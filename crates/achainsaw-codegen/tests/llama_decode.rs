//! `examples/kernels/llama_decode.air`, the single-call fused decode step, against a
//! straightforward f64 reference of the same Llama-style model on a small random model,
//! over several positions (so attention reads earlier cache entries), at every ISA level of
//! the backend selected by `ACHAINSAW_BACKEND`.
//!
//! The reference quantizes activations exactly as the kernel does (f32 arithmetic), but
//! computes everything else in f64, so a value that lands within rounding of a .5 boundary
//! can quantize differently: hidden states are compared with a tolerance of a fraction of a
//! quantization step, and tokens only where the top two logits are clearly apart.

use achainsaw_codegen::cpu::{CpuFeatures, IsaLevel};
use achainsaw_codegen::JitEngine;
use achainsaw_ir::parse_and_validate;

struct Rng(u64);
impl Rng {
    /// Uniform in [-1, 1).
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let x = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D);
        (x >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }
    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next()).collect()
    }
}

/// Weight formats of `qmat_chunk`, by their number in a matrix part.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Fmt {
    Q8 = 0,
    Q4 = 1,
    Q6 = 2,
    Q4K = 3,
    Q5K = 4,
}

impl Fmt {
    const ALL: [Fmt; 5] = [Fmt::Q4, Fmt::Q8, Fmt::Q6, Fmt::Q4K, Fmt::Q5K];

    /// Values per scale.
    fn group(self) -> usize {
        if self == Fmt::Q6 {
            16
        } else {
            32
        }
    }

    /// Q4_K and Q5_K: unsigned values, a scale and a min per 32-block (6-bit codes times f16
    /// factors per 256 values).
    fn has_min(self) -> bool {
        matches!(self, Fmt::Q4K | Fmt::Q5K)
    }
}

/// A Q8_0, Q4_0, Q6_K, Q4_K or Q5_K matrix: logical `q` [rows x k] (in -8..=7 for Q4_0,
/// -32..=31 for Q6_K, 0..=15 for Q4_K, 0..=31 for Q5_K), f16 scales `s` [k/group x rows]
/// and, for Q4_K and Q5_K, mins `m` [k/32 x rows] (weight s * q - m; both the products of a
/// 6-bit code and an f16 factor per 256 values, `codes`), plus the kernel's
/// packed layout (see `qmat_chunk` in std.air): 64-row chunks with, per 32-block, groups of
/// 4 consecutive elements of a row in one 32-bit lane: byte (g * 64 + r) * 4 + q holds row
/// r's element 4g + q plus 128 for Q8_0; for Q4_0 (g < 4) elements 4g + q (low nibble) and
/// 16 + 4g + q (high nibble), offset by 8 (Q4_K: not offset); for Q6_K those nibbles of the
/// low 4 bits of the values offset by 32, then 512 bytes whose byte (g * 64 + r) * 4 + q
/// (g < 2) holds the high 2 bits of element 8p + 4g + q in bit pair p; for Q5_K the nibbles
/// of the low 4 bits, then 256 bytes whose byte r * 4 + q holds the top bit of element 4g + q
/// in bit g and of element 16 + 4g + q in bit 4 + g. The scales follow per chunk: f16 [64]
/// vectors per block (two for Q6_K), or for Q4_K and Q5_K the codes, [64] u16s per block
/// (scale code | min code << 8), then the f16 factors, [64] for the scales then [64] for the
/// mins per 8 blocks.
struct Mat {
    rows: usize,
    k: usize,
    fmt: Fmt,
    q: Vec<i8>,
    s: Vec<f32>,
    m: Vec<f32>,
    packed_q: Vec<u8>,
    packed_s: Vec<u8>,
}

impl Mat {
    fn random(rng: &mut Rng, rows: usize, k: usize, fmt: Fmt) -> Self {
        let nb = k / 32;
        let g = fmt.group();
        let (qmax, smax) = match fmt {
            Fmt::Q8 => (127.0, 73.0),
            Fmt::Q4 => (8.0, 4.6),
            Fmt::Q6 => (32.0, 18.5),
            Fmt::Q4K => (8.0, 4.6),
            Fmt::Q5K => (16.0, 9.2),
        };
        let q: Vec<i8> = (0..rows * k)
            .map(|_| {
                let v = ((rng.next() * qmax).floor() as i32).clamp(-qmax as i32, qmax as i32 - 1);
                // Q4_K and Q5_K values are unsigned: 0 .. 2 qmax.
                (if fmt.has_min() { v + qmax as i32 } else { v }) as i8
            })
            .collect();
        let f16 = |v: f32| half::f16::from_f32(v).to_f32();
        // Entries of roughly unit variance times 1/sqrt(k), as in a trained layer; f16 scales
        // (Q4_0's are often negative).
        let mut s: Vec<f32> = (0..k / g * rows)
            .map(|_| {
                let v = (0.75 + 0.25 * rng.next()) / (smax * (k as f32).sqrt());
                let v = if matches!(fmt, Fmt::Q4 | Fmt::Q6) && rng.next() < 0.0 {
                    -v
                } else {
                    v
                };
                f16(v)
            })
            .collect();
        // Q4_K and Q5_K: codes near the top of their 6 bits times an f16 factor per 256
        // values (some subnormal), and mins near the middle of the values' range times the
        // scale, so the weights center on zero. The codes and factors are [k/32 x rows] and
        // [k/256 x rows] (the last super-block possibly partial).
        let mut m: Vec<f32> = Vec::new();
        let mut codes = (vec![], vec![], vec![], vec![]);
        if fmt.has_min() {
            let ds: Vec<f32> = (0..nb.div_ceil(8) * rows)
                .map(|_| f16((1.0 + 0.1 * rng.next()) / (63.0 * smax * (k as f32).sqrt())))
                .collect();
            let dm: Vec<f32> = ds.iter().map(|&d| f16(d * qmax)).collect();
            let sc: Vec<u8> = (0..nb * rows)
                .map(|_| (56.0 + 3.9 * rng.next()) as u8)
                .collect();
            let mn: Vec<u8> = sc
                .iter()
                .map(|&c| (c as f32 * (1.0 + 0.1 * rng.next())).min(63.0) as u8)
                .collect();
            let sup = |b: usize, r: usize| (b / 8) * rows + r;
            s = (0..nb * rows)
                .map(|i| ds[sup(i / rows, i % rows)] * sc[i] as f32)
                .collect();
            m = (0..nb * rows)
                .map(|i| dm[sup(i / rows, i % rows)] * mn[i] as f32)
                .collect();
            codes = (sc, mn, ds, dm);
        }
        let block_bytes = match fmt {
            Fmt::Q8 => 2048,
            Fmt::Q4 | Fmt::Q4K => 1024,
            Fmt::Q6 => 1536,
            Fmt::Q5K => 1280,
        };
        let mut packed_q = vec![0u8; rows / 64 * nb * block_bytes];
        let mut packed_s = Vec::new();
        for c in 0..rows / 64 {
            for kk in 0..k {
                let (b, t) = (kk / 32, kk % 32);
                let blk = (c * nb + b) * block_bytes;
                for r in 0..64 {
                    let v = q[(c * 64 + r) * k + kk];
                    // Byte q of group g's 32-bit lane for row r.
                    let at = |g: usize, q: usize| blk + (g * 64 + r) * 4 + q;
                    let n = t % 16;
                    match fmt {
                        Fmt::Q8 => packed_q[at(t / 4, t % 4)] = v as u8 ^ 0x80,
                        Fmt::Q4 => packed_q[at(n / 4, n % 4)] |= ((v + 8) as u8) << (4 * (t / 16)),
                        Fmt::Q4K => packed_q[at(n / 4, n % 4)] |= (v as u8) << (4 * (t / 16)),
                        Fmt::Q5K => {
                            let u = v as u8;
                            packed_q[at(n / 4, n % 4)] |= (u & 15) << (4 * (t / 16));
                            packed_q[blk + 1024 + r * 4 + n % 4] |=
                                (u >> 4) << (4 * (t / 16) + n / 4);
                        }
                        Fmt::Q6 => {
                            let u = (v + 32) as u8;
                            packed_q[at(n / 4, n % 4)] |= (u & 15) << (4 * (t / 16));
                            let e = t % 8;
                            packed_q[at(e / 4, e % 4) + 1024] |= (u >> 4) << (2 * (t / 8));
                        }
                    }
                }
            }
            let rows_of = |v: usize| (v * rows + c * 64)..(v * rows + c * 64 + 64);
            let f16_bytes = |src: &[f32]| -> Vec<u8> {
                src.iter()
                    .flat_map(|&v| half::f16::from_f32(v).to_bits().to_le_bytes())
                    .collect()
            };
            if fmt.has_min() {
                let (sc, mn, ds, dm) = &codes;
                for b in 0..nb {
                    for (&c, &n) in sc[rows_of(b)].iter().zip(&mn[rows_of(b)]) {
                        packed_s.extend((c as u16 | (n as u16) << 8).to_le_bytes());
                    }
                }
                for sb in 0..nb.div_ceil(8) {
                    packed_s.extend(f16_bytes(&ds[rows_of(sb)]));
                    packed_s.extend(f16_bytes(&dm[rows_of(sb)]));
                }
            } else {
                for gi in 0..nb * 32 / g {
                    packed_s.extend(f16_bytes(&s[rows_of(gi)]));
                }
            }
        }
        Self {
            rows,
            k,
            fmt,
            q,
            s,
            m,
            packed_q,
            packed_s,
        }
    }

    fn matvec(&self, (xq, dx): &(Vec<i8>, Vec<f32>)) -> Vec<f64> {
        let g = self.fmt.group();
        (0..self.rows)
            .map(|i| {
                (0..self.k / g)
                    .map(|gi| {
                        let x = &xq[gi * g..gi * g + g];
                        let dot: i64 = (0..g)
                            .map(|t| self.q[i * self.k + gi * g + t] as i64 * x[t] as i64)
                            .sum();
                        let mut v = self.s[gi * self.rows + i] as f64 * dot as f64;
                        if self.fmt.has_min() {
                            let sum: i64 = x.iter().map(|&v| v as i64).sum();
                            v -= self.m[gi * self.rows + i] as f64 * sum as f64;
                        }
                        v * dx[gi * g / 32] as f64
                    })
                    .sum()
            })
            .collect()
    }
}

/// A weight matrix as the kernel takes it: parts of whole 64-row chunks stacked by rows,
/// each in its own format (a fused QKV matrix whose Q, K and V differ).
struct Matrix(Vec<Mat>);

impl Matrix {
    fn random(rng: &mut Rng, rows: &[usize], k: usize, fmts: &[Fmt]) -> Self {
        Self(
            rows.iter()
                .zip(fmts)
                .map(|(&r, &f)| Mat::random(rng, r, k, f))
                .collect(),
        )
    }

    /// The parts' 4 words each (weights, scales, format, chunks), which the model table
    /// points to.
    fn words(&self) -> Vec<usize> {
        self.0
            .iter()
            .flat_map(|p| {
                [
                    p.packed_q.as_ptr() as usize,
                    p.packed_s.as_ptr() as usize,
                    p.fmt as usize,
                    p.rows / 64,
                ]
            })
            .collect()
    }

    fn matvec(&self, x: &(Vec<i8>, Vec<f32>)) -> Vec<f64> {
        self.0.iter().flat_map(|p| p.matvec(x)).collect()
    }
}

/// Rounds to bf16 (nearest even), as the kernel stores its KV cache.
fn bf16_round(v: f64) -> f64 {
    half::bf16::from_f32(v as f32).to_f64()
}

/// x quantized per 32-block as the kernel's `quantize_q8` does (f32 arithmetic).
fn quantize(x: &[f32]) -> (Vec<i8>, Vec<f32>) {
    let mut xq = vec![0i8; x.len()];
    let mut dx = vec![0f32; x.len() / 32];
    for b in 0..x.len() / 32 {
        let blk = &x[b * 32..b * 32 + 32];
        let d = blk.iter().fold(0f32, |a, v| a.max(v.abs())) / 127.0;
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };
        dx[b] = d;
        for t in 0..32 {
            let v = blk[t] * id;
            // Halves away from zero, as `vround`.
            xq[b * 32 + t] = v.round() as i32 as i8;
        }
    }
    (xq, dx)
}

struct Layer {
    attn_norm: Vec<f32>,
    qkv: Matrix,
    o: Matrix,
    ffn_norm: Vec<f32>,
    gate_up: Matrix,
    down: Matrix,
}

struct Model {
    d: usize,
    nh: usize,
    nkv: usize,
    hd: usize,
    f: usize,
    vocab: usize,
    max_ctx: usize,
    eps: f32,
    embd: Vec<f32>,
    norm_out: Vec<f32>,
    lm: Matrix,
    cos: Vec<f32>,
    sin: Vec<f32>,
    layers: Vec<Layer>,
}

impl Model {
    #[allow(clippy::too_many_arguments)]
    fn random(
        d: usize,
        nl: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
        f: usize,
        vocab: usize,
        max_ctx: usize,
        mixed: bool,
    ) -> Self {
        let mut rng = Rng(2026);
        let mut norm = |n| (0..n).map(|_| 1.0 + 0.2 * rng.next()).collect::<Vec<f32>>();
        let attn_norms: Vec<Vec<f32>> = (0..nl).map(|_| norm(d)).collect();
        let ffn_norms: Vec<Vec<f32>> = (0..nl).map(|_| norm(d)).collect();
        let norm_out = norm(d);
        let mut rng = Rng(7);
        let half = hd / 2;
        let (mut cos, mut sin) = (Vec::new(), Vec::new());
        for p in 0..max_ctx {
            for j in 0..half {
                let a = p as f64 * 500000f64.powf(-2.0 * j as f64 / hd as f64);
                cos.push(a.cos() as f32);
                sin.push(a.sin() as f32);
            }
        }
        // `mixed`: Q4_0, Q8_0, Q6_K, Q4_K and Q5_K rotate across the matrices of a layer
        // (Q, K and V each in its own) and across layers, and the LM head is Q6_K.
        let layers = attn_norms
            .into_iter()
            .zip(ffn_norms)
            .enumerate()
            .map(|(l, (attn_norm, ffn_norm))| {
                let fmt = |i: usize| {
                    if mixed {
                        Fmt::ALL[(i + 3 * l) % Fmt::ALL.len()]
                    } else {
                        Fmt::Q8
                    }
                };
                Layer {
                    attn_norm,
                    qkv: Matrix::random(
                        &mut rng,
                        &[nh * hd, nkv * hd, nkv * hd],
                        d,
                        &[fmt(0), fmt(1), fmt(2)],
                    ),
                    o: Matrix::random(&mut rng, &[d], nh * hd, &[fmt(3)]),
                    ffn_norm,
                    gate_up: Matrix::random(&mut rng, &[2 * f], d, &[fmt(4)]),
                    down: Matrix::random(&mut rng, &[d], f, &[fmt(5)]),
                }
            })
            .collect();
        Self {
            d,
            nh,
            nkv,
            hd,
            f,
            vocab,
            max_ctx,
            eps: 1e-5,
            embd: rng.vec(vocab * d),
            norm_out,
            lm: Matrix::random(
                &mut rng,
                &[vocab],
                d,
                &[if mixed { Fmt::Q6 } else { Fmt::Q8 }],
            ),
            cos,
            sin,
            layers,
        }
    }

    fn norm_quant(&self, h: &[f64], w: &[f32]) -> (Vec<i8>, Vec<f32>) {
        let ms = h.iter().map(|v| v * v).sum::<f64>() / h.len() as f64;
        let inv = 1.0 / (ms + self.eps as f64).sqrt();
        let xn: Vec<f32> = h
            .iter()
            .zip(w)
            .map(|(v, w)| (v * inv * *w as f64) as f32)
            .collect();
        quantize(&xn)
    }

    fn rope(&self, x: &mut [f64], pos: usize) {
        let half = self.hd / 2;
        for j in 0..half {
            let (c, s) = (
                self.cos[pos * half + j] as f64,
                self.sin[pos * half + j] as f64,
            );
            let (x1, x2) = (x[j], x[half + j]);
            x[j] = x1 * c - x2 * s;
            x[half + j] = x2 * c + x1 * s;
        }
    }

    /// u16s in the kernel's KV cache: per layer K then V, each nkv x cache_ctx x hd, with
    /// cache_ctx = max_ctx rounded up to K's tiles of 32 positions.
    fn cache_len(&self) -> usize {
        self.layers.len() * 2 * self.nkv * self.max_ctx.div_ceil(32) * 32 * self.hd
    }

    /// Where the kernel's cache holds value e of KV head g at position pos of layer l, in K
    /// (tiles of 32 positions, each [hd x 32]) and in V ([nkv][cache_ctx][hd]).
    fn cache_at(&self, l: usize, g: usize, pos: usize, e: usize) -> (usize, usize) {
        let ctx = self.max_ctx.div_ceil(32) * 32;
        let plane = self.nkv * ctx * self.hd;
        let k = l * 2 * plane + (g * ctx + pos / 32 * 32) * self.hd + e * 32 + pos % 32;
        let v = l * 2 * plane + plane + (g * ctx + pos) * self.hd + e;
        (k, v)
    }

    /// One decode step; `cache[l]` is (K, V), each [max_ctx][nkv][hd] holding bf16 values.
    /// Returns the hidden state and the logits.
    fn step(
        &self,
        cache: &mut [(Vec<f64>, Vec<f64>)],
        token: usize,
        pos: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        let (d, hd, nh, nkv) = (self.d, self.hd, self.nh, self.nkv);
        let mut h: Vec<f64> = self.embd[token * d..(token + 1) * d]
            .iter()
            .map(|&v| v as f64)
            .collect();
        for (layer, (kc, vc)) in self.layers.iter().zip(cache.iter_mut()) {
            let qkv = layer.qkv.matvec(&self.norm_quant(&h, &layer.attn_norm));
            let mut q = qkv[..nh * hd].to_vec();
            for head in q.chunks_mut(hd) {
                self.rope(head, pos);
            }
            for g in 0..nkv {
                let mut k = qkv[(nh + g) * hd..(nh + g + 1) * hd].to_vec();
                self.rope(&mut k, pos);
                let v = &qkv[(nh + nkv + g) * hd..(nh + nkv + g + 1) * hd];
                let at = (pos * nkv + g) * hd;
                for j in 0..hd {
                    kc[at + j] = bf16_round(k[j]);
                    vc[at + j] = bf16_round(v[j]);
                }
            }
            let mut attn = vec![0f32; nh * hd];
            for hq in 0..nh {
                let g = hq / (nh / nkv);
                let qh = &q[hq * hd..(hq + 1) * hd];
                let scores: Vec<f64> = (0..=pos)
                    .map(|t| {
                        let kt = &kc[(t * nkv + g) * hd..(t * nkv + g + 1) * hd];
                        qh.iter().zip(kt).map(|(a, b)| a * b).sum::<f64>() / (hd as f64).sqrt()
                    })
                    .collect();
                let m = scores.iter().cloned().fold(f64::MIN, f64::max);
                let p: Vec<f64> = scores.iter().map(|s| (s - m).exp()).collect();
                let total: f64 = p.iter().sum();
                for j in 0..hd {
                    let o: f64 = (0..=pos).map(|t| p[t] * vc[(t * nkv + g) * hd + j]).sum();
                    attn[hq * hd + j] = (o / total) as f32;
                }
            }
            for (hv, ov) in h.iter_mut().zip(layer.o.matvec(&quantize(&attn))) {
                *hv += ov;
            }
            let gu = layer.gate_up.matvec(&self.norm_quant(&h, &layer.ffn_norm));
            let mut act = vec![0f32; self.f];
            for c in 0..self.f / 64 {
                for j in 0..64 {
                    let (g, u) = (gu[128 * c + j], gu[128 * c + 64 + j]);
                    act[64 * c + j] = (g / (1.0 + (-g).exp()) * u) as f32;
                }
            }
            for (hv, dv) in h.iter_mut().zip(layer.down.matvec(&quantize(&act))) {
                *hv += dv;
            }
        }
        let logits = self.lm.matvec(&self.norm_quant(&h, &self.norm_out));
        (h, logits)
    }

    /// The kernel's `model` pointer table and `cfg`, and the matrices' part lists the table
    /// points to (to keep alive while it is used).
    fn tables(&self) -> (Vec<usize>, Vec<i64>, Vec<Vec<usize>>) {
        let p = |v: &[f32]| v.as_ptr() as usize;
        let mut parts = vec![self.lm.words()];
        for l in &self.layers {
            parts.extend([&l.qkv, &l.o, &l.gate_up, &l.down].map(Matrix::words));
        }
        let mp = |i: usize| parts[i].as_ptr() as usize;
        let mut t = vec![
            p(&self.embd),
            p(&self.norm_out),
            mp(0),
            p(&self.cos),
            p(&self.sin),
        ];
        for (i, l) in self.layers.iter().enumerate() {
            let m = 1 + 4 * i;
            t.extend([p(&l.attn_norm), mp(m), mp(m + 1)]);
            t.extend([p(&l.ffn_norm), mp(m + 2), mp(m + 3)]);
        }
        let cfg = [
            self.d,
            self.layers.len(),
            self.nh,
            self.nkv,
            self.hd,
            self.f,
            self.vocab,
            self.max_ctx,
        ]
        .map(|v| v as i64)
        .to_vec();
        (t, cfg, parts)
    }
}

type Decode = extern "C" fn(*const usize, *const i64, *mut u16, i64, i64, *mut f32, f32) -> i64;

fn host_levels() -> Vec<(IsaLevel, CpuFeatures)> {
    let host = CpuFeatures::host();
    IsaLevel::names()
        .iter()
        .map(|n| n.parse::<IsaLevel>().unwrap())
        .filter(|l| l.arch() == host.arch)
        .filter_map(|l| {
            let f = host.capped(l).ok()?;
            (f.max_level() == Some(l)).then_some((l, f))
        })
        .collect()
}

#[test]
fn llama_decode_matches_reference() {
    check_decode(false);
}

/// Q4_0, Q8_0, Q6_K, Q4_K and Q5_K matrices mixed within and across layers (Q, K and V in
/// different formats), and a Q6_K LM head.
#[test]
fn llama_decode_mixed_formats_match_reference() {
    check_decode(true);
}

fn check_decode(mixed: bool) {
    let src = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../examples/kernels/llama_decode.air"
    ))
    .unwrap();
    let module = parse_and_validate(&src).unwrap_or_else(|d| panic!("{d:?}"));
    // d=128, 2 layers, 4 q heads over 2 KV heads of 64, MLP 256, vocab 512, context 8.
    let model = Model::random(128, 2, 4, 2, 64, 256, 512, 8, mixed);
    let (table, cfg, _parts) = model.tables();
    let plane = model.max_ctx * model.nkv * model.hd;
    let modes = host_levels()
        .into_iter()
        .flat_map(|(level, features)| [false, true].map(|fast| (level, features, fast)));
    for (level, features, fast) in modes {
        let mut engine = JitEngine::with_features(&features).unwrap();
        engine.set_fast_math(fast);
        engine.compile_module(&module).unwrap();
        let level = format!(
            "{level}{}{}",
            if fast { " fast_math" } else { "" },
            if mixed { " mixed formats" } else { "" }
        );
        let decode: Decode =
            unsafe { std::mem::transmute(engine.get_fn_ptr("llama_decode").unwrap()) };
        // bf16 bits, as `Model::cache_at` lays them out.
        let mut cache = vec![0u16; model.cache_len()];
        let mut ref_cache = vec![(vec![0f64; plane], vec![0f64; plane]); model.layers.len()];
        let mut token = 7usize;
        let mut checked_tokens = 0;
        for pos in 0..6 {
            let mut h = vec![f32::NAN; model.d];
            let next = decode(
                table.as_ptr(),
                cfg.as_ptr(),
                cache.as_mut_ptr(),
                token as i64,
                pos as i64,
                h.as_mut_ptr(),
                model.eps,
            );
            let (ref_h, logits) = model.step(&mut ref_cache, token, pos);
            let scale = ref_h.iter().fold(0f64, |a, v| a.max(v.abs()));
            for (i, (&got, &want)) in h.iter().zip(&ref_h).enumerate() {
                assert!(
                    (got as f64 - want).abs() <= 2e-3 * scale,
                    "h[{i}] at pos {pos}, {level}: got {got}, want {want} (scale {scale})"
                );
            }
            let bf = |b: u16| half::bf16::from_bits(b).to_f64();
            for (l, (kr, vr)) in ref_cache.iter().enumerate() {
                for j in 0..model.nkv * model.hd {
                    let (g, e) = (j / model.hd, j % model.hd);
                    let (k_at, v_at) = model.cache_at(l, g, pos, e);
                    let (k, v) = (bf(cache[k_at]), bf(cache[v_at]));
                    let (kw, vw) = (
                        kr[pos * model.nkv * model.hd + j],
                        vr[pos * model.nkv * model.hd + j],
                    );
                    assert!(
                        (k - kw).abs() <= 1e-2 * (1.0 + kw.abs()),
                        "K l={l} pos {pos} [{j}] {level}: {k} vs {kw}"
                    );
                    assert!(
                        (v - vw).abs() <= 1e-2 * (1.0 + vw.abs()),
                        "V l={l} pos {pos} [{j}] {level}: {v} vs {vw}"
                    );
                }
            }
            let mut order: Vec<usize> = (0..model.vocab).collect();
            order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
            let lscale = logits.iter().fold(0f64, |a, v| a.max(v.abs()));
            if logits[order[0]] - logits[order[1]] > 1e-2 * lscale {
                assert_eq!(next as usize, order[0], "token at pos {pos}, {level}");
                checked_tokens += 1;
            }
            assert!((next as usize) < model.vocab);
            // Feed the kernel's own token back, so both sides see the same sequence.
            token = next as usize;
        }
        assert!(
            checked_tokens >= 3,
            "only {checked_tokens} clear-cut tokens at {level}"
        );
    }
}

type Prefill =
    extern "C" fn(*const usize, *const i64, *mut u16, *const i64, i64, i64, *mut f32, f32) -> i64;

/// `llama_prefill` over a prompt gives exactly what `llama_decode` gives token by token:
/// the same KV cache bits, final hidden state bits and next token, whether the prompt goes
/// in one call (a 4-token tile and a remainder) or in pieces starting past position 0.
#[test]
fn llama_prefill_matches_decode_exactly() {
    let src = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../examples/kernels/llama_decode.air"
    ))
    .unwrap();
    let module = parse_and_validate(&src).unwrap_or_else(|d| panic!("{d:?}"));
    // A short prompt, and one past two 256-position attention splits; each also in pieces.
    let mut rng = Rng(11);
    let long: Vec<i64> = (0..530)
        .map(|_| (rng.next().abs() * 511.0) as i64)
        .collect();
    let short = vec![7, 300, 42, 511, 0, 99];
    // (mixed formats, prompt, context length, ways to split the prompt into prefill calls)
    type Case<'a> = (bool, &'a [i64], usize, &'a [&'a [usize]]);
    let cases: [Case; 4] = [
        (false, &short, 8, &[&[6], &[1, 5]]),
        (true, &short, 8, &[&[6], &[1, 5]]),
        (false, &long, 530, &[&[530], &[255, 2, 273]]),
        (true, &long, 530, &[&[300, 230]]),
    ];
    for &(mixed, prompt, max_ctx, all_pieces) in &cases {
        let model = Model::random(128, 2, 4, 2, 64, 256, 512, max_ctx, mixed);
        let (table, cfg, _parts) = model.tables();
        let cache_len = model.cache_len();
        for (level, features) in host_levels() {
            let mut engine = JitEngine::with_features(&features).unwrap();
            engine.compile_module(&module).unwrap();
            let decode: Decode =
                unsafe { std::mem::transmute(engine.get_fn_ptr("llama_decode").unwrap()) };
            let prefill: Prefill =
                unsafe { std::mem::transmute(engine.get_fn_ptr("llama_prefill").unwrap()) };
            let ctx = format!("{level}{}", if mixed { " mixed formats" } else { "" });

            let mut cache = vec![0u16; cache_len];
            let mut h = vec![f32::NAN; model.d];
            let mut next = 0;
            for (pos, &tok) in prompt.iter().enumerate() {
                next = decode(
                    table.as_ptr(),
                    cfg.as_ptr(),
                    cache.as_mut_ptr(),
                    tok,
                    pos as i64,
                    h.as_mut_ptr(),
                    model.eps,
                );
            }

            for &pieces in all_pieces {
                let mut p_cache = vec![0u16; cache_len];
                let mut p_h = vec![f32::NAN; model.d];
                let mut p_next = -1;
                let mut pos = 0;
                for &n in pieces {
                    p_next = prefill(
                        table.as_ptr(),
                        cfg.as_ptr(),
                        p_cache.as_mut_ptr(),
                        prompt[pos..].as_ptr(),
                        n as i64,
                        pos as i64,
                        p_h.as_mut_ptr(),
                        model.eps,
                    );
                    pos += n;
                }
                let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                assert_eq!(p_next, next, "next token, pieces {pieces:?}, {ctx}");
                assert_eq!(
                    bits(&p_h),
                    bits(&h),
                    "hidden state, pieces {pieces:?}, {ctx}"
                );
                assert!(p_cache == cache, "KV cache, pieces {pieces:?}, {ctx}");
            }
        }
    }
}

/// A 520-token prompt through `llama_prefill`, then one `llama_decode` step, against the f64
/// reference: attention over three 256-position splits, merged. The kernel rounds f32 K and
/// V values to bf16 and the reference f64 ones, so a value at a rounding boundary can land a
/// bf16 step apart (a few in the first 64 positions); that can flip an int8 quantization of
/// the attention output, and the flip carries into the next layer's K and V. Such flips
/// move the hidden state by up to about 0.5% of its scale, against whole units for a wrong
/// split; `gqa_attention`'s test checks the splits' arithmetic itself to 1e-4.
#[test]
fn llama_long_context_matches_reference() {
    let src = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../examples/kernels/llama_decode.air"
    ))
    .unwrap();
    let module = parse_and_validate(&src).unwrap_or_else(|d| panic!("{d:?}"));
    let model = Model::random(128, 2, 4, 2, 64, 256, 512, 600, false);
    let (table, cfg, _parts) = model.tables();
    let plane = model.max_ctx * model.nkv * model.hd;
    let mut rng = Rng(23);
    let prompt: Vec<i64> = (0..520)
        .map(|_| (rng.next().abs() * 511.0) as i64)
        .collect();
    let mut ref_cache = vec![(vec![0f64; plane], vec![0f64; plane]); model.layers.len()];
    let mut ref_h = Vec::new();
    for (pos, &tok) in prompt.iter().enumerate() {
        ref_h = model.step(&mut ref_cache, tok as usize, pos).0;
    }
    let close = |got: &[f32], want: &[f64], what: &str| {
        let scale = want.iter().fold(0f64, |a, v| a.max(v.abs()));
        for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
            assert!(
                (g as f64 - w).abs() <= 1e-2 * scale,
                "{what} h[{i}]: got {g}, want {w} (scale {scale})"
            );
        }
    };
    for (level, features) in host_levels() {
        let mut engine = JitEngine::with_features(&features).unwrap();
        engine.compile_module(&module).unwrap();
        let decode: Decode =
            unsafe { std::mem::transmute(engine.get_fn_ptr("llama_decode").unwrap()) };
        let prefill: Prefill =
            unsafe { std::mem::transmute(engine.get_fn_ptr("llama_prefill").unwrap()) };
        let mut cache = vec![0u16; model.cache_len()];
        let mut h = vec![f32::NAN; model.d];
        let next = prefill(
            table.as_ptr(),
            cfg.as_ptr(),
            cache.as_mut_ptr(),
            prompt.as_ptr(),
            prompt.len() as i64,
            0,
            h.as_mut_ptr(),
            model.eps,
        );
        close(&h, &ref_h, &format!("after the prompt, {level}"));
        decode(
            table.as_ptr(),
            cfg.as_ptr(),
            cache.as_mut_ptr(),
            next,
            prompt.len() as i64,
            h.as_mut_ptr(),
            model.eps,
        );
        let mut rc = ref_cache.clone();
        let (want, _) = model.step(&mut rc, next as usize, prompt.len());
        close(&h, &want, &format!("decode at position 520, {level}"));
    }
}
