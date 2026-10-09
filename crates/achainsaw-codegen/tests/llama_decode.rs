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

/// Weight formats of `mat_chunk_into`, by their number in the model table.
#[derive(Clone, Copy, PartialEq)]
enum Fmt {
    Q8 = 0,
    Q4 = 1,
    Q6 = 2,
}

impl Fmt {
    /// Values per scale.
    fn group(self) -> usize {
        if self == Fmt::Q6 {
            16
        } else {
            32
        }
    }
}

/// A Q8_0, Q4_0 or Q6_K matrix: logical `q` [rows x k] (in -8..=7 for Q4_0, -32..=31 for
/// Q6_K) and f16 scales `s` [k/group x rows], plus the kernel's packed layout (see
/// `qmat_chunk` in std.air): 64-row chunks with, per 32-block, groups of 4 consecutive
/// elements of a row in one 32-bit lane: byte (g * 64 + r) * 4 + q holds row r's element
/// 4g + q for Q8_0; for Q4_0 (g < 4) elements 4g + q (low nibble) and 16 + 4g + q (high
/// nibble), offset by 8; for Q6_K those nibbles of the low 4 bits of the values offset by
/// 32, then 512 bytes whose byte (g * 64 + r) * 4 + q (g < 2) holds the high 2 bits of
/// element 8p + 4g + q in bit pair p.
struct Mat {
    rows: usize,
    k: usize,
    fmt: Fmt,
    q: Vec<i8>,
    s: Vec<f32>,
    packed_q: Vec<u8>,
    packed_s: Vec<u16>,
}

impl Mat {
    fn random(rng: &mut Rng, rows: usize, k: usize, fmt: Fmt) -> Self {
        let nb = k / 32;
        let g = fmt.group();
        let (qmax, smax) = match fmt {
            Fmt::Q8 => (127.0, 73.0),
            Fmt::Q4 => (8.0, 4.6),
            Fmt::Q6 => (32.0, 18.5),
        };
        let q: Vec<i8> = (0..rows * k)
            .map(|_| {
                ((rng.next() * qmax).floor() as i32).clamp(-qmax as i32, qmax as i32 - 1) as i8
            })
            .collect();
        // Entries of roughly unit variance times 1/sqrt(k), as in a trained layer; f16 scales
        // (Q4_0's are often negative).
        let s: Vec<f32> = (0..k / g * rows)
            .map(|_| {
                let v = (0.75 + 0.25 * rng.next()) / (smax * (k as f32).sqrt());
                let v = if fmt != Fmt::Q8 && rng.next() < 0.0 {
                    -v
                } else {
                    v
                };
                half::f16::from_f32(v).to_f32()
            })
            .collect();
        let block_bytes = match fmt {
            Fmt::Q8 => 2048,
            Fmt::Q4 => 1024,
            Fmt::Q6 => 1536,
        };
        let mut packed_q = vec![0u8; rows / 64 * nb * block_bytes];
        let mut packed_s = vec![0u16; k / g * rows];
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
                        Fmt::Q8 => packed_q[at(t / 4, t % 4)] = v as u8,
                        Fmt::Q4 => packed_q[at(n / 4, n % 4)] |= ((v + 8) as u8) << (4 * (t / 16)),
                        Fmt::Q6 => {
                            let u = (v + 32) as u8;
                            packed_q[at(n / 4, n % 4)] |= (u & 15) << (4 * (t / 16));
                            let e = t % 8;
                            packed_q[at(e / 4, e % 4) + 1024] |= (u >> 4) << (2 * (t / 8));
                        }
                    }
                }
            }
            for gi in 0..k / g {
                for r in 0..64 {
                    packed_s[(c * (k / g) + gi) * 64 + r] =
                        half::f16::from_f32(s[gi * rows + c * 64 + r]).to_bits();
                }
            }
        }
        Self {
            rows,
            k,
            fmt,
            q,
            s,
            packed_q,
            packed_s,
        }
    }

    /// The matrix's 3 words in the kernel's model table.
    fn words(&self) -> [usize; 3] {
        [
            self.packed_q.as_ptr() as usize,
            self.packed_s.as_ptr() as usize,
            self.fmt as usize,
        ]
    }

    fn matvec(&self, (xq, dx): &(Vec<i8>, Vec<f32>)) -> Vec<f64> {
        let g = self.fmt.group();
        (0..self.rows)
            .map(|i| {
                (0..self.k / g)
                    .map(|gi| {
                        let dot: i64 = (0..g)
                            .map(|t| self.q[i * self.k + gi * g + t] as i64 * xq[gi * g + t] as i64)
                            .sum();
                        self.s[gi * self.rows + i] as f64 * dx[gi * g / 32] as f64 * dot as f64
                    })
                    .sum()
            })
            .collect()
    }
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
    qkv: Mat,
    o: Mat,
    ffn_norm: Vec<f32>,
    gate_up: Mat,
    down: Mat,
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
    lm: Mat,
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
        // `mixed`: Q4_0, Q8_0 and Q6_K rotate across the matrices of a layer and across
        // layers, and the LM head is Q6_K.
        let layers = attn_norms
            .into_iter()
            .zip(ffn_norms)
            .enumerate()
            .map(|(l, (attn_norm, ffn_norm))| {
                let fmt = |i: usize| match (mixed, (i + l) % 3) {
                    (false, _) => Fmt::Q8,
                    (true, 0) => Fmt::Q4,
                    (true, 1) => Fmt::Q8,
                    _ => Fmt::Q6,
                };
                Layer {
                    attn_norm,
                    qkv: Mat::random(&mut rng, (nh + 2 * nkv) * hd, d, fmt(0)),
                    o: Mat::random(&mut rng, d, nh * hd, fmt(1)),
                    ffn_norm,
                    gate_up: Mat::random(&mut rng, 2 * f, d, fmt(2)),
                    down: Mat::random(&mut rng, d, f, fmt(3)),
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
            lm: Mat::random(&mut rng, vocab, d, if mixed { Fmt::Q6 } else { Fmt::Q8 }),
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

    /// One decode step; `cache[l]` is (K, V), each [max_ctx][nkv][hd]. Returns the hidden
    /// state and the logits.
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
                kc[at..at + hd].copy_from_slice(&k);
                vc[at..at + hd].copy_from_slice(v);
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

    /// The kernel's `model` pointer table and `cfg`.
    fn tables(&self) -> (Vec<usize>, Vec<i64>) {
        let p = |v: &[f32]| v.as_ptr() as usize;
        let mut t = vec![p(&self.embd), p(&self.norm_out)];
        t.extend(self.lm.words());
        t.extend([p(&self.cos), p(&self.sin)]);
        for l in &self.layers {
            t.push(p(&l.attn_norm));
            t.extend(l.qkv.words());
            t.extend(l.o.words());
            t.push(p(&l.ffn_norm));
            t.extend(l.gate_up.words());
            t.extend(l.down.words());
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
        (t, cfg)
    }
}

type Decode = extern "C" fn(*const usize, *const i64, *mut f32, i64, i64, *mut f32, f32) -> i64;

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

/// Q4_0, Q8_0 and Q6_K matrices mixed within and across layers, and a Q6_K LM head.
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
    let (table, cfg) = model.tables();
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
            if mixed { " mixed Q4/Q8/Q6" } else { "" }
        );
        let decode: Decode =
            unsafe { std::mem::transmute(engine.get_fn_ptr("llama_decode").unwrap()) };
        let mut cache = vec![0f32; model.layers.len() * 2 * plane];
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
            for (l, (kr, vr)) in ref_cache.iter().enumerate() {
                let base = l * 2 * plane + pos * model.nkv * model.hd;
                for j in 0..model.nkv * model.hd {
                    let (k, v) = (cache[base + j] as f64, cache[base + plane + j] as f64);
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
    extern "C" fn(*const usize, *const i64, *mut f32, *const i64, i64, i64, *mut f32, f32) -> i64;

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
    let prompt: [i64; 6] = [7, 300, 42, 511, 0, 99];
    for mixed in [false, true] {
        let model = Model::random(128, 2, 4, 2, 64, 256, 512, 8, mixed);
        let (table, cfg) = model.tables();
        let cache_len = model.layers.len() * 2 * model.max_ctx * model.nkv * model.hd;
        for (level, features) in host_levels() {
            let mut engine = JitEngine::with_features(&features).unwrap();
            engine.compile_module(&module).unwrap();
            let decode: Decode =
                unsafe { std::mem::transmute(engine.get_fn_ptr("llama_decode").unwrap()) };
            let prefill: Prefill =
                unsafe { std::mem::transmute(engine.get_fn_ptr("llama_prefill").unwrap()) };
            let ctx = format!("{level}{}", if mixed { " mixed Q4/Q8/Q6" } else { "" });

            let mut cache = vec![0f32; cache_len];
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

            for pieces in [&[6usize][..], &[1, 5]] {
                let mut p_cache = vec![0f32; cache_len];
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
                assert_eq!(
                    bits(&p_cache),
                    bits(&cache),
                    "KV cache, pieces {pieces:?}, {ctx}"
                );
            }
        }
    }
}
