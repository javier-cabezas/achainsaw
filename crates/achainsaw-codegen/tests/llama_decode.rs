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

/// A Q8 matrix: logical `q` [rows x k] and scales `s` [k/32 x rows], plus the kernel's
/// packed layout (64-row chunks, k-major within a chunk).
struct Q8 {
    rows: usize,
    k: usize,
    q: Vec<i8>,
    s: Vec<f32>,
    packed_q: Vec<i8>,
    packed_s: Vec<f32>,
}

impl Q8 {
    fn random(rng: &mut Rng, rows: usize, k: usize) -> Self {
        let nb = k / 32;
        let q: Vec<i8> = (0..rows * k).map(|_| (rng.next() * 127.0) as i8).collect();
        // Entries of roughly unit variance times 1/sqrt(k), as in a trained layer.
        let s: Vec<f32> = (0..nb * rows)
            .map(|_| (0.75 + 0.25 * rng.next()) / (73.0 * (k as f32).sqrt()))
            .collect();
        let (mut packed_q, mut packed_s) = (vec![0i8; rows * k], vec![0f32; nb * rows]);
        for c in 0..rows / 64 {
            for kk in 0..k {
                for r in 0..64 {
                    packed_q[c * k * 64 + kk * 64 + r] = q[(c * 64 + r) * k + kk];
                }
            }
            for b in 0..nb {
                for r in 0..64 {
                    packed_s[c * nb * 64 + b * 64 + r] = s[b * rows + c * 64 + r];
                }
            }
        }
        Self {
            rows,
            k,
            q,
            s,
            packed_q,
            packed_s,
        }
    }

    fn matvec(&self, (xq, dx): &(Vec<i8>, Vec<f32>)) -> Vec<f64> {
        (0..self.rows)
            .map(|i| {
                (0..self.k / 32)
                    .map(|b| {
                        let dot: i64 = (0..32)
                            .map(|t| self.q[i * self.k + b * 32 + t] as i64 * xq[b * 32 + t] as i64)
                            .sum();
                        self.s[b * self.rows + i] as f64 * dx[b] as f64 * dot as f64
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
            xq[b * 32 + t] = (v + if v >= 0.0 { 0.5 } else { -0.5 }) as i32 as i8;
        }
    }
    (xq, dx)
}

struct Layer {
    attn_norm: Vec<f32>,
    qkv: Q8,
    o: Q8,
    ffn_norm: Vec<f32>,
    gate_up: Q8,
    down: Q8,
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
    lm: Q8,
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
        let layers = attn_norms
            .into_iter()
            .zip(ffn_norms)
            .map(|(attn_norm, ffn_norm)| Layer {
                attn_norm,
                qkv: Q8::random(&mut rng, (nh + 2 * nkv) * hd, d),
                o: Q8::random(&mut rng, d, nh * hd),
                ffn_norm,
                gate_up: Q8::random(&mut rng, 2 * f, d),
                down: Q8::random(&mut rng, d, f),
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
            lm: Q8::random(&mut rng, vocab, d),
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
        let pq = |v: &[i8]| v.as_ptr() as usize;
        let mut t = vec![
            p(&self.embd),
            p(&self.norm_out),
            pq(&self.lm.packed_q),
            p(&self.lm.packed_s),
            p(&self.cos),
            p(&self.sin),
        ];
        for l in &self.layers {
            t.extend([
                p(&l.attn_norm),
                pq(&l.qkv.packed_q),
                p(&l.qkv.packed_s),
                pq(&l.o.packed_q),
                p(&l.o.packed_s),
                p(&l.ffn_norm),
                pq(&l.gate_up.packed_q),
                p(&l.gate_up.packed_s),
                pq(&l.down.packed_q),
                p(&l.down.packed_s),
            ]);
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
    let src = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../examples/kernels/llama_decode.air"
    ))
    .unwrap();
    let module = parse_and_validate(&src).unwrap_or_else(|d| panic!("{d:?}"));
    // d=128, 2 layers, 4 q heads over 2 KV heads of 64, MLP 256, vocab 512, context 8.
    let model = Model::random(128, 2, 4, 2, 64, 256, 512, 8);
    let (table, cfg) = model.tables();
    let plane = model.max_ctx * model.nkv * model.hd;
    for (level, features) in host_levels() {
        let mut engine = JitEngine::with_features(&features).unwrap();
        engine.compile_module(&module).unwrap();
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
