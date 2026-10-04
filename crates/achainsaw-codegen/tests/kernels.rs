//! The example kernels in `examples/kernels/` against scalar references, at every ISA level
//! this host reaches (so `vx` is 128 to 512 bits on LLVM), for lengths that exercise masked
//! tails. Runs on the backend selected by `ACHAINSAW_BACKEND`.

use achainsaw_codegen::cpu::{CpuFeatures, IsaLevel};
use achainsaw_codegen::JitEngine;
use achainsaw_ir::parse_and_validate;

const LENGTHS: [usize; 8] = [1, 2, 3, 7, 16, 17, 100, 1031];

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

/// Compiles `examples/kernels/<name>.air` once per ISA level.
fn engines(name: &str) -> Vec<(IsaLevel, JitEngine)> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../examples/kernels/");
    let src = std::fs::read_to_string(format!("{path}{name}.air")).unwrap();
    let module = parse_and_validate(&src).unwrap_or_else(|d| panic!("{name}: {d:?}"));
    host_levels()
        .into_iter()
        .map(|(level, features)| {
            let mut e = JitEngine::with_features(&features).unwrap();
            e.compile_module(&module).unwrap();
            (level, e)
        })
        .collect()
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let x = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D);
        (x >> 40) as f32 / (1u64 << 24) as f32 * 4.0 - 2.0
    }
    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next()).collect()
    }
}

/// Asserts `got` is within a relative tolerance of `want` (accumulation order differs from
/// the f64 reference with vector width).
fn close(got: f64, want: f64, scale: f64, ctx: &str) {
    let tol = 1e-5 * scale.max(1.0);
    assert!((got - want).abs() <= tol, "{ctx}: got {got}, want {want}");
}

#[test]
fn cosine_similarity_and_l2_match_reference() {
    type Pair = extern "C" fn(*const f32, *const f32, i64) -> f32;
    for (name, f) in [
        ("cosine_similarity", engines("cosine_similarity")),
        ("euclidean_distance", engines("euclidean_distance")),
    ] {
        for (level, engine) in &f {
            let k: Pair = unsafe { std::mem::transmute(engine.get_fn_ptr(name).unwrap()) };
            let mut rng = Rng(11);
            for n in LENGTHS {
                let (a, b) = (rng.vec(n), rng.vec(n));
                let dot: f64 = a.iter().zip(&b).map(|(x, y)| *x as f64 * *y as f64).sum();
                let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum();
                let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum();
                let l2: f64 = a
                    .iter()
                    .zip(&b)
                    .map(|(x, y)| (*x as f64 - *y as f64).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let want = if name == "cosine_similarity" {
                    dot / (na.sqrt() * nb.sqrt() + 1e-12)
                } else {
                    l2
                };
                let got = k(a.as_ptr(), b.as_ptr(), n as i64) as f64;
                close(got, want, want.abs(), &format!("{name} n={n} at {level}"));
            }
        }
    }
}

#[test]
fn rmsnorm_matches_reference() {
    type Rms = extern "C" fn(*const f32, *const f32, *mut f32, i64) -> f32;
    for (level, engine) in engines("rmsnorm") {
        let k: Rms = unsafe { std::mem::transmute(engine.get_fn_ptr("rmsnorm").unwrap()) };
        let mut rng = Rng(5);
        for n in LENGTHS {
            let (x, w) = (rng.vec(n), rng.vec(n));
            // Guard words past the end must stay untouched by the masked stores.
            let mut out = vec![7.0f32; n + 16];
            let inv = k(x.as_ptr(), w.as_ptr(), out.as_mut_ptr(), n as i64) as f64;
            let ms: f64 = x.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / n as f64;
            let want_inv = 1.0 / (ms + 1e-6).sqrt();
            close(
                inv,
                want_inv,
                want_inv,
                &format!("rmsnorm scale n={n} at {level}"),
            );
            for i in 0..n {
                let want = x[i] as f64 * want_inv * w[i] as f64;
                close(
                    out[i] as f64,
                    want,
                    want_inv,
                    &format!("rmsnorm[{i}] n={n} at {level}"),
                );
            }
            assert!(
                out[n..].iter().all(|&v| v == 7.0),
                "rmsnorm wrote past n={n} at {level}"
            );
        }
    }
}

#[test]
fn softmax_matches_reference() {
    type Softmax = extern "C" fn(*const f32, *mut f32, i64) -> f32;
    for (level, engine) in engines("softmax") {
        let k: Softmax = unsafe { std::mem::transmute(engine.get_fn_ptr("softmax").unwrap()) };
        let mut rng = Rng(3);
        for n in LENGTHS {
            // All-negative inputs catch a max that is padded with zeros.
            let x: Vec<f32> = rng.vec(n).iter().map(|v| v * 4.0 - 20.0).collect();
            let mut out = vec![7.0f32; n + 16];
            let total = k(x.as_ptr(), out.as_mut_ptr(), n as i64) as f64;
            let m = x.iter().fold(f64::NEG_INFINITY, |a, &v| a.max(v as f64));
            let e: Vec<f64> = x.iter().map(|&v| (v as f64 - m).exp()).collect();
            let s: f64 = e.iter().sum();
            close(total, s, s, &format!("softmax sum n={n} at {level}"));
            for i in 0..n {
                close(
                    out[i] as f64,
                    e[i] / s,
                    1.0,
                    &format!("softmax[{i}] n={n} at {level}"),
                );
            }
            assert!(
                out[n..].iter().all(|&v| v == 7.0),
                "softmax wrote past n={n} at {level}"
            );
        }
    }
}

#[test]
fn gemv_matches_reference() {
    type Gemv = extern "C" fn(*const f32, *const f32, *mut f32, i64, i64);
    for (level, engine) in engines("gemv_f32") {
        let k: Gemv = unsafe { std::mem::transmute(engine.get_fn_ptr("gemv").unwrap()) };
        let mut rng = Rng(9);
        for (m, kk) in [(1, 1), (3, 17), (5, 64), (33, 100), (8, 1031)] {
            let a = rng.vec(m * kk);
            let x = rng.vec(kk);
            let mut y = vec![0.0f32; m];
            k(a.as_ptr(), x.as_ptr(), y.as_mut_ptr(), m as i64, kk as i64);
            for i in 0..m {
                let terms = (0..kk).map(|j| a[i * kk + j] as f64 * x[j] as f64);
                let scale: f64 = terms.clone().map(f64::abs).sum();
                let want: f64 = terms.sum();
                close(
                    y[i] as f64,
                    want,
                    scale,
                    &format!("gemv {m}x{kk} row {i} at {level}"),
                );
            }
        }
    }
}

#[test]
fn gemm_bf16_matches_reference() {
    type Gemm = extern "C" fn(*mut f32, *const u16, *const u16, i64, i64, i64);
    let bf16 = |v: f32| ((v.to_bits() + 0x7fff + ((v.to_bits() >> 16) & 1)) >> 16) as u16;
    let val = |h: u16| f32::from_bits((h as u32) << 16) as f64;
    for (level, engine) in engines("gemm_bf16") {
        let k: Gemm = unsafe { std::mem::transmute(engine.get_fn_ptr("gemm_bf16").unwrap()) };
        let mut rng = Rng(13);
        for (m, n, kk) in [(1, 1, 1), (4, 17, 9), (33, 20, 64)] {
            let a: Vec<u16> = rng.vec(m * kk).into_iter().map(bf16).collect();
            let b: Vec<u16> = rng.vec(kk * n).into_iter().map(bf16).collect();
            let mut c = vec![1.0f32; m * n];
            k(
                c.as_mut_ptr(),
                a.as_ptr(),
                b.as_ptr(),
                m as i64,
                n as i64,
                kk as i64,
            );
            for i in 0..m {
                for j in 0..n {
                    let terms = (0..kk).map(|p| val(a[i * kk + p]) * val(b[p * n + j]));
                    let scale: f64 = 1.0 + terms.clone().map(f64::abs).sum::<f64>();
                    let want = 1.0 + terms.sum::<f64>();
                    let ctx = format!("gemm_bf16 {m}x{n}x{kk} C[{i},{j}] at {level}");
                    close(c[i * n + j] as f64, want, scale, &ctx);
                }
            }
        }
    }
}

/// One decode step of DeepSeek V4 Pro's sparse MQA attention (FlashAttention-2 blocks over
/// the selected KV entries, with a per-head sink), against an f64 reference on the same
/// bf16 inputs. The kernel rounds the probabilities to bf16 before multiplying by V, as
/// FlashAttention does, so each output may differ by about 2^-8 of its weighted magnitude.
#[test]
fn flash_attention_matches_reference() {
    type Attn =
        extern "C" fn(*const u16, *const u16, *const i32, *const f32, *mut f32, i64, i64, i64, f32);
    let bf16 = |v: f32| ((v.to_bits() + 0x7fff + ((v.to_bits() >> 16) & 1)) >> 16) as u16;
    let val = |h: u16| f32::from_bits((h as u32) << 16) as f64;
    // (heads, head dim, cache entries, selected entries, unused slots)
    let shapes = [
        (1, 8, 4, 1, 0),
        (3, 40, 50, 17, 2),
        (5, 17, 300, 130, 7),
        // DeepSeek V4 Pro: 128 heads, 512-dim shared K=V entries, window 128 + top-1024.
        (128, 512, 4096, 1152, 64),
    ];
    for (level, engine) in engines("flash_attention") {
        let k: Attn = unsafe { std::mem::transmute(engine.get_fn_ptr("flash_attention").unwrap()) };
        let mut rng = Rng(21);
        for (h, d, cache, nk, unused) in shapes {
            let q: Vec<u16> = rng.vec(h * d).into_iter().map(bf16).collect();
            let kv: Vec<u16> = rng.vec(cache * d).into_iter().map(bf16).collect();
            let sink: Vec<f32> = rng.vec(h);
            let mut idx: Vec<i32> = (0..nk).map(|t| ((t * 7919 + 13) % cache) as i32).collect();
            for u in 0..unused {
                idx[(u * 37 + 5) % nk] = -1;
            }
            let scale = 1.0 / (d as f32).sqrt();
            let mut out = vec![0.0f32; h * d];
            k(
                q.as_ptr(),
                kv.as_ptr(),
                idx.as_ptr(),
                sink.as_ptr(),
                out.as_mut_ptr(),
                h as i64,
                d as i64,
                nk as i64,
                scale,
            );

            let valid: Vec<usize> = idx
                .iter()
                .filter(|&&i| i >= 0)
                .map(|&i| i as usize)
                .collect();
            for r in 0..h {
                let scores: Vec<f64> = valid
                    .iter()
                    .map(|&e| {
                        let dot: f64 = (0..d).map(|c| val(q[r * d + c]) * val(kv[e * d + c])).sum();
                        dot * scale as f64
                    })
                    .collect();
                let m = scores.iter().cloned().fold(-1e30f64, f64::max);
                let p: Vec<f64> = scores.iter().map(|s| (s - m).exp()).collect();
                let den = p.iter().sum::<f64>() + (sink[r] as f64 - m).exp();
                for c in 0..d {
                    let num: f64 = valid
                        .iter()
                        .zip(&p)
                        .map(|(&e, w)| w * val(kv[e * d + c]))
                        .sum();
                    let mag: f64 = valid
                        .iter()
                        .zip(&p)
                        .map(|(&e, w)| w * val(kv[e * d + c]).abs())
                        .sum();
                    let want = num / den;
                    let got = out[r * d + c] as f64;
                    let tol = mag / den / 128.0 + 1e-6;
                    assert!(
                        (got - want).abs() <= tol,
                        "flash_attention h={h} d={d} nk={nk} out[{r},{c}] at {level}: got {got}, want {want}, tol {tol}"
                    );
                }
            }
        }
    }
}
