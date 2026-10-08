//! The example kernels in `examples/kernels/` against scalar references, at every ISA level
//! this host reaches (so `vx` is 128 to 512 bits on LLVM), for lengths that exercise masked
//! tails. Runs on the backend selected by `ACHAINSAW_BACKEND`.

use achainsaw_codegen::cpu::{CpuFeatures, IsaLevel};
use achainsaw_codegen::{JitEngine, RtValue};
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

/// Compiles `examples/kernels/<name>.air` once per ISA level, without and with `fast_math`
/// (the kernels' results must not depend on it for finite inputs). Each engine comes with
/// a label naming its level and mode.
fn engines(name: &str) -> Vec<(String, JitEngine)> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../examples/kernels/");
    let src = std::fs::read_to_string(format!("{path}{name}.air")).unwrap();
    let module = parse_and_validate(&src).unwrap_or_else(|d| panic!("{name}: {d:?}"));
    host_levels()
        .into_iter()
        .flat_map(|(level, features)| {
            [false, true].map(|fast| {
                let mut e = JitEngine::with_features(&features).unwrap();
                e.set_fast_math(fast);
                e.compile_module(&module).unwrap();
                let mode = if fast { " fast_math" } else { "" };
                (format!("{level}{mode}"), e)
            })
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
fn gemv_par_matches_reference() {
    type Gemv = extern "C" fn(*const f32, *const f32, *mut f32, i64, i64);
    for (level, engine) in engines("gemv_par") {
        let k: Gemv = unsafe { std::mem::transmute(engine.get_fn_ptr("gemv_par").unwrap()) };
        let mut rng = Rng(21);
        for (m, kk) in [(1, 1), (15, 17), (16, 64), (17, 100), (333, 1031)] {
            let a = rng.vec(m * kk);
            let x = rng.vec(kk);
            let mut y = vec![f32::NAN; m];
            k(a.as_ptr(), x.as_ptr(), y.as_mut_ptr(), m as i64, kk as i64);
            for i in 0..m {
                let terms = (0..kk).map(|j| a[i * kk + j] as f64 * x[j] as f64);
                let scale: f64 = terms.clone().map(f64::abs).sum();
                let want: f64 = terms.sum();
                let ctx = format!("gemv_par {m}x{kk} row {i} at {level}");
                close(y[i] as f64, want, scale, &ctx);
            }
        }
        // The driver: rows of A sum to ((i + j) mod 7 - 3) / 4 terms, so sum(y) is exact.
        let bench = unsafe { engine.call_typed("bench", &[200i64, 70, 2].map(RtValue::I64)) };
        let fill = |i: i64, j: i64| ((i + j) % 7 - 3) as f64 / 4.0;
        let want: f64 = (0..200)
            .map(|i| (0..70).map(|j| fill(i, j) * fill(0, j)).sum::<f64>())
            .sum();
        assert_eq!(
            bench.unwrap(),
            Some(RtValue::F32(want as f32)),
            "bench at {level}"
        );
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
        // Five head groups of 8, the last one partial; no selected entries at all.
        (37, 24, 200, 70, 3),
        (2, 8, 4, 0, 0),
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

#[test]
fn swiglu_matches_reference() {
    type Swiglu = extern "C" fn(*const f32, *const f32, *mut f32, i64);
    let specials = [
        -100.0f32, -88.5, -87.0, -20.0, -1e-3, 0.0, 1e-3, 20.0, 87.0, 88.5, 100.0,
    ];
    for (level, engine) in engines("swiglu") {
        let k: Swiglu = unsafe { std::mem::transmute(engine.get_fn_ptr("swiglu").unwrap()) };
        let mut rng = Rng(31);
        for n in LENGTHS.iter().copied().chain([specials.len()]) {
            let gate: Vec<f32> = if n == specials.len() {
                specials.to_vec()
            } else {
                rng.vec(n).iter().map(|v| v * 6.0).collect()
            };
            let up = rng.vec(n);
            let mut out = vec![f32::NAN; n];
            k(gate.as_ptr(), up.as_ptr(), out.as_mut_ptr(), n as i64);
            for i in 0..n {
                let (g, u) = (gate[i] as f64, up[i] as f64);
                let want = g / (1.0 + (-g).exp()) * u;
                let got = out[i] as f64;
                let tol = 2e-6 * want.abs() + 1e-30;
                assert!(
                    (got - want).abs() <= tol,
                    "swiglu n={n} [{i}] g={g} u={u} at {level}: got {got}, want {want}"
                );
            }
        }
    }
}

#[test]
fn argmax_matches_reference() {
    type Argmax = extern "C" fn(*const f32, i64) -> i64;
    for (level, engine) in engines("argmax") {
        let k: Argmax = unsafe { std::mem::transmute(engine.get_fn_ptr("argmax").unwrap()) };
        let mut rng = Rng(41);
        // Llama 3's vocabulary is 128256 tokens.
        for n in LENGTHS.iter().copied().chain([128256]) {
            let base = rng.vec(n);
            let negative: Vec<f32> = base.iter().map(|v| v - 10.0).collect();
            let mut tie = base.clone();
            let top = base.iter().cloned().fold(f32::MIN, f32::max) + 1.0;
            tie[n / 3] = top;
            tie[n - 1] = top;
            // 64 apart: a multiple of every lane count, so both land in one lane.
            let mut same_lane = base.clone();
            if n > 200 {
                same_lane[n / 4] = top;
                same_lane[n / 4 + 64] = top;
            }
            // Equal maxima across lanes and accumulator sets at every vector width, with the
            // first one in each of the four sets in turn.
            let spreads: Vec<Vec<f32>> = [0, 4, 8, 12]
                .iter()
                .map(|&start| {
                    let mut v = base.clone();
                    if n > 200 {
                        for d in [0, 4, 8, 12, 16, 32] {
                            v[n / 5 + start + d] = top;
                        }
                    }
                    v
                })
                .collect();
            let mut in_tail = base.clone();
            in_tail[n - 1] = top;
            let mut at_zero = base.clone();
            at_zero[0] = top;
            for (name, x) in [
                ("random", &base),
                ("negative", &negative),
                ("tie", &tie),
                ("tie in one lane", &same_lane),
                ("ties across sets +0", &spreads[0]),
                ("ties across sets +4", &spreads[1]),
                ("ties across sets +8", &spreads[2]),
                ("ties across sets +12", &spreads[3]),
                ("tail", &in_tail),
                ("first", &at_zero),
            ] {
                let want = x
                    .iter()
                    .enumerate()
                    .fold(
                        (0, f32::MIN),
                        |(bi, bv), (i, &v)| {
                            if v > bv {
                                (i, v)
                            } else {
                                (bi, bv)
                            }
                        },
                    )
                    .0;
                let got = k(x.as_ptr(), n as i64);
                assert_eq!(got, want as i64, "argmax {name} n={n} at {level}");
            }
        }
    }
}

#[test]
fn rope_matches_reference() {
    type Rope = extern "C" fn(*mut f32, i64, i64, *const f32, *const f32);
    for (level, engine) in engines("rope") {
        let k: Rope = unsafe { std::mem::transmute(engine.get_fn_ptr("rope").unwrap()) };
        let mut rng = Rng(51);
        for (heads, dim, pos) in [
            (1, 2, 0),
            (3, 6, 5),
            (2, 34, 17),
            (32, 64, 1000),
            (8, 128, 77),
        ] {
            let half = dim / 2;
            let freq = |j: usize| 500000f64.powf(-2.0 * j as f64 / dim as f64);
            let cos: Vec<f32> = (0..half)
                .map(|j| (pos as f64 * freq(j)).cos() as f32)
                .collect();
            let sin: Vec<f32> = (0..half)
                .map(|j| (pos as f64 * freq(j)).sin() as f32)
                .collect();
            let x0 = rng.vec(heads * dim);
            let mut x = x0.clone();
            k(
                x.as_mut_ptr(),
                heads as i64,
                dim as i64,
                cos.as_ptr(),
                sin.as_ptr(),
            );
            for h in 0..heads {
                for j in 0..half {
                    let (x1, x2) = (x0[h * dim + j] as f64, x0[h * dim + half + j] as f64);
                    let (c, s) = (cos[j] as f64, sin[j] as f64);
                    let tol = 1e-6 * (x1.abs() + x2.abs()) + 1e-30;
                    for (got, want, which) in [
                        (x[h * dim + j], x1 * c - x2 * s, "x1"),
                        (x[h * dim + half + j], x2 * c + x1 * s, "x2"),
                    ] {
                        assert!(
                            (got as f64 - want).abs() <= tol,
                            "rope {heads}x{dim} head {h} {which}[{j}] at {level}: got {got}, want {want}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn add_rmsnorm_matches_reference() {
    type AddNorm = extern "C" fn(*const f32, *mut f32, *const f32, *mut f32, i64, f32) -> f32;
    for (level, engine) in engines("add_rmsnorm") {
        let k: AddNorm = unsafe { std::mem::transmute(engine.get_fn_ptr("add_rmsnorm").unwrap()) };
        let mut rng = Rng(61);
        for n in LENGTHS {
            let x = rng.vec(n);
            let res0 = rng.vec(n);
            let w = rng.vec(n);
            let mut res = res0.clone();
            let mut out = vec![f32::NAN; n];
            let eps = 1e-5f32;
            k(
                x.as_ptr(),
                res.as_mut_ptr(),
                w.as_ptr(),
                out.as_mut_ptr(),
                n as i64,
                eps,
            );
            let summed: Vec<f32> = res0.iter().zip(&x).map(|(r, v)| r + v).collect();
            assert_eq!(res, summed, "add_rmsnorm residual n={n} at {level}");
            let ms = summed.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / n as f64;
            let inv = 1.0 / (ms + eps as f64).sqrt();
            for i in 0..n {
                let want = summed[i] as f64 * inv * w[i] as f64;
                let got = out[i] as f64;
                close(
                    got,
                    want,
                    (summed[i] as f64 * inv * w[i] as f64).abs(),
                    &format!("add_rmsnorm n={n} [{i}] at {level}"),
                );
            }
        }
    }
}

/// Q8_0 or Q4_0 weights in `q8_gemv`'s / `q4_gemv`'s chunked layout (64 output rows per
/// chunk; per block, groups of 4 consecutive values of a row in one 32-bit lane) with f16
/// scales, plus the logical values q (q - 8 for Q4_0) and the scales as f32. Scales include
/// negative values (Q4_0's usually are), zero and f16 subnormals.
fn q_pack(rng: &mut Rng, m: usize, k: usize, q4: bool) -> (Vec<u8>, Vec<u16>, Vec<i8>, Vec<f32>) {
    let nb = k / 32;
    let q: Vec<i8> = (0..m * k)
        .map(|_| {
            if q4 {
                ((rng.next() * 8.0).floor() as i32).clamp(-8, 7) as i8
            } else {
                (rng.next() * 63.5) as i8
            }
        })
        .collect();
    let s16: Vec<half::f16> = (0..nb * m)
        .map(|i| {
            let v = match i % 17 {
                0 => 0.0,
                1 => 3.0e-6 * rng.next(), // f16 subnormal
                _ => (0.001 + rng.next().abs() * 0.01) * rng.next().signum(),
            };
            half::f16::from_f32(v)
        })
        .collect();
    let s: Vec<f32> = s16.iter().map(|v| v.to_f32()).collect();
    let mut wq = vec![0u8; if q4 { m * k / 2 } else { m * k }];
    let mut sc = vec![0u16; nb * m];
    for c in 0..m / 64 {
        for kk in 0..k {
            for r in 0..64 {
                let v = q[(c * 64 + r) * k + kk];
                // Byte q of group g's 32-bit lane for row r (see `qmat_chunk`).
                let (b, t) = (kk / 32, kk % 32);
                if q4 {
                    let n = t % 16;
                    wq[c * k * 32 + b * 1024 + ((n / 4) * 64 + r) * 4 + n % 4] |=
                        ((v + 8) as u8) << (4 * (t / 16));
                } else {
                    wq[c * k * 64 + b * 2048 + ((t / 4) * 64 + r) * 4 + t % 4] = v as u8;
                }
            }
        }
        for b in 0..nb {
            for r in 0..64 {
                // s is stored [b][i]; the kernel reads chunk c's [b][r].
                sc[c * nb * 64 + b * 64 + r] = s16[b * m + c * 64 + r].to_bits();
            }
        }
    }
    (wq, sc, q, s)
}

/// x quantized per 32-block exactly as `quantize_q8` does (f32 arithmetic).
fn q8_quantize(x: &[f32]) -> (Vec<i8>, Vec<f32>) {
    let mut xq = vec![0i8; x.len()];
    let mut dx = vec![0f32; x.len() / 32];
    for b in 0..x.len() / 32 {
        let blk = &x[b * 32..b * 32 + 32];
        let d = blk.iter().fold(0f32, |a, v| a.max(v.abs())) / 127.0;
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };
        dx[b] = d;
        for t in 0..32 {
            let v = blk[t] * id;
            let r = v + if v >= 0.0 { 0.5 } else { -0.5 };
            xq[b * 32 + t] = r as i32 as i8;
        }
    }
    (xq, dx)
}

#[test]
fn q8_gemv_matches_reference() {
    check_q_gemv("q8_gemv", false);
}

#[test]
fn q4_gemv_matches_reference() {
    check_q_gemv("q4_gemv", true);
}

fn check_q_gemv(name: &str, q4: bool) {
    type Gemv = extern "C" fn(*const u8, *const u16, *const f32, *mut f32, i64, i64);
    for (level, engine) in engines(name) {
        let kern: Gemv = unsafe { std::mem::transmute(engine.get_fn_ptr(name).unwrap()) };
        let mut rng = Rng(71);
        for (m, k) in [(64, 32), (128, 96), (64, 1024), (320, 512)] {
            let (wq, sc, q, s) = q_pack(&mut rng, m, k, q4);
            let mut x: Vec<f32> = rng.vec(k);
            // An all-zero block quantizes with scale 0.
            x[..32].iter_mut().for_each(|v| *v = 0.0);
            let mut y = vec![f32::NAN; m];
            kern(
                wq.as_ptr(),
                sc.as_ptr(),
                x.as_ptr(),
                y.as_mut_ptr(),
                m as i64,
                k as i64,
            );
            let (xq, dx) = q8_quantize(&x);
            for i in 0..m {
                let terms: Vec<f64> = (0..k / 32)
                    .map(|b| {
                        let dot: i64 = (0..32)
                            .map(|t| q[i * k + b * 32 + t] as i64 * xq[b * 32 + t] as i64)
                            .sum();
                        s[b * m + i] as f64 * dx[b] as f64 * dot as f64
                    })
                    .collect();
                let want: f64 = terms.iter().sum();
                let scale: f64 = terms.iter().map(|t| t.abs()).sum();
                let tol = (k / 32 + 4) as f64 * f32::EPSILON as f64 * scale + 1e-30;
                assert!(
                    (y[i] as f64 - want).abs() <= tol,
                    "{name} {m}x{k} y[{i}] at {level}: got {}, want {want}, tol {tol}",
                    y[i]
                );
            }
        }
    }
}
