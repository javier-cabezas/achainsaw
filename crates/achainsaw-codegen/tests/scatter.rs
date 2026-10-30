//! `vscatter` on the backend selected by `ACHAINSAW_BACKEND`: every lane type and vector
//! width, with and without a lane count, at every ISA level this host reaches, against a
//! scalar reference (duplicate indices included: the highest lane stays); object code for
//! every target. `sandbox.rs` covers its bounds checks, `data_stack.rs` read-only data.

use achainsaw_codegen::cpu::{CpuFeatures, IsaLevel};
use achainsaw_codegen::JitEngine;
use achainsaw_ir::parse_and_validate;
use achainsaw_ir::types::Type;

const LANES: [Type; 4] = [Type::I32, Type::I64, Type::F32, Type::F64];
const WIDTHS: [&str; 4] = ["v128", "v256", "v512", "vx"];
/// Table elements; the kernels index from its middle, so indices run from -HALF to HALF - 1.
const HALF: i64 = 512;

type Kernel = extern "C" fn(*mut u8, *const u8, *const u8, i64);

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

fn name(lane: Type, w: &str, counted: bool) -> String {
    format!("s_{lane}_{w}_{}", if counted { "n" } else { "all" })
}

/// `fn s_<lane>_<w>_<n|all>(t:ptr, ip:ptr, vp:ptr, n:i64)`: scatters the vector at `vp` to
/// `t[idx]` for the index vector at `ip`.
fn source() -> String {
    let mut src = String::new();
    for lane in LANES {
        for w in WIDTHS {
            for counted in [false, true] {
                let count = if counted { ", n" } else { "" };
                src += &format!(
                    "fn {}(t:ptr, ip:ptr, vp:ptr, n:i64)\n  b0:\n    idx = ld ip:{w}\n    v = ld vp:{w}\n    vscatter t, idx, v{count}:{lane}\n    ret\n\n",
                    name(lane, w, counted)
                );
            }
        }
    }
    src
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

#[test]
fn scatters_match_a_scalar_reference_at_every_width_and_level() {
    let module = parse_and_validate(&source()).unwrap_or_else(|d| panic!("{d:?}"));
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    for (level, features) in host_levels() {
        let mut engine = JitEngine::with_features(&features).unwrap();
        engine.compile_module(&module).unwrap();
        let vx_bytes = engine.vx_bits() as usize / 8;
        for lane in LANES {
            let size = lane.byte_size();
            for w in WIDTHS {
                let bytes = match w {
                    "v128" => 16,
                    "v256" => 32,
                    "v512" => 64,
                    _ => vx_bytes,
                };
                let lanes = bytes / size;
                for counted in [false, true] {
                    let f: Kernel = unsafe {
                        std::mem::transmute(engine.get_fn_ptr(&name(lane, w, counted)).unwrap())
                    };
                    let counts: Vec<i64> = if counted {
                        vec![-1, 0, 1, lanes as i64 - 1, lanes as i64, lanes as i64 + 5]
                    } else {
                        vec![lanes as i64]
                    };
                    for trial in 0..8 {
                        // Trial 0 reaches both ends of the table, trial 1 sends every lane
                        // to one element and trial 2 pairs of lanes; the rest are random, so
                        // with 1024 elements and up to 64 lanes they collide too.
                        let idx: Vec<i64> = (0..lanes)
                            .map(|i| match (trial, i) {
                                (0, 0) => -HALF,
                                (0, 1) => HALF - 1,
                                (1, _) => 7,
                                (2, _) => (i / 2) as i64 - 3,
                                _ => (rng.next() % (2 * HALF as u64)) as i64 - HALF,
                            })
                            .collect();
                        let mut ip = vec![0u8; bytes];
                        for (i, &x) in idx.iter().enumerate() {
                            ip[i * size..(i + 1) * size].copy_from_slice(&x.to_le_bytes()[..size]);
                        }
                        let vals: Vec<u8> = (0..bytes).map(|_| rng.next() as u8).collect();
                        for &n in &counts {
                            let mut table = vec![0xAAu8; 2 * HALF as usize * size];
                            let mut want = table.clone();
                            let stored = n.clamp(0, lanes as i64) as usize;
                            for i in 0..stored {
                                let at = ((idx[i] + HALF) as usize) * size;
                                want[at..at + size]
                                    .copy_from_slice(&vals[i * size..(i + 1) * size]);
                            }
                            let mid = unsafe { table.as_mut_ptr().add(HALF as usize * size) };
                            f(mid, ip.as_ptr(), vals.as_ptr(), n);
                            for e in 0..2 * HALF as usize {
                                assert_eq!(
                                    table[e * size..(e + 1) * size],
                                    want[e * size..(e + 1) * size],
                                    "{lane} {w} n={n} trial {trial}: element {} at {level}",
                                    e as i64 - HALF
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

/// x86-64 (AVX-512 scatters, lane stores below it) and AArch64 (NEON, SVE) objects on
/// Cranelift and, with the `llvm` feature, LLVM.
#[test]
fn scatters_compile_for_every_target() {
    use achainsaw_codegen::{compile_object, AotTarget, Backend, CodegenOptions};
    let module = parse_and_validate(&source()).unwrap();
    let mut backends = vec![Backend::Cranelift];
    if cfg!(feature = "llvm") {
        backends.push(Backend::Llvm);
    }
    for (triple, cpu, features) in [
        ("x86_64-unknown-linux-gnu", "x86-64", ""),
        ("x86_64-unknown-linux-gnu", "x86-64-v3", ""),
        ("x86_64-unknown-linux-gnu", "x86-64-v4", ""),
        ("aarch64-unknown-linux-gnu", "generic", ""),
        ("aarch64-unknown-linux-gnu", "generic", "+sve"),
        ("aarch64-unknown-linux-gnu", "generic", "+sve2"),
    ] {
        for &backend in &backends {
            if backend == Backend::Cranelift && !features.is_empty() {
                continue;
            }
            let target = AotTarget {
                triple: Some(triple.into()),
                cpu: Some(cpu.into()),
                features: (!features.is_empty()).then(|| features.into()),
            };
            let obj = compile_object(&module, &target, backend, &CodegenOptions::default())
                .unwrap_or_else(|e| panic!("{triple} {cpu} {features} {backend:?}: {e}"));
            assert!(!obj.bytes.is_empty());
        }
    }
}

/// A counted scatter is a masked access to `split_masked_loops`: the loop's full-vector copy
/// scatters without the count, and the tail keeps it. The kernel permutes `x` into `y`.
#[test]
fn counted_scatters_lose_their_count_in_full_vector_loops() {
    use achainsaw_ir::ast::Instruction;
    let src = r#"
fn k(x:ptr, ix:ptr, y:ptr, n:i64)
  b0:
    w = vl f32
    jmp loop(0:i64)
  loop(i:i64):
    more = lt i, n
    br more, body, done
  body:
    rest = sub n, i
    c = ldm ix[i]:vx, rest:i32
    v = ldm x[i]:vx, rest:f32
    vscatter y, c, v, rest:f32
    i2 = add i, w
    jmp loop(i2)
  done:
    ret
"#;
    let module = parse_and_validate(src).unwrap();
    let prepared = achainsaw_ir::prepare_for_codegen(&module);
    let counts: Vec<bool> = prepared.functions[0]
        .blocks
        .iter()
        .flat_map(|b| &b.instructions)
        .filter_map(|i| match i {
            Instruction::VScatter { count, .. } => Some(count.is_some()),
            _ => None,
        })
        .collect();
    assert_eq!(counts.len(), 2, "{counts:?}");
    assert!(
        counts.contains(&true) && counts.contains(&false),
        "{counts:?}"
    );

    let mut e = JitEngine::new().unwrap();
    e.compile_module(&module).unwrap();
    type K = extern "C" fn(*const f32, *const i32, *mut f32, i64);
    let f: K = unsafe { std::mem::transmute(e.get_fn_ptr("k").unwrap()) };
    for n in [0usize, 1, 3, 4, 17, 64, 99] {
        // A permutation of 0..n: 37 is coprime to each n.
        let ix: Vec<i32> = (0..n).map(|i| (i * 37 % n.max(1)) as i32).collect();
        let x: Vec<f32> = (0..n).map(|i| i as f32 * 0.5 + 1.0).collect();
        let mut y = vec![-1.0f32; n + 1];
        f(x.as_ptr(), ix.as_ptr(), y.as_mut_ptr(), n as i64);
        for i in 0..n {
            assert_eq!(y[ix[i] as usize], x[i], "n={n} i={i}");
        }
        assert_eq!(y[n], -1.0, "n={n}: wrote past the end");
    }
}
