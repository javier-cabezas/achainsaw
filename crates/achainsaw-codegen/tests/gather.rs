//! `vgather` on the backend selected by `ACHAINSAW_BACKEND`: every lane type and vector
//! width, with and without a lane count, at every ISA level this host reaches, against a
//! scalar reference; object code for every target. `sandbox.rs` covers its bounds checks.

use achainsaw_codegen::cpu::{CpuFeatures, IsaLevel};
use achainsaw_codegen::JitEngine;
use achainsaw_ir::parse_and_validate;
use achainsaw_ir::types::Type;

const LANES: [Type; 4] = [Type::I32, Type::I64, Type::F32, Type::F64];
const WIDTHS: [&str; 4] = ["v128", "v256", "v512", "vx"];
/// Table elements; the kernels index from its middle, so indices run from -HALF to HALF - 1.
const HALF: i64 = 512;

type Kernel = extern "C" fn(*const u8, *const u8, *mut u8, i64);

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
    format!("g_{lane}_{w}_{}", if counted { "n" } else { "all" })
}

/// `fn g_<lane>_<w>_<n|all>(t:ptr, ip:ptr, out:ptr, n:i64)`: gathers `t[idx]` for the index
/// vector at `ip` into `out`.
fn source() -> String {
    let mut src = String::new();
    for lane in LANES {
        for w in WIDTHS {
            for counted in [false, true] {
                let count = if counted { ", n" } else { "" };
                src += &format!(
                    "fn {}(t:ptr, ip:ptr, out:ptr, n:i64)\n  b0:\n    idx = ld ip:{w}\n    v = vgather t, idx{count}:{lane}\n    st out, v\n    ret\n\n",
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
fn gathers_match_a_scalar_reference_at_every_width_and_level() {
    let module = parse_and_validate(&source()).unwrap_or_else(|d| panic!("{d:?}"));
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    for (level, features) in host_levels() {
        let mut engine = JitEngine::with_features(&features).unwrap();
        engine.compile_module(&module).unwrap();
        let vx_bytes = engine.vx_bits() as usize / 8;
        for lane in LANES {
            let size = lane.byte_size();
            // Distinct bit patterns, so a wrong element shows.
            let table: Vec<u8> = (0..2 * HALF as usize * size)
                .map(|_| rng.next() as u8)
                .collect();
            let mid = unsafe { table.as_ptr().add(HALF as usize * size) };
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
                        let idx: Vec<i64> = (0..lanes)
                            .map(|i| match (trial, i) {
                                (0, 0) => -HALF,
                                (0, 1) => HALF - 1,
                                _ => (rng.next() % (2 * HALF as u64)) as i64 - HALF,
                            })
                            .collect();
                        let mut ip = vec![0u8; bytes];
                        for (i, &x) in idx.iter().enumerate() {
                            ip[i * size..(i + 1) * size].copy_from_slice(&x.to_le_bytes()[..size]);
                        }
                        for &n in &counts {
                            let mut out = vec![0xAAu8; bytes];
                            f(mid, ip.as_ptr(), out.as_mut_ptr(), n);
                            let read = n.clamp(0, lanes as i64) as usize;
                            for i in 0..lanes {
                                let got = &out[i * size..(i + 1) * size];
                                let at = ((idx[i] + HALF) as usize) * size;
                                let want = if i < read {
                                    &table[at..at + size]
                                } else {
                                    &[0u8; 8][..size]
                                };
                                assert_eq!(
                                    got, want,
                                    "{lane} {w} n={n} lane {i} (index {}) at {level}",
                                    idx[i]
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

/// x86-64 (AVX2 and AVX-512 gathers) and AArch64 (NEON, SVE) objects on Cranelift and,
/// with the `llvm` feature, LLVM.
#[test]
fn gathers_compile_for_every_target() {
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

/// A counted gather is a masked access to `split_masked_loops`: the loop's full-vector copy
/// gathers without the count, and the tail keeps it.
#[test]
fn counted_gathers_lose_their_count_in_full_vector_loops() {
    use achainsaw_ir::ast::Instruction;
    let src = r#"
fn k(t:ptr, ix:ptr, y:ptr, n:i64)
  b0:
    w = vl f32
    jmp loop(0:i64)
  loop(i:i64):
    more = lt i, n
    br more, body, done
  body:
    rest = sub n, i
    c = ldm ix[i]:vx, rest:i32
    g = vgather t, c, rest:f32
    stm y[i], g, rest:f32
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
            Instruction::VGather { count, .. } => Some(count.is_some()),
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
    let table: Vec<f32> = (0..100).map(|i| i as f32 * 0.5).collect();
    for n in [0usize, 1, 3, 4, 17, 64, 99] {
        let ix: Vec<i32> = (0..n).map(|i| (i * 37 % 100) as i32).collect();
        let mut y = vec![-1.0f32; n + 1];
        f(table.as_ptr(), ix.as_ptr(), y.as_mut_ptr(), n as i64);
        for i in 0..n {
            assert_eq!(y[i], table[ix[i] as usize], "n={n} i={i}");
        }
        assert_eq!(y[n], -1.0, "n={n}: wrote past the end");
    }
}
