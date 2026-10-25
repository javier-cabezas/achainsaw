//! The math functions (`exp`, `log`, `tanh`, `sin`, `cos` and `vexp` ... `vcos`) on the
//! backend selected by `ACHAINSAW_BACKEND`: scalar and at every vector width, at every ISA
//! level this host reaches, bit for bit against `achainsaw_ir::math::eval` (the definition
//! both backends compile; `achainsaw-ir/tests/math.rs` checks its accuracy).

use achainsaw_codegen::cpu::{CpuFeatures, IsaLevel};
use achainsaw_codegen::JitEngine;
use achainsaw_ir::math::{eval, MathFn};
use achainsaw_ir::parse_and_validate;

const NAMES: [(MathFn, &str); 5] = [
    (MathFn::Exp, "exp"),
    (MathFn::Log, "log"),
    (MathFn::Tanh, "tanh"),
    (MathFn::Sin, "sin"),
    (MathFn::Cos, "cos"),
];

const WIDTHS: [&str; 4] = ["v128", "v256", "v512", "vx"];

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

/// `s_<f>(x:f32)->f32` and `<f>_<width>(p:ptr)`, which replaces the vector at `p` with its
/// image, for every function and width.
fn source() -> String {
    let mut src = String::new();
    for (_, name) in NAMES {
        src += &format!("fn s_{name}(x:f32)->f32\n  b0:\n    r = {name} x\n    ret r\n\n");
        for w in WIDTHS {
            src += &format!(
                "fn {name}_{w}(p:ptr)\n  b0:\n    v = ld p:{w}\n    r = v{name} v:f32\n    st p, r\n    ret\n\n"
            );
        }
    }
    src
}

/// Special values, every power of two and its neighbors, a dense sweep of [-10, 10],
/// values near multiples of pi/2 and random bit patterns.
fn inputs() -> Vec<f32> {
    let mut v = vec![
        0.0,
        -0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        f32::MIN_POSITIVE,
        f32::from_bits(1),
        f32::MAX,
        f32::MIN,
        -87.0,
        88.0,
        0.625,
        -0.625,
        4_194_304.0,
        4_194_305.0,
    ];
    for e in -149..128 {
        let p = 2f32.powi(e);
        for x in [
            p,
            -p,
            f32::from_bits(p.to_bits() + 1),
            f32::from_bits(p.to_bits().wrapping_sub(1)),
        ] {
            v.push(x);
        }
    }
    v.extend((0..20_000).map(|i| -10.0 + i as f32 * 1e-3));
    for k in -1000..1000 {
        let x = k as f32 * std::f32::consts::FRAC_PI_2;
        v.extend([
            x,
            f32::from_bits(x.to_bits() + 1),
            f32::from_bits(x.to_bits().wrapping_sub(1)),
        ]);
    }
    let mut s = 0x9e37_79b9_7f4a_7c15u64;
    for _ in 0..50_000 {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        v.push(f32::from_bits((s >> 32) as u32));
    }
    v
}

fn same(got: f32, want: f32) -> bool {
    got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan())
}

#[test]
fn math_functions_match_their_definition_at_every_width_and_level() {
    let module = parse_and_validate(&source()).unwrap_or_else(|d| panic!("{d:?}"));
    let xs = inputs();
    for (level, features) in host_levels() {
        let mut engine = JitEngine::with_features(&features).unwrap();
        engine.compile_module(&module).unwrap();
        let vx_bytes = engine.vx_bits() as usize / 8;
        for (f, name) in NAMES {
            let want: Vec<f32> = xs.iter().map(|&x| eval(f, x)).collect();
            let scalar: extern "C" fn(f32) -> f32 =
                unsafe { std::mem::transmute(engine.get_fn_ptr(&format!("s_{name}")).unwrap()) };
            for (&x, &w) in xs.iter().zip(&want) {
                let g = scalar(x);
                assert!(
                    same(g, w),
                    "{name}({x:e}) at {level}: got {g:e}, want {w:e}"
                );
            }
            for (wname, bytes) in [("v128", 16), ("v256", 32), ("v512", 64), ("vx", vx_bytes)] {
                let k: extern "C" fn(*mut f32) = unsafe {
                    std::mem::transmute(engine.get_fn_ptr(&format!("{name}_{wname}")).unwrap())
                };
                let lanes = bytes / 4;
                for (chunk, want) in xs.chunks(lanes).zip(want.chunks(lanes)) {
                    let mut buf = vec![0f32; lanes];
                    buf[..chunk.len()].copy_from_slice(chunk);
                    k(buf.as_mut_ptr());
                    for i in 0..chunk.len() {
                        assert!(
                            same(buf[i], want[i]),
                            "v{name} {wname} lane {i} ({:e}) at {level}: got {:e}, want {:e}",
                            chunk[i],
                            buf[i],
                            want[i]
                        );
                    }
                }
            }
        }
    }
}

/// The optimizer folds a math function of a constant to `math::eval`'s value, which is what
/// the compiled code computes.
#[test]
fn folded_math_constants_match_runtime() {
    for (f, name) in NAMES {
        for x in [0.5f32, -3.25, 100.0, 1e-3] {
            let src = format!(
                "fn folded()->f32\n  b0:\n    x = cst {x:e}:f32\n    r = {name} x\n    ret r\n\nfn runtime(x:f32)->f32\n  b0:\n    r = {name} x\n    ret r\n"
            );
            let mut module = parse_and_validate(&src).unwrap();
            achainsaw_ir::opt::optimize_module(&mut module);
            let folded = &module.functions[0];
            assert!(
                folded.blocks[0]
                    .instructions
                    .iter()
                    .all(|i| !matches!(i, achainsaw_ir::ast::Instruction::Unary { .. })),
                "{name} of a constant is folded"
            );
            let mut engine = JitEngine::new().unwrap();
            engine.compile_module(&module).unwrap();
            let a: extern "C" fn() -> f32 =
                unsafe { std::mem::transmute(engine.get_fn_ptr("folded").unwrap()) };
            let b: extern "C" fn(f32) -> f32 =
                unsafe { std::mem::transmute(engine.get_fn_ptr("runtime").unwrap()) };
            assert_eq!(a().to_bits(), b(x).to_bits(), "{name}({x})");
            assert_eq!(a().to_bits(), eval(f, x).to_bits(), "{name}({x})");
        }
    }
}

/// x86-64 and AArch64 objects (SVE's scalable `vx` too) on Cranelift and, with the `llvm`
/// feature, LLVM.
#[test]
fn math_functions_compile_for_every_target() {
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
    ] {
        for &backend in &backends {
            if backend == Backend::Cranelift && features == "+sve" {
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
