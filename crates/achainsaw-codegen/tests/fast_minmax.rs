//! Scalar float `min`/`max` compiled with and without `fast_math`, on the backend selected
//! by `ACHAINSAW_BACKEND`: results follow AIR's semantics for every special value, and the
//! optimizer folds constant operands to exactly what the JIT computes at run time in either
//! mode (it runs before code generation, so its folds must not depend on the mode).

use achainsaw_codegen::{JitEngine, RtValue};
use achainsaw_ir::opt::optimize_module;
use achainsaw_ir::parse_and_validate;

const SPECIALS: [f64; 9] = [
    f64::NAN,
    0.0,
    -0.0,
    1.0,
    -1.0,
    1.5,
    f64::INFINITY,
    f64::NEG_INFINITY,
    3.25e-3,
];

/// AIR's float min/max: by default NaN if either is NaN and -0.0 below +0.0; with
/// `fast_math` `a < b ? a : b` / `a > b ? a : b`.
fn reference(is_max: bool, fast: bool, a: f64, b: f64) -> f64 {
    if fast {
        let pick_a = if is_max { a > b } else { a < b };
        return if pick_a { a } else { b };
    }
    if a.is_nan() || b.is_nan() {
        return f64::NAN;
    }
    if a == b {
        let neg = a.is_sign_negative() || b.is_sign_negative();
        let pos = a.is_sign_positive() || b.is_sign_positive();
        return match (is_max, neg && pos) {
            (true, true) => 0.0,
            (false, true) => -0.0,
            _ => a,
        };
    }
    if is_max {
        a.max(b)
    } else {
        a.min(b)
    }
}

fn same(got: f64, want: f64) -> bool {
    (got.is_nan() && want.is_nan()) || got.to_bits() == want.to_bits()
}

fn run(engine: &JitEngine, name: &str, ty: &str, a: f64, b: f64) -> f64 {
    let args = if ty == "f32" {
        vec![RtValue::F32(a as f32), RtValue::F32(b as f32)]
    } else {
        vec![RtValue::F64(a), RtValue::F64(b)]
    };
    match unsafe { engine.call_typed(name, &args) }.unwrap() {
        Some(RtValue::F32(v)) => v as f64,
        Some(RtValue::F64(v)) => v,
        other => panic!("{name}: {other:?}"),
    }
}

fn source() -> String {
    let mut src = String::new();
    for ty in ["f32", "f64"] {
        for op in ["min", "max"] {
            src += &format!(
                "fn {op}_{ty}(a:{ty}, b:{ty})->{ty}\n  b0:\n    r = {op} a, b\n    ret r\n\n"
            );
        }
    }
    src
}

fn engine(fast_math: bool, src: &str) -> JitEngine {
    let module = parse_and_validate(src).unwrap();
    let mut engine = JitEngine::new().unwrap();
    engine.set_fast_math(fast_math);
    engine.compile_module(&module).unwrap();
    engine
}

#[test]
fn scalar_min_max_follow_air_semantics() {
    for fast in [false, true] {
        let engine = engine(fast, &source());
        for ty in ["f32", "f64"] {
            for (op, is_max) in [("min", false), ("max", true)] {
                let name = format!("{op}_{ty}");
                for &a in &SPECIALS {
                    for &b in &SPECIALS {
                        let got = run(&engine, &name, ty, a, b);
                        let want = reference(is_max, fast, a, b);
                        let want = if ty == "f32" {
                            want as f32 as f64
                        } else {
                            want
                        };
                        assert!(
                            same(got, want),
                            "{name}({a:?}, {b:?}) fast_math={fast} = {got:?}, want {want:?}"
                        );
                    }
                }
            }
        }
    }
}

/// Constant folding gives what the JIT computes in both modes: `min(NaN, 1.0)` used to fold
/// to 1.0 (Rust's `f64::min`) while the program returns NaN.
#[test]
fn folded_min_max_match_runtime() {
    let lit = |v: f64| {
        if v.is_nan() {
            "nan".to_string()
        } else if v.is_infinite() {
            if v > 0.0 { "inf" } else { "-inf" }.to_string()
        } else {
            format!("{v:?}")
        }
    };
    for fast in [false, true] {
        let runtime = engine(fast, &source());
        for ty in ["f32", "f64"] {
            // Constants via `cst` (negative infinity as 0 - inf).
            let mk = |reg: &str, v: f64| {
                if v == f64::NEG_INFINITY {
                    format!("    {reg}0 = cst inf:{ty}\n    {reg}z = cst 0.0:{ty}\n    {reg} = sub {reg}z, {reg}0\n")
                } else {
                    format!("    {reg} = cst {}:{ty}\n", lit(v))
                }
            };
            for op in ["min", "max"] {
                // One function per operand pair, folded together in one module.
                let mut src = String::new();
                for (i, &a) in SPECIALS.iter().enumerate() {
                    for (j, &b) in SPECIALS.iter().enumerate() {
                        src += &format!(
                            "fn k_{i}_{j}()->{ty}\n  b0:\n{}{}    r = {op} a, b\n    ret r\n\n",
                            mk("a", a),
                            mk("b", b)
                        );
                    }
                }
                let mut module = parse_and_validate(&src).unwrap();
                optimize_module(&mut module);
                let mut folded = JitEngine::new().unwrap();
                folded.set_fast_math(fast);
                folded.compile_module(&module).unwrap();
                for (i, &a) in SPECIALS.iter().enumerate() {
                    for (j, &b) in SPECIALS.iter().enumerate() {
                        let got = match unsafe { folded.call_typed(&format!("k_{i}_{j}"), &[]) }
                            .unwrap()
                        {
                            Some(RtValue::F32(v)) => v as f64,
                            Some(RtValue::F64(v)) => v,
                            other => panic!("{other:?}"),
                        };
                        let want = run(&runtime, &format!("{op}_{ty}"), ty, a, b);
                        assert!(
                            same(got, want),
                            "{op}_{ty} fast_math={fast} folded({a:?}, {b:?}) = {got:?}, runtime {want:?}"
                        );
                    }
                }
            }
        }
    }
}
