//! Scalar float `min`/`max` in default and `fast` functions, on the backend selected by
//! `ACHAINSAW_BACKEND`: results follow AIR's semantics for every special value, and the
//! optimizer folds constant operands to exactly what the JIT computes at run time.

use achainsaw_codegen::{JitEngine, RtValue};
use achainsaw_ir::opt::optimize_module;
use achainsaw_ir::{decode_module, encode_module, parse_and_validate, to_air_text};

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

/// AIR's float min/max: by default NaN if either is NaN and -0.0 below +0.0; in `fast`
/// functions `a < b ? a : b` / `a > b ? a : b`.
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
            for (suffix, attr) in [("", ""), ("_fast", " fast")] {
                src += &format!(
                    "fn {op}_{ty}{suffix}(a:{ty}, b:{ty})->{ty}{attr}\n  b0:\n    r = {op} a, b\n    ret r\n\n"
                );
            }
        }
    }
    src
}

#[test]
fn scalar_min_max_follow_air_semantics() {
    let module = parse_and_validate(&source()).unwrap();
    let mut engine = JitEngine::new().unwrap();
    engine.compile_module(&module).unwrap();
    for ty in ["f32", "f64"] {
        for (op, is_max) in [("min", false), ("max", true)] {
            for (suffix, fast) in [("", false), ("_fast", true)] {
                let name = format!("{op}_{ty}{suffix}");
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
                            "{name}({a:?}, {b:?}) = {got:?}, want {want:?}"
                        );
                    }
                }
            }
        }
    }
}

/// Constant folding gives what the JIT computes: `min(NaN, 1.0)` used to fold to 1.0
/// (Rust's `f64::min`) while the program returns NaN.
#[test]
fn folded_min_max_match_runtime() {
    let runtime = {
        let module = parse_and_validate(&source()).unwrap();
        let mut engine = JitEngine::new().unwrap();
        engine.compile_module(&module).unwrap();
        engine
    };
    for ty in ["f32", "f64"] {
        for op in ["min", "max"] {
            for (suffix, attr) in [("", ""), ("_fast", " fast")] {
                for &a in &SPECIALS {
                    for &b in &SPECIALS {
                        let lit = |v: f64| {
                            if v.is_nan() {
                                "nan".to_string()
                            } else if v.is_infinite() {
                                if v > 0.0 { "inf" } else { "-inf" }.to_string()
                            } else {
                                format!("{v:?}")
                            }
                        };
                        // Constants via `cst` (negative infinity as 0 - inf).
                        let mk = |reg: &str, v: f64| {
                            if v == f64::NEG_INFINITY {
                                format!("    {reg}0 = cst inf:{ty}\n    {reg}z = cst 0.0:{ty}\n    {reg} = sub {reg}z, {reg}0\n")
                            } else {
                                format!("    {reg} = cst {}:{ty}\n", lit(v))
                            }
                        };
                        let src = format!(
                            "fn k()->{ty}{attr}\n  b0:\n{}{}    r = {op} a, b\n    ret r\n",
                            mk("a", a),
                            mk("b", b)
                        );
                        let mut module = parse_and_validate(&src).unwrap();
                        optimize_module(&mut module);
                        let mut folded = JitEngine::new().unwrap();
                        folded.compile_module(&module).unwrap();
                        let got = match unsafe { folded.call_typed("k", &[]) }.unwrap() {
                            Some(RtValue::F32(v)) => v as f64,
                            Some(RtValue::F64(v)) => v,
                            other => panic!("{other:?}"),
                        };
                        let want = run(&runtime, &format!("{op}_{ty}{suffix}"), ty, a, b);
                        assert!(
                            same(got, want),
                            "{op}_{ty}{suffix} folded({a:?}, {b:?}) = {got:?}, runtime {want:?}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn fast_attribute_round_trips() {
    let src = "fn k(a:f32, b:f32)->f32 fast\n  b0:\n    r = max a, b\n    ret r\n\nfn plain(fast:i32)->i32\n  b0:\n    r = add fast, 1:i32\n    ret r\n";
    let module = parse_and_validate(src).unwrap();
    assert!(module.functions[0].fast);
    assert!(!module.functions[1].fast);
    let text = to_air_text(&module);
    assert!(text.contains("fn k(a:f32, b:f32)->f32 fast\n"), "{text}");
    let decoded = decode_module(&encode_module(&module).unwrap()).unwrap();
    assert!(decoded.functions[0].fast && !decoded.functions[1].fast);
    assert_eq!(to_air_text(&decoded), text);
    // Without a return type the attribute follows the parameters.
    let module = parse_and_validate("fn v(p:ptr) fast\n  b0:\n    ret\n").unwrap();
    assert!(module.functions[0].fast);
}
