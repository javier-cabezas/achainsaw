//! Differential test: Cranelift and LLVM must agree bit for bit on scalar semantics,
//! including the edge cases AIR defines (division by zero, MIN / -1, shift amounts at or
//! above the bit width, NaN-propagating min/max, saturating float-to-int casts).
#![cfg(feature = "llvm")]

use achainsaw_codegen::cpu::CpuFeatures;
use achainsaw_codegen::{Backend, JitEngine, RtValue};
use achainsaw_ir::parse_and_validate;

const INT_OPS: &[&str] = &[
    "add", "sub", "mul", "div", "rem", "and", "or", "xor", "shl", "shr", "min", "max", "umin",
    "umax", "udiv", "urem", "ushr", "eq", "ne", "lt", "gt", "le", "ge", "ult", "ugt", "ule", "uge",
];
const FLOAT_OPS: &[&str] = &[
    "add", "sub", "mul", "div", "min", "max", "eq", "ne", "lt", "gt", "le", "ge",
];
const INT_TYPES: &[&str] = &["i8", "i16", "i32", "i64"];
const FLOAT_TYPES: &[&str] = &["f32", "f64"];

fn is_cmp(op: &str) -> bool {
    matches!(
        op,
        "eq" | "ne" | "lt" | "gt" | "le" | "ge" | "ult" | "ugt" | "ule" | "uge"
    )
}

fn module_src() -> String {
    let mut src = String::new();
    for ty in INT_TYPES {
        for op in INT_OPS {
            let ret = if is_cmp(op) { "i32" } else { ty };
            src += &format!(
                "fn {op}_{ty}(a:{ty}, b:{ty})->{ret}\n  b0:\n    r = {op} a, b\n    ret r\n\n"
            );
        }
        for op in ["neg", "abs"] {
            src += &format!("fn {op}_{ty}(a:{ty})->{ty}\n  b0:\n    r = {op} a\n    ret r\n\n");
        }
        for fty in FLOAT_TYPES {
            src += &format!(
                "fn itof_{ty}_{fty}(a:{ty})->{fty}\n  b0:\n    r = itof a:{fty}\n    ret r\n\n\
                 fn ftoi_{fty}_{ty}(a:{fty})->{ty}\n  b0:\n    r = ftoi a:{ty}\n    ret r\n\n"
            );
        }
    }
    for ty in FLOAT_TYPES {
        for op in FLOAT_OPS {
            let ret = if is_cmp(op) { "i32" } else { ty };
            src += &format!(
                "fn {op}_{ty}(a:{ty}, b:{ty})->{ret}\n  b0:\n    r = {op} a, b\n    ret r\n\n"
            );
        }
        for op in ["neg", "abs", "sqrt"] {
            src += &format!("fn {op}_{ty}(a:{ty})->{ty}\n  b0:\n    r = {op} a\n    ret r\n\n");
        }
    }
    src += "fn fext_f32(a:f32)->f64\n  b0:\n    r = fext a:f64\n    ret r\n\n";
    src += "fn ftrunc_f64(a:f64)->f32\n  b0:\n    r = ftrunc a:f32\n    ret r\n\n";
    src += "fn h_round(a:f32)->f32\n  b0:\n    h = ftrunc a:f16\n    r = fext h:f32\n    ret r\n\n";
    src +=
        "fn bf_round(a:f32)->f32\n  b0:\n    h = ftrunc a:bf16\n    r = fext h:f32\n    ret r\n\n";
    src += "fn sel(c:i32, a:i64, b:i64)->i64\n  b0:\n    r = select c, a, b\n    ret r\n";
    src
}

fn engines() -> (JitEngine, JitEngine) {
    let module = parse_and_validate(&module_src()).unwrap_or_else(|d| panic!("{d:?}"));
    let features = CpuFeatures::effective().unwrap();
    let mut clif = JitEngine::with_backend(Backend::Cranelift, &features).unwrap();
    let mut llvm = JitEngine::with_backend(Backend::Llvm, &features).unwrap();
    clif.compile_module(&module).unwrap();
    llvm.compile_module(&module).unwrap();
    (clif, llvm)
}

/// Bit pattern of a result; every NaN maps to one value since NaN payloads are not part
/// of AIR's semantics.
fn bits(v: Option<RtValue>) -> u64 {
    match v {
        Some(RtValue::F32(f)) if f.is_nan() => u64::MAX,
        Some(RtValue::F64(f)) if f.is_nan() => u64::MAX,
        Some(RtValue::F32(f)) => f.to_bits() as u64,
        Some(RtValue::F64(f)) => f.to_bits(),
        Some(RtValue::I8(n)) => n as u8 as u64,
        Some(RtValue::I16(n)) => n as u16 as u64,
        Some(RtValue::I32(n)) => n as u32 as u64,
        Some(RtValue::I64(n)) => n as u64,
        Some(RtValue::Ptr(p)) => p as u64,
        None => 0,
    }
}

fn agree(clif: &JitEngine, llvm: &JitEngine, name: &str, args: &[RtValue]) {
    let a = unsafe { clif.call_typed(name, args) }.unwrap();
    let b = unsafe { llvm.call_typed(name, args) }.unwrap();
    assert_eq!(
        bits(a),
        bits(b),
        "{name}{args:?}: cranelift {a:?}, llvm {b:?}"
    );
}

fn int_inputs(bits: u32) -> Vec<i64> {
    let min = if bits == 64 {
        i64::MIN
    } else {
        -(1i64 << (bits - 1))
    };
    let max = if bits == 64 {
        i64::MAX
    } else {
        (1i64 << (bits - 1)) - 1
    };
    let w = bits as i64;
    vec![
        0,
        1,
        -1,
        2,
        -2,
        7,
        -7,
        min,
        max,
        min + 1,
        max - 1,
        w - 1,
        w,
        w + 1,
        -w,
        3 * w + 5,
    ]
}

fn int_val(ty: &str, v: i64) -> RtValue {
    match ty {
        "i8" => RtValue::I8(v as i8),
        "i16" => RtValue::I16(v as i16),
        "i32" => RtValue::I32(v as i32),
        _ => RtValue::I64(v),
    }
}

fn float_inputs() -> Vec<f64> {
    vec![
        0.0,
        -0.0,
        1.0,
        -1.5,
        0.1,
        3.0e9,
        -3.0e9,
        1.0e20,
        -1.0e20,
        2147483647.5,
        -2147483648.5,
        65504.0,
        65520.0,
        1.0e-40,
        f64::MIN_POSITIVE,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
    ]
}

fn float_val(ty: &str, v: f64) -> RtValue {
    if ty == "f32" {
        RtValue::F32(v as f32)
    } else {
        RtValue::F64(v)
    }
}

#[test]
fn integer_ops_agree() {
    let (clif, llvm) = engines();
    for ty in INT_TYPES {
        let ins = int_inputs(ty[1..].parse().unwrap());
        for op in INT_OPS {
            for &a in &ins {
                for &b in &ins {
                    agree(
                        &clif,
                        &llvm,
                        &format!("{op}_{ty}"),
                        &[int_val(ty, a), int_val(ty, b)],
                    );
                }
            }
        }
        for op in ["neg", "abs"] {
            for &a in &ins {
                agree(&clif, &llvm, &format!("{op}_{ty}"), &[int_val(ty, a)]);
            }
        }
    }
}

#[test]
fn float_ops_agree() {
    let (clif, llvm) = engines();
    let ins = float_inputs();
    for ty in FLOAT_TYPES {
        for op in FLOAT_OPS {
            for &a in &ins {
                for &b in &ins {
                    agree(
                        &clif,
                        &llvm,
                        &format!("{op}_{ty}"),
                        &[float_val(ty, a), float_val(ty, b)],
                    );
                }
            }
        }
        for op in ["neg", "abs", "sqrt"] {
            for &a in &ins {
                agree(&clif, &llvm, &format!("{op}_{ty}"), &[float_val(ty, a)]);
            }
        }
    }
}

#[test]
fn conversions_agree() {
    let (clif, llvm) = engines();
    for ity in INT_TYPES {
        for fty in FLOAT_TYPES {
            for &a in &int_inputs(ity[1..].parse().unwrap()) {
                agree(
                    &clif,
                    &llvm,
                    &format!("itof_{ity}_{fty}"),
                    &[int_val(ity, a)],
                );
            }
            for &a in &float_inputs() {
                agree(
                    &clif,
                    &llvm,
                    &format!("ftoi_{fty}_{ity}"),
                    &[float_val(fty, a)],
                );
            }
        }
    }
    for &a in &float_inputs() {
        agree(&clif, &llvm, "fext_f32", &[RtValue::F32(a as f32)]);
        agree(&clif, &llvm, "ftrunc_f64", &[RtValue::F64(a)]);
        agree(&clif, &llvm, "h_round", &[RtValue::F32(a as f32)]);
        agree(&clif, &llvm, "bf_round", &[RtValue::F32(a as f32)]);
    }
    for c in [0, 1, -1, i32::MIN] {
        agree(
            &clif,
            &llvm,
            "sel",
            &[RtValue::I32(c), RtValue::I64(11), RtValue::I64(-22)],
        );
    }
}

#[test]
fn fuel_runs_out_at_the_same_point() {
    let src = "fn count(n:i64)->i64\n  b0:\n    jmp b1(0:i64)\n  b1(i:i64):\n    c = lt i, n\n    br c, b2, b3\n  b2:\n    i2 = add i, 1:i64\n    jmp b1(i2)\n  b3:\n    ret i\n";
    let module = parse_and_validate(src).unwrap();
    let features = CpuFeatures::effective().unwrap();
    let mut remaining = Vec::new();
    for backend in [Backend::Cranelift, Backend::Llvm] {
        let mut e = JitEngine::with_backend(backend, &features).unwrap();
        e.compile_module(&module).unwrap();
        e.set_fuel(Some(1000));
        unsafe { e.call_typed("count", &[RtValue::I64(100)]) }.unwrap();
        remaining.push(achainsaw_codegen::get_remaining_fuel());
        e.set_fuel(Some(150));
        let err = unsafe { e.call_typed("count", &[RtValue::I64(1000)]) }.unwrap_err();
        assert!(
            err.to_string().contains("ERR_OUT_OF_FUEL"),
            "{backend}: {err}"
        );
        e.set_fuel(None);
    }
    assert_eq!(remaining[0], remaining[1]);
}
