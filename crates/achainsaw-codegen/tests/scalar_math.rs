//! Scalar rounding, bit counts, rotations, copysign, fma and unsigned conversions on the
//! backend selected by `ACHAINSAW_BACKEND`, against Rust's own operations (which have the
//! semantics AIR defines): `round` rounds halfway cases away from zero, `roundeven` to even,
//! `clz`/`ctz` of 0 give the bit width, rotations take the amount modulo the width, and
//! `ftoui` saturates (negative values and NaN to 0).

use achainsaw_codegen::{JitEngine, RtValue};
use achainsaw_ir::parse_and_validate;

fn engine(src: &str) -> JitEngine {
    let module = parse_and_validate(src).unwrap_or_else(|d| panic!("{d:?}"));
    let mut e = JitEngine::new().unwrap();
    e.compile_module(&module).unwrap();
    e
}

fn call(e: &JitEngine, name: &str, args: &[RtValue]) -> RtValue {
    unsafe { e.call_typed(name, args) }.unwrap().unwrap()
}

/// Halfway cases, values next to them, integers past f32/f64 precision, and specials.
fn floats() -> Vec<f64> {
    let mut v = vec![
        0.0,
        -0.0,
        0.5,
        -0.5,
        1.5,
        -1.5,
        2.5,
        -2.5,
        3.5,
        0.49999997,
        -0.49999997,
        0.4999999999999999,
        1.0e-40,
        8388609.0,
        16777217.0,
        4503599627370497.0,
        1.0e300,
        123.456,
        -987.654,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
    ];
    v.extend((0..64).map(|i| (i as f64 - 32.0) * 0.25));
    v
}

fn same_f64(got: f64, want: f64) -> bool {
    got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan())
}

/// A rounding op's AIR name and Rust's f64 and f32 versions of it.
type RoundOp = (&'static str, fn(f64) -> f64, fn(f32) -> f32);

#[test]
fn float_rounding_matches_rust() {
    let ops: [RoundOp; 5] = [
        ("floor", f64::floor, f32::floor),
        ("ceil", f64::ceil, f32::ceil),
        ("round", f64::round, f32::round),
        ("roundeven", f64::round_ties_even, f32::round_ties_even),
        ("roundz", f64::trunc, f32::trunc),
    ];
    let mut src = String::new();
    for (op, ..) in ops {
        for ty in ["f32", "f64"] {
            src += &format!("fn {op}_{ty}(a:{ty})->{ty}\n  b0:\n    r = {op} a\n    ret r\n\n");
        }
    }
    let e = engine(&src);
    for (op, f64op, f32op) in ops {
        for x in floats() {
            let RtValue::F64(got) = call(&e, &format!("{op}_f64"), &[RtValue::F64(x)]) else {
                panic!()
            };
            assert!(same_f64(got, f64op(x)), "{op}({x}) = {got}");
            let x32 = x as f32;
            let RtValue::F32(got) = call(&e, &format!("{op}_f32"), &[RtValue::F32(x32)]) else {
                panic!()
            };
            assert!(
                same_f64(got as f64, f32op(x32) as f64),
                "{op}({x32}f32) = {got}"
            );
        }
    }
}

#[test]
fn copysign_and_fma_match_rust() {
    let src = "fn cs(a:f32, b:f32)->f32\n  b0:\n    r = copysign a, b\n    ret r\n\n\
               fn fm(a:f64, b:f64, c:f64)->f64\n  b0:\n    r = fma a, b, c\n    ret r\n\n\
               fn fm32(a:f32, b:f32, c:f32)->f32\n  b0:\n    r = fma a, b, c\n    ret r\n";
    let e = engine(src);
    for a in floats() {
        for b in [0.0, -0.0, 3.0, -7.25, f64::NAN, -f64::NAN] {
            let RtValue::F32(got) =
                call(&e, "cs", &[RtValue::F32(a as f32), RtValue::F32(b as f32)])
            else {
                panic!()
            };
            let want = (a as f32).copysign(b as f32);
            assert_eq!(got.to_bits(), want.to_bits(), "copysign({a}, {b})");
        }
    }
    // a * b + c with one rounding: 1 + 2^-30 squared minus 1 keeps the 2^-60 term.
    let x = 1.0 + 2f64.powi(-30);
    for (a, b, c) in [
        (x, x, -1.0),
        (0.1, 10.0, -1.0),
        (3.0, 7.0, 0.5),
        (1e308, 10.0, -1e308),
    ] {
        let RtValue::F64(got) = call(
            &e,
            "fm",
            &[RtValue::F64(a), RtValue::F64(b), RtValue::F64(c)],
        ) else {
            panic!()
        };
        assert!(same_f64(got, a.mul_add(b, c)), "fma({a}, {b}, {c}) = {got}");
        let (a, b, c) = (a as f32, b as f32, c as f32);
        let RtValue::F32(got) = call(
            &e,
            "fm32",
            &[RtValue::F32(a), RtValue::F32(b), RtValue::F32(c)],
        ) else {
            panic!()
        };
        assert!(
            same_f64(got as f64, a.mul_add(b, c) as f64),
            "fma32({a}, {b}, {c}) = {got}"
        );
    }
}

#[test]
fn bit_counts_and_rotations_match_rust() {
    let mut src = String::new();
    for ty in ["i8", "i16", "i32", "i64"] {
        for op in ["popcnt", "clz", "ctz"] {
            src += &format!("fn {op}_{ty}(a:{ty})->{ty}\n  b0:\n    r = {op} a\n    ret r\n\n");
        }
        for op in ["rotl", "rotr"] {
            src += &format!(
                "fn {op}_{ty}(a:{ty}, b:{ty})->{ty}\n  b0:\n    r = {op} a, b\n    ret r\n\n"
            );
        }
    }
    let e = engine(&src);
    let vals: [u64; 8] = [
        0,
        1,
        0x80,
        0xff,
        0x1234_5678_9abc_def0,
        u64::MAX,
        0x8000_0000,
        6,
    ];
    for (ty, bits) in [("i8", 8u32), ("i16", 16), ("i32", 32), ("i64", 64)] {
        let mask = if bits == 64 {
            u64::MAX
        } else {
            (1u64 << bits) - 1
        };
        let wrap = |v: u64| -> RtValue {
            match bits {
                8 => RtValue::I8(v as i8),
                16 => RtValue::I16(v as i16),
                32 => RtValue::I32(v as i32),
                _ => RtValue::I64(v as i64),
            }
        };
        let unwrap = |v: RtValue| -> u64 {
            (match v {
                RtValue::I8(x) => x as u64,
                RtValue::I16(x) => x as u64,
                RtValue::I32(x) => x as u64,
                RtValue::I64(x) => x as u64,
                other => panic!("{other:?}"),
            }) & mask
        };
        for v in vals {
            let u = v & mask;
            let lz = if u == 0 {
                bits
            } else {
                u.leading_zeros() - (64 - bits)
            };
            let tz = if u == 0 { bits } else { u.trailing_zeros() };
            assert_eq!(
                unwrap(call(&e, &format!("popcnt_{ty}"), &[wrap(u)])),
                u.count_ones() as u64
            );
            assert_eq!(
                unwrap(call(&e, &format!("clz_{ty}"), &[wrap(u)])),
                lz as u64,
                "clz {u:#x}"
            );
            assert_eq!(
                unwrap(call(&e, &format!("ctz_{ty}"), &[wrap(u)])),
                tz as u64,
                "ctz {u:#x}"
            );
            for amount in [
                0u64,
                1,
                3,
                bits as u64 - 1,
                bits as u64,
                bits as u64 + 5,
                u64::MAX,
            ] {
                // The amount has the operand's type; its value modulo the width counts.
                let amount = amount & mask;
                let s = (amount % bits as u64) as u32;
                let (rl, rr) = if s == 0 {
                    (u, u)
                } else {
                    (
                        ((u << s) | (u >> (bits - s))) & mask,
                        ((u >> s) | (u << (bits - s))) & mask,
                    )
                };
                let args = [wrap(u), wrap(amount)];
                assert_eq!(
                    unwrap(call(&e, &format!("rotl_{ty}"), &args)),
                    rl,
                    "rotl {u:#x} {amount}"
                );
                assert_eq!(
                    unwrap(call(&e, &format!("rotr_{ty}"), &args)),
                    rr,
                    "rotr {u:#x} {amount}"
                );
            }
        }
    }
}

#[test]
fn unsigned_conversions_match_rust() {
    let src = "fn u32f(a:i32)->f32\n  b0:\n    r = uitof a:f32\n    ret r\n\n\
               fn u64d(a:i64)->f64\n  b0:\n    r = uitof a:f64\n    ret r\n\n\
               fn u8f(a:i8)->f32\n  b0:\n    r = uitof a:f32\n    ret r\n\n\
               fn fu32(a:f32)->i32\n  b0:\n    r = ftoui a:i32\n    ret r\n\n\
               fn du64(a:f64)->i64\n  b0:\n    r = ftoui a:i64\n    ret r\n";
    let e = engine(src);
    for v in [
        0u64,
        1,
        255,
        0x7fff_ffff,
        0x8000_0000,
        0xffff_ffff,
        0x8000_0000_0000_0001,
        u64::MAX,
    ] {
        assert_eq!(
            call(&e, "u32f", &[RtValue::I32(v as u32 as i32)]),
            RtValue::F32(v as u32 as f32)
        );
        assert_eq!(
            call(&e, "u64d", &[RtValue::I64(v as i64)]),
            RtValue::F64(v as f64)
        );
        assert_eq!(
            call(&e, "u8f", &[RtValue::I8(v as u8 as i8)]),
            RtValue::F32(v as u8 as f32)
        );
    }
    for x in floats()
        .into_iter()
        .chain([4294967295.0, 4294967296.0, 1.8446744073709552e19])
    {
        assert_eq!(
            call(&e, "fu32", &[RtValue::F32(x as f32)]),
            RtValue::I32((x as f32) as u32 as i32),
            "ftoui {x}"
        );
        assert_eq!(
            call(&e, "du64", &[RtValue::F64(x)]),
            RtValue::I64(x as u64 as i64),
            "ftoui {x}"
        );
    }
}
