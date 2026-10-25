//! Accuracy of the math functions (`achainsaw_ir::math`) against f64, and their special
//! values. The backends compile the same definition (`achainsaw-codegen/tests/math.rs`
//! checks them bit for bit). The bounds are the measured worst cases over every f32 input,
//! rounded up; this test samples every 61st bit pattern (607th in debug builds).

use achainsaw_ir::math::{eval, MathFn};

/// Error of `got` in units of the f32 spacing at the true value (0 when both are NaN).
fn ulps(got: f32, want: f64) -> f64 {
    if want.is_nan() || got.is_nan() {
        return if got.is_nan() && want.is_nan() {
            0.0
        } else {
            f64::INFINITY
        };
    }
    let w = want as f32;
    if got == w {
        return 0.0;
    }
    if w.is_infinite() || got.is_infinite() {
        return f64::INFINITY;
    }
    let a = w.abs();
    let spacing = if a < f32::MIN_POSITIVE {
        f32::from_bits(1) as f64
    } else {
        (f32::from_bits(a.to_bits() + 1) - a) as f64
    };
    (got as f64 - want).abs() / spacing
}

#[test]
fn math_functions_are_within_their_error_bounds() {
    type Case = (MathFn, fn(f64) -> f64, f32, f32, f64);
    let cases: [Case; 5] = [
        (MathFn::Exp, f64::exp, -87.0, 88.0, 1.3),
        (MathFn::Log, f64::ln, 0.0, f32::INFINITY, 0.9),
        (
            MathFn::Tanh,
            f64::tanh,
            f32::NEG_INFINITY,
            f32::INFINITY,
            1.4,
        ),
        (MathFn::Sin, f64::sin, -1_048_576.0, 1_048_576.0, 2.1),
        (MathFn::Cos, f64::cos, -1_048_576.0, 1_048_576.0, 1.9),
    ];
    let stride = if cfg!(debug_assertions) { 607 } else { 61 };
    for (f, reference, lo, hi, bound) in cases {
        let mut b: u64 = 0;
        while b <= u32::MAX as u64 {
            let x = f32::from_bits(b as u32);
            if x >= lo && x <= hi {
                let e = ulps(eval(f, x), reference(x as f64));
                assert!(e <= bound, "{f:?}({x:e}) is {e:.3} ulp off (bound {bound})");
            }
            b += stride;
        }
    }
}

#[test]
fn math_functions_have_ieee_special_values() {
    let nan = f32::NAN;
    let inf = f32::INFINITY;
    // (input, exp, log, tanh, sin, cos); exp clamps its input to [-87, 88].
    let rows: [(f32, f32, f32, f32, f32, f32); 7] = [
        (0.0, 1.0, -inf, 0.0, 0.0, 1.0),
        (-0.0, 1.0, -inf, -0.0, -0.0, 1.0),
        (inf, eval(MathFn::Exp, 88.0), inf, 1.0, nan, nan),
        (-inf, eval(MathFn::Exp, -87.0), nan, -1.0, nan, nan),
        (nan, nan, nan, nan, nan, nan),
        (
            -1.0,
            eval(MathFn::Exp, -1.0),
            nan,
            eval(MathFn::Tanh, -1.0),
            eval(MathFn::Sin, -1.0),
            eval(MathFn::Cos, 1.0),
        ),
        (
            4_194_305.0,
            eval(MathFn::Exp, 88.0),
            eval(MathFn::Log, 4_194_305.0),
            1.0,
            nan,
            nan,
        ),
    ];
    for (x, exp, log, tanh, sin, cos) in rows {
        for (f, want) in [
            (MathFn::Exp, exp),
            (MathFn::Log, log),
            (MathFn::Tanh, tanh),
            (MathFn::Sin, sin),
            (MathFn::Cos, cos),
        ] {
            let got = eval(f, x);
            assert!(
                got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan()),
                "{f:?}({x:e}) = {got:e}, want {want:e}"
            );
        }
    }
    // Subnormal inputs of log, and tanh and sin of tiny values.
    assert_eq!(
        eval(MathFn::Log, f32::from_bits(1)),
        (f32::from_bits(1) as f64).ln() as f32
    );
    assert_eq!(eval(MathFn::Tanh, 1e-30), 1e-30);
    assert_eq!(eval(MathFn::Sin, -1e-30), -1e-30);
}

/// The constants keep their f32 values (the Cephes decimals, rounded), and pi/2 in three
/// parts sums to pi/2.
#[test]
fn math_constants_are_pinned() {
    use achainsaw_ir::math::consts;
    let pio2: f64 = consts::PIO2.iter().map(|&c| c as f64).sum();
    assert!((pio2 - std::f64::consts::FRAC_PI_2).abs() < 1e-16);
    assert_eq!(consts::TWO_OVER_PI, std::f32::consts::FRAC_2_PI);
    let bits: Vec<u32> = consts::LOG_POLY
        .iter()
        .chain(&consts::TANH_POLY)
        .chain(&consts::SIN_POLY)
        .chain(&consts::COS_POLY)
        .map(|c| c.to_bits())
        .collect();
    assert_eq!(
        bits,
        [
            0x3d90_21bb,
            0xbdeb_d1b8,
            0x3def_251a,
            0xbdfe_5d4f,
            0x3e11_e9bf,
            0xbe2a_ae50,
            0x3e4c_ceac,
            0xbe7f_fffc,
            0x3eaa_aaaa,
            0xbbba_f0ea,
            0x3ca9_134e,
            0xbd5c_1e2d,
            0x3e08_8393,
            0xbeaa_aa99,
            0xb94c_a1f9,
            0x3c08_839e,
            0xbe2a_aaa3,
            0x37cc_f5ce,
            0xbab6_061a,
            0x3d2a_aaa5,
        ]
    );
}
