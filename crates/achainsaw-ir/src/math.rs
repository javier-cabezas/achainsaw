//! Transcendental functions on f32 (`exp`, `log`, `tanh`, `sin`, `cos` and their vector forms
//! `vexp`, `vlog`, ...) as fixed sequences of AIR operations, so their results are
//! bit-identical on every backend and target.
//!
//! Each function is written once, generic over [`Ops`]: [`eval`] runs it on Rust f32 values
//! (the reference, and the optimizer's constant folding), and [`expand_module`] writes it as
//! the AIR instructions that replace the function's instruction before code generation
//! (`prepare_for_codegen`), scalar or on vectors. Every step is an IEEE operation with one
//! rounding (FMAs included), so the two give the same bits. The same pass writes
//! `vfwidenlo` / `vfwidenhi` of integer lanes as the widening and conversion they stand for.

use std::collections::HashMap;

use crate::ast::{
    vexp, BinaryOp, CastOp, Constant, Function, Instruction, Module, UnaryOp, VBinOp, VCmpOp,
    VShiftOp, VUnaryOp,
};
use crate::diag::Span;
use crate::types::Type;

/// A function `math` defines.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MathFn {
    Exp,
    Log,
    Tanh,
    Sin,
    Cos,
}

impl MathFn {
    pub const ALL: [MathFn; 5] = [
        MathFn::Exp,
        MathFn::Log,
        MathFn::Tanh,
        MathFn::Sin,
        MathFn::Cos,
    ];

    pub fn of_unary(op: UnaryOp) -> Option<Self> {
        match op {
            UnaryOp::Exp => Some(MathFn::Exp),
            UnaryOp::Log => Some(MathFn::Log),
            UnaryOp::Tanh => Some(MathFn::Tanh),
            UnaryOp::Sin => Some(MathFn::Sin),
            UnaryOp::Cos => Some(MathFn::Cos),
            _ => None,
        }
    }

    pub fn of_vunary(op: VUnaryOp) -> Option<Self> {
        match op {
            VUnaryOp::Exp => Some(MathFn::Exp),
            VUnaryOp::Log => Some(MathFn::Log),
            VUnaryOp::Tanh => Some(MathFn::Tanh),
            VUnaryOp::Sin => Some(MathFn::Sin),
            VUnaryOp::Cos => Some(MathFn::Cos),
            _ => None,
        }
    }

    /// The function as a sequence of `o`'s operations on `x`.
    pub fn build<O: Ops>(self, o: &mut O, x: O::F) -> O::F {
        match self {
            MathFn::Exp => exp(o, x),
            MathFn::Log => log(o, x),
            MathFn::Tanh => tanh(o, x),
            MathFn::Sin => sin_cos(o, x, false),
            MathFn::Cos => sin_cos(o, x, true),
        }
    }
}

/// The function on one f32, exactly as the generated AIR computes it.
pub fn eval(f: MathFn, x: f32) -> f32 {
    f.build(&mut Eval, x)
}

/// The operations the functions are written in: f32 values `F`, their bits as i32 `I`, and
/// lane masks `M` (comparison results; `bool` in [`eval`]). Every float operation rounds
/// once, comparisons are false on NaN, and shifts take constant amounts below 32.
pub trait Ops {
    type F: Copy;
    type I: Copy;
    type M: Copy;
    fn f(&mut self, c: f32) -> Self::F;
    fn i(&mut self, c: i32) -> Self::I;
    fn add(&mut self, a: Self::F, b: Self::F) -> Self::F;
    fn sub(&mut self, a: Self::F, b: Self::F) -> Self::F;
    fn mul(&mut self, a: Self::F, b: Self::F) -> Self::F;
    fn div(&mut self, a: Self::F, b: Self::F) -> Self::F;
    /// `a * b + c`, one rounding.
    fn fma(&mut self, a: Self::F, b: Self::F, c: Self::F) -> Self::F;
    fn lt(&mut self, a: Self::F, b: Self::F) -> Self::M;
    fn gt(&mut self, a: Self::F, b: Self::F) -> Self::M;
    fn le(&mut self, a: Self::F, b: Self::F) -> Self::M;
    fn ge(&mut self, a: Self::F, b: Self::F) -> Self::M;
    /// `m ? a : b`.
    fn sel(&mut self, m: Self::M, a: Self::F, b: Self::F) -> Self::F;
    /// Lanes where bit `bit` of `x` is set.
    fn bit_set(&mut self, x: Self::I, bit: u32) -> Self::M;
    fn bits(&mut self, x: Self::F) -> Self::I;
    fn float(&mut self, x: Self::I) -> Self::F;
    /// The integer as the nearest f32.
    fn itof(&mut self, x: Self::I) -> Self::F;
    fn iadd(&mut self, a: Self::I, b: Self::I) -> Self::I;
    fn iand(&mut self, a: Self::I, b: Self::I) -> Self::I;
    fn ior(&mut self, a: Self::I, b: Self::I) -> Self::I;
    fn ixor(&mut self, a: Self::I, b: Self::I) -> Self::I;
    fn shl(&mut self, a: Self::I, n: u32) -> Self::I;
    /// Logical shift right.
    fn ushr(&mut self, a: Self::I, n: u32) -> Self::I;
}

/// Horner's rule by FMAs: `((c[0] x + c[1]) x + c[2]) ...`.
fn horner<O: Ops>(o: &mut O, x: O::F, c: &[f32]) -> O::F {
    let mut q = o.f(c[0]);
    for &k in &c[1..] {
        let k = o.f(k);
        q = o.fma(q, x, k);
    }
    q
}

/// e^x for x clamped to [-87, 88] (see `ast::vexp`): e^-87 (about 1.6e-38) for anything
/// smaller, e^88 for anything larger; NaN gives NaN.
fn exp<O: Ops>(o: &mut O, x: O::F) -> O::F {
    let lo = o.f(vexp::CLAMP_LO);
    let hi = o.f(vexp::CLAMP_HI);
    let below = o.lt(x, lo);
    let x = o.sel(below, lo, x);
    let above = o.gt(x, hi);
    let x = o.sel(above, hi, x);
    let log2e = o.f(vexp::LOG2E);
    let magic = o.f(vexp::MAGIC);
    let t = o.mul(x, log2e);
    let tm = o.add(t, magic);
    let n = o.sub(tm, magic);
    let ln2_hi = o.f(vexp::NEG_LN2_HI);
    let r = o.fma(n, ln2_hi, x);
    let ln2_lo = o.f(vexp::LN2_LO);
    let r = o.fma(n, ln2_lo, r);
    let q = horner(o, r, &vexp::POLY);
    let r2 = o.mul(r, r);
    let one = o.f(1.0);
    let rp = o.add(r, one);
    let p = o.fma(q, r2, rp);
    let bits = o.bits(tm);
    let bias = o.i(vexp::EXP_BIAS);
    let e = o.iadd(bits, bias);
    let e = o.shl(e, 23);
    let scale = o.float(e);
    o.mul(p, scale)
}

/// Constants of `log`, `tanh`, `sin` and `cos`: the Cephes f32 polynomials (logf, tanhf,
/// sinf, cosf), as Cephes prints them.
#[allow(clippy::excessive_precision)]
pub mod consts {
    /// 2^-126, the smallest normal f32: smaller inputs of `log` are scaled by 2^23 first.
    pub const MIN_NORMAL: f32 = f32::MIN_POSITIVE;
    pub const SQRT_HALF: f32 = std::f32::consts::FRAC_1_SQRT_2;
    pub const LOG_POLY: [f32; 9] = [
        7.0376836292e-2,
        -1.1514610310e-1,
        1.1676998740e-1,
        -1.2420140846e-1,
        1.4249322787e-1,
        -1.6668057665e-1,
        2.0000714765e-1,
        -2.4999993993e-1,
        3.3333331174e-1,
    ];
    /// ln 2 in two parts, `LN2_HI` exact in a few bits.
    pub const LN2_HI: f32 = 0.693359375;
    pub const LN2_LO: f32 = -2.12194440e-4;
    /// `tanh` uses its polynomial below this magnitude and `1 - 2 / (e^2|x| + 1)` above.
    pub const TANH_SMALL: f32 = 0.625;
    pub const TANH_POLY: [f32; 5] = [
        -5.70498872745e-3,
        2.06390887954e-2,
        -5.37397155531e-2,
        1.33314422036e-1,
        -3.33332819422e-1,
    ];
    pub const TWO_OVER_PI: f32 = f32::from_bits(0x3f22_f983);
    /// `sin` and `cos` take |x| up to 2^22 (beyond it, |x| 2/pi + 1.5 * 2^23 no longer
    /// rounds to the nearest integer) and give NaN above.
    pub const TRIG_MAX: f32 = 4_194_304.0;
    /// pi / 2 in three parts (each the f32 nearest the rest), for `x - q pi/2` by FMAs.
    pub const PIO2: [f32; 3] = [
        f32::from_bits(0x3fc9_0fdb),
        f32::from_bits(0xb33b_bd2e),
        f32::from_bits(0xa6f7_2ced),
    ];
    pub const SIN_POLY: [f32; 3] = [-1.9515295891e-4, 8.3321608736e-3, -1.6666654611e-1];
    pub const COS_POLY: [f32; 3] = [
        2.443315711809948e-5,
        -1.388731625493765e-3,
        4.166664568298827e-2,
    ];
}

/// Natural logarithm (Cephes logf): x = 2^e m with m in [sqrt(1/2), sqrt(2)), then
/// log(m) = f - f^2/2 + f^3 P(f) for f = m - 1, and e ln 2 in two parts. log(+-0) = -inf,
/// log(+inf) = +inf, log of a negative number or NaN is NaN.
fn log<O: Ops>(o: &mut O, x: O::F) -> O::F {
    use consts::*;
    let min_normal = o.f(MIN_NORMAL);
    let small = o.lt(x, min_normal);
    let scale = o.f(8_388_608.0);
    let scaled = o.mul(x, scale);
    let xs = o.sel(small, scaled, x);
    let adj = o.f(-23.0);
    let zero = o.f(0.0);
    let e_adj = o.sel(small, adj, zero);
    let b = o.bits(xs);
    let biased = o.ushr(b, 23);
    let unbias = o.i(-126);
    let ei = o.iadd(biased, unbias);
    let e = o.itof(ei);
    let e = o.add(e, e_adj);
    // The mantissa as a float in [0.5, 1).
    let mant_mask = o.i(0x007f_ffff);
    let mant = o.iand(b, mant_mask);
    let half = o.i(0x3f00_0000);
    let mb = o.ior(mant, half);
    let m = o.float(mb);
    let sqrt_half = o.f(SQRT_HALF);
    let low = o.lt(m, sqrt_half);
    let one = o.f(1.0);
    let e1 = o.sub(e, one);
    let e = o.sel(low, e1, e);
    let m2 = o.add(m, m);
    let m = o.sel(low, m2, m);
    let f = o.sub(m, one);
    let z = o.mul(f, f);
    let p = horner(o, f, &LOG_POLY);
    let y = o.mul(p, f);
    let y = o.mul(y, z);
    let ln2_lo = o.f(LN2_LO);
    let y = o.fma(e, ln2_lo, y);
    let neg_half = o.f(-0.5);
    let y = o.fma(z, neg_half, y);
    let r = o.add(f, y);
    let ln2_hi = o.f(LN2_HI);
    let r = o.fma(e, ln2_hi, r);
    // +inf, then zeros (and negatives) to -inf, then negatives and NaN to NaN.
    let max = o.f(f32::MAX);
    let inf = o.gt(x, max);
    let r = o.sel(inf, x, r);
    let nonpos = o.le(x, zero);
    let neg_inf = o.f(f32::NEG_INFINITY);
    let r = o.sel(nonpos, neg_inf, r);
    let ok = o.ge(x, zero);
    let nan = o.f(f32::NAN);
    o.sel(ok, r, nan)
}

/// |x| and x's sign bit.
fn abs_sign<O: Ops>(o: &mut O, x: O::F) -> (O::F, O::I) {
    let b = o.bits(x);
    let mag = o.i(0x7fff_ffff);
    let ab = o.iand(b, mag);
    let sign = o.ixor(b, ab);
    (o.float(ab), sign)
}

/// Hyperbolic tangent (Cephes tanhf) of |x|, with x's sign: a + a^3 P(a^2) for a = |x| below
/// 0.625, else 1 - 2 / (e^2a + 1) with `exp` (1 once e^2a clamps at e^88). NaN gives NaN.
fn tanh<O: Ops>(o: &mut O, x: O::F) -> O::F {
    use consts::*;
    let (a, sign) = abs_sign(o, x);
    let z = o.mul(a, a);
    let p = horner(o, z, &TANH_POLY);
    let pz = o.mul(p, z);
    let small = o.fma(pz, a, a);
    let a2 = o.add(a, a);
    let e = exp(o, a2);
    let one = o.f(1.0);
    let d = o.add(e, one);
    let two = o.f(2.0);
    let q = o.div(two, d);
    let large = o.sub(one, q);
    let limit = o.f(TANH_SMALL);
    let is_small = o.lt(a, limit);
    let t = o.sel(is_small, small, large);
    let tb = o.bits(t);
    let tb = o.ior(tb, sign);
    o.float(tb)
}

/// Sine or cosine (Cephes sinf/cosf polynomials) of r = |x| - q pi/2, q the integer nearest
/// |x| 2/pi (rounded by adding and subtracting 1.5 * 2^23), r computed by three FMAs with
/// pi/2 in three parts. Quadrant q (q + 1 for cosine) picks the polynomial and the sign;
/// sine takes x's sign. NaN for |x| above `TRIG_MAX` (where q is no longer exact),
/// infinities and NaN.
fn sin_cos<O: Ops>(o: &mut O, x: O::F, cos: bool) -> O::F {
    use consts::*;
    let (a, sign) = abs_sign(o, x);
    let two_over_pi = o.f(TWO_OVER_PI);
    let t = o.mul(a, two_over_pi);
    let magic = o.f(vexp::MAGIC);
    let tm = o.add(t, magic);
    let q = o.sub(tm, magic);
    let mut r = a;
    for c in PIO2 {
        let c = o.f(-c);
        r = o.fma(q, c, r);
    }
    let z = o.mul(r, r);
    let ps = horner(o, z, &SIN_POLY);
    let psz = o.mul(ps, z);
    let s = o.fma(psz, r, r);
    let pc = horner(o, z, &COS_POLY);
    let pcz = o.mul(pc, z);
    let pcz = o.mul(pcz, z);
    let neg_half = o.f(-0.5);
    let y = o.fma(z, neg_half, pcz);
    let one = o.f(1.0);
    let c = o.add(y, one);
    // The low bits of tm's bits are q's (MAGIC's are zero).
    let mut k = o.bits(tm);
    if cos {
        let one = o.i(1);
        k = o.iadd(k, one);
    }
    let odd = o.bit_set(k, 0);
    let v = o.sel(odd, c, s);
    let two = o.i(2);
    let neg = o.iand(k, two);
    let flip = o.shl(neg, 30);
    let flip = if cos { flip } else { o.ixor(flip, sign) };
    let vb = o.bits(v);
    let vb = o.ixor(vb, flip);
    let v = o.float(vb);
    let max = o.f(TRIG_MAX);
    let ok = o.le(a, max);
    let nan = o.f(f32::NAN);
    o.sel(ok, v, nan)
}

/// [`Ops`] on Rust values.
struct Eval;

impl Ops for Eval {
    type F = f32;
    type I = i32;
    type M = bool;
    fn f(&mut self, c: f32) -> f32 {
        c
    }
    fn i(&mut self, c: i32) -> i32 {
        c
    }
    fn add(&mut self, a: f32, b: f32) -> f32 {
        a + b
    }
    fn sub(&mut self, a: f32, b: f32) -> f32 {
        a - b
    }
    fn mul(&mut self, a: f32, b: f32) -> f32 {
        a * b
    }
    fn div(&mut self, a: f32, b: f32) -> f32 {
        a / b
    }
    fn fma(&mut self, a: f32, b: f32, c: f32) -> f32 {
        a.mul_add(b, c)
    }
    fn lt(&mut self, a: f32, b: f32) -> bool {
        a < b
    }
    fn gt(&mut self, a: f32, b: f32) -> bool {
        a > b
    }
    fn le(&mut self, a: f32, b: f32) -> bool {
        a <= b
    }
    fn ge(&mut self, a: f32, b: f32) -> bool {
        a >= b
    }
    fn sel(&mut self, m: bool, a: f32, b: f32) -> f32 {
        if m {
            a
        } else {
            b
        }
    }
    fn bit_set(&mut self, x: i32, bit: u32) -> bool {
        (x >> bit) & 1 != 0
    }
    fn bits(&mut self, x: f32) -> i32 {
        x.to_bits() as i32
    }
    fn float(&mut self, x: i32) -> f32 {
        f32::from_bits(x as u32)
    }
    fn itof(&mut self, x: i32) -> f32 {
        x as f32
    }
    fn iadd(&mut self, a: i32, b: i32) -> i32 {
        a.wrapping_add(b)
    }
    fn iand(&mut self, a: i32, b: i32) -> i32 {
        a & b
    }
    fn ior(&mut self, a: i32, b: i32) -> i32 {
        a | b
    }
    fn ixor(&mut self, a: i32, b: i32) -> i32 {
        a ^ b
    }
    fn shl(&mut self, a: i32, n: u32) -> i32 {
        a << n
    }
    fn ushr(&mut self, a: i32, n: u32) -> i32 {
        ((a as u32) >> n) as i32
    }
}

/// [`Ops`] that write AIR instructions: scalar f32 and i32 registers when `vec` is `None`,
/// else registers of the vector type `vec` (whose bits are reinterpreted freely, so `bits`
/// and `float` are no-ops) with f32 or i32 lanes. Values are indices into `regs`.
struct Emit {
    out: Vec<Instruction>,
    regs: Vec<String>,
    prefix: String,
    next: usize,
    vec: Option<Type>,
    span: Span,
    floats: HashMap<u32, usize>,
    ints: HashMap<i32, usize>,
}

impl Emit {
    fn new(prefix: String, vec: Option<Type>, span: Span) -> Self {
        Emit {
            out: Vec::new(),
            regs: Vec::new(),
            prefix,
            next: 0,
            vec,
            span,
            floats: HashMap::new(),
            ints: HashMap::new(),
        }
    }

    fn reg(&mut self, name: String) -> usize {
        self.regs.push(name);
        self.regs.len() - 1
    }

    fn name(&self, v: usize) -> String {
        self.regs[v].clone()
    }

    /// A new register, defined by the instruction `make` builds from its name.
    fn push(&mut self, make: impl FnOnce(String, Span) -> Instruction) -> usize {
        self.next += 1;
        let dst = format!("{}_{}", self.prefix, self.next);
        self.out.push(make(dst.clone(), self.span));
        self.reg(dst)
    }

    /// A constant, splat to the vector type.
    fn constant(&mut self, val: Constant, ty: Type) -> usize {
        let c = self.push(|dst, span| Instruction::AssignConst { dst, val, ty, span });
        match self.vec {
            None => c,
            Some(vty) => {
                let src = self.name(c);
                self.push(|dst, span| Instruction::Splat {
                    dst,
                    src,
                    ty: vty,
                    span,
                })
            }
        }
    }

    fn bin(&mut self, op: BinaryOp, vop: VBinOp, lane: Type, a: usize, b: usize) -> usize {
        let (lhs, rhs) = (self.name(a), self.name(b));
        match self.vec {
            None => self.push(|dst, span| Instruction::Binary {
                op,
                dst,
                lhs,
                rhs,
                span,
            }),
            Some(_) => self.push(|dst, span| Instruction::VBinary {
                op: vop,
                dst,
                lhs,
                rhs,
                lane,
                span,
            }),
        }
    }

    fn cmp(&mut self, op: BinaryOp, vop: VCmpOp, a: usize, b: usize) -> usize {
        let (lhs, rhs) = (self.name(a), self.name(b));
        match self.vec {
            None => self.push(|dst, span| Instruction::Binary {
                op,
                dst,
                lhs,
                rhs,
                span,
            }),
            Some(_) => self.push(|dst, span| Instruction::VCmp {
                op: vop,
                dst,
                lhs,
                rhs,
                lane: Type::F32,
                span,
            }),
        }
    }

    fn shift(&mut self, op: BinaryOp, vop: VShiftOp, a: usize, n: u32) -> usize {
        let amount = self.push(|dst, span| Instruction::AssignConst {
            dst,
            val: Constant::Int(n as i64),
            ty: Type::I32,
            span,
        });
        match self.vec {
            None => self.bin(op, VBinOp::Add, Type::I32, a, amount),
            Some(_) => {
                let (src, amount) = (self.name(a), self.name(amount));
                self.push(|dst, span| Instruction::VShift {
                    op: vop,
                    dst,
                    src,
                    amount,
                    lane: Type::I32,
                    span,
                })
            }
        }
    }

    fn bitcast(&mut self, x: usize, ty: Type) -> usize {
        match self.vec {
            None => {
                let src = self.name(x);
                self.push(|dst, span| Instruction::Cast {
                    op: CastOp::Bitcast,
                    dst,
                    src,
                    ty,
                    span,
                })
            }
            Some(_) => x,
        }
    }
}

impl Ops for Emit {
    type F = usize;
    type I = usize;
    type M = usize;
    fn f(&mut self, c: f32) -> usize {
        if let Some(&v) = self.floats.get(&c.to_bits()) {
            return v;
        }
        let v = self.constant(Constant::Float(c as f64), Type::F32);
        self.floats.insert(c.to_bits(), v);
        v
    }
    fn i(&mut self, c: i32) -> usize {
        if let Some(&v) = self.ints.get(&c) {
            return v;
        }
        let v = self.constant(Constant::Int(c as i64), Type::I32);
        self.ints.insert(c, v);
        v
    }
    fn add(&mut self, a: usize, b: usize) -> usize {
        self.bin(BinaryOp::Add, VBinOp::Add, Type::F32, a, b)
    }
    fn sub(&mut self, a: usize, b: usize) -> usize {
        self.bin(BinaryOp::Sub, VBinOp::Sub, Type::F32, a, b)
    }
    fn mul(&mut self, a: usize, b: usize) -> usize {
        self.bin(BinaryOp::Mul, VBinOp::Mul, Type::F32, a, b)
    }
    fn div(&mut self, a: usize, b: usize) -> usize {
        self.bin(BinaryOp::Div, VBinOp::Div, Type::F32, a, b)
    }
    fn fma(&mut self, a: usize, b: usize, c: usize) -> usize {
        let (a, b, c) = (self.name(a), self.name(b), self.name(c));
        match self.vec {
            None => self.push(|dst, span| Instruction::Fma { dst, a, b, c, span }),
            Some(_) => self.push(|dst, span| Instruction::VFma {
                dst,
                a,
                b,
                c,
                lane: Type::F32,
                span,
            }),
        }
    }
    fn lt(&mut self, a: usize, b: usize) -> usize {
        self.cmp(BinaryOp::Lt, VCmpOp::Lt, a, b)
    }
    fn gt(&mut self, a: usize, b: usize) -> usize {
        self.cmp(BinaryOp::Gt, VCmpOp::Gt, a, b)
    }
    fn le(&mut self, a: usize, b: usize) -> usize {
        self.cmp(BinaryOp::Le, VCmpOp::Le, a, b)
    }
    fn ge(&mut self, a: usize, b: usize) -> usize {
        self.cmp(BinaryOp::Ge, VCmpOp::Ge, a, b)
    }
    fn sel(&mut self, m: usize, a: usize, b: usize) -> usize {
        let (cond, then_val, else_val) = (self.name(m), self.name(a), self.name(b));
        match self.vec {
            None => self.push(|dst, span| Instruction::Select {
                dst,
                cond,
                then_val,
                else_val,
                span,
            }),
            Some(_) => self.push(|dst, span| Instruction::VSelect {
                dst,
                mask: cond,
                then_val,
                else_val,
                span,
            }),
        }
    }
    fn bit_set(&mut self, x: usize, bit: u32) -> usize {
        match self.vec {
            // A nonzero i32 selects.
            None => {
                let m = self.i(1 << bit);
                self.iand(x, m)
            }
            // All ones where the bit is set: the bit to the top, then an arithmetic shift.
            Some(_) => {
                let up = self.shift(BinaryOp::Shl, VShiftOp::Shl, x, 31 - bit);
                self.shift(BinaryOp::Shr, VShiftOp::Shr, up, 31)
            }
        }
    }
    fn bits(&mut self, x: usize) -> usize {
        self.bitcast(x, Type::I32)
    }
    fn float(&mut self, x: usize) -> usize {
        self.bitcast(x, Type::F32)
    }
    fn itof(&mut self, x: usize) -> usize {
        let src = self.name(x);
        match self.vec {
            None => self.push(|dst, span| Instruction::Cast {
                op: CastOp::Itof,
                dst,
                src,
                ty: Type::F32,
                span,
            }),
            Some(_) => self.push(|dst, span| Instruction::VUnary {
                op: VUnaryOp::Itof,
                dst,
                src,
                lane: Type::F32,
                span,
            }),
        }
    }
    fn iadd(&mut self, a: usize, b: usize) -> usize {
        self.bin(BinaryOp::Add, VBinOp::Add, Type::I32, a, b)
    }
    fn iand(&mut self, a: usize, b: usize) -> usize {
        self.bin(BinaryOp::And, VBinOp::And, Type::I32, a, b)
    }
    fn ior(&mut self, a: usize, b: usize) -> usize {
        self.bin(BinaryOp::Or, VBinOp::Or, Type::I32, a, b)
    }
    fn ixor(&mut self, a: usize, b: usize) -> usize {
        self.bin(BinaryOp::Xor, VBinOp::Xor, Type::I32, a, b)
    }
    fn shl(&mut self, a: usize, n: u32) -> usize {
        self.shift(BinaryOp::Shl, VShiftOp::Shl, a, n)
    }
    fn ushr(&mut self, a: usize, n: u32) -> usize {
        self.shift(BinaryOp::Ushr, VShiftOp::Ushr, a, n)
    }
}

/// What `expand_module` writes an instruction as.
enum Expansion<'a> {
    /// A math function of `src` (`vec`: whether it is the vector form).
    Math(MathFn, &'a str, bool),
    /// `vfwidenlo`/`vfwidenhi` of integer lanes: `vwidenlo`/`vwidenhi`, then `vitof`.
    IntWiden(VUnaryOp, Type),
}

fn expansion(inst: &Instruction) -> Option<Expansion<'_>> {
    match inst {
        Instruction::Unary { op, src, .. } => {
            Some(Expansion::Math(MathFn::of_unary(*op)?, src, false))
        }
        Instruction::VUnary { op, src, lane, .. } => match op {
            VUnaryOp::FWidenLo | VUnaryOp::FWidenHi if lane.is_int() => {
                Some(Expansion::IntWiden(*op, *lane))
            }
            _ => Some(Expansion::Math(MathFn::of_vunary(*op)?, src, true)),
        },
        _ => None,
    }
}

/// Whether the module has an instruction [`expand_module`] replaces.
pub fn has_math(module: &Module) -> bool {
    module.functions.iter().any(has_expansions)
}

fn has_expansions(f: &Function) -> bool {
    f.blocks
        .iter()
        .any(|b| b.instructions.iter().any(|i| expansion(i).is_some()))
}

/// Replaces every math instruction (`exp` ... `cos`, `vexp` ... `vcos`) with the AIR
/// instructions that compute it, and `vfwidenlo`/`vfwidenhi` of integer lanes with a
/// `vwidenlo`/`vwidenhi` and a `vitof`. Code generation takes modules after this pass, so
/// the backends never see them.
pub fn expand_module(module: &mut Module) {
    let sigs = crate::opt::signatures(module);
    for f in &mut module.functions {
        if has_expansions(f) {
            expand_function(f, &sigs);
        }
    }
}

fn expand_function(f: &mut Function, sigs: &crate::opt::Signatures) {
    let types = crate::opt::infer_reg_types(f, sigs);
    let names = crate::inline::names_in(f);
    let mut serial = 0;
    for bi in 0..f.blocks.len() {
        let old = std::mem::take(&mut f.blocks[bi].instructions);
        let mut new = Vec::with_capacity(old.len());
        for inst in old {
            let Some(how) = expansion(&inst) else {
                new.push(inst);
                continue;
            };
            // A prefix no name in the function starts with.
            let prefix = loop {
                serial += 1;
                let p = format!("__m{serial}");
                if !names.iter().any(|n| n.starts_with(&p)) {
                    break p;
                }
            };
            let dst = inst
                .dst()
                .expect("a math instruction has a destination")
                .to_string();
            let span = inst.span();
            match how {
                Expansion::Math(func, src, vector) => {
                    let vec = vector.then(|| types[src]);
                    let mut e = Emit::new(prefix, vec, span);
                    let x = e.reg(src.to_string());
                    func.build(&mut e, x);
                    // The last instruction defines the result: it takes the destination.
                    let mut out = e.out;
                    let last = out.last_mut().and_then(|i| i.dsts_mut().into_iter().next());
                    *last.expect("a math function ends with an instruction") = dst;
                    new.extend(out);
                }
                Expansion::IntWiden(op, lane) => {
                    let Instruction::VUnary { src, .. } = &inst else {
                        unreachable!()
                    };
                    let (wide, float) = if lane == Type::I16 {
                        (Type::I32, Type::F32)
                    } else {
                        (Type::I64, Type::F64)
                    };
                    let widen = if op == VUnaryOp::FWidenLo {
                        VUnaryOp::WidenLo
                    } else {
                        VUnaryOp::WidenHi
                    };
                    let w = format!("{prefix}_w");
                    new.push(Instruction::VUnary {
                        op: widen,
                        dst: w.clone(),
                        src: src.clone(),
                        lane: wide,
                        span,
                    });
                    new.push(Instruction::VUnary {
                        op: VUnaryOp::Itof,
                        dst,
                        src: w,
                        lane: float,
                        span,
                    });
                }
            }
        }
        f.blocks[bi].instructions = new;
    }
}
