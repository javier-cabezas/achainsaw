use crate::diag::Span;
use crate::types::Type;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Constant {
    Int(i64),
    Float(f64),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    And,
    Or,
    Xor,
    Shl,
    Shr,
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
    // Phase 4 Math & Unsigned Ops
    Min,
    Max,
    Umin,
    Umax,
    Udiv,
    Urem,
    Ushr,
    Ult,
    Ugt,
    Ule,
    Uge,
    /// `copysign a, b`: a's magnitude with b's sign bit (floats).
    Copysign,
    /// Rotations by `b` modulo the bit width (integers).
    Rotl,
    Rotr,
}

impl std::str::FromStr for BinaryOp {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::from_str_opt(s).ok_or(())
    }
}

impl BinaryOp {
    pub fn from_str_opt(s: &str) -> Option<Self> {
        match s {
            "add" => Some(BinaryOp::Add),
            "sub" => Some(BinaryOp::Sub),
            "mul" => Some(BinaryOp::Mul),
            "div" => Some(BinaryOp::Div),
            "rem" => Some(BinaryOp::Rem),
            "and" => Some(BinaryOp::And),
            "or" => Some(BinaryOp::Or),
            "xor" => Some(BinaryOp::Xor),
            "shl" => Some(BinaryOp::Shl),
            "shr" => Some(BinaryOp::Shr),
            "eq" => Some(BinaryOp::Eq),
            "ne" => Some(BinaryOp::Ne),
            "lt" => Some(BinaryOp::Lt),
            "gt" => Some(BinaryOp::Gt),
            "le" => Some(BinaryOp::Le),
            "ge" => Some(BinaryOp::Ge),
            "min" => Some(BinaryOp::Min),
            "max" => Some(BinaryOp::Max),
            "umin" => Some(BinaryOp::Umin),
            "umax" => Some(BinaryOp::Umax),
            "udiv" => Some(BinaryOp::Udiv),
            "urem" => Some(BinaryOp::Urem),
            "ushr" => Some(BinaryOp::Ushr),
            "ult" => Some(BinaryOp::Ult),
            "ugt" => Some(BinaryOp::Ugt),
            "ule" => Some(BinaryOp::Ule),
            "uge" => Some(BinaryOp::Uge),
            "copysign" => Some(BinaryOp::Copysign),
            "rotl" => Some(BinaryOp::Rotl),
            "rotr" => Some(BinaryOp::Rotr),
            _ => None,
        }
    }

    pub fn is_comparison(&self) -> bool {
        matches!(
            self,
            BinaryOp::Eq
                | BinaryOp::Ne
                | BinaryOp::Lt
                | BinaryOp::Gt
                | BinaryOp::Le
                | BinaryOp::Ge
                | BinaryOp::Ult
                | BinaryOp::Ugt
                | BinaryOp::Ule
                | BinaryOp::Uge
        )
    }

    pub fn is_unsigned(&self) -> bool {
        matches!(
            self,
            BinaryOp::Umin
                | BinaryOp::Umax
                | BinaryOp::Udiv
                | BinaryOp::Urem
                | BinaryOp::Ushr
                | BinaryOp::Ult
                | BinaryOp::Ugt
                | BinaryOp::Ule
                | BinaryOp::Uge
        )
    }
}

/// Lane-wise vector binary op: `r = vadd a, b:f32`. `vumin`/`vumax` compare integer lanes
/// as unsigned.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VBinOp {
    Add,
    Sub,
    Mul,
    Div,
    Min,
    Max,
    And,
    Or,
    Xor,
    Copysign,
    Umin,
    Umax,
}

impl VBinOp {
    /// Every op, in AIRB encoding order (append new ops at the end).
    pub const ALL: [VBinOp; 12] = [
        VBinOp::Add,
        VBinOp::Sub,
        VBinOp::Mul,
        VBinOp::Div,
        VBinOp::Min,
        VBinOp::Max,
        VBinOp::And,
        VBinOp::Or,
        VBinOp::Xor,
        VBinOp::Copysign,
        VBinOp::Umin,
        VBinOp::Umax,
    ];

    pub fn from_str_opt(s: &str) -> Option<Self> {
        Self::ALL.iter().copied().find(|op| op.as_str() == s)
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            VBinOp::Add => "vadd",
            VBinOp::Sub => "vsub",
            VBinOp::Mul => "vmul",
            VBinOp::Div => "vdiv",
            VBinOp::Min => "vmin",
            VBinOp::Max => "vmax",
            VBinOp::And => "vand",
            VBinOp::Or => "vor",
            VBinOp::Xor => "vxor",
            VBinOp::Copysign => "vcopysign",
            VBinOp::Umin => "vumin",
            VBinOp::Umax => "vumax",
        }
    }

    pub fn is_bitwise(&self) -> bool {
        matches!(self, VBinOp::And | VBinOp::Or | VBinOp::Xor)
    }
}

/// Set in the size a sandboxed atomic access reports to the runtime's fault hook when its
/// address is misaligned (rather than out of bounds), on both backends.
pub const MISALIGNED_FAULT: u64 = 1 << 63;

/// Atomic read-modify-write op: `old = atomic add p, v` stores `old op v` and returns `old`
/// (`xchg` stores `v`; `min`/`max` compare as signed, `umin`/`umax` as unsigned).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AtomicOp {
    Add,
    Sub,
    And,
    Or,
    Xor,
    Xchg,
    Min,
    Max,
    Umin,
    Umax,
}

impl AtomicOp {
    /// Every op, in AIRB encoding order (append new ops at the end).
    pub const ALL: [AtomicOp; 10] = [
        AtomicOp::Add,
        AtomicOp::Sub,
        AtomicOp::And,
        AtomicOp::Or,
        AtomicOp::Xor,
        AtomicOp::Xchg,
        AtomicOp::Min,
        AtomicOp::Max,
        AtomicOp::Umin,
        AtomicOp::Umax,
    ];

    pub fn from_str_opt(s: &str) -> Option<Self> {
        Self::ALL.iter().copied().find(|op| op.as_str() == s)
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            AtomicOp::Add => "add",
            AtomicOp::Sub => "sub",
            AtomicOp::And => "and",
            AtomicOp::Or => "or",
            AtomicOp::Xor => "xor",
            AtomicOp::Xchg => "xchg",
            AtomicOp::Min => "min",
            AtomicOp::Max => "max",
            AtomicOp::Umin => "umin",
            AtomicOp::Umax => "umax",
        }
    }
}

/// Lane-wise vector comparison producing all-ones (true) or all-zeros (false) lanes. Integer
/// lanes compare as signed, except in `vult`, `vugt`, `vule` and `vuge` (integer lanes only).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VCmpOp {
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
    Ult,
    Ugt,
    Ule,
    Uge,
}

impl VCmpOp {
    /// Every op, in AIRB encoding order (append new ops at the end).
    pub const ALL: [VCmpOp; 10] = [
        VCmpOp::Eq,
        VCmpOp::Ne,
        VCmpOp::Lt,
        VCmpOp::Gt,
        VCmpOp::Le,
        VCmpOp::Ge,
        VCmpOp::Ult,
        VCmpOp::Ugt,
        VCmpOp::Ule,
        VCmpOp::Uge,
    ];

    pub fn from_str_opt(s: &str) -> Option<Self> {
        Self::ALL.iter().copied().find(|op| op.as_str() == s)
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            VCmpOp::Eq => "veq",
            VCmpOp::Ne => "vne",
            VCmpOp::Lt => "vlt",
            VCmpOp::Gt => "vgt",
            VCmpOp::Le => "vle",
            VCmpOp::Ge => "vge",
            VCmpOp::Ult => "vult",
            VCmpOp::Ugt => "vugt",
            VCmpOp::Ule => "vule",
            VCmpOp::Uge => "vuge",
        }
    }

    /// Whether the op compares integer lanes as unsigned.
    pub fn is_unsigned(&self) -> bool {
        matches!(self, VCmpOp::Ult | VCmpOp::Ugt | VCmpOp::Ule | VCmpOp::Uge)
    }
}

/// Scalar op with one operand. Float rounding to an integral value: `floor`, `ceil`,
/// `round` (halfway cases away from zero, as C's `round`), `roundeven` (halfway cases to
/// even) and `roundz` (toward zero); integer bit counts: `popcnt`, `clz` and `ctz` (the bit
/// width for 0).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnaryOp {
    Sqrt,
    Neg,
    Abs,
    Floor,
    Ceil,
    Round,
    RoundEven,
    RoundZ,
    Popcnt,
    Clz,
    Ctz,
    Exp,
    Log,
    Tanh,
    Sin,
    Cos,
}

impl UnaryOp {
    pub const ALL: [UnaryOp; 16] = [
        UnaryOp::Sqrt,
        UnaryOp::Neg,
        UnaryOp::Abs,
        UnaryOp::Floor,
        UnaryOp::Ceil,
        UnaryOp::Round,
        UnaryOp::RoundEven,
        UnaryOp::RoundZ,
        UnaryOp::Popcnt,
        UnaryOp::Clz,
        UnaryOp::Ctz,
        UnaryOp::Exp,
        UnaryOp::Log,
        UnaryOp::Tanh,
        UnaryOp::Sin,
        UnaryOp::Cos,
    ];

    pub fn from_str_opt(s: &str) -> Option<Self> {
        Self::ALL.iter().copied().find(|op| op.as_str() == s)
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            UnaryOp::Sqrt => "sqrt",
            UnaryOp::Neg => "neg",
            UnaryOp::Abs => "abs",
            UnaryOp::Floor => "floor",
            UnaryOp::Ceil => "ceil",
            UnaryOp::Round => "round",
            UnaryOp::RoundEven => "roundeven",
            UnaryOp::RoundZ => "roundz",
            UnaryOp::Popcnt => "popcnt",
            UnaryOp::Clz => "clz",
            UnaryOp::Ctz => "ctz",
            UnaryOp::Exp => "exp",
            UnaryOp::Log => "log",
            UnaryOp::Tanh => "tanh",
            UnaryOp::Sin => "sin",
            UnaryOp::Cos => "cos",
        }
    }

    /// Rounding of a float to an integral float value.
    pub fn is_rounding(&self) -> bool {
        matches!(
            self,
            UnaryOp::Floor | UnaryOp::Ceil | UnaryOp::Round | UnaryOp::RoundEven | UnaryOp::RoundZ
        )
    }

    /// Bit counts of an integer.
    pub fn is_bit_count(&self) -> bool {
        matches!(self, UnaryOp::Popcnt | UnaryOp::Clz | UnaryOp::Ctz)
    }

    /// The f32 math functions of `crate::math`.
    pub fn is_math(&self) -> bool {
        crate::math::MathFn::of_unary(*self).is_some()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CastOp {
    Itof,
    Ftoi,
    Sext,
    Zext,
    Trunc,
    Fext,
    Ftrunc,
    Bitcast,
    /// Unsigned integer to float.
    Uitof,
    /// Float to unsigned integer, saturating (negative and NaN to 0).
    Ftoui,
}

impl CastOp {
    pub fn from_str_opt(s: &str) -> Option<Self> {
        match s {
            "itof" => Some(CastOp::Itof),
            "ftoi" => Some(CastOp::Ftoi),
            "sext" => Some(CastOp::Sext),
            "zext" => Some(CastOp::Zext),
            "trunc" => Some(CastOp::Trunc),
            "fext" => Some(CastOp::Fext),
            "ftrunc" => Some(CastOp::Ftrunc),
            "bitcast" => Some(CastOp::Bitcast),
            "uitof" => Some(CastOp::Uitof),
            "ftoui" => Some(CastOp::Ftoui),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            CastOp::Itof => "itof",
            CastOp::Ftoi => "ftoi",
            CastOp::Sext => "sext",
            CastOp::Zext => "zext",
            CastOp::Trunc => "trunc",
            CastOp::Fext => "fext",
            CastOp::Ftrunc => "ftrunc",
            CastOp::Bitcast => "bitcast",
            CastOp::Uitof => "uitof",
            CastOp::Ftoui => "ftoui",
        }
    }
}

/// Vector op with one operand. Like scalar casts, the suffix names the result lane
/// type: `f = vitof v:f32` (i32 lanes to f32; `:f64` from i64), `i = vftoi v:i32` (f32 to
/// i32, saturating, NaN to 0; `:i64` from f64), `vuitof` / `vftoui` the same with unsigned
/// integers (negative floats and NaN to 0), `w = vwidenlo v:i16` / `vwidenhi` (sign-extends
/// the low / high half of the narrower lanes; `vuwidenlo` / `vuwidenhi` zero-extend),
/// `e = vexp v:f32` (e^x; also `vlog`, `vtanh`, `vsin`, `vcos`, f32 only: see `crate::math`,
/// which writes them as fixed sequences of vector ops, so bit-identical everywhere).
/// The exception is `w = vfwidenlo v:f16` / `vfwidenhi` (also `:bf16`, and `:f32` to f64):
/// the result is the wider float, so the suffix names the float lanes whose low / high half
/// is widened (exactly; a NaN stays a NaN, its payload unspecified). Integer lanes widen to
/// floats too, signed and exactly: `:i16` to f32, `:i32` to f64 (`vwidenlo` then `vitof`,
/// which `crate::math` writes them as).
/// `vabs` and `vneg` wrap on integers (`vabs` of the minimum is the minimum) and clear or flip
/// the sign bit of floats; `vsqrt` is IEEE's correctly rounded square root and `vrsqrt` is
/// `1 / vsqrt(x)` (two roundings, not a hardware estimate, so results are identical
/// everywhere). `vrev` reverses the order of the lanes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VUnaryOp {
    Itof,
    Ftoi,
    WidenLo,
    WidenHi,
    Exp,
    FWidenLo,
    FWidenHi,
    Abs,
    Neg,
    Sqrt,
    Rsqrt,
    Rev,
    Floor,
    Ceil,
    Round,
    RoundEven,
    RoundZ,
    UWidenLo,
    UWidenHi,
    Uitof,
    Ftoui,
    Log,
    Tanh,
    Sin,
    Cos,
}

impl VUnaryOp {
    /// Every op, in AIRB encoding order (append new ops at the end).
    pub const ALL: [VUnaryOp; 25] = [
        VUnaryOp::Itof,
        VUnaryOp::Ftoi,
        VUnaryOp::WidenLo,
        VUnaryOp::WidenHi,
        VUnaryOp::Exp,
        VUnaryOp::FWidenLo,
        VUnaryOp::FWidenHi,
        VUnaryOp::Abs,
        VUnaryOp::Neg,
        VUnaryOp::Sqrt,
        VUnaryOp::Rsqrt,
        VUnaryOp::Rev,
        VUnaryOp::Floor,
        VUnaryOp::Ceil,
        VUnaryOp::Round,
        VUnaryOp::RoundEven,
        VUnaryOp::RoundZ,
        VUnaryOp::UWidenLo,
        VUnaryOp::UWidenHi,
        VUnaryOp::Uitof,
        VUnaryOp::Ftoui,
        VUnaryOp::Log,
        VUnaryOp::Tanh,
        VUnaryOp::Sin,
        VUnaryOp::Cos,
    ];

    pub fn from_str_opt(s: &str) -> Option<Self> {
        Self::ALL.iter().copied().find(|op| op.as_str() == s)
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            VUnaryOp::Itof => "vitof",
            VUnaryOp::Ftoi => "vftoi",
            VUnaryOp::WidenLo => "vwidenlo",
            VUnaryOp::WidenHi => "vwidenhi",
            VUnaryOp::Exp => "vexp",
            VUnaryOp::FWidenLo => "vfwidenlo",
            VUnaryOp::FWidenHi => "vfwidenhi",
            VUnaryOp::Abs => "vabs",
            VUnaryOp::Neg => "vneg",
            VUnaryOp::Sqrt => "vsqrt",
            VUnaryOp::Rsqrt => "vrsqrt",
            VUnaryOp::Rev => "vrev",
            VUnaryOp::Floor => "vfloor",
            VUnaryOp::Ceil => "vceil",
            VUnaryOp::Round => "vround",
            VUnaryOp::RoundEven => "vroundeven",
            VUnaryOp::RoundZ => "vroundz",
            VUnaryOp::UWidenLo => "vuwidenlo",
            VUnaryOp::UWidenHi => "vuwidenhi",
            VUnaryOp::Uitof => "vuitof",
            VUnaryOp::Ftoui => "vftoui",
            VUnaryOp::Log => "vlog",
            VUnaryOp::Tanh => "vtanh",
            VUnaryOp::Sin => "vsin",
            VUnaryOp::Cos => "vcos",
        }
    }

    /// Whether the op converts the low or high half of its lanes to lanes twice as wide.
    pub fn is_widen(&self) -> bool {
        matches!(
            self,
            VUnaryOp::WidenLo
                | VUnaryOp::WidenHi
                | VUnaryOp::UWidenLo
                | VUnaryOp::UWidenHi
                | VUnaryOp::FWidenLo
                | VUnaryOp::FWidenHi
        )
    }

    /// Whether the op works on the low half of its source lanes (`vwidenlo`, `vuwidenlo`,
    /// `vfwidenlo`).
    pub fn is_low_half(&self) -> bool {
        matches!(
            self,
            VUnaryOp::WidenLo | VUnaryOp::UWidenLo | VUnaryOp::FWidenLo
        )
    }

    /// Lane type of the result for the op's suffix `lane` (the suffix itself, except for
    /// `vfwiden*`, whose suffix names the source: f64 from f32 or i32, else f32).
    pub fn result_lane(&self, lane: Type) -> Type {
        match (self, lane) {
            (VUnaryOp::FWidenLo | VUnaryOp::FWidenHi, Type::F32 | Type::I32) => Type::F64,
            (VUnaryOp::FWidenLo | VUnaryOp::FWidenHi, _) => Type::F32,
            _ => lane,
        }
    }

    /// Lane type of the operand for the op's suffix `lane`.
    pub fn source_lane(&self, lane: Type) -> Type {
        match (self, lane) {
            (VUnaryOp::FWidenLo | VUnaryOp::FWidenHi, _) => lane,
            (VUnaryOp::Itof | VUnaryOp::Uitof, Type::F64) => Type::I64,
            (VUnaryOp::Itof | VUnaryOp::Uitof, _) => Type::I32,
            (VUnaryOp::Ftoi | VUnaryOp::Ftoui, Type::I64) => Type::F64,
            (VUnaryOp::Ftoi | VUnaryOp::Ftoui, _) => Type::F32,
            (
                VUnaryOp::WidenLo | VUnaryOp::WidenHi | VUnaryOp::UWidenLo | VUnaryOp::UWidenHi,
                Type::I16,
            ) => Type::I8,
            (
                VUnaryOp::WidenLo | VUnaryOp::WidenHi | VUnaryOp::UWidenLo | VUnaryOp::UWidenHi,
                Type::I32,
            ) => Type::I16,
            (
                VUnaryOp::WidenLo | VUnaryOp::WidenHi | VUnaryOp::UWidenLo | VUnaryOp::UWidenHi,
                _,
            ) => Type::I32,
            (VUnaryOp::Exp | VUnaryOp::Log | VUnaryOp::Tanh | VUnaryOp::Sin | VUnaryOp::Cos, _) => {
                Type::F32
            }
            (
                VUnaryOp::Abs
                | VUnaryOp::Neg
                | VUnaryOp::Sqrt
                | VUnaryOp::Rsqrt
                | VUnaryOp::Rev
                | VUnaryOp::Floor
                | VUnaryOp::Ceil
                | VUnaryOp::Round
                | VUnaryOp::RoundEven
                | VUnaryOp::RoundZ,
                _,
            ) => lane,
        }
    }
}

/// Constants of `vexp`, shared by every backend so its results are bit-identical:
/// e^x = 2^n * p(r) with n = round(x log2 e) (by adding and subtracting `MAGIC`),
/// r = x + n * NEG_LN2_HI + n * LN2_LO, and p the Cephes expf polynomial,
/// p(r) = 1 + r + r^2 * (((((POLY[0] r + POLY[1]) r + POLY[2]) r + POLY[3]) r + POLY[4]) r + POLY[5]).
/// Inputs are clamped to [CLAMP_LO, CLAMP_HI] first.
pub mod vexp {
    pub const CLAMP_LO: f32 = -87.0;
    pub const CLAMP_HI: f32 = 88.0;
    pub const LOG2E: f32 = std::f32::consts::LOG2_E;
    pub const MAGIC: f32 = 12_582_912.0;
    // The Cephes expf constants, written as the shortest decimals of their f32 values.
    pub const NEG_LN2_HI: f32 = -0.693_359_4;
    pub const LN2_LO: f32 = 0.000_212_194_44;
    pub const POLY: [f32; 6] = [
        0.000_198_756_91,
        0.001_398_199_9,
        0.008_333_452,
        0.041_665_796,
        0.166_666_66,
        0.5,
    ];
    /// Added to the bits of `x log2 e + MAGIC` to get the biased exponent n + 127.
    pub const EXP_BIAS: i32 = 127 - 0x4B40_0000;
}

/// Lane-wise shift, the amount taken modulo the lane width as for scalar shifts:
/// `r = vshl v, n:i32` (`vshr` arithmetic, `vushr` logical). The amount is an integer
/// scalar, the same for every lane, or a vector of `v`'s type giving each lane's amount in
/// the matching lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VShiftOp {
    Shl,
    Shr,
    Ushr,
}

impl VShiftOp {
    pub const ALL: [VShiftOp; 3] = [VShiftOp::Shl, VShiftOp::Shr, VShiftOp::Ushr];

    pub fn from_str_opt(s: &str) -> Option<Self> {
        Self::ALL.iter().copied().find(|op| op.as_str() == s)
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            VShiftOp::Shl => "vshl",
            VShiftOp::Shr => "vshr",
            VShiftOp::Ushr => "vushr",
        }
    }
}

/// Two-operand lane permutation (`r = vziplo a, b:f32`). With `n` lanes per vector:
/// `vziplo` interleaves the low halves, `r = [a0, b0, a1, b1, ..., a(n/2-1), b(n/2-1)]`, and
/// `vziphi` the high halves, `r = [a(n/2), b(n/2), ..., a(n-1), b(n-1)]`. `vunziplo` takes the
/// even lanes of `a` then `b`, `r = [a0, a2, ..., b0, b2, ...]`, and `vunziphi` the odd lanes,
/// so `vunziplo`/`vunziphi` of `vziplo a, b` and `vziphi a, b` give back `a` and `b`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VZipOp {
    ZipLo,
    ZipHi,
    UnzipLo,
    UnzipHi,
}

impl VZipOp {
    /// Every op, in AIRB encoding order.
    pub const ALL: [VZipOp; 4] = [
        VZipOp::ZipLo,
        VZipOp::ZipHi,
        VZipOp::UnzipLo,
        VZipOp::UnzipHi,
    ];

    pub fn from_str_opt(s: &str) -> Option<Self> {
        Self::ALL.iter().copied().find(|op| op.as_str() == s)
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            VZipOp::ZipLo => "vziplo",
            VZipOp::ZipHi => "vziphi",
            VZipOp::UnzipLo => "vunziplo",
            VZipOp::UnzipHi => "vunziphi",
        }
    }

    /// Whether the op interleaves (`vzip*`) rather than deinterleaves (`vunzip*`).
    pub fn is_zip(&self) -> bool {
        matches!(self, VZipOp::ZipLo | VZipOp::ZipHi)
    }

    /// Whether the op takes the low halves (`vziplo`) or the even lanes (`vunziplo`).
    pub fn is_lo(&self) -> bool {
        matches!(self, VZipOp::ZipLo | VZipOp::UnzipLo)
    }
}

/// Horizontal reduction of all lanes to a scalar of the lane type: `s = vsum v:f32`.
///
/// Lanes are combined as a recursive-halves tree: `reduce(v) = op(reduce(lo), reduce(hi))`,
/// which pairs adjacent lanes first. Every backend uses this order, so float results are
/// bit-identical for fixed-width vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VectorReduceOp {
    Sum,
    Max,
    Min,
}

impl VectorReduceOp {
    pub fn from_str_opt(s: &str) -> Option<Self> {
        match s {
            "vsum" => Some(VectorReduceOp::Sum),
            "vmaxr" => Some(VectorReduceOp::Max),
            "vminr" => Some(VectorReduceOp::Min),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            VectorReduceOp::Sum => "vsum",
            VectorReduceOp::Max => "vmaxr",
            VectorReduceOp::Min => "vminr",
        }
    }
}

/// Element index of a memory access: `p[i]` addresses `p + i * size`, the size being that of
/// `unit` when given (`p[i:f32]`), else of the access's own element type: the loaded or
/// stored scalar type, or the lane type of `ldm`/`stm`. `reg` is an i64 register.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Index {
    pub reg: String,
    pub unit: Option<Type>,
}

impl Index {
    /// The type whose size scales the index, for an access whose element type is `implied`.
    pub fn unit_or(&self, implied: Type) -> Type {
        self.unit.unwrap_or(implied)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Instruction {
    AssignConst {
        dst: String,
        val: Constant,
        ty: Type,
        span: Span,
    },
    Binary {
        op: BinaryOp,
        dst: String,
        lhs: String,
        rhs: String,
        span: Span,
    },
    Load {
        dst: String,
        ptr: String,
        index: Option<Index>,
        ty: Type,
        span: Span,
    },
    Store {
        ptr: String,
        index: Option<Index>,
        val: String,
        span: Span,
    },
    /// `call f(a)`, `r = call f(a)` or `q, r = call f(a)`: the destinations receive the
    /// callee's results in order (none: results discarded).
    Call {
        dsts: Vec<String>,
        func: String,
        args: Vec<String>,
        span: Span,
    },
    /// Fork-join parallel loop `par n, f(a, b)`: calls `f(i, a, b)` for every `i` in
    /// `[0, n)`, possibly concurrently and in any order, and returns when all calls finish.
    Par {
        count: String,
        func: String,
        args: Vec<String>,
        span: Span,
    },
    // SIMD Intrinsics
    /// Broadcast a scalar to every lane of a vector of type `ty` (default `v128`).
    Splat {
        dst: String,
        src: String,
        ty: Type,
        span: Span,
    },
    ExtractLane {
        dst: String,
        vec: String,
        lane: u32,
        ty: Type,
        span: Span,
    },
    /// Atomic accesses (`AtomicRmw`, `AtomicCas`, `AtomicLoad`, `AtomicStore`) are
    /// sequentially consistent, on an integer (i8 to i64) at an address aligned to its size,
    /// and may take an index like `ld`/`st` (`p[i]`, scaled by the integer's size).
    /// `old = atomic add p, v`: see `AtomicOp`.
    AtomicRmw {
        op: AtomicOp,
        dst: String,
        ptr: String,
        index: Option<Index>,
        val: String,
        span: Span,
    },
    /// `old = atomic cas p, expected, new`: stores `new` if the value is `expected`, and
    /// returns the value it found (the swap happened when `old == expected`).
    AtomicCas {
        dst: String,
        ptr: String,
        index: Option<Index>,
        expected: String,
        new: String,
        span: Span,
    },
    /// `v = atomic ld p:i64`.
    AtomicLoad {
        dst: String,
        ptr: String,
        index: Option<Index>,
        ty: Type,
        span: Span,
    },
    /// `atomic st p, v`.
    AtomicStore {
        ptr: String,
        index: Option<Index>,
        val: String,
        span: Span,
    },
    /// `r = inslane v, x, 3:f32`: `vec` with lane `lane` (lanes of type `ty`) replaced by the
    /// scalar `val`. The index is checked like `extlane`'s.
    InsertLane {
        dst: String,
        vec: String,
        val: String,
        lane: u32,
        ty: Type,
        span: Span,
    },
    // Memory Management Intrinsics
    Alloc {
        dst: String,
        size: String,
        span: Span,
    },
    Free {
        ptr: String,
        span: Span,
    },
    /// `p = stack 384`: the address of a `size`-byte buffer in the function's frame,
    /// 16-byte aligned, holding unspecified bytes. Each `stack` instruction is its own
    /// buffer, the same one every time it runs in a call, and lasts until the function
    /// returns.
    Stack {
        dst: String,
        size: u32,
        span: Span,
    },
    /// `p = addr table`: the address of the module's `data` named `name`.
    Addr {
        dst: String,
        name: String,
        span: Span,
    },
    // Phase 4 Extensions
    Unary {
        op: UnaryOp,
        dst: String,
        src: String,
        span: Span,
    },
    Cast {
        op: CastOp,
        dst: String,
        src: String,
        ty: Type,
        span: Span,
    },
    /// Scalar fused multiply-add `a * b + c` with one rounding (f32, f64).
    Fma {
        dst: String,
        a: String,
        b: String,
        c: String,
        span: Span,
    },
    Select {
        dst: String,
        cond: String,
        then_val: String,
        else_val: String,
        span: Span,
    },
    VectorReduce {
        op: VectorReduceOp,
        dst: String,
        src: String,
        ty: Type,
        span: Span,
    },
    // Lane-typed vector ops. `lane` is the scalar lane type.
    VBinary {
        op: VBinOp,
        dst: String,
        lhs: String,
        rhs: String,
        lane: Type,
        span: Span,
    },
    /// Fused multiply-add `a * b + c` with a single rounding.
    VFma {
        dst: String,
        a: String,
        b: String,
        c: String,
        lane: Type,
        span: Span,
    },
    VCmp {
        op: VCmpOp,
        dst: String,
        lhs: String,
        rhs: String,
        lane: Type,
        span: Span,
    },
    /// Bitwise select: bits of `then_val` where `mask` is 1, else bits of `else_val`.
    VSelect {
        dst: String,
        mask: String,
        then_val: String,
        else_val: String,
        span: Span,
    },
    /// One-operand vector op (see `VUnaryOp`); `lane` is the result lane type.
    VUnary {
        op: VUnaryOp,
        dst: String,
        src: String,
        lane: Type,
        span: Span,
    },
    /// `r = vnarrow lo, hi:i8`: both operands' lanes narrowed to `lane` with signed
    /// saturation, `lo`'s in the low half of the result and `hi`'s in the high half.
    VNarrow {
        dst: String,
        lo: String,
        hi: String,
        lane: Type,
        span: Span,
    },
    /// Lane-wise shift of `src` by `amount`, a scalar or a vector of per-lane amounts (see
    /// `VShiftOp`).
    VShift {
        op: VShiftOp,
        dst: String,
        src: String,
        amount: String,
        lane: Type,
        span: Span,
    },
    /// Lane permutation of two vectors (see `VZipOp`).
    VZip {
        op: VZipOp,
        dst: String,
        lhs: String,
        rhs: String,
        lane: Type,
        span: Span,
    },
    /// `r = vdup v, 3:f32`: lane `lane` of `vec` (lanes of type `ty`) in every lane of a vector
    /// of `vec`'s type. The index is checked like `extlane`'s.
    VDup {
        dst: String,
        vec: String,
        lane: u32,
        ty: Type,
        span: Span,
    },
    /// `r = vdot acc, a, b:i8`: `acc` has i32 lanes and `a`, `b` lanes of type `lane` (i8),
    /// four per i32 lane; lane `i` of the result is `acc[i] + a[4i]*b[4i] + ... +
    /// a[4i+3]*b[4i+3]`, signed products summed exactly and added with i32 wrapping.
    /// `r = vdotu acc, a, b:i8` (`unsigned`) reads `a`'s bytes as unsigned (0..255): one
    /// x86 VNNI `vpdpbusd`, or Arm `usdot` (FEAT_I8MM).
    VDot {
        dst: String,
        acc: String,
        a: String,
        b: String,
        lane: Type,
        unsigned: bool,
        span: Span,
    },
    /// Lane count of `vx` for `lane` (`n = vl f32`), as an i64.
    VLen {
        dst: String,
        lane: Type,
        span: Span,
    },
    /// Tail-masked load: `v = ldm p:vx, n:f32` loads the first `min(max(n, 0), lanes)`
    /// lanes and zeroes the rest. Memory past those lanes is never accessed.
    MaskedLoad {
        dst: String,
        ptr: String,
        index: Option<Index>,
        count: String,
        ty: Type,
        lane: Type,
        span: Span,
    },
    /// `v = vgather p, idx:f32`: lane `i` of the result is the element `p[idx[i]]` of type
    /// `lane`. The index vector `idx` has signed lanes of the same width (i32 for `f32` and
    /// `i32`, i64 for `f64` and `i64`), scaled by the element size, and gives the result's
    /// vector type. With a count, `vgather p, idx, n:f32` reads only the first
    /// `min(max(n, 0), lanes)` lanes and zeroes the rest.
    VGather {
        dst: String,
        ptr: String,
        idx: String,
        count: Option<String>,
        lane: Type,
        span: Span,
    },
    /// Tail-masked store: `stm p, v, n:f32` writes only the first `min(max(n, 0), lanes)` lanes.
    MaskedStore {
        ptr: String,
        index: Option<Index>,
        val: String,
        count: String,
        lane: Type,
        span: Span,
    },
    /// Matrix multiply-accumulate `mm pc, pa, pb, m, n, k:bf16`:
    /// `C[m x n] += A[m x k] * B[k x n]`, all row-major and contiguous. A and B hold
    /// `dtype` elements; C is f32, or i32 when `dtype` is i8.
    MatMul {
        pc: String,
        pa: String,
        pb: String,
        m: String,
        n: String,
        k: String,
        dtype: Type,
        span: Span,
    },
}

impl Instruction {
    /// The instruction's place in the source.
    pub fn span(&self) -> Span {
        match self {
            Instruction::AssignConst { span, .. }
            | Instruction::Binary { span, .. }
            | Instruction::Load { span, .. }
            | Instruction::Store { span, .. }
            | Instruction::AtomicRmw { span, .. }
            | Instruction::AtomicCas { span, .. }
            | Instruction::AtomicLoad { span, .. }
            | Instruction::AtomicStore { span, .. }
            | Instruction::Call { span, .. }
            | Instruction::Par { span, .. }
            | Instruction::Splat { span, .. }
            | Instruction::ExtractLane { span, .. }
            | Instruction::InsertLane { span, .. }
            | Instruction::Alloc { span, .. }
            | Instruction::Free { span, .. }
            | Instruction::Stack { span, .. }
            | Instruction::Addr { span, .. }
            | Instruction::Unary { span, .. }
            | Instruction::Cast { span, .. }
            | Instruction::Fma { span, .. }
            | Instruction::Select { span, .. }
            | Instruction::VectorReduce { span, .. }
            | Instruction::VBinary { span, .. }
            | Instruction::VFma { span, .. }
            | Instruction::VCmp { span, .. }
            | Instruction::VSelect { span, .. }
            | Instruction::VUnary { span, .. }
            | Instruction::VNarrow { span, .. }
            | Instruction::VShift { span, .. }
            | Instruction::VZip { span, .. }
            | Instruction::VDup { span, .. }
            | Instruction::VDot { span, .. }
            | Instruction::VLen { span, .. }
            | Instruction::MaskedLoad { span, .. }
            | Instruction::VGather { span, .. }
            | Instruction::MaskedStore { span, .. }
            | Instruction::MatMul { span, .. } => *span,
        }
    }

    /// Registers defined by this instruction (several for a call of a function returning
    /// several values).
    pub fn dsts(&self) -> Vec<&str> {
        if let Instruction::Call { dsts, .. } = self {
            return dsts.iter().map(String::as_str).collect();
        }
        self.dst().into_iter().collect()
    }

    /// The register defined by this instruction, if it defines exactly one.
    pub fn dst(&self) -> Option<&str> {
        match self {
            Instruction::AssignConst { dst, .. }
            | Instruction::Binary { dst, .. }
            | Instruction::Load { dst, .. }
            | Instruction::Splat { dst, .. }
            | Instruction::ExtractLane { dst, .. }
            | Instruction::InsertLane { dst, .. }
            | Instruction::Alloc { dst, .. }
            | Instruction::Stack { dst, .. }
            | Instruction::Addr { dst, .. }
            | Instruction::Select { dst, .. }
            | Instruction::Unary { dst, .. }
            | Instruction::Cast { dst, .. }
            | Instruction::VectorReduce { dst, .. }
            | Instruction::VBinary { dst, .. }
            | Instruction::VFma { dst, .. }
            | Instruction::VCmp { dst, .. }
            | Instruction::VSelect { dst, .. }
            | Instruction::VLen { dst, .. }
            | Instruction::VUnary { dst, .. }
            | Instruction::VNarrow { dst, .. }
            | Instruction::VShift { dst, .. }
            | Instruction::VZip { dst, .. }
            | Instruction::VDup { dst, .. }
            | Instruction::VDot { dst, .. }
            | Instruction::Fma { dst, .. }
            | Instruction::AtomicRmw { dst, .. }
            | Instruction::AtomicCas { dst, .. }
            | Instruction::AtomicLoad { dst, .. }
            | Instruction::MaskedLoad { dst, .. }
            | Instruction::VGather { dst, .. } => Some(dst),
            Instruction::Call { dsts, .. } => match dsts.as_slice() {
                [d] => Some(d),
                _ => None,
            },
            Instruction::Store { .. }
            | Instruction::AtomicStore { .. }
            | Instruction::Free { .. }
            | Instruction::MaskedStore { .. }
            | Instruction::MatMul { .. }
            | Instruction::Par { .. } => None,
        }
    }

    /// Mutable access to the registers this instruction defines.
    pub fn dsts_mut(&mut self) -> Vec<&mut String> {
        match self {
            Instruction::AssignConst { dst, .. }
            | Instruction::Binary { dst, .. }
            | Instruction::Load { dst, .. }
            | Instruction::Splat { dst, .. }
            | Instruction::ExtractLane { dst, .. }
            | Instruction::InsertLane { dst, .. }
            | Instruction::Alloc { dst, .. }
            | Instruction::Stack { dst, .. }
            | Instruction::Addr { dst, .. }
            | Instruction::Select { dst, .. }
            | Instruction::Unary { dst, .. }
            | Instruction::Cast { dst, .. }
            | Instruction::VectorReduce { dst, .. }
            | Instruction::VBinary { dst, .. }
            | Instruction::VFma { dst, .. }
            | Instruction::VCmp { dst, .. }
            | Instruction::VSelect { dst, .. }
            | Instruction::VLen { dst, .. }
            | Instruction::VUnary { dst, .. }
            | Instruction::VNarrow { dst, .. }
            | Instruction::VShift { dst, .. }
            | Instruction::VZip { dst, .. }
            | Instruction::VDup { dst, .. }
            | Instruction::VDot { dst, .. }
            | Instruction::Fma { dst, .. }
            | Instruction::AtomicRmw { dst, .. }
            | Instruction::AtomicCas { dst, .. }
            | Instruction::AtomicLoad { dst, .. }
            | Instruction::MaskedLoad { dst, .. }
            | Instruction::VGather { dst, .. } => vec![dst],
            Instruction::Call { dsts, .. } => dsts.iter_mut().collect(),
            Instruction::Store { .. }
            | Instruction::AtomicStore { .. }
            | Instruction::Free { .. }
            | Instruction::MaskedStore { .. }
            | Instruction::MatMul { .. }
            | Instruction::Par { .. } => vec![],
        }
    }

    /// Registers read by this instruction, in operand order.
    pub fn operands(&self) -> Vec<&String> {
        let index = self.index().map(|ix| &ix.reg);
        let mut regs = self.direct_operands();
        regs.extend(index);
        regs
    }

    /// The element index of a memory access (`p[i]`), if it has one.
    pub fn index(&self) -> Option<&Index> {
        match self {
            Instruction::Load { index, .. }
            | Instruction::Store { index, .. }
            | Instruction::MaskedLoad { index, .. }
            | Instruction::MaskedStore { index, .. }
            | Instruction::AtomicRmw { index, .. }
            | Instruction::AtomicCas { index, .. }
            | Instruction::AtomicLoad { index, .. }
            | Instruction::AtomicStore { index, .. } => index.as_ref(),
            _ => None,
        }
    }

    fn direct_operands(&self) -> Vec<&String> {
        match self {
            Instruction::AssignConst { .. }
            | Instruction::VLen { .. }
            | Instruction::Stack { .. }
            | Instruction::Addr { .. } => vec![],
            Instruction::Binary { lhs, rhs, .. }
            | Instruction::VBinary { lhs, rhs, .. }
            | Instruction::VCmp { lhs, rhs, .. }
            | Instruction::VZip { lhs, rhs, .. } => vec![lhs, rhs],
            Instruction::Load { ptr, .. }
            | Instruction::AtomicLoad { ptr, .. }
            | Instruction::Free { ptr, .. } => vec![ptr],
            Instruction::Store { ptr, val, .. }
            | Instruction::AtomicRmw { ptr, val, .. }
            | Instruction::AtomicStore { ptr, val, .. } => vec![ptr, val],
            Instruction::AtomicCas {
                ptr, expected, new, ..
            } => vec![ptr, expected, new],
            Instruction::Call { args, .. } => args.iter().collect(),
            Instruction::Par { count, args, .. } => std::iter::once(count).chain(args).collect(),
            Instruction::Splat { src, .. }
            | Instruction::Unary { src, .. }
            | Instruction::Cast { src, .. }
            | Instruction::VUnary { src, .. }
            | Instruction::VectorReduce { src, .. } => vec![src],
            Instruction::VNarrow { lo, hi, .. } => vec![lo, hi],
            Instruction::VShift { src, amount, .. } => vec![src, amount],
            Instruction::ExtractLane { vec, .. } | Instruction::VDup { vec, .. } => vec![vec],
            Instruction::InsertLane { vec, val, .. } => vec![vec, val],
            Instruction::Alloc { size, .. } => vec![size],
            Instruction::Select {
                cond: a,
                then_val: b,
                else_val: c,
                ..
            }
            | Instruction::VSelect {
                mask: a,
                then_val: b,
                else_val: c,
                ..
            }
            | Instruction::VFma { a, b, c, .. }
            | Instruction::Fma { a, b, c, .. }
            | Instruction::VDot {
                acc: a, a: b, b: c, ..
            } => vec![a, b, c],
            Instruction::MaskedLoad { ptr, count, .. } => vec![ptr, count],
            Instruction::VGather {
                ptr, idx, count, ..
            } => [ptr, idx].into_iter().chain(count.as_ref()).collect(),
            Instruction::MaskedStore {
                ptr, val, count, ..
            } => vec![ptr, val, count],
            Instruction::MatMul {
                pc,
                pa,
                pb,
                m,
                n,
                k,
                ..
            } => vec![pc, pa, pb, m, n, k],
        }
    }

    /// Mutable access to the registers read by this instruction (a memory access's index
    /// register last, as in `operands`).
    pub fn operands_mut(&mut self) -> Vec<&mut String> {
        match self {
            Instruction::AssignConst { .. }
            | Instruction::VLen { .. }
            | Instruction::Stack { .. }
            | Instruction::Addr { .. } => vec![],
            Instruction::Binary { lhs, rhs, .. }
            | Instruction::VBinary { lhs, rhs, .. }
            | Instruction::VCmp { lhs, rhs, .. }
            | Instruction::VZip { lhs, rhs, .. } => vec![lhs, rhs],
            Instruction::Load { ptr, index, .. } | Instruction::AtomicLoad { ptr, index, .. } => {
                std::iter::once(ptr)
                    .chain(index.as_mut().map(|ix| &mut ix.reg))
                    .collect()
            }
            Instruction::Free { ptr, .. } => vec![ptr],
            Instruction::Store {
                ptr, val, index, ..
            }
            | Instruction::AtomicRmw {
                ptr, val, index, ..
            }
            | Instruction::AtomicStore {
                ptr, val, index, ..
            } => [ptr, val]
                .into_iter()
                .chain(index.as_mut().map(|ix| &mut ix.reg))
                .collect(),
            Instruction::AtomicCas {
                ptr,
                expected,
                new,
                index,
                ..
            } => [ptr, expected, new]
                .into_iter()
                .chain(index.as_mut().map(|ix| &mut ix.reg))
                .collect(),
            Instruction::Call { args, .. } => args.iter_mut().collect(),
            Instruction::Par { count, args, .. } => {
                std::iter::once(count).chain(args.iter_mut()).collect()
            }
            Instruction::Splat { src, .. }
            | Instruction::Unary { src, .. }
            | Instruction::Cast { src, .. }
            | Instruction::VUnary { src, .. }
            | Instruction::VectorReduce { src, .. } => vec![src],
            Instruction::VNarrow { lo, hi, .. } => vec![lo, hi],
            Instruction::VShift { src, amount, .. } => vec![src, amount],
            Instruction::ExtractLane { vec, .. } | Instruction::VDup { vec, .. } => vec![vec],
            Instruction::InsertLane { vec, val, .. } => vec![vec, val],
            Instruction::Alloc { size, .. } => vec![size],
            Instruction::Select {
                cond: a,
                then_val: b,
                else_val: c,
                ..
            }
            | Instruction::VSelect {
                mask: a,
                then_val: b,
                else_val: c,
                ..
            }
            | Instruction::VFma { a, b, c, .. }
            | Instruction::Fma { a, b, c, .. }
            | Instruction::VDot {
                acc: a, a: b, b: c, ..
            } => vec![a, b, c],
            Instruction::VGather {
                ptr, idx, count, ..
            } => [ptr, idx].into_iter().chain(count.as_mut()).collect(),
            Instruction::MaskedLoad {
                ptr, count, index, ..
            } => [ptr, count]
                .into_iter()
                .chain(index.as_mut().map(|ix| &mut ix.reg))
                .collect(),
            Instruction::MaskedStore {
                ptr,
                val,
                count,
                index,
                ..
            } => [ptr, val, count]
                .into_iter()
                .chain(index.as_mut().map(|ix| &mut ix.reg))
                .collect(),
            Instruction::MatMul {
                pc,
                pa,
                pb,
                m,
                n,
                k,
                ..
            } => vec![pc, pa, pb, m, n, k],
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Terminator {
    Jmp {
        target: String,
        args: Vec<String>,
        span: Span,
    },
    Br {
        cond: String,
        then_block: String,
        then_args: Vec<String>,
        else_block: String,
        else_args: Vec<String>,
        span: Span,
    },
    /// `ret`, `ret v` or `ret a, b` (as many values as the function returns).
    Ret { vals: Vec<String>, span: Span },
    /// `switch v, other(args), 0: a(args), 3: b(args)`: jumps to the case whose value equals
    /// the integer `v`, or to `default` when none does. Case values are distinct.
    Switch {
        val: String,
        default: String,
        default_args: Vec<String>,
        cases: Vec<SwitchCase>,
        span: Span,
    },
}

/// One case of a `switch`: `value: target(args)`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SwitchCase {
    pub value: i64,
    pub target: String,
    pub args: Vec<String>,
}

impl Terminator {
    pub fn span(&self) -> Span {
        match self {
            Terminator::Jmp { span, .. }
            | Terminator::Br { span, .. }
            | Terminator::Ret { span, .. }
            | Terminator::Switch { span, .. } => *span,
        }
    }

    /// Each edge to another block: its target and arguments, in order (`br`: then, else;
    /// `switch`: the default, then the cases).
    pub fn edges(&self) -> Vec<(&String, &Vec<String>)> {
        match self {
            Terminator::Jmp { target, args, .. } => vec![(target, args)],
            Terminator::Br {
                then_block,
                then_args,
                else_block,
                else_args,
                ..
            } => vec![(then_block, then_args), (else_block, else_args)],
            Terminator::Ret { .. } => vec![],
            Terminator::Switch {
                default,
                default_args,
                cases,
                ..
            } => std::iter::once((default, default_args))
                .chain(cases.iter().map(|c| (&c.target, &c.args)))
                .collect(),
        }
    }

    /// `edges`, mutably.
    pub fn edges_mut(&mut self) -> Vec<(&mut String, &mut Vec<String>)> {
        match self {
            Terminator::Jmp { target, args, .. } => vec![(target, args)],
            Terminator::Br {
                then_block,
                then_args,
                else_block,
                else_args,
                ..
            } => vec![(then_block, then_args), (else_block, else_args)],
            Terminator::Ret { .. } => vec![],
            Terminator::Switch {
                default,
                default_args,
                cases,
                ..
            } => std::iter::once((default, default_args))
                .chain(cases.iter_mut().map(|c| (&mut c.target, &mut c.args)))
                .collect(),
        }
    }

    /// Every register the terminator reads: the condition or switched value, then the
    /// edges' arguments (or the returned values).
    pub fn operands(&self) -> Vec<&String> {
        match self {
            Terminator::Jmp { args, .. } => args.iter().collect(),
            Terminator::Br {
                cond,
                then_args,
                else_args,
                ..
            } => std::iter::once(cond)
                .chain(then_args)
                .chain(else_args)
                .collect(),
            Terminator::Ret { vals, .. } => vals.iter().collect(),
            Terminator::Switch {
                val,
                default_args,
                cases,
                ..
            } => std::iter::once(val)
                .chain(default_args)
                .chain(cases.iter().flat_map(|c| &c.args))
                .collect(),
        }
    }

    /// `operands`, mutably.
    pub fn operands_mut(&mut self) -> Vec<&mut String> {
        match self {
            Terminator::Jmp { args, .. } => args.iter_mut().collect(),
            Terminator::Br {
                cond,
                then_args,
                else_args,
                ..
            } => std::iter::once(cond)
                .chain(then_args.iter_mut())
                .chain(else_args.iter_mut())
                .collect(),
            Terminator::Ret { vals, .. } => vals.iter_mut().collect(),
            Terminator::Switch {
                val,
                default_args,
                cases,
                ..
            } => std::iter::once(val)
                .chain(default_args.iter_mut())
                .chain(cases.iter_mut().flat_map(|c| c.args.iter_mut()))
                .collect(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Block {
    pub label: String,
    pub params: Vec<(String, Type)>,
    pub instructions: Vec<Instruction>,
    pub terminator: Terminator,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Function {
    pub name: String,
    pub params: Vec<(String, Type)>,
    /// Result types: none, one (`->i32`) or several (`->(vx, vx)`).
    pub rets: Vec<Type>,
    /// `inline fn`: every call is replaced by the body before code generation
    /// (`crate::inline`); the function also stays callable on its own.
    pub inline: bool,
    pub blocks: Vec<Block>,
    pub span: Span,
}

impl Function {
    /// Whether hosts can call the function through a scalar trampoline: no vector
    /// parameters or results, and at most one result. Others are callable only from AIR.
    pub fn has_trampoline(&self) -> bool {
        self.rets.len() <= 1
            && self
                .params
                .iter()
                .map(|(_, t)| *t)
                .chain(self.rets.iter().copied())
                .all(|t| !t.is_vector())
    }

    /// Byte offsets of the function's `stack` buffers in its frame, in instruction order,
    /// each 16-byte aligned, and their total size: the layout both backends give them.
    pub fn stack_layout(&self) -> (Vec<u32>, u32) {
        let mut offsets = Vec::new();
        let mut total = 0u32;
        for inst in self.blocks.iter().flat_map(|b| &b.instructions) {
            if let Instruction::Stack { size, .. } = inst {
                offsets.push(total);
                total += size.div_ceil(16) * 16;
            }
        }
        (offsets, total)
    }

    /// The single result type, `None` for no result; `Err` with the count for several.
    pub fn single_ret(&self) -> Result<Option<Type>, usize> {
        match self.rets.as_slice() {
            [] => Ok(None),
            [t] => Ok(Some(*t)),
            many => Err(many.len()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExternFunction {
    pub name: String,
    pub params: Vec<(String, Type)>,
    pub ret_type: Option<Type>,
    pub span: Span,
}

/// `data name:ty = v, v, ...`: read-only bytes of the module (the values of type `ty`,
/// little-endian), 64-byte aligned, whose address `addr name` gives. A write to them is an
/// error: `ERR_MEMORY_VIOLATION` in the sandbox, a fault otherwise.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Data {
    pub name: String,
    /// Element type: an integer (i8 to i64) or f32/f64.
    pub ty: Type,
    pub bytes: Vec<u8>,
    pub span: Span,
}

/// Alignment of every `data` table.
pub const DATA_ALIGN: usize = 64;
/// Largest `stack` buffer, in bytes.
pub const MAX_STACK_BYTES: u32 = 1 << 16;

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Module {
    pub extern_functions: Vec<ExternFunction>,
    pub data: Vec<Data>,
    pub functions: Vec<Function>,
}

impl Module {
    /// True when some function runs a `par` loop.
    pub fn uses_par(&self) -> bool {
        self.functions
            .iter()
            .flat_map(|f| &f.blocks)
            .flat_map(|b| &b.instructions)
            .any(|i| matches!(i, Instruction::Par { .. }))
    }
}
