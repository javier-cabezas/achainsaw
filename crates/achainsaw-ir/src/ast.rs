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

/// Lane-wise vector binary op: `r = vadd a, b:f32`.
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
}

impl VBinOp {
    pub const ALL: [VBinOp; 9] = [
        VBinOp::Add,
        VBinOp::Sub,
        VBinOp::Mul,
        VBinOp::Div,
        VBinOp::Min,
        VBinOp::Max,
        VBinOp::And,
        VBinOp::Or,
        VBinOp::Xor,
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
        }
    }

    pub fn is_bitwise(&self) -> bool {
        matches!(self, VBinOp::And | VBinOp::Or | VBinOp::Xor)
    }
}

/// Lane-wise vector comparison producing all-ones (true) or all-zeros (false) lanes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VCmpOp {
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
}

impl VCmpOp {
    pub const ALL: [VCmpOp; 6] = [
        VCmpOp::Eq,
        VCmpOp::Ne,
        VCmpOp::Lt,
        VCmpOp::Gt,
        VCmpOp::Le,
        VCmpOp::Ge,
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
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnaryOp {
    Sqrt,
    Neg,
    Abs,
}

impl UnaryOp {
    pub fn from_str_opt(s: &str) -> Option<Self> {
        match s {
            "sqrt" => Some(UnaryOp::Sqrt),
            "neg" => Some(UnaryOp::Neg),
            "abs" => Some(UnaryOp::Abs),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            UnaryOp::Sqrt => "sqrt",
            UnaryOp::Neg => "neg",
            UnaryOp::Abs => "abs",
        }
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
        }
    }
}

/// Lane-wise vector op with one operand. Like scalar casts, the suffix names the result lane
/// type: `f = vitof v:f32` (i32 lanes to f32), `i = vftoi v:i32` (f32 to i32, saturating, NaN
/// to 0), `w = vwidenlo v:i16` / `vwidenhi` (sign-extends the low / high half of the
/// narrower lanes), `e = vexp v:f32` (e^x with a fixed algorithm, so bit-identical everywhere).
/// The exception is `w = vfwidenlo v:f16` / `vfwidenhi` (also `:bf16`): the result is always
/// f32, so the suffix names the 16-bit float lanes whose low / high half is widened (exactly;
/// a NaN stays a NaN, its payload unspecified).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VUnaryOp {
    Itof,
    Ftoi,
    WidenLo,
    WidenHi,
    Exp,
    FWidenLo,
    FWidenHi,
}

impl VUnaryOp {
    pub const ALL: [VUnaryOp; 7] = [
        VUnaryOp::Itof,
        VUnaryOp::Ftoi,
        VUnaryOp::WidenLo,
        VUnaryOp::WidenHi,
        VUnaryOp::Exp,
        VUnaryOp::FWidenLo,
        VUnaryOp::FWidenHi,
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
        }
    }

    /// Whether the op converts the low or high half of its lanes to lanes twice as wide.
    pub fn is_widen(&self) -> bool {
        matches!(
            self,
            VUnaryOp::WidenLo | VUnaryOp::WidenHi | VUnaryOp::FWidenLo | VUnaryOp::FWidenHi
        )
    }

    /// Whether the op works on the low half of its source lanes (`vwidenlo`, `vfwidenlo`).
    pub fn is_low_half(&self) -> bool {
        matches!(self, VUnaryOp::WidenLo | VUnaryOp::FWidenLo)
    }

    /// Lane type of the result for the op's suffix `lane` (the suffix itself, except for
    /// `vfwiden*`, whose suffix names the source).
    pub fn result_lane(&self, lane: Type) -> Type {
        match self {
            VUnaryOp::FWidenLo | VUnaryOp::FWidenHi => Type::F32,
            _ => lane,
        }
    }

    /// Lane type of the operand for the op's suffix `lane`.
    pub fn source_lane(&self, lane: Type) -> Type {
        match (self, lane) {
            (VUnaryOp::FWidenLo | VUnaryOp::FWidenHi, _) => lane,
            (VUnaryOp::Itof, _) => Type::I32,
            (VUnaryOp::Ftoi, _) => Type::F32,
            (VUnaryOp::WidenLo | VUnaryOp::WidenHi, Type::I16) => Type::I8,
            (VUnaryOp::WidenLo | VUnaryOp::WidenHi, Type::I32) => Type::I16,
            (VUnaryOp::WidenLo | VUnaryOp::WidenHi, _) => Type::I32,
            (VUnaryOp::Exp, _) => Type::F32,
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

/// Lane-wise shift by a scalar amount, taken modulo the lane width as for scalar shifts:
/// `r = vshl v, n:i32` (`vshr` arithmetic, `vushr` logical).
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
        ty: Type,
        span: Span,
    },
    Store {
        ptr: String,
        val: String,
        span: Span,
    },
    Call {
        dst: Option<String>,
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
    /// One-operand lane-wise op (`vitof`, `vftoi`, `vwidenlo`, `vwidenhi`, `vexp`); `lane` is
    /// the result lane type.
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
    /// Lane-wise shift of `src` by the scalar `amount` (see `VShiftOp`).
    VShift {
        op: VShiftOp,
        dst: String,
        src: String,
        amount: String,
        lane: Type,
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
        count: String,
        ty: Type,
        lane: Type,
        span: Span,
    },
    /// Tail-masked store: `stm p, v, n:f32` writes only the first `min(max(n, 0), lanes)` lanes.
    MaskedStore {
        ptr: String,
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
    /// Register defined by this instruction, if any.
    pub fn dst(&self) -> Option<&str> {
        match self {
            Instruction::AssignConst { dst, .. }
            | Instruction::Binary { dst, .. }
            | Instruction::Load { dst, .. }
            | Instruction::Splat { dst, .. }
            | Instruction::ExtractLane { dst, .. }
            | Instruction::Alloc { dst, .. }
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
            | Instruction::MaskedLoad { dst, .. } => Some(dst),
            Instruction::Call { dst, .. } => dst.as_deref(),
            Instruction::Store { .. }
            | Instruction::Free { .. }
            | Instruction::MaskedStore { .. }
            | Instruction::MatMul { .. }
            | Instruction::Par { .. } => None,
        }
    }

    /// Registers read by this instruction, in operand order.
    pub fn operands(&self) -> Vec<&String> {
        match self {
            Instruction::AssignConst { .. } | Instruction::VLen { .. } => vec![],
            Instruction::Binary { lhs, rhs, .. }
            | Instruction::VBinary { lhs, rhs, .. }
            | Instruction::VCmp { lhs, rhs, .. } => vec![lhs, rhs],
            Instruction::Load { ptr, .. } | Instruction::Free { ptr, .. } => vec![ptr],
            Instruction::Store { ptr, val, .. } => vec![ptr, val],
            Instruction::Call { args, .. } => args.iter().collect(),
            Instruction::Par { count, args, .. } => std::iter::once(count).chain(args).collect(),
            Instruction::Splat { src, .. }
            | Instruction::Unary { src, .. }
            | Instruction::Cast { src, .. }
            | Instruction::VUnary { src, .. }
            | Instruction::VectorReduce { src, .. } => vec![src],
            Instruction::VNarrow { lo, hi, .. } => vec![lo, hi],
            Instruction::VShift { src, amount, .. } => vec![src, amount],
            Instruction::ExtractLane { vec, .. } => vec![vec],
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
            | Instruction::VFma { a, b, c, .. } => vec![a, b, c],
            Instruction::MaskedLoad { ptr, count, .. } => vec![ptr, count],
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

    /// Mutable access to the registers read by this instruction.
    pub fn operands_mut(&mut self) -> Vec<&mut String> {
        match self {
            Instruction::AssignConst { .. } | Instruction::VLen { .. } => vec![],
            Instruction::Binary { lhs, rhs, .. }
            | Instruction::VBinary { lhs, rhs, .. }
            | Instruction::VCmp { lhs, rhs, .. } => vec![lhs, rhs],
            Instruction::Load { ptr, .. } | Instruction::Free { ptr, .. } => vec![ptr],
            Instruction::Store { ptr, val, .. } => vec![ptr, val],
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
            Instruction::ExtractLane { vec, .. } => vec![vec],
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
            | Instruction::VFma { a, b, c, .. } => vec![a, b, c],
            Instruction::MaskedLoad { ptr, count, .. } => vec![ptr, count],
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
    Ret {
        val: Option<String>,
        span: Span,
    },
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
    pub ret_type: Option<Type>,
    pub blocks: Vec<Block>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExternFunction {
    pub name: String,
    pub params: Vec<(String, Type)>,
    pub ret_type: Option<Type>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Module {
    pub extern_functions: Vec<ExternFunction>,
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
