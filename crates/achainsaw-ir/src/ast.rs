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

    /// Pre-v2 spellings (`vfadd`, `viadd`, ...) with their implied lane type.
    pub fn from_legacy(s: &str) -> Option<(Self, Type)> {
        match s {
            "vfadd" => Some((VBinOp::Add, Type::F32)),
            "vfsub" => Some((VBinOp::Sub, Type::F32)),
            "vfmul" => Some((VBinOp::Mul, Type::F32)),
            "vfdiv" => Some((VBinOp::Div, Type::F32)),
            "viadd" => Some((VBinOp::Add, Type::I32)),
            "visub" => Some((VBinOp::Sub, Type::I32)),
            "vimul" => Some((VBinOp::Mul, Type::I32)),
            _ => None,
        }
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
    /// Accepts the canonical names and the pre-v2 aliases `vfsum`, `visum`, `vfmax`.
    pub fn from_str_opt(s: &str) -> Option<Self> {
        match s {
            "vsum" | "vfsum" | "visum" => Some(VectorReduceOp::Sum),
            "vmaxr" | "vfmax" => Some(VectorReduceOp::Max),
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
    // AIR v2 lane-typed vector ops. `lane` is the scalar lane type.
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
    /// Lane count of `vx` for `lane` (`n = vl f32`), as an i64.
    VLen {
        dst: String,
        lane: Type,
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
            | Instruction::VLen { dst, .. } => Some(dst),
            Instruction::Call { dst, .. } => dst.as_deref(),
            Instruction::Store { .. } | Instruction::Free { .. } => None,
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
            Instruction::Splat { src, .. }
            | Instruction::Unary { src, .. }
            | Instruction::Cast { src, .. }
            | Instruction::VectorReduce { src, .. } => vec![src],
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
            Instruction::Splat { src, .. }
            | Instruction::Unary { src, .. }
            | Instruction::Cast { src, .. }
            | Instruction::VectorReduce { src, .. } => vec![src],
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
