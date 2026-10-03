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
    // Universal 128-bit SIMD Vector Ops
    VfAdd,
    VfSub,
    VfMul,
    VfDiv,
    ViAdd,
    ViSub,
    ViMul,
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
            "vfadd" => Some(BinaryOp::VfAdd),
            "vfsub" => Some(BinaryOp::VfSub),
            "vfmul" => Some(BinaryOp::VfMul),
            "vfdiv" => Some(BinaryOp::VfDiv),
            "viadd" => Some(BinaryOp::ViAdd),
            "visub" => Some(BinaryOp::ViSub),
            "vimul" => Some(BinaryOp::ViMul),
            _ => None,
        }
    }

    pub fn is_comparison(&self) -> bool {
        matches!(
            self,
            BinaryOp::Eq | BinaryOp::Ne | BinaryOp::Lt | BinaryOp::Gt | BinaryOp::Le | BinaryOp::Ge
        )
    }

    pub fn is_vector(&self) -> bool {
        matches!(
            self,
            BinaryOp::VfAdd
                | BinaryOp::VfSub
                | BinaryOp::VfMul
                | BinaryOp::VfDiv
                | BinaryOp::ViAdd
                | BinaryOp::ViSub
                | BinaryOp::ViMul
        )
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
    Splat {
        dst: String,
        src: String,
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
