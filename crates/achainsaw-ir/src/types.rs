use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Type {
    I8,
    I16,
    I32,
    I64,
    F32,
    F64,
    Ptr,
    /// 128-bit universal SIMD vector (4xf32, 4xi32, etc.)
    /// Architecture is extensible to V256 and V512 in future phases.
    V128,
}

impl Type {
    pub fn from_str_token(s: &str) -> Option<Self> {
        match s {
            "i8" => Some(Type::I8),
            "i16" => Some(Type::I16),
            "i32" => Some(Type::I32),
            "i64" => Some(Type::I64),
            "f32" => Some(Type::F32),
            "f64" => Some(Type::F64),
            "ptr" => Some(Type::Ptr),
            "v128" => Some(Type::V128),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Type::I8 => "i8",
            Type::I16 => "i16",
            Type::I32 => "i32",
            Type::I64 => "i64",
            Type::F32 => "f32",
            Type::F64 => "f64",
            Type::Ptr => "ptr",
            Type::V128 => "v128",
        }
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}
