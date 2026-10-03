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

    /// Width in bits of integer types (`ptr` is 64-bit). `None` for floats and vectors.
    pub fn int_bits(&self) -> Option<u32> {
        match self {
            Type::I8 => Some(8),
            Type::I16 => Some(16),
            Type::I32 => Some(32),
            Type::I64 | Type::Ptr => Some(64),
            _ => None,
        }
    }

    /// Total bit width of any type.
    pub fn bit_width(&self) -> Option<u32> {
        match self {
            Type::I8 => Some(8),
            Type::I16 => Some(16),
            Type::I32 | Type::F32 => Some(32),
            Type::I64 | Type::F64 | Type::Ptr => Some(64),
            Type::V128 => Some(128),
        }
    }

    /// Size in bytes in memory or registers.
    pub fn byte_size(&self) -> usize {
        match self {
            Type::I8 => 1,
            Type::I16 => 2,
            Type::I32 | Type::F32 => 4,
            Type::I64 | Type::F64 | Type::Ptr => 8,
            Type::V128 => 16,
        }
    }

    pub fn is_int(&self) -> bool {
        matches!(self, Type::I8 | Type::I16 | Type::I32 | Type::I64)
    }

    pub fn is_float(&self) -> bool {
        matches!(self, Type::F32 | Type::F64)
    }

    /// Number of lanes a 128-bit vector has when viewed as this scalar type.
    pub fn lane_count(&self) -> Option<u32> {
        match self {
            Type::I8 => Some(16),
            Type::I16 => Some(8),
            Type::I32 | Type::F32 => Some(4),
            Type::I64 | Type::F64 => Some(2),
            _ => None,
        }
    }

    /// True if integer literal `n` is representable in this type, as either a signed
    /// or an unsigned value (so both `-1:i8` and `255:i8` are accepted).
    pub fn int_literal_fits(&self, n: i64) -> bool {
        match self {
            Type::I8 => (i8::MIN as i64..=u8::MAX as i64).contains(&n),
            Type::I16 => (i16::MIN as i64..=u16::MAX as i64).contains(&n),
            Type::I32 => (i32::MIN as i64..=u32::MAX as i64).contains(&n),
            Type::I64 | Type::Ptr => true,
            _ => false,
        }
    }

    /// Wraps `v` to this integer type's width, sign-extended back into an `i64`.
    /// This is the canonical in-register form used by constant folding and codegen.
    pub fn wrap_int(&self, v: i64) -> i64 {
        match self {
            Type::I8 => v as i8 as i64,
            Type::I16 => v as i16 as i64,
            Type::I32 => v as i32 as i64,
            _ => v,
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
