//! Code generation backend selection.

use achainsaw_ir::ast::{Instruction, Module};
use achainsaw_ir::types::Type;
use anyhow::{anyhow, Result};
use std::fmt;
use std::str::FromStr;

/// Environment variable naming the backend that `auto` (and `JitEngine::new`) uses.
pub const BACKEND_ENV: &str = "ACHAINSAW_BACKEND";

/// Code generator for JIT and AOT compilation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// Fast-compiling default; 128-bit vector code.
    Cranelift,
    /// Optional (`llvm` cargo feature); slower to compile, fully optimizing.
    Llvm,
}

impl Backend {
    pub fn as_str(&self) -> &'static str {
        match self {
            Backend::Cranelift => "cranelift",
            Backend::Llvm => "llvm",
        }
    }

    /// Whether this build includes the backend.
    pub fn is_available(&self) -> bool {
        match self {
            Backend::Cranelift => true,
            Backend::Llvm => cfg!(feature = "llvm"),
        }
    }

    /// Backends compiled into this build.
    pub fn available() -> Vec<Backend> {
        [Backend::Cranelift, Backend::Llvm]
            .into_iter()
            .filter(Backend::is_available)
            .collect()
    }

    /// Resolves a user choice without a module: `cranelift`, `llvm`, or `auto`/`None`,
    /// which uses `ACHAINSAW_BACKEND` when set and Cranelift otherwise. Errors if the chosen
    /// backend is not in this build.
    pub fn resolve(choice: Option<&str>) -> Result<Backend> {
        Self::checked(Self::explicit(choice)?.unwrap_or(Backend::Cranelift))
    }

    /// Resolves a user choice for compiling `module`. Like [`Backend::resolve`], except
    /// that `auto` picks LLVM, when this build has it, for modules that use vectors wider
    /// than 128 bits, scalable `vx`, or `mm` (see [`uses_wide_vectors`]); compiling takes
    /// a few milliseconds longer but the code uses the full vector and matrix hardware.
    pub fn resolve_for(choice: Option<&str>, module: &Module) -> Result<Backend> {
        let backend = match Self::explicit(choice)? {
            Some(b) => b,
            None if Backend::Llvm.is_available() && uses_wide_vectors(module) => Backend::Llvm,
            None => Backend::Cranelift,
        };
        Self::checked(backend)
    }

    /// The backend named by `choice` or, for `auto`/`None`, by `ACHAINSAW_BACKEND`.
    fn explicit(choice: Option<&str>) -> Result<Option<Backend>> {
        let named = |s: &str| {
            let s = s.trim();
            (!s.is_empty() && !s.eq_ignore_ascii_case("auto")).then(|| s.to_string())
        };
        if let Some(c) = choice.and_then(named) {
            return c.parse().map(Some);
        }
        match std::env::var(BACKEND_ENV).ok().as_deref().and_then(named) {
            Some(v) => v
                .parse()
                .map(Some)
                .map_err(|e| anyhow!("{e} (from {BACKEND_ENV})")),
            None => Ok(None),
        }
    }

    fn checked(backend: Backend) -> Result<Backend> {
        if !backend.is_available() {
            return Err(anyhow!(
                "[ERR_BACKEND_UNAVAILABLE] The {backend} backend is not in this build; rebuild with the `llvm` feature (and LLVM 22 installed)"
            ));
        }
        Ok(backend)
    }
}

/// Whether `module` uses `v256`, `v512`, `vx` (including `vl`) or `mm`: code that the
/// LLVM backend compiles to wider registers or matrix engines than Cranelift's 128 bits.
pub fn uses_wide_vectors(module: &Module) -> bool {
    let wide = |t: &Type| matches!(t, Type::V256 | Type::V512 | Type::Vx);
    module.functions.iter().any(|f| {
        f.params.iter().any(|(_, t)| wide(t))
            || f.rets.iter().any(wide)
            || f.blocks.iter().any(|b| {
                b.params.iter().any(|(_, t)| wide(t))
                    || b.instructions.iter().any(|i| match i {
                        Instruction::Load { ty, .. }
                        | Instruction::Splat { ty, .. }
                        | Instruction::MaskedLoad { ty, .. }
                        | Instruction::Cast { ty, .. } => wide(ty),
                        Instruction::VLen { .. } | Instruction::MatMul { .. } => true,
                        _ => false,
                    })
            })
    })
}

impl fmt::Display for Backend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for Backend {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "cranelift" | "clif" => Ok(Backend::Cranelift),
            "llvm" => Ok(Backend::Llvm),
            other => Err(anyhow!(
                "[ERR_UNKNOWN_BACKEND] Unknown backend '{other}'; expected cranelift, llvm or auto"
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_reports_availability() {
        assert_eq!("LLVM".parse::<Backend>().unwrap(), Backend::Llvm);
        assert_eq!(
            Backend::resolve(Some("cranelift")).unwrap(),
            Backend::Cranelift
        );
        assert!(Backend::resolve(Some("gcc"))
            .unwrap_err()
            .to_string()
            .contains("ERR_UNKNOWN_BACKEND"));
        assert_eq!(
            Backend::resolve(Some("llvm")).is_ok(),
            cfg!(feature = "llvm")
        );
        assert!(Backend::available().contains(&Backend::Cranelift));
    }

    #[test]
    fn auto_prefers_llvm_only_for_wide_vector_modules() {
        let scalar = achainsaw_ir::parse_and_validate(
            "fn f(x:i32)->i32\n  b0:\n    y = add x, 1:i32\n    ret y\n",
        )
        .unwrap();
        let v128 = achainsaw_ir::parse_and_validate(
            "fn f(p:ptr)\n  b0:\n    v = ld p:v128\n    st p, v\n    ret\n",
        )
        .unwrap();
        let vx =
            achainsaw_ir::parse_and_validate("fn f()->i64\n  b0:\n    n = vl f32\n    ret n\n")
                .unwrap();
        let wide = achainsaw_ir::parse_and_validate(
            "fn f(p:ptr)\n  b0:\n    v = ld p:v512\n    st p, v\n    ret\n",
        )
        .unwrap();
        assert!(!uses_wide_vectors(&scalar));
        assert!(!uses_wide_vectors(&v128));
        assert!(uses_wide_vectors(&vx));
        assert!(uses_wide_vectors(&wide));
        if std::env::var(BACKEND_ENV).is_err() {
            let llvm = cfg!(feature = "llvm");
            assert_eq!(
                Backend::resolve_for(None, &wide).unwrap() == Backend::Llvm,
                llvm
            );
            assert_eq!(
                Backend::resolve_for(Some("auto"), &scalar).unwrap(),
                Backend::Cranelift
            );
        }
        assert_eq!(
            Backend::resolve_for(Some("cranelift"), &wide).unwrap(),
            Backend::Cranelift
        );
    }
}
