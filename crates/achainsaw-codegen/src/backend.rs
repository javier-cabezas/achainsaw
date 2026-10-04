//! Code generation backend selection.

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

    /// Resolves a user choice: `cranelift`, `llvm`, or `auto`/`None`. `auto` uses
    /// `ACHAINSAW_BACKEND` when set, and Cranelift otherwise. Errors if the chosen backend
    /// is not in this build.
    pub fn resolve(choice: Option<&str>) -> Result<Backend> {
        let choice = choice.map(str::trim).filter(|c| !c.is_empty());
        let backend = match choice {
            Some(c) if !c.eq_ignore_ascii_case("auto") => c.parse()?,
            _ => match std::env::var(BACKEND_ENV) {
                Ok(v) if !v.trim().is_empty() && !v.trim().eq_ignore_ascii_case("auto") => {
                    v.parse().map_err(|e| anyhow!("{e} (from {BACKEND_ENV})"))?
                }
                _ => Backend::Cranelift,
            },
        };
        if !backend.is_available() {
            return Err(anyhow!(
                "[ERR_BACKEND_UNAVAILABLE] The {backend} backend is not in this build; rebuild with the `llvm` feature (and LLVM 22 installed)"
            ));
        }
        Ok(backend)
    }
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
}
