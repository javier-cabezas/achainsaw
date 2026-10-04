//! Optional LLVM backend for achainsaw: lowers validated AIR modules to LLVM IR, then JIT
//! compiles them with ORC LLJIT or emits AOT object files.
//!
//! The crate is empty unless the `llvm` feature is enabled, so `cargo build --workspace`
//! works on machines without LLVM. Enable it through `achainsaw-codegen/llvm` (or the CLI
//! and Python `llvm` features) and point `LLVM_SYS_221_PREFIX` at an LLVM 22 install.

#![cfg(feature = "llvm")]

mod aot;
mod jit;
mod lower;
mod target;

pub use aot::{compile_assembly, compile_object};
pub use jit::{LlvmJit, RuntimeHooks};
pub use lower::{trampoline_name, LowerOptions, MatrixUnits, SandboxBounds, VxShape};
pub use target::{host_cpu_name, TargetSpec};
