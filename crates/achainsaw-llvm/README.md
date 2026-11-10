# achainsaw-llvm

The optional LLVM 22 backend of [achainsaw](https://github.com/javier-cabezas/achainsaw): AIR lowered to LLVM IR, an ORC JIT and object emission. Without its `llvm` feature the crate is empty, so the rest of achainsaw builds without LLVM installed; with it, set `LLVM_SYS_221_PREFIX` to an LLVM 22 installation.

It is used through [`achainsaw-codegen`](https://crates.io/crates/achainsaw-codegen)'s `llvm` feature (`cargo install achainsaw --features llvm`). [docs/backends.md](https://github.com/javier-cabezas/achainsaw/blob/main/docs/backends.md) compares the backends. The crate is at 0.x: its API may change between minor versions.
