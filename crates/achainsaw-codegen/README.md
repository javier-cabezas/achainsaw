# achainsaw-codegen

Native code generation for [achainsaw](https://github.com/javier-cabezas/achainsaw)'s AIR: the Cranelift backend (JIT and AOT objects and shared libraries), `JitEngine` with its sandbox (bounds-checked memory, fuel, stack limits), the `par` fork-join runtime, CPU feature detection and ISA caps. With its `llvm` feature it also drives the LLVM 22 backend in [`achainsaw-llvm`](https://crates.io/crates/achainsaw-llvm).

Most users want the [`achainsaw`](https://crates.io/crates/achainsaw) command-line tool and MCP server, or the `achainsaw` Python package; [docs/backends.md](https://github.com/javier-cabezas/achainsaw/blob/main/docs/backends.md) compares the backends. The crate is at 0.x: its API may change between minor versions.
