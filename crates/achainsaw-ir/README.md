# achainsaw-ir

The AIR language of [achainsaw](https://github.com/javier-cabezas/achainsaw), a compiler for the code AI agents write: the lexer and parser of AIR text, the validator (SSA and type checks) and its machine-readable `Diagnostic`s (an `ERR_*` code, a span and context for self-repair), the optimizer, the AIRB bytecode codec, the math functions shared by every backend, and the AIR standard library (`std/std.air`, linked by a module's `use` line).

Most users want the [`achainsaw`](https://crates.io/crates/achainsaw) command-line tool and MCP server, or the `achainsaw` Python package. The language is described in [docs/language.md](https://github.com/javier-cabezas/achainsaw/blob/main/docs/language.md). The crate is at 0.x: AIR, AIRB and this API may change between minor versions.
