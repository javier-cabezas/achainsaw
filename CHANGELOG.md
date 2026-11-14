# Changelog

Each release's changes. achainsaw follows [semantic versioning](https://semver.org) from 1.0;
until then a minor version (0.2, 0.3, ...) may change AIR, AIRB or the Rust and Python APIs,
and a patch version does not.

## 0.1.0 (unreleased)

The first release.

**The language and its tools**
- AIR, a flat SSA language with typed scalars, pointers and vectors (`v128`, `v256`,
  `v512` and `vx`, the CPU's width), several results per function, function handles,
  `switch`, atomics, constant `data`, stack buffers, `inline` functions and a standard
  library linked with `use` ([docs/language.md](docs/language.md)).
- A validator whose errors are JSON `Diagnostic`s with an `ERR_*` code, the source span and
  what it takes to fix them, an optimizer, and AIRB, a compact bytecode.
- Two backends with the same results: Cranelift (JIT and AOT, compiles in under a
  millisecond) and the optional LLVM 22 backend, which uses the CPU's full vector width,
  int8 dot products (VNNI, `sdot`) and matrix engines (AMX, SME)
  ([docs/backends.md](docs/backends.md)).
- `par` loops on every core ([docs/parallel.md](docs/parallel.md)).
- A sandbox for untrusted code: every memory access checked against a private arena,
  fuel against runaway loops, a stack limit, and only C math functions as externals.

**Ways to use it**
- The `achainsaw` CLI: check, run, benchmark, assemble, optimize, and build objects and
  shared libraries ([docs/cli.md](docs/cli.md)).
- An MCP server with seven tools for Claude Code, Claude Desktop and other MCP clients, its
  instructions an AIR primer ([docs/mcp.md](docs/mcp.md)).
- The `achainsaw` Python package, on NumPy arrays ([docs/python.md](docs/python.md)), calls
  from AIR to C functions (`extfn`, [docs/ffi.md](docs/ffi.md)), and AOT objects and shared
  libraries for C, C++ and Rust.

**Kernels and models** (examples, not a stable API)
- 22 kernels in `examples/kernels/`, each benchmarked against NumPy and PyTorch
  ([docs/kernels.md](docs/kernels.md)).
- A single-call LLM decode and prefill in one AIR file, run on GGUF files: Llama 3, Qwen3,
  Qwen3-MoE, gpt-oss, GLM-4.5 and dots.llm1, with Q8_0, Q4_0, Q4_1, Q4_K, Q5_K, Q6_K,
  IQ4_NL and MXFP4 weights, a bf16 or int8 KV cache, and sliding windows in a ring cache
  ([docs/llama.md](docs/llama.md)).

**Packages**
- CLI binaries for Linux (x86-64, AArch64, musl), macOS (x86-64, Apple silicon) and Windows,
  Python wheels for Python 3.11 and later on the same platforms, and the crates
  `achainsaw`, `achainsaw-ir`, `achainsaw-codegen` and `achainsaw-llvm`. The packages and
  binaries use the Cranelift backend; LLVM needs a build with LLVM 22.
