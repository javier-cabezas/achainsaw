# achainsaw

Compiler toolchain for AI agents: AIR (flat SSA IR) text/bytecode -> Cranelift JIT or AOT native code. See README.md for the user-facing tour.

## Layout
- `crates/achainsaw-ir`: lexer, parser, AST, validator (SSA/type checks), optimizer (`opt.rs`), AIRB codec (`binary.rs`), JSON diagnostics (`diag.rs`).
- `crates/achainsaw-codegen`: Cranelift lowering (`lower.rs`), JIT facade over both backends plus the shared runtime hooks (`jit.rs`), backend selection (`backend.rs`), AOT objects/shared libs (`aot.rs`), CPU feature detection, ISA cap, and Cranelift/LLVM target settings (`cpu.rs`).
- `crates/achainsaw-llvm`: optional LLVM backend (empty without its `llvm` feature): AIR -> LLVM IR (`lower.rs`), ORC LLJIT (`jit.rs`), object emission (`aot.rs`).
- `crates/achainsaw-cli`: `achainsaw` binary (`main.rs`) and the MCP stdio server (`mcp.rs`).
- `crates/achainsaw-py`: PyO3 bindings; type stubs in `achainsaw.pyi`.
- `examples/`: `.air` sources with matching `.airb` bytecode; `tests/`: Python e2e tests.

## Commands
- Build: `cargo build --workspace` (release CLI: `cargo build --release -p achainsaw`)
- Rust tests: `cargo test --workspace`
- MCP e2e: `cargo build -p achainsaw && python tests/test_mcp_e2e.py`
- Python binding tests need the extension built first (`maturin develop` or copy `target/*/libachainsaw.so` to `achainsaw.so`), then `python tests/test_py_binding.py`.
- Lint (CI): `cargo fmt --all -- --check` and `cargo clippy --workspace --all-targets -- -D warnings`
- LLVM backend (LLVM 22): `export LLVM_SYS_221_PREFIX=/usr/lib/llvm-22`, then add `--features achainsaw/llvm,achainsaw-codegen/llvm,achainsaw-llvm/llvm` to build/test/clippy. Run the suite on LLVM with `ACHAINSAW_BACKEND=llvm cargo test --workspace --features ...`; CI's `test-llvm` job runs it on both backends.

## Git workflow
- Never commit or push directly to `main`. Do work on a feature branch, push it, and open a pull request against `main` (`gh pr create --base main`).
- Make sure `cargo test --workspace` passes before opening the PR.

## Conventions
- Errors are machine-readable `Diagnostic`s with an `ERR_*` code, span, and `context` for agent self-repair; never plain prose.
- A language change touches lexer -> parser -> validator -> `lower.rs` -> `binary.rs` (AIRB encode/decode) -> `to_air_text`. Regenerate affected `.airb` files with `achainsaw assemble`, but keep the v1 kernel `.airb` files as compatibility fixtures (`crates/achainsaw-ir/tests/vectors.rs` checks them). New AIRB opcodes bump `VERSION` in `binary.rs` and must keep older versions decoding.
- Vector ops: lane-type rules live in `validator.rs` (`vbin_lane_types`, `VFMA_LANE_TYPES`) and apply to every backend. `crates/achainsaw-codegen/tests/vector_ops.rs` compiles every op x lane x width for x86_64 (all levels) and aarch64 and checks results against a scalar reference model; extend its `all_kernels()` and `reference()` for new ops. New v2 mnemonics are `TokenKind::VOp` so they remain valid register names outside the position after `=`. `crates/achainsaw-codegen/tests/half_masked_mm.rs` covers f16/bf16 conversions (exhaustively, against the `half` crate), `ldm`/`stm` (including a guard-page test that faults on any out-of-bounds access), and `mm` (tolerance-based for floats, exact for i8).
- Two backends, one semantics: a language change lowers in both `achainsaw-codegen/src/lower.rs` (Cranelift) and `achainsaw-llvm/src/lower.rs`, with identical results, fuel-check placement, runtime hooks and sandbox checks. Extend `crates/achainsaw-codegen/tests/backend_parity.rs` for new scalar semantics; the rest of the suite runs on both via `ACHAINSAW_BACKEND`. `vx` is 128 bits on Cranelift but 256/512 bits (AVX2/AVX-512) or scalable (`<vscale x ...>`, SVE) on LLVM: tests must size `vx` buffers with `JitEngine::vx_bits()`, and LLVM vector lowering must handle `ScalableVectorValue`s (use the `vec1!`/`vec2!` macros). Scalable code is verified by AOT-compiling for `+sve` locally and by CI's arm64 job (native SVE2 plus a QEMU vector-length sweep).
- LLVM `mm` (`crates/achainsaw-llvm/src/matmul.rs`) dispatches per dtype to AMX, SME or a vector-FMA kernel from `LowerOptions::matrix`; the caller keeps the sandbox check and fuel charge. AMX is compile-only (checked by `mm_kernels_match_target_matrix_engines` in assembly); SME and SVE run under QEMU with `tools/qemu-aarch64/run.sh <features> <vector bytes...>` (needs `qemu-user` and the `aarch64-unknown-linux-gnu` Rust target), which links AOT objects into a freestanding harness.
- Long-running bulk ops must charge fuel in the JIT via `rt_consume_fuel` (see `emit_matmul` in `lower.rs`) so MCP's sandbox still stops runaway programs.
- MCP `air_run` runs code under `JitEngine::enable_sandbox`: `alloc` draws from a private arena, every memory access is bounds-checked against it (`emit_bounds_check` in `lower.rs`; `mm` via `rt_sandbox_check_mm`), `free` must name a live allocation, and function entry checks stack depth. New instructions that touch memory must call the guard, and `crates/achainsaw-codegen/tests/sandbox.rs` must cover them; violations surface as `ERR_MEMORY_VIOLATION` / `ERR_STACK_OVERFLOW`, never a crash.
- `SERVER_INSTRUCTIONS` in `crates/achainsaw-cli/src/mcp.rs` is the AIR primer shown to MCP clients (Claude Code, Claude Desktop). Update it when syntax changes; `test_server_instructions_example_is_valid_air` JIT-runs its example.
- New MCP tools: add a `handle_*` function, a `get_tools_list` entry, a `tools/call` match arm, and a unit test.
- `.mcp.json` registers the local MCP server for Claude Code in this repo.
- Vector ISA work: test lower tiers on one machine with `--isa <level>` or `ACHAINSAW_MAX_ISA` (`sse`, `avx`, `avx2`, `avx512`, `amx`, `neon`, `sve`, `sve2`, `sme`); in tests use `JitEngine::with_features(&CpuFeatures::host().capped(level)?)` since the cap is process-global. Cranelift's generic `x86-64-v3`/`v4` presets omit `has_avx`, so always configure ISAs through `cpu.rs` (it completes feature prerequisites).
