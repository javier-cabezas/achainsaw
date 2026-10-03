# achainsaw

Compiler toolchain for AI agents: AIR (flat SSA IR) text/bytecode -> Cranelift JIT or AOT native code. See README.md for the user-facing tour.

## Layout
- `crates/achainsaw-ir`: lexer, parser, AST, validator (SSA/type checks), optimizer (`opt.rs`), AIRB codec (`binary.rs`), JSON diagnostics (`diag.rs`).
- `crates/achainsaw-codegen`: Cranelift lowering (`lower.rs`), JIT (`jit.rs`), AOT objects/shared libs (`aot.rs`).
- `crates/achainsaw-cli`: `achainsaw` binary (`main.rs`) and the MCP stdio server (`mcp.rs`).
- `crates/achainsaw-py`: PyO3 bindings; type stubs in `achainsaw.pyi`.
- `examples/`: `.air` sources with matching `.airb` bytecode; `tests/`: Python e2e tests.

## Commands
- Build: `cargo build --workspace` (release CLI: `cargo build --release -p achainsaw`)
- Rust tests: `cargo test --workspace`
- MCP e2e: `cargo build -p achainsaw && python tests/test_mcp_e2e.py`
- Python binding tests need the extension built first (`maturin develop` or copy `target/*/libachainsaw.so` to `achainsaw.so`), then `python tests/test_py_binding.py`.
- Lint (CI): `cargo fmt --all -- --check` and `cargo clippy --workspace --all-targets -- -D warnings`

## Git workflow
- Never commit or push directly to `main`. Do work on a feature branch, push it, and open a pull request against `main` (`gh pr create --base main`).
- Make sure `cargo test --workspace` passes before opening the PR.

## Conventions
- Errors are machine-readable `Diagnostic`s with an `ERR_*` code, span, and `context` for agent self-repair; never plain prose.
- A language change touches lexer -> parser -> validator -> `lower.rs` -> `binary.rs` (AIRB encode/decode) -> `to_air_text`. Regenerate affected `.airb` files with `achainsaw assemble`.
- `SERVER_INSTRUCTIONS` in `crates/achainsaw-cli/src/mcp.rs` is the AIR primer shown to MCP clients (Claude Code, Claude Desktop). Update it when syntax changes; `test_server_instructions_example_is_valid_air` JIT-runs its example.
- New MCP tools: add a `handle_*` function, a `get_tools_list` entry, a `tools/call` match arm, and a unit test.
- `.mcp.json` registers the local MCP server for Claude Code in this repo.
