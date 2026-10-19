# Contributing to achainsaw

Thanks for your interest! Bug reports, programs that miscompile, and pull
requests are all welcome.

## Reporting a problem

The most useful reports are:

- **A program that gives wrong results or crashes,** on either backend.
- **A diagnostic that misled a model:** an error whose code, span or context
  didn't help an agent fix its program. Include what the model wrote and what
  it tried next.

Please include the smallest AIR program that shows it, the command or MCP
call (`achainsaw run ... --json`, or the `air_run` arguments), the backend,
and the output of `achainsaw cpu`.

## Building and testing

You need a Rust toolchain, and Python 3 with NumPy for the Python tests.

```sh
cargo build --workspace
cargo test --workspace
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
```

The Python tests need the extension at the repository's root:

```sh
cargo build --release -p achainsaw-py
cp target/release/libachainsaw.so achainsaw.so   # macOS: libachainsaw.dylib
python tests/test_py_binding.py
python tests/test_gguf_decode.py
cargo build -p achainsaw && python tests/test_mcp_e2e.py
python tests/test_readme.py                      # the README's console output
```

Python code is fully typed, which CI checks with `python tools/check_py_types.py`
and `mypy` (strict, configured in `pyproject.toml`).

**The LLVM backend** needs LLVM 22. Run the suite on both backends:

```sh
export LLVM_SYS_221_PREFIX=/usr/lib/llvm-22
FEATURES=achainsaw/llvm,achainsaw-codegen/llvm,achainsaw-llvm/llvm
cargo test --workspace --features $FEATURES
ACHAINSAW_BACKEND=llvm cargo test --workspace --features $FEATURES
```

**Lower ISA levels** run on the same machine with `--isa avx2` (or `sse`,
`avx`, `neon`, ...) or `ACHAINSAW_MAX_ISA`. SVE and SME code runs under QEMU
with `tools/qemu-aarch64/run.sh`, and CI's arm64 job runs the vector tests at
several SVE lengths.

### Performance

Back performance claims with the benchmarks, and compare several runs on an
idle machine:

```sh
python benchmarks/benchmark_kernels.py           # every kernel against NumPy and PyTorch
python benchmarks/benchmark_decode.py --gguf FILE --threads 8 --prompt-tokens 512
python benchmarks/perf_compare.py BASE_DIR HEAD_DIR
```

CI runs `perf_compare.py` on every pull request: it builds the base and the
head, runs both alternately, and fails if a kernel gets more than 25% slower
on either backend. An intended slowdown needs the `perf-regression-ok`
label.

## Layout

| Path | What's there |
|---|---|
| `crates/achainsaw-ir` | lexer, parser, validator, optimizer, inliner, AIRB bytecode, diagnostics, and the standard library (`std/std.air`) |
| `crates/achainsaw-codegen` | Cranelift lowering, the JIT and its runtime (fuel, sandbox, `par`), AOT objects, CPU features |
| `crates/achainsaw-llvm` | the LLVM backend (empty without its `llvm` feature) |
| `crates/achainsaw-cli` | the `achainsaw` binary and the MCP server |
| `crates/achainsaw-py` | Python bindings (PyO3) and their type stubs |
| `examples/` | AIR programs, each with its `.airb` bytecode; `kernels/` holds the benchmarked kernels |
| `benchmarks/` | comparisons with NumPy, PyTorch and llama.cpp; the GGUF reader |
| `tests/` | Python, MCP and AOT end-to-end tests |
| `docs/` | the user documentation |

## Pull requests

- **One language, both backends.** A language change goes through the lexer,
  parser, validator, both backends' `lower.rs`, AIRB encoding and
  `to_air_text`, and gives the same results on Cranelift and LLVM. Regenerate
  the `.airb` files in `examples/` with `achainsaw assemble`.
- **Errors are diagnostics:** an `ERR_*` code, a span, and a JSON context an
  agent can act on, never plain prose.
- **Memory accesses are sandboxed:** a new instruction that touches memory
  calls the sandbox's check, with a test in
  `crates/achainsaw-codegen/tests/sandbox.rs`.
- **Tests:** check new operations against a scalar reference, on every lane
  type and ISA level the operation supports.
- [CLAUDE.md](CLAUDE.md) describes the code's layout and conventions in
  detail, with the pitfalls we've hit. It's written for AI assistants, and
  it's the best map of the code for people too.

By contributing, you agree that your contributions are licensed under the
[Apache License, Version 2.0](LICENSE).
