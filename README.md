# achainsaw

[![CI](https://github.com/javier-cabezas/achainsaw/actions/workflows/ci.yml/badge.svg)](https://github.com/javier-cabezas/achainsaw/actions/workflows/ci.yml)
[![MCP server](https://img.shields.io/badge/MCP-server-blueviolet)](docs/mcp.md)
[![Llama 3.2 1B prefill](https://img.shields.io/badge/Llama%203.2%201B%20prefill-1.7--2.4x%20llama.cpp-brightgreen)](docs/llama.md)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**A compiler for the code AI agents write.** Agents write AIR, a small, flat
SSA language. achainsaw checks it, answers mistakes with JSON that names the
fix, and compiles it in under a millisecond to native code for x86-64 and
AArch64. Claude Code and other MCP clients get it as tools, and run the code
in a sandbox.

```console
$ achainsaw run examples/agent/sum_squares.air --func sum_squares --args 1000000
{
  "backend": "cranelift",
  "compile_time_us": 627,
  "exec_time_us": 500,
  "function": "sum_squares",
  "parse_time_us": 61,
  "result": 333332833333500000,
  "status": "ok",
  "threads": 16,
  "total_time_ms": 1.273
}
```

- **Made for models to write:** one operation per line, no nesting, and
  values that change around loops passed as block parameters. Errors are JSON with an `ERR_*` code, the source span, and
  what it takes to fix them, such as the registers in scope.
- **Safe to run what a model wrote:** through MCP, every load and store is
  checked against the sandbox's memory, runaway loops run out of fuel, and
  deep recursion hits a stack limit. Each is an error code, never a crash.
- **Compiles in under a millisecond** on Cranelift, the default backend.
  The optional LLVM backend takes a few milliseconds and uses the CPU's full
  vector width.
- **Fast code:** vectors that take the CPU's width (SSE to AVX-512, NEON,
  SVE), int8 dot products (VNNI, `sdot`), matrix engines (AMX, SME), and
  `par` loops on every core. A whole Llama 3.2 1B forward pass is one AIR
  file, which prefills 1.7–2.4x faster than llama.cpp on the same CPU and
  decodes as fast (1.9x as fast at 8K tokens of context).
- **Use it from anywhere:** the CLI, an MCP server, Python on NumPy arrays, or
  objects and shared libraries for C, C++ and Rust. CI tests Linux, macOS and
  Windows.

## Contents

- [Quick start](#quick-start)
- [See it work](#see-it-work)
- [Use it from your code](#use-it-from-your-code)
- [The language](#the-language)
- [How it works](#how-it-works)
- [How it compares](#how-it-compares)
- [FAQ](#faq)
- [Roadmap and contributing](#roadmap-and-contributing)

## Quick start

You need a Rust toolchain ([rustup](https://rustup.rs)). Build achainsaw and
run an example:

```sh
git clone https://github.com/javier-cabezas/achainsaw.git
cd achainsaw
cargo build --release -p achainsaw
target/release/achainsaw run examples/agent/sum_squares.air --func sum_squares --args 10
```

Or install only the binary, without cloning:
`cargo install --git https://github.com/javier-cabezas/achainsaw achainsaw`.

**In Claude Code**, install the binary and register its MCP server:

```sh
cargo install --path crates/achainsaw-cli
claude mcp add --scope user achainsaw -- achainsaw mcp
```

Then ask Claude for something like *"count the primes below a million with
achainsaw"*. Claude writes the AIR, checks it with `air_check`, fixes what
the diagnostics point at, and runs it with `air_run`. The server's
instructions teach the model AIR, so it needs no other documentation. Inside
this repository, Claude Code finds the server through [`.mcp.json`](.mcp.json).
[docs/mcp.md](docs/mcp.md) covers Claude Desktop and the seven tools.

**In Python**, install the package (it builds with Rust):

```sh
pip install git+https://github.com/javier-cabezas/achainsaw.git
```

That builds the Cranelift backend; [docs/python.md](docs/python.md#installation)
adds LLVM.

## See it work

**A mistake, and its fix.** Here an agent returns a register it never defined
([examples/agent/sum_squares_bug.air](examples/agent/sum_squares_bug.air)).
The diagnostic names the registers in scope at that point, so the fix is to
pick one:

```console
$ achainsaw check examples/agent/sum_squares_bug.air
{
  "status": "error",
  "error_code": "ERR_UNDEFINED_REG",
  "message": "Register 'total' is used before definition",
  "instruction_index": 0,
  "span": {
    "start": 326,
    "end": 329,
    "line": 14,
    "column": 5
  },
  "context": {
    "available_registers": [
      "acc",
      "i",
      "more",
      "n"
    ],
    "block": "done",
    "function": "sum_squares",
    "target": "total"
  }
}
$ achainsaw run examples/agent/sum_squares.air --func sum_squares --args 10
{
  "backend": "cranelift",
  "compile_time_us": 627,
  "exec_time_us": 0,
  "function": "sum_squares",
  "parse_time_us": 110,
  "result": 285,
  "status": "ok",
  "threads": 16,
  "total_time_ms": 0.792
}
```

**Code that goes wrong stops cleanly.** A loop that never ends runs out of
fuel, which every branch consumes when a run has a budget:

```console
$ achainsaw run examples/infinite_loop.air --fuel 1000000
{
  "error_code": "ERR_EXECUTION",
  "message": "[ERR_OUT_OF_FUEL] Execution halted: loop fuel budget exhausted",
  "status": "error"
}
```

MCP's `air_run` also runs code in a sandbox: memory comes only from `alloc`,
in an arena of its own, and every access outside it fails. This is a read
100 million floats past a 4 KB buffer:

```json
{
  "error_code": "ERR_EXECUTION",
  "message": "[ERR_MEMORY_VIOLATION] Access of 4 bytes at 0x73582bd77410 is outside memory obtained from alloc",
  "status": "error"
}
```

**A Llama in one file.** [`llama_decode.air`](examples/kernels/llama_decode.air)
runs a whole Llama-style forward pass in one call: a token in, the next token
out, with quantized weights (Q4_0, Q8_0, Q4_K, Q5_K, Q6_K) read straight from
a GGUF file. Llama 3.2 1B Instruct, continuing "The capital of France is" on
a laptop's Ryzen 7 8845HS with 8 threads:

```console
$ curl -LO https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_0.gguf
$ python benchmarks/benchmark_decode.py --gguf Llama-3.2-1B-Instruct-Q4_0.gguf --threads 8 --no-numpy --no-torch
Llama 3.2 1B Instruct: 16 layers, 0.79 GB of packed weights (qkv q4, o q4, gate_up q4, down q4/q8, lm q6), loaded in 13.6 s
prompt: 6 tokens; generating up to 32
  AIR cranelift, 8 thr   prompt   7.42 ms/token ( 134.8 tokens/s), decode   16.24 ms/token    61.6 tokens/s   48.9 GB/s of weights
    ' Paris. The capital of Germany is Berlin. The capital of Italy is Rome. The capital of Spain is Madrid. The capital of Portugal is Lisbon. The capital'
  AIR llvm, 8 thr        prompt   3.36 ms/token ( 297.9 tokens/s), decode   15.25 ms/token    65.6 tokens/s   52.1 GB/s of weights
    ' Paris. The capital of Germany is Berlin. The capital of Italy is Rome. The capital of Spain is Madrid. The capital of Portugal is Lisbon. The capital'
```

Against llama.cpp on the same file, threads and machine, measured with its own
`llama-bench` (build 6868a52, with its default f16 KV cache) on prompts of
each length, then 64 generated tokens:

| Context | Prefill, AIR (LLVM) | Prefill, llama.cpp | Decode, AIR (LLVM) | Decode, llama.cpp |
|---|---|---|---|---|
| 64 tokens | 790 tokens/s | 441 tokens/s | 66.3 tokens/s | 65.6 tokens/s |
| 512 tokens | 1036 tokens/s | 437 tokens/s | 67.4 tokens/s | 60.7 tokens/s |
| 2048 tokens | 838 tokens/s | 398 tokens/s | 64.1 tokens/s | 48.6 tokens/s |
| 8192 tokens | 486 tokens/s | 283 tokens/s | 51.9 tokens/s | 27.1 tokens/s |

It prints the same text as llama.cpp, a prefill leaves exactly the state that
decoding the prompt token by token would, and its decode reads weights at
about the memory's bandwidth. [docs/llama.md](docs/llama.md) explains how,
with the numbers for K-quant files and against PyTorch.

**Kernels against NumPy and PyTorch.** [`examples/kernels/`](examples/kernels)
holds 22 kernels for LLM inference, vector search and more: quantized GEMV
and GEMM, attention, norms, activations, RoPE, top-k, a histogram, sparse
matrix-vector products. On LLVM, each is faster
than PyTorch's CPU path for the same operation
([docs/kernels.md](docs/kernels.md)):

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/kernels-vs-numpy-dark.svg">
  <img alt="Speedup of each kernel over NumPy on a log scale, PyTorch, Cranelift and LLVM side by side" src="docs/kernels-vs-numpy-light.svg">
</picture>

## Use it from your code

**Python.** `compile` returns a kernel whose functions take NumPy arrays as
pointers, without copies:

```python
import achainsaw
import numpy as np

kernel = achainsaw.compile(open("examples/saxpy_vx.air").read())
x = np.arange(8, dtype=np.float32)
y = np.ones(8, dtype=np.float32)
kernel.run("saxpy", 2.0, x, y, len(x))  # y = 2 x + y, in place
print(y)                                # [ 1.  3.  5.  7.  9. 11. 13. 15.]
```

Compiling takes about 0.25 ms and a call 0.2 µs. Diagnostics arrive as
dictionaries on `CompilationError` ([docs/python.md](docs/python.md)).

**C, C++ and Rust.** `achainsaw build` writes an object file or a shared
library with C-ABI functions, for x86-64 and AArch64
([docs/cli.md](docs/cli.md#ahead-of-time-compilation-achainsaw-build)):

```sh
achainsaw build examples/kernels/gemv_f32.air --shared
achainsaw --backend llvm build examples/kernels/gemv_f32.air --target-cpu znver4 -o gemv_f32.o
```

**C libraries from AIR.** `extfn` declares C functions, the C math library is
available, and Python can load shared libraries or register callbacks
([docs/ffi.md](docs/ffi.md)).

## The language

A vector loop that runs at the CPU's vector width, from
[`examples/saxpy_vx.air`](examples/saxpy_vx.air). `vl` is the number of
lanes, and the masked `ldm`/`stm` handle the remainder:

```air
fn saxpy(a:f32, x:ptr, y:ptr, n:i64)
  b0:
    va = splat a:vx
    w = vl f32
    jmp vloop(0:i64)
  vloop(i:i64):
    more = lt i, n
    br more, vbody, done
  vbody:
    rest = sub n, i
    xv = ldm x[i]:vx, rest:f32
    yv = ldm y[i]:vx, rest:f32
    r = vfma va, xv, yv:f32
    stm y[i], r, rest:f32
    i2 = add i, w
    jmp vloop(i2)
  done:
    ret
```

The same source runs 128-bit vectors on Cranelift, 256 or 512 bits on AVX2
and AVX-512, and the CPU's own length on SVE. AIR also has integer and float
scalars, f16 and bf16 storage, int8 dot products, matrix multiplies (`mm`),
functions with several results, inline functions, and a standard library
written in AIR:

- [The AIR language](docs/language.md): syntax, types, vector operations, the
  standard library.
- [Parallel loops](docs/parallel.md): `par n, f(args)`, one line for all
  cores.
- [Backends](docs/backends.md): Cranelift and LLVM, vector widths, matrix
  engines.
- [Command line](docs/cli.md), [MCP server](docs/mcp.md),
  [Python](docs/python.md), [Calling C](docs/ffi.md).
- [Kernels](docs/kernels.md) and [Llama in one call](docs/llama.md), with
  their benchmarks.

## How it works

1. **The parser and validator** check every function: types, SSA (each use
   dominated by its definition), block arguments, terminators. A failure is a
   `Diagnostic` with an `ERR_*` code, a span and a JSON context, made for a
   model to act on.
2. **The optimizer** folds constants, simplifies, removes dead code and
   expands `inline fn` calls into their callers, before either backend runs.
3. **Two backends, one semantics.** Cranelift compiles fast, with 128-bit
   vectors. LLVM 22 compiles in milliseconds and uses AVX2, AVX-512, SVE,
   VNNI, AMX and SME. Both give the same results, which the test suite checks
   on every operation, lane type and ISA level, and on Arm at SVE vector lengths
   up to 2048 bits (under QEMU).
4. **The runtime** keeps each function's fuel counter in a register (a
   decrement and a compare per branch, compiled in only under a fuel budget,
   a memory quota or the sandbox, and paid once on entry by loops with a
   known trip count), checks memory in sandboxed runs,
   and runs `par` loops on a lock-free pool of threads.
5. **AIR programs build on AIR:** the quantized matrix kernels, attention and
   softmax helpers are a standard library written in AIR, which a module
   imports with `use`.

## How it compares

- **Asking the agent for C and calling a compiler** needs a toolchain and a
  build on the machine, returns errors as prose, and runs the result
  unsandboxed. Vectorized C targets one ISA at a time with intrinsics, or
  depends on the auto-vectorizer.
- **WebAssembly** runtimes such as Wasmtime also compile with Cranelift, in a
  sandbox. Wasm is a target for other languages, with 128-bit SIMD. AIR is
  written by the model itself, and on LLVM uses the CPU's full vector width,
  int8 dot products and matrix engines, with parallel loops built in.
- **Python JITs** (Numba) and new languages (Mojo) are written by people,
  with nesting and implicit types, and run unsandboxed. AIR is lower level:
  longer, but each line is one operation that the validator can check and
  point at.

## FAQ

**Is AIR shorter than Python or C?** No. A loop takes several times the tokens
of the same loop in Python (in the Llama 3 tokenizer, a scalar dot product is
37 tokens of Python, 59 of C and 130 of AIR, in
[examples/dot_product.air](examples/dot_product.air)). AIR trades that for
programs a model gets right with nothing to balance, errors it can act on, and
native code a millisecond later.

**Should people write AIR?** You can: it's regular and small. But it's meant
to be written by models, and read by people when they need to.

**Is the sandbox a security boundary?** It's built to contain a model's
mistakes: out-of-bounds accesses, runaway loops, unbounded recursion, bad
`free`s. It hasn't been audited against hostile code, so run untrusted code
in a container or VM too.

**Which CPUs and systems?** x86-64 (SSE through AVX-512 and AMX) and AArch64
(NEON, SVE, SME), on Linux, macOS and Windows. The LLVM backend needs LLVM 22
at build time; without it, everything runs on Cranelift.

**GPUs?** No: achainsaw generates CPU code.

## Roadmap and contributing

[ROADMAP.md](ROADMAP.md) lists what's planned: faster K-quant and long-context
inference, more model architectures, and release packages. Programs that
miscompile or diagnostics that mislead a model are the most useful reports.
See [CONTRIBUTING.md](CONTRIBUTING.md) for how to build, test and send
changes.

If achainsaw is useful to you, a star helps others find it.

achainsaw is licensed under the [Apache License, Version 2.0](LICENSE).
