# achainsaw

[![CI](https://github.com/javier-cabezas/achainsaw/actions/workflows/ci.yml/badge.svg)](https://github.com/javier-cabezas/achainsaw/actions/workflows/ci.yml)
[![Release](https://github.com/javier-cabezas/achainsaw/actions/workflows/release.yml/badge.svg)](https://github.com/javier-cabezas/achainsaw/actions/workflows/release.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

> **High-Performance, Token-Minimal Compiler Toolchain Designed Exclusively for AI Agents**

`achainsaw` is a specialized compiler and JIT execution toolchain built from first principles for autonomous LLM agents. It completely discards human-centric syntactic sugar (no curly braces, no indentation sensitivity, no verbose keywords, no English prose compiler errors) to optimize for two uncompromising objectives:

1. **Minimize Agent Processing Time:** Autoregressive LLM decoding latency scales linearly with token count. `achainsaw` uses **AIR (Agent Intermediate Representation)**—a flat, sequential Single Static Assignment (SSA) format with single-token mnemonics. It eliminates nesting hallucinations and reduces token consumption by **60% to 80%** compared to traditional languages.
2. **Generate Highly Efficient Native Code as Fast as Possible:** Powered by **Cranelift** (the code generator behind Wasmtime), `achainsaw` compiles AIR modules directly to bare-metal x86_64 / AArch64 machine code with JIT compilation times under **10 milliseconds** and near-LLVM execution throughput.

---

## ⚡ Key Highlights

- **Linear SSA / Three-Address Code (TAC):** Zero nested expressions (`(+ (* a b) c)`). Eliminates bracket/parenthesis balancing errors and allows causal transformer attention heads to track dependencies with forward sequential ease.
- **Single-Token Mnemonics:** Operators and keywords (`fn`, `cst`, `add`, `mul`, `ld`, `st`, `br`, `jmp`, `ret`) map strictly to single, indivisible tokens in standard BPE vocabularies (OpenAI `cl100k`/`o200k`, Meta LLaMA 3, Google Gemini).
- **Machine-Native Diagnostic Protocol:** Zero natural-language prose error messages. Validation failures immediately produce structured JSON diagnostics with exact instruction indices, expected types, and candidate replacement patches for single-shot agent self-repair.
- **Width-Generic SIMD Vectors:** Fixed `v128`/`v256`/`v512` and scalable `vx` vectors with lane-typed ops (`vadd a, b:f32`) over i8/i16/i32/i64/f32/f64 lanes, fused multiply-add, compares and bitwise select, broadcast, lane extraction, and deterministic horizontal reductions (see [Vector Types & Ops](#-vector-types--ops-air-v2)).
- **One-Line Multi-Core Parallelism:** `par n, f(a, b)` runs `f(i, a, b)` for every `i < n` on all cores and joins, with no threads, locks or closures to get wrong (see [Parallel Loops](#-parallel-loops-par)).
- **Autonomous Dynamic Memory Management:** Built-in bare-metal heap allocator intrinsics (`alloc`, `free`) with 64-bit pointer arithmetic, allowing agents to dynamically allocate, reshape, and reclaim scratchpad buffers.
- **Sub-10ms Bare-Metal JIT:** Direct in-memory compilation and execution via Cranelift with native C-ABI compatibility for host interop.
- **High-Performance Memory Model:** Flat linear memory addressing (`ld`/`st`), 64-bit pointer arithmetic, and scalar numeric types (`i8`, `i16`, `i32`, `i64`, `f32`, `f64`, `ptr`, `v128`).

---

## 📐 Token Efficiency: AIR vs Traditional Code

### Vector Dot Product Kernel

#### Python (~68 tokens, slow runtime):
```python
def dot(a, b, n):
    acc = 0.0
    for i in range(n):
        acc += a[i] * b[i]
    return acc
```

#### C99 (~95 tokens, slow compilation):
```c
float dot(const float* a, const float* b, int n) {
    float acc = 0.0f;
    for (int i = 0; i < n; i++) {
        acc += a[i] * b[i];
    }
    return acc;
}
```

#### achainsaw AIR (~42 tokens, <5ms JIT compilation, native AVX speed):
```air
fn dot(p0:ptr, p1:ptr, n:i64)->f32
  b0:
    zero_f = cst 0.0:f32
    zero_i = cst 0:i64
    jmp b1(zero_i, zero_f)
  b1(i:i64, acc:f32):
    cond = lt i, n
    br cond, b2, b3
  b2:
    four = cst 4:i64
    byte_off = mul i, four
    p0_elem = add p0, byte_off
    p1_elem = add p1, byte_off
    v0 = ld p0_elem:f32
    v1 = ld p1_elem:f32
    prod = mul v0, v1
    next_acc = add acc, prod
    one = cst 1:i64
    next_i = add i, one
    jmp b1(next_i, next_acc)
  b3:
    ret acc
```

---

## 🧮 Vector Types & Ops

Vectors are untyped bit containers; every vector op names the lane type it works on, so one register can be read as `f32` lanes by one op and as `i32` lanes by the next.

| Type | Width | Notes |
|---|---|---|
| `v128`, `v256`, `v512` | 128/256/512 bits | Fixed width; usable in AIR function signatures |
| `vx` | Target maximum (at least 128 bits) | Scalable; `n = vl f32` gives its lane count. Usable in AIR function signatures, not in `extfn` ones |

| Op | Syntax | Lane types |
|---|---|---|
| Load / store | `v = ld p:v256`, `st p, v` | Any alignment |
| Masked load / store | `v = ldm p:vx, n:f32`, `stm p, v, n:f32` touch only the first `n` lanes (a masked load zeroes the rest) | Any |
| Broadcast | `v = splat x:v512` (the vector type is required) | i8 to f64 |
| Arithmetic | `r = vadd a, b:f32` (`vsub`, `vmul`, `vdiv`, `vmin`, `vmax`) | `vmul`: not i8; `vdiv`: f32/f64; `vmin`/`vmax`: not i64 |
| Bitwise | `r = vand a, b:i32` (`vor`, `vxor`) | Any |
| Fused multiply-add | `r = vfma a, b, c:f32` computes `a*b + c` with one rounding | f32, f64 |
| Compare | `m = vlt a, b:f32` (`veq`, `vne`, `vgt`, `vle`, `vge`) gives all-ones lanes where true | Any (signed for integers) |
| Select | `r = vsel m, a, b` takes bits of `a` where `m` is 1, else `b` | n/a |
| Reduce | `s = vsum v:f32` (`vmaxr`, `vminr`) | Any |
| Convert | `f = vitof v:f32` (i32 lanes to f32), `i = vftoi v:i32` (f32 to i32, saturating, NaN to 0); like scalar casts, the suffix is the result lane type | f32, i32 |
| Widen / narrow | `w = vwidenlo v:i16` / `vwidenhi` sign-extend the low / high half of the half-width lanes; `n = vnarrow a, b:i8` saturates both operands' lanes, `a`'s into the low half and `b`'s into the high half | widen: i16, i32, i64; narrow: i8, i16 |
| Shift | `r = vshl v, s:i32` (`vshr` arithmetic, `vushr` logical) by a scalar amount, taken modulo the lane width | i8 to i64 |
| Exponential | `e = vexp v:f32`: e^x within 2 ulp on [-87, 88], inputs clamped to that range, NaN propagated; a fixed algorithm, so results are bit-identical on every backend | f32 |
| Extract | `e = extlane v, 7:f32` | Index checked against the width (`vx`: its guaranteed 128 bits) |
| Lane count | `n = vl f32` | Any |

Semantics are identical on every backend: integer ops wrap, float `vmin`/`vmax` propagate NaN and order `-0.0` below `+0.0`, and reductions use a fixed recursive-halves order (`reduce(v) = op(reduce(lo), reduce(hi))`), so float sums are bit-reproducible for fixed widths.

**Fast min/max:** a function declared `fn softmax(x:ptr, out:ptr, n:i64)->f32 fast` computes its float min/max (`min`, `max`, `vmin`, `vmax`, `vminr`, `vmaxr`) by compare and select, `max(a, b) = a > b ? a : b` and `min(a, b) = a < b ? a : b`, so a NaN operand or two zeros give `b`. That skips the NaN and signed-zero handling, which costs several instructions per op on 128-bit Cranelift vectors, and results stay identical on every backend (they are what x86's `maxps`/`minps` compute). Softmax runs 17% faster on Cranelift and 13% faster on LLVM with it.

Vector-length-agnostic code processes `vl` lanes per iteration and finishes with masked accesses, as in [`examples/saxpy_vx.air`](examples/saxpy_vx.air):
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
    off = mul i, 4:i64
    px = add x, off
    py = add y, off
    xv = ldm px:vx, rest:f32
    yv = ldm py:vx, rest:f32
    r = vfma va, xv, yv:f32
    stm py, r, rest:f32
    i2 = add i, w
    jmp vloop(i2)
  done:
    ret
```

**Half-precision storage types:** `f16` (IEEE binary16) and `bf16` (bfloat16) can be loaded, stored, and converted (`f = fext h:f32`, `h = ftrunc f:bf16`, rounding to nearest-even), but not used in arithmetic or function signatures; convert to `f32` first.

**Matrix multiply:** `mm pc, pa, pb, m, n, k:bf16` computes `C[m x n] += A[m x k] * B[k x n]` on row-major, contiguous matrices. `A`/`B` hold `bf16`, `f16`, or `f32` elements with an `f32` `C`, or `i8` elements with an `i32` `C` (exact, wrapping). Float accumulation order is implementation-defined, so results agree across backends within rounding error rather than bit for bit. Under a fuel budget, `mm` costs one unit per 1024 multiply-adds, charged before it starts.

On the LLVM backend, `mm` uses the best kernel the target has:

| Target | `mm` kernel |
|---|---|
| x86 with AMX (Sapphire/Emerald Rapids: bf16, i8; Granite Rapids: also f16) | 16x16 AMX tiles (`tdpbf16ps`, `tdpbssd`, `tdpfp16ps`) |
| AArch64 with SME (e.g. Apple M4) | ZA-tile outer products (`fmopa`, `bfmopa`, `smopa`) in streaming mode |
| Everything else | broadcast-FMA at full `vx` width (AVX-512/AVX2/SVE/NEON), 4 rows of C per B strip |

On a Zen 4 core (AVX-512), 256x256x256 `mm` runs at about 130 GFLOP/s for f32, 90 for bf16, 70 GOP/s for i8, and 29 for f16. Cranelift's 128-bit kernel reaches about 50, 32, 33 and 9. The AMX path is compiled and checked in assembly but has not run on AMX hardware yet. The SME path is tested under QEMU (`tools/qemu-aarch64/run.sh`); its objects call the SME ABI routine `__arm_tpidr2_save`, provided by GCC 14+ libgcc or compiler-rt.

**Backend support:** the default Cranelift backend runs every vector program, splitting `v256`/`v512` into 128-bit operations, treating `vx` as 128 bits, and running `mm` as 128-bit tiles (4 rows by 8 columns of C, with FMA where the CPU has it). The opt-in [LLVM backend](#10-choosing-a-backend---backend) uses the hardware's full width:

| Target (LLVM backend) | `vx` | Masked `ldm`/`stm` |
|---|---|---|
| x86_64 with AVX-512F | 512 bits (zmm) | AVX-512 k-masks |
| x86_64 with AVX2 | 256 bits (ymm) | `vmaskmov` |
| x86_64 SSE/AVX only, AArch64 NEON (including Apple M4) | 128 bits | per-lane |
| AArch64 with SVE | scalable, the CPU's vector length (`<vscale x ...>`) | `whilelo` predicates |

`v256`/`v512` map to native registers whenever the target has them. On SVE, horizontal reductions follow the same adjacent-pairs tree at the run-time vector length. Programs written against `vl` and `mm` speed up without changes. `vx` width therefore depends on the backend and CPU, so programs must use `vl` rather than assume a lane count; `JitEngine::vx_bits()` and `achainsaw cpu` report it.

---

## 🧵 Parallel Loops (`par`)

`par n, f(a, b)` calls `f(i, a, b)` for every `i` in `0..n` across all cores and returns when every call has finished. It is the only parallel construct: no threads, futures, locks or closures. `f` is an ordinary AIR function whose first parameter is the `i64` index, followed by scalar or `ptr` parameters, and that returns nothing. Values the body needs are passed as arguments, exactly like `call`. From [`examples/kernels/gemv_par.air`](examples/kernels/gemv_par.air):

```air
fn gemv_par(a:ptr, x:ptr, y:ptr, m:i64, k:i64)
  b0:
    m15 = add m, 15:i64
    blocks = udiv m15, 16:i64
    par blocks, gemv_rows(a, x, y, m, k)    # gemv_rows(blk, a, x, y, m, k) does rows 16*blk..16*blk+15
    ret
```

The rules:
- **Write only your own memory.** Calls run concurrently and in any order. Each `i` writes its own memory; to reduce, store per-`i` partial results in an `alloc`'d array and sum them after the `par`.
- **Give each `i` real work.** A row or a block of thousands of elements works well. Dispatching a `par` costs a few microseconds.
- **The validator checks the body.** `ERR_PAR_SIGNATURE` reports a body with the wrong signature and gives the expected one in `context.expected_signature`. External functions (`extfn`) cannot be bodies, and vectors cannot be passed (pass them through memory).

How it runs:
- **Threads.** The calling thread works through the indices together with a process-wide pool of helper threads. Indices are handed out dynamically, so uneven iterations still balance. The pool has `ACHAINSAW_THREADS` threads in all (default: one per core). Helpers that just ran iterations spin for 200 µs before parking, so back-to-back `par` loops do not pay thread wake-ups; the window is short because spinning threads slow down other runtimes' thread pools (OpenBLAS, OpenMP) running in the same process. A `par` with fewer indices than threads uses only that many threads: the other helpers stay parked, so they neither slow down busy threads sharing their core nor need waking for later loops of the same size.
- **Thread cap.** Limit a run with `--threads` on `achainsaw run`, `threads` in MCP `air_run`, `Kernel.set_threads()` in Python, or `JitEngine::set_threads`. `1` runs serially, which makes it easy to measure the speedup.
- **Serial fallbacks.** A `par` inside a `par` body runs serially on its worker. So does a `par` started while another thread's `par` has the pool. AOT objects have no runtime, so `par` compiles to a plain loop there; any order is a valid execution.
- **Fuel.** The fuel budget is shared: each index costs one unit, plus the usual unit per branch. A parallel run therefore uses exactly the fuel of a serial one, and a runaway iteration still ends with `ERR_OUT_OF_FUEL`.
- **Errors and sandbox.** The first failure in any worker stops all of them and is reported from the `par`: `ERR_MEMORY_VIOLATION`, `ERR_STACK_OVERFLOW` or `ERR_OUT_OF_MEMORY`. Sandboxed bodies share the caller's arena and may `alloc`/`free` scratch memory.
- **Python.** `Kernel.run` releases the GIL, so host callbacks can run on `par` workers, and other Python threads keep running.

Speedup of `gemv_par`'s `bench` driver on a Ryzen 7 8845HS (8 cores, 16 threads), from `achainsaw run examples/kernels/gemv_par.air --func bench --args M,K,REPS --threads T`:

| Shape | Backend | 1 thread | 16 threads | Speedup |
|---|---|---|---|---|
| 4096x4096 (64 MB), 50 runs | Cranelift | 632 ms | 75 ms | 8.4x |
| 4096x4096 (64 MB), 50 runs | LLVM (AVX-512) | 180 ms | 74 ms (8 threads) | 2.4x (memory-bound) |
| 1024x512 (2 MB), 2000 runs | Cranelift | 685 ms | 83 ms | 8.3x |
| 1024x512 (2 MB), 2000 runs | LLVM (AVX-512) | 81 ms | 27 ms | 3.1x (2000 `par` dispatches) |

---

## 🛠️ CLI & Agent Usage

### 1. Validate Syntax & SSA Invariants (`--json`)
```bash
achainsaw check examples/sum_loop.air --json
```

**Output:**
```json
{
  "status": "ok",
  "function_count": 1,
  "functions": ["sum_to_n"],
  "block_count": 4,
  "instruction_count": 7
}
```

### 2. Machine-Native Diagnostic on Errors
If an agent hallucinates an undefined register:
```json
{
  "status": "error",
  "error_code": "ERR_UNDEFINED_REG",
  "instruction_index": 0,
  "message": "Register 'undefined_var' is used before definition",
  "span": {
    "start": 42,
    "end": 55,
    "line": 4,
    "column": 16
  },
  "context": {
    "target": "undefined_var",
    "available_registers": ["a", "b", "cst0"]
  }
}
```

### 3. Compile and Run via Cranelift JIT
```bash
achainsaw run examples/sum_loop.air --func sum_to_n --args 100 --json
```

**Output:**
```json
{
  "status": "ok",
  "function": "sum_to_n",
  "result": 5050,
  "parse_time_us": 62,
  "compile_time_us": 3120,
  "exec_time_us": 1,
  "total_time_ms": 3.25
}
```

### 4. Benchmark JIT Compilation Latency
```bash
achainsaw bench examples/fibonacci.air --iters 200
```

### 5. Assemble & Disassemble Compact Binary Bytecode (`.airb`)
AIR modules can be assembled into compact binary bytecode for persistent caching, agent-to-agent IPC, and zero-parse reloading. Every AIR construct round-trips through AIRB, and files carry a format version that decoders check:
```bash
# Assemble text AIR to compact binary bytecode (.airb)
achainsaw assemble examples/fibonacci.air -o examples/fibonacci.airb --json

# Direct execution of pre-parsed binary bytecode
achainsaw run examples/fibonacci.airb --func fib --args 30 --json

# Disassemble binary bytecode back into canonical text AIR
achainsaw disassemble examples/fibonacci.airb
```

### 6. Model Context Protocol (MCP) Server (`achainsaw mcp`)
`achainsaw` includes a native Model Context Protocol (MCP) server communicating over JSON-RPC 2.0 stdio, exposing compiler tools directly to LLM agents and IDE assistants:
```bash
achainsaw mcp
```

**Exposed MCP Tools:**
- **`air_check`**: Validates AIR textual IR or base64 AIRB bytecode syntax and SSA invariants. Returns structured diagnostic metrics or error payloads with line/column pointers and self-repair hints.
- **`air_run`**: JIT compiles and executes AIR functions with arguments, loop fuel budget, and memory quota sandboxing. Code runs in a sandbox: memory comes only from `alloc` within a private arena (`max_memory_mb`, default 64), out-of-bounds accesses and bad `free`s fail with `ERR_MEMORY_VIOLATION`, unbounded recursion with `ERR_STACK_OVERFLOW`, and pointer parameters are rejected.
- **`air_assemble`**: Assembles textual AIR into compact base64-encoded AIRB bytecode with compression metrics.
- **`air_disassemble`**: Decompiles base64 AIRB bytecode back into canonical, human/agent-readable textual AIR.
- **`air_optimize`**: Optimizes IR using constant folding, algebraic simplification, branch folding, and DCE to a fixpoint.
- **`air_target`**: Reports the host's vector features (AVX/AVX2/AVX-512/AMX, NEON/SVE/SME), the active ISA cap, and each backend's vector width (see [CPU Features & ISA Targeting](#9-cpu-features--isa-targeting-achainsaw-cpu)).

The `initialize` response includes server `instructions`, a compact AIR syntax primer that MCP clients inject into the model's context. That lets an agent write valid AIR on the first attempt without this README.

#### Using with Claude Code
The repository ships a project-scoped [`.mcp.json`](.mcp.json) that runs the server via `cargo run --release`. Build once so the first startup does not exceed the MCP connection timeout, then start Claude Code in the repo and approve the `achainsaw` server when prompted:
```bash
cargo build --release -p achainsaw
claude
```

To make the tools available in every project, register an installed binary at user scope instead:
```bash
cargo install --path crates/achainsaw-cli
claude mcp add --scope user achainsaw -- achainsaw mcp
```

#### Using with Claude Desktop
Add the server to `claude_desktop_config.json` (macOS: `~/Library/Application Support/Claude/`, Windows: `%APPDATA%\Claude\`), using the absolute path to the binary, then restart Claude Desktop:
```json
{
  "mcpServers": {
    "achainsaw": {
      "command": "/absolute/path/to/achainsaw",
      "args": ["mcp"]
    }
  }
}
```

### 7. IR Optimization Engine (`achainsaw opt`)
Pre-evaluate constant expressions, simplify algebraic identities (`x + 0 -> x`, `x * 1 -> x`), fold invariant branches, and eliminate dead code and unreachable blocks:
```bash
achainsaw opt examples/opt_demo.air --json
```

**Optimization Telemetry:**
```json
{
  "status": "ok",
  "constants_folded": 2,
  "algebraic_simplifications": 1,
  "branches_folded": 0,
  "dead_instructions_removed": 6,
  "dead_blocks_removed": 1,
  "total_optimizations": 10,
  "iterations": 2,
  "code": "fn demo(x:i32)->i32\n  b0:\n    ret x\n"
}
```

### 8. Ahead-Of-Time (AOT) Compilation (`achainsaw build`)
Compile AIR modules into native object files (`.o`) or linked shared libraries (`.so` / `.dll`) for direct embedding into C, C++, Rust, or Python programs via standard C ABI:
```bash
# Compile to native object file (.o)
achainsaw build examples/fibonacci.air -o examples/fibonacci.o --json

# Compile and link directly into shared library (.dll / .so)
achainsaw build examples/fibonacci.air --shared --json
```

**Build Telemetry:**
```json
{
  "status": "ok",
  "input": "examples/fibonacci.air",
  "output": "examples/fibonacci.dll",
  "object_bytes": 265,
  "shared": true,
  "parse_time_us": 304,
  "compile_time_us": 2563,
  "link_time_us": 112751,
  "total_time_ms": 116.0
}
```

### 9. CPU Features & ISA Targeting (`achainsaw cpu`)
Report the host's vector ISA features (feature names use LLVM spelling), the active ISA cap, and the vector width each code generation backend uses:
```bash
achainsaw cpu
```

```json
{
  "status": "ok",
  "host": {
    "arch": "x86_64",
    "features": ["sse3", "ssse3", "sse4.1", "sse4.2", "popcnt", "cx16", "avx", "f16c", "avx2", "fma", "...", "avx512f", "avx512vl", "avx512bw", "avx512bf16"],
    "max_isa": "avx512",
    "native_vector_bits": 512,
    "sve_vector_bits": null,
    "sme_vector_bits": null
  },
  "isa_cap": null,
  "effective": { "...": "host features after the ISA cap" },
  "backends": {
    "cranelift": { "available": true, "vector_bits": 128, "unused_features": ["f16c", "avx512bw", "avx512cd", "avx512bf16"] },
    "llvm": { "available": false }
  }
}
```

Detection covers SSE through AVX-512 (including BF16/FP16/VNNI) and AMX on x86_64, and NEON, SVE/SVE2 (with vector length), and SME/SME2 on AArch64. Cranelift emits 128-bit vector code (using VEX/EVEX encodings when AVX/AVX-512 are available) and runs `v256`/`v512`/`vx` programs as 128-bit operations. The LLVM backend, when built in, uses every enabled feature, and `backends.llvm` reports `"available": true` with its `vx` width (`vector_bits`) and whether `vx` is scalable (`vx_scalable`, on SVE).

**Cap the ISA level** to exercise lower tiers on a more capable machine (for example, AVX2 code on an AVX-512 host). Levels: `sse`, `avx`, `avx2`, `avx512`, `amx` (x86_64) and `neon`, `sve`, `sve2`, `sme` (AArch64):
```bash
achainsaw run examples/sum_loop.air --func sum_to_n --args 100 --isa avx2
ACHAINSAW_MAX_ISA=sse achainsaw mcp      # every JIT compilation in the server is capped
```

**Target a specific CPU in AOT builds** with LLVM CPU names and feature strings. `+feature` also enables its prerequisites and `-feature` disables everything that depends on it, as in LLVM. Features the backend cannot use are listed in `ignored_features`:
```bash
achainsaw build examples/kernels/gemv_f32.air --target-cpu x86-64-v3
achainsaw build examples/kernels/gemv_f32.air --target-cpu znver4 --target-features -avx512f
achainsaw --backend llvm build examples/kernels/gemv_f32.air --target-cpu sapphirerapids --emit asm   # writes gemv_f32.s
```

---

## ⚡ Chainsaw-BLAS: High-Performance Agent AI Kernel Library

`examples/kernels/` holds verified kernels for LLM inference, vector search, and normalization. They are vector-length agnostic: each processes `vl` lanes per step with FMAs and finishes with masked `ldm`/`stm`, so the same source runs 128-bit vectors on Cranelift and AVX2/AVX-512/SVE-width vectors on LLVM. Every kernel takes element counts, and each `.air` file has matching `.airb` bytecode.

| Kernel | Signature | Use case |
|---|---|---|
| `cosine_similarity.air` | `(a:ptr, b:ptr, n:i64)->f32` | Embedding search and RAG |
| `euclidean_distance.air` | `(a:ptr, b:ptr, n:i64)->f32` | Nearest neighbors, vector quantization |
| `softmax.air` | `(x:ptr, out:ptr, n:i64)->f32` (returns the sum of exponentials) | Attention weights, numerically stable; `exp` vectorized in AIR (no libm call) |
| `rmsnorm.air` | `(x:ptr, w:ptr, out:ptr, n:i64)->f32` (returns the scale) | Token normalization (LLaMA, Mistral, Gemma) |
| `gemv_f32.air` | `(a:ptr, x:ptr, y:ptr, m:i64, k:i64)` | Matrix-vector projection, 4 rows at a time (independent FMA chains, x loaded once per 4 rows) |
| `gemv_par.air` | `gemv_par(a:ptr, x:ptr, y:ptr, m:i64, k:i64)` | The same on all cores, blocks of 16 rows per `par` index; `bench(m, k, reps)->f32` runs it from the CLI or MCP |
| `gemm_bf16.air` | `gemm_bf16(c:ptr, a:ptr, b:ptr, m:i64, n:i64, k:i64)` | bf16 matrix multiply into f32 on all cores: `par` over 16-row blocks, each an `mm` (AMX, SME or FMA) |
| `flash_attention.air` | `(q:ptr, kv:ptr, idx:ptr, sink:ptr, out:ptr, h:i64, d:i64, nk:i64, scale:f32)` | One decode step of sparse multi-query attention with an attention sink, as in DeepSeek V4 Pro (128 heads, 512-dim shared K=V entries, 1152 selected entries). Runs on all cores: one `par` gathers the selected entries, a second runs FlashAttention-2 (blocks of 64 entries, both products on `mm`) for groups of 8 heads |
| `add_rmsnorm.air` | `(x:ptr, res:ptr, w:ptr, out:ptr, n:i64, eps:f32)->f32` | Residual add fused with RMSNorm between sublayers (vLLM's `fused_add_rms_norm`): updates the residual stream and normalizes it in two passes |
| `rope.air` | `(x:ptr, heads:i64, dim:i64, cos:ptr, sin:ptr)` | Rotary position embedding in place, LLaMA/NeoX rotate-half layout, from a row of the cos/sin cache |
| `swiglu.air` | `(gate:ptr, up:ptr, out:ptr, n:i64)` | `silu(gate) * up` with `exp` vectorized in AIR (Cephes polynomial; 2^n built by reading the f32 lanes as i32), no libm call |
| `argmax.air` | `(x:ptr, n:i64)->i64` | Greedy decoding: first index of the largest logit, one pass with four independent (value, index) accumulator sets |
| `q8_gemv.air` | `q8_gemv(wq:ptr, scales:ptr, x:ptr, y:ptr, m:i64, k:i64)` | Q8_0 (llama.cpp) matrix-vector product on all cores: int8 weights in 32-blocks with f32 scales, packed in 64-row chunks; activations quantized on the fly; int8 dots on `mm`, converted to f32 by reading i32 lanes as f32 bits |
| `llama_decode.air` | `llama_decode(model:ptr, cfg:ptr, cache:ptr, token:i64, pos:i64, h:ptr, eps:f32)->i64` | A whole Llama-style decode step in one call, token in, next token out: see [Single-call decode](#single-call-decode-deep-fusion) |

`crates/achainsaw-codegen/tests/kernels.rs` checks every kernel against a scalar reference at each ISA level, including lengths that end in partial vectors. The benchmark verifies them against NumPy and times each backend and ISA level:

```bash
python benchmarks/benchmark_kernels.py                  # host ISA, every available backend
python benchmarks/benchmark_kernels.py --isa all        # also sweep sse/avx/avx2/avx512 (or neon/sve/...)
python benchmarks/benchmark_kernels.py --json out.json  # machine-readable results
```

Compared with NumPy on the same data types (bf16 inputs are stored as bf16 bits and widened in each call, Q8_0 weights stay int8; NumPy's GEMV/GEMM use multithreaded OpenBLAS), from `benchmarks/benchmark_kernels.py` on a Ryzen 7 8845HS (8 cores, 16 threads). Kernels marked "all cores" use `par`; the others run on one core:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/kernels-vs-numpy-dark.svg">
  <img alt="Speedup of each kernel over NumPy on a log scale, Cranelift and LLVM side by side; the numbers are in the table below" src="docs/kernels-vs-numpy-light.svg">
</picture>

| Kernel | NumPy | Cranelift (128-bit) | LLVM (512-bit) |
|---|---|---|---|
| Q8_0 GEMV 4096x4096, all cores | 6.69 ms (int8 widened to f32 per call; NumPy has no int8 matmul) | 471 µs (14.2x) | 148 µs (45.1x) |
| RoPE, 32 heads x 128 | 7.7 µs | 2.0 µs (3.8x) | 0.80 µs (9.6x) |
| Residual add + RMSNorm, n=4096 | 6.3 µs | 4.2 µs (1.5x) | 1.5 µs (4.2x) |
| Cosine similarity, n=1024 | 2.3 µs | 0.70 µs (3.3x) | 0.54 µs (4.3x) |
| RMSNorm, n=4096 | 5.6 µs | 3.9 µs (1.4x) | 1.4 µs (3.9x) |
| Softmax, n=1000 | 2.9 µs | 1.8 µs (1.6x) | 0.93 µs (3.2x) |
| Euclidean distance, n=1024 | 1.4 µs | 0.67 µs (2.1x) | 0.54 µs (2.6x) |
| GEMM bf16 -> f32, 256³, all cores | 172 µs | 251 µs (0.69x) | 76 µs (2.3x) |
| SwiGLU, n=14336 | 11.2 µs | 14.7 µs (0.76x) | 5.0 µs (2.2x) |
| Greedy argmax, vocabulary 128256 | 6.9 µs | 16.2 µs (0.43x) | 4.9 µs (1.4x) |
| Flash attention decode, DeepSeek V4 Pro, all cores | 1.42 ms | 2.20 ms (0.65x) | 1.06 ms (1.3x) |
| GEMV f32 512x1024, all cores | 6.2 µs | 17.7 µs (0.35x) | 9.6 µs (0.64x) |
| GEMV f32 512x1024, 1 core | 6.0 µs (all cores) | 39.8 µs (0.15x) | 21.8 µs (0.27x) |

`python benchmarks/plot_vs_numpy.py results.json docs/kernels-vs-numpy` redraws the figure from `benchmark_kernels.py --json results.json`.

Where the remaining gaps come from:
- **Cranelift's vector width.** Cranelift has only 128-bit vectors, so vector-bound kernels (SwiGLU, argmax, and `mm` in GEMM and attention) do a quarter of the work per instruction of AVX-512 code; on LLVM the same sources beat NumPy.
- **f32 GEMV against multithreaded BLAS.** The 2 MB matrix is cache-resident across calls: OpenBLAS splits it statically, so each core finds its rows in its own L2, while `par` hands rows out dynamically (better under uneven work, worse for this cache reuse) and adds about 2.7 µs of dispatch at 16 threads. The single-core row compares one core with NumPy's 16 threads. With weights streaming from memory, as in LLM decode, the Q8_0 GEMV is 35x faster than NumPy.

Fuel checks are inline (a decrement and a compare per branch, on a counter kept in a register), so loops pay almost nothing for runaway protection. The rare slow path calls the runtime through a stub that preserves every register, so it does not make Cranelift spill loop values: before that, the fuel checks made Cranelift's GEMV 2.6x slower.

#### Single-call decode (deep fusion)

[`examples/kernels/llama_decode.air`](examples/kernels/llama_decode.air) runs a whole decode step of a Llama-style model (RMSNorm, grouped-query attention with RoPE and a KV cache, SwiGLU MLP, Q8_0 weights) in one call: the token id goes in, the greedy next token comes out. Like GPU megakernels ([Hazy Research's Llama-1B megakernel](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles), [AutoMegaKernel](https://arxiv.org/pdf/2606.09682)), it removes the boundaries between the ~100 small kernels of a forward pass; on a CPU that means each elementwise op runs inside the pass that produces or consumes its data:

| Pass | Fused into it |
|---|---|
| RMSNorm (serial) | Q8 quantization of its output |
| QKV projection (`par`, one task per head) | RoPE, KV-cache append |
| Attention (`par`, one task per q head) | vectorized softmax `exp`, Q8 quantization of its output |
| O and down projections (`par`, 64 rows per task) | residual add |
| Gate and up projections (`par`, 64 rows per task) | SwiGLU, Q8 quantization |
| LM head (`par`, 64 rows per task) | argmax: each task keeps only its best logit, so the logits are never written |

That is 2 serial norms and 5 `par` regions per layer. Weights use `q8_gemv`'s chunked layout, so every task owns its output rows and applies its epilogue without a cross-task reduction. `crates/achainsaw-codegen/tests/llama_decode.rs` checks it against an f64 reference model over several positions, at every ISA level on both backends.

On a random model with Llama 3.2 1B's shapes (d 2048, 16 layers, 32 query heads over 8 KV heads of 64, MLP 8192, vocabulary 128256; 1.39 GB of Q8_0 weights) on a Ryzen 7 8845HS, from [`benchmarks/benchmark_decode.py`](benchmarks/benchmark_decode.py):

| | ms/token | tokens/s | weights read |
|---|---|---|---|
| AIR on LLVM, one call per token | 26.5 | 38 | 52 GB/s |
| AIR on Cranelift, one call per token | 31.0 | 32 | 45 GB/s |
| NumPy, same data types (int8 weights and activations, f32 elsewhere) | 781 | 1.3 | 1.8 GB/s |

Both implementations share the weights and the arithmetic (exact int8 block dots, f32 for the embedding, norms, RoPE, KV cache and attention), and all 32 greedy tokens match. Decode is bound by memory bandwidth: the kernel streams each int8 weight once, at 52 GB/s, with no intermediate tensors, while NumPy, lacking an int8 matrix product, widens every weight to f32 on each token. (Dequantizing the weights to f32 ahead of time, at 4x the memory, brings NumPy to about 114 ms/token.)

### 10. Choosing a Backend (`--backend`)
Two code generators share one runtime, so fuel budgets, memory quotas, the MCP sandbox, and the results of scalar and fixed-width vector code are the same on both (`vx` code computes the same values, but `vl` can be larger on LLVM):

| Backend | Compile latency (release, simple kernel) | Code | Availability |
|---|---|---|---|
| `cranelift` | ~0.2–0.3 ms | 128-bit vectors | always |
| `llvm` | ~5–7 ms | fully optimized; full-width AVX2/AVX-512/SVE vectors (`vx` up to 512 bits or scalable) | builds with the `llvm` feature |

Pick one with `--backend` on `run`, `bench` and `build`, the `ACHAINSAW_BACKEND` environment variable, `backend=` in Python, or the `backend` argument of MCP `air_run`. `auto` (the default) uses LLVM, when it is built in, for modules that use `v256`, `v512`, `vx`/`vl` or `mm`, and Cranelift otherwise, so scalar code keeps sub-millisecond compiles. For example, [`examples/saxpy_vx.air`](examples/saxpy_vx.air) on an AVX-512 machine runs 3–4x faster on LLVM.

```bash
# Build with the LLVM backend (needs LLVM 22, e.g. apt install llvm-22-dev)
export LLVM_SYS_221_PREFIX=/usr/lib/llvm-22
cargo build --release -p achainsaw --features llvm

achainsaw --backend llvm run examples/fibonacci.air --func fib --args 30
achainsaw --backend llvm build examples/kernels/gemv_f32.air --target-cpu znver4 --shared
ACHAINSAW_BACKEND=llvm achainsaw mcp     # every air_run in the server uses LLVM
```

AOT builds with `--backend llvm` accept the same `--target`, `--target-cpu` and `--target-features` options and can use every feature (nothing ends up in `ignored_features`). `crates/achainsaw-codegen/tests/backend_parity.rs` checks that both backends agree bit for bit, including on division by zero, out-of-range shifts, NaN handling, and saturating conversions.

---

## 🐍 Python Host Integration (`achainsaw-py`)

For AI agent orchestrators (LangGraph, AutoGen, CrewAI, DSPy), `achainsaw` provides native in-process Python bindings via PyO3 with **zero-copy NumPy memory integration** and microsecond compilation throughput.

### Installation / Build
```bash
pip install .                                     # or, with the LLVM backend:
MATURIN_PEP517_ARGS="--features llvm" pip install .
# then: achainsaw.compile(src, backend="llvm"); achainsaw.available_backends()

cargo build --release -p achainsaw-py
cp target/release/achainsaw.dll achainsaw.pyd  # Windows
# or cp target/release/libachainsaw.so achainsaw.so  # Linux
```

### Microsecond JIT & Zero-Copy NumPy Execution
```python
import achainsaw
import numpy as np

# In-place 128-bit SIMD vector scaling kernel
kernel = achainsaw.compile("""
fn simd_scale(p:ptr, factor:f32, n:i64)
  b0:
    vfactor = splat factor:v128
    zero = cst 0:i64
    jmp b1(zero)
  b1(i:i64):
    cond = lt i, n
    br cond, b2, b3
  b2:
    sixteen = cst 16:i64
    off = mul i, sixteen
    elem_ptr = add p, off
    v = ld elem_ptr:v128
    vscaled = vmul v, vfactor:f32
    st elem_ptr, vscaled
    one = cst 1:i64
    next_i = add i, one
    jmp b1(next_i)
  b3:
    ret
""")

# Contiguous NumPy float32 array
arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32)

# Direct zero-copy mutation in native SIMD registers
kernel.run("simd_scale", arr, 2.5, 2)
print(arr)  # [ 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0 ]
```

### Autonomous Agent Self-Repair via Structured Diagnostics
```python
try:
    achainsaw.compile(bad_air_code)
except achainsaw.CompilationError as e:
    # Machine-native structured dictionary — zero regex or prose parsing needed
    diag = e.diagnostic
    print(diag["error_code"])                         # "ERR_UNDEFINED_REG"
    print(diag["context"]["target"])                  # "bad_register"
    print(diag["context"]["available_registers"])     # ["a", "b"]
    
    # Instant single-shot repair by LLM agent
    repaired_code = bad_air_code.replace(
        diag["context"]["target"],
        diag["context"]["available_registers"][0]
    )
    k = achainsaw.compile(repaired_code)
```

### CPU Features & ISA Cap in Python
```python
report = achainsaw.cpu_features()       # same report as `achainsaw cpu`
print(report["host"]["max_isa"])        # e.g. "avx512"

achainsaw.set_isa_cap("avx2")           # kernels compiled from now on use at most AVX2
kernel = achainsaw.compile(air_kernel)
achainsaw.set_isa_cap(None)             # remove the cap
```

### Binary Bytecode (AIRB) in Python
```python
# Assemble to compact binary payload (for caching or network transfer)
binary_payload = achainsaw.assemble(air_kernel)

# Directly compile pre-parsed binary bytecode (<0.05 ms)
kernel = achainsaw.compile_binary(binary_payload)
result = kernel.run("simd_scale", arr, 2.5, 2)

# Disassemble binary payload back into canonical text AIR
text_ir = achainsaw.disassemble(binary_payload)
```

### ⚡ Performance Comparison

| Metric | Subprocess CLI (`achainsaw.exe`) | In-Process Python (`achainsaw-py`) | Speedup |
| :--- | :--- | :--- | :--- |
| **JIT Compilation Latency** | ~7.95 ms / compile | **0.136 ms / compile** | **58.3x faster** |
| **Compilation Throughput** | ~125 compiles/sec | **7,340+ compiles/sec** | **58.3x throughput** |
| **Execution Call Overhead** | ~7.5 ms (process spawn) | **0.249 µs / call** | **>30,000x faster** |
| **Execution Call Rate** | ~130 calls/sec | **4,011,000+ calls/sec** | **4.0M calls/sec** |
| **NumPy Array Memory** | Serialization required | **Zero-copy (direct pointer)** | **0 MB allocated** |

---

## 🔗 Foreign Function Interface (FFI) & External Symbols

`achainsaw` can seamlessly call functions and access variables from external C/C++ libraries, Python extensions, and system shared objects (`.dll`, `.so`, `.dylib`).

### 1. Minimal `extfn` Syntax
External functions are declared with the token-minimal `extfn` keyword and invoked with standard `call`:
```air
extfn sinf(x:f32)->f32
extfn cosf(x:f32)->f32
extfn sqrtf(x:f32)->f32

fn compute_sincos_norm(x:f32)->f32
  b0:
    s = call sinf(x)
    c = call cosf(x)
    s2 = mul s, s
    c2 = mul c, c
    sum = add s2, c2
    norm = call sqrtf(sum)
    ret norm
```

### 2. Pre-Registered Standard C Math Intrinsics
The JIT engine pre-registers high-performance standard C math functions out of the box with zero runtime overhead:
- **32-bit float:** `sinf`, `cosf`, `tanf`, `sqrtf`, `expf`, `logf`, `powf`, `fabsf`, `floorf`, `ceilf`, `roundf`
- **64-bit float:** `sin`, `cos`, `tan`, `sqrt`, `exp`, `log`, `pow`, `fabs`, `floor`, `ceil`, `round`

### 3. Dynamic Shared Library Loading (`.dll` / `.so`)
Load any dynamic library at runtime. Exported C symbols are resolved automatically during JIT linking:
```python
import achainsaw

# Dynamically load a shared library
achainsaw.load_library("libcustom_math.so")  # or "custom_math.dll"

# Execute kernel calling symbols from the loaded library
kernel = achainsaw.compile("""
extfn custom_dsp_filter(input:ptr, output:ptr, len:i64)
fn process(a:ptr, b:ptr, n:i64)
  b0:
    call custom_dsp_filter(a, b, n)
    ret
""")
```

### 4. Custom Symbol Registration & Callbacks
Register any raw function pointer or host callback directly into `achainsaw`'s symbol table:
```python
import ctypes
import achainsaw

# Create a C callback from Python
callback_type = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32)
def custom_fma(a, b, c):
    return a * b + c
cb = callback_type(custom_fma)

# Register pointer in achainsaw
achainsaw.register_symbol("host_fma", ctypes.cast(cb, ctypes.c_void_p).value)

# Call registered symbol inside an AIR kernel
k = achainsaw.compile("""
extfn host_fma(a:i32, b:i32, c:i32)->i32
fn compute(x:i32, y:i32, z:i32)->i32
  b0:
    res = call host_fma(x, y, z)
    ret res
""")
print(k.run("compute", 3, 4, 5))  # 3 * 4 + 5 = 17
```

### 5. Accessing External Variables
External variables and exported global memory can be directly accessed and mutated via pointers (`ptr`):
- **Method A (Zero-Overhead Pointer Passing):** Pass the address of the external variable directly to the kernel, and read/write it at native CPU speed using `ld` and `st`:
  ```air
  fn increment_counter(p:ptr, delta:i32)->i32
    b0:
      val = ld p:i32
      new_val = add val, delta
      st p, new_val
      ret new_val
  ```
- **Method B (Symbol Address Resolution):** Query any registered or library-exported symbol address directly via `achainsaw.get_symbol_address(name)` or `kernel.lookup_symbol(name)`.

---

## 🏗️ Workspace Structure

```
achainsaw/
├── .github/
│   ├── workflows/
│   │   ├── ci.yml              # Multi-OS matrix CI (Linux, macOS, Windows, Py 3.11-3.13)
│   │   └── release.yml         # Automated multi-target CLI binary and PyPI wheel releases
│   └── dependabot.yml          # Automated Cargo & GitHub Actions dependency tracking
├── Cargo.toml                  # Workspace manifest with unified lints
├── pyproject.toml              # Python packaging manifest with Maturin backend
├── crates/
│   ├── achainsaw-ir/           # Lexer, Parser, AST, SSA Type-Checker, Optimizer, AIRB Codec
│   ├── achainsaw-codegen/      # Cranelift JIT engine & AOT native object / DLL compiler
│   ├── achainsaw-cli/          # Agent CLI driver, MCP JSON-RPC 2.0 stdio server
│   └── achainsaw-py/           # In-process PyO3 host bindings (zero-copy buffer protocol)
├── examples/                   # Each .air has matching .airb bytecode
│   ├── kernels/                # Chainsaw-BLAS: cosine, L2, softmax, RMSNorm, GEMV, GEMM, attention, RoPE, SwiGLU, Q8 GEMV, argmax, single-call decode
│   ├── fibonacci.air           # Iterative Fibonacci (branches, block parameters)
│   ├── sum_loop.air            # Iterative accumulator loop
│   ├── dot_product.air         # Scalar dot product with pointer arithmetic
│   ├── simd_vector_dot.air     # Fixed-width 128-bit SIMD dot product (v128, vfma, vsum)
│   ├── saxpy_vx.air            # Vector-length-agnostic SAXPY (vx, vl, masked tails)
│   ├── ffi_math_intrinsics.air # Calling C math functions via extfn
│   ├── opt_demo.air            # Input for `achainsaw opt`
│   ├── infinite_loop.air       # Stopped by the fuel budget (ERR_OUT_OF_FUEL)
│   └── py_numpy_simd.py        # Calling a kernel on NumPy arrays from Python
├── benchmarks/
│   └── benchmark_kernels.py    # 5,000-iteration BLAS benchmark and verification suite
├── tests/
│   ├── test_py_binding.py      # Python binding & buffer protocol test suite
│   ├── test_mcp_e2e.py         # Model Context Protocol stdio integration tests
│   └── test_aot_e2e.py         # Ahead-Of-Time DLL/shared object compilation tests
└── README.md
```

---

## 🔄 CI/CD & Automated Quality Gates

`achainsaw` enforces a rigorous, multi-platform continuous integration and delivery pipeline via GitHub Actions:

- **Continuous Integration (`ci.yml`):**
  - **Lint & Code Style:** Runs `cargo fmt --all -- --check` and `cargo clippy --workspace --all-targets -- -D warnings`.
  - **Cross-Platform Rust Matrix:** Runs `cargo test --workspace` concurrently on **Ubuntu Linux**, **macOS**, and **Windows**.
  - **Python Test Matrix:** Tests in-process bindings across **Python 3.11, 3.12, and 3.13** on Linux, macOS, and Windows.
  - **E2E & Numerical Verification:** Automatically runs unit tests, the MCP JSON-RPC server test, AOT native compilation test, and Chainsaw-BLAS NumPy numerical accuracy verification.
- **Automated Releases (`release.yml`):**
  - Triggers on version tag push (`v*.*.*`).
  - Cross-compiles standalone release CLI binaries for Linux (`x86_64-gnu`, `x86_64-musl`), Windows (`x86_64-msvc`), and macOS (`x86_64`, `aarch64` Apple Silicon).
  - Builds optimized multi-platform binary wheels (`manylinux`, `windows-x64`, `macos-universal2`) using `maturin-action`.
  - Generates SHA256 checksums and creates a GitHub Release with all binary and wheel assets attached.
- **Automated Dependency Updates (`dependabot.yml`):**
  - Weekly scans for Cargo crate dependencies.
  - Monthly scans for GitHub Actions versions.

---

## 📜 License

Licensed under the [Apache License, Version 2.0](LICENSE).
