# achainsaw

[![CI](https://github.com/javier-cabezas/achainsaw/actions/workflows/ci.yml/badge.svg)](https://github.com/javier-cabezas/achainsaw/actions/workflows/ci.yml)
[![Release](https://github.com/javier-cabezas/achainsaw/actions/workflows/release.yml/badge.svg)](https://github.com/javier-cabezas/achainsaw/actions/workflows/release.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE)

> **High-Performance, Token-Minimal Compiler Toolchain Designed Exclusively for AI Agents**

`achainsaw` is a specialized compiler and JIT execution toolchain built from first principles for autonomous LLM agents. It completely discards human-centric syntactic sugar (no curly braces, no indentation sensitivity, no verbose keywords, no English prose compiler errors) to optimize for two uncompromising objectives:

1. **Minimize Agent Processing Time:** Autoregressive LLM decoding latency scales linearly with token count. `achainsaw` uses **AIR (Agent Intermediate Representation)**—a flat, sequential Single Static Assignment (SSA) format with single-token mnemonics. It eliminates nesting hallucinations and reduces token consumption by **60% to 80%** compared to traditional languages.
2. **Generate Highly Efficient Native Code as Fast as Possible:** Powered by **Cranelift** (the code generator behind Wasmtime), `achainsaw` compiles AIR modules directly to bare-metal x86_64 / AArch64 machine code with JIT compilation times under **10 milliseconds** and near-LLVM execution throughput.

---

## ⚡ Key Highlights

- **Linear SSA / Three-Address Code (TAC):** Zero nested expressions (`(+ (* a b) c)`). Eliminates bracket/parenthesis balancing errors and allows causal transformer attention heads to track dependencies with forward sequential ease.
- **Single-Token Mnemonics:** Operators and keywords (`fn`, `cst`, `add`, `mul`, `ld`, `st`, `br`, `jmp`, `ret`) map strictly to single, indivisible tokens in standard BPE vocabularies (OpenAI `cl100k`/`o200k`, Meta LLaMA 3, Google Gemini).
- **Machine-Native Diagnostic Protocol:** Zero natural-language prose error messages. Validation failures immediately produce structured JSON diagnostics with exact instruction indices, expected types, and candidate replacement patches for single-shot agent self-repair.
- **Universal 128-bit SIMD Primitives:** Hardware-accelerated vectorization (`v128`) supporting 4 parallel single-precision floats (`vfadd`, `vfmul`, `vfsub`, `vfdiv`), 4 parallel 32-bit integers (`viadd`, `vimul`), `splat` broadcast, and lane extraction (`extlane`). *(Architecturally designed for future extension to `v256` AVX2 and `v512` AVX-512).*
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
AIR modules can be assembled into compact binary bytecode for persistent caching, agent-to-agent IPC, and zero-parse reloading:
```bash
# Assemble text AIR to compact binary bytecode (.airb)
achainsaw assemble examples/simd_vector_dot.air -o examples/simd_vector_dot.airb --json

# Direct execution of pre-parsed binary bytecode
achainsaw run examples/simd_vector_dot.airb --func simd_dot --args 1 --json

# Disassemble binary bytecode back into canonical text AIR
achainsaw disassemble examples/simd_vector_dot.airb
```

### 6. Model Context Protocol (MCP) Server (`achainsaw mcp`)
`achainsaw` includes a native Model Context Protocol (MCP) server communicating over JSON-RPC 2.0 stdio, exposing compiler tools directly to LLM agents and IDE assistants:
```bash
achainsaw mcp
```

**Exposed MCP Tools:**
- **`air_check`**: Validates AIR textual IR or base64 AIRB bytecode syntax and SSA invariants. Returns structured diagnostic metrics or error payloads with line/column pointers and self-repair hints.
- **`air_run`**: JIT compiles and executes AIR functions with arguments, loop fuel budget, and memory quota sandboxing.
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

Detection covers SSE through AVX-512 (including BF16/FP16/VNNI) and AMX on x86_64, and NEON, SVE/SVE2 (with vector length), and SME/SME2 on AArch64. Cranelift currently emits 128-bit vector code (using VEX/EVEX encodings when AVX/AVX-512 are available); 256/512-bit, scalable, and matrix code generation is planned for an opt-in LLVM backend.

**Cap the ISA level** to exercise lower tiers on a more capable machine (for example, AVX2 code on an AVX-512 host). Levels: `sse`, `avx`, `avx2`, `avx512`, `amx` (x86_64) and `neon`, `sve`, `sve2`, `sme` (AArch64):
```bash
achainsaw run examples/sum_loop.air --func sum_to_n --args 100 --isa avx2
ACHAINSAW_MAX_ISA=sse achainsaw mcp      # every JIT compilation in the server is capped
```

**Target a specific CPU in AOT builds** with LLVM CPU names and feature strings. `+feature` also enables its prerequisites and `-feature` disables everything that depends on it, as in LLVM. Features the backend cannot use are listed in `ignored_features`:
```bash
achainsaw build examples/kernels/gemv_f32.air --target-cpu x86-64-v3
achainsaw build examples/kernels/gemv_f32.air --target-cpu znver4 --target-features -avx512f
```

---

## ⚡ Chainsaw-BLAS: High-Performance Agent AI Kernel Library

`achainsaw` ships with pre-compiled, mathematically verified AI kernels in `examples/kernels/` targeting LLM inference primitives, vector search, and token normalization:

| Kernel | Source | Bytecode | Primary Use Case | Numerical Error |
|---|---|---|---|---|
| **Cosine Similarity** | `cosine_similarity.air` | `.airb` | High-throughput embedding search & RAG | `< 1e-7` |
| **Euclidean Distance (L2)** | `euclidean_distance.air` | `.airb` | Vector quantization & nearest neighbors | `0.00e+00` |
| **Numerically Stable Softmax** | `softmax.air` | `.airb` | Attention head weighting ($\exp(x_i - \max)/\sum \exp$) | `< 1e-8` |
| **RMSNorm** | `rmsnorm.air` | `.airb` | Transformer token normalization (LLaMA, Mistral, Gemma) | `< 5e-7` |
| **GEMV (f32)** | `gemv_f32.air` | `.airb` | Matrix-vector linear projection | `< 3e-5` |

Run the kernel benchmarks and numerical accuracy verification suite:
```bash
python benchmarks/benchmark_kernels.py
```

---

## 🐍 Python Host Integration (`achainsaw-py`)

For AI agent orchestrators (LangGraph, AutoGen, CrewAI, DSPy), `achainsaw` provides native in-process Python bindings via PyO3 with **zero-copy NumPy memory integration** and microsecond compilation throughput.

### Installation / Build
```bash
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
    vfactor = splat factor
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
    vscaled = vfmul v, vfactor
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
├── examples/
│   ├── kernels/                # Chainsaw-BLAS: Cosine, L2, Softmax, RMSNorm, GEMV
│   ├── sum_loop.air            # Iterative accumulator loop
│   ├── fibonacci.air           # Branching Fibonacci kernel
│   ├── simd_vector_dot.air     # 128-bit SIMD hardware dot product kernel
│   └── ffi_math_intrinsics.air # FFI and C standard math intrinsics kernel
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

Dual-licensed under MIT or Apache-2.0.
