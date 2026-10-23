# Command line

Every command prints JSON (`--json` on `check`, `assemble` and `build`), so an agent or a script can read the results. The [MCP server](mcp.md) offers the same operations as tools.

## Validate syntax and SSA invariants (`achainsaw check`)
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

## Diagnostics
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

## Compile and run (`achainsaw run`)
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

## Benchmark compilation and execution (`achainsaw bench`)
```bash
achainsaw bench examples/fibonacci.air --iters 200
```

## Bytecode (`achainsaw assemble`, `achainsaw disassemble`)
AIR modules can be assembled into compact binary bytecode for persistent caching, agent-to-agent IPC, and zero-parse reloading. Every AIR construct round-trips through AIRB, and files carry a format version that decoders check:
```bash
# Assemble text AIR to compact binary bytecode (.airb)
achainsaw assemble examples/fibonacci.air -o examples/fibonacci.airb --json

# Direct execution of pre-parsed binary bytecode
achainsaw run examples/fibonacci.airb --func fib --args 30 --json

# Disassemble binary bytecode back into canonical text AIR
achainsaw disassemble examples/fibonacci.airb
```


## Optimizer (`achainsaw opt`)
Pre-evaluate constant expressions, simplify algebraic identities (`x + 0 -> x`, `x * 1 -> x`), fold invariant branches, and eliminate dead code and unreachable blocks. The output also shows what code generation does before lowering: `inline fn` calls expanded, and vector loops with masked accesses split into a full-vector copy (see [Vectors](language.md#vectors)):
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
  "loops_split": 0,
  "total_optimizations": 10,
  "iterations": 2,
  "code": "fn demo(x:i32)->i32\n  b0:\n    ret x\n"
}
```

## Ahead-of-time compilation (`achainsaw build`)
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

## CPU features and ISA targeting (`achainsaw cpu`)
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
