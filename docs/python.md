# Python

For AI agent orchestrators (LangGraph, AutoGen, CrewAI, DSPy), `achainsaw` provides native in-process Python bindings via PyO3 with **zero-copy NumPy memory integration** and compiles in a fraction of a millisecond.

## Installation

From GitHub (needs a Rust toolchain; builds the Cranelift backend):

```bash
pip install git+https://github.com/javier-cabezas/achainsaw.git
```

From a checkout, with the LLVM backend too (needs LLVM 22, see [Backends](backends.md#choosing-a-backend---backend)):

```bash
export LLVM_SYS_221_PREFIX=/usr/lib/llvm-22
MATURIN_PEP517_ARGS="--features llvm" pip install .
# then: achainsaw.compile(src, backend="llvm"); achainsaw.available_backends()
```

Or build the extension and copy it next to your code:

```bash
cargo build --release -p achainsaw-py
cp target/release/libachainsaw.so achainsaw.so      # Linux
cp target/release/libachainsaw.dylib achainsaw.so   # macOS
cp target/release/achainsaw.dll achainsaw.pyd       # Windows
```

## Compile and run on NumPy arrays
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

## Diagnostics
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

## CPU features and the ISA cap
```python
report = achainsaw.cpu_features()       # same report as `achainsaw cpu`
print(report["host"]["max_isa"])        # e.g. "avx512"

achainsaw.set_isa_cap("avx2")           # kernels compiled from now on use at most AVX2
kernel = achainsaw.compile(air_kernel)
achainsaw.set_isa_cap(None)             # remove the cap
```

## Bytecode (AIRB)
```python
# Assemble to compact binary payload (for caching or network transfer)
binary_payload = achainsaw.assemble(air_kernel)

# Directly compile pre-parsed binary bytecode (<0.05 ms)
kernel = achainsaw.compile_binary(binary_payload)
result = kernel.run("simd_scale", arr, 2.5, 2)

# Disassemble binary payload back into canonical text AIR
text_ir = achainsaw.disassemble(binary_payload)
```

## Latency

On a Ryzen 7 8845HS, for [`examples/agent/sum_squares.air`](../examples/agent/sum_squares.air) on Cranelift:

| | Python, in process | CLI, one process per run |
|---|---|---|
| Compile | 0.25 ms per `achainsaw.compile` | 2.1 ms per `achainsaw run`, including starting the process and running |
| Call | 0.21 µs per `kernel.run` | |

NumPy arrays are passed as pointers to their data, without copies.
