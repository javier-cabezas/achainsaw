# Backends

achainsaw has two code generators, Cranelift (always built) and LLVM 22 (opt-in), behind one runtime.

## Choosing a backend (`--backend`)
Two code generators share one runtime, so fuel budgets, memory quotas, the MCP sandbox, and the results of scalar and fixed-width vector code are the same on both (`vx` code computes the same values, but `vl` can be larger on LLVM):

| Backend | Compile latency (release, simple kernel) | Code | Availability |
|---|---|---|---|
| `cranelift` | ~0.2–0.3 ms | 128-bit vectors | always |
| `llvm` | ~5–7 ms | fully optimized; full-width AVX2/AVX-512/SVE vectors (`vx` up to 512 bits or scalable) | builds with the `llvm` feature |

Pick one with `--backend` on `run`, `bench` and `build`, the `ACHAINSAW_BACKEND` environment variable, `backend=` in Python, or the `backend` argument of MCP `air_run`. `auto` (the default) uses LLVM, when it is built in, for modules that use `v256`, `v512`, `vx`/`vl` or `mm`, and Cranelift otherwise, so scalar code keeps sub-millisecond compiles. For example, [`examples/saxpy_vx.air`](../examples/saxpy_vx.air) on an AVX-512 machine runs 3–4x faster on LLVM.

```bash
# Build with the LLVM backend (needs LLVM 22, e.g. apt install llvm-22-dev)
export LLVM_SYS_221_PREFIX=/usr/lib/llvm-22
cargo build --release -p achainsaw --features llvm

achainsaw --backend llvm run examples/fibonacci.air --func fib --args 30
achainsaw --backend llvm build examples/kernels/gemv_f32.air --target-cpu znver4 --shared
ACHAINSAW_BACKEND=llvm achainsaw mcp     # every air_run in the server uses LLVM
```

AOT builds with `--backend llvm` accept the same `--target`, `--target-cpu` and `--target-features` options and can use every feature (nothing ends up in `ignored_features`). `crates/achainsaw-codegen/tests/backend_parity.rs` checks that both backends agree bit for bit, including on division by zero, out-of-range shifts, NaN handling, and saturating conversions.

## Vector width

The default Cranelift backend runs every vector program, splitting `v256`/`v512` into 128-bit operations, treating `vx` as 128 bits, and running `mm` as 128-bit tiles (4 rows by 8 columns of C, with FMA where the CPU has it). The opt-in LLVM backend uses the hardware's full width:

| Target (LLVM backend) | `vx` | Masked `ldm`/`stm` |
|---|---|---|
| x86_64 with AVX-512F | 512 bits (zmm) | AVX-512 k-masks |
| x86_64 with AVX2 | 256 bits (ymm) | `vmaskmov` |
| x86_64 SSE/AVX only, AArch64 NEON (including Apple M4) | 128 bits | per-lane |
| AArch64 with SVE | scalable, the CPU's vector length (`<vscale x ...>`) | `whilelo` predicates |

`v256`/`v512` map to native registers whenever the target has them. On SVE, horizontal reductions follow the same adjacent-pairs tree at the run-time vector length. Programs written against `vl` and `mm` speed up without changes. `vx` width therefore depends on the backend and CPU, so programs must use `vl` rather than assume a lane count; `JitEngine::vx_bits()` and `achainsaw cpu` report it.

## Matrix multiply (`mm`)

On the LLVM backend, `mm` uses the best kernel the target has:

| Target | `mm` kernel |
|---|---|
| x86 with AMX (Sapphire/Emerald Rapids: bf16, i8; Granite Rapids: also f16) | 16x16 AMX tiles (`tdpbf16ps`, `tdpbssd`, `tdpfp16ps`) |
| AArch64 with SME (e.g. Apple M4) | ZA-tile outer products (`fmopa`, `bfmopa`, `smopa`) in streaming mode |
| x86 with AVX512-BF16 (Zen 4/5, Cooper Lake), bf16 with fast math | `vdpbf16ps` (two products per f32 lane), tiles of 4 rows of C by 64 columns |
| Everything else | broadcast-FMA at full `vx` width (AVX-512/AVX2/SVE/NEON), tiles of 4 rows of C by 4 vectors (2 on 16-register targets); bf16, f16 and i8 rows of A are widened once per tile, so the inner loop broadcasts them from memory |

On a Zen 4 core (AVX-512), 256x256x256 `mm` runs at about 155 GFLOP/s for f32, 145 for bf16 (290 with fast math, on `vdpbf16ps`; PyTorch's oneDNN gets 282), 125 GOP/s for i8, and 140 for f16. Cranelift's 128-bit kernel reaches about 50, 30, 33 and 8. The AMX path is compiled and checked in assembly but has not run on AMX hardware yet. The SME path is tested under QEMU (`tools/qemu-aarch64/run.sh`); its objects call the SME ABI routine `__arm_tpidr2_save`, provided by GCC 14+ libgcc or compiler-rt.
