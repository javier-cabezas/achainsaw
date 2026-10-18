# The AIR language

AIR (Agent Intermediate Representation) is a flat SSA language: one operation per line, every value named once, control flow as basic blocks. Programs are text (`.air`) or bytecode (`.airb`, the same program; see [the command line](cli.md#bytecode-achainsaw-assemble-achainsaw-disassemble)). The MCP server gives models a condensed version of this page as its instructions (`SERVER_INSTRUCTIONS` in [`mcp.rs`](../crates/achainsaw-cli/src/mcp.rs)).

## Basics

```air
# Sum of x[0..n)
fn sum(x:ptr, n:i64)->f32
  b0:
    jmp loop(0:i64, 0.0:f32)
  loop(i:i64, acc:f32):
    more = lt i, n
    br more, body, done
  body:
    v = ld x[i]:f32
    acc2 = add acc, v
    i2 = add i, 1:i64
    jmp loop(i2, acc2)
  done:
    ret acc
```

- **Types:** `i8` `i16` `i32` `i64` `f32` `f64` `ptr`; `f16` and `bf16` for storage (load, store, convert); vectors `v128` `v256` `v512` and the scalable `vx` (see [Vectors](#vectors)).
- **Functions:** `fn name(a:i32, b:f32)->i32`, with no `->` for no result and `->(i64, f32)` for several. The body is a list of blocks, `label:` or `label(x:i64, acc:f32):`. The first block is the entry: it takes no parameters and nothing branches to it.
- **SSA:** every register is assigned exactly once. Values that change around a loop are block parameters, passed by the branches: `jmp loop(i2, acc2)`.
- **Terminators:** each block ends with one: `jmp b(args)`, `br cond, then(args), else(args)`, `ret v`, `ret a, b` or `ret`.
- **Constants:** `x = cst 5:i32`, or inline in an operand, `add x, 1:i32`.
- **Arithmetic:** `add sub mul div rem and or xor shl shr min max`, and unsigned `udiv urem ushr umin umax`; operands share a type. Compares (`eq ne lt gt le ge`, unsigned `ult ugt ule uge`) give an `i32` 0 or 1, for `br` and `select`.
- **Memory:** `v = ld p:f32`, `st p, v`, `p = alloc n` (bytes), `free p`, pointer arithmetic with `add p, off`. Indexed operands scale by the element size: `ld x[i]:f32` reads `x + 4i`.
- **Casts:** `itof ftoi uitof ftoui sext zext trunc fext ftrunc bitcast`, written `dst = op src:type`.
- **Calls:** `r = call f(a, b)`, `q, m = call divmod(a, b)`; `extfn` declares C functions ([Calling C](ffi.md)); `par n, f(a, b)` runs `f(i, a, b)` for every `i < n` on all cores ([Parallel loops](parallel.md)).
- **Comments** start with `#`.

Every function is checked before it compiles: types, SSA (each use dominated by its definition), block arguments, terminators. A failure is a JSON diagnostic with an `ERR_*` code, the source span and the context to fix it, such as the registers in scope ([Diagnostics](cli.md#diagnostics)).

## Vectors

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
| Int8 dot product | `r = vdot c, a, b:i8` adds to each i32 lane of `c` the four products of the matching i8 lanes of `a` and `b`, exactly (i32 wrapping). One `sdot` on NEON (FEAT_DotProd) and SVE; on x86 with VNNI, `vpdpbusd` on `a ^ 0x80` minus `128 * sum(b)` (it multiplies unsigned by signed bytes), on both backends | i8 |
| Unsigned-by-signed dot product | `r = vdotu c, a, b:i8`: the same with `a`'s bytes unsigned (0..255). One `vpdpbusd` on x86 VNNI and one `usdot` on Arm with FEAT_I8MM (NEON and SVE), on both backends; Arm without I8MM takes two `sdot`s. Use it when one operand is naturally unsigned (4- and 6-bit quantized weights) | i8 |
| Compare | `m = vlt a, b:f32` (`veq`, `vne`, `vgt`, `vle`, `vge`) gives all-ones lanes where true | Any (signed for integers) |
| Select | `r = vsel m, a, b` takes bits of `a` where `m` is 1, else `b` | n/a |
| Reduce | `s = vsum v:f32` (`vmaxr`, `vminr`) | Any |
| Convert | `f = vitof v:f32` (i32 lanes to f32), `i = vftoi v:i32` (f32 to i32, saturating, NaN to 0); like scalar casts, the suffix is the result lane type | f32, i32 |
| Widen / narrow | `w = vwidenlo v:i16` / `vwidenhi` sign-extend the low / high half of the half-width lanes; `n = vnarrow a, b:i8` saturates both operands' lanes, `a`'s into the low half and `b`'s into the high half | widen: i16, i32, i64; narrow: i8, i16 |
| Half-precision floats | `w = vfwidenlo v:f16` / `vfwidenhi` convert the low / high half of the 16-bit float lanes to f32 (the suffix names the source; also `:bf16`); `n = vnarrow a, b:f16` converts f32 lanes to f16 or bf16, rounding to nearest-even like `ftrunc`. Exact and identical on both backends except NaN payloads; LLVM uses F16C or AArch64's `fcvtl`/`fcvtn` where available. `vl`, `ldm` and `stm` count f16/bf16 lanes too | f16, bf16 |
| Shift | `r = vshl v, s:i32` (`vshr` arithmetic, `vushr` logical) by a scalar amount, taken modulo the lane width | i8 to i64 |
| Absolute value / negate | `r = vabs v:i32`, `vneg`: integers wrap (`vabs` of the minimum is the minimum); floats clear or flip the sign bit, NaNs included | i8 to f64 |
| Square root | `r = vsqrt v:f32` (correctly rounded), `vrsqrt` computes `1 / vsqrt(v)` with two roundings, never a hardware estimate, so results are identical on every CPU | f32, f64 |
| Rounding | `r = vfloor v:f32`, `vceil`, `vround` (halfway cases away from zero, as C's `round`), `vroundeven` (to even), `vroundz` (toward zero) | f32, f64 |
| Copy sign | `r = vcopysign a, b:f32`: `a`'s magnitude with `b`'s sign bit | f32, f64 |
| Exponential | `e = vexp v:f32`: e^x within 2 ulp on [-87, 88], inputs clamped to that range, NaN propagated; a fixed algorithm, so results are bit-identical on every backend | f32 |
| Extract | `e = extlane v, 7:f32` | Index checked against the width (`vx`: its guaranteed 128 bits) |
| Broadcast a lane | `r = vdup v, 3:f32` puts lane 3 of `v` in every lane | Any, including f16/bf16; index checked as for `extlane` |
| Reverse | `r = vrev v:f32` reverses the order of the lanes | Any, including f16/bf16 |
| Interleave | `r = vziplo a, b:f32` gives `a0 b0 a1 b1 ...` from the low halves, `vziphi` the same from the high halves | Any, including f16/bf16 |
| Deinterleave | `r = vunziplo a, b:f32` gives the even lanes of `a` then of `b` (`a0 a2 ... b0 b2 ...`), `vunziphi` the odd lanes; they invert `vziplo`/`vziphi` | Any, including f16/bf16 |
| Lane count | `n = vl f32` | Any |

Semantics are identical on every backend: integer ops wrap, float `vmin`/`vmax` propagate NaN and order `-0.0` below `+0.0`, and reductions use a fixed recursive-halves order (`reduce(v) = op(reduce(lo), reduce(hi))`), so float sums are bit-reproducible for fixed widths.

**Permutations** (`vziplo`/`vziphi`, `vunziplo`/`vunziphi`, `vrev`, `vdup`) are defined relative to the vector's own lane count, so vector-length-agnostic code can use them on `vx` at any width: on SVE each is a single instruction (`zip1`/`zip2`, `uzp1`/`uzp2`, `rev`, `dup`) at every vector length. There is deliberately no shuffle by an index list, since its indices would depend on a width that `vx` code does not know. Zips and unzips cover the common layouts: pairs such as interleaved RoPE's `(x[2i], x[2i+1])` deinterleave with `vunziplo`/`vunziphi` and go back with `vziplo`/`vziphi`, and repeated zips transpose small tiles.

**Fast min/max:** float min/max (`min`, `max`, `vmin`, `vmax`, `vminr`, `vmaxr`) propagate NaN and order -0.0 below +0.0 by default. Compiling with fast math (`--fast-math` on `run`, `bench` and `build`, `fast_math=True` in Python's `compile`, `fast_math` in MCP `air_run`, `JitEngine::set_fast_math` or `CodegenOptions` in Rust) lowers the same instructions to compare and select instead, `max(a, b) = a > b ? a : b` and `min(a, b) = a < b ? a : b`, so a NaN operand or two zeros give `b`. That skips the NaN and signed-zero handling, which costs several instructions per op on 128-bit Cranelift vectors, and results stay identical on every backend (they are what x86's `maxps`/`minps` compute). Softmax runs 17% faster on Cranelift and 13% faster on LLVM with it; the [kernel benchmarks](kernels.md) use it. Fast math also lets bf16 `mm` use AMX (`tdpbf16ps`), SME (`bfmopa`) and AVX512-BF16 (`vdpbf16ps`), which treat bf16 subnormal inputs as zero; by default bf16 `mm` uses exact vector FMAs on those CPUs, so it widens every bf16 value exactly on every backend (int8, f16 and f32 `mm` use the matrix engines either way).

Vector-length-agnostic code processes `vl` lanes per iteration and finishes with masked accesses, as in [`examples/saxpy_vx.air`](../examples/saxpy_vx.air):
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

**Half-precision storage types:** `f16` (IEEE binary16) and `bf16` (bfloat16) can be loaded, stored, and converted (`f = fext h:f32`, `h = ftrunc f:bf16`, rounding to nearest-even; vectors with `vfwidenlo`/`vfwidenhi` and `vnarrow`), but not used in arithmetic or function signatures; convert to `f32` first.

**Standard library:** a top-level `use` line links functions from AIR's standard library, written in AIR ([`crates/achainsaw-ir/std/std.air`](../crates/achainsaw-ir/std/std.air)), into the module, together with the library functions they call; they are then ordinary functions to `call` or run with `par`, and work the same in files, MCP `air_run`, Python and AOT builds.

```air
use quantize_q8, qmat_chunk

fn gemv_chunk(c:i64, mat:ptr, xq:ptr, dx:ptr, y:ptr, k:i64)
  b0:
    off = mul c, 256:i64
    yc = add y, off
    call qmat_chunk(c, mat, xq, dx, yc, k)
    ret
```

It holds the building blocks the example kernels share: `quantize_q8` (Q8_0 activations), `qmat_chunk` (64 output rows of a Q8_0, Q4_0, Q6_K, Q4_K or Q5_K matrix, which may be stored in parts of different formats), `rope_half`, `exp_shift` (softmax numerators) and `silu_mul` (SwiGLU). `achainsaw std` lists them with their documentation and `achainsaw std <name>` prints one's source; MCP clients have the `air_std` tool.

**Matrix multiply:** `mm pc, pa, pb, m, n, k:bf16` computes `C[m x n] += A[m x k] * B[k x n]` on row-major, contiguous matrices. `A`/`B` hold `bf16`, `f16`, or `f32` elements with an `f32` `C`, or `i8` elements with an `i32` `C` (exact, wrapping). Float accumulation order is implementation-defined, so results agree across backends within rounding error rather than bit for bit. Under a fuel budget, `mm` costs one unit per 1024 multiply-adds, charged before it starts. Each backend's kernels, AMX and SME among them, are in [Backends](backends.md#matrix-multiply-mm).

## Memory operands, several results and inline functions

**Indexed operands.** `ld`, `st`, `ldm` and `stm` take `p[i]`, which addresses `p + i * size` for an i64 index: the size of the loaded or stored scalar, or the lane type of `ldm`/`stm`. A vector access, or a byte offset, names its unit: `p[i:f32]`, `p[o:i8]`. The sandbox checks the address the index computes.

```air
  body:
    rest = sub n, i
    g = ldm gate[i]:vx, rest:f32      # was: off = mul i, 4:i64; pg = add gate, off; g = ldm pg:vx, rest:f32
    st scores[t], s                   # s:f32 at scores + 4t
```

**Several results.** A function can return several values, `fn divmod(a:i64, b:i64)->(i64, i64)`, with `ret q, r`, received as `q, r = call divmod(x, 10:i64)`. Results of any type cross calls, `vx` included (on SVE, a struct of scalable vectors); hosts call only functions with at most one scalar result.

**Inline functions.** `inline fn` marks a helper whose calls are replaced by its body before compiling, on both backends, JIT and AOT (`achainsaw opt` shows the expansion). A straight-line helper merges into the caller's block, so it adds no branches and no fuel checks; inline functions cannot be recursive (`ERR_RECURSIVE_INLINE`). The standard library's quantized matrix kernels are built this way: `qmat_chunk` (1 token), `qmat_chunk4` (4 tokens) and `qmat_chunk8` (8) call the same inline steps (`qmat_w8`/`qmat_w4`/`qmat_w5`/`qmat_w6` unpack two weight vectors, `qmat_dot` and `qmat_dot1` run `vdotu`s, `qmat_accr`/`qmat_accmr` scale and accumulate), so every token's arithmetic is identical by construction, and `llama_decode.air` reads its tables with `load_cfg`, `model_globals` and `layer_weights`. Together with indexed operands, this cut the standard library from 712 to 447 lines of code (`qmat_chunk` 185 to 80, `qmat_chunk4` 335 to 114) and `llama_decode.air` from 718 to 637, without slowing anything down: in the base-vs-head benchmark comparison every kernel is within noise, and Cranelift's quantized GEMVs and prefill got 6–7% faster (its index shifts fold into x86 addressing modes).

**Scalar math.** Rounding to an integral value, `floor`, `ceil`, `round` (halfway away from zero), `roundeven` and `roundz` (toward zero); bit counts `popcnt`, `clz` and `ctz` (the bit width for 0); `copysign a, b` for floats, `rotl`/`rotr` for integers (amount modulo the width); `r = fma a, b, c` with one rounding; `uitof`/`ftoui` unsigned conversions (`ftoui` saturates, negative values and NaN to 0). All are single instructions or exact sequences, bit-identical on both backends and checked against Rust's own operations (`tests/scalar_math.rs`). `quantize_q8` rounds with `vround`, so it now matches llama.cpp's `roundf` exactly at halfway cases. These mnemonics are recognized only right after `=`, so programs that use them as register names still parse.
