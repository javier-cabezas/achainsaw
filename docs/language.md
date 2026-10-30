# The AIR language

AIR (Agent Intermediate Representation) is a flat SSA language: one operation per line, values that change around loops passed as block parameters, control flow as basic blocks. Programs are text (`.air`) or bytecode (`.airb`, the same program; see [the command line](cli.md#bytecode-achainsaw-assemble-achainsaw-disassemble)). The MCP server gives models a condensed version of this page as its instructions (`SERVER_INSTRUCTIONS` in [`mcp.rs`](../crates/achainsaw-cli/src/mcp.rs)).

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
- **SSA:** a register holds one value. Values that change around a loop are block parameters, passed by the branches: `jmp loop(i2, acc2)`. A name may be defined again: the new definition shadows the old one in the code it dominates, so `loop(i:i64)` and `body(i:i64)` are two registers, and so are the two `i`s of `i = add i, 1:i64`. A use that could see either of two definitions, depending on the path taken (a name defined again in one arm of a branch and used after the join, or in a loop body and not passed back), is `ERR_AMBIGUOUS_REGISTER`.
- **Names:** registers, labels and functions are identifiers. Instruction mnemonics are keywords only right after `=`, so they are names everywhere else (`neg = lt v, 0:f32`, `rem = sub n, i`); only `st`, `free`, `call`, `jmp`, `br`, `ret`, `fn` and `extfn` are reserved (and `switch` starts a terminator when neither `=` nor `,` follows it). An instruction written without its destination (`neg x`) is `ERR_MISSING_DESTINATION`.
- **Terminators:** each block ends with one: `jmp b(args)`, `br cond, then(args), else(args)`, `switch v, other(args), 0: a(args), 3: b(args)`, `ret v`, `ret a, b` or `ret`. A `switch` jumps to the case whose value equals the integer `v`, or to the default (`other`) when none does; each case and the default pass their own arguments, and several may go to one block. Case values are integer literals written signed or unsigned within `v`'s type (`-128` to `255` for `i8`, where `255` and `-1` are the same value, so not both), each at most once (`ERR_INVALID_SWITCH_CASE`). Both backends pick a jump table for dense cases and a tree of compares for sparse ones, and a `switch` costs one fuel check, as `jmp` and `br` do. `switch` starts a terminator only at the start of a line and when neither `=` nor `,` follows it, so it remains a valid register name (`switch = ...`, `switch, y = call f(x)`) and block label; the optimizer folds a `switch` on a constant to a `jmp`.
- **Constants:** `x = cst 5:i32`, or inline in an operand, `add x, 1:i32`; an integer literal of a float type is that float (`cst 0:f32`).
- **Arithmetic:** `add sub mul div rem and or xor shl shr min max`, and unsigned `udiv urem ushr umin umax`; operands share a type. Compares (`eq ne lt gt le ge`, unsigned `ult ugt ule uge`) give an `i32` 0 or 1, for `br` and `select`.
- **Math functions:** `y = exp x`, `log`, `tanh`, `sin`, `cos` on f32, and the same on vector lanes (`r = vsin v:f32`): see [Math functions](#math-functions).
- **Memory:** `v = ld p:f32`, `st p, v`, `p = alloc n` (bytes), `free p`, pointer arithmetic with `add p, off`. Indexed operands scale by the element size: `ld x[i]:f32` reads `x + 4i`. Read-only tables are `data` declarations (`p = addr name`), and fixed-size scratch buffers live in the function's frame (`p = stack 256`): see [Constant data and stack buffers](#constant-data-and-stack-buffers). Memory that `par` calls share takes [atomic accesses](#atomics).
- **Casts:** `itof ftoi uitof ftoui sext zext trunc fext ftrunc bitcast`, written `dst = op src:type`.
- **Calls:** `r = call f(a, b)`, `q, m = call divmod(a, b)`; through a function handle, `h = fnref f` then `r = call_indirect h(a, b)->i64` ([Function handles](#function-handles)); `extfn` declares C functions ([Calling C](ffi.md)); `par n, f(a, b)` runs `f(i, a, b)` for every `i < n` on all cores ([Parallel loops](parallel.md)).
- **Comments** start with `#`.

Every function is checked before it compiles: types, SSA (each use dominated by its definition, and reached by no other), block arguments, terminators. A failure is a JSON diagnostic with an `ERR_*` code, the source span and the context to fix it, such as the registers in scope ([Diagnostics](cli.md#diagnostics)).

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
| Gather | `v = vgather p, idx:f32` reads lane `i` from `p[idx[i]]`; `vgather p, idx, n:f32` reads only the first `n` lanes and zeroes the rest. The indices are signed lanes as wide as the elements, scaled by the element size, and `idx`'s vector type is the result's (see [Gather](#gather)) | f32, i32 (i32 indices); f64, i64 (i64 indices) |
| Scatter | `vscatter p, idx, v:f32` stores lane `i` of `v` to `p[idx[i]]`, in lane order (when lanes share an index, the highest one stays); `vscatter p, idx, v, n:f32` stores only the first `n` lanes. Indices as for `vgather`, and `v` has `idx`'s vector type (see [Gather](#gather)) | f32, i32 (i32 indices); f64, i64 (i64 indices) |
| Broadcast | `v = splat x:v512` (the vector type is required) | i8 to f64 |
| Arithmetic | `r = vadd a, b:f32` (`vsub`, `vmul`, `vdiv`, `vmin`, `vmax`); `vumin`, `vumax` compare as unsigned | `vmul`: not i8; `vdiv`: f32/f64; `vmin`/`vmax`: not i64; `vumin`/`vumax`: i8, i16, i32 |
| Bitwise | `r = vand a, b:i32` (`vor`, `vxor`) | Any |
| Saturating add / subtract | `r = vaddsat a, b:i8` (`vsubsat`) clamps each result to the lane's signed range instead of wrapping, `vuaddsat` / `vusubsat` to its unsigned range (`vusubsat` of a larger value gives 0). One instruction on x86 (`paddsb`, `psubusw`, ...) and Arm (`sqadd`, `uqsub`, ...) on both backends | i8, i16 |
| Multiply high | `r = vmulhi a, b:i16` gives the high half of each lane's double-width product, `(a * b) >> 16` exactly (`:i32`: `>> 32`); `vumulhi` multiplies both as unsigned. For fixed-point products and division by a constant. On LLVM, `pmulhw`/`pmulhuw` (i16) or `pmuldq`/`pmuludq` pairs (i32) on x86, `smull`/`umull` pairs on NEON and one `smulh`/`umulh` on SVE; Cranelift multiplies widened lanes | i16, i32 |
| Bit count | `r = vpopcnt v:i32` counts each lane's set bits, `vclz` its leading zeros, `vctz` its trailing zeros (a zero lane gives the lane width), as scalar `popcnt`, `clz` and `ctz` do. On LLVM, `cnt`/`clz` on NEON and SVE, AVX-512 `vpopcnt`/`vplzcnt` where the CPU has them, else `pshufb` nibble lookups; Cranelift counts the bits of each byte and adds the bytes of a lane | i8 to i64 |
| Fused multiply-add | `r = vfma a, b, c:f32` computes `a*b + c` with one rounding | f32, f64 |
| Int8 dot product | `r = vdot c, a, b:i8` adds to each i32 lane of `c` the four products of the matching i8 lanes of `a` and `b`, exactly (i32 wrapping). One `sdot` on NEON (FEAT_DotProd) and SVE; on x86 with VNNI, `vpdpbusd` on `a ^ 0x80` minus `128 * sum(b)` (it multiplies unsigned by signed bytes), on both backends | i8 |
| Unsigned-by-signed dot product | `r = vdotu c, a, b:i8`: the same with `a`'s bytes unsigned (0..255). One `vpdpbusd` on x86 VNNI and one `usdot` on Arm with FEAT_I8MM (NEON and SVE), on both backends; Arm without I8MM takes two `sdot`s. Use it when one operand is naturally unsigned (4- and 6-bit quantized weights) | i8 |
| Compare | `m = vlt a, b:f32` (`veq`, `vne`, `vgt`, `vle`, `vge`) gives all-ones lanes where true; `vult`, `vugt`, `vule`, `vuge` compare integers as unsigned | Any (signed for integers); unsigned: i8 to i64 |
| Select | `r = vsel m, a, b` takes bits of `a` where `m` is 1, else `b` | n/a |
| Reduce | `s = vsum v:f32` (`vmaxr`, `vminr`) | Any |
| Convert | `f = vitof v:f32` (i32 lanes to f32, or i64 to f64 with `:f64`), `i = vftoi v:i32` (f32 to i32, or f64 to i64 with `:i64`; saturating, NaN to 0); `vuitof` and `vftoui` read and write unsigned integers (`vftoui` takes negative values and NaN to 0). Like scalar casts, the suffix is the result lane type. Cranelift converts f64 to i64 one lane at a time, since x86 has no packed instruction for it | `vitof`, `vuitof`: f32, f64; `vftoi`, `vftoui`: i32, i64 |
| Widen / narrow | `w = vwidenlo v:i16` / `vwidenhi` sign-extend the low / high half of the half-width lanes, `vuwidenlo` / `vuwidenhi` zero-extend them; `n = vnarrow a, b:i8` saturates both operands' lanes, `a`'s into the low half and `b`'s into the high half. Floats: `w = vfwidenlo v:f32` / `vfwidenhi` give f64 lanes (the suffix names the source, as for f16), `n = vnarrow a, b:f32` rounds f64 lanes to f32 (nearest-even, as `ftrunc`). Integers to floats: `w = vfwidenlo v:i16` / `vfwidenhi` convert the low / high half of the i16 lanes to f32, and `:i32` to f64, signed and exactly; they are `vwidenlo` then `vitof` (no CPU converts them in one instruction), and i8 lanes take a `vwidenlo` first | widen: i16, i32, i64; narrow: i8, i16, f32; `vfwidenlo`/`vfwidenhi` from integers: i16, i32 |
| Half-precision floats | `w = vfwidenlo v:f16` / `vfwidenhi` convert the low / high half of the 16-bit float lanes to f32 (the suffix names the source; also `:bf16`); `n = vnarrow a, b:f16` converts f32 lanes to f16 or bf16, rounding to nearest-even like `ftrunc`. Exact and identical on both backends except NaN payloads; LLVM uses F16C or AArch64's `fcvtl`/`fcvtn` where available. `vl`, `ldm` and `stm` count f16/bf16 lanes too | f16, bf16 |
| Shift | `r = vshl v, s:i32` (`vshr` arithmetic, `vushr` logical) by a scalar amount, or by a vector of `v`'s type that gives each lane its own amount; amounts are taken modulo the lane width. On LLVM, per-lane amounts are one variable shift on NEON and SVE, on AVX-512 for every lane size, and on AVX2 for 32- and 64-bit lanes (other lanes take a short sequence); Cranelift has no variable vector shift and shifts each lane separately | i8 to i64 |
| Absolute value / negate | `r = vabs v:i32`, `vneg`: integers wrap (`vabs` of the minimum is the minimum); floats clear or flip the sign bit, NaNs included | i8 to f64 |
| Square root | `r = vsqrt v:f32` (correctly rounded), `vrsqrt` computes `1 / vsqrt(v)` with two roundings, never a hardware estimate, so results are identical on every CPU | f32, f64 |
| Rounding | `r = vfloor v:f32`, `vceil`, `vround` (halfway cases away from zero, as C's `round`), `vroundeven` (to even), `vroundz` (toward zero) | f32, f64 |
| Copy sign | `r = vcopysign a, b:f32`: `a`'s magnitude with `b`'s sign bit | f32, f64 |
| Math functions | `e = vexp v:f32`, `vlog`, `vtanh`, `vsin`, `vcos`: the [math functions](#math-functions) on each lane | f32 |
| Extract | `e = extlane v, 7:f32` | Index checked against the width (`vx`: its guaranteed 128 bits) |
| Insert | `r = inslane v, x, 7:f32`: `v` with lane 7 replaced by the scalar `x` (of the lane type) | i8 to f64; index checked as for `extlane` |
| Broadcast a lane | `r = vdup v, 3:f32` puts lane 3 of `v` in every lane | Any, including f16/bf16; index checked as for `extlane` |
| Reverse | `r = vrev v:f32` reverses the order of the lanes | Any, including f16/bf16 |
| Interleave | `r = vziplo a, b:f32` gives `a0 b0 a1 b1 ...` from the low halves, `vziphi` the same from the high halves | Any, including f16/bf16 |
| Deinterleave | `r = vunziplo a, b:f32` gives the even lanes of `a` then of `b` (`a0 a2 ... b0 b2 ...`), `vunziphi` the odd lanes; they invert `vziplo`/`vziphi` | Any, including f16/bf16 |
| Table lookup | `r = vtbl t, idx:i8`: byte `i` of `r` is byte `idx[i] & 15` of `t`'s first 16 bytes, at any width (load a 16-byte table with `ldm p:vx, 16:i8`). One `pshufb` on x86 (SSSE3; AVX2 and AVX512BW for 256 and 512 bits, on the table broadcast across 128-bit lanes) and one `tbl` on Arm (NEON, and SVE at every vector length), after masking the indices | i8 |
| Lane count | `n = vl f32` | Any |

Semantics are identical on every backend: integer ops wrap, float `vmin`/`vmax` propagate NaN and order `-0.0` below `+0.0`, and reductions use a fixed recursive-halves order (`reduce(v) = op(reduce(lo), reduce(hi))`), so float sums are bit-reproducible for fixed widths.

**Permutations** (`vziplo`/`vziphi`, `vunziplo`/`vunziphi`, `vrev`, `vdup`) are defined relative to the vector's own lane count, so vector-length-agnostic code can use them on `vx` at any width: on SVE each is a single instruction (`zip1`/`zip2`, `uzp1`/`uzp2`, `rev`, `dup`) at every vector length. There is deliberately no shuffle by an index list, since its indices would depend on a width that `vx` code does not know; the one data-dependent permutation, `vtbl`, looks up a 16-byte table that is the same at every width, which covers 4-bit codebooks (a nibble to its int8 or f16 value) and other small lookups. Zips and unzips cover the common layouts: pairs such as interleaved RoPE's `(x[2i], x[2i+1])` deinterleave with `vunziplo`/`vunziphi` and go back with `vziplo`/`vziphi`, and repeated zips transpose small tiles.

**Fast min/max:** float min/max (`min`, `max`, `vmin`, `vmax`, `vminr`, `vmaxr`) propagate NaN and order -0.0 below +0.0 by default. Compiling with fast math (`--fast-math` on `run`, `bench` and `build`, `fast_math=True` in Python's `compile`, `fast_math` in MCP `air_run`, `JitEngine::set_fast_math` or `CodegenOptions` in Rust) lowers the same instructions to compare and select instead, `max(a, b) = a > b ? a : b` and `min(a, b) = a < b ? a : b`, so a NaN operand or two zeros give `b`. That skips the NaN and signed-zero handling, which costs several instructions per op on 128-bit Cranelift vectors, and results stay identical on every backend (they are what x86's `maxps`/`minps` compute). Softmax runs 17% faster on Cranelift and 13% faster on LLVM with it; the [kernel benchmarks](kernels.md) use it. Code can get the same instruction without fast math by writing the compare and select itself, as C's `v < lo ? lo : v` does: `m = vlt v, vlo:f32` then `r = vsel m, vlo, v` (scalar: `c = lt v, lo`, `r = select c, lo, v`). A clamp of 262,144 floats written that way takes 15.3 µs on LLVM (Ryzen 7 8845HS), against 27.6 µs with `vmax`/`vmin`. Fast math also lets bf16 `mm` use AMX (`tdpbf16ps`), SME (`bfmopa`) and AVX512-BF16 (`vdpbf16ps`), which treat bf16 subnormal inputs as zero; by default bf16 `mm` uses exact vector FMAs on those CPUs, so it widens every bf16 value exactly on every backend (int8, f16 and f32 `mm` use the matrix engines either way).

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

Masking every iteration costs nothing extra: before code generation, a loop whose masked counts are `n - i` (with `i` a parameter of the loop header), a `vl`, or the `min` of those, gets a copy in front of it that loads and stores whole vectors while `n - i` covers them, and the original loop runs the tail (`achainsaw opt` shows the copy). Each iteration computes what it did before, on both backends, and the copy never touches memory past the buffers.

**Gather.** `vgather` reads one element per lane from addresses that vector code computed: a lookup table indexed by data, or `x[cols[k]]` in a sparse product. It reads what scalar `ld`s of `p[idx[i]]` would, in any order; with a count, lanes from `n` on are not read, so their indices may be anything. In the sandbox each lane read is checked, and an index out of bounds reports the first such lane. LLVM emits one instruction where the target has a fast one: `vgatherdps`/`vgatherqpd` (and their integer forms) with AVX-512, and on SVE `ld1w`/`ld1d` with vector offsets (`[x0, z0.s, sxtw #2]` for 32-bit indices); for AVX2 CPUs that LLVM does not tune for fast gathers, and on NEON, it loads lane by lane. Cranelift has no gather instruction and always loads lane by lane. A count of `n - i`, as for `ldm`, is dropped in the full-vector copy of the loop.

**Scatter.** `vscatter` is the store counterpart: it writes what scalar `st`s of lane `i` to `p[idx[i]]` would, in lane order, so lanes with the same index leave the highest lane's value (a permutation, or an inverse index such as `pos[keys[i]] = i`; for a histogram, where lanes that collide must add up, use `atomic add` or per-lane counts). With a count, lanes from `n` on are not stored and their indices may be anything. In the sandbox every stored lane is checked before any is written, so an index out of bounds (or into read-only `data`) reports the first such lane and leaves memory as it was. LLVM emits `vscatterdps`/`vscatterqpd` (and their integer forms) with AVX-512 and `st1w`/`st1d` with vector offsets on SVE, and stores lane by lane elsewhere; Cranelift always stores lane by lane. A count of `n - i` is dropped in the full-vector copy of the loop, as for `vgather`. Scattering 1M floats by a permutation that keeps them close (each block of 16 reversed) takes 744 µs on Cranelift and 588 µs on LLVM against the scalar loop's 1801 and 1488 µs (one thread, Ryzen 7 8845HS); by a random permutation the stores wait on memory: 693 and 669 µs against 680 and 826 µs. NumPy's `y[ix] = x` takes 2.2 ms for both.

Gathering pays off when the elements are close: a lookup of 1M floats in a 256-entry table (`k = vftoi` of the scaled input, then `vgather`) runs 1.5x faster than the scalar loop on Cranelift and 1.2x on LLVM. When they are scattered over memory, the loads wait on the cache however they are issued: [`spmv.air`](../examples/kernels/spmv.air)'s `spmv_gather` takes 4% longer than the scalar `spmv` with 16 threads, on both backends ([docs/kernels.md](kernels.md)).

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

It holds the building blocks the example kernels share: `quantize_q8` (Q8_0 activations), `qmat_chunk` (64 output rows of a Q8_0, Q4_0, Q6_K, Q4_K, Q5_K, Q4_1 or IQ4_NL matrix, which may be stored in parts of different formats), `rope_half`, `exp_shift` (softmax numerators) and `silu_mul` (SwiGLU), and `atomic_add_f32` / `atomic_add_f64` for floats that `par` calls share ([Atomics](#atomics)). `achainsaw std` lists them with their documentation and `achainsaw std <name>` prints one's source; MCP clients have the `air_std` tool.

**Matrix multiply:** `mm pc, pa, pb, m, n, k:bf16` computes `C[m x n] += A[m x k] * B[k x n]` on row-major, contiguous matrices. `A`/`B` hold `bf16`, `f16`, or `f32` elements with an `f32` `C`, or `i8` elements with an `i32` `C` (exact, wrapping). Float accumulation order is implementation-defined, so results agree across backends within rounding error rather than bit for bit. Under a fuel budget, `mm` costs one unit per 1024 multiply-adds, charged before it starts. Each backend's kernels, AMX and SME among them, are in [Backends](backends.md#matrix-multiply-mm).

## Math functions

`y = exp x`, `log`, `tanh`, `sin` and `cos` take an f32, and `vexp`, `vlog`, `vtanh`, `vsin` and `vcos` the f32 lanes of a vector (`r = vsin v:f32`). Each is a fixed sequence of AIR operations: Cephes' f32 polynomials evaluated by FMAs, with the exponent and quadrant handled by bit operations, and no libm call. The sequence is written once, in `achainsaw_ir::math`, and replaces the instruction before code generation, so the scalar and vector forms give the same bits on both backends and every target, and the optimizer folds a function of a constant to that same value.

| Function | Largest error | Special values and range |
|---|---|---|
| `exp` | 1.26 ulp on [-87, 88] | The input is clamped to [-87, 88] first, so `exp` of a smaller input is e^-87 (1.6e-38, not 0) and of a larger one e^88 (1.7e38, not inf). NaN gives NaN |
| `log` | 0.83 ulp | log(±0) = -inf, log(+inf) = +inf; negative inputs and NaN give NaN. The bound holds for subnormal inputs too |
| `tanh` | 1.36 ulp | Keeps the sign of zero; ±inf gives ±1, NaN gives NaN |
| `sin`, `cos` | 2.07 ulp (`sin`) and 1.81 ulp (`cos`) for \|x\| ≤ 2^20; up to 53 ulp toward 2^22 | NaN for \|x\| > 2^22 (4194304), where the quadrant can no longer be rounded exactly, and for ±inf and NaN; sin(-0) = -0 |

The errors are the measured worst cases over every f32 input, against f64; `crates/achainsaw-ir/tests/math.rs` checks them on a sample. They are f32 only: `fext` and `ftrunc` convert f64 values.

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

## Function handles

`h = fnref f` gives the handle of the module's function `f`: an `i64` that can be stored in memory, passed, returned and compared, like a function pointer that never exposes a code address. `call_indirect` calls the function whose handle it is given, with the result types it declares:

```air
# A dispatch table filled at run time, then a call through entry k.
fn apply(t:ptr, k:i64, x:i64)->i64
  b0:
    h = ld t[k]:i64
    r = call_indirect h(x)->i64
    ret r
```

`q, m = call_indirect h(a, b)->(i64, i64)` takes several results, and `call_indirect h(p, x)` none. `par n, call_indirect h(a, b)` runs a `par` over the function a handle names, which must take `(i64, ...)` and return nothing; it uses the same fuel as the direct `par`. Handles are 1 plus the function's index in the module (so 0 is never one), valid in the module that takes them. Every call checks its handle: it must name a function of the module whose parameter types are the arguments' and whose results are the declared ones. Otherwise nothing is called, the results are zero, and the run reports `ERR_BAD_INDIRECT_CALL` with the handle (in the sandbox and out of it; AOT objects, which have no runtime, trap). Each backend keeps the module's functions in a table of address and signature id, and a call loads both after the range check, so a forged handle can never jump to anything but a function of the module with the right types. A `call_indirect` whose handle comes from a `fnref` in the same function becomes a direct `call` before code generation (and in `achainsaw opt`), so it can be inlined. Calls through handles cost what calls cost: no fuel themselves, and the callee's checks in the sandbox (stack depth, so recursion through handles stops at the same limit). `fnref` names AIR functions only, not `extfn`s. `fnref` and `call_indirect` are keywords only where an instruction is expected, so they remain valid register names.

## Constant data and stack buffers

**Constant data.** A top-level `data` declaration gives the module a read-only table: an element type (`i8`, `i16`, `i32`, `i64`, `f32` or `f64`) and its values, stored little-endian and 64-byte aligned. `p = addr name` gives its address, which code reads like any other memory. A line that ends with a comma continues on the next, and the disassembler prints 16 values per line. [`crc32.air`](../examples/crc32.air) carries its 256-entry lookup table this way:

```air
data inv_freq:f32 = 1.0, 0.5, 0.25, 0.125,
    0.0625, 0.03125, 0.015625, 0.0078125

fn freq(i:i64)->f32
  b0:
    t = addr inv_freq
    f = ld t[i]:f32
    ret f
```

Integers may be written signed or unsigned (`255` and `-1` are the same `i8`), and an integer of a float type is that float. A program must not write to its data: in the sandbox (MCP `air_run`) a store, atomic, `stm` or `mm` result there is an `ERR_MEMORY_VIOLATION` ("is to read-only data"), and outside it the behavior is undefined (Cranelift places tables in read-only memory, where a write faults; LLVM may assume they never change). The sandbox copies each module's tables into the start of its arena, so they count against its size.

**Stack buffers.** `p = stack n` gives the address of an `n`-byte buffer (a constant from 1 to 65536) in the function's frame, 16-byte aligned, with unspecified contents. Each `stack` instruction is its own buffer: the same one every time it runs during a call, distinct in every call (recursion, each `par` index), and gone when the function returns, so it must not be returned or kept. Pointers to it may be passed to callees and to the calls of a `par` that the function waits for. An inline function's buffers become its caller's. Unlike `alloc`, a buffer costs no call into the runtime and no `free`; [`llama_decode.air`](../examples/kernels/llama_decode.air) keeps its per-task 64-row results in them, and [`histogram.air`](../examples/kernels/histogram.air) its private count tables. Larger or variable-sized buffers still take `alloc`. In the sandbox, whose checks admit only arena memory, a function's buffers come from its arena on entry and go back on return.

`data`, `addr` and `stack` remain valid register and label names.

## Atomics

`par` calls that update the same memory do it through atomic accesses, on integers (`i8` to `i64`):

| Instruction | Effect |
|---|---|
| `old = atomic add p, v` | Stores `old + v` and returns `old`; also `sub`, `and`, `or`, `xor`, `xchg` (stores `v`), `min`, `max` (signed), `umin`, `umax` (unsigned) |
| `old = atomic cas p, expected, new` | Stores `new` if the value is `expected`, and returns the value it found: the swap happened when `old == expected` |
| `v = atomic ld p:i64` | Reads the value |
| `atomic st p, v` | Writes the value |

Every atomic access is sequentially consistent: all threads see them in one order, and a value stored with `st` before an `atomic st` is visible to a thread whose `atomic ld` reads that store. `par` returns only after every call has finished, so the caller reads results with plain `ld`. There is no separate fence. The address must be aligned to the integer's size; in the sandbox (MCP `air_run`), a misaligned atomic is an `ERR_MEMORY_VIOLATION`, and outside it the hardware may fault. Atomics take indexed operands like `ld` and `st` (`atomic add hist[b], 1:i64` adds to `hist + 8b`), and `atomic` remains a valid register name elsewhere.

Floats have no atomic arithmetic. The standard library's `atomic_add_f32` and `atomic_add_f64` (`use atomic_add_f32`, then `old = call atomic_add_f32(total, x)`; inline functions, so no call is made) update the bits with a `cas` loop, which retries when another call changed the value in between; written out, for other operations:

```air
# total += x for an f32 total that other par calls update too
fn add_total(total:ptr, x:f32)
  b0:
    jmp retry
  retry:
    bits = atomic ld total:i32
    f = bitcast bits:f32
    g = add f, x
    nb = bitcast g:i32
    seen = atomic cas total, bits, nb
    ok = eq seen, bits
    br ok, done, retry
  done:
    ret
```

A work queue is a counter that calls take items from, `i = atomic add next, 1:i64`, until `i` reaches the item count. `par` already hands out its indices this way, so a queue of your own gains nothing over a `par` with one index per item: in [`spmv.air`](../examples/kernels/spmv.air), whose rows' costs differ by four orders of magnitude, both take the same time ([docs/kernels.md](kernels.md)). Each backend uses the target's atomic instructions (x86 `lock` prefixes, AArch64 LSE when the target has it, else load-exclusive/store-exclusive loops), and `crates/achainsaw-codegen/tests/atomics.rs` checks every op at every width on both.
