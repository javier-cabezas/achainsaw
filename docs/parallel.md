# Parallel loops (`par`)

`par n, f(a, b)` calls `f(i, a, b)` for every `i` in `0..n` across all cores and returns when every call has finished. It is the only parallel construct: no threads, futures, locks or closures. `f` is an ordinary AIR function whose first parameter is the `i64` index, followed by scalar or `ptr` parameters, and that returns nothing. Values the body needs are passed as arguments, exactly like `call`. From [`examples/kernels/gemv_par.air`](../examples/kernels/gemv_par.air):

```air
fn gemv_par(a:ptr, x:ptr, y:ptr, m:i64, k:i64)
  b0:
    m15 = add m, 15:i64
    blocks = udiv m15, 16:i64
    par blocks, gemv_rows(a, x, y, m, k)    # gemv_rows(blk, a, x, y, m, k) does rows 16*blk..16*blk+15
    ret
```

The rules:
- **Write only your own memory, or use atomics.** Calls run concurrently and in any order. Each `i` writes its own memory with `ld`/`st`; to reduce, store per-`i` partial results in an `alloc`'d array and sum them after the `par`. Memory that several calls update, such as a counter, a histogram or a running minimum, takes [atomic accesses](language.md#atomics): `atomic add hist[b], 1:i64`. A plain `ld` of a location another call writes concurrently gives an unspecified value. Keep atomics on shared locations rare: each one moves the cache line between cores. [`histogram.air`](../examples/kernels/histogram.air) counts each 64 KB chunk privately and merges it with one `atomic add` per bin, 70–80x faster than one per byte ([docs/kernels.md](kernels.md)).
- **Never wait for another index.** A call may spin on an atomic only for work it can finish itself (a `cas` retry). Waiting for another `i` to set a flag can hang: calls may run one after another on a single thread (`--threads 1`, a nested `par`, AOT code).
- **Give each `i` real work.** A row or a block of thousands of elements works well. Dispatching a `par` costs about a microsecond on 16 threads.
- **The validator checks the body.** `ERR_PAR_SIGNATURE` reports a body with the wrong signature and gives the expected one in `context.expected_signature`. External functions (`extfn`) cannot be bodies, and vectors cannot be passed (pass them through memory).

How it runs:
- **Threads.** The calling thread works through the indices together with a process-wide pool of helper threads. Indices are handed out dynamically, so uneven iterations still balance. Starting and joining a `par` takes no locks or allocations, and helpers still waking up when the indices run out skip that `par` instead of delaying it. The pool has `ACHAINSAW_THREADS` threads in all (default: one per core). Helpers that just ran iterations spin for 200 µs before parking, so back-to-back `par` loops do not pay thread wake-ups; the window is short because spinning threads slow down other runtimes' thread pools (OpenBLAS, OpenMP) running in the same process. A `par` with fewer indices than threads uses only that many threads: the other helpers stay parked, so they neither slow down busy threads sharing their core nor need waking for later loops of the same size.
- **Thread cap.** Limit a run with `--threads` on `achainsaw run`, `threads` in MCP `air_run`, `Kernel.set_threads()` in Python, or `JitEngine::set_threads`. `1` runs serially, which makes it easy to measure the speedup.
- **Serial fallbacks.** A `par` inside a `par` body runs serially on its worker. So does a `par` started while another thread's `par` has the pool. AOT objects have no runtime, so `par` compiles to a plain loop there; any order is a valid execution.
- **Fuel.** The fuel budget is shared: each index costs one unit, plus the usual unit per branch. A parallel run therefore uses exactly the fuel of a serial one, and a runaway iteration still ends with `ERR_OUT_OF_FUEL`.
- **Errors and sandbox.** The first failure in any worker stops all of them and is reported from the `par`: `ERR_MEMORY_VIOLATION`, `ERR_STACK_OVERFLOW` or `ERR_OUT_OF_MEMORY`. Sandboxed bodies share the caller's arena and may `alloc`/`free` scratch memory. Each call has its own `stack` buffers, and a caller may pass its own to the calls of a `par` it waits for (as `spmv_queue` does with its counter).
- **Python.** `Kernel.run` releases the GIL, so host callbacks can run on `par` workers, and other Python threads keep running.

Speedup of `gemv_par`'s `bench` driver on a Ryzen 7 8845HS (8 cores, 16 threads), from `achainsaw run examples/kernels/gemv_par.air --func bench --args M,K,REPS --threads T`:

| Shape | Backend | 1 thread | 16 threads | Speedup |
|---|---|---|---|---|
| 4096x4096 (64 MB), 50 runs | Cranelift | 193 ms | 77 ms | 2.5x (memory-bound) |
| 4096x4096 (64 MB), 50 runs | LLVM (AVX-512) | 148 ms | 74 ms | 2.0x (memory-bound) |
| 1024x512 (2 MB), 2000 runs | Cranelift | 90 ms | 17 ms | 5.3x |
| 1024x512 (2 MB), 2000 runs | LLVM (AVX-512) | 43 ms | 9 ms | 4.8x (2000 `par` dispatches) |
