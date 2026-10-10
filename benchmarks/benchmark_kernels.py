"""
Chainsaw-BLAS: verification and benchmarks for the kernels in examples/kernels/,
including flash attention at DeepSeek V4 Pro's decode size.

Every kernel is checked against NumPy, then timed on each available code generation
backend (Cranelift, and LLVM when the build has it) and, with --isa all, at every ISA
level this machine reaches (for example sse, avx, avx2 and avx512 on an AVX-512 host).
The kernels are vector-length agnostic, so the same source runs 128-bit vectors on
Cranelift and up to 512-bit (or SVE-scalable) vectors on LLVM. Kernels compile with
fast_math=True (float min/max as compare and select; the tests cover both modes).

NumPy and, when it is installed, PyTorch (eager, CPU) are timed on the same work as
baselines. NumPy computes with the kernels' data types. PyTorch runs its own fastest CPU
path for each kernel, and its results are checked too. Where its types differ, the case
says so: bf16 outputs for GEMM and attention, and for the quantized GEMVs PyTorch's
weight-only int4 and int8 kernels, which take bf16 activations instead of int8 ones (int8
with one scale per row, since PyTorch has no per-block int8 kernel).

    python benchmarks/benchmark_kernels.py                  # host ISA, all backends
    python benchmarks/benchmark_kernels.py --isa all        # sweep ISA levels too
    python benchmarks/benchmark_kernels.py --quick          # short run (CI)
    python benchmarks/benchmark_kernels.py --json out.json  # machine-readable results
    python benchmarks/benchmark_kernels.py --no-torch       # skip the PyTorch baseline

Exits non-zero if any kernel, or a PyTorch baseline, disagrees with the reference.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None

# Ensure achainsaw module is accessible
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import achainsaw

KERNELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "examples", "kernels"))

ISA_ORDER = {
    "x86_64": ["sse", "avx", "avx2", "avx512", "amx"],
    "aarch64": ["neon", "sve", "sve2", "sme"],
}


def source(name):
    with open(os.path.join(KERNELS_DIR, name + ".air"), "r", encoding="utf-8") as f:
        return f.read()


def bf16_bits(x):
    """Round f32 values to bfloat16 (nearest-even) and return the raw u16 bits."""
    b = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return ((b + 0x7FFF + ((b >> 16) & 1)) >> 16).astype(np.uint16)


def bf16_values(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)


def q8_quantize(x):
    """Q8_0 activations exactly as the kernels' quantize_q8: per 32-block scale
    max|x| / 127 and values rounded half away from zero, in f32. Returns the integer values
    (as f32) and the scales."""
    blocks = x.reshape(-1, 32).astype(np.float32)
    d = np.abs(blocks).max(axis=1) / np.float32(127.0)
    with np.errstate(divide="ignore"):
        inv = np.where(d > 0, np.float32(1.0) / d, np.float32(0.0)).astype(np.float32)
    v = blocks * inv[:, None]
    t = np.trunc(v)
    # Halves away from zero, exactly (C's roundf): every step is exact in f32.
    return (t + np.where(np.abs(v - t) >= np.float32(0.5), np.sign(v), np.float32(0.0))).reshape(-1), d


def np_q8_gemv(q_blocks, scales, x):
    """NumPy Q8_0 GEMV with the kernel's data types: int8 weights q_blocks [k/32 x m x 32],
    f32 scales [k/32 x m], int8-quantized activations, exact integer block dots (computed as a
    batched f32 matmul, exact because every block dot is below 2^24), f32 accumulation."""
    xq, dx = q8_quantize(x)
    nb = q_blocks.shape[0]
    dots = np.matmul(q_blocks.astype(np.float32), xq.reshape(nb, 32, 1))[:, :, 0]
    return ((dots * dx[:, None]) * scales).sum(axis=0)


def rel_err(out, ref):
    """max |out - ref| / max |ref|, for a PyTorch result against the case's f64 reference."""
    out = np.asarray(out.float() if torch is not None and isinstance(out, torch.Tensor) else out,
                     dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    return float(np.max(np.abs(out.reshape(ref.shape) - ref)) / np.max(np.abs(ref)))


def torch_bf16(bits):
    """A PyTorch bf16 tensor over raw bf16 bits (zero copy)."""
    return torch.from_numpy(np.ascontiguousarray(bits).view(np.int16)).view(torch.bfloat16)


def time_call(func, budget_s, min_iters=5, chunks=5):
    """Seconds per call after a short warm-up: the run of about `budget_s` is split into
    `chunks` and the fastest chunk's average is returned, so a transient stall (another
    process, a migrated thread) does not count against the kernel."""
    for _ in range(3):
        func()
    best = float("inf")
    for _ in range(chunks):
        iters, t0 = 0, time.perf_counter()
        while True:
            func()
            iters += 1
            elapsed = time.perf_counter() - t0
            if iters >= min_iters and elapsed >= budget_s / chunks:
                break
        best = min(best, elapsed / iters)
    return best


def tt(x):
    """A PyTorch tensor sharing x's memory, or None without PyTorch (its cases' torch
    callables are then never called)."""
    return torch.from_numpy(x) if torch is not None else None


def make_cases():
    """Each case: name, label, kernel source, args builder, verify(kernel) -> max error,
    tolerance, NumPy baseline, PyTorch baseline with its check(result) -> error (and its own
    tolerance where its types differ), and the work per call (for GFLOP/s)."""
    rng = np.random.default_rng(42)
    cases = []

    dim = 1024  # embedding size of common text-embedding models
    a = rng.standard_normal(dim).astype(np.float32)
    b = rng.standard_normal(dim).astype(np.float32)
    ta, tb = tt(a), tt(b)
    cos_ref = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    cases.append(dict(
        name="cosine_similarity", label=f"Cosine similarity (n={dim})",
        run=lambda k: k(a, b, dim),
        verify=lambda k: abs(k(a, b, dim) - cos_ref), tol=1e-5,
        numpy=lambda: np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12),
        torch=lambda: F.cosine_similarity(ta, tb, dim=0),
        torch_check=lambda out: abs(float(out) - cos_ref),
        flops=6 * dim,
    ))

    l2_ref = float(np.linalg.norm(a - b))
    cases.append(dict(
        name="euclidean_distance", label=f"Euclidean distance (n={dim})",
        run=lambda k: k(a, b, dim),
        verify=lambda k: abs(k(a, b, dim) - l2_ref) / l2_ref, tol=1e-5,
        numpy=lambda: np.linalg.norm(a - b),
        torch=lambda: torch.dist(ta, tb),
        torch_check=lambda out: abs(float(out) - l2_ref) / l2_ref,
        flops=3 * dim,
    ))

    sm_n = 1000
    x_sm = (rng.standard_normal(sm_n) * 4.0).astype(np.float32)
    out_sm = np.zeros(sm_n, dtype=np.float32)
    t_sm = tt(x_sm)
    e = np.exp(x_sm.astype(np.float64) - x_sm.max())
    sm_ref = e / e.sum()

    def verify_softmax(k):
        k(x_sm, out_sm, sm_n)
        return float(np.max(np.abs(out_sm - sm_ref)))

    def np_softmax():
        z = np.exp(x_sm - x_sm.max())
        return z / z.sum()

    cases.append(dict(
        name="softmax", label=f"Softmax (n={sm_n})",
        run=lambda k: k(x_sm, out_sm, sm_n),
        verify=verify_softmax, tol=1e-6, numpy=np_softmax,
        torch=lambda: torch.softmax(t_sm, 0),
        torch_check=lambda out: float(np.max(np.abs(out.numpy() - sm_ref))),
        flops=4 * sm_n,
    ))

    rms_n = 4096
    x_r = rng.standard_normal(rms_n).astype(np.float32)
    w_r = rng.uniform(0.8, 1.2, rms_n).astype(np.float32)
    out_r = np.zeros(rms_n, dtype=np.float32)
    t_xr, t_wr = tt(x_r), tt(w_r)
    xd = x_r.astype(np.float64)
    rms_ref = xd / np.sqrt(np.mean(xd * xd) + 1e-6) * w_r

    def verify_rms(k):
        k(x_r, w_r, out_r, rms_n)
        return float(np.max(np.abs(out_r - rms_ref)))

    cases.append(dict(
        name="rmsnorm", label=f"RMSNorm (n={rms_n})",
        run=lambda k: k(x_r, w_r, out_r, rms_n),
        verify=verify_rms, tol=1e-5,
        numpy=lambda: x_r / np.sqrt(np.mean(x_r * x_r) + 1e-6) * w_r,
        torch=lambda: F.rms_norm(t_xr, (rms_n,), t_wr, eps=1e-6),
        torch_check=lambda out: float(np.max(np.abs(out.numpy() - rms_ref))),
        flops=4 * rms_n,
    ))

    gm, gk = 512, 1024
    A = rng.standard_normal((gm, gk)).astype(np.float32)
    xg = rng.standard_normal(gk).astype(np.float32)
    yg = np.zeros(gm, dtype=np.float32)
    tA, txg = tt(A), tt(xg)
    gemv_ref = A.astype(np.float64) @ xg.astype(np.float64)

    def verify_gemv(k):
        k(A, xg, yg, gm, gk)
        return float(np.max(np.abs(yg - gemv_ref)) / np.max(np.abs(gemv_ref)))

    cases.append(dict(
        name="gemv_f32", label=f"GEMV f32 ({gm}x{gk})",
        run=lambda k: k(A, xg, yg, gm, gk),
        verify=verify_gemv, tol=1e-5, numpy=lambda: A @ xg,
        torch=lambda: torch.mv(tA, txg), torch_check=lambda out: rel_err(out, gemv_ref),
        flops=2 * gm * gk,
    ))

    def verify_gemv_par(k):
        yg[:] = np.nan
        k.run("gemv_par", A, xg, yg, gm, gk)
        return float(np.max(np.abs(yg - gemv_ref)) / np.max(np.abs(gemv_ref)))

    cases.append(dict(
        name="gemv_par", label=f"GEMV f32, all cores via par ({gm}x{gk})",
        run=lambda k: k.run("gemv_par", A, xg, yg, gm, gk),
        verify=verify_gemv_par, tol=1e-5, numpy=lambda: A @ xg,
        torch=lambda: torch.mv(tA, txg), torch_check=lambda out: rel_err(out, gemv_ref),
        flops=2 * gm * gk,
    ))

    n = 256
    Ab = bf16_bits(rng.standard_normal((n, n)))
    Bb = bf16_bits(rng.standard_normal((n, n)))
    Af, Bf = bf16_values(Ab), bf16_values(Bb)
    C = np.zeros((n, n), dtype=np.float32)
    if torch is not None:
        tAb, tBb = torch_bf16(Ab), torch_bf16(Bb)
    gemm_ref = Af.astype(np.float64) @ Bf.astype(np.float64)

    def verify_gemm(k):
        C[:] = 0.0
        k.run("gemm_bf16", C, Ab, Bb, n, n, n)
        return float(np.max(np.abs(C - gemm_ref)) / np.max(np.abs(gemm_ref)))

    cases.append(dict(
        name="gemm_bf16", label=f"GEMM bf16->f32 ({n}x{n}x{n})",
        run=lambda k: k.run("gemm_bf16", C, Ab, Bb, n, n, n),
        verify=verify_gemm, tol=1e-4,
        # Same types as the kernel: bf16 inputs (raw bits; NumPy has no bf16 matmul, so
        # they are widened to f32 in the call), f32 accumulation.
        numpy=lambda: bf16_values(Ab) @ bf16_values(Bb),
        # PyTorch multiplies bf16 natively (f32 accumulation), rounding its output to bf16.
        torch=lambda: torch.mm(tAb, tBb), torch_check=lambda out: rel_err(out, gemm_ref),
        torch_tol=1e-2, flops=2 * n * n * n,
    ))
    # One decode step of DeepSeek V4 Pro's sparse attention: 128 query heads share one
    # 512-dim KV head (K = V) and attend to 1152 selected cache entries (sliding window 128
    # plus the indexer's top-1024), with a per-head attention sink.
    fh, fd, fcache, fnk = 128, 512, 4096, 1152
    fq = bf16_bits(rng.standard_normal((fh, fd)))
    fkv = bf16_bits(rng.standard_normal((fcache, fd)))
    fidx = rng.choice(fcache, size=fnk, replace=False).astype(np.int32)
    fsink = rng.standard_normal(fh).astype(np.float32)
    fout = np.zeros((fh, fd), dtype=np.float32)
    fscale = np.float32(1.0 / np.sqrt(fd))
    qf, kvf = bf16_values(fq).astype(np.float64), bf16_values(fkv).astype(np.float64)
    sel = kvf[fidx]
    scores = (qf @ sel.T) * float(fscale)
    mx = scores.max(axis=1, keepdims=True)
    pr = np.exp(scores - mx)
    den = pr.sum(axis=1, keepdims=True) + np.exp(fsink[:, None].astype(np.float64) - mx)
    flash_ref = (pr @ sel) / den

    def verify_flash(k):
        k.run("flash_attention", fq, fkv, fidx, fsink, fout, fh, fd, fnk, fscale)
        return float(np.max(np.abs(fout - flash_ref)) / np.max(np.abs(flash_ref)))

    def np_flash():
        # Same types as the kernel: bf16 q and kv (widened to f32 here), f32 scores and
        # softmax, probabilities rounded to bf16 for the product with V. The -1 entries are
        # all valid here, so no masking is needed.
        q32, sel32 = bf16_values(fq), bf16_values(fkv[fidx])
        s = (q32 @ sel32.T) * fscale
        m = s.max(axis=1, keepdims=True)
        e = np.exp(s - m)
        p = bf16_values(bf16_bits(e))
        return (p @ sel32) / (e.sum(axis=1, keepdims=True) + np.exp(fsink[:, None] - m))

    if torch is not None:
        # PyTorch's scaled_dot_product_attention in bf16 (f32 softmax inside, bf16 output),
        # all heads as the queries of one KV head. The sink is one more key: a zero row
        # appended to the cache (score 0) plus a mask of the sink logit on its column.
        t_q = torch_bf16(fq).view(1, 1, fh, fd)
        t_kv = torch.cat([torch_bf16(fkv), torch.zeros(1, fd, dtype=torch.bfloat16)])
        t_idx = torch.from_numpy(np.append(fidx, fcache).astype(np.int64))
        t_mask = torch.zeros(1, 1, fh, fnk + 1, dtype=torch.bfloat16)
        t_mask[..., -1] = tt(fsink)

    def torch_flash():
        sel = t_kv[t_idx].view(1, 1, fnk + 1, fd)
        return F.scaled_dot_product_attention(t_q, sel, sel, attn_mask=t_mask,
                                              scale=float(fscale))

    cases.append(dict(
        name="flash_attention", label=f"Flash attention decode, DeepSeek V4 Pro ({fh}x{fd}, {fnk} keys)",
        run=lambda k: k.run("flash_attention", fq, fkv, fidx, fsink, fout, fh, fd, fnk, fscale),
        verify=verify_flash, tol=1e-2, numpy=np_flash,
        torch=torch_flash, torch_check=lambda out: rel_err(out, flash_ref),
        flops=2 * 2 * fh * fnk * fd,
    ))

    # Decode-step glue at Llama 3 8B sizes (d = 4096, MLP 14336, 32 heads of 128,
    # vocabulary 128256).
    sw_n = 14336
    gate = (rng.standard_normal(sw_n) * 3.0).astype(np.float32)
    up = rng.standard_normal(sw_n).astype(np.float32)
    sw_out = np.zeros(sw_n, dtype=np.float32)
    t_gate, t_up = tt(gate), tt(up)
    gd = gate.astype(np.float64)
    sw_ref = gd / (1.0 + np.exp(-gd)) * up

    def verify_swiglu(k):
        k(gate, up, sw_out, sw_n)
        return float(np.max(np.abs(sw_out - sw_ref) / (np.abs(sw_ref) + 1e-30)))

    cases.append(dict(
        name="swiglu", label=f"SwiGLU, vectorized exp (n={sw_n})",
        run=lambda k: k(gate, up, sw_out, sw_n),
        verify=verify_swiglu, tol=2e-6,
        numpy=lambda: gate / (1.0 + np.exp(-gate)) * up,
        torch=lambda: F.silu(t_gate) * t_up,
        torch_check=lambda out: float(np.max(np.abs(out.numpy() - sw_ref) / (np.abs(sw_ref) + 1e-30))),
        flops=8 * sw_n,
    ))

    vocab = 128256
    logits = rng.standard_normal(vocab).astype(np.float32)
    am_ref = int(np.argmax(logits))
    t_logits = tt(logits)
    cases.append(dict(
        name="argmax", label=f"Greedy argmax (vocab={vocab})",
        run=lambda k: k(logits, vocab),
        verify=lambda k: float(k(logits, vocab) != am_ref), tol=0.0,
        numpy=lambda: np.argmax(logits),
        # torch.max along the dimension finds the index faster than torch.argmax.
        torch=lambda: torch.max(t_logits, 0),
        torch_check=lambda out: float(int(out.indices) != am_ref),
        flops=vocab,
    ))

    rh, rd, rpos = 32, 128, 1000
    rhalf = rd // 2
    ang = rpos * 500000.0 ** (-2.0 * np.arange(rhalf) / rd)
    rcos, rsin = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
    rq0 = rng.standard_normal((rh, rd)).astype(np.float32)
    rq = rq0.copy()
    x1, x2 = rq0[:, :rhalf].astype(np.float64), rq0[:, rhalf:].astype(np.float64)
    rope_ref = np.concatenate([x1 * rcos - x2 * rsin, x2 * rcos + x1 * rsin], axis=1)

    def verify_rope(k):
        rq[:] = rq0
        k(rq, rh, rd, rcos, rsin)
        return float(np.max(np.abs(rq - rope_ref)))

    def np_rope():
        a, b = rq0[:, :rhalf], rq0[:, rhalf:]
        return np.concatenate([a * rcos - b * rsin, b * rcos + a * rsin], axis=1)

    # PyTorch has no RoPE operator; models write it out like this.
    t_rq0, t_rcos, t_rsin = tt(rq0), tt(rcos), tt(rsin)

    def torch_rope():
        a, b = t_rq0[:, :rhalf], t_rq0[:, rhalf:]
        return torch.cat([a * t_rcos - b * t_rsin, b * t_rcos + a * t_rsin], dim=1)

    cases.append(dict(
        name="rope", label=f"RoPE, in place ({rh} heads x {rd})",
        run=lambda k: k(rq, rh, rd, rcos, rsin),
        verify=verify_rope, tol=1e-5, numpy=np_rope,
        torch=torch_rope, torch_check=lambda out: float(np.max(np.abs(out.numpy() - rope_ref))),
        flops=6 * rh * rhalf,
    ))

    an = 4096
    # The four buffers are slices of one block, 17 cache lines apart beyond their 16 KB. Left
    # to the allocator, `out` landed exactly 64 KB + 48 bytes after `res`, and this Zen 4
    # then ran the kernel 2x slower in most processes (1.5 vs 3.1 µs): the second pass
    # stores out[i] while loading res a few lanes ahead, apparently taken by the CPU as a
    # possible alias of the load. Fixed offsets keep the timing the same from run to run.
    gap = an + 17 * 16
    block = np.zeros(4 * gap, dtype=np.float32)
    ax, ares, aw, aout = (block[i * gap:i * gap + an] for i in range(4))
    ax[:] = rng.standard_normal(an)
    ares0 = rng.standard_normal(an).astype(np.float32)
    ares[:] = ares0
    aw[:] = rng.uniform(0.8, 1.2, an)
    asum = ares0.astype(np.float64) + ax
    t_ax, t_ares0, t_aw = tt(ax), tt(ares0), tt(aw)
    an_ref = asum / np.sqrt(np.mean(asum * asum) + 1e-5) * aw

    def verify_add_rmsnorm(k):
        ares[:] = ares0
        k(ax, ares, aw, aout, an, 1e-5)
        return float(np.max(np.abs(aout - an_ref)))

    def np_add_rmsnorm():
        r = ares0 + ax
        return r / np.sqrt(np.mean(r * r) + 1e-5) * aw

    cases.append(dict(
        name="add_rmsnorm", label=f"Residual add + RMSNorm (n={an})",
        run=lambda k: k(ax, ares, aw, aout, an, 1e-5),
        verify=verify_add_rmsnorm, tol=1e-5, numpy=np_add_rmsnorm,
        torch=lambda: F.rms_norm(t_ares0 + t_ax, (an,), t_aw, eps=1e-5),
        torch_check=lambda out: float(np.max(np.abs(out.numpy() - an_ref))),
        flops=5 * an,
    ))

    qm, qk = 4096, 4096
    qnb = qk // 32
    qq = rng.integers(-127, 128, size=(qm, qk), dtype=np.int8)
    # f16 scales, as GGUF stores them (NumPy uses the same values in f32).
    qs = (rng.uniform(0.5, 1.0, size=(qnb, qm)) / (73.0 * np.sqrt(qk))).astype(np.float16)
    # Packed layout: 64-row chunks; per 32-block, 4 consecutive values of a row per 32-bit
    # lane, lanes ordered by group then row, values + 128 as unsigned bytes (see qmat_chunk).
    q_packed = np.ascontiguousarray(
        (qq.view(np.uint8) ^ np.uint8(0x80)).reshape(qm // 64, 64, qnb, 8, 4).transpose(0, 2, 3, 1, 4))
    s_packed = np.ascontiguousarray(qs.reshape(qnb, qm // 64, 64).transpose(1, 0, 2))
    qs = qs.astype(np.float32)
    qx = rng.standard_normal(qk).astype(np.float32)
    qy = np.zeros(qm, dtype=np.float32)
    # NumPy keeps the same int8 weights, block-major ([k/32 x m x 32]).
    q_blocks = np.ascontiguousarray(qq.reshape(qm, qnb, 32).transpose(1, 0, 2))
    xq_ints, dxs = q8_quantize(qx)
    dots = np.einsum("rbt,bt->rb", qq.reshape(qm, qnb, 32).astype(np.int64),
                     xq_ints.reshape(qnb, 32).astype(np.int64))
    q8_ref = (dots * qs.T.astype(np.float64) * dxs.astype(np.float64)).sum(axis=1)

    # PyTorch's int8 weight-only GEMV (torchao's CPU path): the same int8 weights, but one
    # scale per row (each row's mean block scale here; it has no per-block int8 kernel) and
    # bf16 activations, so it is checked against its own f64 reference.
    tx_bf16 = tt(qx).bfloat16() if torch is not None else None
    if torch is not None:
        t_w8 = tt(qq)
        t_s8 = tt(qs.mean(axis=0)).bfloat16()
        q8_ref_t = qq.astype(np.float64) @ tx_bf16.double().numpy() * t_s8.double().numpy()

    def verify_q8(k):
        k.run("q8_gemv", q_packed, s_packed, qx, qy, qm, qk)
        return float(np.max(np.abs(qy - q8_ref)) / np.max(np.abs(q8_ref)))

    cases.append(dict(
        name="q8_gemv", label=f"Q8_0 GEMV, all cores ({qm}x{qk})",
        run=lambda k: k.run("q8_gemv", q_packed, s_packed, qx, qy, qm, qk),
        verify=verify_q8, tol=1e-5, numpy=lambda: np_q8_gemv(q_blocks, qs, qx),
        torch=lambda: torch.ops.aten._weight_int8pack_mm(tx_bf16[None], t_w8, t_s8),
        torch_check=lambda out: rel_err(out, q8_ref_t), torch_tol=1e-2,
        flops=2 * qm * qk,
    ))

    # Q4_0: 4-bit weights q in 0..15 (value q - 8), llama.cpp's nibble bytes per block (byte t:
    # values t and t + 16), packed per chunk and block as 1024 bytes in Q8_0's lane order.
    q4 = rng.integers(0, 16, size=(qm, qk), dtype=np.uint8)
    q4s = (rng.uniform(0.5, 1.0, size=(qnb, qm)) / (4.6 * np.sqrt(qk))).astype(np.float16)
    nib = q4.reshape(qm, qnb, 32)
    nib = nib[:, :, :16] | (nib[:, :, 16:] << 4)                     # [m x k/32 x 16]
    q4_packed = np.ascontiguousarray(
        nib.reshape(qm // 64, 64, qnb, 4, 4).transpose(0, 2, 3, 1, 4))
    q4s_packed = np.ascontiguousarray(q4s.reshape(qnb, qm // 64, 64).transpose(1, 0, 2))
    q4s = q4s.astype(np.float32)
    q4_vals = q4.astype(np.int8) - np.int8(8)
    q4_blocks = np.ascontiguousarray(q4_vals.reshape(qm, qnb, 32).transpose(1, 0, 2))
    dots4 = np.einsum("rbt,bt->rb", q4_vals.reshape(qm, qnb, 32).astype(np.int64),
                      xq_ints.reshape(qnb, 32).astype(np.int64))
    q4_ref = (dots4 * q4s.T.astype(np.float64) * dxs.astype(np.float64)).sum(axis=1)

    # PyTorch's int4 weight-only GEMV (torchao's CPU path): the same 4-bit values and
    # 32-blocks, packed by PyTorch, its weights (q - 8) * scale + zero with bf16 scales and
    # zeros (0 here), and bf16 activations; checked against its own f64 reference.
    if torch is not None:
        t_p4 = torch.ops.aten._convert_weight_to_int4pack_for_cpu(tt(q4.astype(np.int32)), 1)
        t_s4 = tt(q4s).bfloat16()
        t_sz4 = torch.stack([t_s4, torch.zeros_like(t_s4)], dim=-1).contiguous()
        dots4t = np.einsum("rbt,bt->rb", q4_vals.reshape(qm, qnb, 32).astype(np.float64),
                           tx_bf16.double().numpy().reshape(qnb, 32))
        q4_ref_t = (dots4t * t_s4.double().numpy().T).sum(axis=1)

    def verify_q4(k):
        k.run("q4_gemv", q4_packed, q4s_packed, qx, qy, qm, qk)
        return float(np.max(np.abs(qy - q4_ref)) / np.max(np.abs(q4_ref)))

    cases.append(dict(
        name="q4_gemv", label=f"Q4_0 GEMV, all cores ({qm}x{qk})",
        run=lambda k: k.run("q4_gemv", q4_packed, q4s_packed, qx, qy, qm, qk),
        verify=verify_q4, tol=1e-5, numpy=lambda: np_q8_gemv(q4_blocks, q4s, qx),
        torch=lambda: torch.ops.aten._weight_int4pack_mm_for_cpu(tx_bf16[None], t_p4, 32, t_sz4),
        torch_check=lambda out: rel_err(out, q4_ref_t), torch_tol=1e-2,
        flops=2 * qm * qk,
    ))
    return cases


def isa_levels(mode):
    report = achainsaw.cpu_features()
    host = report["host"]
    order = ISA_ORDER.get(host["arch"], [])
    top = host.get("max_isa")
    if mode == "host" or top not in order:
        return [None]
    if mode == "all":
        return order[: order.index(top) + 1]
    return [level.strip() for level in mode.split(",")]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--backends", default="all",
                        help="comma-separated backends, or 'all' available (default)")
    parser.add_argument("--isa", default="host",
                        help="'host' (default), 'all' reachable levels, or a comma list")
    parser.add_argument("--quick", action="store_true", help="short timing runs (CI)")
    parser.add_argument("--json", metavar="PATH", help="also write results as JSON")
    parser.add_argument("--no-numpy", action="store_true",
                        help="skip timing NumPy (results still verified against it)")
    parser.add_argument("--no-torch", action="store_true",
                        help="skip the PyTorch baseline (skipped anyway without PyTorch)")
    args = parser.parse_args()
    use_torch = torch is not None and not args.no_torch

    available = achainsaw.available_backends()
    backends = available if args.backends == "all" else args.backends.split(",")
    levels = isa_levels(args.isa)
    budget = 0.02 if args.quick else 0.3
    cases = make_cases()
    original_cap = achainsaw.get_isa_cap()

    print("=" * 96)
    print("CHAINSAW-BLAS KERNEL BENCHMARKS")
    print(f"backends: {', '.join(backends)}  |  ISA levels: "
          f"{', '.join(l or 'host' for l in levels)}  |  host: "
          f"{achainsaw.cpu_features()['host']['max_isa']}")
    if use_torch:
        print(f"PyTorch {torch.__version__}, {torch.get_num_threads()} threads")
    print("=" * 96)

    results, failures = [], []
    configs = [(b, l) for b in backends for l in levels]
    for case in cases:
        src = source(case["name"])
        # Both runtimes busy-wait for a moment after parallel work; let the other's threads
        # settle before timing each one.
        time.sleep(0.02)
        np_s = None if args.no_numpy else time_call(case["numpy"], budget)
        row = dict(kernel=case["name"], label=case["label"],
                   numpy_us=np_s and np_s * 1e6, torch_us=None, runs=[])
        print(f"\n{case['label']}")
        if np_s:
            print(f"    {'NumPy':<22} {np_s * 1e6:10.2f} us  "
                  f"{case['flops'] / np_s / 1e9:8.2f} GFLOP/s")
        if use_torch:
            err = case["torch_check"](case["torch"]())
            tol = case.get("torch_tol", case["tol"])
            if err > tol:
                failures.append(f"{case['name']} on PyTorch: error {err:.2e}")
            time.sleep(0.02)
            pt_s = time_call(case["torch"], budget)
            row["torch_us"], row["torch_error"] = pt_s * 1e6, err
            vs = f"  {np_s / pt_s:6.2f}x NumPy" if np_s else ""
            print(f"    {'PyTorch':<22} {pt_s * 1e6:10.2f} us  "
                  f"{case['flops'] / pt_s / 1e9:8.2f} GFLOP/s{vs}  err {err:.1e} "
                  f"{'PASS' if err <= tol else 'FAIL'}")
        for backend, level in configs:
            achainsaw.set_isa_cap(level)
            kernel = achainsaw.compile(src, backend=backend, fast_math=True)
            time.sleep(0.02)
            err = case["verify"](kernel)
            ok = err <= case["tol"]
            if not ok:
                failures.append(f"{case['name']} on {backend}/{level or 'host'}: error {err:.2e}")
            secs = time_call(lambda: case["run"](kernel), budget)
            tag = f"{backend}@{level or 'host'}"
            vs = f"  {np_s / secs:6.2f}x NumPy" if np_s else ""
            if row["torch_us"]:
                vs += f"  {row['torch_us'] / 1e6 / secs:6.2f}x PyTorch"
            print(f"    {tag:<22} {secs * 1e6:10.2f} us  {case['flops'] / secs / 1e9:8.2f} GFLOP/s"
                  f"{vs}  err {err:.1e} {'PASS' if ok else 'FAIL'}")
            row["runs"].append(dict(backend=backend, isa=level or "host", us=secs * 1e6,
                                    gflops=case["flops"] / secs / 1e9, error=err, ok=ok))
        results.append(row)
    achainsaw.set_isa_cap(original_cap)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(dict(backends=backends, isa_levels=[l or "host" for l in levels],
                           cpu=achainsaw.cpu_features(),
                           torch=use_torch and dict(version=torch.__version__,
                                                    threads=torch.get_num_threads()),
                           results=results), f, indent=2)

    print("\n" + "=" * 96)
    if failures:
        print("[FAIL] " + "; ".join(failures))
        sys.exit(1)
    print(f"[SUCCESS] ALL {len(cases)} KERNELS PASSED NUMERICAL VERIFICATION "
          f"ON {len(configs)} BACKEND/ISA CONFIGURATIONS")


if __name__ == "__main__":
    main()
