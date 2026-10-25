"""
Chainsaw-BLAS: verification and benchmarks for the kernels in examples/kernels/: the
operators of LLM inference (norms, activations, RoPE, attention for decode, quantized GEMV and
prefill GEMM, argmax and top-k sampling), a few BLAS-style ones, and a byte histogram and
a sparse matrix-vector product on all cores, each also in a form for comparison (one atomic
per byte; a work queue of its own instead of `par`).

Every kernel is checked against NumPy, then timed on each available code generation
backend (Cranelift, and LLVM when the build has it) and, with --isa all, at every ISA
level this machine reaches (for example sse, avx, avx2 and avx512 on an AVX-512 host).
The kernels are vector-length agnostic, so the same source runs 128-bit vectors on
Cranelift and up to 512-bit (or SVE-scalable) vectors on LLVM. Kernels compile with
fast_math=True (float min/max as compare and select; the tests cover both modes).

NumPy and, when it is installed, PyTorch (eager, CPU) are timed on the same work as
baselines. NumPy computes with the kernels' data types. PyTorch runs its own fastest CPU
path for each kernel, and its results are checked too. Where its types differ, the case
says so: bf16 outputs for GEMM and attention (bf16 queries too for GQA attention), and for
the quantized GEMVs and GEMM PyTorch's weight-only int4 and int8 kernels, which take bf16
activations instead of int8 ones (int8 with one scale per row, since PyTorch has no
per-block int8 kernel).

    python benchmarks/benchmark_kernels.py                  # host ISA, all backends
    python benchmarks/benchmark_kernels.py --isa all        # sweep ISA levels too
    python benchmarks/benchmark_kernels.py --quick          # short run (CI)
    python benchmarks/benchmark_kernels.py --json out.json  # machine-readable results
    python benchmarks/benchmark_kernels.py --no-torch       # skip the PyTorch baseline

Exits non-zero if any kernel, or a PyTorch baseline, disagrees with the reference.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from collections.abc import Callable
from typing import Any, NotRequired, TypeAlias, TypedDict

import numpy as np
import numpy.typing as npt

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None

# Ensure achainsaw module is accessible
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import achainsaw

F32: TypeAlias = npt.NDArray[np.float32]
F64: TypeAlias = npt.NDArray[np.float64]
U16: TypeAlias = npt.NDArray[np.uint16]
# Arithmetic results whose dtype NumPy's stubs widen (float32 by a scalar to float64).
Floats: TypeAlias = npt.NDArray[np.floating[Any]]


class Case(TypedDict):
    """A kernel to verify and time (see make_cases)."""
    name: str
    # The source file in examples/kernels/ when it is not `name`.
    file: NotRequired[str]
    label: str
    run: Callable[[achainsaw.Kernel], object]
    verify: Callable[[achainsaw.Kernel], float]
    tol: float
    numpy: Callable[[], object]
    torch: Callable[[], Any]
    torch_check: Callable[[Any], float]
    torch_tol: NotRequired[float]
    flops: int


KERNELS_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "examples",
                                                "kernels"))

ISA_ORDER: dict[str, list[str]] = {
    "x86_64": ["sse", "avx", "avx2", "avx512", "amx"],
    "aarch64": ["neon", "sve", "sve2", "sme"],
}


def source(name: str) -> str:
    with open(os.path.join(KERNELS_DIR, name + ".air"), "r", encoding="utf-8") as f:
        return f.read()


def bf16_bits(x: npt.ArrayLike) -> U16:
    """Round f32 values to bfloat16 (nearest-even) and return the raw u16 bits."""
    b: npt.NDArray[np.uint32] = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return ((b + 0x7FFF + ((b >> 16) & 1)) >> 16).astype(np.uint16)


def bf16_values(bits: U16) -> F32:
    return (bits.astype(np.uint32) << 16).view(np.float32)


def q8_quantize(x: F32) -> tuple[F32, Floats]:
    """Q8_0 activations exactly as the kernels' quantize_q8: per 32-block scale
    max|x| / 127 and values rounded half away from zero, in f32. Returns the integer values
    (as f32) and the scales."""
    blocks: F32 = x.reshape(-1, 32).astype(np.float32)
    d: Floats = np.abs(blocks).max(axis=1) / np.float32(127.0)
    with np.errstate(divide="ignore"):
        inv: F32 = np.where(d > 0, np.float32(1.0) / d, np.float32(0.0)).astype(np.float32)
    v: F32 = blocks * inv[:, None]
    t: F32 = np.trunc(v)
    # Halves away from zero, exactly (C's roundf): every step is exact in f32.
    q: F32 = t + np.where(np.abs(v - t) >= np.float32(0.5), np.sign(v), np.float32(0.0))
    return q.reshape(-1), d


def np_q8_gemv(q_blocks: npt.NDArray[np.int8], scales: F32, x: F32) -> Floats:
    """NumPy Q8_0 GEMV with the kernel's data types: int8 weights q_blocks [k/32 x m x 32],
    f32 scales [k/32 x m], int8-quantized activations, exact integer block dots (computed as a
    batched f32 matmul, exact because every block dot is below 2^24), f32 accumulation."""
    xq, dx = q8_quantize(x)
    nb: int = q_blocks.shape[0]
    dots: F32 = np.matmul(q_blocks.astype(np.float32), xq.reshape(nb, 32, 1))[:, :, 0]
    return ((dots * dx[:, None]) * scales).sum(axis=0)


def np_q8_gemm(q_blocks: npt.NDArray[np.int8], scales: F32, xs: F32) -> Floats:
    """np_q8_gemv for n tokens xs [n x k]: one batched f32 matmul over the blocks for every
    token (exact block dots), so the int8 weights are widened once per call."""
    qs: list[tuple[F32, Floats]] = [q8_quantize(x) for x in xs]
    nb: int = q_blocks.shape[0]
    xq: F32 = np.stack([q.reshape(nb, 32) for q, _ in qs], axis=2)     # [k/32 x 32 x n]
    dx: Floats = np.stack([d for _, d in qs], axis=1)                  # [k/32 x n]
    dots: F32 = np.matmul(q_blocks.astype(np.float32), xq)             # [k/32 x m x n]
    return ((dots * dx[:, None, :]) * scales[:, :, None]).sum(axis=0).T


def rel_err(out: Any, ref: npt.ArrayLike) -> float:
    """max |out - ref| / max |ref|, for a PyTorch result against the case's f64 reference."""
    o: F64 = np.asarray(
        out.float() if torch is not None and isinstance(out, torch.Tensor) else out,
        dtype=np.float64)
    r: F64 = np.asarray(ref, dtype=np.float64)
    return float(np.max(np.abs(o.reshape(r.shape) - r)) / np.max(np.abs(r)))


def torch_bf16(bits: U16) -> torch.Tensor:
    """A PyTorch bf16 tensor over raw bf16 bits (zero copy)."""
    return torch.from_numpy(np.ascontiguousarray(bits).view(np.int16)).view(torch.bfloat16)


def time_call(func: Callable[[], object], budget_s: float, min_iters: int = 5,
              chunks: int = 5) -> float:
    """Seconds per call after a short warm-up: the run of about `budget_s` is split into
    `chunks` and the fastest chunk's average is returned, so a transient stall (another
    process, a migrated thread) does not count against the kernel."""
    for _ in range(3):
        func()
    best: float = float("inf")
    for _ in range(chunks):
        iters: int = 0
        t0: float = time.perf_counter()
        while True:
            func()
            iters += 1
            elapsed: float = time.perf_counter() - t0
            if iters >= min_iters and elapsed >= budget_s / chunks:
                break
        best = min(best, elapsed / iters)
    return best


def tt(x: npt.NDArray[Any]) -> Any:
    """A PyTorch tensor sharing x's memory, or None without PyTorch (its cases' torch
    callables are then never called)."""
    return torch.from_numpy(x) if torch is not None else None


def make_cases() -> list[Case]:
    """Each case: name, label, kernel source, args builder, verify(kernel) -> max error,
    tolerance, NumPy baseline, PyTorch baseline with its check(result) -> error (and its own
    tolerance where its types differ), and the work per call (for GFLOP/s)."""
    rng: np.random.Generator = np.random.default_rng(42)
    cases: list[Case] = []

    dim: int = 1024  # embedding size of common text-embedding models
    a: F32 = rng.standard_normal(dim).astype(np.float32)
    b: F32 = rng.standard_normal(dim).astype(np.float32)
    ta: torch.Tensor = tt(a)
    tb: torch.Tensor = tt(b)
    cos_ref: float = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    cases.append(Case(
        name="cosine_similarity", label=f"Cosine similarity (n={dim})",
        run=lambda k: k(a, b, dim),
        verify=lambda k: abs(k(a, b, dim) - cos_ref), tol=1e-5,
        numpy=lambda: np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12),
        torch=lambda: F.cosine_similarity(ta, tb, dim=0),
        torch_check=lambda out: abs(float(out) - cos_ref),
        flops=6 * dim,
    ))

    l2_ref: float = float(np.linalg.norm(a - b))
    cases.append(Case(
        name="euclidean_distance", label=f"Euclidean distance (n={dim})",
        run=lambda k: k(a, b, dim),
        verify=lambda k: abs(k(a, b, dim) - l2_ref) / l2_ref, tol=1e-5,
        numpy=lambda: np.linalg.norm(a - b),
        torch=lambda: torch.dist(ta, tb),
        torch_check=lambda out: abs(float(out) - l2_ref) / l2_ref,
        flops=3 * dim,
    ))

    sm_n: int = 1000
    x_sm: F32 = (rng.standard_normal(sm_n) * 4.0).astype(np.float32)
    out_sm: F32 = np.zeros(sm_n, dtype=np.float32)
    t_sm: torch.Tensor = tt(x_sm)
    e: F64 = np.exp(x_sm.astype(np.float64) - np.max(x_sm))
    sm_ref: F64 = e / e.sum()

    def verify_softmax(k: achainsaw.Kernel) -> float:
        k(x_sm, out_sm, sm_n)
        return float(np.max(np.abs(out_sm - sm_ref)))

    def np_softmax() -> F32:
        z: F32 = np.exp(x_sm - np.max(x_sm))
        p: F32 = z / z.sum()
        return p

    cases.append(Case(
        name="softmax", label=f"Softmax (n={sm_n})",
        run=lambda k: k(x_sm, out_sm, sm_n),
        verify=verify_softmax, tol=1e-6, numpy=np_softmax,
        torch=lambda: torch.softmax(t_sm, 0),
        torch_check=lambda out: float(np.max(np.abs(out.numpy() - sm_ref))),
        flops=4 * sm_n,
    ))

    rms_n: int = 4096
    x_r: F32 = rng.standard_normal(rms_n).astype(np.float32)
    w_r: F32 = rng.uniform(0.8, 1.2, rms_n).astype(np.float32)
    out_r: F32 = np.zeros(rms_n, dtype=np.float32)
    t_xr: torch.Tensor = tt(x_r)
    t_wr: torch.Tensor = tt(w_r)
    xd: F64 = x_r.astype(np.float64)
    rms_ref: F64 = xd / np.sqrt(np.mean(xd * xd) + 1e-6) * w_r

    def verify_rms(k: achainsaw.Kernel) -> float:
        k(x_r, w_r, out_r, rms_n)
        return float(np.max(np.abs(out_r - rms_ref)))

    cases.append(Case(
        name="rmsnorm", label=f"RMSNorm (n={rms_n})",
        run=lambda k: k(x_r, w_r, out_r, rms_n),
        verify=verify_rms, tol=1e-5,
        numpy=lambda: x_r / np.sqrt(np.mean(x_r * x_r) + 1e-6) * w_r,
        torch=lambda: F.rms_norm(t_xr, (rms_n,), t_wr, eps=1e-6),
        torch_check=lambda out: float(np.max(np.abs(out.numpy() - rms_ref))),
        flops=4 * rms_n,
    ))

    gm: int = 512
    gk: int = 1024
    A: F32 = rng.standard_normal((gm, gk)).astype(np.float32)
    xg: F32 = rng.standard_normal(gk).astype(np.float32)
    yg: F32 = np.zeros(gm, dtype=np.float32)
    tA: torch.Tensor = tt(A)
    txg: torch.Tensor = tt(xg)
    gemv_ref: F64 = A.astype(np.float64) @ xg.astype(np.float64)

    def verify_gemv(k: achainsaw.Kernel) -> float:
        k(A, xg, yg, gm, gk)
        return float(np.max(np.abs(yg - gemv_ref)) / np.max(np.abs(gemv_ref)))

    cases.append(Case(
        name="gemv_f32", label=f"GEMV f32 ({gm}x{gk})",
        run=lambda k: k(A, xg, yg, gm, gk),
        verify=verify_gemv, tol=1e-5, numpy=lambda: A @ xg,
        torch=lambda: torch.mv(tA, txg), torch_check=lambda out: rel_err(out, gemv_ref),
        flops=2 * gm * gk,
    ))

    def verify_gemv_par(k: achainsaw.Kernel) -> float:
        yg[:] = np.nan
        k.run("gemv_par", A, xg, yg, gm, gk)
        return float(np.max(np.abs(yg - gemv_ref)) / np.max(np.abs(gemv_ref)))

    cases.append(Case(
        name="gemv_par", label=f"GEMV f32, all cores via par ({gm}x{gk})",
        run=lambda k: k.run("gemv_par", A, xg, yg, gm, gk),
        verify=verify_gemv_par, tol=1e-5, numpy=lambda: A @ xg,
        torch=lambda: torch.mv(tA, txg), torch_check=lambda out: rel_err(out, gemv_ref),
        flops=2 * gm * gk,
    ))

    n: int = 256
    Ab: U16 = bf16_bits(rng.standard_normal((n, n)))
    Bb: U16 = bf16_bits(rng.standard_normal((n, n)))
    Af: F32 = bf16_values(Ab)
    Bf: F32 = bf16_values(Bb)
    C: F32 = np.zeros((n, n), dtype=np.float32)
    if torch is not None:
        tAb: torch.Tensor = torch_bf16(Ab)
        tBb: torch.Tensor = torch_bf16(Bb)
    gemm_ref: F64 = Af.astype(np.float64) @ Bf.astype(np.float64)

    def verify_gemm(k: achainsaw.Kernel) -> float:
        C[:] = 0.0
        k.run("gemm_bf16", C, Ab, Bb, n, n, n)
        return float(np.max(np.abs(C - gemm_ref)) / np.max(np.abs(gemm_ref)))

    cases.append(Case(
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
    fh: int = 128
    fd: int = 512
    fcache: int = 4096
    fnk: int = 1152
    fq: U16 = bf16_bits(rng.standard_normal((fh, fd)))
    fkv: U16 = bf16_bits(rng.standard_normal((fcache, fd)))
    fidx: npt.NDArray[np.int32] = rng.choice(fcache, size=fnk, replace=False).astype(np.int32)
    fsink: F32 = rng.standard_normal(fh).astype(np.float32)
    fout: F32 = np.zeros((fh, fd), dtype=np.float32)
    fscale: np.float32 = np.float32(1.0 / np.sqrt(fd))
    qf: F64 = bf16_values(fq).astype(np.float64)
    kvf: F64 = bf16_values(fkv).astype(np.float64)
    sel: F64 = kvf[fidx]
    scores: F64 = (qf @ sel.T) * float(fscale)
    mx: F64 = scores.max(axis=1, keepdims=True)
    pr: F64 = np.exp(scores - mx)
    den: F64 = pr.sum(axis=1, keepdims=True) + np.exp(fsink[:, None].astype(np.float64) - mx)
    flash_ref: F64 = (pr @ sel) / den

    def verify_flash(k: achainsaw.Kernel) -> float:
        k.run("flash_attention", fq, fkv, fidx, fsink, fout, fh, fd, fnk, fscale)
        return float(np.max(np.abs(fout - flash_ref)) / np.max(np.abs(flash_ref)))

    def np_flash() -> Floats:
        # Same types as the kernel: bf16 q and kv (widened to f32 here), f32 scores and
        # softmax, probabilities rounded to bf16 for the product with V. The -1 entries are
        # all valid here, so no masking is needed.
        q32: F32 = bf16_values(fq)
        sel32: F32 = bf16_values(fkv[fidx])
        s: F32 = (q32 @ sel32.T) * fscale
        m: F32 = s.max(axis=1, keepdims=True)
        e: F32 = np.exp(s - m)
        p: F32 = bf16_values(bf16_bits(e))
        return (p @ sel32) / (e.sum(axis=1, keepdims=True) + np.exp(fsink[:, None] - m))

    if torch is not None:
        # PyTorch's scaled_dot_product_attention in bf16 (f32 softmax inside, bf16 output),
        # all heads as the queries of one KV head. The sink is one more key: a zero row
        # appended to the cache (score 0) plus a mask of the sink logit on its column.
        t_q: torch.Tensor = torch_bf16(fq).view(1, 1, fh, fd)
        t_kv: torch.Tensor = torch.cat([torch_bf16(fkv),
                                        torch.zeros(1, fd, dtype=torch.bfloat16)])
        t_idx: torch.Tensor = torch.from_numpy(np.append(fidx, fcache).astype(np.int64))
        t_mask: torch.Tensor = torch.zeros(1, 1, fh, fnk + 1, dtype=torch.bfloat16)
        t_mask[..., -1] = tt(fsink)

    def torch_flash() -> torch.Tensor:
        t_sel: torch.Tensor = t_kv[t_idx].view(1, 1, fnk + 1, fd)
        return F.scaled_dot_product_attention(t_q, t_sel, t_sel, attn_mask=t_mask,
                                              scale=float(fscale))

    cases.append(Case(
        name="flash_attention",
        label=f"Flash attention decode, DeepSeek V4 Pro ({fh}x{fd}, {fnk} keys)",
        run=lambda k: k.run("flash_attention", fq, fkv, fidx, fsink, fout, fh, fd, fnk, fscale),
        verify=verify_flash, tol=1e-2, numpy=np_flash,
        torch=torch_flash, torch_check=lambda out: rel_err(out, flash_ref),
        flops=2 * 2 * fh * fnk * fd,
    ))

    # One decode step of grouped-query attention at Llama 3 8B's shapes: 32 query heads on 8
    # KV heads of 128 dims, over 4096 positions of a bf16 KV cache (16 MB).
    gh: int = 32
    gkv: int = 8
    ghd: int = 128
    gt: int = 4096
    ggrp: int = gh // gkv
    gq: F32 = rng.standard_normal((gh, ghd)).astype(np.float32)
    gkc: U16 = bf16_bits(rng.standard_normal((gkv, gt, ghd)))
    gvc: U16 = bf16_bits(rng.standard_normal((gkv, gt, ghd)))
    # The kernel's K: tiles of 32 positions, each [hd x 32] (the baselines take gkc's rows).
    gkt: U16 = np.ascontiguousarray(
        gkc.reshape(gkv, gt // 32, 32, ghd).transpose(0, 1, 3, 2))
    gout: F32 = np.zeros((gh, ghd), dtype=np.float32)
    gscale: np.float32 = np.float32(1.0 / np.sqrt(ghd))
    gk64: F64 = bf16_values(gkc).astype(np.float64)
    gv64: F64 = bf16_values(gvc).astype(np.float64)
    gsc: F64 = np.matmul(gq.astype(np.float64).reshape(gkv, ggrp, ghd), gk64.transpose(0, 2, 1))
    gsc *= float(gscale)
    gp: F64 = np.exp(gsc - gsc.max(axis=2, keepdims=True))
    gqa_ref: F64 = (np.matmul(gp, gv64) / gp.sum(axis=2, keepdims=True)).reshape(gh, ghd)

    def np_gqa() -> F32:
        # Same types as the kernel: the bf16 cache widened to f32 in the call.
        k32: F32 = bf16_values(gkc)
        v32: F32 = bf16_values(gvc)
        s: F32 = np.matmul(gq.reshape(gkv, ggrp, ghd), k32.transpose(0, 2, 1)) * gscale
        e: F32 = np.exp(s - s.max(axis=2, keepdims=True))
        o: F32 = np.matmul(e, v32) / e.sum(axis=2, keepdims=True)
        return o.reshape(gh, ghd)

    if torch is not None:
        # bf16 throughout (its SDPA wants one dtype), bf16 output. Each group's query heads
        # go in as that KV head's query positions: the same math, and PyTorch's fastest form
        # (enable_gqa=True with one query position per head takes 18x as long).
        t_gq: torch.Tensor = tt(gq).bfloat16().view(1, gkv, ggrp, ghd)
        t_gk: torch.Tensor = torch_bf16(gkc).view(1, gkv, gt, ghd)
        t_gv: torch.Tensor = torch_bf16(gvc).view(1, gkv, gt, ghd)

    def verify_gqa(k: achainsaw.Kernel) -> float:
        gout[:] = np.nan
        k.run("gqa_attention", gq, gkt, gvc, gout, gh, gkv, ghd, gt, gt)
        return rel_err(gout, gqa_ref)

    cases.append(Case(
        name="gqa_attention",
        label=f"GQA decode attention, Llama 3 8B ({gh}/{gkv} heads x {ghd}, {gt} positions)",
        run=lambda k: k.run("gqa_attention", gq, gkt, gvc, gout, gh, gkv, ghd, gt, gt),
        verify=verify_gqa, tol=1e-5, numpy=np_gqa,
        torch=lambda: F.scaled_dot_product_attention(t_gq, t_gk, t_gv),
        torch_check=lambda out: rel_err(out, gqa_ref), torch_tol=1e-2,
        flops=2 * 2 * gh * gt * ghd,
    ))

    # Decode-step glue at Llama 3 8B sizes (d = 4096, MLP 14336, 32 heads of 128,
    # vocabulary 128256).
    sw_n: int = 14336
    gate: F32 = (rng.standard_normal(sw_n) * 3.0).astype(np.float32)
    up: F32 = rng.standard_normal(sw_n).astype(np.float32)
    sw_out: F32 = np.zeros(sw_n, dtype=np.float32)
    t_gate: torch.Tensor = tt(gate)
    t_up: torch.Tensor = tt(up)
    gd: F64 = gate.astype(np.float64)
    sw_ref: F64 = gd / (1.0 + np.exp(-gd)) * up

    def verify_swiglu(k: achainsaw.Kernel) -> float:
        k(gate, up, sw_out, sw_n)
        return float(np.max(np.abs(sw_out - sw_ref) / (np.abs(sw_ref) + 1e-30)))

    cases.append(Case(
        name="swiglu", label=f"SwiGLU, vectorized exp (n={sw_n})",
        run=lambda k: k(gate, up, sw_out, sw_n),
        verify=verify_swiglu, tol=2e-6,
        numpy=lambda: gate / (1.0 + np.exp(-gate)) * up,
        torch=lambda: F.silu(t_gate) * t_up,
        torch_check=lambda out: float(np.max(np.abs(out.numpy() - sw_ref)
                                             / (np.abs(sw_ref) + 1e-30))),
        flops=8 * sw_n,
    ))

    # GeGLU (Gemma): GELU with PyTorch's tanh approximation on the gate, times up.
    gc: np.float64 = np.sqrt(2.0 / np.pi)
    ginner: F64 = gc * (gd + 0.044715 * gd ** 3)
    ge_ref: F64 = gd * up / (1.0 + np.exp(-2.0 * ginner))

    def np_geglu() -> Floats:
        g: F32 = gate
        return np.float32(0.5) * g * (np.float32(1.0) + np.tanh(
            np.float32(gc) * (g + np.float32(0.044715) * g * g * g))) * up

    def verify_geglu(k: achainsaw.Kernel) -> float:
        k(gate, up, sw_out, sw_n)
        return rel_err(sw_out, ge_ref)

    cases.append(Case(
        name="geglu", label=f"GeGLU, GELU-tanh via vectorized exp (n={sw_n})",
        run=lambda k: k(gate, up, sw_out, sw_n),
        verify=verify_geglu, tol=1e-6, numpy=np_geglu,
        torch=lambda: F.gelu(t_gate, approximate="tanh") * t_up,
        torch_check=lambda out: rel_err(out, ge_ref),
        flops=11 * sw_n,
    ))

    vocab: int = 128256
    logits: F32 = rng.standard_normal(vocab).astype(np.float32)
    am_ref: int = int(np.argmax(logits))
    t_logits: torch.Tensor = tt(logits)
    cases.append(Case(
        name="argmax", label=f"Greedy argmax (vocab={vocab})",
        run=lambda k: k(logits, vocab),
        verify=lambda k: float(k(logits, vocab) != am_ref), tol=0.0,
        numpy=lambda: np.argmax(logits),
        # torch.max along the dimension finds the index faster than torch.argmax.
        torch=lambda: torch.max(t_logits, 0),
        torch_check=lambda out: float(int(out.indices) != am_ref),
        flops=vocab,
    ))

    # Top-k sampling's selection: the 40 largest logits, sorted, with their token ids.
    tk: int = 40
    tk_vals: F32 = np.zeros(tk, dtype=np.float32)
    tk_idx: npt.NDArray[np.int64] = np.zeros(tk, dtype=np.int64)
    tk_ref: npt.NDArray[np.intp] = np.argsort(-logits, kind="stable")[:tk]

    def verify_topk(k: achainsaw.Kernel) -> float:
        tk_idx[:] = -1
        k(logits, vocab, tk, tk_vals, tk_idx)
        return float(not (np.array_equal(tk_idx, tk_ref)
                          and np.array_equal(tk_vals, logits[tk_ref])))

    def np_topk() -> npt.NDArray[np.intp]:
        idx: npt.NDArray[np.intp] = np.argpartition(logits, -tk)[-tk:]
        return idx[np.argsort(-logits[idx])]

    cases.append(Case(
        name="topk", label=f"Top-k, k={tk} (vocab={vocab})",
        run=lambda k: k(logits, vocab, tk, tk_vals, tk_idx),
        verify=verify_topk, tol=0.0, numpy=np_topk,
        torch=lambda: torch.topk(t_logits, tk),
        torch_check=lambda out: float(not np.array_equal(out.indices.numpy(), tk_ref)),
        flops=vocab,
    ))

    rh: int = 32
    rd: int = 128
    rpos: int = 1000
    rhalf: int = rd // 2
    ang: F64 = rpos * 500000.0 ** (-2.0 * np.arange(rhalf) / rd)
    rcos: F32 = np.cos(ang).astype(np.float32)
    rsin: F32 = np.sin(ang).astype(np.float32)
    rq0: F32 = rng.standard_normal((rh, rd)).astype(np.float32)
    rq: F32 = rq0.copy()
    x1: F64 = rq0[:, :rhalf].astype(np.float64)
    x2: F64 = rq0[:, rhalf:].astype(np.float64)
    rope_ref: F64 = np.concatenate([x1 * rcos - x2 * rsin, x2 * rcos + x1 * rsin], axis=1)

    def verify_rope(k: achainsaw.Kernel) -> float:
        rq[:] = rq0
        k(rq, rh, rd, rcos, rsin)
        return float(np.max(np.abs(rq - rope_ref)))

    def np_rope() -> F32:
        a: F32 = rq0[:, :rhalf]
        b: F32 = rq0[:, rhalf:]
        return np.concatenate([a * rcos - b * rsin, b * rcos + a * rsin], axis=1)

    # PyTorch has no RoPE operator; models write it out like this.
    t_rq0: torch.Tensor = tt(rq0)
    t_rcos: torch.Tensor = tt(rcos)
    t_rsin: torch.Tensor = tt(rsin)

    def torch_rope() -> torch.Tensor:
        a: torch.Tensor = t_rq0[:, :rhalf]
        b: torch.Tensor = t_rq0[:, rhalf:]
        return torch.cat([a * t_rcos - b * t_rsin, b * t_rcos + a * t_rsin], dim=1)

    cases.append(Case(
        name="rope", label=f"RoPE, in place ({rh} heads x {rd})",
        run=lambda k: k(rq, rh, rd, rcos, rsin),
        verify=verify_rope, tol=1e-5, numpy=np_rope,
        torch=torch_rope,
        torch_check=lambda out: float(np.max(np.abs(out.numpy() - rope_ref))),
        flops=6 * rh * rhalf,
    ))

    an: int = 4096
    # The four buffers are slices of one block, 17 cache lines apart beyond their 16 KB. Left
    # to the allocator, `out` landed exactly 64 KB + 48 bytes after `res`, and this Zen 4
    # then ran the kernel 2x slower in most processes (1.5 vs 3.1 µs): the second pass
    # stores out[i] while loading res a few lanes ahead, apparently taken by the CPU as a
    # possible alias of the load. Fixed offsets keep the timing the same from run to run.
    gap: int = an + 17 * 16
    block: F32 = np.zeros(4 * gap, dtype=np.float32)
    ax, ares, aw, aout = (block[i * gap:i * gap + an] for i in range(4))
    ax[:] = rng.standard_normal(an)
    ares0: F32 = rng.standard_normal(an).astype(np.float32)
    ares[:] = ares0
    aw[:] = rng.uniform(0.8, 1.2, an)
    asum: F64 = ares0.astype(np.float64) + ax
    t_ax: torch.Tensor = tt(ax)
    t_ares0: torch.Tensor = tt(ares0)
    t_aw: torch.Tensor = tt(aw)
    an_ref: F64 = asum / np.sqrt(np.mean(asum * asum) + 1e-5) * aw

    def verify_add_rmsnorm(k: achainsaw.Kernel) -> float:
        ares[:] = ares0
        k(ax, ares, aw, aout, an, 1e-5)
        return float(np.max(np.abs(aout - an_ref)))

    def np_add_rmsnorm() -> F32:
        r: F32 = ares0 + ax
        y: F32 = r / np.sqrt(np.mean(r * r) + 1e-5) * aw
        return y

    cases.append(Case(
        name="add_rmsnorm", label=f"Residual add + RMSNorm (n={an})",
        run=lambda k: k(ax, ares, aw, aout, an, 1e-5),
        verify=verify_add_rmsnorm, tol=1e-5, numpy=np_add_rmsnorm,
        torch=lambda: F.rms_norm(t_ares0 + t_ax, (an,), t_aw, eps=1e-5),
        torch_check=lambda out: float(np.max(np.abs(out.numpy() - an_ref))),
        flops=5 * an,
    ))

    ln_n: int = 4096
    lx: F32 = (rng.standard_normal(ln_n) + 0.5).astype(np.float32)
    lw: F32 = rng.uniform(0.8, 1.2, ln_n).astype(np.float32)
    lb: F32 = (rng.standard_normal(ln_n) * 0.1).astype(np.float32)
    lout: F32 = np.zeros(ln_n, dtype=np.float32)
    lxd: F64 = lx.astype(np.float64)
    ln_ref: F64 = (lxd - lxd.mean()) / np.sqrt(lxd.var() + 1e-5) * lw + lb
    t_lx: torch.Tensor = tt(lx)
    t_lw: torch.Tensor = tt(lw)
    t_lb: torch.Tensor = tt(lb)

    def verify_layernorm(k: achainsaw.Kernel) -> float:
        k(lx, lw, lb, lout, ln_n, 1e-5)
        return float(np.max(np.abs(lout - ln_ref)))

    def np_layernorm() -> F32:
        d: F32 = lx - lx.mean()
        y: F32 = d / np.sqrt(np.mean(d * d) + np.float32(1e-5)) * lw + lb
        return y

    cases.append(Case(
        name="layernorm", label=f"LayerNorm (n={ln_n})",
        run=lambda k: k(lx, lw, lb, lout, ln_n, 1e-5),
        verify=verify_layernorm, tol=1e-5, numpy=np_layernorm,
        torch=lambda: F.layer_norm(t_lx, (ln_n,), t_lw, t_lb, eps=1e-5),
        torch_check=lambda out: float(np.max(np.abs(out.numpy() - ln_ref))),
        flops=8 * ln_n,
    ))

    qm: int = 4096
    qk: int = 4096
    qnb: int = qk // 32
    qq: npt.NDArray[np.int8] = rng.integers(-127, 128, size=(qm, qk), dtype=np.int8)
    # f16 scales, as GGUF stores them (NumPy uses the same values in f32).
    qs16: npt.NDArray[np.float16] = (rng.uniform(0.5, 1.0, size=(qnb, qm))
                                     / (73.0 * np.sqrt(qk))).astype(np.float16)
    # Packed layout: 64-row chunks; per 32-block, 4 consecutive values of a row per 32-bit
    # lane, lanes ordered by group then row, values + 128 as unsigned bytes (see qmat_chunk).
    q_packed: npt.NDArray[np.uint8] = np.ascontiguousarray(
        (qq.view(np.uint8) ^ np.uint8(0x80)).reshape(qm // 64, 64, qnb, 8, 4).transpose(
            0, 2, 3, 1, 4))
    s_packed: npt.NDArray[np.float16] = np.ascontiguousarray(
        qs16.reshape(qnb, qm // 64, 64).transpose(1, 0, 2))
    qs: F32 = qs16.astype(np.float32)
    qx: F32 = rng.standard_normal(qk).astype(np.float32)
    qy: F32 = np.zeros(qm, dtype=np.float32)
    # NumPy keeps the same int8 weights, block-major ([k/32 x m x 32]).
    q_blocks: npt.NDArray[np.int8] = np.ascontiguousarray(
        qq.reshape(qm, qnb, 32).transpose(1, 0, 2))
    xq_ints, dxs = q8_quantize(qx)
    dots: npt.NDArray[np.int64] = np.einsum("rbt,bt->rb",
                                            qq.reshape(qm, qnb, 32).astype(np.int64),
                                            xq_ints.reshape(qnb, 32).astype(np.int64))
    q8_ref: F64 = (dots.astype(np.float64) * qs.T.astype(np.float64)
                   * dxs.astype(np.float64)).sum(axis=1)

    # PyTorch's int8 weight-only GEMV (torchao's CPU path): the same int8 weights, but one
    # scale per row (each row's mean block scale here; it has no per-block int8 kernel) and
    # bf16 activations, so it is checked against its own f64 reference.
    tx_bf16: torch.Tensor = tt(qx).bfloat16() if torch is not None else None
    if torch is not None:
        t_w8: torch.Tensor = tt(qq)
        t_s8: torch.Tensor = tt(qs.mean(axis=0)).bfloat16()
        q8_ref_t: F64 = qq.astype(np.float64) @ tx_bf16.double().numpy() * \
            t_s8.double().numpy()

    def verify_q8(k: achainsaw.Kernel) -> float:
        k.run("q8_gemv", q_packed, s_packed, qx, qy, qm, qk)
        return float(np.max(np.abs(qy - q8_ref)) / np.max(np.abs(q8_ref)))

    cases.append(Case(
        name="q8_gemv", label=f"Q8_0 GEMV, all cores ({qm}x{qk})",
        run=lambda k: k.run("q8_gemv", q_packed, s_packed, qx, qy, qm, qk),
        verify=verify_q8, tol=1e-5, numpy=lambda: np_q8_gemv(q_blocks, qs, qx),
        torch=lambda: torch.ops.aten._weight_int8pack_mm(tx_bf16[None], t_w8, t_s8),
        torch_check=lambda out: rel_err(out, q8_ref_t), torch_tol=1e-2,
        flops=2 * qm * qk,
    ))

    # Q4_0: 4-bit weights q in 0..15 (value q - 8), llama.cpp's nibble bytes per block (byte t:
    # values t and t + 16), packed per chunk and block as 1024 bytes in Q8_0's lane order.
    q4: npt.NDArray[np.uint8] = rng.integers(0, 16, size=(qm, qk), dtype=np.uint8)
    q4s16: npt.NDArray[np.float16] = (rng.uniform(0.5, 1.0, size=(qnb, qm))
                                      / (4.6 * np.sqrt(qk))).astype(np.float16)
    v4: npt.NDArray[np.uint8] = q4.reshape(qm, qnb, 32)
    nib: npt.NDArray[np.integer[Any]] = v4[:, :, :16] | (v4[:, :, 16:] << 4)  # [m x k/32 x 16]
    q4_packed: npt.NDArray[np.integer[Any]] = np.ascontiguousarray(
        nib.reshape(qm // 64, 64, qnb, 4, 4).transpose(0, 2, 3, 1, 4))
    q4s_packed: npt.NDArray[np.float16] = np.ascontiguousarray(
        q4s16.reshape(qnb, qm // 64, 64).transpose(1, 0, 2))
    q4s: F32 = q4s16.astype(np.float32)
    q4_vals: npt.NDArray[np.int8] = q4.astype(np.int8) - np.int8(8)
    q4_blocks: npt.NDArray[np.int8] = np.ascontiguousarray(
        q4_vals.reshape(qm, qnb, 32).transpose(1, 0, 2))
    dots4: npt.NDArray[np.int64] = np.einsum("rbt,bt->rb",
                                             q4_vals.reshape(qm, qnb, 32).astype(np.int64),
                                             xq_ints.reshape(qnb, 32).astype(np.int64))
    q4_ref: F64 = (dots4.astype(np.float64) * q4s.T.astype(np.float64)
                   * dxs.astype(np.float64)).sum(axis=1)

    # PyTorch's int4 weight-only GEMV (torchao's CPU path): the same 4-bit values and
    # 32-blocks, packed by PyTorch, its weights (q - 8) * scale + zero with bf16 scales and
    # zeros (0 here), and bf16 activations; checked against its own f64 reference.
    if torch is not None:
        t_p4: torch.Tensor = torch.ops.aten._convert_weight_to_int4pack_for_cpu(
            tt(q4.astype(np.int32)), 1)
        t_s4: torch.Tensor = tt(q4s).bfloat16()
        t_sz4: torch.Tensor = torch.stack([t_s4, torch.zeros_like(t_s4)], dim=-1).contiguous()
        dots4t: F64 = np.einsum("rbt,bt->rb", q4_vals.reshape(qm, qnb, 32).astype(np.float64),
                                tx_bf16.double().numpy().reshape(qnb, 32))
        q4_ref_t: F64 = (dots4t * t_s4.double().numpy().T).sum(axis=1)

    def verify_q4(k: achainsaw.Kernel) -> float:
        k.run("q4_gemv", q4_packed, q4s_packed, qx, qy, qm, qk)
        return float(np.max(np.abs(qy - q4_ref)) / np.max(np.abs(q4_ref)))

    cases.append(Case(
        name="q4_gemv", label=f"Q4_0 GEMV, all cores ({qm}x{qk})",
        run=lambda k: k.run("q4_gemv", q4_packed, q4s_packed, qx, qy, qm, qk),
        verify=verify_q4, tol=1e-5, numpy=lambda: np_q8_gemv(q4_blocks, q4s, qx),
        torch=lambda: torch.ops.aten._weight_int4pack_mm_for_cpu(tx_bf16[None], t_p4, 32,
                                                                 t_sz4),
        torch_check=lambda out: rel_err(out, q4_ref_t), torch_tol=1e-2,
        flops=2 * qm * qk,
    ))

    # Prefill: the same Q4_0 matrix on a batch of 32 tokens.
    pn: int = 32
    px: F32 = rng.standard_normal((pn, qk)).astype(np.float32)
    p_out: F32 = np.zeros((pn, qm), dtype=np.float32)
    pxq: list[tuple[F32, Floats]] = [q8_quantize(x) for x in px]
    pxq_ints: F64 = np.stack([q.reshape(qnb, 32) for q, _ in pxq], axis=2).astype(np.float64)
    pdx: F64 = np.stack([d for _, d in pxq], axis=1).astype(np.float64)
    pdots: F64 = np.matmul(q4_blocks.astype(np.float64), pxq_ints)   # [k/32 x m x n]
    q4g_ref: F64 = ((pdots * pdx[:, None, :]) * q4s.astype(np.float64)[:, :, None]).sum(axis=0).T
    if torch is not None:
        tpx_bf16: torch.Tensor = tt(px).bfloat16()
        w4t: F64 = (q4_vals.astype(np.float64).reshape(qm, qnb, 32)
               * t_s4.double().numpy().T[:, :, None]).reshape(qm, qk)
        q4g_ref_t: F64 = tpx_bf16.double().numpy() @ w4t.T

    def verify_q4_gemm(k: achainsaw.Kernel) -> float:
        p_out[:] = np.nan
        k.run("q4_gemm", q4_packed, q4s_packed, px, p_out, qm, qk, pn)
        return rel_err(p_out, q4g_ref)

    cases.append(Case(
        name="q4_gemm", label=f"Q4_0 GEMM, prefill of {pn} tokens, all cores ({qm}x{qk})",
        run=lambda k: k.run("q4_gemm", q4_packed, q4s_packed, px, p_out, qm, qk, pn),
        verify=verify_q4_gemm, tol=1e-5, numpy=lambda: np_q8_gemm(q4_blocks, q4s, px),
        torch=lambda: torch.ops.aten._weight_int4pack_mm_for_cpu(tpx_bf16, t_p4, 32, t_sz4),
        torch_check=lambda out: rel_err(out, q4g_ref_t), torch_tol=1e-2,
        flops=2 * qm * qk * pn,
    ))

    # Byte histogram of 16 MB: private tables per 64 KB chunk merged with one atomic per
    # bin, and for comparison one atomic per byte on the shared bins.
    hn: int = 1 << 24
    hx: npt.NDArray[np.uint8] = rng.integers(0, 256, hn, dtype=np.uint8)
    h_bins: npt.NDArray[np.int64] = np.zeros(256, dtype=np.int64)
    h_ref: npt.NDArray[np.intp] = np.bincount(hx, minlength=256)
    thx: torch.Tensor = tt(hx)

    def histogram_case(func: str, how: str) -> Case:
        def run(k: achainsaw.Kernel) -> None:
            h_bins[:] = 0
            k.run(func, hx, hn, h_bins)

        def verify(k: achainsaw.Kernel) -> float:
            run(k)
            return float(not np.array_equal(h_bins, h_ref))

        return Case(
            name=func, file="histogram", label=f"Byte histogram, {hn >> 20} MB, {how}",
            run=run, verify=verify, tol=0.0,
            numpy=lambda: np.bincount(hx, minlength=256),
            torch=lambda: torch.bincount(thx, minlength=256),
            torch_check=lambda out: float(not np.array_equal(out.numpy(), h_ref)),
            flops=hn,
        )

    cases.append(histogram_case("histogram", "all cores"))
    cases.append(histogram_case("histogram_atomic", "one atomic add per byte, all cores"))

    # Sparse matrix-vector product in CSR form with power-law row lengths (Zipf, as in graph
    # adjacency matrices): most rows have a few nonzeros, a few have tens of thousands. Each
    # row's columns are sorted and distinct (canonical CSR, as scipy and PyTorch keep it).
    sm: int = 1 << 18
    s_lens: npt.NDArray[np.int64] = np.minimum(rng.zipf(1.8, sm), 1 << 16).astype(np.int64)
    s_keys: npt.NDArray[np.int64] = np.unique(
        np.repeat(np.arange(sm, dtype=np.int64), s_lens) * sm
        + rng.integers(0, sm, int(s_lens.sum()), dtype=np.int64))
    s_rowptr: npt.NDArray[np.int64] = np.concatenate(
        [[0], np.cumsum(np.bincount(s_keys // sm, minlength=sm))]).astype(np.int64)
    s_nnz: int = int(s_rowptr[-1])
    s_cols: npt.NDArray[np.int32] = (s_keys % sm).astype(np.int32)
    s_vals: F32 = rng.standard_normal(s_nnz).astype(np.float32)
    s_x: F32 = rng.standard_normal(sm).astype(np.float32)
    s_y: F32 = np.zeros(sm, dtype=np.float32)
    # Every row has a nonzero, so reduceat sums exactly each row's terms.
    spmv_ref: F64 = np.add.reduceat(s_vals.astype(np.float64) * s_x[s_cols], s_rowptr[:-1])
    if torch is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # CSR tensors are a beta feature
            t_csr: torch.Tensor = torch.sparse_csr_tensor(
                tt(s_rowptr), tt(s_cols.astype(np.int64)), tt(s_vals), (sm, sm),
                check_invariants=True)
    t_sx: torch.Tensor = tt(s_x)

    def spmv_case(func: str, how: str) -> Case:
        def run(k: achainsaw.Kernel) -> None:
            k.run(func, s_rowptr, s_cols, s_vals, s_x, s_y, sm)

        def verify(k: achainsaw.Kernel) -> float:
            s_y[:] = np.nan
            run(k)
            return rel_err(s_y, spmv_ref)

        return Case(
            name=func, file="spmv",
            label=f"SpMV, CSR {sm}x{sm} with {s_nnz / 1e6:.1f}M power-law nonzeros, {how}",
            run=run, verify=verify, tol=1e-5,
            # NumPy has no sparse matrices: the products, then each row's sum.
            numpy=lambda: np.add.reduceat(s_vals * s_x[s_cols], s_rowptr[:-1]),
            torch=lambda: torch.mv(t_csr, t_sx),
            torch_check=lambda out: rel_err(out, spmv_ref),
            flops=2 * s_nnz,
        )

    cases.append(spmv_case("spmv", "all cores"))
    cases.append(spmv_case("spmv_queue", "atomic work queue instead of par, all cores"))
    cases.append(spmv_case("spmv_gather", "vgather of x instead of scalar loads, all cores"))

    # The same matrix in spmv_panels' layout: the nonzeros ordered by (task of 1024 rows,
    # panel of 65536 columns, row), with their columns as u16 offsets in the panel. The
    # NumPy and PyTorch conversions are a stable sort by task and panel.
    p_count: int = (sm + 65535) >> 16
    p_rows: npt.NDArray[np.int64] = np.repeat(np.arange(sm, dtype=np.int64), np.diff(s_rowptr))
    p_key: npt.NDArray[np.int64] = (p_rows >> 10) * p_count + (s_cols >> 16)
    p_groups: int = ((sm + 1023) >> 10) * p_count

    def panels_numpy() -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int32],
                                npt.NDArray[np.int32], U16, F32]:
        order: npt.NDArray[np.int64] = np.argsort(p_key, kind="stable")
        key: npt.NDArray[np.int64] = p_key[order]
        rows: npt.NDArray[np.int64] = p_rows[order]
        first: npt.NDArray[np.bool_] = np.ones(s_nnz, dtype=np.bool_)
        first[1:] = (key[1:] != key[:-1]) | (rows[1:] != rows[:-1])
        at: npt.NDArray[np.int64] = np.flatnonzero(first)
        grp: npt.NDArray[np.int64] = np.searchsorted(key[at], np.arange(p_groups + 1))
        start: npt.NDArray[np.int32] = np.append(at, s_nnz).astype(np.int32)
        return (grp, rows[at].astype(np.int32), start, (s_cols[order] & 0xffff).astype(np.uint16),
                s_vals[order])

    def panels_torch() -> tuple[Any, Any, Any, Any, Any]:
        key: Any = tt(p_key)
        order: Any = torch.sort(key, stable=True).indices
        k: Any = key[order]
        rows: Any = tt(p_rows)[order]
        first: Any = torch.ones(s_nnz, dtype=torch.bool)
        first[1:] = (k[1:] != k[:-1]) | (rows[1:] != rows[:-1])
        at: Any = torch.nonzero(first).flatten()
        grp: Any = torch.searchsorted(k[at], torch.arange(p_groups + 1))
        start: Any = torch.cat([at, torch.tensor([s_nnz])]).int()
        return (grp, rows[at].int(), start, (tt(s_cols)[order] & 0xffff).short(),
                tt(s_vals)[order])

    p_ref: tuple[npt.NDArray[np.int64], npt.NDArray[np.int32], npt.NDArray[np.int32], U16,
                 F32] = panels_numpy()
    p_grp: npt.NDArray[np.int64] = np.zeros(p_groups + 1, dtype=np.int64)
    p_row: npt.NDArray[np.int32] = np.zeros(len(p_ref[1]), dtype=np.int32)
    p_start: npt.NDArray[np.int32] = np.zeros(len(p_ref[2]), dtype=np.int32)
    p_cols: U16 = np.zeros(s_nnz, dtype=np.uint16)
    p_vals: F32 = np.zeros(s_nnz, dtype=np.float32)

    def pack_run(k: achainsaw.Kernel) -> None:
        npairs: int = k.run("spmv_panels_count", s_rowptr, s_cols, sm, sm, p_grp)
        k.run("spmv_panels_fill", s_rowptr, s_cols, s_vals, sm, sm, p_grp, p_row, p_start,
              p_cols, p_vals, npairs)

    def pack_check(out: tuple[Any, ...]) -> float:
        same: bool = all(np.array_equal(np.asarray(a).view(np.asarray(b).dtype), b)
                         for a, b in zip(out, p_ref))
        return float(not same)

    def pack_verify(k: achainsaw.Kernel) -> float:
        for a in (p_grp, p_row, p_start, p_cols, p_vals):
            a.fill(0)
        pack_run(k)
        return pack_check((p_grp, p_row, p_start, p_cols, p_vals))

    cases.append(Case(
        name="spmv_pack", file="spmv",
        label=f"CSR to spmv_panels' layout, {s_nnz / 1e6:.1f}M nonzeros, all cores",
        run=pack_run, verify=pack_verify, tol=0.0,
        numpy=panels_numpy, torch=panels_torch,
        torch_check=lambda out: pack_check(tuple(t.numpy() for t in out)),
        flops=s_nnz,
    ))

    def panels_run(k: achainsaw.Kernel) -> None:
        k.run("spmv_panels", p_grp, p_row, p_start, p_cols, p_vals, s_x, s_y, sm, sm)

    def panels_verify(k: achainsaw.Kernel) -> float:
        pack_run(k)
        s_y[:] = np.nan
        panels_run(k)
        return rel_err(s_y, spmv_ref)

    cases.append(Case(
        name="spmv_panels", file="spmv",
        label="SpMV, the same matrix in column panels (spmv_pack), all cores",
        run=panels_run, verify=panels_verify, tol=1e-5,
        # The CSR products, as for spmv.
        numpy=lambda: np.add.reduceat(s_vals * s_x[s_cols], s_rowptr[:-1]),
        torch=lambda: torch.mv(t_csr, t_sx),
        torch_check=lambda out: rel_err(out, spmv_ref),
        flops=2 * s_nnz,
    ))
    return cases


def isa_levels(mode: str) -> list[str | None]:
    report: dict[str, Any] = achainsaw.cpu_features()
    host: dict[str, Any] = report["host"]
    order: list[str | None] = list(ISA_ORDER.get(host["arch"], []))
    top: str | None = host.get("max_isa")
    if mode == "host" or top not in order:
        return [None]
    if mode == "all":
        return order[: order.index(top) + 1]
    return [level.strip() for level in mode.split(",")]


def main() -> None:
    assert __doc__ is not None
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0])
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
    args: argparse.Namespace = parser.parse_args()
    use_torch: bool = torch is not None and not args.no_torch

    available: list[str] = achainsaw.available_backends()
    backends: list[str] = available if args.backends == "all" else args.backends.split(",")
    levels: list[str | None] = isa_levels(args.isa)
    budget: float = 0.02 if args.quick else 0.3
    cases: list[Case] = make_cases()
    original_cap: str | None = achainsaw.get_isa_cap()

    print("=" * 96)
    print("CHAINSAW-BLAS KERNEL BENCHMARKS")
    print(f"backends: {', '.join(backends)}  |  ISA levels: "
          f"{', '.join(l or 'host' for l in levels)}  |  host: "
          f"{achainsaw.cpu_features()['host']['max_isa']}")
    if use_torch:
        print(f"PyTorch {torch.__version__}, {torch.get_num_threads()} threads")
    print("=" * 96)

    results: list[dict[str, Any]] = []
    failures: list[str] = []
    configs: list[tuple[str, str | None]] = [(b, l) for b in backends for l in levels]
    for case in cases:
        src: str = source(case.get("file", case["name"]))
        # Both runtimes busy-wait for a moment after parallel work; let the other's threads
        # settle before timing each one.
        time.sleep(0.02)
        np_s: float | None = None if args.no_numpy else time_call(case["numpy"], budget)
        row: dict[str, Any] = dict(kernel=case["name"], label=case["label"],
                   numpy_us=np_s and np_s * 1e6, torch_us=None, runs=[])
        print(f"\n{case['label']}")
        if np_s:
            print(f"    {'NumPy':<22} {np_s * 1e6:10.2f} us  "
                  f"{case['flops'] / np_s / 1e9:8.2f} GFLOP/s")
        if use_torch:
            err: float = case["torch_check"](case["torch"]())
            tol: float = case.get("torch_tol", case["tol"])
            if err > tol:
                failures.append(f"{case['name']} on PyTorch: error {err:.2e}")
            time.sleep(0.02)
            pt_s: float = time_call(case["torch"], budget)
            row["torch_us"] = pt_s * 1e6
            row["torch_error"] = err
            vs: str = f"  {np_s / pt_s:6.2f}x NumPy" if np_s else ""
            print(f"    {'PyTorch':<22} {pt_s * 1e6:10.2f} us  "
                  f"{case['flops'] / pt_s / 1e9:8.2f} GFLOP/s{vs}  err {err:.1e} "
                  f"{'PASS' if err <= tol else 'FAIL'}")
        for backend, level in configs:
            achainsaw.set_isa_cap(level)
            kernel: achainsaw.Kernel = achainsaw.compile(src, backend=backend, fast_math=True)
            time.sleep(0.02)
            err = case["verify"](kernel)
            ok: bool = err <= case["tol"]
            if not ok:
                failures.append(f"{case['name']} on {backend}/{level or 'host'}: "
                                f"error {err:.2e}")
            secs: float = time_call(lambda: case["run"](kernel), budget)
            tag: str = f"{backend}@{level or 'host'}"
            vs = f"  {np_s / secs:6.2f}x NumPy" if np_s else ""
            if row["torch_us"]:
                vs += f"  {row['torch_us'] / 1e6 / secs:6.2f}x PyTorch"
            print(f"    {tag:<22} {secs * 1e6:10.2f} us  "
                  f"{case['flops'] / secs / 1e9:8.2f} GFLOP/s{vs}  err {err:.1e} "
                  f"{'PASS' if ok else 'FAIL'}")
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
