"""
Chainsaw-BLAS: High-Performance Agent AI Kernel Library Benchmark & Verification.
Benchmarks 128-bit SIMD AIR kernels against NumPy and pure Python:
1. Cosine Similarity (Embedding search & RAG retrieval)
2. Euclidean Distance (L2 Nearest Neighbor search)
3. Numerically Stable Softmax (Attention head weighting)
4. RMSNorm (Transformer token normalization for LLaMA/Mistral/Gemma)
5. GEMV (Matrix-Vector linear projection)
"""

import os
import sys
import time
import numpy as np

# Ensure achainsaw module is accessible
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import achainsaw

KERNELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "examples", "kernels"))


def benchmark_op(func, iters=5000):
    # Warmup
    for _ in range(50):
        func()
    t0 = time.perf_counter()
    for _ in range(iters):
        func()
    t1 = time.perf_counter()
    total_sec = t1 - t0
    avg_us = (total_sec / iters) * 1e6
    ops_sec = iters / total_sec
    return avg_us, ops_sec


def run_benchmarks():
    print("=" * 80)
    print("CHAINSAW-BLAS: HIGH-PERFORMANCE AGENT AI KERNEL BENCHMARKS")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. Cosine Similarity
    # -------------------------------------------------------------------------
    cos_path = os.path.join(KERNELS_DIR, "cosine_similarity.air")
    with open(cos_path, "r", encoding="utf-8") as f:
        cos_source = f.read()

    cos_kernel = achainsaw.compile(cos_source)
    dim = 1024  # 1024-dim embedding (standard for OpenAI text-embedding-3 / Voyage)
    vecs = dim // 4

    np.random.seed(42)
    a = np.random.randn(dim).astype(np.float32)
    b = np.random.randn(dim).astype(np.float32)

    # Verification
    air_cos = cos_kernel(a, b, vecs)
    np_cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    err_cos = abs(air_cos - np_cos)
    assert err_cos < 1e-4, f"Cosine similarity mismatch: AIR={air_cos} vs NP={np_cos}"

    us_cos_air, ops_cos_air = benchmark_op(lambda: cos_kernel(a, b, vecs))
    us_cos_np, ops_cos_np = benchmark_op(lambda: np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    print(f"\n[1] Cosine Similarity (dim={dim}):")
    print(f"    Verification: Max Error = {err_cos:.2e} [PASS]")
    print(f"    achainsaw (SIMD JIT): {us_cos_air:6.2f} µs | {ops_cos_air:10,.0f} ops/sec")
    print(f"    NumPy (Standard):     {us_cos_np:6.2f} µs | {ops_cos_np:10,.0f} ops/sec")
    print(f"    Speedup vs NumPy:     {us_cos_np / us_cos_air:6.2f}x")

    # -------------------------------------------------------------------------
    # 2. Euclidean Distance (L2)
    # -------------------------------------------------------------------------
    l2_path = os.path.join(KERNELS_DIR, "euclidean_distance.air")
    with open(l2_path, "r", encoding="utf-8") as f:
        l2_source = f.read()

    l2_kernel = achainsaw.compile(l2_source)
    air_l2 = l2_kernel(a, b, vecs)
    np_l2 = float(np.linalg.norm(a - b))
    err_l2 = abs(air_l2 - np_l2)
    assert err_l2 < 1e-4, f"L2 distance mismatch: AIR={air_l2} vs NP={np_l2}"

    us_l2_air, ops_l2_air = benchmark_op(lambda: l2_kernel(a, b, vecs))
    us_l2_np, ops_l2_np = benchmark_op(lambda: np.linalg.norm(a - b))

    print(f"\n[2] Euclidean Distance L2 (dim={dim}):")
    print(f"    Verification: Max Error = {err_l2:.2e} [PASS]")
    print(f"    achainsaw (SIMD JIT): {us_l2_air:6.2f} µs | {ops_l2_air:10,.0f} ops/sec")
    print(f"    NumPy (Standard):     {us_l2_np:6.2f} µs | {ops_l2_np:10,.0f} ops/sec")
    print(f"    Speedup vs NumPy:     {us_l2_np / us_l2_air:6.2f}x")

    # -------------------------------------------------------------------------
    # 3. Softmax
    # -------------------------------------------------------------------------
    sm_path = os.path.join(KERNELS_DIR, "softmax.air")
    with open(sm_path, "r", encoding="utf-8") as f:
        sm_source = f.read()

    sm_kernel = achainsaw.compile(sm_source)
    sm_dim = 256  # 256-token attention sequence
    x_sm = np.random.randn(sm_dim).astype(np.float32)
    out_air_sm = np.zeros(sm_dim, dtype=np.float32)

    sm_kernel(x_sm, out_air_sm, sm_dim)
    shift_x = x_sm - np.max(x_sm)
    np_sm = np.exp(shift_x) / np.sum(np.exp(shift_x))
    err_sm = float(np.max(np.abs(out_air_sm - np_sm)))
    assert err_sm < 1e-4, f"Softmax mismatch: max error = {err_sm}"

    us_sm_air, ops_sm_air = benchmark_op(lambda: sm_kernel(x_sm, out_air_sm, sm_dim))
    us_sm_np, ops_sm_np = benchmark_op(lambda: np.exp(x_sm - np.max(x_sm)) / np.sum(np.exp(x_sm - np.max(x_sm))))

    print(f"\n[3] Softmax (tokens={sm_dim}):")
    print(f"    Verification: Max Error = {err_sm:.2e} [PASS]")
    print(f"    achainsaw (JIT):      {us_sm_air:6.2f} µs | {ops_sm_air:10,.0f} ops/sec")
    print(f"    NumPy (Standard):     {us_sm_np:6.2f} µs | {ops_sm_np:10,.0f} ops/sec")
    print(f"    Speedup vs NumPy:     {us_sm_np / us_sm_air:6.2f}x")

    # -------------------------------------------------------------------------
    # 4. RMSNorm
    # -------------------------------------------------------------------------
    rms_path = os.path.join(KERNELS_DIR, "rmsnorm.air")
    with open(rms_path, "r", encoding="utf-8") as f:
        rms_source = f.read()

    rms_kernel = achainsaw.compile(rms_source)
    rms_dim = 1024
    rms_vecs = rms_dim // 4
    x_rms = np.random.randn(rms_dim).astype(np.float32)
    w_rms = np.random.uniform(0.8, 1.2, size=rms_dim).astype(np.float32)
    out_air_rms = np.zeros(rms_dim, dtype=np.float32)

    rms_kernel(x_rms, w_rms, out_air_rms, rms_vecs, float(1.0 / rms_dim))
    np_rms = (x_rms / np.sqrt(np.mean(x_rms**2) + 1e-6)) * w_rms
    err_rms = float(np.max(np.abs(out_air_rms - np_rms)))
    assert err_rms < 1e-4, f"RMSNorm mismatch: max error = {err_rms}"

    us_rms_air, ops_rms_air = benchmark_op(lambda: rms_kernel(x_rms, w_rms, out_air_rms, rms_vecs, float(1.0 / rms_dim)))
    us_rms_np, ops_rms_np = benchmark_op(lambda: (x_rms / np.sqrt(np.mean(x_rms**2) + 1e-6)) * w_rms)

    print(f"\n[4] RMSNorm (dim={rms_dim}):")
    print(f"    Verification: Max Error = {err_rms:.2e} [PASS]")
    print(f"    achainsaw (SIMD JIT): {us_rms_air:6.2f} µs | {ops_rms_air:10,.0f} ops/sec")
    print(f"    NumPy (Standard):     {us_rms_np:6.2f} µs | {ops_rms_np:10,.0f} ops/sec")
    print(f"    Speedup vs NumPy:     {us_rms_np / us_rms_air:6.2f}x")

    # -------------------------------------------------------------------------
    # 5. GEMV (Matrix-Vector Multiplication)
    # -------------------------------------------------------------------------
    gemv_path = os.path.join(KERNELS_DIR, "gemv_f32.air")
    with open(gemv_path, "r", encoding="utf-8") as f:
        gemv_source = f.read()

    gemv_kernel = achainsaw.compile(gemv_source)
    M = 128
    K = 512
    k_vecs = K // 4
    A = np.random.randn(M, K).astype(np.float32)
    x_gemv = np.random.randn(K).astype(np.float32)
    y_air = np.zeros(M, dtype=np.float32)

    gemv_kernel(A, x_gemv, y_air, M, k_vecs)
    np_gemv = np.dot(A, x_gemv)
    err_gemv = float(np.max(np.abs(y_air - np_gemv)))
    assert err_gemv < 1e-3, f"GEMV mismatch: max error = {err_gemv}"

    us_gemv_air, ops_gemv_air = benchmark_op(lambda: gemv_kernel(A, x_gemv, y_air, M, k_vecs))
    us_gemv_np, ops_gemv_np = benchmark_op(lambda: np.dot(A, x_gemv))

    print(f"\n[5] GEMV (M={M}, K={K}):")
    print(f"    Verification: Max Error = {err_gemv:.2e} [PASS]")
    print(f"    achainsaw (SIMD JIT): {us_gemv_air:6.2f} µs | {ops_gemv_air:10,.0f} ops/sec")
    print(f"    NumPy (Standard):     {us_gemv_np:6.2f} µs | {ops_gemv_np:10,.0f} ops/sec")
    print(f"    Speedup vs NumPy:     {us_gemv_np / us_gemv_air:6.2f}x")

    print("\n" + "=" * 80)
    print("[SUCCESS] ALL 5 KERNELS PASSED NUMERICAL VERIFICATION & BENCHMARKS!")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmarks()
