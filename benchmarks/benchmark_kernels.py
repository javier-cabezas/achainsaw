"""
Chainsaw-BLAS: verification and benchmarks for the kernels in examples/kernels/,
including flash attention at DeepSeek V4 Pro's decode size.

Every kernel is checked against NumPy, then timed on each available code generation
backend (Cranelift, and LLVM when the build has it) and, with --isa all, at every ISA
level this machine reaches (for example sse, avx, avx2 and avx512 on an AVX-512 host).
The kernels are vector-length agnostic, so the same source runs 128-bit vectors on
Cranelift and up to 512-bit (or SVE-scalable) vectors on LLVM.

    python benchmarks/benchmark_kernels.py                  # host ISA, all backends
    python benchmarks/benchmark_kernels.py --isa all        # sweep ISA levels too
    python benchmarks/benchmark_kernels.py --quick          # short run (CI)
    python benchmarks/benchmark_kernels.py --json out.json  # machine-readable results

Exits non-zero if any kernel disagrees with NumPy.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

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


def time_call(func, budget_s, min_iters=5):
    """Average seconds per call, running for about `budget_s` after a short warm-up."""
    for _ in range(3):
        func()
    iters, t0 = 0, time.perf_counter()
    while True:
        func()
        iters += 1
        elapsed = time.perf_counter() - t0
        if iters >= min_iters and elapsed >= budget_s:
            return elapsed / iters


def make_cases():
    """Each case: name, label, kernel source, args builder, verify(kernel) -> max error,
    tolerance, NumPy baseline, and the work per call (for GFLOP/s)."""
    rng = np.random.default_rng(42)
    cases = []

    dim = 1024  # embedding size of common text-embedding models
    a = rng.standard_normal(dim).astype(np.float32)
    b = rng.standard_normal(dim).astype(np.float32)
    cos_ref = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    cases.append(dict(
        name="cosine_similarity", label=f"Cosine similarity (n={dim})",
        run=lambda k: k(a, b, dim),
        verify=lambda k: abs(k(a, b, dim) - cos_ref), tol=1e-5,
        numpy=lambda: np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12),
        flops=6 * dim,
    ))

    l2_ref = float(np.linalg.norm(a - b))
    cases.append(dict(
        name="euclidean_distance", label=f"Euclidean distance (n={dim})",
        run=lambda k: k(a, b, dim),
        verify=lambda k: abs(k(a, b, dim) - l2_ref) / l2_ref, tol=1e-5,
        numpy=lambda: np.linalg.norm(a - b),
        flops=3 * dim,
    ))

    sm_n = 1000
    x_sm = (rng.standard_normal(sm_n) * 4.0).astype(np.float32)
    out_sm = np.zeros(sm_n, dtype=np.float32)
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
        verify=verify_softmax, tol=1e-6, numpy=np_softmax, flops=4 * sm_n,
    ))

    rms_n = 4096
    x_r = rng.standard_normal(rms_n).astype(np.float32)
    w_r = rng.uniform(0.8, 1.2, rms_n).astype(np.float32)
    out_r = np.zeros(rms_n, dtype=np.float32)
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
        flops=4 * rms_n,
    ))

    gm, gk = 512, 1024
    A = rng.standard_normal((gm, gk)).astype(np.float32)
    xg = rng.standard_normal(gk).astype(np.float32)
    yg = np.zeros(gm, dtype=np.float32)
    gemv_ref = A.astype(np.float64) @ xg.astype(np.float64)

    def verify_gemv(k):
        k(A, xg, yg, gm, gk)
        return float(np.max(np.abs(yg - gemv_ref)) / np.max(np.abs(gemv_ref)))

    cases.append(dict(
        name="gemv_f32", label=f"GEMV f32 ({gm}x{gk})",
        run=lambda k: k(A, xg, yg, gm, gk),
        verify=verify_gemv, tol=1e-5, numpy=lambda: A @ xg, flops=2 * gm * gk,
    ))

    def verify_gemv_par(k):
        yg[:] = np.nan
        k.run("gemv_par", A, xg, yg, gm, gk)
        return float(np.max(np.abs(yg - gemv_ref)) / np.max(np.abs(gemv_ref)))

    cases.append(dict(
        name="gemv_par", label=f"GEMV f32, all cores via par ({gm}x{gk})",
        run=lambda k: k.run("gemv_par", A, xg, yg, gm, gk),
        verify=verify_gemv_par, tol=1e-5, numpy=lambda: A @ xg, flops=2 * gm * gk,
    ))

    n = 256
    Ab = bf16_bits(rng.standard_normal((n, n)))
    Bb = bf16_bits(rng.standard_normal((n, n)))
    Af, Bf = bf16_values(Ab), bf16_values(Bb)
    C = np.zeros((n, n), dtype=np.float32)
    gemm_ref = Af.astype(np.float64) @ Bf.astype(np.float64)

    def verify_gemm(k):
        C[:] = 0.0
        k(C, Ab, Bb, n, n, n)
        return float(np.max(np.abs(C - gemm_ref)) / np.max(np.abs(gemm_ref)))

    cases.append(dict(
        name="gemm_bf16", label=f"GEMM bf16->f32 ({n}x{n}x{n})",
        run=lambda k: k(C, Ab, Bb, n, n, n),
        verify=verify_gemm, tol=1e-4,
        # NumPy has no bf16 matmul; compare with its f32 matmul on the same values.
        numpy=lambda: Af @ Bf, flops=2 * n * n * n,
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
    qn, kvn = bf16_values(fq), bf16_values(fkv)

    def verify_flash(k):
        k(fq, fkv, fidx, fsink, fout, fh, fd, fnk, fscale)
        return float(np.max(np.abs(fout - flash_ref)) / np.max(np.abs(flash_ref)))

    def np_flash():
        s = (qn @ kvn[fidx].T) * fscale
        m = s.max(axis=1, keepdims=True)
        e = np.exp(s - m)
        return (e @ kvn[fidx]) / (e.sum(axis=1, keepdims=True) + np.exp(fsink[:, None] - m))

    cases.append(dict(
        name="flash_attention", label=f"Flash attention decode, DeepSeek V4 Pro ({fh}x{fd}, {fnk} keys)",
        run=lambda k: k(fq, fkv, fidx, fsink, fout, fh, fd, fnk, fscale),
        verify=verify_flash, tol=1e-2,
        # NumPy has no bf16 matmul; compare with f32 on the same values (multithreaded BLAS).
        numpy=np_flash, flops=2 * 2 * fh * fnk * fd,
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
    args = parser.parse_args()

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
    print("=" * 96)

    results, failures = [], []
    configs = [(b, l) for b in backends for l in levels]
    for case in cases:
        src = source(case["name"])
        np_s = time_call(case["numpy"], budget)
        row = dict(kernel=case["name"], label=case["label"], numpy_us=np_s * 1e6, runs=[])
        print(f"\n{case['label']}")
        print(f"    {'NumPy':<22} {np_s * 1e6:10.2f} us  {case['flops'] / np_s / 1e9:8.2f} GFLOP/s")
        for backend, level in configs:
            achainsaw.set_isa_cap(level)
            kernel = achainsaw.compile(src, backend=backend)
            err = case["verify"](kernel)
            ok = err <= case["tol"]
            if not ok:
                failures.append(f"{case['name']} on {backend}/{level or 'host'}: error {err:.2e}")
            secs = time_call(lambda: case["run"](kernel), budget)
            tag = f"{backend}@{level or 'host'}"
            print(f"    {tag:<22} {secs * 1e6:10.2f} us  {case['flops'] / secs / 1e9:8.2f} GFLOP/s"
                  f"  {np_s / secs:6.2f}x NumPy  err {err:.1e} {'PASS' if ok else 'FAIL'}")
            row["runs"].append(dict(backend=backend, isa=level or "host", us=secs * 1e6,
                                    gflops=case["flops"] / secs / 1e9, error=err, ok=ok))
        results.append(row)
    achainsaw.set_isa_cap(original_cap)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(dict(backends=backends, isa_levels=[l or "host" for l in levels],
                           cpu=achainsaw.cpu_features(), results=results), f, indent=2)

    print("\n" + "=" * 96)
    if failures:
        print("[FAIL] " + "; ".join(failures))
        sys.exit(1)
    print(f"[SUCCESS] ALL {len(cases)} KERNELS PASSED NUMERICAL VERIFICATION "
          f"ON {len(configs)} BACKEND/ISA CONFIGURATIONS")


if __name__ == "__main__":
    main()
