"""
A/B performance check of two builds on one machine, for CI.

    python benchmarks/perf_compare.py BASE_DIR HEAD_DIR [--rounds 5] [--threshold 0.25]
        [--summary PATH] [--allow-regressions]

Each DIR is a checkout with its achainsaw extension (achainsaw.so, or .pyd on Windows) at the
root. Each side runs its own benchmark scripts, so a change to a kernel's source counts the
same as a change to the compiler:
  benchmark_kernels.py   every kernel on every backend (NumPy timing skipped)
  benchmark_decode.py    llama_decode.air on a 4-layer random Q4_0 model
The sides alternate for --rounds rounds, in ABBA order to cancel drift, each run in a fresh
process. Every (benchmark, backend) keeps its best time over the rounds: noise only ever
adds time, and a process can be unlucky as a whole (memory layout), so the minimum is the
stable statistic. A result is a regression when the head's best is slower than the base's
by more than --threshold (relative) and --min-us (absolute).

Exit status: 1 on a regression (0 with --allow-regressions), 2 if the head's benchmarks fail.
If the base's fail (say, a base too old for these scripts), nothing is compared and the
status is 0, with a note in the summary. Benchmarks only one side has are listed but not
compared.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time


def side_env(root):
    paths = [root, os.path.join(root, "benchmarks"), os.environ.get("PYTHONPATH", "")]
    return dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in paths if p))


def supports(script, flag, env):
    """Whether `script --help` (run as the benchmarks run) lists `flag`; a script that
    cannot even print its help is an error, not a missing option."""
    out = subprocess.run([sys.executable, script, "--help"], capture_output=True, text=True,
                         env=env)
    if out.returncode != 0:
        raise RuntimeError(f"{script} --help failed:\n{out.stderr[-3000:]}")
    return flag in out.stdout


def run_side(root, workdir, tag, skipped):
    """One round of benchmarks for the checkout at `root`: {(benchmark, backend): us}.
    Benchmarks this side's scripts cannot run are added to `skipped`."""
    env = side_env(root)
    times = {}
    kernels = os.path.join(root, "benchmarks", "benchmark_kernels.py")
    out = os.path.join(workdir, f"{tag}-kernels.json")
    cmd = [sys.executable, kernels, "--json", out]
    if supports(kernels, "--no-numpy", env):
        cmd.append("--no-numpy")
    res = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=root)
    if res.returncode != 0:
        raise RuntimeError(f"{kernels} failed:\n{res.stdout[-3000:]}\n{res.stderr[-3000:]}")
    for r in json.load(open(out, encoding="utf-8"))["results"]:
        for run in r["runs"]:
            if run["isa"] == "host":
                times[(r["kernel"], run["backend"])] = run["us"]

    decode = os.path.join(root, "benchmarks", "benchmark_decode.py")
    if not (os.path.exists(decode) and supports(decode, "--json", env)):
        skipped.add(f"{tag}: llama_decode (benchmark_decode.py has no --json)")
    else:
        out = os.path.join(workdir, f"{tag}-decode.json")
        cmd = [sys.executable, decode, "--layers", "4", "--tokens", "16", "--weights", "q4",
               "--no-numpy", "--json", out]
        res = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=root)
        if res.returncode != 0:
            raise RuntimeError(f"{decode} failed:\n{res.stdout[-3000:]}\n{res.stderr[-3000:]}")
        for r in json.load(open(out, encoding="utf-8"))["results"]:
            times[("llama_decode (4 layers, Q4_0)", r["backend"])] = r["ms_per_token"] * 1e3
    return times


def fmt_us(us):
    if us >= 1000:
        return f"{us / 1000:.2f} ms"
    return f"{us:.2f} µs" if us < 10 else f"{us:.1f} µs"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("base")
    ap.add_argument("head")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.25,
                    help="relative slowdown that fails (default: %(default)s)")
    ap.add_argument("--min-us", type=float, default=0.3,
                    help="ignore slowdowns smaller than this many µs (default: %(default)s)")
    ap.add_argument("--summary", help="append a Markdown report (e.g. $GITHUB_STEP_SUMMARY)")
    ap.add_argument("--allow-regressions", action="store_true",
                    help="report regressions without failing")
    args = ap.parse_args()

    sides = {"base": os.path.abspath(args.base), "head": os.path.abspath(args.head)}
    best = {"base": {}, "head": {}}
    skipped = set()
    with tempfile.TemporaryDirectory() as workdir:
        for rnd in range(args.rounds):
            order = ["base", "head"] if rnd % 2 == 0 else ["head", "base"]
            for side in order:
                t0 = time.perf_counter()
                try:
                    times = run_side(sides[side], workdir, side, skipped)
                except RuntimeError as e:
                    print(f"[{side}] {e}", file=sys.stderr)
                    if side == "head":
                        sys.exit(2)
                    note = f"The base's benchmarks failed, so nothing was compared:\n```\n{e}\n```\n"
                    print(note)
                    if args.summary:
                        with open(args.summary, "a", encoding="utf-8") as f:
                            f.write(note)
                    sys.exit(0)
                for key, us in times.items():
                    best[side][key] = min(us, best[side].get(key, float("inf")))
                print(f"round {rnd + 1}/{args.rounds} {side}: {len(times)} results "
                      f"in {time.perf_counter() - t0:.0f} s", flush=True)

    rows, regressions = [], []
    for key in sorted(set(best["base"]) | set(best["head"])):
        b, h = best["base"].get(key), best["head"].get(key)
        if b is None or h is None:
            rows.append((key, b, h, None, "only in " + ("head" if b is None else "base")))
            continue
        ratio = h / b
        status = ""
        if ratio > 1 + args.threshold and h - b > args.min_us:
            status = "slower"
            regressions.append(key)
        elif ratio < 1 / (1 + args.threshold) and b - h > args.min_us:
            status = "faster"
        rows.append((key, b, h, ratio, status))
    rows.sort(key=lambda r: -(r[3] or 0))

    lines = [f"### Performance: head vs base (best of {args.rounds} runs each)", "",
             "| Benchmark | Backend | Base | Head | Head / base | |",
             "|---|---|---|---|---|---|"]
    for (name, backend), b, h, ratio, status in rows:
        mark = {"slower": "**slower**", "faster": "faster"}.get(status, status)
        lines.append(f"| {name} | {backend} | {fmt_us(b) if b else '-'} | "
                     f"{fmt_us(h) if h else '-'} | {f'{ratio:.2f}' if ratio else '-'} | {mark} |")
    verdict = (f"{len(regressions)} benchmark(s) slower by more than "
               f"{args.threshold:.0%}" if regressions else
               f"No benchmark slower by more than {args.threshold:.0%}.")
    if regressions and args.allow_regressions:
        verdict += " Allowed by the `perf-regression-ok` label."
    lines += ["", verdict]
    lines += [f"Skipped: {s}." for s in sorted(skipped)]
    report = "\n".join(lines) + "\n"
    print(report)
    if args.summary:
        with open(args.summary, "a", encoding="utf-8") as f:
            f.write(report)
    sys.exit(1 if regressions and not args.allow_regressions else 0)


if __name__ == "__main__":
    main()
