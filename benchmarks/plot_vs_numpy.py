"""
Draws the kernels' speedup over NumPy from `benchmark_kernels.py --json` results, as light
and dark SVG figures for the README (selected with <picture> and prefers-color-scheme).

    python benchmarks/benchmark_kernels.py --json results.json
    python benchmarks/plot_vs_numpy.py results.json docs/kernels-vs-numpy

One row per kernel, sorted by the LLVM speedup; each backend is a bar growing from the 1x
line (NumPy) on a log scale, left when slower and right when faster. Colors and chrome
follow the data-viz reference palette (categorical slots 1-2, validated for both modes).
"""

import json
import math
import sys

SHORT = {
    "cosine_similarity": "Cosine similarity, n=1024",
    "euclidean_distance": "Euclidean distance, n=1024",
    "softmax": "Softmax, n=1000",
    "rmsnorm": "RMSNorm, n=4096",
    "gemv_f32": "GEMV f32 512x1024, 1 core",
    "gemv_par": "GEMV f32 512x1024, all cores",
    "gemm_bf16": "GEMM bf16 256³, all cores",
    "flash_attention": "Flash attention decode, all cores",
    "swiglu": "SwiGLU, n=14336",
    "argmax": "Greedy argmax, 128256 logits",
    "rope": "RoPE, 32 heads x 128",
    "add_rmsnorm": "Residual add + RMSNorm, n=4096",
    "q8_gemv": "Q8_0 GEMV 4096x4096, all cores",
}

THEMES = {
    "light": dict(surface="#fcfcfb", primary="#0b0b0b", secondary="#52514e",
                  muted="#898781", grid="#e1e0d9", baseline="#c3c2b7",
                  series={"cranelift": "#2a78d6", "llvm": "#eb6834"}),
    "dark": dict(surface="#1a1a19", primary="#ffffff", secondary="#c3c2b7",
                 muted="#898781", grid="#2c2c2a", baseline="#383835",
                 series={"cranelift": "#3987e5", "llvm": "#d95926"}),
}
NAMES = {"cranelift": "Cranelift (128-bit vectors)", "llvm": "LLVM (AVX-512)"}
FONT = 'system-ui, -apple-system, &quot;Segoe UI&quot;, sans-serif'


def bar_path(x0, x1, y, h, r=4.0):
    """Horizontal bar from the baseline x0 to the data end x1, rounded only at the data end."""
    r = min(r, abs(x1 - x0) / 2, h / 2)
    if x1 >= x0:
        return (f"M{x0:.1f},{y:.1f} H{x1 - r:.1f} Q{x1:.1f},{y:.1f} {x1:.1f},{y + r:.1f} "
                f"V{y + h - r:.1f} Q{x1:.1f},{y + h:.1f} {x1 - r:.1f},{y + h:.1f} H{x0:.1f} Z")
    return (f"M{x0:.1f},{y:.1f} H{x1 + r:.1f} Q{x1:.1f},{y:.1f} {x1:.1f},{y + r:.1f} "
            f"V{y + h - r:.1f} Q{x1:.1f},{y + h:.1f} {x1 + r:.1f},{y + h:.1f} H{x0:.1f} Z")


def render(rows, backends, theme):
    t = THEMES[theme]
    bar_h, gap, band_gap = 10, 2, 14
    band = len(backends) * bar_h + (len(backends) - 1) * gap
    left, right, top = 250, 32, 104
    plot_w = 520
    width = left + plot_w + right
    height = top + len(rows) * (band + band_gap) + 54
    lo, hi = 0.1, 50.0
    ticks = [0.1, 0.25, 0.5, 1, 2, 5, 10, 25, 50]

    def x(v):
        v = min(max(v, lo), hi)
        return left + (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * plot_w

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
           f'viewBox="0 0 {width} {height}" font-family="{FONT}">',
           f'<title>Speedup of the AIR kernels over NumPy</title>',
           f'<rect width="{width}" height="{height}" rx="8" fill="{t["surface"]}"/>',
           f'<text x="24" y="32" font-size="16" font-weight="600" fill="{t["primary"]}">'
           f'Speedup over NumPy, same data types</text>',
           f'<text x="24" y="52" font-size="12" fill="{t["secondary"]}">'
           f'Ryzen 7 8845HS (8 cores, 16 threads). Right of 1x: faster than NumPy; '
           f'log scale.</text>']
    # Legend on its own row under the subtitle: swatch beside text, text in ink tokens.
    lx = 24
    for be in backends:
        label = NAMES.get(be, be)
        out.append(f'<rect x="{lx}" y="67" width="10" height="10" rx="2" '
                   f'fill="{t["series"][be]}"/>')
        out.append(f'<text x="{lx + 16}" y="76" font-size="12" fill="{t["secondary"]}">{label}</text>')
        lx += 16 + int(6.8 * len(label)) + 24
    # Gridlines and ticks.
    plot_top, plot_bottom = top - 8, height - 46
    for v in ticks:
        gx = x(v)
        color = t["baseline"] if v == 1 else t["grid"]
        out.append(f'<line x1="{gx:.1f}" y1="{plot_top}" x2="{gx:.1f}" y2="{plot_bottom}" '
                   f'stroke="{color}" stroke-width="1"/>')
        label = f"{v:g}x"
        out.append(f'<text x="{gx:.1f}" y="{plot_bottom + 18}" font-size="11" '
                   f'text-anchor="middle" fill="{t["muted"]}">{label}</text>')
    out.append(f'<text x="{x(1) - 8:.1f}" y="{plot_bottom + 36}" font-size="11" '
               f'text-anchor="end" fill="{t["muted"]}">← slower than NumPy</text>')
    out.append(f'<text x="{x(1) + 8:.1f}" y="{plot_bottom + 36}" font-size="11" '
               f'fill="{t["muted"]}">faster than NumPy →</text>')
    # Bars.
    for i, row in enumerate(rows):
        y0 = top + i * (band + band_gap)
        out.append(f'<text x="{left - 12}" y="{y0 + band / 2 + 4:.1f}" font-size="12" '
                   f'text-anchor="end" fill="{t["primary"]}">{row["label"]}</text>')
        for j, be in enumerate(backends):
            s = row["speedup"].get(be)
            if s is None:
                continue
            y = y0 + j * (bar_h + gap)
            out.append(f'<path d="{bar_path(x(1), x(s), y, bar_h)}" fill="{t["series"][be]}">'
                       f'<title>{row["label"]}, {NAMES.get(be, be)}: {s:.2f}x NumPy</title></path>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main():
    results, prefix = sys.argv[1], sys.argv[2]
    data = json.load(open(results, encoding="utf-8"))
    backends = [b for b in ("cranelift", "llvm") if b in data["backends"]]
    rows = []
    for r in data["results"]:
        speedup = {run["backend"]: r["numpy_us"] / run["us"] for run in r["runs"]
                   if run["isa"] == "host"}
        rows.append(dict(label=SHORT.get(r["kernel"], r["label"]), speedup=speedup))
    rows.sort(key=lambda r: -r["speedup"].get("llvm", r["speedup"].get(backends[0], 0)))
    for theme in THEMES:
        with open(f"{prefix}-{theme}.svg", "w", encoding="utf-8") as f:
            f.write(render(rows, backends, theme))


if __name__ == "__main__":
    main()
