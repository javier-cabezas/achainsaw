"""
Single-call LLM decode: examples/kernels/llama_decode.air on a random model with Llama 3.2 1B's
shapes, against a NumPy implementation of the same math.

The AIR kernel runs a whole decode step (embedding, every layer, final norm, LM head and
greedy argmax) in one call. Both implementations use the same data types and arithmetic:
Q8_0 int8 weights with f32 scales, activations quantized to int8 per 32-block on the fly,
exact integer block dots, f32 for everything else (embedding, norms, RoPE, KV cache,
attention). They share the same weights (NumPy reads them block-major), so they should
predict the same tokens. NumPy has no int8 matrix product, so it widens the int8 weights to
f32 in each call and runs the block dots as a batched f32 BLAS matmul (exact here).
The kernel compiles with fast_math=True (float min/max as compare and select).

    python benchmarks/benchmark_decode.py                 # all 16 layers, 32 tokens
    python benchmarks/benchmark_decode.py --layers 4 --tokens 16 --no-numpy
"""

import argparse
import os
import time

import numpy as np

import achainsaw

KERNEL = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "examples", "kernels",
                                      "llama_decode.air"))


class Q8:
    """Q8_0 matrix: q [rows x k] int8 and scales s [k/32 x rows] f32, packed for the kernel
    (64-row chunks, k-major) and, with `numpy`, also block-major for NumPy."""

    def __init__(self, rng, rows, k, numpy):
        nb = k // 32
        q = rng.integers(-127, 128, size=(rows, k), dtype=np.int8)
        self.s = (rng.uniform(0.75, 1.0, size=(nb, rows)) / (73.0 * np.sqrt(k))).astype(np.float32)
        self.nbytes = q.nbytes + self.s.nbytes
        self.packed_q = np.ascontiguousarray(q.reshape(rows // 64, 64, k).transpose(0, 2, 1))
        self.packed_s = np.ascontiguousarray(self.s.reshape(nb, rows // 64, 64).transpose(1, 0, 2))
        if numpy:
            self.q_blocks = np.ascontiguousarray(q.reshape(rows, nb, 32).transpose(1, 0, 2))

    def matvec(self, x):
        """y = W x with the kernel's types: x quantized to int8 per 32-block, exact integer
        block dots (a batched f32 matmul is exact: every dot is below 2^24), f32 sums."""
        xq, dx = quantize(x)
        nb = self.q_blocks.shape[0]
        dots = np.matmul(self.q_blocks.astype(np.float32), xq.reshape(nb, 32, 1))[:, :, 0]
        return ((dots * dx[:, None]) * self.s).sum(axis=0)


def quantize(x):
    """Q8_0 activations exactly as the kernel's quantize_q8: integer values (as f32) and the
    per-32-block scales max|x| / 127, rounded half away from zero in f32."""
    blocks = x.reshape(-1, 32).astype(np.float32)
    d = np.abs(blocks).max(axis=1) / np.float32(127.0)
    with np.errstate(divide="ignore"):
        inv = np.where(d > 0, np.float32(1.0) / d, np.float32(0.0)).astype(np.float32)
    v = blocks * inv[:, None]
    return np.trunc(v + np.where(v >= 0, np.float32(0.5), np.float32(-0.5))).reshape(-1), d


class Model:
    def __init__(self, d, layers, heads, kv_heads, head_dim, ffn, vocab, max_ctx, numpy):
        rng = np.random.default_rng(0)
        self.cfg = np.array([d, layers, heads, kv_heads, head_dim, ffn, vocab, max_ctx],
                            dtype=np.int64)
        self.d, self.nh, self.nkv, self.hd, self.f = d, heads, kv_heads, head_dim, ffn
        self.vocab, self.max_ctx, self.eps = vocab, max_ctx, np.float32(1e-5)
        self.embd = rng.standard_normal((vocab, d), dtype=np.float32)
        self.norm_out = rng.uniform(0.8, 1.2, d).astype(np.float32)
        self.lm = Q8(rng, vocab, d, numpy)
        half = head_dim // 2
        ang = np.arange(max_ctx)[:, None] * 500000.0 ** (-2.0 * np.arange(half) / head_dim)
        self.cos = np.cos(ang).astype(np.float32)
        self.sin = np.sin(ang).astype(np.float32)
        self.layers = []
        for _ in range(layers):
            self.layers.append(dict(
                attn_norm=rng.uniform(0.8, 1.2, d).astype(np.float32),
                qkv=Q8(rng, (heads + 2 * kv_heads) * head_dim, d, numpy),
                o=Q8(rng, d, heads * head_dim, numpy),
                ffn_norm=rng.uniform(0.8, 1.2, d).astype(np.float32),
                gate_up=Q8(rng, 2 * ffn, d, numpy),
                down=Q8(rng, d, ffn, numpy),
            ))
        ptrs = [self.embd, self.norm_out, self.lm.packed_q, self.lm.packed_s, self.cos, self.sin]
        for l in self.layers:
            ptrs += [l["attn_norm"], l["qkv"].packed_q, l["qkv"].packed_s, l["o"].packed_q,
                     l["o"].packed_s, l["ffn_norm"], l["gate_up"].packed_q,
                     l["gate_up"].packed_s, l["down"].packed_q, l["down"].packed_s]
        self.table = np.array([p.ctypes.data for p in ptrs], dtype=np.uint64)

    def weight_bytes(self):
        per_layer = sum(l[n].nbytes for l in self.layers for n in ("qkv", "o", "gate_up", "down"))
        return per_layer + self.lm.nbytes

    def new_cache(self):
        return np.zeros(len(self.layers) * 2 * self.max_ctx * self.nkv * self.hd, dtype=np.float32)

    # NumPy forward pass of the same model.
    def norm(self, h, w):
        return h / np.sqrt(np.mean(h * h) + self.eps) * w

    def rope(self, x, pos):
        half = self.hd // 2
        a, b = x[..., :half], x[..., half:]
        c, s = self.cos[pos], self.sin[pos]
        return np.concatenate([a * c - b * s, b * c + a * s], axis=-1)

    def numpy_step(self, cache, token, pos):
        nh, nkv, hd, f = self.nh, self.nkv, self.hd, self.f
        h = self.embd[token].copy()
        for l, layer in enumerate(self.layers):
            kc, vc = cache[l]
            qkv = layer["qkv"].matvec(self.norm(h, layer["attn_norm"]))
            q = self.rope(qkv[:nh * hd].reshape(nh, hd), pos)
            kc[pos] = self.rope(qkv[nh * hd:(nh + nkv) * hd].reshape(nkv, hd), pos)
            vc[pos] = qkv[(nh + nkv) * hd:].reshape(nkv, hd)
            keys = np.repeat(kc[:pos + 1], nh // nkv, axis=1)       # [t, nh, hd]
            vals = np.repeat(vc[:pos + 1], nh // nkv, axis=1)
            scores = np.einsum("hd,thd->ht", q, keys) / np.sqrt(np.float32(hd))
            p = np.exp(scores - scores.max(axis=1, keepdims=True))
            attn = np.einsum("ht,thd->hd", p, vals) / p.sum(axis=1, keepdims=True)
            h = h + layer["o"].matvec(attn.reshape(-1))
            gu = layer["gate_up"].matvec(self.norm(h, layer["ffn_norm"])).reshape(-1, 2, 64)
            g, u = gu[:, 0].reshape(-1), gu[:, 1].reshape(-1)
            h = h + layer["down"].matvec(g / (1.0 + np.exp(-g)) * u)
        logits = self.lm.matvec(self.norm(h, self.norm_out))
        return int(np.argmax(logits))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--tokens", type=int, default=32, help="tokens decoded per run")
    parser.add_argument("--backends", default="all")
    parser.add_argument("--no-numpy", action="store_true", help="skip the NumPy baseline")
    args = parser.parse_args()

    def build(numpy):
        return Model(d=2048, layers=args.layers, heads=32, kv_heads=8, head_dim=64, ffn=8192,
                     vocab=128256, max_ctx=args.tokens + 8, numpy=numpy)

    t0 = time.perf_counter()
    model = build(numpy=not args.no_numpy)
    wb = model.weight_bytes()
    print(f"Llama 3.2 1B shapes, {args.layers} layers, Q8_0 weights {wb / 1e9:.2f} GB "
          f"(built in {time.perf_counter() - t0:.1f} s); decoding {args.tokens} tokens")
    src = open(KERNEL, encoding="utf-8").read()
    backends = achainsaw.available_backends() if args.backends == "all" else args.backends.split(",")

    runs = {}
    for be in backends:
        kernel = achainsaw.compile(src, backend=be, fast_math=True)
        cache, h = model.new_cache(), np.zeros(model.d, dtype=np.float32)
        token, tokens, times = 1, [], []
        for pos in range(args.tokens):
            t = time.perf_counter()
            token = kernel.run("llama_decode", model.table, model.cfg, cache, token, pos, h,
                               float(model.eps))
            times.append(time.perf_counter() - t)
            tokens.append(token)
        runs[be] = tokens
        ms = np.median(times[1:] or times) * 1e3
        print(f"  AIR {be:<10} {ms:8.2f} ms/token  {1e3 / ms:7.1f} tokens/s  "
              f"{wb / (ms / 1e3) / 1e9:6.1f} GB/s of weights  (threads: {kernel.threads})")

    if not args.no_numpy:
        cache = [(np.zeros((model.max_ctx, model.nkv, model.hd), np.float32),
                  np.zeros((model.max_ctx, model.nkv, model.hd), np.float32))
                 for _ in model.layers]
        token, tokens, times = 1, [], []
        for pos in range(args.tokens):
            t = time.perf_counter()
            token = model.numpy_step(cache, token, pos)
            times.append(time.perf_counter() - t)
            tokens.append(token)
        ms = np.median(times[1:] or times) * 1e3
        print(f"  NumPy, same types {ms:8.2f} ms/token  {1e3 / ms:7.1f} tokens/s  "
              f"{wb / (ms / 1e3) / 1e9:6.1f} GB/s of weights")
        for be, toks in runs.items():
            same = next((i for i, (a, b) in enumerate(zip(toks, tokens)) if a != b), len(toks))
            print(f"  {be}: first {same} of {len(toks)} tokens match NumPy")


if __name__ == "__main__":
    main()
