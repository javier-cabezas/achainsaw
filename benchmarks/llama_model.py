"""
Llama-style models for examples/kernels/llama_decode.air: Q8_0, Q4_0 and Q6_K weight matrices
in the kernel's packed layout, a random model or one loaded from a GGUF file (benchmarks/gguf.py),
the kernel's model table, and a NumPy forward pass with the kernel's data types.

GGUF tensors are converted as follows:
  Q4_0             -> Q4_0, bit for bit
  Q8_0             -> Q8_0, bit for bit
  Q6_K             -> 6-bit values bit for bit, with one f16 scale per 16 values: the f16
                      rounding of Q6_K's super-block scale times its int8 sub-block scale
  anything else    -> dequantized to f32, then Q8_0 as llama.cpp quantizes it (e.g. Q4_1,
                      F16, BF16)
A fused matrix (QKV, gate/up) keeps its parts' format if they share one; otherwise all
parts become Q8_0 (exactly for Q4_0 parts: int8 q - 8 with the same scale). Q and K rows are un-permuted from
llama.cpp's interleaved RoPE order back to Hugging Face's half-split order, which the kernel
uses.
"""

import numpy as np

from gguf import GGUF


def quantize(x):
    """Q8_0 quantization as the kernel's quantize_q8 and llama.cpp's quantize_row_q8_0 do:
    per 32 values, d = max|x| / 127 and q = round(x / d), halves away from zero, in f32.
    Returns (q as float32, d)."""
    blocks = np.asarray(x, dtype=np.float32).reshape(-1, 32)
    d = np.abs(blocks).max(axis=1) / np.float32(127.0)
    with np.errstate(divide="ignore"):
        inv = np.where(d > 0, np.float32(1.0) / d, np.float32(0.0)).astype(np.float32)
    v = blocks * inv[:, None]
    t = np.trunc(v)
    # Halves away from zero, exactly (C's roundf): every step is exact in f32.
    return (t + np.where(np.abs(v - t) >= np.float32(0.5), np.sign(v), np.float32(0.0))).reshape(-1), d


def quantize_rows(w, chunk=4096):
    """f32 [rows x k] -> Q8_0 (q int8 [rows x k/32 x 32], d [rows x k/32]) as llama.cpp's
    quantize_row_q8_0: q from the f32 scale, the scale then stored as f16."""
    rows, k = w.shape
    q = np.empty((rows, k // 32, 32), dtype=np.int8)
    d = np.empty((rows, k // 32), dtype=np.float32)
    for r in range(0, rows, chunk):
        qv, dv = quantize(w[r:r + chunk])
        n = min(chunk, rows - r)
        q[r:r + n] = qv.reshape(n, k // 32, 32).astype(np.int8)
        d[r:r + n] = dv.reshape(n, k // 32).astype(np.float16)
    return q, d


class Mat:
    """A weight matrix of `rows` x `k`, in blocks of 32 along k, with f16 scales `d` (held as
    f32) per `sub` values of a row: [rows x k/sub].
      Q8_0: q int8 [rows x k/32 x 32], sub = 32
      Q4_0: q uint8 [rows x k/32 x 16], llama.cpp's nibble bytes: value t of a block is
            (byte t & 15) - 8 and value t + 16 is (byte t >> 4) - 8; sub = 32
      Q6_K: q int8 [rows x k/32 x 32] in -32..31, sub = 16"""

    FORMATS = {"q8": 0, "q4": 1, "q6": 2}

    def __init__(self, fmt, q, d):
        self.fmt, self.q, self.d = fmt, q, d
        self.rows, self.k = q.shape[0], q.shape[1] * 32
        self.sub = 16 if fmt == "q6" else 32

    @classmethod
    def random(cls, rng, rows, k, fmt):
        nb = k // 32
        if fmt == "q4":
            q = rng.integers(0, 256, size=(rows, nb, 16), dtype=np.uint8)
            scale = 4.6
        elif fmt == "q6":
            q = rng.integers(-32, 32, size=(rows, nb, 32), dtype=np.int8)
            scale = 18.5
        else:
            q = rng.integers(-127, 128, size=(rows, nb, 32), dtype=np.int8)
            scale = 73.0
        # Entries of roughly unit variance times 1/sqrt(k), as in a trained layer.
        n_scales = k // (16 if fmt == "q6" else 32)
        d = rng.uniform(0.75, 1.0, size=(rows, n_scales)) / (scale * np.sqrt(k))
        return cls(fmt, q, d.astype(np.float16).astype(np.float32))

    @classmethod
    def from_tensor(cls, t):
        if t.type_name in ("Q4_0", "Q8_0"):
            b = t.blocks()
            d = np.ascontiguousarray(b[:, :, 0:2]).view(np.float16)[:, :, 0].astype(np.float32)
            if t.type_name == "Q4_0":
                return cls("q4", np.ascontiguousarray(b[:, :, 2:18]), d)
            return cls("q8", np.ascontiguousarray(b[:, :, 2:34]).view(np.int8), d)
        if t.type_name == "Q6_K":
            return cls("q6", *_q6_k_values(t.blocks()))
        q = np.empty((t.rows, t.cols // 32, 32), dtype=np.int8)
        d = np.empty((t.rows, t.cols // 32), dtype=np.float32)
        step = 8192
        for r in range(0, t.rows, step):
            q[r:r + step], d[r:r + step] = quantize_rows(t.dequantize(slice(r, r + step)))
        return cls("q8", q, d)

    def to_q8(self):
        if self.fmt == "q8":
            return self
        if self.fmt == "q6":
            w = self.q.reshape(self.rows, -1, 16) * self.d[:, :, None]
            return Mat("q8", *quantize_rows(w.reshape(self.rows, self.k).astype(np.float32)))
        q = np.concatenate([self.q & 15, self.q >> 4], axis=2).astype(np.int8) - np.int8(8)
        return Mat("q8", q, self.d)

    def take(self, rows):
        return Mat(self.fmt, self.q[rows], self.d[rows])

    @staticmethod
    def concat(mats):
        if len({m.fmt for m in mats}) > 1:
            mats = [m.to_q8() for m in mats]
        return Mat(mats[0].fmt, np.concatenate([m.q for m in mats]),
                   np.concatenate([m.d for m in mats]))

    def values(self):
        """int8 [rows x k/32 x 32]: the integer weights (q - 8 for Q4_0)."""
        if self.fmt == "q4":
            return self.to_q8().q
        return self.q

    def nbytes(self):
        return self.q.nbytes + self.d.nbytes

    def pack(self):
        """The kernel's layout (see qmat_chunk in the AIR standard library): 64-row chunks;
        per chunk and 32-block, groups of 4 consecutive bytes of a row form one 32-bit lane,
        byte (g * 64 + j) * 4 + q holding row j's byte 4g + q: its int8 values plus 128 for Q8_0, its
        nibble bytes for Q4_0 (values 4g + q and 16 + 4g + q); for Q6_K the low 4 bits as
        Q4_0's nibbles, then a 512-byte plane whose byte (g * 64 + j) * 4 + q holds the high
        2 bits of values 8p + 4g + q in bit pair p. Then the f16 scales [k/sub x 64]."""
        c, nb = self.rows // 64, self.k // 32

        def lanes(v, n):
            # [c, 64, nb, n] row bytes -> [c, nb, n/4 groups, 64 rows, 4]
            return v.reshape(c, 64, nb, n // 4, 4).transpose(0, 2, 3, 1, 4).reshape(c, nb, -1)

        if self.fmt == "q4":
            q = lanes(self.q, 16)
        elif self.fmt == "q6":
            u = (self.q.astype(np.int16) + 32).astype(np.uint8).reshape(c, 64, nb, 32)
            lo, hi = u & 15, u >> 4
            nib = lo[..., :16] | (lo[..., 16:] << 4)
            top = hi[..., 0:8] | (hi[..., 8:16] << 2) | (hi[..., 16:24] << 4) | \
                (hi[..., 24:32] << 6)
            q = np.concatenate([lanes(nib, 16), lanes(top, 8)], axis=2)
        else:
            # Q8_0 stored unsigned: each int8 value + 128 (its top bit flipped).
            q = lanes(self.q.view(np.uint8) ^ np.uint8(0x80), 32)
        self.packed_q = np.ascontiguousarray(q)
        per = 32 // self.sub
        self.packed_s = np.ascontiguousarray(
            self.d.reshape(c, 64, nb * per).transpose(0, 2, 1).astype(np.float16))

    def prepare_numpy(self):
        # Integer weights by scale group for np.matmul: [k/sub x rows x sub].
        v = self.values().reshape(self.rows, -1, self.sub)
        self.q_blocks = np.ascontiguousarray(v.transpose(1, 0, 2))
        self.s_blocks = np.ascontiguousarray(self.d.T)

    def matvec(self, x):
        """y = W x with the kernel's types: x quantized to int8 per 32-block, exact integer
        dots per scale group (a batched f32 matmul is exact: every dot is below 2^24), f32
        sums."""
        xq, dx = quantize(x)
        ng = self.q_blocks.shape[0]
        dots = np.matmul(self.q_blocks.astype(np.float32), xq.reshape(ng, self.sub, 1))[:, :, 0]
        dxg = np.repeat(dx, 32 // self.sub)
        return ((dots * dxg[:, None]) * self.s_blocks).sum(axis=0)


def _q6_k_values(b):
    """Q6_K blocks [rows x k/256 x 210] -> (int8 values [rows x k/32 x 32] in -32..31,
    f32 scales [rows x k/16]: the f16 rounding of d * sc)."""
    rows, nsb = b.shape[0], b.shape[1]
    b = b.reshape(-1, 210)
    ql, qh = b[:, 0:128], b[:, 128:192]
    sc = np.ascontiguousarray(b[:, 192:208]).view(np.int8).astype(np.float32)
    d = np.ascontiguousarray(b[:, 208:210]).view(np.float16).astype(np.float32).reshape(-1)
    q = np.empty((b.shape[0], 256), dtype=np.int8)
    for h in range(2):
        l, hh = ql[:, 64 * h:64 * h + 64], qh[:, 32 * h:32 * h + 32]
        parts = [(l[:, :32] & 15) | ((hh & 3) << 4), (l[:, 32:] & 15) | (((hh >> 2) & 3) << 4),
                 (l[:, :32] >> 4) | (((hh >> 4) & 3) << 4), (l[:, 32:] >> 4) | (((hh >> 6) & 3) << 4)]
        for p, v in enumerate(parts):
            q[:, 128 * h + 32 * p:128 * h + 32 * p + 32] = v.astype(np.int16) - 32
    scales = (d[:, None] * sc).astype(np.float16).astype(np.float32)   # [n, 16], f32 product exact
    return q.reshape(rows, nsb * 8, 32), scales.reshape(rows, nsb * 16)


def rope_tables(max_ctx, head_dim, base, freq_factors=None):
    """cos and sin [max_ctx x head_dim/2] as llama.cpp computes them: theta_i = pos *
    base^(-2i/head_dim), by repeated f32 multiplication, divided by the frequency factor."""
    half = head_dim // 2
    ff = np.ones(half, np.float32) if freq_factors is None else freq_factors.astype(np.float32)
    scale = np.float32(base) ** np.float32(-2.0 / head_dim)
    theta = np.arange(max_ctx, dtype=np.float32)
    ang = np.empty((max_ctx, half), dtype=np.float32)
    for i in range(half):
        ang[:, i] = theta / ff[i]
        theta = theta * scale
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


class Model:
    def __init__(self, cfg, embd, norm_out, lm, cos, sin, layers, eps, numpy=False):
        d, nl, nh, nkv, hd, f, vocab, max_ctx = cfg
        self.cfg = np.array(cfg, dtype=np.int64)
        self.d, self.nh, self.nkv, self.hd, self.f = d, nh, nkv, hd, f
        self.vocab, self.max_ctx, self.eps = vocab, max_ctx, np.float32(eps)
        self.embd, self.norm_out, self.lm, self.cos, self.sin = embd, norm_out, lm, cos, sin
        self.layers = layers
        mats = [lm] + [l[n] for l in layers for n in ("qkv", "o", "gate_up", "down")]
        for m in mats:
            m.pack()
            if numpy:
                m.prepare_numpy()
            m.q = None  # the packed copy is all the kernel needs
        words = []

        def mat(m):
            words.extend([m.packed_q.ctypes.data, m.packed_s.ctypes.data, Mat.FORMATS[m.fmt]])

        words += [embd.ctypes.data, norm_out.ctypes.data]
        mat(lm)
        words += [cos.ctypes.data, sin.ctypes.data]
        for l in layers:
            words.append(l["attn_norm"].ctypes.data)
            mat(l["qkv"])
            mat(l["o"])
            words.append(l["ffn_norm"].ctypes.data)
            mat(l["gate_up"])
            mat(l["down"])
        self.table = np.array(words, dtype=np.uint64)

    @classmethod
    def random(cls, d, layers, heads, kv_heads, head_dim, ffn, vocab, max_ctx, fmt="q8",
               numpy=False):
        rng = np.random.default_rng(0)
        ls = []
        for _ in range(layers):
            ls.append(dict(
                attn_norm=rng.uniform(0.8, 1.2, d).astype(np.float32),
                qkv=Mat.random(rng, (heads + 2 * kv_heads) * head_dim, d, fmt),
                o=Mat.random(rng, d, heads * head_dim, fmt),
                ffn_norm=rng.uniform(0.8, 1.2, d).astype(np.float32),
                gate_up=Mat.random(rng, 2 * ffn, d, fmt),
                down=Mat.random(rng, d, ffn, fmt),
            ))
        cos, sin = rope_tables(max_ctx, head_dim, 500000.0)
        return cls((d, layers, heads, kv_heads, head_dim, ffn, vocab, max_ctx),
                   rng.standard_normal((vocab, d), dtype=np.float32),
                   rng.uniform(0.8, 1.2, d).astype(np.float32),
                   Mat.random(rng, vocab, d, fmt), cos, sin, ls, 1e-5, numpy)

    @classmethod
    def from_gguf(cls, path, max_ctx, numpy=False):
        g = path if isinstance(path, GGUF) else GGUF(path)
        m, t = g.meta, g.tensors
        arch = m.get("general.architecture")
        if arch != "llama":
            raise ValueError(f"unsupported architecture {arch!r} (llama only)")
        d = m["llama.embedding_length"]
        nl, nh = m["llama.block_count"], m["llama.attention.head_count"]
        nkv = m.get("llama.attention.head_count_kv", nh)
        hd = m.get("llama.attention.key_length", d // nh)
        f = m["llama.feed_forward_length"]
        vocab = t["token_embd.weight"].rows
        rope_dims = m.get("llama.rope.dimension_count", hd)
        if rope_dims != hd or m.get("llama.attention.value_length", hd) != hd:
            raise ValueError("partial RoPE and differing K/V head sizes are not supported")
        ff = t["rope_freqs.weight"].dequantize()[0] if "rope_freqs.weight" in t else None
        cos, sin = rope_tables(max_ctx, hd, m.get("llama.rope.freq_base", 10000.0), ff)

        embd = np.empty((vocab, d), dtype=np.float32)
        for r in range(0, vocab, 8192):
            embd[r:r + 8192] = t["token_embd.weight"].dequantize(slice(r, r + 8192))
        lm = Mat.from_tensor(t.get("output.weight", t["token_embd.weight"]))

        def hf_rows(n_heads):
            # llama.cpp stores each head's rows as (i, half) pairs interleaved; Hugging Face
            # (and the kernel's RoPE) puts the first halves first.
            return np.arange(n_heads * hd).reshape(n_heads, hd // 2, 2).transpose(0, 2, 1).reshape(-1)

        def norm(name):
            return np.ascontiguousarray(t[name].dequantize()[0])

        layers = []
        for i in range(nl):
            p = f"blk.{i}."
            q = Mat.from_tensor(t[p + "attn_q.weight"]).take(hf_rows(nh))
            k = Mat.from_tensor(t[p + "attn_k.weight"]).take(hf_rows(nkv))
            v = Mat.from_tensor(t[p + "attn_v.weight"])
            gate = Mat.from_tensor(t[p + "ffn_gate.weight"])
            up = Mat.from_tensor(t[p + "ffn_up.weight"])
            # Gate/up chunks interleave: gate rows 64c.., then up rows 64c...
            gu_rows = np.arange(2 * f).reshape(2, f // 64, 64).transpose(1, 0, 2).reshape(-1)
            layers.append(dict(
                attn_norm=norm(p + "attn_norm.weight"),
                qkv=Mat.concat([q, k, v]),
                o=Mat.from_tensor(t[p + "attn_output.weight"]),
                ffn_norm=norm(p + "ffn_norm.weight"),
                gate_up=Mat.concat([gate, up]).take(gu_rows),
                down=Mat.from_tensor(t[p + "ffn_down.weight"]),
            ))
        eps = m.get("llama.attention.layer_norm_rms_epsilon", 1e-5)
        model = cls((d, nl, nh, nkv, hd, f, vocab, max_ctx), embd, norm(
            "output_norm.weight"), lm, cos, sin, layers, eps, numpy)
        model.formats = {n: sorted({l[n].fmt for l in layers}) for n in ("qkv", "o", "gate_up",
                                                                           "down")}
        model.formats["lm"] = [lm.fmt]
        return model

    def weight_bytes(self):
        """Bytes of packed matrices one token reads (embedding row and norms aside)."""
        mats = [self.lm] + [l[n] for l in self.layers for n in ("qkv", "o", "gate_up", "down")]
        return sum(m.packed_q.nbytes + m.packed_s.nbytes for m in mats)

    def new_cache(self):
        return np.zeros(len(self.layers) * 2 * self.max_ctx * self.nkv * self.hd, dtype=np.float32)

    # NumPy forward pass of the same model (needs numpy=True at construction).
    def norm(self, h, w):
        return h / np.sqrt(np.mean(h * h) + self.eps) * w

    def rope(self, x, pos):
        half = self.hd // 2
        a, b = x[..., :half], x[..., half:]
        c, s = self.cos[pos], self.sin[pos]
        return np.concatenate([a * c - b * s, b * c + a * s], axis=-1)

    def numpy_cache(self):
        return [(np.zeros((self.max_ctx, self.nkv, self.hd), np.float32),
                 np.zeros((self.max_ctx, self.nkv, self.hd), np.float32)) for _ in self.layers]

    def numpy_step(self, cache, token, pos):
        nh, nkv, hd = self.nh, self.nkv, self.hd
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
        self.last_h, self.last_logits = h, logits  # what the kernel leaves in `h`, for tests
        return int(np.argmax(logits))
