"""
Llama-style models for examples/kernels/llama_decode.air: Q8_0, Q4_0, Q6_K, Q4_K and Q5_K weight
matrices in the kernel's packed layout, a random model or one loaded from a GGUF file (benchmarks/gguf.py),
the kernel's model table, a NumPy forward pass with the kernel's data types, and a PyTorch
forward pass on PyTorch's weight-only quantized kernels (see Mat.prepare_torch).

GGUF tensors are converted as follows:
  Q4_0             -> Q4_0, bit for bit
  Q8_0             -> Q8_0, bit for bit
  Q6_K             -> 6-bit values bit for bit, with one f16 scale per 16 values: the f16
                      rounding of Q6_K's super-block scale times its int8 sub-block scale
  Q4_K, Q5_K       -> 4- or 5-bit values bit for bit, with an f16 scale and min per 32
                      values: the f16 roundings of the super-block scale times the 6-bit
                      sub-block scale, and of the super-block min times the 6-bit min
  anything else    -> dequantized to f32, then Q8_0 as llama.cpp quantizes it (e.g. Q4_1,
                      F16, BF16)
A fused QKV matrix keeps each part's format (Parts: Q4_K_M files store V as Q6_K in some
layers and Q, K as Q4_K). Gate/up rows interleave by chunk, so they keep their format if
gate and up share one; otherwise both become Q8_0 (exactly for Q4_0: int8 q - 8 with the
same scale). Q and K rows are un-permuted from
llama.cpp's interleaved RoPE order back to Hugging Face's half-split order, which the kernel
uses.
"""

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None

from gguf import GGUF, k_quant_values


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


def f16_round(x):
    return np.asarray(x, dtype=np.float32).astype(np.float16).astype(np.float32)


class Mat:
    """A weight matrix of `rows` x `k`, in blocks of 32 along k, with f16 scales `d` (held as
    f32) per `sub` values of a row: [rows x k/sub].
      Q8_0: q int8 [rows x k/32 x 32], sub = 32
      Q4_0: q uint8 [rows x k/32 x 16], llama.cpp's nibble bytes: value t of a block is
            (byte t & 15) - 8 and value t + 16 is (byte t >> 4) - 8; sub = 32
      Q6_K: q int8 [rows x k/32 x 32] in -32..31, sub = 16
      Q4_K, Q5_K: q uint8 [rows x k/32 x 32] in 0..15 or 0..31, sub = 32, and f16 mins `m`
            [rows x k/32]: weight d * q - m"""

    FORMATS = {"q8": 0, "q4": 1, "q6": 2, "q4k": 3, "q5k": 4}

    def __init__(self, fmt, q, d, m=None):
        self.fmt, self.q, self.d, self.m = fmt, q, d, m
        self.rows, self.k = q.shape[0], q.shape[1] * 32
        self.sub = 16 if fmt == "q6" else 32

    @property
    def has_min(self):
        return self.fmt in ("q4k", "q5k")

    @classmethod
    def random(cls, rng, rows, k, fmt):
        nb = k // 32
        m = None
        if fmt == "q4":
            q = rng.integers(0, 256, size=(rows, nb, 16), dtype=np.uint8)
            scale = 4.6
        elif fmt == "q6":
            q = rng.integers(-32, 32, size=(rows, nb, 32), dtype=np.int8)
            scale = 18.5
        elif fmt in ("q4k", "q5k"):
            top = 16 if fmt == "q4k" else 32
            q = rng.integers(0, top, size=(rows, nb, 32), dtype=np.uint8)
            scale = 4.6 if fmt == "q4k" else 9.2
        else:
            q = rng.integers(-127, 128, size=(rows, nb, 32), dtype=np.int8)
            scale = 73.0
        # Entries of roughly unit variance times 1/sqrt(k), as in a trained layer.
        n_scales = k // (16 if fmt == "q6" else 32)
        d = f16_round(rng.uniform(0.75, 1.0, size=(rows, n_scales)) / (scale * np.sqrt(k)))
        if fmt in ("q4k", "q5k"):
            # Mins near the middle of the values' range, so the weights center on zero.
            m = f16_round(d * (top / 2) * rng.uniform(0.8, 1.2, size=d.shape))
        return cls(fmt, q, d, m)

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
        if t.type_name in ("Q4_K", "Q5_K"):
            b = t.blocks()
            q, sc, mn = k_quant_values(b.reshape(-1, b.shape[2]), 4 if t.type_name == "Q4_K" else 5)
            nb = t.cols // 32
            return cls("q4k" if t.type_name == "Q4_K" else "q5k", q.reshape(t.rows, nb, 32),
                       f16_round(sc).reshape(t.rows, nb), f16_round(mn).reshape(t.rows, nb))
        q = np.empty((t.rows, t.cols // 32, 32), dtype=np.int8)
        d = np.empty((t.rows, t.cols // 32), dtype=np.float32)
        step = 8192
        for r in range(0, t.rows, step):
            q[r:r + step], d[r:r + step] = quantize_rows(t.dequantize(slice(r, r + step)))
        return cls("q8", q, d)

    def dequantize(self):
        """f32 [rows x k]."""
        if self.fmt == "q4":
            return self.to_q8().dequantize()
        w = self.q.reshape(self.rows, -1, self.sub) * self.d[:, :, None]
        if self.has_min:
            w = w - self.m[:, :, None]
        return w.reshape(self.rows, self.k).astype(np.float32)

    def to_q8(self):
        if self.fmt == "q8":
            return self
        if self.fmt == "q4":
            q = np.concatenate([self.q & 15, self.q >> 4], axis=2).astype(np.int8) - np.int8(8)
            return Mat("q8", q, self.d)
        return Mat("q8", *quantize_rows(self.dequantize()))

    def take(self, rows):
        return Mat(self.fmt, self.q[rows], self.d[rows], None if self.m is None else self.m[rows])

    @staticmethod
    def concat(mats):
        """The rows of `mats` in one matrix: in their format if they share one, else Q8_0."""
        if len({m.fmt for m in mats}) > 1:
            mats = [m.to_q8() for m in mats]
        m = None if mats[0].m is None else np.concatenate([x.m for x in mats])
        return Mat(mats[0].fmt, np.concatenate([x.q for x in mats]),
                   np.concatenate([x.d for x in mats]), m)

    def values(self):
        """int8 [rows x k/32 x 32]: the integer weights (q - 8 for Q4_0; unsigned for Q4_K
        and Q5_K, whose mins matvec applies)."""
        if self.fmt == "q4":
            return self.to_q8().q
        return self.q

    def nbytes(self):
        return self.q.nbytes + self.d.nbytes + (0 if self.m is None else self.m.nbytes)

    def parts(self):
        return [self]

    def packed_nbytes(self):
        return self.packed_q.nbytes + self.packed_s.nbytes

    def pack(self):
        """The kernel's layout (see qmat_chunk in the AIR standard library): 64-row chunks;
        per chunk and 32-block, groups of 4 consecutive bytes of a row form one 32-bit lane,
        byte (g * 64 + j) * 4 + q holding row j's byte 4g + q: its int8 values plus 128 for Q8_0, its
        nibble bytes for Q4_0 (values 4g + q and 16 + 4g + q) and Q4_K (the same, from its
        values); for Q6_K the low 4 bits as Q4_0's nibbles, then a 512-byte plane whose byte
        (g * 64 + j) * 4 + q holds the high 2 bits of values 8p + 4g + q in bit pair p; for
        Q5_K the low 4 bits as nibbles, then a 256-byte plane whose byte j * 4 + q holds the
        top bits of values 4g + q (bit g) and 16 + 4g + q (bit 4 + g). Then the f16 scales
        [k/sub x 64]; for Q4_K and Q5_K, scale then min per block."""
        c, nb = self.rows // 64, self.k // 32

        def lanes(v, n):
            # [c, 64, nb, n] row bytes -> [c, nb, n/4 groups, 64 rows, 4]
            return v.reshape(c, 64, nb, n // 4, 4).transpose(0, 2, 3, 1, 4).reshape(c, nb, -1)

        if self.fmt == "q4":
            q = lanes(self.q, 16)
        elif self.has_min:
            u = self.q.reshape(c, 64, nb, 32)
            lo = u & 15
            q = lanes(lo[..., :16] | (lo[..., 16:] << 4), 16)
            if self.fmt == "q5k":
                hi = (u >> 4).reshape(c, 64, nb, 2, 4, 4)       # [.., half, g, q]
                top = np.zeros((c, 64, nb, 4), dtype=np.uint8)
                for g in range(4):
                    top |= (hi[..., 0, g, :] << g) | (hi[..., 1, g, :] << (4 + g))
                q = np.concatenate([q, lanes(top, 4)], axis=2)
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
        sc = self.d.reshape(c, 64, nb * per)
        if self.has_min:
            sc = np.stack([self.d, self.m], axis=2).reshape(c, 64, nb * 2)
        self.packed_s = np.ascontiguousarray(sc.transpose(0, 2, 1).astype(np.float16))

    def words(self):
        """The part's 4 words for qmat_chunk: weights, scales, format, chunks."""
        return [self.packed_q.ctypes.data, self.packed_s.ctypes.data, Mat.FORMATS[self.fmt],
                self.rows // 64]

    def prepare_numpy(self):
        # Integer weights by scale group for np.matmul: [k/sub x rows x sub].
        v = self.values().reshape(self.rows, -1, self.sub)
        self.q_blocks = np.ascontiguousarray(v.transpose(1, 0, 2))
        self.s_blocks = np.ascontiguousarray(self.d.T)
        self.m_blocks = None if self.m is None else np.ascontiguousarray(self.m.T)

    def prepare_torch(self):
        """PyTorch's weight-only kernels (torchao's CPU path), which take bf16 activations.
        Q4_0 and Q4_K become int4 in the same 32-blocks (weight (q - 8) * scale + zero, with
        zero 0 for Q4_0 and 8 scale - min for Q4_K, rounded to bf16); Q8_0, Q6_K and Q5_K are
        requantized to int8 with one scale per row, as PyTorch has no per-block int8 kernel."""
        if self.fmt in ("q4", "q4k"):
            if self.fmt == "q4":
                v = np.concatenate([self.q & 15, self.q >> 4], axis=2)
            else:
                v = self.q
            self.t_w = torch.ops.aten._convert_weight_to_int4pack_for_cpu(
                torch.from_numpy(v.reshape(self.rows, self.k).astype(np.int32)), 1)
            s = torch.from_numpy(np.ascontiguousarray(self.d.T))
            z = torch.zeros_like(s) if self.m is None else \
                8 * s - torch.from_numpy(np.ascontiguousarray(self.m.T))
            self.t_sz = torch.stack([s, z], dim=-1).bfloat16().contiguous()
            return
        q = np.empty((self.rows, self.k), dtype=np.int8)
        s = np.empty(self.rows, dtype=np.float32)
        for r in range(0, self.rows, 4096):
            n = min(4096, self.rows - r)
            w = self.take(slice(r, r + n)).dequantize()
            s[r:r + n] = np.abs(w).max(axis=1) / np.float32(127.0)
            inv = np.where(s[r:r + n] > 0, 1.0 / np.maximum(s[r:r + n], 1e-30), 0.0)
            q[r:r + n] = np.round(w * inv[:, None].astype(np.float32)).astype(np.int8)
        self.t_w, self.t_s = torch.from_numpy(q), torch.from_numpy(s).bfloat16()

    def torch_matvec(self, x):
        """y = W x on PyTorch's kernels (see prepare_torch): x rounded to bf16, y in f32."""
        xb = x.bfloat16()[None]
        if self.fmt in ("q4", "q4k"):
            y = torch.ops.aten._weight_int4pack_mm_for_cpu(xb, self.t_w, 32, self.t_sz)
        else:
            y = torch.ops.aten._weight_int8pack_mm(xb, self.t_w, self.t_s)
        return y[0].float()

    def matvec(self, x):
        """y = W x with the kernel's types: x quantized to int8 per 32-block, exact integer
        dots per scale group (a batched f32 matmul is exact: every dot is below 2^24), f32
        sums."""
        xq, dx = quantize(x)
        ng = self.q_blocks.shape[0]
        dots = np.matmul(self.q_blocks.astype(np.float32), xq.reshape(ng, self.sub, 1))[:, :, 0]
        dxg = np.repeat(dx, 32 // self.sub)
        y = (dots * dxg[:, None]) * self.s_blocks
        if self.m_blocks is not None:
            # Q4_K, Q5_K: minus min * the block's activation sum.
            y -= (xq.reshape(ng, 32).sum(axis=1) * dx)[:, None] * self.m_blocks
        return y.sum(axis=0)


class Parts:
    """A matrix stacked by rows from parts of whole 64-row chunks in their own formats (a
    fused QKV matrix whose Q, K and V differ): qmat_chunk walks its parts' words."""

    def __init__(self, mats):
        self._parts = mats
        self.rows, self.k = sum(m.rows for m in mats), mats[0].k
        self.fmt = "+".join(m.fmt for m in mats)

    @staticmethod
    def of(mats):
        """`mats` stacked: one Mat if they share a format, else Parts."""
        return Mat.concat(mats) if len({m.fmt for m in mats}) == 1 else Parts(mats)

    def parts(self):
        return self._parts

    def pack(self):
        for m in self._parts:
            m.pack()

    def words(self):
        return [w for m in self._parts for w in m.words()]

    def packed_nbytes(self):
        return sum(m.packed_nbytes() for m in self._parts)

    def prepare_numpy(self):
        for m in self._parts:
            m.prepare_numpy()

    def prepare_torch(self):
        for m in self._parts:
            m.prepare_torch()

    def matvec(self, x):
        return np.concatenate([m.matvec(x) for m in self._parts])

    def torch_matvec(self, x):
        return torch.cat([m.torch_matvec(x) for m in self._parts])

    @property
    def q(self):
        return None

    @q.setter
    def q(self, value):
        for m in self._parts:
            m.q = value


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


def bf16_round(x):
    """x rounded to bfloat16 (nearest even), as f32: the values the kernel's KV cache holds."""
    b = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    r = ((b + 0x7FFF + ((b >> 16) & 1)) >> 16) << 16
    return r.astype(np.uint32).view(np.float32).reshape(np.shape(x))


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
    def __init__(self, cfg, embd, norm_out, lm, cos, sin, layers, eps, numpy=False,
                 use_torch=False):
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
            if use_torch:
                m.prepare_torch()
            m.q = None  # the packed copy is all the kernel needs
        # Each matrix's part list (4 words per part), which the table points to.
        self.part_lists = {}

        def mat(m):
            self.part_lists[id(m)] = a = np.array(m.words(), dtype=np.uint64)
            return a.ctypes.data

        words = [embd.ctypes.data, norm_out.ctypes.data, mat(lm), cos.ctypes.data,
                 sin.ctypes.data]
        for l in layers:
            words += [l["attn_norm"].ctypes.data, mat(l["qkv"]), mat(l["o"]),
                      l["ffn_norm"].ctypes.data, mat(l["gate_up"]), mat(l["down"])]
        self.table = np.array(words, dtype=np.uint64)

    @classmethod
    def random(cls, d, layers, heads, kv_heads, head_dim, ffn, vocab, max_ctx, fmt="q8",
               numpy=False, use_torch=False):
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
                   Mat.random(rng, vocab, d, fmt), cos, sin, ls, 1e-5, numpy, use_torch)

    @classmethod
    def from_gguf(cls, path, max_ctx, numpy=False, use_torch=False):
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
                qkv=Parts.of([q, k, v]),
                o=Mat.from_tensor(t[p + "attn_output.weight"]),
                ffn_norm=norm(p + "ffn_norm.weight"),
                gate_up=Mat.concat([gate, up]).take(gu_rows),
                down=Mat.from_tensor(t[p + "ffn_down.weight"]),
            ))
        eps = m.get("llama.attention.layer_norm_rms_epsilon", 1e-5)
        model = cls((d, nl, nh, nkv, hd, f, vocab, max_ctx), embd, norm(
            "output_norm.weight"), lm, cos, sin, layers, eps, numpy, use_torch)
        model.formats = {n: sorted({l[n].fmt for l in layers}) for n in ("qkv", "o", "gate_up",
                                                                           "down")}
        model.formats["lm"] = [lm.fmt]
        return model

    def weight_bytes(self):
        """Bytes of packed matrices one token reads (embedding row and norms aside)."""
        mats = [self.lm] + [l[n] for l in self.layers for n in ("qkv", "o", "gate_up", "down")]
        return sum(m.packed_nbytes() for m in mats)

    def torch_weight_bytes(self):
        """Bytes of PyTorch's matrices one token reads (see Mat.prepare_torch)."""
        mats = [self.lm] + [l[n] for l in self.layers for n in ("qkv", "o", "gate_up", "down")]
        return sum(t.nbytes for m in mats for p in m.parts()
                   for t in ((p.t_w, p.t_sz) if p.fmt in ("q4", "q4k") else (p.t_w, p.t_s)))

    def new_cache(self):
        """The kernel's KV cache: per layer, K then V, each bf16 [nkv x max_ctx x hd] (raw
        u16 bits)."""
        return np.zeros(len(self.layers) * 2 * self.nkv * self.max_ctx * self.hd, dtype=np.uint16)

    # NumPy forward pass of the same model (needs numpy=True at construction).
    def norm(self, h, w):
        return h / np.sqrt(np.mean(h * h) + self.eps) * w

    def rope(self, x, pos):
        half = self.hd // 2
        a, b = x[..., :half], x[..., half:]
        c, s = self.cos[pos], self.sin[pos]
        return np.concatenate([a * c - b * s, b * c + a * s], axis=-1)

    def numpy_cache(self):
        # f32 arrays holding bf16 values, as the kernel's cache does.
        return [(np.zeros((self.max_ctx, self.nkv, self.hd), np.float32),
                 np.zeros((self.max_ctx, self.nkv, self.hd), np.float32)) for _ in self.layers]

    def numpy_step(self, cache, token, pos):
        nh, nkv, hd = self.nh, self.nkv, self.hd
        h = self.embd[token].copy()
        for l, layer in enumerate(self.layers):
            kc, vc = cache[l]
            qkv = layer["qkv"].matvec(self.norm(h, layer["attn_norm"]))
            q = self.rope(qkv[:nh * hd].reshape(nh, hd), pos)
            kc[pos] = bf16_round(self.rope(qkv[nh * hd:(nh + nkv) * hd].reshape(nkv, hd), pos))
            vc[pos] = bf16_round(qkv[(nh + nkv) * hd:].reshape(nkv, hd))
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

    # PyTorch forward pass of the same model (needs use_torch=True at construction): the
    # matrices on PyTorch's weight-only kernels (Mat.torch_matvec), everything else in f32.
    def torch_cache(self):
        # bf16 [nkv x max_ctx x hd], as the kernel's.
        return [tuple(torch.zeros(self.nkv, self.max_ctx, self.hd, dtype=torch.bfloat16)
                      for _ in range(2)) for _ in self.layers]

    def torch_rope(self, x, pos):
        half = self.hd // 2
        a, b = x[..., :half], x[..., half:]
        c, s = torch.from_numpy(self.cos[pos]), torch.from_numpy(self.sin[pos])
        return torch.cat([a * c - b * s, b * c + a * s], dim=-1)

    @torch.inference_mode() if torch is not None else (lambda f: f)
    def torch_step(self, cache, token, pos):
        nh, nkv, hd, d, eps = self.nh, self.nkv, self.hd, self.d, float(self.eps)

        def norm(h, w):
            return F.rms_norm(h, (d,), torch.from_numpy(w), eps=eps)

        h = torch.from_numpy(self.embd[token].copy())
        for l, layer in enumerate(self.layers):
            kc, vc = cache[l]
            qkv = layer["qkv"].torch_matvec(norm(h, layer["attn_norm"]))
            q = self.torch_rope(qkv[:nh * hd].view(nh, hd), pos)
            kc[:, pos] = self.torch_rope(qkv[nh * hd:(nh + nkv) * hd].view(nkv, hd), pos)
            vc[:, pos] = qkv[(nh + nkv) * hd:].view(nkv, hd)
            keys, vals = kc[None, :, :pos + 1], vc[None, :, :pos + 1]  # [1, nkv, t, hd]
            # Each KV head's query heads as its query positions: the same math as
            # enable_gqa=True with one position per head, which leaves SDPA's fast path.
            qg = q.view(1, nkv, nh // nkv, hd).bfloat16()
            attn = F.scaled_dot_product_attention(qg, keys, vals).float()
            h = h + layer["o"].torch_matvec(attn.reshape(-1))
            gu = layer["gate_up"].torch_matvec(norm(h, layer["ffn_norm"])).view(-1, 2, 64)
            g, u = gu[:, 0].reshape(-1), gu[:, 1].reshape(-1)
            h = h + layer["down"].torch_matvec(F.silu(g) * u)
        logits = self.lm.torch_matvec(norm(h, self.norm_out))
        return int(torch.max(logits, 0).indices)
