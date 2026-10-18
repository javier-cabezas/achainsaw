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

from __future__ import annotations

from typing import Any, Literal, TypeAlias, TypedDict

import numpy as np
import numpy.typing as npt

try:
    import torch as torch  # re-exported for benchmark_decode.py
    import torch.nn.functional as F
except ImportError:
    torch = None

from gguf import F32, GGUF, U8, Floats, Ints, Tensor, k_quant_values

I8: TypeAlias = npt.NDArray[np.int8]
# Row selection: a slice or row indices.
Rows: TypeAlias = slice | npt.NDArray[np.intp]


def quantize(x: npt.ArrayLike) -> tuple[F32, Floats]:
    """Q8_0 quantization as the kernel's quantize_q8 and llama.cpp's quantize_row_q8_0 do:
    per 32 values, d = max|x| / 127 and q = round(x / d), halves away from zero, in f32.
    Returns (q as float32, d)."""
    blocks: F32 = np.asarray(x, dtype=np.float32).reshape(-1, 32)
    d: Floats = np.abs(blocks).max(axis=1) / np.float32(127.0)
    with np.errstate(divide="ignore"):
        inv: F32 = np.where(d > 0, np.float32(1.0) / d, np.float32(0.0)).astype(np.float32)
    v: F32 = blocks * inv[:, None]
    t: F32 = np.trunc(v)
    # Halves away from zero, exactly (C's roundf): every step is exact in f32.
    q: F32 = t + np.where(np.abs(v - t) >= np.float32(0.5), np.sign(v), np.float32(0.0))
    return q.reshape(-1), d


def quantize_rows(w: F32, chunk: int = 4096) -> tuple[I8, F32]:
    """f32 [rows x k] -> Q8_0 (q int8 [rows x k/32 x 32], d [rows x k/32]) as llama.cpp's
    quantize_row_q8_0: q from the f32 scale, the scale then stored as f16."""
    rows, k = w.shape
    q: I8 = np.empty((rows, k // 32, 32), dtype=np.int8)
    d: F32 = np.empty((rows, k // 32), dtype=np.float32)
    for r in range(0, rows, chunk):
        qv, dv = quantize(w[r:r + chunk])
        n: int = min(chunk, rows - r)
        q[r:r + n] = qv.reshape(n, k // 32, 32).astype(np.int8)
        d[r:r + n] = dv.reshape(n, k // 32).astype(np.float16)
    return q, d


def f16_round(x: npt.ArrayLike) -> F32:
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

    FORMATS: dict[str, int] = {"q8": 0, "q4": 1, "q6": 2, "q4k": 3, "q5k": 4}

    # The kernel's layout, from pack().
    packed_q: Ints
    packed_s: npt.NDArray[np.float16]
    # For matvec, from prepare_numpy().
    q_blocks: Ints
    s_blocks: F32
    m_blocks: F32 | None
    # For torch_matvec, from prepare_torch(): int4 weights with bf16 scales and zero points
    # (t_sz), or int8 weights with bf16 scales (t_s).
    t_w: torch.Tensor
    t_sz: torch.Tensor
    t_s: torch.Tensor

    def __init__(self, fmt: str, q: Ints, d: F32, m: F32 | None = None) -> None:
        self.fmt: str = fmt
        self.q: Ints = q
        self.d: F32 = d
        self.m: F32 | None = m
        self.rows: int = q.shape[0]
        self.k: int = q.shape[1] * 32
        self.sub: int = 16 if fmt == "q6" else 32

    @property
    def has_min(self) -> bool:
        return self.fmt in ("q4k", "q5k")

    @classmethod
    def random(cls, rng: np.random.Generator, rows: int, k: int, fmt: str) -> Mat:
        nb: int = k // 32
        m: F32 | None = None
        top: int = 0
        if fmt == "q4":
            q: Ints = rng.integers(0, 256, size=(rows, nb, 16), dtype=np.uint8)
            scale: float = 4.6
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
        n_scales: int = k // (16 if fmt == "q6" else 32)
        d: F32 = f16_round(rng.uniform(0.75, 1.0, size=(rows, n_scales)) / (scale * np.sqrt(k)))
        if fmt in ("q4k", "q5k"):
            # Mins near the middle of the values' range, so the weights center on zero.
            m = f16_round(d * (top / 2) * rng.uniform(0.8, 1.2, size=d.shape))
        return cls(fmt, q, d, m)

    @classmethod
    def from_tensor(cls, t: Tensor) -> Mat:
        if t.type_name in ("Q4_0", "Q8_0"):
            b: U8 = t.blocks()
            d: F32 = np.ascontiguousarray(b[:, :, 0:2]).view(np.float16)[:, :, 0].astype(np.float32)
            if t.type_name == "Q4_0":
                return cls("q4", np.ascontiguousarray(b[:, :, 2:18]), d)
            return cls("q8", np.ascontiguousarray(b[:, :, 2:34]).view(np.int8), d)
        if t.type_name == "Q6_K":
            return cls("q6", *_q6_k_values(t.blocks()))
        if t.type_name in ("Q4_K", "Q5_K"):
            b = t.blocks()
            q, sc, mn = k_quant_values(b.reshape(-1, b.shape[2]),
                                       4 if t.type_name == "Q4_K" else 5)
            nb: int = t.cols // 32
            return cls("q4k" if t.type_name == "Q4_K" else "q5k", q.reshape(t.rows, nb, 32),
                       f16_round(sc).reshape(t.rows, nb), f16_round(mn).reshape(t.rows, nb))
        q8: I8 = np.empty((t.rows, t.cols // 32, 32), dtype=np.int8)
        d = np.empty((t.rows, t.cols // 32), dtype=np.float32)
        step: int = 8192
        for r in range(0, t.rows, step):
            q8[r:r + step], d[r:r + step] = quantize_rows(t.dequantize(slice(r, r + step)))
        return cls("q8", q8, d)

    def dequantize(self) -> F32:
        """f32 [rows x k]."""
        if self.fmt == "q4":
            return self.to_q8().dequantize()
        w: F32 = self.q.reshape(self.rows, -1, self.sub) * self.d[:, :, None]
        if self.m is not None:
            w = w - self.m[:, :, None]
        return w.reshape(self.rows, self.k).astype(np.float32)

    def to_q8(self) -> Mat:
        if self.fmt == "q8":
            return self
        if self.fmt == "q4":
            q: I8 = np.concatenate([self.q & 15, self.q >> 4], axis=2).astype(np.int8) - np.int8(8)
            return Mat("q8", q, self.d)
        return Mat("q8", *quantize_rows(self.dequantize()))

    def take(self, rows: Rows) -> Mat:
        return Mat(self.fmt, self.q[rows], self.d[rows], None if self.m is None else self.m[rows])

    @staticmethod
    def concat(mats: list[Mat]) -> Mat:
        """The rows of `mats` in one matrix: in their format if they share one, else Q8_0."""
        if len({m.fmt for m in mats}) > 1:
            mats = [m.to_q8() for m in mats]
        mins: list[F32] = [x.m for x in mats if x.m is not None]
        return Mat(mats[0].fmt, np.concatenate([x.q for x in mats]),
                   np.concatenate([x.d for x in mats]), np.concatenate(mins) if mins else None)

    def drop_values(self) -> None:
        """Frees the unpacked weights once the packed copy is all that is needed."""
        del self.q

    def values(self) -> Ints:
        """int8 [rows x k/32 x 32]: the integer weights (q - 8 for Q4_0; unsigned for Q4_K
        and Q5_K, whose mins matvec applies)."""
        if self.fmt == "q4":
            return self.to_q8().q
        return self.q

    def nbytes(self) -> int:
        return self.q.nbytes + self.d.nbytes + (0 if self.m is None else self.m.nbytes)

    def parts(self) -> list[Mat]:
        return [self]

    def packed_nbytes(self) -> int:
        return self.packed_q.nbytes + self.packed_s.nbytes

    def pack(self) -> None:
        """The kernel's layout (see qmat_chunk in the AIR standard library): 64-row chunks;
        per chunk and 32-block, groups of 4 consecutive bytes of a row form one 32-bit lane,
        byte (g * 64 + j) * 4 + q holding row j's byte 4g + q: its int8 values plus 128 for Q8_0, its
        nibble bytes for Q4_0 (values 4g + q and 16 + 4g + q) and Q4_K (the same, from its
        values); for Q6_K the low 4 bits as Q4_0's nibbles, then a 512-byte plane whose byte
        (g * 64 + j) * 4 + q holds the high 2 bits of values 8p + 4g + q in bit pair p; for
        Q5_K the low 4 bits as nibbles, then a 256-byte plane whose byte j * 4 + q holds the
        top bits of values 4g + q (bit g) and 16 + 4g + q (bit 4 + g). Then the f16 scales
        [k/sub x 64]; for Q4_K and Q5_K, scale then min per block."""
        c: int = self.rows // 64
        nb: int = self.k // 32

        def lanes(v: Ints, n: int) -> Ints:
            # [c, 64, nb, n] row bytes -> [c, nb, n/4 groups, 64 rows, 4]
            return v.reshape(c, 64, nb, n // 4, 4).transpose(0, 2, 3, 1, 4).reshape(c, nb, -1)

        if self.fmt == "q4":
            q: Ints = lanes(self.q, 16)
        elif self.has_min:
            u: Ints = self.q.reshape(c, 64, nb, 32)
            lo: Ints = u & 15
            q = lanes(lo[..., :16] | (lo[..., 16:] << 4), 16)
            if self.fmt == "q5k":
                hi: Ints = (u >> 4).reshape(c, 64, nb, 2, 4, 4)       # [.., half, g, q]
                top: Ints = np.zeros((c, 64, nb, 4), dtype=np.uint8)
                for g in range(4):
                    top |= (hi[..., 0, g, :] << g) | (hi[..., 1, g, :] << (4 + g))
                q = np.concatenate([q, lanes(top, 4)], axis=2)
        elif self.fmt == "q6":
            u = (self.q.astype(np.int16) + 32).astype(np.uint8).reshape(c, 64, nb, 32)
            lo, hi = u & 15, u >> 4
            nib: Ints = lo[..., :16] | (lo[..., 16:] << 4)
            top = hi[..., 0:8] | (hi[..., 8:16] << 2) | (hi[..., 16:24] << 4) | \
                (hi[..., 24:32] << 6)
            q = np.concatenate([lanes(nib, 16), lanes(top, 8)], axis=2)
        else:
            # Q8_0 stored unsigned: each int8 value + 128 (its top bit flipped).
            q = lanes(self.q.view(np.uint8) ^ np.uint8(0x80), 32)
        self.packed_q = np.ascontiguousarray(q)
        per: int = 32 // self.sub
        sc: F32 = self.d.reshape(c, 64, nb * per)
        if self.m is not None:
            sc = np.stack([self.d, self.m], axis=2).reshape(c, 64, nb * 2)
        self.packed_s = np.ascontiguousarray(sc.transpose(0, 2, 1).astype(np.float16))

    def words(self) -> list[int]:
        """The part's 4 words for qmat_chunk: weights, scales, format, chunks."""
        return [self.packed_q.ctypes.data, self.packed_s.ctypes.data, Mat.FORMATS[self.fmt],
                self.rows // 64]

    def prepare_numpy(self) -> None:
        # Integer weights by scale group for np.matmul: [k/sub x rows x sub].
        v: Ints = self.values().reshape(self.rows, -1, self.sub)
        self.q_blocks = np.ascontiguousarray(v.transpose(1, 0, 2))
        self.s_blocks = np.ascontiguousarray(self.d.T)
        self.m_blocks = None if self.m is None else np.ascontiguousarray(self.m.T)

    def prepare_torch(self) -> None:
        """PyTorch's weight-only kernels (torchao's CPU path), which take bf16 activations.
        Q4_0 and Q4_K become int4 in the same 32-blocks (weight (q - 8) * scale + zero, with
        zero 0 for Q4_0 and 8 scale - min for Q4_K, rounded to bf16); Q8_0, Q6_K and Q5_K are
        requantized to int8 with one scale per row, as PyTorch has no per-block int8 kernel."""
        if self.fmt in ("q4", "q4k"):
            if self.fmt == "q4":
                v: Ints = np.concatenate([self.q & 15, self.q >> 4], axis=2)
            else:
                v = self.q
            self.t_w = torch.ops.aten._convert_weight_to_int4pack_for_cpu(
                torch.from_numpy(v.reshape(self.rows, self.k).astype(np.int32)), 1)
            s: torch.Tensor = torch.from_numpy(np.ascontiguousarray(self.d.T))
            z: torch.Tensor = torch.zeros_like(s) if self.m is None else \
                8 * s - torch.from_numpy(np.ascontiguousarray(self.m.T))
            self.t_sz = torch.stack([s, z], dim=-1).bfloat16().contiguous()
            return
        q: I8 = np.empty((self.rows, self.k), dtype=np.int8)
        rs: F32 = np.empty(self.rows, dtype=np.float32)
        for r in range(0, self.rows, 4096):
            n: int = min(4096, self.rows - r)
            w: F32 = self.take(slice(r, r + n)).dequantize()
            rs[r:r + n] = np.abs(w).max(axis=1) / np.float32(127.0)
            inv: npt.NDArray[np.float64] = np.where(
                rs[r:r + n] > 0, 1.0 / np.maximum(rs[r:r + n], 1e-30), 0.0)
            q[r:r + n] = np.round(w * inv[:, None].astype(np.float32)).astype(np.int8)
        self.t_w = torch.from_numpy(q)
        self.t_s = torch.from_numpy(rs).bfloat16()

    def torch_matvec(self, x: torch.Tensor) -> torch.Tensor:
        """y = W x on PyTorch's kernels (see prepare_torch): x rounded to bf16, y in f32."""
        xb: torch.Tensor = x.bfloat16()[None]
        if self.fmt in ("q4", "q4k"):
            y: torch.Tensor = torch.ops.aten._weight_int4pack_mm_for_cpu(xb, self.t_w, 32,
                                                                         self.t_sz)
        else:
            y = torch.ops.aten._weight_int8pack_mm(xb, self.t_w, self.t_s)
        return y[0].float()

    def matvec(self, x: F32) -> F32:
        """y = W x with the kernel's types: x quantized to int8 per 32-block, exact integer
        dots per scale group (a batched f32 matmul is exact: every dot is below 2^24), f32
        sums."""
        xq, dx = quantize(x)
        ng: int = self.q_blocks.shape[0]
        dots: F32 = np.matmul(self.q_blocks.astype(np.float32),
                              xq.reshape(ng, self.sub, 1))[:, :, 0]
        dxg: F32 = np.repeat(dx, 32 // self.sub)
        y: F32 = (dots * dxg[:, None]) * self.s_blocks
        if self.m_blocks is not None:
            # Q4_K, Q5_K: minus min * the block's activation sum.
            y -= (xq.reshape(ng, 32).sum(axis=1) * dx)[:, None] * self.m_blocks
        return y.sum(axis=0)


class Parts:
    """A matrix stacked by rows from parts of whole 64-row chunks in their own formats (a
    fused QKV matrix whose Q, K and V differ): qmat_chunk walks its parts' words."""

    def __init__(self, mats: list[Mat]) -> None:
        self._parts: list[Mat] = mats
        self.rows: int = sum(m.rows for m in mats)
        self.k: int = mats[0].k
        self.fmt: str = "+".join(m.fmt for m in mats)

    @staticmethod
    def of(mats: list[Mat]) -> Mat | Parts:
        """`mats` stacked: one Mat if they share a format, else Parts."""
        return Mat.concat(mats) if len({m.fmt for m in mats}) == 1 else Parts(mats)

    def parts(self) -> list[Mat]:
        return self._parts

    def pack(self) -> None:
        for m in self._parts:
            m.pack()

    def words(self) -> list[int]:
        return [w for m in self._parts for w in m.words()]

    def packed_nbytes(self) -> int:
        return sum(m.packed_nbytes() for m in self._parts)

    def prepare_numpy(self) -> None:
        for m in self._parts:
            m.prepare_numpy()

    def prepare_torch(self) -> None:
        for m in self._parts:
            m.prepare_torch()

    def drop_values(self) -> None:
        for m in self._parts:
            m.drop_values()

    def matvec(self, x: F32) -> F32:
        return np.concatenate([m.matvec(x) for m in self._parts])

    def torch_matvec(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([m.torch_matvec(x) for m in self._parts])


class Layer(TypedDict):
    attn_norm: F32
    qkv: Mat | Parts
    o: Mat
    ffn_norm: F32
    gate_up: Mat
    down: Mat


# A layer's matrices.
MATS: tuple[Literal["qkv", "o", "gate_up", "down"], ...] = ("qkv", "o", "gate_up", "down")


def _q6_k_values(blocks: U8) -> tuple[I8, F32]:
    """Q6_K blocks [rows x k/256 x 210] -> (int8 values [rows x k/32 x 32] in -32..31,
    f32 scales [rows x k/16]: the f16 rounding of d * sc)."""
    rows: int = blocks.shape[0]
    nsb: int = blocks.shape[1]
    b: U8 = blocks.reshape(-1, 210)
    ql: U8 = b[:, 0:128]
    qh: U8 = b[:, 128:192]
    sc: F32 = np.ascontiguousarray(b[:, 192:208]).view(np.int8).astype(np.float32)
    d: F32 = np.ascontiguousarray(b[:, 208:210]).view(np.float16).astype(np.float32).reshape(-1)
    q: I8 = np.empty((b.shape[0], 256), dtype=np.int8)
    for h in range(2):
        l: U8 = ql[:, 64 * h:64 * h + 64]
        hh: U8 = qh[:, 32 * h:32 * h + 32]
        parts: list[Ints] = [
            (l[:, :32] & 15) | ((hh & 3) << 4), (l[:, 32:] & 15) | (((hh >> 2) & 3) << 4),
            (l[:, :32] >> 4) | (((hh >> 4) & 3) << 4), (l[:, 32:] >> 4) | (((hh >> 6) & 3) << 4)]
        for p, v in enumerate(parts):
            q[:, 128 * h + 32 * p:128 * h + 32 * p + 32] = v.astype(np.int16) - 32
    # [n, 16], f32 product exact
    scales: F32 = (d[:, None] * sc).astype(np.float16).astype(np.float32)
    return q.reshape(rows, nsb * 8, 32), scales.reshape(rows, nsb * 16)


def bf16_round(x: npt.ArrayLike) -> F32:
    """x rounded to bfloat16 (nearest even), as f32: the values the kernel's KV cache holds."""
    b: npt.NDArray[np.uint64] = np.ascontiguousarray(x, dtype=np.float32).view(
        np.uint32).astype(np.uint64)
    r: Ints = ((b + 0x7FFF + ((b >> 16) & 1)) >> 16) << 16
    return r.astype(np.uint32).view(np.float32).reshape(np.shape(x))


def rope_tables(max_ctx: int, head_dim: int, base: float,
                freq_factors: F32 | None = None) -> tuple[F32, F32]:
    """cos and sin [max_ctx x head_dim/2] as llama.cpp computes them: theta_i = pos *
    base^(-2i/head_dim), by repeated f32 multiplication, divided by the frequency factor."""
    half: int = head_dim // 2
    ff: F32 = np.ones(half, np.float32) if freq_factors is None else \
        freq_factors.astype(np.float32)
    scale: np.float32 = np.float32(base) ** np.float32(-2.0 / head_dim)
    theta: F32 = np.arange(max_ctx, dtype=np.float32)
    ang: F32 = np.empty((max_ctx, half), dtype=np.float32)
    for i in range(half):
        ang[:, i] = theta / ff[i]
        theta = theta * scale
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


class Model:
    # Each matrix's formats in a GGUF model ("lm" and MATS), from from_gguf.
    formats: dict[str, list[str]]
    # The last numpy_step's hidden state and logits.
    last_h: F32
    last_logits: F32

    def __init__(self, cfg: tuple[int, int, int, int, int, int, int, int], embd: F32,
                 norm_out: F32, lm: Mat, cos: F32, sin: F32, layers: list[Layer], eps: float,
                 numpy: bool = False, use_torch: bool = False) -> None:
        d, nl, nh, nkv, hd, f, vocab, max_ctx = cfg
        self.cfg: npt.NDArray[np.int64] = np.array(cfg, dtype=np.int64)
        self.d: int = d
        self.nh: int = nh
        self.nkv: int = nkv
        self.hd: int = hd
        self.f: int = f
        self.vocab: int = vocab
        self.max_ctx: int = max_ctx
        self.eps: np.float32 = np.float32(eps)
        self.embd: F32 = embd
        self.norm_out: F32 = norm_out
        self.lm: Mat = lm
        self.cos: F32 = cos
        self.sin: F32 = sin
        self.layers: list[Layer] = layers
        for m in self.matrices():
            m.pack()
            if numpy:
                m.prepare_numpy()
            if use_torch:
                m.prepare_torch()
            m.drop_values()  # the packed copy is all the kernel needs
        # Each matrix's part list (4 words per part), which the table points to.
        self.part_lists: dict[int, npt.NDArray[np.uint64]] = {}

        def mat(m: Mat | Parts) -> int:
            a: npt.NDArray[np.uint64] = np.array(m.words(), dtype=np.uint64)
            self.part_lists[id(m)] = a
            return a.ctypes.data

        words: list[int] = [embd.ctypes.data, norm_out.ctypes.data, mat(lm), cos.ctypes.data,
                 sin.ctypes.data]
        for l in layers:
            words += [l["attn_norm"].ctypes.data, mat(l["qkv"]), mat(l["o"]),
                      l["ffn_norm"].ctypes.data, mat(l["gate_up"]), mat(l["down"])]
        self.table: npt.NDArray[np.uint64] = np.array(words, dtype=np.uint64)

    def matrices(self) -> list[Mat | Parts]:
        """The LM head and every layer's matrices."""
        return [self.lm] + [l[n] for l in self.layers for n in MATS]

    @classmethod
    def random(cls, d: int, layers: int, heads: int, kv_heads: int, head_dim: int, ffn: int,
               vocab: int, max_ctx: int, fmt: str = "q8", numpy: bool = False,
               use_torch: bool = False) -> Model:
        rng: np.random.Generator = np.random.default_rng(0)
        ls: list[Layer] = []
        for _ in range(layers):
            ls.append(Layer(
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
    def from_gguf(cls, path: str | GGUF, max_ctx: int, numpy: bool = False,
                  use_torch: bool = False) -> Model:
        g: GGUF = path if isinstance(path, GGUF) else GGUF(path)
        m: dict[str, Any] = g.meta
        t: dict[str, Tensor] = g.tensors
        arch: str | None = m.get("general.architecture")
        if arch != "llama":
            raise ValueError(f"unsupported architecture {arch!r} (llama only)")
        d: int = m["llama.embedding_length"]
        nl: int = m["llama.block_count"]
        nh: int = m["llama.attention.head_count"]
        nkv: int = m.get("llama.attention.head_count_kv", nh)
        hd: int = m.get("llama.attention.key_length", d // nh)
        f: int = m["llama.feed_forward_length"]
        vocab: int = t["token_embd.weight"].rows
        rope_dims: int = m.get("llama.rope.dimension_count", hd)
        if rope_dims != hd or m.get("llama.attention.value_length", hd) != hd:
            raise ValueError("partial RoPE and differing K/V head sizes are not supported")
        ff: F32 | None = t["rope_freqs.weight"].dequantize()[0] \
            if "rope_freqs.weight" in t else None
        cos, sin = rope_tables(max_ctx, hd, m.get("llama.rope.freq_base", 10000.0), ff)

        embd: F32 = np.empty((vocab, d), dtype=np.float32)
        for r in range(0, vocab, 8192):
            embd[r:r + 8192] = t["token_embd.weight"].dequantize(slice(r, r + 8192))
        lm: Mat = Mat.from_tensor(t.get("output.weight", t["token_embd.weight"]))

        def hf_rows(n_heads: int) -> npt.NDArray[np.intp]:
            # llama.cpp stores each head's rows as (i, half) pairs interleaved; Hugging Face
            # (and the kernel's RoPE) puts the first halves first.
            return np.arange(n_heads * hd).reshape(n_heads, hd // 2, 2).transpose(
                0, 2, 1).reshape(-1)

        def norm(name: str) -> F32:
            return np.ascontiguousarray(t[name].dequantize()[0])

        layers: list[Layer] = []
        for i in range(nl):
            p: str = f"blk.{i}."
            q: Mat = Mat.from_tensor(t[p + "attn_q.weight"]).take(hf_rows(nh))
            k: Mat = Mat.from_tensor(t[p + "attn_k.weight"]).take(hf_rows(nkv))
            v: Mat = Mat.from_tensor(t[p + "attn_v.weight"])
            gate: Mat = Mat.from_tensor(t[p + "ffn_gate.weight"])
            up: Mat = Mat.from_tensor(t[p + "ffn_up.weight"])
            # Gate/up chunks interleave: gate rows 64c.., then up rows 64c...
            gu_rows: npt.NDArray[np.intp] = np.arange(2 * f).reshape(2, f // 64, 64).transpose(
                1, 0, 2).reshape(-1)
            layers.append(Layer(
                attn_norm=norm(p + "attn_norm.weight"),
                qkv=Parts.of([q, k, v]),
                o=Mat.from_tensor(t[p + "attn_output.weight"]),
                ffn_norm=norm(p + "ffn_norm.weight"),
                gate_up=Mat.concat([gate, up]).take(gu_rows),
                down=Mat.from_tensor(t[p + "ffn_down.weight"]),
            ))
        eps: float = m.get("llama.attention.layer_norm_rms_epsilon", 1e-5)
        model: Model = cls((d, nl, nh, nkv, hd, f, vocab, max_ctx), embd, norm(
            "output_norm.weight"), lm, cos, sin, layers, eps, numpy, use_torch)
        model.formats = {n: sorted({l[n].fmt for l in layers}) for n in MATS}
        model.formats["lm"] = [lm.fmt]
        return model

    def weight_bytes(self) -> int:
        """Bytes of packed matrices one token reads (embedding row and norms aside)."""
        return sum(m.packed_nbytes() for m in self.matrices())

    def torch_weight_bytes(self) -> int:
        """Bytes of PyTorch's matrices one token reads (see Mat.prepare_torch)."""
        return sum(t.nbytes for m in self.matrices() for p in m.parts()
                   for t in ((p.t_w, p.t_sz) if p.fmt in ("q4", "q4k") else (p.t_w, p.t_s)))

    @property
    def cache_ctx(self) -> int:
        """Positions per KV head in the kernel's cache: max_ctx rounded up to K's tiles."""
        return -(-self.max_ctx // 32) * 32

    def new_cache(self) -> npt.NDArray[np.uint16]:
        """The kernel's KV cache (raw u16 bits of bf16 values): per layer, K then V, each
        nkv x cache_ctx x hd values. V is [nkv x cache_ctx x hd]; K is in tiles of 32
        positions, [nkv x cache_ctx / 32 x hd x 32]."""
        return np.zeros(len(self.layers) * 2 * self.nkv * self.cache_ctx * self.hd,
                        dtype=np.uint16)

    def cache_kv(self, cache: npt.NDArray[np.uint16]
                 ) -> tuple[npt.NDArray[np.uint16], npt.NDArray[np.uint16]]:
        """K and V of a kernel cache, each [layers x nkv x cache_ctx x hd]."""
        planes: npt.NDArray[np.uint16] = cache.reshape(len(self.layers), 2, self.nkv,
                                                       self.cache_ctx, self.hd)
        k: npt.NDArray[np.uint16] = (
            planes[:, 0].reshape(len(self.layers), self.nkv, self.cache_ctx // 32, self.hd, 32)
            .transpose(0, 1, 2, 4, 3).reshape(len(self.layers), self.nkv, self.cache_ctx,
                                              self.hd))
        return k, planes[:, 1]

    # NumPy forward pass of the same model (needs numpy=True at construction).
    def norm(self, h: Floats, w: F32) -> Floats:
        y: Floats = h / np.sqrt(np.mean(h * h) + self.eps) * w
        return y

    def rope(self, x: F32, pos: int) -> F32:
        half: int = self.hd // 2
        a: F32 = x[..., :half]
        b: F32 = x[..., half:]
        c: F32 = self.cos[pos]
        s: F32 = self.sin[pos]
        return np.concatenate([a * c - b * s, b * c + a * s], axis=-1)

    def numpy_cache(self) -> list[tuple[F32, F32]]:
        # f32 arrays holding bf16 values, as the kernel's cache does.
        return [(np.zeros((self.max_ctx, self.nkv, self.hd), np.float32),
                 np.zeros((self.max_ctx, self.nkv, self.hd), np.float32)) for _ in self.layers]

    def numpy_step(self, cache: list[tuple[F32, F32]], token: int, pos: int) -> int:
        nh: int = self.nh
        nkv: int = self.nkv
        hd: int = self.hd
        h: F32 = self.embd[token].copy()
        for l, layer in enumerate(self.layers):
            kc, vc = cache[l]
            qkv: F32 = layer["qkv"].matvec(self.norm(h, layer["attn_norm"]))
            q: F32 = self.rope(qkv[:nh * hd].reshape(nh, hd), pos)
            kc[pos] = bf16_round(self.rope(qkv[nh * hd:(nh + nkv) * hd].reshape(nkv, hd), pos))
            vc[pos] = bf16_round(qkv[(nh + nkv) * hd:].reshape(nkv, hd))
            keys: F32 = np.repeat(kc[:pos + 1], nh // nkv, axis=1)       # [t, nh, hd]
            vals: F32 = np.repeat(vc[:pos + 1], nh // nkv, axis=1)
            scores: F32 = np.einsum("hd,thd->ht", q, keys) / np.sqrt(np.float32(hd))
            p: F32 = np.exp(scores - scores.max(axis=1, keepdims=True))
            attn: F32 = np.einsum("ht,thd->hd", p, vals) / p.sum(axis=1, keepdims=True)
            h = h + layer["o"].matvec(attn.reshape(-1))
            gu: F32 = layer["gate_up"].matvec(self.norm(h, layer["ffn_norm"])).reshape(-1, 2, 64)
            g: F32 = gu[:, 0].reshape(-1)
            u: F32 = gu[:, 1].reshape(-1)
            silu: Floats = g / (1.0 + np.exp(-g))
            h = h + layer["down"].matvec(silu * u)
        logits: F32 = self.lm.matvec(self.norm(h, self.norm_out))
        # What the kernel leaves in `h`, for tests.
        self.last_h = h
        self.last_logits = logits
        return int(np.argmax(logits))

    # PyTorch forward pass of the same model (needs use_torch=True at construction): the
    # matrices on PyTorch's weight-only kernels (Mat.torch_matvec), everything else in f32.
    def torch_cache(self) -> list[tuple[torch.Tensor, ...]]:
        # bf16 [nkv x max_ctx x hd], as the kernel's.
        return [tuple(torch.zeros(self.nkv, self.max_ctx, self.hd, dtype=torch.bfloat16)
                      for _ in range(2)) for _ in self.layers]

    def torch_rope(self, x: torch.Tensor, pos: int) -> torch.Tensor:
        half: int = self.hd // 2
        a: torch.Tensor = x[..., :half]
        b: torch.Tensor = x[..., half:]
        c: torch.Tensor = torch.from_numpy(self.cos[pos])
        s: torch.Tensor = torch.from_numpy(self.sin[pos])
        return torch.cat([a * c - b * s, b * c + a * s], dim=-1)

    def torch_step(self, cache: list[tuple[torch.Tensor, ...]], token: int, pos: int) -> int:
        with torch.inference_mode():
            return self._torch_step(cache, token, pos)

    def _torch_step(self, cache: list[tuple[torch.Tensor, ...]], token: int, pos: int) -> int:
        nh: int = self.nh
        nkv: int = self.nkv
        hd: int = self.hd
        d: int = self.d
        eps: float = float(self.eps)

        def norm(h: torch.Tensor, w: F32) -> torch.Tensor:
            return F.rms_norm(h, (d,), torch.from_numpy(w), eps=eps)

        h: torch.Tensor = torch.from_numpy(self.embd[token].copy())
        for l, layer in enumerate(self.layers):
            kc, vc = cache[l]
            qkv: torch.Tensor = layer["qkv"].torch_matvec(norm(h, layer["attn_norm"]))
            q: torch.Tensor = self.torch_rope(qkv[:nh * hd].view(nh, hd), pos)
            kc[:, pos] = self.torch_rope(qkv[nh * hd:(nh + nkv) * hd].view(nkv, hd), pos)
            vc[:, pos] = qkv[(nh + nkv) * hd:].view(nkv, hd)
            keys: torch.Tensor = kc[None, :, :pos + 1]  # [1, nkv, t, hd]
            vals: torch.Tensor = vc[None, :, :pos + 1]
            # Each KV head's query heads as its query positions: the same math as
            # enable_gqa=True with one position per head, which leaves SDPA's fast path.
            qg: torch.Tensor = q.view(1, nkv, nh // nkv, hd).bfloat16()
            attn: torch.Tensor = F.scaled_dot_product_attention(qg, keys, vals).float()
            h = h + layer["o"].torch_matvec(attn.reshape(-1))
            gu: torch.Tensor = layer["gate_up"].torch_matvec(
                norm(h, layer["ffn_norm"])).view(-1, 2, 64)
            g: torch.Tensor = gu[:, 0].reshape(-1)
            u: torch.Tensor = gu[:, 1].reshape(-1)
            h = h + layer["down"].torch_matvec(F.silu(g) * u)
        logits: torch.Tensor = self.lm.torch_matvec(norm(h, self.norm_out))
        return int(torch.max(logits, 0).indices)
