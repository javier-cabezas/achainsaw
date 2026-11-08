"""
Llama-style models for examples/kernels/llama_decode.air: Q8_0, Q4_0, Q6_K, Q4_K, Q5_K, Q4_1,
IQ4_NL and MXFP4 weight matrices in the kernel's packed layout, a random model or one loaded from a GGUF file (benchmarks/gguf.py),
the kernel's model table, a NumPy forward pass with the kernel's data types, and a PyTorch
forward pass on PyTorch's weight-only quantized kernels (see Mat.prepare_torch).

GGUF tensors are converted as follows:
  Q4_0             -> Q4_0, bit for bit
  Q8_0             -> Q8_0, bit for bit
  Q6_K             -> 6-bit values bit for bit, with one f16 scale per 16 values: the f16
                      rounding of Q6_K's super-block scale times its int8 sub-block scale
  Q4_K, Q5_K       -> bit for bit: 4- or 5-bit values, a 6-bit scale and min per 32
                      values, and the f16 super-block scale and min they multiply
  Q4_1             -> bit for bit: 4-bit values and an f16 scale and min per 32 values
  IQ4_NL           -> bit for bit: 4-bit codes into the int8 codebook and an f16 scale per
                      32 values
  MXFP4            -> bit for bit: 4-bit E2M1 codes and their power-of-two scale per 32
                      values as an f16 (exact for every scale of a trained model, 2^-24 to
                      2^15, and 0 for a block of zeros; a matrix with others becomes Q8_0)
  anything else    -> dequantized to f32, then Q8_0 as llama.cpp quantizes it (e.g. F16,
                      BF16)
A fused QKV matrix keeps each part's format (Parts: Q4_K_M files store V as Q6_K in some
layers and Q, K as Q4_K). Gate/up rows interleave by chunk, so they keep their format if
gate and up share one; otherwise both become Q8_0 (exactly for Q4_0: int8 q - 8 with the
same scale). Q and K rows are un-permuted from
llama.cpp's interleaved RoPE order back to Hugging Face's half-split order, which the kernel
uses.
"""

from __future__ import annotations

import math
from typing import Any, Literal, TypeAlias, TypedDict

import numpy as np
import numpy.typing as npt

try:
    import torch as torch  # re-exported for benchmark_decode.py
    import torch.nn.functional as F
except ImportError:
    torch = None

from gguf import (F32, GGUF, IQ4_NL_VALUES, MXFP4_VALUES, U8, Floats, Ints, Tensor, k_quant_codes,
                  mxfp4_scale)

I8: TypeAlias = npt.NDArray[np.int8]
# Q4_K and Q5_K scales as GGUF stores them: 6-bit scale and min codes [rows x k/32] and the
# super-block's f16 scale and min (as f32) [rows x k/256, rounded up].
KCodes: TypeAlias = tuple[U8, U8, F32, F32]
# Row selection: a slice or row indices.
Rows: TypeAlias = slice | npt.NDArray[np.intp]


def quantize(x: npt.ArrayLike, block: int = 32) -> tuple[F32, Floats]:
    """Q8_0 quantization as the kernel's quantize_q8 and llama.cpp's quantize_row_q8_0 do:
    per `block` values, d = max|x| / 127 and q = round(x / d), halves away from zero, in f32.
    Returns (q as float32, d)."""
    blocks: F32 = np.asarray(x, dtype=np.float32).reshape(-1, block)
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
      Q4_K, Q5_K: q uint8 [rows x k/32 x 32] in 0..15 or 0..31, sub = 32, and mins `m`
            [rows x k/32]: weight d * q - m, where d and m are the f32 products of GGUF's
            6-bit codes and f16 super-block factors (`codes`), which the kernel keeps
      Q4_1: q uint8 [rows x k/32 x 32] in 0..15, sub = 32, and f16 mins `m` [rows x k/32]
            (GGUF's negated): weight d * q - m
      IQ4_NL: q uint8 [rows x k/32 x 16], GGUF's code bytes as Q4_0's (byte t: codes of
            values t and t + 16), each value the codebook entry IQ4_NL_VALUES[code]; sub = 32
      MXFP4: as IQ4_NL with the codebook MXFP4_VALUES (twice E2M1's values) and scales
            that are powers of two (GGUF's E8M0 byte e, 2^(e - 128))"""

    FORMATS: dict[str, int] = {"q8": 0, "q4": 1, "q6": 2, "q4k": 3, "q5k": 4, "q41": 5,
                               "iq4": 6, "mx4": 7}

    # The kernel's layout, from pack().
    packed_q: Ints
    packed_s: U8
    # For matvec, from prepare_numpy().
    q_blocks: Ints
    s_blocks: F32
    m_blocks: F32 | None
    # For torch_matvec, from prepare_torch(): int4 weights with bf16 scales and zero points
    # (t_sz), or int8 weights with bf16 scales (t_s).
    t_w: torch.Tensor
    t_sz: torch.Tensor
    t_s: torch.Tensor

    def __init__(self, fmt: str, q: Ints, d: F32, m: F32 | None = None,
                 codes: KCodes | None = None) -> None:
        self.fmt: str = fmt
        self.q: Ints = q
        self.d: F32 = d
        self.m: F32 | None = m
        self.codes: KCodes | None = codes
        self.rows: int = q.shape[0]
        self.k: int = q.shape[1] * 32
        self.sub: int = 16 if fmt == "q6" else 32

    @property
    def has_min(self) -> bool:
        return self.fmt in ("q4k", "q5k", "q41")

    @classmethod
    def k_quant(cls, fmt: str, q: Ints, codes: KCodes) -> Mat:
        """A Q4_K or Q5_K matrix from its values and GGUF's scale codes and factors."""
        sc, mn, ds, dm = codes
        nb: int = sc.shape[1]
        d: F32 = np.repeat(ds, 8, axis=1)[:, :nb] * sc.astype(np.float32)
        m: F32 = np.repeat(dm, 8, axis=1)[:, :nb] * mn.astype(np.float32)
        return cls(fmt, q, d, m, codes)

    @classmethod
    def random(cls, rng: np.random.Generator, rows: int, k: int, fmt: str) -> Mat:
        nb: int = k // 32
        m: F32 | None = None
        top: int = 0
        if fmt in ("q4", "iq4", "mx4"):
            q: Ints = rng.integers(0, 256, size=(rows, nb, 16), dtype=np.uint8)
            scale: float = {"q4": 4.6, "iq4": 60.0, "mx4": 5.9}[fmt]
        elif fmt == "q6":
            q = rng.integers(-32, 32, size=(rows, nb, 32), dtype=np.int8)
            scale = 18.5
        elif fmt in ("q4k", "q5k", "q41"):
            top = 32 if fmt == "q5k" else 16
            q = rng.integers(0, top, size=(rows, nb, 32), dtype=np.uint8)
            scale = 9.2 if fmt == "q5k" else 4.6
        else:
            q = rng.integers(-127, 128, size=(rows, nb, 32), dtype=np.int8)
            scale = 73.0
        # Entries of roughly unit variance times 1/sqrt(k), as in a trained layer.
        if fmt in ("q4k", "q5k"):
            # Scale codes near the top of their 6 bits, and mins near the middle of the
            # values' range, so the weights center on zero.
            ds: F32 = f16_round(rng.uniform(0.9, 1.1, size=(rows, -(-nb // 8)))
                                / (63 * scale * np.sqrt(k)))
            sc: U8 = rng.integers(48, 64, size=(rows, nb), dtype=np.uint8)
            mn: U8 = np.clip(np.round(sc * rng.uniform(0.8, 1.2, size=sc.shape)), 0,
                             63).astype(np.uint8)
            return cls.k_quant(fmt, q, (sc, mn, ds, f16_round(ds * (top / 2))))
        n_scales: int = k // (16 if fmt == "q6" else 32)
        d: F32 = f16_round(rng.uniform(0.75, 1.0, size=(rows, n_scales)) / (scale * np.sqrt(k)))
        if fmt == "mx4":
            # Powers of two, as E8M0 scales are.
            d = np.exp2(np.round(np.log2(d))).astype(np.float32)
        if fmt == "q41":
            # Mins near the middle of the values' range.
            m = f16_round(d * 8 * rng.uniform(0.9, 1.1, size=d.shape))
        return cls(fmt, q, d, m)

    @classmethod
    def from_tensor(cls, t: Tensor) -> Mat:
        if t.type_name in ("Q4_0", "Q8_0"):
            b: U8 = t.blocks()
            d: F32 = np.ascontiguousarray(b[:, :, 0:2]).view(np.float16)[:, :, 0].astype(np.float32)
            if t.type_name == "Q4_0":
                return cls("q4", np.ascontiguousarray(b[:, :, 2:18]), d)
            return cls("q8", np.ascontiguousarray(b[:, :, 2:34]).view(np.int8), d)
        if t.type_name == "IQ4_NL":
            b = t.blocks()
            dn: F32 = np.ascontiguousarray(b[:, :, 0:2]).view(np.float16)[:, :, 0].astype(
                np.float32)
            return cls("iq4", np.ascontiguousarray(b[:, :, 2:18]), dn)
        if t.type_name == "MXFP4":
            b = t.blocks()
            e: U8 = b[:, :, 0]
            qs: U8 = np.ascontiguousarray(b[:, :, 1:17])
            # A block of zeros (codes 0 and 8, which llama.cpp's quantizer writes with e = 0)
            # takes any scale: 0.
            zero: npt.NDArray[np.bool_] = np.all(((qs & 7) == 0) & ((qs >> 4 & 7) == 0), axis=2)
            if np.all(zero | ((e >= 104) & (e <= 143))):
                return cls("mx4", qs, np.where(zero, np.float32(0), mxfp4_scale(e)))
        if t.type_name == "Q4_1":
            b = t.blocks()
            f: F32 = np.ascontiguousarray(b[:, :, 0:4]).view(np.float16).astype(np.float32)
            nib: U8 = b[:, :, 4:20]
            return cls("q41", np.concatenate([nib & 15, nib >> 4], axis=2), f[:, :, 0],
                       -f[:, :, 1])
        if t.type_name == "Q6_K":
            return cls("q6", *_q6_k_values(t.blocks()))
        if t.type_name in ("Q4_K", "Q5_K"):
            b = t.blocks()
            q, ds, dm, sc, mn = k_quant_codes(b.reshape(-1, b.shape[2]),
                                              4 if t.type_name == "Q4_K" else 5)
            nb: int = t.cols // 32
            return cls.k_quant("q4k" if t.type_name == "Q4_K" else "q5k",
                               q.reshape(t.rows, nb, 32),
                               (sc.reshape(t.rows, nb), mn.reshape(t.rows, nb),
                                ds.reshape(t.rows, nb // 8), dm.reshape(t.rows, nb // 8)))
        q8: I8 = np.empty((t.rows, t.cols // 32, 32), dtype=np.int8)
        d = np.empty((t.rows, t.cols // 32), dtype=np.float32)
        step: int = 8192
        for r in range(0, t.rows, step):
            q8[r:r + step], d[r:r + step] = quantize_rows(t.dequantize(slice(r, r + step)))
        return cls("q8", q8, d)

    def dequantize(self) -> F32:
        """f32 [rows x k]."""
        if self.fmt in ("q4", "iq4", "mx4"):
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
        if self.fmt in ("iq4", "mx4"):
            # Exactly: the codebook values are int8, under the same scales.
            book: npt.NDArray[np.int8] = IQ4_NL_VALUES if self.fmt == "iq4" else MXFP4_VALUES
            return Mat("q8", book[np.concatenate([self.q & 15, self.q >> 4], axis=2)], self.d)
        return Mat("q8", *quantize_rows(self.dequantize()))

    def take(self, rows: Rows) -> Mat:
        codes: KCodes | None = None
        if self.codes is not None:
            sc, mn, ds, dm = self.codes
            codes = (sc[rows], mn[rows], ds[rows], dm[rows])
        return Mat(self.fmt, self.q[rows], self.d[rows], None if self.m is None else self.m[rows],
                   codes)

    @staticmethod
    def concat(mats: list[Mat]) -> Mat:
        """The rows of `mats` in one matrix: in their format if they share one, else Q8_0."""
        if len({m.fmt for m in mats}) > 1:
            mats = [m.to_q8() for m in mats]
        mins: list[F32] = [x.m for x in mats if x.m is not None]
        codes: list[KCodes] = [x.codes for x in mats if x.codes is not None]
        return Mat(mats[0].fmt, np.concatenate([x.q for x in mats]),
                   np.concatenate([x.d for x in mats]), np.concatenate(mins) if mins else None,
                   (np.concatenate([x[0] for x in codes]), np.concatenate([x[1] for x in codes]),
                    np.concatenate([x[2] for x in codes]), np.concatenate([x[3] for x in codes]))
                   if codes else None)

    def drop_values(self) -> None:
        """Frees the unpacked weights once the packed copy is all that is needed."""
        del self.q

    def values(self) -> Ints:
        """int8 [rows x k/32 x 32]: the integer weights (q - 8 for Q4_0, the codebook values for
        IQ4_NL and MXFP4; unsigned for Q4_K, Q5_K and Q4_1, whose mins matvec applies)."""
        if self.fmt in ("q4", "iq4", "mx4"):
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
        nibble bytes for Q4_0 (values 4g + q and 16 + 4g + q), IQ4_NL and MXFP4 (their code
        bytes, the same way) and Q4_K (the same, from its values); for Q6_K the low 4 bits as Q4_0's nibbles, then a 512-byte plane whose byte
        (g * 64 + j) * 4 + q holds the high 2 bits of values 8p + 4g + q in bit pair p; for
        Q5_K the low 4 bits as nibbles, then a 256-byte plane whose byte j * 4 + q holds the
        top bits of values 4g + q (bit g) and 16 + 4g + q (bit 4 + g); Q4_1's values as Q4_K's.
        Then the f16 scales [k/sub x 64] (for Q4_1 per block the scales, then the mins); for
        Q4_K and Q5_K, the 6-bit codes [k/32 x 64] as u16s (scale code | min
        code << 8), then the f16 super-block scales and mins [k/256 x 2 x 64]."""
        c: int = self.rows // 64
        nb: int = self.k // 32

        def lanes(v: Ints, n: int) -> Ints:
            # [c, 64, nb, n] row bytes -> [c, nb, n/4 groups, 64 rows, 4]
            return v.reshape(c, 64, nb, n // 4, 4).transpose(0, 2, 3, 1, 4).reshape(c, nb, -1)

        if self.fmt in ("q4", "iq4", "mx4"):
            # The code bytes; the kernel looks IQ4_NL's and MXFP4's codes up in its codebook.
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
        if self.codes is not None:
            scode, mcode, ds, dm = self.codes
            codes: npt.NDArray[np.uint16] = (scode.astype(np.uint16) | (
                mcode.astype(np.uint16) << np.uint16(8))).reshape(c, 64, nb)
            factors: npt.NDArray[np.float16] = np.stack([ds, dm], axis=2).reshape(
                c, 64, -1).astype(np.float16)
            self.packed_s = np.ascontiguousarray(np.concatenate(
                [np.ascontiguousarray(codes.transpose(0, 2, 1)).reshape(c, -1).view(np.uint8),
                 np.ascontiguousarray(factors.transpose(0, 2, 1)).reshape(c, -1).view(np.uint8)],
                axis=1))
            return
        per: int = 32 // self.sub
        sc: F32 = self.d.reshape(c, 64, nb * per)
        if self.m is not None:
            sc = np.stack([self.d, self.m], axis=2).reshape(c, 64, nb * 2)
        self.packed_s = np.ascontiguousarray(
            sc.transpose(0, 2, 1).astype(np.float16)).view(np.uint8)

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
        Q4_0, Q4_K and Q4_1 become int4 in the same 32-blocks (weight (q - 8) * scale + zero,
        with zero 0 for Q4_0 and 8 scale - min otherwise, rounded to bf16); Q8_0, Q6_K and Q5_K are
        requantized to int8 with one scale per row, as PyTorch has no per-block int8 kernel."""
        if self.fmt in ("q4", "q4k", "q41"):
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
        if self.fmt in ("q4", "q4k", "q41"):
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
            # Q4_K, Q5_K, Q4_1: minus min * the block's activation sum.
            y -= (xq.reshape(ng, 32).sum(axis=1) * dx)[:, None] * self.m_blocks
        return y.sum(axis=0)


def pad_rows(m: Mat, multiple: int = 64) -> Mat:
    """m in Q8_0 with zero rows after its own up to a multiple of `multiple` rows (a router
    whose experts are not a multiple of 64: the kernel computes logits 64 rows at a time and
    reads only the experts' own)."""
    q8: Mat = m if m.fmt == "q8" else m.to_q8()
    pad: int = -q8.rows % multiple
    if not pad:
        return q8
    nb: int = q8.q.shape[1]
    zero: Mat = Mat("q8", np.zeros((pad, nb, 32), np.int8), np.zeros((pad, nb), np.float32))
    return Mat.concat([q8, zero])


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
    # The dense MLP, or None in a mixture-of-experts layer.
    gate_up: Mat | None
    down: Mat | None
    # Qwen3's RMSNorm weights of each q and k head [head_dim], applied before RoPE.
    q_norm: F32 | None
    k_norm: F32 | None
    # A mixture-of-experts layer's router [experts x d] and each expert's gate/up
    # [2 expert_ffn x d] (chunks interleaved as the dense gate/up's) and down [d x expert_ffn].
    router: Mat | None
    experts: list[tuple[Mat, Mat]] | None
    # gpt-oss's biases: of the QKV rows, the O projection, the router's logits, and each
    # expert's gate/up rows (laid out as its rows) and down projection.
    qkv_b: F32 | None
    o_b: F32 | None
    router_b: F32 | None
    expert_b: list[tuple[F32, F32]] | None
    # gpt-oss's attention sink logit per query head, and the positions a query sees (0: all).
    sinks: F32 | None
    window: int


def layer(attn_norm: F32, qkv: Mat | Parts, o: Mat, ffn_norm: F32,
          gate_up: Mat | None = None, down: Mat | None = None, q_norm: F32 | None = None,
          k_norm: F32 | None = None, router: Mat | None = None,
          experts: list[tuple[Mat, Mat]] | None = None, qkv_b: F32 | None = None,
          o_b: F32 | None = None, router_b: F32 | None = None,
          expert_b: list[tuple[F32, F32]] | None = None, sinks: F32 | None = None,
          window: int = 0) -> Layer:
    """A Layer, the parts a model does not have left out."""
    return Layer(attn_norm=attn_norm, qkv=qkv, o=o, ffn_norm=ffn_norm, gate_up=gate_up,
                 down=down, q_norm=q_norm, k_norm=k_norm, router=router, experts=experts,
                 qkv_b=qkv_b, o_b=o_b, router_b=router_b, expert_b=expert_b, sinks=sinks,
                 window=window)


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


def q8_round(x: F32) -> F32:
    """Each row of x rounded to int8 over one scale (quantize with the row as its block) and
    back: the values of the kernel's int8 KV cache and attention queries."""
    q, d = quantize(x, x.shape[-1])
    r: F32 = (q.reshape(-1, x.shape[-1]) * d[:, None]).astype(np.float32)
    return r.reshape(x.shape)


def bf16_round(x: npt.ArrayLike) -> F32:
    """x rounded to bfloat16 (nearest even), as f32: the values the kernel's KV cache holds."""
    b: npt.NDArray[np.uint64] = np.ascontiguousarray(x, dtype=np.float32).view(
        np.uint32).astype(np.uint64)
    r: Ints = ((b + 0x7FFF + ((b >> 16) & 1)) >> 16) << 16
    return r.astype(np.uint32).view(np.float32).reshape(np.shape(x))


class Yarn(TypedDict):
    """YaRN's RoPE scaling (gpt-oss: factor 32 over 4096 positions)."""
    factor: float
    orig_ctx: int
    beta_fast: float
    beta_slow: float


def rope_tables(max_ctx: int, head_dim: int, base: float, freq_factors: F32 | None = None,
                yarn: Yarn | None = None) -> tuple[F32, F32]:
    """cos and sin [max_ctx x head_dim/2] as llama.cpp computes them: theta_i = pos *
    base^(-2i/head_dim), by repeated f32 multiplication, divided by the frequency factor.
    With YaRN (ggml's rope_yarn): each angle mixes theta / factor and theta by a ramp over
    the dimensions between the correction dimensions of beta_fast and beta_slow, and cos and
    sin are scaled by 1 + 0.1 ln(factor) (llama.cpp's attention factor cancels its own)."""
    half: int = head_dim // 2
    ff: F32 = np.ones(half, np.float32) if freq_factors is None else \
        freq_factors.astype(np.float32)
    scale: np.float32 = np.float32(base) ** np.float32(-2.0 / head_dim)
    theta: F32 = np.arange(max_ctx, dtype=np.float32)
    ang: F32 = np.empty((max_ctx, half), dtype=np.float32)
    mscale: np.float32 = np.float32(1.0)
    ramp: F32 = np.zeros(half, np.float32)
    freq_scale: np.float32 = np.float32(1.0)
    if yarn is not None:
        freq_scale = np.float32(1.0 / yarn["factor"])

        def corr_dim(n_rot: float) -> float:
            return head_dim * math.log(yarn["orig_ctx"] / (n_rot * 2 * math.pi)) / (
                2 * math.log(base))

        low: float = max(0.0, math.floor(corr_dim(yarn["beta_fast"])))
        high: float = min(head_dim - 1.0, math.ceil(corr_dim(yarn["beta_slow"])))
        y: F32 = ((np.arange(half) - low) / max(0.001, high - low)).astype(np.float32)
        ramp = (1 - np.clip(y, 0, 1)).astype(np.float32)
        mscale = np.float32(1.0 + 0.1 * math.log(yarn["factor"]))
    for i in range(half):
        extrap: F32 = theta / ff[i]
        if yarn is None:
            ang[:, i] = extrap
        else:
            interp: F32 = freq_scale * extrap
            ang[:, i] = interp * (1 - ramp[i]) + extrap * ramp[i]
        theta = theta * scale
    return (np.cos(ang) * mscale).astype(np.float32), (np.sin(ang) * mscale).astype(np.float32)


class Model:
    # Each matrix's formats in a GGUF model ("lm" and MATS), from from_gguf.
    formats: dict[str, list[str]]
    # The last numpy_step's hidden state and logits.
    last_h: F32
    last_logits: F32

    def __init__(self, cfg: tuple[int, int, int, int, int, int, int, int], embd: F32,
                 norm_out: F32, lm: Mat, cos: F32, sin: F32, layers: list[Layer], eps: float,
                 numpy: bool = False, use_torch: bool = False, kv_q8: bool = False,
                 moe: tuple[int, int, int] | None = None, act: int = 0) -> None:
        d, nl, nh, nkv, hd, f, vocab, max_ctx = cfg
        # A mixture-of-experts model's experts, experts used per token and how their weights
        # are made (the kernel's gating: 0 softmax probabilities, 1 renormalized to sum to 1,
        # 2 a softmax of the picked logits); f is then an expert's MLP width.
        self.moe: tuple[int, int, int] | None = moe
        n_exp, n_used, gating = moe if moe is not None else (0, 0, 0)
        # The MLP's activation: 0 SwiGLU, 1 gpt-oss's clamped SwiGLU.
        self.act: int = act
        # The kernel's cfg: the shapes, the KV cache's format (0 bf16, 1 int8), then the
        # experts, the experts used, an expert's MLP width and the gating (0s if dense), and
        # the activation.
        self.cfg: npt.NDArray[np.int64] = np.array(
            cfg + (int(kv_q8), n_exp, n_used, f if moe is not None else 0, int(gating), act),
            dtype=np.int64)
        self.kv_q8: bool = kv_q8
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
        # Each mixture-of-experts layer's expert table: 4 words per expert (gate/up, down,
        # and their biases or 0).
        self.expert_tables: list[npt.NDArray[np.uint64]] = []

        def opt_mat(m: Mat | None) -> int:
            return 0 if m is None else mat(m)

        def opt(v: F32 | None) -> int:
            return 0 if v is None else v.ctypes.data

        def expert_table(es: list[tuple[Mat, Mat]] | None,
                         bs: list[tuple[F32, F32]] | None) -> int:
            if es is None:
                return 0
            t: npt.NDArray[np.uint64] = np.array(
                [w for e, (gu, dn) in enumerate(es) for w in (
                    mat(gu), mat(dn), opt(bs[e][0]) if bs else 0, opt(bs[e][1]) if bs else 0)],
                dtype=np.uint64)
            self.expert_tables.append(t)
            return t.ctypes.data

        for l in layers:
            words += [l["attn_norm"].ctypes.data, mat(l["qkv"]), mat(l["o"]),
                      l["ffn_norm"].ctypes.data, opt_mat(l["gate_up"]), opt_mat(l["down"]),
                      opt(l["q_norm"]), opt(l["k_norm"]), opt_mat(l["router"]),
                      expert_table(l["experts"], l["expert_b"]), opt(l["qkv_b"]),
                      opt(l["o_b"]), opt(l["router_b"]), opt(l["sinks"]), l["window"]]
        self.table: npt.NDArray[np.uint64] = np.array(words, dtype=np.uint64)

    def matrices(self) -> list[Mat | Parts]:
        """The LM head and every layer's matrices, experts included."""
        out: list[Mat | Parts] = [self.lm]
        for l in self.layers:
            out += [m for m in (l["qkv"], l["o"], l["gate_up"], l["down"], l["router"])
                    if m is not None]
            out += [m for pair in l["experts"] or [] for m in pair]
        return out

    @classmethod
    def random(cls, d: int, layers: int, heads: int, kv_heads: int, head_dim: int, ffn: int,
               vocab: int, max_ctx: int, fmt: str = "q8", numpy: bool = False,
               use_torch: bool = False, kv_q8: bool = False, qk_norm: bool = False,
               moe: tuple[int, int, int] | None = None, gpt_oss: int | None = None) -> Model:
        """A random model; with `moe` (experts, experts used, gating), every layer's MLP is a
        mixture of experts, each `ffn` wide. With `gpt_oss` (a window), gpt-oss's additions:
        QKV, O, router and expert biases, attention sinks, the window in the even layers,
        weights from a softmax of the picked logits, and the clamped SwiGLU."""
        rng: np.random.Generator = np.random.default_rng(0)
        ls: list[Layer] = []
        oss: bool = gpt_oss is not None

        def head_norm() -> F32 | None:
            return rng.uniform(0.8, 1.2, head_dim).astype(np.float32) if qk_norm else None

        def bias(n: int, scale: float) -> F32 | None:
            return rng.uniform(-scale, scale, n).astype(np.float32) if oss else None

        if moe is not None and oss:
            moe = (moe[0], moe[1], 2)
        for i in range(layers):
            n_exp: int = moe[0] if moe else 0
            experts: list[tuple[Mat, Mat]] | None = [
                (Mat.random(rng, 2 * ffn, d, fmt), Mat.random(rng, d, ffn, fmt))
                for _ in range(n_exp)] if moe else None
            expert_b: list[tuple[F32, F32]] | None = [
                (rng.uniform(-0.5, 0.5, 2 * ffn).astype(np.float32),
                 rng.uniform(-0.2, 0.2, d).astype(np.float32))
                for _ in range(n_exp)] if moe and oss else None
            ls.append(layer(
                attn_norm=rng.uniform(0.8, 1.2, d).astype(np.float32),
                qkv=Mat.random(rng, (heads + 2 * kv_heads) * head_dim, d, fmt),
                o=Mat.random(rng, d, heads * head_dim, fmt),
                ffn_norm=rng.uniform(0.8, 1.2, d).astype(np.float32),
                gate_up=None if moe else Mat.random(rng, 2 * ffn, d, fmt),
                down=None if moe else Mat.random(rng, d, ffn, fmt),
                q_norm=head_norm(),
                k_norm=head_norm(),
                router=pad_rows(Mat.random(rng, n_exp, d, "q8")) if moe else None,
                experts=experts,
                qkv_b=bias((heads + 2 * kv_heads) * head_dim, 0.5),
                o_b=bias(d, 0.2),
                router_b=bias(n_exp, 0.5) if moe else None,
                expert_b=expert_b,
                # Sinks around the scores' own range, so they take a share of every softmax.
                sinks=bias(heads, 1.5),
                window=(gpt_oss or 0) if i % 2 == 0 else 0,
            ))
        cos, sin = rope_tables(max_ctx, head_dim, 500000.0)
        return cls((d, layers, heads, kv_heads, head_dim, ffn, vocab, max_ctx),
                   rng.standard_normal((vocab, d), dtype=np.float32),
                   rng.uniform(0.8, 1.2, d).astype(np.float32),
                   Mat.random(rng, vocab, d, fmt), cos, sin, ls, 1e-5, numpy, use_torch, kv_q8,
                   moe, 1 if oss else 0)

    @classmethod
    def from_gguf(cls, path: str | GGUF, max_ctx: int, numpy: bool = False,
                  use_torch: bool = False, kv_q8: bool = False) -> Model:
        g: GGUF = path if isinstance(path, GGUF) else GGUF(path)
        m: dict[str, Any] = g.meta
        t: dict[str, Tensor] = g.tensors
        arch: str | None = m.get("general.architecture")
        if arch not in ("llama", "qwen3", "qwen3moe", "gpt-oss"):
            raise ValueError(f"unsupported architecture {arch!r} (llama, qwen3, qwen3moe and "
                             "gpt-oss only)")
        oss: bool = arch == "gpt-oss"
        a: str = f"{arch}."
        d: int = m[a + "embedding_length"]
        nl: int = m[a + "block_count"]
        nh: int = m[a + "attention.head_count"]
        nkv: int = m.get(a + "attention.head_count_kv", nh)
        hd: int = m.get(a + "attention.key_length", d // nh)
        n_exp: int = m.get(a + "expert_count", 0)
        # A mixture-of-experts model's experts each have an MLP of their own width.
        f: int = m[a + "expert_feed_forward_length"] if n_exp else m[a + "feed_forward_length"]
        # Qwen3-MoE renormalizes the picked experts' probabilities; gpt-oss takes a softmax of
        # their logits.
        moe: tuple[int, int, int] | None = (n_exp, m[a + "expert_used_count"], 2 if oss else 1) \
            if n_exp else None
        vocab: int = t["token_embd.weight"].rows
        rope_dims: int = m.get(a + "rope.dimension_count", hd)
        if rope_dims != hd or m.get(a + "attention.value_length", hd) != hd:
            raise ValueError("partial RoPE and differing K/V head sizes are not supported")
        ff: F32 | None = t["rope_freqs.weight"].dequantize()[0] \
            if "rope_freqs.weight" in t else None
        yarn: Yarn | None = None
        if m.get(a + "rope.scaling.type") == "yarn":
            yarn = Yarn(factor=m[a + "rope.scaling.factor"],
                        orig_ctx=m[a + "rope.scaling.original_context_length"],
                        beta_fast=m.get(a + "rope.scaling.yarn_beta_fast", 32.0),
                        beta_slow=m.get(a + "rope.scaling.yarn_beta_slow", 1.0))
        cos, sin = rope_tables(max_ctx, hd, m.get(a + "rope.freq_base", 10000.0), ff, yarn)
        # gpt-oss: a window in the even layers (llama.cpp's default pattern of 2).
        window: int = m.get(a + "attention.sliding_window", 0) if oss else 0

        embd: F32 = np.empty((vocab, d), dtype=np.float32)
        for r in range(0, vocab, 8192):
            embd[r:r + 8192] = t["token_embd.weight"].dequantize(slice(r, r + 8192))
        lm: Mat = Mat.from_tensor(t.get("output.weight", t["token_embd.weight"]))

        def hf_rows(n_heads: int) -> npt.NDArray[np.intp]:
            # llama.cpp stores each Llama head's rows as (i, half) pairs interleaved; Hugging
            # Face (and the kernel's RoPE) puts the first halves first. Qwen3 uses that order
            # in GGUF files too (NeoX RoPE).
            rows: npt.NDArray[np.intp] = np.arange(n_heads * hd)
            if arch != "llama":
                return rows
            return rows.reshape(n_heads, hd // 2, 2).transpose(0, 2, 1).reshape(-1)

        def norm(name: str) -> F32:
            return np.ascontiguousarray(t[name].dequantize()[0])

        def vec(name: str) -> F32 | None:
            return np.ascontiguousarray(t[name].dequantize().reshape(-1)) if name in t else None

        layers: list[Layer] = []
        for i in range(nl):
            p: str = f"blk.{i}."
            q: Mat = Mat.from_tensor(t[p + "attn_q.weight"]).take(hf_rows(nh))
            k: Mat = Mat.from_tensor(t[p + "attn_k.weight"]).take(hf_rows(nkv))
            v: Mat = Mat.from_tensor(t[p + "attn_v.weight"])
            # Gate/up chunks interleave: gate rows 64c.., then up rows 64c...
            gu_rows: npt.NDArray[np.intp] = np.arange(2 * f).reshape(2, f // 64, 64).transpose(
                1, 0, 2).reshape(-1)
            gate_up: Mat | None = None
            down: Mat | None = None
            router: Mat | None = None
            experts: list[tuple[Mat, Mat]] | None = None
            expert_b: list[tuple[F32, F32]] | None = None
            if moe is None:
                gate: Mat = Mat.from_tensor(t[p + "ffn_gate.weight"])
                up: Mat = Mat.from_tensor(t[p + "ffn_up.weight"])
                gate_up = Mat.concat([gate, up]).take(gu_rows)
                down = Mat.from_tensor(t[p + "ffn_down.weight"])
            else:
                # [experts x expert_ffn x d] tensors (dims innermost first), one expert's
                # rows after another's.
                gates: Mat = Mat.from_tensor(t[p + "ffn_gate_exps.weight"])
                ups: Mat = Mat.from_tensor(t[p + "ffn_up_exps.weight"])
                downs: Mat = Mat.from_tensor(t[p + "ffn_down_exps.weight"])
                experts = []
                for e in range(n_exp):
                    er: npt.NDArray[np.intp] = np.arange(e * f, (e + 1) * f)
                    ed: npt.NDArray[np.intp] = np.arange(e * d, (e + 1) * d)
                    experts.append((Mat.concat([gates.take(er), ups.take(er)]).take(gu_rows),
                                    downs.take(ed)))
                router = pad_rows(Mat.from_tensor(t[p + "ffn_gate_inp.weight"]))
                if p + "ffn_gate_exps.bias" in t:
                    # [experts x expert_ffn] and [experts x d]; gate/up's laid out as its rows.
                    gb: F32 = t[p + "ffn_gate_exps.bias"].dequantize()
                    ub: F32 = t[p + "ffn_up_exps.bias"].dequantize()
                    db: F32 = t[p + "ffn_down_exps.bias"].dequantize()
                    expert_b = [(np.ascontiguousarray(np.concatenate([gb[e], ub[e]])[gu_rows]),
                                 np.ascontiguousarray(db[e])) for e in range(n_exp)]
            qkv_b: F32 | None = None
            if p + "attn_q.bias" in t:
                qb: F32 | None = vec(p + "attn_q.bias")
                kb: F32 | None = vec(p + "attn_k.bias")
                vb: F32 | None = vec(p + "attn_v.bias")
                assert qb is not None and kb is not None and vb is not None
                qkv_b = np.concatenate([qb[hf_rows(nh)], kb[hf_rows(nkv)], vb])
            layers.append(layer(
                attn_norm=norm(p + "attn_norm.weight"),
                qkv=Parts.of([q, k, v]),
                o=Mat.from_tensor(t[p + "attn_output.weight"]),
                # gpt-oss names its MLP norm after attention's output.
                ffn_norm=norm(p + ("post_attention_norm.weight" if oss else "ffn_norm.weight")),
                gate_up=gate_up,
                down=down,
                router=router,
                experts=experts,
                q_norm=norm(p + "attn_q_norm.weight") if p + "attn_q_norm.weight" in t else None,
                k_norm=norm(p + "attn_k_norm.weight") if p + "attn_k_norm.weight" in t else None,
                qkv_b=qkv_b,
                o_b=vec(p + "attn_output.bias"),
                router_b=vec(p + "ffn_gate_inp.bias"),
                expert_b=expert_b,
                sinks=vec(p + "attn_sinks.weight"),
                window=window if i % 2 == 0 else 0,
            ))
        eps: float = m.get(a + "attention.layer_norm_rms_epsilon", 1e-5)
        model: Model = cls((d, nl, nh, nkv, hd, f, vocab, max_ctx), embd, norm(
            "output_norm.weight"), lm, cos, sin, layers, eps, numpy, use_torch, kv_q8, moe,
            1 if oss else 0)
        model.formats = {}
        for n in MATS:
            fmts: set[str] = {mt.fmt for l in layers if (mt := l[n]) is not None}
            if fmts:
                model.formats[n] = sorted(fmts)
        if moe is not None:
            model.formats["experts"] = sorted(
                {mt.fmt for l in layers for pair in l["experts"] or [] for mt in pair})
            model.formats["router"] = sorted(
                {mt.fmt for l in layers if (mt := l["router"]) is not None})
        model.formats["lm"] = [lm.fmt]
        return model

    def weight_bytes(self) -> int:
        """Bytes of packed matrices one token reads (embedding row and norms aside): of a
        mixture of experts, the experts it uses (their mean size times their number)."""
        total: int = 0
        for m in self.matrices():
            total += m.packed_nbytes()
        for l in self.layers:
            es: list[tuple[Mat, Mat]] | None = l["experts"]
            if es and self.moe is not None:
                all_e: int = sum(g.packed_nbytes() + d.packed_nbytes() for g, d in es)
                total -= all_e - all_e * self.moe[1] // len(es)
        return total

    def torch_weight_bytes(self) -> int:
        """Bytes of PyTorch's matrices one token reads (see Mat.prepare_torch)."""
        return sum(t.nbytes for m in self.matrices() for p in m.parts()
                   for t in ((p.t_w, p.t_sz) if p.fmt in ("q4", "q4k", "q41")
                             else (p.t_w, p.t_s)))

    @property
    def cache_ctx(self) -> int:
        """Positions per KV head in the kernel's cache: max_ctx rounded up to K's tiles."""
        return -(-self.max_ctx // 32) * 32

    @property
    def cache_rb(self) -> int:
        """Bytes per position of a KV head's K or V: hd bf16 values, or hd int8 values and
        an f32 scale."""
        return self.hd + 4 if self.kv_q8 else 2 * self.hd

    def kv_ring(self, l: int) -> tuple[int, int]:
        """Layer l's slots per KV head in the kernel's cache and its ring (the kernel's
        kv_ring): with a window, a ring of R = window + 31 + 256 slots rounded up to 32 held
        twice (2 R slots; position p at slot p mod R and R slots on), when that is less than
        cache_ctx; else cache_ctx and 0."""
        win: int = self.layers[l]["window"]
        ring: int = (win + 318) & ~31
        if win and 2 * ring < self.cache_ctx:
            return 2 * ring, ring
        return self.cache_ctx, 0

    def cache_words(self, l: int) -> int:
        """u16 words of layer l in the kernel's cache: K then V, nkv x kv_ring slots each."""
        return 2 * self.nkv * self.kv_ring(l)[0] * self.cache_rb // 2

    def new_cache(self) -> npt.NDArray[np.uint16]:
        """The kernel's KV cache, as u16 words: per layer, K then V, each nkv x ctx_l slots
        (kv_ring) of cache_rb bytes, in tiles of 32 slots. bf16: V is [nkv x ctx_l x hd], K
        [nkv x ctx_l / 32 x hd x 32]. int8: each tile holds the values, K's plus 128 as
        [hd / 4 x 32 x 4] and V's as [32 x hd], then an f32 scale per slot."""
        return np.zeros(sum(self.cache_words(l) for l in range(len(self.layers))),
                        dtype=np.uint16)

    def cache_at(self, cache: npt.NDArray[np.uint16], pos: int) -> tuple[F32, F32]:
        """K and V of position pos in a kernel cache as f32 values, each [layers x nkv x hd]
        (its slot in a ring: pos mod R)."""
        hd: int = self.hd
        out: list[list[F32]] = [[], []]
        at: int = 0
        for l in range(len(self.layers)):
            ctx, ring = self.kv_ring(l)
            words: npt.NDArray[np.uint16] = cache[at:at + self.cache_words(l)]
            at += self.cache_words(l)
            slot: int = pos % ring if ring else pos
            tile: int = slot // 32
            col: int = slot % 32
            if not self.kv_q8:
                bits: npt.NDArray[np.uint32] = words.astype(np.uint32) << np.uint32(16)
                planes: F32 = bits.view(np.float32).reshape(2, self.nkv, ctx // 32, 32 * hd)
                out[0].append(planes[0, :, tile].reshape(self.nkv, hd, 32)[:, :, col])
                out[1].append(planes[1, :, tile].reshape(self.nkv, 32, hd)[:, col])
                continue
            t: U8 = words.view(np.uint8).reshape(2, self.nkv, ctx // 32, 32 * (hd + 4))
            scales: F32 = np.ascontiguousarray(t[:, :, tile, 32 * hd:]).view(np.float32)
            kq: Ints = (t[0, :, tile, :32 * hd] ^ np.uint8(0x80)).view(np.int8).reshape(
                self.nkv, hd // 4, 32, 4)[:, :, col].reshape(self.nkv, hd)
            vq: Ints = t[1, :, tile, :32 * hd].view(np.int8).reshape(self.nkv, 32, hd)[:, col]
            out[0].append((kq * scales[0, :, col, None]).astype(np.float32))
            out[1].append((vq * scales[1, :, col, None]).astype(np.float32))
        k: F32 = np.stack(out[0])
        v: F32 = np.stack(out[1])
        return k, v

    # NumPy forward pass of the same model (needs numpy=True at construction).
    def norm(self, h: Floats, w: F32) -> Floats:
        y: Floats = h / np.sqrt(np.mean(h * h) + self.eps) * w
        return y

    def head_norm(self, x: F32, w: F32 | None) -> F32:
        """RMSNorm of each head (row) of x with the weights w, or x itself without them."""
        if w is None:
            return x
        y: F32 = (x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + self.eps) * w).astype(
            np.float32)
        return y

    def rope(self, x: F32, pos: int) -> F32:
        half: int = self.hd // 2
        a: F32 = x[..., :half]
        b: F32 = x[..., half:]
        c: F32 = self.cos[pos]
        s: F32 = self.sin[pos]
        return np.concatenate([a * c - b * s, b * c + a * s], axis=-1)

    def numpy_cache(self) -> list[tuple[F32, F32]]:
        # f32 arrays holding bf16 (or int8 times scale) values, as the kernel's cache does.
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
            qkv_b: F32 | None = layer["qkv_b"]
            if qkv_b is not None:
                qkv = qkv + qkv_b
            q: F32 = self.rope(self.head_norm(qkv[:nh * hd].reshape(nh, hd), layer["q_norm"]), pos)
            k: F32 = self.rope(self.head_norm(
                qkv[nh * hd:(nh + nkv) * hd].reshape(nkv, hd), layer["k_norm"]), pos)
            v: F32 = qkv[(nh + nkv) * hd:].reshape(nkv, hd)
            if self.kv_q8:
                # int8 K and V per position, and queries per head (attn_split_q8).
                q = q8_round(q)
                kc[pos] = q8_round(k)
                vc[pos] = q8_round(v)
            else:
                kc[pos] = bf16_round(k)
                vc[pos] = bf16_round(v)
            # The layer's window, and each head's sink: one more logit in the softmax, with
            # no value.
            lo: int = max(0, pos + 1 - layer["window"]) if layer["window"] else 0
            keys: F32 = np.repeat(kc[lo:pos + 1], nh // nkv, axis=1)       # [t, nh, hd]
            vals: F32 = np.repeat(vc[lo:pos + 1], nh // nkv, axis=1)
            scores: F32 = np.einsum("hd,thd->ht", q, keys) / np.sqrt(np.float32(hd))
            sinks: F32 | None = layer["sinks"]
            mx: F32 = scores.max(axis=1, keepdims=True)
            if sinks is not None:
                mx = np.maximum(mx, sinks[:, None])
            p: F32 = np.exp(scores - mx)
            total: F32 = p.sum(axis=1, keepdims=True)
            if sinks is not None:
                total = total + np.exp(sinks[:, None] - mx)
            attn: F32 = np.einsum("ht,thd->hd", p, vals) / total
            o: F32 = layer["o"].matvec(attn.reshape(-1))
            o_b: F32 | None = layer["o_b"]
            h = h + (o if o_b is None else o + o_b)
            xn: Floats = self.norm(h, layer["ffn_norm"])
            router: Mat | None = layer["router"]
            if router is None:
                gate_up: Mat | None = layer["gate_up"]
                down: Mat | None = layer["down"]
                assert gate_up is not None and down is not None
                h = h + self.mlp(gate_up, down, xn)
            else:
                h = h + self.moe_mlp(router, layer["experts"] or [], xn, layer["router_b"],
                                     layer["expert_b"])
        logits: F32 = self.lm.matvec(self.norm(h, self.norm_out))
        # What the kernel leaves in `h`, for tests.
        self.last_h = h
        self.last_logits = logits
        return int(np.argmax(logits))

    def mlp(self, gate_up: Mat, down: Mat, xn: Floats,
            bias: tuple[F32, F32] | None = None) -> F32:
        """The MLP: gate/up rows plus their bias, the activation (SwiGLU, or gpt-oss's clamped
        one), then down plus its bias."""
        gu_rows: F32 = gate_up.matvec(xn)
        if bias is not None:
            gu_rows = gu_rows + bias[0]
        gu: F32 = gu_rows.reshape(-1, 2, 64)
        g: F32 = gu[:, 0].reshape(-1)
        u: F32 = gu[:, 1].reshape(-1)
        if self.act == 1:
            x: F32 = np.minimum(g, np.float32(7.0))
            y: F32 = np.clip(u, np.float32(-7.0), np.float32(7.0))
            a: Floats = x / (1.0 + np.exp(np.float32(-1.702) * x)) * (y + 1)
        else:
            a = g / (1.0 + np.exp(-g)) * u
        out: F32 = down.matvec(a)
        return out if bias is None else out + bias[1]

    def moe_mlp(self, router: Mat, experts: list[tuple[Mat, Mat]], xn: Floats,
                router_b: F32 | None = None,
                expert_b: list[tuple[F32, F32]] | None = None) -> F32:
        """The kernel's mixture of experts: the experts with the largest softmax probabilities
        of the router's logits plus its bias (the first on ties), weighted by them
        (renormalized to sum to 1, or a softmax of the picked logits, if the model says so),
        their outputs added in that order."""
        assert self.moe is not None
        n_exp, n_used, gating = self.moe
        logits: F32 = router.matvec(xn)[:n_exp]
        if router_b is not None:
            logits = logits + router_b
        e: Floats = np.exp(logits - np.max(logits))
        p: Floats = e / e.sum()
        picked: list[int] = [int(i) for i in np.argsort(-p, kind="stable")[:n_used]]
        w: Floats = p[picked]
        if gating == 2:
            w = e[picked] / e[picked].sum()
        elif gating == 1:
            w = w / max(float(w.sum()), 2.0 ** -14)
        out: F32 = np.zeros(self.d, np.float32)
        for wi, i in zip(w, picked):
            out = out + np.float32(wi) * self.mlp(experts[i][0], experts[i][1], xn,
                                                  expert_b[i] if expert_b else None)
        return out

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

        def head_norm(x: torch.Tensor, w: F32 | None) -> torch.Tensor:
            return x if w is None else F.rms_norm(x, (hd,), torch.from_numpy(w), eps=eps)

        h: torch.Tensor = torch.from_numpy(self.embd[token].copy())
        for l, layer in enumerate(self.layers):
            if layer["qkv_b"] is not None or layer["sinks"] is not None or layer["window"]:
                raise NotImplementedError("PyTorch runs Llama and Qwen3 attention only")
            kc, vc = cache[l]
            qkv: torch.Tensor = layer["qkv"].torch_matvec(norm(h, layer["attn_norm"]))
            q: torch.Tensor = self.torch_rope(
                head_norm(qkv[:nh * hd].view(nh, hd), layer["q_norm"]), pos)
            kc[:, pos] = self.torch_rope(head_norm(
                qkv[nh * hd:(nh + nkv) * hd].view(nkv, hd), layer["k_norm"]), pos)
            vc[:, pos] = qkv[(nh + nkv) * hd:].view(nkv, hd)
            keys: torch.Tensor = kc[None, :, :pos + 1]  # [1, nkv, t, hd]
            vals: torch.Tensor = vc[None, :, :pos + 1]
            # Each KV head's query heads as its query positions: the same math as
            # enable_gqa=True with one position per head, which leaves SDPA's fast path.
            qg: torch.Tensor = q.view(1, nkv, nh // nkv, hd).bfloat16()
            attn: torch.Tensor = F.scaled_dot_product_attention(qg, keys, vals).float()
            h = h + layer["o"].torch_matvec(attn.reshape(-1))
            gate_up: Mat | None = layer["gate_up"]
            down: Mat | None = layer["down"]
            if gate_up is None or down is None:
                raise NotImplementedError("PyTorch runs dense MLPs only")
            gu: torch.Tensor = gate_up.torch_matvec(
                norm(h, layer["ffn_norm"])).view(-1, 2, 64)
            g: torch.Tensor = gu[:, 0].reshape(-1)
            u: torch.Tensor = gu[:, 1].reshape(-1)
            h = h + down.torch_matvec(F.silu(g) * u)
        logits: torch.Tensor = self.lm.torch_matvec(norm(h, self.norm_out))
        return int(torch.max(logits, 0).indices)
