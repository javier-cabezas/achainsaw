"""
End-to-end test of the GGUF path: writes a tiny Llama GGUF with every tensor type the loader
handles (Q4_0, Q8_0, Q6_K, Q4_K, Q5_K, Q4_1 and IQ4_NL natively; F16 re-quantized to Q8_0),
loads it with
benchmarks/gguf.py and benchmarks/llama_model.py, and checks
  - the dequantizers against the values the file was written from,
  - the loader's conversions (bit for bit where promised; Q and K rows back in Hugging Face
    order; fused matrices, including a QKV matrix whose parts keep different formats),
  - the BPE tokenizer on a small vocabulary,
  - examples/kernels/llama_decode.air against the NumPy forward pass of the loaded model, on
    every available backend;
and the same for a tiny Qwen3 GGUF (q and k head norms, Q and K rows in GGUF order already,
Qwen's pre-tokenizer and no BOS token).
Needs numpy and the achainsaw extension (no download).
"""

import os
import struct
import sys
import tempfile
import unittest
from collections.abc import Callable
from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt

ROOT: str = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
sys.path.insert(0, ROOT)
import achainsaw  # noqa: E402
from gguf import (F32, GGUF, IQ4_NL_VALUES, MXFP4_VALUES, U8, Floats, Ints,  # noqa: E402
                  Tensor, Tokenizer, _bytes_to_unicode, mxfp4_scale)
from llama_model import MATS, Layer, Mat, Model, Parts  # noqa: E402

D: int = 256
LAYERS: int = 3
HEADS: int = 4
KV_HEADS: int = 2
HEAD_DIM: int = 64
FFN: int = 512
VOCAB: int = 384
F16: type[np.float16] = np.float16

# (block bytes, values), and for Q4_K and Q5_K the codes too: see the encoders below.
Encoded: TypeAlias = tuple[npt.NDArray[np.uint8], Floats] | \
    tuple[npt.NDArray[np.uint8], Floats, npt.NDArray[np.uint8]]
Encoder: TypeAlias = Callable[..., Encoded]


# Each encoder takes integer codes and scales and returns (block bytes [rows x row bytes],
# the f32 values llama.cpp's dequantize_row_* gives for them).
def enc_q4_0(rng: np.random.Generator, rows: int, k: int) -> Encoded:
    q: U8 = rng.integers(0, 16, size=(rows, k // 32, 32), dtype=np.uint8)
    d: npt.NDArray[np.float16] = (
        rng.uniform(0.5, 1.0, size=(rows, k // 32)) * rng.choice([-1, 1], size=(rows, k // 32))
        / (4.6 * np.sqrt(k))).astype(F16)
    qs: Ints = q[:, :, :16] | (q[:, :, 16:] << 4)
    raw: U8 = np.concatenate([d[:, :, None].view(np.uint8), qs], axis=2)
    return raw.reshape(rows, -1), ((q.astype(np.float32) - 8) * d[:, :, None].astype(np.float32))


def enc_iq4_nl(rng: np.random.Generator, rows: int, k: int) -> Encoded:
    codes: U8 = rng.integers(0, 16, size=(rows, k // 32, 32), dtype=np.uint8)
    d: npt.NDArray[np.float16] = (rng.uniform(0.5, 1.0, size=(rows, k // 32))
                                  / (60.0 * np.sqrt(k))).astype(F16)
    qs: Ints = codes[:, :, :16] | (codes[:, :, 16:] << 4)
    raw: U8 = np.concatenate([d[:, :, None].view(np.uint8), qs], axis=2)
    vals: F32 = IQ4_NL_VALUES[codes].astype(np.float32) * d[:, :, None].astype(np.float32)
    return raw.reshape(rows, -1), vals


def enc_mxfp4(rng: np.random.Generator, rows: int, k: int) -> Encoded:
    codes: U8 = rng.integers(0, 16, size=(rows, k // 32, 32), dtype=np.uint8)
    # E8M0 scale bytes, 2^(e - 128) of roughly 1 / (5.9 sqrt(k)) (the codebook's rms is 5.9).
    e: U8 = np.round(128 + np.log2(rng.uniform(0.5, 1.0, size=(rows, k // 32))
                                    / (5.9 * np.sqrt(k)))).astype(np.uint8)
    qs: Ints = codes[:, :, :16] | (codes[:, :, 16:] << 4)
    raw: U8 = np.concatenate([e[:, :, None], qs], axis=2).astype(np.uint8)
    vals: F32 = MXFP4_VALUES[codes].astype(np.float32) * mxfp4_scale(e)[:, :, None]
    return raw.reshape(rows, -1), vals


def enc_q8_0(rng: np.random.Generator, rows: int, k: int) -> Encoded:
    q: npt.NDArray[np.int8] = rng.integers(-127, 128, size=(rows, k // 32, 32), dtype=np.int8)
    d: npt.NDArray[np.float16] = (rng.uniform(0.5, 1.0, size=(rows, k // 32))
                                  / (73.0 * np.sqrt(k))).astype(F16)
    raw: U8 = np.concatenate([d[:, :, None].view(np.uint8), q.view(np.uint8)], axis=2)
    return raw.reshape(rows, -1), q.astype(np.float32) * d[:, :, None].astype(np.float32)


def enc_q4_1(rng: np.random.Generator, rows: int, k: int) -> Encoded:
    q: U8 = rng.integers(0, 16, size=(rows, k // 32, 32), dtype=np.uint8)
    d: npt.NDArray[np.float16] = (rng.uniform(0.5, 1.0, size=(rows, k // 32))
                                  / (4.6 * np.sqrt(k))).astype(F16)
    m: npt.NDArray[np.float16] = (-8 * d.astype(np.float32)).astype(F16)
    qs: Ints = q[:, :, :16] | (q[:, :, 16:] << 4)
    raw: U8 = np.concatenate([d[:, :, None].view(np.uint8), m[:, :, None].view(np.uint8), qs],
                             axis=2)
    vals: F32 = q.astype(np.float32) * d[:, :, None].astype(np.float32) + \
        m[:, :, None].astype(np.float32)
    return raw.reshape(rows, -1), vals


def enc_q6_k(rng: np.random.Generator, rows: int, k: int) -> Encoded:
    n: int = rows * k // 256
    q: U8 = rng.integers(0, 64, size=(n, 256), dtype=np.uint8)
    sc: npt.NDArray[np.int8] = rng.integers(-127, 128, size=(n, 16), dtype=np.int8)
    d: npt.NDArray[np.float16] = (rng.uniform(0.5, 1.0, size=n)
                                  / (18.5 * 64 * np.sqrt(k))).astype(F16)
    ql: U8 = np.zeros((n, 128), np.uint8)
    qh: U8 = np.zeros((n, 64), np.uint8)
    for h in range(2):
        q1, q2, q3, q4 = (q[:, 128 * h + 32 * p:128 * h + 32 * p + 32] for p in range(4))
        ql[:, 64 * h:64 * h + 32] = (q1 & 15) | ((q3 & 15) << 4)
        ql[:, 64 * h + 32:64 * h + 64] = (q2 & 15) | ((q4 & 15) << 4)
        qh[:, 32 * h:32 * h + 32] = (q1 >> 4) | ((q2 >> 4) << 2) | ((q3 >> 4) << 4) | \
            ((q4 >> 4) << 6)
    raw: U8 = np.concatenate([ql, qh, sc.view(np.uint8), d[:, None].view(np.uint8)], axis=1)
    ds: F32 = d.astype(np.float32)[:, None] * sc.astype(np.float32)     # [n x 16]
    vals: Floats = np.repeat(ds, 16, axis=1) * (q.astype(np.float32) - 32)
    return raw.reshape(rows, -1), vals.reshape(rows, k)


def enc_k(bits: int) -> Encoder:
    """Q4_K (bits 4) or Q5_K (5): 256-value blocks of 8 sub-blocks, each with a 6-bit scale
    and min under the block's f16 d and dmin; also returns the codes [rows x k]."""
    def enc(rng: np.random.Generator, rows: int, k: int) -> Encoded:
        n: int = rows * k // 256
        q: U8 = rng.integers(0, 1 << bits, size=(n, 256), dtype=np.uint8)
        s6: U8 = rng.integers(0, 64, size=(n, 8), dtype=np.uint8)
        m6: U8 = rng.integers(0, 64, size=(n, 8), dtype=np.uint8)
        d: npt.NDArray[np.float16] = (rng.uniform(0.5, 1.0, size=n)
                                      / (4.6 * (1 << (bits - 4)) * 48 * np.sqrt(k))).astype(F16)
        dmin: npt.NDArray[np.float16] = (d.astype(np.float32) * 8 * (1 << (bits - 4)) * 48
                                         / 32).astype(F16)
        # llama.cpp's packing of the 6-bit scales and mins into 12 bytes.
        sc: U8 = np.zeros((n, 12), np.uint8)
        sc[:, 0:4] = s6[:, 0:4] | ((s6[:, 4:8] >> 4) << 6)
        sc[:, 4:8] = m6[:, 0:4] | ((m6[:, 4:8] >> 4) << 6)
        sc[:, 8:12] = (s6[:, 4:8] & 15) | ((m6[:, 4:8] & 15) << 4)
        lo: Ints = q & 15
        qs: Ints = np.concatenate([lo[:, 64 * p:64 * p + 32] | (lo[:, 64 * p + 32:64 * p + 64] << 4)
                                   for p in range(4)], axis=1)
        head: list[npt.NDArray[Any]] = [d[:, None].view(np.uint8), dmin[:, None].view(np.uint8),
                                        sc]
        if bits == 5:
            qh: Ints = np.zeros((n, 32), np.uint8)
            for j in range(8):
                qh |= (q[:, 32 * j:32 * j + 32] >> 4) << j
            head.append(qh)
        raw: U8 = np.concatenate(head + [qs], axis=1)
        ds: F32 = d.astype(np.float32)[:, None] * s6.astype(np.float32)   # [n x 8]
        dm: F32 = dmin.astype(np.float32)[:, None] * m6.astype(np.float32)
        vals: F32 = q.reshape(n, 8, 32).astype(np.float32) * ds[:, :, None] - dm[:, :, None]
        return raw.reshape(rows, -1), vals.reshape(rows, k), q.reshape(rows, k)
    return enc


def enc_f16(rng: np.random.Generator, rows: int, k: int) -> Encoded:
    v: npt.NDArray[np.float16] = (rng.standard_normal((rows, k)) / np.sqrt(k)).astype(F16)
    return v.view(np.uint8).reshape(rows, -1), v.astype(np.float32)


def enc_f32(rng: np.random.Generator, rows: int, k: int,
            values: npt.NDArray[Any] | None = None) -> Encoded:
    v: F32 = (rng.uniform(0.8, 1.2, (rows, k)) if values is None else values).astype(np.float32)
    return v.view(np.uint8).reshape(rows, -1), v


def need(x: Mat | None) -> Mat:
    """A dense layer's MLP matrix, which it always has."""
    assert x is not None
    return x


ENCODERS: dict[int, Encoder] = {2: enc_q4_0, 8: enc_q8_0, 3: enc_q4_1, 20: enc_iq4_nl,
                                39: enc_mxfp4,
                                14: enc_q6_k, 12: enc_k(4), 13: enc_k(5), 1: enc_f16,
                                0: enc_f32}

# Per layer, the type of each weight: layer 0 is all Q4_0 but an MXFP4 K and a Q4_1 down
# projection;
# layer 1 mixes formats inside the fused QKV (Q8_0, Q4_0, IQ4_NL) and gate/up matrices; layer 2 is laid out like
# a Q4_K_M file's (Q and K Q4_K, V Q6_K), with Q5_K O and down projections.
LAYER_TYPES: list[dict[str, int]] = [
    dict(attn_q=2, attn_k=39, attn_v=2, attn_output=2, ffn_gate=2, ffn_up=2, ffn_down=3),
    dict(attn_q=8, attn_k=2, attn_v=20, attn_output=14, ffn_gate=1, ffn_up=2, ffn_down=2),
    dict(attn_q=12, attn_k=12, attn_v=14, attn_output=13, ffn_gate=12, ffn_up=12, ffn_down=13),
]
SHAPES: dict[str, tuple[int, int]] = dict(
    attn_q=(HEADS * HEAD_DIM, D), attn_k=(KV_HEADS * HEAD_DIM, D),
    attn_v=(KV_HEADS * HEAD_DIM, D), attn_output=(D, HEADS * HEAD_DIM),
    ffn_gate=(FFN, D), ffn_up=(FFN, D), ffn_down=(D, FFN))


def gguf_order(n_heads: int) -> npt.NDArray[np.intp]:
    """Row order llama.cpp's converter gives Q and K (HF row for each GGUF row)."""
    return np.arange(n_heads * HEAD_DIM).reshape(n_heads, 2, HEAD_DIM // 2).transpose(
        0, 2, 1).reshape(-1)


def tiny_vocab() -> tuple[list[str], list[int], list[str]]:
    byte_chars: dict[int, str] = _bytes_to_unicode()
    tokens: list[str] = [byte_chars[b] for b in range(256)]
    merges: list[str] = ["h e", "Ġ t", "Ġt he", "c a", "Ġ ca", "Ġca t"]
    tokens += ["he", "Ġt", "Ġthe", "ca", "Ġca", "Ġcat"]
    types: list[int] = [1] * len(tokens)
    for name in ["<|begin_of_text|>", "<|end_of_text|>", "<|eot_id|>"]:
        tokens.append(name)
        types.append(3)
    while len(tokens) < VOCAB:
        tokens.append(f"<|reserved_special_token_{len(tokens)}|>")
        types.append(3)
    return tokens, types, merges


def write_gguf(path: str, meta: dict[str, tuple[int, Any]],
               tensors: list[tuple[str, list[int], int, U8]]) -> None:
    """meta: {key: (gguf type, value)}; tensors: [(name, dims innermost first, type, bytes)]."""
    def s(x: str) -> bytes:
        b: bytes = x.encode("utf-8")
        return struct.pack("<Q", len(b)) + b

    def value(t: int, v: Any) -> bytes:
        if t == 8:
            return s(v)
        if t == 9:
            et, items = v
            return struct.pack("<IQ", et, len(items)) + b"".join(value(et, i) for i in items)
        return struct.pack({4: "<I", 5: "<i", 6: "<f", 7: "<?"}[t], v)

    out: bytearray = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(meta)))
    for k, (t, v) in meta.items():
        out += s(k) + struct.pack("<I", t) + value(t, v)
    offset: int = 0
    blobs: list[bytes] = []
    for name, dims, t, data in tensors:
        out += s(name) + struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}Q", *dims)
        out += struct.pack("<IQ", t, offset)
        blob: bytes = data.tobytes()
        blob += b"\0" * (-len(blob) % 32)
        blobs.append(blob)
        offset += len(blob)
    out += b"\0" * (-len(out) % 32)
    with open(path, "wb") as f:
        f.write(out)
        for b in blobs:
            f.write(b)


class TestGGUFDecode(unittest.TestCase):
    # The values each tensor was written from, and Q4_K/Q5_K tensors' codes.
    expected: dict[str, Floats]
    k_codes: dict[str, U8]
    tmp: tempfile.TemporaryDirectory[str]
    path: str
    gguf: GGUF

    @classmethod
    def setUpClass(cls) -> None:
        rng: np.random.Generator = np.random.default_rng(5)
        tokens, types, merges = tiny_vocab()
        meta: dict[str, tuple[int, Any]] = {
            "general.architecture": (8, "llama"),
            "general.name": (8, "tiny"),
            "llama.block_count": (4, LAYERS),
            "llama.embedding_length": (4, D),
            "llama.feed_forward_length": (4, FFN),
            "llama.attention.head_count": (4, HEADS),
            "llama.attention.head_count_kv": (4, KV_HEADS),
            "llama.rope.freq_base": (6, 500000.0),
            "llama.rope.dimension_count": (4, HEAD_DIM),
            "llama.attention.layer_norm_rms_epsilon": (6, 1e-5),
            "tokenizer.ggml.model": (8, "gpt2"),
            "tokenizer.ggml.tokens": (9, (8, tokens)),
            "tokenizer.ggml.token_type": (9, (5, types)),
            "tokenizer.ggml.merges": (9, (8, merges)),
            "tokenizer.ggml.bos_token_id": (4, tokens.index("<|begin_of_text|>")),
            "tokenizer.ggml.eos_token_id": (4, tokens.index("<|eot_id|>")),
        }
        cls.expected = {}
        cls.k_codes = {}
        tensors: list[tuple[str, list[int], int, U8]] = []

        def add(name: str, t: int, rows: int, k: int, **kw: Any) -> None:
            raw, vals, *codes = ENCODERS[t](rng, rows, k, **kw)
            cls.expected[name] = vals.reshape(rows, k)
            if codes:
                cls.k_codes[name] = codes[0]
            dims: list[int] = [k] if rows == 1 else [k, rows]
            tensors.append((name, dims, t, raw))

        add("token_embd.weight", 14, VOCAB, D)                 # tied: also the LM head
        add("output_norm.weight", 0, 1, D)
        ff: F32 = np.array([1.0] * 16 + [2.0, 4.0] * 8, np.float32)  # Llama 3 style RoPE factors
        add("rope_freqs.weight", 0, 1, HEAD_DIM // 2, values=ff[None, :])
        for i, types_ in enumerate(LAYER_TYPES):
            add(f"blk.{i}.attn_norm.weight", 0, 1, D)
            add(f"blk.{i}.ffn_norm.weight", 0, 1, D)
            for name, t in types_.items():
                rows, k = SHAPES[name]
                add(f"blk.{i}.{name}.weight", t, rows, k)
        # Windows cannot delete a file that a live memory map still holds.
        cls.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.path = os.path.join(cls.tmp.name, "tiny.gguf")
        write_gguf(cls.path, meta, tensors)
        cls.gguf = GGUF(cls.path)

    @classmethod
    def tearDownClass(cls) -> None:
        del cls.gguf
        cls.tmp.cleanup()

    def test_dequantize_matches_written_values(self) -> None:
        for name, want in self.expected.items():
            got: F32 = self.gguf.tensors[name].dequantize()
            np.testing.assert_array_equal(got, want, err_msg=name)

    def test_loader_conversions(self) -> None:
        m: Model = Model.from_gguf(self.gguf, max_ctx=8, numpy=True)

        def weights(mat: Mat | Parts) -> Floats:
            # The loaded matrix as f32 [rows x k], from its integer values, scales and mins.
            out: list[Floats] = []
            for p in mat.parts():
                v: Floats = p.q_blocks * p.s_blocks[:, :, None]           # [groups, rows, sub]
                if p.m_blocks is not None:
                    v = v - p.m_blocks[:, :, None]
                out.append(v.transpose(1, 0, 2).reshape(p.rows, -1))
            return np.concatenate(out)

        e: dict[str, Floats] = self.expected
        l0, l1, l2 = m.layers
        self.assertEqual((l0["qkv"].fmt, l0["o"].fmt, need(l0["gate_up"]).fmt, need(l0["down"]).fmt),
                         ("q4+mx4+q4", "q4", "q4", "q41"))
        self.assertEqual((l1["qkv"].fmt, l1["o"].fmt, need(l1["gate_up"]).fmt, need(l1["down"]).fmt),
                         ("q8+q4+iq4", "q6", "q8", "q4"))
        self.assertEqual((l2["qkv"].fmt, l2["o"].fmt, need(l2["gate_up"]).fmt, need(l2["down"]).fmt),
                         ("q4k+q4k+q6", "q5k", "q4k", "q5k"))
        self.assertEqual(m.lm.fmt, "q6")
        # Q4_K and Q5_K: values exact, and scales and mins too (6-bit codes times f16 factors).
        for name in ("attn_output", "ffn_down"):
            mat: Mat = l2["o"] if name == "attn_output" else need(l2["down"])
            codes: Ints = mat.q_blocks.transpose(1, 0, 2).reshape(mat.rows, -1)
            np.testing.assert_array_equal(codes, self.k_codes[f"blk.2.{name}.weight"])
        for i, layer in enumerate(m.layers):
            if i == 2:
                # Each part of QKV keeps its format. Q4_K and Q5_K weights are exact; QKV's
                # Q6_K part rounds its scales to f16 (2^-11 relative), which moves a weight by
                # at most 2^-11 |s q|.
                p: str = "blk.2."
                hf_q: Floats = np.empty_like(e[p + "attn_q.weight"])
                hf_q[gguf_order(HEADS)] = e[p + "attn_q.weight"]
                hf_k: Floats = np.empty_like(e[p + "attn_k.weight"])
                hf_k[gguf_order(KV_HEADS)] = e[p + "attn_k.weight"]
                want: dict[str, Floats] = {
                    "qkv": np.concatenate([hf_q, hf_k, e[p + "attn_v.weight"]]),
                    "o": e[p + "attn_output.weight"], "down": e[p + "ffn_down.weight"]}
                gate: Floats = e[p + "ffn_gate.weight"]
                up: Floats = e[p + "ffn_up.weight"]
                want["gate_up"] = np.stack([gate.reshape(-1, 64, D), up.reshape(-1, 64, D)],
                                           1).reshape(-1, D)
                for n in MATS:
                    w: Floats = want[n]
                    ln: object = layer[n]
                    assert isinstance(ln, (Mat, Parts))
                    if ln.fmt in ("q4k", "q5k"):
                        np.testing.assert_array_equal(weights(ln), w, err_msg=f"{p}{n}")
                    else:
                        np.testing.assert_allclose(weights(ln), w,
                                                   atol=2 ** -11 * np.abs(w).max(),
                                                   err_msg=f"{p}{n}")
                continue
            p = f"blk.{i}."
            # Q and K rows back in Hugging Face order; exact for Q4_0, Q8_0, MXFP4 and IQ4_NL (layer 1's
            # QKV in parts of each format; its gate/up re-quantized to Q8_0, see below).
            hf_q = np.empty_like(e[p + "attn_q.weight"])
            hf_q[gguf_order(HEADS)] = e[p + "attn_q.weight"]
            hf_k = np.empty_like(e[p + "attn_k.weight"])
            hf_k[gguf_order(KV_HEADS)] = e[p + "attn_k.weight"]
            want_qkv: Floats = np.concatenate([hf_q, hf_k, e[p + "attn_v.weight"]])
            np.testing.assert_array_equal(weights(layer["qkv"]), want_qkv, err_msg=f"{p}qkv")
            gate = e[p + "ffn_gate.weight"]
            up = e[p + "ffn_up.weight"]
            want_gu: Floats = np.stack([gate.reshape(-1, 64, D), up.reshape(-1, 64, D)],
                                       1).reshape(-1, D)
            got_gu: Floats = weights(need(layer["gate_up"]))
            if LAYER_TYPES[i]["ffn_gate"] == 2:
                np.testing.assert_array_equal(got_gu, want_gu, err_msg=f"{p}gate_up")
            else:  # F16 gate re-quantized to Q8_0
                np.testing.assert_allclose(got_gu, want_gu, atol=np.abs(gate).max() / 100)
            down: Floats = e[p + "ffn_down.weight"]
            # Q4_0 and Q4_1 (layer 0): values, scales and mins exact.
            np.testing.assert_array_equal(weights(need(layer["down"])), down, err_msg=f"{p}down")
        # Q6_K: values exact, scales d * sc rounded to f16.
        got: Floats = weights(m.lm)
        want_lm: Floats = e["token_embd.weight"]
        np.testing.assert_allclose(got, want_lm, rtol=2 ** -11, atol=1e-12)
        np.testing.assert_array_equal(m.embd, want_lm)
        np.testing.assert_array_equal(l1["o"].q_blocks.shape, (HEADS * HEAD_DIM // 16, D, 16))

    def test_tokenizer(self) -> None:
        tok: Tokenizer = Tokenizer(self.gguf)
        ids: dict[str, int] = {t: i for i, t in enumerate(tok.tokens)}
        bos: int = ids["<|begin_of_text|>"]
        eot: int = ids["<|eot_id|>"]
        self.assertEqual(tok.encode("the cat"), [bos, ord("t"), ids["he"], ids["Ġcat"]])
        self.assertEqual(tok.encode(" the<|eot_id|>", bos=False), [ids["Ġthe"], eot])
        self.assertEqual(tok.decode(tok.encode("the cat, the café!")), "the cat, the café!")
        self.assertEqual(tok.decode([bos, ids["Ġcat"], eot], skip_control=False),
                         "<|begin_of_text|> cat<|eot_id|>")

    def test_kernel_matches_numpy(self) -> None:
        for kv_q8 in (False, True):
            with self.subTest(kv_q8=kv_q8):
                self.check_kernel_matches_numpy(kv_q8)

    def check_kernel_matches_numpy(self, kv_q8: bool) -> None:
        m: Model = Model.from_gguf(self.gguf, max_ctx=8, numpy=True, kv_q8=kv_q8)
        with open(os.path.join(ROOT, "examples", "kernels", "llama_decode.air"),
                  encoding="utf-8") as f:
            src: str = f.read()
        for be in achainsaw.available_backends():
            k: achainsaw.Kernel = achainsaw.compile(src, backend=be, fast_math=True)
            cache: npt.NDArray[np.uint16] = m.new_cache()
            h: F32 = np.zeros(m.d, dtype=np.float32)
            np_cache: list[tuple[F32, F32]] = m.numpy_cache()
            token: int = 1
            checked: int = 0
            for pos in range(8):
                got: int = k.run("llama_decode", m.table, m.cfg, cache, token, pos, h,
                                 float(m.eps))
                want: int = m.numpy_step(np_cache, token, pos)
                # The KV cache holds bf16 (or int8) values; NumPy's f32 K and V can round
                # to the neighboring value where they sit at a rounding boundary: one bf16
                # step, or one step of the position's int8 scale. After the first layer, an
                # int8 quantization of an earlier activation that flipped (see below) moves a
                # head's K and V by up to about 1% of their scale too.
                planes: tuple[F32, F32] = m.cache_kv(cache)
                for l, (kc, vc) in enumerate(np_cache):
                    for kv, ref in enumerate((kc[pos], vc[pos])):
                        got_kv: F32 = planes[kv][l, :, pos, :]
                        msg: str = f"{be} cache l={l} pos {pos} kv_q8={kv_q8}"
                        scale_kv: Floats = np.abs(ref).max(axis=1, keepdims=True)
                        flip: Floats = 1e-2 * scale_kv if l > 0 else 0 * scale_kv
                        if kv_q8:
                            step: Floats = scale_kv / 127
                            self.assertTrue(
                                np.all(np.abs(got_kv - ref) <= 1.01 * step + flip), msg)
                        else:
                            self.assertTrue(np.all(
                                np.abs(got_kv - ref) <= 2.0 ** -7 * np.abs(ref) + flip), msg)
                # The next step starts from the kernel's cache, so such a step stays in the
                # position it came from instead of growing through every later one's
                # attention.
                for l, (kc, vc) in enumerate(np_cache):
                    kc[pos] = planes[0][l, :, pos, :]
                    vc[pos] = planes[1][l, :, pos, :]
                # Such a step can flip an int8 quantization of a later attention output,
                # which moves the hidden state by up to about 0.5% of its scale.
                scale: float = float(np.abs(m.last_h).max())
                np.testing.assert_allclose(h, m.last_h, atol=1e-2 * scale,
                                           err_msg=f"{be} pos {pos} kv_q8={kv_q8}")
                top: F32 = np.sort(m.last_logits)[-2:]
                if top[1] - top[0] > 1e-3 * np.abs(m.last_logits).max():
                    self.assertEqual(got, want, f"{be} token at pos {pos}")
                    checked += 1
                token = got
            self.assertGreaterEqual(checked, 4, be)

    def test_prefill_matches_decode(self) -> None:
        # llama_prefill over a prompt leaves exactly the cache and hidden state, and predicts
        # the token, that llama_decode does one token at a time. 13 tokens run a tile of 8, one
        # of 4 and a single token through every layer's formats, with each cache format.
        for kv_q8 in (False, True):
            with self.subTest(kv_q8=kv_q8):
                self.check_prefill_matches_decode(kv_q8)

    def check_prefill_matches_decode(self, kv_q8: bool) -> None:
        m: Model = Model.from_gguf(self.gguf, max_ctx=16, kv_q8=kv_q8)
        with open(os.path.join(ROOT, "examples", "kernels", "llama_decode.air"),
                  encoding="utf-8") as f:
            src: str = f.read()
        prompt: npt.NDArray[np.int64] = np.array([1, 5, 9, 2, 7, 3, 11, 4, 0, 8, 6, 10, 12],
                                                 dtype=np.int64)
        want: int = 0
        for be in achainsaw.available_backends():
            k: achainsaw.Kernel = achainsaw.compile(src, backend=be, fast_math=True)
            cache: npt.NDArray[np.uint16] = m.new_cache()
            h: F32 = np.zeros(m.d, dtype=np.float32)
            for pos, tok in enumerate(prompt):
                want = k.run("llama_decode", m.table, m.cfg, cache, int(tok), pos, h,
                             float(m.eps))
            p_cache: npt.NDArray[np.uint16] = m.new_cache()
            p_h: F32 = np.zeros(m.d, dtype=np.float32)
            got: int = k.run("llama_prefill", m.table, m.cfg, p_cache, prompt, len(prompt), 0, p_h,
                        float(m.eps))
            self.assertEqual(got, want, be)
            np.testing.assert_array_equal(p_h, h, err_msg=be)
            np.testing.assert_array_equal(p_cache, cache, err_msg=be)


class TestQwen3GGUF(TestGGUFDecode):
    """The Llama checks on a tiny qwen3 GGUF; the kernel and prefill tests are inherited."""

    @classmethod
    def setUpClass(cls) -> None:
        rng: np.random.Generator = np.random.default_rng(8)
        tokens, types, merges = tiny_vocab()
        # Merges Qwen's pre-tokenizer never reaches, since it splits digits one by one.
        tokens[-2:] = ["12", "123"]
        types[-2:] = [1, 1]
        merges += ["1 2", "12 3"]
        meta: dict[str, tuple[int, Any]] = {
            "general.architecture": (8, "qwen3"),
            "general.name": (8, "tiny qwen3"),
            "qwen3.block_count": (4, 2),
            "qwen3.embedding_length": (4, D),
            "qwen3.feed_forward_length": (4, FFN),
            "qwen3.attention.head_count": (4, HEADS),
            "qwen3.attention.head_count_kv": (4, KV_HEADS),
            "qwen3.attention.key_length": (4, HEAD_DIM),
            "qwen3.rope.freq_base": (6, 1000000.0),
            "qwen3.attention.layer_norm_rms_epsilon": (6, 1e-6),
            "tokenizer.ggml.model": (8, "gpt2"),
            "tokenizer.ggml.pre": (8, "qwen2"),
            "tokenizer.ggml.tokens": (9, (8, tokens)),
            "tokenizer.ggml.token_type": (9, (5, types)),
            "tokenizer.ggml.merges": (9, (8, merges)),
            "tokenizer.ggml.bos_token_id": (4, tokens.index("<|end_of_text|>")),
            "tokenizer.ggml.add_bos_token": (7, False),
        }
        cls.expected = {}
        cls.k_codes = {}
        tensors: list[tuple[str, list[int], int, U8]] = []

        def add(name: str, t: int, rows: int, k: int, **kw: Any) -> None:
            raw, vals, *_ = ENCODERS[t](rng, rows, k, **kw)
            cls.expected[name] = vals.reshape(rows, k)
            dims: list[int] = [k] if rows == 1 else [k, rows]
            tensors.append((name, dims, t, raw))

        add("token_embd.weight", 8, VOCAB, D)                  # tied: also the LM head
        add("output_norm.weight", 0, 1, D)
        for i, types_ in enumerate([{n: 8 for n in SHAPES}, LAYER_TYPES[2]]):
            add(f"blk.{i}.attn_norm.weight", 0, 1, D)
            add(f"blk.{i}.ffn_norm.weight", 0, 1, D)
            # Head norms scale q and k to unit RMS: weights below 1 keep the scores (and so
            # the softmax's sensitivity to an int8 step of the kernel's queries) moderate.
            for n in ("q", "k"):
                add(f"blk.{i}.attn_{n}_norm.weight", 0, 1, HEAD_DIM,
                    values=rng.uniform(0.25, 0.5, (1, HEAD_DIM)))
            for name, t in types_.items():
                rows, k = SHAPES[name]
                add(f"blk.{i}.{name}.weight", t, rows, k)
        cls.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.path = os.path.join(cls.tmp.name, "qwen3.gguf")
        write_gguf(cls.path, meta, tensors)
        cls.gguf = GGUF(cls.path)

    def test_loader_conversions(self) -> None:
        m: Model = Model.from_gguf(self.gguf, max_ctx=8, numpy=True)
        e: dict[str, Floats] = self.expected
        l0: Layer = m.layers[0]
        # Q and K rows as the file has them (Qwen3's NeoX RoPE already uses the kernel's
        # half-split order); Q8_0 loads exactly.
        qkv: Mat | Parts = l0["qkv"]
        got: Floats = np.concatenate([
            (p.q_blocks * p.s_blocks[:, :, None]).transpose(1, 0, 2).reshape(p.rows, -1)
            for p in qkv.parts()])
        np.testing.assert_array_equal(got, np.concatenate(
            [e["blk.0.attn_q.weight"], e["blk.0.attn_k.weight"], e["blk.0.attn_v.weight"]]))
        for i, layer in enumerate(m.layers):
            for n in ("q_norm", "k_norm"):
                w: F32 | None = layer["q_norm"] if n == "q_norm" else layer["k_norm"]
                assert w is not None
                np.testing.assert_array_equal(w, e[f"blk.{i}.attn_{n}.weight"][0], err_msg=n)
        np.testing.assert_array_equal(m.embd, e["token_embd.weight"])
        self.assertEqual(m.lm.fmt, "q8")
        self.assertEqual(float(m.eps), float(np.float32(1e-6)))

    def test_tokenizer(self) -> None:
        tok: Tokenizer = Tokenizer(self.gguf)
        ids: dict[str, int] = {t: i for i, t in enumerate(tok.tokens)}
        # No BOS (add_bos_token false), and digits one at a time: "123" never merges.
        self.assertEqual(tok.encode("the 123"),
                         [ord("t"), ids["he"], ids["Ġ"], ord("1"), ord("2"), ord("3")])
        self.assertEqual(tok.decode(tok.encode("the cat 42!")), "the cat 42!")


# The tiny Qwen3-MoE: experts per layer, experts used per token, an expert's MLP width.
EXPERTS: int = 64
EXPERTS_USED: int = 4
EXPERT_FFN: int = 256


class TestQwen3MoeGGUF(TestQwen3GGUF):
    """A tiny qwen3moe GGUF: Qwen3's attention, each MLP a mixture of experts stored as 3-D
    tensors (Q8_0 in layer 0; Q4_K gate/up and Q6_K down in layer 1) under an F32 router.
    The kernel, prefill and tokenizer tests are inherited."""

    @classmethod
    def setUpClass(cls) -> None:
        rng: np.random.Generator = np.random.default_rng(13)
        tokens, types, merges = tiny_vocab()
        tokens[-2:] = ["12", "123"]
        types[-2:] = [1, 1]
        merges += ["1 2", "12 3"]
        meta: dict[str, tuple[int, Any]] = {
            "general.architecture": (8, "qwen3moe"),
            "general.name": (8, "tiny qwen3moe"),
            "qwen3moe.block_count": (4, 2),
            "qwen3moe.embedding_length": (4, D),
            "qwen3moe.feed_forward_length": (4, EXPERT_FFN * EXPERTS_USED),
            "qwen3moe.expert_feed_forward_length": (4, EXPERT_FFN),
            "qwen3moe.expert_count": (4, EXPERTS),
            "qwen3moe.expert_used_count": (4, EXPERTS_USED),
            "qwen3moe.attention.head_count": (4, HEADS),
            "qwen3moe.attention.head_count_kv": (4, KV_HEADS),
            "qwen3moe.attention.key_length": (4, HEAD_DIM),
            "qwen3moe.rope.freq_base": (6, 1000000.0),
            "qwen3moe.attention.layer_norm_rms_epsilon": (6, 1e-6),
            "tokenizer.ggml.model": (8, "gpt2"),
            "tokenizer.ggml.pre": (8, "qwen2"),
            "tokenizer.ggml.tokens": (9, (8, tokens)),
            "tokenizer.ggml.token_type": (9, (5, types)),
            "tokenizer.ggml.merges": (9, (8, merges)),
            "tokenizer.ggml.bos_token_id": (4, tokens.index("<|end_of_text|>")),
            "tokenizer.ggml.add_bos_token": (7, False),
        }
        cls.expected = {}
        cls.k_codes = {}
        tensors: list[tuple[str, list[int], int, U8]] = []

        def add(name: str, t: int, rows: int, k: int, dims: list[int] | None = None,
                **kw: Any) -> None:
            raw, vals, *_ = ENCODERS[t](rng, rows, k, **kw)
            cls.expected[name] = vals.reshape(rows, k)
            tensors.append((name, dims or ([k] if rows == 1 else [k, rows]), t, raw))

        add("token_embd.weight", 8, VOCAB, D)
        add("output_norm.weight", 0, 1, D)
        for i, (gu_t, dn_t) in enumerate([(8, 8), (12, 14)]):
            p: str = f"blk.{i}."
            add(p + "attn_norm.weight", 0, 1, D)
            add(p + "ffn_norm.weight", 0, 1, D)
            for n in ("q", "k"):
                add(p + f"attn_{n}_norm.weight", 0, 1, HEAD_DIM,
                    values=rng.uniform(0.25, 0.5, (1, HEAD_DIM)))
            for name in ("attn_q", "attn_k", "attn_v", "attn_output"):
                rows, k = SHAPES[name]
                add(f"{p}{name}.weight", 8, rows, k)
            # The router's logits a few units apart, as a trained model's are.
            add(p + "ffn_gate_inp.weight", 0, EXPERTS, D,
                values=rng.standard_normal((EXPERTS, D)) * 4 / np.sqrt(D))
            for name, t in (("gate", gu_t), ("up", gu_t)):
                add(f"{p}ffn_{name}_exps.weight", t, EXPERTS * EXPERT_FFN, D,
                    dims=[D, EXPERT_FFN, EXPERTS])
            add(p + "ffn_down_exps.weight", dn_t, EXPERTS * D, EXPERT_FFN,
                dims=[EXPERT_FFN, D, EXPERTS])
        cls.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.path = os.path.join(cls.tmp.name, "qwen3moe.gguf")
        write_gguf(cls.path, meta, tensors)
        cls.gguf = GGUF(cls.path)

    def test_loader_conversions(self) -> None:
        m: Model = Model.from_gguf(self.gguf, max_ctx=8, numpy=True)
        e: dict[str, Floats] = self.expected
        self.assertEqual(m.moe, (EXPERTS, EXPERTS_USED, 1))
        self.assertEqual(list(m.cfg[9:]), [EXPERTS, EXPERTS_USED, EXPERT_FFN, 1, 0])
        for i, layer in enumerate(m.layers):
            p: str = f"blk.{i}."
            self.assertIsNone(layer["gate_up"])
            router: Mat | None = layer["router"]
            experts: list[tuple[Mat, Mat]] | None = layer["experts"]
            assert router is not None and experts is not None
            self.assertEqual((router.fmt, router.rows), ("q8", EXPERTS))
            self.assertEqual(len(experts), EXPERTS)
            gates: Floats = e[p + "ffn_gate_exps.weight"]
            ups: Floats = e[p + "ffn_up_exps.weight"]
            downs: Floats = e[p + "ffn_down_exps.weight"]
            for x in (0, 17, EXPERTS - 1):
                gu, dn = experts[x]
                rows: slice = slice(x * EXPERT_FFN, (x + 1) * EXPERT_FFN)
                # Expert x's rows of the 3-D tensors, gate/up chunks interleaved as the dense
                # MLP's; exact in each format (Q6_K values, its scales to f16).
                want_gu: Floats = np.stack([gates[rows].reshape(-1, 64, D),
                                            ups[rows].reshape(-1, 64, D)], 1).reshape(-1, D)
                want_dn: Floats = downs[x * D:(x + 1) * D]
                for got_m, want in ((gu, want_gu), (dn, want_dn)):
                    v: Floats = got_m.q_blocks * got_m.s_blocks[:, :, None]
                    if got_m.m_blocks is not None:
                        v = v - got_m.m_blocks[:, :, None]
                    got: Floats = v.transpose(1, 0, 2).reshape(got_m.rows, -1)
                    np.testing.assert_allclose(got, want, rtol=2 ** -11,
                                               atol=2 ** -11 * np.abs(want).max(),
                                               err_msg=f"{p} expert {x} {got_m.fmt}")
        self.assertEqual(m.formats["experts"], ["q4k", "q6", "q8"])

    def test_experts_not_a_multiple_of_64(self) -> None:
        # 32 experts (gpt-oss has 32, Mixtral 8): the router is padded to 64 zero rows, whose
        # logits the kernel computes but never picks.
        m: Model = Model.random(d=128, layers=2, heads=4, kv_heads=2, head_dim=64, ffn=128,
                                vocab=256, max_ctx=8, numpy=True, qk_norm=True,
                                moe=(32, 4, 1))
        router: Mat | None = m.layers[0]["router"]
        assert router is not None
        self.assertEqual(router.rows, 64)
        with open(os.path.join(ROOT, "examples", "kernels", "llama_decode.air"),
                  encoding="utf-8") as f:
            src: str = f.read()
        for be in achainsaw.available_backends():
            k: achainsaw.Kernel = achainsaw.compile(src, backend=be, fast_math=True)
            cache: npt.NDArray[np.uint16] = m.new_cache()
            h: F32 = np.zeros(m.d, dtype=np.float32)
            np_cache: list[tuple[F32, F32]] = m.numpy_cache()
            token: int = 3
            for pos in range(6):
                got: int = k.run("llama_decode", m.table, m.cfg, cache, token, pos, h,
                                 float(m.eps))
                m.numpy_step(np_cache, token, pos)
                np.testing.assert_allclose(h, m.last_h, atol=1e-2 * np.abs(m.last_h).max(),
                                           err_msg=f"{be} pos {pos}")
                planes: tuple[F32, F32] = m.cache_kv(cache)
                for l, (kc, vc) in enumerate(np_cache):
                    kc[pos] = planes[0][l, :, pos, :]
                    vc[pos] = planes[1][l, :, pos, :]
                token = got



# The tiny gpt-oss: experts, experts used per token, an expert's MLP width, and the window of
# the even layers.
OSS_EXPERTS: int = 32
OSS_USED: int = 4
OSS_FFN: int = 128
OSS_WINDOW: int = 4


def gpt_oss_meta(tokens: list[str], types: list[int], merges: list[str]) -> dict[str, tuple[int, Any]]:
    """A tiny gpt-oss GGUF's metadata: gpt-oss's attention (YaRN over 4096 positions, a window
    in the even layers) and router, at the test's shapes, with the GPT-4o pre-tokenizer."""
    return {
        "general.architecture": (8, "gpt-oss"),
        "general.name": (8, "tiny gpt-oss"),
        "gpt-oss.block_count": (4, 2),
        "gpt-oss.context_length": (4, 131072),
        "gpt-oss.embedding_length": (4, D),
        "gpt-oss.feed_forward_length": (4, OSS_FFN),
        "gpt-oss.expert_feed_forward_length": (4, OSS_FFN),
        "gpt-oss.expert_count": (4, OSS_EXPERTS),
        "gpt-oss.expert_used_count": (4, OSS_USED),
        "gpt-oss.attention.head_count": (4, HEADS),
        "gpt-oss.attention.head_count_kv": (4, KV_HEADS),
        "gpt-oss.attention.key_length": (4, HEAD_DIM),
        "gpt-oss.attention.value_length": (4, HEAD_DIM),
        "gpt-oss.attention.sliding_window": (4, OSS_WINDOW),
        "gpt-oss.attention.layer_norm_rms_epsilon": (6, 1e-5),
        "gpt-oss.rope.freq_base": (6, 150000.0),
        "gpt-oss.rope.scaling.type": (8, "yarn"),
        "gpt-oss.rope.scaling.factor": (6, 32.0),
        "gpt-oss.rope.scaling.original_context_length": (4, 4096),
        "tokenizer.ggml.model": (8, "gpt2"),
        "tokenizer.ggml.pre": (8, "gpt-4o"),
        "tokenizer.ggml.tokens": (9, (8, tokens)),
        "tokenizer.ggml.token_type": (9, (5, types)),
        "tokenizer.ggml.merges": (9, (8, merges)),
        "tokenizer.ggml.bos_token_id": (4, tokens.index("<|begin_of_text|>")),
        "tokenizer.ggml.eos_token_id": (4, tokens.index("<|end_of_text|>")),
        "tokenizer.ggml.add_bos_token": (7, False),
    }


def gpt_oss_tensors(rng: np.random.Generator, expected: dict[str, Floats],
                    expert_types: list[int]) -> list[tuple[str, list[int], int, U8]]:
    """A tiny gpt-oss GGUF's tensors (their values into `expected`): Q8_0 attention with f32
    biases and sinks, an f32 router and its bias, and per layer 3-D expert tensors of the
    given type (MXFP4 in gpt-oss) with f32 biases, and an untied Q8_0 LM head."""
    tensors: list[tuple[str, list[int], int, U8]] = []

    def add(name: str, t: int, rows: int, k: int, dims: list[int] | None = None,
            **kw: Any) -> None:
        raw, vals, *_ = ENCODERS[t](rng, rows, k, **kw)
        expected[name] = vals.reshape(rows, k)
        tensors.append((name, dims or ([k] if rows == 1 else [k, rows]), t, raw))

    def uniform(lo: float, hi: float, *shape: int) -> Floats:
        v: Floats = rng.uniform(lo, hi, shape)
        return v

    add("token_embd.weight", 8, VOCAB, D)
    add("output.weight", 8, VOCAB, D)
    add("output_norm.weight", 0, 1, D)
    for i, et in enumerate(expert_types):
        p: str = f"blk.{i}."
        add(p + "attn_norm.weight", 0, 1, D)
        add(p + "post_attention_norm.weight", 0, 1, D)
        for name in ("attn_q", "attn_k", "attn_v", "attn_output"):
            rows, k = SHAPES[name]
            add(f"{p}{name}.weight", 8, rows, k)
            add(f"{p}{name}.bias", 0, 1, rows, values=uniform(-0.3, 0.3, 1, rows))
        # Sinks around the scores' own range, so they take a share of every softmax.
        add(p + "attn_sinks.weight", 0, 1, HEADS, values=uniform(-1.0, 2.0, 1, HEADS))
        # The router's logits a few units apart, as a trained model's are.
        add(p + "ffn_gate_inp.weight", 0, OSS_EXPERTS, D,
            values=rng.standard_normal((OSS_EXPERTS, D)) * 4 / np.sqrt(D))
        add(p + "ffn_gate_inp.bias", 0, 1, OSS_EXPERTS, values=uniform(-1.0, 1.0, 1, OSS_EXPERTS))
        for name in ("gate", "up"):
            add(f"{p}ffn_{name}_exps.weight", et, OSS_EXPERTS * OSS_FFN, D,
                dims=[D, OSS_FFN, OSS_EXPERTS])
            add(f"{p}ffn_{name}_exps.bias", 0, OSS_EXPERTS, OSS_FFN,
                values=uniform(-0.5, 0.5, OSS_EXPERTS, OSS_FFN))
        add(p + "ffn_down_exps.weight", et, OSS_EXPERTS * D, OSS_FFN,
            dims=[OSS_FFN, D, OSS_EXPERTS])
        add(p + "ffn_down_exps.bias", 0, OSS_EXPERTS, D,
            values=uniform(-0.2, 0.2, OSS_EXPERTS, D))
    return tensors


class TestGptOssGGUF(TestGGUFDecode):
    """A tiny gpt-oss GGUF: attention with QKV and O biases, sinks, YaRN and a window of
    OSS_WINDOW positions in layer 0, then a mixture of 32 MXFP4 experts with biases under a
    biased f32 router (a softmax of the picked logits) and the clamped SwiGLU. The kernel
    (8 positions) and prefill (13 tokens) tests are inherited, so both run past the window."""

    @classmethod
    def setUpClass(cls) -> None:
        rng: np.random.Generator = np.random.default_rng(21)
        tokens, types, merges = tiny_vocab()
        cls.expected = {}
        cls.k_codes = {}
        tensors: list[tuple[str, list[int], int, U8]] = gpt_oss_tensors(
            rng, cls.expected, [39, 39])
        cls.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.path = os.path.join(cls.tmp.name, "gpt-oss.gguf")
        write_gguf(cls.path, gpt_oss_meta(tokens, types, merges), tensors)
        cls.gguf = GGUF(cls.path)

    def test_loader_conversions(self) -> None:
        m: Model = Model.from_gguf(self.gguf, max_ctx=8, numpy=True)
        e: dict[str, Floats] = self.expected
        self.assertEqual(m.moe, (OSS_EXPERTS, OSS_USED, 2))
        self.assertEqual(list(m.cfg[9:]), [OSS_EXPERTS, OSS_USED, OSS_FFN, 2, 1])
        self.assertEqual([l["window"] for l in m.layers], [OSS_WINDOW, 0])
        self.assertEqual(m.formats["experts"], ["mx4"])
        self.assertEqual(m.lm.fmt, "q8")
        gu_rows: npt.NDArray[np.intp] = np.arange(2 * OSS_FFN).reshape(
            2, OSS_FFN // 64, 64).transpose(1, 0, 2).reshape(-1)
        for i, layer in enumerate(m.layers):
            p: str = f"blk.{i}."
            # Biases and sinks as written: QKV's rows in the file's order (NeoX RoPE), each
            # expert's gate/up bias chunk-interleaved as its rows.
            qkv_b: F32 | None = layer["qkv_b"]
            o_b: F32 | None = layer["o_b"]
            sinks: F32 | None = layer["sinks"]
            router_b: F32 | None = layer["router_b"]
            expert_b: list[tuple[F32, F32]] | None = layer["expert_b"]
            assert qkv_b is not None and o_b is not None and sinks is not None
            assert router_b is not None and expert_b is not None
            np.testing.assert_array_equal(qkv_b, np.concatenate(
                [e[f"{p}attn_{n}.bias"][0] for n in ("q", "k", "v")]))
            np.testing.assert_array_equal(o_b, e[p + "attn_output.bias"][0])
            np.testing.assert_array_equal(sinks, e[p + "attn_sinks.weight"][0])
            np.testing.assert_array_equal(router_b, e[p + "ffn_gate_inp.bias"][0])
            np.testing.assert_array_equal(layer["ffn_norm"], e[p + "post_attention_norm.weight"][0])
            for x in (0, OSS_EXPERTS - 1):
                gub, dnb = expert_b[x]
                np.testing.assert_array_equal(gub, np.concatenate(
                    [e[p + "ffn_gate_exps.bias"][x], e[p + "ffn_up_exps.bias"][x]])[gu_rows])
                np.testing.assert_array_equal(dnb, e[p + "ffn_down_exps.bias"][x])
            experts: list[tuple[Mat, Mat]] | None = layer["experts"]
            assert experts is not None
            # MXFP4 bit for bit: values and power-of-two scales.
            dn: Mat = experts[7][1]
            got: Floats = (dn.q_blocks * dn.s_blocks[:, :, None]).transpose(1, 0, 2).reshape(
                dn.rows, -1)
            np.testing.assert_array_equal(got, e[p + "ffn_down_exps.weight"][7 * D:8 * D])
        # YaRN: cos and sin scaled by 1 + 0.1 ln 32 (so cos at position 0), and the fastest
        # dimension's angle unscaled (extrapolated), the slowest's divided by 32.
        mscale: float = 1 + 0.1 * np.log(32.0)
        np.testing.assert_allclose(m.cos[0], mscale, rtol=1e-6)
        ang: F32 = np.arctan2(m.sin[3], m.cos[3])
        self.assertAlmostEqual(float(ang[0]), 3.0, places=5)
        theta_last: float = 3.0 * 150000.0 ** (-2.0 * (HEAD_DIM // 2 - 1) / HEAD_DIM)
        self.assertAlmostEqual(float(ang[-1]), theta_last / 32, places=7)

    def test_tokenizer(self) -> None:
        tok: Tokenizer = Tokenizer(self.gguf)
        # GPT-4o's pre-tokenizer: words split at a lower-to-upper case change and keep their
        # contraction (or a punctuation mark before them), digits in threes, punctuation
        # with a trailing slash or newline.
        self.assertEqual(tok.split.findall("HelloWorld don't 12345 a/b x!!/\nCAPSlock"),
                         ["Hello", "World", " don't", " ", "123", "45", " a", "/b", " x",
                          "!!/\n", "CAPSlock"])
        ids: dict[str, int] = {t: i for i, t in enumerate(tok.tokens)}
        self.assertEqual(tok.encode("the cat"), [ord("t"), ids["he"], ids["Ġcat"]])
        self.assertEqual(tok.decode(tok.encode("the cat 42!")), "the cat 42!")

class TestMxfp4(unittest.TestCase):
    def test_scales_outside_f16_become_q8(self) -> None:
        # Power-of-two scales from 2^-24 to 2^15 are exact f16s, so MXFP4 stays MXFP4; a
        # matrix with any other scale (none in gpt-oss) is dequantized and quantized to Q8_0.
        rng: np.random.Generator = np.random.default_rng(5)
        enc: Encoded = enc_mxfp4(rng, 64, 128)
        raw: U8 = enc[0]
        vals: Floats = enc[1]
        mat: Mat = Mat.from_tensor(Tensor("w", [128, 64], 39, raw))
        self.assertEqual(mat.fmt, "mx4")
        np.testing.assert_array_equal(mat.dequantize(), vals.reshape(64, 128))
        for e in (103, 144):
            out: U8 = raw.copy()
            out[5, 17] = e  # row 5's second block
            want: F32 = Tensor("w", [128, 64], 39, out).dequantize()
            mat = Mat.from_tensor(Tensor("w", [128, 64], 39, out))
            self.assertEqual(mat.fmt, "q8")
            # Q8_0's rounding: at most half a step of its block's scale (the block's max / 127,
            # at most the row's), or the whole value where that scale underflows f16 (the
            # e = 103 block's values, below 2^-21).
            err: Floats = np.abs(mat.dequantize() - want)
            self.assertTrue(np.all(err <= np.abs(want).max(axis=1, keepdims=True) / 200), e)
        # A block of zeros, which llama.cpp writes with e = 0 (any scale fits), stays MXFP4.
        out = raw.copy()
        out[5, 17:34] = 0
        out[6, 17] = 0
        out[6, 18:34] = 0x88  # code 8, -0
        mat = Mat.from_tensor(Tensor("w", [128, 64], 39, out))
        self.assertEqual(mat.fmt, "mx4")
        np.testing.assert_array_equal(mat.dequantize(),
                                      Tensor("w", [128, 64], 39, out).dequantize())


if __name__ == "__main__":
    unittest.main()
