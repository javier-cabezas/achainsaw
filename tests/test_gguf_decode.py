"""
End-to-end test of the GGUF path: writes a tiny Llama GGUF with every tensor type the loader
handles (Q4_0, Q8_0, Q6_K, Q4_K and Q5_K natively; Q4_1 and F16 re-quantized to Q8_0),
loads it with
benchmarks/gguf.py and benchmarks/llama_model.py, and checks
  - the dequantizers against the values the file was written from,
  - the loader's conversions (bit for bit where promised; Q and K rows back in Hugging Face
    order; fused matrices, including a QKV matrix whose parts keep different formats),
  - the BPE tokenizer on a small vocabulary,
  - examples/kernels/llama_decode.air against the NumPy forward pass of the loaded model, on
    every available backend.
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
from gguf import F32, GGUF, U8, Floats, Ints, Tokenizer, _bytes_to_unicode  # noqa: E402
from llama_model import MATS, Mat, Model, Parts  # noqa: E402

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


ENCODERS: dict[int, Encoder] = {2: enc_q4_0, 8: enc_q8_0, 3: enc_q4_1, 14: enc_q6_k,
                                12: enc_k(4), 13: enc_k(5), 1: enc_f16, 0: enc_f32}

# Per layer, the type of each weight: layer 0 is all Q4_0 but a Q4_1 down projection;
# layer 1 mixes formats inside the fused QKV and gate/up matrices; layer 2 is laid out like
# a Q4_K_M file's (Q and K Q4_K, V Q6_K), with Q5_K O and down projections.
LAYER_TYPES: list[dict[str, int]] = [
    dict(attn_q=2, attn_k=2, attn_v=2, attn_output=2, ffn_gate=2, ffn_up=2, ffn_down=3),
    dict(attn_q=8, attn_k=2, attn_v=2, attn_output=14, ffn_gate=1, ffn_up=2, ffn_down=2),
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
        return struct.pack({4: "<I", 5: "<i", 6: "<f"}[t], v)

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
        self.assertEqual((l0["qkv"].fmt, l0["o"].fmt, l0["gate_up"].fmt, l0["down"].fmt),
                         ("q4", "q4", "q4", "q8"))
        self.assertEqual((l1["qkv"].fmt, l1["o"].fmt, l1["gate_up"].fmt, l1["down"].fmt),
                         ("q8+q4+q4", "q6", "q8", "q4"))
        self.assertEqual((l2["qkv"].fmt, l2["o"].fmt, l2["gate_up"].fmt, l2["down"].fmt),
                         ("q4k+q4k+q6", "q5k", "q4k", "q5k"))
        self.assertEqual(m.lm.fmt, "q6")
        # Q4_K and Q5_K: values exact, and scales and mins too (6-bit codes times f16 factors).
        for name in ("attn_output", "ffn_down"):
            mat: Mat = l2["o"] if name == "attn_output" else l2["down"]
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
                    if layer[n].fmt in ("q4k", "q5k"):
                        np.testing.assert_array_equal(weights(layer[n]), w, err_msg=f"{p}{n}")
                    else:
                        np.testing.assert_allclose(weights(layer[n]), w,
                                                   atol=2 ** -11 * np.abs(w).max(),
                                                   err_msg=f"{p}{n}")
                continue
            p = f"blk.{i}."
            # Q and K rows back in Hugging Face order; exact for Q4_0 and Q8_0 (layer 1's
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
            got_gu: Floats = weights(layer["gate_up"])
            if LAYER_TYPES[i]["ffn_gate"] == 2:
                np.testing.assert_array_equal(got_gu, want_gu, err_msg=f"{p}gate_up")
            else:  # F16 gate re-quantized to Q8_0
                np.testing.assert_allclose(got_gu, want_gu, atol=np.abs(gate).max() / 100)
            down: Floats = e[p + "ffn_down.weight"]
            if LAYER_TYPES[i]["ffn_down"] == 2:
                np.testing.assert_array_equal(weights(layer["down"]), down)
            else:  # Q4_1 re-quantized to Q8_0: half a step, plus the scale's f16 rounding
                step: Floats = np.abs(down).reshape(D, -1, 32).max(axis=2) / 127
                err: Floats = np.abs(weights(layer["down"]) - down).reshape(D, -1, 32).max(axis=2)
                self.assertTrue(np.all(err <= step * (0.5 + 127 * 2 ** -11) + 1e-12))
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
        m: Model = Model.from_gguf(self.gguf, max_ctx=8, numpy=True)
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
                # The KV cache holds bf16 values; NumPy's f32 K and V can round to the
                # neighboring bf16 value where they sit at a rounding boundary.
                planes: tuple[npt.NDArray[np.uint16], ...] = m.cache_kv(cache)
                for l, (kc, vc) in enumerate(np_cache):
                    for kv, ref in enumerate((kc[pos], vc[pos])):
                        bits: Ints = planes[kv][l, :, pos, :].astype(np.uint32) << 16
                        np.testing.assert_allclose(bits.view(np.float32), ref, rtol=2.0 ** -7,
                                                   err_msg=f"{be} cache l={l} pos {pos}")
                # Such a step can flip an int8 quantization of a later attention output,
                # which moves the hidden state by up to about 0.5% of its scale.
                scale: float = float(np.abs(m.last_h).max())
                np.testing.assert_allclose(h, m.last_h, atol=1e-2 * scale,
                                           err_msg=f"{be} pos {pos}")
                top: F32 = np.sort(m.last_logits)[-2:]
                if top[1] - top[0] > 1e-3 * np.abs(m.last_logits).max():
                    self.assertEqual(got, want, f"{be} token at pos {pos}")
                    checked += 1
                token = got
            self.assertGreaterEqual(checked, 4, be)

    def test_prefill_matches_decode(self) -> None:
        # llama_prefill over a prompt leaves exactly the cache and hidden state, and predicts
        # the token, that llama_decode does one token at a time. 13 tokens run a tile of 8, one
        # of 4 and a single token through every layer's formats.
        m: Model = Model.from_gguf(self.gguf, max_ctx=16)
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


if __name__ == "__main__":
    unittest.main()
