"""
End-to-end test of the GGUF path: writes a tiny Llama GGUF with every tensor type the loader
handles (Q4_0, Q8_0, Q6_K natively; Q4_1 and F16 re-quantized to Q8_0), loads it with
benchmarks/gguf.py and benchmarks/llama_model.py, and checks
  - the dequantizers against the values the file was written from,
  - the loader's conversions (bit for bit where promised; Q and K rows back in Hugging Face
    order; fused matrices),
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

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
sys.path.insert(0, ROOT)
import achainsaw  # noqa: E402
from gguf import GGUF, Tokenizer, _bytes_to_unicode  # noqa: E402
from llama_model import Model  # noqa: E402

D, LAYERS, HEADS, KV_HEADS, HEAD_DIM, FFN, VOCAB = 256, 2, 4, 2, 64, 512, 384
F16 = np.float16


# Each encoder takes integer codes and scales and returns (block bytes [rows x row bytes],
# the f32 values llama.cpp's dequantize_row_* gives for them).
def enc_q4_0(rng, rows, k):
    q = rng.integers(0, 16, size=(rows, k // 32, 32), dtype=np.uint8)
    d = (rng.uniform(0.5, 1.0, size=(rows, k // 32)) * rng.choice([-1, 1], size=(rows, k // 32))
         / (4.6 * np.sqrt(k))).astype(F16)
    qs = q[:, :, :16] | (q[:, :, 16:] << 4)
    raw = np.concatenate([d[:, :, None].view(np.uint8), qs], axis=2)
    return raw.reshape(rows, -1), ((q.astype(np.float32) - 8) * d[:, :, None].astype(np.float32))


def enc_q8_0(rng, rows, k):
    q = rng.integers(-127, 128, size=(rows, k // 32, 32), dtype=np.int8)
    d = (rng.uniform(0.5, 1.0, size=(rows, k // 32)) / (73.0 * np.sqrt(k))).astype(F16)
    raw = np.concatenate([d[:, :, None].view(np.uint8), q.view(np.uint8)], axis=2)
    return raw.reshape(rows, -1), q.astype(np.float32) * d[:, :, None].astype(np.float32)


def enc_q4_1(rng, rows, k):
    q = rng.integers(0, 16, size=(rows, k // 32, 32), dtype=np.uint8)
    d = (rng.uniform(0.5, 1.0, size=(rows, k // 32)) / (4.6 * np.sqrt(k))).astype(F16)
    m = (-8 * d.astype(np.float32)).astype(F16)
    qs = q[:, :, :16] | (q[:, :, 16:] << 4)
    raw = np.concatenate([d[:, :, None].view(np.uint8), m[:, :, None].view(np.uint8), qs], axis=2)
    vals = q.astype(np.float32) * d[:, :, None].astype(np.float32) + m[:, :, None].astype(np.float32)
    return raw.reshape(rows, -1), vals


def enc_q6_k(rng, rows, k):
    n = rows * k // 256
    q = rng.integers(0, 64, size=(n, 256), dtype=np.uint8)
    sc = rng.integers(-127, 128, size=(n, 16), dtype=np.int8)
    d = (rng.uniform(0.5, 1.0, size=n) / (18.5 * 64 * np.sqrt(k))).astype(F16)
    ql = np.zeros((n, 128), np.uint8)
    qh = np.zeros((n, 64), np.uint8)
    for h in range(2):
        q1, q2, q3, q4 = (q[:, 128 * h + 32 * p:128 * h + 32 * p + 32] for p in range(4))
        ql[:, 64 * h:64 * h + 32] = (q1 & 15) | ((q3 & 15) << 4)
        ql[:, 64 * h + 32:64 * h + 64] = (q2 & 15) | ((q4 & 15) << 4)
        qh[:, 32 * h:32 * h + 32] = (q1 >> 4) | ((q2 >> 4) << 2) | ((q3 >> 4) << 4) | ((q4 >> 4) << 6)
    raw = np.concatenate([ql, qh, sc.view(np.uint8), d[:, None].view(np.uint8)], axis=1)
    ds = d.astype(np.float32)[:, None] * sc.astype(np.float32)          # [n x 16]
    vals = np.repeat(ds, 16, axis=1) * (q.astype(np.float32) - 32)
    return raw.reshape(rows, -1), vals.reshape(rows, k)


def enc_f16(rng, rows, k):
    v = (rng.standard_normal((rows, k)) / np.sqrt(k)).astype(F16)
    return v.view(np.uint8).reshape(rows, -1), v.astype(np.float32)


def enc_f32(rng, rows, k, values=None):
    v = (rng.uniform(0.8, 1.2, (rows, k)) if values is None else values).astype(np.float32)
    return v.view(np.uint8).reshape(rows, -1), v


ENCODERS = {2: enc_q4_0, 8: enc_q8_0, 3: enc_q4_1, 14: enc_q6_k, 1: enc_f16, 0: enc_f32}

# Per layer, the type of each weight: layer 0 is all Q4_0 but a Q4_1 down projection;
# layer 1 mixes formats inside the fused QKV and gate/up matrices.
LAYER_TYPES = [
    dict(attn_q=2, attn_k=2, attn_v=2, attn_output=2, ffn_gate=2, ffn_up=2, ffn_down=3),
    dict(attn_q=8, attn_k=2, attn_v=2, attn_output=14, ffn_gate=1, ffn_up=2, ffn_down=2),
]
SHAPES = dict(attn_q=(HEADS * HEAD_DIM, D), attn_k=(KV_HEADS * HEAD_DIM, D),
              attn_v=(KV_HEADS * HEAD_DIM, D), attn_output=(D, HEADS * HEAD_DIM),
              ffn_gate=(FFN, D), ffn_up=(FFN, D), ffn_down=(D, FFN))


def gguf_order(n_heads):
    """Row order llama.cpp's converter gives Q and K (HF row for each GGUF row)."""
    return np.arange(n_heads * HEAD_DIM).reshape(n_heads, 2, HEAD_DIM // 2).transpose(0, 2, 1).reshape(-1)


def tiny_vocab():
    byte_chars = _bytes_to_unicode()
    tokens = [byte_chars[b] for b in range(256)]
    merges = ["h e", "Ġ t", "Ġt he", "c a", "Ġ ca", "Ġca t"]
    tokens += ["he", "Ġt", "Ġthe", "ca", "Ġca", "Ġcat"]
    types = [1] * len(tokens)
    for name in ["<|begin_of_text|>", "<|end_of_text|>", "<|eot_id|>"]:
        tokens.append(name)
        types.append(3)
    while len(tokens) < VOCAB:
        tokens.append(f"<|reserved_special_token_{len(tokens)}|>")
        types.append(3)
    return tokens, types, merges


def write_gguf(path, meta, tensors):
    """meta: {key: (gguf type, value)}; tensors: [(name, dims innermost first, type, bytes)]."""
    def s(x):
        b = x.encode("utf-8")
        return struct.pack("<Q", len(b)) + b

    def value(t, v):
        if t == 8:
            return s(v)
        if t == 9:
            et, items = v
            return struct.pack("<IQ", et, len(items)) + b"".join(value(et, i) for i in items)
        return struct.pack({4: "<I", 5: "<i", 6: "<f"}[t], v)

    out = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(meta)))
    for k, (t, v) in meta.items():
        out += s(k) + struct.pack("<I", t) + value(t, v)
    offset, blobs = 0, []
    for name, dims, t, data in tensors:
        out += s(name) + struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}Q", *dims)
        out += struct.pack("<IQ", t, offset)
        blob = data.tobytes()
        blob += b"\0" * (-len(blob) % 32)
        blobs.append(blob)
        offset += len(blob)
    out += b"\0" * (-len(out) % 32)
    with open(path, "wb") as f:
        f.write(out)
        for b in blobs:
            f.write(b)


class TestGGUFDecode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(5)
        tokens, types, merges = tiny_vocab()
        meta = {
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
        tensors = []

        def add(name, t, rows, k, **kw):
            raw, vals = ENCODERS[t](rng, rows, k, **kw)
            cls.expected[name] = vals.reshape(rows, k)
            dims = [k] if rows == 1 else [k, rows]
            tensors.append((name, dims, t, raw))

        add("token_embd.weight", 14, VOCAB, D)                 # tied: also the LM head
        add("output_norm.weight", 0, 1, D)
        ff = np.array([1.0] * 16 + [2.0, 4.0] * 8, np.float32)  # Llama 3 style RoPE factors
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
    def tearDownClass(cls):
        del cls.gguf
        cls.tmp.cleanup()

    def test_dequantize_matches_written_values(self):
        for name, want in self.expected.items():
            got = self.gguf.tensors[name].dequantize()
            np.testing.assert_array_equal(got, want, err_msg=name)

    def test_loader_conversions(self):
        m = Model.from_gguf(self.gguf, max_ctx=8, numpy=True)

        def weights(mat):
            # The loaded matrix as f32 [rows x k], from its integer values and scales.
            v = mat.q_blocks * mat.s_blocks[:, :, None]                   # [groups, rows, sub]
            return v.transpose(1, 0, 2).reshape(mat.rows, -1)

        e = self.expected
        l0, l1 = m.layers
        self.assertEqual((l0["qkv"].fmt, l0["o"].fmt, l0["gate_up"].fmt, l0["down"].fmt),
                         ("q4", "q4", "q4", "q8"))
        self.assertEqual((l1["qkv"].fmt, l1["o"].fmt, l1["gate_up"].fmt, l1["down"].fmt),
                         ("q8", "q6", "q8", "q4"))
        self.assertEqual(m.lm.fmt, "q6")
        for i, layer in enumerate(m.layers):
            p = f"blk.{i}."
            # Q and K rows back in Hugging Face order; exact for Q4_0 and Q8_0 (fused into
            # Q8_0 too, in layer 1).
            hf_q = np.empty_like(e[p + "attn_q.weight"])
            hf_q[gguf_order(HEADS)] = e[p + "attn_q.weight"]
            hf_k = np.empty_like(e[p + "attn_k.weight"])
            hf_k[gguf_order(KV_HEADS)] = e[p + "attn_k.weight"]
            want_qkv = np.concatenate([hf_q, hf_k, e[p + "attn_v.weight"]])
            np.testing.assert_array_equal(weights(layer["qkv"]), want_qkv, err_msg=f"{p}qkv")
            gate, up = e[p + "ffn_gate.weight"], e[p + "ffn_up.weight"]
            want_gu = np.stack([gate.reshape(-1, 64, D), up.reshape(-1, 64, D)], 1).reshape(-1, D)
            got_gu = weights(layer["gate_up"])
            if LAYER_TYPES[i]["ffn_gate"] == 2:
                np.testing.assert_array_equal(got_gu, want_gu, err_msg=f"{p}gate_up")
            else:  # F16 gate re-quantized to Q8_0
                np.testing.assert_allclose(got_gu, want_gu, atol=np.abs(gate).max() / 100)
            down = e[p + "ffn_down.weight"]
            if LAYER_TYPES[i]["ffn_down"] == 2:
                np.testing.assert_array_equal(weights(layer["down"]), down)
            else:  # Q4_1 re-quantized to Q8_0: half a step, plus the scale's f16 rounding
                step = np.abs(down).reshape(D, -1, 32).max(axis=2) / 127
                err = np.abs(weights(layer["down"]) - down).reshape(D, -1, 32).max(axis=2)
                self.assertTrue(np.all(err <= step * (0.5 + 127 * 2 ** -11) + 1e-12))
        # Q6_K: values exact, scales d * sc rounded to f16.
        got = weights(m.lm)
        want = e["token_embd.weight"]
        np.testing.assert_allclose(got, want, rtol=2 ** -11, atol=1e-12)
        np.testing.assert_array_equal(m.embd, want)
        np.testing.assert_array_equal(l1["o"].q_blocks.shape, (HEADS * HEAD_DIM // 16, D, 16))

    def test_tokenizer(self):
        tok = Tokenizer(self.gguf)
        ids = {t: i for i, t in enumerate(tok.tokens)}
        bos, eot = ids["<|begin_of_text|>"], ids["<|eot_id|>"]
        self.assertEqual(tok.encode("the cat"), [bos, ord("t"), ids["he"], ids["Ġcat"]])
        self.assertEqual(tok.encode(" the<|eot_id|>", bos=False), [ids["Ġthe"], eot])
        self.assertEqual(tok.decode(tok.encode("the cat, the café!")), "the cat, the café!")
        self.assertEqual(tok.decode([bos, ids["Ġcat"], eot], skip_control=False),
                         "<|begin_of_text|> cat<|eot_id|>")

    def test_kernel_matches_numpy(self):
        m = Model.from_gguf(self.gguf, max_ctx=8, numpy=True)
        with open(os.path.join(ROOT, "examples", "kernels", "llama_decode.air"), encoding="utf-8") as f:
            src = f.read()
        for be in achainsaw.available_backends():
            k = achainsaw.compile(src, backend=be, fast_math=True)
            cache, h = m.new_cache(), np.zeros(m.d, dtype=np.float32)
            np_cache = m.numpy_cache()
            token, checked = 1, 0
            for pos in range(8):
                got = k.run("llama_decode", m.table, m.cfg, cache, token, pos, h, float(m.eps))
                want = m.numpy_step(np_cache, token, pos)
                scale = np.abs(m.last_h).max()
                np.testing.assert_allclose(h, m.last_h, atol=2e-3 * scale,
                                           err_msg=f"{be} pos {pos}")
                top = np.sort(m.last_logits)[-2:]
                if top[1] - top[0] > 1e-3 * np.abs(m.last_logits).max():
                    self.assertEqual(got, want, f"{be} token at pos {pos}")
                    checked += 1
                token = got
            self.assertGreaterEqual(checked, 4, be)


if __name__ == "__main__":
    unittest.main()
