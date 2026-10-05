"""
Minimal GGUF reader (llama.cpp's model format), NumPy only: metadata, tensors as memory-mapped
block arrays, dequantization of the common weight types, and the Llama 3 byte-level BPE
tokenizer stored in the file.

    g = GGUF("Llama-3.2-1B-Instruct-Q4_0.gguf")
    g.meta["llama.block_count"], g.tensors["blk.0.attn_q.weight"].type_name
    w = g.tensors["output_norm.weight"].dequantize()      # float32, [rows x cols]
    tok = Tokenizer(g); ids = tok.encode("Hello"); text = tok.decode(ids)

Shapes are given as GGUF stores them, innermost first: a weight with dims [k, m] is m rows of
k values, each row a run of quantization blocks along k.
"""

import re
import struct

import numpy as np

# ggml type id -> (name, values per block, bytes per block)
GGML_TYPES = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),
    3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34),
    10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
    30: ("BF16", 1, 2),
}

_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q",
            11: "<q", 12: "<d"}


class Tensor:
    def __init__(self, name, dims, type_id, data):
        self.name, self.dims, self.type_id = name, dims, type_id
        self.type_name, self.block_values, self.block_bytes = GGML_TYPES.get(
            type_id, (f"type{type_id}", None, None))
        self.data = data  # uint8 [rows x row bytes], memory-mapped

    @property
    def rows(self):
        return int(np.prod(self.dims[1:], dtype=np.int64)) if len(self.dims) > 1 else 1

    @property
    def cols(self):
        return self.dims[0]

    def blocks(self):
        """uint8 [rows x blocks per row x block bytes]."""
        return self.data.reshape(self.rows, -1, self.block_bytes)

    def dequantize(self, rows=None):
        """float32 [rows x cols], computed as llama.cpp's dequantize_row_* do (f32 math)."""
        sel = slice(None) if rows is None else rows
        name = self.type_name
        if name == "F32":
            return self.data.view(np.float32).reshape(self.rows, self.cols)[sel].copy()
        if name == "F16":
            return self.data.view(np.float16).reshape(self.rows, self.cols)[sel].astype(np.float32)
        if name == "BF16":
            bits = self.data.view(np.uint16).reshape(self.rows, self.cols)[sel]
            return (bits.astype(np.uint32) << 16).view(np.float32)
        decode = _DEQUANT.get(name)
        if decode is None:
            raise ValueError(f"{self.name}: unsupported tensor type {name}")
        b = self.blocks()[sel]
        n_rows = b.shape[0]
        return decode(b.reshape(-1, self.block_bytes)).reshape(n_rows, self.cols)


def _f16(raw):
    return np.ascontiguousarray(raw).view(np.float16).astype(np.float32)


def _deq_q4_0(b):
    d = _f16(b[:, 0:2])
    qs = b[:, 2:18]
    q = np.concatenate([qs & 15, qs >> 4], axis=1).astype(np.float32) - np.float32(8)
    return q * d


def _deq_q4_1(b):
    d, m = _f16(b[:, 0:2]), _f16(b[:, 2:4])
    qs = b[:, 4:20]
    q = np.concatenate([qs & 15, qs >> 4], axis=1).astype(np.float32)
    return q * d + m


def _deq_q8_0(b):
    return np.ascontiguousarray(b[:, 2:34]).view(np.int8).astype(np.float32) * _f16(b[:, 0:2])


def _deq_q6_k(b):
    """256 values per block: 6-bit q (low 4 bits in ql, high 2 in qh) minus 32, times the
    f16 super-scale d and an int8 scale per 16 values."""
    ql, qh = b[:, 0:128], b[:, 128:192]
    sc = np.ascontiguousarray(b[:, 192:208]).view(np.int8).astype(np.float32)
    d = _f16(b[:, 208:210])
    y = np.empty((b.shape[0], 256), dtype=np.float32)
    is_ = np.arange(32) // 16
    for h in range(2):
        l, hh, s = ql[:, 64 * h:64 * h + 64], qh[:, 32 * h:32 * h + 32], sc[:, 8 * h:8 * h + 8]
        parts = [
            (l[:, :32] & 15) | ((hh & 3) << 4),
            (l[:, 32:] & 15) | (((hh >> 2) & 3) << 4),
            (l[:, :32] >> 4) | (((hh >> 4) & 3) << 4),
            (l[:, 32:] >> 4) | (((hh >> 6) & 3) << 4),
        ]
        for p, q in enumerate(parts):
            ds = d * s[:, is_ + 2 * p]
            y[:, 128 * h + 32 * p:128 * h + 32 * p + 32] = ds * (q.astype(np.float32) - np.float32(32))
    return y


_DEQUANT = {"Q4_0": _deq_q4_0, "Q4_1": _deq_q4_1, "Q8_0": _deq_q8_0, "Q6_K": _deq_q6_k}


class GGUF:
    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            self._f = f
            if f.read(4) != b"GGUF":
                raise ValueError(f"{path}: not a GGUF file")
            version, n_tensors, n_kv = struct.unpack("<IQQ", f.read(20))
            if version not in (2, 3):
                raise ValueError(f"{path}: unsupported GGUF version {version}")
            self.meta = {}
            for _ in range(n_kv):
                key = self._str()
                self.meta[key] = self._value(self._u32())
            infos = []
            for _ in range(n_tensors):
                name = self._str()
                n_dims = self._u32()
                dims = list(struct.unpack(f"<{n_dims}Q", f.read(8 * n_dims)))
                type_id = self._u32()
                (offset,) = struct.unpack("<Q", f.read(8))
                infos.append((name, dims, type_id, offset))
            align = self.meta.get("general.alignment", 32)
            data_start = (f.tell() + align - 1) // align * align
        mm = np.memmap(path, dtype=np.uint8, mode="r")
        self.tensors = {}
        for name, dims, type_id, offset in infos:
            _, per_block, block_bytes = GGML_TYPES.get(type_id, (None, 1, 0))
            n = int(np.prod(dims, dtype=np.int64))
            nbytes = n // per_block * block_bytes if block_bytes else 0
            rows = n // dims[0]
            data = mm[data_start + offset:data_start + offset + nbytes]
            self.tensors[name] = Tensor(name, dims, type_id,
                                        data.reshape(rows, -1) if nbytes else data)

    def _u32(self):
        return struct.unpack("<I", self._f.read(4))[0]

    def _str(self):
        (n,) = struct.unpack("<Q", self._f.read(8))
        return self._f.read(n).decode("utf-8", errors="replace")

    def _value(self, t):
        if t in _SCALARS:
            fmt = _SCALARS[t]
            return struct.unpack(fmt, self._f.read(struct.calcsize(fmt)))[0]
        if t == 8:
            return self._str()
        if t == 9:
            et = self._u32()
            (n,) = struct.unpack("<Q", self._f.read(8))
            if et in _SCALARS:
                fmt = _SCALARS[et]
                size = struct.calcsize(fmt)
                return np.frombuffer(self._f.read(size * n), dtype=np.dtype(fmt)).tolist()
            return [self._value(et) for _ in range(n)]
        raise ValueError(f"unknown GGUF value type {t}")


def _bytes_to_unicode():
    """GPT-2's reversible map from bytes to printable characters."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + \
        list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


# Llama 3's pre-tokenizer split (llama.cpp's "llama-bpe"), with \p{L} as [^\W\d_] and \p{N}
# as \d, since Python's `re` has no Unicode properties: exact for letters and decimal digits.
_LLAMA3_SPLIT = re.compile(
    r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])"
    r"|(?:[^\r\n\w]|_)?[^\W\d_]+"
    r"|\d{1,3}"
    r"| ?(?:[^\s\w]|_)+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+")


class Tokenizer:
    """The byte-level BPE tokenizer of a GGUF file whose tokenizer.ggml.model is gpt2 (Llama 3).
    Control tokens (`<|eot_id|>` and the like) in the text are encoded as themselves."""

    def __init__(self, gguf):
        meta = gguf.meta
        if meta.get("tokenizer.ggml.model") != "gpt2":
            raise ValueError("only byte-level BPE (gpt2) tokenizers are supported")
        self.tokens = meta["tokenizer.ggml.tokens"]
        types = meta.get("tokenizer.ggml.token_type", [1] * len(self.tokens))
        self.ids = {t: i for i, t in enumerate(self.tokens)}
        self.ranks = {tuple(m.split(" ", 1)): r for r, m in enumerate(meta["tokenizer.ggml.merges"])}
        self.bos = meta.get("tokenizer.ggml.bos_token_id")
        self.eos = meta.get("tokenizer.ggml.eos_token_id")
        self.special = {self.tokens[i]: i for i, t in enumerate(types) if t in (3, 4)}
        self._special_split = re.compile(
            "(" + "|".join(re.escape(s) for s in sorted(self.special, key=len, reverse=True)) + ")")
        self.control = {i for i, t in enumerate(types) if t == 3}
        self._enc = _bytes_to_unicode()
        self._dec = {c: b for b, c in self._enc.items()}

    def _bpe(self, word):
        if word in self.ids:  # llama-bpe skips merges for whole-vocabulary words
            return [self.ids[word]]
        parts = list(word)
        while len(parts) > 1:
            pairs = [(self.ranks.get((a, b), 1 << 62), i)
                     for i, (a, b) in enumerate(zip(parts, parts[1:]))]
            rank, i = min(pairs)
            if rank == 1 << 62:
                break
            parts[i:i + 2] = [parts[i] + parts[i + 1]]
        return [self.ids[p] for p in parts]

    def encode(self, text, bos=True):
        out = [self.bos] if bos and self.bos is not None else []
        for chunk in self._special_split.split(text) if self.special else [text]:
            if not chunk:
                continue
            if chunk in self.special:
                out.append(self.special[chunk])
                continue
            for piece in _LLAMA3_SPLIT.findall(chunk):
                out += self._bpe("".join(self._enc[b] for b in piece.encode("utf-8")))
        return out

    def decode(self, ids, skip_control=True):
        data = bytearray()
        for i in ids:
            if skip_control and i in self.control:
                continue
            tok = self.tokens[i]
            if i in self.control or tok in self.special:
                data += tok.encode("utf-8")
            else:
                data += bytes(self._dec[c] for c in tok)
        return data.decode("utf-8", errors="replace")
