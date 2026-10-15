"""
Single-call LLM inference: examples/kernels/llama_decode.air on a real Llama GGUF checkpoint
or on a random model with Llama 3.2 1B's shapes, against a NumPy implementation of the same
math and, when PyTorch is installed, a PyTorch one.

The AIR kernel runs the whole prompt in one llama_prefill call (4 tokens per weight load),
then each decode step (embedding, every layer, final norm, LM head and greedy argmax) in one
llama_decode call. Both implementations use the same data types and arithmetic:
Q4_0 or Q8_0 weights with f32 block scales, activations quantized to int8 per 32-block on
the fly, exact integer block dots, a bf16 KV cache, f32 for everything else (embedding, norms,
RoPE, attention). They share the same weights (NumPy reads them block-major), so they should
predict the same tokens. NumPy has no int8 matrix product, so it widens the integer weights
to f32 in each call and runs the block dots as a batched f32 BLAS matmul (exact here).
The kernel compiles with fast_math=True (float min/max as compare and select).

PyTorch (eager, CPU) runs the same model on its weight-only quantized kernels, the path
torchao uses on CPU, with bf16 activations into each matrix, bf16 attention on a bf16 KV
cache, and f32 elsewhere. Q4_0
matrices keep their 4-bit values and 32-blocks; Q8_0 and Q6_K ones are requantized to int8
with one scale per row, as PyTorch has no per-block int8 kernel, so its tokens can differ a
little from the kernel's.

    # Llama 3.2 1B Instruct, Q4_0 (773 MB):
    # https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_0.gguf
    python benchmarks/benchmark_decode.py --gguf Llama-3.2-1B-Instruct-Q4_0.gguf \\
        --prompt "The capital of France is" --tokens 32
    python benchmarks/benchmark_decode.py --gguf model.gguf --chat --prompt "Explain SSA form"
    python benchmarks/benchmark_decode.py --gguf model.gguf --threads 8   # physical cores

    # Random weights (no download):
    python benchmarks/benchmark_decode.py                       # Q4_0, 16 layers, 32 tokens
    python benchmarks/benchmark_decode.py --weights q8 --layers 4 --tokens 16 --no-numpy
    python benchmarks/benchmark_decode.py --prompt-tokens 512 --tokens 1 --no-numpy  # prefill

    # Against llama.cpp on the same file (build llama-bench from github.com/ggml-org/llama.cpp):
    python benchmarks/benchmark_decode.py --gguf model.gguf --threads 8 --prompt-tokens 512 \\
        --tokens 64 --no-numpy --llama-bench ~/llama.cpp/build/bin/llama-bench

llama.cpp runs its own `llama-bench` (3 repetitions) with the same thread count: prefill of a
prompt of the same length, then the same number of generated tokens after a context of that
length (`--n-depth`). It keeps its KV cache in f16 (the kernel's is bf16) and has its own
kernels and rounding, so only its speed is compared.

NumPy and PyTorch run the prompt one token at a time. The prompt speed is the prefill time per prompt
token; the decode speed covers the generated tokens only.
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import achainsaw  # noqa: E402
from gguf import GGUF, Tokenizer  # noqa: E402
from llama_model import Model, torch  # noqa: E402

KERNEL = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "examples", "kernels",
                                      "llama_decode.air"))


def chat_prompt(text):
    """Llama 3's chat format for one user turn, without a system message."""
    return (f"<|start_header_id|>user<|end_header_id|>\n\n{text}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n")


def generate(step, prompt_ids, n_new, stop, prefill=None):
    """Greedy generation with `step(token, pos) -> next token`, and `prefill(tokens) -> next
    token` for the prompt when given (else one step per prompt token). Returns the new
    tokens, the prompt time and the per-token decode times."""
    t0 = time.perf_counter()
    nxt = None
    if prefill:
        nxt = prefill(prompt_ids)
    else:
        for pos, tok in enumerate(prompt_ids):
            nxt = step(tok, pos)
    prompt_s = time.perf_counter() - t0
    out, times = [], []
    pos = len(prompt_ids)
    while len(out) < n_new:
        out.append(nxt)
        if nxt in stop:
            break
        t = time.perf_counter()
        nxt = step(nxt, pos)
        times.append(time.perf_counter() - t)
        pos += 1
    return out, prompt_s, times


def llama_bench(path, gguf, prompt_tokens, gen_tokens, threads):
    """llama.cpp's own benchmark on the same GGUF file: prefill of `prompt_tokens` tokens,
    then `gen_tokens` generated after a context of that length (its --n-depth), as the AIR
    kernel decodes after the prompt. Returns (prompt tokens/s, decode tokens/s, commit)."""
    def run(*extra):
        cmd = [path, "-m", gguf, "-t", str(threads), "-r", "3", "-o", "json", *extra]
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
        return json.loads(out)[0]
    pp = run("-p", str(prompt_tokens), "-n", "0") if prompt_tokens > 1 else None
    tg = run("-p", "0", "-n", str(gen_tokens), "-d", str(prompt_tokens))
    return pp and pp["avg_ts"], tg["avg_ts"], tg["build_commit"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--gguf", metavar="PATH", help="Llama GGUF checkpoint (default: random)")
    parser.add_argument("--prompt", default="The capital of France is",
                        help="prompt for --gguf (default: %(default)r)")
    parser.add_argument("--chat", action="store_true", help="wrap the prompt in the chat format")
    parser.add_argument("--layers", type=int, default=16, help="random model layers")
    parser.add_argument("--weights", choices=["q4", "q8"], default="q4",
                        help="random model weight format (default: q4)")
    parser.add_argument("--tokens", type=int, default=32, help="tokens to generate")
    parser.add_argument("--prompt-tokens", type=int,
                        help="prompt length in random tokens (default: 16 for the random model; "
                             "with --gguf it replaces --prompt, and generation ignores "
                             "end-of-text tokens, as llama-bench does)")
    parser.add_argument("--backends", default="all")
    parser.add_argument("--threads", type=int,
                        help="threads for the kernel's par loops (default: all logical cores)")
    parser.add_argument("--no-numpy", action="store_true", help="skip the NumPy baseline")
    parser.add_argument("--no-torch", action="store_true",
                        help="skip the PyTorch baseline (skipped anyway without PyTorch)")
    parser.add_argument("--llama-bench", metavar="PATH",
                        help="with --gguf: also run llama.cpp's llama-bench on the same file, "
                             "threads, prompt length and token count")
    parser.add_argument("--json", metavar="PATH", help="also write the timings as JSON")
    args = parser.parse_args()
    use_torch = torch is not None and not args.no_torch
    if args.llama_bench and not args.gguf:
        parser.error("--llama-bench needs --gguf")

    t0 = time.perf_counter()
    if args.gguf:
        g = GGUF(args.gguf)
        tok = Tokenizer(g)
        if args.prompt_tokens:
            # Random ordinary tokens (below the special ones), as llama-bench uses; no stop
            # tokens, so every run generates exactly --tokens.
            prompt_ids = [int(t) for t in np.random.default_rng(1).integers(
                0, 128000, args.prompt_tokens)]
            stop = set()
        else:
            prompt = chat_prompt(args.prompt) if args.chat else args.prompt
            prompt_ids = tok.encode(prompt)
            stop = {tok.ids[t] for t in ("<|eot_id|>", "<|end_of_text|>", "<|eom_id|>")
                    if t in tok.ids}
        model = Model.from_gguf(g, max_ctx=len(prompt_ids) + args.tokens + 1,
                                numpy=not args.no_numpy, use_torch=use_torch)
        name = g.meta.get("general.name", os.path.basename(args.gguf))
        formats = ", ".join(f"{k} {'/'.join(v)}" for k, v in model.formats.items())
        print(f"{name}: {len(model.layers)} layers, {model.weight_bytes() / 1e9:.2f} GB of "
              f"packed weights ({formats}), loaded in {time.perf_counter() - t0:.1f} s")
        print(f"prompt: {len(prompt_ids)} tokens; generating up to {args.tokens}")
    else:
        tok, stop = None, set()
        prompt_ids = [int(t) for t in np.random.default_rng(1).integers(
            0, 128256, max(args.prompt_tokens or 16, 1))]
        model = Model.random(d=2048, layers=args.layers, heads=32, kv_heads=8, head_dim=64,
                             ffn=8192, vocab=128256, max_ctx=len(prompt_ids) + args.tokens + 1,
                             fmt=args.weights, numpy=not args.no_numpy, use_torch=use_torch)
        print(f"Llama 3.2 1B shapes, random weights, {args.layers} layers, "
              f"{args.weights.upper()}_0 {model.weight_bytes() / 1e9:.2f} GB "
              f"(built in {time.perf_counter() - t0:.1f} s); prompt of {len(prompt_ids)} "
              f"tokens, decoding {args.tokens}")
    wb = model.weight_bytes()

    src = open(KERNEL, encoding="utf-8").read()
    backends = achainsaw.available_backends() if args.backends == "all" else args.backends.split(",")

    results = []

    def report(label, prompt_s, times, backend=None, nbytes=wb):
        ms = np.median(times[1:] or times) * 1e3
        prompt_ms = prompt_s * 1e3 / len(prompt_ids)
        results.append(dict(backend=backend or label, ms_per_token=ms,
                            prompt_ms_per_token=prompt_ms, prompt_tokens=len(prompt_ids)))
        pre = (f"prompt {prompt_ms:6.2f} ms/token ({1e3 / prompt_ms:6.1f} tokens/s), "
               if len(prompt_ids) > 1 else "")
        print(f"  {label:<22} {pre}decode {ms:7.2f} ms/token  {1e3 / ms:6.1f} tokens/s  "
              f"{nbytes / (ms / 1e3) / 1e9:5.1f} GB/s of weights")

    runs = {}
    for be in backends:
        kernel = achainsaw.compile(src, backend=be, fast_math=True)
        kernel.set_threads(args.threads)
        cache, h = model.new_cache(), np.zeros(model.d, dtype=np.float32)

        def step(token, pos):
            return kernel.run("llama_decode", model.table, model.cfg, cache, token, pos, h,
                              float(model.eps))

        def prefill(ids):
            toks = np.array(ids, dtype=np.int64)
            return kernel.run("llama_prefill", model.table, model.cfg, cache, toks, len(ids), 0,
                              h, float(model.eps))

        out, prompt_s, times = generate(step, prompt_ids, args.tokens, stop, prefill)
        runs[be] = out
        report(f"AIR {be}, {kernel.threads} thr", prompt_s, times, be)
        if tok and not args.prompt_tokens:
            print("    " + repr(tok.decode(out)))

    if not args.no_numpy:
        cache = model.numpy_cache()
        out, prompt_s, times = generate(lambda t, p: model.numpy_step(cache, t, p), prompt_ids,
                                        args.tokens, stop)
        report("NumPy, same types", prompt_s, times, "numpy")
        for be, toks in runs.items():
            same = next((i for i, (a, b) in enumerate(zip(toks, out)) if a != b),
                        min(len(toks), len(out)))
            print(f"  {be}: first {same} of {len(toks)} tokens match NumPy")

    if use_torch:
        if args.threads:
            torch.set_num_threads(args.threads)
        cache = model.torch_cache()
        out, prompt_s, times = generate(lambda t, p: model.torch_step(cache, t, p), prompt_ids,
                                        args.tokens, stop)
        report(f"PyTorch, {torch.get_num_threads()} thr", prompt_s, times, "torch",
               model.torch_weight_bytes())
        if tok:
            print("    " + repr(tok.decode(out)))
        for be, toks in runs.items():
            same = next((i for i, (a, b) in enumerate(zip(toks, out)) if a != b),
                        min(len(toks), len(out)))
            print(f"  {be}: first {same} of {len(toks)} tokens match PyTorch")

    if args.llama_bench:
        threads = args.threads or os.cpu_count()
        pp_ts, tg_ts, commit = llama_bench(args.llama_bench, args.gguf, len(prompt_ids),
                                           args.tokens, threads)
        # Its own prefill and decode rates, as the AIR rows report theirs.
        prompt_s = len(prompt_ids) / pp_ts if pp_ts else 0.0
        report(f"llama.cpp, {threads} thr", prompt_s, [1.0 / tg_ts], "llama.cpp")
        print(f"    llama-bench build {commit}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(dict(model=args.gguf or f"random {args.weights} x{args.layers} layers",
                           weight_bytes=wb, results=results), f, indent=2)


if __name__ == "__main__":
    main()
