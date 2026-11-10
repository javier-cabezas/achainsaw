# Roadmap

What's planned next, roughly in order within each section. Contributions are
welcome: open an issue to discuss an item before starting on a large one.

## Performance

Measured on Llama 3.2 1B Instruct with 8 threads on a Ryzen 7 8845HS (see
[docs/llama.md](docs/llama.md)). Decode is bound by memory bandwidth, so it
follows the bytes read per token; prefill is bound by the int8 dot products
and the scaling of their sums.

1. **Prefill on Cranelift.** Cranelift prefills at 93–172 tokens/s against
   LLVM's 486–1036 (contexts of 64 to 8192 tokens): its vectors are 128
   bits, and it runs tiles of 4 tokens instead of 8.
2. **GLM-4.5 decode.** At GLM-4.5-Air's layer shapes, decode only matches
   llama.cpp's (37–41 against 38 tokens/s), where the other mixtures of
   experts lead by 1.1–1.3x. Its 96 query heads make attention and the QKV
   projection a larger share of each token: profile those passes there.

## Features

- **Gemma 3** in `llama_decode.air`: sliding windows in 5 of every 6 layers
  (the ring cache holds them), q and k head norms, GeGLU, norms after
  attention and after the MLP, a scaled embedding, and its own RoPE base for
  the local layers.
- **DeepSeek proper:** MLA attention (a compressed KV cache, with the
  projections absorbed into the queries and outputs) and group-limited
  routing, for DeepSeek-V2-Lite and V3/R1-shaped models. Today the kernel
  runs DeepSeek-V3's mixture of experts on ordinary attention (GLM-4.5,
  dots.llm1).
- **SentencePiece tokenizers** (Mistral, Llama 2) in `benchmarks/gguf.py`,
  which reads byte-level BPE (Llama 3, Qwen, GPT-4o, GLM) today.
- **Sampling:** temperature, top-k and top-p in the decode step, which is
  greedy today (`topk.air` has the selection).
- **Release packages:** the release workflow builds CLI binaries and Python
  wheels for a version tag. Publishing a release, and the packages to PyPI
  and crates.io, would make `pip install achainsaw` work without Rust.
- **AMX on hardware:** the AMX `mm` kernels are compiled and checked in
  assembly, but haven't run on an AMX CPU yet.

Out of scope: GPUs.
