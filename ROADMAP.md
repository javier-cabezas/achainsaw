# Roadmap

What's planned next, roughly in order within each section. Contributions are
welcome: open an issue to discuss an item before starting on a large one.

## Performance

Measured on Llama 3.2 1B Instruct with 8 threads on a Ryzen 7 8845HS (see
[docs/llama.md](docs/llama.md)). Decode is bound by memory bandwidth, so it
follows the bytes read per token; prefill is bound by the int8 dot products
and the scaling of their sums.

1. **Native Q4_1.** Llama 3.2's Q4_0 file has two Q4_1 down projections,
   which the loader converts to Q8_0: 4% more bytes per token than llama.cpp
   reads.
2. **An int8 KV cache, as an option.** At 8192 tokens of context, decode
   runs at 51.9 tokens/s against 67.4 at 512: the difference is attention
   over the bf16 cache. int8 would halve its reads, and with `vdot` its
   arithmetic too, at some cost in accuracy.
3. **Prefill on Cranelift.** Cranelift prefills at 93–172 tokens/s against
   LLVM's 486–1036 (contexts of 64 to 8192 tokens): its vectors are 128
   bits, and it runs tiles of 4 tokens instead of 8.

## Features

- **More models:** Qwen (biases on Q, K and V) and Gemma (GeGLU, whose
  kernel exists, and its norms) in `llama_decode.air`, and SentencePiece
  tokenizers (Mistral, Llama 2) in `benchmarks/gguf.py`, which reads Llama
  3's BPE today.
- **Sampling:** temperature, top-k and top-p in the decode step, which is
  greedy today (`topk.air` has the selection).
- **Release packages:** the release workflow builds CLI binaries and Python
  wheels for a version tag. Publishing a release, and the packages to PyPI
  and crates.io, would make `pip install achainsaw` work without Rust.
- **AMX on hardware:** the AMX `mm` kernels are compiled and checked in
  assembly, but haven't run on an AMX CPU yet.

Out of scope: GPUs.
