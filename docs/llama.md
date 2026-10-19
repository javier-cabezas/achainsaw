# Llama in one call

[`examples/kernels/llama_decode.air`](../examples/kernels/llama_decode.air) runs a whole decode step of a Llama-style model (RMSNorm, grouped-query attention with RoPE and a bf16 KV cache, SwiGLU MLP, Q4_0/Q8_0/Q6_K/Q4_K/Q5_K weights) in one call: the token id goes in, the greedy next token comes out. Like GPU megakernels ([Hazy Research's Llama-1B megakernel](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles), [AutoMegaKernel](https://arxiv.org/pdf/2606.09682)), it removes the boundaries between the ~100 small kernels of a forward pass; on a CPU that means each elementwise op runs inside the pass that produces or consumes its data:

| Pass | Fused into it |
|---|---|
| RMSNorm (serial) | Q8 quantization of its output |
| QKV projection (`par`, one task per head) | RoPE, bf16 KV-cache append |
| Attention (`par`, one task per KV head and 256-position split) | the group's query heads 4 at a time with their scores and outputs in registers, each K and V vector widened from bf16 once for the 4 (`attn_split`), scores with positions in lanes so none needs a horizontal sum, vectorized softmax `exp` |
| Attention merge (`par`, one task per q head) | the splits merged (`attn_merge`), Q8 quantization of its output |
| O and down projections (`par`, 64 rows per task) | residual add |
| Gate and up projections (`par`, 64 rows per task) | SwiGLU, Q8 quantization |
| LM head (`par`, 64 rows per task) | argmax: each task keeps only its best logit, so the logits are never written |

That is 2 serial norms and 6 `par` regions per layer. The KV cache is bf16 and head-major (`[kv_heads x max_ctx x head_dim]` per layer for V, and the same for K in tiles of 32 positions, each `[head_dim x 32]`, so that one vector holds a value of consecutive positions and a score needs no horizontal sum), so a KV head's positions are contiguous, and attention is flash-decoding: positions split across cores even for one sequence, with the standard library's `attn_split` and `attn_merge` (also behind [`gqa_attention.air`](../examples/kernels/gqa_attention.air)). Each weight matrix has its own format, Q8_0, Q4_0, Q6_K, Q4_K or Q5_K, in the chunked layout of `q8_gemv` and `q4_gemv` (Q6_K adds a plane of high bits and a scale per 16 values; Q4_K and Q5_K, a min per 32 values that the activation sums apply, and Q5_K a plane of top bits), so every task owns its output rows and applies its epilogue without a cross-task reduction. A matrix is a list of parts, so the fused QKV matrix keeps the formats of Q, K and V when they differ. `crates/achainsaw-codegen/tests/llama_decode.rs` checks it against an f64 reference model, with all five formats mixed within and across matrices, over several positions, at every ISA level on both backends.

**Prefill.** `llama_prefill` runs a whole prompt of n tokens through the same passes in one call. Every projection task applies its rows to 8 tokens per weight load (`qmat_batch`, which runs the standard library's `qmat_chunk8` on tiles of 8 tokens, `qmat_chunk4` on a tile of 4 and `qmat_chunk` on the rest; with 128-bit vectors, as on Cranelift, tiles of 4), so the weights are read once per 8 tokens instead of once per token, and the work turns from memory-bound into compute-bound. Norms, RoPE and epilogues run per token, and attention runs one task per tile of 8 tokens and KV head, each token over the cache up to its own position with `llama_decode`'s splits and merges; the splits that all 8 tokens see whole run once for the tile's query heads, so each K and V vector is read once per tile. Each token's operations are exactly those of `llama_decode`, so a prefill leaves the same KV cache, hidden state and next token bit for bit as decoding the prompt token by token (`llama_prefill_matches_decode_exactly`, in one call or in pieces).

**A real model.** [`benchmarks/gguf.py`](../benchmarks/gguf.py) reads llama.cpp's GGUF files with NumPy alone (metadata, tensors, dequantization of Q4_0, Q4_1, Q8_0, Q4_K, Q5_K, Q6_K, F16, BF16 and F32, and the Llama 3 BPE tokenizer, which gives the same ids as `llama-tokenize`), and [`benchmarks/llama_model.py`](../benchmarks/llama_model.py) converts a Llama checkpoint for the kernel: Q4_0 and Q8_0 matrices bit for bit, Q6_K values bit for bit with each 16-value scale (super-block scale times sub-block scale) rounded to f16, Q4_K and Q5_K values bit for bit with each 32-value scale and min rounded to f16, anything else to Q8_0; Q and K rows back from llama.cpp's interleaved RoPE order; RoPE tables with Llama 3's frequency factors. The prompt goes through `llama_prefill`, then each new token through `llama_decode`.

```bash
curl -LO https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_0.gguf
python benchmarks/benchmark_decode.py --gguf Llama-3.2-1B-Instruct-Q4_0.gguf --threads 8 --tokens 64
# Paris. The capital of Germany is Berlin. The capital of Italy is Rome. The capital of ...
```

Llama 3.2 1B Instruct in Q4_0 (Q4_0 layers, two Q4_1 down projections, a Q6_K embedding shared with the LM head), greedy decoding on a Ryzen 7 8845HS with 8 threads (one per core, the fastest setting for each):

| | ms/token | tokens/s | weights read per token |
|---|---|---|---|
| llama.cpp (CPU, build 6868a52, `llama-bench -n 64 -t 8`) | 15.1 | 66.3 | 765 MB at 51 GB/s |
| AIR on LLVM, one call per token | 14.8 | 67.7 | 794 MB at 54 GB/s |
| AIR on Cranelift, one call per token | 15.8 | 63.3 | 794 MB at 50 GB/s |
| PyTorch 2.14, weight-only int4/int8 kernels | 29.7 | 33.7 | 884 MB at 30 GB/s |
| NumPy, same data types | 723 | 1.4 | |

The kernel prints the same text as llama.cpp (`llama-simple`, greedy), and its tokens match the NumPy implementation of the same arithmetic. Both are bound by memory bandwidth at about the same rate; the kernel reads 4% more bytes, because the two Q4_1 matrices become Q8_0 and its Q6_K scales take 0.5 bit more per weight than llama.cpp's packing.

PyTorch runs the same model eagerly on its weight-only quantized kernels (`benchmark_decode.py` uses them when PyTorch is installed; `--no-torch` skips it). Q4_0 matrices become int4 with the same 32-blocks; the Q8_0 down projections and the Q6_K LM head become int8 with one scale per row. Its tokens match too. Its int4 kernel streams weights at about half the rate of LLVM's `vdotu` loop, and it reads the int8 LM head instead of 6.5-bit Q6_K.

**Against llama.cpp, by context length.** `benchmark_decode.py --llama-bench PATH` runs llama.cpp's own `llama-bench` on the same file, thread count, prompt length and number of generated tokens, and prints it as one more row (`--prompt-tokens N` makes the prompt N random tokens, as llama-bench's are):

```bash
python benchmarks/benchmark_decode.py --gguf Llama-3.2-1B-Instruct-Q4_0.gguf --threads 8 \
    --prompt-tokens 512 --tokens 64 --no-numpy --llama-bench ~/llama.cpp/build/bin/llama-bench
```

Prefill of the prompt (one `llama_prefill` call, the mean of 3 after a first one, as llama-bench's `-p` averages 3), then 64 tokens generated after it (`llama_decode`; llama-bench's `-n 64 -d <prompt>`), 8 threads, llama.cpp build 6868a52 with its default f16 KV cache:

| Context | Prefill, AIR on LLVM | Prefill, llama.cpp | Decode, AIR on LLVM | Decode, llama.cpp |
|---|---|---|---|---|
| 64 tokens | 790 tokens/s (1.27 ms/token) | 441 tokens/s (2.27 ms/token) | 66.3 tokens/s (15.1 ms/token) | 65.6 tokens/s (15.2 ms/token) |
| 512 tokens | 1036 tokens/s (0.97 ms/token) | 437 tokens/s (2.29 ms/token) | 67.4 tokens/s (14.8 ms/token) | 60.7 tokens/s (16.5 ms/token) |
| 2048 tokens | 838 tokens/s (1.19 ms/token) | 398 tokens/s (2.52 ms/token) | 64.1 tokens/s (15.6 ms/token) | 48.6 tokens/s (20.6 ms/token) |
| 8192 tokens | 486 tokens/s (2.06 ms/token) | 283 tokens/s (3.53 ms/token) | 51.9 tokens/s (19.3 ms/token) | 27.1 tokens/s (37.0 ms/token) |

- **Prefill is 1.7–2.4x faster than llama.cpp at every length.** `llama_prefill` loads and unpacks each weight vector once per 8 tokens and runs the int8 dots on `vdotu`. Its attention does `llama_decode`'s arithmetic for every token (so a prefill stays bit-identical to decoding), in tiles of 8 tokens: the 256-position splits that all 8 see whole run once for their 32 query heads, so each K and V vector is read once for the tile.
- **Decode is level on short contexts and pulls ahead on long ones**: 1.3x at 2048 tokens, 1.9x at 8192. Both caches take 2 bytes per value; the kernel reads each KV head's positions once for its group's query heads and splits them across cores. Run to run, decode varies by about ±3% and prefill by up to ±10% (this laptop's clocks).
- **Scores with positions in lanes.** K is stored in tiles of 32 positions (`[head_dim x 32]`), so one FMA multiplies a broadcast query value by one K value of up to 32 positions, and no score needs a horizontal sum; scoring and P V then take 4 FMAs each per head and position (head_dim 64, AVX-512). Before, each score was a dot product along K's row and a horizontal sum, which took three quarters of the attention time; at 2048 and 8192 tokens the kernel prefilled 644 and 293 tokens/s and decoded 62.0 and 48.7 tokens/s (alternating the two kernels: 1.24x and 1.64x faster prefill, 1.03x faster decode at both).
- Before the bf16 cache and split attention, the kernel's attention took one query head at a time over an f32 cache, and at 2048 and 8192 tokens it prefilled 287 and 46 tokens/s and decoded 45.6 and 17.5 tokens/s.
- With 16 threads (two per core) at 512 tokens, AIR prefills 1106 tokens/s and decodes 61.0 tokens/s; llama.cpp 540 and 44.7. Both decode faster on 8.
- Cranelift prefills 170, 172, 146 and 93 tokens/s and decodes 56.9, 59.6, 53.9 and 38.3 tokens/s at the four lengths.
- Prefill loads and unpacks each weight vector once per 8 tokens on LLVM (once per 4 before, and still on Cranelift's 128-bit vectors), with the 8 tokens' sums kept in registers across blocks. Measured against the previous kernel the same way, alternating the two, that made 512-token prefill 6% faster on Q4_0, 9% on Q4_K_M and 12% on Q5_K_M. Earlier versions of these tables timed one cold `llama_prefill` call, which came out up to 10% faster or slower than this mean of warm ones.

**K-quants.** Most GGUF downloads are Q4_K_M or Q5_K_M files: Q4_K or Q5_K matrices (256-value super-blocks whose 32-value sub-blocks each have a 6-bit scale and a 6-bit min, so a weight is `scale * q - min`), with the V and down projections of half the layers, and the embedding, in Q6_K. The loader keeps the values bit for bit and rounds each sub-block's scale and min (the super-block's f16 factor times the 6-bit one) to f16, so a Q4_K weight takes 5 bits instead of 4.5; the kernel subtracts each min times the block's activation sum, which `quantize_q8` already computes. The fused QKV matrix is stored in parts, keeping Q and K in Q4_K and V in Q6_K. The same comparison at 512 tokens, the three files measured together (the mean of two runs, each started on an idle machine):

| File | Prefill, AIR on LLVM | Prefill, llama.cpp | Decode, AIR on LLVM | Decode, llama.cpp |
|---|---|---|---|---|
| Q4_0 (773 MB) | 1036 tokens/s | 437 tokens/s | 67.4 tokens/s | 60.7 tokens/s |
| Q4_K_M (808 MB) | 889 tokens/s | 503 tokens/s | 60.7 tokens/s | 57.7 tokens/s |
| Q5_K_M (912 MB) | 876 tokens/s | 787 tokens/s | 54.7 tokens/s | 52.1 tokens/s |

- **Decode stays ahead** by 5% on the K-quant files, both bound by memory bandwidth, although the kernel reads 8% more bytes per token (0.87 and 0.98 GB of packed weights).
- **Prefill is 1.8x faster on Q4_K_M and 1.1x faster on Q5_K_M**, where llama.cpp is faster than on its own Q4_K_M. Prefill is compute-bound: the int8 dots take half of the inner loop, and scaling each block's sums for every token much of the rest, while unpacking Q5_K's top bits for 8 tokens at a time costs little.
- Cranelift prefills 146 and 125 tokens/s and decodes 49.4 and 43.1 tokens/s on the two files.
- The kernel's tokens match NumPy's (`--prompt "The capital of France is"`: 24 of 24 on both files); on random prompts they part only at near-ties, as on Q4_0. PyTorch runs Q4_K on its int4 kernel with the min as its zero point (rounded to bf16) and Q5_K requantized to int8 per row; it decodes Q4_K_M at 30–33 tokens/s.

**Q4 against Q8.** On a random model with the same shapes (`benchmark_decode.py --weights q8|q4`; d 2048, 16 layers, 32 query heads over 8 KV heads of 64, MLP 8192, vocabulary 128256), all Q8_0 or all Q4_0, 8 threads:

| Weights | AIR on LLVM | AIR on Cranelift | PyTorch, weight-only kernels | NumPy, same data types |
|---|---|---|---|---|
| Q8_0, 1.31 GB | 23.9 ms/token (42 tokens/s, 55 GB/s) | 28.4 ms/token | 37.8 ms/token (int8, one scale per row) | 909 ms/token |
| Q4_0, 0.70 GB | 12.7 ms/token (79 tokens/s, 55 GB/s) | 13.8 ms/token | 27.3 ms/token (int4) | 661 ms/token |

On LLVM decode streams the weights at close to the memory's bandwidth (run to run, about ±5%), so Q4_0 is about 2x faster than Q8_0. On Cranelift, `vdot`, in-register unpacking and unrolled groups make the int8 dots cheap enough that Q4_0 streams at 50 GB/s too, close to LLVM. NumPy, lacking an int8 matrix product, widens every weight to f32 on each token. PyTorch's int4 and int8 kernels read the weights at 28–33 GB/s.

Every format on the same random model (`--weights q8|q6|q5k|q4k|q4`), prefill of 512 tokens then decode, 8 threads, best of two runs:

| Weights | Prefill, LLVM | Decode, LLVM | Prefill, Cranelift | Decode, Cranelift |
|---|---|---|---|---|
| Q8_0, 1.31 GB | 1091 tokens/s | 41.7 tokens/s (55 GB/s) | 180 tokens/s | 35.8 tokens/s |
| Q6_K, 1.08 GB | 869 tokens/s | 49.5 tokens/s (54 GB/s) | 124 tokens/s | 39.6 tokens/s |
| Q5_K, 0.93 GB | 965 tokens/s | 59.2 tokens/s (55 GB/s) | 128 tokens/s | 45.1 tokens/s |
| Q4_K, 0.77 GB | 1075 tokens/s | 70.4 tokens/s (54 GB/s) | 156 tokens/s | 56.6 tokens/s |
| Q4_0, 0.70 GB | 1147 tokens/s | 77.4 tokens/s (54 GB/s) | 179 tokens/s | 70.1 tokens/s |

Decode on LLVM follows the bytes per weight. Prefill follows the unpacking and scaling work: Q8_0 needs no unpacking, Q4_0 a mask and a shift, Q4_K adds a min, and Q5_K and Q6_K merge a plane of top bits; Q6_K also scales its two halves separately. With 8 tokens per weight load on LLVM, the slowest format (Q6_K) prefills at 76% of the speed of the fastest. `qmat_chunk` chooses the format once per block, for a strip loop of its own, and unrolls each strip's 4 groups: a branch on the format in every strip, which Cranelift cannot hoist out of the loop, had made its quantized GEMVs up to 28% slower; choosing per block made prefill 20–25% faster on LLVM and decode 30–50% faster on Cranelift.
