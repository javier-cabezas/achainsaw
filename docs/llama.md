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

That is 2 serial norms and 6 `par` regions per layer. The KV cache is bf16 and head-major (`[kv_heads x max_ctx x head_dim]` per layer for V, and the same for K in tiles of 32 positions, each `[head_dim x 32]`, so that one vector holds a value of consecutive positions and a score needs no horizontal sum), so a KV head's positions are contiguous (with an int8 cache, see below, each tile of 32 positions holds int8 values and a scale per position), and attention is flash-decoding: positions split across cores even for one sequence, with the standard library's `attn_split` and `attn_merge` (also behind [`gqa_attention.air`](../examples/kernels/gqa_attention.air)). Each weight matrix has its own format, Q8_0, Q4_0, Q6_K, Q4_K, Q5_K, Q4_1, IQ4_NL or MXFP4, in the chunked layout of `q8_gemv` and `q4_gemv` (Q6_K adds a plane of high bits and a scale per 16 values; Q4_K, Q5_K and Q4_1, a min per 32 values that the activation sums apply, and Q5_K a plane of top bits; IQ4_NL's and MXFP4's 4-bit codes go through a 16-value codebook with `vtbl`), so every task owns its output rows and applies its epilogue without a cross-task reduction. A matrix is a list of parts, so the fused QKV matrix keeps the formats of Q, K and V when they differ. `crates/achainsaw-codegen/tests/llama_decode.rs` checks it against an f64 reference model, with all eight formats mixed within and across matrices, over several positions, at every ISA level on both backends.

**Prefill.** `llama_prefill` runs a whole prompt of n tokens through the same passes in one call. Every projection task applies its rows to 8 tokens per weight load (`qmat_batch`, which runs the standard library's `qmat_chunk8` on tiles of 8 tokens, `qmat_chunk4` on a tile of 4 and `qmat_chunk` on the rest; with 128-bit vectors, as on Cranelift, tiles of 4), so the weights are read once per 8 tokens instead of once per token, and the work turns from memory-bound into compute-bound. Norms, RoPE and epilogues run per token, and attention runs one task per tile of 8 tokens and KV head, each token over the cache up to its own position with `llama_decode`'s splits and merges; the splits that all 8 tokens see whole run once for the tile's query heads, so each K and V vector is read once per tile. Each token's operations are exactly those of `llama_decode`, so a prefill leaves the same KV cache, hidden state and next token bit for bit as decoding the prompt token by token (`llama_prefill_matches_decode_exactly`, in one call or in pieces).

**A real model.** [`benchmarks/gguf.py`](../benchmarks/gguf.py) reads llama.cpp's GGUF files with NumPy alone (metadata, tensors, dequantization of Q4_0, Q4_1, IQ4_NL, MXFP4, Q8_0, Q4_K, Q5_K, Q6_K, F16, BF16 and F32, and the byte-level BPE tokenizer with Llama 3's or Qwen's pre-tokenizer, which gives the same ids as `llama-tokenize`), and [`benchmarks/llama_model.py`](../benchmarks/llama_model.py) converts a Llama checkpoint for the kernel: Q4_0 and Q8_0 matrices bit for bit, Q6_K values bit for bit with each 16-value scale (super-block scale times sub-block scale) rounded to f16, Q4_K and Q5_K bit for bit (values, 6-bit scales and mins, and f16 super-block factors), Q4_1 bit for bit (values and f16 scales and mins), IQ4_NL bit for bit (codes and f16 scales), MXFP4 bit for bit (codes, and power-of-two scales as f16s), anything else to Q8_0; Q and K rows back from llama.cpp's interleaved RoPE order; RoPE tables with Llama 3's frequency factors. The prompt goes through `llama_prefill`, then each new token through `llama_decode`.

```bash
curl -LO https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_0.gguf
python benchmarks/benchmark_decode.py --gguf Llama-3.2-1B-Instruct-Q4_0.gguf --threads 8 --tokens 64
# Paris. The capital of Germany is Berlin. The capital of Italy is Rome. The capital of ...
```

Llama 3.2 1B Instruct in Q4_0 (Q4_0 layers, two Q4_1 down projections, a Q6_K embedding shared with the LM head), greedy decoding on a Ryzen 7 8845HS with 8 threads (one per core, the fastest setting for each):

| | ms/token | tokens/s | weights read per token |
|---|---|---|---|
| llama.cpp (CPU, build 6868a52, `llama-bench -n 64 -t 8`) | 15.0 | 66.6 | 765 MB at 51 GB/s |
| AIR on LLVM, one call per token | 14.2 | 70.2 | 780 MB at 55 GB/s |
| AIR on Cranelift, one call per token | 16.2 | 61.9 | 780 MB at 48 GB/s |
| PyTorch 2.14, weight-only int4/int8 kernels | 28.8 | 34.7 | 870 MB at 30 GB/s |
| NumPy, same data types | 723 | 1.4 | |

The kernel prints the same text as llama.cpp (`llama-simple`, greedy), and its tokens match the NumPy implementation of the same arithmetic. Both are bound by memory bandwidth at about the same rate; the kernel reads 2% more bytes, because its Q6_K scales take 0.5 bit more per weight than llama.cpp's packing. The two Q4_1 matrices keep their format (until they did, the loader converted them to Q8_0, and the kernel read 794 MB per token; in five runs of each, alternating the two, keeping Q4_1 made decode 2% faster on LLVM and prefill 3% slower, as Q8_0 needs no unpacking).

PyTorch runs the same model eagerly on its weight-only quantized kernels (`benchmark_decode.py` uses them when PyTorch is installed; `--no-torch` skips it). Q4_0 and Q4_1 matrices become int4 with the same 32-blocks; the Q6_K LM head becomes int8 with one scale per row. Its tokens match too. Its int4 kernel streams weights at about half the rate of LLVM's `vdotu` loop, and it reads the int8 LM head instead of 6.5-bit Q6_K.

**Against llama.cpp, by context length.** llama.cpp is a git submodule pinned to the build these comparisons use (`third_party/llama.cpp`, build 6868a52); `tools/llama_cpp.sh` builds its `llama-bench` and `llama-simple` into `target/llama.cpp/bin` (CPU only, tuned for the machine). `benchmark_decode.py --llama-bench` then runs `llama-bench` on the same file, thread count, prompt length and number of generated tokens, and prints it as one more row (`--prompt-tokens N` makes the prompt N random tokens, as llama-bench's are; `--llama-bench PATH` takes another build):

```bash
tools/llama_cpp.sh   # checks out the pinned submodule and builds it
python benchmarks/benchmark_decode.py --gguf Llama-3.2-1B-Instruct-Q4_0.gguf --threads 8 \
    --prompt-tokens 512 --tokens 64 --no-numpy --llama-bench
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

**An int8 KV cache.** The 9th word of `cfg` (`--kv-cache int8` in `benchmark_decode.py`) switches the cache to int8 values with an f32 scale per position and KV head: 68 bytes for a position's K (or V) in a head instead of 128 (head_dim 64), so at 8192 tokens a decode step reads 143 MB of cache instead of 268 MB. The standard library's `attn_split_q8` rounds each query head to int8 too and scores 4 values of a position per `vdotu` lane; V is widened to f32, with each position's scale folded into its probability. The same comparison, on LLVM with 8 threads (mean of 2 runs, of 5 at 2048 tokens, alternating the two caches), with llama.cpp on its f16 and its Q8_0 cache (`-ctk q8_0 -ctv q8_0 -fa 1`):

| Context | Decode, bf16 | Decode, int8 | Prefill, bf16 | Prefill, int8 | Decode, llama.cpp f16 / Q8_0 |
|---|---|---|---|---|---|
| 512 tokens | 66.9 tokens/s | 68.6 tokens/s | 908 tokens/s | 992 tokens/s | 59.7 / 59.1 tokens/s |
| 2048 tokens | 62.2 tokens/s | 64.5 tokens/s | 750 tokens/s | 732 tokens/s | 48.9 / 44.7 tokens/s |
| 8192 tokens | 50.9 tokens/s | 57.9 tokens/s | 467 tokens/s | 528 tokens/s | 26.4 / 19.4 tokens/s |

- **Decode is 14% faster at 8192 tokens**, and 3–4% at 512 and 2048, where the cache is a small part of the bytes read. Prefill at 8192 is 13% faster; at shorter lengths the difference is within its noise.
- llama.cpp gets slower with its Q8_0 cache (and prefills 81 tokens/s at 8192, against 271 with f16).
- **Accuracy:** the kernel's tokens match NumPy doing the same rounding (24 of 24 on the default prompt, which both caches continue with the same text). Against the bf16 cache, a 300-token chat answer ("Explain how a compiler turns source code into machine code, step by step.") keeps the same first 83 tokens, then continues differently and as coherently. This is a spot check, not a perplexity measurement.
- On Cranelift at 512 tokens the two caches are level (54–58 tokens/s decode, 153–159 prefill, both within run-to-run noise).
- Prefill loads and unpacks each weight vector once per 8 tokens on LLVM (once per 4 before, and still on Cranelift's 128-bit vectors), with the 8 tokens' sums kept in registers across blocks. Measured against the previous kernel the same way, alternating the two, that made 512-token prefill 6% faster on Q4_0, 9% on Q4_K_M and 12% on Q5_K_M. Earlier versions of these tables timed one cold `llama_prefill` call, which came out up to 10% faster or slower than this mean of warm ones.

**K-quants.** Most GGUF downloads are Q4_K_M or Q5_K_M files: Q4_K or Q5_K matrices (256-value super-blocks whose 32-value sub-blocks each have a 6-bit scale and a 6-bit min, so a weight is `scale * q - min`), with the V and down projections of half the layers, and the embedding, in Q6_K. The loader keeps them as the file stores them: the values, each sub-block's 6-bit scale and min (as one 16-bit pair per row), and each super-block's f16 factors, so a Q4_K weight takes 4.625 bits against llama.cpp's 4.5. The kernel multiplies each code by its factor for every strip of rows (exact in f32, as llama.cpp computes it) and subtracts each min times the block's activation sum, which `quantize_q8` already computes. The fused QKV matrix is stored in parts, keeping Q and K in Q4_K and V in Q6_K. The same comparison at 512 tokens (Q4_0 from the table above; the K-quant files the mean of two runs, with llama.cpp run alongside):

| File | Prefill, AIR on LLVM | Prefill, llama.cpp | Decode, AIR on LLVM | Decode, llama.cpp |
|---|---|---|---|---|
| Q4_0 (773 MB) | 1036 tokens/s | 437 tokens/s | 67.4 tokens/s | 60.7 tokens/s |
| Q4_K_M (808 MB) | 920 tokens/s | 501 tokens/s | 62.1 tokens/s | 59.0 tokens/s |
| Q5_K_M (912 MB) | 841 tokens/s | 821 tokens/s | 54.9 tokens/s | 52.8 tokens/s |

- **Decode stays ahead** by 5% on Q4_K_M and 4% on Q5_K_M, both bound by memory bandwidth, although the kernel reads 3–4% more bytes per token (0.83 and 0.94 GB of packed weights): its Q6_K matrices keep an f16 scale per 16 values, and the codes take 2 bytes per row and block instead of 1.5.
- **Keeping the 6-bit codes** instead of an f16 scale and min per block (0.87 and 0.98 GB) made decode on LLVM 3% faster on both files, alternating the two layouts. Cranelift, whose decode is bound by its instructions rather than by bandwidth, pays for the multiplications: its decode got 6% and 2% slower. Prefill, which scales each strip once for 8 tokens (4 on Cranelift), changed by 1–6% on both backends, within its run-to-run spread.
- **Prefill is 1.8x faster on Q4_K_M and level on Q5_K_M**, where llama.cpp is faster than on its own Q4_K_M. Prefill is compute-bound: the int8 dots take half of the inner loop, and scaling each block's sums for every token much of the rest, while unpacking Q5_K's top bits for 8 tokens at a time costs little.
- Cranelift prefills 147 and 132 tokens/s and decodes 49.5 and 41.3 tokens/s on the two files.
- The kernel's tokens match NumPy's (`--prompt "The capital of France is"`: 24 of 24 on both files); on random prompts they part only at near-ties, as on Q4_0. PyTorch runs Q4_K on its int4 kernel with the min as its zero point (rounded to bf16) and Q5_K requantized to int8 per row; it decodes Q4_K_M at 30–33 tokens/s.

**IQ4_NL.** IQ4_NL files (GGUF type 20) store 4-bit codes into a fixed 16-value int8 codebook (llama.cpp's `kvalues_iq4nl`, −127 to 113, spaced more finely near zero than Q4_0's uniform steps) with an f16 scale per 32 values: Q4_0's size, closer to the weights' distribution. The loader keeps the code bytes and scales bit for bit, and the kernel turns each code into a weight with one `vtbl` per weight vector, into the codebook plus 128, so the bytes are Q8_0's unsigned weights and everything after the lookup is Q8_0's arithmetic. Llama 3.2 1B Instruct in IQ4_NL ([unsloth's file](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-GGUF), 773 MB: IQ4_NL matrices, V in Q5_K in some layers, a Q6_K embedding) at 512 tokens, the mean of two runs with Q4_0 and llama.cpp run alongside:

| File | Prefill, AIR on LLVM | Prefill, llama.cpp | Decode, AIR on LLVM | Decode, llama.cpp |
|---|---|---|---|---|
| Q4_0 | 1000 tokens/s | 440 tokens/s | 67.0 tokens/s | 60.5 tokens/s |
| IQ4_NL | 1001 tokens/s | 468 tokens/s | 67.1 tokens/s | 59.4 tokens/s |

- **IQ4_NL runs as fast as Q4_0** on LLVM: the lookup takes no measurable time in decode, which reads the same 0.78 GB of weights, or in prefill. On a 4096x4096 GEMV the two take 48 and 51 µs with the weights in cache, and 180 µs each when they stream from memory.
- **Against llama.cpp:** 2.1x the prefill and 13% faster decode.
- Cranelift prefills 155 tokens/s (Q4_0: 165) and decodes 55.9 tokens/s (57.5).
- The kernel's tokens match NumPy's (24 of 24 on the default prompt, both backends).

**MXFP4.** gpt-oss stores its experts in MXFP4 (GGUF type 39): per 32 values, 4-bit E2M1 codes (0, 0.5, 1, 1.5, 2, 3, 4, 6 and their negatives) and an E8M0 scale byte e, a power of two. llama.cpp keeps the codebook as twice the E2M1 values, the int8s 0, 1, 2, 3, 4, 6, 8, 12 and negatives (`kvalues_mxfp4`), with the scale 2^(e − 128), so MXFP4 is IQ4_NL with another codebook and the kernel runs IQ4_NL's code on it: `qmat_ntable` builds the `vtbl` table from the format's codebook. Its scales are kept as f16s, which hold every power of two from 2^−24 to 2^15 exactly, so the loader converts MXFP4 tensors bit for bit; a matrix with a scale outside that range (none in gpt-oss) becomes Q8_0. On a random model with Llama 3.2 1B's shapes (512-token prompt, 64 tokens, 8 threads, two runs each), MXFP4 runs at IQ4_NL's speed: on LLVM 896–943 tokens/s of prefill and 72.0–73.6 tokens/s of decode (IQ4_NL: 897–934 and 73.3–73.7; Q4_0: 852–972 and 73.6–75.0), on Cranelift 142–148 and 56.2–57.0 (IQ4_NL: 143–145 and 55.9–56.1).

**Qwen3.** Qwen3's dense models are Llama's architecture with one addition: an RMSNorm of every query and key head (over its `head_dim` values, with learned weights) before RoPE. Each layer's entry in the model table holds two more words for those weights (0 for none, as in Llama), and `qkv_head` applies them in the QKV pass (`qk_norm`), as `llama_prefill` does for each token, so a prefill still equals decoding bit for bit. The loader reads `qwen3` GGUF files: their Q and K rows are already in the half-split order the kernel's RoPE uses (NeoX RoPE, so no un-permuting), the embedding doubles as the LM head (tied), RoPE's base is 10^6, and the tokenizer takes Qwen's pre-tokenizer (digits one at a time) and adds no BOS token. `benchmark_decode.py --chat` writes Qwen3's chat format in its non-thinking mode (an empty `<think>` block). The attention width may differ from the model's: Qwen3-4B has 32 query heads of 128 (4096) over d = 2560.

```bash
curl -LO https://huggingface.co/Qwen/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q8_0.gguf
python benchmarks/benchmark_decode.py --gguf Qwen3-1.7B-Q8_0.gguf --chat \
    --prompt "What is the capital of France? Answer in one sentence." --threads 8 --tokens 24
# The capital of France is Paris.
```

Its tokens match the NumPy implementation on both backends for the chat prompts above, for Qwen3-1.7B in Q8_0 and Qwen3-4B in Q4_K_M (Qwen's files; the 4B one has Q4_K matrices, V and the down projection in Q6_K in half the layers, a Q6_K embedding shared with the LM head). A prompt of 512 tokens then 64 generated, 8 threads, against llama.cpp build 6868a52 (`llama-bench -t 8 -p 512` for the prefill, `-n 64 -d 512` for the decode; the AIR rows run separately, the same day):

| | Prefill | Decode | weights read per token |
|---|---|---|---|
| Qwen3-1.7B Q8_0, AIR on LLVM | 588 tokens/s (1.70 ms/token) | 26.4 tokens/s (37.8 ms/token) | 1.83 GB at 48 GB/s |
| Qwen3-1.7B Q8_0, AIR on Cranelift | 121 tokens/s (8.29 ms/token) | 22.0 tokens/s (45.4 ms/token) | 1.83 GB at 40 GB/s |
| Qwen3-1.7B Q8_0, llama.cpp | 171 tokens/s (5.84 ms/token) | 24.8 tokens/s (40.3 ms/token) | |
| Qwen3-4B Q4_K_M, AIR on LLVM | 204 tokens/s (4.90 ms/token) | 19.4 tokens/s (51.7 ms/token) | 2.59 GB at 50 GB/s |
| Qwen3-4B Q4_K_M, AIR on Cranelift | 23.7 tokens/s (42.3 ms/token) | 14.2 tokens/s (70.4 ms/token) | 2.59 GB at 37 GB/s |
| Qwen3-4B Q4_K_M, llama.cpp | 126 tokens/s (7.94 ms/token) | 17.2 tokens/s (58.0 ms/token) | |

On LLVM the kernel prefills 3.4x (1.7B, Q8_0) and 1.6x (4B, Q4_K_M) faster than llama.cpp, and decodes 6% and 13% faster; decode runs at the memory's bandwidth, as for Llama.

Greedy text can part from llama.cpp's where two tokens come within a few hundredths of a logit. Continuing "The capital of France is", Qwen3-1.7B's NumPy implementation and llama.cpp both write " Paris. The capital of the United States", but " the" leads " Italy" there by 0.08 logits (of about 21), and later " States" leads " Kingdom" by 0.02: the kernel's hidden state, 0.2 to 0.4% of its scale from NumPy's (an int8 rounding of an activation landing one step apart, as the tests allow), takes " Italy" on LLVM and " Kingdom" on Cranelift.

**Mixtures of experts (Qwen3-MoE).** In a mixture-of-experts model each layer's MLP is many small SwiGLU MLPs, the experts, of which a router picks a few per token. `llama_decode` runs Qwen3-MoE's: the router's logits (a matrix of one row per expert), their softmax, the `n_used` experts with the largest probabilities (the first on ties), and their probabilities as weights, renormalized to sum to 1 (Qwen3's `norm_topk_prob`; clamped at 2^-14 as llama.cpp does). `moe_ffn` runs the router serially, then one `par` over the picked experts' gate and up rows (64 rows of one expert per task, the dense MLP's `gate_up` with SwiGLU and Q8 quantization) and one over the output rows, each task adding the picked experts' down projections times their weights in the order they were picked, then the residual. `cfg` gains 4 words (the experts, the experts used per token, an expert's MLP width, renormalization) and each layer's entry 2 (the router, and a table of each expert's gate/up and down matrices); a dense layer has 0s there. The loader reads `qwen3moe` GGUF files, whose experts are 3-D tensors (`ffn_gate_exps` and the like, one expert's rows after another's): each expert becomes a matrix of its own in the file's format, and the F32 router becomes Q8_0, as the activations it multiplies are int8. `llama_prefill` runs each token's experts on its own, so a prefill still equals decoding bit for bit; grouping tokens by expert to read each expert's weights once per group is next.

The kernel matches an f64 reference with mixed expert formats at every ISA level on both backends (`qwen3_moe_matches_reference`), and the NumPy implementation of a `qwen3moe` GGUF. It also matches llama.cpp: on a synthetic `qwen3moe` file (2 layers, 64 experts of which 4 are used, random weights; Qwen3-30B-A3B itself is 18 GB in Q4_K_M, more than this machine's memory), `llama-simple` and the kernel on both backends generate the same 16 tokens greedily, including the two steps where the top two logits are 0.01 and 0.04 apart.

Speed at Qwen3-30B-A3B's layer shapes (d 2048, 32 query heads over 4 KV heads of 128, 128 experts of 768 of which 8 are used, a 151936-token vocabulary), on a random Q4_0 file of 8 such layers (3 GB; the model's 48 do not fit this machine), a prompt of 512 tokens then 64 generated, 8 threads, measured one after the other:

| | Prefill | Decode (after the prompt) | weights read per token |
|---|---|---|---|
| AIR on LLVM | 421 tokens/s (2.37 ms/token) | 113 tokens/s (8.84 ms/token) | 0.43 GB at 49 GB/s |
| AIR on Cranelift | 198 tokens/s (5.06 ms/token) | 84.8 tokens/s (11.8 ms/token) | 0.43 GB at 37 GB/s |
| llama.cpp (build 6868a52, `llama-bench -t 8 -p 512`, `-n 64 -d 512`) | 500 tokens/s (2.00 ms/token) | 86.1 tokens/s (11.6 ms/token) | |

Decode reads the used experts' weights at the memory's bandwidth on LLVM, 1.31x llama.cpp's rate after 512 tokens. Prefill (both from an empty cache) is 0.84x llama.cpp's, because each token runs its own experts while llama.cpp groups the prompt's tokens by expert, reading each expert's weights once for all the tokens that picked it.

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
