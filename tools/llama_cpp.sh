#!/usr/bin/env bash
# Builds the pinned llama.cpp submodule (third_party/llama.cpp, the build docs/llama.md
# compares against) for the performance comparisons: llama-bench and llama-simple, CPU
# only, tuned for this machine (GGML_NATIVE). The binaries land in target/llama.cpp/bin,
# where `benchmarks/benchmark_decode.py --llama-bench` finds them.
#
# Needs git, cmake and a C++ compiler. Usage: tools/llama_cpp.sh
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
src="$root/third_party/llama.cpp"
out="$root/target/llama.cpp"
# Checks out the pinned commit, also over a checkout of another one.
git -C "$root" submodule update --init third_party/llama.cpp
jobs="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"
# Metal is on by default on macOS, and llama-bench would offload every layer to the GPU.
cmake -S "$src" -B "$out" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_METAL=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF >/dev/null
cmake --build "$out" -j "$jobs" --target llama-bench llama-simple
echo "built $(git -C "$src" rev-parse --short HEAD): $out/bin/llama-bench, $out/bin/llama-simple"
