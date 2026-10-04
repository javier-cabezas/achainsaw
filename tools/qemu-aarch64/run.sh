#!/usr/bin/env bash
# Builds the `mm` kernels for AArch64 with the LLVM backend, links them into a freestanding
# harness, and runs it under QEMU at several SVE/SME vector lengths. Lets x86 hosts test
# SVE and SME code paths without ARM hardware or a cross C toolchain.
#
# Needs: an LLVM-enabled `achainsaw` build, `rustup target add aarch64-unknown-linux-gnu`,
# and qemu-user (`qemu-aarch64`, or set QEMU to another binary). SME needs QEMU 9.0 or
# newer: 8.2 mis-emulates some ZA tile accesses.
#
# Usage: tools/qemu-aarch64/run.sh [target-features] [vector lengths in bytes...]
#   tools/qemu-aarch64/run.sh +sve 16 32 64 256       # vector FMA kernel on SVE
#   tools/qemu-aarch64/run.sh +sve,+sme 16 32 64 256  # SME kernel (streaming VL swept too)
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
features="${1:-+sve}"
shift || true
lengths=("${@:-16 32 64 256}")
lengths=(${lengths[@]})
out="${TMPDIR:-/tmp}/achainsaw-qemu-aarch64"
mkdir -p "$out"

achainsaw="${ACHAINSAW:-$root/target/debug/achainsaw}"
"$achainsaw" --backend llvm build "$here/mm.air" --target aarch64-unknown-linux-gnu \
  --target-features "$features" -o "$out/mm.o" >/dev/null

sysroot="$(rustc --print sysroot)"
lld="$(find "$sysroot" -name rust-lld -type f | head -1)"
rustc --edition 2021 --target aarch64-unknown-linux-gnu -O -C panic=abort \
  -C linker="$lld" -C linker-flavor=ld.lld -C relocation-model=static \
  -C link-arg=-static -C link-arg="$out/mm.o" \
  "$here/mm_harness.rs" -o "$out/mm_harness"

status=0
for vl in "${lengths[@]}"; do
  echo "== ${features}: vector length $((vl * 8)) bits"
  "${QEMU:-qemu-aarch64}" -cpu "max,sve-default-vector-length=$vl,sme-default-vector-length=$vl" \
    "$out/mm_harness" || status=1
done
exit $status
