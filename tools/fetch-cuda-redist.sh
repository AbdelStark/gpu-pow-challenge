#!/usr/bin/env bash
# Fetch a minimal CUDA toolchain (nvcc + cudart + cuBLAS headers/libs) from
# NVIDIA's public redistributables into $1 (default: .cuda). Used by CI to
# compile-check solvers and run tools/cuda-host-test on machines without a GPU.
set -euo pipefail
DEST="${1:-.cuda}"
BASE=https://developer.download.nvidia.com/compute/cuda/redist
PKGS=(
  "cuda_nvcc/linux-x86_64/cuda_nvcc-linux-x86_64-12.8.93-archive"
  "cuda_cudart/linux-x86_64/cuda_cudart-linux-x86_64-12.8.90-archive"
  "cuda_cccl/linux-x86_64/cuda_cccl-linux-x86_64-12.8.90-archive"
  "libcublas/linux-x86_64/libcublas-linux-x86_64-12.8.4.1-archive"
)
mkdir -p "$DEST"
if [[ -x "$DEST/bin/nvcc" ]]; then echo "cuda redist already in $DEST"; exit 0; fi
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
for p in "${PKGS[@]}"; do
  echo "fetching $(basename "$p")"
  curl -fsSL "$BASE/$p.tar.xz" -o "$tmp/pkg.tar.xz"
  tar -xJf "$tmp/pkg.tar.xz" -C "$DEST" --strip-components=1
  rm -f "$tmp/pkg.tar.xz"
done
[[ -d "$DEST/lib64" ]] || ln -s lib "$DEST/lib64"
"$DEST/bin/nvcc" --version | tail -1
