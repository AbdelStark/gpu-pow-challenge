#!/usr/bin/env bash
# Compile a solver with the fixed ranked flags. Used by benchmark.sh and CI, so
# what CI checks is exactly what the benchmark builds.
#
#   tools/build-solver.sh <solver-dir> <cuda-arch> <output>
#
# Env: NVCC  path to nvcc (default: PATH). Paths are relative to the repo root.
set -euo pipefail
cd "$(dirname "$0")/.."

[[ $# -eq 3 ]] || { echo "usage: tools/build-solver.sh <solver-dir> <cuda-arch> <output>" >&2; exit 2; }
SRC="$1" ARCH="$2" OUT="$3"
NVCC="${NVCC:-$(command -v nvcc || true)}"
[[ -n "$NVCC" ]] || { echo "build-solver.sh: nvcc not found" >&2; exit 1; }
[[ -d third_party/cutlass/include ]] || { echo "build-solver.sh: run tools/fetch-cutlass.sh first" >&2; exit 1; }

mapfile -t SOURCES < <(find "$SRC" -maxdepth 1 -type f \( -name '*.cu' -o -name '*.cpp' \) | LC_ALL=C sort)
[[ ${#SOURCES[@]} -gt 0 ]] || { echo "build-solver.sh: no .cu/.cpp sources in $SRC" >&2; exit 1; }

echo "-- building solver from $SRC (${#SOURCES[@]} files, -arch=$ARCH)"
"$NVCC" -O3 -std=c++17 -arch="$ARCH" -lineinfo \
  -I worker -I "$SRC" -I third_party/cutlass/include -I third_party/cutlass/tools/util/include \
  --expt-relaxed-constexpr \
  worker/main.cpp "${SOURCES[@]}" \
  -L "$(dirname "$(dirname "$NVCC")")/lib64" -lcublasLt -lcublas \
  -o "$OUT"
