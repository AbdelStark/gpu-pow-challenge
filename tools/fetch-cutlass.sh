#!/usr/bin/env bash
# Fetch the pinned header-only CUTLASS/CuTe into third_party/cutlass and verify
# its commit. Solvers may use it; no other third-party code may be added.
set -euo pipefail
cd "$(dirname "$0")/.."

TAG="v3.9.2"
COMMIT="ad7b2f5e84fcfa124cb02b91d5bd26d238c0459e"
DEST="third_party/cutlass"

if [[ ! -d "$DEST/include" ]]; then
  mkdir -p third_party
  git -c advice.detachedHead=false clone --quiet --depth 1 --branch "$TAG" \
    https://github.com/NVIDIA/cutlass "$DEST"
fi
got="$(git -C "$DEST" rev-parse HEAD)"
[[ "$got" == "$COMMIT" ]] || { echo "fetch-cutlass.sh: $DEST is at $got, expected $COMMIT" >&2; exit 1; }
echo "cutlass $TAG in $DEST"
