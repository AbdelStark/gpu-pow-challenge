#!/usr/bin/env bash
# Build the contestant solver and run the trusted harness for one track.
#
#   ./benchmark.sh [--track <name>] [--note "what I tried"] [extra tpow-bench args]
#
# Tracks: leaderboards/<name>.json (h100-sxm, h200-sxm, h100-sxm-gpucpu, local-gpu, dev-cpu).
# Env:    TPOW_TRACK       default track (h100-sxm)
#         TPOW_SOLVER_SRC  solver source dir (default solvers/<track>, else solvers/h100-sxm)
#         NVCC             path to nvcc (default: PATH, then /usr/local/cuda/bin/nvcc)
#         TPOW_GPU_INDEX   GPU index (default 0)
#
# Steps:
#   1. clean stale outputs (a submission can never pre-seed a score)
#   2. build the TRUSTED harness offline (never links solver code)
#   3. build the UNTRUSTED solver with fixed flags: worker/main.cpp + solver dir
#   4. best-effort public beacon (drand quicknet) for auditability
#   5. take the box-wide GPU lock and run tpow-bench, which sandboxes the solver,
#      times it externally, audits outputs on the CPU and writes the score.
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd -P)"

TRACK="${TPOW_TRACK:-h100-sxm}"
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --track) TRACK="$2"; shift 2 ;;
    --track=*) TRACK="${1#--track=}"; shift ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
LB="leaderboards/${TRACK}.json"
[[ -f "$LB" ]] || { echo "benchmark.sh: no leaderboard config $LB" >&2; exit 2; }
OUT="results/${TRACK}"
BUILD="build/${TRACK}"
mkdir -p "$OUT" "$BUILD"

# 1. Clean slate.
rm -f "$OUT/score.json" "$OUT/run.json" "$BUILD/solver" "$BUILD/beacon.json"

# shellcheck disable=SC1091
. "$HOME/.cargo/env" 2>/dev/null || true
export CARGO_NET_OFFLINE=true

# 2. Trusted harness.
cargo build --release --locked --offline --bin tpow-bench --bin tpow-cpu-solver

# 3. Untrusted solver.
ARCH="$(jq -r '.cuda_arch // empty' "$LB")"
if [[ -z "$ARCH" ]]; then
  SOLVER="$ROOT/target/release/tpow-cpu-solver"      # CPU-only development track
else
  SRC="${TPOW_SOLVER_SRC:-solvers/${TRACK}}"
  [[ -d "$SRC" ]] || SRC="solvers/h100-sxm"
  NVCC="${NVCC:-$(command -v nvcc || true)}"
  [[ -z "$NVCC" && -x /usr/local/cuda/bin/nvcc ]] && NVCC=/usr/local/cuda/bin/nvcc
  [[ -n "$NVCC" ]] || { echo "benchmark.sh: nvcc not found (run ./setup.sh)" >&2; exit 1; }
  mapfile -t SOURCES < <(find "$SRC" -maxdepth 1 -type f \( -name '*.cu' -o -name '*.cpp' \) | LC_ALL=C sort)
  [[ ${#SOURCES[@]} -gt 0 ]] || { echo "benchmark.sh: no .cu/.cpp sources in $SRC" >&2; exit 1; }
  CUDA_LIB="$(dirname "$(dirname "$NVCC")")/lib64"
  echo "-- building solver from $SRC (${#SOURCES[@]} files, -arch=$ARCH)"
  "$NVCC" -O3 -std=c++17 -arch="$ARCH" -lineinfo \
    -I worker -I "$SRC" -I third_party/cutlass/include -I third_party/cutlass/tools/util/include \
    --expt-relaxed-constexpr \
    worker/main.cpp "${SOURCES[@]}" \
    -L "$CUDA_LIB" -lcublasLt -lcublas \
    -o "$BUILD/solver"
  SOLVER="$ROOT/$BUILD/solver"
  # The sandbox hides /home and /root; expose the CUDA libs the solver links.
  export TPOW_WORKER_LD_LIBRARY_PATH="${TPOW_WORKER_LD_LIBRARY_PATH:-$CUDA_LIB}"
fi

# 4. Public randomness beacon (best effort; harness randomness is always mixed in).
BEACON_ARGS=()
if curl -fsS --max-time 5 \
     https://api.drand.sh/52db9ba70e0cc0f6eaf7803dd07447a1f5477735fd3f661792ba94600c84e971/public/latest \
     -o "$BUILD/beacon.json" 2>/dev/null; then
  BEACON_ARGS=(--beacon "$BUILD/beacon.json")
fi

# 5. Run under the box-wide GPU lock (serialises every GPU job on this host).
LOCK="/tmp/tpow-gpu-${TPOW_GPU_INDEX:-0}.lock"
exec 9>"$LOCK"
flock 9
./target/release/tpow-bench --leaderboard "$LB" --solver "$SOLVER" --out-dir "$OUT" \
  "${BEACON_ARGS[@]}" "${ARGS[@]}"
