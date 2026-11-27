#!/usr/bin/env bash
# Idempotent environment setup for the TPoW benchmark.
#
#   ./setup.sh [--track <name>]      (default track: h100-sxm)
#
# Ensures: a C toolchain, bubblewrap (sandbox), the pinned Rust toolchain, the
# trusted harness built offline-ready, and, for GPU tracks, nvcc + nvidia-smi.
set -euo pipefail
cd "$(dirname "$0")"

TRACK="${TPOW_TRACK:-h100-sxm}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --track) TRACK="$2"; shift 2 ;;
    --track=*) TRACK="${1#--track=}"; shift ;;
    *) echo "setup.sh: unknown argument $1" >&2; exit 2 ;;
  esac
done
LB="leaderboards/${TRACK}.json"
[[ -f "$LB" ]] || { echo "setup.sh: no leaderboard config $LB" >&2; exit 2; }

SUDO=""
if [[ ${EUID:-$(id -u)} -ne 0 ]] && command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then SUDO="sudo -n"; fi

apt_install() {
  if command -v apt-get >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    $SUDO apt-get update -qq && $SUDO apt-get install -y -qq --no-install-recommends "$@"
  elif command -v dnf >/dev/null 2>&1; then
    $SUDO dnf install -y "$@"
  else
    return 1
  fi
}

# 1. C toolchain, jq, curl, bubblewrap.
need=()
command -v cc >/dev/null 2>&1 || need+=(gcc libc6-dev)
command -v jq >/dev/null 2>&1 || need+=(jq)
command -v curl >/dev/null 2>&1 || need+=(curl ca-certificates)
command -v bwrap >/dev/null 2>&1 || need+=(bubblewrap)
command -v flock >/dev/null 2>&1 || need+=(util-linux)
if [[ ${#need[@]} -gt 0 ]]; then
  apt_install "${need[@]}" || echo "setup.sh: warning: could not install ${need[*]}" >&2
fi

# 2. Rust (pinned by rust-toolchain.toml).
# shellcheck disable=SC1091
. "$HOME/.cargo/env" 2>/dev/null || true
if ! command -v cargo >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain none
  # shellcheck disable=SC1091
  . "$HOME/.cargo/env"
fi
rustup show active-toolchain >/dev/null   # installs the pinned toolchain if missing

# 3. Trusted harness + reference tools; after this, benchmark.sh runs offline.
cargo fetch --locked
cargo build --release --locked --bin tpow-bench --bin tpow-cpu-solver --bin tpow-kat
./target/release/tpow-kat check tests/vectors/tpow-tiny.json

# 4. GPU tracks need nvcc and a driver.
ARCH="$(jq -r '.cuda_arch // empty' "$LB")"
if [[ -n "$ARCH" ]]; then
  NVCC="${NVCC:-$(command -v nvcc || true)}"
  [[ -z "$NVCC" && -x /usr/local/cuda/bin/nvcc ]] && NVCC=/usr/local/cuda/bin/nvcc
  if [[ -z "$NVCC" ]]; then
    echo "setup.sh: nvcc not found. Install the CUDA toolkit (>= 12.4) or set NVCC=/path/to/nvcc." >&2
    exit 1
  fi
  "$NVCC" --version | tail -1
  # Header-only CUTLASS/CuTe, pinned by commit, available to solvers as
  # -I third_party/cutlass/include (no other third-party code may be added).
  CUTLASS_TAG="v3.9.2"
  CUTLASS_COMMIT="ad7b2f5e84fcfa124cb02b91d5bd26d238c0459e"
  if [[ ! -d third_party/cutlass/include ]]; then
    mkdir -p third_party
    git -c advice.detachedHead=false clone --quiet --depth 1 --branch "$CUTLASS_TAG" \
      https://github.com/NVIDIA/cutlass third_party/cutlass
  fi
  got="$(git -C third_party/cutlass rev-parse HEAD)"
  [[ "$got" == "$CUTLASS_COMMIT" ]] || { echo "setup.sh: third_party/cutlass is at $got, expected $CUTLASS_COMMIT" >&2; exit 1; }
  command -v nvidia-smi >/dev/null 2>&1 || { echo "setup.sh: nvidia-smi not found (no NVIDIA driver?)" >&2; exit 1; }
  nvidia-smi --query-gpu=index,name,driver_version,persistence_mode,power.limit,power.default_limit --format=csv
  if nvidia-smi --query-gpu=persistence_mode --format=csv,noheader | grep -qi disabled; then
    echo "setup.sh: note: persistence mode is off; 'sudo nvidia-smi -pm 1' gives more stable timings" >&2
  fi
fi
echo "setup.sh: ready for track $TRACK"
