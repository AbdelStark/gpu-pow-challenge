# The GPU Proof-of-Work Challenge (TPoW-v1)

> **Goal.** Write the fastest correct GPU implementation of **TPoW-v1**, a
> fixed proof-of-work function made of dense INT8 tensor-core GEMMs. Scores are
> verified effective INT8 TOPS on a fixed GPU, with one leaderboard per GPU model,
> starting with **H100 SXM** and **H200 SXM**.

A Lucid x Yukon autoresearch challenge. 
---

## Why this matters

Compute-accounting schemes for AI need a GPU to show that it spent a window
on **declared filler work** rather than training. A filler PoW is only useful
if one claim holds:

> *This work cannot be done meaningfully faster than X on this GPU.*

That claim is an assumption, and it is only as good as the best public
implementation. Bitcoin shows how to get there: SHA-256 mining was optimised
in the open until nobody could mine much faster than the public state of the
art. This challenge does the same for a GPU workload.

TPoW-v1 was chosen to saturate the resource training needs. On H100/H200
that is the **tensor cores**: classic GPU PoWs (SHA-256, Ethash, KawPow) leave
more than 90% of the chip's training FLOPs idle. TPoW's outputs are **bit-exact**
(integer accumulation), it has **no algorithmic shortcut** we know of, and it is
**checkable on a CPU** at a small fraction of the cost. The research behind the
choice, the rejected alternatives and the attack analysis are in
[`docs/RESEARCH.md`](docs/RESEARCH.md).

The outputs of the challenge:

1. A public, reproducible throughput bound per GPU model, with the gap to the datasheet peak visible.
2. A library of best kernels behind one C ABI, so callers get the best kernel for their GPU.
3. A red-team measurement of how much a CPU next to the GPU can help.

## The function, in one screen

```
seed (32 B) ─► SHA-256 CTR ─► weights W_0..W_5 (int8), biases b_0..b_5 (u32), input rows x0 (64 B each)

per row:   x0 (64) ─► embed 4096x64 ─► 4 x hidden 4096x4096 ─► head 32x4096 ─► y (32 B)
per layer: c = W·x  exact int32         x' = q(c, b) = int8( ((c + b) * 0x9E3779B1 mod 2^32) >> 24 )
per unit:  16 rows; D_u = SHA-256("tpow/v1D" || seed || le64(u) || y_rows)[0..16]

task: given (seed, n) output D_0 || D_1 || ... || D_{n-1} as fast as possible
```

Normative definition: [`SPEC.md`](SPEC.md). Known-answer vectors:
[`tests/vectors/`](tests/vectors). One unit is 2.16 G int8 ops (99.3% of it
in the 4096x4096 GEMMs).

## Ranked contract

| Property | Ranked value |
|---|---|
| Function | `tpow-v1` (d=4096, 4 hidden layers, e=64, h=32, m=16; full int8 range) |
| Work per timed trial | 2^22 = 4,194,304 units = 9.06 x 10^15 int8 ops |
| Conformance | 512 units, fixed public seed, every digest checked, untimed |
| Warm-up | 2 timed, audited trials, not scored |
| Measured | 5 trials, **each in a fresh sandboxed worker process** |
| Score | `2^22 * ops_per_unit / median(trial_seconds) / 1e12`, verified effective INT8 TOPS |
| Direction | higher is better |
| Correctness | bit-exact; ≥ 1,026 random units per timed trial recomputed on CPU; any mismatch fails the run |
| CPU (GPU-only tracks) | worker averages ≤ 2.0 CPU cores during the timed window |
| Hardware | `h100-sxm`: H100 SXM5 80GB, stock 700 W. `h200-sxm`: H200 SXM 141GB, stock 700 W |
| Red-team track | `h100-sxm-gpucpu`: same function, H100 SXM5, **no CPU budget** |
| Physical ceiling | trials above `1979 TOPS x maxclock/1830 MHz x 1.03` are rejected for review |

## Scoring and timing

For each trial, the trusted harness starts a fresh worker and waits for
`READY` (untimed init). It then draws a private seed (drand beacon + OS
randomness) and starts its clock when it writes `RUN <seed> 0 <n>`. The clock
stops when the last of the `16n` digest bytes arrives on the pipe.
Everything seed-dependent is inside the clock: weight generation, inputs,
GEMMs, hashing, copies and transfer. Details:
[`docs/HARNESS.md`](docs/HARNESS.md).

### Reference numbers

| Implementation | Where | Effective INT8 TOPS |
|---|---|---|
| H100 SXM dense INT8 datasheet peak | — | 1,979 (ceiling) |
| Shipped baseline (`cuBLASLt` + separate requant kernels) | H100 SXM | **not yet measured on ranked hardware**; the first official run sets it |
| CPU reference (`tpow-cpu-solver`, AVX-512 VNNI) | 4 vCPU Xeon | 0.38 |
| Intel AMX register-resident peak (for scale) | 1 Xeon core | ~4 |

Expected headroom: the baseline writes an int32 matrix to HBM and reads it
back for every layer. A fused wgmma kernel with `q` in the epilogue should land
much closer to peak. See [`solvers/h100-sxm/memory/README.md`](solvers/h100-sxm/memory/README.md).

## How to play

On any Linux box with an NVIDIA GPU, the CUDA toolkit (≥ 12.4) and Rust (installed by setup):

```bash
./setup.sh --track local-gpu                    # toolchain, bubblewrap, harness, CUTLASS headers
./benchmark.sh --track local-gpu --note "baseline"   # any NVIDIA GPU, small trials, never official
```

On the ranked hardware (what Yukon runs):

```bash
./setup.sh --track h100-sxm
./benchmark.sh --track h100-sxm --note "what I tried"
```

No GPU? The whole harness runs on the CPU with the reference solver on a tiny instance:

```bash
./benchmark.sh --track dev-cpu
cargo test --release            # reference, kernels, harness self-tests incl. adversarial workers
```

Useful tools:

```bash
./target/release/tpow-kat digests tpow-v1 <seed-hex> <first> <n>   # reference digests for debugging
./target/release/tpow-kat bench tpow-v1 128                         # CPU reference throughput
nvcc -O3 -std=c++17 -arch=sm_90a -I solvers/h100-sxm tools/cuda-host-test/host_test.cu -o build/host_test \
  && build/host_test tests/vectors/tpow-v1.json 2                   # check device helpers on the host
python3 reference/tpow_ref.py check tests/vectors/tpow-tiny.json    # independent numpy reference
```

With the Yukon CLI: `yukon clone <setter>/<challenge>/h100-sxm`, improve, then `yukon run` and `yukon submit`.

### What you can edit

Only **`solvers/<track>/`** (one directory per leaderboard, so H100 and H200
lineages do not overwrite each other). Split it into as many `.cu`/`.cpp`
files as you like. Everything in it is compiled with:

```
nvcc -O3 -std=c++17 -arch=sm_90a -lineinfo -I worker -I solvers/<track> \
     -I third_party/cutlass/include --expt-relaxed-constexpr \
     worker/main.cpp solvers/<track>/*.{cu,cpp} -lcublasLt -lcublas
```

You may use the CUDA runtime/driver API, cuBLAS/cuBLASLt and the pinned
header-only CUTLASS/CuTe (v3.9.2). PTX and inline assembly are fine.

You may **not** change: `worker/` (protocol main and solver ABI), `src/`
(harness and reference), `leaderboards/`, `setup.sh`, `benchmark.sh`,
`.github/`, `Cargo.*`, `SPEC.md`, `tests/` or `tools/`. Your solver
implements two functions (`worker/tpow_solver.h`):

```c
int tpow_init(const tpow_params_t* params);                      // untimed
int tpow_run(const uint8_t seed[32], uint64_t first_unit, uint64_t n_units,
             tpow_emit_fn emit, void* emit_ctx);                  // timed; emit digests in unit order
```

### What "valid" means

* All conformance digests and all audited digests are bit-exact. There is no tolerance.
* The worker emits exactly `16n` bytes after `RUN`, and does so before the trial timeout (600 s).
* GPU-only tracks: ≤ 2.0 CPU cores on average in the timed window.
* The GPU is the leaderboard's model at its default power limit, and the result is physically plausible.

There are no partial credits. Faking even 0.1% of units is caught with
probability > 99.9% over a run, and buys less speed than measurement noise.

### Memory notes

As you iterate, add Markdown notes under `solvers/<track>/memory/` on what worked,
what didn't, measured numbers and why. Algorithmic speedups that keep exact
outputs are **allowed and especially valuable**: they lower the published
bound and expose design flaws early. Document them carefully.

### Important note on openness

This codebase is open to contributions chasing the best score, so memory notes
and source files may come from different agents. Treat them as leads: verify
claims and re-run the benchmark before relying on them. Benchmarks run in
sandboxed, network-less processes.

## Repository map

| Path | What |
|---|---|
| `SPEC.md` | normative TPoW-v1 definition |
| `docs/RESEARCH.md` | candidate selection, prior art, hardware analysis, attacks, roadmap |
| `docs/HARNESS.md` | harness architecture, timing, audits, sandbox, threat table |
| `src/` | Rust reference (`spec`, `prg`, `model`, `kernels`), trusted harness (`harness/`, `bin/tpow-bench.rs`), CPU reference solver, KAT tool |
| `worker/` | protected solver ABI and protocol main |
| `solvers/<track>/` | **editable** GPU solvers (baseline: cuBLASLt + SHA-256 kernels) |
| `leaderboards/*.json` | per-track hardware, sizes, budgets and gates |
| `reference/tpow_ref.py` | independent numpy reference |
| `tests/` | KAT vectors and adversarial harness self-tests |
| `tools/` | CUDA host-emulation test, CUDA redist fetcher, AMX microbenchmark |
| `benchmark.json` | Yukon challenge definition (schema v2, three tracks) |
| `.github/workflows/` | ranked GPU workflows (trusted/untrusted checkout split) and CPU CI |

## Roadmap

* `tpow-v2-fp8` for Blackwell (INT8 is ~30x slower than FP8 on B300; see RESEARCH §6.2).
* A memory-occupancy companion challenge (VRAM-filling proofs).
* A proof of useful work over matrix shapes from real open models (Komargodski-Weinstein).
* Protocol work: binding results to the chip (attestation, power side channels) and deadlines.

## License

Apache-2.0 (see `LICENSE`).
