# Harness design

How `tpow-bench` turns an untrusted CUDA program into a trustworthy
throughput number. The design draws on the existing Yukon challenges
(`ecdsafail`, `flock`, `labinius`, `cudafast`) and on published
GPU-benchmark reward-hacking incidents (§9).

## 1. Components and trust boundaries

```
 TRUSTED (default-branch checkout)                    UNTRUSTED (submission)
 ───────────────────────────────────                  ──────────────────────────
 benchmark.sh  ── nvcc, fixed flags ─────────────────> build/<track>/solver
   │                                                   = worker/main.cpp (protected)
   │                                                   + solvers/<track>/*.cu (editable)
   ▼
 tpow-bench (Rust; never links solver code)
   ├─ hardware checks, quiescence, thermal gate (nvidia-smi)
   ├─ spawns ONE FRESH WORKER PER TRIAL inside bubblewrap ─> solver (stdin/stdout pipes)
   ├─ external clock (writes RUN ... reads last digest byte)
   ├─ CPU audits via the reference implementation (src/model.rs)
   └─ score.json, run.json, results.tsv
```

* The **harness** (`src/bin/tpow-bench.rs`, `src/harness/`) and the reference
  (`src/model.rs`) are built from the trusted checkout. They never import or
  load contestant code.
* `worker/main.cpp` and `worker/tpow_solver.h` are protected. Like `flock`'s worker
  wrapper, they are compiled *into* the untrusted binary for convenience; they are
  not a trust boundary.
* The CI workflow checks out the **default branch** as the trusted tree. It copies
  in only `solvers/<track>/` from the submission, after rejecting symlinks and
  special files and enforcing a size cap. It also re-checks that the submission
  commit touches nothing else (`benchmark-gpu.yml`).

## 2. Worker protocol

```
harness -> worker  stdin   PARAMS d=4096 L=4 e=64 h=32 m=16\n
worker  -> harness stdout  READY tpow-v1\n                 (after tpow_init; untimed)
                                   ... harness gates, draws the private seed ...
harness -> worker  stdin   RUN <seed hex> <first_unit> <n_units>\n   <- clock starts after flush
worker  -> harness stdout  n_units * 16 raw digest bytes   <- clock stops at the last byte
harness                    SIGKILL the worker's process group, reap, wipe scratch
```

The worker must stay alive after its last byte until the harness kills it. A
worker that exits on its own fails the trial, because exiting early could hide
CPU time from the `/proc` accounting (§6).

Contestants implement two C functions (`worker/tpow_solver.h`):
`tpow_init` (untimed: contexts, allocation, autotuning, warm-up with any seed)
and `tpow_run` (timed: everything seed-dependent). `emit()` streams digests in
unit order, so early batches can leave while later ones compute.

## 3. Timing semantics

* **Clock:** `Instant::now()` in the trusted process, from immediately after the
  RUN line is flushed to the worker's stdin, until the reader thread has received
  the final digest byte. Weight generation, input generation, all GEMMs, hashing,
  device-to-host copies and pipe transfer are inside. Process start, CUDA
  context creation and `tpow_init` are outside.
* **Why a pipe and not a file:** a GPU cannot DMA into a pipe. A file could be
  `mmap`ed and `cudaHostRegister`ed, letting the GPU keep writing results after
  the solver said "done". Pipe bytes, once read, are final.
* **Fresh worker per trial** (as in `flock`): no state, cache, warmed buffer or
  background thread survives between trials. The process group is killed, and
  bubblewrap's PID namespace plus `--die-with-parent` take daemons with it.
* **Trials:** 512-unit conformance run (untimed), 2 warm-up trials (timed,
  audited, not scored), 5 measured trials. **Score = median** of measured trial
  TOPS. min, max, p10, p90 and spread are reported alongside.
* **Unit of score:** verified effective INT8 TOPS =
  `n_units * ops_per_unit / seconds / 1e12`, with `ops_per_unit = 2,160,066,560`
  (2 x MACs). It is directly comparable to the datasheet dense INT8 peak, and
  `pct_of_peak_int8_dense` is reported.
* **Size:** 2^22 units = 9.06e15 ops per trial, about 4.6 s at 100% of H100 peak.
  This is long enough to reach thermal/power steady state and make pipe and
  launch overheads negligible, and short enough for a 7-trial run in a few minutes.

## 4. Seeds and the public beacon

```
nonce_t = 32 bytes from /dev/urandom, drawn AFTER the worker printed READY
seed_t  = SHA-256("tpow/v1S" || drand_randomness || le64(drand_round) || nonce_t || le32(t))
```

* The **harness nonce** makes the seed unpredictable to the worker even if the
  beacon is stale or unavailable.
* The **drand quicknet** round, fetched best-effort by `benchmark.sh`, makes inputs
  publicly anchored: nobody, including the organiser, can claim tuned inputs.
* `run.json` publishes the beacon, nonces, seeds and the first audited digests of
  every trial, so anyone can recompute those audits with `tpow-kat digests`.

TPoW's cost is data-oblivious, so input choice cannot change timing except
through power draw, and uniform random bytes are the worst case for power.

## 5. Correctness: conformance and audits

* **Conformance:** a fixed public seed (bytes 0..31, the KAT seed). All 512
  units are checked in full before any timing, with a precise error that names
  the first wrong unit and the `tpow-kat` command to reproduce it.
* **Audits:** after each timed trial, `k = 1024` uniformly random distinct units
  plus the first and the last are recomputed on the CPU (AVX-512 VNNI kernel,
  runtime-detected, ~10 ms per unit per few cores). Indices come from
  `/dev/urandom` *after* the digests have arrived. **Any mismatch fails the whole
  run** (no partial credit). See RESEARCH.md §7 for detection probabilities.
* Three independent implementations (Rust, numpy, CUDA device code emulated on
  the host) agree on all known-answer vectors (`tests/vectors/`), checked in CI.

## 6. Plausibility checks

| Check | Rule | Rationale |
|---|---|---|
| GPU identity | product name must contain / not contain configured strings (e.g. H100, not PCIe/NVL) | one hardware spec per leaderboard |
| Power limit | must equal the board default | no capped or raised power |
| Quiescence | no other compute process, utilisation < 5% before starting | shared-box interference |
| Thermal gate | wait until ≤ 45 °C (max 180 s) before each trial; recorded | run-to-run comparability |
| Physical ceiling | reject if TOPS > peak x (max SM clock / reference clock) x 1.03 | an impossible number means a harness bug or an exploit; it goes to human review (cf. KernelGuard's physics-floor rule) |
| CPU budget | GPU-only tracks: worker tree ≤ 2.0 CPU cores on average in the timed window (utime+stime over the whole process tree from /proc) | keeps the GPU leaderboard a GPU number; the red-team track lifts the cap |
| Box lock | `flock /tmp/tpow-gpu-<idx>.lock` around the whole run | serialises GPU jobs on a host (as in `cudafast`) |

**Clocks are deliberately not locked.** An attacker would run at stock
boost behaviour with the default power limit, so that is what the bound must
describe. Clocks, power, temperature and the OR of throttle reasons are sampled
at 100 ms by `nvidia-smi` and reported per trial.

## 7. Sandbox (bubblewrap)

* `--ro-bind / /`; fresh `--dev /dev` with **only** `/dev/nvidiactl`,
  `/dev/nvidia-uvm`, `/dev/nvidia-uvm-tools` and the selected `/dev/nvidia<minor>`
  (`--dev-bind`; a plain bind breaks NVML). `CUDA_VISIBLE_DEVICES=<GPU UUID>`.
* Fresh `--proc` (still exposes `/proc/driver/nvidia`); private tmpfs on `/tmp`,
  `/dev/shm` and `/run`; `/home`, `/root` and `/var/tmp` masked (`TPOW_SANDBOX_HIDE`).
* The solver binary is copied into a private, mode-1777 scratch directory, its only
  writable path and its cwd/HOME/TMPDIR. The directory is wiped after every trial.
* `--unshare-net --unshare-ipc --unshare-uts --unshare-pid --unshare-cgroup-try`,
  optional `--unshare-user --uid 65534`, `--cap-drop ALL`, `--new-session`,
  `--die-with-parent`, `--clearenv`. The harness also sets `PR_SET_PDEATHSIG`
  and its own process group.
* Self-test: `tests/harness.rs::sandbox_blocks_escapes` runs an adversarial worker
  that tries the network, writes to `/etc` and `/usr`, reads the harness's
  environment and lists `/home`. All must fail.
* Some driver/kernel combinations refuse `cuInit` inside unprivileged user
  namespaces. Then set `"sandbox_user_namespace": false` for that host (the other
  isolation layers remain).
* Operator checklist for the GPU host: dedicated runner, persistence mode on,
  MIG off, no MPS, ECC on, pinned driver, `kernel.apparmor_restrict_unprivileged_userns=0`
  (Ubuntu 24.04) or a setuid bwrap.

## 8. Outputs

* `results/<track>/score.json` (Yukon `scorePath`):
  `{"score": <median TOPS>, "metrics": {pct_of_peak_int8_dense, units_per_second, median_seconds, trial_seconds[], warmup_trial_seconds[], tops_min/max/p10/p90, spread_pct, cpu_cores_max, sm_clock_mhz_median, power_w_median, temp_c_max, throttle_reasons_or[], audited_units, audit_failures, conformance_units, official, sandbox, gpu, ...}}`.
  On failure no `score.json` is written, and the process exits nonzero.
* `results/<track>/run.json`: the full transcript (config, GPU info, solver
  SHA-256, beacon, per-trial nonce/seed/time/CPU/telemetry/audit sample, status, failure reason).
* `results/<track>/results.tsv`: one row per run (local history, like `ecdsafail`).
* `official = leaderboard.official && sandboxed && no CLI overrides`.

## 9. Threats from the GPU-benchmark literature and how they are closed

| Incident / exploit | Mitigation here |
|---|---|
| Sakana "AI CUDA Engineer": eval loophole reused stale results, 100x claims | Fresh seed per trial; fresh process; bit-exact audits of every timed trial |
| KernelBench: no-op kernels reading memory the reference already wrote | The reference never runs on the GPU; verification is CPU-only after the worker is dead |
| CUDA-L1: work hidden on a side stream (32.8% of RL kernels) | External wall clock until results are *received*; no in-process timers |
| METR: o3 grabbing the scorer's answer from the call stack, patching `time.time` | Separate process, separate sandbox, external clock; the harness holds no answers before the worker has replied |
| Berkeley RDI: monkey-patching `allclose`, overwriting baselines, `atexit` fake scores | Untrusted code never runs in the scorer; outputs are written only by the trusted process; read-only filesystem |
| Wafer: garbage within tolerance scored 104x | No tolerance: exact digests |
| GPU MODE gist: patching `Event.elapsed_time` | No CUDA-event timing |
| Ecdsafail: forged score via constructors in a shared library | Solver code is a separate executable; nothing contestant-built is linked into the harness |
| Returning before work completes / async DMA into mapped files | Pipe transport; timing stops on receipt |
| Persistent kernels or daemons across trials | killpg + PID namespace + die-with-parent + fresh worker |
| Physically impossible numbers | Ceiling check rejects for review |

## 10. Known limitations

* bubblewrap is not a VM. A GPU-driver kernel exploit is out of scope; run on a
  dedicated, ephemeral host with a patched driver (or under gVisor `nvproxy` on a
  supported driver).
* GPU memory is not scrubbed between processes by the driver. TPoW leaks nothing
  useful through residue (fresh seeds, and the public conformance seed only), but
  operators should not share the box with other tenants.
* The CPU budget is measured, not enforced by cgroups. It is a post-hoc rejection
  rule. A cgroup `cpuset` would be a stricter alternative where the runner allows it.
* The ranked hardware itself (clocks, cooling, power delivery) varies by a few
  percent between boxes. Ranked runs must use the same dedicated host per track.
