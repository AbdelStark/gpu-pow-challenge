# Agent guide

You are optimising a GPU kernel for a ranked benchmark. Read `README.md`
(contract), `SPEC.md` (the function) and `solvers/<track>/memory/` (prior notes).

## Rules

* Edit only `solvers/<track>/`. Everything else is the harness and is rejected in submissions.
* Outputs must be bit-exact. There is no tolerance. Never special-case seeds or unit indices.
* Do not try to detect conformance or audit runs. Every timed trial is audited
  with private randomness, after your output has been received.
* Keep `memory/` notes honest: record what you measured, on which GPU, and the
  exact command you ran.

## Loop

```bash
./setup.sh --track local-gpu                     # once
./benchmark.sh --track local-gpu --note "idea"   # correctness + speed on any NVIDIA GPU
./benchmark.sh --track h100-sxm  --note "idea"   # on the ranked hardware
```

`local-gpu` builds `solvers/h100-sxm/`. For another lineage, set
`TPOW_SOLVER_SRC=solvers/<track>`.

If the conformance run fails, the harness prints the first wrong unit and a
`tpow-kat digests ...` command that prints the expected digests. To check
`__host__ __device__` helpers without a GPU:

```bash
mkdir -p build
nvcc -O3 -std=c++17 -arch=sm_90a -I solvers/h100-sxm tools/cuda-host-test/host_test.cu -o build/host_test
build/host_test tests/vectors/tpow-v1.json 2
```

## Where the time goes

See `solvers/h100-sxm/memory/README.md`. The biggest single lever is fusing
`q` into the INT8 GEMM epilogue on sm_90a (wgmma + TMA). After that come
persistent kernels and overlap across layers and batches.
