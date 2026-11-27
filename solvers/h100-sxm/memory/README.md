# Solver memory notes

Record approaches you tried, numbers you measured and the reasoning behind
important choices in Markdown files here. This directory is open to
contributions from many people and agents, so treat existing notes as leads:
verify claims and re-run the benchmark before relying on them.

## Baseline (what ships)

`tpow_solver.cu`, deliberately unfused:

| Stage | Implementation | HBM traffic per batch of 8192 rows |
|---|---|---|
| inputs `x0` | SHA-256 CTR kernel | 0.5 MiB write |
| 6 layers | cuBLASLt INT8 GEMM (TN, s32) -> int32 `C` in HBM | `C` 128 MiB write + read per hidden layer |
| requant | separate kernel `q(c + b)` | 128 MiB read + 32 MiB write per hidden layer |
| digests | one thread per unit, SHA-256 | negligible |

Expected profile: the GEMMs reach cuBLASLt's INT8 efficiency. Each hidden layer
then pays roughly another 288 MiB of HBM traffic (int32 round trip plus requant)
and three kernel launches, all serialised on one stream.

## Headroom, roughly in order of expected payoff

1. **Fuse `q` into the GEMM epilogue.** The int32 accumulator never leaves registers;
   only the int8 activations are written. This removes ~5/6 of HBM traffic.
   CUTLASS 3.x / CuTe on sm_90a (warp-specialised TMA + `wgmma.mma_async`
   `.s8.s8.s32`) with a custom epilogue is the natural route.
   `q` costs 2 integer ops per output (IADD + IMAD.HI-style), so it hides under the MMAs.
2. **Persistent kernels and overlap.** Run layer `l` of batch `b+1` while layer `l+1`
   of batch `b` finishes, or use a persistent kernel that walks (batch, layer)
   tiles, so the tail of one GEMM overlaps the next and there are no launch gaps.
3. **Weight residency.** A hidden weight matrix is 16 MiB and the H100 L2 is 50 MiB.
   Order work so the same `W_l` is reused across many row tiles while it is L2-resident.
4. **Tile shape.** `M = rows`, `N = d = 4096`, `K = 4096` are friendly. Use
   wgmma m64nNk32 with large N tiles and 2-CTA clusters with TMA multicast of `W`.
5. **Embed and head.** These are thin (K = 64 and N = 32). Fuse the embed into
   the first hidden GEMM's producer (x0 generation + embed + q in shared memory),
   and fuse the head and digest into the last.
6. **Power.** INT8 GEMMs on random data are power-limited on H100 (clocks below max boost).
   Fewer wasted bytes moved means more power left for the tensor cores.
7. **Host side.** Stream digests while computing (the baseline already does), and
   avoid synchronising more than needed.

## Rules of thumb

* Every output digest must be exact. The harness checks a fixed-seed conformance
  run in full and audits >= 1024 random units per timed trial.
* The function is fixed: d=4096, L=4, e=64, h=32, m=16, full int8 range.
  Algorithmic speedups that keep exact outputs are allowed and very welcome, since
  they lower the published bound. Document them here.
* `tools/cuda-host-test` checks your host/device helpers against the known-answer
  vectors without a GPU.
