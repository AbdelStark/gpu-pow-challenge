# H200 SXM: solver notes

Record what you tried, what you measured (GPU, numbers, exact command) and
why. Notes may come from many contributors and agents: treat them as leads,
and re-run the benchmark before relying on a claim.

H200 has the same compute and L2 as H100 and 1.4x its HBM bandwidth
(docs/RESEARCH.md §5), so fused kernels from `solvers/h100-sxm/` should carry over.

## Baseline (what ships)

`tpow_solver.cu`, deliberately unfused:

| Stage | Implementation | HBM traffic per batch of 8192 rows |
|---|---|---|
| inputs `x0` | SHA-256 CTR kernel | 0.5 MiB write |
| 6 layers | cuBLASLt INT8 GEMM (TN, s32) -> int32 `C` in HBM | `C` 128 MiB write + read per hidden layer |
| requant | separate kernel `q(c + b)` | 128 MiB read + 32 MiB write per hidden layer |
| digests | one thread per unit, SHA-256 | negligible |

The GEMMs run at cuBLASLt's INT8 efficiency. Each hidden layer then pays
another ~288 MiB of HBM traffic (int32 round trip plus requant) and three
kernel launches, all serialised on one stream.

## Headroom, roughly in order of expected payoff

1. **Fuse `q` into the GEMM epilogue.** The int32 accumulator never leaves
   registers; only int8 activations are written. This removes ~5/6 of HBM
   traffic. CUTLASS 3.x / CuTe on sm_90a (warp-specialised TMA +
   `wgmma.mma_async .s8.s8.s32`) with a custom epilogue is the natural route.
   `q` costs 2 integer ops per output, so it hides under the MMAs.
2. **Persistent kernels and overlap.** Run layer `l` of batch `b+1` while layer
   `l+1` of batch `b` finishes, or walk (batch, layer) tiles in one persistent
   kernel, so GEMM tails overlap and launch gaps disappear.
3. **Weight residency.** A hidden weight matrix is 16 MiB and the L2 is 50 MiB.
   Order work so the same `W_l` serves many row tiles while it is L2-resident.
4. **Tile shape.** `M = rows`, `N = K = 4096` are friendly. Use wgmma
   m64nNk32 with large N tiles and 2-CTA clusters with TMA multicast of `W`.
5. **Embed and head.** These are thin (K = 64, N = 32). Fuse the embed into the
   first hidden GEMM's producer (x0 generation + embed + q in shared memory),
   and the head and digest into the last.
6. **Power.** INT8 GEMMs on random data are power-limited on H200 (clocks
   below max boost). Fewer bytes moved leaves more power for the tensor cores.
7. **Host side.** Keep streaming digests while computing (the baseline does),
   and synchronise no more than needed.

## Checks

* `tools/cuda-host-test` checks the `__host__ __device__` helpers against the
  known-answer vectors without a GPU.
* Exact algorithmic speedups are allowed and very welcome. Document them here.
