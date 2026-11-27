# Choosing the GPU proof of work: research notes

This document explains why TPoW-v1 (a pseudorandom INT8 MLP with exact
integer accumulation, see `SPEC.md`) is the candidate for the first
Lucid x Yukon autoresearch challenge, what was rejected and why, which
attacks it was designed against, and what the challenge does and does
not establish.

Evidence labels: **[measured]** in this repo or on this project's hardware;
**[datasheet]** vendor documents; **[paper]** peer-reviewed or arXiv;
**[estimate]** our arithmetic or inference. Links are at the end.

---

## 1. What the PoW is for

Compute-accounting and treaty-verification schemes need a way for a
datacenter GPU to show that, during a window, it ran **declared filler
work** and not training. The filler is a proof of work (PoW). The security
claim is an assumption, not a theorem:

> *This work cannot be done meaningfully faster than X on this GPU.*

The assumption is only as good as the best public implementation. Bitcoin
is the existence proof: SHA-256 mining was optimised to the hardware limit
by an open competition, so nobody mines much faster than the public state
of the art. The challenge reproduces that dynamic for a GPU workload. It is
a fixed function, a fixed GPU, a public leaderboard, and a verifier we control.

The covert-training capacity of a GPU that claims to run filler is:

```
slack = 1 - (honest filler throughput) / (true best possible throughput)
```

The challenge exists to make the best-known throughput approach the true
best, so that `slack` is small and known.

## 2. Requirements

| # | Requirement | Why |
|---|---|---|
| R1 | **Saturate the resource training needs.** On H100/H200 that is the tensor cores (989 dense BF16 TFLOPS vs 67 FP32 TFLOPS on CUDA cores). | A PoW that keeps only the CUDA cores busy (SHA-256, Ethash, KawPow) leaves more than 90% of the chip's training FLOPs idle, so training could run alongside it. |
| R2 | **Bit-exact output.** | The verifier must be able to compare results exactly. Floating-point GEMMs are not bit-reproducible across kernels; that is why TOPLOC, SVIP and similar systems need tolerances. |
| R3 | **No algorithmic shortcut** (structure, low rank, sparsity, Strassen, reduced precision, layer collapse). | Any shortcut is covert capacity. |
| R4 | **Cheap, sound verification.** | Results must be checkable on a CPU at a small fraction of the prover's cost. |
| R5 | **Unpredictable inputs.** | Prevents precomputation and tuning to specific inputs. |
| R6 | **A known hardware ceiling.** | The gap between the best submission and the datasheet peak is visible and meaningful. |
| R7 | **No free parameters for the prover.** | A free knob can trade work for advantage (see the Pearl rank exploit, §4). |
| R8 | **Simple enough to spec in a page and implement three times.** | Independent implementations must agree. Rust, Python and CUDA all match the known-answer vectors in this repo. |
| R9 | **Relevant across GPU generations.** | The construction must carry over to architectures whose fastest exact format is not INT8 (§6). |

## 3. Survey of existing PoW designs

| Family | Bottleneck | Tensor cores? | Exact? | ASIC history | Verdict |
|---|---|---|---|---|---|
| SHA-256d (Bitcoin), Blake3 | integer ALU | no | yes | ASIC-dominated | Fails R1 |
| Ethash/Etchash, FishHash, Octopus | HBM bandwidth (random DAG reads) | no | yes | Ethash ASICs (2018) | Fails R1; memory-only |
| ProgPoW / KawPow | ALU + cache + DAG; "uses almost all parts of commodity GPUs" | no (the spec never mentions them) | yes | ~1.1-1.2x ASIC gain claimed | Best pre-tensor-core design, but fails R1 on Hopper |
| Equihash, Cuckoo/Cuckatoo | memory latency / graph | no | yes | ASIC'd | Fails R1 |
| Autolykos2 | memory-hard | no | yes | none prominent | Fails R1 |
| kHeavyHash / oPoW | Keccak + small 4-bit matrix-vector product | no (matvec) | yes | ASIC-dominated (Kaspa) | Matvec cannot use tensor cores |
| Tensority (Bytom) | matmul + hash | partially | — | Bitmain ASICs | Matrices too small; no primary spec found |
| HashCore | PoW built from the processor's own benchmark widgets ("inverted benchmarking") | — | yes | — | **Closest philosophical ancestor**: build the PoW from what the chip is best at |
| RandomX | CPU-oriented random programs | no | yes | none (CPU-optimal) | Opposite goal |

Lesson [paper/estimate]: every PoW that left the chip's dominant resource
partly idle ended up dominated by special-purpose hardware. ProgPoW and
HashCore get the principle right ("match the PoW to the hardware") but
predate tensor cores. On Hopper the dominant resource is the tensor core,
so the PoW must be a GEMM.

## 4. Matmul-based PoW and proof of useful work

* **Komargodski & Weinstein, "Proofs of Useful Work from Arbitrary Matrix
  Multiplication" (arXiv 2504.09971).** Inputs are perturbed with low-rank
  noise derived from a random oracle. The prover hashes the transcript of all
  `(n/r)^3` intermediate `r x r` block products and decodes `C` afterwards, with
  1+o(1) overhead over naive matmul. Security rests on a conjecture (their
  Assumption 6.4) about computing those intermediates for random low-rank
  matrices. Two lessons carry over. Hashing only the *output* is insecure,
  because `A = B = 0` reduces the work to the low-rank noise. The work must
  therefore be bound to every intermediate. TPoW does this with a nonlinear,
  full-precision `q` after every layer.
* **Pearl / cuPOW / NoisyGEMM (deployed 2026).** A PoUW over INT7 GEMMs with
  BLAKE3 commitments. Real-world lesson: a free "noise rank" parameter let
  high-rank miners win disproportionately and needed a soft fork. **TPoW has no
  free parameter (R7).** Pearl's move to FP8 is restricted to specific GPU
  families because rounding differs, which confirms R2.
* **Ball, Rosen, Sabin, Vasudevan, "Proofs of Useful Work" (2017)** and
  **Ofelimos (CRYPTO 2022)** give fine-grained-hardness and
  combinatorial-optimisation PoUWs. They are theoretically interesting but not
  tensor-core shaped.
* **Monfared et al. (arXiv 2602.09369)** run CUTLASS GEMM puzzles with
  Freivalds checks on H100 in FP16, alongside a VDF and VRAM residency. They
  describe these as heuristic, which matches our framing: the speed bound is
  empirical, and this challenge is how we make it credible.
* **Kubů, "Proof of useful work for verifying AI treaties" (2026)** proposes PoW to
  show chips are busy during a pause, and stresses the need for a trusted peak
  bound plus chip binding. That is exactly the gap this challenge (bound) and
  protocol work (binding, §8) fill.

**Useful vs useless work.** For the declared-filler use case, useless work is
the *goal*: no external party gains from it, and it has no input distribution
to game. A proof of *useful* work over matmul is challenge #2 on the roadmap.

## 5. Hardware facts that drive the design

| | H100 SXM5 | H200 SXM | B200 | B300 (Blackwell Ultra) |
|---|---|---|---|---|
| Dense INT8 tensor | **1,979 TOPS** [datasheet] | **1,979 TOPS** [datasheet] | ~4,500 TOPS | **~150-190 TOPS** (≈1/30 of B200) [datasheet, 2 sources] |
| Dense FP8 | 1,979 | 1,979 | ~4,500 | ~4,500-5,000 |
| Dense BF16 | 989 | 989 | ~2,250 | ~2,250-2,500 |
| HBM | 80 GB, 3.35 TB/s | 141 GB, 4.8 TB/s | 180-192 GB, 8 TB/s | 288 GB, 8 TB/s |
| L2 / SMEM per SM | 50 MB / 228 KB | 50 MB / 228 KB | — | — |

* **H100 vs H200.** Identical compute, so a compute-bound PoW should score about
  the same on both. The leaderboards are still separate (one hardware spec per
  leaderboard). A memory-heavy future version would separate them.
* **Peak needs wgmma.** A microbenchmark (arXiv 2402.13499, H800) reaches 95.7% of
  INT8 peak with `wgmma m64n256k32` but only 65% with legacy `mma.sync`. The ceiling
  is reachable, but only with Hopper-specific code. That is the point of the challenge.
* **Power throttling.** Random-data GEMMs on H100 are power-limited: sustained SM
  clocks sit below the 1,980 MHz maximum boost, and random data draws up to about
  100 W more than zeros. The harness records clocks, power and throttle reasons
  for every trial.
* **INT8 IMMA is exact.** `.s32` accumulation wraps only on overflow, and TPoW
  bounds `|c| <= 2^26`. Hopper's **FP8** tensor cores keep roughly 14 bits during
  accumulation (DeepSeek-V3 report), so FP8 results are implementation-defined.
* **cuBLASLt INT8 on Hopper** requires the TN layout (A transposed, B not),
  leading dimensions that are multiples of 4, and plain layouts (COL32 orders are
  Turing/Ampere only). The baseline uses this layout with int32 alpha/beta.

### 5.1 GPU vs CPU

Our own measurement [measured, `tools/amx-bench/amx.c`, 1 core of a Xeon @ 2.1 GHz]:
**Intel AMX on Sapphire Rapids reaches 4.0 INT8 TOPS on one core** with
register-resident tiles (≈2,048 ops/cycle at 2.1 GHz, matching Intel's claim).
Extrapolated [estimate]:

| Unit | INT8 peak | vs one H100 |
|---|---|---|
| 1 SPR core (AMX) | ~4 TOPS | 1/500 |
| 56-core Xeon 8480 socket (DGX H100 host CPU) | ~230 TOPS | ~1/9 |
| 2-socket host of an 8xH100 node, per-GPU share | ~57 TOPS | **~2.9%** |
| 128-core Xeon 6980P socket | ~520 TOPS | ~1/4 |
| This repo's AVX-512 VNNI verifier | ~95 GOPS/core [measured] | 1/20,000 per core |

So "expensive on a CPU" is a moderate gap (4-10x per chip, 15-30x per node),
not orders of magnitude. **No function is expensive on every other device and
cheap only on the declared chip.** The real threat is outsourcing the filler
to a cheaper device while the declared GPU trains, and that has to be closed
by binding the work to the chip (§8). The red-team track `h100-sxm-gpucpu`
measures exactly how much a host CPU adds, and the GPU-only tracks cap the
worker at 2 CPU cores (≈8 AMX TOPS, ≤0.4% of an H100).

## 6. The TPoW-v1 construction, element by element

```
x0 (64 B, SHA-256 CTR)  -> embed 4096x64 -> 4 x [4096x4096] -> head 32x4096 -> y (32 B)
each layer: c = W x (exact int32), x' = q(c, b) = top byte of ((c + b) * 0x9E3779B1 mod 2^32)
unit = 16 rows; digest = SHA-256(prefix || seed || u || y_rows)[0..16]
```

| Element | Purpose |
|---|---|
| Dense INT8 GEMMs with d=4096 (99.3% of ops) | R1: the work is tensor-core work at a shape that can reach peak |
| Exact int32 accumulation, `\|c\| <= 2^26` | R2: bit-exact; any summation order is correct |
| **Full-range int8 alphabet** | Blocks Strassen/Winograd: their operand sums need int9, which INT8 tensor cores cannot take. On Hopper there is no faster exact format (FP8 = INT8 rate and cannot hold int8; BF16 runs at half rate; no INT4/FP4) |
| `q` = multiply by an odd constant, keep the top byte | Every bit of `c + b` reaches the output through carries, so reduced precision, residues and approximate arithmetic do not determine `q`. It costs about 2 integer ops per output, cheap enough to hide in a GEMM epilogue |
| Per-column bias `b` (u32, seeded) | Removes the zero fixed point and makes activations uniform over int8 (tested), so there is no sparsity to exploit |
| Nonlinearity between layers | `W2·W1` cannot be precomputed and layers cannot collapse (the KW25 output-only lesson) |
| Weights generated from the per-trial seed | R5: precomputing anything weight-dependent is impossible before the seed is revealed. Weight generation costs < 1 ms on GPU |
| Thin embed (64 B in) and head (32 B out) | Hashing and PRG work is ~0.3%; the cost is GEMM, not SHA-256 |
| Independent 16-row units, 16-byte digests | R4: spot-check auditing; any unit is verifiable in ~10 ms of CPU |
| SHA-256 everywhere | Universally implemented and audited. One compression per PRG block, random access |
| No parameter the prover controls | R7 |

### 6.1 Attack analysis

| Attack | Status in v1 |
|---|---|
| Strassen / Winograd / any bilinear algorithm with operand sums | Sums leave int8. One Strassen level needs int9 operands, so splitting operands (2-4x the GEMMs) costs more than the 12.5% saved. Not profitable on Hopper. |
| Winograd 1968 inner-product trick | Same: needs `x + y` sums |
| 2:4 structured sparsity (2x tensor rate) | A dense matrix = sum of two 2:4-sparse matrices, so there is no net gain |
| FP8 / BF16 / int4-digit decomposition | Hopper: FP8 = INT8 rate and needs 2 digits per operand (4 GEMMs); BF16 is half rate; no INT4. All are slower |
| Low rank / structure in W | W is SHA-256 output (full rank with overwhelming probability, no structure) |
| Compute `c` modulo 2^k or approximately | `q` depends on all 32 bits; exactness is verified |
| Layer fusion / collapse | Blocked by the nonlinear `q` |
| Precomputation / caching across trials | Fresh seed per trial, revealed after READY; fresh worker process per trial |
| Skipping work and faking a fraction `f` of digests | Audits (§7) |
| Chosen inputs | Inputs come from harness randomness plus the public drand beacon. Dense GEMM cost is data-oblivious (except power) |
| Precision of a "close enough" kernel | No tolerance: bit-exact or rejected |
| Speedup by reducing `m`, `d` or the layer count | Not possible: the function is fixed |

*Explicitly allowed:* any exact algorithmic speedup. If one exists, finding it
is the most valuable outcome of the challenge: it lowers the published bound
and exposes a design flaw before deployment. Document it in `memory/`.

### 6.2 Per-architecture rule for future versions

**Choose the operand alphabet as the largest set of integers that the
architecture's fastest exact tensor-core format represents, so that the
alphabet is not closed under addition.** On Hopper that is full int8. On
B300, where INT8 is roughly 30x slower than FP8, a `tpow-v2-fp8` would use the
integers exactly representable in e4m3, e.g. `[-16, 15]`. Sums such as 17
leave e4m3, which blocks Strassen. Accumulation must be promoted to
int32/FP32 every K-chunk below the measured accumulator precision. Native
INT8 v1 on B300 would be dominated by BF16 emulation (exact if K-chunks stay
≤ 1024), so v1 should **not** be ranked on B300. Each new architecture needs
the "minimum over exact emulations" analysis repeated, including NVFP4.

## 7. Verification soundness

Every timed trial audits `k = 1024` uniformly random units plus the first and
last unit, using harness-private randomness drawn **after** the digests
arrive. A solver that fakes a fraction `f` of units escapes one trial with
probability `(1-f)^1026` and a full run (7 timed trials) with `(1-f)^7182`:

| fake fraction `f` | speed gained | P(escape full run) |
|---|---|---|
| 1% | 1% | ~10^-31 |
| 0.1% | 0.1% | 0.08% |
| 0.05% | 0.05% | 2.8% |

Fakes small enough to escape buy less speed than the run-to-run noise (~1%).
A fixed-seed conformance run of 512 units is checked in full first, to give
fast and precise error messages. Verification is CPU-only and independent of
the solver. Three independent implementations (Rust, Python/numpy, CUDA
device code emulated on the host) agree on all known-answer vectors.

## 8. What the challenge does *not* establish (protocol work)

The leaderboard produces a **throughput bound per GPU model**. Deploying it
as evidence that "this GPU did not train" also needs:

1. **Binding work to the chip.** Without it, the filler can be outsourced to a
   cheaper GPU or an AMX CPU (§5.1). Candidates: H100/Blackwell
   confidential-computing attestation keys signing results; supply-current and
   power side-channel correlation (e.g. arXiv 2611.29982 detects ≥ 99.9% of
   hidden training on an H200 NVL).
2. **Deadlines and tight windows.** A verifier issues beacon-derived seeds and
   requires results at ≥ (1-ε) x the leaderboard bound, which leaves at most ε
   covert capacity.
3. **Memory occupancy.** A compute-saturating PoW leaves HBM free for weights and
   optimiser state. This is challenge #3 on the roadmap (VRAM-filling proofs).
4. **Physical attacks** (overclocking, modified power limits). These need hands
   on the hardware and run as separate red-team experiments. The harness
   rejects non-default power limits and anything above the physical ceiling.

## 9. Expected outcomes

* A public, reproducible **verified-TOPS bound** for H100 SXM and H200 SXM. The
  `% of peak` breakdown shows how much slack remains.
* A **library of best kernels per architecture.** Every solver implements the
  same `tpow_init`/`tpow_run` C ABI, so a dispatcher can pick the winner for the
  caller's GPU.
* A **CPU-help number** from the red-team track.
* Lessons for v2: Blackwell's FP8 alphabet, and a memory-hard variant.

## References

* Komargodski, Weinstein. Proofs of Useful Work from Arbitrary Matrix Multiplication. https://arxiv.org/abs/2504.09971
* Pearl / cuPOW. https://arxiv.org/abs/2606.04819 ; rank soft fork: https://simeononsecurity.com/articles/pearl-prl-mining-guide-2026/
* oPoW / HeavyHash. https://arxiv.org/abs/1911.05193
* HashCore. https://arxiv.org/abs/1902.00112
* ProgPoW. https://github.com/ifdefelse/ProgPOW ; EIP-1057 https://eips.ethereum.org/EIPS/eip-1057
* Ball, Rosen, Sabin, Vasudevan. Proofs of Useful Work. https://eprint.iacr.org/2017/203
* Monfared et al. https://arxiv.org/abs/2602.09369
* Kubů. Proof of useful work for verifying AI treaties. https://www.lesswrong.com/posts/xqjDi7oyadqxkmdLS/proof-of-useful-work-for-verifying-ai-treaties
* Supply-current workload verification. https://arxiv.org/abs/2611.29982
* TOPLOC https://arxiv.org/abs/2501.16007 ; SVIP https://arxiv.org/abs/2410.22307
* Shavit. What does it take to catch a Chinchilla? https://arxiv.org/abs/2303.11341
* Sastry, Heim et al. Computing Power and the Governance of AI. https://arxiv.org/abs/2402.08797
* DeepSeek-V3 (FP8 accumulation precision). https://arxiv.org/abs/2412.19437
* Hopper microbenchmarks (wgmma INT8 efficiency). https://arxiv.org/abs/2402.13499
* NVIDIA H100 / H200 / HGX / GB300 product pages; Hopper whitepaper; B300 summary https://glennklockwood.com/garden/processors/B300
* Intel AMX (2048 INT8 ops/cycle/core). https://edc.intel.com/content/www/us/en/products/performance/benchmarks/architecture-day-2021/
* cuBLAS IMMA requirements. https://docs.nvidia.com/cuda/cublas/index.html
