# TPoW-v1 — Specification

**Status:** v1, frozen for the H100/H200 leaderboards. This document is normative.
Where it disagrees with code, the document wins and the code is a bug.
Reference implementations: Rust (`src/spec.rs`, `src/prg.rs`, `src/model.rs`),
Python (`reference/tpow_ref.py`), CUDA host/device primitives (`solver/tpow_device.cuh`).
Known-answer vectors: `tests/vectors/tpow-v1.json`, `tests/vectors/tpow-tiny.json`.

## 1. Overview

TPoW is a *pseudorandom INT8 multilayer perceptron* applied to independent rows.
All weights and inputs come from a seed. Each layer is an exact integer
matrix product followed by a cheap nonlinear requantisation back to int8.
The work is dominated by dense `int8 x int8 -> int32` GEMMs, the operation
that tensor cores accelerate and that AI training and inference depend on.

The solver returns one 16-byte digest per **unit** (`m` consecutive rows).
A verifier checks any unit independently on a CPU.

## 2. Parameters

| Symbol | Name | tpow-v1 | tpow-tiny (tests only) |
|---|---|---|---|
| `d` | hidden width | 4096 | 256 |
| `L` | hidden layers (`d x d`) | 4 | 2 |
| `e` | input row bytes (embed in-width) | 64 | 64 |
| `h` | head width (output bytes per row) | 32 | 32 |
| `m` | rows per unit | 16 | 4 |

Layers are numbered `l = 0 .. L+1`. Layer `l` has shape `(out_l, in_l)`:

* `l = 0` (embed): `(d, e)`
* `1 <= l <= L` (hidden): `(d, d)`
* `l = L+1` (head): `(h, d)`

Canonical work per row: `ops_row = 2 * (d*e + L*d*d + h*d)` int8 operations
(one multiply-accumulate = 2 ops). For tpow-v1, `ops_row = 135,004,160` and
`ops_unit = m * ops_row = 2,160,066,560`.

## 3. Primitives

All integers are little-endian unless stated otherwise. `H` = SHA-256 (FIPS 180-4).
`int8(v)` reinterprets a byte as two's complement.

### 3.1 PRG

```
block(seed, stream, ctr) = H( "tpow/v1P" || seed[32] || le32(stream) || le64(ctr) )
```

The message is exactly 52 bytes, so each 32-byte block costs one SHA-256
compression. The byte stream `S(seed, stream)` is
`block(seed, stream, 0) || block(seed, stream, 1) || ...`.
Byte `i` of the stream is byte `i mod 32` of block `floor(i/32)`.

Stream identifiers (32-bit):

| Stream | id |
|---|---|
| weights of layer `l` | `0x57000000 OR l` (`'W' << 24`) |
| biases of layer `l` | `0x42000000 OR l` (`'B' << 24`) |
| input rows | `0x58000000` (`'X' << 24`) |

### 3.2 Requantisation

```
q(c, b) = int8( ( (u32(c) + b) * 0x9E3779B1  mod 2^32 ) >> 24 )
```

`c` is the exact int32 accumulator and `b` is a u32 bias. `u32(c)` is two's-complement
reinterpretation. All arithmetic is mod 2^32. The output is the top byte of the
product, read as int8.

## 4. Model generation (per seed)

For each layer `l`:

* `W_l[j][k] = int8( S(seed, 0x57000000|l)[ j*in_l + k ] )` for `0 <= j < out_l`, `0 <= k < in_l`.
* `b_l[j] = le_u32( S(seed, 0x42000000|l)[ 4j .. 4j+4 ] )` for `0 <= j < out_l`.

## 5. Rows and units

Global row `g` has input `x_0 = int8( S(seed, 0x58000000)[ g*e .. g*e + e ] )`.
Because `e` is a multiple of 32, row `g` starts at PRG block `g*e/32`.

Row evaluation:

```
for l in 0 .. L+1:
    c_j      = sum_k W_l[j][k] * x_l[k]      (exact integer; |c_j| < 2^31 is guaranteed)
    x_{l+1}[j] = q(c_j, b_l[j])
y = x_{L+2}                                  (h bytes)
```

Unit `u` (any u64 such that `u*m` does not overflow) consists of rows
`g = u*m + r` for `r = 0 .. m-1`. Its digest is

```
D_u = H( "tpow/v1D" || seed[32] || le64(u) || y_{u*m} || y_{u*m+1} || ... || y_{u*m+m-1} )[0..16]
```

`y` values are serialised as raw bytes (two's complement), `m*h` bytes in total.

## 6. The challenge task

The challenge task is: given `seed` and `n`, output `D_0 || D_1 || ... || D_{n-1}`
(`16n` bytes) as fast as possible. The harness defines the timing, sampling and
scoring rules (docs/HARNESS.md). The function itself has no free parameters.

## 7. Design rationale (informative)

* **Exactness.** The accumulator is bounded: `|c| <= in * 128 * 128 <= 2^26`
  for `in = 4096`. Integer tensor-core accumulation (`.s32`) is therefore exact,
  and results are bit-identical across implementations regardless of summation
  order. This avoids the tolerance problems of floating-point checks.
* **Full-range int8 alphabet.** Operands use all 256 int8 values. Sums of two
  operands need 9 bits and leave the tensor-core input format, so Strassen and
  Winograd style bilinear algorithms (which multiply operand sums) cannot run
  on INT8 tensor cores without splitting operands, which costs more than it saves.
  On Hopper, FP8 runs no faster than INT8 and cannot represent all int8 values,
  BF16 runs at half rate, and there are no INT4/FP4 units. Native INT8 is therefore
  the fastest exact path on sm_90. (Other architectures need their own analysis: docs/RESEARCH.md §6.)
* **Every accumulator bit matters.** `q` multiplies by an odd constant mod 2^32
  and keeps the top byte. Carries propagate every low bit of `c + b` upward, so
  computing `c` modulo a smaller power of two or approximately does not
  determine `q`. The per-column bias removes the zero fixed point and makes the
  activations uniform over int8, so there is no sparsity or structure to exploit.
* **No layer collapse.** `q` is not linear over the integers, so the product
  `W_2 W_1` cannot be precomputed.
* **Data-oblivious cost.** Dense GEMM work does not depend on values. Inputs
  cannot be chosen to make the function cheaper (power draw aside, see HARNESS.md).
* **Thin embed and head.** Inputs are 64 bytes per row (2 PRG blocks) and outputs
  are 32 bytes per row, so hashing is ~0.3% of the work. The `L*d^2` hidden layers
  are 99.3% of `ops_row`, so the cost is tensor-core GEMM.
* **Independent units.** Units can be checked one at a time on a CPU, so the
  verifier can spot-check (docs/HARNESS.md §Audits). A unit costs 2.16 GOP to
  verify (~10 ms on a few AVX-512 VNNI cores).
