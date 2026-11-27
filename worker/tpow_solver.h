/*
 * TPoW solver API — the contract between the protected worker main
 * (worker/main.cpp) and the contestant's solver (solvers/<track>/).
 *
 * PROTECTED FILE: submissions may not modify anything under worker/.
 *
 * Lifecycle (one process per trial, see docs/HARNESS.md):
 *   tpow_init(params)            UNTIMED. Create contexts, allocate, autotune,
 *                                JIT, warm up with any seed you like.
 *   tpow_run(seed, first, n, ..) TIMED. Everything that depends on the seed
 *                                (weights, inputs, GEMMs, digests) happens here.
 *                                Call emit() with digests of consecutive units in
 *                                increasing unit order until exactly n were emitted.
 *                                The clock stops when the harness has received the
 *                                last digest byte, so streaming early batches out
 *                                while later ones compute is encouraged.
 */
#ifndef TPOW_SOLVER_H
#define TPOW_SOLVER_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t d;             /* hidden width (4096 for tpow-v1) */
    uint32_t hidden_layers; /* number of d x d layers (4) */
    uint32_t e;             /* input row bytes (64) */
    uint32_t h;             /* head width (32) */
    uint32_t m;             /* rows per unit (16) */
} tpow_params_t;

#define TPOW_DIGEST_BYTES 16

/* Write `n_units` digests (16 bytes each, unit order). Returns 0 on success. */
typedef int (*tpow_emit_fn)(void* emit_ctx, const uint8_t* digests, uint64_t n_units);

/* Return 0 on success; nonzero aborts the trial. */
int tpow_init(const tpow_params_t* params);
int tpow_run(const uint8_t seed[32], uint64_t first_unit, uint64_t n_units,
             tpow_emit_fn emit, void* emit_ctx);

#ifdef __cplusplus
}
#endif
#endif
