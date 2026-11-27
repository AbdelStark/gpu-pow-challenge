// Intel AMX INT8 peak microbenchmark (register-resident tiles, no memory traffic).
//
//   gcc -O2 -mamx-tile -mamx-int8 -pthread tools/amx-bench/amx.c -o build/amx
//   build/amx <threads>        prints INT8 GOPS (2 ops per multiply-accumulate)
#include <immintrin.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#define ARCH_REQ_XCOMP_PERM 0x1023
#define XFEATURE_XTILEDATA 18
#define MAX_THREADS 256
#define ITERS 20000000L

typedef struct {
    uint8_t palette, start;
    uint8_t reserved[14];
    uint16_t colsb[16];
    uint8_t rows[16];
} tile_cfg_t;

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static void* run(void* arg) {
    (void)arg;
    if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) {
        perror("arch_prctl");
        return NULL;
    }
    tile_cfg_t cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.palette = 1;
    for (int i = 0; i < 8; i++) {
        cfg.colsb[i] = 64;
        cfg.rows[i] = 16;
    }
    _tile_loadconfig(&cfg);

    int8_t a[16 * 64], b[16 * 64];
    int32_t out[16 * 16];
    for (int i = 0; i < 16 * 64; i++) {
        a[i] = (int8_t)(i * 7);
        b[i] = (int8_t)(i * 13);
    }
    _tile_loadd(4, a, 64);
    _tile_loadd(5, a, 64);
    _tile_loadd(6, b, 64);
    _tile_loadd(7, b, 64);
    _tile_zero(0);
    _tile_zero(1);
    _tile_zero(2);
    _tile_zero(3);
    // Four independent accumulators keep the AMX unit busy.
    for (long i = 0; i < ITERS; i++) {
        _tile_dpbssd(0, 4, 6);
        _tile_dpbssd(1, 4, 7);
        _tile_dpbssd(2, 5, 6);
        _tile_dpbssd(3, 5, 7);
    }
    _tile_stored(0, out, 64);
    _tile_release();
    return (void*)(intptr_t)out[0];
}

int main(int argc, char** argv) {
    int threads = argc > 1 ? atoi(argv[1]) : 1;
    if (threads < 1 || threads > MAX_THREADS) {
        fprintf(stderr, "threads must be in [1, %d]\n", MAX_THREADS);
        return 2;
    }
    pthread_t t[MAX_THREADS];
    double t0 = now();
    for (int i = 0; i < threads; i++) pthread_create(&t[i], NULL, run, NULL);
    for (int i = 0; i < threads; i++) pthread_join(t[i], NULL);
    double dt = now() - t0;
    // Per iteration: 4 tile products of 16x16 outputs, each a 64-deep dot product.
    double ops = (double)threads * ITERS * 4 * 16 * 16 * 64 * 2;
    printf("threads=%d  %.1f GOPS total  %.1f GOPS/thread\n", threads, ops / dt / 1e9, ops / dt / 1e9 / threads);
    return 0;
}
