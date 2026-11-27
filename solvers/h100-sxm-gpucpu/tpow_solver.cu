// TPoW-v1 BASELINE GPU solver (EDITABLE — this is what you optimise).
//
// Straightforward, correct, deliberately un-fused:
//   * weights/biases/inputs: SHA-256 counter-mode PRG kernels
//   * each layer: cuBLASLt INT8 GEMM (TN, int32 accumulate) -> int32 C in HBM
//                 -> separate requantisation kernel q(c + b) -> int8
//   * per-unit SHA-256 digest kernel, async D2H copy, host emits batch b-1
//     while the GPU computes batch b.
//
// Obvious headroom (see memory/README.md in this directory): fuse q into the GEMM epilogue
// (never write int32 C to HBM), persistent warp-specialised wgmma kernels,
// overlapping layers of different batches, generating x0 inside the first GEMM.
#include <cublasLt.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <vector>

#include "tpow_solver.h"
#include "tpow_device.cuh"

#define CK(call)                                                                       \
    do {                                                                               \
        cudaError_t e_ = (call);                                                       \
        if (e_ != cudaSuccess) {                                                       \
            std::fprintf(stderr, "solver: %s:%d %s: %s\n", __FILE__, __LINE__, #call,  \
                         cudaGetErrorString(e_));                                      \
            return -1;                                                                 \
        }                                                                              \
    } while (0)

#define CKB(call)                                                                      \
    do {                                                                               \
        cublasStatus_t s_ = (call);                                                    \
        if (s_ != CUBLAS_STATUS_SUCCESS) {                                             \
            std::fprintf(stderr, "solver: %s:%d %s: cublas status %d\n", __FILE__,     \
                         __LINE__, #call, int(s_));                                    \
            return -1;                                                                 \
        }                                                                              \
    } while (0)

namespace {

using tpow::SeedWords;

constexpr int kBatchUnits = 512;           // units per GPU batch
constexpr size_t kWorkspaceBytes = 64u << 20;

// ── kernels ──────────────────────────────────────────────────────────────────

// out[0 .. n_bytes) = PRG stream bytes starting at block `first_block`.
__global__ void prg_fill_kernel(SeedWords seed, uint32_t stream, uint64_t first_block,
                                uint64_t n_bytes, uint8_t* __restrict__ out) {
    uint64_t blk = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x;
    uint64_t off = blk * 32;
    if (off >= n_bytes) return;
    uint32_t w[8];
    tpow::prg_block(seed, stream, first_block + blk, w);
    if (off + 32 <= n_bytes && (reinterpret_cast<uintptr_t>(out + off) & 15) == 0) {
        uint4 v0, v1;
        v0.x = tpow::bswap32(w[0]); v0.y = tpow::bswap32(w[1]); v0.z = tpow::bswap32(w[2]); v0.w = tpow::bswap32(w[3]);
        v1.x = tpow::bswap32(w[4]); v1.y = tpow::bswap32(w[5]); v1.z = tpow::bswap32(w[6]); v1.w = tpow::bswap32(w[7]);
        reinterpret_cast<uint4*>(out + off)[0] = v0;
        reinterpret_cast<uint4*>(out + off)[1] = v1;
    } else {
        for (int j = 0; j < 32 && off + j < n_bytes; ++j) out[off + j] = uint8_t(w[j >> 2] >> (24 - 8 * (j & 3)));
    }
}

// y[r][j] = q(c[r][j], bias[j]); 4 elements per thread (out % 4 == 0).
__global__ void requant_kernel(const int32_t* __restrict__ c, const uint32_t* __restrict__ bias,
                               int8_t* __restrict__ y, uint64_t n4, uint32_t out) {
    uint64_t i = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x;
    if (i >= n4) return;
    int4 v = reinterpret_cast<const int4*>(c)[i];
    uint32_t j = uint32_t((i * 4) % out);
    char4 r;
    r.x = tpow::q(v.x, bias[j]);
    r.y = tpow::q(v.y, bias[j + 1]);
    r.z = tpow::q(v.z, bias[j + 2]);
    r.w = tpow::q(v.w, bias[j + 3]);
    reinterpret_cast<char4*>(y)[i] = r;
}

struct SeedBytes { uint8_t b[32]; };

// One thread per unit: digest over the unit's m*h head-output bytes.
__global__ void digest_kernel(SeedBytes seed, uint64_t first_unit, uint32_t n_units,
                              const int8_t* __restrict__ y, uint32_t unit_bytes,
                              uint8_t* __restrict__ out) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_units) return;
    tpow::unit_digest(seed.b, first_unit + i, y + uint64_t(i) * unit_bytes, unit_bytes, out + 16ull * i);
}

// ── solver state ─────────────────────────────────────────────────────────────

struct LayerDev {
    int out = 0, in = 0;
    int8_t* w = nullptr;
    uint32_t* b = nullptr;
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
    cublasLtMatmulAlgo_t algo{};
};

struct State {
    tpow_params_t p{};
    int rows = 0;  // rows per batch
    cudaStream_t stream = nullptr;
    cublasLtHandle_t lt = nullptr;
    void* workspace = nullptr;
    std::vector<LayerDev> layers;
    int8_t* x0 = nullptr;
    int8_t* act[2] = {nullptr, nullptr};
    int32_t* c = nullptr;
    uint8_t* dig = nullptr;
    uint8_t* hdig[2] = {nullptr, nullptr};
    cudaEvent_t done[2] = {nullptr, nullptr};
} S;

int setup_layer_gemm(LayerDev& L, int rows) {
    tpow::GemmArgs g = tpow::layer_gemm_args(L.out, L.in, rows);
    CKB(cublasLtMatmulDescCreate(&L.op, CUBLAS_COMPUTE_32I, CUDA_R_32I));
    cublasOperation_t ta = g.transa ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t tb = g.transb ? CUBLAS_OP_T : CUBLAS_OP_N;
    CKB(cublasLtMatmulDescSetAttribute(L.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
    CKB(cublasLtMatmulDescSetAttribute(L.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
    // Stored (pre-op) shapes, column-major: A is k x m (opA = T), B is k x n, C is m x n.
    CKB(cublasLtMatrixLayoutCreate(&L.la, CUDA_R_8I, g.k, g.m, g.lda));
    CKB(cublasLtMatrixLayoutCreate(&L.lb, CUDA_R_8I, g.k, g.n, g.ldb));
    CKB(cublasLtMatrixLayoutCreate(&L.lc, CUDA_R_32I, g.m, g.n, g.ldc));
    cublasLtMatmulPreference_t pref;
    CKB(cublasLtMatmulPreferenceCreate(&pref));
    size_t ws = kWorkspaceBytes;
    CKB(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));
    cublasLtMatmulHeuristicResult_t res[8];
    int found = 0;
    CKB(cublasLtMatmulAlgoGetHeuristic(S.lt, L.op, L.la, L.lb, L.lc, L.lc, pref, 8, res, &found));
    cublasLtMatmulPreferenceDestroy(pref);
    if (found == 0) {
        std::fprintf(stderr, "solver: no cuBLASLt int8 algorithm for m=%d n=%d k=%d\n", g.m, g.n, g.k);
        return -1;
    }
    L.algo = res[0].algo;
    return 0;
}

int gemm(const LayerDev& L, const int8_t* x) {
    const int32_t alpha = 1, beta = 0;  // int32 scale type: exact
    CKB(cublasLtMatmul(S.lt, L.op, &alpha, L.w, L.la, x, L.lb, &beta, S.c, L.lc, S.c, L.lc, &L.algo,
                       S.workspace, kWorkspaceBytes, S.stream));
    return 0;
}

int prg_fill(const SeedWords& sw, uint32_t stream, uint64_t first_block, uint64_t n_bytes, void* out) {
    uint64_t blocks = (n_bytes + 31) / 32;
    unsigned grid = unsigned((blocks + 255) / 256);
    prg_fill_kernel<<<grid, 256, 0, S.stream>>>(sw, stream, first_block, n_bytes, static_cast<uint8_t*>(out));
    CK(cudaGetLastError());
    return 0;
}

// Full pipeline for one batch of kBatchUnits units starting at `first_unit`.
int compute_batch(const SeedWords& sw, const SeedBytes& sb, uint64_t first_unit) {
    const tpow_params_t& p = S.p;
    const uint64_t first_row = first_unit * p.m;
    if (prg_fill(sw, tpow::STREAM_X, first_row * (p.e / 32), uint64_t(S.rows) * p.e, S.x0)) return -1;
    const int8_t* in = S.x0;
    int cur = 0;
    for (size_t l = 0; l < S.layers.size(); ++l) {
        const LayerDev& L = S.layers[l];
        if (gemm(L, in)) return -1;
        uint64_t n4 = uint64_t(S.rows) * L.out / 4;
        requant_kernel<<<unsigned((n4 + 255) / 256), 256, 0, S.stream>>>(S.c, L.b, S.act[cur], n4, L.out);
        CK(cudaGetLastError());
        in = S.act[cur];
        cur ^= 1;
    }
    // `in` now holds rows x h head outputs, unit-major (m consecutive rows per unit).
    digest_kernel<<<(kBatchUnits + 127) / 128, 128, 0, S.stream>>>(sb, first_unit, kBatchUnits, in, p.m * p.h, S.dig);
    CK(cudaGetLastError());
    return 0;
}

}  // namespace

extern "C" int tpow_init(const tpow_params_t* params) {
    S.p = *params;
    const tpow_params_t& p = S.p;
    if (p.e % 32 || p.d % 64 || p.h % 16) return -1;
    S.rows = kBatchUnits * int(p.m);
    CK(cudaSetDevice(0));
    CK(cudaStreamCreateWithFlags(&S.stream, cudaStreamNonBlocking));
    CKB(cublasLtCreate(&S.lt));
    CK(cudaMalloc(&S.workspace, kWorkspaceBytes));
    const int nl = int(p.hidden_layers) + 2;
    S.layers.resize(nl);
    for (int l = 0; l < nl; ++l) {
        LayerDev& L = S.layers[l];
        L.out = l == 0 ? int(p.d) : (l <= int(p.hidden_layers) ? int(p.d) : int(p.h));
        L.in = l == 0 ? int(p.e) : int(p.d);
        CK(cudaMalloc(&L.w, size_t(L.out) * L.in));
        CK(cudaMalloc(&L.b, size_t(L.out) * 4));
        if (setup_layer_gemm(L, S.rows)) return -1;
    }
    CK(cudaMalloc(&S.x0, size_t(S.rows) * p.e));
    CK(cudaMalloc(&S.act[0], size_t(S.rows) * p.d));
    CK(cudaMalloc(&S.act[1], size_t(S.rows) * p.d));
    CK(cudaMalloc(&S.c, size_t(S.rows) * p.d * 4));
    CK(cudaMalloc(&S.dig, size_t(kBatchUnits) * 16));
    for (int i = 0; i < 2; ++i) {
        CK(cudaMallocHost(&S.hdig[i], size_t(kBatchUnits) * 16));
        CK(cudaEventCreateWithFlags(&S.done[i], cudaEventDisableTiming));
    }
    // Untimed warm-up with an arbitrary seed: loads kernels, primes cuBLASLt.
    uint8_t warm[32] = {0};
    SeedWords sw = tpow::seed_words(warm);
    SeedBytes sb;
    std::memcpy(sb.b, warm, 32);
    for (int l = 0; l < nl; ++l) {
        LayerDev& L = S.layers[l];
        if (prg_fill(sw, tpow::stream_w(l), 0, uint64_t(L.out) * L.in, L.w)) return -1;
        if (prg_fill(sw, tpow::stream_b(l), 0, uint64_t(L.out) * 4, L.b)) return -1;
    }
    if (compute_batch(sw, sb, 0)) return -1;
    CK(cudaStreamSynchronize(S.stream));
    return 0;
}

extern "C" int tpow_run(const uint8_t seed[32], uint64_t first_unit, uint64_t n_units,
                        tpow_emit_fn emit, void* emit_ctx) {
    const tpow_params_t& p = S.p;
    SeedWords sw = tpow::seed_words(seed);
    SeedBytes sb;
    std::memcpy(sb.b, seed, 32);
    // Seed-dependent weights and biases (inside the timed window).
    for (size_t l = 0; l < S.layers.size(); ++l) {
        LayerDev& L = S.layers[l];
        if (prg_fill(sw, tpow::stream_w(uint32_t(l)), 0, uint64_t(L.out) * L.in, L.w)) return -1;
        if (prg_fill(sw, tpow::stream_b(uint32_t(l)), 0, uint64_t(L.out) * 4, L.b)) return -1;
    }
    (void)p;
    const uint64_t n_batches = (n_units + kBatchUnits - 1) / kBatchUnits;
    for (uint64_t b = 0; b <= n_batches; ++b) {
        if (b < n_batches) {
            // Full batches only; units beyond n are computed and discarded.
            if (compute_batch(sw, sb, first_unit + b * kBatchUnits)) return -1;
            CK(cudaMemcpyAsync(S.hdig[b & 1], S.dig, size_t(kBatchUnits) * 16, cudaMemcpyDeviceToHost, S.stream));
            CK(cudaEventRecord(S.done[b & 1], S.stream));
        }
        if (b > 0) {
            // Emit the previous batch while the GPU works on this one.
            const uint64_t pb = b - 1;
            CK(cudaEventSynchronize(S.done[pb & 1]));
            uint64_t k = n_units - pb * kBatchUnits;
            if (k > kBatchUnits) k = kBatchUnits;
            if (emit(emit_ctx, S.hdig[pb & 1], k) != 0) return -1;
        }
    }
    return 0;
}
