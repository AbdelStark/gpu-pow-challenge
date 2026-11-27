// Host-side conformance test of solvers/<track>/tpow_device.cuh (runs on the CPU; no GPU needed).
//
// Emulates the baseline solver's pipeline on the host using the SAME
// __host__ __device__ functions and the SAME cuBLASLt argument mapping
// (tpow::layer_gemm_args) fed into a literal column-major BLAS emulator, then
// compares unit digests with tests/vectors/<file>.json from the Rust reference.
//
//   nvcc -O3 -std=c++17 -arch=sm_90a -I solvers/h100-sxm tools/cuda-host-test/host_test.cu -o build/host_test
//   build/host_test tests/vectors/tpow-tiny.json [max_units]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "tpow_device.cuh"

static std::string slurp(const char* path) {
    std::ifstream f(path);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static long long json_int(const std::string& s, const std::string& key, size_t from = 0) {
    size_t p = s.find("\"" + key + "\": ", from);
    if (p == std::string::npos) { std::fprintf(stderr, "missing %s\n", key.c_str()); std::exit(2); }
    return std::atoll(s.c_str() + p + key.size() + 4);
}

static std::string hex(const uint8_t* b, size_t n) {
    static const char* H = "0123456789abcdef";
    std::string r;
    for (size_t i = 0; i < n; ++i) { r += H[b[i] >> 4]; r += H[b[i] & 15]; }
    return r;
}

// Byte-exact emulation of prg_fill_kernel.
static void prg_fill(const tpow::SeedWords& sw, uint32_t stream, uint64_t first_block, uint64_t n, uint8_t* out) {
    for (uint64_t blk = 0; blk * 32 < n; ++blk) {
        uint32_t w[8];
        tpow::prg_block(sw, stream, first_block + blk, w);
        for (int j = 0; j < 32 && blk * 32 + j < n; ++j) out[blk * 32 + j] = uint8_t(w[j >> 2] >> (24 - 8 * (j & 3)));
    }
}

// Literal column-major BLAS semantics: C(m x n) = op(A) op(B).
static void blas_gemm_i8(const tpow::GemmArgs& g, const int8_t* A, const int8_t* B, int32_t* C) {
    for (int j = 0; j < g.n; ++j)
        for (int i = 0; i < g.m; ++i) {
            int64_t acc = 0;
            for (int p = 0; p < g.k; ++p) {
                int a = g.transa ? A[p + (size_t)i * g.lda] : A[i + (size_t)p * g.lda];
                int b = g.transb ? B[j + (size_t)p * g.ldb] : B[p + (size_t)j * g.ldb];
                acc += a * b;
            }
            C[i + (size_t)j * g.ldc] = int32_t(acc);
        }
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: host_test <vectors.json> [max_units]\n"); return 2; }
    const std::string js = slurp(argv[1]);
    const int max_units = argc > 2 ? std::atoi(argv[2]) : 1 << 30;
    const uint32_t d = uint32_t(json_int(js, "d")), L = uint32_t(json_int(js, "hidden_layers")),
                   e = uint32_t(json_int(js, "e")), h = uint32_t(json_int(js, "h")), m = uint32_t(json_int(js, "m"));
    uint8_t seed[32];
    for (int i = 0; i < 32; ++i) seed[i] = uint8_t(i);  // KAT seed
    tpow::SeedWords sw = tpow::seed_words(seed);
    int failures = 0;

    // 1. Raw PRG block of stream X.
    {
        uint8_t b[32];
        prg_fill(sw, tpow::STREAM_X, 0, 32, b);
        std::string want = js.substr(js.find("\"x_prg_block0\": \"") + 17, 64);
        if (hex(b, 32) != want) { std::printf("FAIL x_prg_block0 %s != %s\n", hex(b, 32).c_str(), want.c_str()); ++failures; }
        else std::printf("ok   x_prg_block0\n");
    }
    // 2. q spot values (mirror src/spec.rs tests).
    if (tpow::q(1, 0) != int8_t(0x9E) || tpow::q(0, 0) != 0) { std::printf("FAIL q\n"); ++failures; }

    // 3. Weights / biases.
    const int nl = int(L) + 2;
    std::vector<std::vector<int8_t>> W(nl);
    std::vector<std::vector<uint32_t>> Bv(nl);
    std::vector<int> outd(nl), ind(nl);
    for (int l = 0; l < nl; ++l) {
        outd[l] = l == 0 ? int(d) : (l <= int(L) ? int(d) : int(h));
        ind[l] = l == 0 ? int(e) : int(d);
        W[l].resize(size_t(outd[l]) * ind[l]);
        Bv[l].resize(outd[l]);
        prg_fill(sw, tpow::stream_w(l), 0, W[l].size(), reinterpret_cast<uint8_t*>(W[l].data()));
        prg_fill(sw, tpow::stream_b(l), 0, Bv[l].size() * 4, reinterpret_cast<uint8_t*>(Bv[l].data()));
    }
    // 4. Unit digests through the emulated pipeline.
    size_t pos = js.find("\"digests\"");
    int checked = 0;
    while (checked < max_units) {
        size_t dp = js.find("\"digest\": \"", pos);
        if (dp == std::string::npos) break;
        std::string want = js.substr(dp + 11, 32);
        uint64_t unit = uint64_t(json_int(js, "unit", dp));
        pos = dp + 1;
        std::vector<int8_t> x(size_t(m) * e);
        prg_fill(sw, tpow::STREAM_X, unit * m * (e / 32), x.size(), reinterpret_cast<uint8_t*>(x.data()));
        for (int l = 0; l < nl; ++l) {
            tpow::GemmArgs g = tpow::layer_gemm_args(outd[l], ind[l], int(m));
            std::vector<int32_t> C(size_t(m) * outd[l]);
            blas_gemm_i8(g, W[l].data(), x.data(), C.data());
            std::vector<int8_t> y(C.size());
            for (size_t t = 0; t < C.size(); ++t) y[t] = tpow::q(C[t], Bv[l][t % outd[l]]);
            x.swap(y);
        }
        uint8_t dig[16];
        tpow::unit_digest(seed, unit, x.data(), uint32_t(x.size()), dig);
        bool ok = hex(dig, 16) == want;
        std::printf("%s unit %llu digest %s\n", ok ? "ok  " : "FAIL", (unsigned long long)unit, hex(dig, 16).c_str());
        failures += !ok;
        ++checked;
    }
    if (checked == 0) { std::printf("FAIL: no digests checked\n"); return 1; }
    std::printf(failures ? "HOST TEST FAILED (%d)\n" : "HOST TEST OK (%d units)\n", failures ? failures : checked);
    return failures ? 1 : 0;
}
