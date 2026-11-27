// TPoW-v1 building blocks usable on both host and device (EDITABLE).
//
// Everything here mirrors SPEC.md and the Rust reference (src/spec.rs,
// src/prg.rs, src/model.rs). tools/cuda-host-test checks these functions on
// the CPU against tests/vectors/*.json, so they can be validated without a GPU.
#pragma once
#include <stdint.h>

#if defined(__CUDACC__)
#define TPOW_HD __host__ __device__ __forceinline__
#else
#define TPOW_HD inline
#endif

#if defined(__CUDA_ARCH__)
#define TPOW_UNROLL _Pragma("unroll")
#else
#define TPOW_UNROLL
#endif

namespace tpow {

constexpr uint32_t Q_MUL = 0x9E3779B1u;
constexpr uint32_t STREAM_X = 0x58000000u;
TPOW_HD uint32_t stream_w(uint32_t l) { return 0x57000000u | l; }
TPOW_HD uint32_t stream_b(uint32_t l) { return 0x42000000u | l; }

// q(c, b) = int8(((c + b) * Q_MUL mod 2^32) >> 24)
TPOW_HD int8_t q(int32_t c, uint32_t b) {
    return static_cast<int8_t>(static_cast<uint8_t>(((static_cast<uint32_t>(c) + b) * Q_MUL) >> 24));
}

#define TPOW_SHA256_K                                                                              \
    {0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u,    \
     0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu,    \
     0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu,    \
     0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,    \
     0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,    \
     0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,    \
     0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u,    \
     0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,    \
     0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u,    \
     0xc67178f2u}

#if defined(__CUDACC__)
__constant__ uint32_t k_sha256_dev[64] = TPOW_SHA256_K;
#endif
static const uint32_t k_sha256_host[64] = TPOW_SHA256_K;

TPOW_HD uint32_t sha_k(int i) {
#if defined(__CUDA_ARCH__)
    return k_sha256_dev[i];
#else
    return k_sha256_host[i];
#endif
}

TPOW_HD uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

TPOW_HD void sha256_iv(uint32_t st[8]) {
    st[0] = 0x6a09e667u; st[1] = 0xbb67ae85u; st[2] = 0x3c6ef372u; st[3] = 0xa54ff53au;
    st[4] = 0x510e527fu; st[5] = 0x9b05688cu; st[6] = 0x1f83d9abu; st[7] = 0x5be0cd19u;
}

// One SHA-256 compression on 16 big-endian message words.
TPOW_HD void sha256_compress(uint32_t st[8], const uint32_t msg[16]) {
    uint32_t w[16];
TPOW_UNROLL
    for (int i = 0; i < 16; ++i) w[i] = msg[i];
    uint32_t a = st[0], b = st[1], c = st[2], d = st[3], e = st[4], f = st[5], g = st[6], h = st[7];
TPOW_UNROLL
    for (int i = 0; i < 64; ++i) {
        uint32_t wi;
        if (i < 16) {
            wi = w[i];
        } else {
            uint32_t w15 = w[(i - 15) & 15], w2 = w[(i - 2) & 15];
            uint32_t s0 = rotr(w15, 7) ^ rotr(w15, 18) ^ (w15 >> 3);
            uint32_t s1 = rotr(w2, 17) ^ rotr(w2, 19) ^ (w2 >> 10);
            wi = w[i & 15] + s0 + w[(i - 7) & 15] + s1;
            w[i & 15] = wi;
        }
        uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        uint32_t ch = (e & f) ^ (~e & g);
        uint32_t t1 = h + S1 + ch + sha_k(i) + wi;
        uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2 = S0 + mj;
        h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    st[0] += a; st[1] += b; st[2] += c; st[3] += d; st[4] += e; st[5] += f; st[6] += g; st[7] += h;
}

TPOW_HD uint32_t bswap32(uint32_t x) {
    return (x >> 24) | ((x >> 8) & 0xff00u) | ((x << 8) & 0xff0000u) | (x << 24);
}

// Seed as 8 big-endian words (word i = bytes 4i..4i+3 read big-endian).
struct SeedWords { uint32_t w[8]; };

inline SeedWords seed_words(const uint8_t seed[32]) {
    SeedWords s;
    for (int i = 0; i < 8; ++i)
        s.w[i] = (uint32_t(seed[4 * i]) << 24) | (uint32_t(seed[4 * i + 1]) << 16) |
                 (uint32_t(seed[4 * i + 2]) << 8) | uint32_t(seed[4 * i + 3]);
    return s;
}

// PRG block = SHA-256("tpow/v1P" || seed || le32(stream) || le64(ctr)), single compression.
// Output as 8 state words; byte j of the block is byte (j&3) of word j>>2, big-endian.
TPOW_HD void prg_block(const SeedWords& s, uint32_t stream, uint64_t ctr, uint32_t out[8]) {
    uint32_t m[16];
    m[0] = 0x74706f77u;  // "tpow"
    m[1] = 0x2f763150u;  // "/v1P"
TPOW_UNROLL
    for (int i = 0; i < 8; ++i) m[2 + i] = s.w[i];
    m[10] = bswap32(stream);
    m[11] = bswap32(static_cast<uint32_t>(ctr));
    m[12] = bswap32(static_cast<uint32_t>(ctr >> 32));
    m[13] = 0x80000000u;
    m[14] = 0;
    m[15] = 52u * 8u;
    sha256_iv(out);
    sha256_compress(out, m);
}

// Byte-oriented streaming SHA-256 (used for unit digests).
struct Sha256 {
    uint32_t st[8];
    uint32_t buf[16];
    uint32_t nbuf;   // bytes in buf
    uint64_t total;  // bytes absorbed

    TPOW_HD void init() { sha256_iv(st); nbuf = 0; total = 0; }

    TPOW_HD void put(uint8_t byte) {
        uint32_t wi = nbuf >> 2, sh = 24 - 8 * (nbuf & 3);
        if ((nbuf & 3) == 0) buf[wi] = 0;
        buf[wi] |= uint32_t(byte) << sh;
        ++nbuf;
        ++total;
        if (nbuf == 64) { sha256_compress(st, buf); nbuf = 0; }
    }

    TPOW_HD void update(const uint8_t* p, uint32_t n) { for (uint32_t i = 0; i < n; ++i) put(p[i]); }

    TPOW_HD void final(uint8_t out[32]) {
        uint64_t bits = total * 8;
        put(0x80);
        while (nbuf != 56) put(0);
        for (int i = 7; i >= 0; --i) put(static_cast<uint8_t>(bits >> (8 * i)));
        for (int i = 0; i < 8; ++i) {
            out[4 * i] = uint8_t(st[i] >> 24); out[4 * i + 1] = uint8_t(st[i] >> 16);
            out[4 * i + 2] = uint8_t(st[i] >> 8); out[4 * i + 3] = uint8_t(st[i]);
        }
    }
};

// D_u = SHA-256("tpow/v1D" || seed || le64(u) || y_rows)[0..16]
TPOW_HD void unit_digest(const uint8_t seed[32], uint64_t unit, const int8_t* y, uint32_t y_bytes, uint8_t out[16]) {
    Sha256 h;
    h.init();
    const uint8_t prefix[8] = {'t', 'p', 'o', 'w', '/', 'v', '1', 'D'};
    h.update(prefix, 8);
    h.update(seed, 32);
    for (int i = 0; i < 8; ++i) h.put(static_cast<uint8_t>(unit >> (8 * i)));
    h.update(reinterpret_cast<const uint8_t*>(y), y_bytes);
    uint8_t full[32];
    h.final(full);
    for (int i = 0; i < 16; ++i) out[i] = full[i];
}

// cuBLASLt arguments for one layer, in column-major BLAS terms.
// Row-major activations X (rows x in) and row-major weights W (out x in, W[j][k]):
// we want row-major C (rows x out) = X * W^T, i.e. column-major C^T (out x rows)
// = op(A) * op(B) with A = W viewed column-major (in x out, lda=in), opA = T,
//                     B = X viewed column-major (in x rows, ldb=in), opB = N,
//                     C column-major (out x rows, ldc=out).
// This is the "TN" form that INT8 IMMA requires.
struct GemmArgs {
    int m, n, k, lda, ldb, ldc;
    bool transa, transb;
};

TPOW_HD GemmArgs layer_gemm_args(int out, int in, int rows) {
    GemmArgs g;
    g.m = out; g.n = rows; g.k = in;
    g.lda = in; g.ldb = in; g.ldc = out;
    g.transa = true; g.transb = false;
    return g;
}

}  // namespace tpow
