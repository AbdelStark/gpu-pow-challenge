/*
 * Protected worker main: speaks the TPoW worker protocol on stdin/stdout and
 * calls the contestant's solver through worker/tpow_solver.h.
 *
 * PROTECTED FILE: submissions may not modify anything under worker/.
 * This file is a convenience, not a trust boundary: it is compiled into the
 * untrusted solver binary. Trust comes from the external harness (tpow-bench).
 *
 *   stdin : "PARAMS d=.. L=.. e=.. h=.. m=..\n", later "RUN <seed hex> <first> <n>\n"
 *   stdout: "READY tpow-v1\n", later exactly n*16 raw digest bytes
 *   stderr: free-form logs (bounded by the harness)
 */
#include "tpow_solver.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>

namespace {

struct EmitState {
    uint64_t expected = 0;
    uint64_t emitted = 0;
};

bool read_line(std::string& out) {
    out.clear();
    int c;
    while ((c = std::getchar()) != EOF) {
        if (c == '\n') return true;
        out.push_back(static_cast<char>(c));
        if (out.size() > 4096) return false;
    }
    return !out.empty();
}

int write_all(const uint8_t* p, size_t n) {
    while (n > 0) {
        ssize_t k = ::write(STDOUT_FILENO, p, n);
        if (k <= 0) return -1;
        p += k;
        n -= static_cast<size_t>(k);
    }
    return 0;
}

int emit(void* ctx, const uint8_t* digests, uint64_t n_units) {
    auto* st = static_cast<EmitState*>(ctx);
    if (st->emitted + n_units > st->expected) {
        std::fprintf(stderr, "worker: solver emitted too many digests (%llu > %llu)\n",
                     (unsigned long long)(st->emitted + n_units), (unsigned long long)st->expected);
        return -1;
    }
    if (write_all(digests, n_units * TPOW_DIGEST_BYTES) != 0) return -1;
    st->emitted += n_units;
    return 0;
}

int hexval(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

}  // namespace

int main() {
    std::string line;
    if (!read_line(line)) return 1;
    tpow_params_t p{};
    unsigned d, L, e, h, m;
    if (std::sscanf(line.c_str(), "PARAMS d=%u L=%u e=%u h=%u m=%u", &d, &L, &e, &h, &m) != 5) {
        std::fprintf(stderr, "worker: bad PARAMS line\n");
        return 1;
    }
    p.d = d; p.hidden_layers = L; p.e = e; p.h = h; p.m = m;
    if (tpow_init(&p) != 0) {
        std::fprintf(stderr, "worker: tpow_init failed\n");
        return 1;
    }
    std::printf("READY tpow-v1\n");
    std::fflush(stdout);

    while (read_line(line)) {
        if (line.empty()) continue;
        char seed_hex[65] = {0};
        unsigned long long first = 0, n = 0;
        if (std::sscanf(line.c_str(), "RUN %64s %llu %llu", seed_hex, &first, &n) != 3 || std::strlen(seed_hex) != 64) {
            std::fprintf(stderr, "worker: bad RUN line\n");
            return 1;
        }
        uint8_t seed[32];
        for (int i = 0; i < 32; ++i) {
            int hi = hexval(seed_hex[2 * i]), lo = hexval(seed_hex[2 * i + 1]);
            if (hi < 0 || lo < 0) return 1;
            seed[i] = static_cast<uint8_t>(hi << 4 | lo);
        }
        EmitState st;
        st.expected = n;
        if (tpow_run(seed, first, n, emit, &st) != 0) {
            std::fprintf(stderr, "worker: tpow_run failed\n");
            return 1;
        }
        if (st.emitted != n) {
            std::fprintf(stderr, "worker: solver emitted %llu of %llu digests\n",
                         (unsigned long long)st.emitted, n);
            return 1;
        }
    }
    return 0;
}
