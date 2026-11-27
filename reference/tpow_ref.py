#!/usr/bin/env python3
"""Independent Python/numpy reference of TPoW (written from SPEC.md only).

    python3 reference/tpow_ref.py check tests/vectors/tpow-tiny.json [max_units]
    python3 reference/tpow_ref.py digests tpow-v1 <seed-hex> <first> <n>

Slow but simple; used to cross-check the Rust reference and the CUDA device code.
"""
import hashlib
import json
import struct
import sys

import numpy as np

PARAMS = {
    "tpow-v1": dict(d=4096, hidden_layers=4, e=64, h=32, m=16),
    "tpow-tiny": dict(d=256, hidden_layers=2, e=64, h=32, m=4),
}
Q_MUL = 0x9E3779B1


def stream_bytes(seed: bytes, stream: int, first_block: int, nbytes: int) -> bytes:
    out = bytearray()
    blk = first_block
    prefix = b"tpow/v1P" + seed + struct.pack("<I", stream)
    while len(out) < nbytes:
        out += hashlib.sha256(prefix + struct.pack("<Q", blk)).digest()
        blk += 1
    return bytes(out[:nbytes])


def layer_shapes(p):
    d, L, e, h = p["d"], p["hidden_layers"], p["e"], p["h"]
    return [(d, e)] + [(d, d)] * L + [(h, d)]


def generate(p, seed):
    layers = []
    for l, (out, inp) in enumerate(layer_shapes(p)):
        w = np.frombuffer(stream_bytes(seed, 0x57000000 | l, 0, out * inp), dtype=np.int8).reshape(out, inp)
        b = np.frombuffer(stream_bytes(seed, 0x42000000 | l, 0, out * 4), dtype="<u4")
        layers.append((w.astype(np.int64), b.astype(np.uint64)))
    return layers


def q(c: np.ndarray, b: np.ndarray) -> np.ndarray:
    v = ((c.astype(np.int64) & 0xFFFFFFFF).astype(np.uint64) + b) & 0xFFFFFFFF
    v = (v * Q_MUL) & 0xFFFFFFFF
    return (v >> 24).astype(np.uint8).view(np.int8)


def unit_digest(p, seed, layers, unit: int) -> bytes:
    m, e = p["m"], p["e"]
    x = np.frombuffer(stream_bytes(seed, 0x58000000, unit * m * e // 32, m * e), dtype=np.int8).reshape(m, e)
    for w, b in layers:
        c = x.astype(np.int64) @ w.T  # exact: |c| < 2^31
        assert np.abs(c).max() < 2**31
        x = q(c, b[None, :])
    msg = b"tpow/v1D" + seed + struct.pack("<Q", unit) + x.tobytes()
    return hashlib.sha256(msg).digest()[:16]


def main():
    args = sys.argv[1:]
    if args and args[0] == "check":
        with open(args[1]) as f:
            vec = json.load(f)
        max_units = int(args[2]) if len(args) > 2 else 1 << 30
        p = vec["params"]
        seed = bytes.fromhex(vec["seed"])
        assert stream_bytes(seed, 0x58000000, 0, 32).hex() == vec["x_prg_block0"]
        layers = generate(p, seed)
        for l, ((w, b), lv) in enumerate(zip(layers, vec["layers"])):
            assert w[0, :16].astype(np.int8).tobytes().hex() == lv["w_first16"], f"layer {l} weights"
            assert [int(v) for v in b[:4]] == lv["b_first4"], f"layer {l} biases"
        n = 0
        for item in vec["digests"][:max_units]:
            got = unit_digest(p, seed, layers, item["unit"]).hex()
            status = "ok  " if got == item["digest"] else "FAIL"
            print(f"{status} unit {item['unit']} {got}")
            if got != item["digest"]:
                sys.exit(1)
            n += 1
        print(f"PYTHON REFERENCE OK ({vec['function']}, {n} units)")
    elif args and args[0] == "digests" and len(args) == 5:
        p = PARAMS[args[1]]
        seed = bytes.fromhex(args[2])
        layers = generate(p, seed)
        for u in range(int(args[3]), int(args[3]) + int(args[4])):
            print(unit_digest(p, seed, layers, u).hex())
    else:
        print(__doc__)
        sys.exit(2)


if __name__ == "__main__":
    main()
