//! Normative constants and parameters of TPoW-v1 (see SPEC.md).

use serde::{Deserialize, Serialize};

/// 8-byte domain prefix of every PRG block message.
pub const PRG_PREFIX: [u8; 8] = *b"tpow/v1P";
/// 8-byte domain prefix of the unit digest message.
pub const DIGEST_PREFIX: [u8; 8] = *b"tpow/v1D";
/// 8-byte domain prefix used by the harness to derive per-trial seeds.
pub const SEED_PREFIX: [u8; 8] = *b"tpow/v1S";

/// Odd multiplier of the requantisation map `q` (2^32 / golden ratio).
pub const Q_MUL: u32 = 0x9E37_79B1;

/// Bytes of a unit digest returned by a solver (truncated SHA-256).
pub const DIGEST_BYTES: usize = 16;

/// PRG stream id of the input rows `x0`.
pub const STREAM_X: u32 = 0x5800_0000; // 'X' << 24

/// PRG stream id of the weight matrix of layer `l`.
pub const fn stream_w(l: u32) -> u32 {
    0x5700_0000 | l // 'W' << 24
}

/// PRG stream id of the bias vector of layer `l`.
pub const fn stream_b(l: u32) -> u32 {
    0x4200_0000 | l // 'B' << 24
}

/// The requantisation map: `q(c, b) = int8( ((c + b) * Q_MUL mod 2^32) >> 24 )`.
///
/// `c` is the exact int32 accumulator, `b` the per-output-column bias. All
/// arithmetic is modulo 2^32; the result is the top byte reinterpreted as a
/// two's-complement int8. Every bit of `c` influences the output byte, so the
/// accumulator must be computed exactly.
#[inline(always)]
pub fn q(c: i32, b: u32) -> i8 {
    ((c as u32).wrapping_add(b).wrapping_mul(Q_MUL) >> 24) as u8 as i8
}

/// Shape parameters of a TPoW function instance.
///
/// The function is a pseudorandom int8 MLP applied independently to rows:
/// embed (`d x e`), `hidden_layers` hidden layers (`d x d`), head (`h x d`).
/// A *unit* is `m` consecutive rows; the solver returns one digest per unit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Params {
    pub d: u32,
    pub hidden_layers: u32,
    pub e: u32,
    pub h: u32,
    pub m: u32,
}

impl Params {
    /// The ranked function, `tpow-v1`.
    pub const V1: Params = Params { d: 4096, hidden_layers: 4, e: 64, h: 32, m: 16 };
    /// A small instance for tests and CPU-only development. Not ranked.
    pub const TINY: Params = Params { d: 256, hidden_layers: 2, e: 64, h: 32, m: 4 };

    pub fn by_name(name: &str) -> Option<Params> {
        match name {
            "tpow-v1" => Some(Self::V1),
            "tpow-tiny" => Some(Self::TINY),
            _ => None,
        }
    }

    pub fn name(&self) -> String {
        if *self == Self::V1 {
            "tpow-v1".into()
        } else if *self == Self::TINY {
            "tpow-tiny".into()
        } else {
            format!(
                "tpow-custom-d{}-L{}-e{}-h{}-m{}",
                self.d, self.hidden_layers, self.e, self.h, self.m
            )
        }
    }

    /// Total number of layers: embed + hidden + head.
    pub fn num_layers(&self) -> u32 {
        self.hidden_layers + 2
    }

    /// `(out, in)` of layer `l` (0 = embed, 1..=hidden_layers = hidden, last = head).
    pub fn layer_shape(&self, l: u32) -> (usize, usize) {
        let (d, e, h) = (self.d as usize, self.e as usize, self.h as usize);
        if l == 0 {
            (d, e)
        } else if l <= self.hidden_layers {
            (d, d)
        } else if l == self.hidden_layers + 1 {
            (h, d)
        } else {
            panic!("layer {l} out of range")
        }
    }

    /// Canonical operation count per row: 2 x multiply-accumulates.
    pub fn ops_per_row(&self) -> u64 {
        (0..self.num_layers())
            .map(|l| {
                let (o, i) = self.layer_shape(l);
                2 * (o as u64) * (i as u64)
            })
            .sum()
    }

    pub fn ops_per_unit(&self) -> u64 {
        self.m as u64 * self.ops_per_row()
    }

    /// Bytes of all weight matrices.
    pub fn weight_bytes(&self) -> u64 {
        (0..self.num_layers())
            .map(|l| {
                let (o, i) = self.layer_shape(l);
                (o * i) as u64
            })
            .sum()
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.d == 0 || self.e == 0 || self.h == 0 || self.m == 0 {
            return Err("all dimensions must be positive".into());
        }
        if self.e % 32 != 0 {
            return Err("e must be a multiple of 32 (rows start on PRG block boundaries)".into());
        }
        if self.d % 64 != 0 || self.e % 64 != 0 {
            return Err("d and e must be multiples of 64".into());
        }
        if self.h % 16 != 0 {
            return Err("h must be a multiple of 16".into());
        }
        // |c| <= in * 128 * 128 must fit in int32 so that int32 accumulation is exact.
        if self.d > 131_072 {
            return Err("d too large: int32 accumulation would overflow".into());
        }
        if self.num_layers() > 0x00FF_FFFF {
            return Err("too many layers".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v1_numbers() {
        let p = Params::V1;
        p.validate().unwrap();
        assert_eq!(p.num_layers(), 6);
        assert_eq!(p.ops_per_row(), 2 * (4096 * 64 + 4 * 4096 * 4096 + 32 * 4096));
        assert_eq!(p.ops_per_row(), 135_004_160);
        assert_eq!(p.ops_per_unit(), 2_160_066_560);
        assert_eq!(p.weight_bytes(), 4096 * 64 + 4 * 4096 * 4096 + 32 * 4096);
        Params::TINY.validate().unwrap();
    }

    #[test]
    fn q_examples() {
        assert_eq!(q(0, 0), 0);
        // 1 * 0x9E3779B1 >> 24 = 0x9E -> -98
        assert_eq!(q(1, 0), 0x9Eu8 as i8);
        assert_eq!(q(0, 1), q(1, 0));
        assert_eq!(q(-1, 0), ((0xFFFF_FFFFu32.wrapping_mul(Q_MUL)) >> 24) as u8 as i8);
        // Every bit of the accumulator matters: flipping bit 0 of c changes the output.
        for c in [12345i32, -777, 1 << 20] {
            assert_ne!(q(c, 99), q(c ^ 1, 99));
        }
    }
}
