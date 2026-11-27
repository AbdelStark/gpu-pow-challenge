//! The TPoW-v1 PRG: SHA-256 in counter mode, one compression per 32-byte block.
//!
//! block(seed, stream, ctr) = SHA-256( "tpow/v1P" || seed[32] || le32(stream) || le64(ctr) )
//!
//! The 52-byte message always fits a single SHA-256 block, so every PRG block
//! costs exactly one compression and any block is randomly accessible.

use crate::spec::PRG_PREFIX;
use rayon::prelude::*;
use sha2::digest::generic_array::GenericArray;

pub const SHA256_IV: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

/// The padded 64-byte SHA-256 block for a PRG message.
#[inline]
pub fn padded_block(seed: &[u8; 32], stream: u32, counter: u64) -> [u8; 64] {
    let mut blk = [0u8; 64];
    blk[0..8].copy_from_slice(&PRG_PREFIX);
    blk[8..40].copy_from_slice(seed);
    blk[40..44].copy_from_slice(&stream.to_le_bytes());
    blk[44..52].copy_from_slice(&counter.to_le_bytes());
    blk[52] = 0x80;
    blk[56..64].copy_from_slice(&(52u64 * 8).to_be_bytes());
    blk
}

#[inline]
pub fn block(seed: &[u8; 32], stream: u32, counter: u64) -> [u8; 32] {
    let blk = padded_block(seed, stream, counter);
    let mut st = SHA256_IV;
    sha2::compress256(&mut st, &[*GenericArray::from_slice(&blk)]);
    let mut out = [0u8; 32];
    for (i, w) in st.iter().enumerate() {
        out[4 * i..4 * i + 4].copy_from_slice(&w.to_be_bytes());
    }
    out
}

/// Fill `out` with the stream bytes starting at block `first_block`
/// (byte offset `32 * first_block`). The tail of a partial last block is truncated.
pub fn fill(seed: &[u8; 32], stream: u32, first_block: u64, out: &mut [u8]) {
    const PAR_CHUNK: usize = 1 << 16; // bytes per rayon task
    if out.len() >= 4 * PAR_CHUNK {
        out.par_chunks_mut(PAR_CHUNK).enumerate().for_each(|(ci, chunk)| {
            fill_seq(seed, stream, first_block + (ci * PAR_CHUNK / 32) as u64, chunk)
        });
    } else {
        fill_seq(seed, stream, first_block, out);
    }
}

fn fill_seq(seed: &[u8; 32], stream: u32, first_block: u64, out: &mut [u8]) {
    for (i, chunk) in out.chunks_mut(32).enumerate() {
        let b = block(seed, stream, first_block + i as u64);
        chunk.copy_from_slice(&b[..chunk.len()]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    #[test]
    fn block_matches_plain_sha256() {
        let seed: [u8; 32] = core::array::from_fn(|i| i as u8);
        for (stream, ctr) in [(0u32, 0u64), (0x5700_0001, 7), (0x5800_0000, u64::MAX)] {
            let mut msg = Vec::new();
            msg.extend_from_slice(b"tpow/v1P");
            msg.extend_from_slice(&seed);
            msg.extend_from_slice(&stream.to_le_bytes());
            msg.extend_from_slice(&ctr.to_le_bytes());
            assert_eq!(msg.len(), 52);
            let want: [u8; 32] = Sha256::digest(&msg).into();
            assert_eq!(block(&seed, stream, ctr), want);
        }
    }

    #[test]
    fn fill_is_offset_consistent() {
        let seed = [7u8; 32];
        let mut big = vec![0u8; 300_000];
        fill(&seed, 3, 5, &mut big);
        let mut small = vec![0u8; 100];
        fill(&seed, 3, 5 + 1000, &mut small);
        assert_eq!(&big[32_000..32_100], &small[..]);
        assert_eq!(&big[0..32], &block(&seed, 3, 5));
    }
}
