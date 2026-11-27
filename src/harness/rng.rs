//! Harness-private randomness, read straight from the OS.

use std::io::Read;

pub fn os_random<const N: usize>() -> [u8; N] {
    let mut b = [0u8; N];
    std::fs::File::open("/dev/urandom")
        .and_then(|mut f| f.read_exact(&mut b))
        .expect("cannot read /dev/urandom");
    b
}

/// Uniform u64 in [0, n) by rejection sampling.
pub fn below(n: u64) -> u64 {
    assert!(n > 0);
    let zone = u64::MAX - (u64::MAX % n);
    loop {
        let v = u64::from_le_bytes(os_random::<8>());
        if v < zone {
            return v % n;
        }
    }
}

/// `k` distinct uniform indices in [0, n) plus the first and last index.
pub fn audit_indices(n: u64, k: u64) -> Vec<u64> {
    let mut set = std::collections::BTreeSet::new();
    set.insert(0);
    set.insert(n - 1);
    let target = (k + 2).min(n);
    while (set.len() as u64) < target {
        set.insert(below(n));
    }
    set.into_iter().collect()
}
