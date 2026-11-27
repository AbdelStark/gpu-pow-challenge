//! Tiny hex helpers (no external dependency).

pub fn to_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 15) as usize] as char);
    }
    s
}

pub fn from_hex(s: &str) -> Result<Vec<u8>, String> {
    let s = s.trim();
    if s.len() % 2 != 0 {
        return Err(format!("odd-length hex string ({} chars)", s.len()));
    }
    fn nib(c: u8) -> Result<u8, String> {
        match c {
            b'0'..=b'9' => Ok(c - b'0'),
            b'a'..=b'f' => Ok(c - b'a' + 10),
            b'A'..=b'F' => Ok(c - b'A' + 10),
            _ => Err(format!("invalid hex character {:?}", c as char)),
        }
    }
    s.as_bytes()
        .chunks_exact(2)
        .map(|p| Ok((nib(p[0])? << 4) | nib(p[1])?))
        .collect()
}

pub fn seed_from_hex(s: &str) -> Result<[u8; 32], String> {
    let v = from_hex(s)?;
    v.as_slice()
        .try_into()
        .map_err(|_| format!("seed must be 32 bytes, got {}", v.len()))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn roundtrip() {
        let b: Vec<u8> = (0..=255).collect();
        assert_eq!(from_hex(&to_hex(&b)).unwrap(), b);
        assert!(from_hex("abc").is_err());
        assert!(from_hex("zz").is_err());
    }
}
