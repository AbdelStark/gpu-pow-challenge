//! Small statistics helpers.

/// Quantile with linear interpolation at rank (n-1)*q (same convention as flock).
pub fn quantile(xs: &[f64], q: f64) -> f64 {
    assert!(!xs.is_empty());
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pos = (v.len() - 1) as f64 * q;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    v[lo] + (v[hi] - v[lo]) * (pos - lo as f64)
}

pub fn median(xs: &[f64]) -> f64 {
    quantile(xs, 0.5)
}

pub fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    xs.iter().sum::<f64>() / xs.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn med() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&[4.0, 1.0, 2.0, 3.0]), 2.5);
    }
}
