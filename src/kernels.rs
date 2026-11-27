//! CPU int8 GEMM + requantisation kernels used by the reference/verifier.
//!
//! `layer_forward` computes, for every row `x` of `xs` and every output `j`,
//! `y[j] = q( sum_k W[j][k] * x[k], b[j] )` with exact int32 accumulation.
//!
//! Two implementations: an AVX-512 VNNI kernel (runtime-detected) and a
//! portable scalar kernel. Both are exact; tests check they agree.

use crate::spec::q;
use rayon::prelude::*;

/// Rows processed per rayon task. Each weight row (<= 4 KiB) is reused across
/// all rows of a tile while it sits in L1.
const ROW_TILE: usize = 32;

/// One dense layer: `out x inp` int8 weights (row-major, W[j][k] at j*inp+k),
/// per-row weight sums (for the VNNI unsigned-offset trick) and u32 biases.
pub struct Layer {
    pub out: usize,
    pub inp: usize,
    pub w: Vec<i8>,
    pub wsum: Vec<i32>,
    pub b: Vec<u32>,
}

impl Layer {
    pub fn new(out: usize, inp: usize, w: Vec<i8>, b: Vec<u32>) -> Layer {
        assert_eq!(w.len(), out * inp);
        assert_eq!(b.len(), out);
        let wsum = w
            .chunks_exact(inp)
            .map(|r| r.iter().map(|&v| v as i32).sum())
            .collect();
        Layer { out, inp, w, wsum, b }
    }
}

pub fn vnni_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("avx512vnni")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Apply `layer` to `rows` rows stored contiguously in `xs` (rows x layer.inp).
pub fn layer_forward(layer: &Layer, xs: &[i8], rows: usize) -> Vec<i8> {
    layer_forward_impl(layer, xs, rows, vnni_available() && layer.inp % 64 == 0)
}

pub fn layer_forward_impl(layer: &Layer, xs: &[i8], rows: usize, use_vnni: bool) -> Vec<i8> {
    assert_eq!(xs.len(), rows * layer.inp);
    let mut ys = vec![0i8; rows * layer.out];
    ys.par_chunks_mut(ROW_TILE * layer.out)
        .zip(xs.par_chunks(ROW_TILE * layer.inp))
        .for_each(|(ytile, xtile)| {
            #[cfg(target_arch = "x86_64")]
            if use_vnni {
                // SAFETY: guarded by runtime feature detection in `layer_forward`.
                unsafe { tile_vnni(layer, xtile, ytile) };
                return;
            }
            let _ = use_vnni;
            tile_scalar(layer, xtile, ytile);
        });
    ys
}

fn tile_scalar(layer: &Layer, xtile: &[i8], ytile: &mut [i8]) {
    let (inp, out) = (layer.inp, layer.out);
    let nrows = xtile.len() / inp;
    for j in 0..out {
        let w = &layer.w[j * inp..(j + 1) * inp];
        let b = layer.b[j];
        for r in 0..nrows {
            let x = &xtile[r * inp..(r + 1) * inp];
            let c: i32 = x.iter().zip(w).map(|(&a, &bb)| a as i32 * bb as i32).sum();
            ytile[r * out + j] = q(c, b);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn tile_vnni(layer: &Layer, xtile: &[i8], ytile: &mut [i8]) {
    use std::arch::x86_64::*;
    let (inp, out) = (layer.inp, layer.out);
    let nrows = xtile.len() / inp;
    // vpdpbusd multiplies unsigned bytes by signed bytes. Shift activations to
    // unsigned (x + 128 == x ^ 0x80) and correct with 128 * sum_k W[j][k].
    let xu: Vec<u8> = xtile.iter().map(|&v| (v as u8) ^ 0x80).collect();
    let chunks = inp / 64;
    let mut r0 = 0;
    while r0 < nrows {
        let rn = (nrows - r0).min(4);
        for j in 0..out {
            let wrow = layer.w.as_ptr().add(j * inp);
            let corr = 128i32.wrapping_mul(layer.wsum[j]);
            let b = layer.b[j];
            let mut acc = [_mm512_setzero_si512(); 4];
            for c in 0..chunks {
                let wv = _mm512_loadu_si512(wrow.add(64 * c) as *const _);
                for r in 0..rn {
                    let xv = _mm512_loadu_si512(xu.as_ptr().add((r0 + r) * inp + 64 * c) as *const _);
                    acc[r] = _mm512_dpbusd_epi32(acc[r], xv, wv);
                }
            }
            for r in 0..rn {
                let s = _mm512_reduce_add_epi32(acc[r]).wrapping_sub(corr);
                ytile[(r0 + r) * out + j] = q(s, b);
            }
        }
        r0 += rn;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(state: &mut u64) -> u8 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (*state >> 56) as u8
    }

    #[test]
    fn vnni_matches_scalar_including_extremes() {
        if !vnni_available() {
            eprintln!("skipping: no AVX-512 VNNI");
            return;
        }
        let mut s = 42u64;
        for &(out, inp, rows) in &[(64usize, 64usize, 7usize), (96, 256, 37), (32, 4096, 5)] {
            let mut w: Vec<i8> = (0..out * inp).map(|_| lcg(&mut s) as i8).collect();
            let mut xs: Vec<i8> = (0..rows * inp).map(|_| lcg(&mut s) as i8).collect();
            // Extreme rows/cols: all -128 and all 127 to exercise the offset trick.
            for k in 0..inp {
                w[k] = -128;
                xs[k] = -128;
                w[inp + k] = 127;
                xs[inp + k] = 127;
            }
            let b: Vec<u32> = (0..out).map(|i| (i as u32).wrapping_mul(0x01000193)).collect();
            let layer = Layer::new(out, inp, w, b);
            let a = layer_forward_impl(&layer, &xs, rows, false);
            let v = layer_forward_impl(&layer, &xs, rows, true);
            assert_eq!(a, v, "shape {out}x{inp} rows {rows}");
        }
    }
}
