//! The TPoW-v1 function: model generation, row forward pass, unit digests.

use crate::kernels::{layer_forward, Layer};
use crate::prg;
use crate::spec::{stream_b, stream_w, Params, DIGEST_BYTES, DIGEST_PREFIX, STREAM_X};
use rayon::prelude::*;
use sha2::{Digest, Sha256};

/// A fully materialised TPoW instance for one seed.
pub struct Model {
    pub params: Params,
    pub seed: [u8; 32],
    pub layers: Vec<Layer>,
}

impl Model {
    /// Generate all weights and biases from `seed` (SPEC.md §4).
    pub fn generate(params: Params, seed: [u8; 32]) -> Model {
        params.validate().expect("invalid params");
        let layers = (0..params.num_layers())
            .map(|l| {
                let (out, inp) = params.layer_shape(l);
                let mut wb = vec![0u8; out * inp];
                prg::fill(&seed, stream_w(l), 0, &mut wb);
                let w: Vec<i8> = wb.into_iter().map(|v| v as i8).collect();
                let mut bb = vec![0u8; out * 4];
                prg::fill(&seed, stream_b(l), 0, &mut bb);
                let b = bb
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                    .collect();
                Layer::new(out, inp, w, b)
            })
            .collect();
        Model { params, seed, layers }
    }

    /// The `m * e` input bytes of `unit` (rows unit*m .. unit*m+m-1 of stream X).
    pub fn input_rows(&self, unit: u64, out: &mut [i8]) {
        let p = &self.params;
        let row_bytes = p.e as u64;
        assert_eq!(out.len() as u64, p.m as u64 * row_bytes);
        let first_row = unit.checked_mul(p.m as u64).expect("unit index overflow");
        let first_block = first_row * (row_bytes / 32);
        // SAFETY-free reinterpretation: i8 and u8 have identical layout.
        let bytes: &mut [u8] =
            unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, out.len()) };
        prg::fill(&self.seed, STREAM_X, first_block, bytes);
    }

    /// Forward `rows` input rows (rows x e) through all layers, returning rows x h.
    pub fn forward(&self, x0: &[i8], rows: usize) -> Vec<i8> {
        let mut x = layer_forward(&self.layers[0], x0, rows);
        for layer in &self.layers[1..] {
            x = layer_forward(layer, &x, rows);
        }
        x
    }

    /// Head outputs (m x h) of each requested unit, batched for throughput.
    pub fn unit_outputs(&self, units: &[u64]) -> Vec<Vec<i8>> {
        let p = &self.params;
        let (m, e, h) = (p.m as usize, p.e as usize, p.h as usize);
        let mut x0 = vec![0i8; units.len() * m * e];
        x0.par_chunks_mut(m * e)
            .zip(units.par_iter())
            .for_each(|(chunk, &u)| self.input_rows(u, chunk));
        let y = self.forward(&x0, units.len() * m);
        y.chunks_exact(m * h).map(|c| c.to_vec()).collect()
    }

    /// Unit digests of the requested units.
    pub fn unit_digests(&self, units: &[u64]) -> Vec<[u8; DIGEST_BYTES]> {
        // Bound peak memory: process at most 2048 units per batch.
        let mut out = Vec::with_capacity(units.len());
        for batch in units.chunks(2048) {
            let ys = self.unit_outputs(batch);
            out.extend(batch.iter().zip(ys.iter()).map(|(&u, y)| unit_digest(&self.seed, u, y)));
        }
        out
    }
}

/// D_u = SHA-256( "tpow/v1D" || seed || le64(u) || y_{u,0} || ... || y_{u,m-1} )[0..16]
pub fn unit_digest(seed: &[u8; 32], unit: u64, y: &[i8]) -> [u8; DIGEST_BYTES] {
    let mut hasher = Sha256::new();
    hasher.update(DIGEST_PREFIX);
    hasher.update(seed);
    hasher.update(unit.to_le_bytes());
    let bytes: &[u8] = unsafe { std::slice::from_raw_parts(y.as_ptr() as *const u8, y.len()) };
    hasher.update(bytes);
    let full: [u8; 32] = hasher.finalize().into();
    full[..DIGEST_BYTES].try_into().unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::q;

    /// Naive, obviously-correct evaluation of one row, independent of kernels.rs.
    fn naive_row(model: &Model, x0: &[i8]) -> Vec<i8> {
        let mut x = x0.to_vec();
        for layer in &model.layers {
            let mut y = vec![0i8; layer.out];
            for j in 0..layer.out {
                let mut c: i64 = 0;
                for k in 0..layer.inp {
                    c += layer.w[j * layer.inp + k] as i64 * x[k] as i64;
                }
                assert!(c >= i32::MIN as i64 && c <= i32::MAX as i64);
                y[j] = q(c as i32, layer.b[j]);
            }
            x = y;
        }
        x
    }

    #[test]
    fn batched_forward_matches_naive() {
        let p = Params::TINY;
        let model = Model::generate(p, [3u8; 32]);
        let units = [0u64, 1, 17, 1 << 40];
        let ys = model.unit_outputs(&units);
        for (&u, y) in units.iter().zip(&ys) {
            let mut x0 = vec![0i8; (p.m * p.e) as usize];
            model.input_rows(u, &mut x0);
            for r in 0..p.m as usize {
                let row = &x0[r * p.e as usize..(r + 1) * p.e as usize];
                let want = naive_row(&model, row);
                assert_eq!(&y[r * p.h as usize..(r + 1) * p.h as usize], &want[..]);
            }
        }
    }

    #[test]
    fn digests_depend_on_seed_and_unit() {
        let p = Params::TINY;
        let a = Model::generate(p, [1u8; 32]).unit_digests(&[0, 1]);
        let b = Model::generate(p, [2u8; 32]).unit_digests(&[0, 1]);
        assert_ne!(a[0], a[1]);
        assert_ne!(a[0], b[0]);
    }

    #[test]
    fn activations_look_uniform() {
        // q should produce ~uniform int8 activations (no sparsity to exploit).
        let p = Params::TINY;
        let model = Model::generate(p, [9u8; 32]);
        let mut x0 = vec![0i8; 64 * p.e as usize];
        for u in 0..(64 / p.m as u64) {
            model.input_rows(u, &mut x0[(u as usize) * (p.m * p.e) as usize..][..(p.m * p.e) as usize]);
        }
        let x1 = layer_forward(&model.layers[0], &x0, 64);
        let zeros = x1.iter().filter(|&&v| v == 0).count();
        let mean: f64 = x1.iter().map(|&v| v as f64).sum::<f64>() / x1.len() as f64;
        assert!(zeros < x1.len() / 64, "too many zeros: {zeros}");
        assert!(mean.abs() < 3.0, "mean {mean}");
    }
}
