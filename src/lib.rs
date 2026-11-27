//! TPoW-v1 — a tensor-core proof of work.
//!
//! This crate is the *normative reference* for the function specified in
//! `SPEC.md`, plus the trusted benchmark harness (`tpow-bench`) and a CPU
//! reference solver (`tpow-cpu-solver`). Contestant code (CUDA, under
//! `solver/`) is never linked into anything built from this crate.

pub mod harness;
pub mod hexutil;
pub mod kernels;
pub mod model;
pub mod prg;
pub mod protocol;
pub mod spec;

pub use model::{unit_digest, Model};
pub use spec::Params;
