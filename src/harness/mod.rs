//! TRUSTED benchmark harness components (used by `tpow-bench`).
//!
//! Nothing in here ever links or loads contestant code. The contestant's
//! solver is a separate executable that runs in a sandbox and talks to us
//! over pipes (see `crate::protocol` and docs/HARNESS.md).

pub mod config;
pub mod gpu;
pub mod procstat;
pub mod rng;
pub mod sandbox;
pub mod stats;
pub mod worker;
