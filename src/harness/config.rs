//! Leaderboard configuration (leaderboards/*.json).

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GpuRequirement {
    /// Every string must occur in the nvidia-smi product name.
    pub name_must_contain: Vec<String>,
    /// No string may occur in the nvidia-smi product name.
    #[serde(default)]
    pub name_must_not_contain: Vec<String>,
    /// Dense INT8 tensor-core peak at the reference clock (datasheet), TOPS.
    pub peak_int8_dense_tops: f64,
    /// SM clock (MHz) at which `peak_int8_dense_tops` is quoted.
    pub peak_reference_clock_mhz: f64,
    /// Fail if the board power limit differs from its default (capped or raised).
    #[serde(default = "yes")]
    pub require_default_power_limit: bool,
}

fn yes() -> bool {
    true
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Leaderboard {
    pub name: String,
    pub description: String,
    /// Function instance, e.g. "tpow-v1".
    pub function: String,
    /// `null` for CPU-only development leaderboards.
    pub gpu: Option<GpuRequirement>,
    pub units_per_trial: u64,
    /// Units of the untimed fixed-seed conformance run (checked in full).
    pub conformance_units: u64,
    pub warmup_trials: u32,
    pub measured_trials: u32,
    /// Random units re-computed on the CPU per timed trial (first and last unit are always added).
    pub audit_samples_per_trial: u64,
    /// Max average CPU cores the worker tree may burn inside the timed window.
    /// `null` = unlimited (the GPU+CPU red-team leaderboard).
    pub cpu_budget_cores: Option<f64>,
    pub init_timeout_s: u64,
    pub trial_timeout_s: u64,
    /// Wait until GPU temperature <= this before each timed trial (null = off).
    pub thermal_gate_c: Option<f64>,
    pub thermal_gate_max_wait_s: u64,
    /// "bwrap" (required), "auto" (bwrap if present, else unconfined with a warning).
    pub sandbox: String,
    /// Use a user namespace inside the sandbox (uid 65534). Some driver setups
    /// fail cuInit under unprivileged user namespaces; then set false.
    #[serde(default = "yes")]
    pub sandbox_user_namespace: bool,
    /// Whether results from this leaderboard can be official (false for dev configs).
    pub official: bool,
    /// nvcc -arch for the solver build (read by benchmark.sh), e.g. "sm_90a".
    /// `null` for CPU-only leaderboards.
    #[serde(default)]
    pub cuda_arch: Option<String>,
}

impl Leaderboard {
    pub fn load(path: &str) -> Result<Leaderboard, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("read {path}: {e}"))?;
        let lb: Leaderboard =
            serde_json::from_str(&text).map_err(|e| format!("parse {path}: {e}"))?;
        lb.validate()?;
        Ok(lb)
    }

    pub fn validate(&self) -> Result<(), String> {
        if crate::Params::by_name(&self.function).is_none() {
            return Err(format!("unknown function {:?}", self.function));
        }
        if self.units_per_trial < 2 || self.measured_trials == 0 {
            return Err("units_per_trial >= 2 and measured_trials >= 1 required".into());
        }
        if !matches!(self.sandbox.as_str(), "bwrap" | "auto") {
            return Err(format!("sandbox must be \"bwrap\" or \"auto\", got {:?}", self.sandbox));
        }
        Ok(())
    }
}
