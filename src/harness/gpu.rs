//! GPU identification, gates and telemetry via `nvidia-smi` (no NVML binding
//! needed; nvidia-smi ships with every driver).

use serde::Serialize;
use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Clone, Debug, Serialize)]
pub struct GpuInfo {
    pub index: u32,
    pub uuid: String,
    pub name: String,
    pub driver_version: String,
    pub memory_total_mib: String,
    pub power_limit_w: f64,
    pub power_default_limit_w: f64,
    pub clocks_max_sm_mhz: f64,
    pub compute_cap: String,
    pub persistence_mode: String,
    pub pci_bus_id: String,
    pub device_minor: Option<u32>,
}

fn smi(args: &[&str]) -> Result<String, String> {
    let out = Command::new("nvidia-smi")
        .args(args)
        .output()
        .map_err(|e| format!("nvidia-smi: {e}"))?;
    if !out.status.success() {
        return Err(format!("nvidia-smi {:?} failed: {}", args, String::from_utf8_lossy(&out.stderr)));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

fn num(s: &str) -> f64 {
    s.trim().trim_end_matches(" W").trim_end_matches(" MHz").parse().unwrap_or(f64::NAN)
}

pub fn query(index: u32) -> Result<GpuInfo, String> {
    let fields = "index,uuid,name,driver_version,memory.total,power.limit,power.default_limit,clocks.max.sm,compute_cap,persistence_mode,pci.bus_id";
    let out = smi(&[
        &format!("--query-gpu={fields}"),
        "--format=csv,noheader,nounits",
        &format!("--id={index}"),
    ])?;
    let line = out.lines().next().ok_or("nvidia-smi returned no GPU")?;
    let f: Vec<String> = line.split(',').map(|s| s.trim().to_string()).collect();
    if f.len() < 11 {
        return Err(format!("unexpected nvidia-smi output: {line}"));
    }
    let bus = f[10].to_lowercase();
    // /proc/driver/nvidia/gpus/<dddd:bb:dd.f>/information carries "Device Minor".
    let short_bus = if bus.len() > 12 { bus[bus.len() - 12..].to_string() } else { bus.clone() };
    let device_minor = std::fs::read_to_string(format!("/proc/driver/nvidia/gpus/{short_bus}/information"))
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("Device Minor"))
                .and_then(|l| l.split(':').nth(1))
                .and_then(|v| v.trim().parse().ok())
        });
    Ok(GpuInfo {
        index: f[0].parse().unwrap_or(index),
        uuid: f[1].clone(),
        name: f[2].clone(),
        driver_version: f[3].clone(),
        memory_total_mib: f[4].clone(),
        power_limit_w: num(&f[5]),
        power_default_limit_w: num(&f[6]),
        clocks_max_sm_mhz: num(&f[7]),
        compute_cap: f[8].clone(),
        persistence_mode: f[9].clone(),
        pci_bus_id: f[10].clone(),
        device_minor,
    })
}

pub fn temperature_c(index: u32) -> Option<f64> {
    smi(&["--query-gpu=temperature.gpu", "--format=csv,noheader,nounits", &format!("--id={index}")])
        .ok()
        .map(|s| num(s.lines().next().unwrap_or("")))
        .filter(|v| v.is_finite())
}

/// PIDs of compute processes currently holding a context on the GPU.
pub fn compute_pids(index: u32) -> Vec<String> {
    smi(&["--query-compute-apps=pid", "--format=csv,noheader", &format!("--id={index}")])
        .map(|s| s.lines().map(|l| l.trim().to_string()).filter(|l| !l.is_empty()).collect())
        .unwrap_or_default()
}

pub fn utilization_pct(index: u32) -> Option<f64> {
    smi(&["--query-gpu=utilization.gpu", "--format=csv,noheader,nounits", &format!("--id={index}")])
        .ok()
        .map(|s| num(s.lines().next().unwrap_or("")))
        .filter(|v| v.is_finite())
}

/// Wait until no foreign compute process is on the GPU and utilisation is low.
pub fn wait_quiescent(index: u32, max_wait: Duration) -> Result<(), String> {
    let t = Instant::now();
    loop {
        let pids = compute_pids(index);
        let util = utilization_pct(index).unwrap_or(0.0);
        if pids.is_empty() && util < 5.0 {
            return Ok(());
        }
        if t.elapsed() > max_wait {
            return Err(format!("GPU {index} not quiescent: compute pids {pids:?}, utilization {util}%"));
        }
        std::thread::sleep(Duration::from_millis(500));
    }
}

/// Wait until temperature <= gate (returns waited seconds and final temperature).
pub fn thermal_gate(index: u32, gate_c: f64, max_wait: Duration) -> (f64, Option<f64>, bool) {
    let t = Instant::now();
    loop {
        let temp = temperature_c(index);
        match temp {
            Some(c) if c <= gate_c => return (t.elapsed().as_secs_f64(), temp, true),
            None => return (t.elapsed().as_secs_f64(), None, false),
            _ => {}
        }
        if t.elapsed() > max_wait {
            return (t.elapsed().as_secs_f64(), temp, false);
        }
        std::thread::sleep(Duration::from_millis(500));
    }
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct TelemetrySummary {
    pub samples: usize,
    pub sm_clock_mhz_mean: f64,
    pub sm_clock_mhz_min: f64,
    pub mem_clock_mhz_mean: f64,
    pub power_w_mean: f64,
    pub power_w_max: f64,
    pub temp_c_max: f64,
    pub utilization_pct_mean: f64,
    /// Bitwise OR of clocks_event_reasons over the window (see nvidia-smi -q -d PERFORMANCE).
    pub throttle_reasons_or: String,
}

/// Background `nvidia-smi -lms` sampler covering one timed window.
pub struct Sampler {
    child: Child,
    rows: Arc<Mutex<Vec<Vec<String>>>>,
    reader: Option<std::thread::JoinHandle<()>>,
}

impl Sampler {
    pub fn start(index: u32, period_ms: u32) -> Option<Sampler> {
        let mut child = Command::new("nvidia-smi")
            .args([
                "--query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,clocks_throttle_reasons.active",
                "--format=csv,noheader,nounits",
                &format!("--id={index}"),
                &format!("--loop-ms={period_ms}"),
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .ok()?;
        let stdout = child.stdout.take()?;
        let rows = Arc::new(Mutex::new(Vec::new()));
        let rows2 = rows.clone();
        let reader = std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                let f: Vec<String> = line.split(',').map(|s| s.trim().to_string()).collect();
                if f.len() >= 6 {
                    rows2.lock().unwrap().push(f);
                }
            }
        });
        Some(Sampler { child, rows, reader: Some(reader) })
    }

    pub fn stop(mut self) -> TelemetrySummary {
        let _ = self.child.kill();
        let _ = self.child.wait();
        if let Some(r) = self.reader.take() {
            let _ = r.join();
        }
        let rows = self.rows.lock().unwrap();
        let col = |i: usize| -> Vec<f64> { rows.iter().map(|r| num(&r[i])).filter(|v| v.is_finite()).collect() };
        let (sm, mem, pw, tc, ut) = (col(0), col(1), col(2), col(3), col(4));
        let mut reasons = 0u64;
        for r in rows.iter() {
            if let Ok(v) = u64::from_str_radix(r[5].trim_start_matches("0x"), 16) {
                reasons |= v;
            }
        }
        let fmax = |v: &[f64]| v.iter().cloned().fold(f64::NAN, f64::max);
        let fmin = |v: &[f64]| v.iter().cloned().fold(f64::NAN, f64::min);
        TelemetrySummary {
            samples: rows.len(),
            sm_clock_mhz_mean: super::stats::mean(&sm),
            sm_clock_mhz_min: fmin(&sm),
            mem_clock_mhz_mean: super::stats::mean(&mem),
            power_w_mean: super::stats::mean(&pw),
            power_w_max: fmax(&pw),
            temp_c_max: fmax(&tc),
            utilization_pct_mean: super::stats::mean(&ut),
            throttle_reasons_or: format!("{reasons:#018x}"),
        }
    }
}
