//! TRUSTED benchmark harness for TPoW.
//!
//!   tpow-bench --leaderboard leaderboards/h100-sxm.json --solver build/solver \
//!              [--beacon build/beacon.json] [--note "..."] [--gpu-index 0] \
//!              [--units N] [--warmup N] [--trials N] [--audit N]   (dev overrides: never official)
//!
//! Sequence (docs/HARNESS.md): hardware checks -> conformance run (fixed public
//! seed, every unit checked) -> warm-up trials -> measured trials. Every trial
//! runs in a FRESH sandboxed worker; the private seed is drawn only after the
//! worker is READY; the external clock runs from writing RUN to receiving the
//! last digest byte; outputs are audited on the CPU against the reference.
//! Score = median verified effective INT8 TOPS over measured trials.

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tpow::harness::config::Leaderboard;
use tpow::harness::gpu::{self, GpuInfo, Sampler, TelemetrySummary};
use tpow::harness::stats::{median, quantile};
use tpow::harness::worker::{self, WorkerSetup};
use tpow::harness::{rng, sandbox};
use tpow::hexutil::{from_hex, to_hex};
use tpow::spec::{DIGEST_BYTES, SEED_PREFIX};
use tpow::{Model, Params};

struct Args {
    leaderboard: String,
    solver: PathBuf,
    beacon: Option<String>,
    note: String,
    gpu_index: u32,
    units: Option<u64>,
    warmup: Option<u32>,
    trials: Option<u32>,
    audit: Option<u64>,
    out_dir: PathBuf,
}

fn usage() -> ! {
    eprintln!("usage: tpow-bench --leaderboard <json> --solver <exe> [--beacon <json>] [--note <text>] [--gpu-index N] [--units N] [--warmup N] [--trials N] [--audit N] [--out-dir DIR]");
    std::process::exit(2)
}

fn parse_args() -> Args {
    let mut a = Args {
        leaderboard: String::new(),
        solver: PathBuf::new(),
        beacon: None,
        note: String::new(),
        gpu_index: std::env::var("TPOW_GPU_INDEX").ok().and_then(|s| s.parse().ok()).unwrap_or(0),
        units: None,
        warmup: None,
        trials: None,
        audit: None,
        out_dir: PathBuf::from("."),
    };
    let v: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < v.len() {
        let (k, val) = match v[i].split_once('=') {
            Some((k, x)) => (k.to_string(), Some(x.to_string())),
            None => (v[i].clone(), None),
        };
        let mut take = || -> String {
            if let Some(x) = val.clone() {
                x
            } else {
                i += 1;
                v.get(i).cloned().unwrap_or_else(|| usage())
            }
        };
        match k.as_str() {
            "--leaderboard" => a.leaderboard = take(),
            "--solver" => a.solver = PathBuf::from(take()),
            "--beacon" => a.beacon = Some(take()),
            "--note" => a.note = take(),
            "--gpu-index" => a.gpu_index = take().parse().unwrap_or_else(|_| usage()),
            "--units" => a.units = Some(take().parse().unwrap_or_else(|_| usage())),
            "--warmup" => a.warmup = Some(take().parse().unwrap_or_else(|_| usage())),
            "--trials" => a.trials = Some(take().parse().unwrap_or_else(|_| usage())),
            "--audit" => a.audit = Some(take().parse().unwrap_or_else(|_| usage())),
            "--out-dir" => a.out_dir = PathBuf::from(take()),
            "-h" | "--help" => usage(),
            _ => {
                eprintln!("unknown argument {k}");
                usage()
            }
        }
        i += 1;
    }
    if a.leaderboard.is_empty() || a.solver.as_os_str().is_empty() {
        usage()
    }
    a.note = a.note.replace(['\t', '\n'], " ");
    a
}

fn now_unix() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn git_commit() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "nogit".into())
}

fn sha256_file(p: &Path) -> String {
    std::fs::read(p).map(|b| to_hex(&Sha256::digest(&b))).unwrap_or_else(|_| "unreadable".into())
}

/// Beacon round + 32-byte randomness from a drand `public/latest` JSON.
fn load_beacon(path: &Option<String>) -> (u64, [u8; 32], Value) {
    let Some(p) = path else { return (0, [0u8; 32], Value::Null) };
    let parsed = std::fs::read_to_string(p).ok().and_then(|t| serde_json::from_str::<Value>(&t).ok());
    match parsed {
        Some(v) => {
            let round = v["round"].as_u64().unwrap_or(0);
            let rnd = v["randomness"].as_str().and_then(|s| from_hex(s).ok()).filter(|b| b.len() == 32);
            match rnd {
                Some(b) => (round, b.try_into().unwrap(), v),
                None => {
                    eprintln!("warning: beacon file {p} has no 32-byte randomness; using none");
                    (0, [0u8; 32], Value::Null)
                }
            }
        }
        None => {
            eprintln!("warning: could not read beacon file {p}; using none");
            (0, [0u8; 32], Value::Null)
        }
    }
}

/// seed_t = SHA-256("tpow/v1S" || beacon_randomness || le64(round) || nonce_t || le32(t))
fn derive_seed(beacon: &[u8; 32], round: u64, nonce: &[u8; 32], trial: u32) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(SEED_PREFIX);
    h.update(beacon);
    h.update(round.to_le_bytes());
    h.update(nonce);
    h.update(trial.to_le_bytes());
    h.finalize().into()
}

struct Ctx {
    args: Args,
    lb: Leaderboard,
    params: Params,
    gpu: Option<GpuInfo>,
    use_bwrap: bool,
    results_tsv: PathBuf,
}

fn fail(ctx: &Ctx, reason: &str, report: &mut Value) -> ! {
    eprintln!("\n!! tpow-bench FAILED: {reason}");
    report["status"] = json!("FAIL");
    report["fail_reason"] = json!(reason);
    write_json(&ctx.args.out_dir.join("run.json"), report);
    append_tsv(ctx, "FAIL", None, None, &format!("{} | {}", ctx.args.note, reason));
    std::process::exit(1)
}

fn write_json(p: &Path, v: &Value) {
    let tmp = p.with_extension("json.tmp");
    if std::fs::write(&tmp, serde_json::to_string_pretty(v).unwrap() + "\n").is_ok() {
        let _ = std::fs::rename(&tmp, p);
    } else {
        eprintln!("warning: could not write {}", p.display());
    }
}

fn append_tsv(ctx: &Ctx, status: &str, score: Option<f64>, median_s: Option<f64>, note: &str) {
    let new = !ctx.results_tsv.exists();
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&ctx.results_tsv) {
        if new {
            let _ = writeln!(f, "timestamp\tcommit\tleaderboard\tstatus\tscore_tops\tmedian_s\tunits\tgpu\tnote");
        }
        let _ = writeln!(
            f,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            now_unix(),
            git_commit(),
            ctx.lb.name,
            status,
            score.map(|s| format!("{s:.3}")).unwrap_or_default(),
            median_s.map(|s| format!("{s:.4}")).unwrap_or_default(),
            ctx.args.units.unwrap_or(ctx.lb.units_per_trial),
            ctx.gpu.as_ref().map(|g| g.name.clone()).unwrap_or_else(|| "none".into()),
            note.replace(['\t', '\n'], " ")
        );
    }
}

fn setup_for<'a>(ctx: &'a Ctx, label: &str) -> WorkerSetup<'a> {
    WorkerSetup {
        solver: &ctx.args.solver,
        params: ctx.params,
        gpu_uuid: ctx.gpu.as_ref().map(|g| g.uuid.as_str()),
        gpu_minor: ctx.gpu.as_ref().and_then(|g| g.device_minor),
        use_bwrap: ctx.use_bwrap,
        user_namespace: ctx.lb.sandbox_user_namespace,
        init_timeout: Duration::from_secs(ctx.lb.init_timeout_s),
        run_timeout: Duration::from_secs(ctx.lb.trial_timeout_s),
        stderr_prefix: format!("[worker:{label}]"),
    }
}

/// Compare solver digests against the reference on `indices`; returns mismatches.
fn audit(model: &Model, digests: &[u8], first_unit: u64, indices: &[u64]) -> Vec<(u64, String, String)> {
    let units: Vec<u64> = indices.iter().map(|&i| first_unit + i).collect();
    let want = model.unit_digests(&units);
    let mut bad = Vec::new();
    for ((&i, &u), w) in indices.iter().zip(&units).zip(&want) {
        let got = &digests[i as usize * DIGEST_BYTES..(i as usize + 1) * DIGEST_BYTES];
        if got != w {
            bad.push((u, to_hex(got), to_hex(w)));
        }
    }
    bad
}

fn main() {
    let args = parse_args();
    let lb = Leaderboard::load(&args.leaderboard).unwrap_or_else(|e| {
        eprintln!("tpow-bench: {e}");
        std::process::exit(2)
    });
    let params = Params::by_name(&lb.function).unwrap();
    let out_dir = args.out_dir.clone();
    let _ = std::fs::remove_file(out_dir.join("score.json"));
    let overrides = args.units.is_some() || args.warmup.is_some() || args.trials.is_some() || args.audit.is_some();
    let n_units = args.units.unwrap_or(lb.units_per_trial).max(2);
    let n_warm = args.warmup.unwrap_or(lb.warmup_trials);
    let n_meas = args.trials.unwrap_or(lb.measured_trials).max(1);
    let k_audit = args.audit.unwrap_or(lb.audit_samples_per_trial);

    let mut ctx = Ctx {
        results_tsv: out_dir.join("results.tsv"),
        args,
        lb,
        params,
        gpu: None,
        use_bwrap: false,
    };
    let mut report = json!({
        "harness": "tpow-bench",
        "harness_version": env!("CARGO_PKG_VERSION"),
        "commit": git_commit(),
        "started_unix": now_unix(),
        "leaderboard": ctx.lb,
        "function": ctx.params.name(),
        "params": ctx.params,
        "ops_per_unit": ctx.params.ops_per_unit(),
        "units_per_trial": n_units,
        "overrides": overrides,
        "solver_sha256": sha256_file(&ctx.args.solver),
        "note": ctx.args.note,
    });

    println!("=== tpow-bench: {} ({}) ===", ctx.lb.name, ctx.params.name());
    println!("  units/trial : {n_units}  ({:.3e} int8 ops)", n_units as f64 * ctx.params.ops_per_unit() as f64);
    println!("  trials      : {n_warm} warm-up + {n_meas} measured, {k_audit} audited units each");

    // ── Hardware ────────────────────────────────────────────────────────────
    if let Some(req) = ctx.lb.gpu.clone() {
        let g = match gpu::query(ctx.args.gpu_index) {
            Ok(g) => g,
            Err(e) => fail(&ctx, &format!("GPU query failed: {e}"), &mut report),
        };
        println!("  gpu         : {} [{}] driver {} cc {} power {}/{} W max SM {} MHz",
            g.name, g.uuid, g.driver_version, g.compute_cap, g.power_limit_w, g.power_default_limit_w, g.clocks_max_sm_mhz);
        report["gpu"] = json!(g);
        ctx.gpu = Some(g.clone());
        for s in &req.name_must_contain {
            if !g.name.contains(s.as_str()) {
                fail(&ctx, &format!("GPU {:?} does not match leaderboard (needs {:?})", g.name, s), &mut report);
            }
        }
        for s in &req.name_must_not_contain {
            if g.name.contains(s.as_str()) {
                fail(&ctx, &format!("GPU {:?} does not match leaderboard (must not contain {:?})", g.name, s), &mut report);
            }
        }
        if req.require_default_power_limit && (g.power_limit_w - g.power_default_limit_w).abs() > 1.0 {
            fail(&ctx, &format!("power limit {} W differs from default {} W", g.power_limit_w, g.power_default_limit_w), &mut report);
        }
        if let Err(e) = gpu::wait_quiescent(g.index, Duration::from_secs(60)) {
            fail(&ctx, &e, &mut report);
        }
    } else {
        println!("  gpu         : none required (development leaderboard)");
    }

    // ── Sandbox ─────────────────────────────────────────────────────────────
    let have_bwrap = sandbox::bwrap_path().is_some();
    ctx.use_bwrap = have_bwrap && std::env::var("TPOW_UNSAFE_NO_SANDBOX").is_err();
    if !ctx.use_bwrap {
        if ctx.lb.sandbox == "bwrap" {
            fail(&ctx, "bubblewrap is required by this leaderboard but not available (run ./setup.sh)", &mut report);
        }
        eprintln!("!! no sandbox: running the worker UNCONFINED (development only, never official)");
    }
    println!("  sandbox     : {}", if ctx.use_bwrap { "bubblewrap" } else { "NONE" });
    report["sandbox"] = json!(if ctx.use_bwrap { "bwrap" } else { "none" });

    let (round, beacon_rand, beacon_json) = load_beacon(&ctx.args.beacon);
    report["beacon"] = json!({"round": round, "randomness": to_hex(&beacon_rand), "raw": beacon_json});
    println!("  beacon      : {}", if round > 0 { format!("drand round {round}") } else { "none (harness randomness only)".into() });

    // ── Conformance: fixed public seed, every unit checked ─────────────────
    let conf_seed: [u8; 32] = core::array::from_fn(|i| i as u8);
    let n_conf = ctx.lb.conformance_units.max(1);
    println!("\n-- conformance ({n_conf} units, public KAT seed, untimed) --");
    let w = worker::spawn(&setup_for(&ctx, "conformance")).unwrap_or_else(|e| fail(&ctx, &e, &mut report));
    let out = w.run(&conf_seed, 0, n_conf).unwrap_or_else(|e| fail(&ctx, &format!("conformance: {e}"), &mut report));
    let conf_model = Model::generate(ctx.params, conf_seed);
    let all: Vec<u64> = (0..n_conf).collect();
    let bad = audit(&conf_model, &out.digests, 0, &all);
    if let Some((u, got, want)) = bad.first() {
        fail(&ctx, &format!(
            "conformance mismatch on {} of {n_conf} units; first: unit {u} got {got} expected {want} (debug with: tpow-kat digests {} {} 0 {n_conf})",
            bad.len(), ctx.params.name(), to_hex(&conf_seed)), &mut report);
    }
    println!("  all {n_conf} conformance digests match the reference");
    drop(conf_model);

    // ── Timed trials ────────────────────────────────────────────────────────
    let mut trials_json = Vec::new();
    let (mut meas_secs, mut meas_tops, mut warm_secs) = (Vec::new(), Vec::new(), Vec::new());
    let mut meas_cpu = Vec::new();
    let mut meas_tel: Vec<TelemetrySummary> = Vec::new();
    let mut audited_total = 0u64;
    let ops_trial = n_units as f64 * ctx.params.ops_per_unit() as f64;
    for t in 0..(n_warm + n_meas) {
        let phase = if t < n_warm { "warmup" } else { "measured" };
        let w = worker::spawn(&setup_for(&ctx, &format!("{phase}{t}"))).unwrap_or_else(|e| fail(&ctx, &e, &mut report));
        let mut gate = json!(null);
        if let (Some(g), Some(c)) = (&ctx.gpu, ctx.lb.thermal_gate_c) {
            let (waited, temp, ok) = gpu::thermal_gate(g.index, c, Duration::from_secs(ctx.lb.thermal_gate_max_wait_s));
            gate = json!({"gate_c": c, "waited_s": waited, "temp_c": temp, "reached": ok});
        }
        // Private randomness is drawn only now, after the worker declared READY.
        let nonce = rng::os_random::<32>();
        let seed = derive_seed(&beacon_rand, round, &nonce, t);
        let sampler = ctx.gpu.as_ref().and_then(|g| Sampler::start(g.index, 100));
        let res = w.run(&seed, 0, n_units);
        let tel = sampler.map(|s| s.stop()).unwrap_or_default();
        let out = res.unwrap_or_else(|e| fail(&ctx, &format!("trial {t} ({phase}): {e}"), &mut report));
        let tops = ops_trial / out.seconds / 1e12;
        let cores = out.cpu_seconds / out.seconds;

        let model = Model::generate(ctx.params, seed);
        let idx = rng::audit_indices(n_units, k_audit);
        let bad = audit(&model, &out.digests, 0, &idx);
        audited_total += idx.len() as u64;
        let gpu_col = if tel.samples > 0 {
            format!("sm {:>5.0} MHz {:>4.0} W {:>3.0} C", tel.sm_clock_mhz_mean, tel.power_w_mean, tel.temp_c_max)
        } else {
            "no gpu telemetry".to_string()
        };
        println!(
            "  trial {t:>2} {phase:<8} {:>9.4} s {:>9.2} TOPS  cpu {:>5.2} cores  {gpu_col}  audit {}/{} ok  (init {:.1} s)",
            out.seconds, tops, cores, idx.len() - bad.len(), idx.len(), out.init_seconds
        );
        trials_json.push(json!({
            "trial": t, "phase": phase, "seconds": out.seconds, "tops": tops,
            "cpu_seconds": out.cpu_seconds, "cpu_cores": cores, "init_seconds": out.init_seconds,
            "nonce": to_hex(&nonce), "seed": to_hex(&seed), "telemetry": tel, "thermal_gate": gate,
            "audit": {
                "units": idx.len(),
                "failures": bad.len(),
                "sample": idx.iter().take(16).map(|&i| json!({"unit": i, "digest": to_hex(&out.digests[i as usize*16..i as usize*16+16])})).collect::<Vec<_>>(),
            },
        }));
        report["trials"] = json!(trials_json);
        if let Some((u, got, want)) = bad.first() {
            fail(&ctx, &format!("audit failed in trial {t}: {} of {} sampled units wrong; first: unit {u} got {got} expected {want}", bad.len(), idx.len()), &mut report);
        }
        if let Some(req) = ctx.lb.gpu.as_ref().filter(|r| r.peak_int8_dense_tops > 0.0) {
            let max_clock = ctx.gpu.as_ref().map(|g| g.clocks_max_sm_mhz).filter(|c| c.is_finite()).unwrap_or(req.peak_reference_clock_mhz);
            let ceiling = req.peak_int8_dense_tops * (max_clock / req.peak_reference_clock_mhz).max(1.0) * 1.03;
            if tops > ceiling {
                fail(&ctx, &format!("trial {t}: {tops:.1} TOPS exceeds the physical ceiling {ceiling:.1} TOPS of this GPU; result rejected for review"), &mut report);
            }
        }
        if t < n_warm {
            warm_secs.push(out.seconds);
        } else {
            if let Some(budget) = ctx.lb.cpu_budget_cores {
                if cores > budget {
                    fail(&ctx, &format!("trial {t}: worker used {cores:.2} CPU cores on average, budget is {budget} (GPU-only leaderboard)"), &mut report);
                }
            }
            meas_secs.push(out.seconds);
            meas_tops.push(tops);
            meas_cpu.push(cores);
            meas_tel.push(tel);
        }
    }

    // ── Score ───────────────────────────────────────────────────────────────
    let med_s = median(&meas_secs);
    let score = ops_trial / med_s / 1e12;
    let tmin = meas_tops.iter().cloned().fold(f64::INFINITY, f64::min);
    let tmax = meas_tops.iter().cloned().fold(0.0, f64::max);
    let official = ctx.lb.official && !overrides && ctx.use_bwrap;
    let tel_med = |f: &dyn Fn(&TelemetrySummary) -> f64| {
        let v: Vec<f64> = meas_tel.iter().map(f).filter(|x| x.is_finite()).collect();
        if v.is_empty() { Value::Null } else { json!(median(&v)) }
    };
    let pct_peak = ctx.lb.gpu.as_ref().filter(|g| g.peak_int8_dense_tops > 0.0).map(|g| 100.0 * score / g.peak_int8_dense_tops);
    let metrics = json!({
        "unit": "verified effective INT8 TOPS (median of measured trials)",
        "function": ctx.params.name(),
        "leaderboard": ctx.lb.name,
        "official": official,
        "pct_of_peak_int8_dense": pct_peak,
        "units_per_second": n_units as f64 / med_s,
        "units_per_trial": n_units,
        "ops_per_unit": ctx.params.ops_per_unit(),
        "median_seconds": med_s,
        "trial_seconds": meas_secs,
        "warmup_trial_seconds": warm_secs,
        "tops_min": tmin,
        "tops_max": tmax,
        "tops_p10": quantile(&meas_tops, 0.1),
        "tops_p90": quantile(&meas_tops, 0.9),
        "spread_pct": 100.0 * (tmax - tmin) / score,
        "cpu_cores_max": meas_cpu.iter().cloned().fold(0.0, f64::max),
        "sm_clock_mhz_median": tel_med(&|t| t.sm_clock_mhz_mean),
        "power_w_median": tel_med(&|t| t.power_w_mean),
        "temp_c_max": tel_med(&|t| t.temp_c_max),
        "throttle_reasons_or": meas_tel.iter().map(|t| t.throttle_reasons_or.clone()).collect::<Vec<_>>(),
        "audited_units": audited_total,
        "audit_failures": 0,
        "conformance_units": n_conf,
        "sandbox": report["sandbox"],
        "gpu": ctx.gpu.as_ref().map(|g| g.name.clone()),
    });
    let score_json = json!({"score": (score * 1000.0).round() / 1000.0, "metrics": metrics});
    write_json(&out_dir.join("score.json"), &score_json);
    report["status"] = json!("OK");
    report["score"] = score_json.clone();
    report["finished_unix"] = json!(now_unix());
    write_json(&out_dir.join("run.json"), &report);
    append_tsv(&ctx, "OK", Some(score), Some(med_s), &ctx.args.note);

    println!("\n=== result ===");
    println!("  score       : {score:.3} TOPS (median of {n_meas}; min {tmin:.3}, max {tmax:.3})");
    if let Some(p) = pct_peak {
        println!("  of peak     : {p:.1}% of dense INT8 datasheet peak");
    }
    println!("  units/s     : {:.1}", n_units as f64 / med_s);
    println!("  audited     : {audited_total} units + {n_conf} conformance units, 0 mismatches");
    println!("  official    : {official}");
    println!("=== tpow-bench OK ===");
}
