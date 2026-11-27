//! End-to-end self-tests of the trusted harness against honest and adversarial
//! workers (CPU-only; uses the tiny function instance).

use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::Command;

const BENCH: &str = env!("CARGO_BIN_EXE_tpow-bench");
const CPU_SOLVER: &str = env!("CARGO_BIN_EXE_tpow-cpu-solver");
const EVIL: &str = env!("CARGO_BIN_EXE_tpow-evil-solver");

fn tmpdir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("tpow-test-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn have_bwrap() -> bool {
    Command::new("bwrap").args(["--ro-bind", "/", "/", "true"]).output().map(|o| o.status.success()).unwrap_or(false)
}

/// Copy of the adversarial solver with its mode slot patched.
fn evil(mode: &str, dir: &Path) -> PathBuf {
    let mut bin = std::fs::read(EVIL).unwrap();
    let marker = b"EVILMODE:honest";
    let pos = bin.windows(marker.len()).position(|w| w == marker).expect("mode slot");
    let mut slot = [0u8; 48];
    let s = format!("EVILMODE:{mode}");
    slot[..s.len()].copy_from_slice(s.as_bytes());
    bin[pos..pos + 48].copy_from_slice(&slot);
    let p = dir.join(format!("evil-{mode}"));
    std::fs::write(&p, bin).unwrap();
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
    p
}

fn leaderboard(dir: &Path, patch: &[(&str, Value)]) -> PathBuf {
    let base = Path::new(env!("CARGO_MANIFEST_DIR")).join("leaderboards/dev-cpu.json");
    let mut v: Value = serde_json::from_str(&std::fs::read_to_string(base).unwrap()).unwrap();
    v["units_per_trial"] = 3000.into();
    v["warmup_trials"] = 0.into();
    v["measured_trials"] = 2.into();
    v["audit_samples_per_trial"] = 256.into();
    v["conformance_units"] = 16.into();
    for (k, val) in patch {
        v[*k] = val.clone();
    }
    let p = dir.join("lb.json");
    std::fs::write(&p, serde_json::to_string(&v).unwrap()).unwrap();
    p
}

fn bench(dir: &Path, lb: &Path, solver: &Path) -> (bool, String) {
    let out = Command::new(BENCH)
        .args(["--leaderboard", lb.to_str().unwrap(), "--solver", solver.to_str().unwrap(), "--out-dir", dir.to_str().unwrap()])
        .env("TPOW_EVIL_TARGET", "secret-harness-env")
        .output()
        .unwrap();
    let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    (out.status.success(), text)
}

#[test]
fn honest_cpu_solver_scores() {
    let d = tmpdir("honest");
    let (ok, text) = bench(&d, &leaderboard(&d, &[]), Path::new(CPU_SOLVER));
    assert!(ok, "{text}");
    let score: Value = serde_json::from_str(&std::fs::read_to_string(d.join("score.json")).unwrap()).unwrap();
    assert!(score["score"].as_f64().unwrap() > 0.0);
    assert_eq!(score["metrics"]["audit_failures"], 0);
    assert_eq!(score["metrics"]["official"], false);
    let run: Value = serde_json::from_str(&std::fs::read_to_string(d.join("run.json")).unwrap()).unwrap();
    assert_eq!(run["status"], "OK");
    assert!(std::fs::read_to_string(d.join("results.tsv")).unwrap().contains("\tOK\t"));
}

#[test]
fn faking_five_percent_is_caught() {
    let d = tmpdir("fake5");
    let (ok, text) = bench(&d, &leaderboard(&d, &[]), &evil("fake5", &d));
    assert!(!ok);
    assert!(text.contains("conformance mismatch") || text.contains("audit failed"), "{text}");
    assert!(!d.join("score.json").exists());
}

#[test]
fn truncated_output_fails() {
    let d = tmpdir("short");
    let (ok, text) = bench(&d, &leaderboard(&d, &[]), &evil("short", &d));
    assert!(!ok);
    assert!(text.contains("closed stdout"), "{text}");
}

#[test]
fn wrong_ready_line_fails() {
    let d = tmpdir("ready");
    let (ok, text) = bench(&d, &leaderboard(&d, &[]), &evil("bad-ready", &d));
    assert!(!ok);
    assert!(text.contains("expected \"READY tpow-v1\""), "{text}");
}

#[test]
fn init_hang_times_out() {
    let d = tmpdir("hanginit");
    let lb = leaderboard(&d, &[("init_timeout_s", 2.into())]);
    let (ok, text) = bench(&d, &lb, &evil("hang-init", &d));
    assert!(!ok);
    assert!(text.contains("not READY within"), "{text}");
}

#[test]
fn run_hang_times_out() {
    let d = tmpdir("hangrun");
    let lb = leaderboard(&d, &[("trial_timeout_s", 2.into())]);
    let (ok, text) = bench(&d, &lb, &evil("hang-run", &d));
    assert!(!ok);
    assert!(text.contains("exceeded timeout"), "{text}");
}

#[test]
fn cpu_budget_is_enforced() {
    let d = tmpdir("cpuhog");
    let lb = leaderboard(&d, &[("cpu_budget_cores", Value::from(0.25)), ("units_per_trial", 200.into())]);
    let (ok, text) = bench(&d, &lb, &evil("cpu-hog", &d));
    assert!(!ok);
    assert!(text.contains("CPU cores on average"), "{text}");
}

#[test]
fn sandbox_blocks_escapes() {
    if !have_bwrap() {
        eprintln!("skipping: bubblewrap not usable here");
        return;
    }
    let d = tmpdir("escape");
    let lb = leaderboard(&d, &[("sandbox", "bwrap".into())]);
    let (ok, text) = bench(&d, &lb, &evil("escape", &d));
    assert!(ok, "{text}");
    assert!(text.contains("escape attempts all blocked"), "{text}");
}

#[test]
fn exiting_right_after_output_is_rejected() {
    let d = tmpdir("exitearly");
    let (ok, text) = bench(&d, &leaderboard(&d, &[]), &evil("exit-early", &d));
    assert!(!ok);
    assert!(text.contains("must stay alive"), "{text}");
}
