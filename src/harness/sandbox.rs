//! Construction of the sandboxed worker command line.
//!
//! Linux: bubblewrap. Read-only root, fresh /dev with only the NVIDIA device
//! nodes the selected GPU needs, fresh /proc, private /tmp and /dev/shm, no
//! network, no IPC/UTS/cgroup/pid sharing, all capabilities dropped, cleared
//! environment, a single writable scratch directory, die-with-parent.

use std::path::{Path, PathBuf};
use std::process::Command;

pub struct SandboxSpec<'a> {
    pub solver: &'a Path,
    pub scratch: &'a Path,
    /// UUID of the GPU to expose (CUDA_VISIBLE_DEVICES), None for CPU-only runs.
    pub gpu_uuid: Option<&'a str>,
    /// /dev/nvidia<minor> of the selected GPU; None binds every /dev/nvidia<N>.
    pub gpu_minor: Option<u32>,
    pub user_namespace: bool,
}

pub fn bwrap_path() -> Option<PathBuf> {
    std::env::var_os("PATH").and_then(|p| {
        std::env::split_paths(&p).map(|d| d.join("bwrap")).find(|c| c.is_file())
    })
}

fn nvidia_nodes(minor: Option<u32>) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = ["/dev/nvidiactl", "/dev/nvidia-uvm", "/dev/nvidia-uvm-tools"]
        .iter()
        .map(PathBuf::from)
        .filter(|p| p.exists())
        .collect();
    match minor {
        Some(m) => {
            let p = PathBuf::from(format!("/dev/nvidia{m}"));
            if p.exists() {
                v.push(p);
            }
        }
        None => {
            if let Ok(rd) = std::fs::read_dir("/dev") {
                for e in rd.flatten() {
                    let n = e.file_name().to_string_lossy().to_string();
                    if n.len() > 6 && n.starts_with("nvidia") && n[6..].chars().all(|c| c.is_ascii_digit()) {
                        v.push(e.path());
                    }
                }
            }
        }
    }
    v
}

/// Host directories masked with an empty tmpfs inside the sandbox
/// (`TPOW_SANDBOX_HIDE`, colon separated; default `/home:/root:/var/tmp`).
fn hidden_dirs() -> Vec<String> {
    std::env::var("TPOW_SANDBOX_HIDE")
        .unwrap_or_else(|_| "/home:/root:/var/tmp".into())
        .split(':')
        .filter(|s| !s.is_empty())
        .map(String::from)
        .collect()
}

fn base_env(scratch: &Path, gpu_uuid: Option<&str>) -> Vec<(String, String)> {
    let s = scratch.display().to_string();
    let mut env = vec![
        ("PATH".into(), "/usr/local/bin:/usr/bin:/bin".into()),
        ("HOME".into(), s.clone()),
        ("TMPDIR".into(), s.clone()),
        ("CUDA_CACHE_PATH".into(), format!("{s}/.nv-cache")),
        ("LANG".into(), "C".into()),
    ];
    if let Ok(ld) = std::env::var("TPOW_WORKER_LD_LIBRARY_PATH") {
        env.push(("LD_LIBRARY_PATH".into(), ld));
    }
    match gpu_uuid {
        Some(u) => env.push(("CUDA_VISIBLE_DEVICES".into(), u.to_string())),
        None => env.push(("CUDA_VISIBLE_DEVICES".into(), "".into())),
    }
    env
}

/// Sandboxed command (bwrap).
pub fn bwrap_command(bwrap: &Path, spec: &SandboxSpec) -> Command {
    let mut c = Command::new(bwrap);
    let scratch = spec.scratch.display().to_string();
    c.args(["--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc"]);
    if spec.gpu_uuid.is_some() {
        for n in nvidia_nodes(spec.gpu_minor) {
            let s = n.display().to_string();
            c.args(["--dev-bind", &s, &s]);
        }
    }
    c.args(["--tmpfs", "/tmp", "--tmpfs", "/dev/shm", "--tmpfs", "/run"]);
    // Hide host home directories and runner workspaces from the worker; the
    // solver binary itself is copied into the scratch dir before launch.
    for hidden in hidden_dirs() {
        if Path::new(&hidden).is_dir() && !spec.scratch.starts_with(&hidden) {
            c.args(["--tmpfs", &hidden]);
        }
    }
    c.args(["--bind", &scratch, &scratch, "--chdir", &scratch]);
    c.args(["--unshare-net", "--unshare-ipc", "--unshare-uts", "--unshare-pid", "--unshare-cgroup-try"]);
    if spec.user_namespace {
        c.args(["--unshare-user", "--uid", "65534", "--gid", "65534"]);
    }
    c.args(["--cap-drop", "ALL", "--new-session", "--die-with-parent", "--clearenv"]);
    for (k, v) in base_env(spec.scratch, spec.gpu_uuid) {
        c.args(["--setenv", &k, &v]);
    }
    c.arg("--").arg(spec.solver);
    c
}

/// Unconfined command (development fallback only; never official).
pub fn plain_command(spec: &SandboxSpec) -> Command {
    let mut c = Command::new(spec.solver);
    c.current_dir(spec.scratch).env_clear();
    for (k, v) in base_env(spec.scratch, spec.gpu_uuid) {
        c.env(k, v);
    }
    c
}
