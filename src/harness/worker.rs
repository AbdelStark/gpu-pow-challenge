//! One worker lifetime = one trial: spawn (sandboxed) -> PARAMS -> READY ->
//! [gates] -> RUN (clock starts) -> n*16 digest bytes (clock stops) -> kill.

use super::procstat::tree_cpu_seconds;
use super::sandbox::{bwrap_command, bwrap_path, plain_command, SandboxSpec};
use crate::protocol::{params_line, run_line, READY_LINE};
use crate::spec::{Params, DIGEST_BYTES};
use std::io::{BufRead, BufReader, Read, Write};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

/// After the last digest byte, how long the worker is watched for an exit of
/// its own. Outside the clock, so it does not affect the score.
const EXIT_GRACE: Duration = Duration::from_millis(500);

pub struct WorkerSetup<'a> {
    pub solver: &'a Path,
    pub params: Params,
    pub gpu_uuid: Option<&'a str>,
    pub gpu_minor: Option<u32>,
    pub use_bwrap: bool,
    pub user_namespace: bool,
    pub init_timeout: Duration,
    pub run_timeout: Duration,
    pub stderr_prefix: String,
}

pub struct TrialOutput {
    pub digests: Vec<u8>,
    pub seconds: f64,
    pub cpu_seconds: f64,
    pub init_seconds: f64,
}

enum Ev {
    Ready(Instant),
    Done(Vec<u8>, Instant),
    Fail(String),
}

/// A live, READY worker waiting for its RUN command.
pub struct ReadyWorker {
    child: Child,
    pgid: i32,
    scratch: PathBuf,
    stdin: std::process::ChildStdin,
    events: mpsc::Receiver<Ev>,
    expected_tx: mpsc::Sender<u64>,
    init_seconds: f64,
    run_timeout: Duration,
}

fn make_scratch() -> Result<PathBuf, String> {
    let base = std::env::var("TPOW_SCRATCH_BASE").map(PathBuf::from).unwrap_or_else(|_| std::env::temp_dir());
    let name = format!("tpow-worker-{}", crate::hexutil::to_hex(&super::rng::os_random::<8>()));
    let dir = base.join(name);
    std::fs::create_dir(&dir).map_err(|e| format!("create scratch {}: {e}", dir.display()))?;
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o1777)).map_err(|e| e.to_string())?;
    Ok(dir)
}

pub fn spawn(setup: &WorkerSetup) -> Result<ReadyWorker, String> {
    let scratch = make_scratch()?;
    // Stage a private read-only copy of the solver inside the scratch dir so the
    // worker never needs to see the repository / runner workspace.
    let staged = scratch.join("solver");
    std::fs::copy(setup.solver, &staged).map_err(|e| format!("stage solver {}: {e}", setup.solver.display()))?;
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&staged, std::fs::Permissions::from_mode(0o555)).map_err(|e| e.to_string())?;
    }
    let spec = SandboxSpec {
        solver: &staged,
        scratch: &scratch,
        gpu_uuid: setup.gpu_uuid,
        gpu_minor: setup.gpu_minor,
        user_namespace: setup.user_namespace,
    };
    let mut cmd = if setup.use_bwrap {
        bwrap_command(&bwrap_path().ok_or("bwrap not found")?, &spec)
    } else {
        plain_command(&spec)
    };
    cmd.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());
    unsafe {
        cmd.pre_exec(|| {
            // Own process group so the whole tree can be killed at once, and die with us.
            libc::setpgid(0, 0);
            libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL);
            Ok(())
        });
    }
    let t_spawn = Instant::now();
    let mut child = cmd.spawn().map_err(|e| format!("spawn worker: {e}"))?;
    let pgid = child.id() as i32;
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = child.stdout.take().unwrap();
    let stderr = child.stderr.take().unwrap();

    // Forward worker stderr (bounded) for debuggability.
    let prefix = setup.stderr_prefix.clone();
    std::thread::spawn(move || {
        let mut n = 0usize;
        for line in BufReader::new(stderr).lines().map_while(Result::ok) {
            n += 1;
            if n <= 200 {
                eprintln!("{prefix} {}", line.chars().take(400).collect::<String>());
            } else if n == 201 {
                eprintln!("{prefix} (further worker stderr suppressed)");
            }
        }
    });

    let (tx, rx) = mpsc::channel::<Ev>();
    let (exp_tx, exp_rx) = mpsc::channel::<u64>();
    std::thread::spawn(move || {
        // 1. READY line (bounded).
        let mut line = Vec::new();
        let mut byte = [0u8; 1];
        loop {
            match stdout.read(&mut byte) {
                Ok(1) if byte[0] == b'\n' => break,
                Ok(1) => {
                    line.push(byte[0]);
                    if line.len() > 256 {
                        let _ = tx.send(Ev::Fail("worker sent an over-long first line".into()));
                        return;
                    }
                }
                _ => {
                    let _ = tx.send(Ev::Fail("worker exited before READY".into()));
                    return;
                }
            }
        }
        let got = String::from_utf8_lossy(&line).trim().to_string();
        if got != READY_LINE {
            let _ = tx.send(Ev::Fail(format!("expected {READY_LINE:?}, got {got:?}")));
            return;
        }
        let _ = tx.send(Ev::Ready(Instant::now()));
        // 2. Exactly n*16 digest bytes; the clock stops when the last byte lands.
        let n_bytes = match exp_rx.recv() {
            Ok(n) => n as usize,
            Err(_) => return,
        };
        let mut buf = vec![0u8; n_bytes];
        let mut filled = 0;
        while filled < n_bytes {
            match stdout.read(&mut buf[filled..]) {
                Ok(0) => {
                    let _ = tx.send(Ev::Fail(format!("worker closed stdout after {filled}/{n_bytes} digest bytes")));
                    return;
                }
                Ok(k) => filled += k,
                Err(e) => {
                    let _ = tx.send(Ev::Fail(format!("read error: {e}")));
                    return;
                }
            }
        }
        let t_done = Instant::now();
        let _ = tx.send(Ev::Done(buf, t_done));
    });

    let mut w = ReadyWorker {
        child,
        pgid,
        scratch,
        stdin: {
            stdin.write_all(params_line(&setup.params).as_bytes()).map_err(|e| format!("write PARAMS: {e}"))?;
            stdin.flush().map_err(|e| e.to_string())?;
            stdin
        },
        events: rx,
        expected_tx: exp_tx,
        init_seconds: 0.0,
        run_timeout: setup.run_timeout,
    };
    match w.events.recv_timeout(setup.init_timeout) {
        Ok(Ev::Ready(t)) => {
            w.init_seconds = (t - t_spawn).as_secs_f64();
            Ok(w)
        }
        Ok(Ev::Fail(e)) => {
            w.kill();
            Err(format!("worker init failed: {e}"))
        }
        Ok(Ev::Done(..)) => unreachable!(),
        Err(_) => {
            w.kill();
            Err(format!("worker not READY within {:?}", setup.init_timeout))
        }
    }
}

impl ReadyWorker {
    pub fn pid(&self) -> i32 {
        self.child.id() as i32
    }

    /// Send RUN and time until the last digest byte has been received.
    pub fn run(mut self, seed: &[u8; 32], first_unit: u64, n_units: u64) -> Result<TrialOutput, String> {
        let n_bytes = n_units * DIGEST_BYTES as u64;
        self.expected_tx.send(n_bytes).map_err(|_| "reader thread gone".to_string())?;
        let line = run_line(seed, first_unit, n_units);
        let cpu0 = tree_cpu_seconds(self.pid());
        let t0 = Instant::now();
        let wr = self.stdin.write_all(line.as_bytes()).and_then(|_| self.stdin.flush());
        if let Err(e) = wr {
            self.kill();
            return Err(format!("write RUN: {e}"));
        }
        let res = self.events.recv_timeout(self.run_timeout);
        // CPU time is sampled before the kill, while the tree is still alive.
        let cpu1 = tree_cpu_seconds(self.pid());
        // The protocol requires the worker to stay alive until the harness kills
        // it: a worker that exits on its own right after the last byte could hide
        // its CPU time from the /proc accounting above. A single check here would
        // race with an exit already in progress, so watch for a grace period.
        let exited_early = matches!(res, Ok(Ev::Done(..))) && self.exits_within(EXIT_GRACE);
        let init_seconds = self.init_seconds;
        let out = match res {
            Ok(Ev::Done(..)) if exited_early => {
                Err("worker exited on its own after RUN; it must stay alive until the harness stops it".into())
            }
            Ok(Ev::Done(buf, t1)) => Ok(TrialOutput {
                digests: buf,
                seconds: (t1 - t0).as_secs_f64(),
                cpu_seconds: (cpu1 - cpu0).max(0.0),
                init_seconds,
            }),
            Ok(Ev::Fail(e)) => Err(format!("worker failed during RUN: {e}")),
            Ok(Ev::Ready(_)) => Err("protocol error".into()),
            Err(_) => Err(format!("trial exceeded timeout {:?}", self.run_timeout)),
        };
        self.kill();
        out
    }

    fn exits_within(&mut self, grace: Duration) -> bool {
        let t = Instant::now();
        loop {
            if matches!(self.child.try_wait(), Ok(Some(_))) {
                return true;
            }
            if t.elapsed() >= grace {
                return false;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    /// Kill the whole process group, reap, and wipe the scratch directory.
    pub fn kill(&mut self) {
        unsafe {
            libc::killpg(self.pgid, libc::SIGKILL);
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = std::fs::remove_dir_all(&self.scratch);
    }
}

impl Drop for ReadyWorker {
    fn drop(&mut self) {
        if self.scratch.exists() {
            self.kill();
        }
    }
}
