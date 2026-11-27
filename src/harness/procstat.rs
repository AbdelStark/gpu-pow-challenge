//! CPU-time accounting of a sandboxed process tree via /proc.
//!
//! We sum utime+stime+cutime+cstime over the root and all live descendants.
//! A child's own time is counted while it is alive; once reaped it moves into
//! its parent's cutime/cstime, so nothing is double counted.

use std::collections::HashMap;

pub fn clk_tck() -> f64 {
    let v = unsafe { libc::sysconf(libc::_SC_CLK_TCK) };
    if v > 0 { v as f64 } else { 100.0 }
}

fn read_stat(pid: i32) -> Option<(i32, u64)> {
    let s = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    let rest = &s[s.rfind(')')? + 2..];
    let f: Vec<&str> = rest.split_whitespace().collect();
    // rest[0]=state(3) rest[1]=ppid(4) ... utime(14)=rest[11] stime=rest[12] cutime=rest[13] cstime=rest[14]
    let ppid: i32 = f.get(1)?.parse().ok()?;
    let t: u64 = (11..=14).map(|i| f.get(i).and_then(|x| x.parse::<u64>().ok()).unwrap_or(0)).sum();
    Some((ppid, t))
}

/// Total CPU seconds consumed so far by `root` and its live descendants.
pub fn tree_cpu_seconds(root: i32) -> f64 {
    let mut parent: HashMap<i32, i32> = HashMap::new();
    let mut ticks: HashMap<i32, u64> = HashMap::new();
    if let Ok(rd) = std::fs::read_dir("/proc") {
        for ent in rd.flatten() {
            if let Some(pid) = ent.file_name().to_str().and_then(|s| s.parse::<i32>().ok()) {
                if let Some((pp, t)) = read_stat(pid) {
                    parent.insert(pid, pp);
                    ticks.insert(pid, t);
                }
            }
        }
    }
    let mut total = 0u64;
    for (&pid, &t) in &ticks {
        let mut cur = pid;
        let mut hops = 0;
        loop {
            if cur == root {
                total += t;
                break;
            }
            match parent.get(&cur) {
                Some(&p) if p > 0 && hops < 64 => {
                    cur = p;
                    hops += 1;
                }
                _ => break,
            }
        }
    }
    total as f64 / clk_tck()
}
