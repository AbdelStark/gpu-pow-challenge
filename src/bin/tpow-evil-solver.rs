//! ADVERSARIAL worker for harness self-tests (tests/harness.rs). Never ranked.
//!
//! Mode selection: the binary embeds a 48-byte slot `EVILMODE:<mode>` that
//! tests patch in a copy of the executable (the harness stages only the binary,
//! and the sandbox clears the environment, so the binary is the only channel).

use std::io::{BufRead, Write};
use tpow::protocol::{parse_params_line, parse_run_line, READY_LINE};
use tpow::Model;

#[used]
#[no_mangle]
pub static EVIL_MODE_SLOT: [u8; 48] = *b"EVILMODE:honest\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0";

fn mode() -> String {
    // Read through a volatile pointer so the patched bytes are honoured.
    let p = EVIL_MODE_SLOT.as_ptr();
    let bytes: Vec<u8> = (0..48).map(|i| unsafe { std::ptr::read_volatile(p.add(i)) }).collect();
    let s = String::from_utf8_lossy(&bytes[9..]).to_string();
    s.trim_end_matches('\0').to_string()
}

fn try_escape() -> Vec<String> {
    let mut escaped = Vec::new();
    if std::net::TcpStream::connect_timeout(&"1.1.1.1:80".parse().unwrap(), std::time::Duration::from_secs(2)).is_ok() {
        escaped.push("network".to_string());
    }
    for p in ["/etc/tpow-escape-test", "/usr/tpow-escape-test"] {
        if std::fs::write(p, b"x").is_ok() {
            escaped.push(format!("write {p}"));
        }
    }
    if let Ok(target) = std::env::var("TPOW_EVIL_TARGET") {
        escaped.push(format!("env leaked: {target}"));
    }
    if std::path::Path::new("/home").read_dir().map(|mut d| d.next().is_some()).unwrap_or(false) {
        escaped.push("can list /home".into());
    }
    escaped
}

fn main() {
    let mode = mode();
    let stdin = std::io::stdin();
    let mut lines = stdin.lock().lines();
    let params = parse_params_line(&lines.next().unwrap().unwrap()).unwrap();
    let mut out = std::io::stdout().lock();
    match mode.as_str() {
        "bad-ready" => {
            writeln!(out, "READY something-else").unwrap();
            return;
        }
        "hang-init" => loop {
            std::thread::sleep(std::time::Duration::from_secs(1));
        },
        "escape" => {
            let e = try_escape();
            if !e.is_empty() {
                eprintln!("ESCAPED: {e:?}");
                std::process::exit(3);
            }
            eprintln!("escape attempts all blocked");
        }
        _ => {}
    }
    writeln!(out, "{READY_LINE}").unwrap();
    out.flush().unwrap();
    let (seed, first, n) = parse_run_line(&lines.next().unwrap().unwrap()).unwrap();
    let model = Model::generate(params, seed);
    let units: Vec<u64> = (first..first + n).collect();
    let mut digests = model.unit_digests(&units);
    match mode.as_str() {
        // Fake 5% of the units (pretend to have skipped them).
        "fake5" => {
            for (i, d) in digests.iter_mut().enumerate() {
                if (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) % 100 < 5 {
                    d[0] ^= 0xff;
                }
            }
        }
        "short" => {
            digests.truncate(digests.len() / 2);
        }
        "hang-run" => loop {
            std::thread::sleep(std::time::Duration::from_secs(1));
        },
        "cpu-hog" => {
            // Burn extra CPU inside the timed window.
            let t = std::time::Instant::now();
            std::thread::scope(|s| {
                for _ in 0..3 {
                    s.spawn(|| while t.elapsed().as_millis() < 400 { std::hint::black_box(0u64); });
                }
            });
        }
        _ => {}
    }
    for d in &digests {
        out.write_all(d).unwrap();
    }
    out.flush().unwrap();
    if mode == "exit-early" || mode == "short" {
        // Try to hide CPU time from the harness's /proc accounting.
        std::process::exit(0);
    }
    // Protocol: stay alive until the harness stops us.
    for _ in lines {}
}
