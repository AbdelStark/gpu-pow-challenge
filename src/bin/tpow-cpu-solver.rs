//! CPU reference solver speaking the TPoW worker protocol (src/protocol.rs).
//!
//! Used for: harness end-to-end tests without a GPU, developer sanity checks,
//! and as the CPU baseline of the red-team "how much does a CPU help" variant.
//! It is deliberately straightforward; it is NOT the ranked GPU solver.

use std::io::{BufRead, BufWriter, Write};
use tpow::protocol::{parse_params_line, parse_run_line, READY_LINE};
use tpow::Model;

fn main() {
    let stdin = std::io::stdin();
    let mut lines = stdin.lock().lines();
    let first = match lines.next() {
        Some(Ok(l)) => l,
        _ => std::process::exit(1),
    };
    let params = parse_params_line(&first).unwrap_or_else(|e| {
        eprintln!("cpu-solver: {e}");
        std::process::exit(1)
    });
    // Warm the thread pool and kernels before declaring readiness (untimed).
    let _ = Model::generate(tpow::Params::TINY, [0u8; 32]).unit_digests(&[0]);

    let stdout = std::io::stdout();
    let mut out = BufWriter::with_capacity(1 << 20, stdout.lock());
    writeln!(out, "{READY_LINE}").unwrap();
    out.flush().unwrap();

    for line in lines {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        if line.trim().is_empty() {
            continue;
        }
        let (seed, first, n) = parse_run_line(&line).unwrap_or_else(|e| {
            eprintln!("cpu-solver: {e}");
            std::process::exit(1)
        });
        let model = Model::generate(params, seed);
        // Stream digests in batches so the harness sees progress.
        const BATCH: u64 = 256;
        let mut u = first;
        while u < first + n {
            let k = BATCH.min(first + n - u);
            let units: Vec<u64> = (u..u + k).collect();
            for d in model.unit_digests(&units) {
                out.write_all(&d).unwrap();
            }
            u += k;
        }
        out.flush().unwrap();
    }
}
