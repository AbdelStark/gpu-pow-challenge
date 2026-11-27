//! Known-answer tests and developer utilities for TPoW.
//!
//!   tpow-kat gen   <tpow-v1|tpow-tiny> <out.json>     write a KAT vector file
//!   tpow-kat check <vectors.json>                      re-derive and compare a KAT file
//!   tpow-kat digests <params> <seed-hex> <first> <n>   print unit digests (hex, one per line)
//!   tpow-kat bench <params> <n_units>                  CPU reference throughput

use serde_json::{json, Value};
use std::time::Instant;
use tpow::hexutil::{seed_from_hex, to_hex};
use tpow::spec::{stream_b, stream_w, STREAM_X};
use tpow::{prg, Model, Params};

const KAT_UNITS: [u64; 8] = [0, 1, 2, 3, 255, 65_536, (1 << 32) + 5, 1 << 40];

fn kat_seed() -> [u8; 32] {
    core::array::from_fn(|i| i as u8)
}

fn params_arg(s: &str) -> Params {
    Params::by_name(s).unwrap_or_else(|| die(&format!("unknown params {s:?} (tpow-v1 | tpow-tiny)")))
}

fn die(msg: &str) -> ! {
    eprintln!("tpow-kat: {msg}");
    std::process::exit(2)
}

fn i8hex(v: &[i8]) -> String {
    to_hex(unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len()) })
}

fn build(p: Params) -> Value {
    let seed = kat_seed();
    let model = Model::generate(p, seed);
    let layers: Vec<Value> = model
        .layers
        .iter()
        .enumerate()
        .map(|(l, layer)| {
            json!({
                "layer": l,
                "out": layer.out,
                "in": layer.inp,
                "w_prg_block0": to_hex(&prg::block(&seed, stream_w(l as u32), 0)),
                "b_prg_block0": to_hex(&prg::block(&seed, stream_b(l as u32), 0)),
                "w_first16": i8hex(&layer.w[..16]),
                "w_row0_sum": layer.wsum[0],
                "b_first4": layer.b[..4].to_vec(),
            })
        })
        .collect();
    let mut x0 = vec![0i8; (p.m * p.e) as usize];
    model.input_rows(0, &mut x0);
    // Per-layer activations of unit 0, row 0 (first 16 bytes) for debugging.
    let mut act = Vec::new();
    let mut x = x0[..p.e as usize].to_vec();
    for layer in &model.layers {
        x = tpow::kernels::layer_forward(layer, &x, 1);
        act.push(i8hex(&x[..16.min(x.len())]));
    }
    let y0 = model.unit_outputs(&[0]).remove(0);
    let digests = model.unit_digests(&KAT_UNITS);
    json!({
        "function": p.name(),
        "params": p,
        "ops_per_unit": p.ops_per_unit(),
        "seed": to_hex(&seed),
        "x_prg_block0": to_hex(&prg::block(&seed, STREAM_X, 0)),
        "layers": layers,
        "unit0_row0_x0": i8hex(&x0[..p.e as usize]),
        "unit0_row0_activations_first16": act,
        "unit0_y": i8hex(&y0),
        "digests": KAT_UNITS.iter().zip(digests.iter())
            .map(|(u, d)| json!({"unit": u, "digest": to_hex(d)})).collect::<Vec<_>>(),
    })
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("gen") if args.len() == 3 => {
            let v = build(params_arg(&args[1]));
            std::fs::write(&args[2], serde_json::to_string_pretty(&v).unwrap() + "\n")
                .unwrap_or_else(|e| die(&format!("write {}: {e}", args[2])));
            eprintln!("wrote {}", args[2]);
        }
        Some("check") if args.len() == 2 => {
            let text = std::fs::read_to_string(&args[1]).unwrap_or_else(|e| die(&e.to_string()));
            let want: Value = serde_json::from_str(&text).unwrap_or_else(|e| die(&e.to_string()));
            let p: Params = serde_json::from_value(want["params"].clone()).unwrap_or_else(|e| die(&e.to_string()));
            let got = build(p);
            if got != want {
                die(&format!("KAT MISMATCH for {}", args[1]));
            }
            println!("KAT OK: {} ({})", args[1], p.name());
        }
        Some("digests") if args.len() == 5 => {
            let p = params_arg(&args[1]);
            let seed = seed_from_hex(&args[2]).unwrap_or_else(|e| die(&e));
            let first: u64 = args[3].parse().unwrap_or_else(|_| die("bad first"));
            let n: u64 = args[4].parse().unwrap_or_else(|_| die("bad n"));
            let units: Vec<u64> = (first..first + n).collect();
            for d in Model::generate(p, seed).unit_digests(&units) {
                println!("{}", to_hex(&d));
            }
        }
        Some("bench") if args.len() == 3 => {
            let p = params_arg(&args[1]);
            let n: u64 = args[2].parse().unwrap_or_else(|_| die("bad n"));
            let t = Instant::now();
            let model = Model::generate(p, kat_seed());
            let tg = t.elapsed().as_secs_f64();
            let units: Vec<u64> = (0..n).collect();
            let t = Instant::now();
            let d = model.unit_digests(&units);
            let tc = t.elapsed().as_secs_f64();
            let ops = n as f64 * p.ops_per_unit() as f64;
            println!(
                "{}: weights {:.3}s; {} units in {:.3}s = {:.1} units/s, {:.2} TOPS effective (vnni={}, threads={}) [{}]",
                p.name(), tg, n, tc, n as f64 / tc, ops / tc / 1e12,
                tpow::kernels::vnni_available(), rayon::current_num_threads(), to_hex(&d[0])
            );
        }
        _ => die("usage: gen <params> <out.json> | check <file> | digests <params> <seed> <first> <n> | bench <params> <n>"),
    }
}
