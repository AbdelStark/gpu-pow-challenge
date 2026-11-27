//! Worker protocol shared by the harness and the CPU reference solver.
//! See docs/HARNESS.md for the full description.
//!
//! harness -> worker (stdin, text lines):
//!   PARAMS d=<d> L=<hidden_layers> e=<e> h=<h> m=<m>
//!   RUN <seed-hex-64> <first_unit> <n_units>
//! worker -> harness (stdout):
//!   "READY tpow-v1\n" (after PARAMS, once fully initialised)
//!   then, after RUN, exactly n_units * 16 raw digest bytes, in unit order.

use crate::spec::Params;

pub const READY_LINE: &str = "READY tpow-v1";

pub fn params_line(p: &Params) -> String {
    format!("PARAMS d={} L={} e={} h={} m={}\n", p.d, p.hidden_layers, p.e, p.h, p.m)
}

pub fn parse_params_line(line: &str) -> Result<Params, String> {
    let mut it = line.split_whitespace();
    if it.next() != Some("PARAMS") {
        return Err(format!("expected PARAMS line, got {line:?}"));
    }
    let (mut d, mut l, mut e, mut h, mut m) = (None, None, None, None, None);
    for kv in it {
        let (k, v) = kv.split_once('=').ok_or_else(|| format!("bad field {kv:?}"))?;
        let v: u32 = v.parse().map_err(|_| format!("bad value in {kv:?}"))?;
        match k {
            "d" => d = Some(v),
            "L" => l = Some(v),
            "e" => e = Some(v),
            "h" => h = Some(v),
            "m" => m = Some(v),
            _ => return Err(format!("unknown field {k:?}")),
        }
    }
    let p = Params {
        d: d.ok_or("missing d")?,
        hidden_layers: l.ok_or("missing L")?,
        e: e.ok_or("missing e")?,
        h: h.ok_or("missing h")?,
        m: m.ok_or("missing m")?,
    };
    p.validate()?;
    Ok(p)
}

pub fn run_line(seed: &[u8; 32], first_unit: u64, n_units: u64) -> String {
    format!("RUN {} {} {}\n", crate::hexutil::to_hex(seed), first_unit, n_units)
}

pub fn parse_run_line(line: &str) -> Result<([u8; 32], u64, u64), String> {
    let f: Vec<&str> = line.split_whitespace().collect();
    if f.len() != 4 || f[0] != "RUN" {
        return Err(format!("expected RUN line, got {line:?}"));
    }
    let seed = crate::hexutil::seed_from_hex(f[1])?;
    let first = f[2].parse().map_err(|_| "bad first_unit")?;
    let n = f[3].parse().map_err(|_| "bad n_units")?;
    Ok((seed, first, n))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn roundtrip() {
        let p = Params::V1;
        assert_eq!(parse_params_line(&params_line(&p)).unwrap(), p);
        let s = [0xabu8; 32];
        assert_eq!(parse_run_line(&run_line(&s, 5, 9)).unwrap(), (s, 5, 9));
    }
}
