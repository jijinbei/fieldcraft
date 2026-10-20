//! Profile of the explicit step on the Lamé cylinder (a = 10, b = 20, bore pressure 100 MPa,
//! steel, mapped boundary), for the parallelism notes in docs/research/reference-map.md.
//!
//! `cargo run --release -p fieldcraft-eulerian --example rmt_profile -- H [key=value ...]`
//! with keys
//! - `warm=` steps before timing (default 200: the wave has left the bore),
//! - `steps=` timed steps (default 50),
//! - `threads=` a comma-separated list of thread counts (default 1),
//! - `boundary=blurred|mapped` (default mapped),
//! - `pad=` widens the grid around the same cylinder (default 0),
//! - `remote=1` marks the nodes more than CPU_RINGS + 2 rings inside as stepped elsewhere
//!   (their values stay put): the CPU's share of a step with the deep interior on a GPU.
//!
//! Prints one JSON line per thread count: the seconds per step of every phase, the node
//! counts and a checksum of the state (equal across thread counts when the step is
//! deterministic).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use fieldcraft_eulerian::{Boundary, CPU_RINGS, Grid, Limits, Material, NodeKind, Phase, Settings, Shape, Solid};
use std::collections::BTreeMap;
use std::time::Instant;

/// The cylinder on a grid reaching `pad` beyond the usual 5 past the outer surface.
fn lame_solid(h: f64, boundary: Boundary, pad: f64) -> Solid {
    let (a, b) = (10.0, 20.0);
    let n = (2.0 * (b + 5.0 + pad) / h).round() as usize + 1;
    let o = -0.5 * (n as f64 - 1.0) * h;
    let g = Grid::new(n, n, h, [o, o], [false; 2], &Limits::default()).unwrap();
    let m = Material::neo_hookean(210_000.0, 0.3, 1.0).unwrap();
    let c = m.p_wave_speed();
    let settings = Settings { damping: 0.5 * c / b, boundary, ..Settings::default() };
    let mut s = Solid::new(g, m, Shape::Annulus { center: [0.0, 0.0], inner: a, outer: b }, settings, Limits::default()).unwrap();
    s.set_ambient_field(|x| if x[0].hypot(x[1]) < 15.0 { [-100.0, -100.0, 0.0] } else { [0.0; 3] }).unwrap();
    s
}

/// FNV-1a over the bits of ζ and v at the known nodes.
fn checksum(s: &Solid) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    let mut eat = |x: f64| {
        for b in x.to_bits().to_le_bytes() {
            hash ^= u64::from(b);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    for k in 0..s.grid().len() {
        if let (Some(xi), Some(v)) = (s.reference(k), s.velocity(k)) {
            xi.iter().chain(v.iter()).for_each(|&x| eat(x));
        }
    }
    hash
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let h: f64 = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(0.125);
    let kv: BTreeMap<String, String> = args.iter().skip(2).filter_map(|a| a.split_once('=').map(|(k, v)| (k.to_string(), v.to_string()))).collect();
    let get = |k: &str, d: &str| kv.get(k).cloned().unwrap_or_else(|| d.to_string());
    let warm: usize = get("warm", "200").parse().unwrap();
    let steps: usize = get("steps", "50").parse().unwrap();
    let boundary = if get("boundary", "mapped") == "blurred" { Boundary::Blurred } else { Boundary::Mapped };
    let pad: f64 = get("pad", "0").parse().unwrap();
    let remote = get("remote", "0") == "1";
    let threads: Vec<usize> = get("threads", "1").split(',').map(|t| t.parse().unwrap()).collect();
    for &t in &threads {
        let mut s = lame_solid(h, boundary, pad);
        let (times, wall, sum) = fieldcraft_parallel::with_threads(t, || {
            s.run(warm).unwrap();
            if remote {
                let deep = s.deep_interior(CPU_RINGS + 2);
                s.set_remote(Some(deep)).unwrap();
            }
            let mut times: BTreeMap<Phase, f64> = BTreeMap::new();
            let start = Instant::now();
            let mut last = Instant::now();
            s.run_observed(steps, &mut |p| {
                let now = Instant::now();
                *times.entry(p).or_default() += (now - last).as_secs_f64();
                last = now;
            })
            .unwrap();
            (times, start.elapsed().as_secs_f64(), checksum(&s))
        });
        let n = s.grid().len();
        let solid = (0..n).filter(|&k| matches!(s.kind(k), Some(NodeKind::Solid | NodeKind::Fixed))).count();
        let band = (0..n).filter(|&k| s.kind(k) == Some(NodeKind::Band)).count();
        let st = s.cut_stats().unwrap_or_default();
        let phases: Vec<String> =
            Phase::ALL.iter().map(|p| format!("\"{p:?}\":{:.6e}", times.get(p).copied().unwrap_or(0.0) / steps as f64)).collect();
        println!(
            "{{\"h\":{h},\"pad\":{pad},\"remote\":{},\"threads\":{t},\"nodes\":{n},\"solid\":{solid},\"band\":{band},\"cut_cells\":{},\"surface_points\":{},\"sweeps\":{},\"chains\":{},\"longest_chain\":{},\"step\":{:.6e},\"phases\":{{{}}},\"checksum\":\"{sum:016x}\"}}",
            s.remote().map_or(0, |r| r.iter().filter(|&&x| x).count()),
            st.cut_cells,
            st.surface_points,
            st.sweeps,
            st.chains,
            st.longest_chain,
            wall / steps as f64,
            phases.join(",")
        );
    }
}
