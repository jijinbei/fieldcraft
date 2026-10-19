//! A plane P-wave pulse crossing a refined strip, for the research notes (§12): the wave
//! reflected where it enters the finer grid and where it leaves it, and the long-run
//! stability of the subcycled coarse–fine coupling.
//!
//! `cargo run --release -p fieldcraft-eulerian --example refine_wave -- H0 LEVELS WIDTH [key=value ...]`
//! with WIDTH the pulse's half-width in coarse cells. Keys: `visc=` κ (default 0), `periods=`
//! number of trips around the periodic box for the stability run (default 0), `dump=PREFIX`
//! (profiles u_x(x) along the middle row at the measuring times).
//!
//! The box is periodic in both directions, 256 coarse cells along x and 8 across; the strip
//! x ∈ [96, 160] coarse cells is refined LEVELS times. The pulse u_x = A exp(−((x − x₀)/w)²)
//! starts 3w + 8 cells left of the strip, moving +x. The reflection is the largest |u_x|
//! difference to the uniform coarse run (same pulse, same steps) on the nodes left of the strip
//! once the pulse's tail has entered it (entry: centre 3w + 4 cells inside), and inside the
//! strip once the tail has left it (exit), over A.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use fieldcraft_eulerian::{Boundary, Grid, Hierarchy, Limits, Material, Refinement, Settings, Shape};
use std::collections::HashMap;
use std::fmt::Write as _;
use std::sync::Arc;

const NX: usize = 256;
const XA: f64 = 96.0;
const XB: f64 = 160.0;
const NY: usize = 8;
const AMP: f64 = 1e-4;

fn build(h0: f64, levels: usize, width: f64, visc: f64, subcycle: bool, dt_levels: usize, cubic: bool) -> Hierarchy {
    let g = Grid::new(NX, NY, h0, [0.0, 0.0], [true, true], &Limits::default()).unwrap();
    let m = Material::linear_elastic(100.0, 0.3, 1.0).unwrap();
    let c = m.p_wave_speed();
    let (xa, xb) = (XA * h0, XB * h0);
    let region: Arc<dyn Fn([f64; 2]) -> bool + Send + Sync> = Arc::new(move |x: [f64; 2]| x[0] > xa && x[0] < xb);
    let r = Refinement { levels, region: Some(region), subcycle, cubic_hanging: cubic, ..Refinement::default() };
    let set = Settings { viscosity: visc, boundary: Boundary::Mapped, ..Settings::default() };
    // The uniform reference without subcycling takes the refined run's time step.
    let set = Settings { cfl: set.cfl / (1u64 << dt_levels) as f64, ..set };
    let mut s = Hierarchy::new(g, m, Shape::Everywhere, set, r, Limits::default()).unwrap();
    let (x0, w) = ((XA - 3.0 * width - 8.0) * h0, width * h0);
    let period = NX as f64 * h0;
    s.set_motion(move |x| {
        // The periodic image nearest x.
        let d = (x[0] - x0 + 0.5 * period).rem_euclid(period) - 0.5 * period;
        let e = AMP * (-(d / w).powi(2)).exp();
        let du = -2.0 * d / (w * w) * e;
        ([x[0] - e, x[1]], [-c * du, 0.0])
    })
    .unwrap();
    s
}

/// u_x along the middle row: position → value, from the degrees of freedom of every level.
fn profile(s: &Hierarchy) -> Vec<(f64, f64)> {
    let ymid = 0.5 * NY as f64 * s.level(0).unwrap().grid().h();
    let mut by: HashMap<i64, f64> = HashMap::new();
    let ycut = 1e-9;
    for n in s.dofs() {
        if (n.position[1] - ymid).abs() < ycut {
            let u = s.level(n.level).unwrap().displacement(n.index).unwrap()[0];
            let key = (n.position[0] * 1e6).round() as i64;
            by.insert(key, u);
        }
    }
    let mut v: Vec<(f64, f64)> = by.into_iter().map(|(k, u)| (k as f64 * 1e-6, u)).collect();
    v.sort_by(|a, b| a.0.total_cmp(&b.0));
    v
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let h0: f64 = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(1.0);
    let levels: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(1);
    let width: f64 = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(4.0);
    let kv: HashMap<String, String> = args.iter().skip(4).filter_map(|a| a.split_once('=').map(|(k, v)| (k.to_string(), v.to_string()))).collect();
    let get = |k: &str, d: &str| kv.get(k).cloned().unwrap_or_else(|| d.to_string());
    let visc: f64 = get("visc", "0").parse().unwrap();
    let periods: usize = get("periods", "0").parse().unwrap();
    let subcycle = get("subcycle", "1") != "0";
    let cubic = get("hanging", "cubic") == "cubic";
    let mut fine = build(h0, levels, width, visc, subcycle, 0, cubic);
    let mut coarse = build(h0, 0, width, visc, true, if subcycle { 0 } else { levels }, cubic);
    let c = Material::linear_elastic(100.0, 0.3, 1.0).unwrap().p_wave_speed();
    let dt = fine.dt();
    assert!((dt - coarse.dt()).abs() < 1e-15);
    let (xa, xb) = (XA * h0, XB * h0);
    let x0 = (XA - 3.0 * width - 8.0) * h0;
    let steps_to = |x: f64| ((x - x0) / c / dt).round() as usize;
    let mut dump = String::from("t,x,u_refined,u_uniform\n");
    let measure = |s1: &mut Hierarchy, s0: &mut Hierarchy, until: usize, region: (f64, f64), tag: &str, dump: &mut String| -> f64 {
        let n = until.saturating_sub(s1.stats().steps);
        s1.run(n).unwrap();
        s0.run(n).unwrap();
        let (p1, p0) = (profile(s1), profile(s0));
        let m0: HashMap<i64, f64> = p0.iter().map(|&(x, u)| ((x * 1e6).round() as i64, u)).collect();
        let mut worst: f64 = 0.0;
        for &(x, u) in &p1 {
            if let Some(&u0) = m0.get(&((x * 1e6).round() as i64)) {
                writeln!(dump, "{tag},{x},{u},{u0}").unwrap();
                if x > region.0 && x < region.1 {
                    worst = worst.max((u - u0).abs());
                }
            }
        }
        worst / AMP
    };
    // Entry: the pulse's tail just inside the strip; the reflection is left of it.
    let tail = (3.0 * width + 4.0) * h0;
    let entry = measure(&mut fine, &mut coarse, steps_to(xa + tail), (xa - 2.0 * tail, xa - 2.0 * h0), "entry", &mut dump);
    // Exit: the tail just past the strip; the reflection is inside it.
    let exit = measure(&mut fine, &mut coarse, steps_to(xb + tail), (xa + 2.0 * h0, xb - 2.0 * h0), "exit", &mut dump);
    let mut out = format!(
        "{{\"h0\":{h0},\"levels\":{levels},\"width\":{width},\"visc\":{visc},\"entry_reflection\":{entry:.4e},\"exit_reflection\":{exit:.4e},\"nodes\":{:?}",
        fine.stats().nodes
    );
    // Stability: many trips around the box; the largest |u| and |v| at the end of each trip.
    if periods > 0 {
        let per = (NX as f64 * h0 / c / dt).round() as usize;
        let mut amps = Vec::new();
        for _ in 0..periods {
            fine.run(per).unwrap();
            let (mut um, mut vm): (f64, f64) = (0.0, 0.0);
            for n in fine.dofs() {
                let sol = fine.level(n.level).unwrap();
                um = um.max(sol.displacement(n.index).unwrap()[0].abs());
                vm = vm.max(sol.velocity(n.index).unwrap()[0].abs());
            }
            amps.push((um / AMP, vm));
        }
        let um: Vec<String> = amps.iter().map(|a| format!("{:.5}", a.0)).collect();
        write!(out, ",\"trip_max_u\":[{}]", um.join(",")).unwrap();
    }
    out.push('}');
    println!("{out}");
    if let Some(p) = kv.get("dump") {
        std::fs::write(format!("{p}_profiles.csv"), dump).unwrap();
    }
}
