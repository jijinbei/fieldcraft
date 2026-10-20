//! Measurements of the GPU interior for docs/research/reference-map.md.
//!
//! `cargo run --release -p fieldcraft-eulerian-gpu --example rmt_gpu -- CASE [key=value ...]`
//! - `wave n=N steps=S`: a shear wave on a periodic N × N grid with no surface (every node on
//!   the device): seconds per step on the device and on the CPU (`threads=`), and the
//!   difference of the fields after S steps.
//! - `lame h=H steps=S`: the Lamé cylinder with the deep interior on the device and the surface
//!   on the CPU: seconds per step and where they went, against the CPU alone (`pad=` widens
//!   the grid around the same cylinder).
//! - `relax h=H steps=S load=L`: the Lamé cylinder (bore pressure 100 L) relaxed for S steps on
//!   the CPU, on the device with the stable stress and with the stress as written: the stress
//!   differences against the CPU and the errors against the closed form.
//!
//! Keys: `stable=0|1` (the stress from small quantities, default 1), `threads=` for the CPU.
//! Prints one JSON line.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use fieldcraft_eulerian::{Boundary, Grid, Limits, Material, NodeKind, Settings, Shape, Solid};
use fieldcraft_eulerian_gpu::{GpuSolid, Options};
use std::collections::BTreeMap;
use std::time::Instant;

fn wave(n: usize) -> Solid {
    let h = 1.0 / n as f64;
    let g = Grid::new(n, n, h, [0.0, 0.0], [true, true], &Limits::default()).unwrap();
    let m = Material::neo_hookean(210_000.0, 0.3, 1.0).unwrap();
    let mut s = Solid::new(g, m, Shape::Everywhere, Settings { boundary: Boundary::Mapped, ..Settings::default() }, Limits::default()).unwrap();
    // A shear wave along y with a small longitudinal part, amplitude 1e-3 of the speed.
    let c = m.s_wave_speed();
    let tau = std::f64::consts::TAU;
    s.set_motion(|x| (x, [1e-3 * c * (tau * x[1]).sin(), 2e-4 * c * (tau * x[0]).cos()])).unwrap();
    s
}

fn lame(h: f64, pad: f64) -> Solid {
    lame_on(h, 1.0, pad)
}

fn lame_loaded(h: f64, load: f64) -> Solid {
    lame_on(h, load, 0.0)
}

/// The cylinder on a grid reaching `pad` beyond the usual 5 past the outer surface.
fn lame_on(h: f64, load: f64, pad: f64) -> Solid {
    let (a, b) = (10.0, 20.0);
    let n = (2.0 * (b + 5.0 + pad) / h).round() as usize + 1;
    let o = -0.5 * (n as f64 - 1.0) * h;
    let g = Grid::new(n, n, h, [o, o], [false; 2], &Limits::default()).unwrap();
    let m = Material::neo_hookean(210_000.0, 0.3, 1.0).unwrap();
    let c = m.p_wave_speed();
    let settings = Settings { damping: 0.5 * c / b, boundary: Boundary::Mapped, ..Settings::default() };
    let mut s = Solid::new(g, m, Shape::Annulus { center: [0.0, 0.0], inner: a, outer: b }, settings, Limits::default()).unwrap();
    s.set_ambient_field(move |x| if x[0].hypot(x[1]) < 15.0 { [-100.0 * load, -100.0 * load, 0.0] } else { [0.0; 3] }).unwrap();
    s
}

/// Lamé's closed form (a = 10, b = 20, bore pressure p, plane strain): u_r, σ_r, σ_θ.
fn lame_exact(r: f64, p: f64) -> (f64, f64, f64) {
    let (a, b, e, nu) = (10.0f64, 20.0f64, 210_000.0, 0.3);
    let k = p * a * a / (b * b - a * a);
    ((1.0 + nu) * k / e * ((1.0 - 2.0 * nu) * r + b * b / r), k * (1.0 - b * b / (r * r)), k * (1.0 + b * b / (r * r)))
}

/// Polar components (σ_r, σ_θ) of a stress at x.
fn polar(st: [f64; 4], x: [f64; 2]) -> (f64, f64) {
    let r = x[0].hypot(x[1]);
    let (c, s) = (x[0] / r, x[1] / r);
    (st[0] * c * c + st[1] * s * s + 2.0 * st[2] * s * c, st[0] * s * s + st[1] * c * c - 2.0 * st[2] * s * c)
}

/// Against the closed form at pressure p: the largest error of σ_r, σ_θ over the solid nodes
/// more than 2h inside, over p; the mean σ_θ on r = 15 (relative error); the largest speed.
fn lame_errors(s: &Solid, p: f64) -> (f64, f64, f64) {
    let h = s.grid().h();
    let (mut worst, mut hoop, mut count, mut speed) = (0.0f64, 0.0, 0.0, 0.0f64);
    for k in 0..s.grid().len() {
        if s.kind(k) != Some(NodeKind::Solid) {
            continue;
        }
        let x = s.grid().position(k);
        let r = x[0].hypot(x[1]);
        let v = s.velocity(k).unwrap();
        speed = speed.max(v[0].hypot(v[1]));
        let (sr, st) = polar(s.stress(k).unwrap(), x);
        let (_, er, et) = lame_exact(r, p);
        if s.phi(k).unwrap() < -2.0 * h {
            worst = worst.max((sr - er).abs()).max((st - et).abs());
        }
        if (r - 15.0).abs() < 0.5 * h {
            hoop += st;
            count += 1.0;
        }
    }
    (worst / p, hoop / count / lame_exact(15.0, p).2 - 1.0, speed)
}

/// The largest stress difference between two solids over their solid nodes more than 2h
/// inside, and over all their solid nodes.
fn stress_difference(a: &Solid, b: &Solid) -> (f64, f64) {
    let h = a.grid().h();
    let (mut inner, mut all) = (0.0f64, 0.0f64);
    for k in 0..a.grid().len() {
        if a.kind(k) != Some(NodeKind::Solid) || b.kind(k) != Some(NodeKind::Solid) {
            continue;
        }
        let (sa, sb) = (a.stress(k).unwrap(), b.stress(k).unwrap());
        let d = (0..3).map(|c| (sa[c] - sb[c]).abs()).fold(0.0, f64::max);
        all = all.max(d);
        if a.phi(k).unwrap() < -2.0 * h {
            inner = inner.max(d);
        }
    }
    (inner, all)
}

fn relax(kv: &BTreeMap<String, String>) {
    let get = |k: &str, d: &str| kv.get(k).cloned().unwrap_or_else(|| d.to_string());
    let h: f64 = get("h", "0.25").parse().unwrap();
    let load: f64 = get("load", "1").parse().unwrap();
    let steps: usize = get("steps", "8000").parse().unwrap();
    let threads: usize = get("threads", "14").parse().unwrap();
    let p = 100.0 * load;
    let mut cpu = lame_loaded(h, load);
    let t = Instant::now();
    fieldcraft_parallel::with_threads(threads, || cpu.run(steps)).unwrap();
    let cpu_time = t.elapsed().as_secs_f64();
    let (cw, ch, cs) = lame_errors(&cpu, p);
    let mut out = format!(
        "{{\"case\":\"relax\",\"h\":{h},\"load\":{load},\"steps\":{steps},\"cpu\":{{\"time\":{cpu_time:.2},\"worst\":{cw:.4e},\"hoop15\":{ch:.4e},\"speed\":{cs:.3e}}}"
    );
    for stable in [true, false] {
        let mut g = GpuSolid::from_env(lame_loaded(h, load), Options { stable }).unwrap();
        let t = Instant::now();
        fieldcraft_parallel::with_threads(threads, || g.run(steps)).unwrap();
        let time = t.elapsed().as_secs_f64();
        let s = g.into_solid().unwrap();
        let (w, hp, sp) = lame_errors(&s, p);
        let (di, da) = stress_difference(&cpu, &s);
        out += &format!(
            ",\"{}\":{{\"time\":{time:.2},\"worst\":{w:.4e},\"hoop15\":{hp:.4e},\"speed\":{sp:.3e},\"dstress_inner\":{:.4e},\"dstress_all\":{:.4e}}}",
            if stable { "gpu_stable" } else { "gpu_naive" },
            di / p,
            da / p
        );
    }
    println!("{out}}}");
}

/// The largest difference of v over the largest |v|, of ζ over the largest |ζ|, over the solid
/// nodes, and of the stress over `scale`.
fn compare(a: &Solid, b: &Solid, scale: f64) -> (f64, f64, f64) {
    let (mut dv, mut vm, mut dz, mut zm, mut ds) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for k in 0..a.grid().len() {
        if !matches!(a.kind(k), Some(NodeKind::Solid)) {
            continue;
        }
        let ((za, va), (zb, vb)) = (a.node_state(k).unwrap(), b.node_state(k).unwrap());
        dv = dv.max((va[0] - vb[0]).hypot(va[1] - vb[1]));
        vm = vm.max(va[0].hypot(va[1]));
        dz = dz.max((za[0] - zb[0]).hypot(za[1] - zb[1]));
        zm = zm.max(za[0].hypot(za[1]));
        if let (Some(sa), Some(sb)) = (a.stress(k), b.stress(k)) {
            ds = ds.max((0..3).map(|c| (sa[c] - sb[c]).abs()).fold(0.0, f64::max));
        }
    }
    (dv / vm.max(1e-300), dz / zm.max(1e-300), ds / scale)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let case = args.get(1).cloned().unwrap_or_else(|| "wave".into());
    let kv: BTreeMap<String, String> = args.iter().skip(2).filter_map(|a| a.split_once('=').map(|(k, v)| (k.to_string(), v.to_string()))).collect();
    if case == "relax" {
        relax(&kv);
        return;
    }
    let get = |k: &str, d: &str| kv.get(k).cloned().unwrap_or_else(|| d.to_string());
    let steps: usize = get("steps", "200").parse().unwrap();
    let threads: usize = get("threads", "14").parse().unwrap();
    let options = Options { stable: get("stable", "1") == "1" };
    let cpu_steps: usize = get("cpu_steps", &steps.to_string()).parse().unwrap();
    let (solid, scale, label) = match case.as_str() {
        "lame" => {
            let h: f64 = get("h", "0.125").parse().unwrap();
            let pad: f64 = get("pad", "0").parse().unwrap();
            (lame(h, pad), 100.0, format!("\"h\":{h},\"pad\":{pad}"))
        }
        _ => {
            let n: usize = get("n", "256").parse().unwrap();
            (wave(n), 210_000.0 * 1e-3, format!("\"n\":{n}"))
        }
    };
    let nodes = solid.grid().len();
    // The CPU alone.
    let mut cpu = solid.clone();
    let t = Instant::now();
    fieldcraft_parallel::with_threads(threads, || cpu.run(cpu_steps)).unwrap();
    let cpu_step = t.elapsed().as_secs_f64() / cpu_steps as f64;
    // The device with the CPU.
    let t = Instant::now();
    let mut gpu = GpuSolid::from_env(solid, options).unwrap();
    let setup = t.elapsed().as_secs_f64();
    gpu.run(1).unwrap();
    let warm = gpu.timings();
    let t = Instant::now();
    fieldcraft_parallel::with_threads(threads, || gpu.run(steps - 1)).unwrap();
    let gpu_step = t.elapsed().as_secs_f64() / (steps - 1) as f64;
    let tm = gpu.timings();
    let per = |a: f64, b: f64| (a - b) / (tm.steps - warm.steps) as f64;
    let (stepped, sent, received) = gpu.counts();
    let adapter = gpu.adapter_name().to_string();
    let t = Instant::now();
    let solid = gpu.into_solid().unwrap();
    let sync = t.elapsed().as_secs_f64();
    let (dv, dz, ds) = if cpu_steps == steps { compare(&cpu, &solid, scale) } else { (f64::NAN, f64::NAN, f64::NAN) };
    println!(
        "{{\"case\":\"{case}\",{label},\"nodes\":{nodes},\"steps\":{steps},\"adapter\":\"{adapter}\",\"stable\":{},\"threads\":{threads},\"stepped\":{stepped},\"sent\":{sent},\"received\":{received},\"cpu_step\":{cpu_step:.4e},\"gpu_step\":{gpu_step:.4e},\"submit\":{:.4e},\"cpu_part\":{:.4e},\"wait\":{:.4e},\"checks\":{:.4e},\"setup\":{setup:.3},\"sync\":{sync:.4e},\"dv\":{dv:.3e},\"dzeta\":{dz:.3e},\"dstress\":{ds:.3e}}}",
        options.stable,
        per(tm.submit, warm.submit),
        per(tm.cpu, warm.cpu),
        per(tm.wait, warm.wait),
        per(tm.checks, warm.checks),
    );
}
