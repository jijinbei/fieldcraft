//! The GPU interior against the CPU step (double precision, the reference). Stated tolerances
//! (single precision with the stress formed from small quantities; measured on an M4 Max at
//! 1.5e-5 to 2.5e-5 in v, 1e-6 in ζ, 1e-5 of the load in the stress, with a margin):
//! - a periodic shear wave with every node on the device: after 200 steps, v within 1e-4 of the
//!   largest speed, ζ within 1e-5 of the largest, the stress within 2e-5 of μ·(strain amplitude);
//! - the pressurised ring with its deep interior on the device and the surface on the CPU:
//!   after 300 steps, v within 1e-4 of the largest speed, ζ within 2e-5 of the largest, the
//!   stress within 1e-4 of the bore pressure;
//! - a disc translating rigidly across the grid (the work split again as it goes): ξ within
//!   1e-6, v within 5e-5 of 0.81, the stress within 2e-4 of zero (μ = 38.5).
//!
//! Skipped without a GPU adapter unless `FIELDCRAFT_REQUIRE_GPU=1`, which makes a missing
//! adapter a failure; `FIELDCRAFT_GPU` picks the adapter.

use fieldcraft_eulerian::{Boundary, Grid, Limits, Material, NodeKind, Settings, Shape, Solid};
use fieldcraft_eulerian_gpu::{GpuSolid, Options};
use fieldcraft_gpu::select::gpu_required;

/// `solid` on the GPU, or `None` to skip (a failure when a GPU is required).
fn gpu(solid: Solid) -> Option<GpuSolid> {
    match GpuSolid::from_env(solid, Options::default()) {
        Ok(g) => {
            println!("GPU: {}", g.adapter_name());
            Some(g)
        }
        Err(e) if gpu_required() => panic!("a GPU is required: {e}"),
        // Only a missing device skips: a shader that does not compile is a failure anywhere.
        Err(e) if e.to_string().contains("no GPU device") || e.to_string().contains("adapter") => {
            println!("skipping: {e}");
            None
        }
        Err(e) => panic!("{e}"),
    }
}

/// The largest differences over the solid nodes: of v over the largest |v|, of ζ over the
/// largest |ζ|, of the stress (absolute).
fn difference(a: &Solid, b: &Solid) -> (f64, f64, f64) {
    let (mut dv, mut vm, mut dz, mut zm, mut ds) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for k in 0..a.grid().len() {
        if a.kind(k) != Some(NodeKind::Solid) {
            continue;
        }
        let ((za, va), (zb, vb)) = (a.node_state(k).unwrap(), b.node_state(k).unwrap());
        dv = dv.max((va[0] - vb[0]).hypot(va[1] - vb[1]));
        vm = vm.max(va[0].hypot(va[1]));
        dz = dz.max((za[0] - zb[0]).hypot(za[1] - zb[1]));
        zm = zm.max(za[0].hypot(za[1]));
        let (sa, sb) = (a.stress(k).unwrap(), b.stress(k).unwrap());
        ds = ds.max((0..3).map(|c| (sa[c] - sb[c]).abs()).fold(0.0, f64::max));
    }
    (dv / vm, dz / zm, ds)
}

#[test]
fn a_periodic_shear_wave_matches_the_cpu() {
    let n = 96;
    let g = Grid::new(n, n, 1.0 / n as f64, [0.0, 0.0], [true, true], &Limits::default()).unwrap();
    let m = Material::neo_hookean(210_000.0, 0.3, 1.0).unwrap();
    let mut s = Solid::new(g, m, Shape::Everywhere, Settings { boundary: Boundary::Mapped, ..Settings::default() }, Limits::default()).unwrap();
    let (c, tau) = (m.s_wave_speed(), std::f64::consts::TAU);
    s.set_motion(|x| (x, [1e-3 * c * (tau * x[1]).sin(), 2e-4 * c * (tau * x[0]).cos()])).unwrap();
    let mut cpu = s.clone();
    let Some(mut g) = gpu(s) else { return };
    assert_eq!(g.counts(), (n * n, 0, 0), "every node on the device");
    cpu.run(200).unwrap();
    g.run(200).unwrap();
    let solid = g.into_solid().unwrap();
    assert_eq!(solid.time(), cpu.time());
    let (dv, dz, ds) = difference(&cpu, &solid);
    let mu = 210_000.0 / 2.6;
    println!("wave: dv {dv:.2e}, dzeta {dz:.2e}, dstress/(mu 1e-3) {:.2e}", ds / (mu * 1e-3));
    assert!(dv < 1e-4 && dz < 1e-5 && ds < 2e-5 * mu * 1e-3, "{dv:e} {dz:e} {ds:e}");
}

#[test]
fn the_pressurised_ring_matches_the_cpu() {
    let h = 0.25;
    let n = (50.0 / h) as usize + 1;
    let o = -0.5 * (n as f64 - 1.0) * h;
    let grid = Grid::new(n, n, h, [o, o], [false; 2], &Limits::default()).unwrap();
    let m = Material::neo_hookean(210_000.0, 0.3, 1.0).unwrap();
    let settings = Settings { damping: 0.5 * m.p_wave_speed() / 20.0, boundary: Boundary::Mapped, ..Settings::default() };
    let mut s = Solid::new(grid, m, Shape::Annulus { center: [0.03, -0.02], inner: 10.0, outer: 20.0 }, settings, Limits::default()).unwrap();
    s.set_ambient_field(|x| if x[0].hypot(x[1]) < 15.0 { [-100.0, -100.0, 0.0] } else { [0.0; 3] }).unwrap();
    let mut cpu = s.clone();
    let Some(mut g) = gpu(s) else { return };
    let (stepped, sent, received) = g.counts();
    assert!(stepped > 5000 && sent > 0 && received > 0, "{stepped} {sent} {received}");
    cpu.run(300).unwrap();
    g.run(300).unwrap();
    let solid = g.into_solid().unwrap();
    let (dv, dz, ds) = difference(&cpu, &solid);
    println!("ring: dv {dv:.2e}, dzeta {dz:.2e}, dstress/P {:.2e}", ds / 100.0);
    assert!(dv < 1e-4 && dz < 2e-5 && ds < 1e-4 * 100.0, "{dv:e} {dz:e} {ds:e}");
}

/// A disc translating rigidly across the grid: the surface sweeps over ten cells, so the work
/// is split again several times on the way, and every field stays uniform (ξ = x − v t, v
/// constant, no stress) to single precision.
#[test]
fn a_translating_disc_stays_rigid_while_the_split_follows_it() {
    let n = 101;
    let h = 0.02;
    let o = -0.5 * (n as f64 - 1.0) * h;
    let grid = Grid::new(n, n, h, [o, o], [false; 2], &Limits::default()).unwrap();
    let m = Material::neo_hookean(100.0, 0.3, 1.0).unwrap();
    let settings = Settings { boundary: Boundary::Mapped, ..Settings::default() };
    let mut s = Solid::new(grid, m, Shape::Disc { center: [-0.2, 0.1], radius: 0.5 }, settings, Limits::default()).unwrap();
    let v = [0.7, -0.4];
    s.set_motion(|x| (x, v)).unwrap();
    let Some(mut g) = gpu(s) else { return };
    let first = g.counts().0;
    assert!(first > 500, "{first} nodes on the device");
    g.run(400).unwrap();
    let splits = g.timings().splits;
    let s = g.into_solid().unwrap();
    let t = s.time();
    assert!(splits >= 3, "split {splits} times");
    assert!(t * v[0] > 8.0 * h, "moved {}", t * v[0]);
    let (mut exi, mut ev, mut es) = (0.0f64, 0.0f64, 0.0f64);
    for k in 0..s.grid().len() {
        if s.kind(k) != Some(NodeKind::Solid) {
            continue;
        }
        let x = s.grid().position(k);
        let xi = s.reference(k).unwrap();
        exi = exi.max((xi[0] - (x[0] - v[0] * t)).hypot(xi[1] - (x[1] - v[1] * t)));
        let w = s.velocity(k).unwrap();
        ev = ev.max((w[0] - v[0]).hypot(w[1] - v[1]));
        es = es.max(s.stress(k).unwrap().iter().fold(0.0f64, |a, c| a.max(c.abs())));
    }
    println!("translation ({splits} splits): |xi error| {exi:.2e}, |v error| {ev:.2e}, |stress| {es:.2e}");
    // |v| = 0.81, μ = 38.5 (the CPU alone: v to 5e-10, σ to 4e-9).
    assert!(exi < 1e-6 && ev < 5e-5 && es < 2e-4, "{exi:e} {ev:e} {es:e}");
}
