//! Verification of the mapped (sharp cut-cell) boundary.
#![allow(clippy::needless_range_loop)]

use fieldcraft_eulerian::{Boundary, Grid, Limits, Material, NodeKind, Settings, Shape, Solid};

fn centred_grid(n: usize, h: f64) -> Grid {
    let o = -0.5 * (n as f64 - 1.0) * h;
    Grid::new(n, n, h, [o, o], [false; 2], &Limits::default()).unwrap()
}

fn mapped() -> Settings {
    Settings { boundary: Boundary::Mapped, ..Settings::default() }
}

fn disc() -> Shape {
    Shape::Disc { center: [0.013, -0.021], radius: 0.5 }
}

fn solid_nodes(s: &Solid) -> impl Iterator<Item = usize> + '_ {
    (0..s.grid().len()).filter(|&k| matches!(s.kind(k), Some(NodeKind::Solid | NodeKind::Fixed)))
}

/// A neo-Hookean disc held in the homogeneous deformation F by the matching ambient stress
/// stays at rest with the closed-form stress at every node, the surface included.
fn homogeneous(f: [[f64; 2]; 2], expected: [f64; 3]) {
    let m = Material::neo_hookean(100.0, 0.3, 1.0).unwrap();
    let mut s = Solid::new(centred_grid(41, 0.05), m, disc(), mapped(), Limits::default()).unwrap();
    let det = f[0][0] * f[1][1] - f[0][1] * f[1][0];
    let finv = [[f[1][1] / det, -f[0][1] / det], [-f[1][0] / det, f[0][0] / det]];
    s.set_motion(|x| ([finv[0][0] * x[0] + finv[0][1] * x[1], finv[1][0] * x[0] + finv[1][1] * x[1]], [0.0, 0.0])).unwrap();
    s.set_ambient_field(move |_| expected).unwrap();
    s.run(300).unwrap();
    let mut n = 0;
    for k in solid_nodes(&s) {
        let st = s.stress(k).unwrap();
        for c in 0..3 {
            assert!((st[c] - expected[c]).abs() < 1e-9 * 100.0, "σ{c} = {} vs {} at {:?}", st[c], expected[c], s.grid().position(k));
        }
        n += 1;
    }
    assert!(n > 250, "{n}");
    for sp in s.surface_stress() {
        for c in 0..3 {
            assert!((sp.stress[c] - expected[c]).abs() < 1e-9 * 100.0, "surface σ{c} = {} vs {}", sp.stress[c], expected[c]);
        }
    }
    assert!(s.max_speed() < 1e-9, "the disc moved: {}", s.max_speed());
}

#[test]
fn mapped_uniaxial_stretch_is_exact() {
    let (young, nu) = (100.0, 0.3);
    let mu = young / (2.0 * (1.0 + nu));
    let lambda = young * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    let st: f64 = 1.25;
    homogeneous([[st, 0.0], [0.0, 1.0]], [mu * (st * st - 1.0) / st + lambda * st.ln() / st, lambda * st.ln() / st, 0.0]);
}

#[test]
fn mapped_simple_shear_is_exact() {
    let mu = 100.0 / 2.6;
    let g = 0.3;
    homogeneous([[1.0, g], [0.0, 1.0]], [mu * g * g, 0.0, mu * g]);
}

/// Rigid translation of a disc across the grid: every field is uniform, so the cut cells, the
/// surface values and the fluxes reproduce it exactly while the surface sweeps over nodes. Exact
/// to round-off, which the surface map amplifies when the surface passes very close to a node
/// (weights ~1/s): up to 5e-10 in v and 4e-9 in σ here, against 1e-11 for the blurred surface.
#[test]
fn mapped_rigid_translation_is_exact() {
    let m = Material::neo_hookean(100.0, 0.3, 1.0).unwrap();
    let mut s = Solid::new(centred_grid(41, 0.05), m, disc(), mapped(), Limits::default()).unwrap();
    let v = [0.7, -0.4];
    s.set_motion(|x| (x, v)).unwrap();
    s.run(400).unwrap();
    let t = s.time();
    assert!(t * v[0] > 5.0 * 0.05, "moved {}", t * v[0]);
    let mut n = 0;
    for k in solid_nodes(&s) {
        let x = s.grid().position(k);
        let xi = s.reference(k).unwrap();
        assert!((xi[0] - (x[0] - v[0] * t)).abs() < 1e-10 && (xi[1] - (x[1] - v[1] * t)).abs() < 1e-10, "ξ at {x:?}: {xi:?}");
        let vel = s.velocity(k).unwrap();
        assert!((vel[0] - v[0]).abs() < 1e-8 && (vel[1] - v[1]).abs() < 1e-8, "v at {x:?}: {vel:?}");
        let st = s.stress(k).unwrap();
        assert!(st.iter().all(|c| c.abs() < 1e-7), "stress {st:?}");
        n += 1;
    }
    assert!((280..=350).contains(&n), "{n} solid nodes");
    assert!(s.surface_stress().iter().all(|p| p.stress.iter().all(|c| c.abs() < 1e-7)));
}

/// A free disc set spinning rigidly turns a quarter of a revolution: the reference map
/// follows the rotation and the only stress is the small centrifugal response ρω²R².
#[test]
fn mapped_rigid_rotation_turns_the_disc() {
    let m = Material::neo_hookean(100.0, 0.3, 1.0).unwrap();
    let c = [0.013, -0.021];
    let mut s = Solid::new(centred_grid(41, 0.05), m, disc(), mapped(), Limits::default()).unwrap();
    let omega = 1.0;
    s.set_motion(|x| (x, [-omega * (x[1] - c[1]), omega * (x[0] - c[0])])).unwrap();
    let steps = (std::f64::consts::FRAC_PI_2 / omega / s.dt()).round() as usize;
    s.run(steps).unwrap();
    let th = omega * s.time();
    let (ct, st) = (th.cos(), th.sin());
    let mu = 100.0 / 2.6;
    let (mut e_xi, mut e_s): (f64, f64) = (0.0, 0.0);
    for k in solid_nodes(&s) {
        let x = s.grid().position(k);
        let d = [x[0] - c[0], x[1] - c[1]];
        let exact = [c[0] + ct * d[0] + st * d[1], c[1] - st * d[0] + ct * d[1]];
        let xi = s.reference(k).unwrap();
        e_xi = e_xi.max((xi[0] - exact[0]).hypot(xi[1] - exact[1]));
        e_s = e_s.max(s.stress(k).unwrap().iter().fold(0.0f64, |a, v| a.max(v.abs())) / mu);
    }
    eprintln!("mapped rotation by {th:.3}: |ξ error| {e_xi:.3e}, |σ|/μ {e_s:.3e}");
    // ρω²R² / μ = 0.25 / 38.5: the centrifugal stress, about 0.7 % of μ (twice that dynamically).
    assert!(e_s < 0.03, "stress {e_s}");
    assert!(e_xi < 5e-3, "ξ error {e_xi}");
}

/// With no surface (a periodic body), the mapped boundary is the plain stencil: the wave
/// speeds are those of the blurred scheme.
#[test]
fn mapped_wave_speeds_match_the_elastic_constants() {
    let m = Material::neo_hookean(100.0, 0.3, 1.0).unwrap();
    for (shear, exact) in [(false, m.p_wave_speed()), (true, m.s_wave_speed())] {
        let per_wave = 32;
        let h = 1.0 / per_wave as f64;
        let g = Grid::new(per_wave, 5, h, [0.0, 0.0], [true, true], &Limits::default()).unwrap();
        let settings = Settings { viscosity: 0.0, boundary: Boundary::Mapped, ..Settings::default() };
        let mut s = Solid::new(g, m, Shape::Everywhere, settings, Limits::default()).unwrap();
        let k = 2.0 * std::f64::consts::PI;
        let c = if shear { 1 } else { 0 };
        s.set_motion(|x| {
            let mut xi = x;
            xi[c] -= 1e-5 * (k * x[0]).sin();
            (xi, [0.0, 0.0])
        })
        .unwrap();
        let probe = s.grid().index(per_wave / 4, 2);
        let mut prev = (s.time(), s.displacement(probe).unwrap()[c]);
        let mut speed = f64::NAN;
        for _ in 0..100_000 {
            s.run(1).unwrap();
            let now = (s.time(), s.displacement(probe).unwrap()[c]);
            if now.1 <= 0.0 {
                let tq = prev.0 + (now.0 - prev.0) * prev.1 / (prev.1 - now.1);
                speed = std::f64::consts::FRAC_PI_2 / tq / k;
                break;
            }
            prev = now;
        }
        assert!(((speed - exact) / exact).abs() < 2.5e-3, "{} wave: {speed} vs {exact}", if shear { "S" } else { "P" });
    }
}

/// The Lamé cylinder of verification.rs (a = 10, b = 20, p = 100, E = 210000, ν = 0.3, plane
/// strain) with the mapped boundary, relaxed at spacing h: relative errors of u_r(a) (from the
/// fit u_r = A r + B/r over nodes with φ < −2h) and of σθ at the bore surface points.
fn lame_mapped(h: f64) -> (f64, f64) {
    let (a, b, p) = (10.0, 20.0, 100.0);
    let m = Material::neo_hookean(210_000.0, 0.3, 1.0).unwrap();
    let n = (2.0 * (b + 5.0) / h).round() as usize + 1;
    let c = m.p_wave_speed();
    let settings = Settings { damping: 0.5 * c / b, boundary: Boundary::Mapped, ..Settings::default() };
    let mut s = Solid::new(centred_grid(n, h), m, Shape::Annulus { center: [0.0, 0.0], inner: a, outer: b }, settings, Limits::default()).unwrap();
    s.set_ambient_field(move |x| if x[0].hypot(x[1]) < 0.5 * (a + b) { [-p, -p, 0.0] } else { [0.0; 3] }).unwrap();
    let r = s.relax(1e-5 * 0.009 * c / b, 400_000).unwrap();
    assert!(r.converged, "{r:?}");
    let (mut m11, mut m12, mut m22, mut r1, mut r2) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for k in solid_nodes(&s) {
        let x = s.grid().position(k);
        let rr = x[0].hypot(x[1]);
        let u = s.displacement(k).unwrap();
        let ur = (u[0] * x[0] + u[1] * x[1]) / rr;
        if s.phi(k).unwrap() < -2.0 * h {
            m11 += rr * rr;
            m12 += 1.0;
            m22 += 1.0 / (rr * rr);
            r1 += rr * ur;
            r2 += ur / rr;
        }
    }
    let det = m11 * m22 - m12 * m12;
    let (ca, cb) = ((r1 * m22 - r2 * m12) / det, (m11 * r2 - m12 * r1) / det);
    let k = p * a * a / (b * b - a * a);
    let exact_ur = 1.3 * k / 210_000.0 * (0.4 * a + b * b / a);
    let exact_hoop = k * (1.0 + b * b / (a * a));
    let mut worst: f64 = 0.0;
    for sp in s.surface_stress() {
        let x = sp.position;
        let rr = x[0].hypot(x[1]);
        if rr < 0.5 * (a + b) {
            let (cs, sn) = (x[0] / rr, x[1] / rr);
            let hoop = sp.stress[0] * sn * sn + sp.stress[1] * cs * cs - 2.0 * sp.stress[2] * sn * cs;
            worst = worst.max(((hoop - exact_hoop) / exact_hoop).abs());
        }
    }
    let e = ((ca * a + cb / a - exact_ur) / exact_ur).abs();
    eprintln!("mapped Lamé h = {h}: {} steps, u_r(a) error {e:.3e}, worst bore σθ error {worst:.3e}", r.steps);
    (e, worst)
}

/// Ten nodes across the wall: u_r(a) within 0.5 % (the blurred surface is 11 % off) and the
/// hoop stress at every bore surface point within 5 %.
#[test]
fn mapped_thick_cylinder_matches_lame() {
    let (e, hoop) = lame_mapped(1.0);
    assert!(e < 5e-3 && hoop < 0.05, "{e} {hoop}");
}

/// Twenty nodes across the wall: u_r(a) within 0.3 % (the finite-strain law differs from
/// the linear closed form by about 0.18 % at this load) and the bore hoop stress within 1 %.
#[test]
#[ignore = "heavy: cargo xtask test --heavy (CI heavy job)"]
fn mapped_thick_cylinder_converges() {
    let (e, hoop) = lame_mapped(0.5);
    assert!(e < 3e-3 && hoop < 0.01, "{e} {hoop}");
}

#[test]
fn mapped_hostile_inputs_are_refused() {
    let l = Limits::default();
    let m = Material::neo_hookean(100.0, 0.3, 1.0).unwrap();
    let g = Grid::new(16, 16, 0.1, [0.0; 2], [true, false], &l).unwrap();
    assert!(Solid::new(g, m, Shape::Disc { center: [0.8, 0.8], radius: 0.4 }, mapped(), l).is_err());
    let mut s = Solid::new(centred_grid(21, 0.1), m, disc(), mapped(), l).unwrap();
    assert!(s.set_body_force(|_| [f64::NAN, 0.0]).is_err());
    assert!(s.set_ambient_field(|_| [f64::INFINITY, 0.0, 0.0]).is_err());
    // A violent start is reported as an error or survives, never a panic.
    s.set_motion(|x| ([x[0], x[1] + 0.3 * (30.0 * x[0]).sin()], [0.0, 0.0])).ok();
    let _ = s.run(20);
}
