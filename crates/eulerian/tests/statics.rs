//! Verification of the static solve (Newton–Krylov with the multigrid preconditioner) of the
//! mapped boundary: it reaches the equilibrium the dynamic relaxation tends to.
#![allow(clippy::needless_range_loop)]

use fieldcraft_eulerian::{Boundary, Grid, Limits, Material, NodeKind, Settings, Shape, Solid, StaticSettings};

fn centred_grid(n: usize, h: f64) -> Grid {
    let o = -0.5 * (n as f64 - 1.0) * h;
    Grid::new(n, n, h, [o, o], [false; 2], &Limits::default()).unwrap()
}

fn mapped() -> Settings {
    Settings { boundary: Boundary::Mapped, ..Settings::default() }
}

fn solid_nodes(s: &Solid) -> impl Iterator<Item = usize> + '_ {
    (0..s.grid().len()).filter(|&k| matches!(s.kind(k), Some(NodeKind::Solid | NodeKind::Fixed)))
}

/// A neo-Hookean disc loaded by the ambient stress of a homogeneous finite deformation
/// reaches that deformation: the stress is the closed form at every node and surface point
/// (the static solution is the homogeneous one up to the rigid motion the projection fixes).
fn homogeneous_static(expected: [f64; 3]) {
    let m = Material::neo_hookean(100.0, 0.3, 1.0).unwrap();
    let shape = Shape::Disc { center: [0.013, -0.021], radius: 0.5 };
    let mut s = Solid::new(centred_grid(41, 0.05), m, shape, mapped(), Limits::default()).unwrap();
    s.set_ambient_field(move |_| expected).unwrap();
    let rep = s.solve_static(&StaticSettings { increments: 4, ..StaticSettings::default() }).unwrap();
    assert!(rep.converged, "{rep:?}");
    let mut n = 0;
    for k in solid_nodes(&s) {
        let st = s.stress(k).unwrap();
        for c in 0..3 {
            assert!((st[c] - expected[c]).abs() < 1e-7 * 100.0, "σ{c} = {} vs {} at {:?}", st[c], expected[c], s.grid().position(k));
        }
        n += 1;
    }
    assert!(n > 250, "{n}");
    for sp in s.surface_stress() {
        for c in 0..3 {
            assert!((sp.stress[c] - expected[c]).abs() < 1e-7 * 100.0, "surface σ{c} = {} vs {}", sp.stress[c], expected[c]);
        }
    }
}

#[test]
fn static_uniaxial_stretch_is_exact() {
    let (young, nu) = (100.0, 0.3);
    let mu = young / (2.0 * (1.0 + nu));
    let lambda = young * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    let st: f64 = 1.25;
    homogeneous_static([mu * (st * st - 1.0) / st + lambda * st.ln() / st, lambda * st.ln() / st, 0.0]);
}

/// The Lamé cylinder (a = 10, b = 20, p = 100, E = 210000, ν = 0.3) at h = 1: the static
/// solution equals the relaxed one to the relaxation's tolerance, in a few Newton steps of a
/// few tens of Krylov iterations each.
#[test]
fn static_lame_matches_relaxation() {
    let (a, b, p, h): (f64, f64, f64, f64) = (10.0, 20.0, 100.0, 1.0);
    let m = Material::neo_hookean(210_000.0, 0.3, 1.0).unwrap();
    let n = (2.0 * (b + 5.0) / h).round() as usize + 1;
    let c = m.p_wave_speed();
    let settings = Settings { damping: 0.5 * c / b, boundary: Boundary::Mapped, ..Settings::default() };
    let mut s = Solid::new(centred_grid(n, h), m, Shape::Annulus { center: [0.0, 0.0], inner: a, outer: b }, settings, Limits::default()).unwrap();
    s.set_ambient_field(move |x| if x[0].hypot(x[1]) < 0.5 * (a + b) { [-p, -p, 0.0] } else { [0.0; 3] }).unwrap();
    let mut relaxed = s.clone();
    let rep = s.solve_static(&StaticSettings::default()).unwrap();
    assert!(rep.converged && rep.newton.len() <= 6, "{rep:?}");
    assert!(rep.newton.iter().all(|st| st.krylov <= 30), "{:?}", rep.newton);
    // Relaxed ten times tighter than the verification tests.
    let r = relaxed.relax(1e-6 * 0.009 * c / b, 400_000).unwrap();
    assert!(r.converged, "{r:?}");
    let (mut dz, mut umax): (f64, f64) = (0.0, 0.0);
    for k in solid_nodes(&s) {
        assert!(matches!(relaxed.kind(k), Some(NodeKind::Solid)), "node {k} is in the static body only");
        let (p, q) = (s.displacement(k).unwrap(), relaxed.displacement(k).unwrap());
        dz = dz.max((p[0] - q[0]).hypot(p[1] - q[1]));
        umax = umax.max(p[0].hypot(p[1]));
    }
    eprintln!("static vs relaxed Lamé h = 1: max |Δu| {dz:.3e} of max |u| {umax:.3e}");
    assert!(dz < 1e-4 * umax, "{dz} vs {umax}");
}

#[test]
fn static_hostile_settings_are_refused() {
    let l = Limits::default();
    let m = Material::neo_hookean(100.0, 0.3, 1.0).unwrap();
    let shape = Shape::Disc { center: [0.0, 0.0], radius: 0.5 };
    let mut s = Solid::new(centred_grid(21, 0.1), m, shape, mapped(), l).unwrap();
    for bad in [
        StaticSettings { tolerance: f64::NAN, ..StaticSettings::default() },
        StaticSettings { tolerance: 2.0, ..StaticSettings::default() },
        StaticSettings { increments: 0, ..StaticSettings::default() },
        StaticSettings { probe: 4, ..StaticSettings::default() },
        StaticSettings { min_increment: 0.0, ..StaticSettings::default() },
        StaticSettings { restart: 0, ..StaticSettings::default() },
        StaticSettings { sweeps: 0, ..StaticSettings::default() },
    ] {
        assert!(s.solve_static(&bad).is_err(), "{bad:?}");
    }
    let mut blurred = Solid::new(centred_grid(21, 0.1), m, shape, Settings { boundary: Boundary::Blurred, ..Settings::default() }, l).unwrap();
    assert!(blurred.solve_static(&StaticSettings::default()).is_err());
    // An unloaded body is already in equilibrium.
    let rep = s.solve_static(&StaticSettings::default()).unwrap();
    assert!(rep.converged && rep.newton.is_empty(), "{rep:?}");
    // A load that inverts the body is an error or a solution, never a panic.
    s.set_ambient_field(|_| [-1e6, -1e6, 0.0]).unwrap();
    let _ = s.solve_static(&StaticSettings { min_increment: 0.05, ..StaticSettings::default() });
}
