//! Verification of local refinement (`fieldcraft_eulerian::refine`).
#![allow(clippy::needless_range_loop)]

use fieldcraft_eulerian::{Boundary, Grid, Hierarchy, Limits, Material, Refinement, Settings, Shape, Solid};

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

fn refined(levels: usize, h0: f64, shape: Shape, m: Material) -> Hierarchy {
    let n = (1.5 / h0).round() as usize + 1;
    let r = Refinement { levels, band: 4.0, ..Refinement::default() };
    Hierarchy::new(centred_grid(n, h0), m, shape, mapped(), r, Limits::default()).unwrap()
}

/// A homogeneous deformation F held by the matching ambient stress: the composite residual
/// vanishes at every degree of freedom of every level, the run stays at rest and the stress
/// is exact everywhere, across the coarse–fine interfaces and at the surface.
fn patch(f: [[f64; 2]; 2], expected: [f64; 3], levels: usize) {
    let m = Material::neo_hookean(100.0, 0.3, 1.0).unwrap();
    let mut s = refined(levels, 0.1, disc(), m);
    let det = f[0][0] * f[1][1] - f[0][1] * f[1][0];
    let finv = [[f[1][1] / det, -f[0][1] / det], [-f[1][0] / det, f[0][0] / det]];
    s.set_motion(|x| ([finv[0][0] * x[0] + finv[0][1] * x[1], finv[1][0] * x[0] + finv[1][1] * x[1]], [0.0, 0.0])).unwrap();
    s.set_ambient_field(move |_| expected).unwrap();
    let res = s.residual().unwrap();
    let scale = 100.0 * 0.1 * 0.1 / 8.0f64.powi(levels as i32);
    let worst = res.iter().map(|r| r.force[0].abs().max(r.force[1].abs())).fold(0.0, f64::max);
    eprintln!("levels {levels}: {} dofs, worst residual {worst:.3e} (scale {scale:.1e})", res.len());
    assert!(worst < 1e-9 * scale * 1e3, "residual {worst}");
    s.run(60).unwrap();
    assert!(s.max_speed() < 1e-9, "moved: {}", s.max_speed());
    for n in s.dofs() {
        let st = s.stress(&n).unwrap();
        for c in 0..3 {
            assert!((st[c] - expected[c]).abs() < 1e-7, "σ{c} = {} vs {} at {:?} level {}", st[c], expected[c], n.position, n.level);
        }
    }
    for sp in s.surface_stress() {
        for c in 0..3 {
            assert!((sp.stress[c] - expected[c]).abs() < 1e-7, "surface σ{c} = {} vs {}", sp.stress[c], expected[c]);
        }
    }
}

#[test]
fn patch_test_uniaxial_stretch_across_levels() {
    let (young, nu) = (100.0, 0.3);
    let mu = young / (2.0 * (1.0 + nu));
    let lambda = young * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    let st: f64 = 1.25;
    for levels in [1, 2] {
        patch([[st, 0.0], [0.0, 1.0]], [mu * (st * st - 1.0) / st + lambda * st.ln() / st, lambda * st.ln() / st, 0.0], levels);
    }
}

#[test]
fn patch_test_simple_shear_across_levels() {
    let mu = 100.0 / 2.6;
    let g = 0.3;
    for levels in [1, 2] {
        patch([[1.0, g], [0.0, 1.0]], [mu * g * g, 0.0, mu * g], levels);
    }
}

/// Without refinement the hierarchy is the plain mapped solid on the body's part of the grid:
/// the same numbers as the full grid.
#[test]
fn one_level_is_the_plain_grid() {
    let m = Material::linear_elastic(100.0, 0.3, 1.0).unwrap();
    let shape = Shape::Annulus { center: [0.0, 0.0], inner: 0.3, outer: 0.6 };
    let load = |x: [f64; 2]| if x[0].hypot(x[1]) < 0.45 { [-0.1, -0.1, 0.0] } else { [0.0; 3] };
    let mut dense = Solid::new(centred_grid(31, 0.05), m, shape, mapped(), Limits::default()).unwrap();
    dense.set_ambient_field(load).unwrap();
    let r = Refinement { levels: 0, ..Refinement::default() };
    let mut h = Hierarchy::new(centred_grid(31, 0.05), m, shape, mapped(), r, Limits::default()).unwrap();
    h.set_ambient_field(load).unwrap();
    dense.run(40).unwrap();
    h.run(40).unwrap();
    let lv = h.level(0).unwrap();
    let mut n = 0;
    for k in 0..lv.grid().len() {
        let (i, j) = lv.grid().ij(k);
        let kd = dense.grid().index(i, j);
        if let (Some(a), Some(b)) = (lv.displacement(k), dense.displacement(kd)) {
            assert!((a[0] - b[0]).abs() <= 1e-15 && (a[1] - b[1]).abs() <= 1e-15, "{a:?} vs {b:?}");
            n += 1;
        }
    }
    assert!(n > 300, "{n}");
}

/// The control volumes of all levels tile the body: the composite solid area equals that of
/// the uniform grid at the finest spacing (the same cut cells, full cells elsewhere).
#[test]
fn composite_control_volumes_tile_the_body() {
    let m = Material::linear_elastic(100.0, 0.3, 1.0).unwrap();
    let shape = Shape::Annulus { center: [0.01, -0.02], inner: 0.3, outer: 0.65 };
    for levels in [1, 2] {
        let mut h = refined(levels, 0.1, shape, m);
        let area: f64 = h.residual().unwrap().iter().map(|r| r.area).sum();
        let hf = 0.1 / (1 << levels) as f64;
        let n = (1.5 / hf).round() as usize + 1;
        let mut fine = Solid::new(centred_grid(n, hf), m, shape, mapped(), Limits::default()).unwrap();
        fine.run(1).unwrap();
        let exact = std::f64::consts::PI * (0.65f64.powi(2) - 0.3f64.powi(2));
        let mut h2 =
            Hierarchy::new(centred_grid(n, hf), m, shape, mapped(), Refinement { levels: 0, ..Refinement::default() }, Limits::default()).unwrap();
        let uniform: f64 = h2.residual().unwrap().iter().map(|r| r.area).sum();
        eprintln!("levels {levels}: composite area {area:.12}, uniform {uniform:.12}, exact {exact:.12}");
        assert!((area - uniform).abs() < 1e-12 * exact, "{area} vs {uniform}");
    }
}

/// A free body with internal stress and no load: summed over every degree of freedom of every
/// level, the impulses of a coarse step (fluxes, hanging nodes' shares, registers) cancel.
#[test]
fn coarse_fine_fluxes_conserve_momentum() {
    let m = Material::linear_elastic(100.0, 0.3, 1.0).unwrap();
    let mut h = refined(2, 0.1, disc(), m);
    h.set_motion(|x| ([x[0] + 1e-3 * (3.0 * x[1]).sin(), x[1] - 2e-3 * (2.0 * x[0] + x[1]).cos()], [0.0, 0.0])).unwrap();
    h.run(1).unwrap();
    let mut transit = [0.0; 2];
    for _ in 0..20 {
        h.run(1).unwrap();
        let (sum, scale, tr) = h.impulse_audit();
        let net = [sum[0] - tr[0], sum[1] - tr[1]];
        transit = [transit[0] + tr[0], transit[1] + tr[1]];
        eprintln!("impulse {net:?} of {scale:.3e}, in transit {transit:?}");
        assert!(scale > 0.0 && net[0].abs() < 1e-12 * scale && net[1].abs() < 1e-12 * scale, "{net:?} {scale}");
    }
    // The static residual sums to zero too.
    let res = h.residual().unwrap();
    let (fx, fy, mag) = res.iter().fold((0.0, 0.0, 0.0), |a, r| (a.0 + r.force[0], a.1 + r.force[1], a.2 + r.force[0].abs() + r.force[1].abs()));
    assert!(fx.abs() < 1e-12 * mag && fy.abs() < 1e-12 * mag, "{fx} {fy} {mag}");
}

/// A disc translating rigidly across the grid with regrids every few steps (the refined band
/// follows the surface): ζ, v and σ stay exact to round-off at every degree of freedom.
#[test]
fn rigid_translation_through_regrids_is_exact() {
    let m = Material::neo_hookean(100.0, 0.3, 1.0).unwrap();
    let r = Refinement { levels: 2, band: 4.0, regrid_every: 4, ..Refinement::default() };
    let mut s = Hierarchy::new(centred_grid(31, 0.1), m, disc(), mapped(), r, Limits::default()).unwrap();
    let v = [0.7, -0.4];
    s.set_motion(|x| (x, v)).unwrap();
    let before = s.stats();
    s.run(120).unwrap();
    let t = s.time();
    let after = s.stats();
    eprintln!("moved {:.3} in {} regrids; nodes {:?} -> {:?}", t * v[0], after.regrids, before.nodes, after.nodes);
    assert!(t * v[0] > 0.25 && after.regrids == 30, "{t} {after:?}");
    let (mut ex, mut ev, mut es): (f64, f64, f64) = (0.0, 0.0, 0.0);
    for n in s.dofs() {
        let sol = s.level(n.level).unwrap();
        let x = n.position;
        let xi = sol.reference(n.index).unwrap();
        ex = ex.max((xi[0] - (x[0] - v[0] * t)).abs()).max((xi[1] - (x[1] - v[1] * t)).abs());
        let vel = sol.velocity(n.index).unwrap();
        ev = ev.max((vel[0] - v[0]).abs()).max((vel[1] - v[1]).abs());
        es = es.max(s.stress(&n).unwrap().iter().fold(0.0f64, |a, c| a.max(c.abs())));
    }
    eprintln!("errors: ξ {ex:.2e}, v {ev:.2e}, σ {es:.2e}");
    assert!(ex < 1e-10 && ev < 1e-8 && es < 1e-6, "ξ {ex}, v {ev}, σ {es}");
}

/// A regrid hands the momentum of the old control volumes to the new ones: every piece of
/// old mass finds new nodes, and Σ m v over the degrees of freedom is unchanged up to the
/// cut cells next to the surface, which are rebuilt with the band of the new configuration
/// (material that entered the body in the last step now extrapolates too, as it would in the
/// next step) and move in the eighth digit.
#[test]
fn regrid_keeps_the_momentum() {
    let m = Material::linear_elastic(100.0, 0.3, 1.0).unwrap();
    let r = Refinement { levels: 2, band: 6.0, regrid_every: 4, ..Refinement::default() };
    let mut s = Hierarchy::new(centred_grid(31, 0.1), m, disc(), mapped(), r, Limits::default()).unwrap();
    let vel = |x: [f64; 2]| [0.7 - 0.5 * x[1], -0.4 + 0.5 * x[0]];
    s.set_motion(|x| (x, vel(x))).unwrap();
    s.run(30).unwrap();
    let p = |s: &mut Hierarchy| -> [f64; 2] {
        let res = s.residual().unwrap();
        let mut p = [0.0; 2];
        for r in &res {
            let v = s.level(r.node.level).unwrap().velocity(r.node.index).unwrap();
            p[0] += r.area * v[0];
            p[1] += r.area * v[1];
        }
        p
    };
    let p0 = p(&mut s);
    let nodes0 = s.stats().nodes;
    let report = s.regrid().unwrap();
    let p1 = p(&mut s);
    eprintln!("momentum {p0:?} -> {p1:?} ({nodes0:?} -> {:?} nodes); {report:?}", s.stats().nodes);
    assert!(nodes0 != s.stats().nodes, "the regrid changed nothing");
    assert!(report.unassigned == 0.0 && report.mass > 0.7, "{report:?}");
    for c in 0..2 {
        assert!((p1[c] - p0[c]).abs() < 1e-7 * p0[c].abs(), "{p0:?} {p1:?}");
    }
}
