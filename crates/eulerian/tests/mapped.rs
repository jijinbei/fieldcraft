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
