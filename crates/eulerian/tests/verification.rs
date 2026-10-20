//! Verification of the Eulerian (reference map) solid against closed-form answers.
//!
//! Each check states its tolerance and, where the answer is not exact, a convergence check
//! between two resolutions. The numbers are summarised in docs/research/reference-map.md.

use fieldcraft_eulerian::{Boundary, Grid, Limits, Material, NodeKind, Settings, Shape, Solid};

/// A grid of n × n nodes with spacing h, symmetric about the origin.
fn centred_grid(n: usize, h: f64) -> Grid {
    let o = -0.5 * (n as f64 - 1.0) * h;
    Grid::new(n, n, h, [o, o], [false; 2], &Limits::default()).unwrap()
}

fn unit_block() -> Shape {
    Shape::Rectangle { min: [-0.5, -0.5], max: [0.5, 0.5] }
}

fn solid_nodes(s: &Solid) -> impl Iterator<Item = usize> + '_ {
    (0..s.grid().len()).filter(|&k| matches!(s.kind(k), Some(NodeKind::Solid | NodeKind::Fixed)))
}

// ------------------------------------------------------------------ 1. rigid motions

/// Rigid translation: every field is linear, so the advection and the extrapolation are exact
/// and the stress stays zero to round-off, also while the body moves across the grid.
#[test]
fn rigid_translation_is_exact() {
    for boundary in [Boundary::Blurred, Boundary::Ghost] {
        rigid_translation(boundary);
    }
}

fn rigid_translation(boundary: Boundary) {
    let m = Material::neo_hookean(100.0, 0.3, 1.0).unwrap();
    let settings = Settings { boundary, ..Settings::default() };
    let mut s = Solid::new(centred_grid(41, 0.05), m, unit_block(), settings, Limits::default()).unwrap();
    let v = [0.7, -0.4];
    s.set_motion(|x| (x, v)).unwrap();
    let steps = 400;
    s.run(steps).unwrap();
    let t = s.time();
    // The body moved by more than five cells.
    assert!(t * v[0] > 5.0 * 0.05, "moved {}", t * v[0]);
    let mut n = 0;
    for k in solid_nodes(&s) {
        let x = s.grid().position(k);
        let xi = s.reference(k).unwrap();
        assert!((xi[0] - (x[0] - v[0] * t)).abs() < 1e-11 && (xi[1] - (x[1] - v[1] * t)).abs() < 1e-11, "ξ at {x:?}: {xi:?}");
        let vel = s.velocity(k).unwrap();
        // Round-off of the ghost fits (22 × 22 solves) reaches ~1e-11 in v after 400 steps.
        assert!((vel[0] - v[0]).abs() < 1e-10 && (vel[1] - v[1]).abs() < 1e-10, "{boundary:?} v at {x:?}: {vel:?}");
        let st = s.stress(k).unwrap();
        assert!(st.iter().all(|c| c.abs() < 1e-8), "stress {st:?}");
        n += 1;
    }
    // The whole block is still there: 21 × 21 nodes give ~1 / 0.05² = 400 interior nodes.
    assert!((380..=460).contains(&n), "{n} solid nodes");
}

/// The largest error of ξ and of the stress after a rotation by π/4 of the block about the
/// origin under the prescribed rigid velocity field, at spacing h.
fn rotation_errors(h: f64, boundary: Boundary) -> (f64, f64) {
    let m = Material::neo_hookean(100.0, 0.3, 1.0).unwrap();
    let n = (2.0 / h).round() as usize + 1;
    let settings = Settings { boundary, ..Settings::default() };
    let mut s = Solid::new(centred_grid(n, h), m, unit_block(), settings, Limits::default()).unwrap();
    let omega = 1.0;
    let theta = std::f64::consts::FRAC_PI_4;
    let steps = (theta / omega / s.dt()).ceil() as usize;
    s.advect(steps, |x, _| [-omega * x[1], omega * x[0]]).unwrap();
    let t = s.time();
    let (c, sn) = ((omega * t).cos(), (omega * t).sin());
    let mu = 100.0 / 2.6;
    let mut e_xi: f64 = 0.0;
    let mut e_s: f64 = 0.0;
    for k in solid_nodes(&s) {
        let x = s.grid().position(k);
        let exact = [c * x[0] + sn * x[1], -sn * x[0] + c * x[1]];
        let xi = s.reference(k).unwrap();
        e_xi = e_xi.max((xi[0] - exact[0]).hypot(xi[1] - exact[1]));
        let st = s.stress(k).unwrap();
        e_s = e_s.max(st.iter().fold(0.0f64, |a, v| a.max(v.abs())) / mu);
    }
    (e_xi, e_s)
}

/// Rigid rotation: the stress-free state is kept and ξ is transported with the first-order
/// time error of the explicit update; halving h (and Δt) halves the error.
#[test]
fn rigid_rotation_converges() {
    for boundary in [Boundary::Blurred, Boundary::Ghost] {
        let (xi_c, s_c) = rotation_errors(0.05, boundary);
        let (xi_f, s_f) = rotation_errors(0.025, boundary);
        eprintln!("rotation ({boundary:?}): |ξ error| {xi_c:.3e} → {xi_f:.3e}, |σ|/μ {s_c:.3e} → {s_f:.3e}");
        assert!(xi_c < 2e-3 && xi_f < 1e-3, "ξ errors {xi_c} {xi_f}");
        assert!(xi_f < 0.6 * xi_c, "ξ error must fall about linearly: {xi_c} → {xi_f}");
        assert!(s_c < 1e-2 && s_f < 0.6 * s_c, "stress errors {s_c} {s_f}");
    }
}

// ------------------------------------------------------- 2. homogeneous finite strain

/// A neo-Hookean block held in the homogeneous deformation with gradient `f` by the matching
/// surface traction: the stress equals the closed form at every node and the block stays at
/// rest (the linear reference map is reproduced exactly by the stencils and the extrapolation).
fn homogeneous(f: [[f64; 2]; 2], expected: [f64; 3]) {
    for boundary in [Boundary::Blurred, Boundary::Ghost] {
        homogeneous_with(f, expected, boundary);
    }
}

fn homogeneous_with(f: [[f64; 2]; 2], expected: [f64; 3], boundary: Boundary) {
    let m = Material::neo_hookean(100.0, 0.3, 1.0).unwrap();
    let settings = Settings { boundary, ..Settings::default() };
    let mut s = Solid::new(centred_grid(51, 0.05), m, unit_block(), settings, Limits::default()).unwrap();
    let det = f[0][0] * f[1][1] - f[0][1] * f[1][0];
    let finv = [[f[1][1] / det, -f[0][1] / det], [-f[1][0] / det, f[0][0] / det]];
    s.set_motion(|x| ([finv[0][0] * x[0] + finv[0][1] * x[1], finv[1][0] * x[0] + finv[1][1] * x[1]], [0.0, 0.0])).unwrap();
    s.set_ambient(|_| expected).unwrap();
    s.run(300).unwrap();
    let mut n = 0;
    for k in solid_nodes(&s) {
        let st = s.stress(k).unwrap();
        for c in 0..3 {
            assert!((st[c] - expected[c]).abs() < 1e-9 * 100.0, "σ{c} = {} vs {}", st[c], expected[c]);
        }
        n += 1;
    }
    assert!(n > 300);
    assert!(s.max_speed() < 1e-9, "the block moved: {}", s.max_speed());
}

#[test]
fn uniaxial_stretch_matches_the_closed_form() {
    let (young, nu) = (100.0, 0.3);
    let mu = young / (2.0 * (1.0 + nu));
    let lambda = young * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    let st: f64 = 1.25;
    // Plane strain, F = diag(s, 1): σxx = μ(s² − 1)/s + λ ln s / s, σyy = λ ln s / s.
    homogeneous([[st, 0.0], [0.0, 1.0]], [mu * (st * st - 1.0) / st + lambda * st.ln() / st, lambda * st.ln() / st, 0.0]);
}

#[test]
fn simple_shear_matches_the_closed_form() {
    let mu = 100.0 / 2.6;
    let g = 0.3;
    // F = [[1, γ], [0, 1]]: σxx = μγ², σxy = μγ, σyy = 0.
    homogeneous([[1.0, g], [0.0, 1.0]], [mu * g * g, 0.0, mu * g]);
}

// ---------------------------------------------- 3. small strain under surface traction

/// The strains (εxx, εyy) in the middle of a free block pulled by the traction `t` on its x
/// faces, relaxed to rest at spacing h.
fn traction_strain(material: Material, h: f64, t: f64, boundary: Boundary) -> (f64, f64) {
    let n = (2.0 / h).round() as usize + 1;
    let c = material.p_wave_speed();
    let settings = Settings { damping: 2.0 * c, boundary, ..Settings::default() };
    let mut s = Solid::new(centred_grid(n, h), material, unit_block(), settings, Limits::default()).unwrap();
    s.set_ambient(|_| [t, 0.0, 0.0]).unwrap();
    let young = 100.0;
    let r = s.relax(1e-7 * c * t / young, 200_000).unwrap();
    assert!(r.converged, "{r:?}");
    // Least-squares slopes of u_x(x) and u_y(y) over the middle of the block.
    let (mut sxx, mut sxu, mut syy, mut syu, mut cnt) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for k in solid_nodes(&s) {
        let x = s.grid().position(k);
        if x[0].abs() < 0.3 && x[1].abs() < 0.3 {
            let u = s.displacement(k).unwrap();
            sxx += x[0] * x[0];
            sxu += x[0] * u[0];
            syy += x[1] * x[1];
            syu += x[1] * u[1];
            cnt += 1.0;
        }
    }
    assert!(cnt > 10.0);
    (sxu / sxx, syu / syy)
}

/// Plane strain uniaxial traction: εxx = T(1 − ν²)/E, εyy = −Tν(1 + ν)/E. The homogeneous
/// state is reproduced exactly by the scheme, so the error is the relaxation tolerance (linear)
/// and the O(T/E) nonlinearity (neo-Hookean).
#[test]
fn small_strain_traction_matches_linear_elasticity() {
    let (young, nu, t) = (100.0, 0.3, 0.01);
    let exx = t * (1.0 - nu * nu) / young;
    let eyy = -t * nu * (1.0 + nu) / young;
    for (name, m, tol) in
        [("linear", Material::linear_elastic(young, nu, 1.0).unwrap(), 1e-4), ("neo-Hookean", Material::neo_hookean(young, nu, 1.0).unwrap(), 5e-4)]
    {
        for boundary in [Boundary::Blurred, Boundary::Ghost] {
            let (a, b) = traction_strain(m, 0.1, t, boundary);
            eprintln!("{name} ({boundary:?}): εxx {a:.6e} (exact {exx:.6e}), εyy {b:.6e} (exact {eyy:.6e})");
            assert!(((a - exx) / exx).abs() < tol && ((b - eyy) / eyy).abs() < tol, "{name}: {a} {b}");
        }
    }
}

// ------------------------------------------------------------------ 4. wave speeds

/// The phase speed of a standing wave u = A sin(kx) (along x for the P-wave, along y for the
/// S-wave) in a periodic neo-Hookean domain with `per_wave` nodes per wavelength, from the
/// first zero crossing of the displacement at the antinode (a quarter period).
fn wave_speed(per_wave: usize, shear: bool) -> (f64, f64) {
    let m = Material::neo_hookean(100.0, 0.3, 1.0).unwrap();
    let h = 1.0 / per_wave as f64;
    let g = Grid::new(per_wave, 5, h, [0.0, 0.0], [true, true], &Limits::default()).unwrap();
    let settings = Settings { viscosity: 0.0, ..Settings::default() };
    let mut s = Solid::new(g, m, Shape::Everywhere, settings, Limits::default()).unwrap();
    let k = 2.0 * std::f64::consts::PI;
    let amp = 1e-5;
    let c = if shear { 1 } else { 0 };
    s.set_motion(|x| {
        let mut xi = x;
        xi[c] -= amp * (k * x[0]).sin();
        (xi, [0.0, 0.0])
    })
    .unwrap();
    let probe = s.grid().index(per_wave / 4, 2);
    let mut prev = (s.time(), s.displacement(probe).unwrap()[c]);
    for _ in 0..1_000_000 {
        s.run(1).unwrap();
        let now = (s.time(), s.displacement(probe).unwrap()[c]);
        if now.1 <= 0.0 {
            let tq = prev.0 + (now.0 - prev.0) * prev.1 / (prev.1 - now.1);
            let exact = if shear { m.s_wave_speed() } else { m.p_wave_speed() };
            // Quarter period T/4 = π/(2ω), c = ω/k.
            return (std::f64::consts::FRAC_PI_2 / tq / k, exact);
        }
        prev = now;
    }
    (f64::NAN, 0.0)
}

#[test]
fn wave_speeds_match_the_elastic_constants() {
    for shear in [false, true] {
        let (c16, exact) = wave_speed(16, shear);
        let (c32, _) = wave_speed(32, shear);
        let (e16, e32) = (((c16 - exact) / exact).abs(), ((c32 - exact) / exact).abs());
        eprintln!("{} wave: c = {c16:.5} (16/λ), {c32:.5} (32/λ), exact {exact:.5}", if shear { "S" } else { "P" });
        assert!(e16 < 1e-2 && e32 < 2.5e-3, "errors {e16} {e32}");
        assert!(e32 < 0.35 * e16, "second-order convergence expected: {e16} → {e32}");
    }
}

// ----------------------------------------------- 5. thick-walled cylinder (Lamé)

/// The cylinder of the FEM oracle case `tests/oracle/lame-cylinder-plane-strain` (a = 10,
/// b = 20, E = 210000, ν = 0.3), plane strain, as a free ring with the pressure p as the ambient
/// stress in the bore, relaxed at spacing h with the damping γ = `damping` · c_P / (2b).
/// Returns u_r = A r + B/r fitted over the nodes with φ < −2h at r = a and r = 15, and the hoop
/// stress at the nodes nearest r = 15 (averaged), all divided by p.
fn lame_cylinder(h: f64, material: Material, p: f64, boundary: Boundary, damping: f64) -> (f64, f64, f64) {
    let (a, b) = (10.0, 20.0);
    let n = (2.0 * (b + 5.0) / h).round() as usize + 1;
    let c = material.p_wave_speed();
    let settings = Settings { damping: damping * 0.5 * c / b, boundary, ..Settings::default() };
    let mut s =
        Solid::new(centred_grid(n, h), material, Shape::Annulus { center: [0.0, 0.0], inner: a, outer: b }, settings, Limits::default()).unwrap();
    s.set_ambient(|x| if x[0].hypot(x[1]) < 0.5 * (a + b) { [-p, -p, 0.0] } else { [0.0; 3] }).unwrap();
    let ur_scale = 0.009 * p / 100.0;
    let r = s.relax(1e-6 * ur_scale * c / b, 400_000).unwrap();
    assert!(r.converged, "{r:?}");
    assert_eq!(s.surface_fallbacks(), 0);
    eprintln!("Lamé h = {h} ({boundary:?}): {} nodes, {} steps to rest", s.grid().len(), r.steps);
    let (mut m11, mut m12, mut m22, mut r1, mut r2) = (0.0, 0.0, 0.0, 0.0, 0.0);
    let (mut hoop, mut nh) = (0.0, 0.0);
    for k in solid_nodes(&s) {
        let x = s.grid().position(k);
        let rr = x[0].hypot(x[1]);
        let u = s.displacement(k).unwrap();
        let ur = (u[0] * x[0] + u[1] * x[1]) / rr;
        if s.phi(k).unwrap() < -2.0 * h {
            let (f1, f2) = (rr, 1.0 / rr);
            m11 += f1 * f1;
            m12 += f1 * f2;
            m22 += f2 * f2;
            r1 += f1 * ur;
            r2 += f2 * ur;
        }
        if (rr - 15.0).abs() < 0.5 * h {
            let st = s.stress(k).unwrap();
            let (ct, sn) = (x[0] / rr, x[1] / rr);
            hoop += st[0] * sn * sn + st[1] * ct * ct - 2.0 * st[2] * sn * ct;
            nh += 1.0;
        }
    }
    let det = m11 * m22 - m12 * m12;
    let ca = (r1 * m22 - r2 * m12) / det;
    let cb = (m11 * r2 - m12 * r1) / det;
    ((ca * a + cb / a) / p, (ca * 15.0 + cb / 15.0) / p, hoop / nh / p)
}

/// Lamé, plane strain, per unit pressure: u_r = (1 + ν) a² / (E (b² − a²)) ((1 − 2ν) r + b²/r),
/// σ_θ = a² / (b² − a²) (1 + b²/r²).
fn lame_exact(r: f64) -> (f64, f64) {
    let (a, b, e, nu) = (10.0f64, 20.0f64, 210_000.0, 0.3);
    let k = a * a / (b * b - a * a);
    ((1.0 + nu) * k / e * ((1.0 - 2.0 * nu) * r + b * b / r), k * (1.0 + b * b / (r * r)))
}

/// FieldCraft's FEM result for u_r at the bore under p = 100 (quad8, 8 × 16), frozen in
/// tests/oracle/lame-cylinder-plane-strain/fixture.json.
const FEM_UR_BORE: f64 = 0.009079340794556473;

fn lame_report(h: f64, material: Material, p: f64, boundary: Boundary) -> (f64, f64, f64) {
    let (ura, ur15, hoop) = lame_cylinder(h, material, p, boundary, 1.0);
    let (ex_a, _) = lame_exact(10.0);
    let (ex15, hoop15) = lame_exact(15.0);
    let e = ((ura - ex_a) / ex_a, (ur15 - ex15) / ex15, (hoop - hoop15) / hoop15);
    eprintln!(
        "Lamé h = {h} ({boundary:?}): u_r(a)/p {ura:.6e} (exact {ex_a:.6e}, FEM {:.6e}), u_r(15)/p {ur15:.6e} (exact {ex15:.6e}), \
         σθ(15)/p {hoop:.4} (exact {hoop15:.4}); rel. errors {:+.2e} {:+.2e} {:+.2e}",
        FEM_UR_BORE / 100.0,
        e.0,
        e.1,
        e.2
    );
    e
}

/// Ten nodes across the wall, neo-Hookean at p = 100 (strain 1e-3, the ghost fill's traction
/// linearised about its own gradient): the ghost-filled surface is within 3 %, the blurred one about
/// 11 % too stiff (its answer also depends on γ and κ, see below).
#[test]
#[ignore = "heavy: cargo xtask test --heavy (CI heavy job)"]
fn thick_cylinder_matches_lame_coarse() {
    let m = Material::neo_hookean(210_000.0, 0.3, 1.0).unwrap();
    let (ea, e15, eh) = lame_report(1.0, m, 100.0, Boundary::Ghost);
    assert!(ea.abs() < 0.03 && e15.abs() < 0.03 && eh.abs() < 0.03, "{ea} {e15} {eh}");
    let (ea, _, _) = lame_report(1.0, m, 100.0, Boundary::Blurred);
    assert!(ea.abs() < 0.15, "{ea}");
}

/// The static answer of the ghost-filled surface does not depend on the damping used to reach it;
/// the original blurred surface's does (its band nodes integrate a velocity they never act on,
/// and the artificial viscosity turns it into a surface stress).
#[test]
#[ignore = "heavy: cargo xtask test --heavy (CI heavy job)"]
fn the_static_answer_does_not_depend_on_the_damping() {
    let m = Material::linear_elastic(210_000.0, 0.3, 1.0).unwrap();
    let low = lame_cylinder(1.0, m, 0.1, Boundary::Ghost, 0.5).0;
    let high = lame_cylinder(1.0, m, 0.1, Boundary::Ghost, 4.0).0;
    eprintln!("ghost: u_r(a)/p {low:.6e} (γ × 0.5), {high:.6e} (γ × 4)");
    // Equal to the relaxation tolerance (the speed test leaves Δu ∝ v/γ).
    assert!(((low - high) / low).abs() < 1e-4, "{low} {high}");
    let low = lame_cylinder(1.0, m, 0.1, Boundary::Blurred, 0.5).0;
    let high = lame_cylinder(1.0, m, 0.1, Boundary::Blurred, 4.0).0;
    eprintln!("blurred: u_r(a)/p {low:.6e} (γ × 0.5), {high:.6e} (γ × 4)");
    assert!(((low - high) / low).abs() > 0.05, "{low} {high}");
}

/// The ghost-filled surface converges at second order or better: within 0.5 % at twenty nodes
/// across the wall (linear elastic at strain 1e-6, so geometric nonlinearity stays out).
#[test]
#[ignore = "heavy: cargo xtask test --heavy (CI heavy job)"]
fn thick_cylinder_converges_to_lame() {
    let m = Material::linear_elastic(210_000.0, 0.3, 1.0).unwrap();
    let coarse = lame_report(1.0, m, 0.1, Boundary::Ghost);
    let fine = lame_report(0.5, m, 0.1, Boundary::Ghost);
    assert!(fine.0.abs() < 0.005 && fine.1.abs() < 0.005 && fine.2.abs() < 0.006, "{fine:?}");
    assert!(fine.0.abs() < 0.25 * coarse.0.abs() && fine.1.abs() < 0.25 * coarse.1.abs(), "{coarse:?} → {fine:?}");
}

// ------------------------------------------- 6. slab under a body force (linear stress)

/// A slab |x| < L, periodic in y, under the body force ρ g sign(x) with free faces: σxx falls
/// linearly to zero at each face, u is quadratic, and the ghost-filled surface reproduces it to the
/// relaxation tolerance wherever the faces cut the grid; the blurred one is off by O(h)
/// (an effective face shift of 0.06–2.5 h, depending on where the face sits between nodes).
#[test]
fn a_slab_under_a_body_force_is_exact_with_the_ghost_fill() {
    let m = Material::linear_elastic(210_000.0, 0.3, 1.0).unwrap();
    let (l, mu, _) = m.constants();
    let modulus = l + 2.0 * mu;
    let h = 1.0;
    for offset in [0.0, 0.3, 0.7] {
        let big_l = 10.0 + offset;
        let g = 1e-6 * modulus / big_l;
        let grid = Grid::new(37, 5, h, [-18.0, 0.0], [false, true], &Limits::default()).unwrap();
        let c = m.p_wave_speed();
        let settings = Settings { boundary: Boundary::Ghost, viscosity: 0.0, damping: 0.5 * c / big_l, ..Settings::default() };
        let shape = Shape::Rectangle { min: [-big_l, -1e6], max: [big_l, 1e6] };
        let mut s = Solid::new(grid, m, shape, settings, Limits::default()).unwrap();
        s.set_body_force(move |x| {
            [
                if x[0] > 0.0 {
                    g
                } else if x[0] < 0.0 {
                    -g
                } else {
                    0.0
                },
                0.0,
            ]
        })
        .unwrap();
        let u_face = g * big_l * big_l / (2.0 * modulus);
        let r = s.relax(1e-7 * u_face * c / big_l, 200_000).unwrap();
        assert!(r.converged, "{r:?}");
        let mut worst: f64 = 0.0;
        for k in solid_nodes(&s) {
            let x = s.grid().position(k)[0];
            let exact = g / modulus * (big_l * x - x * x.abs() / 2.0);
            worst = worst.max((s.displacement(k).unwrap()[0] - exact).abs() / u_face);
        }
        eprintln!("slab, face {offset} h past a node: max |u error| / u(L) = {worst:.2e}");
        assert!(worst < 1e-5, "{worst}");
    }
}
