//! Accuracy study of the surface treatments of the Eulerian solid (blurred, blurred with the
//! momentum in the body only, the ghost fill with quadratic or cubic fits) against closed-form
//! elastic solutions with a stress gradient at a loaded surface. The numbers are summarised in
//! docs/research/reference-map.md §9.
//!
//! `cargo run --release -p fieldcraft-eulerian --example boundary_study -- <case> [out-dir] [h,h,…] [mode,mode,…]`
//! with case `bar` (slab under a body force), `lame` (thick cylinder), `kirsch` (hole in a
//! plate), `disc` (rotating annulus), `nh` (neo-Hookean cylinder at finite strain), `gamma`
//! (damping independence), `ghostcheck` (ghost and residual errors with the exact Lamé field)
//! or `probe` (step-by-step history); modes `blend`, `blend-sm` (momentum in the body only),
//! `ghost-quadratic`, `ghost`. Prints Markdown tables; writes CSV profiles to `out-dir` (`x`
//! for none). Environment: SEQ (one resolution after another, for timings), NH_P (pressure
//! of `nh`), RAMP (load increments of `nh`, 10), KAPPA / CFL (settings overrides), VERBOSE /
//! CHUNK (relaxation progress), GAMMA (damping factor of `probe`).
//!
//! The small-strain cases are linear elastic at strains of about 1e-6, so geometric
//! nonlinearity (O(strain) relative) stays far below the discretisation errors measured.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::needless_range_loop)]

use fieldcraft_eulerian::{Boundary, Grid, Limits, Material, NodeKind, Settings, Shape, Solid};
use std::fmt::Write as _;

const E: f64 = 210_000.0;
const NU: f64 = 0.3;

fn lame_constants() -> (f64, f64) {
    let mu = E / (2.0 * (1.0 + NU));
    (E * NU / ((1.0 + NU) * (1.0 - 2.0 * NU)), mu)
}

fn is_solid(s: &Solid, k: usize) -> bool {
    matches!(s.kind(k), Some(NodeKind::Solid | NodeKind::Fixed))
}

// ------------------------------------------------------------------------------- bar

/// A slab |x| < L, periodic in y (plane strain, εyy = 0), body force b = g sign(x) per unit
/// mass pulling both halves outwards, free faces. σxx = ρ g (L − |x|), so the stress falls
/// linearly to zero at each face with gradient ρ g; u = (ρ g / M)(L x − x|x|/2), M = λ + 2μ.
struct BarResult {
    /// Effective face position from a fit of u over the interior, minus L (h units).
    shift: f64,
    /// Relative error of σxx at the solid node nearest the face, in units of ρ g h.
    s_last: f64,
    /// Largest |σxx error| over the solid nodes, in units of ρ g h.
    s_max: f64,
    /// Relative error of u at the face (from the fit) vs the exact u(L).
    u_face: f64,
}

fn bar(h: f64, half_nodes: f64, offset: f64, settings: Settings, csv: Option<&mut String>) -> Option<BarResult> {
    let (l_, mu) = lame_constants();
    let m_mod = l_ + 2.0 * mu;
    let rho = 1.0;
    // The face sits `offset` grid spacings past a node.
    let big_l = (half_nodes + offset) * h;
    // Strain ~ 1e-6 at the centre.
    let g = 1e-6 * m_mod / (rho * big_l);
    let n_side = (half_nodes as usize) + 8;
    let nx = 2 * n_side + 1;
    let origin = [-(n_side as f64) * h, 0.0];
    let grid = Grid::new(nx, 5, h, origin, [false, true], &Limits::default()).unwrap();
    let mat = Material::linear_elastic(E, NU, rho).unwrap();
    let c = mat.p_wave_speed();
    let settings = Settings { damping: 0.5 * c / big_l, ..settings };
    let shape = Shape::Rectangle { min: [-big_l, -1e6], max: [big_l, 1e6] };
    let mut s = Solid::new(grid, mat, shape, settings, Limits::default()).unwrap();
    s.set_body_force(move |x| [g * x[0].signum() * (x[0] != 0.0) as u8 as f64, 0.0]).unwrap();
    let u_scale = rho * g * big_l * big_l / (2.0 * m_mod);
    let r = s.relax(1e-7 * u_scale * c / big_l, 200_000);
    match r {
        Ok(r) if r.converged => {}
        other => {
            println!("<!-- not at rest: {other:?} -->");
            return None;
        }
    }
    let exact_u = |x: f64| rho * g / m_mod * (big_l * x - x * x.abs() / 2.0);
    let exact_s = |x: f64| rho * g * (big_l - x.abs());
    // Fit L_eff over the interior: M u/(ρ g) + x²/2 = L_eff x for x > 0.
    let (mut sxx, mut sxy) = (0.0, 0.0);
    let (mut s_last, mut x_last, mut s_max) = (0.0, f64::NEG_INFINITY, 0.0f64);
    let j = 2;
    let mut out = String::new();
    for i in 0..nx {
        let k = s.grid().index(i, j);
        if !is_solid(&s, k) {
            continue;
        }
        let x = s.grid().position(k)[0];
        let u = s.displacement(k).unwrap()[0];
        let st = s.stress(k).unwrap()[0];
        let _ = writeln!(out, "{x},{u},{},{st},{}", exact_u(x), exact_s(x));
        if x > 0.0 && x < big_l - 3.0 * h {
            sxx += x * x;
            sxy += x * (m_mod * u / (rho * g) + x * x / 2.0);
        }
        let err = (st - exact_s(x)) / (rho * g * h);
        // The centre node sits on the kink of the body force (a post-processing artefact of
        // the central difference there), so the maximum is over |x| > 1.5 h.
        if x.abs() > 1.5 * h {
            s_max = s_max.max(err.abs());
        }
        if x > x_last {
            x_last = x;
            s_last = err;
        }
    }
    if let Some(c) = csv {
        c.push_str("x,u,u_exact,sxx,sxx_exact\n");
        c.push_str(&out);
    }
    let l_eff = sxy / sxx;
    let u_face_fit = rho * g / m_mod * (l_eff * big_l - big_l * big_l / 2.0);
    Some(BarResult { shift: (l_eff - big_l) / h, s_last, s_max, u_face: (u_face_fit - exact_u(big_l)) / exact_u(big_l) })
}

/// A surface treatment and, for the ghost fill, the highest degree of its fit.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Mode {
    surface: Boundary,
    degree: usize,
}

impl Mode {
    const BLEND: Mode = Mode { surface: Boundary::Blurred, degree: 3 };
    const BLEND_SM: Mode = Mode { surface: Boundary::BlurredBody, degree: 3 };
    const GHOST: Mode = Mode { surface: Boundary::Ghost, degree: 3 };
    const GHOST2: Mode = Mode { surface: Boundary::Ghost, degree: 2 };
}

fn statics(surface: Mode, blur: f64) -> Settings {
    // No artificial viscosity: it only damps dynamics, and with the blurred surface the band
    // nodes (momentum integrated, ζ extrapolated) never come to rest and drag the surface.
    Settings {
        blur,
        band_layers: ((blur + 2.0).ceil() as usize + 1).max(4),
        viscosity: 0.0,
        boundary: surface.surface,
        ghost_degree: surface.degree,
        ..Settings::default()
    }
}

fn name(surface: Mode) -> &'static str {
    match (surface.surface, surface.degree) {
        (Boundary::Blurred, _) => "blend",
        (Boundary::BlurredBody, _) => "blend-sm",
        (Boundary::Ghost, 2) => "ghost-quadratic",
        (Boundary::Ghost, _) => "ghost",
        (Boundary::Mapped, _) => "mapped",
    }
}

fn bar_study(out: Option<&str>) {
    println!("## bar: slab |x| < L = (10 + θ) h, body force ρ g sign(x), free faces (dσ/dx = ρ g at the face)");
    println!();
    println!("| mode | ε/h | θ | face shift δ/h | σxx err at last node / (ρ g h) | max σxx err / (ρ g h) | u(L) rel. err |");
    println!("|---|---|---|---|---|---|---|");
    let mut runs: Vec<(Mode, f64)> = [1.0, 1.5, 2.5].iter().map(|&b| (Mode::BLEND, b)).collect();
    runs.extend([(Mode::BLEND_SM, 1.5), (Mode::GHOST2, 1.5), (Mode::GHOST, 1.5)]);
    for (surface, blur) in runs {
        for offset in [0.0, 0.25, 0.5, 0.75] {
            let mut csv = String::new();
            let Some(r) = bar(1.0, 10.0, offset, statics(surface, blur), Some(&mut csv)) else {
                println!("| {} | – | {offset} | did not come to rest | | | |", name(surface));
                continue;
            };
            let eps = if matches!(surface.surface, Boundary::Ghost) { "–".to_string() } else { format!("{blur}") };
            println!("| {} | {eps} | {offset} | {:+.3e} | {:+.3e} | {:.3e} | {:+.3e} |", name(surface), r.shift, r.s_last, r.s_max, r.u_face);
            if let Some(dir) = out {
                std::fs::write(format!("{dir}/bar_{}_eps{blur}_off{offset}.csv", name(surface)), csv).unwrap();
            }
        }
    }
    println!();
    println!("Resolution at L ≈ 10 (θ = 0.5, ε = 1.5 h):");
    println!();
    println!("| mode | h | face shift δ (length) | δ/h | u(L) rel. err |");
    println!("|---|---|---|---|---|");
    for surface in [Mode::BLEND, Mode::BLEND_SM, Mode::GHOST2, Mode::GHOST] {
        for h in [1.0, 0.5, 0.25] {
            let Some(r) = bar(h, (10.0 / h).round(), 0.5, statics(surface, 1.5), None) else {
                println!("| {} | {h} | did not come to rest | | |", name(surface));
                continue;
            };
            println!("| {} | {h} | {:+.4e} | {:+.4e} | {:+.3e} |", name(surface), r.shift * h, r.shift, r.u_face);
        }
    }
}

// ------------------------------------------------------------------------- annulus cases

/// A centred square grid covering |x|, |y| ≤ `half` + 6 h, with a node at the origin.
fn centred_grid(half: f64, h: f64) -> Grid {
    let m = ((half / h).ceil() as usize) + 6;
    let n = 2 * m + 1;
    let o = -(m as f64) * h;
    Grid::new(n, n, h, [o, o], [false; 2], &Limits::default()).unwrap()
}

/// Stress components in polar form at x: (σrr, σθθ, σrθ).
fn polar(st: [f64; 3], x: [f64; 2]) -> (f64, f64, f64) {
    let r = x[0].hypot(x[1]);
    let (c, s) = (x[0] / r, x[1] / r);
    let srr = st[0] * c * c + st[1] * s * s + 2.0 * st[2] * s * c;
    let stt = st[0] * s * s + st[1] * c * c - 2.0 * st[2] * s * c;
    let srt = (st[1] - st[0]) * s * c + st[2] * (c * c - s * s);
    (srr, stt, srt)
}

/// Cartesian (σxx, σyy, σxy) from polar components at x.
fn cartesian(srr: f64, stt: f64, srt: f64, x: [f64; 2]) -> [f64; 3] {
    let r = x[0].hypot(x[1]);
    let (c, s) = (x[0] / r, x[1] / r);
    [srr * c * c + stt * s * s - 2.0 * srt * s * c, srr * s * s + stt * c * c + 2.0 * srt * s * c, (srr - stt) * s * c + srt * (c * c - s * s)]
}

/// Relaxes a free annulus `inner` < r < `outer` with the ambient stress and body force given,
/// returning the solid.
#[allow(clippy::too_many_arguments)]
fn relax_annulus(
    h: f64,
    inner: f64,
    outer: f64,
    material: Material,
    settings: Settings,
    gamma: f64,
    ambient: &dyn Fn([f64; 2]) -> [f64; 3],
    body: impl Fn([f64; 2]) -> [f64; 2] + Send + Sync + 'static,
    u_scale: f64,
) -> (Solid, usize) {
    relax_annulus_ramped(h, inner, outer, material, settings, gamma, ambient, body, u_scale, 0)
}

/// As [`relax_annulus`], with the ambient stress applied in `ramp` equal increments of
/// 4 b / c_P each before the final relaxation (quasi-static loading for finite strain).
#[allow(clippy::too_many_arguments)]
fn relax_annulus_ramped(
    h: f64,
    inner: f64,
    outer: f64,
    material: Material,
    settings: Settings,
    gamma: f64,
    ambient: &dyn Fn([f64; 2]) -> [f64; 3],
    body: impl Fn([f64; 2]) -> [f64; 2] + Send + Sync + 'static,
    u_scale: f64,
    ramp: usize,
) -> (Solid, usize) {
    let grid = centred_grid(outer, h);
    let c = material.p_wave_speed();
    let settings = Settings { damping: gamma * 0.5 * c / outer, ..settings };
    let mut s = Solid::new(grid, material, Shape::Annulus { center: [0.0, 0.0], inner, outer }, settings, Limits::default()).unwrap();
    let mut ramp_steps = 0;
    if ramp > 0 {
        let per = (4.0 * outer / c / s.dt()).ceil() as usize;
        for i in 1..=ramp {
            let f = i as f64 / ramp as f64;
            s.set_ambient(|x| ambient(x).map(|v| f * v)).unwrap();
            s.run(per).unwrap();
            ramp_steps += per;
        }
    }
    s.set_ambient(ambient).unwrap();
    s.set_body_force(body).unwrap();
    let tol = 1e-6 * u_scale * c / outer;
    if std::env::var("VERBOSE").is_ok() {
        let mut steps = 0;
        for _ in 0..200 {
            let r = s.relax(tol, std::env::var("CHUNK").ok().and_then(|v| v.parse().ok()).unwrap_or(1000)).unwrap();
            steps += r.steps;
            if steps == r.steps {
                steps += ramp_steps;
            }
            let body = (0..s.grid().len()).filter(|&k| is_solid(&s, k)).count();
            eprintln!("h {h}: {steps} steps, speed {:.3e} (tol {tol:.1e}), fallbacks {}, body nodes {body}", r.max_speed, s.surface_fallbacks());
            if r.converged {
                return (s, steps);
            }
        }
        panic!("not converged");
    }
    let r = s.relax(tol, 3_000_000).unwrap();
    assert!(r.converged, "{r:?}");
    (s, r.steps + ramp_steps)
}

/// Fit of u_r = A r + B/r (+ C r³ when `cubic`) over solid nodes with φ < −2h.
fn fit_ur(s: &Solid, h: f64, cubic: bool) -> Vec<f64> {
    let nb = if cubic { 3 } else { 2 };
    let mut m = vec![vec![0.0; nb + 1]; nb];
    for k in 0..s.grid().len() {
        if !is_solid(s, k) || s.phi(k).unwrap() >= -2.0 * h {
            continue;
        }
        let x = s.grid().position(k);
        let r = x[0].hypot(x[1]);
        let u = s.displacement(k).unwrap();
        let ur = (u[0] * x[0] + u[1] * x[1]) / r;
        let f = [r, 1.0 / r, r * r * r];
        for i in 0..nb {
            for j in 0..nb {
                m[i][j] += f[i] * f[j];
            }
            m[i][nb] += f[i] * ur;
        }
    }
    solve_dense(m)
}

fn solve_dense(mut a: Vec<Vec<f64>>) -> Vec<f64> {
    let n = a.len();
    for c in 0..n {
        let p = (c..n).max_by(|&x, &y| a[x][c].abs().total_cmp(&a[y][c].abs())).unwrap();
        a.swap(c, p);
        for r in c + 1..n {
            let f = a[r][c] / a[c][c];
            for k in c..=n {
                a[r][k] -= f * a[c][k];
            }
        }
    }
    let mut x = vec![0.0; n];
    for r in (0..n).rev() {
        let mut v = a[r][n];
        for k in r + 1..n {
            v -= a[r][k] * x[k];
        }
        x[r] = v / a[r][r];
    }
    x
}

/// Nodal stress (σrr, σθθ) at the solid nodes along the ray from the origin at angle `theta`
/// (0 or π/2: grid lines), extrapolated to radius `r0` by a quadratic through the three solid
/// nodes nearest r0 on the side `dir` (+1 outwards from r0).
fn surface_value(s: &Solid, axis: usize, r0: f64, dir: f64) -> (f64, f64) {
    let g = s.grid();
    let mut pts: Vec<(f64, f64, f64)> = Vec::new();
    for k in 0..g.len() {
        let x = g.position(k);
        if !is_solid(s, k) || x[1 - axis].abs() > 1e-9 || x[axis] <= 0.0 {
            continue;
        }
        let r = x[axis];
        if (r - r0) * dir <= 0.0 {
            continue;
        }
        let st = s.stress(k).unwrap();
        let (srr, stt, _) = polar([st[0], st[1], st[2]], x);
        pts.push((r, srr, stt));
    }
    pts.sort_by(|a, b| (a.0 - r0).abs().total_cmp(&(b.0 - r0).abs()));
    pts.truncate(3);
    let lag = |i: usize| {
        let mut w = 1.0;
        for j in 0..3 {
            if j != i {
                w *= (r0 - pts[j].0) / (pts[i].0 - pts[j].0);
            }
        }
        w
    };
    let mut out = (0.0, 0.0);
    for i in 0..3 {
        out.0 += lag(i) * pts[i].1;
        out.1 += lag(i) * pts[i].2;
    }
    out
}

/// Largest |σ − σ_exact| (Cartesian components) over the solid nodes, and the same over the
/// nodes with φ < −2h.
fn stress_max_error(s: &Solid, h: f64, exact: &dyn Fn([f64; 2]) -> [f64; 3]) -> (f64, f64) {
    let (mut all, mut inner) = (0.0f64, 0.0f64);
    for k in 0..s.grid().len() {
        if !is_solid(s, k) {
            continue;
        }
        let x = s.grid().position(k);
        let st = s.stress(k).unwrap();
        let e = exact(x);
        let err = (0..3).fold(0.0f64, |m, c| m.max((st[c] - e[c]).abs()));
        all = all.max(err);
        if s.phi(k).unwrap() < -2.0 * h {
            inner = inner.max(err);
        }
    }
    (all, inner)
}

fn order(prev: Option<f64>, now: f64, ratio: f64) -> String {
    match prev {
        Some(p) if p != 0.0 && now != 0.0 => format!("{:.2}", (p.abs() / now.abs()).ln() / ratio.ln()),
        _ => "–".into(),
    }
}

// --------------------------------------------------------------------------------- Lamé

const LA: f64 = 10.0;
const LB: f64 = 20.0;
/// Small pressure (strain ~ 1e-6); errors are reported relative.
const LP: f64 = 0.1;

fn lame_exact(r: f64) -> (f64, f64, f64) {
    let k = LP * LA * LA / (LB * LB - LA * LA);
    let ur = (1.0 + NU) * k / E * ((1.0 - 2.0 * NU) * r + LB * LB / r);
    (ur, k * (1.0 - LB * LB / (r * r)), k * (1.0 + LB * LB / (r * r)))
}

struct LameRow {
    h: f64,
    nodes: usize,
    steps: usize,
    ura: f64,
    hoop15: f64,
    srr_a: f64,
    stt_a: f64,
    smax: f64,
    smax_in: f64,
    fallbacks: usize,
    wall: f64,
}

/// The settings of the annulus cases: the committed default κ = 0.1 for every surface. The
/// original blurred surface's static answer depends on it (and on γ); the others do not, but
/// the ghost fill needs it for stability (without it a slowly growing boundary mode
/// appears).
fn lame_settings(surface: Mode) -> Settings {
    // KAPPA and CFL override the defaults for stability experiments.
    let env = |k: &str, d: f64| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
    Settings { viscosity: env("KAPPA", 0.1), cfl: env("CFL", 0.4), ..statics(surface, 1.5) }
}

fn lame(h: f64, surface: Mode, gamma: f64, material: Material, p_scale: f64, out: Option<&str>) -> LameRow {
    let p = LP * p_scale;
    let mid = 0.5 * (LA + LB);
    let amb = move |x: [f64; 2]| if x[0].hypot(x[1]) < mid { [-p, -p, 0.0] } else { [0.0; 3] };
    let t0 = std::time::Instant::now();
    let (s, steps) = relax_annulus(h, LA, LB, material, lame_settings(surface), gamma, &amb, |_| [0.0; 2], lame_exact(LA).0 * p_scale);
    let wall = t0.elapsed().as_secs_f64();
    let c = fit_ur(&s, h, false);
    let ura = (c[0] * LA + c[1] / LA) / p_scale;
    let (mut hoop, mut nh) = (0.0, 0.0);
    for k in 0..s.grid().len() {
        let x = s.grid().position(k);
        let r = x[0].hypot(x[1]);
        if is_solid(&s, k) && (r - 15.0).abs() < 0.5 * h {
            let st = s.stress(k).unwrap();
            hoop += polar([st[0], st[1], st[2]], x).1;
            nh += 1.0;
        }
    }
    let (srr_a, stt_a) = surface_value(&s, 0, LA, 1.0);
    let exact = |x: [f64; 2]| {
        let (_, srr, stt) = lame_exact(x[0].hypot(x[1]));
        let e = cartesian(srr, stt, 0.0, x);
        [e[0] * p_scale, e[1] * p_scale, e[2] * p_scale]
    };
    let (smax, smax_in) = stress_max_error(&s, h, &exact);
    if let Some(dir) = out {
        let mut csv = String::from("x,srr,stt,srr_exact,stt_exact,ur,ur_exact\n");
        let g = s.grid();
        for k in 0..g.len() {
            let x = g.position(k);
            if x[1].abs() < 1e-9 && x[0] > 0.0 && is_solid(&s, k) {
                let st = s.stress(k).unwrap();
                let (_, ex_rr, ex_tt) = lame_exact(x[0]);
                let u = s.displacement(k).unwrap()[0] / p_scale;
                let _ = writeln!(csv, "{},{},{},{ex_rr},{ex_tt},{u},{}", x[0], st[0] / p_scale, st[1] / p_scale, lame_exact(x[0]).0);
            }
        }
        std::fs::write(format!("{dir}/lame_{}_h{h}.csv", name(surface)), csv).unwrap();
    }
    LameRow {
        h,
        nodes: s.grid().len(),
        steps,
        ura,
        hoop15: hoop / nh / p_scale,
        srr_a: srr_a / p_scale,
        stt_a: stt_a / p_scale,
        smax: smax / p_scale,
        smax_in: smax_in / p_scale,
        fallbacks: s.surface_fallbacks(),
        wall,
    }
}

/// Runs `f` over `hs` on one thread each, or one after the other with SEQ set (for timings).
fn over<T: Send>(hs: &[f64], f: impl Fn(f64) -> T + Sync) -> Vec<T> {
    if std::env::var("SEQ").is_ok() {
        return hs.iter().map(|&h| f(h)).collect();
    }
    std::thread::scope(|sc| {
        let f = &f;
        let handles: Vec<_> = hs.iter().map(|&h| sc.spawn(move || f(h))).collect();
        handles.into_iter().map(|t| t.join().unwrap()).collect()
    })
}

fn lame_study(hs: &[f64], surfaces: &[Mode], out: Option<&str>) {
    let mat = Material::linear_elastic(E, NU, 1.0).unwrap();
    let (ura_x, _, _) = lame_exact(LA);
    let (_, _, hoop_x) = lame_exact(15.0);
    let (_, srr_x, stt_x) = lame_exact(LA);
    for &surface in surfaces {
        let rows: Vec<LameRow> = over(hs, |h| lame(h, surface, 1.0, mat, 1.0, out));
        println!("### Lamé, {} (a = 10, b = 20, linear elastic, errors relative; σ errors / p)", name(surface));
        println!();
        println!(
            "| h | nodes | steps | wall s | ms/step | u_r(a) err | order | σθ(15) err | order | σrr(a) / p (exact −1) | σθ(a) err | max σ err / p | order | max σ err / p, φ < −2h | fallbacks |"
        );
        println!("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|");
        let mut prev: Option<(f64, f64, f64)> = None;
        for r in &rows {
            let eu = (r.ura - ura_x) / ura_x;
            let eh = (r.hoop15 - hoop_x) / hoop_x;
            let et = (r.stt_a - stt_x) / stt_x;
            println!(
                "| {} | {} | {} | {:.1} | {:.2} | {eu:+.3e} | {} | {eh:+.3e} | {} | {:+.4} | {et:+.3e} | {:.3e} | {} | {:.3e} | {} |",
                r.h,
                r.nodes,
                r.steps,
                r.wall,
                1e3 * r.wall / r.steps as f64,
                order(prev.map(|p| p.0), eu, 2.0),
                order(prev.map(|p| p.1), eh, 2.0),
                r.srr_a / LP,
                r.smax / LP,
                order(prev.map(|p| p.2), r.smax, 2.0),
                r.smax_in / LP,
                r.fallbacks
            );
            prev = Some((eu, eh, r.smax));
        }
        let _ = srr_x;
        println!();
    }
}

// -------------------------------------------------------------------------------- Kirsch

const KA: f64 = 5.0;
const KR: f64 = 20.0;
const KS: f64 = 0.1;

/// The Kirsch field around a hole of radius a in an infinite plate under σxx = S: (σrr, σθθ, σrθ).
fn kirsch(x: [f64; 2]) -> (f64, f64, f64) {
    let r = x[0].hypot(x[1]);
    let t = x[1].atan2(x[0]);
    let q = KA * KA / (r * r);
    let (c2, s2) = ((2.0 * t).cos(), (2.0 * t).sin());
    let srr = 0.5 * KS * (1.0 - q) + 0.5 * KS * (1.0 - 4.0 * q + 3.0 * q * q) * c2;
    let stt = 0.5 * KS * (1.0 + q) - 0.5 * KS * (1.0 + 3.0 * q * q) * c2;
    let srt = -0.5 * KS * (1.0 + 2.0 * q - 3.0 * q * q) * s2;
    (srr, stt, srt)
}

fn kirsch_xy(x: [f64; 2]) -> [f64; 3] {
    let (a, b, c) = kirsch(x);
    cartesian(a, b, c, x)
}

fn kirsch_case(h: f64, surface: Mode, out: Option<&str>) -> String {
    let mat = Material::linear_elastic(E, NU, 1.0).unwrap();
    let mid = 0.5 * (KA + KR);
    let amb = move |x: [f64; 2]| if x[0].hypot(x[1]) < mid { [0.0; 3] } else { kirsch_xy(x) };
    let u_scale = KS * KR / E;
    let t0 = std::time::Instant::now();
    let (s, steps) = relax_annulus(h, KA, KR, mat, lame_settings(surface), 1.0, &amb, |_| [0.0; 2], u_scale);
    let wall = t0.elapsed().as_secs_f64();
    let (_, t90) = surface_value(&s, 1, KA, 1.0);
    let (_, t0) = surface_value(&s, 0, KA, 1.0);
    let (smax, smax_in) = stress_max_error(&s, h, &kirsch_xy);
    if let Some(dir) = out {
        let mut csv = String::from("x,y,r,theta,stt,stt_exact\n");
        for k in 0..s.grid().len() {
            let x = s.grid().position(k);
            let r = x[0].hypot(x[1]);
            if is_solid(&s, k) && r < KA + 3.0 * h + 1e-9 {
                let st = s.stress(k).unwrap();
                let (_, stt, _) = polar([st[0], st[1], st[2]], x);
                let _ = writeln!(csv, "{},{},{r},{},{},{}", x[0], x[1], x[1].atan2(x[0]), stt / KS, kirsch(x).1 / KS);
            }
        }
        std::fs::write(format!("{dir}/kirsch_{}_h{h}.csv", name(surface)), csv).unwrap();
    }
    format!(
        "| {} | {h} | {} | {steps} | {wall:.1} | {:.2} | {:.4} | {:+.3e} | {:.4} | {:.3e} | {:.3e} | {} |",
        name(surface),
        s.grid().len(),
        1e3 * wall / steps as f64,
        t90 / KS,
        (t90 - 3.0 * KS) / (3.0 * KS),
        t0 / KS,
        smax / KS,
        smax_in / KS,
        s.surface_fallbacks()
    )
}

fn kirsch_study(hs: &[f64], surfaces: &[Mode], out: Option<&str>) {
    println!("### Kirsch: hole a = 5 in a disc R = 20 loaded by the Kirsch traction (σxx = S at infinity)");
    println!();
    println!(
        "| mode | h | nodes | steps | wall s | ms/step | σθ(a, 90°)/S (exact 3) | err | σθ(a, 0°)/S (exact −1) | max σ err / S | max σ err / S, φ < −2h | fallbacks |"
    );
    println!("|---|---|---|---|---|---|---|---|---|---|---|---|");
    for &surface in surfaces {
        let rows: Vec<String> = over(hs, |h| kirsch_case(h, surface, out));
        for r in rows {
            println!("{r}");
        }
    }
    println!();
}

// ------------------------------------------------------------------------ rotating disc

/// ω² with ρ = 1, giving strains of about 1e-6.
fn disc_w2() -> f64 {
    let (l, m) = lame_constants();
    1e-6 * (l + 2.0 * m) / (LB * LB)
}

/// Plane-strain rotating annulus a < r < b, free surfaces: u = C₁ r + C₂/r − K r³,
/// K = ρω²/(8M). Returns (u_r, σrr, σθθ) at r.
fn disc_exact(r: f64) -> (f64, f64, f64) {
    let (l, m) = lame_constants();
    let k = disc_w2() / (8.0 * (l + 2.0 * m));
    // σrr = 2(λ+μ) C₁ − 2μ C₂/r² − (4λ + 6μ) K r²; σθθ = 2(λ+μ) C₁ + 2μ C₂/r² − (4λ + 2μ) K r².
    let srr = |c1: f64, c2: f64, r: f64| 2.0 * (l + m) * c1 - 2.0 * m * c2 / (r * r) - (4.0 * l + 6.0 * m) * k * r * r;
    let a = [[2.0 * (l + m), -2.0 * m / (LA * LA)], [2.0 * (l + m), -2.0 * m / (LB * LB)]];
    let rhs = [(4.0 * l + 6.0 * m) * k * LA * LA, (4.0 * l + 6.0 * m) * k * LB * LB];
    let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    let c1 = (rhs[0] * a[1][1] - a[0][1] * rhs[1]) / det;
    let c2 = (a[0][0] * rhs[1] - a[1][0] * rhs[0]) / det;
    let _ = srr(c1, c2, r);
    (c1 * r + c2 / r - k * r * r * r, srr(c1, c2, r), 2.0 * (l + m) * c1 + 2.0 * m * c2 / (r * r) - (4.0 * l + 2.0 * m) * k * r * r)
}

fn disc_case(h: f64, surface: Mode) -> (String, [f64; 3]) {
    let mat = Material::linear_elastic(E, NU, 1.0).unwrap();
    let w2 = disc_w2();
    let (ura_x, _, stt_a_x) = disc_exact(LA);
    let (s, steps) = relax_annulus(h, LA, LB, mat, lame_settings(surface), 1.0, &|_| [0.0; 3], move |x| [w2 * x[0], w2 * x[1]], ura_x);
    let c = fit_ur(&s, h, true);
    let ura = c[0] * LA + c[1] / LA + c[2] * LA * LA * LA;
    let (_, stt_a) = surface_value(&s, 0, LA, 1.0);
    let exact = |x: [f64; 2]| {
        let (_, a, b) = disc_exact(x[0].hypot(x[1]));
        cartesian(a, b, 0.0, x)
    };
    let (smax, smax_in) = stress_max_error(&s, h, &exact);
    let eu = (ura - ura_x) / ura_x;
    let et = (stt_a - stt_a_x) / stt_a_x;
    let sc = stt_a_x;
    (
        format!(
            "| {} | {h} | {} | {steps} | {eu:+.3e} | {et:+.3e} | {:.3e} | {:.3e} | {} |",
            name(surface),
            s.grid().len(),
            smax / sc,
            smax_in / sc,
            s.surface_fallbacks()
        ),
        [eu, et, smax / sc],
    )
}

fn disc_study(hs: &[f64], surfaces: &[Mode]) {
    println!("### Rotating annulus (a = 10, b = 20, body force ω² x, free surfaces; σ errors / σθ(a))");
    println!();
    println!("| mode | h | nodes | steps | u_r(a) err | σθ(a) err | max σ err | max σ err, φ < −2h | fallbacks |");
    println!("|---|---|---|---|---|---|---|---|---|");
    for &surface in surfaces {
        let rows: Vec<(String, [f64; 3])> = std::thread::scope(|sc| {
            let hs: Vec<_> = hs.iter().map(|&h| sc.spawn(move || disc_case(h, surface))).collect();
            hs.into_iter().map(|t| t.join().unwrap()).collect()
        });
        let mut prev: Option<[f64; 3]> = None;
        for (r, e) in rows {
            let ord = match prev {
                Some(p) => {
                    format!(" (orders {} / {} / {})", order(Some(p[0]), e[0], 2.0), order(Some(p[1]), e[1], 2.0), order(Some(p[2]), e[2], 2.0))
                }
                None => String::new(),
            };
            println!("{r}{ord}");
            prev = Some(e);
        }
    }
    println!();
}

// ------------------------------------------------------- neo-Hookean thick cylinder

/// Bore pressure of the finite-strain cylinder: p/E = 0.038, so the linear solution puts the
/// bore at u_r(a)/a = 7.3 % (hoop strain 7 %): clearly nonlinear, far from inversion.
const NH_P_DEFAULT: f64 = 8000.0;

fn nh_p() -> f64 {
    std::env::var("NH_P").ok().and_then(|v| v.parse().ok()).unwrap_or(NH_P_DEFAULT)
}

/// The deformed position x* of the material at reference radius `r0` along the four grid
/// axes (averaged): ξ along each half-axis by a quadratic through the three body nodes nearest
/// the surface, solved for ξ = r0.
fn deformed_radius(s: &Solid, r0: f64) -> f64 {
    let g = s.grid();
    let mut total = 0.0;
    for (axis, sign) in [(0usize, 1.0f64), (0, -1.0), (1, 1.0), (1, -1.0)] {
        let mut pts: Vec<(f64, f64)> = Vec::new();
        for k in 0..g.len() {
            let x = g.position(k);
            if !is_solid(s, k) || x[1 - axis].abs() > 1e-9 || x[axis] * sign <= 0.0 {
                continue;
            }
            let xi = s.reference(k).unwrap();
            pts.push((x[axis] * sign, xi[axis] * sign));
        }
        pts.sort_by(|a, b| (a.1 - r0).abs().total_cmp(&(b.1 - r0).abs()));
        pts.truncate(3);
        let q = |x: f64| {
            let mut v = 0.0;
            for i in 0..3 {
                let mut w = 1.0;
                for j in 0..3 {
                    if j != i {
                        w *= (x - pts[j].0) / (pts[i].0 - pts[j].0);
                    }
                }
                v += w * pts[i].1;
            }
            v
        };
        // Newton on ξ(x) = r0 from the nearest node.
        let mut x = pts[0].0;
        for _ in 0..30 {
            let d = 1e-6;
            let slope = (q(x + d) - q(x - d)) / (2.0 * d);
            x -= (q(x) - r0) / slope;
        }
        total += x;
    }
    total / 4.0
}

fn nh_case(h: f64, surface: Mode) -> (String, [f64; 4]) {
    let mat = Material::neo_hookean_decoupled(E, NU, 1.0).unwrap();
    let mid = 0.5 * (LA + LB);
    let amb = move |x: [f64; 2]| if x[0].hypot(x[1]) < mid * 1.05 { [-nh_p(), -nh_p(), 0.0] } else { [0.0; 3] };
    let t0 = std::time::Instant::now();
    let ramp: usize = std::env::var("RAMP").ok().and_then(|v| v.parse().ok()).unwrap_or(10);
    let (s, steps) = relax_annulus_ramped(h, LA, LB, mat, lame_settings(surface), 1.0, &amb, |_| [0.0; 2], 0.7, ramp);
    let wall = t0.elapsed().as_secs_f64();
    let xa = deformed_radius(&s, LA);
    let ura = xa - LA;
    let (srr_a, stt_a) = surface_value(&s, 0, xa, 1.0);
    let (mut hoop, mut nh) = (0.0, 0.0);
    for k in 0..s.grid().len() {
        let x = s.grid().position(k);
        let r = x[0].hypot(x[1]);
        if is_solid(&s, k) && (r - 15.0).abs() < 0.5 * h {
            let st = s.stress(k).unwrap();
            hoop += polar([st[0], st[1], st[2]], x).1;
            nh += 1.0;
        }
    }
    let hoop = hoop / nh;
    (
        format!(
            "| {} | {h} | {} | {steps} | {:.1} | {ura:.6} | {:.5} | {hoop:.2} | {:.1} | {:.1} | {} |",
            name(surface),
            s.grid().len(),
            wall,
            ura / LA,
            srr_a,
            stt_a,
            s.surface_fallbacks()
        ),
        [ura, hoop, srr_a, stt_a],
    )
}

fn nh_study(hs: &[f64], surfaces: &[Mode]) {
    println!("### Neo-Hookean (decoupled) thick cylinder a = 10, b = 20, E = 210000, ν = 0.3, bore pressure p = {}", nh_p());
    let (lin_u, _, lin_t) = lame_exact(LA);
    let (_, _, lin_t15) = lame_exact(15.0);
    let sc = nh_p() / LP;
    println!();
    println!(
        "Linear Lamé scaled to p: u_r(a) = {:.6} (u_r/a = {:.4}), σθ(15) = {:.2}, σθ(a) = {:.1}",
        lin_u * sc,
        lin_u * sc / LA,
        lin_t15 * sc,
        lin_t * sc
    );
    println!();
    println!("| mode | h | nodes | steps | wall s | u_r(a) | u_r(a)/a | σθ(r = 15) | σrr at the bore | σθ at the bore | fallbacks |");
    println!("|---|---|---|---|---|---|---|---|---|---|---|");
    for &surface in surfaces {
        // A run that fails (the blurred surface folds at this load) prints its error instead.
        let rows: Vec<(String, [f64; 4])> = over(hs, |h| {
            std::thread::scope(|sc| sc.spawn(|| nh_case(h, surface)).join())
                .unwrap_or_else(|_| (format!("| {} | {h} | – | – | – | failed (see stderr) | | | | | |", name(surface)), [f64::NAN; 4]))
        });
        for (r, _) in &rows {
            println!("{r}");
        }
        // Richardson from the last three: order p from the ratio of differences, then the limit.
        if rows.len() >= 3 {
            let n = rows.len();
            for (q, label) in [(0usize, "u_r(a)"), (1, "σθ(15)"), (3, "σθ(bore)")] {
                let (f1, f2, f3) = (rows[n - 3].1[q], rows[n - 2].1[q], rows[n - 1].1[q]);
                let ord = ((f1 - f2) / (f2 - f3)).abs().ln() / 2f64.ln();
                let lim = f3 + (f3 - f2) / (2f64.powf(ord) - 1.0);
                println!("<!-- {} {label}: observed order {ord:.2}, Richardson limit {lim:.6} -->", name(surface));
            }
        }
    }
    println!();
}

fn parse_surface(s: &str) -> Mode {
    match s {
        "blend" => Mode::BLEND,
        "blend-sm" => Mode::BLEND_SM,
        "ghost-quadratic" => Mode::GHOST2,
        _ => Mode::GHOST,
    }
}

/// The static answer must not depend on the damping used to reach it.
fn gamma_study() {
    let mat = Material::linear_elastic(E, NU, 1.0).unwrap();
    let (ura_x, _, _) = lame_exact(LA);
    println!("### Lamé u_r(a) error vs the relaxation damping γ (× the default 0.5 c/b)");
    println!();
    println!("| mode | h | γ × 0.5 | γ × 1 | γ × 2 | γ × 4 |");
    println!("|---|---|---|---|---|---|");
    for surface in [Mode::BLEND, Mode::BLEND_SM, Mode::GHOST2, Mode::GHOST] {
        for h in [1.0, 0.5] {
            let errs: Vec<f64> = std::thread::scope(|sc| {
                let hs: Vec<_> = [0.5, 1.0, 2.0, 4.0].iter().map(|&g| sc.spawn(move || lame(h, surface, g, mat, 1.0, None))).collect();
                hs.into_iter().map(|t| (t.join().unwrap().ura - ura_x) / ura_x).collect()
            });
            println!("| {} | {h} | {:+.3e} | {:+.3e} | {:+.3e} | {:+.3e} |", name(surface), errs[0], errs[1], errs[2], errs[3]);
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let case = args.get(1).map(String::as_str).unwrap_or("bar");
    let out = args.get(2).map(String::as_str);
    if let Some(dir) = out {
        std::fs::create_dir_all(dir).unwrap();
    }
    match case {
        "bar" => bar_study(out),
        "lame" => {
            let hs: Vec<f64> = args.get(3).map(|v| v.split(',').map(|x| x.parse().unwrap()).collect()).unwrap_or(vec![2.0, 1.0, 0.5]);
            let surfaces: Vec<Mode> = match args.get(4).map(String::as_str) {
                Some(list) => list.split(',').map(parse_surface).collect(),
                None => vec![Mode::BLEND, Mode::BLEND_SM, Mode::GHOST2, Mode::GHOST],
            };
            lame_study(&hs, &surfaces, out)
        }
        "gamma" => gamma_study(),
        "nh" => {
            let hs: Vec<f64> = args.get(3).map(|v| v.split(',').map(|x| x.parse().unwrap()).collect()).unwrap_or(vec![2.0, 1.0, 0.5]);
            let surfaces: Vec<Mode> = match args.get(4).map(String::as_str) {
                Some(list) => list.split(',').map(parse_surface).collect(),
                None => vec![Mode::BLEND, Mode::GHOST],
            };
            nh_study(&hs, &surfaces)
        }
        "kirsch" | "disc" => {
            let hs: Vec<f64> = args.get(3).map(|v| v.split(',').map(|x| x.parse().unwrap()).collect()).unwrap_or(vec![1.0, 0.5]);
            let surfaces: Vec<Mode> = match args.get(4).map(String::as_str) {
                Some(list) => list.split(',').map(parse_surface).collect(),
                None => vec![Mode::BLEND, Mode::BLEND_SM, Mode::GHOST2, Mode::GHOST],
            };
            if case == "kirsch" { kirsch_study(&hs, &surfaces, out) } else { disc_study(&hs, &surfaces) }
        }
        "ghostcheck" => {
            let surface = parse_surface(args.get(3).map(String::as_str).unwrap_or("ghost"));
            for h in [2.0, 1.0, 0.5, 0.25] {
                let mat = Material::linear_elastic(E, NU, 1.0).unwrap();
                let st = Settings { viscosity: 0.0, ..lame_settings(surface) };
                let mut s =
                    Solid::new(centred_grid(LB, h), mat, Shape::Annulus { center: [0.0, 0.0], inner: LA, outer: LB }, st, Limits::default()).unwrap();
                s.set_ambient(|x| if x[0].hypot(x[1]) < 15.0 { [-LP, -LP, 0.0] } else { [0.0; 3] }).unwrap();
                let exact_zeta = |x: [f64; 2]| {
                    let r = x[0].hypot(x[1]).max(1.0);
                    let u = lame_exact(r).0;
                    [-u * x[0] / r, -u * x[1] / r]
                };
                s.set_motion(|x| {
                    let z = exact_zeta(x);
                    ([x[0] + z[0], x[1] + z[1]], [0.0; 2])
                })
                .unwrap();
                let (mut eg, mut eb) = (0.0f64, 0.0f64);
                let mut gloc = ([0.0; 2], 0.0);
                for k in 0..s.grid().len() {
                    let ring = (-1..=1).any(|dj| (-1..=1).any(|di| s.grid().neighbour(k, di, dj).is_some_and(|m| is_solid(&s, m))));
                    if s.kind(k) == Some(NodeKind::Band) && ring {
                        let x = s.grid().position(k);
                        let xi = s.reference(k).unwrap();
                        let z = exact_zeta(x);
                        let e = ((xi[0] - x[0] - z[0]).hypot(xi[1] - x[1] - z[1])) / lame_exact(LA).0;
                        if e > eg {
                            eg = e;
                            gloc = (x, s.phi(k).unwrap() / h);
                        }
                    }
                }
                // Residual force: one half kick from rest with no damping gives v = Δt/2 (∇·σ)/ρ.
                let dt = s.dt();
                let mut loc = ([0.0; 2], 0.0);
                s.run(1).unwrap();
                for k in 0..s.grid().len() {
                    if is_solid(&s, k) {
                        let v = s.velocity(k).unwrap();
                        let e = v[0].hypot(v[1]) / (0.5 * dt) * h / LP;
                        if e > eb {
                            eb = e;
                            loc = (s.grid().position(k), s.phi(k).unwrap() / h);
                        }
                    }
                }
                println!("max residual at {loc:?}, max ghost error at {gloc:?}");
                println!(
                    "h {h}: max ghost ζ error / u_r(a) {eg:.3e}; max residual |∇·σ| h / p {eb:.3e};  fallbacks {}, linear fills {}",
                    s.surface_fallbacks(),
                    s.surface_linear_fills()
                );
            }
        }
        "probe" => {
            let h: f64 = args.get(3).map(|v| v.parse().unwrap()).unwrap_or(1.0);
            let surface = parse_surface(args.get(4).map(String::as_str).unwrap_or("ghost"));
            let mat = Material::linear_elastic(E, NU, 1.0).unwrap();
            let grid = centred_grid(LB, h);
            let c = mat.p_wave_speed();
            let kappa: f64 = std::env::var("KAPPA").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0);
            let gfac: f64 = std::env::var("GAMMA").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0);
            let cfl: f64 = std::env::var("CFL").ok().and_then(|v| v.parse().ok()).unwrap_or(0.4);
            let st = Settings { damping: gfac * 0.5 * c / LB, viscosity: kappa, cfl, ..lame_settings(surface) };
            let mut s = Solid::new(grid, mat, Shape::Annulus { center: [0.0, 0.0], inner: LA, outer: LB }, st, Limits::default()).unwrap();
            s.set_ambient(|x| if x[0].hypot(x[1]) < 15.0 { [-LP, -LP, 0.0] } else { [0.0; 3] }).unwrap();
            let chunk = (500.0 * 0.4 / cfl) as usize;
            for i in 0..40 {
                if let Err(e) = s.run(chunk) {
                    println!("error {e}");
                    break;
                }
                let ura = fit_ur(&s, h, false);
                println!(
                    "{} speed {:.3e} ura err {:+.3e} fallbacks {}",
                    (i + 1) * 250,
                    s.max_speed(),
                    ((ura[0] * LA + ura[1] / LA) - lame_exact(LA).0) / lame_exact(LA).0,
                    s.surface_fallbacks()
                );
            }
        }
        other => eprintln!("unknown case {other}"),
    }
}
