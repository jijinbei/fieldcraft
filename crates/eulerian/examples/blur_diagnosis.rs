//! Where the blurred surface's error comes from: the Lamé cylinder (a = 10, b = 20, bore
//! pressure 100, E = 210000, ν = 0.3, neo-Hookean) relaxed with the blurred surface, varying
//! one thing at a time:
//! - the band's ζ by the linear extrapolation (`lin`) or prescribed from the exact Lamé field
//!   (`exact`, [`Solid::set_exterior_map`]): separates the extrapolation from the blur;
//! - the momentum at every node with H < 1 ([`Boundary::Blurred`], `all`) or only at the
//!   body's nodes ([`Boundary::BlurredBody`], `body`): separates the band nodes' motion;
//! - the blur half-width ε, the artificial viscosity κ and the damping γ (a factor on
//!   0.5 c_P / b).
//!
//! `cargo run --release -p fieldcraft-eulerian --example blur_diagnosis -- [h,h,…]`
//! (default 1,0.5). Prints one line per run: the relative errors of u_r(a) and u_r(b) from a
//! fit u_r = A r + B/r over the nodes with φ < −2h, of the hoop stress at r = 15, and the steps
//! to rest. The numbers are summarised in docs/research/reference-map.md §8.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::needless_range_loop)]

use fieldcraft_eulerian::{Boundary, Grid, Limits, Material, NodeKind, Settings, Shape, Solid};

const A: f64 = 10.0;
const B: f64 = 20.0;
const P: f64 = 100.0;
const E: f64 = 210_000.0;
const NU: f64 = 0.3;

/// Lamé's radial displacement (plane strain).
fn lame_ur(r: f64) -> f64 {
    let k = P * A * A / (B * B - A * A);
    (1.0 + NU) * k / E * ((1.0 - 2.0 * NU) * r + B * B / r)
}

/// Lamé's hoop stress.
fn lame_hoop(r: f64) -> f64 {
    let k = P * A * A / (B * B - A * A);
    k * (1.0 + B * B / (r * r))
}

/// ζ = −u of the exact Lamé field.
fn lame_zeta(x: [f64; 2]) -> [f64; 2] {
    let r = x[0].hypot(x[1]);
    if r < 1e-9 {
        return [0.0; 2];
    }
    let u = lame_ur(r);
    [-u * x[0] / r, -u * x[1] / r]
}

struct Run {
    h: f64,
    exact_band: bool,
    boundary: Boundary,
    blur: f64,
    viscosity: f64,
    damping: f64,
}

fn run(c: &Run) -> String {
    let mat = Material::neo_hookean(E, NU, 1.0).unwrap();
    let h = c.h;
    let n = (2.0 * (B + 5.0) / h).round() as usize + 1;
    let o = -0.5 * (n as f64 - 1.0) * h;
    let g = Grid::new(n, n, h, [o, o], [false; 2], &Limits::default()).unwrap();
    let cp = mat.p_wave_speed();
    let settings = Settings {
        damping: c.damping * 0.5 * cp / B,
        blur: c.blur,
        viscosity: c.viscosity,
        boundary: c.boundary,
        band_layers: ((c.blur + 2.0).ceil() as usize).max(4),
        ..Settings::default()
    };
    let tag = format!(
        "{}/{} eps={} visc={} damping×{}",
        if c.exact_band { "exact" } else { "lin" },
        if c.boundary == Boundary::Blurred { "all" } else { "body" },
        c.blur,
        c.viscosity,
        c.damping
    );
    let mut s = Solid::new(g, mat, Shape::Annulus { center: [0.0, 0.0], inner: A, outer: B }, settings, Limits::default()).unwrap();
    s.set_ambient(|x| if x[0].hypot(x[1]) < 0.5 * (A + B) { [-P, -P, 0.0] } else { [0.0; 3] }).unwrap();
    if c.exact_band {
        s.set_exterior_map(lame_zeta).unwrap();
    }
    let rx = match s.relax(1e-5 * 0.009 * cp / B, 600_000) {
        Ok(r) => r,
        Err(e) => return format!("h={h:<6} {tag:<36} DIVERGED at t={:.3}: {e}", s.time()),
    };
    // Least-squares fit u_r = A r + B / r over the nodes clear of the surface, the hoop stress
    // averaged over the nodes at r = 15.
    let (mut m, mut rhs) = ([[0.0; 2]; 2], [0.0; 2]);
    let (mut hoop, mut nh) = (0.0, 0.0);
    for k in 0..s.grid().len() {
        if s.kind(k) != Some(NodeKind::Solid) {
            continue;
        }
        let x = s.grid().position(k);
        let r = x[0].hypot(x[1]);
        let u = s.displacement(k).unwrap();
        let ur = (u[0] * x[0] + u[1] * x[1]) / r;
        if s.phi(k).unwrap() < -2.0 * h {
            let f = [r, 1.0 / r];
            for i in 0..2 {
                for j in 0..2 {
                    m[i][j] += f[i] * f[j];
                }
                rhs[i] += f[i] * ur;
            }
        }
        if (r - 15.0).abs() < 0.5 * h {
            let st = s.stress(k).unwrap();
            let (ct, sn) = (x[0] / r, x[1] / r);
            hoop += st[0] * sn * sn + st[1] * ct * ct - 2.0 * st[2] * sn * ct;
            nh += 1.0;
        }
    }
    let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    let ca = (rhs[0] * m[1][1] - rhs[1] * m[0][1]) / det;
    let cb = (m[0][0] * rhs[1] - m[1][0] * rhs[0]) / det;
    let fit = |r: f64| ca * r + cb / r;
    let rel = |a: f64, b: f64| 100.0 * (a - b) / b;
    format!(
        "h={h:<6} {tag:<36} u_r(a) {:+8.3} %  u_r(b) {:+8.3} %  σθ(15) {:+8.3} %  ({} steps{})",
        rel(fit(A), lame_ur(A)),
        rel(fit(B), lame_ur(B)),
        rel(hoop / nh, lame_hoop(15.0)),
        rx.steps,
        if rx.converged { "" } else { ", not at rest" }
    )
}

fn main() {
    let hs: Vec<f64> = std::env::args().nth(1).map(|a| a.split(',').map(|v| v.parse().unwrap()).collect()).unwrap_or(vec![1.0, 0.5]);
    let base = |h: f64| Run { h, exact_band: false, boundary: Boundary::Blurred, blur: 1.5, viscosity: 0.1, damping: 1.0 };
    for &h in &hs {
        let mut runs = Vec::new();
        for exact_band in [false, true] {
            for boundary in [Boundary::Blurred, Boundary::BlurredBody] {
                runs.push(Run { exact_band, boundary, ..base(h) });
            }
            for blur in [1.0, 2.0] {
                runs.push(Run { exact_band, blur, ..base(h) });
            }
            runs.push(Run { exact_band, viscosity: 0.0, ..base(h) });
        }
        for damping in [0.5, 2.0, 4.0] {
            runs.push(Run { damping, ..base(h) });
            runs.push(Run { damping, boundary: Boundary::BlurredBody, exact_band: true, ..base(h) });
        }
        for r in &runs {
            println!("{}", run(r));
        }
    }
}
