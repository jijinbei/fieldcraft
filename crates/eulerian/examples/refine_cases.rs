//! Verification runs of local refinement near the surface, for the research notes
//! (docs/research/reference-map.md §12): the Lamé cylinder, the rotating annular disc and the
//! Kirsch hole on a refined hierarchy (mapped boundary on the finest level), relaxed to rest.
//!
//! `cargo run --release -p fieldcraft-eulerian --example refine_cases -- CASE H0 LEVELS [key=value ...]`
//! with CASE one of lame, disc, kirsch; LEVELS = 0 is the uniform grid at H0 (the same code on
//! one level, restricted to the body and a margin). Keys:
//! - `band=` finest band half-width in finest cells (default 6), `nesting=` (default 2),
//! - `law=lin|nh` (default lin), `load=` load multiplier (default 0.01: small strain),
//! - `damp=` damping multiplier (γ = damp · 0.5 c_P / b), `cfl=`, `visc=`,
//! - `tolf=` relaxation tolerance multiplier, `steps=` most coarse steps,
//! - `dump=PREFIX` (writes PREFIX_nodes.csv, PREFIX_leaves.csv, PREFIX_surface.csv).
//!
//! Prints one JSON line of metrics: errors at the surface and in the interior (r within
//! [a + 3, b − 3]), node updates and timings.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

#[path = "common/closed_forms.rs"]
mod closed_forms;
use closed_forms::*;

use fieldcraft_eulerian::{Boundary, Grid, Hierarchy, Limits, Material, Refinement, Settings, Shape};
use std::collections::HashMap;
use std::fmt::Write as _;

/// The exact Cartesian stress and displacement of a case at x (per unit load).
fn exact(case: &str, x: [f64; 2]) -> ([f64; 3], [f64; 2]) {
    let r = x[0].hypot(x[1]);
    let (c, s) = (x[0] / r, x[1] / r);
    let cart = |sr: f64, st: f64| [sr * c * c + st * s * s, sr * s * s + st * c * c, (sr - st) * s * c];
    match case {
        "kirsch" => (kirsch_stress(x), kirsch_u(x)),
        "disc" => {
            let (u, sr, st) = disc(r);
            (cart(sr, st), [u * c, u * s])
        }
        _ => {
            let (u, sr, st) = lame(r);
            (cart(sr, st), [u * c, u * s])
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let case = args.get(1).cloned().unwrap_or_else(|| "lame".into());
    let h0: f64 = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(1.0);
    let levels: usize = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(1);
    let kv: HashMap<String, String> = args.iter().skip(4).filter_map(|a| a.split_once('=').map(|(k, v)| (k.to_string(), v.to_string()))).collect();
    let get = |k: &str, d: &str| kv.get(k).cloned().unwrap_or_else(|| d.to_string());
    let band: f64 = get("band", "6").parse().unwrap();
    let nesting: usize = get("nesting", "2").parse().unwrap();
    let law = get("law", "lin");
    let load: f64 = get("load", "0.01").parse().unwrap();
    let damp: f64 = get("damp", "1").parse().unwrap();
    let cfl: f64 = get("cfl", "0.4").parse().unwrap();
    let visc: f64 = get("visc", "0.1").parse().unwrap();
    let tolf: f64 = get("tolf", "1").parse().unwrap();
    let max_steps: usize = get("steps", "400000").parse().unwrap();
    let mat = if law == "lin" { Material::linear_elastic(E, NU, 1.0).unwrap() } else { Material::neo_hookean(E, NU, 1.0).unwrap() };

    let (a, b) = if case == "kirsch" { (A_K, 20.0) } else { (10.0, 20.0) };
    let n = (2.0 * (b + 5.0) / h0).round() as usize + 1;
    let o = -0.5 * (n as f64 - 1.0) * h0;
    let g = Grid::new(n, n, h0, [o, o], [false; 2], &Limits::default()).unwrap();
    let c = mat.p_wave_speed();
    let settings = Settings { damping: damp * 0.5 * c / b, cfl, viscosity: visc, boundary: Boundary::Mapped, ..Settings::default() };
    let refinement = Refinement { levels, band, nesting, ..Refinement::default() };
    let shape = Shape::Annulus { center: [0.0, 0.0], inner: a, outer: b };
    let t_build = std::time::Instant::now();
    let mut s = Hierarchy::new(g, mat, shape, settings, refinement, Limits::default()).unwrap();
    match case.as_str() {
        "disc" => s.set_body_force(move |x| [load * RW2 * x[0], load * RW2 * x[1]]).unwrap(),
        "kirsch" => s
            .set_ambient_field(move |x| {
                if x[0].hypot(x[1]) < 0.5 * (A_K + 20.0) {
                    [0.0; 3]
                } else {
                    let k = kirsch_stress(x);
                    [load * k[0], load * k[1], load * k[2]]
                }
            })
            .unwrap(),
        _ => s.set_ambient_field(move |x| if x[0].hypot(x[1]) < 15.0 { [-P * load, -P * load, 0.0] } else { [0.0; 3] }).unwrap(),
    }
    let build = t_build.elapsed().as_secs_f64();

    // Relax until the speed, less the best-fit rigid motion, is below the tolerance.
    let tol = tolf * 1e-5 * 0.009 * load * c / b;
    let t0 = std::time::Instant::now();
    let mut steps = 0;
    let mut converged = false;
    while steps < max_steps {
        if let Err(e) = s.run(20) {
            println!("{{\"case\":\"{case}\",\"h0\":{h0},\"levels\":{levels},\"diverged\":\"{e}\",\"t\":{}}}", s.time());
            return;
        }
        steps += 20;
        if deformation_speed(&s) < tol {
            converged = true;
            break;
        }
    }
    let wall = t0.elapsed().as_secs_f64();
    let stats = s.stats();

    // Interior and fit.
    let dofs = s.dofs();
    let sref = match case.as_str() {
        "kirsch" => 3.0 * S_FAR,
        "disc" => disc(a).2,
        _ => lame(a).2,
    };
    let uref = match case.as_str() {
        "kirsch" => kirsch_u([0.0, a])[1].hypot(kirsch_u([0.0, a])[0]).max(kirsch_u([a, 0.0])[0].abs()),
        "disc" => disc(a).0,
        _ => lame(a).0,
    };
    let (mut mm, mut rr) = ([[0.0; 3]; 3], [0.0; 3]);
    let (mut int_s_max, mut int_s_sq, mut int_u_max, mut int_n) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let mut nodes_csv = String::from("level,x,y,role,ux,uy,sxx,syy,sxy,exx,eyy,exy\n");
    for nr in &dofs {
        let sol = s.level(nr.level).unwrap();
        let k = nr.index;
        let x = nr.position;
        let r = x[0].hypot(x[1]);
        let u = sol.displacement(k).unwrap().map(|v| v / load);
        let st = s.stress(nr).map(|v| v.map(|w| w / load));
        let (es, eu) = exact(&case, x);
        if let Some(st) = st {
            if kv.contains_key("dump") {
                writeln!(
                    nodes_csv,
                    "{},{},{},{:?},{},{},{},{},{},{},{},{}",
                    nr.level,
                    x[0],
                    x[1],
                    s.role(nr.level, k).unwrap(),
                    u[0],
                    u[1],
                    st[0],
                    st[1],
                    st[2],
                    es[0],
                    es[1],
                    es[2]
                )
                .unwrap();
            }
            if r > a + 3.0 && r < b - 3.0 {
                let e = (st[0] - es[0]).abs().max((st[1] - es[1]).abs()).max((st[2] - es[2]).abs()) / sref;
                int_s_max = int_s_max.max(e);
                int_s_sq += e * e;
                int_u_max = int_u_max.max((u[0] - eu[0]).hypot(u[1] - eu[1]) / uref);
                int_n += 1.0;
            }
        }
        let hl = sol.grid().h();
        if case != "kirsch" && sol.phi(k).unwrap() < -2.0 * hl {
            let ur = (u[0] * x[0] + u[1] * x[1]) / r;
            let f = if case == "disc" { [r, 1.0 / r, r * r * r] } else { [r, 1.0 / r, 0.0] };
            for i in 0..3 {
                for j in 0..3 {
                    mm[i][j] += f[i] * f[j];
                }
                rr[i] += f[i] * ur;
            }
        }
    }
    if case != "disc" {
        mm[2][2] = 1.0;
    }
    let cf = solve3(&mm, rr);
    let urf = |r: f64| cf[0] * r + cf[1] / r + cf[2] * r * r * r;
    let ur_exact = |r: f64| if case == "disc" { disc(r).0 } else { lame(r).0 };

    // Surface.
    let mut surf_csv = String::from("x,y,nx,ny,sxx,syy,sxy\n");
    let (mut in_worst, mut out_worst, mut rms, mut nin) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for sp in s.surface_stress() {
        let x = sp.position;
        let st = sp.stress.map(|v| v / load);
        writeln!(surf_csv, "{},{},{},{},{},{},{}", x[0], x[1], sp.normal[0], sp.normal[1], st[0], st[1], st[2]).unwrap();
        let r = x[0].hypot(x[1]);
        let (_, stt, _) = polar(st, x);
        let (es, _) = exact(&case, x);
        let (_, ett, _) = polar([es[0], es[1], es[2], 0.0], x);
        let e = (stt - ett).abs() / sref;
        if r < 0.5 * (a + b) {
            in_worst = in_worst.max(e);
            rms += e * e;
            nin += 1.0;
        } else {
            out_worst = out_worst.max(e);
        }
    }
    let mut out = format!(
        "{{\"case\":\"{case}\",\"h0\":{h0},\"levels\":{levels},\"hf\":{},\"band\":{band},\"law\":\"{law}\",\"load\":{load}",
        h0 / (1u64 << levels) as f64
    );
    write!(
        out,
        ",\"steps\":{steps},\"converged\":{converged},\"wall\":{wall:.3},\"build\":{build:.3},\"updates\":{},\"nodes\":{:?},\"dofs\":{:?},\"leaves\":{:?}",
        stats.updates, stats.nodes, stats.dofs, stats.leaves
    )
    .unwrap();
    if case != "kirsch" {
        write!(out, ",\"ur_a\":{:.6e},\"ur_b\":{:.6e}", (urf(a) - ur_exact(a)) / ur_exact(a), (urf(b) - ur_exact(b)) / ur_exact(b)).unwrap();
    }
    write!(
        out,
        ",\"int_s_max\":{int_s_max:.4e},\"int_s_rms\":{:.4e},\"int_u_max\":{int_u_max:.4e},\"int_nodes\":{int_n},\"bore_st_max\":{in_worst:.4e},\"bore_st_rms\":{:.4e},\"outer_st_max\":{out_worst:.4e},\"bore_points\":{nin}}}",
        (int_s_sq / int_n.max(1.0)).sqrt(),
        (rms / nin.max(1.0)).sqrt()
    )
    .unwrap();
    println!("{out}");
    if let Some(prefix) = kv.get("dump") {
        std::fs::write(format!("{prefix}_nodes.csv"), nodes_csv).unwrap();
        std::fs::write(format!("{prefix}_surface.csv"), surf_csv).unwrap();
        let mut leaves = String::from("level,x,y,h\n");
        for (l, p, h) in s.leaf_cells() {
            writeln!(leaves, "{l},{},{},{h}", p[0], p[1]).unwrap();
        }
        std::fs::write(format!("{prefix}_leaves.csv"), leaves).unwrap();
    }
}

fn solve3(m: &[[f64; 3]; 3], r: [f64; 3]) -> [f64; 3] {
    let d = |a: &[[f64; 3]; 3]| {
        a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
    };
    let det = d(m);
    let mut out = [0.0; 3];
    for c in 0..3 {
        let mut a = *m;
        for i in 0..3 {
            a[i][c] = r[i];
        }
        out[c] = d(&a) / det;
    }
    out
}

/// The largest speed of a degree of freedom after removing the best-fit rigid motion.
fn deformation_speed(s: &Hierarchy) -> f64 {
    let pts: Vec<([f64; 2], [f64; 2])> = s.dofs().iter().filter_map(|n| Some((n.position, s.level(n.level)?.velocity(n.index)?))).collect();
    let m = pts.len().max(1) as f64;
    let (mut cx, mut cy, mut vx, mut vy) = (0.0, 0.0, 0.0, 0.0);
    for (x, v) in &pts {
        cx += x[0] / m;
        cy += x[1] / m;
        vx += v[0] / m;
        vy += v[1] / m;
    }
    let (mut num, mut den) = (0.0, 0.0);
    for (x, v) in &pts {
        let (dx, dy) = (x[0] - cx, x[1] - cy);
        num += dx * (v[1] - vy) - dy * (v[0] - vx);
        den += dx * dx + dy * dy;
    }
    let w = if den > 0.0 { num / den } else { 0.0 };
    pts.iter()
        .map(|(x, v)| {
            let (dx, dy) = (x[0] - cx, x[1] - cy);
            (v[0] - (vx - w * dy)).hypot(v[1] - (vy + w * dx))
        })
        .fold(0.0, f64::max)
}
