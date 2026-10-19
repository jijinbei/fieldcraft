//! Verification runs of the Eulerian solid against closed-form answers, for the research notes
//! (docs/research/reference-map.md): the Lamé cylinder, the rotating annular disc and the Kirsch
//! hole, with the blurred or the mapped boundary.
//!
//! `cargo run --release -p fieldcraft-eulerian --example rmt_cases -- CASE H [key=value ...]`
//! with CASE one of lame, disc, kirsch and keys
//! - `boundary=blurred|mapped` (default mapped),
//! - `damp=` damping multiplier (default 1: γ = 0.5 c_P / b),
//! - `cfl=` (default 0.4), `visc=` κ (default 0.1),
//! - `band=lin|exact` (ζ outside the body extrapolated, or the exact solution),
//! - `law=nh|lin` (neo-Hookean or linear elastic, default nh),
//! - `dump=PREFIX` (writes PREFIX_nodes.csv, PREFIX_surface.csv, PREFIX_cut.json).
//!
//! Prints one JSON line of metrics.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use fieldcraft_eulerian::{Boundary, Grid, Limits, Material, NodeKind, Settings, Shape, Solid};
use std::collections::HashMap;
use std::fmt::Write as _;

const E: f64 = 210_000.0;
const NU: f64 = 0.3;
const P: f64 = 100.0;
const RW2: f64 = 0.5;
const S_FAR: f64 = 100.0;

fn lame_constants() -> (f64, f64) {
    let mu = E / (2.0 * (1.0 + NU));
    (E * NU / ((1.0 + NU) * (1.0 - 2.0 * NU)), mu)
}

/// Lamé cylinder a = 10, b = 20, bore pressure P, plane strain: u_r and (σ_r, σ_θ).
fn lame(r: f64) -> (f64, f64, f64) {
    let (a, b) = (10.0f64, 20.0f64);
    let k = P * a * a / (b * b - a * a);
    ((1.0 + NU) * k / E * ((1.0 - 2.0 * NU) * r + b * b / r), k * (1.0 - b * b / (r * r)), k * (1.0 + b * b / (r * r)))
}

/// Rotating annular disc a = 10, b = 20, ρω² = RW2, both surfaces free, plane strain.
fn disc(r: f64) -> (f64, f64, f64) {
    let (a, b) = (10.0f64, 20.0f64);
    let (l, m) = lame_constants();
    let k = RW2 / (8.0 * (l + 2.0 * m));
    let c2 = (4.0 * l + 6.0 * m) * k * a * a * b * b / (2.0 * m);
    let c1 = (4.0 * l + 6.0 * m) * k * (a * a + b * b) / (2.0 * (l + m));
    let ur = c1 * r + c2 / r - k * r * r * r;
    let sr = 2.0 * (l + m) * c1 - 2.0 * m * c2 / (r * r) - (4.0 * l + 6.0 * m) * k * r * r;
    let st = 2.0 * (l + m) * c1 + 2.0 * m * c2 / (r * r) - (4.0 * l + 2.0 * m) * k * r * r;
    (ur, sr, st)
}

/// Kirsch: a hole of radius A_K in an infinite plate under σxx = S_FAR at infinity. Cartesian
/// stress (σxx, σyy, σxy) at x.
const A_K: f64 = 5.0;
fn kirsch_stress(x: [f64; 2]) -> [f64; 3] {
    let r = x[0].hypot(x[1]);
    let th = x[1].atan2(x[0]);
    let q = A_K * A_K / (r * r);
    let s = S_FAR;
    let (c2, s2) = ((2.0 * th).cos(), (2.0 * th).sin());
    let srr = 0.5 * s * (1.0 - q) + 0.5 * s * (1.0 - 4.0 * q + 3.0 * q * q) * c2;
    let stt = 0.5 * s * (1.0 + q) - 0.5 * s * (1.0 + 3.0 * q * q) * c2;
    let srt = -0.5 * s * (1.0 + 2.0 * q - 3.0 * q * q) * s2;
    let (c, sn) = (th.cos(), th.sin());
    [
        srr * c * c + stt * sn * sn - 2.0 * srt * sn * c,
        srr * sn * sn + stt * c * c + 2.0 * srt * sn * c,
        (srr - stt) * sn * c + srt * (c * c - sn * sn),
    ]
}

/// Kirsch displacement (plane strain), for the exact band.
fn kirsch_u(x: [f64; 2]) -> [f64; 2] {
    let r = x[0].hypot(x[1]);
    let th = x[1].atan2(x[0]);
    let mu = E / (2.0 * (1.0 + NU));
    let kappa = 3.0 - 4.0 * NU;
    let a = A_K;
    let s = S_FAR;
    let c2 = (2.0 * th).cos();
    let ur = s / (4.0 * mu) * (r * (kappa - 1.0) / 2.0 + r * c2 + a * a / r * (1.0 + (1.0 + kappa) * c2) - a.powi(4) / r.powi(3) * c2);
    let ut = s / (4.0 * mu) * ((1.0 - kappa) * a * a / r - r - a.powi(4) / r.powi(3)) * (2.0 * th).sin();
    let (c, sn) = (th.cos(), th.sin());
    [ur * c - ut * sn, ur * sn + ut * c]
}

fn polar(st: [f64; 4], x: [f64; 2]) -> (f64, f64, f64) {
    let r = x[0].hypot(x[1]);
    let (c, s) = (x[0] / r, x[1] / r);
    let srr = st[0] * c * c + st[1] * s * s + 2.0 * st[2] * s * c;
    let stt = st[0] * s * s + st[1] * c * c - 2.0 * st[2] * s * c;
    let srt = (st[1] - st[0]) * s * c + st[2] * (c * c - s * s);
    (srr, stt, srt)
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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let case = args.get(1).cloned().unwrap_or_else(|| "lame".into());
    let h: f64 = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(1.0);
    let kv: HashMap<String, String> = args.iter().skip(3).filter_map(|a| a.split_once('=').map(|(k, v)| (k.to_string(), v.to_string()))).collect();
    let get = |k: &str, d: &str| kv.get(k).cloned().unwrap_or_else(|| d.to_string());
    let boundary = if get("boundary", "mapped") == "blurred" { Boundary::Blurred } else { Boundary::Mapped };
    let damp: f64 = get("damp", "1").parse().unwrap();
    let cfl: f64 = get("cfl", "0.4").parse().unwrap();
    let visc: f64 = get("visc", "0.1").parse().unwrap();
    let exact_band = get("band", "lin") == "exact";
    let law = get("law", "nh");
    let max_steps: usize = get("steps", "2000000").parse().unwrap();
    // Load multiplier (the closed forms are linear in the load; stresses are reported per unit
    // multiplier).
    let load: f64 = get("load", "1").parse().unwrap();
    let mat = if law == "lin" { Material::linear_elastic(E, NU, 1.0).unwrap() } else { Material::neo_hookean(E, NU, 1.0).unwrap() };

    let (a, b) = if case == "kirsch" { (A_K, 20.0) } else { (10.0, 20.0) };
    let n = (2.0 * (b + 5.0) / h).round() as usize + 1;
    let o = -0.5 * (n as f64 - 1.0) * h;
    let g = Grid::new(n, n, h, [o, o], [false; 2], &Limits::default()).unwrap();
    let c = mat.p_wave_speed();
    let layers: usize = get("layers", "4").parse().unwrap();
    let settings = Settings { damping: damp * 0.5 * c / b, cfl, viscosity: visc, boundary, band_layers: layers, ..Settings::default() };
    let mut s = Solid::new(g, mat, Shape::Annulus { center: [0.0, 0.0], inner: a, outer: b }, settings, Limits::default()).unwrap();
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
    let ur_exact: fn(f64) -> f64 = if case == "disc" { |r| disc(r).0 } else { |r| lame(r).0 };
    if exact_band {
        match case.as_str() {
            "kirsch" => s.set_exterior_map(move |x| {
                let u = kirsch_u(x);
                [-load * u[0], -load * u[1]]
            }),
            "disc" => s.set_exterior_map(move |x| {
                let r = x[0].hypot(x[1]).max(1e-9);
                let u = load * disc(r).0;
                [-u * x[0] / r, -u * x[1] / r]
            }),
            _ => s.set_exterior_map(move |x| {
                let r = x[0].hypot(x[1]).max(1e-9);
                let u = load * lame(r).0;
                [-u * x[0] / r, -u * x[1] / r]
            }),
        }
        .unwrap();
    }
    let t0 = std::time::Instant::now();
    let tol = 1e-5 * 0.009 * load * c / b;
    if let Some(tr) = kv.get("trace") {
        let every: usize = tr.parse().unwrap();
        for i in 0..max_steps / every {
            if let Err(e) = s.run(every) {
                println!("diverged: {e}");
                return;
            }
            let st = s.cut_stats().unwrap_or_default();
            let nsolid = (0..s.grid().len()).filter(|&k| matches!(s.kind(k), Some(NodeKind::Solid))).count();
            let px: f64 = get("px", "-6").parse().unwrap();
            let py: f64 = get("py", "-8").parse().unwrap();
            let probe = s.grid().index(((px - o) / h).round() as usize, ((py - o) / h).round() as usize);
            print!("solid {nsolid} probe {:?} phi {:+.3e} v {:?} ", s.kind(probe), s.phi(probe).unwrap_or(f64::NAN), s.velocity(probe));
            println!(
                "step {} t {:.4} vmax {:.3e} (tol {tol:.1e}) sweeps {} change {:.2e} fallback {} merged {}",
                (i + 1) * every,
                s.time(),
                s.max_speed(),
                st.sweeps,
                st.last_change,
                st.fallback_points,
                st.merged
            );
        }
        let mut sp: Vec<(f64, usize)> = (0..s.grid().len())
            .filter(|&k| matches!(s.kind(k), Some(NodeKind::Solid)))
            .map(|k| {
                let v = s.velocity(k).unwrap();
                (v[0].hypot(v[1]), k)
            })
            .collect();
        sp.sort_by(|a, b| b.0.total_cmp(&a.0));
        for (v, k) in sp.iter().take(8) {
            let x = s.grid().position(*k);
            println!("  speed {v:.3e} at ({:.1}, {:.1}) r {:.3} phi {:.3}", x[0], x[1], x[0].hypot(x[1]), s.phi(*k).unwrap());
        }
        return;
    }
    let rx = match s.relax(tol, max_steps) {
        Ok(r) => r,
        Err(e) => {
            println!("{{\"case\":\"{case}\",\"h\":{h},\"diverged\":\"{e}\",\"t\":{}}}", s.time());
            return;
        }
    };
    let wall = t0.elapsed().as_secs_f64();

    // u_r fit over nodes with φ < −2h.
    let mut mm = [[0.0; 3]; 3];
    let mut rr = [0.0; 3];
    let (mut hoop, mut nh) = (0.0, 0.0);
    let mut first_a = (f64::NAN, f64::NAN);
    let mut nodes_csv = String::from("x,y,kind,phi,ux,uy,sxx,syy,sxy\n");
    let mut kirsch_node_err: f64 = 0.0;
    // σθ at the first solid node above the hole on the y axis (θ = 90°).
    let mut kirsch_node_peak = (f64::NAN, f64::NAN, f64::NAN);
    for k in 0..s.grid().len() {
        let x = s.grid().position(k);
        let kind = s.kind(k);
        if kv.contains_key("dump") {
            let phi = s.phi(k).unwrap_or(f64::NAN);
            let u = s.displacement(k).unwrap_or([f64::NAN; 2]);
            let st = s.stress(k).unwrap_or([f64::NAN; 4]);
            let kid = match kind {
                Some(NodeKind::Solid) => 0,
                Some(NodeKind::Fixed) => 1,
                Some(NodeKind::Band) => 2,
                _ => 3,
            };
            writeln!(nodes_csv, "{},{},{kid},{phi},{},{},{},{},{}", x[0], x[1], u[0], u[1], st[0], st[1], st[2]).unwrap();
        }
        if kind != Some(NodeKind::Solid) {
            continue;
        }
        let r = x[0].hypot(x[1]);
        let u = s.displacement(k).unwrap();
        let ur = (u[0] * x[0] + u[1] * x[1]) / r / load;
        let st = s.stress(k).unwrap().map(|v| v / load);
        let (srr, stt, _) = polar(st, x);
        if x[1].abs() < 1e-9 && x[0] > 0.0 && (first_a.0.is_nan() || r < first_a.0) {
            first_a = (r, srr);
        }
        if s.phi(k).unwrap() < -2.0 * h {
            let f = if case == "disc" { [r, 1.0 / r, r * r * r] } else { [r, 1.0 / r, 0.0] };
            for i in 0..3 {
                for j in 0..3 {
                    mm[i][j] += f[i] * f[j];
                }
                rr[i] += f[i] * ur;
            }
        }
        if (r - 15.0).abs() < 0.5 * h {
            hoop += stt;
            nh += 1.0;
        }
        if case == "kirsch" && x[0].abs() < 1e-9 && x[1] > 0.0 && (kirsch_node_peak.0.is_nan() || r < kirsch_node_peak.0) {
            let ex = kirsch_stress(x);
            let (_, et, _) = polar([ex[0], ex[1], ex[2], 0.0], x);
            kirsch_node_peak = (r, stt, et);
        }
        if case == "kirsch" && r < a + 1.5 * h {
            let ex = kirsch_stress(x);
            let (_, et, _) = polar([ex[0], ex[1], ex[2], 0.0], x);
            kirsch_node_err = kirsch_node_err.max((stt - et).abs() / (3.0 * S_FAR));
        }
    }
    if case != "disc" {
        mm[2][2] = 1.0;
    }
    let cf = solve3(&mm, rr);
    let urf = |r: f64| cf[0] * r + cf[1] / r + cf[2] * r * r * r;
    let rel = |x: f64, y: f64| 100.0 * (x - y) / y;
    let exact_st15 = if case == "disc" { disc(15.0).2 } else { lame(15.0).2 };
    let exact_srr1 = if case == "disc" { disc(first_a.0).1 } else { lame(first_a.0).1 };

    // Surface: σrr and σθ at the bore surface points.
    let surf = s.surface_stress();
    let mut surf_csv = String::from("x,y,nx,ny,sxx,syy,sxy,szz\n");
    let (mut bore_srr, mut bore_stt, mut nb, mut bore_stt_err, mut bore_srr_err) = (0.0, 0.0, 0.0, 0.0f64, 0.0f64);
    let mut kirsch_peak = (f64::NAN, 0.0f64);
    let mut kirsch_rms = (0.0f64, 0.0f64);
    for sp in &surf {
        writeln!(
            surf_csv,
            "{},{},{},{},{},{},{},{}",
            sp.position[0], sp.position[1], sp.normal[0], sp.normal[1], sp.stress[0], sp.stress[1], sp.stress[2], sp.stress[3]
        )
        .unwrap();
        let r = sp.position[0].hypot(sp.position[1]);
        if r > 0.5 * (a + b) {
            continue;
        }
        let (srr, stt, _) = polar(sp.stress.map(|v| v / load), sp.position);
        bore_srr += srr;
        bore_stt += stt;
        nb += 1.0;
        match case.as_str() {
            "kirsch" => {
                let th = sp.position[1].atan2(sp.position[0]);
                let ex = S_FAR * (1.0 - 2.0 * (2.0 * th).cos());
                kirsch_rms.0 += (stt - ex).powi(2);
                kirsch_rms.1 += 1.0;
                // Peak: the points nearest θ = ±90°.
                let dist = (th.abs() - std::f64::consts::FRAC_PI_2).abs();
                if kirsch_peak.0.is_nan() || dist < kirsch_peak.0 - 1e-12 {
                    kirsch_peak = (dist, stt);
                }
                bore_stt_err = bore_stt_err.max((stt - ex).abs() / (3.0 * S_FAR));
            }
            "disc" => {
                let ex = disc(r);
                bore_stt_err = bore_stt_err.max((stt - ex.2).abs() / ex.2.abs());
                bore_srr_err = bore_srr_err.max((srr - ex.1).abs() / ex.2.abs());
            }
            _ => {
                let ex = lame(r);
                bore_stt_err = bore_stt_err.max((stt - ex.2).abs() / ex.2.abs());
                bore_srr_err = bore_srr_err.max((srr - ex.1).abs() / P);
            }
        }
    }
    let stats = s.cut_stats();
    let mut out = format!(
        "{{\"case\":\"{case}\",\"h\":{h},\"boundary\":\"{:?}\",\"damp\":{damp},\"cfl\":{cfl},\"visc\":{visc},\"band\":\"{}\",\"law\":\"{law}\",\"steps\":{},\"converged\":{},\"wall\":{wall:.3},\"ms_per_step\":{:.4},\"nodes\":{}",
        boundary,
        if exact_band { "exact" } else { "lin" },
        rx.steps,
        rx.converged,
        1000.0 * wall / rx.steps as f64,
        s.grid().len()
    );
    if case != "kirsch" {
        write!(
            out,
            ",\"ur_a\":{:.5},\"ur_b\":{:.5},\"st15\":{:.5},\"srr1_r\":{:.4},\"srr1\":{:.4},\"srr1_exact\":{:.4}",
            rel(urf(a), ur_exact(a)),
            rel(urf(b), ur_exact(b)),
            rel(hoop / nh, exact_st15),
            first_a.0,
            first_a.1,
            exact_srr1
        )
        .unwrap();
    }
    if nb > 0.0 {
        write!(
            out,
            ",\"bore_srr\":{:.4},\"bore_stt\":{:.4},\"bore_points\":{nb},\"bore_stt_maxerr\":{:.5},\"bore_srr_maxerr\":{:.5}",
            bore_srr / nb,
            bore_stt / nb,
            bore_stt_err,
            bore_srr_err
        )
        .unwrap();
    }
    if case == "kirsch" {
        write!(
            out,
            ",\"kirsch_peak\":{:.4},\"kirsch_peak_err\":{:.5},\"kirsch_rms\":{:.5},\"kirsch_node_maxerr\":{:.5},\"kirsch_node_r\":{:.4},\"kirsch_node_stt\":{:.4},\"kirsch_node_exact\":{:.4}",
            kirsch_peak.1,
            100.0 * (kirsch_peak.1 - 3.0 * S_FAR) / (3.0 * S_FAR),
            (kirsch_rms.0 / kirsch_rms.1.max(1.0)).sqrt() / (3.0 * S_FAR) * 100.0,
            100.0 * kirsch_node_err,
            kirsch_node_peak.0,
            kirsch_node_peak.1,
            kirsch_node_peak.2
        )
        .unwrap();
    }
    if let Some(st) = stats {
        write!(
            out,
            ",\"cut_cells\":{},\"surface_points\":{},\"valid_points\":{},\"mapped_faces\":{},\"fallback_points\":{},\"merged\":{},\"lost_area\":{},\"sweeps\":{}",
            st.cut_cells, st.surface_points, st.valid_points, st.mapped_faces, st.fallback_points, st.merged, st.lost_area, st.sweeps
        )
        .unwrap();
    }
    out.push('}');
    println!("{out}");
    if let Some(prefix) = kv.get("dump") {
        std::fs::write(format!("{prefix}_nodes.csv"), nodes_csv).unwrap();
        std::fs::write(format!("{prefix}_surface.csv"), surf_csv).unwrap();
        if let Some(view) = s.cut_view() {
            let mut j = String::from("{\"cells\":[");
            for (i, c) in view.cells.iter().enumerate() {
                if i > 0 {
                    j.push(',');
                }
                let arcs: Vec<String> =
                    c.arcs.iter().map(|a| format!("[[{},{}],[{},{}],[{},{}]]", a[0][0], a[0][1], a[1][0], a[1][1], a[2][0], a[2][1])).collect();
                let pos = s.grid().position(c.node);
                let owners: Vec<String> = c.quarter_owner.iter().map(|o| o.map_or("null".into(), |v| v.to_string())).collect();
                write!(
                    j,
                    "{{\"x\":{},\"y\":{},\"arcs\":[{}],\"area\":[{},{},{},{}],\"owner\":[{}]}}",
                    pos[0],
                    pos[1],
                    arcs.join(","),
                    c.quarter_area[0],
                    c.quarter_area[1],
                    c.quarter_area[2],
                    c.quarter_area[3],
                    owners.join(",")
                )
                .unwrap();
            }
            j.push_str("],\"points\":[");
            for (i, p) in view.points.iter().enumerate() {
                if i > 0 {
                    j.push(',');
                }
                write!(
                    j,
                    "{{\"x\":{},\"y\":{},\"axis\":{},\"sign\":{},\"depth\":{},\"n1\":{},\"n2\":{},\"valid\":{},\"next\":{}}}",
                    p.position[0],
                    p.position[1],
                    p.axis,
                    p.sign,
                    p.depth,
                    p.nodes[0],
                    p.nodes[1],
                    p.valid,
                    p.next.map_or("null".into(), |v| v.to_string())
                )
                .unwrap();
            }
            j.push_str(&format!("],\"h\":{h},\"origin\":{o},\"n\":{n}}}"));
            std::fs::write(format!("{prefix}_cut.json"), j).unwrap();
        }
    }
}
