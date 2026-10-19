//! Cut cells: the solid part of a grid cell that the surface φ = 0 crosses.
//!
//! A cell is the square between four grid nodes, its corners numbered counter-clockwise from
//! the lower left (c0 = (i, j), c1 = (i + 1, j), c2 = (i + 1, j + 1), c3 = (i, j + 1)). Its
//! midlines split it into four quarters, one per corner; the quarters of a node in its four
//! cells form the node's control volume (its dual cell). The midlines are the four half-faces
//! through which neighbouring control volumes exchange momentum:
//!
//! - half-face 0: x = h/2, y ∈ [0, h/2], between c0 and c1 (normal +x);
//! - half-face 1: y = h/2, x ∈ [h/2, h], between c1 and c2 (normal +y);
//! - half-face 2: x = h/2, y ∈ [h/2, h], between c3 and c2 (normal +x);
//! - half-face 3: y = h/2, x ∈ [0, h/2], between c0 and c3 (normal +y).
//!
//! In a cut cell the surface enters through one edge and leaves through another. Each piece of
//! the surface inside the cell is a quadratic curve through the two edge crossings and a third
//! point on φ = 0 above the middle of the chord: the curved side of the cut cell. Areas and
//! first moments come from Green's theorem over the solid part of each quarter's boundary
//! (exact with three Gauss points on every straight and quadratic side), so the solid area of
//! a quarter, the solid pieces of the half-faces and the surface pieces in each quarter all
//! use the same curve, and the boundary of every control volume closes exactly: a constant
//! stress gives zero net force on every cut cell to round-off.
//!
//! Coordinates here are local to the cell (the lower-left corner at the origin).

/// A point or vector in the plane.
pub type P2 = [f64; 2];

/// The root of f on [a, b] where f(a) and f(b) differ in sign (f(a) < 0 ≤ f(b) or the
/// reverse), by the Illinois variant of regula falsi, to `tol` in the argument.
pub fn bracketed_root(f: &dyn Fn(f64) -> f64, mut a: f64, mut b: f64, mut fa: f64, mut fb: f64, tol: f64) -> f64 {
    if fb == 0.0 {
        return b;
    }
    if fa == 0.0 {
        return a;
    }
    let mut side = 0i8;
    for _ in 0..100 {
        let c = if fb != fa { b - fb * (b - a) / (fb - fa) } else { 0.5 * (a + b) };
        let c = if c.is_finite() && c > a.min(b) && c < a.max(b) { c } else { 0.5 * (a + b) };
        let fc = f(c);
        if !fc.is_finite() || fc == 0.0 || (b - a).abs() <= tol {
            return c;
        }
        if (fc < 0.0) == (fb < 0.0) {
            b = c;
            fb = fc;
            if side == -1 {
                fa *= 0.5;
            }
            side = -1;
        } else {
            a = c;
            fa = fc;
            if side == 1 {
                fb *= 0.5;
            }
            side = 1;
        }
        if (b - a).abs() <= tol {
            return if fa.abs() < fb.abs() { a } else { b };
        }
    }
    0.5 * (a + b)
}

/// Two-point Gauss–Legendre rule on [0, 1] (abscissa, weight).
pub const GAUSS2: [(f64, f64); 2] = [(0.211_324_865_405_187_1, 0.5), (0.788_675_134_594_812_9, 0.5)];

/// Three-point Gauss–Legendre rule on [0, 1] (abscissa, weight).
pub const GAUSS3: [(f64, f64); 3] = [(0.112_701_665_379_258_3, 5.0 / 18.0), (0.5, 8.0 / 18.0), (0.887_298_334_620_741_7, 5.0 / 18.0)];

/// A quadratic curve x(t), t ∈ [0, 1], through `p` (t = 0), `m` (t = 1/2) and `q` (t = 1); a
/// straight segment when `m` is the midpoint.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Curve {
    /// Start.
    pub p: P2,
    /// The point at t = 1/2.
    pub m: P2,
    /// End.
    pub q: P2,
}

impl Curve {
    /// The straight segment from `a` to `b`.
    pub fn line(a: P2, b: P2) -> Curve {
        Curve { p: a, m: [0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])], q: b }
    }

    /// Coefficients (c0, c1, c2) of x(t) = c0 + c1 t + c2 t².
    fn coefficients(&self) -> [P2; 3] {
        let mut c = [[0.0; 2]; 3];
        for a in 0..2 {
            c[0][a] = self.p[a];
            c[1][a] = 4.0 * self.m[a] - 3.0 * self.p[a] - self.q[a];
            c[2][a] = 2.0 * self.p[a] + 2.0 * self.q[a] - 4.0 * self.m[a];
        }
        c
    }

    /// The point at parameter t.
    pub fn at(&self, t: f64) -> P2 {
        let c = self.coefficients();
        [c[0][0] + t * (c[1][0] + t * c[2][0]), c[0][1] + t * (c[1][1] + t * c[2][1])]
    }

    /// dx/dt at parameter t.
    pub fn tangent(&self, t: f64) -> P2 {
        let c = self.coefficients();
        [c[1][0] + 2.0 * t * c[2][0], c[1][1] + 2.0 * t * c[2][1]]
    }

    /// The piece between parameters t0 and t1, again a quadratic curve.
    pub fn sub(&self, t0: f64, t1: f64) -> Curve {
        Curve { p: self.at(t0), m: self.at(0.5 * (t0 + t1)), q: self.at(t1) }
    }

    /// The parameters in [0, 1] where coordinate `axis` equals `c`, ascending.
    pub fn hits(&self, axis: usize, c: f64) -> Vec<f64> {
        let k = self.coefficients();
        let (a2, a1, a0) = (k[2][axis], k[1][axis], k[0][axis] - c);
        let scale = a0.abs().max(a1.abs()).max(a2.abs());
        let mut out = Vec::with_capacity(2);
        if !(scale > 0.0) || !scale.is_finite() {
            return out;
        }
        if a2.abs() <= 1e-12 * scale {
            if a1.abs() > 1e-12 * scale {
                out.push(-a0 / a1);
            }
        } else {
            let disc = a1 * a1 - 4.0 * a2 * a0;
            if disc >= 0.0 {
                let sq = disc.sqrt();
                let qq = -0.5 * (a1 + if a1 >= 0.0 { sq } else { -sq });
                if qq != 0.0 {
                    out.push(qq / a2);
                    out.push(a0 / qq);
                } else {
                    out.push(0.0);
                }
            }
        }
        let mut out: Vec<f64> = out.into_iter().filter(|t| t.is_finite() && *t >= -1e-9 && *t <= 1.0 + 1e-9).map(|t| t.clamp(0.0, 1.0)).collect();
        out.sort_by(f64::total_cmp);
        out
    }

    /// Adds this oriented side's contribution to (area, ∫x dA, ∫y dA) by Green's theorem.
    fn green(&self, acc: &mut [f64; 3]) {
        for (t, w) in GAUSS3 {
            let x = self.at(t);
            let d = self.tangent(t);
            acc[0] += w * 0.5 * (x[0] * d[1] - x[1] * d[0]);
            acc[1] += w * 0.5 * x[0] * x[0] * d[1];
            acc[2] -= w * 0.5 * x[1] * x[1] * d[0];
        }
    }
}

/// What a cell needs to be cut: its spacing, which corners are inside the body, where the
/// surface crosses each edge, and the level set at any point of the cell.
pub struct CellInput<'a> {
    /// Grid spacing.
    pub h: f64,
    /// Whether each corner (c0..c3) is inside the body.
    pub inside: [bool; 4],
    /// For edge e (from corner e to corner e + 1, counter-clockwise), the crossing as a fraction
    /// of the edge from corner e, when the corners differ.
    pub cross: [Option<f64>; 4],
    /// The level set at a local point.
    pub phi: &'a dyn Fn(P2) -> f64,
}

/// The solid part of a cut cell.
#[derive(Debug, Clone, Default)]
pub struct CellCut {
    /// The surface pieces in the cell, oriented with the solid on their left (outward normal
    /// (dy, −dx)).
    pub arcs: Vec<Curve>,
    /// The solid area of each quarter.
    pub area: [f64; 4],
    /// ∫ x dA over the solid part of each quarter (local coordinates).
    pub moment: [P2; 4],
    /// The solid pieces of the four half-faces, each from its start towards its end as listed
    /// in the module notes.
    pub faces: [Vec<(P2, P2)>; 4],
    /// The surface pieces lying in each quarter.
    pub quarter_arcs: [Vec<Curve>; 4],
}

/// The local corner positions.
fn corner(h: f64, c: usize) -> P2 {
    match c % 4 {
        0 => [0.0, 0.0],
        1 => [h, 0.0],
        2 => [h, h],
        _ => [0.0, h],
    }
}

fn lerp(a: P2, b: P2, t: f64) -> P2 {
    [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]
}

/// The half-faces as (start, end), local coordinates.
pub fn half_face(h: f64, f: usize) -> (P2, P2) {
    let c = 0.5 * h;
    match f % 4 {
        0 => ([c, 0.0], [c, c]),
        1 => ([c, c], [h, c]),
        2 => ([c, c], [c, h]),
        _ => ([0.0, c], [c, c]),
    }
}

/// Cuts a cell with at least one corner inside and one outside. Returns `None` when the
/// crossings are inconsistent with the corner signs (an edge whose corners differ without a
/// crossing).
pub fn cut_cell(input: &CellInput) -> Option<CellCut> {
    let h = input.h;
    // Crossings in counter-clockwise order with their kind (true = the boundary walk leaves
    // the solid there).
    let mut crossings: Vec<(P2, bool)> = Vec::with_capacity(4);
    for e in 0..4 {
        let (a, b) = (input.inside[e], input.inside[(e + 1) % 4]);
        if a != b {
            let s = input.cross[e]?;
            if !s.is_finite() {
                return None;
            }
            crossings.push((lerp(corner(h, e), corner(h, e + 1), s.clamp(0.0, 1.0)), a));
        }
    }
    let n = crossings.len();
    if n == 0 || !n.is_multiple_of(2) {
        return None;
    }
    // Pair every exit with an entry. Two crossings: one surface piece. Four (a saddle): the
    // level set at the centre decides whether the solid joins the two inside corners (each exit
    // runs to the next entry, cutting off an outside corner) or not (back to the previous one).
    let joined = n == 2 || (input.phi)([0.5 * h, 0.5 * h]) < 0.0;
    let mut arcs = Vec::with_capacity(n / 2);
    for i in 0..n {
        if !crossings[i].1 {
            continue;
        }
        let mut j = i;
        for step in 1..n {
            let k = if joined { (i + step) % n } else { (i + n - step) % n };
            if !crossings[k].1 {
                j = k;
                break;
            }
        }
        if j == i {
            return None;
        }
        let (p, q) = (crossings[i].0, crossings[j].0);
        arcs.push(Curve { p, m: arc_middle(p, q, h, input.phi), q });
    }

    let mut out = CellCut { arcs, ..CellCut::default() };
    // Solid pieces of the half-faces.
    for f in 0..4 {
        let (a, b) = half_face(h, f);
        out.faces[f] = solid_pieces(a, b, &out.arcs, input.phi);
    }
    // Surface pieces per quarter.
    let c = 0.5 * h;
    for arc in &out.arcs {
        let mut ts = vec![0.0, 1.0];
        ts.extend(arc.hits(0, c));
        ts.extend(arc.hits(1, c));
        ts.sort_by(f64::total_cmp);
        ts.dedup_by(|a, b| (*a - *b).abs() < 1e-13);
        for w in ts.windows(2) {
            if w[1] - w[0] < 1e-13 {
                continue;
            }
            let piece = arc.sub(w[0], w[1]);
            let mid = piece.at(0.5);
            out.quarter_arcs[quadrant(mid, c)].push(piece);
        }
    }
    // Areas and moments of each quarter by Green's theorem over its solid boundary.
    for q in 0..4 {
        let mut acc = [0.0; 3];
        for side in quarter_boundary(input, &out, q) {
            side.green(&mut acc);
        }
        out.area[q] = acc[0].max(0.0);
        out.moment[q] = [acc[1], acc[2]];
    }
    Some(out)
}

/// The solid boundary of quarter q, counter-clockwise: the solid pieces of its four sides (two
/// half-edges of the cell, two half-faces) and the surface pieces in it.
fn quarter_boundary(input: &CellInput, cut: &CellCut, q: usize) -> Vec<Curve> {
    let rev = |v: &[(P2, P2)]| -> Vec<(P2, P2)> { v.iter().rev().map(|&(a, b)| (b, a)).collect() };
    let f = &cut.faces;
    let sides: [Vec<(P2, P2)>; 4] = match q {
        0 => [edge_pieces(input, 0, 0.0, 0.5), f[0].clone(), rev(&f[3]), edge_pieces(input, 3, 0.5, 1.0)],
        1 => [edge_pieces(input, 0, 0.5, 1.0), edge_pieces(input, 1, 0.0, 0.5), rev(&f[1]), rev(&f[0])],
        2 => [f[1].clone(), edge_pieces(input, 1, 0.5, 1.0), edge_pieces(input, 2, 0.0, 0.5), rev(&f[2])],
        _ => [f[3].clone(), f[2].clone(), edge_pieces(input, 2, 0.5, 1.0), edge_pieces(input, 3, 0.0, 0.5)],
    };
    let mut out: Vec<Curve> = sides.iter().flatten().map(|&(a, b)| Curve::line(a, b)).collect();
    out.extend(cut.quarter_arcs[q].iter().copied());
    out
}

/// The quarter (corner index) containing a local point.
pub fn quadrant(x: P2, c: f64) -> usize {
    match (x[0] > c, x[1] > c) {
        (false, false) => 0,
        (true, false) => 1,
        (true, true) => 2,
        (false, true) => 3,
    }
}

/// The solid pieces of edge e between fractions `ta` and `tb` (from corner e), in that order,
/// from the corner signs and the crossing.
fn edge_pieces(input: &CellInput, e: usize, ta: f64, tb: f64) -> Vec<(P2, P2)> {
    let (a, b) = (input.inside[e], input.inside[(e + 1) % 4]);
    let (lo, hi) = match (a, b) {
        (true, true) => (0.0, 1.0),
        (false, false) => return Vec::new(),
        (true, false) => (0.0, input.cross[e].unwrap_or(0.0)),
        (false, true) => (input.cross[e].unwrap_or(1.0), 1.0),
    };
    let (u, v) = (ta.min(tb), ta.max(tb));
    let (s0, s1) = (u.max(lo), v.min(hi));
    if !(s1 > s0) {
        return Vec::new();
    }
    let (p0, p1) = (corner(input.h, e), corner(input.h, e + 1));
    if ta <= tb { vec![(lerp(p0, p1, s0), lerp(p0, p1, s1))] } else { vec![(lerp(p0, p1, s1), lerp(p0, p1, s0))] }
}

/// The solid pieces of the axis-aligned segment a → b: split where the surface pieces cross
/// it, each part solid when the level set is negative at its middle.
fn solid_pieces(a: P2, b: P2, arcs: &[Curve], phi: &dyn Fn(P2) -> f64) -> Vec<(P2, P2)> {
    let axis = if (a[0] - b[0]).abs() < (a[1] - b[1]).abs() { 0 } else { 1 };
    let along = 1 - axis;
    let len = b[along] - a[along];
    if len == 0.0 {
        return Vec::new();
    }
    let mut ts = vec![0.0, 1.0];
    for arc in arcs {
        for t in arc.hits(axis, a[axis]) {
            let x = arc.at(t);
            let tau = (x[along] - a[along]) / len;
            if tau > 1e-13 && tau < 1.0 - 1e-13 {
                ts.push(tau);
            }
        }
    }
    ts.sort_by(f64::total_cmp);
    ts.dedup_by(|x, y| (*x - *y).abs() < 1e-13);
    let mut out: Vec<(P2, P2)> = Vec::new();
    for w in ts.windows(2) {
        if !(w[1] > w[0]) {
            continue;
        }
        if phi(lerp(a, b, 0.5 * (w[0] + w[1]))) < 0.0 {
            let (p0, p1) = (lerp(a, b, w[0]), lerp(a, b, w[1]));
            match out.last_mut() {
                Some(last) if last.1 == p0 => last.1 = p1,
                _ => out.push((p0, p1)),
            }
        }
    }
    out
}

/// A point on φ = 0 above the middle of the chord p → q (searched along the chord's normal),
/// or the chord's middle when the level set does not change sign there.
fn arc_middle(p: P2, q: P2, h: f64, phi: &dyn Fn(P2) -> f64) -> P2 {
    let c = [0.5 * (p[0] + q[0]), 0.5 * (p[1] + q[1])];
    let d = [q[0] - p[0], q[1] - p[1]];
    let len = d[0].hypot(d[1]);
    if !(len > 1e-9 * h) {
        return c;
    }
    // Left normal of p → q: the solid side.
    let nrm = [-d[1] / len, d[0] / len];
    let at = |s: f64| [c[0] + s * nrm[0], c[1] + s * nrm[1]];
    let f0 = phi(c);
    if f0 == 0.0 || !f0.is_finite() {
        return c;
    }
    // Inside (φ < 0) at the chord: the surface bulges to the outside (negative s), else inwards.
    let dir = if f0 < 0.0 { -1.0 } else { 1.0 };
    let steps = 16;
    let reach = 0.5 * len;
    let (mut lo, mut flo) = (0.0, f0);
    for k in 1..=steps {
        let s = dir * reach * k as f64 / steps as f64;
        let f = phi(at(s));
        if !f.is_finite() {
            return c;
        }
        if (f < 0.0) != (flo < 0.0) {
            let g = |t: f64| phi(at(t));
            let r = bracketed_root(&g, lo, s, flo, f, 1e-13 * h);
            let m = at(r);
            return [m[0].clamp(0.0, h), m[1].clamp(0.0, h)];
        }
        lo = s;
        flo = f;
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    fn circle(cx: f64, cy: f64, r: f64, inside_disc: bool) -> impl Fn(P2) -> f64 {
        move |x: P2| {
            let d = (x[0] - cx).hypot(x[1] - cy) - r;
            if inside_disc { d } else { -d }
        }
    }

    /// Cuts a unit cell by a level set given analytically (corner signs and edge crossings by
    /// bisection), as the solid does from the grid.
    fn cut(phi: &dyn Fn(P2) -> f64, h: f64) -> CellCut {
        let inside: [bool; 4] = std::array::from_fn(|c| phi(corner(h, c)) < 0.0);
        cut_cell(&CellInput { h, inside, cross: cross_of(phi, h), phi }).unwrap()
    }

    fn cross_of(phi: &dyn Fn(P2) -> f64, h: f64) -> [Option<f64>; 4] {
        std::array::from_fn(|e| {
            let (a, b) = (corner(h, e), corner(h, e + 1));
            if (phi(a) < 0.0) == (phi(b) < 0.0) {
                return None;
            }
            let (mut lo, mut hi) = (0.0, 1.0);
            for _ in 0..80 {
                let m = 0.5 * (lo + hi);
                if (phi(lerp(a, b, m)) < 0.0) == (phi(a) < 0.0) {
                    lo = m;
                } else {
                    hi = m;
                }
            }
            Some(0.5 * (lo + hi))
        })
    }

    #[test]
    fn straight_cut_areas_are_exact() {
        // Half-plane y < 0.3 + 0.2 x in a unit cell.
        let phi = |x: P2| x[1] - 0.3 - 0.2 * x[0];
        let c = cut(&phi, 1.0);
        let total: f64 = c.area.iter().sum();
        assert!((total - 0.4).abs() < 1e-12, "{total}");
        // Quarter 0: x, y < 1/2 under the line: ∫_0^½ min(½, 0.3 + 0.2x) dx = 0.15 + 0.025.
        assert!((c.area[0] - 0.175).abs() < 1e-12, "{:?}", c.area);
        assert!(c.area[2] < 1e-15 && c.area[3] < 1e-15);
    }

    #[test]
    fn curved_cut_area_matches_the_circle_and_closes() {
        // The cell [0, 1]² cut by a circle of radius 3 about (2, -2.4): the solid is the disc.
        let phi = circle(2.0, -2.4, 3.0, true);
        let c = cut(&phi, 1.0);
        // Reference area by fine sampling.
        let n = 2000;
        let mut a = 0.0;
        for i in 0..n {
            for j in 0..n {
                let x = [(i as f64 + 0.5) / n as f64, (j as f64 + 0.5) / n as f64];
                if phi(x) < 0.0 {
                    a += 1.0 / (n * n) as f64;
                }
            }
        }
        let total: f64 = c.area.iter().sum();
        assert!((total - a).abs() < 2e-4, "{total} vs {a}");
        // Closure: ∮ n ds = 0 over the solid boundary of every quarter, so a constant stress
        // gives no net force on any piece of a cut cell.
        let input = CellInput { h: 1.0, inside: std::array::from_fn(|k| phi(corner(1.0, k)) < 0.0), cross: [None; 4], phi: &phi };
        let input = CellInput { cross: cross_of(&phi, 1.0), ..input };
        for q in 0..4 {
            let mut net = [0.0; 2];
            for side in quarter_boundary(&input, &c, q) {
                for (t, w) in GAUSS3 {
                    let d = side.tangent(t);
                    net[0] += w * d[1];
                    net[1] -= w * d[0];
                }
            }
            assert!(net[0].abs() < 1e-14 && net[1].abs() < 1e-14, "quarter {q}: {net:?}");
        }
    }

    #[test]
    fn hits_are_found_on_curves() {
        let c = Curve { p: [0.0, 0.0], m: [0.5, 0.4], q: [1.0, 0.0] };
        let t = c.hits(1, 0.3);
        assert_eq!(t.len(), 2);
        for ti in t {
            assert!((c.at(ti)[1] - 0.3).abs() < 1e-12);
        }
        assert!(c.hits(1, 1.0).is_empty());
    }
}
