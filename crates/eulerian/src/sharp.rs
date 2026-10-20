//! A sharp traction boundary: ghost values of the reference map just outside the body,
//! chosen so that the solid's own stress carries the prescribed traction at the surface.
//!
//! The blurred surface spreads the boundary condition over a band 2ε wide and builds the
//! band's ζ = ξ − x by linear extrapolation; where the stress varies across the band (a
//! loaded surface with a stress gradient: the bore of a pressurised cylinder, a hole in a
//! plate, a slab under a body force) the result is first order and depends on where the
//! surface cuts the grid. Here the solid stress is used alone on every edge that touches the
//! body, and the nodes outside that the stencils reach (the first 8-neighbour ring, the
//! *ghost* nodes) get ζ from a local polynomial that both fits the nearby body nodes and
//! satisfies the traction condition σ(F)·n = t exactly at the surface. This is the ghost-cell
//! idea of sharp-interface immersed boundary methods for Neumann data (the boundary condition
//! imposed at the surface point nearest the ghost node, the extension built from the
//! inside), here as a constrained least-squares fit because the traction couples both
//! components of ζ and involves tangential as well as normal derivatives.
//!
//! For a ghost node g, with offsets d from g in grid spacings:
//! 1. a weighted linear fit of ζ to the body nodes in the window |dᵢ| ≤ R gives a first ζ(d);
//! 2. the surface point d_b on φ₀(x + ζ(x)) = 0 near g, by the projection iteration
//!    d ← d − φ ∇φ/|∇φ|² (bisection on the last step if it crosses the surface); the normal
//!    n = ∇φ/|∇φ| there (∇φ by central differences);
//! 3. the traction t = σ_ambient(x_b)·n (σ_ambient bilinear between nodes: zero for a free
//!    surface, −p I for a pressure, any prescribed stress field);
//! 4. the constrained least-squares fit: minimise Σ w (ζ(dₘ) − ζₘ)² over the body nodes
//!    (w = 1/|dₘ|²) with a polynomial ζ(d) in each component, subject to the two equations
//!    [σ(G₀) + C:(∇ζ(d_b) − G₀)]·n = t: the stress linearised about the gradient G₀ of the
//!    previous fit, C = ∂σ/∂(∇ζ) by central differences of the material law (so any law
//!    works). Solved through the KKT system with partial pivoting;
//! 5. steps 2–4 repeated (`passes`, 2 by default): the second pass moves d_b with the new fit
//!    and linearises about its gradient. The linearisation is then exact at equilibrium
//!    (the residual of the traction equation is σ(G) − t with G the fit's own gradient), and
//!    linear elasticity at small strain is reproduced to O(strain²) after one pass.
//!
//! The ghost value is the fit at d = 0. Every ghost is fitted from body nodes only (never from
//! other ghosts), so the result does not depend on the node order.
//!
//! The polynomial, in order of preference: cubic over the 7 × 7 window (R = 3: on a flat face
//! three layers of body nodes plus the traction fix the 10 + 10 coefficients); quadratic over
//! the 5 × 5 window, then the 7 × 7 one; linear with the same two traction equations
//! (first-order stress, still sharp); else the plain linear extrapolation of the blurred method,
//! counted in `Solid::surface_fallbacks`. A quadratic ζ reproduces a linear stress exactly
//! (the slab under a uniform body force) but its truncation error at a curved surface is
//! large on coarse grids: at the Lamé bore (radius 10 h at h = 1) the ghost error is about
//! 1 % of u_r(a) at h = 1 and the cylinder converges as h^1.8–2.0 from −3.7 % (h = 1); the
//! cubic lowers the ghost error 4–6 times and gives −2.6 % / −0.40 % / −0.06 % at h = 1 /
//! 0.5 / 0.25 (`ghost_degree` in the settings picks the highest degree). Two things that did
//! not help, measured at the Lamé bore: other weights (1/|d|⁴, weights centred on the surface
//! point; the error is truncation, not weighting) and extra traction rows at the edge crossings
//! of the window, which over-constrain the fit's gradient and doubled the error.
//!
//! Exactness: a ζ that is quadratic in x and satisfies the traction condition everywhere on the
//! surface (linear stress) is reproduced exactly, flat or curved surface; a linear ζ
//! (homogeneous deformation, rigid motion) is reproduced for any normal, so corners are exact
//! for homogeneous states.
//!
//! Small cut cells need no special rule: the fit never divides by the distance from a node to
//! the surface (unlike extrapolation along a grid line through the boundary point), so a
//! surface that passes arbitrarily close to a body node or to a ghost node gives a regular
//! system. A boundary point exactly at a node is a value row and a traction row at the same
//! place, which constrain different coefficients. What can fail is the data, not the cut: too
//! few body nodes in the window, or nodes that leave a coefficient free or barely seen
//! (parallel rows of nodes, the inside of a corner); the KKT system is equilibrated (unit
//! diagonal) and a pivot below 1e-6 refuses the fit, then the next fit in the order above.
//! A cubic accepted at a convex corner without that test amplified round-off into a ξ error of
//! 0.05 in the rotating block.
//!
//! The rule for a surface that comes to rest on a node (the other small-cut case) is in
//! `Solid::rebuild`: a body node leaves at φ ≥ 0 but a band node enters only at φ < −h/10, so a
//! ghost may lie up to h/10 inside the body; the projection then runs outwards.
//!
//! Stability: the closure is not symmetric, and without the artificial viscosity (κ = 0) a
//! slowly growing surface mode appears in the Lamé cylinder after a few thousand steps; with
//! the default κ = 0.1 it is damped and the static answer does not depend on κ.
//!
//! Sources: R. Mittal, H. Dong, M. Bozkurttas, F. M. Najjar, A. Vargas, A. von Loebbecke,
//! *A versatile sharp interface immersed boundary method for incompressible flows with
//! complex boundaries*, J. Comput. Phys. 227(10):4825–4852, 2008, doi:10.1016/j.jcp.2008.01.028
//! (ghost cells with the boundary condition at the boundary intercept, for Dirichlet and
//! Neumann data); the traction-constrained polynomial fit is our own formulation of that idea
//! for elasticity.

use crate::grid::Grid;
use crate::material::{Material, Tensor};
use crate::shape::Shape;

/// Coefficients of a cubic in (dx, dy): 1, dx, dy, dx², dx dy, dy², dx³, dx² dy, dx dy², dy³.
const NQ: usize = 10;
/// Smallest accepted pivot of the equilibrated KKT system.
const PIVOT: f64 = 1e-6;

/// The quadratic part of the basis.
const N2: usize = 6;

/// One polynomial per component of ζ, in offsets of grid spacings from the ghost node.
type Poly = [[f64; NQ]; 2];

/// What the body looks like to the fill.
pub(crate) struct Body<'a> {
    pub grid: &'a Grid,
    pub shape: &'a Shape,
    pub material: &'a Material,
    /// Nodes whose ζ is known from the body (solid and fixed).
    pub source: &'a [bool],
    pub zeta: &'a [[f64; 2]],
    /// The prescribed stress outside (σxx, σyy, σxy) at each node.
    pub ambient: &'a [[f64; 3]],
    /// Linearise-and-refit passes.
    pub passes: usize,
    /// Highest polynomial degree of the fit (2 or 3).
    pub degree: usize,
}

/// How a ghost node was filled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Fill {
    /// The cubic fit with the exact traction.
    Cubic,
    /// The quadratic fit with the exact traction.
    Quadratic,
    /// The linear fit with the exact traction (too few nodes for a quadratic).
    Linear,
}

/// The traction-constrained ζ at ghost node `g`, or `None` when no constrained fit exists there
/// (the caller keeps the plain extrapolation).
pub(crate) fn ghost(body: &Body, g: usize) -> Option<([f64; 2], Fill)> {
    // Cubic needs three layers of body nodes (the 7 × 7 window); the quadratic two (5 × 5,
    // then 7 × 7); the linear fit is the last resort.
    if body.degree >= 3
        && let Some(v @ (_, Fill::Cubic)) = window(body, g, 3, NQ)
    {
        return Some(v);
    }
    if let Some(v @ (_, Fill::Quadratic)) = window(body, g, 2, N2) {
        return Some(v);
    }
    window(body, g, 3, N2)
}

/// The constrained fit over the window of half-width `r` with at most `nb` basis functions,
/// falling back to the linear fit when the data do not determine it.
fn window(body: &Body, g: usize, r: isize, nb: usize) -> Option<([f64; 2], Fill)> {
    let samples = gather(body, g, r);
    let mut poly = constrained(&samples, None, 3)?;
    let mut kind = Fill::Linear;
    for _ in 0..body.passes.max(1) {
        let local = Local { body, g, poly: &poly };
        let cond = local.rows_at(local.project(r as f64 + 1.0)?)?;
        if let Some(p) = constrained(&samples, Some(&cond), nb) {
            poly = p;
            kind = if nb == NQ { Fill::Cubic } else { Fill::Quadratic };
        } else {
            poly = constrained(&samples, Some(&cond), 3)?;
            kind = Fill::Linear;
        }
    }
    let v = [poly[0][0], poly[1][0]];
    v.iter().all(|x| x.is_finite()).then_some((v, kind))
}

/// The body nodes in the window of half-width `r` around g: (offset, ζ).
fn gather(body: &Body, g: usize, r: isize) -> Vec<([f64; 2], [f64; 2])> {
    let mut out = Vec::with_capacity(((2 * r + 1) * (2 * r + 1)) as usize);
    for dj in -r..=r {
        for di in -r..=r {
            if let Some(m) = body.grid.neighbour(g, di, dj)
                && body.source.get(m).copied().unwrap_or(false)
                && let Some(z) = body.zeta.get(m)
                && z.iter().all(|x| x.is_finite())
            {
                out.push(([di as f64, dj as f64], *z));
            }
        }
    }
    out
}

/// The traction condition at the surface point nearest g, linearised about the fit `poly`:
/// rows a·c = b over the 12 coefficients, scaled to O(1).
struct Condition {
    rows: [[f64; 2 * NQ]; 2],
    rhs: [f64; 2],
}

fn basis(d: [f64; 2]) -> [f64; NQ] {
    let (x, y) = (d[0], d[1]);
    [1.0, x, y, x * x, x * y, y * y, x * x * x, x * x * y, x * y * y, y * y * y]
}

/// ∂basis/∂d_c.
fn basis_grad(d: [f64; 2]) -> [[f64; NQ]; 2] {
    let (x, y) = (d[0], d[1]);
    [[0.0, 1.0, 0.0, 2.0 * x, y, 0.0, 3.0 * x * x, 2.0 * x * y, y * y, 0.0], [0.0, 0.0, 1.0, 0.0, x, 2.0 * y, 0.0, x * x, 2.0 * x * y, 3.0 * y * y]]
}

fn eval(p: &Poly, d: [f64; 2]) -> [f64; 2] {
    let q = basis(d);
    let mut v = [0.0; 2];
    for r in 0..2 {
        for i in 0..NQ {
            v[r] += p[r][i] * q[i];
        }
    }
    v
}

/// ∇ζ (physical units) of the fit at d: g[r][c] = ∂ζ_r/∂x_c.
fn gradient(p: &Poly, d: [f64; 2], h: f64) -> Tensor {
    let dq = basis_grad(d);
    let mut g = [[0.0; 2]; 2];
    for r in 0..2 {
        for c in 0..2 {
            for i in 0..NQ {
                g[r][c] += p[r][i] * dq[c][i];
            }
            g[r][c] /= h;
        }
    }
    g
}

/// The in-plane Cauchy stress (σxx, σyy, σxy) for ∇ζ = g.
fn stress(m: &Material, g: &Tensor) -> Option<[f64; 3]> {
    let a = [[1.0 + g[0][0], g[0][1]], [g[1][0], 1.0 + g[1][1]]];
    let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    if !(det > 0.0) || !det.is_finite() {
        return None;
    }
    let f = [[a[1][1] / det, -a[0][1] / det], [-a[1][0] / det, a[0][0] / det]];
    let s = m.cauchy(&f).ok()?;
    Some([s[0], s[1], s[2]])
}

/// The fit around one ghost node: the level set it implies and the traction rows.
struct Local<'a> {
    body: &'a Body<'a>,
    g: usize,
    poly: &'a Poly,
}

impl Local<'_> {
    /// φ = φ₀(x + ζ(x)) at offset d (grid spacings) from the ghost.
    fn phi(&self, d: [f64; 2]) -> f64 {
        let h = self.body.grid.h();
        let xg = self.body.grid.position(self.g);
        let z = eval(self.poly, d);
        self.body.shape.phi0([xg[0] + h * d[0] + z[0], xg[1] + h * d[1] + z[1]])
    }

    /// ∇φ in physical units, by central differences (step 1e-4 h).
    fn grad_phi(&self, d: [f64; 2]) -> [f64; 2] {
        let h = self.body.grid.h();
        let e = 1e-4;
        let px = self.phi([d[0] + e, d[1]]) - self.phi([d[0] - e, d[1]]);
        let py = self.phi([d[0], d[1] + e]) - self.phi([d[0], d[1] - e]);
        [px / (2.0 * e * h), py / (2.0 * e * h)]
    }

    /// The surface point near the ghost, within `reach` grid spacings.
    fn project(&self, reach: f64) -> Option<[f64; 2]> {
        let h = self.body.grid.h();
        let phi = |d: [f64; 2]| self.phi(d);
        let grad_phi = |d: [f64; 2]| self.grad_phi(d);
        // The surface point: Newton steps along the local normal, d ← d − φ ∇φ/|∇φ|² (the usual
        // projection onto a level set, with the gradient re-evaluated at each point so that it
        // ends at a corner rather than passing it), and bisection on the last step once it
        // crosses the surface (a kink of φ₀ inside the body cannot make it cycle). A ghost is
        // normally outside (φ > 0), but may lie a hair inside (the entry hysteresis of
        // `Solid::rebuild`), so the side is the sign of φ at the ghost.
        let tol = 1e-12 * h;
        let mut d = [0.0f64, 0.0];
        let mut f = phi(d);
        if !f.is_finite() || f < -0.5 * h {
            return None;
        }
        let side = if f >= 0.0 { 1.0 } else { -1.0 };
        let mut found = f.abs() <= tol;
        for _ in 0..40 {
            if found {
                break;
            }
            let gp = grad_phi(d);
            let g2 = gp[0] * gp[0] + gp[1] * gp[1];
            if !(g2 > 1e-12) || !g2.is_finite() {
                return None;
            }
            let next = [d[0] - f * gp[0] / g2 / h, d[1] - f * gp[1] / g2 / h];
            if !(next[0].hypot(next[1]) <= reach) {
                return None;
            }
            let fn_ = phi(next);
            if !fn_.is_finite() {
                return None;
            }
            if fn_.abs() <= tol {
                (d, found) = (next, true);
                break;
            }
            if side * fn_ > 0.0 {
                (d, f) = (next, fn_);
                continue;
            }
            // Crossed: bisect on the segment d → next (φ has the ghost's sign at d, the other
            // at next).
            let (mut lo, mut hi) = (0.0f64, 1.0f64);
            for _ in 0..60 {
                let mid = 0.5 * (lo + hi);
                let fm = phi([d[0] + mid * (next[0] - d[0]), d[1] + mid * (next[1] - d[1])]);
                if fm.abs() <= tol {
                    (lo, hi) = (mid, mid);
                    break;
                }
                if side * fm > 0.0 {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            let t = 0.5 * (lo + hi);
            d = [d[0] + t * (next[0] - d[0]), d[1] + t * (next[1] - d[1])];
            found = true;
        }
        if !found {
            return None;
        }
        Some(d)
    }

    /// The traction condition at surface point db, linearised about the fit.
    fn rows_at(&self, db: [f64; 2]) -> Option<Condition> {
        let (body, g, poly) = (self.body, self.g, self.poly);
        let h = body.grid.h();
        let grad_phi = |d: [f64; 2]| self.grad_phi(d);
        let gb = grad_phi(db);
        let nb = gb[0].hypot(gb[1]);
        if !(nb > 1e-6) || !nb.is_finite() {
            return None;
        }
        let n = [gb[0] / nb, gb[1] / nb];
        let amb = ambient_at(body, g, db);
        let t = [amb[0] * n[0] + amb[2] * n[1], amb[2] * n[0] + amb[1] * n[1]];

        // Linearise σ about G₀ = ∇ζ(d_b): σ(G) ≈ σ₀ + Σ T[s][r][c] (G − G₀)[r][c].
        let g0 = gradient(poly, db, h);
        let s0 = stress(body.material, &g0)?;
        let mut tan = [[[0.0f64; 2]; 2]; 3];
        let eta = 1e-6;
        for r in 0..2 {
            for c in 0..2 {
                let mut gp = g0;
                let mut gm = g0;
                gp[r][c] += eta;
                gm[r][c] -= eta;
                let sp = stress(body.material, &gp)?;
                let sm = stress(body.material, &gm)?;
                for k in 0..3 {
                    tan[k][r][c] = (sp[k] - sm[k]) / (2.0 * eta);
                }
            }
        }
        // Traction components from (σxx, σyy, σxy).
        let nmat = [[n[0], 0.0, n[1]], [0.0, n[1], n[0]]];
        let dq = basis_grad(db);
        let (_, mu, _) = body.material.constants();
        // Rows in units of the shear modulus per grid spacing, so they are O(1).
        let scale = h / mu;
        let mut cond = Condition { rows: [[0.0; 2 * NQ]; 2], rhs: [0.0; 2] };
        for a in 0..2 {
            let mut lin = 0.0;
            for k in 0..3 {
                let mut tg = 0.0;
                for r in 0..2 {
                    for c in 0..2 {
                        tg += tan[k][r][c] * g0[r][c];
                    }
                }
                lin += nmat[a][k] * (s0[k] - tg);
            }
            cond.rhs[a] = (t[a] - lin) * scale;
            for r in 0..2 {
                for i in 0..NQ {
                    let mut v = 0.0;
                    for k in 0..3 {
                        for c in 0..2 {
                            v += nmat[a][k] * tan[k][r][c] * dq[c][i];
                        }
                    }
                    cond.rows[a][r * NQ + i] = v / h * scale;
                }
            }
        }
        let ok = cond.rows.iter().flatten().chain(cond.rhs.iter()).all(|x| x.is_finite());
        ok.then_some(cond)
    }
}

/// σ_ambient at offset d from node g, bilinear between the four surrounding nodes (the value at
/// g when one of them is off the grid).
fn ambient_at(body: &Body, g: usize, d: [f64; 2]) -> [f64; 3] {
    let own = body.ambient.get(g).copied().unwrap_or([0.0; 3]);
    let (i0, j0) = (d[0].floor(), d[1].floor());
    let (fx, fy) = (d[0] - i0, d[1] - j0);
    let mut out = [0.0; 3];
    for (di, dj, w) in [(0, 0, (1.0 - fx) * (1.0 - fy)), (1, 0, fx * (1.0 - fy)), (0, 1, (1.0 - fx) * fy), (1, 1, fx * fy)] {
        let Some(a) = body.grid.neighbour(g, i0 as isize + di, j0 as isize + dj).and_then(|m| body.ambient.get(m)) else {
            return own;
        };
        for k in 0..3 {
            out[k] += w * a[k];
        }
    }
    out
}

/// The weighted least-squares fit of ζ over `samples` with the first `nb` basis functions
/// (3: linear, 6: quadratic), subject to the traction condition when given. `None` when the
/// system is singular.
fn constrained(samples: &[([f64; 2], [f64; 2])], cond: Option<&Condition>, nb: usize) -> Option<Poly> {
    let nu = 2 * nb;
    let nc = if cond.is_some() { 2 } else { 0 };
    let n = nu + nc;
    // Augmented KKT matrix [[Mᵀ W M, Aᵀ], [A, 0]] | [Mᵀ W y; b], at most 14 × 15.
    let mut a = [[0.0f64; 2 * NQ + 3]; 2 * NQ + 2];
    for (d, y) in samples {
        let r2 = d[0] * d[0] + d[1] * d[1];
        if !(r2 > 0.0) {
            continue;
        }
        let w = 1.0 / r2;
        let q = basis(*d);
        for comp in 0..2 {
            let o = comp * nb;
            for i in 0..nb {
                for j in 0..nb {
                    a[o + i][o + j] += w * q[i] * q[j];
                }
                a[o + i][n] += w * q[i] * y[comp];
            }
        }
    }
    if let Some(c) = cond {
        for row in 0..2 {
            for comp in 0..2 {
                for i in 0..nb {
                    let v = c.rows[row][comp * NQ + i];
                    a[nu + row][comp * nb + i] = v;
                    a[comp * nb + i][nu + row] = v;
                }
            }
            a[nu + row][n] = c.rhs[row];
        }
    }
    // Symmetric equilibration: unit diagonal in the normal-equation block, unit rows for the
    // constraints. The pivot test below is then a scale-free rank test (a monomial the data
    // barely see, as at a corner, gives a tiny pivot and the fit is refused rather than
    // amplifying round-off).
    let mut sc = [1.0f64; 2 * NQ + 2];
    for i in 0..nu {
        if !(a[i][i] > 0.0) || !a[i][i].is_finite() {
            return None;
        }
        sc[i] = 1.0 / a[i][i].sqrt();
    }
    for r in nu..n {
        let norm = (0..nu).map(|c| a[r][c] * a[r][c]).sum::<f64>().sqrt();
        if !(norm > 0.0) || !norm.is_finite() {
            return None;
        }
        sc[r] = 1.0 / norm;
    }
    for r in 0..n {
        for c in 0..n {
            a[r][c] *= sc[r] * sc[c];
        }
        a[r][n] *= sc[r];
    }
    // Gaussian elimination with partial pivoting.
    for col in 0..n {
        let piv = (col..n).max_by(|&x, &y| a[x][col].abs().total_cmp(&a[y][col].abs()))?;
        if !(a[piv][col].abs() > PIVOT) {
            return None;
        }
        a.swap(col, piv);
        for r in col + 1..n {
            let f = a[r][col] / a[col][col];
            if f != 0.0 {
                for c in col..=n {
                    a[r][c] -= f * a[col][c];
                }
            }
        }
    }
    let mut x = [0.0f64; 2 * NQ + 2];
    for r in (0..n).rev() {
        let mut v = a[r][n];
        for c in r + 1..n {
            v -= a[r][c] * x[c];
        }
        x[r] = v / a[r][r];
    }
    for r in 0..n {
        x[r] *= sc[r];
    }
    let mut p = [[0.0; NQ]; 2];
    for comp in 0..2 {
        for i in 0..nb {
            p[comp][i] = x[comp * nb + i];
        }
    }
    p.iter().flatten().all(|v| v.is_finite()).then_some(p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Limits;

    fn setup(shape: Shape, zeta: impl Fn([f64; 2]) -> [f64; 2]) -> (Grid, Vec<bool>, Vec<[f64; 2]>) {
        let grid = Grid::new(21, 21, 0.1, [-1.0, -1.0], [false; 2], &Limits::default()).unwrap();
        let mut source = vec![false; grid.len()];
        let mut z = vec![[0.0; 2]; grid.len()];
        for k in 0..grid.len() {
            let x = grid.position(k);
            z[k] = zeta(x);
            source[k] = shape.phi0([x[0] + z[k][0], x[1] + z[k][1]]) < 0.0;
        }
        (grid, source, z)
    }

    /// A slab x < 0.43 under the uniform body force b: σxx = ρ b (L − x), zero at the face.
    /// ζ is quadratic and the traction is met everywhere on the face, so the fill is exact.
    #[test]
    fn a_quadratic_field_with_a_free_face_is_reproduced() {
        let m = Material::linear_elastic(1000.0, 0.3, 1.0).unwrap();
        let (l, mu, _) = m.constants();
        let mm = l + 2.0 * mu;
        let face = 0.43;
        let k = 1e-3;
        // u = (k/M)(L x − x²/2) along x (σxx = k (L − x)); ζ = −u.
        let shape = Shape::Rectangle { min: [-5.0, -5.0], max: [face, 5.0] };
        let zeta = |x: [f64; 2]| [-(k / mm) * (face * x[0] - 0.5 * x[0] * x[0]), 0.0];
        let (grid, source, z) = setup(shape, zeta);
        let amb = vec![[0.0; 3]; grid.len()];
        let body = Body { grid: &grid, shape: &shape, material: &m, source: &source, zeta: &z, ambient: &amb, passes: 2, degree: 3 };
        let mut n = 0;
        for g in 0..grid.len() {
            let x = grid.position(g);
            if source[g] || !(x[0] < face + 0.15) || x[1].abs() > 0.7 {
                continue;
            }
            let (v, kind) = ghost(&body, g).unwrap();
            assert_eq!(kind, Fill::Cubic);
            let e = zeta(x);
            // The small-strain linearisation of the law leaves O(strain²).
            assert!((v[0] - e[0]).abs() < 1e-12 && (v[1] - e[1]).abs() < 1e-12, "{x:?}: {v:?} vs {e:?}");
            n += 1;
        }
        assert!(n > 10);
    }

    /// A homogeneous neo-Hookean stretch held by the matching ambient stress: the linear map is
    /// reproduced at every ghost node of a block, corners included.
    #[test]
    fn homogeneous_states_are_exact_at_corners() {
        let m = Material::neo_hookean(1000.0, 0.3, 1.0).unwrap();
        let f = [[1.2, 0.1], [0.0, 0.9]];
        let det = f[0][0] * f[1][1] - f[0][1] * f[1][0];
        let fi = [[f[1][1] / det, -f[0][1] / det], [-f[1][0] / det, f[0][0] / det]];
        let s = m.cauchy(&f).unwrap();
        for shape in [Shape::Rectangle { min: [-0.43, -0.37], max: [0.41, 0.33] }, Shape::Rectangle { min: [-0.5, -0.5], max: [0.5, 0.5] }] {
            homogeneous_block(m, f, fi, s, shape);
        }
        // Uniaxial stretch with the y faces on grid nodes.
        let f = [[1.25, 0.0], [0.0, 1.0]];
        let fi = [[0.8, 0.0], [0.0, 1.0]];
        let s = m.cauchy(&f).unwrap();
        homogeneous_block(m, f, fi, s, Shape::Rectangle { min: [-0.5, -0.5], max: [0.5, 0.5] });
    }

    fn homogeneous_block(m: Material, f: [[f64; 2]; 2], fi: [[f64; 2]; 2], s: [f64; 4], shape: Shape) {
        let _ = f;
        let zeta = |x: [f64; 2]| [fi[0][0] * x[0] + fi[0][1] * x[1] - x[0], fi[1][0] * x[0] + fi[1][1] * x[1] - x[1]];
        let (grid, source, z) = setup(shape, zeta);
        let amb = vec![[s[0], s[1], s[2]]; grid.len()];
        let body = Body { grid: &grid, shape: &shape, material: &m, source: &source, zeta: &z, ambient: &amb, passes: 2, degree: 3 };
        let mut n = 0;
        for g in 0..grid.len() {
            let ring = (-1..=1).any(|dj| (-1..=1).any(|di| grid.neighbour(g, di, dj).is_some_and(|q| source[q])));
            if source[g] || !ring {
                continue;
            }
            let (v, _) = ghost(&body, g).unwrap();
            let e = zeta(grid.position(g));
            assert!((v[0] - e[0]).abs() < 1e-11 && (v[1] - e[1]).abs() < 1e-11, "{:?}: {v:?} vs {e:?}", grid.position(g));
            n += 1;
        }
        assert!(n > 20);
    }

    /// A curved surface (a bore) with a quadratic ζ and the ambient stress equal to the field's
    /// own (linear) stress: the traction holds everywhere, the ambient is bilinear-exact, so the
    /// fill must reproduce the field.
    #[test]
    fn a_quadratic_field_at_a_curved_surface_is_reproduced() {
        let m = Material::linear_elastic(1000.0, 0.3, 1.0).unwrap();
        let (l, mu, _) = m.constants();
        let mut c = [[0.0; NQ]; 2];
        c[0][..N2].copy_from_slice(&[1e-6, 2e-6, -1e-6, 3e-6, 1e-6, -2e-6]);
        c[1][..N2].copy_from_slice(&[-2e-6, 1e-6, 1e-6, -1e-6, 2e-6, 1e-6]);
        let zeta = |x: [f64; 2]| {
            let q = basis(x);
            let mut z = [0.0; 2];
            for r in 0..2 {
                for i in 0..NQ {
                    z[r] += c[r][i] * q[i];
                }
            }
            z
        };
        let sig = |x: [f64; 2]| {
            // Small strain: ε = −sym ∇ζ.
            let dq = basis_grad(x);
            let mut g = [[0.0; 2]; 2];
            for r in 0..2 {
                for k in 0..2 {
                    for i in 0..NQ {
                        g[r][k] += c[r][i] * dq[k][i];
                    }
                }
            }
            let (exx, eyy, exy) = (-g[0][0], -g[1][1], -0.5 * (g[0][1] + g[1][0]));
            [l * (exx + eyy) + 2.0 * mu * exx, l * (exx + eyy) + 2.0 * mu * eyy, 2.0 * mu * exy]
        };
        let shape = Shape::Annulus { center: [0.03, -0.02], inner: 0.37, outer: 5.0 };
        let (grid, source, z) = setup(shape, zeta);
        let amb: Vec<_> = (0..grid.len()).map(|k| sig(grid.position(k))).collect();
        let body = Body { grid: &grid, shape: &shape, material: &m, source: &source, zeta: &z, ambient: &amb, passes: 2, degree: 3 };
        let mut worst = 0.0f64;
        for g in 0..grid.len() {
            let ring = (-1..=1).any(|dj| (-1..=1).any(|di| grid.neighbour(g, di, dj).is_some_and(|q| source[q])));
            if source[g] || !ring {
                continue;
            }
            let (v, _) = ghost(&body, g).unwrap();
            let e = zeta(grid.position(g));
            worst = worst.max((v[0] - e[0]).abs().max((v[1] - e[1]).abs()));
        }
        // ζ ~ 1e-6: the geometric nonlinearity of the law leaves O(strain) relative.
        assert!(worst < 1e-10, "{worst}");
    }

    #[test]
    fn singular_systems_are_refused() {
        // Collinear samples cannot fix a linear fit.
        let line: Vec<_> = (1..5).map(|i| ([-(i as f64), 0.0], [1.0, 2.0])).collect();
        assert!(constrained(&line, None, 3).is_none());
        assert!(constrained(&[], None, NQ).is_none());
    }
}
