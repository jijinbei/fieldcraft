//! The mapped boundary: sharp cut cells with the traction imposed on the curved surface.
//!
//! The blurred interface ([`crate::solid::Boundary::Blurred`]) smears the surface over a zone
//! a few cells wide and needs ζ = ξ − x extrapolated across it; where the strain varies, that
//! extrapolation biases the stress the surface carries. Here the surface is kept sharp:
//!
//! - **Control volumes.** Momentum lives at the nodes inside the body (φ < 0). The control
//!   volume of a node is the solid part of its dual cell (the square of side h around it), cut
//!   by the surface φ = 0 ([`crate::cut`]: the surface in each cut cell is a quadratic curve
//!   through its edge crossings). Its lumped mass is ρ times its true solid area. Solid in the
//!   dual cell of a node outside the body joins an inside neighbour: the one across the
//!   surface along the axis where |∂φ/∂x| is largest, else the other axis, else the deepest
//!   inside edge neighbour, else the deepest diagonal one. A control volume still below
//!   [`MERGE_FRACTION`] h² joins its deepest inside edge neighbour. (A straight surface through
//!   or beyond a node always leaves it at least half its dual cell, so merging is only needed
//!   on curvature comparable with h or on features a cell thin.)
//! - **Fluxes.** Neighbouring control volumes exchange ∫σ·n over the solid part of the face
//!   between them. Faces between two full cells keep the plain compact stencil of the interior.
//!   Faces next to a cut cell are integrated with two Gauss points on each solid piece, with σ
//!   from the solid law at F = (∇ξ)⁻¹ given by the surface-fitted map below; no blending. On
//!   the curved side the flux is the prescribed traction t = σ_ambient·n, integrated with three
//!   Gauss points along the quadratic curve with its line Jacobian.
//! - **The surface-fitted map.** Where a grid line along axis d crosses the surface (no more
//!   than about 72° from the normal), the crossing is a *surface point* with its own value ζ_S,
//!   and the two inside nodes n₁, n₂ behind it on the line. Between neighbouring lines the
//!   map T(u, η) = (x_d′ = (l + u) h, x_d = Y(u) − σ η h) takes the square in (u, η) onto the
//!   curved cell: u across the lines, η the depth below the surface along them, Y(u) the
//!   surface position interpolated through the crossings with the same Lagrange shape
//!   functions in u as the field (isoparametric), so η = 0 is exactly the curved side. The
//!   field is ζ(u, η) = Σ_l L_l(u) p_l(η), with p_l the quadratic through (0, ζ_S), (s, ζ_n₁),
//!   (s + 1, ζ_n₂) on line l. Its gradient is ∇ζ = (∂ζ/∂u, ∂ζ/∂η) T′⁻¹, F = (I + ∇ζ)⁻¹.
//! - **The surface values** ζ_S are not extrapolated: at each surface point they make the
//!   traction of the solid law equal the prescribed one, σ(F)·n = t, with F from the map
//!   (normal derivative from the one-sided quadratic, tangential derivative along the surface
//!   from the neighbouring surface points). Newton on the coupled conditions of each chain of
//!   surface points (one family's points on consecutive lines), with a banded
//!   finite-difference Jacobian factorised by LU, warm-started from the previous step.
//!
//! ζ outside the body is still extrapolated (the band of [`crate::solid`]), but only for
//! what needs a neighbour there: the upwind differences of the ξ advection, where the
//! crossings fall on the grid edges (the level set φ₀(ξ) at the outside nodes), and the
//! artificial viscosity's velocity gradient. Neither the stress nor the traction uses it.

use crate::Error;
use crate::cut::{self, CellCut, CellInput, Curve, GAUSS2, GAUSS3, P2};
use crate::material::{Stress, Tensor};
use crate::solid::{NodeKind, Solid, identity_plus, invert};
use std::collections::HashMap;
use std::sync::Arc;

/// The smallest |n_d| of a surface point on a grid line along axis d.
const MIN_NORMAL: f64 = 0.3;
/// Control volumes smaller than this fraction of h² join a neighbour.
pub const MERGE_FRACTION: f64 = 0.25;
/// Width of the blend between the two line families, in |n_x| − |n_y| (±0.2 is about ±8°
/// around the diagonal).
const BLEND: f64 = 0.2;
/// Newton iterations of the surface conditions per step, at most.
const MAX_NEWTON: usize = 20;

/// What a grid cell is: no solid, all solid, or cut (index into the cut cells); or not a
/// leaf of this level of a refined hierarchy (assembled on another level).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CellClass {
    Empty,
    Full,
    Cut(usize),
    Foreign,
}

/// Where a grid line meets the surface.
#[derive(Debug, Clone)]
pub(crate) struct SurfacePoint {
    /// The line runs along axis d.
    d: usize,
    /// Index of the line across d.
    line: usize,
    /// +1 when the outside lies towards +d from n₁.
    sigma: f64,
    /// The inside nodes behind the surface on the line, nearest first.
    n1: usize,
    n2: usize,
    /// Depth of n₁ below the surface, in grid spacings (0 < s ≤ 1).
    s: f64,
    /// Position along d.
    coord: f64,
    /// Position of the line across d.
    across: f64,
    /// Whether the point carries a surface value (the line meets the surface steeply enough
    /// and has neighbours along the surface).
    valid: bool,
    /// The matched points on lines line − 2 ..= line + 2 (index 2 is this one).
    nb: [Option<usize>; 5],
    /// Outward unit normal of the surface curve.
    normal: P2,
    /// ζ at the surface.
    zeta: P2,
    /// ζ_S minus the linear extrapolation from n₂, n₁ (kept to warm-start the next step).
    deviation: P2,
}

impl SurfacePoint {
    fn position(&self) -> P2 {
        let mut x = [0.0; 2];
        x[self.d] = self.coord;
        x[1 - self.d] = self.across;
        x
    }
}

/// Counters of the last mapped step.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CutStats {
    /// Cells the surface crosses.
    pub cut_cells: usize,
    /// Surface points (crossings of grid lines with two inside nodes behind them).
    pub surface_points: usize,
    /// Surface points that carry a surface value.
    pub valid_points: usize,
    /// Faces integrated through the map.
    pub mapped_faces: usize,
    /// Gauss points on those faces where the map was unavailable and the compact stencil
    /// (with extrapolated neighbours) was used instead.
    pub fallback_points: usize,
    /// Control volumes merged into a neighbour for being small.
    pub merged: usize,
    /// Solid area that found no inside node to join (dropped).
    pub lost_area: f64,
    /// Newton iterations of the surface conditions.
    pub sweeps: usize,
    /// The last iteration's largest change of a surface value.
    pub last_change: f64,
    /// Banded LU factorisations of the surface conditions (one per chain and Jacobian).
    pub factorizations: usize,
    /// Evaluations of a surface point's traction condition.
    pub traction_evals: usize,
    /// The largest force component through one face in the last force evaluation.
    pub max_flux: f64,
}

/// The cut cells, control volumes and surface of one step.
#[derive(Debug, Clone)]
pub(crate) struct CutState {
    /// What depends only on which nodes are in the body (shared by the states of one held
    /// topology).
    structure: Arc<Structure>,
    class: Arc<Vec<CellClass>>,
    cells: Vec<(usize, CellCut)>,
    points: Vec<SurfacePoint>,
    /// Surface points by family d and line index.
    lines: [Vec<Vec<usize>>; 2],
    /// The inside node whose control volume holds (most of) each node's solid.
    owner: Vec<Option<usize>>,
    /// How each node's solid, mass and net force are shared between inside nodes (weights
    /// summing to one; the second entry has weight 0 when one node takes all).
    share: Vec<Option<[(usize, f64); 2]>>,
    /// Solid area of each node's dual cell.
    area: Vec<f64>,
    /// ∫ x dA over each node's dual cell.
    moment: Vec<P2>,
    /// Nodes near the surface (corners of cut cells, nodes behind surface points).
    near: Vec<bool>,
    /// Small control volumes merged into a neighbour: (node, the node it joined), in order.
    merges: Vec<(usize, usize)>,
    /// The chains of surface points and the factorised Jacobian of each chain's conditions.
    chains: Vec<Vec<usize>>,
    lus: Vec<Option<Arc<BandLu>>>,
    stats: CutStats,
}

/// The part of a cut state fixed by the node kinds alone: the cell classes, the grid edges the
/// surface crosses, the cut cells and the edges of their crossings, the solid of the full
/// cells in each dual cell, and the outside nodes whose dual cell holds solid. A static solve
/// that holds the topology rebuilds only the geometry on top of it (crossings, cut cells,
/// near-surface areas, shares, surface points), not the whole grid.
#[derive(Debug, Default)]
struct Structure {
    kind: Vec<NodeKind>,
    /// Edges (k, d) between an inside and an outside node, d-major then k ascending.
    cross: Vec<(usize, usize)>,
    /// Cut cells: lower-left node and, for edges e0..e3, the index of the crossing in `cross`.
    cells: Vec<(usize, [Option<usize>; 4])>,
    area_full: Vec<f64>,
    moment_full: Vec<P2>,
    /// Corners of cut cells.
    near_cells: Vec<bool>,
    /// Outside nodes whose dual cell holds solid.
    orphans: Vec<usize>,
    /// The faces with solid on them, in node order (k, then d): the compact stencil's nodes
    /// for a face between two full cells, `None` for a face next to a cut cell.
    faces: Vec<(usize, usize, Option<[usize; 5]>)>,
}

/// Stage one of a cut state: crossings, cells, control volumes and shares.
struct Geometry {
    structure: Arc<Structure>,
    class: Arc<Vec<CellClass>>,
    cross_t: Vec<f64>,
    cells: Vec<(usize, CellCut)>,
    area: Vec<f64>,
    moment: Vec<P2>,
    near: Vec<bool>,
    share: Vec<Option<[(usize, f64); 2]>>,
    owner: Vec<Option<usize>>,
    merges: Vec<(usize, usize)>,
}

/// The discrete choices of a cut state: which nodes are in the body, which surface points
/// carry a value and how they are matched along the surface, and which control volumes are
/// merged. With these held, the forces are continuous functions of ζ (crossings, areas,
/// normals and surface values move, the structure does not), which is what a Newton solve
/// needs; a static solve holds them through each linearisation and updates them between
/// Newton steps.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Topology {
    pub(crate) kind: Vec<NodeKind>,
    /// Per surface point in build order: (n₁, d, outside towards +d), valid, matches.
    points: Vec<((usize, usize, bool), bool, [Option<usize>; 5])>,
    merges: Vec<(usize, usize)>,
}

/// The stress at a point of the surface.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceSample {
    /// Position.
    pub position: [f64; 2],
    /// Outward unit normal.
    pub normal: [f64; 2],
    /// The solid's Cauchy stress there, from the surface-fitted map.
    pub stress: Stress,
}

/// One cut cell for plotting: its lower-left node, the surface pieces (p, m, q) in grid
/// coordinates and the solid area of each quarter with the node that owns it.
#[derive(Debug, Clone, PartialEq)]
pub struct CutCellView {
    /// The cell's lower-left node.
    pub node: usize,
    /// Surface pieces, each through (p, m, q).
    pub arcs: Vec<[[f64; 2]; 3]>,
    /// Solid area of each quarter (c0..c3 counter-clockwise from the lower left).
    pub quarter_area: [f64; 4],
    /// The node owning each quarter's solid.
    pub quarter_owner: [Option<usize>; 4],
}

/// One surface point for plotting.
#[derive(Debug, Clone, PartialEq)]
pub struct SurfacePointView {
    /// Position.
    pub position: [f64; 2],
    /// The grid line runs along this axis.
    pub axis: usize,
    /// +1 when the outside lies towards +axis.
    pub sign: f64,
    /// Depth of the first inside node below the surface, in grid spacings.
    pub depth: f64,
    /// The two inside nodes behind it.
    pub nodes: [usize; 2],
    /// Whether it carries a surface value.
    pub valid: bool,
    /// The point on the next line (+1) it is matched with, by index in the view.
    pub next: Option<usize>,
}

/// The cut cells and surface points of the last mapped step.
#[derive(Debug, Clone, PartialEq)]
pub struct CutView {
    /// Cut cells.
    pub cells: Vec<CutCellView>,
    /// Surface points.
    pub points: Vec<SurfacePointView>,
    /// Counters.
    pub stats: CutStats,
}

/// Lagrange weights and their derivatives at x for the nodes xs (at most four).
fn lagrange(xs: &[f64], x: f64) -> ([f64; 4], [f64; 4]) {
    let mut w = [0.0; 4];
    let mut dw = [0.0; 4];
    let n = xs.len().min(4);
    for i in 0..n {
        let mut p = 1.0;
        for j in 0..n {
            if j != i {
                p *= (x - xs[j]) / (xs[i] - xs[j]);
            }
        }
        w[i] = p;
        let mut s = 0.0;
        for m in 0..n {
            if m == i {
                continue;
            }
            let mut q = 1.0 / (xs[i] - xs[m]);
            for j in 0..n {
                if j != i && j != m {
                    q *= (x - xs[j]) / (xs[i] - xs[j]);
                }
            }
            s += q;
        }
        dw[i] = s;
    }
    (w, dw)
}

/// 3z² − 2z³ on [0, 1], clamped.
fn smoothstep(z: f64) -> f64 {
    let z = z.clamp(0.0, 1.0);
    z * z * (3.0 - 2.0 * z)
}

fn lerp(a: P2, b: P2, t: f64) -> P2 {
    [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]
}

fn traction(s: &[f64; 3], n: P2) -> P2 {
    [s[0] * n[0] + s[2] * n[1], s[2] * n[0] + s[1] * n[1]]
}

impl CutState {
    /// The discrete choices of this state, with the node kinds it was built for.
    pub(crate) fn topology(&self, kind: &[NodeKind]) -> Topology {
        Topology {
            kind: kind.to_vec(),
            points: self.points.iter().map(|p| ((p.n1, p.d, p.sigma > 0.0), p.valid, p.nb)).collect(),
            merges: self.merges.clone(),
        }
    }

    /// The node holding the control volume of node k, when k's solid belongs to one.
    pub(crate) fn owner(&self, k: usize) -> Option<usize> {
        self.owner.get(k).copied().flatten()
    }

    /// The solid area of the control volume held by each node.
    pub(crate) fn control_volumes(&self) -> Vec<f64> {
        let mut cv = vec![0.0; self.area.len()];
        for (k, sh) in self.share.iter().enumerate() {
            for (o, w) in sh.iter().flatten() {
                cv[*o] += w * self.area[k];
            }
        }
        cv
    }

    /// Nodes where the discretisation departs from the interior stencil: corners of cut cells,
    /// nodes behind surface points, outside nodes whose solid is shared out, merged nodes and
    /// the nodes they joined.
    pub(crate) fn special_nodes(&self) -> Vec<bool> {
        let mut out: Vec<bool> = self.near.iter().zip(&self.structure.near_cells).map(|(a, b)| *a || *b).collect();
        for &k in &self.structure.orphans {
            out[k] = true;
        }
        for &(k, m) in &self.merges {
            out[k] = true;
            out[m] = true;
        }
        out
    }

    /// The faces between two full cells with their compact stencil nodes [b, k₊, k₋, b₊, b₋].
    pub(crate) fn full_faces(&self) -> impl Iterator<Item = (usize, usize, [usize; 5])> + '_ {
        self.structure.faces.iter().filter_map(|&(k, d, f)| f.map(|nodes| (k, d, nodes)))
    }

    /// How the solid of node k's dual cell is shared between control volumes.
    pub(crate) fn share(&self, k: usize) -> Option<[(usize, f64); 2]> {
        self.share.get(k).copied().flatten()
    }

    /// Every face with solid on it, (k, d, solid length / h).
    pub(crate) fn solid_faces(&self, solid: &Solid) -> Vec<(usize, usize, f64)> {
        let h = solid.grid.h();
        self.structure
            .faces
            .iter()
            .map(|&(k, d, full)| {
                if full.is_some() {
                    return (k, d, 1.0);
                }
                let lower = if d == 0 { solid.grid.neighbour(k, 0, -1).map(|c| (c, 2)) } else { solid.grid.neighbour(k, -1, 0).map(|c| (c, 1)) };
                let mut len = 0.0;
                for (c0, hf) in [lower, Some((k, if d == 0 { 0 } else { 3 }))].into_iter().flatten() {
                    match self.class[c0] {
                        CellClass::Empty | CellClass::Foreign => {}
                        CellClass::Full => len += 0.5 * h,
                        CellClass::Cut(i) => {
                            for (a, b) in &self.cells[i].1.faces[hf] {
                                len += (b[0] - a[0]).hypot(b[1] - a[1]);
                            }
                        }
                    }
                }
                (k, d, len / h)
            })
            .collect()
    }

    /// The counters of this state.
    pub(crate) fn stats(&self) -> &CutStats {
        &self.stats
    }

    /// ∇ζ at node k from the surface-fitted map, for nodes near the surface.
    pub(crate) fn node_gradient(&self, solid: &Solid, k: usize) -> Option<Tensor> {
        if !*self.near.get(k)? {
            return None;
        }
        solid.reconstruct(self, solid.grid.position(k))
    }
}

impl Solid {
    /// The stress at the surface points of the current configuration, from the surface-fitted
    /// map (empty for the blurred boundary or before the first step).
    pub fn surface_stress(&self) -> Vec<SurfaceSample> {
        let Some(st) = &self.cut else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for (p, pt) in st.points.iter().enumerate() {
            if !pt.valid {
                continue;
            }
            // The blended map the fluxes use (a point on lines that meet the surface obliquely
            // is weighted out where the other family is steeper).
            let Some(g) = self.reconstruct(st, pt.position()).or_else(|| self.surface_gradient(&st.points, p, pt.zeta)) else {
                continue;
            };
            let Some(f) = invert(&identity_plus(&g)) else {
                continue;
            };
            if let Ok(stress) = self.material.cauchy(&f) {
                out.push(SurfaceSample { position: pt.position(), normal: pt.normal, stress });
            }
        }
        out
    }

    /// The cut cells and surface points of the last mapped step, for plotting.
    pub fn cut_view(&self) -> Option<CutView> {
        let st = self.cut.as_ref()?;
        let cells = st
            .cells
            .iter()
            .map(|(c0, cell)| {
                let o = self.grid.position(*c0);
                let g = |p: P2| [o[0] + p[0], o[1] + p[1]];
                let corners = self.cell_corners(*c0).unwrap_or([*c0; 4]);
                CutCellView {
                    node: *c0,
                    arcs: cell.arcs.iter().map(|a| [g(a.p), g(a.m), g(a.q)]).collect(),
                    quarter_area: cell.area,
                    quarter_owner: corners.map(|c| st.owner[c]),
                }
            })
            .collect();
        let points = st
            .points
            .iter()
            .map(|p| SurfacePointView {
                position: p.position(),
                axis: p.d,
                sign: p.sigma,
                depth: p.s,
                nodes: [p.n1, p.n2],
                valid: p.valid,
                next: p.nb[3].filter(|&q| st.points[q].valid),
            })
            .collect();
        Some(CutView { cells, points, stats: st.stats.clone() })
    }

    /// The counters of the last mapped step.
    pub fn cut_stats(&self) -> Option<CutStats> {
        self.cut.as_ref().map(|c| c.stats.clone())
    }

    fn inside(&self, k: usize) -> bool {
        self.kind[k].source()
    }

    fn along(&self, k: usize, d: usize, o: isize) -> Option<usize> {
        if d == 0 { self.grid.neighbour(k, o, 0) } else { self.grid.neighbour(k, 0, o) }
    }

    /// The corners c0..c3 of the cell whose lower-left node is c0 (wrapping periodic axes).
    fn cell_corners(&self, c0: usize) -> Option<[usize; 4]> {
        Some([c0, self.grid.neighbour(c0, 1, 0)?, self.grid.neighbour(c0, 1, 1)?, self.grid.neighbour(c0, 0, 1)?])
    }

    /// Where the surface crosses the edge from a to b = a + e_d, as a fraction from a: the root
    /// of φ₀(x + ζ(x)) along the edge, ζ quadratic through the inside node, the node behind it
    /// and the outside node (linear without the node behind).
    fn edge_crossing(&self, a: usize, b: usize, d: usize) -> Result<f64, Error> {
        let (n_in, n_out, sgn) = if self.inside(a) { (a, b, 1.0) } else { (b, a, -1.0) };
        if !self.kind[n_out].known() {
            return Err(Error::Diverged(format!("the extrapolation band does not reach the surface at node {n_out}")));
        }
        let h = self.grid.h();
        let x_in = self.grid.position(n_in);
        let z1 = self.zeta[n_in];
        let zo = self.zeta[n_out];
        let z2 = self.along(n_in, d, if sgn > 0.0 { -1 } else { 1 }).filter(|&m| self.inside(m)).map(|m| self.zeta[m]);
        let f = |t: f64| {
            let mut xi = x_in;
            xi[d] += sgn * t * h;
            for c in 0..2 {
                let z = match z2 {
                    Some(z2) => z1[c] + 0.5 * t * (zo[c] - z2[c]) + 0.5 * t * t * (zo[c] - 2.0 * z1[c] + z2[c]),
                    None => z1[c] + t * (zo[c] - z1[c]),
                };
                xi[c] += z;
            }
            self.shape.phi0(xi)
        };
        // φ ≥ 0 at the outside node; exactly 0 puts the surface on it. A held node kind that no
        // longer matches φ (a static solve between topology updates) puts the surface on the
        // node that crossed, so the crossing moves continuously to the end of the edge.
        let f1 = f(1.0);
        let t = if !(self.phi[n_in] < 0.0) {
            0.0
        } else if f1 > 0.0 {
            cut::bracketed_root(&f, 0.0, 1.0, self.phi[n_in], f1, 1e-14)
        } else {
            1.0
        };
        Ok(if self.inside(a) { t } else { 1.0 - t })
    }

    /// ∇φ at node k by central differences of the level set (one-sided next to unknown nodes).
    fn phi_gradient(&self, k: usize) -> P2 {
        let h = self.grid.h();
        let mut g = [0.0; 2];
        for (c, gc) in g.iter_mut().enumerate() {
            let p = self.along(k, c, 1).filter(|&m| self.kind[m].known());
            let m = self.along(k, c, -1).filter(|&m| self.kind[m].known());
            *gc = match (p, m) {
                (Some(p), Some(m)) => (self.phi[p] - self.phi[m]) / (2.0 * h),
                (Some(p), None) => (self.phi[p] - self.phi[k]) / h,
                (None, Some(m)) => (self.phi[k] - self.phi[m]) / h,
                (None, None) => 0.0,
            };
        }
        g
    }

    /// The ambient stress at x (in the cell with lower-left node c0): the field when given,
    /// else bilinear in the cell's node values.
    fn ambient_at(&self, x: P2, c0: Option<usize>) -> [f64; 3] {
        let s = self.ambient_unscaled(x, c0);
        [self.load * s[0], self.load * s[1], self.load * s[2]]
    }

    fn ambient_unscaled(&self, x: P2, c0: Option<usize>) -> [f64; 3] {
        if let Some(f) = &self.ambient_field {
            return (f.0)(x);
        }
        let Some(c0) = c0 else {
            return [0.0; 3];
        };
        let Some(c) = self.cell_corners(c0) else {
            return self.ambient[c0];
        };
        let o = self.grid.position(c0);
        let h = self.grid.h();
        let (u, v) = (((x[0] - o[0]) / h).clamp(0.0, 1.0), ((x[1] - o[1]) / h).clamp(0.0, 1.0));
        let w = [(1.0 - u) * (1.0 - v), u * (1.0 - v), u * v, (1.0 - u) * v];
        let mut s = [0.0; 3];
        for q in 0..4 {
            for r in 0..3 {
                s[r] += w[q] * self.ambient[c[q]][r];
            }
        }
        s
    }

    /// Builds the cut cells, the control volumes and the surface, and solves the surface
    /// values.
    pub(crate) fn build_cut(&self) -> Result<CutState, Error> {
        self.build_cut_from(self.cut.as_ref())
    }

    /// As [`Solid::build_cut`], warm-starting the surface values from `prev`.
    fn build_cut_from(&self, prev: Option<&CutState>) -> Result<CutState, Error> {
        self.build_cut_with(prev, None, false, true)
    }

    /// Builds the cut state, warm-starting the surface values from `prev`. With `frozen`, the
    /// surface points' validity and matches and the merges are taken from it instead of being
    /// decided (the node kinds must already be its kinds). With `reuse_lu` and a frozen
    /// topology, the surface conditions are solved by the chord method with the factorised
    /// Jacobians of `prev` (refreshed when a chain converges slowly); without it every
    /// iteration refactorises, as in the dynamic step. Without `solve` the surface values are
    /// left at their warm start (for reading the topology only).
    pub(crate) fn build_cut_with(&self, prev: Option<&CutState>, frozen: Option<&Topology>, reuse_lu: bool, solve: bool) -> Result<CutState, Error> {
        let n = self.grid.len();
        let h = self.grid.h();
        let mut stats = CutStats::default();
        let geo = match (frozen, prev) {
            (Some(t), Some(p)) if p.structure.kind == self.kind => self.geometry_held(p, t)?,
            _ => self.geometry_full(frozen)?,
        };
        let Geometry { structure, class, cross_t, cells, area, moment, mut near, share, owner, merges } = geo;
        stats.cut_cells = cells.len();
        stats.merged = merges.len();
        stats.lost_area = structure.orphans.iter().filter(|&&k| share[k].is_none()).map(|&k| area[k]).sum();

        // Surface points.
        // Warm start: the previous step's surface value as a deviation from the linear
        // extrapolation along its line (which follows the nodes as they move).
        let linear = |n1: usize, n2: usize, s: f64| -> P2 {
            let (z1, z2) = (self.zeta[n1], self.zeta[n2]);
            [z1[0] + s * (z1[0] - z2[0]), z1[1] + s * (z1[1] - z2[1])]
        };
        let previous: HashMap<(usize, usize, bool), P2> = match prev {
            Some(prev) => prev.points.iter().map(|p| ((p.n1, p.d, p.sigma > 0.0), p.deviation)).collect(),
            None => HashMap::new(),
        };
        let mut points = Vec::new();
        let mut lines: [Vec<Vec<usize>>; 2] = [vec![Vec::new(); self.grid.ny()], vec![Vec::new(); self.grid.nx()]];
        for (&(k, d), &t) in structure.cross.iter().zip(&cross_t) {
            {
                let Some(b) = self.along(k, d, 1) else {
                    continue;
                };
                let (n1, out, sigma, s) = if self.inside(k) { (k, b, 1.0, t) } else { (b, k, -1.0, 1.0 - t) };
                let Some(n2) = self.along(n1, d, if sigma > 0.0 { -1 } else { 1 }).filter(|&m| self.inside(m)) else {
                    continue;
                };
                let s = s.max(1e-9);
                let p1 = self.grid.position(n1);
                let g = lerp(self.phi_gradient(n1), self.phi_gradient(out), s);
                let gl = g[0].hypot(g[1]);
                let valid = gl > 0.0 && sigma * g[d] / gl >= MIN_NORMAL;
                let lin = linear(n1, n2, s);
                let dev = previous.get(&(n1, d, sigma > 0.0)).copied().unwrap_or([0.0; 2]);
                let zeta = [lin[0] + dev[0], lin[1] + dev[1]];
                let (i, j) = self.grid.ij(n1);
                let line = if d == 0 { j } else { i };
                lines[d][line].push(points.len());
                points.push(SurfacePoint {
                    d,
                    line,
                    sigma,
                    n1,
                    n2,
                    s,
                    coord: p1[d] + sigma * s * h,
                    across: p1[1 - d],
                    valid,
                    nb: [None; 5],
                    normal: [0.0; 2],
                    zeta,
                    deviation: dev,
                });
            }
        }
        stats.surface_points = points.len();
        if let Some(t) = frozen {
            if t.points.len() != points.len() {
                return Err(Error::Diverged("the held cut topology does not match the surface".into()));
            }
            for (pt, (key, valid, nb)) in points.iter_mut().zip(&t.points) {
                if *key != (pt.n1, pt.d, pt.sigma > 0.0) {
                    return Err(Error::Diverged("the held cut topology does not match the surface".into()));
                }
                pt.valid = *valid;
                pt.nb = *nb;
            }
        }
        // Match each point with the points of the same family and orientation on the
        // neighbouring lines (the nearest along the line), then drop points without neighbours.
        for _ in 0..if frozen.is_some() { 0 } else { 2 } {
            for p in 0..points.len() {
                let mut nb = [None; 5];
                if points[p].valid {
                    nb[2] = Some(p);
                    let (d, line, sigma, coord) = (points[p].d, points[p].line as isize, points[p].sigma, points[p].coord);
                    for off in [-2isize, -1, 1, 2] {
                        let l = line + off;
                        if l < 0 || l as usize >= lines[d].len() {
                            continue;
                        }
                        let mut best: Option<(usize, f64)> = None;
                        for &q in &lines[d][l as usize] {
                            let pq = &points[q];
                            if !pq.valid || pq.sigma != sigma {
                                continue;
                            }
                            let gap = (pq.coord - coord).abs();
                            if gap <= 3.5 * off.unsigned_abs() as f64 * h && best.is_none_or(|b| gap < b.1) {
                                best = Some((q, gap));
                            }
                        }
                        nb[(off + 2) as usize] = best.map(|b| b.0);
                    }
                }
                points[p].nb = nb;
            }
            for pt in points.iter_mut() {
                if pt.valid && pt.nb[1].is_none() && pt.nb[3].is_none() {
                    pt.valid = false;
                }
            }
        }
        // Normals from the surface curve through the matched crossings.
        for p in 0..points.len() {
            if !points[p].valid {
                continue;
            }
            let Some((offs, cnt)) = tangent_stencil(&points[p].nb) else {
                points[p].valid = false;
                continue;
            };
            let xs = offs.map(|o| o as f64);
            let (_, dw) = lagrange(&xs[..cnt], 0.0);
            let mut yp = 0.0;
            for m in 0..cnt {
                if let Some(q) = points[p].nb[(offs[m] + 2) as usize] {
                    yp += dw[m] * points[q].coord;
                }
            }
            let (d, sigma) = (points[p].d, points[p].sigma);
            let len = h.hypot(yp);
            let mut nrm = [0.0; 2];
            nrm[d] = sigma * h / len;
            nrm[1 - d] = -sigma * yp / len;
            points[p].normal = nrm;
        }
        for pt in &points {
            if pt.valid {
                near[pt.n1] = true;
                near[pt.n2] = true;
            }
        }
        stats.valid_points = points.iter().filter(|p| p.valid).count();

        // Surface values from the traction condition: Newton on the coupled conditions of each
        // chain of matched points (one family's points on consecutive lines), the Jacobian by
        // finite differences, banded (a point's condition involves the points up to two lines
        // away), solved by LU with partial pivoting.
        let tol = 1e-13 * h;
        let targets: Vec<P2> = points
            .iter()
            .map(|pt| {
                if pt.valid {
                    let x = pt.position();
                    traction(&self.ambient_at(x, self.cell_of(x)), pt.normal)
                } else {
                    [0.0; 2]
                }
            })
            .collect();
        let chains = surface_chains(&points);
        let chord = reuse_lu && frozen.is_some();
        let mut lus: Vec<Option<Arc<BandLu>>> = match prev {
            Some(p) if chord && p.chains == chains => p.lus.clone(),
            _ => vec![None; chains.len()],
        };
        let mut last = vec![f64::INFINITY; chains.len()];
        for it in 0..if solve { MAX_NEWTON } else { 0 } {
            let mut change: f64 = 0.0;
            for (ci, ch) in chains.iter().enumerate() {
                let m = ch.len();
                let mut r = vec![0.0; 2 * m];
                let mut folded = false;
                stats.traction_evals += m;
                for (i, &p) in ch.iter().enumerate() {
                    match self.traction_residual(&points, p, points[p].zeta, targets[p]) {
                        Ok(v) => {
                            r[2 * i] = v[0];
                            r[2 * i + 1] = v[1];
                        }
                        Err(_) => {
                            // The guess folds the map: restart from the linear extrapolation.
                            let lin = linear(points[p].n1, points[p].n2, points[p].s);
                            if points[p].zeta == lin {
                                return Err(Error::Diverged(format!("the map folded at the surface by node {}", points[p].n1)));
                            }
                            points[p].zeta = lin;
                            folded = true;
                        }
                    }
                }
                if folded {
                    change = f64::INFINITY;
                    lus[ci] = None;
                    continue;
                }
                let lu = match lus.get(ci).and_then(|l| l.clone()) {
                    Some(lu) if chord => lu,
                    _ => {
                        let mut jac = Band::new(2 * m, 5, 5);
                        for (i, &p) in ch.iter().enumerate() {
                            let eps = 1e-7 * h * points[p].s.max(1e-3);
                            for c in 0..2 {
                                let z0 = points[p].zeta;
                                points[p].zeta[c] += eps;
                                for j in i.saturating_sub(2)..=(i + 2).min(m - 1) {
                                    let q = ch[j];
                                    stats.traction_evals += 1;
                                    let rq = self.traction_residual(&points, q, points[q].zeta, targets[q]);
                                    if let Ok(rq) = rq {
                                        for row in 0..2 {
                                            jac.set(2 * j + row, 2 * i + c, (rq[row] - r[2 * j + row]) / eps);
                                        }
                                    }
                                }
                                points[p].zeta = z0;
                            }
                        }
                        stats.factorizations += 1;
                        let lu = Arc::new(jac.factor().ok_or_else(|| Error::Diverged("the surface conditions are singular".into()))?);
                        lus[ci] = Some(lu.clone());
                        lu
                    }
                };
                let rhs: Vec<f64> = r.iter().map(|v| -v).collect();
                let dz = lu.solve(rhs).ok_or_else(|| Error::Diverged("the surface conditions are singular".into()))?;
                let mut chain_change: f64 = 0.0;
                for (i, &p) in ch.iter().enumerate() {
                    let d = [dz[2 * i], dz[2 * i + 1]];
                    if !(d[0].is_finite() && d[1].is_finite()) {
                        return Err(Error::Diverged(format!("the surface value at node {} did not converge", points[p].n1)));
                    }
                    points[p].zeta = [points[p].zeta[0] + d[0], points[p].zeta[1] + d[1]];
                    chain_change = chain_change.max(d[0].abs()).max(d[1].abs());
                }
                // A chord step that contracts slowly: refactorise at the next iteration.
                if chord && chain_change > tol && chain_change > 0.1 * last[ci] {
                    lus[ci] = None;
                }
                last[ci] = chain_change;
                change = change.max(chain_change);
            }
            stats.sweeps = it + 1;
            stats.last_change = change;
            if change <= tol {
                break;
            }
        }
        for pt in points.iter_mut() {
            let lin = linear(pt.n1, pt.n2, pt.s);
            pt.deviation = [pt.zeta[0] - lin[0], pt.zeta[1] - lin[1]];
        }

        let _ = n;
        Ok(CutState { structure, class, cells, points, lines, owner, share, area, moment, near, merges, chains, lus, stats })
    }

    /// The cut cell at lower-left node c0 with corners c and edge crossings cr.
    fn cut_geometry(&self, c0: usize, c: [usize; 4], cr: [Option<f64>; 4]) -> Result<CellCut, Error> {
        let h = self.grid.h();
        let ins = c.map(|k| self.inside(k));
        let origin = self.grid.position(c0);
        let zc = c.map(|k| self.zeta[k]);
        let shape = self.shape;
        let phi = move |x: P2| {
            let (u, v) = (x[0] / h, x[1] / h);
            let w = [(1.0 - u) * (1.0 - v), u * (1.0 - v), u * v, (1.0 - u) * v];
            let mut xi = [origin[0] + x[0], origin[1] + x[1]];
            for q in 0..4 {
                xi[0] += w[q] * zc[q][0];
                xi[1] += w[q] * zc[q][1];
            }
            shape.phi0(xi)
        };
        cut::cut_cell(&CellInput { h, inside: ins, cross: cr, phi: &phi })
            .ok_or_else(|| Error::Diverged(format!("inconsistent surface crossings in the cell at node {c0}")))
    }

    /// Adds the solid of a cut cell's quarters to its corners' dual cells.
    fn add_cut_cell(&self, c0: usize, cell: &CellCut, area: &mut [f64], moment: &mut [P2]) {
        let Some(c) = self.cell_corners(c0) else {
            return;
        };
        let o = self.grid.position(c0);
        for (q, &k) in c.iter().enumerate() {
            area[k] += cell.area[q];
            moment[k][0] += cell.moment[q][0] + cell.area[q] * o[0];
            moment[k][1] += cell.moment[q][1] + cell.area[q] * o[1];
        }
    }

    /// Stage one from scratch: scans the whole grid.
    fn geometry_full(&self, frozen: Option<&Topology>) -> Result<Geometry, Error> {
        let n = self.grid.len();
        let h = self.grid.h();
        // Crossings on every grid edge between an inside and an outside node.
        let mut cross_at = vec![[None::<usize>; 2]; n];
        let mut cross = Vec::new();
        let mut cross_t = Vec::new();
        for d in 0..2 {
            for k in 0..n {
                if let Some(b) = self.along(k, d, 1)
                    && self.inside(k) != self.inside(b)
                {
                    cross_at[k][d] = Some(cross.len());
                    cross.push((k, d));
                    cross_t.push(self.edge_crossing(k, b, d)?);
                }
            }
        }
        // Cells.
        let mut class = vec![CellClass::Empty; n];
        let mut cell_list = Vec::new();
        let mut cells = Vec::new();
        for c0 in 0..n {
            let Some(c) = self.cell_corners(c0) else {
                continue;
            };
            if !self.own_cell(c0) {
                class[c0] = CellClass::Foreign;
                continue;
            }
            let ins = c.map(|k| self.inside(k));
            if ins.iter().all(|&b| b) {
                class[c0] = CellClass::Full;
                continue;
            }
            if !ins.iter().any(|&b| b) {
                continue;
            }
            if c.iter().any(|&k| !self.kind[k].known()) {
                return Err(Error::Diverged(format!("a cut cell at node {c0} has a corner outside the extrapolation band")));
            }
            let edges = [cross_at[c[0]][0], cross_at[c[1]][1], cross_at[c[3]][0], cross_at[c[0]][1]];
            let cr = crossings_of(&edges, &cross_t);
            let cell = self.cut_geometry(c0, c, cr)?;
            class[c0] = CellClass::Cut(cells.len());
            cell_list.push((c0, edges));
            cells.push((c0, cell));
        }
        // Solid area and first moment of each node's dual cell: the full cells, then the cut ones.
        let mut area_full = vec![0.0; n];
        let mut moment_full = vec![[0.0; 2]; n];
        let mut near_cells = vec![false; n];
        let quarter = 0.25 * h * h;
        for c0 in 0..n {
            let Some(c) = self.cell_corners(c0) else {
                continue;
            };
            let o = self.grid.position(c0);
            match class[c0] {
                CellClass::Empty | CellClass::Foreign => {}
                CellClass::Full => {
                    for (q, &k) in c.iter().enumerate() {
                        let qc = match q {
                            0 => [0.25 * h, 0.25 * h],
                            1 => [0.75 * h, 0.25 * h],
                            2 => [0.75 * h, 0.75 * h],
                            _ => [0.25 * h, 0.75 * h],
                        };
                        area_full[k] += quarter;
                        moment_full[k][0] += quarter * (o[0] + qc[0]);
                        moment_full[k][1] += quarter * (o[1] + qc[1]);
                    }
                }
                CellClass::Cut(_) => {
                    for &k in &c {
                        near_cells[k] = true;
                    }
                }
            }
        }
        let mut area = area_full.clone();
        let mut moment = moment_full.clone();
        for (c0, cell) in &cells {
            self.add_cut_cell(*c0, cell, &mut area, &mut moment);
        }

        // Shares: inside nodes keep their control volume; solid in the dual cell of an outside
        // node goes to inside neighbours.
        let mut share: Vec<Option<[(usize, f64); 2]>> = vec![None; n];
        let mut orphans = Vec::new();
        for k in 0..n {
            if !(area[k] > 0.0) {
                continue;
            }
            // A hanging node of a refined level gives its solid to its two masters.
            let slave = self.level.as_ref().and_then(|l| l.slave.get(k).copied().flatten());
            share[k] = if let Some(sl) = slave {
                Some(sl)
            } else if self.inside(k) {
                Some([(k, 1.0), (k, 0.0)])
            } else {
                orphans.push(k);
                self.orphan_shares(k)
            };
        }
        // Small control volumes join their deepest inside edge neighbour (never one the
        // hierarchy sets, nor one on the interface with the coarser level).
        let mut cv = self.level.as_ref().map_or_else(|| vec![0.0; n], |l| l.ext_area.clone());
        cv.resize(n, 0.0);
        for k in 0..n {
            for (o, w) in share[k].into_iter().flatten() {
                cv[o] += w * area[k];
            }
        }
        let mut merges = Vec::new();
        let mut merge = |k: usize, m: usize, share: &mut Vec<Option<[(usize, f64); 2]>>, cv: &mut Vec<f64>| {
            for sh in share.iter_mut().flatten() {
                for e in sh.iter_mut() {
                    if e.0 == k {
                        e.0 = m;
                    }
                }
            }
            cv[m] += cv[k];
            cv[k] = 0.0;
            merges.push((k, m));
        };
        if let Some(t) = frozen {
            for &(k, m) in &t.merges {
                merge(k, m, &mut share, &mut cv);
            }
        } else {
            for k in 0..n {
                let own = share[k].is_some_and(|sh| sh[0] == (k, 1.0));
                let interface = self.level.as_ref().is_some_and(|l| l.master.get(k).copied().unwrap_or(false));
                if !own || self.kind[k] == NodeKind::Fixed || self.frozen(k) || interface || cv[k] >= MERGE_FRACTION * h * h {
                    continue;
                }
                let mut best: Option<usize> = None;
                for (di, dj) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                    if let Some(m) = self.grid.neighbour(k, di, dj)
                        && share[m].is_some_and(|sh| sh[0] == (m, 1.0))
                        && !self.frozen(m)
                        && best.is_none_or(|b| self.phi[m] < self.phi[b])
                        && self.phi[m] < self.phi[k]
                    {
                        best = Some(m);
                    }
                }
                if let Some(m) = best {
                    merge(k, m, &mut share, &mut cv);
                }
            }
        }
        let owner: Vec<Option<usize>> = share.iter().map(|sh| sh.map(owner_of)).collect();
        let near = near_cells.clone();
        // Faces with solid: both halves full (the compact stencil), or one of them cut.
        let mut faces = Vec::new();
        for k in 0..n {
            for d in 0..2 {
                if self.along(k, d, 1).is_none() {
                    continue;
                }
                let lower = if d == 0 { self.grid.neighbour(k, 0, -1) } else { self.grid.neighbour(k, -1, 0) };
                let classes = [lower.map_or(CellClass::Empty, |c| class[c]), class[k]];
                if classes.iter().all(|c| matches!(c, CellClass::Empty | CellClass::Foreign)) {
                    continue;
                }
                let full = classes.iter().all(|c| *c == CellClass::Full);
                faces.push((k, d, if full { self.compact_nodes(k, d) } else { None }));
            }
        }
        let class = Arc::new(class);
        let structure = Arc::new(Structure { kind: self.kind.clone(), cross, cells: cell_list, area_full, moment_full, near_cells, orphans, faces });
        Ok(Geometry { structure, class, cross_t, cells, area, moment, near, share, owner, merges })
    }

    /// Stage one on the structure of `base` (same node kinds) with the topology `topo` held:
    /// only the crossings, the cut cells, the near-surface areas and the orphans' shares are
    /// recomputed.
    fn geometry_held(&self, base: &CutState, topo: &Topology) -> Result<Geometry, Error> {
        let structure = base.structure.clone();
        let mut cross_t = Vec::with_capacity(structure.cross.len());
        for &(k, d) in &structure.cross {
            let b = self.along(k, d, 1).ok_or_else(|| Error::Diverged("a held crossing left the grid".into()))?;
            cross_t.push(self.edge_crossing(k, b, d)?);
        }
        let mut cells = Vec::with_capacity(structure.cells.len());
        let mut area = structure.area_full.clone();
        let mut moment = structure.moment_full.clone();
        for &(c0, edges) in &structure.cells {
            let c = self.cell_corners(c0).ok_or_else(|| Error::Diverged("a held cut cell left the grid".into()))?;
            let cell = self.cut_geometry(c0, c, crossings_of(&edges, &cross_t))?;
            self.add_cut_cell(c0, &cell, &mut area, &mut moment);
            cells.push((c0, cell));
        }
        let mut share = base.share.clone();
        let mut owner = base.owner.clone();
        for &k in &structure.orphans {
            // The orphan's shares, then the held merges in order.
            let mut sh = self.orphan_shares(k);
            if let Some(s) = sh.as_mut() {
                for &(from, to) in &topo.merges {
                    for e in s.iter_mut() {
                        if e.0 == from {
                            e.0 = to;
                        }
                    }
                }
            }
            share[k] = sh;
            owner[k] = sh.map(owner_of);
        }
        let near = structure.near_cells.clone();
        Ok(Geometry { class: base.class.clone(), structure, cross_t, cells, area, moment, near, share, owner, merges: topo.merges.clone() })
    }

    /// The cell (lower-left node) containing x, clamped to the grid; `None` when that node is
    /// not active on a sparse grid.
    fn cell_of(&self, x: P2) -> Option<usize> {
        let o = self.grid.origin();
        let h = self.grid.h();
        let i = (((x[0] - o[0]) / h).floor().max(0.0) as usize).min(self.grid.nx().saturating_sub(2));
        let j = (((x[1] - o[1]) / h).floor().max(0.0) as usize).min(self.grid.ny().saturating_sub(2));
        self.grid.try_index(i, j)
    }

    /// How the solid of the dual cell of outside node m is shared: between the inside
    /// neighbours across the surface along x and along y, weighted smoothly by |∂φ/∂x| against
    /// |∂φ/∂y| (all to one away from the diagonal); else the deepest inside edge neighbour,
    /// else the deepest diagonal one.
    fn orphan_shares(&self, m: usize) -> Option<[(usize, f64); 2]> {
        let g = self.phi_gradient(m);
        let cand = |d: usize| -> Option<usize> {
            if g[d] == 0.0 {
                return None;
            }
            self.along(m, d, if g[d] > 0.0 { -1 } else { 1 }).filter(|&k| self.inside(k))
        };
        match (cand(0), cand(1)) {
            (Some(a), Some(b)) => {
                let gl = g[0].hypot(g[1]);
                let w = smoothstep(((g[0].abs() - g[1].abs()) / (gl * BLEND) + 1.0) * 0.5);
                return Some([(a, w), (b, 1.0 - w)]);
            }
            (Some(a), None) | (None, Some(a)) => return Some([(a, 1.0), (a, 0.0)]),
            (None, None) => {}
        }
        let deepest = |offsets: &[(isize, isize)]| -> Option<usize> {
            let mut best: Option<usize> = None;
            for &(di, dj) in offsets {
                if let Some(k) = self.grid.neighbour(m, di, dj).filter(|&k| self.inside(k))
                    && best.is_none_or(|b| self.phi[k] < self.phi[b])
                {
                    best = Some(k);
                }
            }
            best
        };
        deepest(&[(1, 0), (-1, 0), (0, 1), (0, -1)]).or_else(|| deepest(&[(1, 1), (-1, 1), (1, -1), (-1, -1)])).map(|k| [(k, 1.0), (k, 0.0)])
    }

    /// σ(F)·n − t at surface point p with its surface value set to z.
    fn traction_residual(&self, points: &[SurfacePoint], p: usize, z: P2, t: P2) -> Result<P2, Error> {
        let g = self.surface_gradient(points, p, z).ok_or_else(|| Error::Diverged(format!("no surface map at node {}", points[p].n1)))?;
        let jinv = identity_plus(&g);
        let f = invert(&jinv).ok_or_else(|| Error::Diverged(format!("the reference map folded at the surface by node {}", points[p].n1)))?;
        let s = self.material.cauchy(&f).map_err(|e| Error::Diverged(format!("{e} at the surface by node {}", points[p].n1)))?;
        let tr = traction(&[s[0], s[1], s[2]], points[p].normal);
        Ok([tr[0] - t[0], tr[1] - t[1]])
    }

    /// ∇ζ at surface point p (η = 0) with its surface value set to z: the normal derivative
    /// from the one-sided quadratic along its line, the tangential one from the surface values
    /// of the neighbouring lines, through T′⁻¹.
    fn surface_gradient(&self, points: &[SurfacePoint], p: usize, z: P2) -> Option<Tensor> {
        let pt = &points[p];
        let (offs, cnt) = tangent_stencil(&pt.nb)?;
        let xs = offs.map(|o| o as f64);
        let (_, dw) = lagrange(&xs[..cnt], 0.0);
        let mut zu = [0.0; 2];
        let mut yp = 0.0;
        for m in 0..cnt {
            let q = pt.nb[(offs[m] + 2) as usize]?;
            let zq = if q == p { z } else { points[q].zeta };
            for c in 0..2 {
                zu[c] += dw[m] * zq[c];
            }
            yp += dw[m] * points[q].coord;
        }
        let (_, dl) = lagrange(&[0.0, pt.s, pt.s + 1.0], 0.0);
        let (z1, z2) = (self.zeta[pt.n1], self.zeta[pt.n2]);
        let ze = [dl[0] * z[0] + dl[1] * z1[0] + dl[2] * z2[0], dl[0] * z[1] + dl[1] * z1[1] + dl[2] * z2[1]];
        Some(self.map_gradient(pt.d, pt.sigma, zu, ze, yp))
    }

    /// ∇ζ from (∂ζ/∂u, ∂ζ/∂η) through the inverse Jacobian of T(u, η) = (x_d′ = (l + u) h,
    /// x_d = Y(u) − σ η h), T′ = [[h, 0], [Y′, −σ h]] in (d′, d) components.
    fn map_gradient(&self, d: usize, sigma: f64, zu: P2, ze: P2, yp: f64) -> Tensor {
        let h = self.grid.h();
        // T′ in (d′, d) ordering and its inverse.
        let t = [[h, 0.0], [yp, -sigma * h]];
        let det = t[0][0] * t[1][1] - t[0][1] * t[1][0];
        let ti = [[t[1][1] / det, -t[0][1] / det], [-t[1][0] / det, t[0][0] / det]];
        let mut g = [[0.0; 2]; 2];
        for r in 0..2 {
            // [∂ζ_r/∂x_d′, ∂ζ_r/∂x_d] = [∂ζ_r/∂u, ∂ζ_r/∂η] T′⁻¹.
            let dxp = zu[r] * ti[0][0] + ze[r] * ti[1][0];
            let dxd = zu[r] * ti[0][1] + ze[r] * ti[1][1];
            g[r][1 - d] = dxp;
            g[r][d] = dxd;
        }
        g
    }

    /// ∇ζ at x from the surface-fitted map of the nearest surface; `None` away from a surface
    /// the map covers. Each family of grid lines gives a map; where the surface runs near 45°
    /// both do, and they are blended by how steeply their lines meet the surface (a smooth
    /// weight, so the stress never jumps when the surface turns past the diagonal).
    pub(crate) fn reconstruct(&self, st: &CutState, x: P2) -> Option<Tensor> {
        let mut parts: [Option<(Tensor, f64)>; 2] = [None, None];
        for (d, part) in parts.iter_mut().enumerate() {
            *part = self.reconstruct_family(st, x, d);
        }
        match parts {
            [Some((g0, q0)), Some((g1, q1))] => {
                // Weight of family 0 from q0 − q1 = |n_x| − |n_y| (0 on the diagonal).
                let w0 = smoothstep(((q0 - q1) / BLEND + 1.0) * 0.5);
                let mut g = [[0.0; 2]; 2];
                for r in 0..2 {
                    for c in 0..2 {
                        g[r][c] = w0 * g0[r][c] + (1.0 - w0) * g1[r][c];
                    }
                }
                Some(g)
            }
            [Some((g, _)), None] | [None, Some((g, _))] => Some(g),
            [None, None] => None,
        }
    }

    /// The map of the lines along axis d at x, with |n_d| of the surface there.
    fn reconstruct_family(&self, st: &CutState, x: P2, d: usize) -> Option<(Tensor, f64)> {
        let h = self.grid.h();
        let o = self.grid.origin();
        let t = (x[1 - d] - o[1 - d]) / h;
        let l0 = t.round();
        // The nearest surface point of this family on the lines within two of x, then its
        // match on the line nearest x.
        let li = l0 as isize;
        let mut near: Option<(usize, f64)> = None;
        for l in li - 2..=li + 2 {
            if l < 0 || l as usize >= st.lines[d].len() {
                continue;
            }
            for &q in &st.lines[d][l as usize] {
                let pq = &st.points[q];
                if pq.valid {
                    let px = pq.position();
                    let dist = (px[0] - x[0]).hypot(px[1] - x[1]);
                    if near.is_none_or(|n| dist < n.1) {
                        near = Some((q, dist));
                    }
                }
            }
        }
        let (p, dist) = near?;
        if !(dist <= 2.5 * h) {
            return None;
        }
        let off = li - st.points[p].line as isize;
        if !(-2..=2).contains(&off) {
            return None;
        }
        let p0 = st.points[p].nb[(off + 2) as usize].filter(|&q| st.points[q].valid)?;
        let u = t - l0;
        let pt = &st.points[p0];
        let f = u.floor() as isize;
        let sets: [&[isize]; 4] = [&[f - 1, f, f + 1, f + 2], &[f, f + 1, f + 2], &[f - 1, f, f + 1], &[f, f + 1]];
        let point_at = |off: isize| -> Option<usize> {
            if !(-2..=2).contains(&off) {
                return None;
            }
            pt.nb[(off + 2) as usize].filter(|&q| st.points[q].valid)
        };
        let mut chosen: Option<(&[isize], [usize; 4])> = None;
        for set in sets {
            let mut ids = [p0; 4];
            let mut ok = true;
            for (m, &off) in set.iter().enumerate() {
                match point_at(off) {
                    Some(q) => ids[m] = q,
                    None => ok = false,
                }
            }
            if ok {
                chosen = Some((set, ids));
                break;
            }
        }
        let (set, ids) = chosen?;
        let mut xs = [0.0; 4];
        for (m, &v) in set.iter().enumerate() {
            xs[m] = v as f64;
        }
        let (w, dw) = lagrange(&xs[..set.len()], u);
        let sigma = pt.sigma;
        let mut y = 0.0;
        let mut yp = 0.0;
        for m in 0..set.len() {
            y += w[m] * st.points[ids[m]].coord;
            yp += dw[m] * st.points[ids[m]].coord;
        }
        let eta = sigma * (y - x[d]) / h;
        let mut zu = [0.0; 2];
        let mut ze = [0.0; 2];
        for m in 0..set.len() {
            let q = &st.points[ids[m]];
            let (lw, ldw) = lagrange(&[0.0, q.s, q.s + 1.0], eta);
            let (zs, z1, z2) = (q.zeta, self.zeta[q.n1], self.zeta[q.n2]);
            for c in 0..2 {
                let pv = lw[0] * zs[c] + lw[1] * z1[c] + lw[2] * z2[c];
                let pd = ldw[0] * zs[c] + ldw[1] * z1[c] + ldw[2] * z2[c];
                zu[c] += dw[m] * pv;
                ze[c] += w[m] * pd;
            }
        }
        Some((self.map_gradient(d, sigma, zu, ze, yp), pt.normal[d].abs()))
    }

    /// One dynamic step with the mapped boundary.
    pub(crate) fn step_mapped(&mut self) -> Result<(), Error> {
        let n = self.grid.len();
        let (_, _, rho0) = self.material.constants();
        let h = self.grid.h();
        let dt = self.dt;
        let kick = if self.started { dt } else { 0.5 * dt };
        self.started = true;
        let mu_e = self.settings.viscosity * rho0 * self.material.p_wave_speed() * h;

        // The cut state of the current configuration: kept from the end of the last step, or
        // built now after a change from outside.
        let mut st = match self.cut.take() {
            Some(st) => st,
            None => self.build_cut()?,
        };
        let mut force = self.mapped_forces(&mut st, mu_e)?;
        // 3. Momentum of each control volume (lumped: ρ times its solid area), with the
        // coarser level's part of the interface nodes' control volumes on a refined level.
        let mut cv = vec![0.0; n];
        for k in 0..n {
            for (o, w) in st.share[k].into_iter().flatten() {
                cv[o] += w * st.area[k];
            }
        }
        if let Some(l) = self.level.as_mut() {
            l.force.clone_from(&force);
            l.area.clone_from(&cv);
            for k in 0..n {
                cv[k] += l.ext_area[k];
                force[k][0] += l.ext_force[k][0];
                force[k][1] += l.ext_force[k][1];
            }
        }
        let mut v_new = self.v.clone();
        let mut live = vec![false; n];
        for k in 0..n {
            let kind = self.kind[k];
            if !kind.source() {
                continue;
            }
            live[k] = true;
            if kind == NodeKind::Fixed || st.owner[k] != Some(k) || !(cv[k] > 0.0) {
                continue;
            }
            let det = self.node_gradient(k).map(|g| {
                let j = identity_plus(&g);
                j[0][0] * j[1][1] - j[0][1] * j[1][0]
            });
            let rho = match det {
                Some(d) if d > 0.0 && d.is_finite() => rho0 * d,
                _ => rho0,
            };
            let v = self.v[k];
            let adv = [self.upwind_dot(k, v, |m| self.v[m][0]), self.upwind_dot(k, v, |m| self.v[m][1])];
            for c in 0..2 {
                v_new[k][c] = (v[c] + kick * (force[k][c] / (rho * cv[k]) - adv[c])) / (1.0 + self.settings.damping * kick);
            }
        }
        for k in 0..n {
            if self.kind[k] == NodeKind::Solid
                && let Some(o) = st.owner[k]
                && o != k
            {
                v_new[k] = v_new[o];
            }
        }
        self.v = v_new;

        // 4. Reference map, body, band and velocities as for the blurred boundary.
        self.advect_reference()?;
        self.time += dt;
        self.rebuild()?;
        self.v_live = (0..n).map(|k| live[k] && self.kind[k].known()).collect();
        self.extrapolate_velocity();
        for k in 0..n {
            if self.kind[k].known() && self.v[k].iter().chain(self.zeta[k].iter()).any(|x| !x.is_finite()) {
                return Err(Error::Diverged(format!("non-finite value at node {k} at t = {}", self.time)));
            }
        }
        // The cut state of the new configuration, for the next step and the accessors.
        let mut next = self.build_cut_from(Some(&st))?;
        next.stats.mapped_faces = st.stats.mapped_faces;
        next.stats.fallback_points = st.stats.fallback_points;
        self.cut = Some(next);
        Ok(())
    }

    /// The net force on every control volume and its solid area from the current
    /// configuration, without stepping (the residual of the static problem); with the
    /// coarser level's register on a refined level.
    pub(crate) fn level_forces(&mut self) -> Result<(Vec<[f64; 2]>, Vec<f64>), Error> {
        let n = self.grid.len();
        let (_, _, rho0) = self.material.constants();
        let mu_e = self.settings.viscosity * rho0 * self.material.p_wave_speed() * self.grid.h();
        let mut st = self.build_cut_from(self.cut.as_ref())?;
        let force = self.mapped_forces(&mut st, mu_e)?;
        let mut cv = vec![0.0; n];
        for k in 0..n {
            for (o, w) in st.share[k].into_iter().flatten() {
                cv[o] += w * st.area[k];
            }
        }
        if let Some(l) = self.level.as_mut() {
            l.force.clone_from(&force);
            l.area.clone_from(&cv);
        }
        self.cut = Some(st);
        Ok((force, cv))
    }

    /// ∫ t ds over a surface piece of the cell at `origin`: t = σ_ambient·n, n ds = (dy, −dx).
    fn arc_traction(&self, arc: &Curve, origin: P2, c0: usize) -> P2 {
        let mut f = [0.0; 2];
        for (t, w) in GAUSS3 {
            let p = arc.at(t);
            let x = [origin[0] + p[0], origin[1] + p[1]];
            let d = arc.tangent(t);
            let s = self.ambient_at(x, Some(c0));
            let tr = traction(&s, [d[1], -d[0]]);
            f[0] += w * tr[0];
            f[1] += w * tr[1];
        }
        f
    }

    /// The net force on every control volume (at the node that holds it): fluxes through the faces,
    /// the traction on the curved surface and the body force.
    pub(crate) fn mapped_forces(&self, st: &mut CutState, mu_e: f64) -> Result<Vec<[f64; 2]>, Error> {
        self.mapped_forces_on(st, mu_e, None)
    }

    /// As [`Solid::mapped_forces`], with only the faces that touch a node of `mask` (when given):
    /// the forces are then exact at the control volumes whose own node and whose sharing
    /// neighbours all lie in the mask.
    pub(crate) fn mapped_forces_on(&self, st: &mut CutState, mu_e: f64, mask: Option<&[bool]>) -> Result<Vec<[f64; 2]>, Error> {
        let n = self.grid.len();
        // Net force on the solid of each node's dual cell, then shared out to the control
        // volumes that hold it.
        let mut raw = vec![[0.0f64; 2]; n];
        // 1. Faces.
        let mut stats = st.stats.clone();
        stats.mapped_faces = 0;
        stats.fallback_points = 0;
        stats.max_flux = 0.0;
        let h = self.grid.h();
        let structure = st.structure.clone();
        for &(k, d, full) in &structure.faces {
            if let Some(m) = mask
                && !m[k]
                && self.along(k, d, 1).is_none_or(|b| !m[b])
            {
                continue;
            }
            let Some(nodes) = full else {
                self.face_force(st, k, d, mu_e, &mut raw, &mut stats)?;
                continue;
            };
            // Between two full cells: the compact stencil (as in face_force).
            let b = nodes[0];
            let (Some(sk), Some(sb)) = (st.share[k], st.share[b]) else {
                continue;
            };
            if sk[1].1 == 0.0 && sb[1].1 == 0.0 && sk[0].0 == sb[0].0 {
                continue;
            }
            let e = self.compact_edge_at(k, d, nodes, mu_e)?;
            let sv = [e.stress[0] + e.viscous[0], e.stress[1] + e.viscous[1], e.stress[2] + e.viscous[2]];
            let s = if d == 0 { [sv[0], sv[2]] } else { [sv[2], sv[1]] };
            let f = [s[0] * h, s[1] * h];
            stats.max_flux = stats.max_flux.max(f[0].abs()).max(f[1].abs());
            raw[k][0] += f[0];
            raw[k][1] += f[1];
            raw[b][0] -= f[0];
            raw[b][1] -= f[1];
        }
        st.stats = stats;
        self.add_loads(st, &mut raw);
        let mut force = vec![[0.0f64; 2]; n];
        for k in 0..n {
            for (o, w) in st.share[k].into_iter().flatten() {
                force[o][0] += w * raw[k][0];
                force[o][1] += w * raw[k][1];
            }
        }
        Ok(force)
    }

    /// The external load on every control volume (at the node that holds it): the traction on
    /// the curved surface and the body force.
    pub(crate) fn mapped_loads(&self, st: &CutState) -> Vec<[f64; 2]> {
        let n = self.grid.len();
        let mut raw = vec![[0.0f64; 2]; n];
        self.add_loads(st, &mut raw);
        let mut force = vec![[0.0f64; 2]; n];
        for k in 0..n {
            for (o, w) in st.share[k].into_iter().flatten() {
                force[o][0] += w * raw[k][0];
                force[o][1] += w * raw[k][1];
            }
        }
        force
    }

    /// Adds the traction on the curved surface and the body force to the dual cells' forces.
    fn add_loads(&self, st: &CutState, raw: &mut [[f64; 2]]) {
        let n = self.grid.len();
        for (c0, cell) in st.cells.iter() {
            let Some(c) = self.cell_corners(*c0) else {
                continue;
            };
            let o = self.grid.position(*c0);
            for q in 0..4 {
                for arc in &cell.quarter_arcs[q] {
                    let f = self.arc_traction(arc, o, *c0);
                    raw[c[q]][0] += f[0];
                    raw[c[q]][1] += f[1];
                }
            }
        }
        if let Some(b) = &self.body {
            for k in 0..n {
                if st.area[k] > 0.0 {
                    let c = [st.moment[k][0] / st.area[k], st.moment[k][1] / st.area[k]];
                    let f = (b.0)(c);
                    raw[k][0] += self.load * f[0] * st.area[k];
                    raw[k][1] += self.load * f[1] * st.area[k];
                }
            }
        }
    }

    /// Adds the flux through the face between node k and its +d neighbour to both control
    /// volumes.
    fn face_force(&self, st: &CutState, k: usize, d: usize, mu_e: f64, force: &mut [[f64; 2]], stats: &mut CutStats) -> Result<(), Error> {
        let Some(b) = self.along(k, d, 1) else {
            return Ok(());
        };
        // The two cells holding the halves of the face and the half-face index in each.
        let lower = if d == 0 { self.grid.neighbour(k, 0, -1).map(|c| (c, 2)) } else { self.grid.neighbour(k, -1, 0).map(|c| (c, 1)) };
        let upper = Some((k, if d == 0 { 0 } else { 3 }));
        let halves = [lower, upper];
        let classes = halves.map(|hf| hf.map_or(CellClass::Empty, |(c, _)| st.class[c]));
        if classes.iter().all(|c| matches!(c, CellClass::Empty | CellClass::Foreign)) {
            return Ok(());
        }
        let (Some(sk), Some(sb)) = (st.share[k], st.share[b]) else {
            return Ok(());
        };
        // Inside one control volume: no net effect.
        if sk[1].1 == 0.0 && sb[1].1 == 0.0 && sk[0].0 == sb[0].0 {
            return Ok(());
        }
        let h = self.grid.h();
        let comps = |s: [f64; 3]| if d == 0 { [s[0], s[2]] } else { [s[2], s[1]] };
        let edge = self.compact_edge(k, d, mu_e)?;
        let mut f = [0.0; 2];
        if classes.iter().all(|c| *c == CellClass::Full) {
            let Some(e) = edge else {
                return Ok(());
            };
            let s = comps([e.stress[0] + e.viscous[0], e.stress[1] + e.viscous[1], e.stress[2] + e.viscous[2]]);
            f = [s[0] * h, s[1] * h];
        } else if classes.iter().all(|c| matches!(c, CellClass::Full | CellClass::Foreign)) {
            // Half of the face on this level (a coarse–fine interface runs along the edge): the
            // midpoint rule on that half, the stress at its middle (a quarter of a cell from the
            // edge) interpolated between the edge and the parallel edge on the same side.
            let Some(e) = edge else {
                return Ok(());
            };
            let side: isize = if classes[1] == CellClass::Full { 1 } else { -1 };
            let par = self.along(k, 1 - d, side);
            let e2 = match par {
                Some(p) => self.compact_edge(p, d, mu_e)?,
                None => None,
            };
            let e2 = e2.unwrap_or(e);
            let mut s3 = [0.0; 3];
            for c in 0..3 {
                s3[c] = 0.75 * (e.stress[c] + e.viscous[c]) + 0.25 * (e2.stress[c] + e2.viscous[c]);
            }
            let s = comps(s3);
            f = [0.5 * s[0] * h, 0.5 * s[1] * h];
        } else {
            stats.mapped_faces += 1;
            let visc = edge.map_or([0.0; 3], |e| e.viscous);
            for (half, class) in halves.iter().zip(classes) {
                let Some((c0, hf)) = *half else {
                    continue;
                };
                let o = self.grid.position(c0);
                let pieces: Vec<(P2, P2)> = match class {
                    CellClass::Empty | CellClass::Foreign => continue,
                    CellClass::Full => vec![cut::half_face(h, hf)],
                    CellClass::Cut(i) => st.cells[i].1.faces[hf].clone(),
                };
                for (a, bb) in pieces {
                    let (a, bb) = ([o[0] + a[0], o[1] + a[1]], [o[0] + bb[0], o[1] + bb[1]]);
                    let len = (bb[0] - a[0]).hypot(bb[1] - a[1]);
                    for (t, w) in GAUSS2 {
                        let x = lerp(a, bb, t);
                        let s = match self.reconstruct(st, x) {
                            Some(g) => {
                                let fdef = invert(&identity_plus(&g))
                                    .ok_or_else(|| Error::Diverged(format!("the reference map folded near node {k} at t = {}", self.time)))?;
                                let s = self.material.cauchy(&fdef).map_err(|e| Error::Diverged(format!("{e} near node {k}, t = {}", self.time)))?;
                                [s[0], s[1], s[2]]
                            }
                            None => {
                                stats.fallback_points += 1;
                                match edge {
                                    Some(e) => e.stress,
                                    None => continue,
                                }
                            }
                        };
                        let sv = comps([s[0] + visc[0], s[1] + visc[1], s[2] + visc[2]]);
                        f[0] += w * len * sv[0];
                        f[1] += w * len * sv[1];
                    }
                }
            }
        }
        stats.max_flux = stats.max_flux.max(f[0].abs()).max(f[1].abs());
        force[k][0] += f[0];
        force[k][1] += f[1];
        force[b][0] -= f[0];
        force[b][1] -= f[1];
        Ok(())
    }
}

/// The node holding the larger share.
fn owner_of(sh: [(usize, f64); 2]) -> usize {
    if sh[1].1 > sh[0].1 { sh[1].0 } else { sh[0].0 }
}

/// A cut cell's edge crossings as fractions from corner e along edge e (counter-clockwise):
/// the stored fractions run along +x / +y, so the top and left edges are reversed.
fn crossings_of(edges: &[Option<usize>; 4], t: &[f64]) -> [Option<f64>; 4] {
    let at = |e: usize| edges[e].and_then(|i| t.get(i).copied());
    [at(0), at(1), at(2).map(|v| 1.0 - v), at(3).map(|v| 1.0 - v)]
}

/// The chains of valid surface points: each a family's points on consecutive lines, linked
/// both ways by their matches, in line order.
fn surface_chains(points: &[SurfacePoint]) -> Vec<Vec<usize>> {
    let valid = |q: Option<usize>| q.filter(|&q| points[q].valid);
    let next = |p: usize| valid(points[p].nb[3]).filter(|&q| points[q].nb[1] == Some(p));
    let prev = |p: usize| valid(points[p].nb[1]).filter(|&q| points[q].nb[3] == Some(p));
    let mut seen = vec![false; points.len()];
    let mut chains = Vec::new();
    // Starts first, then whatever is left (a closed chain, cut anywhere).
    let starts: Vec<usize> = (0..points.len()).filter(|&p| points[p].valid && prev(p).is_none()).collect();
    let rest: Vec<usize> = (0..points.len()).filter(|&p| points[p].valid).collect();
    for p0 in starts.into_iter().chain(rest) {
        if seen[p0] {
            continue;
        }
        let mut chain = vec![p0];
        seen[p0] = true;
        let mut p = p0;
        while let Some(q) = next(p) {
            if seen[q] {
                break;
            }
            seen[q] = true;
            chain.push(q);
            p = q;
        }
        chains.push(chain);
    }
    chains
}

/// A square band matrix (kl sub- and ku superdiagonals, room for the fill of partial
/// pivoting), solved by LU with partial pivoting.
#[derive(Debug, Clone)]
struct Band {
    n: usize,
    kl: usize,
    ku: usize,
    w: usize,
    a: Vec<f64>,
}

impl Band {
    fn new(n: usize, kl: usize, ku: usize) -> Band {
        let w = 2 * kl + ku + 1;
        Band { n, kl, ku, w, a: vec![0.0; n * w] }
    }

    /// Storage of (i, j), when inside the band (with the room for fill).
    fn at(&self, i: usize, j: usize) -> Option<usize> {
        let o = (j + self.kl).checked_sub(i)?;
        (o < self.w && i < self.n && j < self.n).then_some(i * self.w + o)
    }

    fn get(&self, i: usize, j: usize) -> f64 {
        self.at(i, j).map_or(0.0, |k| self.a[k])
    }

    /// Sets (i, j); entries outside the band are dropped.
    fn set(&mut self, i: usize, j: usize, v: f64) {
        if j + self.kl >= i
            && j <= i + self.ku
            && let Some(k) = self.at(i, j)
        {
            self.a[k] = v;
        }
    }

    /// Solves A x = b; `None` when singular.
    #[cfg(test)]
    fn solve(self, b: Vec<f64>) -> Option<Vec<f64>> {
        self.factor()?.solve(b)
    }

    /// LU with partial pivoting in place (LINPACK order: the multipliers stay where they were
    /// computed, the row swaps are replayed on the right-hand side); `None` when singular.
    fn factor(mut self) -> Option<BandLu> {
        let n = self.n;
        let reach = self.kl + self.ku;
        let mut pivots = vec![0; n];
        for k in 0..n {
            let last = (k + self.kl).min(n - 1);
            let mut piv = k;
            for i in k + 1..=last {
                if self.get(i, k).abs() > self.get(piv, k).abs() {
                    piv = i;
                }
            }
            let pv = self.get(piv, k);
            if !(pv.abs() > 0.0) || !pv.is_finite() {
                return None;
            }
            pivots[k] = piv;
            let right = (k + reach).min(n - 1);
            if piv != k {
                for j in k..=right {
                    let (x, y) = (self.get(k, j), self.get(piv, j));
                    if let (Some(a), Some(c)) = (self.at(k, j), self.at(piv, j)) {
                        self.a[a] = y;
                        self.a[c] = x;
                    }
                }
            }
            for i in k + 1..=last {
                let f = self.get(i, k) / pv;
                if f != 0.0 {
                    for j in k + 1..=right {
                        if let Some(c) = self.at(i, j) {
                            self.a[c] -= f * self.get(k, j);
                        }
                    }
                }
                if let Some(c) = self.at(i, k) {
                    self.a[c] = f;
                }
            }
        }
        Some(BandLu { band: self, pivots })
    }
}

/// A factorised [`Band`]: U on and above the diagonal, the multipliers below, the pivot rows.
#[derive(Debug, Clone)]
pub(crate) struct BandLu {
    band: Band,
    pivots: Vec<usize>,
}

impl BandLu {
    /// Solves A x = b; `None` when the result is not finite.
    fn solve(&self, mut b: Vec<f64>) -> Option<Vec<f64>> {
        let a = &self.band;
        let n = a.n;
        if b.len() != n {
            return None;
        }
        let reach = a.kl + a.ku;
        for k in 0..n {
            let p = self.pivots[k];
            if p != k {
                b.swap(k, p);
            }
            let last = (k + a.kl).min(n - 1);
            for i in k + 1..=last {
                b[i] -= a.get(i, k) * b[k];
            }
        }
        let mut x = vec![0.0; n];
        for k in (0..n).rev() {
            let right = (k + reach).min(n - 1);
            let mut s = b[k];
            for j in k + 1..=right {
                s -= a.get(k, j) * x[j];
            }
            x[k] = s / a.get(k, k);
        }
        x.iter().all(|v| v.is_finite()).then_some(x)
    }
}

/// The line offsets for the tangential derivative at a surface point: centred when both
/// neighbours exist, else one-sided over three lines, else two.
fn tangent_stencil(nb: &[Option<usize>; 5]) -> Option<([isize; 3], usize)> {
    let has = |o: isize| nb[(o + 2) as usize].is_some();
    if has(-1) && has(1) {
        Some(([-1, 0, 1], 3))
    } else if has(1) && has(2) {
        Some(([0, 1, 2], 3))
    } else if has(-1) && has(-2) {
        Some(([-2, -1, 0], 3))
    } else if has(1) {
        Some(([0, 1, 0], 2))
    } else if has(-1) {
        Some(([-1, 0, 0], 2))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn band_solver_matches_a_dense_solve() {
        // A pentadiagonal system needing pivoting (zero leading diagonal entry).
        let n = 7;
        let mut b = Band::new(n, 2, 2);
        let mut dense = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in i.saturating_sub(2)..=(i + 2).min(n - 1) {
                let v = if i == j && i == 0 { 0.0 } else { 1.0 + ((3 * i + 5 * j) % 7) as f64 * 0.3 - if i == j { 0.0 } else { 0.9 } };
                b.set(i, j, v);
                dense[i][j] = v;
            }
        }
        let x_true: Vec<f64> = (0..n).map(|i| 1.0 + i as f64 * 0.5).collect();
        let rhs: Vec<f64> = (0..n).map(|i| (0..n).map(|j| dense[i][j] * x_true[j]).sum()).collect();
        let x = b.solve(rhs).unwrap();
        for i in 0..n {
            assert!((x[i] - x_true[i]).abs() < 1e-10, "{x:?}");
        }
    }

    #[test]
    fn lagrange_weights_reproduce_quadratics() {
        let xs = [0.0, 0.3, 1.3];
        let f = |x: f64| 2.0 - x + 0.5 * x * x;
        let (w, dw) = lagrange(&xs, 0.7);
        let v: f64 = (0..3).map(|i| w[i] * f(xs[i])).sum();
        let d: f64 = (0..3).map(|i| dw[i] * f(xs[i])).sum();
        assert!((v - f(0.7)).abs() < 1e-14 && (d - (-1.0 + 0.7)).abs() < 1e-14);
    }
}

#[cfg(test)]
mod spectrum {
    use super::*;
    use crate::solid::Boundary;
    use crate::{Grid, Limits, Material, Settings, Shape};
    use std::fmt::Write as _;

    /// Research diagnostic: writes the Jacobian of the control-volume forces with respect to
    /// the nodal displacements of the Lamé cylinder (mapped boundary, linear elastic, at rest
    /// or after `SPECTRUM_STEPS` steps) for an eigenvalue analysis of the explicit step.
    /// `SPECTRUM_H=1 SPECTRUM_OUT=jac.txt cargo test --release -p fieldcraft-eulerian
    /// spectrum -- --ignored`.
    #[test]
    #[ignore = "research diagnostic"]
    fn force_jacobian() {
        let env = |k: &str, d: f64| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
        let h = env("SPECTRUM_H", 2.0);
        let steps = env("SPECTRUM_STEPS", 0.0) as usize;
        let n = (50.0 / h).round() as usize + 1;
        let o = -0.5 * (n as f64 - 1.0) * h;
        let g = Grid::new(n, n, h, [o, o], [false; 2], &Limits::default()).unwrap();
        let m = Material::linear_elastic(210_000.0, 0.3, 1.0).unwrap();
        let c = m.p_wave_speed();
        let set = Settings { damping: 0.5 * c / 20.0, boundary: Boundary::Mapped, viscosity: 0.0, ..Settings::default() };
        let mut s = Solid::new(g, m, Shape::Annulus { center: [0.0, 0.0], inner: 10.0, outer: 20.0 }, set, Limits::default()).unwrap();
        s.set_ambient_field(|x| if x[0].hypot(x[1]) < 15.0 { [-100.0, -100.0, 0.0] } else { [0.0; 3] }).unwrap();
        s.run(steps).unwrap();
        s.v.iter_mut().for_each(|v| *v = [0.0; 2]);
        let mut st = s.build_cut().unwrap();
        let f0 = s.mapped_forces(&mut st, 0.0).unwrap();
        let mut cv = vec![0.0; s.grid.len()];
        for k in 0..s.grid.len() {
            for (o, w) in st.share[k].into_iter().flatten() {
                cv[o] += w * st.area[k];
            }
        }
        let ks: Vec<usize> = (0..s.grid.len()).filter(|&k| s.kind[k].source() && st.owner[k] == Some(k)).collect();
        let mut out = String::new();
        for &k in &ks {
            let x = s.grid.position(k);
            writeln!(out, "node {k} {} {} {} {}", x[0], x[1], cv[k], s.phi[k]).unwrap();
        }
        let eps = 1e-7;
        for &k in &ks {
            for comp in 0..2 {
                let mut t = s.clone();
                // Displace node k and the inside nodes merged into it (u = −ζ).
                for m in 0..t.grid.len() {
                    if st.owner[m] == Some(k) && t.kind[m].source() {
                        t.zeta[m][comp] -= eps;
                    }
                }
                let mut st2 = t.build_cut().unwrap();
                let f1 = t.mapped_forces(&mut st2, 0.0).unwrap();
                let col: Vec<String> =
                    ks.iter().flat_map(|&m| [(f1[m][0] - f0[m][0]) / eps, (f1[m][1] - f0[m][1]) / eps]).map(|v| format!("{v:.9e}")).collect();
                writeln!(out, "col {k} {comp} {}", col.join(" ")).unwrap();
            }
        }
        std::fs::write(std::env::var("SPECTRUM_OUT").unwrap_or_else(|_| "jacobian.txt".into()), out).unwrap();
    }
}
