//! Local refinement near the surface: a hierarchy of nested levels, finer towards φ = 0.
//!
//! The mapped boundary resolves the surface to second order, but a uniform grid pays for the
//! surface's resolution everywhere. Here the grid is a quadtree stored level by level: level 0
//! is the base lattice (spacing h₀), level l a lattice of spacing h₀/2ˡ on which only the
//! nodes of the refined region Rₗ (and a few layers around it) are active, a sparse
//! [`Grid`]. Each level is a [`Solid`] on its sparse grid; the surface (cut cells, surface
//! values, the band) lives on the finest level only.
//!
//! **Regions.** The finest region is the cells of level L − 1 within `band` finest cells of
//! φ = 0 (or straddling it), each split into four. Every coarser region contains the next finer
//! one dilated by `nesting` (≥ 2) of its own cells, so the levels are properly nested; that
//! implies the 2:1 balance of the quadtree across edges and corners. A *leaf* (own) cell of
//! level l is a cell of Rₗ that is not refined further.
//!
//! **Nodes.** On level l every node has a role:
//! - *interior*: all its cells are leaves of this level; updated here;
//! - *master*: on ∂Rₗ at a coarse position (a node of level l − 1 too); updated here, its control
//!   volume completed by the coarser level's cells (the flux register below);
//! - *hanging*: on ∂Rₗ midway between two masters (on the edge of a coarse leaf). Its value is
//!   the cubic interpolation along the interface through the two masters and the next nodes on
//!   the line (quadratic at a corner of the region): a linear interpolation would leave an
//!   O(h²) value error and a first-order stress error in the cells next to it. Its mass and
//!   force go half to each master, which with the cells' half-faces makes the fluxes telescope
//!   and keeps a constant stress free of net force (the patch test) for any interpolation exact
//!   on linear fields;
//! - *ghost*: one or two layers outside Rₗ, for the stencils of the cells at ∂Rₗ; biquadratic
//!   interpolation in space (from coarse nodes that the coarser level advances, and the masters
//!   for points on ∂Rₗ) and linear in time;
//! - *interface* and *covered*: nodes the finer level holds (on ∂Rₗ₊₁ and inside it, a buffer
//!   of two cells); copied from the finer level after its steps.
//!
//! **Fluxes.** Each level assembles only its leaf cells: the four half-faces of a cell carry
//! the flux between its corners, so every flux is computed once and given to both sides with
//! opposite signs. A dual face that the interface cuts in half is integrated over the half in
//! the leaf with the midpoint rule (stress interpolated to a quarter cell from the edge). The
//! force a coarse level's leaves put on its interface nodes is the coarse side of the masters'
//! control volumes: it is handed to the finer level's masters as a register (impulse over the
//! coarse step, spread over the fine substeps), with the coarse side's solid area as extra mass.
//! The composite scheme is therefore conservative: summed over all degrees of freedom of all
//! levels, the impulses of the internal fluxes cancel to round-off.
//!
//! **Time.** Berger–Oliger subcycling with the time-step ratio equal to the refinement ratio:
//! a step of level l is followed by two steps of level l + 1, whose ghosts are interpolated
//! between the coarse states before and after its step; then the finer level's values are
//! copied onto the coarse nodes it covers (positions as they are, velocities averaged over the
//! two substeps, which puts them at the coarse level's half step).
//!
//! **Statics.** [`Hierarchy::residual`] evaluates the composite equilibrium residual of the
//! current configuration (all levels at the same time, no subcycling): the same operators,
//! without the time loop, for a static solver.
//!
//! **Regridding.** [`Hierarchy::regrid`] re-flags the regions from the current surface and
//! moves the state to the new levels: ζ by interpolation from the finest old level holding each
//! point (exact for linear maps), the velocity by the overlap of old and new control volumes
//! (momentum Σ m v kept exactly for the masses of full cells; constant and, away from the
//! interfaces, linear velocity fields reproduced).
//!
//! References for the general scheme: M. J. Berger, J. Oliger, *Adaptive mesh refinement for
//! hyperbolic partial differential equations*, J. Comput. Phys. 53:484–512 (1984) (subcycling,
//! nested grids); M. J. Berger, P. Colella, *Local adaptive mesh refinement for shock
//! hydrodynamics*, J. Comput. Phys. 82:64–84 (1989) (proper nesting, refluxing); hanging nodes
//! as linear constraints as in conforming finite elements on non-conforming quadtrees.

use crate::grid::Grid;
use crate::mapped::SurfaceSample;
use crate::material::{Material, Stress};
use crate::shape::Shape;
use crate::solid::{Boundary, Field, LevelHooks, NodeKind, Settings, Solid};
use crate::{Error, Limits, finite};
use std::sync::Arc;

/// Most refinement levels above the base.
pub const MAX_LEVELS: usize = 4;
/// Layers of ghost nodes outside a refined region.
const GHOST_LAYERS: isize = 2;
/// Layers of covered nodes kept under a finer level, inside its interface.
const BUFFER: isize = 2;

/// A predicate on positions.
pub type Region = Arc<dyn Fn([f64; 2]) -> bool + Send + Sync>;

/// How the grid is refined.
#[derive(Clone)]
pub struct Refinement {
    /// Refinement levels above the base (0 = the base grid alone).
    pub levels: usize,
    /// Half-width of the finest band around φ = 0, in finest cells.
    pub band: f64,
    /// Margin between nested regions, in cells of the coarser one (at least 2).
    pub nesting: usize,
    /// Base cells outside the body kept on the base level when there is no refinement, in base
    /// cells (the band of the surface needs them).
    pub base_margin: f64,
    /// An extra region refined to the finest level (cells whose centre lies in it).
    pub region: Option<Region>,
    /// Coarse steps between regrids in [`Hierarchy::run`] (0: never; the surface must not
    /// leave the finest band in between).
    pub regrid_every: usize,
    /// Subcycling: each level steps twice per step of the coarser one (the default). Without,
    /// every level takes the finest level's time step (for comparison).
    pub subcycle: bool,
    /// Hanging nodes interpolated cubically along the interface (the default); else linearly
    /// between their two masters (the transpose of the force sharing; for comparison).
    pub cubic_hanging: bool,
}

impl Default for Refinement {
    fn default() -> Self {
        Refinement { levels: 2, band: 6.0, nesting: 2, base_margin: 6.0, region: None, regrid_every: 0, subcycle: true, cubic_hanging: true }
    }
}

impl std::fmt::Debug for Refinement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Refinement")
            .field("levels", &self.levels)
            .field("band", &self.band)
            .field("nesting", &self.nesting)
            .field("base_margin", &self.base_margin)
            .field("region", &self.region.is_some())
            .field("regrid_every", &self.regrid_every)
            .field("subcycle", &self.subcycle)
            .field("cubic_hanging", &self.cubic_hanging)
            .finish()
    }
}

impl Refinement {
    fn validate(&self) -> Result<(), Error> {
        if self.levels > MAX_LEVELS {
            return Err(Error::Invalid(format!("at most {MAX_LEVELS} refinement levels, got {}", self.levels)));
        }
        if !(self.band >= 2.0 && self.band <= 64.0) {
            return Err(Error::Invalid(format!("the refined band must lie in [2, 64] finest cells, got {}", self.band)));
        }
        if !(2..=8).contains(&self.nesting) {
            return Err(Error::Invalid(format!("the nesting margin must lie in 2..=8 cells, got {}", self.nesting)));
        }
        if !(self.base_margin >= 0.0 && self.base_margin <= 64.0) {
            return Err(Error::Invalid(format!("the base margin must lie in [0, 64] cells, got {}", self.base_margin)));
        }
        Ok(())
    }
}

/// What a node of a level is (see the module notes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// Updated on this level; all its cells are leaves here.
    Interior,
    /// On the interface with the coarser level at a coarse position; updated here.
    Master,
    /// On the interface with the coarser level between two masters; interpolated.
    Hanging,
    /// Outside this level's region; interpolated from the coarser level.
    Ghost,
    /// On the interface with the finer level; held, copied from it.
    Interface,
    /// Under the finer level; held, copied from it.
    Covered,
}

impl Role {
    /// Whether the hierarchy sets the node's values (the level's solid never changes them).
    pub fn frozen(self) -> bool {
        matches!(self, Role::Hanging | Role::Ghost | Role::Interface | Role::Covered)
    }

    /// Whether the node is a degree of freedom of the composite grid.
    pub fn dof(self) -> bool {
        matches!(self, Role::Interior | Role::Master)
    }
}

/// Where a ghost value comes from.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Src {
    /// A node of the coarser level (interpolated in time).
    Coarse(usize),
    /// A master of this level (its current value).
    Here(usize),
}

/// A hanging node: its two masters (which take its mass and force, half each) and the nodes
/// and weights its value is interpolated from.
#[derive(Debug, Clone)]
struct Hanging {
    node: usize,
    masters: [usize; 2],
    value: Vec<(usize, f64)>,
}

#[derive(Debug, Clone)]
struct Ghost {
    node: usize,
    src: Vec<(Src, f64)>,
}

/// The cells of one level: its region and the cells refined into the next level.
#[derive(Debug, Clone)]
struct LevelPlan {
    nx: usize,
    ny: usize,
    h: f64,
    /// Cells per axis.
    cx: usize,
    cy: usize,
    region: Vec<bool>,
    refined: Vec<bool>,
}

/// The nodes of one level and their links to the neighbouring levels.
#[derive(Debug, Clone)]
struct Topology {
    grid: Grid,
    role: Vec<Role>,
    own_cell: Vec<bool>,
    hanging: Vec<Hanging>,
    ghosts: Vec<Ghost>,
    /// Interface nodes here and the finer level's master at the same point.
    links: Vec<(usize, usize)>,
    /// Interface and covered nodes here and the finer level's node at the same point.
    held: Vec<(usize, usize)>,
}

/// The initial state of a new level.
#[derive(Debug, Clone)]
struct LevelState {
    zeta: Vec<[f64; 2]>,
    v: Vec<[f64; 2]>,
    /// Whether the node's ζ is known (else it starts outside the body).
    known: Vec<bool>,
}

/// What locating a point needs from a level, for old and new levels alike.
struct LevelView<'a> {
    grid: &'a Grid,
    role: &'a [Role],
    own_cell: &'a [bool],
    /// The masters of each hanging node.
    masters: Vec<Option<[usize; 2]>>,
    /// The finer level's master at each interface node.
    link: Vec<Option<usize>>,
}

impl<'a> LevelView<'a> {
    fn new(grid: &'a Grid, role: &'a [Role], own_cell: &'a [bool], hanging: &[Hanging], links: &[(usize, usize)]) -> LevelView<'a> {
        let n = grid.len();
        let mut masters = vec![None; n];
        for hg in hanging {
            masters[hg.node] = Some(hg.masters);
        }
        let mut link = vec![None; n];
        for &(k, m) in links {
            link[k] = Some(m);
        }
        LevelView { grid, role, own_cell, masters, link }
    }

    /// The leaf cell of this level holding x: its lower-left node and the local coordinates.
    fn leaf(&self, x: [f64; 2]) -> Option<(usize, [f64; 2])> {
        let o = self.grid.origin();
        let h = self.grid.h();
        let (u, v) = ((x[0] - o[0]) / h, (x[1] - o[1]) / h);
        let (fi, fj) = (u.floor(), v.floor());
        if !(fi.is_finite() && fj.is_finite()) || fi < 0.0 || fj < 0.0 {
            return None;
        }
        let k = self.grid.try_index(fi as usize, fj as usize)?;
        self.own_cell.get(k).copied().unwrap_or(false).then_some((k, [u - fi, v - fj]))
    }
}

/// The degrees of freedom (level, node, weight) whose control volumes hold point x: the corner
/// of the leaf cell's quarter holding x, resolved through hanging nodes (half to each master)
/// and interface nodes (the finer level's master).
fn owners(views: &[LevelView], x: [f64; 2]) -> Vec<(usize, usize, f64)> {
    for l in (0..views.len()).rev() {
        let vw = &views[l];
        let Some((c0, t)) = vw.leaf(x) else {
            continue;
        };
        let (i, j) = vw.grid.ij(c0);
        let (di, dj) = (isize::from(t[0] >= 0.5), isize::from(t[1] >= 0.5));
        let Some(k) = vw.grid.lattice_shift(i, j, di, dj).and_then(|(a, b)| vw.grid.try_index(a, b)) else {
            return Vec::new();
        };
        return match vw.role.get(k) {
            Some(Role::Interior | Role::Master) => vec![(l, k, 1.0)],
            Some(Role::Hanging) => vw.masters[k].map_or_else(Vec::new, |[a, b]| vec![(l, a, 0.5), (l, b, 0.5)]),
            Some(Role::Interface) => vw.link[k].map_or_else(Vec::new, |m| vec![(l + 1, m, 1.0)]),
            _ => Vec::new(),
        };
    }
    Vec::new()
}

#[derive(Debug, Clone)]
struct Level {
    solid: Solid,
    role: Vec<Role>,
    ghosts: Vec<Ghost>,
    hanging: Vec<Hanging>,
    links: Vec<(usize, usize)>,
    held: Vec<(usize, usize)>,
    zeta_old: Vec<[f64; 2]>,
    v_old: Vec<[f64; 2]>,
    t_old: f64,
    tv_old: f64,
    v_sum: Vec<[f64; 2]>,
    sums: usize,
    /// The coarse force at each link in the last step (for the register's extrapolation).
    reg_prev: Vec<[f64; 2]>,
    /// Whether that force came from a full step.
    reg_prev_started: bool,
}

/// A degree of freedom of the composite grid.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NodeRef {
    /// Level.
    pub level: usize,
    /// Node index on that level's grid.
    pub index: usize,
    /// Position.
    pub position: [f64; 2],
}

/// The composite equilibrium residual at a degree of freedom.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResidualEntry {
    /// The node.
    pub node: NodeRef,
    /// Net force on its control volume (all levels' cells, hanging nodes' shares included).
    pub force: [f64; 2],
    /// Solid area of its control volume.
    pub area: f64,
}

/// What a regrid moved: the momentum and mass (full-cell areas, unit density) of the old
/// degrees of freedom handed to the new ones, and the part of that mass no new degree of
/// freedom took (zero: the transfer conserves Σ m v exactly).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RegridReport {
    /// Σ m v handed over.
    pub momentum: [f64; 2],
    /// Σ m handed over.
    pub mass: f64,
    /// Mass that found no new node.
    pub unassigned: f64,
}

/// Counters of the hierarchy.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct HierarchyStats {
    /// Active nodes per level (all roles).
    pub nodes: Vec<usize>,
    /// Degrees of freedom per level (interior and master nodes inside the body).
    pub dofs: Vec<usize>,
    /// Leaf cells per level.
    pub leaves: Vec<usize>,
    /// Node updates (a degree of freedom advanced by one step of its level) so far.
    pub updates: u64,
    /// Coarse steps so far.
    pub steps: usize,
    /// Regrids so far.
    pub regrids: usize,
}

/// A refined hierarchy of levels around the surface of one body.
#[derive(Debug, Clone)]
pub struct Hierarchy {
    levels: Vec<Level>,
    refinement: Refinement,
    base: Grid,
    material: Material,
    shape: Shape,
    settings: Settings,
    limits: Limits,
    ambient: Option<Field<[f64; 3]>>,
    body: Option<Field<[f64; 2]>>,
    updates: u64,
    steps: usize,
    regrids: usize,
    /// Σ of the impulses on all degrees of freedom, Σ of their magnitudes, and the impulse of
    /// the registers' extrapolation in time, from the last coarse step (a conservation audit).
    audit: ([f64; 2], f64, [f64; 2]),
}

fn wrap(i: isize, n: usize, periodic: bool) -> Option<usize> {
    let n = n as isize;
    if periodic {
        Some(i.rem_euclid(n) as usize)
    } else if (0..n).contains(&i) {
        Some(i as usize)
    } else {
        None
    }
}

/// Quadratic (three nodes at 0, 1, 2) or linear (two nodes at 0, 1) Lagrange weights at t.
fn lagrange_weights(count: usize, t: f64) -> [f64; 3] {
    if count == 3 { [0.5 * (t - 1.0) * (t - 2.0), -t * (t - 2.0), 0.5 * t * (t - 1.0)] } else { [1.0 - t, t, 0.0] }
}

impl LevelPlan {
    fn new(nx: usize, ny: usize, h: f64, periodic: [bool; 2]) -> LevelPlan {
        let cx = if periodic[0] { nx } else { nx - 1 };
        let cy = if periodic[1] { ny } else { ny - 1 };
        LevelPlan { nx, ny, h, cx, cy, region: vec![false; cx * cy], refined: vec![false; cx * cy] }
    }

    fn cell(&self, i: isize, j: isize, periodic: [bool; 2]) -> Option<usize> {
        let i = wrap(i, self.cx, periodic[0])?;
        let j = wrap(j, self.cy, periodic[1])?;
        Some(i + self.cx * j)
    }
}

/// The cells within `m` cells (Chebyshev) of a flagged cell.
fn dilate(p: &LevelPlan, flags: &[bool], m: isize, periodic: [bool; 2]) -> Vec<bool> {
    let mut out = vec![false; flags.len()];
    for cj in 0..p.cy {
        for ci in 0..p.cx {
            if !flags[ci + p.cx * cj] {
                continue;
            }
            for dj in -m..=m {
                for di in -m..=m {
                    if let Some(c) = p.cell(ci as isize + di, cj as isize + dj, periodic) {
                        out[c] = true;
                    }
                }
            }
        }
    }
    out
}

impl Hierarchy {
    /// A hierarchy over the base grid `base` (dense) for the body `shape`, undeformed and at
    /// rest, refined around its surface. Needs the mapped boundary.
    pub fn new(base: Grid, material: Material, shape: Shape, settings: Settings, refinement: Refinement, limits: Limits) -> Result<Hierarchy, Error> {
        refinement.validate()?;
        shape.validate()?;
        if settings.boundary != Boundary::Mapped {
            return Err(Error::Invalid("not supported yet: refinement with the blurred boundary".into()));
        }
        if base.is_sparse() {
            return Err(Error::Invalid("the base of a hierarchy must be a full grid".into()));
        }
        let mut hi = Hierarchy {
            levels: Vec::new(),
            refinement,
            base,
            material,
            shape,
            settings,
            limits,
            ambient: None,
            body: None,
            updates: 0,
            steps: 0,
            regrids: 0,
            audit: ([0.0; 2], 0.0, [0.0; 2]),
        };
        hi.set_motion(|x| (x, [0.0, 0.0]))?;
        Ok(hi)
    }

    /// Sets the initial reference map and velocity, `motion(x) = (ξ, v)`, and builds the
    /// levels around the surface of that configuration.
    pub fn set_motion(&mut self, motion: impl Fn([f64; 2]) -> ([f64; 2], [f64; 2])) -> Result<(), Error> {
        let shape = self.shape;
        let plans = self.plan(&|x| shape.phi0(motion(x).0))?;
        let topo = self.topologies(&plans)?;
        let mut states = Vec::with_capacity(topo.len());
        for t in &topo {
            let mut z = Vec::with_capacity(t.grid.len());
            let mut v = Vec::with_capacity(t.grid.len());
            for k in 0..t.grid.len() {
                let x = t.grid.position(k);
                let (xi, vel) = motion(x);
                for c in 0..2 {
                    finite("initial reference map", xi[c])?;
                    finite("initial velocity", vel[c])?;
                }
                z.push([xi[0] - x[0], xi[1] - x[1]]);
                v.push(vel);
            }
            states.push(LevelState { known: vec![true; z.len()], zeta: z, v });
        }
        self.levels = self.instantiate(topo, states, false, 0.0)?;
        self.synchronise()?;
        Ok(())
    }

    /// Sets the stress outside the body as a function of position (as
    /// [`Solid::set_ambient_field`]) on every level.
    pub fn set_ambient_field(&mut self, stress: impl Fn([f64; 2]) -> [f64; 3] + Send + Sync + 'static) -> Result<(), Error> {
        let f: Arc<dyn Fn([f64; 2]) -> [f64; 3] + Send + Sync> = Arc::new(stress);
        for lv in &mut self.levels {
            let g = f.clone();
            lv.solid.set_ambient_field(move |x| g(x))?;
        }
        self.ambient = Some(Field(f));
        Ok(())
    }

    /// Sets a body force per unit current volume (as [`Solid::set_body_force`]) on every level.
    pub fn set_body_force(&mut self, force: impl Fn([f64; 2]) -> [f64; 2] + Send + Sync + 'static) -> Result<(), Error> {
        let f: Arc<dyn Fn([f64; 2]) -> [f64; 2] + Send + Sync> = Arc::new(force);
        for lv in &mut self.levels {
            let g = f.clone();
            lv.solid.set_body_force(move |x| g(x))?;
        }
        self.body = Some(Field(f));
        Ok(())
    }

    /// The lattice of level l: nodes per axis and spacing.
    fn lattice(&self, l: usize) -> (usize, usize, f64) {
        let p = self.base.periodic();
        let n = |n0: usize, per: bool| if per { n0 << l } else { ((n0 - 1) << l) + 1 };
        (n(self.base.nx(), p[0]), n(self.base.ny(), p[1]), self.base.h() / (1u64 << l) as f64)
    }

    fn lattice_position(&self, l: usize, i: usize, j: usize) -> [f64; 2] {
        let (_, _, h) = self.lattice(l);
        let o = self.base.origin();
        [o[0] + i as f64 * h, o[1] + j as f64 * h]
    }

    /// The regions of every level for the level set `phi_at`.
    fn plan(&self, phi_at: &dyn Fn([f64; 2]) -> f64) -> Result<Vec<LevelPlan>, Error> {
        let per = self.base.periodic();
        let top = self.refinement.levels;
        let mut plans = Vec::with_capacity(top + 1);
        for l in 0..=top {
            let (nx, ny, h) = self.lattice(l);
            let cells = nx.checked_mul(ny).ok_or_else(|| Error::TooLarge("lattice size overflows".into()))?;
            if cells > self.limits.max_nodes.saturating_mul(16) {
                return Err(Error::TooLarge(format!("level {l} has a {nx} × {ny} lattice, the cap is {}", self.limits.max_nodes.saturating_mul(16))));
            }
            plans.push(LevelPlan::new(nx, ny, h, per));
        }
        // φ at the lattice nodes of a level.
        let phis = |l: usize, p: &LevelPlan| -> Vec<f64> {
            let mut out = Vec::with_capacity(p.nx * p.ny);
            for j in 0..p.ny {
                for i in 0..p.nx {
                    out.push(phi_at(self.lattice_position(l, i, j)));
                }
            }
            out
        };
        // Corner φ of a cell: (min |φ|, min φ, max φ).
        let corner_phi = |p: &LevelPlan, phi: &[f64], ci: usize, cj: usize| -> (f64, f64, f64) {
            let (mut a, mut lo, mut hi) = (f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY);
            for (di, dj) in [(0, 0), (1, 0), (1, 1), (0, 1)] {
                let i = wrap(ci as isize + di, p.nx, per[0]).unwrap_or(ci);
                let j = wrap(cj as isize + dj, p.ny, per[1]).unwrap_or(cj);
                let v = phi.get(i + p.nx * j).copied().unwrap_or(f64::INFINITY);
                let v = if v.is_nan() { f64::INFINITY } else { v };
                a = a.min(v.abs());
                lo = lo.min(v);
                hi = hi.max(v);
            }
            (a, lo, hi)
        };
        let base_phi = phis(0, &plans[0]);
        let margin = if top == 0 { self.refinement.base_margin * self.base.h() } else { 0.0 };
        if top > 0 {
            // The finest band, on the cells of level top − 1.
            let l = top - 1;
            let p = &plans[l];
            let phi = phis(l, p);
            let width = self.refinement.band * plans[top].h;
            let mut flags = vec![false; p.cx * p.cy];
            for cj in 0..p.cy {
                for ci in 0..p.cx {
                    let (a, lo, hi) = corner_phi(p, &phi, ci, cj);
                    let mut f = a <= width || (lo < 0.0 && hi >= 0.0);
                    if !f && let Some(r) = &self.refinement.region {
                        let c = self.lattice_position(l, ci, cj);
                        f = r([c[0] + 0.5 * p.h, c[1] + 0.5 * p.h]);
                    }
                    flags[ci + p.cx * cj] = f;
                }
            }
            plans[l].refined = flags;
            // Coarser levels contain the finer region with a margin of `nesting` cells.
            for l in (0..top - 1).rev() {
                let d = dilate(&plans[l + 1], &plans[l + 1].refined, self.refinement.nesting as isize, per);
                let mut flags = vec![false; plans[l].cx * plans[l].cy];
                let fine = &plans[l + 1];
                for cj in 0..fine.cy {
                    for ci in 0..fine.cx {
                        if d[ci + fine.cx * cj]
                            && let Some(c) = plans[l].cell((ci / 2) as isize, (cj / 2) as isize, per)
                        {
                            flags[c] = true;
                        }
                    }
                }
                plans[l].refined = flags;
            }
        }
        // The base region: cells touching the body (within the margin), and around the finer
        // region.
        let around = dilate(&plans[0], &plans[0].refined, self.refinement.nesting as isize, per);
        for cj in 0..plans[0].cy {
            for ci in 0..plans[0].cx {
                let (_, lo, _) = corner_phi(&plans[0], &base_phi, ci, cj);
                let c = ci + plans[0].cx * cj;
                plans[0].region[c] = around[c] || lo < margin;
            }
        }
        for l in 1..=top {
            let (lo, hi) = plans.split_at_mut(l);
            let coarse = &lo[l - 1];
            let fine = &mut hi[0];
            for cj in 0..fine.cy {
                for ci in 0..fine.cx {
                    if let Some(c) = coarse.cell((ci / 2) as isize, (cj / 2) as isize, per)
                        && coarse.refined[c]
                    {
                        fine.region[ci + fine.cx * cj] = true;
                    }
                }
            }
        }
        if !plans[0].region.iter().any(|&r| r) {
            return Err(Error::Diverged("no grid cell touches the body".into()));
        }
        Ok(plans)
    }

    /// The nodes, roles and links of every level.
    fn topologies(&self, plans: &[LevelPlan]) -> Result<Vec<Topology>, Error> {
        let per = self.base.periodic();
        let top = plans.len() - 1;
        let mut lattice_roles: Vec<Vec<Option<Role>>> = Vec::with_capacity(plans.len());
        for (l, p) in plans.iter().enumerate() {
            let n = p.nx * p.ny;
            let mut role: Vec<Option<Role>> = vec![None; n];
            for j in 0..p.ny {
                for i in 0..p.nx {
                    let (mut in_r, mut own, mut refd, mut out) = (0, 0, 0, 0);
                    for (di, dj) in [(-1, -1), (0, -1), (-1, 0), (0, 0)] {
                        let Some(c) = p.cell(i as isize + di, j as isize + dj, per) else {
                            continue;
                        };
                        if p.region[c] {
                            in_r += 1;
                            if p.refined[c] {
                                refd += 1;
                            } else {
                                own += 1;
                            }
                        } else {
                            out += 1;
                        }
                    }
                    if in_r == 0 {
                        continue;
                    }
                    role[i + p.nx * j] = Some(if own == 0 {
                        Role::Covered
                    } else if refd > 0 {
                        if l > 0 && out > 0 {
                            return Err(Error::Invalid(format!("level {l}: a node lies on both interfaces (nesting too tight)")));
                        }
                        Role::Interface
                    } else if l > 0 && out > 0 {
                        match (i % 2, j % 2) {
                            (0, 0) => Role::Master,
                            (1, 1) => return Err(Error::Invalid(format!("level {l}: an interface through a cell centre"))),
                            _ => Role::Hanging,
                        }
                    } else {
                        Role::Interior
                    });
                }
            }
            // Keep the covered nodes within the buffer of an interface node.
            let mut keep = vec![false; n];
            for j in 0..p.ny {
                for i in 0..p.nx {
                    if role[i + p.nx * j] != Some(Role::Interface) {
                        continue;
                    }
                    for dj in -BUFFER..=BUFFER {
                        for di in -BUFFER..=BUFFER {
                            if let (Some(a), Some(b)) = (wrap(i as isize + di, p.nx, per[0]), wrap(j as isize + dj, p.ny, per[1])) {
                                keep[a + p.nx * b] = true;
                            }
                        }
                    }
                }
            }
            for k in 0..n {
                if role[k] == Some(Role::Covered) && !keep[k] {
                    role[k] = None;
                }
            }
            // Ghosts around the interface with the coarser level.
            if l > 0 {
                let mut ghost = vec![false; n];
                for j in 0..p.ny {
                    for i in 0..p.nx {
                        if !matches!(role[i + p.nx * j], Some(Role::Master | Role::Hanging)) {
                            continue;
                        }
                        for dj in -GHOST_LAYERS..=GHOST_LAYERS {
                            for di in -GHOST_LAYERS..=GHOST_LAYERS {
                                if let (Some(a), Some(b)) = (wrap(i as isize + di, p.nx, per[0]), wrap(j as isize + dj, p.ny, per[1]))
                                    && role[a + p.nx * b].is_none()
                                {
                                    ghost[a + p.nx * b] = true;
                                }
                            }
                        }
                    }
                }
                for k in 0..n {
                    if ghost[k] {
                        role[k] = Some(Role::Ghost);
                    }
                }
            }
            let _ = top;
            lattice_roles.push(role);
        }

        // Grids and per-node data.
        let mut topo: Vec<Topology> = Vec::with_capacity(plans.len());
        for (l, p) in plans.iter().enumerate() {
            let lr = &lattice_roles[l];
            let active: Vec<bool> = lr.iter().map(|r| r.is_some()).collect();
            let grid = Grid::sparse(p.nx, p.ny, p.h, self.base.origin(), per, &active, &self.limits)?;
            let n = grid.len();
            let mut role = Vec::with_capacity(n);
            let mut own_cell = vec![false; n];
            for k in 0..n {
                let (i, j) = grid.ij(k);
                role.push(lr[i + p.nx * j].unwrap_or(Role::Ghost));
                if let Some(c) = p.cell(i as isize, j as isize, per) {
                    own_cell[k] = p.region[c] && !p.refined[c];
                }
            }
            let mut hanging = Vec::new();
            for k in 0..n {
                if role[k] != Role::Hanging {
                    continue;
                }
                let (i, j) = grid.ij(k);
                let along = |o: isize| if i % 2 == 1 { grid.lattice_shift(i, j, o, 0) } else { grid.lattice_shift(i, j, 0, o) };
                let at = |o: isize, ok: &dyn Fn(Role) -> bool| along(o).and_then(|(a, b)| grid.try_index(a, b)).filter(|&m| ok(role[m]));
                let master = |r: Role| r == Role::Master;
                let dof = |r: Role| r.dof();
                let (Some(a), Some(b)) = (at(-1, &master), at(1, &master)) else {
                    return Err(Error::Invalid(format!("level {l}: a hanging node at ({i}, {j}) without two masters (2:1 balance)")));
                };
                // The value: Lagrange through the masters and the further nodes on the line
                // that this level advances.
                let mut pts: Vec<(f64, usize)> = vec![(-1.0, a), (1.0, b)];
                if self.refinement.cubic_hanging {
                    if let Some(m) = at(-3, &dof) {
                        pts.push((-3.0, m));
                    }
                    if let Some(m) = at(3, &dof) {
                        pts.push((3.0, m));
                    }
                }
                let mut value = Vec::with_capacity(pts.len());
                for (p, &(xp, m)) in pts.iter().enumerate() {
                    let mut w = 1.0;
                    for (q, &(xq, _)) in pts.iter().enumerate() {
                        if q != p {
                            w *= (0.0 - xq) / (xp - xq);
                        }
                    }
                    value.push((m, w));
                }
                hanging.push(Hanging { node: k, masters: [a, b], value });
            }
            topo.push(Topology { grid, role, own_cell, hanging, ghosts: Vec::new(), links: Vec::new(), held: Vec::new() });
        }

        // Links to the finer level.
        for l in 0..top {
            let (lo, hi) = topo.split_at_mut(l + 1);
            let coarse = &mut lo[l];
            let fine = &hi[0];
            for k in 0..coarse.grid.len() {
                let r = coarse.role[k];
                if !matches!(r, Role::Interface | Role::Covered) {
                    continue;
                }
                let (i, j) = coarse.grid.ij(k);
                let f = fine.grid.try_index(2 * i, 2 * j).ok_or_else(|| Error::Invalid(format!("level {l}: a covered node with no finer node")))?;
                match (r, fine.role[f]) {
                    (Role::Interface, Role::Master) => coarse.links.push((k, f)),
                    (Role::Interface, _) => return Err(Error::Invalid(format!("level {l}: an interface node over a finer non-master"))),
                    (_, Role::Ghost | Role::Hanging) => return Err(Error::Invalid(format!("level {l}: a covered node over a finer ghost"))),
                    _ => {}
                }
                coarse.held.push((k, f));
            }
        }

        // Ghost windows: biquadratic in the coarser level's advanced nodes and this level's masters.
        for l in 1..=top {
            let (lo, hi) = topo.split_at_mut(l);
            let coarse = &lo[l - 1];
            let fine = &mut hi[0];
            let avail = |ci: isize, cj: isize| -> Option<Src> {
                let a = wrap(ci, coarse.grid.nx(), per[0])?;
                let b = wrap(cj, coarse.grid.ny(), per[1])?;
                let k = coarse.grid.try_index(a, b)?;
                match coarse.role[k] {
                    Role::Interior | Role::Master | Role::Hanging => Some(Src::Coarse(k)),
                    Role::Interface => {
                        let f = fine.grid.try_index(2 * a, 2 * b)?;
                        (fine.role[f] == Role::Master).then_some(Src::Here(f))
                    }
                    _ => None,
                }
            };
            let mut ghosts = Vec::new();
            for k in 0..fine.grid.len() {
                if fine.role[k] != Role::Ghost {
                    continue;
                }
                let (i, j) = fine.grid.ij(k);
                let starts = |i: usize| -> Vec<isize> {
                    let c = (i / 2) as isize;
                    if i.is_multiple_of(2) { vec![c - 1, c - 2, c] } else { vec![c - 1, c] }
                };
                let (sx, sy) = (starts(i), starts(j));
                let mut window: Option<Vec<(Src, f64)>> = None;
                let mut combos: Vec<(isize, isize, f64)> = Vec::new();
                for &a in &sx {
                    for &b in &sy {
                        let off = ((a as f64 + 1.0) - i as f64 / 2.0).abs() + ((b as f64 + 1.0) - j as f64 / 2.0).abs();
                        combos.push((a, b, off));
                    }
                }
                combos.sort_by(|p, q| p.2.total_cmp(&q.2));
                'outer: for (a, b, _) in combos {
                    let wx = lagrange_weights(3, i as f64 / 2.0 - a as f64);
                    let wy = lagrange_weights(3, j as f64 / 2.0 - b as f64);
                    let mut src = Vec::with_capacity(9);
                    for q in 0..3 {
                        for p in 0..3 {
                            let w = wx[p] * wy[q];
                            let s = avail(a + p as isize, b + q as isize);
                            match s {
                                Some(s) => {
                                    if w != 0.0 {
                                        src.push((s, w));
                                    }
                                }
                                None if w == 0.0 => {}
                                None => continue 'outer,
                            }
                        }
                    }
                    window = Some(src);
                    break;
                }
                if window.is_none() {
                    // Bilinear from the coarse cell holding the ghost.
                    let (a, b) = ((i / 2) as isize, (j / 2) as isize);
                    let (tx, ty) = (i as f64 / 2.0 - a as f64, j as f64 / 2.0 - b as f64);
                    let mut src = Vec::with_capacity(4);
                    let mut ok = true;
                    for (p, q) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                        let w = (if p == 0 { 1.0 - tx } else { tx }) * (if q == 0 { 1.0 - ty } else { ty });
                        if w == 0.0 {
                            continue;
                        }
                        match avail(a + p, b + q) {
                            Some(s) => src.push((s, w)),
                            None => ok = false,
                        }
                    }
                    if ok {
                        window = Some(src);
                    }
                }
                match window {
                    Some(src) => ghosts.push(Ghost { node: k, src }),
                    None => return Err(Error::Invalid(format!("level {l}: no coarse data around the ghost at ({i}, {j}) (nesting too tight)"))),
                }
            }
            fine.ghosts = ghosts;
        }
        Ok(topo)
    }

    /// Solids for the levels, with the given ζ and v at every node.
    fn instantiate(&self, topo: Vec<Topology>, states: Vec<LevelState>, started: bool, time: f64) -> Result<Vec<Level>, Error> {
        let mut levels = Vec::with_capacity(topo.len());
        for (t, st) in topo.into_iter().zip(states) {
            let LevelState { zeta: z, v, known } = st;
            let n = t.grid.len();
            let mut solid = Solid::new(t.grid, self.material, self.shape, self.settings, self.limits)?;
            let mut hooks = LevelHooks::new(n);
            hooks.own_cell = t.own_cell;
            for k in 0..n {
                hooks.frozen[k] = t.role[k].frozen();
                hooks.master[k] = t.role[k] == Role::Master;
            }
            for hg in &t.hanging {
                hooks.slave[hg.node] = Some([(hg.masters[0], 0.5), (hg.masters[1], 0.5)]);
            }
            solid.level = Some(Box::new(hooks));
            if let Some(f) = &self.ambient {
                let g = f.0.clone();
                solid.set_ambient_field(move |x| g(x))?;
            }
            if let Some(f) = &self.body {
                let g = f.0.clone();
                solid.set_body_force(move |x| g(x))?;
            }
            solid.zeta = z;
            solid.v = v;
            for k in 0..n {
                let inside = known.get(k).copied().unwrap_or(false) && solid.shape.phi0(solid.xi(k)) < 0.0;
                solid.kind[k] = match (t.role[k].frozen(), inside) {
                    (true, true) => NodeKind::Fixed,
                    (false, true) => NodeKind::Solid,
                    (_, false) => NodeKind::Empty,
                };
            }
            solid.rebuild()?;
            solid.v_live = solid.kind.iter().map(|k| k.source()).collect();
            solid.extrapolate_velocity();
            solid.started = started;
            solid.time = time;
            if !self.refinement.subcycle {
                // Every level at the finest level's step.
                solid.dt /= (1u64 << (self.refinement.levels - levels.len().min(self.refinement.levels))) as f64;
            }
            solid.cut = None;
            let zeta_old = solid.zeta.clone();
            let v_old = solid.v.clone();
            levels.push(Level {
                solid,
                role: t.role,
                ghosts: t.ghosts,
                hanging: t.hanging,
                links: t.links,
                held: t.held,
                zeta_old,
                v_old,
                t_old: 0.0,
                tv_old: 0.0,
                v_sum: vec![[0.0; 2]; n],
                sums: 0,
                reg_prev: Vec::new(),
                reg_prev_started: false,
            });
        }
        Ok(levels)
    }

    /// Makes every held, hanging and ghost value consistent with the degrees of freedom at the
    /// current time (finer levels first onto the coarser ones, then the ghosts outwards).
    fn synchronise(&mut self) -> Result<(), Error> {
        for l in (1..self.levels.len()).rev() {
            self.levels[l].sums = 0;
            self.restrict(l);
        }
        for l in 0..self.levels.len() {
            let lv = &mut self.levels[l];
            lv.zeta_old.clone_from(&lv.solid.zeta);
            lv.v_old.clone_from(&lv.solid.v);
            if l > 0 {
                self.fill_ghosts(l, 1.0, 1.0);
            }
            self.update_hanging(l);
        }
        Ok(())
    }

    /// The kind a held or interpolated node gets from its sources' kinds.
    fn derived_kind(all_inside: bool, all_known: bool) -> NodeKind {
        if all_inside {
            NodeKind::Fixed
        } else if all_known {
            NodeKind::Band
        } else {
            NodeKind::Empty
        }
    }

    /// Sets level l's ghosts from level l − 1, interpolated in time with weights θ (ζ) and
    /// θ_v (v) between the coarse states before and after its step.
    fn fill_ghosts(&mut self, l: usize, theta: f64, theta_v: f64) {
        if l == 0 || l >= self.levels.len() {
            return;
        }
        let (lo, hi) = self.levels.split_at_mut(l);
        let coarse = &lo[l - 1];
        let Level { solid, ghosts, .. } = &mut hi[0];
        for g in ghosts.iter() {
            let mut z = [0.0; 2];
            let mut v = [0.0; 2];
            let (mut inside, mut known) = (true, true);
            for &(src, w) in &g.src {
                let (zz, vv, kind) = match src {
                    Src::Coarse(k) => {
                        let (zo, zn) = (coarse.zeta_old[k], coarse.solid.zeta[k]);
                        let (vo, vn) = (coarse.v_old[k], coarse.solid.v[k]);
                        (
                            [zo[0] + theta * (zn[0] - zo[0]), zo[1] + theta * (zn[1] - zo[1])],
                            [vo[0] + theta_v * (vn[0] - vo[0]), vo[1] + theta_v * (vn[1] - vo[1])],
                            coarse.solid.kind[k],
                        )
                    }
                    Src::Here(k) => (solid.zeta[k], solid.v[k], solid.kind[k]),
                };
                inside &= kind.source();
                known &= kind.known();
                for c in 0..2 {
                    z[c] += w * zz[c];
                    v[c] += w * vv[c];
                }
            }
            let k = g.node;
            solid.kind[k] = Self::derived_kind(inside, known);
            if known {
                solid.zeta[k] = z;
                solid.v[k] = v;
                solid.phi[k] = solid.shape.phi0(solid.xi(k));
            }
            solid.v_live[k] = inside;
        }
    }

    /// Sets level l's hanging nodes from their masters.
    fn update_hanging(&mut self, l: usize) {
        let Some(Level { solid, hanging, .. }) = self.levels.get_mut(l) else {
            return;
        };
        for hg in hanging.iter() {
            let k = hg.node;
            let (mut z, mut v) = ([0.0; 2], [0.0; 2]);
            let (mut inside, mut known) = (true, true);
            for &(m, w) in &hg.value {
                inside &= solid.kind[m].source();
                known &= solid.kind[m].known();
                for c in 0..2 {
                    z[c] += w * solid.zeta[m][c];
                    v[c] += w * solid.v[m][c];
                }
            }
            solid.zeta[k] = z;
            solid.v[k] = v;
            solid.kind[k] = Self::derived_kind(inside, known);
            solid.phi[k] = solid.shape.phi0(solid.xi(k));
            solid.v_live[k] = solid.kind[k].source();
        }
    }

    /// Copies level l's values onto the nodes of level l − 1 it holds.
    fn restrict(&mut self, l: usize) {
        if l == 0 || l >= self.levels.len() {
            return;
        }
        let (lo, hi) = self.levels.split_at_mut(l);
        let coarse = &mut lo[l - 1];
        let fine = &hi[0];
        for &(k, f) in &coarse.held {
            let kind = fine.solid.kind[f];
            coarse.solid.zeta[k] = fine.solid.zeta[f];
            coarse.solid.v[k] = if fine.sums > 0 {
                let s = 1.0 / fine.sums as f64;
                [fine.v_sum[f][0] * s, fine.v_sum[f][1] * s]
            } else {
                fine.solid.v[f]
            };
            coarse.solid.kind[k] = Self::derived_kind(kind.source(), kind.known());
            coarse.solid.phi[k] = fine.solid.phi[f];
            coarse.solid.v_live[k] = kind.source();
        }
    }

    /// The kick (velocity step) the solid's next step takes.
    fn kick(s: &Solid) -> f64 {
        if s.started { s.dt } else { 0.5 * s.dt }
    }

    /// The time of the solid's velocity before its next step.
    fn v_time(s: &Solid) -> f64 {
        if s.started { s.time - 0.5 * s.dt } else { s.time }
    }

    /// One step of level l and, recursively, two of every finer level.
    fn advance(&mut self, l: usize) -> Result<(), Error> {
        let kick = {
            let lv = &mut self.levels[l];
            lv.zeta_old.clone_from(&lv.solid.zeta);
            lv.v_old.clone_from(&lv.solid.v);
            lv.t_old = lv.solid.time;
            lv.tv_old = Self::v_time(&lv.solid);
            Self::kick(&lv.solid)
        };
        // The degrees of freedom as the step's forces see them (a node may leave the body in it).
        let dof: Vec<bool> = {
            let lv = &self.levels[l];
            (0..lv.role.len()).map(|k| lv.role[k].dof() && lv.solid.kind[k].source()).collect()
        };
        self.levels[l].solid.step_mapped().map_err(|e| match e {
            Error::Diverged(m) => Error::Diverged(format!("level {l}: {m}")),
            e => e,
        })?;
        self.update_hanging(l);
        {
            let lv = &self.levels[l];
            let updated = (0..lv.role.len()).filter(|&k| lv.role[k].dof() && lv.solid.kind[k] == NodeKind::Solid).count();
            self.updates += updated as u64;
            if let Some(h) = &lv.solid.level {
                for k in 0..lv.role.len() {
                    if dof[k] {
                        let f = [h.force[k][0] + h.ext_force[k][0], h.force[k][1] + h.ext_force[k][1]];
                        self.audit.0[0] += kick * f[0];
                        self.audit.0[1] += kick * f[1];
                        self.audit.1 += kick * (f[0].abs() + f[1].abs());
                    }
                }
            }
        }
        if l + 1 >= self.levels.len() {
            return Ok(());
        }
        // The flux register: the force of the coarse leaves on each interface node, applied to
        // the finer level's master at each of its kicks, and the coarse side's solid area. The
        // coarse kick at tⁿ stands for the force over [tⁿ − Δt/2, tⁿ + Δt/2]; the fine kicks at
        // tⁿ and tⁿ + Δt/2 for [tⁿ − Δt/4, tⁿ + 3Δt/4]. Holding Fⁿ through both would lag the
        // coupling by a quarter coarse step (a first-order error: a P wave reflects in
        // proportion to h/λ); the register is extrapolated in time from the last two coarse
        // forces instead, F(tⁿ + sΔt) = Fⁿ + s (Fⁿ − Fⁿ⁻¹). The impulse that adds over a coarse
        // step, (Δt/4)(Fⁿ − Fⁿ⁻¹), is momentum in transit between the two levels' leapfrog
        // clocks: it telescopes, (Δt/4)(Fᴺ − F⁰) in all, and never accumulates.
        let regs: Vec<(usize, [f64; 2], [f64; 2], f64)> = {
            let lv = &self.levels[l];
            match &lv.solid.level {
                Some(h) => lv
                    .links
                    .iter()
                    .enumerate()
                    .map(|(i, &(k, m))| {
                        let prev =
                            if lv.solid.started && lv.reg_prev.len() == lv.links.len() && lv.reg_prev_started { lv.reg_prev[i] } else { h.force[k] };
                        (m, h.force[k], prev, h.area[k])
                    })
                    .collect(),
                None => Vec::new(),
            }
        };
        {
            let lv = &mut self.levels[l];
            lv.reg_prev = regs.iter().map(|r| r.1).collect();
            // A force from a half kick (the first step) is not a full step's sample.
            lv.reg_prev_started = kick == lv.solid.dt;
        }
        {
            let fine = &mut self.levels[l + 1];
            if let Some(h) = fine.solid.level.as_mut() {
                for &(m, _, _, a) in &regs {
                    h.ext_area[m] = a;
                }
            }
            fine.v_sum.iter_mut().for_each(|v| *v = [0.0; 2]);
            fine.sums = 0;
        }
        let (t_old, t_new) = (self.levels[l].t_old, self.levels[l].solid.time);
        let (tv_old, tv_new) = (self.levels[l].tv_old, Self::v_time(&self.levels[l].solid));
        let ratio = if self.refinement.subcycle { 2 } else { 1 };
        for _ in 0..ratio {
            let f = &self.levels[l + 1].solid;
            let theta = if t_new > t_old { (f.time - t_old) / (t_new - t_old) } else { 1.0 };
            let theta_v = if tv_new > tv_old { (Self::v_time(f) - tv_old) / (tv_new - tv_old) } else { 1.0 };
            let fine_kick = Self::kick(f);
            {
                let fine = &mut self.levels[l + 1];
                if let Some(h) = fine.solid.level.as_mut() {
                    for &(m, fnow, fprev, _) in &regs {
                        for c in 0..2 {
                            let e = fnow[c] + theta * (fnow[c] - fprev[c]);
                            h.ext_force[m][c] = e;
                            self.audit.2[c] += fine_kick * (e - fnow[c]);
                        }
                    }
                }
            }
            self.fill_ghosts(l + 1, theta, theta_v);
            self.advance(l + 1)?;
            let fine = &mut self.levels[l + 1];
            for (acc, v) in fine.v_sum.iter_mut().zip(&fine.solid.v) {
                acc[0] += v[0];
                acc[1] += v[1];
            }
            fine.sums += 1;
        }
        self.restrict(l + 1);
        Ok(())
    }

    /// The level set at x from the current state: at a node, the finest level's value there;
    /// else ζ bilinear in the finest leaf cell holding x, or, where that cell has corners
    /// outside the band, the least φ of its known corners (+∞ without any).
    fn sample_phi(&self, views: &[LevelView], x: [f64; 2]) -> f64 {
        if let Some(z) = self.node_zeta(views, x) {
            return self.shape.phi0([x[0] + z[0], x[1] + z[1]]);
        }
        for l in (0..views.len()).rev() {
            let Some((c0, t)) = views[l].leaf(x) else {
                continue;
            };
            let s = &self.levels[l].solid;
            let Some(c) = s.grid.neighbour(c0, 1, 0).zip(s.grid.neighbour(c0, 1, 1)).zip(s.grid.neighbour(c0, 0, 1)).map(|((a, b), d)| [c0, a, b, d])
            else {
                return f64::INFINITY;
            };
            if c.iter().any(|&k| !s.kind[k].known()) {
                return c.iter().filter(|&&k| s.kind[k].known()).map(|&k| s.phi[k]).fold(f64::INFINITY, f64::min);
            }
            let w = [(1.0 - t[0]) * (1.0 - t[1]), t[0] * (1.0 - t[1]), t[0] * t[1], (1.0 - t[0]) * t[1]];
            let mut xi = x;
            for q in 0..4 {
                xi[0] += w[q] * s.zeta[c[q]][0];
                xi[1] += w[q] * s.zeta[c[q]][1];
            }
            return self.shape.phi0(xi);
        }
        f64::INFINITY
    }

    /// ζ at a node of the finest level that has a node at x (not a ghost) with a known value.
    fn node_zeta(&self, views: &[LevelView], x: [f64; 2]) -> Option<[f64; 2]> {
        let o = self.base.origin();
        for l in (0..views.len()).rev() {
            let s = &self.levels[l].solid;
            let h = s.grid.h();
            let (u, v) = ((x[0] - o[0]) / h, (x[1] - o[1]) / h);
            let (ri, rj) = (u.round(), v.round());
            if (u - ri).abs() < 1e-9
                && (v - rj).abs() < 1e-9
                && ri >= 0.0
                && rj >= 0.0
                && let Some(k) = s.grid.try_index(ri as usize, rj as usize)
                && views[l].role[k] != Role::Ghost
                && s.kind[k].known()
            {
                return Some(s.zeta[k]);
            }
        }
        None
    }

    /// ζ at x from the current state: the value of the finest level that has a node there, else
    /// biquadratic (bilinear when the window is incomplete) in the finest leaf holding x.
    fn sample_zeta(&self, views: &[LevelView], x: [f64; 2]) -> Option<[f64; 2]> {
        if let Some(z) = self.node_zeta(views, x) {
            return Some(z);
        }
        for l in (0..views.len()).rev() {
            let Some((c0, t)) = views[l].leaf(x) else {
                continue;
            };
            let s = &self.levels[l].solid;
            let (i, j) = s.grid.ij(c0);
            // Centred 3 × 3 window around the nearer nodes.
            let (si, sj) = (if t[0] < 0.5 { -1 } else { 0 }, if t[1] < 0.5 { -1 } else { 0 });
            let (wx, wy) = (lagrange_weights(3, t[0] - si as f64), lagrange_weights(3, t[1] - sj as f64));
            let mut z = [0.0; 2];
            let mut ok = true;
            'w: for q in 0..3 {
                for p in 0..3 {
                    match s.grid.lattice_shift(i, j, si + p as isize, sj + q as isize).and_then(|(a, b)| s.grid.try_index(a, b)) {
                        Some(k) if s.kind[k].known() => {
                            z[0] += wx[p] * wy[q] * s.zeta[k][0];
                            z[1] += wx[p] * wy[q] * s.zeta[k][1];
                        }
                        _ => {
                            ok = false;
                            break 'w;
                        }
                    }
                }
            }
            if ok {
                return Some(z);
            }
            let c = [Some(c0), s.grid.neighbour(c0, 1, 0), s.grid.neighbour(c0, 1, 1), s.grid.neighbour(c0, 0, 1)];
            let w = [(1.0 - t[0]) * (1.0 - t[1]), t[0] * (1.0 - t[1]), t[0] * t[1], (1.0 - t[0]) * t[1]];
            let mut z = [0.0; 2];
            for q in 0..4 {
                let k = c[q].filter(|&k| s.kind[k].known())?;
                z[0] += w[q] * s.zeta[k][0];
                z[1] += w[q] * s.zeta[k][1];
            }
            return Some(z);
        }
        None
    }

    /// Rebuilds the levels around the current surface and moves the state onto them: ζ from
    /// the finest old level holding each point; v by the overlap of old and new control
    /// volumes, so Σ m v over the degrees of freedom is unchanged (masses of full cells; the
    /// cut cells lie on the finest level on both sides and keep their nodes).
    pub fn regrid(&mut self) -> Result<RegridReport, Error> {
        self.synchronise()?;
        let started = self.levels.first().is_some_and(|lv| lv.solid.started);
        let time = self.time();
        let old_views: Vec<LevelView> = self
            .levels
            .iter()
            .filter_map(|lv| Some(LevelView::new(&lv.solid.grid, &lv.role, &lv.solid.level.as_ref()?.own_cell, &lv.hanging, &lv.links)))
            .collect();
        if old_views.len() != self.levels.len() {
            return Err(Error::Invalid("a level without its hooks".into()));
        }
        let plans = self.plan(&|x| self.sample_phi(&old_views, x))?;
        let topo = self.topologies(&plans)?;
        // ζ.
        let mut states = Vec::with_capacity(topo.len());
        for t in &topo {
            let n = t.grid.len();
            let mut st = LevelState { zeta: vec![[0.0; 2]; n], v: vec![[0.0; 2]; n], known: vec![false; n] };
            for k in 0..n {
                if let Some(z) = self.sample_zeta(&old_views, t.grid.position(k)) {
                    st.zeta[k] = z;
                    st.known[k] = true;
                }
            }
            states.push(st);
        }
        // v: momentum of every piece of the old control volumes handed to the new ones.
        let new_views: Vec<LevelView> = topo.iter().map(|t| LevelView::new(&t.grid, &t.role, &t.own_cell, &t.hanging, &t.links)).collect();
        let deepest = |views: &[LevelView], plans_cx: usize| -> Vec<Option<usize>> {
            let mut out = vec![None::<usize>; plans_cx];
            for (l, vw) in views.iter().enumerate() {
                for k in 0..vw.grid.len() {
                    if vw.own_cell[k] {
                        let (i, j) = vw.grid.ij(k);
                        let c = (i >> l) + plans[0].cx * (j >> l);
                        if let Some(o) = out.get_mut(c) {
                            *o = Some(o.map_or(l, |m: usize| m.max(l)));
                        }
                    }
                }
            }
            out
        };
        let ncells = plans[0].cx * plans[0].cy;
        let (dold, dnew) = (deepest(&old_views, ncells), deepest(&new_views, ncells));
        let mut mom: Vec<Vec<([f64; 2], f64)>> = topo.iter().map(|t| vec![([0.0; 2], 0.0); t.grid.len()]).collect();
        let mut report = RegridReport::default();
        let o = self.base.origin();
        let h0 = self.base.h();
        for c in 0..ncells {
            let Some(lmax) = dold[c].max(dnew[c]) else {
                continue;
            };
            let (ci, cj) = (c % plans[0].cx, c / plans[0].cx);
            let m = 1usize << (lmax + 1);
            let sub = h0 / m as f64;
            for q in 0..m {
                for p in 0..m {
                    let x = [o[0] + (ci as f64 + (p as f64 + 0.5) / m as f64) * h0, o[1] + (cj as f64 + (q as f64 + 0.5) / m as f64) * h0];
                    let old = owners(&old_views, x);
                    let mut pv = [0.0; 2];
                    let mut mass = 0.0;
                    for &(l, k, w) in &old {
                        let s = &self.levels[l].solid;
                        if s.kind[k].source() {
                            pv[0] += w * s.v[k][0];
                            pv[1] += w * s.v[k][1];
                            mass += w;
                        }
                    }
                    if !(mass > 0.0) {
                        continue;
                    }
                    let a = sub * sub;
                    report.momentum[0] += a * pv[0];
                    report.momentum[1] += a * pv[1];
                    report.mass += a * mass;
                    let new = owners(&new_views, x);
                    let taken: f64 = new.iter().map(|e| e.2).sum();
                    report.unassigned += (1.0 - taken).abs() * a * mass;
                    for (l, k, w) in new {
                        let e = &mut mom[l][k];
                        e.0[0] += w * a * pv[0];
                        e.0[1] += w * a * pv[1];
                        e.1 += w * a * mass;
                    }
                }
            }
        }
        for (l, t) in topo.iter().enumerate() {
            for k in 0..t.grid.len() {
                let (p, m) = mom[l][k];
                if m > 0.0 {
                    states[l].v[k] = [p[0] / m, p[1] / m];
                }
            }
        }
        // Nodes without mass of their own (outside, ghosts, held) take the nearest old value.
        for (l, t) in topo.iter().enumerate() {
            for k in 0..t.grid.len() {
                if mom[l][k].1 > 0.0 {
                    continue;
                }
                let x = t.grid.position(k);
                let own = owners(&old_views, x);
                if let Some(&(ol, ok, _)) = own.first() {
                    states[l].v[k] = self.levels[ol].solid.v[ok];
                }
            }
        }
        drop(new_views);
        drop(old_views);
        self.levels = self.instantiate(topo, states, started, time)?;
        self.synchronise()?;
        self.regrids += 1;
        Ok(report)
    }

    /// Advances `steps` coarse steps (each with the finer levels subcycled).
    pub fn run(&mut self, steps: usize) -> Result<(), Error> {
        if steps > self.limits.max_steps {
            return Err(Error::TooLarge(format!("{steps} steps, the cap is {}", self.limits.max_steps)));
        }
        for _ in 0..steps {
            self.audit = ([0.0; 2], 0.0, [0.0; 2]);
            self.advance(0)?;
            self.steps += 1;
            if self.refinement.regrid_every > 0 && self.steps.is_multiple_of(self.refinement.regrid_every) {
                let r = self.regrid()?;
                if r.unassigned > 1e-12 * r.mass {
                    return Err(Error::Diverged(format!("a regrid left {} of {} of the momentum's mass without a node", r.unassigned, r.mass)));
                }
            }
        }
        Ok(())
    }

    /// The composite equilibrium residual of the current configuration: the net force on every
    /// degree of freedom from the cells of all levels (hanging nodes' shares and the coarse
    /// side of the interface nodes included), with its control volume's solid area. All levels
    /// at the same time; ghosts and held nodes are made consistent first.
    pub fn residual(&mut self) -> Result<Vec<ResidualEntry>, Error> {
        self.synchronise()?;
        let mut forces = Vec::with_capacity(self.levels.len());
        for lv in &mut self.levels {
            if let Some(h) = lv.solid.level.as_mut() {
                h.ext_force.iter_mut().for_each(|f| *f = [0.0; 2]);
            }
            forces.push(lv.solid.level_forces()?);
        }
        for l in 0..self.levels.len().saturating_sub(1) {
            for &(k, m) in &self.levels[l].links {
                let (f, a) = (forces[l].0[k], forces[l].1[k]);
                forces[l + 1].0[m][0] += f[0];
                forces[l + 1].0[m][1] += f[1];
                forces[l + 1].1[m] += a;
            }
        }
        let mut out = Vec::new();
        for (l, lv) in self.levels.iter().enumerate() {
            for k in 0..lv.role.len() {
                if lv.role[k].dof() && lv.solid.kind[k].source() {
                    out.push(ResidualEntry {
                        node: NodeRef { level: l, index: k, position: lv.solid.grid.position(k) },
                        force: forces[l].0[k],
                        area: forces[l].1[k],
                    });
                }
            }
        }
        Ok(out)
    }

    /// The degrees of freedom inside the body.
    pub fn dofs(&self) -> Vec<NodeRef> {
        let mut out = Vec::new();
        for (l, lv) in self.levels.iter().enumerate() {
            for k in 0..lv.role.len() {
                if lv.role[k].dof() && lv.solid.kind[k].source() {
                    out.push(NodeRef { level: l, index: k, position: lv.solid.grid.position(k) });
                }
            }
        }
        out
    }

    /// The solid of level l.
    pub fn level(&self, l: usize) -> Option<&Solid> {
        self.levels.get(l).map(|lv| &lv.solid)
    }

    /// The role of node k of level l.
    pub fn role(&self, l: usize, k: usize) -> Option<Role> {
        self.levels.get(l)?.role.get(k).copied()
    }

    /// Number of levels (refinement levels + 1).
    pub fn level_count(&self) -> usize {
        self.levels.len()
    }

    /// The finest level's solid (where the surface is).
    pub fn finest(&self) -> Option<&Solid> {
        self.levels.last().map(|lv| &lv.solid)
    }

    /// The stress at the surface points (finest level).
    pub fn surface_stress(&self) -> Vec<SurfaceSample> {
        self.finest().map(|s| s.surface_stress()).unwrap_or_default()
    }

    /// The stress at a degree of freedom.
    pub fn stress(&self, node: &NodeRef) -> Option<Stress> {
        self.level(node.level)?.stress(node.index)
    }

    /// The largest speed of a degree of freedom.
    pub fn max_speed(&self) -> f64 {
        self.dofs().iter().filter_map(|n| self.level(n.level)?.velocity(n.index)).map(|v| v[0].hypot(v[1])).fold(0.0, f64::max)
    }

    /// The coarse time step.
    pub fn dt(&self) -> f64 {
        self.levels.first().map_or(0.0, |lv| lv.solid.dt)
    }

    /// The simulated time.
    pub fn time(&self) -> f64 {
        self.levels.first().map_or(0.0, |lv| lv.solid.time)
    }

    /// The impulse audit of the last coarse step: (Σ impulse on all degrees of freedom,
    /// Σ |impulse|, the part of the first from the registers' extrapolation in time). For a
    /// free body without loads the first equals the third to round-off, and the third
    /// telescopes over the steps (momentum in transit between the levels' clocks).
    pub fn impulse_audit(&self) -> ([f64; 2], f64, [f64; 2]) {
        self.audit
    }

    /// The leaf cells of every level: (level, lower-left corner, side), for plotting.
    pub fn leaf_cells(&self) -> Vec<(usize, [f64; 2], f64)> {
        let mut out = Vec::new();
        for (l, lv) in self.levels.iter().enumerate() {
            let Some(h) = &lv.solid.level else {
                continue;
            };
            for k in 0..lv.role.len() {
                if h.own_cell[k] {
                    out.push((l, lv.solid.grid.position(k), lv.solid.grid.h()));
                }
            }
        }
        out
    }

    /// Counters.
    pub fn stats(&self) -> HierarchyStats {
        HierarchyStats {
            nodes: self.levels.iter().map(|lv| lv.role.len()).collect(),
            dofs: self.levels.iter().map(|lv| (0..lv.role.len()).filter(|&k| lv.role[k].dof() && lv.solid.kind[k].source()).count()).collect(),
            leaves: self.levels.iter().map(|lv| lv.solid.level.as_ref().map_or(0, |h| h.own_cell.iter().filter(|&&c| c).count())).collect(),
            updates: self.updates,
            steps: self.steps,
            regrids: self.regrids,
        }
    }
}
