//! The state of an Eulerian solid on the grid and its explicit time step.
//!
//! Per node: the reference map as ζ = ξ − x, the velocity v, the level set φ = φ₀(ξ) and the
//! node's kind. Solid nodes (φ < 0) advect ζ; band nodes (a few layers outside) get ζ by
//! extrapolation every step ([`crate::extrapolate`]). The default surface, [`Boundary::Mapped`],
//! integrates the momentum at the body's nodes over their cut control volumes
//! ([`crate::mapped`]). The step of the other surfaces, below, integrates it at every known
//! node with H_ε(φ) < 1 (the solid and the inner part of the blur zone; the body's nodes only
//! for [`Boundary::BlurredBody`] and [`Boundary::Ghost`]), and the velocity is extrapolated to
//! the rest of the band.
//!
//! One step of the blurred and ghost-filled surfaces (central differences, as the explicit FEM
//! solver: v at half steps, the first step advancing it by half a step from the initial state):
//! 1. edge stresses from ζ and v: σ = (1 − H) (σ_solid(F) + μ_e (∇v + ∇vᵀ)) + H σ_ambient;
//! 2. v ← (v + Δt ((∇·σ + f)/ρ − (v·∇)v)) / (1 + γ Δt) at the momentum nodes (the damping is
//!    implicit, so it never limits Δt; Δt/2 in the first step);
//! 3. ζ ← ζ − Δt (v + (v·∇)ζ) at the solid nodes with the new v (from ∂ξ/∂t + (v·∇)ξ = 0);
//! 4. φ from the new ζ; nodes that left the body drop out, the band is rebuilt, band nodes
//!    with φ < 0 (material that entered) become solid, and v is extrapolated to known nodes
//!    that did not carry it.

use crate::extrapolate::{fit, fit_weights};
use crate::grid::Grid;
use crate::material::{Material, Stress, Tensor};
use crate::par;
use crate::shape::Shape;
use crate::sharp;
use crate::{Error, Limits, finite, positive};
use std::sync::Arc;

/// What a node holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeKind {
    /// Inside the body: ζ advected.
    Solid,
    /// Inside the body, ζ and v held (a clamp).
    Fixed,
    /// Outside the body, ζ extrapolated from the solid.
    Band,
    /// No material and not in the band.
    Empty,
}

impl NodeKind {
    pub(crate) fn known(self) -> bool {
        self != NodeKind::Empty
    }
    pub(crate) fn source(self) -> bool {
        matches!(self, NodeKind::Solid | NodeKind::Fixed)
    }
}

/// Numerical settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Settings {
    /// Δt = cfl · h / c_P (the stable range is below about 0.7 for this stencil).
    pub cfl: f64,
    /// κ of the artificial viscosity μ_e = κ ρ₀ c_P h (Rycroft et al. 2020 use 0.4 with their
    /// incompressible scheme).
    pub viscosity: f64,
    /// Mass-proportional damping rate γ (1/time), for relaxing to static equilibrium.
    pub damping: f64,
    /// Half-width ε of the blurred surface, in grid spacings.
    pub blur: f64,
    /// Number of node layers in the extrapolation band.
    pub band_layers: usize,
    /// How the surface carries its traction.
    pub boundary: Boundary,
    /// Highest degree (2 or 3) of the ghost fit of [`Boundary::Ghost`].
    pub ghost_degree: usize,
}

/// The treatment of the body's surface. [`Boundary::Mapped`] is the default and the one the
/// static solve and the later stages build on; the others are kept for the comparison studies
/// (docs/research/reference-map.md).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Boundary {
    /// The blurred interface of the published RMT: σ = (1 − H_ε) σ_solid + H_ε σ_ambient on
    /// every edge, the momentum integrated wherever H_ε < 1 (Valkov et al. 2015; Rycroft et al.
    /// 2020), including band nodes outside the body whose ζ is extrapolated. First order at a
    /// loaded surface, and its static answer depends on the damping.
    Blurred,
    /// The blurred stress, with the momentum integrated only at the body's nodes (φ < 0) and
    /// v extrapolated into the band. The band nodes of [`Boundary::Blurred`] never come to
    /// rest (their ζ does not follow their velocity), so its static answer depends on the
    /// damping and the artificial viscosity; this variant does not, and exposes the bias of
    /// the extrapolated ζ.
    BlurredBody,
    /// A sharp traction boundary by ghost values: the solid stress alone on every edge that
    /// touches the body, the first ring of nodes outside filled so that σ·n = σ_ambient·n
    /// holds at the surface (module `sharp`), the momentum at the body's nodes only. Kept for
    /// comparison studies.
    Ghost,
    /// Sharp cut cells (`crate::mapped`): momentum only at nodes inside the body, each over the
    /// solid part of its dual cell, the traction imposed on the curved surface, and the
    /// deformation gradient near the surface from a surface-fitted map. Second order.
    #[default]
    Mapped,
}

impl Boundary {
    /// Whether edge stresses are blended with the ambient stress.
    pub(crate) fn blends(self) -> bool {
        matches!(self, Boundary::Blurred | Boundary::BlurredBody)
    }

    /// Whether the momentum is integrated at the body's nodes only.
    pub(crate) fn body_momentum(self) -> bool {
        self != Boundary::Blurred
    }
}

impl Default for Settings {
    fn default() -> Self {
        Settings { cfl: 0.4, viscosity: 0.1, damping: 0.0, blur: 1.5, band_layers: 4, boundary: Boundary::Mapped, ghost_degree: 3 }
    }
}

impl Settings {
    fn validate(&self) -> Result<(), Error> {
        let cfl = positive("cfl", self.cfl)?;
        if cfl > 1.0 {
            return Err(Error::Invalid(format!("cfl must be at most 1, got {cfl}")));
        }
        if !(self.viscosity >= 0.0 && self.viscosity <= 10.0) {
            return Err(Error::Invalid(format!("viscosity κ must lie in [0, 10], got {}", self.viscosity)));
        }
        if !(self.damping >= 0.0) || !self.damping.is_finite() {
            return Err(Error::Invalid(format!("damping must be a non-negative number, got {}", self.damping)));
        }
        if !(self.blur >= 0.5 && self.blur <= 4.0) {
            return Err(Error::Invalid(format!("blur must lie in [0.5, 4] grid spacings, got {}", self.blur)));
        }
        if self.band_layers < 2 || self.band_layers > 8 {
            return Err(Error::Invalid(format!("band_layers must lie in 2..=8, got {}", self.band_layers)));
        }
        if !(2..=3).contains(&self.ghost_degree) {
            return Err(Error::Invalid(format!("ghost_degree must be 2 or 3, got {}", self.ghost_degree)));
        }
        if self.boundary.blends() && (self.band_layers as f64) < self.blur + 2.0 {
            return Err(Error::Invalid("the band must reach past the blur zone by two layers".into()));
        }
        Ok(())
    }
}

/// The outcome of [`Solid::relax`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Relaxation {
    /// Steps taken.
    pub steps: usize,
    /// Whether the speed fell below the tolerance.
    pub converged: bool,
    /// The largest node speed in the body at the end.
    pub max_speed: f64,
}

/// The parts of one explicit step, reported to the observer of [`Solid::run_observed`] as each
/// one finishes (for profiling; the step does the same work with or without an observer).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Phase {
    /// The stress on the faces between full cells (the compact stencil) or, for the blurred
    /// and ghost-filled surfaces, on every edge.
    InteriorFaces,
    /// The stress on the faces next to a cut cell, through the surface-fitted map.
    MappedFaces,
    /// The traction on the curved surface, the body force and the sharing of the net forces
    /// between control volumes.
    SurfaceForces,
    /// The velocity update.
    Momentum,
    /// The advection of the reference map.
    Advection,
    /// The level set, the node kinds and the extrapolation of ζ and v into the band.
    Band,
    /// The surface crossings, the cut cells and the control volumes.
    CutCells,
    /// The surface points, their matching along the surface and their normals.
    SurfacePoints,
    /// The surface values: Newton on the traction conditions of each chain.
    Newton,
}

impl Phase {
    /// Every phase, in step order.
    pub const ALL: [Phase; 9] = [
        Phase::InteriorFaces,
        Phase::MappedFaces,
        Phase::SurfaceForces,
        Phase::Momentum,
        Phase::Advection,
        Phase::Band,
        Phase::CutCells,
        Phase::SurfacePoints,
        Phase::Newton,
    ];
}

/// How deep (in grid spacings) a band node of the ghost-filled surface must lie inside the
/// body to enter it (`Solid::rebuild`).
const ENTRY_DEPTH: f64 = 0.1;

/// Largest window half-width tried by the extrapolation.
const MAX_WINDOW: isize = 4;

/// An Eulerian solid on a grid.
#[derive(Debug, Clone)]
pub struct Solid {
    pub(crate) grid: Grid,
    pub(crate) material: Material,
    pub(crate) shape: Shape,
    pub(crate) settings: Settings,
    pub(crate) limits: Limits,
    pub(crate) zeta: Vec<[f64; 2]>,
    pub(crate) v: Vec<[f64; 2]>,
    pub(crate) phi: Vec<f64>,
    pub(crate) kind: Vec<NodeKind>,
    /// Whether v at the node was integrated (or held) in the last step.
    pub(crate) v_live: Vec<bool>,
    pub(crate) ambient: Vec<[f64; 3]>,
    pub(crate) dt: f64,
    pub(crate) time: f64,
    /// Whether a dynamic step was taken (the first one advances v by half a step).
    pub(crate) started: bool,
    /// The ambient stress as a function of position, when given as one (the mapped boundary
    /// evaluates it at the surface's Gauss points; else it interpolates the node values).
    pub(crate) ambient_field: Option<Field<[f64; 3]>>,
    /// Body force per unit current volume, as a function of position.
    pub(crate) body: Option<Field<[f64; 2]>>,
    /// ζ prescribed outside the body instead of extrapolated (verification only).
    pub(crate) exterior: Option<Field<[f64; 2]>>,
    /// The cut cells and the surface of the last mapped step (for the stress accessors).
    pub(crate) cut: Option<crate::mapped::CutState>,
    /// Ghost nodes of the last rebuild that the ghost fill could not constrain.
    pub(crate) fallbacks: usize,
    /// Ghost nodes of the last rebuild filled by the linear constrained fit.
    pub(crate) linear_fills: usize,
    /// Factor on the ambient stress and the body force (the load parameter of a continuation;
    /// 1 by default). The mapped boundary applies it.
    pub(crate) load: f64,
    /// What the solid needs as one level of a refined hierarchy (`crate::refine`); `None` on a
    /// plain grid.
    pub(crate) level: Option<Box<LevelHooks>>,
}

/// A level of a refined hierarchy as the solid sees it. The hierarchy owns the nodes it sets
/// from other levels (frozen: ghosts, hanging nodes, nodes a finer level covers); the solid
/// assembles only the cells that are leaves on this level, and adds the coarser level's flux
/// and area to the nodes on the coarse–fine interface.
#[derive(Debug, Clone, Default)]
pub(crate) struct LevelHooks {
    /// Whether the cell with lower-left node k is a leaf of this level.
    pub(crate) own_cell: Vec<bool>,
    /// Nodes whose kind and values the hierarchy sets; the solid never changes them.
    pub(crate) frozen: Vec<bool>,
    /// Hanging nodes: the two interface nodes that take their mass and force, with weights.
    pub(crate) slave: Vec<Option<[(usize, f64); 2]>>,
    /// Nodes on the interface with the coarser level (never merged into a neighbour).
    pub(crate) master: Vec<bool>,
    /// Force added to each node from the coarser level's cells (the flux register).
    pub(crate) ext_force: Vec<[f64; 2]>,
    /// Solid area added to each node's control volume from the coarser level's cells.
    pub(crate) ext_area: Vec<f64>,
    /// The net force on each node from this level's cells in the last step.
    pub(crate) force: Vec<[f64; 2]>,
    /// The solid area of each node's control volume in this level's cells in the last step.
    pub(crate) area: Vec<f64>,
}

impl LevelHooks {
    /// Hooks for `n` nodes: every cell a leaf, nothing frozen.
    pub(crate) fn new(n: usize) -> LevelHooks {
        LevelHooks {
            own_cell: vec![true; n],
            frozen: vec![false; n],
            slave: vec![None; n],
            master: vec![false; n],
            ext_force: vec![[0.0; 2]; n],
            ext_area: vec![0.0; n],
            force: vec![[0.0; 2]; n],
            area: vec![0.0; n],
        }
    }
}

/// The stresses at the middle of an edge.
#[derive(Debug, Clone, Copy)]
pub(crate) struct EdgeStress {
    /// The solid's Cauchy stress (σxx, σyy, σxy).
    pub(crate) stress: [f64; 3],
    /// The artificial viscous stress.
    pub(crate) viscous: [f64; 3],
    /// det ∇ξ.
    pub(crate) det: f64,
}

/// A field given as a function of position.
#[derive(Clone)]
pub(crate) struct Field<T>(pub(crate) Arc<dyn Fn([f64; 2]) -> T + Send + Sync>);

impl<T> std::fmt::Debug for Field<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Field(..)")
    }
}

impl Solid {
    /// The body `shape`, undeformed and at rest, on `grid`.
    pub fn new(grid: Grid, material: Material, shape: Shape, settings: Settings, limits: Limits) -> Result<Solid, Error> {
        settings.validate()?;
        shape.validate()?;
        let (_, mu, rho) = material.constants();
        positive("shear modulus", mu)?;
        positive("density", rho)?;
        let n = grid.len();
        if n > limits.max_nodes {
            return Err(Error::TooLarge(format!("{n} grid nodes, the cap is {}", limits.max_nodes)));
        }
        let c = material.p_wave_speed();
        let h = grid.h();
        let mut dt = settings.cfl * h / c;
        if settings.viscosity > 0.0 {
            // Explicit diffusion limit of μ_e (∇v + ∇vᵀ): Δt ≤ ρ h² / (8 μ_e) in 2D.
            dt = dt.min(0.5 * h / (8.0 * settings.viscosity * c));
        }
        positive("time step", dt)?;
        let mut s = Solid {
            grid,
            material,
            shape,
            settings,
            limits,
            zeta: vec![[0.0; 2]; n],
            v: vec![[0.0; 2]; n],
            phi: vec![0.0; n],
            kind: vec![NodeKind::Empty; n],
            v_live: vec![false; n],
            ambient: vec![[0.0; 3]; n],
            dt,
            time: 0.0,
            started: false,
            ambient_field: None,
            body: None,
            exterior: None,
            cut: None,
            fallbacks: 0,
            linear_fills: 0,
            load: 1.0,
            level: None,
        };
        if settings.boundary == Boundary::Mapped && (s.grid.periodic()[0] || s.grid.periodic()[1]) && shape != Shape::Everywhere {
            return Err(Error::Invalid("not supported yet: a mapped surface on a periodic grid".into()));
        }
        for k in 0..n {
            if shape.phi0(s.grid.position(k)) < 0.0 {
                s.kind[k] = NodeKind::Solid;
            }
        }
        s.rebuild()?;
        s.v_live = s.kind.iter().map(|k| k.known()).collect();
        Ok(s)
    }

    /// Sets the initial reference map and velocity: `motion(x)` gives (ξ, v) at grid position x.
    /// The body is then the set of nodes with φ₀(ξ) < 0 (a pre-deformed or moving start).
    pub fn set_motion(&mut self, motion: impl Fn([f64; 2]) -> ([f64; 2], [f64; 2])) -> Result<(), Error> {
        for k in 0..self.grid.len() {
            let x = self.grid.position(k);
            let (xi, v) = motion(x);
            for c in 0..2 {
                finite("initial reference map", xi[c])?;
                finite("initial velocity", v[c])?;
            }
            self.zeta[k] = [xi[0] - x[0], xi[1] - x[1]];
            self.v[k] = v;
            let phi = self.shape.phi0(xi);
            self.kind[k] = if self.kind[k] == NodeKind::Fixed {
                NodeKind::Fixed
            } else if phi < 0.0 {
                NodeKind::Solid
            } else {
                NodeKind::Empty
            };
        }
        self.rebuild()?;
        self.v_live = self.kind.iter().map(|k| k.source()).collect();
        self.extrapolate_velocity();
        self.started = false;
        self.cut = None;
        Ok(())
    }

    /// Sets the stress outside the body (σxx, σyy, σxy) at each grid position: the surface then
    /// carries the traction σ_ambient·n. Zero (a free surface) by default.
    pub fn set_ambient(&mut self, stress: impl Fn([f64; 2]) -> [f64; 3]) -> Result<(), Error> {
        for k in 0..self.grid.len() {
            let s = stress(self.grid.position(k));
            for c in s {
                finite("ambient stress", c)?;
            }
            self.ambient[k] = s;
        }
        // The ghost values carry the traction: refit them for the new one.
        if self.settings.boundary == Boundary::Ghost {
            self.fallbacks = 0;
            self.linear_fills = 0;
            self.sharp_fill();
        }
        self.cut = None;
        Ok(())
    }

    /// Sets the stress outside the body as a function of position, kept for evaluation at any
    /// point (the mapped boundary integrates the traction σ_ambient·n along the curved surface);
    /// the node values are set from it as by [`Solid::set_ambient`].
    pub fn set_ambient_field(&mut self, stress: impl Fn([f64; 2]) -> [f64; 3] + Send + Sync + 'static) -> Result<(), Error> {
        let f: Arc<dyn Fn([f64; 2]) -> [f64; 3] + Send + Sync> = Arc::new(stress);
        let g = f.clone();
        self.set_ambient(move |x| g(x))?;
        self.ambient_field = Some(Field(f));
        self.cut = None;
        Ok(())
    }

    /// Sets a body force f per unit current volume as a function of the grid position (a
    /// centrifugal load ρω²x, gravity ρg). Zero by default. The blurred surface applies it at
    /// the nodes weighted by the material's share 1 − H(φ) (as the LBRMT weights the solid's
    /// external force); the mapped boundary integrates it over each control volume (at its
    /// centroid).
    pub fn set_body_force(&mut self, force: impl Fn([f64; 2]) -> [f64; 2] + Send + Sync + 'static) -> Result<(), Error> {
        for k in 0..self.grid.len() {
            let b = force(self.grid.position(k));
            finite("body force", b[0])?;
            finite("body force", b[1])?;
        }
        self.body = Some(Field(Arc::new(force)));
        Ok(())
    }

    /// Scales the ambient stress and the body force by `factor` (a load ramp). The mapped
    /// boundary applies it; the blurred boundary ignores it.
    pub fn set_load_factor(&mut self, factor: f64) -> Result<(), Error> {
        self.load = finite("load factor", factor)?;
        self.cut = None;
        Ok(())
    }

    /// The load factor.
    pub fn load_factor(&self) -> f64 {
        self.load
    }

    /// For verification only: prescribes ζ = ξ − x outside the body (in the band) instead of
    /// extrapolating it, to measure how much a result depends on the extrapolation.
    pub fn set_exterior_map(&mut self, zeta: impl Fn([f64; 2]) -> [f64; 2] + Send + Sync + 'static) -> Result<(), Error> {
        self.exterior = Some(Field(Arc::new(zeta)));
        self.cut = None;
        self.rebuild()
    }

    /// Holds the solid nodes where `clamp(x)` is true at their current ζ and v.
    pub fn fix(&mut self, clamp: impl Fn([f64; 2]) -> bool) {
        for k in 0..self.grid.len() {
            if self.kind[k] == NodeKind::Solid && clamp(self.grid.position(k)) {
                self.kind[k] = NodeKind::Fixed;
                self.v_live[k] = true;
            }
        }
        self.cut = None;
    }

    /// The grid.
    pub fn grid(&self) -> &Grid {
        &self.grid
    }

    /// The material.
    pub fn material(&self) -> &Material {
        &self.material
    }

    /// The time step.
    pub fn dt(&self) -> f64 {
        self.dt
    }

    /// The simulated time.
    pub fn time(&self) -> f64 {
        self.time
    }

    /// The kind of node k.
    pub fn kind(&self, k: usize) -> Option<NodeKind> {
        self.kind.get(k).copied()
    }

    /// The level set φ₀(ξ) at node k (meaningful at known nodes).
    pub fn phi(&self, k: usize) -> Option<f64> {
        self.kind(k).filter(|k| k.known()).and(self.phi.get(k).copied())
    }

    /// The reference position ξ of the material at node k, at known nodes.
    pub fn reference(&self, k: usize) -> Option<[f64; 2]> {
        self.kind(k).filter(|k| k.known())?;
        let x = self.grid.position(k);
        let z = self.zeta.get(k)?;
        Some([x[0] + z[0], x[1] + z[1]])
    }

    /// The displacement u = x − ξ of the material at node k, at nodes inside the body.
    pub fn displacement(&self, k: usize) -> Option<[f64; 2]> {
        self.kind(k).filter(|k| k.source())?;
        let z = self.zeta.get(k)?;
        Some([-z[0], -z[1]])
    }

    /// The velocity at node k, at known nodes.
    pub fn velocity(&self, k: usize) -> Option<[f64; 2]> {
        self.kind(k).filter(|k| k.known())?;
        self.v.get(k).copied()
    }

    /// The solid's Cauchy stress at node k (inside the body), from ∇ξ by central differences
    /// (one-sided next to unknown nodes).
    pub fn stress(&self, k: usize) -> Option<Stress> {
        self.kind(k).filter(|k| k.source())?;
        let g = match &self.cut {
            Some(cut) if self.settings.boundary == Boundary::Mapped => cut.node_gradient(self, k).or_else(|| self.node_gradient(k))?,
            _ => self.node_gradient(k)?,
        };
        let f = invert(&identity_plus(&g))?;
        self.material.cauchy(&f).ok()
    }

    /// The ghost nodes of the last step that the ghost fill could not constrain (kept the
    /// plain extrapolation); zero for the other surfaces.
    pub fn surface_fallbacks(&self) -> usize {
        self.fallbacks
    }

    /// The ghost nodes of the last step filled by the linear (not quadratic) constrained fit.
    pub fn surface_linear_fills(&self) -> usize {
        self.linear_fills
    }

    /// The largest node speed inside the body.
    pub fn max_speed(&self) -> f64 {
        (0..self.grid.len()).filter(|&k| self.kind[k].source()).map(|k| self.v[k][0].hypot(self.v[k][1])).fold(0.0, f64::max)
    }

    /// Advances `steps` dynamic steps.
    pub fn run(&mut self, steps: usize) -> Result<(), Error> {
        self.run_observed(steps, &mut |_| {})
    }

    /// Advances `steps` dynamic steps, calling `observer` as each [`Phase`] of a step finishes
    /// (a profiler reads a clock there; the results are those of [`Solid::run`]).
    pub fn run_observed(&mut self, steps: usize, observer: &mut (dyn FnMut(Phase) + Send)) -> Result<(), Error> {
        self.check_steps(steps)?;
        for _ in 0..steps {
            self.step(observer)?;
        }
        Ok(())
    }

    /// Steps until the largest speed in the body is below `speed`, at most `max_steps` steps
    /// (checked every 20 steps).
    pub fn relax(&mut self, speed: f64, max_steps: usize) -> Result<Relaxation, Error> {
        positive("relaxation speed", speed)?;
        self.check_steps(max_steps)?;
        let mut steps = 0;
        while steps < max_steps {
            self.step(&mut |_| {})?;
            steps += 1;
            if steps % 20 == 0 && self.max_speed() < speed {
                return Ok(Relaxation { steps, converged: true, max_speed: self.max_speed() });
            }
        }
        Ok(Relaxation { steps, converged: false, max_speed: self.max_speed() })
    }

    /// Advects ζ through `steps` steps of the prescribed velocity field `velocity(x, t)`
    /// (kinematics only: no momentum balance), for checking the transport of ξ.
    pub fn advect(&mut self, steps: usize, velocity: impl Fn([f64; 2], f64) -> [f64; 2]) -> Result<(), Error> {
        self.check_steps(steps)?;
        for _ in 0..steps {
            let t = self.time + 0.5 * self.dt;
            for k in 0..self.grid.len() {
                if self.kind[k].known() && self.kind[k] != NodeKind::Fixed {
                    let v = velocity(self.grid.position(k), t);
                    finite("prescribed velocity", v[0])?;
                    finite("prescribed velocity", v[1])?;
                    self.v[k] = v;
                }
            }
            self.advect_reference()?;
            self.time += self.dt;
            self.rebuild()?;
            self.cut = None;
            // The velocity is prescribed everywhere known.
            for k in 0..self.grid.len() {
                self.v_live[k] = self.kind[k].known();
            }
        }
        Ok(())
    }

    fn check_steps(&self, steps: usize) -> Result<(), Error> {
        if steps > self.limits.max_steps {
            return Err(Error::TooLarge(format!("{steps} steps, the cap is {}", self.limits.max_steps)));
        }
        Ok(())
    }

    /// One dynamic step.
    fn step(&mut self, obs: &mut (dyn FnMut(Phase) + Send)) -> Result<(), Error> {
        // Inside the pool for the whole step: the loops then run on it without a hand-off each.
        fieldcraft_parallel::with_threads(fieldcraft_parallel::current_threads(), || self.step_on_pool(obs))
    }

    fn step_on_pool(&mut self, obs: &mut (dyn FnMut(Phase) + Send)) -> Result<(), Error> {
        if self.settings.boundary == Boundary::Mapped {
            return self.step_mapped(obs);
        }
        let n = self.grid.len();
        let (_, _, rho0) = self.material.constants();
        let h = self.grid.h();
        let dt = self.dt;
        // Central differences: v lives at half steps, so the first kick is half a step.
        let kick = if self.started { dt } else { 0.5 * dt };
        self.started = true;
        let eps = self.settings.blur * h;
        let mu_e = self.settings.viscosity * rho0 * self.material.p_wave_speed() * h;

        // 1. Edge stresses: edge d (0 = +x, 1 = +y) of node k joins k and its +d neighbour.
        let edges = par::try_map(n, |k| -> Result<([[f64; 3]; 2], [f64; 2]), Error> {
            let (s0, d0) = self.edge_stress(k, 0, eps, mu_e)?;
            let (s1, d1) = self.edge_stress(k, 1, eps, mu_e)?;
            Ok(([s0, s1], [d0, d1]))
        })?;
        let edge: Vec<[[f64; 3]; 2]> = edges.iter().map(|e| e.0).collect();
        let edge_det: Vec<[f64; 2]> = edges.iter().map(|e| e.1).collect();
        obs(Phase::InteriorFaces);

        // 2. Momentum at the momentum nodes.
        let updated = par::map(n, |k| -> Option<[f64; 2]> {
            let kind = self.kind[k];
            if kind == NodeKind::Fixed {
                return Some(self.v[k]);
            }
            // The body's own nodes, or (Blurred) every node with some material weight.
            let moves =
                if self.settings.boundary.body_momentum() { kind == NodeKind::Solid } else { kind.known() && heaviside(self.phi[k], eps) < 1.0 };
            if !moves {
                return None;
            }
            let back = |d: usize| -> [f64; 3] {
                let (di, dj) = if d == 0 { (-1, 0) } else { (0, -1) };
                match self.grid.neighbour(k, di, dj) {
                    Some(m) => edge[m][d],
                    None => self.ambient[k],
                }
            };
            let fx = edge[k][0];
            let bx = back(0);
            let fy = edge[k][1];
            let by = back(1);
            // ∇·σ: x-edges carry (σxx, σxy), y-edges (σxy, σyy).
            let mut div = [(fx[0] - bx[0] + fy[2] - by[2]) / h, (fx[2] - bx[2] + fy[1] - by[1]) / h];
            if let Some(b) = &self.body {
                // With a blended stress the force acts on the material's share of the node.
                let w = if self.settings.boundary.blends() { 1.0 - heaviside(self.phi[k], eps) } else { 1.0 };
                let f = (b.0)(self.grid.position(k));
                div = [div[0] + w * f[0], div[1] + w * f[1]];
            }
            // ρ = ρ₀ det ∇ξ, averaged over the node's edges with a solid stress.
            let mut dets = 0.0;
            let mut count = 0.0;
            for det in [edge_det[k][0], edge_det[k][1]] {
                if det.is_finite() {
                    dets += det;
                    count += 1.0;
                }
            }
            for (di, dj, d) in [(-1, 0, 0), (0, -1, 1)] {
                if let Some(m) = self.grid.neighbour(k, di, dj)
                    && edge_det[m][d].is_finite()
                {
                    dets += edge_det[m][d];
                    count += 1.0;
                }
            }
            let rho = if count > 0.0 { rho0 * dets / count } else { rho0 };
            let v = self.v[k];
            let adv = [self.upwind_dot(k, v, |m| self.v[m][0]), self.upwind_dot(k, v, |m| self.v[m][1])];
            let mut out = [0.0; 2];
            for c in 0..2 {
                out[c] = (v[c] + kick * (div[c] / rho - adv[c])) / (1.0 + self.settings.damping * kick);
            }
            Some(out)
        });
        let live: Vec<bool> = updated.iter().map(Option::is_some).collect();
        let v_new: Vec<[f64; 2]> = updated.iter().zip(&self.v).map(|(u, v)| u.unwrap_or(*v)).collect();
        self.v = v_new;
        obs(Phase::Momentum);

        // 3. Reference map with the new velocity.
        self.advect_reference()?;
        self.time += dt;
        obs(Phase::Advection);

        // 4. Body, band and velocities.
        self.rebuild()?;
        self.v_live = par::map(n, |k| live[k] && self.kind[k].known());
        self.extrapolate_velocity();
        self.check_finite()?;
        obs(Phase::Band);
        Ok(())
    }

    /// ζ ← ζ − Δt (v + (v·∇)ζ) at the solid nodes.
    pub(crate) fn advect_reference(&mut self) -> Result<(), Error> {
        let z_new = par::map(self.grid.len(), |k| {
            if self.kind[k] != NodeKind::Solid {
                return self.zeta[k];
            }
            let v = self.v[k];
            let a = [self.upwind_dot(k, v, |m| self.zeta[m][0]), self.upwind_dot(k, v, |m| self.zeta[m][1])];
            let mut z = self.zeta[k];
            for c in 0..2 {
                z[c] = self.zeta[k][c] - self.dt * (v[c] + a[c]);
            }
            z
        });
        self.zeta = z_new;
        Ok(())
    }

    /// An error naming the first known node whose ζ or v is not a finite number.
    pub(crate) fn check_finite(&self) -> Result<(), Error> {
        let bad = par::first(self.grid.len(), |k| self.kind[k].known() && self.v[k].iter().chain(self.zeta[k].iter()).any(|x| !x.is_finite()));
        match bad {
            Some(k) => Err(Error::Diverged(format!("non-finite value at node {k} at t = {}", self.time))),
            None => Ok(()),
        }
    }

    /// (v·∇)a at node k by second-order one-sided upwind differences of the field `a`, first
    /// order next to unknown nodes.
    pub(crate) fn upwind_dot(&self, k: usize, v: [f64; 2], a: impl Fn(usize) -> f64) -> f64 {
        let h = self.grid.h();
        let known = |m: Option<usize>| m.filter(|&m| self.kind[m].known());
        let mut sum = 0.0;
        for d in 0..2 {
            let vd = v[d];
            if vd == 0.0 {
                continue;
            }
            let s: isize = if vd > 0.0 { -1 } else { 1 };
            let off = |o: isize| if d == 0 { (o, 0) } else { (0, o) };
            let (i1, j1) = off(s);
            let (i2, j2) = off(2 * s);
            let up1 = known(self.grid.neighbour(k, i1, j1));
            let up2 = known(self.grid.neighbour(k, i2, j2));
            let a0 = a(k);
            // Derivative along +d.
            let deriv = match (up1, up2) {
                (Some(m1), Some(m2)) => -(s as f64) * (3.0 * a0 - 4.0 * a(m1) + a(m2)) / (2.0 * h),
                (Some(m1), None) => -(s as f64) * (a0 - a(m1)) / h,
                _ => {
                    let (i3, j3) = off(-s);
                    match known(self.grid.neighbour(k, i3, j3)) {
                        Some(m3) => (a(m3) - a0) / (-(s as f64) * h),
                        None => 0.0,
                    }
                }
            };
            sum += vd * deriv;
        }
        sum
    }

    /// The stress on edge d of node k and det ∇ξ there (NaN without a solid stress).
    fn edge_stress(&self, k: usize, d: usize, eps: f64, mu_e: f64) -> Result<([f64; 3], f64), Error> {
        let (ei, ej) = if d == 0 { (1, 0) } else { (0, 1) };
        let Some(b) = self.grid.neighbour(k, ei, ej) else {
            return Ok((self.ambient[k], f64::NAN));
        };
        let amb = avg3(self.ambient[k], self.ambient[b]);
        let ka = self.kind[k];
        let kb = self.kind[b];
        if !ka.known() || !kb.known() {
            return Ok((amb, f64::NAN));
        }
        let hv = if !self.settings.boundary.blends() {
            // The solid stress alone, on every edge with a body node at either end.
            if !ka.source() && !kb.source() {
                return Ok((amb, f64::NAN));
            }
            0.0
        } else {
            heaviside(0.5 * (self.phi[k] + self.phi[b]), eps)
        };
        if hv >= 1.0 {
            return Ok((amb, f64::NAN));
        }
        let Some(e) = self.compact_edge(k, d, mu_e)? else {
            return Ok((amb, f64::NAN));
        };
        let w = 1.0 - hv;
        let (s, visc) = (e.stress, e.viscous);
        let out = [w * (s[0] + visc[0]) + hv * amb[0], w * (s[1] + visc[1]) + hv * amb[1], w * (s[2] + visc[2]) + hv * amb[2]];
        Ok((out, e.det))
    }

    /// The solid and viscous stress at the middle of edge d of node k from the compact stencil
    /// (the two nodes of the edge and their four transverse neighbours); `None` when one of
    /// them is unknown.
    pub(crate) fn compact_edge(&self, k: usize, d: usize, mu_e: f64) -> Result<Option<EdgeStress>, Error> {
        let (ei, ej, ti, tj) = if d == 0 { (1, 0, 0, 1) } else { (0, 1, 1, 0) };
        let Some(b) = self.grid.neighbour(k, ei, ej).filter(|&b| self.kind[b].known() && self.kind[k].known()) else {
            return Ok(None);
        };
        let side = |m: usize, s: isize| self.grid.neighbour(m, s * ti, s * tj).filter(|&q| self.kind[q].known());
        let (Some(ap), Some(am), Some(bp), Some(bm)) = (side(k, 1), side(k, -1), side(b, 1), side(b, -1)) else {
            return Ok(None);
        };
        self.compact_edge_at(k, d, [b, ap, am, bp, bm], mu_e).map(Some)
    }

    /// The nodes of the compact stencil of edge d of node k: its other end b and the transverse
    /// neighbours of k and b ([b, k₊, k₋, b₊, b₋]); `None` when one of them is unknown.
    pub(crate) fn compact_nodes(&self, k: usize, d: usize) -> Option<[usize; 5]> {
        let (ei, ej, ti, tj) = if d == 0 { (1, 0, 0, 1) } else { (0, 1, 1, 0) };
        let b = self.grid.neighbour(k, ei, ej).filter(|&b| self.kind[b].known() && self.kind[k].known())?;
        let side = |m: usize, s: isize| self.grid.neighbour(m, s * ti, s * tj).filter(|&q| self.kind[q].known());
        Some([b, side(k, 1)?, side(k, -1)?, side(b, 1)?, side(b, -1)?])
    }

    /// [`Solid::compact_edge`] on given stencil nodes ([`Solid::compact_nodes`]).
    pub(crate) fn compact_edge_at(&self, k: usize, d: usize, nodes: [usize; 5], mu_e: f64) -> Result<EdgeStress, Error> {
        let [b, ap, am, bp, bm] = nodes;
        let h = self.grid.h();
        // Gradients at the edge: along the edge direction from (k, b), across it from the four
        // transverse neighbours. g[r][c] = ∂field_r/∂x_c.
        let grad = |f: &dyn Fn(usize) -> [f64; 2]| -> Tensor {
            let mut g = [[0.0; 2]; 2];
            for r in 0..2 {
                let along = (f(b)[r] - f(k)[r]) / h;
                let across = (f(ap)[r] + f(bp)[r] - f(am)[r] - f(bm)[r]) / (4.0 * h);
                g[r][d] = along;
                g[r][1 - d] = across;
            }
            g
        };
        let gz = grad(&|m| self.zeta[m]);
        let jinv = identity_plus(&gz);
        let det = jinv[0][0] * jinv[1][1] - jinv[0][1] * jinv[1][0];
        let f =
            invert(&jinv).ok_or_else(|| Error::Diverged(format!("the reference map folded at node {k} (det ∇ξ = {det}) at t = {}", self.time)))?;
        let s = self.material.cauchy(&f).map_err(|e| Error::Diverged(format!("{e} at node {k}, t = {}", self.time)))?;
        let gv = grad(&|m| self.v[m]);
        let viscous = [2.0 * mu_e * gv[0][0], 2.0 * mu_e * gv[1][1], mu_e * (gv[0][1] + gv[1][0])];
        Ok(EdgeStress { stress: [s[0], s[1], s[2]], viscous, det })
    }

    /// ∇ζ at node k by central differences, one-sided next to unknown nodes.
    pub(crate) fn node_gradient(&self, k: usize) -> Option<Tensor> {
        let h = self.grid.h();
        let known = |m: Option<usize>| m.filter(|&m| self.kind[m].known());
        let mut g = [[0.0; 2]; 2];
        for c in 0..2 {
            let (di, dj) = if c == 0 { (1, 0) } else { (0, 1) };
            let p = known(self.grid.neighbour(k, di, dj));
            let m = known(self.grid.neighbour(k, -di, -dj));
            for r in 0..2 {
                g[r][c] = match (p, m) {
                    (Some(p), Some(m)) => (self.zeta[p][r] - self.zeta[m][r]) / (2.0 * h),
                    (Some(p), None) => (self.zeta[p][r] - self.zeta[k][r]) / h,
                    (None, Some(m)) => (self.zeta[k][r] - self.zeta[m][r]) / h,
                    (None, None) => return None,
                };
            }
        }
        Some(g)
    }

    /// Updates φ, drops nodes that left the body, rebuilds the band and admits entering material.
    pub(crate) fn rebuild(&mut self) -> Result<(), Error> {
        let n = self.grid.len();
        let updated = par::map(n, |k| {
            if self.frozen(k) {
                // Set by the refined hierarchy: φ only.
                let phi = if self.kind[k].known() { self.shape.phi0(self.xi(k)) } else { self.phi[k] };
                return (self.kind[k], phi);
            }
            match self.kind[k] {
                NodeKind::Solid | NodeKind::Fixed => {
                    let phi = self.shape.phi0(self.xi(k));
                    let leaves = self.kind[k] == NodeKind::Solid && !(phi < 0.0);
                    (if leaves { NodeKind::Empty } else { self.kind[k] }, phi)
                }
                NodeKind::Band | NodeKind::Empty => (NodeKind::Empty, self.phi[k]),
            }
        });
        for (k, (kind, phi)) in updated.into_iter().enumerate() {
            self.kind[k] = kind;
            self.phi[k] = phi;
        }
        let source: Vec<bool> = par::map(n, |k| self.kind[k].source());
        if !source.iter().any(|&s| s) {
            return Err(Error::Diverged("no grid node lies inside the body".into()));
        }
        let level = self.level.as_deref();
        let frozen = |k: usize| level.is_some_and(|l| l.frozen.get(k).copied().unwrap_or(false));
        let filled = layered_fill(&self.grid, &source, &|k| !frozen(k), self.settings.band_layers, &mut self.zeta);
        if let Some(f) = &self.exterior {
            for &k in &filled {
                self.zeta[k] = (f.0)(self.grid.position(k));
            }
        }
        // Material entering: a band node whose extrapolated ξ lies inside the body. For the
        // ghost-filled surface with hysteresis: a body node leaves at φ ≥ 0 but a band node
        // enters only at φ < −h/10. Without it a node within a hair of a surface that settles
        // on it flips between the two discrete systems (each puts the surface on the other side
        // of the node by O(h²)) and keeps the body ringing (seen in the finite-strain
        // cylinder); with it both states are consistent and either is an O(h²) answer.
        let ghost = self.settings.boundary == Boundary::Ghost;
        let enter = if ghost { -ENTRY_DEPTH * self.grid.h() } else { 0.0 };
        let phis = par::map_items(filled.len(), |i| filled.get(i).map_or(0.0, |&k| self.shape.phi0(self.xi(k))));
        for (&k, phi) in filled.iter().zip(phis) {
            self.phi[k] = phi;
            self.kind[k] = if phi < enter { NodeKind::Solid } else { NodeKind::Band };
        }
        self.fallbacks = 0;
        self.linear_fills = 0;
        if ghost {
            self.sharp_fill();
        }
        Ok(())
    }

    /// Refills the ghost ring (band nodes 8-adjacent to the body) with the traction-constrained
    /// fit, then the outer band layers by the linear extrapolation from the body and the ghosts.
    /// Every ghost is fitted from body nodes only, so the fits run in parallel.
    fn sharp_fill(&mut self) {
        let n = self.grid.len();
        let source: Vec<bool> = par::map(n, |k| self.kind[k].source());
        let ghosts = par::filter(n, |k| {
            self.kind[k] == NodeKind::Band && (-1..=1).any(|dj| (-1..=1).any(|di| self.grid.neighbour(k, di, dj).is_some_and(|m| source[m])))
        });
        let body = sharp::Body {
            grid: &self.grid,
            shape: &self.shape,
            material: &self.material,
            source: &source,
            zeta: &self.zeta,
            ambient: &self.ambient,
            passes: 2,
            degree: self.settings.ghost_degree,
        };
        let values = par::map_items(ghosts.len(), |i| ghosts.get(i).and_then(|&g| sharp::ghost(&body, g)));
        let mut have = source.clone();
        for (&g, value) in ghosts.iter().zip(values) {
            match value {
                Some((z, kind)) => {
                    if kind == sharp::Fill::Linear {
                        self.linear_fills += 1;
                    }
                    self.zeta[g] = z
                }
                None => self.fallbacks += 1,
            }
            have[g] = true;
        }
        // The outer layers from the body and the new ghost values; their kinds are kept.
        let kind = &self.kind;
        let ghost_set = &have;
        layered_fill(&self.grid, &have, &|k| kind[k] == NodeKind::Band && !ghost_set[k], self.settings.band_layers.saturating_sub(1), &mut self.zeta);
        let band = par::filter(n, |k| self.kind[k] == NodeKind::Band);
        let phis = par::map_items(band.len(), |i| band.get(i).map_or(0.0, |&k| self.shape.phi0(self.xi(k))));
        for (&k, phi) in band.iter().zip(phis) {
            self.phi[k] = phi;
        }
    }

    /// The band fill of the current node kinds as a linear map ([`FillPlan`]).
    pub(crate) fn fill_plan(&self) -> FillPlan {
        let source: Vec<bool> = self.kind.iter().map(|k| k.source()).collect();
        let sources = (0..self.grid.len()).filter(|&k| source[k]).collect();
        let band = layered_plan(&self.grid, &source, self.settings.band_layers);
        FillPlan { sources, band }
    }

    /// As [`Solid::rebuild`] with the node kinds held and the band refilled by a plan made for
    /// them: φ at the body and band nodes, and no node enters or leaves the body. Returns the
    /// number of nodes whose kind no longer matches the sign of φ (body nodes with φ ≥ 0, band
    /// nodes with φ < 0), which a static solve treats as a change of the cut topology.
    pub(crate) fn refill_planned(&mut self, plan: &FillPlan) -> usize {
        let mut flips = 0;
        for &k in &plan.sources {
            self.phi[k] = self.shape.phi0(self.xi(k));
            if self.kind[k] == NodeKind::Solid && !(self.phi[k] < 0.0) {
                flips += 1;
            }
        }
        for (k, weights) in &plan.band {
            let mut z = [0.0; 2];
            for &(m, w) in weights {
                z[0] += w * self.zeta[m][0];
                z[1] += w * self.zeta[m][1];
            }
            self.zeta[*k] = match &self.exterior {
                Some(f) => (f.0)(self.grid.position(*k)),
                None => z,
            };
        }
        for (k, _) in &plan.band {
            if self.kind[*k] == NodeKind::Band {
                self.phi[*k] = self.shape.phi0(self.xi(*k));
                if self.phi[*k] < 0.0 {
                    flips += 1;
                }
            }
        }
        flips
    }

    /// Extrapolates v from the nodes that carry it to the other known nodes.
    pub(crate) fn extrapolate_velocity(&mut self) {
        let n = self.grid.len();
        let source: Vec<bool> = par::map(n, |k| self.v_live[k] && self.kind[k].known());
        let kind = &self.kind;
        let level = self.level.as_deref();
        let target = |k: usize| kind[k].known() && !level.is_some_and(|l| l.frozen.get(k).copied().unwrap_or(false));
        let filled = layered_fill(&self.grid, &source, &target, 2 * self.settings.band_layers, &mut self.v);
        let mut done = source;
        for &k in &filled {
            done[k] = true;
        }
        par::update(&mut self.v, |k, v| {
            if target(k) && !done[k] {
                // Isolated: no velocity information nearby.
                *v = [0.0; 2];
            }
        });
    }

    /// Whether the hierarchy sets node k (never changed by this solid).
    pub(crate) fn frozen(&self, k: usize) -> bool {
        self.level.as_ref().is_some_and(|l| l.frozen.get(k).copied().unwrap_or(false))
    }

    /// Whether the cell with lower-left node c0 is assembled on this grid.
    pub(crate) fn own_cell(&self, c0: usize) -> bool {
        self.level.as_ref().is_none_or(|l| l.own_cell.get(c0).copied().unwrap_or(false))
    }

    pub(crate) fn xi(&self, k: usize) -> [f64; 2] {
        let x = self.grid.position(k);
        [x[0] + self.zeta[k][0], x[1] + self.zeta[k][1]]
    }
}

/// Fills `field` at target nodes within `layers` layers (8-neighbour rings) of the source
/// nodes, ring by ring, each node by a linear fit to the nodes of earlier rings (so the result
/// does not depend on the node order). Returns the filled nodes, ascending within each ring.
fn layered_fill(grid: &Grid, source: &[bool], target: &(dyn Fn(usize) -> bool + Sync), layers: usize, field: &mut [[f64; 2]]) -> Vec<usize> {
    let n = grid.len();
    let mut have = source.to_vec();
    let mut filled = Vec::new();
    let next_to_have = |k: usize, have: &[bool]| (-1..=1).any(|dj| (-1..=1).any(|di| grid.neighbour(k, di, dj).is_some_and(|m| have[m])));
    // The first ring by a search of the grid; each later one among the nodes next to the
    // last ring (a node joins a ring once a neighbour has a value, so the rings are the same).
    let mut ring = par::filter(n, |k| !have[k] && target(k) && next_to_have(k, &have));
    // Which layer last listed a node, to list each candidate once.
    let mut listed = vec![usize::MAX; n];
    for layer in 0..layers {
        if layer > 0 {
            ring.retain(|&k| {
                let fresh = listed[k] != layer;
                listed[k] = layer;
                fresh && !have[k]
            });
            ring.sort_unstable();
            let candidates = ring;
            ring = par::filter(candidates.len(), |i| target(candidates[i]) && next_to_have(candidates[i], &have))
                .into_iter()
                .map(|i| candidates[i])
                .collect();
        }
        if ring.is_empty() {
            break;
        }
        let f: &[[f64; 2]] = field;
        let values = par::map_items(ring.len(), |i| {
            let k = ring.get(i).copied()?;
            let mut samples = Vec::with_capacity(81);
            for r in 2..=MAX_WINDOW {
                samples.clear();
                for dj in -r..=r {
                    for di in -r..=r {
                        if let Some(m) = grid.neighbour(k, di, dj)
                            && have[m]
                        {
                            samples.push(([di as f64, dj as f64], f[m]));
                        }
                    }
                }
                if let Some(v) = fit(&samples) {
                    return Some(v);
                }
            }
            None
        });
        let mut next = Vec::with_capacity(4 * ring.len());
        for (&k, value) in ring.iter().zip(values) {
            match value {
                Some(v) => {
                    field[k] = v;
                    have[k] = true;
                    filled.push(k);
                    for dj in -1..=1 {
                        for di in -1..=1 {
                            next.extend(grid.neighbour(k, di, dj));
                        }
                    }
                }
                // Tried again in the next ring.
                None => next.push(k),
            }
        }
        ring = next;
    }
    filled
}

/// The band fill of [`layered_fill`] for fixed source nodes, as a linear map: the band nodes in
/// ring order, each with the weights of its samples (nodes of the body or of earlier rings).
/// The fit's weights depend only on which nodes are known, so while the body's node set is
/// held, refilling the band is one sparse product per band node.
#[derive(Debug, Clone, Default)]
pub(crate) struct FillPlan {
    sources: Vec<usize>,
    band: Vec<(usize, Vec<(usize, f64)>)>,
}

impl FillPlan {
    /// The band nodes in fill order, each with the weights of the nodes it is filled from.
    pub(crate) fn band(&self) -> &[(usize, Vec<(usize, f64)>)] {
        &self.band
    }
}

/// The plan of [`layered_fill`] with every node a target.
fn layered_plan(grid: &Grid, source: &[bool], layers: usize) -> Vec<(usize, Vec<(usize, f64)>)> {
    let n = grid.len();
    let mut have = source.to_vec();
    let mut out = Vec::new();
    // Candidates of the next ring: the unknown 8-neighbours of the nodes known so far, starting
    // from the body's own boundary.
    let mut frontier: Vec<usize> = (0..n).filter(|&k| have[k]).collect();
    let mut offsets = Vec::with_capacity(81);
    let mut nodes = Vec::with_capacity(81);
    for _ in 0..layers {
        let mut ring: Vec<usize> = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for &k in &frontier {
            for dj in -1..=1 {
                for di in -1..=1 {
                    if let Some(m) = grid.neighbour(k, di, dj)
                        && !have[m]
                        && seen.insert(m)
                    {
                        ring.push(m);
                    }
                }
            }
        }
        ring.sort_unstable();
        if ring.is_empty() {
            break;
        }
        let mut entries = Vec::with_capacity(ring.len());
        for &k in &ring {
            let mut entry = None;
            for r in 2..=MAX_WINDOW {
                offsets.clear();
                nodes.clear();
                for dj in -r..=r {
                    for di in -r..=r {
                        if let Some(m) = grid.neighbour(k, di, dj)
                            && have[m]
                        {
                            offsets.push([di as f64, dj as f64]);
                            nodes.push(m);
                        }
                    }
                }
                if let Some(w) = fit_weights(&offsets) {
                    entry = Some(nodes.iter().copied().zip(w).filter(|e| e.1 != 0.0).collect::<Vec<_>>());
                    break;
                }
            }
            entries.push(entry);
        }
        frontier.clear();
        for (&k, entry) in ring.iter().zip(entries) {
            if let Some(e) = entry {
                have[k] = true;
                frontier.push(k);
                out.push((k, e));
            }
        }
    }
    out
}

/// The smoothed Heaviside of Rycroft et al. (2020): 0 for φ ≤ −ε, 1 for φ ≥ ε.
pub fn heaviside(phi: f64, eps: f64) -> f64 {
    if phi <= -eps {
        0.0
    } else if phi >= eps {
        1.0
    } else {
        0.5 * (1.0 + phi / eps + (std::f64::consts::PI * phi / eps).sin() / std::f64::consts::PI)
    }
}

pub(crate) fn identity_plus(g: &Tensor) -> Tensor {
    [[1.0 + g[0][0], g[0][1]], [g[1][0], 1.0 + g[1][1]]]
}

pub(crate) fn invert(a: &Tensor) -> Option<Tensor> {
    let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    if !(det > 0.0) || !det.is_finite() {
        return None;
    }
    Some([[a[1][1] / det, -a[0][1] / det], [-a[1][0] / det, a[0][0] / det]])
}

fn avg3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]), 0.5 * (a[2] + b[2])]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block() -> Solid {
        let l = Limits::default();
        let g = Grid::new(24, 24, 0.1, [-1.15, -1.15], [false; 2], &l).unwrap();
        let m = Material::neo_hookean(100.0, 0.3, 1.0).unwrap();
        Solid::new(g, m, Shape::Rectangle { min: [-0.5, -0.5], max: [0.5, 0.5] }, Settings::default(), l).unwrap()
    }

    #[test]
    fn an_unloaded_body_stays_at_rest_exactly() {
        let mut s = block();
        s.run(50).unwrap();
        assert_eq!(s.max_speed(), 0.0);
    }

    /// A uniform body force accelerates the middle of a free block at f/ρ (no strain there
    /// until the surface's signal arrives).
    #[test]
    fn a_body_force_accelerates_the_body() {
        let mut s = block();
        s.set_body_force(|_| [2.0, -1.0]).unwrap();
        s.run(4).unwrap();
        let k = s.grid().index(11, 11);
        let t = s.time() - 0.5 * s.dt();
        let v = s.velocity(k).unwrap();
        // To the density change of the moving body's first steps (1e-6 relative).
        assert!((v[0] / (2.0 * t) - 1.0).abs() < 1e-5 && (v[1] / -t - 1.0).abs() < 1e-5, "{v:?} at t = {t}");
        assert!(s.set_body_force(|_| [f64::NAN, 0.0]).is_err());
    }

    #[test]
    fn hostile_settings_and_inputs_are_refused() {
        let l = Limits { max_nodes: 10_000, max_steps: 100 };
        let g = Grid::new(10, 10, 0.1, [0.0; 2], [false; 2], &l).unwrap();
        let m = Material::neo_hookean(100.0, 0.3, 1.0).unwrap();
        let sh = Shape::Rectangle { min: [0.2, 0.2], max: [0.7, 0.7] };
        for bad in [
            Settings { cfl: f64::NAN, ..Settings::default() },
            Settings { cfl: 2.0, ..Settings::default() },
            Settings { viscosity: -1.0, ..Settings::default() },
            Settings { damping: f64::INFINITY, ..Settings::default() },
            Settings { blur: 0.0, ..Settings::default() },
            Settings { band_layers: 100, ..Settings::default() },
            Settings { ghost_degree: 4, ..Settings::default() },
        ] {
            assert!(Solid::new(g.clone(), m, sh, bad, l).is_err(), "{bad:?}");
        }
        let outside = Shape::Rectangle { min: [5.0, 5.0], max: [6.0, 6.0] };
        assert!(Solid::new(g.clone(), m, outside, Settings::default(), l).is_err());
        let mut s = Solid::new(g, m, sh, Settings::default(), l).unwrap();
        assert!(s.run(101).is_err());
        assert!(s.relax(f64::NAN, 10).is_err());
        assert!(s.set_ambient(|_| [f64::NAN, 0.0, 0.0]).is_err());
        assert!(s.set_motion(|x| (x, [f64::INFINITY, 0.0])).is_err());
        // A folded reference map is reported, never a panic.
        s.set_motion(|x| ([x[0], 0.45 + 0.2 * (40.0 * x[0]).sin()], [0.0, 0.0])).ok();
        let _ = s.run(5);
    }
}
