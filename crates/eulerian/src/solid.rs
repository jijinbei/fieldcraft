//! The state of an Eulerian solid on the grid and its explicit time step.
//!
//! Per node: the reference map as ζ = ξ − x, the velocity v, the level set φ = φ₀(ξ) and the
//! node's kind. Solid nodes (φ < 0) advect ζ; band nodes (a few layers outside) get ζ by
//! extrapolation every step ([`crate::extrapolate`]); the momentum balance is integrated at
//! every known node with H_ε(φ) < 1 (the solid and the inner part of the blur zone), and the
//! velocity is extrapolated to the rest of the band.
//!
//! One step (central differences, as the explicit FEM solver: v at half steps, the first step
//! advancing it by half a step from the initial state):
//! 1. edge stresses from ζ and v: σ = (1 − H) (σ_solid(F) + μ_e (∇v + ∇vᵀ)) + H σ_ambient;
//! 2. v ← (v + Δt ((∇·σ)/ρ − (v·∇)v)) / (1 + γ Δt) at the momentum nodes (the damping is
//!    implicit, so it never limits Δt; Δt/2 in the first step);
//! 3. ζ ← ζ − Δt (v + (v·∇)ζ) at the solid nodes with the new v (from ∂ξ/∂t + (v·∇)ξ = 0);
//! 4. φ from the new ζ; nodes that left the body drop out, the band is rebuilt, band nodes
//!    with φ < 0 (material that entered) become solid, and v is extrapolated to known nodes
//!    that did not carry it.

use crate::extrapolate::fit;
use crate::grid::Grid;
use crate::material::{Material, Stress, Tensor};
use crate::shape::Shape;
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
    fn known(self) -> bool {
        self != NodeKind::Empty
    }
    fn source(self) -> bool {
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
}

/// The treatment of the body's surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Boundary {
    /// The blurred interface: σ = (1 − H_ε) σ_solid + H_ε σ_ambient on every edge, the
    /// momentum integrated wherever H_ε < 1 (Valkov et al. 2015; Rycroft et al. 2020),
    /// including band nodes outside the body whose ζ is extrapolated.
    #[default]
    Blurred,
    /// The blurred stress, with the momentum integrated only at the body's nodes (φ < 0) and
    /// v extrapolated into the band. The band nodes of [`Boundary::Blurred`] never come to
    /// rest (their ζ does not follow their velocity), so its static answer depends on the
    /// damping and the artificial viscosity; this variant does not, and exposes the bias of
    /// the extrapolated ζ.
    BlurredBody,
}

impl Boundary {
    /// Whether the momentum is integrated at the body's nodes only.
    pub(crate) fn body_momentum(self) -> bool {
        self != Boundary::Blurred
    }
}

impl Default for Settings {
    fn default() -> Self {
        Settings { cfl: 0.4, viscosity: 0.1, damping: 0.0, blur: 1.5, band_layers: 4, boundary: Boundary::Blurred }
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
        if (self.band_layers as f64) < self.blur + 2.0 {
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

/// Largest window half-width tried by the extrapolation.
const MAX_WINDOW: isize = 4;

/// An Eulerian solid on a grid.
#[derive(Debug, Clone)]
pub struct Solid {
    grid: Grid,
    material: Material,
    shape: Shape,
    settings: Settings,
    limits: Limits,
    zeta: Vec<[f64; 2]>,
    v: Vec<[f64; 2]>,
    phi: Vec<f64>,
    kind: Vec<NodeKind>,
    /// Whether v at the node was integrated (or held) in the last step.
    v_live: Vec<bool>,
    ambient: Vec<[f64; 3]>,
    dt: f64,
    time: f64,
    /// Whether a dynamic step was taken (the first one advances v by half a step).
    started: bool,
    /// ζ prescribed outside the body instead of extrapolated (verification only).
    exterior: Option<Field<[f64; 2]>>,
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
            exterior: None,
        };
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
        Ok(())
    }

    /// For verification only: prescribes ζ = ξ − x outside the body (in the band) instead of
    /// extrapolating it, to measure how much a result depends on the extrapolation.
    pub fn set_exterior_map(&mut self, zeta: impl Fn([f64; 2]) -> [f64; 2] + Send + Sync + 'static) -> Result<(), Error> {
        self.exterior = Some(Field(Arc::new(zeta)));
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
        let g = self.node_gradient(k)?;
        let f = invert(&identity_plus(&g))?;
        self.material.cauchy(&f).ok()
    }

    /// The largest node speed inside the body.
    pub fn max_speed(&self) -> f64 {
        (0..self.grid.len()).filter(|&k| self.kind[k].source()).map(|k| self.v[k][0].hypot(self.v[k][1])).fold(0.0, f64::max)
    }

    /// Advances `steps` dynamic steps.
    pub fn run(&mut self, steps: usize) -> Result<(), Error> {
        self.check_steps(steps)?;
        for _ in 0..steps {
            self.step()?;
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
            self.step()?;
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
    fn step(&mut self) -> Result<(), Error> {
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
        let mut edge = vec![[[0.0f64; 3]; 2]; n];
        let mut edge_det = vec![[f64::NAN; 2]; n];
        for k in 0..n {
            for d in 0..2 {
                let (s, det) = self.edge_stress(k, d, eps, mu_e)?;
                edge[k][d] = s;
                edge_det[k][d] = det;
            }
        }

        // 2. Momentum at the momentum nodes.
        let mut live = vec![false; n];
        let mut v_new = self.v.clone();
        for k in 0..n {
            let kind = self.kind[k];
            if kind == NodeKind::Fixed {
                live[k] = true;
                continue;
            }
            // The body's own nodes, or (Blurred) every node with some material weight.
            let moves =
                if self.settings.boundary.body_momentum() { kind == NodeKind::Solid } else { kind.known() && heaviside(self.phi[k], eps) < 1.0 };
            if !moves {
                continue;
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
            let div = [(fx[0] - bx[0] + fy[2] - by[2]) / h, (fx[2] - bx[2] + fy[1] - by[1]) / h];
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
            v_new[k] = out;
            live[k] = true;
        }
        self.v = v_new;

        // 3. Reference map with the new velocity.
        self.advect_reference()?;
        self.time += dt;

        // 4. Body, band and velocities.
        self.rebuild()?;
        self.v_live = (0..n).map(|k| live[k] && self.kind[k].known()).collect();
        self.extrapolate_velocity();
        for k in 0..n {
            if self.kind[k].known() && self.v[k].iter().chain(self.zeta[k].iter()).any(|x| !x.is_finite()) {
                return Err(Error::Diverged(format!("non-finite value at node {k} at t = {}", self.time)));
            }
        }
        Ok(())
    }

    /// ζ ← ζ − Δt (v + (v·∇)ζ) at the solid nodes.
    fn advect_reference(&mut self) -> Result<(), Error> {
        let n = self.grid.len();
        let mut z_new = self.zeta.clone();
        for k in 0..n {
            if self.kind[k] != NodeKind::Solid {
                continue;
            }
            let v = self.v[k];
            let a = [self.upwind_dot(k, v, |m| self.zeta[m][0]), self.upwind_dot(k, v, |m| self.zeta[m][1])];
            for c in 0..2 {
                z_new[k][c] = self.zeta[k][c] - self.dt * (v[c] + a[c]);
            }
        }
        self.zeta = z_new;
        Ok(())
    }

    /// (v·∇)a at node k by second-order one-sided upwind differences of the field `a`, first
    /// order next to unknown nodes.
    fn upwind_dot(&self, k: usize, v: [f64; 2], a: impl Fn(usize) -> f64) -> f64 {
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
        let (ei, ej, ti, tj) = if d == 0 { (1, 0, 0, 1) } else { (0, 1, 1, 0) };
        let Some(b) = self.grid.neighbour(k, ei, ej) else {
            return Ok((self.ambient[k], f64::NAN));
        };
        let amb = avg3(self.ambient[k], self.ambient[b]);
        let ka = self.kind[k];
        let kb = self.kind[b];
        if !ka.known() || !kb.known() {
            return Ok((amb, f64::NAN));
        }
        let phi = 0.5 * (self.phi[k] + self.phi[b]);
        let hv = heaviside(phi, eps);
        if hv >= 1.0 {
            return Ok((amb, f64::NAN));
        }
        let side = |m: usize, s: isize| self.grid.neighbour(m, s * ti, s * tj).filter(|&q| self.kind[q].known());
        let (Some(ap), Some(am), Some(bp), Some(bm)) = (side(k, 1), side(k, -1), side(b, 1), side(b, -1)) else {
            return Ok((amb, f64::NAN));
        };
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
        let visc = [2.0 * mu_e * gv[0][0], 2.0 * mu_e * gv[1][1], mu_e * (gv[0][1] + gv[1][0])];
        let w = 1.0 - hv;
        let out = [w * (s[0] + visc[0]) + hv * amb[0], w * (s[1] + visc[1]) + hv * amb[1], w * (s[2] + visc[2]) + hv * amb[2]];
        Ok((out, det))
    }

    /// ∇ζ at node k by central differences, one-sided next to unknown nodes.
    fn node_gradient(&self, k: usize) -> Option<Tensor> {
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
    fn rebuild(&mut self) -> Result<(), Error> {
        let n = self.grid.len();
        for k in 0..n {
            match self.kind[k] {
                NodeKind::Solid | NodeKind::Fixed => {
                    let xi = self.xi(k);
                    self.phi[k] = self.shape.phi0(xi);
                    if self.kind[k] == NodeKind::Solid && !(self.phi[k] < 0.0) {
                        self.kind[k] = NodeKind::Empty;
                    }
                }
                NodeKind::Band | NodeKind::Empty => self.kind[k] = NodeKind::Empty,
            }
        }
        if !self.kind.iter().any(|k| k.source()) {
            return Err(Error::Diverged("no grid node lies inside the body".into()));
        }
        let source: Vec<bool> = self.kind.iter().map(|k| k.source()).collect();
        let target = vec![true; n];
        let filled = layered_fill(&self.grid, &source, &target, self.settings.band_layers, &mut self.zeta);
        if let Some(f) = &self.exterior {
            for k in 0..n {
                if filled[k] {
                    self.zeta[k] = (f.0)(self.grid.position(k));
                }
            }
        }
        for k in 0..n {
            if filled[k] {
                self.kind[k] = NodeKind::Band;
                self.phi[k] = self.shape.phi0(self.xi(k));
                if self.phi[k] < 0.0 {
                    self.kind[k] = NodeKind::Solid;
                }
            }
        }
        Ok(())
    }

    /// Extrapolates v from the nodes that carry it to the other known nodes.
    fn extrapolate_velocity(&mut self) {
        let n = self.grid.len();
        let source: Vec<bool> = (0..n).map(|k| self.v_live[k] && self.kind[k].known()).collect();
        let target: Vec<bool> = (0..n).map(|k| self.kind[k].known()).collect();
        let filled = layered_fill(&self.grid, &source, &target, 2 * self.settings.band_layers, &mut self.v);
        for k in 0..n {
            if self.kind[k].known() && !source[k] && !filled[k] {
                // Isolated: no velocity information nearby.
                self.v[k] = [0.0; 2];
            }
        }
    }

    fn xi(&self, k: usize) -> [f64; 2] {
        let x = self.grid.position(k);
        [x[0] + self.zeta[k][0], x[1] + self.zeta[k][1]]
    }
}

/// Fills `field` at target nodes within `layers` layers (8-neighbour rings) of the source
/// nodes, ring by ring, each node by a linear fit to the nodes of earlier rings (so the result
/// does not depend on the node order). Returns the filled nodes.
fn layered_fill(grid: &Grid, source: &[bool], target: &[bool], layers: usize, field: &mut [[f64; 2]]) -> Vec<bool> {
    let n = grid.len();
    let mut have = source.to_vec();
    let mut filled = vec![false; n];
    let mut samples = Vec::with_capacity(81);
    for _ in 0..layers {
        let ring: Vec<usize> = (0..n)
            .filter(|&k| !have[k] && target[k])
            .filter(|&k| (-1..=1).any(|dj| (-1..=1).any(|di| grid.neighbour(k, di, dj).is_some_and(|m| have[m]))))
            .collect();
        if ring.is_empty() {
            break;
        }
        let mut values = Vec::with_capacity(ring.len());
        for &k in &ring {
            let mut value = None;
            for r in 2..=MAX_WINDOW {
                samples.clear();
                for dj in -r..=r {
                    for di in -r..=r {
                        if let Some(m) = grid.neighbour(k, di, dj)
                            && have[m]
                        {
                            samples.push(([di as f64, dj as f64], field[m]));
                        }
                    }
                }
                value = fit(&samples);
                if value.is_some() {
                    break;
                }
            }
            values.push(value);
        }
        for (&k, value) in ring.iter().zip(values) {
            if let Some(v) = value {
                field[k] = v;
                have[k] = true;
                filled[k] = true;
            }
        }
    }
    filled
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

fn identity_plus(g: &Tensor) -> Tensor {
    [[1.0 + g[0][0], g[0][1]], [g[1][0], 1.0 + g[1][1]]]
}

fn invert(a: &Tensor) -> Option<Tensor> {
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
