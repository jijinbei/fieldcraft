//! Static equilibrium on the grid: Newton–Krylov with a geometric multigrid preconditioner.
//!
//! Dynamic relaxation ([`Solid::relax`]) reaches equilibrium in a number of steps that grows
//! as 1/h (the explicit step is bounded by h/c_P while the slowest mode's period is fixed).
//! This module solves the equilibrium directly. In statics the reference map is the inverse
//! motion, ξ(x) = X(x); nothing is advected.
//!
//! - **Unknowns**: ζ = ξ − x at the body nodes that hold a control volume (a merged node moves
//!   with the node it joined; held nodes are not unknowns).
//! - **Residual**: the net force on every control volume of the mapped boundary
//!   ([`crate::mapped`]): the face fluxes of σ(F), the traction on the curved surface and the
//!   body force, at the current load factor. The surface values ζ_S come from their own
//!   traction conditions (Newton per chain of surface points) inside every evaluation, and the
//!   band outside the body is refilled from the body nodes, so R(ζ) is the dynamic scheme's
//!   force at rest and R = 0 is its equilibrium.
//! - **Cut topology.** The body {φ₀(ξ) < 0} depends on the unknown ξ, so the set of unknowns
//!   and the cut structure change while Newton converges. Each linearisation and its line
//!   search hold the topology ([`crate::mapped::Topology`]: node kinds, which surface points
//!   carry values and how they are matched, merged control volumes); with it held, crossings,
//!   areas, normals and surface values move continuously with ζ (a body node that would leave
//!   the body puts the surface on itself). After every accepted Newton step the topology is
//!   rebuilt from φ; when it changed, the unknowns are re-indexed (a node that entered the
//!   body starts from its extrapolated ζ) and Newton continues from there. Convergence is
//!   declared only with the residual small *and* the topology consistent with the solution.
//! - **Load continuation**: the ambient stress and the body force are ramped by a load factor
//!   from 0 to the target in increments, each started from a secant predictor through the last
//!   two converged states; an increment whose Newton fails is halved.
//! - **Jacobian: finite-difference directional derivatives (JFNK)**,
//!   J v ≈ (R(ζ + ε v) − R(ζ)) / ε with ε‖v‖∞ = 10⁻⁷ h, inside GMRES. The residual contains an
//!   implicit solve (the surface conditions), root finds (the crossings, the arc midpoints) and
//!   weighted least-squares fits (the band); an analytic Jacobian would have to differentiate
//!   all of them (the surface values through the implicit function theorem), and would have to
//!   follow every change of the discretisation. The difference quotient is the exact
//!   Jacobian's action to O(ε) at one residual evaluation per Krylov iteration, so Newton keeps
//!   its quadratic convergence. Only the preconditioner needs a matrix, and an approximate one
//!   suffices.
//! - **Krylov**: restarted GMRES, right-preconditioned (the operator is non-symmetric near the
//!   surface), with the Eisenstat–Walker forcing term (choice 2).
//! - **Preconditioner matrix** ([`PreconditionerMatrix`]): by default *ghost cells*, assembled
//!   in O(n) without residual evaluations: away from the surface it is the exact Jacobian of the
//!   compact stencil (the tangent of each face flux with respect to its ∇ζ, by differences of
//!   the constitutive law); at the surface every face with solid carries the same stencil times
//!   its solid fraction, ζ at the band nodes expanded through the band fill, which ignores the
//!   cut-cell geometry, the surface map and the surface values. The alternative *probes* the
//!   rows near the surface: their Jacobian is local, so 2 s² evaluations (s = 5) perturb all
//!   unknowns of one colour (node (i, j) with i mod s, j mod s fixed) at once and attribute each
//!   row's response to the perturbed node nearest it (exact for a dependence within s/2 nodes,
//!   the far part of a column lumped onto the nearest node beyond). Both give about the same
//!   Krylov counts; the ghost cells make a rebuild after a topology change nearly free.
//! - **Multigrid**: a V-cycle on Galerkin coarse operators ([`crate::multigrid`] states the
//!   coarsening rule at the cut cells).
//! - **Rigid motions**: a body held by no node is free in the plane. Its equilibrium is then
//!   R ∈ span(MΦ) (the discrete net load accelerates it rigidly, which the relaxation sees as a
//!   rigid drift), with Φ the translations and the rotation about the centre of mass and M the
//!   lumped mass. The residual is measured as QR with Q = I − MΦ(ΦᵀMΦ)⁻¹Φᵀ (inertia relief),
//!   and updates are kept free of rigid motion with P = I − Φ(ΦᵀMΦ)⁻¹ΦᵀM, so GMRES solves
//!   QJP δ = −QR, a consistent singular system.
//! - **Reuse (item 2 of the cost plan)**: the multigrid hierarchy is kept while the topology is
//!   unchanged, across Newton steps and load increments, and rebuilt when it is stale (GMRES
//!   needs more than twice the iterations it took right after the build, or does not converge).
//!   The factorised Jacobian of each surface chain is kept the same way, so the surface solve
//!   inside each residual evaluation is a chord iteration (refactorised when it contracts
//!   slowly). A topology change always rebuilds: a kept hierarchy adapted to the new unknowns
//!   (matched by node, the new ones on their diagonal blocks) cost several times the GMRES
//!   iterations of a fresh one in our tests.

use crate::Error;
use crate::mapped::{CutState, Topology};
use crate::multigrid::{Block, BlockMatrix, Multigrid, MultigridStats, gmres};
use crate::solid::{Boundary, FillPlan, NodeKind, Solid};
use std::sync::Arc;

/// The approximate Jacobian the multigrid preconditioner is built on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreconditionerMatrix {
    /// Ghost cells: every face with solid on it carries the tangent of its compact-stencil flux
    /// times its solid fraction, with ζ at the band nodes expanded through the band fill (a
    /// linear map of the body nodes), and the dual cells' rows shared out to the control
    /// volumes as the forces are. The cut-cell geometry, the surface map and the surface values
    /// are not differentiated. O(n) to assemble, no residual evaluation.
    Ghost,
    /// Probed: interior rows from the tangent, rows near the surface from residual evaluations
    /// with [`StaticSettings::probe`] colours ([`StaticSettings::local_probing`]). Closer to
    /// the Jacobian at the surface, 2 s² residual evaluations per build.
    Probed,
}

/// Settings of the static solve.
#[derive(Debug, Clone, Copy)]
pub struct StaticSettings {
    /// Converged when ‖QR‖∞ ≤ tolerance × the force scale: the larger of the largest external
    /// force on a control volume at the full load and the largest force through one face at
    /// the current state (a body force puts O(h²) on a control volume but O(h) through its
    /// faces, and the residual cannot be resolved below the rounding of those).
    pub tolerance: f64,
    /// Load increments to start with (halved on failure, grown after easy increments).
    pub increments: usize,
    /// Smallest load increment as a fraction of the full load.
    pub min_increment: f64,
    /// Newton iterations per increment, at most.
    pub max_newton: usize,
    /// GMRES restart length.
    pub restart: usize,
    /// GMRES iterations per Newton step, at most.
    pub max_krylov: usize,
    /// Largest Eisenstat–Walker forcing term (relative linear tolerance).
    pub max_forcing: f64,
    /// Probing colour spacing s (odd, 3 ..= 9).
    pub probe: usize,
    /// Gauss–Seidel sweeps before and after each coarse correction.
    pub sweeps: usize,
    /// Keep the multigrid hierarchy while the topology is unchanged.
    pub reuse_preconditioner: bool,
    /// Keep the surface chains' factorised Jacobians (chord iteration).
    pub reuse_surface_lu: bool,
    /// Assemble the preconditioner's interior rows from the constitutive tangent and probe
    /// only the rows near the surface (else probe every row).
    pub local_probing: bool,
    /// The matrix the multigrid hierarchy is built on.
    pub matrix: PreconditionerMatrix,
    /// Topology changes allowed per increment before it counts as failed.
    pub max_topology_changes: usize,
    /// A clock in seconds for the timing counters (none: the times are zero). The crate itself
    /// never reads the time.
    pub clock: Option<fn() -> f64>,
}

impl Default for StaticSettings {
    fn default() -> Self {
        StaticSettings {
            tolerance: 1e-9,
            increments: 1,
            min_increment: 1e-3,
            max_newton: 30,
            restart: 60,
            max_krylov: 300,
            max_forcing: 1e-2,
            probe: 5,
            sweeps: 2,
            reuse_preconditioner: true,
            reuse_surface_lu: true,
            local_probing: true,
            matrix: PreconditionerMatrix::Ghost,
            max_topology_changes: 30,
            clock: None,
        }
    }
}

impl StaticSettings {
    fn validate(&self) -> Result<(), Error> {
        if !(self.tolerance > 0.0 && self.tolerance < 1.0) {
            return Err(Error::Invalid(format!("static tolerance must lie in (0, 1), got {}", self.tolerance)));
        }
        if self.increments == 0 || self.increments > 100_000 {
            return Err(Error::Invalid(format!("increments must lie in 1..=100000, got {}", self.increments)));
        }
        if !(self.min_increment > 0.0 && self.min_increment <= 1.0) {
            return Err(Error::Invalid(format!("min_increment must lie in (0, 1], got {}", self.min_increment)));
        }
        if self.probe < 3 || self.probe > 9 || self.probe.is_multiple_of(2) {
            return Err(Error::Invalid(format!("probe spacing must be odd in 3..=9, got {}", self.probe)));
        }
        if self.max_newton == 0
            || self.max_newton > 1000
            || self.restart == 0
            || self.restart > 1000
            || self.max_krylov == 0
            || self.max_krylov > 100_000
        {
            return Err(Error::Invalid("iteration caps must be positive (Newton ≤ 1000, restart ≤ 1000, Krylov ≤ 100000)".into()));
        }
        if !(self.max_forcing > 0.0 && self.max_forcing < 1.0) || self.sweeps == 0 || self.sweeps > 20 {
            return Err(Error::Invalid("max_forcing must lie in (0, 1) and sweeps in 1..=20".into()));
        }
        Ok(())
    }
}

/// One Newton step.
#[derive(Debug, Clone, PartialEq)]
pub struct NewtonStep {
    /// Load factor.
    pub load: f64,
    /// ‖QR‖∞ / force scale before the step.
    pub residual: f64,
    /// Unknown nodes.
    pub unknowns: usize,
    /// GMRES iterations.
    pub krylov: usize,
    /// The linear solve's relative residual and its target.
    pub linear_residual: f64,
    pub forcing: f64,
    /// Line search step length taken.
    pub step: f64,
    /// Whether the multigrid hierarchy was (re)built for this step.
    pub preconditioner_built: bool,
    /// Whether the step changed the cut topology.
    pub topology_changed: bool,
}

/// Times in seconds (zero without a clock).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct StaticTimes {
    /// All residual evaluations (Newton states, line search, Krylov products, probing).
    pub residual: f64,
    /// Assembling the preconditioner's matrix (the probing evaluations included).
    pub probing: f64,
    /// Multigrid setup (Galerkin products, coarsest factorisation).
    pub setup: f64,
    /// V-cycle applications.
    pub cycles: f64,
    /// Topology rebuilds.
    pub topology: f64,
    /// Parts of the held evaluations: placing the nodes and refilling the band; the cut cells
    /// with the surface solve; the forces.
    pub fill: f64,
    pub cut: f64,
    pub forces: f64,
    /// The whole solve.
    pub total: f64,
}

/// What a static solve did.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StaticReport {
    /// Whether the full load was reached and converged.
    pub converged: bool,
    /// ‖QR‖∞ / force scale at the end.
    pub residual: f64,
    /// The force scale at the end (see [`StaticSettings::tolerance`]).
    pub force_scale: f64,
    /// Load increments accepted and failed, and why each failed one did.
    pub increments: usize,
    pub failed_increments: usize,
    pub failures: Vec<String>,
    /// Every Newton step.
    pub newton: Vec<NewtonStep>,
    /// Residual evaluations: all, and those inside Krylov products and probing.
    pub evaluations: usize,
    pub krylov_evaluations: usize,
    pub probe_evaluations: usize,
    /// Preconditioner builds and topology changes.
    pub preconditioner_builds: usize,
    pub topology_changes: usize,
    /// Surface-chain factorisations and traction-condition evaluations over all evaluations.
    pub surface_factorizations: usize,
    pub surface_traction_evals: usize,
    /// The last hierarchy.
    pub multigrid: MultigridStats,
    /// Times.
    pub times: StaticTimes,
}

/// The outcome of [`Solid::static_linear_test`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LinearTest {
    /// Unknown nodes.
    pub unknowns: usize,
    /// GMRES iterations to the tolerance (or the cap).
    pub iterations: usize,
    /// The relative residual after each iteration.
    pub history: Vec<f64>,
    /// The hierarchy (empty without a preconditioner).
    pub multigrid: MultigridStats,
}

impl StaticReport {
    /// Total GMRES iterations.
    pub fn krylov_iterations(&self) -> usize {
        self.newton.iter().map(|s| s.krylov).sum()
    }
}

/// The linearisation point: a held topology, its unknowns and the state there.
struct Context {
    topo: Topology,
    /// The node holding each unknown's control volume.
    unknowns: Vec<usize>,
    /// The nodes each unknown moves (itself and the nodes merged into it).
    members: Vec<Vec<usize>>,
    /// ζ at every node at the linearisation point.
    base: Vec<[f64; 2]>,
    cut: CutState,
    /// The raw residual at the linearisation point.
    r: Vec<f64>,
    rigid: Option<Rigid>,
    /// The band fill of the held body node set.
    plan: Arc<FillPlan>,
}

/// The rigid motions of a free body, for the projections Q and P.
struct Rigid {
    /// Φ, three columns over the unknowns.
    phi: [Vec<f64>; 3],
    /// Lumped mass per unknown component.
    mass: Vec<f64>,
    /// (ΦᵀMΦ)⁻¹.
    gram_inv: [[f64; 3]; 3],
    centre: [f64; 2],
}

impl Rigid {
    /// Q r = r − MΦ (ΦᵀMΦ)⁻¹ Φᵀ r.
    fn residual(&self, r: &[f64]) -> Vec<f64> {
        let a = self.coefficients(&self.phi.each_ref().map(|p| dot(p, r)));
        let mut out = r.to_vec();
        for (i, o) in out.iter_mut().enumerate() {
            *o -= self.mass[i] * (a[0] * self.phi[0][i] + a[1] * self.phi[1][i] + a[2] * self.phi[2][i]);
        }
        out
    }

    /// P v = v − Φ (ΦᵀMΦ)⁻¹ Φᵀ M v.
    fn update(&self, v: &[f64]) -> Vec<f64> {
        let mv: Vec<f64> = v.iter().zip(&self.mass).map(|(a, m)| a * m).collect();
        let a = self.coefficients(&self.phi.each_ref().map(|p| dot(p, &mv)));
        let mut out = v.to_vec();
        for (i, o) in out.iter_mut().enumerate() {
            *o -= a[0] * self.phi[0][i] + a[1] * self.phi[1][i] + a[2] * self.phi[2][i];
        }
        out
    }

    fn coefficients(&self, t: &[f64; 3]) -> [f64; 3] {
        let g = &self.gram_inv;
        [0, 1, 2].map(|r| g[r][0] * t[0] + g[r][1] * t[1] + g[r][2] * t[2])
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm2(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

/// The nodes within Chebyshev distance r of a node of `mask`.
fn dilate(grid: &crate::Grid, mask: &[bool], r: usize) -> Vec<bool> {
    let r = r as isize;
    let pass = |m: &[bool], along_x: bool| -> Vec<bool> {
        (0..m.len())
            .map(|k| (-r..=r).any(|o| if along_x { grid.neighbour(k, o, 0) } else { grid.neighbour(k, 0, o) }.is_some_and(|q| m[q])))
            .collect()
    };
    pass(&pass(mask, true), false)
}

fn norm_inf(a: &[f64]) -> f64 {
    a.iter().fold(0.0f64, |m, v| m.max(v.abs()))
}

fn invert3(a: &[[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let c =
        |r: usize, s: usize| a[(r + 1) % 3][(s + 1) % 3] * a[(r + 2) % 3][(s + 2) % 3] - a[(r + 1) % 3][(s + 2) % 3] * a[(r + 2) % 3][(s + 1) % 3];
    let det = a[0][0] * c(0, 0) + a[0][1] * c(0, 1) + a[0][2] * c(0, 2);
    let scale = a[0][0].abs().max(a[1][1].abs()).max(a[2][2].abs());
    if !(det.abs() > 1e-14 * scale * scale * scale) || !det.is_finite() {
        return None;
    }
    let mut out = [[0.0; 3]; 3];
    for r in 0..3 {
        for s in 0..3 {
            out[r][s] = c(s, r) / det;
        }
    }
    Some(out)
}

/// The preconditioner kept between Newton steps: the hierarchy, the topology it was built for
/// and the GMRES iterations right after its build.
struct Kept {
    topo: Topology,
    mg: Multigrid,
    first_iterations: Option<usize>,
    stale: bool,
}

/// An evaluation at a trial point.
struct Eval {
    r: Vec<f64>,
    cut: CutState,
}

/// The solver's working state.
struct Run<'a> {
    set: &'a StaticSettings,
    report: StaticReport,
    kept: Option<Kept>,
}

impl Run<'_> {
    fn now(&self) -> f64 {
        self.set.clock.map_or(0.0, |c| c())
    }
}

impl Solid {
    /// Solves the static equilibrium at the current load factor by Newton–Krylov with a
    /// multigrid preconditioner (see [`crate::statics`]), starting from the current state and
    /// ramping the load from zero. The body is left at the solution, at rest, with its cut state
    /// (the stress accessors read it). Only the mapped boundary is supported.
    pub fn solve_static(&mut self, set: &StaticSettings) -> Result<StaticReport, Error> {
        set.validate()?;
        if self.settings.boundary != Boundary::Mapped {
            return Err(Error::Invalid("not supported yet: a static solve with a boundary other than the mapped one".into()));
        }
        let mut run = Run { set, report: StaticReport::default(), kept: None };
        let t0 = run.now();
        let target = self.load;
        for v in self.v.iter_mut() {
            *v = [0.0; 2];
        }
        self.started = false;
        // The force scale: the largest external force on a control volume at the full load.
        self.rebuild()?;
        let st = self.build_cut_with(self.cut.as_ref(), None, false, true, &mut |_| {})?;
        let loads = self.mapped_loads(&st);
        let scale = loads.iter().fold(0.0f64, |m, f| m.max(f[0].abs()).max(f[1].abs()));
        run.report.force_scale = scale;
        self.cut = Some(st);
        if !(scale > 0.0) {
            // No load: the current state is checked as it is.
            self.load = target;
            let ctx = self.context(None, &mut run)?;
            run.report.residual = norm_inf(&ctx.projected(&ctx.r));
            run.report.converged = run.report.residual == 0.0;
            self.finish(ctx);
            run.report.times.total = run.now() - t0;
            return Ok(run.report);
        }
        // Converged states (load, ζ at every node) for the secant predictor.
        let mut history: Vec<(f64, Vec<[f64; 2]>)> = vec![(0.0, self.zeta.clone())];
        let mut done = 0.0;
        let mut inc = target / set.increments as f64;
        let min_inc = set.min_increment * target.abs();
        let mut last_ctx: Option<Context> = None;
        while (target - done).abs() > 1e-14 * target.abs().max(1.0) {
            let next = if (target - done).abs() <= inc.abs() * (1.0 + 1e-12) { target } else { done + inc };
            let saved = (self.zeta.clone(), self.kind.clone(), self.phi.clone(), self.cut.clone());
            // Secant predictor through the last two converged states.
            if let [.., (l0, z0), (l1, z1)] = history.as_slice()
                && (l1 - l0).abs() > 0.0
            {
                let w = (next - l1) / (l1 - l0);
                for k in 0..self.grid.len() {
                    if self.kind[k] == NodeKind::Solid {
                        for c in 0..2 {
                            self.zeta[k][c] = z1[k][c] + w * (z1[k][c] - z0[k][c]);
                        }
                    }
                }
            }
            self.load = next;
            let outcome = self.newton(&mut run, scale, last_ctx.take());
            match outcome {
                Ok(ctx) => {
                    let iterations = run.report.newton.iter().rev().take_while(|s| s.load == next).count();
                    done = next;
                    run.report.increments += 1;
                    history.push((done, self.zeta.clone()));
                    if history.len() > 2 {
                        history.remove(0);
                    }
                    last_ctx = Some(ctx);
                    if iterations <= 4 {
                        inc *= 1.5;
                    }
                }
                Err(e) => {
                    run.report.failed_increments += 1;
                    run.report.failures.push(format!("increment to load factor {next}: {e}"));
                    (self.zeta, self.kind, self.phi, self.cut) = saved;
                    run.kept = None;
                    inc *= 0.5;
                    if inc.abs() < min_inc {
                        self.load = done;
                        return Err(Error::Diverged(format!("the static solve stalled at load factor {done} (last failure: {e})")));
                    }
                }
            }
        }
        let ctx = match last_ctx {
            Some(c) => c,
            None => self.context(None, &mut run)?,
        };
        let scale = scale.max(ctx.cut.stats().max_flux);
        run.report.force_scale = scale;
        run.report.residual = norm_inf(&ctx.projected(&ctx.r)) / scale;
        run.report.converged = run.report.residual <= set.tolerance;
        self.finish(ctx);
        run.report.times.total = run.now() - t0;
        Ok(run.report)
    }

    /// Research diagnostic: the first Newton system at the current state and load,
    /// QJP δ = −QR, solved by GMRES to the relative tolerance `rtol` (at most
    /// `set.max_krylov` iterations), with the multigrid preconditioner or without any. The body
    /// is left at the current state.
    pub fn static_linear_test(&mut self, set: &StaticSettings, rtol: f64, precondition: bool) -> Result<LinearTest, Error> {
        self.linear_test(set, rtol, precondition, false)
    }

    /// As [`Solid::static_linear_test`] with the preconditioner's own assembled matrix as the
    /// operator instead of the Jacobian (to tell the multigrid's quality from the matrix's).
    pub fn static_linear_test_assembled(&mut self, set: &StaticSettings, rtol: f64) -> Result<LinearTest, Error> {
        self.linear_test(set, rtol, true, true)
    }

    fn linear_test(&mut self, set: &StaticSettings, rtol: f64, precondition: bool, assembled: bool) -> Result<LinearTest, Error> {
        set.validate()?;
        if self.settings.boundary != Boundary::Mapped {
            return Err(Error::Invalid("not supported yet: a static solve with a boundary other than the mapped one".into()));
        }
        let mut run = Run { set, report: StaticReport::default(), kept: None };
        self.rebuild()?;
        let ctx = self.context(None, &mut run)?;
        let a0 = if assembled { Some(self.assemble_jacobian(&ctx, &mut run)?) } else { None };
        let mg = if precondition { Some(self.build_preconditioner(&ctx, &mut run)?) } else { None };
        let b: Vec<f64> = ctx.projected(&ctx.r).iter().map(|v| -v).collect();
        let h = self.grid.h();
        let reuse = set.reuse_surface_lu;
        let k = {
            let mut apply = |v: &[f64]| -> Result<Vec<f64>, Error> {
                let pv = ctx.free(v);
                if let Some(a) = &a0 {
                    let mut y = vec![0.0; v.len()];
                    a.mul(&pv, &mut y);
                    return Ok(ctx.projected(&y));
                }
                let vmax = norm_inf(&pv);
                if !(vmax > 0.0) {
                    return Ok(vec![0.0; v.len()]);
                }
                let eps = 1e-7 * h / vmax;
                let delta: Vec<f64> = pv.iter().map(|x| eps * x).collect();
                let e = self.eval(&ctx, &delta, reuse, &mut run)?;
                let jv: Vec<f64> = e.r.iter().zip(&ctx.r).map(|(a, b)| (a - b) / eps).collect();
                Ok(ctx.projected(&jv))
            };
            let mut pre = |v: &[f64]| -> Vec<f64> { mg.as_ref().map_or_else(|| v.to_vec(), |m| m.apply(v)) };
            gmres(&b, &mut apply, &mut pre, rtol, set.restart, set.max_krylov)?
        };
        self.place(&ctx, &vec![0.0; 2 * ctx.unknowns.len()], &mut run);
        Ok(LinearTest {
            unknowns: ctx.unknowns.len(),
            iterations: k.iterations,
            history: k.history,
            multigrid: mg.map(|m| m.stats()).unwrap_or_default(),
        })
    }

    /// Leaves the body at the context's state, at rest, with its cut state.
    fn finish(&mut self, ctx: Context) {
        self.kind = ctx.topo.kind.clone();
        self.zeta = ctx.base;
        self.refill_planned(&ctx.plan);
        self.cut = Some(ctx.cut);
        self.v_live = self.kind.iter().map(|k| k.known()).collect();
        for v in self.v.iter_mut() {
            *v = [0.0; 2];
        }
    }

    /// Newton at the current load from the current state; returns the converged context.
    fn newton(&mut self, run: &mut Run, external: f64, previous: Option<Context>) -> Result<Context, Error> {
        let tolerance = run.set.tolerance;
        let tol_of = |c: &Context| tolerance * external.max(c.cut.stats().max_flux);
        let set = *run.set;
        self.rebuild()?;
        let mut ctx = self.context(previous.as_ref(), run)?;
        let mut changes = 0;
        let mut prev_norm: Option<f64> = None;
        let mut forcing = set.max_forcing;
        for _ in 0..set.max_newton {
            let qr = ctx.projected(&ctx.r);
            let rn = norm_inf(&qr);
            let tol = tol_of(&ctx);
            if rn <= tol {
                return Ok(ctx);
            }
            // Eisenstat–Walker choice 2 (γ = 0.9, α = 2) with its safeguard.
            let r2 = norm2(&qr);
            if let Some(p) = prev_norm {
                let eta = 0.9 * (r2 / p).powi(2);
                let safe = 0.9 * forcing * forcing;
                forcing = if safe > 0.1 { eta.max(safe) } else { eta };
                forcing = forcing.clamp(1e-8, set.max_forcing);
            }
            prev_norm = Some(r2);
            let built = self.ensure_preconditioner(&ctx, run)?;
            let (delta, krylov, lin) = self.linear_solve(&ctx, &qr, forcing, run)?;
            // Line search on ‖QR‖₂ with the topology held; a trial whose body nodes left (or
            // band nodes entered) is re-evaluated with the rebuilt topology.
            let mut alpha = 1.0;
            let mut accepted: Option<(Context, bool)> = None;
            // Seven halvings at most: a Newton direction that needs a shorter step means the held
            // residual is not smooth there; the increment is retried smaller instead.
            for _ in 0..7 {
                let trial: Vec<f64> = delta.iter().map(|d| alpha * d).collect();
                let t = run.now();
                let flips = self.place(&ctx, &trial, run);
                // A trial that moves the surface across a node is judged on the updated
                // topology (the held one has no cut geometry for it).
                if flips == 0 {
                    let e = self.held(&ctx, set.reuse_surface_lu, run);
                    run.report.times.residual += run.now() - t;
                    if let Ok(e) = e
                        && norm2(&ctx.projected(&e.r)) <= (1.0 - 1e-4 * alpha) * r2
                    {
                        let next = Context { base: self.zeta.clone(), cut: e.cut, r: e.r, ..ctx.clone_structure() };
                        accepted = Some((next, false));
                        break;
                    }
                } else {
                    run.report.times.residual += run.now() - t;
                    // The body changed: rebuild the topology at the trial point.
                    let t = run.now();
                    let rebuilt = self.rebuild().and_then(|_| self.context(Some(&ctx), run));
                    run.report.times.topology += run.now() - t;
                    if let Ok(next) = rebuilt
                        && norm2(&next.projected(&next.r)) <= (1.0 - 1e-4 * alpha) * r2
                    {
                        accepted = Some((next, true));
                        break;
                    }
                    // Back to the held topology for the next trial.
                    self.kind = ctx.topo.kind.clone();
                }
                alpha *= 0.5;
            }
            let Some((mut next, mut changed)) = accepted else {
                return Err(Error::Diverged(format!("the line search failed at load factor {} (‖QR‖∞ = {rn:.3e})", self.load)));
            };
            // Converged on the held topology: check that the topology the solution implies is
            // the one held (validity of surface points and merges included).
            if !changed && norm_inf(&next.projected(&next.r)) <= tol_of(&next) {
                let t = run.now();
                let topo = self.current_topology(&next)?;
                run.report.times.topology += run.now() - t;
                if topo != next.topo {
                    next = self.context(Some(&next), run)?;
                    changed = true;
                }
            }
            if changed {
                changes += 1;
                run.report.topology_changes += 1;
                if changes > set.max_topology_changes {
                    return Err(Error::Diverged(format!("the cut topology kept changing at load factor {}", self.load)));
                }
            }
            run.report.newton.push(NewtonStep {
                load: self.load,
                residual: rn / external.max(ctx.cut.stats().max_flux),
                unknowns: ctx.unknowns.len(),
                krylov,
                linear_residual: lin,
                forcing,
                step: alpha,
                preconditioner_built: built,
                topology_changed: changed,
            });
            ctx = next;
        }
        let rn = norm_inf(&ctx.projected(&ctx.r));
        if rn <= tol_of(&ctx) {
            return Ok(ctx);
        }
        Err(Error::Diverged(format!(
            "Newton did not converge at load factor {} (‖QR‖∞ / scale = {:.3e})",
            self.load,
            rn / external.max(ctx.cut.stats().max_flux)
        )))
    }

    /// The topology an unheld build would choose at the context's state.
    fn current_topology(&mut self, ctx: &Context) -> Result<Topology, Error> {
        let kind = self.kind.clone();
        let zeta = self.zeta.clone();
        let phi = self.phi.clone();
        self.zeta = ctx.base.clone();
        self.rebuild()?;
        let st = self.build_cut_with(Some(&ctx.cut), None, false, false, &mut |_| {})?;
        let topo = st.topology(&self.kind);
        self.kind = kind;
        self.zeta = zeta;
        self.phi = phi;
        Ok(topo)
    }

    /// A context at the current state (kinds as they are, after a rebuild): its topology, its
    /// unknowns and the residual there. `warm` warm-starts the surface values.
    fn context(&mut self, warm: Option<&Context>, run: &mut Run) -> Result<Context, Error> {
        let t = run.now();
        let plan = Arc::new(self.fill_plan());
        self.refill_planned(&plan);
        let prev = warm.map(|c| &c.cut).or(self.cut.as_ref());
        let st0 = self.build_cut_with(prev, None, false, false, &mut |_| {})?;
        let topo = st0.topology(&self.kind);
        // With the topology held as decided, solve the surface (reusing the factorisations of
        // the warm state when its topology is the same).
        let same = warm.is_some_and(|w| w.topo == topo);
        let cut =
            self.build_cut_with(if same { warm.map(|w| &w.cut) } else { prev }, Some(&topo), same && run.set.reuse_surface_lu, true, &mut |_| {})?;
        run.report.surface_factorizations += cut.stats().factorizations;
        run.report.surface_traction_evals += cut.stats().traction_evals;
        let n = self.grid.len();
        let mut index = vec![usize::MAX; n];
        let mut unknowns = Vec::new();
        for k in 0..n {
            if self.kind[k] == NodeKind::Solid && cut.owner(k) == Some(k) {
                index[k] = unknowns.len();
                unknowns.push(k);
            }
        }
        if unknowns.is_empty() {
            return Err(Error::Diverged("no body node holds a control volume".into()));
        }
        let mut members: Vec<Vec<usize>> = unknowns.iter().map(|&k| vec![k]).collect();
        for k in 0..n {
            if self.kind[k] != NodeKind::Solid || index[k] != usize::MAX {
                continue;
            }
            // A merged node moves with the node it joined; a node without solid (a body node
            // the surface touches) with its deepest neighbour that is an unknown.
            let target = cut.owner(k).filter(|&o| index[o] != usize::MAX).or_else(|| {
                let mut best: Option<usize> = None;
                for (di, dj) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                    if let Some(m) = self.grid.neighbour(k, di, dj)
                        && index[m] != usize::MAX
                        && best.is_none_or(|b| self.phi[m] < self.phi[b])
                    {
                        best = Some(m);
                    }
                }
                best
            });
            match target {
                Some(o) => members[index[o]].push(k),
                None => return Err(Error::Diverged(format!("body node {k} has no control volume to move with"))),
            }
        }
        let mut cut = cut;
        let force = self.mapped_forces(&mut cut, 0.0)?;
        let r: Vec<f64> = unknowns.iter().flat_map(|&k| force[k]).collect();
        let rigid = self.rigid(&unknowns, &cut);
        run.report.evaluations += 1;
        run.report.times.residual += run.now() - t;
        Ok(Context { topo, unknowns, members, base: self.zeta.clone(), cut, r, rigid, plan })
    }

    /// The rigid motions of a free body (none when a node is held).
    fn rigid(&self, unknowns: &[usize], cut: &CutState) -> Option<Rigid> {
        if self.kind.contains(&NodeKind::Fixed) {
            return None;
        }
        let cv = cut.control_volumes();
        let mut mtot = 0.0;
        let mut c = [0.0; 2];
        for &k in unknowns {
            let x = self.grid.position(k);
            mtot += cv[k];
            c[0] += cv[k] * x[0];
            c[1] += cv[k] * x[1];
        }
        if !(mtot > 0.0) {
            return None;
        }
        let centre = [c[0] / mtot, c[1] / mtot];
        let h = self.grid.h();
        let m2 = 2 * unknowns.len();
        let mut phi = [vec![0.0; m2], vec![0.0; m2], vec![0.0; m2]];
        let mut mass = vec![0.0; m2];
        for (u, &k) in unknowns.iter().enumerate() {
            let x = self.grid.position(k);
            let d = [(x[0] - centre[0]) / h, (x[1] - centre[1]) / h];
            phi[0][2 * u] = 1.0;
            phi[1][2 * u + 1] = 1.0;
            phi[2][2 * u] = -d[1];
            phi[2][2 * u + 1] = d[0];
            mass[2 * u] = cv[k];
            mass[2 * u + 1] = cv[k];
        }
        let mut g = [[0.0; 3]; 3];
        for a in 0..3 {
            for b in 0..3 {
                g[a][b] = phi[a].iter().zip(&phi[b]).zip(&mass).map(|((p, q), m)| p * q * m).sum();
            }
        }
        let gram_inv = invert3(&g)?;
        Some(Rigid { phi, mass, gram_inv, centre })
    }

    /// The residual at the context's base plus `delta` (per unknown), with its topology held.
    /// Leaves the body at that state.
    fn eval(&mut self, ctx: &Context, delta: &[f64], reuse_lu: bool, run: &mut Run) -> Result<Eval, Error> {
        self.place(ctx, delta, run);
        self.held(ctx, reuse_lu, run)
    }

    /// Moves the body nodes to the context's base plus `delta` and refills the band with the
    /// topology held; returns the number of nodes whose kind no longer matches φ.
    fn place(&mut self, ctx: &Context, delta: &[f64], run: &mut Run) -> usize {
        let t = run.now();
        for (u, mem) in ctx.members.iter().enumerate() {
            let d = [delta[2 * u], delta[2 * u + 1]];
            for &m in mem {
                self.zeta[m] = [ctx.base[m][0] + d[0], ctx.base[m][1] + d[1]];
            }
        }
        let flips = self.refill_planned(&ctx.plan);
        run.report.times.fill += run.now() - t;
        flips
    }

    /// The residual at the placed state with the context's topology held.
    fn held(&mut self, ctx: &Context, reuse_lu: bool, run: &mut Run) -> Result<Eval, Error> {
        self.held_on(ctx, reuse_lu, None, run)
    }

    /// As [`Solid::held`] with the forces only on the faces touching `faces` (when given).
    fn held_on(&mut self, ctx: &Context, reuse_lu: bool, faces: Option<&[bool]>, run: &mut Run) -> Result<Eval, Error> {
        run.report.evaluations += 1;
        let t0 = run.now();
        let mut cut = self.build_cut_with(Some(&ctx.cut), Some(&ctx.topo), reuse_lu, true, &mut |_| {})?;
        let t1 = run.now();
        run.report.surface_factorizations += cut.stats().factorizations;
        run.report.surface_traction_evals += cut.stats().traction_evals;
        let force = self.mapped_forces_on(&mut cut, 0.0, faces, &mut |_| {})?;
        run.report.times.cut += t1 - t0;
        run.report.times.forces += run.now() - t1;
        let r: Vec<f64> = ctx.unknowns.iter().flat_map(|&k| force[k]).collect();
        if !r.iter().all(|v| v.is_finite()) {
            return Err(Error::Diverged("non-finite force".into()));
        }
        Ok(Eval { r, cut })
    }

    /// Makes sure a hierarchy for the context's topology is kept; returns whether one was built.
    fn ensure_preconditioner(&mut self, ctx: &Context, run: &mut Run) -> Result<bool, Error> {
        let usable = run.kept.as_ref().is_some_and(|k| !k.stale && k.topo == ctx.topo && run.set.reuse_preconditioner);
        if usable {
            return Ok(false);
        }
        let mg = self.build_preconditioner(ctx, run)?;
        run.report.multigrid = mg.stats();
        run.report.preconditioner_builds += 1;
        run.kept = Some(Kept { topo: ctx.topo.clone(), mg, first_iterations: None, stale: false });
        Ok(true)
    }

    /// The Jacobian at the context for the preconditioner, and its multigrid hierarchy.
    ///
    /// Rows more than two nodes from the surface (from any cut cell, surface point, shared or
    /// merged control volume) are the compact stencil's fluxes; their blocks come from the
    /// tangent of each face's flux with respect to its ∇ζ (by differences of the constitutive
    /// law), exact up to O(ε). The rows near the surface are probed: the unknowns within two
    /// nodes of them are perturbed one colour at a time, and the forces are evaluated only on
    /// the faces those rows need. With `local_probing` off, every row is probed (the reference
    /// for checking the local assembly).
    fn build_preconditioner(&mut self, ctx: &Context, run: &mut Run) -> Result<Multigrid, Error> {
        let a = self.assemble_jacobian(ctx, run)?;
        let t1 = run.now();
        let ij: Vec<(usize, usize)> = ctx.unknowns.iter().map(|&k| self.grid.ij(k)).collect();
        let h = self.grid.h();
        let mg =
            Multigrid::new(a, ij, (self.grid.nx(), self.grid.ny()), self.grid.position(0), h, ctx.rigid.as_ref().map(|r| r.centre), run.set.sweeps)?;
        run.report.times.setup += run.now() - t1;
        Ok(mg)
    }

    /// The assembled approximate Jacobian of [`Solid::build_preconditioner`].
    fn assemble_jacobian(&mut self, ctx: &Context, run: &mut Run) -> Result<BlockMatrix, Error> {
        if run.set.matrix == PreconditionerMatrix::Ghost {
            let t0 = run.now();
            let a = self.assemble_ghost(ctx)?;
            run.report.times.probing += run.now() - t0;
            return Ok(a);
        }
        let s = run.set.probe;
        let h = self.grid.h();
        let eps = 1e-7 * h;
        let nu = ctx.unknowns.len();
        let n = self.grid.len();
        let mut index = vec![usize::MAX; n];
        for (u, &k) in ctx.unknowns.iter().enumerate() {
            index[k] = u;
        }
        let local = run.set.local_probing;
        // Rows near the surface, the unknowns they depend on, and the nodes whose faces their
        // forces need.
        let (near_rows, columns, faces) = if local {
            let special = ctx.cut.special_nodes();
            let near_rows = dilate(&self.grid, &special, 2);
            (near_rows.clone(), dilate(&self.grid, &near_rows, 2), Some(dilate(&self.grid, &near_rows, 1)))
        } else {
            (vec![true; n], vec![true; n], None)
        };
        let mut rows: Vec<Vec<(usize, Block)>> = vec![Vec::new(); nu];
        let t0 = run.now();
        if local {
            self.assemble_interior(ctx, &index, &near_rows, &mut rows)?;
        }
        let half = (s / 2) as isize;
        for ci in 0..s {
            for cj in 0..s {
                let colour: Vec<usize> = (0..nu)
                    .filter(|&u| {
                        let k = ctx.unknowns[u];
                        let (i, j) = self.grid.ij(k);
                        columns[k] && i % s == ci && j % s == cj
                    })
                    .collect();
                if colour.is_empty() {
                    continue;
                }
                for c in 0..2 {
                    let mut delta = vec![0.0; 2 * nu];
                    for &u in &colour {
                        delta[2 * u + c] = eps;
                    }
                    self.place(ctx, &delta, run);
                    let e = self.held_on(ctx, run.set.reuse_surface_lu, faces.as_deref(), run)?;
                    run.report.probe_evaluations += 1;
                    for (w, row) in rows.iter_mut().enumerate() {
                        let k = ctx.unknowns[w];
                        if !near_rows[k] {
                            continue;
                        }
                        let d = [(e.r[2 * w] - ctx.r[2 * w]) / eps, (e.r[2 * w + 1] - ctx.r[2 * w + 1]) / eps];
                        if d[0] == 0.0 && d[1] == 0.0 {
                            continue;
                        }
                        // The perturbed node of this colour nearest the row's node.
                        let (i, j) = self.grid.ij(k);
                        let off = |a: usize, col: usize| -> isize {
                            let r = ((col as isize - a as isize) % s as isize + s as isize) % s as isize;
                            if r > half { r - s as isize } else { r }
                        };
                        let Some(q) = self.grid.neighbour(k, off(i, ci), off(j, cj)) else {
                            continue;
                        };
                        let col = index[q];
                        if col == usize::MAX || !columns[q] {
                            continue;
                        }
                        let mut b = [[0.0; 2]; 2];
                        b[0][c] = d[0];
                        b[1][c] = d[1];
                        row.push((col, b));
                    }
                }
            }
        }
        // Back to the base state.
        self.place(ctx, &vec![0.0; 2 * nu], run);
        let t1 = run.now();
        run.report.times.probing += t1 - t0;
        run.report.times.residual += t1 - t0;
        Ok(BlockMatrix::from_rows(rows))
    }

    /// The ghost-cell approximation of the Jacobian (see [`PreconditionerMatrix::Ghost`]).
    fn assemble_ghost(&self, ctx: &Context) -> Result<BlockMatrix, Error> {
        use std::collections::HashMap;
        let h = self.grid.h();
        let n = self.grid.len();
        let nu = ctx.unknowns.len();
        // The unknown each body node moves with.
        let mut unknown_of = vec![usize::MAX; n];
        for (u, mem) in ctx.members.iter().enumerate() {
            for &m in mem {
                unknown_of[m] = u;
            }
        }
        // Band nodes as combinations of unknowns, in fill order.
        let mut expand: HashMap<usize, Vec<(usize, f64)>> = HashMap::new();
        for (k, weights) in ctx.plan.band() {
            let mut acc: HashMap<usize, f64> = HashMap::new();
            for &(m, w) in weights {
                if unknown_of[m] != usize::MAX {
                    *acc.entry(unknown_of[m]).or_insert(0.0) += w;
                } else if let Some(e) = expand.get(&m) {
                    for &(u, wu) in e {
                        *acc.entry(u).or_insert(0.0) += w * wu;
                    }
                }
            }
            expand.insert(*k, acc.into_iter().collect());
        }
        let columns = |m: usize| -> Vec<(usize, f64)> {
            if unknown_of[m] != usize::MAX { vec![(unknown_of[m], 1.0)] } else { expand.get(&m).cloned().unwrap_or_default() }
        };
        let flux = |g: &crate::material::Tensor, d: usize| -> Result<[f64; 2], Error> {
            let f = crate::solid::invert(&crate::solid::identity_plus(g)).ok_or_else(|| Error::Diverged("the reference map folded".into()))?;
            let s = self.material.cauchy(&f)?;
            Ok(if d == 0 { [s[0] * h, s[2] * h] } else { [s[2] * h, s[1] * h] })
        };
        let mut index = vec![usize::MAX; n];
        for (u, &k) in ctx.unknowns.iter().enumerate() {
            index[k] = u;
        }
        // A dual cell's row goes to the control volumes holding its solid, with their shares.
        let targets = |j: usize| -> [(usize, f64); 2] {
            let mut out = [(usize::MAX, 0.0); 2];
            for (slot, (o, w)) in out.iter_mut().zip(ctx.cut.share(j).into_iter().flatten()) {
                if w != 0.0 && index[o] != usize::MAX {
                    *slot = (index[o], w);
                }
            }
            out
        };
        let mut rows: Vec<Vec<(usize, Block)>> = vec![Vec::new(); nu];
        let step = 1e-7;
        for (k, d, fraction) in ctx.cut.solid_faces(self) {
            if !(fraction > 0.0) {
                continue;
            }
            let Some(nodes) = self.compact_nodes(k, d) else {
                continue;
            };
            let [b, ap, am, bp, bm] = nodes;
            let z = &self.zeta;
            let mut g = [[0.0; 2]; 2];
            for r in 0..2 {
                g[r][d] = (z[b][r] - z[k][r]) / h;
                g[r][1 - d] = (z[ap][r] + z[bp][r] - z[am][r] - z[bm][r]) / (4.0 * h);
            }
            let f0 = flux(&g, d)?;
            let mut sens = [[[0.0; 2]; 2]; 2];
            for r in 0..2 {
                for col in 0..2 {
                    let mut gp = g;
                    gp[r][col] += step;
                    let f1 = flux(&gp, d)?;
                    for comp in 0..2 {
                        sens[comp][r][col] = fraction * (f1[comp] - f0[comp]) / step;
                    }
                }
            }
            let (wa, wc) = (1.0 / h, 1.0 / (4.0 * h));
            let (tk, tb) = (targets(k), targets(b));
            for (m, a_w, c_w) in [(k, -wa, 0.0), (b, wa, 0.0), (ap, 0.0, wc), (bp, 0.0, wc), (am, 0.0, -wc), (bm, 0.0, -wc)] {
                for (u, w) in columns(m) {
                    let mut blk = [[0.0; 2]; 2];
                    for comp in 0..2 {
                        for c in 0..2 {
                            blk[comp][c] = w * (sens[comp][c][d] * a_w + sens[comp][c][1 - d] * c_w);
                        }
                    }
                    // + on k's dual cell, − on b's.
                    for (t, sign) in [(tk, 1.0), (tb, -1.0)] {
                        for (r, ws) in t {
                            if r != usize::MAX {
                                rows[r].push((u, blk.map(|row| row.map(|v| sign * ws * v))));
                            }
                        }
                    }
                }
            }
        }
        // A row left without a diagonal (a control volume no face reaches) keeps the identity
        // scaled like its neighbours, so the smoother stays defined.
        let scale = rows.iter().flatten().fold(0.0f64, |m, e| m.max(e.1[0][0].abs()));
        for (u, row) in rows.iter_mut().enumerate() {
            if !row.iter().any(|e| e.0 == u) {
                row.push((u, [[scale, 0.0], [0.0, scale]]));
            }
        }
        Ok(BlockMatrix::from_rows(rows))
    }

    /// The rows of the unknowns away from the surface: each full face's flux f = σ(F)·n h with
    /// ∇ζ from its six stencil nodes; ∂f/∂(∇ζ) by forward differences of the law (step 10⁻⁷
    /// in ∇ζ), times the stencil weights, added to the face's two nodes (+ at k, − at b).
    fn assemble_interior(&self, ctx: &Context, index: &[usize], near_rows: &[bool], rows: &mut [Vec<(usize, Block)>]) -> Result<(), Error> {
        let h = self.grid.h();
        let flux = |g: &crate::material::Tensor, d: usize| -> Result<[f64; 2], Error> {
            let f = crate::solid::invert(&crate::solid::identity_plus(g)).ok_or_else(|| Error::Diverged("the reference map folded".into()))?;
            let s = self.material.cauchy(&f)?;
            Ok(if d == 0 { [s[0] * h, s[2] * h] } else { [s[2] * h, s[1] * h] })
        };
        let step = 1e-7;
        for (k, d, nodes) in ctx.cut.full_faces() {
            let [b, ap, am, bp, bm] = nodes;
            let rk = (index[k] != usize::MAX && !near_rows[k]).then_some(index[k]);
            let rb = (index[b] != usize::MAX && !near_rows[b]).then_some(index[b]);
            if rk.is_none() && rb.is_none() {
                continue;
            }
            let z = &self.zeta;
            let mut g = [[0.0; 2]; 2];
            for r in 0..2 {
                g[r][d] = (z[b][r] - z[k][r]) / h;
                g[r][1 - d] = (z[ap][r] + z[bp][r] - z[am][r] - z[bm][r]) / (4.0 * h);
            }
            let f0 = flux(&g, d)?;
            // sens[comp][r][col] = ∂f_comp / ∂g[r][col].
            let mut sens = [[[0.0; 2]; 2]; 2];
            for r in 0..2 {
                for col in 0..2 {
                    let mut gp = g;
                    gp[r][col] += step;
                    let f1 = flux(&gp, d)?;
                    for comp in 0..2 {
                        sens[comp][r][col] = (f1[comp] - f0[comp]) / step;
                    }
                }
            }
            // Stencil weights of each node in the along (d) and across (1 − d) derivatives.
            let w_along = 1.0 / h;
            let w_across = 1.0 / (4.0 * h);
            let entries =
                [(k, -w_along, 0.0), (b, w_along, 0.0), (ap, 0.0, w_across), (bp, 0.0, w_across), (am, 0.0, -w_across), (bm, 0.0, -w_across)];
            for (m, wa, wc) in entries {
                let col = index[m];
                if col == usize::MAX {
                    continue;
                }
                let mut blk = [[0.0; 2]; 2];
                for comp in 0..2 {
                    for c in 0..2 {
                        blk[comp][c] = sens[comp][c][d] * wa + sens[comp][c][1 - d] * wc;
                    }
                }
                if let Some(r) = rk {
                    rows[r].push((col, blk));
                }
                if let Some(r) = rb {
                    rows[r].push((col, blk.map(|row| row.map(|v| -v))));
                }
            }
        }
        Ok(())
    }

    /// Solves QJP δ = −QR by GMRES with the kept V-cycle; returns P δ, the iterations and the
    /// relative linear residual. A stale or failing hierarchy is rebuilt once.
    fn linear_solve(&mut self, ctx: &Context, qr: &[f64], forcing: f64, run: &mut Run) -> Result<(Vec<f64>, usize, f64), Error> {
        let b: Vec<f64> = qr.iter().map(|v| -v).collect();
        let h = self.grid.h();
        let set = *run.set;
        let mut total = 0;
        for attempt in 0..2 {
            let Some(kept) = run.kept.take() else {
                return Err(Error::Diverged("no preconditioner".into()));
            };
            let result = {
                let mut apply = |v: &[f64]| -> Result<Vec<f64>, Error> {
                    let pv = ctx.free(v);
                    let vmax = norm_inf(&pv);
                    if !(vmax > 0.0) {
                        return Ok(vec![0.0; v.len()]);
                    }
                    let eps = 1e-7 * h / vmax;
                    let delta: Vec<f64> = pv.iter().map(|x| eps * x).collect();
                    let t = run.now();
                    let e = self.eval(ctx, &delta, set.reuse_surface_lu, run)?;
                    run.report.times.residual += run.now() - t;
                    run.report.krylov_evaluations += 1;
                    let jv: Vec<f64> = e.r.iter().zip(&ctx.r).map(|(a, b)| (a - b) / eps).collect();
                    Ok(ctx.projected(&jv))
                };
                let mut cycles = 0.0;
                let clock = set.clock;
                let mut pre = |v: &[f64]| -> Vec<f64> {
                    let t = clock.map_or(0.0, |c| c());
                    let z = kept.mg.apply(v);
                    cycles += clock.map_or(0.0, |c| c()) - t;
                    z
                };
                let out = gmres(&b, &mut apply, &mut pre, forcing, set.restart, set.max_krylov);
                run.report.times.cycles += cycles;
                out
            };
            let mut kept = kept;
            match result {
                Ok(k) if k.relative_residual <= forcing => {
                    total += k.iterations;
                    let first = *kept.first_iterations.get_or_insert(k.iterations.max(5));
                    if k.iterations > 2 * first {
                        kept.stale = true;
                    }
                    run.kept = Some(kept);
                    return Ok((ctx.free(&k.x), total, k.relative_residual));
                }
                Ok(k) if attempt == 1 || kept.first_iterations.is_none() => {
                    // A fresh hierarchy that does not reach the forcing term: take the step
                    // anyway (the line search judges it).
                    total += k.iterations;
                    run.kept = Some(kept);
                    return Ok((ctx.free(&k.x), total, k.relative_residual));
                }
                Ok(k) => {
                    total += k.iterations;
                    run.kept = None;
                    self.ensure_preconditioner(ctx, run)?;
                }
                Err(e) => {
                    run.kept = Some(kept);
                    return Err(e);
                }
            }
        }
        Err(Error::Diverged("the linear solve failed".into()))
    }
}

impl Context {
    /// Q r (r unchanged for a held body).
    fn projected(&self, r: &[f64]) -> Vec<f64> {
        match &self.rigid {
            Some(g) => g.residual(r),
            None => r.to_vec(),
        }
    }

    /// P v.
    fn free(&self, v: &[f64]) -> Vec<f64> {
        match &self.rigid {
            Some(g) => g.update(v),
            None => v.to_vec(),
        }
    }

    /// A copy of the structure (topology, unknowns, members, rigid modes), with the state fields
    /// to be replaced.
    fn clone_structure(&self) -> Context {
        Context {
            topo: self.topo.clone(),
            unknowns: self.unknowns.clone(),
            members: self.members.clone(),
            base: Vec::new(),
            cut: self.cut.clone(),
            r: Vec::new(),
            rigid: self.rigid.as_ref().map(|g| Rigid { phi: g.phi.clone(), mass: g.mass.clone(), gram_inv: g.gram_inv, centre: g.centre }),
            plan: self.plan.clone(),
        }
    }
}
