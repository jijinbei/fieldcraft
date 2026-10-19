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
//! - **Preconditioner matrix by probing**: the residual's dependence is local, so its Jacobian
//!   is assembled from s² × 2 evaluations (s = 5 by default): all unknowns of one colour (node
//!   (i, j) with i mod s, j mod s fixed) are perturbed together in one component, and the
//!   response at each row is attributed to the perturbed node nearest it. In the interior (a
//!   3 × 3 stencil) this is the exact Jacobian; near the surface, where the surface values and
//!   the band couple further, it lumps the far part of a column onto the nearest node (row sums
//!   kept).
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
//!   slowly).

use crate::Error;
use crate::mapped::{CutState, Topology};
use crate::multigrid::{Block, BlockMatrix, Multigrid, MultigridStats, gmres};
use crate::solid::{Boundary, FillPlan, NodeKind, Solid};
use std::sync::Arc;

/// Settings of the static solve.
#[derive(Debug, Clone, Copy)]
pub struct StaticSettings {
    /// Converged when ‖QR‖∞ ≤ tolerance × the largest external force on a control volume at the
    /// full load.
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
        if self.probe < 3 || self.probe > 9 || self.probe % 2 == 0 {
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
    /// Of which the probing evaluations.
    pub probing: f64,
    /// Multigrid setup (Galerkin products, coarsest factorisation).
    pub setup: f64,
    /// V-cycle applications.
    pub cycles: f64,
    /// Topology rebuilds.
    pub topology: f64,
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
    /// The force scale (largest external force on a control volume at the full load).
    pub force_scale: f64,
    /// Load increments accepted and failed.
    pub increments: usize,
    pub failed_increments: usize,
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
    flips: usize,
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
        let st = self.build_cut_with(self.cut.as_ref(), None, false, true)?;
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
        let tol = set.tolerance * scale;
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
            let outcome = self.newton(&mut run, tol, last_ctx.take());
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
        run.report.residual = norm_inf(&ctx.projected(&ctx.r)) / scale;
        run.report.converged = run.report.residual <= set.tolerance;
        self.finish(ctx);
        run.report.times.total = run.now() - t0;
        Ok(run.report)
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
    fn newton(&mut self, run: &mut Run, tol: f64, previous: Option<Context>) -> Result<Context, Error> {
        let set = *run.set;
        self.rebuild()?;
        let mut ctx = self.context(previous.as_ref(), run)?;
        let mut changes = 0;
        let mut prev_norm: Option<f64> = None;
        let mut forcing = set.max_forcing;
        for _ in 0..set.max_newton {
            let qr = ctx.projected(&ctx.r);
            let rn = norm_inf(&qr);
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
                let flips = self.place(&ctx, &trial);
                // A trial that moves the surface across a node is judged on the updated
                // topology (the held one has no cut geometry for it).
                let e =
                    if flips == 0 { self.held(&ctx, 0, set.reuse_surface_lu, run) } else { Ok(Eval { r: Vec::new(), cut: ctx.cut.clone(), flips }) };
                run.report.times.residual += run.now() - t;
                if let Err(err) = &e
                    && std::env::var("STATIC_DEBUG").is_ok()
                {
                    eprintln!("  trial α {alpha}: {err}");
                }
                if let Ok(e) = e {
                    if e.flips == 0 {
                        let n = norm2(&ctx.projected(&e.r));
                        if n <= (1.0 - 1e-4 * alpha) * r2 {
                            let base = self.zeta.clone();
                            let next = Context { base, cut: e.cut, r: e.r, ..ctx.clone_structure() };
                            accepted = Some((next, false));
                            break;
                        }
                    } else {
                        // The body changed: rebuild the topology at the trial point.
                        let t = run.now();
                        let rebuilt = self.rebuild().and_then(|_| self.context(Some(&ctx), run));
                        run.report.times.topology += run.now() - t;
                        if let Err(err) = &rebuilt
                            && std::env::var("STATIC_DEBUG").is_ok()
                        {
                            eprintln!("  trial α {alpha}: rebuilt: {err}");
                        }
                        if let Ok(next) = rebuilt {
                            let n = norm2(&next.projected(&next.r));
                            if std::env::var("STATIC_DEBUG").is_ok() {
                                let fr = 0.0;
                                let q = next.projected(&next.r);
                                let (mut w, mut wv) = (0, 0.0);
                                for u in 0..next.unknowns.len() {
                                    let v = q[2 * u].hypot(q[2 * u + 1]);
                                    if v > wv {
                                        wv = v;
                                        w = next.unknowns[u];
                                    }
                                }
                                let cv = next.cut.control_volumes();
                                eprintln!(
                                    "  trial α {alpha}: flips {} frozen {:.3e} rebuilt {:.3e} (old {:.3e}); worst node {w} at {:?} φ {:.3e} cv/h² {:.3} |r| {:.3e} was unknown {}",
                                    e.flips,
                                    fr,
                                    n,
                                    r2,
                                    self.grid.position(w),
                                    self.phi[w],
                                    cv[w] / (self.grid.h() * self.grid.h()),
                                    wv,
                                    ctx.unknowns.contains(&w)
                                );
                            }
                            if n <= (1.0 - 1e-4 * alpha) * r2 {
                                accepted = Some((next, true));
                                break;
                            }
                        }
                        // Back to the held topology for the next trial.
                        self.kind = ctx.topo.kind.clone();
                    }
                }
                alpha *= 0.5;
            }
            let Some((mut next, mut changed)) = accepted else {
                return Err(Error::Diverged(format!("the line search failed at load factor {} (‖QR‖∞ = {rn:.3e})", self.load)));
            };
            // Converged on the held topology: check that the topology the solution implies is
            // the one held (validity of surface points and merges included).
            if !changed && norm_inf(&next.projected(&next.r)) <= tol {
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
                residual: rn / run.report.force_scale,
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
        if rn <= tol {
            return Ok(ctx);
        }
        Err(Error::Diverged(format!("Newton did not converge at load factor {} (‖QR‖∞ / scale = {:.3e})", self.load, rn / run.report.force_scale)))
    }

    /// The topology an unheld build would choose at the context's state.
    fn current_topology(&mut self, ctx: &Context) -> Result<Topology, Error> {
        let kind = self.kind.clone();
        let zeta = self.zeta.clone();
        let phi = self.phi.clone();
        self.zeta = ctx.base.clone();
        self.rebuild()?;
        let st = self.build_cut_with(Some(&ctx.cut), None, false, false)?;
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
        let st0 = self.build_cut_with(prev, None, false, false)?;
        let topo = st0.topology(&self.kind);
        // With the topology held as decided, solve the surface (reusing the factorisations of
        // the warm state when its topology is the same).
        let same = warm.is_some_and(|w| w.topo == topo);
        let cut = self.build_cut_with(if same { warm.map(|w| &w.cut) } else { prev }, Some(&topo), same && run.set.reuse_surface_lu, true)?;
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
        let force = self.mapped_forces(&mut cut.clone(), 0.0)?;
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
        let flips = self.place(ctx, delta);
        self.held(ctx, flips, reuse_lu, run)
    }

    /// Moves the body nodes to the context's base plus `delta` and refills the band with the
    /// topology held; returns the number of nodes whose kind no longer matches φ.
    fn place(&mut self, ctx: &Context, delta: &[f64]) -> usize {
        for (u, mem) in ctx.members.iter().enumerate() {
            let d = [delta[2 * u], delta[2 * u + 1]];
            for &m in mem {
                self.zeta[m] = [ctx.base[m][0] + d[0], ctx.base[m][1] + d[1]];
            }
        }
        self.refill_planned(&ctx.plan)
    }

    /// The residual at the placed state with the context's topology held.
    fn held(&mut self, ctx: &Context, flips: usize, reuse_lu: bool, run: &mut Run) -> Result<Eval, Error> {
        run.report.evaluations += 1;
        let mut cut = self.build_cut_with(Some(&ctx.cut), Some(&ctx.topo), reuse_lu, true)?;
        run.report.surface_factorizations += cut.stats().factorizations;
        run.report.surface_traction_evals += cut.stats().traction_evals;
        let force = self.mapped_forces(&mut cut, 0.0)?;
        let r: Vec<f64> = ctx.unknowns.iter().flat_map(|&k| force[k]).collect();
        if !r.iter().all(|v| v.is_finite()) {
            return Err(Error::Diverged("non-finite force".into()));
        }
        Ok(Eval { r, cut, flips })
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

    /// The probed Jacobian at the context and its multigrid hierarchy.
    fn build_preconditioner(&mut self, ctx: &Context, run: &mut Run) -> Result<Multigrid, Error> {
        let s = run.set.probe;
        let h = self.grid.h();
        let eps = 1e-7 * h;
        let nu = ctx.unknowns.len();
        let mut index = vec![usize::MAX; self.grid.len()];
        for (u, &k) in ctx.unknowns.iter().enumerate() {
            index[k] = u;
        }
        let mut rows: Vec<Vec<(usize, Block)>> = vec![Vec::new(); nu];
        let half = (s / 2) as isize;
        let t0 = run.now();
        for ci in 0..s {
            for cj in 0..s {
                let colour: Vec<usize> = (0..nu)
                    .filter(|&u| {
                        let (i, j) = self.grid.ij(ctx.unknowns[u]);
                        i % s == ci && j % s == cj
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
                    let e = self.eval(ctx, &delta, run.set.reuse_surface_lu, run)?;
                    run.report.probe_evaluations += 1;
                    for (w, row) in rows.iter_mut().enumerate() {
                        let d = [(e.r[2 * w] - ctx.r[2 * w]) / eps, (e.r[2 * w + 1] - ctx.r[2 * w + 1]) / eps];
                        if d[0] == 0.0 && d[1] == 0.0 {
                            continue;
                        }
                        // The perturbed node of this colour nearest the row's node.
                        let (i, j) = self.grid.ij(ctx.unknowns[w]);
                        let off = |a: usize, col: usize| -> isize {
                            let r = ((col as isize - a as isize) % s as isize + s as isize) % s as isize;
                            if r > half { r - s as isize } else { r }
                        };
                        let Some(q) = self.grid.neighbour(ctx.unknowns[w], off(i, ci), off(j, cj)) else {
                            continue;
                        };
                        let col = index[q];
                        if col == usize::MAX {
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
        self.eval(ctx, &vec![0.0; 2 * nu], run.set.reuse_surface_lu, run)?;
        let t1 = run.now();
        run.report.times.probing += t1 - t0;
        run.report.times.residual += t1 - t0;
        let a = BlockMatrix::from_rows(rows);
        let ij: Vec<(usize, usize)> = ctx.unknowns.iter().map(|&k| self.grid.ij(k)).collect();
        let mg =
            Multigrid::new(a, ij, (self.grid.nx(), self.grid.ny()), self.grid.position(0), h, ctx.rigid.as_ref().map(|r| r.centre), run.set.sweeps)?;
        run.report.times.setup += run.now() - t1;
        Ok(mg)
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
