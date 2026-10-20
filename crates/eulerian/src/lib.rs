//! FieldCraft's Eulerian solid prototype: the reference map technique (RMT) on a fixed grid.
//!
//! Research step 2 of issue #594 (notes: `docs/research/reference-map.md`). Every grid node
//! stores the reference position ξ(x, t) of the material at x, advected with the material,
//! ∂ξ/∂t + (v·∇)ξ = 0; the deformation gradient is F = (∇ξ)⁻¹, a hyperelastic law gives the
//! Cauchy stress, and the velocity follows from the Eulerian momentum balance
//! ρ(∂v/∂t + (v·∇)v) = ∇·σ + f. No mesh: the solid is the set of nodes where φ₀(ξ) < 0 for a
//! signed distance φ₀ of the body in its reference configuration.
//!
//! - [`grid`]: the regular 2D grid of nodes (periodic or bounded per axis).
//! - [`material`]: plane-strain compressible neo-Hookean and small-strain linear elastic laws.
//! - [`shape`]: reference shapes (their signed distance φ₀).
//! - [`extrapolate`]: weighted least-squares linear extrapolation into the band outside the solid.
//! - `sharp`: a sharp traction boundary by traction-constrained ghost values, [`Boundary::Ghost`].
//! - [`solid`]: the state and the explicit time step.
//! - [`cut`] and [`mapped`]: the default surface, [`Boundary::Mapped`] — sharp cut cells with
//!   the traction imposed on the curved surface and the deformation gradient near it from a
//!   surface-fitted map.
//! - [`statics`] and [`multigrid`]: static equilibrium of the mapped boundary by Newton–Krylov
//!   (finite-difference Jacobian products, GMRES, a geometric multigrid preconditioner) with
//!   load continuation, instead of dynamic relaxation.
//! - [`refine`]: local refinement near the surface — nested levels (a balanced quadtree stored
//!   level by level), conservative coarse–fine fluxes, subcycling in time, regridding.
//!
//! The scheme, in our own implementation from the published equations:
//! - fields at nodes (the nodes a lattice Boltzmann fluid will share later): the reference map
//!   stored as ζ = ξ − x (periodic when the motion is, and free of the cancellation of
//!   storing ξ itself), and the velocity v;
//! - F on the cell edges from compact differences (normal derivative from the two nodes of the
//!   edge, tangential derivative the average of the four transverse differences), the stress
//!   divergence at a node from its four edges (Kamrin, Rycroft & Nave 2012 §6.1; Rycroft et al.
//!   2020 §3.3);
//! - the surface traction is σ_ambient·n, σ_ambient a prescribed stress outside the body (zero
//!   for a free surface, −p I for a pressure, any field), carried by the mapped cut cells of
//!   [`Boundary::Mapped`] (second order; see [`mapped`]), or, for the comparison studies, by
//!   the blurred surface of the papers (the edge stress is (1 − H_ε(φ)) σ_solid + H_ε(φ)
//!   σ_ambient with the smoothed Heaviside of Rycroft et al. 2020; Valkov, Rycroft & Kamrin
//!   2015) or by the ghost fill of [`Boundary::Ghost`] (the solid stress alone, with ghost
//!   values outside the body fitted so that σ·n = σ_ambient·n holds at the surface; see
//!   `sharp`);
//! - ξ in a band of a few nodes outside the solid rebuilt every step by a weighted linear
//!   least-squares fit, layer by layer (after Rycroft et al. 2020 §3.2.2 and Sun & Rycroft 2025,
//!   §3.2 of the preprint), and φ = φ₀(ξ) from it (Valkov et al. 2015);
//! - second-order one-sided upwind advection (Sun & Rycroft 2025, eq. 21 of the preprint), falling back to first
//!   order next to unknown nodes;
//! - central differences in time (v at half steps: v first from the current stress, then ξ
//!   with the new velocity; Belytschko et al., *Nonlinear Finite Elements for Continua and
//!   Structures*, Box 6.1); Δt = α h / c_P with the P-wave speed; an artificial viscosity
//!   μ_e = κ ρ c_P h (Rycroft et al. 2020 eq. 3.28) and an optional mass-proportional damping
//!   −γ v to relax to static equilibrium (dynamic relaxation, as in Kamrin et al. 2012 §6.2).
//!
//! # Sources
//! - K. Kamrin, C. H. Rycroft, J.-C. Nave, *Reference map technique for finite-strain
//!   elasticity and fluid–solid interaction*, J. Mech. Phys. Solids 60(11):1952–1969, 2012,
//!   doi:10.1016/j.jmps.2012.06.003.
//! - B. Valkov, C. H. Rycroft, K. Kamrin, *Eulerian method for multiphase interactions of soft
//!   solid bodies in fluids*, J. Appl. Mech. 82(4):041011, 2015, doi:10.1115/1.4029765.
//! - C. H. Rycroft, C.-H. Wu, Y. Yu, K. Kamrin, *Reference map technique for incompressible
//!   fluid–structure interaction*, J. Fluid Mech. 898:A9, 2020, doi:10.1017/jfm.2020.353.
//! - Y. Sun, C. H. Rycroft, *A fully-integrated lattice Boltzmann method for fluid–structure
//!   interaction*, J. Comput. Phys. 526:113774, 2025, doi:10.1016/j.jcp.2025.113774 (read
//!   from the preprint, arXiv:2402.12696).
//! - J. Bonet, R. D. Wood, *Nonlinear Continuum Mechanics for Finite Element Analysis*, 2nd
//!   ed., Cambridge University Press 2008 (the compressible neo-Hookean law).
//!
//! Deterministic on any number of threads: the step's loops over nodes, cells and surface
//! points run on `fieldcraft-parallel`'s pool, every value is computed by the operations of the
//! sequential loop and every sum is formed in its order, so the same input gives the same
//! numbers bit for bit on one thread or many. The surface chains' Newton iterations stay in
//! chain order (each chain sees the values the earlier ones just took).
//! [`Solid::run_observed`] reports the step's [`Phase`]s for profiling. Sizes and step counts
//! are capped by [`Limits`].
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]
#![forbid(unsafe_code)]
// Small fixed-size tensor kernels read as the textbook formulas with index loops.
#![allow(clippy::needless_range_loop)]

pub mod cut;
pub mod extrapolate;
pub mod grid;
pub mod mapped;
pub mod material;
pub mod multigrid;
mod par;
pub mod refine;
pub mod shape;
mod sharp;
pub mod solid;
pub mod statics;

pub use grid::Grid;
pub use mapped::{CutStats, CutView, SurfaceSample};
pub use material::Material;
pub use multigrid::MultigridStats;
pub use refine::{Hierarchy, NodeRef, Refinement, ResidualEntry, Role};
pub use shape::Shape;
pub use solid::{Boundary, NodeKind, Phase, Relaxation, Settings, Solid};
pub use statics::{LinearTest, NewtonStep, PreconditionerMatrix, StaticReport, StaticSettings, StaticTimes};

/// Errors of the Eulerian solver.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum Error {
    /// A parameter is out of range or not a finite number.
    #[error("invalid parameter: {0}")]
    Invalid(String),
    /// A size or count is above its cap.
    #[error("too large: {0}")]
    TooLarge(String),
    /// The solution blew up (a non-finite value or an inverted deformation).
    #[error("the solution became unusable: {0}")]
    Diverged(String),
}

/// Caps on the work one solid may ask for.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Limits {
    /// Largest number of grid nodes.
    pub max_nodes: usize,
    /// Largest number of time steps in one call.
    pub max_steps: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Limits { max_nodes: 4_000_000, max_steps: 10_000_000 }
    }
}

/// A finite number, or an `Invalid` error naming it.
pub(crate) fn finite(name: &str, x: f64) -> Result<f64, Error> {
    if x.is_finite() { Ok(x) } else { Err(Error::Invalid(format!("{name} must be a finite number, got {x}"))) }
}

/// A finite positive number, or an `Invalid` error naming it.
pub(crate) fn positive(name: &str, x: f64) -> Result<f64, Error> {
    if x.is_finite() && x > 0.0 { Ok(x) } else { Err(Error::Invalid(format!("{name} must be a positive number, got {x}"))) }
}
