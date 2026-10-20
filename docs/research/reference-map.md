# Eulerian solids on the lattice: the reference map technique

Research notes for issue #594: solids described on a fixed grid (the lattice the LB fluid
already uses) instead of a separate FEM mesh, and the lattice Boltzmann coupling of such solids
(LBRMT). This page is our own summary, written from the papers listed at the end; the papers
themselves are kept locally and never committed. The prototype that follows these notes is
`crates/eulerian` (solid only, 2D plane strain); fluid coupling is a later step.

## 1. The idea

A Lagrangian solver follows material points. The reference map technique (RMT) does the
opposite: every grid point x stores the reference (undeformed) position ξ(x, t) of the material
that sits at x now. ξ is the inverse of the motion χ(X, t), and it is carried along with the
material, so it obeys the pure advection equation

    ∂ξ/∂t + (v · ∇) ξ = 0.

Its spatial gradient is the inverse deformation gradient, so

    F = (∇ξ)⁻¹,   J = det F,   ρ = ρ₀ / J,

and any finite-strain constitutive law σ = σ(F) (hyperelastic, or with internal variables
stored and advected beside ξ) gives the Cauchy stress at every grid point. The velocity follows
from the Eulerian momentum balance

    ρ (∂v/∂t + (v · ∇) v) = ∇ · σ + ρ b.

Everything lives on one fixed grid: no mesh, no remeshing, no Lagrangian markers. Displacement
boundary conditions are natural (the boundary's ξ is the boundary's reference position), and
a pre-strained state is just a non-identity initial ξ. Unlike Eulerian methods that advect F
itself, F stays compatible (it is a gradient), so no curl constraint has to be enforced.

The price: ξ is only defined where material is, the boundary of the solid is not on the grid,
and the stress needs a derivative of ξ, so every stencil that touches the boundary needs values
of ξ outside the solid.

## 2. Tracking the solid region

- Kamrin, Rycroft & Nave (2012) use a level set φ whose zero contour is the deformed boundary,
  with a sharp-interface treatment: values just outside the body are extrapolated linearly
  along grid lines from inside values and the boundary condition, and points that cross into
  the body during a step are given values by interpolation along the normal. Cut cells smaller
  than about a tenth of a cell are handled the same way.
- Valkov, Rycroft & Kamrin (2015) observe that the boundary is already encoded in ξ: if
  φ₀(X) is a signed distance to the body in the reference configuration, φ(x) = φ₀(ξ(x)) marks
  the deformed body without advecting a separate level set. The incompressible RMT (Rycroft,
  Wu, Yu & Kamrin 2020) and the LBRMT use this and reinitialise φ₀(ξ) to a signed distance in a
  narrow band each step (straddling points by a bicubic interpolant and Newton iterations, the
  rest by fast marching).
- From Valkov et al. (2015) on, the interface is blurred: a smoothed Heaviside H_ε(φ) over a
  transition zone of half-width ε (one to two cells) mixes solid and fluid stress and density,
  σ = H σ_f + (1 − H) σ_s, ρ = H ρ_f + (1 − H) ρ_s. Interface jump conditions are then satisfied
  implicitly in the zone rather than by explicit ghost-value fixes; only ξ is extrapolated
  (two scalar fields per step instead of the twenty of the sharp-interface method, by the
  authors' count).

## 3. Extrapolating ξ: the hard part

The solid stress is needed wherever H < 1, which includes the outer half of the blur zone
(0 < φ < ε), and its finite differences reach one more cell; ξ must therefore exist in a band
of a few cells outside the solid. ξ there is not advected (that region behaves like fluid and
would shear without bound, making the solid stress meaningless); it is rebuilt from the solid
each step.

- Sharp-interface RMT (2012): linear extrapolation along grid lines and PDE-based extension
  (Aslam 2004) of every needed field — many fields, and the papers report that accumulated
  extrapolation error could destabilise the interface (pressure spikes when points enter the
  solid).
- Valkov et al. (2015) introduce a constrained extrapolation that enforces physical and
  smoothness requirements on the extrapolated map.
- Rycroft et al. (2020) and the LBRMT replace this by a local least-squares fit: for each
  outside point, in order of increasing distance (2020: increasing φ; LBRMT: layer by layer
  outwards), fit a linear map ξ ≈ A x + c to all known values within a square window
  (half-width 3 in 2020, 2 in the LBRMT), weighted by distance, widening the window when the
  fit is degenerate (collinear points). The averaging blurs ξ far from the interface, which the
  authors argue is harmless because only the first cells matter, and it damps high-frequency
  noise.

Why it is hard: the extrapolated ξ decides (a) the stress in the blur zone, hence the traction
the surface carries, (b) where the interface is next step (φ₀(ξ) at the band points), so
material entering a cell gets an extrapolated, not an advected, reference position, and (c)
the deformation gradient computed across the interface. A linear fit is exact for homogeneous
deformation and first-order wrong where the strain varies, and it is the main source of the
first-order error near the boundary that the blurred methods show. Points where the fit is
degenerate (thin features, corners, two bodies touching) need special handling.

## 4. Numerical schemes in the papers

| | Kamrin, Rycroft & Nave 2012 | Valkov et al. 2015 | Rycroft et al. 2020 (IncRMT) | Sun & Rycroft 2025 (LBRMT) |
|---|---|---|---|---|
| fluid | compressible, explicit | compressible Navier–Stokes, explicit | incompressible, projection (Chorin), MAC + FE projections, multigrid | LB, D2Q9 BGK, Guo forcing |
| solid law | general hyperelastic, plane strain | compressible neo-Hookean | incompressible neo-Hookean (deviatoric) | incompressible neo-Hookean (deviatoric) |
| grid | v, ξ at cell corners; F, σ at cell centres | v, ξ corners, σ centres | cell-centred fields, stress on cell edges | LB nodes; F, σ on half-edges |
| advection | WENO (Liu, Osher & Chan 1994) | second-order upwinded ENO | second-order explicit Godunov scheme with Colella's limited slopes | second-order one-sided upwind differences |
| time | Euler step inside TVD RK3 (Shu & Osher 1988) | explicit | second order (projection) | LB step |
| interface | sharp, level set, extrapolation | blurred, φ = φ₀(ξ), one extrapolation | blurred, least-squares extrapolation, reinitialised φ | as IncRMT, blur zone + extrapolation zone of width ≥ 1.5Δx + 2ε |
| stabilisation | solid viscosity | artificial viscosity | artificial viscosity μ_e = κ ρ_s c_s h in the solid (κ = 0.4) | the LB fluid's own viscosity acts inside the solid |

Common points:
- **Stress stencil.** The deformation gradient is evaluated on cell edges (or centres) from
  compact differences of ξ: on a vertical edge ∂ξ/∂x from the two nodes it joins, ∂ξ/∂y as the
  average of the four transverse differences; the stress divergence at a node is then the
  difference of the edge stresses. This is compact (a 3 × 3 node stencil) and avoids the
  odd–even decoupling of a wide centred stencil.
- **CFL.** Explicit, limited by the solid wave speed: Δt ≲ α h / c with c = √(G/ρ) (shear) in
  the incompressible papers, the P-wave speed for compressible solids, plus the viscous limits
  Δt ≲ ρ h² / (2 μ (1 + q) d) of the fluid and the artificial viscosity.
- **Statics** are reached by relaxation: add damping (viscosity) and run the dynamics to rest
  (2012, §6.2: the washer, the flower). A good initial guess for ξ (a pre-strained map that
  already satisfies the boundary displacements) shortens the relaxation.
- **Verification** in the 2012 paper: a finite-strain travelling wave in a periodic domain
  (second-order convergence of ξ, v and stress), the Levinson–Burgess washer under finite
  rotation (closed form), uniform motion of a pre-deformed disk (exact to round-off because
  every field is linear). The incompressible paper's FSI tests converge at about order 1.8 in
  the L1 and L2 norms, with the largest errors at the interface (the blur zone alters the
  equations over a width O(h)) and first order in L∞ at the corners of a square solid.

## 5. How the LBRMT couples to the lattice

Sun & Rycroft (J. Comput. Phys. 2025; preprint 2024; Sun's 2020 master's thesis is an earlier
version of the method) put both phases on the D2Q9 lattice with one velocity field:

1. Advect ξ inside the solid (φ < 0) with the LB velocity; extrapolate ξ layer by layer into
   the extrapolation zone; relabel nodes with φ = φ₀(ξ).
2. Compute the solid's deviatoric stress on half-edges from ∇ξ, then the force density
   F = ∇ · [(1 − H_ε(φ)) τ_s] + (1 − H_ε) f_ext,s + H_ε f_ext,f.
3. Hand F to the LB collision as an external force with Guo's forcing (and the half-force
   velocity correction). The fluid stress is never computed explicitly: it is in the LB
   populations, and it also acts inside the solid, where it serves as the stabilising
   viscosity the IncRMT added by hand.
4. A "smooth flux correction" keeps the solid–fluid density difference: the equilibrium is
   shifted by the target density field ρ_t = H ρ_f + (1 − H) ρ_s, moving the excess flux back
   onto the rest population so mass and momentum stay conserved.

No interface mesh, no force spreading between Lagrangian and Eulerian points, no coupling
iteration; multi-body contact is a repulsive stress between overlapping blur zones (a separate
ξ field per body). Validation: a deformable disk in a lid-driven cavity against published
results, rotating and settling solids, suspensions mixing.

For FieldCraft this maps onto `crates/lbm` (D3Q19 TRT with Guo forcing already): the solid
supplies a force density per lattice cell, the lattice supplies the velocity; the prototype
keeps velocity and ξ at grid nodes for that reason.

## 6. Open problems (ours and the papers')

- **Accuracy at the boundary.** The blurred interface is formally first order at the
  interface (O(h) zone); the free surface sits
  somewhere in a zone ±ε wide, and ξ in that zone is extrapolated. Stresses at the surface (the
  numbers an analyst wants, e.g. the hoop stress at a bore) are the least accurate.
  §8 measures the bias; §10 removes it.
- **Extrapolation robustness**: thin parts a cell or two thick, sharp corners, near-contact,
  points that enter the solid with an extrapolated reference position.
- **Mixing / large distortion.** ξ keeps the link to the reference configuration, so extreme
  shearing (stirring) degrades ∇ξ; the 2012 paper suggests periodic reinitialisation of ξ with
  a stored F.
- **Level-set reinitialisation** cost and robustness (the IncRMT falls back to a first-order
  method when Newton fails).
- **Stability** relies on artificial viscosity or the LB viscosity; the solid's damping is
  then tied to grid or fluid parameters.
- **Stiff solids.** The explicit step is set by the solid's wave speed; for engineering metals
  in a slow flow, c_solid ≫ flow speed and the solid dominates the step count. The LBRMT's
  examples use soft solids (G comparable with fluid stresses).
- **Incompressible vs compressible.** The LB fluid is weakly compressible; the LBRMT's solid is
  incompressible neo-Hookean with the pressure from the lattice. A compressible solid on the
  lattice needs its volumetric stress in F.
- **3D** is direct in principle (3 × 3 ∇ξ, D3Q19) but the band and extrapolation cost grow
  with the surface.
- **Cost vs FEM** for a solid alone: an RMT grid resolves the whole bounding box and needs a
  dynamic relaxation to reach a static answer; it is not a substitute for the FEM solver for
  statics, only a route to FSI with large solid motion.

## 7. The prototype and its verification

`crates/eulerian` (`fieldcraft-eulerian`, layer 2, no command yet): 2D plane strain on a
regular grid, f64, sequential and deterministic, no panics, capped sizes and step counts.
This section describes the first version, which follows the published scheme with its blurred
surface (`Boundary::Blurred`).
The default surface is now the mapped one of §10; the blurred one stays for the comparison
studies.

- Nodes carry ζ = ξ − x (periodic when the motion is) and v — the nodes a lattice will share.
- F on cell edges from compact differences; ∇·σ at nodes from the four edges.
- Blurred surface (ε = 1.5 h, the smoothed Heaviside of the 2020 paper); outside the body a
  prescribed *ambient stress* (0 for a free surface, −p I in a pressurised bore, any field), so
  surface tractions are σ_ambient·n with no surface tracking.
- φ = φ₀(ξ) (Valkov et al.), band of four 8-neighbour rings rebuilt each step by a weighted
  linear least-squares fit using only earlier rings (independent of node order, mirror
  symmetric). Band nodes with φ < 0 become solid (material entering a cell).
- Second-order one-sided upwind advection (first order next to unknown nodes); central
  differences in time (v at half steps), Δt = 0.4 h / c_P; artificial viscosity
  μ_e = κ ρ c_P h (κ = 0.1 by default); implicit mass-proportional damping for statics.
- Compressible neo-Hookean (Bonet & Wood) and small-strain linear elasticity.
- Later additions: the decoupled neo-Hookean of FieldCraft's FEM (μ/2 (Ī₁ − 3) +
  κ/2 (J − 1)²) for finite-strain comparisons, and a body force per unit volume.

Verification (`crates/eulerian/tests/verification.rs`):

| check | reference | resolution | result | tolerance |
|---|---|---|---|---|
| rigid translation, body crossing > 5 cells | ξ = x − vt, σ = 0 | h = 0.05 | ξ, v exact to 1e-11, ‖σ‖ < 1e-8 | 1e-11 / 1e-8 |
| rigid rotation by π/4 (prescribed velocity) | ξ = R(−θ)x, σ = 0 | h = 0.05 / 0.025 | max ‖ξ error‖ 4.7e-4 → 2.4e-4; max ‖σ‖/μ 3.4e-3 → 1.7e-3 (first order: the explicit update of ξ) | < 2e-3, ratio < 0.6 |
| uniaxial stretch s = 1.25, neo-Hookean | σxx = μ(s²−1)/s + λ ln s/s, σyy = λ ln s/s | h = 0.05 | exact to 1e-9 relative, block stays at rest | 1e-9 |
| simple shear γ = 0.3, neo-Hookean | σxx = μγ², σxy = μγ, σyy = 0 | h = 0.05 | exact to 1e-9, at rest | 1e-9 |
| free block under traction T, T/E = 1e-4 | εxx = T(1−ν²)/E, εyy = −Tν(1+ν)/E | h = 0.1 | linear: 9.0992e-5 / −3.9002e-5 (exact 9.1e-5 / −3.9e-5); neo-Hookean within 5.2e-5 relative | 1e-4 / 5e-4 |
| P-wave, periodic standing wave | √((λ+2μ)/ρ) = 11.6024 | 16 / 32 nodes per wavelength | 11.5395 / 11.5867 (−0.54 % / −0.14 %, second order) | 1 % / 0.25 %, ratio < 0.35 |
| S-wave | √(μ/ρ) = 6.2017 | 16 / 32 | 6.1638 / 6.1922 (−0.61 % / −0.15 %) | same |
| thick cylinder, plane strain (a = 10, b = 20, p = 100, E = 210000, ν = 0.3) | Lamé; FieldCraft FEM (quad8, oracle `lame-cylinder-plane-strain`) u_r(a) = 0.0090793 | h = 1 / 0.5 (10 / 20 nodes across the wall) | u_r(a) 0.008088 / 0.008789 (−10.9 % / −3.2 %); u_r(15) −11.2 % / −3.3 %; σθ(15) 81.9 / 89.3 vs 92.6 (−11.6 % / −3.5 %) | 15 % / 4 %, ratio < 0.4 |

u_r(a) is evaluated from a fit u_r = A r + B/r over the nodes with φ < −2h; the FEM gets the
same number to 3e-6 on a 8 × 16 quad8 quarter.
The checks except the Lamé ones now run with the blurred and the ghost-filled surfaces (§9).
The rigid motions, patch tests and wave speeds also run with the mapped surface
(`tests/mapped.rs`, §10).

Where it falls short:
- **The free surface.** Homogeneous states are exact (linear fields are reproduced by every
  stencil and by the extrapolation). With a curved surface and a strain gradient across it,
  the blur zone makes the ring too stiff: −11 % with ten nodes across the wall, −3.2 % with
  twenty (observed order ≈ 1.8, as the incompressible paper reports for its FSI tests). A wider
  blur is stiffer (ε = 1.0 h: −8.5 %; 2.5 h: −16.8 % at h = 1). Stresses at the surface itself
  are not usable; the numbers above are a node layer or more inside.
  §8 finds the cause.
- **Extrapolation** is linear: first-order wrong where the strain varies, and its fit needs
  three non-collinear nodes, so features thinner than about two cells are not represented.
- **Statics by relaxation.** The cylinder takes 1200 steps (2601 nodes) at h = 1 and about
  4000 steps (10201 nodes) at h = 0.5 to come to rest; the FEM solves the same problem with one
  sparse factorisation of under a thousand unknowns, more accurately. For a solid alone the RMT is
  not competitive with the FEM; its case is FSI with large solid motion on the fluid's grid.
- **Rotation** is transported with a first-order time error (explicit update of ξ with the
  current ∇ξ); second order in time would need a predictor–corrector or an RK stage.
- Not done: 3D, threads, contact between bodies, prescribed moving boundaries, the
  incompressible law the LBRMT uses, the LB coupling (step 3).

## 8. Why the blurred surface is biased

Two suspects: the blur itself (the boundary condition spread over a band 2ε wide) and the
band's ζ, a linear extrapolation that continues the interior strain past the surface. The
`blur_diagnosis` example separates them on the Lamé cylinder (neo-Hookean, p = 100; errors of
u_r(a) from the fit over φ < −2h, relative), with two switches the solid keeps for this:
`Solid::set_exterior_map` prescribes the exact Lamé ζ in the band instead of extrapolating
it, and `Boundary::BlurredBody` integrates the momentum only at the body's nodes (φ < 0) with
v extrapolated into the band ("sm" in the study notes).

| band ζ | momentum at | ε / h | κ | γ × | h = 1 | h = 0.5 | h = 0.25 |
|---|---|---|---|---|---|---|---|
| extrapolated | φ < ε (published) | 1.5 | 0.1 | 1 | −10.9 % | −3.17 % | −0.85 % |
| exact | φ < ε | 1.5 | 0.1 | 1 | +0.30 % | +0.11 % | – (unstable) |
| extrapolated | φ < 0 | 1.5 | 0.1 | 1 | −39.1 % | −24.2 % | −13.8 % |
| exact | φ < 0 | 1.5 | 0.1 | 1 | +0.31 % | +0.088 % | +0.017 % |
| exact | φ < ε | 1.0 / 2.0 | 0.1 | 1 | +0.15 % / +0.53 % | +0.075 % / +0.18 % | |
| extrapolated | φ < ε | 1.0 / 2.0 | 0.1 | 1 | −8.5 % / −13.9 % | −2.3 % / −4.5 % | |
| extrapolated | φ < ε | 1.5 | 0 | 1 | −39.2 % | −24.2 % | |
| extrapolated | φ < ε | 1.5 | 0.1 | 0.5 / 2 / 4 | −7.5 % / −15.4 % / −20.7 % | | |
| exact | φ < 0 | 1.5 | 0.1 | 0.5 / 2 / 4 | +0.31 % / +0.31 % / +0.31 % | | |

What this shows:
- **The extrapolation is the bias.** With the exact ζ in the band the published scheme is
  within 0.3 % at h = 1; the blur width alone costs a few tenths of a percent (+0.15 / +0.30 /
  +0.53 % for ε = 1.0 / 1.5 / 2.0 h).
- **The published −10.9 % is a cancellation.** The band nodes with 0 < φ < ε integrate
  momentum, but their ζ is overwritten by the extrapolation every step, so they never come to
  rest, and the artificial viscosity turns their velocity into a surface stress that offsets
  part of the extrapolation error. Without the
  band's momentum (φ < 0), or without the viscosity, the true extrapolation error shows:
  −39 % at h = 1, order 0.4–1.0.
- **The static answer depends on the damping** used to reach it: −7.5 % to −20.7 % for
  γ × 0.5 to × 4 at h = 1, and on κ. With the momentum in the body and the exact band, both
  drop out.
- Measured and dropped (not committed): quadratic least-squares extrapolation (−3.7 % at h = 1,
  +0.16 % at h = 0.25, but noisy and divergent without viscosity), harmonic extensions (worse
  and unstable), other blur kernels (none removes the bias; the best with the exact band gives
  +0.10 % at h = 1), averaging over grid rotations (the angular spread also comes from the
  extrapolation).

A flat face isolates the same effect without curvature or staircase (a slab |x| < L under a
body force, σxx falling linearly to zero at the faces): the blurred surface puts the face at
an effective position off by a fixed fraction of h that depends mostly on where the face cuts
the grid, so the stress error is O(1) · (dσ/dn) · h, first order.

| | ε / h | face θ h past a node: 0 | 0.25 | 0.5 | 0.75 |
|---|---|---|---|---|---|
| blurred, effective face shift δ / h | 1.0 | +0.06 | +0.25 | +0.82 | +2.51 |
| | 1.5 | +0.13 | +0.30 | +0.66 | +1.47 |
| | 2.5 | +0.24 | +0.39 | +0.63 | +1.01 |
| blurred, σxx error at the last node / (ρ g h) | 1.5 | 0.82 | 0.91 | 1.21 | 1.98 |

δ / h is the same at h = 1, 0.5 and 0.25 (0.658 h at θ = 0.5, ε = 1.5 h): first order.

So a sharp surface needs two things: the traction imposed where the surface is, by the
solid's own stress (no blending), and no stress computed from extrapolated ζ.

## 9. A sharp boundary by ghost values (kept for comparison)

`Boundary::Ghost` (`crates/eulerian/src/sharp.rs`) was the first sharp surface. It is kept as
an option for the comparison studies; §10's mapped cut cells replace it.

Momentum is integrated at the body's nodes only; every edge with a body node at either end
uses the solid stress alone (no blending); the first 8-neighbour ring of nodes outside (the
ghosts) gets ζ from a local constrained least-squares fit: a polynomial ζ(d) per component
fitted to the body nodes in a 7 × 7 (cubic) or 5 × 5 (quadratic) window, subject to the two
equations [σ(G₀) + C:(∇ζ − G₀)]·n = σ_ambient·n at the surface point nearest the ghost
(the stress linearised about the previous fit, C by differences of the material law; two
passes, so the traction is exact at equilibrium for any law). The surface point is the
projection d ← d − φ∇φ/|∇φ|² onto φ₀(x + ζ(x)) = 0; the normal is ∇φ there. Outer band
layers stay linear (they only feed φ and the upwind advection).

Rules for the hard cases:
- small cut cells: nothing divides by the cut distance, so a surface arbitrarily close to a
  node gives a regular system;
- a surface that comes to rest on a node: body nodes leave at φ ≥ 0, band nodes enter only at
  φ < −h/10 (without this hysteresis the finite-strain cylinder rang forever, 8 bore nodes
  flipping in and out); a ghost may therefore lie up to h/10 inside, and the projection runs
  from either side;
- fits the data barely determine (the inside of a convex corner): the KKT system is
  equilibrated and a pivot below 1e-6 refuses the fit; then the quadratic, then a linear fit
  with the same traction, then the plain extrapolation (counted; zero in every case below);
- other things measured and dropped: other fit weights (no change: the error is truncation),
  extra traction rows at the window's edge crossings (doubled the error: they over-constrain
  the fit's gradient), the quadratic fit (exact for linear stress, but 3–5 times the cubic's
  error at a curved surface).

Stability: with no artificial viscosity (κ = 0) a growing surface mode appears after a few
thousand steps in the cylinder, at any time step (CFL 0.4 and 0.2 alike): the closure is not
symmetric. κ = 0.02–0.1 damps it, with or without the relaxation damping, and the static
answer does not depend on κ or γ.

Verification (linear elastic at strain 1e-6 unless noted; `examples/boundary_study.rs`). The
slab of §8 is exact to the relaxation tolerance wherever its faces cut the grid (effective
face shift −7e-7 to −6e-6 h). Lamé cylinder (a = 10, b = 20; relative errors; σ at the bore
extrapolated from the three nearest nodes on y = 0):

| h | blurred u_r(a) | order | ghost quadratic u_r(a) | order | ghost cubic u_r(a) | order | cubic σθ(15) | cubic σrr(a)/p (−1) | cubic σθ(a) |
|---|---|---|---|---|---|---|---|---|---|
| 2 | −2.61e-1 | – | −1.07e-1 | – | −1.42e-1 | – | −1.33e-1 | −0.838 | −1.80e-1 |
| 1 | −1.12e-1 | 1.22 | −3.69e-2 | 1.54 | −2.57e-2 | 2.47 | −2.71e-2 | −0.986 | −4.61e-2 |
| 0.5 | −3.45e-2 | 1.70 | −1.11e-2 | 1.73 | −4.01e-3 | 2.68 | −5.32e-3 | −1.0015 | −1.01e-2 |
| 0.25 | −1.01e-2 | 1.78 | −3.12e-3 | 1.83 | −6.29e-4 | 2.67 | −1.25e-3 | −1.0011 | −2.15e-3 |
| 0.125 | −2.75e-3 | 1.87 | −7.64e-4 | 2.03 | −5.89e-5 | 3.42 | +6.6e-5 | −1.0004 | −4.14e-4 |

The maximum nodal stress error of the cubic fit converges at order 2.2 (1.5e-2 p at h = 0.5).
The blurred surface's σrr at the bore is −0.03 / −0.46 / −0.72 / −0.86 / −0.93 p for h = 2 …
0.125; the ghost fill reaches −p within 0.2 % from h = 0.5.

Kirsch hole (a = 5 in a disc R = 20 with the Kirsch traction on r = R, so the finite domain
is exact; σθ at the hole edge extrapolated along the axis):

| h | blurred σθ(a, 90°)/S (3) | σθ(a, 0°)/S (−1) | ghost quadratic | | ghost cubic | |
|---|---|---|---|---|---|---|
| 1 | 2.184 | −0.254 | 3.169 | −1.338 | 3.166 | −1.321 |
| 0.5 | 2.693 | −0.668 | 3.056 | −1.111 | 3.010 | −1.054 |
| 0.25 | 2.938 | −0.897 | 3.020 | −1.036 | 2.997 | −1.007 |
| 0.125 | 3.002 | −0.972 | 3.006 | −1.011 | 2.998 | −1.000 |

The blurred value at 90° crosses 3 S on its way from below (+0.05 % at h = 0.125 is a
coincidence: its error around the hole oscillates between −0.02 S and +0.07 S at h = 0.25,
against ±0.02 S for the cubic fit). Maximum nodal stress error: blurred 0.135 S, cubic 5.4e-3 S
at h = 0.125.

Rotating annulus (a = 10, b = 20, free surfaces, centrifugal body force, closed form
u = C₁ r + C₂/r − K r³; non-harmonic):

| h | blurred u_r(a) | σθ(a) | ghost cubic u_r(a) | σθ(a) | max σ error / σθ(a) |
|---|---|---|---|---|---|
| 1 | −7.8e-2 | −2.3e-2 | −1.96e-2 | −2.8e-2 | 2.5e-2 |
| 0.5 | −2.4e-2 | +1.8e-2 | −3.0e-3 | −6.2e-3 | 5.6e-3 |
| 0.25 | −6.8e-3 | +1.9e-2 | −4.8e-4 | −1.3e-3 | 1.2e-3 |

(orders of the ghost fill 2.7 / 2.2 / 2.2; the blurred σθ(a) does not converge). The static
answer no longer depends on the damping: u_r(a) at h = 1 equal to 2e-5 for γ × 0.5 and × 4
(blurred: −7.9 % against −20.9 %).

**Finite strain: neo-Hookean cylinder.** The decoupled neo-Hookean law of FieldCraft's FEM
(μ, κ from E = 210000, ν = 0.3), bore pressure p = 8000 (p/E = 0.038; the linear solution
scaled to it is u_r(a)/a = 7.3 %, the nonlinear one 8.2 %), applied in ten increments.
Reference: FieldCraft's FEM, nonlinear static with follower pressure, plane-strain quad8
quarter, converged by refinement (16 × 32 → 32 × 64: u_r(a) 0.815032 → 0.815032, σθ at r = 15
in the deformed body 8777.3 → 8777.5).

| | u_r(a) | σθ(r = 15) | σθ at the bore | σrr at the bore |
|---|---|---|---|---|
| linear Lamé scaled to p | 0.726 | 7407 | 13333 | −8000 |
| FEM 32 × 64 | 0.815032 | 8777.5 | 14700 | −7998 |
| ghost, h = 2 | 0.6440 | 7114 | 11548 | −7188 |
| ghost, h = 1 | 0.7924 | 8471 | 13972 | −8151 |
| ghost, h = 0.5 | 0.8108 | 8714 | 14558 | −8040 |
| ghost, h = 0.25 | 0.8144 | 8765 | 14677 | −8009 |
| ghost, h = 0.125 | 0.81499 (−4.7e-5) | 8777.9 | 14697 | −8004 |
| Richardson (0.5, 0.25, 0.125) | 0.81510 (order 2.7) | 8782.5 (order 1.95) | 14701 (order 2.6) | |

The blurred surface folds (det ∇ξ < 0) during the load ramp at every resolution at
p = 8000 and at p = 4000, and at some resolutions at p = 2000–3000.

**Cost** (Apple M4 Max, release build, one thread per run, cylinder, linear):

| h | nodes | blurred steps | ms/step | wall s | ghost steps | ms/step | wall s | ghost u_r(a) error |
|---|---|---|---|---|---|---|---|---|
| 2 | 1089 | 780 | 0.15 | 0.1 | 1200 | 1.1 | 1.3 | 1.4e-1 |
| 1 | 2809 | 1860 | 0.35 | 0.6 | 2140 | 2.3 | 4.9 | 2.6e-2 |
| 0.5 | 8649 | 4240 | 0.93 | 3.9 | 4300 | 4.8 | 20.5 | 4.0e-3 |
| 0.25 | 29929 | 7820 | 2.9 | 22.4 | 9120 | 10.6 | 96.5 | 6.3e-4 |
| 0.125 | 110889 | 16480 | 9.8 | 161 | 17220 | 34.4 | 591 | 5.9e-5 |

A profile at h = 0.25: ghost fits 65 %, grid neighbour lookups 12 %, the linear band fill 11 %,
edge stresses and momentum 6 %. Steps to rest grow as 1/h, so a static answer by relaxation
costs O(h⁻³) in 2D: 10³–10⁵ times the FEM, which reaches u_r(a) within 6e-3 on 1 × 2 quad8
elements, 6e-4 on 2 × 4 and 4e-5 on 4 × 8 (121 nodes; σθ(15) within 8e-4), each in about
1 ms, and the finite-strain case in 0.04–0.9 s (8 × 16 to 32 × 64).

## 10. Mapped cut cells (the default surface)

`Boundary::Mapped` (`crates/eulerian/src/{cut,mapped}.rs`) is the default surface. It removes
both causes of §8: no blending, no stress from extrapolated ξ, momentum only inside the body.

- **Control volumes.** A node inside the body (φ < 0) carries momentum for the solid part of
  its dual cell. The surface in each cut grid cell is a quadratic curve through its two edge
  crossings and a third point on φ₀(ξ) = 0 above the chord. Areas, centroids, the solid
  pieces of the faces and the surface pieces of each quarter cell come from Green's theorem
  over that curve (three Gauss points per side, exact for it), so every control volume closes
  exactly and a constant stress puts no net force on any cut cell. The lumped mass is ρ times
  the true solid area. Solid in the dual cell of an outside node is shared between the inside
  neighbours across the surface along x and along y, with smooth weights in |∂φ/∂x| against
  |∂φ/∂y|; its net force and mass go with it. A control volume below a quarter of h² would
  join its deepest inside neighbour (never needed in the cases below).
- **Fluxes.** Faces between full cells keep the compact interior stencil. Faces next to a cut
  cell integrate σ(F)·n with two Gauss points on each solid piece; on the curved side the flux
  is the prescribed traction σ_ambient·n with the curve's line Jacobian.
- **The deformation gradient near the surface** comes from a surface-fitted map. Where a grid
  line along axis d crosses the surface (up to about 72° from the normal) there is a surface
  point with its own value ξ_S. Between neighbouring lines, T(u, η) = (x_d′ = (l + u) h,
  x_d = Y(u) − σ η h) maps the unit square onto a cell whose side η = 0 is the surface, Y(u)
  the crossings interpolated with the same Lagrange shape functions in u as the field;
  ζ(u, η) = Σ L_l(u) p_l(η), p_l quadratic through ξ_S and the two inside nodes on line l;
  ∇ζ = (∂ζ/∂u, ∂ζ/∂η) T′⁻¹ and F = (I + ∇ζ)⁻¹. Near the diagonal both families of lines give
  a map and the two are blended smoothly.
- **Surface values** satisfy σ(F)·n = t at every surface point: Newton on the conditions of
  each chain of points along the surface, banded finite-difference Jacobian, LU with partial
  pivoting, warm-started from the previous step (two or three iterations).
- **The band** is still filled by the linear least-squares extrapolation, but only the ξ
  advection (upwind neighbours) and the crossing positions use it.

Three things the first version got wrong and the final one fixes: a discrete choice that can
flip between steps makes the force discontinuous and the explicit scheme chatters instead of
coming to rest (the family of lines chosen at 45°, the inside node that takes an outside
node's solid); Gauss–Seidel along the surface diverges at the ends of a family of lines on a
tight curve (hence the banded Newton); the cut state must belong to the configuration it is
used with (it is rebuilt at the end of each step).

Verification (`crates/eulerian/tests/mapped.rs`, and the `rmt_cases` example for the
studies; same cases and metrics as before, u_r(a) from the fit over φ < −2h):

| check | result |
|---|---|
| uniaxial stretch 1.25, simple shear 0.3 (neo-Hookean disc held by the matching ambient stress) | exact to 1e-9 at every node and surface point, at rest |
| rigid translation of a disc across the grid | exact to round-off; the surface map amplifies it where the surface passes very close to a node (1/s weights): v to 5e-10, σ to 4e-9 (blurred: 1e-11) |
| rigid rotation of a free disc by 90° (dynamic) | ‖σ‖/μ 3.0e-3 (the centrifugal response ρω²R²/μ = 6.5e-3), ξ within 2e-3 |
| P- and S-wave speeds (no surface) | the plain stencil: same as §7 |

Lamé cylinder, |u_r(a) error| (%):

| h | blurred | blur + exact ξ in band (§8) | mapped, p = 100 neo-Hookean | mapped, small strain (load × 0.01) |
|---|---|---|---|---|
| 2 | 30.3 | – | 0.85 | 0.99 |
| 1 | 10.9 | 0.31 | 0.154 | 0.023 |
| 0.5 | 3.17 | 0.088 | 0.181 | 0.0004 |
| 0.25 | 0.85 | 0.017 | 0.184 | 0.0018 |
| 0.125 | 0.11 | – | 0.185 | 0.0016 |

The mapped error at p = 100 levels at +0.18 %: that is the finite-strain answer (the pressure
acts on the expanded bore, u is measured at the current position), not a discretisation
error; with the load cut a hundredfold it falls to 0.0018 %, the same floor scaled. Below h =
0.5 the small-strain error sits on that floor, so the order is read off the stresses: the
worst σθ over the bore surface points (small strain) is 6.7, 1.4, 0.32, 0.082, 0.026 % for h =
2 … 0.125 (orders 2.3, 2.2, 2.0, 1.7). Rotating disc (ρω² = 0.5, both surfaces free): u_r(a)
small strain −6.9, −0.015, +0.003, +0.002, +0.001 %; worst bore σθ 4.2, 0.89, 0.20, 0.052,
0.017 % (blurred u_r(a): −47, −7.6, −2.2, −0.54, −0.04 %). Kirsch hole (a = 5 in a disc of
radius 20 with the exact traction outside): RMS error of σθ around the hole edge 3.4, 3.7,
0.53, 0.12, 0.027 % of 3S (orders 2.8, 2.2, 2.1 from h = 1); σθ(a, 90°) within 0.04 % of 3S
from h = 0.5; the blurred surface's worst node next to the hole is 30, 18, 6.2, 3.6, 2.1 % off.

- **Damping** no longer moves the answer: u_r(a) at h = 0.5 is 0.00022 / 0.00036 / 0.00041 %
  for γ × 0.5 / 1 / 4 (blurred: −1.9 / −3.2 / −7.7 %), nor does the artificial viscosity
  (κ = 0.02: +0.150 % against +0.154 % at h = 1; blurred −22.6 % against −10.9 %).
- **The band** decides nothing measurable: 2, 4 or 6 extrapolation layers give identical
  results; prescribing the exact ξ there moves u_r(a) by 0.004 % (h = 1) and 0.015 % (0.5).
- **Time step.** With κ = 0.02 (so the viscous limit does not cap Δt) the mapped scheme relaxes
  up to Δt = 0.7 h/c_P at h = 1 and 0.8 at 0.5, the blurred one up to 0.8; the default 0.4 has
  margin. The largest eigenvalue of the mapped operator is 0.22–0.26 of the explicit limit at
  0.4. The operator is not symmetric near the surface (asymmetry 0.17 of its largest entry at
  h = 1); its complex eigenvalues grow by up to 0.5 % per step without damping, so undamped
  dynamics need the artificial viscosity (κ = 0.1 suffices; the blurred scheme without it
  diverges at t = 0.5).
- **Cost** per step on one core (Lamé; blurred / mapped, ms): h = 2: 0.10 / 0.66; 1: 0.31 /
  1.5; 0.5: 1.0 / 3.5; 0.25: 3.6 / 9.0; 0.125: 13.8 / 26.6. The cut cells (≈ 0.9 ms at h =
  0.5) and the surface solve (≈ 0.6 ms) grow with the surface, about × 2.2 and × 1.5 per
  halving of h; the rest is the interior's O(n) work, so the ratio falls from 6.6 to 1.9.

**The three surfaces side by side** (small strain; the blurred and ghost columns from §9's
study at strain 1e-6, the mapped one from this section's at load × 0.01; per-step costs from
the two studies' own runs on one core):

| | h | blurred | ghost fill (cubic) | mapped |
|---|---|---|---|---|
| Lamé u_r(a) error | 1 | −11.2 % | −2.57 % | 0.023 % |
| | 0.5 | −3.45 % | −0.40 % | 0.0004 % |
| | 0.25 | −1.01 % | −0.063 % | 0.0018 % (floor) |
| Kirsch σθ(a, 90°)/S (exact 3) | 0.5 | 2.693 | 3.010 | within 0.04 % of 3 |
| | 0.25 | 2.938 | 2.997 | within 0.04 % of 3 |
| rotating disc u_r(a) error | 1 | −7.8 % | −1.96 % | −0.015 % |
| | 0.5 | −2.4 % | −0.30 % | +0.003 % |
| static answer, γ × 0.5 against × 4 | 1 / 0.5 | −7.9 / −20.9 % (h = 1) | equal to 2e-5 (h = 1) | 0.00022 / 0.00041 % (h = 0.5) |
| order of the surface stress | | none (σθ(a) does not converge) | 2.2 | 2.0–2.3 |
| ms per step, h = 0.125 | | 9.8–13.8 | 34.4 | 26.6 |

## Papers

All read from the open-access versions listed; journal details checked against the published
DOI records (Crossref) where marked.

1. K. Kamrin, J.-C. Nave, *An Eulerian approach to the simulation of deformable solids:
   Application to finite-strain elasticity*, arXiv:0901.3799 (2009).
   <https://arxiv.org/abs/0901.3799> — the first RMT paper (journal version not checked).
2. K. Kamrin, C. H. Rycroft, J.-C. Nave, *Reference map technique for finite-strain elasticity
   and fluid–solid interaction*, J. Mech. Phys. Solids 60(11):1952–1969 (2012).
   doi:[10.1016/j.jmps.2012.06.003](https://doi.org/10.1016/j.jmps.2012.06.003) (verified).
   Read from the author copy <https://www.math.mcgill.ca/jcnave/papers/elasticity.pdf>.
3. B. Valkov, C. H. Rycroft, K. Kamrin, *Eulerian method for multiphase interactions of soft
   solid bodies in fluids*, J. Appl. Mech. 82(4):041011 (2015).
   doi:[10.1115/1.4029765](https://doi.org/10.1115/1.4029765) (verified). Preprint
   arXiv:1409.6183, titled *Eulerian method for fluid–structure interaction and submerged
   solid–solid contact problems*.
4. C. H. Rycroft, C.-H. Wu, Y. Yu, K. Kamrin, *Reference map technique for incompressible
   fluid–structure interaction*, J. Fluid Mech. 898:A9 (2020).
   doi:[10.1017/jfm.2020.353](https://doi.org/10.1017/jfm.2020.353) (verified). Preprint
   arXiv:1810.03015.
5. Y. Sun, C. H. Rycroft, *A fully-integrated lattice Boltzmann method for fluid–structure
   interaction*, J. Comput. Phys. 526:113774 (2025).
   doi:[10.1016/j.jcp.2025.113774](https://doi.org/10.1016/j.jcp.2025.113774) (verified).
   Read from the preprint arXiv:2402.12696 (2024), <https://arxiv.org/abs/2402.12696>; section
   and equation numbers cited in the code are the preprint's.
6. Y. Sun, *A lattice Boltzmann implementation of the reference map technique*, master's
   thesis, Harvard University (2020).
   <https://people.math.wisc.edu/~chr/publish/2020_Sun.pdf>.

For the sharp boundaries (§9, §10), journal records checked against Crossref (metadata only;
the methods were written from the equations, these are pointers to the established ideas, not
sources of code or text):

7. R. Mittal, H. Dong, M. Bozkurttas, F. M. Najjar, A. Vargas, A. von Loebbecke, *A versatile
   sharp interface immersed boundary method for incompressible flows with complex
   boundaries*, J. Comput. Phys. 227(10):4825–4852 (2008). doi:10.1016/j.jcp.2008.01.028
   (verified): ghost cells with the boundary condition at the boundary intercept.
8. P. T. Barton, D. Drikakis, *An Eulerian method for multi-component problems in non-linear
   elasticity with sliding interfaces*, J. Comput. Phys. 229(15):5518–5540 (2010).
   doi:10.1016/j.jcp.2010.04.012 (record verified; its ghost-fluid treatment not checked).
9. T. D. Aslam, *A partial differential equation approach to multidimensional extrapolation*,
   J. Comput. Phys. 193(1):349–355 (2004). doi:10.1016/j.jcp.2003.08.001 (verified).
10. H. Johansen, P. Colella, *A Cartesian grid embedded boundary method for Poisson's equation
    on irregular domains*, J. Comput. Phys. 147(1):60–85 (1998). doi:10.1006/jcph.1998.5965
    (verified): conservative cut cells.
11. C. S. Peskin, *The immersed boundary method*, Acta Numerica 11:479–517 (2002).
    doi:10.1017/S0962492902000077 (verified).

Background cited by these papers and used in the prototype: J. Bonet, R. D. Wood, *Nonlinear
Continuum Mechanics for Finite Element Analysis*, 2nd ed., Cambridge 2008 (the compressible
neo-Hookean law σ = (μ/J)(B − I) + (λ ln J / J) I); T. D. Aslam, *A partial differential
equation approach to multidimensional extrapolation*, J. Comput. Phys. 193(1):349–355 (2004).
