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

Background cited by these papers and used in the prototype: J. Bonet, R. D. Wood, *Nonlinear
Continuum Mechanics for Finite Element Analysis*, 2nd ed., Cambridge 2008 (the compressible
neo-Hookean law σ = (μ/J)(B − I) + (λ ln J / J) I); T. D. Aslam, *A partial differential
equation approach to multidimensional extrapolation*, J. Comput. Phys. 193(1):349–355 (2004).
