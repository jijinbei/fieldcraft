//! Stepping the deep interior elsewhere (a GPU): the hooks a device backend needs.
//!
//! Far from the surface the mapped step is a pure stencil on full cells: the compact edge
//! stress, the momentum of a full control volume (h² of solid, its own) and the upwind
//! advection of ζ. A backend may take those nodes over: [`Solid::set_remote`] marks them,
//! and the CPU step then skips their faces (between two remote nodes), their momentum and
//! their advection, while doing everything else (the band, the cut cells, the surface chains,
//! the faces next to the surface) as before. The backend steps the remote nodes with
//! [`StepConstants`] and exchanges with the CPU, every step, the values of the nodes on either
//! side of the seam within the stencil's reach (two rings): ζ and v of the remote nodes next to
//! the CPU's, and of the CPU's next to the remote ones ([`Solid::node_state`],
//! [`Solid::set_node_state`]).
//!
//! The remote nodes must stay deep inside the body: [`Solid::deep_interior`] finds the solid
//! nodes with only solid nodes within a number of rings. The CPU needs its own values up to
//! [`CPU_RINGS`] rings inside the surface (the band's extrapolation windows, the surface map
//! and the cut cells' corners), so a backend marks only nodes deeper than that and checks
//! as the body moves that they still are.

use crate::material::Material;
use crate::solid::{Boundary, NodeKind, Solid};
use crate::{Error, finite, par};

/// Rings inside the surface whose values the CPU step computes itself: the band's
/// extrapolation windows reach four rings inside, the surface map and the cut cells' corners
/// two, the stencils of the faces next to the surface two more.
pub const CPU_RINGS: usize = 6;

/// The constants of the step a device needs to step the deep interior.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StepConstants {
    /// Nodes along x and y.
    pub nx: usize,
    pub ny: usize,
    /// Whether each axis wraps.
    pub periodic: [bool; 2],
    /// Node spacing.
    pub h: f64,
    /// The time step.
    pub dt: f64,
    /// The material's Lamé constants and reference density.
    pub lambda: f64,
    pub mu: f64,
    pub density: f64,
    /// Compressible neo-Hookean (else small-strain linear elastic; [`Solid::set_remote`]
    /// refuses the decoupled law).
    pub neo_hookean: bool,
    /// The artificial viscosity μ_e = κ ρ₀ c_P h.
    pub viscosity: f64,
    /// The mass-proportional damping rate γ.
    pub damping: f64,
}

impl Solid {
    /// The constants of the step (see [`StepConstants`]).
    pub fn step_constants(&self) -> StepConstants {
        let (lambda, mu, density) = self.material.constants();
        StepConstants {
            nx: self.grid.nx(),
            ny: self.grid.ny(),
            periodic: self.grid.periodic(),
            h: self.grid.h(),
            dt: self.dt,
            lambda,
            mu,
            density,
            neo_hookean: matches!(self.material, Material::NeoHookean { .. }),
            viscosity: self.settings.viscosity * density * self.material.p_wave_speed() * self.grid.h(),
            damping: self.settings.damping,
        }
    }

    /// What the next step multiplies the acceleration by: Δt/2 for the first step (the
    /// velocity lives at half steps), else Δt.
    pub fn next_kick(&self) -> f64 {
        if self.started { self.dt } else { 0.5 * self.dt }
    }

    /// Marks the nodes stepped elsewhere (`None` clears it): the CPU step then skips the faces
    /// between two of them, their momentum and their advection, and keeps their ζ and v as
    /// [`Solid::set_node_state`] last set them. Only for the mapped boundary, and only nodes
    /// inside the body that are not held: everything within [`CPU_RINGS`] rings of a node
    /// outside the body is refused (the CPU step needs those values).
    pub fn set_remote(&mut self, remote: Option<Vec<bool>>) -> Result<(), Error> {
        let Some(remote) = remote else {
            self.remote = None;
            return Ok(());
        };
        if self.settings.boundary != Boundary::Mapped {
            return Err(Error::Invalid("not supported yet: stepping nodes elsewhere with a boundary other than the mapped one".into()));
        }
        if matches!(self.material, Material::NeoHookeanDecoupled { .. }) {
            return Err(Error::Invalid("not supported yet: stepping nodes elsewhere with the decoupled neo-Hookean law".into()));
        }
        if remote.len() != self.grid.len() {
            return Err(Error::Invalid(format!("{} remote flags for {} nodes", remote.len(), self.grid.len())));
        }
        let deep = self.deep_interior(CPU_RINGS);
        if let Some(k) = par::first(remote.len(), |k| remote[k] && !deep[k]) {
            return Err(Error::Invalid(format!("node {k} is within {CPU_RINGS} rings of the surface: the CPU must step it")));
        }
        let all = remote.iter().zip(&self.kind).all(|(&r, k)| r || !k.known());
        self.remote = Some(crate::solid::Remote { nodes: remote, all });
        Ok(())
    }

    /// The nodes stepped elsewhere, when some are.
    pub fn remote(&self) -> Option<&[bool]> {
        self.remote.as_ref().map(|r| r.nodes.as_slice())
    }

    /// The solid (not held) nodes with only solid or held nodes within `rings` rings (the
    /// square of side 2 rings + 1 around them, wrapping periodic axes; a square reaching past a
    /// bounded edge of the grid does not count).
    pub fn deep_interior(&self, rings: usize) -> Vec<bool> {
        let (nx, ny) = (self.grid.nx(), self.grid.ny());
        let periodic = self.grid.periodic();
        let r = rings as isize;
        // Non-source nodes per row window, then per column window of those counts.
        let bad: Vec<u32> = par::map(self.grid.len(), |k| u32::from(!self.kind[k].source()));
        let window = |len: usize, wrap: bool, at: &dyn Fn(usize) -> u32, i: usize| -> u32 {
            let mut c = 0;
            for o in -r..=r {
                let t = i as isize + o;
                let t = if wrap {
                    t.rem_euclid(len as isize)
                } else if (0..len as isize).contains(&t) {
                    t
                } else {
                    return 1;
                };
                c += at(t as usize);
            }
            c
        };
        let rows: Vec<u32> = par::map(self.grid.len(), |k| {
            let (i, j) = self.grid.ij(k);
            window(nx, periodic[0], &|t| bad[t + nx * j], i)
        });
        par::map(self.grid.len(), |k| {
            let (i, j) = self.grid.ij(k);
            self.kind[k] == NodeKind::Solid && window(ny, periodic[1], &|t| rows[i + nx * t], j) == 0
        })
    }

    /// ζ = ξ − x and v at node k, whatever its kind.
    pub fn node_state(&self, k: usize) -> Option<([f64; 2], [f64; 2])> {
        Some((*self.zeta.get(k)?, *self.v.get(k)?))
    }

    /// Sets ζ and v at node k (a remote node's new values, from the device that steps it).
    pub fn set_node_state(&mut self, k: usize, zeta: [f64; 2], v: [f64; 2]) -> Result<(), Error> {
        for x in zeta.iter().chain(v.iter()) {
            finite("node state", *x)?;
        }
        let (Some(z), Some(w)) = (self.zeta.get_mut(k), self.v.get_mut(k)) else {
            return Err(Error::Invalid(format!("no node {k}")));
        };
        *z = zeta;
        *w = v;
        Ok(())
    }

    /// The body force per unit volume at grid position x (times the load factor), when one
    /// is set.
    pub fn body_force_at(&self, x: [f64; 2]) -> Option<[f64; 2]> {
        self.body.as_ref().map(|b| {
            let f = (b.0)(x);
            [self.load * f[0], self.load * f[1]]
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Grid, Limits, Settings, Shape};

    fn disc(settings: Settings, material: Material) -> Solid {
        let l = Limits::default();
        let g = Grid::new(41, 41, 0.05, [-1.0, -1.0], [false; 2], &l).unwrap();
        Solid::new(g, material, Shape::Disc { center: [0.01, -0.02], radius: 0.8 }, settings, l).unwrap()
    }

    /// The step leaves the remote nodes as set, and nodes near the surface are refused.
    #[test]
    fn remote_nodes_are_left_alone_and_nodes_near_the_surface_refused() {
        let m = Material::neo_hookean(100.0, 0.3, 1.0).unwrap();
        let mut s = disc(Settings::default(), m);
        s.set_ambient_field(|_| [-1.0, -1.0, 0.0]).unwrap();
        let deep = s.deep_interior(CPU_RINGS);
        let n = s.grid().len();
        assert!(deep.iter().filter(|&&d| d).count() > 100);
        // A solid node that is not deep enough.
        let shallow = (0..n).find(|&k| s.kind(k) == Some(NodeKind::Solid) && !deep[k]).unwrap();
        let mut bad = deep.clone();
        bad[shallow] = true;
        assert!(s.set_remote(Some(bad)).is_err());
        assert!(s.set_remote(Some(vec![false; 3])).is_err());
        s.set_remote(Some(deep.clone())).unwrap();
        let before: Vec<_> = (0..n).filter(|&k| deep[k]).map(|k| s.node_state(k)).collect();
        s.run(5).unwrap();
        let after: Vec<_> = (0..n).filter(|&k| deep[k]).map(|k| s.node_state(k)).collect();
        assert_eq!(before, after);
        assert!(s.max_speed() > 0.0, "the CPU's nodes moved");
        // Only the mapped boundary and the laws a device has.
        let mut blurred = disc(Settings { boundary: Boundary::Blurred, ..Settings::default() }, m);
        assert!(blurred.set_remote(Some(blurred.deep_interior(CPU_RINGS))).is_err());
        let mut decoupled = disc(Settings::default(), Material::neo_hookean_decoupled(100.0, 0.3, 1.0).unwrap());
        assert!(decoupled.set_remote(Some(decoupled.deep_interior(CPU_RINGS))).is_err());
    }
}
