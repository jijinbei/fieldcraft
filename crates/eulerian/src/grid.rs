//! The regular 2D grid of nodes.
//!
//! Node (i, j) sits at `origin + (i h, j h)`, i in 0..nx, j in 0..ny, stored row by row
//! (index i + nx j). Each axis is periodic or bounded; across a periodic axis the neighbour of
//! the last node is the first, and since neighbours are addressed by offsets the distance to
//! a neighbour is always the offset times h.
//!
//! A grid may be *sparse*: only a chosen set of the lattice's nodes is active (one level of a
//! refined hierarchy, [`crate::refine`]). Node indices then run over the active nodes only, in
//! the same row-by-row order, and a neighbour that is not active is `None`, as off a bounded
//! edge. Everything that walks the grid through [`Grid::len`] and [`Grid::neighbour`] works on
//! a sparse grid unchanged, at a cost proportional to the active nodes.

use crate::{Error, Limits, finite, positive};
use std::sync::Arc;

/// Marks a lattice node that is not active.
const INACTIVE: u32 = u32::MAX;

/// A regular grid of nodes with spacing `h`.
#[derive(Debug, Clone, PartialEq)]
pub struct Grid {
    nx: usize,
    ny: usize,
    h: f64,
    origin: [f64; 2],
    periodic: [bool; 2],
    /// The active nodes of a sparse grid; `None` when every lattice node is active.
    sparse: Option<Arc<Sparse>>,
}

/// The active nodes of a sparse grid.
#[derive(Debug, PartialEq)]
struct Sparse {
    /// (i, j) of each active node, row by row.
    nodes: Vec<(u32, u32)>,
    /// The active index of each lattice node (i + nx j), or [`INACTIVE`].
    slot: Vec<u32>,
}

impl Grid {
    /// A grid of `nx × ny` nodes; a periodic axis needs at least 5 nodes (the widest stencil).
    pub fn new(nx: usize, ny: usize, h: f64, origin: [f64; 2], periodic: [bool; 2], limits: &Limits) -> Result<Grid, Error> {
        positive("grid spacing", h)?;
        finite("grid origin x", origin[0])?;
        finite("grid origin y", origin[1])?;
        if nx < 3 || ny < 3 {
            return Err(Error::Invalid(format!("the grid needs at least 3 × 3 nodes, got {nx} × {ny}")));
        }
        if (periodic[0] && nx < 5) || (periodic[1] && ny < 5) {
            return Err(Error::Invalid("a periodic axis needs at least 5 nodes".into()));
        }
        let n = nx.checked_mul(ny).ok_or_else(|| Error::TooLarge("grid size overflows".into()))?;
        if n > limits.max_nodes {
            return Err(Error::TooLarge(format!("{n} grid nodes, the cap is {}", limits.max_nodes)));
        }
        let far = origin[0].abs().max(origin[1].abs()) + h * nx.max(ny) as f64;
        finite("grid extent", far)?;
        Ok(Grid { nx, ny, h, origin, periodic, sparse: None })
    }

    /// A sparse grid: the lattice of [`Grid::new`] with only the nodes where `active[i + nx j]`
    /// is true. The lattice itself may hold up to 16 times the node cap; the active nodes are
    /// capped as a dense grid's.
    pub fn sparse(nx: usize, ny: usize, h: f64, origin: [f64; 2], periodic: [bool; 2], active: &[bool], limits: &Limits) -> Result<Grid, Error> {
        let lattice = Limits { max_nodes: limits.max_nodes.saturating_mul(16), ..*limits };
        let mut g = Grid::new(nx, ny, h, origin, periodic, &lattice)?;
        if active.len() != nx * ny {
            return Err(Error::Invalid(format!("the active mask has {} entries for a {nx} × {ny} lattice", active.len())));
        }
        let count = active.iter().filter(|&&a| a).count();
        if count > limits.max_nodes || count >= INACTIVE as usize {
            return Err(Error::TooLarge(format!("{count} active nodes, the cap is {}", limits.max_nodes)));
        }
        if count == 0 {
            return Err(Error::Invalid("a sparse grid needs at least one active node".into()));
        }
        let mut nodes = Vec::with_capacity(count);
        let mut slot = vec![INACTIVE; nx * ny];
        for (l, &a) in active.iter().enumerate() {
            if a {
                slot[l] = nodes.len() as u32;
                nodes.push(((l % nx) as u32, (l / nx) as u32));
            }
        }
        g.sparse = Some(Arc::new(Sparse { nodes, slot }));
        Ok(g)
    }

    /// Whether only some lattice nodes are active.
    pub fn is_sparse(&self) -> bool {
        self.sparse.is_some()
    }

    /// The position of lattice node (0, 0).
    pub fn origin(&self) -> [f64; 2] {
        self.origin
    }

    /// The index of lattice node (i, j) when it is active (and inside the lattice).
    pub fn try_index(&self, i: usize, j: usize) -> Option<usize> {
        if i >= self.nx || j >= self.ny {
            return None;
        }
        match &self.sparse {
            None => Some(i + self.nx * j),
            Some(s) => s.slot.get(i + self.nx * j).copied().filter(|&v| v != INACTIVE).map(|v| v as usize),
        }
    }

    /// The lattice node at offset (di, dj) from lattice node (i, j), wrapping periodic axes,
    /// whether or not it is active; `None` off a bounded edge.
    pub fn lattice_shift(&self, i: usize, j: usize, di: isize, dj: isize) -> Option<(usize, usize)> {
        Some((shift(i, di, self.nx, self.periodic[0])?, shift(j, dj, self.ny, self.periodic[1])?))
    }

    /// Nodes along x.
    pub fn nx(&self) -> usize {
        self.nx
    }

    /// Nodes along y.
    pub fn ny(&self) -> usize {
        self.ny
    }

    /// Node spacing.
    pub fn h(&self) -> f64 {
        self.h
    }

    /// Number of (active) nodes.
    pub fn len(&self) -> usize {
        match &self.sparse {
            None => self.nx * self.ny,
            Some(s) => s.nodes.len(),
        }
    }

    /// Whether the grid has no nodes (never: a grid has at least 3 × 3).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether each axis is periodic.
    pub fn periodic(&self) -> [bool; 2] {
        self.periodic
    }

    /// The length of the grid's period along each axis (nodes × h).
    pub fn period(&self) -> [f64; 2] {
        [self.nx as f64 * self.h, self.ny as f64 * self.h]
    }

    /// The index of node (i, j). On a sparse grid, the index of an active node and `len()`
    /// (out of range) for any other; [`Grid::try_index`] says which.
    #[inline]
    pub fn index(&self, i: usize, j: usize) -> usize {
        match &self.sparse {
            None => i + self.nx * j,
            Some(_) => self.try_index(i, j).unwrap_or(self.len()),
        }
    }

    /// The lattice (i, j) of a node index.
    #[inline]
    pub fn ij(&self, k: usize) -> (usize, usize) {
        match &self.sparse {
            None => (k % self.nx, k / self.nx),
            Some(s) => s.nodes.get(k).map_or((usize::MAX, usize::MAX), |&(i, j)| (i as usize, j as usize)),
        }
    }

    /// The position of node k.
    #[inline]
    pub fn position(&self, k: usize) -> [f64; 2] {
        let (i, j) = self.ij(k);
        [self.origin[0] + i as f64 * self.h, self.origin[1] + j as f64 * self.h]
    }

    /// The node at offset (di, dj) from node k, wrapping periodic axes; `None` off a bounded edge.
    #[inline]
    pub fn neighbour(&self, k: usize, di: isize, dj: isize) -> Option<usize> {
        let (i, j) = self.ij(k);
        if i >= self.nx || j >= self.ny {
            return None;
        }
        let i = shift(i, di, self.nx, self.periodic[0])?;
        let j = shift(j, dj, self.ny, self.periodic[1])?;
        match &self.sparse {
            None => Some(i + self.nx * j),
            Some(s) => s.slot.get(i + self.nx * j).copied().filter(|&v| v != INACTIVE).map(|v| v as usize),
        }
    }
}

#[inline]
fn shift(i: usize, d: isize, n: usize, periodic: bool) -> Option<usize> {
    let n = n as isize;
    let t = i as isize + d;
    if periodic {
        Some(t.rem_euclid(n) as usize)
    } else if (0..n).contains(&t) {
        Some(t as usize)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neighbours_wrap_on_periodic_axes_only() {
        let g = Grid::new(5, 4, 0.5, [0.0, 0.0], [true, false], &Limits::default()).unwrap();
        let k = g.index(4, 0);
        assert_eq!(g.neighbour(k, 1, 0), Some(g.index(0, 0)));
        assert_eq!(g.neighbour(k, 0, -1), None);
        assert_eq!(g.position(g.index(2, 3)), [1.0, 1.5]);
    }

    #[test]
    fn a_sparse_grid_walks_its_active_nodes_only() {
        let l = Limits::default();
        let mut active = vec![false; 6 * 5];
        for (i, j) in [(1, 1), (2, 1), (1, 2), (5, 4), (0, 4)] {
            active[i + 6 * j] = true;
        }
        let g = Grid::sparse(6, 5, 0.5, [1.0, 0.0], [true, false], &active, &l).unwrap();
        assert_eq!(g.len(), 5);
        let k = g.try_index(1, 1).unwrap();
        assert_eq!(g.ij(k), (1, 1));
        assert_eq!(g.position(k), [1.5, 0.5]);
        assert_eq!(g.neighbour(k, 1, 0), g.try_index(2, 1));
        assert_eq!(g.neighbour(k, -1, 0), None);
        // Wraps along the periodic axis to an active node.
        assert_eq!(g.neighbour(g.try_index(5, 4).unwrap(), 1, 0), g.try_index(0, 4));
        assert_eq!(g.try_index(3, 3), None);
        assert_eq!(g.index(3, 3), g.len());
        assert!(Grid::sparse(6, 5, 0.5, [0.0; 2], [false; 2], &[false; 30], &l).is_err());
        assert!(Grid::sparse(6, 5, 0.5, [0.0; 2], [false; 2], &[true; 7], &l).is_err());
    }

    #[test]
    fn hostile_sizes_are_refused() {
        let l = Limits { max_nodes: 100, max_steps: 10 };
        assert!(Grid::new(11, 10, 1.0, [0.0; 2], [false; 2], &l).is_err());
        assert!(Grid::new(usize::MAX, 2, 1.0, [0.0; 2], [false; 2], &l).is_err());
        assert!(Grid::new(5, 5, f64::NAN, [0.0; 2], [false; 2], &l).is_err());
        assert!(Grid::new(5, 5, 1.0, [f64::INFINITY, 0.0], [false; 2], &l).is_err());
        assert!(Grid::new(4, 9, 1.0, [0.0; 2], [true, false], &l).is_err());
    }
}
