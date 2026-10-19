//! The regular 2D grid of nodes.
//!
//! Node (i, j) sits at `origin + (i h, j h)`, i in 0..nx, j in 0..ny, stored row by row
//! (index i + nx j). Each axis is periodic or bounded; across a periodic axis the neighbour of
//! the last node is the first, and since neighbours are addressed by offsets the distance to
//! a neighbour is always the offset times h.

use crate::{Error, Limits, finite, positive};

/// A regular grid of nodes with spacing `h`.
#[derive(Debug, Clone, PartialEq)]
pub struct Grid {
    nx: usize,
    ny: usize,
    h: f64,
    origin: [f64; 2],
    periodic: [bool; 2],
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
        Ok(Grid { nx, ny, h, origin, periodic })
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

    /// Number of nodes.
    pub fn len(&self) -> usize {
        self.nx * self.ny
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

    /// The index of node (i, j).
    pub fn index(&self, i: usize, j: usize) -> usize {
        i + self.nx * j
    }

    /// The (i, j) of a node index.
    pub fn ij(&self, k: usize) -> (usize, usize) {
        (k % self.nx, k / self.nx)
    }

    /// The position of node k.
    pub fn position(&self, k: usize) -> [f64; 2] {
        let (i, j) = self.ij(k);
        [self.origin[0] + i as f64 * self.h, self.origin[1] + j as f64 * self.h]
    }

    /// The node at offset (di, dj) from node k, wrapping periodic axes; `None` off a bounded edge.
    pub fn neighbour(&self, k: usize, di: isize, dj: isize) -> Option<usize> {
        let (i, j) = self.ij(k);
        let i = shift(i, di, self.nx, self.periodic[0])?;
        let j = shift(j, dj, self.ny, self.periodic[1])?;
        Some(self.index(i, j))
    }
}

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
    fn hostile_sizes_are_refused() {
        let l = Limits { max_nodes: 100, max_steps: 10 };
        assert!(Grid::new(11, 10, 1.0, [0.0; 2], [false; 2], &l).is_err());
        assert!(Grid::new(usize::MAX, 2, 1.0, [0.0; 2], [false; 2], &l).is_err());
        assert!(Grid::new(5, 5, f64::NAN, [0.0; 2], [false; 2], &l).is_err());
        assert!(Grid::new(5, 5, 1.0, [f64::INFINITY, 0.0], [false; 2], &l).is_err());
        assert!(Grid::new(4, 9, 1.0, [0.0; 2], [true, false], &l).is_err());
    }
}
