//! Geometric multigrid for the linearised static problem on the grid, and restarted GMRES.
//!
//! The unknowns are the two components of ζ at the body nodes that hold a control volume; the
//! matrix is stored in 2 × 2 blocks, one row of blocks per such node, in node order.
//!
//! **Hierarchy.** Level l + 1 is the grid of every second node of level l (node (I, J) at
//! fine (2I, 2J)). A fine node takes its value from its coarse parents by bilinear
//! interpolation: weight 1 from a coincident parent, ½ from each of the two parents of a node
//! on a coarse line, ¼ from each of the four parents of a cell centre; the same weight for
//! both components. **Coarsening rule at the surface:** a coarse node is a coarse unknown when
//! the fine node on it is a fine unknown (injection of the body), and also when a fine unknown
//! would otherwise have no coarse parent (its parent of largest weight is added). A fine
//! unknown interpolates from the parents that are coarse unknowns, its weights renormalised to
//! sum to one: constants are reproduced everywhere, linear fields (the rigid rotation) exactly
//! in the interior and to O(h) at the surface. Every coarse unknown has a fine row of its own
//! (the node on it, or the unknown it was added for), so P has full column rank. (Taking every
//! parent of a fine unknown instead keeps linear fields exact but makes P rank deficient where
//! several coarse nodes outside the body parent only the same cut-cell node, and the Galerkin
//! operator singular.) The coarse operators are Galerkin products
//! A_{l+1} = Pᵀ A_l P, so the cut cells, the surface conditions and the merged control volumes
//! enter every level through the fine operator and no cut geometry is rebuilt on a coarse grid.
//!
//! **Smoother.** Point-block Gauss–Seidel (each node's 2 × 2 diagonal block inverted), forward
//! before the coarse correction and backward after it, so the V-cycle is a fixed linear operator
//! and plain (not flexible) GMRES can use it.
//!
//! **Coarsest level.** A dense LU with partial pivoting. A body held by no node (free in the
//! plane) has the three rigid motions as near-null vectors of every level; there the coarsest
//! system is bordered with them, [A Φ; Φᵀ 0], which solves it in the complement of the rigid
//! motions and stays regular when A is singular on them.
//!
//! GMRES (Saad & Schultz 1986): restarted, right-preconditioned (so it monitors the true
//! residual), modified Gram–Schmidt, Givens rotations.

use crate::Error;

/// A 2 × 2 block, row-major.
pub(crate) type Block = [[f64; 2]; 2];

/// Coarsest level size (nodes) for the dense solve. Coarsening further puts the surface of
/// the test bodies within one or two coarse cells of each other (an annulus wall of 10 at
/// spacing 4), where the renormalised interpolation is crude; stopping at about 240 nodes
/// (spacing 2 there) took the iterations at h = 0.0625 from 28 to 25.
const COARSEST: usize = 400;
/// Most levels.
const MAX_LEVELS: usize = 12;

/// A sparse matrix of 2 × 2 blocks in compressed rows.
#[derive(Debug, Clone, Default)]
pub(crate) struct BlockMatrix {
    rows: usize,
    ptr: Vec<usize>,
    col: Vec<usize>,
    val: Vec<Block>,
}

impl BlockMatrix {
    /// From rows of (column, block) entries (any order, duplicates summed).
    pub(crate) fn from_rows(rows: Vec<Vec<(usize, Block)>>) -> BlockMatrix {
        let n = rows.len();
        let mut ptr = Vec::with_capacity(n + 1);
        let mut col = Vec::new();
        let mut val: Vec<Block> = Vec::new();
        ptr.push(0);
        for mut row in rows {
            row.sort_by_key(|e| e.0);
            let start = col.len();
            for (c, b) in row {
                if col.len() > start && col.last() == Some(&c) {
                    if let Some(last) = val.last_mut() {
                        add_block(last, &b, 1.0);
                    }
                } else {
                    col.push(c);
                    val.push(b);
                }
            }
            ptr.push(col.len());
        }
        BlockMatrix { rows: n, ptr, col, val }
    }

    /// Rows of blocks.
    pub(crate) fn rows(&self) -> usize {
        self.rows
    }

    /// Stored blocks.
    pub(crate) fn blocks(&self) -> usize {
        self.col.len()
    }

    /// y = A x.
    pub(crate) fn mul(&self, x: &[f64], y: &mut [f64]) {
        for r in 0..self.rows {
            let mut s = [0.0; 2];
            for e in self.ptr[r]..self.ptr[r + 1] {
                let c = self.col[e];
                let b = &self.val[e];
                s[0] += b[0][0] * x[2 * c] + b[0][1] * x[2 * c + 1];
                s[1] += b[1][0] * x[2 * c] + b[1][1] * x[2 * c + 1];
            }
            y[2 * r] = s[0];
            y[2 * r + 1] = s[1];
        }
    }

    fn diagonal(&self, r: usize) -> Block {
        for e in self.ptr[r]..self.ptr[r + 1] {
            if self.col[e] == r {
                return self.val[e];
            }
        }
        [[0.0; 2]; 2]
    }

    /// One Gauss–Seidel sweep with the inverted diagonal blocks, forward or backward.
    fn sweep(&self, dinv: &[Block], b: &[f64], x: &mut [f64], forward: bool) {
        for i in 0..self.rows {
            let r = if forward { i } else { self.rows - 1 - i };
            let mut s = [b[2 * r], b[2 * r + 1]];
            for e in self.ptr[r]..self.ptr[r + 1] {
                let c = self.col[e];
                if c == r {
                    continue;
                }
                let m = &self.val[e];
                s[0] -= m[0][0] * x[2 * c] + m[0][1] * x[2 * c + 1];
                s[1] -= m[1][0] * x[2 * c] + m[1][1] * x[2 * c + 1];
            }
            let d = &dinv[r];
            x[2 * r] = d[0][0] * s[0] + d[0][1] * s[1];
            x[2 * r + 1] = d[1][0] * s[0] + d[1][1] * s[1];
        }
    }
}

fn add_block(a: &mut Block, b: &Block, w: f64) {
    for r in 0..2 {
        for c in 0..2 {
            a[r][c] += w * b[r][c];
        }
    }
}

/// The inverse of a diagonal block; a zero block (a node without stiffness) is left alone.
fn invert_block(d: &Block) -> Block {
    let det = d[0][0] * d[1][1] - d[0][1] * d[1][0];
    let scale = d.iter().flatten().fold(0.0f64, |a, v| a.max(v.abs()));
    if !(scale > 0.0) || !(det.abs() > 1e-12 * scale * scale) {
        return if scale > 0.0 { [[1.0 / scale, 0.0], [0.0, 1.0 / scale]] } else { [[0.0; 2]; 2] };
    }
    [[d[1][1] / det, -d[0][1] / det], [-d[1][0] / det, d[0][0] / det]]
}

/// One level of the hierarchy: its operator, the inverted diagonal blocks and the
/// interpolation from the next coarser level (parents of each node of this level).
#[derive(Debug, Clone)]
struct Level {
    a: BlockMatrix,
    dinv: Vec<Block>,
    /// Up to four (coarse node, weight) per node; empty on the coarsest level.
    parents: Vec<Vec<(usize, f64)>>,
}

/// The coarsest level's factorised dense matrix (bordered with the rigid motions when the
/// body is free).
#[derive(Debug, Clone)]
struct DenseLu {
    n: usize,
    a: Vec<f64>,
    piv: Vec<usize>,
    /// Unknowns of the level itself (the rest are the multipliers of the border).
    size: usize,
}

impl DenseLu {
    fn factor(n: usize, mut a: Vec<f64>, size: usize) -> Result<DenseLu, Error> {
        let mut piv = vec![0; n];
        let scale = a.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        for k in 0..n {
            let mut p = k;
            for i in k + 1..n {
                if a[i * n + k].abs() > a[p * n + k].abs() {
                    p = i;
                }
            }
            let pv = a[p * n + k];
            if !(pv.abs() > 1e-14 * scale) || !pv.is_finite() {
                return Err(Error::Diverged(format!("the coarsest multigrid level is singular (pivot {k} of {n}: {pv:e}, scale {scale:e})")));
            }
            piv[k] = p;
            if p != k {
                for j in 0..n {
                    a.swap(k * n + j, p * n + j);
                }
            }
            for i in k + 1..n {
                let f = a[i * n + k] / pv;
                a[i * n + k] = f;
                if f != 0.0 {
                    for j in k + 1..n {
                        a[i * n + j] -= f * a[k * n + j];
                    }
                }
            }
        }
        Ok(DenseLu { n, a, piv, size })
    }

    fn solve(&self, b: &[f64]) -> Vec<f64> {
        let n = self.n;
        let mut x = vec![0.0; n];
        x[..b.len().min(n)].copy_from_slice(&b[..b.len().min(n)]);
        for k in 0..n {
            let p = self.piv[k];
            if p != k {
                x.swap(k, p);
            }
            for i in k + 1..n {
                x[i] -= self.a[i * n + k] * x[k];
            }
        }
        for k in (0..n).rev() {
            let mut s = x[k];
            for j in k + 1..n {
                s -= self.a[k * n + j] * x[j];
            }
            x[k] = s / self.a[k * n + k];
        }
        x.truncate(self.size);
        x
    }
}

/// A multigrid V-cycle preconditioner for a block matrix on grid nodes.
#[derive(Debug, Clone)]
pub(crate) struct Multigrid {
    levels: Vec<Level>,
    coarse: DenseLu,
    sweeps: usize,
}

/// Counters of a hierarchy.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct MultigridStats {
    /// Levels including the finest and the coarsest.
    pub levels: usize,
    /// Unknown nodes on the finest level.
    pub fine_nodes: usize,
    /// Unknown nodes on the coarsest level.
    pub coarse_nodes: usize,
    /// Stored blocks over all levels divided by those of the finest (the operator complexity).
    pub complexity: f64,
}

impl Multigrid {
    /// Builds the hierarchy for `a`, whose row r belongs to grid node `ij[r]` of a grid of
    /// `dims` nodes at `origin` with spacing `h`. With `rigid` (the centre of the body), the
    /// coarsest system is bordered with the translations and the rotation about it.
    pub(crate) fn new(
        a: BlockMatrix,
        ij: Vec<(usize, usize)>,
        dims: (usize, usize),
        origin: [f64; 2],
        h: f64,
        rigid: Option<[f64; 2]>,
        sweeps: usize,
    ) -> Result<Multigrid, Error> {
        let mut levels = Vec::new();
        let mut a = a;
        let mut ij = ij;
        let mut dims = dims;
        let mut spacing = h;
        while a.rows() > COARSEST && levels.len() + 1 < MAX_LEVELS {
            let cdims = (dims.0 / 2 + 1, dims.1 / 2 + 1);
            // Coarse unknowns: the coarse nodes on a fine unknown, then for every fine unknown
            // left without one among its parents, its parent of largest weight (first in order).
            let mut kept = vec![false; cdims.0 * cdims.1];
            for &(i, j) in &ij {
                if i % 2 == 0 && j % 2 == 0 && i / 2 < cdims.0 && j / 2 < cdims.1 {
                    kept[i / 2 + cdims.0 * (j / 2)] = true;
                }
            }
            let mut parents_grid: Vec<Vec<(usize, f64)>> = Vec::with_capacity(ij.len());
            for &(i, j) in &ij {
                let along = |i: usize, n: usize| -> Vec<(usize, f64)> {
                    if i.is_multiple_of(2) { vec![(i / 2, 1.0)] } else { [(i / 2, 0.5), (i / 2 + 1, 0.5)].into_iter().filter(|p| p.0 < n).collect() }
                };
                let mut ps = Vec::with_capacity(4);
                for (ci, wi) in along(i, cdims.0) {
                    for &(cj, wj) in &along(j, cdims.1) {
                        ps.push((ci + cdims.0 * cj, wi * wj));
                    }
                }
                parents_grid.push(ps);
            }
            for ps in &parents_grid {
                if !ps.iter().any(|p| kept[p.0])
                    && let Some(best) = ps.iter().fold(None::<(usize, f64)>, |b, p| if b.is_none_or(|b| p.1 > b.1) { Some(*p) } else { b })
                {
                    kept[best.0] = true;
                }
            }
            let mut id = vec![usize::MAX; cdims.0 * cdims.1];
            let mut cij = Vec::new();
            for (g, slot) in id.iter_mut().enumerate() {
                if kept[g] {
                    *slot = cij.len();
                    cij.push((g % cdims.0, g / cdims.0));
                }
            }
            // Each fine unknown interpolates from its kept parents, the weights renormalised.
            let parents: Vec<Vec<(usize, f64)>> = parents_grid
                .into_iter()
                .map(|ps| {
                    let total: f64 = ps.iter().filter(|p| kept[p.0]).map(|p| p.1).sum();
                    ps.into_iter().filter(|p| kept[p.0]).map(|(g, w)| (id[g], w / total)).collect()
                })
                .collect();
            // Galerkin product Pᵀ A P, row by coarse row through the children.
            let nc = cij.len();
            let mut children: Vec<Vec<(usize, f64)>> = vec![Vec::new(); nc];
            for (f, ps) in parents.iter().enumerate() {
                for &(c, w) in ps {
                    children[c].push((f, w));
                }
            }
            let mut mark = vec![usize::MAX; nc];
            let mut slot = vec![0usize; nc];
            let mut rows = Vec::with_capacity(nc);
            for (c, ch) in children.iter().enumerate() {
                let mut row: Vec<(usize, Block)> = Vec::new();
                for &(f, wf) in ch {
                    for e in a.ptr[f]..a.ptr[f + 1] {
                        let g = a.col[e];
                        for &(cc, wg) in &parents[g] {
                            if mark[cc] != c {
                                mark[cc] = c;
                                slot[cc] = row.len();
                                row.push((cc, [[0.0; 2]; 2]));
                            }
                            if let Some(entry) = row.get_mut(slot[cc]) {
                                add_block(&mut entry.1, &a.val[e], wf * wg);
                            }
                        }
                    }
                }
                rows.push(row);
            }
            let coarse = BlockMatrix::from_rows(rows);
            let dinv = (0..a.rows()).map(|r| invert_block(&a.diagonal(r))).collect();
            levels.push(Level { a, dinv, parents });
            a = coarse;
            ij = cij;
            dims = cdims;
            spacing *= 2.0;
        }
        // Coarsest: dense, bordered with the rigid motions of a free body.
        let n = a.rows();
        let border = if rigid.is_some() { 3 } else { 0 };
        let m = 2 * n + border;
        let mut dense = vec![0.0; m * m];
        for r in 0..n {
            for e in a.ptr[r]..a.ptr[r + 1] {
                let c = a.col[e];
                for p in 0..2 {
                    for q in 0..2 {
                        dense[(2 * r + p) * m + 2 * c + q] = a.val[e][p][q];
                    }
                }
            }
        }
        if let Some(centre) = rigid {
            // Scale the border like the matrix so the pivots stay balanced.
            let s = a.val.iter().flatten().flatten().fold(0.0f64, |x, v| x.max(v.abs())).max(1e-300);
            for (r, &(i, j)) in ij.iter().enumerate() {
                let x = [origin[0] + i as f64 * spacing - centre[0], origin[1] + j as f64 * spacing - centre[1]];
                let modes = [[1.0, 0.0], [0.0, 1.0], [-x[1] / spacing, x[0] / spacing]];
                for (k, md) in modes.iter().enumerate() {
                    for p in 0..2 {
                        dense[(2 * r + p) * m + 2 * n + k] = s * md[p];
                        dense[(2 * n + k) * m + 2 * r + p] = s * md[p];
                    }
                }
            }
        }
        let coarse = DenseLu::factor(m, dense, 2 * n)?;
        let dinv = (0..a.rows()).map(|r| invert_block(&a.diagonal(r))).collect();
        levels.push(Level { a, dinv, parents: Vec::new() });
        Ok(Multigrid { levels, coarse, sweeps: sweeps.max(1) })
    }

    /// The counters of the hierarchy.
    pub(crate) fn stats(&self) -> MultigridStats {
        let fine = self.levels.first().map_or(0, |l| l.a.blocks()).max(1);
        let total: usize = self.levels.iter().map(|l| l.a.blocks()).sum();
        MultigridStats {
            levels: self.levels.len(),
            fine_nodes: self.levels.first().map_or(0, |l| l.a.rows()),
            coarse_nodes: self.levels.last().map_or(0, |l| l.a.rows()),
            complexity: total as f64 / fine as f64,
        }
    }

    /// One V-cycle from a zero guess: an approximation of A⁻¹ b.
    pub(crate) fn apply(&self, b: &[f64]) -> Vec<f64> {
        self.cycle(0, b)
    }

    fn cycle(&self, l: usize, b: &[f64]) -> Vec<f64> {
        let lv = &self.levels[l];
        if l + 1 == self.levels.len() {
            return self.coarse.solve(b);
        }
        let n = lv.a.rows();
        let mut x = vec![0.0; 2 * n];
        for _ in 0..self.sweeps {
            lv.a.sweep(&lv.dinv, b, &mut x, true);
        }
        let mut ax = vec![0.0; 2 * n];
        lv.a.mul(&x, &mut ax);
        let nc = self.levels[l + 1].a.rows();
        let mut bc = vec![0.0; 2 * nc];
        for (f, ps) in lv.parents.iter().enumerate() {
            for &(c, w) in ps {
                bc[2 * c] += w * (b[2 * f] - ax[2 * f]);
                bc[2 * c + 1] += w * (b[2 * f + 1] - ax[2 * f + 1]);
            }
        }
        let xc = self.cycle(l + 1, &bc);
        for (f, ps) in lv.parents.iter().enumerate() {
            for &(c, w) in ps {
                x[2 * f] += w * xc[2 * c];
                x[2 * f + 1] += w * xc[2 * c + 1];
            }
        }
        for _ in 0..self.sweeps {
            lv.a.sweep(&lv.dinv, b, &mut x, false);
        }
        x
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

/// The outcome of [`gmres`].
#[derive(Debug, Clone)]
pub(crate) struct Krylov {
    /// The solution.
    pub(crate) x: Vec<f64>,
    /// Operator applications.
    pub(crate) iterations: usize,
    /// ‖b − A x‖ / ‖b‖ (the GMRES estimate).
    pub(crate) relative_residual: f64,
    /// The relative residual after each iteration.
    pub(crate) history: Vec<f64>,
}

/// Restarted GMRES with right preconditioning: solves A x = b to ‖b − A x‖ ≤ rtol ‖b‖ with at
/// most `max_iter` applications of A; `apply(v)` is A v, `precondition(v)` approximates A⁻¹ v.
pub(crate) fn gmres(
    b: &[f64],
    apply: &mut dyn FnMut(&[f64]) -> Result<Vec<f64>, Error>,
    precondition: &mut dyn FnMut(&[f64]) -> Vec<f64>,
    rtol: f64,
    restart: usize,
    max_iter: usize,
) -> Result<Krylov, Error> {
    let n = b.len();
    let mut x = vec![0.0; n];
    let bnorm = norm(b);
    if !(bnorm > 0.0) {
        return Ok(Krylov { x, iterations: 0, relative_residual: 0.0, history: Vec::new() });
    }
    let m = restart.max(1);
    let mut total = 0;
    let mut history = Vec::new();
    loop {
        let r: Vec<f64> = if total == 0 {
            b.to_vec()
        } else {
            let ax = apply(&x)?;
            total += 1;
            b.iter().zip(&ax).map(|(p, q)| p - q).collect()
        };
        let beta = norm(&r);
        let mut rel = beta / bnorm;
        if rel <= rtol || total >= max_iter {
            return Ok(Krylov { x, iterations: total, relative_residual: rel, history });
        }
        let mut v: Vec<Vec<f64>> = vec![r.iter().map(|q| q / beta).collect()];
        let mut z: Vec<Vec<f64>> = Vec::with_capacity(m);
        let mut hcol: Vec<Vec<f64>> = Vec::with_capacity(m);
        let mut cs: Vec<(f64, f64)> = Vec::with_capacity(m);
        let mut g = vec![0.0; m + 1];
        g[0] = beta;
        let mut k = 0;
        while k < m && total < max_iter {
            let zk = precondition(&v[k]);
            let mut w = apply(&zk)?;
            total += 1;
            z.push(zk);
            let mut hk = vec![0.0; k + 2];
            for (i, vi) in v.iter().enumerate() {
                let hij = dot(&w, vi);
                hk[i] = hij;
                for (wq, vq) in w.iter_mut().zip(vi) {
                    *wq -= hij * vq;
                }
            }
            let wn = norm(&w);
            hk[k + 1] = wn;
            for (i, &(c, s)) in cs.iter().enumerate() {
                let (a0, a1) = (hk[i], hk[i + 1]);
                hk[i] = c * a0 + s * a1;
                hk[i + 1] = -s * a0 + c * a1;
            }
            let den = hk[k].hypot(hk[k + 1]);
            let (c, s) = if den > 0.0 { (hk[k] / den, hk[k + 1] / den) } else { (1.0, 0.0) };
            hk[k] = den;
            hk[k + 1] = 0.0;
            cs.push((c, s));
            g[k + 1] = -s * g[k];
            g[k] *= c;
            hcol.push(hk);
            k += 1;
            rel = g[k].abs() / bnorm;
            history.push(rel);
            if rel <= rtol || !(wn > 0.0) {
                break;
            }
            v.push(w.iter().map(|q| q / wn).collect());
        }
        // Back substitution H y = g, then x += Z y.
        let mut y = vec![0.0; k];
        for i in (0..k).rev() {
            let mut s = g[i];
            for j in i + 1..k {
                s -= hcol[j][i] * y[j];
            }
            y[i] = if hcol[i][i] != 0.0 { s / hcol[i][i] } else { 0.0 };
        }
        for (j, zj) in z.iter().enumerate() {
            for (xq, zq) in x.iter_mut().zip(zj) {
                *xq += y[j] * zq;
            }
        }
        if rel <= rtol || total >= max_iter {
            if !x.iter().all(|q| q.is_finite()) {
                return Err(Error::Diverged("the Krylov solve produced a non-finite update".into()));
            }
            return Ok(Krylov { x, iterations: total, relative_residual: rel, history });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The 5-point Laplacian (both components, coupled weakly) on an n × n grid with the
    /// boundary held: the V-cycle preconditioned GMRES needs a mesh-independent iteration count.
    fn laplace(n: usize) -> (BlockMatrix, Vec<(usize, usize)>) {
        let mut rows = Vec::new();
        let mut ij = Vec::new();
        let id = |i: usize, j: usize| (i - 1) + (n - 2) * (j - 1);
        for j in 1..n - 1 {
            for i in 1..n - 1 {
                let mut row = vec![(id(i, j), [[4.2, 0.1], [0.1, 4.2]])];
                for (di, dj) in [(-1isize, 0isize), (1, 0), (0, -1), (0, 1)] {
                    let (a, b) = (i as isize + di, j as isize + dj);
                    if a >= 1 && b >= 1 && (a as usize) < n - 1 && (b as usize) < n - 1 {
                        row.push((id(a as usize, b as usize), [[-1.0, 0.0], [0.0, -1.0]]));
                    }
                }
                rows.push(row);
                ij.push((i, j));
            }
        }
        (BlockMatrix::from_rows(rows), ij)
    }

    #[test]
    fn multigrid_gmres_iterations_do_not_grow_with_the_grid() {
        let mut its = Vec::new();
        for n in [33, 65, 129] {
            let (a, ij) = laplace(n);
            let mg = Multigrid::new(a.clone(), ij, (n, n), [0.0; 2], 1.0, None, 2).unwrap();
            let b: Vec<f64> = (0..2 * a.rows()).map(|k| ((k * 7919) % 13) as f64 - 6.0).collect();
            let mut apply = |v: &[f64]| -> Result<Vec<f64>, Error> {
                let mut y = vec![0.0; v.len()];
                a.mul(v, &mut y);
                Ok(y)
            };
            let mut pre = |v: &[f64]| mg.apply(v);
            let k = gmres(&b, &mut apply, &mut pre, 1e-8, 30, 200).unwrap();
            let mut ax = vec![0.0; b.len()];
            a.mul(&k.x, &mut ax);
            let res: f64 = ax.iter().zip(&b).map(|(p, q)| (p - q) * (p - q)).sum::<f64>().sqrt();
            assert!(res <= 1e-7 * norm(&b), "{res}");
            its.push(k.iterations);
        }
        assert!(its.iter().all(|&i| i <= 12) && its[2] <= its[0] + 2, "{its:?}");
    }

    #[test]
    fn gmres_without_preconditioner_solves_a_nonsymmetric_system() {
        // Convection–diffusion in 1D (non-symmetric), 40 unknowns stored as 20 blocks.
        let n = 20;
        let rows: Vec<Vec<(usize, Block)>> = (0..n)
            .map(|r| {
                let mut row = vec![(r, [[2.0, -0.7], [-1.3, 2.0]])];
                if r > 0 {
                    row.push((r - 1, [[-0.3, 0.0], [-0.6, -0.3]]));
                }
                if r + 1 < n {
                    row.push((r + 1, [[-0.6, 0.0], [0.0, -0.4]]));
                }
                row
            })
            .collect();
        let a = BlockMatrix::from_rows(rows);
        let b: Vec<f64> = (0..2 * n).map(|k| (k as f64).sin()).collect();
        let mut apply = |v: &[f64]| -> Result<Vec<f64>, Error> {
            let mut y = vec![0.0; v.len()];
            a.mul(v, &mut y);
            Ok(y)
        };
        let mut pre = |v: &[f64]| v.to_vec();
        let k = gmres(&b, &mut apply, &mut pre, 1e-10, 8, 400).unwrap();
        assert!(k.relative_residual <= 1e-10, "{}", k.relative_residual);
    }
}
