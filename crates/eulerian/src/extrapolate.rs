//! Weighted least-squares linear extrapolation from known nodes to a node outside the solid.
//!
//! The reference map outside the body is rebuilt each step from the nearby nodes inside it,
//! after the local regression of Rycroft, Wu, Yu & Kamrin (2020, §3.2.2) and Sun & Rycroft
//! (2025, §3.2 of the preprint): fit a linear map a + b·dx to the known values in a square
//! window around the target and evaluate it at the target. Our weights are 1/|dx|² (dx in grid spacings), so the
//! nearest nodes dominate; the papers weight by level-set differences. A fit with collinear
//! samples is degenerate and returns `None`, and the caller widens the window. A linear fit
//! reproduces any linear field exactly, so homogeneous deformations and rigid motions are
//! extrapolated without error.

/// The fitted value at offset (0, 0) from samples `(offset, value)`, offsets in grid spacings;
/// `None` when fewer than three non-collinear samples are given or a value is not finite.
pub fn fit<const N: usize>(samples: &[([f64; 2], [f64; N])]) -> Option<[f64; N]> {
    // Normal equations Mᵀ W M c = Mᵀ W y for the basis (1, dx, dy).
    let mut m = [[0.0f64; 3]; 3];
    let mut rhs = [[0.0f64; N]; 3];
    for (d, y) in samples {
        let r2 = d[0] * d[0] + d[1] * d[1];
        if !(r2 > 0.0) || !r2.is_finite() || y.iter().any(|v| !v.is_finite()) {
            continue;
        }
        let w = 1.0 / r2;
        let b = [1.0, d[0], d[1]];
        for r in 0..3 {
            for c in 0..3 {
                m[r][c] += w * b[r] * b[c];
            }
            for k in 0..N {
                rhs[r][k] += w * b[r] * y[k];
            }
        }
    }
    // Cramer's rule on the 3 × 3 system, with a scale-free degeneracy test.
    let det = det3(&m);
    let scale = m[0][0] * m[1][1] * m[2][2];
    if !(scale > 0.0) || !(det.abs() > 1.0e-9 * scale) {
        return None;
    }
    // Only the constant coefficient (the value at the target) is needed.
    let mut out = [0.0; N];
    for k in 0..N {
        let mut a = m;
        for r in 0..3 {
            a[r][0] = rhs[r][k];
        }
        out[k] = det3(&a) / det;
    }
    out.iter().all(|v| v.is_finite()).then_some(out)
}

fn det3(a: &[[f64; 3]; 3]) -> f64 {
    a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_fields_are_exact_and_collinear_samples_are_refused() {
        let f = |d: [f64; 2]| [2.0 + 0.5 * d[0] - 3.0 * d[1], -1.0 + d[1]];
        let pts = [[-1.0, 0.0], [-1.0, 1.0], [-2.0, -1.0], [0.0, -1.0], [-1.0, -1.0]];
        let s: Vec<_> = pts.iter().map(|&d| (d, f(d))).collect();
        let v = fit(&s).unwrap();
        assert!((v[0] - 2.0).abs() < 1e-12 && (v[1] + 1.0).abs() < 1e-12);
        let line: Vec<_> = [[-1.0, 0.0], [-2.0, 0.0], [-3.0, 0.0]].iter().map(|&d| (d, f(d))).collect();
        assert!(fit(&line).is_none());
        assert!(fit::<2>(&[]).is_none());
        assert!(fit(&[([1.0, 0.0], [f64::NAN]), ([0.0, 1.0], [1.0]), ([1.0, 1.0], [1.0])]).is_none());
    }
}
