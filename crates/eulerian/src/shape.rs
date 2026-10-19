//! Reference shapes: the body in its undeformed configuration as a signed distance φ₀(X),
//! negative inside. The deformed body is then {x : φ₀(ξ(x)) < 0} (Valkov, Rycroft & Kamrin
//! 2015): no level set is advected separately.

use crate::{Error, finite, positive};

/// φ₀ of a body that fills the whole grid (no surface).
const EVERYWHERE: f64 = -1.0e30;

/// A body in its reference configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Shape {
    /// The whole grid (periodic problems).
    Everywhere,
    /// An axis-aligned rectangle.
    Rectangle { min: [f64; 2], max: [f64; 2] },
    /// A ring between two concentric circles.
    Annulus { center: [f64; 2], inner: f64, outer: f64 },
    /// A disc.
    Disc { center: [f64; 2], radius: f64 },
}

impl Shape {
    /// Checks the shape's numbers.
    pub fn validate(&self) -> Result<(), Error> {
        match *self {
            Shape::Everywhere => Ok(()),
            Shape::Rectangle { min, max } => {
                for v in [min[0], min[1], max[0], max[1]] {
                    finite("rectangle corner", v)?;
                }
                if min[0] < max[0] && min[1] < max[1] {
                    Ok(())
                } else {
                    Err(Error::Invalid("the rectangle's min corner must be below and left of its max corner".into()))
                }
            }
            Shape::Annulus { center, inner, outer } => {
                finite("annulus centre x", center[0])?;
                finite("annulus centre y", center[1])?;
                positive("annulus inner radius", inner)?;
                positive("annulus outer radius", outer)?;
                if inner < outer { Ok(()) } else { Err(Error::Invalid("the annulus needs inner < outer".into())) }
            }
            Shape::Disc { center, radius } => {
                finite("disc centre x", center[0])?;
                finite("disc centre y", center[1])?;
                positive("disc radius", radius)?;
                Ok(())
            }
        }
    }

    /// The signed distance φ₀ at reference position `x` (negative inside; exact outside the
    /// rectangle's corners too).
    pub fn phi0(&self, x: [f64; 2]) -> f64 {
        match *self {
            Shape::Everywhere => EVERYWHERE,
            Shape::Rectangle { min, max } => {
                let c = [0.5 * (min[0] + max[0]), 0.5 * (min[1] + max[1])];
                let half = [0.5 * (max[0] - min[0]), 0.5 * (max[1] - min[1])];
                let q = [(x[0] - c[0]).abs() - half[0], (x[1] - c[1]).abs() - half[1]];
                let outside = (q[0].max(0.0).powi(2) + q[1].max(0.0).powi(2)).sqrt();
                outside + q[0].max(q[1]).min(0.0)
            }
            Shape::Annulus { center, inner, outer } => {
                let r = (x[0] - center[0]).hypot(x[1] - center[1]);
                (inner - r).max(r - outer)
            }
            Shape::Disc { center, radius } => (x[0] - center[0]).hypot(x[1] - center[1]) - radius,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signed_distances() {
        let r = Shape::Rectangle { min: [0.0, 0.0], max: [2.0, 1.0] };
        assert_eq!(r.phi0([1.0, 0.5]), -0.5);
        assert_eq!(r.phi0([3.0, 0.5]), 1.0);
        assert!((r.phi0([3.0, 2.0]) - 2f64.sqrt()).abs() < 1e-15);
        let a = Shape::Annulus { center: [0.0, 0.0], inner: 1.0, outer: 2.0 };
        assert_eq!(a.phi0([1.5, 0.0]), -0.5);
        assert_eq!(a.phi0([0.25, 0.0]), 0.75);
        assert!(Shape::Annulus { center: [0.0, 0.0], inner: 2.0, outer: 1.0 }.validate().is_err());
        let d = Shape::Disc { center: [1.0, 0.0], radius: 2.0 };
        assert_eq!(d.phi0([1.0, 0.5]), -1.5);
        assert!(Shape::Disc { center: [0.0, 0.0], radius: 0.0 }.validate().is_err());
        assert!(Shape::Rectangle { min: [0.0, f64::NAN], max: [1.0, 1.0] }.validate().is_err());
    }
}
