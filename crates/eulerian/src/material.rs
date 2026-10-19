//! Plane-strain constitutive laws: Cauchy stress from the in-plane deformation gradient.
//!
//! Plane strain: F₃₃ = 1 and no out-of-plane shear, so the left Cauchy–Green tensor is
//! B = F Fᵀ in the plane with B₃₃ = 1, and J = det F of the 2 × 2 block.
//!
//! - **Neo-Hookean** (compressible, Bonet & Wood 2008):
//!   σ = (μ/J)(B − I) + (λ ln J / J) I, so σ₃₃ = λ ln J / J. Stress free under any rotation;
//!   tends to linear elasticity with Lamé constants λ, μ at small strain.
//! - **Linear elastic** (small strain): ε = sym(F − I), σ = λ tr ε I + 2 μ ε, σ₃₃ = λ tr ε.
//!   Not objective: for checks of the small-strain limit only.

use crate::{Error, positive};

/// A stress in the plane-strain state: `[σxx, σyy, σxy, σzz]`.
pub type Stress = [f64; 4];

/// A 2 × 2 tensor, row-major: `t[r][c]`.
pub type Tensor = [[f64; 2]; 2];

/// An isotropic elastic law with Lamé constants λ, μ (shear modulus) and reference density ρ₀.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Material {
    /// Compressible neo-Hookean.
    NeoHookean { lambda: f64, mu: f64, density: f64 },
    /// Small-strain linear elasticity.
    LinearElastic { lambda: f64, mu: f64, density: f64 },
}

impl Material {
    /// A neo-Hookean material from Young's modulus and Poisson's ratio (the small-strain values).
    pub fn neo_hookean(young: f64, poisson: f64, density: f64) -> Result<Material, Error> {
        let (lambda, mu) = lame(young, poisson)?;
        Ok(Material::NeoHookean { lambda, mu, density: positive("density", density)? })
    }

    /// A linear elastic material from Young's modulus and Poisson's ratio.
    pub fn linear_elastic(young: f64, poisson: f64, density: f64) -> Result<Material, Error> {
        let (lambda, mu) = lame(young, poisson)?;
        Ok(Material::LinearElastic { lambda, mu, density: positive("density", density)? })
    }

    /// (λ, μ, ρ₀).
    pub fn constants(&self) -> (f64, f64, f64) {
        match *self {
            Material::NeoHookean { lambda, mu, density } | Material::LinearElastic { lambda, mu, density } => (lambda, mu, density),
        }
    }

    /// The small-strain P-wave speed √((λ + 2μ)/ρ₀).
    pub fn p_wave_speed(&self) -> f64 {
        let (l, m, r) = self.constants();
        ((l + 2.0 * m) / r).sqrt()
    }

    /// The small-strain shear wave speed √(μ/ρ₀).
    pub fn s_wave_speed(&self) -> f64 {
        let (_, m, r) = self.constants();
        (m / r).sqrt()
    }

    /// The Cauchy stress for the in-plane deformation gradient `f`.
    pub fn cauchy(&self, f: &Tensor) -> Result<Stress, Error> {
        match *self {
            Material::NeoHookean { lambda, mu, .. } => {
                let j = f[0][0] * f[1][1] - f[0][1] * f[1][0];
                if !(j > 0.0) {
                    return Err(Error::Diverged(format!("inverted or degenerate deformation, J = {j}")));
                }
                let bxx = f[0][0] * f[0][0] + f[0][1] * f[0][1];
                let byy = f[1][0] * f[1][0] + f[1][1] * f[1][1];
                let bxy = f[0][0] * f[1][0] + f[0][1] * f[1][1];
                let vol = lambda * j.ln() / j;
                let m = mu / j;
                Ok([m * (bxx - 1.0) + vol, m * (byy - 1.0) + vol, m * bxy, vol])
            }
            Material::LinearElastic { lambda, mu, .. } => {
                let exx = f[0][0] - 1.0;
                let eyy = f[1][1] - 1.0;
                let exy = 0.5 * (f[0][1] + f[1][0]);
                let tr = lambda * (exx + eyy);
                Ok([tr + 2.0 * mu * exx, tr + 2.0 * mu * eyy, 2.0 * mu * exy, tr])
            }
        }
    }
}

/// Lamé constants (λ, μ) from E and ν, refusing values without a positive-definite energy.
fn lame(young: f64, poisson: f64) -> Result<(f64, f64), Error> {
    positive("Young's modulus", young)?;
    if !(poisson > -1.0 && poisson < 0.5) {
        return Err(Error::Invalid(format!("Poisson's ratio must lie in (-1, 0.5), got {poisson}")));
    }
    let mu = young / (2.0 * (1.0 + poisson));
    let lambda = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson));
    Ok((lambda, mu))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-12 * (1.0 + b.abs())
    }

    #[test]
    fn neo_hookean_is_stress_free_under_rotation() {
        let m = Material::neo_hookean(1000.0, 0.3, 1.0).unwrap();
        let (c, s) = (0.7_f64.cos(), 0.7_f64.sin());
        let st = m.cauchy(&[[c, -s], [s, c]]).unwrap();
        assert!(st.iter().all(|x| x.abs() < 1e-10), "{st:?}");
    }

    #[test]
    fn neo_hookean_closed_forms() {
        let m = Material::neo_hookean(1000.0, 0.3, 1.0).unwrap();
        let (l, mu, _) = m.constants();
        // Uniaxial stretch s along x (plane strain).
        let s: f64 = 1.3;
        let st = m.cauchy(&[[s, 0.0], [0.0, 1.0]]).unwrap();
        assert!(close(st[0], mu * (s * s - 1.0) / s + l * s.ln() / s));
        assert!(close(st[1], l * s.ln() / s));
        // Simple shear γ: σxx = μγ², σxy = μγ, σyy = σzz = 0.
        let g = 0.4;
        let st = m.cauchy(&[[1.0, g], [0.0, 1.0]]).unwrap();
        assert!(close(st[0], mu * g * g) && close(st[2], mu * g) && st[1].abs() < 1e-12 && st[3].abs() < 1e-12);
    }

    #[test]
    fn inverted_deformation_and_hostile_constants_are_errors() {
        let m = Material::neo_hookean(1000.0, 0.3, 1.0).unwrap();
        assert!(m.cauchy(&[[-1.0, 0.0], [0.0, 1.0]]).is_err());
        assert!(m.cauchy(&[[f64::NAN, 0.0], [0.0, 1.0]]).is_err());
        assert!(Material::neo_hookean(1000.0, 0.5, 1.0).is_err());
        assert!(Material::neo_hookean(-1.0, 0.3, 1.0).is_err());
        assert!(Material::linear_elastic(1.0, 0.3, 0.0).is_err());
    }
}
