//! Closed-form answers of the verification cases, shared by the study examples: the Lamé
//! cylinder, the rotating annular disc and the Kirsch hole (plane strain, linear elastic).
#![allow(dead_code)]

pub const E: f64 = 210_000.0;
pub const NU: f64 = 0.3;
pub const P: f64 = 100.0;
pub const RW2: f64 = 0.5;
pub const S_FAR: f64 = 100.0;

pub fn lame_constants() -> (f64, f64) {
    let mu = E / (2.0 * (1.0 + NU));
    (E * NU / ((1.0 + NU) * (1.0 - 2.0 * NU)), mu)
}

/// Lamé cylinder a = 10, b = 20, bore pressure P, plane strain: u_r and (σ_r, σ_θ).
pub fn lame(r: f64) -> (f64, f64, f64) {
    let (a, b) = (10.0f64, 20.0f64);
    let k = P * a * a / (b * b - a * a);
    ((1.0 + NU) * k / E * ((1.0 - 2.0 * NU) * r + b * b / r), k * (1.0 - b * b / (r * r)), k * (1.0 + b * b / (r * r)))
}

/// Rotating annular disc a = 10, b = 20, ρω² = RW2, both surfaces free, plane strain.
pub fn disc(r: f64) -> (f64, f64, f64) {
    let (a, b) = (10.0f64, 20.0f64);
    let (l, m) = lame_constants();
    let k = RW2 / (8.0 * (l + 2.0 * m));
    let c2 = (4.0 * l + 6.0 * m) * k * a * a * b * b / (2.0 * m);
    let c1 = (4.0 * l + 6.0 * m) * k * (a * a + b * b) / (2.0 * (l + m));
    let ur = c1 * r + c2 / r - k * r * r * r;
    let sr = 2.0 * (l + m) * c1 - 2.0 * m * c2 / (r * r) - (4.0 * l + 6.0 * m) * k * r * r;
    let st = 2.0 * (l + m) * c1 + 2.0 * m * c2 / (r * r) - (4.0 * l + 2.0 * m) * k * r * r;
    (ur, sr, st)
}

/// Kirsch: a hole of radius A_K in an infinite plate under σxx = S_FAR at infinity. Cartesian
/// stress (σxx, σyy, σxy) at x.
pub const A_K: f64 = 5.0;
pub fn kirsch_stress(x: [f64; 2]) -> [f64; 3] {
    let r = x[0].hypot(x[1]);
    let th = x[1].atan2(x[0]);
    let q = A_K * A_K / (r * r);
    let s = S_FAR;
    let (c2, s2) = ((2.0 * th).cos(), (2.0 * th).sin());
    let srr = 0.5 * s * (1.0 - q) + 0.5 * s * (1.0 - 4.0 * q + 3.0 * q * q) * c2;
    let stt = 0.5 * s * (1.0 + q) - 0.5 * s * (1.0 + 3.0 * q * q) * c2;
    let srt = -0.5 * s * (1.0 + 2.0 * q - 3.0 * q * q) * s2;
    let (c, sn) = (th.cos(), th.sin());
    [
        srr * c * c + stt * sn * sn - 2.0 * srt * sn * c,
        srr * sn * sn + stt * c * c + 2.0 * srt * sn * c,
        (srr - stt) * sn * c + srt * (c * c - sn * sn),
    ]
}

/// Kirsch displacement (plane strain), for the exact band.
pub fn kirsch_u(x: [f64; 2]) -> [f64; 2] {
    let r = x[0].hypot(x[1]);
    let th = x[1].atan2(x[0]);
    let mu = E / (2.0 * (1.0 + NU));
    let kappa = 3.0 - 4.0 * NU;
    let a = A_K;
    let s = S_FAR;
    let c2 = (2.0 * th).cos();
    let ur = s / (4.0 * mu) * (r * (kappa - 1.0) / 2.0 + r * c2 + a * a / r * (1.0 + (1.0 + kappa) * c2) - a.powi(4) / r.powi(3) * c2);
    let ut = s / (4.0 * mu) * ((1.0 - kappa) * a * a / r - r - a.powi(4) / r.powi(3)) * (2.0 * th).sin();
    let (c, sn) = (th.cos(), th.sin());
    [ur * c - ut * sn, ur * sn + ut * c]
}

pub fn polar(st: [f64; 4], x: [f64; 2]) -> (f64, f64, f64) {
    let r = x[0].hypot(x[1]);
    let (c, s) = (x[0] / r, x[1] / r);
    let srr = st[0] * c * c + st[1] * s * s + 2.0 * st[2] * s * c;
    let stt = st[0] * s * s + st[1] * c * c - 2.0 * st[2] * s * c;
    let srt = (st[1] - st[0]) * s * c + st[2] * (c * c - s * s);
    (srr, stt, srt)
}
