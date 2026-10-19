// The deep interior of fieldcraft-eulerian's mapped step, in single precision: the CPU's
// compact edge stress (`Solid::compact_edge`), the momentum of a full control volume and the
// upwind advection of ζ (`Solid::step_mapped`, `Solid::advect_reference`), for the nodes the
// host marks as stepped here. Every stencil node of such a node is inside the body (the host
// marks only nodes CPU_RINGS deep), so no kernel checks for unknown neighbours.
//
// The state is one vec4 per node, (ζx, ζy, vx, vy) with ζ = ξ − x, in two buffers: `cur` holds
// step n, `next` receives step n + 1 (the host swaps the bind groups).
//
// The stress: with G = ∇ζ (small), F = (I + G)⁻¹. Computed as written (B − I = F Fᵀ − I and
// ln J of J ≈ 1) single precision loses the strain to cancellation: at a strain of 1e-4 a
// rounding of 6e-8 in B or J is a relative error of 6e-4 in the stress. With STABLE = 1 the
// stress is formed from small quantities only: H = F − I = −F G, B − I = H + Hᵀ + H Hᵀ,
// 1/J = det(I + G) = 1 + a with a = tr G + det G, ln J = −log1p(a), and log1p by its series
// near 0 (WGSL's log has an absolute error of 2⁻²¹ near 1, which is no use for log(1 + 1e-4)).

override STABLE: u32 = 1u;

struct Params {
    nx: u32,
    ny: u32,
    // Bit 0: x wraps, bit 1: y wraps.
    periodic: u32,
    // 1: compressible neo-Hookean, 0: small-strain linear elastic.
    neo_hookean: u32,
    h: f32,
    dt: f32,
    kick: f32,
    damping: f32,
    lambda: f32,
    mu: f32,
    density: f32,
    viscosity: f32,
    // Nodes in the exchange lists.
    upload_count: u32,
    download_count: u32,
    groups_x: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> P: Params;
// Bit 0: stepped here. Bit 1: face (k, +x) needed here. Bit 2: face (k, +y) needed here.
@group(0) @binding(1) var<storage, read> flags: array<u32>;
@group(0) @binding(2) var<storage, read_write> cur: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> next: array<vec4<f32>>;
// The flux through faces (k, +x) (xy) and (k, +y) (zw).
@group(0) @binding(4) var<storage, read_write> flux: array<vec4<f32>>;
// Body force per unit volume at each node (zero without one).
@group(0) @binding(5) var<storage, read> body: array<vec2<f32>>;
// Exchange: the nodes whose values the host sends, and their values; the nodes whose new
// values the host reads, and those values.
@group(0) @binding(6) var<storage, read> upload_nodes: array<u32>;
@group(0) @binding(7) var<storage, read> upload_values: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read> download_nodes: array<u32>;
@group(0) @binding(9) var<storage, read_write> download_values: array<vec4<f32>>;
// Bit 0: a stepped node folded (det ∇ξ ≤ 0) or a value stopped being a number.
@group(0) @binding(10) var<storage, read_write> status: atomic<u32>;

fn index(gid: vec3<u32>) -> u32 {
    return gid.x + gid.y * P.groups_x * 256u;
}

// The node at offset (di, dj) from k (wrapping periodic axes; the host guarantees bounded
// axes are never crossed by a stepped node's stencil).
fn nb(k: u32, di: i32, dj: i32) -> u32 {
    let nx = i32(P.nx);
    let ny = i32(P.ny);
    var i = i32(k % P.nx) + di;
    var j = i32(k / P.nx) + dj;
    if (P.periodic & 1u) != 0u {
        i = (i % nx + nx) % nx;
    }
    if (P.periodic & 2u) != 0u {
        j = (j % ny + ny) % ny;
    }
    i = clamp(i, 0, nx - 1);
    j = clamp(j, 0, ny - 1);
    return u32(i + nx * j);
}

fn log1p_small(a: f32) -> f32 {
    if abs(a) < 0.1 {
        // 2 atanh(s), s = a / (2 + a): |s| < 0.053, the series to s⁹ is exact in f32.
        let s = a / (2.0 + a);
        let s2 = s * s;
        return 2.0 * s * (1.0 + s2 * (1.0 / 3.0 + s2 * (0.2 + s2 * (1.0 / 7.0 + s2 / 9.0))));
    }
    return log(1.0 + a);
}

// The Cauchy stress (σxx, σyy, σxy) for G = ∇ζ (rows: components of ζ, columns: ∂/∂x, ∂/∂y),
// or a NaN when the map folds.
fn cauchy(g: mat2x2<f32>) -> vec3<f32> {
    // g[c][r] = ∂ζ_r/∂x_c (WGSL matrices are column-major).
    let g00 = g[0][0];
    let g01 = g[1][0];
    let g10 = g[0][1];
    let g11 = g[1][1];
    // det(I + G) = 1 + a.
    let a = g00 + g11 + (g00 * g11 - g01 * g10);
    let dj = 1.0 + a;
    if !(dj > 0.0) {
        return vec3<f32>(bitcast<f32>(0x7fc00000u));
    }
    // F = (I + G)⁻¹.
    let f00 = (1.0 + g11) / dj;
    let f01 = -g01 / dj;
    let f10 = -g10 / dj;
    let f11 = (1.0 + g00) / dj;
    var h00: f32;
    var h01: f32;
    var h10: f32;
    var h11: f32;
    if STABLE == 1u {
        // H = F − I = −F G.
        h00 = -(f00 * g00 + f01 * g10);
        h01 = -(f00 * g01 + f01 * g11);
        h10 = -(f10 * g00 + f11 * g10);
        h11 = -(f10 * g01 + f11 * g11);
    } else {
        h00 = f00 - 1.0;
        h01 = f01;
        h10 = f10;
        h11 = f11 - 1.0;
    }
    if P.neo_hookean == 0u {
        let exx = h00;
        let eyy = h11;
        let exy = 0.5 * (h01 + h10);
        let tr = P.lambda * (exx + eyy);
        return vec3<f32>(tr + 2.0 * P.mu * exx, tr + 2.0 * P.mu * eyy, 2.0 * P.mu * exy);
    }
    var bxx: f32;
    var byy: f32;
    var bxy: f32;
    var lnj: f32;
    if STABLE == 1u {
        // B − I = H + Hᵀ + H Hᵀ; ln J = −log1p(a).
        bxx = 2.0 * h00 + h00 * h00 + h01 * h01;
        byy = 2.0 * h11 + h10 * h10 + h11 * h11;
        bxy = h01 + h10 + h00 * h10 + h01 * h11;
        lnj = -log1p_small(a);
    } else {
        bxx = f00 * f00 + f01 * f01 - 1.0;
        byy = f10 * f10 + f11 * f11 - 1.0;
        bxy = f00 * f10 + f01 * f11;
        lnj = log(1.0 / dj);
    }
    // σ = (μ/J)(B − I) + (λ ln J / J) I, 1/J = det(I + G).
    let vol = P.lambda * lnj * dj;
    let m = P.mu * dj;
    return vec3<f32>(m * bxx + vol, m * byy + vol, m * bxy);
}

// The flux through face (k, +d): the compact stencil's stress (solid and viscous) times h, the
// components a face of normal +d carries.
fn face_flux(k: u32, d: u32) -> vec2<f32> {
    var ei = 1;
    var ej = 0;
    var ti = 0;
    var tj = 1;
    if d == 1u {
        ei = 0;
        ej = 1;
        ti = 1;
        tj = 0;
    }
    let b = nb(k, ei, ej);
    let ap = nb(k, ti, tj);
    let am = nb(k, -ti, -tj);
    let bp = nb(b, ti, tj);
    let bm = nb(b, -ti, -tj);
    let sk = cur[k];
    let sb = cur[b];
    let sap = cur[ap];
    let sam = cur[am];
    let sbp = cur[bp];
    let sbm = cur[bm];
    let along = (sb - sk) / P.h;
    let across = (sap + sbp - sam - sbm) / (4.0 * P.h);
    // Columns: ∂/∂x, ∂/∂y of (ζx, ζy) and of (vx, vy).
    var gz: mat2x2<f32>;
    var gv: mat2x2<f32>;
    if d == 0u {
        gz = mat2x2<f32>(along.xy, across.xy);
        gv = mat2x2<f32>(along.zw, across.zw);
    } else {
        gz = mat2x2<f32>(across.xy, along.xy);
        gv = mat2x2<f32>(across.zw, along.zw);
    }
    let s = cauchy(gz);
    // gv[c][r] = ∂v_r/∂x_c.
    let visc = vec3<f32>(2.0 * P.viscosity * gv[0][0], 2.0 * P.viscosity * gv[1][1], P.viscosity * (gv[1][0] + gv[0][1]));
    let t = s + visc;
    if d == 0u {
        return vec2<f32>(t.x, t.z) * P.h;
    }
    return vec2<f32>(t.z, t.y) * P.h;
}

// The host's values at the nodes it steps next to the ones stepped here.
@compute @workgroup_size(256)
fn upload(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = index(gid);
    if i >= P.upload_count {
        return;
    }
    cur[upload_nodes[i]] = upload_values[i];
}

@compute @workgroup_size(256)
fn faces(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = index(gid);
    if k >= P.nx * P.ny {
        return;
    }
    let f = flags[k];
    var out = vec4<f32>(0.0);
    if (f & 2u) != 0u {
        out = vec4<f32>(face_flux(k, 0u), out.zw);
    }
    if (f & 4u) != 0u {
        out = vec4<f32>(out.xy, face_flux(k, 1u));
    }
    flux[k] = out;
}

// (v·∇)a at node k by second-order one-sided upwind differences; `comp` picks a from the state
// (0: ζ, 1: v).
fn upwind(k: u32, v: vec2<f32>, comp: u32) -> vec2<f32> {
    var sum = vec2<f32>(0.0);
    for (var d = 0u; d < 2u; d++) {
        let vd = v[d];
        if vd == 0.0 {
            continue;
        }
        var s = 1;
        if vd > 0.0 {
            s = -1;
        }
        var m1: u32;
        var m2: u32;
        if d == 0u {
            m1 = nb(k, s, 0);
            m2 = nb(k, 2 * s, 0);
        } else {
            m1 = nb(k, 0, s);
            m2 = nb(k, 0, 2 * s);
        }
        var a0: vec2<f32>;
        var a1: vec2<f32>;
        var a2: vec2<f32>;
        if comp == 0u {
            a0 = cur[k].xy;
            a1 = cur[m1].xy;
            a2 = cur[m2].xy;
        } else {
            a0 = cur[k].zw;
            a1 = cur[m1].zw;
            a2 = cur[m2].zw;
        }
        let deriv = -f32(s) * (3.0 * a0 - 4.0 * a1 + a2) / (2.0 * P.h);
        sum += vd * deriv;
    }
    return sum;
}

@compute @workgroup_size(256)
fn momentum(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = index(gid);
    if k >= P.nx * P.ny {
        return;
    }
    let s = cur[k];
    if (flags[k] & 1u) == 0u {
        next[k] = s;
        return;
    }
    // The net force on the full control volume, summed in the CPU's face order.
    let fy0 = flux[nb(k, 0, -1)].zw;
    let fx0 = flux[nb(k, -1, 0)].xy;
    let here = flux[k];
    let force = (((vec2<f32>(0.0) - fy0) - fx0) + here.xy) + here.zw + body[k] * (P.h * P.h);
    // ρ = ρ₀ det ∇ξ from central differences.
    let gx = (cur[nb(k, 1, 0)].xy - cur[nb(k, -1, 0)].xy) / (2.0 * P.h);
    let gy = (cur[nb(k, 0, 1)].xy - cur[nb(k, 0, -1)].xy) / (2.0 * P.h);
    let det = (1.0 + gx.x) * (1.0 + gy.y) - gy.x * gx.y;
    var rho = P.density;
    if det > 0.0 {
        rho = P.density * det;
    }
    let v = s.zw;
    let adv = upwind(k, v, 1u);
    let vn = (v + P.kick * (force / (rho * P.h * P.h) - adv)) / (1.0 + P.damping * P.kick);
    if !(det > 0.0) || !all(abs(vn) < vec2<f32>(3.0e38)) {
        atomicOr(&status, 1u);
    }
    next[k] = vec4<f32>(s.xy, vn);
}

@compute @workgroup_size(256)
fn advect(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = index(gid);
    if k >= P.nx * P.ny || (flags[k] & 1u) == 0u {
        return;
    }
    let v = next[k].zw;
    let a = upwind(k, v, 0u);
    let z = cur[k].xy - P.dt * (v + a);
    if !all(abs(z) < vec2<f32>(3.0e38)) {
        atomicOr(&status, 1u);
    }
    next[k] = vec4<f32>(z, v);
}

// The new values of the stepped nodes next to the host's.
@compute @workgroup_size(256)
fn download(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = index(gid);
    if i >= P.download_count {
        return;
    }
    download_values[i] = next[download_nodes[i]];
}
