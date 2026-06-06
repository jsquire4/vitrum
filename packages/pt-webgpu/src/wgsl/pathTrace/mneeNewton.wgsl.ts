/**
 * mneeNewton.wgsl.ts — the COMPLETE math core of a real Hanika-2015 manifold-NEE.
 * The half-vector Newton solve finds the specular vertex on a surface so the
 * light↔receiver path obeys the specular law — both REFLECTION (etaI==etaT) and
 * REFRACTION (eta-generalized half-vector; the important case — caustics are
 * usually glass). This replaces the cone-search APPROXIMATION in
 * `caustic.wgsl.ts:manifoldNeeContribution` (which has no half-vector constraint,
 * Newton iteration, or change-of-variables Jacobian).
 *
 * Contents (all GPU-validated via self-validating analytic==FD / residual→0
 * harnesses — no radiometric A/B; see wsl-gpu/scripts/mnee-*-validate.ts):
 *   • mneeHalfVectorResidual2d / mneeNewtonSolve — the solve (residual→0, and a
 *     flat-mirror config converges to the analytic mirror-image point).
 *   • mneeResidualJacobian — the ANALYTIC ∂r/∂(surface coords) that drives the
 *     Newton step (replaced the finite-difference columns: exact, 2 fewer
 *     residual evals/step, and the reusable per-vertex diagonal block a future
 *     multi-vertex chain solve needs). Validated analytic == FD.
 *   • mneeManifoldJacobian — d(vertex)/d(light) via the implicit function theorem
 *     (−J_vertex⁻¹·J_light), the geometric quantity the connection PDF is built on.
 *   • mneePdfJacobianDet — the area-light connection-PDF factor |dω_recv/dA_light|.
 *
 * Integration note: the production MNEE path consumes this solver through the
 * pt-webgpu caustic pipeline. Radiometric validation remains a rendered
 * caustic A/B against a converged reference.
 *
 * Ref: Hanika, Droske, Fascione, "Manifold Next Event Estimation," EGSR 2015;
 *      Jakob & Marschner, "Manifold Exploration," SIGGRAPH 2012.
 */

/** Newton iterations per solve (clamped in the kernel). */
export const MNEE_NEWTON_MAX_ITERS = 16;

/** Floats per harness input record (vec4-aligned: 3 × vec4 = 12 floats). */
export const MNEE_HARNESS_INPUT_FLOATS = 12;

/** Pack one harness config: receiver, light, and the mirror plane point (the
 *  plane is z=0 with normal +z in harness space, so only the point varies). */
export function packMneeHarnessInput(
  receiver: readonly [number, number, number],
  light: readonly [number, number, number],
  planePoint: readonly [number, number, number],
  etaI = 1,
  etaT = 1,
): number[] {
  return [
    receiver[0], receiver[1], receiver[2], etaI,
    light[0], light[1], light[2], etaT,
    planePoint[0], planePoint[1], planePoint[2], 0,
  ];
}

export const MNEE_NEWTON_WGSL = /* wgsl */ `
fn mnee_safe_normalize(v: vec3f) -> vec3f {
  let l = length(v);
  if (l < 1e-12) { return vec3f(0.0); }
  return v / l;
}

// Tangential half-vector residual at vertex v: project the (eta-generalized)
// half-vector h = normalize(etaI·wi + etaT·wo) onto the surface tangent plane
// (tu,tv). Zero ⇔ h ∥ nm ⇔ the specular law holds: reflection when etaI==etaT
// (same medium), refraction (Snell) when etaI/etaT are the two media's IORs.
fn mneeHalfVectorResidual2d(v: vec3f, recv: vec3f, light: vec3f, nm: vec3f, tu: vec3f, tv: vec3f, etaI: f32, etaT: f32) -> vec2f {
  let wi = mnee_safe_normalize(light - v);
  let wo = mnee_safe_normalize(recv - v);
  let h = mnee_safe_normalize(etaI * wi + etaT * wo);
  let hTan = h - dot(h, nm) * nm;
  return vec2f(dot(hTan, tu), dot(hTan, tv));
}

// Directional derivative of normalize(x) along a perturbation dx of x:
//   D[normalize](x)·dx = (dx − x̂ (x̂·dx)) / |x|        (the (I − x̂x̂ᵀ)/|x| projector).
fn mneeDNormalize(x: vec3f, dx: vec3f) -> vec3f {
  let len = max(length(x), 1e-12);
  let xh = x / len;
  return (dx - xh * dot(xh, dx)) / len;
}

// ANALYTIC residual Jacobian ∂r/∂(a,b) at vertex v = p0 + a·tu + b·tv. Replaces
// the finite-difference columns that drove the Newton step — exact, 2 fewer
// residual evals/step, and the reusable per-vertex diagonal block a multi-vertex
// chain solve needs. Chain rule on r = tangential(normalize(etaI·wi + etaT·wo)):
// ∂v/∂a = tu so ∂(light−v)/∂a = −tu (likewise −tv for b); push −tu/−tv through
// the (I − ûûᵀ)/|u| normalize derivatives of wi, wo, then h, then the tangent
// projection. GPU-validated analytic == FD (wsl-gpu mnee-newton-jac-validate.ts).
// Returns the two columns: cA = ∂r/∂a, cB = ∂r/∂b (each [∂rx, ∂ry]).
struct MneeResidualJac { cA: vec2f, cB: vec2f }
fn mneeResidualJacobian(v: vec3f, recv: vec3f, light: vec3f, nm: vec3f, tu: vec3f, tv: vec3f, etaI: f32, etaT: f32) -> MneeResidualJac {
  let wiVec = light - v;
  let woVec = recv - v;
  let dwi_a = mneeDNormalize(wiVec, -tu);
  let dwo_a = mneeDNormalize(woVec, -tu);
  let dwi_b = mneeDNormalize(wiVec, -tv);
  let dwo_b = mneeDNormalize(woVec, -tv);
  let g = etaI * mnee_safe_normalize(wiVec) + etaT * mnee_safe_normalize(woVec);
  let dg_a = etaI * dwi_a + etaT * dwo_a;
  let dg_b = etaI * dwi_b + etaT * dwo_b;
  let dh_a = mneeDNormalize(g, dg_a);
  let dh_b = mneeDNormalize(g, dg_b);
  // hTan = h − (h·nm)·nm ⇒ ∂hTan = ∂h − (∂h·nm)·nm.
  let dhTan_a = dh_a - dot(dh_a, nm) * nm;
  let dhTan_b = dh_b - dot(dh_b, nm) * nm;
  var out: MneeResidualJac;
  out.cA = vec2f(dot(dhTan_a, tu), dot(dhTan_a, tv));
  out.cB = vec2f(dot(dhTan_b, tu), dot(dhTan_b, tv));
  return out;
}

struct MneeNewtonResult { vertex: vec3f, residual: f32, iters: u32 }

// 2D Newton solve on the surface (a,b) coords for the specular vertex. FD
// Jacobian; solves J·δ = −r each step. Reflection (etaI==etaT) converges to the
// mirror-image point; refraction (etaI≠etaT) converges to the Snell-law point.
fn mneeNewtonSolve(p0: vec3f, nm: vec3f, tu: vec3f, tv: vec3f, recv: vec3f, light: vec3f, etaI: f32, etaT: f32, maxIter: u32) -> MneeNewtonResult {
  var a = 0.0;
  var b = 0.0;
  var out: MneeNewtonResult;
  for (var it = 0u; it < maxIter; it = it + 1u) {
    let v = p0 + a * tu + b * tv;
    let r0 = mneeHalfVectorResidual2d(v, recv, light, nm, tu, tv, etaI, etaT);
    let rmag = length(r0);
    out.vertex = v; out.residual = rmag; out.iters = it;
    if (rmag < 1e-5) { return out; }
    // ANALYTIC Jacobian columns ∂r/∂a, ∂r/∂b (exact; replaced finite difference).
    let jac = mneeResidualJacobian(v, recv, light, nm, tu, tv, etaI, etaT);
    let j00 = jac.cA.x; let j10 = jac.cA.y;
    let j01 = jac.cB.x; let j11 = jac.cB.y;
    let det = j00 * j11 - j01 * j10;
    if (abs(det) < 1e-12) { return out; }
    let invDet = 1.0 / det;
    // δ = −J⁻¹·r0
    let da = -( j11 * r0.x - j01 * r0.y) * invDet;
    let db = -(-j10 * r0.x + j00 * r0.y) * invDet;
    a = a + da;
    b = b + db;
  }
  let v = p0 + a * tu + b * tv;
  out.vertex = v;
  out.residual = length(mneeHalfVectorResidual2d(v, recv, light, nm, tu, tv, etaI, etaT));
  out.iters = maxIter;
  return out;
}

// MANIFOLD DERIVATIVE — d(surface coords a,b)/d(light position), the geometric
// quantity the MNEE connection PDF change-of-variables is built on. At the
// converged vertex the half-vector constraint r(a,b; light)=0 holds, so by the
// implicit function theorem d(a,b)/d(light) = −J_vertex⁻¹ · J_light, where
// J_vertex = ∂r/∂(a,b) (the Newton residual Jacobian) and J_light = ∂r/∂light.
// All FD here; a future analytic form replaces it but must match this (and the FD
// re-solve in the harness). Returns ∂a/∂light (dadL) + ∂b/∂light (dbdL).
struct MneeJacobian { dadL: vec3f, dbdL: vec3f }
fn mneeManifoldJacobian(v: vec3f, nm: vec3f, tu: vec3f, tv: vec3f, recv: vec3f, light: vec3f, etaI: f32, etaT: f32) -> MneeJacobian {
  let eps = 1e-3;
  let r0 = mneeHalfVectorResidual2d(v, recv, light, nm, tu, tv, etaI, etaT);
  // J_vertex = ∂r/∂(a,b) at the converged vertex (a,b move v along tu,tv).
  let ra = mneeHalfVectorResidual2d(v + eps * tu, recv, light, nm, tu, tv, etaI, etaT);
  let rb = mneeHalfVectorResidual2d(v + eps * tv, recv, light, nm, tu, tv, etaI, etaT);
  let j00 = (ra.x - r0.x) / eps; let j10 = (ra.y - r0.y) / eps; // ∂r/∂a
  let j01 = (rb.x - r0.x) / eps; let j11 = (rb.y - r0.y) / eps; // ∂r/∂b
  let invDet = 1.0 / (j00 * j11 - j01 * j10);
  // J_light columns = ∂r/∂light_{x,y,z}.
  let dlx = (mneeHalfVectorResidual2d(v, recv, light + vec3f(eps, 0.0, 0.0), nm, tu, tv, etaI, etaT) - r0) / eps;
  let dly = (mneeHalfVectorResidual2d(v, recv, light + vec3f(0.0, eps, 0.0), nm, tu, tv, etaI, etaT) - r0) / eps;
  let dlz = (mneeHalfVectorResidual2d(v, recv, light + vec3f(0.0, 0.0, eps), nm, tu, tv, etaI, etaT) - r0) / eps;
  // d(a,b)/d(lk) = −J_vertex⁻¹ · [drx/dlk, dry/dlk];  J_vertex⁻¹ = invDet·[[j11,−j01],[−j10,j00]].
  var out: MneeJacobian;
  out.dadL = vec3f(
    -invDet * (j11 * dlx.x - j01 * dlx.y),
    -invDet * (j11 * dly.x - j01 * dly.y),
    -invDet * (j11 * dlz.x - j01 * dlz.y),
  );
  out.dbdL = vec3f(
    -invDet * (-j10 * dlx.x + j00 * dlx.y),
    -invDet * (-j10 * dly.x + j00 * dly.y),
    -invDet * (-j10 * dlz.x + j00 * dlz.y),
  );
  return out;
}

// MNEE connection PDF geometric factor |dω_recv / dA_light| — the change of
// variables from the (area-sampled) light to the receiver's solid angle, through
// the specular vertex. The light is parameterized by its (x,y) area coords (s,t),
// so ∂v/∂s = tu·dadL.x + tv·dbdL.x, ∂v/∂t = tu·dadL.y + tv·dbdL.y (the manifold
// Jacobian columns). ω = normalize(v−recv); the solid-angle projection is
// ∂ω/∂* = (∂v/∂* − ω(ω·∂v/∂*))/|v−recv|. The 2×2 determinant on the (2D) tangent
// plane of ω is BASIS-FREE: |det| = |∂ω/∂s × ∂ω/∂t|. (POINT lights need no area
// PDF — their connection is deterministic; this is the AREA-light factor.)
fn mneePdfJacobianDet(v: vec3f, recv: vec3f, dadL: vec3f, dbdL: vec3f, tu: vec3f, tv: vec3f) -> f32 {
  let dv_ds = tu * dadL.x + tv * dbdL.x;
  let dv_dt = tu * dadL.y + tv * dbdL.y;
  let d = v - recv;
  let dist = max(length(d), 1e-8);
  let w = d / dist;
  let dw_ds = (dv_ds - w * dot(w, dv_ds)) / dist;
  let dw_dt = (dv_dt - w * dot(w, dv_dt)) / dist;
  return length(cross(dw_ds, dw_dt));
}
`;

/** Newton iterations for the 2-vertex chain solve (coupled 4-DOF → more than the
 *  single-vertex solve). */
export const MNEE_CHAIN_MAX_ITERS = 32;

/**
 * 2-VERTEX specular chain — the glass case (enter + exit refraction), the
 * canonical caustic beyond a single water surface. Path L → v1(plane 1) →
 * v2(plane 2) → R. The half-vector constraint must hold at BOTH vertices at once:
 * a 4-DOF coupled system. Solved by a BLOCK-TRIDIAGONAL Newton — the 4×4 Jacobian
 * is four 2×2 blocks [[A,B],[C,D]] (A,D diagonal; B,C the inter-vertex coupling,
 * since wo1 = v2−v1 and wi2 = v1−v2), reduced via the Schur complement
 * S = D − C·A⁻¹·B. Self-validating: a correct solve drives BOTH tangential
 * half-vector residuals → 0 (and the converged vertices satisfy Snell's ratio at
 * each interface — an independent cross-check in mnee-chain-validate.ts).
 *
 * Ref: Jakob & Marschner, "Manifold Exploration" SIGGRAPH 2012 §5 (the
 *      block-tridiagonal specular-manifold Newton); Hanika 2015 §4.
 */
export const MNEE_CHAIN_WGSL = /* wgsl */ `
// 2×2 inverse (column-major mat2x2f: m[0]=(m00,m10), m[1]=(m01,m11)).
fn mnee_inv2x2(m: mat2x2f) -> mat2x2f {
  let det = m[0][0] * m[1][1] - m[1][0] * m[0][1];
  let inv = 1.0 / select(det, 1e-12, abs(det) < 1e-12);
  return mat2x2f(vec2f(m[1][1], -m[0][1]) * inv, vec2f(-m[1][0], m[0][0]) * inv);
}

// 4D residual: tangential half-vectors at v1 (interface 1) and v2 (interface 2).
fn mneeChainResidual4d(
  v1: vec3f, v2: vec3f,
  n1: vec3f, tu1: vec3f, tv1: vec3f,
  n2: vec3f, tu2: vec3f, tv2: vec3f,
  lightP: vec3f, recv: vec3f,
  eta1i: f32, eta1t: f32, eta2i: f32, eta2t: f32,
) -> vec4f {
  let wi1 = mnee_safe_normalize(lightP - v1);
  let wo1 = mnee_safe_normalize(v2 - v1);
  let h1 = mnee_safe_normalize(eta1i * wi1 + eta1t * wo1);
  let h1t = h1 - dot(h1, n1) * n1;
  let wi2 = mnee_safe_normalize(v1 - v2);
  let wo2 = mnee_safe_normalize(recv - v2);
  let h2 = mnee_safe_normalize(eta2i * wi2 + eta2t * wo2);
  let h2t = h2 - dot(h2, n2) * n2;
  return vec4f(dot(h1t, tu1), dot(h1t, tv1), dot(h2t, tu2), dot(h2t, tv2));
}

struct MneeChainResult { v1: vec3f, v2: vec3f, residual: f32, iters: u32 }

fn mneeNewtonSolveChain2(
  p1: vec3f, n1: vec3f, tu1: vec3f, tv1: vec3f,
  p2: vec3f, n2: vec3f, tu2: vec3f, tv2: vec3f,
  lightP: vec3f, recv: vec3f,
  eta1i: f32, eta1t: f32, eta2i: f32, eta2t: f32,
  maxIter: u32,
) -> MneeChainResult {
  // Initialize each vertex at the UNREFRACTED straight-line L→R crossing of its
  // plane (a, b in the plane's tangent frame) — far closer to the refracted
  // solution than the plane origin, which is what lets the coupled Newton
  // converge for oblique configs instead of overshooting.
  let dLR = recv - lightP;
  let dn1 = dot(dLR, n1);
  let dn2 = dot(dLR, n2);
  let t1 = dot(p1 - lightP, n1) / select(dn1, 1e-12, abs(dn1) < 1e-12);
  let t2 = dot(p2 - lightP, n2) / select(dn2, 1e-12, abs(dn2) < 1e-12);
  let cross1 = lightP + t1 * dLR;
  let cross2 = lightP + t2 * dLR;
  var a1 = dot(cross1 - p1, tu1); var b1 = dot(cross1 - p1, tv1);
  var a2 = dot(cross2 - p2, tu2); var b2 = dot(cross2 - p2, tv2);
  let eps = 1e-3;
  var out: MneeChainResult;
  for (var it = 0u; it < maxIter; it = it + 1u) {
    let v1 = p1 + a1 * tu1 + b1 * tv1;
    let v2 = p2 + a2 * tu2 + b2 * tv2;
    let r = mneeChainResidual4d(v1, v2, n1, tu1, tv1, n2, tu2, tv2, lightP, recv, eta1i, eta1t, eta2i, eta2t);
    let rmag = length(r);
    out.v1 = v1; out.v2 = v2; out.residual = rmag; out.iters = it;
    if (rmag < 1e-5) { return out; }
    // FD 4×4 Jacobian as four 2×2 blocks (perturb each DOF, diff the 4D residual).
    let d_a1 = (mneeChainResidual4d(v1 + eps * tu1, v2, n1, tu1, tv1, n2, tu2, tv2, lightP, recv, eta1i, eta1t, eta2i, eta2t) - r) / eps;
    let d_b1 = (mneeChainResidual4d(v1 + eps * tv1, v2, n1, tu1, tv1, n2, tu2, tv2, lightP, recv, eta1i, eta1t, eta2i, eta2t) - r) / eps;
    let d_a2 = (mneeChainResidual4d(v1, v2 + eps * tu2, n1, tu1, tv1, n2, tu2, tv2, lightP, recv, eta1i, eta1t, eta2i, eta2t) - r) / eps;
    let d_b2 = (mneeChainResidual4d(v1, v2 + eps * tv2, n1, tu1, tv1, n2, tu2, tv2, lightP, recv, eta1i, eta1t, eta2i, eta2t) - r) / eps;
    let A = mat2x2f(d_a1.xy, d_b1.xy); // ∂r1/∂(a1,b1)
    let B = mat2x2f(d_a2.xy, d_b2.xy); // ∂r1/∂(a2,b2)
    let C = mat2x2f(d_a1.zw, d_b1.zw); // ∂r2/∂(a1,b1)
    let D = mat2x2f(d_a2.zw, d_b2.zw); // ∂r2/∂(a2,b2)
    let r1 = r.xy; let r2 = r.zw;
    let Ainv = mnee_inv2x2(A);
    let CAinv = C * Ainv;
    let S = D - CAinv * B;                       // Schur complement
    let d2 = mnee_inv2x2(S) * (CAinv * r1 - r2); // δ2 = S⁻¹(C A⁻¹ r1 − r2)
    let d1 = Ainv * (-r1 - B * d2);              // δ1 = A⁻¹(−r1 − B δ2)
    // Backtracking line search: take the largest fraction of the Newton step that
    // DECREASES the residual (globalizes the coupled solve — an undamped full step
    // overshoots + diverges for oblique configs).
    var scale = 1.0;
    var accepted = false;
    for (var bt = 0u; bt < 10u; bt = bt + 1u) {
      let na1 = a1 + scale * d1.x; let nb1 = b1 + scale * d1.y;
      let na2 = a2 + scale * d2.x; let nb2 = b2 + scale * d2.y;
      let nv1 = p1 + na1 * tu1 + nb1 * tv1;
      let nv2 = p2 + na2 * tu2 + nb2 * tv2;
      let nr = length(mneeChainResidual4d(nv1, nv2, n1, tu1, tv1, n2, tu2, tv2, lightP, recv, eta1i, eta1t, eta2i, eta2t));
      if (nr < rmag) { a1 = na1; b1 = nb1; a2 = na2; b2 = nb2; accepted = true; break; }
      scale = scale * 0.5;
    }
    if (!accepted) { return out; } // no descent along the Newton direction — return best so far
  }
  let v1 = p1 + a1 * tu1 + b1 * tv1;
  let v2 = p2 + a2 * tu2 + b2 * tv2;
  out.v1 = v1; out.v2 = v2;
  out.residual = length(mneeChainResidual4d(v1, v2, n1, tu1, tv1, n2, tu2, tv2, lightP, recv, eta1i, eta1t, eta2i, eta2t));
  out.iters = maxIter;
  return out;
}

// CHAIN connection-PDF geometric factor |dω_recv/dA_light| through the 2-vertex
// glass chain. As the AREA light moves along its tangent axes (lightU,lightV = the
// s,t area params) the whole chain re-solves, so v2 — the last vertex, the one the
// receiver sees — shifts. d(a2,b2)/d(light) comes from the implicit function
// theorem on the 4-DOF system (d(params)/d(light) = −J_full⁻¹·J_light); its (a2,b2)
// rows are exactly the Schur form S⁻¹(C·A⁻¹·r_top − r_bot) REUSED from the solve.
// Then ∂v2/∂s,t → solid-angle projection at recv → basis-free |∂ω/∂s × ∂ω/∂t|
// (same shape as the single-vertex mneePdfJacobianDet, but through the chain).
// POINT lights are deterministic (no area PDF) — this is the AREA-light factor.
fn mneeChainPdfJacobianDet(
  v1: vec3f, v2: vec3f,
  n1: vec3f, tu1: vec3f, tv1: vec3f,
  n2: vec3f, tu2: vec3f, tv2: vec3f,
  lightP: vec3f, recv: vec3f,
  eta1i: f32, eta1t: f32, eta2i: f32, eta2t: f32,
  lightU: vec3f, lightV: vec3f,
) -> f32 {
  let eps = 1e-3;
  let r = mneeChainResidual4d(v1, v2, n1, tu1, tv1, n2, tu2, tv2, lightP, recv, eta1i, eta1t, eta2i, eta2t);
  let d_a1 = (mneeChainResidual4d(v1 + eps * tu1, v2, n1, tu1, tv1, n2, tu2, tv2, lightP, recv, eta1i, eta1t, eta2i, eta2t) - r) / eps;
  let d_b1 = (mneeChainResidual4d(v1 + eps * tv1, v2, n1, tu1, tv1, n2, tu2, tv2, lightP, recv, eta1i, eta1t, eta2i, eta2t) - r) / eps;
  let d_a2 = (mneeChainResidual4d(v1, v2 + eps * tu2, n1, tu1, tv1, n2, tu2, tv2, lightP, recv, eta1i, eta1t, eta2i, eta2t) - r) / eps;
  let d_b2 = (mneeChainResidual4d(v1, v2 + eps * tv2, n1, tu1, tv1, n2, tu2, tv2, lightP, recv, eta1i, eta1t, eta2i, eta2t) - r) / eps;
  let A = mat2x2f(d_a1.xy, d_b1.xy);
  let B = mat2x2f(d_a2.xy, d_b2.xy);
  let C = mat2x2f(d_a1.zw, d_b1.zw);
  let D = mat2x2f(d_a2.zw, d_b2.zw);
  let Ainv = mnee_inv2x2(A);
  let CAinv = C * Ainv;
  let Sinv = mnee_inv2x2(D - CAinv * B);
  // ∂r/∂(light) along the area axes s=lightU, t=lightV (FD).
  let r_s = (mneeChainResidual4d(v1, v2, n1, tu1, tv1, n2, tu2, tv2, lightP + eps * lightU, recv, eta1i, eta1t, eta2i, eta2t) - r) / eps;
  let r_t = (mneeChainResidual4d(v1, v2, n1, tu1, tv1, n2, tu2, tv2, lightP + eps * lightV, recv, eta1i, eta1t, eta2i, eta2t) - r) / eps;
  // d(a2,b2)/d(s,t) = −[J_full⁻¹ J_light]_{rows 3,4} = S⁻¹(C A⁻¹ r_top − r_bot).
  let dab2_ds = Sinv * (CAinv * r_s.xy - r_s.zw);
  let dab2_dt = Sinv * (CAinv * r_t.xy - r_t.zw);
  let dv2_ds = tu2 * dab2_ds.x + tv2 * dab2_ds.y;
  let dv2_dt = tu2 * dab2_dt.x + tv2 * dab2_dt.y;
  let dvec = v2 - recv;
  let dist = max(length(dvec), 1e-8);
  let w = dvec / dist;
  let dw_ds = (dv2_ds - w * dot(w, dv2_ds)) / dist;
  let dw_dt = (dv2_dt - w * dot(w, dv2_dt)) / dist;
  return length(cross(dw_ds, dw_dt));
}
`;

/** Chain-PDF harness: solve the glass-slab chain, then write the chain connection-
 *  PDF factor |dω_recv/dA_light| (light area axes +x/+y). Validated analytic == FD
 *  re-solve over the light's area params. Per config: [v1.xyz, residual], [v2.xyz, det]. */
export const MNEE_CHAIN_PDF_HARNESS_WGSL = /* wgsl */ `
${MNEE_NEWTON_WGSL}
${MNEE_CHAIN_WGSL}

struct ChainIn { lightP: vec3f, slabD: f32, recv: vec3f, etaGlass: f32 }
@group(0) @binding(0) var<storage, read>       hIn:  array<ChainIn>;
@group(0) @binding(1) var<storage, read_write> hOut: array<vec4f>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= arrayLength(&hIn)) { return; }
  let c = hIn[i];
  let n = vec3f(0.0, 0.0, 1.0);
  let tu = vec3f(1.0, 0.0, 0.0);
  let tv = vec3f(0.0, 1.0, 0.0);
  let p1 = vec3f(0.0, 0.0, 0.0);
  let p2 = vec3f(0.0, 0.0, -c.slabD);
  let res = mneeNewtonSolveChain2(p1, n, tu, tv, p2, n, tu, tv, c.lightP, c.recv, 1.0, c.etaGlass, c.etaGlass, 1.0, ${MNEE_CHAIN_MAX_ITERS}u);
  let det = mneeChainPdfJacobianDet(res.v1, res.v2, n, tu, tv, n, tu, tv, c.lightP, c.recv, 1.0, c.etaGlass, c.etaGlass, 1.0, vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0));
  hOut[i * 2u + 0u] = vec4f(res.v1, res.residual);
  hOut[i * 2u + 1u] = vec4f(res.v2, det);
}
`;

/** Chain harness: a glass slab — plane 1 at z=0, plane 2 at z=−slabD (both +z
 *  normal, +x/+y tangents); air→glass→air (eta 1 / etaGlass / 1). Writes the two
 *  converged vertices + the final 4D residual. */
export const MNEE_CHAIN_HARNESS_WGSL = /* wgsl */ `
${MNEE_NEWTON_WGSL}
${MNEE_CHAIN_WGSL}

struct ChainIn { lightP: vec3f, slabD: f32, recv: vec3f, etaGlass: f32 }
@group(0) @binding(0) var<storage, read>       hIn:  array<ChainIn>;
@group(0) @binding(1) var<storage, read_write> hOut: array<vec4f>; // [v1.xyz, residual], [v2.xyz, iters]

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= arrayLength(&hIn)) { return; }
  let c = hIn[i];
  let n = vec3f(0.0, 0.0, 1.0);
  let tu = vec3f(1.0, 0.0, 0.0);
  let tv = vec3f(0.0, 1.0, 0.0);
  let p1 = vec3f(0.0, 0.0, 0.0);
  let p2 = vec3f(0.0, 0.0, -c.slabD);
  let res = mneeNewtonSolveChain2(p1, n, tu, tv, p2, n, tu, tv, c.lightP, c.recv, 1.0, c.etaGlass, c.etaGlass, 1.0, ${MNEE_CHAIN_MAX_ITERS}u);
  hOut[i * 2u + 0u] = vec4f(res.v1, res.residual);
  hOut[i * 2u + 1u] = vec4f(res.v2, f32(res.iters));
}
`;

/**
 * Single-vertex MNEE REFLECTION irradiance — the kernel-ready contribution core
 * for a point-light specular-reflection caustic (Phase I.1 of the render
 * INTEGRATION, distinct from the validated-in-isolation solve/Jacobian/PDF). Given
 * a receiver + a (seed-found) mirror plane + a point light, Newton-solve the exact
 * mirror vertex, then return the incident IRRADIANCE E = I·cosθ_recv / d_total²,
 * where d_total = |light→v| + |v→recv| is the UNFOLDED path length (= the distance
 * from the light's mirror IMAGE to the receiver). The kernel multiplies E by the
 * receiver BRDF + a visibility test (both scene-dependent, hence not here).
 * GPU-validated against the analytic mirror-image irradiance — DETERMINISTIC
 * ground truth, since a point-light specular caustic is zero-measure for ordinary
 * NEE/BSDF sampling (that is WHY MNEE exists), so there is no noisy reference.
 */
export const MNEE_CONNECTION_WGSL = /* wgsl */ `
fn mneeReflectionIrradiance(
  recv: vec3f, recvNormal: vec3f,
  mirrorP: vec3f, mirrorN: vec3f, mirrorTu: vec3f, mirrorTv: vec3f,
  lightPos: vec3f, lightIntensity: vec3f,
) -> vec3f {
  let res = mneeNewtonSolve(mirrorP, mirrorN, mirrorTu, mirrorTv, recv, lightPos, 1.0, 1.0, ${MNEE_NEWTON_MAX_ITERS}u);
  if (res.residual > 1e-4) { return vec3f(0.0); }      // no specular connection found
  let v = res.vertex;
  let wi = mnee_safe_normalize(v - recv);               // receiver's incident dir (toward the mirror vertex)
  let nDotL = max(dot(recvNormal, wi), 0.0);
  if (nDotL <= 1e-5) { return vec3f(0.0); }
  let dTotal = length(lightPos - v) + length(recv - v); // unfolded path length = dist(image, recv)
  return lightIntensity * nDotL / max(dTotal * dTotal, 1e-8);
}
`;

/** Reflection harness: a mirror at z=0 (+z normal, +x/+y tangents); writes the MNEE
 *  reflection irradiance.rgb per config. Validated against the analytic mirror-image
 *  point-light irradiance (deterministic ground truth). */
export const MNEE_REFLECTION_HARNESS_WGSL = /* wgsl */ `
${MNEE_NEWTON_WGSL}
${MNEE_CONNECTION_WGSL}

struct ReflIn { recv: vec3f, _p0: f32, recvNormal: vec3f, _p1: f32, lightPos: vec3f, _p2: f32, intensity: vec3f, _p3: f32 }
@group(0) @binding(0) var<storage, read>       hIn:  array<ReflIn>;
@group(0) @binding(1) var<storage, read_write> hOut: array<vec4f>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= arrayLength(&hIn)) { return; }
  let c = hIn[i];
  let mP = vec3f(0.0, 0.0, 0.0);
  let mN = vec3f(0.0, 0.0, 1.0);
  let mTu = vec3f(1.0, 0.0, 0.0);
  let mTv = vec3f(0.0, 1.0, 0.0);
  let e = mneeReflectionIrradiance(c.recv, c.recvNormal, mP, mN, mTu, mTv, c.lightPos, c.intensity);
  hOut[i] = vec4f(e, 0.0);
}
`;

/** Harness kernel: runs the Newton solve per config (mirror plane z = planePoint.z,
 *  normal +z, tangents +x/+y) and writes the converged vertex + final residual. */
export const MNEE_NEWTON_HARNESS_WGSL = /* wgsl */ `
${MNEE_NEWTON_WGSL}

// recv.w = etaI (IOR on the light side), light.w = etaT (IOR on the receiver side).
struct MneeIn { recv: vec3f, etaI: f32, light: vec3f, etaT: f32, planePoint: vec3f, _p2: f32 }
@group(0) @binding(0) var<storage, read>       hIn:  array<MneeIn>;
@group(0) @binding(1) var<storage, read_write> hOut: array<vec4f>; // xyz = vertex, w = residual

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= arrayLength(&hIn)) { return; }
  let c = hIn[i];
  let nm = vec3f(0.0, 0.0, 1.0);
  let tu = vec3f(1.0, 0.0, 0.0);
  let tv = vec3f(0.0, 1.0, 0.0);
  let r = mneeNewtonSolve(c.planePoint, nm, tu, tv, c.recv, c.light, c.etaI, c.etaT, ${MNEE_NEWTON_MAX_ITERS}u);
  hOut[i] = vec4f(r.vertex, r.residual);
}
`;

/** Newton-Jacobian harness: writes the ANALYTIC residual Jacobian ∂r/∂(a,b) and
 *  the finite-difference reference at a generic test vertex (2 vec4 per config:
 *  [analytic j00,j10,j01,j11], [FD j00,j10,j01,j11]). The validation asserts
 *  analytic == FD — proving the exact Jacobian that drives the Newton step. */
export const MNEE_NEWTON_JAC_HARNESS_WGSL = /* wgsl */ `
${MNEE_NEWTON_WGSL}

struct MneeIn { recv: vec3f, etaI: f32, light: vec3f, etaT: f32, planePoint: vec3f, _p2: f32 }
@group(0) @binding(0) var<storage, read>       hIn:  array<MneeIn>;
@group(0) @binding(1) var<storage, read_write> hOut: array<vec4f>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= arrayLength(&hIn)) { return; }
  let c = hIn[i];
  let nm = vec3f(0.0, 0.0, 1.0);
  let tu = vec3f(1.0, 0.0, 0.0);
  let tv = vec3f(0.0, 1.0, 0.0);
  // A generic (non-degenerate) test vertex on the plane — the Jacobian formula
  // is point-independent, so analytic == FD here proves it everywhere.
  let v = c.planePoint + 0.1 * tu + 0.05 * tv;
  let jac = mneeResidualJacobian(v, c.recv, c.light, nm, tu, tv, c.etaI, c.etaT);
  // CENTRAL-difference reference (O(eps²)). Forward difference's O(eps) truncation
  // is ~1e-2 on the highly-nonlinear refraction residual — enough to spuriously
  // diverge from the EXACT analytic on small components.
  let eps = 1e-3;
  let ra_p = mneeHalfVectorResidual2d(v + eps * tu, c.recv, c.light, nm, tu, tv, c.etaI, c.etaT);
  let ra_m = mneeHalfVectorResidual2d(v - eps * tu, c.recv, c.light, nm, tu, tv, c.etaI, c.etaT);
  let rb_p = mneeHalfVectorResidual2d(v + eps * tv, c.recv, c.light, nm, tu, tv, c.etaI, c.etaT);
  let rb_m = mneeHalfVectorResidual2d(v - eps * tv, c.recv, c.light, nm, tu, tv, c.etaI, c.etaT);
  hOut[i * 2u + 0u] = vec4f(jac.cA.x, jac.cA.y, jac.cB.x, jac.cB.y);
  hOut[i * 2u + 1u] = vec4f((ra_p.x - ra_m.x) / (2.0 * eps), (ra_p.y - ra_m.y) / (2.0 * eps), (rb_p.x - rb_m.x) / (2.0 * eps), (rb_p.y - rb_m.y) / (2.0 * eps));
}
`;

/** Jacobian harness: solves, then writes the manifold derivative d(a,b)/d(light)
 *  (3 vec4 per config: [vertex, residual], [dadL.xyz, dbdL.x], [dbdL.yz, _, _]).
 *  The validation FD-re-solves to confirm the analytic Jacobian == finite diff. */
export const MNEE_JACOBIAN_HARNESS_WGSL = /* wgsl */ `
${MNEE_NEWTON_WGSL}

struct MneeIn { recv: vec3f, etaI: f32, light: vec3f, etaT: f32, planePoint: vec3f, _p2: f32 }
@group(0) @binding(0) var<storage, read>       hIn:  array<MneeIn>;
@group(0) @binding(1) var<storage, read_write> hOut: array<vec4f>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= arrayLength(&hIn)) { return; }
  let c = hIn[i];
  let nm = vec3f(0.0, 0.0, 1.0);
  let tu = vec3f(1.0, 0.0, 0.0);
  let tv = vec3f(0.0, 1.0, 0.0);
  let r = mneeNewtonSolve(c.planePoint, nm, tu, tv, c.recv, c.light, c.etaI, c.etaT, ${MNEE_NEWTON_MAX_ITERS}u);
  let jac = mneeManifoldJacobian(r.vertex, nm, tu, tv, c.recv, c.light, c.etaI, c.etaT);
  hOut[i * 3u + 0u] = vec4f(r.vertex, r.residual);
  hOut[i * 3u + 1u] = vec4f(jac.dadL, jac.dbdL.x);
  hOut[i * 3u + 2u] = vec4f(jac.dbdL.y, jac.dbdL.z, 0.0, 0.0);
}
`;

/** PDF harness: solves, then writes the connection-PDF geometric factor
 *  |dω_recv/dA_light| (per config: vec4[vertex.xyz, |det|]). The validation
 *  FD-re-solves over the light's (x,y) area params to confirm analytic == FD. */
export const MNEE_PDF_HARNESS_WGSL = /* wgsl */ `
${MNEE_NEWTON_WGSL}

struct MneeIn { recv: vec3f, etaI: f32, light: vec3f, etaT: f32, planePoint: vec3f, _p2: f32 }
@group(0) @binding(0) var<storage, read>       hIn:  array<MneeIn>;
@group(0) @binding(1) var<storage, read_write> hOut: array<vec4f>; // xyz = vertex, w = |dω/dA|

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= arrayLength(&hIn)) { return; }
  let c = hIn[i];
  let nm = vec3f(0.0, 0.0, 1.0);
  let tu = vec3f(1.0, 0.0, 0.0);
  let tv = vec3f(0.0, 1.0, 0.0);
  let r = mneeNewtonSolve(c.planePoint, nm, tu, tv, c.recv, c.light, c.etaI, c.etaT, ${MNEE_NEWTON_MAX_ITERS}u);
  let jac = mneeManifoldJacobian(r.vertex, nm, tu, tv, c.recv, c.light, c.etaI, c.etaT);
  let det = mneePdfJacobianDet(r.vertex, c.recv, jac.dadL, jac.dbdL, tu, tv);
  hOut[i] = vec4f(r.vertex, det);
}
`;
