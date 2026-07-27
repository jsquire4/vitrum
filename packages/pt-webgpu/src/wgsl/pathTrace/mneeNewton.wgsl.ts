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

export const MNEE_NEWTON_WGSL = /* wgsl */ `
const MNEE_F32_EPSILON = 1.1920928955078125e-7;
const MNEE_SQRT_F32_EPSILON = 3.4526698300124393e-4;
const MNEE_MAX_RELATIVE_LENGTH_FLOOR = 0.01;

fn mneePointCoordinateScale(point: vec3f) -> f32 {
  return max(abs(point.x), max(abs(point.y), abs(point.z)));
}

// x = maximum absolute coordinate, y = maximum local edge/span.
fn mneeLocalScales3(a: vec3f, b: vec3f, c: vec3f) -> vec2f {
  return vec2f(
    max(mneePointCoordinateScale(a),
      max(mneePointCoordinateScale(b), mneePointCoordinateScale(c))),
    max(length(a - b), max(length(a - c), length(b - c))),
  );
}

fn mneeLengthFloorFromScales(scales: vec2f) -> f32 {
  return max(
    max(params.triIntersectEpsilon, 0.0),
    max(
      scales.x * (4.0 * MNEE_F32_EPSILON),
      max(scales.y * (4.0 * MNEE_F32_EPSILON), bitcast<f32>(0x00800000u)),
    ),
  );
}

fn mneeFdStepFromScales(scales: vec2f) -> f32 {
  return max(
    mneeLengthFloorFromScales(scales),
    scales.y * MNEE_SQRT_F32_EPSILON,
  );
}

// Half-vector residuals are dimensionless. Convert the local representable
// length floor to an angular floor instead of imposing a metre-scale constant.
fn mneeResidualToleranceFromScales(scales: vec2f) -> f32 {
  let lengthFloor = mneeLengthFloorFromScales(scales);
  return max(
    16.0 * MNEE_F32_EPSILON,
    min(
      lengthFloor / max(scales.y, lengthFloor),
      MNEE_MAX_RELATIVE_LENGTH_FLOOR,
    ),
  );
}

fn mneeScalesRepresentable(scales: vec2f) -> bool {
  let lengthFloor = mneeLengthFloorFromScales(scales);
  return scales.x == scales.x && scales.y == scales.y &&
    abs(scales.x) < INFINITY && abs(scales.y) < INFINITY &&
    scales.y > bitcast<f32>(0x00800000u) &&
    lengthFloor < scales.y * MNEE_MAX_RELATIVE_LENGTH_FLOOR;
}

fn mneeSafeSignedDenominator(value: f32, magnitudeFloor: f32) -> f32 {
  let fallback = select(-magnitudeFloor, magnitudeFloor, value >= 0.0);
  return select(fallback, value, abs(value) > magnitudeFloor);
}

fn mnee_safe_normalize(v: vec3f) -> vec3f {
  let l = length(v);
  if (l < bitcast<f32>(0x00800000u)) { return vec3f(0.0); }
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
  let len = max(length(x), bitcast<f32>(0x00800000u));
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
    let solverScales = mneeLocalScales3(v, recv, light);
    if (!mneeScalesRepresentable(solverScales)) { return out; }
    if (rmag < mneeResidualToleranceFromScales(solverScales)) {
      return out;
    }
    // ANALYTIC Jacobian columns ∂r/∂a, ∂r/∂b (exact; replaced finite difference).
    let jac = mneeResidualJacobian(v, recv, light, nm, tu, tv, etaI, etaT);
    let j00 = jac.cA.x; let j10 = jac.cA.y;
    let j01 = jac.cB.x; let j11 = jac.cB.y;
    let det = j00 * j11 - j01 * j10;
    let determinantScale = max(
      length(jac.cA) * length(jac.cB),
      bitcast<f32>(0x00800000u),
    );
    let determinantFloor = determinantScale * (32.0 * MNEE_F32_EPSILON);
    if (!(abs(det) > determinantFloor) || !(abs(det) < INFINITY)) {
      return out;
    }
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
  let solverScales = mneeLocalScales3(v, recv, light);
  var out: MneeJacobian;
  out.dadL = vec3f(0.0);
  out.dbdL = vec3f(0.0);
  if (!mneeScalesRepresentable(solverScales)) { return out; }
  let eps = mneeFdStepFromScales(solverScales);
  let r0 = mneeHalfVectorResidual2d(v, recv, light, nm, tu, tv, etaI, etaT);
  // J_vertex = ∂r/∂(a,b) at the converged vertex (a,b move v along tu,tv).
  let ra = mneeHalfVectorResidual2d(v + eps * tu, recv, light, nm, tu, tv, etaI, etaT);
  let rb = mneeHalfVectorResidual2d(v + eps * tv, recv, light, nm, tu, tv, etaI, etaT);
  let j00 = (ra.x - r0.x) / eps; let j10 = (ra.y - r0.y) / eps; // ∂r/∂a
  let j01 = (rb.x - r0.x) / eps; let j11 = (rb.y - r0.y) / eps; // ∂r/∂b
  let determinant = j00 * j11 - j01 * j10;
  let determinantFloor = max(
    length(vec2f(j00, j10)) * length(vec2f(j01, j11)) *
      (32.0 * MNEE_F32_EPSILON),
    bitcast<f32>(0x00800000u),
  );
  if (!(abs(determinant) > determinantFloor) ||
      !(abs(determinant) < INFINITY)) {
    return out;
  }
  let invDet = 1.0 / determinant;
  // J_light columns = ∂r/∂light_{x,y,z}.
  let dlx = (mneeHalfVectorResidual2d(v, recv, light + vec3f(eps, 0.0, 0.0), nm, tu, tv, etaI, etaT) - r0) / eps;
  let dly = (mneeHalfVectorResidual2d(v, recv, light + vec3f(0.0, eps, 0.0), nm, tu, tv, etaI, etaT) - r0) / eps;
  let dlz = (mneeHalfVectorResidual2d(v, recv, light + vec3f(0.0, 0.0, eps), nm, tu, tv, etaI, etaT) - r0) / eps;
  // d(a,b)/d(lk) = −J_vertex⁻¹ · [drx/dlk, dry/dlk];  J_vertex⁻¹ = invDet·[[j11,−j01],[−j10,j00]].
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
  let solverScales = mneeLocalScales3(v, recv, v);
  if (!mneeScalesRepresentable(solverScales)) { return 0.0; }
  let dv_ds = tu * dadL.x + tv * dbdL.x;
  let dv_dt = tu * dadL.y + tv * dbdL.y;
  let d = v - recv;
  let dist = length(d);
  if (dist <= mneeLengthFloorFromScales(solverScales)) {
    return 0.0;
  }
  let w = d / dist;
  let dw_ds = (dv_ds - w * dot(w, dv_ds)) / dist;
  let dw_dt = (dv_dt - w * dot(w, dv_dt)) / dist;
  let determinant = length(cross(dw_ds, dw_dt));
  if (!(determinant >= 0.0) || !(determinant < INFINITY)) { return 0.0; }
  return determinant;
}

// Same area-light connection-PDF factor, but for an arbitrary finite-emitter
// differential basis. lightU/lightV are the world-space emitter tangent axes for
// local coordinates (s,t), so dA = |lightU×lightV| ds dt. The IFT rows dadL/dbdL
// are derivatives w.r.t. world-space light motion; chain them through lightU/V,
// project the resulting vertex motion to receiver solid angle, then normalize by
// the emitter area scale. This is what production rect/disc/mesh MNEE needs:
// unlike the historical mneePdfJacobianDet x/y harness helper, it does not assume
// the area light lives in the world XY plane.
fn mneePdfJacobianDetAxes(
  v: vec3f,
  recv: vec3f,
  dadL: vec3f,
  dbdL: vec3f,
  tu: vec3f,
  tv: vec3f,
  lightU: vec3f,
  lightV: vec3f,
) -> f32 {
  let solverScales = mneeLocalScales3(v, recv, v);
  if (!mneeScalesRepresentable(solverScales)) { return 0.0; }
  let areaScale = length(cross(lightU, lightV));
  if (!(areaScale > bitcast<f32>(0x00800000u)) ||
      !(areaScale < INFINITY)) { return 0.0; }
  let da_ds = dot(dadL, lightU);
  let db_ds = dot(dbdL, lightU);
  let da_dt = dot(dadL, lightV);
  let db_dt = dot(dbdL, lightV);
  let dv_ds = tu * da_ds + tv * db_ds;
  let dv_dt = tu * da_dt + tv * db_dt;
  let d = v - recv;
  let dist = length(d);
  if (dist <= mneeLengthFloorFromScales(solverScales)) {
    return 0.0;
  }
  let w = d / dist;
  let dw_ds = (dv_ds - w * dot(w, dv_ds)) / dist;
  let dw_dt = (dv_dt - w * dot(w, dv_dt)) / dist;
  let determinant = length(cross(dw_ds, dw_dt)) / areaScale;
  if (!(determinant >= 0.0) || !(determinant < INFINITY)) { return 0.0; }
  return determinant;
}
`;

/** Newton iterations for the 2-vertex chain solve (coupled 4-DOF → more than the
 *  single-vertex solve). */
export const MNEE_CHAIN_MAX_ITERS = 32;
/** Publicly supported upper bound for `causticOptions.mneeMaxChainLength`. */
export const MNEE_CHAIN_MAX_VERTICES = 8;

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
fn mneeMat2Invertible(m: mat2x2f) -> bool {
  let determinant = m[0][0] * m[1][1] - m[1][0] * m[0][1];
  let determinantFloor = max(
    length(m[0]) * length(m[1]) * (32.0 * MNEE_F32_EPSILON),
    bitcast<f32>(0x00800000u),
  );
  return abs(determinant) > determinantFloor &&
    abs(determinant) < INFINITY &&
    determinantFloor < INFINITY;
}

// 2×2 inverse (column-major mat2x2f: m[0]=(m00,m10), m[1]=(m01,m11)).
fn mnee_inv2x2(m: mat2x2f) -> mat2x2f {
  let det = m[0][0] * m[1][1] - m[1][0] * m[0][1];
  let detFloor = max(
    length(m[0]) * length(m[1]) * (32.0 * MNEE_F32_EPSILON),
    bitcast<f32>(0x00800000u),
  );
  let inv = 1.0 / mneeSafeSignedDenominator(det, detFloor);
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

// D9.4 — FD 4×4 Jacobian as four 2×2 blocks (perturb each DOF, finite-diff the 4D
// residual). Returns the full 4×4 matrix stored column-major: columns are
// d_a1, d_b1, d_a2, d_b2 (each a vec4f). Recover blocks:
//   A = mat2x2f(J[0].xy, J[1].xy)   B = mat2x2f(J[2].xy, J[3].xy)
//   C = mat2x2f(J[0].zw, J[1].zw)   D = mat2x2f(J[2].zw, J[3].zw)
fn mneeChainFdJacobian4x4(
  v1: vec3f, v2: vec3f,
  n1: vec3f, tu1: vec3f, tv1: vec3f,
  n2: vec3f, tu2: vec3f, tv2: vec3f,
  lightP: vec3f, recv: vec3f,
  eta1i: f32, eta1t: f32, eta2i: f32, eta2t: f32,
  r: vec4f, eps: f32,
) -> mat4x4f {
  let d_a1 = (mneeChainResidual4d(v1 + eps * tu1, v2, n1, tu1, tv1, n2, tu2, tv2, lightP, recv, eta1i, eta1t, eta2i, eta2t) - r) / eps;
  let d_b1 = (mneeChainResidual4d(v1 + eps * tv1, v2, n1, tu1, tv1, n2, tu2, tv2, lightP, recv, eta1i, eta1t, eta2i, eta2t) - r) / eps;
  let d_a2 = (mneeChainResidual4d(v1, v2 + eps * tu2, n1, tu1, tv1, n2, tu2, tv2, lightP, recv, eta1i, eta1t, eta2i, eta2t) - r) / eps;
  let d_b2 = (mneeChainResidual4d(v1, v2 + eps * tv2, n1, tu1, tv1, n2, tu2, tv2, lightP, recv, eta1i, eta1t, eta2i, eta2t) - r) / eps;
  return mat4x4f(d_a1, d_b1, d_a2, d_b2);
}

struct MneeChainResult { v1: vec3f, v2: vec3f, residual: f32, iters: u32 }

fn mneeNewtonSolveChain2(
  p1: vec3f, n1: vec3f, tu1: vec3f, tv1: vec3f,
  p2: vec3f, n2: vec3f, tu2: vec3f, tv2: vec3f,
  lightP: vec3f, recv: vec3f,
  eta1i: f32, eta1t: f32, eta2i: f32, eta2t: f32,
  maxIter: u32,
) -> MneeChainResult {
  var out: MneeChainResult;
  out.v1 = p1;
  out.v2 = p2;
  out.residual = INFINITY;
  out.iters = 0u;
  // Initialize each vertex at the UNREFRACTED straight-line L→R crossing of its
  // plane (a, b in the plane's tangent frame) — far closer to the refracted
  // solution than the plane origin, which is what lets the coupled Newton
  // converge for oblique configs instead of overshooting.
  let dLR = recv - lightP;
  let dn1 = dot(dLR, n1);
  let dn2 = dot(dLR, n2);
  let chainScales = max(
    mneeLocalScales3(p1, p2, lightP),
    mneeLocalScales3(p1, p2, recv),
  );
  if (!mneeScalesRepresentable(chainScales)) { return out; }
  let denominatorFloor = mneeLengthFloorFromScales(chainScales);
  let t1 = dot(p1 - lightP, n1) /
    mneeSafeSignedDenominator(dn1, denominatorFloor);
  let t2 = dot(p2 - lightP, n2) /
    mneeSafeSignedDenominator(dn2, denominatorFloor);
  let cross1 = lightP + t1 * dLR;
  let cross2 = lightP + t2 * dLR;
  var a1 = dot(cross1 - p1, tu1); var b1 = dot(cross1 - p1, tv1);
  var a2 = dot(cross2 - p2, tu2); var b2 = dot(cross2 - p2, tv2);
  let eps = mneeFdStepFromScales(chainScales);
  let residualTolerance = mneeResidualToleranceFromScales(chainScales);
  for (var it = 0u; it < maxIter; it = it + 1u) {
    let v1 = p1 + a1 * tu1 + b1 * tv1;
    let v2 = p2 + a2 * tu2 + b2 * tv2;
    let r = mneeChainResidual4d(v1, v2, n1, tu1, tv1, n2, tu2, tv2, lightP, recv, eta1i, eta1t, eta2i, eta2t);
    let rmag = length(r);
    out.v1 = v1; out.v2 = v2; out.residual = rmag; out.iters = it;
    if (rmag < residualTolerance) { return out; }
    // FD 4×4 Jacobian (D9.4 shared helper). Columns: d_a1,d_b1,d_a2,d_b2.
    let J = mneeChainFdJacobian4x4(v1, v2, n1, tu1, tv1, n2, tu2, tv2, lightP, recv, eta1i, eta1t, eta2i, eta2t, r, eps);
    let A = mat2x2f(J[0].xy, J[1].xy); // ∂r1/∂(a1,b1)
    let B = mat2x2f(J[2].xy, J[3].xy); // ∂r1/∂(a2,b2)
    let C = mat2x2f(J[0].zw, J[1].zw); // ∂r2/∂(a1,b1)
    let D = mat2x2f(J[2].zw, J[3].zw); // ∂r2/∂(a2,b2)
    let r1 = r.xy; let r2 = r.zw;
    if (!mneeMat2Invertible(A)) { return out; }
    let Ainv = mnee_inv2x2(A);
    let CAinv = C * Ainv;
    let S = D - CAinv * B;                       // Schur complement
    if (!mneeMat2Invertible(S)) { return out; }
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

// Fixed-capacity N-vertex manifold chain used by the production 3..8 vertex
// estimator. Each residual depends only on its previous/current/next vertex, so
// the Newton Jacobian is block tridiagonal with 2x2 blocks. The block Thomas
// solve below is O(N), preserves all inter-vertex coupling, and avoids a fake
// dense 16x16 inverse. Facet planes and event eta pairs are supplied by the
// probability-known proposal in caustic.wgsl.ts.
struct MneeBoundedChainGeometry {
  count: u32,
  // 0 = finite point/area endpoint, 1 = directional endpoint.
  sourceMode: u32,
  // Direction from the first manifold vertex toward a directional source.
  sourceDirection: vec3f,
  planeP: array<vec3f, ${MNEE_CHAIN_MAX_VERTICES}>,
  normal: array<vec3f, ${MNEE_CHAIN_MAX_VERTICES}>,
  tangentU: array<vec3f, ${MNEE_CHAIN_MAX_VERTICES}>,
  tangentV: array<vec3f, ${MNEE_CHAIN_MAX_VERTICES}>,
}

fn mneeBoundedChainScales(
  geometry: MneeBoundedChainGeometry,
  lightP: vec3f,
  recv: vec3f,
) -> vec2f {
  var coordinateScale = max(
    mneePointCoordinateScale(lightP), mneePointCoordinateScale(recv),
  );
  var localSpan = length(recv - lightP);
  var previous = lightP;
  for (var index = 0u; index < ${MNEE_CHAIN_MAX_VERTICES}u; index = index + 1u) {
    if (index >= geometry.count) { break; }
    let point = geometry.planeP[index];
    coordinateScale = max(coordinateScale, mneePointCoordinateScale(point));
    localSpan = max(
      localSpan,
      max(length(point - previous), length(recv - point)),
    );
    previous = point;
  }
  return vec2f(coordinateScale, localSpan);
}

struct MneeBoundedChainMedia {
  etaI: array<f32, ${MNEE_CHAIN_MAX_VERTICES}>,
  etaT: array<f32, ${MNEE_CHAIN_MAX_VERTICES}>,
}

struct MneeBoundedChainResult {
  vertices: array<vec3f, ${MNEE_CHAIN_MAX_VERTICES}>,
  residual: f32,
  iters: u32,
  valid: u32,
}

fn mneeBoundedChainResidualAt(
  vertices: array<vec3f, ${MNEE_CHAIN_MAX_VERTICES}>,
  geometry: MneeBoundedChainGeometry,
  media: MneeBoundedChainMedia,
  vertexIndex: u32,
  lightP: vec3f,
  recv: vec3f,
) -> vec2f {
  let v = vertices[vertexIndex];
  var previous = lightP;
  if (vertexIndex > 0u) { previous = vertices[vertexIndex - 1u]; }
  var next = recv;
  if (vertexIndex + 1u < geometry.count) { next = vertices[vertexIndex + 1u]; }
  var wi = mnee_safe_normalize(previous - v);
  if (vertexIndex == 0u && geometry.sourceMode == 1u) {
    wi = mnee_safe_normalize(geometry.sourceDirection);
  }
  let wo = mnee_safe_normalize(next - v);
  let h = mnee_safe_normalize(
    media.etaI[vertexIndex] * wi + media.etaT[vertexIndex] * wo,
  );
  let n = geometry.normal[vertexIndex];
  let hTan = h - dot(h, n) * n;
  return vec2f(
    dot(hTan, geometry.tangentU[vertexIndex]),
    dot(hTan, geometry.tangentV[vertexIndex]),
  );
}

fn mneeBoundedChainResidualMax(
  vertices: array<vec3f, ${MNEE_CHAIN_MAX_VERTICES}>,
  geometry: MneeBoundedChainGeometry,
  media: MneeBoundedChainMedia,
  lightP: vec3f,
  recv: vec3f,
) -> f32 {
  var residualMax = 0.0;
  for (var i = 0u; i < ${MNEE_CHAIN_MAX_VERTICES}u; i = i + 1u) {
    if (i >= geometry.count) { break; }
    residualMax = max(
      residualMax,
      length(mneeBoundedChainResidualAt(vertices, geometry, media, i, lightP, recv)),
    );
  }
  return residualMax;
}

fn mnee_mat2_det(m: mat2x2f) -> f32 {
  return m[0][0] * m[1][1] - m[1][0] * m[0][1];
}

fn mneeNewtonSolveChainBounded(
  geometry: MneeBoundedChainGeometry,
  media: MneeBoundedChainMedia,
  lightP: vec3f,
  recv: vec3f,
  seedVertices: array<vec3f, ${MNEE_CHAIN_MAX_VERTICES}>,
  maxIter: u32,
) -> MneeBoundedChainResult {
  var out: MneeBoundedChainResult;
  out.vertices = seedVertices;
  out.residual = 1e20;
  out.iters = 0u;
  out.valid = 0u;
  if (geometry.count == 0u || geometry.count > ${MNEE_CHAIN_MAX_VERTICES}u) {
    return out;
  }

  var coordA: array<f32, ${MNEE_CHAIN_MAX_VERTICES}>;
  var coordB: array<f32, ${MNEE_CHAIN_MAX_VERTICES}>;
  for (var initIndex = 0u; initIndex < ${MNEE_CHAIN_MAX_VERTICES}u; initIndex = initIndex + 1u) {
    if (initIndex >= geometry.count) { break; }
    let offset = seedVertices[initIndex] - geometry.planeP[initIndex];
    coordA[initIndex] = dot(offset, geometry.tangentU[initIndex]);
    coordB[initIndex] = dot(offset, geometry.tangentV[initIndex]);
  }

  let solverScales = mneeBoundedChainScales(geometry, lightP, recv);
  if (!mneeScalesRepresentable(solverScales)) { return out; }
  let eps = mneeFdStepFromScales(solverScales);
  let residualTolerance = mneeResidualToleranceFromScales(solverScales);
  for (var it = 0u; it < maxIter; it = it + 1u) {
    var vertices: array<vec3f, ${MNEE_CHAIN_MAX_VERTICES}>;
    for (var vi = 0u; vi < ${MNEE_CHAIN_MAX_VERTICES}u; vi = vi + 1u) {
      if (vi >= geometry.count) { break; }
      vertices[vi] = geometry.planeP[vi] +
        coordA[vi] * geometry.tangentU[vi] +
        coordB[vi] * geometry.tangentV[vi];
    }
    let residualMax = mneeBoundedChainResidualMax(
      vertices, geometry, media, lightP, recv,
    );
    out.vertices = vertices;
    out.residual = residualMax;
    out.iters = it;
    if (residualMax < residualTolerance) {
      out.valid = 1u;
      return out;
    }

    var residual: array<vec2f, ${MNEE_CHAIN_MAX_VERTICES}>;
    var lower: array<mat2x2f, ${MNEE_CHAIN_MAX_VERTICES}>;
    var diagonal: array<mat2x2f, ${MNEE_CHAIN_MAX_VERTICES}>;
    var upper: array<mat2x2f, ${MNEE_CHAIN_MAX_VERTICES}>;
    for (var ji = 0u; ji < ${MNEE_CHAIN_MAX_VERTICES}u; ji = ji + 1u) {
      if (ji >= geometry.count) { break; }
      let r0 = mneeBoundedChainResidualAt(vertices, geometry, media, ji, lightP, recv);
      residual[ji] = r0;

      var perturbed = vertices;
      perturbed[ji] = vertices[ji] + eps * geometry.tangentU[ji];
      let ownA = (
        mneeBoundedChainResidualAt(perturbed, geometry, media, ji, lightP, recv) - r0
      ) / eps;
      perturbed[ji] = vertices[ji] + eps * geometry.tangentV[ji];
      let ownB = (
        mneeBoundedChainResidualAt(perturbed, geometry, media, ji, lightP, recv) - r0
      ) / eps;
      diagonal[ji] = mat2x2f(ownA, ownB);

      lower[ji] = mat2x2f(vec2f(0.0), vec2f(0.0));
      if (ji > 0u) {
        perturbed = vertices;
        perturbed[ji - 1u] = vertices[ji - 1u] + eps * geometry.tangentU[ji - 1u];
        let prevA = (
          mneeBoundedChainResidualAt(perturbed, geometry, media, ji, lightP, recv) - r0
        ) / eps;
        perturbed[ji - 1u] = vertices[ji - 1u] + eps * geometry.tangentV[ji - 1u];
        let prevB = (
          mneeBoundedChainResidualAt(perturbed, geometry, media, ji, lightP, recv) - r0
        ) / eps;
        lower[ji] = mat2x2f(prevA, prevB);
      }

      upper[ji] = mat2x2f(vec2f(0.0), vec2f(0.0));
      if (ji + 1u < geometry.count) {
        perturbed = vertices;
        perturbed[ji + 1u] = vertices[ji + 1u] + eps * geometry.tangentU[ji + 1u];
        let nextA = (
          mneeBoundedChainResidualAt(perturbed, geometry, media, ji, lightP, recv) - r0
        ) / eps;
        perturbed[ji + 1u] = vertices[ji + 1u] + eps * geometry.tangentV[ji + 1u];
        let nextB = (
          mneeBoundedChainResidualAt(perturbed, geometry, media, ji, lightP, recv) - r0
        ) / eps;
        upper[ji] = mat2x2f(nextA, nextB);
      }
    }

    // Block Thomas elimination for J delta = -r.
    var cPrime: array<mat2x2f, ${MNEE_CHAIN_MAX_VERTICES}>;
    var dPrime: array<vec2f, ${MNEE_CHAIN_MAX_VERTICES}>;
    var singular = false;
    for (var fi = 0u; fi < ${MNEE_CHAIN_MAX_VERTICES}u; fi = fi + 1u) {
      if (fi >= geometry.count) { break; }
      var denom = diagonal[fi];
      var rhs = -residual[fi];
      if (fi > 0u) {
        denom = denom - lower[fi] * cPrime[fi - 1u];
        rhs = rhs - lower[fi] * dPrime[fi - 1u];
      }
      if (!mneeMat2Invertible(denom)) {
        singular = true;
        break;
      }
      let invDenom = mnee_inv2x2(denom);
      if (fi + 1u < geometry.count) {
        cPrime[fi] = invDenom * upper[fi];
      } else {
        cPrime[fi] = mat2x2f(vec2f(0.0), vec2f(0.0));
      }
      dPrime[fi] = invDenom * rhs;
    }
    if (singular) { return out; }

    var delta: array<vec2f, ${MNEE_CHAIN_MAX_VERTICES}>;
    var reverseIndex = geometry.count;
    loop {
      if (reverseIndex == 0u) { break; }
      reverseIndex = reverseIndex - 1u;
      delta[reverseIndex] = dPrime[reverseIndex];
      if (reverseIndex + 1u < geometry.count) {
        delta[reverseIndex] = delta[reverseIndex] -
          cPrime[reverseIndex] * delta[reverseIndex + 1u];
      }
    }

    // Globalize the coupled solve with a residual-decreasing line search.
    var scale = 1.0;
    var accepted = false;
    for (var bt = 0u; bt < 10u; bt = bt + 1u) {
      var trialVertices = vertices;
      for (var ti = 0u; ti < ${MNEE_CHAIN_MAX_VERTICES}u; ti = ti + 1u) {
        if (ti >= geometry.count) { break; }
        let trialA = coordA[ti] + scale * delta[ti].x;
        let trialB = coordB[ti] + scale * delta[ti].y;
        trialVertices[ti] = geometry.planeP[ti] +
          trialA * geometry.tangentU[ti] +
          trialB * geometry.tangentV[ti];
      }
      let trialResidual = mneeBoundedChainResidualMax(
        trialVertices, geometry, media, lightP, recv,
      );
      if (trialResidual < residualMax) {
        for (var ai = 0u; ai < ${MNEE_CHAIN_MAX_VERTICES}u; ai = ai + 1u) {
          if (ai >= geometry.count) { break; }
          coordA[ai] = coordA[ai] + scale * delta[ai].x;
          coordB[ai] = coordB[ai] + scale * delta[ai].y;
        }
        accepted = true;
        break;
      }
      scale = scale * 0.5;
    }
    if (!accepted) { return out; }
  }

  var finalVertices: array<vec3f, ${MNEE_CHAIN_MAX_VERTICES}>;
  for (var finalIndex = 0u; finalIndex < ${MNEE_CHAIN_MAX_VERTICES}u; finalIndex = finalIndex + 1u) {
    if (finalIndex >= geometry.count) { break; }
    finalVertices[finalIndex] = geometry.planeP[finalIndex] +
      coordA[finalIndex] * geometry.tangentU[finalIndex] +
      coordB[finalIndex] * geometry.tangentV[finalIndex];
  }
  out.vertices = finalVertices;
  out.residual = mneeBoundedChainResidualMax(
    finalVertices, geometry, media, lightP, recv,
  );
  out.iters = maxIter;
  if (out.residual <= residualTolerance) { out.valid = 1u; }
  return out;
}

fn mneeBoundedChainFocusingDet(
  geometry: MneeBoundedChainGeometry,
  media: MneeBoundedChainMedia,
  solved: MneeBoundedChainResult,
  lightP: vec3f,
  recv: vec3f,
  recvTu: vec3f,
  recvTv: vec3f,
  maxIter: u32,
) -> f32 {
  if (solved.valid == 0u || geometry.count == 0u) { return 0.0; }
  let solverScales = mneeBoundedChainScales(geometry, lightP, recv);
  if (!mneeScalesRepresentable(solverScales)) { return 0.0; }
  let eps = mneeFdStepFromScales(solverScales);
  let baseDirection = mnee_safe_normalize(solved.vertices[0] - lightP);
  let solveU = mneeNewtonSolveChainBounded(
    geometry, media, lightP, recv + recvTu * eps, solved.vertices, maxIter,
  );
  let solveV = mneeNewtonSolveChainBounded(
    geometry, media, lightP, recv + recvTv * eps, solved.vertices, maxIter,
  );
  if (solveU.valid == 0u || solveV.valid == 0u) { return 0.0; }
  let derivativeU = (
    mnee_safe_normalize(solveU.vertices[0] - lightP) - baseDirection
  ) / eps;
  let derivativeV = (
    mnee_safe_normalize(solveV.vertices[0] - lightP) - baseDirection
  ) / eps;
  let determinant = length(cross(derivativeU, derivativeV));
  if (!(determinant >= 0.0) || !(determinant < INFINITY)) { return 0.0; }
  return determinant;
}

// Directional-source irradiance transport. Perturb one square metre of receiver
// tangent area, re-solve the whole chain, and measure the projected source-side
// footprint at vertex zero. This Jacobian already contains receiver cosine.
fn mneeBoundedChainDirectionalFocusingDet(
  geometry: MneeBoundedChainGeometry,
  media: MneeBoundedChainMedia,
  solved: MneeBoundedChainResult,
  lightP: vec3f,
  recv: vec3f,
  recvTu: vec3f,
  recvTv: vec3f,
  maxIter: u32,
) -> f32 {
  if (solved.valid == 0u || geometry.count == 0u || geometry.sourceMode != 1u) {
    return 0.0;
  }
  let solverScales = mneeBoundedChainScales(geometry, lightP, recv);
  if (!mneeScalesRepresentable(solverScales)) { return 0.0; }
  let eps = mneeFdStepFromScales(solverScales);
  let solveU = mneeNewtonSolveChainBounded(
    geometry, media, lightP, recv + recvTu * eps, solved.vertices, maxIter,
  );
  let solveV = mneeNewtonSolveChainBounded(
    geometry, media, lightP, recv + recvTv * eps, solved.vertices, maxIter,
  );
  if (solveU.valid == 0u || solveV.valid == 0u) { return 0.0; }
  let derivativeU = (solveU.vertices[0] - solved.vertices[0]) / eps;
  let derivativeV = (solveV.vertices[0] - solved.vertices[0]) / eps;
  let determinant = abs(dot(
    cross(derivativeU, derivativeV),
    mnee_safe_normalize(geometry.sourceDirection),
  ));
  if (!(determinant >= 0.0) || !(determinant < INFINITY)) { return 0.0; }
  return determinant;
}

// Area endpoint change of variables: dA_light -> dOmega_receiver through the
// complete solved chain. The light axes are orthonormal physical metre axes, so
// the returned determinant is per unit emitter area.
fn mneeBoundedChainAreaPdfDet(
  geometry: MneeBoundedChainGeometry,
  media: MneeBoundedChainMedia,
  solved: MneeBoundedChainResult,
  lightP: vec3f,
  recv: vec3f,
  lightU: vec3f,
  lightV: vec3f,
  maxIter: u32,
) -> f32 {
  if (solved.valid == 0u || geometry.count == 0u || geometry.sourceMode != 0u) {
    return 0.0;
  }
  let solverScales = mneeBoundedChainScales(geometry, lightP, recv);
  if (!mneeScalesRepresentable(solverScales)) { return 0.0; }
  let eps = mneeFdStepFromScales(solverScales);
  let lightAreaScale = length(cross(lightU, lightV));
  if (!(lightAreaScale > mneeLengthFloorFromScales(solverScales) *
      mneeLengthFloorFromScales(solverScales)) ||
      !(lightAreaScale < INFINITY)) { return 0.0; }
  let tu = mnee_safe_normalize(lightU);
  let lightN = mnee_safe_normalize(cross(lightU, lightV));
  let tv = mnee_safe_normalize(cross(lightN, tu));
  let solveU = mneeNewtonSolveChainBounded(
    geometry, media, lightP + tu * eps, recv, solved.vertices, maxIter,
  );
  let solveV = mneeNewtonSolveChainBounded(
    geometry, media, lightP + tv * eps, recv, solved.vertices, maxIter,
  );
  if (solveU.valid == 0u || solveV.valid == 0u) { return 0.0; }
  let last = geometry.count - 1u;
  let baseDirection = mnee_safe_normalize(solved.vertices[last] - recv);
  let derivativeU = (
    mnee_safe_normalize(solveU.vertices[last] - recv) - baseDirection
  ) / eps;
  let derivativeV = (
    mnee_safe_normalize(solveV.vertices[last] - recv) - baseDirection
  ) / eps;
  let determinant = length(cross(derivativeU, derivativeV));
  if (!(determinant >= 0.0) || !(determinant < INFINITY)) { return 0.0; }
  return determinant;
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
  let chainScales = max(
    mneeLocalScales3(v1, v2, lightP),
    mneeLocalScales3(v1, v2, recv),
  );
  if (!mneeScalesRepresentable(chainScales)) { return 0.0; }
  let eps = mneeFdStepFromScales(chainScales);
  let r = mneeChainResidual4d(v1, v2, n1, tu1, tv1, n2, tu2, tv2, lightP, recv, eta1i, eta1t, eta2i, eta2t);
  // FD 4×4 Jacobian (D9.4 shared helper). Columns: d_a1,d_b1,d_a2,d_b2.
  let J = mneeChainFdJacobian4x4(v1, v2, n1, tu1, tv1, n2, tu2, tv2, lightP, recv, eta1i, eta1t, eta2i, eta2t, r, eps);
  let A = mat2x2f(J[0].xy, J[1].xy);
  let B = mat2x2f(J[2].xy, J[3].xy);
  let C = mat2x2f(J[0].zw, J[1].zw);
  let D = mat2x2f(J[2].zw, J[3].zw);
  if (!mneeMat2Invertible(A)) { return 0.0; }
  let Ainv = mnee_inv2x2(A);
  let CAinv = C * Ainv;
  let S = D - CAinv * B;
  if (!mneeMat2Invertible(S)) { return 0.0; }
  let Sinv = mnee_inv2x2(S);
  // ∂r/∂(light) along the area axes s=lightU, t=lightV (FD).
  let r_s = (mneeChainResidual4d(v1, v2, n1, tu1, tv1, n2, tu2, tv2, lightP + eps * lightU, recv, eta1i, eta1t, eta2i, eta2t) - r) / eps;
  let r_t = (mneeChainResidual4d(v1, v2, n1, tu1, tv1, n2, tu2, tv2, lightP + eps * lightV, recv, eta1i, eta1t, eta2i, eta2t) - r) / eps;
  // d(a2,b2)/d(s,t) = −[J_full⁻¹ J_light]_{rows 3,4} = S⁻¹(C A⁻¹ r_top − r_bot).
  let dab2_ds = Sinv * (CAinv * r_s.xy - r_s.zw);
  let dab2_dt = Sinv * (CAinv * r_t.xy - r_t.zw);
  let dv2_ds = tu2 * dab2_ds.x + tv2 * dab2_ds.y;
  let dv2_dt = tu2 * dab2_dt.x + tv2 * dab2_dt.y;
  let dvec = v2 - recv;
  let dist = length(dvec);
  if (dist <= mneeLengthFloorFromScales(mneeLocalScales3(v2, recv, v2))) {
    return 0.0;
  }
  let w = dvec / dist;
  let dw_ds = (dv2_ds - w * dot(w, dv2_ds)) / dist;
  let dw_dt = (dv2_dt - w * dot(w, dv2_dt)) / dist;
  let determinant = length(cross(dw_ds, dw_dt));
  if (!(determinant >= 0.0) || !(determinant < INFINITY)) { return 0.0; }
  return determinant;
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
  let solverScales = mneeLocalScales3(res.vertex, recv, lightPos);
  if (!mneeScalesRepresentable(solverScales) ||
      res.residual > mneeResidualToleranceFromScales(solverScales)) {
    return vec3f(0.0);
  }
  let v = res.vertex;
  let wi = mnee_safe_normalize(v - recv);               // receiver's incident dir (toward the mirror vertex)
  let nDotL = max(dot(recvNormal, wi), 0.0);
  if (nDotL <= 1e-5) { return vec3f(0.0); }
  let dTotal = length(lightPos - v) + length(recv - v); // unfolded path length = dist(image, recv)
  if (dTotal <= mneeLengthFloorFromScales(solverScales)) {
    return vec3f(0.0);
  }
  return lightIntensity * nDotL / (dTotal * dTotal);
}
`;

// NOTE: Harness shaders (MNEE_NEWTON_HARNESS_WGSL, MNEE_NEWTON_JAC_HARNESS_WGSL,
// MNEE_JACOBIAN_HARNESS_WGSL, MNEE_PDF_HARNESS_WGSL, MNEE_CHAIN_HARNESS_WGSL,
// MNEE_CHAIN_PDF_HARNESS_WGSL, MNEE_REFLECTION_HARNESS_WGSL) and the TypeScript
// helpers (packMneeHarnessInput, MNEE_HARNESS_INPUT_FLOATS) live in
// mneeNewton.harness.wgsl.ts — kept separate from this production module.
