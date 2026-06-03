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
 * REMAINING (radiometric, deliberately NOT here): wiring these into
 * `manifoldNeeContribution` to replace the cone-search — its validation is a
 * caustic render against a converged reference (design the scene offline first).
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
