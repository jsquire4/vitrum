/**
 * mneeNewton.wgsl.ts — the CORE of a real Hanika-2015 manifold-NEE: the
 * half-vector Newton solve that finds the specular vertex on a surface so the
 * light↔receiver path obeys the specular law (reflection here; refraction is the
 * eta-generalized half-vector — a follow-up). This replaces the cone-search
 * APPROXIMATION in `caustic.wgsl.ts:manifoldNeeContribution` (which has no
 * half-vector constraint, Newton iteration, or change-of-variables Jacobian).
 *
 * The solve is SELF-VALIDATING: the tangential half-vector residual must converge
 * to ~0, and for a flat mirror the converged vertex must equal the analytic
 * mirror-image reflection point — so the GPU harness here (`mnee-newton-validate.ts`)
 * checks correctness without any radiometric A/B. The Jacobian is finite-difference
 * (the analytic geometric Jacobian needed for the connection PDF is the next
 * increment); FD is enough to DRIVE the Newton step + prove convergence.
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
): number[] {
  return [
    receiver[0], receiver[1], receiver[2], 0,
    light[0], light[1], light[2], 0,
    planePoint[0], planePoint[1], planePoint[2], 0,
  ];
}

export const MNEE_NEWTON_WGSL = /* wgsl */ `
fn mnee_safe_normalize(v: vec3f) -> vec3f {
  let l = length(v);
  if (l < 1e-12) { return vec3f(0.0); }
  return v / l;
}

// Tangential half-vector residual at vertex v: project h = normalize(wi+wo) onto
// the surface tangent plane (tu,tv). Zero ⇔ h ∥ nm ⇔ the reflection law holds.
fn mneeHalfVectorResidual2d(v: vec3f, recv: vec3f, light: vec3f, nm: vec3f, tu: vec3f, tv: vec3f) -> vec2f {
  let wi = mnee_safe_normalize(light - v);
  let wo = mnee_safe_normalize(recv - v);
  let h = mnee_safe_normalize(wi + wo);
  let hTan = h - dot(h, nm) * nm;
  return vec2f(dot(hTan, tu), dot(hTan, tv));
}

struct MneeNewtonResult { vertex: vec3f, residual: f32, iters: u32 }

// 2D Newton solve on the surface (a,b) coords for the reflection specular vertex.
// FD Jacobian; solves J·δ = −r each step. Converges to the mirror-image point.
fn mneeNewtonReflect(p0: vec3f, nm: vec3f, tu: vec3f, tv: vec3f, recv: vec3f, light: vec3f, maxIter: u32) -> MneeNewtonResult {
  var a = 0.0;
  var b = 0.0;
  let eps = 1e-3;
  var out: MneeNewtonResult;
  for (var it = 0u; it < maxIter; it = it + 1u) {
    let v = p0 + a * tu + b * tv;
    let r0 = mneeHalfVectorResidual2d(v, recv, light, nm, tu, tv);
    let rmag = length(r0);
    out.vertex = v; out.residual = rmag; out.iters = it;
    if (rmag < 1e-5) { return out; }
    // FD Jacobian columns: ∂r/∂a, ∂r/∂b.
    let ra = mneeHalfVectorResidual2d(p0 + (a + eps) * tu + b * tv, recv, light, nm, tu, tv);
    let rb = mneeHalfVectorResidual2d(p0 + a * tu + (b + eps) * tv, recv, light, nm, tu, tv);
    let j00 = (ra.x - r0.x) / eps; let j10 = (ra.y - r0.y) / eps;
    let j01 = (rb.x - r0.x) / eps; let j11 = (rb.y - r0.y) / eps;
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
  out.residual = length(mneeHalfVectorResidual2d(v, recv, light, nm, tu, tv));
  out.iters = maxIter;
  return out;
}
`;

/** Harness kernel: runs the Newton solve per config (mirror plane z = planePoint.z,
 *  normal +z, tangents +x/+y) and writes the converged vertex + final residual. */
export const MNEE_NEWTON_HARNESS_WGSL = /* wgsl */ `
${MNEE_NEWTON_WGSL}

struct MneeIn { recv: vec3f, _p0: f32, light: vec3f, _p1: f32, planePoint: vec3f, _p2: f32 }
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
  let r = mneeNewtonReflect(c.planePoint, nm, tu, tv, c.recv, c.light, ${MNEE_NEWTON_MAX_ITERS}u);
  hOut[i] = vec4f(r.vertex, r.residual);
}
`;
