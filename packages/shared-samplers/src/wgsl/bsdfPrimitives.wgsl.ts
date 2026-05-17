/**
 * Canonical BSDF / sampling-frame WGSL primitives — single source of truth
 * across @vitrum/walkaround-hybrid and @vitrum/pt-webgpu.
 *
 * W2-C6 dedup (premium-grade-refactor-20260517 §W2): the following helpers
 * were declared in BOTH consumer packages with minor cosmetic drift
 * (capitalisation, argument order, defensive guards).  This module is the
 * canonical declaration.
 *
 * Functions exported:
 *   fn buildONB(n: vec3f, T: ptr<function, vec3f>, B: ptr<function, vec3f>)
 *       Build an orthonormal basis (T, B, N) around a unit normal `n`.
 *       Uses the "world-up vs world-right fallback" form (branch on
 *       |n.y| > 0.999) — matches walkaround-hybrid's pre-W2-C6 spelling.
 *       Pre-W2-C6 pt-webgpu used the lower-case `buildOnb`; that name is
 *       provided as an alias so existing pt-webgpu call sites compile
 *       unchanged.
 *
 *   fn sampleCosineHemisphere(n: vec3f, rng: ptr<function, u32>) -> vec3f
 *       Cosine-weighted hemisphere sample in world space around the unit
 *       normal `n`.  Concentric-square form (Pharr §A.5.3).  Pre-W2-C6
 *       pt-webgpu used `cosineHemisphereSample(rng, n)` with swapped
 *       argument order; that spelling is provided as an alias.
 *
 *   fn cosineHemispherePdf(n: vec3f, wi: vec3f) -> f32
 *       PDF of `sampleCosineHemisphere`, evaluated for an arbitrary `wi`.
 *
 *   fn fresnelSchlick(cosTheta: f32, F0: vec3f) -> vec3f
 *       Schlick 1994 approximation to Fresnel reflectance.  Uses the
 *       defensively-clamped pt-webgpu form (clamp(1 - cosTheta, 0, 1))
 *       which is a no-op when callers pre-saturate cosTheta with
 *       `max(0, dot(n, h))` (walkaround's contract) and a safety net
 *       otherwise.
 *
 * The GGX D / G / Smith helpers (distributionGGX vs ggxD / geometrySmith
 * vs smithG1) intentionally remain per-consumer:
 *   - walkaround takes roughness directly and rolls a² = (rough²)² inside;
 *   - pt-webgpu accepts the pre-squared α = rough² and exposes G1 / D
 *     separately so the caller controls the joint Smith product (Heitz
 *     2014 §6 separable form, paired with VNDF sampling).
 * Unifying these would change the API contract on at least one consumer
 * and risk subtle behaviour drift — out of scope for W2-C6, which
 * targets only unambiguous wins.
 *
 * References:
 *   - Pharr, Jakob, Humphreys.  Physically Based Rendering 4th ed.  §A.5.3
 *     (cosine-weighted hemisphere sampling — concentric-square form).
 *   - Schlick, C. (1994).  "An Inexpensive BRDF Model for Physically-based
 *     Rendering."  Computer Graphics Forum 13(3):233-246.
 *   - Heitz, E. (2014).  "Understanding the Masking-Shadowing Function in
 *     Microfacet-Based BRDFs."  JCGT 3(2):48-107.
 */

export const BSDF_PRIMITIVES_WGSL = /* wgsl */ `

// ============================================================
// BSDF / sampling-frame primitives (canonical — @vitrum/shared-samplers)
// ============================================================

// Build an orthonormal basis (T, B, N) around a unit normal.  Branches on
// |n.y| > 0.999 to avoid a degenerate cross-product when n is near the
// world-up axis.
fn buildONB(n: vec3f, T: ptr<function, vec3f>, B: ptr<function, vec3f>) {
  var up = vec3f(0.0, 1.0, 0.0);
  if (abs(n.y) > 0.999) { up = vec3f(1.0, 0.0, 0.0); }
  *T = normalize(cross(up, n));
  *B = cross(n, *T);
}

// Pre-W2-C6 pt-webgpu used the lowercase \`buildOnb\` spelling. Alias.
fn buildOnb(n: vec3f, T: ptr<function, vec3f>, B: ptr<function, vec3f>) {
  buildONB(n, T, B);
}

// Cosine-weighted hemisphere sample around the unit normal n.
// Returns a unit-length world-space direction.
fn sampleCosineHemisphere(n: vec3f, rng: ptr<function, u32>) -> vec3f {
  let xi = rand_f32_2(rng);
  let r = sqrt(xi.x);
  let phi = 2.0 * 3.14159265358979 * xi.y;
  let localDir = vec3f(r * cos(phi), r * sin(phi), sqrt(max(0.0, 1.0 - xi.x)));
  var T: vec3f; var B: vec3f;
  buildONB(n, &T, &B);
  return localDir.x * T + localDir.y * B + localDir.z * n;
}

// Pre-W2-C6 pt-webgpu used \`cosineHemisphereSample(rng, n)\` with swapped
// argument order. Alias preserves existing pt-webgpu call sites.
fn cosineHemisphereSample(rng: ptr<function, u32>, n: vec3f) -> vec3f {
  return sampleCosineHemisphere(n, rng);
}

fn cosineHemispherePdf(n: vec3f, wi: vec3f) -> f32 {
  return max(0.0, dot(n, wi)) * 0.31830988618;
}

// Schlick 1994 approximation to Fresnel reflectance.
// cosTheta is defensively clamped to [0, 1] — callers in walkaround
// already pre-saturate via max(0, dot(n, h)), so the clamp is a no-op
// there; pt-webgpu callers benefit from the safety net.
fn fresnelSchlick(cosTheta: f32, F0: vec3f) -> vec3f {
  let m = clamp(1.0 - cosTheta, 0.0, 1.0);
  let m2 = m * m;
  let m5 = m2 * m2 * m;
  return F0 + (vec3f(1.0) - F0) * m5;
}

`;
