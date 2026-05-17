/**
 * Canonical BVH-traversal WGSL primitives — single source of truth.
 *
 * W2-C1 dedup (premium-grade-refactor-20260517 §W2): four shader files
 * historically inlined near-identical copies of the IEEE-safe inverse-
 * direction helper and the Möller-Trumbore triangle intersection, with
 * cosmetic variation (parameter renames, hardcoded vs. UBO-plumbed
 * triangle-epsilon).  The duplicates lived in:
 *
 *   - packages/walkaround-hybrid/src/shaders/common.wgsl.ts
 *   - packages/walkaround-hybrid/src/ddgi/wgsl/probeUpdateRays.wgsl.ts
 *   - packages/walkaround-hybrid/src/rc/wgsl/probeRayCast.wgsl.ts
 *   - packages/pt-webgpu/src/wgsl/common.wgsl.ts
 *
 * The full BVH-stack-traversal wrappers (`bvhIntersectFirstHit`,
 * `bvhIntersectAny`, `bvhTraceFirstHit`, `traceMeshBvh`) each return a
 * consumer-specific result struct (HitResult vs. IntersectionResult vs.
 * SceneHit) and bind different geometry-buffer types, so they cannot share
 * one signature without breaking the existing GPU bind-group layouts.
 * What IS safely shareable is the per-primitive math: the inverse-direction
 * helper and the Möller-Trumbore triangle test.  This module exports those.
 *
 * Functions exported:
 *   fn safeInvDir(d: vec3f) -> vec3f
 *       Williams 2005 §4 IEEE-safe inverse-direction.  When a direction
 *       component is exactly zero, 1/0 = ±Inf is IEEE-valid but
 *       0 * ±Inf = NaN can poison the slab test when the ray origin
 *       coincides with an AABB face.  We substitute a finite large value.
 *       WGSL sign(0) == 0, so for a zero component sign(d.x) * 1e30 == 0,
 *       and (bMin - origin) * 0 == 0 — the axis contributes zero to
 *       tNear/tFar, which is correct: a zero-direction ray cannot enter
 *       or exit the slab through that axis.
 *
 *   fn intersectTriangle(origin: vec3f, dir: vec3f, a: vec3f, b: vec3f,
 *                        c: vec3f, triEps: f32) -> f32
 *       Möller-Trumbore triangle intersection; returns t or INFINITY.
 *       Caller supplies the coplanarity floor (`triEps`) as a parameter
 *       rather than reading from a module-scope constant.  This keeps the
 *       primitive compilable when concatenated into shaders that bind
 *       different UBO structs (each consumer threads its own per-frame
 *       epsilon — e.g. WalkaroundUBO.triIntersectEpsilon,
 *       CascadeUniforms.triIntersectEpsilon, FrameParams.triIntersectEpsilon).
 *       Returns INFINITY if INFINITY is in scope; callers MUST declare
 *       `const INFINITY = 1e20;` or equivalent before concatenation.
 *
 * References:
 *   Williams 2005, "An Efficient and Robust Ray-Box Intersection Algorithm",
 *   JGT 10(1):49-54.
 *   Möller & Trumbore 1997, "Fast, Minimum Storage Ray/Triangle
 *   Intersection", JGT 2(1):21-28.
 */

export const BVH_TRAVERSE_WGSL = /* wgsl */ `
fn safeInvDir(d: vec3f) -> vec3f {
  return vec3f(
    select(1.0 / d.x, sign(d.x) * 1e30, abs(d.x) < 1e-30),
    select(1.0 / d.y, sign(d.y) * 1e30, abs(d.y) < 1e-30),
    select(1.0 / d.z, sign(d.z) * 1e30, abs(d.z) < 1e-30),
  );
}

fn intersectTriangle(origin: vec3f, dir: vec3f, a: vec3f, b: vec3f, c: vec3f, triEps: f32) -> f32 {
  let e1 = b - a;
  let e2 = c - a;
  let h = cross(dir, e2);
  let det = dot(e1, h);
  if (abs(det) < triEps) { return INFINITY; }
  let invDet = 1.0 / det;
  let s = origin - a;
  let u = dot(s, h) * invDet;
  if (u < 0.0 || u > 1.0) { return INFINITY; }
  let q = cross(s, e1);
  let v = dot(dir, q) * invDet;
  if (v < 0.0 || u + v > 1.0) { return INFINITY; }
  let t = dot(e2, q) * invDet;
  if (t < triEps) { return INFINITY; }
  return t;
}
`;
