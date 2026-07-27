/**
 * vecMath.ts — small pure Vec3 helpers shared by the host-CPU scene packers.
 *
 * The path-tracing backends (`@vitrum/pt-webgl2` lights/mesh-area packers,
 * `@vitrum/pt-webgpu` emitter packer) each re-spelled `cross`/`normalize`/
 * `tangentBasis`/`lengthOf` inline. This module single-sources them so the
 * packed emitter geometry (u/v basis synthesis for disc/spot emitters,
 * triangle areas) is computed identically everywhere.
 *
 * SEMANTICS NOTE — {@link normalize} returns the zero vector `[0,0,0]` for a
 * near-degenerate input (`len < 1e-12`). This matches the pt-webgl2 lights
 * packer's historical behaviour EXACTLY and is what {@link tangentBasis}
 * depends on; it is deliberately NOT the "pass the input through" convention
 * used by `@vitrum/shared-bvh`'s `v3Normalize` (that one is for BVH transform
 * math with a different degeneracy contract). Do not "unify" the two without a
 * byte-identity A/B on both consumers.

 */
import { requireFiniteVec3 } from './numericGuards.js';


export type Vec3Tuple = readonly [number, number, number];

/** Euclidean length of a Vec3. */
export function vecLength(v: Vec3Tuple): number {
  requireFiniteVec3(v, 'vecLength.v');
  return Math.hypot(v[0], v[1], v[2]);
}

/** Cross product a × b. */
export function vecCross(a: Vec3Tuple, b: Vec3Tuple): [number, number, number] {
  requireFiniteVec3(a, 'vecCross.a');
  requireFiniteVec3(b, 'vecCross.b');
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

/**
 * Normalize a Vec3. Returns `[0,0,0]` when the length is below `1e-12`
 * (the pt-webgl2 lights-packer degeneracy contract — see the module note).
 */
export function vecNormalize(v: Vec3Tuple): [number, number, number] {
  const len = vecLength(v);
  if (len < 1e-12) return [0, 0, 0];
  requireFiniteVec3(v, 'vecNormalize.v');
  return [v[0] / len, v[1] / len, v[2] / len];
}

/**
 * Build two unit tangent vectors spanning the plane perpendicular to `n`.
 * Used to synthesize the (u, v) basis for emitters whose core representation
 * gives an axis but no explicit in-plane axes: disc-area supplies a radius,
 * while spot is a delta-position source and needs only an orientation basis.
 * Deterministic so packed data is stable across calls.
 */
export function tangentBasis(n: Vec3Tuple): {
  t: [number, number, number];
  b: [number, number, number];
} {
  const nn = vecNormalize(n);
  // Pick the world axis least aligned with nn to avoid degeneracy.
  if (vecLength(nn) < 1e-12) {
    return { t: [0, 0, 1], b: [1, 0, 0] };
  }
  const ref: Vec3Tuple = Math.abs(nn[0]) > 0.9 ? [0, 1, 0] : [1, 0, 0];
  const t = vecNormalize(vecCross(ref, nn));
  const b = vecNormalize(vecCross(nn, t));
  return { t, b };
}
