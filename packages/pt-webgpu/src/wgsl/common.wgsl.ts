import { PCG_WGSL } from '@vitrum/shared-samplers';
import { SAFE_INV_DIR_WGSL, MOLLER_TRUMBORE_WGSL } from '@vitrum/shared-bvh';

/**
 * Early shared WGSL include for pt-webgpu.
 *
 * This captures transferable, renderer-agnostic pieces from stainedGlass:
 * - PCG RNG
 * - BVH node/ray/hit structs aligned with three-mesh-bvh's packed layout
 * - Triangle intersection and basic utilities
 *
 * ReSTIR/DDGI-specific reservoir and lighting logic is intentionally excluded.
 */
export const PT_WEBGPU_COMMON_WGSL = /* wgsl */ `
const PI = 3.14159265358979;
const INV_PI = 0.31830988618;
const INV_2PI = 0.15915494309189535;
const INFINITY = 1e20;

struct BVHNode {
  boundsMin: array<f32, 3>,
  boundsMax: array<f32, 3>,
  rightChildOrTriOffset: u32,
  splitAxisOrTriCount: u32,
};

struct Ray {
  origin: vec3f,
  direction: vec3f,
};

struct HitResult {
  didHit: bool,
  dist: f32,
  triIndex: u32,
  bary: vec3f,
  normal: vec3f,
};

${PCG_WGSL}

fn safe_normalize(v: vec3f) -> vec3f {
  let len = length(v);
  if (len < 1e-8) {
    return vec3f(0.0, 1.0, 0.0);
  }
  return v / len;
}

${SAFE_INV_DIR_WGSL}

${MOLLER_TRUMBORE_WGSL}

// intersectTriangle: pt-webgpu's scalar-returning wrapper over the canonical
// shared Moller-Trumbore core (mollerTrumboreCore, prepended above as
// MOLLER_TRUMBORE_WGSL from @vitrum/shared-bvh). The MATH is now single-sourced
// in shared-bvh -- both this wrapper and the canonical struct-returning
// intersectTriangle in BVH_INTERSECT_WGSL delegate to the same core.
//
// pt-webgpu keeps its own thin f32-returning wrapper (rather than composing the
// full BVH_INTERSECT_WGSL) because its traversal kernels define their own
// BVHNode / Ray / SceneHit / HitResult structs, which would collide with the
// ones BVH_INTERSECT_WGSL declares. The wrapper just unpacks the core's
// TriHit: returns the hit distance t on a hit, INFINITY on a miss. The
// three pt-webgpu call sites (traceMeshBvh in intersection/intersectionLite,
// intersectMeshAreaLightRay in connect) compare the returned f32 against their
// own t-bounds, so the f32 contract is preserved.
//
// NUMERICS NOTE (V7): switching to the canonical core changes edge behaviour --
// the core uses triEps-tolerant SIGNED barycentric tests (u/v/w < -triEps)
// instead of the old strict u<0||u>1 / v<0||u+v>1 tests, so hits grazing a
// triangle edge by less than triEps are now accepted (closes shared-edge
// cracks). The hit distance t for interior hits is algebraically unchanged.
// The __tests__/cpuTracer.ts oracle mirrors this same core so it stays in sync.
fn intersectTriangle(origin: vec3f, dir: vec3f, a: vec3f, b: vec3f, c: vec3f) -> f32 {
  let core = mollerTrumboreCore(origin, dir, a, b, c, params.triIntersectEpsilon);
  if (!core.hit) {
    return INFINITY;
  }
  return core.t;
}
`;
