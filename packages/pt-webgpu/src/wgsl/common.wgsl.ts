import { PCG_WGSL } from '@vitrum/shared-samplers';
import { SAFE_INV_DIR_WGSL } from '@vitrum/shared-bvh';

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

// intersectTriangle: pt-webgpu keeps its own scalar-returning Möller–Trumbore
// (h = cross(dir, e2), det = dot(e1, h)) here rather than composing the
// canonical @vitrum/shared-bvh intersectTriangle. The canonical returns an
// IntersectionResult (struct), takes a triEps parameter, uses the
// geometric-normal determinant form (n = cross(e1,e2), det = -dot(dir,n)) with
// triEps-tolerant barycentric tests, and is bundled inside BVH_INTERSECT_WGSL
// alongside BVHNode / Ray struct definitions that would collide with the ones
// declared in this module. This scalar form is also mirrored by the
// __tests__/cpuTracer.ts GPU-acceptance oracle, so swapping it is not a pure
// dedup. safeInvDir IS now shared (imported above as SAFE_INV_DIR_WGSL).
fn intersectTriangle(origin: vec3f, dir: vec3f, a: vec3f, b: vec3f, c: vec3f) -> f32 {
  let e1 = b - a;
  let e2 = c - a;
  let h = cross(dir, e2);
  let det = dot(e1, h);
  if (abs(det) < params.triIntersectEpsilon) {
    return INFINITY;
  }
  let invDet = 1.0 / det;
  let s = origin - a;
  let u = dot(s, h) * invDet;
  if (u < 0.0 || u > 1.0) {
    return INFINITY;
  }
  let q = cross(s, e1);
  let v = dot(dir, q) * invDet;
  if (v < 0.0 || u + v > 1.0) {
    return INFINITY;
  }
  let t = dot(e2, q) * invDet;
  if (t < params.triIntersectEpsilon) {
    return INFINITY;
  }
  return t;
}
`;
