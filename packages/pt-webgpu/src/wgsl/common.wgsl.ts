/**
 * Early shared WGSL include for pt-webgpu.
 *
 * This captures transferable, renderer-agnostic pieces from stainedGlass:
 * - PCG RNG (canonical @vitrum/shared-samplers, since W2-C6)
 * - BSDF sampling-frame primitives — buildONB / buildOnb,
 *   sampleCosineHemisphere / cosineHemisphereSample, cosineHemispherePdf,
 *   fresnelSchlick (canonical @vitrum/shared-samplers, since W2-C6)
 * - BVH node/ray/hit structs aligned with three-mesh-bvh's packed layout
 * - Triangle intersection and basic utilities
 *
 * ReSTIR/DDGI-specific reservoir and lighting logic is intentionally excluded.
 *
 * W2-C6 — the duplicated PCG + BSDF-sampling primitives that pt-webgpu and
 * walkaround-hybrid previously declared independently now live in a single
 * canonical module at @vitrum/shared-samplers.  We concat-import the
 * canonical bytes here so the rest of `pathTraceBruteforce.wgsl.ts` can
 * still resolve every symbol by name without going through the W1-R6
 * include-graph (pt-webgpu does not use the include-graph today).
 */
import { PCG_WGSL, BSDF_PRIMITIVES_WGSL } from '@vitrum/shared-samplers';

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

${BSDF_PRIMITIVES_WGSL}

fn safeInvDir(d: vec3f) -> vec3f {
  // Williams 2005 §4 IEEE-safe form. When a direction component is
  // exactly zero, 0 * Inf = NaN poisons the slab test if the ray origin
  // coincides with an AABB face. Substitute a tiny signed value.
  return vec3f(
    select(1.0 / d.x, sign(d.x) * 1e30, abs(d.x) < 1e-30),
    select(1.0 / d.y, sign(d.y) * 1e30, abs(d.y) < 1e-30),
    select(1.0 / d.z, sign(d.z) * 1e30, abs(d.z) < 1e-30),
  );
}

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
