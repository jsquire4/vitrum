/**
 * Early shared WGSL include for pt-webgpu.
 *
 * This captures transferable, renderer-agnostic pieces from stainedGlass:
 * - PCG RNG
 * - BVH node/ray/hit structs aligned with three-mesh-bvh's packed layout
 * - Triangle intersection and basic utilities
 *
 * ReSTIR/DDGI-specific reservoir and lighting logic is intentionally excluded.
 *
 * W2-C1: safeInvDir + intersectTriangle (Moller-Trumbore) come from the
 * canonical @vitrum/shared-bvh/wgsl/bvhTraverse.wgsl.ts module and are
 * concatenated at the tail of this shader header.  intersectTriangle takes
 * triEps as a function parameter (the canonical signature); call sites in
 * pt-webgpu pass `params.triIntersectEpsilon` from the FrameParams UBO.
 */
import { BVH_TRAVERSE_WGSL } from '@vitrum/shared-bvh';

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

fn pcgInit(px: u32, py: u32, frameSeed: u32) -> u32 {
  var state = px * 1664525u + py * 1013904223u + frameSeed * 22695477u;
  state ^= state >> 17u;
  state ^= state << 31u;
  state ^= state >> 11u;
  return state;
}

fn pcgNext(state: ptr<function, u32>) -> u32 {
  (*state) = (*state) * 747796405u + 2891336453u;
  var word = (((*state) >> (((*state) >> 28u) + 4u)) ^ (*state)) * 277803737u;
  word = (word >> 22u) ^ word;
  return word;
}

fn rand_f32(state: ptr<function, u32>) -> f32 {
  return f32(pcgNext(state)) / f32(0xFFFFFFFFu);
}

fn safe_normalize(v: vec3f) -> vec3f {
  let len = length(v);
  if (len < 1e-8) {
    return vec3f(0.0, 1.0, 0.0);
  }
  return v / len;
}

// W2-C1: safeInvDir and intersectTriangle live in
// @vitrum/shared-bvh/wgsl/bvhTraverse.wgsl.ts (canonical).  The canonical
// intersectTriangle takes triEps as a parameter (so the primitive does not
// depend on the host UBO struct shape); pt-webgpu's two call sites
// (traceMeshBvh and intersectMeshAreaLightRay) thread
// params.triIntersectEpsilon explicitly.
${BVH_TRAVERSE_WGSL}
`;
