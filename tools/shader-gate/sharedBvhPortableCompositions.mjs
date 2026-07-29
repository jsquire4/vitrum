// @ts-nocheck -- Deno loads the production TypeScript WGSL exports directly.
/**
 * Verbatim shared-BVH shader compositions used by both the Naga and Chromium
 * Tint gates. The library strings are interpolated without source rewriting.
 */
import {
  BVH_CAST_SHADOW_MASK_WGSL,
  BVH_CAST_SHADOW_PREDICATE_WGSL,
  BVH_INTERSECT_WGSL,
  CWBVH_INTERSECT_WGSL,
  TLAS_TRAVERSAL_WGSL,
} from "../../packages/shared-bvh/src/index.ts";

const binaryTlasBindingsWgsl = /* wgsl */ `
@group(0) @binding(0) var<storage, read> bvh: array<BVHNode>;
@group(0) @binding(1) var<storage, read> bvh_index: array<vec4u>;
@group(0) @binding(2) var<storage, read> bvh_position: array<vec4f>;
@group(0) @binding(3) var<storage, read> tlasNodes: array<BVHNode>;
@group(0) @binding(4) var<storage, read> tlasInstanceIndices: array<u32>;
@group(0) @binding(5) var<storage, read> tlasBlasRoots: array<u32>;
@group(0) @binding(6) var<storage, read> tlasInstanceWorldToLocal: array<vec4f>;
@group(0) @binding(7) var<storage, read> tlasInstanceLocalToWorld: array<vec4f>;
`;

const binaryTlasCoreWgsl = `${BVH_INTERSECT_WGSL}
${TLAS_TRAVERSAL_WGSL}
${binaryTlasBindingsWgsl}`;

const binaryTlasGateWgsl = `${binaryTlasCoreWgsl}
@compute @workgroup_size(1)
fn binaryTlasGateMain() {
  var ray: Ray;
  ray.origin = vec3f(0.0, 0.0, 1.0);
  ray.direction = vec3f(0.0, 0.0, -1.0);
  let closest = traceTlasFirstHit(0u, ray, 1e-5);
  let anyHit = traceTlasAny(0u, ray.origin, ray.direction, 1e20, 1e-5, false);
  _ = closest.didHit || anyHit;
}
`;

const binaryTlasCastMaskGateWgsl = `${binaryTlasCoreWgsl}
${BVH_CAST_SHADOW_MASK_WGSL}
@group(0) @binding(8) var castMask: texture_2d<u32>;
@compute @workgroup_size(1)
fn binaryTlasCastMaskGateMain() {
  _ = traceTlasAnyCastMask(
    0u, vec3f(0.0, 0.0, 1.0), vec3f(0.0, 0.0, -1.0),
    1e20, 1e-5, false, castMask, 1u,
  );
}
`;

const binaryTlasCastPredicateGateWgsl = `${binaryTlasCoreWgsl}
fn bvhCastShadowDisabledForTri(triIdx: u32) -> bool {
  return triIdx == 0xffffffffu;
}
${BVH_CAST_SHADOW_PREDICATE_WGSL}
@compute @workgroup_size(1)
fn binaryTlasCastPredicateGateMain() {
  _ = traceTlasAnyCastPredicate(
    0u, vec3f(0.0, 0.0, 1.0), vec3f(0.0, 0.0, -1.0),
    1e20, 1e-5, false,
  );
}
`;

const cwbvhGateWgsl = `${CWBVH_INTERSECT_WGSL}
@group(0) @binding(0) var<storage, read> cwbvhNodeBounds: array<CwbvhNodeBounds>;
@group(0) @binding(1) var<storage, read> cwbvhChildBoundsPacked: array<u32>;
@group(0) @binding(2) var<storage, read> cwbvhChildMeta: array<CwbvhChildMeta>;
@group(0) @binding(3) var<storage, read> cwbvhChildCount: array<u32>;
@group(0) @binding(4) var<storage, read> bvh_index: array<vec4u>;
@group(0) @binding(5) var<storage, read> bvh_position: array<vec4f>;
@group(0) @binding(6) var<storage, read_write> cwbvhGateOut: array<u32>;

@compute @workgroup_size(1)
fn cwbvhGateMain() {
  var ray: CwbvhRay;
  ray.origin = vec3f(0.0, 0.0, 1.0);
  ray.direction = vec3f(0.0, 0.0, -1.0);
  let closest = cwbvhIntersectFirstHitRangeFromRoot(
    ray, 1e-5, 1e-5, 1e20, 0u, 0u, false,
  );
  cwbvhGateOut[0] = closest.status * 2u + select(0u, 1u, closest.didHit);
}
`;

export const SHARED_BVH_PORTABLE_COMPOSITIONS = Object.freeze([
  Object.freeze({
    name: "shared-bvh/binaryTlas-verbatim",
    code: binaryTlasGateWgsl,
    entryPoint: "binaryTlasGateMain",
  }),
  Object.freeze({
    name: "shared-bvh/binaryTlas-castMask-verbatim",
    code: binaryTlasCastMaskGateWgsl,
    entryPoint: "binaryTlasCastMaskGateMain",
  }),
  Object.freeze({
    name: "shared-bvh/binaryTlas-castPredicate-verbatim",
    code: binaryTlasCastPredicateGateWgsl,
    entryPoint: "binaryTlasCastPredicateGateMain",
  }),
  Object.freeze({
    name: "shared-bvh/cwbvhIntersect-verbatim",
    code: cwbvhGateWgsl,
    entryPoint: "cwbvhGateMain",
  }),
]);
