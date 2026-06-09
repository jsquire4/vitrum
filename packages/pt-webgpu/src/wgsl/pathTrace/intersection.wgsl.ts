/**
 * Intersection module — analytic-shape intersectors + BVH traversal.
 *
 * Bundled here:
 *  - `SceneHit` struct + `SHAPE_*` discriminants
 *  - AABB intersection (`intersectAabb`, `intersectAabbDetailed`)
 *  - Analytic-shape local-frame intersectors:
 *      - `intersectSphereLocal`
 *      - `intersectCylinderLocal`
 *      - `intersectCapsuleLocal`
 *      - `intersectHChannelLocal` (window-came H-channel)
 *  - World↔local transform helpers (`transformPointCols`,
 *    `transformDirectionCols`, `transformNormalFromWorldToLocalCols`)
 *  - BVH traversal wrappers (`traceMeshBvh`, `traceAnalyticShapes`,
 *    `traceClosest`, `traceAny`); TLAS traverse-into-BLAS from `@vitrum/shared-bvh`
 *  - `hitMaterialId` — bridges SceneHit → material payload index for the
 *    main kernel and caustic dispatch.
 *
 * Depends on the FrameParams bindings and material constants declared in
 * `material.wgsl.ts`; the Möller-Trumbore `intersectTriangle` lives in
 * `common.wgsl.ts` and is referenced through the shared global scope.
 */
import { TLAS_SCENE_HIT_TRAVERSAL_WGSL } from '@vitrum/shared-bvh';
import { PT_WEBGPU_INTERSECTION_CORE_WGSL } from './intersectionCore.wgsl.js';

function tlasSceneHitTraversalWithInstanceIndex(wgsl: string): string {
  const withInit = wgsl.replace(
    `  (*hit).normal = vec3f(0.0, 1.0, 0.0);
  var stack: array<u32, 64>;`,
    `  (*hit).normal = vec3f(0.0, 1.0, 0.0);
  (*hit).baryVW = vec2f(0.0);
  (*hit).instanceIndex = INVALID_TLAS_INSTANCE_INDEX;
  var stack: array<u32, 64>;`,
  );
  if (withInit === wgsl) {
    throw new Error('TLAS SceneHit init anchor changed; update pt-webgpu instance-index augmentation.');
  }

  const withAssignment = withInit.replace(
    `          (*hit).triIndex = localHit.triIndex;
          (*hit).normal = transformNormalFromWorldToLocalCols(w2l0, w2l1, w2l2, localHit.normal);
          // Barycentric weights are space-invariant`,
    `          (*hit).triIndex = localHit.triIndex;
          (*hit).normal = transformNormalFromWorldToLocalCols(w2l0, w2l1, w2l2, localHit.normal);
          (*hit).instanceIndex = instIdx;
          // Barycentric weights are space-invariant`,
  );
  if (withAssignment === withInit) {
    throw new Error('TLAS SceneHit assignment anchor changed; update pt-webgpu instance-index augmentation.');
  }

  return withAssignment;
}

export const PT_WEBGPU_PATH_TRACE_INTERSECTION_WGSL = /* wgsl */ `
${PT_WEBGPU_INTERSECTION_CORE_WGSL}

fn traceAnalyticShapes(
  ray: Ray,
  tMin: f32,
  tMaxBound: f32,
  closest: bool,
  hit: ptr<function, SceneHit>,
) -> bool {
  let analyticTotal = min(params.analyticCount, arrayLength(&analyticHeaders));
  for (var ai = 0u; ai < analyticTotal; ai = ai + 1u) {
    let header = analyticHeaders[ai];
    let shapeId = u32(max(header.x, 0.0));
    let paramOffset = u32(max(header.z, 0.0));
    let matBase = ai * 4u;
    if (matBase + 3u >= arrayLength(&analyticWorldToLocal) || matBase + 3u >= arrayLength(&analyticLocalToWorld)) {
      continue;
    }
    let w2l0 = analyticWorldToLocal[matBase];
    let w2l1 = analyticWorldToLocal[matBase + 1u];
    let w2l2 = analyticWorldToLocal[matBase + 2u];
    let w2l3 = analyticWorldToLocal[matBase + 3u];
    let l2w0 = analyticLocalToWorld[matBase];
    let l2w1 = analyticLocalToWorld[matBase + 1u];
    let l2w2 = analyticLocalToWorld[matBase + 2u];
    let l2w3 = analyticLocalToWorld[matBase + 3u];
    var localRay: Ray;
    localRay.origin = transformPointCols(w2l0, w2l1, w2l2, w2l3, ray.origin);
    localRay.direction = transformDirectionCols(w2l0, w2l1, w2l2, ray.direction);
    var localN = vec3f(0.0, 1.0, 0.0);
    var localT = INFINITY;
    let p0 = select(vec4f(0.0), analyticParams[paramOffset], paramOffset < arrayLength(&analyticParams));
    let p1 = select(vec4f(0.0), analyticParams[paramOffset + 1u], paramOffset + 1u < arrayLength(&analyticParams));
    if (shapeId == SHAPE_SPHERE) {
      localT = intersectSphereLocal(localRay, p0.xyz, max(p0.w, 1e-4), &localN);
    } else if (shapeId == SHAPE_BOX) {
      localT = intersectAabbDetailed(localRay, p0.xyz - p1.xyz, p0.xyz + p1.xyz, 1e-4, INFINITY, &localN);
    } else if (shapeId == SHAPE_CAPSULE) {
      localT = intersectCapsuleLocal(localRay, p0.xyz, p1.xyz, max(p1.w, 1e-4), &localN);
    } else if (shapeId == SHAPE_CYLINDER) {
      localT = intersectCylinderLocal(localRay, p0.xyz, max(p0.w, 1e-4), max(p1.x, 1e-4), &localN);
    } else if (shapeId == SHAPE_H_CHANNEL_CAME) {
      localT = intersectHChannelLocal(localRay, p0.x, p0.y, p0.z, p0.w, &localN);
    }
    if (localT <= tMin || localT >= INFINITY) {
      continue;
    }
    let localHitPos = localRay.origin + localRay.direction * localT;
    let worldHitPos = transformPointCols(l2w0, l2w1, l2w2, l2w3, localHitPos);
    let worldT = dot(worldHitPos - ray.origin, ray.direction);
    let bound = select(tMaxBound, (*hit).dist, closest);
    if (worldT > tMin && worldT < bound) {
      if (!closest) {
        return true;
      }
      (*hit).didHit = true;
      (*hit).dist = worldT;
      (*hit).triIndex = params.triangleCount + ai;
      (*hit).normal = transformNormalFromWorldToLocalCols(w2l0, w2l1, w2l2, localN);
      (*hit).baryVW = vec2f(0.0);
      (*hit).instanceIndex = INVALID_TLAS_INSTANCE_INDEX;
    }
  }
  return false;
}

${tlasSceneHitTraversalWithInstanceIndex(TLAS_SCENE_HIT_TRAVERSAL_WGSL)}

fn traceClosest(ray: Ray, tMin: f32, tMax: f32) -> SceneHit {
  var hit: SceneHit;
  _ = traceTlasClosest(ray, tMin, tMax, &hit);
  _ = traceAnalyticShapes(ray, tMin, tMax, true, &hit);
  return hit;
}

fn traceAny(ray: Ray, tMin: f32, tMax: f32) -> bool {
  if (traceTlasAny(ray, tMin, tMax)) {
    return true;
  }
  var hit: SceneHit;
  if (traceAnalyticShapes(ray, tMin, tMax, false, &hit)) {
    return true;
  }
  return false;
}

fn hitMaterialId(hit: SceneHit) -> u32 {
  if (hit.triIndex < params.triangleCount) {
    return select(0u, triMaterialIds[hit.triIndex], hit.triIndex < arrayLength(&triMaterialIds));
  }
  let analyticIndex = hit.triIndex - params.triangleCount;
  if (analyticIndex < arrayLength(&analyticHeaders)) {
    return u32(max(analyticHeaders[analyticIndex].y, 0.0));
  }
  return 0u;
}
`;
