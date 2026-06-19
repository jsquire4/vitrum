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
import {
  CWBVH_INTERSECT_CORE_WGSL,
  TLAS_SCENE_HIT_TRAVERSAL_WGSL,
  TLAS_SCENE_HIT_INIT_ANCHOR,
  TLAS_SCENE_HIT_ASSIGNMENT_ANCHOR,
} from '@vitrum/shared-bvh';
import { PT_WEBGPU_INTERSECTION_CORE_WGSL } from './intersectionCore.wgsl.js';

function tlasSceneHitTraversalWithInstanceIndex(wgsl: string): string {
  // Use the named anchor constants from @vitrum/shared-bvh so any edit to the
  // traversal WGSL is caught at the source (updating the anchor constant) rather
  // than by grep for a stale hardcoded string here.
  const withInit = wgsl.replace(
    TLAS_SCENE_HIT_INIT_ANCHOR,
    `  (*hit).normal = vec3f(0.0, 1.0, 0.0);\n  (*hit).baryVW = vec2f(0.0);\n  (*hit).instanceIndex = INVALID_TLAS_INSTANCE_INDEX;\n  var stack: array<u32, 64>;`,
  );
  if (withInit === wgsl) {
    throw new Error('TLAS SceneHit init anchor changed; update pt-webgpu instance-index augmentation.');
  }

  const withAssignment = withInit.replace(
    TLAS_SCENE_HIT_ASSIGNMENT_ANCHOR,
    `          (*hit).triIndex = localHit.triIndex;\n          (*hit).normal = transformNormalFromWorldToLocalCols(w2l0, w2l1, w2l2, localHit.normal);\n          (*hit).instanceIndex = instIdx;\n          // Barycentric weights are space-invariant`,
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

function replaceWgslFunction(source: string, name: string, replacement: string): string {
  const signature = `fn ${name}(`;
  const start = source.indexOf(signature);
  if (start < 0) {
    throw new Error(`WGSL function ${name} not found`);
  }
  const bodyStart = source.indexOf('{', start);
  if (bodyStart < 0) {
    throw new Error(`WGSL function ${name} has no body`);
  }
  let depth = 0;
  for (let i = bodyStart; i < source.length; i += 1) {
    const ch = source[i];
    if (ch === '{') depth += 1;
    if (ch === '}') {
      depth -= 1;
      if (depth === 0) {
        return `${source.slice(0, start)}${replacement}${source.slice(i + 1)}`;
      }
    }
  }
  throw new Error(`WGSL function ${name} body is unterminated`);
}

function insertBeforeWgslFunction(source: string, name: string, insertion: string): string {
  const signature = `fn ${name}(`;
  const start = source.indexOf(signature);
  if (start < 0) {
    throw new Error(`WGSL function ${name} not found`);
  }
  return `${source.slice(0, start)}${insertion}${source.slice(start)}`;
}

function ptWebgpuCwbvhCoreWgsl(): string {
  let out = CWBVH_INTERSECT_CORE_WGSL;
  const pointerParams = [
    'cwbvhNodeBounds',
    'cwbvhChildBoundsPacked',
    'cwbvhChildMeta',
    'cwbvhChildCount',
    'bvh_index',
    'bvh_position',
  ];
  for (const name of pointerParams) {
    out = out.replace(new RegExp(`\\n  ${name}: ptr<storage, array<[^\\n]+>, read>,`, 'g'), '');
    out = out.replace(new RegExp(`\\n    &${name},`, 'g'), '');
    out = out.replace(new RegExp(`\\n    ${name},`, 'g'), '');
  }
  out = out
    .replace(/cwbvhLoadChildBounds\(\s*cwbvhNodeBounds,\s*cwbvhChildBoundsPacked,\s*/g, 'cwbvhLoadChildBounds(')
    .replace(/\(\*cwbvhNodeBounds\)/g, 'cwbvhNodeBounds')
    .replace(/\(\*cwbvhChildBoundsPacked\)/g, 'cwbvhChildBoundsPacked')
    .replace(/\(\*cwbvhChildMeta\)/g, 'cwbvhChildMeta')
    .replace(/\(\*cwbvhChildCount\)/g, 'cwbvhChildCount')
    .replace(/\(\*bvh_index\)/g, 'indices')
    .replace(/\(\*bvh_position\)/g, 'positions');
  return out;
}

const PT_WEBGPU_CWBVH_BINDINGS_AND_WRAPPERS_WGSL = /* wgsl */ `
${ptWebgpuCwbvhCoreWgsl()}

@group(3) @binding(12) var<storage, read> cwbvhNodeBounds: array<CwbvhNodeBounds>;
@group(3) @binding(13) var<storage, read> cwbvhChildBoundsPacked: array<u32>;
@group(3) @binding(14) var<storage, read> cwbvhChildMeta: array<CwbvhChildMeta>;
@group(3) @binding(15) var<storage, read> cwbvhChildCount: array<u32>;
@group(3) @binding(16) var<storage, read> cwbvhTlasBlasRoots: array<u32>;

fn traceMeshCwbvhClosest(
  ray: Ray,
  tMin: f32,
  tMaxBound: f32,
  hit: ptr<function, SceneHit>,
  rootNode: u32,
  captureShadingDetails: bool,
) -> bool {
  (*hit).didHit = false;
  (*hit).dist = tMaxBound;
  (*hit).triIndex = 0u;
  (*hit).normal = vec3f(0.0, 1.0, 0.0);
  (*hit).baryVW = vec2f(0.0);
  (*hit).instanceIndex = INVALID_TLAS_INSTANCE_INDEX;

  let nodeCount = arrayLength(&cwbvhChildCount);
  if (nodeCount == 0u || rootNode >= nodeCount) {
    return false;
  }
  var cRay: CwbvhRay;
  cRay.origin = ray.origin;
  cRay.direction = ray.direction;
  let cHit = cwbvhIntersectFirstHitRangeFromRoot(
    cRay,
    params.triIntersectEpsilon,
    tMin,
    tMaxBound,
    nodeCount,
    rootNode,
    false,
  );
  if (!cHit.didHit || cHit.dist <= tMin || cHit.dist >= tMaxBound || cHit.triIndex >= min(params.triangleCount, arrayLength(&indices))) {
    return false;
  }

  var shadeNormal = cHit.normal;
  var shadeBaryVW = vec2f(cHit.barycoord.y, cHit.barycoord.z);
  if (captureShadingDetails) {
    let tri = indices[cHit.triIndex];
    if (tri.x < arrayLength(&positions) && tri.y < arrayLength(&positions) && tri.z < arrayLength(&positions)) {
      let a = positions[tri.x].xyz;
      let b = positions[tri.y].xyz;
      let c = positions[tri.z].xyz;
      shadeNormal = safe_normalize(cross(b - a, c - a));
      if (tri.x < arrayLength(&normals) && tri.y < arrayLength(&normals) && tri.z < arrayLength(&normals)) {
        let na = normals[tri.x].xyz;
        let nb = normals[tri.y].xyz;
        let nc = normals[tri.z].xyz;
        shadeNormal = safe_normalize(na * cHit.barycoord.x + nb * cHit.barycoord.y + nc * cHit.barycoord.z);
      }
    }
  }
  (*hit).didHit = true;
  (*hit).dist = cHit.dist;
  (*hit).triIndex = cHit.triIndex;
  (*hit).normal = shadeNormal;
  (*hit).baryVW = shadeBaryVW;
  (*hit).instanceIndex = INVALID_TLAS_INSTANCE_INDEX;
  return true;
}

fn traceMeshCwbvhAny(
  ray: Ray,
  tMin: f32,
  tMaxBound: f32,
  rootNode: u32,
) -> bool {
  let nodeCount = arrayLength(&cwbvhChildCount);
  if (nodeCount == 0u || rootNode >= nodeCount) {
    return false;
  }

  var stack: array<u32, CWBVH_INTERSECT_STACK_DEPTH>;
  var stackPtr = 0u;
  stack[stackPtr] = rootNode;
  stackPtr = stackPtr + 1u;
  let invDir = safeInvDir(ray.direction);

  while (stackPtr > 0u) {
    stackPtr = stackPtr - 1u;
    let nodeIndex = stack[stackPtr];
    if (nodeIndex >= nodeCount) {
      continue;
    }
    let count = min(cwbvhChildCount[nodeIndex], CWBVH_CHILDREN);
    for (var slot = 0u; slot < count; slot = slot + 1u) {
      let childIndex = nodeIndex * CWBVH_CHILDREN + slot;
      let childInfo = cwbvhChildMeta[childIndex];
      if (childInfo.kind == CWBVH_CHILD_EMPTY) {
        continue;
      }
      let bounds = cwbvhLoadChildBounds(nodeIndex, slot);
      let childT = cwbvhAabbEntry(ray.origin, invDir, bounds.boundsMin, bounds.boundsMax, tMaxBound);
      if (childT == CWBVH_INTERSECT_INFINITY) {
        continue;
      }
      if (childInfo.kind == CWBVH_CHILD_NODE) {
        if (stackPtr >= CWBVH_INTERSECT_STACK_DEPTH) {
          // Conservative any-hit overflow policy: prefer occlusion over a light leak.
          return true;
        }
        stack[stackPtr] = childInfo.indexOrOffset;
        stackPtr = stackPtr + 1u;
      } else if (childInfo.kind == CWBVH_CHILD_LEAF) {
        for (var i = 0u; i < childInfo.triCount; i = i + 1u) {
          let triIdx = childInfo.indexOrOffset + i;
          if (triIdx >= min(params.triangleCount, arrayLength(&indices))) {
            continue;
          }
          if (triShadowCastDisabled(triIdx)) {
            continue;
          }
          let tri = indices[triIdx];
          if (tri.x >= arrayLength(&positions) || tri.y >= arrayLength(&positions) || tri.z >= arrayLength(&positions)) {
            continue;
          }
          let a = positions[tri.x].xyz;
          let b = positions[tri.y].xyz;
          let c = positions[tri.z].xyz;
          let hitT = intersectTriangle(ray.origin, ray.direction, a, b, c);
          if (hitT > tMin && hitT < tMaxBound) {
            return true;
          }
        }
      }
    }
  }
  return false;
}

fn traceTlasClosestCwbvh(ray: Ray, tMin: f32, tMax: f32, hit: ptr<function, SceneHit>) -> bool {
  if (params.tlasNodeCount == 0u || arrayLength(&tlasNodes) == 0u || arrayLength(&tlasInstanceIndices) == 0u) {
    return traceMeshCwbvhClosest(ray, tMin, tMax, hit, 0u, true);
  }
  (*hit).didHit = false;
  (*hit).dist = tMax;
  (*hit).triIndex = 0u;
  (*hit).normal = vec3f(0.0, 1.0, 0.0);
  (*hit).baryVW = vec2f(0.0);
  (*hit).instanceIndex = INVALID_TLAS_INSTANCE_INDEX;
  var stack: array<u32, 64>;
  var stackPtr = 0u;
  stack[stackPtr] = 0u;
  stackPtr = stackPtr + 1u;
  while (stackPtr > 0u) {
    stackPtr = stackPtr - 1u;
    let nodeIdx = stack[stackPtr];
    if (nodeIdx >= min(params.tlasNodeCount, arrayLength(&tlasNodes))) { continue; }
    let node = tlasNodes[nodeIdx];
    let bmin = vec3f(node.boundsMin[0], node.boundsMin[1], node.boundsMin[2]);
    let bmax = vec3f(node.boundsMax[0], node.boundsMax[1], node.boundsMax[2]);
    if (!intersectAabb(ray, bmin, bmax, tMin, (*hit).dist)) { continue; }
    let splitOrCount = node.splitAxisOrTriCount;
    if ((splitOrCount & LEAFNODE_FLAG) == LEAFNODE_FLAG) {
      let count = splitOrCount & 0x0000ffffu;
      let start = node.rightChildOrTriOffset;
      for (var i = 0u; i < count; i = i + 1u) {
        let permIdx = start + i;
        if (permIdx >= arrayLength(&tlasInstanceIndices)) { continue; }
        let instIdx = tlasInstanceIndices[permIdx];
        let m = instIdx * 4u;
        if (m + 3u >= arrayLength(&tlasInstanceWorldToLocal) || m + 3u >= arrayLength(&tlasInstanceLocalToWorld)) { continue; }
        let w2l0 = tlasInstanceWorldToLocal[m];
        let w2l1 = tlasInstanceWorldToLocal[m + 1u];
        let w2l2 = tlasInstanceWorldToLocal[m + 2u];
        let w2l3 = tlasInstanceWorldToLocal[m + 3u];
        let l2w0 = tlasInstanceLocalToWorld[m];
        let l2w1 = tlasInstanceLocalToWorld[m + 1u];
        let l2w2 = tlasInstanceLocalToWorld[m + 2u];
        let l2w3 = tlasInstanceLocalToWorld[m + 3u];
        var localRay: Ray;
        localRay.origin = transformPointCols(w2l0, w2l1, w2l2, w2l3, ray.origin);
        localRay.direction = transformDirectionCols(w2l0, w2l1, w2l2, ray.direction);
        var localHit: SceneHit;
        let blasRoot = select(0u, cwbvhTlasBlasRoots[instIdx], instIdx < arrayLength(&cwbvhTlasBlasRoots));
        _ = traceMeshCwbvhClosest(localRay, tMin, INFINITY, &localHit, blasRoot, true);
        if (localHit.didHit && localHit.dist > tMin && localHit.dist < (*hit).dist) {
          let localHitPos = localRay.origin + localRay.direction * localHit.dist;
          let worldHitPos = transformPointCols(l2w0, l2w1, l2w2, l2w3, localHitPos);
          let worldDist = dot(worldHitPos - ray.origin, ray.direction);
          if (worldDist <= tMin || worldDist >= (*hit).dist) { continue; }
          (*hit).didHit = true;
          (*hit).dist = worldDist;
          (*hit).triIndex = localHit.triIndex;
          (*hit).normal = transformNormalFromWorldToLocalCols(w2l0, w2l1, w2l2, localHit.normal);
          (*hit).instanceIndex = instIdx;
          (*hit).baryVW = localHit.baryVW;
        }
      }
    } else {
      let leftChild = nodeIdx + 1u;
      let rightChild = nodeIdx + node.rightChildOrTriOffset;
      if (stackPtr + 2u < 64u) {
        stack[stackPtr] = rightChild; stackPtr = stackPtr + 1u;
        stack[stackPtr] = leftChild; stackPtr = stackPtr + 1u;
      } else {
        return (*hit).didHit;
      }
    }
  }
  return (*hit).didHit;
}

fn traceTlasAnyCwbvh(ray: Ray, tMin: f32, tMax: f32) -> bool {
  if (params.tlasNodeCount == 0u || arrayLength(&tlasNodes) == 0u || arrayLength(&tlasInstanceIndices) == 0u) {
    return traceMeshCwbvhAny(ray, tMin, tMax, 0u);
  }
  var stack: array<u32, 64>;
  var stackPtr = 0u;
  stack[stackPtr] = 0u;
  stackPtr = stackPtr + 1u;
  while (stackPtr > 0u) {
    stackPtr = stackPtr - 1u;
    let nodeIdx = stack[stackPtr];
    if (nodeIdx >= min(params.tlasNodeCount, arrayLength(&tlasNodes))) { continue; }
    let node = tlasNodes[nodeIdx];
    let bmin = vec3f(node.boundsMin[0], node.boundsMin[1], node.boundsMin[2]);
    let bmax = vec3f(node.boundsMax[0], node.boundsMax[1], node.boundsMax[2]);
    if (!intersectAabb(ray, bmin, bmax, tMin, tMax)) { continue; }
    let splitOrCount = node.splitAxisOrTriCount;
    if ((splitOrCount & LEAFNODE_FLAG) == LEAFNODE_FLAG) {
      let count = splitOrCount & 0x0000ffffu;
      let start = node.rightChildOrTriOffset;
      for (var i = 0u; i < count; i = i + 1u) {
        let permIdx = start + i;
        if (permIdx >= arrayLength(&tlasInstanceIndices)) { continue; }
        let instIdx = tlasInstanceIndices[permIdx];
        let m = instIdx * 4u;
        if (m + 3u >= arrayLength(&tlasInstanceWorldToLocal)) { continue; }
        let w2l0 = tlasInstanceWorldToLocal[m];
        let w2l1 = tlasInstanceWorldToLocal[m + 1u];
        let w2l2 = tlasInstanceWorldToLocal[m + 2u];
        let w2l3 = tlasInstanceWorldToLocal[m + 3u];
        var localRay: Ray;
        localRay.origin = transformPointCols(w2l0, w2l1, w2l2, w2l3, ray.origin);
        localRay.direction = transformDirectionCols(w2l0, w2l1, w2l2, ray.direction);
        var localTMax = tMax;
        if (tMax < INFINITY * 0.5) {
          let localEnd = transformPointCols(w2l0, w2l1, w2l2, w2l3, ray.origin + ray.direction * tMax);
          localTMax = max(dot(localEnd - localRay.origin, localRay.direction), tMin);
        }
        let blasRoot = select(0u, cwbvhTlasBlasRoots[instIdx], instIdx < arrayLength(&cwbvhTlasBlasRoots));
        if (traceMeshCwbvhAny(localRay, tMin, localTMax, blasRoot)) {
          return true;
        }
      }
    } else {
      let leftChild = nodeIdx + 1u;
      let rightChild = nodeIdx + node.rightChildOrTriOffset;
      if (stackPtr + 2u < 64u) {
        stack[stackPtr] = rightChild; stackPtr = stackPtr + 1u;
        stack[stackPtr] = leftChild; stackPtr = stackPtr + 1u;
      } else {
        // Conservative any-hit overflow policy: prefer occlusion over a light leak.
        return true;
      }
    }
  }
  return false;
}
`;

const PT_WEBGPU_TRACE_CLOSEST_CWBVH_WGSL = /* wgsl */ `fn traceClosest(ray: Ray, tMin: f32, tMax: f32) -> SceneHit {
  var hit: SceneHit;
  _ = traceTlasClosestCwbvh(ray, tMin, tMax, &hit);
  _ = traceAnalyticShapes(ray, tMin, tMax, true, &hit);
  return hit;
}
`;

const PT_WEBGPU_TRACE_ANY_CWBVH_WGSL = /* wgsl */ `fn traceAny(ray: Ray, tMin: f32, tMax: f32) -> bool {
  if (traceTlasAnyCwbvh(ray, tMin, tMax)) {
    return true;
  }
  var hit: SceneHit;
  if (traceAnalyticShapes(ray, tMin, tMax, false, &hit)) {
    return true;
  }
  return false;
}
`;

export interface PtWebgpuIntersectionComposeOptions {
  readonly cwbvhClosest?: boolean;
}

export function composePtWebgpuPathTraceIntersectionWgsl(
  opts: PtWebgpuIntersectionComposeOptions = {},
): string {
  if (opts.cwbvhClosest !== true) {
    return PT_WEBGPU_PATH_TRACE_INTERSECTION_WGSL;
  }
  const withCwbvh = insertBeforeWgslFunction(
    PT_WEBGPU_PATH_TRACE_INTERSECTION_WGSL,
    'traceClosest',
    PT_WEBGPU_CWBVH_BINDINGS_AND_WRAPPERS_WGSL,
  );
  const withCwbvhClosest = replaceWgslFunction(withCwbvh, 'traceClosest', PT_WEBGPU_TRACE_CLOSEST_CWBVH_WGSL);
  return replaceWgslFunction(withCwbvhClosest, 'traceAny', PT_WEBGPU_TRACE_ANY_CWBVH_WGSL);
}
