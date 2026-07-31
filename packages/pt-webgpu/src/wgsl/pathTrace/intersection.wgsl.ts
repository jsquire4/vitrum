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
    `  (*hit).normal = vec3f(0.0, 1.0, 0.0);\n  (*hit).frontFace = false;\n  (*hit).baryVW = vec2f(0.0);\n  (*hit).instanceIndex = INVALID_TLAS_INSTANCE_INDEX;\n}`,
  );
  if (withInit === wgsl) {
    throw new Error('TLAS SceneHit init anchor changed; update pt-webgpu instance-index augmentation.');
  }

  const withAssignment = withInit.replace(
    TLAS_SCENE_HIT_ASSIGNMENT_ANCHOR,
    `      (*hit).triIndex = localHit.triIndex;\n      (*hit).normal = transformNormalFromWorldToLocalCols(w2l0, w2l1, w2l2, localHit.normal);\n      let orientationPreserving = transformLinearOrientationSign(l2w0, l2w1, l2w2) > 0.0;\n      (*hit).frontFace = select(!localHit.frontFace, localHit.frontFace, orientationPreserving);\n      (*hit).instanceIndex = instIdx;\n      // Barycentric weights are space-invariant`,
  );
  if (withAssignment === withInit) {
    throw new Error('TLAS SceneHit assignment anchor changed; update pt-webgpu instance-index augmentation.');
  }

  // The shared traversal intentionally supports a root-0 merged-BVH fallback.
  // That fallback cannot provide a primitive/instance identity: entry and exit
  // faces of one closed mesh have different triangle ids, so a triangle id is
  // not a valid medium-boundary token.  The full pt-webgpu tier always uploads
  // a TLAS for renderable mesh instances; when every instance is deliberately
  // skipped (for example because all transforms are singular), fail closed on
  // mesh traversal instead of resurrecting untransformed root-0 geometry.
  const closestFallback = `  if (params.tlasNodeCount == 0u || arrayLength(&tlasNodes) == 0u || arrayLength(&tlasInstanceIndices) == 0u) {
    return traceMeshBvh(ray, tMin, tMax, true, hit, 0u, true);
  }`;
  const closestFailClosed = `  if (params.tlasNodeCount == 0u || arrayLength(&tlasNodes) == 0u || arrayLength(&tlasInstanceIndices) == 0u) {
    tlasResetSceneHit(hit, tMax);
    return false;
  }`;
  const withClosestFailClosed = withAssignment.replace(
    closestFallback,
    closestFailClosed,
  );
  if (withClosestFailClosed === withAssignment) {
    throw new Error(
      'TLAS SceneHit closest root-0 fallback changed; update pt-webgpu fail-closed traversal.',
    );
  }

  const anyFallback = `  if (params.tlasNodeCount == 0u || arrayLength(&tlasNodes) == 0u || arrayLength(&tlasInstanceIndices) == 0u) {
    var meshHit: SceneHit;
    return traceMeshBvh(ray, tMin, tMax, false, &meshHit, 0u, false);
  }`;
  const anyFailClosed = `  if (params.tlasNodeCount == 0u || arrayLength(&tlasNodes) == 0u || arrayLength(&tlasInstanceIndices) == 0u) {
    return false;
  }`;
  const withAnyFailClosed = withClosestFailClosed.replace(
    anyFallback,
    anyFailClosed,
  );
  if (withAnyFailClosed === withClosestFailClosed) {
    throw new Error(
      'TLAS SceneHit any-hit root-0 fallback changed; update pt-webgpu fail-closed traversal.',
    );
  }

  return withAnyFailClosed;
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
    let materialId = u32(max(header.y, 0.0));
    let paramOffset = u32(max(header.z, 0.0));
    if (!closest && materialShadowCastDisabled(materialId)) {
      continue;
    }
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
    let localDirectionScale = max(
      abs(localRay.direction.x),
      max(abs(localRay.direction.y), abs(localRay.direction.z)),
    );
    if (!(localDirectionScale > 0.0) || localDirectionScale > 3.402823e38) {
      continue;
    }
    var localN = vec3f(0.0, 1.0, 0.0);
    var localT = INFINITY;
    let p0 = select(vec4f(0.0), analyticParams[paramOffset], paramOffset < arrayLength(&analyticParams));
    let p1 = select(vec4f(0.0), analyticParams[paramOffset + 1u], paramOffset + 1u < arrayLength(&analyticParams));
    if (shapeId == SHAPE_SPHERE) {
      localT = intersectSphereLocal(localRay, p0.xyz, p0.w, &localN);
    } else if (shapeId == SHAPE_BOX && all(p1.xyz > vec3f(0.0))) {
      localT = intersectAabbDetailed(
        localRay, p0.xyz - p1.xyz, p0.xyz + p1.xyz, 0.0, INFINITY, &localN,
      );
    } else if (shapeId == SHAPE_CAPSULE) {
      localT = intersectCapsuleLocal(localRay, p0.xyz, p1.xyz, p1.w, &localN);
    } else if (shapeId == SHAPE_CYLINDER) {
      localT = intersectCylinderLocal(localRay, p0.xyz, p0.w, p1.x, &localN);
    } else if (shapeId == SHAPE_H_CHANNEL_CAME) {
      localT = intersectHChannelLocal(localRay, p0.x, p0.y, p0.z, p0.w, &localN);
    }
    if (!(localT > 0.0) || localT >= INFINITY) {
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
      (*hit).frontFace = dot((*hit).normal, ray.direction) < 0.0;
      (*hit).baryVW = vec2f(0.0);
      (*hit).instanceIndex = INVALID_TLAS_INSTANCE_INDEX;
    }
  }
  return false;
}

${tlasSceneHitTraversalWithInstanceIndex(TLAS_SCENE_HIT_TRAVERSAL_WGSL)}

fn traceClosestRaw(ray: Ray, tMin: f32, tMax: f32) -> SceneHit {
  var hit: SceneHit;
  _ = traceTlasClosest(ray, tMin, tMax, &hit);
  _ = traceAnalyticShapes(ray, tMin, tMax, true, &hit);
  return hit;
}

fn nextSidedTraversalCursor(cursor: f32, hitDist: f32) -> f32 {
  let step = max(
    max(params.triIntersectEpsilon, 1.175494351e-38),
    abs(hitDist) * (4.0 * 1.192092896e-7),
  );
  let fromHit = hitDist + step;
  return select(fromHit, cursor + step, !(fromHit > cursor));
}

fn traceClosest(ray: Ray, tMin: f32, tMax: f32) -> SceneHit {
  var cursor = tMin;
  var hit = traceClosestRaw(ray, cursor, tMax);
  loop {
    if (!hit.didHit) { return hit; }
    let matId = hitMaterialId(hit);
    if (materialAcceptsSidedHit(matId, hit.frontFace)) { return hit; }
    let nextCursor = nextSidedTraversalCursor(cursor, hit.dist);
    if (!(nextCursor > cursor) || !(nextCursor < tMax)) {
      hit.didHit = false;
      return hit;
    }
    cursor = nextCursor;
    hit = traceClosestRaw(ray, cursor, tMax);
  }
  return hit;
}

fn traceAny(
  ray: Ray,
  tMin: f32,
  tMax: f32,
  rng: ptr<function, PtRngState>,
) -> bool {
  var cursor = tMin;
  loop {
    let hit = traceClosestRaw(ray, cursor, tMax);
    if (!hit.didHit) { return false; }
    let matId = hitMaterialId(hit);
    if (
      materialAcceptsSidedHit(matId, hit.frontFace) &&
      !materialShadowCastDisabled(matId) &&
      !alphaTestPassThrough(
        matId, hit.triIndex, hit.baryVW, hit.instanceIndex, rng,
      )
    ) {
      return true;
    }
    let nextCursor = nextSidedTraversalCursor(cursor, hit.dist);
    if (!(nextCursor > cursor) || !(nextCursor < tMax)) { return false; }
    cursor = nextCursor;
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

const MEDIUM_BOUNDARY_KIND_TLAS: u32 = 0u;
const MEDIUM_BOUNDARY_KIND_ANALYTIC: u32 = 1u;
const MEDIUM_BOUNDARY_KIND_INVALID: u32 = 0xffffffffu;

fn mediumBoundaryIdentity(triIndex: u32, instanceIndex: u32) -> vec2u {
  if (instanceIndex != INVALID_TLAS_INSTANCE_INDEX) {
    return vec2u(MEDIUM_BOUNDARY_KIND_TLAS, instanceIndex);
  }
  if (triIndex >= params.triangleCount) {
    let analyticIndex = triIndex - params.triangleCount;
    let analyticTotal = min(params.analyticCount, arrayLength(&analyticHeaders));
    if (analyticIndex < analyticTotal) {
      return vec2u(MEDIUM_BOUNDARY_KIND_ANALYTIC, analyticIndex);
    }
  }
  // A merged mesh hit has no primitive/instance identity. A triangle id cannot
  // stand in for a closed boundary because entry and exit use different faces.
  return vec2u(MEDIUM_BOUNDARY_KIND_INVALID);
}

fn mediumBoundaryIsValid(boundary: vec2u) -> bool {
  return boundary.x != MEDIUM_BOUNDARY_KIND_INVALID;
}

fn mediumBoundaryMatches(kind: u32, index: u32, boundary: vec2u) -> bool {
  return mediumBoundaryIsValid(boundary) &&
    kind == boundary.x && index == boundary.y;
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

const PT_WEBGPU_CWBVH_BINDINGS_AND_WRAPPERS_WGSL = /* wgsl */ `
${CWBVH_INTERSECT_CORE_WGSL}

@group(3) @binding(12) var<storage, read> cwbvhNodeBounds: array<CwbvhNodeBounds>;
@group(3) @binding(13) var<storage, read> cwbvhChildBoundsPacked: array<u32>;
@group(3) @binding(14) var<storage, read> cwbvhChildMeta: array<CwbvhChildMeta>;
@group(3) @binding(15) var<storage, read> cwbvhChildCount: array<u32>;
@group(3) @binding(16) var<storage, read> cwbvhTlasBlasRoots: array<vec4u>;

fn cwbvhLoadNodeBounds(index: u32) -> CwbvhNodeBounds { return cwbvhNodeBounds[index]; }
fn cwbvhNodeBoundsCount() -> u32 { return arrayLength(&cwbvhNodeBounds); }
fn cwbvhLoadChildBoundsWord(index: u32) -> u32 { return cwbvhChildBoundsPacked[index]; }
fn cwbvhChildBoundsWordCount() -> u32 { return arrayLength(&cwbvhChildBoundsPacked); }
fn cwbvhLoadChildMeta(index: u32) -> CwbvhChildMeta { return cwbvhChildMeta[index]; }
fn cwbvhChildMetaCount() -> u32 { return arrayLength(&cwbvhChildMeta); }
fn cwbvhLoadChildCount(index: u32) -> u32 { return cwbvhChildCount[index]; }
fn cwbvhChildCountCount() -> u32 { return arrayLength(&cwbvhChildCount); }
fn cwbvhLoadIndex(index: u32) -> vec4u { return indices[index]; }
fn cwbvhIndexCount() -> u32 { return arrayLength(&indices); }
fn cwbvhLoadPosition(index: u32) -> vec4f { return positions[index]; }
fn cwbvhPositionCount() -> u32 { return arrayLength(&positions); }

const CWBVH_ROOT_PAIR_MAGIC = 0x43574256u;
const CWBVH_BINARY_ROOT_FACTOR = 0x9e3779b1u;
const CWBVH_WIDE_ROOT_FACTOR = 0x85ebca6bu;

fn traceMeshCwbvhClosest(
  ray: Ray,
  tMin: f32,
  tMaxBound: f32,
  hit: ptr<function, SceneHit>,
  rootNode: u32,
  binaryRootNode: u32,
  captureShadingDetails: bool,
) -> bool {
  (*hit).didHit = false;
  (*hit).dist = tMaxBound;
  (*hit).triIndex = 0u;
  (*hit).normal = vec3f(0.0, 1.0, 0.0);
  (*hit).frontFace = false;
  (*hit).baryVW = vec2f(0.0);
  (*hit).instanceIndex = INVALID_TLAS_INSTANCE_INDEX;

  let nodeCount = arrayLength(&cwbvhChildCount);
  if (nodeCount == 0u || rootNode >= nodeCount) {
    return traceMeshBvh(ray, tMin, tMaxBound, true, hit, binaryRootNode, captureShadingDetails);
  }
  var cRay: CwbvhRay;
  cRay.origin = ray.origin;
  cRay.direction = ray.direction;
  let cHit = cwbvhIntersectFirstHitRangeFromRoot(
    cRay,
    tMin,
    tMaxBound,
    nodeCount,
    rootNode,
    false,
  );
  if (cHit.status != CWBVH_STATUS_COMPLETE) {
    return traceMeshBvh(ray, tMin, tMaxBound, true, hit, binaryRootNode, captureShadingDetails);
  }
  if (!cHit.didHit || cHit.dist <= tMin || cHit.dist >= tMaxBound) {
    return false;
  }
  if (cHit.triIndex >= min(params.triangleCount, arrayLength(&indices))) {
    return traceMeshBvh(ray, tMin, tMaxBound, true, hit, binaryRootNode, captureShadingDetails);
  }

  // CWBVH stores its geometric normal oriented against the incident ray;
  // SceneHit keeps the authored-winding normal and records side separately.
  var shadeNormal = cHit.normal * cHit.side;
  var shadeBaryVW = vec2f(cHit.barycoord.y, cHit.barycoord.z);
  if (captureShadingDetails) {
    let tri = indices[cHit.triIndex];
    if (tri.x < arrayLength(&positions) && tri.y < arrayLength(&positions) && tri.z < arrayLength(&positions)) {
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
  (*hit).frontFace = cHit.side > 0.0;
  (*hit).baryVW = shadeBaryVW;
  (*hit).instanceIndex = INVALID_TLAS_INSTANCE_INDEX;
  return true;
}

fn traceTlasClosestCwbvhFallback(
  ray: Ray,
  tMin: f32,
  tMax: f32,
  hit: ptr<function, SceneHit>,
) -> bool {
  let fallbackHit = traceTlasClosest(ray, tMin, tMax, hit);
  tlasTraversalStatusCode = TLAS_TRAVERSAL_STATUS_FALLBACK;
  return fallbackHit;
}

fn traceTlasClosestCwbvh(ray: Ray, tMin: f32, tMax: f32, hit: ptr<function, SceneHit>) -> bool {
  tlasTraversalStatusCode = TLAS_TRAVERSAL_STATUS_COMPLETE;
  if (params.tlasNodeCount == 0u || arrayLength(&tlasNodes) == 0u || arrayLength(&tlasInstanceIndices) == 0u) {
    return traceTlasClosest(ray, tMin, tMax, hit);
  }
  (*hit).didHit = false;
  (*hit).dist = tMax;
  (*hit).triIndex = 0u;
  (*hit).normal = vec3f(0.0, 1.0, 0.0);
  (*hit).frontFace = false;
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
        let localStart = transformPointCols(
          w2l0, w2l1, w2l2, w2l3, ray.origin + ray.direction * tMin,
        );
        let localTMin = max(dot(localStart - localRay.origin, localRay.direction), 0.0);
        var localTMax = INFINITY;
        if ((*hit).dist < INFINITY * 0.5) {
          let localEnd = transformPointCols(
            w2l0, w2l1, w2l2, w2l3, ray.origin + ray.direction * (*hit).dist,
          );
          localTMax = max(dot(localEnd - localRay.origin, localRay.direction), localTMin);
        }
        var localHit: SceneHit;
        if (instIdx >= arrayLength(&cwbvhTlasBlasRoots) || instIdx >= arrayLength(&tlasBlasRoots)) {
          return traceTlasClosestCwbvhFallback(ray, tMin, tMax, hit);
        }
        let rootPair = cwbvhTlasBlasRoots[instIdx];
        if (
          rootPair.x != CWBVH_ROOT_PAIR_MAGIC ||
          rootPair.y == 0xffffffffu || rootPair.z == 0xffffffffu ||
          rootPair.y != tlasBlasRoots[instIdx] ||
          rootPair.y >= min(params.bvhNodeCount, arrayLength(&bvhNodes)) ||
          rootPair.z >= arrayLength(&cwbvhChildCount) ||
          rootPair.w != (rootPair.x ^ rootPair.y * CWBVH_BINARY_ROOT_FACTOR ^ rootPair.z * CWBVH_WIDE_ROOT_FACTOR)
        ) {
          return traceTlasClosestCwbvhFallback(ray, tMin, tMax, hit);
        }
        let binaryBlasRoot = rootPair.y;
        let blasRoot = rootPair.z;
        _ = traceMeshCwbvhClosest(localRay, localTMin, localTMax, &localHit, blasRoot, binaryBlasRoot, true);
        if (localHit.didHit && localHit.dist > localTMin) {
          let localHitPos = localRay.origin + localRay.direction * localHit.dist;
          let worldHitPos = transformPointCols(l2w0, l2w1, l2w2, l2w3, localHitPos);
          let worldDist = dot(worldHitPos - ray.origin, ray.direction);
          if (worldDist <= tMin || worldDist >= (*hit).dist) { continue; }
          (*hit).didHit = true;
          (*hit).dist = worldDist;
          (*hit).triIndex = localHit.triIndex;
          (*hit).normal = transformNormalFromWorldToLocalCols(w2l0, w2l1, w2l2, localHit.normal);
          let orientationPreserving =
            transformLinearOrientationSign(l2w0, l2w1, l2w2) > 0.0;
          (*hit).frontFace = select(!localHit.frontFace, localHit.frontFace, orientationPreserving);
          (*hit).instanceIndex = instIdx;
          (*hit).baryVW = localHit.baryVW;
        }
      }
    } else {
      let leftChild = nodeIdx + 1u;
      let rightChild = nodeIdx + node.rightChildOrTriOffset;
      if (stackPtr + 2u <= 64u) {
        stack[stackPtr] = rightChild; stackPtr = stackPtr + 1u;
        stack[stackPtr] = leftChild; stackPtr = stackPtr + 1u;
      } else {
        return traceTlasClosestCwbvhFallback(ray, tMin, tMax, hit);
      }
    }
  }
  return (*hit).didHit;
}

`;

const PT_WEBGPU_TRACE_CLOSEST_CWBVH_WGSL = /* wgsl */ `fn traceClosestRaw(ray: Ray, tMin: f32, tMax: f32) -> SceneHit {
  var hit: SceneHit;
  _ = traceTlasClosestCwbvh(ray, tMin, tMax, &hit);
  _ = traceAnalyticShapes(ray, tMin, tMax, true, &hit);
  return hit;
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
    'traceClosestRaw',
    PT_WEBGPU_CWBVH_BINDINGS_AND_WRAPPERS_WGSL,
  );
  return replaceWgslFunction(withCwbvh, 'traceClosestRaw', PT_WEBGPU_TRACE_CLOSEST_CWBVH_WGSL);
}
