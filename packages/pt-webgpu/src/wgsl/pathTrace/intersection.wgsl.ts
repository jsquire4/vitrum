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
    `  (*hit).position = vec3f(0.0);\n  (*hit).normal = vec3f(0.0, 1.0, 0.0);\n  (*hit).frontFace = false;\n  (*hit).baryVW = vec2f(0.0);\n  (*hit).instanceIndex = INVALID_TLAS_INSTANCE_INDEX;\n  (*hit).zeroEdgeMask = 0u;\n}`,
  );
  if (withInit === wgsl) {
    throw new Error('TLAS SceneHit init anchor changed; update pt-webgpu instance-index augmentation.');
  }

  const withAssignment = withInit.replace(
    TLAS_SCENE_HIT_ASSIGNMENT_ANCHOR,
    `      (*hit).triIndex = localHit.triIndex;\n      (*hit).position = worldHitPos;\n      (*hit).normal = transformNormalFromWorldToLocalCols(w2l0, w2l1, w2l2, localHit.normal);\n      let orientationPreserving = transformLinearOrientationSign(l2w0, l2w1, l2w2) > 0.0;\n      (*hit).frontFace = select(!localHit.frontFace, localHit.frontFace, orientationPreserving);\n      (*hit).instanceIndex = instIdx;\n      (*hit).zeroEdgeMask = localHit.zeroEdgeMask;\n      // Barycentric weights are space-invariant`,
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

  const closestCall =
    `_ = traceMeshBvh(localRay, localTMin, localTMax, true, &localHit, blasRoot, true);`;
  const closestWithContext =
    `opticalSetTraversalInstance(instIdx);\n    ${closestCall}`;
  const withClosestContext = withAnyFailClosed.replace(
    closestCall,
    closestWithContext,
  );
  if (withClosestContext === withAnyFailClosed) {
    throw new Error(
      'TLAS closest BLAS call changed; update pt-webgpu source-feature context routing.',
    );
  }

  const anyCall =
    `return traceMeshBvh(localRay, localTMin, localTMax, false, &localHit, blasRoot, false);`;
  const anyWithContext =
    `opticalSetTraversalInstance(instIdx);\n  ${anyCall}`;
  const withAnyContext = withClosestContext.replace(anyCall, anyWithContext);
  if (withAnyContext === withClosestContext) {
    throw new Error(
      'TLAS any-hit BLAS call changed; update pt-webgpu source-feature context routing.',
    );
  }

  const localMinLine =
    `  let localTMin = max(dot(localStart - localRay.origin, localRay.direction), 0.0);`;
  const localMinWithContinuation =
    `  let localTMin = select(\n    max(dot(localStart - localRay.origin, localRay.direction), 0.0),\n    0.0,\n    opticalContinuationSourceIsActive(),\n  );`;
  const withConservativeContinuationMin = withAnyContext.replaceAll(
    localMinLine,
    localMinWithContinuation,
  );
  if (
    withConservativeContinuationMin === withAnyContext ||
    withConservativeContinuationMin.includes(localMinLine)
  ) {
    throw new Error(
      'TLAS local lower-bound conversion changed; update exact continuation routing.',
    );
  }

  const reconstructedLocalHitPoint =
    `    let localHitPos = localRay.origin + localRay.direction * localHit.dist;`;
  const representedLocalHitPoint =
    `    let localHitPos = localHit.position;`;
  const withRepresentedHitPoint = withConservativeContinuationMin.replaceAll(
    reconstructedLocalHitPoint,
    representedLocalHitPoint,
  );
  if (
    withRepresentedHitPoint === withConservativeContinuationMin ||
    withRepresentedHitPoint.includes(reconstructedLocalHitPoint)
  ) {
    throw new Error(
      'TLAS local hit-point reconstruction changed; update represented-point routing.',
    );
  }

  return withRepresentedHitPoint;
}

export const PT_WEBGPU_PATH_TRACE_INTERSECTION_WGSL = /* wgsl */ `
fn opticalEncodedBoundaryId(triIndex: u32, instanceIndex: u32) -> u32 {
  if (
    triIndex >= min(params.triangleCount, arrayLength(&indices)) ||
    instanceIndex == INVALID_TLAS_INSTANCE_INDEX ||
    instanceIndex >= arrayLength(&opticalInstanceBoundaryIdBasePlusOne)
  ) {
    return 0u;
  }
  let componentPlusOne = indices[triIndex].w;
  let basePlusOne = opticalInstanceBoundaryIdBasePlusOne[instanceIndex];
  if (
    componentPlusOne == 0u || basePlusOne == 0u ||
    componentPlusOne - 1u > 0xffffffffu - basePlusOne
  ) {
    return 0u;
  }
  return basePlusOne + componentPlusOne - 1u;
}

var<private> opticalContinuationSource: OpticalSourceFeature;
var<private> opticalTraversalInstanceIndex: u32;

fn opticalSetTraversalInstance(instanceIndex: u32) {
  opticalTraversalInstanceIndex = instanceIndex;
}

fn opticalContinuationSourceIsActive() -> bool {
  return opticalContinuationSource.kind != OPTICAL_SOURCE_FEATURE_INVALID;
}

fn opticalClearContinuationSource() {
  opticalContinuationSource = opticalSourceFeatureInvalid();
}

fn opticalTraversalSuppressesTriangle(
  triangleIndex: u32,
  tri: vec4u,
  a: vec3f,
  b: vec3f,
  c: vec3f,
) -> bool {
  if (
    !opticalContinuationSourceIsActive() ||
    opticalTraversalInstanceIndex == INVALID_TLAS_INSTANCE_INDEX ||
    opticalTraversalInstanceIndex == 0xffffffffu
  ) {
    return false;
  }
  let representedId = opticalTraversalInstanceIndex + 1u;
  return opticalSourceFeatureSuppressesTriangle(
    opticalContinuationSource,
    opticalEncodedBoundaryId(triangleIndex, opticalTraversalInstanceIndex),
    representedId,
    triangleIndex,
    materialTexturePointToWorld(a, opticalTraversalInstanceIndex),
    materialTexturePointToWorld(b, opticalTraversalInstanceIndex),
    materialTexturePointToWorld(c, opticalTraversalInstanceIndex),
  );
}

${PT_WEBGPU_INTERSECTION_CORE_WGSL}

fn opticalSetContinuationSourceFromHit(hit: SceneHit) -> bool {
  if (
    hit.triIndex >= min(params.triangleCount, arrayLength(&indices)) ||
    hit.instanceIndex == INVALID_TLAS_INSTANCE_INDEX ||
    hit.instanceIndex == 0xffffffffu
  ) {
    opticalClearContinuationSource();
    return false;
  }
  let tri = indices[hit.triIndex];
  if (
    tri.x >= arrayLength(&positions) ||
    tri.y >= arrayLength(&positions) ||
    tri.z >= arrayLength(&positions)
  ) {
    opticalClearContinuationSource();
    return false;
  }
  opticalContinuationSource = opticalCreateSourceFeature(
    opticalEncodedBoundaryId(hit.triIndex, hit.instanceIndex),
    hit.instanceIndex + 1u,
    hit.triIndex,
    hit.zeroEdgeMask,
    materialTexturePointToWorld(positions[tri.x].xyz, hit.instanceIndex),
    materialTexturePointToWorld(positions[tri.y].xyz, hit.instanceIndex),
    materialTexturePointToWorld(positions[tri.z].xyz, hit.instanceIndex),
  );
  return opticalContinuationSourceIsActive();
}

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
      (*hit).position = worldHitPos;
      (*hit).triIndex = params.triangleCount + ai;
      (*hit).normal = transformNormalFromWorldToLocalCols(w2l0, w2l1, w2l2, localN);
      (*hit).frontFace = dot((*hit).normal, ray.direction) < 0.0;
      (*hit).baryVW = vec2f(0.0);
      (*hit).instanceIndex = INVALID_TLAS_INSTANCE_INDEX;
      (*hit).zeroEdgeMask = 0u;
    }
  }
  return false;
}

${tlasSceneHitTraversalWithInstanceIndex(TLAS_SCENE_HIT_TRAVERSAL_WGSL)}

// Full-tier optical replay intentionally walks every represented TLAS instance
// but traverses each instance's BLAS, rather than scanning triangles. This is a
// launch-time operation and preserves the exact same represented transform and
// geometry stream as radiance transport without introducing a crossing cap.
fn traceOpticalBoundaryClosest(
  ray: Ray,
  exclusiveMinT: f32,
  tMax: f32,
) -> OpticalBoundaryHit {
  var result: OpticalBoundaryHit;
  opticalResetBoundaryHit(&result, tMax);
  if (
    params.tlasNodeCount == 0u ||
    arrayLength(&tlasNodes) == 0u ||
    arrayLength(&tlasInstanceIndices) == 0u
  ) {
    result.valid = false;
    return result;
  }
  let instanceLimit = min(
    arrayLength(&tlasBlasRoots),
    min(
      arrayLength(&tlasInstanceWorldToLocal) / 4u,
      arrayLength(&tlasInstanceLocalToWorld) / 4u,
    ),
  );
  if (instanceLimit == 0u) {
    result.valid = false;
    return result;
  }
  for (var instIdx = 0u; instIdx < instanceLimit; instIdx = instIdx + 1u) {
    let matrixBase = instIdx * 4u;
    let w2l0 = tlasInstanceWorldToLocal[matrixBase];
    let w2l1 = tlasInstanceWorldToLocal[matrixBase + 1u];
    let w2l2 = tlasInstanceWorldToLocal[matrixBase + 2u];
    let w2l3 = tlasInstanceWorldToLocal[matrixBase + 3u];
    let l2w0 = tlasInstanceLocalToWorld[matrixBase];
    let l2w1 = tlasInstanceLocalToWorld[matrixBase + 1u];
    let l2w2 = tlasInstanceLocalToWorld[matrixBase + 2u];
    let l2w3 = tlasInstanceLocalToWorld[matrixBase + 3u];
    var localRay: Ray;
    localRay.origin = transformPointCols(
      w2l0, w2l1, w2l2, w2l3, ray.origin,
    );
    localRay.direction = transformDirectionCols(
      w2l0, w2l1, w2l2, ray.direction,
    );
    // Do not convert the exclusive world cursor to one rounded local-space
    // threshold. Under a non-uniform transform that threshold can round upward
    // and skip the adjacent representable boundary. Replay from the fixed local
    // origin, discard already-consumed events only after converting each exact
    // hit back to world t, and stop at this instance's first remaining event.
    // The loop has no crossing ceiling; only the live medium stack is bounded.
    var localCursor = 0.0;
    loop {
      var localHit: OpticalLocalBoundaryHit;
      traceOpticalMeshBvhLocal(
        localRay,
        localCursor,
        INFINITY,
        tlasBlasRoots[instIdx],
        &localHit,
      );
      if (!localHit.valid) {
        result.valid = false;
        return result;
      }
      if (!localHit.didHit) { break; }
      if (!(localHit.dist > localCursor)) {
        result.valid = false;
        return result;
      }
      let localHitPoint =
        localRay.origin + localRay.direction * localHit.dist;
      let worldHitPoint = transformPointCols(
        l2w0, l2w1, l2w2, l2w3, localHitPoint,
      );
      let worldT = dot(worldHitPoint - ray.origin, ray.direction);
      if (!(worldT == worldT) || abs(worldT) > 3.402823e38) {
        result.valid = false;
        return result;
      }
      if (!(worldT > exclusiveMinT)) {
        localCursor = localHit.dist;
        continue;
      }
      if (!(worldT < tMax)) { break; }
      let boundary = mediumBoundaryIdentity(localHit.triIndex, instIdx);
      if (!mediumBoundaryIsValid(boundary)) {
        result.valid = false;
        return result;
      }
      let orientationPreserving =
        transformLinearOrientationSign(l2w0, l2w1, l2w2) > 0.0;
      let frontFace = select(
        !localHit.frontFace, localHit.frontFace, orientationPreserving,
      );
      if (!result.didHit || worldT < result.dist) {
        result.didHit = true;
        result.ambiguous = localHit.ambiguous;
        result.tangent = localHit.tangent;
      result.dist = worldT;
      result.triIndex = localHit.triIndex;
      result.instanceIndex = instIdx;
      result.baryVW = localHit.baryVW;
      result.matId = triMaterialIds[localHit.triIndex].x;
        result.boundary = boundary;
        result.frontFace = frontFace;
      } else if (bitcast<u32>(worldT) == bitcast<u32>(result.dist)) {
        if (!opticalBoundaryHitSameEvent(
          boundary, frontFace, result.boundary, result.frontFace,
        )) {
          result.ambiguous = true;
        }
      }
      break;
    }
  }
  return result;
}

fn traceClosestRaw(ray: Ray, tMin: f32, tMax: f32) -> SceneHit {
  var hit: SceneHit;
  _ = traceTlasClosest(ray, tMin, tMax, &hit);
  _ = traceAnalyticShapes(ray, tMin, tMax, true, &hit);
  return hit;
}

fn nextSidedTraversalCursor(cursor: f32, hitDist: f32) -> f32 {
  _ = cursor;
  return hitDist;
}

fn traceClosest(ray: Ray, tMin: f32, tMax: f32) -> SceneHit {
  // A post-transmission ray starts at the exact accepted hit and owns an exact
  // source-feature token. Its lower bound is therefore zero-exclusive; the
  // public clearance policy must not erase an arbitrarily close next surface.
  var cursor = select(tMin, 0.0, opticalContinuationSourceIsActive());
  var hit = traceClosestRaw(ray, cursor, tMax);
  loop {
    if (!hit.didHit) {
      opticalClearContinuationSource();
      return hit;
    }
    let matId = hitMaterialId(hit);
    if (materialAcceptsSidedHit(matId, hit.frontFace)) {
      opticalClearContinuationSource();
      return hit;
    }
    let nextCursor = nextSidedTraversalCursor(cursor, hit.dist);
    if (!(nextCursor > cursor) || !(nextCursor < tMax)) {
      hit.didHit = false;
      opticalClearContinuationSource();
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
  var cursor = select(tMin, 0.0, opticalContinuationSourceIsActive());
  loop {
    let hit = traceClosestRaw(ray, cursor, tMax);
    if (!hit.didHit) {
      opticalClearContinuationSource();
      return false;
    }
    let matId = hitMaterialId(hit);
    if (
      materialAcceptsSidedHit(matId, hit.frontFace) &&
      !materialShadowCastDisabled(matId) &&
      !alphaTestPassThrough(
        matId, hit.triIndex, hit.baryVW, hit.instanceIndex, rng,
      )
    ) {
      opticalClearContinuationSource();
      return true;
    }
    let nextCursor = nextSidedTraversalCursor(cursor, hit.dist);
    if (!(nextCursor > cursor) || !(nextCursor < tMax)) {
      opticalClearContinuationSource();
      return false;
    }
    cursor = nextCursor;
  }
  opticalClearContinuationSource();
  return false;
}

fn hitMaterialId(hit: SceneHit) -> u32 {
  if (hit.triIndex < params.triangleCount) {
    return select(0u, triMaterialIds[hit.triIndex].x, hit.triIndex < arrayLength(&triMaterialIds));
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
  (*hit).position = vec3f(0.0);
  (*hit).triIndex = 0u;
  (*hit).normal = vec3f(0.0, 1.0, 0.0);
  (*hit).frontFace = false;
  (*hit).baryVW = vec2f(0.0);
  (*hit).instanceIndex = INVALID_TLAS_INSTANCE_INDEX;
  (*hit).zeroEdgeMask = 0u;

  // Continuation filtering needs the exact represented face/edge/vertex fan;
  // the wide traversal does not expose every tied candidate. Use the binary
  // watertight traversal for this one post-transmission segment.
  if (opticalContinuationSourceIsActive()) {
    return traceMeshBvh(
      ray, tMin, tMaxBound, true, hit, binaryRootNode, captureShadingDetails,
    );
  }

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

  let acceptedTri = indices[cHit.triIndex];
  if (
    acceptedTri.x >= arrayLength(&positions) ||
    acceptedTri.y >= arrayLength(&positions) ||
    acceptedTri.z >= arrayLength(&positions)
  ) {
    return traceMeshBvh(
      ray, tMin, tMaxBound, true, hit, binaryRootNode, captureShadingDetails,
    );
  }
  let exactHit = opticalWatertightTriangleIntersect(
    ray.origin,
    ray.direction,
    positions[acceptedTri.x].xyz,
    positions[acceptedTri.y].xyz,
    positions[acceptedTri.z].xyz,
    tMin,
  );
  if (
    !exactHit.hit ||
    bitcast<u32>(exactHit.t) != bitcast<u32>(cHit.dist)
  ) {
    return traceMeshBvh(
      ray, tMin, tMaxBound, true, hit, binaryRootNode, captureShadingDetails,
    );
  }

  // CWBVH stores its geometric normal oriented against the incident ray;
  // SceneHit keeps the authored-winding normal and records side separately.
  var shadeNormal = exactHit.normal * exactHit.side;
  var shadeBaryVW = vec2f(exactHit.bary.y, exactHit.bary.z);
  if (captureShadingDetails) {
    let tri = indices[cHit.triIndex];
    if (tri.x < arrayLength(&positions) && tri.y < arrayLength(&positions) && tri.z < arrayLength(&positions)) {
      if (tri.x < arrayLength(&normals) && tri.y < arrayLength(&normals) && tri.z < arrayLength(&normals)) {
        let na = normals[tri.x].xyz;
        let nb = normals[tri.y].xyz;
        let nc = normals[tri.z].xyz;
        shadeNormal = safe_normalize(
          na * exactHit.bary.x + nb * exactHit.bary.y + nc * exactHit.bary.z,
        );
      }
    }
  }
  (*hit).didHit = true;
  (*hit).dist = cHit.dist;
  (*hit).position = opticalCanonicalHitPoint(
    exactHit,
    positions[acceptedTri.x].xyz,
    positions[acceptedTri.y].xyz,
    positions[acceptedTri.z].xyz,
  );
  (*hit).triIndex = cHit.triIndex;
  (*hit).normal = shadeNormal;
  (*hit).frontFace = exactHit.side > 0.0;
  (*hit).baryVW = shadeBaryVW;
  (*hit).instanceIndex = INVALID_TLAS_INSTANCE_INDEX;
  (*hit).zeroEdgeMask = exactHit.zeroEdgeMask;
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
  if (opticalContinuationSourceIsActive()) {
    return traceTlasClosest(ray, 0.0, tMax, hit);
  }
  if (params.tlasNodeCount == 0u || arrayLength(&tlasNodes) == 0u || arrayLength(&tlasInstanceIndices) == 0u) {
    return traceTlasClosest(ray, tMin, tMax, hit);
  }
  (*hit).didHit = false;
  (*hit).dist = tMax;
  (*hit).position = vec3f(0.0);
  (*hit).triIndex = 0u;
  (*hit).normal = vec3f(0.0, 1.0, 0.0);
  (*hit).frontFace = false;
  (*hit).baryVW = vec2f(0.0);
  (*hit).instanceIndex = INVALID_TLAS_INSTANCE_INDEX;
  (*hit).zeroEdgeMask = 0u;
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
        opticalSetTraversalInstance(instIdx);
        _ = traceMeshCwbvhClosest(localRay, localTMin, localTMax, &localHit, blasRoot, binaryBlasRoot, true);
        if (localHit.didHit && localHit.dist > localTMin) {
          let localHitPos = localHit.position;
          let worldHitPos = transformPointCols(l2w0, l2w1, l2w2, l2w3, localHitPos);
          let worldDist = dot(worldHitPos - ray.origin, ray.direction);
          if (worldDist <= tMin || worldDist >= (*hit).dist) { continue; }
          (*hit).didHit = true;
          (*hit).dist = worldDist;
          (*hit).position = worldHitPos;
          (*hit).triIndex = localHit.triIndex;
          (*hit).normal = transformNormalFromWorldToLocalCols(w2l0, w2l1, w2l2, localHit.normal);
          let orientationPreserving =
            transformLinearOrientationSign(l2w0, l2w1, l2w2) > 0.0;
          (*hit).frontFace = select(!localHit.frontFace, localHit.frontFace, orientationPreserving);
          (*hit).instanceIndex = instIdx;
          (*hit).baryVW = localHit.baryVW;
          (*hit).zeroEdgeMask = localHit.zeroEdgeMask;
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
