/**
 * Scene traversal — canonical BVH/TLAS intersection helpers + the
 * merged-world-BVH-vs-TLAS+local-BLAS dispatch wrappers (PR-3).
 *
 * Split out of common.wgsl.ts (T9-stepA). Injects the canonical
 * `BVH_INTERSECT_WGSL` + `TLAS_TRAVERSAL_WGSL` from `@vitrum/shared-bvh`
 * (single source of truth for BVHNode, Ray, IntersectionResult,
 * intersectTriangle, bvhIntersectFirstHit/Any, traceTlas*), then defines
 * `traceSceneFirstHit` plus the cast-mask-aware shadow wrapper which pick the
 * path from `ubo.bvhMode` / `ubo.tlasNodeCount`.
 */

import type { WgslModule } from '../pipeline/wgslComposer.js';
import {
  BVH_CAST_SHADOW_MASK_WGSL,
  BVH_INTERSECT_STACK_DEPTH,
  BVH_INTERSECT_CORE_WGSL,
  OPTICAL_WATERTIGHT_TRIANGLE_WGSL,
  TLAS_TRAVERSAL_STACK_DEPTH,
  TLAS_TRAVERSAL_CORE_WGSL,
} from '@vitrum/shared-bvh';
import { SCENE_STORAGE_ARENA_WGSL } from './sceneStorageArena.wgsl.js';

const OPTICAL_TRAVERSAL_BEGIN_MARKER =
  '// vitrum:optical-scene-traversal-begin';
const OPTICAL_TRAVERSAL_END_MARKER =
  '// vitrum:optical-scene-traversal-end';

export const SCENE_TRAVERSAL_WGSL = /* wgsl */ `// ============================================================
// BVH structs + intersection helpers — canonical from @vitrum/shared-bvh
// (sweep-20260518/moller-trumbore-canonical). Single source of truth for
// BVHNode, Ray, IntersectionResult, safeInvDir, intersectTriangle,
// bvhIntersectFirstHit, bvhIntersectAny. Pre-canonical inline copies were
// here (lines 128-164 + 480-735 in the pre-refactor file).
//
// Migration notes:
//   - The canonical return type is IntersectionResult (superset). The
//     pre-canonical HitResult is gone; its bary field is now barycoord,
//     and triIndex is now indices.w (matches DDGI / RC conventions).
//   - Cast-mask-aware any-hit traversal carries both skipGlass and packed
//     cast-shadow/alpha flags. Tinted transmission uses the dedicated
//     per-channel visibility walk in surfaceTextures.wgsl.
// ============================================================
${BVH_INTERSECT_CORE_WGSL}
${SCENE_STORAGE_ARENA_WGSL}
${TLAS_TRAVERSAL_CORE_WGSL}
${OPTICAL_WATERTIGHT_TRIANGLE_WGSL}

// Scene traversal — merged world BVH vs TLAS+local BLAS (PR-3).
fn traceSceneFirstHit(
  bvhMode: u32,
  tlasNodeCount: u32,
  ray: Ray,
  triEps: f32,
) -> IntersectionResult {
  if (bvhNodeCapacity() == 0u || bvhIndexCount() == 0u) {
    return tlasEmptyFirstHit();
  }
  if (bvhMode == 1u && tlasNodeCount > 0u) {
    return traceTlasFirstHit(tlasNodeCount, ray, triEps);
  }
  return bvhIntersectFirstHit(ray, triEps);
}

${OPTICAL_TRAVERSAL_BEGIN_MARKER}
// --------------------------------------------------------------------------
// Exact optical-boundary traversal
// --------------------------------------------------------------------------

struct OpticalSceneTriangle {
  valid: u32,
  indices: vec3u,
  a: vec3f,
  b: vec3f,
  c: vec3f,
  uvA: vec2f,
  uvB: vec2f,
  uvC: vec2f,
  materialWord: u32,
};

struct OpticalSceneBoundaryEvent {
  status: u32,
  t: f32,
  encodedBoundaryId: u32,
  side: f32,
  representedPrimitiveInstanceId: u32,
  zeroEdgeMask: u32,
  hit: IntersectionResult,
};

fn sceneOpticalEncodedBoundaryId(
  useTlas: bool,
  triangleIndex: u32,
  instanceIndex: u32,
) -> u32 {
  if (triangleIndex >= sceneOpticalTriangleIdentityCount()) { return 0u; }
  let componentPlusOne = sceneLoadOpticalTriangleIdentity(triangleIndex).x;
  if (componentPlusOne == 0u) { return 0u; }
  let baseIndex = select(0u, instanceIndex, useTlas);
  if (baseIndex >= sceneOpticalInstanceBoundaryIdBaseCount()) { return 0u; }
  let basePlusOne = sceneLoadOpticalInstanceBoundaryIdBasePlusOne(baseIndex);
  if (basePlusOne == 0u) { return 0u; }
  let componentOffset = componentPlusOne - 1u;
  if (componentOffset > 0xffffffffu - basePlusOne) { return 0u; }
  return basePlusOne + componentOffset;
}

fn sceneOpticalRepresentedPrimitiveInstanceId(
  useTlas: bool,
  triangleIndex: u32,
  instanceIndex: u32,
) -> u32 {
  if (triangleIndex >= sceneOpticalTriangleIdentityCount()) { return 0u; }
  if (useTlas) {
    if (
      instanceIndex >= sceneOpticalInstanceBoundaryIdBaseCount() ||
      instanceIndex == 0xffffffffu
    ) { return 0u; }
    return instanceIndex + 1u;
  }
  return sceneLoadOpticalTriangleIdentity(triangleIndex).y;
}

fn sceneLoadOpticalWorldTriangle(
  useTlas: bool,
  triangleIndex: u32,
  instanceIndex: u32,
) -> OpticalSceneTriangle {
  var triangle: OpticalSceneTriangle;
  triangle.valid = 0u;
  if (
    triangleIndex >= bvhIndexCount() ||
    triangleIndex >= sceneOpticalTriangleIdentityCount()
  ) { return triangle; }
  let indexEntry = bvhLoadIndex(triangleIndex);
  if (any(indexEntry.xyz >= vec3u(bvhPositionCount()))) { return triangle; }
  let pa = bvhLoadPosition(indexEntry.x);
  let pb = bvhLoadPosition(indexEntry.y);
  let pc = bvhLoadPosition(indexEntry.z);
  triangle.indices = indexEntry.xyz;
  triangle.a = pa.xyz;
  triangle.b = pb.xyz;
  triangle.c = pc.xyz;
  triangle.uvA = unpack2x16float(bitcast<u32>(pa.w));
  triangle.uvB = unpack2x16float(bitcast<u32>(pb.w));
  triangle.uvC = unpack2x16float(bitcast<u32>(pc.w));
  triangle.materialWord = indexEntry.w;
  if (useTlas) {
    let matrixBase = instanceIndex * 4u;
    if (
      instanceIndex > 0x3fffffffu ||
      matrixBase + 3u >= tlasLocalToWorldColumnCount()
    ) { return triangle; }
    let c0 = tlasLoadLocalToWorldColumn(matrixBase);
    let c1 = tlasLoadLocalToWorldColumn(matrixBase + 1u);
    let c2 = tlasLoadLocalToWorldColumn(matrixBase + 2u);
    let c3 = tlasLoadLocalToWorldColumn(matrixBase + 3u);
    triangle.a = tlasTransformPointCols(c0, c1, c2, c3, triangle.a);
    triangle.b = tlasTransformPointCols(c0, c1, c2, c3, triangle.b);
    triangle.c = tlasTransformPointCols(c0, c1, c2, c3, triangle.c);
  }
  if (
    any(triangle.a != triangle.a) || any(triangle.b != triangle.b) ||
    any(triangle.c != triangle.c) ||
    any(abs(triangle.a) > vec3f(3.402823e38)) ||
    any(abs(triangle.b) > vec3f(3.402823e38)) ||
    any(abs(triangle.c) > vec3f(3.402823e38))
  ) { return triangle; }
  triangle.valid = 1u;
  return triangle;
}

fn opticalSceneBoundaryEventEmpty() -> OpticalSceneBoundaryEvent {
  var result: OpticalSceneBoundaryEvent;
  result.status = OPTICAL_BOUNDARY_EVENT_NONE;
  result.t = 3.402823e38;
  result.encodedBoundaryId = 0u;
  result.side = 0.0;
  result.representedPrimitiveInstanceId = 0u;
  result.zeroEdgeMask = 0u;
  result.hit = tlasEmptyFirstHit();
  return result;
}

fn opticalSceneAccumulateBoundaryTriangle(
  useTlas: bool,
  triangleIndex: u32,
  instanceIndex: u32,
  ray: Ray,
  exclusiveMinT: f32,
  sourceFeature: OpticalSourceFeature,
  accumulator: ptr<function, OpticalBoundaryEventAccumulator>,
  selected: ptr<function, OpticalSceneBoundaryEvent>,
) {
  let encodedBoundaryId = sceneOpticalEncodedBoundaryId(
    useTlas, triangleIndex, instanceIndex,
  );
  // Zero is an intentionally valid thin-sheet topology lane, but this query
  // enumerates persistent bulk-boundary events only.
  if (encodedBoundaryId == 0u) { return; }
  let representedId = sceneOpticalRepresentedPrimitiveInstanceId(
    useTlas, triangleIndex, instanceIndex,
  );
  let triangle = sceneLoadOpticalWorldTriangle(
    useTlas, triangleIndex, instanceIndex,
  );
  if (triangle.valid == 0u || representedId == 0u) {
    (*accumulator).invalidInput = 1u;
    return;
  }
  if (opticalSourceFeatureSuppressesTriangle(
    sourceFeature,
    encodedBoundaryId,
    representedId,
    triangleIndex,
    triangle.a,
    triangle.b,
    triangle.c,
  )) { return; }
  let exact = opticalWatertightTriangleIntersect(
    ray.origin,
    ray.direction,
    triangle.a,
    triangle.b,
    triangle.c,
    exclusiveMinT,
  );
  if (!exact.hit) { return; }

  let replacesSelected =
    (*accumulator).hasCandidate == 0u || exact.t < (*accumulator).t;
  opticalBoundaryEventAccumulate(
    accumulator,
    exact.t,
    encodedBoundaryId,
    exact.side,
  );
  if (replacesSelected) {
    (*selected).t = exact.t;
    (*selected).encodedBoundaryId = encodedBoundaryId;
    (*selected).side = exact.side;
    (*selected).representedPrimitiveInstanceId = representedId;
    (*selected).zeroEdgeMask = exact.zeroEdgeMask;
    (*selected).hit.didHit = true;
    (*selected).hit.indices = vec4u(triangle.indices, triangleIndex);
    (*selected).hit.normal = exact.normal;
    (*selected).hit.barycoord = exact.bary;
    (*selected).hit.side = exact.side;
    (*selected).hit.dist = exact.t;
    (*selected).hit.matColorPacked = triangle.materialWord;
    (*selected).hit.uv =
      exact.bary.x * triangle.uvA +
      exact.bary.y * triangle.uvB +
      exact.bary.z * triangle.uvC;
    (*selected).hit.instanceIndex = instanceIndex;
  }
}

fn opticalSceneTraverseBlasBoundaryCandidates(
  useTlas: bool,
  instanceIndex: u32,
  rootNode: u32,
  ray: Ray,
  exclusiveMinT: f32,
  sourceFeature: OpticalSourceFeature,
  accumulator: ptr<function, OpticalBoundaryEventAccumulator>,
  selected: ptr<function, OpticalSceneBoundaryEvent>,
) {
  var traversalRay = ray;
  if (useTlas) {
    let matrixBase = instanceIndex * 4u;
    if (
      instanceIndex > 0x3fffffffu ||
      matrixBase + 3u >= tlasWorldToLocalColumnCount()
    ) {
      (*accumulator).invalidInput = 1u;
      return;
    }
    let c0 = tlasLoadWorldToLocalColumn(matrixBase);
    let c1 = tlasLoadWorldToLocalColumn(matrixBase + 1u);
    let c2 = tlasLoadWorldToLocalColumn(matrixBase + 2u);
    let c3 = tlasLoadWorldToLocalColumn(matrixBase + 3u);
    traversalRay.origin = tlasTransformPointCols(c0, c1, c2, c3, ray.origin);
    traversalRay.direction = tlasTransformDirectionCols(c0, c1, c2, ray.direction);
  }
  let invDirection = safeInvDir(traversalRay.direction);
  var stack: array<u32, ${BVH_INTERSECT_STACK_DEPTH}>;
  var stackPointer = 0u;
  stack[0] = rootNode;
  stackPointer = 1u;
  while (stackPointer > 0u) {
    stackPointer -= 1u;
    let nodeIndex = stack[stackPointer];
    if (nodeIndex >= bvhNodeCapacity()) {
      (*accumulator).invalidInput = 1u;
      return;
    }
    let node = bvhLoadNode(nodeIndex);
    let boundsMin = vec3f(
      node.boundsMin[0], node.boundsMin[1], node.boundsMin[2],
    );
    let boundsMax = vec3f(
      node.boundsMax[0], node.boundsMax[1], node.boundsMax[2],
    );
    let t0 = (boundsMin - traversalRay.origin) * invDirection;
    let t1 = (boundsMax - traversalRay.origin) * invDirection;
    let tNear = max(max(min(t0.x, t1.x), min(t0.y, t1.y)), min(t0.z, t1.z));
    let tFar = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z));
    if (tNear > tFar || tFar < 0.0) { continue; }

    let splitOrCount = node.splitAxisOrTriCount;
    if ((splitOrCount & BVH_LEAFNODE_FLAG) == BVH_LEAFNODE_FLAG) {
      let triangleCount = splitOrCount & 0x0000ffffu;
      let triangleOffset = node.rightChildOrTriOffset;
      for (var triangle = 0u; triangle < triangleCount; triangle += 1u) {
        opticalSceneAccumulateBoundaryTriangle(
          useTlas,
          triangleOffset + triangle,
          instanceIndex,
          ray,
          exclusiveMinT,
          sourceFeature,
          accumulator,
          selected,
        );
      }
    } else {
      if (stackPointer + 2u > ${BVH_INTERSECT_STACK_DEPTH}u) {
        (*accumulator).invalidInput = 1u;
        return;
      }
      stack[stackPointer] = nodeIndex + node.rightChildOrTriOffset;
      stackPointer += 1u;
      stack[stackPointer] = nodeIndex + 1u;
      stackPointer += 1u;
    }
  }
}

fn opticalSceneTraverseAllTlasInstances(
  ray: Ray,
  exclusiveMinT: f32,
  sourceFeature: OpticalSourceFeature,
  accumulator: ptr<function, OpticalBoundaryEventAccumulator>,
  selected: ptr<function, OpticalSceneBoundaryEvent>,
) {
  for (
    var permutationIndex = 0u;
    permutationIndex < tlasInstanceIndexCount();
    permutationIndex += 1u
  ) {
    let instanceIndex = tlasLoadInstanceIndex(permutationIndex);
    if (instanceIndex >= tlasBlasRootCount()) {
      (*accumulator).invalidInput = 1u;
      return;
    }
    opticalSceneTraverseBlasBoundaryCandidates(
      true,
      instanceIndex,
      tlasLoadBlasRoot(instanceIndex),
      ray,
      exclusiveMinT,
      sourceFeature,
      accumulator,
      selected,
    );
  }
}

fn traceSceneOpticalBoundaryEvent(
  bvhMode: u32,
  tlasNodeCount: u32,
  ray: Ray,
  exclusiveMinT: f32,
  sourceFeature: OpticalSourceFeature,
) -> OpticalSceneBoundaryEvent {
  var accumulator = opticalBoundaryEventAccumulatorInit();
  var selected = opticalSceneBoundaryEventEmpty();
  if (bvhNodeCapacity() == 0u || bvhIndexCount() == 0u) {
    return selected;
  }
  if (
    bvhMode == 1u && tlasNodeCount > 0u &&
    tlasNodeCapacity() > 0u && tlasInstanceIndexCount() > 0u
  ) {
    var stack: array<u32, ${TLAS_TRAVERSAL_STACK_DEPTH}>;
    var stackPointer = 1u;
    stack[0] = 0u;
    var overflowed = false;
    while (stackPointer > 0u) {
      stackPointer -= 1u;
      let nodeIndex = stack[stackPointer];
      if (nodeIndex >= min(tlasNodeCount, tlasNodeCapacity())) {
        accumulator.invalidInput = 1u;
        break;
      }
      let node = tlasLoadNode(nodeIndex);
      let boundsMin = vec3f(
        node.boundsMin[0], node.boundsMin[1], node.boundsMin[2],
      );
      let boundsMax = vec3f(
        node.boundsMax[0], node.boundsMax[1], node.boundsMax[2],
      );
      if (!tlasIntersectAabb(
        ray,
        boundsMin,
        boundsMax,
        max(exclusiveMinT, 0.0),
        accumulator.t,
      )) { continue; }
      let splitOrCount = node.splitAxisOrTriCount;
      if ((splitOrCount & BVH_LEAFNODE_FLAG) == BVH_LEAFNODE_FLAG) {
        let count = splitOrCount & 0x0000ffffu;
        let start = node.rightChildOrTriOffset;
        for (var entry = 0u; entry < count; entry += 1u) {
          let permutationIndex = start + entry;
          if (permutationIndex >= tlasInstanceIndexCount()) {
            accumulator.invalidInput = 1u;
            break;
          }
          let instanceIndex = tlasLoadInstanceIndex(permutationIndex);
          if (instanceIndex >= tlasBlasRootCount()) {
            accumulator.invalidInput = 1u;
            break;
          }
          opticalSceneTraverseBlasBoundaryCandidates(
            true,
            instanceIndex,
            tlasLoadBlasRoot(instanceIndex),
            ray,
            exclusiveMinT,
            sourceFeature,
            &accumulator,
            &selected,
          );
        }
      } else {
        if (stackPointer + 2u > ${TLAS_TRAVERSAL_STACK_DEPTH}u) {
          overflowed = true;
          break;
        }
        stack[stackPointer] = nodeIndex + node.rightChildOrTriOffset;
        stackPointer += 1u;
        stack[stackPointer] = nodeIndex + 1u;
        stackPointer += 1u;
      }
    }
    if (overflowed) {
      accumulator = opticalBoundaryEventAccumulatorInit();
      selected = opticalSceneBoundaryEventEmpty();
      opticalSceneTraverseAllTlasInstances(
        ray,
        exclusiveMinT,
        sourceFeature,
        &accumulator,
        &selected,
      );
    }
  } else {
    opticalSceneTraverseBlasBoundaryCandidates(
      false,
      0u,
      0u,
      ray,
      exclusiveMinT,
      sourceFeature,
      &accumulator,
      &selected,
    );
  }
  let event = opticalBoundaryEventFinalize(accumulator);
  selected.status = event.status;
  selected.t = event.t;
  selected.encodedBoundaryId = event.encodedBoundaryId;
  selected.side = event.side;
  if (event.status == OPTICAL_BOUNDARY_EVENT_INVALID) {
    selected.hit.didHit = false;
  }
  return selected;
}

// Every ordinary/Moller transmissive hit must be replayable against the exact
// chosen represented world triangle before it can become a continuation
// source feature. A miss here is a fail-closed transport result, not a cue to
// infer ownership from tolerant barycentrics.
fn traceSceneRetraceOpticalHit(
  bvhMode: u32,
  tlasNodeCount: u32,
  ray: Ray,
  ordinaryHit: IntersectionResult,
  exclusiveMinT: f32,
) -> OpticalWatertightHit {
  let useTlas = bvhMode == 1u && tlasNodeCount > 0u;
  let triangle = sceneLoadOpticalWorldTriangle(
    useTlas,
    ordinaryHit.indices.w,
    ordinaryHit.instanceIndex,
  );
  if (triangle.valid == 0u) { return opticalWatertightMiss(); }
  return opticalWatertightTriangleIntersect(
    ray.origin,
    ray.direction,
    triangle.a,
    triangle.b,
    triangle.c,
    exclusiveMinT,
  );
}

fn sceneOpticalSourceFeatureForExactHit(
  bvhMode: u32,
  tlasNodeCount: u32,
  hit: IntersectionResult,
  exactHit: OpticalWatertightHit,
) -> OpticalSourceFeature {
  if (!exactHit.hit) { return opticalSourceFeatureInvalid(); }
  let useTlas = bvhMode == 1u && tlasNodeCount > 0u;
  let triangle = sceneLoadOpticalWorldTriangle(
    useTlas,
    hit.indices.w,
    hit.instanceIndex,
  );
  if (triangle.valid == 0u) { return opticalSourceFeatureInvalid(); }
  return opticalCreateSourceFeature(
    sceneOpticalEncodedBoundaryId(
      useTlas, hit.indices.w, hit.instanceIndex,
    ),
    sceneOpticalRepresentedPrimitiveInstanceId(
      useTlas, hit.indices.w, hit.instanceIndex,
    ),
    hit.indices.w,
    exactHit.zeroEdgeMask,
    triangle.a,
    triangle.b,
    triangle.c,
  );
}

// Closest-hit traversal for a ray launched directly from an accepted optical
// interface.  The ray origin remains the exact represented hit point and the
// open t interval starts at zero; only the exact crossed face/edge/vertex fan
// is suppressed.  This is deliberately separate from the ordinary epsilon
// traversal so a nearby distinct boundary cannot be skipped by an origin step
// or a scene-scale tMin.
struct OpticalSourceAwareFirstHit {
  valid: u32,
  hit: IntersectionResult,
};

fn opticalSceneSourceAwareFirstHitInit() -> OpticalSourceAwareFirstHit {
  var result: OpticalSourceAwareFirstHit;
  result.valid = 1u;
  result.hit = tlasEmptyFirstHit();
  return result;
}

fn opticalSceneConsiderSourceAwareTriangle(
  useTlas: bool,
  triangleIndex: u32,
  instanceIndex: u32,
  ray: Ray,
  exclusiveMinT: f32,
  sourceFeature: OpticalSourceFeature,
  result: ptr<function, OpticalSourceAwareFirstHit>,
) {
  let triangle = sceneLoadOpticalWorldTriangle(
    useTlas, triangleIndex, instanceIndex,
  );
  if (triangle.valid == 0u) {
    (*result).valid = 0u;
    return;
  }
  let encodedBoundaryId = sceneOpticalEncodedBoundaryId(
    useTlas, triangleIndex, instanceIndex,
  );
  let representedId = sceneOpticalRepresentedPrimitiveInstanceId(
    useTlas, triangleIndex, instanceIndex,
  );
  if (opticalSourceFeatureSuppressesTriangle(
    sourceFeature,
    encodedBoundaryId,
    representedId,
    triangleIndex,
    triangle.a,
    triangle.b,
    triangle.c,
  )) { return; }

  var candidate = intersectTriangle(
    ray.origin,
    ray.direction,
    triangle.a,
    triangle.b,
    triangle.c,
    exclusiveMinT,
  );
  if (
    !candidate.didHit || !(candidate.dist > exclusiveMinT) ||
    !(candidate.dist < (*result).hit.dist)
  ) { return; }
  candidate.indices = vec4u(triangle.indices, triangleIndex);
  candidate.matColorPacked = triangle.materialWord;
  candidate.uv =
    candidate.barycoord.x * triangle.uvA +
    candidate.barycoord.y * triangle.uvB +
    candidate.barycoord.z * triangle.uvC;
  candidate.instanceIndex = instanceIndex;
  (*result).hit = candidate;
}

fn opticalSceneTraverseBlasSourceAwareFirstHit(
  useTlas: bool,
  instanceIndex: u32,
  rootNode: u32,
  ray: Ray,
  exclusiveMinT: f32,
  sourceFeature: OpticalSourceFeature,
  result: ptr<function, OpticalSourceAwareFirstHit>,
) {
  if ((*result).valid == 0u || rootNode >= bvhNodeCapacity()) {
    (*result).valid = 0u;
    return;
  }
  var traversalRay = ray;
  if (useTlas) {
    if (instanceIndex > 0x3fffffffu) {
      (*result).valid = 0u;
      return;
    }
    let matrixBase = instanceIndex * 4u;
    if (matrixBase + 3u >= tlasWorldToLocalColumnCount()) {
      (*result).valid = 0u;
      return;
    }
    traversalRay.origin = tlasTransformPointCols(
      tlasLoadWorldToLocalColumn(matrixBase),
      tlasLoadWorldToLocalColumn(matrixBase + 1u),
      tlasLoadWorldToLocalColumn(matrixBase + 2u),
      tlasLoadWorldToLocalColumn(matrixBase + 3u),
      ray.origin,
    );
    traversalRay.direction = tlasTransformDirectionCols(
      tlasLoadWorldToLocalColumn(matrixBase),
      tlasLoadWorldToLocalColumn(matrixBase + 1u),
      tlasLoadWorldToLocalColumn(matrixBase + 2u),
      ray.direction,
    );
  }
  let invDirection = safeInvDir(traversalRay.direction);
  var stack: array<u32, ${BVH_INTERSECT_STACK_DEPTH}>;
  var stackPointer = 1u;
  stack[0] = rootNode;
  while (stackPointer > 0u) {
    stackPointer -= 1u;
    let nodeIndex = stack[stackPointer];
    if (nodeIndex >= bvhNodeCapacity()) {
      (*result).valid = 0u;
      return;
    }
    let node = bvhLoadNode(nodeIndex);
    let boundsMin = vec3f(
      node.boundsMin[0], node.boundsMin[1], node.boundsMin[2],
    );
    let boundsMax = vec3f(
      node.boundsMax[0], node.boundsMax[1], node.boundsMax[2],
    );
    let t0 = (boundsMin - traversalRay.origin) * invDirection;
    let t1 = (boundsMax - traversalRay.origin) * invDirection;
    let tNear = max(max(min(t0.x, t1.x), min(t0.y, t1.y)), min(t0.z, t1.z));
    let tFar = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z));
    if (tNear > tFar || tFar < 0.0) { continue; }

    let splitOrCount = node.splitAxisOrTriCount;
    if ((splitOrCount & BVH_LEAFNODE_FLAG) == BVH_LEAFNODE_FLAG) {
      let triangleCount = splitOrCount & 0x0000ffffu;
      let triangleOffset = node.rightChildOrTriOffset;
      if (
        triangleOffset > bvhIndexCount() ||
        triangleCount > bvhIndexCount() - triangleOffset
      ) {
        (*result).valid = 0u;
        return;
      }
      for (var triangle = 0u; triangle < triangleCount; triangle += 1u) {
        opticalSceneConsiderSourceAwareTriangle(
          useTlas,
          triangleOffset + triangle,
          instanceIndex,
          ray,
          exclusiveMinT,
          sourceFeature,
          result,
        );
        if ((*result).valid == 0u) { return; }
      }
    } else {
      if (stackPointer + 2u > ${BVH_INTERSECT_STACK_DEPTH}u) {
        (*result).valid = 0u;
        return;
      }
      stack[stackPointer] = nodeIndex + node.rightChildOrTriOffset;
      stackPointer += 1u;
      stack[stackPointer] = nodeIndex + 1u;
      stackPointer += 1u;
    }
  }
}

fn opticalSceneTraverseAllTlasInstancesSourceAwareFirstHit(
  ray: Ray,
  exclusiveMinT: f32,
  sourceFeature: OpticalSourceFeature,
  result: ptr<function, OpticalSourceAwareFirstHit>,
) {
  for (
    var permutationIndex = 0u;
    permutationIndex < tlasInstanceIndexCount();
    permutationIndex += 1u
  ) {
    let instanceIndex = tlasLoadInstanceIndex(permutationIndex);
    if (instanceIndex >= tlasBlasRootCount()) {
      (*result).valid = 0u;
      return;
    }
    opticalSceneTraverseBlasSourceAwareFirstHit(
      true,
      instanceIndex,
      tlasLoadBlasRoot(instanceIndex),
      ray,
      exclusiveMinT,
      sourceFeature,
      result,
    );
    if ((*result).valid == 0u) { return; }
  }
}

fn traceSceneFirstHitWithOpticalSourceExclusion(
  bvhMode: u32,
  tlasNodeCount: u32,
  ray: Ray,
  exclusiveMinT: f32,
  sourceFeature: OpticalSourceFeature,
) -> OpticalSourceAwareFirstHit {
  var result = opticalSceneSourceAwareFirstHitInit();
  if (
    sourceFeature.kind == OPTICAL_SOURCE_FEATURE_INVALID ||
    sourceFeature.representedPrimitiveInstanceId == 0u ||
    exclusiveMinT != exclusiveMinT || exclusiveMinT < 0.0
  ) {
    result.valid = 0u;
    return result;
  }
  if (bvhNodeCapacity() == 0u || bvhIndexCount() == 0u) {
    return result;
  }
  if (
    bvhMode == 1u && tlasNodeCount > 0u &&
    tlasNodeCapacity() > 0u && tlasInstanceIndexCount() > 0u
  ) {
    var stack: array<u32, ${TLAS_TRAVERSAL_STACK_DEPTH}>;
    var stackPointer = 1u;
    stack[0] = 0u;
    var overflowed = false;
    while (stackPointer > 0u) {
      stackPointer -= 1u;
      let nodeIndex = stack[stackPointer];
      if (nodeIndex >= min(tlasNodeCount, tlasNodeCapacity())) {
        result.valid = 0u;
        return result;
      }
      let node = tlasLoadNode(nodeIndex);
      let boundsMin = vec3f(
        node.boundsMin[0], node.boundsMin[1], node.boundsMin[2],
      );
      let boundsMax = vec3f(
        node.boundsMax[0], node.boundsMax[1], node.boundsMax[2],
      );
      if (!tlasIntersectAabb(
        ray, boundsMin, boundsMax, 0.0, result.hit.dist,
      )) { continue; }
      let splitOrCount = node.splitAxisOrTriCount;
      if ((splitOrCount & BVH_LEAFNODE_FLAG) == BVH_LEAFNODE_FLAG) {
        let count = splitOrCount & 0x0000ffffu;
        let start = node.rightChildOrTriOffset;
        if (
          start > tlasInstanceIndexCount() ||
          count > tlasInstanceIndexCount() - start
        ) {
          result.valid = 0u;
          return result;
        }
        for (var entry = 0u; entry < count; entry += 1u) {
          let instanceIndex = tlasLoadInstanceIndex(start + entry);
          if (instanceIndex >= tlasBlasRootCount()) {
            result.valid = 0u;
            return result;
          }
          opticalSceneTraverseBlasSourceAwareFirstHit(
            true,
            instanceIndex,
            tlasLoadBlasRoot(instanceIndex),
            ray,
            exclusiveMinT,
            sourceFeature,
            &result,
          );
          if (result.valid == 0u) { return result; }
        }
      } else {
        if (stackPointer + 2u > ${TLAS_TRAVERSAL_STACK_DEPTH}u) {
          overflowed = true;
          break;
        }
        stack[stackPointer] = nodeIndex + node.rightChildOrTriOffset;
        stackPointer += 1u;
        stack[stackPointer] = nodeIndex + 1u;
        stackPointer += 1u;
      }
    }
    if (overflowed) {
      result = opticalSceneSourceAwareFirstHitInit();
      opticalSceneTraverseAllTlasInstancesSourceAwareFirstHit(
        ray, exclusiveMinT, sourceFeature, &result,
      );
    }
    return result;
  }
  opticalSceneTraverseBlasSourceAwareFirstHit(
    false,
    0u,
    0u,
    ray,
    exclusiveMinT,
    sourceFeature,
    &result,
  );
  return result;
}

${OPTICAL_TRAVERSAL_END_MARKER}
${BVH_CAST_SHADOW_MASK_WGSL}

// SHADOW-01 — castShadow-aware occlusion wrapper for the ReSTIR **DI** shadow
// predicates (ris.wgsl candidate visibility, ReSTIR-GI visibility, GRIS
// reconnection visibility, and shadingTerms.wgsl shading / analytic / sun
// visibility). The leaf loops skip triangles whose bvh_material word has bit 0 set
// (castShadow:false — packBVHRoughMetalFromCore) or bit 2 set
// (scalar alpha discarded). Callers pass the
// module-scope bvh_material texture + BVH_MATERIAL_TEX_WIDTH so this module
// stays binding-free. DDGI / RC use the sibling predicate-backed shared-bvh
// traversal because those passes carry material flags through MaterialEntry
// buffers rather than this texture.
fn traceSceneAnyCastMask(
  bvhMode: u32,
  tlasNodeCount: u32,
  origin: vec3f,
  dir: vec3f,
  tMax: f32,
  triEps: f32,
  skipGlass: bool,
  castMask: texture_2d<u32>,
  castMaskWidth: u32,
) -> bool {
  if (bvhNodeCapacity() == 0u || bvhIndexCount() == 0u) {
    return false;
  }
  if (bvhMode == 1u && tlasNodeCount > 0u) {
    return traceTlasAnyCastMask(
      tlasNodeCount,
      origin,
      dir,
      tMax,
      triEps,
      skipGlass,
      castMask,
      castMaskWidth,
    );
  }
  return bvhIntersectAnyAtRootCastMask(origin, dir, tMax, triEps, skipGlass, 0u, castMask, castMaskWidth);
}

// ─── WS1 (2026-05-29) — smooth shading normal via barycentric per-vertex blend ─
//
// Mirrors the DDGI precedent (probeUpdateRays.wgsl.ts:443-454): interpolate the
// per-vertex normals at the hit's barycentric coordinate, normalize, and apply
// the hit side so back-face hits get a consistently-oriented normal —
//   n = normalize(w·n0 + u·n1 + v·n2) · side.
//
// This replaces the faceted geometric face normal (hit.normal, cross(e1,e2))
// for SHADING. The geometric normal must still be used for ray-origin offsets /
// backface bias by the caller (a smooth normal can point into the surface near
// a silhouette edge, which would self-intersect the offset ray).
//
// TLAS mode (V21): the geometry arena holds LOCAL-space BLAS normals, so the blended
// shading normal is transformed to WORLD by the hit instance's inverse-transpose
// (tlasTransformNormalFromLocalCols with the instance world-to-local columns —
// the SAME transform traceTlasFirstHit applies to the geometric normal). The
// caller passes isTlas + the three world-to-local columns (read from the
// arena-backed world-to-local loader at instanceIndex*4); merged
// mode passes isTlas=false and the blend is already world-space. (The earlier
// wave kept the geometric normal in TLAS — that left smooth shading dormant on
// every multi-mesh / instanced scene, which all auto-select TLAS.)
//
// Degenerate guard: if the blended vector collapses (antipodal vertex normals
// across a thin/folded triangle) we fall back to the geometric face normal so
// the result stays finite + unit-length.
// Takes the three per-vertex normals BY VALUE (n0/n1/n2) rather than the
// arena-backed normal stream by pointer: Naga (wgpu-native / Firefox) rejects
// storage-buffer pointer function parameters, so a value-arg signature is naga-native
// and needs no shader-rewrite shim. Callers load sceneLoadBvhNormal(hit.indices.xyz)
// inline at the call site (indexing a module-scope storage global is fine; only
// passing it as a storage-buffer pointer param is the Naga gap). Caught by the wsl-gpu
// T1 smoke gate (lavapipe/naga) — the prior ptr-param form failed to compile.
// In TLAS mode the per-vertex normals (n0/n1/n2) are LOCAL-space BLAS normals, so
// the blended shading normal is transformed to world by the SAME inverse-transpose
// the geometric normal uses (tlasTransformNormalFromLocalCols with the instance's
// world-to-local columns). The caller reads those columns through the shared
// TLAS-arena loader (instanceIndex*4) and passes them BY VALUE —
// Naga rejects storage-buffer pointer params, but value vec4f args + a bool are naga-native.
// In merged-world mode isTlas is false and the blend is already world-space.
fn smoothShadingNormal(
  hit: IntersectionResult,
  geoNormal: vec3f,
  n0: vec3f,
  n1: vec3f,
  n2: vec3f,
  isTlas: bool,
  w2l0: vec4f,
  w2l1: vec4f,
  w2l2: vec4f,
) -> vec3f {
  let blended =
    hit.barycoord.x * n0 +
    hit.barycoord.y * n1 +
    hit.barycoord.z * n2;
  let blendedScale = max(abs(blended.x), max(abs(blended.y), abs(blended.z)));
  if (!(blendedScale > 0.0) || blendedScale > 3.402823466e38) {
    return geoNormal;
  }
  var n = safe_normalize(blended);
  if (isTlas) {
    let worldN = tlasTransformNormalFromLocalCols(w2l0, w2l1, w2l2, n);
    let worldScale = max(abs(worldN.x), max(abs(worldN.y), abs(worldN.z)));
    if (!(worldScale > 0.0) || worldScale > 3.402823466e38) {
      return geoNormal;
    }
    // hit.side is world-winding parity corrected by TLAS traversal. Apply
    // the transform parity to the authored local normal before multiplying by
    // that side so the final shading normal remains face-forward on mirrored
    // instances as well as non-mirrored ones.
    n = safe_normalize(worldN) * tlasLinearOrientationSign(w2l0, w2l1, w2l2);
  }
  return n * hit.side;
}

`;

/** T9-stepA — focused WGSL_MODULES entry split out of `common`. */
export const SCENE_TRAVERSAL_MODULE: WgslModule = {
  name: "sceneTraversal",
  source: SCENE_TRAVERSAL_WGSL,
  requires: [],
};

/**
 * Binding-agnostic exact optical traversal block used by non-composed probe
 * shaders that expose the canonical BVH/TLAS loader names themselves.
 * Keeping this as an extraction from {@link SCENE_TRAVERSAL_WGSL} prevents the
 * containment and fixed-origin source-exclusion algorithms from forking.
 */
export function sceneOpticalTraversalWgslForBindings(): string {
  const begin = SCENE_TRAVERSAL_WGSL.indexOf(OPTICAL_TRAVERSAL_BEGIN_MARKER);
  const end = SCENE_TRAVERSAL_WGSL.indexOf(
    OPTICAL_TRAVERSAL_END_MARKER,
    begin + OPTICAL_TRAVERSAL_BEGIN_MARKER.length,
  );
  if (begin < 0 || end <= begin) {
    throw new Error('sceneTraversal: optical traversal markers are missing or inverted.');
  }
  return SCENE_TRAVERSAL_WGSL.slice(
    begin + OPTICAL_TRAVERSAL_BEGIN_MARKER.length,
    end,
  );
}
