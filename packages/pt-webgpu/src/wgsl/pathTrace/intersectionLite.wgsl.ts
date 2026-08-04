import { PT_WEBGPU_INTERSECTION_CORE_WGSL } from './intersectionCore.wgsl.js';

export const PT_WEBGPU_PATH_TRACE_INTERSECTION_LITE_WGSL = /* wgsl */ `
fn opticalEncodedBoundaryId(triIndex: u32, instanceIndex: u32) -> u32 {
  _ = instanceIndex;
  if (triIndex >= min(params.triangleCount, arrayLength(&indices))) {
    return 0u;
  }
  return indices[triIndex].w;
}

var<private> opticalContinuationSource: OpticalSourceFeature;

fn opticalContinuationSourceIsActive() -> bool {
  return opticalContinuationSource.kind != OPTICAL_SOURCE_FEATURE_INVALID;
}

fn opticalClearContinuationSource() {
  opticalContinuationSource = opticalSourceFeatureInvalid();
}

fn opticalTriangleRepresentedId(triangleIndex: u32) -> u32 {
  return select(
    0u,
    triMaterialIds[triangleIndex].y,
    triangleIndex < arrayLength(&triMaterialIds),
  );
}

fn opticalTraversalSuppressesTriangle(
  triangleIndex: u32,
  tri: vec4u,
  a: vec3f,
  b: vec3f,
  c: vec3f,
) -> bool {
  if (!opticalContinuationSourceIsActive()) { return false; }
  let representedId = opticalTriangleRepresentedId(triangleIndex);
  if (representedId == 0u) { return false; }
  return opticalSourceFeatureSuppressesTriangle(
    opticalContinuationSource,
    opticalEncodedBoundaryId(triangleIndex, INVALID_TLAS_INSTANCE_INDEX),
    representedId,
    triangleIndex,
    a,
    b,
    c,
  );
}

${PT_WEBGPU_INTERSECTION_CORE_WGSL}

fn opticalSetContinuationSourceFromHit(hit: SceneHit) -> bool {
  if (hit.triIndex >= min(params.triangleCount, arrayLength(&indices))) {
    opticalClearContinuationSource();
    return false;
  }
  let tri = indices[hit.triIndex];
  let representedId = opticalTriangleRepresentedId(hit.triIndex);
  if (
    representedId == 0u ||
    tri.x >= arrayLength(&positions) ||
    tri.y >= arrayLength(&positions) ||
    tri.z >= arrayLength(&positions)
  ) {
    opticalClearContinuationSource();
    return false;
  }
  opticalContinuationSource = opticalCreateSourceFeature(
    opticalEncodedBoundaryId(hit.triIndex, INVALID_TLAS_INSTANCE_INDEX),
    representedId,
    hit.triIndex,
    hit.zeroEdgeMask,
    positions[tri.x].xyz,
    positions[tri.y].xyz,
    positions[tri.z].xyz,
  );
  return opticalContinuationSourceIsActive();
}

fn traceOpticalBoundaryClosest(
  ray: Ray,
  exclusiveMinT: f32,
  tMax: f32,
) -> OpticalBoundaryHit {
  var result: OpticalBoundaryHit;
  opticalResetBoundaryHit(&result, tMax);
  var localHit: OpticalLocalBoundaryHit;
  traceOpticalMeshBvhLocal(ray, exclusiveMinT, tMax, 0u, &localHit);
  if (!localHit.valid) {
    result.valid = false;
    return result;
  }
  if (!localHit.didHit) { return result; }
  let boundary = mediumBoundaryIdentity(
    localHit.triIndex, INVALID_TLAS_INSTANCE_INDEX,
  );
  if (
    !mediumBoundaryIsValid(boundary) ||
    localHit.triIndex >= arrayLength(&triMaterialIds)
  ) {
    result.valid = false;
    return result;
  }
  result.didHit = true;
  result.ambiguous = localHit.ambiguous;
  result.tangent = localHit.tangent;
  result.dist = localHit.dist;
  result.triIndex = localHit.triIndex;
  result.instanceIndex = INVALID_TLAS_INSTANCE_INDEX;
  result.baryVW = localHit.baryVW;
  result.matId = triMaterialIds[localHit.triIndex].x;
  result.boundary = boundary;
  result.frontFace = localHit.frontFace;
  return result;
}

fn traceClosestRaw(ray: Ray, tMin: f32, tMax: f32) -> SceneHit {
  var hit: SceneHit;
  _ = traceMeshBvh(ray, tMin, tMax, true, &hit, 0u, true);
  return hit;
}

fn hitMaterialId(hit: SceneHit) -> u32 {
  if (hit.triIndex < params.triangleCount) {
    return select(0u, triMaterialIds[hit.triIndex].x, hit.triIndex < arrayLength(&triMaterialIds));
  }
  return 0u;
}

fn nextSidedTraversalCursor(cursor: f32, hitDist: f32) -> f32 {
  _ = cursor;
  return hitDist;
}

fn traceClosest(ray: Ray, tMin: f32, tMax: f32) -> SceneHit {
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

fn traceAny(ray: Ray, tMin: f32, tMax: f32) -> bool {
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
      !materialShadowCastDisabled(matId)
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
`;
