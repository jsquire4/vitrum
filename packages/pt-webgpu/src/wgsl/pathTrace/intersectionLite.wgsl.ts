import { PT_WEBGPU_INTERSECTION_CORE_WGSL } from './intersectionCore.wgsl.js';

export const PT_WEBGPU_PATH_TRACE_INTERSECTION_LITE_WGSL = /* wgsl */ `
${PT_WEBGPU_INTERSECTION_CORE_WGSL}

fn traceClosestRaw(ray: Ray, tMin: f32, tMax: f32) -> SceneHit {
  var hit: SceneHit;
  _ = traceMeshBvh(ray, tMin, tMax, true, &hit, 0u, true);
  return hit;
}

fn hitMaterialId(hit: SceneHit) -> u32 {
  if (hit.triIndex < params.triangleCount) {
    return select(0u, triMaterialIds[hit.triIndex], hit.triIndex < arrayLength(&triMaterialIds));
  }
  return 0u;
}

fn nextSidedTraversalCursor(cursor: f32, hitDist: f32) -> f32 {
  let step = max(max(params.triIntersectEpsilon, 1e-5), abs(hitDist) * 1e-6);
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

fn traceAny(ray: Ray, tMin: f32, tMax: f32) -> bool {
  var cursor = tMin;
  loop {
    let hit = traceClosestRaw(ray, cursor, tMax);
    if (!hit.didHit) { return false; }
    let matId = hitMaterialId(hit);
    if (
      materialAcceptsSidedHit(matId, hit.frontFace) &&
      !materialShadowCastDisabled(matId)
    ) {
      return true;
    }
    let nextCursor = nextSidedTraversalCursor(cursor, hit.dist);
    if (!(nextCursor > cursor) || !(nextCursor < tMax)) { return false; }
    cursor = nextCursor;
  }
  return false;
}
`;
