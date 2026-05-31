import { PT_WEBGPU_INTERSECTION_CORE_WGSL } from './intersectionCore.wgsl.js';

export const PT_WEBGPU_PATH_TRACE_INTERSECTION_LITE_WGSL = /* wgsl */ `
${PT_WEBGPU_INTERSECTION_CORE_WGSL}

fn traceClosest(ray: Ray, tMin: f32, tMax: f32) -> SceneHit {
  var hit: SceneHit;
  _ = traceMeshBvh(ray, tMin, tMax, true, &hit, 0u, true);
  return hit;
}

fn traceAny(ray: Ray, tMin: f32, tMax: f32) -> bool {
  var hit: SceneHit;
  return traceMeshBvh(ray, tMin, tMax, false, &hit, 0u, false);
}

fn hitMaterialId(hit: SceneHit) -> u32 {
  if (hit.triIndex < params.triangleCount) {
    return select(0u, triMaterialIds[hit.triIndex], hit.triIndex < arrayLength(&triMaterialIds));
  }
  return 0u;
}
`;
