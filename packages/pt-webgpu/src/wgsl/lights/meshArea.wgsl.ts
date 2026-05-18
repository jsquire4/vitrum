/**
 * Mesh-area light — BSDF-direction-mode intersection for sum-MIS (Veach 1997
 * Ch. 9). The legacy `sampleMeshAreaLight` per-light sampling helper has been
 * removed; the main kernel's per-light loop in `main.wgsl.ts` handles
 * one-light-pick area sampling inline (sweep finding F9).
 */
export const PT_WEBGPU_LIGHT_MESH_AREA_WGSL = /* wgsl */ `
// Intersect the BSDF sample ray against mesh area light index li.
// Accepts a light index so BSDF->light MIS can check all lights.
// Ref: Veach 1997 Ch. 9 -- sum-MIS over all lights (D9 decision).
fn intersectMeshAreaLightRay(li: u32, rayOrigin: vec3f, rayDir: vec3f, distOut: ptr<function, f32>, lightPdfOut: ptr<function, f32>) -> bool {
  let mb = li * 4u;
  let a = meshAreaLights[mb].xyz;
  let b = meshAreaLights[mb + 1u].xyz;
  let c = meshAreaLights[mb + 2u].xyz;
  let t = intersectTriangle(rayOrigin, rayDir, a, b, c);
  if (t <= 1e-4 || t >= INFINITY) {
    return false;
  }
  let lightNormal = safe_normalize(cross(b - a, c - a));
  let cosLight = max(dot(lightNormal, -rayDir), 0.0);
  if (cosLight <= 0.0) {
    return false;
  }
  let area = max(0.5 * length(cross(b - a, c - a)), 1e-6);
  *distOut = t;
  *lightPdfOut = (t * t) / max(cosLight * area, 1e-6);
  return true;
}
`;
