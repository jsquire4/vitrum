/**
 * Rect-area light — BSDF-direction-mode intersection for sum-MIS (Veach 1997
 * Ch. 9). The sampling-mode path is inlined in `main.wgsl.ts`'s per-light
 * loop (D9 decision: BSDF connections iterate every rect, sample picks one).
 */
export const PT_WEBGPU_LIGHT_RECT_AREA_WGSL = /* wgsl */ `
// Intersect the BSDF sample ray against rect area light index li.
// Accepts a light index so BSDF->light MIS can check all lights.
// Ref: Veach, E. PhD thesis, Stanford 1997, Ch. 9 -- power-heuristic MIS;
//      sum-MIS over all lights is unbiased (D9 decision).
fn intersectRectAreaLightRay(li: u32, rayOrigin: vec3f, rayDir: vec3f, distOut: ptr<function, f32>, lightPdfOut: ptr<function, f32>) -> bool {
  let rb = li * 4u;
  let rectPos = rectAreaLights[rb].xyz;
  let uAxis = rectAreaLights[rb + 1u].xyz;
  let vAxis = rectAreaLights[rb + 2u].xyz;
  let lightNormal = safe_normalize(cross(uAxis, vAxis));
  let denom = dot(lightNormal, rayDir);
  if (abs(denom) < 1e-6) {
    return false;
  }
  let t = dot(lightNormal, rectPos - rayOrigin) / denom;
  if (t <= 1e-4) {
    return false;
  }
  let p = rayOrigin + rayDir * t;
  let rel = p - rectPos;
  let uLen2 = max(dot(uAxis, uAxis), 1e-6);
  let vLen2 = max(dot(vAxis, vAxis), 1e-6);
  let uCoord = dot(rel, uAxis) / uLen2;
  let vCoord = dot(rel, vAxis) / vLen2;
  if (abs(uCoord) > 1.0 || abs(vCoord) > 1.0) {
    return false;
  }
  let cosLight = max(dot(lightNormal, -rayDir), 0.0);
  if (cosLight <= 0.0) {
    return false;
  }
  let area = max(4.0 * length(cross(uAxis, vAxis)), 1e-6);
  *distOut = t;
  *lightPdfOut = (t * t) / max(cosLight * area, 1e-6);
  return true;
}
`;
