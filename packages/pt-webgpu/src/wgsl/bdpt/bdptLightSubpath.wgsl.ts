/**
 * BDPT light-subpath extension (bounce k>0) — compute pass.
 * Bounce 0 is filled on the host (CPU) or by the fork emitter sampler.
 * @see packages/three-gpu-pathtracer/.../bdpt_light_subpath.glsl.js
 */
export const PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL = /* wgsl */ `
@compute @workgroup_size(1, 1, 1)
fn bdptExtendLightSubpath(@builtin(global_invocation_id) gid: vec3u) {
  if (params.bdptEnabled == 0u) {
    return;
  }
  let col = i32(gid.x);
  let maxB = i32(params.bdptMaxLightBounces);
  if (col <= 0 || col >= maxB) {
    return;
  }
  let prevCol = col - 1;
  let v0prev = textureLoad(bdptLightPath, vec2i(prevCol, 0), 0);
  let v1prev = textureLoad(bdptLightPath, vec2i(prevCol, 1), 0);
  let v2prev = textureLoad(bdptLightPath, vec2i(prevCol, 2), 0);
  if (v0prev.w == BDPT_KIND_INVALID) {
    textureStore(bdptLightPath, vec2i(col, 0), vec4f(0.0, 0.0, 0.0, BDPT_KIND_INVALID));
    textureStore(bdptLightPath, vec2i(col, 1), vec4f(0.0));
    textureStore(bdptLightPath, vec2i(col, 2), vec4f(0.0));
    return;
  }
  var rng = pcgInit(u32(col), 0u, params.frameSeed ^ params.frameIndex);
  let prevPos = v0prev.xyz;
  let prevNormal = v1prev.xyz;
  let prevThroughput = v2prev.xyz;
  let hemi = cosineHemisphereSample(&rng, prevNormal);
  let scatterDir = hemi.wi;
  let cosScatter = max(dot(prevNormal, scatterDir), 0.0);
  let pdfScatter = hemi.pdf;
  if (pdfScatter <= 0.0) {
    textureStore(bdptLightPath, vec2i(col, 0), vec4f(0.0, 0.0, 0.0, BDPT_KIND_INVALID));
    textureStore(bdptLightPath, vec2i(col, 1), vec4f(0.0));
    textureStore(bdptLightPath, vec2i(col, 2), vec4f(0.0));
    return;
  }
  var ray: Ray;
  ray.origin = prevPos + prevNormal * 1e-4;
  ray.direction = scatterDir;
  let hit = traceClosest(ray, 1e-4, 1e30);
  if (!hit.didHit) {
    textureStore(bdptLightPath, vec2i(col, 0), vec4f(0.0, 0.0, 0.0, BDPT_KIND_INVALID));
    textureStore(bdptLightPath, vec2i(col, 1), vec4f(0.0));
    textureStore(bdptLightPath, vec2i(col, 2), vec4f(0.0));
    return;
  }
  let matIdx = hitMaterialId(hit);
  let mat = decodeMaterial(matIdx);
  if (mat.transmission > 0.5 && mat.roughness < 0.05) {
    textureStore(bdptLightPath, vec2i(col, 0), vec4f(0.0, 0.0, 0.0, BDPT_KIND_INVALID));
    textureStore(bdptLightPath, vec2i(col, 1), vec4f(0.0));
    textureStore(bdptLightPath, vec2i(col, 2), vec4f(0.0));
    return;
  }
  let newPos = ray.origin + ray.direction * hit.dist;
  let newNormal = safe_normalize(hit.normal);
  let newThroughput = prevThroughput * mat.baseColor * cosScatter / pdfScatter;
  let gTerm = bdptGeometricTerm(prevPos, prevNormal, newPos, newNormal);
  let pdfFwd = pdfScatter * max(gTerm, 0.0);
  let toPrev = safe_normalize(prevPos - newPos);
  let cosRev = max(dot(newNormal, toPrev), 0.0);
  let pdfRev = (cosRev * INV_PI) * max(gTerm, 0.0);
  textureStore(bdptLightPath, vec2i(col, 0), vec4f(newPos, 0.0));
  textureStore(bdptLightPath, vec2i(col, 1), vec4f(newNormal, pdfFwd));
  textureStore(bdptLightPath, vec2i(col, 2), vec4f(newThroughput, pdfRev));
}
`;
