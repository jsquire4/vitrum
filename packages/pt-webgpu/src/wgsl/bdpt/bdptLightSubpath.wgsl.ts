/**
 * BDPT light-subpath compute — bounce 0 (emitter) + extension bounces k>0.
 * @see packages/three-gpu-pathtracer/.../bdpt_light_subpath.glsl.js
 */
export const PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL = /* wgsl */ `
fn bdptWriteInvalid(col: i32) {
  textureStore(bdptLightPath, vec2i(col, 0), vec4f(0.0, 0.0, 0.0, BDPT_KIND_INVALID));
  textureStore(bdptLightPath, vec2i(col, 1), vec4f(0.0));
  textureStore(bdptLightPath, vec2i(col, 2), vec4f(0.0));
}

fn bdptWriteBounce0(col: i32, rng: ptr<function, u32>) {
  if (params.meshAreaLightCount > 0u) {
    let mb = 0u;
    let a = meshAreaLights[mb].xyz;
    let b = meshAreaLights[mb + 1u].xyz;
    let c = meshAreaLights[mb + 2u].xyz;
    let rad = meshAreaLights[mb + 3u].rgb;
    let emitPos = (a + b + c) / 3.0;
    let e1 = b - a;
    let e2 = c - a;
    let n = cross(e1, e2);
    let nLen = length(n);
    if (nLen < 1e-8) {
      bdptWriteInvalid(col);
      return;
    }
    let emitNormal = n / nLen;
    let hemi = cosineHemisphereSample(rng, emitNormal);
    let cosEmit = max(dot(emitNormal, hemi.wi), 0.0);
    let pdfHemi = hemi.pdf;
    let pdfJoint = max(pdfHemi, 1e-8);
    let emitThroughput = rad * cosEmit / pdfJoint;
    textureStore(bdptLightPath, vec2i(col, 0), vec4f(emitPos, 0.0));
    textureStore(bdptLightPath, vec2i(col, 1), vec4f(emitNormal, pdfJoint));
    textureStore(bdptLightPath, vec2i(col, 2), vec4f(emitThroughput, pdfHemi));
    return;
  }
  if (params.pointLightCount > 0u) {
    let pos = pointLights[0].xyz;
    let rad = pointLights[1].rgb;
    let pdfFwd = 1.0;
    textureStore(bdptLightPath, vec2i(col, 0), vec4f(pos, 0.0));
    textureStore(bdptLightPath, vec2i(col, 1), vec4f(0.0, 1.0, 0.0, pdfFwd));
    textureStore(bdptLightPath, vec2i(col, 2), vec4f(rad / pdfFwd, pdfFwd));
    return;
  }
  if (params.lightDir.w > 1e-6) {
    let lightDir = safe_normalize(params.lightDir.xyz);
    let emitPos = -lightDir * 50.0;
    let irr = params.lightDir.w;
    let pdfFwd = 1.0;
    textureStore(bdptLightPath, vec2i(col, 0), vec4f(emitPos, 0.0));
    textureStore(bdptLightPath, vec2i(col, 1), vec4f(lightDir, pdfFwd));
    textureStore(bdptLightPath, vec2i(col, 2), vec4f(vec3f(irr / pdfFwd), pdfFwd));
    return;
  }
  bdptWriteInvalid(col);
}

@compute @workgroup_size(1, 1, 1)
fn bdptExtendLightSubpath(@builtin(global_invocation_id) gid: vec3u) {
  if (params.bdptEnabled == 0u) {
    return;
  }
  let col = i32(gid.x);
  let maxB = i32(params.bdptMaxLightBounces);
  if (col < 0 || col >= maxB) {
    return;
  }
  var rng = pcgInit(u32(col), 0u, params.frameSeed ^ params.frameIndex);

  if (col == 0) {
    bdptWriteBounce0(col, &rng);
    return;
  }

  let prevCol = col - 1;
  let v0prev = textureLoad(bdptLightPath, vec2i(prevCol, 0), 0);
  let v1prev = textureLoad(bdptLightPath, vec2i(prevCol, 1), 0);
  let v2prev = textureLoad(bdptLightPath, vec2i(prevCol, 2), 0);
  if (v0prev.w == BDPT_KIND_INVALID) {
    bdptWriteInvalid(col);
    return;
  }
  let prevPos = v0prev.xyz;
  let prevNormal = v1prev.xyz;
  let prevThroughput = v2prev.xyz;
  let hemi = cosineHemisphereSample(&rng, prevNormal);
  let scatterDir = hemi.wi;
  let cosScatter = max(dot(prevNormal, scatterDir), 0.0);
  let pdfScatter = hemi.pdf;
  if (pdfScatter <= 0.0) {
    bdptWriteInvalid(col);
    return;
  }
  var ray: Ray;
  ray.origin = prevPos + prevNormal * 1e-4;
  ray.direction = scatterDir;
  let hit = traceClosest(ray, 1e-4, 1e30);
  if (!hit.didHit) {
    bdptWriteInvalid(col);
    return;
  }
  let matIdx = hitMaterialId(hit);
  let mat = decodeMaterial(matIdx);
  if (mat.transmission > 0.5 && mat.roughness < 0.05) {
    bdptWriteInvalid(col);
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
