/**
 * BDPT light-subpath compute — bounce 0 (emitter) + extension bounces k>0.
 * Bounce 0 uses power-weighted discrete emitter pick (fork `randomLightSample` parity).
 * @see packages/three-gpu-pathtracer/.../bdpt_light_subpath.glsl.js
 */
export const PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL = /* wgsl */ `
fn bdptLightLuminance(c: vec3f) -> f32 {
  return max(dot(c, vec3f(0.2126, 0.7152, 0.0722)), 1e-20);
}

fn bdptHasEnvironmentEmitter() -> bool {
  return hasEnvironmentMap() || params.environmentSun.w > 1e-6;
}

fn bdptEnvironmentPower() -> f32 {
  if (hasEnvironmentMap()) {
    let dims = environmentDimensions();
    let count = dims.x * dims.y;
    if (count > 0u && arrayLength(&environmentMapCdf) >= count + 1u) {
      return max(environmentMapCdf[count], 1e-20);
    }
  }
  if (params.environmentSun.w > 1e-6) {
    return max(params.environmentSun.w, 1e-20) * (4.0 * PI);
  }
  return 1e-20;
}

fn bdptEmitterCount() -> u32 {
  var n = 0u;
  if (params.lightDir.w > 1e-6) {
    n = n + 1u;
  }
  n = n + params.pointLightCount;
  n = n + params.spotLightCount;
  n = n + params.rectAreaLightCount;
  n = n + params.meshAreaLightCount;
  if (bdptHasEnvironmentEmitter()) {
    n = n + 1u;
  }
  return n;
}

fn bdptEmitterPower(flatIdx: u32) -> f32 {
  var cur = 0u;
  if (params.lightDir.w > 1e-6) {
    if (cur == flatIdx) {
      return bdptLightLuminance(vec3f(params.lightDir.w));
    }
    cur = cur + 1u;
  }
  for (var pi = 0u; pi < params.pointLightCount; pi = pi + 1u) {
    if (cur == flatIdx) {
      let rad = pointLights[pi * 2u + 1u].rgb;
      return bdptLightLuminance(rad);
    }
    cur = cur + 1u;
  }
  for (var si = 0u; si < params.spotLightCount; si = si + 1u) {
    if (cur == flatIdx) {
      let sb = si * 3u;
      let srad = spotLights[sb + 2u].rgb;
      return bdptLightLuminance(srad);
    }
    cur = cur + 1u;
  }
  for (var ri = 0u; ri < params.rectAreaLightCount; ri = ri + 1u) {
    if (cur == flatIdx) {
      let rb = ri * 4u;
      let ru = rectAreaLights[rb + 1u].xyz;
      let rv = rectAreaLights[rb + 2u].xyz;
      let rr = rectAreaLights[rb + 3u].rgb;
      let area = max(4.0 * length(cross(ru, rv)), 1e-6);
      return area * bdptLightLuminance(rr);
    }
    cur = cur + 1u;
  }
  for (var mi = 0u; mi < params.meshAreaLightCount; mi = mi + 1u) {
    if (cur == flatIdx) {
      let mb = mi * 4u;
      let a = meshAreaLights[mb].xyz;
      let b = meshAreaLights[mb + 1u].xyz;
      let c = meshAreaLights[mb + 2u].xyz;
      let mr = meshAreaLights[mb + 3u].rgb;
      let area = max(0.5 * length(cross(b - a, c - a)), 1e-6);
      return area * bdptLightLuminance(mr);
    }
    cur = cur + 1u;
  }
  if (bdptHasEnvironmentEmitter()) {
    if (cur == flatIdx) {
      return bdptEnvironmentPower();
    }
  }
  return 1e-20;
}

fn bdptPickEmitterFlat(rng: ptr<function, u32>, totalPower: f32, emitterCount: u32) -> u32 {
  if (emitterCount == 0u) {
    return 0u;
  }
  let u = rand_f32(rng) * totalPower;
  var cum = 0.0;
  for (var i = 0u; i < emitterCount; i = i + 1u) {
    let w = bdptEmitterPower(i);
    cum = cum + w;
    if (u <= cum) {
      return i;
    }
  }
  return emitterCount - 1u;
}

fn bdptWriteInvalid(col: i32) {
  bdptLightPath[bdptLightPathIndex(col, 0u)] = vec4f(0.0, 0.0, 0.0, BDPT_KIND_INVALID);
  bdptLightPath[bdptLightPathIndex(col, 1u)] = vec4f(0.0);
  bdptLightPath[bdptLightPathIndex(col, 2u)] = vec4f(0.0);
}

fn bdptFinishBounce0(
  col: i32,
  emitPos: vec3f,
  emitNormal: vec3f,
  emitRad: vec3f,
  pdfLight: f32,
  rng: ptr<function, u32>,
) {
  let hemi = cosineHemisphereSample(rng, emitNormal);
  let cosEmit = max(dot(emitNormal, hemi.wi), 0.0);
  let pdfHemi = hemi.pdf;
  let pdfJoint = max(pdfLight * pdfHemi, 1e-8);
  let emitThroughput = emitRad * cosEmit / pdfJoint;
  bdptLightPath[bdptLightPathIndex(col, 0u)] = vec4f(emitPos, 0.0);
  bdptLightPath[bdptLightPathIndex(col, 1u)] = vec4f(emitNormal, pdfJoint);
  bdptLightPath[bdptLightPathIndex(col, 2u)] = vec4f(emitThroughput, pdfHemi);
}

fn bdptWriteBounce0(col: i32, rng: ptr<function, u32>) {
  let emitterCount = bdptEmitterCount();
  if (emitterCount == 0u) {
    bdptWriteInvalid(col);
    return;
  }
  var totalPower = 0.0;
  for (var i = 0u; i < emitterCount; i = i + 1u) {
    totalPower = totalPower + bdptEmitterPower(i);
  }
  let flat = bdptPickEmitterFlat(rng, totalPower, emitterCount);
  let discretePdf = bdptEmitterPower(flat) / max(totalPower, 1e-20);

  var cur = 0u;
  if (params.lightDir.w > 1e-6) {
    if (cur == flat) {
      let lightDir = safe_normalize(params.lightDir.xyz);
      let emitPos = -lightDir * 50.0;
      let irr = params.lightDir.w;
      bdptFinishBounce0(col, emitPos, lightDir, vec3f(irr), discretePdf, rng);
      return;
    }
    cur = cur + 1u;
  }
  for (var pi = 0u; pi < params.pointLightCount; pi = pi + 1u) {
    if (cur == flat) {
      let pos = pointLights[pi * 2u].xyz;
      let rad = pointLights[pi * 2u + 1u].rgb;
      bdptFinishBounce0(col, pos, vec3f(0.0, 1.0, 0.0), rad, discretePdf, rng);
      return;
    }
    cur = cur + 1u;
  }
  for (var si = 0u; si < params.spotLightCount; si = si + 1u) {
    if (cur == flat) {
      let sb = si * 3u;
      let spos = spotLights[sb].xyz;
      let saxis = spotLights[sb + 1u];
      let srad = spotLights[sb + 2u].rgb;
      let spotDir = safe_normalize(saxis.xyz);
      bdptFinishBounce0(col, spos, spotDir, srad, discretePdf, rng);
      return;
    }
    cur = cur + 1u;
  }
  for (var ri = 0u; ri < params.rectAreaLightCount; ri = ri + 1u) {
    if (cur == flat) {
      let rb = ri * 4u;
      let rpos = rectAreaLights[rb].xyz;
      let ru = rectAreaLights[rb + 1u].xyz;
      let rv = rectAreaLights[rb + 2u].xyz;
      let rr = rectAreaLights[rb + 3u].rgb;
      let u = rand_f32(rng) * 2.0 - 1.0;
      let v = rand_f32(rng) * 2.0 - 1.0;
      let emitPos = rpos + ru * u + rv * v;
      let emitNormal = safe_normalize(cross(ru, rv));
      bdptFinishBounce0(col, emitPos, emitNormal, rr, discretePdf, rng);
      return;
    }
    cur = cur + 1u;
  }
  for (var mi = 0u; mi < params.meshAreaLightCount; mi = mi + 1u) {
    if (cur == flat) {
      let mb = mi * 4u;
      let a = meshAreaLights[mb].xyz;
      let b = meshAreaLights[mb + 1u].xyz;
      let c = meshAreaLights[mb + 2u].xyz;
      let mr = meshAreaLights[mb + 3u].rgb;
      let r1 = rand_f32(rng);
      let r2 = rand_f32(rng);
      let su = sqrt(r1);
      let uu = 1.0 - su;
      let vv = r2 * su;
      let ww = 1.0 - uu - vv;
      let emitPos = a * uu + b * vv + c * ww;
      let e1 = b - a;
      let e2 = c - a;
      let n = cross(e1, e2);
      let nLen = length(n);
      if (nLen < 1e-8) {
        bdptWriteInvalid(col);
        return;
      }
      let emitNormal = n / nLen;
      bdptFinishBounce0(col, emitPos, emitNormal, mr, discretePdf, rng);
      return;
    }
    cur = cur + 1u;
  }
  if (bdptHasEnvironmentEmitter() && cur == flat) {
    let envSample = sampleEnvironmentImportance(rng);
    if (envSample.pdf > 1e-8) {
      let pdfLight = discretePdf * envSample.pdf;
      let emitDir = envSample.wi;
      let emitPos = -emitDir * 50.0;
      bdptFinishBounce0(col, emitPos, emitDir, envSample.value, pdfLight, rng);
      return;
    }
    if (params.environmentSun.w > 1e-6) {
      let sunDir = safe_normalize(params.environmentSun.xyz);
      let emitPos = -sunDir * 50.0;
      bdptFinishBounce0(col, emitPos, sunDir, vec3f(params.environmentSun.w), discretePdf, rng);
      return;
    }
    bdptWriteInvalid(col);
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
  let v0prev = bdptLightPath[bdptLightPathIndex(prevCol, 0u)];
  let v1prev = bdptLightPath[bdptLightPathIndex(prevCol, 1u)];
  let v2prev = bdptLightPath[bdptLightPathIndex(prevCol, 2u)];
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
  // Store SOLID-ANGLE pdfs (NO baked-in geometry term). The full Veach §10.3
  // connection sweep converts SA→area on the fly via ConvertDensity (PBRT
  // Vertex::ConvertDensity), so baking G here would double-apply the Jacobian.
  let pdfFwd = pdfScatter;                       // SA forward (Lambertian cosθ/π)
  let toPrev = safe_normalize(prevPos - newPos);
  let cosRev = max(dot(newNormal, toPrev), 0.0);
  let pdfRev = cosRev * INV_PI;                  // SA reverse (Lambertian cosθ/π)
  bdptLightPath[bdptLightPathIndex(col, 0u)] = vec4f(newPos, 0.0);
  bdptLightPath[bdptLightPathIndex(col, 1u)] = vec4f(newNormal, pdfFwd);
  bdptLightPath[bdptLightPathIndex(col, 2u)] = vec4f(newThroughput, pdfRev);
}
`;
