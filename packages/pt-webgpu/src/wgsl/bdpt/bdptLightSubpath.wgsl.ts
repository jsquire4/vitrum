/**
 * BDPT light-subpath compute — bounce 0 (emitter) + extension bounces k>0.
 * Bounce 0 uses the power-weighted discrete emitter pick used by the native
 * WebGL2 parity path.
 */
export const PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL = /* wgsl */ `
fn bdptLightLuminance(c: vec3f) -> f32 {
  // Canonical Rec.709 luminance() from LUMINANCE_WGSL (@vitrum/shared-samplers),
  // composed into the trace shader before this module (pathTraceBruteforce
  // .wgsl.ts:82). Keep the 1e-20 floor for the power-weighted emitter pick.
  return max(luminance(c), 1e-20);
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

fn bdptDistantEmitterPosition(lightDir: vec3f) -> vec3f {
  let center = vec3f(params.sceneCenterX, params.sceneCenterY, params.sceneCenterZ);
  let dist = max(params.sceneRadius * 4.0, 1.0);
  return center - lightDir * dist;
}

fn bdptEmitterCount() -> u32 {
  var n = params.directionalLightCount;
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
  for (var di = 0u; di < params.directionalLightCount; di = di + 1u) {
    if (cur == flatIdx) {
      let dBase = di * 2u;
      let dIrrMean = directionalLights[dBase + 1u];
      return bdptLightLuminance(dIrrMean.rgb);
    }
    cur = cur + 1u;
  }
  for (var pi = 0u; pi < params.pointLightCount; pi = pi + 1u) {
    if (cur == flatIdx) {
      // H51-D: stride 3; radiance is at slot 1
      let rad = pointLights[pi * 3u + 1u].rgb;
      return bdptLightLuminance(rad);
    }
    cur = cur + 1u;
  }
  for (var si = 0u; si < params.spotLightCount; si = si + 1u) {
    if (cur == flatIdx) {
      // H51-D: stride 4; radiance is at slot 2 (.rgb)
      let sb = si * 4u;
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
      let rshapeB = rectAreaLights[rb + 3u];
      let rr = rshapeB.rgb;
      // Disc area = π·|u|²; rect area = 4·|u×v|.
      let isDiscB = abs(rshapeB.w - 1.0) < 0.5;
      let area = select(
        max(4.0 * length(cross(ru, rv)), 1e-6),
        max(PI * dot(ru, ru), 1e-6),
        isDiscB,
      );
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

// A9 — row-3 light-vertex BSDF record. .w = matId (>=0 ⇒ a real surface vertex
// whose BSDF the connection evaluates; -1 ⇒ legacy pseudo-emitter Lambertian
// profile; -2 ⇒ finite area emitter whose throughput already includes pdfArea).
// .xyz = wo toward the PREVIOUS light vertex (the eval's outgoing direction).
const BDPT_LV_EMITTER_MATID: f32 = -1.0;
const BDPT_LV_AREA_EMITTER_MATID: f32 = -2.0;
fn bdptWriteLvBsdf(col: i32, matId: f32, woTowardPrev: vec3f) {
  bdptLightPath[bdptLightPathIndex(col, 3u)] = vec4f(woTowardPrev, matId);
}

// Texture-map payload for surface light vertices. Row 4 keeps the hit-local
// coordinate system needed by the same material sampling helpers as the eye path.
fn bdptWriteLvMaterialPayload(col: i32, triIndex: u32, baryVW: vec2f, instanceIndex: u32) {
  bdptLightPath[bdptLightPathIndex(col, 4u)] = vec4f(bitcast<f32>(triIndex), baryVW.x, baryVW.y, bitcast<f32>(instanceIndex));
}

fn bdptClearLvMaterialPayload(col: i32) {
  bdptLightPath[bdptLightPathIndex(col, 4u)] = vec4f(0.0);
}

struct BdptSampledMaterial {
  baseColor: vec3f,
  roughness: f32,
  metallic: f32,
  transmission: f32,
  ior: f32,
  clearcoat: f32,
  clearcoatRoughness: f32,
  sheen: f32,
  sheenRoughness: f32,
  sheenColor: vec3f,
  iridescence: f32,
  iridescenceIor: f32,
  iridescenceThicknessMin: f32,
  iridescenceThicknessMax: f32,
  specularColor: vec3f,
  specularIntensity: f32,
  anisotropy: f32,
  anisotropyRotation: f32,
  clearcoatNormal: vec3f,
}

fn bdptSampleMaterialAtPayload(matId: u32, payload: vec4f, shadingNormal: vec3f) -> BdptSampledMaterial {
  let triIndex = bitcast<u32>(payload.x);
  let baryVW = payload.yz;
  let instanceIndex = bitcast<u32>(payload.w);
  let mat = decodeMaterial(matId);
  var out: BdptSampledMaterial;
  out.baseColor = mat.baseColor * sampleVertexColor(triIndex, baryVW).rgb * sampleBaseColorTexture(matId, triIndex, baryVW).rgb;
  out.baseColor = out.baseColor * sampleAoFactor(matId, triIndex, baryVW);
  let orm = sampleOrmTexture(matId, triIndex, baryVW);
  out.roughness = clamp(mat.roughness * orm.g, 0.02, 1.0);
  out.metallic = clamp(mat.metallic * orm.b, 0.0, 1.0);
  out.transmission = clamp(mat.transmission * sampleTransmissionTexture(matId, triIndex, baryVW), 0.0, 1.0);
  out.ior = mat.ior;
  out.clearcoat = clamp(mat.clearcoat * sampleClearcoatTexture(matId, triIndex, baryVW), 0.0, 1.0);
  out.clearcoatRoughness = clamp(mat.clearcoatRoughness * sampleClearcoatRoughnessTexture(matId, triIndex, baryVW), 0.0, 1.0);
  out.sheen = mat.sheen;
  out.sheenRoughness = clamp(mat.sheenRoughness * sampleSheenRoughnessTexture(matId, triIndex, baryVW), 0.0, 1.0);
  out.sheenColor = clamp(mat.sheenColor * sampleSheenColorTexture(matId, triIndex, baryVW), vec3f(0.0), vec3f(1.0));
  out.iridescence = clamp(mat.iridescence * sampleIridescenceTexture(matId, triIndex, baryVW), 0.0, 1.0);
  let iridescenceThicknessSample = sampleIridescenceThicknessTexture(matId, triIndex, baryVW);
  out.iridescenceThicknessMin = mat.iridescenceThicknessMin;
  out.iridescenceThicknessMax = mat.iridescenceThicknessMax;
  if (iridescenceThicknessSample >= 0.0) {
    let iridescenceThickness = mix(mat.iridescenceThicknessMin, mat.iridescenceThicknessMax, iridescenceThicknessSample);
    out.iridescenceThicknessMin = iridescenceThickness;
    out.iridescenceThicknessMax = iridescenceThickness;
    if (iridescenceThickness <= 0.0) { out.iridescence = 0.0; }
  }
  out.iridescenceIor = mat.iridescenceIor;
  out.specularColor = clamp(mat.specularColor * sampleSpecularColorTexture(matId, triIndex, baryVW), vec3f(0.0), vec3f(1.0));
  out.specularIntensity = clamp(mat.specularIntensity * sampleSpecularIntensityTexture(matId, triIndex, baryVW), 0.0, 1.0);
  out.anisotropy = materialAnisotropy(matId, triIndex, baryVW);
  out.anisotropyRotation = materialAnisotropyRotation(matId, triIndex, baryVW);
  out.clearcoatNormal = applyClearcoatNormalMap(matId, triIndex, baryVW, shadingNormal, instanceIndex);
  return out;
}

fn bdptWriteInvalid(col: i32) {
  bdptLightPath[bdptLightPathIndex(col, 0u)] = vec4f(0.0, 0.0, 0.0, BDPT_KIND_INVALID);
  bdptLightPath[bdptLightPathIndex(col, 1u)] = vec4f(0.0);
  bdptLightPath[bdptLightPathIndex(col, 2u)] = vec4f(0.0);
  bdptLightPath[bdptLightPathIndex(col, 3u)] = vec4f(0.0, 0.0, 0.0, BDPT_LV_EMITTER_MATID);
  bdptClearLvMaterialPayload(col);
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
  // Emitter vertex → Lambertian/emission profile in the connection (matId < 0).
  bdptWriteLvBsdf(col, BDPT_LV_EMITTER_MATID, emitNormal);
  bdptClearLvMaterialPayload(col);
}

fn bdptFinishBounce0Area(
  col: i32,
  emitPos: vec3f,
  emitNormal: vec3f,
  emitRad: vec3f,
  pdfLight: f32,
  pdfArea: f32,
  rng: ptr<function, u32>,
) {
  let hemi = cosineHemisphereSample(rng, emitNormal);
  let pdfHemi = hemi.pdf;
  let pdfPos = max(pdfLight * pdfArea, 1e-8);
  let emitThroughput = emitRad / pdfPos;
  bdptLightPath[bdptLightPathIndex(col, 0u)] = vec4f(emitPos, 0.0);
  bdptLightPath[bdptLightPathIndex(col, 1u)] = vec4f(emitNormal, pdfPos);
  bdptLightPath[bdptLightPathIndex(col, 2u)] = vec4f(emitThroughput, pdfHemi);
  bdptWriteLvBsdf(col, BDPT_LV_AREA_EMITTER_MATID, emitNormal);
  bdptClearLvMaterialPayload(col);
}

// A9 — ISOTROPIC point-emitter bounce-0 finish. A point light emits uniformly over
// the FULL sphere (intensity I, no cosine falloff at the source) — modelling it as
// a cosine-weighted hemisphere about a fabricated normal (0,1,0) was wrong (it
// biased the emitted direction distribution + threw away the lower hemisphere). The
// directional sampling pdf is the uniform-sphere density 1/(4π); there is no cosEmit
// factor (a point emitter has no surface cosine). The first scatter vertex's stored
// emitNormal is set to the SAMPLED direction so the k>0 extension's
// cosineHemisphere(prevNormal) keeps a sensible local frame at the first surface hit
// — but the EMITTER vertex itself carries the isotropic pdf. emitRad here is the
// point-light radiant intensity (radiance·sr is handled by the 1/dist² at the
// connection's geometry term, the same as the megakernel NEE).
fn bdptFinishBounce0Isotropic(
  col: i32,
  emitPos: vec3f,
  emitRad: vec3f,
  pdfLight: f32,
  rng: ptr<function, u32>,
) {
  let dir = uniformSphere(vec2f(rand_f32(rng), rand_f32(rng)));
  let pdfDir = 0.25 * INV_PI;                 // 1/(4π), uniform sphere
  let pdfJoint = max(pdfLight * pdfDir, 1e-8);
  // No surface cosine for an isotropic point source.
  let emitThroughput = emitRad / pdfJoint;
  // Store the sampled direction as the emitter "normal" so the first extension
  // bounce scatters about a consistent outgoing frame.
  bdptLightPath[bdptLightPathIndex(col, 0u)] = vec4f(emitPos, 0.0);
  bdptLightPath[bdptLightPathIndex(col, 1u)] = vec4f(dir, pdfJoint);
  bdptLightPath[bdptLightPathIndex(col, 2u)] = vec4f(emitThroughput, pdfDir);
  // Point emitter vertex → emitter profile in the connection (matId < 0).
  bdptWriteLvBsdf(col, BDPT_LV_EMITTER_MATID, dir);
  bdptClearLvMaterialPayload(col);
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
  for (var di = 0u; di < params.directionalLightCount; di = di + 1u) {
    if (cur == flat) {
      let dBase = di * 2u;
      let dDirAD = directionalLights[dBase];
      let dIrrMean = directionalLights[dBase + 1u];
      let lightDir = safe_normalize(dDirAD.xyz);
      let emitPos = bdptDistantEmitterPosition(lightDir);
      bdptFinishBounce0(col, emitPos, lightDir, dIrrMean.rgb, discretePdf, rng);
      return;
    }
    cur = cur + 1u;
  }
  for (var pi = 0u; pi < params.pointLightCount; pi = pi + 1u) {
    if (cur == flat) {
      // H51-D: stride 3; position at slot 0, radiance at slot 1
      let pos = pointLights[pi * 3u].xyz;
      let rad = pointLights[pi * 3u + 1u].rgb;
      // A9 — ISOTROPIC point emitter (uniform sphere, no cosine-up about a
      // fabricated normal). See bdptFinishBounce0Isotropic.
      bdptFinishBounce0Isotropic(col, pos, rad, discretePdf, rng);
      return;
    }
    cur = cur + 1u;
  }
  for (var si = 0u; si < params.spotLightCount; si = si + 1u) {
    if (cur == flat) {
      // H51-D: stride 4; position at slot 0, dir+cosOuter at slot 1, radiance+cosInner at slot 2
      let sb = si * 4u;
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
      let rshapeS = rectAreaLights[rb + 3u];
      let rr = rshapeS.rgb;
      let isDiscS = abs(rshapeS.w - 1.0) < 0.5;
      let xi1s = rand_f32(rng);
      let xi2s = rand_f32(rng);
      var emitPos: vec3f;
      var areaS: f32;
      if (isDiscS) {
        let rrad = length(ru);
        let a = xi1s * 2.0 - 1.0;
        let b = xi2s * 2.0 - 1.0;
        var cr: f32; var cphi: f32;
        if (abs(a) >= abs(b)) {
          cr = a; cphi = (PI / 4.0) * (b / max(abs(a), 1e-9));
        } else {
          cr = b; cphi = (PI / 2.0) - (PI / 4.0) * (a / max(abs(b), 1e-9));
        }
        emitPos = rpos + ru * (cr * cos(cphi)) + rv * (cr * sin(cphi));
        areaS = max(PI * rrad * rrad, 1e-6);
      } else {
        emitPos = rpos + ru * (xi1s * 2.0 - 1.0) + rv * (xi2s * 2.0 - 1.0);
        areaS = max(4.0 * length(cross(ru, rv)), 1e-6);
      }
      let emitNormal = safe_normalize(cross(ru, rv));
      bdptFinishBounce0Area(col, emitPos, emitNormal, rr, discretePdf, 1.0 / areaS, rng);
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
      let areaM = max(0.5 * nLen, 1e-6);
      bdptFinishBounce0Area(col, emitPos, emitNormal, mr, discretePdf, 1.0 / areaM, rng);
      return;
    }
    cur = cur + 1u;
  }
  if (bdptHasEnvironmentEmitter() && cur == flat) {
    let envSample = sampleEnvironmentImportance(rng);
    if (envSample.pdf > 1e-8) {
      let pdfLight = discretePdf * envSample.pdf;
      let emitDir = envSample.wi;
      let emitPos = bdptDistantEmitterPosition(emitDir);
      bdptFinishBounce0(col, emitPos, emitDir, envSample.value, pdfLight, rng);
      return;
    }
    if (params.environmentSun.w > 1e-6) {
      let sunDir = safe_normalize(params.environmentSun.xyz);
      let emitPos = bdptDistantEmitterPosition(sunDir);
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
  // SINGLE-WORKGROUP sequential build — DELIBERATELY RETAINED (A9 #5). This WAS one
  // workgroup per column, each reading column-1 from the buffer — a cross-workgroup
  // data race, since workgroup execution order within one dispatch is spec-undefined
  // (it only "worked" because drivers serialize tiny dispatches; that's why V18
  // passed). Building the whole light subpath in ONE invocation makes each column
  // read the previous vertex THIS thread just wrote → sequentially correct, no race.
  //
  // WHY SERIAL STAYS (A9 #5): the light SUBPATH is an inherently sequential Markov
  // chain — vertex k's position/throughput/pdf depend on vertex k-1's (the BSDF
  // sample is rooted at the previous vertex). A parallel build would need either
  // (a) a barrier-synchronised single-workgroup pass (one thread per column, a
  // workgroupBarrier between columns) — which is STILL serial in the critical path
  // and adds barrier cost for ≤8 columns, OR (b) speculative/wavefront schemes that
  // reintroduce the cross-invocation hazard the V25 fix closed. For a ONE-shared-
  // path-per-frame light subpath (the host issues dispatchWorkgroups(1,1,1)) the
  // serial build over ≤ 8 columns is trivially cheap; the variance reduction comes
  // from accumulating one independent subpath PER FRAME (frameSeed-decorrelated),
  // not from a wide per-frame build. A per-pixel parallel light subpath (the true
  // throughput win) is a SEPARATE, larger redesign (per-pixel light-path buffers +
  // the connection reading the matching pixel's subpath) — tracked, not in A9.
  //
  // A9 raised the OTHER limits instead: maxLightBounces cap 3 → 8 (deeper transport),
  // a REAL glossy/specular BSDF at each light vertex (was Lambertian), and an
  // isotropic point-emitter model. (Caught + designed via the wsl-gpu gate; V25.)
  let maxB = i32(params.bdptMaxLightBounces);
  let seed = params.frameSeed ^ params.frameIndex;

  // Column 0 — the emitter vertex.
  var rng0 = pcgInit(0u, 0u, seed);
  bdptWriteBounce0(0, &rng0);

  // Columns 1..maxB-1 — extend from the previous column (written just above /
  // last loop iteration, by THIS same invocation: no inter-workgroup hazard).
  for (var col = 1; col < maxB; col = col + 1) {
    var rng = pcgInit(u32(col), 0u, seed);
    let prevCol = col - 1;
    let v0prev = bdptLightPath[bdptLightPathIndex(prevCol, 0u)];
    let v1prev = bdptLightPath[bdptLightPathIndex(prevCol, 1u)];
    let v2prev = bdptLightPath[bdptLightPathIndex(prevCol, 2u)];
    // Row 3 of prevCol: .xyz = woAtPrev (outgoing direction at prevPos toward the
    // vertex before it), .w = prevMatId (< 0 for emitter, >= 0 for surface).
    let v3prev = bdptLightPath[bdptLightPathIndex(prevCol, 3u)];
    if (v0prev.w == BDPT_KIND_INVALID) {
      bdptWriteInvalid(col);
      continue;
    }
    let prevPos = v0prev.xyz;
    let prevNormal = v1prev.xyz;
    let prevThroughput = v2prev.xyz;
    let woAtPrev = v3prev.xyz;    // outgoing direction at prevPos (toward its own predecessor)
    let prevMatId = v3prev.w;     // < 0 = emitter vertex, >= 0 = surface vertex
  
    // ── BDPT light-subpath estimator coherence: stored throughput/pdfFwd now
    // describe the traced segment (was a sampled-then-discarded direction),
    // 2026-06-10 — RENDER-CHANGING for bdpt:true, A/B pending V28-B.
    //
    // The fix: sample ONE direction at prevPos from the REAL BSDF (not the old
    // two-step: cosine-hemisphere trace + discard + real-BSDF sample at newPos).
    // That same direction is used to (a) extend the path (trace) and (b) compute
    // the stored throughput update (f·|cos|/pdf of the traced segment) and pdfFwd
    // (the generation density of the traced segment). Following PBRT §16.3,
    // pdfRev(prevCol) is then patched to the sampled-direction pdf (reciprocal
    // BSDF convention: rev density at k = fwd density at k used to reach k+1).
    //
    // For the emitter vertex (prevMatId < 0) the cosine-hemisphere direction
    // already describes the emitter emission profile; we keep it for emitter→1st
    // and apply the BSDF-throughput correction below instead of the old approach.
    var scatterDir = vec3f(0.0);
    var pdfScatter = 0.0;
    var fPrev = vec3f(0.0);
    var cosPrev = 0.0;
  
    if (prevMatId < 0.0) {
      // Emitter vertex: sample cosine hemisphere about prevNormal (= emitter
      // surface or direction normal). The emitter throughput already carries the
      // emitted radiance scaled by the emitter pdf; the cosine hemisphere scatter
      // here represents the direction-distribution component.
      let hemi = cosineHemisphereSample(&rng, prevNormal);
      scatterDir = hemi.wi;
      pdfScatter = hemi.pdf;
      cosPrev = max(dot(prevNormal, scatterDir), 0.0);
      // PTWG-BDPT-01 (2026-06-15): finite area emitters are sampled by AREA at
      // bounce 0, and this extension samples the outgoing direction in SOLID ANGLE.
      // The rendering-equation estimator therefore needs the usual cos/pdfΩ = π
      // factor after the first surface hit:
      //   Le * A/pPick * cos / (cos/π) = Le * A * π / pPick.
      //
      // Legacy pseudo-emitters (directional/spot/env/point sentinels) already bake
      // their direction-density normalization at bounce 0, so keep their old
      // INV_PI branch to avoid double-applying π there.
      fPrev = select(vec3f(INV_PI), vec3f(1.0), prevMatId == BDPT_LV_AREA_EMITTER_MATID);
    } else {
      // Surface vertex: sample the real BSDF at prevPos (outgoing = woAtPrev,
      // the direction that brought the path to prevPos from its predecessor).
      let prevPayload = bdptLightPath[bdptLightPathIndex(prevCol, 4u)];
      let prevMat = bdptSampleMaterialAtPayload(u32(prevMatId), prevPayload, prevNormal);
      let prevBc = prevMat.baseColor;
      let prevRough = max(prevMat.roughness, 0.02);
      let prevMetal = prevMat.metallic;
      let cosOPrev = max(dot(prevNormal, woAtPrev), 0.0);
      let f0Prev = materialSpecularF0(prevBc, prevMetal, prevMat.specularColor, prevMat.specularIntensity);
      let fresPrev = fresnelSchlick(cosOPrev, f0Prev);
      let bsPrev = sampleNextBounceDirectionWithClearcoatNormal(
        &rng,
        -woAtPrev,
        prevPos,
        prevNormal,
        prevNormal,
        prevMat.clearcoatNormal,
        prevBc,
        prevRough,
        prevMetal,
        0.0,
        prevMat.ior,
        fresPrev,
        vec3f(1.0),
        false,
        prevMat.clearcoat,
        prevMat.clearcoatRoughness,
        prevMat.sheen,
        prevMat.sheenRoughness,
        prevMat.sheenColor,
        prevMat.anisotropy,
        prevMat.anisotropyRotation,
      );
      scatterDir = bsPrev.sampledDir;
      pdfScatter = brdfDirectionalPdfFullSampledWithClearcoatNormal(prevBc, prevRough, prevMetal, 0.0, prevMat.ior,
                                      prevNormal, prevMat.clearcoatNormal, woAtPrev, scatterDir,
                                      prevMat.clearcoat, prevMat.clearcoatRoughness,
                                      prevMat.sheen, prevMat.sheenRoughness,
                                      prevMat.iridescence, prevMat.iridescenceIor,
                                      prevMat.iridescenceThicknessMin, prevMat.iridescenceThicknessMax,
                                      prevMat.specularColor, prevMat.specularIntensity,
                                      prevMat.anisotropy, prevMat.anisotropyRotation);
      cosPrev = max(dot(prevNormal, scatterDir), 0.0);
      fPrev = evaluateBrdfFullWithClearcoatNormal(
        prevBc, prevRough, prevMetal, prevNormal, prevMat.clearcoatNormal, woAtPrev, scatterDir,
        prevMat.clearcoat, prevMat.clearcoatRoughness,
        prevMat.sheen, prevMat.sheenRoughness, prevMat.sheenColor,
        prevMat.iridescence, prevMat.iridescenceIor,
        prevMat.iridescenceThicknessMin, prevMat.iridescenceThicknessMax,
        prevMat.specularColor, prevMat.specularIntensity,
        prevMat.anisotropy, prevMat.anisotropyRotation,
      );
    }
  
    if (pdfScatter <= 1e-8 || cosPrev <= 1e-5) {
      bdptWriteInvalid(col);
      continue;
    }
  
    var ray: Ray;
    ray.origin = prevPos + prevNormal * 1e-4;
    ray.direction = scatterDir;
    let hit = traceClosest(ray, 1e-4, 1e30);
    if (!hit.didHit) {
      bdptWriteInvalid(col);
      continue;
    }
    let matIdx = hitMaterialId(hit);
    let matPayload = vec4f(bitcast<f32>(hit.triIndex), hit.baryVW.x, hit.baryVW.y, bitcast<f32>(hit.instanceIndex));
    let mat = bdptSampleMaterialAtPayload(matIdx, matPayload, safe_normalize(hit.normal));
    // Perfect-specular TRANSMISSION (glass) is non-reconnectable on the light path —
    // the reconnection-shift / Veach connection assumes a non-singular BSDF at the
    // connectable vertex. (A glossy/rough refractive vertex IS handled by the real
    // BSDF sample below; only the near-delta refraction terminates the subpath.)
    if (mat.transmission > 0.5 && mat.roughness < 0.05) {
      bdptWriteInvalid(col);
      continue;
    }
    let newPos = ray.origin + ray.direction * hit.dist;
    let newNormal = safe_normalize(hit.normal);
    // Front-relative shading normal at the new vertex (toward the incoming light dir).
    var nsFront = select(-newNormal, newNormal, dot(newNormal, -scatterDir) > 0.0);
    nsFront = applyNormalMap(matIdx, hit.triIndex, hit.baryVW, nsFront, hit.instanceIndex);
    nsFront = applyBumpMap(matIdx, hit.triIndex, hit.baryVW, nsFront, hit.instanceIndex);
    // Outgoing direction at newPos toward the previous vertex (= -scatterDir).
    let woLp = -scatterDir;
  
    // Throughput update: carry the prefix throughput * f·|cos|/pdf of THIS traced
    // segment. pdfFwd = generation density of scatterDir at prevPos (SA measure,
    // no baked-in geometry term — the §10.3 ConvertDensity handles SA→area).
    let pdfFwd = pdfScatter;
    let newThroughput = prevThroughput * fPrev * cosPrev / pdfFwd;
  
    // pdfRev at col (Lambertian placeholder): the true value requires knowing the
    // NEXT scatter direction (which isn't known until col+1 is built). We store a
    // Lambertian placeholder here and PATCH it when col+1 is built (the PBRT
    // RandomWalk convention: prev.pdfRev = pdf_of_scatter_at_prev). For the
    // deepest vertex (no further extension) the placeholder is used directly;
    // the §10.3 connection overrides pdfRev at L_c and L_{c-1} anyway.
    let pdfRevPlaceholder = max(dot(nsFront, woLp), 0.0) * INV_PI;
  
    // PATCH pdfRev of prevCol (PBRT RandomWalk convention, Item-3 fix 2026-06-10):
    //
    // pdfRev at prevCol = the solid-angle density of generating the REVERSE direction
    // (woAtPrev = direction back toward prevCol's predecessor) given the INCOMING
    // direction at prevCol is scatterDir.
    //
    // For LAMBERTIAN / EMITTER vertices: pdfFwd == pdfRev (cosine hemisphere is
    // symmetric w.r.t. the Lambertian BSDF), so pdfFwd is the correct patch value.
    //
    // For GLOSSY / VNDF vertices: the VNDF pdf is NOT symmetric.  The forward pdf
    // was brdfDirectionalPdfFullSampled(prevNormal, woAtPrev, scatterDir); the reverse pdf is
    // brdfDirectionalPdfFullSampled(prevNormal, scatterDir, woAtPrev) — outgoing
    // and incoming swapped.  Using pdfFwd as pdfRev for VNDF lobes biases the MIS weights but
    // NOT the contribution value (the MIS sum still integrates to an unbiased
    // estimator — incorrect pdfRev inflates or deflates strategy weights without
    // introducing energy).  The PBRT §16.3 analysis bounds the variance penalty to
    // at most the VNDF/Lambertian pdf ratio at the specific angle (~10% typical,
    // up to ~3× at grazing angles on metallic surfaces).
    //
    // We compute the true pdfRev for both cases:
    //   - emitter (prevMatId < 0): pdfRev = pdfFwd (Lambertian symmetric)
    //   - surface (prevMatId >= 0): pdfRev = brdfDirectionalPdfFullSampled(prevNormal, scatterDir, woAtPrev)
    var pdfRevAtPrev = pdfFwd; // correct default for emitter + Lambertian vertices
    if (prevMatId >= 0.0) {
      // Surface vertex: compute the reverse pdf by swapping wo/wi in the BSDF pdf.
      // prevBc/prevRough/prevMetal/prevMat.ior are in scope from the surface branch
      // above (the emitter branch does not reach this code path since prevMatId < 0).
      let prevPayloadForRev = bdptLightPath[bdptLightPathIndex(prevCol, 4u)];
      let prevMatForRev = bdptSampleMaterialAtPayload(u32(prevMatId), prevPayloadForRev, prevNormal);
      let prevBcRev = prevMatForRev.baseColor;
      let prevRoughRev = max(prevMatForRev.roughness, 0.02);
      let prevMetalRev = prevMatForRev.metallic;
      // Reverse: incoming = scatterDir, outgoing (toward prevCol's predecessor) = woAtPrev.
      pdfRevAtPrev = brdfDirectionalPdfFullSampledWithClearcoatNormal(prevBcRev, prevRoughRev, prevMetalRev, 0.0,
                                        prevMatForRev.ior, prevNormal, prevMatForRev.clearcoatNormal, scatterDir, woAtPrev,
                                        prevMatForRev.clearcoat, prevMatForRev.clearcoatRoughness,
                                        prevMatForRev.sheen, prevMatForRev.sheenRoughness,
                                        prevMatForRev.iridescence, prevMatForRev.iridescenceIor,
                                        prevMatForRev.iridescenceThicknessMin, prevMatForRev.iridescenceThicknessMax,
                                        prevMatForRev.specularColor, prevMatForRev.specularIntensity,
                                        prevMatForRev.anisotropy, prevMatForRev.anisotropyRotation);
    }
    let old_r2prev = bdptLightPath[bdptLightPathIndex(prevCol, 2u)];
    bdptLightPath[bdptLightPathIndex(prevCol, 2u)] = vec4f(old_r2prev.xyz, pdfRevAtPrev);
  
    bdptLightPath[bdptLightPathIndex(col, 0u)] = vec4f(newPos, 0.0);
    bdptLightPath[bdptLightPathIndex(col, 1u)] = vec4f(nsFront, pdfFwd);
    bdptLightPath[bdptLightPathIndex(col, 2u)] = vec4f(newThroughput, pdfRevPlaceholder);
    // A9 — record the reached vertex's matId + wo toward the previous light vertex so
    // the §10.3 connection evaluates the REAL light-vertex BSDF (glossy/metallic).
    bdptWriteLvBsdf(col, f32(matIdx), woLp);
    bdptWriteLvMaterialPayload(col, hit.triIndex, hit.baryVW, hit.instanceIndex);
  }
}
`;
