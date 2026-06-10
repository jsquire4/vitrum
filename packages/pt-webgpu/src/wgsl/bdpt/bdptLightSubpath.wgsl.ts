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

// A9 — row-3 light-vertex BSDF record. .x = matId (>=0 ⇒ a real surface vertex
// whose BSDF the connection evaluates; < 0 ⇒ emitter/invalid → Lambertian profile).
// .yzw = wo toward the PREVIOUS light vertex (the eval's outgoing direction).
const BDPT_LV_EMITTER_MATID: f32 = -1.0;
fn bdptWriteLvBsdf(col: i32, matId: f32, woTowardPrev: vec3f) {
  bdptLightPath[bdptLightPathIndex(col, 3u)] = vec4f(woTowardPrev, matId);
}

fn bdptWriteInvalid(col: i32) {
  bdptLightPath[bdptLightPathIndex(col, 0u)] = vec4f(0.0, 0.0, 0.0, BDPT_KIND_INVALID);
  bdptLightPath[bdptLightPathIndex(col, 1u)] = vec4f(0.0);
  bdptLightPath[bdptLightPathIndex(col, 2u)] = vec4f(0.0);
  bdptLightPath[bdptLightPathIndex(col, 3u)] = vec4f(0.0, 0.0, 0.0, BDPT_LV_EMITTER_MATID);
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
  if (v0prev.w == BDPT_KIND_INVALID) {
    bdptWriteInvalid(col);
    continue;
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
  let mat = decodeMaterial(matIdx);
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
  let nsFront = select(-newNormal, newNormal, dot(newNormal, -scatterDir) > 0.0);

  // ── A9 — REAL BSDF light-subpath extension (glossy/specular, not Lambertian) ──
  // Sample the NEXT light-path direction from the visible-vertex BSDF (the same
  // diffuse/glossy partition the eye path + the ReSTIR-PT producer use), so a
  // glossy/metallic light-path vertex carries the correct lobe + pdf. The stored
  // pdfFwd/pdfRev are the REAL brdfDirectionalPdf (SA measure; the §10.3 connection
  // sweep converts SA→area via ConvertDensity, so NO baked-in geometry term here).
  // The throughput is the unbiased BSDF MC estimator f·cos/pdf.
  let woLp = -scatterDir;                        // outgoing toward the previous vertex
  let bc = mat.baseColor;
  let rough = max(mat.roughness, 0.02);
  let metal = mat.metallic;
  var tanT: vec3f;
  var tanB: vec3f;
  buildOnb(nsFront, &tanT, &tanB);
  let cosOlp = max(dot(nsFront, woLp), 0.0);
  let f0lp = mix(vec3f(0.04), bc, metal);
  let freslp = fresnelSchlick(cosOlp, f0lp);
  let specProbRaw = clamp(mix(0.04, 0.96, max(luminance(freslp), metal)), 0.04, 0.96);
  let specProb = specProbRaw;
  var nextDir = vec3f(0.0);
  if (rand_f32(&rng) < specProb) {
    let gs = glossyReflectionSample(&rng, woLp, nsFront, tanT, tanB, rough);
    nextDir = gs.wi;
  } else {
    let cs = cosineHemisphereSample(&rng, nsFront);
    nextDir = cs.wi;
  }
  let cosNext = dot(nsFront, nextDir);
  if (cosNext <= 1e-5) {
    bdptWriteInvalid(col);
    continue;
  }
  // The REAL forward directional pdf (mixture: spec·glossy + diff·cosine) — the
  // density that produced nextDir, used as pdfFwd in the §10.3 MIS sweep.
  let pdfFwd = brdfDirectionalPdf(bc, rough, metal, 0.0, mat.ior, nsFront, woLp, nextDir);
  if (pdfFwd <= 1e-8) {
    bdptWriteInvalid(col);
    continue;
  }
  let fLp = evaluateBrdf(bc, rough, metal, nsFront, woLp, nextDir);
  // Light-path throughput: carry the prefix throughput · f·cos/pdf of THIS bounce.
  let newThroughput = prevThroughput * fLp * cosNext / pdfFwd;
  // pdfRev: the reverse directional density at THIS vertex toward the previous
  // vertex (swap wo↔wi in brdfDirectionalPdf) — the §10.3 sweep needs the true
  // non-symmetric reverse for a glossy lobe (Lambertian was symmetric).
  let toPrev = safe_normalize(prevPos - newPos);
  let pdfRev = brdfDirectionalPdf(bc, rough, metal, 0.0, mat.ior, nsFront, nextDir, toPrev);
  bdptLightPath[bdptLightPathIndex(col, 0u)] = vec4f(newPos, 0.0);
  bdptLightPath[bdptLightPathIndex(col, 1u)] = vec4f(nsFront, pdfFwd);
  bdptLightPath[bdptLightPathIndex(col, 2u)] = vec4f(newThroughput, pdfRev);
  // A9 — record the reached vertex's matId + wo toward the previous light vertex so
  // the §10.3 connection evaluates the REAL light-vertex BSDF (glossy/metallic).
  bdptWriteLvBsdf(col, f32(matIdx), woLp);
  }
}
`;
