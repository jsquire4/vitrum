/**
 * restirPtProducer.wgsl.ts — the ReSTIR-PT reconnection-sample PRODUCER pass.
 *
 * A SEPARATE `@compute` entry point (`restirPtProduce`) — it does NOT modify the
 * megakernel (`kernel.wgsl.ts`). It traces, per full-res pixel:
 *   1. the PRIMARY ray → the visible vertex xv (path prefix; prefix length 1),
 *   2. ONE BSDF-sampled bounce off xv → the reconnection vertex xs,
 *   3. the SUFFIX radiance Lo leaving xs back toward xv,
 * then seeds a 1-sample RIS reservoir, finalises W, and stores it.
 *
 * This is the hero-stack analogue of walkaround-hybrid's risGi producer
 * (`risGi.wgsl.ts` → `updateReservoirGI` + `finaliseGIReservoirW` +
 * `refreshPhase0Cache`); the reservoir ADT + target + finalize are mirrored from
 * `reservoirPtHero.wgsl.ts` (itself a port of `reservoirGi.wgsl.ts`).
 *
 * ════════════════════════════════════════════════════════════════════════════
 * The EXACT Lo definition (energy-consistency — read this)
 * ════════════════════════════════════════════════════════════════════════════
 * Lo is the OUTGOING RADIANCE LEAVING xs back toward xv — i.e. the radiance the
 * suffix path (everything from xs onward) carries along the direction xs → xv,
 * measured with the SUFFIX throughput initialised to 1 at xs (it does NOT carry
 * the prefix throughput, and it does NOT include the reconnection-edge BRDF/cos
 * at xv). Concretely Lo accumulates, with a suffix throughput that starts at 1:
 *   • the EMISSION of xs as seen from xv (xs's own emissive, if any), PLUS
 *   • NEE at xs (direct lighting: the analytic directional/point/env connection
 *     with the visible-vertex-independent BRDF at xs), PLUS
 *   • onward INDIRECT bounces from xs (a short BSDF-sampled walk with RR),
 *     including emissive-on-hit / env on those further vertices.
 *
 * WHY this exact split: the RESOLVE pass reconstructs the full path contribution
 * as  prefixThroughput · f_bsdf(xv; wo→wi) · cos(nv, wi) · Lo · W.
 * If Lo folded in the prefix f·cos (or the prefix throughput), resolve would
 * DOUBLE-apply it. Keeping Lo = "radiance leaving xs, suffix-throughput-1" makes
 * the reconnection split clean and matches Lin 2022's cached L_o at x_s.
 *
 * The producer's 1-sample RIS candidate weight is  w = p̂ / p_src  where
 *   p̂    = restirPtTargetAt(xv, nv, woV, mat, xs, Lo)
 *          = luminance(f_bsdf·cos·Lo), the integrand-matching scalar target, and
 *   p_src = the REAL directional pdf that generated the xv → xs edge.
 * After finalize W = w_sum/p̂ = 1/p_src, and resolve forms f·cos·Lo/p_src — the
 * unbiased single-bounce estimator. p̂ cancels (see reservoirPtHero.wgsl.ts
 * unbiasedness note); storing the REAL p_src is what makes a GLOSSY xv unbiased.
 *
 * ════════════════════════════════════════════════════════════════════════════
 * ENERGY-CONSISTENCY RISKS the producer guards (and the ones it does NOT)
 * ════════════════════════════════════════════════════════════════════════════
 * GUARDED:
 *   • TRANSMISSIVE or singular/delta visible vertex → EMPTY reservoir (M = 0).
 *     The stable shift is finite same-side reflection. Opaque glossy dielectric,
 *     metallic, clearcoat, and sheen mixtures remain reusable; transmission
 *     awaits an opposite-hemisphere shift and measure conversion.
 *   • PRIMARY MISS / degenerate reconnection edge (xv≈xs, cos≤0) → EMPTY.
 *   • p_src ≤ 0 (the sampled direction has zero forward density) → EMPTY.
 *   • RECONNECTION-RAY ESCAPE (the bounce ray leaves the scene) → NOT empty: a FAR
 *     synthetic reconnection vertex along the escape direction with Lo = the
 *     environment radiance, mirroring the proven GI producer (risGi.wgsl.ts:252-
 *     257). Writing empty here would zero xv and the temporal pass would lose the
 *     pixel's entire history (the open-Cornell-face ~22%-of-lit-pixels dropout) —
 *     the env-escape indirect IS part of this pixel's reuse, not solely the
 *     megakernel's job. See the reconnection-trace block below.
 *
 * HONEST REGIME LIMITS:
 *   • Lo SUFFIX TRUNCATION: the suffix walk is bounce-limited + RR-terminated, so
 *     Lo is a finite-bounce estimate of the true outgoing radiance — the SAME
 *     truncation the megakernel applies, but the reconnection-vertex suffix
 *     budget here is independent of the megakernel's bounce budget (the wiring
 *     step may want them aligned for A/B parity against the unidirectional path).
 *   • Lo is a SINGLE stochastic estimate per produced reservoir (one suffix
 *     sample). Across frames the temporal reuse averages many such Lo, but a
 *     fixed Lo per reservoir means the reconnection sample's suffix is NOT
 *     re-estimated under reuse — standard for ReSTIR-PT reconnection (the suffix
 *     is the cached, shift-invariant tail).
 *
 * ── Bind groups ─────────────────────────────────────────────────────────────
 * This pass composes the SHARED pt-webgpu modules (for traceClosest / evaluateBrdf
 * / NEE helpers), which already own @group(0..3). The ReSTIR-PT reservoir output
 * is declared in @group(4) to avoid any collision with the inherited groups; the
 * WIRING step builds a pipeline layout that includes @group(4) and (because the
 * producer statically uses tracing + NEE) groups 0/1/2. (maxBindGroups ≥ 5 — see
 * the compose-module note; the wiring agent may relocate this group if a target
 * adapter caps maxBindGroups at 4.)
 */

export const RESTIR_PT_PRODUCER_WGSL = /* wgsl */ `
// ReSTIR-PT producer output — the per-pixel reconnection reservoir (strided u32).
@group(4) @binding(0) var<storage, read_write> rpt_reservoirOut: array<u32>;
@group(4) @binding(4) var<uniform> rptParams: RestirPtParams;

// Stable scope: finite same-side reflection only. Transmission is deliberately
// excluded until its opposite-hemisphere shift and measure conversion are wired.
// The second predicate is the production BSDF's canonical finite-event classifier;
// opaque diffuse, glossy, metallic, clearcoat, and sheen mixtures all pass.
fn rptIsReusableVisibleVertex(
  roughness: f32, metallic: f32, transmission: f32, clearcoat: f32, sheen: f32,
) -> bool {
  if (transmission > 0.0) { return false; }
  return bsdfHasFiniteConnectionSupport(
    roughness, metallic, transmission, clearcoat, sheen,
  );
}

// Motion-stable surface coordinates for temporal correspondence. Mesh
// barycentrics are invariant under instance transforms and skinning; analytic
// shapes use their authored local frame. The triangle/shape and instance ids
// carried beside this value disambiguate equal coordinates on different objects.
fn rptVisibleSurfaceParam(hit: SceneHit, worldPos: vec3f) -> vec3f {
  if (hit.triIndex < params.triangleCount) {
    return vec3f(hit.baryVW, 0.0);
  }
  let analyticIndex = hit.triIndex - params.triangleCount;
  let base = analyticIndex * 4u;
  if (base + 3u >= arrayLength(&analyticWorldToLocal)) {
    return vec3f(0.0);
  }
  let localPoint = transformPointCols(
    analyticWorldToLocal[base],
    analyticWorldToLocal[base + 1u],
    analyticWorldToLocal[base + 2u],
    analyticWorldToLocal[base + 3u],
    worldPos,
  );
  return rptNormalizeAnalyticSurfaceParam(analyticIndex, localPoint);
}

struct RptAlphaTraceHit {
  hit: SceneHit,
  rayOrigin: vec3f,
  valid: bool,
}

struct RptSuffixEstimate {
  Lo: vec3f,
  valid: bool,
}

struct RptSuffixMaterial {
  baseColor: vec3f,
  roughness: f32,
  metallic: f32,
  emissive: vec3f,
  normal: vec3f,
  clearcoatNormal: vec3f,
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
  envMapIntensity: f32,
  isUnlit: bool,
}

// Trace a suffix segment through stochastic/masked alpha the same way the main
// kernel and visible-vertex producer do. The returned origin is the post-skip ray
// origin, so callers reconstruct hit positions against the ray that actually hit.
fn rptTraceClosestAfterAlpha(rayIn: Ray, rng: ptr<function, PtRngState>) -> RptAlphaTraceHit {
  var ray = rayIn;
  var hit = traceClosest(ray, 1e-4, INFINITY);
  var valid = true;
  // Permit at most eight alpha pass-throughs, then INSPECT the next hit.  A
  // ninth pass-through means this bounded trace did not find the segment's real
  // endpoint; it is invalid rather than an opaque hit at the truncation layer.
  for (var aSkip = 0u; aSkip < 9u; aSkip = aSkip + 1u) {
    if (!hit.didHit) {
      break;
    }
    let passesThrough = alphaTestPassThrough(
      hitMaterialId(hit), hit.triIndex, hit.baryVW, hit.instanceIndex, rng,
    );
    if (!passesThrough) { break; }
    if (aSkip == 8u) {
      valid = false;
      break;
    }
    ray.origin = ray.origin + ray.direction * (hit.dist + 1e-4);
    hit = traceClosest(ray, 1e-4, INFINITY);
  }
  var out: RptAlphaTraceHit;
  out.hit = hit;
  out.rayOrigin = ray.origin;
  out.valid = valid;
  return out;
}

fn rptInvalidSuffixEstimate() -> RptSuffixEstimate {
  var out: RptSuffixEstimate;
  out.Lo = vec3f(0.0);
  out.valid = false;
  return out;
}

// Full-tier suffix material decode for the reconnection vertex and its onward
// hits. This mirrors the main shade prologue's hit-local material stack: maps,
// normal/bump perturbation, layer tint, thin-film reflect tint, spectral albedo,
// and KHR_materials_specular. The visible-vertex source-lobe sampler/PDF lives
// below in rptSampleSourceReconnectionDirection / rptSourceDirectionalPdfFull.
fn rptSuffixMaterialAtHit(hit: SceneHit, incomingDir: vec3f, wo: vec3f, heroLambda: f32) -> RptSuffixMaterial {
  let matId = hitMaterialId(hit);
  let mat = decodeMaterial(matId);
  let isFrontFace = hit.frontFace;
  var out: RptSuffixMaterial;
  out.baseColor = mat.baseColor * sampleVertexColor(hit.triIndex, hit.baryVW).rgb * sampleBaseColorTexture(matId, hit.triIndex, hit.baryVW, hit.instanceIndex).rgb;
  out.baseColor = out.baseColor * sampleAoFactor(matId, hit.triIndex, hit.baryVW, hit.instanceIndex);
  let ormSample = sampleOrmTexture(matId, hit.triIndex, hit.baryVW, hit.instanceIndex);
  out.roughness = clamp(mat.roughness * ormSample.g, 0.0, 1.0);
  out.metallic = clamp(mat.metallic * ormSample.b, 0.0, 1.0);
  out.emissive = mat.emissive * sampleEmissiveTexture(matId, hit.triIndex, hit.baryVW, hit.instanceIndex).rgb;
  out.transmission = clamp(mat.transmission * sampleTransmissionTexture(matId, hit.triIndex, hit.baryVW, hit.instanceIndex), 0.0, 1.0);
  out.ior = mat.ior;
  if (params.spectralEnabled != 0u && mat.dispersionAbbe >= 1.0) {
    out.ior = cauchyIorAtLambda(heroLambda, mat.ior, mat.dispersionAbbe);
  }
  out.normal = select(-hit.normal, hit.normal, isFrontFace);
  out.normal = applyNormalMap(matId, hit.triIndex, hit.baryVW, out.normal, hit.instanceIndex, isFrontFace);
  out.normal = applyBumpMap(matId, hit.triIndex, hit.baryVW, out.normal, hit.instanceIndex);
  out.clearcoatNormal = applyClearcoatNormalMap(matId, hit.triIndex, hit.baryVW, out.normal, hit.instanceIndex);

  out.clearcoat = clamp(mat.clearcoat * sampleClearcoatTexture(matId, hit.triIndex, hit.baryVW, hit.instanceIndex), 0.0, 1.0);
  out.clearcoatRoughness = clamp(mat.clearcoatRoughness * sampleClearcoatRoughnessTexture(matId, hit.triIndex, hit.baryVW, hit.instanceIndex), 0.0, 1.0);
  out.sheen = mat.sheen;
  out.sheenRoughness = clamp(mat.sheenRoughness * sampleSheenRoughnessTexture(matId, hit.triIndex, hit.baryVW, hit.instanceIndex), 0.0, 1.0);
  out.sheenColor = clamp(mat.sheenColor * sampleSheenColorTexture(matId, hit.triIndex, hit.baryVW, hit.instanceIndex), vec3f(0.0), vec3f(1.0));
  out.iridescence = clamp(mat.iridescence * sampleIridescenceTexture(matId, hit.triIndex, hit.baryVW, hit.instanceIndex), 0.0, 1.0);
  let iridescenceThicknessSample = sampleIridescenceThicknessTexture(matId, hit.triIndex, hit.baryVW, hit.instanceIndex);
  out.iridescenceThicknessMin = mat.iridescenceThicknessMin;
  out.iridescenceThicknessMax = mat.iridescenceThicknessMax;
  if (iridescenceThicknessSample >= 0.0) {
    let iridescenceThickness = mix(mat.iridescenceThicknessMin, mat.iridescenceThicknessMax, iridescenceThicknessSample);
    out.iridescenceThicknessMin = iridescenceThickness;
    out.iridescenceThicknessMax = iridescenceThickness;
    if (iridescenceThickness <= 0.0) { out.iridescence = 0.0; }
  }
  out.iridescenceIor = mat.iridescenceIor;
  out.specularColor = clamp(mat.specularColor * sampleSpecularColorTexture(matId, hit.triIndex, hit.baryVW, hit.instanceIndex), vec3f(0.0), vec3f(1.0));
  out.specularIntensity = clamp(mat.specularIntensity * sampleSpecularIntensityTexture(matId, hit.triIndex, hit.baryVW, hit.instanceIndex), 0.0, 1.0);
  if (params.spectralEnabled != 0u) {
    out.sheenColor = vec3f(spectralRgbFactorAtHero(out.sheenColor, heroLambda));
    out.specularColor = vec3f(spectralRgbFactorAtHero(out.specularColor, heroLambda));
  }
  out.anisotropy = materialAnisotropy(matId, hit.triIndex, hit.baryVW, hit.instanceIndex);
  out.anisotropyRotation = materialAnisotropyRotation(matId, hit.triIndex, hit.baryVW, out.normal, hit.instanceIndex);
  out.envMapIntensity = materialEnvMapIntensity(matId);
  out.isUnlit = mat.isUnlit;

  let layerTx = clamp(select(mat.backLayerTx, mat.frontLayerTx, isFrontFace), vec3f(0.0), vec3f(1.0));
  let layerRoughness = select(mat.backLayerRoughness, mat.frontLayerRoughness, isFrontFace);
  if (layerRoughness >= 0.0) {
    out.roughness = clamp(layerRoughness, 0.0, 1.0);
  }
  let layerWeight = select(
    layerTx,
    activeLayerWeightRgb(layerTx, heroLambda, true),
    params.spectralEnabled != 0u && luminance(layerTx) < 0.999,
  );
  out.baseColor = out.baseColor * layerWeight;
  if (params.spectralEnabled != 0u) {
    let reflScalar = spectralCombinedReflectanceAtHero(
      out.baseColor,
      mat.baseColor,
      mat.spectralReflCoeffs,
      mat.hasSpectralReflectance,
      heroLambda,
    );
    out.baseColor = vec3f(reflScalar);
  }
  return out;
}

fn rptSampleDirectionalCone(rng: ptr<function, PtRngState>, axisIn: vec3f, angularDiameter: f32) -> vec3f {
  var sampleDir = safe_normalize(axisIn);
  if (angularDiameter > 0.0) {
    let cosHalfAngle = cos(angularDiameter * 0.5);
    let xi1 = rand_f32(rng);
    let xi2 = rand_f32(rng);
    let cosTheta = mix(cosHalfAngle, 1.0, xi1);
    let sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
    let phi = 6.28318530718 * xi2;
    let tangentX = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), abs(sampleDir.x) > 0.9);
    let basisY = normalize(cross(sampleDir, tangentX));
    let basisX = cross(basisY, sampleDir);
    sampleDir = normalize(sinTheta * cos(phi) * basisX + sinTheta * sin(phi) * basisY + cosTheta * sampleDir);
  }
  return sampleDir;
}

// Direct-lighting NEE at the RECONNECTION vertex xs (visible-vertex independent).
// Adds the rect-/disc-/mesh-area + directional + point + spot + environment
// connections with the SAME analytic estimators the megakernel uses, at the suffix throughput
// passed in (the suffix starts at throughput 1 at xs). The rect-area branch is
// ENERGY-CRITICAL (it is the engine's primary area-light NEE — see its inline
// note); the delta + env branches are full-weight, the area branch is the
// un-MIS-weighted area-measure connection (the producer has no BSDF-side
// emissive-on-hit to complement an MIS split — see the rect-area branch). Uses a
// per-light loop (no light-tree selection) to keep the producer's group footprint
// minimal and the Lo estimate well-defined. Returns the radiance leaving xs.
fn rptDirectAtVertex(
  rng: ptr<function, PtRngState>,
  pos: vec3f,
  normal: vec3f,
  clearcoatNormal: vec3f,
  wo: vec3f,
  baseColor: vec3f,
  roughness: f32,
  metallic: f32,
  clearcoat: f32,
  transmission: f32,
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
  suffixThroughput: vec3f,
  heroLambda: f32,
  envMapIntensity: f32,
) -> vec3f {
  var contrib = vec3f(0.0);
  // Directional lights: full-weight light-sampled estimator, mirroring the
  // megakernel's delta/soft-cone branch. Use the packed N-directional RGB
  // records instead of the legacy scalar lightDir mirror so ReSTIR-PT suffix Lo
  // keeps chroma and multiple directional emitters.
  for (var di = 0u; di < params.directionalLightCount; di = di + 1u) {
    let dBase = di * 2u;
    let dDirAD = directionalLights[dBase];
    let dIrrMean = directionalLights[dBase + 1u];
    if (dIrrMean.w > 0.0) {
      let angDiamRaw = dDirAD.w;
      let dirShadowDisabled = angDiamRaw < 0.0;
      let angDiam = select(angDiamRaw, -1.0 - angDiamRaw, dirShadowDisabled);
      let lightDir = rptSampleDirectionalCone(rng, dDirAD.xyz, angDiam);
      let nDotL = max(0.0, dot(normal, lightDir));
      if (nDotL > 0.0) {
        let shadowRay = Ray(pos + normal * 1e-3, lightDir);
        if (dirShadowDisabled || !traceAny(shadowRay, 1e-4, INFINITY)) {
          let brdf = evaluateFiniteSameSideBrdfFullWithClearcoatNormal(
            baseColor, roughness, metallic, transmission, normal, clearcoatNormal, wo, lightDir,
            clearcoat, clearcoatRoughness, sheen, sheenRoughness, sheenColor,
            iridescence, iridescenceIor, iridescenceThicknessMin, iridescenceThicknessMax,
            specularColor, specularIntensity,
            anisotropy, anisotropyRotation,
          );
          let dIrrOut = select(dIrrMean.rgb, spectralEmissionAtHero(dIrrMean.rgb, heroLambda), params.spectralEnabled != 0u);
          contrib = contrib + suffixThroughput * brdf * nDotL * dIrrOut;
        }
      }
    }
  }
  // Rect/disc area lights: the dominant area-light NEE the suffix Lo needs.
  // Shape discriminator in emission.w: ≈ 0 → rect, ≈ 1 → analytic disc.
  // FULL WEIGHT (no MIS) — see inline comment in kernel.wgsl.ts rect loop for rationale.
  // Native analytic disc emitters replace the 32-triangle fan, 2026-06-10 —
  // RENDER-CHANGING for disc-lit scenes, A/B in R9-B.
  for (var ri = 0u; ri < params.rectAreaLightCount; ri = ri + 1u) {
    let rb = ri * 4u;
    let rpos = rectAreaLights[rb].xyz;
    let rectShadowDisabled = rectAreaLights[rb].w > 0.5;
    let ru = rectAreaLights[rb + 1u].xyz;
    let rv = rectAreaLights[rb + 2u].xyz;
    let rshapeR = rectAreaLights[rb + 3u];
    let rr = rshapeR.rgb;
    let isDiscR = abs(rshapeR.w - 1.0) < 0.5;
    let xi1r = rand_f32(rng);
    let xi2r = rand_f32(rng);
    var lpos: vec3f;
    var area: f32;
    if (isDiscR) {
      let rrad = length(ru);
      let disc = concentricDiscSample(
        vec2f(xi1r * 2.0 - 1.0, xi2r * 2.0 - 1.0),
      );
      lpos = rpos + ru * disc.x + rv * disc.y;
      area = PI * rrad * rrad;
    } else {
      lpos = rpos + ru * (xi1r * 2.0 - 1.0) + rv * (xi2r * 2.0 - 1.0);
      area = 4.0 * length(cross(ru, rv));
    }
    let toLight = lpos - pos;
    let dist2 = dot(toLight, toLight);
    if (dist2 <= 0.0 || area <= 0.0) { continue; }
    let dist = sqrt(dist2);
    let wi = toLight / dist;
    let nDotL = max(dot(normal, wi), 0.0);
    if (nDotL > 0.0) {
      let lightNormal = safe_normalize(cross(ru, rv));
      let cosLight = max(dot(lightNormal, -wi), 0.0);
      if (cosLight > 0.0) {
        let lightPdf = dist2 / (cosLight * area);
        let shadowRay = Ray(pos + normal * 1e-3, wi);
        if (rectShadowDisabled || !traceAny(shadowRay, 1e-4, max(dist - 2e-3, 1e-3))) {
          let brdf = evaluateFiniteSameSideBrdfFullWithClearcoatNormal(
            baseColor, roughness, metallic, transmission, normal, clearcoatNormal, wo, wi,
            clearcoat, clearcoatRoughness, sheen, sheenRoughness, sheenColor,
            iridescence, iridescenceIor, iridescenceThicknessMin, iridescenceThicknessMax,
            specularColor, specularIntensity,
            anisotropy, anisotropyRotation,
          );
          let rrOut = select(rr, spectralEmissionAtHero(rr, heroLambda), params.spectralEnabled != 0u);
          contrib = contrib + suffixThroughput * brdf * nDotL * rrOut / lightPdf;
        }
      }
    }
  }
  // Point lights (delta): full weight, no MIS.
  // H51-D: stride 3 (3 vec4 = 12 f32): position, radiance, [distance, decay, 0, 0]
  for (var pi = 0u; pi < params.pointLightCount; pi = pi + 1u) {
    let base = pi * 3u;
    let lp = pointLights[base].xyz;
    let rad = pointLights[base + 1u].rgb;
    let ptExtra = pointLights[base + 2u];
    let ptMaxDist = ptExtra.x;
    let ptDecay   = ptExtra.y;
    let ptShadowDisabled = ptExtra.z > 0.5;
    let toPoint = lp - pos;
    let dist2 = max(dot(toPoint, toPoint), 1e-5);
    let dist = sqrt(dist2);
    if (ptMaxDist > 0.0 && dist > ptMaxDist) { continue; }
    let wi = toPoint / dist;
    let nDotL = max(0.0, dot(normal, wi));
    if (nDotL > 0.0) {
      let shadowRay = Ray(pos + normal * 1e-3, wi);
      if (ptShadowDisabled || !traceAny(shadowRay, 1e-4, max(dist - 2e-3, 1e-3))) {
        let attenuation = pointSpotDistanceAttenuation(dist, ptMaxDist, ptDecay);
        let brdf = evaluateFiniteSameSideBrdfFullWithClearcoatNormal(
          baseColor, roughness, metallic, transmission, normal, clearcoatNormal, wo, wi,
          clearcoat, clearcoatRoughness, sheen, sheenRoughness, sheenColor,
          iridescence, iridescenceIor, iridescenceThicknessMin, iridescenceThicknessMax,
          specularColor, specularIntensity,
          anisotropy, anisotropyRotation,
        );
        let radOut = select(rad, spectralEmissionAtHero(rad, heroLambda), params.spectralEnabled != 0u);
        contrib = contrib + suffixThroughput * brdf * nDotL * radOut * attenuation;
      }
    }
  }
  // Spot lights (delta): full weight, no MIS.
  // H14-B + H51-D: mirrors kernel NEE; stride 4 (4 vec4 = 16 f32).
  for (var si = 0u; si < params.spotLightCount; si = si + 1u) {
    let sb = si * 4u;
    let spos = spotLights[sb].xyz;
    let saxis = spotLights[sb + 1u];
    let sradW = spotLights[sb + 2u];
    let spExtra = spotLights[sb + 3u];
    let spotDir = safe_normalize(saxis.xyz);
    let cosOuter = saxis.w;
    let cosInner = sradW.w;
    let srad = sradW.rgb;
    let spMaxDist = spExtra.x;
    let spDecay   = spExtra.y;
    let spShadowDisabled = spExtra.z > 0.5;
    let toSpot = spos - pos;
    let dist2 = max(dot(toSpot, toSpot), 1e-5);
    let dist = sqrt(dist2);
    if (spMaxDist > 0.0 && dist > spMaxDist) { continue; }
    let wi = toSpot / dist;
    let coneCos = dot(-wi, spotDir);
    if (coneCos >= cosOuter) {
      let nDotL = max(0.0, dot(normal, wi));
      if (nDotL > 0.0) {
        let shadowRay = Ray(pos + normal * 1e-3, wi);
        if (spShadowDisabled || !traceAny(shadowRay, 1e-4, max(dist - 2e-3, 1e-3))) {
          let softness = smoothstep(cosOuter, max(cosInner, cosOuter + 1e-6), coneCos);
          let attenuation = pointSpotDistanceAttenuation(dist, spMaxDist, spDecay);
          let brdf = evaluateFiniteSameSideBrdfFullWithClearcoatNormal(
            baseColor, roughness, metallic, transmission, normal, clearcoatNormal, wo, wi,
            clearcoat, clearcoatRoughness, sheen, sheenRoughness, sheenColor,
            iridescence, iridescenceIor, iridescenceThicknessMin, iridescenceThicknessMax,
            specularColor, specularIntensity,
            anisotropy, anisotropyRotation,
          );
          let sradOut = select(srad, spectralEmissionAtHero(srad, heroLambda), params.spectralEnabled != 0u);
          contrib = contrib + suffixThroughput * brdf * nDotL * softness * sradOut * attenuation;
        }
      }
    }
  }
  // Mesh-area lights: full-weight area-measure connection for implicit and
  // explicit mesh emitters. This mirrors the main kernel's triangle sampler
  // without light-selection compensation because this producer loops all lights.
  for (var mi = 0u; mi < params.meshAreaLightCount; mi = mi + 1u) {
    let mb = meshAreaLightBase(mi);
    let a = meshAreaLights[mb].xyz;
    let b = meshAreaLights[mb + 1u].xyz;
    let c = meshAreaLights[mb + 2u].xyz;
    let r1 = rand_f32(rng);
    let r2 = rand_f32(rng);
    let su = sqrt(r1);
    let uu = 1.0 - su;
    let vv = r2 * su;
    let ww = 1.0 - uu - vv;
    let lpos = a * uu + b * vv + c * ww;
    let mr = sampleMeshAreaLightRadiance(
      mi, vec3f(uu, vv, ww), lpos,
    );
    let toLight = lpos - pos;
    let dist2 = dot(toLight, toLight);
    let area = 0.5 * length(cross(b - a, c - a));
    if (dist2 <= 0.0 || area <= 0.0) { continue; }
    let dist = sqrt(dist2);
    let wi = toLight / dist;
    let nDotL = max(dot(normal, wi), 0.0);
    if (nDotL > 0.0) {
      let lightNormal = safe_normalize(cross(b - a, c - a));
      let cosLight = max(dot(lightNormal, -wi), 0.0);
      if (cosLight > 0.0) {
        let lightPdf = dist2 / (cosLight * area);
        let shadowRay = Ray(pos + normal * 1e-3, wi);
        if (meshAreaLights[mb + 3u].w > 0.5 || !traceAny(shadowRay, 1e-4, max(dist - 2e-3, 1e-3))) {
          let brdf = evaluateFiniteSameSideBrdfFullWithClearcoatNormal(
            baseColor, roughness, metallic, transmission, normal, clearcoatNormal, wo, wi,
            clearcoat, clearcoatRoughness, sheen, sheenRoughness, sheenColor,
            iridescence, iridescenceIor, iridescenceThicknessMin, iridescenceThicknessMax,
            specularColor, specularIntensity,
            anisotropy, anisotropyRotation,
          );
          let mrOut = select(mr, spectralEmissionAtHero(mr, heroLambda), params.spectralEnabled != 0u);
          contrib = contrib + suffixThroughput * brdf * nDotL * mrOut / lightPdf;
        }
      }
    }
  }
  // Environment (importance-sampled if a map is present), MIS vs the BRDF pdf.
  if (hasEnvironmentMap() || params.environmentSun.w > 0.0) {
    var envDir = vec3f(0.0, 1.0, 0.0);
    var envColor = vec3f(0.0);
    var envPdf = 0.0;
    let envSample = sampleEnvironmentImportance(rng);
    if (envSample.pdf > 0.0) {
      envDir = envSample.wi;
      envColor = envSample.value;
      envPdf = envSample.pdf;
    } else {
      let diffSample = cosineHemisphereSample(rng, normal);
      envDir = diffSample.wi;
      envColor = sampleEnvironmentColor(envDir);
      envPdf = diffSample.pdf;
    }
    let nDotL = max(dot(normal, envDir), 0.0);
    if (nDotL > 1e-6) {
      let shadowRay = Ray(pos + normal * 1e-3, envDir);
      if (!traceAny(shadowRay, 1e-4, INFINITY)) {
        let brdf = evaluateFiniteSameSideBrdfFullWithClearcoatNormal(
          baseColor, roughness, metallic, transmission, normal, clearcoatNormal, wo, envDir,
          clearcoat, clearcoatRoughness, sheen, sheenRoughness, sheenColor,
          iridescence, iridescenceIor, iridescenceThicknessMin, iridescenceThicknessMax,
          specularColor, specularIntensity,
          anisotropy, anisotropyRotation,
        );
        // The suffix continuation below is sampled from a cosine hemisphere,
        // not the full authored BSDF mixture. MIS must compare the environment
        // proposal against that actual continuation density.
        let continuationPdf = max(dot(normal, envDir), 0.0) * INV_PI;
        // Mirror the megakernel environment estimators: spectral mode evaluates
        // env radiance at the hero wavelength, and the current surface's
        // envMapIntensity scales both the NEE and BSDF-escape halves.
        let envColorOut = select(envColor, spectralEmissionAtHero(envColor, heroLambda), params.spectralEnabled != 0u) * envMapIntensity;
        let misWeight = powerHeuristic(envPdf, continuationPdf);
        contrib = contrib + suffixThroughput * brdf * nDotL * envColorOut * misWeight / max(envPdf, 1e-8);
      }
    }
  }
  return contrib;
}

// The SUFFIX radiance Lo leaving the reconnection vertex xs toward xv. A short
// BSDF-sampled walk ROOTED AT xs with suffix throughput initialised to 1 (the
// prefix throughput / reconnection-edge BRDF at xv are applied at RESOLVE, NOT
// here — see the producer header). Accumulates xs's own emission (as seen from
// xv) + NEE at each suffix vertex + emissive/env on onward hits, exactly as the
// megakernel does for the corresponding suffix segment, but rooted at xs with
// throughput 1. The reconnection vertex's material is passed in DIRECTLY (the
// caller already has the SceneHit at xs) so b==0 needs no re-trace; onward
// bounces decode the new hit's material in the loop.
//   - xsHit / incomingDir: the reconnection hit plus the ray direction that
//     reached it, so material maps and normal/bump perturbation use the real
//     barycentric/instance payload.
//   - reconDir: the direction xs → xv (wo at xs; the outgoing direction Lo is on).
fn rptComputeLoAtReconnection(
  rng: ptr<function, PtRngState>,
  xs: vec3f,
  xsHit: SceneHit,
  incomingDir: vec3f,
  reconDir: vec3f,
  heroLambda: f32,
  suffixBounces: u32,
) -> RptSuffixEstimate {
  var Lo = vec3f(0.0);
  var suffixThroughput = vec3f(1.0);
  var wo = reconDir;          // wo at the current suffix vertex (toward xv at b==0)
  var pos = xs;
  var hit = xsHit;
  var rayDir = incomingDir;
  // xs's emission seen from xv is a directly-viewed term (no prior diffuse bounce
  // MIS-accounts for it), so the first iteration adds it ungated.
  var prevAllowsAreaMis = false;

  for (var b = 0u; b < suffixBounces; b = b + 1u) {
    let sm = rptSuffixMaterialAtHit(hit, rayDir, wo, heroLambda);
    // The suffix estimator is a finite same-side/cosine walk. A transmissive
    // hit requires delta refraction plus medium state, which this bounded
    // reconnection estimator does not store. Reject the ENTIRE candidate:
    // publishing Lo accumulated before this hit would be a biased partial suffix.
    if (sm.transmission > 0.0) { return rptInvalidSuffixEstimate(); }
    let normal = sm.normal;
    let baseColor = sm.baseColor;
    let roughness = sm.roughness;
    let metallic = sm.metallic;
    let emissive = select(sm.emissive, spectralEmissionAtHero(sm.emissive, heroLambda), params.spectralEnabled != 0u);

    if (sm.isUnlit) {
      Lo = Lo + suffixThroughput * baseColor;
      break;
    }

    // Emission of this suffix vertex along wo (toward xv at b==0). Gated so an
    // onward diffuse bounce's NEE does not double-count the next hit's emission.
    if (!prevAllowsAreaMis) {
      Lo = Lo + suffixThroughput * emissive;
    }
    // Direct lighting (NEE) at this suffix vertex.
    Lo = Lo + rptDirectAtVertex(
      rng, pos, normal, sm.clearcoatNormal, wo, baseColor, roughness, metallic, sm.transmission,
      sm.clearcoat, sm.clearcoatRoughness, sm.sheen, sm.sheenRoughness, sm.sheenColor,
      sm.iridescence, sm.iridescenceIor, sm.iridescenceThicknessMin, sm.iridescenceThicknessMax,
      sm.specularColor, sm.specularIntensity,
      sm.anisotropy, sm.anisotropyRotation,
      suffixThroughput,
      heroLambda,
      sm.envMapIntensity,
    );

    // Sample the next onward direction with the cosine (diffuse) lobe — the robust
    // default that keeps Lo well-defined for any onward surface; the reconnection
    // vertex was validated diffuse-ish by the reusable-visible-vertex gate, and the
    // onward indirect is a second-order term whose exact lobe choice changes
    // variance, not the mean.
    let cosSample = cosineHemisphereSample(rng, normal);
    let nextDir = cosSample.wi;
    let nDotNext = max(dot(normal, nextDir), 0.0);
    if (nDotNext <= 1e-5) { break; }
    // Onward-bounce throughput = the EXACT cosine-sampling MC estimator
    //   f·cos / pdf   with pdf = cos·INV_PI   (cosSample.pdf),
    // evaluated with the FULL BRDF (evaluateBrdf, whose diffuse kd uses the
    // HALF-VECTOR Fresnel). This is critical at GRAZING wo: the prior throughput
    // "(1 - fresnelSchlick(dot(n,wo)))*baseColor" applied the VIEW-ANGLE Fresnel,
    // which -> 1 as wo grazes, collapsing the diffuse transport toward 0 (a ~10%/
    // bounce energy loss that COMPOUNDED with suffix depth — the ReSTIR-PT reuse
    // ~15% deficit; the deeper a suffix bounce, the more oblique its wo). The
    // physically-correct Lambertian cosine-sample throughput has NO view-angle
    // collapse (f·cos/pdf = albedo for an ideal diffuse lobe). Verified exact
    // (ratio 1.000 ∀ wo angle) vs dense-quadrature in wsl-gpu/scripts/
    // restir-pt-onward-jsmodel.ts. evaluateBrdf also folds the small onward
    // specular response in, matching the megakernel's onward transport.
    let fOnward = evaluateFiniteSameSideBrdfFullWithClearcoatNormal(
      baseColor, roughness, metallic, sm.transmission, normal, sm.clearcoatNormal, wo, nextDir,
      sm.clearcoat, sm.clearcoatRoughness, sm.sheen, sm.sheenRoughness, sm.sheenColor,
      sm.iridescence, sm.iridescenceIor, sm.iridescenceThicknessMin, sm.iridescenceThicknessMax,
      sm.specularColor, sm.specularIntensity,
      sm.anisotropy, sm.anisotropyRotation,
    );
    suffixThroughput = suffixThroughput * fOnward * nDotNext / max(cosSample.pdf, 1e-8);
    prevAllowsAreaMis = true; // diffuse onward bounce: next emission handled by NEE/MIS.

    let nextTrace = rptTraceClosestAfterAlpha(Ray(pos + normal * 1e-3, nextDir), rng);
    if (!nextTrace.valid) { return rptInvalidSuffixEstimate(); }
    let nextHit = nextTrace.hit;
    if (!nextHit.didHit) {
      let envRgb = sampleEnvironmentColor(nextDir);
      let envContribution = select(envRgb, spectralEmissionAtHero(envRgb, heroLambda), params.spectralEnabled != 0u);
      let envNeePdf = environmentNeeProposalPdf(nextDir, normal);
      let escapeMisWeight = powerHeuristic(cosSample.pdf, envNeePdf);
      Lo = Lo + suffixThroughput * envContribution * sm.envMapIntensity * escapeMisWeight;
      break;
    }
    pos = nextTrace.rayOrigin + nextDir * nextHit.dist;
    hit = nextHit;
    rayDir = nextDir;
    wo = -nextDir;

    // Russian roulette on the suffix throughput.
    if (b >= 1u) {
      let surv = clamp(max(suffixThroughput.r, max(suffixThroughput.g, suffixThroughput.b)), 0.05, 0.95);
      if (rand_f32(rng) > surv) { break; }
      suffixThroughput = suffixThroughput / surv;
    }
  }
  var out: RptSuffixEstimate;
  out.Lo = Lo;
  out.valid = rptFiniteVec3(Lo);
  return out;
}

fn rptSourceLobeWeightSum(clearcoat: f32, sheen: f32) -> f32 {
  return max(1.0 + max(clearcoat, 0.0) + max(sheen, 0.0), 1e-4);
}

fn rptSampleSourceReconnectionDirection(
  rng: ptr<function, PtRngState>,
  wo: vec3f,
  normal: vec3f,
  clearcoatNormal: vec3f,
  tanT: vec3f,
  tanB: vec3f,
  roughness: f32,
  metallic: f32,
  fresnel: vec3f,
  clearcoat: f32,
  clearcoatRoughness: f32,
  sheen: f32,
  sheenRoughness: f32,
  anisotropy: f32,
) -> vec3f {
  let lobeWeightSum = rptSourceLobeWeightSum(clearcoat, sheen);
  let xiSource = rand_f32(rng) * lobeWeightSum;
  if (xiSource < 1.0) {
    let baseSpecProb = clamp(mix(0.04, 0.96, max(luminance(fresnel), metallic)), 0.04, 0.96);
    let baseDiffProb = max(0.0, 1.0 - baseSpecProb);
    let sumProb = max(baseSpecProb + baseDiffProb, 1e-4);
    let specProb = baseSpecProb / sumProb;
    if (rand_f32(rng) < specProb) {
      var bs: BsdfSample;
      if (anisotropy > 0.0) {
        bs = glossyReflectionSampleAnisotropic(rng, wo, normal, tanT, tanB, roughness, anisotropy);
      } else {
        bs = glossyReflectionSample(rng, wo, normal, tanT, tanB, roughness);
      }
      return bs.wi;
    }
    let bs = cosineHemisphereSample(rng, normal);
    return bs.wi;
  }
  if (xiSource < 1.0 + max(clearcoat, 0.0)) {
    var ccTanT: vec3f;
    var ccTanB: vec3f;
    buildOnb(clearcoatNormal, &ccTanT, &ccTanB);
    let bs = glossyReflectionSample(rng, wo, clearcoatNormal, ccTanT, ccTanB, clearcoatRoughness);
    return bs.wi;
  }
  let bs = charlieSheenSample(rng, wo, normal, tanT, tanB, sheenRoughness);
  return bs.wi;
}

fn rptSourceDirectionalPdfFull(
  baseColor: vec3f,
  roughness: f32,
  metallic: f32,
  transmission: f32,
  ior: f32,
  normal: vec3f,
  clearcoatNormal: vec3f,
  wo: vec3f,
  wi: vec3f,
  clearcoat: f32,
  clearcoatRoughness: f32,
  sheen: f32,
  sheenRoughness: f32,
  iridescence: f32,
  iridescenceIor: f32,
  iridescenceThicknessMin: f32,
  iridescenceThicknessMax: f32,
  specularColor: vec3f,
  specularIntensity: f32,
  anisotropy: f32,
  anisotropyRotation: f32,
) -> f32 {
  return brdfDirectionalPdfFullSampledWithClearcoatNormal(
    baseColor, roughness, metallic, transmission, ior, normal, clearcoatNormal, wo, wi,
    clearcoat, clearcoatRoughness, sheen, sheenRoughness,
    iridescence, iridescenceIor, iridescenceThicknessMin, iridescenceThicknessMax,
    specularColor, specularIntensity,
    anisotropy, anisotropyRotation,
  );
}

@compute @workgroup_size(8, 8, 1)
fn restirPtProduce(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= params.width || gid.y >= params.height) {
    return;
  }
  let pixelIdx = gid.y * params.width + gid.x;

  var rng = pcgInit(gid.x, gid.y, ptRngFrameKey(params.frameSeed, params.frameIndex));
  let jitter = vec2f(rand_f32(&rng), rand_f32(&rng));
  var primaryRay = generatePrimaryRay(gid.x, gid.y, jitter);
  var heroLambda = params.heroLambdaNm;
  if (params.spectralEnabled != 0u) {
    let hero = sampleHeroWavelengthMIS(rand_f32(&rng), rand_f32(&rng));
    heroLambda = hero.x;
  }

  // ── 1. Primary ray → visible vertex xv (the path prefix) ──
  let vTrace = rptTraceClosestAfterAlpha(primaryRay, &rng);
  let vHit = vTrace.hit;
  if (!vTrace.valid || !vHit.didHit) {
    storeReservoirPTHero_rw(&rpt_reservoirOut, pixelIdx, emptyReservoirPTHero());
    return;
  }
  primaryRay.origin = vTrace.rayOrigin;
  let vMatId = hitMaterialId(vHit);
  let vMat = decodeMaterial(vMatId);
  let xv = primaryRay.origin + primaryRay.direction * vHit.dist;
  let vIsFront = dot(vHit.normal, primaryRay.direction) < 0.0;
  var nv = select(-vHit.normal, vHit.normal, vIsFront);
  nv = applyNormalMap(vMatId, vHit.triIndex, vHit.baryVW, nv, vHit.instanceIndex, vIsFront);
  nv = applyBumpMap(vMatId, vHit.triIndex, vHit.baryVW, nv, vHit.instanceIndex);
  let clearcoatNormalV = applyClearcoatNormalMap(vMatId, vHit.triIndex, vHit.baryVW, nv, vHit.instanceIndex);
  let woV = -primaryRay.direction; // eye-side direction at xv
  var baseColorV = vMat.baseColor * sampleVertexColor(vHit.triIndex, vHit.baryVW).rgb * sampleBaseColorTexture(vMatId, vHit.triIndex, vHit.baryVW, vHit.instanceIndex).rgb;
  baseColorV = baseColorV * sampleAoFactor(vMatId, vHit.triIndex, vHit.baryVW, vHit.instanceIndex);
  let ormSampleV = sampleOrmTexture(vMatId, vHit.triIndex, vHit.baryVW, vHit.instanceIndex);
  var roughnessV = clamp(vMat.roughness * ormSampleV.g, 0.0, 1.0);
  var metallicV = clamp(vMat.metallic * ormSampleV.b, 0.0, 1.0);
  var transmissionV = clamp(vMat.transmission * sampleTransmissionTexture(vMatId, vHit.triIndex, vHit.baryVW, vHit.instanceIndex), 0.0, 1.0);
  var clearcoatV = clamp(vMat.clearcoat * sampleClearcoatTexture(vMatId, vHit.triIndex, vHit.baryVW, vHit.instanceIndex), 0.0, 1.0);
  var clearcoatRoughnessV = clamp(vMat.clearcoatRoughness * sampleClearcoatRoughnessTexture(vMatId, vHit.triIndex, vHit.baryVW, vHit.instanceIndex), 0.0, 1.0);
  var sheenV = vMat.sheen;
  var sheenRoughnessV = clamp(vMat.sheenRoughness * sampleSheenRoughnessTexture(vMatId, vHit.triIndex, vHit.baryVW, vHit.instanceIndex), 0.0, 1.0);
  var sheenColorV = clamp(vMat.sheenColor * sampleSheenColorTexture(vMatId, vHit.triIndex, vHit.baryVW, vHit.instanceIndex), vec3f(0.0), vec3f(1.0));
  var iridescenceV = clamp(vMat.iridescence * sampleIridescenceTexture(vMatId, vHit.triIndex, vHit.baryVW, vHit.instanceIndex), 0.0, 1.0);
  let iridescenceThicknessSampleV = sampleIridescenceThicknessTexture(vMatId, vHit.triIndex, vHit.baryVW, vHit.instanceIndex);
  var iridescenceThicknessMinV = vMat.iridescenceThicknessMin;
  var iridescenceThicknessMaxV = vMat.iridescenceThicknessMax;
  if (iridescenceThicknessSampleV >= 0.0) {
    let iridescenceThicknessV = mix(vMat.iridescenceThicknessMin, vMat.iridescenceThicknessMax, iridescenceThicknessSampleV);
    iridescenceThicknessMinV = iridescenceThicknessV;
    iridescenceThicknessMaxV = iridescenceThicknessV;
    if (iridescenceThicknessV <= 0.0) { iridescenceV = 0.0; }
  }
  let iridescenceIorV = vMat.iridescenceIor;
  var specularColorV = clamp(vMat.specularColor * sampleSpecularColorTexture(vMatId, vHit.triIndex, vHit.baryVW, vHit.instanceIndex), vec3f(0.0), vec3f(1.0));
  var specularIntensityV = clamp(vMat.specularIntensity * sampleSpecularIntensityTexture(vMatId, vHit.triIndex, vHit.baryVW, vHit.instanceIndex), 0.0, 1.0);
  if (params.spectralEnabled != 0u) {
    sheenColorV = vec3f(spectralRgbFactorAtHero(sheenColorV, heroLambda));
    specularColorV = vec3f(spectralRgbFactorAtHero(specularColorV, heroLambda));
  }
  let anisotropyV = materialAnisotropy(vMatId, vHit.triIndex, vHit.baryVW, vHit.instanceIndex);
  let anisotropyRotationV = materialAnisotropyRotation(vMatId, vHit.triIndex, vHit.baryVW, nv, vHit.instanceIndex);
  var iorV = vMat.ior;
  if (params.spectralEnabled != 0u && vMat.dispersionAbbe >= 1.0) {
    iorV = cauchyIorAtLambda(heroLambda, vMat.ior, vMat.dispersionAbbe);
  }
  let layerTxV = clamp(select(vMat.backLayerTx, vMat.frontLayerTx, vIsFront), vec3f(0.0), vec3f(1.0));
  let layerRoughnessV = select(vMat.backLayerRoughness, vMat.frontLayerRoughness, vIsFront);
  if (layerRoughnessV >= 0.0) {
    roughnessV = clamp(layerRoughnessV, 0.0, 1.0);
  }
  let layerWeightV = select(
    layerTxV,
    activeLayerWeightRgb(layerTxV, heroLambda, true),
    params.spectralEnabled != 0u && luminance(layerTxV) < 0.999,
  );
  baseColorV = baseColorV * layerWeightV;
  if (params.spectralEnabled != 0u) {
    let reflScalarV = spectralCombinedReflectanceAtHero(
      baseColorV,
      vMat.baseColor,
      vMat.spectralReflCoeffs,
      vMat.hasSpectralReflectance,
      heroLambda,
    );
    baseColorV = vec3f(reflScalarV);
  }

  // Unlit, transmissive, or singular/delta visible vertex → empty. A TMM
  // thin-film stack is an exact discrete R/T/A interface in the megakernel;
  // it cannot be represented by this finite-direction reconnection reservoir.
  if (vMat.isUnlit || vMat.thinFilmEnabled ||
      !rptIsReusableVisibleVertex(roughnessV, metallicV, transmissionV, clearcoatV, sheenV)) {
    storeReservoirPTHero_rw(&rpt_reservoirOut, pixelIdx, emptyReservoirPTHero());
    return;
  }

  // ── 2. One BSDF-sampled bounce off xv → reconnection vertex xs ──
  // Sample the reconnection direction wi from the visible-vertex BSDF.
  // The hero stack stores the REAL source pdf so a glossy xv stays unbiased.
  var tanT: vec3f;
  var tanB: vec3f;
  buildOnb(nv, &tanT, &tanB);
  if (anisotropyV > 0.0) {
    let c = cos(anisotropyRotationV);
    let s = sin(anisotropyRotationV);
    let rotatedT = c * tanT + s * tanB;
    let rotatedB = -s * tanT + c * tanB;
    tanT = rotatedT;
    tanB = rotatedB;
  }
  let cosO = max(dot(nv, woV), 0.0);
  let f0BaseV = materialSpecularF0(baseColorV, metallicV, specularColorV, specularIntensityV);
  let f0V = iridescenceModifiedF0(
    f0BaseV,
    iridescenceV,
    iridescenceIorV,
    iridescenceThicknessMinV,
    iridescenceThicknessMaxV,
    cosO,
  );
  let fresV = fresnelSchlick(cosO, f0V);
  // Sample from the normalized source-lobe mixture used by pdfSrc below:
  //   p_src = (p_base + clearcoat*p_clearcoat + sheen*p_sheen) /
  //           (1 + clearcoat + sheen)
  // where p_base preserves the base specular/diffuse split, p_clearcoat is a
  // clearcoat-roughness VNDF reflection, and p_sheen is the Charlie half-vector
  // sheen sampler used by brdfDirectionalPdfFull. This keeps the reservoir source
  // density honest.
  let wiRecon = rptSampleSourceReconnectionDirection(
    &rng,
    woV,
    nv,
    clearcoatNormalV,
    tanT,
    tanB,
    roughnessV,
    metallicV,
    fresV,
    clearcoatV,
    clearcoatRoughnessV,
    sheenV,
    sheenRoughnessV,
    anisotropyV,
  );
  let nDotRecon = dot(nv, wiRecon);
  if (nDotRecon <= 1e-5) {
    storeReservoirPTHero_rw(&rpt_reservoirOut, pixelIdx, emptyReservoirPTHero());
    return;
  }
  // The REAL source directional pdf that generated wi (unbiasedness-
  // critical — resolve forms f·cos·Lo/p_src; see the header). A degenerate
  // single sample (p_src ≈ 0, e.g. a near-mirror VNDF lobe sampled back-facing)
  // writes empty: a cosine "retry" is NOT taken because it would make p_src
  // conditional on the glossy sample having failed (a direction-dependent density
  // the mixture brdfDirectionalPdf does not capture) — that mismatch would BIAS
  // the f·cos·Lo/p_src estimator. p_src ≤ 0 is rare (and its f·cos contribution is
  // ~0 anyway), so dropping the single frame's sample is the correct, unbiased
  // choice; the temporal history is re-seeded the next non-degenerate frame.
  let pdfSrc = rptSourceDirectionalPdfFull(
    baseColorV, roughnessV, metallicV, 0.0, iorV, nv, clearcoatNormalV, woV, wiRecon,
    clearcoatV, clearcoatRoughnessV, sheenV, sheenRoughnessV,
    iridescenceV, iridescenceIorV, iridescenceThicknessMinV, iridescenceThicknessMaxV,
    specularColorV, specularIntensityV,
    anisotropyV, anisotropyRotationV,
  );
  if (!rptFinitePositive(pdfSrc) || pdfSrc <= 1e-8) {
    storeReservoirPTHero_rw(&rpt_reservoirOut, pixelIdx, emptyReservoirPTHero());
    return;
  }

  let reconRay = Ray(xv + nv * 1e-3, wiRecon);
  let sTrace = rptTraceClosestAfterAlpha(reconRay, &rng);
  let sHit = sTrace.hit;
  if (!sTrace.valid) {
    storeReservoirPTHero_rw(&rpt_reservoirOut, pixelIdx, emptyReservoirPTHero());
    return;
  }
  var xs: vec3f;
  var ns: vec3f;
  var Lo: vec3f;
  if (!sHit.didHit) {
    // The megakernel already owns E0's environment NEE and complementary
    // BSDF-environment connection. Publishing an environment reconnection here
    // would estimate the same direct path a second time. An empty producer
    // reservoir makes the composite kernel keep tracing the ordinary full path.
    storeReservoirPTHero_rw(&rpt_reservoirOut, pixelIdx, emptyReservoirPTHero());
    return;
  } else {
    xs = sTrace.rayOrigin + reconRay.direction * sHit.dist;

    // ── 3. Suffix radiance Lo leaving xs toward xv ──
    let reconDirToXv = safe_normalize(xv - xs); // wo at xs (Lo is measured here)
    let sReservoirMat = rptSuffixMaterialAtHit(sHit, reconRay.direction, reconDirToXv, heroLambda);
    ns = sReservoirMat.normal;
    // Suffix bounce budget: bounded short walk (the reconnection-vertex tail). Kept
    // modest; the wiring step can align it with the megakernel bounce budget for
    // A/B parity. maxBounces is the host's path budget; the suffix is at most that.
    let suffixBounces = max(1u, min(params.maxBounces, 4u));
    let suffixEstimate = rptComputeLoAtReconnection(
      &rng, xs, sHit, reconRay.direction, reconDirToXv, heroLambda, suffixBounces,
    );
    if (!suffixEstimate.valid) {
      storeReservoirPTHero_rw(&rpt_reservoirOut, pixelIdx, emptyReservoirPTHero());
      return;
    }
    Lo = suffixEstimate.Lo;
  }

  // ── Seed a 1-sample RIS reservoir, finalise, store ──
  var r = emptyReservoirPTHero();
  r.xv = xv; r.nv = nv;
  r.woV = restirpt_safe_normalize(woV);
  r.heroLambdaV = heroLambda;
  r.isFrontFaceV = vIsFront;
  r.materialIdV = vMatId;
  r.instanceIndexV = vHit.instanceIndex;
  r.triangleIndexV = vHit.triIndex;
  r.surfaceParamV = rptVisibleSurfaceParam(vHit, xv);
  r.clearcoatNormalV = clearcoatNormalV;
  r.albV = baseColorV; r.roughnessV = roughnessV; r.metalV = metallicV;
  r.clearcoatV = clearcoatV;
  r.clearcoatRoughnessV = clearcoatRoughnessV;
  r.sheenV = sheenV;
  r.sheenRoughnessV = sheenRoughnessV;
  r.sheenColorV = sheenColorV;
  r.iridescenceV = iridescenceV;
  r.iridescenceIorV = iridescenceIorV;
  r.iridescenceThicknessMinV = iridescenceThicknessMinV;
  r.iridescenceThicknessMaxV = iridescenceThicknessMaxV;
  r.specularColorV = specularColorV;
  r.specularIntensityV = specularIntensityV;
  r.anisotropyV = anisotropyV;
  r.anisotropyRotationV = anisotropyRotationV;
  // Candidate target (integrand-matching: f_bsdf·cos·Lo with the visible-vertex
  // BRDF) for the single reconnection sample. The candidate weight is p̂ / p_src.
  // For 1-sample RIS p̂ cancels in W = w_sum/p̂ = 1/p_src, so this does NOT change the
  // producer's mean — it sets the cross-frame-consistent p̂ the temporal MIS
  // resamples against (the same target finalise uses). Canonicalize Lo before
  // either evaluation: update/finalise operates in the compact stored-Lo domain,
  // so mixing a raw candidate p̂ with the packed winner p̂ would break the exact
  // one-candidate cancellation whenever shared-exponent quantization changes Lo.
  Lo = rptCanonicalizeStoredLo(Lo);
  let pHat = restirPtTargetForDomainAtHero(r, heroLambda, woV, xs, Lo);
  let wCandidate = select(0.0, pHat / pdfSrc, pdfSrc > 1e-8);
  if (!updateReservoirPT(
    &r, xs, ns, Lo, heroLambda, pdfSrc, wCandidate, &rng,
  )) {
    storeReservoirPTHero_rw(&rpt_reservoirOut, pixelIdx, emptyReservoirPTHero());
    return;
  }
  // GRIS finalize: W = w_sum / p̂ (NO /M — the temporal pass folds with MIS).
  finaliseReservoirPTWGris(&r, rptParams.wCap);
  // Validate the selected one-edge reconnection path xv → xs.
  refreshReconnectionStatePT(&r);

  storeReservoirPTHero_rw(&rpt_reservoirOut, pixelIdx, r);
}
`;
