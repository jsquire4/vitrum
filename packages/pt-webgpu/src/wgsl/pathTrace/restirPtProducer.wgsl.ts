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
 * as  prefixThroughput · f_bsdf(xv; wo→wi_recon) · cos(nv, wi_recon) · Lo · W.
 * If Lo folded in the prefix f·cos (or the prefix throughput), resolve would
 * DOUBLE-apply it. Keeping Lo = "radiance leaving xs, suffix-throughput-1" makes
 * the reconnection split clean and matches Lin 2022's cached L_o at x_s.
 *
 * The producer's 1-sample RIS candidate weight is  w = p̂ / p_src  where
 *   p̂    = restirPtTargetAt(xv, nv, woV, mat, xs, Lo) = luminance(f_bsdf·cos·Lo)  (integrand-matching, B3)
 *          (the diffuse-cosine resampling proxy — a SCALAR heuristic), and
 *   p_src = the REAL directional pdf that GENERATED wi_recon at xv
 *          (brdfDirectionalPdf at the visible vertex).
 * After finalize W = w_sum/p̂ = 1/p_src, and resolve forms f·cos·Lo/p_src — the
 * unbiased single-bounce estimator. p̂ cancels (see reservoirPtHero.wgsl.ts
 * unbiasedness note); storing the REAL p_src is what makes a GLOSSY xv unbiased.
 *
 * ════════════════════════════════════════════════════════════════════════════
 * ENERGY-CONSISTENCY RISKS the producer guards (and the ones it does NOT)
 * ════════════════════════════════════════════════════════════════════════════
 * GUARDED:
 *   • SPECULAR / TRANSMISSIVE visible vertex → write an EMPTY reservoir (M = 0).
 *     A near-singular prefix BSDF cannot be reused via the geometric reconnection
 *     shift (reusing a neighbour's Lo through a different wi_recon is invalid),
 *     and the diffuse-cosine target is meaningless there. Matches the GI producer
 *     writing empty for glass/metal primaries. The pixel simply does not reuse.
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
 * HONEST UNGUARDED bias (documented regime limits, NOT bugs):
 *   • MODERATELY-GLOSSY xv: the reconnection shift is geometric (holds xs fixed,
 *     re-roots the edge) but the glossy BRDF at xv is direction-sensitive; the
 *     diffuse-cosine proxy target under-weights it, so cross-pixel/temporal reuse
 *     of such a reservoir is APPROXIMATE. The PRODUCER itself is unbiased for
 *     glossy xv (it stores the real p_src); the bias is introduced by the REUSE
 *     pass (temporal feedback can drift on glossy surfaces). Prefix-1 reconnection
 *     reuse is physically exact only for a DIFFUSE visible vertex.
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

// Visible-vertex classification: a sufficiently-diffuse primary hit is reusable;
// a specular / transmissive one is not (see the producer header). Mirrors the GI
// producer's "write empty for glass/metal" gate.
fn rptIsReusableVisibleVertex(roughness: f32, metallic: f32, transmission: f32) -> bool {
  // Transmissive → never (the reconnection edge is a refraction, not a reusable
  // diffuse/glossy bounce). Near-mirror metal → never. Otherwise reusable, with
  // the documented glossy-reuse approximation for the moderate-roughness middle.
  if (transmission > 0.01) { return false; }
  if (rptParams.allowGlossyReuse == 0u) {
    // Default to the regime the prefix-1 reconnection shift can validate
    // radiometrically: diffuse-ish visible vertices. Glossy/metallic visible
    // domains can still be explored via experimentalGlossyReuse, but they are
    // not admitted into the temporal/spatial GRIS feedback loop by default.
    return metallic <= 0.05 && roughness >= 0.35;
  }
  if (metallic > 0.5 && roughness < 0.08) { return false; }
  return roughness >= 0.08;
}

struct RptAlphaTraceHit {
  hit: SceneHit,
  rayOrigin: vec3f,
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
fn rptTraceClosestAfterAlpha(rayIn: Ray, rng: ptr<function, u32>) -> RptAlphaTraceHit {
  var ray = rayIn;
  var hit = traceClosest(ray, 1e-4, INFINITY);
  for (var aSkip = 0u; aSkip < 8u; aSkip = aSkip + 1u) {
    if (!hit.didHit || !alphaTestPassThrough(hitMaterialId(hit), hit.triIndex, hit.baryVW, rng)) {
      break;
    }
    ray.origin = ray.origin + ray.direction * (hit.dist + 1e-4);
    hit = traceClosest(ray, 1e-4, INFINITY);
  }
  var out: RptAlphaTraceHit;
  out.hit = hit;
  out.rayOrigin = ray.origin;
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
  let isFrontFace = dot(hit.normal, incomingDir) < 0.0;
  var out: RptSuffixMaterial;
  out.baseColor = mat.baseColor * sampleVertexColor(hit.triIndex, hit.baryVW).rgb * sampleBaseColorTexture(matId, hit.triIndex, hit.baryVW).rgb;
  out.baseColor = out.baseColor * sampleAoFactor(matId, hit.triIndex, hit.baryVW);
  let ormSample = sampleOrmTexture(matId, hit.triIndex, hit.baryVW);
  out.roughness = clamp(mat.roughness * ormSample.g, 0.02, 1.0);
  out.metallic = clamp(mat.metallic * ormSample.b, 0.0, 1.0);
  out.emissive = mat.emissive * sampleEmissiveTexture(matId, hit.triIndex, hit.baryVW).rgb;
  out.transmission = clamp(mat.transmission * sampleTransmissionTexture(matId, hit.triIndex, hit.baryVW), 0.0, 1.0);
  out.ior = mat.ior;
  if (params.spectralEnabled != 0u && mat.dispersionAbbe >= 1.0) {
    out.ior = cauchyIorAtLambda(heroLambda, mat.ior, mat.dispersionAbbe);
  }
  out.normal = select(-hit.normal, hit.normal, isFrontFace);
  out.normal = applyNormalMap(matId, hit.triIndex, hit.baryVW, out.normal, hit.instanceIndex, isFrontFace);
  out.normal = applyBumpMap(matId, hit.triIndex, hit.baryVW, out.normal, hit.instanceIndex);
  out.clearcoatNormal = applyClearcoatNormalMap(matId, hit.triIndex, hit.baryVW, out.normal, hit.instanceIndex);

  out.clearcoat = clamp(mat.clearcoat * sampleClearcoatTexture(matId, hit.triIndex, hit.baryVW), 0.0, 1.0);
  out.clearcoatRoughness = clamp(mat.clearcoatRoughness * sampleClearcoatRoughnessTexture(matId, hit.triIndex, hit.baryVW), 0.0, 1.0);
  out.sheen = mat.sheen;
  out.sheenRoughness = clamp(mat.sheenRoughness * sampleSheenRoughnessTexture(matId, hit.triIndex, hit.baryVW), 0.0, 1.0);
  out.sheenColor = clamp(mat.sheenColor * sampleSheenColorTexture(matId, hit.triIndex, hit.baryVW), vec3f(0.0), vec3f(1.0));
  out.iridescence = clamp(mat.iridescence * sampleIridescenceTexture(matId, hit.triIndex, hit.baryVW), 0.0, 1.0);
  let iridescenceThicknessSample = sampleIridescenceThicknessTexture(matId, hit.triIndex, hit.baryVW);
  out.iridescenceThicknessMin = mat.iridescenceThicknessMin;
  out.iridescenceThicknessMax = mat.iridescenceThicknessMax;
  if (iridescenceThicknessSample >= 0.0) {
    let iridescenceThickness = mix(mat.iridescenceThicknessMin, mat.iridescenceThicknessMax, iridescenceThicknessSample);
    out.iridescenceThicknessMin = iridescenceThickness;
    out.iridescenceThicknessMax = iridescenceThickness;
    if (iridescenceThickness <= 0.0) { out.iridescence = 0.0; }
  }
  out.iridescenceIor = mat.iridescenceIor;
  out.specularColor = clamp(mat.specularColor * sampleSpecularColorTexture(matId, hit.triIndex, hit.baryVW), vec3f(0.0), vec3f(1.0));
  out.specularIntensity = clamp(mat.specularIntensity * sampleSpecularIntensityTexture(matId, hit.triIndex, hit.baryVW), 0.0, 1.0);
  out.anisotropy = materialAnisotropy(matId, hit.triIndex, hit.baryVW);
  out.anisotropyRotation = materialAnisotropyRotation(matId, hit.triIndex, hit.baryVW);
  out.envMapIntensity = materialEnvMapIntensity(matId);
  out.isUnlit = mat.isUnlit;

  let layerTx = clamp(select(mat.backLayerTx, mat.frontLayerTx, isFrontFace), vec3f(0.0), vec3f(1.0));
  let layerRoughness = select(mat.backLayerRoughness, mat.frontLayerRoughness, isFrontFace);
  if (layerRoughness >= 0.0) {
    out.roughness = clamp(layerRoughness, 0.02, 1.0);
  }
  let layerWeight = select(
    layerTx,
    activeLayerWeightRgb(layerTx, heroLambda, true),
    params.spectralEnabled != 0u && luminance(layerTx) < 0.999,
  );
  out.baseColor = out.baseColor * layerWeight;
  if (mat.thinFilmEnabled) {
    let viewCos = clamp(dot(out.normal, wo), 0.0, 1.0);
    var thinFilmReflectTint = vec3f(1.0);
    if (params.spectralEnabled != 0u) {
      let rt = thinFilmTmmRt(
        matId,
        mat.thinFilmLayerCountU,
        heroLambda,
        out.ior,
        mat.thinFilmIncidentIor,
        mat.thinFilmAngleDependent,
        viewCos,
      );
      thinFilmReflectTint = vec3f(clamp(rt.x, 0.0, 1.0));
    } else {
      let rtR = thinFilmTmmRt(matId, mat.thinFilmLayerCountU, 630.0, out.ior, mat.thinFilmIncidentIor, mat.thinFilmAngleDependent, viewCos);
      let rtG = thinFilmTmmRt(matId, mat.thinFilmLayerCountU, 540.0, out.ior, mat.thinFilmIncidentIor, mat.thinFilmAngleDependent, viewCos);
      let rtB = thinFilmTmmRt(matId, mat.thinFilmLayerCountU, 460.0, out.ior, mat.thinFilmIncidentIor, mat.thinFilmAngleDependent, viewCos);
      thinFilmReflectTint = clamp(vec3f(rtR.x, rtG.x, rtB.x), vec3f(0.0), vec3f(1.0));
    }
    let layerStrength = clamp(0.12 + 0.06 * f32(mat.thinFilmLayerCountU), 0.0, 0.55);
    let filmStrength = clamp(layerStrength * (1.0 - out.roughness), 0.0, 0.6);
    out.baseColor = mix(out.baseColor, out.baseColor * thinFilmReflectTint, filmStrength);
  }
  if (params.spectralEnabled != 0u) {
    let reflScalar = select(
      max(luminance(out.baseColor), 0.0),
      evalJakobHanikaSpectrum(mat.spectralReflCoeffs, heroLambda),
      mat.hasSpectralReflectance,
    );
    out.baseColor = vec3f(reflScalar);
  }
  return out;
}

fn rptSampleDirectionalCone(rng: ptr<function, u32>, axisIn: vec3f, angularDiameter: f32) -> vec3f {
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
  rng: ptr<function, u32>,
  pos: vec3f,
  normal: vec3f,
  clearcoatNormal: vec3f,
  wo: vec3f,
  baseColor: vec3f,
  roughness: f32,
  metallic: f32,
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
    if (dIrrMean.w > 1e-6) {
      let angDiamRaw = dDirAD.w;
      let dirShadowDisabled = angDiamRaw < 0.0;
      let angDiam = select(angDiamRaw, -1.0 - angDiamRaw, dirShadowDisabled);
      let lightDir = rptSampleDirectionalCone(rng, dDirAD.xyz, angDiam);
      let nDotL = max(0.0, dot(normal, lightDir));
      if (nDotL > 0.0) {
        let shadowRay = Ray(pos + normal * 1e-3, lightDir);
        if (dirShadowDisabled || !traceAny(shadowRay, 1e-4, INFINITY)) {
          let brdf = evaluateBrdfFullWithClearcoatNormal(
            baseColor, roughness, metallic, normal, clearcoatNormal, wo, lightDir,
            clearcoat, clearcoatRoughness, sheen, sheenRoughness, sheenColor,
            iridescence, iridescenceIor, iridescenceThicknessMin, iridescenceThicknessMax,
            specularColor, specularIntensity,
            anisotropy, anisotropyRotation,
          );
          contrib = contrib + suffixThroughput * brdf * nDotL * dIrrMean.rgb;
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
      let a = xi1r * 2.0 - 1.0;
      let b = xi2r * 2.0 - 1.0;
      var cr: f32; var cphi: f32;
      if (abs(a) >= abs(b)) {
        cr = a; cphi = (PI / 4.0) * (b / max(abs(a), 1e-9));
      } else {
        cr = b; cphi = (PI / 2.0) - (PI / 4.0) * (a / max(abs(b), 1e-9));
      }
      lpos = rpos + ru * (cr * cos(cphi)) + rv * (cr * sin(cphi));
      area = max(PI * rrad * rrad, 1e-6);
    } else {
      lpos = rpos + ru * (xi1r * 2.0 - 1.0) + rv * (xi2r * 2.0 - 1.0);
      area = max(4.0 * length(cross(ru, rv)), 1e-6);
    }
    let toLight = lpos - pos;
    let dist2 = max(dot(toLight, toLight), 1e-6);
    let dist = sqrt(dist2);
    let wi = toLight / dist;
    let nDotL = max(dot(normal, wi), 0.0);
    if (nDotL > 0.0) {
      let lightNormal = safe_normalize(cross(ru, rv));
      let cosLight = max(dot(lightNormal, -wi), 0.0);
      if (cosLight > 0.0) {
        let lightPdf = dist2 / max(cosLight * area, 1e-6);
        let shadowRay = Ray(pos + normal * 1e-3, wi);
        if (rectShadowDisabled || !traceAny(shadowRay, 1e-4, max(dist - 2e-3, 1e-3))) {
          let brdf = evaluateBrdfFullWithClearcoatNormal(
            baseColor, roughness, metallic, normal, clearcoatNormal, wo, wi,
            clearcoat, clearcoatRoughness, sheen, sheenRoughness, sheenColor,
            iridescence, iridescenceIor, iridescenceThicknessMin, iridescenceThicknessMax,
            specularColor, specularIntensity,
            anisotropy, anisotropyRotation,
          );
          contrib = contrib + suffixThroughput * brdf * nDotL * rr / max(lightPdf, 1e-6);
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
        let attenuation = select(1.0 / dist2, pow(max(dist, 1.0), -ptDecay), ptDecay > 0.01);
        let brdf = evaluateBrdfFullWithClearcoatNormal(
          baseColor, roughness, metallic, normal, clearcoatNormal, wo, wi,
          clearcoat, clearcoatRoughness, sheen, sheenRoughness, sheenColor,
          iridescence, iridescenceIor, iridescenceThicknessMin, iridescenceThicknessMax,
          specularColor, specularIntensity,
          anisotropy, anisotropyRotation,
        );
        contrib = contrib + suffixThroughput * brdf * nDotL * rad * attenuation;
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
          let attenuation = select(1.0 / dist2, pow(max(dist, 1.0), -spDecay), spDecay > 0.01);
          let brdf = evaluateBrdfFullWithClearcoatNormal(
            baseColor, roughness, metallic, normal, clearcoatNormal, wo, wi,
            clearcoat, clearcoatRoughness, sheen, sheenRoughness, sheenColor,
            iridescence, iridescenceIor, iridescenceThicknessMin, iridescenceThicknessMax,
            specularColor, specularIntensity,
            anisotropy, anisotropyRotation,
          );
          contrib = contrib + suffixThroughput * brdf * nDotL * softness * srad * attenuation;
        }
      }
    }
  }
  // Mesh-area lights: full-weight area-measure connection for implicit and
  // explicit mesh emitters. This mirrors the main kernel's triangle sampler
  // without light-selection compensation because this producer loops all lights.
  for (var mi = 0u; mi < params.meshAreaLightCount; mi = mi + 1u) {
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
    let lpos = a * uu + b * vv + c * ww;
    let toLight = lpos - pos;
    let dist2 = max(dot(toLight, toLight), 1e-6);
    let dist = sqrt(dist2);
    let wi = toLight / dist;
    let nDotL = max(dot(normal, wi), 0.0);
    if (nDotL > 0.0) {
      let lightNormal = safe_normalize(cross(b - a, c - a));
      let cosLight = max(dot(lightNormal, -wi), 0.0);
      if (cosLight > 0.0) {
        let area = max(0.5 * length(cross(b - a, c - a)), 1e-6);
        let lightPdf = dist2 / max(cosLight * area, 1e-6);
        let shadowRay = Ray(pos + normal * 1e-3, wi);
        if (meshAreaLights[mb + 3u].w > 0.5 || !traceAny(shadowRay, 1e-4, max(dist - 2e-3, 1e-3))) {
          let brdf = evaluateBrdfFullWithClearcoatNormal(
            baseColor, roughness, metallic, normal, clearcoatNormal, wo, wi,
            clearcoat, clearcoatRoughness, sheen, sheenRoughness, sheenColor,
            iridescence, iridescenceIor, iridescenceThicknessMin, iridescenceThicknessMax,
            specularColor, specularIntensity,
            anisotropy, anisotropyRotation,
          );
          contrib = contrib + suffixThroughput * brdf * nDotL * mr / max(lightPdf, 1e-6);
        }
      }
    }
  }
  // Environment (importance-sampled if a map is present), MIS vs the BRDF pdf.
  if (hasEnvironmentMap() || params.environmentSun.w > 1e-6) {
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
      envPdf = max(environmentPdf(envDir), 1e-8);
    }
    let nDotL = max(dot(normal, envDir), 0.0);
    if (nDotL > 1e-6) {
      let shadowRay = Ray(pos + normal * 1e-3, envDir);
      if (!traceAny(shadowRay, 1e-4, INFINITY)) {
        let brdf = evaluateBrdfFullWithClearcoatNormal(
          baseColor, roughness, metallic, normal, clearcoatNormal, wo, envDir,
          clearcoat, clearcoatRoughness, sheen, sheenRoughness, sheenColor,
          iridescence, iridescenceIor, iridescenceThicknessMin, iridescenceThicknessMax,
          specularColor, specularIntensity,
          anisotropy, anisotropyRotation,
        );
        let brdfPdf = brdfDirectionalPdfFullSampledWithClearcoatNormal(
          baseColor, roughness, metallic, 0.0, 1.0, normal, clearcoatNormal, wo, envDir,
          clearcoat, clearcoatRoughness, sheen, sheenRoughness,
          iridescence, iridescenceIor, iridescenceThicknessMin, iridescenceThicknessMax,
          specularColor, specularIntensity,
          anisotropy, anisotropyRotation,
        );
        // Mirror the megakernel environment estimators: spectral mode evaluates
        // env radiance at the hero wavelength, and the current surface's
        // envMapIntensity scales both the NEE and BSDF-escape halves.
        let envColorOut = select(envColor, spectralEmissionAtHero(envColor, heroLambda), params.spectralEnabled != 0u) * envMapIntensity;
        let misWeight = powerHeuristic(envPdf, brdfPdf);
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
  rng: ptr<function, u32>,
  xs: vec3f,
  xsHit: SceneHit,
  incomingDir: vec3f,
  reconDir: vec3f,
  heroLambda: f32,
  suffixBounces: u32,
) -> vec3f {
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
    let normal = sm.normal;
    let baseColor = sm.baseColor;
    let roughness = max(sm.roughness, 0.02);
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
      rng, pos, normal, sm.clearcoatNormal, wo, baseColor, roughness, metallic,
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
    let fOnward = evaluateBrdfFullWithClearcoatNormal(
      baseColor, roughness, metallic, normal, sm.clearcoatNormal, wo, nextDir,
      sm.clearcoat, sm.clearcoatRoughness, sm.sheen, sm.sheenRoughness, sm.sheenColor,
      sm.iridescence, sm.iridescenceIor, sm.iridescenceThicknessMin, sm.iridescenceThicknessMax,
      sm.specularColor, sm.specularIntensity,
      sm.anisotropy, sm.anisotropyRotation,
    );
    suffixThroughput = suffixThroughput * fOnward * nDotNext / max(cosSample.pdf, 1e-8);
    prevAllowsAreaMis = true; // diffuse onward bounce: next emission handled by NEE/MIS.

    let nextTrace = rptTraceClosestAfterAlpha(Ray(pos + normal * 1e-3, nextDir), rng);
    let nextHit = nextTrace.hit;
    if (!nextHit.didHit) {
      let envRgb = sampleEnvironmentColor(nextDir);
      let envContribution = select(envRgb, spectralEmissionAtHero(envRgb, heroLambda), params.spectralEnabled != 0u);
      Lo = Lo + suffixThroughput * envContribution * sm.envMapIntensity;
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
  return Lo;
}

fn rptSourceLobeWeightSum(clearcoat: f32, sheen: f32) -> f32 {
  return max(1.0 + max(clearcoat, 0.0) + max(sheen, 0.0), 1e-4);
}

fn rptSampleSourceReconnectionDirection(
  rng: ptr<function, u32>,
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
      if (anisotropy > 1e-4) {
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
  var vHit = traceClosest(primaryRay, 1e-4, INFINITY);
  for (var aSkip = 0u; aSkip < 8u; aSkip = aSkip + 1u) {
    if (!vHit.didHit || !alphaTestPassThrough(hitMaterialId(vHit), vHit.triIndex, vHit.baryVW, &rng)) {
      break;
    }
    primaryRay.origin = primaryRay.origin + primaryRay.direction * (vHit.dist + 1e-4);
    vHit = traceClosest(primaryRay, 1e-4, INFINITY);
  }
  if (!vHit.didHit) {
    storeReservoirPTHero_rw(&rpt_reservoirOut, pixelIdx, emptyReservoirPTHero());
    return;
  }
  let vMatId = hitMaterialId(vHit);
  let vMat = decodeMaterial(vMatId);
  let xv = primaryRay.origin + primaryRay.direction * vHit.dist;
  let vIsFront = dot(vHit.normal, primaryRay.direction) < 0.0;
  var nv = select(-vHit.normal, vHit.normal, vIsFront);
  nv = applyNormalMap(vMatId, vHit.triIndex, vHit.baryVW, nv, vHit.instanceIndex, vIsFront);
  nv = applyBumpMap(vMatId, vHit.triIndex, vHit.baryVW, nv, vHit.instanceIndex);
  let clearcoatNormalV = applyClearcoatNormalMap(vMatId, vHit.triIndex, vHit.baryVW, nv, vHit.instanceIndex);
  let woV = -primaryRay.direction; // eye-side direction at xv
  var baseColorV = vMat.baseColor * sampleVertexColor(vHit.triIndex, vHit.baryVW).rgb * sampleBaseColorTexture(vMatId, vHit.triIndex, vHit.baryVW).rgb;
  baseColorV = baseColorV * sampleAoFactor(vMatId, vHit.triIndex, vHit.baryVW);
  let ormSampleV = sampleOrmTexture(vMatId, vHit.triIndex, vHit.baryVW);
  var roughnessV = clamp(vMat.roughness * ormSampleV.g, 0.02, 1.0);
  var metallicV = clamp(vMat.metallic * ormSampleV.b, 0.0, 1.0);
  var transmissionV = clamp(vMat.transmission * sampleTransmissionTexture(vMatId, vHit.triIndex, vHit.baryVW), 0.0, 1.0);
  var clearcoatV = clamp(vMat.clearcoat * sampleClearcoatTexture(vMatId, vHit.triIndex, vHit.baryVW), 0.0, 1.0);
  var clearcoatRoughnessV = clamp(vMat.clearcoatRoughness * sampleClearcoatRoughnessTexture(vMatId, vHit.triIndex, vHit.baryVW), 0.0, 1.0);
  var sheenV = vMat.sheen;
  var sheenRoughnessV = clamp(vMat.sheenRoughness * sampleSheenRoughnessTexture(vMatId, vHit.triIndex, vHit.baryVW), 0.0, 1.0);
  var sheenColorV = clamp(vMat.sheenColor * sampleSheenColorTexture(vMatId, vHit.triIndex, vHit.baryVW), vec3f(0.0), vec3f(1.0));
  var iridescenceV = clamp(vMat.iridescence * sampleIridescenceTexture(vMatId, vHit.triIndex, vHit.baryVW), 0.0, 1.0);
  let iridescenceThicknessSampleV = sampleIridescenceThicknessTexture(vMatId, vHit.triIndex, vHit.baryVW);
  var iridescenceThicknessMinV = vMat.iridescenceThicknessMin;
  var iridescenceThicknessMaxV = vMat.iridescenceThicknessMax;
  if (iridescenceThicknessSampleV >= 0.0) {
    let iridescenceThicknessV = mix(vMat.iridescenceThicknessMin, vMat.iridescenceThicknessMax, iridescenceThicknessSampleV);
    iridescenceThicknessMinV = iridescenceThicknessV;
    iridescenceThicknessMaxV = iridescenceThicknessV;
    if (iridescenceThicknessV <= 0.0) { iridescenceV = 0.0; }
  }
  let iridescenceIorV = vMat.iridescenceIor;
  var specularColorV = clamp(vMat.specularColor * sampleSpecularColorTexture(vMatId, vHit.triIndex, vHit.baryVW), vec3f(0.0), vec3f(1.0));
  var specularIntensityV = clamp(vMat.specularIntensity * sampleSpecularIntensityTexture(vMatId, vHit.triIndex, vHit.baryVW), 0.0, 1.0);
  let anisotropyV = materialAnisotropy(vMatId, vHit.triIndex, vHit.baryVW);
  let anisotropyRotationV = materialAnisotropyRotation(vMatId, vHit.triIndex, vHit.baryVW);
  var iorV = vMat.ior;
  if (params.spectralEnabled != 0u && vMat.dispersionAbbe >= 1.0) {
    iorV = cauchyIorAtLambda(heroLambda, vMat.ior, vMat.dispersionAbbe);
  }
  let layerTxV = clamp(select(vMat.backLayerTx, vMat.frontLayerTx, vIsFront), vec3f(0.0), vec3f(1.0));
  let layerRoughnessV = select(vMat.backLayerRoughness, vMat.frontLayerRoughness, vIsFront);
  if (layerRoughnessV >= 0.0) {
    roughnessV = clamp(layerRoughnessV, 0.02, 1.0);
  }
  let layerWeightV = select(
    layerTxV,
    activeLayerWeightRgb(layerTxV, heroLambda, true),
    params.spectralEnabled != 0u && luminance(layerTxV) < 0.999,
  );
  baseColorV = baseColorV * layerWeightV;
  if (vMat.thinFilmEnabled) {
    let viewCosV = clamp(dot(nv, woV), 0.0, 1.0);
    var thinFilmReflectTintV = vec3f(1.0);
    if (params.spectralEnabled != 0u) {
      let rt = thinFilmTmmRt(
        vMatId,
        vMat.thinFilmLayerCountU,
        heroLambda,
        iorV,
        vMat.thinFilmIncidentIor,
        vMat.thinFilmAngleDependent,
        viewCosV,
      );
      thinFilmReflectTintV = vec3f(clamp(rt.x, 0.0, 1.0));
    } else {
      let rtR = thinFilmTmmRt(vMatId, vMat.thinFilmLayerCountU, 630.0, iorV, vMat.thinFilmIncidentIor, vMat.thinFilmAngleDependent, viewCosV);
      let rtG = thinFilmTmmRt(vMatId, vMat.thinFilmLayerCountU, 540.0, iorV, vMat.thinFilmIncidentIor, vMat.thinFilmAngleDependent, viewCosV);
      let rtB = thinFilmTmmRt(vMatId, vMat.thinFilmLayerCountU, 460.0, iorV, vMat.thinFilmIncidentIor, vMat.thinFilmAngleDependent, viewCosV);
      thinFilmReflectTintV = clamp(vec3f(rtR.x, rtG.x, rtB.x), vec3f(0.0), vec3f(1.0));
    }
    let layerStrengthV = clamp(0.12 + 0.06 * f32(vMat.thinFilmLayerCountU), 0.0, 0.55);
    let filmStrengthV = clamp(layerStrengthV * (1.0 - roughnessV), 0.0, 0.6);
    baseColorV = mix(baseColorV, baseColorV * thinFilmReflectTintV, filmStrengthV);
  }
  if (params.spectralEnabled != 0u) {
    let reflScalarV = select(
      max(luminance(baseColorV), 0.0),
      evalJakobHanikaSpectrum(vMat.spectralReflCoeffs, heroLambda),
      vMat.hasSpectralReflectance,
    );
    baseColorV = vec3f(reflScalarV);
  }

  // Specular / transmissive visible vertex → not reusable; write empty.
  if (vMat.isUnlit || !rptIsReusableVisibleVertex(roughnessV, metallicV, transmissionV)) {
    storeReservoirPTHero_rw(&rpt_reservoirOut, pixelIdx, emptyReservoirPTHero());
    return;
  }

  // ── 2. One BSDF-sampled bounce off xv → reconnection vertex xs ──
  // Sample the reconnection direction wi_recon from the visible-vertex BSDF.
  // The hero stack stores the REAL source pdf so a glossy xv stays unbiased.
  var tanT: vec3f;
  var tanB: vec3f;
  buildOnb(nv, &tanT, &tanB);
  if (anisotropyV > 1e-4) {
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
  // The REAL source directional pdf that generated wi_recon (unbiasedness-
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
  if (pdfSrc <= 1e-8) {
    storeReservoirPTHero_rw(&rpt_reservoirOut, pixelIdx, emptyReservoirPTHero());
    return;
  }

  let reconRay = Ray(xv + nv * 1e-3, wiRecon);
  let sTrace = rptTraceClosestAfterAlpha(reconRay, &rng);
  let sHit = sTrace.hit;
  var xs: vec3f;
  var ns: vec3f;
  var Lo: vec3f;
  if (!sHit.didHit) {
    // The reconnection ray ESCAPED to the environment. DO NOT write empty — that
    // would (a) drop this pixel's indirect entirely and (b) zero xv so the
    // temporal pass cannot reproject and the pixel loses ALL its history, leaving
    // it permanently black (the open-Cornell-face dropout that made ~22% of lit
    // pixels read zero). Mirror the PROVEN GI producer (risGi.wgsl.ts:252-257):
    // synthesize a FAR reconnection vertex along the escape direction and treat
    // the environment radiance as the outgoing radiance Lo leaving it. The
    // reconnection shift holds this synthetic xs fixed exactly like a real one
    // (ns = −wi_recon ⇒ cosθ_out = 1; the half-G dist² term is finite & large,
    // so the Jacobian is well-defined). sampleEnvironmentColor matches the
    // megakernel's escape term (kernel.wgsl.ts:326).
    let kReconEscapeDist: f32 = 100.0; // GI RECONNECT_MAX_DIST analogue
    xs = reconRay.origin + wiRecon * kReconEscapeDist;
    ns = -wiRecon;
    let reconEnvRgb = sampleEnvironmentColor(wiRecon);
    let reconEnv = select(reconEnvRgb, spectralEmissionAtHero(reconEnvRgb, heroLambda), params.spectralEnabled != 0u);
    let envScaleV = materialEnvMapIntensity(vMatId);
    Lo = reconEnv * envScaleV;
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
    Lo = rptComputeLoAtReconnection(&rng, xs, sHit, reconRay.direction, reconDirToXv, heroLambda, suffixBounces);
  }

  // ── Seed a 1-sample RIS reservoir, finalise, store ──
  var r = emptyReservoirPTHero();
  r.xv = xv; r.nv = nv;
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
  r.prefixVertexCount = 1u;
  // Candidate target (integrand-matching: f_bsdf·cos·Lo with the visible-vertex
  // BRDF) for the single reconnection sample. The candidate weight is p̂ / p_src.
  // For 1-sample RIS p̂ cancels in W = w_sum/p̂ = 1/p_src, so this does NOT change the
  // producer's mean — it sets the cross-frame-consistent p̂ the temporal MIS
  // resamples against (the same target finalise uses).
  let pHat = restirPtTargetForDomain(r, woV, xs, Lo);
  let wCandidate = select(0.0, pHat / pdfSrc, pdfSrc > 1e-8);
  updateReservoirPT(&r, xs, ns, Lo, pdfSrc, wCandidate, &rng);
  // GRIS finalize: W = w_sum / p̂ (NO /M — the temporal pass folds with MIS).
  finaliseReservoirPTWGris(&r, rptParams.wCap, params.cameraPos.xyz);
  // Refresh the reconnection-shift cache from the chosen base edge xv → xs.
  refreshReconnectionCachePT(&r, params.cameraPos.xyz, params.frameSeed ^ pixelIdx);

  storeReservoirPTHero_rw(&rpt_reservoirOut, pixelIdx, r);
}
`;
