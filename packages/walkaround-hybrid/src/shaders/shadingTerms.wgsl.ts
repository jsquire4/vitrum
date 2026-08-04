/**
 * shadingTerms.wgsl.ts — the 8 per-lighting-term helper functions for
 * the walkaround-hybrid shade pass.
 *
 * Exported as a BARE STRING (`SHADING_TERMS_WGSL`), NOT a WgslModule.
 *
 * ── Why not a WgslModule? (D5.8b — resolved 2026-06-11) ──────────────────────
 * The lo_* helpers reference @group(N)/@binding(M) declarations and the
 * WalkaroundUBO binding that are declared in SHADE_WGSL's own body (not in any
 * required module). The wgslComposer emits required modules BEFORE the
 * consumer's own source, so a WgslModule for these helpers would be placed
 * before those declarations exist in the concatenated string → naga rejects
 * the undefined names.
 *
 * The proven alternative (the temporalGiCommon.wgsl.ts pattern) is to export
 * the text as a raw string and interpolate it into SHADE_WGSL's template
 * literal at the exact position where the inline block used to live.  The
 * final composed shader string is byte-identical to the original; only the
 * TypeScript file organisation improves.
 *
 * ── Binding / symbol contract ─────────────────────────────────────────────────
 * The includer (SHADE_WGSL) MUST have already declared, before this string is
 * interpolated:
 *
 *   Textures / buffers (group 1):
 *     bvh            : array<BVHNode>           @group(1) @binding(0)
 *     bvh_index      : array<vec4u>             @group(1) @binding(1)
 *     bvh_position   : array<vec4f>             @group(1) @binding(2)
 *     emitters       : array<EmitterTri>        @group(1) @binding(3)
 *     emitterCdf     : array<f32>               @group(1) @binding(4)
 *     bvh_beer       : texture_2d<u32>          @group(1) @binding(5)
 *     tlasNodes      : array<BVHNode>           @group(1) @binding(6)
 *     tlasInstanceIndices    : array<u32>       @group(1) @binding(7)
 *     tlasBlasRoots  : array<u32>               @group(1) @binding(8)
 *     tlasInstanceWorldToLocal : array<vec4f>   @group(1) @binding(9)
 *     tlasInstanceLocalToWorld : array<vec4f>   @group(1) @binding(10)
 *     bvh_normal     : array<vec4f>             @group(1) @binding(11)
 *     restir_gi_bvh_emissive : texture_2d<f32>  @group(1) @binding(12)
 *     analytic_lights: texture_2d<f32>          @group(1) @binding(13)
 *     bvh_material   : texture_2d<u32>          @group(1) @binding(14)
 *
 *   UBO (group 2):
 *     ubo : WalkaroundUBO    @group(2) @binding(0)
 *
 *   Reservoirs (group 0):
 *     spatialReservoir      : array<u32>  @group(0) @binding(7)
 *     reservoirGiCurrent    : array<u32>  @group(0) @binding(11)
 *
 *   Structs / constants declared by required modules:
 *     BVHNode, EmitterTri, ReservoirDI, ReservoirGI, WalkaroundUBO
 *     INV_PI, ENV_SAMPLE_SENTINEL, BVH_BEER_TEX_WIDTH, BVH_MATERIAL_TEX_WIDTH
 *     SPEC_GI_ROUGH_MAX (declared in this string itself — not a pre-req)
 *
 *   Functions declared by required modules (common, surfaceTextures,
 *   ddgiGridUbo, sampleCascadeC0, stainedGlassShade, environmentSample):
 *     traceSceneAny, traceSceneAlphaTintTransmittanceTextured (SHADOW-01 / ALPHA-03),
 *     traceSceneAlphaTintTransmittanceTextured,
 *     traceSceneFirstHit, loadReservoirDI_rw,
 *     loadReservoirGI_rw, sampleEmitterPoint, emitterGeometry,
 *     rich-material GGX evaluators, decodeSurfaceTextureId,
 *     surfaceTextureMod, decodeMaterialColor, decodeIsMetal,
 *     decodeRoughMetal, decodeIor, envHasMap, envDirFromXi, envRadiance,
 *     sampleCascadeC0, safe_normalize, rand_f32, pcgInit,
 *     smoothShadingNormal, rcParams (struct field)
 *
 *   Local helper declared in SHADE_WGSL before this interpolation:
 *     loadSpatialDI (calls loadReservoirDI_rw on spatialReservoir)
 *
 * ── D5.8b history ────────────────────────────────────────────────────────────
 * Originally the lo_* block lived inline in SHADE_WGSL and carried a D5.8b
 * deferral comment explaining why a WgslModule extraction was blocked. The
 * temporalGiCommon raw-string pattern was the proven resolution: export the
 * text verbatim here and interpolate it with
 * `${SHADING_TERMS_WGSL}` at the same position in SHADE_WGSL's template
 * literal. Resolved 2026-06-11; byte-identical to the original composed
 * shader string.
 */
import { analyticLightFalloffWgsl } from './analyticLightFalloff.wgsl.js';
import { giBilinearWeightsWgsl, giBilinearCornerSelectWgsl } from './giBilinearGather.wgsl.js';
import { NATIVE_GLASS_GI_WGSL } from './risGiGlassWalk.wgsl.js';

export const SHADING_TERMS_WGSL = /* wgsl */ `// ── Stained-glass sun glow for primary glass hits ─────────────────────────
//
// Le ≈ attenuationColor × transmission × sunIntensity × |sunDot| × textureMod.
// attenuationColor is read from bvh_beer (Beer-Lambert visible color =
// pow(rawAttCol, thickness/attDist)) — separate from bvhIndex.w which
// carries the RAW attCol used by emitter Le and tinted-visibility.
//
// Generic transmissive glass is not emissive. This legacy stained-glass glow is
// gated behind the same opt-in sun-caustic flag as the other cathedral-window
// direct-light terms, so ordinary glass scenes do not receive un-authored sun
// radiance from HybridEngineOptions.primaryLightIntensity.
// Reservoir weights deliberately remain logarithmic until every positive
// radiometric factor has been incorporated. Converting logW to a linear f32
// first would discard otherwise representable products in both directions:
// e.g. 2^-200 * 2^100 and 2^200 * 2^-100 are both 2^-100.
const RESTIR_SHADE_MAX_FINITE_F32: f32 = 3.402823466e38;
const RESTIR_SHADE_LOG_ZERO: f32 = -3.402823466e38;
const RESTIR_SHADE_LOG2_OVERFLOW_F32: f32 = 128.0;
const RESTIR_SHADE_LOG2_ROUND_TO_ZERO_F32: f32 = -150.0;

fn restirShadeFinite(value: f32) -> bool {
  return value >= -RESTIR_SHADE_MAX_FINITE_F32 &&
    value <= RESTIR_SHADE_MAX_FINITE_F32;
}

fn restirShadeValidLog(value: f32) -> bool {
  return value > RESTIR_SHADE_LOG_ZERO && restirShadeFinite(value);
}

fn restirShadeAppendLogFactor(logValue: f32, logFactor: f32) -> f32 {
  if (!restirShadeValidLog(logValue) || !restirShadeValidLog(logFactor)) {
    return RESTIR_SHADE_LOG_ZERO;
  }
  let result = logValue + logFactor;
  if (result != result) { return RESTIR_SHADE_LOG_ZERO; }
  if (result > RESTIR_SHADE_MAX_FINITE_F32) {
    return RESTIR_SHADE_MAX_FINITE_F32;
  }
  if (result <= RESTIR_SHADE_LOG_ZERO) {
    return RESTIR_SHADE_LOG_ZERO;
  }
  return result;
}

fn restirShadeAppendPositiveFactor(logValue: f32, factor: f32) -> f32 {
  if (!restirShadeValidLog(logValue) ||
      !restirShadeFinite(factor) ||
      !(factor > 0.0)) {
    return RESTIR_SHADE_LOG_ZERO;
  }
  return restirShadeAppendLogFactor(logValue, log2(factor));
}

fn restirShadeRemovePositiveFactor(logValue: f32, factor: f32) -> f32 {
  if (!restirShadeValidLog(logValue) ||
      !restirShadeFinite(factor) ||
      !(factor > 0.0)) {
    return RESTIR_SHADE_LOG_ZERO;
  }
  let result = logValue - log2(factor);
  if (result != result) { return RESTIR_SHADE_LOG_ZERO; }
  if (result > RESTIR_SHADE_MAX_FINITE_F32) {
    return RESTIR_SHADE_MAX_FINITE_F32;
  }
  if (result <= RESTIR_SHADE_LOG_ZERO) {
    return RESTIR_SHADE_LOG_ZERO;
  }
  return result;
}

fn restirShadeLogProductChannel(
  logScale: f32,
  a: f32,
  b: f32,
  c: f32,
) -> f32 {
  var result = restirShadeAppendPositiveFactor(logScale, a);
  result = restirShadeAppendPositiveFactor(result, b);
  result = restirShadeAppendPositiveFactor(result, c);
  return result;
}

fn restirShadeLogProduct3(
  logScale: f32,
  a: vec3f,
  b: vec3f,
  c: vec3f,
) -> vec3f {
  return vec3f(
    restirShadeLogProductChannel(logScale, a.x, b.x, c.x),
    restirShadeLogProductChannel(logScale, a.y, b.y, c.y),
    restirShadeLogProductChannel(logScale, a.z, b.z, c.z),
  );
}

fn restirShadeLogAdd(a: f32, b: f32) -> f32 {
  if (!restirShadeValidLog(a)) { return b; }
  if (!restirShadeValidLog(b)) { return a; }
  let hi = max(a, b);
  let lo = min(a, b);
  let result = hi + log2(1.0 + exp2(lo - hi));
  if (result != result) { return RESTIR_SHADE_LOG_ZERO; }
  return min(result, RESTIR_SHADE_MAX_FINITE_F32);
}

fn restirShadeLogAdd3(a: vec3f, b: vec3f) -> vec3f {
  return vec3f(
    restirShadeLogAdd(a.x, b.x),
    restirShadeLogAdd(a.y, b.y),
    restirShadeLogAdd(a.z, b.z),
  );
}

fn restirShadeExp2Clamped(logValue: f32) -> f32 {
  if (!restirShadeValidLog(logValue) ||
      logValue <= RESTIR_SHADE_LOG2_ROUND_TO_ZERO_F32) {
    return 0.0;
  }
  if (logValue >= RESTIR_SHADE_LOG2_OVERFLOW_F32) {
    return RESTIR_SHADE_MAX_FINITE_F32;
  }
  return min(exp2(logValue), RESTIR_SHADE_MAX_FINITE_F32);
}

fn restirShadeExp2Clamped3(logValue: vec3f) -> vec3f {
  return vec3f(
    restirShadeExp2Clamped(logValue.x),
    restirShadeExp2Clamped(logValue.y),
    restirShadeExp2Clamped(logValue.z),
  );
}

fn restirShadeNormaliseLogSum(logSum: vec3f, denominator: f32) -> vec3f {
  if (!restirShadeFinite(denominator) || !(denominator > 0.0)) {
    return vec3f(RESTIR_SHADE_LOG_ZERO);
  }
  return vec3f(
    restirShadeRemovePositiveFactor(logSum.x, denominator),
    restirShadeRemovePositiveFactor(logSum.y, denominator),
    restirShadeRemovePositiveFactor(logSum.z, denominator),
  );
}

fn restirShadeAppendPositiveFactor3(
  logValue: vec3f,
  factor: vec3f,
) -> vec3f {
  return vec3f(
    restirShadeAppendPositiveFactor(logValue.x, factor.x),
    restirShadeAppendPositiveFactor(logValue.y, factor.y),
    restirShadeAppendPositiveFactor(logValue.z, factor.z),
  );
}

// Log-domain form of applyHomogeneousVolumeSingleScatterDirectional. GI
// contributions remain logarithmic until after bilinear accumulation, so this
// evaluates the same linear operator without introducing an exp2 endpoint per
// candidate. Every input contribution is non-negative.
fn restirShadeDirectionalVolumeLog(
  logRadiance: vec3f,
  albedo: vec3f,
  scatter: vec4f,
  pathLength: f32,
  normal: vec3f,
  wo: vec3f,
  wi: vec3f,
) -> vec3f {
  let sigmaS = max(scatter.rgb, vec3f(0.0));
  if (all(sigmaS <= vec3f(0.0)) || pathLength <= 0.0) {
    return logRadiance;
  }

  var logLuminance = RESTIR_SHADE_LOG_ZERO;
  logLuminance = restirShadeLogAdd(
    logLuminance,
    restirShadeAppendPositiveFactor(logRadiance.x, 0.2126),
  );
  logLuminance = restirShadeLogAdd(
    logLuminance,
    restirShadeAppendPositiveFactor(logRadiance.y, 0.7152),
  );
  logLuminance = restirShadeLogAdd(
    logLuminance,
    restirShadeAppendPositiveFactor(logRadiance.z, 0.0722),
  );
  let phase = henyeyGreensteinPhase(
    dot(safe_normalize(-wi), safe_normalize(wo)),
    scatter.a,
  );
  var logSource = vec3f(logLuminance);
  logSource = restirShadeAppendPositiveFactor3(
    logSource,
    max(albedo, vec3f(0.0)),
  );
  logSource = vec3f(
    restirShadeAppendPositiveFactor(logSource.x, phase),
    restirShadeAppendPositiveFactor(logSource.y, phase),
    restirShadeAppendPositiveFactor(logSource.z, phase),
  );

  let projectedCosine = abs(dot(
    safe_normalize(normal),
    safe_normalize(wo),
  ));
  if (projectedCosine <= 0.0) { return logSource; }
  let distance = pathLength / projectedCosine;
  let transmittance = homogeneousBeerTransmittanceRgb(sigmaS, distance);
  let logAttenuated = restirShadeAppendPositiveFactor3(
    logRadiance,
    transmittance,
  );
  let logScattered = restirShadeAppendPositiveFactor3(
    logSource,
    vec3f(1.0) - transmittance,
  );
  return restirShadeLogAdd3(logAttenuated, logScattered);
}

fn restirShadeDemodulateLog(
  logPhysicalRadiance: vec3f,
  albedo: vec3f,
) -> vec3f {
  return vec3f(
    restirShadeRemovePositiveFactor(logPhysicalRadiance.x, albedo.x),
    restirShadeRemovePositiveFactor(logPhysicalRadiance.y, albedo.y),
    restirShadeRemovePositiveFactor(logPhysicalRadiance.z, albedo.z),
  );
}

fn restirShadeDemodulateChannel(value: f32, albedo: f32) -> f32 {
  if (!restirShadeFinite(value) || value < 0.0 ||
      !restirShadeFinite(albedo) || !(albedo > 0.0)) {
    return 0.0;
  }
  return min(value / albedo, RESTIR_SHADE_MAX_FINITE_F32);
}

fn restirShadeDemodulate(
  physicalRadiance: vec3f,
  albedo: vec3f,
) -> vec3f {
  return vec3f(
    restirShadeDemodulateChannel(physicalRadiance.x, albedo.x),
    restirShadeDemodulateChannel(physicalRadiance.y, albedo.y),
    restirShadeDemodulateChannel(physicalRadiance.z, albedo.z),
  );
}

// DDGI fallback and RC are already integrated over direction. They retain the
// aggregate volume overload, but still own the same diffuse/material factors
// before being safely demodulated for the indirect buffer.
fn restirShadeAggregateDiffuseDemodulated(
  lighting: vec3f,
  albedo: vec3f,
  diffuseWeight: f32,
  layerTransmission: vec3f,
  scatter: vec4f,
  pathLength: f32,
  normal: vec3f,
  wo: vec3f,
) -> vec3f {
  let sigmaS = max(scatter.rgb, vec3f(0.0));
  if (all(sigmaS <= vec3f(0.0)) || pathLength <= 0.0) {
    return lighting * diffuseWeight * layerTransmission;
  }
  let physical = applyHomogeneousVolumeSingleScatter(
    lighting * albedo * diffuseWeight * layerTransmission,
    albedo,
    scatter,
    pathLength,
    normal,
    wo,
  );
  return restirShadeDemodulate(physical, albedo);
}

fn lo_emit(
  matColor:         vec4f,
  normal:           vec3f,
  isGlass:          bool,
  uv:               vec2f,
  uv1:              vec2f,
  matColorPacked:   u32,
  triIndex:         u32,
) -> vec3f {
  if (!isGlass) { return vec3f(0.0); }
  if ((ubo.stainedGlassFlags & SG_FLAG_SUN_CAUSTIC) == 0u) { return vec3f(0.0); }
  let sunDot = abs(dot(ubo.sunDirection, normal));
  if (sunDot <= 0.05) { return vec3f(0.0); }
  let trans = matColor.a;
  let texId = decodeSurfaceTextureId(matColorPacked);
  let texMod = surfaceTextureMod(uv, texId);
  // WS1 — beer is a texture now: map the triangle index to a texel.
  let beerCoord = vec2u(triIndex % BVH_BEER_TEX_WIDTH, triIndex / BVH_BEER_TEX_WIDTH);
  let beerPacked = textureLoad(bvh_beer, vec2i(beerCoord), 0).r;
  var beerAlbedo = vec3f(
    f32((beerPacked >> 24u) & 0xFFu) / 255.0,
    f32((beerPacked >> 16u) & 0xFFu) / 255.0,
    f32((beerPacked >>  8u) & 0xFFu) / 255.0,
  );
  var thicknessMapScale = 1.0;
  let thicknessMap = sampleMaterialAtlasRawAtOffset(
    triIndex,
    MATERIAL_MAP_THICKNESS_TEXEL_OFFSET,
    uv,
    uv1,
  );
  if (thicknessMap.valid != 0u) {
    thicknessMapScale = materialOpticalThicknessMapScale(
      triIndex,
      thicknessMap.value.g,
    );
  }
  beerAlbedo = applyThicknessMapToBeerTint(triIndex, uv, uv1, beerAlbedo);
  beerAlbedo = materialSpectralAttenuation(
    triIndex,
    materialOpticalThickness(triIndex) * thicknessMapScale,
    beerAlbedo,
  );
  return beerAlbedo * trans * ubo.sunIntensity * sunDot * texMod;
}

// --- Camera-visible emitters: self-emission glow on a primary hit -----------
//
// Emissive-mesh surfaces are NEE-only in walkaround (the ReSTIR-DI emitter list
// lights RECEIVERS); their own pixels render black to the camera. This returns
// the hit triangle's HDR emissive radiance Le from the canonical GI emissive
// texture so the emitter glows directly. The texel layout matches bvh_beer
// (BVH_BEER_TEX_WIDTH width; both per-triangle textures share it).
//
// No double-count: this is the emitter's OWN pixel, a different surface point
// than the receivers ReSTIR-DI shades; and lo_emit (glass Beer-Lambert) is the
// transmissive case, while packBVHEmissiveLe packs the emissive branch ONLY.
fn lo_emitterGlow(triIndex: u32) -> vec3f {
  let coord = vec2u(triIndex % BVH_BEER_TEX_WIDTH, triIndex / BVH_BEER_TEX_WIDTH);
  return textureLoad(restir_gi_bvh_emissive, vec2i(coord), 0).rgb;
}

// --- H41: Analytic point/spot NEE (ADDITIVE, separate from RIS area-emitter pool) ---
//
// Iterates the analytic_lights buffer for point and spot emitters. This term
// is SEPARATE from lo_direct (which samples the ReSTIR area-emitter reservoir).
// Adding it additively prevents PDF contamination — the two estimators address
// disjoint light sources (area emitters vs analytic point/spot), so their
// contributions sum without double-counting.
//
// Light model: inverse-square falloff with ε denominator floor to avoid
// division-by-zero at the light position. Spot falloff: smoothstep from
// cosOuter to cosInner. Point: cosInner=1, cosOuter=0 → smoothstep(0,1,x) = x²(3-2x) is not 1 at x≥1, so use cosOuter < cosInner always.
//
// Shadow: deterministic RGB alpha/material-transmission traversal (not
// stochastic — no variance per pixel, no need for reservoir denoising).
// Manifold mode delegates material transmission to its explicit path estimator.
//
// Metals receive their conductor lobe; glass receives reflection-only while
// its transmission remains owned by the bounded dielectric transport path.
${analyticLightFalloffWgsl('analytic')}

// Strategy code 2 is manifold-nee. In that mode the explicit specular-path
// estimator owns material-transmission paths, so ordinary NEE keeps alpha-only
// visibility but cannot also pass directly through a refractive boundary.
fn manifoldNeeOwnsMaterialTransmission() -> bool {
  return ubo.sunAngular.z >= 1.5;
}

// Direct illumination follows the continuous transmission mix: the opaque
// base response fades with (1-transmission), while reflection remains present
// and the complementary BTDF is evaluated by the explicit glass transport.
fn evalDirectSurfaceBrdf(
  albedo: vec3f,
  rough: f32,
  metal: f32,
  specular: vec4f,
  anisotropy: vec2f,
  anisotropyTangent: vec3f,
  anisotropyBitangent: vec3f,
  iridescence: vec4f,
  clearcoat: vec2f,
  sheen: vec4f,
  sheenRoughness: f32,
  normal: vec3f,
  clearcoatNormal: vec3f,
  wo: vec3f,
  wi: vec3f,
  transmission: f32,
  layerTransmission: vec3f,
  reflectionLayerTransmission: vec3f,
) -> vec3f {
  let mixedClosure = evalGGXReflectionWithTransmissionMix(
    albedo, rough, metal, specular.rgb, specular.a,
    anisotropy.x, anisotropy.y, iridescence,
    clearcoat.x, clearcoat.y, sheen.a, sheenRoughness, sheen.rgb,
    anisotropyTangent, anisotropyBitangent,
    normal, clearcoatNormal, wo, wi, transmission,
  );
  let reflectionClosure =
    evalGGXSpecularOnlyWithSpecularClearcoatSheenWithAnisotropyFrame(
      albedo, rough, metal, specular.rgb, specular.a,
      anisotropy.x, anisotropy.y, iridescence,
      clearcoat.x, clearcoat.y, sheen.a, sheenRoughness, sheen.rgb,
      anisotropyTangent, anisotropyBitangent,
      normal, clearcoatNormal, wo, wi,
    );
  return applyMaterialLayerTransmissionToBrdf(
    mixedClosure,
    reflectionClosure,
    layerTransmission,
    reflectionLayerTransmission,
  );
}

struct AnalyticNeeAliasDraw {
  index: u32,
  pmf: f32,
}

fn analyticNeeHashU32(seed: u32) -> u32 {
  var state = seed * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

fn analyticNeeHashToF32(seed: u32) -> f32 {
  return f32(analyticNeeHashU32(seed) >> 8u) * (1.0 / 16777216.0);
}

fn analyticNeeAliasColumn(seed: u32, count: u32) -> u32 {
  let threshold = ((0xffffffffu % count) + 1u) % count;
  var word = analyticNeeHashU32(seed);
  loop {
    if (word >= threshold) { return word % count; }
    word = analyticNeeHashU32(word ^ 0x27d4eb2du);
  }
  return 0u;
}

fn analyticNeeAliasDraw(
  count: u32,
  aliasOffset: u32,
  dims: vec2u,
  seed: u32,
) -> AnalyticNeeAliasDraw {
  let column = analyticNeeAliasColumn(seed, count);
  let coord = aliasOffset + column;
  let entry = textureLoad(analytic_lights, vec2i(i32(coord % dims.x), i32(coord / dims.x)), 0);
  let aliasIndex = bitcast<u32>(entry.y);
  let selected = select(aliasIndex, column, analyticNeeHashToF32(seed ^ 0x85ebca6bu) < entry.x);
  let selectedCoord = aliasOffset + selected;
  let selectedEntry = textureLoad(
    analytic_lights,
    vec2i(i32(selectedCoord % dims.x), i32(selectedCoord / dims.x)),
    0,
  );
  var draw: AnalyticNeeAliasDraw;
  draw.index = selected;
  draw.pmf = selectedEntry.z;
  return draw;
}

fn lo_analyticNEE(
  pos:      vec3f,
  normal:   vec3f,
  clearcoatNormal: vec3f,
  geoNormal: vec3f,
  shadowContainingMedia: MaterialShadowContainingMedia,
  albedo:   vec3f,
  rough:    f32,
  metal:    f32,
  specular: vec4f,
  anisotropy: vec2f,
  anisotropyTangent: vec3f,
  anisotropyBitangent: vec3f,
  iridescence: vec4f,
  clearcoat: vec2f,
  sheen:    vec4f,
  sheenRoughness: f32,
  wo:       vec3f,
  transmission: f32,
  layerTransmission: vec3f,
  reflectionLayerTransmission: vec3f,
  volumeScattering: vec4f,
  bulkThickness: f32,
  isMetal:  bool,
) -> vec3f {
  // Opaque materials receive the full BRDF. Glass receives its Fresnel/GGX
  // reflection families only; refracted analytic NEE is a separate estimator.
  // V28/H41 analytic-light texture layout: texel 0 is a self-describing header
  // (x = light count), followed by 4×vec4f per point/spot light. The header is
  // required because the CPU side must allocate a non-empty placeholder texture
  // for zero-light scenes; deriving count from texture dimensions would treat
  // that placeholder as a real zeroed light.
  let analyticDims = textureDimensions(analytic_lights);
  let analyticHeader = textureLoad(analytic_lights, vec2i(0, 0), 0);
  let count = u32(max(analyticHeader.x, 0.0));
  let aliasOffset = u32(max(analyticHeader.y, 0.0));
  let texelCount = analyticDims.x * analyticDims.y;
  if (
    count == 0u ||
    aliasOffset != 1u + count * 4u ||
    aliasOffset + count > texelCount
  ) { return vec3f(0.0); }
  var Lo = vec3f(0.0);
  let sampleCount = min(count, 4u);
  let posBits = bitcast<vec3u>(pos);
  let seedBase = ubo.frameSeed ^ posBits.x ^ (posBits.y * 0x9e3779b9u) ^ (posBits.z * 0x85ebca6bu);
  for (var sampleIndex = 0u; sampleIndex < sampleCount; sampleIndex++) {
    var li = sampleIndex;
    var estimatorWeight = 1.0;
    if (count > 4u) {
      let draw = analyticNeeAliasDraw(
        count,
        aliasOffset,
        analyticDims,
        seedBase ^ (sampleIndex * 0xc2b2ae35u),
      );
      if (draw.pmf <= 0.0) { continue; }
      li = draw.index;
      estimatorWeight = 1.0 / (f32(sampleCount) * draw.pmf);
    }
    let base = 1u + li * 4u;
    let light0 = textureLoad(analytic_lights, vec2i(i32(base % analyticDims.x), i32(base / analyticDims.x)), 0);
    let light1 = textureLoad(analytic_lights, vec2i(i32((base + 1u) % analyticDims.x), i32((base + 1u) / analyticDims.x)), 0);
    let light2 = textureLoad(analytic_lights, vec2i(i32((base + 2u) % analyticDims.x), i32((base + 2u) / analyticDims.x)), 0);
    let light3 = textureLoad(analytic_lights, vec2i(i32((base + 3u) % analyticDims.x), i32((base + 3u) / analyticDims.x)), 0);
    let lightPos  = light0.xyz;
    let lightLe   = light1.xyz;
    let lightDir  = light2.xyz;
    let cosInner  = light2.w;
    let cosOuter  = light3.x;
    let castShadowDisabled = light3.y > 0.5;
    let cutoffDistance = light3.z;
    let decay = light3.w;

    let toL  = lightPos - pos;
    let dist = safe_length(toL);
    if (!(dist > 0.0)) { continue; }
    let wi   = safe_normalize(toL);
    let nDotL = dot(normal, wi);
    if (nDotL <= 0.0) { continue; }

    // Spot cone attenuation. For a point (cosInner=1, cosOuter=0):
    //   cosTheta = dot(-lightDir, wi) but lightDir=(0,0,0) so we skip cone.
    //   cone = 1.0 for points (omnidirectional).
    let cone = analyticSpotConeFalloff(lightDir, wi, cosInner, cosOuter);
    if (cone <= 0.0) { continue; }

    var shadowT = vec3f(1.0);
    if (!castShadowDisabled) {
      // Shadow ray — same ownership-aware RGB traversal as lo_direct, offset
      // along the geometric normal.
      // SHADOW-01 / ALPHA-03 — DI shadow rays skip castShadow:false geometry
      // and attenuate through readable alpha-blend coverage in the material atlas;
      // transparent glass additionally applies the Beer/material tint per channel.
      shadowT = traceSceneAlphaTintTransmittanceTexturedWithContainingMedia(
        ubo.bvhMode, ubo.tlasNodeCount,

        pos + geoNormal * walkaroundRayOriginBias(), wi,
        max(0.0, dist - walkaroundRayEndMargin()), ubo.triIntersectEpsilon,
        bvh_material, BVH_MATERIAL_TEX_WIDTH, bvh_beer,
        shadowContainingMedia,
        manifoldNeeOwnsMaterialTransmission());
      if (max(max(shadowT.x, shadowT.y), shadowT.z) <= 0.0) { continue; }
    }

    // Authored range/decay falloff; default decay=2 preserves inverse-square.
    let attenuation = analyticPointSpotAttenuation(dist, cutoffDistance, decay, ubo.emitterDist2Floor);
    let layeredBrdf = evalDirectSurfaceBrdf(albedo, rough, metal, specular, anisotropy, anisotropyTangent, anisotropyBitangent, iridescence, clearcoat, sheen, sheenRoughness, normal, clearcoatNormal, wo, wi, transmission, layerTransmission, reflectionLayerTransmission);
    let unvolumedContribution = lightLe * shadowT * layeredBrdf *
      cone * attenuation * estimatorWeight;
    let contribution = applyHomogeneousVolumeSingleScatterDirectional(
      unvolumedContribution, albedo, volumeScattering, bulkThickness,
      normal, wo, wi,
    );
    // evalGGX* already includes the receiver cosine; nDotL is only a gate here.
    Lo += contribution;
  }
  return Lo;
}

// --- Direct lighting (ReSTIR DI) ---
//
// The ReSTIR target and this consumer share the same material-domain rule:
// opaque receivers use the full BRDF, while transmissive dielectrics use only
// reflection families. Their separately sampled transmission is not folded
// into this direct-light reservoir.
fn lo_direct(
  pixelIdx: u32,
  pos:      vec3f,
  normal:   vec3f,
  clearcoatNormal: vec3f,
  geoNormal: vec3f,
  shadowContainingMedia: MaterialShadowContainingMedia,
  wo:       vec3f,
  albedo:   vec3f,
  rough:    f32,
  metal:    f32,
  specular: vec4f,
  anisotropy: vec2f,
  anisotropyTangent: vec3f,
  anisotropyBitangent: vec3f,
  iridescence: vec4f,
  clearcoat: vec2f,
  sheen:    vec4f,
  sheenRoughness: f32,
  envMapIntensity: f32,
  transmission: f32,
  layerTransmission: vec3f,
  reflectionLayerTransmission: vec3f,
  volumeScattering: vec4f,
  bulkThickness: f32,
  isMetal:  bool,
  rng:      ptr<function, u32>,
) -> vec3f {
  // Glass is deliberately not an early-out: evalDirectSurfaceBrdf retains its
  // Fresnel/GGX reflection lobe without inventing an opaque diffuse lobe.
  let r = loadSpatialDI(pixelIdx);
  if (r.M == 0u || !restirShadeValidLog(r.logW)) { return vec3f(0.0); }
  let lid = r.lightId;

  // Wave 4 — ENV_SAMPLE_SENTINEL: the reservoir's winning candidate was an HDRI
  // importance-sampled direction. Decode xi → world direction and shade with the
  // env radiance × BRDF × exp2(logW). Visibility is evaluated here at the
  // target receiver and was deliberately not baked into the source reservoir,
  // so temporal/spatial shifts cannot carry stale source occlusion.
  if (lid == ENV_SAMPLE_SENTINEL) {
    if (!envHasMap()) { return vec3f(0.0); }
    let envDir = envDirFromXi(r.xi);
    let nDotL = max(0.0, dot(normal, envDir));
    if (nDotL <= 0.0) { return vec3f(0.0); }
    let envColor = walkaroundScaleEnvironmentRadiance(
      envRadiance(envDir),
      envMapIntensity,
    );
    let layeredBrdfE = evalDirectSurfaceBrdf(albedo, rough, metal, specular, anisotropy, anisotropyTangent, anisotropyBitangent, iridescence, clearcoat, sheen, sheenRoughness, normal, clearcoatNormal, wo, envDir, transmission, layerTransmission, reflectionLayerTransmission);
    let shadowTint = traceSceneAlphaTintTransmittanceTexturedWithContainingMedia(
      ubo.bvhMode, ubo.tlasNodeCount,

      pos + geoNormal * walkaroundRayOriginBias(), envDir, INFINITY,
      ubo.triIntersectEpsilon,
      bvh_material, BVH_MATERIAL_TEX_WIDTH, bvh_beer,
      shadowContainingMedia,
      manifoldNeeOwnsMaterialTransmission());
    let shadowScalar = clamp(luminance(shadowTint), 0.0, 1.0);
    if (shadowScalar <= 0.0) { return vec3f(0.0); }
    let responseLog = restirShadeDirectionalVolumeLog(
      restirShadeLogProduct3(r.logW, envColor, layeredBrdfE, shadowTint),
      albedo, volumeScattering, bulkThickness,
      normal, wo, envDir,
    );
    return restirShadeExp2Clamped3(responseLog);
  }

  if (lid >= ubo.emitterCount) { return vec3f(0.0); }
  let e  = sceneLoadEmitter(lid);
  // Consume the exact sample that won the reservoir. Re-rolling a fresh point
  // here breaks the ReSTIR identity between candidate p̂, finalization p̂, and
  // shaded contribution for large close emitters.
  let ls = sampleEmitterPoint(e, r.xi);
  let toL = ls.pos - pos;
  let dist = safe_length(toL);
  if (!(dist > 0.0)) { return vec3f(0.0); }
  let wi    = safe_normalize(toL);
  let nDotL = max(0.0, dot(normal, wi));
  let nlDotL = emitterTriCosineTowardReceiver(e, -wi);
  if (nDotL <= 0.0 || nlDotL <= 0.0) { return vec3f(0.0); }
  if (!emitterTriCastShadowDisabled(e)) {
    // The ownership-aware RGB walk carries alpha, material tint, and Beer
    // attenuation unless manifold NEE explicitly owns material transmission.
    // WS1 — offset the shadow-ray origin along the GEOMETRIC normal (the smooth
    // shading normal can dip below the surface near silhouettes → self-hit).
    // SHADOW-01 / ALPHA-03 — DI shadow rays skip castShadow:false geometry,
    // apply atlas-backed texture-alpha cutouts, and carry glass Beer tint through
    // the material atlas. This target tint is applied exactly once; the
    // reservoir stores no source-receiver visibility term.
    let shadowTint = traceSceneAlphaTintTransmittanceTexturedWithContainingMedia(
      ubo.bvhMode, ubo.tlasNodeCount,

      pos + geoNormal * walkaroundRayOriginBias(), wi,
      max(0.0, dist - walkaroundRayEndMargin()), ubo.triIntersectEpsilon,
      bvh_material, BVH_MATERIAL_TEX_WIDTH, bvh_beer,
      shadowContainingMedia,
      manifoldNeeOwnsMaterialTransmission());
    let shadowScalar = clamp(luminance(shadowTint), 0.0, 1.0);
    if (shadowScalar <= 0.0) { return vec3f(0.0); }
    let G    = emitterGeometry(nlDotL, dist * dist, ubo.emitterDist2Floor);
    let layeredBrdf = evalDirectSurfaceBrdf(albedo, rough, metal, specular, anisotropy, anisotropyTangent, anisotropyBitangent, iridescence, clearcoat, sheen, sheenRoughness, normal, clearcoatNormal, wo, wi, transmission, layerTransmission, reflectionLayerTransmission);
    let Le = sampleEmitterLeAtXi(e, r.xi);
    let geometryLogW = restirShadeAppendPositiveFactor(r.logW, G);
    let responseLog = restirShadeDirectionalVolumeLog(
      restirShadeLogProduct3(
        geometryLogW, Le, layeredBrdf, shadowTint,
      ),
      albedo, volumeScattering, bulkThickness,
      normal, wo, wi,
    );
    return restirShadeExp2Clamped3(responseLog);
  }
  let G    = emitterGeometry(nlDotL, dist * dist, ubo.emitterDist2Floor);
  let layeredBrdf = evalDirectSurfaceBrdf(albedo, rough, metal, specular, anisotropy, anisotropyTangent, anisotropyBitangent, iridescence, clearcoat, sheen, sheenRoughness, normal, clearcoatNormal, wo, wi, transmission, layerTransmission, reflectionLayerTransmission);
  let Le = sampleEmitterLeAtXi(e, r.xi);
  let geometryLogW = restirShadeAppendPositiveFactor(r.logW, G);
  let responseLog = restirShadeDirectionalVolumeLog(
    restirShadeLogProduct3(
      geometryLogW, Le, layeredBrdf, vec3f(1.0),
    ),
    albedo, volumeScattering, bulkThickness,
    normal, wo, wi,
  );
  return restirShadeExp2Clamped3(responseLog);
}

// --- Direct sun NEE (default-on, item 4 plan/residue-closure-plan-2026-06-10.md) ----
//
// Casts one deterministic shadow ray toward the sun direction and evaluates the
// full opaque BRDF or the glass reflection families when the sun lane is live
// (sunIntensity > eps). Transmission and stained-glass caustics stay separate.
//
// ── No-double-count argument ───────────────────────────────────────────────────
// DDGI probes evaluate evalSunLight at the PROBE BOUNCE SURFACE (walls, floor,
// etc.), store Lo_probe = (baseColor/π) · (direct_sun_at_bounce + indirect), and
// the receiver reads that atlas to get INDIRECT irradiance = sun → wall → receiver
// (the lo_indirect term). lo_sunNEE adds sun → receiver DIRECT — a disjoint path.
//
// Concretely: every probe ray that hits a wall records the sun's contribution at
// THAT wall; the blend pass averages those records into the irradiance atlas; the
// shade pass draws from the atlas as the per-surface INDIRECT term (lo_indirect).
// lo_sunNEE below traces the ray from the PRIMARY-HIT SURFACE to the sun, which
// is a DIFFERENT surface and a DIFFERENT ray path. The two estimates do not share
// any common path segment, so they are strictly non-overlapping:
//
//   DDGI indirect: sun → bounce_wall → (probe stores E/π) → receiver reads E·albedo
//   lo_sunNEE:     sun → (shadow test along sunDir) → receiver BRDF evaluation
//
// The DDGI term carries the first-bounce INDIRECT contribution. lo_sunNEE is the
// DIRECT one-bounce term at the receiver. A double-count would only arise if the
// probe also stored "direct sun AT THE RECEIVER" in its atlas, which it cannot —
// probes are placed at grid positions, not at primary-hit surfaces.
//
// ── lo_sg_caustic relationship ────────────────────────────────────────────────
// lo_sg_caustic (SG_FLAG_SUN_CAUSTIC, stained-glass flag) is an opt-in stylized
// enhancement of this same direct-sun family. The default lo_sunNEE path owns
// physically tinted alpha/transmission visibility for every eligible receiver;
// the stained-glass lane adds its authored causticBoost treatment only when the
// host explicitly enables it. With the flag off, lo_sunNEE is the sole direct-sun
// term. With it on, the caustic lane is intentionally additive rather than an
// independent unbiased estimator.
fn lo_sunNEE(
  gid:       vec2u,
  pos:       vec3f,
  normal:    vec3f,
  clearcoatNormal: vec3f,
  geoNormal: vec3f,
  shadowContainingMedia: MaterialShadowContainingMedia,
  albedo:    vec3f,
  rough:     f32,
  metal:     f32,
  specular:  vec4f,
  anisotropy: vec2f,
  anisotropyTangent: vec3f,
  anisotropyBitangent: vec3f,
  iridescence: vec4f,
  clearcoat: vec2f,
  sheen:     vec4f,
  sheenRoughness: f32,
  wo:        vec3f,
  transmission: f32,
  layerTransmission: vec3f,
  reflectionLayerTransmission: vec3f,
  volumeScattering: vec4f,
  bulkThickness: f32,
) -> vec3f {
  // Skip if sun is absent (no directional emitter or intensity essentially zero).
  if (!(ubo.sunIntensity > 0.0)) { return vec3f(0.0); }
  // Sun direction: ubo.sunDirection is the unit vector from world origin toward
  // the sun. Apply the same authored soft-sun angular spread as lo_sg_caustic.
  // Frame-scramble the per-pixel disk sample so temporal accumulation integrates
  // the finite sun instead of converging to one permanently frozen direction.
  let sunBase = ubo.sunDirection;
  let sunAngularRadius = max(ubo.sunAngular.x, 0.0);
  let xi = pixelHash2(gid, ubo.frameSeed ^ 0x53474341u);
  let upRef = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), abs(sunBase.y) < 0.99);
  let tan = safe_normalize(cross(upRef, sunBase));
  let bit = cross(sunBase, tan);
  let r2  = sunAngularRadius * sqrt(xi.x);
  let phi = 6.2831853 * xi.y;
  let toSun = safe_normalize(sunBase + tan * (r2 * cos(phi)) + bit * (r2 * sin(phi)));

  let nDotSun = dot(normal, toSun);
  if (nDotSun <= 0.0) { return vec3f(0.0); }

  // Shadow ray — offset along geometric normal (same pattern as lo_direct / lo_analyticNEE).
  // Generic direct sun now uses the same RGB alpha/transmission/thickness/Beer
  // visibility as analytic NEE and transparent OIT. The stained-glass caustic
  // term remains an optional boosted artistic estimator layered on top.
  // SHADOW-01 / ALPHA-03 — DI shadow rays skip castShadow:false geometry and
  // attenuate through readable alpha-blend coverage in the material atlas.
  var sunShadowT = vec3f(1.0);
  if ((ubo.stainedGlassFlags & SHADE_FLAG_DIRECT_SUN_SHADOW_DISABLED) == 0u) {
    sunShadowT = traceSceneAlphaTintTransmittanceTexturedWithContainingMedia(
      ubo.bvhMode, ubo.tlasNodeCount,

      pos + geoNormal * walkaroundRayOriginBias(), toSun, INFINITY,
      ubo.triIntersectEpsilon,
      bvh_material, BVH_MATERIAL_TEX_WIDTH, bvh_beer,
      shadowContainingMedia,
      manifoldNeeOwnsMaterialTransmission());
    if (max(max(sunShadowT.x, sunShadowT.y), sunShadowT.z) <= 0.0) { return vec3f(0.0); }
  }

  // Opaque full-BRDF / glass reflection-only evaluation. The helper already
  // folds in nDotSun as its NdotL term.
  let layeredBrdf = evalDirectSurfaceBrdf(albedo, rough, metal, specular, anisotropy, anisotropyTangent, anisotropyBitangent, iridescence, clearcoat, sheen, sheenRoughness, normal, clearcoatNormal, wo, toSun, transmission, layerTransmission, reflectionLayerTransmission);
  let unvolumedContribution = vec3f(ubo.sunIntensity) *
    sunShadowT * layeredBrdf;
  let contribution = applyHomogeneousVolumeSingleScatterDirectional(
    unvolumedContribution, albedo, volumeScattering, bulkThickness,
    normal, wo, toSun,
  );
  // Sun irradiance: ubo.sunIntensity is the directional emitter intensity.
  // No distance falloff — directional lights have infinite distance.
  return contribution;
}

// T5 — the sun-caustic + sky-aperture stained-glass-specific lighting terms
// were extracted into stainedGlassShade.wgsl.ts (lo_sg_caustic /
// lo_sg_aperture), opt-in behind ubo.stainedGlassFlags. shade no longer
// carries stained-glass knowledge; it just calls the two helpers below in
// the per-term composition. SHADE_MODULE.requires lists stainedGlassShade
// so the composer emits those bodies ahead of SHADE_WGSL.

// --- Indirect lighting (Sprint 16 — ReSTIR-GI one-bounce resampling) ---
//
// The trilinear DDGI atlas read had visible cell-grid splotches on
// smooth walls (structural single-bounce limitation). ReSTIR-GI runs
// a half-res RIS pass that picks ONE probe-direction sample per pixel
// by importance, then resamples spatially+temporally (Sprints 17-18).
// The result is per-pixel screen-space — the probe grid stops being
// the per-pixel basis, so cell artefacts go away.
//
// Lo_indirect_lighting = Lo * exp2(logW) * cos(N, wi) * INV_PI
//   - Lo, logW from the GI reservoir (half-res; bilinear-blend across 4 cells)
//   - cos × INV_PI is the receiver Lambertian BRDF response
//   - each directional sample first receives albedo, the continuous
//     metal/transmission diffuse share, face-layer transfer, and volume response;
//     safe demodulation then removes albedo for the Schied 2017 §4.1 denoiser
//     contract, and indirectCombine re-multiplies it after filtering.
// Pure conductors or fully transmissive receivers naturally produce zero diffuse
// weight. Intermediate materials retain their physical continuous share; native
// camera-prefix transmission and specular GI remain separate lanes.
//
// Sprint 18 follow-up — bilinear blend across 4 surrounding half-res
// reservoirs.  The original nearest-neighbour read halfPx = gid/2u made
// every 2x2 full-res quad share one chosen sample; adjacent quads picked
// different random samples, so the indirect signal had a sharp 2-pixel
// discontinuity at every quad boundary.  risGi re-rolls samples each frame,
// so the discontinuity pattern shifted every frame and the temporal
// accumulator could not converge to a fixed point.  Blending 4 neighbours
// with bilinear weights in the producer's center-aligned reservoir coordinate
// eliminates the quad grid without shifting the field by half a cell.
fn giReservoirVisibility(g: ReservoirGI) -> f32 {
  if (g.historyEpoch != bitcast<u32>(ubo.sunAngular.y)) { return 0.0; }
  return clamp(g.sampleVisibility, 0.0, 1.0);
}

fn giReservoirDirectionVector(g: ReservoirGI, receiverPosition: vec3f) -> vec3f {
  if (g.sampleKind == GI_SAMPLE_ENVIRONMENT) {
    return g.wi_recon;
  }
  return g.xs - receiverPosition;
}

fn coarseGiDomainCompatible(
  g: ReservoirGI,
  receiverPosition: vec3f,
  receiverNormal: vec3f,
  receiverMaterialKey: u32,
) -> bool {
  return
    g.receiverMaterialKey == receiverMaterialKey &&
    dot(receiverNormal, g.nv) >= ubo.restirGiSpatialNormalDotMin &&
    abs(dot(receiverPosition - g.xv, g.nv))
      <= ubo.restirGiSpatialCoplanarTol;
}

fn giReservoirAtNativeReceiver(g: ReservoirGI, receiverPosition: vec3f) -> bool {
  // RIS and shade use the same primary-ray generator and intersection path for
  // the center pixel. Exact equality identifies the only receiver at which an
  // angular/NRC Lo was actually evaluated; a geometric tolerance would still
  // admit a different thin-film angle or volume path length.
  return all(receiverPosition == g.xv);
}

fn giReservoirRequiresNativeReceiver(g: ReservoirGI) -> bool {
  return (g.sampleFlags & (
    GI_SAMPLE_FLAG_LOCAL_TECHNIQUE |
    GI_SAMPLE_FLAG_LOCAL_ESTIMATOR
  )) != 0u;
}
fn lo_indirect(
  gid:     vec2u,
  dims:    vec2u,
  pos:     vec3f,
  normal:  vec3f,
  geoNormal: vec3f,
  wo:      vec3f,
  albedo:  vec3f,
  layerTransmission: vec3f,
  volumeScattering: vec4f,
  bulkThickness: f32,
  receiverMaterialKey: u32,
  envMapIntensity: f32,
  transmission: f32,
  metal:   f32,
) -> vec3f {
  var Lo_indirect = vec3f(0.0);
  var indirectLogSum = vec3f(RESTIR_SHADE_LOG_ZERO);
  let diffuseWeight = (1.0 - clamp(metal, 0.0, 1.0)) *
    (1.0 - clamp(transmission, 0.0, 1.0));
${giBilinearWeightsWgsl()}
  var totalW: f32 = 0.0;
  // Confidence accumulator — bilinear-weighted ReSTIR-GI sample count over the
  // same 4 half-res reservoirs that build Lo_indirect. The reservoir M is the
  // effective number of resampled candidates the temporal+spatial passes have
  // integrated into this pixel (Bitterli 2020 / Ouyang 2021 ReSTIR-GI). A
  // higher M ⇒ lower-variance, more-trustworthy estimate. We weight each
  // reservoir's M by its bilinear contribution so the per-pixel confidence
  // matches the radiance blend exactly.
  var Maccum: f32 = 0.0;
  for (var k: u32 = 0u; k < 4u; k = k + 1u) {
${giBilinearCornerSelectWgsl()}
    let giIdx = hy * halfDims.x + hx;
    let g = loadReservoirGI_rw(giIdx);
    if (g.M == 0u || !restirShadeValidLog(g.logW)) { continue; }
    if (g.historyEpoch != bitcast<u32>(ubo.sunAngular.y)) { continue; }
    if (g.prefixVertexCount != GI_PREFIX_RECONNECTABLE) { continue; }
    if (!coarseGiDomainCompatible(g, pos, normal, receiverMaterialKey)) { continue; }
    let atNativeReceiver = giReservoirAtNativeReceiver(g, pos);
    if (giReservoirRequiresNativeReceiver(g) && !atNativeReceiver) { continue; }
    if (g.sampleKind == GI_SAMPLE_SURFACE
     && !grisSurfaceSuffixReceiverSupported(pos, g.xs, g.ns)) { continue; }
    var logDomainJacobian = 0.0;
    var grisTint = vec3f(giReservoirVisibility(g));
    let recastTint = !atNativeReceiver ||
      (g.sampleFlags & GI_SAMPLE_FLAG_RECAST_TINT) != 0u;
    if (!atNativeReceiver) {
      logDomainJacobian = grisLogDomainToCanonicalJacobian(
        g.xv,
        pos,
        g.sampleKind,
        g.xs,
        g.ns,
      );
    }
    if (recastTint) {
      grisTint = grisProxyTintAt(
        pos,
        normal,
        geoNormal,
        g.sampleKind,
        g.xs,
        g.wi_recon,
      );
    }
    if (!restirShadeValidLog(logDomainJacobian)) { continue; }
    if (max(grisTint.r, max(grisTint.g, grisTint.b)) <= 0.0) { continue; }
    let toS = giReservoirDirectionVector(g, pos);
    let distS = safe_length(toS);
    if (!(distS > 0.0)) { continue; }
    let wi = safe_normalize(toS);
    let cosTheta = max(0.0, dot(normal, wi));
    var receiverLo = g.Lo;
    if (g.sampleKind == GI_SAMPLE_ENVIRONMENT) {
      receiverLo = walkaroundScaleEnvironmentRadiance(
        receiverLo,
        envMapIntensity,
      );
    }
    // Build the physical diffuse response for this candidate before volume.
    // The directional operator is linear for non-negative radiance, so each
    // distinct bilinear wi must be processed before accumulation. Demodulation
    // happens afterward solely to preserve the denoiser buffer contract.
    var contributionLogW = restirShadeAppendLogFactor(g.logW, logDomainJacobian);
    contributionLogW = restirShadeAppendPositiveFactor(contributionLogW, INV_PI);
    contributionLogW = restirShadeAppendPositiveFactor(contributionLogW, cosTheta);
    contributionLogW = restirShadeAppendPositiveFactor(contributionLogW, bw);
    contributionLogW = restirShadeAppendPositiveFactor(
      contributionLogW,
      diffuseWeight,
    );
    var physicalDiffuseLog = restirShadeLogProduct3(
      contributionLogW,
      receiverLo,
      grisTint,
      albedo,
    );
    physicalDiffuseLog = restirShadeAppendPositiveFactor3(
      physicalDiffuseLog,
      layerTransmission,
    );
    physicalDiffuseLog = restirShadeDirectionalVolumeLog(
      physicalDiffuseLog,
      albedo,
      volumeScattering,
      bulkThickness,
      normal,
      wo,
      wi,
    );
    let demodulatedDiffuseLog = restirShadeDemodulateLog(
      physicalDiffuseLog,
      albedo,
    );
    indirectLogSum = restirShadeLogAdd3(
      indirectLogSum,
      demodulatedDiffuseLog,
    );
    Maccum = Maccum + f32(g.M) * bw;
    totalW = totalW + bw;
  }
  // Effective per-pixel ReSTIR-GI sample count (bilinear-averaged M). 0 when no
  // valid reservoir contributed — that pixel has *no* ReSTIR estimate, so the
  // receiver-local DDGI fallback below becomes Lo_indirect. RC may then blend
  // with that fallback when it is enabled and carries energy.
  var Meff: f32 = 0.0;
  if (totalW > 0.0) {
    Lo_indirect = restirShadeExp2Clamped3(
      restirShadeNormaliseLogSum(indirectLogSum, totalW),
    );
    Meff = Maccum / totalW;
  } else {
    // Deterministic discontinuity/local-suffix fallback: never transplant a
    // source-domain reservoir normalization or angular Lo across receivers.
    // The DDGI atlas is receiver-local and remains well-defined at every scale.
    Lo_indirect = restirShadeAggregateDiffuseDemodulated(
      sampleDDGIAtPoint(pos, normal) * INV_PI,
      albedo,
      diffuseWeight,
      layerTransmission,
      volumeScattering,
      bulkThickness,
      normal,
      wo,
    );
  }
  // W8 Phase 3 — confidence-ratio (balance-heuristic) composition with the
  // Sannikov 2023 Radiance Cascades cascade-0 estimate. Both estimators
  // estimate the SAME diffuse-indirect signal. Their reliability-weighted convex
  // blend is a deliberate bias/variance heuristic (RC itself is biased), chosen
  // per pixel instead of by a single host scalar.
  //
  // Confidence proxies (both ∈ [0,1]):
  //   c_restir = m            — ReSTIR-GI's normalised effective sample count
  //                             m = clamp(Meff / restirGiMClamp, 0, 1). The
  //                             temporal M-clamp is the host's "fully
  //                             converged" reference; ReSTIR variance falls
  //                             ~1/M, so m is a monotone reliability proxy
  //                             (NOT W — a high W usually means a low p̂ /
  //                             rare-sample spike, i.e. *less* reliable).
  //   c_rc = rcWeight·(1 - m) — RC is a low-variance but biased deterministic
  //                             probe integrator with no per-pixel sample
  //                             count, so its reliability is a fixed host
  //                             PRIOR (rcWeight) gated by how *unreliable*
  //                             ReSTIR is here (1 - m). When ReSTIR is well
  //                             converged (m→1) RC's weight fades out; on a
  //                             fresh disocclusion (m→0) RC's stable estimate
  //                             fills in. rcWeight stays the global RC trust
  //                             knob and the disabled-path off-switch.
  //
  // Balance heuristic: w_rc = c_rc / (c_rc + c_restir), w_restir = 1 - w_rc.
  // Degenerate guard: when neither estimator is confident (c_rc + c_restir ≈ 0,
  // i.e. no valid reservoir AND rcWeight 0) we force w_restir = 1. In that
  // case Lo_indirect is the receiver-local DDGI fallback computed above.
  //
  // rc-disabled bit-identity: the host binds an all-zero rcParams placeholder
  // when RC is off (DDGIBindingState.setRCInputs(null)), so rcParams.enabled==0
  // ⇒ sampleCascadeC0 returns exactly vec3f(0), AND rcWeight==0.0 ⇒ c_rc==0 ⇒
  // w_rc==0, w_restir==1.0 exactly ⇒ result == Lo_indirect, byte-for-byte
  // identical to the pre-Phase-3 path.
  let Lo_rc = sampleCascadeC0(pos, normal);
  let Lo_rcDemodulated = restirShadeAggregateDiffuseDemodulated(
    Lo_rc,
    albedo,
    diffuseWeight,
    layerTransmission,
    volumeScattering,
    bulkThickness,
    normal,
    wo,
  );
  let m = clamp(Meff / f32(max(ubo.restirGiMClamp, 1u)), 0.0, 1.0);
  let cRestir = m;
  // RC-has-energy gate (algorithm-combination-fitness fix, 2026-06-07):
  // the confidence MIS hands RC weight rcWeight*(1-m) purely from the host
  // toggle, with no check that RC's cascades actually carry radiance. When RC
  // is disabled (rcWeight=0 ⇒ Lo_rc=0), or a cascade is empty for a given
  // pixel, a non-zero weight would REPLACE the (correct) ReSTIR-GI estimate
  // with RC's zero → black indirect. Gating cRc on the already-computed Lo_rc
  // closes that: an empty cascade forces cRc=0 ⇒ wRestirGi=1 ⇒ ReSTIR-GI keeps
  // full weight. (RC's probe cast DOES sample sun + emissive geometry + env +
  // rect-area emitter NEE since 1e893fa — see walkaround-rc probeRayCast; the
  // gate guards RC-off / empty-cascade pixels, NOT a light-model mismatch.)
  // Bit-identity preserved both ways — RC off ⇒ rcWeight=0 AND Lo_rc=0, RC
  // on-with-energy ⇒ the gate is 1.0 and the blend is unchanged.
  let rcHasEnergy = max(Lo_rc.r, max(Lo_rc.g, Lo_rc.b)) > 0.0;
  let cRc = clamp(rcParams.rcWeight, 0.0, 1.0) * (1.0 - m) * select(0.0, 1.0, rcHasEnergy);
  let cSum = cRestir + cRc;
  // max() in the denominator keeps the (always-evaluated) select arm finite —
  // no inf/NaN to leak even though select discards it when cSum ≈ 0.
  var wRc = 0.0;
  if (cSum > 0.0) { wRc = cRc / cSum; }
  let wRestirGi = 1.0 - wRc;
  // Every lane above already owns the continuous metal/transmission diffuse
  // share, layer transfer, volume response, and safe demodulation exactly once.
  return wRestirGi * Lo_indirect + wRc * Lo_rcDemodulated;
}

// Camera-prefix glass transport is evaluated at the exact full-resolution
// primary pixel and consumed immediately. It never enters the half-resolution
// GI reservoir or any temporal/spatial/bilinear reuse path.
${NATIVE_GLASS_GI_WGSL}

// --- B1: Glossy/metal SPECULAR indirect (ReSTIR-GI sample × GGX specular lobe) -
//
// The ReSTIR-GI reservoir is a DIFFUSE-irradiance cache: its candidates are
// cosine-hemisphere sampled and its target p̂ = luminance(Lo)·cosθ·INV_PI is the
// Lambertian receiver response (unchanged by B1 — preserving GRIS reuse
// correctness + the diffuse-default invariant). lo_indirect consumes that
// demodulated-diffuse channel.
//
// For glossy/metal receivers the diffuse lobe is absent (metal) or minor
// (glossy); their indirect is a SPECULAR reflection of the same reconnection
// radiance Lo arriving from xs. This term evaluates the GGX specular lobe of the
// receiver material against that stored sample (wi = normalize(xs − pos)) and
// returns it as UN-demodulated radiance — so it joins the DIRECT channel (it is
// NOT proportional to the diffuse albedo, so it must bypass indirectCombine's
// albedo re-modulation; metals carry their reflectance tint in the specular F0).
//
// Consistency: this reuses the SAME reservoir sample (xs, Lo, logW) that the
// diffuse target selected — a deterministic BSDF re-weighting of a chosen
// sample (the same pattern lo_emit / lo_emitterGlow use), NOT a second
// estimator. p-hat remains an importance target; this final physical integrand
// owns its exact reflection layer and directional volume response. Single nearest GI reservoir
// (no bilinear blend) — the specular lobe is higher-frequency, so the half-res
// blur of the diffuse path is undesirable here.
//
// Gate: mirrors restir_gi_receiver_has_specular_lobes so the GI producer's
// receiver-lobe p-hat and the shade-side consumer stay in lock-step. A
// default-diffuse scene (rough 0.85, metal 0, default specular controls) gets
// EXACTLY zero specular indirect — preserving the diffuse-default invariant
// byte-for-byte.
const SPEC_GI_ROUGH_MAX: f32 = 0.6;
fn lo_indirectSpecular(
  gid:    vec2u,
  dims:   vec2u,
  pos:    vec3f,
  normal: vec3f,
  geoNormal: vec3f,
  clearcoatNormal: vec3f,
  wo:     vec3f,
  albedo: vec3f,
  rough:  f32,
  metal:  f32,
  specular: vec4f,
  anisotropy: vec2f,
  anisotropyTangent: vec3f,
  anisotropyBitangent: vec3f,
  iridescence: vec4f,
  clearcoat: vec2f,
  sheen: vec4f,
  sheenRoughness: f32,
  receiverMaterialKey: u32,
  envMapIntensity: f32,
  transmission: f32,
  reflectionLayerTransmission: vec3f,
  volumeScattering: vec4f,
  bulkThickness: f32,
) -> vec3f {
  let specularDelta = max(
    max(abs(specular.r - 1.0), abs(specular.g - 1.0)),
    max(abs(specular.b - 1.0), abs(specular.a - 1.0)),
  );
  if (transmission <= 0.0 && metal <= 0.0 && rough >= SPEC_GI_ROUGH_MAX && specularDelta <= 0.0 && abs(anisotropy.x) <= 0.0 && clearcoat.x <= 0.0 && sheen.a <= 0.0 && iridescence.x <= 0.0) { return vec3f(0.0); }
  let halfDims = restirGiDimensions();
  let giCoord = restirGiCoordForFullPixel(gid);
  let hx = giCoord.x;
  let hy = giCoord.y;
  let giIdx = hy * halfDims.x + hx;
  let g = loadReservoirGI_rw(giIdx);
  if (g.M == 0u || !restirShadeValidLog(g.logW)) { return vec3f(0.0); }
  if (g.historyEpoch != bitcast<u32>(ubo.sunAngular.y)) {
    return vec3f(0.0);
  }
  if (g.prefixVertexCount != GI_PREFIX_RECONNECTABLE) { return vec3f(0.0); }
  if (!coarseGiDomainCompatible(g, pos, normal, receiverMaterialKey)) { return vec3f(0.0); }
  let atNativeReceiver = giReservoirAtNativeReceiver(g, pos);
  if (giReservoirRequiresNativeReceiver(g) && !atNativeReceiver) {
    return vec3f(0.0);
  }
  if (g.sampleKind == GI_SAMPLE_SURFACE
   && !grisSurfaceSuffixReceiverSupported(pos, g.xs, g.ns)) {
    return vec3f(0.0);
  }
  var logDomainJacobian = 0.0;
  var grisTint = vec3f(giReservoirVisibility(g));
  let recastTint = !atNativeReceiver ||
    (g.sampleFlags & GI_SAMPLE_FLAG_RECAST_TINT) != 0u;
  if (!atNativeReceiver) {
    logDomainJacobian = grisLogDomainToCanonicalJacobian(
      g.xv,
      pos,
      g.sampleKind,
      g.xs,
      g.ns,
    );
  }
  if (recastTint) {
    grisTint = grisProxyTintAt(
      pos,
      normal,
      geoNormal,
      g.sampleKind,
      g.xs,
      g.wi_recon,
    );
  }
  if (!restirShadeValidLog(logDomainJacobian)) { return vec3f(0.0); }
  if (max(grisTint.r, max(grisTint.g, grisTint.b)) <= 0.0) {
    return vec3f(0.0);
  }
  let toS = giReservoirDirectionVector(g, pos);
  let distS = safe_length(toS);
  if (!(distS > 0.0)) { return vec3f(0.0); }
  let wi = safe_normalize(toS);
  var receiverLo = g.Lo;
  if (g.sampleKind == GI_SAMPLE_ENVIRONMENT) {
    receiverLo = walkaroundScaleEnvironmentRadiance(
      receiverLo,
      envMapIntensity,
    );
  }
  // The specular-only evaluator already includes the NdotL cosine + conductor F0.
  let specBrdf = evalGGXSpecularOnlyWithSpecularClearcoatSheenWithAnisotropyFrame(albedo, rough, metal, specular.rgb, specular.a, anisotropy.x, anisotropy.y, iridescence, clearcoat.x, clearcoat.y, sheen.a, sheenRoughness, sheen.rgb, anisotropyTangent, anisotropyBitangent, normal, clearcoatNormal, wo, wi);
  let contributionLogW = restirShadeAppendLogFactor(
    g.logW,
    logDomainJacobian,
  );
  var physicalSpecularLog = restirShadeLogProduct3(
    contributionLogW,
    receiverLo,
    specBrdf,
    grisTint,
  );
  physicalSpecularLog = restirShadeAppendPositiveFactor3(
    physicalSpecularLog,
    reflectionLayerTransmission,
  );
  physicalSpecularLog = restirShadeDirectionalVolumeLog(
    physicalSpecularLog,
    albedo,
    volumeScattering,
    bulkThickness,
    normal,
    wo,
    wi,
  );
  return restirShadeExp2Clamped3(physicalSpecularLog);
}`;
