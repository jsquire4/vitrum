/**
 * Sprint 16 — ReSTIR-GI initial-candidate RIS pass.
 *
 * Reference: Majercik et al. 2021, "Dynamic Diffuse Global Illumination
 * Resampling," SIGGRAPH 2021, §4.2 (initial-sample RIS).
 *
 * Per-pixel:
 *   1. Re-cast primary ray. Misses produce an empty reservoir; glass walks up
 *      to four dielectric interfaces to the first diffuse receiver, and
 *      rich receivers keep their material-aware lobe target below.
 *   2. RIS over M_GI = 8 candidates. Each candidate samples a
 *      cosine-weighted hemisphere direction; the reconnection vertex
 *      is the first BVH hit along that direction (or sky).
 *      Outgoing radiance Lo at the reconnection vertex is computed by
 *      sampling the DDGI irradiance atlas, then applying the hit surface's
 *      mapped material response: diffuse albedo / π for ordinary suffixes,
 *      or the extension-aware GGX/clearcoat/sheen proxy for rich suffixes.
 *   3. p̂ = luminance(receiver contribution). Diffuse defaults are
 *      luminance(Lo) × cos(N_visible, wi) × INV_PI; rich receivers add the
 *      glossy/clearcoat/sheen lobes that shade will consume.
 *      pdf_source = the candidate's source pdf. Without path guiding this is
 *        the pure cosine-hemisphere pdf cos/π, so diffuse-default
 *        w_i = p̂/pdf = luminance(Lo)
 *        (the cosθ cancels). With PPG guided sampling (ubo.ppgEnabled == 1) a
 *        Bernoulli(α) chooses guided-dTree vs cosine sampling, and the source
 *        pdf becomes the DEFENSIVE MIXTURE
 *          p_src = α·p_guide(wi) + (1−α)·cos/π        (Müller 2017 §3.4)
 *        evaluated for whichever wi was drawn; the explicit weight is then
 *        w_i = p̂ / p_src. α = 0 (PPG off) still reduces to luminance(Lo) for
 *        default diffuse receivers, while rich receivers keep their lobe target.
 *   4. The stochastic textured first-hit walk owns native segment occupancy.
 *      Its realized candidate therefore uses visibility 1; any partial-blend
 *      decision makes the accumulated native estimator identity-only for
 *      reuse. Shifted target evaluation traces expected tinted visibility.
 *   5. Word 11 stores H=log2(W_uncapped*p̂_selected); word 7 stores only the
 *      explicit firefly-cap-limited log2(W) used by final shading.
 *
 * Half-resolution: dispatches W/2 × H/2 invocations. The visible point
 * is the center of each 2×2 quad in full-res coords. The shade pass
 * reconstructs full-res indirect via reservoir read at gid.xy / 2.
 *
 * Bindings:
 *   group(0) — compact GI frame group (reservoir only; the pass re-casts its
 *              primary ray and therefore does not bind the raster G-buffer)
 *   group(1) — scene (BVH + emitters; reuse existing layout)
 *   group(2) — ubo (camera matrices, frameSeed, aoFullTexture)
 *   group(3) — hybrid (DDGI atlas + sampler + grid params at bindings 0-3;
 *              RC cascade-0 + params at 4-5; W9 PPG sTree/dTree/dTreeOffsets
 *              storage buffers at 6-8, declared by the `ppgPdf` module and
 *              read only when ubo.ppgEnabled == 1)
 *   The GI reservoir buffer is bound as @group(0) @binding(11), added
 *   to the frame BGL by the Sprint 16 pipeline machinery.
 */

import type { WgslModule } from '../pipeline/wgslComposer.js';
import { reservoirGiAccessorsWgsl } from './reservoirGi.wgsl.js';

export const RIS_GI_WGSL = /* wgsl */ `

@group(0) @binding(11) var<storage, read_write> reservoirGiCurrent: array<u32>;

${reservoirGiAccessorsWgsl({ storeReadWriteBinding: 'reservoirGiCurrent' })}

@group(1) @binding(5) var bvh_beer: texture_2d<u32>;
// WS1 (2026-05-29) — bvh_normal is declared by materialAtlas.wgsl so alpha
// cutout traversal and GI shading share the same UV1/normal source.
// B1-ior-per-tri (2026-06-10) — per-triangle roughness+metalness+IOR texture.
// Declared here so the glass-walk Snell solve can decode per-tri IOR via decodeIor().
// Layout: bits[31:24]=rough×255, bits[23:16]=metal×255, bits[15:8]=ior_quantized.
// IOR decode: 1.0 + ((byte - 1) / 254) * 2.0; byte 0 is the infinite-IOR sentinel.
@group(1) @binding(14) var bvh_material: texture_2d<u32>;

@group(2) @binding(0) var<uniform> ubo: WalkaroundUBO;
// Sprint 9 — adaptive sampling tier (r32uint, full-res). 1 = low variance,
// 2 = medium, 4 = high. Read at the centre of each half-res 2×2 quad to
// scale the RIS candidate count: high-variance pixels get more candidates
// where they're needed, low-variance pixels save the compute.
@group(2) @binding(2) var gi_tier: texture_2d<u32>;

// D5.1+D5.2: DDGIGridUBO struct, @group(3) @binding(3) ddgiGrid UBO, and
// sampleDDGIAtPoint are now provided by the shared ddgiGridUbo module.
@group(3) @binding(0) var ddgiIrradiance: texture_2d<f32>;
@group(3) @binding(1) var ddgiVisibility: texture_2d<f32>;

// Base RIS-GI candidate count. Scaled per pixel by adaptive-sampling tier:
// tier=1 → M_GI_eff = 4; tier=2 → 8 (default); tier=4 → 16.
const M_GI_BASE: u32 = 8u;

const RESTIR_GI_SUFFIX_MAX_INTERFACES: u32 = 8u;
const RESTIR_GI_DIELECTRIC_EVENT_INVALID: u32 = 0u;
const RESTIR_GI_DIELECTRIC_EVENT_REFLECTION: u32 = 1u;
const RESTIR_GI_DIELECTRIC_EVENT_TRANSMISSION: u32 = 2u;

fn restirGiSuffixChannel(value: vec3f, channel: u32) -> f32 {
  if (channel == 0u) { return value.r; }
  if (channel == 1u) { return value.g; }
  return value.b;
}

fn restirGiSuffixSetChannel(
  value: ptr<function, vec3f>,
  channel: u32,
  component: f32,
) {
  if (channel == 0u) { (*value).r = component; }
  if (channel == 1u) { (*value).g = component; }
  if (channel == 2u) { (*value).b = component; }
}

fn restirGiSuffixDiagonalTransfer(value: vec3f) -> mat3x3f {
  return mat3x3f(
    vec3f(value.x, 0.0, 0.0),
    vec3f(0.0, value.y, 0.0),
    vec3f(0.0, 0.0, value.z),
  );
}

// Linear form of the directionally aggregated homogeneous single-scatter
// contract. A matrix is required because the in-scatter source is driven by
// luminance and therefore mixes downstream RGB channels. Each dispersion lane
// still traces its own geometry; this operator preserves the same bounded
// cross-channel approximation as the native full-resolution glass path.
fn restirGiSuffixSegmentTransfer(
  absorption: vec3f,
  scatter: vec4f,
  albedo: vec3f,
  distance: f32,
) -> mat3x3f {
  let boundedAbsorption = clamp(absorption, vec3f(0.0), vec3f(1.0));
  if (!(distance > 0.0)) {
    return restirGiSuffixDiagonalTransfer(boundedAbsorption);
  }
  let sigmaS = max(scatter.rgb, vec3f(0.0));
  let sigmaA = -log(max(boundedAbsorption, vec3f(1e-30))) / distance;
  let sigmaT = sigmaA + sigmaS;
  let transmittance = boundedAbsorption *
    homogeneousBeerTransmittanceRgb(sigmaS, distance);
  var scatterAlbedo = vec3f(0.0);
  if (sigmaT.x > 0.0) { scatterAlbedo.x = sigmaS.x / sigmaT.x; }
  if (sigmaT.y > 0.0) { scatterAlbedo.y = sigmaS.y / sigmaT.y; }
  if (sigmaT.z > 0.0) { scatterAlbedo.z = sigmaS.z / sigmaT.z; }
  let sourceScale = max(albedo, vec3f(0.0)) * scatterAlbedo *
    (vec3f(1.0) - transmittance) *
    henyeyGreensteinPhase(0.0, clamp(scatter.a, -0.99, 0.99));
  return mat3x3f(
    vec3f(transmittance.x, 0.0, 0.0) + sourceScale * 0.2126,
    vec3f(0.0, transmittance.y, 0.0) + sourceScale * 0.7152,
    vec3f(0.0, 0.0, transmittance.z) + sourceScale * 0.0722,
  );
}

fn restirGiSuffixTransferFinite(transfer: mat3x3f) -> bool {
  let maximum = max(
    max(max(abs(transfer[0].x), abs(transfer[0].y)), abs(transfer[0].z)),
    max(
      max(max(abs(transfer[1].x), abs(transfer[1].y)), abs(transfer[1].z)),
      max(max(abs(transfer[2].x), abs(transfer[2].y)), abs(transfer[2].z)),
    ),
  );
  return maximum < 1e30;
}

fn restirGiSuffixTransferredChannel(
  transfer: mat3x3f,
  radiance: vec3f,
  channel: u32,
) -> f32 {
  return restirGiSuffixChannel(transfer * radiance, channel);
}

// Raw exit-face Beer reference plus the signed internal thickness convention:
// positive values cap path length through the actual exit texel; negative one
// denotes synthetic zero-thickness bulk whose full closed segment is used.
fn restirGiSuffixBeerReferenceForHit(
  hit: IntersectionResult,
  baseThickness: f32,
) -> vec4f {
  let beerCoord = vec2u(
    hit.indices.w % BVH_BEER_TEX_WIDTH,
    hit.indices.w / BVH_BEER_TEX_WIDTH,
  );
  let packedBeer = textureLoad(bvh_beer, vec2i(beerCoord), 0).r;
  let baseBeer = vec3f(
    f32((packedBeer >> 24u) & 0xffu) / 255.0,
    f32((packedBeer >> 16u) & 0xffu) / 255.0,
    f32((packedBeer >> 8u) & 0xffu) / 255.0,
  );
  return vec4f(
    baseBeer,
    baseThickness,
  );
}

fn restirGiSuffixTransportDistance(
  authoredThickness: f32,
  thicknessMapScale: f32,
  segmentDistance: f32,
) -> f32 {
  let hasAuthoredThickness = authoredThickness > 0.0;
  let referenceThickness = select(1.0, authoredThickness, hasAuthoredThickness);
  return select(
    segmentDistance,
    min(
      segmentDistance,
      referenceThickness * clamp(thicknessMapScale, 0.0, 1.0),
    ),
    hasAuthoredThickness,
  );
}

fn restirGiSuffixThicknessMapScaleForHit(
  hit: IntersectionResult,
) -> f32 {
  let thicknessMap = sampleMaterialAtlasRawAtOffsetForHit(
    hit, MATERIAL_MAP_THICKNESS_TEXEL_OFFSET,
  );
  if (thicknessMap.valid == 0u) { return 1.0; }
  return materialOpticalThicknessMapScale(
    hit.indices.w,
    thicknessMap.value.g,
  );
}

struct RestirGiDielectricLobe {
  direction: vec3f,
  weightRgb: vec3f,
  microfacetCos: f32,
  kind: u32,
  valid: u32,
};

// Sample one requested lobe of a rough-dielectric interface. The caller owns
// the discrete unit-envelope reflection-vs-transmission selection; this helper
// divides only by the continuous visible-normal/direction proposal. That keeps
// Fresnel reflection persistent at every authored scalar-transmission value and
// lets paired bulk exits cross without paying that scalar a second time.
fn restirGiSampleDielectricLobe(
  hit: IntersectionResult,
  n: vec3f,
  anisotropyTangent: vec3f,
  anisotropyBitangent: vec3f,
  wo: vec3f,
  rough: f32,
  anisotropy: f32,
  anisotropyRotation: f32,
  etaIncidentRgb: vec3f,
  etaTargetRgb: vec3f,
  frontFacing: bool,
  layer: vec4f,
  channel: u32,
  chooseReflection: bool,
  rng: ptr<function, u32>,
) -> RestirGiDielectricLobe {
  var out: RestirGiDielectricLobe;
  out.direction = vec3f(0.0);
  out.weightRgb = vec3f(0.0);
  out.microfacetCos = 0.0;
  out.kind = RESTIR_GI_DIELECTRIC_EVENT_INVALID;
  out.valid = 0u;

  let etaIncident = restirGiSuffixChannel(etaIncidentRgb, channel);
  let etaTarget = restirGiSuffixChannel(etaTargetRgb, channel);
  let nDotWo = dot(n, wo);
  if (
    !(nDotWo > 0.0) || etaIncident <= 0.0 || etaTarget <= 0.0
  ) { return out; }

  let authoredRoughness = clamp(rough, 0.0, 1.0);
  let aniso = clamp(anisotropy, 0.0, 1.0);
  var wm = n;
  if (authoredRoughness > 0.0) {
    if (aniso > 0.0) {
      let frame = anisotropyTangentFrameFromBasis(
        n,
        anisotropyTangent,
        anisotropyBitangent,
        anisotropyRotation,
      );
      let axes = ggxDielectricTransmissionAxes(
        authoredRoughness, aniso,
      );
      let woT = vec3f(
        dot(wo, frame[0]), dot(wo, frame[1]), dot(wo, n),
      );
      let wmT = ggxSampleVndfTangentAnisotropic(
        woT, axes.x, axes.y, rng,
      );
      wm = safe_normalize(
        wmT.x * frame[0] + wmT.y * frame[1] + wmT.z * n,
      );
    } else {
      let reflectedProposal = ggxSampleVndf(
        n, wo, authoredRoughness, rng,
      );
      wm = safe_normalize(wo + reflectedProposal);
      if (dot(wm, n) < 0.0) { wm = -wm; }
    }
  }

  let woDotM = dot(wo, wm);
  if (!(woDotM > 0.0)) { return out; }
  let reflectedDirection = safe_normalize(reflect(-wo, wm));
  let refractedRaw = refract(-wo, wm, etaIncident / etaTarget);
  let tir = dot(refractedRaw, refractedRaw) <= 1e-12;

  var transmittanceRgb = dielectricInterfaceTransmissionRgb(
    woDotM, etaIncidentRgb, etaTargetRgb,
  );
  var reflectanceRgb = vec3f(1.0) - transmittanceRgb;
  let film = materialThinFilmResponse(
    hit.indices.w, frontFacing, woDotM,
  );
  if (film.present != 0u) {
    reflectanceRgb = film.reflectance;
    transmittanceRgb = film.transmittance;
  }
  let etaRatioRgb = max(etaIncidentRgb, vec3f(1e-6)) /
    max(etaTargetRgb, vec3f(1e-6));
  let sin2TargetRgb = etaRatioRgb * etaRatioRgb *
    (1.0 - woDotM * woDotM);
  let tirRgb = sin2TargetRgb >= vec3f(1.0);
  reflectanceRgb = select(reflectanceRgb, vec3f(1.0), tirRgb);
  transmittanceRgb = select(transmittanceRgb, vec3f(0.0), tirRgb);
  let layerTransferRgb = faceLayerTransmission(layer);
  reflectanceRgb = clamp(
    reflectanceRgb, vec3f(0.0), vec3f(1.0),
  ) * layerTransferRgb;
  transmittanceRgb = clamp(
    transmittanceRgb, vec3f(0.0), vec3f(1.0),
  ) * layerTransferRgb;
  let reflectance = restirGiSuffixChannel(reflectanceRgb, channel);
  let transmittance = restirGiSuffixChannel(
    transmittanceRgb, channel,
  );
  var directionalBaseWeight = 0.0;
  if (chooseReflection) {
    let nDotWi = dot(n, reflectedDirection);
    if (!(nDotWi > 0.0) || !(reflectance > 0.0)) { return out; }
    if (authoredRoughness <= 0.0) {
      directionalBaseWeight = 1.0;
    } else {
      var D = 0.0;
      var G1o = 0.0;
      var G1i = 0.0;
      if (aniso > 0.0) {
        let frame = anisotropyTangentFrameFromBasis(
          n,
          anisotropyTangent,
          anisotropyBitangent,
          anisotropyRotation,
        );
        let axes = ggxDielectricTransmissionAxes(
          authoredRoughness, aniso,
        );
        D = distributionGGXAnisotropic(
          n, frame[0], frame[1], wm, axes.x, axes.y,
        );
        G1o = geometrySmithGGXAnisotropicG1(
          n, frame[0], frame[1], wo, axes.x, axes.y,
        );
        G1i = geometrySmithGGXAnisotropicG1(
          n, frame[0], frame[1], reflectedDirection, axes.x, axes.y,
        );
      } else {
        let alpha = authoredRoughness * authoredRoughness;
        let alpha2 = alpha * alpha;
        D = distributionGGX(dot(n, wm), authoredRoughness);
        G1o = smithG1GGX(nDotWo, alpha2);
        G1i = smithG1GGX(nDotWi, alpha2);
      }
      let directionPdf = D * G1o / (4.0 * nDotWo);
      let frBase = D * G1o * G1i /
        (4.0 * nDotWo * nDotWi);
      if (!(directionPdf > 0.0) || !(frBase > 0.0)) { return out; }
      directionalBaseWeight = frBase * nDotWi / directionPdf;
    }
    out.direction = reflectedDirection;
    out.kind = RESTIR_GI_DIELECTRIC_EVENT_REFLECTION;
    out.weightRgb = reflectanceRgb * directionalBaseWeight;
  } else {
    if (tir || !(transmittance > 0.0)) { return out; }
    let refractedDirection = safe_normalize(refractedRaw);
    let nDotWiAbs = abs(dot(n, refractedDirection));
    let wiDotM = dot(refractedDirection, wm);
    let etap = etaTarget / etaIncident;
    let denom = wiDotM + woDotM / etap;
    if (
      dot(n, refractedDirection) >= 0.0 || !(nDotWiAbs > 0.0) ||
      wiDotM >= 0.0 || abs(denom) <= 1e-8
    ) { return out; }
    if (authoredRoughness <= 0.0) {
      directionalBaseWeight = 1.0 / (etap * etap);
    } else {
      var D = 0.0;
      var G1o = 0.0;
      var G1i = 0.0;
      if (aniso > 0.0) {
        let frame = anisotropyTangentFrameFromBasis(
          n,
          anisotropyTangent,
          anisotropyBitangent,
          anisotropyRotation,
        );
        let axes = ggxDielectricTransmissionAxes(
          authoredRoughness, aniso,
        );
        D = distributionGGXAnisotropic(
          n, frame[0], frame[1], wm, axes.x, axes.y,
        );
        G1o = geometrySmithGGXAnisotropicG1(
          n, frame[0], frame[1], wo, axes.x, axes.y,
        );
        G1i = geometrySmithGGXAnisotropicG1(
          n, frame[0], frame[1], -refractedDirection, axes.x, axes.y,
        );
      } else {
        let alpha = authoredRoughness * authoredRoughness;
        let alpha2 = alpha * alpha;
        D = distributionGGX(dot(n, wm), authoredRoughness);
        G1o = smithG1GGX(nDotWo, alpha2);
        G1i = smithG1GGX(nDotWiAbs, alpha2);
      }
      let microfacetPdf = D * G1o * abs(woDotM) / nDotWo;
      let directionPdf = microfacetPdf * abs(wiDotM) /
        (denom * denom);
      let ftBase = D * G1o * G1i *
        abs(wiDotM * woDotM /
          (nDotWiAbs * nDotWo * denom * denom)) /
        (etap * etap);
      if (!(directionPdf > 0.0) || !(ftBase > 0.0)) { return out; }
      directionalBaseWeight = ftBase * nDotWiAbs / directionPdf;
    }
    out.direction = refractedDirection;
    out.kind = RESTIR_GI_DIELECTRIC_EVENT_TRANSMISSION;
    out.weightRgb = transmittanceRgb * directionalBaseWeight;
  }

  let selectedWeight = restirGiSuffixChannel(out.weightRgb, channel);
  let maximumWeight = max(
    out.weightRgb.x, max(out.weightRgb.y, out.weightRgb.z),
  );
  if (!(selectedWeight > 0.0) || maximumWeight >= 1e30) {
    out.weightRgb = vec3f(0.0);
    out.kind = RESTIR_GI_DIELECTRIC_EVENT_INVALID;
    return out;
  }
  out.microfacetCos = woDotM;
  out.valid = 1u;
  return out;
}

// Trace the ordinary surface closure of a transmissive first-hit GI suffix.
// This estimator is deliberately receiver-local: rough-interface draws, full
// reflection/refraction state, nested media, spectral Beer transfer, and its
// terminal receiver are not represented by the compact reservoir. Segment
// volume transport carries absorption, scattering extinction, and the same
// 3x3 cross-channel single-scatter source as the native camera glass path.
fn traceRestirGiDielectricSuffixChannel(
  firstHit: IntersectionResult,
  firstSourceFeature: OpticalSourceFeature,
  firstPos: vec3f,
  firstSegmentDistance: f32,
  firstShadingNormal: vec3f,
  firstDirection: vec3f,
  firstEmissionLo: vec3f,
  firstOpaqueLo: vec3f,
  firstTransmission: f32,
  channel: u32,
  rng: ptr<function, u32>,
) -> f32 {
  var prefixTransfer = restirGiSuffixDiagonalTransfer(vec3f(1.0));
  var accumulatedRadiance = 0.0;
  var rayDirection = safe_normalize(firstDirection);
  var currentHit = firstHit;
  var currentPos = firstPos;
  var currentShadingNormal = firstShadingNormal;
  var currentOpaqueLo = firstOpaqueLo;
  var currentTransmission = clamp(firstTransmission, 0.0, 1.0);
  var mediumDepth = 0u;
  var mediumIor: array<vec3f, 8>;
  var mediumTri: array<u32, 8>;
  var mediumMaterialId: array<u32, 8>;
  var mediumInstance: array<u32, 8>;
  var mediumBeer: array<vec3f, 8>;
  var mediumThickness: array<f32, 8>;
  var mediumThicknessMapScale: array<f32, 8>;
  var mediumAlbedo: array<vec3f, 8>;
  var mediumScatter: array<vec4f, 8>;
  var mediumTransmissionPaid: array<u32, 8>;
  var interfaceCount = 0u;
  var continuationSourceFeature = firstSourceFeature;

  // Reconstruct from the actual incoming-segment launch point and direction;
  // this makes firstHit itself the exit payload when that launch is inside a
  // medium and avoids arbitrary containment directions or origin stepping.
  let suffixOrigin = firstPos - rayDirection * firstSegmentDistance;
  let containing = materialShadowClassifyContainingMedia(
    ubo.bvhMode,
    ubo.tlasNodeCount,
    suffixOrigin,
    rayDirection,
    ubo.triIntersectEpsilon,
    bvh_material,
    BVH_MATERIAL_TEX_WIDTH,
    bvh_beer,
  );
  if (
    containing.valid == 0u ||
    containing.state.depth > RESTIR_GI_SUFFIX_MAX_INTERFACES
  ) { return 0.0; }
  for (var seed = 0u; seed < containing.state.depth; seed = seed + 1u) {
    let triIndex = containing.state.tri[seed];
    let materialCoord = vec2u(
      triIndex % BVH_MATERIAL_TEX_WIDTH,
      triIndex / BVH_MATERIAL_TEX_WIDTH,
    );
    let materialWord = textureLoad(
      bvh_material, vec2i(materialCoord), 0,
    ).r;
    mediumIor[seed] = materialDispersionIorRgb(
      triIndex, decodeIor(materialWord),
    );
    mediumTri[seed] = triIndex;
    mediumMaterialId[seed] = containing.state.materialId[seed];
    mediumInstance[seed] = containing.state.instance[seed];
    mediumBeer[seed] = containing.state.tint[seed];
    mediumThickness[seed] = containing.state.thickness[seed];
    mediumThicknessMapScale[seed] =
      containing.state.thicknessMapScale[seed];
    mediumAlbedo[seed] = containing.state.albedo[seed];
    mediumScatter[seed] = containing.state.scattering[seed];
    // A suffix born inside this medium has not yet paid the material's scalar
    // transmission. Its first forward exit owns that one-time factor.
    mediumTransmissionPaid[seed] = 0u;
  }
  mediumDepth = containing.state.depth;

  // When the suffix origin is already inside a closed medium, the first hit is
  // its exit. Apply the actual candidate segment before evaluating that exit;
  // otherwise a containing medium would lose all absorption/scattering
  // extinction between the visible point and its first boundary.
  if (mediumDepth > 0u && firstSegmentDistance > 0.0) {
    let top = mediumDepth - 1u;
    let referenceThickness = select(
      1.0, mediumThickness[top], mediumThickness[top] > 0.0,
    );
    let transportDistance = restirGiSuffixTransportDistance(
      mediumThickness[top],
      mediumThicknessMapScale[top],
      firstSegmentDistance,
    );
    let distanceScale = transportDistance / referenceThickness;
    let fallbackBeer = pow(
      clamp(mediumBeer[top], vec3f(0.0), vec3f(1.0)),
      vec3f(distanceScale),
    );
    let spectralTransfer = materialSpectralAttenuation(
      mediumTri[top],
      transportDistance,
      fallbackBeer,
    );
    prefixTransfer = prefixTransfer * restirGiSuffixSegmentTransfer(
      spectralTransfer,
      mediumScatter[top],
      mediumAlbedo[top],
      transportDistance,
    );
    if (!restirGiSuffixTransferFinite(prefixTransfer)) {
      return accumulatedRadiance;
    }
  }

  // The first receiver's emission travels over that same incoming segment.
  // Accumulate it only after the containing-medium transfer so it is neither
  // unattenuated at an inside-volume exit nor paid again by the caller.
  accumulatedRadiance = accumulatedRadiance +
    restirGiSuffixTransferredChannel(
      prefixTransfer, firstEmissionLo, channel,
    );

  // Eight dielectric interface events plus a terminal opaque/environment
  // query. Thin sheets may spend the budget on reciprocal internal bounces.
  for (
    var depth = 0u;
    depth <= RESTIR_GI_SUFFIX_MAX_INTERFACES;
    depth = depth + 1u
  ) {
    let materialCoord = vec2u(
      currentHit.indices.w % BVH_MATERIAL_TEX_WIDTH,
      currentHit.indices.w / BVH_MATERIAL_TEX_WIDTH,
    );
    let materialWord = textureLoad(
      bvh_material, vec2i(materialCoord), 0,
    ).r;
    let scalarMaterial = decodeMaterialColor(currentHit.matColorPacked);
    let mappedTransmission = clamp(currentTransmission, 0.0, 1.0);
    let currentUseTlas = ubo.bvhMode == 1u && ubo.tlasNodeCount > 0u;
    let currentBoundaryId = sceneOpticalEncodedBoundaryId(
      currentUseTlas, currentHit.indices.w, currentHit.instanceIndex,
    );
    let currentRepresentedId = sceneOpticalRepresentedPrimitiveInstanceId(
      currentUseTlas, currentHit.indices.w, currentHit.instanceIndex,
    );

    let materialLayer = sampleFaceLayerControls(
      currentHit.indices.w, currentHit.side >= 0.0,
    );
    let mappedBaseRoughness = sampleMaterialScalarMap(
      currentHit,
      MATERIAL_MAP_SLOT_ROUGHNESS,
      1u,
      decodeRoughMetal(materialWord).x,
    );
    let materialRoughness = faceLayerRoughness(
      mappedBaseRoughness,
      materialLayer,
    );
    let materialAnisotropy = sampleAnisotropyControls(currentHit);
    let materialIor = materialDispersionIorRgb(
      currentHit.indices.w, decodeIor(materialWord),
    );
    let baseMaterialThickness = materialShadowAuthoredThickness(currentHit);
    let hasBulkTopology =
      materialHasTransmission(scalarMaterial.a) &&
      currentBoundaryId != 0u;
    if (
      materialHasTransmission(scalarMaterial.a) &&
      currentRepresentedId == 0u
    ) {
      return accumulatedRadiance;
    }

    let entering = currentHit.side >= 0.0;
    var pairedPaidExit = false;
    if (hasBulkTopology && !entering && mediumDepth > 0u) {
      let top = mediumDepth - 1u;
      pairedPaidExit =
        mediumMaterialId[top] == currentBoundaryId &&
        mediumInstance[top] == currentRepresentedId &&
        mediumTransmissionPaid[top] != 0u;
    }
    if (!(mappedTransmission > 0.0) && !pairedPaidExit) {
      return accumulatedRadiance + restirGiSuffixTransferredChannel(
        prefixTransfer, currentOpaqueLo, channel,
      );
    }
    if (
      hasBulkTopology && !entering &&
      mediumDepth == 0u && mappedTransmission > 0.0
    ) {
      return accumulatedRadiance;
    }
    if (interfaceCount >= RESTIR_GI_SUFFIX_MAX_INTERFACES) {
      return accumulatedRadiance;
    }

    let alignedNormal = select(
      -currentShadingNormal,
      currentShadingNormal,
      dot(currentShadingNormal, currentHit.normal) >= 0.0,
    );
    let faceNormal = select(
      -alignedNormal,
      alignedNormal,
      dot(rayDirection, alignedNormal) < 0.0,
    );
    if (dot(rayDirection, faceNormal) >= -1e-6) {
      return accumulatedRadiance;
    }
    let anisotropyFrame = materialTangentFrameForHit(
      currentHit, faceNormal, MATERIAL_MAP_ANISOTROPY_TEXEL_OFFSET,
    );

    var incidentIor = vec3f(1.0);
    if (mediumDepth > 0u) {
      incidentIor = mediumIor[mediumDepth - 1u];
    }
    var targetIor = materialIor;
    if (hasBulkTopology && !entering) {
      let top = mediumDepth - 1u;
      if (
        mediumMaterialId[top] != currentBoundaryId ||
        mediumInstance[top] != currentRepresentedId
      ) {
        return accumulatedRadiance;
      }
      targetIor = vec3f(1.0);
      if (mediumDepth > 1u) {
        targetIor = mediumIor[mediumDepth - 2u];
      }
    }

    // The reflection family has a unit envelope: Fresnel reflection persists
    // at t=1, while the complementary opaque source contributes (1-t) only
    // while this material traversal's scalar is unpaid. A paired paid bulk
    // exit uses tFactor=1 and has no second opaque/transmission material split.
    let transmissionPhysicalWeight = select(
      mappedTransmission, 1.0, pairedPaidExit,
    );
    let idealTransmissionBranchPdf = transmissionPhysicalWeight /
      (1.0 + transmissionPhysicalWeight);
    let transmissionBranchPdf = represented_bernoulli_probability_f32(
      idealTransmissionBranchPdf,
    );
    let reflectionBranchPdf = 1.0 - transmissionBranchPdf;
    let chooseTransmission =
      transmissionBranchPdf > 0.0 &&
      rand_f32(rng) < transmissionBranchPdf;

    if (!chooseTransmission) {
      let opaquePhysicalWeight = select(
        1.0 - mappedTransmission, 0.0, pairedPaidExit,
      );
      if (opaquePhysicalWeight > 0.0) {
        accumulatedRadiance = accumulatedRadiance +
          restirGiSuffixTransferredChannel(
            prefixTransfer, currentOpaqueLo, channel,
          ) *
          opaquePhysicalWeight / reflectionBranchPdf;
      }
    }

    let interfaceLobe = restirGiSampleDielectricLobe(
      currentHit,
      faceNormal,
      anisotropyFrame.tangent,
      anisotropyFrame.bitangent,
      -rayDirection,
      materialRoughness,
      materialAnisotropy.x,
      materialAnisotropy.y,
      incidentIor,
      targetIor,
      currentHit.side >= 0.0,
      materialLayer,
      channel,
      !chooseTransmission,
      rng,
    );
    interfaceCount = interfaceCount + 1u;
    if (interfaceLobe.valid == 0u) {
      return accumulatedRadiance;
    }
    if (chooseTransmission) {
      prefixTransfer = prefixTransfer * restirGiSuffixDiagonalTransfer(
        interfaceLobe.weightRgb *
          transmissionPhysicalWeight / transmissionBranchPdf,
      );
    } else {
      prefixTransfer = prefixTransfer * restirGiSuffixDiagonalTransfer(
        interfaceLobe.weightRgb / reflectionBranchPdf,
      );
    }
    if (!restirGiSuffixTransferFinite(prefixTransfer)) {
      return accumulatedRadiance;
    }
    rayDirection = interfaceLobe.direction;

    if (chooseTransmission) {
      if (!hasBulkTopology) {
        // A zero-thickness sheet is a reciprocal two-boundary slab. Preserve
        // reflected energy inside the slab by alternating authored faces until
        // it exits or exhausts the same global interface budget.
        let slabSmoothNormal = restir_gi_smooth_normal_for_hit(
          currentHit, currentHit.normal,
        );
        var slabFrontFacing = currentHit.side < 0.0;
        var slabExited = false;
        loop {
          if (interfaceCount >= RESTIR_GI_SUFFIX_MAX_INTERFACES) {
            break;
          }
          let slabLayer = sampleFaceLayerControls(
            currentHit.indices.w, slabFrontFacing,
          );
          let slabRoughness = faceLayerRoughness(
            mappedBaseRoughness, slabLayer,
          );
          let slabMappedNormal = applyBumpMapForHit(
            currentHit,
            applyNormalMapForSideForHit(
              currentHit, slabSmoothNormal, slabFrontFacing,
            ),
          );
          let slabAlignedNormal = select(
            -slabMappedNormal,
            slabMappedNormal,
            dot(slabMappedNormal, currentHit.normal) >= 0.0,
          );
          let slabNormal = select(
            -slabAlignedNormal,
            slabAlignedNormal,
            dot(rayDirection, slabAlignedNormal) < 0.0,
          );
          let slabFrame = materialTangentFrameForHit(
            currentHit,
            slabNormal,
            MATERIAL_MAP_ANISOTROPY_TEXEL_OFFSET,
          );
          // The authored scalar was paid at the sheet's forward interface.
          // Reciprocal internal boundaries therefore sample only the unit
          // reflection/transmission optical envelope (one half each), with no
          // local diffuse source and no further scalar-transmission factor.
          let slabTransmissionPdf = represented_bernoulli_probability_f32(0.5);
          let slabChooseTransmission =
            rand_f32(rng) < slabTransmissionPdf;
          let slabLobe = restirGiSampleDielectricLobe(
            currentHit,
            slabNormal,
            slabFrame.tangent,
            slabFrame.bitangent,
            -rayDirection,
            slabRoughness,
            materialAnisotropy.x,
            materialAnisotropy.y,
            materialIor,
            incidentIor,
            slabFrontFacing,
            slabLayer,
            channel,
            !slabChooseTransmission,
            rng,
          );
          interfaceCount = interfaceCount + 1u;
          if (slabLobe.valid == 0u) { break; }
          let slabSelectedPdf = select(
            1.0 - slabTransmissionPdf,
            slabTransmissionPdf,
            slabChooseTransmission,
          );
          prefixTransfer = prefixTransfer * restirGiSuffixDiagonalTransfer(
            slabLobe.weightRgb / slabSelectedPdf,
          );
          if (!restirGiSuffixTransferFinite(prefixTransfer)) { break; }
          rayDirection = slabLobe.direction;
          if (slabChooseTransmission) {
            slabExited = true;
            break;
          }
          slabFrontFacing = !slabFrontFacing;
        }
        if (!slabExited) { return accumulatedRadiance; }
      } else if (entering) {
        if (mediumDepth >= RESTIR_GI_SUFFIX_MAX_INTERFACES) {
          return accumulatedRadiance;
        }
        let beerReference = restirGiSuffixBeerReferenceForHit(
          currentHit, baseMaterialThickness,
        );
        mediumIor[mediumDepth] = materialIor;
        mediumTri[mediumDepth] = currentHit.indices.w;
        mediumMaterialId[mediumDepth] = currentBoundaryId;
        mediumInstance[mediumDepth] = currentRepresentedId;
        mediumBeer[mediumDepth] = beerReference.rgb;
        mediumThickness[mediumDepth] = beerReference.a;
        mediumThicknessMapScale[mediumDepth] =
          restirGiSuffixThicknessMapScaleForHit(currentHit);
        let mediumVertexColor = sampleVertexColorForHit(currentHit);
        mediumAlbedo[mediumDepth] = sampleBaseColorMap(
          currentHit,
          scalarMaterial.rgb * mediumVertexColor.rgb,
        );
        mediumScatter[mediumDepth] = sampleVolumeScatteringControls(
          currentHit.indices.w,
        );
        mediumTransmissionPaid[mediumDepth] = 1u;
        mediumDepth = mediumDepth + 1u;
      } else if (mediumDepth > 0u) {
        mediumDepth = mediumDepth - 1u;
      }
    }

    if (!restirGiSuffixTransferFinite(prefixTransfer)) {
      return accumulatedRadiance;
    }

    let nextOrigin = currentPos;
    let nextRay = Ray(nextOrigin, rayDirection);
    let sourceAware = traceSceneFirstHitAlphaMaskTexturedWithOpticalSource(
      ubo.bvhMode,
      ubo.tlasNodeCount,
      nextRay,
      continuationSourceFeature,
      bvh_material,
      BVH_MATERIAL_TEX_WIDTH,
      pcgNext(rng),
    );
    if (sourceAware.valid == 0u) { return accumulatedRadiance; }
    var nextHit = sourceAware.hit;
    if (!nextHit.didHit) {
      if (mediumDepth == 0u) {
        return accumulatedRadiance + restirGiSuffixTransferredChannel(
          prefixTransfer, envRadiance(rayDirection), channel,
        );
      }
      return accumulatedRadiance;
    }

    var acceptedNextSourceFeature = opticalSourceFeatureInvalid();
    if (packedMaterialHasTransmission(nextHit.matColorPacked)) {
      let exactNext = traceSceneRetraceOpticalHit(
        ubo.bvhMode, ubo.tlasNodeCount, nextRay, nextHit, 0.0,
      );
      let nextSourceFeature = sceneOpticalSourceFeatureForExactHit(
        ubo.bvhMode, ubo.tlasNodeCount, nextHit, exactNext,
      );
      if (
        !exactNext.hit ||
        nextSourceFeature.kind == OPTICAL_SOURCE_FEATURE_INVALID
      ) { return accumulatedRadiance; }
      let nextUseTlas = ubo.bvhMode == 1u && ubo.tlasNodeCount > 0u;
      let nextExactTriangle = sceneLoadOpticalWorldTriangle(
        nextUseTlas, nextHit.indices.w, nextHit.instanceIndex,
      );
      if (nextExactTriangle.valid == 0u) { return accumulatedRadiance; }
      nextHit.normal = exactNext.normal;
      nextHit.barycoord = exactNext.bary;
      nextHit.side = exactNext.side;
      nextHit.dist = exactNext.t;
      nextHit.uv = exactNext.bary.x * nextExactTriangle.uvA +
        exactNext.bary.y * nextExactTriangle.uvB +
        exactNext.bary.z * nextExactTriangle.uvC;
      acceptedNextSourceFeature = nextSourceFeature;
    }
    let nextUseTlas = ubo.bvhMode == 1u && ubo.tlasNodeCount > 0u;
    let nextBoundaryId = sceneOpticalEncodedBoundaryId(
      nextUseTlas, nextHit.indices.w, nextHit.instanceIndex,
    );
    let nextRepresentedId = sceneOpticalRepresentedPrimitiveInstanceId(
      nextUseTlas, nextHit.indices.w, nextHit.instanceIndex,
    );

    if (mediumDepth > 0u) {
      let top = mediumDepth - 1u;
      let segmentDistance = nextHit.dist;
      var segmentTri = mediumTri[top];
      var segmentBeer = mediumBeer[top];
      var segmentThickness = mediumThickness[top];
      var segmentThicknessMapScale = mediumThicknessMapScale[top];
      var segmentAlbedo = mediumAlbedo[top];
      var segmentScatter = mediumScatter[top];
      if (
        packedMaterialHasTransmission(nextHit.matColorPacked) &&
        nextHit.side < 0.0 &&
        nextBoundaryId == mediumMaterialId[top] &&
        nextRepresentedId == mediumInstance[top]
      ) {
        let exitReference = restirGiSuffixBeerReferenceForHit(
          nextHit, materialShadowAuthoredThickness(nextHit),
        );
        segmentTri = nextHit.indices.w;
        segmentBeer = exitReference.rgb;
        segmentThickness = exitReference.a;
        segmentThicknessMapScale =
          restirGiSuffixThicknessMapScaleForHit(nextHit);
        let exitScalar = decodeMaterialColor(nextHit.matColorPacked);
        let exitVertexColor = sampleVertexColorForHit(nextHit);
        segmentAlbedo = sampleBaseColorMap(
          nextHit, exitScalar.rgb * exitVertexColor.rgb,
        );
        segmentScatter = sampleVolumeScatteringControls(nextHit.indices.w);
      }
      let referenceThickness = select(
        1.0, segmentThickness, segmentThickness > 0.0,
      );
      let transportDistance = restirGiSuffixTransportDistance(
        segmentThickness,
        segmentThicknessMapScale,
        segmentDistance,
      );
      let distanceScale = transportDistance / referenceThickness;
      let fallbackBeer = pow(
        clamp(segmentBeer, vec3f(0.0), vec3f(1.0)),
        vec3f(distanceScale),
      );
      let spectralTransfer = materialSpectralAttenuation(
        segmentTri,
        transportDistance,
        fallbackBeer,
      );
      prefixTransfer = prefixTransfer * restirGiSuffixSegmentTransfer(
        spectralTransfer,
        segmentScatter,
        segmentAlbedo,
        transportDistance,
      );
      if (!restirGiSuffixTransferFinite(prefixTransfer)) {
        return accumulatedRadiance;
      }
    }

    let nextPos = nextOrigin + rayDirection * nextHit.dist;
    let nextSmoothNormal = restir_gi_smooth_normal_for_hit(
      nextHit, nextHit.normal,
    );
    let nextShadingNormal = applyBumpMapForHit(
      nextHit,
      applyNormalMapForHit(nextHit, nextSmoothNormal),
    );
    let nextIrradiance = min(
      sampleDDGIAtPoint(nextPos, nextShadingNormal),
      vec3f(ubo.restirGiIrrClamp),
    );
    let nextCoord = vec2u(
      nextHit.indices.w % BVH_MATERIAL_TEX_WIDTH,
      nextHit.indices.w / BVH_MATERIAL_TEX_WIDTH,
    );
    let nextWord = textureLoad(bvh_material, vec2i(nextCoord), 0).r;
    let nextPayload = sampleRestirGIHitMaterialForHit(
      nextHit,
      nextSmoothNormal,
      nextShadingNormal,
      nextIrradiance,
      rayDirection,
      nextWord,
    );
    if (decodeIsUnlitMaterial(nextWord)) {
      return accumulatedRadiance + restirGiSuffixTransferredChannel(
        prefixTransfer, nextPayload.Lo, channel,
      );
    }

    let nextTransmission = clamp(nextPayload.transmission, 0.0, 1.0);
    accumulatedRadiance = accumulatedRadiance +
      restirGiSuffixTransferredChannel(
        prefixTransfer, nextPayload.emissionLo, channel,
      );
    currentHit = nextHit;
    currentPos = nextPos;
    currentShadingNormal = nextShadingNormal;
    currentOpaqueLo = nextPayload.opaqueLo;
    currentTransmission = nextTransmission;
    continuationSourceFeature = acceptedNextSourceFeature;
  }

  return accumulatedRadiance;
}

fn traceRestirGiDielectricSuffix(
  firstHitInput: IntersectionResult,
  firstPosInput: vec3f,
  firstSegmentDistanceInput: f32,
  firstShadingNormal: vec3f,
  firstDirection: vec3f,
  firstEmissionLo: vec3f,
  firstOpaqueLo: vec3f,
  firstTransmission: f32,
  rng: ptr<function, u32>,
) -> vec3f {
  var result = vec3f(0.0);
  var firstHit = firstHitInput;
  var firstPos = firstPosInput;
  var firstSegmentDistance = firstSegmentDistanceInput;
  let exactDirection = safe_normalize(firstDirection);
  let replayDistance = max(firstSegmentDistance, 1.0);
  let replayRay = Ray(
    firstPos - exactDirection * replayDistance,
    exactDirection,
  );
  let exactFirst = traceSceneRetraceOpticalHit(
    ubo.bvhMode, ubo.tlasNodeCount, replayRay, firstHit, 0.0,
  );
  let firstSourceFeature = sceneOpticalSourceFeatureForExactHit(
    ubo.bvhMode, ubo.tlasNodeCount, firstHit, exactFirst,
  );
  if (
    !exactFirst.hit ||
    firstSourceFeature.kind == OPTICAL_SOURCE_FEATURE_INVALID
  ) { return result; }
  let firstUseTlas = ubo.bvhMode == 1u && ubo.tlasNodeCount > 0u;
  let firstTriangle = sceneLoadOpticalWorldTriangle(
    firstUseTlas, firstHit.indices.w, firstHit.instanceIndex,
  );
  if (firstTriangle.valid == 0u) { return result; }
  firstHit.normal = exactFirst.normal;
  firstHit.barycoord = exactFirst.bary;
  firstHit.side = exactFirst.side;
  firstHit.dist = exactFirst.t;
  firstHit.uv = exactFirst.bary.x * firstTriangle.uvA +
    exactFirst.bary.y * firstTriangle.uvB +
    exactFirst.bary.z * firstTriangle.uvC;
  firstPos = replayRay.origin + replayRay.direction * exactFirst.t;
  let exactFirstSmoothNormal = restir_gi_smooth_normal_for_hit(
    firstHit, firstHit.normal,
  );
  let exactFirstShadingNormal = applyBumpMapForHit(
    firstHit,
    applyNormalMapForHit(firstHit, exactFirstSmoothNormal),
  );
  // One parent draw owns the suffix. Each RGB channel receives an identical
  // mutable RNG state, providing common-random-number correlation while the
  // caller's parent stream advances by a fixed amount independent of channel
  // divergence or interface count.
  let suffixRngSeed = pcgNext(rng);
  for (var channel = 0u; channel < 3u; channel = channel + 1u) {
    var channelRng = suffixRngSeed;
    let component = traceRestirGiDielectricSuffixChannel(
      firstHit,
      firstSourceFeature,
      firstPos,
      firstSegmentDistance,
      exactFirstShadingNormal,
      firstDirection,
      firstEmissionLo,
      firstOpaqueLo,
      firstTransmission,
      channel,
      &channelRng,
    );
    restirGiSuffixSetChannel(&result, channel, component);
  }
  return result;
}

// sampleCosineHemisphere is the canonical helper from @vitrum/shared-samplers'
// bsdfPrimitives.wgsl, injected into the composed shade module via composeWgsl
// (SHARED_PRIMITIVES_MODULE → BSDF_PRIMITIVES_WGSL).

@compute @workgroup_size(8, 8, 1)
fn risGiMain(@builtin(global_invocation_id) gid: vec3u) {
  let fullDims = ubo.screenSize;
  let halfDims = restirGiDimensions();
  if (any(gid.xy >= halfDims)) { return; }

  let pixelIdxGi = gid.y * halfDims.x + gid.x;

  // Sample point in full-res: centre of the 2×2 quad.
  let fullPx = restirGiFullPixel(gid.xy);

  var rng = pcgInit(
    gid.x ^ (ubo.frameSeed * 0xA5A5u),
    gid.y ^ (ubo.frameSeed * 0x5A5Au),
    ubo.frameSeed ^ 0xC1A2u,
  );

  // Re-cast primary ray to find the visible surface.
  let vp = ubo.projMatrix * ubo.viewMatrix;
  let invVP = invertMat4_common(vp);
  let primaryRay = generatePrimaryRay_common(
    fullPx.x, fullPx.y, fullDims.x, fullDims.y, ubo.cameraPos, invVP,
  );
  let hit = traceSceneFirstHitAlphaMaskTexturedOpaqueOnly(
    ubo.bvhMode, ubo.tlasNodeCount,

    primaryRay, ubo.triIntersectEpsilon,
    bvh_material, BVH_MATERIAL_TEX_WIDTH, 0u);
  if (!hit.didHit) {
    storeReservoirGI_rw(pixelIdxGi, emptyReservoirGI());
    return;
  }

  let pos = primaryRay.origin + primaryRay.direction * hit.dist;
  // WS1 — smooth shading normal (visible-point normal + hemisphere frame);
  // geometric normal kept for the bounce-ray offset. V21 — applies in TLAS too
  // (transform the LOCAL blend to world by the hit instance inverse-transpose).
  let geoNormal = hit.normal;
  let n_isTlas = ubo.bvhMode == 1u;
  let n_base = hit.instanceIndex * 4u;
  let n_ok = n_isTlas && n_base + 2u < tlasWorldToLocalColumnCount();
  let n_i = select(0u, n_base, n_ok);
  let smoothNormal = smoothShadingNormal(
    hit, geoNormal,
    sceneLoadBvhNormal(hit.indices.x).xyz, sceneLoadBvhNormal(hit.indices.y).xyz, sceneLoadBvhNormal(hit.indices.z).xyz,
    n_ok,
    tlasLoadWorldToLocalColumn(n_i), tlasLoadWorldToLocalColumn(n_i + 1u), tlasLoadWorldToLocalColumn(n_i + 2u),
  );
  let normalMapped = applyNormalMapForHit(hit, smoothNormal);
  let normal = applyBumpMapForHit(hit, normalMapped);
  // B1 (road-to-100) — metals/glossy now get a GI reservoir. The reservoir uses
  // cosine-hemisphere candidates and the visible-point receiver-lobe target.
  // Diffuse-default p̂ remains luminance(Lo)·cosθ·INV_PI; rich receivers add
  // their specular/clearcoat/sheen lobes. The suffix Lo is material-aware:
  // DDGI irradiance is multiplied by mapped diffuse albedo / π, then authored
  // emission/light maps, face-layer transmission, and homogeneous volume
  // response are applied. shade then reflects
  // this stored radiance off the receiver via the visible material's indirect
  // lobes (for metals/glossy, shade.lo_indirectSpecular), so the old empty punt
  // is gone without widening the GI reservoir payload.
  //
  // Glass primaries build their reservoir at the first opaque surface reached
  // by the bounded dielectric-prefix walk below. Every crossed interface gets
  // its own Snell/Fresnel solve and medium-stack update; TIR or budget overflow
  // fails closed rather than relabelling a glass exit face as the receiver.
  let scalarMatColor = decodeMaterialColor(hit.matColorPacked);
  let matColor = vec4f(
    scalarMatColor.rgb,
    sampleTransmissionMapForHit(hit, scalarMatColor.a),
  );
  let currentGrisEpoch = bitcast<u32>(ubo.sunAngular.y);
  // Camera-prefix transmission is evaluated at the exact full-resolution
  // primary pixel inside shadeMain and never enters this buffer. This ordinary
  // reservoir remains live at transmissive primaries for their continuous
  // (1-transmission) opaque share and the reflection lobes that exist at every
  // transmission value, including the fully transmissive endpoint.

  let receiverMaterialCoord = vec2u(
    hit.indices.w % BVH_MATERIAL_TEX_WIDTH,
    hit.indices.w / BVH_MATERIAL_TEX_WIDTH,
  );
  let receiverMaterialWord = textureLoad(
    bvh_material,
    vec2i(receiverMaterialCoord),
    0,
  ).r;
  let receiverPayload = sampleRestirDIMaterialPayloadForHit(
    hit,
    smoothNormal,
    normal,
    matColor.rgb,
    receiverMaterialWord,
    safe_normalize(-primaryRay.direction),
  );

  var r: ReservoirGI = emptyReservoirGI();
  var giWrs = representedWrsInit();
  r.xv = pos;
  r.nv = normal;
  r.receiverMaterialKey = restir_gi_receiver_domain_key(
    hit.matColorPacked,
    receiverMaterialWord,
    hit.indices.w,
    select(0u, hit.instanceIndex, ubo.bvhMode == 1u),
    receiverPayload,
    matColor.a,
  );
  r.historyEpoch = currentGrisEpoch;
  // Preserve the receiver as a sampling technique even when every scheduled
  // draw has zero target mass; its M still normalizes other reused candidates.
  r.prefixVertexCount = GI_PREFIX_RECONNECTABLE;

  // Adaptive-sampling tier read at the full-res quad centre. Clamped to
  // [1,4] in case the sample-budget pass emits a bad/uninitialised value
  // (first frame writes vec4u(2,0,0,0) by default). M_GI scales linearly.
  let tier_raw = textureLoad(gi_tier, vec2i(fullPx), 0).r;
  let tier = clamp(tier_raw, 1u, 4u);
  let M_GI = M_GI_BASE * tier / 2u;

  // ── PPG guided-sampling mixing weight (Müller 2017 §3.4) ──────────────────
  // α is the fraction of RIS candidates drawn from the learned dTree. It is
  // ubo.ppgMixAlpha when PPG guided sampling is live (ubo.ppgEnabled == 1) and
  // EXACTLY 0 otherwise. The host writes ppgEnabled=0 / ppgMixAlpha=0 whenever
  // PPG is off (see uboUpdater.ts), so on the PPG-off path:
  //   - the Bernoulli branch below is gated on alpha > 0.0, so NO extra RNG
  //     draw is consumed → the rng stream stays stable, and
  //   - p_src = (1−0)·p_cos = cosθ/π, so the explicit RIS weight uses the
  //     receiver-lobe target divided by the cosine source pdf. For default
  //     diffuse receivers this algebraically reduces to luminance(Lo); rich
  //     material receivers now guide the reservoir by their actual lobes.
  let ppgGuidedOn = ubo.ppgEnabled == 1u;
  let alpha = represented_bernoulli_probability_f32(
    select(0.0, ubo.ppgMixAlpha, ppgGuidedOn),
  );

  for (var i: u32 = 0u; i < M_GI; i = i + 1u) {
    // Draw a candidate direction wi. When alpha > 0, flip a Bernoulli(alpha):
    // heads → sample the learned dTree (guided), tails → cosine hemisphere.
    // When alpha == 0 we take the cosine branch WITHOUT consuming a Bernoulli
    // draw, preserving the exact pre-PPG rng sequence (bit-identity).
    var wi: vec3f;
    if (alpha > 0.0) {
      let bern = rand_f32(&rng);
      if (bern < alpha) {
        // Guided: sample ∝ leaf flux from the dTree for this shading cell.
        wi = ppgSampleGuidedDir(pos, &rng);
      } else {
        wi = sampleCosineHemisphere(normal, &rng);
      }
    } else {
      // Cosine-weighted hemisphere candidate (pre-PPG path, bit-identical).
      wi = sampleCosineHemisphere(normal, &rng);
    }
    let cosTheta = max(0.0, dot(normal, wi));
    if (!reservoirGiFinite(cosTheta) || !(cosTheta > 0.0)) {
      recordInvalidReservoirGICandidate(&r, GI_SAMPLE_SURFACE, currentGrisEpoch);
      continue;
    }

    // Trace from the visible point along wi. Reconnection vertex is the
    // first BVH hit (or a scene-scale-relative sky-miss proxy).
    // WS1 — offset the bounce-ray origin along the GEOMETRIC normal.
    let bounceRay = Ray(pos + geoNormal * walkaroundRayOriginBias(), wi);
    let bounceTrace = traceSceneFirstHitAlphaMaskTexturedWithMetadata(
      ubo.bvhMode, ubo.tlasNodeCount,

      bounceRay, ubo.triIntersectEpsilon,
      bvh_material, BVH_MATERIAL_TEX_WIDTH,
      ubo.frameSeed ^ (i * 0x85ebca6bu) ^ 0x4749424eu,
    );
    let bounceHit = bounceTrace.hit;

    var xs:  vec3f;
    var ns:  vec3f;
    var Lo:  vec3f;
    var sampleKind: u32 = GI_SAMPLE_ENVIRONMENT;
    var sampleFlags: u32 = 0u;

    if (bounceHit.didHit) {
      sampleKind = GI_SAMPLE_SURFACE;
      xs = bounceRay.origin + wi * bounceHit.dist;
      // GRIS's area/solid-angle Jacobian and front/back support use the actual
      // geometric plane. Keep that face-forward geometric normal in the
      // reservoir; normal/bump mapping remains part of the suffix material Lo.
      ns = bounceHit.normal;
      let smoothNs = restir_gi_smooth_normal_for_hit(bounceHit, bounceHit.normal);
      let shadingNs = applyBumpMapForHit(
        bounceHit,
        applyNormalMapForHit(bounceHit, smoothNs),
      );
      // Sample DDGI atlas at the reconnection vertex along its normal —
      // gives the incoming irradiance there. Modulate by the hit surface's
      // material response for outgoing radiance toward the visible pt.
      //
      // Defensive cap on the atlas read.  DDGI probes within ~1 spacing of
      // an area light catch Le directly during the probe trace, so atlas
      // reads of 5..8 are possible — but for a Cornell-scale scene with
      // Le=12, legitimate near-light wall irradiance is also in this band,
      // so we can't reject those samples without truncating real indirect.
      // The cap (Cornell-tuned default 5.0, exposed via ubo.restirGiIrrClamp)
      // admits the realistic indirect range while bounding pathological DDGI
      // atlas readings (which would otherwise produce ~10× per-channel spikes
      // in Lo). The previous tighter reject+cap (>2.0 reject, min 1.0) was
      // over-truncating: the magnitude audit showed it was a 5-10× *under*-
      // energizer of the indirect channel.
      let irrAtXs = min(
        sampleDDGIAtPoint(xs, shadingNs),
        vec3f(ubo.restirGiIrrClamp),
      );
      let xsRmCoord = vec2u(
        bounceHit.indices.w % BVH_MATERIAL_TEX_WIDTH,
        bounceHit.indices.w / BVH_MATERIAL_TEX_WIDTH,
      );
      let xsMaterialWord = textureLoad(bvh_material, vec2i(xsRmCoord), 0).r;
      let xsPayload = sampleRestirGIHitMaterialForHit(
        bounceHit,
        smoothNs,
        shadingNs,
        irrAtXs,
        wi,
        xsMaterialWord,
      );
      Lo = xsPayload.Lo;
      let xsTransmission = clamp(xsPayload.transmission, 0.0, 1.0);
      if (
        !decodeIsUnlitMaterial(xsMaterialWord) &&
        xsTransmission > 0.0
      ) {
        // A transmissive suffix is a stochastic local estimator: the compact
        // reservoir cannot retain its rough-interface draws, spectral medium
        // stack, or terminal receiver. Emission is deterministic. The suffix
        // samples a unit-envelope persistent-reflection family (including the
        // complementary opaque share) against t-weighted transmission, and a
        // paired closed-volume exit never pays authored t a second time.
        sampleFlags = GI_SAMPLE_FLAG_LOCAL_ESTIMATOR;
        let dielectricLo = traceRestirGiDielectricSuffix(
          bounceHit,
          xs,
          bounceHit.dist,
          shadingNs,
          wi,
          xsPayload.emissionLo,
          xsPayload.opaqueLo,
          xsTransmission,
          &rng,
        );
        Lo = dielectricLo;
      }
      if (xsPayload.localSelectedSuffix != 0u) {
        sampleFlags = GI_SAMPLE_FLAG_LOCAL_ESTIMATOR;
      }
    } else {
      // Sky miss — the GI ray escaped the scene. B3: sample the directional IBL
      // map along wi (rotationY-aware) as the reconnection radiance; envRadiance
      // falls back to the scalar skyTint × skyIrradiance with no HDRI bound
      // (no-HDRI byte-identity: the cosine RIS shortcut below is unchanged).
      xs = pos + wi * walkaroundReconnectMaxDistance();
      ns = -wi;
      Lo = envRadiance(wi);
    }

    if (bounceTrace.requiresNativeEstimator != 0u) {
      sampleFlags = GI_SAMPLE_FLAG_LOCAL_ESTIMATOR;
    }

    // Evaluate the same mapped receiver lobes that shade will consume. The
    // diffuse-default path still reduces to luminance(Lo) * cos(theta) / pi,
    // while glossy/metal/clearcoat/sheen receivers retain their actual target.
    // The first-hit walker has already realized every alpha-mask, sidedness,
    // and partial-blend occupancy decision on pos -> xs (or sky).
    // Retracing deterministic transmittance here would pay skipped blend
    // coverage a second time. Shifted GRIS evaluation owns its own expected
    // visibility in grisProxyVisibilityAt instead.
    let candidateVisibility: f32 = 1.0;
    var receiverLo = Lo;
    if (sampleKind == GI_SAMPLE_ENVIRONMENT) {
      receiverLo = walkaroundScaleEnvironmentRadiance(
        receiverLo,
        receiverPayload.envMapIntensity,
      );
    }
    let receiverPHat = restir_gi_receiver_phat_from_payload(
      pos,
      normal,
      receiverPayload.clearcoatNormal,
      safe_normalize(-primaryRay.direction),
      receiverPayload,
      matColor.a,
      xs,
      receiverLo,
    );
    let logPHat = reservoirGiLogPositiveProduct(receiverPHat, candidateVisibility);
    if (!reservoirGiValidLog(logPHat)) {
      recordInvalidReservoirGICandidate(&r, sampleKind, currentGrisEpoch);
      continue;
    }
    // RIS candidate weight w = p̂ / p_src.
    //
    // ppg-OFF (alpha == 0): the RIS source pdf is the pure cosine pdf
    //   p_src = cosθ/π. Diffuse defaults still reduce to luminance(Lo);
    //   rich-material receivers keep the full p̂ / p_src ratio.
    //
    // ppg-ON (alpha > 0): the RIS source pdf is the DEFENSIVE MIXTURE
    //   (Müller §3.4)   p_src = α·p_guide(wi) + (1−α)·p_cos(wi)
    //   evaluated for WHICHEVER wi was chosen (guided OR cosine). Evaluating
    //   BOTH pdfs for the chosen direction is what keeps the mixture estimator
    //   unbiased — we cannot reuse the cosine shortcut because the cosθ no
    //   longer cancels against a pure-cosine denominator.
    //     p_cos   = cosθ/π                    (cosine-hemisphere solid-angle pdf)
    //     p_guide = ppgEvalPdf(pos, wi)       (dTree solid-angle pdf; mirrors
    //               the CPU dTreePdf in dTree.ts exactly)
    var logPSrc: f32;
    let pCos = cosTheta * INV_PI;
    if (alpha > 0.0) {
      let pGuide = ppgEvalPdf(pos, wi);
      logPSrc = reservoirGiLogProposalMixture(alpha, pGuide, pCos);
    } else {
      logPSrc = reservoirGiLogPositive(pCos);
    }
    if (!reservoirGiValidLog(logPSrc)) {
      recordInvalidReservoirGICandidate(&r, sampleKind, currentGrisEpoch);
      continue;
    }
    let logWeight = logPHat - logPSrc;
    if (!reservoirGiValidLog(logWeight)) {
      recordInvalidReservoirGICandidate(&r, sampleKind, currentGrisEpoch);
      continue;
    }
    updateReservoirGIWithMetadata(
      &r, &giWrs, xs, ns, Lo, sampleKind, wi,
      sampleFlags,
      logPHat, candidateVisibility, currentGrisEpoch, logWeight, &rng,
    );
  }

  finaliseGIReservoirFromNativeWrs(&r, giWrs, ubo.restirGiWCap);

  // ── Generalized-reuse producer metadata (Lin et al. 2022, §5) ────────────
  // The metadata is consumed by the canonical temporal/spatial passes and
  // by the final shading visibility gate. refreshGrisMetadata derives the
  // reconnection direction, distance, outgoing cosine, and prefix count from
  // the selected edge; the winning candidate's sample kind, native target,
  // visibility, and history epoch remain intact for transformed-density reuse.
  refreshGrisMetadata(&r);

  storeReservoirGI_rw(pixelIdxGi, r);
}
`;

const RESTIR_GI_DIELECTRIC_SUFFIX_START =
  'const RESTIR_GI_SUFFIX_MAX_INTERFACES: u32 = 8u;';
const RESTIR_GI_DIELECTRIC_SUFFIX_END =
  '// sampleCosineHemisphere is the canonical helper';
const restirGiDielectricSuffixStart = RIS_GI_WGSL.indexOf(
  RESTIR_GI_DIELECTRIC_SUFFIX_START,
);
const restirGiDielectricSuffixEnd = RIS_GI_WGSL.indexOf(
  RESTIR_GI_DIELECTRIC_SUFFIX_END,
  restirGiDielectricSuffixStart,
);
if (
  restirGiDielectricSuffixStart < 0 ||
  restirGiDielectricSuffixEnd <= restirGiDielectricSuffixStart
) {
  throw new Error('RIS-GI dielectric suffix fragment markers are missing or reordered');
}

/**
 * Exact byte slice of the ordinary RIS-GI dielectric suffix implementation.
 * NRC embeds this same implementation instead of maintaining a second glass
 * state machine. Keeping the canonical body inside `RIS_GI_WGSL` preserves the
 * default shader byte-for-byte; marker validation fails module loading if a
 * future edit accidentally makes the reusable boundary ambiguous.
 */
export const RESTIR_GI_DIELECTRIC_SUFFIX_WGSL = RIS_GI_WGSL.slice(
  restirGiDielectricSuffixStart,
  restirGiDielectricSuffixEnd,
);

/** W1-R6 — declarative include-graph entry.
 *  T9-stepC — narrowed from `['common', 'ddgiSample']` to the modules this
 *  half-res GI-RIS pass actually references:
 *    - `WalkaroundUBO` / `INV_PI`            → walkaroundUbo
 *    - primary cast (`traceScene*` / `BVHNode` / `Ray`) → sceneTraversal
 *    - `ReservoirGI` / `emptyReservoirGI` / `updateReservoirGIWithMetadata` /
 *      `storeReservoirGI_rw`                 → reservoirGi
 *    - `pcgInit` / `luminance` / `sampleCosineHemisphere` → sharedPrimitives
 *    - `decodeMaterialColor` / `decodeIsMetal` / `decodeRoughMetal` / `decodeIor`
 *      / `BVH_MATERIAL_TEX_WIDTH`              → materialDecode
 *    - `invertMat4_common` / `generatePrimaryRay_common` → cameraRays
 *    - `ddgiSample`                          → ddgiSample
 *  Drops emitterSampling / welfordTail (unused).
 *  ReSTIR-GI material parity adds `restirGiMaterial` (normal/bump maps, mapped
 *  base color, rough/metal, and extension-aware suffix radiance).
 *  W9 guided sampling — adds `ppgPdf` (declares the group(3) PPG tree buffers
 *  + provides ppgEvalPdf / ppgSampleGuidedDir). Listed AFTER sharedPrimitives
 *  so `rand_f32` is defined before ppgPdf's source, and after ddgiSample so
 *  the group(3) DDGI bindings (0-3) precede the PPG ones (6-8).
 *  Verified complete by the static ident-resolution gate. */
export const RIS_GI_MODULE: WgslModule = {
  name: 'risGi',
  source: RIS_GI_WGSL,
  // D5.1+D5.2: ddgiSample replaced by ddgiGridUbo (which requires ddgiSample
  // transitively, and adds the DDGIGridUBO struct + binding + sampleDDGIAtPoint).
  requires: ['walkaroundUbo', 'sceneTraversal', 'reservoirGi', 'sharedPrimitives', 'materialDecode', 'materialAtlas', 'surfaceTextures', 'restirGiMaterial', 'cameraRays', 'ddgiGridUbo', 'ppgPdf', 'environmentSample'],
};
