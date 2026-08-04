/**
 * Independent path-suffix teacher for Neural Radiance Caching.
 *
 * This is a raw WGSL fragment because it deliberately consumes the bindings
 * declared by the NRC GI-RIS pass.  Unlike the cache query, the teacher never
 * reads DDGI or the cache itself: it traces a bounded continuation path,
 * evaluates mapped materials, performs next-event estimation for every finite
 * emitter plus analytic and directional lights, and uses Russian roulette.
 *
 * Müller et al. 2021 train NRC from independently traced suffix radiance.  A
 * same-candidate DDGI label is only distillation and cannot improve on DDGI.
 */

import { analyticLightFalloffWgsl } from './analyticLightFalloff.wgsl.js';

export const NRC_INDEPENDENT_SUFFIX_WGSL = /* wgsl */ `

${analyticLightFalloffWgsl('nrc_teacher')}

// The target is the radiance of this explicitly bounded four-vertex suffix.
// Direct lighting is evaluated at the terminal vertex, so the only omitted
// terms are paths requiring more than three continuation edges.
const NRC_TEACHER_MAX_VERTICES: u32 = 4u;
const NRC_TEACHER_RR_START: u32 = 2u;
const NRC_TEACHER_SPECULAR_MIX: f32 = 0.5;
const NRC_TEACHER_MEDIUM_CAPACITY: u32 = 4u;
const NRC_TEACHER_SHEET_MAX_INTERFACES: u32 = 8u;
const NRC_TEACHER_DIELECTRIC_EVENT_INVALID: u32 = 0u;
const NRC_TEACHER_DIELECTRIC_EVENT_REFLECTION: u32 = 1u;
const NRC_TEACHER_DIELECTRIC_EVENT_TRANSMISSION: u32 = 2u;

struct NrcTeacherAnalyticAliasDraw {
  index: u32,
  pmf: f32,
}

fn nrcTeacherAnalyticAliasColumn(
  count: u32,
  rng: ptr<function, u32>,
) -> u32 {
  let threshold = ((0xffffffffu % count) + 1u) % count;
  var word = pcgNext(rng);
  loop {
    if (word >= threshold) { return word % count; }
    word = pcgNext(rng);
  }
  return 0u;
}

fn nrcTeacherAnalyticAliasDraw(
  count: u32,
  aliasOffset: u32,
  dims: vec2u,
  rng: ptr<function, u32>,
) -> NrcTeacherAnalyticAliasDraw {
  let column = nrcTeacherAnalyticAliasColumn(count, rng);
  let coord = aliasOffset + column;
  let entry = textureLoad(analytic_lights, vec2i(i32(coord % dims.x), i32(coord / dims.x)), 0);
  let aliasIndex = bitcast<u32>(entry.y);
  let selected = select(aliasIndex, column, rand_f32(rng) < entry.x);
  let selectedCoord = aliasOffset + selected;
  let selectedEntry = textureLoad(
    analytic_lights,
    vec2i(i32(selectedCoord % dims.x), i32(selectedCoord / dims.x)),
    0,
  );
  var draw: NrcTeacherAnalyticAliasDraw;
  draw.index = selected;
  draw.pmf = selectedEntry.z;
  return draw;
}

fn nrcTeacherMax3(v: vec3f) -> f32 {
  return max(v.x, max(v.y, v.z));
}

fn nrcTeacherFinite3(v: vec3f) -> bool {
  return all(v == v) && nrcTeacherMax3(abs(v)) < 1e30;
}

fn nrcTeacherTransferMax(transfer: mat3x3f) -> f32 {
  return max(
    nrcTeacherMax3(abs(transfer[0])),
    max(
      nrcTeacherMax3(abs(transfer[1])),
      nrcTeacherMax3(abs(transfer[2])),
    ),
  );
}

fn nrcTeacherChannel(value: vec3f, channel: u32) -> f32 {
  if (channel == 0u) { return value.r; }
  if (channel == 1u) { return value.g; }
  return value.b;
}

// Evaluate the persistent rough-dielectric interface with exact unpolarised
// Fresnel (or the authored thin-film response) instead of the Schlick F0 used
// by the opaque material model. This is f*cos at the receiver. The interface
// remains below the authored sheen/clearcoat layers and therefore pays their
// base-lobe attenuation, but it does not pay thin-film transmittance: a film's
// reflectance is already the absolute reflected share.
struct NrcTeacherInterfaceReflection {
  exact: vec3f,
  canonicalSchlick: vec3f,
  exactFresnel: vec3f,
  canonicalFresnel: vec3f,
  reflectionScale: f32,
  microfacetCos: f32,
}

fn nrcTeacherExactInterfaceReflection(
  hit: IntersectionResult,
  payload: RestirDIMaterialPayload,
  normal: vec3f,
  wo: vec3f,
  wi: vec3f,
  etaIncident: vec3f,
  etaTarget: vec3f,
  dielectricInterface: bool,
) -> NrcTeacherInterfaceReflection {
  var out: NrcTeacherInterfaceReflection;
  out.exact = vec3f(0.0);
  out.canonicalSchlick = vec3f(0.0);
  out.exactFresnel = vec3f(0.0);
  out.canonicalFresnel = vec3f(0.0);
  out.reflectionScale = 0.0;
  out.microfacetCos = 0.0;
  if (!dielectricInterface) { return out; }
  let nDotWo = dot(normal, wo);
  let nDotWi = dot(normal, wi);
  if (!(nDotWo > 0.0) || !(nDotWi > 0.0)) {
    return out;
  }
  let halfVector = safe_normalize(wo + wi);
  let woDotM = dot(wo, halfVector);
  if (!(woDotM > 0.0)) { return out; }
  var exactReflectance = vec3f(1.0) - dielectricInterfaceTransmissionRgb(
    woDotM, etaIncident, etaTarget,
  );
  let film = materialThinFilmResponse(
    hit.indices.w, hit.side >= 0.0, woDotM,
  );
  if (film.present != 0u) {
    exactReflectance = film.reflectance;
  }
  let etaRatio = max(etaIncident, vec3f(1e-6)) /
    max(etaTarget, vec3f(1e-6));
  let sin2Target = etaRatio * etaRatio *
    (1.0 - woDotM * woDotM);
  exactReflectance = select(
    clamp(exactReflectance, vec3f(0.0), vec3f(1.0)),
    vec3f(1.0),
    sin2Target >= vec3f(1.0),
  );

  let continuousRoughness = boundedContinuousGgxRoughness(payload.rough);
  let aniso = clamp(payload.anisotropy.x, 0.0, 1.0);
  var D = 0.0;
  var G = 0.0;
  if (aniso > 0.0) {
    let frame = anisotropyTangentFrameFromBasis(
      normal,
      payload.anisotropyTangent,
      payload.anisotropyBitangent,
      payload.anisotropy.y,
    );
    let axes = anisotropyAxes(continuousRoughness, aniso);
    D = distributionGGXAnisotropic(
      normal, frame[0], frame[1], halfVector, axes.x, axes.y,
    );
    G = geometrySmithGGXAnisotropic(
      normal, frame[0], frame[1], wo, wi, axes.x, axes.y,
    );
  } else {
    D = distributionGGX(
      max(0.0, dot(normal, halfVector)), continuousRoughness,
    );
    G = geometrySmith(nDotWo, nDotWi, continuousRoughness);
  }
  if (!(D > 0.0) || !(G > 0.0)) { return out; }

  let sheenAttenuation = sheenBaseAttenuation(
    payload.sheen.a,
    payload.sheenRoughness,
    payload.sheen.rgb,
    normal,
    wo,
    wi,
  );
  let clearcoatAttenuation = clearcoatBaseAttenuation(
    payload.clearcoat.x,
    payload.clearcoatNormal,
    wo,
  );
  let etaSum = max(etaIncident + etaTarget, vec3f(1e-6));
  let etaDelta = (etaTarget - etaIncident) / etaSum;
  let canonicalF0 = clamp(
    etaDelta * etaDelta, vec3f(0.0), vec3f(1.0),
  );
  let reflectionScale = (D * G / (4.0 * nDotWo)) *
    sheenAttenuation * clearcoatAttenuation;
  let canonicalFresnel = fresnelSchlick(woDotM, canonicalF0);
  out.exact = exactReflectance * reflectionScale;
  out.canonicalSchlick = canonicalFresnel * reflectionScale;
  out.exactFresnel = exactReflectance;
  out.canonicalFresnel = canonicalFresnel;
  out.reflectionScale = reflectionScale;
  out.microfacetCos = woDotM;
  return out;
}

fn nrcTeacherMaterialResponse(
  hit: IntersectionResult,
  payload: RestirDIMaterialPayload,
  normal: vec3f,
  wo: vec3f,
  wi: vec3f,
  transmission: f32,
  etaIncident: vec3f,
  etaTarget: vec3f,
  dielectricInterface: bool,
) -> vec3f {
  if (dielectricInterface) {
    // The opaque evaluator owns authored conductor/specular/iridescence,
    // diffuse, sheen, and clearcoat controls. Remove only the canonical base
    // dielectric lobe that the exact interface replaces. A signed relative
    // residual lets a specular extension deliberately reduce as well as
    // increase the opaque-share response without making the total negative.
    let authoredSpecular = sampleSpecularControls(hit);
    let authoredIridescence = sampleIridescenceControls(hit);
    let opaqueClosure =
      evalGGXWithSpecularClearcoatSheenWithAnisotropyFrame(
        payload.albedo,
        payload.rough,
        payload.metal,
        authoredSpecular.rgb,
        authoredSpecular.a,
        payload.anisotropy.x,
        payload.anisotropy.y,
        authoredIridescence,
        payload.clearcoat.x,
        payload.clearcoat.y,
        payload.sheen.a,
        payload.sheenRoughness,
        payload.sheen.rgb,
        payload.anisotropyTangent,
        payload.anisotropyBitangent,
        normal,
        payload.clearcoatNormal,
        wo,
        wi,
      );
    let interfaceReflection = nrcTeacherExactInterfaceReflection(
      hit,
      payload,
      normal,
      wo,
      wi,
      etaIncident,
      etaTarget,
      true,
    );
    let authoredF0 = iridescenceModifiedF0(
      materialF0(
        payload.albedo,
        payload.metal,
        authoredSpecular.rgb,
        authoredSpecular.a,
      ),
      authoredIridescence,
      interfaceReflection.microfacetCos,
    );
    let authoredFresnel = fresnelSchlick(
      interfaceReflection.microfacetCos, authoredF0,
    );
    let canonicalSupport =
      interfaceReflection.canonicalFresnel > vec3f(1e-6);
    let authoredBase = select(
      vec3f(0.0),
      authoredFresnel * interfaceReflection.reflectionScale,
      canonicalSupport,
    );
    // Express custom specular/conductor/iridescence as a signed relative
    // change to the exact interface rather than subtracting an absolute bare
    // Schlick lobe. This retains authored reductions without allowing them to
    // drive a destructive-interference film below zero. Where canonical IOR
    // has no support, the authored lobe remains in the opaque body instead.
    let authoredExact = select(
      interfaceReflection.exact,
      interfaceReflection.exactFresnel * authoredFresnel /
        max(interfaceReflection.canonicalFresnel, vec3f(1e-6)) *
        interfaceReflection.reflectionScale,
      canonicalSupport,
    );
    let signedInterfaceResidual = authoredExact -
      interfaceReflection.exact;
    let opaqueBody = opaqueClosure - authoredBase;
    return interfaceReflection.exact *
        payload.reflectionLayerTransmission +
      (opaqueBody + signedInterfaceResidual) *
        (1.0 - clamp(transmission, 0.0, 1.0)) *
        payload.layerTransmission;
  }

  let mixedClosure = evalGGXReflectionWithTransmissionMix(
    payload.albedo,
    payload.rough,
    payload.metal,
    payload.specular.rgb,
    payload.specular.a,
    payload.anisotropy.x,
    payload.anisotropy.y,
    payload.iridescence,
    payload.clearcoat.x,
    payload.clearcoat.y,
    payload.sheen.a,
    payload.sheenRoughness,
    payload.sheen.rgb,
    payload.anisotropyTangent,
    payload.anisotropyBitangent,
    normal,
    payload.clearcoatNormal,
    wo,
    wi,
    transmission,
  );
  let reflectionClosure =
    evalGGXSpecularOnlyWithSpecularClearcoatSheenWithAnisotropyFrame(
      payload.albedo,
      payload.rough,
      payload.metal,
      payload.specular.rgb,
      payload.specular.a,
      payload.anisotropy.x,
      payload.anisotropy.y,
      payload.iridescence,
      payload.clearcoat.x,
      payload.clearcoat.y,
      payload.sheen.a,
      payload.sheenRoughness,
      payload.sheen.rgb,
      payload.anisotropyTangent,
      payload.anisotropyBitangent,
      normal,
      payload.clearcoatNormal,
      wo,
      wi,
    );
  return applyMaterialLayerTransmissionToBrdf(
    mixedClosure,
    reflectionClosure,
    payload.layerTransmission,
    payload.reflectionLayerTransmission,
  );
}

fn nrcTeacherDirectionalIncidentResponse(
  hit: IntersectionResult,
  payload: RestirDIMaterialPayload,
  normal: vec3f,
  wo: vec3f,
  wi: vec3f,
  transmission: f32,
  etaIncident: vec3f,
  etaTarget: vec3f,
  dielectricInterface: bool,
  incidentRadiance: vec3f,
  explicitSegmentVolume: bool,
) -> vec3f {
  let rawResponse = incidentRadiance * nrcTeacherMaterialResponse(
    hit, payload, normal, wo, wi, transmission,
    etaIncident, etaTarget, dielectricInterface,
  );
  let proxyResponse = applyHomogeneousVolumeSingleScatterDirectional(
    rawResponse,
    payload.albedo,
    payload.volumeScattering,
    payload.bulkThickness,
    normal,
    wo,
    wi,
  );
  return select(proxyResponse, rawResponse, explicitSegmentVolume);
}

fn nrcTeacherShadowTint(
  pos: vec3f,
  geoNormal: vec3f,
  wi: vec3f,
  tMax: f32,
  initialMediumState: MaterialShadowMediumState,
) -> vec3f {
  let offsetNormal = select(-geoNormal, geoNormal, dot(geoNormal, wi) >= 0.0);
  return traceSceneAlphaTintTransmittanceTexturedWithState(
    ubo.bvhMode, ubo.tlasNodeCount,

    pos + offsetNormal * walkaroundRayOriginBias(), wi, tMax, ubo.triIntersectEpsilon,
    bvh_material, BVH_MATERIAL_TEX_WIDTH, bvh_beer,
    initialMediumState, false,
  );
}

// One exact-CDF area-light sample.  The source density is converted from area
// to solid angle by G = cos(light)/distance^2, leaving f*cos at the receiver.
fn nrcTeacherAreaNee(
  hit: IntersectionResult,
  pos: vec3f,
  geoNormal: vec3f,
  normal: vec3f,
  wo: vec3f,
  payload: RestirDIMaterialPayload,
  transmission: f32,
  etaIncident: vec3f,
  etaTarget: vec3f,
  dielectricInterface: bool,
  explicitSegmentVolume: bool,
  shadowMediumState: MaterialShadowMediumState,
  rng: ptr<function, u32>,
) -> vec3f {
  let count = min(ubo.emitterCount, sceneEmitterCount());
  if (count == 0u) { return vec3f(0.0); }

  let lid = sampleEmitterIdx(count, rand_f32(rng));
  let selectionPmf = emitterCdfPmf(count, lid);
  if (!(selectionPmf > 0.0)) { return vec3f(0.0); }
  let xi = vec2f(rand_f32(rng), rand_f32(rng));
  let emitter = sceneLoadEmitter(lid);
  let ls = sampleEmitterPoint(emitter, xi);
  let toLight = ls.pos - pos;
  let dist = safe_length(toLight);
  if (!(dist > 0.0)) { return vec3f(0.0); }
  let dist2 = dist * dist;
  let wi = safe_normalize(toLight);
  let cosLight = emitterTriCosineTowardReceiver(emitter, -wi);
  if (cosLight <= 0.0 || dot(normal, wi) <= 0.0) {
    return vec3f(0.0);
  }

  var visibility = vec3f(1.0);
  if (!emitterTriCastShadowDisabled(emitter)) {
    visibility = nrcTeacherShadowTint(
      pos, geoNormal, wi, max(0.0, dist - walkaroundRayEndMargin()),
      shadowMediumState,
    );
    if (nrcTeacherMax3(visibility) <= 0.0) { return vec3f(0.0); }
  }

  let pdfArea = selectionPmf * ls.pdfArea;
  let geometry = emitterGeometry(cosLight, dist2, ubo.emitterDist2Floor);
  let Le = sampleEmitterLeAtXi(emitter, xi);
  if (!(pdfArea > 0.0)) { return vec3f(0.0); }
  return nrcTeacherDirectionalIncidentResponse(
    hit, payload, normal, wo, wi, transmission,
    etaIncident, etaTarget, dielectricInterface,
    Le * visibility * geometry / pdfArea,
    explicitSegmentVolume,
  );
}

// Point/spot emitters are delta lights, hence the deterministic sum is already
// their exact direct-light estimator and needs no sampling-pdf division.
fn nrcTeacherAnalyticNee(
  hit: IntersectionResult,
  pos: vec3f,
  geoNormal: vec3f,
  normal: vec3f,
  wo: vec3f,
  payload: RestirDIMaterialPayload,
  transmission: f32,
  etaIncident: vec3f,
  etaTarget: vec3f,
  dielectricInterface: bool,
  explicitSegmentVolume: bool,
  shadowMediumState: MaterialShadowMediumState,
  rng: ptr<function, u32>,
) -> vec3f {
  let dims = textureDimensions(analytic_lights);
  let texelCount = dims.x * dims.y;
  let header = textureLoad(analytic_lights, vec2i(0, 0), 0);
  let count = u32(max(header.x, 0.0));
  let aliasOffset = u32(max(header.y, 0.0));
  if (
    count == 0u ||
    aliasOffset != 1u + count * 4u ||
    aliasOffset + count > texelCount
  ) { return vec3f(0.0); }
  var direct = vec3f(0.0);
  let sampleCount = min(count, 4u);
  for (var sampleIndex = 0u; sampleIndex < sampleCount; sampleIndex = sampleIndex + 1u) {
    var li = sampleIndex;
    var estimatorWeight = 1.0;
    if (count > 4u) {
      let draw = nrcTeacherAnalyticAliasDraw(count, aliasOffset, dims, rng);
      if (draw.pmf <= 0.0) { continue; }
      li = draw.index;
      estimatorWeight = 1.0 / (f32(sampleCount) * draw.pmf);
    }
    let base = 1u + li * 4u;
    let l0 = textureLoad(analytic_lights, vec2i(i32(base % dims.x), i32(base / dims.x)), 0);
    let l1 = textureLoad(analytic_lights, vec2i(i32((base + 1u) % dims.x), i32((base + 1u) / dims.x)), 0);
    let l2 = textureLoad(analytic_lights, vec2i(i32((base + 2u) % dims.x), i32((base + 2u) / dims.x)), 0);
    let l3 = textureLoad(analytic_lights, vec2i(i32((base + 3u) % dims.x), i32((base + 3u) / dims.x)), 0);
    let toLight = l0.xyz - pos;
    let dist = safe_length(toLight);
    if (!(dist > 0.0)) { continue; }
    let wi = safe_normalize(toLight);
    if (dot(normal, wi) <= 0.0) { continue; }
    let cone = nrc_teacherSpotConeFalloff(l2.xyz, wi, l2.w, l3.x);
    if (cone <= 0.0) { continue; }
    var visibility = vec3f(1.0);
    if (l3.y <= 0.5) {
      visibility = nrcTeacherShadowTint(
        pos, geoNormal, wi, max(0.0, dist - walkaroundRayEndMargin()),
        shadowMediumState,
      );
      if (nrcTeacherMax3(visibility) <= 0.0) { continue; }
    }
    let attenuation = nrc_teacherPointSpotAttenuation(
      dist, l3.z, l3.w, ubo.emitterDist2Floor,
    );
    direct = direct + nrcTeacherDirectionalIncidentResponse(
      hit, payload, normal, wo, wi, transmission,
      etaIncident, etaTarget, dielectricInterface,
      l1.xyz * visibility * cone * attenuation * estimatorWeight,
      explicitSegmentVolume,
    );
  }
  return direct;
}

// The renderer interprets sunIntensity as integrated directional irradiance.
// Sampling its authored angular disc therefore estimates its directional
// average directly (no solid-angle division), matching the live shade model.
fn nrcTeacherSunNee(
  hit: IntersectionResult,
  pos: vec3f,
  geoNormal: vec3f,
  normal: vec3f,
  wo: vec3f,
  payload: RestirDIMaterialPayload,
  transmission: f32,
  etaIncident: vec3f,
  etaTarget: vec3f,
  dielectricInterface: bool,
  explicitSegmentVolume: bool,
  shadowMediumState: MaterialShadowMediumState,
  rng: ptr<function, u32>,
) -> vec3f {
  if (!(ubo.sunIntensity > 0.0)) { return vec3f(0.0); }
  let sunBase = safe_normalize(ubo.sunDirection);
  let up = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), abs(sunBase.y) < 0.99);
  let tangent = safe_normalize(cross(up, sunBase));
  let bitangent = cross(sunBase, tangent);
  let radius = max(ubo.sunAngular.x, 0.0) * sqrt(rand_f32(rng));
  let phi = 2.0 * PI * rand_f32(rng);
  let wi = safe_normalize(sunBase + radius * (cos(phi) * tangent + sin(phi) * bitangent));
  if (dot(normal, wi) <= 0.0) { return vec3f(0.0); }
  var visibility = vec3f(1.0);
  if ((ubo.stainedGlassFlags & SHADE_FLAG_DIRECT_SUN_SHADOW_DISABLED) == 0u) {
    visibility = nrcTeacherShadowTint(
      pos, geoNormal, wi, INFINITY, shadowMediumState,
    );
    if (nrcTeacherMax3(visibility) <= 0.0) { return vec3f(0.0); }
  }
  return nrcTeacherDirectionalIncidentResponse(
    hit, payload, normal, wo, wi, transmission,
    etaIncident, etaTarget, dielectricInterface,
    vec3f(ubo.sunIntensity) * visibility,
    explicitSegmentVolume,
  );
}

struct NrcTeacherDielectricLobe {
  direction: vec3f,
  weightRgb: vec3f,
  microfacetCos: f32,
  kind: u32,
  tir: u32,
  valid: u32,
}

// Conditional rough-dielectric lobe using one canonical Heitz visible normal
// and exact dielectric Fresnel. The ordinary outer estimator leaves TIR in its
// same-side family. A reciprocal sheet's internal walk instead sets
// collapseTirToReflection: both discrete requests then denote the same reflected
// event, whose effective discrete probability is one rather than one half.
fn nrcTeacherSampleDielectricLobe(
  hit: IntersectionResult,
  normal: vec3f,
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
  collapseTirToReflection: bool,
  rng: ptr<function, u32>,
) -> NrcTeacherDielectricLobe {
  var out: NrcTeacherDielectricLobe;
  out.direction = vec3f(0.0);
  out.weightRgb = vec3f(0.0);
  out.microfacetCos = 0.0;
  out.kind = NRC_TEACHER_DIELECTRIC_EVENT_INVALID;
  out.tir = 0u;
  out.valid = 0u;

  let etaIncident = nrcTeacherChannel(etaIncidentRgb, channel);
  let etaTarget = nrcTeacherChannel(etaTargetRgb, channel);
  let nDotWo = dot(normal, wo);
  if (!(nDotWo > 0.0) || etaIncident <= 0.0 || etaTarget <= 0.0) {
    return out;
  }

  let authoredRoughness = clamp(rough, 0.0, 1.0);
  let aniso = clamp(anisotropy, 0.0, 1.0);
  var wm = normal;
  if (authoredRoughness > 0.0) {
    if (aniso > 0.0) {
      let frame = anisotropyTangentFrameFromBasis(
        normal,
        anisotropyTangent,
        anisotropyBitangent,
        anisotropyRotation,
      );
      let axes = ggxDielectricTransmissionAxes(authoredRoughness, aniso);
      let woT = vec3f(
        dot(wo, frame[0]), dot(wo, frame[1]), dot(wo, normal),
      );
      let wmT = ggxSampleVndfTangentAnisotropic(
        woT, axes.x, axes.y, rng,
      );
      wm = safe_normalize(
        wmT.x * frame[0] + wmT.y * frame[1] + wmT.z * normal,
      );
    } else {
      let reflectedProposal = ggxSampleVndf(
        normal, wo, authoredRoughness, rng,
      );
      wm = safe_normalize(wo + reflectedProposal);
      if (dot(wm, normal) < 0.0) { wm = -wm; }
    }
  }

  let woDotM = dot(wo, wm);
  if (!(woDotM > 0.0)) { return out; }
  let reflectedDirection = safe_normalize(reflect(-wo, wm));
  let refractedRaw = refract(-wo, wm, etaIncident / etaTarget);
  let tir = dot(refractedRaw, refractedRaw) <= 1e-12;
  out.tir = select(0u, 1u, tir);

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
  let reflectance = nrcTeacherChannel(reflectanceRgb, channel);
  let transmittance = nrcTeacherChannel(transmittanceRgb, channel);

  var sampleReflection = chooseReflection;
  if (tir && collapseTirToReflection) {
    sampleReflection = true;
  }
  var directionalBaseWeight = 0.0;
  if (sampleReflection) {
    let nDotWi = dot(normal, reflectedDirection);
    if (!(nDotWi > 0.0) || !(reflectance > 0.0)) { return out; }
    if (authoredRoughness <= 0.0) {
      directionalBaseWeight = 1.0;
    } else {
      var D = 0.0;
      var G1o = 0.0;
      var G1i = 0.0;
      if (aniso > 0.0) {
        let frame = anisotropyTangentFrameFromBasis(
          normal,
          anisotropyTangent,
          anisotropyBitangent,
          anisotropyRotation,
        );
        let axes = ggxDielectricTransmissionAxes(
          authoredRoughness, aniso,
        );
        D = distributionGGXAnisotropic(
          normal, frame[0], frame[1], wm, axes.x, axes.y,
        );
        G1o = geometrySmithGGXAnisotropicG1(
          normal, frame[0], frame[1], wo, axes.x, axes.y,
        );
        G1i = geometrySmithGGXAnisotropicG1(
          normal, frame[0], frame[1], reflectedDirection, axes.x, axes.y,
        );
      } else {
        let alpha = authoredRoughness * authoredRoughness;
        let alpha2 = alpha * alpha;
        D = distributionGGX(dot(normal, wm), authoredRoughness);
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
    out.kind = NRC_TEACHER_DIELECTRIC_EVENT_REFLECTION;
    out.weightRgb = reflectanceRgb * directionalBaseWeight;
  } else {
    if (tir || !(transmittance > 0.0)) { return out; }
    let refractedDirection = safe_normalize(refractedRaw);
    let nDotWiAbs = abs(dot(normal, refractedDirection));
    let wiDotM = dot(refractedDirection, wm);
    let etap = etaTarget / etaIncident;
    let denom = wiDotM + woDotM / etap;
    if (
      dot(normal, refractedDirection) >= 0.0 || !(nDotWiAbs > 0.0) ||
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
          normal,
          anisotropyTangent,
          anisotropyBitangent,
          anisotropyRotation,
        );
        let axes = ggxDielectricTransmissionAxes(
          authoredRoughness, aniso,
        );
        D = distributionGGXAnisotropic(
          normal, frame[0], frame[1], wm, axes.x, axes.y,
        );
        G1o = geometrySmithGGXAnisotropicG1(
          normal, frame[0], frame[1], wo, axes.x, axes.y,
        );
        G1i = geometrySmithGGXAnisotropicG1(
          normal, frame[0], frame[1], -refractedDirection, axes.x, axes.y,
        );
      } else {
        let alpha = authoredRoughness * authoredRoughness;
        let alpha2 = alpha * alpha;
        D = distributionGGX(dot(normal, wm), authoredRoughness);
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
    out.kind = NRC_TEACHER_DIELECTRIC_EVENT_TRANSMISSION;
    out.weightRgb = transmittanceRgb * directionalBaseWeight;
  }

  let selectedWeight = nrcTeacherChannel(out.weightRgb, channel);
  if (!(selectedWeight > 0.0) || nrcTeacherMax3(out.weightRgb) >= 1e30) {
    out.weightRgb = vec3f(0.0);
    out.kind = NRC_TEACHER_DIELECTRIC_EVENT_INVALID;
    return out;
  }
  out.microfacetCos = woDotM;
  out.valid = 1u;
  return out;
}

// Raw bvh_beer reference. The actual exit-face thickness texel caps a positive
// authored distance; synthetic zero-thickness bulk uses the complete closed
// segment and intentionally ignores that map.
fn nrcTeacherMappedBeerReference(hit: IntersectionResult) -> vec3f {
  let coord = vec2u(
    hit.indices.w % BVH_BEER_TEX_WIDTH,
    hit.indices.w / BVH_BEER_TEX_WIDTH,
  );
  let packed = textureLoad(bvh_beer, vec2i(coord), 0).r;
  let tint = vec3f(
    f32((packed >> 24u) & 0xffu),
    f32((packed >> 16u) & 0xffu),
    f32((packed >> 8u) & 0xffu),
  ) / 255.0;
  return tint;
}

fn nrcTeacherThicknessMapScale(hit: IntersectionResult) -> f32 {
  let thicknessMap = sampleMaterialAtlasRawAtOffset(
    hit.indices.w,
    MATERIAL_MAP_THICKNESS_TEXEL_OFFSET,
    hit.uv,
    materialAtlasUv1ForHit(hit),
  );
  if (thicknessMap.valid == 0u) { return 1.0; }
  return materialOpticalThicknessMapScale(hit.indices.w, thicknessMap.value.g);
}

fn nrcTeacherBeerForSegment(
  triIndex: u32,
  mappedReferenceTint: vec3f,
  authoredThickness: f32,
  thicknessMapScale: f32,
  segmentLength: f32,
) -> vec3f {
  if (segmentLength <= 0.0) { return vec3f(1.0); }
  let hasAuthoredThickness = authoredThickness > 0.0;
  let referenceThickness = select(1.0, authoredThickness, hasAuthoredThickness);
  let transportDistance = select(
    segmentLength,
    min(
      segmentLength,
      referenceThickness * clamp(thicknessMapScale, 0.0, 1.0),
    ),
    hasAuthoredThickness,
  );
  let rgbBeer = pow(
    clamp(mappedReferenceTint, vec3f(0.0), vec3f(1.0)),
    vec3f(transportDistance / referenceThickness),
  );
  return materialSpectralAttenuation(
    triIndex,
    transportDistance,
    rgbBeer,
  );
}

fn nrcTeacherTransportDistance(
  authoredThickness: f32,
  thicknessMapScale: f32,
  segmentLength: f32,
) -> f32 {
  let hasAuthoredThickness = authoredThickness > 0.0;
  let referenceThickness = select(1.0, authoredThickness, hasAuthoredThickness);
  return select(
    segmentLength,
    min(
      segmentLength,
      referenceThickness * clamp(thicknessMapScale, 0.0, 1.0),
    ),
    hasAuthoredThickness,
  );
}

struct NrcTeacherContainingMedia {
  valid: u32,
  depth: u32,
  materialId: array<u32, 4>,
  tri: array<u32, 4>,
  instance: array<u32, 4>,
  ior: array<vec3f, 4>,
  mappedBeerReference: array<vec3f, 4>,
  authoredThickness: array<f32, 4>,
  thicknessMapScale: array<f32, 4>,
  albedo: array<vec3f, 4>,
  scatter: array<vec4f, 4>,
  transmissionPaid: array<u32, 4>,
}

// Reconstruct all authored closed bulk media containing the suffix origin with
// the shared exact outward event scan. The actual exit face supplies every
// optical payload and component/range identity owns LIFO pairing.
fn nrcTeacherClassifyContainingMedia(
  initialHit: IntersectionResult,
  initialPos: vec3f,
  initialWo: vec3f,
) -> NrcTeacherContainingMedia {
  var out: NrcTeacherContainingMedia;
  out.valid = 0u;
  out.depth = 0u;
  let classified = materialShadowClassifyContainingMedia(
    ubo.bvhMode,
    ubo.tlasNodeCount,
    initialPos,
    safe_normalize(initialWo),
    ubo.triIntersectEpsilon,
    bvh_material,
    BVH_MATERIAL_TEX_WIDTH,
    bvh_beer,
  );
  if (
    classified.valid == 0u ||
    classified.state.depth > NRC_TEACHER_MEDIUM_CAPACITY
  ) { return out; }
  for (var seed = 0u; seed < classified.state.depth; seed = seed + 1u) {
    let triIndex = classified.state.tri[seed];
    let materialCoord = vec2u(
      triIndex % BVH_MATERIAL_TEX_WIDTH,
      triIndex / BVH_MATERIAL_TEX_WIDTH,
    );
    let materialWord = textureLoad(
      bvh_material, vec2i(materialCoord), 0,
    ).r;
    out.materialId[seed] = classified.state.materialId[seed];
    out.tri[seed] = triIndex;
    out.instance[seed] = classified.state.instance[seed];
    out.ior[seed] = materialDispersionIorRgb(
      triIndex, decodeIor(materialWord),
    );
    out.mappedBeerReference[seed] = classified.state.tint[seed];
    out.authoredThickness[seed] = classified.state.thickness[seed];
    out.thicknessMapScale[seed] =
      classified.state.thicknessMapScale[seed];
    out.albedo[seed] = classified.state.albedo[seed];
    out.scatter[seed] = classified.state.scattering[seed];
    // No prefix event observed this entry. The first paired forward exit pays
    // the authored scalar exactly once before the seed is popped.
    out.transmissionPaid[seed] = 0u;
  }
  out.depth = classified.state.depth;
  out.valid = 1u;
  return out;
}

// Receiver-local sources use the same ownership and transfer as the live
// ReSTIR-GI suffix. Baked light is irradiance (not scene emission), so it is
// always present, converted to outgoing diffuse radiance, and reduced by the
// opaque share. Emitter radiance remains conditional because NEE owns it after
// a non-delta continuation.
fn nrcTeacherLocalSourceForHit(
  hit: IntersectionResult,
  payload: RestirDIMaterialPayload,
  normal: vec3f,
  wo: vec3f,
  transmission: f32,
  includeEmitterEmission: bool,
  explicitSegmentVolume: bool,
) -> vec3f {
  var surfaceEmission = vec3f(0.0);
  if (includeEmitterEmission) {
    surfaceEmission = restir_gi_surface_emission_for_hit(hit);
  }
  let bakedDiffuse = payload.albedo * INV_PI * sampleLightMap(hit) *
    (1.0 - clamp(transmission, 0.0, 1.0));
  let rawSource =
    (surfaceEmission + bakedDiffuse) * payload.layerTransmission;
  let proxySource = applyHomogeneousVolumeSingleScatter(
    rawSource,
    payload.albedo,
    payload.volumeScattering,
    payload.bulkThickness,
    normal,
    wo,
  );
  return select(proxySource, rawSource, explicitSegmentVolume);
}

// Independent online teacher.  It intentionally has no DDGI/cache arguments
// and never calls sampleDDGIAtPoint or nrcQueryRadiance.
fn nrcTraceIndependentSuffixForChannel(
  initialHit: IntersectionResult,
  initialSourceFeature: OpticalSourceFeature,
  initialPos: vec3f,
  initialWo: vec3f,
  containingMedia: NrcTeacherContainingMedia,
  channel: u32,
  rng: ptr<function, u32>,
) -> vec3f {
  var currentHit = initialHit;
  var currentPos = initialPos;
  var wo = safe_normalize(initialWo);
  var throughput = restirGiSuffixDiagonalTransfer(vec3f(1.0));
  var radiance = vec3f(0.0);
  var arrivedWithoutNeeOwner = true;
  var segmentLength = 0.0;
  var mediumDepth = 0u;
  var mediumMaterialId: array<u32, 4>;
  var mediumTri: array<u32, 4>;
  var mediumInstance: array<u32, 4>;
  var mediumIor: array<vec3f, 4>;
  var mediumMappedBeerReference: array<vec3f, 4>;
  var mediumAuthoredThickness: array<f32, 4>;
  var mediumThicknessMapScale: array<f32, 4>;
  var mediumAlbedo: array<vec3f, 4>;
  var mediumScatter: array<vec4f, 4>;
  var mediumTransmissionPaid: array<u32, 4>;
  var currentSourceFeature = initialSourceFeature;

  for (
    var seed = 0u;
    seed < containingMedia.depth;
    seed = seed + 1u
  ) {
    mediumMaterialId[seed] = containingMedia.materialId[seed];
    mediumTri[seed] = containingMedia.tri[seed];
    mediumInstance[seed] = containingMedia.instance[seed];
    mediumIor[seed] = containingMedia.ior[seed];
    mediumMappedBeerReference[seed] =
      containingMedia.mappedBeerReference[seed];
    mediumAuthoredThickness[seed] =
      containingMedia.authoredThickness[seed];
    mediumThicknessMapScale[seed] =
      containingMedia.thicknessMapScale[seed];
    mediumAlbedo[seed] = containingMedia.albedo[seed];
    mediumScatter[seed] = containingMedia.scatter[seed];
    mediumTransmissionPaid[seed] =
      containingMedia.transmissionPaid[seed];
  }
  mediumDepth = containingMedia.depth;

  for (var depth = 0u; depth < NRC_TEACHER_MAX_VERTICES; depth = depth + 1u) {
    let currentUseTlas = ubo.bvhMode == 1u && ubo.tlasNodeCount > 0u;
    let currentBoundaryId = sceneOpticalEncodedBoundaryId(
      currentUseTlas, currentHit.indices.w, currentHit.instanceIndex,
    );
    let currentRepresentedId = sceneOpticalRepresentedPrimitiveInstanceId(
      currentUseTlas, currentHit.indices.w, currentHit.instanceIndex,
    );
    if (mediumDepth > 0u && segmentLength > 0.0) {
      let top = mediumDepth - 1u;
      var segmentTri = mediumTri[top];
      var segmentBeerReference = mediumMappedBeerReference[top];
      var segmentAuthoredThickness = mediumAuthoredThickness[top];
      var segmentThicknessMapScale = mediumThicknessMapScale[top];
      var segmentAlbedo = mediumAlbedo[top];
      var segmentScatter = mediumScatter[top];
      if (
        packedMaterialHasTransmission(currentHit.matColorPacked) &&
        currentHit.side < 0.0 &&
        currentBoundaryId == mediumMaterialId[top] &&
        currentRepresentedId == mediumInstance[top]
      ) {
        segmentTri = currentHit.indices.w;
        segmentBeerReference = nrcTeacherMappedBeerReference(currentHit);
        segmentAuthoredThickness = materialShadowAuthoredThickness(currentHit);
        segmentThicknessMapScale = nrcTeacherThicknessMapScale(currentHit);
        let exitScalar = decodeMaterialColor(currentHit.matColorPacked);
        let exitVertexColor = sampleVertexColorForHit(currentHit);
        segmentAlbedo = sampleBaseColorMap(
          currentHit, exitScalar.rgb * exitVertexColor.rgb,
        );
        segmentScatter = sampleVolumeScatteringControls(currentHit.indices.w);
      }
      let absorptionTransfer = nrcTeacherBeerForSegment(
        segmentTri,
        segmentBeerReference,
        segmentAuthoredThickness,
        segmentThicknessMapScale,
        segmentLength,
      );
      let transportDistance = nrcTeacherTransportDistance(
        segmentAuthoredThickness,
        segmentThicknessMapScale,
        segmentLength,
      );
      throughput = throughput * restirGiSuffixSegmentTransfer(
        absorptionTransfer,
        segmentScatter,
        segmentAlbedo,
        transportDistance,
      );
      if (
        !restirGiSuffixTransferFinite(throughput) ||
        nrcTeacherTransferMax(throughput) <= 0.0
      ) {
        break;
      }
    }

    let geoNormal = currentHit.normal;
    let smoothNormal = restir_gi_smooth_normal_for_hit(currentHit, geoNormal);
    let normal = applyBumpMapForHit(
      currentHit,
      applyNormalMapForHit(currentHit, smoothNormal),
    );
    let wordCoord = vec2u(
      currentHit.indices.w % BVH_MATERIAL_TEX_WIDTH,
      currentHit.indices.w / BVH_MATERIAL_TEX_WIDTH,
    );
    let materialWord = textureLoad(bvh_material, vec2i(wordCoord), 0).r;
    let scalar = decodeMaterialColor(currentHit.matColorPacked);
    let transmission = clamp(
      sampleTransmissionMapForHit(currentHit, scalar.a), 0.0, 1.0,
    );
    let payload = sampleRestirDIMaterialPayloadForHit(
      currentHit, smoothNormal, normal, scalar.rgb, materialWord, wo,
    );
    // Containment is a stable authored material/geometry property. Neither a
    // mapped transmission texel nor mapped thickness G may change thin-vs-bulk
    // topology or material+instance stack ownership.
    let authoredTransmissionTopology = materialHasTransmission(scalar.a);
    let authoredThickness = materialShadowAuthoredThickness(currentHit);
    if (authoredTransmissionTopology && currentRepresentedId == 0u) { break; }
    let bulkMedium =
      authoredTransmissionTopology && currentBoundaryId != 0u;
    let thinSheet = authoredTransmissionTopology && !bulkMedium;
    let entering = currentHit.side >= 0.0;

    let materialIor = materialDispersionIorRgb(
      currentHit.indices.w, decodeIor(materialWord),
    );
    var etaIncident = vec3f(1.0);
    if (mediumDepth > 0u) {
      etaIncident = mediumIor[mediumDepth - 1u];
    }
    var etaTarget = materialIor;
    var pairedBulkExit = false;
    var pairedPaidExit = false;
    if (bulkMedium && !entering && mediumDepth > 0u) {
      let top = mediumDepth - 1u;
      pairedBulkExit =
        mediumMaterialId[top] == currentBoundaryId &&
        mediumInstance[top] == currentRepresentedId;
      if (pairedBulkExit) {
        pairedPaidExit = mediumTransmissionPaid[top] != 0u;
        etaTarget = vec3f(1.0);
        if (mediumDepth > 1u) {
          etaTarget = mediumIor[mediumDepth - 2u];
        }
      }
    }
    if (
      bulkMedium && !entering && !pairedBulkExit
    ) { break; }

    var transmissionPhysicalWeight = transmission;
    if (pairedPaidExit) {
      transmissionPhysicalWeight = 1.0;
    }
    // The same effective t owns every opaque-vs-transmissive partition. Once a
    // bulk entry paid scalar transmission, its paired exit is a pure dielectric
    // boundary: exact reflection remains, but no complementary opaque share or
    // baked diffuse source is reopened at that second boundary.
    let closureTransmission = transmissionPhysicalWeight;

    // Seed each NEE proposal from the incident-side medium stack before the
    // current boundary event. Compact out castShadow:false media: their entry
    // boundaries are absent to an empty-state shadow walk, so a start-inside
    // proposal must not resurrect their Beer/scatter extinction either.
    var shadowMediumState = materialShadowEmptyMediumState();
    var shadowDepth = 0u;
    for (var shadowSource = 0u; shadowSource < mediumDepth; shadowSource = shadowSource + 1u) {
      let shadowTri = mediumTri[shadowSource];
      let shadowWordCoord = vec2u(
        shadowTri % BVH_MATERIAL_TEX_WIDTH,
        shadowTri / BVH_MATERIAL_TEX_WIDTH,
      );
      let shadowMaterialWord = textureLoad(
        bvh_material, vec2i(shadowWordCoord), 0,
      ).r;
      if ((shadowMaterialWord & 1u) != 0u) {
        continue;
      }
      shadowMediumState.materialId[shadowDepth] = mediumMaterialId[shadowSource];
      shadowMediumState.tri[shadowDepth] = shadowTri;
      shadowMediumState.instance[shadowDepth] = mediumInstance[shadowSource];
      shadowMediumState.tint[shadowDepth] = mediumMappedBeerReference[shadowSource];
      shadowMediumState.thickness[shadowDepth] = mediumAuthoredThickness[shadowSource];
      shadowMediumState.thicknessMapScale[shadowDepth] =
        mediumThicknessMapScale[shadowSource];
      shadowMediumState.scattering[shadowDepth] = mediumScatter[shadowSource];
      shadowMediumState.albedo[shadowDepth] = mediumAlbedo[shadowSource];
      shadowMediumState.distance[shadowDepth] = 0.0;
      shadowMediumState.transmissionPaid[shadowDepth] =
        mediumTransmissionPaid[shadowSource];
      shadowDepth = shadowDepth + 1u;
    }
    shadowMediumState.depth = shadowDepth;
    let exactDielectricInterface =
      authoredTransmissionTopology &&
      (!bulkMedium || entering || pairedBulkExit);

    if (decodeIsUnlitMaterial(materialWord)) {
      radiance = radiance + throughput *
        payload.albedo * payload.layerTransmission;
      break;
    }

    // NEE owns same-side reflection connections only when this exact triangle
    // is represented in the live mesh-emitter proposal. The alpha lane is
    // rebuilt with the emitter list, so implicit skipEmitter surfaces retain
    // BSDF-hit Le while explicit mesh-area ownership remains authoritative.
    let emissionCoord = vec2u(
      currentHit.indices.w % BVH_MATERIAL_TEX_WIDTH,
      currentHit.indices.w / BVH_MATERIAL_TEX_WIDTH,
    );
    let areaNeeOwnsHit =
      min(ubo.emitterCount, sceneEmitterCount()) > 0u &&
      textureLoad(
        restir_gi_bvh_emissive, vec2i(emissionCoord), 0,
      ).a >= 0.5;
    let includeEmitterEmission =
      depth == 0u || arrivedWithoutNeeOwner || !areaNeeOwnsHit;
    radiance = radiance + throughput * nrcTeacherLocalSourceForHit(
      currentHit,
      payload,
      normal,
      wo,
      closureTransmission,
      includeEmitterEmission,
      bulkMedium,
    );
    // NEE samples the complete same-side reflection family at every material
    // value. Transmission is a separate below-surface family, so an emitter
    // reached through it keeps emission ownership at the next vertex.
    radiance = radiance + throughput * (
      nrcTeacherAreaNee(
        currentHit, currentPos, geoNormal, normal, wo, payload,
        closureTransmission,
        etaIncident, etaTarget, exactDielectricInterface, bulkMedium,
        shadowMediumState, rng,
      ) +
      nrcTeacherAnalyticNee(
        currentHit, currentPos, geoNormal, normal, wo, payload,
        closureTransmission,
        etaIncident, etaTarget, exactDielectricInterface, bulkMedium,
        shadowMediumState, rng,
      ) +
      nrcTeacherSunNee(
        currentHit, currentPos, geoNormal, normal, wo, payload,
        closureTransmission,
        etaIncident, etaTarget, exactDielectricInterface, bulkMedium,
        shadowMediumState, rng,
      )
    );

    if (depth + 1u >= NRC_TEACHER_MAX_VERTICES) { break; }

    var nextDir = vec3f(0.0, 1.0, 0.0);
    var nextThroughput = throughput;
    var nextArrivesWithoutNeeOwner = false;

    // A unit-envelope reflection family (persistent Fresnel reflection plus
    // the complementary opaque share) competes with the physical transmission
    // event. Authored scalar transmission is paid once on bulk entry; its paired
    // exit retains unit support and cannot pay the same material traversal again.
    // A reciprocal thin sheet has no stack entry and pays its scalar once around
    // the two-interface estimator below.
    let idealTransmissionBranchPdf =
      transmissionPhysicalWeight / (1.0 + transmissionPhysicalWeight);
    let transmissionBranchPdf = represented_bernoulli_probability_f32(
      idealTransmissionBranchPdf,
    );
    let reflectionBranchPdf = 1.0 - transmissionBranchPdf;
    var chooseTransmission = false;
    if (transmissionBranchPdf > 0.0) {
      chooseTransmission = rand_f32(rng) < transmissionBranchPdf;
    }

    if (chooseTransmission) {
      // Bounded rough-dielectric interface walk. IntersectionResult.normal is
      // face-forward, so authored side owns medium entry/exit. Geometry, the
      // Walter Jacobian/PDF, and the nested IOR stack all use this represented
      // RGB channel; the wrapper below repeats the correlated trace for R/G/B.
      // A TIR draw has zero transmission integrand here: its exact reflected
      // mass is evaluated by nrcTeacherExactInterfaceReflection in the
      // independently selected same-side family.

      let interfaceLayer = sampleFaceLayerControls(
        currentHit.indices.w, currentHit.side >= 0.0,
      );
      let interfaceLobe = nrcTeacherSampleDielectricLobe(
        currentHit,
        normal,
        payload.anisotropyTangent,
        payload.anisotropyBitangent,
        wo,
        payload.rough,
        payload.anisotropy.x,
        payload.anisotropy.y,
        etaIncident,
        etaTarget,
        currentHit.side >= 0.0,
        interfaceLayer,
        channel,
        false,
        false,
        rng,
      );
      if (
        interfaceLobe.valid == 0u ||
        interfaceLobe.kind != NRC_TEACHER_DIELECTRIC_EVENT_TRANSMISSION
      ) { break; }
      nextThroughput = nextThroughput * restirGiSuffixDiagonalTransfer(
        interfaceLobe.weightRgb *
          transmissionPhysicalWeight / transmissionBranchPdf,
      );

      if (thinSheet) {
        // A geometric sheet represents a reciprocal two-boundary slab. Internal
        // rough-interface reflection alternates authored faces until a sampled
        // transmission exits. At exact TIR both discrete requests collapse to
        // reflection, so the effective discrete probability is one.
        let mappedBaseRoughness = sampleMaterialScalarMap(
          currentHit,
          MATERIAL_MAP_SLOT_ROUGHNESS,
          1u,
          decodeRoughMetal(materialWord).x,
        );
        var slabFrontFacing = currentHit.side < 0.0;
        var slabWo = -interfaceLobe.direction;
        var slabExited = false;
        for (
          var slabInterface = 1u;
          slabInterface < NRC_TEACHER_SHEET_MAX_INTERFACES;
          slabInterface = slabInterface + 1u
        ) {
          let slabLayer = sampleFaceLayerControls(
            currentHit.indices.w, slabFrontFacing,
          );
          let slabRoughness = faceLayerRoughness(
            mappedBaseRoughness, slabLayer,
          );
          let slabMappedNormal = applyBumpMapForHit(
            currentHit,
            applyNormalMapForSideForHit(
              currentHit, smoothNormal, slabFrontFacing,
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
            dot(slabWo, slabAlignedNormal) > 0.0,
          );
          let slabFrame = materialTangentFrameForHit(
            currentHit,
            slabNormal,
            MATERIAL_MAP_ANISOTROPY_TEXEL_OFFSET,
          );
          let slabTransmissionPdf = represented_bernoulli_probability_f32(0.5);
          let slabChooseTransmission =
            rand_f32(rng) < slabTransmissionPdf;
          let slabLobe = nrcTeacherSampleDielectricLobe(
            currentHit,
            slabNormal,
            slabFrame.tangent,
            slabFrame.bitangent,
            slabWo,
            slabRoughness,
            payload.anisotropy.x,
            payload.anisotropy.y,
            etaTarget,
            etaIncident,
            slabFrontFacing,
            slabLayer,
            channel,
            !slabChooseTransmission,
            true,
            rng,
          );
          if (slabLobe.valid == 0u) { break; }
          let slabSelectedPdf = select(
            1.0 - slabTransmissionPdf,
            slabTransmissionPdf,
            slabChooseTransmission,
          );
          let slabDiscretePdf = select(
            slabSelectedPdf, 1.0, slabLobe.tir != 0u,
          );
          nextThroughput = nextThroughput * restirGiSuffixDiagonalTransfer(
            slabLobe.weightRgb / slabDiscretePdf,
          );
          if (
            !restirGiSuffixTransferFinite(nextThroughput) ||
            nrcTeacherTransferMax(nextThroughput) <= 0.0
          ) { break; }
          nextDir = slabLobe.direction;
          if (
            slabLobe.kind == NRC_TEACHER_DIELECTRIC_EVENT_TRANSMISSION
          ) {
            slabExited = true;
            break;
          }
          slabFrontFacing = !slabFrontFacing;
          slabWo = -nextDir;
        }
        if (!slabExited) { break; }
      } else {
        nextDir = interfaceLobe.direction;
        if (entering) {
          if (mediumDepth >= NRC_TEACHER_MEDIUM_CAPACITY) { break; }
          mediumMaterialId[mediumDepth] = currentBoundaryId;
          mediumTri[mediumDepth] = currentHit.indices.w;
          mediumInstance[mediumDepth] = currentRepresentedId;
          mediumIor[mediumDepth] = materialIor;
          mediumMappedBeerReference[mediumDepth] =
            nrcTeacherMappedBeerReference(currentHit);
          mediumAuthoredThickness[mediumDepth] = authoredThickness;
          mediumThicknessMapScale[mediumDepth] =
            nrcTeacherThicknessMapScale(currentHit);
          mediumAlbedo[mediumDepth] = payload.albedo;
          mediumScatter[mediumDepth] = payload.volumeScattering;
          mediumTransmissionPaid[mediumDepth] = 1u;
          mediumDepth = mediumDepth + 1u;
        } else {
          mediumDepth = mediumDepth - 1u;
        }
      }
      // Transmission NEE is not part of this bounded teacher, so an emitter
      // reached by this family has no competing NEE owner even when rough.
      nextArrivesWithoutNeeOwner = true;
    } else {
      let specularMixProbability = represented_bernoulli_probability_f32(
        NRC_TEACHER_SPECULAR_MIX,
      );
      if (rand_f32(rng) < specularMixProbability) {
        nextDir = ggxSampleVndf(normal, wo, payload.rough, rng);
      } else {
        nextDir = sampleCosineHemisphere(normal, rng);
      }
      let cosNext = max(0.0, dot(normal, nextDir));
      if (cosNext <= 0.0) { break; }
      let pdfCos = cosNext * INV_PI;
      let pdfSpec = ggxVndfReflectionPdf(normal, wo, nextDir, payload.rough);
      let proposalPdf =
        specularMixProbability * pdfSpec +
        (1.0 - specularMixProbability) * pdfCos;
      if (!(proposalPdf > 0.0)) { break; }
      nextThroughput = nextThroughput * restirGiSuffixDiagonalTransfer(
        nrcTeacherMaterialResponse(
          currentHit,
          payload,
          normal,
          wo,
          nextDir,
          closureTransmission,
          etaIncident,
          etaTarget,
          exactDielectricInterface,
        ) / proposalPdf / reflectionBranchPdf,
      );
    }

    if (
      !restirGiSuffixTransferFinite(nextThroughput) ||
      nrcTeacherTransferMax(nextThroughput) <= 0.0
    ) {
      break;
    }
    if (depth + 1u >= NRC_TEACHER_RR_START) {
      let survive = represented_bernoulli_probability_f32(
        clamp(nrcTeacherTransferMax(nextThroughput), 0.05, 0.95),
      );
      if (rand_f32(rng) >= survive) { break; }
      nextThroughput = nextThroughput * (1.0 / survive);
    }

    var nextRay: Ray;
    var nextHit: IntersectionResult;
    let nextAlphaSeed = pcgNext(rng);
    if (currentSourceFeature.kind != OPTICAL_SOURCE_FEATURE_INVALID) {
      nextRay = Ray(currentPos, nextDir);
      let sourceAware = traceSceneFirstHitAlphaMaskTexturedWithOpticalSource(
        ubo.bvhMode,
        ubo.tlasNodeCount,
        nextRay,
        currentSourceFeature,
        bvh_material,
        BVH_MATERIAL_TEX_WIDTH,
        nextAlphaSeed,
      );
      if (sourceAware.valid == 0u) { break; }
      nextHit = sourceAware.hit;
    } else {
      let offsetNormal = select(
        -geoNormal, geoNormal, dot(nextDir, geoNormal) >= 0.0,
      );
      nextRay = Ray(
        currentPos + offsetNormal * walkaroundRayOriginBias(),
        nextDir,
      );
      nextHit = traceSceneFirstHitAlphaMaskTextured(
        ubo.bvhMode,
        ubo.tlasNodeCount,
        nextRay,
        ubo.triIntersectEpsilon,
        bvh_material,
        BVH_MATERIAL_TEX_WIDTH,
        nextAlphaSeed,
      );
    }
    if (!nextHit.didHit) {
      // An escaped ray with an unpaired bulk entry would otherwise omit an
      // unbounded in-medium segment. Open/malformed volumes fail closed.
      if (mediumDepth != 0u) { break; }
      let receiverEnvironment = walkaroundScaleEnvironmentRadiance(
        envRadiance(nextDir),
        payload.envMapIntensity,
      );
      radiance = radiance + nextThroughput * receiverEnvironment;
      break;
    }

    var nextSourceFeature = opticalSourceFeatureInvalid();
    if (packedMaterialHasTransmission(nextHit.matColorPacked)) {
      let exactNext = traceSceneRetraceOpticalHit(
        ubo.bvhMode, ubo.tlasNodeCount, nextRay, nextHit, 0.0,
      );
      nextSourceFeature = sceneOpticalSourceFeatureForExactHit(
        ubo.bvhMode, ubo.tlasNodeCount, nextHit, exactNext,
      );
      if (
        !exactNext.hit ||
        nextSourceFeature.kind == OPTICAL_SOURCE_FEATURE_INVALID
      ) { break; }
      let nextUseTlas = ubo.bvhMode == 1u && ubo.tlasNodeCount > 0u;
      let nextTriangle = sceneLoadOpticalWorldTriangle(
        nextUseTlas, nextHit.indices.w, nextHit.instanceIndex,
      );
      if (nextTriangle.valid == 0u) { break; }
      nextHit.normal = exactNext.normal;
      nextHit.barycoord = exactNext.bary;
      nextHit.side = exactNext.side;
      nextHit.dist = exactNext.t;
      nextHit.uv = exactNext.bary.x * nextTriangle.uvA +
        exactNext.bary.y * nextTriangle.uvB +
        exactNext.bary.z * nextTriangle.uvC;
    }

    let nextPos = nextRay.origin + nextDir * nextHit.dist;
    segmentLength = length(nextPos - currentPos);
    currentPos = nextPos;
    currentHit = nextHit;
    wo = -nextDir;
    throughput = nextThroughput;
    arrivedWithoutNeeOwner = nextArrivesWithoutNeeOwner;
    currentSourceFeature = nextSourceFeature;
  }

  return select(vec3f(0.0), max(radiance, vec3f(0.0)), nrcTeacherFinite3(radiance));
}

// Trace one correlated path per represented RGB basis wavelength. Sharing the
// initial RNG state correlates lobe/VNDF decisions while allowing channel IOR
// to drive distinct Snell geometry, BTDF density/Jacobian, Beer transfer, and
// terminal radiance. Only the represented component from each path is retained.
fn nrcTraceIndependentSuffix(
  initialHitInput: IntersectionResult,
  initialPosInput: vec3f,
  initialWo: vec3f,
  rng: ptr<function, u32>,
) -> vec3f {
  var initialHit = initialHitInput;
  var initialPos = initialPosInput;
  var initialSourceFeature = opticalSourceFeatureInvalid();
  if (packedMaterialHasTransmission(initialHit.matColorPacked)) {
    let incomingDirection = -safe_normalize(initialWo);
    let replayRay = Ray(initialPos - incomingDirection, incomingDirection);
    let exactInitial = traceSceneRetraceOpticalHit(
      ubo.bvhMode, ubo.tlasNodeCount, replayRay, initialHit, 0.0,
    );
    initialSourceFeature = sceneOpticalSourceFeatureForExactHit(
      ubo.bvhMode, ubo.tlasNodeCount, initialHit, exactInitial,
    );
    if (
      !exactInitial.hit ||
      initialSourceFeature.kind == OPTICAL_SOURCE_FEATURE_INVALID
    ) { return vec3f(0.0); }
    let initialUseTlas = ubo.bvhMode == 1u && ubo.tlasNodeCount > 0u;
    let initialTriangle = sceneLoadOpticalWorldTriangle(
      initialUseTlas, initialHit.indices.w, initialHit.instanceIndex,
    );
    if (initialTriangle.valid == 0u) { return vec3f(0.0); }
    initialHit.normal = exactInitial.normal;
    initialHit.barycoord = exactInitial.bary;
    initialHit.side = exactInitial.side;
    initialHit.dist = exactInitial.t;
    initialHit.uv = exactInitial.bary.x * initialTriangle.uvA +
      exactInitial.bary.y * initialTriangle.uvB +
      exactInitial.bary.z * initialTriangle.uvC;
    initialPos = replayRay.origin + replayRay.direction * exactInitial.t;
  }
  // Containment topology and all stored optical properties are RGB-valued and
  // independent of the represented dispersion lane. Reconstruct them once,
  // then reuse the result for the three correlated channel paths.
  let containingMedia = nrcTeacherClassifyContainingMedia(
    initialHit, initialPos, initialWo,
  );
  if (containingMedia.valid == 0u) { return vec3f(0.0); }
  let sharedRng = (*rng);
  var rngR = sharedRng;
  var rngG = sharedRng;
  var rngB = sharedRng;
  let radianceR = nrcTraceIndependentSuffixForChannel(
    initialHit, initialSourceFeature,
    initialPos, initialWo, containingMedia, 0u, &rngR,
  );
  let radianceG = nrcTraceIndependentSuffixForChannel(
    initialHit, initialSourceFeature,
    initialPos, initialWo, containingMedia, 1u, &rngG,
  );
  let radianceB = nrcTraceIndependentSuffixForChannel(
    initialHit, initialSourceFeature,
    initialPos, initialWo, containingMedia, 2u, &rngB,
  );
  (*rng) = rngR ^ (rngG * 0x9e3779b9u) ^ (rngB * 0x85ebca6bu);
  return vec3f(radianceR.r, radianceG.g, radianceB.b);
}
`;
