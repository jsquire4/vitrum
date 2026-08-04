import { MNEE_CHAIN_MAX_ITERS, MNEE_CHAIN_MAX_VERTICES } from './mneeNewton.wgsl.js';
import {
  MNEE_FACET_TABLE_MAGIC,
  MNEE_GUIDED_FACET_PROBABILITY,
} from '../../scene/mneeFacetCandidates.js';

/**
 * Production caustic transport for the full pt-webgpu tier.
 *
 * Strategy 1 is one bounded planar-manifold NEE estimator. It samples chain
 * lengths 1..8, reflection/transmission events, exact TLAS facet identities,
 * and every explicit emitter family under one non-overlapping ownership path.
 * Strategy 2 is the progressive SPPM gather populated by the photon pre-pass.
 */
export const PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL = /* wgsl */ `


// Exact bounded draws and full-support facet proposals for unified MNEE.

// Unbiased bounded u32 draw. Unlike floor(rand_f32 * f32(count)), this keeps
// every identity reachable above 2^24 and removes modulo bias at all bounds.
fn causticUniformEmitterIndex(rng: ptr<function, PtRngState>, count: u32) -> u32 {
  return ptRandBoundedU32(rng, count);
}

// A proposal names one entry from the host-packed exact TLAS membership table.
// Each identity is a real (global triangle, global instance) BLAS pair; no
// Cartesian-product holes or hidden success-conditioned normalizer remain.
struct MneeFacetProposal {
  valid: u32,
  triIndex: u32,
  instanceIndex: u32,
  _pad: u32,
  p: vec3f,
  n: vec3f,
  pdf: f32,
};

fn mneeFacetCandidateBase() -> u32 {
  return params.analyticCount * 2u;
}

fn mneeFacetCandidateCount() -> u32 {
  let base = mneeFacetCandidateBase();
  if (base >= arrayLength(&analyticParams)) { return 0u; }
  let tableHeader = analyticParams[base];
  if (bitcast<u32>(tableHeader.x) != ${MNEE_FACET_TABLE_MAGIC}u) { return 0u; }
  return min(
    bitcast<u32>(tableHeader.y),
    arrayLength(&analyticParams) - base - 1u,
  );
}


fn mneeFacetFromIdentity(triIndex: u32, instanceIndex: u32) -> MneeFacetProposal {
  var proposal: MneeFacetProposal;
  proposal.valid = 0u;
  proposal.triIndex = triIndex;
  proposal.instanceIndex = instanceIndex;
  proposal._pad = 0u;
  proposal.p = vec3f(0.0);
  proposal.n = vec3f(0.0, 1.0, 0.0);
  proposal.pdf = 0.0;
  if (triIndex >= min(params.triangleCount, arrayLength(&indices)) ||
      instanceIndex >= arrayLength(&tlasBlasRoots)) {
    return proposal;
  }
  let tri = indices[triIndex];
  let positionCount = arrayLength(&positions);
  if (tri.x >= positionCount || tri.y >= positionCount || tri.z >= positionCount) {
    return proposal;
  }
  let a = materialTexturePointToWorld(positions[tri.x].xyz, instanceIndex);
  let b = materialTexturePointToWorld(positions[tri.y].xyz, instanceIndex);
  let c = materialTexturePointToWorld(positions[tri.z].xyz, instanceIndex);
  let edge1 = b - a;
  let edge2 = c - a;
  let areaMeasure = measureAreaVector(edge1, edge2, 0.5);
  if (areaMeasure.valid == 0u) { return proposal; }

  proposal.valid = 1u;
  proposal.triIndex = triIndex;
  proposal.instanceIndex = instanceIndex;
  let positionScale = max(
    max(abs(a.x), max(abs(a.y), abs(a.z))),
    max(
      max(abs(b.x), max(abs(b.y), abs(b.z))),
      max(abs(c.x), max(abs(c.y), abs(c.z))),
    ),
  );
  proposal.p = vec3f(0.0);
  if (positionScale > 0.0 && positionScale <= 3.402823e38) {
    proposal.p =
      ((a / positionScale) + (b / positionScale) + (c / positionScale)) *
      (positionScale / 3.0);
  }
  proposal.n = areaMeasure.normal;
  return proposal;
}

fn mneeFacetCandidateAt(index: u32) -> MneeFacetProposal {
  var invalid = mneeFacetFromIdentity(0xffffffffu, INVALID_TLAS_INSTANCE_INDEX);
  let count = mneeFacetCandidateCount();
  if (index >= count) { return invalid; }
  let packed = analyticParams[mneeFacetCandidateBase() + 1u + index];
  let triIndex = bitcast<u32>(packed.x);
  let instanceIndex = bitcast<u32>(packed.y);
  var proposal = mneeFacetFromIdentity(triIndex, instanceIndex);
  proposal.pdf = 1.0 / f32(count);
  return proposal;
}


struct MneeFacetOptics {
  matId: u32,
  boundaryKind: u32,
  boundaryIndex: u32,
  boundaryComponent: u32,
  baseColor: vec3f,
  specularColor: vec3f,
  sigmaA: vec3f,
  sigmaT: vec3f,
  roughness: f32,
  oppositeRoughness: f32,
  oppositeLayerWeight: vec3f,
  metallic: f32,
  transmission: f32,
  ior: f32,
  specularIntensity: f32,
  iridescence: f32,
  iridescenceIor: f32,
  iridescenceThicknessMin: f32,
  iridescenceThicknessMax: f32,
  coverage: f32,
  volumeThickness: f32,
  hasVolumeThickness: bool,
  isBulkMedium: bool,
  isThinSheet: bool,
  isUnlit: bool,
  thinFilmEnabled: bool,
  thinFilmLayerCount: u32,
  thinFilmIncidentIor: f32,
  thinFilmAngleDependent: bool,
  frontFace: bool,
}

fn mneeFacetBaryVW(facet: MneeFacetProposal, worldPoint: vec3f) -> vec2f {
  let tri = indices[facet.triIndex];
  let a = materialTexturePointToWorld(positions[tri.x].xyz, facet.instanceIndex);
  let b = materialTexturePointToWorld(positions[tri.y].xyz, facet.instanceIndex);
  let c = materialTexturePointToWorld(positions[tri.z].xyz, facet.instanceIndex);
  let e0 = b - a;
  let e1 = c - a;
  let ep = worldPoint - a;
  let d00 = dot(e0, e0);
  let d01 = dot(e0, e1);
  let d11 = dot(e1, e1);
  let d20 = dot(ep, e0);
  let d21 = dot(ep, e1);
  let denom = d00 * d11 - d01 * d01;
  if (abs(denom) <= 1e-16) { return vec2f(0.0); }
  let vRaw = (d11 * d20 - d01 * d21) / denom;
  let wRaw = (d00 * d21 - d01 * d20) / denom;
  let u = max(1.0 - vRaw - wRaw, 0.0);
  let v = max(vRaw, 0.0);
  let w = max(wRaw, 0.0);
  let sum = max(u + v + w, 1e-8);
  return vec2f(v / sum, w / sum);
}

fn mneeMaterialSigmaA(matId: u32, mat: DecodedMaterial, heroLambda: f32) -> vec3f {
  var sigmaA = select(vec3f(0.0), max(mat.sigmaA, vec3f(0.0)), mat.hasSigmaA);
  if (mat.hasSpectralAttenuation && mat.spectralSampleCount > 0u) {
    if (params.spectralEnabled != 0u) {
      sigmaA = vec3f(max(sampleMaterialSpectralMu(matId, heroLambdaTo01(heroLambda)), 0.0));
    } else {
      sigmaA = max(vec3f(
        sampleMaterialSpectralMu(matId, 0.15),
        sampleMaterialSpectralMu(matId, 0.50),
        sampleMaterialSpectralMu(matId, 0.85),
      ), vec3f(0.0));
    }
  } else if (params.spectralEnabled != 0u) {
    sigmaA = vec3f(spectralRgbFactorAtHero(sigmaA, heroLambda));
  }
  return sigmaA;
}

fn mneeMaterialSigmaT(matId: u32, mat: DecodedMaterial, heroLambda: f32) -> vec3f {
  let sigmaA = mneeMaterialSigmaA(matId, mat, heroLambda);
  let sigmaS = select(
    max(mat.scatteringRgb, vec3f(0.0)),
    vec3f(spectralRgbFactorAtHero(max(mat.scatteringRgb, vec3f(0.0)), heroLambda)),
    params.spectralEnabled != 0u,
  );
  return max(sigmaA + sigmaS, vec3f(0.0));
}

// Deterministic expected surface coverage for the solved manifold vertex.
// This is the probability that alphaTestPassThrough keeps the same authored
// surface in the ordinary path kernel: opaque=1, mask=0/1, blend=alpha.
fn mneeFacetCoverage(
  matId: u32,
  triIndex: u32,
  baryVW: vec2f,
  instanceIndex: u32,
) -> f32 {
  let base = materialTextureDescriptorBase(matId);
  if (!materialTextureDescriptorSpanValid(base, 0u, 4u)) { return 0.0; }
  let alphaMode = materialTextureExactU32(
    materialTexDescriptors[base + 1u].x,
    3u,
  );
  if (alphaMode == 0xffffffffu) { return 0.0; }
  if (alphaMode == 0u) { return 1.0; }
  let alphaCutoff = materialTexDescriptors[base + 1u].y;
  let opacity = materialTexDescriptors[base + 1u].z;
  if (
    !materialTextureFiniteF32(alphaCutoff) ||
    !materialTextureFiniteF32(opacity) ||
    alphaCutoff < 0.0 || alphaCutoff > 1.0 ||
    opacity < 0.0 || opacity > 1.0
  ) { return 0.0; }
  let alphaRaw =
    sampleBaseColorTexture(matId, triIndex, baryVW, instanceIndex).a *
      sampleVertexColor(triIndex, baryVW).a *
      sampleAlphaTexture(matId, triIndex, baryVW, instanceIndex) *
      opacity;
  if (!materialTextureFiniteF32(alphaRaw)) { return 0.0; }
  let alpha = clamp(alphaRaw, 0.0, 1.0);
  if (alphaMode == 1u) {
    return select(0.0, 1.0, alpha >= alphaCutoff);
  }
  return alpha;
}

// Rehydrate the selected interface at the SOLVED point. MNEE cannot use only
// decodeMaterial(): maps, vertex color, face layers, spectral reflectance, and
// volume-thickness textures are part of the authored optical boundary too.
fn mneeFacetOpticsAt(
  facet: MneeFacetProposal,
  worldPoint: vec3f,
  incomingDir: vec3f,
  heroLambda: f32,
) -> MneeFacetOptics {
  let matId = triMaterialIds[facet.triIndex].x;
  let mat = decodeMaterial(matId);
  let baryVW = mneeFacetBaryVW(facet, worldPoint);
  let frontFace = dot(facet.n, incomingDir) < 0.0;
  var baseColor = mat.baseColor *
    sampleVertexColor(facet.triIndex, baryVW).rgb *
    sampleBaseColorTexture(matId, facet.triIndex, baryVW, facet.instanceIndex).rgb *
    sampleAoFactor(matId, facet.triIndex, baryVW, facet.instanceIndex);
  let orm = sampleOrmTexture(matId, facet.triIndex, baryVW, facet.instanceIndex);
  var roughness = clamp(mat.roughness * orm.g, 0.0, 1.0);
  let metallic = clamp(mat.metallic * orm.b, 0.0, 1.0);
  let transmission = clamp(
    mat.transmission * sampleTransmissionTexture(
      matId, facet.triIndex, baryVW, facet.instanceIndex,
    ),
    0.0, 1.0,
  );
  var iridescence = clamp(
    mat.iridescence * sampleIridescenceTexture(
      matId, facet.triIndex, baryVW, facet.instanceIndex,
    ),
    0.0, 1.0,
  );
  var iridescenceThicknessMin = mat.iridescenceThicknessMin;
  var iridescenceThicknessMax = mat.iridescenceThicknessMax;
  let iridescenceThicknessSample = sampleIridescenceThicknessTexture(
    matId, facet.triIndex, baryVW, facet.instanceIndex,
  );
  if (iridescenceThicknessSample >= 0.0) {
    let iridescenceThickness = mix(
      iridescenceThicknessMin,
      iridescenceThicknessMax,
      iridescenceThicknessSample,
    );
    iridescenceThicknessMin = iridescenceThickness;
    iridescenceThicknessMax = iridescenceThickness;
    if (iridescenceThickness <= 0.0) { iridescence = 0.0; }
  }
  var specularColor = max(
    mat.specularColor * sampleSpecularColorTexture(
      matId, facet.triIndex, baryVW, facet.instanceIndex,
    ),
    vec3f(0.0),
  );
  let specularIntensity = clamp(
    mat.specularIntensity * sampleSpecularIntensityTexture(
      matId, facet.triIndex, baryVW, facet.instanceIndex,
    ),
    0.0, 1.0,
  );
  if (params.spectralEnabled != 0u) {
    specularColor = vec3f(spectralRgbFactorAtHero(specularColor, heroLambda));
  }
  let layerTx = clamp(
    select(mat.backLayerTx, mat.frontLayerTx, frontFace),
    vec3f(0.0), vec3f(1.0),
  );
  let oppositeLayerTx = clamp(
    select(mat.frontLayerTx, mat.backLayerTx, frontFace),
    vec3f(0.0), vec3f(1.0),
  );
  let layerRoughness = select(
    mat.backLayerRoughness, mat.frontLayerRoughness, frontFace,
  );
  let oppositeLayerRoughness = select(
    mat.frontLayerRoughness, mat.backLayerRoughness, frontFace,
  );
  var oppositeRoughness = roughness;
  if (layerRoughness >= 0.0) {
    roughness = clamp(layerRoughness, 0.0, 1.0);
  }
  if (oppositeLayerRoughness >= 0.0) {
    oppositeRoughness = clamp(oppositeLayerRoughness, 0.0, 1.0);
  }
  baseColor = baseColor * select(
    layerTx,
    activeLayerWeightRgb(layerTx, heroLambda, true),
    params.spectralEnabled != 0u && luminance(layerTx) < 0.999,
  );
  if (params.spectralEnabled != 0u) {
    baseColor = vec3f(spectralCombinedReflectanceAtHero(
      baseColor, mat.baseColor, mat.spectralReflCoeffs,
      mat.hasSpectralReflectance, heroLambda,
    ));
  }
  var ior = mat.ior;
  if (params.spectralEnabled != 0u && mat.dispersionAbbe > 0.0) {
    ior = cauchyIorAtLambda(heroLambda, mat.ior, mat.dispersionAbbe);
  }
  var volumeThickness = mat.volumeThickness;
  var hasVolumeThickness = mat.hasVolumeThickness;
  let thicknessSample = sampleVolumeThicknessTexture(
    matId, facet.triIndex, baryVW, facet.instanceIndex,
  );
  if (thicknessSample >= 0.0 && hasVolumeThickness) {
    volumeThickness = max(volumeThickness * thicknessSample, 0.0);
  }
  var out: MneeFacetOptics;
  out.matId = matId;
  let boundary = mediumBoundaryIdentity(
    facet.triIndex, facet.instanceIndex,
  );
  out.boundaryKind = boundary.x;
  out.boundaryIndex = boundary.y;
  out.boundaryComponent = boundary.z;
  out.baseColor = baseColor;
  out.specularColor = specularColor;
  out.sigmaA = mneeMaterialSigmaA(matId, mat, heroLambda);
  out.sigmaT = mneeMaterialSigmaT(matId, mat, heroLambda);
  out.roughness = roughness;
  out.oppositeRoughness = oppositeRoughness;
  out.oppositeLayerWeight = select(
    oppositeLayerTx,
    activeLayerWeightRgb(oppositeLayerTx, heroLambda, true),
    params.spectralEnabled != 0u && luminance(oppositeLayerTx) < 0.999,
  );
  out.metallic = metallic;
  out.transmission = transmission;
  out.ior = ior;
  out.specularIntensity = specularIntensity;
  out.iridescence = iridescence;
  out.iridescenceIor = mat.iridescenceIor;
  out.iridescenceThicknessMin = iridescenceThicknessMin;
  out.iridescenceThicknessMax = iridescenceThicknessMax;
  out.coverage = mneeFacetCoverage(
    matId, facet.triIndex, baryVW, facet.instanceIndex,
  );
  out.volumeThickness = volumeThickness;
  out.hasVolumeThickness = hasVolumeThickness;
  out.isBulkMedium = mat.isBulkMedium;
  out.isThinSheet = mat.isThinSheet;
  out.isUnlit = mat.isUnlit;
  out.thinFilmEnabled = mat.thinFilmEnabled;
  out.thinFilmLayerCount = mat.thinFilmLayerCountU;
  out.thinFilmIncidentIor = mat.thinFilmIncidentIor;
  out.thinFilmAngleDependent = mat.thinFilmAngleDependent;
  out.frontFace = frontFace;
  return out;
}

// Match the production sampler's genuine Dirac domains. Thin film changes the
// interface Fresnel but not its measure: only the reflection/refraction branches
// of a smooth transmissive dielectric are delta. Opaque metallic GGX remains
// finite measure, including at the shared alpha floor.
fn mneeFacetHasDeltaReflection(optics: MneeFacetOptics) -> bool {
  if (optics.isUnlit || optics.coverage <= 0.0) { return false; }
  return optics.metallic == 0.0 &&
    optics.transmission > 0.0 &&
    bsdfDielectricIsSmooth(optics.roughness);
}

fn mneeFacetHasDeltaTransmission(optics: MneeFacetOptics) -> bool {
  if (optics.isUnlit || optics.coverage <= 0.0) { return false; }
  return optics.metallic == 0.0 &&
    optics.transmission > 0.0 &&
    bsdfDielectricIsSmooth(optics.roughness) &&
    (!optics.isThinSheet ||
      bsdfDielectricIsSmooth(optics.oppositeRoughness));
}

const MNEE_CHAIN_EVENT_REFLECTION = 0u;
const MNEE_CHAIN_EVENT_TRANSMISSION = 1u;
const MNEE_SOURCE_FINITE = 0u;
const MNEE_SOURCE_DIRECTIONAL = 1u;
const MNEE_GUIDED_MIX_PROBABILITY = ${MNEE_GUIDED_FACET_PROBABILITY};

fn mneeFacetIdentityEqual(a: MneeFacetProposal, b: MneeFacetProposal) -> bool {
  return a.valid != 0u && b.valid != 0u &&
    a.triIndex == b.triIndex && a.instanceIndex == b.instanceIndex;
}

// The manifold solver constrains geometric facet planes. A normal, layer-
// normal, clearcoat-normal, or bump map changes the ordinary estimator's
// interface frame without changing that plane, so MNEE must not claim the
// event until its constraint/Jacobian is mapped-normal aware.
fn mneeFacetHasMappedInterface(facet: MneeFacetProposal) -> bool {
  if (
    facet.valid == 0u ||
    facet.triIndex >= arrayLength(&triMaterialIds)
  ) {
    return true;
  }
  let matId = triMaterialIds[facet.triIndex].x;
  let base = materialTextureDescriptorBase(matId);
  if (!materialTextureDescriptorSpanValid(base, 0u, MATERIAL_TEX_VEC4_STRIDE)) {
    return true;
  }
  let layerCount = textureNumLayers(materialTexturesLinear);
  let normalIdx = materialTextureLayerIndex(
    materialTexDescriptors[base].y,
    layerCount,
  );
  let bumpIdx = materialTextureLayerIndex(
    materialTexDescriptors[base + 3u].w,
    layerCount,
  );
  let clearcoatNormalIdx = materialTextureLayerIndex(
    materialTexDescriptors[base + MATERIAL_TEX_CLEARCOAT_NORMAL].x,
    layerCount,
  );
  let layerNormals = materialTexDescriptors[base + MATERIAL_TEX_LAYER_NORMAL];
  let frontLayerNormalIdx = materialTextureLayerIndex(layerNormals.x, layerCount);
  let backLayerNormalIdx = materialTextureLayerIndex(layerNormals.z, layerCount);
  return normalIdx >= 0 || bumpIdx >= 0 || clearcoatNormalIdx >= 0 ||
    frontLayerNormalIdx >= 0 || backLayerNormalIdx >= 0;
}

// Intermediate MNEE throughput/Jacobians currently model the base dielectric
// interface only. Active clearcoat and sheen are outer layers that attenuate
// the production lower-layer response directionally, so fail closed until the
// manifold estimator carries that exact layered response.
fn mneeFacetHasUnsupportedOuterLayer(facet: MneeFacetProposal) -> bool {
  if (
    facet.valid == 0u ||
    facet.triIndex >= arrayLength(&triMaterialIds)
  ) {
    return true;
  }
  let mat = decodeMaterial(triMaterialIds[facet.triIndex].x);
  return mat.clearcoat > 0.0 || mat.sheen > 0.0;
}

fn mneeFacetSupportsEvent(
  facet: MneeFacetProposal,
  point: vec3f,
  incomingTravel: vec3f,
  eventMode: u32,
  heroLambda: f32,
) -> bool {
  if (
    mneeFacetHasMappedInterface(facet) ||
    mneeFacetHasUnsupportedOuterLayer(facet)
  ) { return false; }
  let optics = mneeFacetOpticsAt(facet, point, incomingTravel, heroLambda);
  return select(
    mneeFacetHasDeltaReflection(optics),
    mneeFacetHasDeltaTransmission(optics),
    eventMode == MNEE_CHAIN_EVENT_TRANSMISSION,
  );
}

fn mneeGuidedFiniteFacet(
  sourcePoint: vec3f,
  recv: vec3f,
  ordinal: u32,
  eventMode: u32,
  heroLambda: f32,
) -> MneeFacetProposal {
  var invalid = mneeFacetFromIdentity(0xffffffffu, INVALID_TLAS_INSTANCE_INDEX);
  let sourceToRecv = recv - sourcePoint;
  let totalDistance = length(sourceToRecv);
  let endpointEpsilon = mneeScaleAwareEpsilon(sourcePoint, totalDistance);
  if (totalDistance <= 2.0 * endpointEpsilon) { return invalid; }
  let travel = sourceToRecv / totalDistance;
  var origin = sourcePoint + travel * endpointEpsilon;
  var remaining = totalDistance - endpointEpsilon;
  var compatibleIndex = 0u;
  for (var step = 0u; step < 16u; step = step + 1u) {
    let epsilon = mneeScaleAwareEpsilon(origin, remaining);
    let hit = traceClosest(Ray(origin, travel), epsilon, remaining + epsilon);
    if (!hit.didHit) { break; }
    let point = origin + travel * hit.dist;
    if (hit.triIndex < params.triangleCount &&
        hit.instanceIndex != INVALID_TLAS_INSTANCE_INDEX) {
      var facet = mneeFacetFromIdentity(hit.triIndex, hit.instanceIndex);
      facet.p = point;
      if (mneeFacetSupportsEvent(facet, point, travel, eventMode, heroLambda)) {
        if (compatibleIndex == ordinal) { return facet; }
        compatibleIndex = compatibleIndex + 1u;
      }
    }
    let advance = hit.dist + mneeScaleAwareEpsilon(point, remaining);
    origin = origin + travel * advance;
    remaining = remaining - advance;
    if (remaining <= mneeScaleAwareEpsilon(origin, remaining)) { break; }
  }
  return invalid;
}

fn mneeGuidedDirectionalFacet(
  recv: vec3f,
  towardLight: vec3f,
  reverseOrdinal: u32,
  eventMode: u32,
  heroLambda: f32,
) -> MneeFacetProposal {
  var origin = recv + towardLight * mneeScaleAwareEpsilon(recv, 0.0);
  var compatibleIndex = 0u;
  for (var step = 0u; step < 16u; step = step + 1u) {
    let epsilon = mneeScaleAwareEpsilon(origin, 0.0);
    let hit = traceClosest(Ray(origin, towardLight), epsilon, INFINITY);
    if (!hit.didHit) { break; }
    let point = origin + towardLight * hit.dist;
    if (hit.triIndex < params.triangleCount &&
        hit.instanceIndex != INVALID_TLAS_INSTANCE_INDEX) {
      var facet = mneeFacetFromIdentity(hit.triIndex, hit.instanceIndex);
      facet.p = point;
      if (mneeFacetSupportsEvent(
        facet, point, -towardLight, eventMode, heroLambda,
      )) {
        if (compatibleIndex == reverseOrdinal) { return facet; }
        compatibleIndex = compatibleIndex + 1u;
      }
    }
    origin = origin + towardLight *
      (hit.dist + mneeScaleAwareEpsilon(point, hit.dist));
  }
  return mneeFacetFromIdentity(0xffffffffu, INVALID_TLAS_INSTANCE_INDEX);
}

// Exact conditional proposal: half of the mass follows the deterministic
// straight-line specular guide and half remains uniform over every valid
// potential-delta facet identity. The latter preserves full support; the
// returned PMF includes both mixture components when they select the same facet.
fn mneeProposeConditionalFacet(
  rng: ptr<function, PtRngState>,
  sourceMode: u32,
  sourcePoint: vec3f,
  towardLight: vec3f,
  recv: vec3f,
  guideOrdinal: u32,
  eventMode: u32,
  heroLambda: f32,
) -> MneeFacetProposal {
  let count = mneeFacetCandidateCount();
  if (count == 0u) {
    return mneeFacetFromIdentity(0xffffffffu, INVALID_TLAS_INSTANCE_INDEX);
  }
  var guided = mneeFacetFromIdentity(
    0xffffffffu, INVALID_TLAS_INSTANCE_INDEX,
  );
  if (sourceMode == MNEE_SOURCE_DIRECTIONAL) {
    guided = mneeGuidedDirectionalFacet(
      recv, towardLight, guideOrdinal, eventMode, heroLambda,
    );
  } else {
    guided = mneeGuidedFiniteFacet(
      sourcePoint, recv, guideOrdinal, eventMode, heroLambda,
    );
  }
  let guidedAvailable = guided.valid != 0u;
  var selected = guided;
  if (!guidedAvailable || rand_f32(rng) >= MNEE_GUIDED_MIX_PROBABILITY) {
    selected = mneeFacetCandidateAt(causticUniformEmitterIndex(rng, count));
  }
  if (mneeFacetHasMappedInterface(selected)) {
    return mneeFacetFromIdentity(0xffffffffu, INVALID_TLAS_INSTANCE_INDEX);
  }
  if (!guidedAvailable) {
    selected.pdf = 1.0 / f32(count);
    return selected;
  }
  selected.pdf = (1.0 - MNEE_GUIDED_MIX_PROBABILITY) / f32(count);
  if (mneeFacetIdentityEqual(selected, guided)) {
    selected.pdf = selected.pdf + MNEE_GUIDED_MIX_PROBABILITY;
  }
  return selected;
}

fn mneeFacetReflectionFactorWithEta(
  optics: MneeFacetOptics,
  microfacetCos: f32,
  heroLambda: f32,
  etaTOverI: f32,
) -> vec3f {
  let film = ThinFilmInterface(
    optics.thinFilmEnabled, optics.matId, optics.thinFilmLayerCount,
    optics.thinFilmIncidentIor, optics.ior,
    optics.thinFilmAngleDependent, optics.frontFace,
    params.spectralEnabled != 0u, heroLambda, optics.transmission,
  );
  return optics.coverage * materialDielectricLayeredInterface(
    microfacetCos, etaTOverI,
    optics.iridescence, optics.iridescenceIor,
    optics.iridescenceThicknessMin, optics.iridescenceThicknessMax,
    optics.specularColor, optics.specularIntensity, film,
  ).reflectance;
}


fn mneeFacetTransmissionFactorWithEta(
  optics: MneeFacetOptics,
  microfacetCos: f32,
  heroLambda: f32,
  etaTOverI: f32,
  incidentIor: f32,
) -> vec3f {
  let film = ThinFilmInterface(
    optics.thinFilmEnabled, optics.matId, optics.thinFilmLayerCount,
    optics.thinFilmIncidentIor, optics.ior,
    optics.thinFilmAngleDependent, optics.frontFace,
    params.spectralEnabled != 0u, heroLambda, optics.transmission,
  );
  let interfaceResponse = materialDielectricLayeredInterface(
    microfacetCos, etaTOverI,
    optics.iridescence, optics.iridescenceIor,
    optics.iridescenceThicknessMin, optics.iridescenceThicknessMax,
    optics.specularColor, optics.specularIntensity, film,
  );
  if (!optics.isThinSheet) {
    return optics.coverage * optics.baseColor * optics.transmission *
      interfaceResponse.baseTransmittance;
  }
  // A represented thin sheet is an augmented two-interface delta event. The
  // manifold geometry sees no persistent eta transition, but its physical
  // entry still refracts into the authored dielectric and its reciprocal
  // virtual exit refracts back to the incident medium. Evaluate the coherent
  // coating only on entry, the bare exit once, and both authored face layers.
  let entryEta = optics.ior / max(incidentIor, 1e-4);
  let sin2Incident = max(1.0 - microfacetCos * microfacetCos, 0.0);
  let sin2Internal = sin2Incident /
    max(entryEta * entryEta, 1e-8);
  if (!(sin2Internal < 1.0)) { return vec3f(0.0); }
  let internalCos = sqrt(max(1.0 - sin2Internal, 0.0));
  let entryResponse = materialDielectricLayeredInterface(
    microfacetCos, entryEta,
    optics.iridescence, optics.iridescenceIor,
    optics.iridescenceThicknessMin, optics.iridescenceThicknessMax,
    optics.specularColor, optics.specularIntensity, film,
  );
  let exitResponse = materialDielectricLayeredInterface(
    internalCos, 1.0 / max(entryEta, 1e-4),
    optics.iridescence, optics.iridescenceIor,
    optics.iridescenceThicknessMin, optics.iridescenceThicknessMax,
    optics.specularColor, optics.specularIntensity, bsdfNoThinFilm(),
  );
  return optics.coverage * optics.baseColor * optics.transmission *
    entryResponse.baseTransmittance * exitResponse.baseTransmittance *
    max(optics.oppositeLayerWeight, vec3f(0.0));
}



fn mneeEmitterRadiance(rgb: vec3f, heroLambda: f32) -> vec3f {
  return select(
    rgb, spectralEmissionAtHero(rgb, heroLambda),
    params.spectralEnabled != 0u,
  );
}

// Scale-aware MNEE ray tolerance. triIntersectEpsilon retains its engine-wide
// absolute t-min contract. Distance and coordinate terms add four f32 ULPs so
// neither tiny configured scenes nor large translated scenes use a fixed metre
// offset that is outside their representable scale.
fn mneeScaleAwareEpsilon(point: vec3f, distanceHintRaw: f32) -> f32 {
  let finiteHint = select(
    0.0, abs(distanceHintRaw),
    distanceHintRaw == distanceHintRaw && abs(distanceHintRaw) < INFINITY,
  );
  let coordinateScale = max(abs(point.x), max(abs(point.y), abs(point.z)));
  let distanceTolerance = finiteHint * (4.0 * 1.1920928955078125e-7);
  let ulpTolerance = coordinateScale * (4.0 * 1.1920928955078125e-7);
  return max(
    max(max(params.triIntersectEpsilon, 0.0), distanceTolerance),
    max(ulpTolerance, bitcast<f32>(0x00800000u)),
  );
}

fn mneeSaturatedExpRgb(logValue: vec3f, enabled: vec3<bool>) -> vec3f {
  let maxFinite = bitcast<f32>(0x7f7fffffu);
  let maxLog = log(maxFinite) - 1e-3;
  return select(
    vec3f(0.0),
    exp(min(logValue, vec3f(maxLog))),
    enabled,
  );
}

fn mneeSaturatedAddRgb(lhs: vec3f, rhs: vec3f) -> vec3f {
  let maxFinite = vec3f(bitcast<f32>(0x7f7fffffu));
  return lhs + min(max(rhs, vec3f(0.0)), max(maxFinite - lhs, vec3f(0.0)));
}


// Trace to a solved endpoint and require the exact proposed triangle+instance.
// This simultaneously verifies finite-facet membership, rejects Cartesian-domain
// pairs absent from the selected BLAS, and rejects any nearer occluder.
fn mneeTraceReachesFacet(
  origin: vec3f,
  endpoint: vec3f,
  facet: MneeFacetProposal,
) -> bool {
  let delta = endpoint - origin;
  let distance = length(delta);
  let epsilon = mneeScaleAwareEpsilon(origin, distance);
  if (!(distance > 2.0 * epsilon)) { return false; }
  let direction = delta / distance;
  let rayOrigin = origin + direction * epsilon;
  let expectedDistance = distance - epsilon;
  let tolerance = mneeScaleAwareEpsilon(endpoint, distance);
  let hit = traceClosest(
    Ray(rayOrigin, direction),
    max(epsilon * 0.25, bitcast<f32>(0x00800000u)),
    expectedDistance + tolerance,
  );
  return hit.didHit &&
    hit.triIndex == facet.triIndex &&
    hit.instanceIndex == facet.instanceIndex &&
    abs(hit.dist - expectedDistance) <= tolerance;
}

// Conservative external-leg visibility. Only numerical self-hits on the exact
// selected triangle+instance may be stepped through. A distinct mirror or glass
// facet is a real occluder unless it is explicitly part of the solved chain.
fn mneeSegmentBlockedExceptFacet(
  origin: vec3f,
  dir: vec3f,
  maxDist: f32,
  selfFacet: MneeFacetProposal,
) -> bool {
  let finiteDistance = maxDist == maxDist && abs(maxDist) < INFINITY;
  let firstEpsilon = mneeScaleAwareEpsilon(origin, maxDist);
  if (finiteDistance && maxDist <= 2.0 * firstEpsilon) { return false; }
  var segOrigin = origin + dir * firstEpsilon;
  var remaining = select(maxDist, maxDist - firstEpsilon, finiteDistance);
  for (var stepN = 0u; stepN < 4u; stepN = stepN + 1u) {
    let epsilon = mneeScaleAwareEpsilon(segOrigin, remaining);
    let traceMax = select(
      INFINITY,
      max(remaining + epsilon, epsilon),
      finiteDistance,
    );
    let segHit = traceClosest(
      Ray(segOrigin, dir), epsilon, traceMax,
    );
    if (!segHit.didHit) { return false; }
    let exactSelf = segHit.triIndex == selfFacet.triIndex &&
      segHit.instanceIndex == selfFacet.instanceIndex;
    if (!exactSelf) { return true; }
    let hitPoint = segOrigin + dir * segHit.dist;
    let advance = segHit.dist + mneeScaleAwareEpsilon(hitPoint, remaining);
    segOrigin = segOrigin + dir * advance;
    if (finiteDistance) {
      remaining = remaining - advance;
      if (remaining <= mneeScaleAwareEpsilon(segOrigin, remaining)) {
        return false;
      }
    }
  }
  return true;
}

fn configuredMneeIterations(maxSupported: u32) -> u32 {
  return clamp(params.mneeMaxIterations, 1u, maxSupported);
}


// ── General bounded MNEE chain (1..8 vertices) ──────────────────────────────
// Length, every facet, and every reflection/transmission event are proposed from
// explicit distributions. Invalid topology/material/visibility combinations are
// zero-valued samples; successful paths divide by the complete proposal PMF.
// The coupled solver is the O(N) block-tridiagonal Newton implementation in
// mneeNewton.wgsl.ts, not a sequence of independent one-vertex solves.
struct MneeChainMediumStack {
  depth: u32,
  valid: u32,
  matIds: array<u32, ${MNEE_CHAIN_MAX_VERTICES}>,
  boundaryKinds: array<u32, ${MNEE_CHAIN_MAX_VERTICES}>,
  boundaryIndices: array<u32, ${MNEE_CHAIN_MAX_VERTICES}>,
  boundaryComponents: array<u32, ${MNEE_CHAIN_MAX_VERTICES}>,
  iors: array<f32, ${MNEE_CHAIN_MAX_VERTICES}>,
  sigmaT: array<vec3f, ${MNEE_CHAIN_MAX_VERTICES}>,
  remainingDistance: array<f32, ${MNEE_CHAIN_MAX_VERTICES}>,
}

struct MneeChainBoundaryState {
  stack: MneeChainMediumStack,
  etaI: f32,
  etaT: f32,
}

struct MneeChainSegmentState {
  stack: MneeChainMediumStack,
  transmittance: vec3f,
}

fn mneeChainPushMedium(
  stackIn: MneeChainMediumStack,
  optics: MneeFacetOptics,
) -> MneeChainMediumStack {
  var stack = stackIn;
  if (
    stack.depth >= ${MNEE_CHAIN_MAX_VERTICES}u ||
    !mediumBoundaryIsValid(vec3u(
      optics.boundaryKind,
      optics.boundaryIndex,
      optics.boundaryComponent,
    ))
  ) {
    stack.valid = 0u;
    return stack;
  }
  let index = stack.depth;
  stack.matIds[index] = optics.matId;
  stack.boundaryKinds[index] = optics.boundaryKind;
  stack.boundaryIndices[index] = optics.boundaryIndex;
  stack.boundaryComponents[index] = optics.boundaryComponent;
  stack.iors[index] = max(optics.ior, 1e-4);
  stack.sigmaT[index] = optics.sigmaT;
  stack.remainingDistance[index] = select(
    3.402823e38,
    max(optics.volumeThickness, 0.0),
    optics.hasVolumeThickness,
  );
  stack.depth = stack.depth + 1u;
  return stack;
}

fn mneeChainStackAtLaunch(
  origin: vec3f,
  direction: vec3f,
  heroLambda: f32,
) -> MneeChainMediumStack {
  var stack: MneeChainMediumStack;
  stack.depth = 0u;
  stack.valid = 1u;
  let containment = opticalContainmentAlongRay(origin, direction);
  if (!containment.valid || containment.depth > ${MNEE_CHAIN_MAX_VERTICES}u) {
    stack.valid = 0u;
    return stack;
  }
  stack.depth = containment.depth;
  for (var index = 0u; index < stack.depth; index = index + 1u) {
    let matId = containment.matIds[index];
    let mat = materialAtOpticalBoundary(
      matId,
      containment.triIndices[index],
      containment.baryVWs[index],
      containment.instanceIndices[index],
    );
    if (!mat.isBulkMedium) {
      stack.valid = 0u;
      return stack;
    }
    var mediumIor = mat.ior;
    if (params.spectralEnabled != 0u && mat.dispersionAbbe > 0.0) {
      mediumIor = cauchyIorAtLambda(
        heroLambda, mat.ior, mat.dispersionAbbe,
      );
    }
    let boundary = containment.boundaries[index];
    stack.matIds[index] = matId;
    stack.boundaryKinds[index] = boundary.x;
    stack.boundaryIndices[index] = boundary.y;
    stack.boundaryComponents[index] = boundary.z;
    stack.iors[index] = max(mediumIor, 1e-4);
    stack.sigmaT[index] = mneeMaterialSigmaT(matId, mat, heroLambda);
    stack.remainingDistance[index] =
      materialAttenuationDistance(INFINITY, mat);
  }
  return stack;
}

// Match the eye walk's nested-medium eta lookup. Launch points are reconstructed
// before the chain is evaluated; a first back face is never inferred locally.
fn mneeChainPrepareBoundary(
  stackIn: MneeChainMediumStack,
  optics: MneeFacetOptics,
) -> MneeChainBoundaryState {
  var stack = stackIn;
  var etaI = 1.0;
  if (stack.depth > 0u) {
    etaI = stack.iors[stack.depth - 1u];
  } else if (!optics.frontFace && optics.isBulkMedium) {
    etaI = max(optics.ior, 1e-4);
  }
  var etaT = max(optics.ior, 1e-4);
  if (!optics.isBulkMedium) {
    // A compound sheet returns to its incident medium after its virtual second
    // interface, so it introduces no persistent eta transition in the chain.
    etaT = etaI;
  } else if (!optics.frontFace) {
    etaT = 1.0;
    if (
      stack.depth == 0u ||
      stack.matIds[stack.depth - 1u] != optics.matId ||
      !mediumBoundaryMatches(
        stack.boundaryKinds[stack.depth - 1u],
        stack.boundaryIndices[stack.depth - 1u],
        stack.boundaryComponents[stack.depth - 1u],
        vec3u(
          optics.boundaryKind,
          optics.boundaryIndex,
          optics.boundaryComponent,
        ),
      )
    ) {
      stack.valid = 0u;
    } else if (stack.depth > 1u) {
      etaT = stack.iors[stack.depth - 2u];
    }
  }
  var out: MneeChainBoundaryState;
  out.stack = stack;
  out.etaI = etaI;
  out.etaT = etaT;
  return out;
}

fn mneeChainCommitTransmission(
  stackIn: MneeChainMediumStack,
  optics: MneeFacetOptics,
) -> MneeChainMediumStack {
  var stack = stackIn;
  if (!optics.isBulkMedium) { return stack; }
  if (optics.frontFace) {
    return mneeChainPushMedium(stack, optics);
  }
  if (
    stack.depth == 0u ||
    stack.matIds[stack.depth - 1u] != optics.matId ||
    !mediumBoundaryMatches(
      stack.boundaryKinds[stack.depth - 1u],
      stack.boundaryIndices[stack.depth - 1u],
      stack.boundaryComponents[stack.depth - 1u],
      vec3u(
        optics.boundaryKind,
        optics.boundaryIndex,
        optics.boundaryComponent,
      ),
    )
  ) {
    stack.valid = 0u;
    return stack;
  }
  stack.depth = stack.depth - 1u;
  return stack;
}

fn mneeChainAttenuateSegment(
  stackIn: MneeChainMediumStack,
  segmentDistance: f32,
) -> MneeChainSegmentState {
  var out: MneeChainSegmentState;
  out.stack = stackIn;
  out.transmittance = vec3f(1.0);
  if (out.stack.depth == 0u) { return out; }
  let top = out.stack.depth - 1u;
  let distanceInMedium = min(segmentDistance, out.stack.remainingDistance[top]);
  out.transmittance = materialBeer(
    out.stack.sigmaT[top], distanceInMedium,
  );
  out.stack.remainingDistance[top] = max(
    out.stack.remainingDistance[top] - distanceInMedium, 0.0,
  );
  return out;
}

const MNEE_EMITTER_DIRECTIONAL = 0u;
const MNEE_EMITTER_POINT = 1u;
const MNEE_EMITTER_SPOT = 2u;
const MNEE_EMITTER_AREA = 3u;
const MNEE_GENERAL_PROPOSAL_COUNT = 4u;

struct MneeEmitterSample {
  valid: u32,
  kind: u32,
  sourceMode: u32,
  shadowDisabled: u32,
  twoSided: u32,
  position: vec3f,
  towardLight: vec3f,
  radiance: vec3f,
  areaU: vec3f,
  areaV: vec3f,
  area: f32,
  maxDistance: f32,
  decay: f32,
  spotAxis: vec3f,
  cosOuter: f32,
  cosInner: f32,
  selectionPdf: f32,
}

fn mneeSampleEmitter(
  rng: ptr<function, PtRngState>,
) -> MneeEmitterSample {
  var out: MneeEmitterSample;
  out.valid = 0u;
  out.kind = MNEE_EMITTER_POINT;
  out.sourceMode = MNEE_SOURCE_FINITE;
  out.shadowDisabled = 0u;
  out.twoSided = 0u;
  out.position = vec3f(0.0);
  out.towardLight = vec3f(0.0, 1.0, 0.0);
  out.radiance = vec3f(0.0);
  out.areaU = vec3f(1.0, 0.0, 0.0);
  out.areaV = vec3f(0.0, 1.0, 0.0);
  out.area = 0.0;
  out.maxDistance = 0.0;
  out.decay = 0.0;
  out.spotAxis = vec3f(0.0, -1.0, 0.0);
  out.cosOuter = -1.0;
  out.cosInner = -1.0;
  out.selectionPdf = 0.0;

  let directionalCount = params.directionalLightCount;
  let pointCount = params.pointLightCount;
  let spotCount = params.spotLightCount;
  let rectCount = params.rectAreaLightCount;
  let meshCount = params.meshAreaLightCount;
  let total = directionalCount + pointCount + spotCount + rectCount + meshCount;
  if (total == 0u) { return out; }
  let picked = causticUniformEmitterIndex(rng, total);
  out.selectionPdf = 1.0 / f32(total);
  out.valid = 1u;

  var cursor = 0u;
  if (picked < cursor + directionalCount) {
    let index = picked - cursor;
    let base = index * 2u;
    let dirAndDiameter = directionalLights[base];
    let rawDiameter = dirAndDiameter.w;
    out.shadowDisabled = select(0u, 1u, rawDiameter < 0.0);
    let diameter = select(rawDiameter, -1.0 - rawDiameter, rawDiameter < 0.0);
    out.kind = MNEE_EMITTER_DIRECTIONAL;
    out.sourceMode = MNEE_SOURCE_DIRECTIONAL;
    out.towardLight = sampleDirectionalCone(
      rng, safe_normalize(dirAndDiameter.xyz), max(diameter, 0.0),
    );
    out.radiance = directionalLights[base + 1u].rgb;
    return out;
  }
  cursor = cursor + directionalCount;

  if (picked < cursor + pointCount) {
    let index = picked - cursor;
    let base = index * POINT_LIGHT_VEC4_STRIDE;
    let extra = pointLights[base + 2u];
    out.kind = MNEE_EMITTER_POINT;
    out.position = pointLights[base].xyz;
    out.radiance = pointLights[base + 1u].rgb;
    out.maxDistance = extra.x;
    out.decay = extra.y;
    out.shadowDisabled = select(0u, 1u, extra.z > 0.5);
    return out;
  }
  cursor = cursor + pointCount;

  if (picked < cursor + spotCount) {
    let index = picked - cursor;
    let base = index * SPOT_LIGHT_VEC4_STRIDE;
    let axis = spotLights[base + 1u];
    let radianceAndInner = spotLights[base + 2u];
    let extra = spotLights[base + 3u];
    out.kind = MNEE_EMITTER_SPOT;
    out.position = spotLights[base].xyz;
    out.spotAxis = safe_normalize(axis.xyz);
    out.cosOuter = axis.w;
    out.cosInner = radianceAndInner.w;
    out.radiance = radianceAndInner.rgb;
    out.maxDistance = extra.x;
    out.decay = extra.y;
    out.shadowDisabled = select(0u, 1u, extra.z > 0.5);
    return out;
  }
  cursor = cursor + spotCount;

  if (picked < cursor + rectCount) {
    let index = picked - cursor;
    let base = index * 4u;
    let centerAndShadow = rectAreaLights[base];
    let u = rectAreaLights[base + 1u].xyz;
    let v = rectAreaLights[base + 2u].xyz;
    let radianceAndShape = rectAreaLights[base + 3u];
    let xi = vec2f(rand_f32(rng), rand_f32(rng));
    let isDisc = abs(radianceAndShape.w - 1.0) < 0.5;
    let areaMeasure = measureAreaVector(
      u, v, select(4.0, PI, isDisc),
    );
    if (isDisc) {
      let disc = concentricDiscSample(xi * 2.0 - vec2f(1.0));
      out.position = centerAndShadow.xyz + u * disc.x + v * disc.y;
    } else {
      out.position = centerAndShadow.xyz +
        u * (xi.x * 2.0 - 1.0) + v * (xi.y * 2.0 - 1.0);
    }
    out.area = areaMeasure.area;
    out.kind = MNEE_EMITTER_AREA;
    out.areaU = u;
    out.areaV = v;
    out.radiance = radianceAndShape.rgb;
    out.shadowDisabled = select(0u, 1u, centerAndShadow.w > 0.5);
    out.valid = areaMeasure.valid;
    return out;
  }
  cursor = cursor + rectCount;

  let index = picked - cursor;
  let base = meshAreaLightBase(index);
  let a = meshAreaLights[base].xyz;
  let b = meshAreaLights[base + 1u].xyz;
  let c = meshAreaLights[base + 2u].xyz;
  let r1 = rand_f32(rng);
  let r2 = rand_f32(rng);
  let root = sqrt(r1);
  let wa = 1.0 - root;
  let wb = r2 * root;
  let wc = 1.0 - wa - wb;
  out.kind = MNEE_EMITTER_AREA;
  out.position = a * wa + b * wb + c * wc;
  out.areaU = b - a;
  out.areaV = c - a;
  let areaMeasure = measureAreaVector(out.areaU, out.areaV, 0.5);
  out.area = areaMeasure.area;
  out.radiance = sampleMeshAreaLightRadiance(
    index, vec3f(wa, wb, wc), out.position,
  );
  out.shadowDisabled = select(
    0u, 1u, meshAreaLights[base + 3u].w > 0.5,
  );
  out.twoSided = select(0u, 1u, meshAreaLightIsTwoSided(index));
  out.valid = areaMeasure.valid;
  return out;
}

fn mneeEmitterScalarFactor(
  emitter: MneeEmitterSample,
  firstVertex: vec3f,
  pathDistance: f32,
) -> f32 {
  if (emitter.kind == MNEE_EMITTER_POINT) {
    return pointSpotPathMeasureScale(
      pathDistance, emitter.maxDistance, emitter.decay,
    );
  }
  if (emitter.kind == MNEE_EMITTER_SPOT) {
    let lightToVertex = safe_normalize(firstVertex - emitter.position);
    let coneCos = dot(lightToVertex, emitter.spotAxis);
    if (coneCos < emitter.cosOuter) { return 0.0; }
    let softness = smoothstep(
      emitter.cosOuter, max(emitter.cosInner, emitter.cosOuter + 1e-6), coneCos,
    );
    return softness * pointSpotPathMeasureScale(
      pathDistance, emitter.maxDistance, emitter.decay,
    );
  }
  if (emitter.kind == MNEE_EMITTER_AREA) {
    let areaMeasure = measureAreaVector(
      emitter.areaU, emitter.areaV, 1.0,
    );
    if (areaMeasure.valid == 0u) { return 0.0; }
    let lightNormal = areaMeasure.normal;
    let signedCosine = dot(
      lightNormal, safe_normalize(firstVertex - emitter.position),
    );
    return select(
      0.0,
      1.0,
      select(signedCosine > 1e-5, abs(signedCosine) > 1e-5, emitter.twoSided != 0u),
    );
  }
  return 1.0;
}


// Unified planar-manifold estimator. It owns chain lengths 1..8 for every
// explicit emitter family, so the old point-only one/two/general helpers below
// are no longer composed as separate estimators (which would overlap paths).
fn boundedManifoldCaustic(
  rng: ptr<function, PtRngState>,
  hitPos: vec3f,
  normal: vec3f,
  clearcoatNormal: vec3f,
  wo: vec3f,
  baseColor: vec3f,
  roughness: f32,
  metallic: f32,
  transmission: f32,
  etaTOverI: f32,
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
  thinFilm: ThinFilmInterface,
  throughput: vec3f,
  heroLambda: f32,
) -> vec3f {
  let maximumLength = min(params.mneeMaxChainLength, ${MNEE_CHAIN_MAX_VERTICES}u);
  if (maximumLength == 0u || mneeFacetCandidateCount() == 0u) {
    return vec3f(0.0);
  }
  let emitter = mneeSampleEmitter(rng);
  if (emitter.valid == 0u ||
      max(emitter.radiance.r, max(emitter.radiance.g, emitter.radiance.b)) <= 0.0) {
    return vec3f(0.0);
  }
  let lengthSelectionPdf = 1.0 / f32(maximumLength);
  var recvTu: vec3f;
  var recvTv: vec3f;
  buildOnb(normal, &recvTu, &recvTv);
  var contribution = vec3f(0.0);

  for (var attempt = 0u; attempt < MNEE_GENERAL_PROPOSAL_COUNT; attempt = attempt + 1u) {
    let chainLength = 1u + ptRandBoundedU32(rng, maximumLength);
    var geometry: MneeBoundedChainGeometry;
    geometry.count = chainLength;
    geometry.sourceMode = emitter.sourceMode;
    geometry.sourceDirection = emitter.towardLight;
    var media: MneeBoundedChainMedia;
    var facets: array<MneeFacetProposal, ${MNEE_CHAIN_MAX_VERTICES}>;
    var eventModes: array<u32, ${MNEE_CHAIN_MAX_VERTICES}>;
    var seedVertices: array<vec3f, ${MNEE_CHAIN_MAX_VERTICES}>;
    var seedStack: MneeChainMediumStack;
    seedStack.depth = 0u;
    seedStack.valid = 1u;
    var proposalValid = true;
    var previousSeed = emitter.position;
    let finiteDirection = hitPos - emitter.position;

    for (var ci = 0u; ci < ${MNEE_CHAIN_MAX_VERTICES}u; ci = ci + 1u) {
      if (ci >= chainLength) { break; }
      let eventMode = select(
        MNEE_CHAIN_EVENT_REFLECTION,
        MNEE_CHAIN_EVENT_TRANSMISSION,
        rand_f32(rng) >= 0.5,
      );
      eventModes[ci] = eventMode;
      let guideOrdinal = select(
        ci, chainLength - 1u - ci,
        emitter.sourceMode == MNEE_SOURCE_DIRECTIONAL,
      );
      let facet = mneeProposeConditionalFacet(
        rng, emitter.sourceMode, emitter.position, emitter.towardLight,
        hitPos, guideOrdinal, eventMode, heroLambda,
      );
      facets[ci] = facet;
      if (facet.valid == 0u || !(facet.pdf > 0.0)) {
        proposalValid = false;
        break;
      }
      geometry.planeP[ci] = facet.p;
      geometry.normal[ci] = facet.n;
      // Naga's SPIR-V backend cannot materialize dynamically indexed array
      // element pointers as function-call arguments. Build into local addressable
      // values and then store the completed frame in the chain arrays.
      var tangentU = vec3f(0.0);
      var tangentV = vec3f(0.0);
      buildOnb(facet.n, &tangentU, &tangentV);
      geometry.tangentU[ci] = tangentU;
      geometry.tangentV[ci] = tangentV;

      var seed = facet.p;
      if (emitter.sourceMode == MNEE_SOURCE_DIRECTIONAL) {
        let denominator = dot(emitter.towardLight, facet.n);
        if (abs(denominator) > 1e-7) {
          let t = dot(facet.p - hitPos, facet.n) / denominator;
          seed = hitPos + emitter.towardLight * t;
        }
      } else {
        let denominator = dot(finiteDirection, facet.n);
        if (abs(denominator) > 1e-7) {
          let t = dot(facet.p - emitter.position, facet.n) / denominator;
          seed = emitter.position + finiteDirection * t;
        }
      }
      seedVertices[ci] = seed;

      var seedIncoming = safe_normalize(facet.p - previousSeed);
      if (ci == 0u && emitter.sourceMode == MNEE_SOURCE_DIRECTIONAL) {
        seedIncoming = -emitter.towardLight;
      } else if (ci == 0u) {
        seedStack = mneeChainStackAtLaunch(
          emitter.position, seedIncoming, heroLambda,
        );
        if (seedStack.valid == 0u) {
          proposalValid = false;
          break;
        }
      }
      let seedOptics = mneeFacetOpticsAt(
        facet, facet.p, seedIncoming, heroLambda,
      );
      let boundary = mneeChainPrepareBoundary(seedStack, seedOptics);
      seedStack = boundary.stack;
      if (seedStack.valid == 0u) {
        proposalValid = false;
        break;
      }
      if (eventMode == MNEE_CHAIN_EVENT_REFLECTION) {
        media.etaI[ci] = 1.0;
        media.etaT[ci] = 1.0;
      } else {
        media.etaI[ci] = boundary.etaI;
        media.etaT[ci] = boundary.etaT;
        seedStack = mneeChainCommitTransmission(seedStack, seedOptics);
        if (seedStack.valid == 0u) {
          proposalValid = false;
          break;
        }
      }
      previousSeed = facet.p;
    }
    if (!proposalValid) { continue; }

    let solverLightPoint = select(
      emitter.position,
      hitPos + emitter.towardLight,
      emitter.sourceMode == MNEE_SOURCE_DIRECTIONAL,
    );
    let solved = mneeNewtonSolveChainBounded(
      geometry, media, solverLightPoint, hitPos, seedVertices,
      configuredMneeIterations(${MNEE_CHAIN_MAX_ITERS}u),
    );
    if (solved.valid == 0u) { continue; }

    if (!mneeTraceReachesFacet(
      hitPos,
      solved.vertices[chainLength - 1u],
      facets[chainLength - 1u],
    )) { continue; }
    var identityValid = true;
    var reverseIndex = chainLength - 1u;
    loop {
      if (reverseIndex == 0u) { break; }
      let targetIndex = reverseIndex - 1u;
      if (!mneeTraceReachesFacet(
        solved.vertices[reverseIndex],
        solved.vertices[targetIndex],
        facets[targetIndex],
      )) {
        identityValid = false;
        break;
      }
      reverseIndex = reverseIndex - 1u;
    }
    if (!identityValid) { continue; }

    if (emitter.shadowDisabled == 0u) {
      if (emitter.sourceMode == MNEE_SOURCE_DIRECTIONAL) {
        if (mneeSegmentBlockedExceptFacet(
          solved.vertices[0],
          emitter.towardLight, INFINITY, facets[0],
        )) { continue; }
      } else {
        let toSource = emitter.position - solved.vertices[0];
        let sourceDistance = length(toSource);
        let sourceDirection = toSource / max(sourceDistance, 1e-8);
        if (mneeSegmentBlockedExceptFacet(
          solved.vertices[0],
          sourceDirection, sourceDistance, facets[0],
        )) { continue; }
      }
    }

    var solvedStack: MneeChainMediumStack;
    solvedStack.depth = 0u;
    solvedStack.valid = 1u;
    if (emitter.sourceMode != MNEE_SOURCE_DIRECTIONAL) {
      let solvedLaunchDirection = safe_normalize(
        solved.vertices[0] - emitter.position,
      );
      solvedStack = mneeChainStackAtLaunch(
        emitter.position, solvedLaunchDirection, heroLambda,
      );
      if (solvedStack.valid == 0u) { continue; }
    }
    var logInterfaceNumerator = vec3f(0.0);
    var interfacePositive = vec3<bool>(true);
    var volumeTransmittance = vec3f(1.0);
    var logFacetEventPdf = 0.0;
    var previousPoint = emitter.position;
    var pathDistance = 0.0;
    var physicsValid = true;
    for (var pi = 0u; pi < ${MNEE_CHAIN_MAX_VERTICES}u; pi = pi + 1u) {
      if (pi >= chainLength) { break; }
      let vertex = solved.vertices[pi];
      var incomingTravel = safe_normalize(vertex - previousPoint);
      var segmentDistance = length(vertex - previousPoint);
      if (pi == 0u && emitter.sourceMode == MNEE_SOURCE_DIRECTIONAL) {
        incomingTravel = -emitter.towardLight;
        segmentDistance = 0.0;
      }
      if (mneeFacetHasMappedInterface(facets[pi])) {
        physicsValid = false;
        break;
      }
      let optics = mneeFacetOpticsAt(
        facets[pi], vertex, incomingTravel, heroLambda,
      );
      let boundary = mneeChainPrepareBoundary(solvedStack, optics);
      solvedStack = boundary.stack;
      if (solvedStack.valid == 0u) {
        physicsValid = false;
        break;
      }
      pathDistance = pathDistance + segmentDistance;
      let segment = mneeChainAttenuateSegment(solvedStack, segmentDistance);
      solvedStack = segment.stack;
      volumeTransmittance = volumeTransmittance * segment.transmittance;
      let microfacetCos = abs(dot(-incomingTravel, facets[pi].n));
      let etaRatio = boundary.etaT / max(boundary.etaI, 1e-4);
      var eventFactor = vec3f(0.0);

      if (eventModes[pi] == MNEE_CHAIN_EVENT_REFLECTION) {
        if (!mneeFacetHasDeltaReflection(optics)) {
          physicsValid = false;
          break;
        }
        eventFactor = mneeFacetReflectionFactorWithEta(
          optics, microfacetCos, heroLambda, etaRatio,
        );
      } else {
        if (!mneeFacetHasDeltaTransmission(optics) ||
            abs(media.etaI[pi] - boundary.etaI) > 1e-3 ||
            abs(media.etaT[pi] - boundary.etaT) > 1e-3) {
          physicsValid = false;
          break;
        }
        eventFactor = mneeFacetTransmissionFactorWithEta(
          optics, microfacetCos, heroLambda, etaRatio, boundary.etaI,
        );
        solvedStack = mneeChainCommitTransmission(solvedStack, optics);
        if (solvedStack.valid == 0u) {
          physicsValid = false;
          break;
        }
      }
      let conditionalPdf = facets[pi].pdf * 0.5;
      logFacetEventPdf = logFacetEventPdf + log(conditionalPdf);
      interfacePositive = interfacePositive & (eventFactor > vec3f(0.0));
      logInterfaceNumerator = logInterfaceNumerator +
        log(max(eventFactor, vec3f(bitcast<f32>(0x00800000u))));
      previousPoint = vertex;
    }
    if (!physicsValid || !any(interfacePositive)) {
      continue;
    }
    let receiverDistance = length(hitPos - previousPoint);
    pathDistance = pathDistance + receiverDistance;
    let receiverSegment = mneeChainAttenuateSegment(
      solvedStack, receiverDistance,
    );
    volumeTransmittance = volumeTransmittance * receiverSegment.transmittance;

    let wi = safe_normalize(solved.vertices[chainLength - 1u] - hitPos);
    let nDotL = max(dot(normal, wi), 0.0);
    if (nDotL <= 1e-5) { continue; }
    let fr = evaluateFiniteBsdfFullWithClearcoatNormal(
      baseColor, roughness, metallic, transmission, etaTOverI,
      normal, clearcoatNormal, wo, wi,
      clearcoat, clearcoatRoughness, sheen, sheenRoughness, sheenColor,
      iridescence, iridescenceIor, iridescenceThicknessMin, iridescenceThicknessMax,
      specularColor, specularIntensity,
      anisotropy, anisotropyRotation, thinFilm, true,
    );
    let emitterFactor = mneeEmitterScalarFactor(
      emitter, solved.vertices[0], pathDistance,
    );
    if (!(emitterFactor > 0.0) || !(emitterFactor < INFINITY)) { continue; }

    var pathMeasure = 0.0;
    if (emitter.kind == MNEE_EMITTER_AREA) {
      let areaDet = mneeBoundedChainAreaPdfDet(
        geometry, media, solved, emitter.position, hitPos,
        emitter.areaU, emitter.areaV,
        configuredMneeIterations(${MNEE_CHAIN_MAX_ITERS}u),
      );
      if (!(areaDet > 0.0) || !(areaDet < INFINITY)) { continue; }
      if (!(emitter.area > 0.0) || !(emitter.area < INFINITY)) { continue; }
      let endpointPdf = (1.0 / emitter.area) / areaDet;
      if (!(endpointPdf > 0.0) || !(endpointPdf < INFINITY)) { continue; }
      // Exact ownership replaces the old one-sided MIS weight: the eye kernel
      // suppresses mesh-emitter hits reached through 1..configuredMax delta
      // events, and BDPT rejects the same bounded all-delta finite-source
      // prefix. MNEE is therefore the sole estimator for this area-light family.
      pathMeasure = nDotL * emitter.area * areaDet;
    } else if (emitter.sourceMode == MNEE_SOURCE_DIRECTIONAL) {
      pathMeasure = mneeBoundedChainDirectionalFocusingDet(
        geometry, media, solved, solverLightPoint, hitPos, recvTu, recvTv,
        configuredMneeIterations(${MNEE_CHAIN_MAX_ITERS}u),
      );
    } else {
      pathMeasure = mneeBoundedChainFocusingDet(
        geometry, media, solved, emitter.position, hitPos, recvTu, recvTv,
        configuredMneeIterations(${MNEE_CHAIN_MAX_ITERS}u),
      );
    }
    if (!(pathMeasure > 0.0) || !(pathMeasure < INFINITY)) { continue; }
    let lightOut = mneeEmitterRadiance(emitter.radiance, heroLambda);
    let positiveChannels = interfacePositive &
      (throughput > vec3f(0.0)) &
      (fr > vec3f(0.0)) &
      (lightOut > vec3f(0.0)) &
      (volumeTransmittance > vec3f(0.0));
    let minPositive = vec3f(bitcast<f32>(0x00800000u));
    let logContribution =
      log(max(throughput, minPositive)) +
      log(max(fr, minPositive)) +
      log(max(lightOut, minPositive)) +
      logInterfaceNumerator - vec3f(logFacetEventPdf) +
      log(max(volumeTransmittance, minPositive)) +
      vec3f(
        log(pathMeasure) + log(emitterFactor) -
        log(lengthSelectionPdf) - log(emitter.selectionPdf) -
        log(f32(MNEE_GENERAL_PROPOSAL_COUNT))
      );
    let sampleContribution = mneeSaturatedExpRgb(
      logContribution, positiveChannels,
    );
    contribution = mneeSaturatedAddRgb(contribution, sampleContribution);
  }
  return contribution;
}


fn manifoldNeeContribution(
  rng: ptr<function, PtRngState>,
  hitPos: vec3f,
  normal: vec3f,
  clearcoatNormal: vec3f,
  wo: vec3f,
  baseColor: vec3f,
  roughness: f32,
  metallic: f32,
  transmission: f32,
  etaTOverI: f32,
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
  thinFilm: ThinFilmInterface,
  heroLambda: f32,
  throughput: vec3f,
) -> vec3f {
  // One ownership path for LS+E, LSS+E, ... through the configured bound.
  // Endpoint and event families are sampled inside this estimator, so no
  // specialized sibling can double count a path.
  return boundedManifoldCaustic(
    rng, hitPos, normal, clearcoatNormal, wo, baseColor, roughness, metallic,
    transmission, etaTOverI,
    clearcoat, clearcoatRoughness, sheen, sheenRoughness, sheenColor,
    iridescence, iridescenceIor, iridescenceThicknessMin, iridescenceThicknessMax,
    specularColor, specularIntensity,
    anisotropy, anisotropyRotation, thinFilm,
    throughput, heroLambda,
  );
}

// ── SPPM gather (causticStrategy == 2) ────────────────────────────────────────
// A4-progressive: true Hachisuka & Jensen 2009 SPPM with per-pixel progressive
// statistics (τ, linear R, N). The photon-emission pass (sppmEmitPhotons in
// sppmBindings.wgsl.ts / the separate sppmPhotonPass pipeline) runs BEFORE the
// megakernel each frame and re-populates the hash grid with fresh photons.
// This update calls sppmUpdateSurfaceProgressive which:
//   (1) reads surface (τ, R, N) from sppmPixelStats[pixelIndex*2],
//   (2) collects M new photons within the current radius R,
//   (3) applies the Hachisuka §4 update: N'=N+αM, ratio=N'/(N+M),
//       R'²=R²·ratio, τ'=(τ+Φ_M)·ratio,
//   (4) writes (τ', R'², N') back. Readback is a separate final-kernel step.
//
// The active gather consumes only the current frame's linked hash grid.
//
// Accumulator interaction: this function mutates one visible-point measure but
// does not return radiance. The kernel reads the current surface+volume
// cumulative estimate once per SPPM-owned frame, including no-receiver frames;
// that avoids applying receiver-event probability twice.
//
// Provenance: Hachisuka & Jensen 2009 "Stochastic Progressive Photon Mapping"
// (ACM SIGGRAPH Asia 2009 §4); Knaus & Zwicker 2011 formulation of the
// progressive update rule.
fn photonMapUpdateProgressive(
  pixelIndex: u32,
  hitPos: vec3f,
  normal: vec3f,
  clearcoatNormal: vec3f,
  wo: vec3f,
  baseColor: vec3f,
  roughness: f32,
  metallic: f32,
  transmission: f32,
  etaTOverI: f32,
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
  thinFilm: ThinFilmInterface,
  throughput: vec3f,
  heroLambda: f32,
  heroPdf: f32,
  absorbedFluxInvPdf: f32,
) {
  sppmUpdateSurfaceProgressive(
    pixelIndex, hitPos, normal, clearcoatNormal, wo,
    baseColor, roughness, metallic, transmission, etaTOverI,
    clearcoat, clearcoatRoughness, sheen, sheenRoughness, sheenColor,
    iridescence, iridescenceIor, iridescenceThicknessMin, iridescenceThicknessMax,
    specularColor, specularIntensity,
    anisotropy, anisotropyRotation, thinFilm,
    throughput,
    heroLambda, heroPdf, absorbedFluxInvPdf,
  );
}
`;
