/**
 * BDPT eye↔light connection — full Veach §10.3 multi-strategy MIS (WebGPU).
 *
 * Computes the power-heuristic (β=2) MIS weight for ONE explicit connection
 * (current eye-subpath bounce `E_e` → one stored light-subpath vertex `L_c`) by
 * enumerating ALL Veach §10.3 strategy path-pdfs over the merged path
 *
 *   v[0]=L_0(emitter) … v[c]=L_c | v[c+1]=E_e … v[c+1+e]=E_0 | v[n-1]=camera
 *
 * with `n = c+e+3` and `selectedS = c+1`. This is the canonical PBRT-v4
 * `MISWeight` recurrence: a pure ratio of AREA-measure forward/reverse densities
 * walked over the actual vertices, with the per-vertex solid-angle pdf converted
 * to area on the fly via `convertDensitySAtoArea` (PBRT `Vertex::ConvertDensity`,
 * a DESTINATION-cosine-only "half-G"). The four pdfs straddling the connection
 * edge are recomputed here from the connection geometry (PBRT's pt/ptMinus/qs/
 * qsMinus pdfRev overrides); the eye-side overrides use `brdfDirectionalPdf` with
 * wo/wi as required (D1 — PBRT-correct non-symmetric reverse density). The eye
 * prefix (E_0…E_e construction-time SA pdfs + pos/normal/specular) is read from a
 * fixed per-invocation private array threaded through the eye loop. Emitter endpoints
 * keep the Lambertian/emission endpoint profile; surface light vertices use the
 * row-3/row-4 material payload and real BSDF/PDF path from the light-subpath
 * kernel.
 *
 * This is a 1:1 port of the CPU reference `bdpt/bdptConnectionMisFull.ts`, which
 * is pinned to `@vitrum/shared-samplers`'s `bdptConnectionMIS_full` /
 * `buildBDPTStrategyPDFs_full` oracle to ~1e-12. The whole module compiles only
 * into the full-tier shader, and the kernel calls `evaluateBdptConnection` only
 * under `params.bdptEnabled != 0u`, so the BDPT-off path is bit-identical.
 *
 * References:
 *   Veach 1997 §10.3 (BDPT MIS weights), §9.2 (power heuristic, β=2),
 *     §10.3.5 (specular zero-weight rule), §8.3.2 (geometry term).
 *   Pharr et al. 2023, PBR 4e §16.3.5 Eq. 16.16; integrators.cpp MISWeight.
 *   @vitrum/shared-samplers: bdptConnectionMIS_full, buildBDPTStrategyPDFs_full.
 */
export const PT_WEBGPU_BDPT_CONNECTION_WGSL = /* wgsl */ `
const BDPT_KIND_INVALID: f32 = 3.0;
const BDPT_KIND_DELTA: f32 = 1.0;
const BDPT_MAX_MERGED: u32 = 19u; // c(<=7) + e(<=7) + 3, with headroom
const BDPT_MAX_INFINITE_STRATEGIES: u32 = 10u;

const BDPT_MEDIUM_STACK_LIMIT: u32 = 8u;
const BDPT_LV_MEDIUM_MATID: f32 = -7.0;
const BDPT_NO_MEDIUM: u32 = 0xffffffffu;

const BDPT_UNBOUNDED_MEDIUM_DISTANCE: f32 = 3.402823e38;
struct BdptMediumLayer {
  matId: u32,
  boundaryKind: u32,
  boundaryIndex: u32,
  ior: f32,
  sigmaA: vec3f,
  sigmaT: vec3f,
  sigmaS: vec3f,
  g: f32,
  remainingDistance: f32,
}

fn bdptMaterialSigmaA(
  matId: u32,
  mat: DecodedMaterial,
  heroLambda: f32,
) -> vec3f {
  var sigmaA = select(vec3f(0.0), max(mat.sigmaA, vec3f(0.0)), mat.hasSigmaA);
  if (mat.hasSpectralAttenuation && mat.spectralSampleCount > 0u) {
    if (params.spectralEnabled != 0u) {
      let mu = sampleMaterialSpectralMu(matId, heroLambdaTo01(heroLambda));
      sigmaA = vec3f(max(mu, 0.0));
    } else {
      let muR = sampleMaterialSpectralMu(matId, 0.15);
      let muG = sampleMaterialSpectralMu(matId, 0.50);
      let muB = sampleMaterialSpectralMu(matId, 0.85);
      sigmaA = max(vec3f(muR, muG, muB), vec3f(0.0));
    }
  }
  if (!mat.hasSpectralAttenuation && params.spectralEnabled != 0u) {
    sigmaA = vec3f(spectralRgbFactorAtHero(sigmaA, heroLambda));
  }
  return sigmaA;
}

fn bdptMaterialSigmaS(mat: DecodedMaterial, heroLambda: f32) -> vec3f {
  // Packed RGB already contains the scalar fallback; RGB overrides it exactly.
  let sigmaS = mat.scatteringRgb;
  return select(sigmaS, vec3f(spectralRgbFactorAtHero(sigmaS, heroLambda)), params.spectralEnabled != 0u);
}

fn bdptMaterialSigmaT(
  matId: u32,
  mat: DecodedMaterial,
  heroLambda: f32,
) -> vec3f {
  return max(
    bdptMaterialSigmaA(matId, mat, heroLambda) + bdptMaterialSigmaS(mat, heroLambda),
    vec3f(0.0),
  );
}

fn bdptHeroSigmaT(sigmaT: vec3f) -> f32 {
  return select(
    max(sigmaT.x, max(sigmaT.y, sigmaT.z)),
    sigmaT.x,
    params.spectralEnabled != 0u,
  );
}

fn bdptSegmentDistanceDensity(
  mediumMatId: u32,
  distance: f32,
  remainingDistance: f32,
  destinationIsMedium: bool,
  heroLambda: f32,
) -> f32 {
  if (mediumMatId == BDPT_NO_MEDIUM) { return 1.0; }
  let mat = decodeMaterial(mediumMatId);
  let sigmaT = bdptMaterialSigmaT(mediumMatId, mat, heroLambda);
  let heroSigmaT = bdptHeroSigmaT(sigmaT);
  if (heroSigmaT <= 0.0) { return select(1.0, 0.0, destinationIsMedium); }

  let effectiveDistance = min(
    max(distance, 0.0), max(remainingDistance, 0.0),
  );
  let survival = exp(-heroSigmaT * effectiveDistance);
  return select(survival, heroSigmaT * survival, destinationIsMedium);
}
fn bdptMediumLayer(
  matId: u32,
  mat: DecodedMaterial,
  heroLambda: f32,
  boundary: vec2u,
) -> BdptMediumLayer {
  let sigmaS = bdptMaterialSigmaS(mat, heroLambda);
  let sigmaT = bdptMaterialSigmaT(matId, mat, heroLambda);
  return BdptMediumLayer(
    matId,
    boundary.x,
    boundary.y,
    max(mat.ior, 1e-4),
    bdptMaterialSigmaA(matId, mat, heroLambda),
    sigmaT,
    sigmaS,
    clamp(mat.scatteringAnisotropy, -0.999999, 0.999999),
    materialAttenuationDistance(INFINITY, mat),
  );
}

fn bdptMediumLayerMatchesBoundary(
  layer: BdptMediumLayer,
  matId: u32,
  boundary: vec2u,
) -> bool {
  return layer.matId == matId &&
    mediumBoundaryMatches(layer.boundaryKind, layer.boundaryIndex, boundary);
}



struct BdptEndpointMedium {
  matId: u32,
  remainingDistance: f32,
}

fn bdptNoEndpointMedium() -> BdptEndpointMedium {
  return BdptEndpointMedium(
    BDPT_NO_MEDIUM, BDPT_UNBOUNDED_MEDIUM_DISTANCE,
  );
}

fn bdptSelectEndpointMedium(
  isMedium: bool,
  normal: vec3f,
  direction: vec3f,
  mediumMatId: u32,
  mediumRemainingDistance: f32,
  incidentMediumMatId: u32,
  incidentMediumRemainingDistance: f32,
  transmittedMediumMatId: u32,
  transmittedMediumRemainingDistance: f32,
) -> BdptEndpointMedium {
  if (isMedium) {
    return BdptEndpointMedium(mediumMatId, mediumRemainingDistance);
  }
  if (dot(normal, direction) >= 0.0) {
    return BdptEndpointMedium(
      incidentMediumMatId, incidentMediumRemainingDistance,
    );
  }
  return BdptEndpointMedium(
    transmittedMediumMatId, transmittedMediumRemainingDistance,
  );
}

fn bdptSharedEdgeMedium(
  a: BdptEndpointMedium,
  b: BdptEndpointMedium,
) -> BdptEndpointMedium {
  if (a.matId != b.matId) {
    return BdptEndpointMedium(BDPT_NO_MEDIUM, -1.0);
  }
  return BdptEndpointMedium(
    a.matId, min(a.remainingDistance, b.remainingDistance),
  );
}

fn bdptEndpointEdgeDistanceDensity(
  a: BdptEndpointMedium,
  b: BdptEndpointMedium,
  distance: f32,
  destinationIsMedium: bool,
  heroLambda: f32,
) -> f32 {
  let edgeMedium = bdptSharedEdgeMedium(a, b);
  if (edgeMedium.remainingDistance < 0.0) { return 0.0; }
  return bdptSegmentDistanceDensity(
    edgeMedium.matId, distance, edgeMedium.remainingDistance,
    destinationIsMedium, heroLambda,
  );
}

// ── Eye-subpath scratch stack (D2) ──────────────────────────────────────────
// The eye walk and every connection strategy execute sequentially inside ONE
// path-trace invocation. Its bounded (max 8) vertex stack therefore belongs in
// invocation-private memory, not a viewport-sized storage buffer. The former
// width×height×depth×64-B allocation reached ~759 MiB at 1080p/6 bounces and
// silently disabled explicitly requested BDPT. This fixed private array retains
// the exact fields and strategies without any resolution-dependent allocation.
// Medium vertices carry nrm=0 because volume measure has no surface cosine.
struct BdptEyeVtx {
  pos: vec3f,
  nrm: vec3f,
  pdfFwd: f32,
  pdfRev: f32,
  spec: bool,
  mediumRemainingDistance: f32,
  incidentMediumMatId: u32,
  incidentMediumRemainingDistance: f32,
  transmittedMediumMatId: u32,
  transmittedMediumRemainingDistance: f32,
  medium: bool,
  mediumG: f32,
  mediumMatId: u32,
}

const BDPT_MAX_EYE_DEPTH: u32 = 8u;
var<private> bdptEyeStackPrivate: array<BdptEyeVtx, 8>;

fn bdptEyeStackStore(
  d: u32,
  pos: vec3f,
  nrm: vec3f,
  pdfFwd: f32,
  pdfRev: f32,
  spec: bool,
  medium: bool,
  mediumG: f32,
  mediumMatId: u32,
  mediumRemainingDistance: f32,
  incidentMediumMatId: u32,
  incidentMediumRemainingDistance: f32,
  transmittedMediumMatId: u32,
  transmittedMediumRemainingDistance: f32,
) {
  if (d >= min(params.bdptMaxEyeDepth, BDPT_MAX_EYE_DEPTH)) { return; }
  var v: BdptEyeVtx;
  v.pos = pos;
  v.nrm = nrm;
  v.pdfFwd = max(pdfFwd, 0.0);
  v.pdfRev = max(pdfRev, 0.0);
  v.spec = spec;
  v.medium = medium;
  v.mediumG = mediumG;
  v.mediumMatId = mediumMatId;
  v.mediumRemainingDistance = mediumRemainingDistance;
  v.incidentMediumMatId = incidentMediumMatId;
  v.incidentMediumRemainingDistance = incidentMediumRemainingDistance;
  v.transmittedMediumMatId = transmittedMediumMatId;
  v.transmittedMediumRemainingDistance = transmittedMediumRemainingDistance;
  bdptEyeStackPrivate[d] = v;
}

// Patch only the pdfFwd of an already-stored slot, preserving its position and
// its specular sign convention (specular slots keep their negative sentinel).
fn bdptEyeStackSetFwd(d: u32, pdfFwd: f32) {
  if (d >= min(params.bdptMaxEyeDepth, BDPT_MAX_EYE_DEPTH)) { return; }
  bdptEyeStackPrivate[d].pdfFwd = max(pdfFwd, 0.0);
}

// Record the event sampled at this eye vertex after the connection work.
fn bdptEyeStackSetSpec(d: u32, spec: bool) {
  if (d >= min(params.bdptMaxEyeDepth, BDPT_MAX_EYE_DEPTH)) { return; }
  bdptEyeStackPrivate[d].spec = spec;
}

fn bdptEyeStackLoad(d: u32) -> BdptEyeVtx {
  return bdptEyeStackPrivate[min(d, BDPT_MAX_EYE_DEPTH - 1u)];
}

// SPPM stores light prefixes that contain one or more specular events before
// their finite diffuse receiver. A BDPT connection to a later non-delta light
// vertex would otherwise estimate that same E-…-D-S*-L path family. Vertex 0 is
// deliberately excluded: point/spot/directional source endpoints are delta
// launch distributions, not an interior specular scattering event.
fn bdptLightPrefixContainsInteriorDelta(connectionIndex: u32) -> bool {
  for (var j = 1u; j < connectionIndex; j = j + 1u) {
    let row0 = bdptLightPath[bdptLightPathIndex(i32(j), 0u)];
    if (row0.w == BDPT_KIND_DELTA) { return true; }
  }
  return false;
}

// MNEE owns exactly L-S[1..configuredMax]-D: every light-side interior
// vertex before the finite connection vertex must be delta, and the chain must
// fit the configured bounded solver. A merely mixed finite/delta prefix is not
// a manifold-NEE path and must remain available to BDPT.
fn bdptLightPrefixIsMneeOwned(connectionIndex: u32) -> bool {
  if (connectionIndex <= 1u) { return false; }
  let chainLength = connectionIndex - 1u;
  if (chainLength > min(params.mneeMaxChainLength, 8u)) { return false; }
  for (var j = 1u; j < connectionIndex; j = j + 1u) {
    let row0 = bdptLightPath[bdptLightPathIndex(i32(j), 0u)];
    if (row0.w != BDPT_KIND_DELTA) { return false; }
  }
  return true;
}

fn bdptGeometricTerm(
  posX: vec3f,
  nX: vec3f,
  mediumX: bool,
  posY: vec3f,
  nY: vec3f,
  mediumY: bool,
) -> f32 {
  let d = posY - posX;
  let dist2 = dot(d, d);
  if (dist2 <= 1e-12) {
    return 0.0;
  }
  let w = d * inverseSqrt(dist2);
  let cosX = select(abs(dot(nX, w)), 1.0, mediumX);
  let cosY = select(abs(dot(nY, -w)), 1.0, mediumY);
  return (cosX * cosY) / dist2;
}

struct BdptLogDensity {
  value: f32,
  valid: bool,
}

fn bdptFinitePositive(value: f32) -> bool {
  return value > 0.0 && value - value == 0.0;
}

// Log-domain PBRT Vertex::ConvertDensity. Scaling the edge before dot/square
// avoids overflow on long finite edges; adding/subtracting logarithms avoids
// underflow for near-grazing destination cosines.
fn bdptLogDensitySAtoArea(
  pdfSA: f32,
  fromPos: vec3f,
  destPos: vec3f,
  destNorm: vec3f,
  destIsMedium: bool,
) -> BdptLogDensity {
  if (!bdptFinitePositive(pdfSA)) {
    return BdptLogDensity(0.0, false);
  }
  let d = destPos - fromPos;
  let edgeScale = max(abs(d.x), max(abs(d.y), abs(d.z)));
  if (edgeScale == 0.0) {
    return BdptLogDensity(0.0, false);
  }
  if (!bdptFinitePositive(edgeScale)) {
    return BdptLogDensity(0.0, false);
  }
  let scaledEdge = d / edgeScale;
  let scaledDist2 = dot(scaledEdge, scaledEdge);
  let edgeDir = scaledEdge * inverseSqrt(scaledDist2);
  let cosDest = select(abs(dot(destNorm, edgeDir)), 1.0, destIsMedium);
  if (!bdptFinitePositive(cosDest)) {
    return BdptLogDensity(0.0, false);
  }
  let logDist2 = 2.0 * log(edgeScale) + log(scaledDist2);
  return BdptLogDensity(log(pdfSA) + log(cosDest) - logDist2, true);
}

fn bdptInfiniteLaunchLogArea(
  launchPdf: f32,
  receiverNormal: vec3f,
  receiverToSource: vec3f,
  receiverIsMedium: bool,
) -> BdptLogDensity {
  if (!bdptFinitePositive(launchPdf)) {
    return BdptLogDensity(0.0, false);
  }
  let projection = select(
    abs(dot(safe_normalize(receiverNormal), safe_normalize(receiverToSource))),
    1.0,
    receiverIsMedium,
  );
  if (!bdptFinitePositive(projection)) {
    return BdptLogDensity(0.0, false);
  }
  // Infinite roots launch parallel rays from a scene-bounding disk. Disk area
  // maps to receiver area by orthogonal projection; there is no 1/r² factor.
  return BdptLogDensity(log(launchPdf) + log(projection), true);
}

fn bdptInfiniteRootLaunchPdf(directionPdf: f32) -> f32 {
  let emitterCount = bdptEmitterCount();
  if (emitterCount == 0u || !bdptFinitePositive(directionPdf)) { return 0.0; }
  let radius = max(params.sceneRadius, 1e-3);
  return directionPdf /
    (f32(emitterCount) * PI * radius * radius);
}

// Complete infinite-emitter strategy family for a fixed geometric path:
// p0 = eye escape, p1 = distant NEE, p2+ = explicit infinite-root BDPT.
// The selected p0/p1 draw is normalized to log-density zero and every other
// technique is reconstructed by exact area-measure ratios.
fn bdptInfiniteEyeFamilyWeight(
  selectedS: u32,
  pureEyeImplemented: bool,
  pureEyeSampledDelta: bool,
  pureEyePdf: f32,
  neePdf: f32,
  launchPdf: f32,
  receiverToSource: vec3f,
  currentPosition: vec3f,
  currentNormal: vec3f,
  currentIsMedium: bool,
  currentIncomingEyePdf: f32,
  currentSwappedPdf: f32,
  currentConnectionDelta: bool,
  eyeDepth: u32,
) -> f32 {
  if (selectedS > 1u || eyeDepth >= BDPT_MAX_EYE_DEPTH) { return 0.0; }
  if (selectedS == 0u && pureEyeSampledDelta) { return 1.0; }
  if (!bdptFinitePositive(neePdf)) {
    return select(0.0, 1.0, selectedS == 0u);
  }

  var logPdfs: array<f32, BDPT_MAX_INFINITE_STRATEGIES>;
  var validPdfs: array<bool, BDPT_MAX_INFINITE_STRATEGIES>;
  for (var strategy = 0u; strategy < BDPT_MAX_INFINITE_STRATEGIES;
       strategy = strategy + 1u) {
    logPdfs[strategy] = 0.0;
    validPdfs[strategy] = false;
  }
  validPdfs[1u] = true;
  if (pureEyeImplemented && bdptFinitePositive(pureEyePdf)) {
    logPdfs[0u] = log(pureEyePdf) - log(neePdf);
    validPdfs[0u] = true;
  }

  let pathVertexCount = eyeDepth + 3u;
  if (
    eyeDepth >= 1u && params.bdptMaxLightBounces >= 2u &&
    bdptFinitePositive(launchPdf)
  ) {
    let previous = bdptEyeStackLoad(eyeDepth - 1u);
    let transitionBlocked = currentConnectionDelta || previous.spec;
    let launchArea = bdptInfiniteLaunchLogArea(
      launchPdf, currentNormal, receiverToSource, currentIsMedium,
    );
    let incomingEyeArea = bdptLogDensitySAtoArea(
      currentIncomingEyePdf,
      previous.pos,
      currentPosition,
      currentNormal,
      currentIsMedium,
    );
    if (!transitionBlocked && launchArea.valid && incomingEyeArea.valid) {
      var logRatio = launchArea.value - log(neePdf) - incomingEyeArea.value;
      logPdfs[2u] = logRatio;
      validPdfs[2u] = true;

      for (var strategy = 3u; strategy < BDPT_MAX_INFINITE_STRATEGIES;
           strategy = strategy + 1u) {
        if (
          strategy > params.bdptMaxLightBounces ||
          strategy > pathVertexCount - 2u
        ) { break; }
        // Addition precedes subtraction deliberately: every admitted strategy
        // satisfies strategy <= eyeDepth + 1, so this cannot wrap as u32.
        let destinationDepth = eyeDepth + 2u - strategy;
        if (destinationDepth < 1u) { break; }
        let destination = bdptEyeStackLoad(destinationDepth);
        let predecessor = bdptEyeStackLoad(destinationDepth - 1u);
        if (destination.spec || predecessor.spec) { break; }
        var lightwardPosition = currentPosition;
        var forwardPdf = currentSwappedPdf;
        if (strategy > 3u) {
          lightwardPosition = bdptEyeStackLoad(destinationDepth + 1u).pos;
          forwardPdf = destination.pdfFwd;
        }
        let forwardArea = bdptLogDensitySAtoArea(
          forwardPdf,
          lightwardPosition,
          destination.pos,
          destination.nrm,
          destination.medium,
        );
        let reverseArea = bdptLogDensitySAtoArea(
          destination.pdfRev,
          predecessor.pos,
          destination.pos,
          destination.nrm,
          destination.medium,
        );
        if (!forwardArea.valid || !reverseArea.valid) { break; }
        logRatio = logRatio + forwardArea.value - reverseArea.value;
        logPdfs[strategy] = logRatio;
        validPdfs[strategy] = true;
      }
    }
  }

  if (!validPdfs[selectedS]) { return 0.0; }
  var maxPowerLog = 2.0 * logPdfs[selectedS];
  for (var strategy = 0u; strategy < BDPT_MAX_INFINITE_STRATEGIES;
       strategy = strategy + 1u) {
    if (validPdfs[strategy]) {
      maxPowerLog = max(maxPowerLog, 2.0 * logPdfs[strategy]);
    }
  }
  var denominator = 0.0;
  for (var strategy = 0u; strategy < BDPT_MAX_INFINITE_STRATEGIES;
       strategy = strategy + 1u) {
    if (validPdfs[strategy]) {
      denominator = denominator +
        exp(2.0 * logPdfs[strategy] - maxPowerLog);
    }
  }
  if (!(denominator > 0.0)) { return 0.0; }
  return exp(2.0 * logPdfs[selectedS] - maxPowerLog) / denominator;
}

// Lambertian outgoing SA density at a light-subpath vertex along dir: |cosθ|/π.
fn bdptLambertDirPdf(n: vec3f, dir: vec3f) -> f32 {
  return abs(dot(n, normalize(dir))) * INV_PI;
}

// Exact finite-direction density for a transmissive dielectric connection
// endpoint. The shared sampler chooses the base dielectric with probability
// 1/(1+clearcoat+sheen), then chooses diffuse reflection or one GGX VNDF
// dielectric event. Smooth reflection/refraction are delta events and therefore
// have zero finite-direction density; rough reflection and refraction are here.
fn bdptTransmissiveConnectionPdf(
  roughness: f32,
  transmission: f32,
  etaTOverI: f32,
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
  var evalNormal = normal;
  var evalClearcoatNormal = clearcoatNormal;
  var evalEta = max(etaTOverI, 1e-4);
  if (dot(evalNormal, wo) < 0.0) {
    evalNormal = -evalNormal;
    evalClearcoatNormal = -evalClearcoatNormal;
    evalEta = 1.0 / evalEta;
  }
  let nDotV = dot(evalNormal, wo);
  let nDotL = dot(evalNormal, wi);
  if (nDotV <= 1e-5) { return 0.0; }
  let eventProbabilities = bsdfDielectricFiniteEventProbabilities(
    roughness, transmission, evalEta, evalNormal, wo, wi,
    iridescence, iridescenceIor,
    iridescenceThicknessMin, iridescenceThicknessMax,
    specularColor, specularIntensity,
  );
  let lobeWeightSum = brdfExtensionLobeWeightSum(clearcoat, sheen);
  if (nDotL < -1e-5) {
    return (
      eventProbabilities.z *
      bsdfRoughTransmissionPdf(
        roughness, evalEta, evalNormal, wo, wi,
        anisotropy, anisotropyRotation,
      )
    ) / lobeWeightSum;
  }
  if (nDotL <= 1e-5) { return 0.0; }

  let pdfSpec = bsdfSpecularReflectionPdf(
    roughness, evalNormal, wo, wi, anisotropy, anisotropyRotation,
  );
  let clearcoatDensity =
    max(clearcoat, 0.0) * clearcoatPdf(
      clearcoat, clearcoatRoughness, evalClearcoatNormal, wo, wi,
    );
  let sheenDensity =
    max(sheen, 0.0) * charlieSheenPdf(
      sheen, sheenRoughness, evalNormal, wo, wi,
    );
  return (
    eventProbabilities.x * pdfSpec + eventProbabilities.y * nDotL * INV_PI +
      clearcoatDensity + sheenDensity
  ) / lobeWeightSum;
}

// Marginal directional density of the complete sampled BSDF mixture. BDPT
// strategy PDFs cannot condition on the auxiliary lobe id: connection vertices
// do not carry that random choice, so every reverse recurrence uses this same
// marginal density. The transmissive branch performs side reorientation before
// evaluating rough R/T plus finite reflection lobes.
fn bdptMarginalSurfacePdf(
  baseColor: vec3f,
  roughness: f32,
  metallic: f32,
  transmission: f32,
  etaTOverI: f32,
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
  if (transmission > 0.0 && metallic == 0.0) {
    return bdptTransmissiveConnectionPdf(
      roughness, transmission, etaTOverI, normal, clearcoatNormal, wo, wi,
      clearcoat, clearcoatRoughness, sheen, sheenRoughness,
      iridescence, iridescenceIor,
      iridescenceThicknessMin, iridescenceThicknessMax,
      specularColor, specularIntensity,
      anisotropy, anisotropyRotation,
    );
  }
  return brdfDirectionalPdfFullSampledWithClearcoatNormal(
    baseColor, roughness, metallic, transmission, etaTOverI,
    normal, clearcoatNormal, wo, wi,
    clearcoat, clearcoatRoughness, sheen, sheenRoughness,
    iridescence, iridescenceIor,
    iridescenceThicknessMin, iridescenceThicknessMax,
    specularColor, specularIntensity, anisotropy, anisotropyRotation,
  );
}

// Merged-path vertex assembled on the fly from the light chain (texture) + the
// eye stack (scratch buffer) + the connection-induced straddle overrides.
struct BdptMergedVtx {
  pos: vec3f,
  nrm: vec3f,
  pdfFwd: f32,
  pdfRev: f32,
  spec: bool,
  medium: bool,
  mediumG: f32,
  mediumMatId: u32,
}

// Address one merged vertex by index i in [0, n-1]. The connection-induced
// straddle overrides are applied here so the sweep reads a coherent vertex set.
//   light side  : i <= c
//   eye side     : i in [c+1, c+1+e]  → eye depth = e - (i - (c+1))
//   camera       : i == n-1
fn bdptMergedVertex(
  i: u32,
  c: u32,        // light connection vertex index (lightChain length - 1)
  e: u32,        // eye connection vertex depth (eyeChain length - 1)
  n: u32,        // merged length
  // connection geometry / overrides
  fwdEe: f32,        // merged pdfFwd(E_e)        (light Lambert toward E_e)
  fwdEeMinus: f32,   // merged pdfFwd(E_{e-1})    (eye BSDF, wo=connDir)
  revLc: f32,        // merged pdfRev(L_c)        (eye BSDF, wi=connDir)
  revLcMinus: f32,   // merged pdfRev(L_{c-1})    (light Lambert toward L_{c-1})
  camPos: vec3f,
  camNrm: vec3f,
) -> BdptMergedVtx {
  var v: BdptMergedVtx;
  if (i <= c) {
    // Light side: read scratch-buffer column i, rows 0/1/2.
    let l0 = bdptLightPath[bdptLightPathIndex(i32(i), 0u)];
    let l1 = bdptLightPath[bdptLightPathIndex(i32(i), 1u)];
    let l2 = bdptLightPath[bdptLightPathIndex(i32(i), 2u)];
    let l3 = bdptLightPath[bdptLightPathIndex(i32(i), 3u)];
    let l5 = bdptLightPath[bdptLightPathIndex(i32(i), 5u)];
    v.pos = l0.xyz;
    v.nrm = l1.xyz;
    v.pdfFwd = l1.w;          // stored SA pdfFwd (NO baked-in G; emitter = area endpoint)
    v.pdfRev = l2.w;          // stored SA pdfRev (Lambertian construction)
    v.medium = l3.w == BDPT_LV_MEDIUM_MATID;
    v.mediumG = select(0.0, l5.y, v.medium);
    v.mediumMatId = bitcast<u32>(l5.w);
    // Point/spot position is singular, but its emitted direction has a finite
    // density. Do not classify the root as an interior delta-scatter barrier:
    // c=0 endpoint connections and c=1 extended-light strategies must compete.
    v.spec = l0.w == BDPT_KIND_DELTA;
    if (i == c && l3.w >= 0.0) { v.spec = fwdEe <= 0.0 || revLc <= 0.0; }
    if (i == c) { v.pdfRev = revLc; }
    else if (c >= 1u && i == c - 1u) { v.pdfRev = revLcMinus; }
    return v;
  }
  if (i == n - 1u) {
    v.pos = camPos;
    v.nrm = camNrm;
    v.pdfFwd = 1.0;
    v.pdfRev = 1.0;
    v.spec = false;
    v.medium = false;
    v.mediumG = 0.0;
    v.mediumMatId = BDPT_NO_MEDIUM;
    return v;
  }
  // Eye side: merged i in [c+1, c+1+e] → eye scratch index d = e - (i - (c+1)).
  let off = i - (c + 1u);
  let d = e - off;
  let es = bdptEyeStackLoad(d);
  v.pos = es.pos;
  v.nrm = es.nrm;
  v.pdfFwd = es.pdfFwd;       // merged forward (swapped-BSDF reverse density)
  v.pdfRev = es.pdfRev;       // merged reverse (scatter pdf that produced E_d)
  v.spec = es.spec;
  if (off == 0u) { v.pdfFwd = fwdEe; }            // E_e   override
  v.medium = es.medium;
  v.mediumG = es.mediumG;
  v.mediumMatId = es.mediumMatId;
  if (off == 0u) { v.spec = fwdEe <= 0.0 || revLc <= 0.0; }
  else if (off == 1u) { v.pdfFwd = fwdEeMinus; }  // E_{e-1} override
  return v;
}

// Log area-measure forward density of merged vertex i (edge v_{i-1}→v_i).
fn bdptFwdLogArea(i: u32, c: u32, e: u32, n: u32,
  fwdEe: f32, fwdEeMinus: f32, revLc: f32, revLcMinus: f32, camPos: vec3f, camNrm: vec3f) -> BdptLogDensity {
  let v = bdptMergedVertex(i, c, e, n, fwdEe, fwdEeMinus, revLc, revLcMinus, camPos, camNrm);
  if (i == 0u) {
    if (!bdptFinitePositive(v.pdfFwd)) {
      return BdptLogDensity(0.0, false);
    }
    return BdptLogDensity(log(v.pdfFwd), true);
  }
  let prev = bdptMergedVertex(i - 1u, c, e, n, fwdEe, fwdEeMinus, revLc, revLcMinus, camPos, camNrm);
  return bdptLogDensitySAtoArea(v.pdfFwd, prev.pos, v.pos, v.nrm, v.medium);
}

// Log area-measure reverse density of merged vertex i (edge v_{i+1}→v_i).
fn bdptRevLogArea(i: u32, c: u32, e: u32, n: u32,
  fwdEe: f32, fwdEeMinus: f32, revLc: f32, revLcMinus: f32, camPos: vec3f, camNrm: vec3f) -> BdptLogDensity {
  let v = bdptMergedVertex(i, c, e, n, fwdEe, fwdEeMinus, revLc, revLcMinus, camPos, camNrm);
  if (i == n - 1u) {
    if (!bdptFinitePositive(v.pdfRev)) {
      return BdptLogDensity(0.0, false);
    }
    return BdptLogDensity(log(v.pdfRev), true);
  }
  let next = bdptMergedVertex(i + 1u, c, e, n, fwdEe, fwdEeMinus, revLc, revLcMinus, camPos, camNrm);
  return bdptLogDensitySAtoArea(v.pdfRev, next.pos, v.pos, v.nrm, v.medium);
}

// Full Veach §10.3 power-heuristic MIS weight for the selected strategy
// selectedS = c+1. Mirrors buildBDPTStrategyPDFs_full + bdptConnectionMIS_full.
fn bdptMISWeightFull(
  c: u32, e: u32, n: u32, selectedS: u32,
  infiniteRoot: bool,
  infiniteEnvironmentRoot: bool,
  infiniteSourceDirection: vec3f,
  infiniteNeePdf: f32,
  infiniteLaunchPdf: f32,
  infiniteEyeEscapePdf: f32,
  infiniteEyeEscapeDelta: bool,
  fwdEe: f32, fwdEeMinus: f32, revLc: f32, revLcMinus: f32, camPos: vec3f, camNrm: vec3f,
) -> f32 {
  if (n == 0u || selectedS >= n) { return 0.0; }
  // The power heuristic is invariant to a common density scale, so the
  // reference log-density is exactly zero. Every other technique is represented
  // only by a sum of log-ratios relative to that reference.
  var logPdfs: array<f32, BDPT_MAX_MERGED>;
  var validPdfs: array<bool, BDPT_MAX_MERGED>;
  for (var k = 0u; k < n; k = k + 1u) {
    logPdfs[k] = 0.0;
    validPdfs[k] = false;
  }
  validPdfs[selectedS] = true;

  // Left sweep: log p_{s-1} = log p_s + log pRev(s-1) - log pFwd(s-1).
  {
    var logP = 0.0;
    var s = selectedS;
    loop {
      if (s == 0u) { break; }
      let flip = bdptMergedVertex(s - 1u, c, e, n, fwdEe, fwdEeMinus, revLc, revLcMinus, camPos, camNrm);
      var neighborSpec = false;
      if (s >= 2u) {
        let nb = bdptMergedVertex(s - 2u, c, e, n, fwdEe, fwdEeMinus, revLc, revLcMinus, camPos, camNrm);
        neighborSpec = nb.spec;
      }
      if (flip.spec || neighborSpec) { break; }
      let logFwd = bdptFwdLogArea(s - 1u, c, e, n, fwdEe, fwdEeMinus, revLc, revLcMinus, camPos, camNrm);
      let logRev = bdptRevLogArea(s - 1u, c, e, n, fwdEe, fwdEeMinus, revLc, revLcMinus, camPos, camNrm);
      if (!logFwd.valid || !logRev.valid) { break; }
      logP = logP + logRev.value - logFwd.value;
      logPdfs[s - 1u] = logP;
      validPdfs[s - 1u] = true;
      s = s - 1u;
    }
  }
  // Right sweep: log p_{s+1} = log p_s + log pFwd(s) - log pRev(s).
  {
    var logP = 0.0;
    var s = selectedS;
    loop {
      if (s >= n - 1u) { break; }
      let flip = bdptMergedVertex(s, c, e, n, fwdEe, fwdEeMinus, revLc, revLcMinus, camPos, camNrm);
      let nb = bdptMergedVertex(s + 1u, c, e, n, fwdEe, fwdEeMinus, revLc, revLcMinus, camPos, camNrm);
      if (flip.spec || nb.spec) { break; }
      let logFwd = bdptFwdLogArea(s, c, e, n, fwdEe, fwdEeMinus, revLc, revLcMinus, camPos, camNrm);
      let logRev = bdptRevLogArea(s, c, e, n, fwdEe, fwdEeMinus, revLc, revLcMinus, camPos, camNrm);
      if (!logFwd.valid || !logRev.valid) { break; }
      logP = logP + logFwd.value - logRev.value;
      logPdfs[s + 1u] = logP;
      validPdfs[s + 1u] = true;
      s = s + 1u;
    }
  }

  if (infiniteRoot) {
    // The generic recurrence treats v0 as a finite point. Replace p0/p1 with
    // the true infinite-source measures relative to p2: parallel launch-disk
    // projection for p2, distant NEE for p1, and eye escape for p0.
    validPdfs[0u] = false;
    validPdfs[1u] = false;
    if (
      n > 3u && validPdfs[2u] &&
      bdptFinitePositive(infiniteNeePdf) &&
      bdptFinitePositive(infiniteLaunchPdf)
    ) {
      let firstReceiver = bdptMergedVertex(
        1u, c, e, n, fwdEe, fwdEeMinus, revLc, revLcMinus, camPos, camNrm,
      );
      let cameraward = bdptMergedVertex(
        2u, c, e, n, fwdEe, fwdEeMinus, revLc, revLcMinus, camPos, camNrm,
      );
      let launchArea = bdptInfiniteLaunchLogArea(
        infiniteLaunchPdf,
        firstReceiver.nrm,
        infiniteSourceDirection,
        firstReceiver.medium,
      );
      let eyeArea = bdptLogDensitySAtoArea(
        firstReceiver.pdfRev,
        cameraward.pos,
        firstReceiver.pos,
        firstReceiver.nrm,
        firstReceiver.medium,
      );
      if (launchArea.valid && eyeArea.valid) {
        logPdfs[1u] = logPdfs[2u] + log(infiniteNeePdf) -
          launchArea.value + eyeArea.value;
        validPdfs[1u] = true;
        if (
          infiniteEnvironmentRoot && !infiniteEyeEscapeDelta &&
          bdptFinitePositive(infiniteEyeEscapePdf)
        ) {
          logPdfs[0u] = logPdfs[1u] + log(infiniteEyeEscapePdf) -
            log(infiniteNeePdf);
          validPdfs[0u] = true;
        }
      }
    }
  }

  // Keep only the explicit strategies that this bounded kernel actually samples.
  // Infinite roots additionally admit p0 eye escape and p1 distant NEE; finite
  // roots retain only explicit p1..p(n-2) connection strategies.
  let maxLightVertices = min(params.bdptMaxLightBounces, 8u);
  let maxEyeVertices = min(params.bdptMaxEyeDepth, 8u);
  for (var k = 0u; k < n; k = k + 1u) {
    var validExplicitStrategy = k >= 1u && k <= n - 2u;
    if (infiniteRoot) {
      validExplicitStrategy =
        (k == 0u && infiniteEnvironmentRoot) ||
        k == 1u || (k >= 2u && k <= n - 2u);
    }
    if (validExplicitStrategy) {
      let lightVertices = k;
      let eyeVertices = n - k - 1u;
      validExplicitStrategy =
        (k < 2u || lightVertices <= maxLightVertices) &&
        eyeVertices <= maxEyeVertices;
    }
    if (!validExplicitStrategy) { validPdfs[k] = false; }
  }

  if (!validPdfs[selectedS]) { return 0.0; }
  // Power heuristic (β=2) evaluated with a max-shifted log-sum-exp. All
  // exponent arguments are <= 0, so neither squaring nor the denominator can
  // overflow; extremely small relative weights may harmlessly round to zero.
  var maxPowerLog = 2.0 * logPdfs[selectedS];
  for (var k = 0u; k < n; k = k + 1u) {
    if (validPdfs[k]) {
      maxPowerLog = max(maxPowerLog, 2.0 * logPdfs[k]);
    }
  }
  var denom = 0.0;
  for (var k = 0u; k < n; k = k + 1u) {
    if (validPdfs[k]) {
      denom = denom + exp(2.0 * logPdfs[k] - maxPowerLog);
    }
  }
  if (denom <= 0.0) { return 0.0; }
  let selectedNumerator =
    exp(2.0 * logPdfs[selectedS] - maxPowerLog);
  return selectedNumerator / denom;
}

// Forward scatter pdf at the eye vertex (the old hardcoded eyePdfFwd=1.0 is
// gone): the real per-vertex BSDF scatter densities now flow through the eye
// scratch stack (pdfRev) and the connection-induced overrides (revLc / fwdEeMinus
// via brdfDirectionalPdf), so no scalar eyePdfFwd argument is needed here.
fn evaluateBdptConnection(
  eyePos: vec3f,
  eyeNormal: vec3f,
  eyeClearcoatNormal: vec3f,
  eyeWo: vec3f,
  eyeThroughput: vec3f,
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
  eyeDepth: u32,
  eyeIsMedium: bool,
  eyeMediumG: f32,
  eyeMediumMatId: u32,
  lightVtxIdx: i32,
) -> vec3f {
  let lv0 = bdptLightPath[bdptLightPathIndex(lightVtxIdx, 0u)];
  let lv1 = bdptLightPath[bdptLightPathIndex(lightVtxIdx, 1u)];
  let lv2 = bdptLightPath[bdptLightPathIndex(lightVtxIdx, 2u)];
  if (lv0.w == BDPT_KIND_INVALID) {
    return vec3f(0.0);
  }
  let c = u32(lightVtxIdx);
  let activeCausticMode = causticMode();
  let sppmOwnsLightPrefix = activeCausticMode == 2u &&
    bdptLightPrefixContainsInteriorDelta(c);
  let mneeOwnsLightPrefix = activeCausticMode == 1u &&
    bdptLightPrefixIsMneeOwned(c);
  if (sppmOwnsLightPrefix || mneeOwnsLightPrefix) {
    return vec3f(0.0);
  }
  let lightPos = lv0.xyz;
  let lightNormal = lv1.xyz;
  let lightThroughput = lv2.xyz;
  let toLight = lightPos - eyePos;
  let dist = length(toLight);
  if (dist < 1e-4) {
    return vec3f(0.0);
  }
  let connDir = toLight / dist;                 // E_e → L_c
  let lv3 = bdptLightPath[bdptLightPathIndex(lightVtxIdx, 3u)];
  let lv4 = bdptLightPath[bdptLightPathIndex(lightVtxIdx, 4u)];
  let lv5 = bdptLightPath[bdptLightPathIndex(lightVtxIdx, 5u)];
  let lv6 = bdptLightPath[bdptLightPathIndex(lightVtxIdx, 6u)];
  let lvMatId = lv3.w;
  // Infinite endpoints are launch constructs on a scene-bounding disk, not
  // finite points that an eye vertex may connect to. Ordinary directional/env
  // direct-light and escape estimators own c=0; BDPT starts at the first real
  // scattering vertex (c>=1).
  if (lightVtxIdx == 0 && (
      lvMatId == BDPT_LV_DIRECTIONAL_EMITTER_MATID ||
      lvMatId == BDPT_LV_ENVIRONMENT_EMITTER_MATID
  )) {
    return vec3f(0.0);
  }
  let isPointEndpoint = lightVtxIdx == 0 && lvMatId == BDPT_LV_POINT_EMITTER_MATID;
  let lightIsMedium = lvMatId == BDPT_LV_MEDIUM_MATID;
  let lightMediumG = select(0.0, lv5.y, lightIsMedium);
  let lightMediumMatId = bitcast<u32>(lv5.w);
  let lightMediumRemainingDistance = lv6.y;
  let lightIncidentMediumMatId = bitcast<u32>(lv6.x);
  let lightIncidentMediumRemainingDistance = lv6.y;
  let lightTransmittedMediumMatId = bitcast<u32>(lv6.z);
  let lightTransmittedMediumRemainingDistance = lv6.w;
  let eyeRecord = bdptEyeStackLoad(eyeDepth);
  let isSpotEndpoint = lightVtxIdx == 0 && lvMatId == BDPT_LV_SPOT_EMITTER_MATID;
  if (!eyeIsMedium && !bsdfHasFiniteConnectionSupport(
    roughness, metallic, transmission, clearcoat, sheen,
  )) {
    return vec3f(0.0);
  }
  let cosEye = select(abs(dot(eyeNormal, connDir)), 1.0, eyeIsMedium);
  if (cosEye <= 0.0) {
    return vec3f(0.0);
  }
  var gTerm = bdptGeometricTerm(
    eyePos, eyeNormal, eyeIsMedium,
    lightPos, lightNormal, lightIsMedium,
  );
  var connectionTransmittance = vec3f(1.0);
  let eyeConnectionMedium = bdptSelectEndpointMedium(
    eyeIsMedium, eyeNormal, connDir,
    eyeMediumMatId, eyeRecord.mediumRemainingDistance,
    eyeRecord.incidentMediumMatId, eyeRecord.incidentMediumRemainingDistance,
    eyeRecord.transmittedMediumMatId,
    eyeRecord.transmittedMediumRemainingDistance,
  );
  let lightConnectionMedium = bdptSelectEndpointMedium(
    lightIsMedium, lightNormal, -connDir,
    lightMediumMatId, lightMediumRemainingDistance,
    lightIncidentMediumMatId, lightIncidentMediumRemainingDistance,
    lightTransmittedMediumMatId,
    lightTransmittedMediumRemainingDistance,
  );
  let connectionMedium = bdptSharedEdgeMedium(
    eyeConnectionMedium, lightConnectionMedium,
  );
  if (connectionMedium.remainingDistance < 0.0) {
    return vec3f(0.0);
  }
  if (connectionMedium.matId != BDPT_NO_MEDIUM) {
    let connectionMaterial = decodeMaterial(connectionMedium.matId);
    let connectionSigmaT = bdptMaterialSigmaT(
      connectionMedium.matId, connectionMaterial, bdptInvocationHeroLambdaNm,
    );
    let connectionDistance = min(
      dist, max(connectionMedium.remainingDistance, 0.0),
    );
    connectionTransmittance =
      exp(-connectionSigmaT * connectionDistance);
  }
  if (isPointEndpoint) {
    gTerm = cosEye * pointSpotDistanceAttenuation(dist, lv4.y, lv4.z);
  } else if (isSpotEndpoint) {
    let cosOuter = lv4.w;
    let cosInner = lv3.x;
    let cosTheta = dot(lightNormal, -connDir);
    if (cosTheta <= cosOuter) {
      return vec3f(0.0);
    }
    let softness = smoothstep(cosOuter, max(cosInner, cosOuter + 1e-6), cosTheta);
    gTerm = cosEye * softness * pointSpotDistanceAttenuation(
      dist, lv4.y, lv4.z,
    );
  }
  if (gTerm <= 0.0) {
    return vec3f(0.0);
  }
  let lightEmitterCastShadowDisabled = lightVtxIdx == 0 && lvMatId < 0.0 && lv4.x > 0.5;
  let shadowRay = Ray(eyePos + connDir * 1e-3, connDir);
  if (!lightEmitterCastShadowDisabled && traceAny(shadowRay, 1e-4, max(dist - 2e-3, 1e-3))) {
    return vec3f(0.0);
  }
  var eyeBrdf = vec3f(
    hgPhase(dot(-eyeWo, connDir), eyeMediumG),
  );
  if (!eyeIsMedium) {
    eyeBrdf = evaluateFiniteBsdfFullWithClearcoatNormal(
      baseColor, roughness, metallic, transmission, etaTOverI,
      eyeNormal, eyeClearcoatNormal, eyeWo, connDir,
      clearcoat, clearcoatRoughness, sheen, sheenRoughness, sheenColor,
      iridescence, iridescenceIor, iridescenceThicknessMin,
      iridescenceThicknessMax, specularColor, specularIntensity,
      anisotropy, anisotropyRotation, false,
    );
  }
  // bdptGeometricTerm already contributes the receiver cosine for this edge.
  // Keep the BSDF value itself here so the connection does not double-count
  // cos(theta) at the eye endpoint.
  let eyeBsdfCosTheta = eyeBrdf;
  let cosLight = select(max(dot(lightNormal, -connDir), 0.0), 1.0, lightIsMedium);
  if (lvMatId < 0.0 && !isPointEndpoint && !isSpotEndpoint && cosLight <= 0.0) {
    return vec3f(0.0);
  }
  // A9 — REAL light-vertex BSDF at L_c (row 3: matId + wo-toward-prev). For a
  // surface vertex (matId >= 0) evaluate the actual BSDF scattering the incoming
  // light direction (toward L_{c-1}, = lvWoPrev) to the connection direction (L_c →
  // E_e, = -connDir); for the legacy pseudo-emitter vertex (matId == -1) keep
  // the diffuse emission/Lambertian profile cosθ/π. Finite area emitters
  // (matId == -2) already carry Le/(pdfPick·pdfArea), so their endpoint factor
  // is 1 and the geometry term owns the emitter cosine. This makes a glossy/metallic light-path
  // vertex's connection consistent with the glossy light-subpath BUILD (else the
  // BSDF mismatch between build and connect biases the estimate).
  let lvWoPrev = lv3.xyz;
  var lightBsdfCosTheta = vec3f(1.0);
  if (lvMatId == -1.0) {
    lightBsdfCosTheta = vec3f(cosLight / PI);
  }
  if (lightIsMedium) {
    lightBsdfCosTheta =
      vec3f(hgPhase(dot(lvWoPrev, connDir), lightMediumG));
  }
  if (lvMatId >= 0.0) {
    let lvMat = bdptSampleMaterialAtPayload(u32(lvMatId), lv4, lightNormal, lvWoPrev, bdptInvocationHeroLambdaNm);
    // The TMM stack is sampled as a discrete R/T/A event by the light random
    // walk. It has no finite connection density at this vertex.
    if (lvMat.thinFilmEnabled) {
      return vec3f(0.0);
    }
    if (!bsdfHasFiniteConnectionSupport(
      lvMat.roughness, lvMat.metallic, lvMat.transmission,
      lvMat.clearcoat, lvMat.sheen,
    )) {
      return vec3f(0.0);
    }
    let lvBrdf = evaluateFiniteBsdfFullWithClearcoatNormal(
      lvMat.baseColor, lvMat.roughness, lvMat.metallic,
      lvMat.transmission, max(lv5.x, 1e-4),
      lightNormal, lvMat.clearcoatNormal, lvWoPrev, -connDir,
      lvMat.clearcoat, lvMat.clearcoatRoughness, lvMat.sheen, lvMat.sheenRoughness, lvMat.sheenColor,
      lvMat.iridescence, lvMat.iridescenceIor, lvMat.iridescenceThicknessMin, lvMat.iridescenceThicknessMax,
      lvMat.specularColor, lvMat.specularIntensity,
      lvMat.anisotropy, lvMat.anisotropyRotation, true,
    );
    // bdptGeometricTerm already contributes the light-vertex cosine.
    lightBsdfCosTheta = lvBrdf;
  }

  // ── Full §10.3 MIS weight ──────────────────────────────────────────────────
  let e = eyeDepth;
  let n = c + e + 3u;
  let selectedS = c + 1u;
  if (n > BDPT_MAX_MERGED) {
    return vec3f(0.0); // depth out of scratch range (should not happen)
  }
  let camPos = params.cameraPos.xyz;
  let camNrm = normalize(camPos - eyePos);

  // Connection-induced straddle overrides (PBRT MISWeight remapping).
  let lcToE = -connDir;                          // L_c → E_e
  // A9 — forward arrival density at E_e from L_c: the REAL light-vertex BSDF pdf
  // (incoming = lvWoPrev, outgoing = lcToE) for a surface vertex; Lambertian for the
  // emitter (matId < 0). Keeps the MIS pdf bookkeeping consistent with the glossy
  // light-vertex BSDF used in lightBsdfCosTheta.
  var fwdEe = bdptLambertDirPdf(lightNormal, lcToE);
  if (isPointEndpoint) {
    fwdEe = 0.25 * INV_PI;
  } else if (isSpotEndpoint) {
    let solidAngle = 2.0 * PI * (1.0 - lv4.w);
    fwdEe = select(0.0, 1.0 / solidAngle, solidAngle > 0.0);
  } else if (lightIsMedium) {
    fwdEe = hgPhase(dot(lvWoPrev, connDir), lightMediumG);
  } else if (lvMatId >= 0.0) {
    let lvMatF = bdptSampleMaterialAtPayload(u32(lvMatId), lv4, lightNormal, lvWoPrev, bdptInvocationHeroLambdaNm);
    if (lvMatF.transmission > 0.0 && lvMatF.metallic == 0.0) {
      fwdEe = bdptTransmissiveConnectionPdf(
        lvMatF.roughness,
        lvMatF.transmission,
        max(lv5.x, 1e-4),
        lightNormal,
        lvMatF.clearcoatNormal,
        lvWoPrev,
        lcToE,
        lvMatF.clearcoat,
        lvMatF.clearcoatRoughness,
        lvMatF.sheen,
        lvMatF.sheenRoughness,
        lvMatF.iridescence,
        lvMatF.iridescenceIor,
        lvMatF.iridescenceThicknessMin,
        lvMatF.iridescenceThicknessMax,
        lvMatF.specularColor,
        lvMatF.specularIntensity,
        lvMatF.anisotropy,
        lvMatF.anisotropyRotation,
      );
    } else {
      fwdEe = brdfDirectionalPdfFullSampledWithClearcoatNormal(
        lvMatF.baseColor, lvMatF.roughness, lvMatF.metallic,
        0.0, lvMatF.ior, lightNormal, lvMatF.clearcoatNormal, lvWoPrev, lcToE,
        lvMatF.clearcoat, lvMatF.clearcoatRoughness, lvMatF.sheen, lvMatF.sheenRoughness,
        lvMatF.iridescence, lvMatF.iridescenceIor, lvMatF.iridescenceThicknessMin, lvMatF.iridescenceThicknessMax,
        lvMatF.specularColor, lvMatF.specularIntensity,
        lvMatF.anisotropy, lvMatF.anisotropyRotation,
      );
    }
  }
  fwdEe = fwdEe * bdptSegmentDistanceDensity(
    connectionMedium.matId,
    dist,
    connectionMedium.remainingDistance,
    eyeIsMedium,
    bdptInvocationHeroLambdaNm,
  );

  // E_{e-1} position from scratch (if e>=1); else camera endpoint.
  var eeMinusPos = camPos;
  if (e >= 1u) {
    let prevEye = bdptEyeStackLoad(e - 1u);
    eeMinusPos = prevEye.pos;
  }
  let eeToPrev = normalize(eeMinusPos - eyePos);  // E_e → E_{e-1} (or → camera at e=0)
  var revLc = 0.0;
  if (eyeIsMedium) {
    revLc = hgPhase(dot(-eeToPrev, connDir), eyeMediumG);
  } else if (transmission > 0.0 && metallic == 0.0) {
    revLc = bdptTransmissiveConnectionPdf(
      roughness,
      transmission,
      etaTOverI,
      eyeNormal,
      eyeClearcoatNormal,
      eeToPrev,
      connDir,
      clearcoat,
      clearcoatRoughness,
      sheen,
      sheenRoughness,
      iridescence,
      iridescenceIor,
      iridescenceThicknessMin,
      iridescenceThicknessMax,
      specularColor,
      specularIntensity,
      anisotropy,
      anisotropyRotation,
    );
  } else {
    revLc = brdfDirectionalPdfFullSampledWithClearcoatNormal(
      baseColor, roughness, metallic, transmission, etaTOverI, eyeNormal, eyeClearcoatNormal, eeToPrev, connDir,
      clearcoat, clearcoatRoughness, sheen, sheenRoughness,
      iridescence, iridescenceIor, iridescenceThicknessMin, iridescenceThicknessMax,
      specularColor, specularIntensity,
      anisotropy, anisotropyRotation,
    );
  }
  revLc = revLc * bdptSegmentDistanceDensity(
    connectionMedium.matId,
    dist,
    connectionMedium.remainingDistance,
    lightIsMedium,
    bdptInvocationHeroLambdaNm,
  );
  var fwdEeMinus = 0.0;
  if (e >= 1u) {
    if (eyeIsMedium) {
      fwdEeMinus = hgPhase(dot(-connDir, eeToPrev), eyeMediumG);
    } else if (transmission > 0.0 && metallic == 0.0) {
      fwdEeMinus = bdptTransmissiveConnectionPdf(
        roughness,
        transmission,
        etaTOverI,
        eyeNormal,
        eyeClearcoatNormal,
        connDir,
        eeToPrev,
        clearcoat,
        clearcoatRoughness,
        sheen,
        sheenRoughness,
        iridescence,
        iridescenceIor,
        iridescenceThicknessMin,
        iridescenceThicknessMax,
        specularColor,
        specularIntensity,
        anisotropy,
        anisotropyRotation,
      );
    } else {
      fwdEeMinus = brdfDirectionalPdfFullSampledWithClearcoatNormal(
        baseColor, roughness, metallic, transmission, etaTOverI, eyeNormal, eyeClearcoatNormal, connDir, eeToPrev,
        clearcoat, clearcoatRoughness, sheen, sheenRoughness,
        iridescence, iridescenceIor, iridescenceThicknessMin, iridescenceThicknessMax,
        specularColor, specularIntensity,
        anisotropy, anisotropyRotation,
      );
    }
    let prevEyeForDensity = bdptEyeStackLoad(e - 1u);
    let currentEyePrefixMedium = bdptSelectEndpointMedium(
      eyeIsMedium, eyeNormal, eeToPrev,
      eyeMediumMatId, eyeRecord.mediumRemainingDistance,
      eyeRecord.incidentMediumMatId, eyeRecord.incidentMediumRemainingDistance,
      eyeRecord.transmittedMediumMatId,
      eyeRecord.transmittedMediumRemainingDistance,
    );
    let previousEyePrefixMedium = bdptSelectEndpointMedium(
      prevEyeForDensity.medium, prevEyeForDensity.nrm, -eeToPrev,
      prevEyeForDensity.mediumMatId,
      prevEyeForDensity.mediumRemainingDistance,
      prevEyeForDensity.incidentMediumMatId,
      prevEyeForDensity.incidentMediumRemainingDistance,
      prevEyeForDensity.transmittedMediumMatId,
      prevEyeForDensity.transmittedMediumRemainingDistance,
    );
    fwdEeMinus = fwdEeMinus * bdptEndpointEdgeDistanceDensity(
      currentEyePrefixMedium, previousEyePrefixMedium,
      distance(eyePos, prevEyeForDensity.pos),
      prevEyeForDensity.medium,
      bdptInvocationHeroLambdaNm,
    );
  }
  // L_{c-1} override: reverse density at L_c toward L_{c-1}. REAL BSDF pdf (outgoing
  // = lcToE toward E_e, incoming = direction to L_{c-1}) for a surface vertex;
  // Lambertian for the emitter.
  var revLcMinus = 0.0;
  if (c >= 1u) {
    let lcm0 = bdptLightPath[bdptLightPathIndex(i32(c - 1u), 0u)];
    let lcToLcMinus = normalize(lcm0.xyz - lightPos);
    if (lightIsMedium) {
      revLcMinus = hgPhase(dot(connDir, lcToLcMinus), lightMediumG);
    } else if (lvMatId >= 0.0) {
      let lvMatR = bdptSampleMaterialAtPayload(u32(lvMatId), lv4, lightNormal, lvWoPrev, bdptInvocationHeroLambdaNm);
      if (lvMatR.transmission > 0.0 && lvMatR.metallic == 0.0) {
        revLcMinus = bdptTransmissiveConnectionPdf(
          lvMatR.roughness,
          lvMatR.transmission,
          max(lv5.x, 1e-4),
          lightNormal,
          lvMatR.clearcoatNormal,
          lcToE,
          lcToLcMinus,
          lvMatR.clearcoat,
          lvMatR.clearcoatRoughness,
          lvMatR.sheen,
          lvMatR.sheenRoughness,
          lvMatR.iridescence,
          lvMatR.iridescenceIor,
          lvMatR.iridescenceThicknessMin,
          lvMatR.iridescenceThicknessMax,
          lvMatR.specularColor,
          lvMatR.specularIntensity,
          lvMatR.anisotropy,
          lvMatR.anisotropyRotation,
        );
      } else {
        revLcMinus = brdfDirectionalPdfFullSampledWithClearcoatNormal(
          lvMatR.baseColor, lvMatR.roughness, lvMatR.metallic,
          0.0, lvMatR.ior, lightNormal, lvMatR.clearcoatNormal, lcToE, lcToLcMinus,
          lvMatR.clearcoat, lvMatR.clearcoatRoughness, lvMatR.sheen, lvMatR.sheenRoughness,
          lvMatR.iridescence, lvMatR.iridescenceIor, lvMatR.iridescenceThicknessMin, lvMatR.iridescenceThicknessMax,
          lvMatR.specularColor, lvMatR.specularIntensity,
          lvMatR.anisotropy, lvMatR.anisotropyRotation,
        );
      }
    } else {
      revLcMinus = bdptLambertDirPdf(lightNormal, lcToLcMinus);
    }
    let lcm3 = bdptLightPath[bdptLightPathIndex(i32(c - 1u), 3u)];
    let lcm1 = bdptLightPath[bdptLightPathIndex(i32(c - 1u), 1u)];
    let lcm5 = bdptLightPath[bdptLightPathIndex(i32(c - 1u), 5u)];
    let lcm6 = bdptLightPath[bdptLightPathIndex(i32(c - 1u), 6u)];
    let previousLightIsMedium = lcm3.w == BDPT_LV_MEDIUM_MATID;
    let currentLightPrefixMedium = bdptSelectEndpointMedium(
      lightIsMedium, lightNormal, lcToLcMinus,
      lightMediumMatId, lightMediumRemainingDistance,
      lightIncidentMediumMatId, lightIncidentMediumRemainingDistance,
      lightTransmittedMediumMatId,
      lightTransmittedMediumRemainingDistance,
    );
    let previousLightPrefixMedium = bdptSelectEndpointMedium(
      previousLightIsMedium, lcm1.xyz, -lcToLcMinus,
      bitcast<u32>(lcm5.w), lcm6.y,
      bitcast<u32>(lcm6.x), lcm6.y,
      bitcast<u32>(lcm6.z), lcm6.w,
    );
    revLcMinus = revLcMinus * bdptEndpointEdgeDistanceDensity(
      currentLightPrefixMedium, previousLightPrefixMedium,
      distance(lightPos, lcm0.xyz),
      previousLightIsMedium,
      bdptInvocationHeroLambdaNm,
    );
  }

  let root1 = bdptLightPath[bdptLightPathIndex(0, 1u)];
  let root3 = bdptLightPath[bdptLightPathIndex(0, 3u)];
  let root4 = bdptLightPath[bdptLightPathIndex(0, 4u)];
  let infiniteRoot =
    root3.w == BDPT_LV_DIRECTIONAL_EMITTER_MATID ||
    root3.w == BDPT_LV_ENVIRONMENT_EMITTER_MATID;
  let infiniteEnvironmentRoot =
    root3.w == BDPT_LV_ENVIRONMENT_EMITTER_MATID;
  let infiniteSourceDirection = safe_normalize(root1.xyz);
  let infiniteNeePdf = select(0.0, root4.z, infiniteRoot);
  let infiniteLaunchPdf = select(0.0, root1.w * root4.y, infiniteRoot);
  var infiniteEyeEscapePdf = 0.0;
  var infiniteEyeEscapeDelta = false;
  if (infiniteRoot && c >= 1u) {
    let first0 = bdptLightPath[bdptLightPathIndex(1, 0u)];
    let first1 = bdptLightPath[bdptLightPathIndex(1, 1u)];
    let first3 = bdptLightPath[bdptLightPathIndex(1, 3u)];
    let first4 = bdptLightPath[bdptLightPathIndex(1, 4u)];
    let first5 = bdptLightPath[bdptLightPathIndex(1, 5u)];
    let firstIsMedium = first3.w == BDPT_LV_MEDIUM_MATID;
    var firstCamerawardPosition = eyePos;
    if (c >= 2u) {
      firstCamerawardPosition =
        bdptLightPath[bdptLightPathIndex(2, 0u)].xyz;
    }
    let firstCamerawardDirection = safe_normalize(
      firstCamerawardPosition - first0.xyz,
    );
    infiniteEyeEscapeDelta = first0.w == BDPT_KIND_DELTA;
    if (firstIsMedium) {
      infiniteEyeEscapePdf = hgPhase(
        dot(-firstCamerawardDirection, infiniteSourceDirection), first5.y,
      );
    } else if (first3.w >= 0.0 && !infiniteEyeEscapeDelta) {
      let firstMaterial = bdptSampleMaterialAtPayload(
        u32(first3.w), first4, first1.xyz, firstCamerawardDirection,
        bdptInvocationHeroLambdaNm,
      );
      if (!firstMaterial.thinFilmEnabled && bsdfHasFiniteConnectionSupport(
        firstMaterial.roughness,
        firstMaterial.metallic,
        firstMaterial.transmission,
        firstMaterial.clearcoat,
        firstMaterial.sheen,
      )) {
        if (
          firstMaterial.transmission > 0.0 &&
          firstMaterial.metallic == 0.0
        ) {
          infiniteEyeEscapePdf = bdptTransmissiveConnectionPdf(
            firstMaterial.roughness,
            firstMaterial.transmission,
            max(first5.x, 1e-4),
            first1.xyz,
            firstMaterial.clearcoatNormal,
            firstCamerawardDirection,
            infiniteSourceDirection,
            firstMaterial.clearcoat,
            firstMaterial.clearcoatRoughness,
            firstMaterial.sheen,
            firstMaterial.sheenRoughness,
            firstMaterial.iridescence,
            firstMaterial.iridescenceIor,
            firstMaterial.iridescenceThicknessMin,
            firstMaterial.iridescenceThicknessMax,
            firstMaterial.specularColor,
            firstMaterial.specularIntensity,
            firstMaterial.anisotropy,
            firstMaterial.anisotropyRotation,
          );
        } else {
          infiniteEyeEscapePdf =
            brdfDirectionalPdfFullSampledWithClearcoatNormal(
              firstMaterial.baseColor,
              firstMaterial.roughness,
              firstMaterial.metallic,
              firstMaterial.transmission,
              max(first5.x, 1e-4),
              first1.xyz,
              firstMaterial.clearcoatNormal,
              firstCamerawardDirection,
              infiniteSourceDirection,
              firstMaterial.clearcoat,
              firstMaterial.clearcoatRoughness,
              firstMaterial.sheen,
              firstMaterial.sheenRoughness,
              firstMaterial.iridescence,
              firstMaterial.iridescenceIor,
              firstMaterial.iridescenceThicknessMin,
              firstMaterial.iridescenceThicknessMax,
              firstMaterial.specularColor,
              firstMaterial.specularIntensity,
              firstMaterial.anisotropy,
              firstMaterial.anisotropyRotation,
            );
        }
      }
    }
  }

  let misW = bdptMISWeightFull(
    c, e, n, selectedS,
    infiniteRoot,
    infiniteEnvironmentRoot,
    infiniteSourceDirection,
    infiniteNeePdf,
    infiniteLaunchPdf,
    infiniteEyeEscapePdf,
    infiniteEyeEscapeDelta,
    fwdEe, fwdEeMinus, revLc, revLcMinus, camPos, camNrm,
  );
  if (misW <= 0.0) {
    return vec3f(0.0);
  }

  var contribution = lightThroughput * lightBsdfCosTheta * gTerm * eyeBsdfCosTheta * misW;
  contribution = contribution * eyeThroughput;
  // WGSL has no isNan/isInf builtins (those calls fail to resolve on Dawn).
  // For every finite f32, x-x is exactly zero; NaN and ±Inf instead produce
  contribution = contribution * connectionTransmittance;
  // NaN, which compares unequal to zero. Negative radiance is invalid here
  // because every physical factor in this estimator is non-negative.
  let finiteProbe = contribution - contribution;
  let isInvalid =
    any(finiteProbe != vec3f(0.0)) ||
    any(contribution < vec3f(0.0));
  if (isInvalid) {
    return vec3f(0.0);
  }
  return contribution;
}
`;
