import type { WgslModule } from '../pipeline/wgslComposer.js';

/**
 * Fixed-offset specular-manifold solver used by the bounded realtime SMS
 * estimator. Residual finite differences re-evaluate smooth, normal, bump, and
 * face-layer normals at every perturbed point, following Zeltner et al. 2020.
 */
export const MANIFOLD_SMS_SOLVER_WGSL = /* wgsl */ `
const SMS_MAX_VERTICES = 8u;
const SMS_F32_EPSILON = 1.1920928955078125e-7;
const SMS_MIN_NORMAL_F32 = 1.1754943508222875e-38;
const SMS_EVENT_REFLECTION = 0u;
const SMS_EVENT_TRANSMISSION = 1u;

fn smsConfiguredIterations() -> u32 {
  return clamp(bitcast<u32>(ubo.sunAngular.w) & 0xffu, 1u, 32u);
}

fn smsConfiguredChainLength() -> u32 {
  return clamp((bitcast<u32>(ubo.sunAngular.w) >> 8u) & 0xffu, 1u, SMS_MAX_VERTICES);
}

fn smsConfiguredMultiplicityTrials() -> u32 {
  return clamp((bitcast<u32>(ubo.sunAngular.w) >> 16u) & 0xffu, 1u, 32u);
}

// Rejection-based bounded draw. Every u32 identity remains reachable and no
// modulo or f32 mantissa bias is introduced for large scenes.
fn smsUniformIndex(rng: ptr<function, u32>, count: u32) -> u32 {
  if (count <= 1u) { return 0u; }
  let threshold = ((0xffffffffu % count) + 1u) % count;
  loop {
    let word = pcgNext(rng);
    if (word >= threshold) { return word % count; }
  }
  return 0u;
}

struct SmsFacet {
  valid: u32,
  triIndex: u32,
  instanceIndex: u32,
  encodedBoundaryId: u32,
  representedPrimitiveInstanceId: u32,
  indices: vec4u,
  a: vec3f,
  b: vec3f,
  c: vec3f,
  planePoint: vec3f,
  geometricNormal: vec3f,
  planeU: vec3f,
  planeV: vec3f,
  area: f32,
  // Discrete topology probability only. Uniform-area initialization is a
  // root-finding law whose basin probability is inverted by the recurrence;
  // dividing by its point density again would double count it.
  pairPmf: f32,
};

struct SmsFacetDraw {
  facet: SmsFacet,
  seed: vec3f,
};

fn smsInvalidFacet() -> SmsFacet {
  var facet: SmsFacet;
  facet.valid = 0u;
  facet.triIndex = 0xffffffffu;
  facet.instanceIndex = 0xffffffffu;
  facet.encodedBoundaryId = 0u;
  facet.representedPrimitiveInstanceId = 0u;
  facet.indices = vec4u(0u);
  facet.a = vec3f(0.0);
  facet.b = vec3f(0.0);
  facet.c = vec3f(0.0);
  facet.planePoint = vec3f(0.0);
  facet.geometricNormal = vec3f(0.0, 1.0, 0.0);
  facet.planeU = vec3f(1.0, 0.0, 0.0);
  facet.planeV = vec3f(0.0, 0.0, 1.0);
  facet.area = 0.0;
  facet.pairPmf = 0.0;
  return facet;
}

fn smsPointToWorld(point: vec3f, instanceIndex: u32) -> vec3f {
  let base = instanceIndex * 4u;
  if (ubo.bvhMode != 1u || base + 3u >= tlasLocalToWorldColumnCount()) {
    return point;
  }
  return tlasTransformPointCols(
    tlasLoadLocalToWorldColumn(base),
    tlasLoadLocalToWorldColumn(base + 1u),
    tlasLoadLocalToWorldColumn(base + 2u),
    tlasLoadLocalToWorldColumn(base + 3u),
    point,
  );
}

fn smsFacetFromIdentity(
  triIndex: u32,
  instanceIndex: u32,
  pairPmf: f32,
) -> SmsFacet {
  var facet = smsInvalidFacet();
  if (triIndex >= bvhIndexCount()) { return facet; }
  let indices = bvhLoadIndex(triIndex);
  if (indices.x >= bvhPositionCount() || indices.y >= bvhPositionCount() ||
      indices.z >= bvhPositionCount()) { return facet; }
  let a = smsPointToWorld(bvhLoadPosition(indices.x).xyz, instanceIndex);
  let b = smsPointToWorld(bvhLoadPosition(indices.y).xyz, instanceIndex);
  let c = smsPointToWorld(bvhLoadPosition(indices.z).xyz, instanceIndex);
  let areaVector = cross(b - a, c - a);
  let area2 = length(areaVector);
  let useTlas = ubo.bvhMode == 1u && ubo.tlasNodeCount > 0u;
  let representedPrimitiveInstanceId =
    sceneOpticalRepresentedPrimitiveInstanceId(
      useTlas, triIndex, instanceIndex,
    );
  if (!(area2 > 0.0) || !(area2 < INFINITY) ||
      !(pairPmf > 0.0) || !(pairPmf < INFINITY) ||
      representedPrimitiveInstanceId == 0u) { return facet; }
  let planeU = normalize(b - a);
  let normal = areaVector / area2;
  facet.valid = 1u;
  facet.triIndex = triIndex;
  facet.instanceIndex = instanceIndex;
  facet.encodedBoundaryId = sceneOpticalEncodedBoundaryId(
    useTlas, triIndex, instanceIndex,
  );
  facet.representedPrimitiveInstanceId =
    representedPrimitiveInstanceId;
  facet.indices = indices;
  facet.a = a;
  facet.b = b;
  facet.c = c;
  facet.planePoint = (a + b + c) / 3.0;
  facet.geometricNormal = normal;
  facet.planeU = planeU;
  facet.planeV = normalize(cross(normal, planeU));
  facet.area = 0.5 * area2;
  facet.pairPmf = pairPmf;
  return facet;
}

fn smsDrawFacet(rng: ptr<function, u32>) -> SmsFacetDraw {
  var out: SmsFacetDraw;
  out.facet = smsInvalidFacet();
  out.seed = vec3f(0.0);
  let domainCount = sceneMneeFacetDomainCount();
  if (domainCount == 0u) { return out; }
  let column = smsUniformIndex(rng, domainCount);
  let columnAlias = sceneLoadMneeFacetDomainAlias(column);
  let q = bitcast<f32>(columnAlias.x);
  if (!(q >= 0.0 && q <= 1.0)) { return out; }
  let domainIndex = select(columnAlias.y, column, rand_f32(rng) < q);
  if (domainIndex >= domainCount) { return out; }
  let domain = sceneLoadMneeFacetDomainBase(domainIndex);
  let represented = sceneLoadMneeFacetDomainAlias(domainIndex);
  let domainPmf = bitcast<f32>(represented.z);
  if (domain.y == 0u || domain.w == 0u ||
      !(domainPmf > 0.0) || !(domainPmf < INFINITY)) { return out; }
  let triOffset = smsUniformIndex(rng, domain.y);
  let instanceOffset = smsUniformIndex(rng, domain.w);
  let pairPmf = domainPmf / (f32(domain.y) * f32(domain.w));
  out.facet = smsFacetFromIdentity(
    domain.x + triOffset,
    domain.z + instanceOffset,
    pairPmf,
  );
  let xi = rand2(rng);
  let s = sqrt(xi.x);
  let bary = vec3f(1.0 - s, s * (1.0 - xi.y), s * xi.y);
  out.seed = bary.x * out.facet.a + bary.y * out.facet.b + bary.z * out.facet.c;
  return out;
}

fn smsFacetBarycentric(facet: SmsFacet, point: vec3f) -> vec3f {
  let e0 = facet.b - facet.a;
  let e1 = facet.c - facet.a;
  let ep = point - facet.a;
  let d00 = dot(e0, e0);
  let d01 = dot(e0, e1);
  let d11 = dot(e1, e1);
  let d20 = dot(ep, e0);
  let d21 = dot(ep, e1);
  let denominator = d00 * d11 - d01 * d01;
  if (!(abs(denominator) > 0.0)) { return vec3f(-1.0); }
  let v = (d11 * d20 - d01 * d21) / denominator;
  let w = (d00 * d21 - d01 * d20) / denominator;
  return vec3f(1.0 - v - w, v, w);
}

fn smsFacetContains(facet: SmsFacet, point: vec3f) -> bool {
  let bary = smsFacetBarycentric(facet, point);
  let tolerance = 32.0 * SMS_F32_EPSILON;
  return all(bary >= vec3f(-tolerance)) && all(bary <= vec3f(1.0 + tolerance));
}

struct SmsSurfaceFrame {
  hit: IntersectionResult,
  normal: vec3f,
  tangent: vec3f,
  bitangent: vec3f,
  valid: u32,
};

// Duff et al. 2017 orthonormal basis: continuous except at the unavoidable
// south-pole branch and independent of UV tangent availability.
fn smsBuildFrame(normal: vec3f, tangent: ptr<function, vec3f>, bitangent: ptr<function, vec3f>) {
  if (normal.z < -0.9999999) {
    *tangent = vec3f(0.0, -1.0, 0.0);
    *bitangent = vec3f(-1.0, 0.0, 0.0);
    return;
  }
  let a = 1.0 / (1.0 + normal.z);
  let b = -normal.x * normal.y * a;
  *tangent = vec3f(1.0 - normal.x * normal.x * a, b, -normal.x);
  *bitangent = vec3f(b, 1.0 - normal.y * normal.y * a, -normal.y);
}

fn smsSurfaceFrameAt(
  facet: SmsFacet,
  point: vec3f,
  incomingTravel: vec3f,
) -> SmsSurfaceFrame {
  var out: SmsSurfaceFrame;
  var invalidHit: IntersectionResult;
  invalidHit.didHit = false;
  invalidHit.indices = vec4u(0u);
  invalidHit.normal = vec3f(0.0, 1.0, 0.0);
  invalidHit.barycoord = vec3f(0.0);
  invalidHit.side = 1.0;
  invalidHit.dist = 0.0;
  invalidHit.matColorPacked = 0u;
  invalidHit.uv = vec2f(0.0);
  invalidHit.instanceIndex = 0u;
  out.hit = invalidHit;
  out.normal = vec3f(0.0, 1.0, 0.0);
  out.tangent = vec3f(1.0, 0.0, 0.0);
  out.bitangent = vec3f(0.0, 0.0, 1.0);
  out.valid = 0u;
  let bary = smsFacetBarycentric(facet, point);
  let side = select(-1.0, 1.0, dot(incomingTravel, facet.geometricNormal) < 0.0);
  var hit: IntersectionResult;
  hit.didHit = true;
  hit.indices = facet.indices;
  hit.indices.w = facet.triIndex;
  hit.normal = side * facet.geometricNormal;
  hit.barycoord = bary;
  hit.side = side;
  hit.dist = 0.0;
  hit.matColorPacked = facet.indices.w;
  let p0 = bvhLoadPosition(facet.indices.x);
  let p1 = bvhLoadPosition(facet.indices.y);
  let p2 = bvhLoadPosition(facet.indices.z);
  hit.uv = bary.x * materialAtlasPackedUvFromVec4(p0) +
    bary.y * materialAtlasPackedUvFromVec4(p1) +
    bary.z * materialAtlasPackedUvFromVec4(p2);
  hit.instanceIndex = facet.instanceIndex;
  let base = facet.instanceIndex * 4u;
  let hasTlasNormal = ubo.bvhMode == 1u &&
    base + 2u < tlasWorldToLocalColumnCount();
  var smoothNormal: vec3f;
  if (hasTlasNormal) {
    smoothNormal = smoothShadingNormal(
      hit,
      hit.normal,
      sceneLoadBvhNormal(facet.indices.x).xyz,
      sceneLoadBvhNormal(facet.indices.y).xyz,
      sceneLoadBvhNormal(facet.indices.z).xyz,
      true,
      tlasLoadWorldToLocalColumn(base),
      tlasLoadWorldToLocalColumn(base + 1u),
      tlasLoadWorldToLocalColumn(base + 2u),
    );
  } else {
    smoothNormal = smoothShadingNormal(
      hit,
      hit.normal,
      sceneLoadBvhNormal(facet.indices.x).xyz,
      sceneLoadBvhNormal(facet.indices.y).xyz,
      sceneLoadBvhNormal(facet.indices.z).xyz,
      false,
      vec4f(0.0), vec4f(0.0), vec4f(0.0),
    );
  }
  var mapped = applyBumpMapForHit(hit, applyNormalMapForHit(hit, smoothNormal));
  mapped = select(-mapped, mapped, dot(mapped, hit.normal) >= 0.0);
  let mappedLength = length(mapped);
  if (!(mappedLength > 0.0) || !(mappedLength < INFINITY)) { return out; }
  mapped = mapped / mappedLength;
  var tangent: vec3f;
  var bitangent: vec3f;
  smsBuildFrame(mapped, &tangent, &bitangent);
  out.hit = hit;
  out.normal = mapped;
  out.tangent = tangent;
  out.bitangent = bitangent;
  out.valid = 1u;
  return out;
}

struct SmsFacetOptics {
  roughness: f32,
  transmission: f32,
  iorRgb: vec3f,
  layerTransmission: vec3f,
  bulkMedium: u32,
  thickness: f32,
  materialWord: u32,
  metalness: f32,
  surfaceCoverage: f32,
  frame: SmsSurfaceFrame,
};

fn smsFacetOpticsAt(facet: SmsFacet, point: vec3f, incomingTravel: vec3f) -> SmsFacetOptics {
  let frame = smsSurfaceFrameAt(facet, point, incomingTravel);
  var out: SmsFacetOptics;
  out.roughness = 0.0;
  out.transmission = 0.0;
  out.iorRgb = vec3f(1.0);
  out.layerTransmission = vec3f(0.0);
  out.bulkMedium = 0u;
  out.thickness = 0.0;
  out.materialWord = 0u;
  out.metalness = 0.0;
  out.surfaceCoverage = 0.0;
  out.frame = frame;
  if (frame.valid == 0u) { return out; }
  let coord = vec2u(
    facet.triIndex % BVH_MATERIAL_TEX_WIDTH,
    facet.triIndex / BVH_MATERIAL_TEX_WIDTH,
  );
  let materialWord = textureLoad(bvh_material, vec2i(coord), 0).r;
  // castShadow:false surfaces do not own direct-light paths. Alpha masks are
  // binary support; alpha blend is a coverage mixture, so its covered fraction
  // weights this explicit surface path while ordinary NEE owns the complement.
  let alpha = materialAlphaCoverageForHit(frame.hit, materialWord);
  if (!materialSideAdmittedForHit(frame.hit) ||
      (materialWord & 1u) != 0u ||
      alpha.scalarDiscarded != 0u ||
      (alpha.mode == 1u && alpha.coverage < alpha.cutoff) ||
      alpha.mode > 2u ||
      (alpha.mode == 2u && !(alpha.coverage > 0.0))) {
    out.frame.valid = 0u;
    return out;
  }
  let roughMetal = decodeRoughMetal(materialWord);
  let scalar = decodeMaterialColor(facet.indices.w);
  let layer = sampleFaceLayerControls(facet.triIndex, frame.hit.side >= 0.0);
  out.roughness = faceLayerRoughness(
    sampleMaterialScalarMap(
      frame.hit,
      MATERIAL_MAP_SLOT_ROUGHNESS,
      1u,
      roughMetal.x,
    ),
    layer,
  );
  out.transmission = materialShadowMappedTransmission(frame.hit);
  out.iorRgb = materialDispersionIorRgb(facet.triIndex, decodeIor(materialWord));
  out.layerTransmission = faceLayerTransmission(layer);
  out.bulkMedium = select(0u, 1u, facet.encodedBoundaryId != 0u);
  out.thickness = materialShadowAuthoredThickness(frame.hit);
  out.materialWord = materialWord;
  out.metalness = roughMetal.y;
  out.surfaceCoverage = select(1.0, alpha.coverage, alpha.mode == 2u);
  out.frame = frame;
  return out;
}

struct SmsOffsetNormal {
  local: vec3f,
  proposalPdf: f32,
  roughness: f32,
  eventProposalPmf: f32,
  valid: u32,
};

// Heitz 2018 VNDF proposal at the seed. The sampled local normal is frozen for
// every Newton/Jacobian/multiplicity solve in this proposal, as required by the
// fixed-offset SMS construction. Roughness zero is an explicit delta mass.
fn smsSampleOffsetNormal(
  optics: SmsFacetOptics,
  incomingTravel: vec3f,
  event: u32,
  eventProposalPmf: f32,
  rng: ptr<function, u32>,
) -> SmsOffsetNormal {
  var out: SmsOffsetNormal;
  out.local = vec3f(0.0, 0.0, 1.0);
  out.proposalPdf = 1.0;
  out.roughness = optics.roughness;
  out.eventProposalPmf = eventProposalPmf;
  out.valid = 0u;
  if (optics.frame.valid == 0u ||
      !(eventProposalPmf > 0.0) || !(eventProposalPmf < INFINITY) ||
      (event == SMS_EVENT_TRANSMISSION &&
       (!(optics.transmission > 0.0) || optics.metalness > 0.0))) { return out; }
  if (!(optics.roughness > 0.0)) {
    out.valid = 1u;
    return out;
  }
  let wo = -safe_normalize(incomingTravel);
  let woLocal = vec3f(
    dot(wo, optics.frame.tangent),
    dot(wo, optics.frame.bitangent),
    dot(wo, optics.frame.normal),
  );
  if (!(woLocal.z > 0.0)) { return out; }
  let alpha = optics.roughness * optics.roughness;
  let local = ggxSampleVndfTangent(woLocal, alpha, rng);
  let woDotM = dot(woLocal, local);
  let D = distributionGGX(local.z, optics.roughness);
  let G1 = smithG1GGX(woLocal.z, alpha * alpha);
  let pdf = D * G1 * abs(woDotM) / woLocal.z;
  if (!(pdf > 0.0) || !(pdf < INFINITY)) { return out; }
  out.local = local;
  out.proposalPdf = pdf;
  out.valid = 1u;
  return out;
}

struct SmsChainGeometry {
  count: u32,
  sourceMode: u32,
  sourceDirection: vec3f,
  facets: array<SmsFacet, 8>,
  offsets: array<SmsOffsetNormal, 8>,
  events: array<u32, 8>,
};

struct SmsChainMedia {
  etaI: array<f32, 8>,
  etaT: array<f32, 8>,
};

struct SmsChainResult {
  vertices: array<vec3f, 8>,
  residual: f32,
  iterations: u32,
  valid: u32,
};

fn smsPointScale(point: vec3f) -> f32 {
  return max(1.0, max(abs(point.x), max(abs(point.y), abs(point.z))));
}

fn smsFiniteDifferenceStep(geometry: SmsChainGeometry, source: vec3f, receiver: vec3f) -> f32 {
  var scale = max(smsPointScale(source), smsPointScale(receiver));
  var span = length(receiver - source);
  for (var i = 0u; i < SMS_MAX_VERTICES; i = i + 1u) {
    if (i >= geometry.count) { break; }
    scale = max(scale, smsPointScale(geometry.facets[i].planePoint));
    span = max(span, length(geometry.facets[i].planePoint - receiver));
  }
  return max(
    8.0 * SMS_F32_EPSILON * scale,
    sqrt(SMS_F32_EPSILON) * max(span, scale * 0.01),
  );
}

fn smsResidualAt(
  vertices: array<vec3f, 8>,
  geometry: SmsChainGeometry,
  media: SmsChainMedia,
  index: u32,
  source: vec3f,
  receiver: vec3f,
) -> vec2f {
  let vertex = vertices[index];
  var previous = source;
  if (index > 0u) { previous = vertices[index - 1u]; }
  var next = receiver;
  if (index + 1u < geometry.count) { next = vertices[index + 1u]; }
  var towardSource = safe_normalize(previous - vertex);
  var incomingTravel = safe_normalize(vertex - previous);
  if (index == 0u && geometry.sourceMode == 1u) {
    towardSource = safe_normalize(geometry.sourceDirection);
    incomingTravel = -towardSource;
  }
  let towardReceiver = safe_normalize(next - vertex);
  let frame = smsSurfaceFrameAt(geometry.facets[index], vertex, incomingTravel);
  if (frame.valid == 0u) { return vec2f(INFINITY); }
  var halfVector = safe_normalize(towardSource + towardReceiver);
  if (geometry.events[index] == SMS_EVENT_TRANSMISSION) {
    halfVector = safe_normalize(
      media.etaI[index] * towardSource + media.etaT[index] * towardReceiver,
    );
  }
  if (dot(halfVector, frame.normal) < 0.0) { halfVector = -halfVector; }
  return vec2f(dot(halfVector, frame.tangent), dot(halfVector, frame.bitangent)) -
    geometry.offsets[index].local.xy;
}

fn smsResidualMax(
  vertices: array<vec3f, 8>,
  geometry: SmsChainGeometry,
  media: SmsChainMedia,
  source: vec3f,
  receiver: vec3f,
) -> f32 {
  var maximum = 0.0;
  for (var i = 0u; i < SMS_MAX_VERTICES; i = i + 1u) {
    if (i >= geometry.count) { break; }
    maximum = max(maximum, length(smsResidualAt(vertices, geometry, media, i, source, receiver)));
  }
  return maximum;
}

fn smsMat2Det(value: mat2x2f) -> f32 {
  return value[0][0] * value[1][1] - value[1][0] * value[0][1];
}

fn smsMat2Invertible(value: mat2x2f) -> bool {
  let determinant = smsMat2Det(value);
  let scale = max(abs(value[0][0]), max(abs(value[0][1]), max(abs(value[1][0]), abs(value[1][1]))));
  let absoluteDeterminant = abs(determinant);
  return scale > 0.0 && scale < INFINITY &&
    absoluteDeterminant >= SMS_MIN_NORMAL_F32 &&
    absoluteDeterminant > 64.0 * SMS_F32_EPSILON * scale * scale &&
    absoluteDeterminant < INFINITY;
}

fn smsSolutionInsideFacets(
  vertices: array<vec3f, 8>,
  geometry: SmsChainGeometry,
) -> bool {
  for (var i = 0u; i < SMS_MAX_VERTICES; i = i + 1u) {
    if (i >= geometry.count) { break; }
    if (!smsFacetContains(geometry.facets[i], vertices[i])) { return false; }
  }
  return true;
}

fn smsMat2Inverse(value: mat2x2f) -> mat2x2f {
  let reciprocal = 1.0 / smsMat2Det(value);
  return mat2x2f(
    vec2f(value[1][1], -value[0][1]) * reciprocal,
    vec2f(-value[1][0], value[0][0]) * reciprocal,
  );
}

fn smsSolveChain(
  geometry: SmsChainGeometry,
  media: SmsChainMedia,
  source: vec3f,
  receiver: vec3f,
  seeds: array<vec3f, 8>,
) -> SmsChainResult {
  var out: SmsChainResult;
  out.vertices = seeds;
  out.residual = INFINITY;
  out.iterations = 0u;
  out.valid = 0u;
  if (geometry.count == 0u || geometry.count > SMS_MAX_VERTICES) { return out; }
  var coordA: array<f32, 8>;
  var coordB: array<f32, 8>;
  for (var i = 0u; i < SMS_MAX_VERTICES; i = i + 1u) {
    if (i >= geometry.count) { break; }
    let offset = seeds[i] - geometry.facets[i].planePoint;
    coordA[i] = dot(offset, geometry.facets[i].planeU);
    coordB[i] = dot(offset, geometry.facets[i].planeV);
  }
  let epsilon = smsFiniteDifferenceStep(geometry, source, receiver);
  let tolerance = max(32.0 * SMS_F32_EPSILON, 4.0 * epsilon / max(length(receiver - source), epsilon));
  let maxIterations = smsConfiguredIterations();
  for (var iteration = 0u; iteration < 32u; iteration = iteration + 1u) {
    if (iteration >= maxIterations) { break; }
    var vertices: array<vec3f, 8>;
    for (var vi = 0u; vi < SMS_MAX_VERTICES; vi = vi + 1u) {
      if (vi >= geometry.count) { break; }
      vertices[vi] = geometry.facets[vi].planePoint +
        coordA[vi] * geometry.facets[vi].planeU +
        coordB[vi] * geometry.facets[vi].planeV;
    }
    let currentResidual = smsResidualMax(vertices, geometry, media, source, receiver);
    out.vertices = vertices;
    out.residual = currentResidual;
    out.iterations = iteration;
    if (currentResidual <= tolerance && smsSolutionInsideFacets(vertices, geometry)) {
      out.valid = 1u;
      return out;
    }
    var residual: array<vec2f, 8>;
    var lower: array<mat2x2f, 8>;
    var diagonal: array<mat2x2f, 8>;
    var upper: array<mat2x2f, 8>;
    for (var ji = 0u; ji < SMS_MAX_VERTICES; ji = ji + 1u) {
      if (ji >= geometry.count) { break; }
      let r0 = smsResidualAt(vertices, geometry, media, ji, source, receiver);
      residual[ji] = r0;
      var perturbed = vertices;
      perturbed[ji] = vertices[ji] + epsilon * geometry.facets[ji].planeU;
      let ownA = (smsResidualAt(perturbed, geometry, media, ji, source, receiver) - r0) / epsilon;
      perturbed[ji] = vertices[ji] + epsilon * geometry.facets[ji].planeV;
      let ownB = (smsResidualAt(perturbed, geometry, media, ji, source, receiver) - r0) / epsilon;
      diagonal[ji] = mat2x2f(ownA, ownB);
      lower[ji] = mat2x2f(vec2f(0.0), vec2f(0.0));
      if (ji > 0u) {
        perturbed = vertices;
        perturbed[ji - 1u] = vertices[ji - 1u] + epsilon * geometry.facets[ji - 1u].planeU;
        let prevA = (smsResidualAt(perturbed, geometry, media, ji, source, receiver) - r0) / epsilon;
        perturbed[ji - 1u] = vertices[ji - 1u] + epsilon * geometry.facets[ji - 1u].planeV;
        let prevB = (smsResidualAt(perturbed, geometry, media, ji, source, receiver) - r0) / epsilon;
        lower[ji] = mat2x2f(prevA, prevB);
      }
      upper[ji] = mat2x2f(vec2f(0.0), vec2f(0.0));
      if (ji + 1u < geometry.count) {
        perturbed = vertices;
        perturbed[ji + 1u] = vertices[ji + 1u] + epsilon * geometry.facets[ji + 1u].planeU;
        let nextA = (smsResidualAt(perturbed, geometry, media, ji, source, receiver) - r0) / epsilon;
        perturbed[ji + 1u] = vertices[ji + 1u] + epsilon * geometry.facets[ji + 1u].planeV;
        let nextB = (smsResidualAt(perturbed, geometry, media, ji, source, receiver) - r0) / epsilon;
        upper[ji] = mat2x2f(nextA, nextB);
      }
    }
    var cPrime: array<mat2x2f, 8>;
    var dPrime: array<vec2f, 8>;
    var singular = false;
    for (var fi = 0u; fi < SMS_MAX_VERTICES; fi = fi + 1u) {
      if (fi >= geometry.count) { break; }
      var denominator = diagonal[fi];
      var rhs = -residual[fi];
      if (fi > 0u) {
        denominator = denominator - lower[fi] * cPrime[fi - 1u];
        rhs = rhs - lower[fi] * dPrime[fi - 1u];
      }
      if (!smsMat2Invertible(denominator)) { singular = true; break; }
      let inverse = smsMat2Inverse(denominator);
      cPrime[fi] = mat2x2f(vec2f(0.0), vec2f(0.0));
      if (fi + 1u < geometry.count) {
        cPrime[fi] = inverse * upper[fi];
      }
      dPrime[fi] = inverse * rhs;
    }
    if (singular) { return out; }
    var delta: array<vec2f, 8>;
    var reverse = geometry.count;
    loop {
      if (reverse == 0u) { break; }
      reverse = reverse - 1u;
      delta[reverse] = dPrime[reverse];
      if (reverse + 1u < geometry.count) {
        delta[reverse] = delta[reverse] - cPrime[reverse] * delta[reverse + 1u];
      }
    }
    var step = 1.0;
    var accepted = false;
    for (var line = 0u; line < 10u; line = line + 1u) {
      var trial = vertices;
      for (var ti = 0u; ti < SMS_MAX_VERTICES; ti = ti + 1u) {
        if (ti >= geometry.count) { break; }
        trial[ti] = geometry.facets[ti].planePoint +
          (coordA[ti] + step * delta[ti].x) * geometry.facets[ti].planeU +
          (coordB[ti] + step * delta[ti].y) * geometry.facets[ti].planeV;
      }
      if (smsResidualMax(trial, geometry, media, source, receiver) < currentResidual) {
        for (var ai = 0u; ai < SMS_MAX_VERTICES; ai = ai + 1u) {
          if (ai >= geometry.count) { break; }
          coordA[ai] = coordA[ai] + step * delta[ai].x;
          coordB[ai] = coordB[ai] + step * delta[ai].y;
        }
        accepted = true;
        break;
      }
      step = step * 0.5;
    }
    if (!accepted) { return out; }
  }
  // A step accepted by the final permitted iteration must be re-evaluated.
  var finalVertices: array<vec3f, 8>;
  for (var finalIndex = 0u; finalIndex < SMS_MAX_VERTICES; finalIndex = finalIndex + 1u) {
    if (finalIndex >= geometry.count) { break; }
    finalVertices[finalIndex] = geometry.facets[finalIndex].planePoint +
      coordA[finalIndex] * geometry.facets[finalIndex].planeU +
      coordB[finalIndex] * geometry.facets[finalIndex].planeV;
  }
  out.vertices = finalVertices;
  out.residual = smsResidualMax(finalVertices, geometry, media, source, receiver);
  out.iterations = maxIterations;
  if (out.residual <= tolerance && smsSolutionInsideFacets(finalVertices, geometry)) {
    out.valid = 1u;
  }
  return out;
}

fn smsSameSolution(a: SmsChainResult, b: SmsChainResult, geometry: SmsChainGeometry, source: vec3f, receiver: vec3f) -> bool {
  if (a.valid == 0u || b.valid == 0u) { return false; }
  let tolerance = 8.0 * smsFiniteDifferenceStep(geometry, source, receiver);
  for (var i = 0u; i < SMS_MAX_VERTICES; i = i + 1u) {
    if (i >= geometry.count) { break; }
    if (distance(a.vertices[i], b.vertices[i]) > tolerance) { return false; }
  }
  return true;
}

fn smsFiniteFocusingDet(
  geometry: SmsChainGeometry,
  media: SmsChainMedia,
  solved: SmsChainResult,
  source: vec3f,
  receiver: vec3f,
  receiverU: vec3f,
  receiverV: vec3f,
) -> f32 {
  let epsilon = smsFiniteDifferenceStep(geometry, source, receiver);
  let solveU = smsSolveChain(geometry, media, source, receiver + receiverU * epsilon, solved.vertices);
  let solveV = smsSolveChain(geometry, media, source, receiver + receiverV * epsilon, solved.vertices);
  if (solveU.valid == 0u || solveV.valid == 0u) { return 0.0; }
  let base = safe_normalize(solved.vertices[0] - source);
  let derivativeU = (safe_normalize(solveU.vertices[0] - source) - base) / epsilon;
  let derivativeV = (safe_normalize(solveV.vertices[0] - source) - base) / epsilon;
  let determinant = length(cross(derivativeU, derivativeV));
  return select(0.0, determinant, determinant > 0.0 && determinant < INFINITY);
}

fn smsDirectionalFocusingDet(
  geometry: SmsChainGeometry,
  media: SmsChainMedia,
  solved: SmsChainResult,
  source: vec3f,
  receiver: vec3f,
  receiverU: vec3f,
  receiverV: vec3f,
) -> f32 {
  let epsilon = smsFiniteDifferenceStep(geometry, source, receiver);
  let solveU = smsSolveChain(geometry, media, source, receiver + receiverU * epsilon, solved.vertices);
  let solveV = smsSolveChain(geometry, media, source, receiver + receiverV * epsilon, solved.vertices);
  if (solveU.valid == 0u || solveV.valid == 0u) { return 0.0; }
  let derivativeU = (solveU.vertices[0] - solved.vertices[0]) / epsilon;
  let derivativeV = (solveV.vertices[0] - solved.vertices[0]) / epsilon;
  let determinant = abs(dot(cross(derivativeU, derivativeV), safe_normalize(geometry.sourceDirection)));
  return select(0.0, determinant, determinant > 0.0 && determinant < INFINITY);
}

fn smsAreaFocusingDet(
  geometry: SmsChainGeometry,
  media: SmsChainMedia,
  solved: SmsChainResult,
  source: vec3f,
  receiver: vec3f,
  lightU: vec3f,
  lightV: vec3f,
) -> f32 {
  let epsilon = smsFiniteDifferenceStep(geometry, source, receiver);
  let solveU = smsSolveChain(geometry, media, source + lightU * epsilon, receiver, solved.vertices);
  let solveV = smsSolveChain(geometry, media, source + lightV * epsilon, receiver, solved.vertices);
  if (solveU.valid == 0u || solveV.valid == 0u) { return 0.0; }
  let last = geometry.count - 1u;
  let base = safe_normalize(solved.vertices[last] - receiver);
  let derivativeU = (safe_normalize(solveU.vertices[last] - receiver) - base) / epsilon;
  let derivativeV = (safe_normalize(solveV.vertices[last] - receiver) - base) / epsilon;
  let determinant = length(cross(derivativeU, derivativeV));
  return select(0.0, determinant, determinant > 0.0 && determinant < INFINITY);
}
`;

export const MANIFOLD_SMS_SOLVER_MODULE: WgslModule = {
  name: 'manifoldSmsSolver',
  source: MANIFOLD_SMS_SOLVER_WGSL,
  requires: ['common', 'materialAtlas', 'surfaceTextures', 'ggxBrdf'],
};
