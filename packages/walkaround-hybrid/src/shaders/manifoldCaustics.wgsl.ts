import type { WgslModule } from '../pipeline/wgslComposer.js';
import { analyticLightFalloffWgsl } from './analyticLightFalloff.wgsl.js';

/**
 * Bounded sample-space-manifold-walk caustics for the realtime renderer.
 * Root multiplicity is deliberately a supported approximation: W_K=min(T,K)
 * has downward residual bias at finite K, exposed by mneeMultiplicityTrials.
 */
export const MANIFOLD_CAUSTICS_WGSL = /* wgsl */ `
const SMS_SOURCE_SUN = 0u;
const SMS_SOURCE_ANALYTIC = 1u;
const SMS_SOURCE_AREA = 2u;
const SMS_SOURCE_ENVIRONMENT = 3u;

${analyticLightFalloffWgsl('smsAnalytic')}

fn smsChannel(value: vec3f, channel: u32) -> f32 {
  if (channel == 0u) { return value.r; }
  if (channel == 1u) { return value.g; }
  return value.b;
}

fn smsPositiveFiniteRgb(value: vec3f) -> bool {
  return all(value >= vec3f(0.0)) &&
    max(value.r, max(value.g, value.b)) > 0.0 &&
    max(value.r, max(value.g, value.b)) < INFINITY;
}

struct SmsEndpoint {
  valid: u32,
  family: u32,
  sourceMode: u32,
  position: vec3f,
  towardLight: vec3f,
  radiance: vec3f,
  selectionPdf: f32,
  endpointPdf: f32,
  normal: vec3f,
  tangent: vec3f,
  bitangent: vec3f,
  xi: vec2f,
  area: f32,
  castShadowDisabled: u32,
  twoSided: u32,
  analyticDirection: vec3f,
  cosInner: f32,
  cosOuter: f32,
  cutoffDistance: f32,
  decay: f32,
};

fn smsInvalidEndpoint() -> SmsEndpoint {
  var endpoint: SmsEndpoint;
  endpoint.valid = 0u;
  endpoint.family = SMS_SOURCE_SUN;
  endpoint.sourceMode = 0u;
  endpoint.position = vec3f(0.0);
  endpoint.towardLight = vec3f(0.0, 1.0, 0.0);
  endpoint.radiance = vec3f(0.0);
  endpoint.selectionPdf = 0.0;
  endpoint.endpointPdf = 0.0;
  endpoint.normal = vec3f(0.0, -1.0, 0.0);
  endpoint.tangent = vec3f(1.0, 0.0, 0.0);
  endpoint.bitangent = vec3f(0.0, 0.0, 1.0);
  endpoint.xi = vec2f(0.0);
  endpoint.area = 0.0;
  endpoint.castShadowDisabled = 0u;
  endpoint.twoSided = 0u;
  endpoint.analyticDirection = vec3f(0.0);
  endpoint.cosInner = 1.0;
  endpoint.cosOuter = 0.0;
  endpoint.cutoffDistance = 0.0;
  endpoint.decay = 0.0;
  return endpoint;
}

struct SmsAnalyticLayout {
  valid: u32,
  count: u32,
  aliasOffset: u32,
  dims: vec2u,
};

fn smsAnalyticLayout() -> SmsAnalyticLayout {
  var out: SmsAnalyticLayout;
  out.valid = 0u;
  out.count = 0u;
  out.aliasOffset = 0u;
  out.dims = textureDimensions(analytic_lights);
  let header = textureLoad(analytic_lights, vec2i(0, 0), 0);
  if (!(header.x >= 0.0) || !(header.x < INFINITY) ||
      !(header.y >= 0.0) || !(header.y < INFINITY) ||
      header.x != floor(header.x) || header.y != floor(header.y)) { return out; }
  let count = u32(header.x);
  let aliasOffset = u32(header.y);
  let texelCount = out.dims.x * out.dims.y;
  if (count == 0u || aliasOffset != 1u + count * 4u ||
      aliasOffset + count > texelCount) { return out; }
  out.valid = 1u;
  out.count = count;
  out.aliasOffset = aliasOffset;
  return out;
}

struct SmsAliasDraw {
  valid: u32,
  index: u32,
  pmf: f32,
};

fn smsArenaAliasDraw(rng: ptr<function, u32>, count: u32) -> SmsAliasDraw {
  var out: SmsAliasDraw;
  out.valid = 0u;
  out.index = 0u;
  out.pmf = 0.0;
  if (count == 0u || sceneEmitterAliasCount() != count) { return out; }
  let column = smsUniformIndex(rng, count);
  let entry = sceneLoadEmitterAlias(column);
  let q = bitcast<f32>(entry.x);
  if (!(q >= 0.0 && q <= 1.0)) { return out; }
  let index = select(entry.y, column, rand_f32(rng) < q);
  if (index >= count) { return out; }
  let represented = sceneLoadEmitterAlias(index);
  let pmf = bitcast<f32>(represented.z);
  if (!(pmf > 0.0) || !(pmf < INFINITY)) { return out; }
  out.valid = 1u;
  out.index = index;
  out.pmf = pmf;
  return out;
}

fn smsAnalyticAliasDraw(
  rng: ptr<function, u32>,
  analyticLayout: SmsAnalyticLayout,
) -> SmsAliasDraw {
  var out: SmsAliasDraw;
  out.valid = 0u;
  out.index = 0u;
  out.pmf = 0.0;
  if (analyticLayout.valid == 0u) { return out; }
  let column = smsUniformIndex(rng, analyticLayout.count);
  let coord = analyticLayout.aliasOffset + column;
  let entry = textureLoad(
    analytic_lights,
    vec2i(i32(coord % analyticLayout.dims.x), i32(coord / analyticLayout.dims.x)), 0,
  );
  if (!(entry.x >= 0.0 && entry.x <= 1.0)) { return out; }
  let aliasIndex = bitcast<u32>(entry.y);
  if (aliasIndex >= analyticLayout.count) { return out; }
  let index = select(aliasIndex, column, rand_f32(rng) < entry.x);
  if (index >= analyticLayout.count) { return out; }
  let selectedCoord = analyticLayout.aliasOffset + index;
  let selected = textureLoad(
    analytic_lights,
    vec2i(i32(selectedCoord % analyticLayout.dims.x), i32(selectedCoord / analyticLayout.dims.x)), 0,
  );
  if (!(selected.z > 0.0) || !(selected.z < INFINITY)) { return out; }
  out.valid = 1u;
  out.index = index;
  out.pmf = selected.z;
  return out;
}

fn smsSunAvailable() -> bool {
  return ubo.sunIntensity > 0.0 && ubo.sunIntensity < INFINITY &&
    dot(ubo.sunDirection, ubo.sunDirection) > 0.0 &&
    ubo.sunAngular.x >= 0.0 && ubo.sunAngular.x <= 1.5707963267948966;
}

fn smsEnvironmentAvailable() -> bool {
  if (envHasMap()) {
    return envParams.intensity > 0.0 && envParams.intensity < INFINITY;
  }
  return smsPositiveFiniteRgb(walkaroundScaleEnvironmentRadiance(
    ubo.skyTint,
    ubo.skyIrradiance,
  ));
}

fn smsEndpointFamilyCount(analyticLayout: SmsAnalyticLayout) -> u32 {
  var count = 0u;
  if (smsSunAvailable()) { count = count + 1u; }
  if (analyticLayout.valid != 0u) { count = count + 1u; }
  if (sceneEmitterCount() > 0u && sceneEmitterAliasCount() == sceneEmitterCount()) {
    count = count + 1u;
  }
  if (smsEnvironmentAvailable()) { count = count + 1u; }
  return count;
}

fn smsFamilyAtOrdinal(ordinal: u32, analyticLayout: SmsAnalyticLayout) -> u32 {
  var cursor = 0u;
  if (smsSunAvailable()) {
    if (cursor == ordinal) { return SMS_SOURCE_SUN; }
    cursor = cursor + 1u;
  }
  if (analyticLayout.valid != 0u) {
    if (cursor == ordinal) { return SMS_SOURCE_ANALYTIC; }
    cursor = cursor + 1u;
  }
  if (sceneEmitterCount() > 0u && sceneEmitterAliasCount() == sceneEmitterCount()) {
    if (cursor == ordinal) { return SMS_SOURCE_AREA; }
    cursor = cursor + 1u;
  }
  return SMS_SOURCE_ENVIRONMENT;
}

fn smsSampleSunEndpoint(
  rng: ptr<function, u32>,
  familyPdf: f32,
  receiver: vec3f,
) -> SmsEndpoint {
  var out = smsInvalidEndpoint();
  if (!smsSunAvailable()) { return out; }
  let axis = safe_normalize(ubo.sunDirection);
  let radius = ubo.sunAngular.x;
  var direction = axis;
  var directionalPdf = 1.0;
  var radiance = vec3f(ubo.sunIntensity);
  if (radius > 0.0) {
    var tangent: vec3f;
    var bitangent: vec3f;
    smsBuildFrame(axis, &tangent, &bitangent);
    let xi = rand2(rng);
    let cosTheta = mix(1.0, cos(radius), xi.x);
    let sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
    let phi = 2.0 * PI * xi.y;
    direction = safe_normalize(
      axis * cosTheta + tangent * (sinTheta * cos(phi)) +
      bitangent * (sinTheta * sin(phi)),
    );
    let solidAngle = 2.0 * PI * (1.0 - cos(radius));
    directionalPdf = 1.0 / solidAngle;
    radiance = vec3f(ubo.sunIntensity / solidAngle);
  }
  out.valid = 1u;
  out.family = SMS_SOURCE_SUN;
  out.sourceMode = 1u;
  out.position = receiver + direction;
  out.towardLight = direction;
  out.radiance = radiance;
  out.selectionPdf = familyPdf;
  out.endpointPdf = directionalPdf;
  out.castShadowDisabled = select(
    0u, 1u,
    (ubo.stainedGlassFlags & SHADE_FLAG_DIRECT_SUN_SHADOW_DISABLED) != 0u,
  );
  return out;
}

fn smsSampleAnalyticEndpoint(
  rng: ptr<function, u32>,
  familyPdf: f32,
  analyticLayout: SmsAnalyticLayout,
) -> SmsEndpoint {
  var out = smsInvalidEndpoint();
  let draw = smsAnalyticAliasDraw(rng, analyticLayout);
  if (draw.valid == 0u) { return out; }
  let base = 1u + draw.index * 4u;
  let light0 = textureLoad(analytic_lights, vec2i(i32(base % analyticLayout.dims.x), i32(base / analyticLayout.dims.x)), 0);
  let light1 = textureLoad(analytic_lights, vec2i(i32((base + 1u) % analyticLayout.dims.x), i32((base + 1u) / analyticLayout.dims.x)), 0);
  let light2 = textureLoad(analytic_lights, vec2i(i32((base + 2u) % analyticLayout.dims.x), i32((base + 2u) / analyticLayout.dims.x)), 0);
  let light3 = textureLoad(analytic_lights, vec2i(i32((base + 3u) % analyticLayout.dims.x), i32((base + 3u) / analyticLayout.dims.x)), 0);
  if (!smsPositiveFiniteRgb(light1.xyz) || !all(abs(light0.xyz) < vec3f(INFINITY))) { return out; }
  out.valid = 1u;
  out.family = SMS_SOURCE_ANALYTIC;
  out.position = light0.xyz;
  out.radiance = light1.xyz;
  out.selectionPdf = familyPdf * draw.pmf;
  out.endpointPdf = 1.0;
  out.castShadowDisabled = select(0u, 1u, light3.y > 0.5);
  out.analyticDirection = light2.xyz;
  out.cosInner = light2.w;
  out.cosOuter = light3.x;
  out.cutoffDistance = light3.z;
  out.decay = light3.w;
  return out;
}

fn smsSampleAreaEndpoint(
  rng: ptr<function, u32>,
  familyPdf: f32,
) -> SmsEndpoint {
  var out = smsInvalidEndpoint();
  let count = sceneEmitterCount();
  let draw = smsArenaAliasDraw(rng, count);
  if (draw.valid == 0u) { return out; }
  let emitter = sceneLoadEmitter(draw.index);
  if (!(emitter.area > 0.0) || !(emitter.area < INFINITY)) { return out; }
  let xi = rand2(rng);
  let sample = sampleEmitterPoint(emitter, xi);
  let radiance = sampleEmitterLeAtXi(emitter, xi);
  if (!smsPositiveFiniteRgb(radiance)) { return out; }
  var tangent: vec3f;
  var bitangent: vec3f;
  smsBuildFrame(safe_normalize(emitter.normal), &tangent, &bitangent);
  out.valid = 1u;
  out.family = SMS_SOURCE_AREA;
  out.position = sample.pos;
  out.radiance = radiance;
  out.selectionPdf = familyPdf * draw.pmf;
  out.endpointPdf = sample.pdfArea;
  out.normal = safe_normalize(emitter.normal);
  out.tangent = tangent;
  out.bitangent = bitangent;
  out.xi = xi;
  out.area = sample.area;
  out.castShadowDisabled = select(
    0u, 1u, emitterTriCastShadowDisabled(emitter),
  );
  out.twoSided = select(0u, 1u, emitterTriIsTwoSided(emitter));
  return out;
}

fn smsSampleEnvironmentEndpoint(
  rng: ptr<function, u32>,
  familyPdf: f32,
  receiver: vec3f,
) -> SmsEndpoint {
  var out = smsInvalidEndpoint();
  var direction: vec3f;
  var radiance: vec3f;
  var directionalPdf: f32;
  if (envHasMap()) {
    let sample = envImportanceSample(rng);
    direction = sample.dir;
    radiance = sample.color;
    directionalPdf = sample.pdf;
  } else {
    let xi = rand2(rng);
    let y = 1.0 - 2.0 * xi.x;
    let radius = sqrt(max(0.0, 1.0 - y * y));
    let phi = 2.0 * PI * xi.y;
    direction = vec3f(radius * cos(phi), y, radius * sin(phi));
    radiance = walkaroundScaleEnvironmentRadiance(
      ubo.skyTint,
      ubo.skyIrradiance,
    );
    directionalPdf = 1.0 / (4.0 * PI);
  }
  if (!smsPositiveFiniteRgb(radiance) ||
      !(directionalPdf > 0.0) || !(directionalPdf < INFINITY)) { return out; }
  out.valid = 1u;
  out.family = SMS_SOURCE_ENVIRONMENT;
  out.sourceMode = 1u;
  out.position = receiver + direction;
  out.towardLight = safe_normalize(direction);
  out.radiance = radiance;
  out.selectionPdf = familyPdf;
  out.endpointPdf = directionalPdf;
  return out;
}

fn smsSampleEndpoint(
  rng: ptr<function, u32>,
  receiver: vec3f,
) -> SmsEndpoint {
  let analyticLayout = smsAnalyticLayout();
  let familyCount = smsEndpointFamilyCount(analyticLayout);
  if (familyCount == 0u) { return smsInvalidEndpoint(); }
  let familyPdf = 1.0 / f32(familyCount);
  let family = smsFamilyAtOrdinal(smsUniformIndex(rng, familyCount), analyticLayout);
  if (family == SMS_SOURCE_SUN) {
    return smsSampleSunEndpoint(rng, familyPdf, receiver);
  }
  if (family == SMS_SOURCE_ANALYTIC) {
    return smsSampleAnalyticEndpoint(rng, familyPdf, analyticLayout);
  }
  if (family == SMS_SOURCE_AREA) {
    return smsSampleAreaEndpoint(rng, familyPdf);
  }
  return smsSampleEnvironmentEndpoint(rng, familyPdf, receiver);
}

struct SmsMediaBuild {
  media: SmsChainMedia,
  valid: u32,
};

fn smsEmptyContainingMedia() -> MaterialShadowContainingMedia {
  var out: MaterialShadowContainingMedia;
  out.valid = 1u;
  out.state = materialShadowEmptyMediumState();
  return out;
}

fn smsSourceIncomingDirection(
  endpoint: SmsEndpoint,
  firstVertex: vec3f,
) -> vec3f {
  if (endpoint.sourceMode == 1u) {
    return -safe_normalize(endpoint.towardLight);
  }
  return safe_normalize(firstVertex - endpoint.position);
}

fn smsSourceContainingMedia(
  endpoint: SmsEndpoint,
  firstVertex: vec3f,
) -> MaterialShadowContainingMedia {
  if (endpoint.sourceMode == 1u) {
    return smsEmptyContainingMedia();
  }
  return materialShadowClassifyContainingMedia(
    ubo.bvhMode,
    ubo.tlasNodeCount,
    endpoint.position,
    smsSourceIncomingDirection(endpoint, firstVertex),
    ubo.triIntersectEpsilon,
    bvh_material,
    BVH_MATERIAL_TEX_WIDTH,
    bvh_beer,
  );
}

fn smsReceiverContainingMedia(
  receiver: vec3f,
  lastVertex: vec3f,
) -> MaterialShadowContainingMedia {
  return materialShadowClassifyContainingMedia(
    ubo.bvhMode,
    ubo.tlasNodeCount,
    receiver,
    safe_normalize(receiver - lastVertex),
    ubo.triIntersectEpsilon,
    bvh_material,
    BVH_MATERIAL_TEX_WIDTH,
    bvh_beer,
  );
}

fn smsTriangleIorChannel(triIndex: u32, channel: u32) -> f32 {
  let coord = vec2u(
    triIndex % BVH_MATERIAL_TEX_WIDTH,
    triIndex / BVH_MATERIAL_TEX_WIDTH,
  );
  let materialWord = textureLoad(bvh_material, vec2i(coord), 0).r;
  return smsChannel(
    materialDispersionIorRgb(triIndex, decodeIor(materialWord)),
    channel,
  );
}

fn smsBuildMedia(
  geometry: SmsChainGeometry,
  endpoint: SmsEndpoint,
  seeds: array<vec3f, 8>,
  receiver: vec3f,
  channel: u32,
) -> SmsMediaBuild {
  var out: SmsMediaBuild;
  out.valid = 0u;
  if (geometry.count == 0u) { return out; }
  let sourceContaining = smsSourceContainingMedia(endpoint, seeds[0]);
  let receiverContaining = smsReceiverContainingMedia(
    receiver, seeds[geometry.count - 1u],
  );
  if (sourceContaining.valid == 0u || receiverContaining.valid == 0u) {
    return out;
  }
  var stackBoundaryId: array<u32, 16>;
  var stackRepresentedId: array<u32, 16>;
  var stackIor: array<f32, 16>;
  var depth = sourceContaining.state.depth;
  if (depth > MATERIAL_SHADOW_MEDIUM_CAPACITY) { return out; }
  for (var seed = 0u; seed < MATERIAL_SHADOW_MEDIUM_CAPACITY; seed += 1u) {
    if (seed >= depth) { break; }
    let ior = smsTriangleIorChannel(
      sourceContaining.state.tri[seed], channel,
    );
    if (!(ior > 0.0) || !(ior < INFINITY)) { return out; }
    stackBoundaryId[seed] = sourceContaining.state.materialId[seed];
    stackRepresentedId[seed] = sourceContaining.state.instance[seed];
    stackIor[seed] = ior;
  }
  var previous = endpoint.position;
  for (var i = 0u; i < SMS_MAX_VERTICES; i = i + 1u) {
    if (i >= geometry.count) { break; }
    var incoming = safe_normalize(seeds[i] - previous);
    if (i == 0u && endpoint.sourceMode == 1u) {
      incoming = -endpoint.towardLight;
    }
    let optics = smsFacetOpticsAt(geometry.facets[i], seeds[i], incoming);
    if (optics.frame.valid == 0u) { return out; }
    let materialIor = smsChannel(optics.iorRgb, channel);
    if (!(materialIor > 0.0) || !(materialIor < INFINITY)) { return out; }
    let entering = dot(incoming, geometry.facets[i].geometricNormal) < 0.0;
    var currentIor = 1.0;
    if (depth > 0u) { currentIor = stackIor[depth - 1u]; }
    out.media.etaI[i] = currentIor;
    var boundaryTargetIor = materialIor;
    if (optics.bulkMedium != 0u && !entering) {
      if (
        depth == 0u ||
        stackBoundaryId[depth - 1u] !=
          geometry.facets[i].encodedBoundaryId ||
        stackRepresentedId[depth - 1u] !=
          geometry.facets[i].representedPrimitiveInstanceId
      ) { return out; }
      boundaryTargetIor = 1.0;
      if (depth > 1u) { boundaryTargetIor = stackIor[depth - 2u]; }
    }
    out.media.etaT[i] = boundaryTargetIor;
    // Reflection observes the same physical boundary IORs but does not mutate
    // the medium stack. The residual itself is eta-independent for reflection.
    if (geometry.events[i] == SMS_EVENT_REFLECTION) {
      previous = seeds[i];
      continue;
    }
    if (optics.bulkMedium != 0u) {
      if (entering) {
        if (depth >= MATERIAL_SHADOW_MEDIUM_CAPACITY) { return out; }
        stackBoundaryId[depth] = geometry.facets[i].encodedBoundaryId;
        stackRepresentedId[depth] =
          geometry.facets[i].representedPrimitiveInstanceId;
        stackIor[depth] = materialIor;
        depth = depth + 1u;
      } else {
        depth = depth - 1u;
      }
    }
    let frame = optics.frame;
    let wo = -incoming;
    let wm = safe_normalize(
      frame.tangent * geometry.offsets[i].local.x +
      frame.bitangent * geometry.offsets[i].local.y +
      frame.normal * geometry.offsets[i].local.z,
    );
    let cosIncident = dot(wo, wm);
    if (!(cosIncident > 0.0)) { return out; }
    let eta = out.media.etaI[i] / out.media.etaT[i];
    if (eta * eta * (1.0 - cosIncident * cosIncident) >= 1.0) { return out; }
    previous = seeds[i];
  }
  if (depth != receiverContaining.state.depth) { return out; }
  for (var destination = 0u; destination < MATERIAL_SHADOW_MEDIUM_CAPACITY; destination += 1u) {
    if (destination >= depth) { break; }
    if (
      stackBoundaryId[destination] !=
        receiverContaining.state.materialId[destination] ||
      stackRepresentedId[destination] !=
        receiverContaining.state.instance[destination]
    ) { return out; }
  }
  out.valid = 1u;
  return out;
}

fn smsSeedOnFacet(facet: SmsFacet, rng: ptr<function, u32>) -> vec3f {
  let xi = rand2(rng);
  let s = sqrt(xi.x);
  let bary = vec3f(1.0 - s, s * (1.0 - xi.y), s * xi.y);
  return bary.x * facet.a + bary.y * facet.b + bary.z * facet.c;
}

fn smsMultiplicitySeed(
  frameSeed: u32,
  pixelIndex: u32,
  channel: u32,
  proposal: u32,
  oneBasedTrial: u32,
) -> u32 {
  var state = frameSeed ^
    (pixelIndex * 0x9e3779b9u) ^
    (channel * 0x85ebca6bu) ^
    (proposal * 0xc2b2ae35u) ^
    (oneBasedTrial * 0x27d4eb2du) ^
    0x534d534du;
  return pcgNext(&state);
}

fn smsMaterialMapAbsent(triIndex: u32, metaOffset: u32) -> bool {
  let mapDescriptor = textureLoad(
    baseColorMapMeta,
    baseColorMapMetaCoord(triIndex, metaOffset),
    0,
  );
  return i32(mapDescriptor.x) < 0;
}

// Exact W=1 bypass only for a proven restricted event family. Failure to
// observe another root is never evidence of uniqueness.
fn smsProvesUniquePlanarDeltaTransmission(
  geometry: SmsChainGeometry,
  media: SmsChainMedia,
  endpoint: SmsEndpoint,
  receiver: vec3f,
  solved: SmsChainResult,
) -> bool {
  if (geometry.count != 1u || geometry.events[0] != SMS_EVENT_TRANSMISSION ||
      geometry.offsets[0].roughness != 0.0 || ubo.bvhMode == 1u) { return false; }
  let facet = geometry.facets[0];
  if (!smsMaterialMapAbsent(facet.triIndex, MATERIAL_MAP_NORMAL_TEXEL_OFFSET) ||
      !smsMaterialMapAbsent(facet.triIndex, MATERIAL_MAP_BUMP_TEXEL_OFFSET) ||
      !smsMaterialMapAbsent(facet.triIndex, MATERIAL_MAP_FRONT_LAYER_NORMAL_TEXEL_OFFSET) ||
      !smsMaterialMapAbsent(facet.triIndex, MATERIAL_MAP_BACK_LAYER_NORMAL_TEXEL_OFFSET)) {
    return false;
  }
  let n0 = sceneLoadBvhNormal(facet.indices.x).xyz;
  let n1 = sceneLoadBvhNormal(facet.indices.y).xyz;
  let n2 = sceneLoadBvhNormal(facet.indices.z).xyz;
  if (!all(n0 == n1) || !all(n1 == n2) ||
      !all(cross(n0, facet.geometricNormal) == vec3f(0.0))) { return false; }
  let sourceDistance = select(
    dot(endpoint.position - facet.planePoint, facet.geometricNormal),
    dot(endpoint.towardLight, facet.geometricNormal),
    endpoint.sourceMode == 1u,
  );
  let receiverDistance = dot(receiver - facet.planePoint, facet.geometricNormal);
  if (sourceDistance == 0.0 || receiverDistance == 0.0 ||
      sourceDistance * receiverDistance >= 0.0) { return false; }
  var incoming = safe_normalize(solved.vertices[0] - endpoint.position);
  if (endpoint.sourceMode == 1u) { incoming = -endpoint.towardLight; }
  let optics = smsFacetOpticsAt(facet, solved.vertices[0], incoming);
  // A one-vertex bulk chain cannot both enter and leave a closed component.
  // The structural W=1 proof is therefore restricted to the represented
  // reciprocal thin-sheet event (encoded boundary zero).
  if (optics.frame.valid == 0u || optics.bulkMedium != 0u) { return false; }
  let cosIncident = dot(-incoming, optics.frame.normal);
  if (!(cosIncident > 0.0) || !(media.etaI[0] > 0.0) || !(media.etaT[0] > 0.0)) {
    return false;
  }
  let eta = media.etaI[0] / media.etaT[0];
  return eta * eta * (1.0 - cosIncident * cosIncident) < 1.0;
}

fn smsBoundedMultiplicityEstimate(
  geometry: SmsChainGeometry,
  media: SmsChainMedia,
  endpoint: SmsEndpoint,
  receiver: vec3f,
  primary: SmsChainResult,
  pixelIndex: u32,
  channel: u32,
) -> f32 {
  if (smsProvesUniquePlanarDeltaTransmission(geometry, media, endpoint, receiver, primary)) {
    return 1.0;
  }
  let cap = smsConfiguredMultiplicityTrials();
  for (var trial = 1u; trial <= 32u; trial = trial + 1u) {
    if (trial > cap) { break; }
    var recurrenceRng = smsMultiplicitySeed(
      ubo.frameSeed, pixelIndex, channel, 0u, trial,
    );
    var seeds: array<vec3f, 8>;
    for (var i = 0u; i < SMS_MAX_VERTICES; i = i + 1u) {
      if (i >= geometry.count) { break; }
      seeds[i] = smsSeedOnFacet(geometry.facets[i], &recurrenceRng);
    }
    let recurrence = smsSolveChain(
      geometry, media, endpoint.position, receiver, seeds,
    );
    if (smsSameSolution(primary, recurrence, geometry, endpoint.position, receiver)) {
      return f32(trial);
    }
  }
  return f32(cap);
}

struct SmsSegmentVisibility {
  reachesFacet: u32,
  alphaTransmittance: f32,
};

fn smsFacetSourceFeature(
  facet: SmsFacet,
  incomingOrigin: vec3f,
  sourcePosition: vec3f,
) -> OpticalSourceFeature {
  let delta = sourcePosition - incomingOrigin;
  let distanceToSource = safe_length(delta);
  if (!(distanceToSource > 0.0) || !(distanceToSource < INFINITY)) {
    return opticalSourceFeatureInvalid();
  }
  let exact = opticalWatertightTriangleIntersect(
    incomingOrigin,
    safe_normalize(delta),
    facet.a,
    facet.b,
    facet.c,
    0.0,
  );
  if (!exact.hit) { return opticalSourceFeatureInvalid(); }
  return opticalCreateSourceFeature(
    facet.encodedBoundaryId,
    facet.representedPrimitiveInstanceId,
    facet.triIndex,
    exact.zeroEdgeMask,
    facet.a,
    facet.b,
    facet.c,
  );
}

// The receiver is an ordinary diffuse/opaque launch, so its own G-buffer
// surface uses the ordinary geometric offset. Once a manifold interface has
// accepted the path, the sibling fixed-origin walker below takes over.
fn smsTraceReceiverReachesFacet(
  origin: vec3f,
  targetPosition: vec3f,
  facet: SmsFacet,
) -> SmsSegmentVisibility {
  var out: SmsSegmentVisibility;
  out.reachesFacet = 0u;
  out.alphaTransmittance = 1.0;
  let delta = targetPosition - origin;
  let distanceToTarget = safe_length(delta);
  if (!(distanceToTarget > 0.0) || !(distanceToTarget < INFINITY)) { return out; }
  let direction = safe_normalize(delta);
  let step = walkaroundRayOriginBias();
  var traveled = step;
  var ray = Ray();
  ray.origin = origin + direction * traveled;
  ray.direction = direction;
  let tolerance = max(
    8.0 * walkaroundRayOriginBias(),
    32.0 * SMS_F32_EPSILON * distanceToTarget,
  );
  let surfaceBudget = materialShadowWorldSurfaceBudget(
    ubo.bvhMode, ubo.tlasNodeCount,
  );
  for (var layer = 0u; layer < surfaceBudget; layer = layer + 1u) {
    let hit = traceSceneFirstHit(
      ubo.bvhMode, ubo.tlasNodeCount, ray, ubo.triIntersectEpsilon,
    );
    if (!hit.didHit) { return out; }
    let totalDistance = traveled + hit.dist;
    let sameFacet = hit.indices.w == facet.triIndex &&
      (ubo.bvhMode != 1u || hit.instanceIndex == facet.instanceIndex);
    if (sameFacet && abs(totalDistance - distanceToTarget) <= tolerance) {
      out.reachesFacet = 1u;
      return out;
    }
    if (totalDistance >= distanceToTarget - tolerance) { return out; }
    let word = textureLoad(
      bvh_material,
      vec2i(
        i32(hit.indices.w % BVH_MATERIAL_TEX_WIDTH),
        i32(hit.indices.w / BVH_MATERIAL_TEX_WIDTH),
      ),
      0,
    ).r;
    // Non-target refractive coverage is not part of this frozen topology.
    // skipGlass=false makes its covered fraction opaque, while castShadow:
    // false, mask holes, and alpha-blend uncovered coverage pass exactly once.
    let alphaT = materialShadowTransmittanceForHit(hit, word, false);
    if (!(alphaT > 0.0)) { return out; }
    out.alphaTransmittance = out.alphaTransmittance * alphaT;
    if (!(out.alphaTransmittance > 0.0)) { return out; }
    traveled = totalDistance + step;
    if (traveled >= distanceToTarget - tolerance) { return out; }
    ray.origin = origin + direction * traveled;
  }
  return out;
}

// Accepted specular interfaces remain at the exact represented point. The
// crossed face/edge/vertex fan is excluded by its exact watertight source
// feature, while exclusive-minT advances on the original ray without an
// origin step that could jump a nearby boundary.
fn smsTraceOpticalSourceReachesFacet(
  origin: vec3f,
  targetPosition: vec3f,
  facet: SmsFacet,
  sourceFeature: OpticalSourceFeature,
) -> SmsSegmentVisibility {
  var out: SmsSegmentVisibility;
  out.reachesFacet = 0u;
  out.alphaTransmittance = 1.0;
  let delta = targetPosition - origin;
  let distanceToTarget = safe_length(delta);
  if (!(distanceToTarget > 0.0) || !(distanceToTarget < INFINITY) ||
      sourceFeature.kind == OPTICAL_SOURCE_FEATURE_INVALID) { return out; }
  let ray = Ray(origin, safe_normalize(delta));
  var exclusiveMinT = 0.0;
  let tolerance = max(
    8.0 * walkaroundRayOriginBias(),
    32.0 * SMS_F32_EPSILON * distanceToTarget,
  );
  let surfaceBudget = materialShadowWorldSurfaceBudget(
    ubo.bvhMode, ubo.tlasNodeCount,
  );
  for (var layer = 0u; layer < surfaceBudget; layer = layer + 1u) {
    let traced = traceSceneFirstHitWithOpticalSourceExclusion(
      ubo.bvhMode,
      ubo.tlasNodeCount,
      ray,
      exclusiveMinT,
      sourceFeature,
    );
    if (traced.valid == 0u) { return out; }
    let hit = traced.hit;
    if (!hit.didHit) { return out; }
    let sameFacet = hit.indices.w == facet.triIndex &&
      (ubo.bvhMode != 1u || hit.instanceIndex == facet.instanceIndex);
    if (sameFacet && abs(hit.dist - distanceToTarget) <= tolerance) {
      out.reachesFacet = 1u;
      return out;
    }
    if (!(hit.dist > exclusiveMinT) ||
        hit.dist >= distanceToTarget - tolerance) { return out; }
    let word = textureLoad(
      bvh_material,
      vec2i(
        i32(hit.indices.w % BVH_MATERIAL_TEX_WIDTH),
        i32(hit.indices.w / BVH_MATERIAL_TEX_WIDTH),
      ),
      0,
    ).r;
    let alphaT = materialShadowTransmittanceForHit(hit, word, false);
    if (!(alphaT > 0.0)) { return out; }
    out.alphaTransmittance = out.alphaTransmittance * alphaT;
    if (!(out.alphaTransmittance > 0.0)) { return out; }
    exclusiveMinT = hit.dist;
  }
  return out;
}

fn smsChainIdentityVisible(
  geometry: SmsChainGeometry,
  solved: SmsChainResult,
  receiver: vec3f,
) -> f32 {
  var alphaTransmittance = 1.0;
  let last = geometry.count - 1u;
  let receiverSegment = smsTraceReceiverReachesFacet(
    receiver, solved.vertices[last], geometry.facets[last],
  );
  if (receiverSegment.reachesFacet == 0u) {
    return 0.0;
  }
  alphaTransmittance = alphaTransmittance * receiverSegment.alphaTransmittance;
  var reverse = last;
  loop {
    if (reverse == 0u) { break; }
    let previousVertexIndex = reverse - 1u;
    let sourceFeature = smsFacetSourceFeature(
      geometry.facets[reverse],
      solved.vertices[previousVertexIndex],
      solved.vertices[reverse],
    );
    if (sourceFeature.kind == OPTICAL_SOURCE_FEATURE_INVALID) { return 0.0; }
    let segment = smsTraceOpticalSourceReachesFacet(
      solved.vertices[reverse], solved.vertices[previousVertexIndex],
      geometry.facets[previousVertexIndex], sourceFeature,
    );
    if (segment.reachesFacet == 0u) { return 0.0; }
    alphaTransmittance = alphaTransmittance * segment.alphaTransmittance;
    reverse = previousVertexIndex;
  }
  return alphaTransmittance;
}

fn smsEndpointSourceFeature(
  endpoint: SmsEndpoint,
  firstVertex: vec3f,
  firstFacet: SmsFacet,
) -> OpticalSourceFeature {
  var incomingOrigin = endpoint.position;
  if (endpoint.sourceMode == 1u) {
    incomingOrigin = firstVertex + safe_normalize(endpoint.towardLight);
  }
  return smsFacetSourceFeature(firstFacet, incomingOrigin, firstVertex);
}

fn smsTraceExternalAlphaFromOpticalSource(
  origin: vec3f,
  direction: vec3f,
  tMax: f32,
  sourceFeature: OpticalSourceFeature,
) -> f32 {
  if (sourceFeature.kind == OPTICAL_SOURCE_FEATURE_INVALID ||
      tMax != tMax || tMax < 0.0) { return 0.0; }
  let ray = Ray(origin, direction);
  var exclusiveMinT = 0.0;
  var transmittance = 1.0;
  let surfaceBudget = materialShadowWorldSurfaceBudget(
    ubo.bvhMode, ubo.tlasNodeCount,
  );
  for (var layer = 0u; layer < surfaceBudget; layer = layer + 1u) {
    let traced = traceSceneFirstHitWithOpticalSourceExclusion(
      ubo.bvhMode,
      ubo.tlasNodeCount,
      ray,
      exclusiveMinT,
      sourceFeature,
    );
    if (traced.valid == 0u) { return 0.0; }
    let hit = traced.hit;
    if (!hit.didHit || hit.dist >= tMax) {
      return clamp(transmittance, 0.0, 1.0);
    }
    if (!(hit.dist > exclusiveMinT)) { return 0.0; }
    let word = textureLoad(
      bvh_material,
      vec2i(
        i32(hit.indices.w % BVH_MATERIAL_TEX_WIDTH),
        i32(hit.indices.w / BVH_MATERIAL_TEX_WIDTH),
      ),
      0,
    ).r;
    // Every covered material-transmission boundary belongs to the frozen SMS
    // topology. Unselected covered glass is therefore opaque here; only
    // castShadow:false, alpha holes, and represented blend coverage pass.
    let alphaT = materialShadowTransmittanceForHit(hit, word, false);
    if (!(alphaT > 0.0)) { return 0.0; }
    transmittance = transmittance * alphaT;
    if (!(transmittance > 0.0)) { return 0.0; }
    exclusiveMinT = hit.dist;
  }
  return 0.0;
}

fn smsEndpointTransport(
  endpoint: SmsEndpoint,
  firstVertex: vec3f,
  firstFacet: SmsFacet,
) -> vec3f {
  var direction: vec3f;
  var distanceToSource = INFINITY;
  var distanceSquaredToSource = 1.0;
  var scalar = 1.0;
  if (endpoint.sourceMode == 1u) {
    direction = endpoint.towardLight;
  } else {
    let delta = endpoint.position - firstVertex;
    distanceToSource = safe_length(delta);
    if (!(distanceToSource > 0.0) ||
        !(distanceToSource < INFINITY)) { return vec3f(0.0); }
    distanceSquaredToSource = distanceToSource * distanceToSource;
    direction = safe_normalize(delta);
  }
  if (endpoint.family == SMS_SOURCE_ANALYTIC) {
    // The generalized finite-endpoint determinant already carries the
    // physical inverse-square geometry. Remove that factor from the authored
    // point/spot attenuation here (as direct-emitter sampling does) so it is
    // not applied twice; non-quadratic decay/range controls remain as a ratio.
    scalar = smsAnalyticSpotConeFalloff(
      endpoint.analyticDirection, direction, endpoint.cosInner, endpoint.cosOuter,
    ) * smsAnalyticPointSpotAttenuation(
      distanceToSource, endpoint.cutoffDistance, endpoint.decay, ubo.emitterDist2Floor,
    ) * distanceSquaredToSource;
  } else if (endpoint.family == SMS_SOURCE_AREA) {
    let signedCosine = dot(endpoint.normal, -direction);
    let emitterCosine = select(
      max(signedCosine, 0.0),
      abs(signedCosine),
      endpoint.twoSided != 0u,
    );
    if (emitterCosine <= 0.0) { return vec3f(0.0); }
  }
  if (!(scalar > 0.0) || !(scalar < INFINITY)) { return vec3f(0.0); }
  var visibility = vec3f(1.0);
  if (endpoint.castShadowDisabled == 0u) {
    let maximum = select(
      max(0.0, distanceToSource - walkaroundRayEndMargin()),
      INFINITY,
      endpoint.sourceMode == 1u,
    );
    let sourceFeature = smsEndpointSourceFeature(
      endpoint, firstVertex, firstFacet,
    );
    visibility = vec3f(smsTraceExternalAlphaFromOpticalSource(
      firstVertex,
      direction,
      maximum,
      sourceFeature,
    ));
  }
  return endpoint.radiance * scalar * visibility;
}

fn smsBeerTint(optics: SmsFacetOptics) -> vec3f {
  return materialShadowAuthoredBeerTint(optics.frame.hit, bvh_beer);
}

fn smsThicknessMapScale(optics: SmsFacetOptics) -> f32 {
  return materialShadowThicknessMapScale(optics.frame.hit);
}

fn smsInterfaceFactor(
  geometry: SmsChainGeometry,
  media: SmsChainMedia,
  index: u32,
  vertex: vec3f,
  previous: vec3f,
  next: vec3f,
  endpoint: SmsEndpoint,
  channel: u32,
  payMaterialTransmission: bool,
) -> f32 {
  var incoming = safe_normalize(vertex - previous);
  if (index == 0u && endpoint.sourceMode == 1u) { incoming = -endpoint.towardLight; }
  let towardNext = safe_normalize(next - vertex);
  let optics = smsFacetOpticsAt(geometry.facets[index], vertex, incoming);
  if (optics.frame.valid == 0u) { return 0.0; }
  let n = optics.frame.normal;
  let wo = -incoming;
  let wi = towardNext;
  let wm = safe_normalize(
    optics.frame.tangent * geometry.offsets[index].local.x +
    optics.frame.bitangent * geometry.offsets[index].local.y +
    n * geometry.offsets[index].local.z,
  );
  let nDotWo = dot(n, wo);
  let woDotM = dot(wo, wm);
  if (!(nDotWo > 0.0) || !(woDotM > 0.0)) { return 0.0; }
  let film = materialThinFilmResponse(
    geometry.facets[index].triIndex, optics.frame.hit.side >= 0.0, woDotM,
  );
  if (geometry.events[index] == SMS_EVENT_REFLECTION) {
    let nDotWi = dot(n, wi);
    let wiDotM = dot(wi, wm);
    if (!(nDotWi > 0.0) || !(wiDotM > 0.0)) { return 0.0; }
    var fresnel = dielectricFresnelExact(woDotM, media.etaI[index], media.etaT[index]);
    if (optics.metalness > 0.0) {
      fresnel = mix(fresnel, smsChannel(decodeMaterialColor(geometry.facets[index].indices.w).rgb, channel), optics.metalness);
    }
    if (film.present != 0u) { fresnel = smsChannel(film.reflectance, channel); }
    if (!(fresnel > 0.0)) { return 0.0; }
    if (!(optics.roughness > 0.0)) {
      return optics.surfaceCoverage * fresnel;
    }
    let alpha = optics.roughness * optics.roughness;
    let D = distributionGGX(dot(n, wm), optics.roughness);
    let G = smithG1GGX(nDotWo, alpha * alpha) * smithG1GGX(nDotWi, alpha * alpha);
    // eval(BRDF)*cos is in outgoing-direction measure. Offset normals are
    // sampled in half-vector measure, whose reflection Jacobian is 4|wi.m|.
    let halfVectorJacobian = 4.0 * abs(wiDotM);
    return optics.surfaceCoverage * D * G * fresnel /
      (4.0 * nDotWo) * halfVectorJacobian;
  }
  if (!(optics.transmission > 0.0) || optics.metalness > 0.0) { return 0.0; }
  let nDotWi = abs(dot(n, wi));
  let wiDotM = dot(wi, wm);
  if (dot(n, wi) >= 0.0 || !(nDotWi > 0.0) || !(wiDotM < 0.0)) { return 0.0; }
  let etaI = media.etaI[index];
  let etaT = media.etaT[index];
  let etap = etaT / etaI;
  let denominator = wiDotM + woDotM / etap;
  if (denominator == 0.0) { return 0.0; }
  var transmission = 1.0 - dielectricFresnelExact(woDotM, etaI, etaT);
  if (film.present != 0u) { transmission = smsChannel(film.transmittance, channel); }
  transmission = transmission * smsChannel(optics.layerTransmission, channel);
  if (payMaterialTransmission) {
    transmission = transmission * optics.transmission;
  }
  if (optics.bulkMedium == 0u) {
    let exitLayer = sampleFaceLayerControls(
      geometry.facets[index].triIndex, optics.frame.hit.side < 0.0,
    );
    transmission = transmission *
      (1.0 - dielectricFresnelExact(abs(wiDotM), etaT, etaI)) *
      smsChannel(faceLayerTransmission(exitLayer), channel);
  }
  if (!(transmission > 0.0)) { return 0.0; }
  if (!(optics.roughness > 0.0)) {
    return optics.surfaceCoverage * transmission / (etap * etap);
  }
  let alpha = optics.roughness * optics.roughness;
  let D = distributionGGX(dot(n, wm), optics.roughness);
  let G = smithG1GGX(nDotWo, alpha * alpha) * smithG1GGX(nDotWi, alpha * alpha);
  let denominator2 = denominator * denominator;
  let ft = transmission * D * G *
    abs(wiDotM * woDotM / (nDotWi * nDotWo * denominator2)) /
    (etap * etap);
  // Refraction half-vector change of variables for
  // h = normalize(wi + (etaI/etaT) * wo).
  let halfVectorJacobian = denominator2 * etap * etap / abs(woDotM);
  return optics.surfaceCoverage * ft * nDotWi * halfVectorJacobian;
}

struct SmsPathEvaluation {
  factor: f32,
  valid: u32,
};

fn smsEvaluatePath(
  geometry: SmsChainGeometry,
  media: SmsChainMedia,
  endpoint: SmsEndpoint,
  solved: SmsChainResult,
  receiver: vec3f,
  channel: u32,
) -> SmsPathEvaluation {
  var out: SmsPathEvaluation;
  out.factor = 0.0;
  out.valid = 0u;
  if (geometry.count == 0u) { return out; }
  let sourceContaining = smsSourceContainingMedia(
    endpoint, solved.vertices[0],
  );
  let receiverContaining = smsReceiverContainingMedia(
    receiver, solved.vertices[geometry.count - 1u],
  );
  if (sourceContaining.valid == 0u || receiverContaining.valid == 0u) {
    return out;
  }
  var mediumBoundaryId: array<u32, 16>;
  var mediumRepresentedId: array<u32, 16>;
  var mediumDistance: array<f32, 16>;
  var mediumTransmissionPaid: array<u32, 16>;
  var depth = sourceContaining.state.depth;
  if (depth > MATERIAL_SHADOW_MEDIUM_CAPACITY) { return out; }
  for (var seed = 0u; seed < MATERIAL_SHADOW_MEDIUM_CAPACITY; seed += 1u) {
    if (seed >= depth) { break; }
    mediumBoundaryId[seed] = sourceContaining.state.materialId[seed];
    mediumRepresentedId[seed] = sourceContaining.state.instance[seed];
    mediumDistance[seed] = 0.0;
    mediumTransmissionPaid[seed] = 0u;
  }
  var previous = endpoint.position;
  var factor = 1.0;
  for (var i = 0u; i < SMS_MAX_VERTICES; i = i + 1u) {
    if (i >= geometry.count) { break; }
    let vertex = solved.vertices[i];
    var segmentDistance = length(vertex - previous);
    if (i == 0u && endpoint.sourceMode == 1u) { segmentDistance = 0.0; }
    if (depth > 0u && segmentDistance > 0.0) {
      let top = depth - 1u;
      mediumDistance[top] = mediumDistance[top] + segmentDistance;
    }
    var incoming = safe_normalize(vertex - previous);
    if (i == 0u && endpoint.sourceMode == 1u) {
      incoming = -endpoint.towardLight;
    }
    let entering = dot(
      incoming, geometry.facets[i].geometricNormal,
    ) < 0.0;
    let optics = smsFacetOpticsAt(geometry.facets[i], vertex, incoming);
    if (optics.frame.valid == 0u) { return out; }
    var payMaterialTransmission = optics.bulkMedium == 0u || entering;
    if (optics.bulkMedium != 0u && !entering) {
      if (
        depth == 0u ||
        mediumBoundaryId[depth - 1u] !=
          geometry.facets[i].encodedBoundaryId ||
        mediumRepresentedId[depth - 1u] !=
          geometry.facets[i].representedPrimitiveInstanceId
      ) { return out; }
      payMaterialTransmission = mediumTransmissionPaid[depth - 1u] == 0u;
    }
    var next = receiver;
    if (i + 1u < geometry.count) { next = solved.vertices[i + 1u]; }
    let interfaceFactor = smsInterfaceFactor(
      geometry, media, i, vertex, previous, next, endpoint, channel,
      payMaterialTransmission,
    );
    if (!(interfaceFactor > 0.0) || !(interfaceFactor < INFINITY)) { return out; }
    factor = factor * interfaceFactor;
    if (geometry.events[i] == SMS_EVENT_TRANSMISSION &&
        optics.bulkMedium != 0u) {
      if (entering) {
        if (depth >= MATERIAL_SHADOW_MEDIUM_CAPACITY) { return out; }
        mediumBoundaryId[depth] = geometry.facets[i].encodedBoundaryId;
        mediumRepresentedId[depth] =
          geometry.facets[i].representedPrimitiveInstanceId;
        mediumDistance[depth] = 0.0;
        mediumTransmissionPaid[depth] = 1u;
        depth = depth + 1u;
      } else {
        let top = depth - 1u;
        factor = factor * smsChannel(
          materialShadowBeerForSegment(
            geometry.facets[i].triIndex,
            smsBeerTint(optics),
            optics.thickness,
            smsThicknessMapScale(optics),
            sampleVolumeScatteringControls(geometry.facets[i].triIndex),
            mediumDistance[top],
          ),
          channel,
        );
        depth = top;
      }
    }
    previous = vertex;
  }
  let finalDistance = length(receiver - previous);
  if (depth > 0u && finalDistance > 0.0) {
    let top = depth - 1u;
    mediumDistance[top] = mediumDistance[top] + finalDistance;
  }
  if (depth != receiverContaining.state.depth) { return out; }
  for (var live = 0u; live < MATERIAL_SHADOW_MEDIUM_CAPACITY; live += 1u) {
    if (live >= depth) { break; }
    if (
      mediumBoundaryId[live] != receiverContaining.state.materialId[live] ||
      mediumRepresentedId[live] != receiverContaining.state.instance[live]
    ) { return out; }
    factor = factor * smsChannel(
      materialShadowBeerForSegment(
        receiverContaining.state.tri[live],
        receiverContaining.state.tint[live],
        receiverContaining.state.thickness[live],
        receiverContaining.state.thicknessMapScale[live],
        receiverContaining.state.scattering[live],
        mediumDistance[live],
      ),
      channel,
    );
  }
  if (!(factor > 0.0) || !(factor < INFINITY)) { return out; }
  out.factor = factor;
  out.valid = 1u;
  return out;
}

fn lo_manifold_caustic(
  gid: vec2u,
  receiver: vec3f,
  normal: vec3f,
  clearcoatNormal: vec3f,
  wo: vec3f,
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
  isGlass: bool,
) -> vec3f {
  if (isGlass || sceneMneeFacetDomainCount() == 0u) { return vec3f(0.0); }
  let pixelIndex = gid.y * ubo.screenSize.x + gid.x;
  var rng = pcgInit(gid.x, gid.y, ubo.frameSeed ^ 0x4d4e4545u);
  let endpoint = smsSampleEndpoint(&rng, receiver);
  if (endpoint.valid == 0u || !(endpoint.selectionPdf > 0.0) ||
      !(endpoint.endpointPdf > 0.0)) { return vec3f(0.0); }
  let configuredLength = smsConfiguredChainLength();
  let chainLength = 1u + smsUniformIndex(&rng, configuredLength);
  let lengthPmf = 1.0 / f32(configuredLength);
  var geometry: SmsChainGeometry;
  geometry.count = chainLength;
  geometry.sourceMode = endpoint.sourceMode;
  geometry.sourceDirection = endpoint.towardLight;
  var seeds: array<vec3f, 8>;
  var previousSeed = endpoint.position;
  var proposalDensity = lengthPmf * endpoint.selectionPdf * endpoint.endpointPdf;
  for (var i = 0u; i < SMS_MAX_VERTICES; i = i + 1u) {
    if (i >= chainLength) { break; }
    let draw = smsDrawFacet(&rng);
    if (draw.facet.valid == 0u) { return vec3f(0.0); }
    geometry.facets[i] = draw.facet;
    seeds[i] = draw.seed;
    var incoming = safe_normalize(draw.seed - previousSeed);
    if (i == 0u && endpoint.sourceMode == 1u) { incoming = -endpoint.towardLight; }
    let optics = smsFacetOpticsAt(draw.facet, draw.seed, incoming);
    if (optics.frame.valid == 0u) { return vec3f(0.0); }
    let transmissionSupported = optics.transmission > 0.0 && optics.metalness <= 0.0;
    var event = SMS_EVENT_REFLECTION;
    var eventPmf = 1.0;
    if (transmissionSupported) {
      let transmissionEventPmf = represented_bernoulli_probability_f32(0.5);
      event = select(
        SMS_EVENT_REFLECTION,
        SMS_EVENT_TRANSMISSION,
        rand_f32(&rng) < transmissionEventPmf,
      );
      eventPmf = select(
        1.0 - transmissionEventPmf,
        transmissionEventPmf,
        event == SMS_EVENT_TRANSMISSION,
      );
    }
    geometry.events[i] = event;
    let offset = smsSampleOffsetNormal(optics, incoming, event, eventPmf, &rng);
    if (offset.valid == 0u) { return vec3f(0.0); }
    geometry.offsets[i] = offset;
    // The recurrence re-draws uniform-area seeds on these frozen facets and
    // estimates the inverse convergence-basin probability. Keep only the
    // discrete facet/instance topology PMF here; q_A is already canceled.
    proposalDensity = proposalDensity * draw.facet.pairPmf *
      offset.proposalPdf * offset.eventProposalPmf;
    previousSeed = draw.seed;
  }
  if (!(proposalDensity > 0.0) || !(proposalDensity < INFINITY)) { return vec3f(0.0); }
  var receiverTangent: vec3f;
  var receiverBitangent: vec3f;
  smsBuildFrame(normal, &receiverTangent, &receiverBitangent);
  var result = vec3f(0.0);
  for (var channel = 0u; channel < 3u; channel = channel + 1u) {
    let built = smsBuildMedia(
      geometry, endpoint, seeds, receiver, channel,
    );
    if (built.valid == 0u) { continue; }
    let solved = smsSolveChain(geometry, built.media, endpoint.position, receiver, seeds);
    if (solved.valid == 0u) { continue; }
    let chainVisibility = smsChainIdentityVisible(geometry, solved, receiver);
    if (!(chainVisibility > 0.0)) { continue; }
    let path = smsEvaluatePath(geometry, built.media, endpoint, solved, receiver, channel);
    if (path.valid == 0u) { continue; }
    let endpointTransport = smsEndpointTransport(
      endpoint, solved.vertices[0], geometry.facets[0],
    );
    let light = smsChannel(endpointTransport, channel);
    if (!(light > 0.0) || !(light < INFINITY)) { continue; }
    let wi = safe_normalize(solved.vertices[chainLength - 1u] - receiver);
    let receiverCosine = dot(normal, wi);
    if (!(receiverCosine > 0.0)) { continue; }
    var focusing = 0.0;
    if (endpoint.family == SMS_SOURCE_AREA) {
      focusing = smsAreaFocusingDet(
        geometry, built.media, solved, endpoint.position, receiver,
        endpoint.tangent, endpoint.bitangent,
      );
    } else if (endpoint.sourceMode == 1u) {
      focusing = smsDirectionalFocusingDet(
        geometry, built.media, solved, endpoint.position, receiver,
        receiverTangent, receiverBitangent,
      );
    } else {
      focusing = smsFiniteFocusingDet(
        geometry, built.media, solved, endpoint.position, receiver,
        receiverTangent, receiverBitangent,
      );
    }
    if (!(focusing > 0.0) || !(focusing < INFINITY)) { continue; }
    let multiplicityWeight = smsBoundedMultiplicityEstimate(
      geometry, built.media, endpoint, receiver, solved, pixelIndex, channel,
    );
    // The canonical layered receiver evaluation includes N·L. Point and
    // directional focusing determinants already contain receiver
    // foreshortening, so remove that one cosine for those endpoint families;
    // area focusing still needs the complete BRDF·cos response.
    let receiverBrdfCosine =
      evalGGXWithSpecularClearcoatSheenWithAnisotropyFrame(
        albedo, rough, metal, specular.rgb, specular.a,
        anisotropy.x, anisotropy.y, iridescence,
        clearcoat.x, clearcoat.y, sheen.a, sheenRoughness, sheen.rgb,
        anisotropyTangent, anisotropyBitangent,
        normal, clearcoatNormal, wo, wi,
      );
    let receiverResponse = receiverBrdfCosine * select(
      1.0 / max(receiverCosine, 1e-6),
      1.0,
      endpoint.family == SMS_SOURCE_AREA,
    );
    let contribution = smsChannel(receiverResponse, channel) *
      light * path.factor * chainVisibility * focusing * multiplicityWeight /
      proposalDensity;
    if (!(contribution > 0.0) || !(contribution < INFINITY)) { continue; }
    if (channel == 0u) { result.r = result.r + contribution; }
    if (channel == 1u) { result.g = result.g + contribution; }
    if (channel == 2u) { result.b = result.b + contribution; }
  }
  return result;
}
`;

export const MANIFOLD_CAUSTICS_MODULE: WgslModule = {
  name: 'manifoldCaustics',
  source: MANIFOLD_CAUSTICS_WGSL,
  requires: [
    'common',
    'materialAtlas',
    'ggxBrdf',
    'emitterSampling',
    'emitterLeAtXi',
    'environmentSample',
    'manifoldSmsSolver',
  ],
};
