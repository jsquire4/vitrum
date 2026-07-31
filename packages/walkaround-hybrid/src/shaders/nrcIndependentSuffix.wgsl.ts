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

fn nrcTeacherMaterialResponse(
  payload: RestirDIMaterialPayload,
  normal: vec3f,
  wo: vec3f,
  wi: vec3f,
) -> vec3f {
  let fCos = evalGGXWithSpecularClearcoatSheenWithAnisotropyFrame(
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
  ) * payload.layerTransmission;
  return applyHomogeneousVolumeSingleScatterDirectional(
    fCos,
    payload.albedo,
    payload.volumeScattering,
    payload.bulkThickness,
    normal,
    wo,
    wi,
  );
}

fn nrcTeacherShadowTint(
  pos: vec3f,
  geoNormal: vec3f,
  wi: vec3f,
  tMax: f32,
) -> vec3f {
  let offsetNormal = select(-geoNormal, geoNormal, dot(geoNormal, wi) >= 0.0);
  return traceSceneAlphaTintTransmittanceTextured(
    ubo.bvhMode, ubo.tlasNodeCount,

    pos + offsetNormal * walkaroundRayOriginBias(), wi, tMax, ubo.triIntersectEpsilon,
    bvh_material, BVH_MATERIAL_TEX_WIDTH, bvh_beer,
  );
}

// One exact-CDF area-light sample.  The source density is converted from area
// to solid angle by G = cos(light)/distance^2, leaving f*cos at the receiver.
fn nrcTeacherAreaNee(
  pos: vec3f,
  geoNormal: vec3f,
  normal: vec3f,
  wo: vec3f,
  payload: RestirDIMaterialPayload,
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
    );
    if (nrcTeacherMax3(visibility) <= 0.0) { return vec3f(0.0); }
  }

  let pdfArea = selectionPmf * ls.pdfArea;
  let geometry = emitterGeometry(cosLight, dist2, ubo.emitterDist2Floor);
  let response = nrcTeacherMaterialResponse(payload, normal, wo, wi);
  let Le = sampleEmitterLeAtXi(emitter, xi);
  if (!(pdfArea > 0.0)) { return vec3f(0.0); }
  return Le * visibility * response * geometry / pdfArea;
}

// Point/spot emitters are delta lights, hence the deterministic sum is already
// their exact direct-light estimator and needs no sampling-pdf division.
fn nrcTeacherAnalyticNee(
  pos: vec3f,
  geoNormal: vec3f,
  normal: vec3f,
  wo: vec3f,
  payload: RestirDIMaterialPayload,
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
      );
      if (nrcTeacherMax3(visibility) <= 0.0) { continue; }
    }
    let attenuation = nrc_teacherPointSpotAttenuation(
      dist, l3.z, l3.w, ubo.emitterDist2Floor,
    );
    direct = direct + l1.xyz * visibility *
      nrcTeacherMaterialResponse(payload, normal, wo, wi) * cone * attenuation *
      estimatorWeight;
  }
  return direct;
}

// The renderer interprets sunIntensity as integrated directional irradiance.
// Sampling its authored angular disc therefore estimates its directional
// average directly (no solid-angle division), matching the live shade model.
fn nrcTeacherSunNee(
  pos: vec3f,
  geoNormal: vec3f,
  normal: vec3f,
  wo: vec3f,
  payload: RestirDIMaterialPayload,
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
    visibility = nrcTeacherShadowTint(pos, geoNormal, wi, INFINITY);
    if (nrcTeacherMax3(visibility) <= 0.0) { return vec3f(0.0); }
  }
  return vec3f(ubo.sunIntensity) * visibility *
    nrcTeacherMaterialResponse(payload, normal, wo, wi);
}

fn nrcTeacherAuthoredBeerTint(hit: IntersectionResult) -> vec3f {
  let coord = vec2u(
    hit.indices.w % BVH_MATERIAL_TEX_WIDTH,
    hit.indices.w / BVH_MATERIAL_TEX_WIDTH,
  );
  let packed = textureLoad(bvh_beer, vec2i(coord), 0).r;
  let tint = vec3f(
    f32((packed >> 24u) & 0xffu),
    f32((packed >> 16u) & 0xffu),
    f32((packed >> 8u) & 0xffu),
  ) / 255.0;
  return applyThicknessMapToBeerTint(
    hit.indices.w,
    hit.uv,
    materialAtlasUv1ForHit(hit),
    tint,
  );
}

fn nrcTeacherBeerForSegment(
  triIndex: u32,
  authoredTint: vec3f,
  authoredThickness: f32,
  segmentLength: f32,
) -> vec3f {
  if (segmentLength <= 0.0) { return vec3f(1.0); }
  if (!(authoredThickness > 0.0)) { return vec3f(0.0); }
  let rgbBeer = pow(
    clamp(authoredTint, vec3f(0.0), vec3f(1.0)),
    vec3f(segmentLength / authoredThickness),
  );
  return materialSpectralAttenuation(triIndex, segmentLength, rgbBeer);
}

// Independent online teacher.  It intentionally has no DDGI/cache arguments
// and never calls sampleDDGIAtPoint or nrcQueryRadiance.
fn nrcTraceIndependentSuffix(
  initialHit: IntersectionResult,
  initialPos: vec3f,
  initialWo: vec3f,
  rng: ptr<function, u32>,
) -> vec3f {
  var currentHit = initialHit;
  var currentPos = initialPos;
  var wo = safe_normalize(initialWo);
  var throughput = vec3f(1.0);
  var radiance = vec3f(0.0);
  var arrivedThroughDelta = true;
  var segmentLength = 0.0;
  var mediumDepth = 0u;
  var mediumMaterialWord: array<u32, 4>;
  var mediumTri: array<u32, 4>;
  var mediumInstance: array<u32, 4>;
  var mediumIor: array<f32, 4>;
  var mediumAuthoredTint: array<vec3f, 4>;
  var mediumAuthoredThickness: array<f32, 4>;

  for (var depth = 0u; depth < NRC_TEACHER_MAX_VERTICES; depth = depth + 1u) {
    if (mediumDepth > 0u && segmentLength > 0.0) {
      let top = mediumDepth - 1u;
      throughput = throughput * nrcTeacherBeerForSegment(
        mediumTri[top],
        mediumAuthoredTint[top],
        mediumAuthoredThickness[top],
        segmentLength,
      );
      if (
        !nrcTeacherFinite3(throughput) ||
        nrcTeacherMax3(throughput) <= 0.0
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
    let transmission = sampleTransmissionMapForHit(currentHit, scalar.a);
    let payload = sampleRestirDIMaterialPayloadForHit(
      currentHit, smoothNormal, normal, scalar.rgb, materialWord, wo,
    );

    if (decodeIsUnlitMaterial(materialWord)) {
      radiance = radiance + throughput * payload.albedo;
      break;
    }

    // NEE owns non-delta emitter connections.  Emission reached through the
    // preceding diffuse/glossy continuation is suppressed to avoid double
    // counting; camera/cache and dielectric-delta arrivals retain it.
    if (depth == 0u || arrivedThroughDelta) {
      radiance = radiance + throughput * restir_gi_surface_emission_for_hit(currentHit);
    }
    var nextDir = vec3f(0.0, 1.0, 0.0);
    var nextThroughput = throughput;
    var nextIsDelta = false;

    if (materialHasTransmission(transmission)) {
      if (depth + 1u >= NRC_TEACHER_MAX_VERTICES) { break; }
      // Bounded dielectric interface walk. IntersectionResult.normal is
      // face-forward, so authored side — not dot(ray, normal) — owns medium
      // entry/exit. Fresnel branch probabilities cancel their delta weights.
      let incident = -wo;
      let entering = currentHit.side >= 0.0;
      let materialIor = materialDispersionIorRgb(
        currentHit.indices.w, decodeIor(materialWord),
      ).g;
      let thinSheet = payload.bulkThickness <= 0.0;
      if (!thinSheet && !entering && mediumDepth == 0u) { break; }

      var etaIncident = 1.0;
      if (mediumDepth > 0u) {
        etaIncident = mediumIor[mediumDepth - 1u];
      }
      var etaTarget = materialIor;
      if (!thinSheet && !entering) {
        let top = mediumDepth - 1u;
        if (
          mediumMaterialWord[top] != materialWord ||
          mediumInstance[top] != currentHit.instanceIndex
        ) {
          break;
        }
        etaTarget = 1.0;
        if (mediumDepth > 1u) {
          etaTarget = mediumIor[mediumDepth - 2u];
        }
      }

      let etaRatio = etaIncident / max(etaTarget, 1e-6);
      let orientedNormal = select(
        -normal,
        normal,
        dot(incident, normal) < 0.0,
      );
      let cosIAbs = clamp(-dot(incident, orientedNormal), 0.0, 1.0);
      let sin2T = etaRatio * etaRatio * max(0.0, 1.0 - cosIAbs * cosIAbs);
      let r0 = (etaTarget - etaIncident) /
        max(etaTarget + etaIncident, 1e-6);
      let interfaceFresnel = clamp(
        r0 * r0 + (1.0 - r0 * r0) * pow(1.0 - cosIAbs, 5.0),
        0.0,
        1.0,
      );
      let thinTransmission = (1.0 - interfaceFresnel) *
        (1.0 - interfaceFresnel);
      let branchFresnel = select(
        interfaceFresnel,
        1.0 - thinTransmission,
        thinSheet,
      );
      if (sin2T >= 1.0 || rand_f32(rng) < branchFresnel) {
        nextDir = safe_normalize(reflect(incident, orientedNormal));
      } else {
        if (thinSheet) {
          // A geometric sheet represents reciprocal entry+exit boundaries and
          // has no interior Beer segment.
          nextDir = incident;
          let reverseLayer = sampleFaceLayerControls(
            currentHit.indices.w,
            currentHit.side < 0.0,
          );
          nextThroughput = nextThroughput *
            transmission *
            payload.layerTransmission *
            clamp(reverseLayer.rgb, vec3f(0.0), vec3f(1.0));
        } else {
          let cosT = sqrt(max(0.0, 1.0 - sin2T));
          nextDir = safe_normalize(
            etaRatio * incident +
            (etaRatio * cosIAbs - cosT) * orientedNormal,
          );
          nextThroughput = nextThroughput *
            payload.layerTransmission *
            (etaRatio * etaRatio);
          if (entering) {
            nextThroughput = nextThroughput * transmission;
            if (mediumDepth >= NRC_TEACHER_MAX_VERTICES) { break; }
            mediumMaterialWord[mediumDepth] = materialWord;
            mediumTri[mediumDepth] = currentHit.indices.w;
            mediumInstance[mediumDepth] = currentHit.instanceIndex;
            mediumIor[mediumDepth] = materialIor;
            mediumAuthoredTint[mediumDepth] =
              nrcTeacherAuthoredBeerTint(currentHit);
            mediumAuthoredThickness[mediumDepth] = payload.bulkThickness;
            mediumDepth = mediumDepth + 1u;
          } else {
            mediumDepth = mediumDepth - 1u;
          }
        }
      }
      nextIsDelta = true;
    } else {
      radiance = radiance + throughput * (
        nrcTeacherAreaNee(currentPos, geoNormal, normal, wo, payload, rng) +
        nrcTeacherAnalyticNee(currentPos, geoNormal, normal, wo, payload, rng) +
        nrcTeacherSunNee(currentPos, geoNormal, normal, wo, payload, rng)
      );

      if (depth + 1u >= NRC_TEACHER_MAX_VERTICES) { break; }

      if (rand_f32(rng) < NRC_TEACHER_SPECULAR_MIX) {
        nextDir = ggxSampleVndf(normal, wo, payload.rough, rng);
      } else {
        nextDir = sampleCosineHemisphere(normal, rng);
      }
      let cosNext = max(0.0, dot(normal, nextDir));
      if (cosNext <= 0.0) { break; }
      let pdfCos = cosNext * INV_PI;
      let pdfSpec = ggxVndfReflectionPdf(normal, wo, nextDir, payload.rough);
      let proposalPdf =
        NRC_TEACHER_SPECULAR_MIX * pdfSpec +
        (1.0 - NRC_TEACHER_SPECULAR_MIX) * pdfCos;
      if (!(proposalPdf > 0.0)) { break; }
      nextThroughput = nextThroughput *
        nrcTeacherMaterialResponse(payload, normal, wo, nextDir) / proposalPdf;
    }

    if (!nrcTeacherFinite3(nextThroughput) || nrcTeacherMax3(nextThroughput) <= 0.0) {
      break;
    }
    if (depth + 1u >= NRC_TEACHER_RR_START) {
      let survive = clamp(nrcTeacherMax3(nextThroughput), 0.05, 0.95);
      if (rand_f32(rng) >= survive) { break; }
      nextThroughput = nextThroughput / survive;
    }

    let offsetNormal = select(-geoNormal, geoNormal, dot(geoNormal, nextDir) >= 0.0);
    let nextRay = Ray(
      currentPos + offsetNormal * walkaroundRayOriginBias(),
      nextDir,
    );
    let nextHit = traceSceneFirstHitAlphaMaskTextured(
      ubo.bvhMode, ubo.tlasNodeCount,

      nextRay, ubo.triIntersectEpsilon,
      bvh_material, BVH_MATERIAL_TEX_WIDTH,
      pcgNext(rng),
    );
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

    let nextPos = nextRay.origin + nextDir * nextHit.dist;
    segmentLength = length(nextPos - currentPos);
    currentPos = nextPos;
    currentHit = nextHit;
    wo = -nextDir;
    throughput = nextThroughput;
    arrivedThroughDelta = nextIsDelta;
  }

  return select(vec3f(0.0), max(radiance, vec3f(0.0)), nrcTeacherFinite3(radiance));
}
`;
