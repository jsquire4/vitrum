/**
 * Homogeneous-volume direct-light and phase-connection estimators.
 *
 * All finite-emitter sampling follows the same packed emitter order and
 * light-tree PMF as the surface estimator. Selection probability is compensated
 * outside the conditional phase/light MIS, matching the surface convention.
 */
export const PT_WEBGPU_MEDIUM_NEE_WGSL = /* wgsl */ `
struct DirectLightSelection {
  emitterIndex: u32,
  invPdf: f32,
};

fn sampleCanonicalDirectLight(
  position: vec3f,
  lightCount: u32,
  sumAll: bool,
  rng: ptr<function, PtRngState>,
) -> DirectLightSelection {
  if (sumAll) { return DirectLightSelection(0u, 1.0); }
  let treeActive =
    params.lightTreeEnabled != 0u && params.lightTreeNodeCount > 0u;
  if (treeActive) {
    let selected = sampleLightTree(
      position, LT_DIST2_FLOOR, params.lightTreeNodeCount, rng,
    );
    if (
      selected.emitterIndex >= 0 && selected.pdf > 0.0 &&
      u32(selected.emitterIndex) < lightCount
    ) {
      return DirectLightSelection(
        u32(selected.emitterIndex), 1.0 / selected.pdf,
      );
    }
  }
  let index = u32(min(
    floor(rand_f32(rng) * f32(lightCount)), f32(lightCount - 1u),
  ));
  return DirectLightSelection(index, f32(lightCount));
}

fn hasDistantDirectEnvironment() -> bool {
  return hasEnvironmentMap();
}

fn distantDirectEmitterCount() -> u32 {
  return params.directionalLightCount + select(0u, 1u, hasDistantDirectEnvironment());
}

fn distantDirectEmitterGlobalIndex(localIndex: u32) -> u32 {
  if (localIndex < params.directionalLightCount) { return localIndex; }
  return params.directionalLightCount + params.pointLightCount +
    params.spotLightCount + params.rectAreaLightCount +
    params.meshAreaLightCount;
}

fn distantDirectEnvironmentPower() -> f32 {
  let dims = environmentDimensions();
  let count = dims.x * dims.y;
  if (count > 0u && arrayLength(&environmentMapCdf) >= count + 1u) {
    return max(environmentMapCdf[count], 0.0);
  }
  return 0.0;
}

fn distantDirectEmitterPower(localIndex: u32) -> f32 {
  let diskArea = PI * params.sceneRadius * params.sceneRadius;
  if (localIndex < params.directionalLightCount) {
    let irradiance = directionalLights[localIndex * 2u + 1u].rgb;
    return diskArea * max(luminance(irradiance), 0.0);
  }
  return diskArea * distantDirectEnvironmentPower();
}

fn distantDirectSelectionPdf(globalIndex: u32) -> f32 {
  if (params.directLightingMode == 1u) { return 1.0; }
  let count = distantDirectEmitterCount();
  if (count == 0u) { return 0.0; }
  var localIndex = params.directionalLightCount;
  if (globalIndex < params.directionalLightCount) {
    localIndex = globalIndex;
  } else if (
    !hasDistantDirectEnvironment() ||
    globalIndex != params.directionalLightCount + params.pointLightCount +
      params.spotLightCount + params.rectAreaLightCount +
      params.meshAreaLightCount
  ) {
    return 0.0;
  }
  var totalPower = 0.0;
  for (var i = 0u; i < count; i = i + 1u) {
    totalPower = totalPower + distantDirectEmitterPower(i);
  }
  if (!(totalPower > 0.0)) { return 0.0; }
  return distantDirectEmitterPower(localIndex) / totalPower;
}

fn sampleDistantDirectLight(
  sumAll: bool,
  rng: ptr<function, PtRngState>,
) -> DirectLightSelection {
  let count = distantDirectEmitterCount();
  if (count == 0u || sumAll) { return DirectLightSelection(0u, 1.0); }
  var totalPower = 0.0;
  for (var i = 0u; i < count; i = i + 1u) {
    totalPower = totalPower + distantDirectEmitterPower(i);
  }
  if (!(totalPower > 0.0)) {
    return DirectLightSelection(0xffffffffu, 0.0);
  }
  let pickTarget = rand_f32(rng) * totalPower;
  var cumulative = 0.0;
  for (var i = 0u; i < count; i = i + 1u) {
    let power = distantDirectEmitterPower(i);
    cumulative = cumulative + power;
    if (power > 0.0 && pickTarget < cumulative) {
      return DirectLightSelection(
        distantDirectEmitterGlobalIndex(i), totalPower / power,
      );
    }
  }
  return DirectLightSelection(0xffffffffu, 0.0);
}

struct MediumVisibility {
  transmittance: vec3f,
  visible: bool,
};

fn mediumMaterialAtHit(hit: SceneHit) -> DecodedMaterial {
  let matId = hitMaterialId(hit);
  var mat = decodeMaterial(matId);
  let thicknessSample = sampleVolumeThicknessTexture(
    matId, hit.triIndex, hit.baryVW, hit.instanceIndex,
  );
  if (thicknessSample >= 0.0) {
    mat.volumeThickness = max(mat.volumeThickness * thicknessSample, 0.0);
    mat.hasVolumeThickness = true;
  }
  return mat;
}

// A straight endpoint connection may cross only an optical-identity boundary.
// Any IOR transition belongs to a specular/refraction path family: accepting it
// here would neither bend the ray nor include the interface Fresnel factors.
// Those paths remain with BDPT/SPPM instead of biasing this NEE estimator.
fn mediumStraightBoundaryIsNull(incidentIor: f32, transmittedIor: f32) -> bool {
  let scale = max(max(abs(incidentIor), abs(transmittedIor)), 1.0);
  return abs(incidentIor - transmittedIor) <= 1e-4 * scale;
}

fn traceMediumVisibility(
  origin: vec3f,
  direction: vec3f,
  maxDistance: f32,
  sourceLayer: BdptMediumLayer,
  sourceStack: ptr<
    function,
    array<BdptMediumLayer, BDPT_MEDIUM_STACK_LIMIT>
  >,
  sourceDepth: u32,
  heroLambda: f32,
  ignoreOpaque: bool,
  rng: ptr<function, PtRngState>,
) -> MediumVisibility {
  var result = MediumVisibility(vec3f(1.0), false);
  if (
    sourceDepth == 0u ||
    sourceDepth > BDPT_MEDIUM_STACK_LIMIT ||
    !mediumBoundaryIsValid(
      vec2u(sourceLayer.boundaryKind, sourceLayer.boundaryIndex),
    )
  ) {
    return result;
  }
  var stack: array<BdptMediumLayer, BDPT_MEDIUM_STACK_LIMIT>;
  for (var i = 0u; i < sourceDepth; i = i + 1u) {
    let source = (*sourceStack)[i];
    if (!mediumBoundaryIsValid(
      vec2u(source.boundaryKind, source.boundaryIndex),
    )) {
      return result;
    }
    stack[i] = source;
  }
  let sourceTop = stack[sourceDepth - 1u];
  if (
    sourceTop.matId != sourceLayer.matId ||
    sourceTop.boundaryKind != sourceLayer.boundaryKind ||
    sourceTop.boundaryIndex != sourceLayer.boundaryIndex
  ) {
    return result;
  }
  // The scatter point has already consumed part of the top layer's finite
  // attenuation distance. Preserve every enclosing layer, but replace the top
  // with the caller's post-flight state.
  stack[sourceDepth - 1u] = sourceLayer;
  var depth = sourceDepth;
  var rayOrigin = origin;
  var travelled = 0.0;
  let surfaceHitLimit = sceneSurfaceHitLimit();
  var surfaceHitCount = 0u;

  loop {
    let remaining = max(maxDistance - travelled, 0.0);
    if (remaining <= 1e-5) {
      result.visible = true;
      return result;
    }
    let shadowRay = Ray(rayOrigin, direction);
    let hit = traceClosest(shadowRay, 1e-4, remaining);
    let segment = select(remaining, hit.dist, hit.didHit);
    if (depth > 0u) {
      let top = depth - 1u;
      let attenDistance = min(segment, stack[top].remainingDistance);
      result.transmittance = result.transmittance *
        exp(-stack[top].sigmaT * max(attenDistance, 0.0));
      stack[top].remainingDistance =
        max(stack[top].remainingDistance - attenDistance, 0.0);
    }
    if (!hit.didHit) {
      result.visible = true;
      return result;
    }
    // Observe the final miss after exactly surfaceHitLimit pass-through
    // surfaces. A further hit is impossible under the published scene support,
    // so fail closed instead of silently accepting an unbounded/corrupt walk.
    if (surfaceHitCount >= surfaceHitLimit) {
      return result;
    }
    surfaceHitCount = surfaceHitCount + 1u;

    let matId = hitMaterialId(hit);
    if (alphaTestPassThrough(
      matId, hit.triIndex, hit.baryVW, hit.instanceIndex, rng,
    )) {
      let advance = hit.dist + 1e-4;
      rayOrigin = rayOrigin + direction * advance;
      travelled = travelled + advance;
      continue;
    }

    let mat = mediumMaterialAtHit(hit);
    let transmission = clamp(
      mat.transmission * sampleTransmissionTexture(
        matId, hit.triIndex, hit.baryVW, hit.instanceIndex,
      ),
      0.0, 1.0,
    );
    if (!mat.isTranslucent || transmission <= 0.0) {
      if (!ignoreOpaque) { return result; }
      let advance = hit.dist + 1e-4;
      rayOrigin = rayOrigin + direction * advance;
      travelled = travelled + advance;
      continue;
    }

    result.transmittance = result.transmittance * transmission;
    let frontFace = hit.frontFace;
    let boundary = mediumBoundaryIdentity(hit.triIndex, hit.instanceIndex);
    if (!mediumBoundaryIsValid(boundary)) { return result; }
    var incidentIor = 1.0;
    if (depth > 0u) {
      incidentIor = stack[depth - 1u].ior;
    }
    var transmittedIor = max(mat.ior, 1e-4);
    if (!frontFace) {
      if (
        depth == 0u ||
        !bdptMediumLayerMatchesBoundary(stack[depth - 1u], matId, boundary)
      ) {
        return result;
      }
      transmittedIor = 1.0;
      if (depth > 1u) {
        transmittedIor = stack[depth - 2u].ior;
      }
    }
    if (!mediumStraightBoundaryIsNull(incidentIor, transmittedIor)) {
      return result;
    }
    if (frontFace) {
      if (depth >= BDPT_MEDIUM_STACK_LIMIT) { return result; }
      stack[depth] = bdptMediumLayer(matId, mat, heroLambda, boundary);
      depth = depth + 1u;
    } else {
      depth = depth - 1u;
    }

    let advance = hit.dist + 1e-4;
    rayOrigin = rayOrigin + direction * advance;
    travelled = travelled + advance;
  }
  return result;
}

struct MediumEmitterSample {
  wi: vec3f,
  radiance: vec3f,
  distance: f32,
  pdf: f32,
  delta: bool,
  ignoreOpaque: bool,
  valid: bool,
};

fn invalidMediumEmitterSample() -> MediumEmitterSample {
  return MediumEmitterSample(
    vec3f(0.0, 1.0, 0.0), vec3f(0.0), 0.0, 0.0, false, false, false,
  );
}

fn sampleMediumEmitter(
  flat: u32,
  position: vec3f,
  rng: ptr<function, PtRngState>,
) -> MediumEmitterSample {
  var current = 0u;
  for (var di = 0u; di < params.directionalLightCount; di = di + 1u) {
    if (current == flat) {
      let base = di * 2u;
      let dirRecord = directionalLights[base];
      let irradiance = directionalLights[base + 1u];
      if (irradiance.w <= 0.0) { return invalidMediumEmitterSample(); }
      let rawDiameter = dirRecord.w;
      let ignoreOpaque = rawDiameter < 0.0;
      let diameter = select(rawDiameter, -1.0 - rawDiameter, ignoreOpaque);
      let wi = sampleDirectionalCone(
        rng, safe_normalize(dirRecord.xyz), diameter,
      );
      let directionIsDelta = diameter <= 0.0;
      var directionPdf = 1.0;
      if (!directionIsDelta) {
        let coneQuarterSin = sin(diameter * 0.25);
        directionPdf =
          1.0 / (4.0 * PI * coneQuarterSin * coneQuarterSin);
      }
      return MediumEmitterSample(
        wi, irradiance.rgb * select(1.0, directionPdf, !directionIsDelta),
        INFINITY, directionPdf,
        directionIsDelta, ignoreOpaque, true,
      );
    }
    current = current + 1u;
  }

  for (var pi = 0u; pi < params.pointLightCount; pi = pi + 1u) {
    if (current == flat) {
      let base = pi * 3u;
      let toLight = pointLights[base].xyz - position;
      let distance = length(toLight);
      let extra = pointLights[base + 2u];
      if (distance <= 1e-5 || (extra.x > 0.0 && distance > extra.x)) {
        return invalidMediumEmitterSample();
      }
      let attenuation =
        pointSpotDistanceAttenuation(distance, extra.x, extra.y);
      return MediumEmitterSample(
        toLight / distance, pointLights[base + 1u].rgb * attenuation,
        distance, 1.0, true, extra.z > 0.5, true,
      );
    }
    current = current + 1u;
  }

  for (var si = 0u; si < params.spotLightCount; si = si + 1u) {
    if (current == flat) {
      let base = si * 4u;
      let toLight = spotLights[base].xyz - position;
      let distance = length(toLight);
      let axisOuter = spotLights[base + 1u];
      let radianceInner = spotLights[base + 2u];
      let extra = spotLights[base + 3u];
      if (distance <= 1e-5 || (extra.x > 0.0 && distance > extra.x)) {
        return invalidMediumEmitterSample();
      }
      let wi = toLight / distance;
      let coneCos = dot(-wi, safe_normalize(axisOuter.xyz));
      if (coneCos < axisOuter.w) { return invalidMediumEmitterSample(); }
      let softness = smoothstep(
        axisOuter.w, max(radianceInner.w, axisOuter.w + 1e-6), coneCos,
      );
      let attenuation =
        pointSpotDistanceAttenuation(distance, extra.x, extra.y);
      return MediumEmitterSample(
        wi, radianceInner.rgb * softness * attenuation,
        distance, 1.0, true, extra.z > 0.5, true,
      );
    }
    current = current + 1u;
  }

  for (var ri = 0u; ri < params.rectAreaLightCount; ri = ri + 1u) {
    if (current == flat) {
      let base = ri * 4u;
      let center = rectAreaLights[base].xyz;
      let uAxis = rectAreaLights[base + 1u].xyz;
      let vAxis = rectAreaLights[base + 2u].xyz;
      let shape = rectAreaLights[base + 3u];
      let disc = abs(shape.w - 1.0) < 0.5;
      let xi = vec2f(rand_f32(rng), rand_f32(rng));
      var lightPosition: vec3f;
      var area: f32;
      if (disc) {
        let uv = concentricDiscSample(xi * 2.0 - vec2f(1.0));
        lightPosition = center + uAxis * uv.x + vAxis * uv.y;
        area = PI * length(cross(uAxis, vAxis));
      } else {
        lightPosition =
          center + uAxis * (xi.x * 2.0 - 1.0) +
          vAxis * (xi.y * 2.0 - 1.0);
        area = 4.0 * length(cross(uAxis, vAxis));
      }
      let toLight = lightPosition - position;
      let distance2 = dot(toLight, toLight);
      if (distance2 <= 0.0 || area <= 0.0) {
        return invalidMediumEmitterSample();
      }
      let distance = sqrt(distance2);
      let wi = toLight / distance;
      let normal = safe_normalize(cross(uAxis, vAxis));
      let cosLight = max(dot(normal, -wi), 0.0);
      if (cosLight <= 0.0) { return invalidMediumEmitterSample(); }
      return MediumEmitterSample(
        wi, shape.rgb, distance, distance2 / (cosLight * area),
        false, rectAreaLights[base].w > 0.5, true,
      );
    }
    current = current + 1u;
  }

  for (var mi = 0u; mi < params.meshAreaLightCount; mi = mi + 1u) {
    if (current == flat) {
      let base = meshAreaLightBase(mi);
      let a = meshAreaLights[base].xyz;
      let b = meshAreaLights[base + 1u].xyz;
      let c = meshAreaLights[base + 2u].xyz;
      let emission = meshAreaLights[base + 3u];
      let r1 = rand_f32(rng);
      let r2 = rand_f32(rng);
      let su = sqrt(r1);
      let lightPosition =
        a * (1.0 - su) + b * (r2 * su) + c * ((1.0 - r2) * su);
      let emissionRadiance = sampleMeshAreaLightRadiance(
        mi, vec3f(1.0 - su, r2 * su, (1.0 - r2) * su), lightPosition,
      );
      let toLight = lightPosition - position;
      let distance2 = dot(toLight, toLight);
      let crossEdges = cross(b - a, c - a);
      let area = 0.5 * length(crossEdges);
      if (distance2 <= 0.0 || area <= 0.0) {
        return invalidMediumEmitterSample();
      }
      let distance = sqrt(distance2);
      let wi = toLight / distance;
      let cosLight = max(dot(safe_normalize(crossEdges), -wi), 0.0);
      if (cosLight <= 0.0) { return invalidMediumEmitterSample(); }
      return MediumEmitterSample(
        wi, emissionRadiance, distance,
        distance2 / (cosLight * area),
        false, emission.w > 0.5, true,
      );
    }
    current = current + 1u;
  }

  if (current == flat) {
    let env = sampleEnvironmentImportance(rng);
    if (env.pdf > 0.0) {
      return MediumEmitterSample(
        env.wi, env.value, INFINITY, env.pdf, false, false, true,
      );
    }
    return invalidMediumEmitterSample();
  }
  return invalidMediumEmitterSample();
}

fn mediumNeeForEmitter(
  flat: u32,
  selectionInvPdf: f32,
  position: vec3f,
  travelDirection: vec3f,
  layer: BdptMediumLayer,
  mediumStack: ptr<
    function,
    array<BdptMediumLayer, BDPT_MEDIUM_STACK_LIMIT>
  >,
  mediumDepth: u32,
  throughput: vec3f,
  heroLambda: f32,
  bdptCrossFamilyEnabled: bool,
  eyeDepth: u32,
  currentIncomingEyePdf: f32,
  rng: ptr<function, PtRngState>,
) -> vec3f {
  if (!(selectionInvPdf > 0.0)) { return vec3f(0.0); }
  let light = sampleMediumEmitter(flat, position, rng);
  if (!light.valid || light.pdf <= 0.0) { return vec3f(0.0); }
  let phasePdf = hgPhase(dot(travelDirection, light.wi), layer.g);
  let visibility = traceMediumVisibility(
    position, light.wi,
    select(light.distance, max(light.distance - 2e-3, 1e-4), light.distance < INFINITY * 0.5),
    layer, mediumStack, mediumDepth, heroLambda,
    light.ignoreOpaque, rng,
  );
  if (!visibility.visible) { return vec3f(0.0); }
  // A phase-sampled ray cannot terminate on an analytic directional cone:
  // mediumPhaseEmitterConnection owns only rect/mesh/environment endpoints.
  // Without BDPT there is therefore no complementary estimator to share this
  // soft-directional sample with, so its local NEE weight must remain one.
  let softDirectionalWithoutComplement =
    flat < params.directionalLightCount && !light.delta &&
    !bdptCrossFamilyEnabled;
  var mis = select(
    powerHeuristic(light.pdf, phasePdf), 1.0,
    light.delta || softDirectionalWithoutComplement,
  );
  if (bdptCrossFamilyEnabled) {
    let environmentGlobalIndex =
      params.directionalLightCount + params.pointLightCount +
      params.spotLightCount + params.rectAreaLightCount +
      params.meshAreaLightCount;
    let isEnvironment = flat == environmentGlobalIndex;
    let selectionPdf = 1.0 / selectionInvPdf;
    mis = bdptInfiniteEyeFamilyWeight(
      1u,
      isEnvironment,
      false,
      phasePdf,
      selectionPdf * light.pdf,
      bdptInfiniteRootLaunchPdf(light.pdf),
      light.wi,
      position,
      vec3f(0.0),
      true,
      currentIncomingEyePdf,
      phasePdf,
      false,
      eyeDepth,
    );
  }
  let spectralRadiance = select(
    light.radiance,
    spectralEmissionAtHero(light.radiance, heroLambda),
    params.spectralEnabled != 0u,
  );
  return throughput * visibility.transmittance * spectralRadiance *
    phasePdf * mis * selectionInvPdf / light.pdf;
}

fn mediumPhaseEmitterConnection(
  position: vec3f,
  travelDirection: vec3f,
  sampledDirection: vec3f,
  phasePdf: f32,
  layer: BdptMediumLayer,
  mediumStack: ptr<
    function,
    array<BdptMediumLayer, BDPT_MEDIUM_STACK_LIMIT>
  >,
  mediumDepth: u32,
  throughput: vec3f,
  heroLambda: f32,
  includeFiniteEmitters: bool,
  bdptCrossFamilyEnabled: bool,
  eyeDepth: u32,
  currentIncomingEyePdf: f32,
  rng: ptr<function, PtRngState>,
) -> vec3f {
  var bestDistance = INFINITY;
  var bestPdf = 0.0;
  var bestEmission = vec3f(0.0);
  var bestIgnoreOpaque = false;

  if (includeFiniteEmitters) {
  for (var ri = 0u; ri < params.rectAreaLightCount; ri = ri + 1u) {
    var distance = INFINITY;
    var lightPdf = 0.0;
    if (
      intersectRectAreaLightRay(
        ri, position, sampledDirection, &distance, &lightPdf,
      ) && distance < bestDistance
    ) {
      bestDistance = distance;
      bestPdf = lightPdf;
      bestEmission = rectAreaLights[ri * 4u + 3u].rgb;
      bestIgnoreOpaque = rectAreaLights[ri * 4u].w > 0.5;
    }
  }
  for (var mi = 0u; mi < params.meshAreaLightCount; mi = mi + 1u) {
    var distance = INFINITY;
    var lightPdf = 0.0;
    if (
      intersectMeshAreaLightRay(
        mi, position, sampledDirection, &distance, &lightPdf,
    ) && distance < bestDistance
    ) {
      bestDistance = distance;
      bestPdf = lightPdf;
      let lightPoint = position + sampledDirection * distance;
      let meshBase = meshAreaLightBase(mi);
      bestEmission = sampleMeshAreaLightRadiance(
        mi, meshAreaLightWeightsAtPoint(mi, lightPoint), lightPoint,
      );
      bestIgnoreOpaque = meshAreaLights[meshBase + 3u].w > 0.5;
    }
  }

  if (bestPdf > 0.0) {
    let visibility = traceMediumVisibility(
      position, sampledDirection, max(bestDistance - 2e-3, 1e-4),
      layer, mediumStack, mediumDepth, heroLambda,
      bestIgnoreOpaque, rng,
    );
    if (!visibility.visible) { return vec3f(0.0); }
    let emission = select(
      bestEmission, spectralEmissionAtHero(bestEmission, heroLambda),
      params.spectralEnabled != 0u,
    );
    return throughput * visibility.transmittance * emission *
      powerHeuristic(phasePdf, bestPdf);
  }
  }

  let envPdf = environmentPdf(sampledDirection);
  if (envPdf <= 0.0) { return vec3f(0.0); }
  let visibility = traceMediumVisibility(
    position, sampledDirection, INFINITY,
    layer, mediumStack, mediumDepth, heroLambda, false, rng,
  );
  if (!visibility.visible) { return vec3f(0.0); }
  let envRgb = sampleEnvironmentColor(sampledDirection);
  let emission = select(
    envRgb, spectralEmissionAtHero(envRgb, heroLambda),
    params.spectralEnabled != 0u,
  );
  var mis = powerHeuristic(phasePdf, envPdf);
  if (bdptCrossFamilyEnabled) {
    let environmentGlobalIndex =
      params.directionalLightCount + params.pointLightCount +
      params.spotLightCount + params.rectAreaLightCount +
      params.meshAreaLightCount;
    let selectionPdf =
      distantDirectSelectionPdf(environmentGlobalIndex);
    mis = bdptInfiniteEyeFamilyWeight(
      0u,
      true,
      false,
      phasePdf,
      selectionPdf * envPdf,
      bdptInfiniteRootLaunchPdf(envPdf),
      sampledDirection,
      position,
      vec3f(0.0),
      true,
      currentIncomingEyePdf,
      phasePdf,
      false,
      eyeDepth,
    );
  }
  return throughput * visibility.transmittance * emission * mis;
}
`;
