/**
 * Invocation-private BDPT light subpath — bounce 0 (source) + extensions k>0.
 * The source family is selected by an unbiased full-u32 uniform draw over every
 * finite and infinite endpoint. Directional/environment roots launch from the
 * scene-bounding disk and carry the distant-NEE density needed for cross-family
 * p0/p1/p2+ MIS with eye escape and direct-light replay.
 */
export const PT_WEBGPU_BDPT_LIGHT_SUBPATH_WGSL = /* wgsl */ `
fn bdptHasEnvironmentEndpoint() -> bool {
  return hasEnvironmentMap();
}

fn bdptEmitterCount() -> u32 {
  return params.directionalLightCount + params.pointLightCount +
    params.spotLightCount + params.rectAreaLightCount +
    params.meshAreaLightCount + select(0u, 1u, bdptHasEnvironmentEndpoint());
}

fn bdptRandomU32(rng: ptr<function, PtRngState>) -> u32 {
  // Sobol's primary dimensions expose 24 high-quality bits. Combine two draws
  // so both Sobol and PCG paths cover the complete u32 identity domain.
  let high24 = pcgNext(rng) & 0xffffff00u;
  let low8 = pcgNext(rng) >> 24u;
  return high24 | low8;
}

fn bdptPickEmitterFlat(rng: ptr<function, PtRngState>, emitterCount: u32) -> u32 {
  if (emitterCount <= 1u) { return 0u; }
  let threshold = ((0xffffffffu % emitterCount) + 1u) % emitterCount;
  loop {
    let word = bdptRandomU32(rng);
    if (word >= threshold) { return word % emitterCount; }
  }
  return 0u;
}

// Row 3 stores a real material id (>=0) or an explicit emitter endpoint kind.
// Negative kinds are deliberately distinct because point and spot endpoints
// have different directional measures and connection geometry.
// .xyz = wo toward the PREVIOUS light vertex (the eval's outgoing direction).
const BDPT_LV_EMITTER_MATID: f32 = -1.0;
const BDPT_LV_AREA_EMITTER_MATID: f32 = -2.0;
const BDPT_LV_POINT_EMITTER_MATID: f32 = -4.0;
const BDPT_LV_SPOT_EMITTER_MATID: f32 = -5.0;
const BDPT_LV_DIRECTIONAL_EMITTER_MATID: f32 = -8.0;
const BDPT_LV_ENVIRONMENT_EMITTER_MATID: f32 = -9.0;
fn bdptWriteLvBsdf(col: i32, matId: f32, woTowardPrev: vec3f) {
  bdptLightPath[bdptLightPathIndex(col, 3u)] = vec4f(woTowardPrev, matId);
}

// Texture-map payload for surface light vertices. Row 4 keeps the hit-local
// coordinate system needed by the same material sampling helpers as the eye path.
// The high bit of triIndex stores the front-face side for front/back layer parity.
fn bdptWriteLvMaterialPayload(col: i32, triIndex: u32, baryVW: vec2f, instanceIndex: u32, isFrontFace: bool) {
  let sideBit = select(0u, 0x80000000u, isFrontFace);
  bdptLightPath[bdptLightPathIndex(col, 4u)] = vec4f(bitcast<f32>((triIndex & 0x7fffffffu) | sideBit), baryVW.x, baryVW.y, bitcast<f32>(instanceIndex));
}

fn bdptWriteLvInterfaceEta(
  col: i32,
  etaTOverI: f32,
  incidentIor: f32,
  transmittedIor: f32,
) {
  bdptLightPath[bdptLightPathIndex(col, 5u)] =
    vec4f(etaTOverI, incidentIor, transmittedIor, 0.0);
}
fn bdptWriteLvMediumContext(col: i32, mediumMatId: u32) {
  let eta = bdptLightPath[bdptLightPathIndex(col, 5u)];
  bdptLightPath[bdptLightPathIndex(col, 5u)] =
    vec4f(eta.xyz, bitcast<f32>(mediumMatId));
}

fn bdptWriteLvMediumSides(
  col: i32,
  incidentMedium: BdptEndpointMedium,
  transmittedMedium: BdptEndpointMedium,
) {
  bdptLightPath[bdptLightPathIndex(col, 6u)] = vec4f(
    bitcast<f32>(incidentMedium.matId), incidentMedium.remainingDistance,
    bitcast<f32>(transmittedMedium.matId), transmittedMedium.remainingDistance,
  );
}

fn bdptWriteLvMediumPayload(col: i32, layer: BdptMediumLayer) {
  bdptLightPath[bdptLightPathIndex(col, 5u)] =
    vec4f(1.0, layer.g, 1.0, bitcast<f32>(layer.matId));
  let endpointMedium = BdptEndpointMedium(
    layer.matId, layer.remainingDistance,
  );
  bdptWriteLvMediumSides(col, endpointMedium, endpointMedium);
}

fn bdptClearLvMaterialPayload(col: i32) {
  bdptLightPath[bdptLightPathIndex(col, 4u)] = vec4f(0.0);
  bdptWriteLvInterfaceEta(col, 1.0, 1.0, 1.0);
  bdptWriteLvMediumContext(col, BDPT_NO_MEDIUM);
  bdptWriteLvMediumSides(col, bdptNoEndpointMedium(), bdptNoEndpointMedium());
}

// Bounce-0 emitter payload:
//   x = castShadowDisabled, y = cutoffDistance, z = decay, w = spot cosOuter.
// Spot cosInner lives in row 3.x; surface vertices overwrite row 4 with their
// hit-local material payload.
fn bdptWriteLvEmitterPayload(
  col: i32,
  castShadowDisabled: bool,
  cutoffDistance: f32,
  decay: f32,
  spotCosOuter: f32,
) {
  bdptLightPath[bdptLightPathIndex(col, 4u)] = vec4f(
    select(0.0, 1.0, castShadowDisabled), cutoffDistance, decay, spotCosOuter,
  );
}

fn bdptMaterialWithVolumeThickness(
  matId: u32, triIndex: u32, baryVW: vec2f, instanceIndex: u32,
) -> DecodedMaterial {
  var mat = decodeMaterial(matId);
  let thicknessSample = sampleVolumeThicknessTexture(
    matId, triIndex, baryVW, instanceIndex,
  );
  if (thicknessSample >= 0.0) {
    mat.volumeThickness = max(mat.volumeThickness * thicknessSample, 0.0);
    mat.hasVolumeThickness = true;
  }
  return mat;
}

struct BdptSampledMaterial {
  baseColor: vec3f,
  roughness: f32,
  metallic: f32,
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
  clearcoatNormal: vec3f,
  thinFilm: ThinFilmInterface,
}

fn bdptSampleMaterialAtPayload(matId: u32, payload: vec4f, shadingNormal: vec3f, woTowardPrev: vec3f, heroLambda: f32) -> BdptSampledMaterial {
  let triWord = bitcast<u32>(payload.x);
  let triIndex = triWord & 0x7fffffffu;
  let isFrontFace = (triWord & 0x80000000u) != 0u;
  let baryVW = payload.yz;
  let instanceIndex = bitcast<u32>(payload.w);
  let mat = bdptMaterialWithVolumeThickness(matId, triIndex, baryVW, instanceIndex);
  var out: BdptSampledMaterial;
  out.baseColor = mat.baseColor * sampleVertexColor(triIndex, baryVW).rgb * sampleBaseColorTexture(matId, triIndex, baryVW, instanceIndex).rgb;
  out.baseColor = out.baseColor * sampleAoFactor(matId, triIndex, baryVW, instanceIndex);
  let orm = sampleOrmTexture(matId, triIndex, baryVW, instanceIndex);
  out.roughness = clamp(mat.roughness * orm.g, 0.0, 1.0);
  out.metallic = clamp(mat.metallic * orm.b, 0.0, 1.0);
  out.transmission = clamp(mat.transmission * sampleTransmissionTexture(matId, triIndex, baryVW, instanceIndex), 0.0, 1.0);
  out.ior = mat.ior;
  if (params.spectralEnabled != 0u && mat.dispersionAbbe >= 1.0) {
    out.ior = cauchyIorAtLambda(heroLambda, mat.ior, mat.dispersionAbbe);
  }
  out.clearcoat = clamp(mat.clearcoat * sampleClearcoatTexture(matId, triIndex, baryVW, instanceIndex), 0.0, 1.0);
  out.clearcoatRoughness = clamp(mat.clearcoatRoughness * sampleClearcoatRoughnessTexture(matId, triIndex, baryVW, instanceIndex), 0.0, 1.0);
  out.sheen = mat.sheen;
  out.sheenRoughness = clamp(mat.sheenRoughness * sampleSheenRoughnessTexture(matId, triIndex, baryVW, instanceIndex), 0.0, 1.0);
  out.sheenColor = clamp(mat.sheenColor * sampleSheenColorTexture(matId, triIndex, baryVW, instanceIndex), vec3f(0.0), vec3f(1.0));
  out.iridescence = clamp(mat.iridescence * sampleIridescenceTexture(matId, triIndex, baryVW, instanceIndex), 0.0, 1.0);
  let iridescenceThicknessSample = sampleIridescenceThicknessTexture(matId, triIndex, baryVW, instanceIndex);
  out.iridescenceThicknessMin = mat.iridescenceThicknessMin;
  out.iridescenceThicknessMax = mat.iridescenceThicknessMax;
  if (iridescenceThicknessSample >= 0.0) {
    let iridescenceThickness = mix(mat.iridescenceThicknessMin, mat.iridescenceThicknessMax, iridescenceThicknessSample);
    out.iridescenceThicknessMin = iridescenceThickness;
    out.iridescenceThicknessMax = iridescenceThickness;
    if (iridescenceThickness <= 0.0) { out.iridescence = 0.0; }
  }
  out.iridescenceIor = mat.iridescenceIor;
  out.specularColor = max(
    mat.specularColor * sampleSpecularColorTexture(
      matId, triIndex, baryVW, instanceIndex,
    ),
    vec3f(0.0),
  );
  out.specularIntensity = clamp(mat.specularIntensity * sampleSpecularIntensityTexture(matId, triIndex, baryVW, instanceIndex), 0.0, 1.0);
  if (params.spectralEnabled != 0u) {
    out.sheenColor = vec3f(spectralRgbFactorAtHero(out.sheenColor, heroLambda));
    out.specularColor = vec3f(spectralRgbFactorAtHero(out.specularColor, heroLambda));
  }
  out.anisotropy = materialAnisotropy(matId, triIndex, baryVW, instanceIndex);
  out.anisotropyRotation = materialAnisotropyRotation(
    matId, triIndex, baryVW, shadingNormal, instanceIndex,
  );
  out.clearcoatNormal = applyClearcoatNormalMap(matId, triIndex, baryVW, shadingNormal, instanceIndex);
  out.thinFilm = ThinFilmInterface(
    mat.thinFilmEnabled,
    matId,
    mat.thinFilmLayerCountU,
    mat.thinFilmIncidentIor,
    out.ior,
    mat.thinFilmAngleDependent,
    isFrontFace,
    params.spectralEnabled != 0u,
    heroLambda,
    out.transmission,
  );
  let layerTx = clamp(select(mat.backLayerTx, mat.frontLayerTx, isFrontFace), vec3f(0.0), vec3f(1.0));
  let layerRoughness = select(mat.backLayerRoughness, mat.frontLayerRoughness, isFrontFace);
  if (layerRoughness >= 0.0) {
    out.roughness = clamp(layerRoughness, 0.0, 1.0);
  }
  let layerWeight = select(
    layerTx,
    activeLayerWeightRgb(layerTx, heroLambda, true),
    params.spectralEnabled != 0u && luminance(layerTx) < 0.999,
  );
  out.baseColor = out.baseColor * layerWeight;
  if (params.spectralEnabled != 0u) {
    let reflScalar = spectralCombinedReflectanceAtHero(
      out.baseColor,
      mat.baseColor,
      mat.spectralReflCoeffs,
      mat.hasSpectralReflectance,
      heroLambda,
    );
    out.baseColor = vec3f(reflScalar);
  }
  return out;
}

fn bdptWriteInvalid(col: i32) {
  bdptLightPath[bdptLightPathIndex(col, 0u)] = vec4f(0.0, 0.0, 0.0, BDPT_KIND_INVALID);
  bdptLightPath[bdptLightPathIndex(col, 1u)] = vec4f(0.0);
  bdptLightPath[bdptLightPathIndex(col, 2u)] = vec4f(0.0);
  bdptLightPath[bdptLightPathIndex(col, 3u)] = vec4f(0.0, 0.0, 0.0, BDPT_LV_EMITTER_MATID);
  bdptClearLvMaterialPayload(col);
}
fn bdptWriteMediumVertex(
  col: i32,
  pos: vec3f,
  woTowardPrev: vec3f,
  throughput: vec3f,
  pdfFwd: f32,
  pdfRev: f32,
  layer: BdptMediumLayer,
) {
  bdptLightPath[bdptLightPathIndex(col, 0u)] = vec4f(pos, 0.0);
  bdptLightPath[bdptLightPathIndex(col, 1u)] = vec4f(vec3f(0.0), pdfFwd);
  bdptLightPath[bdptLightPathIndex(col, 2u)] = vec4f(throughput, pdfRev);
  bdptWriteLvBsdf(col, BDPT_LV_MEDIUM_MATID, woTowardPrev);
  bdptLightPath[bdptLightPathIndex(col, 4u)] = vec4f(0.0);
  bdptWriteLvMediumPayload(col, layer);
}


// Bounce 0 samples only the finite endpoint position. Direction sampling belongs to the extension edge; this
// prevents its PDF from contaminating direct eye-to-emitter connections.
fn bdptFinishBounce0Endpoint(
  col: i32,
  emitPos: vec3f,
  emitAxis: vec3f,
  endpointData: vec3f,
  emitRad: vec3f,
  pdfPosition: f32,
  emitterKind: f32,
  castShadowDisabled: bool,
  cutoffDistance: f32,
  decay: f32,
  spotCosOuter: f32,
) {
  if (!(pdfPosition > 0.0) || pdfPosition != pdfPosition ||
      pdfPosition > bitcast<f32>(0x7f7fffffu)) {
    bdptWriteInvalid(col);
    return;
  }
  let pdfPos = pdfPosition;
  let spectralEmit = spectralEmissionAtHero(emitRad, bdptInvocationHeroLambdaNm);
  let emitThroughput = select(emitRad, spectralEmit, params.spectralEnabled != 0u) / pdfPos;
  bdptLightPath[bdptLightPathIndex(col, 0u)] = vec4f(emitPos, 0.0);
  bdptLightPath[bdptLightPathIndex(col, 1u)] = vec4f(emitAxis, pdfPos);
  bdptLightPath[bdptLightPathIndex(col, 2u)] = vec4f(emitThroughput, 0.0);
  // A valid endpoint must initialise both sides of row 6 to the vacuum
  // sentinel. Leaving a freshly allocated zero row here made material id 0
  // look like an authored medium; bdptSharedEdgeMedium then rejected every
  // c=0 endpoint edge against the eye-side vacuum and maxLightBounces:1
  // produced a black estimator.
  bdptClearLvMaterialPayload(col);
  bdptWriteLvBsdf(col, emitterKind, endpointData);
  bdptWriteLvEmitterPayload(
    col,
    castShadowDisabled,
    cutoffDistance,
    decay,
    spotCosOuter,
  );
}

struct BdptInfiniteLaunch {
  position: vec3f,
  towardSource: vec3f,
  travelDirection: vec3f,
  positionPdf: f32,
}

fn bdptInfiniteLaunchDisk(
  towardSourceIn: vec3f,
  rng: ptr<function, PtRngState>,
) -> BdptInfiniteLaunch {
  let towardSource = safe_normalize(towardSourceIn);
  let radius = max(params.sceneRadius, 1e-3);
  let center = vec3f(
    params.sceneCenterX, params.sceneCenterY, params.sceneCenterZ,
  ) + towardSource * radius;
  var tangent: vec3f;
  var bitangent: vec3f;
  buildOnb(towardSource, &tangent, &bitangent);
  let disc = concentricDiscSample(
    vec2f(rand_f32(rng), rand_f32(rng)) * 2.0 - vec2f(1.0),
  );
  return BdptInfiniteLaunch(
    center + radius * (disc.x * tangent + disc.y * bitangent),
    towardSource,
    -towardSource,
    1.0 / (PI * radius * radius),
  );
}

fn bdptSampleDirectionalCone(
  rng: ptr<function, PtRngState>,
  axisIn: vec3f,
  angularDiameter: f32,
) -> vec3f {
  let axis = safe_normalize(axisIn);
  if (angularDiameter <= 0.0) { return axis; }
  let cosHalfAngle = cos(angularDiameter * 0.5);
  let cosTheta = mix(cosHalfAngle, 1.0, rand_f32(rng));
  let sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
  let phi = 2.0 * PI * rand_f32(rng);
  var tangent: vec3f;
  var bitangent: vec3f;
  buildOnb(axis, &tangent, &bitangent);
  return safe_normalize(
    sinTheta * cos(phi) * tangent +
    sinTheta * sin(phi) * bitangent +
    cosTheta * axis,
  );
}

fn bdptFinishInfiniteEndpoint(
  col: i32,
  launch: BdptInfiniteLaunch,
  radiance: vec3f,
  selectionPdf: f32,
  directionPdf: f32,
  neePdf: f32,
  sourceDirectionWeight: f32,
  emitterKind: f32,
  directionIsDelta: bool,
  castShadowDisabled: bool,
) {
  let pdfPosition = selectionPdf * launch.positionPdf;
  if (!(pdfPosition > 0.0) || pdfPosition != pdfPosition ||
      pdfPosition > bitcast<f32>(0x7f7fffffu) ||
      !(directionPdf > 0.0) || directionPdf != directionPdf ||
      directionPdf > bitcast<f32>(0x7f7fffffu) ||
      !(neePdf > 0.0) || neePdf != neePdf ||
      neePdf > bitcast<f32>(0x7f7fffffu) ||
      !(sourceDirectionWeight > 0.0) ||
      sourceDirectionWeight != sourceDirectionWeight ||
      sourceDirectionWeight > bitcast<f32>(0x7f7fffffu)) {
    bdptWriteInvalid(col);
    return;
  }
  let spectralEmit = spectralEmissionAtHero(
    radiance, bdptInvocationHeroLambdaNm,
  );
  let emitted = select(
    radiance, spectralEmit, params.spectralEnabled != 0u,
  );
  bdptLightPath[bdptLightPathIndex(col, 0u)] = vec4f(
    launch.position,
    select(0.0, BDPT_KIND_DELTA, directionIsDelta),
  );
  bdptLightPath[bdptLightPathIndex(col, 1u)] =
    vec4f(launch.towardSource, pdfPosition);
  bdptLightPath[bdptLightPathIndex(col, 2u)] =
    vec4f(emitted * sourceDirectionWeight / pdfPosition, 0.0);
  bdptClearLvMaterialPayload(col);
  bdptWriteLvBsdf(col, emitterKind, launch.travelDirection);
  bdptLightPath[bdptLightPathIndex(col, 4u)] = vec4f(
    select(0.0, 1.0, castShadowDisabled),
    directionPdf,
    neePdf,
    select(0.0, 1.0, directionIsDelta),
  );
}

fn bdptWriteBounce0(col: i32, rng: ptr<function, PtRngState>) {
  let emitterCount = bdptEmitterCount();
  if (emitterCount == 0u) {
    bdptWriteInvalid(col);
    return;
  }
  let flat = bdptPickEmitterFlat(rng, emitterCount);
  let discretePdf = 1.0 / f32(emitterCount);

  var cur = 0u;
  for (var di = 0u; di < params.directionalLightCount; di = di + 1u) {
    if (cur == flat) {
      let directional = directionalLights[di * 2u];
      let irradiance = directionalLights[di * 2u + 1u].rgb;
      let rawAngularDiameter = directional.w;
      let castShadowDisabled = rawAngularDiameter < 0.0;
      let angularDiameter = max(select(
        rawAngularDiameter, -1.0 - rawAngularDiameter, castShadowDisabled,
      ), 0.0);
      let towardSource = bdptSampleDirectionalCone(
        rng, directional.xyz, angularDiameter,
      );
      let launch = bdptInfiniteLaunchDisk(towardSource, rng);
      let directionIsDelta = angularDiameter <= 0.0;
      // 1-cos(d/2) = 2*sin(d/4)^2 avoids catastrophic cancellation for
      // sun-sized and smaller authored cones while preserving the exact cone
      // solid-angle measure.
      var directionPdf = 1.0;
      if (!directionIsDelta) {
        let coneQuarterSin = sin(angularDiameter * 0.25);
        directionPdf =
          1.0 / (4.0 * PI * coneQuarterSin * coneQuarterSin);
      }
      // Packed directional RGB is irradiance, not radiance. The authored
      // source is invariant to the soft-shadow cone size, so its directional
      // measure carries p_dir and cancels the extension's 1/p_dir factor.
      let sourceDirectionWeight = select(
        1.0, directionPdf, !directionIsDelta,
      );
      bdptFinishInfiniteEndpoint(
        col, launch, irradiance, discretePdf, directionPdf,
        distantDirectSelectionPdf(cur) * directionPdf,
        sourceDirectionWeight,
        BDPT_LV_DIRECTIONAL_EMITTER_MATID,
        directionIsDelta, castShadowDisabled,
      );
      return;
    }
    cur = cur + 1u;
  }
  for (var pi = 0u; pi < params.pointLightCount; pi = pi + 1u) {
    if (cur == flat) {
      // H51-D: stride 3; position at slot 0, radiance at slot 1
      let pos = pointLights[pi * 3u].xyz;
      let rad = pointLights[pi * 3u + 1u].rgb;
      let ptExtra = pointLights[pi * 3u + 2u];
      bdptFinishBounce0Endpoint(
        col,
        pos,
        vec3f(0.0, 1.0, 0.0),
        vec3f(0.0),
        rad,
        discretePdf,
        BDPT_LV_POINT_EMITTER_MATID,
        ptExtra.z > 0.5,
        ptExtra.x,
        ptExtra.y,
        0.0,
      );
      return;
    }
    cur = cur + 1u;
  }
  for (var si = 0u; si < params.spotLightCount; si = si + 1u) {
    if (cur == flat) {
      // H51-D: stride 4; position at slot 0, dir+cosOuter at slot 1, radiance+cosInner at slot 2
      let sb = si * 4u;
      let spos = spotLights[sb].xyz;
      let saxis = spotLights[sb + 1u];
      let sradW = spotLights[sb + 2u];
      let spotDir = safe_normalize(saxis.xyz);
      let spExtra = spotLights[sb + 3u];
      bdptFinishBounce0Endpoint(
        col,
        spos,
        spotDir,
        vec3f(sradW.w, 0.0, 0.0),
        sradW.rgb,
        discretePdf,
        BDPT_LV_SPOT_EMITTER_MATID,
        spExtra.z > 0.5,
        spExtra.x,
        spExtra.y,
        saxis.w,
      );
      return;
    }
    cur = cur + 1u;
  }
  for (var ri = 0u; ri < params.rectAreaLightCount; ri = ri + 1u) {
    if (cur == flat) {
      let rb = ri * 4u;
      let rbase = rectAreaLights[rb];
      let rpos = rbase.xyz;
      let ru = rectAreaLights[rb + 1u].xyz;
      let rv = rectAreaLights[rb + 2u].xyz;
      let rshapeS = rectAreaLights[rb + 3u];
      let rr = rshapeS.rgb;
      let isDiscS = abs(rshapeS.w - 1.0) < 0.5;
      let xi1s = rand_f32(rng);
      let xi2s = rand_f32(rng);
      var emitPos: vec3f;
      var areaS: f32;
      if (isDiscS) {
        let disc = concentricDiscSample(
          vec2f(xi1s * 2.0 - 1.0, xi2s * 2.0 - 1.0),
        );
        emitPos = rpos + ru * disc.x + rv * disc.y;
        areaS = PI * length(cross(ru, rv));
      } else {
        emitPos = rpos + ru * (xi1s * 2.0 - 1.0) + rv * (xi2s * 2.0 - 1.0);
        areaS = 4.0 * length(cross(ru, rv));
      }
      if (areaS <= 0.0) {
        bdptWriteInvalid(col);
        return;
      }
      let emitNormal = safe_normalize(cross(ru, rv));
      bdptFinishBounce0Endpoint(
        col,
        emitPos,
        emitNormal,
        emitNormal,
        rr,
        discretePdf / areaS,
        BDPT_LV_AREA_EMITTER_MATID,
        rbase.w > 0.5,
        0.0,
        0.0,
        0.0,
      );
      return;
    }
    cur = cur + 1u;
  }
  for (var mi = 0u; mi < params.meshAreaLightCount; mi = mi + 1u) {
    if (cur == flat) {
      let mb = meshAreaLightBase(mi);
      let a = meshAreaLights[mb].xyz;
      let b = meshAreaLights[mb + 1u].xyz;
      let c = meshAreaLights[mb + 2u].xyz;
      let r1 = rand_f32(rng);
      let r2 = rand_f32(rng);
      let su = sqrt(r1);
      let uu = 1.0 - su;
      let vv = r2 * su;
      let ww = 1.0 - uu - vv;
      let emitPos = a * uu + b * vv + c * ww;
      let mr = sampleMeshAreaLightRadiance(
        mi, vec3f(uu, vv, ww), emitPos,
      );
      let e1 = b - a;
      let e2 = c - a;
      let n = cross(e1, e2);
      let nLen = length(n);
      if (!(nLen > 0.0)) {
        bdptWriteInvalid(col);
        return;
      }
      let emitNormal = n / nLen;
      let areaM = 0.5 * nLen;
      bdptFinishBounce0Endpoint(
        col,
        emitPos,
        emitNormal,
        emitNormal,
        mr,
        discretePdf / areaM,
        BDPT_LV_AREA_EMITTER_MATID,
        meshAreaLights[mb + 3u].w > 0.5,
        0.0,
        0.0,
        0.0,
      );
      return;
    }
    cur = cur + 1u;
  }
  if (bdptHasEnvironmentEndpoint() && cur == flat) {
    let environment = sampleEnvironmentImportance(rng);
    let towardSource = environment.wi;
    let radiance = environment.value;
    let directionPdf = environment.pdf;
    if (!(directionPdf > 0.0) || any(radiance < vec3f(0.0))) {
      bdptWriteInvalid(col);
      return;
    }
    let launch = bdptInfiniteLaunchDisk(towardSource, rng);
    bdptFinishInfiniteEndpoint(
      col, launch, radiance, discretePdf, directionPdf,
      distantDirectSelectionPdf(cur) * directionPdf, 1.0,
      BDPT_LV_ENVIRONMENT_EMITTER_MATID, false, false,
    );
    return;
  }
  bdptWriteInvalid(col);
}

fn bdptBuildInvocationLightSubpath(pixel: vec2u) {
  if (params.bdptEnabled == 0u) {
    return;
  }
  // One sequential Markov chain per path-trace invocation. The fixed private
  // stack is bounded to eight vertices and never scales with viewport size.
  let maxB = i32(min(params.bdptMaxLightBounces, BDPT_MAX_LIGHT_DEPTH));
  let seed = ptRngFrameKey(params.frameSeed ^ 0xb17d5eedu, params.frameIndex);

  // Column 0 — the emitter vertex.
  var rng0 = pcgInit(pixel.x, pixel.y, seed);
  bdptWriteBounce0(0, &rng0);

  var mediumStack: array<BdptMediumLayer, BDPT_MEDIUM_STACK_LIMIT>;
  var mediumDepth = 0u;

  // Columns 1..maxB-1 — extend from the previous column (written just above /
  // last loop iteration, by THIS same invocation: no inter-workgroup hazard).
  for (var col = 1; col < maxB; col = col + 1) {
    var rng = pcgInit(
      pixel.x ^ (u32(col) * 0x9e3779b9u),
      pixel.y ^ (u32(col) * 0x85ebca6bu),
      seed,
    );
    let prevCol = col - 1;
    let v0prev = bdptLightPath[bdptLightPathIndex(prevCol, 0u)];
    let v1prev = bdptLightPath[bdptLightPathIndex(prevCol, 1u)];
    let v2prev = bdptLightPath[bdptLightPathIndex(prevCol, 2u)];
    // Row 3 of prevCol: .xyz = woAtPrev (outgoing direction at prevPos toward the
    // vertex before it), .w = prevMatId (< 0 for emitter, >= 0 for surface).
    let v3prev = bdptLightPath[bdptLightPathIndex(prevCol, 3u)];
    if (v0prev.w == BDPT_KIND_INVALID) {
      bdptWriteInvalid(col);
      continue;
    }
    let prevPos = v0prev.xyz;
    let prevNormal = v1prev.xyz;
    let prevThroughput = v2prev.xyz;
    let woAtPrev = v3prev.xyz;    // outgoing direction at prevPos (toward its own predecessor)
    let prevMatId = v3prev.w;     // < 0 = emitter vertex, >= 0 = surface vertex

    // Surface extensions use the shared BounceSample for direction, event
    // density, throughput, ray offset, and delta classification.
    var scatterDir = vec3f(0.0);
    var pdfScatter = 0.0;
    var fPrev = vec3f(0.0);
    var cosPrev = 0.0;
    var surfaceThroughputMul = vec3f(0.0);
    var surfaceRayOrigin = prevPos;
    var sampledDelta = false;

    if (prevMatId == BDPT_LV_MEDIUM_MATID) {
      if (mediumDepth == 0u) {
        bdptWriteInvalid(col);
        continue;
      }
      let medium = mediumStack[mediumDepth - 1u];
      let travelIn = -woAtPrev;
      scatterDir =
        sampleHenyeyGreenstein(&rng, travelIn, medium.g);
      pdfScatter = hgPhase(dot(travelIn, scatterDir), medium.g);
      surfaceThroughputMul = vec3f(1.0);
      surfaceRayOrigin = prevPos;
    } else if (prevMatId < 0.0) {
      if (
        prevMatId == BDPT_LV_DIRECTIONAL_EMITTER_MATID ||
        prevMatId == BDPT_LV_ENVIRONMENT_EMITTER_MATID
      ) {
        let emitterPayload =
          bdptLightPath[bdptLightPathIndex(prevCol, 4u)];
        scatterDir = safe_normalize(woAtPrev);
        pdfScatter = emitterPayload.y;
        cosPrev = 1.0;
        fPrev = vec3f(1.0);
      } else if (prevMatId == BDPT_LV_AREA_EMITTER_MATID) {
        let hemi = cosineHemisphereSample(&rng, prevNormal);
        scatterDir = hemi.wi;
        pdfScatter = hemi.pdf;
        cosPrev = max(dot(prevNormal, scatterDir), 0.0);
        fPrev = vec3f(1.0);
      } else if (prevMatId == BDPT_LV_POINT_EMITTER_MATID) {
        scatterDir = uniformSphere(vec2f(rand_f32(&rng), rand_f32(&rng)));
        pdfScatter = 0.25 * INV_PI;
        cosPrev = 1.0;
        fPrev = vec3f(1.0);
      } else if (prevMatId == BDPT_LV_SPOT_EMITTER_MATID) {
        let emitterPayload = bdptLightPath[bdptLightPathIndex(prevCol, 4u)];
        let cosOuter = emitterPayload.w;
        let cosInner = woAtPrev.x;
        let cosTheta = mix(cosOuter, 1.0, rand_f32(&rng));
        let sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
        let phi = 2.0 * PI * rand_f32(&rng);
        var st: vec3f;
        var sb: vec3f;
        buildOnb(prevNormal, &st, &sb);
        scatterDir = safe_normalize(
          sinTheta * cos(phi) * st + sinTheta * sin(phi) * sb + cosTheta * prevNormal,
        );
        let solidAngle = 2.0 * PI * (1.0 - cosOuter);
        pdfScatter = select(0.0, 1.0 / solidAngle, solidAngle > 0.0);
        let softness = smoothstep(cosOuter, max(cosInner, cosOuter + 1e-6), cosTheta);
        fPrev = vec3f(softness);
        cosPrev = 1.0;

      } else {
        let hemi = cosineHemisphereSample(&rng, prevNormal);
        scatterDir = hemi.wi;
        pdfScatter = hemi.pdf;
        cosPrev = max(dot(prevNormal, scatterDir), 0.0);
        fPrev = vec3f(INV_PI);
      }
    } else {
      // Surface extension must use the event returned by the shared sampler.
      // Re-evaluating an opaque BRDF here discarded transmission, its discrete
      // Fresnel probability, and the sampler's side-correct ray offset.
      let prevPayload = bdptLightPath[bdptLightPathIndex(prevCol, 4u)];
      let prevInterface = bdptLightPath[bdptLightPathIndex(prevCol, 5u)];
      let prevEtaTOverI = max(prevInterface.x, 1e-4);
      let prevTriWordForMaterial = bitcast<u32>(prevPayload.x);
      let prevDecodedMat = bdptMaterialWithVolumeThickness(
        u32(prevMatId), prevTriWordForMaterial & 0x7fffffffu,
        prevPayload.yz, bitcast<u32>(prevPayload.w),
      );
      let prevMat = bdptSampleMaterialAtPayload(
        u32(prevMatId), prevPayload, prevNormal, woAtPrev, bdptInvocationHeroLambdaNm,
      );
      let prevRough = prevMat.roughness;
      let cosOPrev = max(dot(prevNormal, woAtPrev), 0.0);
      let f0BasePrev = materialSpecularF0(
        prevMat.baseColor,
        prevMat.metallic,
        prevEtaTOverI,
        prevMat.specularColor,
        prevMat.specularIntensity,
      );
      let f0Prev = iridescenceModifiedF0(
        f0BasePrev,
        prevMat.iridescence,
        prevMat.iridescenceIor,
        prevMat.iridescenceThicknessMin,
        prevMat.iridescenceThicknessMax,
        cosOPrev,
      );
      let prevTriWord = bitcast<u32>(prevPayload.x);
      let prevIsFrontFace = (prevTriWord & 0x80000000u) != 0u;
      let prevBoundary = mediumBoundaryIdentity(
        prevTriWord & 0x7fffffffu,
        bitcast<u32>(prevPayload.w),
      );
      let prevCrossesMedium =
        prevDecodedMat.isTranslucent && prevMat.transmission > 0.0;
      if (prevCrossesMedium && !mediumBoundaryIsValid(prevBoundary)) {
        bdptWriteInvalid(col);
        continue;
      }
      let prevGeometricNormal = select(-prevNormal, prevNormal, prevIsFrontFace);
      let prevThinFilm = prevMat.thinFilm;
      let bsPrev = sampleNextBounceDirectionWithClearcoatNormal(
        &rng,
        -woAtPrev,
        prevPos,
        prevGeometricNormal,
        prevNormal,
        prevMat.clearcoatNormal,
        prevMat.baseColor,
        prevRough,
        prevMat.metallic,
        prevMat.transmission,
        prevEtaTOverI,
        true,
        materialSpecularFresnelSchlick(
          cosOPrev, f0Prev, prevMat.metallic, prevMat.specularIntensity,
        ),
        prevMat.iridescence,
        prevMat.iridescenceIor,
        prevMat.iridescenceThicknessMin,
        prevMat.iridescenceThicknessMax,
        prevMat.specularColor,
        prevMat.specularIntensity,
        prevThinFilm,
        prevDecodedMat.isTranslucent,
        prevMat.clearcoat,
        prevMat.clearcoatRoughness,
        prevMat.sheen,
        prevMat.sheenRoughness,
        prevMat.sheenColor,
        prevMat.anisotropy,
        prevMat.anisotropyRotation,
      );
      scatterDir = bsPrev.sampledDir;
      pdfScatter = bsPrev.sampledEventPdf;
      surfaceThroughputMul = bsPrev.throughputMul;
      surfaceRayOrigin = bsPrev.newRayOrigin;
      sampledDelta = bsPrev.sampledIsDelta;

      if (bsPrev.enteredMedium) {
        if (mediumDepth >= BDPT_MEDIUM_STACK_LIMIT) {
          bdptWriteInvalid(col);
          continue;
        }
        mediumStack[mediumDepth] = bdptMediumLayer(
          u32(prevMatId), prevDecodedMat, bdptInvocationHeroLambdaNm,
          prevBoundary,
        );
        mediumDepth = mediumDepth + 1u;
      } else if (bsPrev.exitedMedium) {
        if (
          mediumDepth == 0u ||
          !bdptMediumLayerMatchesBoundary(
            mediumStack[mediumDepth - 1u],
            u32(prevMatId),
            prevBoundary,
          )
        ) {
          bdptWriteInvalid(col);
          continue;
        }
        mediumDepth = mediumDepth - 1u;
      }
    }

    if (pdfScatter <= 0.0 || (prevMatId < 0.0 && prevMatId != BDPT_LV_MEDIUM_MATID && cosPrev <= 1e-5)) {
      bdptWriteInvalid(col);
      continue;
    }

    // PBRT RandomWalk reverse-density convention: the swapped directional PDF
    // evaluated at L_{i-1} after sampling L_i describes reverse ARRIVAL at
    // L_{i-2}, not at L_{i-1}. L_{i-1}.row2.w was seeded when that vertex was
    // created with the reverse distance/medium density of edge
    // L_{i-2}<->L_{i-1}; combine the two factors and patch L_{i-2}. The final
    // two vertices adjacent to a connection are replaced by the straddle
    // overrides in bdptConnection.wgsl.ts, so their intermediate rows are safe.
    if (prevCol >= 1) {
      var swappedDirectionalPdf = 0.0;
      if (prevMatId == BDPT_LV_MEDIUM_MATID) {
        let prevMediumPayload =
          bdptLightPath[bdptLightPathIndex(prevCol, 5u)];
        swappedDirectionalPdf = hgPhase(
          dot(-woAtPrev, scatterDir), prevMediumPayload.y,
        );
      } else if (prevMatId >= 0.0 && !sampledDelta) {
        let prevPayloadForRev =
          bdptLightPath[bdptLightPathIndex(prevCol, 4u)];
        let prevInterfaceForRev =
          bdptLightPath[bdptLightPathIndex(prevCol, 5u)];
        let prevMatForRev = bdptSampleMaterialAtPayload(
          u32(prevMatId), prevPayloadForRev, prevNormal, woAtPrev,
          bdptInvocationHeroLambdaNm,
        );
        swappedDirectionalPdf = bdptMarginalSurfacePdf(
          prevMatForRev.baseColor, prevMatForRev.roughness,
          prevMatForRev.metallic, prevMatForRev.transmission,
          max(prevInterfaceForRev.x, 1e-4), prevNormal,
          prevMatForRev.clearcoatNormal, scatterDir, woAtPrev,
          prevMatForRev.clearcoat, prevMatForRev.clearcoatRoughness,
          prevMatForRev.sheen, prevMatForRev.sheenRoughness,
          prevMatForRev.iridescence, prevMatForRev.iridescenceIor,
          prevMatForRev.iridescenceThicknessMin,
          prevMatForRev.iridescenceThicknessMax,
          prevMatForRev.specularColor, prevMatForRev.specularIntensity,
          prevMatForRev.anisotropy, prevMatForRev.anisotropyRotation,
          prevMatForRev.thinFilm,
        );
      }
      let reverseEdgeDensity = max(v2prev.w, 0.0);
      let predecessorCol = prevCol - 1;
      let predecessorRow2 =
        bdptLightPath[bdptLightPathIndex(predecessorCol, 2u)];
      bdptLightPath[bdptLightPathIndex(predecessorCol, 2u)] = vec4f(
        predecessorRow2.xyz,
        swappedDirectionalPdf * reverseEdgeDensity,
      );
    }
    if (sampledDelta) {
      let oldPrevKind = bdptLightPath[bdptLightPathIndex(prevCol, 0u)];
      bdptLightPath[bdptLightPathIndex(prevCol, 0u)] =
        vec4f(oldPrevKind.xyz, BDPT_KIND_DELTA);
    }

    var ray: Ray;
    let emitterRayOrigin = prevPos + scatterDir * 1e-4;
    ray.origin = select(surfaceRayOrigin, emitterRayOrigin, prevMatId < 0.0 && prevMatId != BDPT_LV_MEDIUM_MATID);
    ray.direction = scatterDir;
    let alphaTraceOrigin = ray.origin;
    var alphaAdvance = 0.0;
    var hit = traceClosest(ray, 1e-4, 1e30);
    let alphaSurfaceHitLimit = sceneSurfaceHitLimit();
    var alphaSurfaceHitCount = 0u;
    var alphaTraversalValid = true;
    loop {
      if (!hit.didHit) { break; }
      if (alphaSurfaceHitCount >= alphaSurfaceHitLimit) {
        alphaTraversalValid = false;
        break;
      }
      alphaSurfaceHitCount = alphaSurfaceHitCount + 1u;
      if (!alphaTestPassThrough(
        hitMaterialId(hit), hit.triIndex, hit.baryVW, hit.instanceIndex, &rng,
      )) { break; }
      let alphaStep = hit.dist + 1e-4;
      alphaAdvance = alphaAdvance + alphaStep;
      ray.origin = ray.origin + ray.direction * alphaStep;
      hit = traceClosest(ray, 1e-4, 1e30);
    }
    if (!alphaTraversalValid) {
      bdptWriteInvalid(col);
      continue;
    }
    ray.origin = alphaTraceOrigin;
    if (hit.didHit) { hit.dist = hit.dist + alphaAdvance; }
    var segmentForwardDensity = 1.0;
    var segmentReverseDensity = 1.0;
    var segmentWeight = vec3f(1.0);
    var mediumCollision = false;
    var collisionDistance = 0.0;
    // If an emitter/camera starts inside a closed volume there is no preceding
    // entry interface to seed the stack. A first back-face hit identifies the
    // homogeneous layer for this segment; sample it exactly like an explicit
    // entered layer instead of applying an eye/light-asymmetric Beer fallback.
    if (mediumDepth == 0u && hit.didHit) {
      let inferredMatId = hitMaterialId(hit);
      let inferredMat = bdptMaterialWithVolumeThickness(
        inferredMatId, hit.triIndex, hit.baryVW, hit.instanceIndex,
      );
      let inferredBackFace = !hit.frontFace;
      if (inferredBackFace && inferredMat.isTranslucent) {
        let inferredBoundary = mediumBoundaryIdentity(
          hit.triIndex, hit.instanceIndex,
        );
        if (!mediumBoundaryIsValid(inferredBoundary)) {
          bdptWriteInvalid(col);
          continue;
        }
        mediumStack[0u] =
          bdptMediumLayer(
            inferredMatId, inferredMat, bdptInvocationHeroLambdaNm,
            inferredBoundary,
          );
        mediumDepth = 1u;
      }
    }
    if (mediumDepth > 0u) {
      let topIndex = mediumDepth - 1u;
      let layer = mediumStack[topIndex];
      let heroSigmaT = bdptHeroSigmaT(layer.sigmaT);
      if (heroSigmaT > 0.0) {
        let surfaceDistance = select(1e30, hit.dist, hit.didHit);
        let segmentLimit = min(surfaceDistance, layer.remainingDistance);
        let xiFlight = rand_f32(&rng);
        let freeFlightDistance =
          -log(max(1.0 - xiFlight, 1e-9)) / heroSigmaT;
        collisionDistance = min(freeFlightDistance, segmentLimit);
        let heroSurvival = exp(-heroSigmaT * collisionDistance);
        let transmittance =
          exp(-layer.sigmaT * collisionDistance);
        mediumCollision = freeFlightDistance < segmentLimit;
        segmentForwardDensity = select(
          heroSurvival,
          heroSigmaT * heroSurvival,
          mediumCollision,
        );
        segmentReverseDensity = select(
          heroSurvival,
          heroSigmaT * heroSurvival,
          prevMatId == BDPT_LV_MEDIUM_MATID,
        );
        segmentWeight = select(
          transmittance / max(heroSurvival, 1e-20),
          layer.sigmaS * transmittance /
            max(heroSigmaT * heroSurvival, 1e-20),
          mediumCollision,
        );
        mediumStack[topIndex].remainingDistance =
          max(layer.remainingDistance - collisionDistance, 0.0);
      }
    }
    if (mediumCollision) {
      let mediumVertexPos =
        ray.origin + ray.direction * collisionDistance;
      let mediumLayerAtVertex = mediumStack[mediumDepth - 1u];
      let pdfFwdMedium = pdfScatter * segmentForwardDensity;
      let mediumThroughput =
        prevThroughput * surfaceThroughputMul * segmentWeight;
      bdptWriteMediumVertex(
        col, mediumVertexPos, -scatterDir, mediumThroughput,
        pdfFwdMedium, segmentReverseDensity,
        mediumLayerAtVertex,
      );
      continue;
    }
    if (!hit.didHit) {
      bdptWriteInvalid(col);
      continue;
    }
    let matIdx = hitMaterialId(hit);
    let isFrontFaceHit = hit.frontFace;
    let hitDecodedMat = bdptMaterialWithVolumeThickness(
      matIdx, hit.triIndex, hit.baryVW, hit.instanceIndex,
    );
    let hitBoundary = mediumBoundaryIdentity(hit.triIndex, hit.instanceIndex);
    if (hitDecodedMat.isTranslucent && !mediumBoundaryIsValid(hitBoundary)) {
      bdptWriteInvalid(col);
      continue;
    }

    var incidentIor = 1.0;
    if (mediumDepth > 0u) {
      incidentIor = mediumStack[mediumDepth - 1u].ior;
    } else if (!isFrontFaceHit) {
      incidentIor = max(hitDecodedMat.ior, 1e-4);
    }
    var transmittedIor = max(hitDecodedMat.ior, 1e-4);
    if (!isFrontFaceHit) {
      transmittedIor = 1.0;
      if (hitDecodedMat.isTranslucent) {
        if (
          mediumDepth == 0u ||
          !bdptMediumLayerMatchesBoundary(
            mediumStack[mediumDepth - 1u], matIdx, hitBoundary,
          )
        ) {
          bdptWriteInvalid(col);
          continue;
        }
        if (mediumDepth > 1u) {
          transmittedIor = mediumStack[mediumDepth - 2u].ior;
        }
      }
    }
    let hitEtaTOverI = transmittedIor / max(incidentIor, 1e-4);
    let newPos = ray.origin + ray.direction * hit.dist;
    var incidentMedium = bdptNoEndpointMedium();
    if (mediumDepth > 0u) {
      let incidentLayer = mediumStack[mediumDepth - 1u];
      incidentMedium = BdptEndpointMedium(
        incidentLayer.matId, incidentLayer.remainingDistance,
      );
    }
    var transmittedMedium = bdptNoEndpointMedium();
    if (hitDecodedMat.isTranslucent) {
      if (isFrontFaceHit) {
        let enteredLayer = bdptMediumLayer(
          matIdx, hitDecodedMat, bdptInvocationHeroLambdaNm, hitBoundary,
        );
        transmittedMedium = BdptEndpointMedium(
          enteredLayer.matId, enteredLayer.remainingDistance,
        );
      } else {
        if (mediumDepth > 1u) {
          let belowLayer = mediumStack[mediumDepth - 2u];
          transmittedMedium = BdptEndpointMedium(
            belowLayer.matId, belowLayer.remainingDistance,
          );
        }
      }
    }
    let newNormal = safe_normalize(hit.normal);
    // Front-relative shading normal at the new vertex (toward the incoming light dir).
    var nsFront = select(-newNormal, newNormal, isFrontFaceHit);
    nsFront = applyNormalMap(matIdx, hit.triIndex, hit.baryVW, nsFront, hit.instanceIndex, isFrontFaceHit);
    nsFront = applyBumpMap(matIdx, hit.triIndex, hit.baryVW, nsFront, hit.instanceIndex);
    // Outgoing direction at newPos toward the previous vertex (= -scatterDir).
    let woLp = -scatterDir;

    // Throughput update: carry the prefix throughput * f·|cos|/pdf of THIS traced
    // segment. pdfFwd = generation density of scatterDir at prevPos (SA measure,
    // no baked-in geometry term — the §10.3 ConvertDensity handles SA→area).
    let pdfFwd = pdfScatter * segmentForwardDensity;
    var newThroughput = prevThroughput * surfaceThroughputMul * segmentWeight;
    if (prevMatId < 0.0 && prevMatId != BDPT_LV_MEDIUM_MATID) {
      // Emitter events are represented explicitly by f*cos/pdf above.
      newThroughput = prevThroughput * fPrev * cosPrev / pdfScatter * segmentWeight;
    }
    if (
      prevCol == 0 &&
      (prevMatId == BDPT_LV_POINT_EMITTER_MATID || prevMatId == BDPT_LV_SPOT_EMITTER_MATID)
    ) {
      let emitterPayload = bdptLightPath[bdptLightPathIndex(prevCol, 4u)];
      newThroughput = newThroughput * pointSpotPathMeasureScale(
        hit.dist, emitterPayload.y, emitterPayload.z,
      );
    }

    // row2.w initially holds the reverse distance/collision density of the
    // incoming edge. Building the successor evaluates the swapped directional
    // density at this vertex and moves the product to its predecessor, matching
    // PBRT's RandomWalk pdfRev-arrival convention.
    let pdfRevPlaceholder = segmentReverseDensity;

    bdptLightPath[bdptLightPathIndex(col, 0u)] = vec4f(newPos, 0.0);
    bdptLightPath[bdptLightPathIndex(col, 1u)] = vec4f(nsFront, pdfFwd);
    bdptLightPath[bdptLightPathIndex(col, 2u)] = vec4f(newThroughput, pdfRevPlaceholder);
    // A9 — record the reached vertex's matId + wo toward the previous light vertex so
    // the §10.3 connection evaluates the REAL light-vertex BSDF (glossy/metallic).
    bdptWriteLvBsdf(col, f32(matIdx), woLp);
    bdptWriteLvMaterialPayload(col, hit.triIndex, hit.baryVW, hit.instanceIndex, isFrontFaceHit);
    bdptWriteLvInterfaceEta(col, hitEtaTOverI, incidentIor, transmittedIor);
    var activeMediumMatId = BDPT_NO_MEDIUM;
    if (mediumDepth > 0u) {
      activeMediumMatId = mediumStack[mediumDepth - 1u].matId;
    }
    bdptWriteLvMediumContext(col, activeMediumMatId);
    bdptWriteLvMediumSides(col, incidentMedium, transmittedMedium);
  }
}
`;
