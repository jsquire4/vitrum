import type { WgslModule } from '../pipeline/wgslComposer.js';

/**
 * Generic bounded refractive-caustic estimator for the realtime hybrid path.
 * It follows the same exact VNDF-sampled rough dielectric interfaces as the
 * camera, ReSTIR-GI, DDGI, and RC transport paths. Stained-glass flags only
 * retain optional calibration.
 */
export const REFRACTIVE_CAUSTICS_WGSL = /* wgsl */ `
struct RefractiveCausticPath {
  throughput: f32,
  direction: vec3f,
  sawGlass: u32,
  escaped: u32,
  eligible: u32,
};

fn refractiveCausticChannel(value: vec3f, channel: u32) -> f32 {
  if (channel == 0u) { return value.r; }
  if (channel == 1u) { return value.g; }
  return value.b;
}

fn traceRefractiveCausticPath(
  origin: vec3f,
  initialDirection: vec3f,
  channel: u32,
) -> RefractiveCausticPath {
  var out: RefractiveCausticPath;
  out.throughput = 1.0;
  out.direction = safe_normalize(initialDirection);
  out.sawGlass = 0u;
  out.escaped = 0u;
  out.eligible = 1u;
  var ray = Ray();
  ray.origin = origin;
  ray.direction = out.direction;
  var mediumDepth = 0u;
  var mediumIor: array<f32, 4>;
  var mediumTri: array<u32, 4>;
  var mediumMaterialWord: array<u32, 4>;
  var mediumInstance: array<u32, 4>;
  var mediumBeer: array<f32, 4>;
  var mediumThickness: array<f32, 4>;
  var interfaceCount = 0u;

  // Two faces per closed slab plus one nested shell. Overflow is rejected,
  // never treated as visibility, so the work and energy are both bounded.
  // Four interface slots plus one terminal visibility query. The terminal hit
  // may be a miss (successful escape) or an opaque blocker, never interface 5.
  for (var depth = 0u; depth <= 4u; depth = depth + 1u) {
    let hit = traceSceneFirstHitAlphaMaskTextured(
      ubo.bvhMode, ubo.tlasNodeCount,
      ray, ubo.triIntersectEpsilon,
      bvh_material, BVH_MATERIAL_TEX_WIDTH,
      ubo.frameSeed ^ (channel * 0x85ebca6bu) ^ (depth * 0x9e3779b9u) ^ 0x43415553u,
    );
    if (!hit.didHit) {
      out.direction = ray.direction;
      // A scene-global refractive-caustic gate means most receiver probes can
      // miss every glass primitive. That path belongs to lo_sunNEE's baseline,
      // not to the refractive residual: mark it ineligible so the caller adds
      // the matching baseline sample instead of treating the miss as a zero
      // estimate and subtracting unobstructed direct sun.
      if (out.sawGlass == 0u) {
        out.eligible = 0u;
      }
      out.escaped = select(0u, 1u, mediumDepth == 0u && out.sawGlass != 0u);
      return out;
    }

    // Integrate the geometric segment through the current medium before the
    // next interface. The packed RGB Beer tint represents authored thickness;
    // rescale it to the actual segment while spectral curves consume scene-
    // length distance directly.
    if (mediumDepth > 0u) {
      let top = mediumDepth - 1u;
      let segmentDistance = hit.dist + ubo.triIntersectEpsilon * 4.0;
      let distanceScale = segmentDistance / max(mediumThickness[top], 1e-5);
      let fallback = pow(clamp(mediumBeer[top], 0.0, 1.0), distanceScale);
      let spectral = materialSpectralAttenuation(
        mediumTri[top], segmentDistance, vec3f(fallback),
      );
      out.throughput = out.throughput * refractiveCausticChannel(spectral, channel);
    }

    let scalar = decodeMaterialColor(hit.matColorPacked);
    let transmission = sampleTransmissionMapForHit(hit, scalar.a);
    if (!materialHasTransmission(transmission)) {
      out.throughput = 0.0;
      return out;
    }
    out.sawGlass = 1u;

    let materialCoord = vec2u(
      hit.indices.w % BVH_MATERIAL_TEX_WIDTH,
      hit.indices.w / BVH_MATERIAL_TEX_WIDTH,
    );
    let materialWord = textureLoad(bvh_material, vec2i(materialCoord), 0).r;
    let layerControls = sampleFaceLayerControls(hit.indices.w, hit.side >= 0.0);
    let materialRoughness = faceLayerRoughness(
      sampleMaterialScalarMap(
        hit, MATERIAL_MAP_SLOT_ROUGHNESS, 1u, decodeRoughMetal(materialWord).x,
      ),
      layerControls,
    );
    let iorRgb = materialDispersionIorRgb(hit.indices.w, decodeIor(materialWord));
    let materialIor = refractiveCausticChannel(iorRgb, channel);
    var materialThickness = max(materialOpticalThickness(hit.indices.w), 0.0);
    let thicknessTexel = sampleMaterialAtlasRawAtOffset(
      hit.indices.w,
      MATERIAL_MAP_THICKNESS_TEXEL_OFFSET,
      hit.uv,
      materialAtlasUv1ForHit(hit),
    );
    if (thicknessTexel.x >= 0.0) {
      materialThickness = materialThickness *
        clamp(thicknessTexel.g, 0.0, 1.0);
    }
    let incidentDirection = ray.direction;
    let entering = hit.side >= 0.0;
    if (materialThickness > 0.0 && !entering && mediumDepth == 0u) {
      out.throughput = 0.0;
      return out;
    }
    let interfaceCost = select(1u, 2u, materialThickness <= 0.0);
    if (interfaceCount + interfaceCost > 4u) {
      out.throughput = 0.0;
      return out;
    }
    interfaceCount = interfaceCount + interfaceCost;
    let interfaceIsTlas = ubo.bvhMode == 1u;
    let interfaceBase = hit.instanceIndex * 4u;
    let interfaceNormalOk = interfaceIsTlas &&
      interfaceBase + 2u < tlasWorldToLocalColumnCount();
    let interfaceNormalIndex = select(0u, interfaceBase, interfaceNormalOk);
    let interfaceSmoothNormal = smoothShadingNormal(
      hit, hit.normal,
      sceneLoadBvhNormal(hit.indices.x).xyz,
      sceneLoadBvhNormal(hit.indices.y).xyz,
      sceneLoadBvhNormal(hit.indices.z).xyz,
      interfaceNormalOk,
      tlasLoadWorldToLocalColumn(interfaceNormalIndex),
      tlasLoadWorldToLocalColumn(interfaceNormalIndex + 1u),
      tlasLoadWorldToLocalColumn(interfaceNormalIndex + 2u),
    );
    let interfaceMappedNormal = applyBumpMapForHit(
      hit, applyNormalMapForHit(hit, interfaceSmoothNormal),
    );
    let alignedInterfaceNormal = select(
      -interfaceMappedNormal,
      interfaceMappedNormal,
      dot(interfaceMappedNormal, hit.normal) >= 0.0,
    );
    let faceNormal = select(
      -alignedInterfaceNormal,
      alignedInterfaceNormal,
      dot(incidentDirection, alignedInterfaceNormal) < 0.0,
    );
    if (dot(incidentDirection, faceNormal) >= -1e-6) {
      // A pathological mapped normal must not turn the transport solve into an
      // invalid negative-cosine refraction. Keep this raw-call correction zero.
      out.throughput = 0.0;
      out.eligible = 0u;
      return out;
    }
    var incidentIor = 1.0;
    if (mediumDepth > 0u) {
      incidentIor = mediumIor[mediumDepth - 1u];
    } else if (materialThickness > 0.0 && !entering) {
      incidentIor = materialIor;
    }
    var targetIor = materialIor;
    if (materialThickness <= 0.0) {
      targetIor = materialIor;
    } else if (entering) {
      if (mediumDepth >= 4u) {
        out.throughput = 0.0;
        return out;
      }
      targetIor = materialIor;
    } else {
      let top = mediumDepth - 1u;
      if (
        mediumMaterialWord[top] != materialWord ||
        mediumInstance[top] != hit.instanceIndex
      ) {
        out.throughput = 0.0;
        return out;
      }
      if (mediumDepth > 1u) {
        targetIor = mediumIor[mediumDepth - 2u];
      }
    }

    var interfaceRng =
      ubo.frameSeed ^ (channel * 0x85ebca6bu) ^
      (depth * 0x9e3779b9u) ^ hit.indices.w ^ 0x43415553u;
    let interfaceBtdf = ggxSampleDielectricTransmission(
      faceNormal,
      -ray.direction,
      materialRoughness,
      incidentIor,
      targetIor,
      &interfaceRng,
    );
    if (interfaceBtdf.valid == 0u) {
      out.throughput = 0.0;
      return out;
    }
    let interfaceCos = interfaceBtdf.microfacetCos;
    var interfaceT = refractiveCausticChannel(
      dielectricInterfaceTransmissionRgb(
        interfaceCos, vec3f(incidentIor), vec3f(targetIor),
      ),
      channel,
    );
    let film = materialThinFilmResponse(
      hit.indices.w, hit.side >= 0.0, interfaceCos,
    );
    if (film.present != 0u) {
      interfaceT = refractiveCausticChannel(film.transmittance, channel);
    }

    var interfaceWeight = interfaceBtdf.weight;
    if (interfaceBtdf.transmission > 0.0) {
      interfaceWeight = interfaceWeight * interfaceT / interfaceBtdf.transmission;
    }
    interfaceWeight = interfaceWeight *
      refractiveCausticChannel(faceLayerTransmission(layerControls), channel);

    if (materialThickness <= 0.0) {
      // A thin sheet still crosses two reciprocal rough interfaces. Sample the
      // opposite authored face rather than silently preserving the direction.
      let exitLayer = sampleFaceLayerControls(hit.indices.w, hit.side < 0.0);
      let exitRoughness = faceLayerRoughness(materialRoughness, exitLayer);
      var exitRng = interfaceRng ^ 0xb7e15162u;
      let exitBtdf = ggxSampleDielectricTransmission(
        faceNormal,
        -interfaceBtdf.direction,
        exitRoughness,
        targetIor,
        incidentIor,
        &exitRng,
      );
      if (exitBtdf.valid == 0u) {
        out.throughput = 0.0;
        return out;
      }
      var exitT = refractiveCausticChannel(
        dielectricInterfaceTransmissionRgb(
          exitBtdf.microfacetCos, vec3f(targetIor), vec3f(incidentIor),
        ),
        channel,
      );
      let exitFilm = materialThinFilmResponse(
        hit.indices.w, hit.side < 0.0, exitBtdf.microfacetCos,
      );
      if (exitFilm.present != 0u) {
        exitT = refractiveCausticChannel(exitFilm.transmittance, channel);
      }
      var exitWeight = exitBtdf.weight;
      if (exitBtdf.transmission > 0.0) {
        exitWeight = exitWeight * exitT / exitBtdf.transmission;
      }
      exitWeight = exitWeight *
        refractiveCausticChannel(faceLayerTransmission(exitLayer), channel);
      out.throughput = out.throughput *
        interfaceWeight * exitWeight * transmission;
      let thinHitPos = ray.origin + incidentDirection * hit.dist;
      ray.direction = exitBtdf.direction;
      ray.origin = thinHitPos + ray.direction * (ubo.triIntersectEpsilon * 4.0);
      out.direction = ray.direction;
      continue;
    }

    out.throughput = out.throughput * interfaceWeight;
    var entryBeer = 1.0;
    let entryThickness = materialThickness;
    if (entering) {
      out.throughput = out.throughput * transmission;
      let beerCoord = vec2u(
        hit.indices.w % BVH_BEER_TEX_WIDTH,
        hit.indices.w / BVH_BEER_TEX_WIDTH,
      );
      let packedBeer = textureLoad(bvh_beer, vec2i(beerCoord), 0).r;
      var rgbBeer = vec3f(
        f32((packedBeer >> 24u) & 0xffu) / 255.0,
        f32((packedBeer >> 16u) & 0xffu) / 255.0,
        f32((packedBeer >> 8u) & 0xffu) / 255.0,
      );
      rgbBeer = applyThicknessMapToBeerTint(
        hit.indices.w, hit.uv, materialAtlasUv1ForHit(hit), rgbBeer,
      );
      entryBeer = refractiveCausticChannel(rgbBeer, channel);
    }

    if (entering) {
      mediumIor[mediumDepth] = materialIor;
      mediumTri[mediumDepth] = hit.indices.w;
      mediumMaterialWord[mediumDepth] = materialWord;
      mediumInstance[mediumDepth] = hit.instanceIndex;
      mediumBeer[mediumDepth] = entryBeer;
      mediumThickness[mediumDepth] = entryThickness;
      mediumDepth = mediumDepth + 1u;
    } else if (mediumDepth > 0u) {
      mediumDepth = mediumDepth - 1u;
    }
    let hitPos = ray.origin + incidentDirection * hit.dist;
    ray.direction = interfaceBtdf.direction;
    ray.origin = hitPos + ray.direction * (ubo.triIntersectEpsilon * 4.0);
    out.direction = ray.direction;
  }
  out.throughput = 0.0;
  return out;
}

fn lo_refractive_caustic(
  gid: vec2u,
  pos: vec3f,
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
  isMetal: bool,
) -> vec3f {
  if (ubo.sunAngular.z >= 1.5) {
    return lo_manifold_caustic(
      gid, pos, normal, clearcoatNormal, wo,
      albedo, rough, metal, specular, anisotropy,
      anisotropyTangent, anisotropyBitangent,
      iridescence, clearcoat, sheen, sheenRoughness,
      isGlass,
    );
  }
  if (ubo.sunAngular.z < 0.5 || !(ubo.sunIntensity > 0.0) || isGlass || isMetal) {
    return vec3f(0.0);
  }
  let sunBase = safe_normalize(ubo.sunDirection);
  if (dot(normal, sunBase) <= 0.0) { return vec3f(0.0); }

  var probe = Ray();
  probe.origin = pos + normal * walkaroundRayOriginBias();
  probe.direction = sunBase;

  // Core's directional-emitter contract bounds angularDiameter to [0, PI], so
  // the radius must be finite and in [0, PI/2]. Reject an invalid raw UBO here
  // before any trigonometry; zero-radius authored suns use a numerical 1 mrad
  // disk solely to keep the bounded proposal density finite.
  if (!(ubo.sunAngular.x >= 0.0 && ubo.sunAngular.x <= 1.5707963268)) {
    return vec3f(0.0);
  }
  let sunRadius = max(ubo.sunAngular.x, 0.001);
  let searchRadius = min(0.35, max(sunRadius * 8.0, 0.12));
  let upRef = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), abs(sunBase.y) < 0.99);
  let tangent = safe_normalize(cross(upRef, sunBase));
  let bitangent = cross(sunBase, tangent);
  let omegaSearch = 6.2831853 * (1.0 - cos(searchRadius));
  let omegaSun = 6.2831853 * (1.0 - cos(sunRadius));
  // Uniform-solid-angle proposal conversion. Both solid angles are positive,
  // finite and contract-bounded above, so this is the exact (unclamped)
  // omegaSearch / omegaSun weight. Keeping the large but finite weight is what
  // makes the two-candidate residual unbiased inside this bounded path model.
  let jacobianWeight = omegaSearch / omegaSun;
  // lo_sunNEE owns the straight tinted estimator. Unsupported rough paths
  // contribute this baseline to the correction estimator, producing exactly
  // zero correction instead of a spurious negative caustic.
  let baselineT = traceSceneAlphaTintTransmittanceTextured(
    ubo.bvhMode, ubo.tlasNodeCount,
    probe.origin, sunBase, INFINITY, ubo.triIntersectEpsilon,
    bvh_material, BVH_MATERIAL_TEX_WIDTH, bvh_beer,
  );
  let baseline = baselineT * max(0.0, dot(normal, sunBase));
  var estimate = vec3f(0.0);

  // Two frame-scrambled stratified candidates are enough for the realtime
  // path; temporal/spatial filtering integrates the bounded residual noise.
  for (var candidate = 0u; candidate < 2u; candidate = candidate + 1u) {
    let xi0 = pixelHash2(
      gid + vec2u(candidate * 131u, candidate * 313u),
      ubo.frameSeed ^ 0x52434658u,
    );
    let cosTheta = mix(1.0, cos(searchRadius), (f32(candidate) + xi0.x) * 0.5);
    let sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
    let phi = 6.2831853 * xi0.y;
    let candidateDir = safe_normalize(
      sunBase * cosTheta + tangent * (sinTheta * cos(phi)) + bitangent * (sinTheta * sin(phi)),
    );
    let receiverCosine = max(0.0, dot(normal, candidateDir));
    for (var channel = 0u; channel < 3u; channel = channel + 1u) {
      let path = traceRefractiveCausticPath(probe.origin, candidateDir, channel);
      if (path.eligible == 0u) {
        let fallback = refractiveCausticChannel(baseline, channel) * 0.5;
        if (channel == 0u) { estimate.r = estimate.r + fallback; }
        if (channel == 1u) { estimate.g = estimate.g + fallback; }
        if (channel == 2u) { estimate.b = estimate.b + fallback; }
        continue;
      }
      let reachesSun = path.escaped != 0u && dot(path.direction, sunBase) >= cos(sunRadius);
      if (reachesSun) {
        let contribution = path.throughput * receiverCosine * jacobianWeight * 0.5;
        if (channel == 0u) { estimate.r = estimate.r + contribution; }
        if (channel == 1u) { estimate.g = estimate.g + contribution; }
        if (channel == 2u) { estimate.b = estimate.b + contribution; }
      }
    }
  }

  // lo_sunNEE already owns the straight tinted estimator. Return the signed
  // path-space correction so a flat parallel slab does not double direct sun,
  // while focusing brightens and defocusing darkens the physically same term.
  var correction = (estimate - baseline) * vec3f(ubo.sunIntensity) * albedo * INV_PI;
  if ((ubo.stainedGlassFlags & SG_FLAG_SUN_CAUSTIC) != 0u) {
    let positive = max(correction, vec3f(0.0));
    correction = min(correction, vec3f(0.0)) +
      min(positive, vec3f(ubo.causticVisClamp * ubo.sunIntensity)) * ubo.causticBoost;
  }
  return correction;
}
`;

export const REFRACTIVE_CAUSTICS_MODULE: WgslModule = {
  name: 'refractiveCaustics',
  source: REFRACTIVE_CAUSTICS_WGSL,
  requires: ['common', 'materialAtlas', 'ggxBrdf', 'manifoldCaustics'],
};
