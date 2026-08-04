import {
  composeShadePrologueWgsl,
  SHADE_PROLOGUE_EMISSIVE_COMMENT_LITE,
} from './shadePrologue.wgsl.js';
import { PT_WEBGPU_PATH_TRACE_KERNEL_CORE_WGSL } from './kernelCore.wgsl.js';

/**
 * Kernel module — primary-ray generation, motion-vector projection, Russian
 * roulette helpers, per-pixel accumulation, and the `@compute` entry point
 * that ties every other module together.
 *
 * Bundled here:
 *  - `generatePrimaryRay` — inverse-VP camera ray + sub-pixel jitter
 *  - `projectToNdc` — VP-clip projection used for motion vectors
 *  - `causticMode` — UBO accessor for the caustic-strategy selector
 *  - `RRResult` struct + `russianRoulette` — bounce-termination helper
 *  - `accumulateFrame` — output texture writes + variance-moments update
 *  - `main` — the @compute @workgroup_size(8,8,1) kernel that walks each ray
 *
 * This module is the LAST concatenated chunk because it consumes every other
 * module: traceClosest/hitMaterialId (intersection), decodeMaterial /
 * thinFilmTmmRt / fresnelSchlick / sampleMaterialSpectralMu (material),
 * sampleNextBounceDirection / cosineHemisphereSample / evaluateBrdf /
 * brdfDirectionalPdf (bsdf), sampleEnvironmentColor / sampleEnvironmentImportance /
 * environmentPdf / bsdfAreaLightConnectionContribution /
 * bsdfEnvironmentConnectionContribution (connect), and manifoldNeeContribution /
 * photonMapContribution (caustic).
 */
/**
 * Lite kernel — directional + procedural-sky direct lighting only; no motion
 * vectors or variance-moments buffer (compatibility tier bind layout).
 */
export const PT_WEBGPU_PATH_TRACE_KERNEL_LITE_WGSL = /* wgsl */ `
${PT_WEBGPU_PATH_TRACE_KERNEL_CORE_WGSL}

const LITE_MEDIUM_STACK_LIMIT = 8u;

struct LiteMediumLayer {
  matId: u32,
  boundary: vec3u,
  ior: f32,
  sigmaA: vec3f,
  remainingDistance: f32,
};

fn liteMediumSigmaA(matId: u32, mat: DecodedMaterial, heroLambda: f32) -> vec3f {
  var spectralMu = vec3f(mat.spectralAvgMu);
  if (mat.spectralSampleCount > 0u) {
    if (params.spectralEnabled != 0u) {
      spectralMu = vec3f(
        sampleMaterialSpectralMu(matId, heroLambdaTo01(heroLambda)),
      );
    } else {
      spectralMu = vec3f(
        sampleMaterialSpectralMu(matId, 0.15),
        sampleMaterialSpectralMu(matId, 0.50),
        sampleMaterialSpectralMu(matId, 0.85),
      );
    }
  }
  var sigmaA = select(
    vec3f(0.0), max(spectralMu, vec3f(0.0)), mat.hasSpectralAttenuation,
  );
  if (!mat.hasSpectralAttenuation && mat.hasSigmaA) {
    sigmaA = max(mat.sigmaA, vec3f(0.0));
  }
  if (params.spectralEnabled != 0u && !mat.hasSpectralAttenuation) {
    sigmaA = vec3f(spectralRgbFactorAtHero(sigmaA, heroLambda));
  }
  return sigmaA;
}

fn liteMediumLayer(
  matId: u32,
  mat: DecodedMaterial,
  heroLambda: f32,
  boundary: vec3u,
) -> LiteMediumLayer {
  return LiteMediumLayer(
    matId,
    boundary,
    max(mat.ior, 1e-4),
    liteMediumSigmaA(matId, mat, heroLambda),
    materialAttenuationDistance(INFINITY, mat),
  );
}

fn liteMediumLayerMatches(
  layer: LiteMediumLayer,
  matId: u32,
  boundary: vec3u,
) -> bool {
  return layer.matId == matId && all(layer.boundary == boundary);
}

fn accumulateFrame(
  gid: vec3u,
  radiance: vec3f,
  firstHitValid: bool,
  firstHitPos: vec3f,
  firstHitNormal: vec3f,
  firstHitAlbedo: vec3f,
  firstHitDepth: f32,
) {
  let sampleColor = max(radiance, vec3f(0.0));

  let pixelIndex = gid.y * params.width + gid.x;
  var accum = accumBuffer[pixelIndex];
  accum = accum + vec4f(sampleColor, 1.0);
  accumBuffer[pixelIndex] = accum;
  let display = accum.xyz / max(accum.w, 1.0);
  textureStore(outputTexture, vec2i(gid.xy), vec4f(display, 1.0));
  if (firstHitValid) {
    textureStore(normalDepthTexture, vec2i(gid.xy), vec4f(firstHitNormal * 0.5 + vec3f(0.5), firstHitDepth));
    textureStore(albedoTexture, vec2i(gid.xy), vec4f(firstHitAlbedo, 1.0));
  } else {
    textureStore(normalDepthTexture, vec2i(gid.xy), vec4f(0.5, 1.0, 0.5, 0.0));
    textureStore(albedoTexture, vec2i(gid.xy), vec4f(0.0, 0.0, 0.0, 0.0));
  }
  // The lite tier has no moment history and does not advertise variance.
  // Keep the layout-compatible storage target deterministic without exposing a
  // raw luminance image under the canonical variance semantic.
  textureStore(varianceTexture, vec2i(gid.xy), vec4f(0.0));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= params.width || gid.y >= params.height) {
    return;
  }

  var rng = pcgInit(gid.x, gid.y, ptRngFrameKey(params.frameSeed, params.frameIndex));
  let jitter = vec2f(rand_f32(&rng), rand_f32(&rng));
  var ray = generatePrimaryRay(gid.x, gid.y, jitter);
  let primaryRayOrigin = ray.origin;

  var heroLambda = params.heroLambdaNm;
  var heroPdf = params.heroPdf;
  if (params.spectralEnabled != 0u) {
    let hero = sampleHeroWavelengthMIS(rand_f32(&rng), rand_f32(&rng));
    heroLambda = hero.x;
    heroPdf = hero.y;
  }

  var radiance = vec3f(0.0);
  var throughput = vec3f(1.0);
  let bounceLimit = max(1u, min(params.maxBounces, 8u));
  var firstHitValid = false;
  var firstHitPos = vec3f(0.0);
  var firstHitNormal = vec3f(0.0, 1.0, 0.0);
  var firstHitAlbedo = vec3f(0.0);
  var firstHitDepth = 0.0;
  // Camera-visible emitters: gate the emissive-on-hit term to the camera ray +
  // post-refraction paths (see kernel.wgsl.ts for the rationale). Init false so a
  // directly-viewed emitter glows.
  var prevSampleAllowsAreaMis = false;

  // Reconstruct the ordered authored-bulk state at the exact camera origin.
  // The live nesting limit is eight; traversal itself has no crossing ceiling.
  var mediumStack: array<LiteMediumLayer, LITE_MEDIUM_STACK_LIMIT>;
  let cameraContainment = opticalContainmentAlongRay(ray.origin, ray.direction);
  var mediumDepth = cameraContainment.depth;
  if (!cameraContainment.valid) {
    accumulateFrame(
      gid, vec3f(0.0), false, vec3f(0.0), vec3f(0.0, 1.0, 0.0),
      vec3f(0.0), 0.0,
    );
    return;
  }
  for (var mediumIndex = 0u; mediumIndex < mediumDepth; mediumIndex = mediumIndex + 1u) {
    let initialMatId = cameraContainment.matIds[mediumIndex];
    mediumStack[mediumIndex] = liteMediumLayer(
      initialMatId,
      decodeMaterial(initialMatId),
      heroLambda,
      cameraContainment.boundaries[mediumIndex],
    );
    if (!(mediumStack[mediumIndex].remainingDistance >= 0.0)) {
      accumulateFrame(
        gid, vec3f(0.0), false, vec3f(0.0), vec3f(0.0, 1.0, 0.0),
        vec3f(0.0), 0.0,
      );
      return;
    }
  }

  for (var bounce = 0u; bounce < bounceLimit; bounce = bounce + 1u) {
    let sppmOwnsCurrentEmission = false;
    let hit = traceClosest(ray, ptRayTMin(), INFINITY);
    if (mediumDepth > 0u) {
      let topIndex = mediumDepth - 1u;
      let segmentDistance = select(INFINITY, hit.dist, hit.didHit);
      let distanceInMedium = min(
        segmentDistance, mediumStack[topIndex].remainingDistance,
      );
      if (!(distanceInMedium >= 0.0)) { break; }
      throughput = throughput * materialBeer(
        mediumStack[topIndex].sigmaA, distanceInMedium,
      );
      // Keep an unbounded medium unbounded: INFINITY - INFINITY is NaN.
      if (mediumStack[topIndex].remainingDistance <= PT_F32_MAX) {
        mediumStack[topIndex].remainingDistance = max(
          mediumStack[topIndex].remainingDistance - distanceInMedium, 0.0,
        );
      }
    }
    if (!hit.didHit) {
      // Diffuse/glossy BSDF environment paths were already added with their MIS
      // weight at the previous vertex. Only camera and delta/specular-transmission
      // escapes need the raw miss pickup.
      if (!prevSampleAllowsAreaMis) {
        // A3 — spectralise the env at the hero λ in spectral mode (RGB unchanged).
        let envRgb = sampleEnvironmentColor(ray.direction);
        let envContribution = select(envRgb, spectralEmissionAtHero(envRgb, heroLambda), params.spectralEnabled != 0u);
        radiance = radiance + throughput *
          ptScaleEnvironmentRadiance(envContribution, 1.0);
      }
      break;
    }

${composeShadePrologueWgsl(SHADE_PROLOGUE_EMISSIVE_COMMENT_LITE)}
    let throughputAtVertex = throughput;
    let surfaceMediumBoundary = mediumBoundaryIdentity(
      hit.triIndex, hit.instanceIndex,
    );
    let surfaceCrossesMedium = mat.isBulkMedium;
    if (surfaceCrossesMedium && !mediumBoundaryIsValid(surfaceMediumBoundary)) {
      break;
    }
    var incidentIor = 1.0;
    if (mediumDepth > 0u) {
      incidentIor = mediumStack[mediumDepth - 1u].ior;
    }
    var transmittedIor = max(ior, 1e-4);
    if (surfaceCrossesMedium && !isFrontFace) {
      if (
        mediumDepth == 0u ||
        !liteMediumLayerMatches(
          mediumStack[mediumDepth - 1u], matId, surfaceMediumBoundary,
        )
      ) {
        break;
      }
      transmittedIor = 1.0;
      if (mediumDepth > 1u) {
        transmittedIor = mediumStack[mediumDepth - 2u].ior;
      }
    }
    let surfaceEtaTOverI = transmittedIor / max(incidentIor, 1e-4);
    let cosThetaO = max(0.0, dot(normal, wo));
    let f0Base = materialSpecularF0(
      baseColor, metallic, surfaceEtaTOverI,
      mat.specularColor, mat.specularIntensity,
    );
    let f0 = iridescenceModifiedF0(
      f0Base,
      mat.iridescence,
      mat.iridescenceIor,
      mat.iridescenceThicknessMin,
      mat.iridescenceThicknessMax,
      cosThetaO,
    );
    let fresnel = materialSpecularFresnelSchlick(
      cosThetaO, f0, metallic, mat.specularIntensity,
    );

    // B12 — lite-tier NEE: directional + env/sky + point + spot + rect-area.
    // Directional/point/spot/rect data is loaded from liteLightTex (binding 14) via textureLoad.
    // Counts come from the UBO (params.directionalLightCount / pointLightCount /
    // spotLightCount / rectAreaLightCount).
    // liteLightTex layout (1-row, consecutive vec4 texels):
    //   [0, directionalLightCount*2):              directional records (2 texels/light)
    //   [dirOff, dirOff + pointLightCount*3):      point records  (3 texels/light)
    //   [pointOff, pointOff + spotLightCount*4):   spot records   (4 texels/light)
    //   [spotOff,  spotOff  + rectAreaLightCount*4): rect records (4 texels/light)
    let liteDirBase = 0u;
    let litePtBase = params.directionalLightCount * 2u;
    let liteSpBase = litePtBase + params.pointLightCount * 3u;
    let liteRcBase = liteSpBase + params.spotLightCount * 4u;

    var lightCount = 0u;
    lightCount = lightCount + params.directionalLightCount;
    lightCount = lightCount + params.pointLightCount;
    lightCount = lightCount + params.spotLightCount;
    lightCount = lightCount + params.rectAreaLightCount;
    if (hasEnvironmentMap()) {
      lightCount = lightCount + 1u;
    }
    if (lightCount > 0u) {
      let sumDirectLighting = params.directLightingMode == 1u;
      let picked = ptRandBoundedU32(&rng, lightCount);
      let directLightingScale = select(f32(lightCount), 1.0, sumDirectLighting);
      var current = 0u;
      var directLi = vec3f(0.0);
      for (var di = 0u; di < params.directionalLightCount; di = di + 1u) {
        if (sumDirectLighting || current == picked) {
          let dBase = liteDirBase + di * 2u;
          let dDirAD = textureLoad(liteLightTex, vec2i(i32(dBase), 0), 0);
          let dIrrMean = textureLoad(liteLightTex, vec2i(i32(dBase + 1u), 0), 0);
          if (dIrrMean.w > 0.0) {
            var lightDir = safe_normalize(dDirAD.xyz);
            // SHADOW-01 — emitter castShadow:false is sign-encoded into the
            // angularDiameter lane (packed = -1 - angularDiameter).
            let angDiamRaw = dDirAD.w;
            let dirShadowDisabled = angDiamRaw < 0.0;
            let angDiam = select(angDiamRaw, -1.0 - angDiamRaw, dirShadowDisabled);
            if (!ptDirectionalConeIsDelta(angDiam)) {
              let xi1 = rand_f32(&rng);
              let xi2 = rand_f32(&rng);
              let sinCosTheta = ptDirectionalConeSinCos(angDiam, xi1);
              let sinTheta = sinCosTheta.x;
              let cosTheta = sinCosTheta.y;
              let phi = 6.28318530718 * xi2;
              let tangentX = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), abs(lightDir.x) > 0.9);
              let basisY = normalize(cross(lightDir, tangentX));
              let basisX = cross(basisY, lightDir);
              lightDir = normalize(sinTheta * cos(phi) * basisX + sinTheta * sin(phi) * basisY + cosTheta * lightDir);
            }
            let directOffsetNormal = select(-normal, normal, dot(normal, lightDir) > 0.0);
            let shadowRay = Ray(
              hitPos + directOffsetNormal * ptRayOriginBias(),
              lightDir,
            );
            var visibility = SurfaceVisibility(vec3f(1.0), true);
            if (!dirShadowDisabled) {
              visibility = traceSurfaceVisibility(
                shadowRay, ptRayTMin(), INFINITY,
                heroLambda, incidentIor,
              );
            }
            if (visibility.visible) {
              let nDotL = abs(dot(normal, lightDir));
              let brdf = evaluateFiniteBsdfFullWithClearcoatNormal(baseColor, roughness, metallic, transmission, surfaceEtaTOverI, normal, normal, wo, lightDir,
                mat.clearcoat, mat.clearcoatRoughness, mat.sheen, mat.sheenRoughness, mat.sheenColor,
                mat.iridescence, mat.iridescenceIor, mat.iridescenceThicknessMin, mat.iridescenceThicknessMax,
                mat.specularColor, mat.specularIntensity,
                0.0, 0.0, thinFilm, false);
              let dirIrrOut = select(dIrrMean.rgb, spectralEmissionAtHero(dIrrMean.rgb, heroLambda), params.spectralEnabled != 0u);
              directLi = directLi + throughput * brdf * nDotL * dirIrrOut *
                visibility.transmittance;
            }
          }
        }
        current = current + 1u;
      }
      // B12 — point lights (delta; stride 3 texels: pos, rad, [dist, decay, 0, 0]).
      for (var pi = 0u; pi < params.pointLightCount; pi = pi + 1u) {
        if (sumDirectLighting || current == picked) {
          let base = litePtBase + pi * 3u;
          let lp  = textureLoad(liteLightTex, vec2i(i32(base),      0), 0).xyz;
          let rad = textureLoad(liteLightTex, vec2i(i32(base + 1u), 0), 0).rgb;
          let extra = textureLoad(liteLightTex, vec2i(i32(base + 2u), 0), 0);
          let ptMaxDist = extra.x;
          let ptDecay   = extra.y;
          let toPoint = lp - hitPos;
          let dist = safe_length(toPoint);
          if (!(dist > 0.0) || (ptMaxDist > 0.0 && dist > ptMaxDist)) {
            current = current + 1u;
            continue;
          }
          let wi = toPoint / dist;
          let pointOffsetNormal = select(-normal, normal, dot(normal, wi) > 0.0);
          let pointShadowRay = Ray(
            hitPos + pointOffsetNormal * ptRayOriginBias(),
            wi,
          );
          // SHADOW-01 — extra.z carries the emitter castShadowDisabled flag.
          var visibility = SurfaceVisibility(vec3f(1.0), true);
          if (!(extra.z > 0.5)) {
            visibility = traceSurfaceVisibility(
              pointShadowRay,
              ptRayTMin(),
              ptFiniteSegmentTMax(dist),
              heroLambda,
              incidentIor,
            );
          }
          if (visibility.visible) {
            let nDotL = abs(dot(normal, wi));
            let brdf = evaluateFiniteBsdfFullWithClearcoatNormal(baseColor, roughness, metallic, transmission, surfaceEtaTOverI, normal, normal, wo, wi,
              mat.clearcoat, mat.clearcoatRoughness, mat.sheen, mat.sheenRoughness, mat.sheenColor,
              mat.iridescence, mat.iridescenceIor, mat.iridescenceThicknessMin, mat.iridescenceThicknessMax,
              mat.specularColor, mat.specularIntensity,
              0.0, 0.0, thinFilm, false);
            let attenuation = pointSpotDistanceAttenuation(dist, ptMaxDist, ptDecay);
            let radOut = select(rad, spectralEmissionAtHero(rad, heroLambda), params.spectralEnabled != 0u);
            directLi = directLi + throughput * brdf * nDotL * radOut *
              attenuation * visibility.transmittance;
          }
        }
        current = current + 1u;
      }
      // B12 — spot lights (delta; stride 4 texels: pos, dir+cosOuter, rad+cosInner, [dist,decay,0,0]).
      for (var si = 0u; si < params.spotLightCount; si = si + 1u) {
        if (sumDirectLighting || current == picked) {
          let sb2 = liteSpBase + si * 4u;
          let spos   = textureLoad(liteLightTex, vec2i(i32(sb2),      0), 0).xyz;
          let saxis  = textureLoad(liteLightTex, vec2i(i32(sb2 + 1u), 0), 0);
          let sradW  = textureLoad(liteLightTex, vec2i(i32(sb2 + 2u), 0), 0);
          let spExtra = textureLoad(liteLightTex, vec2i(i32(sb2 + 3u), 0), 0);
          let spotDir  = safe_normalize(saxis.xyz);
          let cosOuter = saxis.w;
          let cosInner = sradW.w;
          let srad     = sradW.rgb;
          let spMaxDist = spExtra.x;
          let spDecay   = spExtra.y;
          let toSpot = spos - hitPos;
          let dist = safe_length(toSpot);
          if (!(dist > 0.0) || (spMaxDist > 0.0 && dist > spMaxDist)) {
            current = current + 1u;
            continue;
          }
          let wi = toSpot / dist;
          let coneCos = dot(-wi, spotDir);
          if (coneCos >= cosOuter) {
            let spotOffsetNormal = select(-normal, normal, dot(normal, wi) > 0.0);
            let spotShadowRay = Ray(
              hitPos + spotOffsetNormal * ptRayOriginBias(),
              wi,
            );
            // SHADOW-01 — spExtra.z carries the emitter castShadowDisabled flag.
            var visibility = SurfaceVisibility(vec3f(1.0), true);
            if (!(spExtra.z > 0.5)) {
              visibility = traceSurfaceVisibility(
                spotShadowRay,
                ptRayTMin(),
                ptFiniteSegmentTMax(dist),
                heroLambda,
                incidentIor,
              );
            }
            if (visibility.visible) {
              let nDotL = abs(dot(normal, wi));
              let softness = smoothstep(cosOuter, max(cosInner, cosOuter + 1e-6), coneCos);
              let attenuation = pointSpotDistanceAttenuation(dist, spMaxDist, spDecay);
              let brdf = evaluateFiniteBsdfFullWithClearcoatNormal(baseColor, roughness, metallic, transmission, surfaceEtaTOverI, normal, normal, wo, wi,
                mat.clearcoat, mat.clearcoatRoughness, mat.sheen, mat.sheenRoughness, mat.sheenColor,
                mat.iridescence, mat.iridescenceIor, mat.iridescenceThicknessMin, mat.iridescenceThicknessMax,
                mat.specularColor, mat.specularIntensity,
                0.0, 0.0, thinFilm, false);
              let sradOut = select(srad, spectralEmissionAtHero(srad, heroLambda), params.spectralEnabled != 0u);
              directLi = directLi + throughput * brdf * nDotL * softness * sradOut *
                attenuation * visibility.transmittance;
            }
          }
        }
        current = current + 1u;
      }
      // B12 — rect/disc-area lights (area; stride 4 texels: rpos, ru, rv, rshape).
      // Shape discriminator in rshape.w: ≈ 0 → rect, ≈ 1 → analytic disc.
      // Disc sampling uses the concentric-disc map (Shirley & Chiu 1997);
      // area = π·|u×v|.
      // Lite rect/disc lights are analytic records, not scene geometry. The
      // paired BSDF->area-light half lives in connectLite and intersects the
      // same liteLightTex records, so this NEE half applies the matching MIS
      // weight just like the full tier.
      // Native analytic disc emitters replace the 32-triangle fan, 2026-06-10 —
      // RENDER-CHANGING for disc-lit scenes, A/B in R9-B.
      for (var ri = 0u; ri < params.rectAreaLightCount; ri = ri + 1u) {
        if (sumDirectLighting || current == picked) {
          let rb2 = liteRcBase + ri * 4u;
          let rpos = textureLoad(liteLightTex, vec2i(i32(rb2),      0), 0).xyz;
          let ru   = textureLoad(liteLightTex, vec2i(i32(rb2 + 1u), 0), 0).xyz;
          let rv   = textureLoad(liteLightTex, vec2i(i32(rb2 + 2u), 0), 0).xyz;
          let rshapeL = textureLoad(liteLightTex, vec2i(i32(rb2 + 3u), 0), 0);
          let rr   = rshapeL.rgb;
          let isDiscL = abs(rshapeL.w - 1.0) < 0.5;
          let areaMeasure = measureAreaVector(
            ru, rv, select(4.0, PI, isDiscL),
          );
          let xi1l = rand_f32(&rng);
          let xi2l = rand_f32(&rng);
          var lpos: vec3f;
          if (isDiscL) {
            // D9.11 — Shirley & Chiu 1997 concentric-disc map via shared kernelCore helper.
            let discL = concentricDiscSample(vec2f(xi1l * 2.0 - 1.0, xi2l * 2.0 - 1.0));
            lpos = rpos + ru * discL.x + rv * discL.y;
          } else {
            lpos = rpos + ru * (xi1l * 2.0 - 1.0) + rv * (xi2l * 2.0 - 1.0);
          }
          let toLight = lpos - hitPos;
          let dist = safe_length(toLight);
          let wi = safe_normalize(toLight);
          let nDotL = abs(dot(normal, wi));
          if (
            dist > 0.0 && nDotL > 0.0 &&
            areaMeasure.valid != 0u
          ) {
            let brdf = evaluateFiniteBsdfFullWithClearcoatNormal(baseColor, roughness, metallic, transmission, surfaceEtaTOverI, normal, normal, wo, wi,
              mat.clearcoat, mat.clearcoatRoughness, mat.sheen, mat.sheenRoughness, mat.sheenColor,
              mat.iridescence, mat.iridescenceIor, mat.iridescenceThicknessMin, mat.iridescenceThicknessMax,
              mat.specularColor, mat.specularIntensity,
              0.0, 0.0, thinFilm, false);
            let lightNormal = areaMeasure.normal;
            let cosLight = max(dot(lightNormal, -wi), 0.0);
            if (cosLight > 0.0) {
              let lightPdf =
                ptAreaToSolidAnglePdf(dist, cosLight, areaMeasure);
              if (!(lightPdf > 0.0)) {
                current = current + 1u;
                continue;
              }
              let brdfPdf = brdfDirectionalPdfFullSampled(baseColor, roughness, metallic, transmission, surfaceEtaTOverI, normal, wo, wi,
                mat.clearcoat, mat.clearcoatRoughness, mat.sheen, mat.sheenRoughness,
                mat.iridescence, mat.iridescenceIor, mat.iridescenceThicknessMin, mat.iridescenceThicknessMax,
                mat.specularColor, mat.specularIntensity,
                0.0, 0.0, thinFilm);
              let misWeight = powerHeuristic(lightPdf, brdfPdf);
              let rectOffsetNormal = select(-normal, normal, dot(normal, wi) > 0.0);
              let shadowRay = Ray(
                hitPos + rectOffsetNormal * ptRayOriginBias(),
                wi,
              );
              // SHADOW-01 — rect record texel 0 .w carries castShadowDisabled.
              let rectShadowDisabledL = textureLoad(liteLightTex, vec2i(i32(rb2), 0), 0).w > 0.5;
              var visibility = SurfaceVisibility(vec3f(1.0), true);
              if (!rectShadowDisabledL) {
                visibility = traceSurfaceVisibility(
                  shadowRay,
                  ptRayTMin(),
                  ptFiniteSegmentTMax(dist),
                  heroLambda,
                  incidentIor,
                );
              }
              if (visibility.visible) {
                let rrOut = select(rr, spectralEmissionAtHero(rr, heroLambda), params.spectralEnabled != 0u);
                directLi = directLi + throughput * brdf * nDotL * rrOut *
                  visibility.transmittance * misWeight / lightPdf;
              }
            }
          }
        }
        current = current + 1u;
      }
      if (hasEnvironmentMap() && (sumDirectLighting || current == picked)) {
        var envDir = vec3f(0.0, 1.0, 0.0);
        var envColor = vec3f(0.0);
        var envPdf = 0.0;
        let envSample = sampleEnvironmentImportance(&rng);
        if (envSample.pdf > 0.0) {
          envDir = envSample.wi;
          envColor = envSample.value;
          envPdf = envSample.pdf;
        } else {
          envDir = uniformSphere(vec2f(rand_f32(&rng), rand_f32(&rng)));
          envColor = sampleEnvironmentColor(envDir);
          envPdf = 0.25 * INV_PI;
        }
        let nDotL = abs(dot(normal, envDir));
        if (nDotL > 0.0) {
          let envOffsetNormal = select(-normal, normal, dot(normal, envDir) > 0.0);
          let shadowRay = Ray(
            hitPos + envOffsetNormal * ptRayOriginBias(),
            envDir,
          );
          let visibility = traceSurfaceVisibility(
            shadowRay, ptRayTMin(), INFINITY,
            heroLambda, incidentIor,
          );
          if (visibility.visible) {
            let brdf = evaluateFiniteBsdfFullWithClearcoatNormal(baseColor, roughness, metallic, transmission, surfaceEtaTOverI, normal, normal, wo, envDir,
              mat.clearcoat, mat.clearcoatRoughness, mat.sheen, mat.sheenRoughness, mat.sheenColor,
              mat.iridescence, mat.iridescenceIor, mat.iridescenceThicknessMin, mat.iridescenceThicknessMax,
              mat.specularColor, mat.specularIntensity,
              0.0, 0.0, thinFilm, false);
            let brdfPdf = brdfDirectionalPdfFullSampled(baseColor, roughness, metallic, transmission, surfaceEtaTOverI, normal, wo, envDir,
              mat.clearcoat, mat.clearcoatRoughness, mat.sheen, mat.sheenRoughness,
              mat.iridescence, mat.iridescenceIor, mat.iridescenceThicknessMin, mat.iridescenceThicknessMax,
              mat.specularColor, mat.specularIntensity,
              0.0, 0.0, thinFilm);
            // A3 — spectralise the env radiance at the hero λ (RGB unchanged).
            let envColorOut = ptScaleEnvironmentRadiance(
              select(
                envColor,
                spectralEmissionAtHero(envColor, heroLambda),
                params.spectralEnabled != 0u,
              ),
              1.0,
            );
            let misWeight = powerHeuristic(envPdf, brdfPdf);
            directLi = directLi + throughput * brdf * nDotL * envColorOut *
              visibility.transmittance * misWeight / max(envPdf, 1e-8);
          }
        }
      }
      radiance = radiance + directLi * directLightingScale;
    }

    let caustic = causticMode();
    if (caustic == 1u) {
      radiance = radiance + manifoldNeeContribution(
        &rng,
        hitPos,
        normal,
        normal,
        wo,
        baseColor,
        roughness,
        metallic,
        transmission,
        surfaceEtaTOverI,
        mat.clearcoat,
        mat.clearcoatRoughness,
        mat.sheen,
        mat.sheenRoughness,
        mat.sheenColor,
        mat.iridescence,
        mat.iridescenceIor,
        mat.iridescenceThicknessMin,
        mat.iridescenceThicknessMax,
        mat.specularColor,
        mat.specularIntensity,
        0.0,
        0.0,
        thinFilm,
        heroLambda,
        throughputAtVertex,
      );
    } else if (caustic == 2u) {
      photonMapUpdateProgressive(
        0u,
        hitPos,
        normal,
        normal,
        wo,
        baseColor,
        roughness,
        metallic,
        transmission,
        surfaceEtaTOverI,
        mat.clearcoat,
        mat.clearcoatRoughness,
        mat.sheen,
        mat.sheenRoughness,
        mat.sheenColor,
        mat.iridescence,
        mat.iridescenceIor,
        mat.iridescenceThicknessMin,
        mat.iridescenceThicknessMax,
        mat.specularColor,
        mat.specularIntensity,
        0.0,
        0.0,
        thinFilm,
        throughputAtVertex,
        heroLambda,
        heroPdf,
        1.0,
      );
    }

    let bs = sampleNextBounceDirection(
      &rng,
      ray.direction,
      hitPos,
      hit.normal,
      normal,
      baseColor,
      roughness,
      metallic,
      transmission,
      surfaceEtaTOverI,
      false,
      fresnel,
      mat.iridescence,
      mat.iridescenceIor,
      mat.iridescenceThicknessMin,
      mat.iridescenceThicknessMax,
      mat.specularColor,
      mat.specularIntensity,
      thinFilm,
      mat.isThinSheet,
      oppositeNormal,
      oppositeRoughness,
      oppositeLayerW,
      surfaceCrossesMedium,
      mat.clearcoat,
      mat.clearcoatRoughness,
      mat.sheen,
      mat.sheenRoughness,
      mat.sheenColor,
      0.0, // anisotropy — lite tier has no aniso texture bindings; always isotropic
      0.0, // anisotropyRotation
    );
    if (bs.enteredMedium) {
      if (mediumDepth >= LITE_MEDIUM_STACK_LIMIT) { break; }
      mediumStack[mediumDepth] = liteMediumLayer(
        matId, mat, heroLambda, surfaceMediumBoundary,
      );
      if (!(mediumStack[mediumDepth].remainingDistance >= 0.0)) { break; }
      mediumDepth = mediumDepth + 1u;
    } else if (bs.exitedMedium) {
      if (
        mediumDepth == 0u ||
        !liteMediumLayerMatches(
          mediumStack[mediumDepth - 1u], matId, surfaceMediumBoundary,
        )
      ) {
        break;
      }
      mediumDepth = mediumDepth - 1u;
    }
    let crossedTransmissiveInterface =
      bs.sampledLobe == BSDF_LOBE_DELTA_TRANSMISSION ||
      bs.sampledLobe == BSDF_LOBE_ROUGH_TRANSMISSION ||
      bs.sampledLobe == BSDF_LOBE_COMPOUND_THIN_SHEET_TRANSMISSION;
    ray.direction = bs.newRayDir;
    throughput = throughput * bs.throughputMul;
    let sampledDir = bs.sampledDir;
    let sampleAllowsAreaMis = bs.sampleAllowsAreaMis;
    prevSampleAllowsAreaMis = sampleAllowsAreaMis;

    if (sampleAllowsAreaMis) {
      radiance = radiance + bsdfAreaLightConnectionContribution(
        hitPos,
        normal,
        wo,
        sampledDir,
        baseColor,
        roughness,
        metallic,
        transmission,
        surfaceEtaTOverI,
        mat.clearcoat,
        mat.clearcoatRoughness,
        mat.sheen,
        mat.sheenRoughness,
        mat.sheenColor,
        mat.iridescence,
        mat.iridescenceIor,
        mat.iridescenceThicknessMin,
        mat.iridescenceThicknessMax,
        mat.specularColor,
        mat.specularIntensity,
        thinFilm,
        throughputAtVertex,
        heroLambda,
      );
      radiance = radiance + bsdfEnvironmentConnectionContribution(
        hitPos,
        normal,
        wo,
        sampledDir,
        baseColor,
        roughness,
        metallic,
        transmission,
        surfaceEtaTOverI,
        mat.clearcoat,
        mat.clearcoatRoughness,
        mat.sheen,
        mat.sheenRoughness,
        mat.sheenColor,
        mat.iridescence,
        mat.iridescenceIor,
        mat.iridescenceThicknessMin,
        mat.iridescenceThicknessMax,
        mat.specularColor,
        mat.specularIntensity,
        thinFilm,
        throughputAtVertex,
        heroLambda,
      );
    }

    // Keep the exact source token scoped to one outgoing path query. All
    // connection visibility queries above originate at this vertex and must
    // not inherit continuation suppression from the sampled path edge.
    if (crossedTransmissiveInterface) {
      if (!opticalSetContinuationSourceFromHit(hit)) { break; }
      ray.origin = hitPos;
    } else {
      opticalClearContinuationSource();
      ray.origin = bs.newRayOrigin;
    }

    if (bounce > 2u) {
      let rr = russianRoulette(&rng, throughput);
      if (!rr.survives) { break; }
      throughput = throughput * rr.throughputMul;
    }
  }

  var outRadiance = radiance;
  if (params.spectralEnabled != 0u) {
    outRadiance = heroWavelengthToRgb(heroLambda, luminance(radiance), heroPdf);
  }

  accumulateFrame(
    gid,
    outRadiance,
    firstHitValid,
    firstHitPos,
    firstHitNormal,
    firstHitAlbedo,
    firstHitDepth,
  );
}
`;
