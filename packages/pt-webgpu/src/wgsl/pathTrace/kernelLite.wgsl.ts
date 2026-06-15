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
  let sampleLum = luminance(sampleColor);
  textureStore(outputTexture, vec2i(gid.xy), vec4f(display, 1.0));
  if (firstHitValid) {
    textureStore(normalDepthTexture, vec2i(gid.xy), vec4f(firstHitNormal, firstHitDepth));
    textureStore(albedoTexture, vec2i(gid.xy), vec4f(firstHitAlbedo, 1.0));
  } else {
    textureStore(normalDepthTexture, vec2i(gid.xy), vec4f(0.0, 0.0, 0.0, 0.0));
    textureStore(albedoTexture, vec2i(gid.xy), vec4f(0.0, 0.0, 0.0, 0.0));
  }
  textureStore(varianceTexture, vec2i(gid.xy), vec4f(sampleLum, sampleLum, sampleLum, 1.0));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= params.width || gid.y >= params.height) {
    return;
  }

  var rng = pcgInit(gid.x, gid.y, params.frameSeed ^ params.frameIndex);
  let jitter = vec2f(rand_f32(&rng), rand_f32(&rng));
  var ray = generatePrimaryRay(gid.x, gid.y, jitter);

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

  for (var bounce = 0u; bounce < bounceLimit; bounce = bounce + 1u) {
    let hit = traceClosest(ray, 1e-4, INFINITY);
    if (!hit.didHit) {
      // A3 — spectralise the env at the hero λ in spectral mode (RGB unchanged).
      let envRgb = sampleEnvironmentColor(ray.direction);
      let envContribution = select(envRgb, spectralEmissionAtHero(envRgb, heroLambda), params.spectralEnabled != 0u);
      radiance = radiance + throughput * envContribution;
      break;
    }

${composeShadePrologueWgsl(SHADE_PROLOGUE_EMISSIVE_COMMENT_LITE)}
    let throughputAtVertex = throughput;
    if (transmission > 0.0 && isTranslucent) {
      var spectralMu = vec3f(spectralAvgMu);
      if (spectralSampleCount > 0u) {
        if (params.spectralEnabled != 0u) {
          let mu = sampleMaterialSpectralMu(matId, heroLambdaTo01(heroLambda));
          spectralMu = vec3f(mu);
        } else {
          let sampledMuR = sampleMaterialSpectralMu(matId, 0.15);
          let sampledMuG = sampleMaterialSpectralMu(matId, 0.50);
          let sampledMuB = sampleMaterialSpectralMu(matId, 0.85);
          spectralMu = vec3f(sampledMuR, sampledMuG, sampledMuB);
        }
      }
      // Lite tier keeps the legacy Beer-Lambert absorption (no volumetric
      // random walk — WS4 §4 degradation policy). Prefer the spectral
      // attenuation curve; otherwise fall back to the host-derived σ_a
      // (from attenuationColor/attenuationDistance) so that material data is
      // not silently dead on the compatibility tier.
      var sigmaA = select(vec3f(0.0), max(spectralMu, vec3f(0.0)), hasSpectralAttenuation);
      if (!hasSpectralAttenuation && mat.hasSigmaA) {
        sigmaA = max(mat.sigmaA, vec3f(0.0));
      }
      let sigmaS = max(scatteringRgb, vec3f(scatteringCoeff));
      let sigmaT = max(sigmaA + sigmaS, vec3f(0.0));
      if (max(sigmaT.x, max(sigmaT.y, sigmaT.z)) > 0.0) {
        throughput = throughput * exp(-sigmaT * materialAttenuationDistance(hit.dist, mat));
      }
      if (scatteringCoeff > 0.0) {
        let anisotropyBoost = 1.0 + 0.5 * scatteringAnisotropy;
        radiance = radiance + throughputAtVertex * sigmaS * (0.02 * scatteringCoeff * anisotropyBoost);
      }
    }
    let cosThetaO = max(0.0, dot(normal, wo));
    let f0 = materialSpecularF0(baseColor, metallic, mat.specularColor, mat.specularIntensity);
    let fresnel = fresnelSchlick(cosThetaO, f0);

    // B12 — lite-tier NEE: directional + env/sky + point + spot + rect-area.
    // Point/spot/rect data is loaded from liteLightTex (binding 14) via textureLoad.
    // Counts come from the UBO (params.pointLightCount / spotLightCount / rectAreaLightCount).
    // liteLightTex layout (1-row, consecutive vec4 texels):
    //   [0, pointLightCount*3):                   point records  (3 texels/light)
    //   [pointLightCount*3, +spotLightCount*4):    spot records   (4 texels/light)
    //   [that offset, +rectAreaLightCount*4):      rect records   (4 texels/light)
    let litePtBase = 0u;
    let liteSpBase = params.pointLightCount * 3u;
    let liteRcBase = liteSpBase + params.spotLightCount * 4u;

    var lightCount = 0u;
    if (params.lightDir.w > 1e-6) {
      lightCount = lightCount + 1u;
    }
    lightCount = lightCount + params.pointLightCount;
    lightCount = lightCount + params.spotLightCount;
    lightCount = lightCount + params.rectAreaLightCount;
    if (hasEnvironmentMap() || params.environmentSun.w > 1e-6) {
      lightCount = lightCount + 1u;
    }
    if (lightCount > 0u) {
      let picked = u32(min(floor(rand_f32(&rng) * f32(lightCount)), f32(lightCount - 1u)));
      var current = 0u;
      var directLi = vec3f(0.0);
      if (params.lightDir.w > 1e-6) {
        if (current == picked) {
          let lightDir = safe_normalize(params.lightDir.xyz);
          let shadowRay = Ray(hitPos + normal * 1e-3, lightDir);
          if (!traceAny(shadowRay, 1e-4, INFINITY)) {
            let nDotL = max(0.0, dot(normal, lightDir));
            let brdf = evaluateBrdfFull(baseColor, roughness, metallic, normal, wo, lightDir,
              mat.clearcoat, mat.clearcoatRoughness, mat.sheen, mat.sheenRoughness, mat.sheenColor,
              mat.iridescence, mat.iridescenceIor, mat.iridescenceThicknessMin, mat.iridescenceThicknessMax,
              mat.specularColor, mat.specularIntensity,
              0.0, 0.0);
            directLi = throughput * brdf * nDotL * params.lightDir.w;
          }
        }
        current = current + 1u;
      }
      // B12 — point lights (delta; stride 3 texels: pos, rad, [dist, decay, 0, 0]).
      for (var pi = 0u; pi < params.pointLightCount; pi = pi + 1u) {
        if (current == picked) {
          let base = litePtBase + pi * 3u;
          let lp  = textureLoad(liteLightTex, vec2i(i32(base),      0), 0).xyz;
          let rad = textureLoad(liteLightTex, vec2i(i32(base + 1u), 0), 0).rgb;
          let extra = textureLoad(liteLightTex, vec2i(i32(base + 2u), 0), 0);
          let ptMaxDist = extra.x;
          let ptDecay   = extra.y;
          let toPoint = lp - hitPos;
          let dist2 = max(dot(toPoint, toPoint), 1e-5);
          let dist = sqrt(dist2);
          if (ptMaxDist > 0.0 && dist > ptMaxDist) {
            current = current + 1u;
            continue;
          }
          let wi = toPoint / dist;
          let pointShadowRay = Ray(hitPos + normal * 1e-3, wi);
          // SHADOW-01 — extra.z carries the emitter castShadowDisabled flag.
          if (extra.z > 0.5 || !traceAny(pointShadowRay, 1e-4, max(dist - 2e-3, 1e-3))) {
            let nDotL = max(0.0, dot(normal, wi));
            let brdf = evaluateBrdfFull(baseColor, roughness, metallic, normal, wo, wi,
              mat.clearcoat, mat.clearcoatRoughness, mat.sheen, mat.sheenRoughness, mat.sheenColor,
              mat.iridescence, mat.iridescenceIor, mat.iridescenceThicknessMin, mat.iridescenceThicknessMax,
              mat.specularColor, mat.specularIntensity,
              0.0, 0.0);
            let attenuation = select(1.0 / dist2, pow(max(dist, 1.0), -ptDecay), ptDecay > 0.01);
            let radOut = select(rad, spectralEmissionAtHero(rad, heroLambda), params.spectralEnabled != 0u);
            directLi = throughput * brdf * nDotL * radOut * attenuation;
          }
        }
        current = current + 1u;
      }
      // B12 — spot lights (delta; stride 4 texels: pos, dir+cosOuter, rad+cosInner, [dist,decay,0,0]).
      for (var si = 0u; si < params.spotLightCount; si = si + 1u) {
        if (current == picked) {
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
          let dist2 = max(dot(toSpot, toSpot), 1e-5);
          let dist = sqrt(dist2);
          if (spMaxDist > 0.0 && dist > spMaxDist) {
            current = current + 1u;
            continue;
          }
          let wi = toSpot / dist;
          let coneCos = dot(-wi, spotDir);
          if (coneCos >= cosOuter) {
            let spotShadowRay = Ray(hitPos + normal * 1e-3, wi);
            // SHADOW-01 — spExtra.z carries the emitter castShadowDisabled flag.
            if (spExtra.z > 0.5 || !traceAny(spotShadowRay, 1e-4, max(dist - 2e-3, 1e-3))) {
              let nDotL = max(0.0, dot(normal, wi));
              let softness = smoothstep(cosOuter, max(cosInner, cosOuter + 1e-6), coneCos);
              let attenuation = select(1.0 / dist2, pow(max(dist, 1.0), -spDecay), spDecay > 0.01);
              let brdf = evaluateBrdfFull(baseColor, roughness, metallic, normal, wo, wi,
                mat.clearcoat, mat.clearcoatRoughness, mat.sheen, mat.sheenRoughness, mat.sheenColor,
                mat.iridescence, mat.iridescenceIor, mat.iridescenceThicknessMin, mat.iridescenceThicknessMax,
                mat.specularColor, mat.specularIntensity,
                0.0, 0.0);
              let sradOut = select(srad, spectralEmissionAtHero(srad, heroLambda), params.spectralEnabled != 0u);
              directLi = throughput * brdf * nDotL * softness * sradOut * attenuation;
            }
          }
        }
        current = current + 1u;
      }
      // B12 — rect/disc-area lights (area; stride 4 texels: rpos, ru, rv, rshape).
      // Shape discriminator in rshape.w: ≈ 0 → rect, ≈ 1 → analytic disc.
      // Disc sampling uses the concentric-disc map (Shirley & Chiu 1997); area = π·|u|².
      // Lite rect/disc lights are analytic records, not scene geometry. The
      // paired BSDF->area-light half lives in connectLite and intersects the
      // same liteLightTex records, so this NEE half applies the matching MIS
      // weight just like the full tier.
      // Native analytic disc emitters replace the 32-triangle fan, 2026-06-10 —
      // RENDER-CHANGING for disc-lit scenes, A/B in R9-B.
      for (var ri = 0u; ri < params.rectAreaLightCount; ri = ri + 1u) {
        if (current == picked) {
          let rb2 = liteRcBase + ri * 4u;
          let rpos = textureLoad(liteLightTex, vec2i(i32(rb2),      0), 0).xyz;
          let ru   = textureLoad(liteLightTex, vec2i(i32(rb2 + 1u), 0), 0).xyz;
          let rv   = textureLoad(liteLightTex, vec2i(i32(rb2 + 2u), 0), 0).xyz;
          let rshapeL = textureLoad(liteLightTex, vec2i(i32(rb2 + 3u), 0), 0);
          let rr   = rshapeL.rgb;
          let isDiscL = abs(rshapeL.w - 1.0) < 0.5;
          let xi1l = rand_f32(&rng);
          let xi2l = rand_f32(&rng);
          var lpos: vec3f;
          var area: f32;
          if (isDiscL) {
            // D9.11 — Shirley & Chiu 1997 concentric-disc map via shared kernelCore helper.
            let rradL = length(ru);
            let discL = concentricDiscSample(vec2f(xi1l * 2.0 - 1.0, xi2l * 2.0 - 1.0));
            lpos = rpos + ru * discL.x + rv * discL.y;
            area = max(PI * rradL * rradL, 1e-6);
          } else {
            lpos = rpos + ru * (xi1l * 2.0 - 1.0) + rv * (xi2l * 2.0 - 1.0);
            area = max(4.0 * length(cross(ru, rv)), 1e-6);
          }
          let toLight = lpos - hitPos;
          let dist2 = max(dot(toLight, toLight), 1e-6);
          let dist = sqrt(dist2);
          let wi = toLight / dist;
          let nDotL = max(dot(normal, wi), 0.0);
          if (nDotL > 0.0) {
            let brdf = evaluateBrdfFull(baseColor, roughness, metallic, normal, wo, wi,
              mat.clearcoat, mat.clearcoatRoughness, mat.sheen, mat.sheenRoughness, mat.sheenColor,
              mat.iridescence, mat.iridescenceIor, mat.iridescenceThicknessMin, mat.iridescenceThicknessMax,
              mat.specularColor, mat.specularIntensity,
              0.0, 0.0);
            let lightNormal = safe_normalize(cross(ru, rv));
            let cosLight = max(dot(lightNormal, -wi), 0.0);
            if (cosLight > 0.0) {
              let lightPdf = dist2 / max(cosLight * area, 1e-6);
              let brdfPdf = brdfDirectionalPdfFullSampled(baseColor, roughness, metallic, transmission, ior, normal, wo, wi,
                mat.clearcoat, mat.clearcoatRoughness, mat.sheen, mat.sheenRoughness,
                mat.iridescence, mat.iridescenceIor, mat.iridescenceThicknessMin, mat.iridescenceThicknessMax,
                mat.specularColor, mat.specularIntensity,
                0.0, 0.0);
              let misWeight = powerHeuristic(lightPdf, brdfPdf);
              let shadowRay = Ray(hitPos + normal * 1e-3, wi);
              // SHADOW-01 — rect record texel 0 .w carries castShadowDisabled.
              let rectShadowDisabledL = textureLoad(liteLightTex, vec2i(i32(rb2), 0), 0).w > 0.5;
              if (rectShadowDisabledL || !traceAny(shadowRay, 1e-4, max(dist - 2e-3, 1e-3))) {
                let rrOut = select(rr, spectralEmissionAtHero(rr, heroLambda), params.spectralEnabled != 0u);
                directLi = throughput * brdf * nDotL * rrOut * misWeight / max(lightPdf, 1e-6);
              }
            }
          }
        }
        current = current + 1u;
      }
      if ((hasEnvironmentMap() || params.environmentSun.w > 1e-6) && current == picked) {
        var envDir = vec3f(0.0, 1.0, 0.0);
        var envColor = vec3f(0.0);
        var envPdf = 0.0;
        let envSample = sampleEnvironmentImportance(&rng);
        if (envSample.pdf > 0.0) {
          envDir = envSample.wi;
          envColor = envSample.value;
          envPdf = envSample.pdf;
        } else {
          let diffSample = cosineHemisphereSample(&rng, normal);
          envDir = diffSample.wi;
          envColor = sampleEnvironmentColor(envDir);
          envPdf = max(environmentPdf(envDir), 1e-8);
        }
        let nDotL = max(dot(normal, envDir), 0.0);
        if (nDotL > 1e-6) {
          let shadowRay = Ray(hitPos + normal * 1e-3, envDir);
          if (!traceAny(shadowRay, 1e-4, INFINITY)) {
            let brdf = evaluateBrdfFull(baseColor, roughness, metallic, normal, wo, envDir,
              mat.clearcoat, mat.clearcoatRoughness, mat.sheen, mat.sheenRoughness, mat.sheenColor,
              mat.iridescence, mat.iridescenceIor, mat.iridescenceThicknessMin, mat.iridescenceThicknessMax,
              mat.specularColor, mat.specularIntensity,
              0.0, 0.0);
            let brdfPdf = brdfDirectionalPdfFullSampled(baseColor, roughness, metallic, transmission, ior, normal, wo, envDir,
              mat.clearcoat, mat.clearcoatRoughness, mat.sheen, mat.sheenRoughness,
              mat.iridescence, mat.iridescenceIor, mat.iridescenceThicknessMin, mat.iridescenceThicknessMax,
              mat.specularColor, mat.specularIntensity,
              0.0, 0.0);
            // A3 — spectralise the env radiance at the hero λ (RGB unchanged).
            let envColorOut = select(envColor, spectralEmissionAtHero(envColor, heroLambda), params.spectralEnabled != 0u);
            let misWeight = powerHeuristic(envPdf, brdfPdf);
            directLi = throughput * brdf * nDotL * envColorOut * misWeight / max(envPdf, 1e-8);
          }
        }
      }
      radiance = radiance + directLi * f32(lightCount);
    }

    let caustic = causticMode();
    if (caustic == 1u) {
      radiance = radiance + manifoldNeeContribution(
        &rng,
        hitPos,
        normal,
        wo,
        baseColor,
        roughness,
        metallic,
        transmission,
        ior,
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
        throughputAtVertex,
      );
    } else if (caustic == 2u) {
      radiance = radiance + photonMapContribution(
        &rng,
        hitPos,
        normal,
        wo,
        baseColor,
        roughness,
        metallic,
        transmission,
        throughputAtVertex,
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
      ior,
      fresnel,
      thinFilmTransmitTint,
      isTranslucent,
      mat.clearcoat,
      mat.clearcoatRoughness,
      mat.sheen,
      mat.sheenRoughness,
      mat.sheenColor,
      0.0, // anisotropy — lite tier has no aniso texture bindings; always isotropic
      0.0, // anisotropyRotation
    );
    ray.origin = bs.newRayOrigin;
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
        ior,
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
        ior,
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
        throughputAtVertex,
      );
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
