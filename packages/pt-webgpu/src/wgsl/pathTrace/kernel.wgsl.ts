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
export const PT_WEBGPU_PATH_TRACE_KERNEL_WGSL = /* wgsl */ `
fn generatePrimaryRay(px: u32, py: u32, jitter: vec2f) -> Ray {
  let uv = (vec2f(f32(px), f32(py)) + jitter) / vec2f(f32(params.width), f32(params.height));
  let ndc = vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
  let far4 = params.invViewProj * vec4f(ndc, 1.0, 1.0);
  let near4 = params.invViewProj * vec4f(ndc, -1.0, 1.0);
  let farW = far4.xyz / far4.w;
  let nearW = near4.xyz / near4.w;
  var ray: Ray;
  ray.origin = params.cameraPos.xyz;
  ray.direction = safe_normalize(farW - nearW);
  return ray;
}

fn projectToNdc(pos: vec3f, vp: mat4x4f) -> vec2f {
  let clip = vp * vec4f(pos, 1.0);
  let invW = 1.0 / max(abs(clip.w), 1e-8);
  return clip.xy * invW;
}


fn causticMode() -> u32 {
  return params.causticStrategy;
}

struct RRResult {
  survives: bool,
  throughputMul: f32,
}

fn russianRoulette(rng: ptr<function, u32>, throughput: vec3f) -> RRResult {
  let survival = clamp(max(throughput.r, max(throughput.g, throughput.b)), 0.1, 0.95);
  var result: RRResult;
  if (rand_f32(rng) > survival) {
    result.survives = false;
    result.throughputMul = 1.0;
    return result;
  }
  result.survives = true;
  result.throughputMul = 1.0 / survival;
  return result;
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
  let sampleLum = luminance(sampleColor);
  var moments = varianceMomentsBuffer[pixelIndex];
  moments.x = moments.x + sampleLum;
  moments.y = moments.y + sampleLum * sampleLum;
  moments.z = moments.z + 1.0;
  varianceMomentsBuffer[pixelIndex] = moments;

  let display = accum.xyz / max(accum.w, 1.0);
  let count = max(moments.z, 1.0);
  let mean = moments.x / count;
  let varL = max(0.0, moments.y / count - mean * mean);
  textureStore(outputTexture, vec2i(gid.xy), vec4f(display, 1.0));
  if (firstHitValid) {
    textureStore(normalDepthTexture, vec2i(gid.xy), vec4f(firstHitNormal, firstHitDepth));
    textureStore(albedoTexture, vec2i(gid.xy), vec4f(firstHitAlbedo, 1.0));
    let ndc = projectToNdc(firstHitPos, params.viewProj);
    let prevNdc = projectToNdc(firstHitPos, params.prevViewProj);
    let motionPx = (ndc - prevNdc) * 0.5 * vec2f(f32(params.width), f32(params.height));
    textureStore(motionVectorsTexture, vec2i(gid.xy), vec4f(motionPx, 0.0, 1.0));
  } else {
    textureStore(normalDepthTexture, vec2i(gid.xy), vec4f(0.0, 0.0, 0.0, 0.0));
    textureStore(albedoTexture, vec2i(gid.xy), vec4f(0.0, 0.0, 0.0, 0.0));
    textureStore(motionVectorsTexture, vec2i(gid.xy), vec4f(0.0, 0.0, 0.0, 1.0));
  }
  textureStore(varianceTexture, vec2i(gid.xy), vec4f(varL, varL, varL, 1.0));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= params.width || gid.y >= params.height) {
    return;
  }

  var rng = pcgInit(gid.x, gid.y, params.frameSeed ^ params.frameIndex);
  let jitter = vec2f(rand_f32(&rng), rand_f32(&rng));
  var ray = generatePrimaryRay(gid.x, gid.y, jitter);

  // BDPT eye-subpath scratch stack — bind this pixel for the deeply-nested
  // stack helpers (bdptEyeStackStore/Load) used by the full §10.3 connection.
  if (params.bdptEnabled != 0u) {
    bdptSetCurrentPixel(gid.y * params.width + gid.x);
  }
  // Forward scatter pdf at the previous eye vertex (the density that produced
  // the current vertex). For the primary hit E_0 the "previous vertex" is the
  // pinhole camera; its importance directional pdf is modelled as 1.0 (We for
  // an aperture-less pinhole — the one vertex without an aperture model). This
  // replaces the old hardcoded eyePdfFwd=1.0 for all SCENE-surface vertices,
  // where the real BSDF scatter pdf now flows in.
  var bdptPrevScatterPdf = 1.0;
  var bdptPrevPos = params.cameraPos.xyz;

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

  for (var bounce = 0u; bounce < bounceLimit; bounce = bounce + 1u) {
    let hit = traceClosest(ray, 1e-4, INFINITY);
    if (!hit.didHit) {
      radiance = radiance + throughput * sampleEnvironmentColor(ray.direction);
      break;
    }

    let matId = hitMaterialId(hit);
    let mat = decodeMaterial(matId);
    var baseColor = mat.baseColor;
    var roughness = mat.roughness;
    let emissive = mat.emissive;
    let metallic = mat.metallic;
    let transmission = mat.transmission;
    var ior = mat.ior;
    if (params.spectralEnabled != 0u && mat.dispersionAbbe >= 1.0) {
      ior = cauchyIorAtLambda(heroLambda, mat.ior, mat.dispersionAbbe);
    }
    let scatteringCoeff = mat.scatteringCoeff;
    let scatteringAnisotropy = mat.scatteringAnisotropy;
    let scatteringRgb = mat.scatteringRgb;
    let hasSpectralAttenuation = mat.hasSpectralAttenuation;
    let frontLayerTx = mat.frontLayerTx;
    let frontLayerRoughness = mat.frontLayerRoughness;
    let backLayerTx = mat.backLayerTx;
    let backLayerRoughness = mat.backLayerRoughness;
    let thinFilmEnabled = mat.thinFilmEnabled;
    let thinFilmLayerCountU = mat.thinFilmLayerCountU;
    let thinFilmIncidentIor = mat.thinFilmIncidentIor;
    let thinFilmAngleDependent = mat.thinFilmAngleDependent;
    let spectralAvgMu = mat.spectralAvgMu;
    let spectralSampleCount = mat.spectralSampleCount;
    let isTranslucent = mat.isTranslucent;

    radiance = radiance + throughput * emissive;

    let hitPos = ray.origin + ray.direction * hit.dist;
    let isFrontFace = dot(hit.normal, ray.direction) < 0.0;
    let normal = select(-hit.normal, hit.normal, isFrontFace);
    let layerTx = clamp(select(backLayerTx, frontLayerTx, isFrontFace), vec3f(0.0), vec3f(1.0));
    let layerRoughness = select(backLayerRoughness, frontLayerRoughness, isFrontFace);
    if (layerRoughness >= 0.0) {
      roughness = clamp(layerRoughness, 0.02, 1.0);
    }
    let layerW = select(
      layerTx,
      activeLayerWeightRgb(layerTx, heroLambda, true),
      params.spectralEnabled != 0u && luminance(layerTx) < 0.999,
    );
    baseColor = baseColor * layerW;
    if (!firstHitValid) {
      firstHitValid = true;
      firstHitPos = hitPos;
      firstHitNormal = normal;
      firstHitAlbedo = baseColor;
      firstHitDepth = hit.dist;
    }
    let wo = -ray.direction;
    var thinFilmReflectTint = vec3f(1.0);
    var thinFilmTransmitTint = vec3f(1.0);
    if (thinFilmEnabled) {
      let viewCos = clamp(dot(normal, wo), 0.0, 1.0);
      if (params.spectralEnabled != 0u) {
        let rt = thinFilmTmmRt(
          matId,
          thinFilmLayerCountU,
          heroLambda,
          ior,
          thinFilmIncidentIor,
          thinFilmAngleDependent,
          viewCos,
        );
        thinFilmReflectTint = vec3f(clamp(rt.x, 0.0, 1.0));
        thinFilmTransmitTint = vec3f(clamp(rt.y, 0.0, 1.0));
      } else {
        let rtR = thinFilmTmmRt(matId, thinFilmLayerCountU, 630.0, ior, thinFilmIncidentIor, thinFilmAngleDependent, viewCos);
        let rtG = thinFilmTmmRt(matId, thinFilmLayerCountU, 540.0, ior, thinFilmIncidentIor, thinFilmAngleDependent, viewCos);
        let rtB = thinFilmTmmRt(matId, thinFilmLayerCountU, 460.0, ior, thinFilmIncidentIor, thinFilmAngleDependent, viewCos);
        thinFilmReflectTint = clamp(vec3f(rtR.x, rtG.x, rtB.x), vec3f(0.0), vec3f(1.0));
        thinFilmTransmitTint = clamp(vec3f(rtR.y, rtG.y, rtB.y), vec3f(0.0), vec3f(1.0));
      }
      let layerStrength = clamp(0.12 + 0.06 * f32(thinFilmLayerCountU), 0.0, 0.55);
      let filmStrength = clamp(layerStrength * (1.0 - roughness), 0.0, 0.6);
      baseColor = mix(baseColor, baseColor * thinFilmReflectTint, filmStrength);
    }
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
      let sigmaA = select(vec3f(0.0), max(spectralMu, vec3f(0.0)), hasSpectralAttenuation);
      let sigmaS = max(scatteringRgb, vec3f(scatteringCoeff));
      let sigmaT = max(sigmaA + sigmaS, vec3f(0.0));
      if (max(sigmaT.x, max(sigmaT.y, sigmaT.z)) > 0.0) {
        throughput = throughput * exp(-sigmaT * hit.dist);
      }
      if (scatteringCoeff > 0.0) {
        let anisotropyBoost = 1.0 + 0.5 * scatteringAnisotropy;
        radiance = radiance + throughputAtVertex * sigmaS * (0.02 * scatteringCoeff * anisotropyBoost);
      }
    }
    let cosThetaO = max(0.0, dot(normal, wo));
    let f0 = mix(vec3f(0.04), baseColor, metallic);
    let fresnel = fresnelSchlick(cosThetaO, f0);

    var lightCount = 0u;
    if (params.lightDir.w > 1e-6) {
      lightCount = lightCount + 1u;
    }
    lightCount = lightCount + params.pointLightCount;
    lightCount = lightCount + params.spotLightCount;
    lightCount = lightCount + params.rectAreaLightCount;
    lightCount = lightCount + params.meshAreaLightCount;
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
            let brdf = evaluateBrdf(baseColor, roughness, metallic, normal, wo, lightDir);
            directLi = throughput * brdf * nDotL * params.lightDir.w;
          }
        }
        current = current + 1u;
      }
      for (var pi = 0u; pi < params.pointLightCount; pi = pi + 1u) {
        if (current == picked) {
          let base = pi * 2u;
          let lp = pointLights[base].xyz;
          let rad = pointLights[base + 1u].rgb;
          let toPoint = lp - hitPos;
          let dist2 = max(dot(toPoint, toPoint), 1e-5);
          let dist = sqrt(dist2);
          let wi = toPoint / dist;
          let pointShadowRay = Ray(hitPos + normal * 1e-3, wi);
          if (!traceAny(pointShadowRay, 1e-4, max(dist - 2e-3, 1e-3))) {
            let nDotL = max(0.0, dot(normal, wi));
            let brdf = evaluateBrdf(baseColor, roughness, metallic, normal, wo, wi);
            directLi = throughput * brdf * nDotL * (rad / dist2);
          }
        }
        current = current + 1u;
      }
      for (var si = 0u; si < params.spotLightCount; si = si + 1u) {
        if (current == picked) {
          let sb = si * 3u;
          let spos = spotLights[sb].xyz;
          let saxis = spotLights[sb + 1u];
          let srad = spotLights[sb + 2u].rgb;
          let spotDir = safe_normalize(saxis.xyz);
          let cosOuter = saxis.w;
          let toSpot = spos - hitPos;
          let dist2 = max(dot(toSpot, toSpot), 1e-5);
          let dist = sqrt(dist2);
          let wi = toSpot / dist;
          let coneCos = dot(-wi, spotDir);
          if (coneCos >= cosOuter) {
            let spotShadowRay = Ray(hitPos + normal * 1e-3, wi);
            if (!traceAny(spotShadowRay, 1e-4, max(dist - 2e-3, 1e-3))) {
              let nDotL = max(0.0, dot(normal, wi));
              let softness = smoothstep(cosOuter, 1.0, coneCos);
              let brdf = evaluateBrdf(baseColor, roughness, metallic, normal, wo, wi);
              directLi = throughput * brdf * nDotL * softness * (srad / dist2);
            }
          }
        }
        current = current + 1u;
      }
      for (var ri = 0u; ri < params.rectAreaLightCount; ri = ri + 1u) {
        if (current == picked) {
          let rb = ri * 4u;
          let rpos = rectAreaLights[rb].xyz;
          let ru = rectAreaLights[rb + 1u].xyz;
          let rv = rectAreaLights[rb + 2u].xyz;
          let rr = rectAreaLights[rb + 3u].rgb;
          let u = rand_f32(&rng) * 2.0 - 1.0;
          let v = rand_f32(&rng) * 2.0 - 1.0;
          let lpos = rpos + ru * u + rv * v;
          let toLight = lpos - hitPos;
          let dist2 = max(dot(toLight, toLight), 1e-6);
          let dist = sqrt(dist2);
          let wi = toLight / dist;
          let nDotL = max(dot(normal, wi), 0.0);
          if (nDotL > 0.0) {
            let brdf = evaluateBrdf(baseColor, roughness, metallic, normal, wo, wi);
            let lightNormal = safe_normalize(cross(ru, rv));
            let cosLight = max(dot(lightNormal, -wi), 0.0);
            if (cosLight > 0.0) {
              let area = max(4.0 * length(cross(ru, rv)), 1e-6);
              let lightPdf = dist2 / max(cosLight * area, 1e-6);
              let brdfPdf = brdfDirectionalPdf(baseColor, roughness, metallic, transmission, ior, normal, wo, wi);
              let misWeight = powerHeuristic(lightPdf, brdfPdf);
              let shadowRay = Ray(hitPos + normal * 1e-3, wi);
              if (!traceAny(shadowRay, 1e-4, max(dist - 2e-3, 1e-3))) {
                directLi = throughput * brdf * nDotL * rr * misWeight / max(lightPdf, 1e-6);
              }
            }
          }
        }
        current = current + 1u;
      }
      for (var mi = 0u; mi < params.meshAreaLightCount; mi = mi + 1u) {
        if (current == picked) {
          let mb = mi * 4u;
          let a = meshAreaLights[mb].xyz;
          let b = meshAreaLights[mb + 1u].xyz;
          let c = meshAreaLights[mb + 2u].xyz;
          let mr = meshAreaLights[mb + 3u].rgb;
          let r1 = rand_f32(&rng);
          let r2 = rand_f32(&rng);
          let su = sqrt(r1);
          let uu = 1.0 - su;
          let vv = r2 * su;
          let ww = 1.0 - uu - vv;
          let lpos = a * uu + b * vv + c * ww;
          let toLight = lpos - hitPos;
          let dist2 = max(dot(toLight, toLight), 1e-6);
          let dist = sqrt(dist2);
          let wi = toLight / dist;
          let nDotL = max(dot(normal, wi), 0.0);
          if (nDotL > 0.0) {
            let brdf = evaluateBrdf(baseColor, roughness, metallic, normal, wo, wi);
            let lightNormal = safe_normalize(cross(b - a, c - a));
            let cosLight = max(dot(lightNormal, -wi), 0.0);
            if (cosLight > 0.0) {
              let area = max(0.5 * length(cross(b - a, c - a)), 1e-6);
              let lightPdf = dist2 / max(cosLight * area, 1e-6);
              let brdfPdf = brdfDirectionalPdf(baseColor, roughness, metallic, transmission, ior, normal, wo, wi);
              let misWeight = powerHeuristic(lightPdf, brdfPdf);
              let shadowRay = Ray(hitPos + normal * 1e-3, wi);
              if (!traceAny(shadowRay, 1e-4, max(dist - 2e-3, 1e-3))) {
                directLi = throughput * brdf * nDotL * mr * misWeight / max(lightPdf, 1e-6);
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
            let brdf = evaluateBrdf(baseColor, roughness, metallic, normal, wo, envDir);
            let brdfPdf = brdfDirectionalPdf(baseColor, roughness, metallic, transmission, ior, normal, wo, envDir);
            let misWeight = powerHeuristic(envPdf, brdfPdf);
            directLi = throughput * brdf * nDotL * envColor * misWeight / max(envPdf, 1e-8);
          }
        }
      }
      radiance = radiance + directLi * f32(lightCount);
    }

    if (params.bdptEnabled != 0u) {
      // Push this eye vertex (E_bounce) onto the scratch stack BEFORE connecting:
      // pdfRev = forward scatter pdf at the previous vertex that produced E_bounce
      // (camera importance 1.0 for the primary hit). pdfFwd is filled one
      // iteration later by the swapped-direction reverse density (and overridden
      // by the connection straddle terms when this vertex is E_e / E_{e-1}).
      let eyeIsSpecular = transmission > 0.5 && roughness < 0.05;
      bdptEyeStackStore(bounce, hitPos, normal, 0.0, bdptPrevScatterPdf, eyeIsSpecular);
      // Skip the primary hit (bounce 0): an explicit connection there would
      // double-count with the unidirectional NEE above (fork !state.firstRay).
      if (bounce > 0u) {
        let maxLv = min(params.bdptMaxLightBounces, 3u);
        for (var lvi = 0u; lvi < maxLv; lvi++) {
          radiance = radiance + evaluateBdptConnection(
            hitPos,
            normal,
            wo,
            throughputAtVertex,
            baseColor,
            roughness,
            metallic,
            transmission,
            ior,
            bounce,
            i32(lvi),
          );
        }
      }
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
    );
    ray.origin = bs.newRayOrigin;
    ray.direction = bs.newRayDir;
    throughput = throughput * bs.throughputMul;
    let sampledDir = bs.sampledDir;
    let sampleAllowsAreaMis = bs.sampleAllowsAreaMis;

    if (params.bdptEnabled != 0u) {
      // The forward scatter pdf of the chosen next direction at E_bounce — fed
      // to the next iteration as E_{bounce+1}'s reverse density. (eyePdfFwd is
      // now this real value, not the old hardcoded 1.0.)
      let scatterPdfFwd = brdfDirectionalPdf(
        baseColor, roughness, metallic, transmission, ior, normal, wo, sampledDir);
      // Merged pdfFwd(E_{bounce-1}): swapped-direction reverse density at the
      // PREVIOUS eye vertex toward E_bounce, using the natural next eye direction
      // as wo (PBRT camera[d-1].pdfRev set while at camera[d]). Write into the
      // previous scratch slot (D1 — non-symmetric reverse density).
      if (bounce >= 1u) {
        let toPrev = safe_normalize(bdptPrevPos - hitPos);
        let swappedRev = brdfDirectionalPdf(
          baseColor, roughness, metallic, transmission, ior, normal, sampledDir, toPrev);
        bdptEyeStackSetFwd(bounce - 1u, swappedRev);
      }
      bdptPrevScatterPdf = scatterPdfFwd;
      bdptPrevPos = hitPos;
    }

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
        throughputAtVertex,
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
