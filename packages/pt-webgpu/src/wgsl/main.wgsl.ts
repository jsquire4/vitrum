/**
 * Main compute kernel + per-pixel orchestration.
 *
 *  - `generatePrimaryRay` — primary-ray generation from inverse VP + sub-pixel jitter
 *  - `projectToNdc` — VP-clip projection used for motion vectors
 *  - `causticMode` — UBO accessor for the caustic-strategy selector
 *  - `RRResult` / `russianRoulette` — bounce-termination helper
 *  - `accumulateFrame` — output-texture writes + variance moments update
 *  - `main` — the @compute kernel that walks each ray through the integrator
 *
 * The kernel composes BVH traversal, light sampling, BSDF sampling, MNEE /
 * photon-map caustic strategies, and the layered direction sampler — all of
 * which live in their own modules. This file is intentionally the only place
 * that sees the full integrator's control flow.
 */
export const PT_WEBGPU_MAIN_WGSL = /* wgsl */ `
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
    let ior = mat.ior;
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

    radiance = radiance + throughput * emissive;

    let hitPos = ray.origin + ray.direction * hit.dist;
    let isFrontFace = dot(hit.normal, ray.direction) < 0.0;
    let normal = select(-hit.normal, hit.normal, isFrontFace);
    let layerTx = clamp(select(backLayerTx, frontLayerTx, isFrontFace), vec3f(0.0), vec3f(1.0));
    let layerRoughness = select(backLayerRoughness, frontLayerRoughness, isFrontFace);
    if (layerRoughness >= 0.0) {
      roughness = clamp(layerRoughness, 0.02, 1.0);
    }
    baseColor = baseColor * layerTx;
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
      let rtR = thinFilmTmmRt(matId, thinFilmLayerCountU, 630.0, ior, thinFilmIncidentIor, thinFilmAngleDependent, viewCos);
      let rtG = thinFilmTmmRt(matId, thinFilmLayerCountU, 540.0, ior, thinFilmIncidentIor, thinFilmAngleDependent, viewCos);
      let rtB = thinFilmTmmRt(matId, thinFilmLayerCountU, 460.0, ior, thinFilmIncidentIor, thinFilmAngleDependent, viewCos);
      thinFilmReflectTint = clamp(vec3f(rtR.x, rtG.x, rtB.x), vec3f(0.0), vec3f(1.0));
      thinFilmTransmitTint = clamp(vec3f(rtR.y, rtG.y, rtB.y), vec3f(0.0), vec3f(1.0));
      let layerStrength = clamp(0.12 + 0.06 * f32(thinFilmLayerCountU), 0.0, 0.55);
      let filmStrength = clamp(layerStrength * (1.0 - roughness), 0.0, 0.6);
      baseColor = mix(baseColor, baseColor * thinFilmReflectTint, filmStrength);
    }
    let throughputAtVertex = throughput;
    if (transmission > 0.0) {
      let sampledMuR = sampleMaterialSpectralMu(matId, 0.15);
      let sampledMuG = sampleMaterialSpectralMu(matId, 0.50);
      let sampledMuB = sampleMaterialSpectralMu(matId, 0.85);
      let spectralMu = select(
        vec3f(spectralAvgMu),
        vec3f(sampledMuR, sampledMuG, sampledMuB),
        spectralSampleCount > 0u,
      );
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
              let brdfPdf = brdfDirectionalPdf(baseColor, roughness, metallic, transmission, normal, wo, wi);
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
              let brdfPdf = brdfDirectionalPdf(baseColor, roughness, metallic, transmission, normal, wo, wi);
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
        let sampled = sampleEnvironmentImportance(&rng, &envDir, &envColor, &envPdf);
        if (!sampled) {
          envDir = cosineHemisphereSample(&rng, normal);
          envColor = sampleEnvironmentColor(envDir);
          envPdf = max(environmentPdf(envDir), 1e-8);
        }
        let nDotL = max(dot(normal, envDir), 0.0);
        if (nDotL > 1e-6) {
          let shadowRay = Ray(hitPos + normal * 1e-3, envDir);
          if (!traceAny(shadowRay, 1e-4, INFINITY)) {
            let brdf = evaluateBrdf(baseColor, roughness, metallic, normal, wo, envDir);
            let brdfPdf = brdfDirectionalPdf(baseColor, roughness, metallic, transmission, normal, wo, envDir);
            let misWeight = powerHeuristic(envPdf, brdfPdf);
            directLi = throughput * brdf * nDotL * envColor * misWeight / max(envPdf, 1e-8);
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
        throughputAtVertex,
      );
    }

    if (bounce > 2u) {
      let rr = russianRoulette(&rng, throughput);
      if (!rr.survives) { break; }
      throughput = throughput * rr.throughputMul;
    }
  }

  accumulateFrame(
    gid,
    radiance,
    firstHitValid,
    firstHitPos,
    firstHitNormal,
    firstHitAlbedo,
    firstHitDepth,
  );
}
`;
