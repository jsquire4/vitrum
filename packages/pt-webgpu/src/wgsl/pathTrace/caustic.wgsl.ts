/**
 * Caustic module — the two strategy paths the main kernel dispatches when
 * `params.causticStrategy != 0`.
 *
 * Bundled here:
 *  - `perturbAroundDirection` — cone-jittered direction sampler used by MNEE
 *  - `traceSpecularTransmissiveChain` — multi-bounce specular-transmissive
 *    chain walker (shared between MNEE and the photon-map gather kernel)
 *  - `manifoldNeeContribution` — caustic strategy mode 1 (Hanika et al. 2015
 *    manifold next-event estimation)
 *  - `photonMapContribution` — caustic strategy mode 2 (Jensen 1996 photon
 *    mapping with a tiny in-shader photon pass + Gaussian gather kernel)
 *
 * Depends on FrameParams bindings (materials, lightDir, pointLights,
 * spotLights) from `material.wgsl.ts`, evaluateBrdf + brdfDirectionalPdf and
 * `buildOnb` from `bsdf.wgsl.ts`, and traceClosest/traceAny/hitMaterialId
 * from `intersection.wgsl.ts`.
 */
export const PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL = /* wgsl */ `
fn perturbAroundDirection(baseDir: vec3f, xi: vec2f, coneAngle: f32) -> vec3f {
  var t: vec3f;
  var b: vec3f;
  buildOnb(baseDir, &t, &b);
  let cosThetaMin = cos(coneAngle);
  let cosTheta = mix(cosThetaMin, 1.0, xi.x);
  let sinTheta = sqrt(max(1.0 - cosTheta * cosTheta, 0.0));
  let phi = 2.0 * PI * xi.y;
  let local = vec3f(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
  return safe_normalize(local.x * t + local.y * b + local.z * baseDir);
}

fn traceSpecularTransmissiveChain(
  startPos: vec3f,
  startNormal: vec3f,
  startDir: vec3f,
  maxChain: u32,
  exitPos: ptr<function, vec3f>,
  exitDir: ptr<function, vec3f>,
  chainAttenuation: ptr<function, vec3f>,
) -> bool {
  var ray = Ray(startPos + startNormal * 1e-3, safe_normalize(startDir));
  var att = vec3f(1.0);
  for (var step = 0u; step < 8u; step = step + 1u) {
    if (step >= maxChain) {
      *exitPos = ray.origin;
      *exitDir = ray.direction;
      *chainAttenuation = att;
      return true;
    }
    let hit = traceClosest(ray, 1e-4, INFINITY);
    if (!hit.didHit) {
      *exitPos = ray.origin;
      *exitDir = ray.direction;
      *chainAttenuation = att;
      return true;
    }
    let matId = hitMaterialId(hit);
    let m0Index = matId * MATERIAL_VEC4_STRIDE;
    let m2Index = m0Index + 2u;
    let m0 = select(vec4f(1.0, 1.0, 1.0, 0.5), materials[m0Index], m0Index < arrayLength(&materials));
    let m2 = select(vec4f(0.0, 1.5, 0.0, 0.0), materials[m2Index], m2Index < arrayLength(&materials));
    let transmission = clamp(m2.x, 0.0, 1.0);
    if (transmission <= 1e-4) {
      return false;
    }
    let ior = clamp(m2.y, 1.0, 2.5);
    let hitPos = ray.origin + ray.direction * hit.dist;
    let frontFace = dot(ray.direction, hit.normal) < 0.0;
    let surfaceNormal = select(-hit.normal, hit.normal, frontFace);
    let eta = select(ior, 1.0 / ior, frontFace);
    let refr = refract(ray.direction, surfaceNormal, eta);
    let hasRefr = dot(refr, refr) > 1e-8;
    let nextDir = select(reflect(ray.direction, surfaceNormal), safe_normalize(refr), hasRefr);
    att = att * mix(vec3f(1.0), clamp(m0.rgb, vec3f(0.0), vec3f(1.0)), 0.2) * max(transmission, 0.05);
    if (max(att.r, max(att.g, att.b)) < 1e-4) {
      return false;
    }
    ray.origin = hitPos + nextDir * 1e-3;
    ray.direction = nextDir;
  }
  *exitPos = ray.origin;
  *exitDir = ray.direction;
  *chainAttenuation = att;
  return true;
}

fn manifoldNeeContribution(
  rng: ptr<function, u32>,
  hitPos: vec3f,
  normal: vec3f,
  wo: vec3f,
  baseColor: vec3f,
  roughness: f32,
  metallic: f32,
  transmission: f32,
  ior: f32,
  throughput: vec3f,
) -> vec3f {
  if (transmission <= 1e-4 || params.lightDir.w <= 1e-6) {
    return vec3f(0.0);
  }
  let mneeSteps = clamp(params.mneeMaxIterations, 1u, 8u);
  let maxChain = clamp(params.mneeMaxChainLength, 1u, 8u);
  let baseLightDir = safe_normalize(params.lightDir.xyz);
  let coneAngle = mix(0.01, 0.12, clamp(roughness, 0.0, 1.0));
  var contribution = vec3f(0.0);
  for (var step = 0u; step < 8u; step = step + 1u) {
    if (step >= mneeSteps) {
      break;
    }
    let jitter = vec2f(rand_f32(rng), rand_f32(rng));
    let candidateDir = perturbAroundDirection(baseLightDir, jitter, coneAngle);
    let nDotL = max(dot(normal, candidateDir), 0.0);
    if (nDotL <= 1e-5) {
      continue;
    }
    var exitPos = vec3f(0.0);
    var exitDir = vec3f(0.0, 1.0, 0.0);
    var chainAtt = vec3f(1.0);
    if (!traceSpecularTransmissiveChain(hitPos, normal, candidateDir, maxChain, &exitPos, &exitDir, &chainAtt)) {
      continue;
    }
    let align = max(dot(exitDir, baseLightDir), 0.0);
    if (align <= 0.75) {
      continue;
    }
    let visibilityRay = Ray(exitPos + exitDir * 1e-3, baseLightDir);
    if (traceAny(visibilityRay, 1e-4, INFINITY)) {
      continue;
    }
    let brdf = evaluateBrdf(baseColor, roughness, metallic, normal, wo, candidateDir);
    let brdfPdf = brdfDirectionalPdf(baseColor, roughness, metallic, transmission, ior, normal, wo, candidateDir);
    let conePdf = 1.0 / max(2.0 * PI * (1.0 - cos(coneAngle)), 1e-6);
    let samplePdf = conePdf / f32(mneeSteps);
    let misWeight = powerHeuristic(samplePdf, brdfPdf);
    let lightRadiance = vec3f(params.lightDir.w) * align;
    contribution = contribution +
      throughput * chainAtt * brdf * nDotL * lightRadiance * misWeight / max(samplePdf, 1e-6);
  }
  return contribution;
}

fn photonMapContribution(
  rng: ptr<function, u32>,
  hitPos: vec3f,
  normal: vec3f,
  wo: vec3f,
  baseColor: vec3f,
  roughness: f32,
  metallic: f32,
  transmission: f32,
  throughput: vec3f,
) -> vec3f {
  var availableLightCount = 0u;
  if (params.lightDir.w > 1e-6) { availableLightCount = availableLightCount + 1u; }
  if (params.pointLightCount > 0u) { availableLightCount = availableLightCount + params.pointLightCount; }
  if (params.spotLightCount > 0u) { availableLightCount = availableLightCount + params.spotLightCount; }
  if (availableLightCount == 0u) { return vec3f(0.0); }
  let photonCount = u32(clamp(f32(params.mneeMaxIterations) * 2.0, 8.0, 32.0));
  let maxChain = clamp(params.mneeMaxChainLength, 1u, 8u);
  // Photon-gather radius in world units. Hardcoded at 0.35 for the current
  // calibration scene. Exposed as a named local so the photon density / cell
  // size relationship is easy to tune in one place. Future: lift to a params
  // field if hosts need scene-relative tuning.
  let gatherRadius = 0.35;
  let gatherRadius2 = gatherRadius * gatherRadius;
  var contribution = vec3f(0.0);
  for (var photonIdx = 0u; photonIdx < 32u; photonIdx = photonIdx + 1u) {
    if (photonIdx >= photonCount) {
      break;
    }
    let pick = u32(min(floor(rand_f32(rng) * f32(availableLightCount)), f32(availableLightCount - 1u)));
    var current = 0u;
    var photonOrigin = hitPos;
    var photonDir = vec3f(0.0, 1.0, 0.0);
    var photonFlux = vec3f(0.0);
    var seeded = false;
    if (params.lightDir.w > 1e-6) {
      if (current == pick) {
        photonOrigin = hitPos - safe_normalize(params.lightDir.xyz) * 24.0;
        photonDir = safe_normalize(params.lightDir.xyz);
        photonFlux = vec3f(params.lightDir.w);
        seeded = true;
      }
      current = current + 1u;
    }
    if (params.pointLightCount > 0u) {
      if (pick >= current && pick < current + params.pointLightCount) {
        let pointIdx = pick - current;
        let pointBase = pointIdx * 2u;
        photonOrigin = pointLights[pointBase].xyz;
        photonDir = uniformSphere(vec2f(rand_f32(rng), rand_f32(rng)));
        photonFlux = pointLights[pointBase + 1u].rgb;
        seeded = true;
      }
      current = current + params.pointLightCount;
    }
    if (params.spotLightCount > 0u && pick >= current && pick < current + params.spotLightCount) {
      let spotIdx = pick - current;
      let spotBase = spotIdx * 3u;
      photonOrigin = spotLights[spotBase].xyz;
      let coneXi = vec2f(rand_f32(rng), rand_f32(rng));
      let cosMin = spotLights[spotBase + 1u].w;
      let cosTheta = mix(cosMin, 1.0, coneXi.x);
      let sinTheta = sqrt(max(1.0 - cosTheta * cosTheta, 0.0));
      let phi = 2.0 * PI * coneXi.y;
      let local = vec3f(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
      let spotAxis = safe_normalize(-spotLights[spotBase + 1u].xyz);
      var t: vec3f;
      var b: vec3f;
      buildOnb(spotAxis, &t, &b);
      photonDir = safe_normalize(local.x * t + local.y * b + local.z * spotAxis);
      photonFlux = spotLights[spotBase + 2u].rgb;
      seeded = true;
    }
    if (!seeded) {
      continue;
    }
    var ray = Ray(photonOrigin + photonDir * 1e-3, photonDir);
    var flux = photonFlux / max(f32(photonCount), 1.0);
    for (var bounce = 0u; bounce < 8u; bounce = bounce + 1u) {
      if (bounce >= maxChain) { break; }
      let hit = traceClosest(ray, 1e-4, INFINITY);
      if (!hit.didHit) { break; }
      let matId = hitMaterialId(hit);
      let m0Index = matId * MATERIAL_VEC4_STRIDE;
      let m2Index = m0Index + 2u;
      let m0 = select(vec4f(1.0, 1.0, 1.0, 0.5), materials[m0Index], m0Index < arrayLength(&materials));
      let m2 = select(vec4f(0.0, 1.5, 0.0, 0.0), materials[m2Index], m2Index < arrayLength(&materials));
      let mTransmission = clamp(m2.x, 0.0, 1.0);
      let mIor = clamp(m2.y, 1.0, 2.5);
      let hp = ray.origin + ray.direction * hit.dist;
      let dist2ToReceiver = dot(hp - hitPos, hp - hitPos);
      if (dist2ToReceiver <= gatherRadius2) {
        let wi = -ray.direction;
        let nDotL = max(dot(normal, wi), 0.0);
        if (nDotL > 1e-6) {
          let receiverBrdf = evaluateBrdf(baseColor, roughness, metallic, normal, wo, wi);
          let kernel = exp(-dist2ToReceiver / max(2.0 * gatherRadius2, 1e-6)) / max(PI * gatherRadius2, 1e-6);
          contribution = contribution + throughput * flux * receiverBrdf * nDotL * kernel;
        }
      }
      if (mTransmission <= 1e-4) {
        break;
      }
      let frontFace = dot(ray.direction, hit.normal) < 0.0;
      let n = select(-hit.normal, hit.normal, frontFace);
      let eta = select(mIor, 1.0 / mIor, frontFace);
      let refr = refract(ray.direction, n, eta);
      let hasRefr = dot(refr, refr) > 1e-8;
      let nextDir = select(reflect(ray.direction, n), safe_normalize(refr), hasRefr);
      flux = flux * mix(vec3f(1.0), clamp(m0.rgb, vec3f(0.0), vec3f(1.0)), 0.2) * max(mTransmission, 0.05);
      if (max(flux.r, max(flux.g, flux.b)) < 1e-5) {
        break;
      }
      ray.origin = hp + nextDir * 1e-3;
      ray.direction = nextDir;
    }
  }
  let strategyScale = 1.0 + 0.25 * transmission;
  return contribution * strategyScale;
}
`;
