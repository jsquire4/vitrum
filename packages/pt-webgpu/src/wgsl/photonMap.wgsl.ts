/**
 * Photon-map caustic estimator — caustic strategy mode 2.
 *
 * Implements a single-pass progressive photon mapping variant: per-receiver
 * pixel, emit a small number of photons from one of the available lights
 * (sun / point / spot), trace them through specular-transmissive media,
 * and gather their flux at the current shading point using a Gaussian
 * kernel of fixed radius.
 *
 * Ref: Jensen, H.W. "Global Illumination using Photon Maps." EGSR 1996.
 *      http://graphics.stanford.edu/courses/cs348b-01/course29.hanrahan.pdf
 */
export const PT_WEBGPU_PHOTON_MAP_WGSL = /* wgsl */ `
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
  if (params.pointLightCount > 0u) { availableLightCount = availableLightCount + 1u; }
  if (params.spotLightCount > 0u) { availableLightCount = availableLightCount + 1u; }
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
      if (current == pick) {
        photonOrigin = pointLights[0].xyz;
        photonDir = uniformSphere(vec2f(rand_f32(rng), rand_f32(rng)));
        photonFlux = pointLights[1].rgb;
        seeded = true;
      }
      current = current + 1u;
    }
    if (params.spotLightCount > 0u && current == pick) {
      photonOrigin = spotLights[0].xyz;
      let coneXi = vec2f(rand_f32(rng), rand_f32(rng));
      let cosMin = spotLights[1].w;
      let cosTheta = mix(cosMin, 1.0, coneXi.x);
      let sinTheta = sqrt(max(1.0 - cosTheta * cosTheta, 0.0));
      let phi = 2.0 * PI * coneXi.y;
      let local = vec3f(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
      let spotAxis = safe_normalize(-spotLights[1].xyz);
      var t: vec3f;
      var b: vec3f;
      buildOnb(spotAxis, &t, &b);
      photonDir = safe_normalize(local.x * t + local.y * b + local.z * spotAxis);
      photonFlux = spotLights[2].rgb;
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
