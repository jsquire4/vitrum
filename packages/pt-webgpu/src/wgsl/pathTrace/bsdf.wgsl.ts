/**
 * BSDF module — BRDF evaluation, directional PDF, diffuse / glossy samplers,
 * and the layered single-bounce direction sampler (`sampleNextBounceDirection`).
 *
 * Bundled here:
 *  - `evaluateBrdf` — Cook-Torrance unified diffuse + specular BRDF eval
 *  - `brdfDirectionalPdf` — three-lobe MIS-aware directional PDF (VNDF
 *    reflection PDF aligned with `glossyReflectionSample`)
 *  - `buildOnb` — orthonormal basis around a surface normal
 *  - `cosineHemisphereSample` — Lambertian diffuse sampler returning BsdfSample
 *  - `sampleGgxVndfTangent` — Heitz 2018 VNDF Algorithm 1
 *  - `glossyReflectionSample` — VNDF reflection sampler in world space
 *  - `BounceSample` struct + `sampleNextBounceDirection` — layered
 *    dielectric / glossy / diffuse partition that drives the main kernel
 *
 * Depends on Fresnel / microfacet primitives (`fresnelSchlick`, `frDielectric`,
 * `ggxD`, `smithG1`) and `luminance` from `material.wgsl.ts`.
 */
export const PT_WEBGPU_PATH_TRACE_BSDF_WGSL = /* wgsl */ `
fn evaluateBrdf(baseColor: vec3f, roughness: f32, metallic: f32, normal: vec3f, wo: vec3f, wi: vec3f) -> vec3f {
  let nDotL = max(dot(normal, wi), 0.0);
  let nDotV = max(dot(normal, wo), 0.0);
  if (nDotL <= 1e-5 || nDotV <= 1e-5) {
    return vec3f(0.0);
  }
  let h = safe_normalize(wi + wo);
  let nDotH = max(dot(normal, h), 0.0);
  let vDotH = max(dot(wo, h), 0.0);
  let f0 = mix(vec3f(0.04), baseColor, metallic);
  let f = fresnelSchlick(vDotH, f0);
  let alpha = max(roughness * roughness, 1e-3);
  let d = ggxD(nDotH, alpha);
  let g = smithG1(nDotV, roughness) * smithG1(nDotL, roughness);
  let spec = (d * g) * f / max(4.0 * nDotV * nDotL, 1e-6);
  let kd = (vec3f(1.0) - f) * (1.0 - metallic);
  let diff = kd * baseColor * INV_PI;
  return diff + spec;
}

fn brdfDirectionalPdf(baseColor: vec3f, roughness: f32, metallic: f32, transmission: f32, normal: vec3f, wo: vec3f, wi: vec3f) -> f32 {
  let wiDotN = dot(normal, wi);
  let woDotN = dot(normal, wo);
  let nDotV = max(woDotN, 0.0);
  if (nDotV <= 1e-5) {
    return 0.0;
  }
  let h = safe_normalize(wi + wo);
  let nDotH = max(dot(normal, h), 0.0);
  let vDotH = max(dot(wo, h), 1e-6);
  let f0 = mix(vec3f(0.04), baseColor, metallic);
  let fresnel = fresnelSchlick(vDotH, f0);
  let baseSpecProb = clamp(mix(0.04, 0.96, max(luminance(fresnel), metallic)), 0.04, 0.96);
  let baseTransProb = clamp(transmission * (1.0 - metallic), 0.0, 0.95);
  let baseDiffProb = max(0.0, (1.0 - metallic) * (1.0 - transmission));
  let sumProb = max(baseSpecProb + baseTransProb + baseDiffProb, 1e-4);
  let specProb = baseSpecProb / sumProb;
  let transProb = baseTransProb / sumProb;
  let diffProb = baseDiffProb / sumProb;
  let sameHemisphere = wiDotN * woDotN > 0.0;
  if (!sameHemisphere) {
    let nDotT = max(abs(wiDotN), 1e-5);
    let pdfTransApprox = nDotT * INV_PI;
    return max(transProb * pdfTransApprox, 1e-8);
  }
  let nDotL = max(wiDotN, 0.0);
  if (nDotL <= 1e-5) {
    return 0.0;
  }
  let alpha = max(roughness * roughness, 1e-3);
  let d = ggxD(nDotH, alpha);
  // VNDF reflection PDF (Heitz 2018 JCGT 7(4) §3, Eq. 17):
  //   p_VNDF(h | wo) = D(h) · G1(wo) · max(0, wo·h) / (N·wo)
  // With reflection Jacobian dω_h/dω_wi = 1/(4·|wo·h|), this collapses to
  //   p_VNDF(wi | wo) = D(h) · G1(wo) / (4 · N·wo)
  // which matches the glossyReflectionSample sampler (sampleGgxVndfTangent).
  // Earlier revisions used the NDF half-vector PDF (d · N·h / (4 · wo·h));
  // that distribution and the VNDF sampler disagree, biasing MIS weights.
  let g1Wo = smithG1(nDotV, roughness);
  let pdfSpec = (d * g1Wo) / max(4.0 * nDotV, 1e-6);
  let pdfDiff = nDotL * INV_PI;
  return diffProb * pdfDiff + specProb * pdfSpec;
}

fn buildOnb(n: vec3f, t: ptr<function, vec3f>, b: ptr<function, vec3f>) {
  var up = vec3f(0.0, 1.0, 0.0);
  if (abs(n.y) > 0.999) {
    up = vec3f(1.0, 0.0, 0.0);
  }
  *t = normalize(cross(up, n));
  *b = cross(n, *t);
}

// Cosine-weighted hemisphere sampler — diffuse BRDF.
// Returns a BsdfSample where wi is the sampled world-space direction,
// pdf = cos(θ)/π, and value = vec3f(INV_PI) (unitless Lambertian kernel;
// callers multiply by albedo at the throughput level — matches the existing
// pattern in sampleNextBounceDirection).
// Same RNG consumption (two rand_f32 calls) and identical sampled direction
// as the prior vec3f-returning signature.
fn cosineHemisphereSample(rng: ptr<function, u32>, n: vec3f) -> BsdfSample {
  let u1 = rand_f32(rng);
  let u2 = rand_f32(rng);
  let r = sqrt(u1);
  let phi = 2.0 * PI * u2;
  let cosTheta = sqrt(max(0.0, 1.0 - u1));
  let local = vec3f(r * cos(phi), r * sin(phi), cosTheta);
  var t: vec3f;
  var b: vec3f;
  buildOnb(n, &t, &b);
  var result: BsdfSample;
  result.wi = safe_normalize(local.x * t + local.y * b + local.z * n);
  result.pdf = cosTheta * INV_PI;
  result.value = vec3f(INV_PI);
  return result;
}

/**
 * Heitz 2018 VNDF sample (Algorithm 1).
 * Input: wo in surface tangent-space (N = +Z); alpha = roughness².
 * Output: sampled half-vector h in tangent-space.
 * Ref: Heitz, E. "Sampling the GGX Distribution of Visible Normals."
 *      JCGT 7(4):1–13, 2018. https://jcgt.org/published/0007/04/01/paper.pdf
 */
fn sampleGgxVndfTangent(wo: vec3f, alpha: f32, rng: ptr<function, u32>) -> vec3f {
  // Step 1: stretch wo into the unit-roughness configuration.
  let Vh = safe_normalize(vec3f(alpha * wo.x, alpha * wo.y, wo.z));
  // Step 2: ONB around Vh (Frisvad-style, no branching on y).
  let lensq = Vh.x * Vh.x + Vh.y * Vh.y;
  let T1 = select(
    vec3f(1.0, 0.0, 0.0),
    vec3f(-Vh.y, Vh.x, 0.0) * inverseSqrt(lensq),
    lensq > 1e-10,
  );
  let T2 = cross(Vh, T1);
  // Step 3: sample point on unit disc with polar mapping, project onto hemisphere.
  let u1 = rand_f32(rng);
  let u2 = rand_f32(rng);
  let r   = sqrt(u1);
  let phi = 2.0 * PI * u2;
  let t1  = r * cos(phi);
  var t2  = r * sin(phi);
  let s   = 0.5 * (1.0 + Vh.z);
  // Lerp between the two extreme projections to match the hemisphere distribution.
  t2 = (1.0 - s) * sqrt(max(0.0, 1.0 - t1 * t1)) + s * t2;
  // Step 4: reproject onto hemisphere, unstretch back to ellipsoid frame.
  let Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0, 1.0 - t1 * t1 - t2 * t2)) * Vh;
  return safe_normalize(vec3f(alpha * Nh.x, alpha * Nh.y, max(1e-6, Nh.z)));
}

/**
 * Sample a glossy reflection direction via Heitz 2018 VNDF.
 * All inputs in WORLD space; n is the surface normal; t, b are
 * surface-tangent ONB axes (caller computes via buildOnb).
 * Returns a BsdfSample where:
 *   wi    — world-space reflection direction
 *   pdf   — GGX half-vector PDF d * nDotH / (4 * vDotH), matching the
 *           convention used by brdfDirectionalPdf for MIS consistency
 *   value — unitless microfacet specular kernel D * G / (4 * nDotV * nDotL);
 *           Fresnel and albedo are integrated by callers at the throughput
 *           level (matches sampleNextBounceDirection's existing pattern).
 * Same RNG consumption (two rand_f32 calls inside sampleGgxVndfTangent) and
 * identical sampled direction as the prior vec3f-returning signature.
 * Ref: Heitz 2018 VNDF Algorithm 1 (see sampleGgxVndfTangent above);
 *      PBR4e §9.6 for the BRDF kernel decomposition.
 */
fn glossyReflectionSample(rng: ptr<function, u32>, wo: vec3f, n: vec3f, t: vec3f, b: vec3f, roughness: f32) -> BsdfSample {
  let alpha   = max(roughness * roughness, 0.001);
  let woLocal = vec3f(dot(wo, t), dot(wo, b), dot(wo, n));
  let hLocal  = sampleGgxVndfTangent(woLocal, alpha, rng);
  let hWorld  = safe_normalize(hLocal.x * t + hLocal.y * b + hLocal.z * n);
  let wi      = safe_normalize(reflect(-wo, hWorld));

  var result: BsdfSample;
  result.wi = wi;
  // Compute pdf + value at the sampled wi. These are populated so future MIS
  // code paths can read them without redoing the eval; today's callers in
  // sampleNextBounceDirection still only consume .wi.
  let nDotV = max(dot(n, wo), 1e-6);
  let nDotL = max(dot(n, wi), 0.0);
  let nDotH = max(dot(n, hWorld), 0.0);
  let vDotH = max(dot(wo, hWorld), 1e-6);
  let d = ggxD(nDotH, alpha);
  if (nDotL <= 1e-5) {
    result.pdf = 0.0;
    result.value = vec3f(0.0);
  } else {
    result.pdf = d * nDotH / max(4.0 * vDotH, 1e-6);
    let g = smithG1(nDotV, roughness) * smithG1(nDotL, roughness);
    result.value = vec3f((d * g) / max(4.0 * nDotV * nDotL, 1e-6));
  }
  return result;
}

struct BounceSample {
  newRayOrigin: vec3f,
  newRayDir: vec3f,
  throughputMul: vec3f,
  sampledDir: vec3f,
  sampleAllowsAreaMis: bool,
}

fn sampleNextBounceDirection(
  rng: ptr<function, u32>,
  incomingDir: vec3f,
  hitPos: vec3f,
  hitNormal: vec3f,
  normal: vec3f,
  baseColor: vec3f,
  roughness: f32,
  metallic: f32,
  transmission: f32,
  ior: f32,
  fresnel: vec3f,
  thinFilmTransmitTint: vec3f,
) -> BounceSample {
  // Build surface-tangent ONB once; shared by both glossy-reflect call sites.
  var tanT: vec3f;
  var tanB: vec3f;
  buildOnb(normal, &tanT, &tanB);

  var result: BounceSample;
  result.sampledDir = vec3f(0.0);
  result.sampleAllowsAreaMis = false;

  // -----------------------------------------------------------------------
  // Transmissive (dielectric) surface: Fresnel-weighted reflect/refract
  // partition per PBR4e §9.3 FrDielectric.
  // Ref: Pharr, Jakob, Humphreys. PBR 4th ed. §9.3 "Specular Reflection and
  //      Transmission" — DielectricBxDF::Sample_f.
  //      https://pbr-book.org/4ed/Reflection_Models/Dielectric_BSDF
  // -----------------------------------------------------------------------
  if (transmission > 0.0 && metallic == 0.0) {
    let cosThetaI = abs(dot(-incomingDir, normal));
    let R = frDielectric(cosThetaI, ior);  // PBR4e §9.3 FrDielectric
    let xi = rand_f32(rng);
    let frontFace = dot(incomingDir, hitNormal) < 0.0;
    if (xi < R) {
      // Fresnel-weighted specular reflection branch.
      // frDielectric returns 1.0 for TIR, so TIR is handled automatically
      // (the refract branch is never taken when R == 1).
      let wo = -incomingDir; // eye-side direction
      result.newRayOrigin = hitPos + normal * 1e-3;
      let bs = glossyReflectionSample(rng, wo, normal, tanT, tanB, roughness);
      result.sampledDir = bs.wi;
      result.newRayDir = bs.wi;
      result.sampleAllowsAreaMis = true;
      // MC estimator for VNDF sampling of the GGX BRDF (Heitz 2018):
      //   f·cosθ / p_VNDF = [D·G·F / (4·NdotV·NdotL)] · NdotL
      //                    / [D·G1(wo) / (4·NdotV)]
      //                    = F · G1(wi)
      // The Fresnel branch probability R is the partition weight, so the
      // throughput multiplier is F · G1(wi) / R.
      let nDotL = max(dot(normal, result.sampledDir), 0.0);
      let g1Wi = smithG1(nDotL, roughness);
      result.throughputMul = fresnel * g1Wi / max(R, 1e-4);
    } else {
      // Fresnel-weighted refraction branch.
      let eta = select(ior, 1.0 / ior, frontFace);
      let refr = refract(incomingDir, normal, eta);
      let outDir = safe_normalize(refr);
      let offsetN = select(-normal, normal, dot(outDir, normal) > 0.0);
      result.newRayOrigin = hitPos + offsetN * 1e-3;
      result.sampledDir = outDir;
      result.newRayDir = outDir;
      // Divide by branch probability (1 - R); apply thin-film transmittance tint.
      result.throughputMul = mix(vec3f(1.0), baseColor, 0.15) * thinFilmTransmitTint / max(1.0 - R, 1e-4);
    }
    return result;
  }

  // -----------------------------------------------------------------------
  // Non-transmissive surface: heuristic specular / diffuse partition.
  // -----------------------------------------------------------------------
  let baseSpecProb = clamp(mix(0.04, 0.96, max(luminance(fresnel), metallic)), 0.04, 0.96);
  let baseDiffProb = max(0.0, 1.0 - baseSpecProb);
  let sumProb = max(baseSpecProb + baseDiffProb, 1e-4);
  let specProb = baseSpecProb / sumProb;
  let diffProb = baseDiffProb / sumProb;
  let xi2 = rand_f32(rng);
  if (xi2 < specProb) {
    // Glossy specular reflection — Heitz 2018 VNDF.
    // Ref: Heitz 2018 VNDF Algorithm 1 (see glossyReflectionSample).
    // MC estimator for VNDF sampling collapses to F · G1(wi). Without the
    // G1(wi) factor (or with the NDF half-vector PDF) grazing reflections
    // are over-estimated; see brdfDirectionalPdf for the matching PDF.
    let wo = -incomingDir;
    result.newRayOrigin = hitPos + normal * 1e-3;
    let bs = glossyReflectionSample(rng, wo, normal, tanT, tanB, roughness);
    result.sampledDir = bs.wi;
    result.newRayDir = bs.wi;
    result.sampleAllowsAreaMis = true;
    let nDotL = max(dot(normal, result.sampledDir), 0.0);
    let g1Wi = smithG1(nDotL, roughness);
    result.throughputMul = fresnel * g1Wi / max(specProb, 1e-4);
  } else {
    result.newRayOrigin = hitPos + normal * 1e-3;
    let bs = cosineHemisphereSample(rng, normal);
    result.sampledDir = bs.wi;
    result.newRayDir = bs.wi;
    result.sampleAllowsAreaMis = true;
    let kd = (vec3f(1.0) - fresnel) * (1.0 - metallic);
    result.throughputMul = (kd * baseColor) / max(diffProb, 1e-4);
  }
  return result;
}
`;
