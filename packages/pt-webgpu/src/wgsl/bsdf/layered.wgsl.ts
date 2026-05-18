/**
 * Layered BSDF — single-bounce direction sampler.
 *
 * `sampleNextBounceDirection` partitions sampling across three lobes:
 *  - Dielectric (Fresnel-weighted reflect / refract) — PBR4e §9.3
 *  - Glossy specular — Heitz 2018 VNDF (see `bsdf/glossy.wgsl.ts`)
 *  - Cosine-hemisphere diffuse — Lambertian
 *
 * `BounceSample` packs the sampled direction with traversal bookkeeping
 * (new ray origin, throughput multiplier, area-MIS allow flag) so the
 * main kernel can advance the path in one call.
 *
 * Note (W4-A4): the plan envisions per-BSDF sample/PDF/eval triples returning
 * a shared `BsdfSample {wi, pdf, value}`. That signature unification is a
 * separate follow-up. Today's `BounceSample` keeps the legacy shape so this
 * refactor is purely code-movement (byte-equivalent compiled WGSL).
 */
export const PT_WEBGPU_BSDF_LAYERED_WGSL = /* wgsl */ `
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
      result.sampledDir = glossyReflectionSample(rng, wo, normal, tanT, tanB, roughness);
      result.newRayDir = result.sampledDir;
      result.sampleAllowsAreaMis = true;
      // Divide by branch probability R (unbiased estimator).
      result.throughputMul = fresnel / max(R, 1e-4);
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
    let wo = -incomingDir;
    result.newRayOrigin = hitPos + normal * 1e-3;
    result.sampledDir = glossyReflectionSample(rng, wo, normal, tanT, tanB, roughness);
    result.newRayDir = result.sampledDir;
    result.sampleAllowsAreaMis = true;
    result.throughputMul = fresnel / max(specProb, 1e-4);
  } else {
    result.newRayOrigin = hitPos + normal * 1e-3;
    result.sampledDir = cosineHemisphereSample(rng, normal);
    result.newRayDir = result.sampledDir;
    result.sampleAllowsAreaMis = true;
    let kd = (vec3f(1.0) - fresnel) * (1.0 - metallic);
    result.throughputMul = (kd * baseColor) / max(diffProb, 1e-4);
  }
  return result;
}
`;
