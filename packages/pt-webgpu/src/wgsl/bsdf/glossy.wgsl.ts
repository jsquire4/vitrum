/**
 * Glossy microfacet sampling — Heitz 2018 VNDF + helper ONB + cosine-hemisphere.
 *
 *  - `buildOnb`, `cosineHemisphereSample` — cosine-weighted hemisphere sample
 *    (also used by the diffuse lobe in the main kernel).
 *  - `sampleGgxVndfTangent` — Heitz 2018 VNDF Algorithm 1 (tangent-space).
 *  - `glossyReflectionSample` — world-space wrapper used by both glossy and
 *    dielectric reflection branches.
 *
 * Ref: Heitz, E. "Sampling the GGX Distribution of Visible Normals."
 *      JCGT 7(4):1–13, 2018. https://jcgt.org/published/0007/04/01/paper.pdf
 */
export const PT_WEBGPU_BSDF_GLOSSY_WGSL = /* wgsl */ `
fn buildOnb(n: vec3f, t: ptr<function, vec3f>, b: ptr<function, vec3f>) {
  var up = vec3f(0.0, 1.0, 0.0);
  if (abs(n.y) > 0.999) {
    up = vec3f(1.0, 0.0, 0.0);
  }
  *t = normalize(cross(up, n));
  *b = cross(n, *t);
}

fn cosineHemisphereSample(rng: ptr<function, u32>, n: vec3f) -> vec3f {
  let u1 = rand_f32(rng);
  let u2 = rand_f32(rng);
  let r = sqrt(u1);
  let phi = 2.0 * PI * u2;
  let local = vec3f(r * cos(phi), r * sin(phi), sqrt(max(0.0, 1.0 - u1)));
  var t: vec3f;
  var b: vec3f;
  buildOnb(n, &t, &b);
  return safe_normalize(local.x * t + local.y * b + local.z * n);
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
 * Returns the world-space reflection direction.
 * Ref: Heitz 2018 VNDF Algorithm 1 (see sampleGgxVndfTangent above).
 */
fn glossyReflectionSample(rng: ptr<function, u32>, wo: vec3f, n: vec3f, t: vec3f, b: vec3f, roughness: f32) -> vec3f {
  let alpha   = max(roughness * roughness, 0.001);
  let woLocal = vec3f(dot(wo, t), dot(wo, b), dot(wo, n));
  let hLocal  = sampleGgxVndfTangent(woLocal, alpha, rng);
  let hWorld  = safe_normalize(hLocal.x * t + hLocal.y * b + hLocal.z * n);
  return safe_normalize(reflect(-wo, hWorld));
}
`;
