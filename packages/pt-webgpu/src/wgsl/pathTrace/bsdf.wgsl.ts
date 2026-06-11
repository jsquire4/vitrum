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
// ============================================================
// H52 — Disney extension lobes: clearcoat / sheen / iridescence
// ============================================================

// ── Iridescence (thin-film Fresnel modification of specular F0) ───────────────
// Ported from packages/pt-webgl2/src/glsl/shader/bsdf/iridescence_functions.glsl.js
// which implements Belcour & Barla, "A Practical Extension to Microfacet Theory
// for the Modeling of Varying Iridescence," ACM TOG 36(4) (SIGGRAPH 2017).
// https://hal.archives-ouvertes.fr/hal-01518344/document
//
// WGSL translation notes: component-wise comparisons use select() (no ternary),
// vector comparisons use all()/any().  No implicit vec3 ternary in WGSL.
//
// Refs: glTF KHR_materials_iridescence; Belcour & Barla 2017 (§4 Analytic
//       Spectral Integration); Schlick 1994.
//
// XYZ → sRGB (Rec. 709) colour matrix (row-major → applied as column multiply).
fn iridXyzToRec709(xyz: vec3f) -> vec3f {
  return vec3f(
     3.2404542 * xyz.x - 1.5371385 * xyz.y - 0.4985314 * xyz.z,
    -0.9692660 * xyz.x + 1.8760108 * xyz.y + 0.0415560 * xyz.z,
     0.0556434 * xyz.x - 0.2040259 * xyz.y + 1.0572252 * xyz.z,
  );
}

// F0 → IOR (Schlick inverse — clamp F0 away from 1 to avoid sqrt(0)/div-by-zero).
fn iridFresnel0ToIor(f0: vec3f) -> vec3f {
  let sqrtF0 = sqrt(clamp(f0, vec3f(0.0), vec3f(0.9999)));
  return (vec3f(1.0) + sqrtF0) / (vec3f(1.0) - sqrtF0);
}

// Scalar IOR → F0.
fn iridIorToFresnel0Scalar(transmittedIor: f32, incidentIor: f32) -> f32 {
  let r = (transmittedIor - incidentIor) / (transmittedIor + incidentIor);
  return r * r;
}

// Vec3 IOR → F0.
fn iridIorToFresnel0Vec(transmittedIor: vec3f, incidentIor: f32) -> vec3f {
  let r = (transmittedIor - vec3f(incidentIor)) / (transmittedIor + vec3f(incidentIor));
  return r * r;
}

// Schlick Fresnel at a scalar F0 (used for the air-film interface).
fn iridSchlickScalar(cosTheta: f32, f0: f32) -> f32 {
  let m = clamp(1.0 - cosTheta, 0.0, 1.0);
  let m2 = m * m;
  return f0 + (1.0 - f0) * m2 * m2 * m;
}

// Schlick Fresnel at a vec3 F0 (used for the film-substrate interface).
fn iridSchlickVec(cosTheta: f32, f0: vec3f) -> vec3f {
  let m = clamp(1.0 - cosTheta, 0.0, 1.0);
  let m2 = m * m;
  return f0 + (vec3f(1.0) - f0) * m2 * m2 * m;
}

// Evaluate the CIE-XYZ sensitivity functions as a sum of Gaussians (Belcour 2017 §4).
// OPD = optical path difference in metres (≈ 2·n·d·cosTheta).
// shift = per-channel phase shift vec (from interface phase mismatch).
fn iridEvalSensitivity(OPD: f32, shift: vec3f) -> vec3f {
  let phase = 2.0 * PI * OPD * 1.0e-9;
  let val = vec3f(5.4856e-13, 4.4201e-13, 5.2481e-13);
  let pos = vec3f(1.6810e+06, 1.7953e+06, 2.2084e+06);
  let vari = vec3f(4.3278e+09, 9.3046e+09, 6.6121e+09);
  var xyz = val * sqrt(2.0 * PI * vari) * cos(pos * phase + shift) * exp(-phase * phase * vari);
  // Extra Gaussian term for the X channel (Belcour 2017 supplemental).
  xyz.x = xyz.x + 9.7470e-14 * sqrt(2.0 * PI * 4.5282e+09)
      * cos(2.2399e+06 * phase + shift.x) * exp(-4.5282e+09 * phase * phase);
  xyz = xyz / 1.0685e-7;
  return iridXyzToRec709(xyz);
}

// Full iridescence Fresnel (Belcour & Barla 2017, §4 Analytic Spectral Integration).
// outsideIOR: medium IOR above the thin film (typically 1.0).
// eta2:        thin-film IOR.
// cosTheta1:  cosine of incidence angle in the outside medium.
// thicknessNm: thin-film thickness in nanometres.
// baseF0:      substrate F0 (the specular colour the iridescence modifies).
// Returns the iridescent reflectance in sRGB.  Negative values are clamped to 0.
fn evalIridescence(
  outsideIOR: f32,
  eta2: f32,
  cosTheta1: f32,
  thicknessNm: f32,
  baseF0: vec3f,
) -> vec3f {
  // Force iridescenceIor → outsideIOR when thickness → 0 (graceful fade-out).
  let iridescenceIor = mix(outsideIOR, eta2, smoothstep(0.0, 0.03, thicknessNm));

  // Snell's law at the air-film interface → cosine in the film.
  let sinTheta2Sq = (outsideIOR / iridescenceIor) * (outsideIOR / iridescenceIor)
      * max(0.0, 1.0 - cosTheta1 * cosTheta1);
  let cosTheta2Sq = 1.0 - sinTheta2Sq;
  // TIR → return full white reflectance (degenerate — the film is opaque).
  if (cosTheta2Sq < 0.0) { return vec3f(1.0); }
  let cosTheta2 = sqrt(cosTheta2Sq);

  // ── First interface (outside ↔ film) ──────────────────────────────────────
  let R0_scalar = iridIorToFresnel0Scalar(iridescenceIor, outsideIOR);
  let R12 = iridSchlickScalar(cosTheta1, R0_scalar);
  let R21 = R12;           // symmetric (non-absorbing dielectric)
  let T121 = 1.0 - R12;
  // Phase shift at first interface: π when going into a denser medium.
  let phi12 = select(0.0, PI, iridescenceIor < outsideIOR);
  let phi21 = PI - phi12;

  // ── Second interface (film ↔ substrate) ───────────────────────────────────
  let baseIOR = iridFresnel0ToIor(clamp(baseF0, vec3f(0.0), vec3f(0.9999)));
  let R1_vec = iridIorToFresnel0Vec(baseIOR, iridescenceIor);
  let R23 = iridSchlickVec(cosTheta2, R1_vec);
  // Phase shift per channel: π when going from film into a less-dense substrate.
  // WGSL: no implicit component-wise ternary — use select() per component.
  var phi23 = vec3f(0.0);
  phi23.x = select(0.0, PI, baseIOR.x < iridescenceIor);
  phi23.y = select(0.0, PI, baseIOR.y < iridescenceIor);
  phi23.z = select(0.0, PI, baseIOR.z < iridescenceIor);

  // ── Compound terms (Belcour 2017 §4, Eq. 5–7) ────────────────────────────
  let OPD = 2.0 * iridescenceIor * thicknessNm * cosTheta2;
  let phi = vec3f(phi21) + phi23;
  let R123 = clamp(R12 * R23, vec3f(1e-5), vec3f(0.9999));
  let r123 = sqrt(R123);
  let Rs = (T121 * T121) * R23 / (vec3f(1.0) - R123);

  // m = 0 (DC) term.
  let C0 = vec3f(R12) + Rs;
  var I = C0;

  // m = 1, 2 (Dirac pairs).
  var Cm = Rs - vec3f(T121);
  for (var m = 1; m <= 2; m = m + 1) {
    Cm = Cm * r123;
    let Sm = 2.0 * iridEvalSensitivity(f32(m) * OPD, f32(m) * phi);
    I = I + Cm * Sm;
  }

  return max(I, vec3f(0.0));
}

// Mix the base specular F0 with the iridescent Fresnel based on the iridescence
// factor.  thicknessNm is sampled linearly between min and max using viewDotN.
// Returns a modified F0 that the caller substitutes into fresnelSchlick.
// Ref: glTF KHR_materials_iridescence §3.
fn iridescenceModifiedF0(
  baseF0: vec3f,
  iridescence: f32,
  iridescenceIor: f32,
  thicknessMin: f32,
  thicknessMax: f32,
  cosTheta: f32,
) -> vec3f {
  if (iridescence < 1e-4) {
    return baseF0; // zero-default: numerically identical to pre-H52 path.
  }
  let thicknessNm = mix(thicknessMin, thicknessMax, clamp(cosTheta, 0.0, 1.0));
  let iridF = evalIridescence(1.0, iridescenceIor, cosTheta, thicknessNm, baseF0);
  return mix(baseF0, iridF, iridescence);
}

// ── Clearcoat (additive GGX specular at fixed IOR 1.5) ────────────────────────
// Ref: glTF KHR_materials_clearcoat (Spec rev 3.0) §3.
//      Burley, "Physically-Based Shading at Disney," SIGGRAPH 2012 §5.4.
// The clearcoat lobe is an additive GGX specular layer at a fixed IOR of 1.5
// (F0 = 0.04), independent of the base metallic/roughness lobe.
// The caller supplies the ALREADY COMPUTED clearcoat roughness (= clearcoatRoughness²
// clamped below 1e-3 as for the base GGX); the clearcoat scalar weights the result.
// The lobe uses the same Cook-Torrance estimator as evaluateBrdf's specular branch.
// evalClearcoatLobe returns the BRDF kernel (WITHOUT nDotL) so it can be
// summed with evaluateBrdf and the caller multiplies by nDotL once, matching
// the convention used throughout the kernel's NEE paths.
fn evalClearcoatLobe(
  clearcoat: f32,
  clearcoatRoughness: f32,
  normal: vec3f,
  wo: vec3f,
  wi: vec3f,
) -> vec3f {
  if (clearcoat < 1e-4) { return vec3f(0.0); } // zero-default short-circuit.
  let nDotL = max(dot(normal, wi), 0.0);
  let nDotV = max(dot(normal, wo), 0.0);
  if (nDotL <= 1e-5 || nDotV <= 1e-5) { return vec3f(0.0); }
  let h = safe_normalize(wi + wo);
  let nDotH = max(dot(normal, h), 0.0);
  let vDotH = max(dot(wo, h), 0.0);
  // Fixed IOR 1.5 → F0 = ((1.5-1)/(1.5+1))² = 0.04.
  let f0cc = vec3f(0.04);
  let f = fresnelSchlick(vDotH, f0cc);
  let alpha = max(clearcoatRoughness * clearcoatRoughness, 1e-3);
  let d = ggxD(nDotH, alpha);
  let g = smithG1(nDotV, clearcoatRoughness) * smithG1(nDotL, clearcoatRoughness);
  // BRDF kernel (no nDotL) — caller multiplies by nDotL together with the base lobe.
  let spec = (d * g) * f / max(4.0 * nDotV * nDotL, 1e-6);
  return clearcoat * spec;
}

// Clearcoat PDF contribution for brdfDirectionalPdf.
// Weighted by clearcoat scalar; the base lobe PDF is extended by this term.
// Returns 0 when clearcoat == 0 (zero-default: identical to pre-H52 PDF).
fn clearcoatPdf(clearcoat: f32, clearcoatRoughness: f32, normal: vec3f, wo: vec3f, wi: vec3f) -> f32 {
  if (clearcoat < 1e-4) { return 0.0; }
  let nDotV = max(dot(normal, wo), 0.0);
  if (nDotV <= 1e-5) { return 0.0; }
  let nDotL = max(dot(normal, wi), 0.0);
  if (nDotL <= 1e-5) { return 0.0; }
  let h = safe_normalize(wi + wo);
  let nDotH = max(dot(normal, h), 0.0);
  let alpha = max(clearcoatRoughness * clearcoatRoughness, 1e-3);
  let d = ggxD(nDotH, alpha);
  let g1Wo = smithG1(nDotV, clearcoatRoughness);
  return (d * g1Wo) / max(4.0 * nDotV, 1e-6);
}

// ── Sheen (Charlie distribution retro-reflective lobe) ────────────────────────
// Ref: glTF KHR_materials_sheen §3; Estevez & Kulla, "Production Friendly
//      Microfacet Sheen BRDF," SIGGRAPH 2017.
// The Charlie NDF: D_c(h; α) = (2 + 1/α) * sin(θ_h)^(1/α) / (2π).
// The sheen lobe is EVALUATION-ONLY (no dedicated sampler — the cosine-hemisphere
// sampler covers it indirectly).  The brdfDirectionalPdf for the sheen term returns
// a cosine-hemisphere approximation; v1 documents this as an accepted bias that
// keeps the sampler simple.  The sheen contribution is typically small enough
// (grazing-only velvet-like highlight) that the variance impact is negligible.
// When sheen == 0 the function returns vec3(0) — zero-default invariant.
fn charlieD(nDotH: f32, alpha: f32) -> f32 {
  let invAlpha = 1.0 / max(alpha, 1e-4);
  let sinThetaH = sqrt(max(0.0, 1.0 - nDotH * nDotH));
  return (2.0 + invAlpha) * pow(sinThetaH, invAlpha) / (2.0 * PI);
}

// Neubelt-Pettineo visibility (Neubelt & Pettineo 2013 approximation for sheen).
fn sheenVisibility(nDotL: f32, nDotV: f32) -> f32 {
  return 1.0 / max(4.0 * (nDotL + nDotV - nDotL * nDotV), 1e-6);
}

// evalSheenLobe returns the BRDF kernel (WITHOUT nDotL) matching the convention
// of evaluateBrdf (caller multiplies by nDotL once for the full NEE contribution).
// Evaluation-only: no dedicated sampler (cosine-hemisphere covers it indirectly).
// The PDF bookkeeping for the sheen lobe uses a cosine-hemisphere approximation
// (see brdfDirectionalPdfFull below).  This is documented as an accepted v1 bias;
// the sheen lobe is a small grazing-angle velvet highlight whose variance impact
// from the mismatched PDF is negligible.
fn evalSheenLobe(
  sheen: f32,
  sheenRoughness: f32,
  sheenColor: vec3f,
  normal: vec3f,
  wo: vec3f,
  wi: vec3f,
) -> vec3f {
  if (sheen < 1e-4) { return vec3f(0.0); } // zero-default short-circuit.
  let nDotL = max(dot(normal, wi), 0.0);
  let nDotV = max(dot(normal, wo), 0.0);
  if (nDotL <= 1e-5 || nDotV <= 1e-5) { return vec3f(0.0); }
  let h = safe_normalize(wi + wo);
  let nDotH = max(dot(normal, h), 0.0);
  let alpha = max(sheenRoughness * sheenRoughness, 1e-3);
  let d = charlieD(nDotH, alpha);
  let vis = sheenVisibility(nDotL, nDotV);
  // BRDF kernel (no nDotL) — caller multiplies by nDotL together with the base lobe.
  return sheen * sheenColor * d * vis;
}

// ── Item 7 — Anisotropic GGX (Heitz 2018 VNDF generalisation) ────────────────
//
// Standard anisotropic formulation with Burley aspect ratio convention:
//   aspect  = sqrt(1 - 0.9 · anisotropy)
//   αx = roughness² / aspect     (stretches ALONG tangent T)
//   αy = roughness² · aspect     (compresses along bitangent B)
// When anisotropy == 0, aspect = 1, αx == αy == roughness² → isotropic.
//
// Anisotropic GGX NDF:
//   D_aniso(h) = 1 / (π · αx · αy · ((hT/αx)² + (hB/αy)² + hN²)²)
//
// Anisotropic Smith G1 (height-correlated form):
//   Λ_aniso(v) = (-1 + sqrt(1 + (vT·αx)² + (vB·αy)²) / vN)) / 2    (λ function)
//   G1_aniso(v) = 1 / (1 + Λ_aniso(v))
//              = 2·vN / (vN + sqrt(vN² + (vT·αx)² + (vB·αy)²))
//
// Anisotropic VNDF sample (Heitz 2018 Algorithm 1, ellipsoidal stretch):
//   Stretch wo by (αx, αy) then apply the same hemisphere projection as the
//   isotropic case; unstretch the result by (αx, αy).
//
// Tangent frame: buildOnb(normal, &t, &b), then rotated by anisotropyRotation
//   t' = cos(rot)·t + sin(rot)·b
//   b' = -sin(rot)·t + cos(rot)·b
// Rotation aligns the anisotropy direction with the author's intent.
//
// Refs: Heitz 2018 JCGT 7(4) — "Sampling the GGX Distribution of Visible
//       Normals"; Burley 2012 Disney BRDF §3 (aspect/roughness parameterisation).
//
// GUARD (zero-anisotropy invariant):
// Every aniso function is gated on anisotropy > 0. Callers check the scalar first
// and fall back to the existing isotropic path for zero-anisotropy materials, so
// pre-existing renders are NUMERICALLY IDENTICAL when anisotropy == 0.

fn ggxDAnis(hT: f32, hB: f32, hN: f32, ax: f32, ay: f32) -> f32 {
  let d = (hT / ax) * (hT / ax) + (hB / ay) * (hB / ay) + hN * hN;
  return 1.0 / max(PI * ax * ay * d * d, 1e-10);
}

fn smithG1Anis(vT: f32, vB: f32, vN: f32, ax: f32, ay: f32) -> f32 {
  let vN2 = max(vN * vN, 1e-10);
  let numer = 2.0 * vN;
  let denom = vN + sqrt(vN2 + (vT * ax) * (vT * ax) + (vB * ay) * (vB * ay));
  return numer / max(denom, 1e-6);
}

// Anisotropic VNDF sample — all vectors in surface TANGENT SPACE (N = +Z, T = +X, B = +Y).
// Input wo in tangent space; ax, ay = per-axis alpha. Returns the sampled half-vector h.
// Follows Heitz 2018 Algorithm 1, §3 (ellipsoidal stretch + unit-sphere projection).
fn sampleGgxVndfAnisTangent(wo: vec3f, ax: f32, ay: f32, rng: ptr<function, u32>) -> vec3f {
  // Step 1: stretch wo into the unit-roughness configuration.
  let Vh = safe_normalize(vec3f(ax * wo.x, ay * wo.y, wo.z));
  // Step 2: ONB around Vh (Frisvad-style, same as isotropic).
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
  t2 = (1.0 - s) * sqrt(max(0.0, 1.0 - t1 * t1)) + s * t2;
  // Step 4: reproject onto hemisphere, ANISOTROPICALLY unstretch.
  let Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0, 1.0 - t1 * t1 - t2 * t2)) * Vh;
  return safe_normalize(vec3f(ax * Nh.x, ay * Nh.y, max(1e-6, Nh.z)));
}

// Anisotropic glossy reflection sampler in WORLD SPACE.
// Tangent frame t, b come from the caller (buildOnb then rotated by anisotropyRotation).
// Returns a BsdfSample consistent with the anisotropic eval+pdf triple.
fn glossyReflectionSampleAnisotropic(
  rng: ptr<function, u32>,
  wo: vec3f,
  n: vec3f,
  t: vec3f,
  b: vec3f,
  roughness: f32,
  anisotropy: f32,
) -> BsdfSample {
  let alpha = max(roughness * roughness, 0.001);
  let aspect = sqrt(max(1.0 - 0.9 * anisotropy, 1e-4));
  let ax = max(alpha / aspect, 1e-4);
  let ay = max(alpha * aspect, 1e-4);
  // Transform wo into tangent space (T=+X, B=+Y, N=+Z).
  let woLocal = vec3f(dot(wo, t), dot(wo, b), dot(wo, n));
  let hLocal  = sampleGgxVndfAnisTangent(woLocal, ax, ay, rng);
  let hWorld  = safe_normalize(hLocal.x * t + hLocal.y * b + hLocal.z * n);
  let wi      = safe_normalize(reflect(-wo, hWorld));

  var result: BsdfSample;
  result.wi = wi;
  let nDotV = max(dot(n, wo), 1e-6);
  let nDotL = max(dot(n, wi), 0.0);
  if (nDotL <= 1e-5) {
    result.pdf = 0.0;
    result.value = vec3f(0.0);
  } else {
    // Anisotropic VNDF PDF:  p(wi) = D_aniso(h) · G1_aniso(wo) / (4 · NdotV)
    let hT = dot(hWorld, t);
    let hB = dot(hWorld, b);
    let hN = max(dot(hWorld, n), 0.0);
    let woT = dot(wo, t);
    let woB = dot(wo, b);
    let woN = max(dot(wo, n), 1e-6);
    let d   = ggxDAnis(hT, hB, hN, ax, ay);
    let g1  = smithG1Anis(woT, woB, woN, ax, ay);
    result.pdf = (d * g1) / max(4.0 * nDotV, 1e-6);
    // Anisotropic BRDF kernel (no nDotL): D·G1(wo)·G1(wi)·F / (4·NdotV·NdotL)
    // F is NOT included here (caller applies Fresnel at throughput level, same as
    // the isotropic path in glossyReflectionSample). We return D·G1(wo)·G1(wi) /
    // (4·NdotV·NdotL) as 'value' — identical role to the isotropic (d*g)/(4*NdotV*NdotL).
    let wiT = dot(wi, t);
    let wiB = dot(wi, b);
    let wiN = max(dot(wi, n), 1e-6);
    let g1i = smithG1Anis(wiT, wiB, wiN, ax, ay);
    result.value = vec3f((d * g1 * g1i) / max(4.0 * nDotV * nDotL, 1e-6));
  }
  return result;
}

// Evaluate anisotropic GGX specular BRDF kernel (WITHOUT nDotL).
// Returns the anisotropic Cook-Torrance specular BRDF kernel D·G·F/(4·NdotV·NdotL).
// Requires the pre-rotated tangent frame (t, b) aligned to the anisotropy direction.
fn evalBrdfSpecAnisotropic(
  f0: vec3f,
  roughness: f32,
  anisotropy: f32,
  normal: vec3f,
  t: vec3f,
  b: vec3f,
  wo: vec3f,
  wi: vec3f,
) -> vec3f {
  let nDotV = max(dot(normal, wo), 1e-6);
  let nDotL = max(dot(normal, wi), 0.0);
  if (nDotL <= 1e-5 || nDotV <= 1e-5) { return vec3f(0.0); }
  let h = safe_normalize(wo + wi);
  let vDotH = max(dot(wo, h), 1e-6);
  let f = fresnelSchlick(vDotH, f0);
  let alpha = max(roughness * roughness, 1e-3);
  let aspect = sqrt(max(1.0 - 0.9 * anisotropy, 1e-4));
  let ax = max(alpha / aspect, 1e-4);
  let ay = max(alpha * aspect, 1e-4);
  let hT = dot(h, t);
  let hB = dot(h, b);
  let hN = max(dot(h, normal), 0.0);
  let woT = dot(wo, t);
  let woB = dot(wo, b);
  let woN = max(dot(wo, normal), 1e-6);
  let wiT = dot(wi, t);
  let wiB = dot(wi, b);
  let wiN = max(dot(wi, normal), 1e-6);
  let d = ggxDAnis(hT, hB, hN, ax, ay);
  let g = smithG1Anis(woT, woB, woN, ax, ay) * smithG1Anis(wiT, wiB, wiN, ax, ay);
  return (d * g) * f / max(4.0 * nDotV * nDotL, 1e-6);
}

// Anisotropic VNDF reflection PDF (for brdfDirectionalPdf).
// D_aniso(h) · G1_aniso(wo) / (4 · NdotV), matching glossyReflectionSampleAnisotropic.
fn brdfAnisotropicSpecPdf(
  roughness: f32,
  anisotropy: f32,
  normal: vec3f,
  t: vec3f,
  b: vec3f,
  wo: vec3f,
  wi: vec3f,
) -> f32 {
  let nDotV = max(dot(normal, wo), 1e-6);
  let nDotL = max(dot(normal, wi), 0.0);
  if (nDotL <= 1e-5) { return 0.0; }
  let h = safe_normalize(wo + wi);
  let alpha = max(roughness * roughness, 1e-3);
  let aspect = sqrt(max(1.0 - 0.9 * anisotropy, 1e-4));
  let ax = max(alpha / aspect, 1e-4);
  let ay = max(alpha * aspect, 1e-4);
  let hT = dot(h, t);
  let hB = dot(h, b);
  let hN = max(dot(h, normal), 0.0);
  let woT = dot(wo, t);
  let woB = dot(wo, b);
  let woN = max(dot(wo, normal), 1e-6);
  let d   = ggxDAnis(hT, hB, hN, ax, ay);
  let g1  = smithG1Anis(woT, woB, woN, ax, ay);
  return (d * g1) / max(4.0 * nDotV, 1e-6);
}

// ── H52 extended BRDF evaluation (base + clearcoat + sheen + iridescence) ─────
// evaluateBrdfFull: adds the three Disney extension lobes to the base Cook-Torrance
// BRDF.  Returns the BRDF kernel (WITHOUT nDotL); callers multiply by nDotL once.
// When all extension scalars are 0 the result is identical to evaluateBrdf.
//
// iridescence modifies the base specular F0 BEFORE the Cook-Torrance evaluation
// (it is NOT an additive lobe — it replaces the F0 that governs diffuse/specular
// partition and the specular highlight colour).  Clearcoat and sheen ARE additive.
//
// Refs: glTF KHR_materials_clearcoat, KHR_materials_sheen, KHR_materials_iridescence;
//       Belcour & Barla 2017 (iridescence); Estevez & Kulla 2017 (sheen).
fn evaluateBrdfFull(
  baseColor: vec3f,
  roughness: f32,
  metallic: f32,
  normal: vec3f,
  wo: vec3f,
  wi: vec3f,
  clearcoat: f32,
  clearcoatRoughness: f32,
  sheen: f32,
  sheenRoughness: f32,
  sheenColor: vec3f,
  iridescence: f32,
  iridescenceIor: f32,
  iridescenceThicknessMin: f32,
  iridescenceThicknessMax: f32,
  anisotropy: f32,
  anisotropyRotation: f32,
) -> vec3f {
  let nDotL = max(dot(normal, wi), 0.0);
  let nDotV = max(dot(normal, wo), 0.0);
  if (nDotL <= 1e-5 || nDotV <= 1e-5) { return vec3f(0.0); }
  let h = safe_normalize(wi + wo);
  let nDotH = max(dot(normal, h), 0.0);
  let vDotH = max(dot(wo, h), 0.0);

  // Iridescence-modified F0 (modifies diffuse/specular partition + specular colour).
  let f0base = mix(vec3f(0.04), baseColor, metallic);
  let f0 = iridescenceModifiedF0(
    f0base, iridescence, iridescenceIor,
    iridescenceThicknessMin, iridescenceThicknessMax, vDotH,
  );

  let f = fresnelSchlick(vDotH, f0);
  let kd = (vec3f(1.0) - f) * (1.0 - metallic);
  let diff = kd * baseColor * INV_PI;

  // Item 7 — anisotropic GGX specular lobe.
  // When anisotropy == 0 the guard falls through to the isotropic Cook-Torrance path
  // (byte-identical render for zero-anisotropy materials).
  var spec: vec3f;
  var ms: vec3f;
  if (anisotropy > 1e-4) {
    // Build tangent frame and rotate by anisotropyRotation.
    var tanT: vec3f;
    var tanB: vec3f;
    buildOnb(normal, &tanT, &tanB);
    let c = cos(anisotropyRotation);
    let s = sin(anisotropyRotation);
    let anisoT = c * tanT + s * tanB;
    let anisoB = -s * tanT + c * tanB;
    spec = evalBrdfSpecAnisotropic(f0, roughness, anisotropy, normal, anisoT, anisoB, wo, wi);
    // B9 — Kulla-Conty for anisotropic path: use the isotropic E LUT as an
    // approximation (the LUT is rotationally symmetric; anisotropy changes the
    // per-direction albedo distribution but the average E(μ) stays similar).
    ms = ggxMultiscatterLobe(f0, roughness, nDotV, nDotL);
  } else {
    let alpha = max(roughness * roughness, 1e-3);
    let d = ggxD(nDotH, alpha);
    let g = smithG1(nDotV, roughness) * smithG1(nDotL, roughness);
    spec = (d * g) * f / max(4.0 * nDotV * nDotL, 1e-6);
    // B9 — Kulla-Conty multiscatter energy compensation.
    ms = ggxMultiscatterLobe(f0, roughness, nDotV, nDotL);
  }
  let base = diff + spec + ms;

  // Additive extension lobes (each returns BRDF kernel, no nDotL factor).
  let cc = evalClearcoatLobe(clearcoat, clearcoatRoughness, normal, wo, wi);
  let sh = evalSheenLobe(sheen, sheenRoughness, sheenColor, normal, wo, wi);
  return base + cc + sh;
}

// brdfDirectionalPdfFull: the pdf for the full lobe mixture used in MIS.
// The base pdf comes from brdfDirectionalPdf; the clearcoat and sheen terms add
// their (weighted) pdfs.  The sheen PDF uses a cosine-hemisphere approximation
// (v1 documented bias — see evalSheenLobe).
// When all extension scalars are 0 the result is identical to brdfDirectionalPdf.
fn brdfDirectionalPdfFull(
  baseColor: vec3f,
  roughness: f32,
  metallic: f32,
  transmission: f32,
  ior: f32,
  normal: vec3f,
  wo: vec3f,
  wi: vec3f,
  clearcoat: f32,
  clearcoatRoughness: f32,
  sheen: f32,
  sheenRoughness: f32,
  iridescence: f32,
  iridescenceIor: f32,
  iridescenceThicknessMin: f32,
  iridescenceThicknessMax: f32,
  anisotropy: f32,
  anisotropyRotation: f32,
) -> f32 {
  // Item 7 — when anisotropic, replace the isotropic specular PDF with the
  // anisotropic VNDF PDF. The diffuse/trans lobe probabilities stay identical.
  var basePdf: f32;
  if (anisotropy > 1e-4) {
    // Compute lobe probabilities (same as brdfDirectionalPdf).
    let wiDotN = dot(normal, wi);
    let woDotN = dot(normal, wo);
    let nDotV = max(woDotN, 0.0);
    if (nDotV <= 1e-5) { return 0.0; }
    let h = safe_normalize(wi + wo);
    let vDotH = max(dot(wo, h), 1e-6);
    let f0 = mix(vec3f(0.04), baseColor, metallic);
    let fresnel = fresnelSchlick(vDotH, f0);
    let baseSpecProb = clamp(mix(0.04, 0.96, max(luminance(fresnel), metallic)), 0.04, 0.96);
    let baseTransProb = clamp(transmission * (1.0 - metallic), 0.0, 0.95);
    let baseDiffProb = max(0.0, (1.0 - metallic) * (1.0 - transmission));
    let sumProb = max(baseSpecProb + baseTransProb + baseDiffProb, 1e-4);
    let specProb = baseSpecProb / sumProb;
    let diffProb = baseDiffProb / sumProb;
    let sameHemisphere = wiDotN * woDotN > 0.0;
    if (!sameHemisphere) { return 0.0; }
    let nDotL = max(wiDotN, 0.0);
    if (nDotL <= 1e-5) { return 0.0; }
    // Build rotated tangent frame.
    var tanT: vec3f;
    var tanB: vec3f;
    buildOnb(normal, &tanT, &tanB);
    let c = cos(anisotropyRotation);
    let s = sin(anisotropyRotation);
    let anisoT = c * tanT + s * tanB;
    let anisoB = -s * tanT + c * tanB;
    let pdfSpec = brdfAnisotropicSpecPdf(roughness, anisotropy, normal, anisoT, anisoB, wo, wi);
    let pdfDiff = nDotL * INV_PI;
    basePdf = diffProb * pdfDiff + specProb * pdfSpec;
  } else {
    basePdf = brdfDirectionalPdf(baseColor, roughness, metallic, transmission, ior, normal, wo, wi);
  }
  // Clearcoat PDF: VNDF GGX at clearcoat roughness, weighted by clearcoat scalar.
  let ccPdf = clearcoat * clearcoatPdf(clearcoat, clearcoatRoughness, normal, wo, wi);
  // Sheen PDF approximation: cosine-hemisphere (v1 accepted bias — see evalSheenLobe).
  let nDotL = max(dot(normal, wi), 0.0);
  let sheenPdf = sheen * nDotL * INV_PI;
  // Iridescence does NOT add a new sampling lobe (it modifies F0 of the existing
  // specular lobe, which the base brdfDirectionalPdf already accounts for).
  // These parameters are present for API symmetry with evaluateBrdfFull.
  // WGSL does not penalise unused function parameters.
  // Total pdf: sum of all lobe pdfs.
  let total = basePdf + ccPdf + sheenPdf;
  return total;
}

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
  // B9 — Kulla-Conty multiscatter energy compensation (see evaluateBrdfFull).
  let ms = ggxMultiscatterLobe(f0, roughness, nDotV, nDotL);
  return diff + spec + ms;
}

fn brdfDirectionalPdf(
  baseColor: vec3f,
  roughness: f32,
  metallic: f32,
  transmission: f32,
  ior: f32,
  normal: vec3f,
  wo: vec3f,
  wi: vec3f,
) -> f32 {
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
    // Delta-refraction lobe (D4): the sampler draws a deterministic transmitted
    // direction (Snell's law), so the refraction pdf is a Dirac delta — it
    // contributes zero probability density for any specific direction query.
    // Returning 0 here is unbiased: the NEE weight at connection sites resolves
    // to the full light-pdf denominator (MIS collapses to pure NEE weighting on
    // delta lobes). All brdfDirectionalPdf call sites guard against pdf <= 1e-6
    // so division by zero never occurs.
    // Decision H13/D4 (h-remediation-plan §3): prior finite cosine*eta^2 pdf
    // did not match the deterministic sampler in sampleNextBounceDirection.
    return 0.0;
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
 *   pdf   — VNDF reflection PDF D(h) * G1(wo) / (4 * NdotV), matching
 *           brdfDirectionalPdf's specular branch for MIS consistency
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
  // pdf and value are populated for MIS consumers; current sampleNextBounceDirection
  // callers recompute g1Wi independently.
  let nDotV = max(dot(n, wo), 1e-6);
  let nDotL = max(dot(n, wi), 0.0);
  let nDotH = max(dot(n, hWorld), 0.0);
  let vDotH = max(dot(wo, hWorld), 1e-6);
  let d = ggxD(nDotH, alpha);
  if (nDotL <= 1e-5) {
    result.pdf = 0.0;
    result.value = vec3f(0.0);
  } else {
    let g1Wo = smithG1(nDotV, roughness);
    result.pdf = (d * g1Wo) / max(4.0 * nDotV, 1e-6);
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
  // WS4 — medium-crossing events for the volumetric random walk. Set true on
  // the dielectric REFRACTION branch (the only branch that crosses the
  // surface): entered when refracting into a translucent front face, exited
  // when refracting out through a back face. Reflection / diffuse / specular
  // branches stay on the same side, so both remain false there.
  enteredMedium: bool,
  exitedMedium: bool,
}

// D9.1 — shared anisotropy axis helper (deduplicates two identical blocks in
// sampleNextBounceDirection). Returns vec2f(ax, ay).
fn computeAnisotropicAxes(alpha: f32, anisotropy: f32) -> vec2f {
  let aspect = sqrt(max(1.0 - 0.9 * anisotropy, 1e-4));
  let ax = max(alpha / aspect, 1e-4);
  let ay = max(alpha * aspect, 1e-4);
  return vec2f(ax, ay);
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
  isTranslucent: bool,
  anisotropy: f32,
  anisotropyRotation: f32,
) -> BounceSample {
  // Build surface-tangent ONB once; shared by both glossy-reflect call sites.
  // Item 7 — if anisotropic, rotate the tangent frame by anisotropyRotation so
  // both tangent and bitangent align with the authored anisotropy direction.
  var tanT: vec3f;
  var tanB: vec3f;
  buildOnb(normal, &tanT, &tanB);
  if (anisotropy > 1e-4) {
    let c = cos(anisotropyRotation);
    let s = sin(anisotropyRotation);
    let newT = c * tanT + s * tanB;
    let newB = -s * tanT + c * tanB;
    tanT = newT;
    tanB = newB;
  }

  var result: BounceSample;
  result.sampledDir = vec3f(0.0);
  result.sampleAllowsAreaMis = false;
  result.enteredMedium = false;
  result.exitedMedium = false;

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
      // Item 7 — use anisotropic sampler when anisotropy > 0.
      var bs: BsdfSample;
      if (anisotropy > 1e-4) {
        bs = glossyReflectionSampleAnisotropic(rng, wo, normal, tanT, tanB, roughness, anisotropy);
      } else {
        bs = glossyReflectionSample(rng, wo, normal, tanT, tanB, roughness);
      }
      result.sampledDir = bs.wi;
      result.newRayDir = bs.wi;
      result.sampleAllowsAreaMis = true;
      // MC estimator for VNDF sampling of the GGX BRDF (Heitz 2018):
      //   f·cosθ / p_VNDF = [D·G·F / (4·NdotV·NdotL)] · NdotL
      //                    / [D·G1(wo) / (4·NdotV)]
      //                    = F · G1(wi)
      // MC estimator: F · G1(wi) / R (same derivation for iso and aniso paths;
      // only the G1 function differs). nDotL = dot(n, wi).
      let nDotL = max(dot(normal, result.sampledDir), 0.0);
      var g1Wi: f32;
      if (anisotropy > 1e-4) {
        let axes = computeAnisotropicAxes(max(roughness * roughness, 1e-3), anisotropy);
        let wiT = dot(result.sampledDir, tanT);
        let wiB = dot(result.sampledDir, tanB);
        g1Wi = smithG1Anis(wiT, wiB, max(nDotL, 1e-6), axes.x, axes.y);
      } else {
        g1Wi = smithG1(nDotL, roughness);
      }
      // B9 — multiscatter energy boost on the sampled specular reflection. The
      // VNDF sampler covers single-scatter only; scale by the Kulla-Conty factor
      // 1 + F_avg·(1−E_ss)/E_ss so the sampled estimator recovers the lost
      // multi-bounce energy (1 at low roughness → unchanged smooth surfaces).
      let nDotVcc = max(dot(normal, wo), 0.0);
      let msBoost = ggxMultiscatterBoost(fresnel, roughness, nDotVcc);
      result.throughputMul = fresnel * g1Wi * msBoost / max(R, 1e-4);
    } else {
      // Fresnel-weighted refraction branch — the only branch that crosses the
      // surface, so it is where the volumetric random walk enters / exits the
      // medium (WS4). A front-face refraction of a translucent dielectric
      // enters the medium; a back-face refraction exits it.
      let eta = select(ior, 1.0 / ior, frontFace);
      let refr = refract(incomingDir, normal, eta);
      let outDir = safe_normalize(refr);
      let offsetN = select(-normal, normal, dot(outDir, normal) > 0.0);
      result.newRayOrigin = hitPos + offsetN * 1e-3;
      result.sampledDir = outDir;
      result.newRayDir = outDir;
      // B10 — physical refraction transmittance. The energy partition is already
      // Fresnel-consistent: the refraction branch carries probability (1 − R) and
      // the throughput divides by it, so a clear (white) dielectric transmits 1·
      // (1 − R)/(1 − R) = 1 — no arbitrary tint. The SURFACE transmittance colour
      // is the material's baseColor (the dielectric's interface transmission
      // colour, e.g. tinted glass) — NOT the old phenomenological
      // mix(vec3(1), baseColor, 0.15) which faded any glass 85 % toward white and
      // hid its colour. Bulk Beer-Lambert μ(λ) absorption (the physically-correct
      // path-length-dependent colouring of a participating medium) is applied
      // SEPARATELY in the kernel's transmissive block / volumetric walk; this
      // factor is the thin-interface transmittance only. In spectral mode
      // baseColor is the scalar Jakob-Hanika reflectance S(λ), so the surface
      // transmittance is genuinely wavelength-resolved here too.
      //
      // η² radiance scaling: the symmetric BSDF (light/eye) requires the radiance
      // to scale by (η_t/η_i)² across a refraction (PBR4e §9.5.2, eq. 9.13 — the
      // "non-symmetry due to refraction" factor). We DELIBERATELY OMIT it: this is
      // a UNIDIRECTIONAL eye-path tracer where the camera measures radiance and
      // the radiance-scaling factor cancels for a closed light↔eye round trip
      // through equal media (entering then exiting the same glass), which is the
      // overwhelmingly common case (a glass object in air). Including it on only
      // one crossing would over/under-brighten enclosed glass; the BDPT light
      // subpath (the bidirectional consumer that WOULD need it) has its own
      // medium accounting. Ref: PBR4e §9.5.2; Veach 1997 §5 (importance vs.
      // radiance transport asymmetry). This is the same decision the pt-webgl2
      // and walkaround dielectric BSDFs make.
      result.throughputMul = baseColor * thinFilmTransmitTint / max(1.0 - R, 1e-4);
      result.enteredMedium = isTranslucent && frontFace;
      result.exitedMedium = isTranslucent && !frontFace;
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
    // Item 7 — use anisotropic sampler when anisotropy > 0; isotropic otherwise.
    // The tangent frame (tanT, tanB) is already rotated by anisotropyRotation above.
    let wo = -incomingDir;
    result.newRayOrigin = hitPos + normal * 1e-3;
    var bs2: BsdfSample;
    if (anisotropy > 1e-4) {
      bs2 = glossyReflectionSampleAnisotropic(rng, wo, normal, tanT, tanB, roughness, anisotropy);
    } else {
      bs2 = glossyReflectionSample(rng, wo, normal, tanT, tanB, roughness);
    }
    result.sampledDir = bs2.wi;
    result.newRayDir = bs2.wi;
    result.sampleAllowsAreaMis = true;
    let nDotL2 = max(dot(normal, result.sampledDir), 0.0);
    var g1Wi2: f32;
    if (anisotropy > 1e-4) {
      let axes2 = computeAnisotropicAxes(max(roughness * roughness, 1e-3), anisotropy);
      g1Wi2 = smithG1Anis(dot(result.sampledDir, tanT), dot(result.sampledDir, tanB), max(nDotL2, 1e-6), axes2.x, axes2.y);
    } else {
      g1Wi2 = smithG1(nDotL2, roughness);
    }
    // B9 — multiscatter energy boost on the sampled specular reflection (see the
    // dielectric-reflection branch above).
    let nDotVcc = max(dot(normal, wo), 0.0);
    let msBoost = ggxMultiscatterBoost(fresnel, roughness, nDotVcc);
    result.throughputMul = fresnel * g1Wi2 * msBoost / max(specProb, 1e-4);
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

/**
 * Henyey-Greenstein phase function + importance sampler (WS4 volumetric SSS).
 *
 * Emitted as a SEPARATE WGSL chunk so the kernel composer can include it ONLY
 * when the volumetric walk is compiled in (BDPT off). Keeping it out of the
 * always-included BSDF module is what makes the BDPT-on shader free of any SSS
 * symbols (the structural compile-time gate) — no dead phase-function code.
 *
 * p(cosθ; g) = 1/(4π) · (1 - g²) / (1 + g² - 2·g·cosθ)^{3/2}
 * integrates to 1 over the sphere; g ∈ (-1,1) is the mean cosine (forward
 * scatter for g>0, back-scatter for g<0). The sampler draws cosθ ∝ p with the
 * closed-form inversion, azimuth uniform, so f/pdf = 1 and the phase-sampled
 * estimator is unbiased without an explicit weight.
 *
 * Ref: Henyey, L. G., & Greenstein, J. L. "Diffuse radiation in the galaxy."
 *      Astrophys. J. 93:70-83 (1941).
 *      Pharr, Jakob, Humphreys. PBR 4th ed. §11.3 "Phase Functions" — the HG
 *      phase function and its sampling routine (eq. 11.7).
 */
export const PT_WEBGPU_PATH_TRACE_HG_PHASE_WGSL = /* wgsl */ `
const INV_4PI = 0.07957747154594767;

fn hgPhase(cosTheta: f32, g: f32) -> f32 {
  let denom = 1.0 + g * g - 2.0 * g * cosTheta;
  return INV_4PI * (1.0 - g * g) / max(pow(denom, 1.5), 1e-9);
}

// Sample a world-space direction from the HG phase function around the
// incoming-photon travel direction wIn (the direction the ray is travelling).
// Returns the scattered direction; the implied pdf equals hgPhase(cosθ, g).
fn sampleHenyeyGreenstein(rng: ptr<function, u32>, wIn: vec3f, g: f32) -> vec3f {
  let u1 = rand_f32(rng);
  let u2 = rand_f32(rng);
  var cosTheta: f32;
  if (abs(g) < 1e-3) {
    cosTheta = 1.0 - 2.0 * u1; // isotropic
  } else {
    let sq = (1.0 - g * g) / (1.0 + g - 2.0 * g * u1);
    // PBRT closed form returns cosθ vs wo = -travel (mean -g); negate so it is
    // measured against the travel direction wIn (mean +g, forward scatter for
    // g>0) — consistent with hgPhase(dot(travel, ·), g) used in the kernel NEE.
    cosTheta = (1.0 + g * g - sq * sq) / (2.0 * g);
  }
  cosTheta = clamp(cosTheta, -1.0, 1.0);
  let sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
  let phi = 2.0 * PI * u2;
  // Build an ONB around wIn and place the sampled (θ measured FROM wIn).
  var t: vec3f;
  var b: vec3f;
  buildOnb(wIn, &t, &b);
  return safe_normalize(sinTheta * cos(phi) * t + sinTheta * sin(phi) * b + cosTheta * wIn);
}
`;
