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
import {
  PT_WEBGPU_MICROFACET_ALPHA_FLOOR,
  ROUGH_DIELECTRIC_SMOOTH_THRESHOLD,
} from '../../math/roughDielectric.js';

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
  if (iridescence <= 0.0) {
    return baseF0; // zero-default: numerically identical to pre-H52 path.
  }
  let thicknessNm = mix(thicknessMin, thicknessMax, clamp(cosTheta, 0.0, 1.0));
  let iridF = evalIridescence(1.0, iridescenceIor, cosTheta, thicknessNm, baseF0);
  return mix(baseF0, iridF, iridescence);
}

// SPEC-01 — KHR_materials_specular dielectric F0 composition.
// specularColor/specularIntensity scale the dielectric 4% baseline; metallic
// surfaces still use baseColor as F0. Defaults ([1,1,1], 1) reproduce the old path.
fn materialSpecularF0(
  baseColor: vec3f,
  metallic: f32,
  specularColor: vec3f,
  specularIntensity: f32,
) -> vec3f {
  let dielectricF0 = clamp(
    vec3f(0.04) * clamp(specularColor, vec3f(0.0), vec3f(1.0)) * clamp(specularIntensity, 0.0, 1.0),
    vec3f(0.0),
    vec3f(1.0),
  );
  return mix(dielectricF0, baseColor, metallic);
}

// Rough dielectric transport uses exact IOR Fresnel for the no-extension
// baseline, while KHR_materials_specular and KHR_materials_iridescence author
// the dielectric's coloured F0. Apply those authored controls as a ratio to the
// legacy 4%-F0 Schlick curve so the default values preserve frDielectric
// bit-for-bit for every IOR, including non-1.5 interfaces. The same function is
// consumed by finite evaluation, directional PDFs, and source sampling.
fn materialDielectricFresnel(
  cosTheta: f32,
  etaTOverI: f32,
  iridescence: f32,
  iridescenceIor: f32,
  iridescenceThicknessMin: f32,
  iridescenceThicknessMax: f32,
  specularColor: vec3f,
  specularIntensity: f32,
) -> vec3f {
  let exactF = frDielectric(abs(cosTheta), max(etaTOverI, 1e-4));
  // Total internal reflection is achromatic unit reflectance; material F0
  // controls cannot turn a physically unavailable transmission event back on.
  if (exactF >= 1.0) { return vec3f(1.0); }

  let authoredF0 = iridescenceModifiedF0(
    materialSpecularF0(
      vec3f(1.0), 0.0, specularColor, specularIntensity,
    ),
    iridescence, iridescenceIor,
    iridescenceThicknessMin, iridescenceThicknessMax, abs(cosTheta),
  );
  let baselineSchlick = fresnelSchlick(abs(cosTheta), vec3f(0.04));
  let authoredSchlick = fresnelSchlick(abs(cosTheta), authoredF0);
  return clamp(
    vec3f(exactF) * authoredSchlick / max(baselineSchlick, vec3f(1e-6)),
    vec3f(0.0),
    vec3f(1.0),
  );
}

// ── Clearcoat (additive GGX specular at fixed IOR 1.5) ────────────────────────
// Ref: glTF KHR_materials_clearcoat (Spec rev 3.0) §3.
//      Burley, "Physically-Based Shading at Disney," SIGGRAPH 2012 §5.4.
// The clearcoat lobe is an additive GGX specular layer at a fixed IOR of 1.5
// (F0 = 0.04), independent of the base metallic/roughness lobe.
// The caller supplies the ALREADY COMPUTED clearcoat roughness (= clearcoatRoughness²
// evaluated with the shared finite-alpha numerical floor used by base GGX); the
// clearcoat scalar weights the result.
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
  if (clearcoat <= 0.0) { return vec3f(0.0); } // zero-default short-circuit.
  let nDotL = max(dot(normal, wi), 0.0);
  let nDotV = max(dot(normal, wo), 0.0);
  if (nDotL <= 1e-5 || nDotV <= 1e-5) { return vec3f(0.0); }
  let h = safe_normalize(wi + wo);
  let nDotH = max(dot(normal, h), 0.0);
  let vDotH = max(dot(wo, h), 0.0);
  // Fixed IOR 1.5 → F0 = ((1.5-1)/(1.5+1))² = 0.04.
  let f0cc = vec3f(0.04);
  let f = fresnelSchlick(vDotH, f0cc);
  let alpha = max(clearcoatRoughness * clearcoatRoughness, ${PT_WEBGPU_MICROFACET_ALPHA_FLOOR});
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
  if (clearcoat <= 0.0) { return 0.0; }
  let nDotV = max(dot(normal, wo), 0.0);
  if (nDotV <= 1e-5) { return 0.0; }
  let nDotL = max(dot(normal, wi), 0.0);
  if (nDotL <= 1e-5) { return 0.0; }
  let h = safe_normalize(wi + wo);
  let nDotH = max(dot(normal, h), 0.0);
  let alpha = max(clearcoatRoughness * clearcoatRoughness, ${PT_WEBGPU_MICROFACET_ALPHA_FLOOR});
  let d = ggxD(nDotH, alpha);
  let g1Wo = smithG1(nDotV, clearcoatRoughness);
  return (d * g1Wo) / max(4.0 * nDotV, 1e-6);
}

// ── Sheen (Charlie distribution retro-reflective lobe) ────────────────────────
// Ref: glTF KHR_materials_sheen §3; Estevez & Kulla, "Production Friendly
//      Microfacet Sheen BRDF," SIGGRAPH 2017.
// The Charlie NDF: D_c(h; α) = (2 + 1/α) * sin(θ_h)^(1/α) / (2π).
// The sampled path uses a matching Charlie half-vector sampler below:
//   sin(theta_h) = u^(1 / (1/alpha + 2)),
// which is the inverse CDF for p_h(h)=D_c(h)*cos(theta_h).
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
fn evalSheenLobe(
  sheen: f32,
  sheenRoughness: f32,
  sheenColor: vec3f,
  normal: vec3f,
  wo: vec3f,
  wi: vec3f,
) -> vec3f {
  if (sheen <= 0.0) { return vec3f(0.0); } // zero-default short-circuit.
  let nDotL = max(dot(normal, wi), 0.0);
  let nDotV = max(dot(normal, wo), 0.0);
  if (nDotL <= 1e-5 || nDotV <= 1e-5) { return vec3f(0.0); }
  let h = safe_normalize(wi + wo);
  let nDotH = max(dot(normal, h), 0.0);
  let alpha = max(sheenRoughness * sheenRoughness, ${PT_WEBGPU_MICROFACET_ALPHA_FLOOR});
  let d = charlieD(nDotH, alpha);
  let vis = sheenVisibility(nDotL, nDotV);
  // BRDF kernel (no nDotL) — caller multiplies by nDotL together with the base lobe.
  return sheen * sheenColor * d * vis;
}

fn charlieSheenPdf(
  sheen: f32,
  sheenRoughness: f32,
  normal: vec3f,
  wo: vec3f,
  wi: vec3f,
) -> f32 {
  if (sheen <= 0.0) { return 0.0; }
  let nDotL = max(dot(normal, wi), 0.0);
  let nDotV = max(dot(normal, wo), 0.0);
  if (nDotL <= 1e-5 || nDotV <= 1e-5) { return 0.0; }
  let h = safe_normalize(wi + wo);
  let nDotH = max(dot(normal, h), 0.0);
  let vDotH = max(dot(wo, h), 1e-6);
  let alpha = max(sheenRoughness * sheenRoughness, ${PT_WEBGPU_MICROFACET_ALPHA_FLOOR});
  return (charlieD(nDotH, alpha) * nDotH) / max(4.0 * vDotH, 1e-6);
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
fn sampleGgxVndfAnisTangent(wo: vec3f, ax: f32, ay: f32, rng: ptr<function, PtRngState>) -> vec3f {
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
  rng: ptr<function, PtRngState>,
  wo: vec3f,
  n: vec3f,
  t: vec3f,
  b: vec3f,
  roughness: f32,
  anisotropy: f32,
) -> BsdfSample {
  let alpha = max(roughness * roughness, ${PT_WEBGPU_MICROFACET_ALPHA_FLOOR});
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
  let alpha = max(roughness * roughness, ${PT_WEBGPU_MICROFACET_ALPHA_FLOOR});
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
  let alpha = max(roughness * roughness, ${PT_WEBGPU_MICROFACET_ALPHA_FLOOR});
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

fn anisotropicProjectedRoughness(
  dir: vec3f,
  t: vec3f,
  b: vec3f,
  roughness: f32,
  anisotropy: f32,
) -> f32 {
  let alpha = max(roughness * roughness, ${PT_WEBGPU_MICROFACET_ALPHA_FLOOR});
  let axes = computeAnisotropicAxes(alpha, anisotropy);
  let dT = dot(dir, t);
  let dB = dot(dir, b);
  let tangentLen2 = dT * dT + dB * dB;
  let projectionBlend = clamp(0.15 * anisotropy, 0.0, 0.15);
  if (tangentLen2 <= 1e-6) {
    let projectedNormal = sqrt(clamp(0.5 * (axes.x + axes.y), 1e-4, 1.0));
    return mix(roughness, projectedNormal, projectionBlend);
  }
  let alphaEff = sqrt(((dT * axes.x) * (dT * axes.x) + (dB * axes.y) * (dB * axes.y)) / tangentLen2);
  let projected = sqrt(clamp(alphaEff, 1e-4, 1.0));
  return mix(roughness, projected, projectionBlend);
}

fn anisotropicAverageRoughness(roughness: f32, anisotropy: f32) -> f32 {
  let alpha = max(roughness * roughness, ${PT_WEBGPU_MICROFACET_ALPHA_FLOOR});
  let axes = computeAnisotropicAxes(alpha, anisotropy);
  let alphaRms = sqrt(0.5 * (axes.x * axes.x + axes.y * axes.y));
  let projected = sqrt(clamp(alphaRms, 1e-4, 1.0));
  return mix(roughness, projected, clamp(0.15 * anisotropy, 0.0, 0.15));
}

fn anisotropicMultiscatterScale(anisotropy: f32, roughnessForScale: f32) -> f32 {
  // The GGX E table is isotropic. As anisotropy grows, keep the projected lookup
  // conservative instead of over-promising native anisotropic multiscatter closure.
  // Medium-gloss anisotropic VNDF paths are already close to furnace closure; the
  // empirical correction mainly belongs to rough lobes where single scatter loses
  // obvious energy.
  let anisoReduction = smoothstep(0.0, 0.35, clamp(anisotropy, 0.0, 1.0));
  return mix(1.0, 0.6, anisoReduction) * smoothstep(0.35, 0.9, roughnessForScale);
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
fn evaluateBrdfFullWithClearcoatNormal(
  baseColor: vec3f,
  roughness: f32,
  metallic: f32,
  normal: vec3f,
  clearcoatNormal: vec3f,
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
  specularColor: vec3f,
  specularIntensity: f32,
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
  let f0base = materialSpecularF0(baseColor, metallic, specularColor, specularIntensity);
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
  if (anisotropy > 0.0) {
    // Build tangent frame and rotate by anisotropyRotation.
    var tanT: vec3f;
    var tanB: vec3f;
    buildOnb(normal, &tanT, &tanB);
    let c = cos(anisotropyRotation);
    let s = sin(anisotropyRotation);
    let anisoT = c * tanT + s * tanB;
    let anisoB = -s * tanT + c * tanB;
    spec = evalBrdfSpecAnisotropic(f0, roughness, anisotropy, normal, anisoT, anisoB, wo, wi);
    // B9 — anisotropy-aware Kulla-Conty approximation. The E LUT is still the
    // isotropic GGX table, but view/light lookups use projected roughness along
    // the authored anisotropy axes instead of ignoring the lobe stretch.
    let roughnessAvg = anisotropicAverageRoughness(roughness, anisotropy);
    ms = anisotropicMultiscatterScale(anisotropy, roughnessAvg) * ggxMultiscatterLobeRoughness(
      f0,
      anisotropicProjectedRoughness(wo, anisoT, anisoB, roughness, anisotropy),
      anisotropicProjectedRoughness(wi, anisoT, anisoB, roughness, anisotropy),
      roughnessAvg,
      nDotV,
      nDotL,
    );
  } else {
    let alpha = max(roughness * roughness, ${PT_WEBGPU_MICROFACET_ALPHA_FLOOR});
    let d = ggxD(nDotH, alpha);
    let g = smithG1(nDotV, roughness) * smithG1(nDotL, roughness);
    spec = (d * g) * f / max(4.0 * nDotV * nDotL, 1e-6);
    // B9 — Kulla-Conty multiscatter energy compensation.
    ms = ggxMultiscatterLobe(f0, roughness, nDotV, nDotL);
  }
  let base = diff + spec + ms;

  // Additive extension lobes (each returns BRDF kernel, no nDotL factor).
  let cc = evalClearcoatLobe(clearcoat, clearcoatRoughness, clearcoatNormal, wo, wi);
  let sh = evalSheenLobe(sheen, sheenRoughness, sheenColor, normal, wo, wi);
  return base + cc + sh;
}

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
  specularColor: vec3f,
  specularIntensity: f32,
  anisotropy: f32,
  anisotropyRotation: f32,
) -> vec3f {
  return evaluateBrdfFullWithClearcoatNormal(
    baseColor, roughness, metallic, normal, normal, wo, wi,
    clearcoat, clearcoatRoughness, sheen, sheenRoughness, sheenColor,
    iridescence, iridescenceIor, iridescenceThicknessMin, iridescenceThicknessMax,
    specularColor, specularIntensity, anisotropy, anisotropyRotation,
  );
}
fn evaluateFiniteSameSideBrdfFullWithClearcoatNormal(
  baseColor: vec3f, roughness: f32, metallic: f32, transmission: f32,
  normal: vec3f, clearcoatNormal: vec3f, wo: vec3f, wi: vec3f,
  clearcoat: f32, clearcoatRoughness: f32,
  sheen: f32, sheenRoughness: f32, sheenColor: vec3f,
  iridescence: f32, iridescenceIor: f32,
  iridescenceThicknessMin: f32, iridescenceThicknessMax: f32,
  specularColor: vec3f, specularIntensity: f32,
  anisotropy: f32, anisotropyRotation: f32,
) -> vec3f {
  let finiteBaseColor = select(
    baseColor,
    baseColor * (1.0 - clamp(transmission, 0.0, 1.0)),
    transmission > 0.0 && metallic == 0.0,
  );
  return evaluateBrdfFullWithClearcoatNormal(
    finiteBaseColor, roughness, metallic, normal, clearcoatNormal, wo, wi,
    clearcoat, clearcoatRoughness, sheen, sheenRoughness, sheenColor,
    iridescence, iridescenceIor, iridescenceThicknessMin, iridescenceThicknessMax,
    specularColor, specularIntensity, anisotropy, anisotropyRotation,
  );
}
fn evaluateFiniteBsdfFullWithClearcoatNormal(
  baseColor: vec3f, roughness: f32, metallic: f32, transmission: f32,
  etaTOverI: f32,
  normal: vec3f, clearcoatNormal: vec3f, wo: vec3f, wi: vec3f,
  clearcoat: f32, clearcoatRoughness: f32,
  sheen: f32, sheenRoughness: f32, sheenColor: vec3f,
  iridescence: f32, iridescenceIor: f32,
  iridescenceThicknessMin: f32, iridescenceThicknessMax: f32,
  specularColor: vec3f, specularIntensity: f32,
  anisotropy: f32, anisotropyRotation: f32,
  transportModeImportance: bool,
) -> vec3f {
  if (
    dot(normal, wo) > 1e-5 && dot(normal, wi) < -1e-5 &&
    transmission > 0.0 && metallic == 0.0
  ) {
    // A delta interface has no finite solid-angle density and therefore cannot
    // participate in explicit finite-light/environment sampling. Rough
    // transmission is continuous and is evaluated below for matched MIS.
    if (bsdfDielectricIsSmooth(roughness)) { return vec3f(0.0); }
    let ft = evaluateRoughDielectricTransmission(
      roughness, etaTOverI, normal, wo, wi,
      anisotropy, anisotropyRotation, transportModeImportance,
      iridescence, iridescenceIor,
      iridescenceThicknessMin, iridescenceThicknessMax,
      specularColor, specularIntensity,
    );
    return baseColor * clamp(transmission, 0.0, 1.0) * ft;
  }
  if (transmission > 0.0 && metallic == 0.0) {
    let cosO = dot(normal, wo);
    let cosI = dot(normal, wi);
    if (cosO <= 1e-5 || cosI <= 1e-5) { return vec3f(0.0); }
    let macroF = materialDielectricFresnel(
      abs(cosO), etaTOverI,
      iridescence, iridescenceIor,
      iridescenceThicknessMin, iridescenceThicknessMax,
      specularColor, specularIntensity,
    );
    let diffuse = baseColor * (vec3f(1.0) - macroF) *
      (1.0 - clamp(transmission, 0.0, 1.0)) * INV_PI;
    let specular = evaluateRoughDielectricReflection(
      roughness, etaTOverI, normal, wo, wi,
      anisotropy, anisotropyRotation,
      iridescence, iridescenceIor,
      iridescenceThicknessMin, iridescenceThicknessMax,
      specularColor, specularIntensity,
    );
    let cc = evalClearcoatLobe(
      clearcoat, clearcoatRoughness, clearcoatNormal, wo, wi,
    );
    let sh = evalSheenLobe(
      sheen, sheenRoughness, sheenColor, normal, wo, wi,
    );
    return diffuse + specular + cc + sh;
  }
  return evaluateFiniteSameSideBrdfFullWithClearcoatNormal(
    baseColor, roughness, metallic, transmission,
    normal, clearcoatNormal, wo, wi,
    clearcoat, clearcoatRoughness,
    sheen, sheenRoughness, sheenColor,
    iridescence, iridescenceIor,
    iridescenceThicknessMin, iridescenceThicknessMax,
    specularColor, specularIntensity,
    anisotropy, anisotropyRotation,
  );
}


fn bsdfDielectricFiniteEventProbabilities(
  roughness: f32,
  transmission: f32,
  etaTOverI: f32,
  normal: vec3f,
  wo: vec3f,
  wi: vec3f,
  iridescence: f32,
  iridescenceIor: f32,
  iridescenceThicknessMin: f32,
  iridescenceThicknessMax: f32,
  specularColor: vec3f,
  specularIntensity: f32,
) -> vec3f {
  let t = clamp(transmission, 0.0, 1.0);
  let oriented = bsdfOrientDielectricInterface(normal, wo, etaTOverI);
  let macroF = materialDielectricFresnel(
    abs(dot(oriented.normal, wo)), oriented.etaTOverI,
    iridescence, iridescenceIor,
    iridescenceThicknessMin, iridescenceThicknessMax,
    specularColor, specularIntensity,
  );
  let macroFProbability = clamp(luminance(macroF), 0.0, 1.0);
  let diffuseProbability = (1.0 - macroFProbability) * (1.0 - t);
  let dielectricProbability = max(1.0 - diffuseProbability, 0.0);
  var wm: vec3f;
  if (dot(normal, wo) * dot(normal, wi) > 0.0) {
    wm = safe_normalize(wo + wi);
    if (dot(wm, oriented.normal) < 0.0) { wm = -wm; }
  } else {
    wm = bsdfRoughTransmissionHalfVector(
      normal, wo, wi, etaTOverI,
    );
  }
  let microfacetF = materialDielectricFresnel(
    abs(dot(wo, wm)), oriented.etaTOverI,
    iridescence, iridescenceIor,
    iridescenceThicknessMin, iridescenceThicknessMax,
    specularColor, specularIntensity,
  );
  let microfacetFProbability = clamp(luminance(microfacetF), 0.0, 1.0);
  let dielectricNorm = max(
    microfacetFProbability + t * (1.0 - microfacetFProbability), 1e-8,
  );
  return vec3f(
    dielectricProbability * microfacetFProbability / dielectricNorm,
    diffuseProbability,
    dielectricProbability * t * (1.0 - microfacetFProbability) / dielectricNorm,
  );
}


fn brdfFiniteBaseLobeWeights(
  baseColor: vec3f, metallic: f32, transmission: f32, etaTOverI: f32,
  nDotV: f32, specularColor: vec3f, specularIntensity: f32,
  iridescence: f32, iridescenceIor: f32,
  iridescenceThicknessMin: f32, iridescenceThicknessMax: f32,
) -> vec2f {
  if (transmission > 0.0 && metallic == 0.0) {
    let fresnelProbability = clamp(luminance(materialDielectricFresnel(
      abs(nDotV), max(etaTOverI, 1e-4),
      iridescence, iridescenceIor,
      iridescenceThicknessMin, iridescenceThicknessMax,
      specularColor, specularIntensity,
    )), 0.0, 1.0);
    let diffuseProbability =
      (1.0 - fresnelProbability) * (1.0 - clamp(transmission, 0.0, 1.0));
    return vec2f(fresnelProbability, diffuseProbability);
  }
  let f0Base = materialSpecularF0(baseColor, metallic, specularColor, specularIntensity);
  let f0 = iridescenceModifiedF0(
    f0Base, iridescence, iridescenceIor,
    iridescenceThicknessMin, iridescenceThicknessMax, nDotV,
  );
  let fresnel = fresnelSchlick(nDotV, f0);
  let specProbability =
    clamp(mix(0.04, 0.96, max(luminance(fresnel), metallic)), 0.04, 0.96);
  return vec2f(specProbability, 1.0 - specProbability);
}
// brdfDirectionalPdfFull: the pdf for the full lobe mixture used in MIS.
// The base pdf comes from brdfDirectionalPdf; the clearcoat and sheen terms add
// their (weighted) pdfs. The sheen PDF mirrors the Charlie half-vector sampler.
// When all extension scalars are 0 the result is identical to brdfDirectionalPdf.
fn brdfDirectionalPdfFullWithClearcoatNormal(
  baseColor: vec3f,
  roughness: f32,
  metallic: f32,
  transmission: f32,
  etaTOverI: f32,
  normal: vec3f,
  clearcoatNormal: vec3f,
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
  specularColor: vec3f,
  specularIntensity: f32,
  anisotropy: f32,
  anisotropyRotation: f32,
) -> f32 {
  // Item 7 — when anisotropic, replace the isotropic specular PDF with the
  // anisotropic VNDF PDF. The diffuse/trans lobe probabilities stay identical.
  var basePdf: f32;
  if (anisotropy > 0.0) {
    // Compute lobe probabilities (same as brdfDirectionalPdf).
    let wiDotN = dot(normal, wi);
    let woDotN = dot(normal, wo);
    let nDotV = max(woDotN, 0.0);
    if (nDotV <= 1e-5) { return 0.0; }
    let lobeWeights = brdfFiniteBaseLobeWeights(
      baseColor, metallic, transmission, etaTOverI,
      nDotV, specularColor, specularIntensity,
      iridescence, iridescenceIor,
      iridescenceThicknessMin, iridescenceThicknessMax,
    );
    var specWeight = lobeWeights.x;
    var diffWeight = lobeWeights.y;
    if (transmission > 0.0 && metallic == 0.0) {
      let eventProbabilities = bsdfDielectricFiniteEventProbabilities(
        roughness, transmission, etaTOverI, normal, wo, wi,
        iridescence, iridescenceIor,
        iridescenceThicknessMin, iridescenceThicknessMax,
        specularColor, specularIntensity,
      );
      specWeight = eventProbabilities.x;
      diffWeight = eventProbabilities.y;
    }
    let sameHemisphere = wiDotN * woDotN > 0.0;
    if (!sameHemisphere) {
      if (transmission <= 0.0 || metallic != 0.0) { return 0.0; }
      let eventProbabilities = bsdfDielectricFiniteEventProbabilities(
        roughness, transmission, etaTOverI, normal, wo, wi,
        iridescence, iridescenceIor,
        iridescenceThicknessMin, iridescenceThicknessMax,
        specularColor, specularIntensity,
      );
      return eventProbabilities.z * bsdfRoughTransmissionPdf(
        roughness, etaTOverI, normal, wo, wi,
        anisotropy, anisotropyRotation,
      );
    }
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
    basePdf = diffWeight * pdfDiff + specWeight * pdfSpec;
  } else {
    basePdf = brdfDirectionalPdfWithIridescence(
      baseColor, roughness, metallic, transmission, etaTOverI, normal, wo, wi,
      specularColor, specularIntensity,
      iridescence, iridescenceIor, iridescenceThicknessMin, iridescenceThicknessMax,
    );
  }
  // Clearcoat PDF: VNDF GGX at clearcoat roughness, weighted by clearcoat scalar.
  let ccPdf = clearcoat * clearcoatPdf(clearcoat, clearcoatRoughness, clearcoatNormal, wo, wi);
  // Sheen PDF: Charlie half-vector sampler matching evalSheenLobe.
  let sheenPdf = sheen * charlieSheenPdf(sheen, sheenRoughness, normal, wo, wi);
  // Iridescence does NOT add a new sampling lobe; it modifies the F0 of the
  // existing specular lobe, so the base PDF helper folds it into the lobe split.
  // Total pdf: sum of all lobe pdfs.
  let total = basePdf + ccPdf + sheenPdf;
  return total;
}

fn brdfDirectionalPdfFull(
  baseColor: vec3f,
  roughness: f32,
  metallic: f32,
  transmission: f32,
  etaTOverI: f32,
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
  specularColor: vec3f,
  specularIntensity: f32,
  anisotropy: f32,
  anisotropyRotation: f32,
) -> f32 {
  return brdfDirectionalPdfFullWithClearcoatNormal(
    baseColor, roughness, metallic, transmission, etaTOverI, normal, normal, wo, wi,
    clearcoat, clearcoatRoughness, sheen, sheenRoughness,
    iridescence, iridescenceIor, iridescenceThicknessMin, iridescenceThicknessMax,
    specularColor, specularIntensity, anisotropy, anisotropyRotation,
  );
}

fn brdfExtensionLobeWeightSum(clearcoat: f32, sheen: f32) -> f32 {
  return max(1.0 + max(clearcoat, 0.0) + max(sheen, 0.0), 1.0);
}

fn brdfDirectionalPdfFullSampledWithClearcoatNormal(
  baseColor: vec3f,
  roughness: f32,
  metallic: f32,
  transmission: f32,
  etaTOverI: f32,
  normal: vec3f,
  clearcoatNormal: vec3f,
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
  specularColor: vec3f,
  specularIntensity: f32,
  anisotropy: f32,
  anisotropyRotation: f32,
) -> f32 {
  return brdfDirectionalPdfFullWithClearcoatNormal(
    baseColor, roughness, metallic, transmission, etaTOverI, normal, clearcoatNormal, wo, wi,
    clearcoat, clearcoatRoughness, sheen, sheenRoughness,
    iridescence, iridescenceIor, iridescenceThicknessMin, iridescenceThicknessMax,
    specularColor, specularIntensity,
    anisotropy, anisotropyRotation,
  ) / brdfExtensionLobeWeightSum(clearcoat, sheen);
}

fn brdfDirectionalPdfFullSampled(
  baseColor: vec3f,
  roughness: f32,
  metallic: f32,
  transmission: f32,
  etaTOverI: f32,
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
  specularColor: vec3f,
  specularIntensity: f32,
  anisotropy: f32,
  anisotropyRotation: f32,
) -> f32 {
  return brdfDirectionalPdfFullSampledWithClearcoatNormal(
    baseColor, roughness, metallic, transmission, etaTOverI, normal, normal, wo, wi,
    clearcoat, clearcoatRoughness, sheen, sheenRoughness,
    iridescence, iridescenceIor, iridescenceThicknessMin, iridescenceThicknessMax,
    specularColor, specularIntensity, anisotropy, anisotropyRotation,
  );
}

fn evaluateBrdf(
  baseColor: vec3f,
  roughness: f32,
  metallic: f32,
  normal: vec3f,
  wo: vec3f,
  wi: vec3f,
  specularColor: vec3f,
  specularIntensity: f32,
) -> vec3f {
  let nDotL = max(dot(normal, wi), 0.0);
  let nDotV = max(dot(normal, wo), 0.0);
  if (nDotL <= 1e-5 || nDotV <= 1e-5) {
    return vec3f(0.0);
  }
  let h = safe_normalize(wi + wo);
  let nDotH = max(dot(normal, h), 0.0);
  let vDotH = max(dot(wo, h), 0.0);
  let f0 = materialSpecularF0(baseColor, metallic, specularColor, specularIntensity);
  let f = fresnelSchlick(vDotH, f0);
  let alpha = max(roughness * roughness, ${PT_WEBGPU_MICROFACET_ALPHA_FLOOR});
  let d = ggxD(nDotH, alpha);
  let g = smithG1(nDotV, roughness) * smithG1(nDotL, roughness);
  let spec = (d * g) * f / max(4.0 * nDotV * nDotL, 1e-6);
  let kd = (vec3f(1.0) - f) * (1.0 - metallic);
  let diff = kd * baseColor * INV_PI;
  // B9 — Kulla-Conty multiscatter energy compensation (see evaluateBrdfFull).
  let ms = ggxMultiscatterLobe(f0, roughness, nDotV, nDotL);
  return diff + spec + ms;
}

fn brdfDirectionalPdfWithIridescence(
  baseColor: vec3f,
  roughness: f32,
  metallic: f32,
  transmission: f32,
  etaTOverI: f32,
  normal: vec3f,
  wo: vec3f,
  wi: vec3f,
  specularColor: vec3f,
  specularIntensity: f32,
  iridescence: f32,
  iridescenceIor: f32,
  iridescenceThicknessMin: f32,
  iridescenceThicknessMax: f32,
) -> f32 {
  let wiDotN = dot(normal, wi);
  let woDotN = dot(normal, wo);
  let nDotV = max(woDotN, 0.0);
  if (nDotV <= 1e-5) {
    return 0.0;
  }
  let h = safe_normalize(wi + wo);
  let nDotH = max(dot(normal, h), 0.0);
  let lobeWeights = brdfFiniteBaseLobeWeights(
    baseColor, metallic, transmission, etaTOverI,
    nDotV, specularColor, specularIntensity,
    iridescence, iridescenceIor,
    iridescenceThicknessMin, iridescenceThicknessMax,
  );
  var specWeight = lobeWeights.x;
  var diffWeight = lobeWeights.y;
  if (transmission > 0.0 && metallic == 0.0) {
    let eventProbabilities = bsdfDielectricFiniteEventProbabilities(
      roughness, transmission, etaTOverI, normal, wo, wi,
      iridescence, iridescenceIor,
      iridescenceThicknessMin, iridescenceThicknessMax,
      specularColor, specularIntensity,
    );
    specWeight = eventProbabilities.x;
    diffWeight = eventProbabilities.y;
  }
  let sameHemisphere = wiDotN * woDotN > 0.0;
  if (!sameHemisphere) {
    if (transmission <= 0.0 || metallic != 0.0) { return 0.0; }
    let eventProbabilities = bsdfDielectricFiniteEventProbabilities(
      roughness, transmission, etaTOverI, normal, wo, wi,
      iridescence, iridescenceIor,
      iridescenceThicknessMin, iridescenceThicknessMax,
      specularColor, specularIntensity,
    );
    return eventProbabilities.z * bsdfRoughTransmissionPdf(
      roughness, etaTOverI, normal, wo, wi, 0.0, 0.0,
    );
  }
  let nDotL = max(wiDotN, 0.0);
  if (nDotL <= 1e-5) {
    return 0.0;
  }
  let alpha = max(roughness * roughness, ${PT_WEBGPU_MICROFACET_ALPHA_FLOOR});
  let d = ggxD(nDotH, alpha);
  // VNDF reflection PDF (Heitz 2018 JCGT 7(4) §3, Eq. 17):
  //   p_VNDF(h | wo) = D(h) · G1(wo) · max(0, wo·h) / (N·wo)
  // With reflection Jacobian dω_h/dω_wi = 1/(4·|wo·h|), this collapses to
  //   p_VNDF(wi | wo) = D(h) · G1(wo) / (4 · N·wo)
  // which matches the glossyReflectionSample sampler (sampleGgxVndfTangent).
  // Earlier revisions used the NDF half-vector PDF (d · N·h / (4 · wo·h));
  // that distribution and the VNDF sampler disagree, biasing MIS weights.
  let g1Wo = smithG1(nDotV, roughness);
  let pdfSpec = select((d * g1Wo) / max(4.0 * nDotV, 1e-6), 0.0, bsdfDielectricIsSmooth(roughness));
  let pdfDiff = nDotL * INV_PI;
  return diffWeight * pdfDiff + specWeight * pdfSpec;
}

fn brdfDirectionalPdf(
  baseColor: vec3f,
  roughness: f32,
  metallic: f32,
  transmission: f32,
  etaTOverI: f32,
  normal: vec3f,
  wo: vec3f,
  wi: vec3f,
  specularColor: vec3f,
  specularIntensity: f32,
) -> f32 {
  return brdfDirectionalPdfWithIridescence(
    baseColor, roughness, metallic, transmission, etaTOverI, normal, wo, wi,
    specularColor, specularIntensity,
    0.0, 1.3, 0.0, 0.0,
  );
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
fn cosineHemisphereSample(rng: ptr<function, PtRngState>, n: vec3f) -> BsdfSample {
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

fn charlieSheenSample(
  rng: ptr<function, PtRngState>,
  wo: vec3f,
  n: vec3f,
  t: vec3f,
  b: vec3f,
  sheenRoughness: f32,
) -> BsdfSample {
  let u1 = rand_f32(rng);
  let u2 = rand_f32(rng);
  let alpha = max(sheenRoughness * sheenRoughness, ${PT_WEBGPU_MICROFACET_ALPHA_FLOOR});
  let invAlpha = 1.0 / max(alpha, 1e-4);
  let sinThetaH = pow(u1, 1.0 / (invAlpha + 2.0));
  let cosThetaH = sqrt(max(0.0, 1.0 - sinThetaH * sinThetaH));
  let phi = 2.0 * PI * u2;
  let hWorld = safe_normalize(
    sinThetaH * cos(phi) * t +
    sinThetaH * sin(phi) * b +
    cosThetaH * n
  );
  let wi = safe_normalize(reflect(-wo, hWorld));

  var result: BsdfSample;
  result.wi = wi;
  let nDotL = max(dot(n, wi), 0.0);
  let nDotV = max(dot(n, wo), 0.0);
  if (nDotL <= 1e-5 || nDotV <= 1e-5) {
    result.pdf = 0.0;
    result.value = vec3f(0.0);
  } else {
    result.pdf = charlieSheenPdf(1.0, sheenRoughness, n, wo, wi);
    let h = safe_normalize(wi + wo);
    let nDotH = max(dot(n, h), 0.0);
    result.value = vec3f(charlieD(nDotH, alpha) * sheenVisibility(nDotL, nDotV));
  }
  return result;
}

/**
 * Heitz 2018 VNDF sample (Algorithm 1).
 * Input: wo in surface tangent-space (N = +Z); alpha = roughness².
 * Output: sampled half-vector h in tangent-space.
 * Ref: Heitz, E. "Sampling the GGX Distribution of Visible Normals."
 *      JCGT 7(4):1–13, 2018. https://jcgt.org/published/0007/04/01/paper.pdf
 */
fn sampleGgxVndfTangent(wo: vec3f, alpha: f32, rng: ptr<function, PtRngState>) -> vec3f {
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
fn glossyReflectionSample(rng: ptr<function, PtRngState>, wo: vec3f, n: vec3f, t: vec3f, b: vec3f, roughness: f32) -> BsdfSample {
  let alpha   = max(roughness * roughness, ${PT_WEBGPU_MICROFACET_ALPHA_FLOOR});
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

const BSDF_LOBE_NONE: u32 = 0u;
const BSDF_LOBE_SPECULAR_REFLECTION: u32 = 1u;
const BSDF_LOBE_DIFFUSE_REFLECTION: u32 = 2u;
const BSDF_LOBE_CLEARCOAT: u32 = 3u;
const BSDF_LOBE_SHEEN: u32 = 4u;
const BSDF_LOBE_DELTA_TRANSMISSION: u32 = 5u;
const BSDF_LOBE_ROUGH_TRANSMISSION: u32 = 6u;
const BSDF_LOBE_DELTA_REFLECTION: u32 = 7u;

// Exactly zero roughness is a Dirac interface. Every authored positive value
// has finite connection support, even when the microfacet implementation uses
// an internal numerical alpha floor to keep the continuous density stable.
fn bsdfDielectricIsSmooth(roughness: f32) -> bool {
  return roughness <= ${ROUGH_DIELECTRIC_SMOOTH_THRESHOLD};
}
fn bsdfHasFiniteConnectionSupport(
  roughness: f32,
  metallic: f32,
  transmission: f32,
  clearcoat: f32,
  sheen: f32,
) -> bool {
  let allDeltaDielectric =
    metallic == 0.0 &&
    transmission >= 1.0 &&
    bsdfDielectricIsSmooth(roughness);
  return !allDeltaDielectric || clearcoat > 0.0 || sheen > 0.0;
}



struct BounceSample {
  newRayOrigin: vec3f,
  newRayDir: vec3f,
  throughputMul: vec3f,
  sampledDir: vec3f,
  sampleAllowsAreaMis: bool,
  // Density of the event that was actually sampled. Continuous events use a
  // solid-angle density; delta transmission uses its discrete probability
  // mass. BDPT must not reconstruct this from the material after the fact.
  sampledEventPdf: f32,
  sampledIsDelta: bool,
  sampledLobe: u32,
  // eta_t / eta_i for the sampled interface, including nested dielectrics.
  sampledEtaTOverI: f32,
  // Continuous events remain eligible for ordinary area-light MIS.
  // WS4 — medium-crossing events for the volumetric random walk. Set true on
  // the dielectric REFRACTION branch (the only branch that crosses the
  // surface): entered when refracting into a translucent front face, exited
  // when refracting out through a back face. Reflection / diffuse / specular
  // branches stay on the same side, so both remain false there.
  enteredMedium: bool,
  exitedMedium: bool,
}

// Finite events are sampled from a normalized mixture of the base,
// clearcoat, and sheen proposals. Once a direction has been generated, the
// path estimator and every MIS consumer must use that mixture's marginal
// density and the complete finite BSDF at the sampled direction. Keeping the
// selected proposal's density/value here would put forward and reverse BDPT
// densities in different path spaces and would omit overlapping lobes.
fn finalizeFiniteBounceSampleWithClearcoatNormal(
  result: ptr<function, BounceSample>,
  baseColor: vec3f,
  roughness: f32,
  metallic: f32,
  transmission: f32,
  etaTOverI: f32,
  transportModeImportance: bool,
  normal: vec3f,
  clearcoatNormal: vec3f,
  wo: vec3f,
  clearcoat: f32,
  clearcoatRoughness: f32,
  sheen: f32,
  sheenRoughness: f32,
  sheenColor: vec3f,
  iridescence: f32,
  iridescenceIor: f32,
  iridescenceThicknessMin: f32,
  iridescenceThicknessMax: f32,
  specularColor: vec3f,
  specularIntensity: f32,
  anisotropy: f32,
  anisotropyRotation: f32,
) {
  if (
    (*result).sampledIsDelta ||
    (*result).sampledLobe == BSDF_LOBE_NONE
  ) {
    return;
  }
  let wi = (*result).sampledDir;
  let marginalPdf = brdfDirectionalPdfFullSampledWithClearcoatNormal(
    baseColor, roughness, metallic, transmission, etaTOverI,
    normal, clearcoatNormal, wo, wi,
    clearcoat, clearcoatRoughness, sheen, sheenRoughness,
    iridescence, iridescenceIor,
    iridescenceThicknessMin, iridescenceThicknessMax,
    specularColor, specularIntensity, anisotropy, anisotropyRotation,
  );
  let finiteBsdf = evaluateFiniteBsdfFullWithClearcoatNormal(
    baseColor, roughness, metallic, transmission, etaTOverI,
    normal, clearcoatNormal, wo, wi,
    clearcoat, clearcoatRoughness,
    sheen, sheenRoughness, sheenColor,
    iridescence, iridescenceIor,
    iridescenceThicknessMin, iridescenceThicknessMax,
    specularColor, specularIntensity,
    anisotropy, anisotropyRotation, transportModeImportance,
  );
  let cosine = abs(dot(normal, wi));
  if (marginalPdf <= 0.0 || cosine <= 1e-8) {
    (*result).sampledEventPdf = 0.0;
    (*result).throughputMul = vec3f(0.0);
    (*result).sampleAllowsAreaMis = false;
    return;
  }
  (*result).sampledEventPdf = marginalPdf;
  (*result).throughputMul = finiteBsdf * cosine / marginalPdf;
  (*result).sampleAllowsAreaMis = true;
}

// D9.1 — shared anisotropy axis helper (deduplicates two identical blocks in
// sampleNextBounceDirection). Returns vec2f(ax, ay).
fn computeAnisotropicAxes(alpha: f32, anisotropy: f32) -> vec2f {
  let aspect = sqrt(max(1.0 - 0.9 * anisotropy, 1e-4));
  let ax = max(alpha / aspect, 1e-4);
  let ay = max(alpha * aspect, 1e-4);
  return vec2f(ax, ay);
}

struct BsdfOrientedDielectric {
  normal: vec3f,
  etaTOverI: f32,
}

// Material interface eta is stored for the path's incident side. Reversing a
// rough-transmission edge moves wo to the other side, so both the shading
// normal and eta ratio must be reversed before evaluating a directional PDF.
fn bsdfOrientDielectricInterface(
  normal: vec3f,
  wo: vec3f,
  etaTOverI: f32,
) -> BsdfOrientedDielectric {
  let eta = max(etaTOverI, 1e-4);
  if (dot(normal, wo) >= 0.0) {
    return BsdfOrientedDielectric(normal, eta);
  }
  return BsdfOrientedDielectric(-normal, 1.0 / eta);
}
// Visible-normal density and rough dielectric transmission use the same GGX
// distribution.  This is the Walter 2007 / PBRT-v4 rough-dielectric model:
// one sampled microfacet normal, Fresnel selection at that normal, and the
// refraction Jacobian that maps its density into solid angle.
fn bsdfGgxVisibleNormalPdf(
  roughness: f32,
  anisotropy: f32,
  anisotropyRotation: f32,
  normal: vec3f,
  wo: vec3f,
  wm: vec3f,
) -> f32 {
  let nDotO = max(dot(normal, wo), 1e-6);
  let oDotM = dot(wo, wm);
  if (nDotO <= 1e-5 || oDotM <= 1e-6 || dot(normal, wm) <= 0.0) {
    return 0.0;
  }
  if (anisotropy > 0.0) {
    var tangent: vec3f;
    var bitangent: vec3f;
    buildOnb(normal, &tangent, &bitangent);
    let c = cos(anisotropyRotation);
    let s = sin(anisotropyRotation);
    let t = c * tangent + s * bitangent;
    let b = -s * tangent + c * bitangent;
    let axes = computeAnisotropicAxes(
      max(roughness * roughness, ${PT_WEBGPU_MICROFACET_ALPHA_FLOOR}), anisotropy,
    );
    let d = ggxDAnis(
      dot(wm, t), dot(wm, b), max(dot(wm, normal), 0.0),
      axes.x, axes.y,
    );
    let g1 = smithG1Anis(
      dot(wo, t), dot(wo, b), nDotO, axes.x, axes.y,
    );
    return d * g1 * oDotM / nDotO;
  }
  let alpha = max(roughness * roughness, ${PT_WEBGPU_MICROFACET_ALPHA_FLOOR});
  return ggxD(max(dot(normal, wm), 0.0), alpha) *
    smithG1(nDotO, roughness) * oDotM / nDotO;
}

fn bsdfRoughTransmissionHalfVector(
  normal: vec3f,
  wo: vec3f,
  wi: vec3f,
  etaTOverI: f32,
) -> vec3f {
  let oriented = bsdfOrientDielectricInterface(normal, wo, etaTOverI);
  var wm = safe_normalize(wo + wi * oriented.etaTOverI);
  if (dot(wm, oriented.normal) < 0.0) { wm = -wm; }
  return wm;
}

fn bsdfRoughTransmissionPdf(
  roughness: f32,
  etaTOverI: f32,
  normal: vec3f,
  wo: vec3f,
  wi: vec3f,
  anisotropy: f32,
  anisotropyRotation: f32,
) -> f32 {
  if (bsdfDielectricIsSmooth(roughness)) { return 0.0; }
  let oriented = bsdfOrientDielectricInterface(normal, wo, etaTOverI);
  let cosO = dot(oriented.normal, wo);
  let cosI = dot(oriented.normal, wi);
  if (cosO <= 1e-5 || cosI >= -1e-5) { return 0.0; }
  let eta = oriented.etaTOverI;
  let wm = bsdfRoughTransmissionHalfVector(normal, wo, wi, etaTOverI);
  if (dot(wm, wi) * cosI < 0.0 || dot(wm, wo) * cosO < 0.0) {
    return 0.0;
  }
  let denomTerm = dot(wi, wm) + dot(wo, wm) / eta;
  let jacobian = abs(dot(wi, wm)) / max(denomTerm * denomTerm, 1e-10);
  return bsdfGgxVisibleNormalPdf(
    roughness, anisotropy, anisotropyRotation, oriented.normal, wo, wm,
  ) * jacobian;
}

fn evaluateRoughDielectricTransmission(
  roughness: f32,
  etaTOverI: f32,
  normal: vec3f,
  wo: vec3f,
  wi: vec3f,
  anisotropy: f32,
  anisotropyRotation: f32,
  transportModeImportance: bool,
  iridescence: f32,
  iridescenceIor: f32,
  iridescenceThicknessMin: f32,
  iridescenceThicknessMax: f32,
  specularColor: vec3f,
  specularIntensity: f32,
) -> vec3f {
  if (bsdfDielectricIsSmooth(roughness)) { return vec3f(0.0); }
  let oriented = bsdfOrientDielectricInterface(normal, wo, etaTOverI);
  let cosO = dot(oriented.normal, wo);
  let cosI = dot(oriented.normal, wi);
  if (cosO <= 1e-5 || cosI >= -1e-5) { return vec3f(0.0); }
  let eta = oriented.etaTOverI;
  let wm = bsdfRoughTransmissionHalfVector(normal, wo, wi, etaTOverI);
  if (dot(wm, wi) * cosI < 0.0 || dot(wm, wo) * cosO < 0.0) {
    return vec3f(0.0);
  }
  var d: f32;
  var g: f32;
  if (anisotropy > 0.0) {
    var tangent: vec3f;
    var bitangent: vec3f;
    buildOnb(oriented.normal, &tangent, &bitangent);
    let c = cos(anisotropyRotation);
    let s = sin(anisotropyRotation);
    let t = c * tangent + s * bitangent;
    let b = -s * tangent + c * bitangent;
    let axes = computeAnisotropicAxes(
      max(roughness * roughness, ${PT_WEBGPU_MICROFACET_ALPHA_FLOOR}), anisotropy,
    );
    d = ggxDAnis(
      dot(wm, t), dot(wm, b), max(dot(wm, oriented.normal), 0.0),
      axes.x, axes.y,
    );

    g = smithG1Anis(dot(wo, t), dot(wo, b), cosO, axes.x, axes.y) *
      smithG1Anis(dot(wi, t), dot(wi, b), abs(cosI), axes.x, axes.y);
  } else {
    d = ggxD(max(dot(oriented.normal, wm), 0.0), max(roughness * roughness, ${PT_WEBGPU_MICROFACET_ALPHA_FLOOR}));
    g = smithG1(cosO, roughness) * smithG1(abs(cosI), roughness);
  }
  let F = materialDielectricFresnel(
    abs(dot(wo, wm)), eta,
    iridescence, iridescenceIor,
    iridescenceThicknessMin, iridescenceThicknessMax,
    specularColor, specularIntensity,
  );
  let denomTerm = dot(wi, wm) + dot(wo, wm) / eta;
  var ft = d * (vec3f(1.0) - F) * g *
    abs(dot(wi, wm) * dot(wo, wm) /
      max(abs(denomTerm * denomTerm * cosI * cosO), 1e-10));
  if (!transportModeImportance) { ft /= eta * eta; }
  return ft;
}
fn evaluateRoughDielectricReflection(
  roughness: f32,
  etaTOverI: f32,
  normal: vec3f,
  wo: vec3f,
  wi: vec3f,
  anisotropy: f32,
  anisotropyRotation: f32,
  iridescence: f32,
  iridescenceIor: f32,
  iridescenceThicknessMin: f32,
  iridescenceThicknessMax: f32,
  specularColor: vec3f,
  specularIntensity: f32,
) -> vec3f {
  if (bsdfDielectricIsSmooth(roughness)) { return vec3f(0.0); }
  let cosO = dot(normal, wo);
  let cosI = dot(normal, wi);
  if (cosO <= 1e-5 || cosI <= 1e-5) { return vec3f(0.0); }
  let wm = safe_normalize(wo + wi);
  var d: f32;
  var g: f32;
  if (anisotropy > 0.0) {
    var tangent: vec3f;
    var bitangent: vec3f;
    buildOnb(normal, &tangent, &bitangent);
    let c = cos(anisotropyRotation);
    let s = sin(anisotropyRotation);
    let t = c * tangent + s * bitangent;
    let b = -s * tangent + c * bitangent;
    let axes = computeAnisotropicAxes(
      max(roughness * roughness, ${PT_WEBGPU_MICROFACET_ALPHA_FLOOR}), anisotropy,
    );
    d = ggxDAnis(
      dot(wm, t), dot(wm, b), max(dot(wm, normal), 0.0),
      axes.x, axes.y,
    );
    g = smithG1Anis(dot(wo, t), dot(wo, b), cosO, axes.x, axes.y) *
      smithG1Anis(dot(wi, t), dot(wi, b), cosI, axes.x, axes.y);
  } else {
    d = ggxD(max(dot(normal, wm), 0.0), max(roughness * roughness, ${PT_WEBGPU_MICROFACET_ALPHA_FLOOR}));
    g = smithG1(cosO, roughness) * smithG1(cosI, roughness);
  }
  let F = materialDielectricFresnel(
    abs(dot(wo, wm)), etaTOverI,
    iridescence, iridescenceIor,
    iridescenceThicknessMin, iridescenceThicknessMax,
    specularColor, specularIntensity,
  );
  return d * g * F / max(4.0 * cosO * cosI, 1e-10);
}




fn bsdfSpecularReflectionPdf(
  roughness: f32,
  normal: vec3f,
  wo: vec3f,
  wi: vec3f,
  anisotropy: f32,
  anisotropyRotation: f32,
) -> f32 {
  if (bsdfDielectricIsSmooth(roughness)) { return 0.0; }
  let nDotV = dot(normal, wo);
  let nDotL = dot(normal, wi);
  if (nDotV <= 1e-5 || nDotL <= 1e-5) { return 0.0; }
  if (anisotropy > 0.0) {
    var tangent: vec3f;
    var bitangent: vec3f;
    buildOnb(normal, &tangent, &bitangent);
    let c = cos(anisotropyRotation);
    let s = sin(anisotropyRotation);
    let rotatedTangent = c * tangent + s * bitangent;
    let rotatedBitangent = -s * tangent + c * bitangent;
    return brdfAnisotropicSpecPdf(
      roughness, anisotropy, normal, rotatedTangent, rotatedBitangent, wo, wi,
    );
  }
  let h = safe_normalize(wo + wi);
  let nDotH = max(dot(normal, h), 0.0);
  let alpha = max(roughness * roughness, ${PT_WEBGPU_MICROFACET_ALPHA_FLOOR});
  let d = ggxD(nDotH, alpha);
  return d * smithG1(nDotV, roughness) / max(4.0 * nDotV, 1e-6);
}

fn sampleNextBounceDirectionWithClearcoatNormal(
  rng: ptr<function, PtRngState>,
  incomingDir: vec3f,
  hitPos: vec3f,
  hitNormal: vec3f,
  normal: vec3f,
  clearcoatNormal: vec3f,
  baseColor: vec3f,
  roughness: f32,
  metallic: f32,
  transmission: f32,
  etaTOverI: f32,
  transportModeImportance: bool,
  fresnel: vec3f,
  iridescence: f32,
  iridescenceIor: f32,
  iridescenceThicknessMin: f32,
  iridescenceThicknessMax: f32,
  specularColor: vec3f,
  specularIntensity: f32,
  thinFilm: ThinFilmInterface,
  isTranslucent: bool,
  clearcoat: f32,
  clearcoatRoughness: f32,
  sheen: f32,
  sheenRoughness: f32,
  sheenColor: vec3f,
  anisotropy: f32,
  anisotropyRotation: f32,
) -> BounceSample {
  // Build surface-tangent ONB once; shared by both glossy-reflect call sites.
  // Item 7 — if anisotropic, rotate the tangent frame by anisotropyRotation so
  // both tangent and bitangent align with the authored anisotropy direction.
  var tanT: vec3f;
  var tanB: vec3f;
  buildOnb(normal, &tanT, &tanB);
  if (anisotropy > 0.0) {
    let c = cos(anisotropyRotation);
    let s = sin(anisotropyRotation);
    let newT = c * tanT + s * tanB;
    let newB = -s * tanT + c * tanB;
    tanT = newT;
    tanB = newB;
  }

  var result: BounceSample;
  result.newRayOrigin = hitPos;
  result.newRayDir = vec3f(0.0);
  result.throughputMul = vec3f(0.0);
  result.sampledDir = vec3f(0.0);
  result.sampledEventPdf = 0.0;
  result.sampledIsDelta = false;
  result.sampledLobe = BSDF_LOBE_NONE;
  result.sampledEtaTOverI = max(etaTOverI, 1e-4);
  result.sampleAllowsAreaMis = false;
  result.enteredMedium = false;
  result.exitedMedium = false;

  // Coherent R/T/A owns the smooth-interface event distribution.
  if (thinFilm.enabled) {
    let wo = -incomingDir;
    let rt = thinFilmTransportRt(thinFilm, abs(dot(wo, normal)));
    let pReflect = clamp(rt.reflectanceEnergy, 0.0, 1.0);
    let pTransmit = clamp(
      rt.transmittanceEnergy, 0.0, max(0.0, 1.0 - pReflect),
    );
    let xiFilm = rand_f32(rng);
    let frontFace = dot(incomingDir, hitNormal) < 0.0;
    if (xiFilm < pReflect && pReflect > 1e-8) {
      let outDir = safe_normalize(reflect(incomingDir, normal));
      result.newRayOrigin = hitPos + normal * 1e-3;
      result.newRayDir = outDir;
      result.sampledDir = outDir;
      result.throughputMul = rt.reflectance / pReflect;
      result.sampledEventPdf = pReflect;
      result.sampledIsDelta = true;
      result.sampledLobe = BSDF_LOBE_DELTA_REFLECTION;
      return result;
    }
    if (xiFilm < pReflect + pTransmit && pTransmit > 1e-8) {
      let etaIOverT = 1.0 / max(etaTOverI, 1e-4);
      let refracted = refract(incomingDir, normal, etaIOverT);
      if (dot(refracted, refracted) <= 1e-12) { return result; }
      let outDir = safe_normalize(refracted);
      let offsetN = select(-normal, normal, dot(outDir, normal) > 0.0);
      let etaScale = select(
        etaIOverT * etaIOverT, 1.0, transportModeImportance,
      );
      result.newRayOrigin = hitPos + offsetN * 1e-3;
      result.newRayDir = outDir;
      result.sampledDir = outDir;
      result.throughputMul =
        baseColor * rt.transmittance * etaScale / pTransmit;
      result.sampledEventPdf = pTransmit;
      result.sampledIsDelta = true;
      result.sampledLobe = BSDF_LOBE_DELTA_TRANSMISSION;
      result.enteredMedium = isTranslucent && frontFace;
      result.exitedMedium = isTranslucent && !frontFace;
      return result;
    }
    // The remainder is physical absorption; zero throughput terminates.
    return result;
  }

  // -----------------------------------------------------------------------
  // Transmissive (dielectric) surface: Fresnel-weighted reflect/refract
  // partition per PBR4e §9.3 FrDielectric.
  // Ref: Pharr, Jakob, Humphreys. PBR 4th ed. §9.3 "Specular Reflection and
  //      Transmission" — DielectricBxDF::Sample_f.
  //      https://pbr-book.org/4ed/Reflection_Models/Dielectric_BSDF
  // -----------------------------------------------------------------------
  if (transmission > 0.0 && metallic == 0.0) {
    let lobeWeightSum = brdfExtensionLobeWeightSum(clearcoat, sheen);
    let clearcoatWeight = max(clearcoat, 0.0);
    let sheenWeight = max(sheen, 0.0);
    // Transmissive dielectrics use the same normalized source-lobe mixture as
    // opaque materials: the base lobe is the Fresnel reflect/refract partition,
    // while clearcoat and sheen remain additive same-side reflection lobes.
    let xiLobe = rand_f32(rng) * lobeWeightSum;
    if (xiLobe < 1.0) {
      let wo = -incomingDir;
      let etaIOverT = 1.0 / max(etaTOverI, 1e-4);
      var wm = normal;
      if (!bsdfDielectricIsSmooth(roughness)) {
        if (anisotropy > 0.0) {
          let woLocal = vec3f(
            dot(wo, tanT), dot(wo, tanB), dot(wo, normal),
          );
          let axes = computeAnisotropicAxes(
            max(roughness * roughness, ${PT_WEBGPU_MICROFACET_ALPHA_FLOOR}), anisotropy,
          );
          let wmLocal = sampleGgxVndfAnisTangent(
            woLocal, axes.x, axes.y, rng,
          );
          wm = safe_normalize(
            wmLocal.x * tanT + wmLocal.y * tanB + wmLocal.z * normal,
          );
        } else {
          let wmLocal = sampleGgxVndfTangent(
            vec3f(dot(wo, tanT), dot(wo, tanB), dot(wo, normal)),
            max(roughness * roughness, ${PT_WEBGPU_MICROFACET_ALPHA_FLOOR}),
            rng,
          );
          wm = safe_normalize(
            wmLocal.x * tanT + wmLocal.y * tanB + wmLocal.z * normal,
          );
        }
      }
      let refractedDir = refract(incomingDir, wm, etaIOverT);
      var microfacetF = materialDielectricFresnel(
        abs(dot(wo, wm)), etaTOverI,
        iridescence, iridescenceIor,
        iridescenceThicknessMin, iridescenceThicknessMax,
        specularColor, specularIntensity,
      );
      if (dot(refractedDir, refractedDir) <= 1e-12) {
        microfacetF = vec3f(1.0);
      }
      let transmissionWeight = clamp(transmission, 0.0, 1.0);
      let macroF = materialDielectricFresnel(
        abs(dot(wo, normal)), etaTOverI,
        iridescence, iridescenceIor,
        iridescenceThicknessMin, iridescenceThicknessMax,
        specularColor, specularIntensity,
      );
      let macroFProbability = clamp(luminance(macroF), 0.0, 1.0);
      let diffuseProbability =
        (1.0 - macroFProbability) * (1.0 - transmissionWeight);
      let dielectricProbability = max(1.0 - diffuseProbability, 0.0);
      let microfacetFProbability =
        clamp(luminance(microfacetF), 0.0, 1.0);
      let dielectricNorm = max(
        microfacetFProbability +
          transmissionWeight * (1.0 - microfacetFProbability),
        1e-8,
      );
      let reflectionProbability =
        dielectricProbability * microfacetFProbability / dielectricNorm;
      let transmissionProbability =
        dielectricProbability * transmissionWeight *
        (1.0 - microfacetFProbability) / dielectricNorm;
      let xiBase = xiLobe;
      let frontFace = dot(incomingDir, hitNormal) < 0.0;
      if (xiBase < reflectionProbability) {
        // Fresnel-weighted specular reflection branch.
        // materialDielectricFresnel preserves frDielectric's 1.0 for TIR, so
        // the refract branch is never selected when transmission is impossible.
        result.newRayOrigin = hitPos + normal * 1e-3;
        var bs: BsdfSample;
        bs.wi = safe_normalize(reflect(incomingDir, wm));
        bs.pdf = bsdfSpecularReflectionPdf(
          roughness, normal, wo, bs.wi, anisotropy, anisotropyRotation,
        );
        bs.value = vec3f(0.0);
        if (!bsdfDielectricIsSmooth(roughness) && (
          dot(normal, bs.wi) <= 1e-5 || bs.pdf <= 0.0
        )) { return result; }

        if (bsdfDielectricIsSmooth(roughness)) {
          let wiDelta = safe_normalize(reflect(incomingDir, normal));
          result.sampledDir = wiDelta;
          result.newRayDir = wiDelta;
          result.sampledEventPdf = reflectionProbability / lobeWeightSum;
          result.sampledIsDelta = true;
          result.sampleAllowsAreaMis = false;
          result.sampledLobe = BSDF_LOBE_DELTA_REFLECTION;
          result.throughputMul =
            microfacetF * lobeWeightSum / max(reflectionProbability, 1e-10);
        } else {
        result.sampledDir = bs.wi;
        result.newRayDir = bs.wi;
        result.sampledEventPdf = (reflectionProbability / lobeWeightSum) * bs.pdf;
        result.sampleAllowsAreaMis = true;
        result.sampledLobe = BSDF_LOBE_SPECULAR_REFLECTION;
        // MC estimator for VNDF sampling of the GGX BRDF (Heitz 2018):
        //   f·cosθ / p_VNDF = [D·G·F / (4·NdotV·NdotL)] · NdotL
        //                    / [D·G1(wo) / (4·NdotV)]
        //                    = F · G1(wi)
        // MC estimator: F · G1(wi) / R (same derivation for iso and aniso paths;
        // only the G1 function differs). nDotL = dot(n, wi).
        let nDotL = max(dot(normal, result.sampledDir), 0.0);
        var g1Wi: f32;
        if (anisotropy > 0.0) {
          let axes = computeAnisotropicAxes(max(roughness * roughness, ${PT_WEBGPU_MICROFACET_ALPHA_FLOOR}), anisotropy);
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
        let projectedRoughnessV = anisotropicProjectedRoughness(wo, tanT, tanB, roughness, anisotropy);
        let msBoostRaw = select(
          ggxMultiscatterBoost(microfacetF, roughness, nDotVcc),
          ggxMultiscatterBoostRoughness(
            microfacetF,
            projectedRoughnessV,
            nDotVcc,
          ),
          anisotropy > 0.0,
        );
        let msBoost = select(
          msBoostRaw,
          vec3f(1.0) + (msBoostRaw - vec3f(1.0)) * anisotropicMultiscatterScale(anisotropy, projectedRoughnessV),
          anisotropy > 0.0,
        );
        result.throughputMul = microfacetF * g1Wi * msBoost * lobeWeightSum / max(reflectionProbability, 1e-10);
        }
      } else if (xiBase < reflectionProbability + transmissionProbability) {
        // Fresnel-weighted refraction branch — the only branch that crosses the
        // surface, so it is where the volumetric random walk enters / exits the
        // medium (WS4). A front-face refraction of a translucent dielectric
        // enters the medium; a back-face refraction exits it.
        if (dot(refractedDir, refractedDir) <= 1e-12) {
          return result;
        }
        let outDir = safe_normalize(refractedDir);
        let offsetN = select(-normal, normal, dot(outDir, normal) > 0.0);
        result.newRayOrigin = hitPos + offsetN * 1e-3;
        result.sampledDir = outDir;
        result.newRayDir = outDir;
        if (bsdfDielectricIsSmooth(roughness)) {
          result.sampledEventPdf = transmissionProbability / lobeWeightSum;
          result.sampledIsDelta = true;
          result.sampledLobe = BSDF_LOBE_DELTA_TRANSMISSION;
        } else {
          result.sampledEventPdf =
            (transmissionProbability / lobeWeightSum) *
            bsdfRoughTransmissionPdf(
              roughness, etaTOverI, normal, wo, outDir,
              anisotropy, anisotropyRotation,
            );
          result.sampleAllowsAreaMis = result.sampledEventPdf > 0.0;
          if (dot(normal, outDir) >= -1e-5 || result.sampledEventPdf <= 0.0) {
            return result;
          }
          result.sampledLobe = BSDF_LOBE_ROUGH_TRANSMISSION;
        }
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
        // Refraction is asymmetric between radiance and importance transport
        // (PBR4e §9.5.2; Veach 1997 §5). Eye paths apply (η_i/η_t)²; light
        // subpaths transport importance and use unit eta scaling. A closed
        // enter/exit sequence in equal endpoint media cancels the two
        // radiance-mode factors.
        if (bsdfDielectricIsSmooth(roughness)) {
          let etaScale = select(
            etaIOverT * etaIOverT, 1.0, transportModeImportance,
          );
          result.throughputMul =
            baseColor * transmissionWeight * (vec3f(1.0) - microfacetF) *
            lobeWeightSum * etaScale / max(transmissionProbability, 1e-10);
        } else {
          let ft = evaluateRoughDielectricTransmission(
            roughness, etaTOverI, normal, wo, outDir,
            anisotropy, anisotropyRotation, transportModeImportance,
            iridescence, iridescenceIor,
            iridescenceThicknessMin, iridescenceThicknessMax,
            specularColor, specularIntensity,
          );
          result.throughputMul =
            baseColor * transmissionWeight * ft *
            abs(dot(normal, outDir)) /
            max(result.sampledEventPdf, 1e-10);
        }
        result.enteredMedium = isTranslucent && frontFace;
        result.exitedMedium = isTranslucent && !frontFace;
      } else {
        result.newRayOrigin = hitPos + normal * 1e-3;
        let bsDiffuse = cosineHemisphereSample(rng, normal);
        result.sampledDir = bsDiffuse.wi;
        result.newRayDir = bsDiffuse.wi;
        result.sampledEventPdf =
          (diffuseProbability / lobeWeightSum) * bsDiffuse.pdf;
        result.sampledLobe = BSDF_LOBE_DIFFUSE_REFLECTION;
        result.sampleAllowsAreaMis = true;
        let kd = (vec3f(1.0) - macroF) * (1.0 - transmissionWeight);
        result.throughputMul =
          kd * baseColor * lobeWeightSum / max(diffuseProbability, 1e-10);
      }
    } else if (xiLobe < 1.0 + clearcoatWeight) {
      let wo = -incomingDir;
      var ccTanT: vec3f;
      var ccTanB: vec3f;
      buildOnb(clearcoatNormal, &ccTanT, &ccTanB);
      result.newRayOrigin = hitPos + clearcoatNormal * 1e-3;
      let bsCc = glossyReflectionSample(rng, wo, clearcoatNormal, ccTanT, ccTanB, clearcoatRoughness);
      result.sampledDir = bsCc.wi;
      result.newRayDir = bsCc.wi;
      result.sampleAllowsAreaMis = true;
      let nDotCc = max(dot(clearcoatNormal, result.sampledDir), 0.0);
      let ccPdf = clearcoatPdf(clearcoat, clearcoatRoughness, clearcoatNormal, wo, result.sampledDir);
      let ccDensity = (clearcoatWeight / lobeWeightSum) * ccPdf;
      result.sampledEventPdf = ccDensity;
      let ccBrdf = evalClearcoatLobe(clearcoat, clearcoatRoughness, clearcoatNormal, wo, result.sampledDir);
      result.throughputMul = ccBrdf * nDotCc / max(ccDensity, 1e-8);
      result.sampledLobe = BSDF_LOBE_CLEARCOAT;
    } else {
      result.newRayOrigin = hitPos + normal * 1e-3;
      let bs = charlieSheenSample(rng, -incomingDir, normal, tanT, tanB, sheenRoughness);
      result.sampledDir = bs.wi;
      result.newRayDir = bs.wi;
      result.sampleAllowsAreaMis = true;
      let nDotSh = max(dot(normal, result.sampledDir), 0.0);
      let shPdf = bs.pdf;
      let shDensity = (sheenWeight / lobeWeightSum) * shPdf;
      result.sampledEventPdf = shDensity;
      let shBrdf = evalSheenLobe(sheen, sheenRoughness, sheenColor, normal, -incomingDir, result.sampledDir);
      result.throughputMul = shBrdf * nDotSh / max(shDensity, 1e-8);
      result.sampledLobe = BSDF_LOBE_SHEEN;
    }
    finalizeFiniteBounceSampleWithClearcoatNormal(
      &result,
      baseColor, roughness, metallic, transmission, etaTOverI,
      transportModeImportance, normal, clearcoatNormal, -incomingDir,
      clearcoat, clearcoatRoughness, sheen, sheenRoughness, sheenColor,
      iridescence, iridescenceIor,
      iridescenceThicknessMin, iridescenceThicknessMax,
      specularColor, specularIntensity, anisotropy, anisotropyRotation,
    );
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
  let lobeWeightSum = brdfExtensionLobeWeightSum(clearcoat, sheen);
  let clearcoatWeight = max(clearcoat, 0.0);
  let sheenWeight = max(sheen, 0.0);
  // Draw once from the normalized source-lobe mixture:
  //   p = (p_base + clearcoat*p_clearcoat + sheen*p_sheen)/(1+clearcoat+sheen).
  // When both extension weights are zero this is exactly the historical xi2.
  let xiLobe = rand_f32(rng) * lobeWeightSum;
  if (xiLobe < 1.0) {
    let xiBase = xiLobe;
    if (xiBase < specProb) {
    // Glossy specular reflection — Heitz 2018 VNDF.
    // Item 7 — use anisotropic sampler when anisotropy > 0; isotropic otherwise.
    // The tangent frame (tanT, tanB) is already rotated by anisotropyRotation above.
    let wo = -incomingDir;
    result.newRayOrigin = hitPos + normal * 1e-3;
    var bs2: BsdfSample;
    if (anisotropy > 0.0) {
      bs2 = glossyReflectionSampleAnisotropic(rng, wo, normal, tanT, tanB, roughness, anisotropy);
    } else {
      bs2 = glossyReflectionSample(rng, wo, normal, tanT, tanB, roughness);
    }
    result.sampledDir = bs2.wi;
    result.newRayDir = bs2.wi;
    result.sampledEventPdf = (specProb / lobeWeightSum) * bs2.pdf;
    result.sampledLobe = BSDF_LOBE_SPECULAR_REFLECTION;
    result.sampleAllowsAreaMis = true;
    let nDotL2 = max(dot(normal, result.sampledDir), 0.0);
    var g1Wi2: f32;
    if (anisotropy > 0.0) {
      let axes2 = computeAnisotropicAxes(max(roughness * roughness, ${PT_WEBGPU_MICROFACET_ALPHA_FLOOR}), anisotropy);
      g1Wi2 = smithG1Anis(dot(result.sampledDir, tanT), dot(result.sampledDir, tanB), max(nDotL2, 1e-6), axes2.x, axes2.y);
    } else {
      g1Wi2 = smithG1(nDotL2, roughness);
    }
    // B9 — multiscatter energy boost on the sampled specular reflection (see the
    // dielectric-reflection branch above).
    let nDotVcc = max(dot(normal, wo), 0.0);
    let projectedRoughnessV = anisotropicProjectedRoughness(wo, tanT, tanB, roughness, anisotropy);
    let msBoostRaw = select(
      ggxMultiscatterBoost(fresnel, roughness, nDotVcc),
      ggxMultiscatterBoostRoughness(
        fresnel,
        projectedRoughnessV,
        nDotVcc,
      ),
      anisotropy > 0.0,
    );
    let msBoost = select(
      msBoostRaw,
      vec3f(1.0) + (msBoostRaw - vec3f(1.0)) * anisotropicMultiscatterScale(anisotropy, projectedRoughnessV),
      anisotropy > 0.0,
    );
    result.throughputMul = fresnel * g1Wi2 * msBoost * lobeWeightSum / max(specProb, 1e-4);
    } else {
      result.newRayOrigin = hitPos + normal * 1e-3;
      let bs = cosineHemisphereSample(rng, normal);
      result.sampledDir = bs.wi;
      result.newRayDir = bs.wi;
      result.sampledEventPdf = (diffProb / lobeWeightSum) * bs.pdf;
      result.sampledLobe = BSDF_LOBE_DIFFUSE_REFLECTION;
      result.sampleAllowsAreaMis = true;
      let kd = (vec3f(1.0) - fresnel) * (1.0 - metallic);
      result.throughputMul = (kd * baseColor) * lobeWeightSum / max(diffProb, 1e-4);
    }
  } else if (xiLobe < 1.0 + clearcoatWeight) {
    let wo = -incomingDir;
    var ccTanT: vec3f;
    var ccTanB: vec3f;
    buildOnb(clearcoatNormal, &ccTanT, &ccTanB);
    result.newRayOrigin = hitPos + clearcoatNormal * 1e-3;
    let bsCc = glossyReflectionSample(rng, wo, clearcoatNormal, ccTanT, ccTanB, clearcoatRoughness);
    result.sampledDir = bsCc.wi;
    result.newRayDir = bsCc.wi;
    result.sampleAllowsAreaMis = true;
    let nDotCc = max(dot(clearcoatNormal, result.sampledDir), 0.0);
    let ccPdf = clearcoatPdf(clearcoat, clearcoatRoughness, clearcoatNormal, wo, result.sampledDir);
    let ccDensity = (clearcoatWeight / lobeWeightSum) * ccPdf;
    result.sampledEventPdf = ccDensity;
    result.sampledLobe = BSDF_LOBE_CLEARCOAT;
    let ccBrdf = evalClearcoatLobe(clearcoat, clearcoatRoughness, clearcoatNormal, wo, result.sampledDir);
    result.throughputMul = ccBrdf * nDotCc / max(ccDensity, 1e-8);
  } else {
    result.newRayOrigin = hitPos + normal * 1e-3;
    let bs = charlieSheenSample(rng, -incomingDir, normal, tanT, tanB, sheenRoughness);
    result.sampledDir = bs.wi;
    result.newRayDir = bs.wi;
    result.sampleAllowsAreaMis = true;
    let nDotSh = max(dot(normal, result.sampledDir), 0.0);
    let shPdf = bs.pdf;
    let shDensity = (sheenWeight / lobeWeightSum) * shPdf;
    result.sampledEventPdf = shDensity;
    result.sampledLobe = BSDF_LOBE_SHEEN;
    let shBrdf = evalSheenLobe(sheen, sheenRoughness, sheenColor, normal, -incomingDir, result.sampledDir);
    result.throughputMul = shBrdf * nDotSh / max(shDensity, 1e-8);
  }
  finalizeFiniteBounceSampleWithClearcoatNormal(
    &result,
    baseColor, roughness, metallic, transmission, etaTOverI,
    transportModeImportance, normal, clearcoatNormal, -incomingDir,
    clearcoat, clearcoatRoughness, sheen, sheenRoughness, sheenColor,
    iridescence, iridescenceIor,
    iridescenceThicknessMin, iridescenceThicknessMax,
    specularColor, specularIntensity, anisotropy, anisotropyRotation,
  );
  return result;
}

fn sampleNextBounceDirection(
  rng: ptr<function, PtRngState>,
  incomingDir: vec3f,
  hitPos: vec3f,
  hitNormal: vec3f,
  normal: vec3f,
  baseColor: vec3f,
  roughness: f32,
  metallic: f32,
  transmission: f32,
  etaTOverI: f32,
  transportModeImportance: bool,
  fresnel: vec3f,
  iridescence: f32,
  iridescenceIor: f32,
  iridescenceThicknessMin: f32,
  iridescenceThicknessMax: f32,
  specularColor: vec3f,
  specularIntensity: f32,
  thinFilm: ThinFilmInterface,
  isTranslucent: bool,
  clearcoat: f32,
  clearcoatRoughness: f32,
  sheen: f32,
  sheenRoughness: f32,
  sheenColor: vec3f,
  anisotropy: f32,
  anisotropyRotation: f32,
) -> BounceSample {
  return sampleNextBounceDirectionWithClearcoatNormal(
    rng,
    incomingDir,
    hitPos,
    hitNormal,
    normal,
    normal,
    baseColor,
    roughness,
    metallic,
    transmission,
    etaTOverI,
    transportModeImportance,
    fresnel,
    iridescence,
    iridescenceIor,
    iridescenceThicknessMin,
    iridescenceThicknessMax,
    specularColor,
    specularIntensity,
    thinFilm,
    isTranslucent,
    clearcoat,
    clearcoatRoughness,
    sheen,
    sheenRoughness,
    sheenColor,
    anisotropy,
    anisotropyRotation,
  );
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

fn hgPhase(cosThetaRaw: f32, gRaw: f32) -> f32 {
  // Evaluate the denominator around the lobe axis without subtracting two
  // nearly equal O(1) values.  This matters for authored |g| close to one:
  // the old max(..., 1e-9) flattened the physical peak by many orders of
  // magnitude.  Clamping only excludes the singular delta limit |g| = 1.
  let g = clamp(gRaw, -0.999999, 0.999999);
  let a = abs(g);
  let alignedCos = select(-clamp(cosThetaRaw, -1.0, 1.0),
                          clamp(cosThetaRaw, -1.0, 1.0), g >= 0.0);
  let oneMinusA = 1.0 - a;
  let denom = oneMinusA * oneMinusA + 2.0 * a * (1.0 - alignedCos);
  return INV_4PI * (oneMinusA * (1.0 + a)) /
    (denom * sqrt(denom));
}

// Sample a world-space direction from the HG phase function around the
// incoming-photon travel direction wIn (the direction the ray is travelling).
// Returns the scattered direction; the implied pdf equals hgPhase(cosθ, g).
fn sampleHenyeyGreenstein(rng: ptr<function, PtRngState>, wIn: vec3f, gRaw: f32) -> vec3f {
  let u1 = rand_f32(rng);
  let u2 = rand_f32(rng);
  let g = clamp(gRaw, -0.999999, 0.999999);
  let a = abs(g);
  let q = 1.0 - 2.0 * u1;
  var alignedCosTheta: f32;
  if (a < 0.125) {
    // Algebraically exact rational form of the HG inverse.  Unlike treating a
    // small non-zero g as isotropic, it preserves every authored anisotropy and
    // avoids the 0/0 cancellation in the usual closed form as g approaches 0.
    let d = 1.0 + a * q;
    let numerator =
      2.0 * q + a * (q * q + 3.0) +
      2.0 * a * a * q + a * a * a * (q * q - 1.0);
    alignedCosTheta = numerator / (2.0 * d * d);
  } else {
    // Factor the two cancellation-prone expressions for the strongly
    // anisotropic case.  This remains the exact HG inverse, not an approximation.
    let oneMinusA = 1.0 - a;
    let ratio = (oneMinusA * (1.0 + a)) /
      (oneMinusA + 2.0 * a * (1.0 - u1));
    alignedCosTheta = (1.0 + a * a - ratio * ratio) / (2.0 * a);
  }
  let cosTheta = clamp(select(-alignedCosTheta, alignedCosTheta, g >= 0.0), -1.0, 1.0);
  let sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
  let phi = 2.0 * PI * u2;
  // Build an ONB around wIn and place the sampled (θ measured FROM wIn).
  var t: vec3f;
  var b: vec3f;
  buildOnb(wIn, &t, &b);
  return safe_normalize(sinTheta * cos(phi) * t + sinTheta * sin(phi) * b + cosTheta * wIn);
}
`;
