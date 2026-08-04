/**
 * BSDF module — BRDF evaluation, directional PDF, diffuse / glossy samplers,
 * and the layered single-bounce direction sampler (`sampleNextBounceDirection`).
 *
 * Production entry points bundled here:
 *  - `evaluateBrdfFullWithClearcoatNormal` and
 *    `evaluateFiniteBsdfFullWithClearcoatNormal` — layered finite-BSDF
 *    evaluation, including distinct base and clearcoat normals
 *  - `brdfDirectionalPdfFullSampledWithClearcoatNormal` — MIS-aware marginal
 *    directional PDF aligned with the layered samplers
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
// factor. KHR_materials_iridescence defines an absent thickness texture as an
// implicit sample of 1, so the authored maximum is used here. Texture-backed
// paths collapse min/max to the sampled thickness before reaching this helper.
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
  let thicknessNm = max(thicknessMax, 0.0);
  let iridF = evalIridescence(1.0, iridescenceIor, cosTheta, thicknessNm, baseF0);
  return mix(baseF0, iridF, iridescence);
}

// KHR_materials_ior + KHR_materials_specular dielectric F0 composition.
// Khronos requires clamping IOR-derived F0 × specularColor BEFORE applying
// specularIntensity. This order is observable for color factors above one.
fn materialSpecularF0(
  baseColor: vec3f,
  metallic: f32,
  etaTOverI: f32,
  specularColor: vec3f,
  specularIntensity: f32,
) -> vec3f {
  let eta = max(abs(etaTOverI), 1e-8);
  let ratio = (eta - 1.0) / (eta + 1.0);
  let iorF0 = ratio * ratio;
  let coloredDielectricF0 = min(
    vec3f(iorF0) * max(specularColor, vec3f(0.0)),
    vec3f(1.0),
  );
  let dielectricF0 =
    coloredDielectricF0 * clamp(specularIntensity, 0.0, 1.0);
  return mix(dielectricF0, baseColor, clamp(metallic, 0.0, 1.0));
}

fn materialSpecularFresnelSchlick(
  cosTheta: f32,
  f0: vec3f,
  metallic: f32,
  specularIntensity: f32,
) -> vec3f {
  let dielectricF90 = vec3f(clamp(specularIntensity, 0.0, 1.0));
  let f90 = mix(dielectricF90, vec3f(1.0), clamp(metallic, 0.0, 1.0));
  let x = clamp(1.0 - cosTheta, 0.0, 1.0);
  let x2 = x * x;
  return f0 + (f90 - f0) * x2 * x2 * x;
}

// The same Khronos F0/F90 construction is consumed by finite evaluation,
// directional PDFs, and source sampling. Exact Fresnel remains the TIR oracle:
// material controls cannot create transmission when Snell's law forbids it.
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
  let interfaceEta = max(abs(etaTOverI), 1e-8);
  let cosI = clamp(abs(cosTheta), 0.0, 1.0);
  let sinI2 = max(0.0, 1.0 - cosI * cosI);
  // Total internal reflection is achromatic unit reflectance; material F0
  // controls cannot turn a physically unavailable transmission event back on.
  if (interfaceEta < 1.0 && sinI2 >= interfaceEta * interfaceEta) {
    return vec3f(1.0);
  }

  let authoredF0 = iridescenceModifiedF0(
    materialSpecularF0(
      vec3f(1.0), 0.0, etaTOverI, specularColor, specularIntensity,
    ),
    iridescence, iridescenceIor,
    iridescenceThicknessMin, iridescenceThicknessMax, abs(cosTheta),
  );
  return materialSpecularFresnelSchlick(
    abs(cosTheta), authoredF0, 0.0, specularIntensity,
  );
}

// A coherent stack replaces (rather than adds to) the authored bare interface
// inside the rough microfacet lobe. TMM already includes the final
// incident-medium -> Material.ior substrate boundary, so adding its R to the
// authored Fresnel would count that boundary twice.
//
// Material metallic/baseColor/KHR-specular controls still define the authored
// bare-interface reflectance. We preserve them with a coated-vs-bare odds
// replacement:
//
//   odds_coated = odds_authored * (R_stack/T_stack) / (R_bare/T_bare)
//
// and allocate the TMM surviving energy R_stack+T_stack by those odds. This has
// three important invariants:
//   1. authored == physical bare Fresnel -> exactly the TMM R/T;
//   2. a zero-layer TMM stack -> exactly authored F and 1-F;
//   3. R_out + T_out == R_stack + T_stack <= 1, preserving film absorption.
//
// This is the bounded "spectral modulation ratio" construction used when an
// artistic base Fresnel must coexist with a physical coating transfer function;
// it does not invent a second optical interface.
struct BsdfLayeredInterfaceResponse {
  reflectance: vec3f,
  baseTransmittance: vec3f,
}

fn bsdfLayeredInterfaceResponse(
  baseFresnel: vec3f,
  thinFilm: ThinFilmInterface,
  microfacetCos: f32,
) -> BsdfLayeredInterfaceResponse {
  var response: BsdfLayeredInterfaceResponse;
  let baseF = clamp(baseFresnel, vec3f(0.0), vec3f(1.0));
  let baseT = vec3f(1.0) - baseF;
  response.reflectance = baseF;
  response.baseTransmittance = baseT;
  if (!thinFilm.enabled) { return response; }

  // KHR transmission is a substrate branch selection, not an optical property
  // of the coherent stack. Evaluate raw stack R/T here and apply the authored
  // transmission scalar later, alongside the ordinary diffuse/transmission
  // partition. This also avoids double-applying transmissionMap.
  var opticalFilm = thinFilm;
  opticalFilm.transmissionScale = 1.0;
  let filmRt = thinFilmTransportRt(
    opticalFilm, clamp(abs(microfacetCos), 0.0, 1.0),
  );
  let etaIncident = select(
    thinFilm.substrateIor, thinFilm.incidentIor, thinFilm.frontFace,
  );
  let etaTransmitted = select(
    thinFilm.incidentIor, thinFilm.substrateIor, thinFilm.frontFace,
  );
  let interfaceCos = select(
    1.0, clamp(abs(microfacetCos), 0.0, 1.0), thinFilm.angleDependent,
  );
  let bareR = frDielectric(
    interfaceCos,
    max(etaTransmitted / max(etaIncident, 1e-4), 1e-4),
  );
  let bareT = 1.0 - bareR;
  let reflectedWeight =
    baseF * filmRt.reflectance / max(bareR, 1e-6);
  let transmittedWeight =
    baseT * filmRt.transmittance / max(bareT, 1e-6);
  let weightSum = reflectedWeight + transmittedWeight;
  var reflectedFraction = clamp(
    select(
      baseF,
      reflectedWeight / max(weightSum, vec3f(1e-20)),
      weightSum > vec3f(1e-20),
    ),
    vec3f(0.0),
    vec3f(1.0),
  );
  // Exact one-sided events (notably reverse-incidence TIR) must not turn into
  // 0/0 odds when an artistic authored channel is exactly zero or one.
  reflectedFraction = select(
    reflectedFraction,
    vec3f(1.0),
    (filmRt.transmittance <= vec3f(1e-20)) &
      (filmRt.reflectance > vec3f(1e-20)),
  );
  reflectedFraction = select(
    reflectedFraction,
    vec3f(0.0),
    (filmRt.reflectance <= vec3f(1e-20)) &
      (filmRt.transmittance > vec3f(1e-20)),
  );
  let survivingEnergy = clamp(
    filmRt.reflectance + filmRt.transmittance,
    vec3f(0.0),
    vec3f(1.0),
  );
  response.reflectance = survivingEnergy * reflectedFraction;
  response.baseTransmittance =
    survivingEnergy * (vec3f(1.0) - reflectedFraction);
  return response;
}

fn materialDielectricLayeredInterface(
  cosTheta: f32,
  etaTOverI: f32,
  iridescence: f32,
  iridescenceIor: f32,
  iridescenceThicknessMin: f32,
  iridescenceThicknessMax: f32,
  specularColor: vec3f,
  specularIntensity: f32,
  thinFilm: ThinFilmInterface,
) -> BsdfLayeredInterfaceResponse {
  // ThinFilmStack overrides KHR_materials_iridescence by core contract.
  let activeIridescence = select(iridescence, 0.0, thinFilm.enabled);
  let baseF = materialDielectricFresnel(
    cosTheta, etaTOverI,
    activeIridescence, iridescenceIor,
    iridescenceThicknessMin, iridescenceThicknessMax,
    specularColor, specularIntensity,
  );
  return bsdfLayeredInterfaceResponse(baseF, thinFilm, cosTheta);
}

fn bsdfNoThinFilm() -> ThinFilmInterface {
  return ThinFilmInterface(
    false, 0u, 0u, 1.0, 1.5, false, true, false, 550.0, 1.0,
  );
}

// ── Clearcoat (outer GGX layer at fixed IOR 1.5) ─────────────────────────────
// Ref: glTF KHR_materials_clearcoat (Spec rev 3.0) §3.
//      Burley, "Physically-Based Shading at Disney," SIGGRAPH 2012 §5.4.
// The clearcoat lobe is an outer GGX specular layer at a fixed IOR of 1.5
// (F0 = 0.04). Its directional Fresnel attenuates the base-plus-sheen response
// before the coat reflection is added.
// The caller supplies the ALREADY COMPUTED clearcoat roughness (= clearcoatRoughness²
// evaluated with the shared finite-alpha numerical floor used by base GGX); the
// clearcoat scalar weights the result.
// The lobe uses the same Cook-Torrance estimator as the base specular branch.
// evalClearcoatLobe returns the BRDF kernel (WITHOUT nDotL) so it can be
// summed with the lower layers and the caller multiplies by nDotL once, matching
// the convention used throughout the kernel's NEE paths.
fn clearcoatLayerWeight(
  clearcoat: f32,
  clearcoatNormal: vec3f,
  wo: vec3f,
) -> f32 {
  // KHR_materials_clearcoat evaluates the fixed-IOR coat Fresnel at the
  // view/clearcoat-normal angle. This weight is shared by the coat reflection
  // and attenuation of every lower layer.
  let vDotN = clamp(abs(dot(clearcoatNormal, wo)), 0.0, 1.0);
  let fcc = fresnelSchlick(vDotN, vec3f(0.04)).x;
  return clamp(clearcoat, 0.0, 1.0) * fcc;
}

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
  // Fixed IOR 1.5 → F0 = ((1.5-1)/(1.5+1))² = 0.04. The ratified glTF
  // layering model evaluates that Fresnel at abs(V·Nc), not V·H.
  let layerWeight = clearcoatLayerWeight(clearcoat, normal, wo);
  let alpha = max(clearcoatRoughness * clearcoatRoughness, ${PT_WEBGPU_MICROFACET_ALPHA_FLOOR});
  let d = ggxD(nDotH, alpha);
  let g = smithG1(nDotV, clearcoatRoughness) * smithG1(nDotL, clearcoatRoughness);
  // BRDF kernel (no nDotL) — caller multiplies by nDotL together with the base lobe.
  let spec = (d * g) / max(4.0 * nDotV * nDotL, 1e-6);
  return vec3f(layerWeight * spec);
}

// KHR_materials_clearcoat layering:
//   f_layered = f_below * (1 - clearcoat * Fcc) + f_clearcoat.
// Use the same view/clearcoat-normal Fresnel as evalClearcoatLobe so the
// attenuation and reflected coat cannot disagree directionally.
fn clearcoatBaseAttenuation(
  clearcoat: f32,
  clearcoatNormal: vec3f,
  wo: vec3f,
  _wi: vec3f,
) -> f32 {
  if (clearcoat <= 0.0) { return 1.0; }
  return clamp(1.0 - clearcoatLayerWeight(clearcoat, clearcoatNormal, wo), 0.0, 1.0);
}

// Clearcoat PDF contribution for the layered directional PDF.
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
// of the layered evaluator (caller multiplies by nDotL once for the full NEE
// contribution).
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

// Estevez & Kulla 2017, section 5: directional-albedo compensation for a
// sheen layer. The fitted directional albedo bounds the energy removed from
// the base by the brighter of the incoming/outgoing directions.
fn sheenDirectionalAlbedo(cosThetaRaw: f32, alpha: f32) -> f32 {
  let cosTheta = clamp(cosThetaRaw, 0.0, 1.0);
  let c = 1.0 - cosTheta;
  let c3 = c * c * c;
  return 0.65584461 * c3 +
    1.0 / (4.16526551 + exp(-7.97291361 * sqrt(alpha) + 6.33516894));
}

fn sheenBaseAttenuation(
  sheen: f32,
  sheenRoughness: f32,
  sheenColor: vec3f,
  normal: vec3f,
  wo: vec3f,
  wi: vec3f,
) -> f32 {
  if (sheen <= 0.0) { return 1.0; }
  // Sheen is an outer layer over the complete base material. Use magnitudes
  // so the same bounded directional-albedo loss applies to a base event that
  // crosses the interface; the sheen reflection lobe itself remains same-side.
  let nDotL = clamp(abs(dot(normal, wi)), 0.0, 1.0);
  let nDotV = clamp(abs(dot(normal, wo)), 0.0, 1.0);
  let alpha = max(sheenRoughness, 0.07);
  let alpha2 = alpha * alpha;
  let maxSheenColor = max(max(sheenColor.r, sheenColor.g), sheenColor.b);
  let eWo = sheenDirectionalAlbedo(nDotV, alpha2);
  let eWi = sheenDirectionalAlbedo(nDotL, alpha2);
  let fullSheenScale = min(
    1.0 - maxSheenColor * eWo,
    1.0 - maxSheenColor * eWi,
  );
  return clamp(mix(1.0, fullSheenScale, clamp(sheen, 0.0, 1.0)), 0.0, 1.0);
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
  fresnel: vec3f,
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
  return (d * g) * fresnel / max(4.0 * nDotV * nDotL, 1e-6);
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
  let anisotropicScale = 0.6 * smoothstep(0.35, 0.9, roughnessForScale);
  return mix(1.0, anisotropicScale, anisoReduction);
}

// ── H52 extended BRDF evaluation (base + clearcoat + sheen + iridescence) ─────
// Adds the three Disney extension lobes to the base Cook-Torrance BRDF. Returns
// the BRDF kernel (WITHOUT nDotL); callers multiply by nDotL once. When all
// extension scalars are 0 the result is the base Cook-Torrance response.
//
// iridescence modifies the base specular F0 BEFORE the Cook-Torrance evaluation
// (it is NOT an additive lobe — it replaces the F0 that governs diffuse/specular
// partition and the specular highlight colour). Clearcoat and sheen are ordered
// outer layers: each attenuates the complete response below it before its own
// same-side reflection lobe is added.
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
  thinFilm: ThinFilmInterface,
) -> vec3f {
  let nDotL = max(dot(normal, wi), 0.0);
  let nDotV = max(dot(normal, wo), 0.0);
  if (nDotL <= 1e-5 || nDotV <= 1e-5) { return vec3f(0.0); }
  let h = safe_normalize(wi + wo);
  let nDotH = max(dot(normal, h), 0.0);
  let vDotH = max(dot(wo, h), 0.0);

  // Iridescence-modified F0 (modifies diffuse/specular partition + specular colour).
  let f0base = materialSpecularF0(
    baseColor, metallic, thinFilm.substrateIor,
    specularColor, specularIntensity,
  );
  let iridescentF0 = iridescenceModifiedF0(
    f0base, iridescence, iridescenceIor,
    iridescenceThicknessMin, iridescenceThicknessMax, vDotH,
  );
  // ThinFilmStack overrides the single-layer iridescence model.
  let f0 = select(iridescentF0, f0base, thinFilm.enabled);
  let interfaceResponse = bsdfLayeredInterfaceResponse(
    materialSpecularFresnelSchlick(
      vDotH, f0, metallic, specularIntensity,
    ),
    thinFilm, vDotH,
  );
  let f = interfaceResponse.reflectance;
  let kd = interfaceResponse.baseTransmittance * (1.0 - metallic);
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
    spec = evalBrdfSpecAnisotropic(f, roughness, anisotropy, normal, anisoT, anisoB, wo, wi);
    // B9 — anisotropy-aware Kulla-Conty approximation. The E LUT is still the
    // isotropic GGX table, but view/light lookups use projected roughness along
    // the authored anisotropy axes instead of ignoring the lobe stretch.
    let roughnessAvg = anisotropicAverageRoughness(roughness, anisotropy);
    ms = anisotropicMultiscatterScale(anisotropy, roughnessAvg) * ggxMultiscatterLobeRoughness(
      bsdfLayeredInterfaceResponse(f0, thinFilm, nDotV).reflectance,
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
    ms = ggxMultiscatterLobe(
      bsdfLayeredInterfaceResponse(f0, thinFilm, nDotV).reflectance,
      roughness, nDotV, nDotL,
    );
  }
  let base = diff + spec + ms;

  // Layered extension lobes (each returns a BRDF kernel, no nDotL factor).
  // Sheen sits over the base; the outer clearcoat attenuates everything below.
  let cc = evalClearcoatLobe(clearcoat, clearcoatRoughness, clearcoatNormal, wo, wi);
  let sh = evalSheenLobe(sheen, sheenRoughness, sheenColor, normal, wo, wi);
  let sheenAttenuation = sheenBaseAttenuation(
    sheen, sheenRoughness, sheenColor, normal, wo, wi,
  );
  let clearcoatAttenuation = clearcoatBaseAttenuation(
    clearcoat, clearcoatNormal, wo, wi,
  );
  return (base * sheenAttenuation + sh) * clearcoatAttenuation + cc;
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
  thinFilm: ThinFilmInterface,
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
    specularColor, specularIntensity, anisotropy, anisotropyRotation, thinFilm,
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
  thinFilm: ThinFilmInterface,
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
      specularColor, specularIntensity, thinFilm,
    );
    let sheenAttenuation = sheenBaseAttenuation(
      sheen, sheenRoughness, sheenColor, normal, wo, wi,
    );
    let clearcoatAttenuation = clearcoatBaseAttenuation(
      clearcoat, clearcoatNormal, wo, wi,
    );
    // Cross-side transport sees only the transmitted base response. The outer
    // layer losses still apply, but their reflection lobes cannot contribute on
    // the opposite side of the interface.
    return baseColor * clamp(transmission, 0.0, 1.0) * ft *
      sheenAttenuation * clearcoatAttenuation;
  }
  if (transmission > 0.0 && metallic == 0.0) {
    let cosO = dot(normal, wo);
    let cosI = dot(normal, wi);
    if (cosO <= 1e-5 || cosI <= 1e-5) { return vec3f(0.0); }
    let macroInterface = materialDielectricLayeredInterface(
      abs(cosO), etaTOverI,
      iridescence, iridescenceIor,
      iridescenceThicknessMin, iridescenceThicknessMax,
      specularColor, specularIntensity, thinFilm,
    );
    let diffuse = baseColor * macroInterface.baseTransmittance *
      (1.0 - clamp(transmission, 0.0, 1.0)) * INV_PI;
    let specular = evaluateRoughDielectricReflection(
      roughness, etaTOverI, normal, wo, wi,
      anisotropy, anisotropyRotation,
      iridescence, iridescenceIor,
      iridescenceThicknessMin, iridescenceThicknessMax,
      specularColor, specularIntensity, thinFilm,
    );
    let cc = evalClearcoatLobe(
      clearcoat, clearcoatRoughness, clearcoatNormal, wo, wi,
    );
    let sh = evalSheenLobe(
      sheen, sheenRoughness, sheenColor, normal, wo, wi,
    );
    let sheenAttenuation = sheenBaseAttenuation(
      sheen, sheenRoughness, sheenColor, normal, wo, wi,
    );
    let clearcoatAttenuation = clearcoatBaseAttenuation(
      clearcoat, clearcoatNormal, wo, wi,
    );
    return ((diffuse + specular) * sheenAttenuation + sh) *
      clearcoatAttenuation + cc;
  }
  return evaluateFiniteSameSideBrdfFullWithClearcoatNormal(
    baseColor, roughness, metallic, transmission,
    normal, clearcoatNormal, wo, wi,
    clearcoat, clearcoatRoughness,
    sheen, sheenRoughness, sheenColor,
    iridescence, iridescenceIor,
    iridescenceThicknessMin, iridescenceThicknessMax,
    specularColor, specularIntensity,
    anisotropy, anisotropyRotation, thinFilm,
  );
}


// Represent the two nested Bernoulli decisions used by the dielectric base
// lobe. The first draw chooses diffuse versus dielectric; the second chooses
// reflection versus transmission conditioned on dielectric. Returning the
// joint probabilities alongside the conditional threshold keeps sampling,
// forward/reverse PDFs, and throughput on the exact same finite RNG measure.
// Layout: joint reflection, joint diffuse, joint transmission,
//         P(reflection | dielectric).
fn bsdfRepresentedDielectricEventProbabilities(
  diffuseWeight: f32,
  reflectionWeight: f32,
  transmissionWeight: f32,
) -> vec4f {
  let d = max(diffuseWeight, 0.0);
  let r = max(reflectionWeight, 0.0);
  let t = max(transmissionWeight, 0.0);
  let total = d + r + t;
  if (!(total > 0.0)) { return vec4f(0.0); }
  let dielectricTotal = r + t;
  if (!(dielectricTotal > 0.0)) {
    return vec4f(0.0, 1.0, 0.0, 0.0);
  }
  if (!(d > 0.0)) {
    // Keep the exact endpoint without asking a rounded ratio to recover it.
    let reflectionConditional = select(
      1.0,
      represented_bernoulli_probability_f32(r / dielectricTotal),
      t > 0.0,
    );
    return vec4f(
      reflectionConditional, 0.0, 1.0 - reflectionConditional,
      reflectionConditional,
    );
  }
  // Form the smaller side of each ratio. If one physical weight is many ulps
  // below the other, forming the larger ratio first can round it to exactly
  // one before the represented helper gets a chance to preserve both supports.
  var diffuseProbability = 0.0;
  var dielectricProbability = 0.0;
  if (d <= dielectricTotal) {
    diffuseProbability = represented_bernoulli_probability_f32(d / total);
    dielectricProbability = 1.0 - diffuseProbability;
  } else {
    dielectricProbability = represented_bernoulli_probability_f32(
      dielectricTotal / total,
    );
    diffuseProbability = 1.0 - dielectricProbability;
  }
  var reflectionConditional = 0.0;
  if (!(r > 0.0)) {
    reflectionConditional = 0.0;
  } else if (!(t > 0.0)) {
    reflectionConditional = 1.0;
  } else if (r <= t) {
    reflectionConditional = represented_bernoulli_probability_f32(
      r / dielectricTotal,
    );
  } else {
    reflectionConditional = 1.0 - represented_bernoulli_probability_f32(
      t / dielectricTotal,
    );
  }
  let reflectionProbability = dielectricProbability * reflectionConditional;
  let transmissionProbability = dielectricProbability - reflectionProbability;
  return vec4f(
    reflectionProbability,
    diffuseProbability,
    transmissionProbability,
    reflectionConditional,
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
  thinFilm: ThinFilmInterface,
) -> vec3f {
  let t = clamp(transmission, 0.0, 1.0);
  let oriented = bsdfOrientDielectricInterface(normal, wo, etaTOverI);
  let macroInterface = materialDielectricLayeredInterface(
    abs(dot(oriented.normal, wo)), oriented.etaTOverI,
    iridescence, iridescenceIor,
    iridescenceThicknessMin, iridescenceThicknessMax,
    specularColor, specularIntensity, thinFilm,
  );
  let diffuseProbability = clamp(
    luminance(macroInterface.baseTransmittance) * (1.0 - t),
    0.0,
    1.0,
  );
  var wm: vec3f;
  if (dot(normal, wo) * dot(normal, wi) > 0.0) {
    wm = safe_normalize(wo + wi);
    if (dot(wm, oriented.normal) < 0.0) { wm = -wm; }
  } else {
    wm = bsdfRoughTransmissionHalfVector(
      normal, wo, wi, etaTOverI,
    );
  }
  let microfacetInterface = materialDielectricLayeredInterface(
    abs(dot(wo, wm)), oriented.etaTOverI,
    iridescence, iridescenceIor,
    iridescenceThicknessMin, iridescenceThicknessMax,
    specularColor, specularIntensity, thinFilm,
  );
  let microfacetFProbability =
    clamp(luminance(microfacetInterface.reflectance), 0.0, 1.0);
  let microfacetTProbability =
    clamp(luminance(microfacetInterface.baseTransmittance), 0.0, 1.0);
  return bsdfRepresentedDielectricEventProbabilities(
    diffuseProbability,
    microfacetFProbability,
    t * microfacetTProbability,
  ).xyz;
}


fn brdfFiniteBaseLobeWeights(
  baseColor: vec3f, metallic: f32, transmission: f32, etaTOverI: f32,
  nDotV: f32, specularColor: vec3f, specularIntensity: f32,
  iridescence: f32, iridescenceIor: f32,
  iridescenceThicknessMin: f32, iridescenceThicknessMax: f32,
  thinFilm: ThinFilmInterface,
) -> vec2f {
  if (transmission > 0.0 && metallic == 0.0) {
    let interfaceResponse = materialDielectricLayeredInterface(
      abs(nDotV), max(etaTOverI, 1e-4),
      iridescence, iridescenceIor,
      iridescenceThicknessMin, iridescenceThicknessMax,
      specularColor, specularIntensity, thinFilm,
    );
    let fresnelProbability =
      clamp(luminance(interfaceResponse.reflectance), 0.0, 1.0);
    let diffuseProbability =
      clamp(luminance(interfaceResponse.baseTransmittance), 0.0, 1.0) *
      (1.0 - clamp(transmission, 0.0, 1.0));
    return vec2f(fresnelProbability, diffuseProbability);
  }
  let f0Base = materialSpecularF0(
    baseColor, metallic, etaTOverI, specularColor, specularIntensity,
  );
  let iridescentF0 = iridescenceModifiedF0(
    f0Base, iridescence, iridescenceIor,
    iridescenceThicknessMin, iridescenceThicknessMax, nDotV,
  );
  let f0 = select(iridescentF0, f0Base, thinFilm.enabled);
  let fresnel = bsdfLayeredInterfaceResponse(
    materialSpecularFresnelSchlick(
      nDotV, f0, metallic, specularIntensity,
    ),
    thinFilm, nDotV,
  ).reflectance;
  let specProbability = represented_bernoulli_probability_f32(
    clamp(mix(0.04, 0.96, max(luminance(fresnel), metallic)), 0.04, 0.96),
  );
  return vec2f(specProbability, 1.0 - specProbability);
}

// Exact source-lobe probabilities for the finite rand_f32 domain. The first
// draw selects base versus extensions; a second independent draw selects
// clearcoat versus sheen when an extension was chosen. This preserves every
// authored positive lobe and exposes the joint masses used by marginal PDFs.
// Layout: joint base, joint clearcoat, joint sheen,
//         P(clearcoat | extension).
fn brdfRepresentedExtensionLobeProbabilities(
  clearcoat: f32,
  sheen: f32,
) -> vec4f {
  let cc = max(clearcoat, 0.0);
  let sh = max(sheen, 0.0);
  let extensionTotal = cc + sh;
  if (!(extensionTotal > 0.0)) {
    return vec4f(1.0, 0.0, 0.0, 0.0);
  }
  let total = 1.0 + extensionTotal;
  var baseProbability = 0.0;
  var extensionProbability = 0.0;
  if (extensionTotal <= 1.0) {
    extensionProbability = represented_bernoulli_probability_f32(
      extensionTotal / total,
    );
    baseProbability = 1.0 - extensionProbability;
  } else {
    baseProbability = represented_bernoulli_probability_f32(1.0 / total);
    extensionProbability = 1.0 - baseProbability;
  }
  var clearcoatConditional = 0.0;
  if (!(cc > 0.0)) {
    clearcoatConditional = 0.0;
  } else if (!(sh > 0.0)) {
    clearcoatConditional = 1.0;
  } else if (cc <= sh) {
    clearcoatConditional = represented_bernoulli_probability_f32(
      cc / extensionTotal,
    );
  } else {
    clearcoatConditional = 1.0 - represented_bernoulli_probability_f32(
      sh / extensionTotal,
    );
  }
  let clearcoatProbability = extensionProbability * clearcoatConditional;
  return vec4f(
    baseProbability,
    clearcoatProbability,
    extensionProbability - clearcoatProbability,
    clearcoatConditional,
  );
}

fn brdfDirectionalPdfThinFilm(
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
  thinFilm: ThinFilmInterface,
) -> f32 {
  let wiDotN = dot(normal, wi);
  let woDotN = dot(normal, wo);
  let nDotV = max(woDotN, 0.0);
  if (nDotV <= 1e-5) { return 0.0; }
  let lobeWeights = brdfFiniteBaseLobeWeights(
    baseColor, metallic, transmission, etaTOverI,
    nDotV, specularColor, specularIntensity,
    iridescence, iridescenceIor,
    iridescenceThicknessMin, iridescenceThicknessMax, thinFilm,
  );
  var specWeight = lobeWeights.x;
  var diffWeight = lobeWeights.y;
  if (transmission > 0.0 && metallic == 0.0) {
    let eventProbabilities = bsdfDielectricFiniteEventProbabilities(
      roughness, transmission, etaTOverI, normal, wo, wi,
      iridescence, iridescenceIor,
      iridescenceThicknessMin, iridescenceThicknessMax,
      specularColor, specularIntensity, thinFilm,
    );
    specWeight = eventProbabilities.x;
    diffWeight = eventProbabilities.y;
  }
  let sameHemisphere = wiDotN * woDotN > 0.0;
  let extensionProbabilities = brdfRepresentedExtensionLobeProbabilities(
    clearcoat, sheen,
  );
  if (!sameHemisphere) {
    if (transmission <= 0.0 || metallic != 0.0) { return 0.0; }
    let eventProbabilities = bsdfDielectricFiniteEventProbabilities(
      roughness, transmission, etaTOverI, normal, wo, wi,
      iridescence, iridescenceIor,
      iridescenceThicknessMin, iridescenceThicknessMax,
      specularColor, specularIntensity, thinFilm,
    );
    return extensionProbabilities.x * eventProbabilities.z *
      bsdfRoughTransmissionPdf(
      roughness, etaTOverI, normal, wo, wi,
      anisotropy, anisotropyRotation,
    );
  }
  let nDotL = max(wiDotN, 0.0);
  if (nDotL <= 1e-5) { return 0.0; }
  var pdfSpec: f32;
  if (anisotropy > 0.0) {
    var tanT: vec3f;
    var tanB: vec3f;
    buildOnb(normal, &tanT, &tanB);
    let c = cos(anisotropyRotation);
    let s = sin(anisotropyRotation);
    let anisoT = c * tanT + s * tanB;
    let anisoB = -s * tanT + c * tanB;
    pdfSpec = brdfAnisotropicSpecPdf(
      roughness, anisotropy, normal, anisoT, anisoB, wo, wi,
    );
  } else {
    let h = safe_normalize(wo + wi);
    let nDotH = max(dot(normal, h), 0.0);
    let alpha = max(
      roughness * roughness, ${PT_WEBGPU_MICROFACET_ALPHA_FLOOR},
    );
    pdfSpec = ggxD(nDotH, alpha) * smithG1(nDotV, roughness) /
      max(4.0 * nDotV, 1e-6);
  }
  pdfSpec = select(
    pdfSpec,
    0.0,
    bsdfBaseReflectionIsDelta(roughness, metallic, transmission),
  );
  let basePdf = diffWeight * nDotL * INV_PI + specWeight * pdfSpec;
  return extensionProbabilities.x * basePdf +
    extensionProbabilities.y * clearcoatPdf(
      clearcoat, clearcoatRoughness, clearcoatNormal, wo, wi,
    ) +
    extensionProbabilities.z * charlieSheenPdf(
      sheen, sheenRoughness, normal, wo, wi,
    );
}
// PDF for the full lobe mixture used in MIS. The clearcoat and sheen terms add
// their weighted PDFs to the base mixture. The sheen PDF mirrors the Charlie
// half-vector sampler. With all extension scalars at zero, this reduces to the
// base diffuse/specular/transmission mixture.
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
  thinFilm: ThinFilmInterface,
) -> f32 {
  if (thinFilm.enabled) {
    return brdfDirectionalPdfThinFilm(
      baseColor, roughness, metallic, transmission, etaTOverI,
      normal, clearcoatNormal, wo, wi,
      clearcoat, clearcoatRoughness, sheen, sheenRoughness,
      iridescence, iridescenceIor,
      iridescenceThicknessMin, iridescenceThicknessMax,
      specularColor, specularIntensity,
      anisotropy, anisotropyRotation, thinFilm,
    );
  }
  let extensionProbabilities = brdfRepresentedExtensionLobeProbabilities(
    clearcoat, sheen,
  );
  // Item 7 — when anisotropic, replace the isotropic specular PDF with the
  // anisotropic VNDF PDF. The diffuse/trans lobe probabilities stay identical.
  var basePdf: f32;
  if (anisotropy > 0.0) {
    // Compute the base lobe probabilities shared with the sampling path.
    let wiDotN = dot(normal, wi);
    let woDotN = dot(normal, wo);
    let nDotV = max(woDotN, 0.0);
    if (nDotV <= 1e-5) { return 0.0; }
    let lobeWeights = brdfFiniteBaseLobeWeights(
      baseColor, metallic, transmission, etaTOverI,
      nDotV, specularColor, specularIntensity,
      iridescence, iridescenceIor,
      iridescenceThicknessMin, iridescenceThicknessMax, thinFilm,
    );
    var specWeight = lobeWeights.x;
    var diffWeight = lobeWeights.y;
    if (transmission > 0.0 && metallic == 0.0) {
      let eventProbabilities = bsdfDielectricFiniteEventProbabilities(
        roughness, transmission, etaTOverI, normal, wo, wi,
        iridescence, iridescenceIor,
        iridescenceThicknessMin, iridescenceThicknessMax,
        specularColor, specularIntensity, thinFilm,
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
        specularColor, specularIntensity, thinFilm,
      );
      return extensionProbabilities.x * eventProbabilities.z *
        bsdfRoughTransmissionPdf(
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
    let pdfSpec = select(
      brdfAnisotropicSpecPdf(
        roughness, anisotropy, normal, anisoT, anisoB, wo, wi,
      ),
      0.0,
      bsdfBaseReflectionIsDelta(roughness, metallic, transmission),
    );
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
  let ccPdf = clearcoatPdf(
    clearcoat, clearcoatRoughness, clearcoatNormal, wo, wi,
  );
  // Sheen PDF: Charlie half-vector sampler matching evalSheenLobe.
  let sheenPdf = charlieSheenPdf(
    sheen, sheenRoughness, normal, wo, wi,
  );
  // Iridescence does NOT add a new sampling lobe; it modifies the F0 of the
  // existing specular lobe, so the base PDF helper folds it into the lobe split.
  // Total pdf: sum of all lobe pdfs.
  let total = extensionProbabilities.x * basePdf +
    extensionProbabilities.y * ccPdf + extensionProbabilities.z * sheenPdf;
  return total;
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
  thinFilm: ThinFilmInterface,
) -> f32 {
  return brdfDirectionalPdfFullWithClearcoatNormal(
    baseColor, roughness, metallic, transmission, etaTOverI, normal, clearcoatNormal, wo, wi,
    clearcoat, clearcoatRoughness, sheen, sheenRoughness,
    iridescence, iridescenceIor, iridescenceThicknessMin, iridescenceThicknessMax,
    specularColor, specularIntensity,
    anisotropy, anisotropyRotation, thinFilm,
  );
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
  thinFilm: ThinFilmInterface,
) -> f32 {
  return brdfDirectionalPdfFullSampledWithClearcoatNormal(
    baseColor, roughness, metallic, transmission, etaTOverI, normal, normal, wo, wi,
    clearcoat, clearcoatRoughness, sheen, sheenRoughness,
    iridescence, iridescenceIor, iridescenceThicknessMin, iridescenceThicknessMax,
    specularColor, specularIntensity, anisotropy, anisotropyRotation, thinFilm,
  );
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
    iridescenceThicknessMin, iridescenceThicknessMax, bsdfNoThinFilm(),
  );
  var specWeight = lobeWeights.x;
  var diffWeight = lobeWeights.y;
  if (transmission > 0.0 && metallic == 0.0) {
    let eventProbabilities = bsdfDielectricFiniteEventProbabilities(
      roughness, transmission, etaTOverI, normal, wo, wi,
      iridescence, iridescenceIor,
      iridescenceThicknessMin, iridescenceThicknessMax,
      specularColor, specularIntensity, bsdfNoThinFilm(),
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
      specularColor, specularIntensity, bsdfNoThinFilm(),
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
  let pdfSpec = select(
    (d * g1Wo) / max(4.0 * nDotV, 1e-6),
    0.0,
    bsdfBaseReflectionIsDelta(roughness, metallic, transmission),
  );
  let pdfDiff = nDotL * INV_PI;
  return diffWeight * pdfDiff + specWeight * pdfSpec;
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
 *           the layered directional PDF's specular branch for MIS consistency
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
// One represented non-bulk sheet, sampled as entry refraction followed by a
// reciprocal virtual exit refraction. This is one path event with a joint
// forward/reverse density and never changes the authored medium stack.
const BSDF_LOBE_COMPOUND_THIN_SHEET_TRANSMISSION: u32 = 8u;

// Exactly zero roughness is a Dirac interface. Every authored positive value
// has finite connection support, even when the microfacet implementation uses
// an internal numerical alpha floor to keep the continuous density stable.
fn bsdfDielectricIsSmooth(roughness: f32) -> bool {
  return roughness <= ${ROUGH_DIELECTRIC_SMOOTH_THRESHOLD};
}

// The base-reflection sampler is Dirac only in the transmissive dielectric
// branch. Opaque and metallic materials remain finite GGX at the shared
// numerical alpha floor, including when their authored roughness is exactly 0.
fn bsdfBaseReflectionIsDelta(
  roughness: f32,
  metallic: f32,
  transmission: f32,
) -> bool {
  return metallic == 0.0 &&
    transmission > 0.0 &&
    bsdfDielectricIsSmooth(roughness);
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
  // Reverse density of the sampled event. Compound sheets need this explicit
  // joint density because no arbitrary-direction one-interface evaluator can
  // reconstruct their two latent microfacet draws.
  sampledReverseEventPdf: f32,
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
  thinFilm: ThinFilmInterface,
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
    specularColor, specularIntensity, anisotropy, anisotropyRotation, thinFilm,
  );
  let finiteBsdf = evaluateFiniteBsdfFullWithClearcoatNormal(
    baseColor, roughness, metallic, transmission, etaTOverI,
    normal, clearcoatNormal, wo, wi,
    clearcoat, clearcoatRoughness,
    sheen, sheenRoughness, sheenColor,
    iridescence, iridescenceIor,
    iridescenceThicknessMin, iridescenceThicknessMax,
    specularColor, specularIntensity,
    anisotropy, anisotropyRotation, thinFilm, transportModeImportance,
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
  thinFilm: ThinFilmInterface,
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
  let interfaceResponse = materialDielectricLayeredInterface(
    abs(dot(wo, wm)), eta,
    iridescence, iridescenceIor,
    iridescenceThicknessMin, iridescenceThicknessMax,
    specularColor, specularIntensity, thinFilm,
  );
  let denomTerm = dot(wi, wm) + dot(wo, wm) / eta;
  var ft = d * interfaceResponse.baseTransmittance * g *
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
  thinFilm: ThinFilmInterface,
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
  let interfaceResponse = materialDielectricLayeredInterface(
    abs(dot(wo, wm)), etaTOverI,
    iridescence, iridescenceIor,
    iridescenceThicknessMin, iridescenceThicknessMax,
    specularColor, specularIntensity, thinFilm,
  );
  return d * g * interfaceResponse.reflectance /
    max(4.0 * cosO * cosI, 1e-10);
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

struct ThinSheetInterfaceSample {
  valid: bool,
  wi: vec3f,
  throughput: vec3f,
  pdfFwd: f32,
  pdfRev: f32,
  sampledDelta: bool,
};

// Conditional transmission sample for exactly one dielectric interface. It
// deliberately excludes material color, face-layer absorption, and the outer
// lobe-selection probability. Those factors belong once to the compound sheet
// event; the entry interface may evaluate the authored coherent TMM stack,
// while the virtual exit calls this helper with bsdfNoThinFilm().
fn sampleThinSheetInterface(
  rng: ptr<function, PtRngState>,
  wo: vec3f,
  normal: vec3f,
  roughness: f32,
  etaTOverI: f32,
  transportModeImportance: bool,
  iridescence: f32,
  iridescenceIor: f32,
  iridescenceThicknessMin: f32,
  iridescenceThicknessMax: f32,
  specularColor: vec3f,
  specularIntensity: f32,
  anisotropy: f32,
  anisotropyRotation: f32,
  thinFilm: ThinFilmInterface,
) -> ThinSheetInterfaceSample {
  var result: ThinSheetInterfaceSample;
  result.valid = false;
  result.wi = vec3f(0.0);
  result.throughput = vec3f(0.0);
  result.pdfFwd = 0.0;
  result.pdfRev = 0.0;
  result.sampledDelta = bsdfDielectricIsSmooth(roughness);

  let oriented = bsdfOrientDielectricInterface(normal, wo, etaTOverI);
  if (!(dot(oriented.normal, wo) > 1e-5)) { return result; }
  var wm = oriented.normal;
  if (!result.sampledDelta) {
    var tangent: vec3f;
    var bitangent: vec3f;
    buildOnb(oriented.normal, &tangent, &bitangent);
    if (anisotropy > 0.0) {
      let c = cos(anisotropyRotation);
      let s = sin(anisotropyRotation);
      let rotatedTangent = c * tangent + s * bitangent;
      let rotatedBitangent = -s * tangent + c * bitangent;
      tangent = rotatedTangent;
      bitangent = rotatedBitangent;
    }
    let woLocal = vec3f(
      dot(wo, tangent), dot(wo, bitangent), dot(wo, oriented.normal),
    );
    if (anisotropy > 0.0) {
      let axes = computeAnisotropicAxes(
        max(roughness * roughness, ${PT_WEBGPU_MICROFACET_ALPHA_FLOOR}),
        anisotropy,
      );
      let wmLocal = sampleGgxVndfAnisTangent(
        woLocal, axes.x, axes.y, rng,
      );
      wm = safe_normalize(
        wmLocal.x * tangent + wmLocal.y * bitangent +
          wmLocal.z * oriented.normal,
      );
    } else {
      let wmLocal = sampleGgxVndfTangent(
        woLocal,
        max(roughness * roughness, ${PT_WEBGPU_MICROFACET_ALPHA_FLOOR}),
        rng,
      );
      wm = safe_normalize(
        wmLocal.x * tangent + wmLocal.y * bitangent +
          wmLocal.z * oriented.normal,
      );
    }
  }

  let wiRaw = refract(-wo, wm, 1.0 / max(oriented.etaTOverI, 1e-4));
  // The bounded two-interface estimator owns no internal-reflection chain.
  // Exit TIR is therefore a null transmission draw, never redirected into the
  // entry reflection proposal.
  if (!(dot(wiRaw, wiRaw) > 1e-12)) { return result; }
  result.wi = safe_normalize(wiRaw);
  if (result.sampledDelta) {
    let interfaceResponse = materialDielectricLayeredInterface(
      abs(dot(wo, oriented.normal)), oriented.etaTOverI,
      iridescence, iridescenceIor,
      iridescenceThicknessMin, iridescenceThicknessMax,
      specularColor, specularIntensity, thinFilm,
    );
    let etaIOverT = 1.0 / max(oriented.etaTOverI, 1e-4);
    let etaScale = select(
      etaIOverT * etaIOverT, 1.0, transportModeImportance,
    );
    result.throughput = interfaceResponse.baseTransmittance * etaScale;
    result.pdfFwd = 1.0;
    result.pdfRev = 1.0;
  } else {
    result.pdfFwd = bsdfRoughTransmissionPdf(
      roughness, etaTOverI, normal, wo, result.wi,
      anisotropy, anisotropyRotation,
    );
    result.pdfRev = bsdfRoughTransmissionPdf(
      roughness, etaTOverI, normal, result.wi, wo,
      anisotropy, anisotropyRotation,
    );
    if (!(result.pdfFwd > 0.0) || !(result.pdfRev > 0.0)) { return result; }
    let interfaceValue = evaluateRoughDielectricTransmission(
      roughness, etaTOverI, normal, wo, result.wi,
      anisotropy, anisotropyRotation, transportModeImportance,
      iridescence, iridescenceIor,
      iridescenceThicknessMin, iridescenceThicknessMax,
      specularColor, specularIntensity, thinFilm,
    );
    result.throughput = interfaceValue * abs(dot(normal, result.wi)) /
      result.pdfFwd;
  }
  result.valid =
    result.pdfFwd > 0.0 && result.pdfFwd <= 3.402823e38 &&
    result.pdfRev > 0.0 && result.pdfRev <= 3.402823e38 &&
    all(result.throughput >= vec3f(0.0)) &&
    all(result.throughput == result.throughput) &&
    all(result.throughput <= vec3f(3.402823e38));
  return result;
}

// Exact straight-connection response of a smooth represented sheet. A rough
// sheet has two latent microfacet draws and therefore no arbitrary-direction
// visibility evaluator; it returns zero and remains nonconnectable. The return
// value is the complete sheet attenuation, or zero for mismatch/TIR/invalid.
fn thinSheetExactVisibilityTransmission(
  hit: SceneHit,
  rayDirectionRaw: vec3f,
  heroLambda: f32,
  incidentIor: f32,
) -> vec3f {
  let matId = hitMaterialId(hit);
  let mat = decodeMaterial(matId);
  if (!mat.isThinSheet || mat.isUnlit) { return vec3f(0.0); }
  let rayDirection = safe_normalize(rayDirectionRaw);
  if (!(dot(rayDirection, rayDirection) > 0.0)) { return vec3f(0.0); }
  let frontFace = hit.frontFace;
  let interfaceBaseNormal = select(-hit.normal, hit.normal, frontFace);
  var normal = interfaceBaseNormal;
  normal = applyNormalMap(
    matId, hit.triIndex, hit.baryVW, normal, hit.instanceIndex, frontFace,
  );
  normal = applyBumpMap(
    matId, hit.triIndex, hit.baryVW, normal, hit.instanceIndex,
  );
  var exitNormal = -interfaceBaseNormal;
  exitNormal = applyNormalMap(
    matId, hit.triIndex, hit.baryVW, exitNormal, hit.instanceIndex, !frontFace,
  );
  exitNormal = applyBumpMap(
    matId, hit.triIndex, hit.baryVW, exitNormal, hit.instanceIndex,
  );
  var clearcoatNormal = interfaceBaseNormal;
  clearcoatNormal = applyClearcoatNormalMap(
    matId, hit.triIndex, hit.baryVW, clearcoatNormal, hit.instanceIndex,
  );
  let orm = sampleOrmTexture(
    matId, hit.triIndex, hit.baryVW, hit.instanceIndex,
  );
  var entryRoughness = clamp(mat.roughness * orm.g, 0.0, 1.0);
  var exitRoughness = entryRoughness;
  let entryLayerRoughness = select(
    mat.backLayerRoughness, mat.frontLayerRoughness, frontFace,
  );
  let exitLayerRoughness = select(
    mat.frontLayerRoughness, mat.backLayerRoughness, frontFace,
  );
  if (entryLayerRoughness >= 0.0) {
    entryRoughness = clamp(entryLayerRoughness, 0.0, 1.0);
  }
  if (exitLayerRoughness >= 0.0) {
    exitRoughness = clamp(exitLayerRoughness, 0.0, 1.0);
  }
  if (
    !bsdfDielectricIsSmooth(entryRoughness) ||
    !bsdfDielectricIsSmooth(exitRoughness)
  ) {
    return vec3f(0.0);
  }

  let transmission = clamp(
    mat.transmission * sampleTransmissionTexture(
      matId, hit.triIndex, hit.baryVW, hit.instanceIndex,
    ),
    0.0, 1.0,
  );
  if (!(transmission > 0.0)) { return vec3f(0.0); }
  var baseColor = mat.baseColor *
    sampleVertexColor(hit.triIndex, hit.baryVW).rgb *
    sampleBaseColorTexture(
      matId, hit.triIndex, hit.baryVW, hit.instanceIndex,
    ).rgb *
    sampleAoFactor(matId, hit.triIndex, hit.baryVW, hit.instanceIndex);
  let entryLayerTx = clamp(
    select(mat.backLayerTx, mat.frontLayerTx, frontFace),
    vec3f(0.0), vec3f(1.0),
  );
  let exitLayerTx = clamp(
    select(mat.frontLayerTx, mat.backLayerTx, frontFace),
    vec3f(0.0), vec3f(1.0),
  );
  let entryLayerWeight = select(
    entryLayerTx,
    activeLayerWeightRgb(entryLayerTx, heroLambda, true),
    params.spectralEnabled != 0u && luminance(entryLayerTx) < 0.999,
  );
  let exitLayerWeight = select(
    exitLayerTx,
    activeLayerWeightRgb(exitLayerTx, heroLambda, true),
    params.spectralEnabled != 0u && luminance(exitLayerTx) < 0.999,
  );
  baseColor = baseColor * entryLayerWeight;
  if (params.spectralEnabled != 0u) {
    baseColor = vec3f(spectralCombinedReflectanceAtHero(
      baseColor, mat.baseColor, mat.spectralReflCoeffs,
      mat.hasSpectralReflectance, heroLambda,
    ));
  }
  var materialIor = mat.ior;
  if (params.spectralEnabled != 0u && mat.dispersionAbbe > 0.0) {
    materialIor = cauchyIorAtLambda(
      heroLambda, mat.ior, mat.dispersionAbbe,
    );
  }
  let entryEta = max(materialIor, 1e-4) / max(incidentIor, 1e-4);
  let entryWo = -rayDirection;
  let entryOriented = bsdfOrientDielectricInterface(
    normal, entryWo, entryEta,
  );
  let internalRaw = refract(
    -entryWo, entryOriented.normal,
    1.0 / max(entryOriented.etaTOverI, 1e-4),
  );
  if (!(dot(internalRaw, internalRaw) > 1e-12)) { return vec3f(0.0); }
  let internalDirection = safe_normalize(internalRaw);
  var iridescence = clamp(
    mat.iridescence * sampleIridescenceTexture(
      matId, hit.triIndex, hit.baryVW, hit.instanceIndex,
    ),
    0.0, 1.0,
  );
  var iridescenceThicknessMin = mat.iridescenceThicknessMin;
  var iridescenceThicknessMax = mat.iridescenceThicknessMax;
  let iridescenceThicknessSample = sampleIridescenceThicknessTexture(
    matId, hit.triIndex, hit.baryVW, hit.instanceIndex,
  );
  if (iridescenceThicknessSample >= 0.0) {
    let thickness = mix(
      iridescenceThicknessMin,
      iridescenceThicknessMax,
      iridescenceThicknessSample,
    );
    iridescenceThicknessMin = thickness;
    iridescenceThicknessMax = thickness;
    if (thickness <= 0.0) { iridescence = 0.0; }
  }
  var specularColor = max(
    mat.specularColor * sampleSpecularColorTexture(
      matId, hit.triIndex, hit.baryVW, hit.instanceIndex,
    ),
    vec3f(0.0),
  );
  if (params.spectralEnabled != 0u) {
    specularColor = vec3f(spectralRgbFactorAtHero(
      specularColor, heroLambda,
    ));
  }
  let specularIntensity = clamp(
    mat.specularIntensity * sampleSpecularIntensityTexture(
      matId, hit.triIndex, hit.baryVW, hit.instanceIndex,
    ),
    0.0, 1.0,
  );
  let entryFilm = ThinFilmInterface(
    mat.thinFilmEnabled, matId, mat.thinFilmLayerCountU,
    mat.thinFilmIncidentIor, materialIor, mat.thinFilmAngleDependent,
    frontFace, params.spectralEnabled != 0u, heroLambda, transmission,
  );
  let entryResponse = materialDielectricLayeredInterface(
    abs(dot(entryWo, entryOriented.normal)), entryOriented.etaTOverI,
    iridescence, mat.iridescenceIor,
    iridescenceThicknessMin, iridescenceThicknessMax,
    specularColor, specularIntensity, entryFilm,
  );
  let exitWo = -internalDirection;
  let exitOriented = bsdfOrientDielectricInterface(
    exitNormal, exitWo, entryEta,
  );
  let finalRaw = refract(
    -exitWo, exitOriented.normal,
    1.0 / max(exitOriented.etaTOverI, 1e-4),
  );
  if (!(dot(finalRaw, finalRaw) > 1e-12)) { return vec3f(0.0); }
  let finalDirection = safe_normalize(finalRaw);
  if (dot(finalDirection, rayDirection) < 1.0 - 1e-5) {
    return vec3f(0.0);
  }
  let exitResponse = materialDielectricLayeredInterface(
    abs(dot(exitWo, exitOriented.normal)), exitOriented.etaTOverI,
    iridescence, mat.iridescenceIor,
    iridescenceThicknessMin, iridescenceThicknessMax,
    specularColor, specularIntensity, bsdfNoThinFilm(),
  );
  let clearcoat = clamp(
    mat.clearcoat * sampleClearcoatTexture(
      matId, hit.triIndex, hit.baryVW, hit.instanceIndex,
    ),
    0.0, 1.0,
  );
  var sheenColor = clamp(
    mat.sheenColor * sampleSheenColorTexture(
      matId, hit.triIndex, hit.baryVW, hit.instanceIndex,
    ),
    vec3f(0.0), vec3f(1.0),
  );
  if (params.spectralEnabled != 0u) {
    sheenColor = vec3f(spectralRgbFactorAtHero(sheenColor, heroLambda));
  }
  let sheenRoughness = clamp(
    mat.sheenRoughness * sampleSheenRoughnessTexture(
      matId, hit.triIndex, hit.baryVW, hit.instanceIndex,
    ),
    0.0, 1.0,
  );
  let outerLayerAttenuation =
    sheenBaseAttenuation(
      mat.sheen, sheenRoughness, sheenColor,
      normal, entryWo, internalDirection,
    ) *
    clearcoatBaseAttenuation(
      clearcoat, clearcoatNormal, entryWo, internalDirection,
    );
  let attenuation = baseColor * transmission *
    entryResponse.baseTransmittance * exitResponse.baseTransmittance *
    max(exitLayerWeight, vec3f(0.0)) * outerLayerAttenuation;
  if (
    any(attenuation != attenuation) ||
    any(attenuation < vec3f(0.0)) ||
    any(attenuation > vec3f(3.402823e38))
  ) {
    return vec3f(0.0);
  }
  return attenuation;
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
  isThinSheet: bool,
  oppositeNormal: vec3f,
  oppositeRoughness: f32,
  oppositeLayerWeight: vec3f,
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
  result.sampledReverseEventPdf = 0.0;
  result.sampledIsDelta = false;
  result.sampledLobe = BSDF_LOBE_NONE;
  result.sampledEtaTOverI = max(etaTOverI, 1e-4);
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
    let extensionProbabilities = brdfRepresentedExtensionLobeProbabilities(
      clearcoat, sheen,
    );
    // Transmissive dielectrics use the same normalized source-lobe mixture as
    // opaque materials: the base lobe is the Fresnel reflect/refract partition,
    // while clearcoat and sheen provide same-side proposals. The shared finite
    // evaluator applies their ordered lower-layer attenuation after sampling.
    if (rand_f32(rng) < extensionProbabilities.x) {
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
      var microfacetInterface = materialDielectricLayeredInterface(
        abs(dot(wo, wm)), etaTOverI,
        iridescence, iridescenceIor,
        iridescenceThicknessMin, iridescenceThicknessMax,
        specularColor, specularIntensity, thinFilm,
      );
      if (dot(refractedDir, refractedDir) <= 1e-12) {
        microfacetInterface.reflectance = vec3f(1.0);
        microfacetInterface.baseTransmittance = vec3f(0.0);
      }
      let transmissionWeight = clamp(transmission, 0.0, 1.0);
      let macroInterface = materialDielectricLayeredInterface(
        abs(dot(wo, normal)), etaTOverI,
        iridescence, iridescenceIor,
        iridescenceThicknessMin, iridescenceThicknessMax,
        specularColor, specularIntensity, thinFilm,
      );
      let diffuseProbability = clamp(
        luminance(macroInterface.baseTransmittance) *
          (1.0 - transmissionWeight),
        0.0,
        1.0,
      );
      let microfacetFProbability =
        clamp(luminance(microfacetInterface.reflectance), 0.0, 1.0);
      let microfacetTProbability =
        clamp(luminance(microfacetInterface.baseTransmittance), 0.0, 1.0);
      let eventProbabilities = bsdfRepresentedDielectricEventProbabilities(
        diffuseProbability,
        microfacetFProbability,
        transmissionWeight * microfacetTProbability,
      );
      let reflectionProbability = eventProbabilities.x;
      let transmissionProbability = eventProbabilities.z;
      let chooseDiffuse = rand_f32(rng) < eventProbabilities.y;
      var chooseReflection = false;
      if (!chooseDiffuse) {
        chooseReflection = rand_f32(rng) < eventProbabilities.w;
      }
      let frontFace = dot(incomingDir, hitNormal) < 0.0;
      if (chooseReflection) {
        // Fresnel-weighted specular reflection branch.
        // materialDielectricFresnel preserves frDielectric's 1.0 for TIR, so
        // the refract branch is never selected when transmission is impossible.
        result.newRayOrigin = hitPos + normal * ptRayOriginBias();
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
          result.sampledEventPdf =
            extensionProbabilities.x * reflectionProbability;
          result.sampledIsDelta = true;
          result.sampleAllowsAreaMis = false;
          result.sampledLobe = BSDF_LOBE_DELTA_REFLECTION;
          let sheenAttenuation = sheenBaseAttenuation(
            sheen, sheenRoughness, sheenColor, normal, wo, wiDelta,
          );
          let clearcoatAttenuation = clearcoatBaseAttenuation(
            clearcoat, clearcoatNormal, wo, wiDelta,
          );
          result.throughputMul =
            microfacetInterface.reflectance * sheenAttenuation *
            clearcoatAttenuation / max(result.sampledEventPdf, 1e-10);
        } else {
          result.sampledDir = bs.wi;
          result.newRayDir = bs.wi;
          result.sampledLobe = BSDF_LOBE_SPECULAR_REFLECTION;
          // The shared finite finalizer below owns the marginal density and the
          // complete layered-BSDF estimator for every continuous event.
        }
      } else if (!chooseDiffuse) {
        // Fresnel-weighted refraction branch — the only branch that crosses the
        // surface, so it is where the volumetric random walk enters / exits the
        // medium (WS4). A front-face refraction of a translucent dielectric
        // enters the medium; a back-face refraction exits it.
        if (dot(refractedDir, refractedDir) <= 1e-12) {
          return result;
        }
        let outDir = safe_normalize(refractedDir);
        let offsetN = select(-normal, normal, dot(outDir, normal) > 0.0);
        result.newRayOrigin = hitPos + offsetN * ptRayOriginBias();
        result.sampledDir = outDir;
        result.newRayDir = outDir;
        if (bsdfDielectricIsSmooth(roughness)) {
          result.sampledEventPdf =
            extensionProbabilities.x * transmissionProbability;
          result.sampledIsDelta = true;
          result.sampledLobe = BSDF_LOBE_DELTA_TRANSMISSION;
        } else {
          let roughTransmissionProposalPdf =
            (extensionProbabilities.x * transmissionProbability) *
            bsdfRoughTransmissionPdf(
              roughness, etaTOverI, normal, wo, outDir,
              anisotropy, anisotropyRotation,
            );
          if (dot(normal, outDir) >= -1e-5 || roughTransmissionProposalPdf <= 0.0) {
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
          let sheenAttenuation = sheenBaseAttenuation(
            sheen, sheenRoughness, sheenColor, normal, wo, outDir,
          );
          let clearcoatAttenuation = clearcoatBaseAttenuation(
            clearcoat, clearcoatNormal, wo, outDir,
          );
          result.throughputMul =
            baseColor * transmissionWeight *
            microfacetInterface.baseTransmittance *
            sheenAttenuation * clearcoatAttenuation * etaScale /
            max(extensionProbabilities.x * transmissionProbability, 1e-10);
        } else {
          // The shared finite finalizer below owns the marginal density and the
          // complete layered-BSDF estimator for every continuous event.
        }
        result.enteredMedium = isTranslucent && frontFace;
        result.exitedMedium = isTranslucent && !frontFace;
      } else {
        result.newRayOrigin = hitPos + normal * ptRayOriginBias();
        let bsDiffuse = cosineHemisphereSample(rng, normal);
        result.sampledDir = bsDiffuse.wi;
        result.newRayDir = bsDiffuse.wi;
        result.sampledLobe = BSDF_LOBE_DIFFUSE_REFLECTION;
      }
    } else if (rand_f32(rng) < extensionProbabilities.w) {
      let wo = -incomingDir;
      var ccTanT: vec3f;
      var ccTanB: vec3f;
      buildOnb(clearcoatNormal, &ccTanT, &ccTanB);
      result.newRayOrigin =
        hitPos + clearcoatNormal * ptRayOriginBias();
      let bsCc = glossyReflectionSample(rng, wo, clearcoatNormal, ccTanT, ccTanB, clearcoatRoughness);
      result.sampledDir = bsCc.wi;
      result.newRayDir = bsCc.wi;
      result.sampledLobe = BSDF_LOBE_CLEARCOAT;
    } else {
      result.newRayOrigin = hitPos + normal * ptRayOriginBias();
      let bs = charlieSheenSample(rng, -incomingDir, normal, tanT, tanB, sheenRoughness);
      result.sampledDir = bs.wi;
      result.newRayDir = bs.wi;
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
      thinFilm,
    );
    if (
      isThinSheet && (
        result.sampledLobe == BSDF_LOBE_DELTA_TRANSMISSION ||
        result.sampledLobe == BSDF_LOBE_ROUGH_TRANSMISSION
      )
    ) {
      let entryWo = -incomingDir;
      var entryPdfRev = 1.0;
      if (!bsdfDielectricIsSmooth(roughness)) {
        entryPdfRev = bsdfRoughTransmissionPdf(
          roughness, etaTOverI, normal, result.sampledDir, entryWo,
          anisotropy, anisotropyRotation,
        );
      }
      let exitWo = -result.sampledDir;
      let exitThinFilm = bsdfNoThinFilm();
      let exitSample = sampleThinSheetInterface(
        rng, exitWo, oppositeNormal, oppositeRoughness, etaTOverI,
        transportModeImportance,
        iridescence, iridescenceIor,
        iridescenceThicknessMin, iridescenceThicknessMax,
        specularColor, specularIntensity,
        anisotropy, anisotropyRotation, exitThinFilm,
      );
      let reverseEventProbabilities = bsdfDielectricFiniteEventProbabilities(
        oppositeRoughness, transmission, etaTOverI,
        oppositeNormal, exitSample.wi, exitWo,
        iridescence, iridescenceIor,
        iridescenceThicknessMin, iridescenceThicknessMax,
        specularColor, specularIntensity, exitThinFilm,
      );
      let reverseTransmissionWeight = reverseEventProbabilities.z *
        brdfRepresentedExtensionLobeProbabilities(clearcoat, sheen).x;
      let jointPdfFwd = result.sampledEventPdf * exitSample.pdfFwd;
      let jointPdfRev = reverseTransmissionWeight *
        exitSample.pdfRev * entryPdfRev;
      let compoundThroughput = result.throughputMul *
        exitSample.throughput * max(oppositeLayerWeight, vec3f(0.0));
      if (
        !exitSample.valid || !(entryPdfRev > 0.0) ||
        !(jointPdfFwd > 0.0) || jointPdfFwd > 3.402823e38 ||
        !(jointPdfRev > 0.0) || jointPdfRev > 3.402823e38 ||
        any(compoundThroughput != compoundThroughput) ||
        any(compoundThroughput < vec3f(0.0)) ||
        any(compoundThroughput > vec3f(3.402823e38))
      ) {
        result.newRayDir = vec3f(0.0);
        result.sampledDir = vec3f(0.0);
        result.throughputMul = vec3f(0.0);
        result.sampledEventPdf = 0.0;
        result.sampledReverseEventPdf = 0.0;
        result.sampledIsDelta = false;
        result.sampledLobe = BSDF_LOBE_NONE;
        result.sampleAllowsAreaMis = false;
        result.enteredMedium = false;
        result.exitedMedium = false;
        return result;
      }
      result.newRayOrigin = hitPos;
      result.newRayDir = exitSample.wi;
      result.sampledDir = exitSample.wi;
      result.throughputMul = compoundThroughput;
      result.sampledEventPdf = jointPdfFwd;
      result.sampledReverseEventPdf = jointPdfRev;
      result.sampledIsDelta =
        bsdfDielectricIsSmooth(roughness) && exitSample.sampledDelta;
      result.sampledLobe = BSDF_LOBE_COMPOUND_THIN_SHEET_TRANSMISSION;
      result.sampledEtaTOverI = 1.0;
      // A rough augmented sheet has two latent microfacet directions. There is
      // no arbitrary-direction finite BSDF for that augmented path space, so it
      // is deliberately excluded from ordinary NEE/BDPT connection strategies.
      result.sampleAllowsAreaMis = false;
      result.enteredMedium = false;
      result.exitedMedium = false;
    }
    return result;
  }

  // -----------------------------------------------------------------------
  // Non-transmissive surface: heuristic specular / diffuse partition.
  // -----------------------------------------------------------------------
  let woOpaque = -incomingDir;
  let nDotVOpaque = max(dot(normal, woOpaque), 0.0);
  let opaqueBaseF0 = materialSpecularF0(
    baseColor, metallic, etaTOverI, specularColor, specularIntensity,
  );
  let opaqueBaseFresnel = select(
    fresnel,
    materialSpecularFresnelSchlick(
      nDotVOpaque, opaqueBaseF0, metallic, specularIntensity,
    ),
    thinFilm.enabled,
  );
  let opaqueInterface = bsdfLayeredInterfaceResponse(
    opaqueBaseFresnel, thinFilm, nDotVOpaque,
  );
  let specProb = represented_bernoulli_probability_f32(clamp(mix(
    0.04, 0.96,
    max(luminance(opaqueInterface.reflectance), metallic),
  ), 0.04, 0.96));
  let extensionProbabilities = brdfRepresentedExtensionLobeProbabilities(
    clearcoat, sheen,
  );
  if (rand_f32(rng) < extensionProbabilities.x) {
    if (rand_f32(rng) < specProb) {
      // Glossy specular reflection — Heitz 2018 VNDF.
      // Item 7 — use anisotropic sampler when anisotropy > 0; isotropic otherwise.
      // The tangent frame (tanT, tanB) is already rotated by anisotropyRotation above.
      let wo = -incomingDir;
      result.newRayOrigin = hitPos + normal * ptRayOriginBias();
      var bs2: BsdfSample;
      if (anisotropy > 0.0) {
        bs2 = glossyReflectionSampleAnisotropic(rng, wo, normal, tanT, tanB, roughness, anisotropy);
      } else {
        bs2 = glossyReflectionSample(rng, wo, normal, tanT, tanB, roughness);
      }
      result.sampledDir = bs2.wi;
      result.newRayDir = bs2.wi;
      result.sampledLobe = BSDF_LOBE_SPECULAR_REFLECTION;
      // The shared finite finalizer below owns the marginal density and the
      // complete layered-BSDF estimator for every continuous event.
    } else {
      result.newRayOrigin = hitPos + normal * ptRayOriginBias();
      let bs = cosineHemisphereSample(rng, normal);
      result.sampledDir = bs.wi;
      result.newRayDir = bs.wi;
      result.sampledLobe = BSDF_LOBE_DIFFUSE_REFLECTION;
    }
  } else if (rand_f32(rng) < extensionProbabilities.w) {
    let wo = -incomingDir;
    var ccTanT: vec3f;
    var ccTanB: vec3f;
    buildOnb(clearcoatNormal, &ccTanT, &ccTanB);
    result.newRayOrigin =
      hitPos + clearcoatNormal * ptRayOriginBias();
    let bsCc = glossyReflectionSample(rng, wo, clearcoatNormal, ccTanT, ccTanB, clearcoatRoughness);
    result.sampledDir = bsCc.wi;
    result.newRayDir = bsCc.wi;
    result.sampledLobe = BSDF_LOBE_CLEARCOAT;
  } else {
    result.newRayOrigin = hitPos + normal * ptRayOriginBias();
    let bs = charlieSheenSample(rng, -incomingDir, normal, tanT, tanB, sheenRoughness);
    result.sampledDir = bs.wi;
    result.newRayDir = bs.wi;
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
    thinFilm,
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
  isThinSheet: bool,
  oppositeNormal: vec3f,
  oppositeRoughness: f32,
  oppositeLayerWeight: vec3f,
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
    isThinSheet,
    oppositeNormal,
    oppositeRoughness,
    oppositeLayerWeight,
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
  let q = 1.0 - 2.0 * u1;
  var cosTheta: f32;
  if (abs(g) < 0.125) {
    // Algebraically exact rational form of the HG inverse.  Unlike treating a
    // small non-zero g as isotropic, it preserves every authored anisotropy and
    // avoids the 0/0 cancellation in the usual closed form as g approaches 0.
    // Keep signed g throughout: replacing it with abs(g) changes the random-
    // variate mapping on the back-scattering side and breaks continuity at 0.
    let d = 1.0 + g * q;
    let numerator =
      2.0 * q + g * (q * q + 3.0) +
      2.0 * g * g * q + g * g * g * (q * q - 1.0);
    cosTheta = numerator / (2.0 * d * d);
  } else {
    // Factor the two cancellation-prone expressions for the strongly
    // anisotropic case.  This remains the exact HG inverse, not an approximation.
    let ratio = (1.0 - g * g) / (1.0 + g * q);
    cosTheta = (1.0 + g * g - ratio * ratio) / (2.0 * g);
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
