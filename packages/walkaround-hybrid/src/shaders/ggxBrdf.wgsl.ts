/**
 * GGX BRDF — simplified Lambertian diffuse + GGX specular.
 *
 * Split out of common.wgsl.ts (T9-stepA): `distributionGGX`,
 * `geometrySchlickGGX`, `geometrySmith`, and the `evalGGX` entry point used
 * by the ReSTIR p̂ helper and shade. Depends on `PI`/`INV_PI` (walkaroundUbo
 * module), `safe_normalize`, and `fresnelSchlick` (shared primitives module);
 * `common` aggregates all three so the symbols are in scope.
 */

import type { WgslModule } from '../pipeline/wgslComposer.js';

// INTENTIONAL per-backend divergence (complexity-sweep 2026-06-02, verified + kept
// — NOT accidental duplication): distributionGGX/geometrySchlickGGX below are kept
// local rather than shared with pt-webgpu's ggxD/smithG1 or @vitrum/shared-samplers,
// because the backends floor roughness differently — walkaround floors `rough` at
// 0.01 (via evalGGX) with no denominator floor; pt-webgpu floors alpha=rough² at
// 1e-3 plus a 1e-6 denom floor. They produce different low-roughness specular
// (rough=0.02 → a²≈1.6e-7 here vs 1e-6 in pt-webgpu); unifying would change
// rendering. See @vitrum/shared-samplers/wgsl/bsdfPrimitives.wgsl.ts for the
// reference (unfloored) form.
export const GGX_BRDF_WGSL = /* wgsl */ `// ============================================================
// GGX BRDF (simplified Lambertian + GGX specular)
// ============================================================

// GGX NDF
fn distributionGGX(NdotH: f32, rough: f32) -> f32 {
  let a = rough * rough;
  let a2 = a * a;
  let d = NdotH * NdotH * (a2 - 1.0) + 1.0;
  return a2 / (PI * d * d);
}

// Smith G1 (Schlick approximation)
fn geometrySchlickGGX(NdotV: f32, rough: f32) -> f32 {
  let r = rough + 1.0;
  let k = r * r / 8.0;
  return NdotV / (NdotV * (1.0 - k) + k);
}

fn geometrySmith(NdotV: f32, NdotL: f32, rough: f32) -> f32 {
  return geometrySchlickGGX(NdotV, rough) * geometrySchlickGGX(NdotL, rough);
}

fn materialF0(albedo: vec3f, metal: f32, specularColor: vec3f, specularIntensity: f32) -> vec3f {
  let dielectricF0 = vec3f(0.04) * clamp(specularColor, vec3f(0.0), vec3f(1.0)) * clamp(specularIntensity, 0.0, 1.0);
  return mix(dielectricF0, albedo, clamp(metal, 0.0, 1.0));
}

// KHR_materials_iridescence approximation for walkaround's shade-owned GGX
// evaluations. Iridescence modifies the existing base F0; it is not a separate
// sampled lobe in this backend, so ReSTIR candidate PDFs remain base-lobe only.
// Ported by convention from pt-webgpu's Belcour/Barla 2017 helper.
fn iridXyzToRec709(xyz: vec3f) -> vec3f {
  return vec3f(
     3.2404542 * xyz.x - 1.5371385 * xyz.y - 0.4985314 * xyz.z,
    -0.9692660 * xyz.x + 1.8760108 * xyz.y + 0.0415560 * xyz.z,
     0.0556434 * xyz.x - 0.2040259 * xyz.y + 1.0572252 * xyz.z,
  );
}

fn iridFresnel0ToIor(f0: vec3f) -> vec3f {
  let sqrtF0 = sqrt(clamp(f0, vec3f(0.0), vec3f(0.9999)));
  return (vec3f(1.0) + sqrtF0) / (vec3f(1.0) - sqrtF0);
}

fn iridIorToFresnel0Scalar(transmittedIor: f32, incidentIor: f32) -> f32 {
  let r = (transmittedIor - incidentIor) / (transmittedIor + incidentIor);
  return r * r;
}

fn iridIorToFresnel0Vec(transmittedIor: vec3f, incidentIor: f32) -> vec3f {
  let r = (transmittedIor - vec3f(incidentIor)) / (transmittedIor + vec3f(incidentIor));
  return r * r;
}

fn iridSchlickScalar(cosTheta: f32, f0: f32) -> f32 {
  let m = clamp(1.0 - cosTheta, 0.0, 1.0);
  let m2 = m * m;
  return f0 + (1.0 - f0) * m2 * m2 * m;
}

fn iridSchlickVec(cosTheta: f32, f0: vec3f) -> vec3f {
  let m = clamp(1.0 - cosTheta, 0.0, 1.0);
  let m2 = m * m;
  return f0 + (vec3f(1.0) - f0) * m2 * m2 * m;
}

fn iridEvalSensitivity(OPD: f32, shift: vec3f) -> vec3f {
  let phase = 2.0 * PI * OPD * 1.0e-9;
  let val = vec3f(5.4856e-13, 4.4201e-13, 5.2481e-13);
  let pos = vec3f(1.6810e+06, 1.7953e+06, 2.2084e+06);
  let vari = vec3f(4.3278e+09, 9.3046e+09, 6.6121e+09);
  var xyz = val * sqrt(2.0 * PI * vari) * cos(pos * phase + shift) * exp(-phase * phase * vari);
  xyz.x = xyz.x + 9.7470e-14 * sqrt(2.0 * PI * 4.5282e+09)
      * cos(2.2399e+06 * phase + shift.x) * exp(-4.5282e+09 * phase * phase);
  xyz = xyz / 1.0685e-7;
  return iridXyzToRec709(xyz);
}

fn evalIridescence(
  outsideIOR: f32,
  eta2: f32,
  cosTheta1: f32,
  thicknessNm: f32,
  baseF0: vec3f,
) -> vec3f {
  let iridescenceIor = mix(outsideIOR, eta2, smoothstep(0.0, 0.03, thicknessNm));
  let sinTheta2Sq = (outsideIOR / iridescenceIor) * (outsideIOR / iridescenceIor)
      * max(0.0, 1.0 - cosTheta1 * cosTheta1);
  let cosTheta2Sq = 1.0 - sinTheta2Sq;
  if (cosTheta2Sq < 0.0) { return vec3f(1.0); }
  let cosTheta2 = sqrt(cosTheta2Sq);

  let R0_scalar = iridIorToFresnel0Scalar(iridescenceIor, outsideIOR);
  let R12 = iridSchlickScalar(cosTheta1, R0_scalar);
  let T121 = 1.0 - R12;
  let phi12 = select(0.0, PI, iridescenceIor < outsideIOR);
  let phi21 = PI - phi12;

  let baseIOR = iridFresnel0ToIor(clamp(baseF0, vec3f(0.0), vec3f(0.9999)));
  let R1_vec = iridIorToFresnel0Vec(baseIOR, iridescenceIor);
  let R23 = iridSchlickVec(cosTheta2, R1_vec);
  var phi23 = vec3f(0.0);
  phi23.x = select(0.0, PI, baseIOR.x < iridescenceIor);
  phi23.y = select(0.0, PI, baseIOR.y < iridescenceIor);
  phi23.z = select(0.0, PI, baseIOR.z < iridescenceIor);

  let OPD = 2.0 * iridescenceIor * thicknessNm * cosTheta2;
  let phi = vec3f(phi21) + phi23;
  let R123 = clamp(R12 * R23, vec3f(1e-5), vec3f(0.9999));
  let r123 = sqrt(R123);
  let Rs = (T121 * T121) * R23 / (vec3f(1.0) - R123);

  let C0 = vec3f(R12) + Rs;
  var I = C0;
  var Cm = Rs - vec3f(T121);
  for (var m = 1; m <= 2; m = m + 1) {
    Cm = Cm * r123;
    let Sm = 2.0 * iridEvalSensitivity(f32(m) * OPD, f32(m) * phi);
    I = I + Cm * Sm;
  }
  return max(I, vec3f(0.0));
}

fn iridescenceModifiedF0(baseF0: vec3f, iridescence: vec4f, cosTheta: f32) -> vec3f {
  let factor = clamp(iridescence.x, 0.0, 1.0);
  if (factor < 1e-4) {
    return baseF0;
  }
  let thicknessNm = mix(max(0.0, iridescence.z), max(0.0, iridescence.w), clamp(cosTheta, 0.0, 1.0));
  let iridF = evalIridescence(1.0, max(1.0, iridescence.y), cosTheta, thicknessNm, baseF0);
  return mix(baseF0, iridF, factor);
}

fn anisotropyAxes(rough: f32, anisotropy: f32) -> vec2f {
  let alpha = max(rough * rough, 1e-4);
  let aspect = sqrt(max(1.0 - 0.9 * clamp(anisotropy, 0.0, 1.0), 1e-4));
  return vec2f(max(alpha / aspect, 1e-4), max(alpha * aspect, 1e-4));
}

fn anisotropyTangentFrame(n: vec3f, rotation: f32) -> mat3x3f {
  var up = vec3f(0.0, 1.0, 0.0);
  if (abs(n.y) > 0.999) { up = vec3f(1.0, 0.0, 0.0); }
  let t0 = normalize(cross(up, n));
  let b0 = cross(n, t0);
  let c = cos(rotation);
  let s = sin(rotation);
  let t = normalize(t0 * c + b0 * s);
  let b = normalize(-t0 * s + b0 * c);
  return mat3x3f(t, b, n);
}

fn anisotropyTangentFrameFromBasis(n: vec3f, tangentBasis: vec3f, bitangentBasis: vec3f, rotation: f32) -> mat3x3f {
  var t0 = tangentBasis - n * dot(n, tangentBasis);
  let tLen2 = dot(t0, t0);
  if (tLen2 <= 1e-8) {
    return anisotropyTangentFrame(n, rotation);
  }
  t0 = t0 * inverseSqrt(tLen2);

  var b0 = bitangentBasis - n * dot(n, bitangentBasis) - t0 * dot(t0, bitangentBasis);
  let bLen2 = dot(b0, b0);
  if (bLen2 <= 1e-8) {
    b0 = normalize(cross(n, t0));
  } else {
    b0 = b0 * inverseSqrt(bLen2);
  }

  let c = cos(rotation);
  let s = sin(rotation);
  let t = normalize(t0 * c + b0 * s);
  let b = normalize(-t0 * s + b0 * c);
  return mat3x3f(t, b, n);
}

fn distributionGGXAnisotropic(n: vec3f, t: vec3f, b: vec3f, h: vec3f, alphaX: f32, alphaY: f32) -> f32 {
  let NdotH = max(0.0, dot(n, h));
  let TdotH = dot(t, h);
  let BdotH = dot(b, h);
  let denom = (TdotH / alphaX) * (TdotH / alphaX) +
              (BdotH / alphaY) * (BdotH / alphaY) +
              NdotH * NdotH;
  return 1.0 / max(PI * alphaX * alphaY * denom * denom, 1e-8);
}

fn geometrySmithGGXAnisotropicG1(n: vec3f, t: vec3f, b: vec3f, v: vec3f, alphaX: f32, alphaY: f32) -> f32 {
  let NdotV = max(1e-4, dot(n, v));
  let TdotV = dot(t, v);
  let BdotV = dot(b, v);
  let root = sqrt(alphaX * alphaX * TdotV * TdotV + alphaY * alphaY * BdotV * BdotV + NdotV * NdotV);
  return (2.0 * NdotV) / max(NdotV + root, 1e-6);
}

fn geometrySmithGGXAnisotropic(n: vec3f, t: vec3f, b: vec3f, wo: vec3f, wi: vec3f, alphaX: f32, alphaY: f32) -> f32 {
  return geometrySmithGGXAnisotropicG1(n, t, b, wo, alphaX, alphaY) *
         geometrySmithGGXAnisotropicG1(n, t, b, wi, alphaX, alphaY);
}

// Evaluate GGX BRDF (diffuse + specular).
// albedo: base color, rough: roughness, metalness baked into F0.
fn evalGGXWithSpecular(
  albedo: vec3f,
  rough: f32,
  metal: f32,
  specularColor: vec3f,
  specularIntensity: f32,
  n: vec3f,
  wo: vec3f,
  wi: vec3f,
) -> vec3f {
  let h = safe_normalize(wo + wi);
  let NdotL = max(0.0, dot(n, wi));
  let NdotV = max(1e-4, dot(n, wo));
  let NdotH = max(0.0, dot(n, h));
  let VdotH = max(0.0, dot(wo, h));
  if (NdotL < 1e-6 || NdotV < 1e-6) { return vec3f(0.0); }

  let F0 = materialF0(albedo, metal, specularColor, specularIntensity);
  let F   = fresnelSchlick(VdotH, F0);
  let D   = distributionGGX(NdotH, max(0.01, rough));
  let G   = geometrySmith(NdotV, NdotL, max(0.01, rough));

  let specular = (D * G * F) / (4.0 * NdotV * NdotL);
  let diffuse  = (1.0 - F) * (1.0 - metal) * albedo * INV_PI;
  return (diffuse + specular) * NdotL;
}

fn evalGGXWithSpecularAnisotropy(
  albedo: vec3f,
  rough: f32,
  metal: f32,
  specularColor: vec3f,
  specularIntensity: f32,
  anisotropy: f32,
  anisotropyRotation: f32,
  iridescence: vec4f,
  n: vec3f,
  wo: vec3f,
  wi: vec3f,
) -> vec3f {
  let frame = anisotropyTangentFrame(n, 0.0);
  return evalGGXWithSpecularAnisotropyFrame(
    albedo,
    rough,
    metal,
    specularColor,
    specularIntensity,
    anisotropy,
    anisotropyRotation,
    iridescence,
    frame[0],
    frame[1],
    n,
    wo,
    wi,
  );
}

fn evalGGXWithSpecularAnisotropyFrame(
  albedo: vec3f,
  rough: f32,
  metal: f32,
  specularColor: vec3f,
  specularIntensity: f32,
  anisotropy: f32,
  anisotropyRotation: f32,
  iridescence: vec4f,
  anisotropyTangent: vec3f,
  anisotropyBitangent: vec3f,
  n: vec3f,
  wo: vec3f,
  wi: vec3f,
) -> vec3f {
  let aniso = clamp(anisotropy, 0.0, 1.0);
  if (aniso <= 1e-4 && iridescence.x <= 1e-4) {
    return evalGGXWithSpecular(albedo, rough, metal, specularColor, specularIntensity, n, wo, wi);
  }

  let h = safe_normalize(wo + wi);
  let NdotL = max(0.0, dot(n, wi));
  let NdotV = max(1e-4, dot(n, wo));
  let NdotH = max(0.0, dot(n, h));
  let VdotH = max(0.0, dot(wo, h));
  if (NdotL < 1e-6 || NdotV < 1e-6) { return vec3f(0.0); }

  let F0 = iridescenceModifiedF0(materialF0(albedo, metal, specularColor, specularIntensity), iridescence, VdotH);
  let F = fresnelSchlick(VdotH, F0);
  var D: f32;
  var G: f32;
  if (aniso <= 1e-4) {
    D = distributionGGX(NdotH, max(0.01, rough));
    G = geometrySmith(NdotV, NdotL, max(0.01, rough));
  } else {
    let frame = anisotropyTangentFrameFromBasis(n, anisotropyTangent, anisotropyBitangent, anisotropyRotation);
    let axes = anisotropyAxes(max(0.01, rough), aniso);
    D = distributionGGXAnisotropic(n, frame[0], frame[1], h, axes.x, axes.y);
    G = geometrySmithGGXAnisotropic(n, frame[0], frame[1], wo, wi, axes.x, axes.y);
  }

  let specular = (D * G * F) / (4.0 * NdotV * NdotL);
  let diffuse = (1.0 - F) * (1.0 - metal) * albedo * INV_PI;
  return (diffuse + specular) * NdotL;
}

fn evalGGX(albedo: vec3f, rough: f32, metal: f32, n: vec3f, wo: vec3f, wi: vec3f) -> vec3f {
  return evalGGXWithSpecular(albedo, rough, metal, vec3f(1.0), 1.0, n, wo, wi);
}

fn evalClearcoatLobe(clearcoat: f32, clearcoatRoughness: f32, n: vec3f, wo: vec3f, wi: vec3f) -> vec3f {
  let cc = clamp(clearcoat, 0.0, 1.0);
  if (cc < 1e-4) { return vec3f(0.0); }
  let h = safe_normalize(wo + wi);
  let NdotL = max(0.0, dot(n, wi));
  let NdotV = max(1e-4, dot(n, wo));
  let NdotH = max(0.0, dot(n, h));
  let VdotH = max(0.0, dot(wo, h));
  if (NdotL < 1e-6 || NdotV < 1e-6) { return vec3f(0.0); }

  // KHR_materials_clearcoat uses a dielectric top coat at IOR 1.5 (F0 ≈ 0.04).
  let rough = max(0.01, clamp(clearcoatRoughness, 0.0, 1.0));
  let F = fresnelSchlick(VdotH, vec3f(0.04));
  let D = distributionGGX(NdotH, rough);
  let G = geometrySmith(NdotV, NdotL, rough);
  let specular = (D * G * F) / (4.0 * NdotV * NdotL);
  return cc * specular * NdotL;
}

// KHR_materials_sheen: Charlie distribution plus Neubelt-Pettineo visibility.
// This returns the full lobe contribution including NdotL, matching evalGGX().
fn charlieD(nDotH: f32, alpha: f32) -> f32 {
  let invAlpha = 1.0 / max(alpha, 1e-4);
  let sinThetaH = sqrt(max(0.0, 1.0 - nDotH * nDotH));
  return (2.0 + invAlpha) * pow(sinThetaH, invAlpha) / (2.0 * PI);
}

fn sheenVisibility(nDotL: f32, nDotV: f32) -> f32 {
  return 1.0 / max(4.0 * (nDotL + nDotV - nDotL * nDotV), 1e-6);
}

fn evalSheenLobe(sheen: f32, sheenRoughness: f32, sheenColor: vec3f, n: vec3f, wo: vec3f, wi: vec3f) -> vec3f {
  let sh = clamp(sheen, 0.0, 1.0);
  if (sh < 1e-4) { return vec3f(0.0); }
  let NdotL = max(0.0, dot(n, wi));
  let NdotV = max(0.0, dot(n, wo));
  if (NdotL < 1e-6 || NdotV < 1e-6) { return vec3f(0.0); }
  let h = safe_normalize(wo + wi);
  let NdotH = max(0.0, dot(n, h));
  let alpha = max(clamp(sheenRoughness, 0.0, 1.0) * clamp(sheenRoughness, 0.0, 1.0), 1e-3);
  let D = charlieD(NdotH, alpha);
  let V = sheenVisibility(NdotL, NdotV);
  return sh * clamp(sheenColor, vec3f(0.0), vec3f(1.0)) * D * V * NdotL;
}

fn evalGGXWithSpecularClearcoatSheen(
  albedo: vec3f,
  rough: f32,
  metal: f32,
  specularColor: vec3f,
  specularIntensity: f32,
  anisotropy: f32,
  anisotropyRotation: f32,
  iridescence: vec4f,
  clearcoat: f32,
  clearcoatRoughness: f32,
  sheen: f32,
  sheenRoughness: f32,
  sheenColor: vec3f,
  n: vec3f,
  clearcoatNormal: vec3f,
  wo: vec3f,
  wi: vec3f,
) -> vec3f {
  let frame = anisotropyTangentFrame(n, 0.0);
  return evalGGXWithSpecularClearcoatSheenWithAnisotropyFrame(
    albedo,
    rough,
    metal,
    specularColor,
    specularIntensity,
    anisotropy,
    anisotropyRotation,
    iridescence,
    clearcoat,
    clearcoatRoughness,
    sheen,
    sheenRoughness,
    sheenColor,
    frame[0],
    frame[1],
    n,
    clearcoatNormal,
    wo,
    wi,
  );
}

fn evalGGXWithSpecularClearcoatSheenWithAnisotropyFrame(
  albedo: vec3f,
  rough: f32,
  metal: f32,
  specularColor: vec3f,
  specularIntensity: f32,
  anisotropy: f32,
  anisotropyRotation: f32,
  iridescence: vec4f,
  clearcoat: f32,
  clearcoatRoughness: f32,
  sheen: f32,
  sheenRoughness: f32,
  sheenColor: vec3f,
  anisotropyTangent: vec3f,
  anisotropyBitangent: vec3f,
  n: vec3f,
  clearcoatNormal: vec3f,
  wo: vec3f,
  wi: vec3f,
) -> vec3f {
  return evalGGXWithSpecularAnisotropyFrame(albedo, rough, metal, specularColor, specularIntensity, anisotropy, anisotropyRotation, iridescence, anisotropyTangent, anisotropyBitangent, n, wo, wi)
       + evalClearcoatLobe(clearcoat, clearcoatRoughness, clearcoatNormal, wo, wi)
       + evalSheenLobe(sheen, sheenRoughness, sheenColor, n, wo, wi);
}

// ── B9 (road-to-100) — Kulla-Conty multiple-scattering energy compensation ───
//
// Single-scattering GGX (the D·G·F lobe above) loses energy at high roughness
// because the Smith masking-shadowing term only accounts for the FIRST microfacet
// bounce — light that scatters between facets and re-emerges is dropped. The loss
// is severe for rough metals (a rough=1 conductor furnace test reads ~50% dark).
// Kulla & Conty 2017 ("Revisiting Physically Based Shading at Imageworks", SIGGRAPH
// Course) restore it with a multiscatter lobe whose magnitude is driven by the
// directional albedo E(μ) and average albedo Eavg of the single-scatter lobe.
//
// We use the analytic Eavg/E fit (Karis / Kulla-Conty white-furnace closed form)
// rather than a precomputed LUT — naga has no texture dependency, the same fit is
// the CPU furnace-test mirror, and it matches the pt-webgpu B9 lobe by CONVENTION
// (both packages use the directional-albedo-driven compensation, not a shared LUT).
//
//   E(μ,α)    ≈ directional albedo of the GGX-Smith single-scatter lobe.
//   Eavg(α)   = 2·∫₀¹ E(μ,α)·μ dμ  (cosine-weighted hemispherical average).
//
// The compensation BRDF (Kulla-Conty Eq. for the multiscatter lobe) is
//   f_ms(μo,μi) = (1 − E(μo))·(1 − E(μi)) / (π·(1 − Eavg))
// tinted by the Fresnel-average colour factor F_ms so conductors keep their hue
// across the extra bounces:
//   F_ms = Favg² · Eavg / (1 − Favg·(1 − Eavg))     (Kulla-Conty colour series)
// with Favg the hemispherical Fresnel average ≈ F0 + (1 − F0)/21 (Karis).
//
// At low roughness E→1 and Eavg→1, so f_ms → 0 — the compensation vanishes and the
// lobe is the pure single-scatter form. This is APPLIED ONLY in evalGGXSpecularOnly
// (the glossy/metal GI path, already gated to non-default surfaces by shade's
// SPEC_GI_ROUGH_MAX), so default-diffuse + direct-light evalGGX stay byte-identical.

// Analytic GGX-Smith directional albedo fit E(μ,α). Rational fit to the white-
// furnace single-scatter integral; monotone, E(μ,0)=1, E→ small at μ→0 & α→1.
fn ggxDirectionalAlbedo(mu: f32, rough: f32) -> f32 {
  let a = clamp(rough, 0.0, 1.0);
  let c = clamp(mu, 0.0, 1.0);
  // Karis-style biased fit: the single-scatter lobe retains ~ (1 − a·a·(1−c))
  // of grazing energy; tuned so the furnace residual (1 − E − ∫f_ms) → 0.
  let a2 = a * a;
  return clamp(1.0 - a2 * (1.0 - c) * (0.75 + 0.25 * c), 0.0, 1.0);
}

// Eavg(α) = 2·∫₀¹ E(μ,α)·μ dμ for the fit above (closed form of the integral of
// ggxDirectionalAlbedo against the cosine measure). Used to normalise f_ms.
fn ggxAverageAlbedo(rough: f32) -> f32 {
  let a = clamp(rough, 0.0, 1.0);
  let a2 = a * a;
  // ∫₀¹ 2μ·(1 − a²(1−μ)(0.75+0.25μ)) dμ = 1 − a²·(0.75·(1/3) + ...) → collapse:
  //   = 1 − a²·(7/24)   (exact integral of the fit; keeps Eavg∈[0,1]).
  return clamp(1.0 - a2 * (7.0 / 24.0), 0.0, 1.0);
}

// Hemispherical Fresnel average (Karis approximation, colour-preserving).
fn fresnelAverage(F0: vec3f) -> vec3f {
  return F0 + (vec3f(1.0) - F0) * (1.0 / 21.0);
}

// Kulla-Conty multiscatter lobe contribution (NOT × NdotL — the caller folds the
// cosine in). Returns vec3f(0) at low roughness (Eavg→1) so the single-scatter
// path is unchanged. Eq. refs in the block comment above.
fn ggxMultiscatter(F0: vec3f, rough: f32, NdotV: f32, NdotL: f32) -> vec3f {
  let Eo = ggxDirectionalAlbedo(NdotV, rough);
  let Ei = ggxDirectionalAlbedo(NdotL, rough);
  let Eavg = ggxAverageAlbedo(rough);
  let oneMinusEavg = 1.0 - Eavg;
  if (oneMinusEavg < 1e-4) { return vec3f(0.0); }       // low-roughness: no comp.
  let fms = ((1.0 - Eo) * (1.0 - Ei)) / (PI * oneMinusEavg);
  let Favg = fresnelAverage(F0);
  // Colour-series multiscatter Fresnel: keeps conductor hue over extra bounces.
  let Fms = (Favg * Favg * Eavg) / (vec3f(1.0) - Favg * oneMinusEavg);
  return Fms * fms;
}

// B1 (road-to-100) — GGX SPECULAR lobe ONLY (no diffuse term), full energy
// including NdotL and the metal/dielectric Fresnel F0. Used by the glossy/metal
// GI path: the diffuse-indirect channel stays albedo-demodulated (Schied 2017),
// so the specular lobe is evaluated separately here and added to the
// UN-demodulated direct channel (it is NOT proportional to the diffuse albedo,
// so it must bypass the indirectCombine albedo re-modulation). F0 = mix(0.04,
// albedo, metal) — conductor reflectance tint comes from baseColor when metal.
//
// B9 (road-to-100) — the Kulla-Conty multiscatter lobe (ggxMultiscatter) is ADDED
// to the single-scatter lobe here so rough metals/glossy surfaces re-gain the
// inter-facet energy the Smith G term drops. This is the glossy/metal GI path
// only (shade gates it to non-default surfaces via SPEC_GI_ROUGH_MAX), so
// default-diffuse + the direct-light evalGGX above remain byte-identical.
fn evalGGXSpecularOnlyWithSpecular(
  albedo: vec3f,
  rough: f32,
  metal: f32,
  specularColor: vec3f,
  specularIntensity: f32,
  n: vec3f,
  wo: vec3f,
  wi: vec3f,
) -> vec3f {
  let h = safe_normalize(wo + wi);
  let NdotL = max(0.0, dot(n, wi));
  let NdotV = max(1e-4, dot(n, wo));
  let NdotH = max(0.0, dot(n, h));
  let VdotH = max(0.0, dot(wo, h));
  if (NdotL < 1e-6 || NdotV < 1e-6) { return vec3f(0.0); }
  let F0 = materialF0(albedo, metal, specularColor, specularIntensity);
  let F  = fresnelSchlick(VdotH, F0);
  let D  = distributionGGX(NdotH, max(0.01, rough));
  let G  = geometrySmith(NdotV, NdotL, max(0.01, rough));
  let specular = (D * G * F) / (4.0 * NdotV * NdotL);
  let ms = ggxMultiscatter(F0, max(0.01, rough), NdotV, NdotL);
  return (specular + ms) * NdotL;
}

fn evalGGXSpecularOnlyWithSpecularAnisotropy(
  albedo: vec3f,
  rough: f32,
  metal: f32,
  specularColor: vec3f,
  specularIntensity: f32,
  anisotropy: f32,
  anisotropyRotation: f32,
  iridescence: vec4f,
  n: vec3f,
  wo: vec3f,
  wi: vec3f,
) -> vec3f {
  let frame = anisotropyTangentFrame(n, 0.0);
  return evalGGXSpecularOnlyWithSpecularAnisotropyFrame(
    albedo,
    rough,
    metal,
    specularColor,
    specularIntensity,
    anisotropy,
    anisotropyRotation,
    iridescence,
    frame[0],
    frame[1],
    n,
    wo,
    wi,
  );
}

fn evalGGXSpecularOnlyWithSpecularAnisotropyFrame(
  albedo: vec3f,
  rough: f32,
  metal: f32,
  specularColor: vec3f,
  specularIntensity: f32,
  anisotropy: f32,
  anisotropyRotation: f32,
  iridescence: vec4f,
  anisotropyTangent: vec3f,
  anisotropyBitangent: vec3f,
  n: vec3f,
  wo: vec3f,
  wi: vec3f,
) -> vec3f {
  let aniso = clamp(anisotropy, 0.0, 1.0);
  if (aniso <= 1e-4 && iridescence.x <= 1e-4) {
    return evalGGXSpecularOnlyWithSpecular(albedo, rough, metal, specularColor, specularIntensity, n, wo, wi);
  }

  let h = safe_normalize(wo + wi);
  let NdotL = max(0.0, dot(n, wi));
  let NdotV = max(1e-4, dot(n, wo));
  let NdotH = max(0.0, dot(n, h));
  let VdotH = max(0.0, dot(wo, h));
  if (NdotL < 1e-6 || NdotV < 1e-6) { return vec3f(0.0); }

  let F0 = iridescenceModifiedF0(materialF0(albedo, metal, specularColor, specularIntensity), iridescence, VdotH);
  let F = fresnelSchlick(VdotH, F0);
  var D: f32;
  var G: f32;
  if (aniso <= 1e-4) {
    D = distributionGGX(NdotH, max(0.01, rough));
    G = geometrySmith(NdotV, NdotL, max(0.01, rough));
  } else {
    let frame = anisotropyTangentFrameFromBasis(n, anisotropyTangent, anisotropyBitangent, anisotropyRotation);
    let axes = anisotropyAxes(max(0.01, rough), aniso);
    D = distributionGGXAnisotropic(n, frame[0], frame[1], h, axes.x, axes.y);
    G = geometrySmithGGXAnisotropic(n, frame[0], frame[1], wo, wi, axes.x, axes.y);
  }
  let specular = (D * G * F) / (4.0 * NdotV * NdotL);
  let ms = ggxMultiscatter(F0, max(0.01, rough), NdotV, NdotL);
  return (specular + ms) * NdotL;
}

fn evalGGXSpecularOnly(albedo: vec3f, rough: f32, metal: f32, n: vec3f, wo: vec3f, wi: vec3f) -> vec3f {
  return evalGGXSpecularOnlyWithSpecular(albedo, rough, metal, vec3f(1.0), 1.0, n, wo, wi);
}

fn evalGGXSpecularOnlyWithSpecularClearcoatSheen(
  albedo: vec3f,
  rough: f32,
  metal: f32,
  specularColor: vec3f,
  specularIntensity: f32,
  anisotropy: f32,
  anisotropyRotation: f32,
  iridescence: vec4f,
  clearcoat: f32,
  clearcoatRoughness: f32,
  sheen: f32,
  sheenRoughness: f32,
  sheenColor: vec3f,
  n: vec3f,
  clearcoatNormal: vec3f,
  wo: vec3f,
  wi: vec3f,
) -> vec3f {
  let frame = anisotropyTangentFrame(n, 0.0);
  return evalGGXSpecularOnlyWithSpecularClearcoatSheenWithAnisotropyFrame(
    albedo,
    rough,
    metal,
    specularColor,
    specularIntensity,
    anisotropy,
    anisotropyRotation,
    iridescence,
    clearcoat,
    clearcoatRoughness,
    sheen,
    sheenRoughness,
    sheenColor,
    frame[0],
    frame[1],
    n,
    clearcoatNormal,
    wo,
    wi,
  );
}

fn evalGGXSpecularOnlyWithSpecularClearcoatSheenWithAnisotropyFrame(
  albedo: vec3f,
  rough: f32,
  metal: f32,
  specularColor: vec3f,
  specularIntensity: f32,
  anisotropy: f32,
  anisotropyRotation: f32,
  iridescence: vec4f,
  clearcoat: f32,
  clearcoatRoughness: f32,
  sheen: f32,
  sheenRoughness: f32,
  sheenColor: vec3f,
  anisotropyTangent: vec3f,
  anisotropyBitangent: vec3f,
  n: vec3f,
  clearcoatNormal: vec3f,
  wo: vec3f,
  wi: vec3f,
) -> vec3f {
  return evalGGXSpecularOnlyWithSpecularAnisotropyFrame(albedo, rough, metal, specularColor, specularIntensity, anisotropy, anisotropyRotation, iridescence, anisotropyTangent, anisotropyBitangent, n, wo, wi)
       + evalClearcoatLobe(clearcoat, clearcoatRoughness, clearcoatNormal, wo, wi)
       + evalSheenLobe(sheen, sheenRoughness, sheenColor, n, wo, wi);
}

// ── B16 (road-to-100) — GGX VNDF importance sampler (Heitz 2018) ─────────────
//
// Samples a glossy reflection direction wi ∝ the GGX visible-normal distribution
// for the DI BRDF candidate (ris.wgsl M_BRDF loop). Returns the world-space wi.
// The matching solid-angle pdf is ggxVndfReflectionPdf below — they MUST agree
// (a sampler/pdf mismatch biases the RIS source-pdf bookkeeping).
//
// Ref: Heitz, E. "Sampling the GGX Distribution of Visible Normals." JCGT 7(4),
//      2018. Algorithm 1 (tangent-space VNDF half-vector), reflected about it.
//      Ported by CONVENTION from pt-webgpu bsdf.wgsl sampleGgxVndfTangent.
fn ggxBuildOnb(n: vec3f, t: ptr<function, vec3f>, b: ptr<function, vec3f>) {
  var up = vec3f(0.0, 1.0, 0.0);
  if (abs(n.y) > 0.999) { up = vec3f(1.0, 0.0, 0.0); }
  *t = normalize(cross(up, n));
  *b = cross(n, *t);
}

// Heitz 2018 Algorithm 1 — tangent-space VNDF half-vector (wo, N=+Z).
fn ggxSampleVndfTangent(wo: vec3f, alpha: f32, rng: ptr<function, u32>) -> vec3f {
  let Vh = safe_normalize(vec3f(alpha * wo.x, alpha * wo.y, wo.z));
  let lensq = Vh.x * Vh.x + Vh.y * Vh.y;
  let T1 = select(vec3f(1.0, 0.0, 0.0), vec3f(-Vh.y, Vh.x, 0.0) * inverseSqrt(lensq), lensq > 1e-10);
  let T2 = cross(Vh, T1);
  let u1 = rand_f32(rng);
  let u2 = rand_f32(rng);
  let r = sqrt(u1);
  let phi = 2.0 * PI * u2;
  let t1 = r * cos(phi);
  var t2 = r * sin(phi);
  let s = 0.5 * (1.0 + Vh.z);
  t2 = (1.0 - s) * sqrt(max(0.0, 1.0 - t1 * t1)) + s * t2;
  let Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0, 1.0 - t1 * t1 - t2 * t2)) * Vh;
  return safe_normalize(vec3f(alpha * Nh.x, alpha * Nh.y, max(1e-6, Nh.z)));
}

// Sample a world-space glossy reflection direction wi via VNDF. n = surface
// normal, wo = world view dir (toward camera). Returns wi (may point below the
// surface — the caller checks dot(n,wi) > 0).
//
// B16 (road-to-100) — alpha floor: alpha = max(rough², 1e-4). Matches
// ggxVndfReflectionPdf exactly so the RIS source-pdf bookkeeping is unbiased.
// (Prior code floored at 1e-3 here vs 1e-4 in the pdf — sampler/pdf mismatch.)
fn ggxSampleVndf(n: vec3f, wo: vec3f, rough: f32, rng: ptr<function, u32>) -> vec3f {
  let alpha = max(rough * rough, 1e-4);
  var t: vec3f; var b: vec3f;
  ggxBuildOnb(n, &t, &b);
  // wo into tangent space (N = +Z).
  let woT = vec3f(dot(wo, t), dot(wo, b), dot(wo, n));
  let hT = ggxSampleVndfTangent(woT, alpha, rng);
  let h = safe_normalize(hT.x * t + hT.y * b + hT.z * n);
  return reflect(-wo, h);
}

// Exact Smith G1 for GGX (Heitz 2014 / Walter 2007).
// G1(nv, a²) = 2·nv / (nv + sqrt(a² + (1 − a²)·nv²))
// Used ONLY in the VNDF density (below); the shading G term (geometrySmith)
// keeps the Schlick approximation intentionally — changing it would alter
// shading. The VNDF density REQUIRES the exact G1 (the sampler draws from the
// exact distribution; using the Schlick approx here biases source-pdf values).
fn smithG1GGX(nv: f32, a2: f32) -> f32 {
  return 2.0 * nv / (nv + sqrt(a2 + (1.0 - a2) * nv * nv));
}

// VNDF reflection PDF in SOLID-ANGLE measure (Heitz 2018 §3 Eq. 17 + reflection
// Jacobian): p(wi) = D(h)·G1(wo,α²) / (4·NdotV). MUST match ggxSampleVndf so
// the RIS source pdf is exact.
//
// B16 (road-to-100) — two fixes applied here:
//   1. Alpha floor: alpha² = max(rough², 1e-4) — matches ggxSampleVndf exactly.
//      (Prior code: a = max(0.01, rough); a² could reach 1e-4 correctly, but the
//      floor was on rough not alpha — now explicit and shared with the sampler.)
//   2. G1 form: smithG1GGX (exact Smith) replaces geometrySchlickGGX (Schlick
//      k=(r+1)²/8 shading approximation). The exact G1 is required because the
//      VNDF distribution is DEFINED in terms of the exact Smith G1 (Heitz 2014
//      Eq. 2). Using the Schlick form here made pdf(sample) mismatch the true
//      VNDF density — the RIS M-BRDF weight was subtly biased at all roughnesses.
fn ggxVndfReflectionPdf(n: vec3f, wo: vec3f, wi: vec3f, rough: f32) -> f32 {
  let h = safe_normalize(wo + wi);
  let NdotV = max(1e-4, dot(n, wo));
  let NdotH = max(0.0, dot(n, h));
  if (NdotH <= 0.0) { return 0.0; }
  let a2 = max(rough * rough, 1e-4);
  let a  = sqrt(a2);
  let D  = distributionGGX(NdotH, a);
  let g1 = smithG1GGX(NdotV, a2);
  return (D * g1) / max(4.0 * NdotV, 1e-6);
}

`;

/** T9-stepA — focused WGSL_MODULES entry split out of `common`. */
export const GGX_BRDF_MODULE: WgslModule = {
  name: "ggxBrdf",
  source: GGX_BRDF_WGSL,
  requires: [],
};
