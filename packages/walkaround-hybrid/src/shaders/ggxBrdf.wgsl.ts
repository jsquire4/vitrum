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

fn evalGGXWithSpecularAndClearcoat(
  albedo: vec3f,
  rough: f32,
  metal: f32,
  specularColor: vec3f,
  specularIntensity: f32,
  clearcoat: f32,
  clearcoatRoughness: f32,
  n: vec3f,
  wo: vec3f,
  wi: vec3f,
) -> vec3f {
  return evalGGXWithSpecular(albedo, rough, metal, specularColor, specularIntensity, n, wo, wi)
       + evalClearcoatLobe(clearcoat, clearcoatRoughness, n, wo, wi);
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

fn evalGGXSpecularOnly(albedo: vec3f, rough: f32, metal: f32, n: vec3f, wo: vec3f, wi: vec3f) -> vec3f {
  return evalGGXSpecularOnlyWithSpecular(albedo, rough, metal, vec3f(1.0), 1.0, n, wo, wi);
}

fn evalGGXSpecularOnlyWithSpecularAndClearcoat(
  albedo: vec3f,
  rough: f32,
  metal: f32,
  specularColor: vec3f,
  specularIntensity: f32,
  clearcoat: f32,
  clearcoatRoughness: f32,
  n: vec3f,
  wo: vec3f,
  wi: vec3f,
) -> vec3f {
  return evalGGXSpecularOnlyWithSpecular(albedo, rough, metal, specularColor, specularIntensity, n, wo, wi)
       + evalClearcoatLobe(clearcoat, clearcoatRoughness, n, wo, wi);
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
