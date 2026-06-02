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

// Evaluate GGX BRDF (diffuse + specular).
// albedo: base color, rough: roughness, metalness baked into F0.
fn evalGGX(albedo: vec3f, rough: f32, metal: f32, n: vec3f, wo: vec3f, wi: vec3f) -> vec3f {
  let h = safe_normalize(wo + wi);
  let NdotL = max(0.0, dot(n, wi));
  let NdotV = max(1e-4, dot(n, wo));
  let NdotH = max(0.0, dot(n, h));
  let VdotH = max(0.0, dot(wo, h));
  if (NdotL < 1e-6 || NdotV < 1e-6) { return vec3f(0.0); }

  let F0 = mix(vec3f(0.04), albedo, metal);
  let F   = fresnelSchlick(VdotH, F0);
  let D   = distributionGGX(NdotH, max(0.01, rough));
  let G   = geometrySmith(NdotV, NdotL, max(0.01, rough));

  let specular = (D * G * F) / (4.0 * NdotV * NdotL);
  let diffuse  = (1.0 - F) * (1.0 - metal) * albedo * INV_PI;
  return (diffuse + specular) * NdotL;
}

`;

/** T9-stepA — focused WGSL_MODULES entry split out of `common`. */
export const GGX_BRDF_MODULE: WgslModule = {
  name: "ggxBrdf",
  source: GGX_BRDF_WGSL,
  requires: [],
};
