/**
 * Canonical ReSTIR-DI target distribution p̂.
 *
 * W2-C7 canonicalisation (premium-grade-refactor-20260517 §W2):
 *   Bitterli 2020 §4.3 mandates that the target function p̂ used during
 *   RIS selection, temporal reuse, and spatial reuse MUST be identical
 *   across all three passes. Pre-refactor this was hand-copied as three
 *   near-identical functions:
 *     ris.wgsl.ts:60-79      computePHat   (scalar args)
 *     temporal.wgsl.ts:85-102 computePHat_t (PrimarySurface arg)
 *     spatial.wgsl.ts:67-83   computePHat_s (PrimarySurface arg)
 *   Drift was prevented only by manual diff-discipline. This module
 *   structurally enforces the invariant by providing the single shared
 *   implementation that every consumer calls.
 *
 *  Inputs: `lid` = emitter index, `surf` = primary surface receiver
 *          struct (defined in common.wgsl).
 *  Output: scalar luminance of e.Le × evalGGX × emitterGeometry — the
 *          quantity Bitterli 2020 calls p̂(x) in eq. (3).
 *
 *  Citations: see CREDITS.md → ReSTIR / Bitterli 2020.
 */

import type { WgslModule } from '../pipeline/wgslComposer.js';

export const RESTIR_PHAT_WGSL = /* wgsl */ `

// Receiver-independent target used only when a coarse reservoir is shared by
// multiple full-resolution shading receivers. Its support depends solely on
// the sampled light/environment source; receiver BRDF, geometry, sidedness,
// depth, and visibility are re-evaluated by the final full-resolution consumer.
fn restir_di_coarse_proposal_phat(lid: u32, xi: vec2f) -> f32 {
  if (lid == ENV_SAMPLE_SENTINEL) {
    return max(0.0, luminance(envRadiance(envDirFromXi(xi))));
  }
  if (lid >= ubo.emitterCount) { return 0.0; }
  return max(0.0, luminance(sampleEmitterLeAtXi(sceneLoadEmitter(lid), xi)));
}

// ============================================================
// Canonical ReSTIR-DI target distribution p̂.
//   Same body all three pre-refactor copies (computePHat /
//   computePHat_t / computePHat_s) used. Bitterli 2020 §4.3
//   MUST BE IDENTICAL — structurally enforced via this shared
//   declaration site.
//
// Wave 4: renamed to restir_di_compute_phat_xi(lid, xi, surf) — takes the
// reservoir's stored xi so every candidate family re-evaluates the same sample
// it stored: emitter candidates use sampleEmitterPoint(e, xi), and
// ENV_SAMPLE_SENTINEL decodes xi back to the sampled HDRI direction. The old
// name is kept as a thin wrapper for any callers that haven't been updated
// (should be zero after Wave 4).
// ============================================================

fn restir_di_eval_surface_brdf(surf: PrimarySurface, wi: vec3f) -> vec3f {
  var brdf = evalGGXWithSpecularClearcoatSheenWithAnisotropyFrame(
    surf.albedo,
    surf.rough,
    surf.metal,
    surf.specular.rgb,
    surf.specular.a,
    surf.anisotropy.x,
    surf.anisotropy.y,
    surf.iridescence,
    surf.clearcoat.x,
    surf.clearcoat.y,
    surf.sheen.a,
    surf.sheenRoughness,
    surf.sheen.rgb,
    surf.anisotropyTangent,
    surf.anisotropyBitangent,
    surf.normal,
    surf.clearcoatNormal,
    surf.wo,
    wi,
  );
  // A transmissive dielectric still reflects its Fresnel/GGX lobe. Its base
  // color is absorption/transmission tint, not an opaque Lambertian lobe, so
  // direct-light target evaluation must retain only the specular family.
  if (surf.isGlass) {
    brdf = evalGGXSpecularOnlyWithSpecularClearcoatSheenWithAnisotropyFrame(
      surf.albedo,
      surf.rough,
      surf.metal,
      surf.specular.rgb,
      surf.specular.a,
      surf.anisotropy.x,
      surf.anisotropy.y,
      surf.iridescence,
      surf.clearcoat.x,
      surf.clearcoat.y,
      surf.sheen.a,
      surf.sheenRoughness,
      surf.sheen.rgb,
      surf.anisotropyTangent,
      surf.anisotropyBitangent,
      surf.normal,
      surf.clearcoatNormal,
      surf.wo,
      wi,
    );
  }
  brdf = surf.layerTransmission * brdf;
  return applyHomogeneousVolumeSingleScatterDirectional(
    brdf, surf.albedo, surf.volumeScattering, surf.bulkThickness,
    surf.normal, surf.wo, wi,
  );
}

// ENV branch: p̂ = luminance(envRadiance(dir) * full material BRDF(... dir)) — no geometry
// term (the IBL is at infinity: no cosθ_light, no 1/dist² falloff). This is the
// solid-angle measure p̂ consistent with the source pdf stored by the sampled
// environment candidate.
fn restir_di_compute_phat_xi(lid: u32, xi: vec2f, surf: PrimarySurface) -> f32 {
  if (!surf.hit) { return 0.0; }
  // Wave 4 — ENV_SAMPLE_SENTINEL: decode xi → world direction, evaluate env p̂.
  if (lid == ENV_SAMPLE_SENTINEL) {
    if (!envHasMap()) { return 0.0; }
    let wi = envDirFromXi(xi);
    let nDotL = max(0.0, dot(surf.normal, wi));
    if (nDotL <= 0.0) { return 0.0; }
    let color = walkaroundScaleEnvironmentRadiance(
      envRadiance(wi),
      surf.envMapIntensity,
    );
    let brdf  = restir_di_eval_surface_brdf(surf, wi);
    return luminance(color * brdf);
  }
  let e = sceneLoadEmitter(lid);
  let ls = sampleEmitterPoint(e, xi);
  let toL = ls.pos - surf.pos;
  let dist = safe_length(toL);
  if (!(dist > 0.0)) { return 0.0; }
  let dist2 = dist * dist;
  let wi = safe_normalize(toL);
  let nDotL  = max(0.0, dot(surf.normal, wi));
  let nlDotL = emitterTriCosineTowardReceiver(e, -wi);
  if (nDotL <= 0.0 || nlDotL <= 0.0) { return 0.0; }
  // evalGGX already multiplies by NdotL (receiver cosine); G is the emitter
  // geometry term only: cos(emitter) / dist² with the emitterDist2Floor
  // clamp applied consistently with shade.wgsl (sweep finding Bug 3).
  let G    = emitterGeometry(nlDotL, dist2, ubo.emitterDist2Floor);
  let brdf = restir_di_eval_surface_brdf(surf, wi);
  let Le = sampleEmitterLeAtXi(e, xi);
  return luminance(Le * brdf * G);
}

`;

/** W2-C7 — declarative include-graph entry for the canonical p̂.
 *  Wave 4: adds `environmentSample` to the requires chain so the
 *  ENV_SAMPLE_SENTINEL branch can call envHasMap() / envRadiance() /
 *  envDirFromXi(). `environmentSample` declares the scene-group bindings
 *  15-19; all passes that compose restirPHat already bind the full scene BGL
 *  (which includes those slots), so no bind-group layout change is needed. */
export const RESTIR_PHAT_MODULE: WgslModule = {
  name: 'restirPHat',
  source: RESTIR_PHAT_WGSL,
  // `common` for PrimarySurface, emitters, emitterGeometry, full BRDF helpers, luminance, ubo.
  // `emitterLeAtXi` for emissive-map texel Le on source-indexed mesh-material emitters.
  // `environmentSample` for ENV_SAMPLE_SENTINEL / envHasMap / envRadiance / envDirFromXi.
  requires: ['common', 'emitterLeAtXi', 'environmentSample'],
};
