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

// ============================================================
// Canonical ReSTIR-DI target distribution p̂.
//   Same body all three pre-refactor copies (computePHat /
//   computePHat_t / computePHat_s) used. Bitterli 2020 §4.3
//   MUST BE IDENTICAL — structurally enforced via this shared
//   declaration site.
// ============================================================
fn restir_di_compute_phat_from_surface(lid: u32, surf: PrimarySurface) -> f32 {
  if (!surf.hit) { return 0.0; }
  let e = emitters[lid];
  let centroid = (e.vA + e.vB + e.vC) / 3.0;
  let toL = centroid - surf.pos;
  let dist2 = dot(toL, toL);
  if (dist2 < 1e-8) { return 0.0; }
  let wi     = toL / sqrt(dist2);
  let nDotL  = max(0.0, dot(surf.normal, wi));
  let nlDotL = max(0.0, dot(-e.normal, wi));
  if (nDotL < 1e-6 || nlDotL < 1e-6) { return 0.0; }
  // evalGGX already multiplies by NdotL (receiver cosine); G is the emitter
  // geometry term only: cos(emitter) / dist² with the emitterDist2Floor
  // clamp applied consistently with shade.wgsl (sweep finding Bug 3).
  let G    = emitterGeometry(nlDotL, dist2, ubo.emitterDist2Floor);
  let brdf = evalGGX(surf.albedo, surf.rough, surf.metal, surf.normal, surf.wo, wi);
  return luminance(e.Le * brdf * G);
}
`;

/** W2-C7 — declarative include-graph entry for the canonical p̂. */
export const RESTIR_PHAT_MODULE: WgslModule = {
  name: 'restirPHat',
  source: RESTIR_PHAT_WGSL,
  // Depends on `common` for PrimarySurface, emitters, emitterGeometry, evalGGX,
  // luminance, and the ubo binding.
  requires: ['common'],
};
