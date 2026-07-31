/**
 * spatialGiCommon.wgsl.ts — helpers used by the canonical spatial-GI pass.
 *
 * Holds the candidate-count constants (`K_SPATIAL_GI`, `M_CLAMP_SPATIAL`)
 * and the disc-sample helper `sampleDiscPx`.
 *
 * `sampleDiscPx` references `ubo.restirGiSpatialRadiusPx` (WalkaroundUBO)
 * and `rand_f32` (sharedPrimitives).  Both are already in every consumer's
 * dep closure via `walkaroundUbo` and `sharedPrimitives`.
 */

import type { WgslModule } from '../pipeline/wgslComposer.js';

export const SPATIAL_GI_COMMON_WGSL = /* wgsl */ `
const K_SPATIAL_GI: u32 = 5u;
const M_CLAMP_SPATIAL: u32 = 500u;
// SPATIAL_RADIUS_GI / NORMAL_DOT_MIN_S / COPLANAR_TOL_S now live on the
// WalkaroundUBO so library consumers can override the Cornell-tuned defaults:
//   ubo.restirGiSpatialRadiusPx         (default 12.0 — half-res pixels)
//   ubo.restirGiSpatialNormalDotMin     (default 0.906 ≈ cos(25°))
//   ubo.restirGiSpatialCoplanarTol      (default 0.05 — 5 cm world units)
//
// Coplanar-distance tolerance rationale: neighbour must lie within this
// perpendicular distance of the centre pixel's tangent plane.  Replaces the
// older camera-distance ratio test (DEPTH_REL_TOL_S) which rejected
// neighbours in corner geometry where the same wall recedes from the camera
// at a steep angle — verified via reservoir probe that the camera-ratio
// test was rejecting essentially all 5 neighbours on left-wall-near-back-
// corner pixels, locking each pixel into its own initial-RIS sample.  The
// plane test instead asks "are these points on the same surface" which is
// what the spatial filter actually needs.

fn sampleDiscPx(rng: ptr<function, u32>) -> vec2f {
  var radius = ubo.restirGiSpatialRadiusPx;
  if (restirReservoirScaleValue() > 1u) {
    radius = max(1.0, radius / f32(restirReservoirScaleValue()));
  }
  let r = radius * sqrt(rand_f32(rng));
  let phi = 6.2831853 * rand_f32(rng);
  return vec2f(r * cos(phi), r * sin(phi));
}
`;

/** Focused module for the shared spatial-GI constants + disc-sample helper.
 *  The canonical generalized-reuse root requires this so each declaration
 *  appears exactly once in the composed closure.
 *  `sampleDiscPx` forward-references `ubo` (walkaroundUbo) and `rand_f32`
 *  (sharedPrimitives) — both are always in every consumer's dep closure, so
 *  this module needs no `requires` of its own (mirrors the reservoirDi/reservoirGi
 *  pattern of forward-referencing without declaring the dep). */
export const SPATIAL_GI_COMMON_MODULE: WgslModule = {
  name: 'spatialGiCommon',
  source: SPATIAL_GI_COMMON_WGSL,
  requires: [],
};
