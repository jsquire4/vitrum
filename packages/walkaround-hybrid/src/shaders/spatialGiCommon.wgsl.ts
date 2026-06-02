/**
 * spatialGiCommon.wgsl.ts — helpers shared verbatim by BOTH spatial-GI
 * bodies (the OFF/default `SPATIAL_GI_WGSL` and the GRIS opt-in
 * `SPATIAL_GI_GRIS_WGSL`).
 *
 * Holds the candidate-count constants (`K_SPATIAL_GI`, `M_CLAMP_SPATIAL`)
 * and the disc-sample helper `sampleDiscPx`.  Both bodies source this once
 * instead of inlining two byte-identical copies.
 *
 * Mirrors the `temporalGiCommon.wgsl.ts` pattern: the fragment begins exactly
 * where the inline copies began (the constant declarations) and ends at the
 * closing brace of `sampleDiscPx`, so each body interpolates it via the
 * `spatialGiCommon` requires dep with no surrounding-byte change.
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
  let r = ubo.restirGiSpatialRadiusPx * sqrt(rand_f32(rng));
  let phi = 6.2831853 * rand_f32(rng);
  return vec2f(r * cos(phi), r * sin(phi));
}
`;

/** Focused module for the shared spatial-GI constants + disc-sample helper.
 *  Both spatial-GI compile roots (OFF default + GRIS ON) require this so the
 *  declarations appear exactly once in each composed closure.
 *  `sampleDiscPx` forward-references `ubo` (walkaroundUbo) and `rand_f32`
 *  (sharedPrimitives) — both are always in every consumer's dep closure, so
 *  this module needs no `requires` of its own (mirrors the reservoirDi/reservoirGi
 *  pattern of forward-referencing without declaring the dep). */
export const SPATIAL_GI_COMMON_MODULE: WgslModule = {
  name: 'spatialGiCommon',
  source: SPATIAL_GI_COMMON_WGSL,
  requires: [],
};
