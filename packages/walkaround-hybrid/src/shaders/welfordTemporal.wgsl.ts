/**
 * Temporal Welford pass — updates per-pixel running mean/M2 of shaded
 * luminance for SVGF's temporal variance branch (Sprint 9 Decision 13).
 *
 * Prepended with COMMON_WGSL for WelfordVariance + welfordUpdate.
 */

import type { WgslModule } from '../pipeline/wgslComposer.js';

export const WELFORD_TEMPORAL_WGSL = /* wgsl */ `

struct WelfordTemporalUBO {
  sampleN:     u32,   // 1-based frame index since camera/scene stable window
  forceReset:  u32,   // non-zero → treat prev state as (0,0)
  _pad0:       u32,
  _pad1:       u32,
};

const LUM_W: vec3f = vec3f(0.2126, 0.7152, 0.0722);
fn luminance_welford(c: vec3f) -> f32 {
  return dot(c, LUM_W);
}

@group(0) @binding(0) var w_hdr:  texture_2d<f32>;
@group(0) @binding(1) var w_prev: texture_2d<f32>;
@group(0) @binding(2) var w_out:  texture_storage_2d<rg32float, write>;
@group(0) @binding(3) var<uniform> w_ubo: WelfordTemporalUBO;

@compute @workgroup_size(16, 16, 1)
fn welfordTemporalMain(@builtin(global_invocation_id) gid: vec3u) {
  let dims = textureDimensions(w_hdr);
  if (any(gid.xy >= dims)) { return; }

  let lum = luminance_welford(textureLoad(w_hdr, gid.xy, 0).rgb);
  let raw = textureLoad(w_prev, gid.xy, 0);
  // WGSL select() only accepts scalar/vecN types — split the struct pick
  // into per-field selects to keep the same forceReset-zero-state semantic.
  // The prior single-call select(WelfordVariance, WelfordVariance, bool)
  // failed to compile on the WebGPU spec validator.
  let reset = w_ubo.forceReset != 0u;
  let prevMean = select(raw.r, 0.0, reset);
  let prevM2   = select(raw.g, 0.0, reset);
  let prevState = WelfordVariance(prevMean, prevM2);
  let n = max(1u, w_ubo.sampleN);
  let next = welfordUpdate(prevState, lum, n);
  textureStore(w_out, gid.xy, vec4f(next.mean, next.m2, 0.0, 0.0));
}
`;

/** W1-R6 — declarative include-graph entry. */
export const WELFORD_TEMPORAL_MODULE: WgslModule = {
  name: 'welfordTemporal',
  source: WELFORD_TEMPORAL_WGSL,
  requires: ['common'],
};
