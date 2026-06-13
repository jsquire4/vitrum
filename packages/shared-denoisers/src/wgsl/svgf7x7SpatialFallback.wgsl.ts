/**
 * svgf7x7SpatialFallback.wgsl.ts — Schied 2017 §4.3 spatial variance fallback.
 *
 * For newly-disoccluded pixels (history < SVGF_SPATIAL_FALLBACK_HISTORY_THRESHOLD),
 * temporal moment data is insufficient to estimate variance. This pass computes
 * a 7×7 cross-bilateral spatial variance estimate from the current frame's noisy
 * radiance, using luminance as the signal. The result is merged into the variance
 * texture so the à-trous chain reads it in place of the zero written by
 * svgfVarianceFromMomentsMain for those pixels.
 *
 * The 7×7 spatial variance uses a simple box kernel (no edge-stop); Schied §4.3
 * shows this is sufficient for history=0 pixels since they have no history
 * coherence to protect.
 *
 * Bind group 0 layout (entry point: svgf7x7FallbackMain):
 *   binding 0 — texture_2d<f32>                        currColor    (rgba16float noisy current frame)
 *   binding 1 — texture_2d<u32>                        historyIn    (r16uint history length)
 *   binding 2 — texture_2d<f32>                        varianceIn   (rg32float from svgfVarianceFromMomentsMain)
 *   binding 3 — texture_storage_2d<rgba32float, write> varianceOut  (merged output)
 *
 * References:
 *   Schied et al. HPG 2017 §4.3.
 */

import { LUMINANCE_WGSL } from '@vitrum/shared-samplers';
import { SVGF_HISTORY_MIN_FOR_MOMENTS } from './svgfVarianceFromMoments.wgsl.js';

/** Pixels with history below this threshold use the 7×7 spatial variance estimate. */
export const SVGF_SPATIAL_FALLBACK_HISTORY_THRESHOLD = SVGF_HISTORY_MIN_FOR_MOMENTS;

/** Must match @workgroup_size in svgf7x7FallbackMain. */
export const SVGF_7X7_SPATIAL_FALLBACK_WORKGROUP_SIZE = 16 as const;

export const SVGF_7X7_SPATIAL_FALLBACK_WGSL = /* wgsl */ `
${LUMINANCE_WGSL}
const SVGF_SPATIAL_THRESHOLD: u32 = ${SVGF_SPATIAL_FALLBACK_HISTORY_THRESHOLD}u;

// luminance(c) is the canonical Rec.709 helper from LUMINANCE_WGSL above.

@group(0) @binding(0) var sfb_currColor:    texture_2d<f32>;
@group(0) @binding(1) var sfb_historyIn:    texture_2d<u32>;
@group(0) @binding(2) var sfb_varianceIn:   texture_2d<f32>;
@group(0) @binding(3) var sfb_varianceOut:  texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(16, 16, 1)
fn svgf7x7FallbackMain(@builtin(global_invocation_id) gid: vec3u) {
  let dims = textureDimensions(sfb_currColor);
  if (any(gid.xy >= dims)) { return; }

  let h = textureLoad(sfb_historyIn, gid.xy, 0).r;

  // Pass through pixels with sufficient history — those already have
  // variance from the moments pass.
  if (h >= SVGF_SPATIAL_THRESHOLD) {
    let existing = textureLoad(sfb_varianceIn, gid.xy, 0);
    textureStore(sfb_varianceOut, gid.xy, existing);
    return;
  }

  // 7×7 box-filter spatial variance estimate (Schied §4.3).
  // Compute E[L] and E[L²] over a 7×7 neighborhood.
  var sumL  = 0.0;
  var sumL2 = 0.0;
  var n     = 0u;

  for (var dy: i32 = -3; dy <= 3; dy++) {
    for (var dx: i32 = -3; dx <= 3; dx++) {
      let p = vec2i(gid.xy) + vec2i(dx, dy);
      if (p.x < 0 || p.y < 0 || u32(p.x) >= dims.x || u32(p.y) >= dims.y) {
        continue;
      }
      let lum = luminance(textureLoad(sfb_currColor, vec2u(p), 0).rgb);
      sumL  += lum;
      sumL2 += lum * lum;
      n     += 1u;
    }
  }

  var spatialVariance = 0.0;
  if (n > 1u) {
    let mean = sumL / f32(n);
    // Biased population estimator: (ΣL²/n) − μ².
    spatialVariance = max(0.0, sumL2 / f32(n) - mean * mean);
  }

  // Write spatial fallback variance. Preserve .g (history) from the moments pass.
  let histG = textureLoad(sfb_varianceIn, gid.xy, 0).g;
  textureStore(sfb_varianceOut, gid.xy, vec4f(spatialVariance, histG, 0.0, 0.0));
}
`;
