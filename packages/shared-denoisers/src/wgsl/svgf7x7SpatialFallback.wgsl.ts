/**
 * svgf7x7SpatialFallback.wgsl.ts — Schied 2017 §4.3 spatial variance fallback.
 *
 * For newly-disoccluded pixels (history < SVGF_SPATIAL_FALLBACK_HISTORY_THRESHOLD),
 * temporal moment data is insufficient to estimate variance. This pass computes
 * a 7×7 cross-bilateral spatial variance estimate from the current frame's
 * noisy radiance. Luminance is the signal and current depth/normal buffers keep
 * unrelated surfaces out of the estimate.
 *
 * Bind group 0 layout (entry point: svgf7x7FallbackMain):
 *   binding 0 — texture_2d<f32>                        currColor
 *   binding 1 — texture_2d<u32>                        historyIn
 *   binding 2 — texture_2d<f32>                        varianceIn
 *   binding 3 — texture_storage_2d<rgba32float, write> varianceOut
 *   binding 4 — texture_2d<f32>                        currNormal
 *   binding 5 — texture_2d<f32>                        currDepth
 *
 * Reference:
 *   Schied et al., "Spatiotemporal Variance-Guided Filtering", HPG 2017 §4.3.
 */

import { LUMINANCE_WGSL } from '@vitrum/shared-samplers';
import { SVGF_HISTORY_MIN_FOR_MOMENTS } from './svgfVarianceFromMoments.wgsl.js';
import {
  STANDALONE_DEPTH_TEXTURE_LAYOUT,
  normalDepthWgslDepthComponent,
  type NormalDepthTextureLayout,
} from '../normalDepthEncoding.js';

/** Pixels with history below this threshold use the 7×7 spatial variance estimate. */
export const SVGF_SPATIAL_FALLBACK_HISTORY_THRESHOLD = SVGF_HISTORY_MIN_FOR_MOMENTS;

/** Must match @workgroup_size in svgf7x7FallbackMain. */
export const SVGF_7X7_SPATIAL_FALLBACK_WORKGROUP_SIZE = 16 as const;

export function buildSvgf7x7SpatialFallbackWgsl(
  depthLayout: NormalDepthTextureLayout = STANDALONE_DEPTH_TEXTURE_LAYOUT,
): string {
  const depthComponent = normalDepthWgslDepthComponent(depthLayout);
  return /* wgsl */ `
${LUMINANCE_WGSL}
const SVGF_SPATIAL_THRESHOLD: u32 = ${SVGF_SPATIAL_FALLBACK_HISTORY_THRESHOLD}u;
const SVGF_FALLBACK_NORMAL_EXPONENT: f32 = 128.0;
const SVGF_FALLBACK_RELATIVE_DEPTH_SIGMA: f32 = 0.1;

@group(0) @binding(0) var sfb_currColor:    texture_2d<f32>;
@group(0) @binding(1) var sfb_historyIn:    texture_2d<u32>;
@group(0) @binding(2) var sfb_varianceIn:   texture_2d<f32>;
@group(0) @binding(3) var sfb_varianceOut:  texture_storage_2d<rgba32float, write>;
@group(0) @binding(4) var sfb_currNormal:   texture_2d<f32>;
@group(0) @binding(5) var sfb_currDepth:    texture_2d<f32>;

@compute @workgroup_size(16, 16, 1)
fn svgf7x7FallbackMain(@builtin(global_invocation_id) gid: vec3u) {
  let dims = textureDimensions(sfb_currColor);
  if (any(gid.xy >= dims)) { return; }

  let history = textureLoad(sfb_historyIn, gid.xy, 0).r;
  if (history >= SVGF_SPATIAL_THRESHOLD) {
    textureStore(
      sfb_varianceOut,
      gid.xy,
      textureLoad(sfb_varianceIn, gid.xy, 0),
    );
    return;
  }

  let centerNormal = textureLoad(sfb_currNormal, gid.xy, 0).xyz * 2.0 - 1.0;
  let centerDepth = textureLoad(sfb_currDepth, gid.xy, 0).${depthComponent};
  var weightedLuminance = 0.0;
  var weightedLuminanceSquared = 0.0;
  var weightSum = 0.0;

  for (var dy: i32 = -3; dy <= 3; dy += 1) {
    for (var dx: i32 = -3; dx <= 3; dx += 1) {
      let p = vec2i(gid.xy) + vec2i(dx, dy);
      if (any(p < vec2i(0)) || any(vec2u(p) >= dims)) {
        continue;
      }
      let pu = vec2u(p);
      let sampleNormal = textureLoad(sfb_currNormal, pu, 0).xyz * 2.0 - 1.0;
      let sampleDepth = textureLoad(sfb_currDepth, pu, 0).${depthComponent};
      let normalWeight = pow(
        clamp(dot(centerNormal, sampleNormal), 0.0, 1.0),
        SVGF_FALLBACK_NORMAL_EXPONENT,
      );
      let depthScale = max(1e-3, max(abs(centerDepth), abs(sampleDepth)));
      let depthWeight = exp(
        -abs(sampleDepth - centerDepth)
        / (SVGF_FALLBACK_RELATIVE_DEPTH_SIGMA * depthScale + 1e-4),
      );
      let weight = normalWeight * depthWeight;
      let sampleLuminance = luminance(textureLoad(sfb_currColor, pu, 0).rgb);
      weightedLuminance += weight * sampleLuminance;
      weightedLuminanceSquared += weight * sampleLuminance * sampleLuminance;
      weightSum += weight;
    }
  }

  var spatialVariance = 0.0;
  if (weightSum > 1e-6) {
    let mean = weightedLuminance / weightSum;
    spatialVariance = max(
      0.0,
      weightedLuminanceSquared / weightSum - mean * mean,
    );
  }

  let historyAsFloat = textureLoad(sfb_varianceIn, gid.xy, 0).g;
  textureStore(
    sfb_varianceOut,
    gid.xy,
    vec4f(spatialVariance, historyAsFloat, 0.0, 0.0),
  );
}
`;
}

/** Standalone one-shot ABI: depth is supplied by a dedicated R texture. */
export const SVGF_7X7_SPATIAL_FALLBACK_WGSL = buildSvgf7x7SpatialFallbackWgsl();
