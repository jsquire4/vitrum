/**
 * Schied 2017 SVGF variance-guided à-trous reconstruction pass.
 *
 * Unlike the package's standalone `atrousVariance` denoiser, this kernel
 * carries the luminance variance through every wavelet level.  Iteration zero
 * reads the full-precision variance estimate produced by the moments/fallback
 * passes.  Later iterations read the variance written into the previous color
 * level's alpha channel.  RGBA16F is intentional here: the real-time SVGF
 * signal already uses half-float ping-pong storage and the propagated variance
 * is clamped to the finite half-float range before storage.
 *
 * Schied et al. 2017 §4.3 requires two distinct variance operations:
 *   1. a 3×3 Gaussian prefilter used only to stabilize the luminance
 *      edge-stopping function; and
 *   2. squared-weight variance propagation to steer the next wavelet level.
 *
 * Bind group 0 (entry point `svgfRealAtrousMain`):
 *   0 — texture_2d<f32>                        color + prior variance in alpha
 *   1 — texture_storage_2d<rgba16float, write> filtered color + variance
 *   2 — texture_2d<f32>                        packed world normal
 *   3 — texture_2d<f32>                        linear depth
 *   4 — texture_2d<f32>                        initial full-precision variance
 *   5 — AtrousVarianceAtrousUBO
 *
 * Reference:
 *   Schied et al., "Spatiotemporal Variance-Guided Filtering: Real-Time
 *   Reconstruction for Path-Traced Global Illumination", HPG 2017, §4.3.
 */

import { LUMINANCE_WGSL } from '@vitrum/shared-samplers';
import { ATROUS_VARIANCE_KERNEL_WGSL } from './atrousKernel.wgsl.js';
import {
  STANDALONE_DEPTH_TEXTURE_LAYOUT,
  normalDepthWgslDepthComponent,
  type NormalDepthTextureLayout,
} from '../normalDepthEncoding.js';

export const SVGF_REAL_ATROUS_WORKGROUP_SIZE = 16 as const;

export function buildSvgfRealAtrousWgsl(
  depthLayout: NormalDepthTextureLayout = STANDALONE_DEPTH_TEXTURE_LAYOUT,
): string {
  const depthComponent = normalDepthWgslDepthComponent(depthLayout);
  return /* wgsl */ `
${LUMINANCE_WGSL}

struct AtrousVarianceAtrousUBO {
  iteration:   u32,
  sigmaColor:  f32,
  sigmaNormal: f32,
  sigmaDepth:  f32,
};

@group(0) @binding(0) var svgfAtrous_input:           texture_2d<f32>;
@group(0) @binding(1) var svgfAtrous_output:          texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var svgfAtrous_normal:          texture_2d<f32>;
@group(0) @binding(3) var svgfAtrous_depth:           texture_2d<f32>;
@group(0) @binding(4) var svgfAtrous_initialVariance: texture_2d<f32>;
@group(0) @binding(5) var<uniform> svgfAtrous_ubo:     AtrousVarianceAtrousUBO;

${ATROUS_VARIANCE_KERNEL_WGSL}

fn svgfVarianceAt(p: vec2u) -> f32 {
  if (svgfAtrous_ubo.iteration == 0u) {
    return max(0.0, textureLoad(svgfAtrous_initialVariance, p, 0).r);
  }
  return max(0.0, textureLoad(svgfAtrous_input, p, 0).a);
}

// Schied §4.3: this Gaussian is used only by the luminance edge stop.  The
// un-prefiltered variance is propagated below so repeated levels do not apply
// the Gaussian twice.
fn svgfPrefilterVariance3x3(center: vec2u, dims: vec2u) -> f32 {
  var weightedVariance = 0.0;
  var weightSum = 0.0;
  for (var dy: i32 = -1; dy <= 1; dy += 1) {
    for (var dx: i32 = -1; dx <= 1; dx += 1) {
      let p = vec2i(center) + vec2i(dx, dy);
      if (any(p < vec2i(0)) || any(vec2u(p) >= dims)) {
        continue;
      }
      let wx = select(2.0, 1.0, dx != 0);
      let wy = select(2.0, 1.0, dy != 0);
      let gaussianWeight = wx * wy;
      weightedVariance += gaussianWeight * svgfVarianceAt(vec2u(p));
      weightSum += gaussianWeight;
    }
  }
  return select(0.0, weightedVariance / weightSum, weightSum > 0.0);
}

fn svgfDepthAt(p: vec2u) -> f32 {
  return textureLoad(svgfAtrous_depth, p, 0).${depthComponent};
}

@compute @workgroup_size(16, 16, 1)
fn svgfRealAtrousMain(@builtin(global_invocation_id) gid: vec3u) {
  let dims = textureDimensions(svgfAtrous_input);
  if (any(gid.xy >= dims)) {
    return;
  }

  let centerColor = textureLoad(svgfAtrous_input, gid.xy, 0).rgb;
  let centerNormal = textureLoad(svgfAtrous_normal, gid.xy, 0).xyz * 2.0 - 1.0;
  let centerDepth = svgfDepthAt(gid.xy);
  let centerVariance = svgfVarianceAt(gid.xy);

  // A zero depth is the no-hit sentinel.  Negative depth is a valid packed
  // glass-primary marker in the walkaround renderer and must still be filtered.
  if (centerDepth == 0.0) {
    textureStore(
      svgfAtrous_output,
      gid.xy,
      vec4f(centerColor, min(centerVariance, 65504.0)),
    );
    return;
  }

  let xPrev = select(gid.x - 1u, gid.x, gid.x == 0u);
  let xNext = min(gid.x + 1u, dims.x - 1u);
  let yPrev = select(gid.y - 1u, gid.y, gid.y == 0u);
  let yNext = min(gid.y + 1u, dims.y - 1u);
  let depthGradient = vec2f(
    0.5 * (svgfDepthAt(vec2u(xNext, gid.y)) - svgfDepthAt(vec2u(xPrev, gid.y))),
    0.5 * (svgfDepthAt(vec2u(gid.x, yNext)) - svgfDepthAt(vec2u(gid.x, yPrev))),
  );

  let filteredVariance = svgfPrefilterVariance3x3(gid.xy, dims);
  let luminanceDenominator =
    svgfAtrous_ubo.sigmaColor * sqrt(max(filteredVariance, 0.0));
  let centerLuminance = luminance(centerColor);
  let normalExponent = max(1.0, svgfAtrous_ubo.sigmaNormal);
  let stepWidth = i32(1u << svgfAtrous_ubo.iteration);

  var colorNumerator = vec3f(0.0);
  var varianceNumerator = 0.0;
  var weightSum = 0.0;

  for (var dy: i32 = -2; dy <= 2; dy += 1) {
    for (var dx: i32 = -2; dx <= 2; dx += 1) {
      let pixelOffset = vec2i(dx, dy) * stepWidth;
      let p = vec2i(gid.xy) + pixelOffset;
      if (any(p < vec2i(0)) || any(vec2u(p) >= dims)) {
        continue;
      }
      let pu = vec2u(p);
      let sampleColor = textureLoad(svgfAtrous_input, pu, 0).rgb;
      let sampleNormal = textureLoad(svgfAtrous_normal, pu, 0).xyz * 2.0 - 1.0;
      let sampleDepth = svgfDepthAt(pu);

      let luminanceDelta = abs(luminance(sampleColor) - centerLuminance);
      var luminanceDistance = 0.0;
      if (luminanceDelta > 0.0) {
        luminanceDistance = 3.402823466e38;
        if (luminanceDenominator > 0.0) {
          luminanceDistance = luminanceDelta / luminanceDenominator;
        }
      }
      let projectedDepthGradient =
        abs(dot(depthGradient, vec2f(pixelOffset)));
      let depthDenominator =
        svgfAtrous_ubo.sigmaDepth * projectedDepthGradient;
      let depthDelta = abs(sampleDepth - centerDepth);
      var depthDistance = 0.0;
      if (depthDelta > 0.0) {
        depthDistance = 3.402823466e38;
        if (depthDenominator > 0.0) {
          depthDistance = depthDelta / depthDenominator;
        }
      }
      let normalWeight =
        pow(clamp(dot(centerNormal, sampleNormal), 0.0, 1.0), normalExponent);
      let edgeWeight =
        exp(-max(0.0, luminanceDistance) - max(0.0, depthDistance)) * normalWeight;

      let kernelIndex = u32((dy + 2) * 5 + (dx + 2));
      let weight = ATROUS_VARIANCE_KERNEL[kernelIndex] * edgeWeight;
      let sampleVariance = svgfVarianceAt(pu);

      colorNumerator += weight * sampleColor;
      // Variance of a weighted sum of uncorrelated samples.  The squared
      // weights are essential; an ordinary weighted average is not equivalent.
      varianceNumerator += weight * weight * sampleVariance;
      weightSum += weight;
    }
  }

  let hasSupport = weightSum > 1e-8;
  let filteredColor = select(centerColor, colorNumerator / weightSum, hasSupport);
  let propagatedVariance = select(
    centerVariance,
    varianceNumerator / (weightSum * weightSum),
    hasSupport,
  );
  textureStore(
    svgfAtrous_output,
    gid.xy,
    vec4f(filteredColor, min(max(0.0, propagatedVariance), 65504.0)),
  );
}
`;
}

/** Standalone one-shot ABI: depth is supplied in a dedicated R texture. */
export const SVGF_REAL_ATROUS_WGSL = buildSvgfRealAtrousWgsl();
