/**
 * atrousVariance.wgsl.ts — à-trous wavelet edge-stop filter + per-pixel variance scalar lookup.
 *
 * NOT a Schied 2017 SVGF implementation. What this module provides:
 *   - À-trous wavelet edge-stop filtering (Dammertz 2010).
 *   - Per-pixel variance scalar lookup from Welford temporal accumulation
 *     (Sprint 9) or 3×3 spatial estimate (early frames).
 *
 * The defining SVGF temporal stages (bilinear reprojection, disocclusion
 * detection, per-pixel history length, variance-guided α-clamp, Schied Eq. 4
 * edge-stop form) are absent. Real Schied 2017 SVGF is tracked in
 * plan/sprint-svgf-real-future.md.
 *
 * Previously named svgf.wgsl.ts; renamed by sweep-2026-05-11 D3 to match
 * what the implementation actually does.
 *
 * Two compute entry points:
 *
 *   svgfVarianceMain — Variance estimation pass.
 *     When temporal history is scarce (frame count below ATROUS_VARIANCE_TEMPORAL_MIN_FRAME_COUNT), estimates variance
 *     from a 3×3 spatial neighborhood. When temporal history is stable,
 *     falls back to the Welford running variance already in the variance buffer.
 *     Writes a per-pixel scalar variance estimate into the output.
 *
 *   svgfAtrousMain — À-trous wavelet pass with variance-guided edge stops.
 *     5 iterations (step widths 1, 2, 4, 8, 16). Each iteration reads
 *     edge-stopping weights modulated by the per-pixel variance estimate.
 *     Configured by iteration index uniform; the host dispatches this 5× with
 *     incrementing iteration values.
 *
 * Bind groups:
 *
 *   group 0 — variance estimation pass (svgfVarianceMain):
 *     binding 0 — texture_2d<f32>                        inputColor     (noisy RGBA16F)
 *     binding 1 — texture_2d<f32>                        prevRadiance   (previous frame RGBA16F)
 *     binding 2 — texture_2d<f32>                        gbufferNormal  (RGBA16F, .xyz = world normal)
 *     binding 3 — texture_2d<f32>                        gbufferDepth   (RGBA16F or R32F, .r = linear depth)
 *     binding 4 — texture_2d<f32>                        motionVectors  (RG32F, .xy = screen-space motion)
 *     binding 5 — texture_2d<f32>                        varianceIn     (RG32F — WelfordVariance mean+m2)
 *     binding 6 — texture_storage_2d<rg32float, write>   varianceOut    (estimated scalar variance per pixel)
 *     binding 7 — var<uniform> AtrousVarianceVarianceUBO
 *
 *   group 0 — à-trous pass (svgfAtrousMain):
 *     binding 0 — texture_2d<f32>                        inputColor     (RGBA16F — ping-pong input)
 *     binding 1 — texture_storage_2d<rgba16float, write> outputColor    (RGBA16F — ping-pong output)
 *     binding 2 — texture_2d<f32>                        gbufferNormal  (RGBA16F, .xyz = world normal)
 *     binding 3 — texture_2d<f32>                        gbufferDepth   (RGBA16F or R32F, .r = linear depth)
 *     binding 4 — texture_2d<f32>                        varianceMap    (RG32F — .r = estimated variance)
 *     binding 5 — var<uniform> AtrousVarianceAtrousUBO
 *
 * WelfordVariance struct:
 *   This shader declares a local copy of WelfordVariance matching the layout
 *   in walkaround-hybrid/src/shaders/common.wgsl.ts (Decision 13, Sprint 9).
 *   Cross-package WGSL string imports are not supported (each package exports
 *   its own self-contained WGSL string). The struct fields are byte-for-byte
 *   identical and the comment below links to the canonical definition.
 *   @see walkaround-hybrid/src/shaders/common.wgsl.ts — WelfordVariance @version 1
 *
 * References:
 *   Dammertz, Hanika, Keller "Edge-Avoiding À-Trous Wavelet Transform for
 *   fast Global Illumination Filtering". HPG 2010.
 *
 *   WelfordVariance layout: canonical in ./welfordVariance.wgsl.ts.
 *   Decision 13 — versioned struct pinned Sprint 9 (2026-05-09).
 */

import { ATROUS_VARIANCE_TEMPORAL_MIN_FRAME_COUNT } from '../atrousVarianceConstants.js';
import { WELFORD_VARIANCE_WGSL } from './welfordVariance.wgsl.js';
import { SVGF_ATROUS_KERNEL_WGSL } from './atrousKernel.wgsl.js';

/** Must match `@workgroup_size` in this module's compute entry points. */
export const ATROUS_VARIANCE_COMPUTE_WORKGROUP_SIZE = 16 as const;

export const ATROUS_VARIANCE_WGSL = /* wgsl */ `
// Temporal branch threshold — single source: ../atrousVarianceConstants.ts ATROUS_VARIANCE_TEMPORAL_MIN_FRAME_COUNT
const SVGF_TEMPORAL_VARIANCE_MIN_FRAMES: u32 = ${ATROUS_VARIANCE_TEMPORAL_MIN_FRAME_COUNT}u;

// ============================================================
// WelfordVariance — canonical struct + helpers from welfordVariance.wgsl.ts.
// ============================================================
${WELFORD_VARIANCE_WGSL}

// ============================================================
// Constants
// ============================================================
const PI       = 3.14159265358979;
const INV_PI   = 0.31830988618;
// Rec. 709 luminance weights — canonical value; identical copies exist
// in atrous.wgsl.ts, spatialFilter.wgsl.ts, hdrLuminanceBilateral.wgsl.ts.
const LUM_W    = vec3f(0.2126, 0.7152, 0.0722);

// ============================================================
// Variance estimation uniforms
// ============================================================
struct AtrousVarianceVarianceUBO {
  frameCount:  u32,   // cumulative frames since last camera reset (0 = first frame)
  _pad0:       u32,
  _pad1:       u32,
  _pad2:       u32,
};

// ============================================================
// À-trous iteration uniforms
// ============================================================
struct AtrousVarianceAtrousUBO {
  iteration:   u32,   // 0-4 (step width = 1 << iteration)
  sigmaColor:  f32,   // color edge-stop σ (default 10.0)
  sigmaNormal: f32,   // normal edge-stop σ as exponent (default 128.0)
  sigmaDepth:  f32,   // depth edge-stop σ in world units (default 1.0)
};

// ============================================================
// Variance Estimation Pass — svgfVarianceMain
//
// Selects the best available variance estimate per pixel:
//   - frameCount < SVGF_TEMPORAL_VARIANCE_MIN_FRAMES: spatial 3×3 variance from the noisy color buffer.
//     This covers the first few frames where temporal Welford data is
//     scarce (mean is still far from steady state).
//   - frameCount >= SVGF_TEMPORAL_VARIANCE_MIN_FRAMES: Welford variance from the running buffer. This
//     is a temporally stable estimate that converges over time.
//
// Output (varianceOut): RG32Float texture.
//   .r = scalar luminance variance estimate (used by the à-trous pass).
//   .g = frameCount cast to f32 (informational — lets downstream passes
//        read the convergence state without an extra uniform).
// ============================================================

@group(0) @binding(0) var varIn_inputColor:   texture_2d<f32>;
@group(0) @binding(1) var varIn_prevRadiance:  texture_2d<f32>;
@group(0) @binding(2) var varIn_gbufNormal:    texture_2d<f32>;
@group(0) @binding(3) var varIn_gbufDepth:     texture_2d<f32>;
@group(0) @binding(4) var varIn_motionVec:     texture_2d<f32>;
@group(0) @binding(5) var varIn_varianceIn:    texture_2d<f32>;
@group(0) @binding(6) var varOut_varianceOut:  texture_storage_2d<rg32float, write>;
@group(0) @binding(7) var<uniform>  varUBO:   AtrousVarianceVarianceUBO;

fn luminance(c: vec3f) -> f32 {
  return dot(c, LUM_W);
}

@compute @workgroup_size(16, 16, 1)
fn svgfVarianceMain(@builtin(global_invocation_id) gid: vec3u) {
  let dims = textureDimensions(varIn_inputColor);
  if (any(gid.xy >= dims)) { return; }

  let frameCount = varUBO.frameCount;

  var variance: f32;

  if (frameCount < SVGF_TEMPORAL_VARIANCE_MIN_FRAMES) {
    // ── Spatial 3×3 variance from noisy color ────────────────────────────
    // Compute sample mean and sum-of-squared-deviations over a 3×3 window.
    var sum    = 0.0;
    var sumSq  = 0.0;
    var n      = 0u;

    for (var dy = -1; dy <= 1; dy++) {
      for (var dx = -1; dx <= 1; dx++) {
        let p = vec2i(gid.xy) + vec2i(dx, dy);
        if (any(p < vec2i(0)) || any(vec2u(p) >= dims)) { continue; }
        let lum = luminance(textureLoad(varIn_inputColor, vec2u(p), 0).rgb);
        sum   += lum;
        sumSq += lum * lum;
        n     += 1u;
      }
    }

    if (n > 1u) {
      let mean = sum / f32(n);
      // Biased estimator: (ΣxΒ² / n) - μ². Variance of ~9 samples in steady-
      // state gives a good initial edge-stop signal without division by n-1.
      variance = max(0.0, sumSq / f32(n) - mean * mean);
    } else {
      variance = 0.0;
    }
  } else {
    // ── Temporal Welford variance from running buffer ─────────────────────
    // varianceIn texel: .r = mean, .g = M2. Welford: variance = M2 / (n-1).
    let raw = textureLoad(varIn_varianceIn, gid.xy, 0);
    let state = WelfordVariance(raw.r, raw.g);
    variance = welfordVariance(state, frameCount);
  }

  textureStore(varOut_varianceOut, gid.xy, vec4f(variance, f32(frameCount), 0.0, 0.0));
}

// ============================================================
// À-trous Wavelet Pass — svgfAtrousMain
//
// 5×5 B3-spline kernel with step width = 1 << iteration.
// Edge-stopping weights:
//   wc — color: exp(-|lum_p - lum_center|² / (sigmaColor² × variance + ε))
//               Variance-guided: high variance → wider color tolerance.
//   wn — normal: pow(max(0, dot(n_p, n_center)), sigmaNormal)
//   wz — depth: exp(-(z_p - z_center)² / sigmaDepth²)
//
// Ping-pong usage: host swaps inputColor / outputColor between iterations.
// Iteration 0 → step width 1 (finest detail).
// Iteration 4 → step width 16 (coarsest low-frequency pass).
// ============================================================

// Separate binding namespace for the à-trous pass so both entry points
// can coexist in the same shader module without binding conflicts.
@group(0) @binding(0) var atrous_inputColor:  texture_2d<f32>;
@group(0) @binding(1) var atrous_outputColor: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var atrous_gbufNormal:  texture_2d<f32>;
@group(0) @binding(3) var atrous_gbufDepth:   texture_2d<f32>;
@group(0) @binding(4) var atrous_varianceMap: texture_2d<f32>;
@group(0) @binding(5) var<uniform> atrousUBO: AtrousVarianceAtrousUBO;

// 5×5 B3 spline kernel — injected from shared TS constant (atrousKernel.wgsl.ts).
${SVGF_ATROUS_KERNEL_WGSL}

@compute @workgroup_size(16, 16, 1)
fn svgfAtrousMain(@builtin(global_invocation_id) gid: vec3u) {
  let dims = textureDimensions(atrous_inputColor);
  if (any(gid.xy >= dims)) { return; }

  let cCenter = textureLoad(atrous_inputColor, gid.xy, 0).rgb;
  // Match walkaround atrous.wgsl: packed normal (0..1) → world normal; depth in .w
  let nCenter = textureLoad(atrous_gbufNormal, gid.xy, 0).xyz * 2.0 - 1.0;
  let zCenter = textureLoad(atrous_gbufDepth, gid.xy, 0).w;

  // Sky / miss pixels pass through unfiltered.
  if (zCenter <= 0.0) {
    textureStore(atrous_outputColor, gid.xy, vec4f(cCenter, 1.0));
    return;
  }

  // Per-pixel variance estimate from the variance pass.
  let varEstimate = max(0.0, textureLoad(atrous_varianceMap, gid.xy, 0).r);

  // Step width doubles each iteration: 1, 2, 4, 8, 16.
  let sw = i32(1u << atrousUBO.iteration);

  let lumCenter = luminance(cCenter);

  var sumColor  = vec3f(0.0);
  var sumWeight = 0.0;

  let sigC2 = atrousUBO.sigmaColor  * atrousUBO.sigmaColor;
  let sigN  = max(1.0, atrousUBO.sigmaNormal);
  let sigZ2 = atrousUBO.sigmaDepth  * atrousUBO.sigmaDepth + 1e-6;

  for (var dy = -2; dy <= 2; dy++) {
    for (var dx = -2; dx <= 2; dx++) {
      let off = vec2i(dx, dy) * sw;
      let p   = vec2i(gid.xy) + off;
      if (any(p < vec2i(0)) || any(vec2u(p) >= dims)) { continue; }
      let pu  = vec2u(p);

      let cP = textureLoad(atrous_inputColor, pu, 0).rgb;
      let nP = textureLoad(atrous_gbufNormal, pu, 0).xyz * 2.0 - 1.0;
      let zP = textureLoad(atrous_gbufDepth,  pu, 0).w;

      let kIdx = u32((dy + 2) * 5 + (dx + 2));
      let h    = SVGF_KERNEL[kIdx];

      // ── Variance-guided color edge stop ─────────────────────────────────
      // Tolerance scales with sqrt(variance): noisy pixels accept wider
      // color neighborhoods, converged pixels apply tighter edges.
      let lumP = luminance(cP);
      let dLum = lumP - lumCenter;
      // Add a small epsilon to the denominator so the first-frame case
      // (zero variance) still allows some neighborhood smoothing.
      let colorDenom = sigC2 * (varEstimate + 0.001) + 1e-6;
      let wc = exp(-dLum * dLum / colorDenom);

      // ── Normal edge stop ────────────────────────────────────────────────
      let dn = max(0.0, dot(nCenter, nP));
      let wn = pow(dn, sigN);

      // ── Depth edge stop ─────────────────────────────────────────────────
      let wz = exp(-(zP - zCenter) * (zP - zCenter) / sigZ2);

      let w  = h * wc * wn * wz;
      sumColor  += cP * w;
      sumWeight += w;
    }
  }

  let result = select(cCenter, sumColor / sumWeight, sumWeight > 1e-6);
  textureStore(atrous_outputColor, gid.xy, vec4f(result, 1.0));
}
`;
