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
 * edge-stop form) are absent. à-trous + variance denoiser; not Schied SVGF —
 * see svgfRealWebGPU.ts for real SVGF.
 *
 * Two legacy-named compute entry points (the exported names are retained for
 * compatibility; they do not mean this module is a Schied-SVGF implementation):
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
 *     binding 0 — texture_2d<f32>                        inputColor   (noisy RGBA16F)
 *     binding 1 — texture_2d<f32>                        varianceIn   (RG32F — WelfordVariance mean+m2)
 *     binding 2 — texture_storage_2d<r32float, write>    varianceOut  (estimated scalar variance per pixel)
 *     binding 3 — var<uniform> AtrousVarianceVarianceUBO
 *
 *   group 0 — à-trous pass (svgfAtrousMain):
 *     binding 0 — texture_2d<f32>                        inputColor     (RGBA16F — ping-pong input)
 *     binding 1 — texture_storage_2d<rgba16float, write> outputColor    (RGBA16F — ping-pong output)
 *     binding 2 — texture_2d<f32>                        gbufferNormal  (RGBA16F, .xyz = world normal)
 *     binding 3 — texture_2d<f32>                        gbufferDepth   (RGBA16F or R32F, .x = linear depth)
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
 *
 * INTENTIONAL DIVERGENCE from walkaround-hybrid's SVGF edge-stop:
 *   • This module uses a variance-guided LUMINANCE edge-stop weight:
 *       w_l = exp(-|lum(c) - lum(c')|² / (σ_l² * var + ε))
 *     This is the Dammertz 2010 standalone à-trous form — a single luminance
 *     channel drives the edge-stop, and variance modulates the sigma.
 *   • walkaround-hybrid's SVGF (svgfRealWebGPU.ts, via ATROUS_PASS_WGSL) uses
 *     Schied 2017 Eq. 4 — separate σ_l / σ_n / σ_d terms, a 3-component
 *     g-buffer edge-stop (luminance, world normal, linear depth), and a
 *     per-pixel history-length-clamped α. The two formulas produce different
 *     filtering behaviour; they are NOT interchangeable.
 *   • Do NOT "unify" these into a single shared WGSL kernel without verifying
 *     that both consumers produce identical output on their respective test
 *     scenes (the behavioral gate covers both).
 */

import { LUMINANCE_WGSL } from '@vitrum/shared-samplers';
import { ATROUS_VARIANCE_TEMPORAL_MIN_FRAME_COUNT } from '../atrousVarianceConstants.js';
import { WELFORD_VARIANCE_WGSL } from './welfordVariance.wgsl.js';
import { ATROUS_VARIANCE_KERNEL_WGSL } from './atrousKernel.wgsl.js';
import {
  STANDALONE_DEPTH_TEXTURE_LAYOUT,
  normalDepthWgslDepthComponent,
  type NormalDepthTextureLayout,
} from '../normalDepthEncoding.js';

/** Must match `@workgroup_size` in this module's compute entry points. */
export const ATROUS_VARIANCE_COMPUTE_WORKGROUP_SIZE = 16 as const;

export function buildAtrousVarianceWgsl(
  depthLayout: NormalDepthTextureLayout = STANDALONE_DEPTH_TEXTURE_LAYOUT,
): string {
  const depthComponent = normalDepthWgslDepthComponent(depthLayout);
  return /* wgsl */ `
// Canonical Rec.709 luminance — @vitrum/shared-samplers/wgsl/luminance.wgsl.
// Provides const LUM_W709 + fn luminance(c: vec3f) -> f32. The local fn
// luminance defined below at "Variance Estimation Pass" delegates to this.
${LUMINANCE_WGSL}

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
  sigmaColor:  f32,   // color edge-stop σ (default 4.0)
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
// Output (varianceOut): R32Float texture containing the scalar luminance
// variance estimate used by the à-trous pass. Frame count is consumed here
// from the UBO and is not duplicated into an unread texture lane.
// ============================================================

@group(0) @binding(0) var varIn_inputColor:   texture_2d<f32>;
@group(0) @binding(1) var varIn_varianceIn:    texture_2d<f32>;
@group(0) @binding(2) var varOut_varianceOut:  texture_storage_2d<r32float, write>;
@group(0) @binding(3) var<uniform>  varUBO:   AtrousVarianceVarianceUBO;

// fn luminance(c: vec3f) — canonical from LUMINANCE_WGSL above; uses LUM_W709.

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

  textureStore(varOut_varianceOut, gid.xy, vec4f(variance, 0.0, 0.0, 0.0));
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
${ATROUS_VARIANCE_KERNEL_WGSL}

@compute @workgroup_size(16, 16, 1)
fn svgfAtrousMain(@builtin(global_invocation_id) gid: vec3u) {
  let dims = textureDimensions(atrous_inputColor);
  if (any(gid.xy >= dims)) { return; }

  let cCenter = textureLoad(atrous_inputColor, gid.xy, 0).rgb;
  // Normal uses the packed affine encoding; depth component is selected by the
  // host-declared physical texture layout.
  let nCenter = textureLoad(atrous_gbufNormal, gid.xy, 0).xyz * 2.0 - 1.0;
  let zCenter = textureLoad(atrous_gbufDepth, gid.xy, 0).${depthComponent};

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
  let sigZ2 = atrousUBO.sigmaDepth * atrousUBO.sigmaDepth;

  for (var dy = -2; dy <= 2; dy++) {
    for (var dx = -2; dx <= 2; dx++) {
      let off = vec2i(dx, dy) * sw;
      let p   = vec2i(gid.xy) + off;
      if (any(p < vec2i(0)) || any(vec2u(p) >= dims)) { continue; }
      let pu  = vec2u(p);

      let cP = textureLoad(atrous_inputColor, pu, 0).rgb;
      let nP = textureLoad(atrous_gbufNormal, pu, 0).xyz * 2.0 - 1.0;
      let zP = textureLoad(atrous_gbufDepth,  pu, 0).${depthComponent};

      let kIdx = u32((dy + 2) * 5 + (dx + 2));
      let h    = ATROUS_VARIANCE_KERNEL[kIdx];

      // ── Variance-guided color edge stop ─────────────────────────────────
      // Tolerance scales with sqrt(variance): noisy pixels accept wider
      // color neighborhoods, converged pixels apply tighter edges.
      let lumP = luminance(cP);
      let dLum = lumP - lumCenter;
      let colorDenom = sigC2 * varEstimate;
      var wc = 1.0;
      if (dLum != 0.0) {
        wc = 0.0;
        if (colorDenom > 0.0) {
          wc = exp(-dLum * dLum / colorDenom);
        }
      }

      // ── Normal edge stop ────────────────────────────────────────────────
      // Clamp both ends: finite-but-imperfect normal inputs and f32 roundoff
      // must never turn the high-exponent edge stop into pow(x>1, 128).
      let dn = clamp(dot(nCenter, nP), 0.0, 1.0);
      let wn = pow(dn, sigN);

      // ── Depth edge stop ─────────────────────────────────────────────────
      let depthDelta = zP - zCenter;
      var wz = 1.0;
      if (depthDelta != 0.0) {
        wz = 0.0;
        if (sigZ2 > 0.0) {
          wz = exp(-(depthDelta * depthDelta) / sigZ2);
        }
      }

      let w  = h * wc * wn * wz;
      sumColor  += cP * w;
      sumWeight += w;
    }
  }

  var result = cCenter;
  if (sumWeight > 0.0) {
    result = sumColor / sumWeight;
  }
  textureStore(atrous_outputColor, gid.xy, vec4f(result, 1.0));
}
`;
}

/** Standalone ABI: depth is supplied by a dedicated R texture. */
export const ATROUS_VARIANCE_WGSL = buildAtrousVarianceWgsl();
