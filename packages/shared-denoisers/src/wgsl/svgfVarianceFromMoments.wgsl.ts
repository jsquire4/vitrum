/**
 * svgfVarianceFromMoments.wgsl.ts — Schied 2017 SVGF Eq. 5 variance pass.
 *
 * For pixels with sufficient history (h ≥ SVGF_HISTORY_MIN_FOR_MOMENTS),
 * computes per-pixel luminance variance from the blended first and second
 * moment textures:
 *
 *   Var_i = max(0, M2_i − M1_i²)   — Schied Eq. 5
 *
 * For pixels with history < SVGF_HISTORY_MIN_FOR_MOMENTS (see svgf7x7SpatialFallback.wgsl.ts),
 * this pass writes a zero (the fallback is used instead and the two outputs
 * are combined by the host by the time the atrous chain reads).
 *
 * Bind group 0 layout (entry point: svgfVarianceFromMomentsMain):
 *   binding 0 — texture_2d<f32>                        momentsIn     (rg32float M1, M2)
 *   binding 1 — texture_2d<u32>                        historyIn     (r16uint history length)
 *   binding 2 — texture_storage_2d<rgba32float, write> varianceOut   (.r = variance, .g = h cast to f32)
 *
 * References:
 *   Schied et al. HPG 2017 §4, Equation 5.
 */

/** Frames of history below which this pass writes 0 (7×7 fallback takes over). */
export const SVGF_HISTORY_MIN_FOR_MOMENTS = 4 as const;

/** Must match @workgroup_size in svgfVarianceFromMomentsMain. */
export const SVGF_VARIANCE_FROM_MOMENTS_WORKGROUP_SIZE = 16 as const;

export const SVGF_VARIANCE_FROM_MOMENTS_WGSL = /* wgsl */ `
const SVGF_HISTORY_THRESHOLD: u32 = ${SVGF_HISTORY_MIN_FOR_MOMENTS}u;

@group(0) @binding(0) var vmom_momentsIn:   texture_2d<f32>;
@group(0) @binding(1) var vmom_historyIn:   texture_2d<u32>;
@group(0) @binding(2) var vmom_varianceOut: texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(16, 16, 1)
fn svgfVarianceFromMomentsMain(@builtin(global_invocation_id) gid: vec3u) {
  let dims = textureDimensions(vmom_momentsIn);
  if (any(gid.xy >= dims)) { return; }

  let h  = textureLoad(vmom_historyIn, gid.xy, 0).r;
  var variance = 0.0;

  if (h >= SVGF_HISTORY_THRESHOLD) {
    let m = textureLoad(vmom_momentsIn, gid.xy, 0).rg;
    let m1 = m.r;
    let m2 = m.g;
    // Eq. 5: Var = M2 - M1². Clamp to [0,∞).
    variance = max(0.0, m2 - m1 * m1);
  }
  // When h < threshold: variance = 0 → svgf7x7SpatialFallback covers those pixels.

  textureStore(vmom_varianceOut, gid.xy, vec4f(variance, f32(h), 0.0, 0.0));
}
`;
