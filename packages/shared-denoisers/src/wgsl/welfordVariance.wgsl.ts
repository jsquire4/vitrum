/**
 * WelfordVariance — canonical per-pixel running variance state.
 *
 * Single source of truth for the WelfordVariance WGSL struct and its
 * update / variance helpers. Imported by:
 *  - `walkaround-hybrid/src/shaders/common.wgsl.ts` (concatenates into COMMON_WGSL)
 *  - `shared-denoisers/src/wgsl/atrousVariance.wgsl.ts` (concatenates into ATROUS_VARIANCE_WGSL)
 *
 * Layout (RG32Float texel):
 *   r = mean (running average of luminance)
 *   g = M2  (sum of squared deltas; variance = M2 / (n - 1))
 *
 * n is implicit from sample counter — host passes per-frame sample-count
 * uniform, shaders compute variance = welford.g / (n - 1).
 *
 * Decision 13 (locked 2026-05-09): versioned named struct prevents
 * independent re-declarations across sprints / packages. Bump
 * WELFORD_VARIANCE_VERSION when the layout changes.
 */

export const WELFORD_VARIANCE_VERSION = 1;

export const WELFORD_VARIANCE_WGSL = /* wgsl */ `
// Decision 13 (locked 2026-05-09): versioned struct prevents independent
// re-declarations across sprints/packages. Bump WELFORD_VARIANCE_VERSION
// when the layout changes.
//
// Layout (RG32Float texel):
//   r = mean (running average of luminance)
//   g = M2  (sum of squared deltas; variance = M2 / (n - 1))
//
// @version 1 (Sprint 9, 2026-05-09).
struct WelfordVariance {
  mean: f32,
  m2:   f32,
};

// Online Welford update for one new sample.
// prev: current running state, sample: new luminance value, n: new sample count (1-based).
// Returns: updated state.
fn welfordUpdate(prev: WelfordVariance, sample: f32, n: u32) -> WelfordVariance {
  let delta = sample - prev.mean;
  let mean  = prev.mean + delta / f32(n);
  let m2    = prev.m2   + delta * (sample - mean);
  return WelfordVariance(mean, m2);
}

// Compute unbiased sample variance from the Welford state.
// Returns 0 for n < 2 (not enough samples for a meaningful estimate).
fn welfordVariance(state: WelfordVariance, n: u32) -> f32 {
  if (n < 2u) { return 0.0; }
  return state.m2 / f32(n - 1u);
}
`;
