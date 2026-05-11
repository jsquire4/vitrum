/**
 * sampleBudget.wgsl — Variance-driven per-pixel sample-tier compute shader.
 *
 * Sprint 9 — Walkaround adaptive sampling. Reads the per-pixel Welford
 * variance state written by the welford-temporal pass and classifies each
 * pixel into a sample tier byte that downstream passes consume.
 *
 * Tier semantics:
 *   1 = high-confidence pixel   (variance < threshold_low  → 1 ray/frame)
 *   2 = medium-confidence pixel (variance < threshold_high → 2 rays/frame)
 *   4 = low-confidence pixel    (variance ≥ threshold_high → 4 rays/frame)
 *
 * Threshold defaults (host-overridable via the SampleBudgetUniforms UBO):
 *   threshold_low  = 0.01  — pixel is visually converged; reduce sampling
 *   threshold_high = 0.10  — pixel needs more samples but isn't urgent
 *
 * These values were chosen by calibration against the walkaround AABB
 * accumulator: a pixel with luminance variance < 0.01 after temporal
 * blending is perceptually stable on a calibrated display; variance > 0.10
 * produces visible noise even through the à-trous denoiser.
 *
 * Pipeline placement:
 *   slot 0 — sample-budget   (this pass)
 *     ↓ writes tier texture (r32uint)
 *   slot 1 — ris
 *     ↓ (tier texture is currently informational only — RIS does not yet
 *        consume it; future work may use tier to gate M_LIGHT)
 *   slot 5 — shade
 *     ↓ checkerboard sparse write (Sprint 9 companion of this pass)
 *
 * The variance texture this pass reads is the Welford state written by
 * the welford-temporal pass on the PREVIOUS frame (read via the ping-pong
 * slot that is NOT being written this frame). The first frame has no
 * meaningful variance so all pixels classify to tier 4 (low confidence).
 *
 * Bindings deliberately use `texture_2d<f32>` (sampled, unfilterable-float)
 * for the variance input instead of a storage-read view: WebGPU does not
 * support read-access for rg32float storage textures across all browsers,
 * but the source variance texture was created with TEXTURE_BINDING usage
 * for exactly this read path.
 *
 * Dependencies: WELFORD_VARIANCE_WGSL (canonical struct + helpers) is
 * injected below. Do NOT prepend COMMON_WGSL — that would cause a
 * redeclaration error for WelfordVariance.
 *
 * @version 2 (Sprint 9 wire-in, 2026-05-11)
 */

import { WELFORD_VARIANCE_WGSL } from '@vitrum/shared-denoisers';

export const SAMPLE_BUDGET_WGSL = /* wgsl */ `

// ── Uniforms ─────────────────────────────────────────────────────────────────

struct SampleBudgetUniforms {
  threshold_low:  f32,   // variance below this → tier 1 (converged)
  threshold_high: f32,   // variance below this → tier 2 (needs some samples)
  screenWidth:    u32,
  screenHeight:   u32,
};

@group(0) @binding(0) var<uniform>            u_budget: SampleBudgetUniforms;

// RG32Float — r=mean, g=M2. Written by the welford-temporal pass each frame.
// Bound as texture_2d<f32> (unfilterable) because storage-read of rg32float
// is not portably supported across WebGPU browsers.
@group(0) @binding(1) var                     t_variance: texture_2d<f32>;

// r32uint — tier byte per pixel (1, 2, or 4). Read by next-frame shade dispatch.
@group(0) @binding(2) var                     t_tier_out: texture_storage_2d<r32uint, write>;

// Sample counter (per-frame, same value for all pixels in a frame).
// Host writes this as part of the per-frame UBO update.
struct SampleCountUniforms {
  sampleCount: u32,
  _pad0:       u32,
  _pad1:       u32,
  _pad2:       u32,
};
@group(0) @binding(3) var<uniform>            u_sampleCount: SampleCountUniforms;

// ── WelfordVariance (canonical, injected from @vitrum/shared-denoisers) ─────
${WELFORD_VARIANCE_WGSL}

// ── Tier classification ───────────────────────────────────────────────────────

fn sampleTierFromVariance(v: f32, threshold_low: f32, threshold_high: f32) -> u32 {
  if (v < threshold_low)  { return 1u; }
  if (v < threshold_high) { return 2u; }
  return 4u;
}

// ── Compute entry point ───────────────────────────────────────────────────────

@compute @workgroup_size(8, 8, 1)
fn sampleBudgetKernel(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let px = globalId.x;
  let py = globalId.y;

  // Guard against out-of-bounds invocations from workgroup padding.
  if (px >= u_budget.screenWidth || py >= u_budget.screenHeight) { return; }

  // Load per-pixel Welford state. r=mean, g=M2 (RG32Float texel).
  let raw = textureLoad(t_variance, vec2<i32>(i32(px), i32(py)), 0);
  let state = WelfordVariance(raw.r, raw.g);

  // Compute unbiased variance. Returns 0 for n < 2.
  let n = u_sampleCount.sampleCount;
  let variance = welfordVariance(state, n);

  // Classify tier.
  let tier = sampleTierFromVariance(variance, u_budget.threshold_low, u_budget.threshold_high);

  // Write tier to per-pixel tile.
  textureStore(t_tier_out, vec2<u32>(px, py), vec4<u32>(tier, 0u, 0u, 0u));
}
`;
