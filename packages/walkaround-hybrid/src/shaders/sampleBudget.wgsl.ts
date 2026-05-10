/**
 * sampleBudget.wgsl — Variance-driven per-pixel sample-tier compute shader.
 *
 * Sprint 9 (walkaround only). Reads the per-pixel WelfordVariance state from
 * a RG32Float storage texture, computes a dispatch-tier byte per pixel, and
 * writes it to a sample-count texture (r32uint).
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
 * Integration status: DEFERRED — shader is complete and runnable but is NOT
 * wired into the dispatch pipeline. Sprint 10a or a follow-up GPU-verification
 * sprint will insert this pass between the temporal accumulator and the next
 * RIS pass. See plan/sprint-9-walkaround-integration.md for wiring details.
 *
 * Dependencies: COMMON_WGSL (WelfordVariance struct, welfordVariance fn).
 *
 * @version 1 (Sprint 9, 2026-05-09)
 */

export const SAMPLE_BUDGET_WGSL = /* wgsl */ `

// ── Uniforms ─────────────────────────────────────────────────────────────────

struct SampleBudgetUniforms {
  threshold_low:  f32,   // variance below this → tier 1 (converged)
  threshold_high: f32,   // variance below this → tier 2 (needs some samples)
  screenWidth:    u32,
  screenHeight:   u32,
};

@group(0) @binding(0) var<uniform>            u_budget: SampleBudgetUniforms;

// RG32Float — r=mean, g=M2. Written by the temporal accumulator pass each frame.
@group(0) @binding(1) var                     t_variance: texture_storage_2d<rg32float, read>;

// r32uint — tier byte per pixel (1, 2, or 4). Read by next-frame RIS dispatch.
@group(0) @binding(2) var                     t_tier_out: texture_storage_2d<r32uint, write>;

// Sample counter (per-frame, same value for all pixels in a frame).
// Host writes this as part of the per-frame UBO update.
@group(0) @binding(3) var<uniform>            u_sampleCount: u32;

// ── WelfordVariance (inlined from COMMON_WGSL) ───────────────────────────────
//
// NOTE: In production use, this shader string is concatenated AFTER COMMON_WGSL
// so WelfordVariance, welfordUpdate, and welfordVariance are available without
// redeclaration. The inline copy below is for standalone validation only.
// When concatenated, remove the struct/fn declarations marked [INLINE-COPY].

// [INLINE-COPY] struct WelfordVariance { mean: f32, m2: f32, };

// [INLINE-COPY] fn welfordVariance(state: WelfordVariance, n: u32) -> f32 {
//   if (n < 2u) { return 0.0; }
//   return state.m2 / f32(n - 1u);
// }

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
  let raw = textureLoad(t_variance, vec2<u32>(px, py));
  let state = WelfordVariance(raw.r, raw.g);

  // Compute unbiased variance. Returns 0 for n < 2.
  let n = u_sampleCount;
  let variance = welfordVariance(state, n);

  // Classify tier.
  let tier = sampleTierFromVariance(variance, u_budget.threshold_low, u_budget.threshold_high);

  // Write tier to per-pixel tile.
  textureStore(t_tier_out, vec2<u32>(px, py), vec4<u32>(tier, 0u, 0u, 0u));
}
`;
