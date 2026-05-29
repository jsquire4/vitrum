/**
 * bmfrConstants.ts — Tunable defaults for the BMFR denoiser.
 *
 * BMFR = Koskela et al. 2019, "Blockwise Multi-Order Feature Regression for
 * Real-Time Path-Tracing Reconstruction" (ACM TOG 38(5)).
 *
 * Single source of truth for the host dispatcher (`bmfrWebGPU.ts`), the WGSL
 * kernel (`wgsl/bmfr.wgsl.ts` interpolates these), and the walkaround-hybrid
 * registry entry.
 */

/**
 * Temporal-accumulation EMA weight on the CURRENT reconstructed frame.
 *
 * `out = (1-α)·history + α·current`. Koskela 2019 use an exponential moving
 * average of the reconstructed (regressed) frame; α = 0.2 keeps ~5 frames of
 * effective history, balancing noise reduction against ghosting on motion.
 * Disocclusion / large-motion handling is the host's responsibility (reset
 * history when reprojection fails); for the one-shot CPU-backed dispatcher the
 * first frame uses α = 1 (no history).
 */
export const BMFR_DEFAULT_TEMPORAL_ALPHA = 0.2 as const;

/**
 * Characteristic world-space scale used to normalise block-local positions
 * before the squared feature terms are formed. The fit subtracts the block's
 * mean position and divides by this scale so `p` lands roughly in [-1, 1] and
 * `p²` in [0, 1], keeping the normal matrix well-conditioned regardless of the
 * scene's absolute world units. Hosts with very large or very small scenes
 * override via `BmfrWebGPUOptions.positionScale`.
 */
export const BMFR_DEFAULT_POSITION_SCALE = 4.0 as const;
