/**
 * Central numeric policy for SVGF host paths and WGSL (see `wgsl/svgf.wgsl.ts`).
 *
 * `SVGF_TEMPORAL_VARIANCE_MIN_FRAME_COUNT` must stay in sync with the WGSL
 * constant `SVGF_TEMPORAL_VARIANCE_MIN_FRAMES` (injected from this value).
 */

/** Frames required before the variance pass reads temporal Welford data (matches WGSL branch). */
export const SVGF_TEMPORAL_VARIANCE_MIN_FRAME_COUNT = 4 as const;

/** Default à-trous dispatch count when callers omit `atrousIterations`.
 *  Reduced from 5 to 3: the iter-4 pass's step=16 with a 5x5 kernel reaches
 *  ±32 pixels, which over-smoothed soft-shadow penumbras into stair-step
 *  blocks on flat surfaces (visible as blocky shadows on the Cornell floor
 *  under area-light boxes). Three iterations (steps 1, 2, 4) give an
 *  effective footprint ~14 pixels — enough to clean ReSTIR-DI noise without
 *  smearing penumbra detail. */
export const SVGF_DEFAULT_ATROUS_ITERATIONS = 3 as const;

/** Hard cap on à-trous iterations per `runSvgfWebGPU` call (GPU time guardrail). */
export const SVGF_MAX_ATROUS_ITERATIONS = 12 as const;

/**
 * Upper bound for `frameCount` packed into SVGF variance UBO (CPU/host guardrail).
 * Values above this are saturated before packing; keeps accidental URL spikes bounded.
 */
export const SVGF_FRAME_COUNT_INPUT_GUARD_MAX = 1_000_000 as const;
