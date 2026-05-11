/**
 * Central numeric policy for SVGF host paths and WGSL (see `wgsl/svgf.wgsl.ts`).
 *
 * `SVGF_TEMPORAL_VARIANCE_MIN_FRAME_COUNT` must stay in sync with the WGSL
 * constant `SVGF_TEMPORAL_VARIANCE_MIN_FRAMES` (injected from this value).
 */

/** Frames required before the variance pass reads temporal Welford data (matches WGSL branch). */
export const SVGF_TEMPORAL_VARIANCE_MIN_FRAME_COUNT = 4 as const;

/** Default à-trous dispatch count when callers omit `atrousIterations`. */
export const SVGF_DEFAULT_ATROUS_ITERATIONS = 5 as const;

/** Hard cap on à-trous iterations per `runSvgfWebGPU` call (GPU time guardrail). */
export const SVGF_MAX_ATROUS_ITERATIONS = 12 as const;

/**
 * Upper bound for `frameCount` packed into SVGF variance UBO (CPU/host guardrail).
 * Values above this are saturated before packing; keeps accidental URL spikes bounded.
 */
export const SVGF_FRAME_COUNT_INPUT_GUARD_MAX = 1_000_000 as const;
