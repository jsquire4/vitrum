/**
 * Matches HDR accumulation semantics: α at or below this threshold means **no samples**
 * for divide-by-α readback and tile-variance luminance (see adaptiveTileWeights GLSL).
 */
export const ZERO_SAMPLE_COUNT_EPSILON = 1e-6;
