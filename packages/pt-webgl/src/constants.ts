import type { FrameQualitySettings } from '@vitrum/core';

export const PT_TARGET_SAMPLES = 192;

/** Per-scene PT sample-target tiers (shaded fixtures need higher counts
 *  because emissive-shade caustics are sampled without MIS in
 *  three-gpu-pathtracer 0.0.23 — boost converges within the same time budget). */
export const PT_TARGET_SAMPLES_BASE     = 192;
export const PT_TARGET_SAMPLES_FIXTURES = 240;
export const PT_BOUNCES = 5;
export const PT_FILTERED_GLOSSY_FACTOR = 0.25;
export const PT_RESOLUTION_FACTOR = 1.0;
export const PT_LOW_RES_SCALE = 0.25;

// Timing budgets (PT_HONEYCOMB_TIMING_BUDGET_MS etc.) are intentionally not
// extracted — they are e2e test gate parameters, not engine configuration.

/** Editor preview quality: lower sample count, glossy filtering on. Pass as
 *  `FrameInput.quality` for interactive PT editing. */
export const PT_PREVIEW_OPTIONS: Partial<FrameQualitySettings> = {
  samplesTarget:        PT_TARGET_SAMPLES_BASE,
  bounces:              PT_BOUNCES,
  filteredGlossyFactor: PT_FILTERED_GLOSSY_FACTOR,
} as const;

/** Hero / final-render quality: 2112 samples, 10 bounces, no glossy filter.
 *  Phase 4 normalMap-perturbed shadow rays benefit from the extra indirect
 *  bounce; harmless when Phase 4 isn't active (renders cleaner SDSDS). */
export const PT_FINAL_OPTIONS: Partial<FrameQualitySettings> = {
  samplesTarget:        PT_TARGET_SAMPLES_BASE * 11, // 2112 — head-room above the 2048 calibration target
  bounces:              PT_BOUNCES + 5,              // 10 bounces
  filteredGlossyFactor: 0,
} as const;
