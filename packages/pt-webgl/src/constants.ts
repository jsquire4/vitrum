import type { FrameQualitySettings } from '@vitrum/core';

export const PT_TARGET_SAMPLES = 192;

/** Per-scene PT sample-target tiers (shaded fixtures need higher counts
 *  because emissive-shade caustics are sampled without MIS in
 *  three-gpu-pathtracer 0.0.23 — boost converges within the same time budget). */
export const PT_TARGET_SAMPLES_BASE     = 192;
export const PT_TARGET_SAMPLES_FIXTURES = 240;

/**
 * Sprint 1 (Phase 6): Preview uses 3 bounces for ~25–40 fps interactive on
 * desktop. Final uses 10 bounces for production hero renders.
 *
 * PT_BOUNCES is kept as a backward-compat alias for PT_PREVIEW_BOUNCES.
 * New code should reference PT_PREVIEW_BOUNCES or PT_FINAL_BOUNCES directly.
 */
export const PT_PREVIEW_BOUNCES = 3;
export const PT_FINAL_BOUNCES   = 10;
/** @deprecated Use PT_PREVIEW_BOUNCES. Kept for backward compatibility. */
export const PT_BOUNCES = PT_PREVIEW_BOUNCES;

export const PT_FILTERED_GLOSSY_FACTOR = 0.25;
/** Full-resolution factor for hero / final-quality frames (1.0× viewport). */
export const PT_RESOLUTION_FACTOR = 1.0;
/**
 * Phase 6 Sprint 1: preview path traces at 0.5× width/height vs `FrameInput.viewport`
 * before bilinear resolve — targets ~25–40 fps desktop interactive budgets.
 */
export const PT_PREVIEW_RESOLUTION_FACTOR = 0.5;
export const PT_LOW_RES_SCALE = 0.25;

// Timing budgets (PT_HONEYCOMB_TIMING_BUDGET_MS etc.) are intentionally not
// extracted — they are e2e test gate parameters, not engine configuration.

/** Editor preview quality: lower sample count, glossy filtering on. Pass as
 *  `FrameInput.quality` for interactive PT editing.
 *
 *  Sprint 1 (Phase 6): bounces 3, resolutionFactor 0.5 for interactive preview
 *  (~25–40 fps desktop). PT_FINAL_OPTIONS stays 10 bounces / 1.0× resolution. */
export const PT_PREVIEW_OPTIONS: Partial<FrameQualitySettings> = {
  samplesTarget:        PT_TARGET_SAMPLES_BASE,
  bounces:              PT_PREVIEW_BOUNCES,
  resolutionFactor:     PT_PREVIEW_RESOLUTION_FACTOR,
  filteredGlossyFactor: PT_FILTERED_GLOSSY_FACTOR,
} as const;

/** Hero / final-render quality: 2112 samples, 10 bounces, no glossy filter.
 *  Phase 4 normalMap-perturbed shadow rays benefit from the extra indirect
 *  bounce; harmless when Phase 4 isn't active (renders cleaner SDSDS). */
export const PT_FINAL_OPTIONS: Partial<FrameQualitySettings> = {
  samplesTarget:        PT_TARGET_SAMPLES_BASE * 11, // 2112 — head-room above the 2048 calibration target
  bounces:              PT_FINAL_BOUNCES,            // 10 bounces
  filteredGlossyFactor: 0,
} as const;
