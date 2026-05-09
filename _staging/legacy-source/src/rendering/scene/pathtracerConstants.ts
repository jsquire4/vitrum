/**
 * Path-tracer convergence settings — single source of truth for both the
 * `<Pathtracer>` wrapper props in `PathTracingLayer.tsx` and the e2e timing
 * gate in `10-pathtracing-timing.spec.ts`. Keeping the value in one place
 * means bumping the convergence target updates the gate automatically.
 *
 * Tuning rationale documented in PathTracingLayer.tsx and the
 * project_m6_pt_perf_tuning memory note.
 */

export const PT_TARGET_SAMPLES = 192;
/**
 * Per-scene PT sample-target tiers (S6-T10). Shaded fixtures (table lamp,
 * pendant lamp, ceiling flush) need higher sample counts because emissive-
 * shade caustics are sampled WITHOUT MIS in three-gpu-pathtracer 0.0.23 —
 * boost converges within the same time budget.
 */
export const PT_TARGET_SAMPLES_BASE     = 192;
export const PT_TARGET_SAMPLES_FIXTURES = 240;
export const PT_BOUNCES = 5;
export const PT_FILTERED_GLOSSY_FACTOR = 0.25;
export const PT_RESOLUTION_FACTOR = 1.0;
export const PT_LOW_RES_SCALE = 0.25;

/**
 * Per-scene PT preview-quality time budgets. Each spec asserts its scene
 * converges within these.
 *   honeycomb  — bare panel + sun-catcher (10-pathtracing-timing)
 *   lightbox   — panel + lightbox + perimeter LED + tea light (S4-T22)
 *   full room  — panel + room asset + 7 default fixtures (S5-T7, S6-T11)
 */
export const PT_HONEYCOMB_TIMING_BUDGET_MS  = 60_000;
export const PT_LIGHTBOX_TIMING_BUDGET_MS   = 90_000;
export const PT_FULL_SCENE_TIMING_BUDGET_MS = 120_000;

/**
 * S7-T7: Final-render mode. PT_PREVIEW is the editor preview (the
 * existing convergence target); PT_FINAL is the share-quality output —
 * sharper caustics + lower-noise shading at a 240s budget.
 *
 * The "Final render" UI checkbox in the PT control swaps PT_PREVIEW →
 * PT_FINAL via usePTPipelineConfig().
 */
export interface PTPipelineConfig {
  samples: number;
  bounces: number;
  filteredGlossyFactor: number;
  budgetMs: number;
}

export const PT_PREVIEW: PTPipelineConfig = {
  samples: PT_TARGET_SAMPLES_BASE,
  bounces: PT_BOUNCES,
  filteredGlossyFactor: PT_FILTERED_GLOSSY_FACTOR,
  budgetMs: PT_HONEYCOMB_TIMING_BUDGET_MS,
};

/** Phase 5: PT_FINAL bumped for photorealism convergence. With area-
 *  light sun (Phase 1.1) introducing a real solid-angle for sun NEE,
 *  and outdoor HDRIs (Phase 1.2) providing rich HDR sampling, SDS
 *  caustic paths converge cleaner than under the directional sun but
 *  also see more variance per sample. samples 2048 (was 1536) closes
 *  the residual noise floor; 900s budget (was 480s) accommodates a
 *  hero 2K render at the new sample count. */
export const PT_FINAL: PTPipelineConfig = {
  samples: PT_TARGET_SAMPLES_BASE * 11, // 2112 — head-room above the 2048 calibration target
  bounces: PT_BOUNCES + 5,              // 10 bounces — Phase 4 normalMap-perturbed shadow rays
                                        // benefit from one extra indirect bounce; harmless when
                                        // Phase 4 isn't yet active (still renders cleaner SDSDS)
  filteredGlossyFactor: 0,
  budgetMs: 900_000,                    // 15-minute budget for a hero 2K render
};
