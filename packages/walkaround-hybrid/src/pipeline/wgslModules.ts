/**
 * Central WGSL module registry — the single source of truth for the
 * declarative include-graph introduced by W1-R6.
 *
 * Every shader-string `*.wgsl.ts` file in this package exports a sibling
 * `*_MODULE: WgslModule` declaration. This file collects them all into
 * `WGSL_MODULES`, which `composeWgsl` consults to resolve `requires`.
 *
 * Shared-denoisers WGSL is wrapped here (not in `@vitrum/shared-denoisers`)
 * so the shared package stays raw-string-only and the include-graph
 * concerns live entirely in walkaround-hybrid. The `requires` arrays here
 * reflect the historical concat patterns from pipelineCompiler.ts /
 * atrousVariance.ts / svgfReal.ts pre-R6:
 *   - ATROUS_WGSL was prepended with COMMON_WGSL  → requires: ['common']
 *   - TEMPORAL_ACCUM_WGSL was NOT concatenated   → requires: []
 *   - ATROUS_VARIANCE_WGSL was self-contained (carries its own PI/INV_PI/
 *     LUM_W/WelfordVariance) — the anti-duplication-by-comment at
 *     pipelineCompiler.ts:131 (pre-R6) becomes structural here:
 *     requires: [].
 *   - SVGF_REPROJECTION_WGSL, SVGF_VARIANCE_FROM_MOMENTS_WGSL,
 *     SVGF_7X7_SPATIAL_FALLBACK_WGSL are all self-contained: pre-R6 they
 *     were compiled standalone (no COMMON_WGSL prepend in svgfReal.ts).
 *     → requires: [].
 *   - WELFORD_VARIANCE_WGSL is a fragment that other modules
 *     template-interpolate (it is part of COMMON_WGSL's source string and
 *     of SAMPLE_BUDGET_WGSL's source string already). It is registered
 *     here for completeness so consumers of just-the-WelfordVariance
 *     struct could opt in, but the include-graph never resolves it via
 *     `requires` today.
 */

import {
  ATROUS_WGSL,
  ATROUS_VARIANCE_WGSL,
  TEMPORAL_ACCUM_WGSL,
  WELFORD_VARIANCE_WGSL,
  SVGF_REPROJECTION_WGSL,
  SVGF_VARIANCE_FROM_MOMENTS_WGSL,
  SVGF_7X7_SPATIAL_FALLBACK_WGSL,
} from '@vitrum/shared-denoisers';

import { COMMON_MODULE } from '../shaders/common.wgsl.js';
import { SURFACE_TEXTURES_MODULE } from '../shaders/surfaceTextures.wgsl.js';
import { STAINED_GLASS_SURFACE_MODS_MODULE } from '../shaders/stained-glass/surfaceMods.wgsl.js';
import { GLASS_VISIBILITY_MODULE } from '../shaders/glassVisibility.wgsl.js';
import { RIS_MODULE } from '../shaders/ris.wgsl.js';
import { TEMPORAL_MODULE } from '../shaders/temporal.wgsl.js';
import { SPATIAL_MODULE } from '../shaders/spatial.wgsl.js';
import { SHADE_MODULE } from '../shaders/shade.wgsl.js';
import { RIS_GI_MODULE } from '../shaders/risGi.wgsl.js';
import { TEMPORAL_GI_MODULE } from '../shaders/temporalGi.wgsl.js';
import { SPATIAL_GI_MODULE } from '../shaders/spatialGi.wgsl.js';
import { WELFORD_TEMPORAL_MODULE } from '../shaders/welfordTemporal.wgsl.js';
import { SAMPLE_BUDGET_MODULE } from '../shaders/sampleBudget.wgsl.js';
import { RESOLVE_MODULE } from '../shaders/resolve.wgsl.js';
import { GTAO_MODULE } from '../shaders/gtao.wgsl.js';
import { GTAO_UPSAMPLE_MODULE } from '../shaders/gtaoUpsample.wgsl.js';
import { INDIRECT_COMBINE_MODULE } from '../shaders/indirectCombine.wgsl.js';
import { INDIRECT_TEMPORAL_ACCUM_MODULE } from '../shaders/indirectTemporalAccum.wgsl.js';
import { COMPOSITE_VERT_MODULE, COMPOSITE_FRAG_MODULE } from '../shaders/composite.wgsl.js';
import { DDGI_SAMPLE_MODULE } from '../ddgi/ddgiSampleWgsl.js';
import { PPG_UPDATE_MODULE } from '../ppg/ppgUpdate.wgsl.js';
import { PPG_GUIDE_MODULE } from '../ppg/ppgGuide.wgsl.js';
import type { WgslModule } from './wgslComposer.js';

// Re-exports so consumers can import every module from a single, central index.
export {
  COMMON_MODULE,
  SURFACE_TEXTURES_MODULE,
  STAINED_GLASS_SURFACE_MODS_MODULE,
  GLASS_VISIBILITY_MODULE,
  RIS_MODULE,
  TEMPORAL_MODULE,
  SPATIAL_MODULE,
  SHADE_MODULE,
  RIS_GI_MODULE,
  TEMPORAL_GI_MODULE,
  SPATIAL_GI_MODULE,
  WELFORD_TEMPORAL_MODULE,
  SAMPLE_BUDGET_MODULE,
  RESOLVE_MODULE,
  GTAO_MODULE,
  GTAO_UPSAMPLE_MODULE,
  INDIRECT_COMBINE_MODULE,
  INDIRECT_TEMPORAL_ACCUM_MODULE,
  COMPOSITE_VERT_MODULE,
  COMPOSITE_FRAG_MODULE,
  DDGI_SAMPLE_MODULE,
  PPG_UPDATE_MODULE,
  PPG_GUIDE_MODULE,
};

// ── Shared-denoisers wrappers ──────────────────────────────────────────────
// These wrap the raw-string exports from `@vitrum/shared-denoisers` in
// WgslModule envelopes so the include-graph can resolve them by name.
// shared-denoisers itself stays untouched.

/** Pre-R6 concat: `COMMON_WGSL + ATROUS_WGSL` (pipelineCompiler.ts:112). */
export const ATROUS_MODULE: WgslModule = {
  name: 'atrous',
  source: ATROUS_WGSL,
  requires: ['common'],
};

/** Pre-R6 concat: `TEMPORAL_ACCUM_WGSL` standalone (pipelineCompiler.ts:295). */
export const TEMPORAL_ACCUM_MODULE: WgslModule = {
  name: 'temporalAccum',
  source: TEMPORAL_ACCUM_WGSL,
  requires: [],
};

/** Pre-R6: ATROUS_VARIANCE_WGSL is self-contained — declares its own PI,
 *  INV_PI, LUM_W, and WelfordVariance struct. Pre-R6 atrousVariance.ts:148
 *  had an anti-duplication-by-comment explaining why COMMON_WGSL was NOT
 *  prepended; W1-R6 turns that comment into structure: `requires: []`. */
export const ATROUS_VARIANCE_MODULE: WgslModule = {
  name: 'atrousVariance',
  source: ATROUS_VARIANCE_WGSL,
  requires: [],
};

/** Pre-R6 svgfReal.ts:72-80: all three SVGF entries compiled standalone
 *  (no COMMON_WGSL prepend). Each is self-contained. */
export const SVGF_REPROJECTION_MODULE: WgslModule = {
  name: 'svgfReprojection',
  source: SVGF_REPROJECTION_WGSL,
  requires: [],
};
export const SVGF_VARIANCE_FROM_MOMENTS_MODULE: WgslModule = {
  name: 'svgfVarianceFromMoments',
  source: SVGF_VARIANCE_FROM_MOMENTS_WGSL,
  requires: [],
};
export const SVGF_7X7_SPATIAL_FALLBACK_MODULE: WgslModule = {
  name: 'svgf7x7SpatialFallback',
  source: SVGF_7X7_SPATIAL_FALLBACK_WGSL,
  requires: [],
};

/** Fragment-only — registered for completeness; never resolved via `requires`
 *  because the modules that need WelfordVariance template-interpolate the
 *  raw string into their own source. */
export const WELFORD_VARIANCE_MODULE: WgslModule = {
  name: 'welfordVariance',
  source: WELFORD_VARIANCE_WGSL,
  requires: [],
};

// ── The registry ──────────────────────────────────────────────────────────
/**
 * Single-source-of-truth registry. `composeWgsl(rootModule, WGSL_MODULES)`
 * resolves every `requires` name against this map.
 */
export const WGSL_MODULES: ReadonlyMap<string, WgslModule> = new Map<string, WgslModule>([
  // Foundation
  [COMMON_MODULE.name, COMMON_MODULE],
  [WELFORD_VARIANCE_MODULE.name, WELFORD_VARIANCE_MODULE],

  // Walkaround-local shader helpers
  [SURFACE_TEXTURES_MODULE.name, SURFACE_TEXTURES_MODULE],
  // W7-H6 split: stained-glass surface mods (host) + glass visibility (library).
  // The eventual home of `surfaceMods` is the `@vitrum/stained-glass-extensions`
  // package (W3-D2+D3); it stays here until that lands.
  [STAINED_GLASS_SURFACE_MODS_MODULE.name, STAINED_GLASS_SURFACE_MODS_MODULE],
  [GLASS_VISIBILITY_MODULE.name, GLASS_VISIBILITY_MODULE],
  [DDGI_SAMPLE_MODULE.name, DDGI_SAMPLE_MODULE],

  // ReSTIR-DI passes
  [RIS_MODULE.name, RIS_MODULE],
  [TEMPORAL_MODULE.name, TEMPORAL_MODULE],
  [SPATIAL_MODULE.name, SPATIAL_MODULE],
  [SHADE_MODULE.name, SHADE_MODULE],

  // ReSTIR-GI passes
  [RIS_GI_MODULE.name, RIS_GI_MODULE],
  [TEMPORAL_GI_MODULE.name, TEMPORAL_GI_MODULE],
  [SPATIAL_GI_MODULE.name, SPATIAL_GI_MODULE],

  // Sprint 9 — adaptive sampling
  [WELFORD_TEMPORAL_MODULE.name, WELFORD_TEMPORAL_MODULE],
  [SAMPLE_BUDGET_MODULE.name, SAMPLE_BUDGET_MODULE],
  [RESOLVE_MODULE.name, RESOLVE_MODULE],

  // Sprint 15 — GTAO
  [GTAO_MODULE.name, GTAO_MODULE],
  [GTAO_UPSAMPLE_MODULE.name, GTAO_UPSAMPLE_MODULE],

  // Sprint 18 — indirect channel
  [INDIRECT_COMBINE_MODULE.name, INDIRECT_COMBINE_MODULE],
  [INDIRECT_TEMPORAL_ACCUM_MODULE.name, INDIRECT_TEMPORAL_ACCUM_MODULE],

  // Composite (vert + frag)
  [COMPOSITE_VERT_MODULE.name, COMPOSITE_VERT_MODULE],
  [COMPOSITE_FRAG_MODULE.name, COMPOSITE_FRAG_MODULE],

  // PPG (Müller 2017 — opt-in, compiled only when ppgEnabled)
  [PPG_UPDATE_MODULE.name, PPG_UPDATE_MODULE],
  [PPG_GUIDE_MODULE.name, PPG_GUIDE_MODULE],

  // Shared-denoisers wrappers
  [ATROUS_MODULE.name, ATROUS_MODULE],
  [TEMPORAL_ACCUM_MODULE.name, TEMPORAL_ACCUM_MODULE],
  [ATROUS_VARIANCE_MODULE.name, ATROUS_VARIANCE_MODULE],
  [SVGF_REPROJECTION_MODULE.name, SVGF_REPROJECTION_MODULE],
  [SVGF_VARIANCE_FROM_MOMENTS_MODULE.name, SVGF_VARIANCE_FROM_MOMENTS_MODULE],
  [SVGF_7X7_SPATIAL_FALLBACK_MODULE.name, SVGF_7X7_SPATIAL_FALLBACK_MODULE],
]);
