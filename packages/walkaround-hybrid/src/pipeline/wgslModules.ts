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
  BMFR_WGSL,
} from '@vitrum/shared-denoisers';
import { LUMINANCE_WGSL, OCTAHEDRAL_CORE_WGSL } from '@vitrum/shared-samplers';

import { COMMON_MODULE } from '../shaders/common.wgsl.js';
// T9-stepA — the eleven focused modules split out of `common`. `common`
// itself `requires` these in canonical order; they are registered here so
// the composer can resolve those `requires` names.
import { WALKAROUND_UBO_MODULE } from '../shaders/walkaroundUbo.wgsl.js';
import { SCENE_TRAVERSAL_MODULE } from '../shaders/sceneTraversal.wgsl.js';
import { RESERVOIR_DI_MODULE } from '../shaders/reservoirDi.wgsl.js';
import { RESERVOIR_GI_MODULE } from '../shaders/reservoirGi.wgsl.js';
import { SHARED_PRIMITIVES_MODULE } from '../shaders/sharedPrimitives.wgsl.js';
import { GGX_BRDF_MODULE } from '../shaders/ggxBrdf.wgsl.js';
import { MATERIAL_DECODE_MODULE } from '../shaders/materialDecode.wgsl.js';
import { MATERIAL_ATLAS_MODULE } from '../shaders/materialAtlas.wgsl.js';
import { EMITTER_SAMPLING_MODULE } from '../shaders/emitterSampling.wgsl.js';
import { JACOBIAN_SHIFT_MODULE } from '../shaders/jacobianShift.wgsl.js';
import { GRIS_REUSE_MODULE } from '../shaders/grisReuse.wgsl.js';
import { CAMERA_RAYS_MODULE } from '../shaders/cameraRays.wgsl.js';
import { WELFORD_TAIL_MODULE } from '../shaders/welfordTail.wgsl.js';
import { SURFACE_TEXTURES_MODULE } from '../shaders/surfaceTextures.wgsl.js';
import { RESTIR_PHAT_MODULE } from '../shaders/restirPHat.wgsl.js';
import { RESTIR_CAST_PRIMARY_MODULE } from '../shaders/restirCastPrimary.wgsl.js';
import { RESTIR_GI_MATERIAL_MODULE } from '../shaders/restirGiMaterial.wgsl.js';
import { LIGHT_TREE_MODULE } from '../shaders/lightTree.wgsl.js';
import { REGIR_MODULE, REGIR_BUILD_MODULE } from '../shaders/regir.wgsl.js';
import { RIS_MODULE } from '../shaders/ris.wgsl.js';
import { TEMPORAL_MODULE } from '../shaders/temporal.wgsl.js';
import { SPATIAL_MODULE } from '../shaders/spatial.wgsl.js';
import { SHADE_MODULE } from '../shaders/shade.wgsl.js';
import { STAINED_GLASS_SHADE_MODULE } from '../shaders/stainedGlassShade.wgsl.js';
import { MOTION_VECTORS_MODULE } from '../shaders/motionVectors.wgsl.js';
import { SAMPLE_CASCADE_C0_MODULE } from '../shaders/sampleCascadeC0.wgsl.js';
import { RIS_GI_MODULE } from '../shaders/risGi.wgsl.js';
import { TEMPORAL_GI_MODULE, TEMPORAL_GI_GRIS_MODULE } from '../shaders/temporalGi.wgsl.js';
import { SPATIAL_GI_MODULE, SPATIAL_GI_GRIS_MODULE } from '../shaders/spatialGi.wgsl.js';
import { SPATIAL_GI_COMMON_MODULE } from '../shaders/spatialGiCommon.wgsl.js';
import { WELFORD_TEMPORAL_MODULE } from '../shaders/welfordTemporal.wgsl.js';
import { SAMPLE_BUDGET_MODULE } from '../shaders/sampleBudget.wgsl.js';
import { RESOLVE_MODULE } from '../shaders/resolve.wgsl.js';
import { CB_PREFILL_MODULE } from '../shaders/cbPrefill.wgsl.js';
import { SCREEN_COORD_HELPERS_MODULE } from '../shaders/screenCoordHelpers.wgsl.js';
import { GTAO_COMMON_MODULE } from '../shaders/gtaoCommon.wgsl.js';
import { GTAO_MODULE } from '../shaders/gtao.wgsl.js';
import { GTAO_UPSAMPLE_MODULE } from '../shaders/gtaoUpsample.wgsl.js';
import { INDIRECT_COMBINE_MODULE } from '../shaders/indirectCombine.wgsl.js';
import { INDIRECT_TEMPORAL_ACCUM_MODULE } from '../shaders/indirectTemporalAccum.wgsl.js';
import { TRANSPARENT_OIT_MODULE } from '../shaders/transparentOit.wgsl.js';
import { COMPOSITE_VERT_MODULE, COMPOSITE_FRAG_MODULE } from '../shaders/composite.wgsl.js';
import { DDGI_SAMPLE_MODULE, DDGI_GRID_UBO_MODULE } from '../ddgi/ddgiSampleWgsl.js';
import { ENVIRONMENT_SAMPLE_MODULE } from '../shaders/environmentSample.wgsl.js';
import { PPG_TREE_LAYOUT_MODULE } from '../ppg/ppgTreeLayout.wgsl.js';
import { PPG_UPDATE_MODULE } from '../ppg/ppgUpdate.wgsl.js';
import { PPG_PDF_MODULE } from '../ppg/ppgPdf.wgsl.js';
import { NEURAL_PACK_MODULE } from '../shaders/neuralPack.wgsl.js';
import { NEURAL_UNPACK_MODULE } from '../shaders/neuralUnpack.wgsl.js';
import type { WgslModule } from './wgslComposer.js';

// Re-exports for consumers that pull individual modules by name (e.g.,
// pipelineCompiler.ts iterates them when assembling individual compute
// pipelines). The five symbols that knip flagged as dead re-exports
// (COMMON_MODULE, SURFACE_TEXTURES_MODULE, RESTIR_PHAT_MODULE,
// RESTIR_CAST_PRIMARY_MODULE, DDGI_SAMPLE_MODULE) are intentionally
// omitted here — they're consumed only via the WGSL_MODULES Map below,
// so the named re-export was unused weight on the public surface.
export {
  RIS_MODULE,
  REGIR_BUILD_MODULE,
  TEMPORAL_MODULE,
  SPATIAL_MODULE,
  SHADE_MODULE,
  MOTION_VECTORS_MODULE,
  RIS_GI_MODULE,
  TEMPORAL_GI_MODULE,
  TEMPORAL_GI_GRIS_MODULE,
  SPATIAL_GI_MODULE,
  SPATIAL_GI_GRIS_MODULE,
  WELFORD_TEMPORAL_MODULE,
  SAMPLE_BUDGET_MODULE,
  RESOLVE_MODULE,
  CB_PREFILL_MODULE,
  GTAO_MODULE,
  GTAO_UPSAMPLE_MODULE,
  INDIRECT_COMBINE_MODULE,
  INDIRECT_TEMPORAL_ACCUM_MODULE,
  TRANSPARENT_OIT_MODULE,
  COMPOSITE_VERT_MODULE,
  COMPOSITE_FRAG_MODULE,
  PPG_UPDATE_MODULE,
};

// ── Shared-denoisers wrappers ──────────────────────────────────────────────
// These wrap the raw-string exports from `@vitrum/shared-denoisers` in
// WgslModule envelopes so the include-graph can resolve them by name.
// shared-denoisers itself stays untouched.

/**
 * Canonical Rec.709 luminance helper from @vitrum/shared-samplers.
 * Modules can `requires: ['luminance']` to pull `fn luminance(c: vec3f)`
 * + `const LUM_W709` without the weight of the full `common` module.
 *
 * Note: `common` itself defines `fn luminance` for legacy reasons, so a
 * module that already requires 'common' should NOT also require 'luminance'
 * — that would emit two definitions and WGSL would reject the redefinition.
 */
const LUMINANCE_MODULE: WgslModule = {
  name: 'luminance',
  source: LUMINANCE_WGSL,
  requires: [],
};

/**
 * Canonical octahedral encode/decode pair from @vitrum/shared-samplers.
 * Consumers `requires: ['octahedralCore']` for `fn octEncode(dir: vec3f) -> vec2f`
 * and `fn octDecode(oct: vec2f) -> vec3f` (Cigolle et al. 2014 — A Survey
 * of Efficient Representations for Independent Unit Vectors).
 *
 * Note: `common` does NOT include octahedral, so modules that want it
 * must declare the dependency explicitly.
 */
const OCTAHEDRAL_CORE_MODULE: WgslModule = {
  name: 'octahedralCore',
  source: OCTAHEDRAL_CORE_WGSL,
  requires: [],
};

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

/** BMFR (Koskela 2019) per-block feature-regression kernel. Self-contained:
 *  declares its own UBO struct, feature-row + Householder-QR helpers, and the
 *  bmfrMain entry point. `requires: []`. */
export const BMFR_MODULE: WgslModule = {
  name: 'bmfr',
  source: BMFR_WGSL,
  requires: [],
};

/** Fragment-only — registered for completeness; never resolved via `requires`
 *  because the modules that need WelfordVariance template-interpolate the
 *  raw string into their own source. */
const WELFORD_VARIANCE_MODULE: WgslModule = {
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
  // T9-stepA — focused modules `common` aggregates (canonical order).
  [WALKAROUND_UBO_MODULE.name, WALKAROUND_UBO_MODULE],
  [SCENE_TRAVERSAL_MODULE.name, SCENE_TRAVERSAL_MODULE],
  [RESERVOIR_DI_MODULE.name, RESERVOIR_DI_MODULE],
  [RESERVOIR_GI_MODULE.name, RESERVOIR_GI_MODULE],
  [SHARED_PRIMITIVES_MODULE.name, SHARED_PRIMITIVES_MODULE],
  [GGX_BRDF_MODULE.name, GGX_BRDF_MODULE],
  [MATERIAL_DECODE_MODULE.name, MATERIAL_DECODE_MODULE],
  [MATERIAL_ATLAS_MODULE.name, MATERIAL_ATLAS_MODULE],
  [EMITTER_SAMPLING_MODULE.name, EMITTER_SAMPLING_MODULE],
  [JACOBIAN_SHIFT_MODULE.name, JACOBIAN_SHIFT_MODULE],
  // GRIS / ReSTIR-PT reconnection-shift + pairwise MIS (Lin et al. 2022).
  // Consumed by spatialGi / temporalGi when ubo.restirPtReuse == 1.
  [GRIS_REUSE_MODULE.name, GRIS_REUSE_MODULE],
  [CAMERA_RAYS_MODULE.name, CAMERA_RAYS_MODULE],
  [WELFORD_TAIL_MODULE.name, WELFORD_TAIL_MODULE],
  [LUMINANCE_MODULE.name, LUMINANCE_MODULE],
  [OCTAHEDRAL_CORE_MODULE.name, OCTAHEDRAL_CORE_MODULE],
  [WELFORD_VARIANCE_MODULE.name, WELFORD_VARIANCE_MODULE],

  // Walkaround-local shader helpers
  [SURFACE_TEXTURES_MODULE.name, SURFACE_TEXTURES_MODULE],
  [DDGI_SAMPLE_MODULE.name, DDGI_SAMPLE_MODULE],
  // D5.1+D5.2 — shared DDGIGridUBO struct + binding(3) + sampleDDGIAtPoint wrapper.
  // Requires ddgiSample; consumed by risGi, risGiNrc, shade.
  [DDGI_GRID_UBO_MODULE.name, DDGI_GRID_UBO_MODULE],
  // B3 — directional IBL env bindings + lookup/importance helpers (scene group
  // bindings 15-19). Required by ris/risGi/shade; bindings are runtime-gated by
  // envParams.hasEnv (0 ⇒ scalar-tint fallback, no-HDRI byte-identity).
  [ENVIRONMENT_SAMPLE_MODULE.name, ENVIRONMENT_SAMPLE_MODULE],

  // W2-C7+C9 — canonical ReSTIR-DI primitives
  [RESTIR_PHAT_MODULE.name, RESTIR_PHAT_MODULE],
  [RESTIR_CAST_PRIMARY_MODULE.name, RESTIR_CAST_PRIMARY_MODULE],
  [RESTIR_GI_MATERIAL_MODULE.name, RESTIR_GI_MATERIAL_MODULE],
  // Light-tree DI light-selection traversal (RIS-only @group(3) buffer).
  [LIGHT_TREE_MODULE.name, LIGHT_TREE_MODULE],
  // ReGIR grid sampling (RIS read path, reuses the combined @group(3) buffer)
  // + the grid-build kernel (its own @group(0) read_write binding).
  [REGIR_MODULE.name, REGIR_MODULE],
  [REGIR_BUILD_MODULE.name, REGIR_BUILD_MODULE],

  // ReSTIR-DI passes
  [RIS_MODULE.name, RIS_MODULE],
  [TEMPORAL_MODULE.name, TEMPORAL_MODULE],
  [SPATIAL_MODULE.name, SPATIAL_MODULE],
  [SHADE_MODULE.name, SHADE_MODULE],
  // T5 — stained-glass-specific lighting physics (opt-in via UBO flag).
  [STAINED_GLASS_SHADE_MODULE.name, STAINED_GLASS_SHADE_MODULE],
  [MOTION_VECTORS_MODULE.name, MOTION_VECTORS_MODULE],
  [SAMPLE_CASCADE_C0_MODULE.name, SAMPLE_CASCADE_C0_MODULE],

  // ReSTIR-GI passes. The GRIS (restirPtReuse ON) variants are separate
  // compile-roots, composed only when the host opts in — see
  // spatialGi.wgsl.ts / temporalGi.wgsl.ts headers + pipelineCompiler.ts.
  [RIS_GI_MODULE.name, RIS_GI_MODULE],
  [TEMPORAL_GI_MODULE.name, TEMPORAL_GI_MODULE],
  [TEMPORAL_GI_GRIS_MODULE.name, TEMPORAL_GI_GRIS_MODULE],
  [SPATIAL_GI_COMMON_MODULE.name, SPATIAL_GI_COMMON_MODULE],
  [SPATIAL_GI_MODULE.name, SPATIAL_GI_MODULE],
  [SPATIAL_GI_GRIS_MODULE.name, SPATIAL_GI_GRIS_MODULE],

  // Sprint 9 — adaptive sampling
  [WELFORD_TEMPORAL_MODULE.name, WELFORD_TEMPORAL_MODULE],
  [SAMPLE_BUDGET_MODULE.name, SAMPLE_BUDGET_MODULE],
  // D5.4 — screenCoordHelpers registered before resolve/cb-prefill which require it.
  [SCREEN_COORD_HELPERS_MODULE.name, SCREEN_COORD_HELPERS_MODULE],
  [RESOLVE_MODULE.name, RESOLVE_MODULE],
  [CB_PREFILL_MODULE.name, CB_PREFILL_MODULE],

  // Sprint 15 — GTAO
  [GTAO_COMMON_MODULE.name, GTAO_COMMON_MODULE],
  [GTAO_MODULE.name, GTAO_MODULE],
  [GTAO_UPSAMPLE_MODULE.name, GTAO_UPSAMPLE_MODULE],

  // Sprint 18 — indirect channel
  [INDIRECT_COMBINE_MODULE.name, INDIRECT_COMBINE_MODULE],
  [INDIRECT_TEMPORAL_ACCUM_MODULE.name, INDIRECT_TEMPORAL_ACCUM_MODULE],
  [TRANSPARENT_OIT_MODULE.name, TRANSPARENT_OIT_MODULE],

  // Composite (vert + frag)
  [COMPOSITE_VERT_MODULE.name, COMPOSITE_VERT_MODULE],
  [COMPOSITE_FRAG_MODULE.name, COMPOSITE_FRAG_MODULE],

  // PPG (Müller 2017 — opt-in, compiled only when ppgEnabled)
  [PPG_TREE_LAYOUT_MODULE.name, PPG_TREE_LAYOUT_MODULE],
  [PPG_UPDATE_MODULE.name, PPG_UPDATE_MODULE],
  // PPG guided-sampling pdf-eval + sampler for gi-ris (always available so
  // risGi's include-graph resolves; the bindings it declares are gated by
  // ubo.ppgEnabled at runtime).
  [PPG_PDF_MODULE.name, PPG_PDF_MODULE],

  // Shared-denoisers wrappers
  [ATROUS_MODULE.name, ATROUS_MODULE],
  [TEMPORAL_ACCUM_MODULE.name, TEMPORAL_ACCUM_MODULE],
  [ATROUS_VARIANCE_MODULE.name, ATROUS_VARIANCE_MODULE],
  [SVGF_REPROJECTION_MODULE.name, SVGF_REPROJECTION_MODULE],
  [SVGF_VARIANCE_FROM_MOMENTS_MODULE.name, SVGF_VARIANCE_FROM_MOMENTS_MODULE],
  [SVGF_7X7_SPATIAL_FALLBACK_MODULE.name, SVGF_7X7_SPATIAL_FALLBACK_MODULE],
  [BMFR_MODULE.name, BMFR_MODULE],

  // Neural denoiser pack/unpack compute shaders (Issue 2 — extracted from
  // NeuralDenoiser.initialize inline literals; character-identical source).
  [NEURAL_PACK_MODULE.name, NEURAL_PACK_MODULE],
  [NEURAL_UNPACK_MODULE.name, NEURAL_UNPACK_MODULE],
]);
