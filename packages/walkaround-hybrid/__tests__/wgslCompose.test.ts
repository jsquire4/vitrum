/**
 * W1-R6 — declarative WGSL include-graph composer tests.
 *
 * Three test groups:
 *  1. `composeWgsl` semantics — cycle detection, dedup, unknown require,
 *     dep-first ordering. Independent of any real shader source.
 *  2. Module declarations — every entry in `WGSL_MODULES` resolves cleanly
 *     and references only declared names.
 *  3. Bit-identical gate — every compose() call mirrors the pre-R6 concat
 *     pattern for the corresponding pipeline. This is the BIT-IDENTICAL
 *     guarantee the W1-R6 plan calls out as mandatory.
 */

import { describe, expect, it } from 'vitest';

import {
  ATROUS_WGSL,
  ATROUS_VARIANCE_WGSL,
  SVGF_7X7_SPATIAL_FALLBACK_WGSL,
  SVGF_REPROJECTION_WGSL,
  SVGF_VARIANCE_FROM_MOMENTS_WGSL,
  TEMPORAL_ACCUM_WGSL,
} from '@vitrum/shared-denoisers';
import { LUMINANCE_WGSL, OCTAHEDRAL_CORE_WGSL } from '@vitrum/shared-samplers';

import { composeWgsl, type WgslModule } from '../src/pipeline/wgslComposer.js';
import {
  ATROUS_MODULE,
  ATROUS_VARIANCE_MODULE,
  COMPOSITE_FRAG_MODULE,
  COMPOSITE_VERT_MODULE,
  GTAO_MODULE,
  GTAO_UPSAMPLE_MODULE,
  INDIRECT_COMBINE_MODULE,
  INDIRECT_TEMPORAL_ACCUM_MODULE,
  MOTION_VECTORS_MODULE,
  PPG_UPDATE_MODULE,
  RESOLVE_MODULE,
  RIS_GI_MODULE,
  RIS_MODULE,
  SAMPLE_BUDGET_MODULE,
  SHADE_MODULE,
  SPATIAL_GI_MODULE,
  SPATIAL_GI_GRIS_MODULE,
  SPATIAL_MODULE,
  SVGF_7X7_SPATIAL_FALLBACK_MODULE,
  SVGF_REPROJECTION_MODULE,
  SVGF_VARIANCE_FROM_MOMENTS_MODULE,
  TEMPORAL_ACCUM_MODULE,
  TEMPORAL_GI_MODULE,
  TEMPORAL_GI_GRIS_MODULE,
  TEMPORAL_MODULE,
  WELFORD_TEMPORAL_MODULE,
  WGSL_MODULES,
} from '../src/pipeline/wgslModules.js';
import { buildSymbolUniverse, checkCrossModuleResolution } from './wgslIdentResolution.js';
import { COMMON_WGSL } from '../src/shaders/common.wgsl.js';
// T9-stepC — focused-module sources, for the narrowed-pass byte assertions.
import { WALKAROUND_UBO_WGSL } from '../src/shaders/walkaroundUbo.wgsl.js';
import { SCENE_TRAVERSAL_WGSL } from '../src/shaders/sceneTraversal.wgsl.js';
import { RESERVOIR_GI_WGSL } from '../src/shaders/reservoirGi.wgsl.js';
import { SHARED_PRIMITIVES_WGSL } from '../src/shaders/sharedPrimitives.wgsl.js';
import { MATERIAL_DECODE_WGSL } from '../src/shaders/materialDecode.wgsl.js';
import { CAMERA_RAYS_WGSL } from '../src/shaders/cameraRays.wgsl.js';
import { JACOBIAN_SHIFT_WGSL } from '../src/shaders/jacobianShift.wgsl.js';
import { GRIS_REUSE_WGSL } from '../src/shaders/grisReuse.wgsl.js';
import { WELFORD_TAIL_WGSL } from '../src/shaders/welfordTail.wgsl.js';
import { MOTION_VECTORS_WGSL } from '../src/shaders/motionVectors.wgsl.js';
import { COMPOSITE_FRAG_WGSL, COMPOSITE_VERT_WGSL } from '../src/shaders/composite.wgsl.js';
import { GTAO_COMMON_WGSL } from '../src/shaders/gtaoCommon.wgsl.js';
import { GTAO_WGSL } from '../src/shaders/gtao.wgsl.js';
import { GTAO_UPSAMPLE_WGSL } from '../src/shaders/gtaoUpsample.wgsl.js';
import { INDIRECT_COMBINE_WGSL } from '../src/shaders/indirectCombine.wgsl.js';
import { INDIRECT_TEMPORAL_ACCUM_WGSL } from '../src/shaders/indirectTemporalAccum.wgsl.js';
import { RESOLVE_WGSL } from '../src/shaders/resolve.wgsl.js';
import { SCREEN_COORD_HELPERS_WGSL } from '../src/shaders/screenCoordHelpers.wgsl.js';
import { RESTIR_PHAT_WGSL } from '../src/shaders/restirPHat.wgsl.js';
import { RESTIR_CAST_PRIMARY_WGSL } from '../src/shaders/restirCastPrimary.wgsl.js';
import { RIS_WGSL } from '../src/shaders/ris.wgsl.js';
import { LIGHT_TREE_WGSL } from '../src/shaders/lightTree.wgsl.js';
import { REGIR_WGSL } from '../src/shaders/regir.wgsl.js';
import { RIS_GI_WGSL } from '../src/shaders/risGi.wgsl.js';
import { ENVIRONMENT_SAMPLE_WGSL } from '../src/shaders/environmentSample.wgsl.js';
import { PPG_PDF_WGSL } from '../src/ppg/ppgPdf.wgsl.js';
import { SAMPLE_BUDGET_WGSL } from '../src/shaders/sampleBudget.wgsl.js';
import { SHADE_WGSL } from '../src/shaders/shade.wgsl.js';
import { SAMPLE_CASCADE_C0_WGSL } from '../src/shaders/sampleCascadeC0.wgsl.js';
import { STAINED_GLASS_SHADE_WGSL } from '../src/shaders/stainedGlassShade.wgsl.js';
import { SPATIAL_WGSL } from '../src/shaders/spatial.wgsl.js';
import { SPATIAL_GI_WGSL, SPATIAL_GI_GRIS_WGSL } from '../src/shaders/spatialGi.wgsl.js';
import { SPATIAL_GI_COMMON_WGSL } from '../src/shaders/spatialGiCommon.wgsl.js';
import { SURFACE_TEXTURES_WGSL } from '../src/shaders/surfaceTextures.wgsl.js';
import { TEMPORAL_WGSL } from '../src/shaders/temporal.wgsl.js';
import { TEMPORAL_GI_WGSL, TEMPORAL_GI_GRIS_WGSL } from '../src/shaders/temporalGi.wgsl.js';
import { WELFORD_TEMPORAL_WGSL } from '../src/shaders/welfordTemporal.wgsl.js';
import { DDGI_SAMPLE_WGSL, DDGI_GRID_UBO_WGSL } from '../src/ddgi/ddgiSampleWgsl.js';
import { PPG_TREE_LAYOUT_WGSL } from '../src/ppg/ppgTreeLayout.wgsl.js';
import { PPG_UPDATE_WGSL } from '../src/ppg/ppgUpdate.wgsl.js';

// ──────────────────────────────────────────────────────────────────────────
// 1. composeWgsl semantics
// ──────────────────────────────────────────────────────────────────────────
describe('composeWgsl — semantics', () => {
  // Tiny synthetic graph used by the semantic tests. Independent of the
  // real WGSL_MODULES registry so a regression in module declarations
  // doesn't cascade into composer-level failures.
  const A: WgslModule = { name: 'A', source: '<A>', requires: [] };
  const B: WgslModule = { name: 'B', source: '<B>', requires: ['A'] };
  const C: WgslModule = { name: 'C', source: '<C>', requires: ['A'] };
  const D: WgslModule = { name: 'D', source: '<D>', requires: ['B', 'C'] };

  const synthetic = new Map<string, WgslModule>([
    ['A', A], ['B', B], ['C', C], ['D', D],
  ]);

  it('emits a leaf module with no deps as just its source', () => {
    expect(composeWgsl(A, synthetic)).toBe('<A>');
  });

  it('prepends a single dep in dep-first order', () => {
    expect(composeWgsl(B, synthetic)).toBe('<A><B>');
  });

  it('deduplicates a dep reached via multiple paths (diamond)', () => {
    // D requires [B, C]; both require A. A must appear ONCE.
    expect(composeWgsl(D, synthetic)).toBe('<A><B><C><D>');
  });

  it('does NOT include the root in its own dependency emission set', () => {
    // If composeWgsl(D) erroneously added D to `emitted` before emitting
    // deps, this would still pass — so additionally assert that the
    // result ends with D and contains D exactly once.
    const out = composeWgsl(D, synthetic);
    expect(out.endsWith('<D>')).toBe(true);
    expect(out.split('<D>').length - 1).toBe(1);
  });

  it('throws on a dependency cycle', () => {
    const X: WgslModule = { name: 'X', source: '<X>', requires: ['Y'] };
    const Y: WgslModule = { name: 'Y', source: '<Y>', requires: ['X'] };
    const cyclic = new Map<string, WgslModule>([['X', X], ['Y', Y]]);
    expect(() => composeWgsl(X, cyclic)).toThrow(/cycle/i);
  });

  it('throws on a self-referencing module', () => {
    const S: WgslModule = { name: 'S', source: '<S>', requires: ['S'] };
    const selfRef = new Map<string, WgslModule>([['S', S]]);
    expect(() => composeWgsl(S, selfRef)).toThrow(/cycle/i);
  });

  it('throws on an unknown require name with a helpful message', () => {
    const Q: WgslModule = { name: 'Q', source: '<Q>', requires: ['missing'] };
    const reg = new Map<string, WgslModule>([['Q', Q]]);
    expect(() => composeWgsl(Q, reg)).toThrow(/unknown module 'missing'/);
  });

  it('is deterministic: same inputs → same output every call', () => {
    const a = composeWgsl(D, synthetic);
    const b = composeWgsl(D, synthetic);
    const c = composeWgsl(D, synthetic);
    expect(a).toBe(b);
    expect(b).toBe(c);
  });
});

// ──────────────────────────────────────────────────────────────────────────
// 2. WGSL_MODULES — every declared module resolves cleanly
// ──────────────────────────────────────────────────────────────────────────
describe('WGSL_MODULES registry', () => {
  it('contains every module-name keyed by its own .name', () => {
    for (const [key, mod] of WGSL_MODULES) {
      expect(mod.name).toBe(key);
    }
  });

  it('every `requires` entry resolves to a module in the registry', () => {
    for (const mod of WGSL_MODULES.values()) {
      for (const dep of mod.requires) {
        expect(
          WGSL_MODULES.has(dep),
          `Module '${mod.name}' requires unknown '${dep}'`,
        ).toBe(true);
      }
    }
  });

  it('every module composes without throwing', () => {
    for (const mod of WGSL_MODULES.values()) {
      expect(
        () => composeWgsl(mod, WGSL_MODULES),
        `composeWgsl failed for '${mod.name}'`,
      ).not.toThrow();
    }
  });
});

// ──────────────────────────────────────────────────────────────────────────
// 3. Bit-identical gate — composed strings match pre-R6 concat patterns
// ──────────────────────────────────────────────────────────────────────────
describe('composeWgsl — bit-identical to pre-R6 concat patterns', () => {
  // The pre-R6 patterns lived in `pipelineCompiler.ts` and the two denoiser
  // entries. They are restated here as plain string concatenation so the
  // assertion that "the include-graph emits the same bytes" is mechanical
  // and stays auditable.

  // W2-C7+C9 update: ris/temporal/spatial now depend on the canonical
  // restirPHat / restirCastPrimary helpers (Bitterli 2020 §4.3 — the p̂
  // function MUST BE IDENTICAL across the three passes, structurally
  // enforced by sharing the declaration site). The composed output gains
  // RESTIR_PHAT_WGSL after COMMON_WGSL (and RESTIR_CAST_PRIMARY_WGSL for
  // the two reuse passes); the function bodies themselves moved verbatim
  // from the three consumer files into the shared modules, so the total
  // composed-bytes for any pass equals the pre-refactor composed bytes
  // with the duplicated helper bodies relocated to the shared section.

  it('common includes PR-3 traceScene helpers and TLAS traversal', () => {
    expect(COMMON_WGSL).toContain('fn traceSceneFirstHit');
    expect(COMMON_WGSL).toContain('fn traceTlasFirstHit');
    expect(COMMON_WGSL).toContain('bvhMode:');
    expect(COMMON_WGSL).toContain('tlasNodeCount:');
  });

  it('ris: COMMON_WGSL + ENVIRONMENT_SAMPLE_WGSL + RESTIR_PHAT_WGSL + LIGHT_TREE_WGSL + REGIR_WGSL + RIS_WGSL', () => {
    // RIS_MODULE.requires === ['restirPHat', 'regir', 'environmentSample'].
    // Wave 4: restirPHat now requires ['common', 'environmentSample'] (the
    // ENV_SAMPLE_SENTINEL branch calls envHasMap/envRadiance/envDirFromXi).
    // DFS order: common (via restirPHat.requires[0]) → environmentSample
    // (via restirPHat.requires[1]) → restirPHat → lightTree (via regir) →
    // regir → environmentSample (already emitted, skipped) → ris.
    // environmentSample now appears at position 2 (before restirPHat) rather
    // than at the end, because restirPHat pulls it in first.
    expect(composeWgsl(RIS_MODULE, WGSL_MODULES)).toBe(
      COMMON_WGSL + ENVIRONMENT_SAMPLE_WGSL + RESTIR_PHAT_WGSL + LIGHT_TREE_WGSL + REGIR_WGSL + RIS_WGSL,
    );
  });

  it('temporal: COMMON_WGSL + ENVIRONMENT_SAMPLE_WGSL + RESTIR_PHAT_WGSL + RESTIR_CAST_PRIMARY_WGSL + TEMPORAL_WGSL', () => {
    // Wave 4: restirPHat now requires ['common', 'environmentSample'] so
    // environmentSample is emitted before restirPHat in all passes that
    // depend on restirPHat (temporal, spatial, ris).
    expect(composeWgsl(TEMPORAL_MODULE, WGSL_MODULES)).toBe(
      COMMON_WGSL + ENVIRONMENT_SAMPLE_WGSL + RESTIR_PHAT_WGSL + RESTIR_CAST_PRIMARY_WGSL + TEMPORAL_WGSL,
    );
  });

  it('spatial: COMMON_WGSL + ENVIRONMENT_SAMPLE_WGSL + RESTIR_PHAT_WGSL + RESTIR_CAST_PRIMARY_WGSL + SPATIAL_WGSL', () => {
    // Wave 4: see temporal note above — same dep change.
    expect(composeWgsl(SPATIAL_MODULE, WGSL_MODULES)).toBe(
      COMMON_WGSL + ENVIRONMENT_SAMPLE_WGSL + RESTIR_PHAT_WGSL + RESTIR_CAST_PRIMARY_WGSL + SPATIAL_WGSL,
    );
  });

  it('shade: COMMON_WGSL + SURFACE_TEXTURES_WGSL + DDGI_SAMPLE_WGSL + DDGI_GRID_UBO_WGSL + OCTAHEDRAL_CORE_WGSL + SAMPLE_CASCADE_C0_WGSL + STAINED_GLASS_SHADE_WGSL + SHADE_WGSL', () => {
    // W8 Phase 3 (2026-05-18) — original shade composition.
    //
    // T5 (2026-05-28) — SHADE_MODULE.requires ends with 'stainedGlassShade'.
    // B3 — SHADE_MODULE.requires ends with 'environmentSample'.
    //
    // D5.1+D5.2 (2026-06-10) — 'ddgiSample' replaced by 'ddgiGridUbo' in
    // SHADE_MODULE.requires. ddgiGridUbo requires ['ddgiSample'], so the
    // composition now emits ddgiSample (no deps) then ddgiGridUbo (DDGIGridUBO
    // struct + binding(3) + sampleDDGIAtPoint) before the cascade/stainedGlass
    // modules and shade body. common is emitted once via dedup.
    expect(composeWgsl(SHADE_MODULE, WGSL_MODULES)).toBe(
      COMMON_WGSL +
      SURFACE_TEXTURES_WGSL +
      DDGI_SAMPLE_WGSL +
      DDGI_GRID_UBO_WGSL +
      OCTAHEDRAL_CORE_WGSL +
      SAMPLE_CASCADE_C0_WGSL +
      STAINED_GLASS_SHADE_WGSL +
      ENVIRONMENT_SAMPLE_WGSL +
      SHADE_WGSL,
    );
  });

  it('atrous: COMMON_WGSL + ATROUS_WGSL', () => {
    expect(composeWgsl(ATROUS_MODULE, WGSL_MODULES)).toBe(COMMON_WGSL + ATROUS_WGSL);
  });

  it('compositeVert: standalone (no prepend)', () => {
    expect(composeWgsl(COMPOSITE_VERT_MODULE, WGSL_MODULES)).toBe(COMPOSITE_VERT_WGSL);
  });

  it('compositeFrag: standalone (no prepend)', () => {
    expect(composeWgsl(COMPOSITE_FRAG_MODULE, WGSL_MODULES)).toBe(COMPOSITE_FRAG_WGSL);
  });

  it('sampleBudget: WELFORD_TAIL_WGSL + SAMPLE_BUDGET_WGSL', () => {
    expect(composeWgsl(SAMPLE_BUDGET_MODULE, WGSL_MODULES)).toBe(WELFORD_TAIL_WGSL + SAMPLE_BUDGET_WGSL);
  });

  it('resolve: screenCoordHelpers + RESOLVE_WGSL (D5.4 dedup — clampCoord extracted)', () => {
    // D5.4: clampCoord moved to screenCoordHelpers; RESOLVE_MODULE now
    // requires: ['screenCoordHelpers']. The composed output is the shared helper
    // prepended to the resolve body (byte-identical rule: SCREEN_COORD_HELPERS_WGSL + RESOLVE_WGSL).
    expect(composeWgsl(RESOLVE_MODULE, WGSL_MODULES)).toBe(SCREEN_COORD_HELPERS_WGSL + RESOLVE_WGSL);
  });

  it('gtao: GTAO_COMMON_WGSL + GTAO_WGSL', () => {
    expect(composeWgsl(GTAO_MODULE, WGSL_MODULES)).toBe(GTAO_COMMON_WGSL + GTAO_WGSL);
  });

  it('gtaoUpsample: GTAO_COMMON_WGSL + GTAO_UPSAMPLE_WGSL', () => {
    expect(composeWgsl(GTAO_UPSAMPLE_MODULE, WGSL_MODULES)).toBe(GTAO_COMMON_WGSL + GTAO_UPSAMPLE_WGSL);
  });

  // T9-stepC — the GI passes were narrowed off the full `common` aggregate to
  // the minimal focused-module subset each references. The composed output is
  // now strictly smaller than `COMMON_WGSL + …`; the exact narrowed
  // composition is pinned below (focused-module sources concatenated in the
  // pass's declared `requires` order, deps-first, deduped).
  it('risGi (narrowed): walkaroundUbo + sceneTraversal + reservoirGi + sharedPrimitives + materialDecode + cameraRays + ddgiGridUbo(ddgiSample) + ppgPdf(+ppgTreeLayout) + environmentSample + RIS_GI', () => {
    // W9 guided sampling — risGi now requires `ppgPdf` (the gi-ris dTree
    // pdf-eval + guided sampler). ppgPdf requires only `ppgTreeLayout` (the
    // 2026-06-09 equal-area fix dropped its octahedralCore dependency — it now
    // uses an inline cylindrical map), so the composer emits PPG_TREE_LAYOUT_WGSL
    // then PPG_PDF_WGSL after DDGI_GRID_UBO, before the RIS_GI root source. No
    // module in this unit calls the shared octEncode/octDecode (ddgiSample
    // inlines its own), so OCTAHEDRAL_CORE_WGSL is correctly absent here.
    //
    // D5.1+D5.2 (2026-06-10) — 'ddgiSample' replaced by 'ddgiGridUbo' which
    // requires ['ddgiSample']. ddgiSample emits first (no deps), then ddgiGridUbo
    // (DDGIGridUBO struct + @group(3) @binding(3) + sampleDDGIAtPoint wrapper).
    expect(composeWgsl(RIS_GI_MODULE, WGSL_MODULES)).toBe(
      WALKAROUND_UBO_WGSL +
      SCENE_TRAVERSAL_WGSL +
      RESERVOIR_GI_WGSL +
      SHARED_PRIMITIVES_WGSL +
      MATERIAL_DECODE_WGSL +
      CAMERA_RAYS_WGSL +
      DDGI_SAMPLE_WGSL +
      DDGI_GRID_UBO_WGSL +
      PPG_TREE_LAYOUT_WGSL +
      PPG_PDF_WGSL +
      ENVIRONMENT_SAMPLE_WGSL +
      RIS_GI_WGSL,
    );
  });

  // ── GI reuse passes: the DEFAULT (restirPtReuse OFF) module is the verbatim
  // Sprint-17 pass — no sceneTraversal/grisReuse, NO @group(1). The GRIS (ON)
  // variant is a SEPARATE compile-root composed only when the host opts in.
  // This split is the f8df9a4 black-frame fix: an opt-in feature must not change
  // the default pipeline structure. (The structural-no-group(1) assertion lives
  // in giStructuralGate.test.ts.) ──
  it('temporalGi OFF (default, narrowed): walkaroundUbo + sceneTraversal + reservoirGi + sharedPrimitives + jacobianShift + cameraRays + TEMPORAL_GI', () => {
    expect(composeWgsl(TEMPORAL_GI_MODULE, WGSL_MODULES)).toBe(
      WALKAROUND_UBO_WGSL +
      SCENE_TRAVERSAL_WGSL +
      RESERVOIR_GI_WGSL +
      SHARED_PRIMITIVES_WGSL +
      JACOBIAN_SHIFT_WGSL +
      CAMERA_RAYS_WGSL +
      TEMPORAL_GI_WGSL,
    );
  });

  it('temporalGi ON (GRIS): walkaroundUbo + sceneTraversal + reservoirGi + sharedPrimitives + cameraRays + grisReuse + materialDecode + TEMPORAL_GI_GRIS', () => {
    // GRIS variant adds `grisReuse` (shift + pairwise-MIS math) and uses
    // `sceneTraversal` (the reconnection-visibility ray's traceSceneAny);
    // `sceneTraversal`/`cameraRays` were already in the closure. It DROPS
    // `jacobianShift` — the GRIS path uses grisShiftJacobian, not the legacy
    // clamped Jacobian.
    expect(composeWgsl(TEMPORAL_GI_GRIS_MODULE, WGSL_MODULES)).toBe(
      WALKAROUND_UBO_WGSL +
      SCENE_TRAVERSAL_WGSL +
      RESERVOIR_GI_WGSL +
      SHARED_PRIMITIVES_WGSL +
      CAMERA_RAYS_WGSL +
      GRIS_REUSE_WGSL +
      MATERIAL_DECODE_WGSL +
      TEMPORAL_GI_GRIS_WGSL,
    );
  });

  it('spatialGi OFF (default, narrowed): walkaroundUbo + spatialGiCommon + reservoirGi + sharedPrimitives + jacobianShift + SPATIAL_GI', () => {
    // Task3 — K_SPATIAL_GI / M_CLAMP_SPATIAL / sampleDiscPx hoisted into the
    // `spatialGiCommon` module (registered dep) so both OFF and GRIS ON roots
    // share a single declaration. `spatialGiCommon` has requires:[] so it
    // lands immediately after walkaroundUbo (the first declared dep).
    expect(composeWgsl(SPATIAL_GI_MODULE, WGSL_MODULES)).toBe(
      WALKAROUND_UBO_WGSL +
      SPATIAL_GI_COMMON_WGSL +
      RESERVOIR_GI_WGSL +
      SHARED_PRIMITIVES_WGSL +
      JACOBIAN_SHIFT_WGSL +
      SPATIAL_GI_WGSL,
    );
  });

  it('spatialGi ON (GRIS): walkaroundUbo + spatialGiCommon + sceneTraversal + reservoirGi + sharedPrimitives + grisReuse + materialDecode + SPATIAL_GI_GRIS', () => {
    // GRIS variant adds `sceneTraversal` (the reconnection-visibility ray's
    // traceSceneAny + BVHNode) and `grisReuse` (the shift + pairwise-MIS math),
    // and DROPS `jacobianShift` (grisShiftJacobian replaces the legacy reuse).
    // Task3 — `spatialGiCommon` (requires:[]) is now the second declared dep,
    // so it lands immediately after walkaroundUbo, before sceneTraversal.
    expect(composeWgsl(SPATIAL_GI_GRIS_MODULE, WGSL_MODULES)).toBe(
      WALKAROUND_UBO_WGSL +
      SPATIAL_GI_COMMON_WGSL +
      SCENE_TRAVERSAL_WGSL +
      RESERVOIR_GI_WGSL +
      SHARED_PRIMITIVES_WGSL +
      GRIS_REUSE_WGSL +
      MATERIAL_DECODE_WGSL +
      SPATIAL_GI_GRIS_WGSL,
    );
  });

  it('motionVectors (narrowed): walkaroundUbo + sceneTraversal + sharedPrimitives + cameraRays + MOTION_VECTORS', () => {
    expect(composeWgsl(MOTION_VECTORS_MODULE, WGSL_MODULES)).toBe(
      WALKAROUND_UBO_WGSL +
      SCENE_TRAVERSAL_WGSL +
      SHARED_PRIMITIVES_WGSL +
      CAMERA_RAYS_WGSL +
      MOTION_VECTORS_WGSL,
    );
  });

  it('indirectCombine: standalone (no prepend)', () => {
    expect(composeWgsl(INDIRECT_COMBINE_MODULE, WGSL_MODULES)).toBe(INDIRECT_COMBINE_WGSL);
  });

  it('indirectTemporalAccum: standalone (no prepend)', () => {
    expect(composeWgsl(INDIRECT_TEMPORAL_ACCUM_MODULE, WGSL_MODULES)).toBe(
      INDIRECT_TEMPORAL_ACCUM_WGSL,
    );
  });

  it('temporalAccum: standalone (no prepend)', () => {
    expect(composeWgsl(TEMPORAL_ACCUM_MODULE, WGSL_MODULES)).toBe(TEMPORAL_ACCUM_WGSL);
  });

  // Denoiser entries.

  it('welfordTemporal (narrowed): LUMINANCE_WGSL + WELFORD_TAIL_WGSL + WELFORD_TEMPORAL_WGSL', () => {
    // T9-stepC — narrowed off `common`: this pass binds its own
    // WelfordTemporalUBO and only needs `luminance` (Rec.709) + the
    // WelfordVariance struct/helpers (welfordTail wrapper).
    expect(composeWgsl(WELFORD_TEMPORAL_MODULE, WGSL_MODULES)).toBe(
      LUMINANCE_WGSL + WELFORD_TAIL_WGSL + WELFORD_TEMPORAL_WGSL,
    );
  });

  it('atrousVariance: self-contained — composer adds nothing', () => {
    // Pre-R6 anti-duplication-by-comment at pipelineCompiler.ts:131 +
    // atrousVariance.ts:148: this string declares its own PI/INV_PI/LUM_W/
    // WelfordVariance. The structural fix is `requires: []`.
    expect(composeWgsl(ATROUS_VARIANCE_MODULE, WGSL_MODULES)).toBe(ATROUS_VARIANCE_WGSL);
  });

  it('svgfReprojection: standalone (no prepend)', () => {
    expect(composeWgsl(SVGF_REPROJECTION_MODULE, WGSL_MODULES)).toBe(SVGF_REPROJECTION_WGSL);
  });

  it('svgfVarianceFromMoments: standalone (no prepend)', () => {
    expect(composeWgsl(SVGF_VARIANCE_FROM_MOMENTS_MODULE, WGSL_MODULES)).toBe(
      SVGF_VARIANCE_FROM_MOMENTS_WGSL,
    );
  });

  it('svgf7x7SpatialFallback: standalone (no prepend)', () => {
    expect(composeWgsl(SVGF_7X7_SPATIAL_FALLBACK_MODULE, WGSL_MODULES)).toBe(
      SVGF_7X7_SPATIAL_FALLBACK_WGSL,
    );
  });

  // PPG (Müller 2017) — ppgUpdate requires canonical luminance (W8
  // follow-up cleanup). Guided sampling is inlined in gi-ris via ppgPdf.

  it('ppgUpdate: LUMINANCE_WGSL + PPG_TREE_LAYOUT_WGSL + PPG_UPDATE_WGSL', () => {
    // 2026-06-09 equal-area fix: ppgUpdate dropped its octahedralCore require
    // (the octEncode training-direction call became the inline cylindrical map),
    // so OCTAHEDRAL_CORE_WGSL is no longer composed into this standalone kernel.
    const composed = composeWgsl(PPG_UPDATE_MODULE, WGSL_MODULES);
    expect(composed).toBe(LUMINANCE_WGSL + PPG_TREE_LAYOUT_WGSL + PPG_UPDATE_WGSL);
  });

});

// ──────────────────────────────────────────────────────────────────────────
// 4. T9-stepC — static cross-module identifier-resolution gate
//
// There is no GPU in CI to catch a `requires`-narrowed pass that dropped a
// module it actually needs. This gate tokenizes each pass's composed closure
// and asserts every LIBRARY symbol the closure references is declared inside
// that closure. It is the safety net that justifies narrowing each pass's
// `requires` below the full `common` aggregate.
// ──────────────────────────────────────────────────────────────────────────
describe('T9-stepC — static cross-module identifier resolution', () => {
  // The symbol universe = module-scope declarations across every library
  // module a pass may include (common's 11 focused modules + the non-common
  // helpers). A reference to any of these names can ONLY be satisfied by
  // including the declaring module, so this is the set we resolve against.
  const LIBRARY_MODULE_NAMES = [
    'walkaroundUbo', 'sceneTraversal', 'reservoirDi', 'reservoirGi',
    'sharedPrimitives', 'ggxBrdf', 'materialDecode', 'emitterSampling',
    'jacobianShift', 'cameraRays', 'welfordTail',
    'luminance', 'octahedralCore',
    'surfaceTextures', 'ddgiSample', 'sampleCascadeC0', 'stainedGlassShade',
    'restirPHat', 'restirCastPrimary', 'lightTree',
  ];
  const symbolUniverse = buildSymbolUniverse(
    LIBRARY_MODULE_NAMES.map((n) => {
      const m = WGSL_MODULES.get(n);
      if (!m) throw new Error(`library module '${n}' missing from WGSL_MODULES`);
      return m.source;
    }),
  );

  // Every compute/render pass that is the root of a compiled pipeline. The
  // gate runs against each, narrowed or not — a regression that narrows the
  // wrong module would surface as an unresolved library symbol here.
  const ROOT_PASSES = [
    'ris', 'temporal', 'spatial', 'shade',
    'risGi', 'temporalGi', 'spatialGi',
    // GRIS (restirPtReuse ON) compile-roots — composed only when the host opts
    // in, but the ident-resolution gate must still cover their closures.
    'temporalGiGris', 'spatialGiGris',
    'welfordTemporal', 'motionVectors',
    'sampleBudget', 'resolve', 'gtao', 'gtaoUpsample',
    'indirectCombine', 'indirectTemporalAccum', 'atrous',
  ];

  it('every root pass resolves all referenced library symbols in its closure', () => {
    for (const name of ROOT_PASSES) {
      const mod = WGSL_MODULES.get(name);
      expect(mod, `root pass '${name}' missing from WGSL_MODULES`).toBeDefined();
      const composed = composeWgsl(mod!, WGSL_MODULES);
      const { missing, used } = checkCrossModuleResolution({
        ownSource: mod!.source,
        composed,
        symbolUniverse,
      });
      expect(
        missing,
        `pass '${name}' references library symbols not declared in its narrowed ` +
        `closure: [${missing.join(', ')}] (it uses [${used.join(', ')}])`,
      ).toEqual([]);
    }
  });

  // Spot-pin the actual narrowing achieved: each narrowed pass must NOT pull
  // the modules it was proven not to reference. This guards against a future
  // accidental re-widening back to `['common']`.
  it('narrowed passes do not reference the modules they dropped', () => {
    const droppedExpectations: Record<string, readonly string[]> = {
      // welfordTemporal binds its own UBO + only needs luminance/welford.
      welfordTemporal: [
        'WalkaroundUBO', 'BVHNode', 'traceSceneFirstHit', 'evalGGX',
        'sampleEmitterPoint', 'jacobianReconnectionShift', 'ReservoirDI',
      ],
      // motionVectors: pure reprojection, no reservoirs / BRDF / emitters.
      motionVectors: [
        'evalGGX', 'sampleEmitterPoint', 'loadReservoirDI_rw',
        'jacobianReconnectionShift', 'WelfordVariance',
      ],
      // spatialGi (OFF, default): no primary cast, no BRDF, no emitters, NO
      // scene traversal, NO GRIS. This is the verbatim Sprint-17 pass — the
      // f8df9a4 black-frame fix moved the @group(1) scene BVH + grisReuse into
      // the SEPARATE `spatialGiGris` root, so the default closure must NOT pull
      // them (this is exactly the structural guarantee that was missing).
      spatialGi: [
        'evalGGX', 'sampleEmitterPoint', 'BVHNode', 'traceSceneFirstHit',
        'generatePrimaryRay_common', 'grisShiftJacobian', 'grisTargetAt',
      ],
      // spatialGiGris (ON): adds sceneTraversal (reconnection-visibility ray) +
      // grisReuse, but still no primary cast / BRDF / emitter, and DROPS the
      // legacy jacobianReconnectionShift (grisShiftJacobian replaces it).
      spatialGiGris: [
        'evalGGX', 'sampleEmitterPoint', 'generatePrimaryRay_common',
        'jacobianReconnectionShift',
      ],
      // temporalGi (OFF, default): reprojects (needs cameraRays) but no BRDF /
      // emitters, NO GRIS. The default closure must NOT pull grisReuse.
      temporalGi: [
        'evalGGX', 'sampleEmitterPoint', 'WelfordVariance',
        'grisShiftJacobian', 'grisTargetAt',
      ],
      // temporalGiGris (ON): adds grisReuse + uses sceneTraversal, drops the
      // legacy jacobianReconnectionShift.
      temporalGiGris: [
        'evalGGX', 'sampleEmitterPoint', 'WelfordVariance',
        'jacobianReconnectionShift',
      ],
      // risGi: casts primary + DDGI, but no emitter sampling / GGX / welford.
      risGi: ['evalGGX', 'sampleEmitterPoint', 'WelfordVariance', 'jacobianReconnectionShift'],
    };
    for (const [name, dropped] of Object.entries(droppedExpectations)) {
      const composed = composeWgsl(WGSL_MODULES.get(name)!, WGSL_MODULES);
      const composedDecls = buildSymbolUniverse([composed]);
      for (const sym of dropped) {
        expect(
          composedDecls.has(sym),
          `narrowed pass '${name}' unexpectedly still pulls a module declaring '${sym}'`,
        ).toBe(false);
      }
    }
  });
});

// ──────────────────────────────────────────────────────────────────────────
// 4b. Theme-C temporalGiCommon dedup — byte-identity pin.
//
// Task 2.2 Item 5 hoisted the geometric-rejection consts + projectToPrevHalfPx
// (shared verbatim between the OFF and GRIS temporal-GI bodies) into the
// `temporalGiCommon` fragment, and DELETED the dead `worldFromHalfPx_temporal`
// helper (defined in both copies, called by neither — the pass reprojects via
// rCur.xv). The composed strings must therefore be byte-identical EXCEPT for the
// absence of that one dead function. These pins enforce exactly that.
// ──────────────────────────────────────────────────────────────────────────
describe('Theme-C — temporalGiCommon dedup (byte-identity minus the deleted dead fn)', () => {
  // The dead helper that Item 5 deleted (one of two sanctioned deletions). It
  // must be absent from BOTH temporal-GI bodies and from their composed roots.
  const deadFnSig = 'fn worldFromHalfPx_temporal';

  it('the dead worldFromHalfPx_temporal helper is gone from both temporal-GI bodies', () => {
    expect(TEMPORAL_GI_WGSL).not.toContain(deadFnSig);
    expect(TEMPORAL_GI_GRIS_WGSL).not.toContain(deadFnSig);
  });

  it('the shared helpers (consts + projectToPrevHalfPx) appear exactly once per body', () => {
    for (const body of [TEMPORAL_GI_WGSL, TEMPORAL_GI_GRIS_WGSL]) {
      expect(body.split('const DEPTH_REL_TOL: f32 = 0.1;').length - 1).toBe(1);
      expect(body.split('const NORMAL_DOT_MIN: f32 = 0.906;').length - 1).toBe(1);
      expect(body.split('fn projectToPrevHalfPx(').length - 1).toBe(1);
    }
  });

  it('both temporal-GI compose-roots are free of the dead fn (composer adds nothing that resurrects it)', () => {
    expect(composeWgsl(TEMPORAL_GI_MODULE, WGSL_MODULES)).not.toContain(deadFnSig);
    expect(composeWgsl(TEMPORAL_GI_GRIS_MODULE, WGSL_MODULES)).not.toContain(deadFnSig);
  });
});

// ──────────────────────────────────────────────────────────────────────────
// 5. Theme-D guard — the @group(1) scene BVH binding block is NOT hoistable
//    into a shared `sceneBindings` fragment without changing the composed
//    bytes, so it stays inlined per consumer.
//
// Two independent reasons (either alone blocks a byte-identical hoist):
//
//  (a) The composer is PREPEND-ONLY (deps emitted first, root source last —
//      see wgslComposer.ts). The scene-binding block lives in the MIDDLE of
//      each consumer's source (after the @group(0) frame bindings, before the
//      @group(2) UBO). Hoisting it to a `requires` fragment would emit it at
//      the TOP of the composed string, reordering bytes — the exact failure
//      mode common.wgsl.ts documents for honest inter-sibling `requires`.
//
//  (b) The block is NOT identical across consumers: every scene-traversal pass
//      carries `bvh_normal` (binding 11), but shade adds `bvh_beer`
//      (binding 5) + `bvh_emissive` (binding 12); risGi omits
//      `emitters`/`emitterCdf` (bindings 3/4). A single shared fragment can
//      reproduce none of them verbatim.
//
// This guard pins the divergence so a future agent doesn't naively "dedup" the
// block and silently change a composed shader. The DDGIGridUBO struct (byte-
// identical between shade/risGi) is likewise NOT hoisted — same reason (a): it
// sits mid-file after the @group(3) texture bindings.
// ──────────────────────────────────────────────────────────────────────────
describe('Theme-D — scene @group(1) binding block stays inlined (not hoistable byte-identically)', () => {
  it('bvh_normal (binding 11) is present in every scene-traversal pass', () => {
    const norm = '@group(1) @binding(11) var<storage, read> bvh_normal: array<vec4f>;';
    expect(RIS_WGSL).toContain(norm);
    expect(SHADE_WGSL).toContain(norm);
    expect(RIS_GI_WGSL).toContain(norm);
    expect(TEMPORAL_WGSL).toContain(norm);
    expect(SPATIAL_WGSL).toContain(norm);
  });

  it('shade carries bvh_beer (binding 5) + bvh_emissive (binding 12) that no other consumer has', () => {
    expect(SHADE_WGSL).toContain('@group(1) @binding(5) var bvh_beer: texture_2d<u32>;');
    expect(SHADE_WGSL).toContain('@group(1) @binding(12) var bvh_emissive: texture_2d<f32>;');
    expect(RIS_WGSL).not.toContain('bvh_beer');
    expect(TEMPORAL_WGSL).not.toContain('bvh_beer');
  });

  it('risGi omits emitters/emitterCdf (bindings 3/4) that ris/temporal/spatial declare', () => {
    const emit = '@group(1) @binding(3) var<storage, read> emitters:     array<EmitterTri>;';
    expect(RIS_WGSL).toContain(emit);
    expect(TEMPORAL_WGSL).toContain(emit);
    expect(SPATIAL_WGSL).toContain(emit);
    expect(RIS_GI_WGSL).not.toContain(emit);
  });

  it('the binding subsets span ≥3 distinct shapes — no single shared fragment reproduces all', () => {
    // Reduce each consumer to its @group(1) binding-name set; assert at least
    // three distinct shapes exist (ris, shade, risGi all differ), which is what
    // makes a single hoisted `sceneBindings` fragment impossible.
    // 2026-06-06 (G-P0.1 smooth-normal reuse consistency): temporal + spatial
    // gained bvh_normal @binding(11), which made temporal's group(1) block
    // IDENTICAL to ris's. B1 (road-to-100, 2026-06-10) then RE-DIVERGED them:
    // ris declares bvh_material @binding(14) INLINE (its candidate p̂ decodes
    // real roughness/metal), while temporal/spatial pull bvh_material via the
    // shared restirCastPrimary module (a SEPARATE source), so temporal.wgsl's
    // own group(1) block no longer carries binding 14. The ≥3-distinct-shapes
    // rationale for keeping the block inlined holds even more strongly now.
    const group1Lines = (src: string): string =>
      src
        .split('\n')
        .filter((l) => l.includes('@group(1) @binding'))
        .join('\n');
    const ris = group1Lines(RIS_WGSL);
    const shade = group1Lines(SHADE_WGSL);
    const temporal = group1Lines(TEMPORAL_WGSL);
    const risGi = group1Lines(RIS_GI_WGSL);
    expect(ris).not.toBe(shade);
    expect(ris).not.toBe(temporal); // B1 — ris gained inline bvh_material @binding(14); temporal gets it via restirCastPrimary
    expect(ris).not.toBe(risGi);
    expect(shade).not.toBe(temporal);
    // ris/temporal still share every binding EXCEPT the B1 bvh_material line.
    expect(temporal).toBe(ris.split('\n').filter((l) => !l.includes('binding(14)')).join('\n'));
  });
});
