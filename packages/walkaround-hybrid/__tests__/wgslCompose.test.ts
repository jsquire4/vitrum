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
  PPG_GUIDE_MODULE,
  PPG_UPDATE_MODULE,
  RESOLVE_MODULE,
  RIS_GI_MODULE,
  RIS_MODULE,
  SAMPLE_BUDGET_MODULE,
  SHADE_MODULE,
  SPATIAL_GI_MODULE,
  SPATIAL_MODULE,
  SVGF_7X7_SPATIAL_FALLBACK_MODULE,
  SVGF_REPROJECTION_MODULE,
  SVGF_VARIANCE_FROM_MOMENTS_MODULE,
  TEMPORAL_ACCUM_MODULE,
  TEMPORAL_GI_MODULE,
  TEMPORAL_MODULE,
  WELFORD_TEMPORAL_MODULE,
  WGSL_MODULES,
} from '../src/pipeline/wgslModules.js';
import { COMMON_WGSL } from '../src/shaders/common.wgsl.js';
import { COMPOSITE_FRAG_WGSL, COMPOSITE_VERT_WGSL } from '../src/shaders/composite.wgsl.js';
import { GTAO_WGSL } from '../src/shaders/gtao.wgsl.js';
import { GTAO_UPSAMPLE_WGSL } from '../src/shaders/gtaoUpsample.wgsl.js';
import { INDIRECT_COMBINE_WGSL } from '../src/shaders/indirectCombine.wgsl.js';
import { INDIRECT_TEMPORAL_ACCUM_WGSL } from '../src/shaders/indirectTemporalAccum.wgsl.js';
import { RESOLVE_WGSL } from '../src/shaders/resolve.wgsl.js';
import { RESTIR_PHAT_WGSL } from '../src/shaders/restirPHat.wgsl.js';
import { RESTIR_CAST_PRIMARY_WGSL } from '../src/shaders/restirCastPrimary.wgsl.js';
import { RIS_WGSL } from '../src/shaders/ris.wgsl.js';
import { RIS_GI_WGSL } from '../src/shaders/risGi.wgsl.js';
import { SAMPLE_BUDGET_WGSL } from '../src/shaders/sampleBudget.wgsl.js';
import { SHADE_WGSL } from '../src/shaders/shade.wgsl.js';
import { SAMPLE_CASCADE_C0_WGSL } from '../src/shaders/sampleCascadeC0.wgsl.js';
import { SPATIAL_WGSL } from '../src/shaders/spatial.wgsl.js';
import { SPATIAL_GI_WGSL } from '../src/shaders/spatialGi.wgsl.js';
import { SURFACE_TEXTURES_WGSL } from '../src/shaders/surfaceTextures.wgsl.js';
import { TEMPORAL_WGSL } from '../src/shaders/temporal.wgsl.js';
import { TEMPORAL_GI_WGSL } from '../src/shaders/temporalGi.wgsl.js';
import { WELFORD_TEMPORAL_WGSL } from '../src/shaders/welfordTemporal.wgsl.js';
import { DDGI_SAMPLE_WGSL } from '../src/ddgi/ddgiSampleWgsl.js';
import { PPG_GUIDE_WGSL } from '../src/ppg/ppgGuide.wgsl.js';
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

  it('ris: COMMON_WGSL + RESTIR_PHAT_WGSL + RIS_WGSL', () => {
    expect(composeWgsl(RIS_MODULE, WGSL_MODULES)).toBe(
      COMMON_WGSL + RESTIR_PHAT_WGSL + RIS_WGSL,
    );
  });

  it('temporal: COMMON_WGSL + RESTIR_PHAT_WGSL + RESTIR_CAST_PRIMARY_WGSL + TEMPORAL_WGSL', () => {
    expect(composeWgsl(TEMPORAL_MODULE, WGSL_MODULES)).toBe(
      COMMON_WGSL + RESTIR_PHAT_WGSL + RESTIR_CAST_PRIMARY_WGSL + TEMPORAL_WGSL,
    );
  });

  it('spatial: COMMON_WGSL + RESTIR_PHAT_WGSL + RESTIR_CAST_PRIMARY_WGSL + SPATIAL_WGSL', () => {
    expect(composeWgsl(SPATIAL_MODULE, WGSL_MODULES)).toBe(
      COMMON_WGSL + RESTIR_PHAT_WGSL + RESTIR_CAST_PRIMARY_WGSL + SPATIAL_WGSL,
    );
  });

  it('shade: COMMON_WGSL + SURFACE_TEXTURES_WGSL + DDGI_SAMPLE_WGSL + OCTAHEDRAL_CORE_WGSL + SAMPLE_CASCADE_C0_WGSL + SHADE_WGSL', () => {
    // W8 Phase 3 (2026-05-18) — SHADE_MODULE.requires now includes
    // 'sampleCascadeC0', which itself requires 'octahedralCore'. The
    // composer emits dependencies depth-first, so the order becomes:
    //   common → surfaceTextures (requires common) → ddgiSample (requires
    //   common) → sampleCascadeC0 (requires common + octahedralCore) →
    //   shade. `common` is emitted once at the top via the dedup rule.
    //   `octahedralCore` is emitted just before sampleCascadeC0 since it
    //   has no other deps.
    expect(composeWgsl(SHADE_MODULE, WGSL_MODULES)).toBe(
      COMMON_WGSL +
      SURFACE_TEXTURES_WGSL +
      DDGI_SAMPLE_WGSL +
      OCTAHEDRAL_CORE_WGSL +
      SAMPLE_CASCADE_C0_WGSL +
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

  it('sampleBudget: standalone (no prepend)', () => {
    expect(composeWgsl(SAMPLE_BUDGET_MODULE, WGSL_MODULES)).toBe(SAMPLE_BUDGET_WGSL);
  });

  it('resolve: standalone (no prepend)', () => {
    expect(composeWgsl(RESOLVE_MODULE, WGSL_MODULES)).toBe(RESOLVE_WGSL);
  });

  it('gtao: standalone (no prepend)', () => {
    expect(composeWgsl(GTAO_MODULE, WGSL_MODULES)).toBe(GTAO_WGSL);
  });

  it('gtaoUpsample: standalone (no prepend)', () => {
    expect(composeWgsl(GTAO_UPSAMPLE_MODULE, WGSL_MODULES)).toBe(GTAO_UPSAMPLE_WGSL);
  });

  it('risGi: COMMON_WGSL + DDGI_SAMPLE_WGSL + RIS_GI_WGSL', () => {
    expect(composeWgsl(RIS_GI_MODULE, WGSL_MODULES)).toBe(
      COMMON_WGSL + DDGI_SAMPLE_WGSL + RIS_GI_WGSL,
    );
  });

  it('temporalGi: COMMON_WGSL + TEMPORAL_GI_WGSL', () => {
    expect(composeWgsl(TEMPORAL_GI_MODULE, WGSL_MODULES)).toBe(
      COMMON_WGSL + TEMPORAL_GI_WGSL,
    );
  });

  it('spatialGi: COMMON_WGSL + SPATIAL_GI_WGSL', () => {
    expect(composeWgsl(SPATIAL_GI_MODULE, WGSL_MODULES)).toBe(
      COMMON_WGSL + SPATIAL_GI_WGSL,
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

  it('welfordTemporal: COMMON_WGSL + WELFORD_TEMPORAL_WGSL', () => {
    expect(composeWgsl(WELFORD_TEMPORAL_MODULE, WGSL_MODULES)).toBe(
      COMMON_WGSL + WELFORD_TEMPORAL_WGSL,
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

  // PPG (Müller 2017) — ppgUpdate now requires canonical luminance (W8
  // follow-up cleanup); ppgGuide is still standalone.

  it('ppgUpdate: prepends LUMINANCE_WGSL only', () => {
    const composed = composeWgsl(PPG_UPDATE_MODULE, WGSL_MODULES);
    expect(composed).toBe(`${LUMINANCE_WGSL}${PPG_UPDATE_WGSL}`);
  });

  it('ppgGuide: standalone (no prepend)', () => {
    expect(composeWgsl(PPG_GUIDE_MODULE, WGSL_MODULES)).toBe(PPG_GUIDE_WGSL);
  });
});
