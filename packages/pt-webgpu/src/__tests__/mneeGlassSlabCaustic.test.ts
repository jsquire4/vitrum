import { describe, expect, it } from 'vitest';
import { PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL as code } from '../wgsl/pathTrace/caustic.wgsl.js';
import { PT_WEBGPU_TRACE_WGSL, composePtWebgpuTraceWgsl } from '../wgsl/pathTraceBruteforce.wgsl.js';

describe('unified MNEE multi-interface closure', () => {
  it('owns chain lengths one through eight in one coupled estimator', () => {
    expect(code).toContain('let maximumLength = min(params.mneeMaxChainLength, 8u);');
    expect(code).toContain('let chainLength = 1u + min(');
    expect(code).toContain('var facets: array<MneeFacetProposal, 8>;');
    expect(code).toContain('return boundedManifoldCaustic(');
    expect(code).not.toContain('fn pointLightGlassSlabCaustic(');
    expect(code).not.toContain('fn pointLightBoundedChainCaustic(');
  });

  it('pins exact facet sequence validation and the full-support mixture', () => {
    expect(code).toContain('let targetIndex = reverseIndex - 1u;');
    expect(code).toContain('facets[targetIndex],');
    expect(code).toContain('fn mneeProposeConditionalFacet(');
    expect(code).toContain('selected.pdf = 1.0 / f32(count);');
    expect(code).toContain('selected.pdf = selected.pdf + MNEE_GUIDED_MIX_PROBABILITY;');
  });

  it('keeps N^8 proposal weights in an explicit finite log representation', () => {
    expect(code).toContain('var logFacetEventPdf = 0.0;');
    expect(code).toContain('logInterfaceNumerator - vec3f(logFacetEventPdf)');
    expect(code).toContain('fn mneeSaturatedExpRgb(');
    expect(code).toContain('fn mneeSaturatedAddRgb(');
    expect(code).not.toContain('weightedInterfaceProduct');
  });

  it('uses scale-aware offsets and composes no duplicate estimator', () => {
    expect(code).toContain('fn mneeScaleAwareEpsilon(');
    const unified = code.slice(code.indexOf('fn boundedManifoldCaustic('), code.indexOf('fn manifoldNeeContribution('));
    expect(unified).not.toContain('* 1e-3');
    expect(unified).not.toContain('- 2e-3');
    for (const source of [PT_WEBGPU_TRACE_WGSL, composePtWebgpuTraceWgsl(true)]) {
      expect(source).toContain('fn mneeNewtonSolveChainBounded(');
      expect(source).toContain('fn boundedManifoldCaustic(');
      expect(source).not.toContain('fn pointLightGlassSlabCaustic(');
    }
  });
});
