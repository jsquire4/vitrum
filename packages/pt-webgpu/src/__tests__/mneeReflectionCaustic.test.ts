import { describe, expect, it } from 'vitest';
import { PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL as code } from '../wgsl/pathTrace/caustic.wgsl.js';
import { PT_WEBGPU_TRACE_WGSL, composePtWebgpuTraceWgsl } from '../wgsl/pathTraceBruteforce.wgsl.js';

describe('unified MNEE reflection closure', () => {
  it('routes reflection through the bounded coupled solver', () => {
    expect(code).toContain('fn boundedManifoldCaustic(');
    expect(code).toContain('if (!mneeFacetHasDeltaReflection(optics))');
    expect(code).toContain('eventFactor = mneeFacetReflectionFactorWithEta(');
    expect(code).toContain('let solved = mneeNewtonSolveChainBounded(');
  });

  it('samples all explicit emitter domains and the area determinant', () => {
    expect(code).toContain('let total = directionalCount + pointCount + spotCount + rectCount + meshCount;');
    expect(code).toContain('let base = index * POINT_LIGHT_VEC4_STRIDE;');
    expect(code).toContain('let base = index * SPOT_LIGHT_VEC4_STRIDE;');
    expect(code).toContain('let areaDet = mneeBoundedChainAreaPdfDet(');
  });

  it('keeps exact identity, visibility, and glossy-metal receiver response', () => {
    expect(code).toContain('hit.instanceIndex == facet.instanceIndex &&');
    expect(code).toContain('fn mneeSegmentBlockedExceptFacet(');
    const unified = code.slice(code.indexOf('fn boundedManifoldCaustic('), code.indexOf('fn manifoldNeeContribution('));
    expect(unified).toContain('let fr = evaluateBrdfFullWithClearcoatNormal(');
    expect(unified).not.toContain('causticReceiverRejected');
  });

  it('composes one estimator after the solver in both variants', () => {
    for (const source of [PT_WEBGPU_TRACE_WGSL, composePtWebgpuTraceWgsl(true)]) {
      expect(source.indexOf('fn mneeNewtonSolveChainBounded(')).toBeGreaterThan(0);
      expect(source.indexOf('fn boundedManifoldCaustic('))
        .toBeGreaterThan(source.indexOf('fn mneeNewtonSolveChainBounded('));
      expect(source).not.toContain('fn pointLightReflectionCaustic(');
    }
  });
});
