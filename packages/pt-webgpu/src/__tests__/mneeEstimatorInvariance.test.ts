import { describe, expect, it } from 'vitest';
import { PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL } from '../wgsl/pathTrace/caustic.wgsl.js';

function expectedUniformEmitterEstimate(contributions: readonly number[]): number {
  const count = contributions.length;
  return contributions.reduce((sum, contribution) => sum + contribution * count, 0) / count;
}

function expectedUniformFacetEstimate(contributions: readonly number[]): number {
  const domainSize = contributions.length;
  return contributions.reduce(
    (expectation, contribution) => expectation + contribution / domainSize * domainSize,
    0,
  );
}

function expectedUniformOrderedPairEstimate(contributions: readonly (readonly number[])[]): number {
  const domainSize = contributions.length;
  const pairPmf = 1 / (domainSize * domainSize);
  return contributions.reduce(
    (outer, row) => outer + row.reduce(
      (inner, contribution) => inner + pairPmf * contribution / pairPmf,
      0,
    ),
    0,
  );
}

describe('MNEE estimator invariants', () => {
  it('recovers the sum of every point emitter with one uniform emitter sample', () => {
    const contributions = [0.125, 1.75, 0, 4.5, 0.625];
    expect(expectedUniformEmitterEstimate(contributions)).toBeCloseTo(
      contributions.reduce((sum, contribution) => sum + contribution, 0),
      13,
    );
  });

  it('does not condition point-emitter selection on a non-zero contribution', () => {
    expect(expectedUniformEmitterEstimate([3, 0])).toBeCloseTo(3, 13);
  });

  it('is invariant when one point emitter is split into equivalent emitters', () => {
    const total = 7.5;
    for (const count of [1, 2, 4, 8, 32]) {
      expect(expectedUniformEmitterEstimate(Array.from({ length: count }, () => total / count)))
        .toBeCloseTo(total, 13);
    }
  });

  it('keeps the finite-area joint density invariant under equal-area partitioning', () => {
    const totalArea = 12;
    const manifoldJacobian = 0.375;
    const expectedJointPdf = 1 / totalArea / manifoldJacobian;

    for (const count of [1, 2, 4, 8]) {
      const selectionPdf = 1 / count;
      const conditionalAreaPdf = 1 / (totalArea / count) / manifoldJacobian;
      expect(selectionPdf * conditionalAreaPdf).toBeCloseTo(expectedJointPdf, 13);
    }
  });

  it('recovers heterogeneous finite-emitter integrals under uniform emitter selection', () => {
    const emitters = [
      { area: 1.5, radiance: 2, jacobian: 0.5 },
      { area: 4, radiance: 0.25, jacobian: 0.75 },
      { area: 0.5, radiance: 9, jacobian: 0.2 },
    ];
    const selectionPdf = 1 / emitters.length;
    const expected = emitters.reduce(
      (sum, emitter) => sum + emitter.radiance * emitter.area * emitter.jacobian,
      0,
    );
    const enumeratedExpectation = emitters.reduce((sum, emitter) => {
      const conditionalPdf = 1 / emitter.area / emitter.jacobian;
      const jointPdf = selectionPdf * conditionalPdf;
      return sum + selectionPdf * emitter.radiance / jointPdf;
    }, 0);
    expect(enumeratedExpectation).toBeCloseTo(expected, 13);
  });

  it('recovers every exact TLAS facet-table member', () => {
    const perFacet = [1.25, 3.5, 0.125];
    expect(expectedUniformFacetEstimate(perFacet)).toBeCloseTo(4.875, 13);
  });

  it('recovers ordered two-interface chains with the full pair PMF', () => {
    const perPair = [
      [0, 0, 0.75],
      [0, 0, 0],
      [0, 2.25, 0],
    ];
    expect(expectedUniformOrderedPairEstimate(perPair)).toBeCloseTo(3, 13);
  });
});

describe('MNEE estimator implementation pins', () => {
  it('uses one unbiased explicit-emitter draw and its reciprocal PMF', () => {
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain(
      'let picked = causticUniformEmitterIndex(rng, total);',
    );
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain(
      'out.selectionPdf = 1.0 / f32(total);',
    );
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain(
      'log(lengthSelectionPdf) - log(emitter.selectionPdf)',
    );
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).not.toContain(
      'min(params.pointLightCount, 16u)',
    );
  });

  it('uses the joint emitter-selection and conditional-area density exactly once', () => {
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain(
      'let endpointPdf = (1.0 / emitter.area) / areaDet;',
    );
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain(
      'if (!(endpointPdf > 0.0) || !(endpointPdf < INFINITY))',
    );
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain(
      'log(lengthSelectionPdf) + log(emitter.selectionPdf)',
    );
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain(
      'pathMeasure = nDotL * emitter.area * areaDet;',
    );
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).not.toContain(
      'pathMeasure = nDotL * emitter.area * areaDet * misWeight;',
    );
  });

  it('uses exact TLAS identities and a full-support guided/uniform mixture', () => {
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain(
      'fn mneeProposeConditionalFacet(',
    );
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain(
      'selected.pdf = 1.0 / f32(count);',
    );
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain(
      'selected.pdf = (1.0 - MNEE_GUIDED_MIX_PROBABILITY) / f32(count);',
    );
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain(
      'hit.triIndex == facet.triIndex &&',
    );
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).toContain(
      'hit.instanceIndex == facet.instanceIndex &&',
    );
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).not.toContain('let seedHit =');
    expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).not.toContain('found = true;');
  });

  it('has one unified ownership path and no retired specialized estimators', () => {
    const dispatcher = PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL.slice(
      PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL.indexOf('fn manifoldNeeContribution('),
      PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL.indexOf('// ── SPPM gather'),
    );
    expect(dispatcher).toContain('return boundedManifoldCaustic(');
    for (const retired of [
      'pointLightReflectionCaustic', 'finiteAreaReflectionCaustic',
      'pointLightRefractionCaustic', 'pointLightGlassSlabCaustic',
      'pointLightBoundedChainCaustic',
    ]) {
      expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).not.toContain(retired);
    }
    for (const removed of [
      'perturbAroundDirection', 'traceSpecularTransmissiveChain',
      'conePdf', 'align <= 0.75', 'transmissiveContribution',
    ]) {
      expect(PT_WEBGPU_PATH_TRACE_CAUSTIC_WGSL).not.toContain(removed);
    }
  });
});
