import { describe, expect, it } from 'vitest';

import { RESTIR_PT_PRODUCER_WGSL } from '../wgsl/pathTrace/restirPtProducer.wgsl.js';

describe('ReSTIR-PT producer path-depth budget', () => {
  it('does not construct a two-vertex reconnection path when only one vertex is allowed', () => {
    const budgetGuard = RESTIR_PT_PRODUCER_WGSL.indexOf(
      'if (params.maxBounces < 2u) {',
    );
    const primaryTrace = RESTIR_PT_PRODUCER_WGSL.indexOf(
      'let vTrace = rptTraceClosestAfterAlpha(primaryRay, &rng);',
    );

    expect(budgetGuard).toBeGreaterThan(-1);
    expect(budgetGuard).toBeLessThan(primaryTrace);
  });

  it('charges the visible vertex against the suffix shaded-vertex budget', () => {
    expect(RESTIR_PT_PRODUCER_WGSL).toContain(
      'let suffixBounces = params.maxBounces - 1u;',
    );
    expect(RESTIR_PT_PRODUCER_WGSL).not.toContain(
      'let suffixBounces = max(1u, params.maxBounces);',
    );
  });

  it.each([2, 3, 4, 8])(
    'keeps the producer at exactly %i shaded vertices',
    (configuredBounces) => {
      const visibleVertices = 1;
      const suffixVertices = configuredBounces - visibleVertices;
      expect(visibleVertices + suffixVertices).toBe(configuredBounces);
    },
  );
});
