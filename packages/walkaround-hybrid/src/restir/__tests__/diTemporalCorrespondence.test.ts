import { describe, expect, it } from 'vitest';
import {
  diTemporalSurfaceCorresponds,
  type DiCorrespondenceSurface,
} from './support/diTemporalCorrespondence.js';
import { TEMPORAL_WGSL } from '../../shaders/temporal.wgsl.js';

const current: DiCorrespondenceSurface = {
  hit: true,
  position: [0, 0, 10],
  normal: [0, 0, 1],
  depth: 10,
  triangleId: 17,
  instanceId: 4,
  materialKey: 0x12345678,
};

describe('ReSTIR-DI temporal correspondence', () => {
  it('accepts the same recast surface within subpixel tolerances', () => {
    expect(diTemporalSurfaceCorresponds(
      current,
      {
        ...current,
        position: [0.01, -0.01, 10.01],
        depth: 10.01,
      },
      [0, 0, 0],
      0.001,
    )).toBe(true);
  });

  it.each([
    ['animated instance', { instanceId: 5 }],
    ['topology change', { triangleId: 18 }],
    ['material replacement', { materialKey: 0x87654321 }],
    ['depth disocclusion', { depth: 8 }],
    ['world-position disocclusion', { position: [0, 0, 8] as const, depth: 10 }],
    ['normal discontinuity', { normal: [0, 0, -1] as const }],
  ])('rejects %s', (_label, mutation) => {
    expect(diTemporalSurfaceCorresponds(
      current,
      { ...current, ...mutation },
      [0, 0, 0],
      0.001,
    )).toBe(false);
  });

  it('recasts the historical pixel against the current BVH before reuse', () => {
    expect(TEMPORAL_WGSL).toContain('castPrimaryFromInvVP');
    expect(TEMPORAL_WGSL).toContain('temporalSurfaceCorresponds');
    expect(TEMPORAL_WGSL).toContain(
      'previousSurfaceNow.instanceId != currentSurface.instanceId',
    );
    expect(TEMPORAL_WGSL).not.toContain('implicit gate');
  });

  it('allows valid previous history to recover an empty current reservoir safely', () => {
    expect(TEMPORAL_WGSL).toContain('if (previous.M == 0u)');
    expect(TEMPORAL_WGSL).not.toContain('if (current.M == 0u)');
    expect(TEMPORAL_WGSL).toContain(
      'if (currentSupport > 0u && reservoirDiHasEstimatorNumerator(current))',
    );
    expect(TEMPORAL_WGSL).toContain(
      'if (wrs.hasSelection)',
    );
    expect(TEMPORAL_WGSL).not.toMatch(/current\.W\s*>\s*0\.0/);
  });
});
