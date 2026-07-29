import { describe, expect, it } from 'vitest';
import { NRC_INDEPENDENT_SUFFIX_WGSL } from '../nrcIndependentSuffix.wgsl.js';
import { TRANSPARENT_OIT_WGSL } from '../transparentOit.wgsl.js';

function sourceBetween(source: string, start: string, end: string): string {
  const startIndex = source.indexOf(start);
  const endIndex = source.indexOf(end, startIndex);
  expect(startIndex).toBeGreaterThanOrEqual(0);
  expect(endIndex).toBeGreaterThan(startIndex);
  return source.slice(startIndex, endIndex);
}

describe('emitter distance-floor parity', () => {
  it('regularizes transparent OIT mesh-area geometry terms with the runtime floor', () => {
    const areaNee = sourceBetween(
      TRANSPARENT_OIT_WGSL,
      'fn oitLayerAreaEmitterNEE(',
      'fn oitLayerRadiance(',
    );

    expect(areaNee).toContain(
      'let G = emitterGeometry(nlDotL, dist2, ubo.emitterDist2Floor);',
    );
    expect(areaNee).not.toContain('nlDotL / dist2');
  });

  it('regularizes NRC teacher mesh-area geometry terms with the runtime floor', () => {
    const areaNee = sourceBetween(
      NRC_INDEPENDENT_SUFFIX_WGSL,
      'fn nrcTeacherAreaNee(',
      'fn nrcTeacherAnalyticNee(',
    );

    expect(areaNee).toContain(
      'let geometry = emitterGeometry(cosLight, dist2, ubo.emitterDist2Floor);',
    );
    expect(areaNee).not.toContain('cosLight / dist2');
  });
});
