import { describe, expect, it } from 'vitest';
import * as BsdfFns from '../bsdf_functions.glsl.js';

const bsdf_functions = (BsdfFns as unknown as Record<string, string>)['bsdf_functions']!;

function sourceBetween(startNeedle: string, endNeedle: string): string {
  const start = bsdf_functions.indexOf(startNeedle);
  expect(start, `${startNeedle} must exist`).toBeGreaterThanOrEqual(0);
  const end = bsdf_functions.indexOf(endNeedle, start + startNeedle.length);
  expect(end, `${endNeedle} must follow ${startNeedle}`).toBeGreaterThan(start);
  return bsdf_functions.slice(start, end);
}

function sourceFrom(startNeedle: string): string {
  const start = bsdf_functions.indexOf(startNeedle);
  expect(start, `${startNeedle} must exist`).toBeGreaterThanOrEqual(0);
  return bsdf_functions.slice(start);
}

describe('pt-webgl2 BSDF lobe-selection PDF policy', () => {
  it('uses one sampling-lobe policy helper for sampled and evaluated PDFs', () => {
    const helper = sourceBetween('void getSamplingLobeWeights', 'float bsdfEval');
    expect(helper).toContain('getLobeWeights( wo, wo, vec3( 0, 0, 1 ), clearcoatWo, surf');

    const result = sourceBetween('float bsdfResult', '// Sprint 7: SSS');
    const sample = sourceFrom('ScatterRecord bsdfSample');
    expect(result).toContain('getSamplingLobeWeights( wo, clearcoatWo, surf');
    expect(sample).toContain('getSamplingLobeWeights( wo, clearcoatWo, surf');
  });

  it('does not recompute mixture probabilities from arbitrary light directions in bsdfResult', () => {
    const result = sourceBetween('float bsdfResult', '// Sprint 7: SSS');
    expect(result).not.toContain('getLobeWeights( wo, wi');
    expect(result).not.toContain('getHalfVector( wo, wi');
  });
});
