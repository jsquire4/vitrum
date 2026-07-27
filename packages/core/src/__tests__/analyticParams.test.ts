import { describe, expect, it } from 'vitest';
import {
  ANALYTIC_PARAM_LENGTH,
  decodeAnalyticParams,
  encodeAnalyticParams,
  validateAnalyticParams,
} from '../scene/analyticParams.js';

describe('analyticParams', () => {
  it('round-trips sphere params', () => {
    const packed = encodeAnalyticParams('sphere', [0, 1, 2, 0.5]);
    expect(packed.length).toBe(ANALYTIC_PARAM_LENGTH.sphere);
    expect(decodeAnalyticParams('sphere', packed)).toEqual([0, 1, 2, 0.5]);
  });

  it('validateAnalyticParams rejects wrong length', () => {
    expect(() => validateAnalyticParams('box', new Float32Array(4))).toThrow(/expects 6/);
  });

  it('rejects non-finite values before tessellation can clamp or propagate them', () => {
    expect(() => validateAnalyticParams(
      'sphere',
      new Float32Array([0, 0, 0, Number.NaN]),
    )).toThrow(/params\[3\].*finite/);
    expect(() => encodeAnalyticParams('box', [0, 0, 0, 1, Number.POSITIVE_INFINITY, 1]))
      .toThrow(/params\[4\].*finite/);
  });

  it.each([
    ['sphere', [0, 0, 0, 0], 'radius'],
    ['box', [0, 0, 0, 1, -1, 1], 'half-height'],
    ['capsule', [0, 0, 0, 0, 1, 0, -0.1], 'radius'],
    ['cylinder', [0, 0, 0, 1, 0], 'half-height'],
  ] as const)('rejects invalid %s dimensions', (shape, values, label) => {
    expect(() => validateAnalyticParams(shape, new Float32Array(values)))
      .toThrow(new RegExp(`${label}.*> 0`));
  });

  it('rejects an H-channel web that collapses its authored profile', () => {
    expect(() => validateAnalyticParams(
      'h-channel-came',
      new Float32Array([10, 2, 4, 2]),
    )).toThrow(/webThickness must be smaller/);
  });

  it('rejects runtime shapes and packed-array types outside the public contract', () => {
    expect(() => validateAnalyticParams(
      'torus' as never,
      new Float32Array(4),
    )).toThrow(/unsupported shape/);
    expect(() => validateAnalyticParams(
      'sphere',
      [0, 0, 0, 1] as unknown as Float32Array,
    )).toThrow(/Float32Array/);
  });
});
