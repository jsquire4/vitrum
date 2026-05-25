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
});
