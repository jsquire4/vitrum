/**
 * Smoke test: AnalyticParamsByShape typed tuple map and ANALYTIC_PARAM_LENGTH
 * are importable from the @vitrum/core public surface (core/src/index.ts →
 * scene/index.ts → analyticParams.ts).
 *
 * Hosts building primitives programmatically need the typed layout to encode
 * params without relying on an internal path import.
 */

import { describe, expect, it } from 'vitest';
import {
  ANALYTIC_PARAM_LENGTH,
  encodeAnalyticParams,
  decodeAnalyticParams,
  type AnalyticParamsByShape,
} from '../index.js';

describe('AnalyticParamsByShape + ANALYTIC_PARAM_LENGTH public export (@vitrum/core)', () => {
  it('ANALYTIC_PARAM_LENGTH is exported from the package root with all five shapes', () => {
    expect(ANALYTIC_PARAM_LENGTH.sphere).toBe(4);
    expect(ANALYTIC_PARAM_LENGTH.box).toBe(6);
    expect(ANALYTIC_PARAM_LENGTH.capsule).toBe(7);
    expect(ANALYTIC_PARAM_LENGTH.cylinder).toBe(5);
    expect(ANALYTIC_PARAM_LENGTH['h-channel-came']).toBe(4);
  });

  it('AnalyticParamsByShape typed tuple round-trips via encodeAnalyticParams / decodeAnalyticParams', () => {
    // Compile-time: the params variable is typed via AnalyticParamsByShape.
    const params: AnalyticParamsByShape['sphere'] = [1, 2, 3, 0.5];
    const packed = encodeAnalyticParams('sphere', params);
    const decoded = decodeAnalyticParams('sphere', packed);
    expect(decoded).toEqual([1, 2, 3, 0.5]);
  });

  it('box params encode and decode correctly', () => {
    const params: AnalyticParamsByShape['box'] = [0, 0, 0, 1, 2, 3];
    const packed = encodeAnalyticParams('box', params);
    expect(packed.length).toBe(ANALYTIC_PARAM_LENGTH.box);
    expect(Array.from(decodeAnalyticParams('box', packed))).toEqual([0, 0, 0, 1, 2, 3]);
  });
});
