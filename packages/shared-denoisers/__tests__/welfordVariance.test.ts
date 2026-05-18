import { describe, expect, it } from 'vitest';
import {
  WELFORD_VARIANCE_WGSL,
  WELFORD_VARIANCE_VERSION,
  ATROUS_VARIANCE_WGSL,
} from '../src/index.js';
import { COMMON_WGSL } from '../../walkaround-hybrid/src/shaders/common.wgsl.js';

describe('WelfordVariance canonical WGSL', () => {
  it('declares the v1 struct fields in the locked order', () => {
    // Any field-order change must update WELFORD_VARIANCE_VERSION and every
    // consumer's reads.
    expect(WELFORD_VARIANCE_WGSL).toContain('struct WelfordVariance {');
    expect(WELFORD_VARIANCE_WGSL).toMatch(
      /struct WelfordVariance \{\s*mean: f32,\s*m2:\s*f32,\s*\}/,
    );
  });

  it('carries the RG32Float layout documentation that consumers test for', () => {
    expect(WELFORD_VARIANCE_WGSL).toContain('Decision 13');
    expect(WELFORD_VARIANCE_WGSL).toContain('RG32Float');
    expect(WELFORD_VARIANCE_WGSL).toContain('r = mean');
    expect(WELFORD_VARIANCE_WGSL).toContain('g = M2');
  });

  it('pins version 1', () => {
    expect(WELFORD_VARIANCE_VERSION).toBe(1);
  });

  it('is injected into walkaround-hybrid COMMON_WGSL', () => {
    expect(COMMON_WGSL).toContain('struct WelfordVariance');
    expect(COMMON_WGSL).toContain('fn welfordUpdate');
    expect(COMMON_WGSL).toContain('fn welfordVariance');
  });

  it('is injected into ATROUS_VARIANCE_WGSL', () => {
    expect(ATROUS_VARIANCE_WGSL).toContain('struct WelfordVariance');
    expect(ATROUS_VARIANCE_WGSL).toContain('fn welfordVariance');
  });

  it('appears exactly once in each consumer (no duplicate declarations)', () => {
    const commonOccurrences = COMMON_WGSL.split('struct WelfordVariance').length - 1;
    const atrousVarianceOccurrences =
      ATROUS_VARIANCE_WGSL.split('struct WelfordVariance').length - 1;
    expect(commonOccurrences).toBe(1);
    expect(atrousVarianceOccurrences).toBe(1);
  });
});
