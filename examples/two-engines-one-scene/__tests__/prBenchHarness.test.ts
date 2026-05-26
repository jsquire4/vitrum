import { describe, expect, it } from 'vitest';
import { percentile95 } from '../src/prBenchHarness.js';

describe('percentile95', () => {
  it('returns the 95th percentile of sorted samples', () => {
    const values = Array.from({ length: 100 }, (_, i) => i + 1);
    expect(percentile95(values)).toBe(96);
  });

  it('returns 0 for empty input', () => {
    expect(percentile95([])).toBe(0);
  });
});
