import { describe, expect, it } from 'vitest';
import { accumulationSeed } from './accumulationSeed.js';

describe('accumulationSeed', () => {
  it('is deterministic and changes with every input domain', () => {
    const base = accumulationSeed(37, 11, 5);
    expect(accumulationSeed(37, 11, 5)).toBe(base);
    expect(accumulationSeed(38, 11, 5)).not.toBe(base);
    expect(accumulationSeed(37, 12, 5)).not.toBe(base);
    expect(accumulationSeed(37, 11, 6)).not.toBe(base);
  });

  it('does not collapse the accumulated-sample sequence for fixed host inputs', () => {
    const seeds = Array.from(
      { length: 4_096 },
      (_, sample) => accumulationSeed(0xffff_ffff, 0xffff_ffff, sample),
    );
    expect(new Set(seeds).size).toBe(seeds.length);
  });

  it('returns a signed GLint while retaining the full uint32 bit domain', () => {
    for (const value of [
      accumulationSeed(0, 0, 0),
      accumulationSeed(0xffff_ffff, 0xffff_ffff, 0xffff_ffff),
    ]) {
      expect(Number.isInteger(value)).toBe(true);
      expect(value).toBeGreaterThanOrEqual(-0x8000_0000);
      expect(value).toBeLessThanOrEqual(0x7fff_ffff);
    }
  });
});
