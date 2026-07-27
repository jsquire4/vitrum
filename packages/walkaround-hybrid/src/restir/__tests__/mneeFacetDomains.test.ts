import { describe, expect, it } from 'vitest';
import { packMneeFacetDomains } from '../bvhCore.js';

describe('compact MNEE facet domains', () => {
  it('represents a huge triangle-instance Cartesian product in O(bindings)', () => {
    const records = packMneeFacetDomains([{
      triStart: 17,
      triCount: 2,
      instanceStart: 23,
      instanceCount: 1_000_000_000,
    }]);
    expect(records).toHaveLength(8);
    expect(Array.from(records.slice(0, 4))).toEqual([17, 2, 23, 1_000_000_000]);
    expect(new Float32Array(records.buffer)[6]).toBe(1);
  });

  it('stores the PMF represented by the quantized alias table', () => {
    const records = packMneeFacetDomains([
      { triStart: 0, triCount: 1, instanceStart: 0, instanceCount: 1 },
      { triStart: 1, triCount: 3, instanceStart: 1, instanceCount: 1 },
    ]);
    const f32 = new Float32Array(records.buffer);
    expect(f32[6]).toBeCloseTo(0.25, 6);
    expect(f32[14]).toBeCloseTo(0.75, 6);
  });

  it('rejects unsafe or overflowing half-open domains', () => {
    expect(() => packMneeFacetDomains([{
      triStart: 0xffff_ffff,
      triCount: 2,
      instanceStart: 0,
      instanceCount: 1,
    }])).toThrow(/half-open range/);
    expect(() => packMneeFacetDomains([{
      triStart: 0,
      triCount: 0xffff_ffff,
      instanceStart: 0,
      instanceCount: 0xffff_ffff,
    }])).toThrow(/safe integer/);
  });
});
