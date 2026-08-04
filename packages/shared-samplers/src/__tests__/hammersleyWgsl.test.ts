import { describe, expect, it } from 'vitest';

import { HAMMERSLEY_WGSL } from '../wgsl/hammersley.wgsl.js';

function reverseBits32(value: number): number {
  let bits = value >>> 0;
  bits = ((bits << 16) | (bits >>> 16)) >>> 0;
  bits = (((bits & 0x5555_5555) << 1) | ((bits & 0xaaaa_aaaa) >>> 1)) >>> 0;
  bits = (((bits & 0x3333_3333) << 2) | ((bits & 0xcccc_cccc) >>> 2)) >>> 0;
  bits = (((bits & 0x0f0f_0f0f) << 4) | ((bits & 0xf0f0_f0f0) >>> 4)) >>> 0;
  return (((bits & 0x00ff_00ff) << 8) | ((bits & 0xff00_ff00) >>> 8)) >>> 0;
}

function radicalInverseVdc24(value: number): number {
  return (reverseBits32(value) >>> 8) * 2 ** -24;
}

describe('HAMMERSLEY_WGSL', () => {
  it('uses an exactly representable 24-bit [0, 1) radical-inverse lattice', () => {
    expect(HAMMERSLEY_WGSL).toContain(
      'return f32(bits >> 8u) * 5.960464477539063e-8;',
    );
    expect(HAMMERSLEY_WGSL).not.toContain(
      'return f32(bits) * 2.3283064365386963e-10;',
    );

    for (const value of [0, 1, 2, 3, 0x7fff_ffff, 0xffff_fffe, 0xffff_ffff]) {
      const sample = radicalInverseVdc24(value);
      expect(sample).toBeGreaterThanOrEqual(0);
      expect(sample).toBeLessThan(1);
      expect(Math.fround(sample)).toBe(sample);
    }
  });
});
