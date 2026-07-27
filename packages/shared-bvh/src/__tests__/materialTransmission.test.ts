import { describe, expect, it } from 'vitest';

import {
  buildMaterialTransmissionPredicatesWGSL,
  packedMaterialHasTransmission,
  quantizePackedMaterialTransmission,
} from '../wgsl/materialTransmission.wgsl.js';

describe('canonical packed physical-transmission semantics', () => {
  it('reserves code zero for opaque and preserves every finite positive value', () => {
    expect(quantizePackedMaterialTransmission(0)).toBe(0);
    expect(quantizePackedMaterialTransmission(-1)).toBe(0);
    expect(quantizePackedMaterialTransmission(Number.NaN)).toBe(0);
    expect(quantizePackedMaterialTransmission(Number.POSITIVE_INFINITY)).toBe(0);
    expect(quantizePackedMaterialTransmission(Number.MIN_VALUE)).toBe(1);
    expect(quantizePackedMaterialTransmission(0.001)).toBe(1);
    expect(quantizePackedMaterialTransmission(1)).toBe(15);
    expect(quantizePackedMaterialTransmission(2)).toBe(15);
  });

  it('reads only the physical-transmission nibble, never the low metadata nibble', () => {
    expect(packedMaterialHasTransmission(0x00)).toBe(false);
    expect(packedMaterialHasTransmission(0x0F)).toBe(false);
    expect(packedMaterialHasTransmission(0x10)).toBe(true);
    expect(packedMaterialHasTransmission(0xF0)).toBe(true);
  });

  it('generates identical nonzero predicates for packed and sampled forms', () => {
    const wgsl = buildMaterialTransmissionPredicatesWGSL({
      packedFunctionName: 'packedHasTransmission',
      sampledFunctionName: 'sampledHasTransmission',
    });
    expect(wgsl).toContain('((packedMaterial >> 4u) & 0xFu) != 0u');
    expect(wgsl).toContain('return transmission > 0.0;');
    expect(wgsl).not.toContain('opacity');
    expect(wgsl).not.toMatch(/[<>]=?\s*0\.[1-9]/);
  });
});
