import { describe, expect, it } from 'vitest';

import {
  packRadianceRgbProductF32,
  packRadianceRgbScaleF32,
} from '../radianceFloat32.js';

describe('canonical radiance Float32 publication', () => {
  it('folds RGB×intensity in the exact staged shader order', () => {
    const packed = packRadianceRgbScaleF32(
      [0.1, 0.2, 0.3],
      0.7,
      'test emitter',
    );
    const expected = [0.1, 0.2, 0.3].map((channel) =>
      Math.fround(Math.fround(channel) * Math.fround(0.7))
    );
    expect(packed.value).toEqual([
      Math.fround(0.1),
      Math.fround(0.2),
      Math.fround(0.3),
    ]);
    expect(packed.scale).toBe(Math.fround(0.7));
    expect(packed.scaled).toEqual(expected);
  });

  it('rejects binary32 overflow and a positive product collapsing to black', () => {
    expect(() => packRadianceRgbScaleF32(
      [3e38, 0, 0],
      2,
      'overflow emitter',
    )).toThrow(/remain finite/i);

    const minSubnormal = Math.fround(2 ** -149);
    expect(() => packRadianceRgbScaleF32(
      [minSubnormal, 0, 0],
      0.5,
      'underflow emitter',
    )).toThrow(/underflow completely/i);
  });

  it('publishes mapped RGB modulation exactly and rejects total underflow', () => {
    const product = packRadianceRgbProductF32(
      [0.1, 0.2, 0.3],
      [0.7, 0.6, 0.5],
      'mapped emitter',
    );
    expect(product).toEqual([
      Math.fround(Math.fround(0.1) * Math.fround(0.7)),
      Math.fround(Math.fround(0.2) * Math.fround(0.6)),
      Math.fround(Math.fround(0.3) * Math.fround(0.5)),
    ]);

    const minSubnormal = Math.fround(2 ** -149);
    expect(() => packRadianceRgbProductF32(
      [minSubnormal, 0, 0],
      [0.5, 1, 1],
      'mapped underflow',
    )).toThrow(/underflow completely/i);
  });
});
