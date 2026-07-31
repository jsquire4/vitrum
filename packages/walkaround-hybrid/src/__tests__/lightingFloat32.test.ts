import { describe, expect, it } from 'vitest';
import {
  canonicalizeLightingDirectionF32,
  multiplyNonNegativeLightingFloat32,
  packFiniteLightingFloat32,
  packLightingRgbScaleEnvelopeF32,
  packNonNegativeLightingFloat32,
  packNonNegativeLightingRgbF32,
} from '../lightingFloat32.js';
import { snapshotDdgiLights } from '../ddgi/types.js';

const F32_MAX = Math.fround(3.4028234663852886e38);
const F32_MIN_SUBNORMAL = Math.fround(1.401298464324817e-45);

describe('lighting Float32 publication policy', () => {
  it('distinguishes authored zero from positive scalar underflow', () => {
    expect(packNonNegativeLightingFloat32(0, 'light')).toBe(0);
    expect(packNonNegativeLightingFloat32(F32_MIN_SUBNORMAL, 'light'))
      .toBe(F32_MIN_SUBNORMAL);
    expect(() => packNonNegativeLightingFloat32(Number.MIN_VALUE, 'light'))
      .toThrow(/remain positive/);
  });

  it('rejects signed and non-negative scalar overflow before publication', () => {
    expect(packFiniteLightingFloat32(F32_MAX, 'coordinate')).toBe(F32_MAX);
    expect(() => packFiniteLightingFloat32(Number.MAX_VALUE, 'coordinate'))
      .toThrow(/remain finite/);
    expect(() => packNonNegativeLightingFloat32(Number.MAX_VALUE, 'radiance'))
      .toThrow(/remain finite/);
  });

  it('normalizes huge and subnormal directions stably into fresh f32 tuples', () => {
    const huge: [number, number, number] = [
      Number.MAX_VALUE,
      Number.MAX_VALUE,
      Number.MAX_VALUE,
    ];
    const packedHuge = canonicalizeLightingDirectionF32(huge, 'sun');
    expect(Math.hypot(...packedHuge)).toBeCloseTo(1, 6);
    expect(packedHuge[0]).toBeCloseTo(1 / Math.sqrt(3), 6);

    const tiny: [number, number, number] = [Number.MIN_VALUE, 0, 0];
    const packedTiny = canonicalizeLightingDirectionF32(tiny, 'sun');
    expect(packedTiny).toEqual([1, 0, 0]);

    huge[0] = 0;
    expect(packedHuge[0]).not.toBe(0);
  });

  it('allows one RGB lane to underflow but rejects complete non-black collapse', () => {
    expect(packNonNegativeLightingRgbF32(
      [Number.MIN_VALUE, 1, 0],
      'color',
    )).toEqual([0, 1, 0]);
    expect(() => packNonNegativeLightingRgbF32(
      [Number.MIN_VALUE, Number.MIN_VALUE, Number.MIN_VALUE],
      'color',
    )).toThrow(/collapse completely/);
  });

  it('rejects overflow and positive collapse in scalar f32 products', () => {
    expect(multiplyNonNegativeLightingFloat32(0, F32_MAX, 'product')).toBe(0);
    expect(() => multiplyNonNegativeLightingFloat32(F32_MAX, 2, 'product'))
      .toThrow(/remain finite/);
    expect(() => multiplyNonNegativeLightingFloat32(
      F32_MIN_SUBNORMAL,
      F32_MIN_SUBNORMAL,
      'product',
    )).toThrow(/underflow/);
  });

  it('returns the exact staged RGB×intensity tuple and rejects total collapse', () => {
    expect(packLightingRgbScaleEnvelopeF32(
      [0.25, Number.MIN_VALUE, 1],
      2,
      'sun',
    )).toEqual({
      value: [0.25, 0, 1],
      scale: 2,
      scaled: [0.5, 0, 2],
    });
    expect(() => packLightingRgbScaleEnvelopeF32(
      [F32_MIN_SUBNORMAL, 0, 0],
      F32_MIN_SUBNORMAL,
      'sun',
    )).toThrow(/underflow completely/);
    expect(() => packLightingRgbScaleEnvelopeF32(
      [2, 0, 0],
      F32_MAX,
      'sun',
    )).toThrow(/remain finite/);
  });

  it('deep-snapshots and canonicalizes every nested DDGI light field', () => {
    const host = [{
      kind: 'fixture' as const,
      on: true,
      intensity: 1.00000001,
      position: { x: 1.00000001, y: Number.MIN_VALUE, z: -2 },
      color: { r: 0.25, g: Number.MIN_VALUE, b: 1 },
      spotAxis: { x: 10, y: 0, z: 0 },
      spotCosInner: 0.90000001,
      spotCosOuter: 0.80000001,
      distance: 4.00000001,
      decay: 2.00000001,
    }];
    const snapshot = snapshotDdgiLights(host);
    host[0]!.position.x = 99;
    host[0]!.color.r = 99;
    host[0]!.spotAxis.x = -10;

    expect(snapshot).toEqual([{
      kind: 'fixture',
      on: true,
      intensity: Math.fround(1.00000001),
      position: { x: Math.fround(1.00000001), y: 0, z: -2 },
      color: { r: 0.25, g: 0, b: 1 },
      spotAxis: { x: 1, y: 0, z: 0 },
      spotCosInner: Math.fround(0.90000001),
      spotCosOuter: Math.fround(0.80000001),
      distance: Math.fround(4.00000001),
      decay: Math.fround(2.00000001),
    }]);
  });
});
