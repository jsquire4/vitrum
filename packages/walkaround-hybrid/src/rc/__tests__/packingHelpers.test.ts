import { describe, expect, it } from 'vitest';
import type { DDGILight } from '../../ddgi/types.js';
import {
  packRCLights,
  RCLightBufferHeaderOffset,
  RCLightEntryOffset,
} from '../packingHelpers.js';

const makeFixtureLights = (count: number): DDGILight[] =>
  Array.from({ length: count }, (_, i) => ({
    kind: 'fixture',
    on: true,
    intensity: 1,
    position: { x: i, y: 0, z: 0 },
  }));

describe('packRCLights runtime alias ABI', () => {
  it('retains every active light beyond the former fixed cap', () => {
    const packed = packRCLights(makeFixtureLights(17));
    const words = new Uint32Array(packed);
    expect(words[0]).toBe(17);
    expect(words[1]).toBe(4);
    expect(words[2]).toBe(4 + 17 * 16);
    expect(packed.byteLength).toBe(16 + 17 * (64 + 16));
  });

  it('stores positive represented PMFs for every positive-power light', () => {
    const packed = packRCLights(makeFixtureLights(17));
    const words = new Uint32Array(packed);
    const floats = new Float32Array(packed);
    const aliasWord = words[2]!;
    let sum = 0;
    for (let index = 0; index < 17; index += 1) {
      const pmf = floats[aliasWord + index * 4 + 2]!;
      expect(pmf).toBeGreaterThan(0);
      sum += pmf;
    }
    expect(sum).toBeCloseTo(1, 6);
  });

  it('publishes exact f32 radiance operands and normalized directions', () => {
    const packed = packRCLights([{
      kind: 'sun',
      on: true,
      intensity: 0.2,
      direction: { x: 0, y: -10, z: 0 },
      color: { r: 0.1, g: 0.3, b: 0.7 },
      angularRadius: 0.01,
    }]);
    const words = new Uint32Array(packed);
    const floats = new Float32Array(packed);
    const base = words[RCLightBufferHeaderOffset.entriesWordOffset / 4]!;
    expect(floats[base + RCLightEntryOffset.intensity / 4])
      .toBe(Math.fround(0.2));
    expect(Array.from(floats.slice(
      base + RCLightEntryOffset.direction / 4,
      base + RCLightEntryOffset.direction / 4 + 3,
    ))).toEqual([0, -1, 0]);
    expect(Array.from(floats.slice(
      base + RCLightEntryOffset.color / 4,
      base + RCLightEntryOffset.color / 4 + 3,
    ))).toEqual([
      Math.fround(0.1),
      Math.fround(0.3),
      Math.fround(0.7),
    ]);
  });

  it('keeps host-only alias weights in f64 for maximum finite f32 radiance', () => {
    const f32Max = Math.fround(3.4028234663852886e38);
    const packed = packRCLights([{
      kind: 'fixture',
      on: true,
      intensity: f32Max,
      position: { x: 0, y: 0, z: 0 },
      color: { r: 1, g: 1, b: 1 },
    }]);
    const words = new Uint32Array(packed);
    const floats = new Float32Array(packed);
    const entry = words[RCLightBufferHeaderOffset.entriesWordOffset / 4]!;
    const alias = words[RCLightBufferHeaderOffset.aliasWordOffset / 4]!;
    expect(floats[entry + RCLightEntryOffset.intensity / 4]).toBe(f32Max);
    expect(floats[alias]).toBe(1);
    expect(floats[alias + 2]).toBe(1);
  });

  it('rejects invalid, overflowing, and completely collapsing radiance', () => {
    expect(() => packRCLights([{
      kind: 'fixture',
      on: true,
      intensity: -1,
      position: { x: 0, y: 0, z: 0 },
    }])).toThrow(/finite and non-negative/);

    expect(() => packRCLights([{
      kind: 'fixture',
      on: true,
      intensity: 2,
      position: { x: 0, y: 0, z: 0 },
      color: { r: Math.fround(3.4028234663852886e38), g: 0, b: 0 },
    }])).toThrow(/remain finite in Float32/);

    const small = 2 ** -80;
    expect(() => packRCLights([{
      kind: 'fixture',
      on: true,
      intensity: small,
      position: { x: 0, y: 0, z: 0 },
      color: { r: small, g: 0, b: 0 },
    }])).toThrow(/underflow completely to zero/);

    expect(() => packRCLights([{
      kind: 'fixture',
      on: true,
      intensity: Number.MIN_VALUE,
      position: { x: 0, y: 0, z: 0 },
    }])).toThrow(/remain positive after Float32 packing/);
  });
});
