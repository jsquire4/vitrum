import { describe, expect, it } from 'vitest';
import type { DDGILight } from '../../ddgi/types.js';
import { packRCLights } from '../packingHelpers.js';

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
});
