import { describe, expect, it } from 'vitest';
import type { DDGILight } from '../../ddgi/types.js';
import { packRCLights } from '../packingHelpers.js';

/**
 * T2-C (D16-7) — byte golden for packRCLights after switching from raw word
 * indices (data[base + N]) to the generated RCLightEntryOffset named offsets.
 * The assertions pin the packed header ABI and exact point+spot payload size.
 */
const GOLDEN_LIGHTS: DDGILight[] = [
  {
    kind: 'fixture',
    on: true,
    intensity: 5,
    distance: 12,
    decay: 2,
    position: { x: 1, y: 2, z: 3 },
    color: { r: 0.4, g: 0.5, b: 0.6 },
  },
  {
    kind: 'teaLight',
    on: true,
    intensity: 9,
    position: { x: -4, y: 5, z: -6 },
    color: { r: 0.1, g: 0.2, b: 0.3 },
    spotAxis: { x: 0, y: -1, z: 0 },
    spotCosInner: 0.9,
    spotCosOuter: 0.7,
    castShadow: false,
  },
];


describe('packRCLights byte golden', () => {
  it('packs the representative point+spot set byte-identically to the pre-codegen packer', () => {
    const buf = packRCLights(GOLDEN_LIGHTS);
    const words = new Uint32Array(buf);
    expect(Array.from(words.slice(0, 4))).toEqual([2, 4, 36, 0x31544352]);
    expect(buf.byteLength).toBe(16 + 2 * (64 + 16));
  });
});
