import { describe, expect, it } from 'vitest';
import type { DDGILight } from '../../ddgi/types.js';
import { packRCLights } from '../packingHelpers.js';

/**
 * T2-C (D16-7) — byte golden for packRCLights after switching from raw word
 * indices (data[base + N]) to the generated RCLightEntryOffset named offsets.
 * The u32 array below was captured from the pre-refactor packer and pins the
 * exact 1040-byte RCLightBuffer wire format (point + spot entry + zero tail).
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

const GOLDEN_U32 = [2,0,0,0,1,1094713344,1073741824,0,1065353216,1073741824,1077936128,1084227584,0,0,0,1065353216,1053609165,1056964608,1058642330,0,2147483650,0,1073741824,0,3229614080,1084227584,3233808384,1091567616,0,3212836864,0,1063675494,1036831949,1045220557,1050253722,1060320051,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0];

describe('packRCLights byte golden', () => {
  it('packs the representative point+spot set byte-identically to the pre-codegen packer', () => {
    const buf = packRCLights(GOLDEN_LIGHTS);
    expect(Array.from(new Uint32Array(buf))).toEqual(GOLDEN_U32);
    expect(buf.byteLength).toBe(1040);
  });
});
