import { describe, expect, it } from 'vitest';
import { RCParamsOffset } from '../src/rc/rcParamsLayout.generated.js';
import { packRCParams } from '../src/HybridEngineRC.js';

describe('RCParamsOffset codegen alignment', () => {
  it('matches packRCParams field byte offsets', () => {
    expect(RCParamsOffset.probeOriginWorld).toBe(0);
    expect(RCParamsOffset.rcWeight).toBe(12);
    expect(RCParamsOffset.roomSize).toBe(16);
    expect(RCParamsOffset.enabled).toBe(28);
    expect(RCParamsOffset.probeCount).toBe(32);
    expect(RCParamsOffset.raysPerProbe).toBe(44);
    expect(RCParamsOffset.rayGridSize).toBe(48);
    expect(RCParamsOffset.rcWeight / 4).toBe(3);
  });

  it('packRCParams writes at generated offsets', () => {
    const buf = packRCParams([1, 2, 3], [4, 5, 6], [7, 8, 9], 64, 0.25, true);
    const view = new DataView(buf);
    expect(view.getFloat32(RCParamsOffset.probeOriginWorld, true)).toBe(1);
    expect(view.getFloat32(RCParamsOffset.rcWeight, true)).toBe(0.25);
    expect(view.getFloat32(RCParamsOffset.roomSize, true)).toBe(4);
    expect(view.getUint32(RCParamsOffset.enabled, true)).toBe(1);
    expect(view.getUint32(RCParamsOffset.probeCount, true)).toBe(7);
    expect(view.getUint32(RCParamsOffset.raysPerProbe, true)).toBe(64);
    expect(view.getUint32(RCParamsOffset.rayGridSize, true)).toBe(8);
  });
});
