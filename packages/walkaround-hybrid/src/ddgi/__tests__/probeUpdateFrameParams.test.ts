import { describe, expect, it } from 'vitest';
import {
  DDGI_PROBE_BLEND_HYSTERESIS,
  haltonSO3AxisAngleFromFrameIndex,
  packProbeUpdateBlendParams,
} from '../probeUpdateFrameParams.js';
describe('probeUpdateFrameParams', () => {
  it('halton rotation is deterministic per frame index', () => {
    const a = haltonSO3AxisAngleFromFrameIndex(42);
    const b = haltonSO3AxisAngleFromFrameIndex(42);
    expect(a).toEqual(b);
    expect(a[0]).not.toBe(a[1]);
  });

  it('blend params pack hysteresis constant', () => {
    const buf = packProbeUpdateBlendParams(8);
    const f32 = new Float32Array(buf);
    const u32 = new Uint32Array(buf);
    expect(u32[0]).toBe(2);
    expect(f32[1]).toBeCloseTo(DDGI_PROBE_BLEND_HYSTERESIS, 6);
  });
});
