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

  // H16 — `DDGI.invalidateProbeCache()` → `ProbeUpdatePass.requestFullBlend()`
  // sets a one-shot flag that makes the next blend upload pass
  // `hysteresisOverride = 0.0`. With the blend kernel's
  // `mix(coeff, prev, hysteresis)` (probeUpdateBlend.wgsl:156), hysteresis=0 is a
  // FULL REPLACE (`newValue = freshSample`) — the atlas clears in one stride
  // window instead of decaying at 0.97 over hundreds of frames. This pins the
  // UBO surface of that fix so the override can't silently regress to the
  // steady-state constant.
  it('H16: full-replace override packs hysteresis=0 (vs steady-state 0.97)', () => {
    const steady = new Float32Array(packProbeUpdateBlendParams(8));
    const fullReplace = new Float32Array(packProbeUpdateBlendParams(8, undefined, 0.0));
    expect(steady[1]).toBeCloseTo(0.97, 6);          // default = steady-state EMA
    expect(fullReplace[1]).toBe(0);                  // invalidate path = clear
    // an arbitrary override is honoured verbatim (not clamped to the constant).
    expect(new Float32Array(packProbeUpdateBlendParams(8, undefined, 0.5))[1]).toBeCloseTo(0.5, 6);
    // probesPerFrame still derives from the divisor (override touches only hysteresis).
    expect(new Uint32Array(packProbeUpdateBlendParams(8, undefined, 0.0))[0]).toBe(2);
  });
});
