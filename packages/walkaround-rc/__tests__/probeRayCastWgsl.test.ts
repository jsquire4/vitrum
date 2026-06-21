import { describe, expect, it } from 'vitest';
import { PROBE_RAY_CAST_WGSL } from '../src/wgsl/probeRayCast.wgsl.js';

describe('PROBE_RAY_CAST_WGSL material UV decode', () => {
  it('matches walkaround f16 UV packing in vec4.w lanes', () => {
    expect(PROBE_RAY_CAST_WGSL).toContain('fn rcPackedUvFromVec4(v: vec4f) -> vec2f');
    expect(PROBE_RAY_CAST_WGSL).toContain('unpack2x16float(bitcast<u32>(v.w))');
    expect(PROBE_RAY_CAST_WGSL).not.toContain('unpack2x16unorm(bitcast<u32>(v.w))');
  });
});
