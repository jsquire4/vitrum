import { describe, expect, it } from 'vitest';

import {
  NORMAL_DEPTH_NO_HIT_NORMAL,
  NORMAL_DEPTH_DECODE_WGSL,
  PACKED_NORMAL_DEPTH_TEXTURE_LAYOUT,
  STANDALONE_DEPTH_TEXTURE_LAYOUT,
  buildAtrousVarianceWgsl,
  buildSvgfReprojectionWgsl,
  decodeNormalDepthWorldNormal,
  float32ToFloat16Bits,
  rgba16fBufferToRgbF32,
  normalDepthWgslDepthComponent,
} from '../src/index.js';

describe('normalDepth encoding', () => {
  it('uses encoded world-up as the canonical no-hit normal', () => {
    expect(NORMAL_DEPTH_NO_HIT_NORMAL).toEqual([0.5, 1, 0.5]);
    expect(decodeNormalDepthWorldNormal(...NORMAL_DEPTH_NO_HIT_NORMAL)).toEqual([0, 1, 0]);
  });

  it('decodes the full affine normal range', () => {
    expect(decodeNormalDepthWorldNormal(0, 0.5, 1)).toEqual([-1, 0, 1]);
  });

  it('selects depth from the declared physical texture layout', () => {
    expect(normalDepthWgslDepthComponent(STANDALONE_DEPTH_TEXTURE_LAYOUT)).toBe('r');
    expect(normalDepthWgslDepthComponent(PACKED_NORMAL_DEPTH_TEXTURE_LAYOUT)).toBe('w');

    const standaloneAtrous = buildAtrousVarianceWgsl(STANDALONE_DEPTH_TEXTURE_LAYOUT);
    const packedAtrous = buildAtrousVarianceWgsl(PACKED_NORMAL_DEPTH_TEXTURE_LAYOUT);
    const standaloneSvgf = buildSvgfReprojectionWgsl(STANDALONE_DEPTH_TEXTURE_LAYOUT);
    const packedSvgf = buildSvgfReprojectionWgsl(PACKED_NORMAL_DEPTH_TEXTURE_LAYOUT);

    expect(standaloneAtrous).toContain(
      'textureLoad(atrous_gbufDepth, gid.xy, 0).r',
    );
    expect(packedAtrous).toContain(
      'textureLoad(atrous_gbufDepth, gid.xy, 0).w',
    );
    expect(standaloneSvgf).toContain(
      'textureLoad(reproj_currDepth,  gid.xy, 0).r',
    );
    expect(packedSvgf).toContain(
      'textureLoad(reproj_currDepth,  gid.xy, 0).w',
    );
  });

  it('keeps depth and normal discontinuity weights independent', () => {
    const depthWeight = (a: number, b: number): number => Math.exp(-Math.abs(a - b));
    const normalWeight = (
      a: readonly [number, number, number],
      b: readonly [number, number, number],
    ): number =>
      Math.max(0, a[0] * b[0] + a[1] * b[1] + a[2] * b[2]);

    const normal = [0, 1, 0] as const;
    expect(depthWeight(1, 9)).toBeLessThan(0.001);
    expect(normalWeight(normal, normal)).toBe(1);

    const perpendicular = [1, 0, 0] as const;
    expect(depthWeight(4, 4)).toBe(1);
    expect(normalWeight(normal, perpendicular)).toBe(0);
  });

  it('shares the packed-normal affine decoder with WGSL consumers', () => {
    expect(NORMAL_DEPTH_DECODE_WGSL).toContain(
      'return encoded * 2.0 - vec3f(1.0);',
    );
    expect(decodeNormalDepthWorldNormal(0.25, 0.5, 0.75)).toEqual([-0.5, 0, 0.5]);
  });

  it('decodes packed half-float readback through the OIDN conversion path', () => {
    const packed = new Uint16Array([
      float32ToFloat16Bits(0.5),
      float32ToFloat16Bits(1),
      float32ToFloat16Bits(0.5),
      float32ToFloat16Bits(0),
    ]);

    expect(
      Array.from(
        rgba16fBufferToRgbF32(
          packed.buffer,
          8,
          1,
          1,
          decodeNormalDepthWorldNormal,
        ),
      ),
    ).toEqual([0, 1, 0]);
  });
});
