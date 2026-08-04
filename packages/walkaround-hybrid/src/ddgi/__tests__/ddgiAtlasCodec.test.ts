import { describe, expect, it } from 'vitest';

import {
  DDGI_ATLAS_SAFE_BLOCK_MAX,
  DDGI_VISIBILITY_DISTANCE_MAX,
} from '../ddgiNumericLimits.js';
import { DDGI_ATLAS_CODEC_WGSL } from '../wgsl/ddgiAtlasCodec.wgsl.js';

describe('DDGI rgba16float block codec', () => {
  it('emits the visibility-distance ceiling from the exact canonical f32 bits', () => {
    const encoded = new ArrayBuffer(4);
    const view = new DataView(encoded);
    view.setFloat32(0, DDGI_VISIBILITY_DISTANCE_MAX, false);

    expect(view.getUint32(0, false)).toBe(0x5f7f_efff);
    expect(DDGI_ATLAS_CODEC_WGSL).toContain(
      'const DDGI_ATLAS_VISIBILITY_DISTANCE_MAX: f32 = 1.8442239374570553344e19;',
    );
    expect(Math.fround(1.8442239374570553344e19)).toBe(
      DDGI_VISIBILITY_DISTANCE_MAX,
    );
    expect(DDGI_ATLAS_CODEC_WGSL).not.toContain(
      'DDGI_ATLAS_VISIBILITY_DISTANCE_MAX: f32 = 18442239374570553000',
    );
  });

  it('keeps legacy zero/one exponent lanes self-describing and rejects malformed metadata', () => {
    expect(DDGI_ATLAS_CODEC_WGSL).toContain(
      'if (stored == 0 || stored == 1) { return 0; }',
    );
    expect(DDGI_ATLAS_CODEC_WGSL).toContain('lane != round(lane)');
    expect(DDGI_ATLAS_CODEC_WGSL).toContain('lane < -149.0');
    expect(DDGI_ATLAS_CODEC_WGSL).toContain('lane > 115.0');
    expect(DDGI_ATLAS_CODEC_WGSL).toContain(
      '(exponent == 114 && abs(mantissa) >= 16384.0)',
    );
  });

  it('rounds the top producer mantissa inward and keeps the visibility second moment ordered', () => {
    const highestPublished = Math.fround(16_376 * 2 ** 114);
    const firstOverflow = Math.fround(16_384 * 2 ** 114);
    const largestVisibilitySquare = Math.fround(
      DDGI_VISIBILITY_DISTANCE_MAX * DDGI_VISIBILITY_DISTANCE_MAX,
    );

    expect(Number.isFinite(highestPublished)).toBe(true);
    expect(highestPublished).toBeLessThanOrEqual(DDGI_ATLAS_SAFE_BLOCK_MAX);
    expect(firstOverflow).toBe(Number.POSITIVE_INFINITY);
    expect(Number.isFinite(largestVisibilitySquare)).toBe(true);
    expect(largestVisibilitySquare).toBeLessThanOrEqual(
      DDGI_ATLAS_SAFE_BLOCK_MAX,
    );
    expect(DDGI_ATLAS_CODEC_WGSL).toContain(
      'publishMantissa = select(-16376.0, 16376.0, publishMantissa > 0.0);',
    );
    expect(DDGI_ATLAS_CODEC_WGSL).toContain(
      'max(secondMoment, ddgiAtlasSaturatingMul(mean, mean))',
    );
    expect(DDGI_ATLAS_CODEC_WGSL).toContain(
      'let orderedSecondMoment = max(',
    );
  });
});
