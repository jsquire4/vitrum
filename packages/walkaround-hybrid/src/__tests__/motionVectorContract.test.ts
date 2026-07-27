import { describe, expect, it } from 'vitest';

import { SVGF_REPROJECTION_WGSL } from '@vitrum/shared-denoisers';
import { CB_PREFILL_MODULE } from '../shaders/cbPrefill.wgsl.js';
import { MOTION_VECTORS_WGSL } from '../shaders/motionVectors.wgsl.js';
import { RESOLVE_WGSL } from '../shaders/resolve.wgsl.js';
import { TEMPORAL_WGSL } from '../shaders/temporal.wgsl.js';
import { TEMPORAL_GI_COMMON_WGSL } from '../shaders/temporalGiCommon.wgsl.js';

type V2 = readonly [number, number];

function ndcToFramebufferPixel(ndc: V2, width: number, height: number): V2 {
  return [
    (ndc[0] * 0.5 + 0.5) * width,
    (0.5 - ndc[1] * 0.5) * height,
  ];
}

function previousMinusCurrentPixelMotion(
  currentNdc: V2,
  previousNdc: V2,
  width: number,
  height: number,
): V2 {
  return [
    (previousNdc[0] - currentNdc[0]) * width * 0.5,
    -(previousNdc[1] - currentNdc[1]) * height * 0.5,
  ];
}

describe('walkaround motion-vector contract', () => {
  it('round-trips ±x/±y reprojection at anisotropic resolution', () => {
    const width = 1920;
    const height = 720;
    const cases: readonly [V2, V2][] = [
      [[-0.7, 0.4], [-0.2, 0.4]],
      [[0.6, -0.1], [0.1, -0.1]],
      [[0.2, -0.8], [0.2, -0.3]],
      [[-0.4, 0.7], [-0.4, 0.1]],
      [[0.55, -0.65], [-0.35, 0.45]],
    ];

    for (const [currentNdc, previousNdc] of cases) {
      const currentPixel = ndcToFramebufferPixel(currentNdc, width, height);
      const previousPixel = ndcToFramebufferPixel(previousNdc, width, height);
      const motion = previousMinusCurrentPixelMotion(
        currentNdc,
        previousNdc,
        width,
        height,
      );
      expect(currentPixel[0] + motion[0]).toBeCloseTo(previousPixel[0], 10);
      expect(currentPixel[1] + motion[1]).toBeCloseTo(previousPixel[1], 10);
    }
  });

  it('converts NDC to pixels once at the producer and never again downstream', () => {
    expect(MOTION_VECTORS_WGSL).toContain(
      'ndcDelta.x * f32(dims.x) * 0.5',
    );
    expect(MOTION_VECTORS_WGSL).toContain(
      '-ndcDelta.y * f32(dims.y) * 0.5',
    );

    const checkerboardConsumers = [CB_PREFILL_MODULE.source, RESOLVE_WGSL];
    for (const source of checkerboardConsumers) {
      expect(source).toContain('vec2i(round(mv))');
      expect(source).toContain('+ deltaPx');
      expect(source).not.toMatch(/mv\.[xy]\s*\*\s*f32\([WH]\)/);
      expect(source).not.toMatch(/-\s*dxPx|-\s*dyPx/);
    }

    expect(SVGF_REPROJECTION_WGSL).toContain('vec2f(gid.xy) + mv');
    expect(SVGF_REPROJECTION_WGSL).not.toMatch(
      /mv\.[xy]\s*\*\s*f32\([^)]*(width|height|dims)/i,
    );
  });

  it('keeps DI/GI temporal reprojection matrix-based, not motion-texture based', () => {
    for (const source of [TEMPORAL_WGSL, TEMPORAL_GI_COMMON_WGSL]) {
      expect(source).toContain('prevViewProjMatrix');
      expect(source).not.toMatch(/motion(Vector|Vec|Texture)/i);
    }
  });
});
