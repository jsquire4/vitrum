import { describe, expect, it } from 'vitest';
import {
  assertPtWebgpuDistantLaunchDiskRepresentable,
  F32_MIN_NORMAL,
  ptWebgpuRayOriginBias,
  ptWebgpuRayTMin,
  resolvePtWebgpuSceneRadius,
} from '../scene/sceneScalePolicy.js';

describe('pt-webgpu scene-scale ray policy', () => {
  it('preserves every finite positive scene radius without a world-unit floor', () => {
    for (const radius of [1e-30, 1, 1e30]) {
      expect(resolvePtWebgpuSceneRadius([0, 0, 0], radius)).toBe(radius);
    }
  });

  it('scales ray origin and traversal thresholds homogeneously', () => {
    const ordinaryBias = ptWebgpuRayOriginBias(1);
    const ordinaryTMin = ptWebgpuRayTMin(1);
    for (const scale of [1e-30, 1e30]) {
      expect(ptWebgpuRayOriginBias(scale) / ordinaryBias).toBeCloseTo(
        scale,
        12,
      );
      expect(ptWebgpuRayTMin(scale) / ordinaryTMin).toBeCloseTo(
        scale,
        12,
      );
    }
  });

  it('gives a truly degenerate scene only a coordinate-relative f32-safe scale', () => {
    expect(resolvePtWebgpuSceneRadius([0, 0, 0], 0)).toBe(F32_MIN_NORMAL);
    expect(resolvePtWebgpuSceneRadius([1024, 0, 0], 0)).toBe(2 ** -10);
  });

  it('raises the origin bias above the coordinate ULP for translated scenes', () => {
    const center: readonly [number, number, number] = [1e8, 0, 0];
    const bias = ptWebgpuRayOriginBias(1, center);
    const packedCenter = Math.fround(center[0]);
    expect(Math.fround(packedCenter + bias)).toBeGreaterThan(packedCenter);
    expect(bias).toBeGreaterThan(ptWebgpuRayOriginBias(1));
  });

  it('rejects invalid bounds and non-positive ray-policy inputs', () => {
    expect(() => resolvePtWebgpuSceneRadius([0, 0, 0], -1)).toThrow(
      RangeError,
    );
    expect(() => resolvePtWebgpuSceneRadius([0, Number.NaN, 0], 1)).toThrow(
      RangeError,
    );
    expect(() => ptWebgpuRayOriginBias(0)).toThrow(RangeError);
    expect(() => ptWebgpuRayTMin(Number.POSITIVE_INFINITY)).toThrow(
      RangeError,
    );
    expect(() => resolvePtWebgpuSceneRadius([0, 0, 0], 1e-50)).toThrow(
      RangeError,
    );
    expect(() => resolvePtWebgpuSceneRadius([0, 0, 0], 1e39)).toThrow(
      RangeError,
    );
    expect(() => resolvePtWebgpuSceneRadius([1e300, 0, 0], 1)).toThrow(
      RangeError,
    );
  });

  it('accepts only distant-launch radii whose area density is representable in f32', () => {
    expect(() => assertPtWebgpuDistantLaunchDiskRepresentable(1)).not.toThrow();
    expect(() => assertPtWebgpuDistantLaunchDiskRepresentable(1e19)).not.toThrow();
    expect(() => assertPtWebgpuDistantLaunchDiskRepresentable(1e-30))
      .toThrow(/launch-disk area/);
    expect(() => assertPtWebgpuDistantLaunchDiskRepresentable(1.1e19))
      .toThrow(/launch-disk area/);
  });
});
