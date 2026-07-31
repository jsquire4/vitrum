import { describe, expect, it } from 'vitest';
import { sceneGeometryStatsNeedRefresh } from '../sceneMutationRouter.js';
import {
  sppmSceneBoundsFromCenterRadius,
  sppmSceneBoundsFromPackedPositions,
} from '../sppmParams.js';

describe('SPPM mutation-safe scene bounds', () => {
  it('converts the packed-scene root sphere without reading stale local BLAS vertices', () => {
    expect(
      sppmSceneBoundsFromCenterRadius([10, -2, 4], 5),
    ).toEqual({
      initialRadius: 0.1,
      extent: 5,
      center: [10, -2, 4],
    });
    expect(
      sppmSceneBoundsFromCenterRadius([Number.NaN, 0, 0], 5),
    ).toBeNull();
    expect(
      sppmSceneBoundsFromCenterRadius([0, 0, 0], -1),
    ).toBeNull();
  });

  it('matches the old packed-position result for the equivalent AABB', () => {
    const packed = new Float32Array([
      -2, -1, -3, 0,
      4, 5, 3, 0,
    ]);
    const scanned = sppmSceneBoundsFromPackedPositions(packed)!;
    expect(
      sppmSceneBoundsFromCenterRadius(scanned.center, scanned.extent),
    ).toEqual(scanned);
  });

  it('preserves non-degenerate scene scales without a metre-scale floor', () => {
    for (const scale of [1e-30, 1, 1e30]) {
      const bounds = sppmSceneBoundsFromCenterRadius([0, 0, 0], scale)!;
      expect(bounds.extent).toBe(scale);
      expect(bounds.initialRadius / bounds.extent).toBeCloseTo(0.02, 14);
    }
  });

  it('refreshes derived stats for transform-only TLAS and analytic mutations', () => {
    expect(sceneGeometryStatsNeedRefresh(
      { tlasNodes: new Uint32Array(8) },
      false,
    )).toBe(true);
    expect(sceneGeometryStatsNeedRefresh(
      { analyticLocalToWorld: new Float32Array(16) },
      false,
    )).toBe(true);
    expect(sceneGeometryStatsNeedRefresh(
      { materials: new Float32Array(4) },
      false,
    )).toBe(false);
    expect(sceneGeometryStatsNeedRefresh({}, true)).toBe(true);
  });
});
