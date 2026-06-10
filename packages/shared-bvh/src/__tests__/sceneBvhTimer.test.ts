/**
 * H34-g unit tests — SceneBvh slow-rebuild timer covers real work
 *
 * Verifies that onSlowRebuild fires when performance.now() reports the total
 * elapsed time (merge + build) as exceeding the 50 ms threshold. Uses a mock
 * on performance.now() to make the timer predictable.
 */

import { describe, expect, it, vi } from 'vitest';
import type { Scene } from '@vitrum/core';
import { SceneBvh } from '../sceneBvh.js';

function minimalScene(): Scene {
  return {
    primitives: [
      {
        kind: 'mesh',
        id: 'tri',
        positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
        material: { baseColor: [1, 1, 1], roughness: 0.5, metallic: 0 },
      },
    ],
    emitters: [],
    environment: { kind: 'none' },
  };
}

describe('H34-g: SceneBvh slow-rebuild timer', () => {
  it('onSlowRebuild fires when the timer (started before merge) exceeds 50 ms', () => {
    // Mock performance.now: first call → 0 (t0 start), subsequent calls → 100 (t1 end).
    let callCount = 0;
    const nowSpy = vi.spyOn(performance, 'now').mockImplementation(() => {
      callCount += 1;
      return callCount === 1 ? 0 : 100;  // 100 ms elapsed
    });

    const slowCb = vi.fn();
    const bvh = new SceneBvh({ onSlowRebuild: slowCb });
    bvh.updateFromCore(minimalScene());

    expect(slowCb).toHaveBeenCalledOnce();
    expect(slowCb.mock.calls[0]![0]).toBeGreaterThan(50);

    nowSpy.mockRestore();
  });

  it('onSlowRebuild does NOT fire when elapsed < 50 ms', () => {
    let callCount = 0;
    const nowSpy = vi.spyOn(performance, 'now').mockImplementation(() => {
      callCount += 1;
      return callCount === 1 ? 0 : 10;  // 10 ms elapsed — under threshold
    });

    const slowCb = vi.fn();
    const bvh = new SceneBvh({ onSlowRebuild: slowCb });
    bvh.updateFromCore(minimalScene());

    expect(slowCb).not.toHaveBeenCalled();

    nowSpy.mockRestore();
  });
});
