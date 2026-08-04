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

function emptyScene(): Scene {
  return { primitives: [], emitters: [], environment: { kind: 'none' } };
}

/**
 * Deterministic clock: every `performance.now()` reads the current value and
 * then advances it by `advanceMsPerCall`. A single `updateFromCore` that runs
 * the merge reads the clock exactly twice (start + report), so each such call
 * measures exactly `advanceMsPerCall` — no matter which return path it takes.
 */
function mockAdvancingClock(advanceMsPerCall: number): { restore: () => void } {
  let nowMs = 0;
  const spy = vi.spyOn(performance, 'now').mockImplementation(() => {
    const value = nowMs;
    nowMs += advanceMsPerCall;
    return value;
  });
  return { restore: () => spy.mockRestore() };
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

  it('reports a slow merge that ends in an empty scene', () => {
    const clock = mockAdvancingClock(100);
    const slowCb = vi.fn();

    const bvh = new SceneBvh({ onSlowRebuild: slowCb });
    bvh.updateFromCore(emptyScene());

    // The merge ran (and cost 100 ms) even though it produced no triangles.
    expect(bvh.buffers).toBeNull();
    expect(slowCb).toHaveBeenCalledOnce();
    expect(slowCb.mock.calls[0]![0]).toBeGreaterThan(50);

    clock.restore();
  });

  it('reports a slow merge that ends in a content-fingerprint match', () => {
    const clock = mockAdvancingClock(100);
    const slowCb = vi.fn();
    const scene = minimalScene();

    const bvh = new SceneBvh({ onSlowRebuild: slowCb });
    bvh.updateFromCore(scene);
    const first = bvh.buffers;
    expect(first).not.toBeNull();
    expect(slowCb).toHaveBeenCalledOnce();

    // Second call re-runs the whole merge and then discovers the content is
    // unchanged. Buffers are retained by identity — that is the fingerprint
    // early return — but the merge was just as expensive as the first call.
    bvh.updateFromCore(scene);
    expect(bvh.buffers).toBe(first);
    expect(slowCb).toHaveBeenCalledTimes(2);
    expect(slowCb.mock.calls[1]![0]).toBeGreaterThan(50);

    clock.restore();
  });

  it('stays silent on the sceneVersionTag fast path, which does no work', () => {
    const clock = mockAdvancingClock(100);
    const slowCb = vi.fn();

    const bvh = new SceneBvh({ onSlowRebuild: slowCb });
    bvh.updateFromCore(minimalScene(), { sceneVersionTag: 7 });
    expect(slowCb).toHaveBeenCalledOnce();

    // Same tag → the merge is skipped entirely, so there is no elapsed time to
    // report and the clock is never read.
    bvh.updateFromCore(minimalScene(), { sceneVersionTag: 7 });
    expect(slowCb).toHaveBeenCalledOnce();

    clock.restore();
  });
});
