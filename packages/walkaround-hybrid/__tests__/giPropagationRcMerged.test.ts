/**
 * PR-5.3 wiring — `propagateBvhToGiSubsystems` must drive the RC merged-mode
 * in-place refit (`refitMergedInstance`) when a primitive moves on a merged-mode
 * BVH, instead of leaving the RC merged geometry stale.
 *
 * This pins the *wiring* (the GI-propagation cascade calls the right RC method
 * with the right data); the refit math itself is pinned by `rcMergedRefit.test.ts`.
 *
 * The moved instance's re-derived WORLD positions arrive as the stride-4
 * `bvhBuffers.bvhPositions.cpuData` — the same buffer the post-update refit path
 * (transformRefit) has already updated in place. The test asserts that exact
 * buffer reaches `refitMergedInstance`.
 */

import { describe, expect, it, vi } from 'vitest';
import { propagateBvhToGiSubsystems, type GiPropagationDeps } from '../src/HybridEngineGiPropagation.js';
import type { RCSubsystem } from '../src/HybridEngineRC.js';
import type { DDGI } from '../src/ddgi/DDGI.js';
import type { SceneBVHBuffers } from '../src/restir/bvhCompute.js';

/** Mock RC subsystem recording every refit/sync/setScene entry point. */
function makeMockRc(refitMergedReturns = true): {
  rc: RCSubsystem;
  refitMergedInstance: ReturnType<typeof vi.fn>;
  refitCascadeBounds: ReturnType<typeof vi.fn>;
  syncRestirBvhBuffers: ReturnType<typeof vi.fn>;
  setScene: ReturnType<typeof vi.fn>;
} {
  const refitMergedInstance = vi.fn(() => refitMergedReturns) as ReturnType<typeof vi.fn>;
  const refitCascadeBounds = vi.fn();
  const syncRestirBvhBuffers = vi.fn();
  const setScene = vi.fn();
  const rc = {
    refitMergedInstance,
    refitCascadeBounds,
    syncRestirBvhBuffers,
    setScene,
  } as unknown as RCSubsystem;
  return { rc, refitMergedInstance, refitCascadeBounds, syncRestirBvhBuffers, setScene };
}

/** Mock DDGI (only `syncRestirBvhBuffers` is reachable from the cascade). */
function makeMockDdgi(): DDGI {
  return { syncRestirBvhBuffers: vi.fn() } as unknown as DDGI;
}

/** Minimal merged-mode `SceneBVHBuffers` stub: only `bvhMode` + the stride-4
 *  `bvhPositions.cpuData` the merged refit reads are load-bearing here. */
function mergedBuffers(stride4Positions: Float32Array): SceneBVHBuffers {
  return {
    bvhMode: 'merged',
    bvhPositions: {
      cpuData: stride4Positions.buffer,
      byteLength: stride4Positions.byteLength,
      count: stride4Positions.length / 4,
    },
  } as unknown as SceneBVHBuffers;
}

const RC_REFIT_BOUNDS = {
  min: [4.5, -0.5, -0.5] as const,
  max: [5.5, 0.5, 0.5] as const,
};

describe('propagateBvhToGiSubsystems — RC merged moving-instance wiring (PR-5.3)', () => {
  it('calls refitMergedInstance with the moved instance world positions on a merged-mode move', () => {
    // Two vertices, stride-4 (xyz + UV-packed .w), already shifted +5 X by the
    // upstream transformRefit (simulated here).
    const positions = new Float32Array([5, 0, 0, 0, 6, 1, 1, 0]);
    const { rc, refitMergedInstance, refitCascadeBounds, setScene } = makeMockRc(true);

    const deps: GiPropagationDeps = {
      ddgi: makeMockDdgi(),
      rc,
      bvhBuffers: mergedBuffers(positions),
      lastScene: null,
      syncDdgi: false,
      allowRcSceneRebuild: true,
      ensureThreeSceneRoot: () => null,
      rcRefitBounds: RC_REFIT_BOUNDS,
    };

    propagateBvhToGiSubsystems(deps);

    expect(refitMergedInstance).toHaveBeenCalledTimes(1);
    const [posArg, minArg, maxArg] = refitMergedInstance.mock.calls[0]!;
    // Same world positions the post-update refit wrote into bvhPositions.cpuData.
    expect(Array.from(posArg as Float32Array)).toEqual(Array.from(positions));
    expect(minArg).toEqual(RC_REFIT_BOUNDS.min);
    expect(maxArg).toEqual(RC_REFIT_BOUNDS.max);

    // Fast path succeeded → no rebuild, no plain cascade-bounds-only fallback.
    expect(setScene).not.toHaveBeenCalled();
    expect(refitCascadeBounds).not.toHaveBeenCalled();
  });

  it('falls back to a full setScene rebuild when the merged fast path declines', () => {
    const positions = new Float32Array([0, 0, 0, 0]);
    const { rc, refitMergedInstance, refitCascadeBounds, setScene } = makeMockRc(false);
    const root = {} as unknown as ReturnType<GiPropagationDeps['ensureThreeSceneRoot']>;

    const deps: GiPropagationDeps = {
      ddgi: makeMockDdgi(),
      rc,
      bvhBuffers: mergedBuffers(positions),
      lastScene: null,
      syncDdgi: false,
      allowRcSceneRebuild: true,
      ensureThreeSceneRoot: () => root,
      rcRefitBounds: RC_REFIT_BOUNDS,
    };

    propagateBvhToGiSubsystems(deps);

    expect(refitMergedInstance).toHaveBeenCalledTimes(1);
    expect(setScene).toHaveBeenCalledTimes(1);
    expect(setScene).toHaveBeenCalledWith(root);
    // setScene refits the cascade bounds itself — no separate refitCascadeBounds.
    expect(refitCascadeBounds).not.toHaveBeenCalled();
  });

  it('keeps the cascade probe grid in sync when the fast path declines and no rebuild is allowed', () => {
    const positions = new Float32Array([0, 0, 0, 0]);
    const { rc, refitMergedInstance, refitCascadeBounds, setScene } = makeMockRc(false);

    const deps: GiPropagationDeps = {
      ddgi: makeMockDdgi(),
      rc,
      bvhBuffers: mergedBuffers(positions),
      lastScene: null,
      syncDdgi: false,
      allowRcSceneRebuild: false, // per-frame-style: no merged rebuild
      ensureThreeSceneRoot: () => null,
      rcRefitBounds: RC_REFIT_BOUNDS,
    };

    propagateBvhToGiSubsystems(deps);

    expect(refitMergedInstance).toHaveBeenCalledTimes(1);
    expect(setScene).not.toHaveBeenCalled();
    // Floor: cascade bounds still track the new AABB.
    expect(refitCascadeBounds).toHaveBeenCalledTimes(1);
    expect(refitCascadeBounds).toHaveBeenCalledWith(RC_REFIT_BOUNDS.min, RC_REFIT_BOUNDS.max);
  });

  it('does NOT call refitMergedInstance in TLAS mode (uses cascade refit + buffer sync instead)', () => {
    const { rc, refitMergedInstance, refitCascadeBounds, syncRestirBvhBuffers } = makeMockRc(true);
    const tlasBuffers = { bvhMode: 'tlas' } as unknown as SceneBVHBuffers;

    const deps: GiPropagationDeps = {
      ddgi: makeMockDdgi(),
      rc,
      bvhBuffers: tlasBuffers,
      lastScene: null,
      syncDdgi: false,
      allowRcSceneRebuild: true,
      ensureThreeSceneRoot: () => null,
      rcRefitBounds: RC_REFIT_BOUNDS,
    };

    propagateBvhToGiSubsystems(deps);

    expect(refitMergedInstance).not.toHaveBeenCalled();
    expect(refitCascadeBounds).toHaveBeenCalledWith(RC_REFIT_BOUNDS.min, RC_REFIT_BOUNDS.max);
    expect(syncRestirBvhBuffers).toHaveBeenCalledWith(tlasBuffers);
  });
});
