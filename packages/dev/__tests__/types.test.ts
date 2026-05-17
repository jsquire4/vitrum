/**
 * types.test.ts — Smoke tests for @vitrum/dev shared types.
 *
 * Validates that:
 *  1. The types module imports cleanly.
 *  2. DebuggableEngine is a strict supertype of Engine — a real engine that
 *     does not implement onFrame/onProgress/debug still passes a typeof check.
 *  3. The optional chaining guards in the overlay components work correctly
 *     on a plain Engine-shaped object.
 */

import { describe, it, expect } from 'vitest';
import type { DebuggableEngine, FrameStats, ProgressStats } from '../src/types.js';
import type { Engine, EngineCapabilities, EngineState } from '@vitrum/core';

// ── Minimal Engine stub ───────────────────────────────────────────────────────
// Implements the Engine contract but NOT the DebuggableEngine extensions.
// This simulates an engine before T3.E / T3.G land.

function makeMinimalEngine(): Engine {
  const caps: EngineCapabilities = {
    supportsIncrementalScene: false,
    supportsMotionBlur: false,
    supportsAuxBuffers: false,
    accumulates: false,
    maxSamplesPerPixel: Infinity,
    maxBounces: 8,
    supportedAnalyticShapes: new Set(['sphere']),
    supportedEmitterKinds: new Set(['directional']),
    causticStrategy: 'none',
  };
  let state: EngineState = 'ready';
  return {
    get state() { return state; },
    capabilities: caps,
    setScene() { /* no-op */ },
    renderFrame() {
      return { kind: 'skipped', reason: 'no-scene' } as const;
    },
    reset() { /* no-op */ },
    pause() { state = 'paused'; },
    resume() { state = 'ready'; },
    dispose() { state = 'disposed'; },
  };
}

// ────────────────────────────────────────────────────────────────────────────

describe('@vitrum/dev types', () => {
  it('module imports cleanly (no throw on import)', async () => {
    // If this test runs, the module loaded without error.
    const mod = await import('../src/types.js');
    expect(mod).toBeDefined();
  });

  it('a plain Engine can be cast to DebuggableEngine without runtime error', () => {
    const engine = makeMinimalEngine() as DebuggableEngine;
    // All T3.E/T3.G fields are optional — accessing them must not throw.
    expect(engine.onFrame).toBeUndefined();
    expect(engine.onProgress).toBeUndefined();
    expect(engine.debug).toBeUndefined();
  });

  it('typeof-guard for onFrame works correctly on a minimal engine', () => {
    const engine = makeMinimalEngine() as DebuggableEngine;
    const hasOnFrame = typeof engine.onFrame === 'function';
    expect(hasOnFrame).toBe(false);
  });

  it('typeof-guard for debug.setDenoiserEnabled works on a minimal engine', () => {
    const engine = makeMinimalEngine() as DebuggableEngine;
    const hasSetDenoiser = typeof engine.debug?.setDenoiserEnabled === 'function';
    expect(hasSetDenoiser).toBe(false);
  });

  it('a DebuggableEngine with onFrame implemented can subscribe + unsubscribe', () => {
    const engine = makeMinimalEngine() as DebuggableEngine;
    const calls: FrameStats[] = [];
    // Manually attach onFrame (simulates T3.E landing)
    (engine as unknown as Record<string, unknown>)['onFrame'] = (cb: (s: FrameStats) => void) => {
      cb({ frameTimeMs: 16.7 });
      return () => { /* no-op unsubscribe */ };
    };
    if (typeof engine.onFrame === 'function') {
      const unsub = engine.onFrame((stats) => calls.push(stats));
      unsub(); // must not throw
    }
    expect(calls).toHaveLength(1);
    expect(calls[0]?.frameTimeMs).toBeCloseTo(16.7, 5);
  });

  it('ProgressStats fraction field is in [0, 1] by contract shape', () => {
    const p: ProgressStats = { kind: 'ddgi-warmup', current: 5, target: 10, fraction: 0.5 };
    expect(p.fraction).toBeGreaterThanOrEqual(0);
    expect(p.fraction).toBeLessThanOrEqual(1);
  });
});
