// progressiveSeedProxy.test.ts — the createEngine idempotent-dispose proxy
// conditionally forwards the progressive walkaround→PT seed SOURCE
// (getProgressiveSeedTexture) and SINK (seedAccumulator), gated on the
// respective capabilities, with dispose-safe fallbacks (source → null, sink →
// no-op). This forwarding is what lets `createProgressiveEngine` drive the
// ProgressiveHandoffCoordinator over the WRAPPED engines — without it the
// coordinator's optional-chained seed calls resolve to undefined and the seed
// silently no-ops (the bug the dzn e2e surfaced: byte-identical seeded/unseeded
// arms).

import { describe, it, expect, vi } from 'vitest';
import type {
  Engine,
  EngineCapabilities,
  EngineState,
  BackendTexture,
} from '@vitrum/core';
import { wrapWithIdempotentDispose } from '../createEngine.js';

function capabilities(over: Partial<EngineCapabilities>): EngineCapabilities {
  return {
    supportsIncrementalScene: false,
    supportsAddRemovePrimitive: false,
    supportsAuxBuffers: false,
    accumulates: true,
    maxSamplesPerPixel: 1,
    maxBounces: 1,
    supportedAnalyticShapes: new Set(),
    supportedEmitterKinds: new Set(),
    supportedPrimitiveKinds: new Set(),
    supportedEnvironmentKinds: new Set(),
    presentationMode: 'offscreen-texture',
    experimentalFeatures: new Set(),
    causticStrategy: 'none',
    ...over,
  } as unknown as EngineCapabilities;
}

const FAKE_TEX = { __seed: true } as unknown as BackendTexture;
const SEED_RESULT = { texture: FAKE_TEX, width: 320, height: 180 };

function makeEngine(opts: {
  caps: Partial<EngineCapabilities>;
  withSource?: boolean;
  withSink?: boolean;
}) {
  const getProgressiveSeedTexture = vi.fn(() => SEED_RESULT);
  const seedAccumulator = vi.fn();
  const engine: Engine = {
    get state(): EngineState { return 'ready'; },
    get capabilities() { return capabilities(opts.caps); },
    setScene: vi.fn(),
    renderFrame: vi.fn(() => ({ kind: 'skipped', samplesAccumulated: 0, isConverged: false })),
    reset: vi.fn(),
    pause: vi.fn(),
    resume: vi.fn(),
    dispose: vi.fn(),
    ...(opts.withSource ? { getProgressiveSeedTexture } : {}),
    ...(opts.withSink ? { seedAccumulator } : {}),
  };
  return { engine, getProgressiveSeedTexture, seedAccumulator };
}

describe('createEngine proxy — progressive seed source/sink forwarding', () => {
  it('forwards getProgressiveSeedTexture when supportsProgressiveSeedSource + method present', () => {
    const { engine, getProgressiveSeedTexture } = makeEngine({
      caps: { supportsProgressiveSeedSource: true },
      withSource: true,
    });
    const proxy = wrapWithIdempotentDispose(engine, () => {});
    expect(typeof proxy.getProgressiveSeedTexture).toBe('function');
    expect(proxy.getProgressiveSeedTexture!()).toBe(SEED_RESULT);
    expect(getProgressiveSeedTexture).toHaveBeenCalledTimes(1);
  });

  it('forwards seedAccumulator when supportsAccumulatorSeed + method present', () => {
    const { engine, seedAccumulator } = makeEngine({
      caps: { supportsAccumulatorSeed: true },
      withSink: true,
    });
    const proxy = wrapWithIdempotentDispose(engine, () => {});
    expect(typeof proxy.seedAccumulator).toBe('function');
    proxy.seedAccumulator!(FAKE_TEX, { weight: 4, width: 64, height: 64 });
    expect(seedAccumulator).toHaveBeenCalledWith(FAKE_TEX, { weight: 4, width: 64, height: 64 });
  });

  it('omits the methods when the capability is NOT advertised (even if the method exists)', () => {
    // Capability flags absent → unforwarded, regardless of the method being present.
    const { engine } = makeEngine({ caps: {}, withSource: true, withSink: true });
    const proxy = wrapWithIdempotentDispose(engine, () => {});
    expect(proxy.getProgressiveSeedTexture).toBeUndefined();
    expect(proxy.seedAccumulator).toBeUndefined();
  });

  it('omits the methods when the backend does not implement them (e.g. pt-webgl2)', () => {
    const { engine } = makeEngine({
      caps: { supportsProgressiveSeedSource: true, supportsAccumulatorSeed: true },
      withSource: false,
      withSink: false,
    });
    const proxy = wrapWithIdempotentDispose(engine, () => {});
    expect(proxy.getProgressiveSeedTexture).toBeUndefined();
    expect(proxy.seedAccumulator).toBeUndefined();
  });

  it('is dispose-safe: source → null, sink → no-op, without touching the torn-down backend', () => {
    const { engine, getProgressiveSeedTexture, seedAccumulator } = makeEngine({
      caps: { supportsProgressiveSeedSource: true, supportsAccumulatorSeed: true },
      withSource: true,
      withSink: true,
    });
    const proxy = wrapWithIdempotentDispose(engine, () => {});
    proxy.dispose();
    expect(proxy.getProgressiveSeedTexture!()).toBeNull();
    proxy.seedAccumulator!(FAKE_TEX, { weight: 4, width: 64, height: 64 });
    // Neither disposed fallback forwards to the (torn-down) engine.
    expect(getProgressiveSeedTexture).not.toHaveBeenCalled();
    expect(seedAccumulator).not.toHaveBeenCalled();
  });
});
