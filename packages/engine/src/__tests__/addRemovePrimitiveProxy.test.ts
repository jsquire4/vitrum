// addRemovePrimitiveProxy.test.ts — regression guard for the wiring-audit bug:
// the createEngine idempotent-dispose proxy must forward addPrimitive /
// removePrimitive (gated on capabilities.supportsAddRemovePrimitive), which it
// previously dropped entirely — making whole-primitive add/remove unreachable
// through the facade even though every backend implements it.

import { describe, it, expect, vi } from 'vitest';
import type {
  Engine,
  EngineCapabilities,
  EngineState,
  ScenePrimitive,
} from '@vitrum/core';
import { wrapWithIdempotentDispose } from '../createEngine.js';

function capabilities(supportsAddRemove: boolean): EngineCapabilities {
  return {
    supportsIncrementalScene: false,
    supportsAddRemovePrimitive: supportsAddRemove,
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
  } as unknown as EngineCapabilities;
}

function makeEngine(opts: { supportsAddRemove: boolean; hasMethods: boolean }) {
  const add = vi.fn((_p: ScenePrimitive) => {});
  const remove = vi.fn((_id: ScenePrimitive['id']) => {});
  const caps = capabilities(opts.supportsAddRemove);
  const engine: Engine = {
    get state(): EngineState { return 'ready'; },
    get capabilities() { return caps; },
    setScene: vi.fn(),
    renderFrame: vi.fn(() => ({ kind: 'skipped', samplesAccumulated: 0, isConverged: false })),
    reset: vi.fn(),
    pause: vi.fn(),
    resume: vi.fn(),
    dispose: vi.fn(),
    ...(opts.hasMethods
      ? {
          addPrimitive: (p: ScenePrimitive) => add(p),
          removePrimitive: (id: ScenePrimitive['id']) => remove(id),
        }
      : {}),
  };
  return { engine, add, remove };
}

const PRIM = { id: 'p1', kind: 'mesh' } as unknown as ScenePrimitive;

describe('createEngine proxy — addPrimitive / removePrimitive forwarding', () => {
  it('forwards both when the backend implements them and capability is advertised', () => {
    const { engine, add, remove } = makeEngine({ supportsAddRemove: true, hasMethods: true });
    const proxy = wrapWithIdempotentDispose(engine, () => {});
    expect(typeof proxy.addPrimitive).toBe('function');
    expect(typeof proxy.removePrimitive).toBe('function');
    proxy.addPrimitive!(PRIM);
    proxy.removePrimitive!('p1');
    expect(add).toHaveBeenCalledTimes(1);
    expect(remove).toHaveBeenCalledWith('p1');
  });

  it('omits both when the capability is not advertised (even if methods exist)', () => {
    const { engine } = makeEngine({ supportsAddRemove: false, hasMethods: true });
    const proxy = wrapWithIdempotentDispose(engine, () => {});
    expect(proxy.addPrimitive).toBeUndefined();
    expect(proxy.removePrimitive).toBeUndefined();
  });

  it('omits both when the backend does not implement them', () => {
    const { engine } = makeEngine({ supportsAddRemove: true, hasMethods: false });
    const proxy = wrapWithIdempotentDispose(engine, () => {});
    expect(proxy.addPrimitive).toBeUndefined();
    expect(proxy.removePrimitive).toBeUndefined();
  });

  it('no-ops forwarded calls after dispose (does not reach the backend)', () => {
    const { engine, add } = makeEngine({ supportsAddRemove: true, hasMethods: true });
    const proxy = wrapWithIdempotentDispose(engine, () => {});
    proxy.dispose();
    proxy.addPrimitive!(PRIM);
    expect(add).not.toHaveBeenCalled();
  });
});
