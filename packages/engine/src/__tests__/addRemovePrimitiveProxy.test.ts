// addRemovePrimitiveProxy.test.ts — regression guard for the wiring-audit bug:
// the createEngine idempotent-dispose proxy must forward addPrimitive /
// removePrimitive (gated on capabilities.supportsAddRemovePrimitive), which it
// previously dropped entirely — making whole-primitive add/remove unreachable
// through the facade even though every backend implements it.

import { describe, it, expect, vi } from 'vitest';
import type {
  Engine,
  ScenePrimitive,
} from '@vitrum/core';
import { wrapWithIdempotentDispose } from '../createEngine.js';
import { stubCapabilities, stubEngine } from './fixtures/stubEngine.js';

function makeEngine(opts: { supportsAddRemove: boolean; hasMethods: boolean }) {
  const add = vi.fn((_p: ScenePrimitive) => {});
  const remove = vi.fn((_id: ScenePrimitive['id']) => {});
  const caps = stubCapabilities({ supportsAddRemovePrimitive: opts.supportsAddRemove });
  const engine: Engine = {
    ...stubEngine(caps),
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
