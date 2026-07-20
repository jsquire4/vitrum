// inverseSessionProxy.test.ts — WS5: the createEngine idempotent-dispose proxy
// conditionally forwards createInverseSession and refuses to open a session
// after dispose (mirrors the optional-method forwarding of updateEnvironment etc).

import { describe, it, expect, vi } from 'vitest';
import type {
  Engine,
  InverseSession,
  InverseSessionOptions,
} from '@vitrum/core';
import { wrapWithIdempotentDispose } from '../createEngine.js';
import { stubEngine } from './fixtures/stubEngine.js';

function fakeSession(): InverseSession {
  return {
    parameterCount: 1,
    method: 'finite-difference',
    step: async () => ({ step: 0, loss: 0, values: [[0]], gradient: [[0]] }),
    currentValues: () => [[0]],
    dispose: vi.fn(),
  };
}

function makeEngine(withInverse: boolean) {
  const createInverse = vi.fn((_opts: InverseSessionOptions): InverseSession => fakeSession());
  const engine: Engine = {
    ...stubEngine(),
    ...(withInverse
      ? { createInverseSession: (opts: InverseSessionOptions) => createInverse(opts) }
      : {}),
  };
  return { engine, createInverse };
}

const OPTS: InverseSessionOptions = {
  target: { data: new Float32Array(3), width: 1, height: 1, channels: 3 },
  parameters: [{ path: 'materials.x.baseColor', kind: 'rgb' }],
};

describe('createEngine proxy — createInverseSession forwarding', () => {
  it('forwards when the backend implements it', () => {
    const { engine, createInverse } = makeEngine(true);
    const proxy = wrapWithIdempotentDispose(engine, () => {});
    expect(typeof proxy.createInverseSession).toBe('function');
    const session = proxy.createInverseSession!(OPTS);
    expect(createInverse).toHaveBeenCalledTimes(1);
    expect(session.parameterCount).toBe(1);
  });

  it('omits the method when the backend does not implement it', () => {
    const { engine } = makeEngine(false);
    const proxy = wrapWithIdempotentDispose(engine, () => {});
    expect(proxy.createInverseSession).toBeUndefined();
  });

  it('throws when opening a session after dispose', () => {
    const { engine } = makeEngine(true);
    const proxy = wrapWithIdempotentDispose(engine, () => {});
    proxy.dispose();
    expect(() => proxy.createInverseSession!(OPTS)).toThrow(/disposed/);
  });
});
