/**
 * engineErrorSurface.test.ts — unit tests for the item-28 EngineError surface
 * on pt-webgl2 (PTEngineWebGL2).
 *
 * The pt-webgl2 backend surfaces errors via the WebGL `webglcontextlost` canvas
 * event (there is no GPUDevice).  The mock canvas inside `createMockGl()` already
 * exposes a test-helper `dispatchEvent(type, event)` — we use that to fire a
 * synthetic context-lost event and assert the subscriber receives the right
 * EngineError shape.
 *
 * Coverage:
 *   - onError() is present and returns an unsubscribe function.
 *   - webglcontextlost → subscriber fires with kind='context-lost', fatal=true.
 *   - Unsubscribe prevents further delivery; idempotent.
 *   - dispose() clears subscribers; post-dispose context-lost is silently ignored.
 */

import { describe, expect, it, vi } from 'vitest';
import type { EngineError } from '@vitrum/core';
import { createPTEngine_WebGL2 } from '../index.js';
import { createMockGl } from './mockGl.js';

// ── Helpers ────────────────────────────────────────────────────────────────────

/** Build a mock GL whose canvas exposes `dispatchEvent` so tests can fire
 *  synthetic webglcontextlost events. */
function makeGl() {
  const gl = createMockGl();
  // The canvas object returned by createMockGl has a dispatchEvent test helper.
  const canvas = (gl as unknown as { canvas: {
    dispatchEvent(type: string, event: Event): void;
    addEventListener: unknown;
  } }).canvas;
  return { gl, canvas };
}

/** A minimal fake webglcontextlost event (just needs `preventDefault`). */
function fakeContextLostEvent(): Event {
  return { preventDefault: vi.fn() } as unknown as Event;
}

// ── Tests ─────────────────────────────────────────────────────────────────────

describe('pt-webgl2 EngineError surface — onError subscription', () => {
  it('onError is a function and returns an unsubscribe function', async () => {
    const { gl } = makeGl();
    const engine = await createPTEngine_WebGL2({ device: gl });
    expect(typeof engine.onError).toBe('function');
    const unsub = engine.onError!((err: EngineError) => void err);
    expect(typeof unsub).toBe('function');
    engine.dispose();
  });

  it('unsubscribe removes the callback and is idempotent', async () => {
    const { gl } = makeGl();
    const engine = await createPTEngine_WebGL2({ device: gl });
    const received: EngineError[] = [];
    const unsub = engine.onError!((err) => received.push(err));
    unsub();
    unsub(); // idempotent — must not throw
    engine.dispose();
  });
});

describe('pt-webgl2 EngineError surface — webglcontextlost routing', () => {
  it('context-lost event fires subscriber with kind="context-lost" and fatal=true', async () => {
    const { gl, canvas } = makeGl();
    const engine = await createPTEngine_WebGL2({ device: gl });
    const received: EngineError[] = [];
    engine.onError!((err) => received.push(err));

    canvas.dispatchEvent('webglcontextlost', fakeContextLostEvent());

    expect(received).toHaveLength(1);
    expect(received[0]!.kind).toBe('context-lost');
    expect(received[0]!.fatal).toBe(true);
    expect(typeof received[0]!.message).toBe('string');
    engine.dispose();
  });

  it('context-lost is routed before dispose — subscriber receives it', async () => {
    const { gl, canvas } = makeGl();
    const engine = await createPTEngine_WebGL2({ device: gl });
    const received: EngineError[] = [];
    engine.onError!((err) => received.push(err));

    canvas.dispatchEvent('webglcontextlost', fakeContextLostEvent());
    expect(received).toHaveLength(1);
    engine.dispose();
  });

  it('unsubscribe before context-lost prevents delivery', async () => {
    const { gl, canvas } = makeGl();
    const engine = await createPTEngine_WebGL2({ device: gl });
    const received: EngineError[] = [];
    const unsub = engine.onError!((err) => received.push(err));
    unsub();

    canvas.dispatchEvent('webglcontextlost', fakeContextLostEvent());
    expect(received).toHaveLength(0);
    engine.dispose();
  });

  it('multiple subscribers all receive the error', async () => {
    const { gl, canvas } = makeGl();
    const engine = await createPTEngine_WebGL2({ device: gl });
    const a: EngineError[] = [];
    const b: EngineError[] = [];
    engine.onError!((err) => a.push(err));
    engine.onError!((err) => b.push(err));

    canvas.dispatchEvent('webglcontextlost', fakeContextLostEvent());
    expect(a).toHaveLength(1);
    expect(b).toHaveLength(1);
    engine.dispose();
  });
});

describe('pt-webgl2 EngineError surface — dispose clears subscribers', () => {
  it('dispose clears subscriber list — no delivery after dispose', async () => {
    const { gl, canvas } = makeGl();
    const engine = await createPTEngine_WebGL2({ device: gl });
    const received: EngineError[] = [];
    engine.onError!((err) => received.push(err));
    engine.dispose();

    canvas.dispatchEvent('webglcontextlost', fakeContextLostEvent());
    // After dispose, #onErrorSubs is cleared → no callback fires.
    expect(received).toHaveLength(0);
  });
});
