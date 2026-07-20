// onErrorCaptureFrameProxy.test.ts — regression guard for the H61-class bug
// where `wrapWithIdempotentDispose` silently hid `onError` and `captureFrame`.
//
// Bug: both methods were absent from the OPTIONAL_METHOD_PROXIES table and the
// bespoke-forward section; vanilla.ts's `engine.onError ? …` check was always
// false (device-loss recovery dead) and captureFrame always returned
// Promise.resolve(null) regardless of what the backend returned.
//
// Tests:
//   (a) wrapped engine forwards onError subscription and returns the backend unsub
//   (b) wrapped captureFrame forwards and returns the backend Promise value
//   (c) after dispose, onError returns a no-op unsub; captureFrame resolves null
//   (d) ledger guard — asserts the proxy exposes every Engine method that
//       the BACKEND_PROMISE_LEDGER advertises for all three shipping backends;
//       fails fast when a future optional method is added to the ledger but not
//       forwarded by the proxy (kills the H61 class permanently).

import { describe, it, expect, vi } from 'vitest';
import type {
  Engine,
  EngineCapabilities,
  EngineState,
  CapturedFrame,
} from '@vitrum/core';
import { BACKEND_PROMISE_LEDGER } from '@vitrum/core';
import { wrapWithIdempotentDispose } from '../createEngine.js';
import { stubCapabilities, stubEngine } from './fixtures/stubEngine.js';

// ── Helpers ────────────────────────────────────────────────────────────────────

function baseCapabilities(): EngineCapabilities {
  return stubCapabilities({
    supportsIncrementalScene: true,
    supportsAddRemovePrimitive: true,
    incrementalPatchSupport: {
      transform: true,
      positions: true,
      material: true,
      emitter: true,
      topology: true,
    },
  });
}

const FAKE_FRAME: CapturedFrame = {
  width: 1,
  height: 1,
  rgba: new Float32Array([0, 0, 0, 1]),
};

function makeEngine() {
  const backendUnsub = vi.fn();
  const onErrorSpy = vi.fn((_cb: (e: unknown) => void) => backendUnsub as () => void);
  const captureFrameSpy = vi.fn(async () => FAKE_FRAME as CapturedFrame | null);
  const engine: Engine = {
    ...stubEngine(baseCapabilities()),
    onError: onErrorSpy,
    captureFrame: captureFrameSpy,
  };
  return { engine, onErrorSpy, captureFrameSpy, backendUnsub };
}

// ── (a) onError forwarding pre-dispose ────────────────────────────────────────

describe('proxy — onError forwarding', () => {
  it('(a) pre-dispose: proxy.onError is defined and forwards the subscription', () => {
    const { engine, onErrorSpy, backendUnsub } = makeEngine();
    const proxy = wrapWithIdempotentDispose(engine, () => {});

    expect(typeof proxy.onError).toBe('function');

    const cb = vi.fn();
    const unsub = proxy.onError!(cb);

    // The backend spy was called with the callback.
    expect(onErrorSpy).toHaveBeenCalledTimes(1);
    expect(onErrorSpy).toHaveBeenCalledWith(cb);

    // The returned unsub IS the backend's unsub (not the post-dispose no-op).
    expect(unsub).toBe(backendUnsub);
  });

  it('(c-onError) post-dispose: onError returns a callable no-op unsub, does NOT forward', () => {
    const { engine, onErrorSpy } = makeEngine();
    const proxy = wrapWithIdempotentDispose(engine, () => {});
    proxy.dispose();

    const cb = vi.fn();
    const unsub = proxy.onError!(cb);

    // Must not have reached the backend.
    expect(onErrorSpy).not.toHaveBeenCalled();

    // Returns a callable no-op (not undefined, not the backend unsub).
    expect(typeof unsub).toBe('function');
    expect(() => unsub()).not.toThrow();
  });
});

// ── (b) captureFrame forwarding pre-dispose ────────────────────────────────────

describe('proxy — captureFrame forwarding', () => {
  it('(b) pre-dispose: proxy.captureFrame is defined and returns the backend Promise value', async () => {
    const { engine, captureFrameSpy } = makeEngine();
    const proxy = wrapWithIdempotentDispose(engine, () => {});

    expect(typeof proxy.captureFrame).toBe('function');

    const result = await proxy.captureFrame!();

    expect(captureFrameSpy).toHaveBeenCalledTimes(1);
    expect(result).toBe(FAKE_FRAME);
  });

  it('(b) pre-dispose: passes options through to the backend', async () => {
    const { engine, captureFrameSpy } = makeEngine();
    const proxy = wrapWithIdempotentDispose(engine, () => {});
    await proxy.captureFrame!({ colorSpace: 'output' });
    expect(captureFrameSpy).toHaveBeenCalledWith({ colorSpace: 'output' });
  });

  it('(c-captureFrame) post-dispose: resolves null without forwarding to the backend', async () => {
    const { engine, captureFrameSpy } = makeEngine();
    const proxy = wrapWithIdempotentDispose(engine, () => {});
    proxy.dispose();

    const result = await proxy.captureFrame!();

    // Must be null — the contract says null when no frame is available.
    expect(result).toBeNull();
    // Must NOT have touched the (torn-down) backend.
    expect(captureFrameSpy).not.toHaveBeenCalled();
  });
});

// ── (d) Ledger guard — H61 class kill-switch ─────────────────────────────────
//
// For every optional Engine method that ALL THREE shipping backends advertise in
// BACKEND_PROMISE_LEDGER.methodPromises, assert that a proxy wrapping a backend
// that implements that method exposes it (i.e. proxy[method] is defined).
//
// Design: if a future optional method is added to the ledger AND all three
// backends set it true, but the proxy table or bespoke-forward section doesn't
// forward it, this test fails — no more silent H61-class regressions.

describe('proxy — ledger completeness guard (H61-class kill-switch)', () => {
  it('proxy exposes every method that all three backends advertise and the engine implements', () => {
    const ledger = BACKEND_PROMISE_LEDGER;

    // Collect the keys where ALL three backends advertise `true`.
    const allBackendsTrue = (
      Object.keys(ledger['walkaround-hybrid'].methodPromises) as Array<
        keyof typeof ledger['walkaround-hybrid']['methodPromises']
      >
    ).filter((key) =>
      ledger['walkaround-hybrid'].methodPromises[key] === true &&
      ledger['pt-webgl2'].methodPromises[key] === true &&
      ledger['pt-webgpu'].methodPromises[key] === true,
    );

    // Build an engine that implements every one of those methods as a no-op/stub.
    const stubs: Record<string, unknown> = {};
    for (const key of allBackendsTrue) {
      // Skip non-callable ledger keys (e.g. 'debug' is a property, not a method).
      if (key === 'debug') continue;
      stubs[key] = vi.fn(() => {
        // Methods that return an unsub, a Promise, or a value all need stubs.
        if (key === 'onError' || key === 'onFrame' || key === 'onProgress') return () => {};
        if (key === 'captureFrame') return Promise.resolve(null);
        return null;
      });
    }

    const engine: Engine = {
      get state(): EngineState { return 'ready'; },
      get capabilities() { return baseCapabilities(); },
      setScene: vi.fn(),
      renderFrame: (_: import('@vitrum/core').FrameInput) => ({ kind: 'skipped' as const, samplesAccumulated: 0, isConverged: false }),
      reset: vi.fn(),
      pause: vi.fn(),
      resume: vi.fn(),
      dispose: vi.fn(),
      ...stubs,
    };

    const proxy = wrapWithIdempotentDispose(engine, () => {});

    const missing: string[] = [];
    for (const key of allBackendsTrue) {
      if (key === 'debug') continue; // property, not a method — checked separately
      if (typeof (proxy as unknown as Record<string, unknown>)[key] !== 'function') {
        missing.push(key);
      }
    }

    expect(
      missing,
      `Proxy does NOT forward these methods that all 3 backends advertise: ${missing.join(', ')}. ` +
        'Add them to OPTIONAL_METHOD_PROXIES or the bespoke-forward section in idempotentDispose.ts.',
    ).toHaveLength(0);
  });
});
