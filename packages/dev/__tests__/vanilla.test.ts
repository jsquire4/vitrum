/**
 * vanilla.test.ts — Tests for attachDebugOverlays (vanilla DOM overlay).
 *
 * Uses a minimal JSDOM-compatible DOM setup (vitest provides a happy-dom /
 * jsdom environment when configured). We mock the minimal DOM APIs the
 * vanilla overlay uses: createElement, appendChild, removeChild, and the
 * basic style assignments.
 *
 * Note: requestAnimationFrame is not available in the vitest node environment.
 * We mock it below so the rAF fallback path is exercised without hanging.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import type { Engine, EngineCapabilities, EngineState } from '@vitrum/core';
import type { DebuggableEngine } from '../src/types.js';

// ── rAF mock ──────────────────────────────────────────────────────────────────
// The vanilla overlay calls requestAnimationFrame when engine.onFrame is absent.
// We need to mock it so the test isn't async/waiting.

let rafCallbacks: Map<number, FrameRequestCallback> = new Map();
let rafCounter = 0;

function mockRequestAnimationFrame(cb: FrameRequestCallback): number {
  const id = ++rafCounter;
  rafCallbacks.set(id, cb);
  return id;
}
function mockCancelAnimationFrame(id: number): void {
  rafCallbacks.delete(id);
}

beforeEach(() => {
  rafCallbacks = new Map();
  rafCounter = 0;
  vi.stubGlobal('requestAnimationFrame', mockRequestAnimationFrame);
  vi.stubGlobal('cancelAnimationFrame', mockCancelAnimationFrame);
});

afterEach(() => {
  vi.unstubAllGlobals();
});

// ── Minimal engine ────────────────────────────────────────────────────────────

function makeEngine(): Engine {
  const caps: EngineCapabilities = {
    supportsIncrementalScene: false,
    supportsAuxBuffers: false,
    accumulates: false,
    maxSamplesPerPixel: Infinity,
    maxBounces: 4,
    supportedAnalyticShapes: new Set(),
    supportedEmitterKinds: new Set(),
    causticStrategy: 'none',
  };
  let s: EngineState = 'ready';
  return {
    get state() { return s; },
    capabilities: caps,
    setScene() {},
    renderFrame() {
      return { kind: 'skipped', samplesAccumulated: 0, isConverged: false };
    },
    reset() {},
    pause() { s = 'paused'; },
    resume() { s = 'ready'; },
    dispose() { s = 'disposed'; },
  };
}

// ── Minimal container ─────────────────────────────────────────────────────────
// We build a real DOM container (JSDOM / happy-dom provided by vitest env).
// If the test environment is pure node (no DOM), we skip DOM-dependent tests.

const hasDom = typeof document !== 'undefined';

// ────────────────────────────────────────────────────────────────────────────

describe('attachDebugOverlays', () => {
  it('module imports cleanly', async () => {
    const mod = await import('../src/vanilla.js');
    expect(typeof mod.attachDebugOverlays).toBe('function');
  });

  it.skipIf(!hasDom)(
    'attaches DOM nodes to the container',
    async () => {
      const { attachDebugOverlays } = await import('../src/vanilla.js');
      const container = document.createElement('div');
      document.body.appendChild(container);

      const engine = makeEngine() as DebuggableEngine;
      const handle = attachDebugOverlays(engine, container, {
        overlays: ['frameTime'],
      });

      expect(container.children.length).toBeGreaterThan(0);

      handle.dispose();
      expect(container.children.length).toBe(0);

      document.body.removeChild(container);
    }
  );

  it.skipIf(!hasDom)(
    'dispose() is idempotent (can be called twice)',
    async () => {
      const { attachDebugOverlays } = await import('../src/vanilla.js');
      const container = document.createElement('div');
      document.body.appendChild(container);

      const engine = makeEngine() as DebuggableEngine;
      const handle = attachDebugOverlays(engine, container, {
        overlays: ['frameTime'],
      });

      handle.dispose();
      expect(() => handle.dispose()).not.toThrow();

      document.body.removeChild(container);
    }
  );

  it.skipIf(!hasDom)(
    'engine.onFrame path: subscribes and unsubscribes cleanly',
    async () => {
      const { attachDebugOverlays } = await import('../src/vanilla.js');
      const container = document.createElement('div');
      document.body.appendChild(container);

      let subscribedCb: ((s: { frameTimeMs: number }) => void) | null = null;
      let unsubscribeCalled = false;

      const engine = makeEngine() as DebuggableEngine;
      (engine as unknown as Record<string, unknown>)['onFrame'] = (cb: (s: { frameTimeMs: number }) => void) => {
        subscribedCb = cb;
        return () => { unsubscribeCalled = true; };
      };

      const handle = attachDebugOverlays(engine, container, { overlays: ['frameTime'] });

      // Simulate a frame event
      if (subscribedCb !== null) {
        (subscribedCb as (s: { frameTimeMs: number }) => void)({ frameTimeMs: 16.67 });
      }
      // DOM nodes exist
      expect(container.children.length).toBeGreaterThan(0);

      handle.dispose();
      expect(unsubscribeCalled).toBe(true);
      expect(container.children.length).toBe(0);

      document.body.removeChild(container);
    }
  );

  it.skipIf(!hasDom)(
    'rAF fallback path: starts rAF loop and cancels on dispose',
    async () => {
      const { attachDebugOverlays } = await import('../src/vanilla.js');
      const container = document.createElement('div');
      document.body.appendChild(container);

      // Engine has no onFrame — should fall back to rAF
      const engine = makeEngine() as DebuggableEngine;
      const handle = attachDebugOverlays(engine, container, { overlays: ['frameTime'] });

      expect(rafCallbacks.size).toBeGreaterThan(0);

      handle.dispose();
      expect(rafCallbacks.size).toBe(0);

      document.body.removeChild(container);
    }
  );

  it.skipIf(!hasDom)(
    'stub overlays emit console.warn and render badge',
    async () => {
      const { attachDebugOverlays } = await import('../src/vanilla.js');
      const container = document.createElement('div');
      document.body.appendChild(container);

      const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => { /* suppress */ });

      const engine = makeEngine() as DebuggableEngine;
      const handle = attachDebugOverlays(engine, container, {
        overlays: ['ddgiAtlas'],
      });

      expect(warnSpy).toHaveBeenCalledWith(
        expect.stringContaining('[attachDebugOverlays]'),
      );

      handle.dispose();
      warnSpy.mockRestore();
      document.body.removeChild(container);
    }
  );
});
