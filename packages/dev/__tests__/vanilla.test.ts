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
import {
  asBackendTexture,
  type EngineCapabilities,
  type EngineState,
  type Scene,
} from '@vitrum/core';
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

function makeEngine(overrides: Partial<DebuggableEngine> = {}): DebuggableEngine {
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
    ...overrides,
  };
}

function makeScene(): Scene {
  return {
    primitives: [
      {
        kind: 'mesh',
        id: 'mesh-1',
        positions: new Float32Array([
          0, 0, 0,
          1, 0, 0,
          0, 1, 0,
        ]),
        normals: new Float32Array([
          0, 0, 1,
          0, 0, 1,
          0, 0, 1,
        ]),
        material: {
          baseColor: [0.25, 0.5, 0.75],
          roughness: 0.4,
          metallic: 0.1,
        },
      },
    ],
    emitters: [],
    environment: { kind: 'none' },
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

      const engine = makeEngine();
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

      const engine = makeEngine();
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

      const engine = makeEngine();
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
      const engine = makeEngine();
      const handle = attachDebugOverlays(engine, container, { overlays: ['frameTime'] });

      expect(rafCallbacks.size).toBeGreaterThan(0);

      handle.dispose();
      expect(rafCallbacks.size).toBe(0);

      document.body.removeChild(container);
    }
  );

  it.skipIf(!hasDom)(
    'ddgiAtlas renders truthful diagnostics without React-only warnings',
    async () => {
      const { attachDebugOverlays } = await import('../src/vanilla.js');
      const container = document.createElement('div');
      document.body.appendChild(container);

      const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => { /* suppress */ });

      const engine = makeEngine();
      const handle = attachDebugOverlays(engine, container, {
        overlays: ['ddgiAtlas'],
      });

      expect(warnSpy).not.toHaveBeenCalled();
      expect(container.textContent).toContain('DDGI / capabilities');
      expect(container.textContent).toContain('debugSurface');
      expect(container.textContent).toContain('texture api unavailable');

      handle.dispose();
      warnSpy.mockRestore();
      document.body.removeChild(container);
    }
  );

  it.skipIf(!hasDom)(
    'denoiser diagnostics toggles only when debug setter exists',
    async () => {
      const { attachDebugOverlays } = await import('../src/vanilla.js');
      const container = document.createElement('div');
      document.body.appendChild(container);

      let enabled = true;
      const setDenoiserEnabled = vi.fn((next: boolean) => { enabled = next; });
      const engine = makeEngine({
        debug: {
          isDenoiserEnabled: () => enabled,
          setDenoiserEnabled,
        },
      });

      const handle = attachDebugOverlays(engine, container, {
        overlays: ['denoiserToggle'],
      });

      expect(container.textContent).toContain('denoiser on [D]');
      (container.firstElementChild as HTMLElement).click();
      expect(setDenoiserEnabled).toHaveBeenCalledWith(false);
      expect(container.textContent).toContain('denoiser off [D]');

      handle.dispose();
      document.body.removeChild(container);
    }
  );

  it.skipIf(!hasDom)(
    'giSignalSplit reports debug textures and last rendered frame outputs',
    async () => {
      const { attachDebugOverlays } = await import('../src/vanilla.js');
      const container = document.createElement('div');
      document.body.appendChild(container);

      const engine = makeEngine({
        debug: {
          giSignalTextures: () => ({
            direct: {},
            indirect: null,
            ao: {},
            total: {},
          }) as never,
          device: () => null,
        },
        renderFrame() {
          return {
            kind: 'rendered',
            samplesAccumulated: 3,
            isConverged: false,
            primaryRadiance: asBackendTexture<'test', object>({}),
            normalDepth: asBackendTexture<'test', object>({}),
            albedo: asBackendTexture<'test', object>({}),
          };
        },
      });
      const originalRenderFrame = engine.renderFrame;

      const handle = attachDebugOverlays(engine, container, {
        overlays: ['giSignalSplit'],
      });

      expect(container.textContent).toContain('waiting');
      engine.renderFrame({} as never);
      expect(container.textContent).toContain('rendered, spp 3, active');
      expect(container.textContent).toContain('normalDepth');
      expect(container.textContent).toContain('motion');
      expect(container.textContent).toContain('missing');

      handle.dispose();
      expect(engine.renderFrame).toBe(originalRenderFrame);
      document.body.removeChild(container);
    }
  );

  it.skipIf(!hasDom)(
    'bvhVisualizer renders node statistics when debug.bvhNodes exists',
    async () => {
      const { attachDebugOverlays } = await import('../src/vanilla.js');
      const container = document.createElement('div');
      document.body.appendChild(container);

      const nodes = new Float32Array([
        0, 0, 0, 1, 1, 1, 0, 0,
        0, 0, 0, 0.5, 1, 1, 1, 0,
        0.5, 0, 0, 1, 1, 1, 1, 0,
      ]);
      const engine = makeEngine({
        debug: {
          bvhNodes: () => nodes,
        },
      });

      const handle = attachDebugOverlays(engine, container, {
        overlays: ['bvhVisualizer'],
      });

      expect(container.textContent).toContain('BVH structure');
      expect(container.textContent).toContain('nodes');
      expect(container.textContent).toContain('3');
      expect(container.textContent).toContain('max depth');
      expect(container.textContent).toContain('1');

      handle.dispose();
      document.body.removeChild(container);
    }
  );

  it.skipIf(!hasDom)(
    'materialInspector reports picked primitive details when debug picking exists',
    async () => {
      const { attachDebugOverlays } = await import('../src/vanilla.js');
      const container = document.createElement('div');
      const canvas = document.createElement('canvas');
      canvas.width = 200;
      canvas.height = 100;
      Object.defineProperty(canvas, 'getBoundingClientRect', {
        value: () => ({
          left: 0,
          top: 0,
          width: 100,
          height: 50,
          right: 100,
          bottom: 50,
          x: 0,
          y: 0,
          toJSON: () => ({}),
        }),
      });
      container.appendChild(canvas);
      document.body.appendChild(container);

      const pickPrimitive = vi.fn(() => 'mesh-1');
      const engine = makeEngine({
        capabilities: {
          ...makeEngine().capabilities,
          supportsIncrementalScene: true,
          incrementalPatchSupport: {
            transform: false,
            positions: false,
            material: true,
            emitter: false,
            topology: false,
          },
        },
        debug: { pickPrimitive },
      });

      const handle = attachDebugOverlays(engine, container, {
        overlays: ['materialInspector'],
        scene: makeScene(),
        canvas,
      });

      canvas.dispatchEvent(new MouseEvent('click', { clientX: 50, clientY: 25 }));
      expect(pickPrimitive).toHaveBeenCalledWith(100, 50);
      expect(container.textContent).toContain('mesh-1');
      expect(container.textContent).toContain('mesh');
      expect(container.textContent).toContain('base [0.25,0.50,0.75]');
      expect(container.textContent).toContain('native');

      handle.dispose();
      document.body.removeChild(container);
    }
  );
});
