// H54 — lifecycle tests for attachVitrum using the happy-dom DOM stub.
//
// These tests verify the H30 canvas initial sizing, the RAF tick, the
// ResizeObserver wiring, and the dispose path. They also verify the H31-d
// RAF self-stop behaviour after N consecutive renderFrame throws.
//
// The real backend engines cannot be constructed in this environment (no GPU),
// so we use a minimal mock engine that satisfies the @vitrum/core Engine
// contract. attachVitrum itself is tested against its exported pure helpers
// (detectWebGPUSwapChain, toPhysicalViewport, etc.) plus an end-to-end
// attachVitrum call with the mock engine injected via the createEngine seam.

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { Window } from 'happy-dom';
import type { Engine, FrameInput, FrameOutput } from '@vitrum/core';
import {
  toPhysicalViewport,
  resolveQualityOption,
  detectWebGPUSwapChain,
} from '../src/lifecycle/vanilla.js';

// ────────────────────────────────────────────────────────────────────────────
// Pure-helper tests (no DOM needed)
// ────────────────────────────────────────────────────────────────────────────

describe('toPhysicalViewport', () => {
  it('floors CSS × DPR and clamps to 1', () => {
    const vp = toPhysicalViewport(1280, 720, 2);
    expect(vp.width).toBe(2560);
    expect(vp.height).toBe(1440);
    expect(vp.devicePixelRatio).toBe(2);
  });

  it('clamps zero-size to 1×1', () => {
    const vp = toPhysicalViewport(0, 0, 1);
    expect(vp.width).toBe(1);
    expect(vp.height).toBe(1);
  });

  it('treats non-finite DPR as 1', () => {
    const vp = toPhysicalViewport(100, 100, NaN);
    expect(vp.devicePixelRatio).toBe(1);
    expect(vp.width).toBe(100);
  });

  it('treats negative DPR as 1', () => {
    const vp = toPhysicalViewport(100, 100, -2);
    expect(vp.devicePixelRatio).toBe(1);
  });
});

describe('resolveQualityOption', () => {
  it('returns undefined for null/undefined', () => {
    expect(resolveQualityOption(undefined)).toBeUndefined();
    expect(resolveQualityOption(null as never)).toBeUndefined();
  });

  it('passes through a static quality value', () => {
    const q = { samplesPerFrame: 4 } as FrameInput['quality'];
    expect(resolveQualityOption(q)).toBe(q);
  });

  it('invokes a getter function and returns its result', () => {
    const q = { samplesPerFrame: 8 } as FrameInput['quality'];
    const getter = vi.fn(() => q);
    expect(resolveQualityOption(getter)).toBe(q);
    expect(getter).toHaveBeenCalledOnce();
  });
});

// ────────────────────────────────────────────────────────────────────────────
// DOM-backed tests: canvas initial sizing (H30) + RAF loop + ResizeObserver
// ────────────────────────────────────────────────────────────────────────────

/**
 * Minimal mock engine that records renderFrame calls. Satisfies the Engine
 * contract so attachVitrum can use it without a real GPU device.
 */
function makeMockEngine(renderOverride?: (input: FrameInput) => FrameOutput): Engine {
  const renders: FrameInput[] = [];
  const capabilities = {
    presentationMode: 'offscreen-texture' as const,
    supportedAnalyticShapes: new Set(),
    supportedEmitterKinds: new Set(),
    supportedPrimitiveKinds: new Set(['mesh'] as const),
    supportedEnvironmentKinds: new Set(),
    supportsIncrementalScene: false,
    supportsAuxBuffers: false,
    accumulates: true,
    maxSamplesPerPixel: Infinity,
    maxBounces: 8,
  };
  return {
    get state() { return 'ready' as const; },
    get capabilities() { return capabilities; },
    setScene: vi.fn(),
    renderFrame(input: FrameInput): FrameOutput {
      renders.push(input);
      return renderOverride
        ? renderOverride(input)
        : { kind: 'skipped', samplesAccumulated: 0, isConverged: false };
    },
    reset: vi.fn(),
    pause: vi.fn(),
    resume: vi.fn(),
    dispose: vi.fn(),
    // expose renders for assertions
    _renders: renders,
  } as unknown as Engine;
}

describe('attachVitrum with happy-dom + mock engine', () => {
  let happyWindow: Window;
  // Save / restore globals that attachVitrum reads.
  let savedWindow: typeof globalThis.window | undefined;
  let savedDocument: typeof globalThis.document | undefined;
  let savedRequestAnimationFrame: typeof globalThis.requestAnimationFrame | undefined;
  let savedCancelAnimationFrame: typeof globalThis.cancelAnimationFrame | undefined;
  let savedResizeObserver: typeof globalThis.ResizeObserver | undefined;

  beforeEach(() => {
    happyWindow = new Window({ url: 'http://localhost/' });

    // Inject happy-dom globals so the vanilla.ts module picks them up.
    savedWindow = (globalThis as { window?: typeof globalThis.window }).window;
    savedDocument = (globalThis as { document?: typeof globalThis.document }).document;
    savedRequestAnimationFrame = (globalThis as { requestAnimationFrame?: typeof globalThis.requestAnimationFrame }).requestAnimationFrame;
    savedCancelAnimationFrame = (globalThis as { cancelAnimationFrame?: typeof globalThis.cancelAnimationFrame }).cancelAnimationFrame;
    savedResizeObserver = (globalThis as { ResizeObserver?: typeof globalThis.ResizeObserver }).ResizeObserver;

    (globalThis as Record<string, unknown>).window = happyWindow;
    (globalThis as Record<string, unknown>).document = happyWindow.document;
    (globalThis as Record<string, unknown>).requestAnimationFrame = happyWindow.requestAnimationFrame.bind(happyWindow);
    (globalThis as Record<string, unknown>).cancelAnimationFrame = happyWindow.cancelAnimationFrame.bind(happyWindow);
    (globalThis as Record<string, unknown>).ResizeObserver = happyWindow.ResizeObserver;
  });

  afterEach(() => {
    (globalThis as Record<string, unknown>).window = savedWindow;
    (globalThis as Record<string, unknown>).document = savedDocument;
    (globalThis as Record<string, unknown>).requestAnimationFrame = savedRequestAnimationFrame;
    (globalThis as Record<string, unknown>).cancelAnimationFrame = savedCancelAnimationFrame;
    (globalThis as Record<string, unknown>).ResizeObserver = savedResizeObserver;
    happyWindow.close();
    vi.restoreAllMocks();
  });

  it('H30 — initial canvas.width/height is set before the first frame', async () => {
    // Inline-import attachVitrum so it reads the globals we just set.
    const { attachVitrum } = await import('../src/lifecycle/vanilla.js');

    const canvas = happyWindow.document.createElement('canvas') as unknown as HTMLCanvasElement;
    // Simulate a CSS-styled canvas: clientWidth=0 in happy-dom (no layout),
    // so the fallback path uses canvas.width/height (300×150 default).
    expect(canvas.width).toBe(300);
    expect(canvas.height).toBe(150);

    const engine = makeMockEngine();
    const { asMat4 } = await import('@vitrum/core');
    const identity = asMat4(new Float32Array([1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]));
    const camera = {
      updateMatrixWorld: vi.fn(),
      matrixWorldInverse: { elements: identity },
      projectionMatrix: { elements: identity },
      position: { x: 0, y: 0, z: 0 },
    };

    // Override createEngine so attachVitrum uses our mock engine.
    const createEngineModule = await import('../src/createEngine.js');
    const originalCreateEngine = createEngineModule.createEngine;
    vi.spyOn(createEngineModule, 'createEngine').mockResolvedValue(engine as ReturnType<typeof createEngineModule.createEngine> extends Promise<infer T> ? T : never);

    const scene = {
      primitives: [],
      emitters: [],
      environment: { kind: 'none' as const },
      version: 1,
    };

    const handle = await attachVitrum({
      canvas,
      scene,
      camera,
    });

    // H30: backing store should have been written synchronously before first RAF.
    // Since clientWidth=0 in happy-dom, the fallback uses canvas.width×DPR.
    // DPR in happy-dom = 1 (window.devicePixelRatio), so result = 300×150.
    expect(canvas.width).toBe(300);
    expect(canvas.height).toBe(150);

    handle.dispose();
    vi.restoreAllMocks();
    // Restore for next test
    vi.spyOn(createEngineModule, 'createEngine').mockImplementation(originalCreateEngine);
  });

  it('H30 — CSS size × DPR backing store is applied before createEngine runs', async () => {
    const { attachVitrum } = await import('../src/lifecycle/vanilla.js');
    const createEngineModule = await import('../src/createEngine.js');

    Object.defineProperty(happyWindow, 'devicePixelRatio', { value: 2, configurable: true });
    const canvas = happyWindow.document.createElement('canvas') as unknown as HTMLCanvasElement;
    Object.defineProperty(canvas, 'clientWidth', { value: 640, configurable: true });
    Object.defineProperty(canvas, 'clientHeight', { value: 360, configurable: true });

    const engine = makeMockEngine();
    const { asMat4 } = await import('@vitrum/core');
    const identity = asMat4(new Float32Array([1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]));
    const camera = {
      updateMatrixWorld: vi.fn(),
      matrixWorldInverse: { elements: identity },
      projectionMatrix: { elements: identity },
      position: { x: 0, y: 0, z: 0 },
    };
    const scene = {
      primitives: [],
      emitters: [],
      environment: { kind: 'none' as const },
    };

    const createSpy = vi.spyOn(createEngineModule, 'createEngine').mockImplementation(async (opts) => {
      expect(opts.canvas.width).toBe(1280);
      expect(opts.canvas.height).toBe(720);
      return engine as ReturnType<typeof createEngineModule.createEngine> extends Promise<infer T> ? T : never;
    });

    const handle = await attachVitrum({ canvas, scene, camera });

    expect(createSpy).toHaveBeenCalledOnce();
    expect(canvas.width).toBe(1280);
    expect(canvas.height).toBe(720);
    handle.dispose();
  });

  it('forwards onWarning into createEngine options', async () => {
    const { attachVitrum } = await import('../src/lifecycle/vanilla.js');
    const createEngineModule = await import('../src/createEngine.js');

    const canvas = happyWindow.document.createElement('canvas') as unknown as HTMLCanvasElement;
    const engine = makeMockEngine();
    const { asMat4 } = await import('@vitrum/core');
    const identity = asMat4(new Float32Array([1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]));
    const camera = {
      updateMatrixWorld: vi.fn(),
      matrixWorldInverse: { elements: identity },
      projectionMatrix: { elements: identity },
      position: { x: 0, y: 0, z: 0 },
    };
    const scene = {
      primitives: [],
      emitters: [],
      environment: { kind: 'none' as const },
    };
    const onWarning = vi.fn();

    const createSpy = vi.spyOn(createEngineModule, 'createEngine').mockResolvedValue(
      engine as ReturnType<typeof createEngineModule.createEngine> extends Promise<infer T> ? T : never,
    );

    const handle = await attachVitrum({ canvas, scene, camera, onWarning });

    expect(createSpy).toHaveBeenCalledWith(expect.objectContaining({ onWarning }));
    handle.dispose();
  });

  it('forwards gltfAsset into createEngine options', async () => {
    const { attachVitrum } = await import('../src/lifecycle/vanilla.js');
    const createEngineModule = await import('../src/createEngine.js');

    const canvas = happyWindow.document.createElement('canvas') as unknown as HTMLCanvasElement;
    const engine = makeMockEngine();
    const { asMat4 } = await import('@vitrum/core');
    const identity = asMat4(new Float32Array([1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]));
    const camera = {
      updateMatrixWorld: vi.fn(),
      matrixWorldInverse: { elements: identity },
      projectionMatrix: { elements: identity },
      position: { x: 0, y: 0, z: 0 },
    };
    const scene = {
      primitives: [],
      emitters: [],
      environment: { kind: 'none' as const },
    };
    const gltfAsset = { recommendedBackend: { backend: 'pt-webgl2' as const } };

    const createSpy = vi.spyOn(createEngineModule, 'createEngine').mockResolvedValue(
      engine as ReturnType<typeof createEngineModule.createEngine> extends Promise<infer T> ? T : never,
    );

    const handle = await attachVitrum({ canvas, scene, camera, gltfAsset });

    expect(createSpy).toHaveBeenCalledWith(expect.objectContaining({ gltfAsset }));
    handle.dispose();
  });

  it('H30 — ResizeObserver is wired to the canvas after attach', async () => {
    const { attachVitrum } = await import('../src/lifecycle/vanilla.js');
    const createEngineModule = await import('../src/createEngine.js');

    // Track ResizeObserver construction.
    const observeSpy = vi.fn();
    const disconnectSpy = vi.fn();
    const OriginalResizeObserver = happyWindow.ResizeObserver;
    (globalThis as Record<string, unknown>).ResizeObserver = class MockResizeObserver {
      constructor(cb: ResizeObserverCallback) {
        void cb; // capture the callback (not needed for this assertion)
      }
      observe(target: Element): void { observeSpy(target); }
      disconnect(): void { disconnectSpy(); }
      unobserve(): void {}
    };

    const canvas = happyWindow.document.createElement('canvas') as unknown as HTMLCanvasElement;
    const engine = makeMockEngine();
    const { asMat4 } = await import('@vitrum/core');
    const identity = asMat4(new Float32Array([1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]));
    const camera = {
      updateMatrixWorld: vi.fn(),
      matrixWorldInverse: { elements: identity },
      projectionMatrix: { elements: identity },
      position: { x: 0, y: 0, z: 0 },
    };

    vi.spyOn(createEngineModule, 'createEngine').mockResolvedValue(engine as ReturnType<typeof createEngineModule.createEngine> extends Promise<infer T> ? T : never);

    const scene = {
      primitives: [],
      emitters: [],
      environment: { kind: 'none' as const },
      version: 1,
    };

    const handle = await attachVitrum({ canvas, scene, camera });

    // ResizeObserver should have been observing the canvas.
    expect(observeSpy).toHaveBeenCalledWith(canvas);

    handle.dispose();

    // ResizeObserver should disconnect on dispose.
    expect(disconnectSpy).toHaveBeenCalled();

    // Restore
    (globalThis as Record<string, unknown>).ResizeObserver = OriginalResizeObserver;
    vi.restoreAllMocks();
  });

  it('H31-d — RAF loop self-stops after 5 consecutive renderFrame throws', async () => {
    const { attachVitrum } = await import('../src/lifecycle/vanilla.js');
    const createEngineModule = await import('../src/createEngine.js');

    const errors: { recoverable: boolean }[] = [];
    let throwCount = 0;
    const engine = makeMockEngine(() => {
      throwCount++;
      throw new Error(`renderFrame fail #${throwCount}`);
    });

    const { asMat4 } = await import('@vitrum/core');
    const identity = asMat4(new Float32Array([1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]));
    const camera = {
      updateMatrixWorld: vi.fn(),
      matrixWorldInverse: { elements: identity },
      projectionMatrix: { elements: identity },
      position: { x: 0, y: 0, z: 0 },
    };

    vi.spyOn(createEngineModule, 'createEngine').mockResolvedValue(engine as ReturnType<typeof createEngineModule.createEngine> extends Promise<infer T> ? T : never);

    const canvas = happyWindow.document.createElement('canvas') as unknown as HTMLCanvasElement;
    const scene = {
      primitives: [],
      emitters: [],
      environment: { kind: 'none' as const },
      version: 1,
    };

    const handle = await attachVitrum({
      canvas,
      scene,
      camera,
      onError: (_err, event) => { errors.push({ recoverable: event.recoverable }); },
    });

    // Advance RAF ticks until the loop self-stops.
    // happy-dom's requestAnimationFrame fires on runUntilComplete/tick.
    // We manually run ticks. The loop stops after 5 throws.
    await happyWindow.happyDOM.waitUntilComplete();

    // After the loop stops, the non-recoverable error should have been reported.
    const nonRecoverables = errors.filter((e) => !e.recoverable);
    expect(nonRecoverables.length).toBeGreaterThanOrEqual(1);

    handle.dispose();
    vi.restoreAllMocks();
  });

  it('H30 fix — ResizeObserver callback updates canvas.width/height (backing store)', async () => {
    // Verifies Bug 1: after a resize, canvas.width/height must track the new
    // CSS × DPR size so the swapchain textures are the right physical size.
    const { attachVitrum } = await import('../src/lifecycle/vanilla.js');
    const createEngineModule = await import('../src/createEngine.js');

    // Capture the ResizeObserver callback so we can fire it manually.
    let capturedCallback: ResizeObserverCallback | null = null;
    (globalThis as Record<string, unknown>).ResizeObserver = class MockRO {
      constructor(cb: ResizeObserverCallback) { capturedCallback = cb; }
      observe() {}
      disconnect() {}
      unobserve() {}
    };

    const canvas = happyWindow.document.createElement('canvas') as unknown as HTMLCanvasElement;
    canvas.width = 300;
    canvas.height = 150;

    const engine = makeMockEngine();
    const { asMat4 } = await import('@vitrum/core');
    const identity = asMat4(new Float32Array([1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]));
    const camera = {
      updateMatrixWorld: vi.fn(),
      matrixWorldInverse: { elements: identity },
      projectionMatrix: { elements: identity },
      position: { x: 0, y: 0, z: 0 },
    };

    vi.spyOn(createEngineModule, 'createEngine').mockResolvedValue(
      engine as ReturnType<typeof createEngineModule.createEngine> extends Promise<infer T> ? T : never,
    );

    const handle = await attachVitrum({
      canvas,
      scene: { primitives: [], emitters: [], environment: { kind: 'none' as const } },
      camera,
    });

    // Simulate a resize to 800×600 CSS pixels at DPR=1.
    expect(capturedCallback).not.toBeNull();
    const resizeEntry = [
      { contentRect: { width: 800, height: 600 } } as unknown as ResizeObserverEntry,
    ];
    (capturedCallback as unknown as ResizeObserverCallback)(resizeEntry, {} as ResizeObserver);

    // Canvas backing store must have been updated to the new physical size.
    expect(canvas.width).toBe(800);
    expect(canvas.height).toBe(600);

    handle.dispose();
    vi.restoreAllMocks();
  });
});

// ────────────────────────────────────────────────────────────────────────────
// detectWebGPUSwapChain — pure helper, no DOM needed
// ────────────────────────────────────────────────────────────────────────────

describe('detectWebGPUSwapChain', () => {
  it('returns null context for a canvas with no webgpu context support', () => {
    // Create a minimal fake canvas that returns null for getContext('webgpu').
    const fakeCanvas = {
      getContext: (id: string) => id === 'webgpu' ? null : null,
    } as unknown as HTMLCanvasElement;
    const result = detectWebGPUSwapChain(fakeCanvas);
    expect(result.context).toBeNull();
    expect(result.format).toBeUndefined();
  });

  it('returns null context when getContext throws', () => {
    const fakeCanvas = {
      getContext: () => { throw new Error('no webgpu'); },
    } as unknown as HTMLCanvasElement;
    const result = detectWebGPUSwapChain(fakeCanvas);
    expect(result.context).toBeNull();
    expect(result.format).toBeUndefined();
  });
});
