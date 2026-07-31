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
import type { EngineWithBackendId } from '../src/createEngine.js';
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

  it('uses a supplied engine and re-targets a supplied scene controller without constructing another engine', async () => {
    const { attachVitrum } = await import('../src/lifecycle/vanilla.js');
    const createEngineModule = await import('../src/createEngine.js');
    const createSpy = vi.spyOn(createEngineModule, 'createEngine')
      .mockRejectedValue(new Error('createEngine should not be called for supplied engines'));

    const canvas = happyWindow.document.createElement('canvas') as unknown as HTMLCanvasElement;
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
    const engine = Object.assign(makeMockEngine(), { backendId: 'pt-webgl2' as const }) as EngineWithBackendId;
    const sceneController = { attachEngine: vi.fn() };

    const handle = await attachVitrum({
      canvas,
      scene,
      camera,
      engine,
      sceneController,
    });

    expect(createSpy).not.toHaveBeenCalled();
    expect(handle.engine).toBe(engine);
    expect(handle.backendId).toBe('pt-webgl2');
    expect(sceneController.attachEngine).toHaveBeenCalledWith(engine, { setScene: false });

    handle.dispose();
    expect(engine.dispose).toHaveBeenCalled();
  });

  it('advances a supplied scene controller with RAF delta seconds when playback is enabled', async () => {
    const scheduledFrames: FrameRequestCallback[] = [];
    (globalThis as Record<string, unknown>).requestAnimationFrame = vi.fn((cb: FrameRequestCallback) => {
      scheduledFrames.push(cb);
      return scheduledFrames.length;
    });
    (globalThis as Record<string, unknown>).cancelAnimationFrame = vi.fn();

    const { attachVitrum } = await import('../src/lifecycle/vanilla.js');
    const createEngineModule = await import('../src/createEngine.js');
    const createSpy = vi.spyOn(createEngineModule, 'createEngine')
      .mockRejectedValue(new Error('createEngine should not be called for supplied engines'));

    const canvas = happyWindow.document.createElement('canvas') as unknown as HTMLCanvasElement;
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
    const engine = Object.assign(makeMockEngine(), { backendId: 'pt-webgl2' as const }) as EngineWithBackendId;
    const sceneController = {
      animations: [{}],
      attachEngine: vi.fn(),
      advance: vi.fn(),
    };

    const handle = await attachVitrum({
      canvas,
      scene,
      camera,
      engine,
      sceneController,
      sceneControllerPlayback: { loop: false },
    });

    expect(createSpy).not.toHaveBeenCalled();
    expect(scheduledFrames).toHaveLength(1);

    scheduledFrames.shift()?.(1000);
    expect(sceneController.advance).not.toHaveBeenCalled();

    scheduledFrames.shift()?.(1016);
    expect(sceneController.advance).toHaveBeenCalledTimes(1);
    const [deltaSeconds, advanceOptions] = sceneController.advance.mock.calls[0]!;
    expect(deltaSeconds).toBeCloseTo(0.016, 6);
    expect(advanceOptions).toEqual({ engine, loop: false });

    handle.dispose();
  });

  it('forwards previous view/projection matrices starting on the second RAF tick', async () => {
    const scheduledFrames: FrameRequestCallback[] = [];
    (globalThis as Record<string, unknown>).requestAnimationFrame = vi.fn((cb: FrameRequestCallback) => {
      scheduledFrames.push(cb);
      return scheduledFrames.length;
    });
    (globalThis as Record<string, unknown>).cancelAnimationFrame = vi.fn();

    const { attachVitrum } = await import('../src/lifecycle/vanilla.js');
    const createEngineModule = await import('../src/createEngine.js');
    const createSpy = vi.spyOn(createEngineModule, 'createEngine')
      .mockRejectedValue(new Error('createEngine should not be called for supplied engines'));

    const canvas = happyWindow.document.createElement('canvas') as unknown as HTMLCanvasElement;
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
    const engine = Object.assign(makeMockEngine(), { backendId: 'pt-webgl2' as const }) as
      EngineWithBackendId & { readonly _renders: FrameInput[] };

    const handle = await attachVitrum({
      canvas,
      scene,
      camera,
      engine,
    });

    expect(createSpy).not.toHaveBeenCalled();
    expect(scheduledFrames).toHaveLength(1);

    scheduledFrames.shift()?.(1000);
    scheduledFrames.shift()?.(1016);

    expect(engine._renders).toHaveLength(2);
    expect(engine._renders[0]?.prevViewMatrix).toBeUndefined();
    expect(engine._renders[0]?.prevProjMatrix).toBeUndefined();
    expect(engine._renders[1]?.prevViewMatrix).toBe(engine._renders[0]?.viewMatrix);
    expect(engine._renders[1]?.prevProjMatrix).toBe(engine._renders[0]?.projMatrix);

    handle.dispose();
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

  it('contains camera/frame-preparation failures and applies the RAF self-stop policy', async () => {
    const scheduledFrames: FrameRequestCallback[] = [];
    const requestFrame = vi.fn((callback: FrameRequestCallback) => {
      scheduledFrames.push(callback);
      return scheduledFrames.length;
    });
    const cancelFrame = vi.fn();
    (globalThis as Record<string, unknown>).requestAnimationFrame = requestFrame;
    (globalThis as Record<string, unknown>).cancelAnimationFrame = cancelFrame;

    const { attachVitrum } = await import('../src/lifecycle/vanilla.js');
    const renderSpy = vi.fn((): FrameOutput => ({
      kind: 'skipped',
      samplesAccumulated: 0,
      isConverged: false,
    }));
    const engine = Object.assign(makeMockEngine(renderSpy), {
      backendId: 'pt-webgl2' as const,
    }) as EngineWithBackendId;
    const identity = new Float32Array([
      1,0,0,0,
      0,1,0,0,
      0,0,1,0,
      0,0,0,1,
    ]);
    const preparationFailure = new Error('camera preparation failed');
    const events: Array<{ readonly error: unknown; readonly recoverable: boolean }> = [];
    const handle = await attachVitrum({
      canvas: happyWindow.document.createElement('canvas') as unknown as HTMLCanvasElement,
      engine,
      scene: { primitives: [], emitters: [], environment: { kind: 'none' } },
      camera: {
        updateMatrixWorld: () => { throw preparationFailure; },
        matrixWorldInverse: { elements: identity },
        projectionMatrix: { elements: identity },
        position: { x: 0, y: 0, z: 0 },
      },
      onError: (error, event) => {
        events.push({ error, recoverable: event.recoverable });
      },
    });

    for (let attempt = 0; attempt < 5; attempt += 1) {
      const callback = scheduledFrames.shift();
      expect(callback).toBeDefined();
      expect(() => callback?.(attempt * 16)).not.toThrow();
    }

    expect(renderSpy).not.toHaveBeenCalled();
    expect(events).toHaveLength(5);
    expect(events.every(({ error }) => error === preparationFailure)).toBe(true);
    expect(events.slice(0, 4).every(({ recoverable }) => recoverable)).toBe(true);
    expect(events[4]?.recoverable).toBe(false);
    expect(cancelFrame).toHaveBeenCalledOnce();

    handle.dispose();
  });

  it('commits frame index and previous matrices only after a rendered output', async () => {
    const scheduledFrames: FrameRequestCallback[] = [];
    (globalThis as Record<string, unknown>).requestAnimationFrame = vi.fn(
      (callback: FrameRequestCallback) => {
        scheduledFrames.push(callback);
        return scheduledFrames.length;
      },
    );
    (globalThis as Record<string, unknown>).cancelAnimationFrame = vi.fn();
    vi.spyOn(console, 'error').mockImplementation(() => {});

    const outputs: Array<'skipped' | 'throw' | 'rendered'> = [
      'skipped',
      'throw',
      'rendered',
      'rendered',
    ];
    let outputIndex = 0;
    const engine = Object.assign(
      makeMockEngine(() => {
        const output = outputs[outputIndex++];
        if (output === 'throw') throw new Error('injected render failure');
        if (output === 'rendered') {
          return {
            kind: 'rendered',
            primaryRadiance: {},
            samplesAccumulated: 1,
            isConverged: false,
          } as unknown as FrameOutput;
        }
        return {
          kind: 'skipped',
          samplesAccumulated: 0,
          isConverged: false,
        };
      }),
      { backendId: 'pt-webgl2' as const },
    ) as EngineWithBackendId;
    const view = new Float32Array([
      1,0,0,0,
      0,1,0,0,
      0,0,1,0,
      0,0,0,1,
    ]);
    const projection = new Float32Array(view);
    const handle = await (await import('../src/lifecycle/vanilla.js')).attachVitrum({
      canvas: happyWindow.document.createElement('canvas') as unknown as HTMLCanvasElement,
      engine,
      scene: { primitives: [], emitters: [], environment: { kind: 'none' } },
      camera: {
        updateMatrixWorld: vi.fn(),
        matrixWorldInverse: { elements: view },
        projectionMatrix: { elements: projection },
        position: { x: 0, y: 0, z: 0 },
      },
    });

    for (let frame = 0; frame < outputs.length; frame += 1) {
      projection[0] = frame + 1;
      const callback = scheduledFrames.shift();
      expect(callback).toBeDefined();
      callback?.(frame * 16);
    }

    const inputs = (
      engine as EngineWithBackendId & { readonly _renders: readonly FrameInput[] }
    )._renders;
    expect(inputs.map(({ frameIndex }) => frameIndex)).toEqual([0, 0, 0, 1]);
    expect(inputs[0]?.prevProjMatrix).toBeUndefined();
    expect(inputs[1]?.prevProjMatrix).toBeUndefined();
    expect(inputs[2]?.prevProjMatrix).toBeUndefined();
    expect(inputs[3]?.prevProjMatrix?.[0]).toBe(3);

    handle.dispose();
  });

  it('H30 fix — ResizeObserver updates the backing store and calls swapchain-optional setSize', async () => {
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
    const setSize = vi.fn();
    (engine.capabilities as { presentationMode: string }).presentationMode = 'swapchain-optional';
    (engine as Engine & { setSize(width: number, height: number): void }).setSize = setSize;
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
    expect(setSize).toHaveBeenCalledWith(800, 600);

    handle.dispose();
    vi.restoreAllMocks();
  });

  it('rolls back the constructed engine and partial observer when initial observation fails', async () => {
    const { attachVitrum } = await import('../src/lifecycle/vanilla.js');
    const failure = new Error('observe failed');
    const disconnect = vi.fn();
    (globalThis as Record<string, unknown>).ResizeObserver = class FailingRO {
      observe(): void { throw failure; }
      disconnect(): void { disconnect(); }
      unobserve(): void {}
    };
    const engine = Object.assign(makeMockEngine(), {
      backendId: 'pt-webgl2' as const,
      backendProfileId: 'pt-webgl2' as const,
      profileId: 'pt-webgl2' as const,
    }) as unknown as EngineWithBackendId;
    const canvas = happyWindow.document.createElement('canvas') as unknown as HTMLCanvasElement;
    const identity = new Float32Array([1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]);

    await expect(attachVitrum({
      canvas,
      engine,
      scene: { primitives: [], emitters: [], environment: { kind: 'none' } },
      camera: {
        updateMatrixWorld: vi.fn(),
        matrixWorldInverse: { elements: identity },
        projectionMatrix: { elements: identity },
        position: { x: 0, y: 0, z: 0 },
      },
    })).rejects.toBe(failure);

    expect(disconnect).toHaveBeenCalledOnce();
    expect(engine.dispose).toHaveBeenCalledOnce();
  });

  it('rolls back earlier subscriptions, DOM hooks, and engine when a later subscription fails', async () => {
    const { attachVitrum } = await import('../src/lifecycle/vanilla.js');
    const failure = new Error('progress subscription failed');
    const disconnect = vi.fn();
    const unsubFrame = vi.fn();
    (globalThis as Record<string, unknown>).ResizeObserver = class TrackingRO {
      observe(): void {}
      disconnect(): void { disconnect(); }
      unobserve(): void {}
    };
    const engine = Object.assign(makeMockEngine(), {
      backendId: 'pt-webgl2' as const,
      backendProfileId: 'pt-webgl2' as const,
      profileId: 'pt-webgl2' as const,
      onFrame: vi.fn(() => unsubFrame),
      onProgress: vi.fn(() => { throw failure; }),
    }) as unknown as EngineWithBackendId;
    const removeEventListener = vi.spyOn(document, 'removeEventListener');
    const requestFrame = vi.spyOn(globalThis, 'requestAnimationFrame');
    const canvas = happyWindow.document.createElement('canvas') as unknown as HTMLCanvasElement;
    const identity = new Float32Array([1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]);

    await expect(attachVitrum({
      canvas,
      engine,
      scene: { primitives: [], emitters: [], environment: { kind: 'none' } },
      camera: {
        updateMatrixWorld: vi.fn(),
        matrixWorldInverse: { elements: identity },
        projectionMatrix: { elements: identity },
        position: { x: 0, y: 0, z: 0 },
      },
      onFrame: vi.fn(),
      onProgress: vi.fn(),
    })).rejects.toBe(failure);

    expect(unsubFrame).toHaveBeenCalledOnce();
    expect(disconnect).toHaveBeenCalledOnce();
    expect(removeEventListener).toHaveBeenCalledWith(
      'visibilitychange',
      expect.any(Function),
    );
    expect(requestFrame).not.toHaveBeenCalled();
    expect(engine.dispose).toHaveBeenCalledOnce();
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
