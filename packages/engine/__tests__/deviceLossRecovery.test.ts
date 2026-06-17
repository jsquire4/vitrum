// deviceLossRecovery.test.ts — auto-recreate on device-loss / context-lost.
//
// Tests the `autoRecreateOnDeviceLoss` option added to `attachVitrum`:
//   • Fatal device-lost/context-lost → dispose called, factory called again, loop resumes.
//   • Non-fatal errors do NOT trigger a recreate.
//   • Retry cap: after AUTO_RECREATE_MAX_ATTEMPTS in the window the loop stays stopped.
//   • GI export/import called around the recreate when the surface is present.
//   • The stable handle facade exposes the NEW engine after recreate.
//
// The tests use the happy-dom DOM stub (same approach as attachVitrumLoop.test.ts)
// and inject a mock createEngine via vi.spyOn, so no real GPU is required.

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { Window } from 'happy-dom';
import type { EngineError, FrameInput, FrameOutput } from '@vitrum/core';
import type { GIStateSnapshot } from '@vitrum/walkaround-hybrid';
import { AUTO_RECREATE_MAX_ATTEMPTS, AUTO_RECREATE_WINDOW_MS } from '../src/lifecycle/vanilla.js';

// ────────────────────────────────────────────────────────────────────────────
// Mock engine factories
// ────────────────────────────────────────────────────────────────────────────

/** A minimal Engine-like object for testing.
 *  Returns plain object cast to `unknown` so test sites can cast freely. */
function makeMockEngine(options?: {
  withGIState?: boolean;
  giSnapshot?: GIStateSnapshot | null;
  renderOverride?: (input: FrameInput) => FrameOutput;
}): Record<string, unknown> {
  const onErrorCbs: ((e: EngineError) => void)[] = [];

  const obj: Record<string, unknown> = {
    backendId: 'pt-webgl2',
    state: 'ready' as const,
    capabilities: {
      presentationMode: 'offscreen-texture',
      supportedAnalyticShapes: new Set(),
      supportedEmitterKinds: new Set(),
      supportedPrimitiveKinds: new Set(['mesh']),
      supportedEnvironmentKinds: new Set(),
      supportsIncrementalScene: false,
      supportsAddRemovePrimitive: false,
      supportsAuxBuffers: false,
      accumulates: true,
      maxSamplesPerPixel: Infinity,
      maxBounces: 8,
      causticStrategy: 'none',
      experimentalFeatures: new Set(),
    },
    setScene: vi.fn(),
    renderFrame(input: FrameInput): FrameOutput {
      return options?.renderOverride
        ? options.renderOverride(input)
        : { kind: 'skipped', samplesAccumulated: 0, isConverged: false };
    },
    reset: vi.fn(),
    pause: vi.fn(),
    resume: vi.fn(),
    dispose: vi.fn(),
    onError(cb: (e: EngineError) => void): () => void {
      onErrorCbs.push(cb);
      return () => {
        const idx = onErrorCbs.indexOf(cb);
        if (idx !== -1) onErrorCbs.splice(idx, 1);
      };
    },
    // Test helper — fires an EngineError to all onError subscribers.
    _onErrorCbs: onErrorCbs,
    _fireError(e: EngineError): void {
      for (const cb of [...onErrorCbs]) {
        try { cb(e); } catch { /* subscriber errors must not stop other subscribers — ignore */ }
      }
    },
  };

  if (options?.withGIState) {
    const snapshot = options.giSnapshot ?? null;
    obj.exportGIState = vi.fn(async (): Promise<GIStateSnapshot | null> => snapshot);
    obj.importGIState = vi.fn((_s: GIStateSnapshot): boolean => true);
  }

  return obj;
}

// Typed accessors for test assertions — avoid re-casting at every call site.
function asDispose(eng: Record<string, unknown>) {
  return eng.dispose as ReturnType<typeof vi.fn>;
}
function asExportGIState(eng: Record<string, unknown>) {
  return eng.exportGIState as (ReturnType<typeof vi.fn>) | undefined;
}
function asImportGIState(eng: Record<string, unknown>) {
  return eng.importGIState as (ReturnType<typeof vi.fn>) | undefined;
}
function fireError(eng: Record<string, unknown>, e: EngineError) {
  (eng._fireError as (e: EngineError) => void)(e);
}

/** Create a fake EngineError for testing. */
function makeLossError(
  kind: EngineError['kind'] = 'device-lost',
): EngineError {
  return { kind, message: `test ${kind}`, fatal: true };
}

// ────────────────────────────────────────────────────────────────────────────
// DOM setup (mirrors attachVitrumLoop.test.ts pattern)
// ────────────────────────────────────────────────────────────────────────────

describe('attachVitrum — autoRecreateOnDeviceLoss', () => {
  let happyWindow: Window;
  let savedWindow: unknown;
  let savedDocument: unknown;
  let savedRequestAnimationFrame: unknown;
  let savedCancelAnimationFrame: unknown;
  let savedResizeObserver: unknown;

  beforeEach(() => {
    happyWindow = new Window({ url: 'http://localhost/' });
    savedWindow = (globalThis as Record<string, unknown>).window;
    savedDocument = (globalThis as Record<string, unknown>).document;
    savedRequestAnimationFrame = (globalThis as Record<string, unknown>).requestAnimationFrame;
    savedCancelAnimationFrame = (globalThis as Record<string, unknown>).cancelAnimationFrame;
    savedResizeObserver = (globalThis as Record<string, unknown>).ResizeObserver;

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

  // ── Shared helpers ──────────────────────────────────────────────────────

  function makeCanvas(): HTMLCanvasElement {
    return happyWindow.document.createElement('canvas') as unknown as HTMLCanvasElement;
  }

  async function makeCamera() {
    const { asMat4 } = await import('@vitrum/core');
    const identity = asMat4(new Float32Array([1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]));
    return {
      updateMatrixWorld: vi.fn(),
      matrixWorldInverse: { elements: identity },
      projectionMatrix: { elements: identity },
      position: { x: 0, y: 0, z: 0 },
    };
  }

  const scene = {
    primitives: [],
    emitters: [],
    environment: { kind: 'none' as const },
    version: 1,
  };

  // ── Test: fatal device-lost triggers dispose + factory + loop resume ────

  it('fatal device-lost: dispose called, factory called again, loop resumes', async () => {
    const { attachVitrum } = await import('../src/lifecycle/vanilla.js');
    const createEngineModule = await import('../src/createEngine.js');

    const engine1 = makeMockEngine();
    const engine2 = makeMockEngine();
    let callCount = 0;
    vi.spyOn(createEngineModule, 'createEngine').mockImplementation(async () => {
      callCount++;
      return callCount === 1
        ? (engine1 as never)
        : (engine2 as never);
    });

    const canvas = makeCanvas();
    const camera = await makeCamera();
    const errorsSeen: EngineError[] = [];

    const handle = await attachVitrum({
      canvas,
      scene,
      camera,
      autoRecreateOnDeviceLoss: true,
      onEngineError(err) { errorsSeen.push(err); },
    });

    // Verify we got engine1 initially.
    expect(handle.engine).toBe(engine1);

    // Fire a fatal device-lost on the first engine.
    fireError(engine1, makeLossError('device-lost'));

    // Allow the async recreate microtasks to settle.
    await new Promise<void>((resolve) => setTimeout(resolve, 20));

    // createEngine should have been called twice.
    expect(createEngineModule.createEngine).toHaveBeenCalledTimes(2);

    // First engine should have been disposed.
    expect(asDispose(engine1)).toHaveBeenCalledTimes(1);

    // Handle should now expose engine2.
    expect(handle.engine).toBe(engine2);

    // Host received the error before the recreate.
    expect(errorsSeen).toHaveLength(1);
    expect(errorsSeen[0]!.kind).toBe('device-lost');

    handle.dispose();
  });

  // ── Test: context-lost (WebGL) also triggers recreate ──────────────────

  it('fatal context-lost: triggers recreate (WebGL path)', async () => {
    const { attachVitrum } = await import('../src/lifecycle/vanilla.js');
    const createEngineModule = await import('../src/createEngine.js');

    const engine1 = makeMockEngine();
    const engine2 = makeMockEngine();
    let callCount = 0;
    vi.spyOn(createEngineModule, 'createEngine').mockImplementation(async () => {
      callCount++;
      return callCount === 1
        ? (engine1 as never)
        : (engine2 as never);
    });

    const handle = await attachVitrum({
      canvas: makeCanvas(),
      scene,
      camera: await makeCamera(),
      autoRecreateOnDeviceLoss: true,
    });

    fireError(engine1, makeLossError('context-lost'));
    await new Promise<void>((resolve) => setTimeout(resolve, 20));

    expect(asDispose(engine1)).toHaveBeenCalledTimes(1);
    expect(handle.engine).toBe(engine2);

    handle.dispose();
  });

  // ── Test: non-fatal errors do NOT trigger recreate ──────────────────────

  it('non-fatal errors do NOT trigger recreate', async () => {
    const { attachVitrum } = await import('../src/lifecycle/vanilla.js');
    const createEngineModule = await import('../src/createEngine.js');

    const engine1 = makeMockEngine();
    vi.spyOn(createEngineModule, 'createEngine').mockResolvedValue(
      engine1 as never,
    );

    const handle = await attachVitrum({
      canvas: makeCanvas(),
      scene,
      camera: await makeCamera(),
      autoRecreateOnDeviceLoss: true,
    });

    // Non-fatal validation error.
    fireError(engine1, { kind: 'gpu-validation', message: 'shader error', fatal: false });
    await new Promise<void>((resolve) => setTimeout(resolve, 20));

    // createEngine called exactly once (initial construction only).
    expect(createEngineModule.createEngine).toHaveBeenCalledTimes(1);
    expect(asDispose(engine1)).not.toHaveBeenCalled();
    expect(handle.engine).toBe(engine1);

    handle.dispose();
  });

  // ── Test: autoRecreateOnDeviceLoss: false — loss does NOT trigger recreate

  it('when autoRecreateOnDeviceLoss is false, fatal loss does NOT trigger recreate', async () => {
    const { attachVitrum } = await import('../src/lifecycle/vanilla.js');
    const createEngineModule = await import('../src/createEngine.js');

    const engine1 = makeMockEngine();
    vi.spyOn(createEngineModule, 'createEngine').mockResolvedValue(
      engine1 as never,
    );

    const errorsSeen: EngineError[] = [];
    const handle = await attachVitrum({
      canvas: makeCanvas(),
      scene,
      camera: await makeCamera(),
      autoRecreateOnDeviceLoss: false,
      onEngineError(err) { errorsSeen.push(err); },
    });

    fireError(engine1, makeLossError('device-lost'));
    await new Promise<void>((resolve) => setTimeout(resolve, 20));

    // Still only one createEngine call.
    expect(createEngineModule.createEngine).toHaveBeenCalledTimes(1);
    // Engine was NOT disposed by the recreate path.
    expect(asDispose(engine1)).not.toHaveBeenCalled();
    // But onEngineError still fired.
    expect(errorsSeen).toHaveLength(1);

    handle.dispose();
  });

  // ── Test: retry cap ─────────────────────────────────────────────────────

  it('retry cap: stops recreating after AUTO_RECREATE_MAX_ATTEMPTS within the window', async () => {
    const { attachVitrum } = await import('../src/lifecycle/vanilla.js');
    const createEngineModule = await import('../src/createEngine.js');

    // We need AUTO_RECREATE_MAX_ATTEMPTS + 1 engines (1 initial + one per recreate attempt).
    const engines = Array.from({ length: AUTO_RECREATE_MAX_ATTEMPTS + 2 }, () => makeMockEngine());
    let callIdx = 0;
    vi.spyOn(createEngineModule, 'createEngine').mockImplementation(async () => {
      const eng = engines[callIdx++]!;
      return eng as never;
    });

    const handle = await attachVitrum({
      canvas: makeCanvas(),
      scene,
      camera: await makeCamera(),
      autoRecreateOnDeviceLoss: true,
    });

    // Fire AUTO_RECREATE_MAX_ATTEMPTS + 1 fatal errors (one more than the cap).
    for (let i = 0; i <= AUTO_RECREATE_MAX_ATTEMPTS; i++) {
      const currentEngine = engines[i]!;
      fireError(currentEngine, makeLossError('device-lost'));
      // Wait for the recreate to settle before firing the next error.
      await new Promise<void>((resolve) => setTimeout(resolve, 20));
    }

    // The factory should have been called exactly 1 (initial) + AUTO_RECREATE_MAX_ATTEMPTS
    // (allowed recreates) times — the (MAX+1)th attempt is blocked by the cap.
    expect(createEngineModule.createEngine).toHaveBeenCalledTimes(
      1 + AUTO_RECREATE_MAX_ATTEMPTS,
    );

    handle.dispose();
  });

  // ── Test: GI export/import called when surface is present ───────────────

  it('GI export/import called when the engine exposes the GI-state surface', async () => {
    const { attachVitrum } = await import('../src/lifecycle/vanilla.js');
    const createEngineModule = await import('../src/createEngine.js');

    const fakeSnapshot = {
      dims: { x: 2, y: 2, z: 2 },
      origin: [0, 0, 0] as [number, number, number],
      spacing: 1,
      irrW: 4, irrH: 4,
      visW: 4, visH: 4,
      irrData: new Uint16Array(4 * 4 * 4),
      visData: new Uint16Array(4 * 4 * 4),
    } as GIStateSnapshot;

    const engine1 = makeMockEngine({ withGIState: true, giSnapshot: fakeSnapshot });
    const engine2 = makeMockEngine({ withGIState: true, giSnapshot: null });
    let callCount = 0;
    vi.spyOn(createEngineModule, 'createEngine').mockImplementation(async () => {
      callCount++;
      return callCount === 1
        ? (engine1 as never)
        : (engine2 as never);
    });

    const handle = await attachVitrum({
      canvas: makeCanvas(),
      scene,
      camera: await makeCamera(),
      autoRecreateOnDeviceLoss: true,
    });

    expect(handle.engine).toBe(engine1);

    // Fire a fatal device-lost.
    fireError(engine1, makeLossError('device-lost'));
    await new Promise<void>((resolve) => setTimeout(resolve, 20));

    // exportGIState should have been called on engine1 before dispose.
    expect(asExportGIState(engine1)).toHaveBeenCalledTimes(1);

    // importGIState should have been called on engine2 with the snapshot.
    expect(asImportGIState(engine2)).toHaveBeenCalledTimes(1);
    expect(asImportGIState(engine2)).toHaveBeenCalledWith(fakeSnapshot);

    // Handle now exposes engine2.
    expect(handle.engine).toBe(engine2);

    handle.dispose();
  });

  it('reports GI export failure while still recreating the engine', async () => {
    const { attachVitrum } = await import('../src/lifecycle/vanilla.js');
    const createEngineModule = await import('../src/createEngine.js');

    const exportError = new Error('export failed');
    const engine1 = makeMockEngine({ withGIState: true, giSnapshot: null });
    const engine2 = makeMockEngine({ withGIState: true, giSnapshot: null });
    engine1.exportGIState = vi.fn(async () => {
      throw exportError;
    });
    let callCount = 0;
    vi.spyOn(createEngineModule, 'createEngine').mockImplementation(async () => {
      callCount++;
      return callCount === 1
        ? (engine1 as never)
        : (engine2 as never);
    });

    const errorsSeen: Array<{ error: unknown; event: { phase: string; recoverable: boolean; backend?: string } }> = [];
    const handle = await attachVitrum({
      canvas: makeCanvas(),
      scene,
      camera: await makeCamera(),
      autoRecreateOnDeviceLoss: true,
      onError(error, event) {
        errorsSeen.push({ error, event });
      },
    });

    fireError(engine1, makeLossError('device-lost'));
    await new Promise<void>((resolve) => setTimeout(resolve, 20));

    expect(handle.engine).toBe(engine2);
    expect(errorsSeen).toEqual([
      {
        error: exportError,
        event: { phase: 'attach:gi-export', backend: 'pt-webgl2', recoverable: true },
      },
    ]);
    expect(asDispose(engine1)).toHaveBeenCalledTimes(1);

    handle.dispose();
  });

  it('reports GI import failure while keeping the recreated engine active', async () => {
    const { attachVitrum } = await import('../src/lifecycle/vanilla.js');
    const createEngineModule = await import('../src/createEngine.js');

    const fakeSnapshot = {
      dims: { x: 2, y: 2, z: 2 },
      origin: [0, 0, 0] as [number, number, number],
      spacing: 1,
      irrW: 4, irrH: 4,
      visW: 4, visH: 4,
      irrData: new Uint16Array(4 * 4 * 4),
      visData: new Uint16Array(4 * 4 * 4),
    } as GIStateSnapshot;
    const importError = new Error('import failed');
    const engine1 = makeMockEngine({ withGIState: true, giSnapshot: fakeSnapshot });
    const engine2 = makeMockEngine({ withGIState: true, giSnapshot: null });
    engine2.importGIState = vi.fn((_s: GIStateSnapshot): boolean => {
      throw importError;
    });
    let callCount = 0;
    vi.spyOn(createEngineModule, 'createEngine').mockImplementation(async () => {
      callCount++;
      return callCount === 1
        ? (engine1 as never)
        : (engine2 as never);
    });

    const errorsSeen: Array<{ error: unknown; event: { phase: string; recoverable: boolean; backend?: string } }> = [];
    const handle = await attachVitrum({
      canvas: makeCanvas(),
      scene,
      camera: await makeCamera(),
      autoRecreateOnDeviceLoss: true,
      onError(error, event) {
        errorsSeen.push({ error, event });
      },
    });

    fireError(engine1, makeLossError('device-lost'));
    await new Promise<void>((resolve) => setTimeout(resolve, 20));

    expect(handle.engine).toBe(engine2);
    expect(asImportGIState(engine2)).toHaveBeenCalledWith(fakeSnapshot);
    expect(errorsSeen).toEqual([
      {
        error: importError,
        event: { phase: 'attach:gi-import', backend: 'pt-webgl2', recoverable: true },
      },
    ]);

    handle.dispose();
  });

  // ── Test: GI export absent — recreate proceeds without import ───────────

  it('GI export NOT called when engine does not expose exportGIState', async () => {
    const { attachVitrum } = await import('../src/lifecycle/vanilla.js');
    const createEngineModule = await import('../src/createEngine.js');

    // engine1 has NO GI surface; engine2 also lacks it.
    const engine1 = makeMockEngine({ withGIState: false });
    const engine2 = makeMockEngine({ withGIState: false });
    let callCount = 0;
    vi.spyOn(createEngineModule, 'createEngine').mockImplementation(async () => {
      callCount++;
      return callCount === 1
        ? (engine1 as never)
        : (engine2 as never);
    });

    const handle = await attachVitrum({
      canvas: makeCanvas(),
      scene,
      camera: await makeCamera(),
      autoRecreateOnDeviceLoss: true,
    });

    fireError(engine1, makeLossError('device-lost'));
    await new Promise<void>((resolve) => setTimeout(resolve, 20));

    // Recreate happened — engine2 is now active.
    expect(handle.engine).toBe(engine2);
    // No GI methods exist — nothing to assert on; test passes if no throw.
    expect(createEngineModule.createEngine).toHaveBeenCalledTimes(2);

    handle.dispose();
  });

  // ── Test: handle.dispose() during recreate stops the loop cleanly ───────

  it('handle.dispose() during recreate stops the loop without restarting it', async () => {
    const { attachVitrum } = await import('../src/lifecycle/vanilla.js');
    const createEngineModule = await import('../src/createEngine.js');

    const engine1 = makeMockEngine();
    let createEngineCallCount = 0;
    let resolveRecreate!: () => void;
    const recreateStarted = new Promise<void>((res) => { resolveRecreate = res; });

    vi.spyOn(createEngineModule, 'createEngine').mockImplementation((async () => {
      createEngineCallCount++;
      if (createEngineCallCount === 1) {
        // Initial construction — return immediately.
        return engine1;
      }
      // Second call (recreate) — signal that recreate started, then hang briefly.
      resolveRecreate();
      await new Promise<void>((res) => setTimeout(res, 50));
      return engine1;
    }) as never);

    const handle = await attachVitrum({
      canvas: makeCanvas(),
      scene,
      camera: await makeCamera(),
      autoRecreateOnDeviceLoss: true,
    });

    // Trigger recreate.
    fireError(engine1, makeLossError('device-lost'));
    // Wait until the recreate async path has entered createEngine.
    await recreateStarted;

    // Dispose the handle while recreate is in flight.
    handle.dispose();

    // Let the recreate finish.
    await new Promise<void>((resolve) => setTimeout(resolve, 80));

    // After the recreate resolves, the loop should stay stopped (disposed=true).
    // createEngine was called twice (initial + one recreate).
    expect(createEngineCallCount).toBe(2);
  });

  it('reports auto-recreate construction failure as non-recoverable', async () => {
    const { attachVitrum } = await import('../src/lifecycle/vanilla.js');
    const createEngineModule = await import('../src/createEngine.js');

    const engine1 = makeMockEngine();
    const recreateError = new Error('replacement create failed');
    vi.spyOn(createEngineModule, 'createEngine')
      .mockResolvedValueOnce(engine1 as never)
      .mockRejectedValueOnce(recreateError);

    const errorsSeen: Array<{ error: unknown; event: { phase: string; recoverable: boolean; backend?: string } }> = [];
    const handle = await attachVitrum({
      canvas: makeCanvas(),
      scene,
      camera: await makeCamera(),
      autoRecreateOnDeviceLoss: true,
      onError(error, event) {
        errorsSeen.push({ error, event });
      },
    });

    fireError(engine1, makeLossError('device-lost'));
    await new Promise<void>((resolve) => setTimeout(resolve, 20));

    expect(createEngineModule.createEngine).toHaveBeenCalledTimes(2);
    expect(asDispose(engine1)).toHaveBeenCalledTimes(1);
    expect(handle.engine).toBe(engine1);
    expect(errorsSeen).toEqual([
      {
        error: recreateError,
        event: { phase: 'attach:auto-recreate', recoverable: false },
      },
    ]);

    handle.dispose();
  });
});

// ── Exported constants ─────────────────────────────────────────────────────

describe('AUTO_RECREATE constants', () => {
  it('AUTO_RECREATE_MAX_ATTEMPTS is exported and positive', () => {
    expect(AUTO_RECREATE_MAX_ATTEMPTS).toBeGreaterThan(0);
  });

  it('AUTO_RECREATE_WINDOW_MS is exported and positive', () => {
    expect(AUTO_RECREATE_WINDOW_MS).toBeGreaterThan(0);
  });
});
