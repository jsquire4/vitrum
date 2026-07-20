// lifecycleRecreate.test.ts — R1 lifecycle regression pins.
//
// V1-2: a dispose() that arrives while the async auto-recreate is awaiting the
//       replacement engine must NOT install/subscribe that late engine — it
//       must be disposed and the recreate state machine reset.
// V1-3: hitting the auto-recreate retry cap must stop the RAF loop, cancel the
//       pending RAF, unsubscribe telemetry, and dispose the fatal engine
//       (previously the cap branch leaked the engine + left RAF running until
//       the H31-d self-stop threshold).

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { Engine, EngineError, Scene } from '@vitrum/core';

const createEngineMock = vi.hoisted(() => vi.fn());

vi.mock('../createEngine.js', () => ({
  createEngine: createEngineMock,
}));

import { attachVitrum, type AttachVitrumRecreateEngineContext } from '../lifecycle/vanilla.js';
import type { EngineWithBackendId } from '../createEngineInternals.js';

const sceneA: Scene = { primitives: [], emitters: [], environment: { kind: 'none' } };

const identityElements = new Float32Array([
  1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 1, 0,
  0, 0, 0, 1,
]);

function makeCanvas(): HTMLCanvasElement {
  return {
    width: 300,
    height: 150,
    clientWidth: 300,
    clientHeight: 150,
    getContext: vi.fn(() => null),
  } as unknown as HTMLCanvasElement;
}

function makeCamera() {
  return {
    updateMatrixWorld: vi.fn(),
    matrixWorldInverse: { elements: identityElements },
    projectionMatrix: { elements: identityElements },
    position: { x: 0, y: 0, z: 0 },
  };
}

function makeEngine(backendId = 'pt-webgl2', retainedScene: Scene | null = sceneA) {
  const errorCallbacks: Array<(err: EngineError) => void> = [];
  const unsubFrame = vi.fn();
  const unsubProgress = vi.fn();
  const unsubError = vi.fn();
  const engine = {
    backendId,
    state: 'ready',
    capabilities: {},
    setScene: vi.fn(),
    renderFrame: vi.fn(),
    reset: vi.fn(),
    pause: vi.fn(),
    resume: vi.fn(),
    dispose: vi.fn(),
    getScene: vi.fn(() => retainedScene),
    onFrame: vi.fn(() => unsubFrame),
    onProgress: vi.fn(() => unsubProgress),
    onError: vi.fn((cb: (err: EngineError) => void) => {
      errorCallbacks.push(cb);
      return unsubError;
    }),
  } as unknown as Engine;
  return { engine, errorCallbacks, unsubFrame, unsubProgress, unsubError };
}

const LOSS: EngineError = { kind: 'device-lost', fatal: true, message: 'lost' } as EngineError;

describe('attachVitrum lifecycle recreate (R1)', () => {
  const originalRaf = globalThis.requestAnimationFrame;
  const originalCancelRaf = globalThis.cancelAnimationFrame;

  beforeEach(() => {
    createEngineMock.mockReset();
    Object.defineProperty(globalThis, 'requestAnimationFrame', {
      value: vi.fn(() => 1),
      configurable: true,
    });
    Object.defineProperty(globalThis, 'cancelAnimationFrame', {
      value: vi.fn(),
      configurable: true,
    });
  });

  afterEach(() => {
    Object.defineProperty(globalThis, 'requestAnimationFrame', {
      value: originalRaf,
      configurable: true,
    });
    Object.defineProperty(globalThis, 'cancelAnimationFrame', {
      value: originalCancelRaf,
      configurable: true,
    });
  });

  it('V1-2: disposes the late engine when dispose() arrives mid-recreate (does not install/subscribe it)', async () => {
    const first = makeEngine('pt-webgpu');
    const late = makeEngine('pt-webgpu');

    // Gate the recreate so we can dispose the handle while it is awaiting.
    let releaseRecreate!: () => void;
    const recreateGate = new Promise<void>((resolve) => { releaseRecreate = resolve; });
    const recreateEngine = vi.fn(async (_ctx: AttachVitrumRecreateEngineContext) => {
      await recreateGate;
      return late.engine as unknown as EngineWithBackendId;
    });

    const handle = await attachVitrum({
      canvas: makeCanvas(),
      engine: first.engine as never,
      scene: sceneA,
      camera: makeCamera(),
      autoRecreateOnDeviceLoss: true,
      recreateEngine,
    });

    // Trigger the fatal loss → performAutoRecreate starts and awaits the gate.
    first.errorCallbacks[0]!(LOSS);
    await vi.waitFor(() => expect(recreateEngine).toHaveBeenCalledTimes(1));

    // Dispose the handle WHILE the recreate is still awaiting.
    handle.dispose();
    expect(first.engine.dispose).toHaveBeenCalled();

    // Now let the recreate resolve — the late engine must be disposed and
    // never installed as the handle's engine.
    releaseRecreate();
    await vi.waitFor(() => expect(late.engine.dispose).toHaveBeenCalledTimes(1));

    // The late engine must NOT have been subscribed (never installed).
    expect(late.engine.onError).not.toHaveBeenCalled();
    // The stable handle still points at the original (now-disposed) engine.
    expect(handle.engine.renderFrame).toBe(first.engine.renderFrame);
  });

  it('V1-3: hitting the retry cap stops RAF, cancels the pending frame, unsubscribes, and disposes the fatal engine', async () => {
    const first = makeEngine('pt-webgpu');
    const second = makeEngine('pt-webgpu');
    const third = makeEngine('pt-webgpu');
    // Each recreate returns a fresh engine; the loss is re-fired on the newest.
    const recreateEngine = vi.fn()
      .mockResolvedValueOnce(second.engine as unknown as EngineWithBackendId)
      .mockResolvedValueOnce(third.engine as unknown as EngineWithBackendId);

    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

    const handle = await attachVitrum({
      canvas: makeCanvas(),
      engine: first.engine as never,
      scene: sceneA,
      camera: makeCamera(),
      autoRecreateOnDeviceLoss: true,
      recreateEngine,
    });

    // Attempt 1: loss on first → recreate to second.
    first.errorCallbacks[0]!(LOSS);
    await vi.waitFor(() => expect(recreateEngine).toHaveBeenCalledTimes(1));
    await vi.waitFor(() => expect(second.engine.onError).toHaveBeenCalled());

    // Attempt 2: loss on second → recreate to third (fills the 2-attempt budget).
    second.errorCallbacks[0]!(LOSS);
    await vi.waitFor(() => expect(recreateEngine).toHaveBeenCalledTimes(2));
    await vi.waitFor(() => expect(third.engine.onError).toHaveBeenCalled());

    const cancelSpy = vi.mocked(globalThis.cancelAnimationFrame);
    cancelSpy.mockClear();

    // 3rd loss on third → cap exceeded → cap branch teardown.
    third.errorCallbacks[0]!(LOSS);

    // No further recreate attempted.
    expect(recreateEngine).toHaveBeenCalledTimes(2);
    // The fatal (third) engine is disposed.
    expect(third.engine.dispose).toHaveBeenCalledTimes(1);
    // Its telemetry subscriptions were torn down.
    expect(third.unsubError).toHaveBeenCalled();
    // The pending RAF was cancelled.
    expect(cancelSpy).toHaveBeenCalled();

    handle.dispose();
    errorSpy.mockRestore();
  });
});
