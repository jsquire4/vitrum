import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { Engine, EngineError, Scene } from '@vitrum/core';

const createEngineMock = vi.hoisted(() => vi.fn());

vi.mock('../createEngine.js', () => ({
  createEngine: createEngineMock,
}));

import { attachVitrum } from '../lifecycle/vanilla.js';

const sceneA: Scene = { primitives: [], emitters: [], environment: { kind: 'none' } };
const sceneB: Scene = {
  primitives: [],
  emitters: [],
  environment: { kind: 'none' },
};
const sceneC: Scene = {
  primitives: [],
  emitters: [],
  environment: { kind: 'none' },
};

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

function makeEngine(backendId = 'pt-webgl2', retainedScene: Scene | null = null) {
  const errorCallbacks: Array<(err: EngineError) => void> = [];
  const setScene = vi.fn();
  const getScene = vi.fn(() => retainedScene);
  const engine = {
    backendId,
    state: 'ready',
    capabilities: {},
    setScene,
    renderFrame: vi.fn(),
    reset: vi.fn(),
    pause: vi.fn(),
    resume: vi.fn(),
    dispose: vi.fn(),
    getScene,
    onError: vi.fn((cb: (err: EngineError) => void) => {
      errorCallbacks.push(cb);
      return vi.fn();
    }),
  } as unknown as Engine;
  return { engine, errorCallbacks, setScene, getScene };
}

describe('attachVitrum auto-recreate scene tracking', () => {
  const originalRequestAnimationFrame = globalThis.requestAnimationFrame;
  const originalCancelAnimationFrame = globalThis.cancelAnimationFrame;

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
      value: originalRequestAnimationFrame,
      configurable: true,
    });
    Object.defineProperty(globalThis, 'cancelAnimationFrame', {
      value: originalCancelAnimationFrame,
      configurable: true,
    });
  });

  it('recreates with the latest scene set through the exposed engine handle', async () => {
    const first = makeEngine();
    const second = makeEngine();
    createEngineMock
      .mockResolvedValueOnce(first.engine)
      .mockResolvedValueOnce(second.engine);

    const handle = await attachVitrum({
      canvas: makeCanvas(),
      scene: sceneA,
      camera: makeCamera(),
      autoRecreateOnDeviceLoss: true,
    });

    handle.engine.setScene(sceneB);
    expect(first.setScene).toHaveBeenCalledWith(sceneB);

    first.errorCallbacks[0]!({
      kind: 'device-lost',
      fatal: true,
      message: 'lost',
    } as EngineError);

    await vi.waitFor(() => expect(createEngineMock).toHaveBeenCalledTimes(2));
    expect(createEngineMock.mock.calls[1]![0].scene).toBe(sceneB);

    handle.dispose();
  });

  it('recreates with the backend-retained live scene when fast paths bypass lifecycle setScene tracking', async () => {
    const first = makeEngine('pt-webgl2', sceneC);
    const second = makeEngine();
    createEngineMock
      .mockResolvedValueOnce(first.engine)
      .mockResolvedValueOnce(second.engine);

    const handle = await attachVitrum({
      canvas: makeCanvas(),
      scene: sceneA,
      camera: makeCamera(),
      autoRecreateOnDeviceLoss: true,
    });

    first.errorCallbacks[0]!({
      kind: 'device-lost',
      fatal: true,
      message: 'lost',
    } as EngineError);

    await vi.waitFor(() => expect(createEngineMock).toHaveBeenCalledTimes(2));
    expect(createEngineMock.mock.calls[1]![0].scene).toBe(sceneC);

    handle.dispose();
  });

  it('warns and falls back to the tracked scene when backend scene snapshot throws', async () => {
    const first = makeEngine('pt-webgl2');
    const second = makeEngine();
    const snapshotError = new Error('snapshot failed');
    const onError = vi.fn();
    createEngineMock
      .mockResolvedValueOnce(first.engine)
      .mockResolvedValueOnce(second.engine);

    const handle = await attachVitrum({
      canvas: makeCanvas(),
      scene: sceneA,
      camera: makeCamera(),
      autoRecreateOnDeviceLoss: true,
      onError,
    });

    handle.engine.setScene(sceneB);
    first.getScene.mockImplementationOnce(() => {
      throw snapshotError;
    });

    first.errorCallbacks[0]!({
      kind: 'device-lost',
      fatal: true,
      message: 'lost',
    } as EngineError);

    await vi.waitFor(() => expect(createEngineMock).toHaveBeenCalledTimes(2));
    expect(onError).toHaveBeenCalledWith(snapshotError, {
      phase: 'attach:auto-recreate',
      backend: 'pt-webgl2',
      recoverable: true,
    });
    expect(createEngineMock.mock.calls[1]![0].scene).toBe(sceneB);

    handle.dispose();
  });

  it('exposes the selected backend id through the stable attach handle', async () => {
    const first = makeEngine('pt-webgl2');
    const second = makeEngine('walkaround-hybrid');
    createEngineMock
      .mockResolvedValueOnce(first.engine)
      .mockResolvedValueOnce(second.engine);

    const handle = await attachVitrum({
      canvas: makeCanvas(),
      scene: sceneA,
      camera: makeCamera(),
      autoRecreateOnDeviceLoss: true,
    });

    expect(handle.backendId).toBe('pt-webgl2');

    first.errorCallbacks[0]!({
      kind: 'device-lost',
      fatal: true,
      message: 'lost',
    } as EngineError);

    await vi.waitFor(() => expect(handle.backendId).toBe('walkaround-hybrid'));

    handle.dispose();
  });

  it('refreshes WebGPU swapchain plumbing after auto-recreate changes backend class', async () => {
    let rafCallback: FrameRequestCallback | undefined;
    Object.defineProperty(globalThis, 'requestAnimationFrame', {
      value: vi.fn((cb: FrameRequestCallback) => {
        rafCallback = cb;
        return 1;
      }),
      configurable: true,
    });

    const swapChainView = { tag: 'view-after-recreate' } as unknown as GPUTextureView;
    const webgpuContext = {
      getConfiguration: vi.fn(() => ({ format: 'rgba8unorm' as GPUTextureFormat })),
      getCurrentTexture: vi.fn(() => ({
        createView: vi.fn(() => swapChainView),
      })),
    } as unknown as GPUCanvasContext;
    let exposeWebGpuContext = false;
    const canvas = makeCanvas();
    vi.mocked(canvas.getContext).mockImplementation((kind: string) =>
      kind === 'webgpu' && exposeWebGpuContext ? webgpuContext : null,
    );

    const first = makeEngine('pt-webgl2');
    const second = makeEngine('walkaround-hybrid');
    createEngineMock
      .mockResolvedValueOnce(first.engine)
      .mockResolvedValueOnce(second.engine);

    const handle = await attachVitrum({
      canvas,
      scene: sceneA,
      camera: makeCamera(),
      autoRecreateOnDeviceLoss: true,
    });

    exposeWebGpuContext = true;
    first.errorCallbacks[0]!({
      kind: 'device-lost',
      fatal: true,
      message: 'lost',
    } as EngineError);

    await vi.waitFor(() => expect(handle.backendId).toBe('walkaround-hybrid'));

    rafCallback?.(123);

    expect(second.engine.renderFrame).toHaveBeenCalledTimes(1);
    expect(second.engine.renderFrame).toHaveBeenCalledWith(expect.objectContaining({
      swapChainView,
      swapChainFormat: 'rgba8unorm',
    }));

    handle.dispose();
  });
});
