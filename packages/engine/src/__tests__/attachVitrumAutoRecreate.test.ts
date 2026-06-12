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

function makeEngine() {
  const errorCallbacks: Array<(err: EngineError) => void> = [];
  const setScene = vi.fn();
  const engine = {
    state: 'ready',
    capabilities: {},
    setScene,
    renderFrame: vi.fn(),
    reset: vi.fn(),
    pause: vi.fn(),
    resume: vi.fn(),
    dispose: vi.fn(),
    onError: vi.fn((cb: (err: EngineError) => void) => {
      errorCallbacks.push(cb);
      return vi.fn();
    }),
  } as unknown as Engine;
  return { engine, errorCallbacks, setScene };
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
});
