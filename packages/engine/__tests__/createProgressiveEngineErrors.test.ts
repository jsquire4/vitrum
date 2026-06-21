import { afterEach, describe, expect, it, vi } from 'vitest';
import type { Engine, EngineCapabilities, FrameOutput, Scene } from '@vitrum/core';
import type { CreateEngineErrorEvent, CreateEngineOptions } from '../src/createEngine.js';

const constructorMocks = vi.hoisted(() => ({
  constructWalkaround: vi.fn(),
  constructPathTracerWebGPU: vi.fn(),
}));

vi.mock('../src/createEngine.js', () => ({
  constructWalkaround: constructorMocks.constructWalkaround,
  constructPathTracerWebGPU: constructorMocks.constructPathTracerWebGPU,
}));

import { createProgressiveEngine } from '../src/createProgressiveEngine.js';

const originalNavigator = globalThis.navigator;

afterEach(() => {
  constructorMocks.constructWalkaround.mockReset();
  constructorMocks.constructPathTracerWebGPU.mockReset();
  Object.defineProperty(globalThis, 'navigator', {
    value: originalNavigator,
    configurable: true,
  });
});

const SKIPPED: FrameOutput = {
  kind: 'skipped',
  samplesAccumulated: 0,
  isConverged: false,
};

const SCENE: Scene = {
  primitives: [
    {
      kind: 'mesh',
      id: 'tri',
      positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
      normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
      material: { baseColor: [1, 1, 1], roughness: 0.5, metallic: 0 },
    },
  ],
  emitters: [],
  environment: { kind: 'none' },
};

function makeCapabilities(
  caps: Partial<EngineCapabilities>,
): EngineCapabilities {
  return {
    supportsIncrementalScene: false,
    supportsAddRemovePrimitive: false,
    supportsAuxBuffers: false,
    accumulates: false,
    supportsProgressiveSeedSource: false,
    supportsAccumulatorSeed: false,
    maxSamplesPerPixel: 1,
    maxBounces: 1,
    supportedAnalyticShapes: new Set(),
    supportedEmitterKinds: new Set(),
    presentationMode: 'offscreen-texture',
    causticStrategy: 'none',
    ...caps,
  };
}

function makeEngine(caps: Partial<EngineCapabilities>): Engine {
  return {
    state: 'ready',
    capabilities: makeCapabilities(caps),
    setScene: vi.fn(),
    renderFrame: vi.fn(() => SKIPPED),
    reset: vi.fn(),
    pause: vi.fn(),
    resume: vi.fn(),
    dispose: vi.fn(),
  };
}

function makeCanvas(configure?: () => void): HTMLCanvasElement {
  return {
    getContext: (kind: string) => (
      kind === 'webgpu'
        ? { configure: configure ?? vi.fn() }
        : null
    ),
  } as unknown as HTMLCanvasElement;
}

function installWebGpu(): GPUDevice {
  const fakeDevice = { destroy: vi.fn() } as unknown as GPUDevice;
  const fakeAdapter = {
    limits: {
      maxStorageBuffersPerShaderStage: 128,
      maxStorageTexturesPerShaderStage: 128,
    },
    requestDevice: vi.fn(async () => fakeDevice),
  };
  Object.defineProperty(globalThis, 'navigator', {
    value: {
      gpu: {
        requestAdapter: vi.fn(async () => fakeAdapter),
        getPreferredCanvasFormat: () => 'bgra8unorm' as GPUTextureFormat,
      },
    },
    configurable: true,
  });
  return fakeDevice;
}

function installSuccessfulSubEngines(): void {
  constructorMocks.constructWalkaround.mockResolvedValue(
    makeEngine({ supportsProgressiveSeedSource: true }),
  );
  constructorMocks.constructPathTracerWebGPU.mockResolvedValue(
    makeEngine({ supportsAccumulatorSeed: true }),
  );
}

describe('createProgressiveEngine error callbacks', () => {
  it('forwards sub-engine construction error events without dropping phase/backend metadata', async () => {
    installWebGpu();
    const reported = new Error('sub-build');
    const event: CreateEngineErrorEvent = {
      phase: 'create:walkaround-hybrid',
      backend: 'walkaround-hybrid',
      recoverable: true,
    };
    constructorMocks.constructWalkaround.mockImplementation(
      async (opts: CreateEngineOptions) => {
        opts.onError?.(reported, event);
        return makeEngine({ supportsProgressiveSeedSource: true });
      },
    );
    constructorMocks.constructPathTracerWebGPU.mockResolvedValue(
      makeEngine({ supportsAccumulatorSeed: true }),
    );
    const onError = vi.fn();

    await createProgressiveEngine({
      canvas: makeCanvas(),
      scene: SCENE,
      onError,
    });

    expect(onError).toHaveBeenCalledWith(reported, event);
  });

  it('reports the facade canvas-configure failure with the canonical event shape', async () => {
    installWebGpu();
    installSuccessfulSubEngines();
    const configureError = new Error('configure failed');
    const onError = vi.fn();

    await createProgressiveEngine({
      canvas: makeCanvas(() => { throw configureError; }),
      scene: SCENE,
      onError,
    });

    expect(onError).toHaveBeenCalledWith(configureError, {
      phase: 'canvas-configure',
      backend: 'walkaround-hybrid',
      recoverable: true,
    });
  });

  it('guards throwing adapter-profile callbacks during progressive construction', async () => {
    const device = installWebGpu();
    installSuccessfulSubEngines();
    const onAdapterProfile = vi.fn(() => {
      throw new Error('host profile callback failed');
    });

    const handle = await createProgressiveEngine({
      canvas: makeCanvas(),
      scene: SCENE,
      onAdapterProfile,
    });

    expect(onAdapterProfile).toHaveBeenCalledTimes(1);
    expect(constructorMocks.constructWalkaround).toHaveBeenCalledTimes(1);
    expect(constructorMocks.constructPathTracerWebGPU).toHaveBeenCalledTimes(1);
    expect(device.destroy).not.toHaveBeenCalled();
    handle.dispose();
    expect(device.destroy).toHaveBeenCalledTimes(1);
  });
});
