import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { Engine, FrameInput, Scene } from '@vitrum/core';

const constructWalkaroundMock = vi.hoisted(() => vi.fn());
const constructPathTracerWebGPUMock = vi.hoisted(() => vi.fn());

vi.mock('../createEngine.js', () => ({
  constructWalkaround: constructWalkaroundMock,
  constructPathTracerWebGPU: constructPathTracerWebGPUMock,
}));

import { createProgressiveEngine } from '../createProgressiveEngine.js';

const scene: Scene = { primitives: [], emitters: [], environment: { kind: 'none' } };
const originalNavigator = globalThis.navigator;

function makeEngine(capability: 'seed-source' | 'seed-sink'): Engine {
  return {
    state: 'ready',
    capabilities: {
      ...(capability === 'seed-source' ? { supportsProgressiveSeedSource: true } : {}),
      ...(capability === 'seed-sink' ? { supportsAccumulatorSeed: true } : {}),
    },
    setScene: vi.fn(),
    renderFrame: vi.fn((_input: FrameInput) => ({
      kind: 'skipped',
      samplesAccumulated: 0,
      isConverged: false,
    })),
    reset: vi.fn(),
    pause: vi.fn(),
    resume: vi.fn(),
    dispose: vi.fn(),
  } as unknown as Engine;
}

function makeDevice(): GPUDevice {
  return {
    destroy: vi.fn(),
    limits: {
      maxStorageBuffersPerShaderStage: 64,
      maxStorageTexturesPerShaderStage: 16,
    },
  } as unknown as GPUDevice;
}

function makeAdapter(device: GPUDevice): GPUAdapter {
  return {
    limits: {
      maxStorageBuffersPerShaderStage: 64,
      maxStorageTexturesPerShaderStage: 16,
    },
    requestDevice: vi.fn(async () => device),
  } as unknown as GPUAdapter;
}

function makeThrowingConfigureCanvas(error: Error): HTMLCanvasElement {
  return {
    getContext: vi.fn((kind: string) => (
      kind === 'webgpu'
        ? { configure: vi.fn(() => { throw error; }) }
        : null
    )),
  } as unknown as HTMLCanvasElement;
}

describe('createProgressiveEngine canvas plumbing diagnostics', () => {
  beforeEach(() => {
    constructWalkaroundMock.mockReset();
    constructPathTracerWebGPUMock.mockReset();
  });

  afterEach(() => {
    Object.defineProperty(globalThis, 'navigator', {
      value: originalNavigator,
      configurable: true,
    });
  });

  it('reports final canvas-configure failures through the structured onError callback', async () => {
    const device = makeDevice();
    const adapter = makeAdapter(device);
    Object.defineProperty(globalThis, 'navigator', {
      value: {
        gpu: {
          requestAdapter: vi.fn(async () => adapter),
          getPreferredCanvasFormat: () => 'rgba8unorm',
        },
      },
      configurable: true,
    });
    constructWalkaroundMock.mockResolvedValue(makeEngine('seed-source'));
    constructPathTracerWebGPUMock.mockResolvedValue(makeEngine('seed-sink'));
    const configureError = new Error('configure failed');
    const onError = vi.fn();

    const handle = await createProgressiveEngine({
      canvas: makeThrowingConfigureCanvas(configureError),
      scene,
      onError,
    });

    expect(onError).toHaveBeenCalledWith(configureError, {
      phase: 'canvas-configure',
      backend: 'walkaround-hybrid',
      recoverable: true,
    });

    handle.dispose();
  });
});
