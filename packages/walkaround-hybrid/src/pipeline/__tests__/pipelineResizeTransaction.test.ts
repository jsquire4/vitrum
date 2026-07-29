import { beforeAll, describe, expect, it, vi } from 'vitest';
import type { FrameResources } from '../resourceManager.js';
import type { WalkaroundGPUPipeline } from '../WalkaroundGPUPipeline.js';
import type { HybridEngine } from '../../HybridEngine.js';

type MockResource = {
  readonly label: string;
  readonly destroy: ReturnType<typeof vi.fn>;
};

let createFrameResources: typeof import('../resourceManager.js').createFrameResources;
let Pipeline: typeof import('../WalkaroundGPUPipeline.js').WalkaroundGPUPipeline;
let Engine: typeof import('../../HybridEngine.js').HybridEngine;

beforeAll(async () => {
  Object.assign(globalThis, {
    GPUBufferUsage: { COPY_SRC: 1, COPY_DST: 2, STORAGE: 4, UNIFORM: 8 },
    GPUTextureUsage: {
      COPY_SRC: 1,
      COPY_DST: 2,
      TEXTURE_BINDING: 4,
      STORAGE_BINDING: 8,
    },
  });
  ({ createFrameResources } = await import('../resourceManager.js'));
  ({ WalkaroundGPUPipeline: Pipeline } = await import('../WalkaroundGPUPipeline.js'));
  ({ HybridEngine: Engine } = await import('../../HybridEngine.js'));
});

function makeDevice(): {
  readonly device: GPUDevice;
  readonly created: MockResource[];
} {
  const created: MockResource[] = [];
  const allocate = (descriptor: { label?: string }): MockResource => {
    const resource: MockResource = {
      label: descriptor.label ?? 'unlabelled',
      destroy: vi.fn(),
    };
    created.push(resource);
    return resource;
  };
  return {
    device: {
      createTexture: vi.fn(allocate),
      createBuffer: vi.fn(allocate),
      createSampler: vi.fn(() => ({})),
      queue: { writeTexture: vi.fn(), writeBuffer: vi.fn() },
    } as unknown as GPUDevice,
    created,
  };
}

function makePipeline(
  device: GPUDevice,
  resources: FrameResources,
  onPpgResize: () => void,
  onDenoiserResize: () => void,
): WalkaroundGPUPipeline {
  const pipeline = Object.create(Pipeline.prototype) as WalkaroundGPUPipeline;
  Object.assign(pipeline as unknown as Record<string, unknown>, {
    _initialized: true,
    _width: 16,
    _height: 16,
    _device: device,
    _res: resources,
    _gtaoDownscale: 2,
    _denoiserMode: 'none',
    _ppg: { onResize: vi.fn(onPpgResize) },
    _activeDenoiser: { resize: vi.fn(onDenoiserResize) },
    _frameCount: 7,
    _resourceCache: { clear: vi.fn() },
    _accumPingPongIndex: 1,
    _accumFrameIndex: 9,
    _indirectAccumPingPongRef: { value: 1 },
    _varianceTrackerPingPongRef: { value: 1 },
    _lastCameraPos: [4, 5, 6],
  });
  return pipeline;
}

describe('WalkaroundGPUPipeline resize transaction', () => {
  it('keeps the live frame set when optional PPG preparation fails', () => {
    const { device, created } = makeDevice();
    const live = createFrameResources(device, 16, 16);
    const liveResources = created.splice(0);
    const pipeline = makePipeline(
      device,
      live,
      () => {
        throw new Error('injected PPG resize failure');
      },
      () => undefined,
    );

    expect(() => pipeline.resize(32, 24)).toThrow('injected PPG resize failure');

    const state = pipeline as unknown as Record<string, unknown>;
    expect(state._res).toBe(live);
    expect(state._width).toBe(16);
    expect(state._height).toBe(16);
    for (const resource of liveResources) {
      expect(resource.destroy).not.toHaveBeenCalled();
    }
    expect(created.length).toBeGreaterThan(20);
    for (const resource of created) {
      expect(resource.destroy).toHaveBeenCalledOnce();
    }
  });

  it('publishes the complete replacement before retiring the old frame set', () => {
    const { device, created } = makeDevice();
    const live = createFrameResources(device, 16, 16);
    const liveResources = created.splice(0);
    const pipeline = makePipeline(
      device,
      live,
      () => undefined,
      () => undefined,
    );

    pipeline.resize(32, 24);

    const state = pipeline as unknown as Record<string, unknown>;
    expect(state._res).not.toBe(live);
    expect(state._width).toBe(32);
    expect(state._height).toBe(24);
    for (const resource of liveResources) {
      expect(resource.destroy).toHaveBeenCalledOnce();
    }
    for (const resource of created) {
      expect(resource.destroy).not.toHaveBeenCalled();
    }
  });
});

describe('HybridEngine size transaction', () => {
  it('does not publish dimensions when pipeline replacement fails', () => {
    const engine = Object.create(Engine.prototype) as HybridEngine;
    const resize = vi.fn(() => {
      throw new Error('injected pipeline resize failure');
    });
    Object.assign(engine as unknown as Record<string, unknown>, {
      _state: 'ready',
      _width: 16,
      _height: 16,
      _internalWidth: 8,
      _internalHeight: 8,
      _resolutionFactor: 0.5,
      _pipeline: { resize },
      _materialWarner: { warnInvalidSetSize: vi.fn() },
    });

    expect(() => engine.setSize(32, 24)).toThrow('injected pipeline resize failure');

    const state = engine as unknown as Record<string, unknown>;
    expect(state._width).toBe(16);
    expect(state._height).toBe(16);
    expect(state._internalWidth).toBe(8);
    expect(state._internalHeight).toBe(8);
  });
});
