import { beforeAll, describe, expect, it, vi } from 'vitest';
import type { FrameResources } from '../resourceManager.js';
import type { WalkaroundGPUPipeline } from '../WalkaroundGPUPipeline.js';
import type { HybridEngine } from '../../HybridEngine.js';
import type { SceneBVHBuffers } from '../../restir/bvhTypes.js';
import { RESERVOIR_DI_STRIDE_BYTES } from '../../restir/reservoirDiLayout.js';
import { RESERVOIR_GI_STRIDE_BYTES } from '../../gi/giLayout.js';
import { SceneMutationFinalizationError } from '../../SceneMutationTransaction.js';

type MockResource = {
  readonly label: string;
  readonly size: number;
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
  const allocate = (
    descriptor: { label?: string; size?: number | GPUExtent3D },
  ): MockResource => {
    const resource: MockResource = {
      label: descriptor.label ?? 'unlabelled',
      size: typeof descriptor.size === 'number' ? descriptor.size : 0,
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
  preparePpgResize: () => {
    commit(): void;
    rollback(): void;
    finalize(): void;
  },
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
    _gtaoEnabled: true,
    _checkerboard: false,
    _reservoirScale: 1,
    _denoiserMode: 'none',
    _ppg: { prepareResize: vi.fn(preparePpgResize) },
    _activeDenoiser: {
      prepareResize: vi.fn(() => {
        onDenoiserResize();
        return {
          commit: vi.fn(),
          rollback: vi.fn(),
          finalize: vi.fn(),
        };
      }),
    },
    _frameCount: 7,
    _resourceCache: { clear: vi.fn() },
    _accumPingPongIndex: 1,
    _accumFrameIndex: 9,
    _grisHistoryEpoch: 3,
    _temporalHistoryClearPending: false,
    _temporalHistoryFullRatePending: false,
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
      () => ({
        commit: vi.fn(),
        rollback: vi.fn(),
        finalize: vi.fn(),
      }),
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

  it('retires every old resource even when cache retirement reports an error', () => {
    const { device, created } = makeDevice();
    const live = createFrameResources(device, 16, 16);
    const liveResources = created.splice(0);
    const ppgMutation = {
      commit: vi.fn(),
      rollback: vi.fn(),
      finalize: vi.fn(),
    };
    const pipeline = makePipeline(
      device,
      live,
      () => ppgMutation,
      () => undefined,
    );
    const state = pipeline as unknown as {
      _res: FrameResources;
      _resourceCache: { clear: ReturnType<typeof vi.fn> };
    };
    state._resourceCache.clear.mockImplementationOnce(() => {
      throw new Error('injected cache retirement failure');
    });

    expect(() => pipeline.resize(32, 24)).toThrow(
      /old-generation retirement failed/,
    );

    expect(state._res).not.toBe(live);
    expect(ppgMutation.finalize).toHaveBeenCalledOnce();
    for (const resource of liveResources) {
      expect(resource.destroy).toHaveBeenCalledOnce();
    }
    for (const resource of created) {
      expect(resource.destroy).not.toHaveBeenCalled();
    }
  });

  it('keeps the old generation live through commit and restores it on rollback', () => {
    const { device, created } = makeDevice();
    const live = createFrameResources(device, 16, 16);
    const liveResources = created.splice(0);
    const pipeline = makePipeline(
      device,
      live,
      () => ({
        commit: vi.fn(),
        rollback: vi.fn(),
        finalize: vi.fn(),
      }),
      () => undefined,
    );

    const mutation = pipeline.prepareResize(32, 24, 2);
    const candidates = [...created];
    const state = pipeline as unknown as Record<string, unknown>;
    expect(state._res).toBe(live);
    expect(state._width).toBe(16);
    expect(state._height).toBe(16);
    expect(state._reservoirScale).toBe(1);

    mutation.commit();
    expect(state._res).toBe(mutation.candidateFrameResources);
    expect(state._width).toBe(32);
    expect(state._height).toBe(24);
    expect(state._reservoirScale).toBe(2);
    expect(state._accumFrameIndex).toBe(0);
    for (const resource of liveResources) {
      expect(resource.destroy).not.toHaveBeenCalled();
    }

    mutation.rollback();
    expect(state._res).toBe(live);
    expect(state._width).toBe(16);
    expect(state._height).toBe(16);
    expect(state._reservoirScale).toBe(1);
    expect(state._accumPingPongIndex).toBe(1);
    expect(state._accumFrameIndex).toBe(9);
    expect(state._grisHistoryEpoch).toBe(3);
    for (const resource of liveResources) {
      expect(resource.destroy).not.toHaveBeenCalled();
    }
    for (const resource of candidates) {
      expect(resource.destroy).toHaveBeenCalledOnce();
    }
  });

  it('rolls the reversible resize back when the final scene submission rejects', () => {
    const events: string[] = [];
    const candidate = { marker: 'candidate' } as unknown as FrameResources;
    const resizeMutation = {
      candidateFrameResources: candidate,
      commit: vi.fn(() => events.push('resize.commit')),
      rollback: vi.fn(() => events.push('resize.rollback')),
      finalize: vi.fn(() => events.push('resize.finalize')),
    };
    const sceneMutation = {
      commit: vi.fn(() => {
        events.push('scene.commit');
        throw new Error('injected queue submission failure');
      }),
      rollback: vi.fn(() => events.push('scene.rollback')),
      finalize: vi.fn(() => events.push('scene.finalize')),
    };
    const pipeline = Object.create(Pipeline.prototype) as WalkaroundGPUPipeline;
    const prepareResize = vi.fn(() => resizeMutation);
    const prepareSceneMutation = vi.fn(() => sceneMutation);
    Object.assign(pipeline as unknown as Record<string, unknown>, {
      prepareResize,
      prepareSceneMutation,
    });

    const mutation = pipeline.prepareSceneMutationAndResize(
      { resetAccumulator: true },
      {} as SceneBVHBuffers,
      [],
      32,
      24,
      2,
    );
    expect(prepareSceneMutation).toHaveBeenCalledWith(
      { resetAccumulator: true },
      {},
      [],
      { frameResources: candidate, width: 32, height: 24 },
    );
    expect(() => mutation.commit()).toThrow('injected queue submission failure');
    expect(events).toEqual([
      'resize.commit',
      'scene.commit',
      'scene.rollback',
      'resize.rollback',
    ]);
    expect(resizeMutation.finalize).not.toHaveBeenCalled();
    expect(sceneMutation.finalize).not.toHaveBeenCalled();
  });

  it('rolls back prepared PPG state when the denoiser rejects the candidate size', () => {
    const { device, created } = makeDevice();
    const live = createFrameResources(device, 16, 16);
    const liveResources = created.splice(0);
    const mutation = {
      commit: vi.fn(),
      rollback: vi.fn(),
      finalize: vi.fn(),
    };
    const pipeline = makePipeline(
      device,
      live,
      () => mutation,
      () => {
        throw new Error('injected denoiser resize failure');
      },
    );

    expect(() => pipeline.resize(32, 24)).toThrow('injected denoiser resize failure');

    const state = pipeline as unknown as Record<string, unknown>;
    expect(state._res).toBe(live);
    expect(state._width).toBe(16);
    expect(state._height).toBe(16);
    expect(mutation.commit).not.toHaveBeenCalled();
    expect(mutation.rollback).toHaveBeenCalledOnce();
    expect(mutation.finalize).not.toHaveBeenCalled();
    for (const resource of liveResources) {
      expect(resource.destroy).not.toHaveBeenCalled();
    }
    for (const resource of created) {
      expect(resource.destroy).toHaveBeenCalledOnce();
    }
  });

  it('transactionally replaces same-size resources when only reservoir scale changes', () => {
    const { device, created } = makeDevice();
    const live = createFrameResources(device, 16, 16);
    const liveResources = created.splice(0);
    const pipeline = makePipeline(
      device,
      live,
      () => ({
        commit: vi.fn(),
        rollback: vi.fn(),
        finalize: vi.fn(),
      }),
      () => undefined,
    );

    pipeline.resize(16, 16, 2);

    const state = pipeline as unknown as {
      _res: FrameResources;
      _width: number;
      _height: number;
      _reservoirScale: number;
    };
    expect(state._res).not.toBe(live);
    expect(state._width).toBe(16);
    expect(state._height).toBe(16);
    expect(state._reservoirScale).toBe(2);
    expect(state._res.restirDI.reservoirCurrentBuffer.size)
      .toBe(8 * 8 * RESERVOIR_DI_STRIDE_BYTES);
    expect(state._res.restirGI.reservoirGiCurrentBuffer.size)
      .toBe(4 * 4 * RESERVOIR_GI_STRIDE_BYTES);
    for (const resource of liveResources) {
      expect(resource.destroy).toHaveBeenCalledOnce();
    }
    for (const resource of created) {
      expect(resource.destroy).not.toHaveBeenCalled();
    }
  });

  it('rejects an invalid scale before allocating a replacement generation', () => {
    const { device, created } = makeDevice();
    const live = createFrameResources(device, 16, 16);
    created.splice(0);
    const pipeline = makePipeline(
      device,
      live,
      () => ({
        commit: vi.fn(),
        rollback: vi.fn(),
        finalize: vi.fn(),
      }),
      () => undefined,
    );

    expect(() => pipeline.resize(32, 24, 5)).toThrow(
      /reservoirScale must be an integer in \[1, 4\]/,
    );
    expect(created).toHaveLength(0);
    const state = pipeline as unknown as Record<string, unknown>;
    expect(state._res).toBe(live);
    expect(state._reservoirScale).toBe(1);
  });
});

describe('HybridEngine size transaction', () => {
  it('does not report a phantom old generation before pipeline allocation', () => {
    const engine = Object.create(Engine.prototype) as HybridEngine;
    const resolveFrameResourceResolution = vi.fn(() => ({
      effectiveWidth: 16,
      effectiveHeight: 12,
      restirReservoirScale: 2,
    }));
    Object.assign(engine as unknown as Record<string, unknown>, {
      _state: 'initializing',
      _width: 16,
      _height: 16,
      _internalWidth: 8,
      _internalHeight: 8,
      _resolutionFactor: 0.5,
      _lastScene: null,
      _frameResourceResolution: { persistentBytes: 123 },
      _resolveFrameResourceResolution: resolveFrameResourceResolution,
      _pipeline: null,
    });

    engine.setSize(32, 24);

    expect(resolveFrameResourceResolution)
      .toHaveBeenCalledWith(16, 12, 0, undefined);
    const state = engine as unknown as Record<string, unknown>;
    expect(state._width).toBe(32);
    expect(state._height).toBe(24);
    expect(state._internalWidth).toBe(16);
    expect(state._internalHeight).toBe(12);
  });

  it('does not publish dimensions when pipeline replacement fails', () => {
    const engine = Object.create(Engine.prototype) as HybridEngine;
    const resize = vi.fn(() => {
      throw new Error('injected pipeline resize failure');
    });
    const resolveFrameResourceResolution = vi.fn(() => ({
      effectiveWidth: 16,
      effectiveHeight: 12,
      restirReservoirScale: 2,
    }));
    Object.assign(engine as unknown as Record<string, unknown>, {
      _state: 'ready',
      _width: 16,
      _height: 16,
      _internalWidth: 8,
      _internalHeight: 8,
      _resolutionFactor: 0.5,
      _lastScene: null,
      _frameResourceResolution: { persistentBytes: 123 },
      _resolveFrameResourceResolution: resolveFrameResourceResolution,
      _pipeline: { resize },
      _materialWarner: { warnInvalidSetSize: vi.fn() },
    });

    expect(() => engine.setSize(32, 24)).toThrow('injected pipeline resize failure');

    const state = engine as unknown as Record<string, unknown>;
    expect(resolveFrameResourceResolution)
      .toHaveBeenCalledWith(16, 12, 123, undefined);
    expect(resize).toHaveBeenCalledWith(16, 12, 2);
    expect(state._width).toBe(16);
    expect(state._height).toBe(16);
    expect(state._internalWidth).toBe(8);
    expect(state._internalHeight).toBe(8);
  });

  it('publishes committed dimensions before surfacing retirement failure', () => {
    const engine = Object.create(Engine.prototype) as HybridEngine;
    const retirementFailure = new SceneMutationFinalizationError(
      [new Error('injected retirement failure')],
      'injected committed resize retirement failure',
    );
    const resize = vi.fn(() => {
      throw retirementFailure;
    });
    const nextResolution = {
      effectiveWidth: 16,
      effectiveHeight: 12,
      restirReservoirScale: 2,
    };
    const resolveFrameResourceResolution = vi.fn(() => nextResolution);
    Object.assign(engine as unknown as Record<string, unknown>, {
      _state: 'ready',
      _width: 16,
      _height: 16,
      _internalWidth: 8,
      _internalHeight: 8,
      _resolutionFactor: 0.5,
      _lastScene: null,
      _frameResourceResolution: { persistentBytes: 123 },
      _resolveFrameResourceResolution: resolveFrameResourceResolution,
      _pipeline: { resize },
      _materialWarner: { warnInvalidSetSize: vi.fn() },
    });

    expect(() => engine.setSize(32, 24)).toThrow(retirementFailure);

    const state = engine as unknown as Record<string, unknown>;
    expect(resize).toHaveBeenCalledWith(16, 12, 2);
    expect(state._width).toBe(32);
    expect(state._height).toBe(24);
    expect(state._internalWidth).toBe(16);
    expect(state._internalHeight).toBe(12);
    expect(state._frameResourceResolution).toBe(nextResolution);
  });
});
