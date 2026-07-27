import { describe, expect, it, vi } from 'vitest';
import type { Scene } from '@vitrum/core';
import type { SceneBVHBuffers } from '../restir/bvhTypes.js';
import type { CollectedBvhMutation } from '../pipeline/CollectingBvhUpdateSink.js';
import { WalkaroundGPUPipeline } from '../pipeline/WalkaroundGPUPipeline.js';
import type { PreparedSceneMutation } from '../SceneMutationTransaction.js';

const noOpMutation = (): PreparedSceneMutation => ({
  commit: vi.fn(),
  rollback: vi.fn(),
  finalize: vi.fn(),
});

interface EncoderStub {
  clearBuffer: (...args: never[]) => void;
  finish: () => GPUCommandBuffer;
}

interface DeviceStub {
  device: GPUDevice;
  encoder: EncoderStub;
  submit: () => void;
}

function makeDevice(submitImpl?: () => void): DeviceStub {
  const encoder: EncoderStub = {
    clearBuffer: vi.fn(),
    finish: vi.fn(() => ({ label: 'mutation-command-buffer' } as GPUCommandBuffer)),
  };
  const submit = vi.fn(submitImpl ?? (() => undefined));
  const device = {
    createCommandEncoder: vi.fn(() => encoder),
    queue: { submit },
  } as unknown as GPUDevice;
  return { device, encoder, submit };
}

type PipelinePrivate = {
  _initialized: boolean;
  _grisReuseStructural: boolean;
  _grisHistoryEpoch: number;
  _grisHistoryClearPending: boolean;
  _accumFrameIndex: number;
  _learningBvhPositionsCpuData: ArrayBuffer | null;
  _nrc: null;
  _res: {
    restirGI: {
      reservoirGiCurrentBuffer: GPUBuffer;
      reservoirGiPreviousBuffer: GPUBuffer;
      reservoirGiSpatialBuffer: GPUBuffer;
    };
  };
  _bvhHost: {
    initialized: boolean;
    updateEnvironment: (...args: never[]) => void;
    prepareMutation: () => PreparedSceneMutation;
    prepareEmitterLightingReplacement: () => PreparedSceneMutation;
  };
  _regir: {
    prepareForSceneBvh: () => PreparedSceneMutation;
  };
  _ppg: {
    prepareResetForSceneBvh: () => PreparedSceneMutation;
  };
};

function makePipeline(submitImpl?: () => void): {
  pipeline: WalkaroundGPUPipeline;
  state: PipelinePrivate;
  gpu: DeviceStub;
} {
  const gpu = makeDevice(submitImpl);
  const pipeline = new WalkaroundGPUPipeline(gpu.device, 64, 64);
  const state = pipeline as unknown as PipelinePrivate;
  state._initialized = true;
  state._grisReuseStructural = true;
  state._grisHistoryEpoch = 1;
  state._grisHistoryClearPending = false;
  state._accumFrameIndex = 0;
  state._learningBvhPositionsCpuData = new ArrayBuffer(64);
  state._nrc = null;
  state._res = {
    restirGI: {
      reservoirGiCurrentBuffer: { size: 120 } as GPUBuffer,
      reservoirGiPreviousBuffer: { size: 120 } as GPUBuffer,
      reservoirGiSpatialBuffer: { size: 120 } as GPUBuffer,
    },
  };
  state._bvhHost = {
    initialized: true,
    updateEnvironment: vi.fn(),
    prepareMutation: vi.fn(() => noOpMutation()),
    prepareEmitterLightingReplacement: vi.fn(() => noOpMutation()),
  };
  state._regir = {
    prepareForSceneBvh: vi.fn(() => noOpMutation()),
  };
  state._ppg = {
    prepareResetForSceneBvh: vi.fn(() => noOpMutation()),
  };
  return { pipeline, state, gpu };
}

const nextBvh = {} as SceneBVHBuffers;
const nextScene = {
  primitives: [],
  emitters: [],
  environment: { kind: 'none' },
} as Scene;

function materialMutation(): CollectedBvhMutation {
  const bytes = new ArrayBuffer(4);
  return {
    material: {
      index: { byteOffset: 0, data: bytes },
      beer: { data: bytes, triCount: 1 },
      emissive: { data: bytes, triCount: 1 },
    },
    resetAccumulator: true,
  };
}

describe('GRIS history epoch mutation boundary', () => {
  it('is inert in compact mode and becomes active only for the fixed GRIS layout', () => {
    const { pipeline, state } = makePipeline();
    state._grisReuseStructural = false;
    state._accumFrameIndex = 9;

    pipeline.requestAccumReset();
    expect(state._accumFrameIndex).toBe(0);
    expect(state._grisHistoryEpoch).toBe(1);
    expect(state._grisHistoryClearPending).toBe(false);

    state._grisReuseStructural = true;
    state._accumFrameIndex = 9;
    pipeline.requestAccumReset();
    expect(state._grisHistoryEpoch).toBe(2);
    expect(state._grisHistoryClearPending).toBe(true);
  });

  it('never publishes epoch zero when the u32 counter wraps', () => {
    const { pipeline, state } = makePipeline();
    state._grisHistoryEpoch = 0xffffffff;
    pipeline.requestAccumReset();
    expect(state._grisHistoryEpoch).toBe(1);
    expect(state._grisHistoryClearPending).toBe(true);
  });

  it('invalidates history for a lighting/environment mutation', () => {
    const { pipeline, state } = makePipeline();
    state._grisHistoryEpoch = 7;
    state._accumFrameIndex = 12;

    pipeline.updateDirectionalEnvironment(null, 0.25, 1.5);

    expect(state._bvhHost.updateEnvironment).toHaveBeenCalledWith(
      expect.anything(),
      null,
      0.25,
      1.5,
    );
    expect(state._accumFrameIndex).toBe(0);
    expect(state._grisHistoryEpoch).toBe(8);
    expect(state._grisHistoryClearPending).toBe(true);
  });

  it('makes emitter publication epoch invalidation transactional', () => {
    const { pipeline, state } = makePipeline();
    state._grisHistoryEpoch = 11;
    state._accumFrameIndex = 6;
    const prepared = pipeline.prepareEmitterLightingMutation(nextBvh, nextScene);

    prepared.commit();
    expect(state._accumFrameIndex).toBe(0);
    expect(state._grisHistoryEpoch).toBe(12);
    expect(state._grisHistoryClearPending).toBe(true);

    prepared.rollback();
    expect(state._accumFrameIndex).toBe(6);
    expect(state._grisHistoryEpoch).toBe(11);
    expect(state._grisHistoryClearPending).toBe(false);
  });

  it('clears all widened reservoirs in the submitted material transaction', () => {
    const { pipeline, state, gpu } = makePipeline();
    state._grisHistoryEpoch = 20;
    state._accumFrameIndex = 14;
    const prepared = pipeline.prepareSceneMutation(materialMutation(), nextBvh);

    expect(gpu.encoder.clearBuffer).toHaveBeenCalledTimes(3);
    prepared.commit();

    expect(gpu.submit).toHaveBeenCalledTimes(1);
    expect(state._accumFrameIndex).toBe(0);
    expect(state._grisHistoryEpoch).toBe(21);
    expect(state._grisHistoryClearPending).toBe(false);
  });

  it('invalidates geometry history in the same submitted transaction', () => {
    const { pipeline, state, gpu } = makePipeline();
    state._grisHistoryEpoch = 30;
    const mutation: CollectedBvhMutation = {
      nodes: [{
        byteOffset: 0,
        data: new Uint32Array([1, 2, 3, 4]).buffer,
      }],
      resetAccumulator: true,
    };
    const prepared = pipeline.prepareSceneMutation(mutation, nextBvh);

    expect(state._ppg.prepareResetForSceneBvh).toHaveBeenCalledTimes(1);
    expect(gpu.encoder.clearBuffer).toHaveBeenCalledTimes(3);
    prepared.commit();
    expect(state._grisHistoryEpoch).toBe(31);
    expect(state._grisHistoryClearPending).toBe(false);
  });

  it('rolls epoch, pending-clear, and accumulator state back when clear submit fails', () => {
    const { pipeline, state, gpu } = makePipeline(() => {
      throw new Error('submit rejected');
    });
    state._grisHistoryEpoch = 41;
    state._grisHistoryClearPending = true;
    state._accumFrameIndex = 17;
    const prepared = pipeline.prepareSceneMutation(materialMutation(), nextBvh);

    expect(gpu.encoder.clearBuffer).toHaveBeenCalledTimes(3);
    expect(() => prepared.commit()).toThrow('submit rejected');
    expect(state._grisHistoryEpoch).toBe(41);
    expect(state._grisHistoryClearPending).toBe(true);
    expect(state._accumFrameIndex).toBe(17);
  });
});
