import { describe, expect, it, vi } from 'vitest';
import type { SceneBVHBuffers } from '../../restir/bvhTypes.js';
import type { PreparedSceneMutation } from '../../SceneMutationTransaction.js';
import type { CollectedBvhMutation } from '../CollectingBvhUpdateSink.js';
import { WalkaroundGPUPipeline } from '../WalkaroundGPUPipeline.js';

function participant() {
  return {
    commit: vi.fn(),
    rollback: vi.fn(),
    finalize: vi.fn(),
  } satisfies PreparedSceneMutation;
}

function recordingParticipant(
  id: string,
  events: string[],
  failCommit = false,
): PreparedSceneMutation {
  return {
    commit: () => {
      events.push(`commit:${id}`);
      if (failCommit) throw new Error(`${id} commit failure`);
    },
    rollback: () => {
      events.push(`rollback:${id}`);
    },
    finalize: () => {
      events.push(`finalize:${id}`);
    },
  };
}

describe('WalkaroundGPUPipeline scene-mutation submit boundary', () => {
  it('submits GPU skin commands as the sole prefix and rolls back all staged state on submit failure', () => {
    const finalCommand = {} as GPUCommandBuffer;
    const skinCommand = {} as GPUCommandBuffer;
    const encoder = {
      finish: vi.fn(() => finalCommand),
    } as unknown as GPUCommandEncoder;
    const submit = vi.fn(() => {
      throw new Error('injected submit failure');
    });
    const device = {
      createCommandEncoder: vi.fn(() => encoder),
      queue: { submit },
    } as unknown as GPUDevice;
    const bvhMutation = participant();
    const regirMutation = participant();
    const ppgMutation = participant();
    const bvhHost = { prepareMutation: vi.fn(() => bvhMutation) };
    const regir = { prepareForSceneBvh: vi.fn(() => regirMutation) };
    const originalShadow = new Uint8Array(64);
    originalShadow.fill(7);
    const pipeline = Object.create(WalkaroundGPUPipeline.prototype) as WalkaroundGPUPipeline;
    Object.assign(pipeline as unknown as Record<string, unknown>, {
      _initialized: true,
      _device: device,
      _bvhHost: bvhHost,
      _regir: regir,
      _nrc: null,
      _ppg: { prepareResetForSceneBvh: vi.fn(() => ppgMutation) },
      _res: {},
      _width: 8,
      _height: 8,
      _learningBvhPositionsCpuData: originalShadow.buffer.slice(0),
      _accumFrameIndex: 9,
    });
    const mutation: CollectedBvhMutation = {
      positions: [
        { byteOffset: 0, data: new Uint8Array(16).fill(1).buffer },
        { byteOffset: 16, data: new Uint8Array(16).fill(2).buffer },
      ],
      learningPositions: [
        { byteOffset: 32, data: new Uint8Array(16).fill(3).buffer },
      ],
      normals: [
        { byteOffset: 0, data: new Uint8Array(16).fill(4).buffer },
        { byteOffset: 16, data: new Uint8Array(16).fill(5).buffer },
      ],
      resetAccumulator: true,
    };
    const prepared = pipeline.prepareSceneMutation(
      mutation,
      {} as SceneBVHBuffers,
      [skinCommand],
    );

    expect(() => prepared.commit()).toThrow(/submit failure/);
    expect(submit).toHaveBeenCalledTimes(1);
    expect(submit).toHaveBeenCalledWith([skinCommand, finalCommand]);
    expect(bvhMutation.commit).toHaveBeenCalledTimes(1);
    expect(regirMutation.commit).toHaveBeenCalledTimes(1);
    expect(bvhMutation.rollback).toHaveBeenCalledTimes(1);
    expect(regirMutation.rollback).toHaveBeenCalledTimes(1);
    const internals = pipeline as unknown as {
      _learningBvhPositionsCpuData: ArrayBuffer;
      _accumFrameIndex: number;
    };
    expect([...new Uint8Array(internals._learningBvhPositionsCpuData)])
      .toEqual([...originalShadow]);
    expect(internals._accumFrameIndex).toBe(9);
    prepared.rollback();
    expect(submit).toHaveBeenCalledTimes(1);
  });

  it('reverse-rolls every dependency after a nested commit failure', () => {
    const events: string[] = [];
    const encoder = {
      finish: vi.fn(() => ({} as GPUCommandBuffer)),
    } as unknown as GPUCommandEncoder;
    const pipeline = Object.create(
      WalkaroundGPUPipeline.prototype,
    ) as WalkaroundGPUPipeline;
    Object.assign(pipeline as unknown as Record<string, unknown>, {
      _initialized: true,
      _device: {
        createCommandEncoder: vi.fn(() => encoder),
        queue: { submit: vi.fn() },
      } as unknown as GPUDevice,
      _bvhHost: {
        prepareMutation: vi.fn(() => recordingParticipant('bvh', events)),
      },
      _ppg: {
        prepareResetForSceneBvh: vi.fn(
          () => recordingParticipant('ppg', events, true),
        ),
      },
      _nrc: {
        prepareSceneReset: vi.fn(() => recordingParticipant('nrc', events)),
      },
      _regir: {
        prepareForSceneBvh: vi.fn(
          () => recordingParticipant('regir', events),
        ),
      },
      _res: {},
      _width: 8,
      _height: 8,
      _learningBvhPositionsCpuData: new ArrayBuffer(0),
      _accumFrameIndex: 0,
      _grisHistoryEpoch: 0,
      _temporalHistoryClearPending: false,
    });
    const prepared = pipeline.prepareSceneMutation(
      {
        nodes: [{ byteOffset: 0, data: new ArrayBuffer(32) }],
        resetAccumulator: false,
      },
      {
        bvhPositions: { cpuData: new ArrayBuffer(0) },
      } as SceneBVHBuffers,
    );

    expect(() => prepared.commit()).toThrow('ppg commit failure');
    expect(events).toEqual([
      'commit:bvh',
      'commit:ppg',
      'rollback:regir',
      'rollback:nrc',
      'rollback:ppg',
      'rollback:bvh',
    ]);
  });

  it('retires nested scene dependencies in reverse preparation order', () => {
    const events: string[] = [];
    const encoder = {
      finish: vi.fn(() => ({} as GPUCommandBuffer)),
    } as unknown as GPUCommandEncoder;
    const pipeline = Object.create(
      WalkaroundGPUPipeline.prototype,
    ) as WalkaroundGPUPipeline;
    Object.assign(pipeline as unknown as Record<string, unknown>, {
      _initialized: true,
      _device: {
        createCommandEncoder: vi.fn(() => encoder),
        queue: { submit: vi.fn() },
      } as unknown as GPUDevice,
      _bvhHost: {
        prepareMutation: vi.fn(() => recordingParticipant('bvh', events)),
      },
      _ppg: {
        prepareResetForSceneBvh: vi.fn(
          () => recordingParticipant('ppg', events),
        ),
      },
      _nrc: {
        prepareSceneReset: vi.fn(() => recordingParticipant('nrc', events)),
      },
      _regir: {
        prepareForSceneBvh: vi.fn(
          () => recordingParticipant('regir', events),
        ),
      },
      _res: {},
      _width: 8,
      _height: 8,
      _learningBvhPositionsCpuData: new ArrayBuffer(0),
      _accumFrameIndex: 0,
      _grisHistoryEpoch: 0,
      _temporalHistoryClearPending: false,
    });
    const prepared = pipeline.prepareSceneMutation(
      {
        nodes: [{ byteOffset: 0, data: new ArrayBuffer(32) }],
        resetAccumulator: false,
      },
      {
        bvhPositions: { cpuData: new ArrayBuffer(0) },
      } as SceneBVHBuffers,
    );

    prepared.commit();
    prepared.finalize();
    expect(events).toEqual([
      'commit:bvh',
      'commit:ppg',
      'commit:nrc',
      'commit:regir',
      'finalize:regir',
      'finalize:nrc',
      'finalize:ppg',
      'finalize:bvh',
    ]);
  });
});
