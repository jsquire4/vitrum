import { describe, expect, it, vi } from 'vitest';
import type { Scene } from '@vitrum/core';
import { WalkaroundGPUPipeline } from '../WalkaroundGPUPipeline.js';
import type { SceneBVHBuffers } from '../../restir/bvhTypes.js';
import type { PreparedSceneMutation } from '../../SceneMutationTransaction.js';

interface PipelineInternals {
  _initialized: boolean;
  _accumFrameIndex: number;
  _device: GPUDevice;
  _bvhHost: {
    prepareEmitterLightingReplacement(
      device: GPUDevice,
      bvh: SceneBVHBuffers,
      scene: Scene,
    ): PreparedSceneMutation;
  };
  _regir: {
    prepareForSceneBvh(bvh: SceneBVHBuffers): PreparedSceneMutation;
  };
}

function participant(
  name: string,
  events: string[],
  state: { live: boolean; finalized: boolean },
  failCommit = false,
): PreparedSceneMutation {
  return {
    commit: () => {
      events.push(`${name}:commit`);
      state.live = true;
      if (failCommit) throw new Error(`${name}:failure`);
    },
    rollback: () => {
      events.push(`${name}:rollback`);
      state.live = false;
    },
    finalize: () => {
      events.push(`${name}:finalize`);
      state.finalized = true;
    },
  };
}

function fixture(failRegir = false) {
  const events: string[] = [];
  const bvhState = { live: false, finalized: false };
  const regirState = { live: false, finalized: false };
  const bvh = {} as SceneBVHBuffers;
  const scene: Scene = {
    primitives: [],
    emitters: [],
    environment: { kind: 'none' },
  };
  const bvhPrepare = vi.fn(() => participant('bvh', events, bvhState));
  const regirPrepare = vi.fn(() =>
    participant('regir', events, regirState, failRegir));
  const pipeline = Object.create(
    WalkaroundGPUPipeline.prototype,
  ) as WalkaroundGPUPipeline;
  Object.assign(pipeline as unknown as PipelineInternals, {
    _initialized: true,
    _accumFrameIndex: 17,
    _device: {} as GPUDevice,
    _bvhHost: { prepareEmitterLightingReplacement: bvhPrepare },
    _regir: { prepareForSceneBvh: regirPrepare },
  });
  return {
    pipeline,
    internals: pipeline as unknown as PipelineInternals,
    bvh,
    scene,
    events,
    bvhState,
    regirState,
    bvhPrepare,
    regirPrepare,
  };
}

describe('WalkaroundGPUPipeline.prepareEmitterLightingMutation', () => {
  it('publishes BvhHost + ReGIR + accum reset and reverses all three', () => {
    const f = fixture();
    const prepared = f.pipeline.prepareEmitterLightingMutation(f.bvh, f.scene);

    expect(f.events).toEqual([]);
    prepared.commit();
    expect(f.bvhState.live).toBe(true);
    expect(f.regirState.live).toBe(true);
    expect(f.internals._accumFrameIndex).toBe(0);

    prepared.rollback();
    expect(f.bvhState.live).toBe(false);
    expect(f.regirState.live).toBe(false);
    expect(f.internals._accumFrameIndex).toBe(17);
    expect(f.events).toEqual([
      'bvh:commit',
      'regir:commit',
      'regir:rollback',
      'bvh:rollback',
    ]);
  });

  it('finalizes candidates only after the outer transaction accepts commit', () => {
    const f = fixture();
    const prepared = f.pipeline.prepareEmitterLightingMutation(f.bvh, f.scene);
    prepared.commit();
    prepared.finalize();

    expect(f.bvhState.finalized).toBe(true);
    expect(f.regirState.finalized).toBe(true);
    expect(f.internals._accumFrameIndex).toBe(0);
  });

  it('rolls BvhHost and the throwing ReGIR participant back on commit failure', () => {
    const f = fixture(true);
    const prepared = f.pipeline.prepareEmitterLightingMutation(f.bvh, f.scene);

    expect(() => prepared.commit()).toThrow('regir:failure');
    expect(f.bvhState.live).toBe(false);
    expect(f.regirState.live).toBe(false);
    expect(f.internals._accumFrameIndex).toBe(17);
    expect(f.events).toContain('bvh:rollback');
    expect(f.events).toContain('regir:rollback');
  });
});
