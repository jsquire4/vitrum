import { describe, expect, it, vi } from 'vitest';
import { WalkaroundGPUPipeline } from '../WalkaroundGPUPipeline.js';
import type { FrameResources } from '../resourceManager.js';
import type { SceneBVHBuffers } from '../../restir/bvhTypes.js';

type PipelineMutationHarness = {
  refreshBvhRefit: WalkaroundGPUPipeline['refreshBvhRefit'];
  replaceBvhAndEmitters: WalkaroundGPUPipeline['replaceBvhAndEmitters'];
  _initialized: boolean;
  _res: FrameResources;
  _learningBvhPositionsCpuData: ArrayBuffer;
  _bvhHost: {
    refreshBvhRefit: ReturnType<typeof vi.fn>;
    refreshBvhNodesOnly: ReturnType<typeof vi.fn>;
    refreshTlasRefit: ReturnType<typeof vi.fn>;
    refreshBvhFullRebuild: ReturnType<typeof vi.fn>;
    replaceBvhAndEmitters: ReturnType<typeof vi.fn>;
  };
  _ppg: {
    resetForSceneBvh: ReturnType<typeof vi.fn>;
  };
  _nrc: {
    resetForSceneBounds: ReturnType<typeof vi.fn>;
  };
  _regir: {
    refreshAfterEmitterRebuild: ReturnType<typeof vi.fn>;
  };
};

function makePipelineHarness(initialPositions: Float32Array): PipelineMutationHarness {
  const pipeline = new WalkaroundGPUPipeline({} as GPUDevice, 64, 32) as unknown as Record<string, unknown>;
  pipeline._initialized = true;
  pipeline._res = {};
  pipeline._learningBvhPositionsCpuData = initialPositions.buffer.slice(0);
  pipeline._bvhHost = {
    refreshBvhRefit: vi.fn(),
    refreshBvhNodesOnly: vi.fn(),
    refreshTlasRefit: vi.fn(),
    refreshBvhFullRebuild: vi.fn(),
    replaceBvhAndEmitters: vi.fn(),
  };
  pipeline._ppg = {
    resetForSceneBvh: vi.fn(),
  };
  pipeline._nrc = {
    resetForSceneBounds: vi.fn(),
  };
  pipeline._regir = {
    refreshAfterEmitterRebuild: vi.fn(),
  };
  return pipeline as unknown as PipelineMutationHarness;
}

describe('WalkaroundGPUPipeline learned scene mutation reset', () => {
  it('patches the BVH position shadow and cold-restarts PPG/NRC after refit updates', () => {
    const initial = new Float32Array([
      0, 0, 0, 0,
      1, 1, 1, 0,
    ]);
    const pipeline = makePipelineHarness(initial);
    const replacement = new Float32Array([10, 20, 30, 0]);

    pipeline.refreshBvhRefit(
      new ArrayBuffer(16),
      { byteOffset: 4 * 4, data: replacement.buffer },
    );

    const ppg = pipeline._ppg;
    const nrc = pipeline._nrc;
    expect(ppg.resetForSceneBvh).toHaveBeenCalledTimes(1);
    const bvhArg = ppg.resetForSceneBvh.mock.calls[0]?.[0] as { bvhPositions: { cpuData: ArrayBuffer } };
    expect(Array.from(new Float32Array(bvhArg.bvhPositions.cpuData))).toEqual([
      0, 0, 0, 0,
      10, 20, 30, 0,
    ]);
    expect(nrc.resetForSceneBounds).toHaveBeenCalledTimes(1);
    expect(nrc.resetForSceneBounds.mock.calls[0]?.[1][2]).toBeGreaterThan(30);
  });

  it('publishes the replacement shadow to PPG, NRC, and ReGIR after the BVH swap', () => {
    const pipeline = makePipelineHarness(new Float32Array([0, 0, 0, 0]));
    const replacement = new Float32Array([
      -2, -1, 0, 0,
      4, 3, 2, 0,
    ]);
    const bvh = {
      bvhPositions: { cpuData: replacement.buffer },
      lightTreeNodeCount: 2,
      lightTreeEnabled: true,
    } as SceneBVHBuffers;

    pipeline.replaceBvhAndEmitters(bvh);

    expect(pipeline._bvhHost.replaceBvhAndEmitters).toHaveBeenCalledTimes(1);
    expect(Array.from(new Float32Array(pipeline._learningBvhPositionsCpuData)))
      .toEqual(Array.from(replacement));
    expect(pipeline._ppg.resetForSceneBvh).toHaveBeenCalledTimes(1);
    expect(pipeline._nrc.resetForSceneBounds).toHaveBeenCalledTimes(1);
    expect(pipeline._regir.refreshAfterEmitterRebuild).toHaveBeenCalledWith(bvh);
  });
});
