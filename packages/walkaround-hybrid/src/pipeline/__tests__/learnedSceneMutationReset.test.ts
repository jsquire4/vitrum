import { describe, expect, it, vi } from 'vitest';
import { WalkaroundGPUPipeline } from '../WalkaroundGPUPipeline.js';
import type { FrameResources } from '../resourceManager.js';

type PipelineMutationHarness = {
  refreshBvhRefit: WalkaroundGPUPipeline['refreshBvhRefit'];
  _initialized: boolean;
  _res: FrameResources;
  _learningBvhPositionsCpuData: ArrayBuffer;
  _bvhHost: {
    refreshBvhRefit: ReturnType<typeof vi.fn>;
    refreshBvhNodesOnly: ReturnType<typeof vi.fn>;
    refreshTlasRefit: ReturnType<typeof vi.fn>;
    refreshBvhFullRebuild: ReturnType<typeof vi.fn>;
  };
  _ppg: {
    resetForSceneBvh: ReturnType<typeof vi.fn>;
  };
  _nrc: {
    resetForSceneBounds: ReturnType<typeof vi.fn>;
  };
};

function makePipelineHarness(initialPositions: Float32Array): PipelineMutationHarness {
  const pipeline = new WalkaroundGPUPipeline({} as GPUDevice, 64, 32) as unknown as Record<string, unknown>;
  pipeline._initialized = true;
  pipeline._res = {} as FrameResources;
  pipeline._learningBvhPositionsCpuData = initialPositions.buffer.slice(0);
  pipeline._bvhHost = {
    refreshBvhRefit: vi.fn(),
    refreshBvhNodesOnly: vi.fn(),
    refreshTlasRefit: vi.fn(),
    refreshBvhFullRebuild: vi.fn(),
  };
  pipeline._ppg = {
    resetForSceneBvh: vi.fn(),
  };
  pipeline._nrc = {
    resetForSceneBounds: vi.fn(),
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
});
