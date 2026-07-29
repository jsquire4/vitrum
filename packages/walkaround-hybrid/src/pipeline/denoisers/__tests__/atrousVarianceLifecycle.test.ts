import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { AtrousVarianceDenoiser } from '../atrousVariance.js';
import type { DenoiserInitContext } from '../index.js';

interface TrackedBuffer {
  readonly gpu: GPUBuffer;
  readonly destroy: ReturnType<typeof vi.fn>;
}

function trackedBuffer(): TrackedBuffer {
  const destroy = vi.fn();
  return {
    gpu: { destroy } as unknown as GPUBuffer,
    destroy,
  };
}

function refs(denoiser: AtrousVarianceDenoiser): readonly GPUBuffer[] {
  const seam = denoiser as unknown as {
    _welfordUboRef: { buf?: GPUBuffer };
    _varianceUboRef: { buf?: GPUBuffer };
    _atrousUboRef: { buf?: GPUBuffer };
  };
  return [
    seam._welfordUboRef.buf!,
    seam._varianceUboRef.buf!,
    seam._atrousUboRef.buf!,
  ];
}

function pipelines(denoiser: AtrousVarianceDenoiser): readonly GPUComputePipeline[] {
  const seam = denoiser as unknown as {
    _welfordPipeline: GPUComputePipeline;
    _variancePipeline: GPUComputePipeline;
    _atrousPipeline: GPUComputePipeline;
  };
  return [
    seam._welfordPipeline,
    seam._variancePipeline,
    seam._atrousPipeline,
  ];
}

describe('AtrousVarianceDenoiser initialize lifecycle', () => {
  beforeEach(() => {
    vi.stubGlobal('GPUBufferUsage', { UNIFORM: 1, COPY_DST: 2 });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('publishes a complete UBO generation, rolls back failed reinitialize, then retires the prior generation', async () => {
    const created: TrackedBuffer[] = [];
    let allocation = 0;
    let failAt = -1;
    let pipelineAttempt = 0;
    let failPipelineAt = -1;
    const createBuffer = vi.fn(() => {
      allocation++;
      if (allocation === failAt) {
        throw new Error(`forced allocation ${allocation}`);
      }
      const record = trackedBuffer();
      created.push(record);
      return record.gpu;
    });
    const shaderModule = {
      getCompilationInfo: vi.fn(async () => ({ messages: [] })),
    } as unknown as GPUShaderModule;
    const createComputePipelineAsync = vi.fn(async (
      descriptor: GPUComputePipelineDescriptor,
    ) => {
      pipelineAttempt++;
      if (pipelineAttempt === failPipelineAt) {
        throw new Error(`forced pipeline ${pipelineAttempt}`);
      }
      return {
        descriptor,
        pipelineAttempt,
        getBindGroupLayout: vi.fn(),
      } as unknown as GPUComputePipeline;
    });
    const device = {
      createShaderModule: vi.fn(() => shaderModule),
      createComputePipelineAsync,
      createBuffer,
    } as unknown as GPUDevice;
    const context = {
      device,
      width: 8,
      height: 8,
    } as DenoiserInitContext;
    const denoiser = new AtrousVarianceDenoiser();

    await denoiser.initialize(context);
    const firstGeneration = refs(denoiser);
    const firstPipelines = pipelines(denoiser);
    const firstRecords = created.slice(0, 3);
    expect(firstGeneration).toEqual(firstRecords.map((record) => record.gpu));

    // Pipeline candidates are also private until the matching UBO cohort is
    // complete. A compile failure must not mix old buffers with new pipelines.
    failPipelineAt = pipelineAttempt + 2;
    await expect(denoiser.initialize(context))
      .rejects.toThrow(`forced pipeline ${failPipelineAt}`);
    expect(pipelines(denoiser)).toEqual(firstPipelines);
    expect(refs(denoiser)).toEqual(firstGeneration);
    expect(created).toHaveLength(3);
    for (const record of firstRecords) {
      expect(record.destroy).not.toHaveBeenCalled();
    }

    // Reinitialize allocates two private candidates, then fails on the third.
    // The published cohort must remain byte/identity stable and usable.
    failPipelineAt = -1;
    failAt = allocation + 3;
    await expect(denoiser.initialize(context))
      .rejects.toThrow(`forced allocation ${failAt}`);
    expect(pipelines(denoiser)).toEqual(firstPipelines);
    expect(refs(denoiser)).toEqual(firstGeneration);
    for (const record of firstRecords) {
      expect(record.destroy).not.toHaveBeenCalled();
    }
    for (const record of created.slice(3, 5)) {
      expect(record.destroy).toHaveBeenCalledOnce();
    }

    failAt = -1;
    await denoiser.initialize(context);
    const thirdGeneration = refs(denoiser);
    const thirdPipelines = pipelines(denoiser);
    expect(thirdGeneration).not.toEqual(firstGeneration);
    expect(thirdPipelines).not.toEqual(firstPipelines);
    for (const record of firstRecords) {
      expect(record.destroy).toHaveBeenCalledOnce();
    }

    denoiser.dispose();
    denoiser.dispose();
    for (const buffer of thirdGeneration) {
      const record = created.find((candidate) => candidate.gpu === buffer)!;
      expect(record.destroy).toHaveBeenCalledOnce();
    }
  });

  it('rejects a candidate alias without destroying the live generation', async () => {
    const initial = [trackedBuffer(), trackedBuffer(), trackedBuffer()];
    let returnAlias = false;
    let initialIndex = 0;
    const device = {
      createShaderModule: vi.fn(() => ({
        getCompilationInfo: async () => ({ messages: [] }),
      })),
      createComputePipelineAsync: vi.fn(async () => ({
        getBindGroupLayout: vi.fn(),
      })),
      createBuffer: vi.fn(() => {
        if (returnAlias) return initial[0]!.gpu;
        return initial[initialIndex++]!.gpu;
      }),
    } as unknown as GPUDevice;
    const denoiser = new AtrousVarianceDenoiser();
    const context = { device, width: 8, height: 8 } as DenoiserInitContext;

    await denoiser.initialize(context);
    returnAlias = true;
    await expect(denoiser.initialize(context)).rejects.toThrow(/aliases a live resource/);
    expect(refs(denoiser)).toEqual(initial.map((record) => record.gpu));
    for (const record of initial) {
      expect(record.destroy).not.toHaveBeenCalled();
    }

    denoiser.dispose();
  });

  it('does not publish or allocate UBOs when dispose supersedes an in-flight initialize', async () => {
    let pipelineAttempt = 0;
    let resolveFinalPipeline!: (pipeline: GPUComputePipeline) => void;
    const finalPipeline = new Promise<GPUComputePipeline>((resolve) => {
      resolveFinalPipeline = resolve;
    });
    const createComputePipelineAsync = vi.fn(async () => {
      pipelineAttempt++;
      if (pipelineAttempt === 3) return finalPipeline;
      return {
        pipelineAttempt,
        getBindGroupLayout: vi.fn(),
      } as unknown as GPUComputePipeline;
    });
    const createBuffer = vi.fn();
    const device = {
      createShaderModule: vi.fn(() => ({
        getCompilationInfo: async () => ({ messages: [] }),
      })),
      createComputePipelineAsync,
      createBuffer,
    } as unknown as GPUDevice;
    const denoiser = new AtrousVarianceDenoiser();
    const initializing = denoiser.initialize({
      device,
      width: 8,
      height: 8,
    } as DenoiserInitContext);

    await vi.waitFor(() => {
      expect(createComputePipelineAsync).toHaveBeenCalledTimes(3);
    });
    denoiser.dispose();
    resolveFinalPipeline({
      pipelineAttempt: 3,
      getBindGroupLayout: vi.fn(),
    } as unknown as GPUComputePipeline);

    await expect(initializing).resolves.toBeUndefined();
    expect(createBuffer).not.toHaveBeenCalled();
    expect(refs(denoiser)).toEqual([undefined, undefined, undefined]);
    expect(pipelines(denoiser)).toEqual([undefined, undefined, undefined]);
  });
});
