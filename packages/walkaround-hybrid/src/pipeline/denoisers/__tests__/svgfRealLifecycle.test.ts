import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest';

import { SVGFRealDenoiser } from '../svgfReal.js';

type TestBuffer = GPUBuffer & {
  readonly destroy: ReturnType<typeof vi.fn>;
};

type TestPipeline = GPUComputePipeline & {
  readonly generation: number;
};

const previousGpuBufferUsage = (
  globalThis as { GPUBufferUsage?: unknown }
).GPUBufferUsage;

beforeAll(() => {
  vi.stubGlobal('GPUBufferUsage', {
    UNIFORM: 1,
    COPY_DST: 2,
  });
});

afterAll(() => {
  if (previousGpuBufferUsage === undefined) {
    delete (globalThis as { GPUBufferUsage?: unknown }).GPUBufferUsage;
  } else {
    (globalThis as { GPUBufferUsage?: unknown }).GPUBufferUsage =
      previousGpuBufferUsage;
  }
});

function shaderModule(): GPUShaderModule {
  return {
    getCompilationInfo: vi.fn(async () => ({ messages: [] })),
  } as unknown as GPUShaderModule;
}

function pipeline(generation: number): TestPipeline {
  return { generation } as TestPipeline;
}

function buffer(): TestBuffer {
  return {
    destroy: vi.fn(),
  } as unknown as TestBuffer;
}

describe('SVGFRealDenoiser initialization lifecycle', () => {
  it('publishes a complete generation or preserves every live pipeline and buffer', async () => {
    const created: TestBuffer[] = [];
    const createdPipelines: TestPipeline[] = [];
    let failAtCreatedCount: number | null = null;
    let pipelineAttempt = 0;
    let failAtPipelineAttempt: number | null = null;
    const device = {
      queue: { writeBuffer: vi.fn() },
      createShaderModule: vi.fn(shaderModule),
      createComputePipelineAsync: vi.fn(async () => {
        pipelineAttempt += 1;
        if (pipelineAttempt === failAtPipelineAttempt) {
          throw new Error('injected atrous pipeline failure');
        }
        const next = pipeline(pipelineAttempt);
        createdPipelines.push(next);
        return next;
      }),
      createBuffer: vi.fn(() => {
        if (
          failAtCreatedCount != null &&
          created.length >= failAtCreatedCount
        ) {
          throw new Error('injected UBO allocation failure');
        }
        const next = buffer();
        created.push(next);
        return next;
      }),
    } as unknown as GPUDevice;
    const denoiser = new SVGFRealDenoiser();
    const context = { device } as never;
    const internals = denoiser as unknown as {
      _reprojPipeline: GPUComputePipeline;
      _momentsPipeline: GPUComputePipeline;
      _fallbackPipeline: GPUComputePipeline;
      _atrousPipeline: GPUComputePipeline;
      _reprojUboRef: { buf?: GPUBuffer };
      _atrousUboRefs: Array<{ buf?: GPUBuffer }>;
    };

    await expect(denoiser.initialize(context)).resolves.toBeUndefined();
    expect(created).toHaveLength(6);
    expect(createdPipelines).toHaveLength(4);
    const livePipelines = [
      internals._reprojPipeline,
      internals._momentsPipeline,
      internals._fallbackPipeline,
      internals._atrousPipeline,
    ];
    const liveReproj = internals._reprojUboRef.buf;
    const liveAtrous = internals._atrousUboRefs.map((ref) => ref.buf);
    const expectLivePipelinesPreserved = () => {
      expect(internals._reprojPipeline).toBe(livePipelines[0]);
      expect(internals._momentsPipeline).toBe(livePipelines[1]);
      expect(internals._fallbackPipeline).toBe(livePipelines[2]);
      expect(internals._atrousPipeline).toBe(livePipelines[3]);
    };

    // A reinitialize compiles three candidates and then fails on the atrous
    // pipeline. No candidate buffer exists yet and the entire live generation
    // must remain installed.
    failAtPipelineAttempt = 8;
    await expect(denoiser.initialize(context)).rejects.toThrow(
      'injected atrous pipeline failure',
    );
    expect(created).toHaveLength(6);
    expectLivePipelinesPreserved();
    expect(internals._reprojUboRef.buf).toBe(liveReproj);
    expect(internals._atrousUboRefs.map((ref) => ref.buf)).toEqual(liveAtrous);

    // The replacement creates its reprojection UBO and one wavelet UBO, then
    // fails. Its four pipeline candidates remain unpublished, both candidate
    // buffers are retired, and every live identity remains installed.
    failAtPipelineAttempt = null;
    failAtCreatedCount = 8;
    await expect(denoiser.initialize(context)).rejects.toThrow(
      'injected UBO allocation failure',
    );

    expect(created).toHaveLength(8);
    expect(created[6]!.destroy).toHaveBeenCalledTimes(1);
    expect(created[7]!.destroy).toHaveBeenCalledTimes(1);
    expectLivePipelinesPreserved();
    expect(internals._reprojUboRef.buf).toBe(liveReproj);
    expect(internals._atrousUboRefs.map((ref) => ref.buf)).toEqual(liveAtrous);
    for (const live of created.slice(0, 6)) {
      expect(live.destroy).not.toHaveBeenCalled();
    }

    denoiser.dispose();
    for (const live of created.slice(0, 6)) {
      expect(live.destroy).toHaveBeenCalledTimes(1);
    }
  });
});
