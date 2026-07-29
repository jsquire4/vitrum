import { beforeAll, describe, expect, it, vi } from 'vitest';
import type { EngineError } from '@vitrum/core';
import { FramePublicationTransaction } from '../FramePublication.js';
import { WalkaroundGPUPipeline } from '../WalkaroundGPUPipeline.js';

type SubmitHarness = Record<string, unknown> & {
  _submitAndRunPostSubmitHooks(
    encoder: GPUCommandEncoder,
    publication: FramePublicationTransaction,
    frameCount: number,
    labels: readonly never[],
  ): void;
};

beforeAll(() => {
  Object.assign(globalThis, { GPUMapMode: { READ: 1 } });
});

function makeHarness(options: {
  readonly afterFrameSubmit?: () => void;
  readonly mapAsync?: () => Promise<void>;
  readonly onError: (error: EngineError) => void;
}): SubmitHarness {
  const pipeline = Object.create(WalkaroundGPUPipeline.prototype) as SubmitHarness;
  pipeline._initialized = true;
  pipeline._onError = options.onError;
  pipeline._device = { queue: { submit: vi.fn() } };
  pipeline._activeDenoiser = options.afterFrameSubmit == null
    ? null
    : { afterFrameSubmit: options.afterFrameSubmit };
  const readback = {
    mapAsync: options.mapAsync ?? (() => Promise.resolve()),
    getMappedRange: () => new ArrayBuffer(0),
    unmap: vi.fn(),
  };
  pipeline._tsState = {
    querySet: null,
    resolveBuffer: {},
    readbackA: readback,
    readbackB: readback,
    readbackInFlight: null,
    readbackGeneration: 0,
    disposed: false,
    periodNs: 1,
    lastGpuTimings: { previous: 1 },
    lastGpuTimingsFrame: 7,
  };
  pipeline.lastGpuTimings = {};
  pipeline.lastGpuTimingsFrame = -1;
  return pipeline;
}

describe('accepted-frame post-submit contract', () => {
  it('contains denoiser and timestamp kickoff throws after publishing exactly once', () => {
    const denoiserError = new Error('denoiser hook failed');
    const timestampError = new Error('mapAsync threw synchronously');
    const onError = vi.fn();
    const afterFrameSubmit = vi.fn(() => { throw denoiserError; });
    const mapAsync = vi.fn(() => { throw timestampError; }) as unknown as () => Promise<void>;
    const pipeline = makeHarness({ afterFrameSubmit, mapAsync, onError });
    const publication = new FramePublicationTransaction();
    const accepted = vi.fn();
    publication.stage(accepted);
    const commandBuffer = {} as GPUCommandBuffer;
    const encoder = { finish: vi.fn(() => commandBuffer) } as unknown as GPUCommandEncoder;

    expect(() => pipeline._submitAndRunPostSubmitHooks(
      encoder,
      publication,
      2,
      [],
    )).not.toThrow();

    expect(publication.state).toBe('accepted');
    expect(accepted).toHaveBeenCalledOnce();
    expect(afterFrameSubmit).toHaveBeenCalledOnce();
    expect(mapAsync).toHaveBeenCalledOnce();
    expect((pipeline._device as { queue: { submit: ReturnType<typeof vi.fn> } }).queue.submit)
      .toHaveBeenCalledWith([commandBuffer]);
    expect((pipeline._tsState as { readbackInFlight: unknown }).readbackInFlight).toBeNull();
    expect(onError).toHaveBeenCalledTimes(2);
    expect(onError.mock.calls.map(([error]) => error.raw)).toEqual([
      denoiserError,
      timestampError,
    ]);
    expect(onError.mock.calls.every(([error]) => error.fatal === false)).toBe(true);

    publication.accept();
    publication.abort();
    expect(accepted).toHaveBeenCalledOnce();
  });

  it('contains accepted publication callback errors and still runs later hooks', () => {
    const publicationError = new Error('accepted callback failed');
    const onError = vi.fn();
    const afterFrameSubmit = vi.fn();
    const pipeline = makeHarness({ afterFrameSubmit, onError });
    const publication = new FramePublicationTransaction();
    const laterAccepted = vi.fn();
    publication.stage(() => { throw publicationError; });
    publication.stage(laterAccepted);

    expect(() => pipeline._submitAndRunPostSubmitHooks(
      { finish: () => ({} as GPUCommandBuffer) } as GPUCommandEncoder,
      publication,
      2,
      [],
    )).not.toThrow();

    expect(publication.state).toBe('accepted');
    expect(laterAccepted).toHaveBeenCalledOnce();
    expect(afterFrameSubmit).toHaveBeenCalledOnce();
    expect(onError).toHaveBeenCalledOnce();
    expect(onError.mock.calls[0]![0]).toMatchObject({
      fatal: false,
      raw: expect.objectContaining({ errors: [publicationError] }),
    });
  });
});
