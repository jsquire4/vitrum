import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  buildPassLayout,
  disposeTimestampState,
  initTimestampQueries,
  kickTimestampReadback,
  makeTimestampState,
  readTimestampsOnce,
} from '../timestampQueries.js';

function trackedResource(): {
  readonly resource: GPUBuffer;
  readonly destroy: ReturnType<typeof vi.fn>;
} {
  const destroy = vi.fn();
  return {
    resource: { destroy } as unknown as GPUBuffer,
    destroy,
  };
}

describe('timestamp-query lifecycle', () => {
  beforeEach(() => {
    vi.stubGlobal('GPUBufferUsage', {
      QUERY_RESOLVE: 1,
      COPY_SRC: 2,
      COPY_DST: 4,
      MAP_READ: 8,
    });
    vi.stubGlobal('GPUMapMode', { READ: 1 });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('defaults off when import-meta DEV is absent and enables only explicit DEV=true', () => {
    const querySet = {
      destroy: vi.fn(),
    } as unknown as GPUQuerySet;
    const buffers = [trackedResource(), trackedResource(), trackedResource()];
    const device = {
      features: new Set(['timestamp-query']),
      createQuerySet: vi.fn(() => querySet),
      createBuffer: vi.fn(() => buffers.shift()!.resource),
    } as unknown as GPUDevice;
    const state = makeTimestampState();

    initTimestampQueries(device, state, {});
    expect(device.createQuerySet).not.toHaveBeenCalled();
    expect(device.createBuffer).not.toHaveBeenCalled();

    initTimestampQueries(device, state, { DEV: true });
    expect(device.createQuerySet).toHaveBeenCalledOnce();
    expect(device.createBuffer).toHaveBeenCalledTimes(3);
    expect(state.querySet).toBe(querySet);

    disposeTimestampState(state);
  });

  it('rolls back a partial initialization without publishing it', () => {
    const queryDestroy = vi.fn();
    const first = trackedResource();
    const second = trackedResource();
    const state = makeTimestampState();
    const device = {
      features: new Set(['timestamp-query']),
      createQuerySet: vi.fn(() => ({
        destroy: queryDestroy,
      })),
      createBuffer: vi.fn()
        .mockReturnValueOnce(first.resource)
        .mockReturnValueOnce(second.resource)
        .mockImplementationOnce(() => {
          throw new Error('forced readback allocation failure');
        }),
    } as unknown as GPUDevice;

    expect(() => initTimestampQueries(device, state, { DEV: true }))
      .toThrow('forced readback allocation failure');
    expect(queryDestroy).toHaveBeenCalledOnce();
    expect(first.destroy).toHaveBeenCalledOnce();
    expect(second.destroy).toHaveBeenCalledOnce();
    expect(state.querySet).toBeNull();
    expect(state.resolveBuffer).toBeNull();
    expect(state.readbackA).toBeNull();
    expect(state.readbackB).toBeNull();
  });

  it.each([
    ['missing DEV opt-in', {}, new Set(['timestamp-query'])],
    ['adapter without timestamp-query', { DEV: true }, new Set<string>()],
  ] as const)('retires a prior generation when disabled by %s', (
    _reason,
    environment,
    features,
  ) => {
    const queryDestroy = vi.fn();
    const resolve = trackedResource();
    const readbackA = trackedResource();
    const readbackB = trackedResource();
    const state = makeTimestampState();
    state.querySet = { destroy: queryDestroy } as unknown as GPUQuerySet;
    state.resolveBuffer = resolve.resource;
    state.readbackA = readbackA.resource;
    state.readbackB = readbackB.resource;
    const device = {
      features,
      createQuerySet: vi.fn(),
      createBuffer: vi.fn(),
    } as unknown as GPUDevice;
    const consoleLog = vi.spyOn(console, 'log').mockImplementation(() => undefined);

    try {
      initTimestampQueries(device, state, environment);
      expect(queryDestroy).toHaveBeenCalledOnce();
      expect(resolve.destroy).toHaveBeenCalledOnce();
      expect(readbackA.destroy).toHaveBeenCalledOnce();
      expect(readbackB.destroy).toHaveBeenCalledOnce();
      expect(state).toMatchObject({
        querySet: null,
        resolveBuffer: null,
        readbackA: null,
        readbackB: null,
        disposed: true,
      });
      expect(device.createQuerySet).not.toHaveBeenCalled();
      expect(device.createBuffer).not.toHaveBeenCalled();
    } finally {
      consoleLog.mockRestore();
    }
  });

  it('destroys every owned resource while mapAsync is in flight and owns its rejection', async () => {
    let rejectMapping!: (reason: unknown) => void;
    const mapping = new Promise<void>((_resolve, reject) => {
      rejectMapping = reject;
    });
    const queryDestroy = vi.fn();
    const resolve = trackedResource();
    const readbackA = trackedResource();
    const readbackB = trackedResource();
    Object.assign(readbackA.resource, {
      mapAsync: vi.fn(() => mapping),
      getMappedRange: vi.fn(),
      unmap: vi.fn(),
    });
    const state = makeTimestampState();
    state.querySet = { destroy: queryDestroy } as unknown as GPUQuerySet;
    state.resolveBuffer = resolve.resource;
    state.readbackA = readbackA.resource;
    state.readbackB = readbackB.resource;

    kickTimestampReadback(state, 2, []);
    expect(state.readbackInFlight).toBe('A');
    disposeTimestampState(state);

    expect(queryDestroy).toHaveBeenCalledOnce();
    expect(resolve.destroy).toHaveBeenCalledOnce();
    expect(readbackA.destroy).toHaveBeenCalledOnce();
    expect(readbackB.destroy).toHaveBeenCalledOnce();
    expect(state).toMatchObject({
      querySet: null,
      resolveBuffer: null,
      readbackA: null,
      readbackB: null,
      readbackInFlight: null,
      disposed: true,
    });

    rejectMapping(new Error('destroyed while mapping'));
    await mapping.catch(() => undefined);
    await Promise.resolve();
    expect(state.lastGpuTimingsFrame).toBe(-1);
    expect(() => disposeTimestampState(state)).not.toThrow();
    expect(readbackA.destroy).toHaveBeenCalledOnce();
  });

  it('retires the one-shot diagnostic readback when mapAsync rejects', async () => {
    const readback = trackedResource();
    Object.assign(readback.resource, {
      mapAsync: vi.fn(() => Promise.reject(new Error('diagnostic map failed'))),
      unmap: vi.fn(),
    });
    const state = makeTimestampState();
    state.querySet = { destroy: vi.fn() } as unknown as GPUQuerySet;
    state.resolveBuffer = trackedResource().resource;
    const encoder = {
      resolveQuerySet: vi.fn(),
      copyBufferToBuffer: vi.fn(),
      finish: vi.fn(() => ({} as GPUCommandBuffer)),
    };
    const device = {
      createBuffer: vi.fn(() => readback.resource),
      createCommandEncoder: vi.fn(() => encoder),
      queue: { submit: vi.fn() },
    } as unknown as GPUDevice;
    const layout = buildPassLayout({ denoiserMode: 'none' });

    await expect(readTimestampsOnce(device, state, layout))
      .rejects.toThrow('diagnostic map failed');
    expect(readback.destroy).toHaveBeenCalledOnce();
    expect((readback.resource as unknown as { unmap: ReturnType<typeof vi.fn> }).unmap)
      .not.toHaveBeenCalled();
  });
});
