import { describe, expect, it, vi } from 'vitest';
import { installWebGPUPolyfills } from '../../../__tests__/helpers/webgpuPolyfills.js';
import type { RestirGISnapshot } from '../../giStateSnapshot.js';
import type { FrameResources } from '../resourceManager.js';
import { WalkaroundGPUPipeline } from '../WalkaroundGPUPipeline.js';
import { RESERVOIR_GI_STRIDE_U32 } from '../../gi/giLayout.js';

installWebGPUPolyfills();

const WIDTH = 4;
const HEIGHT = 4;
const STRIDE_U32 = RESERVOIR_GI_STRIDE_U32;
const RESERVOIR_U32 = 2 * 2 * STRIDE_U32;
const RESERVOIR_BYTES = RESERVOIR_U32 * Uint32Array.BYTES_PER_ELEMENT;

type TrackedBuffer = GPUBuffer & {
  readonly backing: ArrayBuffer;
  readonly destroy: ReturnType<typeof vi.fn>;
  readonly getMappedRange: ReturnType<typeof vi.fn>;
  readonly unmap: ReturnType<typeof vi.fn>;
};

type PipelineInternals = {
  _initialized: boolean;
  _res: FrameResources;
  _resourceCache: { clear(): void };
  _accumFrameIndex: number;
  _grisHistoryEpoch: number;
  _temporalHistoryClearPending: boolean;
  _temporalHistoryFullRatePending: boolean;
};

type TrackedCohort = {
  reservoirGiCurrentBuffer: TrackedBuffer;
  reservoirGiPreviousBuffer: TrackedBuffer;
  reservoirGiSpatialBuffer: TrackedBuffer;
};

function trackedBuffer(size = RESERVOIR_BYTES): TrackedBuffer {
  const backing = new ArrayBuffer(size);
  return {
    size,
    backing,
    destroy: vi.fn(),
    getMappedRange: vi.fn(() => backing),
    unmap: vi.fn(),
  } as unknown as TrackedBuffer;
}

function snapshot(overrides: Partial<RestirGISnapshot> = {}): RestirGISnapshot {
  const make = (seed: number): Uint32Array => {
    const values = new Uint32Array(RESERVOIR_U32);
    const floats = new Float32Array(values.buffer);
    const floatLanes = [
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18,
      20, 21, 22, 23, 26,
    ];
    for (let record = 0; record < 4; record += 1) {
      const base = record * STRIDE_U32;
      for (const lane of floatLanes) {
        floats[base + lane] = (seed + record + lane) * 0.001;
      }
      values[base + 15] = (seed + record) >>> 0;
      values[base + 19] = (seed * 17 + record) >>> 0;
      values[base + 24] = 1;
      values[base + 25] = record & 1;
      values[base + 27] = 37;
    }
    return values;
  };
  return {
    halfW: 2,
    halfH: 2,
    strideU32: STRIDE_U32,
    current: make(1),
    previous: make(2),
    spatial: make(3),
    ...overrides,
  };
}

function makeDevice(options: {
  failAllocationAt?: number;
  failMapAt?: number;
  failUnmapAt?: number;
  wrongMappedSizeAt?: number;
  aliasAt?: number;
  aliasBuffer?: GPUBuffer;
} = {}) {
  const candidates: TrackedBuffer[] = [];
  let allocations = 0;
  const device = {
    createBuffer: vi.fn((descriptor: GPUBufferDescriptor) => {
      allocations += 1;
      if (allocations === options.failAllocationAt) {
        throw new Error(`injected allocation failure ${allocations}`);
      }
      if (allocations === options.aliasAt && options.aliasBuffer != null) {
        return options.aliasBuffer;
      }
      const candidate = trackedBuffer(Number(descriptor.size));
      if (allocations === options.failMapAt) {
        candidate.getMappedRange.mockImplementationOnce(() => {
          throw new Error(`injected map failure ${allocations}`);
        });
      }
      if (allocations === options.wrongMappedSizeAt) {
        candidate.getMappedRange.mockReturnValueOnce(
          new ArrayBuffer(Math.max(0, Number(descriptor.size) - 4)),
        );
      }
      if (allocations === options.failUnmapAt) {
        candidate.unmap.mockImplementationOnce(() => {
          throw new Error(`injected unmap failure ${allocations}`);
        });
      }
      candidates.push(candidate);
      return candidate;
    }),
    queue: {},
  } as unknown as GPUDevice;
  return { device, candidates };
}

function makePipeline(
  device: GPUDevice,
  live: TrackedCohort = {
    reservoirGiCurrentBuffer: trackedBuffer(),
    reservoirGiPreviousBuffer: trackedBuffer(),
    reservoirGiSpatialBuffer: trackedBuffer(),
  },
) {
  const pipeline = new WalkaroundGPUPipeline(device, WIDTH, HEIGHT);
  const state = pipeline as unknown as PipelineInternals;
  state._initialized = true;
  state._res = { restirGI: live } as unknown as FrameResources;
  const clear = vi.spyOn(state._resourceCache, 'clear');
  return { pipeline, state, live, clear };
}

function expectLiveUnchanged(
  state: PipelineInternals,
  live: TrackedCohort,
): void {
  expect(state._res.restirGI).toBe(live);
  expect(live.reservoirGiCurrentBuffer.destroy).not.toHaveBeenCalled();
  expect(live.reservoirGiPreviousBuffer.destroy).not.toHaveBeenCalled();
  expect(live.reservoirGiSpatialBuffer.destroy).not.toHaveBeenCalled();
}

describe('ReSTIR-GI reservoir import transaction', () => {
  it.each([
    ['null', null],
    ['empty object', {}],
    [
      'missing arrays',
      { halfW: 2, halfH: 2, strideU32: STRIDE_U32 },
    ],
  ])('returns false instead of throwing for malformed JS preflight input: %s', (_label, value) => {
    const gpu = makeDevice();
    const made = makePipeline(gpu.device);

    expect(
      made.pipeline.canImportRestirGIReservoirs(
        value as unknown as RestirGISnapshot,
      ),
    ).toBe(false);
    expect(gpu.device.createBuffer).not.toHaveBeenCalled();
    expectLiveUnchanged(made.state, made.live);
  });

  it.each([
    ['wrong grid', { halfW: 3 }],
    ['retired compact stride', { strideU32: 20 }],
    ['short current', { current: new Uint32Array(RESERVOIR_U32 - 1) }],
    ['long previous', { previous: new Uint32Array(RESERVOIR_U32 + 1) }],
    ['short spatial', { spatial: new Uint32Array(RESERVOIR_U32 - 1) }],
  ] as const)('rejects %s without allocation or live publication', (_label, override) => {
    const gpu = makeDevice();
    const made = makePipeline(gpu.device);

    expect(
      made.pipeline.prepareRestirGIReservoirImport(
        gpu.device,
        snapshot(override),
      ),
    ).toBeNull();
    expect(gpu.device.createBuffer).not.toHaveBeenCalled();
    expect(made.clear).not.toHaveBeenCalled();
    expectLiveUnchanged(made.state, made.live);
  });

  it.each(['current', 'previous', 'spatial'] as const)(
    'rejects a non-finite logical %s reservoir before candidate allocation',
    (bufferName) => {
      const gpu = makeDevice();
      const made = makePipeline(gpu.device);
      const value = snapshot();
      value[bufferName][7] = 0x7f80_0000;

      expect(
        made.pipeline.prepareRestirGIReservoirImport(gpu.device, value),
      ).toBeNull();
      expect(gpu.device.createBuffer).not.toHaveBeenCalled();
      expect(made.clear).not.toHaveBeenCalled();
      expectLiveUnchanged(made.state, made.live);
    },
  );

  it('rejects a mixed live history epoch before candidate allocation', () => {
    const gpu = makeDevice();
    const made = makePipeline(gpu.device);
    const value = snapshot();
    value.previous[27] = 38;

    expect(
      made.pipeline.prepareRestirGIReservoirImport(gpu.device, value),
    ).toBeNull();
    expect(gpu.device.createBuffer).not.toHaveBeenCalled();
    expect(made.clear).not.toHaveBeenCalled();
    expectLiveUnchanged(made.state, made.live);
  });

  it('rejects a different GPUDevice without allocating on either device', () => {
    const owner = makeDevice();
    const foreign = makeDevice();
    const made = makePipeline(owner.device);

    expect(
      made.pipeline.prepareRestirGIReservoirImport(
        foreign.device,
        snapshot(),
      ),
    ).toBeNull();
    expect(owner.device.createBuffer).not.toHaveBeenCalled();
    expect(foreign.device.createBuffer).not.toHaveBeenCalled();
    expectLiveUnchanged(made.state, made.live);
  });

  it.each([
    ['allocation 1', { failAllocationAt: 1 }],
    ['allocation 2', { failAllocationAt: 2 }],
    ['allocation 3', { failAllocationAt: 3 }],
    ['map 1', { failMapAt: 1 }],
    ['map 2', { failMapAt: 2 }],
    ['map 3', { failMapAt: 3 }],
    ['mapped size 1', { wrongMappedSizeAt: 1 }],
    ['mapped size 2', { wrongMappedSizeAt: 2 }],
    ['mapped size 3', { wrongMappedSizeAt: 3 }],
    ['unmap 1', { failUnmapAt: 1 }],
    ['unmap 2', { failUnmapAt: 2 }],
    ['unmap 3', { failUnmapAt: 3 }],
  ] as const)('retires every candidate and retains the live generation on %s failure', (_label, options) => {
    const gpu = makeDevice(options);
    const made = makePipeline(gpu.device);

    expect(() =>
      made.pipeline.prepareRestirGIReservoirImport(gpu.device, snapshot()),
    ).toThrow();
    expectLiveUnchanged(made.state, made.live);
    expect(made.clear).not.toHaveBeenCalled();
    for (const candidate of gpu.candidates) {
      expect(candidate.destroy).toHaveBeenCalledOnce();
    }
  });

  it('does not destroy a live buffer when a faulty device aliases it as a candidate', () => {
    const liveCurrent = trackedBuffer();
    const live = {
      reservoirGiCurrentBuffer: liveCurrent,
      reservoirGiPreviousBuffer: trackedBuffer(),
      reservoirGiSpatialBuffer: trackedBuffer(),
    };
    const gpu = makeDevice({ aliasAt: 1, aliasBuffer: liveCurrent });
    const made = makePipeline(gpu.device, live);

    expect(() =>
      made.pipeline.prepareRestirGIReservoirImport(gpu.device, snapshot()),
    ).toThrow(/aliases/);
    expectLiveUnchanged(made.state, made.live);
  });

  it('rejects an aliased live cohort before allocating candidates', () => {
    const repeated = trackedBuffer();
    const gpu = makeDevice();
    const made = makePipeline(gpu.device, {
      reservoirGiCurrentBuffer: repeated,
      reservoirGiPreviousBuffer: repeated,
      reservoirGiSpatialBuffer: trackedBuffer(),
    });

    expect(
      made.pipeline.prepareRestirGIReservoirImport(gpu.device, snapshot()),
    ).toBeNull();
    expect(gpu.device.createBuffer).not.toHaveBeenCalled();
    expectLiveUnchanged(made.state, made.live);
  });

  it('prepares without invalidating live bind groups and rolls back publication if commit invalidation fails', () => {
    const gpu = makeDevice();
    const made = makePipeline(gpu.device);
    made.clear.mockImplementationOnce(() => {
      throw new Error('injected cache clear failure');
    });

    const transaction = made.pipeline.prepareRestirGIReservoirImport(
      gpu.device,
      snapshot(),
    );
    expect(transaction).not.toBeNull();
    expect(made.clear).not.toHaveBeenCalled();
    expect(() => transaction!.commit()).toThrow('injected cache clear failure');
    expectLiveUnchanged(made.state, made.live);
    expect(made.clear).toHaveBeenCalledTimes(2);
    for (const candidate of gpu.candidates) {
      expect(candidate.destroy).toHaveBeenCalledOnce();
    }
  });

  it('rolls back a prepared candidate cohort without changing live identities', () => {
    const gpu = makeDevice();
    const made = makePipeline(gpu.device);
    const transaction = made.pipeline.prepareRestirGIReservoirImport(
      gpu.device,
      snapshot(),
    );

    expect(transaction).not.toBeNull();
    expectLiveUnchanged(made.state, made.live);
    expect(made.clear).not.toHaveBeenCalled();
    transaction!.rollback();
    transaction!.rollback();
    transaction!.commit();

    expectLiveUnchanged(made.state, made.live);
    expect(made.clear).not.toHaveBeenCalled();
    for (const candidate of gpu.candidates) {
      expect(candidate.destroy).toHaveBeenCalledOnce();
    }
  });

  it('publishes all three buffers, supports rollback, and retires old buffers only at finalize', () => {
    const gpu = makeDevice();
    const made = makePipeline(gpu.device);
    made.state._accumFrameIndex = 19;
    made.state._grisHistoryEpoch = 11;
    made.state._temporalHistoryClearPending = true;
    made.state._temporalHistoryFullRatePending = false;
    const state = snapshot();
    const transaction = made.pipeline.prepareRestirGIReservoirImport(
      gpu.device,
      state,
    );

    expect(transaction).not.toBeNull();
    expectLiveUnchanged(made.state, made.live);
    transaction!.commit();
    transaction!.commit();

    const published = made.state._res.restirGI;
    expect(published).not.toBe(made.live);
    expect(published.reservoirGiCurrentBuffer).toBe(gpu.candidates[0]);
    expect(published.reservoirGiPreviousBuffer).toBe(gpu.candidates[1]);
    expect(published.reservoirGiSpatialBuffer).toBe(gpu.candidates[2]);
    expect(made.state._accumFrameIndex).toBe(0);
    expect(made.state._grisHistoryEpoch).toBe(37);
    expect(made.state._temporalHistoryClearPending).toBe(false);
    expect(made.state._temporalHistoryFullRatePending).toBe(true);
    expect(Array.from(new Uint32Array(gpu.candidates[0]!.backing))).toEqual(
      Array.from(state.current),
    );
    expect(Array.from(new Uint32Array(gpu.candidates[1]!.backing))).toEqual(
      Array.from(state.previous),
    );
    expect(Array.from(new Uint32Array(gpu.candidates[2]!.backing))).toEqual(
      Array.from(state.spatial),
    );
    expect(made.live.reservoirGiCurrentBuffer.destroy).not.toHaveBeenCalled();
    expect(made.live.reservoirGiPreviousBuffer.destroy).not.toHaveBeenCalled();
    expect(made.live.reservoirGiSpatialBuffer.destroy).not.toHaveBeenCalled();
    for (const candidate of gpu.candidates) {
      expect(candidate.destroy).not.toHaveBeenCalled();
    }

    transaction!.rollback();
    expectLiveUnchanged(made.state, made.live);
    expect(made.state._accumFrameIndex).toBe(19);
    expect(made.state._grisHistoryEpoch).toBe(11);
    expect(made.state._temporalHistoryClearPending).toBe(true);
    expect(made.state._temporalHistoryFullRatePending).toBe(false);
    expect(made.clear).toHaveBeenCalledTimes(2);
    for (const candidate of gpu.candidates) {
      expect(candidate.destroy).toHaveBeenCalledOnce();
    }
  });

  it('finalize retires the old generation only after publication is irreversible', () => {
    const gpu = makeDevice();
    const made = makePipeline(gpu.device);
    const transaction = made.pipeline.prepareRestirGIReservoirImport(
      gpu.device,
      snapshot(),
    )!;

    transaction.commit();
    expect(made.clear).toHaveBeenCalledOnce();
    expect(made.live.reservoirGiCurrentBuffer.destroy).not.toHaveBeenCalled();
    transaction.finalize();
    transaction.finalize();
    transaction.rollback();

    expect(made.live.reservoirGiCurrentBuffer.destroy).toHaveBeenCalledOnce();
    expect(made.live.reservoirGiPreviousBuffer.destroy).toHaveBeenCalledOnce();
    expect(made.live.reservoirGiSpatialBuffer.destroy).toHaveBeenCalledOnce();
    for (const candidate of gpu.candidates) {
      expect(candidate.destroy).not.toHaveBeenCalled();
    }
  });
});
