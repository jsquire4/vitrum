import { describe, expect, it, vi } from 'vitest';
import { installWebGPUPolyfills } from '../../../__tests__/helpers/webgpuPolyfills.js';
import type { FrameResources } from '../resourceManager.js';
import { WalkaroundGPUPipeline } from '../WalkaroundGPUPipeline.js';
import type { RestirDISnapshot } from '../../restir/restirDiStateSnapshot.js';
import { RESERVOIR_DI_STRIDE_U32 } from '../../restir/reservoirDiLayout.js';

installWebGPUPolyfills();

const WIDTH = 4;
const HEIGHT = 4;
const RESERVOIR_U32 = WIDTH * HEIGHT * RESERVOIR_DI_STRIDE_U32;
const RESERVOIR_BYTES =
  RESERVOIR_U32 * Uint32Array.BYTES_PER_ELEMENT;

type TrackedBuffer = GPUBuffer & {
  readonly backing: ArrayBuffer;
  readonly destroy: ReturnType<typeof vi.fn>;
  readonly getMappedRange: ReturnType<typeof vi.fn>;
  readonly unmap: ReturnType<typeof vi.fn>;
};

type TrackedCohort = {
  reservoirCurrentBuffer: TrackedBuffer;
  reservoirPreviousBuffer: TrackedBuffer;
  reservoirSpatialBuffer: TrackedBuffer;
};

type PipelineInternals = {
  _initialized: boolean;
  _width: number;
  _height: number;
  _reservoirScale: number;
  _res: FrameResources;
  _resourceCache: { clear(): void };
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

function makePayload(seed: number): Uint32Array {
  const data = new Uint32Array(RESERVOIR_U32);
  const floats = new Float32Array(data.buffer);
  for (let record = 0; record < WIDTH * HEIGHT; record += 1) {
    const base = record * RESERVOIR_DI_STRIDE_U32;
    data[base] = (seed + record) >>> 0;
    data[base + 1] = 2;
    floats[base + 2] = 4 + seed;
    floats[base + 3] = 0.5;
    floats[base + 4] = 0.25;
    floats[base + 5] = 0.75;
    data[base + 6] = 2;
    data[base + 7] = 0;
  }
  return data;
}

function snapshot(
  overrides: Partial<RestirDISnapshot> = {},
): RestirDISnapshot {
  return {
    width: WIDTH,
    height: HEIGHT,
    strideU32: RESERVOIR_DI_STRIDE_U32,
    current: makePayload(1),
    previous: makePayload(2),
    spatial: makePayload(3),
    ...overrides,
  };
}

function makeDevice() {
  const candidates: TrackedBuffer[] = [];
  const device = {
    createBuffer: vi.fn((descriptor: GPUBufferDescriptor) => {
      const buffer = trackedBuffer(Number(descriptor.size));
      candidates.push(buffer);
      return buffer;
    }),
    queue: {},
  } as unknown as GPUDevice;
  return { device, candidates };
}

function makePipeline(
  device: GPUDevice,
  dimensions: {
    readonly width?: number;
    readonly height?: number;
    readonly reservoirScale?: number;
  } = {},
) {
  const live: TrackedCohort = {
    reservoirCurrentBuffer: trackedBuffer(),
    reservoirPreviousBuffer: trackedBuffer(),
    reservoirSpatialBuffer: trackedBuffer(),
  };
  const pipeline = new WalkaroundGPUPipeline(
    device,
    dimensions.width ?? WIDTH,
    dimensions.height ?? HEIGHT,
  );
  const state = pipeline as unknown as PipelineInternals;
  state._initialized = true;
  state._reservoirScale = dimensions.reservoirScale ?? 1;
  state._res = { restirDI: live } as unknown as FrameResources;
  const clear = vi.spyOn(state._resourceCache, 'clear');
  return { pipeline, state, live, clear };
}

describe('ReSTIR-DI reservoir import transaction', () => {
  it('accepts only the DI grid implied by internal dimensions and active scale', () => {
    const gpu = makeDevice();
    const made = makePipeline(gpu.device, {
      width: 8,
      height: 8,
      reservoirScale: 2,
    });

    expect(made.pipeline.canImportRestirDIReservoirs(snapshot())).toBe(true);
    made.state._reservoirScale = 4;
    expect(made.pipeline.canImportRestirDIReservoirs(snapshot())).toBe(false);
    expect(gpu.device.createBuffer).not.toHaveBeenCalled();
    expect(made.state._res.restirDI).toBe(made.live);
  });

  it('rejects malformed or foreign-device input before allocation', () => {
    const owner = makeDevice();
    const foreign = makeDevice();
    const made = makePipeline(owner.device);

    expect(
      made.pipeline.prepareRestirDIReservoirImport(
        owner.device,
        snapshot({ width: WIDTH + 1 }),
      ),
    ).toBeNull();
    expect(
      made.pipeline.prepareRestirDIReservoirImport(
        foreign.device,
        snapshot(),
      ),
    ).toBeNull();
    expect(owner.device.createBuffer).not.toHaveBeenCalled();
    expect(foreign.device.createBuffer).not.toHaveBeenCalled();
    expect(made.state._res.restirDI).toBe(made.live);
  });

  it('keeps the exact live cohort through prepare and restores it on rollback', () => {
    const gpu = makeDevice();
    const made = makePipeline(gpu.device);
    const state = snapshot();
    const transaction = made.pipeline.prepareRestirDIReservoirImport(
      gpu.device,
      state,
    );

    expect(transaction).not.toBeNull();
    expect(made.state._res.restirDI).toBe(made.live);
    expect(made.clear).not.toHaveBeenCalled();

    transaction!.commit();
    const published = made.state._res.restirDI;
    expect(published.reservoirCurrentBuffer).toBe(gpu.candidates[0]);
    expect(published.reservoirPreviousBuffer).toBe(gpu.candidates[1]);
    expect(published.reservoirSpatialBuffer).toBe(gpu.candidates[2]);
    expect(
      new Uint32Array(gpu.candidates[0]!.backing),
    ).toEqual(state.current);
    expect(made.clear).toHaveBeenCalledOnce();
    for (const buffer of Object.values(made.live)) {
      expect(buffer.destroy).not.toHaveBeenCalled();
    }

    transaction!.rollback();
    expect(made.state._res.restirDI).toBe(made.live);
    expect(made.clear).toHaveBeenCalledTimes(2);
    for (const candidate of gpu.candidates) {
      expect(candidate.destroy).toHaveBeenCalledOnce();
    }
    for (const buffer of Object.values(made.live)) {
      expect(buffer.destroy).not.toHaveBeenCalled();
    }
  });

  it('retires the old cohort only after a committed restore is finalized', () => {
    const gpu = makeDevice();
    const made = makePipeline(gpu.device);
    const transaction = made.pipeline.prepareRestirDIReservoirImport(
      gpu.device,
      snapshot(),
    )!;

    transaction.commit();
    transaction.finalize();
    transaction.rollback();

    for (const buffer of Object.values(made.live)) {
      expect(buffer.destroy).toHaveBeenCalledOnce();
    }
    for (const candidate of gpu.candidates) {
      expect(candidate.destroy).not.toHaveBeenCalled();
    }
  });
});
