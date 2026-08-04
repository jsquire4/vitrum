import { describe, expect, it, vi } from 'vitest';
import type { EngineWarning } from '@vitrum/core';
import { REPRESENTED_PROPOSAL_BUCKET_COUNT } from '@vitrum/shared-samplers';
import { dTreeAccumulateFlux } from '../../ppg/dTree.js';
import { buildSTree } from '../../ppg/sTree.js';
import {
  DTREE_HEADER_F32,
  DTREE_NODE_F32,
  serialiseSTree,
} from '../../ppg/serialise.js';
import type { STree } from '../../ppg/types.js';
import { PPGCoordinator } from '../PPGCoordinator.js';
import type { FrameResources } from '../resourceManager.js';
import { installWebGPUPolyfills } from '../../../__tests__/helpers/webgpuPolyfills.js';
import { createPpgQueryArenaLayout } from '../../ppg/ppgQueryArena.js';

installWebGPUPolyfills();

const BOUNDS = {
  min: [0, 0, 0] as [number, number, number],
  max: [1, 1, 1] as [number, number, number],
};

type Internals = {
  _enabled: boolean;
  _maxSpatialCells?: number;
  _maxDTreeNodesPerCell?: number;
  _sceneAABB: typeof BOUNDS;
  _sTree: STree | null;
  _frameResourcesGeneration: number;
  _trainingDispatchesSinceRefine: number;
  _trainingReadbackFailures: number;
  _lastTrainingReadbackErrorMessage: string | null;
  _fluxReadbackInFlight: boolean;
  _trainingEpochState: 'collecting' | 'readback' | 'retry-pending' | 'failed' | 'disposed';
  _width: number;
  _height: number;
  _fluxReadbackBuffer: GPUBuffer | null;
  _cellCountReadbackBuffer: GPUBuffer | null;
  _mergeFluxAndRefine(
    rawFlux: Float32Array,
    cellCounts: Uint32Array,
    frameResources: FrameResources,
    maxSpatialCells: number,
    maxDTreeNodesPerCell: number,
  ): void;
};

type TrackedBuffer = GPUBuffer & {
  bytes: Uint8Array;
  label?: string;
  destroy: ReturnType<typeof vi.fn>;
  getMappedRange?: ReturnType<typeof vi.fn>;
  unmap?: ReturnType<typeof vi.fn>;
};

function destination(size: number, fill = 0x5a): TrackedBuffer {
  const bytes = new Uint8Array(size);
  bytes.fill(fill);
  return { size, bytes, destroy: vi.fn() } as unknown as TrackedBuffer;
}

function frameResources(maxDTreeNodesPerCell = 4): FrameResources {
  const queryArenaLayout = createPpgQueryArenaLayout({
    sTreeCapacityBytes: 512,
    dTreeCapacityBytes: 512,
    dTreeOffsetsCapacityBytes: 32,
    maxSpatialCells: 8,
    maxDTreeNodesPerCell,
  });
  return {
    ppg: {
      queryArenaBuf: destination(queryArenaLayout.byteLength),
      queryArenaLayout,
      queryArenaEpoch: 1,
      fluxAtomicsBuf: destination(
        8 * maxDTreeNodesPerCell * Uint32Array.BYTES_PER_ELEMENT,
      ),
      cellSampleCountsBuf: destination(32),
      updateUboBuffer: destination(16),
    },
  } as unknown as FrameResources;
}

function treeWithFlux(flux: number): STree {
  const tree = buildSTree(BOUNDS, 0);
  dTreeAccumulateFlux(tree.dTrees[0]!, [0.25, 0.25], flux);
  return tree;
}

function treeWithFluxInBounds(
  flux: number,
  bounds: typeof BOUNDS,
): STree {
  const tree = buildSTree(bounds, 0);
  dTreeAccumulateFlux(tree.dTrees[0]!, [0.25, 0.25], flux);
  return tree;
}

function snapshot(flux = 9) {
  return {
    maxSpatialCells: 8,
    maxDTreeNodesPerCell: 4,
    ...serialiseSTree(treeWithFlux(flux), 4),
    sceneBoundsMin: BOUNDS.min,
    sceneBoundsMax: BOUNDS.max,
  };
}

function coordinator(
  device: GPUDevice,
  warnings: EngineWarning[] = [],
  bounds: typeof BOUNDS = BOUNDS,
): { value: PPGCoordinator; state: Internals; oldTree: STree } {
  const value = new PPGCoordinator(device, {
    onWarning: (warning) => warnings.push(warning),
  });
  const oldTree = treeWithFluxInBounds(1, bounds);
  const state = value as unknown as Internals;
  state._enabled = true;
  state._maxSpatialCells = 8;
  state._maxDTreeNodesPerCell = 4;
  state._sceneAABB = bounds;
  state._sTree = oldTree;
  state._frameResourcesGeneration = 7;
  return { value, state, oldTree };
}

function snapshotForBounds(bounds: typeof BOUNDS, flux = 9) {
  return {
    maxSpatialCells: 8,
    maxDTreeNodesPerCell: 4,
    ...serialiseSTree(treeWithFluxInBounds(flux, bounds), 4),
    sceneBoundsMin: bounds.min.map(Math.fround) as [number, number, number],
    sceneBoundsMax: bounds.max.map(Math.fround) as [number, number, number],
  };
}

function deferred(): {
  promise: Promise<void>;
  resolve(): void;
} {
  let resolve!: () => void;
  const promise = new Promise<void>((done) => { resolve = done; });
  return { promise, resolve };
}

function uploadDevice(options: {
  failAllocationAt?: number;
  failCopyAt?: number;
  failCreateEncoder?: boolean;
  failClearAt?: number;
  failFinish?: boolean;
  failSubmit?: boolean;
  failWorkDoneSync?: boolean;
  failWriteAt?: number;
  aliasAllocationAt?: number;
  aliasBuffer?: GPUBuffer;
  submittedWork?: Promise<void>;
  onSubmit?: () => void;
} = {}) {
  const staging: TrackedBuffer[] = [];
  const copies: unknown[][] = [];
  const clears: unknown[][] = [];
  let allocations = 0;
  let copyCalls = 0;
  let clearCalls = 0;
  let writeCalls = 0;
  const encoder = {
    copyBufferToBuffer: vi.fn((...args: unknown[]) => {
      copyCalls++;
      if (copyCalls === options.failCopyAt) throw new Error('injected copy failure');
      copies.push(args);
      const source = args[0] as TrackedBuffer;
      const sourceOffset = Number(args[1]);
      const target = args[2] as TrackedBuffer;
      const targetOffset = Number(args[3]);
      const byteLength = Number(args[4]);
      target.bytes.set(
        source.bytes.subarray(sourceOffset, sourceOffset + byteLength),
        targetOffset,
      );
    }),
    clearBuffer: vi.fn((...args: unknown[]) => {
      clearCalls++;
      if (clearCalls === options.failClearAt) {
        throw new Error('injected clear failure');
      }
      clears.push(args);
      const target = args[0] as TrackedBuffer;
      const offset = Number(args[1] ?? 0);
      const byteLength = Number(args[2] ?? target.size - offset);
      target.bytes.fill(0, offset, offset + byteLength);
    }),
    finish: vi.fn(() => {
      if (options.failFinish) throw new Error('injected finish failure');
      return {} as GPUCommandBuffer;
    }),
  };
  const queue = {
    submit: vi.fn(() => {
      options.onSubmit?.();
      if (options.failSubmit) throw new Error('injected submit failure');
    }),
    onSubmittedWorkDone: vi.fn(() => {
      if (options.failWorkDoneSync) {
        throw new Error('injected completion kickoff failure');
      }
      return options.submittedWork ?? Promise.resolve();
    }),
    writeBuffer: vi.fn((
      target: TrackedBuffer,
      targetOffset: number,
      data: ArrayBuffer | ArrayBufferView,
      dataOffset = 0,
      byteLength?: number,
    ) => {
      writeCalls++;
      if (writeCalls === options.failWriteAt) {
        throw new Error('injected write failure');
      }
      const view = ArrayBuffer.isView(data)
        ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
        : new Uint8Array(data);
      const length = byteLength ?? view.byteLength - dataOffset;
      target.bytes.set(
        view.subarray(dataOffset, dataOffset + length),
        targetOffset,
      );
    }),
  };
  const device = {
    createBuffer: vi.fn((descriptor: GPUBufferDescriptor) => {
      allocations++;
      if (allocations === options.failAllocationAt) {
        throw new Error('injected allocation failure');
      }
      if (
        allocations === options.aliasAllocationAt &&
        options.aliasBuffer !== undefined
      ) {
        const alias = options.aliasBuffer as TrackedBuffer;
        staging.push(alias);
        return alias;
      }
      const mapped = new ArrayBuffer(Number(descriptor.size));
      const buffer = {
        label: descriptor.label,
        size: Number(descriptor.size),
        bytes: new Uint8Array(mapped),
        getMappedRange: vi.fn(() => mapped),
        unmap: vi.fn(),
        destroy: vi.fn(),
      } as unknown as TrackedBuffer;
      staging.push(buffer);
      return buffer;
    }),
    createCommandEncoder: vi.fn(() => {
      if (options.failCreateEncoder) {
        throw new Error('injected encoder allocation failure');
      }
      return encoder;
    }),
    queue,
  } as unknown as GPUDevice;
  return { device, staging, copies, clears, encoder, queue };
}

async function flushMicrotasks(): Promise<void> {
  await Promise.resolve();
  await Promise.resolve();
}

type AllocatedPPG = Extract<
  FrameResources['ppg'],
  { queryArenaBuf: GPUBuffer }
>;

function ppgBuffers(ppg: AllocatedPPG): TrackedBuffer[] {
  return [
    ppg.queryArenaBuf,
    ppg.fluxAtomicsBuf,
    ppg.cellSampleCountsBuf,
    ppg.updateUboBuffer,
  ] as TrackedBuffer[];
}

function captureBytes(ppg: AllocatedPPG): Uint8Array[] {
  return ppgBuffers(ppg).map((buffer) => buffer.bytes.slice());
}

function expectBytes(ppg: AllocatedPPG, expected: readonly Uint8Array[]): void {
  ppgBuffers(ppg).forEach((buffer, index) => {
    expect(buffer.bytes).toEqual(expected[index]);
  });
}

function queryDTreeView(ppg: AllocatedPPG): Float32Array {
  const queryArena = ppg.queryArenaBuf as TrackedBuffer;
  return new Float32Array(
    queryArena.bytes.buffer,
    queryArena.bytes.byteOffset + ppg.queryArenaLayout.dTreeByteOffset,
    ppg.queryArenaLayout.dTreeCapacityBytes / Float32Array.BYTES_PER_ELEMENT,
  );
}

describe('PPG snapshot import transaction', () => {
  it('keeps the persistent ABI canonical while publishing represented query buckets', () => {
    const gpu = uploadDevice();
    const made = coordinator(gpu.device);
    const resources = frameResources();

    expect(made.value.importSTree(snapshot(), resources)).toBe(true);
    const live = resources.ppg as AllocatedPPG;
    expect(queryDTreeView(live)[DTREE_HEADER_F32 + 6]).toBe(
      REPRESENTED_PROPOSAL_BUCKET_COUNT,
    );

    const exported = made.value.exportSTree();
    expect(exported).not.toBeNull();
    // Leaf lane 6 is still the canonical firstChild=-1 sentinel on disk.
    expect(exported!.dTreeBuf[DTREE_HEADER_F32 + 6]).toBe(-1);
  });

  it('accepts large-world scene bounds after their float32 wire round-trip', () => {
    const bounds = {
      min: [
        1_000_000_033,
        -2_000_000_017,
        3_000_000_049,
      ] as [number, number, number],
      max: [
        1_000_001_033,
        -1_999_999_017,
        3_000_001_049,
      ] as [number, number, number],
    };
    const warnings: EngineWarning[] = [];
    const gpu = uploadDevice();
    const made = coordinator(gpu.device, warnings, bounds);

    expect(
      made.value.importSTree(snapshotForBounds(bounds), frameResources()),
    ).toBe(true);
    expect(
      warnings.some(
        (warning) =>
          warning.code ===
          'walkaround-hybrid.ppg-import-scene-bounds-mismatch',
      ),
    ).toBe(false);
  });

  it('rejects a distinct large-world scene bound before GPU mutation', () => {
    const bounds = {
      min: [
        1_000_000_033,
        -2_000_000_017,
        3_000_000_049,
      ] as [number, number, number],
      max: [
        1_000_001_033,
        -1_999_999_017,
        3_000_001_049,
      ] as [number, number, number],
    };
    const warnings: EngineWarning[] = [];
    const gpu = uploadDevice();
    const made = coordinator(gpu.device, warnings, bounds);
    const snap = snapshotForBounds(bounds);
    snap.sceneBoundsMin[0] += 256;

    expect(made.value.importSTree(snap, frameResources())).toBe(false);
    expect(made.state._sTree).toBe(made.oldTree);
    expect(made.state._frameResourcesGeneration).toBe(7);
    expect(gpu.device.createBuffer).not.toHaveBeenCalled();
    expect(gpu.queue.submit).not.toHaveBeenCalled();
    expect(warnings.at(-1)?.code).toBe(
      'walkaround-hybrid.ppg-import-scene-bounds-mismatch',
    );
  });

  it('rejects malformed state before allocation, encoding, submission, or publication', () => {
    const warnings: EngineWarning[] = [];
    const gpu = uploadDevice();
    const { value, state, oldTree } = coordinator(gpu.device, warnings);
    const malformed = snapshot();
    malformed.dTreeBuf[4] = Number.NaN;

    expect(value.importSTree(malformed, frameResources())).toBe(false);
    expect(state._sTree).toBe(oldTree);
    expect(state._frameResourcesGeneration).toBe(7);
    expect(gpu.device.createBuffer).not.toHaveBeenCalled();
    expect(gpu.device.createCommandEncoder).not.toHaveBeenCalled();
    expect(gpu.queue.submit).not.toHaveBeenCalled();
    expect(warnings.at(-1)?.code).toBe('walkaround-hybrid.ppg-import-malformed-snapshot');
  });

  it('probes compatibility without allocating, warning, or mutating state', () => {
    const warnings: EngineWarning[] = [];
    const gpu = uploadDevice();
    const made = coordinator(gpu.device, warnings);
    const resources = frameResources();
    const live = resources.ppg as AllocatedPPG;
    const before = captureBytes(live);
    const malformed = snapshot();
    malformed.dTreeBuf[4] = Number.NaN;

    expect(made.value.canImportSTree(snapshot(), resources)).toBe(true);
    expect(made.value.canImportSTree(malformed, resources)).toBe(false);
    expect(made.state._sTree).toBe(made.oldTree);
    expect(made.state._frameResourcesGeneration).toBe(7);
    expect(resources.ppg).toBe(live);
    expectBytes(live, before);
    expect(gpu.device.createBuffer).not.toHaveBeenCalled();
    expect(gpu.queue.submit).not.toHaveBeenCalled();
    expect(warnings).toEqual([]);
  });

  it('prepares an isolated cohort, then restores exact live CPU/GPU identity on rollback', () => {
    const gpu = uploadDevice();
    const made = coordinator(gpu.device);
    const resources = frameResources();
    const live = resources.ppg as AllocatedPPG;
    const before = captureBytes(live);
    const fluxReadback = destination(16, 0x31);
    const countReadback = destination(16, 0x32);
    made.state._trainingDispatchesSinceRefine = 11;
    made.state._trainingReadbackFailures = 2;
    made.state._lastTrainingReadbackErrorMessage = 'old error';
    made.state._fluxReadbackInFlight = true;
    made.state._trainingEpochState = 'retry-pending';
    made.state._fluxReadbackBuffer = fluxReadback;
    made.state._cellCountReadbackBuffer = countReadback;

    const transaction = made.value.prepareSTreeImport(snapshot(), resources);
    expect(transaction).not.toBeNull();
    expect(gpu.queue.submit).toHaveBeenCalledOnce();
    expect(gpu.clears).toHaveLength(2);
    expect(resources.ppg).toBe(live);
    expect(made.state._sTree).toBe(made.oldTree);
    expect(made.state._frameResourcesGeneration).toBe(7);
    expectBytes(live, before);
    expect(gpu.staging).toHaveLength(4);
    expect(gpu.staging[1]?.bytes.every((byte) => byte === 0)).toBe(true);
    expect(gpu.staging[2]?.bytes.every((byte) => byte === 0)).toBe(true);

    transaction!.commit();
    expect(resources.ppg).not.toBe(live);
    expect(made.state._sTree?.dTrees[0]?.totalFlux).toBe(9);
    expect(made.state._frameResourcesGeneration).toBe(8);
    expect(made.state._trainingDispatchesSinceRefine).toBe(0);
    expect(made.state._trainingReadbackFailures).toBe(0);
    expect(made.state._lastTrainingReadbackErrorMessage).toBeNull();
    expect(made.state._fluxReadbackInFlight).toBe(false);
    expect(made.state._trainingEpochState).toBe('collecting');
    for (const buffer of ppgBuffers(live)) {
      expect(buffer.destroy).not.toHaveBeenCalled();
    }
    expect(fluxReadback.destroy).not.toHaveBeenCalled();
    expect(countReadback.destroy).not.toHaveBeenCalled();

    transaction!.rollback();
    expect(resources.ppg).toBe(live);
    expect(made.state._sTree).toBe(made.oldTree);
    expect(made.state._frameResourcesGeneration).toBe(7);
    expect(made.state._trainingDispatchesSinceRefine).toBe(11);
    expect(made.state._trainingReadbackFailures).toBe(2);
    expect(made.state._lastTrainingReadbackErrorMessage).toBe('old error');
    expect(made.state._fluxReadbackInFlight).toBe(true);
    expect(made.state._trainingEpochState).toBe('retry-pending');
    expectBytes(live, before);
    for (const buffer of gpu.staging) {
      expect(buffer.destroy).toHaveBeenCalledOnce();
    }
    for (const buffer of ppgBuffers(live)) {
      expect(buffer.destroy).not.toHaveBeenCalled();
    }
    expect(fluxReadback.destroy).not.toHaveBeenCalled();
    expect(countReadback.destroy).not.toHaveBeenCalled();
  });

  it('finalize retires only the old cohort/readbacks and keeps the cold replacement live', () => {
    const gpu = uploadDevice();
    const made = coordinator(gpu.device);
    const resources = frameResources();
    const live = resources.ppg as AllocatedPPG;
    const fluxReadback = destination(16);
    const countReadback = destination(16);
    made.state._fluxReadbackBuffer = fluxReadback;
    made.state._cellCountReadbackBuffer = countReadback;
    const transaction = made.value.prepareSTreeImport(snapshot(), resources)!;

    transaction.commit();
    const replacement = resources.ppg as AllocatedPPG;
    transaction.finalize();

    expect(resources.ppg).toBe(replacement);
    expect(made.state._sTree?.dTrees[0]?.totalFlux).toBe(9);
    expect(made.state._trainingEpochState).toBe('collecting');
    expect(
      (replacement.fluxAtomicsBuf as TrackedBuffer).bytes.every(
        (byte) => byte === 0,
      ),
    ).toBe(true);
    expect(
      (replacement.cellSampleCountsBuf as TrackedBuffer).bytes.every(
        (byte) => byte === 0,
      ),
    ).toBe(true);
    for (const buffer of ppgBuffers(live)) {
      expect(buffer.destroy).toHaveBeenCalledOnce();
    }
    for (const buffer of ppgBuffers(replacement)) {
      expect(buffer.destroy).not.toHaveBeenCalled();
    }
    expect(fluxReadback.destroy).toHaveBeenCalledOnce();
    expect(countReadback.destroy).toHaveBeenCalledOnce();
  });

  it.each([
    ['allocation-1', { failAllocationAt: 1 }],
    ['allocation-2', { failAllocationAt: 2 }],
    ['allocation-3', { failAllocationAt: 3 }],
    ['allocation-4', { failAllocationAt: 4 }],
    ['tree-write-1', { failWriteAt: 1 }],
    ['tree-write-2', { failWriteAt: 2 }],
    ['tree-write-3', { failWriteAt: 3 }],
    ['header-write', { failWriteAt: 4 }],
    ['ubo-write', { failWriteAt: 5 }],
    ['encoder', { failCreateEncoder: true }],
    ['clear-1', { failClearAt: 1 }],
    ['clear-2', { failClearAt: 2 }],
    ['finish', { failFinish: true }],
    ['submit', { failSubmit: true }],
  ] as const)('keeps live handles, bytes, and CPU state exact on %s failure', (_label, options) => {
    const warnings: EngineWarning[] = [];
    const gpu = uploadDevice(options);
    const made = coordinator(gpu.device, warnings);
    const resources = frameResources();
    const live = resources.ppg as AllocatedPPG;
    const before = captureBytes(live);

    expect(made.value.importSTree(snapshot(), resources)).toBe(false);
    expect(resources.ppg).toBe(live);
    expect(made.state._sTree).toBe(made.oldTree);
    expect(made.state._frameResourcesGeneration).toBe(7);
    expectBytes(live, before);
    for (const buffer of gpu.staging) {
      expect(buffer.destroy).toHaveBeenCalledOnce();
    }
    for (const buffer of ppgBuffers(live)) {
      expect(buffer.destroy).not.toHaveBeenCalled();
    }
    expect(warnings.at(-1)?.code).toBe(
      'walkaround-hybrid.ppg-import-upload-failed',
    );
  });

  it('rejects a candidate/live alias without touching or destroying the live buffer', () => {
    const resources = frameResources();
    const live = resources.ppg as AllocatedPPG;
    const before = captureBytes(live);
    const gpu = uploadDevice({
      aliasAllocationAt: 1,
      aliasBuffer: live.queryArenaBuf,
    });
    const made = coordinator(gpu.device);

    expect(made.value.importSTree(snapshot(), resources)).toBe(false);
    expect(resources.ppg).toBe(live);
    expect(made.state._sTree).toBe(made.oldTree);
    expectBytes(live, before);
    expect(live.queryArenaBuf.destroy).not.toHaveBeenCalled();
    for (const buffer of gpu.staging.slice(1)) {
      expect(buffer.destroy).toHaveBeenCalledOnce();
    }
  });
});

describe('PPG refine publication transaction', () => {
  it('publishes represented subtree and leaf buckets after a training refinement', () => {
    const gpu = uploadDevice();
    const made = coordinator(gpu.device);
    const resources = frameResources(5);
    const trained = buildSTree(BOUNDS, 1);
    dTreeAccumulateFlux(trained.dTrees[0]!, [0.25, 0.25], 1);
    made.state._sTree = trained;
    made.state._maxDTreeNodesPerCell = 5;

    made.state._mergeFluxAndRefine(
      new Float32Array(5),
      new Uint32Array(1),
      resources,
      8,
      5,
    );

    const query = queryDTreeView(resources.ppg as AllocatedPPG);
    const rootBase = DTREE_HEADER_F32;
    const firstLeafBase = DTREE_HEADER_F32 + DTREE_NODE_F32;
    expect(query[rootBase + 5]).toBe(REPRESENTED_PROPOSAL_BUCKET_COUNT);
    expect(query[firstLeafBase + 6]).toBe(
      REPRESENTED_PROPOSAL_BUCKET_COUNT,
    );
  });

  it.each([
    ['allocation', { failAllocationAt: 2 }],
    ['encode', { failCopyAt: 2 }],
    ['finish', { failFinish: true }],
    ['submit', { failSubmit: true }],
  ] as const)('retains the live CPU/GPU guide and retires staged candidates on %s failure', (_label, options) => {
    const gpu = uploadDevice(options);
    const made = coordinator(gpu.device);
    const resources = frameResources();
    const ppg = resources.ppg as Extract<FrameResources['ppg'], { queryArenaBuf: GPUBuffer }>;

    expect(() => made.state._mergeFluxAndRefine(
      new Float32Array(4),
      new Uint32Array(1),
      resources,
      8,
      4,
    )).toThrow();

    expect(made.state._sTree).toBe(made.oldTree);
    expect(ppg.queryArenaEpoch).toBe(1);
    for (const buffer of gpu.staging) {
      expect(buffer.destroy).toHaveBeenCalledOnce();
    }
  });

  it('submits payload, header, and bounded clears atomically before swapping the CPU guide', async () => {
    const completion = deferred();
    const observed: {
      made?: ReturnType<typeof coordinator>;
      ppg?: Extract<FrameResources['ppg'], { queryArenaBuf: GPUBuffer }>;
    } = {};
    const gpu = uploadDevice({
      submittedWork: completion.promise,
      onSubmit: () => {
        expect(observed.made?.state._sTree).toBe(observed.made?.oldTree);
        expect(observed.ppg?.queryArenaEpoch).toBe(1);
      },
    });
    const made = coordinator(gpu.device);
    const resources = frameResources();
    const ppg = resources.ppg as Extract<FrameResources['ppg'], { queryArenaBuf: GPUBuffer }>;
    observed.made = made;
    observed.ppg = ppg;

    made.state._mergeFluxAndRefine(
      new Float32Array(4),
      new Uint32Array(1),
      resources,
      8,
      4,
    );

    expect(gpu.queue.submit).toHaveBeenCalledOnce();
    expect(gpu.copies).toHaveLength(4);
    expect(gpu.copies[3]?.[3]).toBe(0);
    expect(gpu.clears.map((call) => call.slice(1))).toEqual([
      [0, 16],
      [0, 4],
    ]);
    expect(ppg.queryArenaEpoch).toBe(2);
    expect(made.state._sTree).not.toBe(made.oldTree);
    for (const buffer of gpu.staging) {
      expect(buffer.destroy).not.toHaveBeenCalled();
    }

    completion.resolve();
    await flushMicrotasks();
    for (const buffer of gpu.staging) {
      expect(buffer.destroy).toHaveBeenCalledOnce();
    }
  });
});

describe('PPG resize transaction', () => {
  it('restores resources and every published lifecycle field after rollback', () => {
    const gpu = uploadDevice();
    const made = coordinator(gpu.device);
    const resources = frameResources();
    const previous = resources.ppg as AllocatedPPG;
    made.state._frameResourcesGeneration = 11;
    made.state._fluxReadbackInFlight = true;
    made.state._trainingEpochState = 'retry-pending';
    made.state._trainingReadbackFailures = 2;
    made.state._trainingDispatchesSinceRefine = 5;
    made.state._lastTrainingReadbackErrorMessage = 'retained failure';
    made.state._width = 16;
    made.state._height = 12;

    const mutation = made.value.prepareResize(resources, 32, 24, 7);
    const candidate = resources.ppg as AllocatedPPG;
    expect(candidate).not.toBe(previous);
    // Preparation may populate the isolated frame candidate, but it must not
    // publish coordinator lifecycle state.
    expect(made.state._frameResourcesGeneration).toBe(11);
    expect(made.state._trainingEpochState).toBe('retry-pending');
    expect(made.state._width).toBe(16);
    expect(made.state._height).toBe(12);

    mutation.commit();
    expect(made.state._frameResourcesGeneration).toBe(12);
    expect(made.state._trainingEpochState).toBe('collecting');
    expect(made.state._width).toBe(32);
    expect(made.state._height).toBe(24);

    mutation.rollback();
    expect(resources.ppg).toBe(previous);
    expect(made.state._frameResourcesGeneration).toBe(11);
    expect(made.state._fluxReadbackInFlight).toBe(true);
    expect(made.state._trainingEpochState).toBe('retry-pending');
    expect(made.state._trainingReadbackFailures).toBe(2);
    expect(made.state._trainingDispatchesSinceRefine).toBe(5);
    expect(made.state._lastTrainingReadbackErrorMessage).toBe('retained failure');
    expect(made.state._width).toBe(16);
    expect(made.state._height).toBe(12);
    for (const buffer of ppgBuffers(previous)) {
      expect(buffer.destroy).not.toHaveBeenCalled();
    }
    for (const buffer of ppgBuffers(candidate)) {
      expect(buffer.destroy).toHaveBeenCalledOnce();
    }
  });

  it('retires the prior PPG generation only after successful finalization', () => {
    const gpu = uploadDevice();
    const made = coordinator(gpu.device);
    const resources = frameResources();
    const previous = resources.ppg as AllocatedPPG;

    const mutation = made.value.prepareResize(resources, 32, 24, 7);
    const candidate = resources.ppg as AllocatedPPG;
    mutation.commit();

    for (const buffer of ppgBuffers(previous)) {
      expect(buffer.destroy).not.toHaveBeenCalled();
    }
    for (const buffer of ppgBuffers(candidate)) {
      expect(buffer.destroy).not.toHaveBeenCalled();
    }

    mutation.finalize();
    expect(resources.ppg).toBe(candidate);
    for (const buffer of ppgBuffers(previous)) {
      expect(buffer.destroy).toHaveBeenCalledOnce();
    }
    for (const buffer of ppgBuffers(candidate)) {
      expect(buffer.destroy).not.toHaveBeenCalled();
    }
  });
});

describe('PPG coordinator initialization transaction', () => {
  it('publishes the query-only bucket overlay during a scene reset', () => {
    const gpu = uploadDevice();
    const made = coordinator(gpu.device);
    const resources = frameResources(5);
    made.state._maxDTreeNodesPerCell = 5;

    made.value.resetForSceneBvh(
      {
        bvhPositions: {
          cpuData: new Float32Array([
            -2, -3, -4, 0,
            5, 6, 7, 0,
          ]).buffer,
        },
      },
      resources,
      16,
      16,
    );

    const query = queryDTreeView(resources.ppg as AllocatedPPG);
    // A cold depth-1 root is interior. Canonical lane 5 is -1; the query view
    // overlays the represented zero-bucket subtree count instead.
    expect(query[DTREE_HEADER_F32 + 5]).toBe(0);
    expect(made.value.exportSTree()!.dTreeBuf[DTREE_HEADER_F32 + 5]).toBe(-1);
  });

  it('retains prior coordinator/resources and destroys the complete candidate on upload failure', () => {
    const gpu = uploadDevice({ failWriteAt: 3 });
    const made = coordinator(gpu.device);
    const resources = frameResources();
    const previous = resources.ppg as Extract<FrameResources['ppg'], { queryArenaBuf: GPUBuffer }>;

    expect(() => made.value.initialize(
      {
        bvhPositions: {
          cpuData: new Float32Array([0, 0, 0, 0]).buffer,
        },
      } as never,
      resources,
      16,
      16,
      true,
      0.5,
      8,
      4,
    )).toThrow('injected write failure');

    expect(resources.ppg).toBe(previous);
    expect(made.state._sTree).toBe(made.oldTree);
    expect(made.state._frameResourcesGeneration).toBe(7);
    expect(previous.queryArenaBuf.destroy).not.toHaveBeenCalled();
    expect(gpu.staging).toHaveLength(4);
    for (const buffer of gpu.staging) {
      expect(buffer.destroy).toHaveBeenCalledOnce();
    }
  });
});
