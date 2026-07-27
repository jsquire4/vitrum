import { describe, expect, it, vi } from 'vitest';
import type { EngineWarning } from '@vitrum/core';
import { dTreeAccumulateFlux } from '../../ppg/dTree.js';
import { buildSTree } from '../../ppg/sTree.js';
import { serialiseSTree } from '../../ppg/serialise.js';
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
  _mergeFluxAndRefine(
    rawFlux: Float32Array,
    cellCounts: Uint32Array,
    frameResources: FrameResources,
    maxSpatialCells: number,
    maxDTreeNodesPerCell: number,
  ): void;
};

type TrackedBuffer = GPUBuffer & {
  destroy: ReturnType<typeof vi.fn>;
  getMappedRange?: ReturnType<typeof vi.fn>;
  unmap?: ReturnType<typeof vi.fn>;
};

function destination(size: number): GPUBuffer {
  return { size, destroy: vi.fn() } as unknown as GPUBuffer;
}

function frameResources(): FrameResources {
  const queryArenaLayout = createPpgQueryArenaLayout({
    sTreeCapacityBytes: 512,
    dTreeCapacityBytes: 512,
    dTreeOffsetsCapacityBytes: 32,
    maxSpatialCells: 8,
    maxDTreeNodesPerCell: 4,
  });
  return {
    ppg: {
      queryArenaBuf: destination(queryArenaLayout.byteLength),
      queryArenaLayout,
      queryArenaEpoch: 1,
      fluxAtomicsBuf: destination(128),
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
  failFinish?: boolean;
  failSubmit?: boolean;
  failWorkDoneSync?: boolean;
  failWriteAt?: number;
  submittedWork?: Promise<void>;
  onSubmit?: () => void;
} = {}) {
  const staging: TrackedBuffer[] = [];
  const copies: unknown[][] = [];
  const clears: unknown[][] = [];
  let allocations = 0;
  let copyCalls = 0;
  let writeCalls = 0;
  const encoder = {
    copyBufferToBuffer: vi.fn((...args: unknown[]) => {
      copyCalls++;
      if (copyCalls === options.failCopyAt) throw new Error('injected copy failure');
      copies.push(args);
    }),
    clearBuffer: vi.fn((...args: unknown[]) => { clears.push(args); }),
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
    writeBuffer: vi.fn(() => {
      writeCalls++;
      if (writeCalls === options.failWriteAt) {
        throw new Error('injected write failure');
      }
    }),
  };
  const device = {
    createBuffer: vi.fn((descriptor: GPUBufferDescriptor) => {
      allocations++;
      if (allocations === options.failAllocationAt) {
        throw new Error('injected allocation failure');
      }
      const mapped = new ArrayBuffer(Number(descriptor.size));
      const buffer = {
        size: Number(descriptor.size),
        getMappedRange: vi.fn(() => mapped),
        unmap: vi.fn(),
        destroy: vi.fn(),
      } as unknown as TrackedBuffer;
      staging.push(buffer);
      return buffer;
    }),
    createCommandEncoder: vi.fn(() => encoder),
    queue,
  } as unknown as GPUDevice;
  return { device, staging, copies, clears, encoder, queue };
}

async function flushMicrotasks(): Promise<void> {
  await Promise.resolve();
  await Promise.resolve();
}

describe('PPG snapshot import transaction', () => {
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

  it.each([
    ['allocation', { failAllocationAt: 2 }],
    ['encode', { failCopyAt: 2 }],
    ['finish', { failFinish: true }],
    ['submit', { failSubmit: true }],
  ] as const)('retains both guides and destroys each staging buffer once on %s failure', (_label, options) => {
    const warnings: EngineWarning[] = [];
    const gpu = uploadDevice(options);
    const { value, state, oldTree } = coordinator(gpu.device, warnings);

    expect(value.importSTree(snapshot(), frameResources())).toBe(false);
    expect(state._sTree).toBe(oldTree);
    expect(state._frameResourcesGeneration).toBe(7);
    for (const buffer of gpu.staging) expect(buffer.destroy).toHaveBeenCalledOnce();
    expect(warnings.at(-1)?.code).toBe('walkaround-hybrid.ppg-import-upload-failed');
  });

  it('submits one complete replacement before publishing CPU state and releases staging after completion', async () => {
    const completion = deferred();
    const observed: { state?: Internals; oldTree?: STree } = {};
    const gpu = uploadDevice({
      submittedWork: completion.promise,
      onSubmit: () => {
        expect(observed.state?._sTree).toBe(observed.oldTree);
        expect(observed.state?._frameResourcesGeneration).toBe(7);
      },
    });
    const made = coordinator(gpu.device);
    const state = made.state;
    const oldTree = made.oldTree;
    observed.state = state;
    observed.oldTree = oldTree;

    expect(made.value.importSTree(snapshot(), frameResources())).toBe(true);
    expect(gpu.queue.submit).toHaveBeenCalledOnce();
    expect(gpu.copies).toHaveLength(4);
    // The arena header/epoch is the final copy, after all three payload ranges.
    expect(gpu.copies[3]?.[3]).toBe(0);
    expect(gpu.clears).toHaveLength(2);
    expect(state._sTree).not.toBe(oldTree);
    expect(state._sTree?.dTrees[0]?.totalFlux).toBe(9);
    expect(state._frameResourcesGeneration).toBe(8);
    for (const buffer of gpu.staging) expect(buffer.destroy).not.toHaveBeenCalled();

    completion.resolve();
    await flushMicrotasks();
    for (const buffer of gpu.staging) expect(buffer.destroy).toHaveBeenCalledOnce();
  });

  it('keeps a published import successful and releases staging when completion tracking throws', () => {
    const gpu = uploadDevice({ failWorkDoneSync: true });
    const { value, state, oldTree } = coordinator(gpu.device);

    expect(value.importSTree(snapshot(), frameResources())).toBe(true);
    expect(gpu.queue.submit).toHaveBeenCalledOnce();
    expect(state._sTree).not.toBe(oldTree);
    expect(state._sTree?.dTrees[0]?.totalFlux).toBe(9);
    expect(state._frameResourcesGeneration).toBe(8);
    for (const buffer of gpu.staging) {
      expect(buffer.destroy).toHaveBeenCalledOnce();
    }
  });
});

describe('PPG refine publication transaction', () => {
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

describe('PPG coordinator initialization transaction', () => {
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
