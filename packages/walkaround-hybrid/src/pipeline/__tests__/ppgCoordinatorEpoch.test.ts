import { describe, expect, it, vi } from 'vitest';
import { buildSTree } from '../../ppg/sTree.js';
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
  _trainingDispatchesSinceRefine: number;
  _trainingReadbackFailures: number;
  _frameResourcesGeneration: number;
};

function gpuBuffer(size: number, label: string): GPUBuffer {
  return { size, label, destroy: vi.fn() } as unknown as GPUBuffer;
}

function resources(): FrameResources {
  const queryArenaLayout = createPpgQueryArenaLayout({
    sTreeCapacityBytes: 512,
    dTreeCapacityBytes: 512,
    dTreeOffsetsCapacityBytes: 32,
    maxSpatialCells: 8,
    maxDTreeNodesPerCell: 4,
  });
  return {
    ppg: {
      queryArenaBuf: gpuBuffer(queryArenaLayout.byteLength, 'ppg-query-arena'),
      queryArenaLayout,
      queryArenaEpoch: 1,
      fluxAtomicsBuf: gpuBuffer(128, 'ppg-fluxAtomics'),
      cellSampleCountsBuf: gpuBuffer(32, 'ppg-cellSampleCounts'),
      updateUboBuffer: gpuBuffer(16, 'ppg-update-ubo'),
    },
  } as unknown as FrameResources;
}

function deferred() {
  let resolve!: () => void;
  const promise = new Promise<void>((done) => { resolve = done; });
  return { promise, resolve };
}

function harness(fluxMap: () => Promise<void>) {
  const readbacks: Array<GPUBuffer & { destroy: ReturnType<typeof vi.fn> }> = [];
  const copyBufferToBuffer = vi.fn();
  const clearBuffer = vi.fn();
  const submit = vi.fn();
  const writeBuffer = vi.fn();
  const createBuffer = vi.fn((descriptor: GPUBufferDescriptor) => {
    const label = descriptor.label ?? '';
    if (label === 'ppg-flux-readback' || label === 'ppg-cellcount-readback') {
      const buffer = {
        size: Number(descriptor.size),
        label,
        destroy: vi.fn(),
        mapAsync: vi.fn(label === 'ppg-flux-readback' ? fluxMap : () => Promise.resolve()),
        getMappedRange: vi.fn(() => new ArrayBuffer(Number(descriptor.size))),
        unmap: vi.fn(),
      } as unknown as GPUBuffer & { destroy: ReturnType<typeof vi.fn> };
      readbacks.push(buffer);
      return buffer;
    }
    const storage = new ArrayBuffer(Number(descriptor.size));
    return {
      size: Number(descriptor.size),
      label,
      destroy: vi.fn(),
      getMappedRange: vi.fn(() => storage),
      unmap: vi.fn(),
    } as unknown as GPUBuffer;
  });
  const device = {
    createBuffer,
    createCommandEncoder: vi.fn(() => ({
      copyBufferToBuffer,
      clearBuffer,
      finish: vi.fn(() => ({})),
    })),
    queue: {
      submit,
      writeBuffer,
      onSubmittedWorkDone: vi.fn(() => Promise.resolve()),
    },
  } as unknown as GPUDevice;
  return {
    device,
    readbacks,
    submit,
    writeBuffer,
    copyBufferToBuffer,
    createBuffer,
  };
}

function coordinator(device: GPUDevice) {
  const value = new PPGCoordinator(device, { onError: () => undefined });
  const state = value as unknown as Internals;
  state._enabled = true;
  state._maxSpatialCells = 8;
  state._maxDTreeNodesPerCell = 4;
  state._sceneAABB = BOUNDS;
  state._sTree = buildSTree(BOUNDS, 0);
  return { value, state };
}

async function settle(): Promise<void> {
  for (let i = 0; i < 12; i++) await Promise.resolve();
}

describe('PPG sealed training epochs', () => {
  it('rejects fractional refine intervals without starting a GPU epoch', () => {
    const gpu = harness(() => Promise.resolve());
    const { value } = coordinator(gpu.device);
    expect(() => value.maybeRunTrainingRefine(resources(), true, 1.5)).toThrow(
      /positive safe integer/,
    );
    expect(gpu.createBuffer).not.toHaveBeenCalled();
    expect(gpu.submit).not.toHaveBeenCalled();
  });

  it('blocks deposits across retries, enters durable failed state, and recovers explicitly', async () => {
    let attempts = 0;
    const gpu = harness(() => {
      attempts++;
      return attempts <= 3
        ? Promise.reject(new Error(`readback-${attempts}`))
        : Promise.resolve();
    });
    const { value, state } = coordinator(gpu.device);
    const frame = resources();

    value.maybeRunTrainingRefine(frame, true, 1);
    await settle();
    expect(value.trainingStatus).toBe('retry-pending');
    expect(value.trainingDispatchAllowed).toBe(false);
    expect(state._trainingReadbackFailures).toBe(1);
    expect(state._trainingDispatchesSinceRefine).toBe(0);

    value.maybeRunTrainingRefine(frame, true, 1);
    await settle();
    expect(value.trainingStatus).toBe('retry-pending');
    expect(state._trainingReadbackFailures).toBe(2);
    expect(state._trainingDispatchesSinceRefine).toBe(0);

    value.maybeRunTrainingRefine(frame, true, 1);
    await settle();
    expect(value.trainingStatus).toBe('failed');
    expect(value.trainingDispatchAllowed).toBe(false);
    const submissionsAtFailure = gpu.submit.mock.calls.length;
    value.maybeRunTrainingRefine(frame, true, 1);
    expect(gpu.submit).toHaveBeenCalledTimes(submissionsAtFailure);

    expect(value.requestTrainingRecovery()).toBe(true);
    expect(value.trainingStatus).toBe('retry-pending');
    value.maybeRunTrainingRefine(frame, false, 1);
    await vi.waitFor(() => expect(value.trainingStatus).toBe('collecting'));
    expect(value.trainingDispatchAllowed).toBe(true);
    expect(state._trainingReadbackFailures).toBe(0);
    expect(value.requestTrainingRecovery()).toBe(false);
  });

  it.each(['resize', 'mutation', 'dispose'] as const)(
    'ignores a stale async readback after %s and preserves the lifecycle publication',
    async (operation) => {
      const pending = deferred();
      const gpu = harness(() => pending.promise);
      const { value, state } = coordinator(gpu.device);
      const frame = resources();
      const initialGeneration = state._frameResourcesGeneration;

      value.maybeRunTrainingRefine(frame, true, 1);
      await vi.waitFor(() => {
        const flux = gpu.readbacks.find((buffer) => buffer.label === 'ppg-flux-readback') as
          | (GPUBuffer & { mapAsync: ReturnType<typeof vi.fn> })
          | undefined;
        expect(flux?.mapAsync).toHaveBeenCalledOnce();
      });

      if (operation === 'resize') {
        value.onResize(frame, 16, 16, 0);
      } else if (operation === 'mutation') {
        value.resetForSceneBvh(
          { bvhPositions: { cpuData: new Float32Array([2, 2, 2, 0, 4, 4, 4, 0]).buffer } },
          frame,
          16,
          16,
        );
      } else {
        value.dispose();
      }
      const publishedTree = state._sTree;
      const publishedGeneration = state._frameResourcesGeneration;
      const writesAfterLifecycle = gpu.writeBuffer.mock.calls.length;
      expect(publishedGeneration).toBeGreaterThan(initialGeneration);
      expect(value.trainingStatus).toBe(operation === 'dispose' ? 'disposed' : 'collecting');

      pending.resolve();
      await settle();
      expect(state._sTree).toBe(publishedTree);
      expect(state._frameResourcesGeneration).toBe(publishedGeneration);
      expect(gpu.writeBuffer).toHaveBeenCalledTimes(writesAfterLifecycle);
      expect(value.trainingStatus).toBe(operation === 'dispose' ? 'disposed' : 'collecting');
      for (const readback of gpu.readbacks) expect(readback.destroy).toHaveBeenCalledOnce();
    },
  );
});
