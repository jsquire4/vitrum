import { describe, expect, it, vi } from 'vitest';

import { installWebGPUPolyfills } from '../../../../__tests__/helpers/webgpuPolyfills.js';
import type { BGLCache } from '../../../bglTypes.js';
import { DEFAULT_NRC_CONFIG, NrcSubsystem, type NrcConfig } from '../nrcSubsystem.js';
import { computeNrcResourceFootprint } from '../nrcPreflight.js';

installWebGPUPolyfills();

const TEST_CONFIG: Partial<NrcConfig> = {
  levels: 1,
  featuresPerEntry: 1,
  tableSize: 4,
  nMin: 2,
  growth: 2,
  oneBlobBins: 1,
  width: 16,
  hidden: 0,
  spreadC: 0.01,
  recordCap: 4,
  learningRate: 0.01,
  tableLearningRate: 0.1,
  useF16: false,
  tileB: 4,
  warmupSteps: 1,
};

const AABB_MIN = [-1, -1, -1] as const;
const AABB_MAX = [1, 1, 1] as const;

type Failure =
  | { readonly kind: 'create-buffer'; readonly ordinal: number }
  | { readonly kind: 'write-buffer'; readonly ordinal: number }
  | { readonly kind: 'pipeline'; readonly ordinal: number }
  | { readonly kind: 'bind-group-layout' }
  | { readonly kind: 'bind-group' };

class TrackedBuffer {
  readonly label: string;
  readonly size: number;
  mapState: GPUBufferMapState;
  private readonly throwOnDestroy: boolean;
  private readonly bytes: ArrayBuffer;
  readonly destroy = vi.fn(() => {
    if (this.throwOnDestroy) throw new Error('injected destroy failure');
  });
  readonly unmap = vi.fn(() => {
    this.mapState = 'unmapped';
  });

  readonly getMappedRange = vi.fn(() => this.bytes);
  readonly mapAsync = vi.fn(async () => {
    this.mapState = 'mapped';
  });

  writeFloat(index: number, value: number): void {
    new Float32Array(this.bytes)[index] = value;
  }

  constructor(label: string, size: number, throwOnDestroy = false, mapped = false) {
    this.label = label;
    this.size = size;
    this.bytes = new ArrayBuffer(size);
    this.mapState = mapped ? 'mapped' : 'unmapped';
    this.throwOnDestroy = throwOnDestroy;
  }
}

interface Harness {
  readonly device: GPUDevice;
  readonly cache: BGLCache;
  readonly buffers: TrackedBuffer[];
  readonly createBufferCount: number;
  readonly writeBufferCount: number;
  readonly pipelineCount: number;
  readonly pipelineReached: Promise<void>;
  setFailure(failure: Failure | undefined): void;
  releasePipeline(): void;
}

function makeHarness(options: {
  failure?: Failure;
  pausePipelineOrdinal?: number;
  throwOnDestroyOrdinal?: number;
} = {}): Harness {
  let failure = options.failure;
  let createBufferCount = 0;
  let writeBufferCount = 0;
  let pipelineCount = 0;
  let releasePausedPipeline: (() => void) | undefined;
  let signalPipelineReached: (() => void) | undefined;
  const pipelineReached = new Promise<void>((resolve) => {
    signalPipelineReached = resolve;
  });
  const pausedPipeline = new Promise<void>((resolve) => {
    releasePausedPipeline = resolve;
  });
  const buffers: TrackedBuffer[] = [];
  const cache: BGLCache = {};

  const injected = (stage: string, ordinal?: number): Error =>
    new Error('injected NRC initialization failure at ' + stage +
      (ordinal === undefined ? '' : ' #' + ordinal));

  const device = {
    limits: { maxComputeWorkgroupStorageSize: 32_768 },
    features: new Set<GPUFeatureName>(['shader-f16']),
    queue: {
      writeBuffer: () => {
        writeBufferCount++;
        if (failure?.kind === 'write-buffer' &&
            failure.ordinal === writeBufferCount) {
          throw injected('writeBuffer', writeBufferCount);
        }
      },
      submit: vi.fn(),
    },
    createBuffer: (descriptor: GPUBufferDescriptor) => {
      createBufferCount++;
      if (failure?.kind === 'create-buffer' &&
          failure.ordinal === createBufferCount) {
        throw injected('createBuffer', createBufferCount);
      }
      const buffer = new TrackedBuffer(
        String(descriptor.label ?? 'unlabelled-' + createBufferCount),
        Number(descriptor.size),
        options.throwOnDestroyOrdinal === createBufferCount,
        descriptor.mappedAtCreation === true,
      );
      buffers.push(buffer);
      return buffer;
    },
    createCommandEncoder: () => ({}),
    createShaderModule: () => ({
      getCompilationInfo: async () => ({ messages: [] }),
    }),
    createComputePipelineAsync: async () => {
      pipelineCount++;
      if (options.pausePipelineOrdinal === pipelineCount) {
        signalPipelineReached?.();
        await pausedPipeline;
      }
      if (failure?.kind === 'pipeline' &&
          failure.ordinal === pipelineCount) {
        throw injected('pipeline', pipelineCount);
      }
      return {
        getBindGroupLayout: () => ({}),
      };
    },
    createBindGroupLayout: () => {
      if (failure?.kind === 'bind-group-layout') {
        throw injected('bindGroupLayout');
      }
      return {};
    },
    createBindGroup: () => {
      if (failure?.kind === 'bind-group') {
        throw injected('bindGroup');
      }
      return { label: 'nrc-bind-group' };
    },
  } as unknown as GPUDevice;

  return {
    device,
    cache,
    buffers,
    get createBufferCount() { return createBufferCount; },
    get writeBufferCount() { return writeBufferCount; },
    get pipelineCount() { return pipelineCount; },
    pipelineReached,
    setFailure(next) { failure = next; },
    releasePipeline() { releasePausedPipeline?.(); },
  };
}

function makeSubsystem(harness: Harness): NrcSubsystem {
  return new NrcSubsystem(harness.device, harness.cache, TEST_CONFIG);
}

function expectAllDestroyedOnce(buffers: readonly TrackedBuffer[]): void {
  for (const buffer of buffers) {
    expect(buffer.destroy, buffer.label).toHaveBeenCalledTimes(1);
  }
}

async function expectRolledBack(failure: Failure): Promise<void> {
  const harness = makeHarness({ failure });
  const subsystem = makeSubsystem(harness);

  await expect(subsystem.initialize(AABB_MIN, AABB_MAX))
    .rejects.toThrow(/injected NRC initialization failure/);

  expect(subsystem.lifecycleState).toBe('new');
  expect(() => subsystem.queryBindings()).toThrow(/requires state 'ready'/);
  expectAllDestroyedOnce(harness.buffers);
  const readback = harness.buffers.find(
    (buffer) => buffer.label === 'nrc-records-readback',
  );
  if (readback) {
    expect(readback.unmap).toHaveBeenCalledTimes(1);
  }
}

describe('NrcSubsystem initialization transaction', () => {
  it('publishes 49 persistent buffers and peaks at 50 with one readback', async () => {
    const harness = makeHarness();
    const subsystem = makeSubsystem(harness);

    await subsystem.initialize(AABB_MIN, AABB_MAX);

    expect(subsystem.lifecycleState).toBe('ready');
    expect(subsystem.queryBindings()).toMatchObject({
      inferenceArenaBuffer: { label: 'nrc-inference-arena-active' },
      runtimeArenaBuffer: { label: 'nrc-runtime-arena' },
      configBuffer: { label: 'nrc-cfg' },
    });
    expect(harness.createBufferCount).toBe(49);
    const resolvedConfig: NrcConfig = { ...DEFAULT_NRC_CONFIG, ...TEST_CONFIG };
    const footprint = computeNrcResourceFootprint(resolvedConfig);
    const expectedSizes = Object.values(footprint.persistentAllocations)
      .flatMap((entry) => Array<number>(entry.count).fill(entry.bytesEach))
      .sort((a, b) => a - b);
    expect(harness.buffers.map((buffer) => buffer.size).sort((a, b) => a - b))
      .toEqual(expectedSizes);
    expect(harness.buffers.reduce((sum, buffer) => sum + buffer.size, 0))
      .toBe(footprint.persistentBufferBytes);
    expect(subsystem.diagnostics()).toMatchObject({
      persistentBufferCount: footprint.persistentBufferCount,
      persistentBufferBytes: footprint.persistentBufferBytes,
      peakResidentBufferCount: footprint.peakResidentBufferCount,
      peakResidentBufferBytes: footprint.peakResidentBufferBytes,
    });
    expect(harness.writeBufferCount).toBe(14);
    expect(harness.pipelineCount).toBe(7);
    for (const buffer of harness.buffers) {
      expect(buffer.destroy, buffer.label).not.toHaveBeenCalled();
    }

    subsystem.recordCopyForReadback({
      copyBufferToBuffer: vi.fn(),
    } as unknown as GPUCommandEncoder);
    expect(harness.createBufferCount).toBe(50);
    expect(harness.buffers.at(-1)?.size).toBe(footprint.readbackBytes);
    expect(harness.buffers.reduce((sum, buffer) => sum + buffer.size, 0))
      .toBe(footprint.peakResidentBufferBytes);
    subsystem.dispose();
    expect(subsystem.lifecycleState).toBe('disposed');
    expectAllDestroyedOnce(harness.buffers);
  });

  it('pins the f16 51-persistent to 52-readback transition and exact bytes', async () => {
    const harness = makeHarness();
    const cfg: NrcConfig = {
      ...DEFAULT_NRC_CONFIG,
      ...TEST_CONFIG,
      useF16: true,
    };
    const footprint = computeNrcResourceFootprint(cfg);
    const subsystem = new NrcSubsystem(harness.device, harness.cache, cfg);

    await subsystem.initialize(AABB_MIN, AABB_MAX);
    expect(harness.createBufferCount).toBe(51);
    expect(harness.buffers.reduce((sum, buffer) => sum + buffer.size, 0))
      .toBe(footprint.persistentBufferBytes);

    subsystem.recordCopyForReadback({
      copyBufferToBuffer: vi.fn(),
    } as unknown as GPUCommandEncoder);
    expect(harness.createBufferCount).toBe(52);
    expect(harness.buffers.at(-1)?.size).toBe(footprint.readbackBytes);
    expect(harness.buffers.reduce((sum, buffer) => sum + buffer.size, 0))
      .toBe(footprint.peakResidentBufferBytes);

    subsystem.dispose();
    expectAllDestroyedOnce(harness.buffers);
  });

  it('records repeated two-generation trainer windows without allocating another GPU buffer', async () => {
    const harness = makeHarness();
    const subsystem = makeSubsystem(harness);
    await subsystem.initialize(AABB_MIN, AABB_MAX);
    const allocationCount = harness.createBufferCount;

    const internals = subsystem as unknown as {
      _trainer: {
        setBatch(x: Float32Array, y: Float32Array): void;
        recordTrainStep(
          encoder: GPUCommandEncoder,
          learningRate: number,
          activeSamples: number,
        ): {
          commitCpu(): void;
          rollback(): void;
          finalizeSuccess(): void;
        } | null;
      };
      _tableTrainer: {
        recordStep(
          encoder: GPUCommandEncoder,
          positions: Float32Array,
          activeSamples: number,
        ): {
          commitCpu(): void;
          rollback(): void;
          finalizeSuccess(): void;
        };
      };
      _batchX: Float32Array;
      _batchY: Float32Array;
      _batchPos: Float32Array;
    };
    const pass = {
      setPipeline: vi.fn(),
      setBindGroup: vi.fn(),
      dispatchWorkgroups: vi.fn(),
      end: vi.fn(),
    };
    const encoder = {
      copyBufferToBuffer: vi.fn(),
      clearBuffer: vi.fn(),
      beginComputePass: vi.fn(() => pass),
    } as unknown as GPUCommandEncoder;

    internals._trainer.setBatch(internals._batchX, internals._batchY);
    for (let epoch = 0; epoch < 4; epoch++) {
      const mlp = internals._trainer.recordTrainStep(encoder, 0.01, 4);
      expect(mlp).not.toBeNull();
      const table = internals._tableTrainer.recordStep(
        encoder,
        internals._batchPos,
        4,
      );
      expect(harness.createBufferCount).toBe(allocationCount);
      mlp!.commitCpu();
      table.commitCpu();
      mlp!.finalizeSuccess();
      table.finalizeSuccess();
    }

    expect(harness.createBufferCount).toBe(49);
    subsystem.dispose();
    expectAllDestroyedOnce(harness.buffers);
  });

  it('rejects an undersized aggregate budget before the first GPU allocation', async () => {
    const harness = makeHarness();
    const base: NrcConfig = { ...DEFAULT_NRC_CONFIG, ...TEST_CONFIG };
    const peak = computeNrcResourceFootprint(base).peakResidentBufferBytes;
    const subsystem = new NrcSubsystem(harness.device, harness.cache, {
      ...TEST_CONFIG,
      maxNrcResidentBytes: peak - 1,
    });

    await expect(subsystem.initialize(AABB_MIN, AABB_MAX))
      .rejects.toThrow(/host maxNrcResidentBytes budget/);
    expect(harness.createBufferCount).toBe(0);
    expect(harness.buffers).toHaveLength(0);
    expect(subsystem.lifecycleState).toBe('new');
  });

  it('rejects an internal training failure after rolling back the candidate generation', async () => {
    const harness = makeHarness();
    const subsystem = makeSubsystem(harness);
    await subsystem.initialize(AABB_MIN, AABB_MAX);
    subsystem.recordCopyForReadback({
      copyBufferToBuffer: vi.fn(),
    } as unknown as GPUCommandEncoder);
    const readback = harness.buffers.at(-1)!;
    // One non-zero encoded lane marks slot zero as a real (dark-target) record.
    readback.writeFloat(0, 1);

    const rollback = vi.fn();
    const raw = new Error('table trainer setup failed');
    const internals = subsystem as unknown as {
      _trainer: {
        recordTrainStep: (...args: unknown[]) => unknown;
      };
      _tableTrainer: {
        recordStep: (...args: unknown[]) => unknown;
      };
    };
    internals._trainer.recordTrainStep = vi.fn(() => ({
      candidateWeightBuffer: {} as GPUBuffer,
      candidateBiasBuffer: {} as GPUBuffer,
      commitCpu: vi.fn(),
      rollback,
      finalizeSuccess: vi.fn(),
    }));
    internals._tableTrainer.recordStep = vi.fn(() => {
      throw raw;
    });

    await expect(subsystem.trainFromRecords()).rejects.toBe(raw);
    expect(rollback).toHaveBeenCalledOnce();
    expect(subsystem.diagnostics().trainingFailures).toBe(1);
    expect(readback.destroy).toHaveBeenCalledOnce();

    // Failure returns the readback state to idle; the next frame can reserve a
    // fresh generation-tagged ticket instead of wedging training permanently.
    subsystem.recordCopyForReadback({
      copyBufferToBuffer: vi.fn(),
    } as unknown as GPUCommandEncoder);
    expect(harness.createBufferCount).toBe(51);
    subsystem.dispose();
  });

  it('rolls back after every trainer, table-trainer, and subsystem buffer allocation', async () => {
    for (let ordinal = 1; ordinal <= 49; ordinal++) {
      await expectRolledBack({ kind: 'create-buffer', ordinal });
    }
  });

  it('rolls back after every initialization queue upload', async () => {
    for (let ordinal = 1; ordinal <= 14; ordinal++) {
      await expectRolledBack({ kind: 'write-buffer', ordinal });
    }
  });

  it('rolls back after every trainer and table-trainer pipeline build', async () => {
    for (let ordinal = 1; ordinal <= 7; ordinal++) {
      await expectRolledBack({ kind: 'pipeline', ordinal });
    }
  });

  it('rolls back trainer bind-group creation failures', async () => {
    await expectRolledBack({ kind: 'bind-group' });
  });

  it('continues cleanup when a trainer, external, or table buffer destroy throws', async () => {
    for (const throwOnDestroyOrdinal of [1, 33, 34, 46, 49]) {
      const harness = makeHarness({
        failure: { kind: 'bind-group' },
        throwOnDestroyOrdinal,
      });
      const subsystem = makeSubsystem(harness);

      await expect(subsystem.initialize(AABB_MIN, AABB_MAX))
        .rejects.toThrow(/injected NRC initialization failure at bindGroup/);
      expect(subsystem.lifecycleState).toBe('new');
      expectAllDestroyedOnce(harness.buffers);
    }
  });

  it('allows a clean retry after a transient initialization failure', async () => {
    const harness = makeHarness({ failure: { kind: 'bind-group' } });
    const subsystem = makeSubsystem(harness);

    await expect(subsystem.initialize(AABB_MIN, AABB_MAX)).rejects.toThrow();
    const failedBuffers = [...harness.buffers];
    expectAllDestroyedOnce(failedBuffers);
    expect(subsystem.lifecycleState).toBe('new');

    harness.setFailure(undefined);
    await subsystem.initialize(AABB_MIN, AABB_MAX);

    expect(subsystem.lifecycleState).toBe('ready');
    const liveBuffers = harness.buffers.slice(failedBuffers.length);
    expect(liveBuffers).toHaveLength(49);
    for (const buffer of liveBuffers) {
      expect(buffer.destroy, buffer.label).not.toHaveBeenCalled();
    }

    subsystem.dispose();
    expectAllDestroyedOnce(liveBuffers);
  });

  it('keeps dispose terminal when it races an asynchronous trainer build', async () => {
    const harness = makeHarness({ pausePipelineOrdinal: 1 });
    const subsystem = makeSubsystem(harness);
    const initialization = subsystem.initialize(AABB_MIN, AABB_MAX);
    const rejection = expect(initialization).rejects.toThrow(
      /disposed during initialize.*terminal/,
    );

    await harness.pipelineReached;
    subsystem.dispose();
    expect(subsystem.lifecycleState).toBe('disposed');
    harness.releasePipeline();

    await rejection;
    expect(subsystem.lifecycleState).toBe('disposed');
    expectAllDestroyedOnce(harness.buffers);
    await expect(subsystem.initialize(AABB_MIN, AABB_MAX))
      .rejects.toThrow(/disposed is terminal/);
  });
});
