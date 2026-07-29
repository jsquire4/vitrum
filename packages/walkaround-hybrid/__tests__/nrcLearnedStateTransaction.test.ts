import { describe, expect, it, vi } from 'vitest';

import { installWebGPUPolyfills } from './helpers/webgpuPolyfills.js';
import {
  NrcSubsystem,
  type NrcConfig,
} from '../src/neural/nrc/nrcSubsystem.js';
import {
  createNrcInferenceArenaLayout,
  createNrcRuntimeArenaLayout,
} from '../src/neural/nrc/nrcArena.js';
import {
  nrcStateShape,
  type NrcLearnedStateSnapshot,
} from '../src/neural/nrc/nrcStateSnapshot.js';
import { NRC_DIAGNOSTIC_BYTES } from '../src/neural/nrc/nrcDiagnostics.js';

installWebGPUPolyfills();

interface MemoryBuffer {
  readonly label: string;
  readonly size: number;
  readonly bytes: Uint8Array;
  mapState: GPUBufferMapState;
  mapAsync(mode: GPUMapModeFlags, offset?: number, size?: number): Promise<void>;
  getMappedRange(offset?: number, size?: number): ArrayBuffer;
  unmap(): void;
  destroy(): void;
}

function memoryBuffer(
  label: string,
  size: number,
  initial?: Float32Array,
): MemoryBuffer {
  const bytes = new Uint8Array(size);
  if (initial) {
    bytes.set(new Uint8Array(initial.buffer, initial.byteOffset, initial.byteLength));
  }
  return {
    label,
    size,
    bytes,
    mapState: 'unmapped',
    async mapAsync() {
      this.mapState = 'mapped';
    },
    getMappedRange(offset = 0, mappedSize = size - offset) {
      if (offset === 0 && mappedSize === size) return bytes.buffer;
      return bytes.slice(offset, offset + mappedSize).buffer;
    },
    unmap() {
      this.mapState = 'unmapped';
    },
    destroy: vi.fn(),
  };
}

const config: Partial<NrcConfig> = {
  levels: 1,
  featuresPerEntry: 1,
  tableSize: 2,
  nMin: 2,
  growth: 2,
  oneBlobBins: 1,
  width: 10,
  hidden: 0,
  spreadC: 0.01,
  recordCap: 1,
  learningRate: 0.01,
  tableLearningRate: 0.1,
  useF16: false,
  tileB: 1,
  warmupSteps: 2,
};

function learnedState(): NrcLearnedStateSnapshot {
  const resolved = new NrcSubsystem({} as GPUDevice, {} as never, config).cfg;
  const stateConfig = {
    levels: resolved.levels,
    featuresPerEntry: resolved.featuresPerEntry,
    tableSize: resolved.tableSize,
    nMin: resolved.nMin,
    growth: resolved.growth,
    oneBlobBins: resolved.oneBlobBins,
    width: resolved.width,
    hidden: resolved.hidden,
    spreadC: resolved.spreadC,
    recordCap: resolved.recordCap,
    learningRate: resolved.learningRate,
    tableLearningRate: resolved.tableLearningRate,
    useF16: resolved.useF16,
    tileB: resolved.tileB,
    warmupSteps: resolved.warmupSteps ?? 8,
  };
  const shape = nrcStateShape(stateConfig);
  return {
    config: stateConfig,
    sceneBoundsMin: [-1, -2, -3],
    sceneBoundsMax: [4, 5, 6],
    trainedSteps: 7,
    mlp: {
      weights: Float32Array.from(
        { length: shape.weightScalars },
        (_, index) => index + 0.25,
      ),
      biases: Float32Array.from(
        { length: shape.biasScalars },
        (_, index) => -index - 0.5,
      ),
      firstMomentWeights: new Float32Array(shape.weightScalars).fill(-0.1),
      secondMomentWeights: new Float32Array(shape.weightScalars).fill(0.2),
      firstMomentBiases: new Float32Array(shape.biasScalars).fill(-0.3),
      secondMomentBiases: new Float32Array(shape.biasScalars).fill(0.4),
      adamT: 11,
    },
    hashGrid: {
      tables: new Float32Array(shape.tableScalars).fill(0.01),
      firstMoment: new Float32Array(shape.tableScalars).fill(-0.02),
      secondMoment: new Float32Array(shape.tableScalars).fill(0.03),
      adamT: 13,
    },
  };
}

function memoryDevice() {
  const copies: Array<readonly [MemoryBuffer, MemoryBuffer, number, number, number]> = [];
  const encoder = {
    copyBufferToBuffer(
      source: MemoryBuffer,
      sourceOffset: number,
      destination: MemoryBuffer,
      destinationOffset: number,
      size: number,
    ) {
      copies.push([source, destination, sourceOffset, destinationOffset, size]);
      destination.bytes.set(
        source.bytes.subarray(sourceOffset, sourceOffset + size),
        destinationOffset,
      );
    },
    clearBuffer(buffer: MemoryBuffer, offset = 0, size = buffer.size - offset) {
      buffer.bytes.fill(0, offset, offset + size);
    },
    finish: vi.fn(() => ({ label: 'nrc-state-command-buffer' })),
  };
  const device = {
    createBuffer: vi.fn((descriptor: GPUBufferDescriptor) => {
      const buffer = memoryBuffer(String(descriptor.label ?? 'buffer'), Number(descriptor.size));
      if (descriptor.mappedAtCreation) buffer.mapState = 'mapped';
      return buffer;
    }),
    createCommandEncoder: vi.fn(() => encoder),
    queue: {
      submit: vi.fn(),
      onSubmittedWorkDone: vi.fn(async () => {}),
    },
  } as unknown as GPUDevice;
  return { device, encoder, copies };
}

function asGpu(buffer: MemoryBuffer): GPUBuffer {
  return buffer as unknown as GPUBuffer;
}

function installReadyState(
  subsystem: NrcSubsystem,
  state: NrcLearnedStateSnapshot,
  deviceHarness: ReturnType<typeof memoryDevice>,
) {
  const shape = nrcStateShape(state.config);
  const source = (label: string, values: Float32Array): GPUBuffer =>
    asGpu(memoryBuffer(label, values.byteLength, values));
  const oldInference = memoryBuffer('old-inference', 2048);
  const spareInference = memoryBuffer('spare-inference', 2048);
  const oldTables = source('old-tables', state.hashGrid.tables);
  const candidateWeights = memoryBuffer('candidate-weights', state.mlp.weights.byteLength);
  const candidateBiases = memoryBuffer('candidate-biases', state.mlp.biases.byteLength);
  const candidateTables = memoryBuffer('candidate-tables', state.hashGrid.tables.byteLength);
  const mlpCommit = vi.fn();
  const mlpRollback = vi.fn();
  const mlpFinalize = vi.fn();
  const tableCommit = vi.fn();
  const tableRollback = vi.fn();
  const tableFinalize = vi.fn();
  const trainer = {
    stateBuffers: vi.fn(() => ({
      weights: source('weights', state.mlp.weights),
      biases: source('biases', state.mlp.biases),
      firstMomentWeights: source('mW', state.mlp.firstMomentWeights),
      secondMomentWeights: source('vW', state.mlp.secondMomentWeights),
      firstMomentBiases: source('mB', state.mlp.firstMomentBiases),
      secondMomentBiases: source('vB', state.mlp.secondMomentBiases),
      weightScalars: shape.weightScalars,
      biasScalars: shape.biasScalars,
      adamT: state.mlp.adamT,
    })),
    prepareStateRestore: vi.fn(() => ({
      candidateWeightBuffer: asGpu(candidateWeights),
      candidateBiasBuffer: asGpu(candidateBiases),
      commitCpu: mlpCommit,
      rollback: mlpRollback,
      finalizeSuccess: mlpFinalize,
    })),
  };
  const tableTrainer = {
    stateBuffers: vi.fn(() => ({
      tables: oldTables,
      firstMoment: source('table-m', state.hashGrid.firstMoment),
      secondMoment: source('table-v', state.hashGrid.secondMoment),
      tableScalars: shape.tableScalars,
      adamT: state.hashGrid.adamT,
    })),
    prepareStateRestore: vi.fn(() => ({
      candidateTableBuffer: asGpu(candidateTables),
      commitCpu: tableCommit,
      rollback: tableRollback,
      finalizeSuccess: tableFinalize,
    })),
  };
  const inferenceLayout = createNrcInferenceArenaLayout({
    weightsBytes: state.mlp.weights.byteLength,
    biasesBytes: state.mlp.biases.byteLength,
    tablesBytes: state.hashGrid.tables.byteLength,
    levelsBytes: state.config.levels * 16,
  });
  const runtimeLayout = createNrcRuntimeArenaLayout({
    diagnosticsBytes: NRC_DIAGNOSTIC_BYTES,
    claimsBytes: 4,
    recordsBytes: 64,
  });
  Object.assign(subsystem, {
    _lifecycleState: 'ready',
    _trainer: trainer,
    _tableTrainer: tableTrainer,
    _tablesBuf: oldTables,
    _levelsBuf: asGpu(memoryBuffer('levels', state.config.levels * 16)),
    _cfgUbo: asGpu(memoryBuffer('cfg', 48)),
    _activeInferenceArena: asGpu(oldInference),
    _spareInferenceArena: asGpu(spareInference),
    _runtimeArena: asGpu(memoryBuffer('runtime', runtimeLayout.byteSize)),
    _trainerDiagnosticsBuffer: asGpu(memoryBuffer(
      'trainer-diagnostics',
      NRC_DIAGNOSTIC_BYTES,
    )),
    _inferenceLayout: inferenceLayout,
    _runtimeLayout: runtimeLayout,
    _inferenceEpoch: 2,
    _runtimeEpoch: 2,
    _generation: 4,
    _recordStride: 16,
    _trainedSteps: state.trainedSteps,
    _lastGpuDiagnostics: new Uint32Array(5).fill(9),
    _sceneBoundsMin: [...state.sceneBoundsMin],
    _sceneBoundsMax: [...state.sceneBoundsMax],
  });
  return {
    trainer,
    tableTrainer,
    oldInference,
    spareInference,
    oldTables,
    candidateTables,
    mlpCommit,
    mlpRollback,
    mlpFinalize,
    tableCommit,
    tableRollback,
    tableFinalize,
    deviceHarness,
  };
}

describe('NrcSubsystem learned-state persistence', () => {
  it('exports all nine trainable buffers through one coherent submission', async () => {
    const state = learnedState();
    const harness = memoryDevice();
    const subsystem = new NrcSubsystem(harness.device, {} as never, config);
    installReadyState(subsystem, state, harness);

    const exported = await subsystem.exportLearnedState();

    expect(exported).not.toBeNull();
    expect(exported?.trainedSteps).toBe(state.trainedSteps);
    expect(exported?.mlp.adamT).toBe(state.mlp.adamT);
    expect(exported?.hashGrid.adamT).toBe(state.hashGrid.adamT);
    expect(Array.from(exported!.mlp.weights)).toEqual(Array.from(state.mlp.weights));
    expect(Array.from(exported!.mlp.secondMomentBiases))
      .toEqual(Array.from(state.mlp.secondMomentBiases));
    expect(Array.from(exported!.hashGrid.secondMoment))
      .toEqual(Array.from(state.hashGrid.secondMoment));
    expect(harness.copies).toHaveLength(9);
    expect(harness.device.queue.submit).toHaveBeenCalledOnce();
  });

  it('publishes and rolls back the complete learned-state handle cohort', () => {
    const state = learnedState();
    const replacement = {
      ...state,
      trainedSteps: 19,
      mlp: { ...state.mlp, adamT: 23 },
      hashGrid: { ...state.hashGrid, adamT: 29 },
    };
    const deviceHarness = memoryDevice();
    const subsystem = new NrcSubsystem(deviceHarness.device, {} as never, config);
    const harness = installReadyState(subsystem, state, deviceHarness);
    const transaction = subsystem.prepareLearnedStateImport(
      deviceHarness.encoder as unknown as GPUCommandEncoder,
      replacement,
    );

    expect(transaction).not.toBeNull();
    expect((subsystem as unknown as { _tablesBuf: GPUBuffer })._tablesBuf)
      .toBe(harness.oldTables);
    transaction!.commit();
    expect(harness.mlpCommit).toHaveBeenCalledOnce();
    expect(harness.tableCommit).toHaveBeenCalledOnce();
    expect((subsystem as unknown as { _tablesBuf: GPUBuffer })._tablesBuf)
      .toBe(harness.candidateTables);
    expect((subsystem as unknown as { _activeInferenceArena: GPUBuffer })._activeInferenceArena)
      .toBe(harness.spareInference);
    expect(subsystem.diagnostics().trainedSteps).toBe(19);

    transaction!.rollback();
    expect(harness.mlpRollback).toHaveBeenCalledOnce();
    expect(harness.tableRollback).toHaveBeenCalledOnce();
    expect((subsystem as unknown as { _tablesBuf: GPUBuffer })._tablesBuf)
      .toBe(harness.oldTables);
    expect((subsystem as unknown as { _activeInferenceArena: GPUBuffer })._activeInferenceArena)
      .toBe(harness.oldInference);
    expect(subsystem.diagnostics().trainedSteps).toBe(state.trainedSteps);
  });

  it('rejects config and scene-bound mismatches before trainer preparation', () => {
    const state = learnedState();
    const deviceHarness = memoryDevice();
    const subsystem = new NrcSubsystem(deviceHarness.device, {} as never, config);
    const harness = installReadyState(subsystem, state, deviceHarness);
    const mismatch = {
      ...state,
      sceneBoundsMax: [4, 5, 7] as const,
    };

    expect(subsystem.canImportLearnedState(mismatch)).toBe(false);
    expect(subsystem.prepareLearnedStateImport(
      deviceHarness.encoder as unknown as GPUCommandEncoder,
      mismatch,
    )).toBeNull();
    expect(harness.trainer.prepareStateRestore).not.toHaveBeenCalled();
    expect(harness.tableTrainer.prepareStateRestore).not.toHaveBeenCalled();
  });
});
