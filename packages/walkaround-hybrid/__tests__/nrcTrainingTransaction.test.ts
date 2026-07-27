import { describe, expect, it, vi, type Mock } from 'vitest';

import { installWebGPUPolyfills } from './helpers/webgpuPolyfills.js';
import { NrcSubsystem } from '../src/neural/nrc/nrcSubsystem.js';
import { createNrcInferenceArenaLayout } from '../src/neural/nrc/nrcArena.js';

installWebGPUPolyfills();

interface Deferred {
  readonly promise: Promise<void>;
  resolve(): void;
  reject(error: unknown): void;
}

function deferred(): Deferred {
  let resolve!: () => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<void>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

interface TrackedBuffer {
  readonly label: string;
  mapState: GPUBufferMapState;
  mapAsync: Mock<[], Promise<void>>;
  getMappedRange: Mock<[], ArrayBuffer>;
  unmap: Mock<[], void>;
  destroy: Mock<[], void>;
}

function trackedBuffer(label: string, mapped?: ArrayBuffer): TrackedBuffer {
  const buffer: TrackedBuffer = {
    label,
    mapState: 'unmapped',
    mapAsync: vi.fn(async () => {}),
    getMappedRange: vi.fn(() => mapped ?? new ArrayBuffer(4)),
    unmap: vi.fn(() => { buffer.mapState = 'unmapped'; }),
    destroy: vi.fn(),
  };
  return buffer;
}

interface NumericModelState {
  parameters: Uint32Array;
  moments: Uint32Array;
  table: Uint32Array;
}

function modelChecksum(state: NumericModelState): number {
  let hash = 0x811c9dc5;
  for (const words of [state.parameters, state.moments, state.table]) {
    for (const word of words) hash = Math.imul(hash ^ word, 0x01000193) >>> 0;
  }
  return hash;
}

interface TransactionHarnessOptions {
  readonly mapWait?: Promise<void>;
  readonly completion?: Promise<void>;
  readonly gateError?: Error;
  readonly retirementError?: Error;
}

function makeTransactionHarness(options: TransactionHarnessOptions = {}) {
  const events: string[] = [];
  const mapped = new ArrayBuffer(48);
  new Float32Array(mapped, 0, 7).set([1, 2, 3, 4, 5, 6, 7]);
  new Uint32Array(mapped, 28, 5).set([9, 8, 7, 6, 5]);
  const readback = trackedBuffer('readback', mapped);
  readback.mapState = 'pending';
  readback.mapAsync = vi.fn(async () => {
    await (options.mapWait ?? Promise.resolve());
    readback.mapState = 'mapped';
  });

  const oldTables = trackedBuffer('old-tables');
  const oldInference = trackedBuffer('old-inference');
  const candidateInference = trackedBuffer('candidate-inference');
  const runtimeArena = trackedBuffer('runtime-arena');
  const candidateWeight = trackedBuffer('candidate-weight');
  const candidateBias = trackedBuffer('candidate-bias');
  const candidateTable = trackedBuffer('candidate-table');
  const oldNumericState: NumericModelState = {
    parameters: new Uint32Array([0x3f800000, 0x40000000, 0x40400000]),
    moments: new Uint32Array([0, 1, 2, 3]),
    table: new Uint32Array([11, 12, 13, 14]),
  };
  const candidateNumericState: NumericModelState = {
    parameters: oldNumericState.parameters.slice(),
    moments: oldNumericState.moments.slice(),
    table: oldNumericState.table.slice(),
  };
  candidateNumericState.parameters[1] = 0x40800000;
  candidateNumericState.moments[2] = 0x3dcccccd;
  candidateNumericState.table[3] = 99;
  const liveNumericState: NumericModelState = {
    parameters: oldNumericState.parameters,
    moments: oldNumericState.moments,
    table: oldNumericState.table,
  };

  let mlpClosed = false;
  let mlpCommitted = false;
  const mlpCommit = vi.fn(() => {
    events.push('mlp-commit');
    liveNumericState.parameters = candidateNumericState.parameters;
    liveNumericState.moments = candidateNumericState.moments;
    mlpCommitted = true;
  });
  const mlpRollback = vi.fn(() => {
    events.push('mlp-rollback');
    if (mlpClosed) return;
    if (mlpCommitted) {
      liveNumericState.parameters = oldNumericState.parameters;
      liveNumericState.moments = oldNumericState.moments;
    }
    mlpClosed = true;
    candidateWeight.destroy();
    candidateBias.destroy();
  });
  const mlpFinalize = vi.fn(() => {
    events.push('mlp-retire');
    if (options.retirementError) throw options.retirementError;
    mlpClosed = true;
  });
  const mlpTransaction = {
    candidateWeightBuffer: candidateWeight as unknown as GPUBuffer,
    candidateBiasBuffer: candidateBias as unknown as GPUBuffer,
    commitCpu: mlpCommit,
    rollback: mlpRollback,
    finalizeSuccess: mlpFinalize,
  };

  let tableClosed = false;
  let tableCommitted = false;
  const tableCommit = vi.fn(() => {
    events.push('table-commit');
    liveNumericState.table = candidateNumericState.table;
    tableCommitted = true;
  });
  const tableRollback = vi.fn(() => {
    events.push('table-rollback');
    if (tableClosed) return;
    if (tableCommitted) liveNumericState.table = oldNumericState.table;
    tableClosed = true;
    candidateTable.destroy();
  });
  const tableFinalize = vi.fn(() => {
    events.push('table-retire');
    tableClosed = true;
  });
  const tableTransaction = {
    candidateTableBuffer: candidateTable as unknown as GPUBuffer,
    commitCpu: tableCommit,
    rollback: tableRollback,
    finalizeSuccess: tableFinalize,
  };

  const encoder = {
    copyBufferToBuffer: vi.fn(),
    finish: vi.fn(() => ({ label: 'combined-command-buffer' })),
  };
  const submit = vi.fn(() => { events.push('submit'); });
  const onSubmittedWorkDone = vi.fn(() => options.completion ?? Promise.resolve());
  const writeBuffer = vi.fn(() => {
    events.push('gate-write');
    if (options.gateError) throw options.gateError;
  });
  const headerStaging: TrackedBuffer[] = [];
  const createBuffer = vi.fn((descriptor: GPUBufferDescriptor) => {
    const buffer = trackedBuffer(
      String(descriptor.label ?? 'buffer'),
      new ArrayBuffer(Number(descriptor.size)),
    );
    if (descriptor.mappedAtCreation) buffer.mapState = 'mapped';
    headerStaging.push(buffer);
    return buffer;
  });
  const device = {
    queue: { submit, onSubmittedWorkDone, writeBuffer },
    createCommandEncoder: vi.fn(() => encoder),
    createBuffer,
  } as unknown as GPUDevice;

  const trainer = {
    wMaster: new Float32Array(1),
    bMaster: new Float32Array(1),
    setBatch: vi.fn(),
    recordTrainStep: vi.fn(() => {
      events.push('mlp-record');
      return mlpTransaction;
    }),
    dispose: vi.fn(() => { mlpRollback(); }),
  };
  const tableTrainer = {
    recordStep: vi.fn(() => {
      events.push('table-record');
      return tableTransaction;
    }),
    dispose: vi.fn(() => { tableRollback(); }),
  };

  const subsystem = new NrcSubsystem(
    device,
    { nrc: { label: 'nrc-layout' } } as never,
    { recordCap: 1 },
  );
  const inferenceLayout = createNrcInferenceArenaLayout({
    weightsBytes: trainer.wMaster.byteLength,
    biasesBytes: trainer.bMaster.byteLength,
    tablesBytes: subsystem.cfg.levels * subsystem.cfg.tableSize
      * subsystem.cfg.featuresPerEntry * Float32Array.BYTES_PER_ELEMENT,
    levelsBytes: subsystem.cfg.levels * 16,
  });
  const ticket = { buffer: readback, generation: 4, sequence: 1, destroyed: false };
  Object.assign(subsystem, {
    _lifecycleState: 'ready',
    _readbackState: { kind: 'copy-recorded', ticket },
    _generation: 4,
    _staleReadbacks: 0,
    _trainingFailures: 0,
    _trainedSteps: 0,
    _recordStride: 7,
    _recordByteSize: 28,
    _readbackByteSize: 48,
    _inW: 1,
    _lastGpuDiagnostics: new Uint32Array(5),
    _batchX: new Float32Array(1),
    _batchY: new Float32Array(3),
    _batchPos: new Float32Array(3),
    _trainer: trainer,
    _tableTrainer: tableTrainer,
    _tablesBuf: oldTables,
    _levelsBuf: trackedBuffer('levels'),
    _cfgUbo: trackedBuffer('cfg'),
    _activeInferenceArena: oldInference,
    _spareInferenceArena: candidateInference,
    _runtimeArena: runtimeArena,
    _inferenceLayout: inferenceLayout,
    _inferenceEpoch: 1,
  });

  return {
    subsystem,
    events,
    readback,
    oldTables,
    oldInference,
    candidateInference,
    runtimeArena,
    candidateWeight,
    candidateBias,
    candidateTable,
    mlpTransaction,
    tableTransaction,
    trainer,
    tableTrainer,
    encoder,
    submit,
    onSubmittedWorkDone,
    oldNumericState,
    candidateNumericState,
    liveNumericState,
    writeBuffer,
    createBuffer,
    headerStaging,
  };
}

function internal(subsystem: NrcSubsystem): Record<string, unknown> {
  return subsystem as unknown as Record<string, unknown>;
}

async function reachSubmission(
  harness: ReturnType<typeof makeTransactionHarness>,
): Promise<void> {
  for (let turn = 0; turn < 8 && harness.submit.mock.calls.length === 0; turn++) {
    await Promise.resolve();
  }
  expect(harness.submit).toHaveBeenCalledOnce();
}

describe('NrcSubsystem training transaction', () => {
  it('rolls back both candidates when submitted work completion rejects', async () => {
    const completion = Promise.reject(new Error('device lost after submit'));
    completion.catch(() => {});
    const harness = makeTransactionHarness({ completion });
    const oldChecksum = modelChecksum(harness.oldNumericState);
    expect(modelChecksum(harness.candidateNumericState)).not.toBe(oldChecksum);

    await expect(harness.subsystem.trainFromRecords())
      .rejects.toThrow('device lost after submit');

    expect(harness.submit).toHaveBeenCalledOnce();
    expect(harness.mlpTransaction.commitCpu).not.toHaveBeenCalled();
    expect(harness.tableTransaction.commitCpu).not.toHaveBeenCalled();
    expect(harness.mlpTransaction.rollback).toHaveBeenCalledOnce();
    expect(harness.tableTransaction.rollback).toHaveBeenCalledOnce();
    expect(internal(harness.subsystem)._tablesBuf).toBe(harness.oldTables);
    expect(internal(harness.subsystem)._activeInferenceArena).toBe(harness.oldInference);
    expect(harness.subsystem.diagnostics().trainedSteps).toBe(0);
    expect(harness.subsystem.diagnostics().trainingFailures).toBe(1);
    expect(modelChecksum(harness.oldNumericState)).toBe(oldChecksum);
    expect(modelChecksum(harness.liveNumericState)).toBe(oldChecksum);
    expect(harness.candidateWeight.destroy).toHaveBeenCalledOnce();
    expect(harness.candidateBias.destroy).toHaveBeenCalledOnce();
    expect(harness.candidateTable.destroy).toHaveBeenCalledOnce();
    expect(harness.readback.unmap).toHaveBeenCalledOnce();
    expect(harness.readback.destroy).toHaveBeenCalledOnce();
    expect(internal(harness.subsystem)._readbackState).toEqual({ kind: 'idle' });
    expect(Array.from(internal(harness.subsystem)._lastGpuDiagnostics as Uint32Array))
      .toEqual([9, 8, 7, 6, 5]);
  });

  it('discards a completed submission when mutation changes its generation', async () => {
    const completion = deferred();
    const harness = makeTransactionHarness({ completion: completion.promise });
    const training = harness.subsystem.trainFromRecords();
    await reachSubmission(harness);

    Object.assign(harness.subsystem, { _generation: 5 });
    completion.resolve();
    await training;

    expect(harness.mlpTransaction.commitCpu).not.toHaveBeenCalled();
    expect(harness.tableTransaction.commitCpu).not.toHaveBeenCalled();
    expect(harness.mlpTransaction.rollback).toHaveBeenCalledOnce();
    expect(harness.tableTransaction.rollback).toHaveBeenCalledOnce();
    expect(internal(harness.subsystem)._tablesBuf).toBe(harness.oldTables);
    expect(internal(harness.subsystem)._activeInferenceArena).toBe(harness.oldInference);
    expect(harness.subsystem.diagnostics().staleReadbacks).toBe(1);
    expect(harness.writeBuffer).not.toHaveBeenCalled();
  });

  it('cancels a pending map on dispose and destroys its ticket exactly once', async () => {
    const mapWait = deferred();
    const harness = makeTransactionHarness({ mapWait: mapWait.promise });
    const training = harness.subsystem.trainFromRecords();
    await Promise.resolve();

    harness.subsystem.dispose();
    mapWait.resolve();
    await training;

    expect(harness.trainer.setBatch).not.toHaveBeenCalled();
    expect(harness.submit).not.toHaveBeenCalled();
    expect(harness.readback.unmap).toHaveBeenCalledOnce();
    expect(harness.readback.destroy).toHaveBeenCalledOnce();
    expect(harness.subsystem.lifecycleState).toBe('disposed');
  });

  it('disposes in-flight candidates once while a submission is pending', async () => {
    const completion = deferred();
    const harness = makeTransactionHarness({ completion: completion.promise });
    const training = harness.subsystem.trainFromRecords();
    await reachSubmission(harness);

    harness.subsystem.dispose();
    completion.resolve();
    await training;

    expect(harness.mlpTransaction.commitCpu).not.toHaveBeenCalled();
    expect(harness.tableTransaction.commitCpu).not.toHaveBeenCalled();
    expect(harness.candidateWeight.destroy).toHaveBeenCalledOnce();
    expect(harness.candidateBias.destroy).toHaveBeenCalledOnce();
    expect(harness.candidateTable.destroy).toHaveBeenCalledOnce();
    expect(harness.readback.unmap).toHaveBeenCalledOnce();
    expect(harness.readback.destroy).toHaveBeenCalledOnce();
    expect(harness.subsystem.lifecycleState).toBe('disposed');
  });

  it('keeps old state bit-identical when the final gate write fails', async () => {
    const harness = makeTransactionHarness({ gateError: new Error('gate write failed') });

    await expect(harness.subsystem.trainFromRecords())
      .rejects.toThrow('gate write failed');

    expect(harness.events).toEqual([
      'mlp-record',
      'table-record',
      'submit',
      'gate-write',
      'mlp-rollback',
      'table-rollback',
    ]);
    expect(harness.mlpTransaction.commitCpu).not.toHaveBeenCalled();
    expect(harness.tableTransaction.commitCpu).not.toHaveBeenCalled();
    expect(internal(harness.subsystem)._tablesBuf).toBe(harness.oldTables);
    expect(internal(harness.subsystem)._activeInferenceArena).toBe(harness.oldInference);
    expect(harness.subsystem.diagnostics().trainedSteps).toBe(0);
  });

  it('publishes once, reports retirement cleanup failures, and never rolls back live state', async () => {
    const harness = makeTransactionHarness({
      retirementError: new Error('old-buffer retirement failed'),
    });

    await expect(harness.subsystem.trainFromRecords())
      .rejects.toThrow(/failed to retire one or more previous resources/);

    expect(harness.events).toEqual([
      'mlp-record',
      'table-record',
      'submit',
      'gate-write',
      'mlp-commit',
      'table-commit',
      'mlp-retire',
      'table-retire',
    ]);
    expect(harness.encoder.finish).toHaveBeenCalledOnce();
    expect(harness.submit).toHaveBeenCalledOnce();
    expect(harness.mlpTransaction.rollback).not.toHaveBeenCalled();
    expect(harness.tableTransaction.rollback).not.toHaveBeenCalled();
    expect(internal(harness.subsystem)._tablesBuf).toBe(harness.candidateTable);
    expect(internal(harness.subsystem)._activeInferenceArena).toBe(harness.candidateInference);
    expect(internal(harness.subsystem)._spareInferenceArena).toBe(harness.oldInference);
    expect(harness.subsystem.diagnostics().trainedSteps).toBe(1);
    expect(modelChecksum(harness.liveNumericState)).toBe(
      modelChecksum(harness.candidateNumericState),
    );
    expect(modelChecksum(harness.liveNumericState)).not.toBe(
      modelChecksum(harness.oldNumericState),
    );
    expect(harness.subsystem.diagnostics().trainingFailures).toBe(1);
    expect(harness.candidateWeight.destroy).not.toHaveBeenCalled();
    expect(harness.candidateBias.destroy).not.toHaveBeenCalled();
    expect(harness.candidateTable.destroy).not.toHaveBeenCalled();
    expect(internal(harness.subsystem)._readbackState).toEqual({ kind: 'idle' });
    expect(Array.from(internal(harness.subsystem)._lastGpuDiagnostics as Uint32Array))
      .toEqual([9, 8, 7, 6, 5]);
  });
});

describe('NrcSubsystem readback memory bound', () => {
  it('keeps at most one staging allocation live across repeated epochs', async () => {
    let live = 0;
    let peak = 0;
    const readbacks: TrackedBuffer[] = [];
    const createBuffer = vi.fn((descriptor: GPUBufferDescriptor) => {
      live++;
      peak = Math.max(peak, live);
      const mapped = new ArrayBuffer(Number(descriptor.size));
      const buffer = trackedBuffer(String(descriptor.label), mapped);
      let destroyed = false;
      buffer.mapAsync = vi.fn(async () => { buffer.mapState = 'mapped'; });
      buffer.destroy = vi.fn(() => {
        if (destroyed) return;
        destroyed = true;
        live--;
      });
      readbacks.push(buffer);
      return buffer;
    });
    const device = { createBuffer } as unknown as GPUDevice;
    const subsystem = new NrcSubsystem(device, {} as never, { recordCap: 1 });
    Object.assign(subsystem, {
      _lifecycleState: 'ready',
      _readbackState: { kind: 'idle' },
      _generation: 1,
      _recordStride: 7,
      _recordByteSize: 28,
      _readbackByteSize: 48,
      _inW: 1,
      _lastGpuDiagnostics: new Uint32Array(5),
      _batchX: new Float32Array(1),
      _batchY: new Float32Array(3),
      _batchPos: new Float32Array(3),
      _runtimeArena: trackedBuffer('runtime-arena'),
      _runtimeLayout: {
        recordsByteOffset: 256,
        diagnosticsByteOffset: 512,
      },
      _trainer: {},
      _tableTrainer: {},
    });
    const copyBufferToBuffer = vi.fn();
    const encoder = { copyBufferToBuffer } as unknown as GPUCommandEncoder;

    for (let epoch = 0; epoch < 32; epoch++) {
      subsystem.recordCopyForReadback(encoder);
      await subsystem.trainFromRecords();
      expect(internal(subsystem)._readbackState).toEqual({ kind: 'idle' });
    }

    expect(createBuffer).toHaveBeenCalledTimes(32);
    expect(copyBufferToBuffer).toHaveBeenCalledTimes(64);
    expect(peak).toBe(1);
    expect(live).toBe(0);
    for (const buffer of readbacks) {
      expect(buffer.destroy).toHaveBeenCalledOnce();
    }
  });
});
