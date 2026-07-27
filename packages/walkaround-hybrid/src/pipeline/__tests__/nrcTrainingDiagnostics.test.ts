import { describe, expect, it, vi } from 'vitest';
import type { EngineError } from '@vitrum/core';
import { WalkaroundGPUPipeline } from '../WalkaroundGPUPipeline.js';

type TrainingHarness = Record<string, unknown> & {
  _tickSubsystemTraining(passLayout: unknown): void;
};

async function flushTrainingCatch(): Promise<void> {
  await Promise.resolve();
  await Promise.resolve();
  await new Promise<void>((resolve) => { setTimeout(resolve, 0); });
}

function makeHarness(
  trainFromRecords: () => Promise<void>,
  onError: (error: EngineError) => void,
): TrainingHarness {
  const pipeline = Object.create(WalkaroundGPUPipeline.prototype) as TrainingHarness;
  pipeline._initialized = true;
  pipeline._onError = onError;
  pipeline._lastNrcTrainingErrorMessage = null;
  pipeline._nrc = { trainFromRecords };
  pipeline._ppg = { maybeRunTrainingRefine: vi.fn() };
  pipeline._res = {};
  pipeline._frameCount = 3;
  return pipeline;
}

describe('WalkaroundGPUPipeline NRC training diagnostics', () => {
  it('reports rejected NRC training as a non-fatal EngineError', async () => {
    const raw = new Error('mapAsync failed');
    const trainFromRecords = vi.fn(() => Promise.reject(raw));
    const onError = vi.fn();
    const pipeline = makeHarness(trainFromRecords, onError);

    pipeline._tickSubsystemTraining({});
    await flushTrainingCatch();

    expect(onError).toHaveBeenCalledTimes(1);
    expect(onError).toHaveBeenCalledWith({
      kind: 'render',
      message: '[WalkaroundGPUPipeline] NRC training transaction failed; the last committed NRC generation remains valid. mapAsync failed',
      fatal: false,
      raw,
    });
  });

  it('contains and reports a synchronous NRC training throw', () => {
    const raw = new Error('synchronous training setup failed');
    const trainFromRecords = vi.fn(() => { throw raw; });
    const onError = vi.fn();
    const pipeline = makeHarness(trainFromRecords, onError);

    expect(() => pipeline._tickSubsystemTraining({})).not.toThrow();
    expect(onError).toHaveBeenCalledOnce();
    expect(onError).toHaveBeenCalledWith({
      kind: 'render',
      message: '[WalkaroundGPUPipeline] NRC training transaction failed; the last committed NRC generation remains valid. synchronous training setup failed',
      fatal: false,
      raw,
    });
  });

  it('contains a synchronous PPG refine throw and still starts NRC training', async () => {
    const raw = new Error('PPG readback setup failed');
    const trainFromRecords = vi.fn(() => Promise.resolve());
    const onError = vi.fn();
    const pipeline = makeHarness(trainFromRecords, onError);
    pipeline._ppg = {
      maybeRunTrainingRefine: vi.fn(() => { throw raw; }),
    };

    expect(() => pipeline._tickSubsystemTraining({})).not.toThrow();
    await flushTrainingCatch();
    expect(trainFromRecords).toHaveBeenCalledOnce();
    expect(onError).toHaveBeenCalledWith({
      kind: 'render',
      message: "[WalkaroundGPUPipeline] accepted frame post-submit hook 'PPG training/refine' failed; GPU submission remains accepted. PPG readback setup failed",
      fatal: false,
      raw,
    });
  });

  it('dedupes repeated NRC training failures until a step succeeds', async () => {
    const raw = new Error('transient readback failure');
    const trainFromRecords = vi.fn()
      .mockRejectedValueOnce(raw)
      .mockRejectedValueOnce(raw)
      .mockResolvedValueOnce(undefined)
      .mockRejectedValueOnce(raw);
    const onError = vi.fn();
    const pipeline = makeHarness(trainFromRecords, onError);

    pipeline._tickSubsystemTraining({});
    await flushTrainingCatch();
    pipeline._tickSubsystemTraining({});
    await flushTrainingCatch();
    expect(onError).toHaveBeenCalledTimes(1);

    pipeline._tickSubsystemTraining({});
    await flushTrainingCatch();
    pipeline._tickSubsystemTraining({});
    await flushTrainingCatch();
    expect(onError).toHaveBeenCalledTimes(2);
  });

  it('suppresses late NRC training failures after dispose clears initialized state', async () => {
    const trainFromRecords = vi.fn(() => Promise.reject(new Error('disposed')));
    const onError = vi.fn();
    const pipeline = makeHarness(trainFromRecords, onError);
    pipeline._initialized = false;

    pipeline._tickSubsystemTraining({});
    await flushTrainingCatch();

    expect(onError).not.toHaveBeenCalled();
  });
});
