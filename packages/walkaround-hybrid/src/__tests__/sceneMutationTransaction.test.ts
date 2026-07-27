import { describe, expect, it } from 'vitest';
import {
  commitSceneMutations,
  prepareSceneMutations,
  runSceneMutationCleanups,
  SceneMutationCleanupError,
  SceneMutationFinalizationError,
  SceneMutationRollbackError,
  type PreparedSceneMutation,
} from '../SceneMutationTransaction.js';

function captureThrown(operation: () => void): unknown {
  try {
    operation();
  } catch (error) {
    return error;
  }
  throw new Error('expected operation to throw');
}

describe('scene mutation transaction coordinator', () => {
  it('preserves the prepare failure and every reverse-rollback failure', () => {
    const events: string[] = [];
    const primary = new Error('prepare-primary');
    const rollback0 = new Error('rollback-0');
    const rollback1 = new Error('rollback-1');
    const participant = (
      id: number,
      rollbackFailure: Error,
    ): PreparedSceneMutation => ({
      commit: () => events.push(`commit-${id}`),
      rollback: () => {
        events.push(`rollback-${id}`);
        throw rollbackFailure;
      },
      finalize: () => events.push(`finalize-${id}`),
    });

    const thrown = captureThrown(() => prepareSceneMutations([
      () => {
        events.push('prepare-0');
        return participant(0, rollback0);
      },
      () => {
        events.push('prepare-1');
        return participant(1, rollback1);
      },
      () => {
        events.push('prepare-2');
        throw primary;
      },
    ]));

    expect(events).toEqual([
      'prepare-0',
      'prepare-1',
      'prepare-2',
      'rollback-1',
      'rollback-0',
    ]);
    expect(thrown).toBeInstanceOf(SceneMutationRollbackError);
    if (!(thrown instanceof SceneMutationRollbackError)) return;
    expect(thrown.primaryError).toBe(primary);
    expect(thrown.errors).toEqual([primary, rollback1, rollback0]);
  });

  it('rolls every participant back in global reverse preparation order', () => {
    const events: string[] = [];
    const primary = new Error('commit-primary');
    const rollback0 = new Error('rollback-0');
    const rollback1 = new Error('rollback-1');
    const rollback2 = new Error('rollback-2');
    const rollback3 = new Error('rollback-3');
    const make = (
      id: number,
      commit: () => void,
      rollbackFailure: Error,
    ): PreparedSceneMutation => ({
      commit: () => {
        events.push(`commit-${id}`);
        commit();
      },
      rollback: () => {
        events.push(`rollback-${id}`);
        throw rollbackFailure;
      },
      finalize: () => events.push(`finalize-${id}`),
    });
    const prepared = [
      make(0, () => undefined, rollback0),
      make(1, () => {
        throw primary;
      }, rollback1),
      make(2, () => undefined, rollback2),
      make(3, () => undefined, rollback3),
    ];

    const thrown = captureThrown(() => commitSceneMutations(prepared));

    expect(events).toEqual([
      'commit-0',
      'commit-1',
      'rollback-3',
      'rollback-2',
      'rollback-1',
      'rollback-0',
    ]);
    expect(thrown).toBeInstanceOf(SceneMutationRollbackError);
    if (!(thrown instanceof SceneMutationRollbackError)) return;
    expect(thrown.errors).toEqual([
      primary,
      rollback3,
      rollback2,
      rollback1,
      rollback0,
    ]);
  });

  it('runs every finalizer and reports retirement failures after publication', () => {
    const events: string[] = [];
    const retirement0 = new Error('retirement-0');
    const retirement2 = new Error('retirement-2');
    const make = (
      id: number,
      retirementFailure?: Error,
    ): PreparedSceneMutation => ({
      commit: () => events.push(`commit-${id}`),
      rollback: () => events.push(`rollback-${id}`),
      finalize: () => {
        events.push(`finalize-${id}`);
        if (retirementFailure != null) throw retirementFailure;
      },
    });

    const thrown = captureThrown(() => commitSceneMutations([
      make(0, retirement0),
      make(1),
      make(2, retirement2),
    ]));

    expect(events).toEqual([
      'commit-0',
      'commit-1',
      'commit-2',
      'finalize-2',
      'finalize-1',
      'finalize-0',
    ]);
    expect(thrown).toBeInstanceOf(SceneMutationFinalizationError);
    if (!(thrown instanceof SceneMutationFinalizationError)) return;
    expect(thrown.committed).toBe(true);
    expect(thrown.errors).toEqual([retirement2, retirement0]);
  });

  it('flattens nested cleanup failures while still executing later owners', () => {
    const events: string[] = [];
    const first = new Error('first');
    const second = new Error('second');
    const third = new Error('third');

    const thrown = captureThrown(() => runSceneMutationCleanups([
      () => runSceneMutationCleanups([
        () => {
          events.push('first');
          throw first;
        },
        () => {
          events.push('second');
          throw second;
        },
      ], 'nested cleanup failed'),
      () => {
        events.push('third');
        throw third;
      },
    ], 'outer cleanup failed'));

    expect(events).toEqual(['first', 'second', 'third']);
    expect(thrown).toBeInstanceOf(SceneMutationCleanupError);
    if (!(thrown instanceof SceneMutationCleanupError)) return;
    expect(thrown.errors).toEqual([first, second, third]);
  });

  it('rethrows the original failure unchanged when rollback succeeds', () => {
    const primary = new Error('primary');
    const participant: PreparedSceneMutation = {
      commit: () => {
        throw primary;
      },
      rollback: () => undefined,
      finalize: () => undefined,
    };

    expect(captureThrown(() => commitSceneMutations([participant]))).toBe(
      primary,
    );
  });
});
