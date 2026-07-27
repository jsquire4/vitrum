/**
 * A prepared scene mutation has three deliberately separate phases:
 *
 * - `commit` publishes only already-created state and must not destroy the old
 *   generation;
 * - `rollback` restores the old generation (or discards an uncommitted
 *   candidate);
 * - `finalize` releases the old generation after every participant committed.
 *
 * Keeping destruction out of `commit` is what lets a later participant fail
 * without leaving the engine split across scene generations.
 */
export interface PreparedSceneMutation {
  commit(): void;
  rollback(): void;
  finalize(): void;
}

export type SceneMutationCleanup = () => void;

/** Every failure from a best-effort cleanup sequence, in execution order. */
export class SceneMutationCleanupError extends AggregateError {
  constructor(errors: readonly unknown[], message: string) {
    super(errors, message);
    this.name = 'SceneMutationCleanupError';
  }
}

/** Publication failed and one or more rollback participants also failed. */
export class SceneMutationRollbackError extends AggregateError {
  readonly primaryError: unknown;

  constructor(primaryError: unknown, rollbackErrors: readonly unknown[], message: string) {
    super([primaryError, ...rollbackErrors], message);
    this.name = 'SceneMutationRollbackError';
    this.primaryError = primaryError;
  }
}

/**
 * Publication completed, but one or more old-generation retirement steps
 * failed. Callers must not roll the committed generation back.
 */
export class SceneMutationFinalizationError extends AggregateError {
  readonly committed = true;

  constructor(errors: readonly unknown[], message: string) {
    super(errors, message);
    this.name = 'SceneMutationFinalizationError';
  }
}

function collectSceneMutationCleanupErrors(
  cleanups: readonly SceneMutationCleanup[],
): unknown[] {
  const errors: unknown[] = [];
  for (const cleanup of cleanups) {
    try {
        cleanup();
      } catch (error) {
        if (error instanceof SceneMutationCleanupError) {
          for (const nestedError of error.errors as readonly unknown[]) {
            errors.push(nestedError);
          }
      } else {
        errors.push(error);
      }
    }
  }
  return errors;
}

/** Execute every cleanup and report the complete failure set. */
export function runSceneMutationCleanups(
  cleanups: readonly SceneMutationCleanup[],
  message: string,
): void {
  const errors = collectSceneMutationCleanupErrors(cleanups);
  if (errors.length > 0) {
    throw new SceneMutationCleanupError(errors, message);
  }
}

/**
 * Execute every rollback and then rethrow the primary failure. When rollback
 * also fails, the resulting AggregateError preserves the primary at index zero
 * followed by every rollback failure in execution order.
 */
export function rethrowWithSceneMutationCleanup(
  primaryError: unknown,
  cleanups: readonly SceneMutationCleanup[],
  message: string,
): never {
  const rollbackErrors = collectSceneMutationCleanupErrors(cleanups);
  if (rollbackErrors.length > 0) {
    throw new SceneMutationRollbackError(
      primaryError,
      rollbackErrors,
      message,
    );
  }
  throw primaryError;
}

/**
 * Prepare a group of participants without leaking candidates when a later
 * prepare step throws. The callback form is intentional: it also gives tests a
 * precise failure-injection boundary for every subsystem.
 */
export function prepareSceneMutations(
  factories: ReadonlyArray<() => PreparedSceneMutation>,
): PreparedSceneMutation[] {
  const prepared: PreparedSceneMutation[] = [];
  try {
    for (const factory of factories) prepared.push(factory());
    return prepared;
  } catch (error) {
    rethrowWithSceneMutationCleanup(
      error,
      [...prepared].reverse().map((mutation) => () => mutation.rollback()),
      'scene mutation preparation failed and rollback also failed',
    );
  }
}

/** Commit all participants, reverse-rollback on failure, then release old state. */
export function commitSceneMutations(
  prepared: readonly PreparedSceneMutation[],
): void {
  let committed = 0;
  try {
    for (; committed < prepared.length; committed += 1) {
      prepared[committed]!.commit();
    }
  } catch (error) {
    // Participants are prepared provider -> consumer. Roll every prepared
    // participant back in global reverse preparation order so no provider is
    // retired while an uncommitted consumer candidate can still reference it.
    // Rollback must be idempotent and safe after a partial commit, including
    // for the participant whose commit threw.
    const rollbacks = [...prepared]
      .reverse()
      .map((mutation): SceneMutationCleanup => () => mutation.rollback());
    rethrowWithSceneMutationCleanup(
      error,
      rollbacks,
      'scene mutation publication failed and rollback also failed',
    );
  }

  // Factories are dependency ordered. Retire in reverse so consumers release
  // their old-generation state before providers destroy the resources behind it.
  const finalizationErrors = collectSceneMutationCleanupErrors(
    [...prepared].reverse().map((mutation) => () => mutation.finalize()),
  );
  if (finalizationErrors.length > 0) {
    throw new SceneMutationFinalizationError(
      finalizationErrors,
      'scene mutation committed, but old-generation retirement failed',
    );
  }
}
