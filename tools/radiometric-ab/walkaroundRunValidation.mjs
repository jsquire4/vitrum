// @ts-check

/** @typedef {{ complete: ReadonlySet<string>, partial: ReadonlySet<string> }} CaseVerdictContract */

/** @type {Readonly<Record<string, CaseVerdictContract>>} */
const CASE_VERDICT_CONTRACT = Object.freeze({
  a8: Object.freeze({
    complete: new Set(['NEGLIGIBLE']),
    partial: new Set(['SMALL', 'MODERATE', 'SIGNIFICANT']),
  }),
  sun: Object.freeze({
    complete: new Set(['PASS']),
    partial: new Set(['PASS-PARTIAL']),
  }),
  glass: Object.freeze({
    complete: new Set(['PASS']),
    partial: new Set(['FINDING', 'SMOKE']),
  }),
  glossy: Object.freeze({
    complete: new Set(['PASS']),
    partial: new Set(['FINDING', 'PASS-WEAK']),
  }),
});

export class WalkaroundRunValidationError extends Error {
  /** @param {string} code @param {string} message */
  constructor(code, message) {
    super(message);
    this.name = 'WalkaroundRunValidationError';
    this.code = code;
  }
}

/** @param {unknown} value @returns {value is Record<string, any>} */
function isRecord(value) {
  return value != null && typeof value === 'object' && !Array.isArray(value);
}

/**
 * @param {{ mtimeMs: number, size: number, text: string } | null} beforeState
 * @param {{ mtimeMs: number, size: number, text: string } | null} afterState
 * @param {number} startedAtMs
 */
function assertFreshResult(beforeState, afterState, startedAtMs) {
  if (afterState == null) {
    throw new WalkaroundRunValidationError(
      'walkaround-ab-missing-result',
      'The native WebGPU harness exited 0 but did not write a readable result artifact.',
    );
  }
  if (!Number.isFinite(afterState.mtimeMs) || !Number.isSafeInteger(afterState.size)) {
    throw new WalkaroundRunValidationError(
      'walkaround-ab-invalid-result',
      'The native WebGPU harness wrote a result artifact with invalid file metadata.',
    );
  }
  const changed = beforeState == null ||
    afterState.mtimeMs !== beforeState.mtimeMs ||
    afterState.size !== beforeState.size ||
    afterState.text !== beforeState.text;
  // Leave room for filesystems whose timestamp precision rounds to whole seconds.
  const writtenDuringRun = afterState.mtimeMs >= startedAtMs - 2_000;
  if (!changed || !writtenDuringRun) {
    throw new WalkaroundRunValidationError(
      'walkaround-ab-stale-result',
      'The native WebGPU harness exited 0 without replacing the result artifact during this run.',
    );
  }
}

/**
 * @param {unknown} payload
 * @param {{
 *   expectedCaseIds: string[],
 *   beforeState: { mtimeMs: number, size: number, text: string } | null,
 *   afterState: { mtimeMs: number, size: number, text: string } | null,
 *   startedAtMs: number,
 * }} options
 */
export function validateWalkaroundRunResult(payload, options) {
  assertFreshResult(options.beforeState, options.afterState, options.startedAtMs);
  if (!isRecord(payload)) {
    throw new WalkaroundRunValidationError(
      'walkaround-ab-invalid-result',
      'The native WebGPU harness exited 0 but wrote a non-object result artifact.',
    );
  }

  const expectedCaseIds = [...new Set(options.expectedCaseIds)];
  if (expectedCaseIds.length === 0 || expectedCaseIds.length !== options.expectedCaseIds.length) {
    throw new WalkaroundRunValidationError(
      'walkaround-ab-invalid-case-selection',
      'The walkaround result validator requires a non-empty, unique case selection.',
    );
  }

  /** @type {Record<string, string>} */
  const caseVerdicts = {};
  /** @type {string[]} */
  const partialCaseIds = [];
  for (const id of expectedCaseIds) {
    const contract = CASE_VERDICT_CONTRACT[id];
    if (contract == null) {
      throw new WalkaroundRunValidationError(
        'walkaround-ab-invalid-case-selection',
        `Unknown walkaround regression case: ${id}`,
      );
    }
    const row = payload[id];
    if (!isRecord(row) || typeof row.verdict !== 'string') {
      throw new WalkaroundRunValidationError(
        'walkaround-ab-incomplete-result',
        `The result artifact is missing a verdict for: ${id}`,
      );
    }
    const verdict = row.verdict;
    if (verdict === 'FAIL' || verdict === 'ERROR') {
      throw new WalkaroundRunValidationError(
        'walkaround-ab-nonpass-result',
        `The ${id} case returned ${verdict}.`,
      );
    }
    if (contract.complete.has(verdict)) {
      caseVerdicts[id] = verdict;
      continue;
    }
    if (contract.partial.has(verdict)) {
      caseVerdicts[id] = verdict;
      partialCaseIds.push(`${id}:${verdict}`);
      continue;
    }
    throw new WalkaroundRunValidationError(
      'walkaround-ab-invalid-verdict',
      `The ${id} case returned an unknown verdict: ${verdict}`,
    );
  }

  return {
    caseVerdicts,
    partialCaseIds,
    partial: partialCaseIds.length > 0,
  };
}
