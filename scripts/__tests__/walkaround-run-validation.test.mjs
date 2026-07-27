import assert from 'node:assert/strict';
import test from 'node:test';

import {
  validateWalkaroundRunResult,
  WalkaroundRunValidationError,
} from '../../tools/radiometric-ab/walkaroundRunValidation.mjs';

const beforeState = { mtimeMs: 1_000, size: 2, text: '{}' };
const afterState = { mtimeMs: 3_000, size: 40, text: '{"fresh":true}' };
const startedAtMs = 2_500;

function validate(payload, overrides = {}) {
  return validateWalkaroundRunResult(payload, {
    expectedCaseIds: ['a8', 'sun', 'glass', 'glossy'],
    beforeState,
    afterState,
    startedAtMs,
    ...overrides,
  });
}

test('walkaround run validator accepts only case-specific complete verdicts', () => {
  const result = validate({
    a8: { verdict: 'NEGLIGIBLE' },
    sun: { verdict: 'PASS' },
    glass: { verdict: 'PASS' },
    glossy: { verdict: 'PASS' },
  });
  assert.equal(result.partial, false);
  assert.deepEqual(result.caseVerdicts, {
    a8: 'NEGLIGIBLE',
    sun: 'PASS',
    glass: 'PASS',
    glossy: 'PASS',
  });
});

test('walkaround run validator classifies bounded diagnostic verdicts as partial', () => {
  const result = validate({
    a8: { verdict: 'SMALL' },
    sun: { verdict: 'PASS-PARTIAL' },
    glass: { verdict: 'FINDING' },
    glossy: { verdict: 'PASS-WEAK' },
  });
  assert.equal(result.partial, true);
  assert.deepEqual(result.partialCaseIds, [
    'a8:SMALL',
    'sun:PASS-PARTIAL',
    'glass:FINDING',
    'glossy:PASS-WEAK',
  ]);
});

test('walkaround run validator rejects FAIL, ERROR, and arbitrary verdicts', () => {
  for (const verdict of ['FAIL', 'ERROR', 'UNKNOWN', 'NEGLIGIBLE']) {
    assert.throws(
      () => validateWalkaroundRunResult(
        { glass: { verdict } },
        {
          expectedCaseIds: ['glass'],
          beforeState,
          afterState,
          startedAtMs,
        },
      ),
      (error) => error instanceof WalkaroundRunValidationError &&
        ['walkaround-ab-nonpass-result', 'walkaround-ab-invalid-verdict'].includes(error.code),
    );
  }
});

test('walkaround run validator rejects unchanged and pre-run result artifacts', () => {
  const payload = { sun: { verdict: 'PASS' } };
  assert.throws(
    () => validateWalkaroundRunResult(payload, {
      expectedCaseIds: ['sun'],
      beforeState,
      afterState: beforeState,
      startedAtMs,
    }),
    (error) => error instanceof WalkaroundRunValidationError &&
      error.code === 'walkaround-ab-stale-result',
  );
  assert.throws(
    () => validateWalkaroundRunResult(payload, {
      expectedCaseIds: ['sun'],
      beforeState,
      afterState: { mtimeMs: 100, size: 20, text: '{"changed":true}' },
      startedAtMs,
    }),
    (error) => error instanceof WalkaroundRunValidationError &&
      error.code === 'walkaround-ab-stale-result',
  );
});

test('walkaround run validator grades only cases selected for this run', () => {
  const result = validateWalkaroundRunResult(
    {
      glossy: { verdict: 'PASS' },
      glass: { verdict: 'FAIL' },
    },
    {
      expectedCaseIds: ['glossy'],
      beforeState,
      afterState,
      startedAtMs,
    },
  );
  assert.deepEqual(result.caseVerdicts, { glossy: 'PASS' });
});
