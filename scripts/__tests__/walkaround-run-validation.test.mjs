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

function validGlossyPassRow() {
  return {
    id: 'GLOSSY',
    description: 'fixture',
    spp: 16,
    resolution: '128x128',
    regions: { sample: { x0: 32, y0: 32, x1: 96, y1: 96 } },
    sampleRegion: 'visible-back-wall-center-crop',
    metal: { sampleRegionLum: 0.1, floorLum: 0.1, overall: 0.1 },
    diffuse: { sampleRegionLum: 0.05, floorLum: 0.05, overall: 0.05 },
    sampleRatio: 2,
    floorRatio: 2,
    delta: { sampleRegionLum: 0.05, floorLum: 0.05, overall: 0.05 },
    expectedMinSampleRatio: 0.8,
    expectedMinFloorRatio: 0.8,
    minSignalDelta: 1e-4,
    materialEffectObserved: true,
    promotion: {
      defaultReady: false,
      blocker: 'ddgi-irradiance-cache-not-ggx-filtered-radiance',
      requiredEvidence: 'fixture evidence',
    },
    renderTimeSec: 1,
    verdict: 'PASS',
    notes: ['fixture'],
    qualityProfile: 'fixture',
    renderConfig: { width: 128, height: 128, spp: 16 },
  };
}

function validSignificantA8Row() {
  const biased = {
    overall: 0.1,
    floor: 0.2,
    ceiling: 0.3,
    leftWall: 0.4,
    rightWall: 0.5,
  };
  const unbiased = {
    overall: 0.17,
    floor: 0.21,
    ceiling: 0.32,
    leftWall: 0.43,
    rightWall: 0.54,
  };
  return {
    id: 'A8',
    description: 'fixture',
    spp: 16,
    resolution: '128x128',
    regions: {
      floor: { x0: 0, y0: 96, x1: 128, y1: 128 },
      ceiling: { x0: 0, y0: 0, x1: 128, y1: 24 },
      leftWall: { x0: 0, y0: 24, x1: 24, y1: 96 },
      rightWall: { x0: 104, y0: 24, x1: 128, y1: 96 },
    },
    biased,
    unbiased,
    delta: Object.fromEntries(
      Object.keys(biased).map((key) => [key, unbiased[key] - biased[key]]),
    ),
    renderTimeSec: 1,
    verdict: 'SIGNIFICANT',
    notes: ['fixture'],
    qualityProfile: 'fixture',
    renderConfig: { width: 128, height: 128, spp: 16 },
  };
}

test('walkaround run validator rejects verdict-only complete and partial rows', () => {
  for (const payload of [
    {
      a8: { verdict: 'NEGLIGIBLE' },
      sun: { verdict: 'PASS' },
      glass: { verdict: 'PASS' },
      glossy: { verdict: 'PASS' },
    },
    {
      a8: { verdict: 'SMALL' },
      sun: { verdict: 'PASS-PARTIAL' },
      glass: { verdict: 'FINDING' },
      glossy: { verdict: 'PASS-WEAK' },
    },
  ]) {
    assert.throws(
      () => validate(payload),
      (error) =>
        error instanceof WalkaroundRunValidationError &&
        error.code === 'walkaround-ab-invalid-result-evidence',
    );
  }
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

test('walkaround run validator fails closed on a measured SIGNIFICANT A8 regression', () => {
  assert.throws(
    () => validateWalkaroundRunResult(
      { a8: validSignificantA8Row() },
      {
        expectedCaseIds: ['a8'],
        beforeState,
        afterState,
        startedAtMs,
      },
    ),
    (error) =>
      error instanceof WalkaroundRunValidationError &&
      error.code === 'walkaround-ab-nonpass-result',
  );
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
      glossy: validGlossyPassRow(),
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
