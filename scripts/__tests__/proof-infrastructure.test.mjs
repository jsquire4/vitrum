import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

import { resolveUniqueVisibleCaptureSurface } from '../../tools/benchmark-runner/captureSurfaceValidation.mjs';
import {
  evaluateCaptureFreshness,
  GAP_CLOSURE_CAPTURE_PROVENANCE_SCHEMA,
  mechanicalGapClosureRow,
  selectGapClosureScenarios,
  strictGapClosureProblems,
} from '../../tools/benchmark-runner/gapClosureProofValidation.mjs';
import {
  validateGeneratedDznStatusNumericFields,
  validateDznStatusNumericFields,
} from '../../tools/behavioral-gate/dznStatusNumericValidation.mjs';
import {
  configMatchesBehavioralFilter,
  readOptionalNonEmptyFlagValue,
} from '../../tools/behavioral-gate/selectorValidation.mjs';

function fakePage(visibility) {
  const candidates = visibility.map((visible, index) => ({
    index,
    async isVisible() {
      return visible;
    },
  }));
  return {
    locator() {
      return {
        async count() {
          return candidates.length;
        },
        nth(index) {
          return candidates[index];
        },
      };
    },
  };
}

test('Playwright capture selection requires exactly one visible surface', async () => {
  await assert.rejects(
    resolveUniqueVisibleCaptureSurface(fakePage([]), 'canvas'),
    /0 total and 0 visible surfaces/,
  );
  await assert.rejects(
    resolveUniqueVisibleCaptureSurface(fakePage([true, true]), 'canvas'),
    /2 total and 2 visible surfaces/,
  );
  const selected = await resolveUniqueVisibleCaptureSurface(
    fakePage([false, true, false]),
    'canvas',
  );
  assert.equal(selected.index, 1);
});

test('gap-closure filters fail on zero-match, unknown, or duplicate selections', () => {
  const scenarios = [{ scenarioId: 'one' }, { scenarioId: 'two' }];
  assert.deepEqual(selectGapClosureScenarios(scenarios, ['two']), [scenarios[1]]);
  assert.throws(() => selectGapClosureScenarios(scenarios, ['missing']), /matched no scenario/);
  assert.throws(() => selectGapClosureScenarios([], ['missing']), /matched no scenario/);
  assert.throws(() => selectGapClosureScenarios(scenarios, ['one', 'one']), /duplicate ids/);
});

test('mechanical gap closure is skipped and cannot satisfy strict proof', () => {
  const mechanical = mechanicalGapClosureRow(
    { scenarioId: 'mechanical-row' },
    {
      baselineExists: true,
      baselineHash: 'a'.repeat(64),
      baselinePath: 'baseline.png',
    },
  );
  assert.equal(mechanical.status, 'SKIPPED');
  assert.equal(mechanical.evidenceClass, 'MECHANICAL');
  assert.equal(mechanical.passFail, 'SKIPPED');
  assert.deepEqual(strictGapClosureProblems([mechanical]), ['mechanical-row: passFail=SKIPPED']);
});

test('strict gap proof requires fresh candidate provenance distinct from baseline', () => {
  const row = {
    scenarioId: 'live-row',
    passFail: 'PASS',
    evidenceClass: 'LIVE_GPU',
    captureProvenance: {
      schema: GAP_CLOSURE_CAPTURE_PROVENANCE_SCHEMA,
      baseline: {
        artifactPath: 'baseline.png',
        mtimeMs: 1_000,
        size: 128,
        sha256: 'a'.repeat(64),
      },
      candidates: [
        {
          artifactPath: 'candidate.png',
          mtimeMs: Date.parse('2026-07-29T00:00:00.100Z'),
          size: 256,
          sha256: 'b'.repeat(64),
          fresh: true,
          previousArtifact: null,
          captureStartedAt: '2026-07-29T00:00:00.000Z',
          capturedAt: '2026-07-29T00:00:00.100Z',
        },
      ],
    },
  };
  assert.deepEqual(strictGapClosureProblems([row]), []);
  row.captureProvenance.candidates[0].artifactPath = 'baseline.png';
  assert.deepEqual(strictGapClosureProblems([row]), [
    'live-row: candidate provenance is stale or not distinct',
  ]);
  const candidate = row.captureProvenance.candidates[0];
  candidate.artifactPath = 'candidate.png';
  candidate.previousArtifact = {
    artifactPath: candidate.artifactPath,
    mtimeMs: candidate.mtimeMs,
    size: candidate.size,
    sha256: candidate.sha256,
  };
  assert.deepEqual(strictGapClosureProblems([row]), [
    'live-row: candidate provenance is stale or not distinct',
  ]);
});

test('capture freshness rejects unchanged and pre-run artifacts', () => {
  const before = { mtimeMs: 1_000, size: 4, sha256: 'a'.repeat(64) };
  assert.equal(evaluateCaptureFreshness(before, before, 1_000).fresh, false);
  assert.equal(
    evaluateCaptureFreshness(before, { mtimeMs: 500, size: 5, sha256: 'b'.repeat(64) }, 5_000)
      .fresh,
    false,
  );
  assert.equal(
    evaluateCaptureFreshness(before, { mtimeMs: 5_100, size: 5, sha256: 'b'.repeat(64) }, 5_000)
      .fresh,
    true,
  );
});

function validDznFixture() {
  return {
    expected: {
      path: 'fixture.json',
      configs: [
        {
          label: 'pt/mutation-material',
          goldenStatus: 'ok',
          mutationKind: 'material',
        },
      ],
    },
    status: {
      timeoutMs: 420_000,
      exitStatus: 0,
      summary: {
        totalConfigs: 1,
        failures: 0,
        knownResiduals: 0,
      },
      configs: [
        {
          label: 'pt/mutation-material',
          luminance: 0.1,
          gpuErrors: 0,
          rmse: 1,
          meanAbs: 0.5,
          maxAbs: 4,
          thresholds: {
            maxRmse: 8,
            maxMeanAbs: 4,
            maxAbs: 48,
          },
          mutationMeanAbs: 3,
          mutationMaxAbs: 12,
        },
      ],
    },
  };
}

test('DZN numeric evidence rejects missing, string, NaN, and infinite fields', () => {
  const valid = validDznFixture();
  assert.doesNotThrow(() => validateDznStatusNumericFields(valid.status, valid.expected));

  for (const mutation of [
    (status) => {
      delete status.configs[0].mutationMeanAbs;
    },
    (status) => {
      status.configs[0].mutationMeanAbs = '3';
    },
    (status) => {
      status.configs[0].mutationMeanAbs = Number.NaN;
    },
    (status) => {
      status.configs[0].mutationMeanAbs = Number.POSITIVE_INFINITY;
    },
    (status) => {
      status.summary.totalConfigs = undefined;
    },
    (status) => {
      status.configs[0].thresholds.maxAbs = null;
    },
  ]) {
    const { status, expected } = validDznFixture();
    mutation(status);
    assert.throws(
      () => validateDznStatusNumericFields(status, expected),
      /is missing|must be a number-typed finite value/,
    );
  }
});

test('new DZN PASS output fails closed when parsing loses a numeric token', () => {
  const { status } = validDznFixture();
  status.configs[0].goldenStatus = 'ok';
  status.configs[0].mutationKind = 'material';
  assert.doesNotThrow(() => validateGeneratedDznStatusNumericFields(status));
  delete status.configs[0].mutationKind;
  status.configs[0].mutationMaxAbs = null;
  assert.throws(
    () => validateGeneratedDznStatusNumericFields(status),
    /mutationMaxAbs must be a number-typed finite value/,
  );

  const missingRows = validDznFixture().status;
  missingRows.configs = [];
  assert.throws(
    () => validateGeneratedDznStatusNumericFields(missingRows),
    /must contain at least one parsed config/,
  );
});

test('behavioral-gate value selectors reject omitted, empty, and duplicate filters', () => {
  assert.equal(readOptionalNonEmptyFlagValue([], '--filter'), '');
  assert.equal(readOptionalNonEmptyFlagValue(['--filter', 'gltf'], '--filter'), 'gltf');
  assert.equal(readOptionalNonEmptyFlagValue(['--filter=gltf'], '--filter'), 'gltf');
  assert.throws(
    () => readOptionalNonEmptyFlagValue(['--filter'], '--filter'),
    /requires a non-empty value/,
  );
  assert.throws(
    () => readOptionalNonEmptyFlagValue(['--filter='], '--filter'),
    /requires a non-empty value/,
  );
  assert.throws(
    () => readOptionalNonEmptyFlagValue(['--filter', '--require-full-tier'], '--filter'),
    /requires a non-empty value/,
  );
  assert.throws(
    () => readOptionalNonEmptyFlagValue(['--filter=gltf', '--filter', 'default'], '--filter'),
    /may be provided at most once/,
  );
});

test('behavioral-gate default filter selects only the two canonical default lanes', () => {
  assert.equal(configMatchesBehavioralFilter('pt/default', 'default'), true);
  assert.equal(configMatchesBehavioralFilter('wh/default', 'default'), true);
  assert.equal(configMatchesBehavioralFilter('pt/sobol-default', 'default'), false);
  assert.equal(configMatchesBehavioralFilter('pt/sobol-default', 'sobol-default'), true);
  assert.equal(
    configMatchesBehavioralFilter('pt/cwbvh-broader-gltf-material-sweep', 'gltf', true),
    false,
  );
});

test('new DZN PASS output rejects a zero-evidence summary', () => {
  const { status } = validDznFixture();
  status.summary.totalConfigs = 0;
  status.configs = [];
  assert.throws(
    () => validateGeneratedDznStatusNumericFields(status),
    /must contain at least one parsed config/,
  );
});

test('CWBVH TLAS any-hit proof uses an independent traversal result', () => {
  const source = readFileSync(
    new URL('../../tools/behavioral-gate/cwbvh-parity-oracle.mjs', import.meta.url),
    'utf8',
  );
  assert.match(source, /fn gateTlasAny\(/);
  assert.match(source, /let anyHitResult = gateTlasAny\(/);
  assert.match(
    source,
    /anyHitResult\.status \* 2u \+ select\(0u, 1u, anyHitResult\.didHit\)/,
  );
  assert.match(source, /closest\.status,\s*anyHitResult\.status,\s*0xc0b7a11eu/s);
  assert.doesNotMatch(
    source,
    /closest\.status \* 2u \+ select\(0u, 1u, closest\.didHit\),\s*closest\.status,\s*closest\.status,\s*0xc0b7a11eu/s,
  );
});
