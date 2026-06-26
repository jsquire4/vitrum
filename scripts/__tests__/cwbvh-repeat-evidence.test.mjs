import assert from 'node:assert/strict';
import { dirname, resolve } from 'node:path';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import test from 'node:test';

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..');
const validationQueueCheckerPath = resolve(repoRoot, 'tools', 'road-to-100', 'check-validation-queue.mjs');
const {
  buildCwbvhRepeatCampaignSummary,
  CWBVH_REPEAT_FILTERS,
  DEFAULT_REPEAT_DZN_TIMEOUT_MS,
  MIN_REPEAT_COUNT_PER_WORKLOAD,
  summarizeCwbvhRepeatEvidence,
} = await import(resolve(repoRoot, 'tools', 'behavioral-gate', 'run-cwbvh-default-promotion-repeats.mjs'));

test('CWBVH repeat evidence discards warmup and classifies uniform faster samples', () => {
  const records = makeRecords(({ runIndex, label }) => ({
    ratio: runIndex === 0 ? 1.4 : 0.82 + (label.length % 3) * 0.01,
  }), MIN_REPEAT_COUNT_PER_WORKLOAD + 1);

  const summary = summarizeCwbvhRepeatEvidence(records, { warmupCount: 1 });

  assert.equal(summary.harness, 'cwbvh-default-promotion-repeat-proof');
  assert.equal(summary.verdict, 'PASS');
  assert.equal(summary.classification, 'uniform-faster');
  assert.equal(summary.promotion.defaultReady, true);
  assert.equal(summary.sampleCountPerWorkload, MIN_REPEAT_COUNT_PER_WORKLOAD);
  assert.equal(summary.allWorkloadsHaveRequiredRepeats, true);
  assert.equal(summary.failures.length, 0);
  for (const workload of summary.workloads) {
    assert.equal(workload.sampleCount, MIN_REPEAT_COUNT_PER_WORKLOAD);
    assert.equal(workload.totalRunCount, MIN_REPEAT_COUNT_PER_WORKLOAD + 1);
    assert.ok(workload.ratios.every((ratio) => ratio < 0.95));
    assert.equal(workload.slowOrNeutralSampleCount, 0);
  }
});

test('CWBVH repeat evidence stays partial when any workload median is slow', () => {
  const records = makeRecords(({ label }) => ({
    ratio: label === 'pt/cwbvh-broader-gltf-material-sweep' ? 1.08 : 0.86,
  }), MIN_REPEAT_COUNT_PER_WORKLOAD + 1);

  const summary = summarizeCwbvhRepeatEvidence(records, { warmupCount: 1 });

  assert.equal(summary.verdict, 'PASS-PARTIAL');
  assert.equal(summary.classification, 'mixed');
  assert.equal(summary.promotion.defaultReady, false);
  const gltf = summary.workloads.find((row) => row.label === 'pt/cwbvh-broader-gltf-material-sweep');
  assert.equal(gltf.slowOrNeutralSampleCount, MIN_REPEAT_COUNT_PER_WORKLOAD);
});

test('CWBVH repeat evidence fails closed on too few post-warmup samples', () => {
  const records = makeRecords(() => ({ ratio: 0.8 }), MIN_REPEAT_COUNT_PER_WORKLOAD);

  const summary = summarizeCwbvhRepeatEvidence(records, { warmupCount: 1 });

  assert.equal(summary.verdict, 'PASS-PARTIAL');
  assert.equal(summary.classification, 'uniform-faster');
  assert.equal(summary.promotion.defaultReady, false);
  assert.equal(summary.sampleCountPerWorkload, MIN_REPEAT_COUNT_PER_WORKLOAD - 1);
  assert.equal(summary.allWorkloadsHaveRequiredRepeats, false);
});

test('CWBVH repeat evidence reports insufficient samples when only warmup rows exist', () => {
  const records = makeRecords(() => ({ ratio: 0.8 }), 1);

  const summary = summarizeCwbvhRepeatEvidence(records, { warmupCount: 1 });

  assert.equal(summary.verdict, 'PASS-PARTIAL');
  assert.equal(summary.classification, 'insufficient-samples');
  assert.equal(summary.promotion.defaultReady, false);
  assert.equal(summary.sampleCountPerWorkload, 0);
  assert.equal(summary.allWorkloadsHaveAnySamples, false);
  assert.equal(summary.allWorkloadsHaveRequiredRepeats, false);
});

test('CWBVH repeat evidence rejects phase/run-index mismatches', () => {
  const records = makeRecords(() => ({ ratio: 0.8 }), MIN_REPEAT_COUNT_PER_WORKLOAD + 1);
  records[0].phase = 'sample';

  const summary = summarizeCwbvhRepeatEvidence(records, { warmupCount: 1 });

  assert.equal(summary.verdict, 'PASS-PARTIAL');
  assert.equal(summary.promotion.defaultReady, false);
  assert.equal(summary.failures.length, 1);
  assert.equal(summary.failures[0].reason, 'phase-mismatch');
  assert.equal(summary.failures[0].expectedPhase, 'warmup');
});

test('CWBVH repeat evidence records invalid shard rows as failures', () => {
  const records = makeRecords(() => ({ ratio: 0.8 }), MIN_REPEAT_COUNT_PER_WORKLOAD + 1);
  records[0].status.verdict = 'FAIL';

  const summary = summarizeCwbvhRepeatEvidence(records, { warmupCount: 1 });

  assert.equal(summary.verdict, 'PASS-PARTIAL');
  assert.equal(summary.promotion.defaultReady, false);
  assert.equal(summary.failures.length, 1);
  assert.equal(summary.failures[0].reason, 'status-not-pass');
});

test('CWBVH repeat evidence rejects rows without GPU and memory sanity proof', () => {
  const records = makeRecords(() => ({ ratio: 0.8 }), MIN_REPEAT_COUNT_PER_WORKLOAD + 1);
  const row = records[0].status.configs[0];
  row.gpuErrors = 1;
  row.nan = true;
  row.cwbvhMemoryBytes = 0;

  const summary = summarizeCwbvhRepeatEvidence(records, { warmupCount: 1 });

  assert.equal(summary.verdict, 'PASS-PARTIAL');
  assert.equal(summary.promotion.defaultReady, false);
  assert.equal(summary.failures.length, 1);
  assert.equal(summary.failures[0].reason, 'invalid-cwbvh-row');
  assert.equal(summary.failures[0].label, CWBVH_REPEAT_FILTERS[0].labels[0]);
});

test('CWBVH repeat campaign summaries do not promote interrupted captures', () => {
  const records = makeRecords(() => ({ ratio: 0.8 }), MIN_REPEAT_COUNT_PER_WORKLOAD + 1);

  const summary = buildCwbvhRepeatCampaignSummary(records, {
    repeats: MIN_REPEAT_COUNT_PER_WORKLOAD,
    warmupCount: 1,
    campaignStatus: 'interrupted',
    failure: {
      runIndex: 6,
      filter: 'cwbvh-broader',
      exitStatus: 124,
    },
    recordsPath: 'tools/behavioral-gate/cwbvh-default-promotion-repeat-records.json',
  });

  assert.equal(summary.campaignStatus, 'interrupted');
  assert.equal(summary.verdict, 'PASS-PARTIAL');
  assert.equal(summary.classification, 'uniform-faster');
  assert.equal(summary.promotion.defaultReady, false);
  assert.equal(summary.promotion.blockedBy, 'repeat-campaign-incomplete');
  assert.equal(summary.failure.exitStatus, 124);
});

test('CWBVH repeat campaign summaries pin the promotion-sized dzn timeout', () => {
  const records = makeRecords(() => ({ ratio: 0.8 }), MIN_REPEAT_COUNT_PER_WORKLOAD + 1);

  const summary = buildCwbvhRepeatCampaignSummary(records, {
    repeats: MIN_REPEAT_COUNT_PER_WORKLOAD,
    warmupCount: 1,
    recordsPath: 'tools/behavioral-gate/cwbvh-default-promotion-repeat-records.json',
  });

  assert.equal(summary.dznTimeoutMs, DEFAULT_REPEAT_DZN_TIMEOUT_MS);
  assert.match(summary.command, /--dzn-timeout-ms=900000/);
});

test('CWBVH default-promotion summary cites the completed repeat evidence without promoting', async () => {
  const summary = JSON.parse(await readFile(
    resolve(repoRoot, 'tools', 'behavioral-gate', 'cwbvh-default-promotion-status.json'),
    'utf8',
  ));
  const repeatStatus = JSON.parse(await readFile(
    resolve(repoRoot, 'tools', 'behavioral-gate', 'cwbvh-default-promotion-repeat-status.json'),
    'utf8',
  ));

  assert.equal(summary.verdict, 'PASS-PARTIAL');
  assert.equal(summary.promotion.defaultReady, false);
  assert.equal(summary.repeatEvidence.status, 'completed-five-sample-warmup-discarded-nonpromoting');
  assert.equal(summary.repeatEvidence.sampleCountPerWorkload, repeatStatus.sampleCountPerWorkload);
  assert.equal(summary.repeatEvidence.warmupDiscardedPerWorkload, repeatStatus.warmupDiscardedPerWorkload);
  assert.equal(summary.repeatEvidence.classification, repeatStatus.classification);
  assert.equal(summary.repeatEvidence.defaultPromotionEligible, false);
  assert.match(summary.repeatEvidence.residual, /one material-lobe-map fast outlier/);
});

test('CWBVH committed repeat status is derived from raw repeat records', async () => {
  const repeatStatus = JSON.parse(await readFile(
    resolve(repoRoot, 'tools', 'behavioral-gate', 'cwbvh-default-promotion-repeat-status.json'),
    'utf8',
  ));
  const repeatRecords = JSON.parse(await readFile(
    resolve(repoRoot, 'tools', 'behavioral-gate', 'cwbvh-default-promotion-repeat-records.json'),
    'utf8',
  ));
  const recomputed = summarizeCwbvhRepeatEvidence(repeatRecords.records, {
    warmupCount: repeatStatus.warmupDiscardedPerWorkload,
  });

  assert.deepEqual(repeatStatus.workloads, recomputed.workloads);
  assert.equal(repeatStatus.classification, recomputed.classification);
  assert.equal(repeatStatus.sampleCountPerWorkload, recomputed.sampleCountPerWorkload);

  const tamperedRecords = structuredClone(repeatRecords.records);
  const sample = tamperedRecords.find((record) => record.phase === 'sample' && record.filter === 'cwbvh-broader');
  const row = sample.status.configs.find((entry) => entry.label === 'pt/cwbvh-broader-material-lobe-maps');
  row.cwbvhRenderMsRatio = 0.5;
  const tamperedSummary = summarizeCwbvhRepeatEvidence(tamperedRecords, {
    warmupCount: repeatStatus.warmupDiscardedPerWorkload,
  });

  assert.notDeepEqual(tamperedSummary.workloads, repeatStatus.workloads);
});

test('Road checker pins CWBVH promotion provenance blocks', async () => {
  const validationQueueChecker = await readFile(validationQueueCheckerPath, 'utf8');

  assert.match(validationQueueChecker, /REQUIRED_CWBVH_PROMOTION_SOURCE_STATUS_PATHS/);
  assert.match(validationQueueChecker, /vitrum\.cwbvh-default-promotion\.provenance\.v1/);
  assert.match(validationQueueChecker, /repeat-status-provenance\.v1/);
  assert.match(validationQueueChecker, /repeat-records-provenance\.v1/);
  assert.match(validationQueueChecker, /cwbvhPromotionProvenance/);
  assert.match(validationQueueChecker, /cwbvhRepeatStatusProvenance/);
  assert.match(validationQueueChecker, /cwbvhRepeatRecordsProvenance/);
  assert.match(validationQueueChecker, /assertSha256DigestRows/);
  const checker = await readFile(
    resolve(repoRoot, 'tools', 'behavioral-gate', 'check-cwbvh-default-promotion-status.mjs'),
    'utf8',
  );
  assert.match(checker, /runIndex .* must be phase/);
});
function makeRecords(ratioFor, runCount) {
  const records = [];
  for (let runIndex = 0; runIndex < runCount; runIndex += 1) {
    for (const entry of CWBVH_REPEAT_FILTERS) {
      records.push({
        runIndex,
        phase: runIndex === 0 ? 'warmup' : 'sample',
        filter: entry.filter,
        status: makeStatus(entry, runIndex, ratioFor),
      });
    }
  }
  return records;
}

function makeStatus(entry, runIndex, ratioFor) {
  return {
    harness: 'behavioral-gate:dzn',
    verdict: 'PASS',
    filter: entry.filter,
    goldenVariant: 'dzn-full',
    summary: { totalConfigs: entry.labels.length, failures: 0, knownResiduals: 0 },
    configs: entry.labels.map((label, labelIndex) => {
      const { ratio } = ratioFor({ runIndex, label, labelIndex });
      const binaryMs = 1000 + runIndex * 10 + labelIndex;
      return {
        verdict: 'PASS',
        label,
        rawStatus: 'OK',
        tier: 'full',
        luminance: 0.25,
        gpuErrors: 0,
        nan: false,
        cwbvhParityKind: 'binary',
        cwbvhParityRmse: 0,
        cwbvhParityMeanAbs: 0,
        cwbvhParityMaxAbs: 0,
        cwbvhPerfKind: 'same-scene',
        cwbvhBinaryMemoryBytes: 4096,
        cwbvhMemoryBytes: 3072,
        cwbvhBinaryRenderMs: binaryMs,
        cwbvhRenderMs: binaryMs * ratio,
        cwbvhRenderMsRatio: ratio,
      };
    }),
  };
}
