#!/usr/bin/env node
// Collect repeated CWBVH default-promotion timing evidence from the existing
// dzn behavioral-gate shards. This is intentionally separate from proof-check:
// it is a promotion-data capture lane, not a cheap source/proof invariant.

import { mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, resolve } from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath, pathToFileURL } from 'node:url';

export const CWBVH_REPEAT_FILTERS = [
  {
    filter: 'cwbvh-binary-parity',
    labels: ['pt/cwbvh-binary-parity'],
  },
  {
    filter: 'cwbvh-complex-parity',
    labels: ['pt/cwbvh-complex-parity'],
  },
  {
    filter: 'cwbvh-broader',
    labels: [
      'pt/cwbvh-broader-material-lobes',
      'pt/cwbvh-broader-material-lobe-maps',
      'pt/cwbvh-broader-gltf-material-sweep',
    ],
  },
];

export const MIN_REPEAT_COUNT_PER_WORKLOAD = 5;
const DEFAULT_WARMUP_COUNT = 1;
const DEFAULT_OUTPUT = 'tools/behavioral-gate/cwbvh-default-promotion-repeat-status.json';
const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '../..');

export function summarizeCwbvhRepeatEvidence(records, options = {}) {
  const warmupCount = readNonNegativeInteger(options.warmupCount ?? DEFAULT_WARMUP_COUNT, 'warmupCount');
  const expectedLabels = CWBVH_REPEAT_FILTERS.flatMap((entry) => entry.labels);
  const byLabel = new Map(expectedLabels.map((label) => [label, []]));
  const runCountByLabel = new Map(expectedLabels.map((label) => [label, 0]));
  const failures = [];

  for (const record of records) {
    const runIndex = Number(record?.runIndex);
    const status = record?.status;
    const filter = String(record?.filter ?? status?.filter ?? '');
    const filterSpec = CWBVH_REPEAT_FILTERS.find((entry) => entry.filter === filter);
    if (!filterSpec) {
      failures.push({ runIndex, filter, reason: 'unexpected-filter' });
      continue;
    }
    if (status?.verdict !== 'PASS' || status?.goldenVariant !== 'dzn-full' || status?.summary?.failures !== 0) {
      failures.push({ runIndex, filter, reason: 'status-not-pass' });
      continue;
    }
    if (!Array.isArray(status.configs)) {
      failures.push({ runIndex, filter, reason: 'missing-configs' });
      continue;
    }
    for (const label of filterSpec.labels) {
      const row = status.configs.find((entry) => entry?.label === label);
      if (!isValidCwbvhTimingRow(row)) {
        failures.push({ runIndex, filter, label, reason: 'invalid-cwbvh-row' });
        continue;
      }
      runCountByLabel.set(label, (runCountByLabel.get(label) ?? 0) + 1);
      if (runIndex >= warmupCount) {
        byLabel.get(label)?.push({
          runIndex,
          ratio: Number(row.cwbvhRenderMsRatio),
          binaryMs: Number(row.cwbvhBinaryRenderMs),
          cwbvhMs: Number(row.cwbvhRenderMs),
        });
      }
    }
  }

  const workloads = expectedLabels.map((label) => {
    const samples = byLabel.get(label) ?? [];
    const ratios = samples.map((sample) => sample.ratio);
    return {
      label,
      totalRunCount: runCountByLabel.get(label) ?? 0,
      sampleCount: samples.length,
      ratios,
      minRatio: ratios.length ? Math.min(...ratios) : null,
      medianRatio: median(ratios),
      meanRatio: mean(ratios),
      maxRatio: ratios.length ? Math.max(...ratios) : null,
      fastSampleCount: ratios.filter((ratio) => ratio < 0.95).length,
      slowOrNeutralSampleCount: ratios.filter((ratio) => ratio >= 1).length,
    };
  });
  const sampleCountPerWorkload = Math.min(...workloads.map((row) => row.sampleCount));
  const allWorkloadsHaveRequiredRepeats = sampleCountPerWorkload >= MIN_REPEAT_COUNT_PER_WORKLOAD;
  const allMedianFast = workloads.every((row) => Number(row.medianRatio) < 0.95);
  const allMedianSlowOrNeutral = workloads.every((row) => Number(row.medianRatio) >= 1);
  const classification = allMedianFast
    ? 'uniform-faster'
    : (allMedianSlowOrNeutral ? 'uniform-slower' : 'mixed');

  return {
    harness: 'cwbvh-default-promotion-repeat-proof',
    verdict: allWorkloadsHaveRequiredRepeats && classification === 'uniform-faster' && failures.length === 0
      ? 'PASS'
      : 'PASS-PARTIAL',
    adapterScope: 'wsl-dzn-real-adapter',
    warmupDiscardedPerWorkload: warmupCount,
    sampleCountPerWorkload,
    minRepeatCountPerWorkload: MIN_REPEAT_COUNT_PER_WORKLOAD,
    allWorkloadsHaveRequiredRepeats,
    classification,
    promotion: {
      defaultReady: allWorkloadsHaveRequiredRepeats && classification === 'uniform-faster' && failures.length === 0,
      requiredEvidence: 'multiple warmup-discarded repeats per workload on browser/real-adapter hosts',
    },
    workloads,
    failures,
  };
}

function isValidCwbvhTimingRow(row) {
  return row != null &&
    row.verdict === 'PASS' &&
    row.rawStatus === 'OK' &&
    row.tier === 'full' &&
    row.cwbvhParityKind === 'binary' &&
    Number(row.cwbvhParityRmse) <= 1 &&
    Number(row.cwbvhParityMeanAbs) <= 0.5 &&
    Number(row.cwbvhParityMaxAbs) <= 8 &&
    row.cwbvhPerfKind === 'same-scene' &&
    Number(row.cwbvhBinaryRenderMs) > 0 &&
    Number(row.cwbvhRenderMs) > 0 &&
    Number(row.cwbvhRenderMsRatio) > 0;
}

function mean(values) {
  if (values.length === 0) return null;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function median(values) {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1
    ? sorted[mid]
    : (sorted[mid - 1] + sorted[mid]) / 2;
}

function readNonNegativeInteger(value, label) {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < 0) {
    throw new Error(`${label} must be a non-negative integer, got ${value}`);
  }
  return parsed;
}

function readPositiveInteger(value, label) {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new Error(`${label} must be a positive integer, got ${value}`);
  }
  return parsed;
}

function readFlagValue(args, name, fallback = null) {
  const eq = args.find((arg) => arg.startsWith(`${name}=`));
  if (eq) return eq.slice(name.length + 1);
  const index = args.indexOf(name);
  return index >= 0 ? (args[index + 1] ?? '') : fallback;
}

function commandFor(filter) {
  return `npm run behavioral-gate:dzn -- --filter ${filter} --require-full-tier`;
}

function runCampaign(args) {
  const repeats = readPositiveInteger(readFlagValue(args, '--repeats', String(MIN_REPEAT_COUNT_PER_WORKLOAD)), '--repeats');
  const warmupCount = readNonNegativeInteger(readFlagValue(args, '--warmup', String(DEFAULT_WARMUP_COUNT)), '--warmup');
  const outputPath = resolve(repoRoot, readFlagValue(args, '--status', DEFAULT_OUTPUT));
  const totalRuns = repeats + warmupCount;
  const tempDir = mkdtempSync(resolve(tmpdir(), 'vitrum-cwbvh-repeat-'));
  const records = [];

  try {
    for (let runIndex = 0; runIndex < totalRuns; runIndex += 1) {
      for (const entry of CWBVH_REPEAT_FILTERS) {
        const statusPath = resolve(tempDir, `${String(runIndex).padStart(2, '0')}-${entry.filter}.json`);
        const result = spawnSync(
          'npm',
          ['run', 'behavioral-gate:dzn', '--', '--filter', entry.filter, '--require-full-tier'],
          {
            cwd: repoRoot,
            encoding: 'utf8',
            stdio: 'inherit',
            env: {
              ...process.env,
              VITRUM_BEHAVIORAL_GATE_DZN_STATUS_PATH: statusPath,
            },
          },
        );
        if (result.error) throw result.error;
        if (result.status !== 0) {
          throw new Error(`${commandFor(entry.filter)} failed with exit status ${result.status}`);
        }
        records.push({
          runIndex,
          phase: runIndex < warmupCount ? 'warmup' : 'sample',
          filter: entry.filter,
          status: JSON.parse(readFileSync(statusPath, 'utf8')),
        });
      }
    }
    const summary = {
      generatedAt: new Date().toISOString(),
      command: `node tools/behavioral-gate/run-cwbvh-default-promotion-repeats.mjs --repeats=${repeats} --warmup=${warmupCount}`,
      filters: CWBVH_REPEAT_FILTERS.map((entry) => ({
        filter: entry.filter,
        labels: entry.labels,
        command: commandFor(entry.filter),
      })),
      ...summarizeCwbvhRepeatEvidence(records, { warmupCount }),
    };
    mkdirSync(dirname(outputPath), { recursive: true });
    writeFileSync(outputPath, `${JSON.stringify(summary, null, 2)}\n`);
    console.log(`[cwbvh-default-promotion-repeats] wrote ${outputPath}`);
    console.log(`[cwbvh-default-promotion-repeats] verdict=${summary.verdict} classification=${summary.classification} sampleCountPerWorkload=${summary.sampleCountPerWorkload}`);
    return summary.verdict === 'PASS' ? 0 : 2;
  } finally {
    if (!args.includes('--keep-temp')) {
      rmSync(tempDir, { recursive: true, force: true });
    }
  }
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  try {
    process.exit(runCampaign(process.argv.slice(2)));
  } catch (err) {
    console.error(`[cwbvh-default-promotion-repeats] ${err instanceof Error ? err.message : String(err)}`);
    process.exit(1);
  }
}
