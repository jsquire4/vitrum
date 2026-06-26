#!/usr/bin/env -S deno run --sloppy-imports --allow-read
// @ts-check

const STATUS_FILES = [
  {
    path: "tools/behavioral-gate/behavioral-gate-dzn-cwbvh-binary-parity-status.json",
    command: "npm run behavioral-gate:dzn -- --filter cwbvh-binary-parity --require-full-tier",
    filter: "cwbvh-binary-parity",
    labels: ["pt/cwbvh-binary-parity"],
  },
  {
    path: "tools/behavioral-gate/behavioral-gate-dzn-cwbvh-complex-parity-status.json",
    command: "npm run behavioral-gate:dzn -- --filter cwbvh-complex-parity --require-full-tier",
    filter: "cwbvh-complex-parity",
    labels: ["pt/cwbvh-complex-parity"],
  },
  {
    path: "tools/behavioral-gate/behavioral-gate-dzn-cwbvh-broader-status.json",
    command: "npm run behavioral-gate:dzn -- --filter cwbvh-broader --require-full-tier",
    filter: "cwbvh-broader",
    labels: [
      "pt/cwbvh-broader-material-lobes",
      "pt/cwbvh-broader-material-lobe-maps",
      "pt/cwbvh-broader-gltf-material-sweep",
    ],
  },
];

const SOURCE_GUARD = "packages/pt-webgpu/src/index.ts";
const SUMMARY_STATUS = "tools/behavioral-gate/cwbvh-default-promotion-status.json";
const REPEAT_STATUS = "tools/behavioral-gate/cwbvh-default-promotion-repeat-status.json";
const REPEAT_RECORDS = "tools/behavioral-gate/cwbvh-default-promotion-repeat-records.json";
const REPEAT_HARNESS = "tools/behavioral-gate/run-cwbvh-default-promotion-repeats.mjs";
const MIN_SLOW_OR_NEUTRAL_RATIO = 1.0;
const MIN_FAST_RATIO = 0.95;
const MIN_SLOW_OR_NEUTRAL_ROWS = 2;
const MIN_REPEAT_COUNT_PER_WORKLOAD = 5;
const REQUIRED_ADAPTER_SCOPE = "browser/real-adapter";

/** @param {string} message @returns {never} */
function fail(message) {
  throw new Error(`[cwbvh-default-promotion-proof-check] ${message}`);
}

/** @param {unknown} value @param {string} label */
function requireFinite(value, label) {
  const n = Number(value);
  if (!Number.isFinite(n)) fail(`${label} must be finite, got ${value}`);
  return n;
}

/** @param {number[]} values */
function mean(values) {
  if (values.length === 0) return null;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

/** @param {number[]} values */
function median(values) {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1
    ? sorted[mid]
    : (sorted[mid - 1] + sorted[mid]) / 2;
}

/** @param {unknown} value */
function isCompletedDznRepeatRecord(value) {
  if (value == null || typeof value !== "object") return false;
  const record = /** @type {{ phase?: unknown, status?: { verdict?: unknown, goldenVariant?: unknown, timeoutMs?: unknown } }} */ (value);
  return (
    (record.phase === "warmup" || record.phase === "sample") &&
    record.status?.verdict === "PASS" &&
    record.status?.goldenVariant === "dzn-full" &&
    record.status?.timeoutMs === 900000
  );
}

/** @param {string} source */
function assertSourceStillOptIn(source) {
  if (!source.includes("opts.bvhTraversal === 'cwbvh-closest-experimental'")) {
    fail("pt-webgpu no longer gates CWBVH behind an explicit bvhTraversal option");
  }
  if (!source.includes(": 'binary';")) {
    fail("pt-webgpu source no longer visibly defaults bvhTraversal to binary");
  }
  if (!source.includes("renderer parity/performance A/B is still required before default promotion")) {
    fail("pt-webgpu CWBVH opt-in warning no longer preserves the promotion caveat");
  }
}

/** @param {string} source */
function assertRepeatHarness(source) {
  for (const needle of [
    "cwbvh-default-promotion-repeat-proof",
    "MIN_REPEAT_COUNT_PER_WORKLOAD = 5",
    "warmupDiscardedPerWorkload",
    "cwbvh-default-promotion-repeat-records",
    "campaignStatus",
    "insufficient-samples",
    "DEFAULT_REPEAT_DZN_TIMEOUT_MS = 900_000",
    "--dzn-timeout-ms",
    "VITRUM_BEHAVIORAL_GATE_DZN_TIMEOUT_MS: String(dznTimeoutMs)",
    "recordsPath",
    "statusVerdict",
    "VITRUM_BEHAVIORAL_GATE_DZN_STATUS_PATH",
    "behavioral-gate:dzn",
    "Number(row.gpuErrors) === 0",
    "row.nan === false",
    "Number(row.luminance) >= 0.005",
    "Number(row.cwbvhBinaryMemoryBytes) > 0",
    "Number(row.cwbvhMemoryBytes) > 0",
  ]) {
    if (!source.includes(needle)) fail(`${REPEAT_HARNESS}: missing ${needle}`);
  }
}

/**
 * @param {Record<string, any>} status
 * @param {{ path: string; command: string; filter: string; labels: readonly string[] }} expected
 */
function assertStatusHeader(status, expected) {
  if (status.harness !== "behavioral-gate:dzn") fail(`${expected.path}: unexpected harness ${status.harness}`);
  if (status.verdict !== "PASS") fail(`${expected.path}: committed verdict is ${status.verdict}`);
  if (status.command !== expected.command) fail(`${expected.path}: unexpected command ${status.command}`);
  if (status.filter !== expected.filter) fail(`${expected.path}: unexpected filter ${status.filter}`);
  if (status.goldenVariant !== "dzn-full") fail(`${expected.path}: expected dzn-full golden variant`);
  if (status.exitStatus !== 0) fail(`${expected.path}: exitStatus must be 0, got ${status.exitStatus}`);
  if (status.summary?.totalConfigs !== expected.labels.length || status.summary?.failures !== 0) {
    fail(`${expected.path}: unexpected summary ${JSON.stringify(status.summary)}`);
  }
}

/** @param {Record<string, any>} row @param {string} path */
function assertCwbvhRow(row, path) {
  if (row == null || typeof row !== "object") fail(`${path}: invalid CWBVH row`);
  const label = row.label;
  if (row.verdict !== "PASS") fail(`${path}: ${label} verdict is ${row.verdict}`);
  if (row.rawStatus !== "OK") fail(`${path}: ${label} rawStatus is ${row.rawStatus}`);
  if (row.tier !== "full") fail(`${path}: ${label} tier is ${row.tier}`);
  if (row.gpuErrors !== 0) fail(`${path}: ${label} gpuErrors must be 0, got ${row.gpuErrors}`);
  if (row.nan !== false) fail(`${path}: ${label} nan must be false, got ${row.nan}`);
  if (!(row.luminance >= 0.005)) fail(`${path}: ${label} luminance below bound: ${row.luminance}`);
  if (row.cwbvhParityKind !== "binary") fail(`${path}: ${label} cwbvhParityKind is ${row.cwbvhParityKind}`);
  if (row.cwbvhParityRmse > 1 || row.cwbvhParityMeanAbs > 0.5 || row.cwbvhParityMaxAbs > 8) {
    fail(`${path}: ${label} CWBVH parity exceeds thresholds`);
  }
  if (row.cwbvhPerfKind !== "same-scene") fail(`${path}: ${label} cwbvhPerfKind is ${row.cwbvhPerfKind}`);
  const ratio = requireFinite(row.cwbvhRenderMsRatio, `${path}: ${label} cwbvhRenderMsRatio`);
  if (requireFinite(row.cwbvhBinaryRenderMs, `${path}: ${label} cwbvhBinaryRenderMs`) <= 0) {
    fail(`${path}: ${label} binary timing must be positive`);
  }
  if (requireFinite(row.cwbvhRenderMs, `${path}: ${label} cwbvhRenderMs`) <= 0) {
    fail(`${path}: ${label} cwbvh timing must be positive`);
  }
  if (requireFinite(row.cwbvhBinaryMemoryBytes, `${path}: ${label} cwbvhBinaryMemoryBytes`) <= 0) {
    fail(`${path}: ${label} binary memory must be positive`);
  }
  if (requireFinite(row.cwbvhMemoryBytes, `${path}: ${label} cwbvhMemoryBytes`) <= 0) {
    fail(`${path}: ${label} cwbvh memory must be positive`);
  }
  return { label, ratio };
}

/**
 * @param {Array<{ label: string; ratio: number }>} perfRows
 * @param {Array<{ label: string; ratio: number }>} slowOrNeutral
 * @param {Array<{ label: string; ratio: number }>} fast
 * @param {string} classification
 */
async function assertSummaryStatus(perfRows, slowOrNeutral, fast, classification) {
  const status = JSON.parse(await Deno.readTextFile(SUMMARY_STATUS));
  const repeatStatus = JSON.parse(await Deno.readTextFile(REPEAT_STATUS));
  if (status.harness !== "cwbvh-default-promotion-proof") fail(`${SUMMARY_STATUS}: harness mismatch`);
  if (status.verdict !== "PASS-PARTIAL") fail(`${SUMMARY_STATUS}: verdict must stay PASS-PARTIAL until default promotion is proven`);
  if (status.promotion?.defaultReady !== false) fail(`${SUMMARY_STATUS}: promotion.defaultReady must be false`);
  if (status.promotion?.classification !== classification) {
    fail(`${SUMMARY_STATUS}: promotion.classification expected ${classification}, got ${status.promotion?.classification}`);
  }
  if (!String(status.promotion?.requiredEvidence ?? "").includes("browser/real-adapter throughput A/B")) {
    fail(`${SUMMARY_STATUS}: requiredEvidence must name browser/real-adapter throughput A/B`);
  }
  if (status.measurementSufficiency?.status !== "single-sample-insufficient-for-default-promotion") {
    fail(`${SUMMARY_STATUS}: measurementSufficiency must keep single-sample timing evidence out of default promotion`);
  }
  if (status.measurementSufficiency?.sampleCountPerWorkload !== 1) {
    fail(`${SUMMARY_STATUS}: measurementSufficiency.sampleCountPerWorkload must reflect committed single-run timing rows`);
  }
  if (status.measurementSufficiency?.minRepeatCountPerWorkload !== MIN_REPEAT_COUNT_PER_WORKLOAD) {
    fail(`${SUMMARY_STATUS}: measurementSufficiency.minRepeatCountPerWorkload drifted`);
  }
  if (status.measurementSufficiency?.requiredAdapterScope !== REQUIRED_ADAPTER_SCOPE) {
    fail(`${SUMMARY_STATUS}: measurementSufficiency.requiredAdapterScope must name browser/real-adapter validation`);
  }
  if (status.measurementSufficiency?.defaultPromotionEligible !== false) {
    fail(`${SUMMARY_STATUS}: measurementSufficiency.defaultPromotionEligible must remain false`);
  }
  if (!String(status.measurementSufficiency?.requiredEvidence ?? "").includes("multiple repeats per workload")) {
    fail(`${SUMMARY_STATUS}: measurementSufficiency.requiredEvidence must name repeat-count evidence`);
  }
  if (
    status.repeatEvidence?.status !== "completed-five-sample-warmup-discarded-nonpromoting" ||
    status.repeatEvidence?.harness !== "cwbvh-default-promotion-repeat-proof" ||
    status.repeatEvidence?.sourceStatus !== REPEAT_STATUS ||
    status.repeatEvidence?.sourceRecords !== REPEAT_RECORDS ||
    status.repeatEvidence?.campaignStatus !== repeatStatus.campaignStatus ||
    status.repeatEvidence?.adapterScope !== repeatStatus.adapterScope ||
    status.repeatEvidence?.sampleCountPerWorkload !== repeatStatus.sampleCountPerWorkload ||
    status.repeatEvidence?.warmupDiscardedPerWorkload !== repeatStatus.warmupDiscardedPerWorkload ||
    status.repeatEvidence?.minRepeatCountPerWorkload !== repeatStatus.minRepeatCountPerWorkload ||
    status.repeatEvidence?.classification !== repeatStatus.classification ||
    status.repeatEvidence?.defaultPromotionEligible !== false ||
    status.repeatEvidence?.requiredAdapterScope !== REQUIRED_ADAPTER_SCOPE
  ) {
    fail(`${SUMMARY_STATUS}: repeatEvidence must mirror the completed five-sample repeat status without promoting CWBVH`);
  }
  if (
    !String(status.repeatEvidence?.requiredEvidence ?? "").includes("browser/real-adapter throughput A/B") ||
    !String(status.repeatEvidence?.residual ?? "").includes("one material-lobe-map fast outlier")
  ) {
    fail(`${SUMMARY_STATUS}: repeatEvidence must preserve the browser/adapter tail and material-lobe-map outlier`);
  }
  if (status.thresholds?.slowOrNeutralRatio !== MIN_SLOW_OR_NEUTRAL_RATIO) {
    fail(`${SUMMARY_STATUS}: slowOrNeutralRatio threshold drifted`);
  }
  if (status.thresholds?.fastRatio !== MIN_FAST_RATIO) {
    fail(`${SUMMARY_STATUS}: fastRatio threshold drifted`);
  }
  if (status.thresholds?.minSlowOrNeutralRows !== MIN_SLOW_OR_NEUTRAL_ROWS) {
    fail(`${SUMMARY_STATUS}: minSlowOrNeutralRows threshold drifted`);
  }
  if (status.thresholds?.minRepeatCountPerWorkload !== MIN_REPEAT_COUNT_PER_WORKLOAD) {
    fail(`${SUMMARY_STATUS}: minRepeatCountPerWorkload threshold drifted`);
  }
  if (status.rowCount !== perfRows.length) fail(`${SUMMARY_STATUS}: rowCount mismatch`);
  if (status.slowOrNeutralCount !== slowOrNeutral.length) fail(`${SUMMARY_STATUS}: slowOrNeutralCount mismatch`);
  if (status.fastCount !== fast.length) fail(`${SUMMARY_STATUS}: fastCount mismatch`);
  const expectedRatios = perfRows.map((row) => ({ label: row.label, ratio: row.ratio }));
  if (JSON.stringify(status.ratios) !== JSON.stringify(expectedRatios)) {
    fail(`${SUMMARY_STATUS}: ratios do not match committed dzn CWBVH status rows`);
  }
  const expectedSources = STATUS_FILES.map((entry) => entry.path);
  if (JSON.stringify(status.sourceStatuses) !== JSON.stringify(expectedSources)) {
    fail(`${SUMMARY_STATUS}: sourceStatuses do not match checker inputs`);
  }
}

async function assertRepeatCaptureStatus() {
  const status = JSON.parse(await Deno.readTextFile(REPEAT_STATUS));
  const records = JSON.parse(await Deno.readTextFile(REPEAT_RECORDS));
  const allRecords = /** @type {Array<Record<string, any>>} */ (Array.isArray(records.records) ? records.records : []);
  const warmupRecords = allRecords.filter((record) => record?.phase === "warmup");
  const sampleRecords = allRecords.filter((record) => record?.phase === "sample");
  if (
    status.harness !== "cwbvh-default-promotion-repeat-proof" ||
    status.command !== "node tools/behavioral-gate/run-cwbvh-default-promotion-repeats.mjs --repeats=5 --warmup=1 --dzn-timeout-ms=900000" ||
    status.recordsPath !== REPEAT_RECORDS ||
    status.verdict !== "PASS-PARTIAL" ||
    status.campaignStatus !== "complete" ||
    status.classification !== "uniform-slower" ||
    status.requestedRepeatsPerWorkload !== MIN_REPEAT_COUNT_PER_WORKLOAD ||
    status.warmupDiscardedPerWorkload !== 1 ||
    status.sampleCountPerWorkload !== MIN_REPEAT_COUNT_PER_WORKLOAD ||
    status.dznTimeoutMs !== 900000 ||
    status.allWorkloadsHaveAnySamples !== true ||
    status.allWorkloadsHaveRequiredRepeats !== true ||
    status.promotion?.defaultReady !== false
  ) {
    fail(`${REPEAT_STATUS}: must pin the completed five-sample warmup-discarded uniform-slower repeat campaign`);
  }
  if (
    records.harness !== "cwbvh-default-promotion-repeat-records" ||
    records.campaignStatus !== "complete" ||
    records.failure !== null ||
    allRecords.length !== 18 ||
    warmupRecords.length !== 3 ||
    sampleRecords.length !== 15 ||
    !allRecords.every(isCompletedDznRepeatRecord)
  ) {
    fail(`${REPEAT_RECORDS}: must preserve the completed five-sample dzn repeat records plus one warmup per shard`);
  }
  const recomputed = summarizeRepeatRecords(allRecords, status.warmupDiscardedPerWorkload);
  if (JSON.stringify(status.workloads) !== JSON.stringify(recomputed.workloads)) {
    fail(`${REPEAT_STATUS}: workloads must match ${REPEAT_RECORDS}`);
  }
  if (
    status.sampleCountPerWorkload !== recomputed.sampleCountPerWorkload ||
    status.allWorkloadsHaveAnySamples !== recomputed.allWorkloadsHaveAnySamples ||
    status.allWorkloadsHaveRequiredRepeats !== recomputed.allWorkloadsHaveRequiredRepeats ||
    status.classification !== recomputed.classification
  ) {
    fail(`${REPEAT_STATUS}: aggregate repeat fields must match ${REPEAT_RECORDS}`);
  }
}

/**
 * @param {Array<Record<string, any>>} records
 * @param {number} warmupCount
 */
function summarizeRepeatRecords(records, warmupCount) {
  const expectedLabels = STATUS_FILES.flatMap((entry) => entry.labels);
  /** @type {Map<string, number[]>} */
  const byLabel = new Map(expectedLabels.map((label) => [label, []]));
  /** @type {Map<string, number>} */
  const runCountByLabel = new Map(expectedLabels.map((label) => [label, 0]));
  for (const record of records) {
    const expected = STATUS_FILES.find((entry) => entry.filter === record?.filter);
    if (expected == null) fail(`${REPEAT_RECORDS}: unexpected filter ${record?.filter}`);
    const recordStatus = record.status;
    assertStatusHeader(recordStatus, expected);
    if (recordStatus.timeoutMs !== 900000) fail(`${REPEAT_RECORDS}: ${record.filter} timeoutMs must be 900000`);
    const configs = /** @type {Array<Record<string, any>>} */ (recordStatus.configs ?? []);
    for (const label of expected.labels) {
      const row = configs.find((entry) => entry.label === label);
      if (row == null) fail(`${REPEAT_RECORDS}: ${record.filter} missing ${label}`);
      const { ratio } = assertCwbvhRow(row, REPEAT_RECORDS);
      runCountByLabel.set(label, (runCountByLabel.get(label) ?? 0) + 1);
      if (record.runIndex >= warmupCount) byLabel.get(label)?.push(ratio);
    }
  }
  const workloads = expectedLabels.map((label) => {
    const ratios = byLabel.get(label) ?? [];
    return {
      label,
      totalRunCount: runCountByLabel.get(label) ?? 0,
      sampleCount: ratios.length,
      ratios,
      minRatio: ratios.length ? Math.min(...ratios) : null,
      medianRatio: median(ratios),
      meanRatio: mean(ratios),
      maxRatio: ratios.length ? Math.max(...ratios) : null,
      fastSampleCount: ratios.filter((ratio) => ratio < MIN_FAST_RATIO).length,
      slowOrNeutralSampleCount: ratios.filter((ratio) => ratio >= MIN_SLOW_OR_NEUTRAL_RATIO).length,
    };
  });
  const sampleCountPerWorkload = Math.min(...workloads.map((row) => row.sampleCount));
  const allWorkloadsHaveAnySamples = sampleCountPerWorkload > 0;
  const allWorkloadsHaveRequiredRepeats = sampleCountPerWorkload >= MIN_REPEAT_COUNT_PER_WORKLOAD;
  const allMedianFast = allWorkloadsHaveAnySamples && workloads.every((row) => Number(row.medianRatio) < MIN_FAST_RATIO);
  const allMedianSlowOrNeutral = allWorkloadsHaveAnySamples && workloads.every((row) => Number(row.medianRatio) >= MIN_SLOW_OR_NEUTRAL_RATIO);
  const classification = !allWorkloadsHaveAnySamples
    ? "insufficient-samples"
    : (allMedianFast ? "uniform-faster" : (allMedianSlowOrNeutral ? "uniform-slower" : "mixed"));
  return {
    workloads,
    sampleCountPerWorkload,
    allWorkloadsHaveAnySamples,
    allWorkloadsHaveRequiredRepeats,
    classification,
  };
}

const source = await Deno.readTextFile(SOURCE_GUARD);
assertSourceStillOptIn(source);
const repeatHarness = await Deno.readTextFile(REPEAT_HARNESS);
assertRepeatHarness(repeatHarness);

/** @type {Array<{ label: string; ratio: number }>} */
const perfRows = [];
for (const expected of STATUS_FILES) {
  const status = JSON.parse(await Deno.readTextFile(expected.path));
  assertStatusHeader(status, expected);
  for (const label of expected.labels) {
    const configs = /** @type {Array<Record<string, any>>} */ (status.configs ?? []);
    const row = configs.find((entry) => entry.label === label);
    if (row == null) fail(`${expected.path}: missing ${label}`);
    perfRows.push(assertCwbvhRow(row, expected.path));
  }
}

if (perfRows.length !== 5) fail(`expected 5 CWBVH performance rows, got ${perfRows.length}`);
const slowOrNeutral = perfRows.filter((row) => row.ratio >= MIN_SLOW_OR_NEUTRAL_RATIO);
const fast = perfRows.filter((row) => row.ratio < MIN_FAST_RATIO);
if (slowOrNeutral.length < MIN_SLOW_OR_NEUTRAL_ROWS) {
  fail(
    `default-promotion blocker disappeared: expected at least two slow/neutral rows ` +
      `(ratio >= ${MIN_SLOW_OR_NEUTRAL_RATIO}), got ${slowOrNeutral.length}`,
  );
}

const ratioSummary = perfRows.map((row) => `${row.label}=${row.ratio.toFixed(3)}`).join(", ");
const classification = fast.length > 0 ? "mixed" : "uniform-slower";
await assertSummaryStatus(perfRows, slowOrNeutral, fast, classification);
await assertRepeatCaptureStatus();
console.log(
  `[cwbvh-default-promotion-proof-check] PASS (CWBVH remains opt-in; ${classification} dzn ratios: ${ratioSummary})`,
);
