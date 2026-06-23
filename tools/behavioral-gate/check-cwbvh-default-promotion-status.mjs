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
const MIN_SLOW_OR_NEUTRAL_RATIO = 1.0;
const MIN_FAST_RATIO = 0.95;

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

const source = await Deno.readTextFile(SOURCE_GUARD);
assertSourceStillOptIn(source);

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
if (slowOrNeutral.length < 2) {
  fail(
    `default-promotion blocker disappeared: expected at least two slow/neutral rows ` +
      `(ratio >= ${MIN_SLOW_OR_NEUTRAL_RATIO}), got ${slowOrNeutral.length}`,
  );
}

const ratioSummary = perfRows.map((row) => `${row.label}=${row.ratio.toFixed(3)}`).join(", ");
const classification = fast.length > 0 ? "mixed" : "uniform-slower";
console.log(
  `[cwbvh-default-promotion-proof-check] PASS (CWBVH remains opt-in; ${classification} dzn ratios: ${ratioSummary})`,
);
