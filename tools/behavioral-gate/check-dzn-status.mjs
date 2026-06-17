#!/usr/bin/env -S deno run --sloppy-imports --allow-read
// @ts-check
// Verifies committed WSL dzn behavioral-gate status artifacts.

const EXPECTED = [
  {
    path: "tools/behavioral-gate/behavioral-gate-dzn-gltf-material-sweep-status.json",
    command: "npm run behavioral-gate:dzn -- --filter gltf-material-sweep --require-full-tier",
    filter: "gltf-material-sweep",
    totalConfigs: 1,
    configs: [
      {
        label: "pt/gltf-material-sweep",
        tier: "full",
        goldenStatus: "ok",
        maxRmse: 8,
        maxMeanAbs: 4,
        maxAbs: 48,
      },
    ],
  },
  {
    path: "tools/behavioral-gate/behavioral-gate-dzn-mutation-status.json",
    command: "npm run behavioral-gate:dzn -- --filter mutation --require-full-tier",
    filter: "mutation",
    totalConfigs: 3,
    configs: [
      { label: "pt/mutation-material", tier: "full", mutationKind: "material", minMeanAbs: 2, minMaxAbs: 8 },
      { label: "pt/mutation-environment", tier: "full", mutationKind: "environment", minMeanAbs: 2, minMaxAbs: 8 },
      { label: "pt/mutation-emitter", tier: "full", mutationKind: "emitter", minMeanAbs: 2, minMaxAbs: 8 },
    ],
  },
];

function fail(message) {
  throw new Error(`[behavioral-gate-dzn-status-check] ${message}`);
}

function sameJson(a, b) {
  return JSON.stringify(a) === JSON.stringify(b);
}

for (const expected of EXPECTED) {
  const url = new URL(`../../${expected.path}`, import.meta.url);
  const status = JSON.parse(await Deno.readTextFile(url));

  if (status.harness !== "behavioral-gate:dzn") fail(`${expected.path}: harness mismatch`);
  if (status.verdict !== "PASS") fail(`${expected.path}: verdict must be PASS, got ${status.verdict}`);
  if (status.command !== expected.command) fail(`${expected.path}: command mismatch`);
  if (status.filter !== expected.filter) fail(`${expected.path}: filter mismatch`);
  if (status.exitStatus !== 0) fail(`${expected.path}: exitStatus must be 0`);
  if (status.signal != null) fail(`${expected.path}: signal must be null`);
  if (status.summary?.totalConfigs !== expected.totalConfigs) fail(`${expected.path}: totalConfigs mismatch`);
  if (status.summary?.failures !== 0) fail(`${expected.path}: failures must be 0`);
  if (status.summary?.knownResiduals !== 0) fail(`${expected.path}: knownResiduals must be 0`);

  const byLabel = new Map((status.configs ?? []).map((config) => [config.label, config]));
  for (const expectedConfig of expected.configs) {
    const config = byLabel.get(expectedConfig.label);
    if (config == null) fail(`${expected.path}: missing ${expectedConfig.label}`);
    if (config.verdict !== "PASS") fail(`${expected.path}: ${expectedConfig.label} verdict mismatch`);
    if (config.rawStatus !== "OK") fail(`${expected.path}: ${expectedConfig.label} rawStatus mismatch`);
    if (config.tier !== expectedConfig.tier) fail(`${expected.path}: ${expectedConfig.label} tier mismatch`);
    if (config.gpuErrors !== 0) fail(`${expected.path}: ${expectedConfig.label} gpuErrors must be 0`);
    if (config.nan !== false) fail(`${expected.path}: ${expectedConfig.label} nan must be false`);

    if (expectedConfig.goldenStatus != null) {
      if (config.goldenStatus !== expectedConfig.goldenStatus) {
        fail(`${expected.path}: ${expectedConfig.label} goldenStatus mismatch`);
      }
      if (config.rmse > expectedConfig.maxRmse) fail(`${expected.path}: ${expectedConfig.label} RMSE exceeds bound`);
      if (config.meanAbs > expectedConfig.maxMeanAbs) fail(`${expected.path}: ${expectedConfig.label} meanAbs exceeds bound`);
      if (config.maxAbs > expectedConfig.maxAbs) fail(`${expected.path}: ${expectedConfig.label} maxAbs exceeds bound`);
      if (!sameJson(config.thresholds, {
        maxRmse: expectedConfig.maxRmse,
        maxMeanAbs: expectedConfig.maxMeanAbs,
        maxAbs: expectedConfig.maxAbs,
      })) {
        fail(`${expected.path}: ${expectedConfig.label} thresholds mismatch`);
      }
    }

    if (expectedConfig.mutationKind != null) {
      if (config.mutationKind !== expectedConfig.mutationKind) {
        fail(`${expected.path}: ${expectedConfig.label} mutationKind mismatch`);
      }
      if (config.mutationMeanAbs < expectedConfig.minMeanAbs) {
        fail(`${expected.path}: ${expectedConfig.label} mutationMeanAbs below bound`);
      }
      if (config.mutationMaxAbs < expectedConfig.minMaxAbs) {
        fail(`${expected.path}: ${expectedConfig.label} mutationMaxAbs below bound`);
      }
    }
  }
}

console.log("[behavioral-gate-dzn-status-check] PASS (2 committed dzn status artifacts)");
