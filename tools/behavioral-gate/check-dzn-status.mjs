#!/usr/bin/env -S deno run --sloppy-imports --allow-read
// @ts-check
// Verifies committed WSL dzn behavioral-gate status artifacts.

const EXPECTED = [
  {
    path: "tools/behavioral-gate/behavioral-gate-dzn-gltf-material-sweep-status.json",
    command: "npm run behavioral-gate:dzn -- --filter gltf-material-sweep --require-full-tier",
    filter: "gltf-material-sweep",
    verdict: "PASS",
    exitStatus: 0,
    totalConfigs: 1,
    failures: 0,
    configs: [
      {
        label: "pt/gltf-material-sweep",
        verdict: "PASS",
        rawStatus: "OK",
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
    verdict: "PASS",
    exitStatus: 0,
    totalConfigs: 3,
    failures: 0,
    configs: [
      { label: "pt/mutation-material", verdict: "PASS", rawStatus: "OK", tier: "full", mutationKind: "material", minMutationMeanAbs: 2, minMutationMaxAbs: 8 },
      { label: "pt/mutation-environment", verdict: "PASS", rawStatus: "OK", tier: "full", mutationKind: "environment", minMutationMeanAbs: 2, minMutationMaxAbs: 8 },
      { label: "pt/mutation-emitter", verdict: "PASS", rawStatus: "OK", tier: "full", mutationKind: "emitter", minMutationMeanAbs: 2, minMutationMaxAbs: 8 },
    ],
  },
  {
    path: "tools/behavioral-gate/behavioral-gate-dzn-gltf-status.json",
    command: "npm run behavioral-gate:dzn -- --filter gltf --require-full-tier",
    filter: "gltf",
    verdict: "FAIL",
    exitStatus: 1,
    totalConfigs: 11,
    failures: 3,
    configs: [
      { label: "pt/gltf-unlit", verdict: "PASS", rawStatus: "OK", tier: "full" },
      { label: "pt/gltf-textured-pbr", verdict: "PASS", rawStatus: "OK", tier: "full" },
      { label: "pt/gltf-transmission", verdict: "PASS", rawStatus: "OK", tier: "full" },
      { label: "pt/gltf-skinned-animation", verdict: "PASS", rawStatus: "OK", tier: "full" },
      { label: "pt/gltf-draco-mock", verdict: "PASS", rawStatus: "OK", tier: "full" },
      { label: "pt/gltf-point-line-fallback", verdict: "PASS", rawStatus: "OK", tier: "full", goldenStatus: "ok", maxRmse: 8, maxMeanAbs: 4, maxAbs: 48 },
      { label: "pt/gltf-triangle-strip-fan", verdict: "PASS", rawStatus: "OK", tier: "full", goldenStatus: "ok", maxRmse: 8, maxMeanAbs: 4, maxAbs: 48 },
      { label: "pt/gltf-material-sweep", verdict: "PASS", rawStatus: "OK", tier: "full", goldenStatus: "ok", maxRmse: 8, maxMeanAbs: 4, maxAbs: 48 },
      { label: "pt/gltf-real-box-textured", verdict: "FAIL", rawStatus: "GOLDEN-DELTA", tier: "full", goldenStatus: "FAIL", minGoldenRmse: 8, minGoldenMaxAbs: 48 },
      { label: "pt/gltf-real-draco", verdict: "FAIL", rawStatus: "GOLDEN-DELTA", tier: "full", goldenStatus: "FAIL", minGoldenRmse: 8, minGoldenMaxAbs: 48 },
      { label: "pt/gltf-real-meshopt", verdict: "FAIL", rawStatus: "GOLDEN-DELTA", tier: "full", goldenStatus: "FAIL", minGoldenRmse: 8, minGoldenMaxAbs: 48 },
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
  if (status.verdict !== expected.verdict) fail(`${expected.path}: verdict must be ${expected.verdict}, got ${status.verdict}`);
  if (status.command !== expected.command) fail(`${expected.path}: command mismatch`);
  if (status.filter !== expected.filter) fail(`${expected.path}: filter mismatch`);
  if (status.exitStatus !== expected.exitStatus) fail(`${expected.path}: exitStatus must be ${expected.exitStatus}`);
  if (status.signal != null) fail(`${expected.path}: signal must be null`);
  if (status.summary?.totalConfigs !== expected.totalConfigs) fail(`${expected.path}: totalConfigs mismatch`);
  if (status.summary?.failures !== expected.failures) fail(`${expected.path}: failures mismatch`);
  if (status.summary?.knownResiduals !== 0) fail(`${expected.path}: knownResiduals must be 0`);

  const byLabel = new Map((status.configs ?? []).map((config) => [config.label, config]));
  for (const expectedConfig of expected.configs) {
    const config = byLabel.get(expectedConfig.label);
    if (config == null) fail(`${expected.path}: missing ${expectedConfig.label}`);
    if (config.verdict !== expectedConfig.verdict) fail(`${expected.path}: ${expectedConfig.label} verdict mismatch`);
    if (config.rawStatus !== expectedConfig.rawStatus) fail(`${expected.path}: ${expectedConfig.label} rawStatus mismatch`);
    if (config.tier !== expectedConfig.tier) fail(`${expected.path}: ${expectedConfig.label} tier mismatch`);
    if (config.gpuErrors !== 0) fail(`${expected.path}: ${expectedConfig.label} gpuErrors must be 0`);
    if (config.nan !== false) fail(`${expected.path}: ${expectedConfig.label} nan must be false`);

    if (expectedConfig.goldenStatus != null) {
      if (config.goldenStatus !== expectedConfig.goldenStatus) {
        fail(`${expected.path}: ${expectedConfig.label} goldenStatus mismatch`);
      }
      if (expectedConfig.maxRmse != null && expectedConfig.maxMeanAbs != null && expectedConfig.maxAbs != null) {
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
    }

    if (expectedConfig.minGoldenRmse != null && !(config.rmse > expectedConfig.minGoldenRmse)) {
      fail(`${expected.path}: ${expectedConfig.label} RMSE should preserve known dzn golden-delta finding`);
    }
    if (expectedConfig.minGoldenMaxAbs != null && !(config.maxAbs > expectedConfig.minGoldenMaxAbs)) {
      fail(`${expected.path}: ${expectedConfig.label} maxAbs should preserve known dzn golden-delta finding`);
    }

    if (expectedConfig.mutationKind != null) {
      if (config.mutationKind !== expectedConfig.mutationKind) {
        fail(`${expected.path}: ${expectedConfig.label} mutationKind mismatch`);
      }
      if (config.mutationMeanAbs < expectedConfig.minMutationMeanAbs) {
        fail(`${expected.path}: ${expectedConfig.label} mutationMeanAbs below bound`);
      }
      if (config.mutationMaxAbs < expectedConfig.minMutationMaxAbs) {
        fail(`${expected.path}: ${expectedConfig.label} mutationMaxAbs below bound`);
      }
    }
  }
}

console.log("[behavioral-gate-dzn-status-check] PASS (3 committed dzn status artifacts: 2 pass, 1 known finding)");
