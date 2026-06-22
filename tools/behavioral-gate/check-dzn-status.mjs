#!/usr/bin/env -S deno run --sloppy-imports --allow-read
// @ts-check
// Verifies committed WSL dzn behavioral-gate status artifacts.

/** @typedef {Record<string, any>} JsonRecord */
/**
 * @typedef {JsonRecord & {
 *   label: string,
 *   verdict: string,
 *   rawStatus: string,
 *   tier?: string | null,
 * }} ExpectedConfig
 */
/**
 * @typedef {{
 *   path: string,
 *   command: string,
 *   filter: string,
 *   goldenVariant?: string,
 *   verdict: string,
 *   exitStatus: number,
 *   totalConfigs: number,
 *   failures: number,
 *   configs: ExpectedConfig[],
 * }} ExpectedStatus
 */

const EXPECTED = /** @type {ExpectedStatus[]} */ ([
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
    goldenVariant: "dzn-full",
    verdict: "PASS",
    exitStatus: 0,
    totalConfigs: 8,
    failures: 0,
    configs: [
      { label: "pt/mutation-material", verdict: "PASS", rawStatus: "OK", tier: "full", mutationKind: "material", minMutationMeanAbs: 2, minMutationMaxAbs: 8 },
      { label: "pt/mutation-environment", verdict: "PASS", rawStatus: "OK", tier: "full", mutationKind: "environment", minMutationMeanAbs: 2, minMutationMaxAbs: 8 },
      { label: "pt/mutation-emitter", verdict: "PASS", rawStatus: "OK", tier: "full", mutationKind: "emitter", minMutationMeanAbs: 2, minMutationMaxAbs: 8 },
      { label: "pt/mutation-transform", verdict: "PASS", rawStatus: "OK", tier: "full", mutationKind: "transform", minMutationMeanAbs: 2, minMutationMaxAbs: 8 },
      { label: "pt/mutation-topology", verdict: "PASS", rawStatus: "OK", tier: "full", mutationKind: "topology", minMutationMeanAbs: 2, minMutationMaxAbs: 8 },
      { label: "pt/mutation-instanced-count", verdict: "PASS", rawStatus: "OK", tier: "full", mutationKind: "instanced-count", minMutationMeanAbs: 2, minMutationMaxAbs: 8 },
      { label: "pt/mutation-add-primitive", verdict: "PASS", rawStatus: "OK", tier: "full", mutationKind: "add-primitive", minMutationMeanAbs: 2, minMutationMaxAbs: 8 },
      { label: "pt/mutation-remove-primitive", verdict: "PASS", rawStatus: "OK", tier: "full", mutationKind: "remove-primitive", minMutationMeanAbs: 2, minMutationMaxAbs: 8 },
    ],
  },
  {
    path: "tools/behavioral-gate/behavioral-gate-dzn-wh-mutation-status.json",
    command: "npm run behavioral-gate:dzn -- --filter wh/mutation --require-full-tier",
    filter: "wh/mutation",
    goldenVariant: "dzn-full",
    verdict: "PASS",
    exitStatus: 0,
    totalConfigs: 8,
    failures: 0,
    configs: [
      { label: "wh/mutation-material", verdict: "PASS", rawStatus: "OK", tier: null, minLuminance: 0.005, mutationKind: "material", minMutationMeanAbs: 2, minMutationMaxAbs: 8 },
      { label: "wh/mutation-environment", verdict: "PASS", rawStatus: "OK", tier: null, minLuminance: 0.005, mutationKind: "environment", minMutationMeanAbs: 2, minMutationMaxAbs: 8 },
      { label: "wh/mutation-emitter", verdict: "PASS", rawStatus: "OK", tier: null, minLuminance: 0.005, mutationKind: "emitter", minMutationMeanAbs: 2, minMutationMaxAbs: 8 },
      { label: "wh/mutation-transform", verdict: "PASS", rawStatus: "OK", tier: null, minLuminance: 0.005, mutationKind: "transform", minMutationMeanAbs: 2, minMutationMaxAbs: 8 },
      { label: "wh/mutation-topology", verdict: "PASS", rawStatus: "OK", tier: null, minLuminance: 0.005, mutationKind: "topology", minMutationMeanAbs: 2, minMutationMaxAbs: 8 },
      { label: "wh/mutation-instanced-count", verdict: "PASS", rawStatus: "OK", tier: null, minLuminance: 0.005, mutationKind: "instanced-count", minMutationMeanAbs: 2, minMutationMaxAbs: 8 },
      { label: "wh/mutation-add-primitive", verdict: "PASS", rawStatus: "OK", tier: null, minLuminance: 0.005, mutationKind: "add-primitive", minMutationMeanAbs: 2, minMutationMaxAbs: 8 },
      { label: "wh/mutation-remove-primitive", verdict: "PASS", rawStatus: "OK", tier: null, minLuminance: 0.005, mutationKind: "remove-primitive", minMutationMeanAbs: 2, minMutationMaxAbs: 8 },
    ],
  },
  {
    path: "tools/behavioral-gate/behavioral-gate-dzn-default-status.json",
    command: "npm run behavioral-gate:dzn -- --filter default --require-full-tier",
    filter: "default",
    goldenVariant: "dzn-full",
    verdict: "PASS",
    exitStatus: 0,
    totalConfigs: 2,
    failures: 0,
    configs: [
      { label: "pt/default", verdict: "PASS", rawStatus: "OK", tier: "full", minLuminance: 0.005 },
      { label: "wh/default", verdict: "PASS", rawStatus: "OK", tier: null, minLuminance: 0.005 },
    ],
  },
  {
    path: "tools/behavioral-gate/behavioral-gate-dzn-lite-tier-status.json",
    command: "npm run behavioral-gate:dzn -- --filter lite-tier --require-full-tier",
    filter: "lite-tier",
    goldenVariant: "dzn-full",
    verdict: "PASS",
    exitStatus: 0,
    totalConfigs: 1,
    failures: 0,
    configs: [
      { label: "pt/lite-tier", verdict: "PASS", rawStatus: "OK", tier: "lite", minLuminance: 0.005 },
    ],
  },
  {
    path: "tools/behavioral-gate/behavioral-gate-dzn-sobol-default-status.json",
    command: "npm run behavioral-gate:dzn -- --filter sobol-default --require-full-tier",
    filter: "sobol-default",
    goldenVariant: "dzn-full",
    verdict: "PASS",
    exitStatus: 0,
    totalConfigs: 1,
    failures: 0,
    configs: [
      { label: "pt/sobol-default", verdict: "PASS", rawStatus: "OK", tier: "full", minLuminance: 0.005 },
    ],
  },
  {
    path: "tools/behavioral-gate/behavioral-gate-dzn-sobol-bdpt-status.json",
    command: "npm run behavioral-gate:dzn -- --filter sobol-bdpt --require-full-tier",
    filter: "sobol-bdpt",
    goldenVariant: "dzn-full",
    verdict: "PASS",
    exitStatus: 0,
    totalConfigs: 1,
    failures: 0,
    configs: [
      { label: "pt/sobol-bdpt", verdict: "PASS", rawStatus: "OK", tier: "full", minLuminance: 0.005 },
    ],
  },
  {
    path: "tools/behavioral-gate/behavioral-gate-dzn-sobol-lite-status.json",
    command: "npm run behavioral-gate:dzn -- --filter sobol-lite --require-full-tier",
    filter: "sobol-lite",
    goldenVariant: "dzn-full",
    verdict: "PASS",
    exitStatus: 0,
    totalConfigs: 1,
    failures: 0,
    configs: [
      { label: "pt/sobol-lite", verdict: "PASS", rawStatus: "OK", tier: "lite", minLuminance: 0.005 },
    ],
  },
  {
    path: "tools/behavioral-gate/behavioral-gate-dzn-sobol-restirptreuse-status.json",
    command: "npm run behavioral-gate:dzn -- --filter sobol-restirPtReuse --require-full-tier",
    filter: "sobol-restirPtReuse",
    goldenVariant: "dzn-full",
    verdict: "PASS",
    exitStatus: 0,
    totalConfigs: 1,
    failures: 0,
    configs: [
      { label: "pt/sobol-restirPtReuse", verdict: "PASS", rawStatus: "OK", tier: "full", minLuminance: 0.005 },
    ],
  },
  {
    path: "tools/behavioral-gate/behavioral-gate-dzn-cwbvh-binary-parity-status.json",
    command: "npm run behavioral-gate:dzn -- --filter cwbvh-binary-parity --require-full-tier",
    filter: "cwbvh-binary-parity",
    goldenVariant: "dzn-full",
    verdict: "PASS",
    exitStatus: 0,
    totalConfigs: 1,
    failures: 0,
    configs: [
      {
        label: "pt/cwbvh-binary-parity",
        verdict: "PASS",
        rawStatus: "OK",
        tier: "full",
        minLuminance: 0.005,
        cwbvhParityKind: "binary",
        maxCwbvhParityRmse: 1,
        maxCwbvhParityMeanAbs: 0.5,
        maxCwbvhParityMaxAbs: 8,
        minCwbvhRenderMsRatio: 1.0,
      },
    ],
  },
  {
    path: "tools/behavioral-gate/behavioral-gate-dzn-cwbvh-complex-parity-status.json",
    command: "npm run behavioral-gate:dzn -- --filter cwbvh-complex-parity --require-full-tier",
    filter: "cwbvh-complex-parity",
    goldenVariant: "dzn-full",
    verdict: "PASS",
    exitStatus: 0,
    totalConfigs: 1,
    failures: 0,
    configs: [
      {
        label: "pt/cwbvh-complex-parity",
        verdict: "PASS",
        rawStatus: "OK",
        tier: "full",
        minLuminance: 0.005,
        cwbvhParityKind: "binary",
        maxCwbvhParityRmse: 1,
        maxCwbvhParityMeanAbs: 0.5,
        maxCwbvhParityMaxAbs: 8,
        minCwbvhRenderMsRatio: 1.0,
      },
    ],
  },
  {
    path: "tools/behavioral-gate/behavioral-gate-dzn-spectral-status.json",
    command: "npm run behavioral-gate:dzn -- --filter spectral --require-full-tier",
    filter: "spectral",
    goldenVariant: "dzn-full",
    verdict: "PASS",
    exitStatus: 0,
    totalConfigs: 3,
    failures: 0,
    configs: [
      { label: "pt/spectral", verdict: "PASS", rawStatus: "OK", tier: "full", minLuminance: 0.005 },
      { label: "pt/spectral+photon", verdict: "PASS", rawStatus: "OK", tier: "full", minLuminance: 0.005 },
      { label: "pt/spectral+bdpt", verdict: "PASS", rawStatus: "OK", tier: "full", minLuminance: 0.005 },
    ],
  },
  {
    path: "tools/behavioral-gate/behavioral-gate-dzn-skinned-status.json",
    command: "npm run behavioral-gate:dzn -- --filter skinned --require-full-tier",
    filter: "skinned",
    goldenVariant: "dzn-full",
    verdict: "PASS",
    exitStatus: 0,
    totalConfigs: 3,
    failures: 0,
    configs: [
      { label: "pt/skinned-mesh", verdict: "PASS", rawStatus: "OK", tier: "full", minLuminance: 0.005 },
      { label: "pt/gltf-skinned-animation", verdict: "PASS", rawStatus: "OK", tier: "full", minLuminance: 0.005 },
      { label: "wh/skinned-mesh", verdict: "PASS", rawStatus: "OK", tier: null, minLuminance: 0.005 },
    ],
  },
  {
    path: "tools/behavioral-gate/behavioral-gate-dzn-analytic-status.json",
    command: "npm run behavioral-gate:dzn -- --filter analytic --require-full-tier",
    filter: "analytic",
    goldenVariant: "dzn-full",
    verdict: "PASS",
    exitStatus: 0,
    totalConfigs: 1,
    failures: 0,
    configs: [
      { label: "pt/analytic-sphere", verdict: "PASS", rawStatus: "OK", tier: "full", minLuminance: 0.005 },
    ],
  },
  {
    path: "tools/behavioral-gate/behavioral-gate-dzn-bdpt-status.json",
    command: "npm run behavioral-gate:dzn -- --filter bdpt --require-full-tier",
    filter: "bdpt",
    goldenVariant: "dzn-full",
    verdict: "PASS",
    exitStatus: 0,
    totalConfigs: 2,
    failures: 0,
    configs: [
      { label: "pt/bdpt", verdict: "PASS", rawStatus: "OK", tier: "full", minLuminance: 0.005 },
      { label: "pt/spectral+bdpt", verdict: "PASS", rawStatus: "OK", tier: "full", minLuminance: 0.005 },
    ],
  },
  {
    path: "tools/behavioral-gate/behavioral-gate-dzn-restirptreuse-status.json",
    command: "npm run behavioral-gate:dzn -- --filter restirPtReuse --require-full-tier",
    filter: "restirPtReuse",
    goldenVariant: "dzn-full",
    verdict: "PASS",
    exitStatus: 0,
    totalConfigs: 1,
    failures: 0,
    configs: [
      { label: "pt/restirPtReuse", verdict: "PASS", rawStatus: "OK", tier: "full", minLuminance: 0.005 },
    ],
  },
  {
    path: "tools/behavioral-gate/behavioral-gate-dzn-caustic-status.json",
    command: "npm run behavioral-gate:dzn -- --filter caustic --require-full-tier",
    filter: "caustic",
    goldenVariant: "dzn-full",
    verdict: "PASS",
    exitStatus: 0,
    totalConfigs: 2,
    failures: 0,
    configs: [
      { label: "pt/caustic-manifold", verdict: "PASS", rawStatus: "OK", tier: "full", minLuminance: 0.005 },
      { label: "pt/caustic-photon", verdict: "PASS", rawStatus: "OK", tier: "full", minLuminance: 0.005 },
    ],
  },
  {
    path: "tools/behavioral-gate/behavioral-gate-dzn-photon-status.json",
    command: "npm run behavioral-gate:dzn -- --filter photon --require-full-tier",
    filter: "photon",
    goldenVariant: "dzn-full",
    verdict: "PASS",
    exitStatus: 0,
    totalConfigs: 2,
    failures: 0,
    configs: [
      { label: "pt/caustic-photon", verdict: "PASS", rawStatus: "OK", tier: "full", minLuminance: 0.005 },
      { label: "pt/spectral+photon", verdict: "PASS", rawStatus: "OK", tier: "full", minLuminance: 0.005 },
    ],
  },
  {
    path: "tools/behavioral-gate/behavioral-gate-dzn-light-status.json",
    command: "npm run behavioral-gate:dzn -- --filter light --require-full-tier",
    filter: "light",
    goldenVariant: "dzn-full",
    verdict: "PASS",
    exitStatus: 0,
    totalConfigs: 4,
    failures: 0,
    configs: [
      { label: "pt/point-light", verdict: "PASS", rawStatus: "OK", tier: "full", minLuminance: 0.005 },
      { label: "pt/disc-light", verdict: "PASS", rawStatus: "OK", tier: "full", minLuminance: 0.005 },
      { label: "pt/spot-light", verdict: "PASS", rawStatus: "OK", tier: "full", minLuminance: 0.005 },
      { label: "pt/lite+point-light", verdict: "PASS", rawStatus: "OK", tier: "lite", minLuminance: 0.005 },
    ],
  },
  {
    path: "tools/behavioral-gate/behavioral-gate-dzn-directional-status.json",
    command: "npm run behavioral-gate:dzn -- --filter directional --require-full-tier",
    filter: "directional",
    goldenVariant: "dzn-full",
    verdict: "PASS",
    exitStatus: 0,
    totalConfigs: 2,
    failures: 0,
    configs: [
      { label: "pt/directional-2", verdict: "PASS", rawStatus: "OK", tier: "full", minLuminance: 0.005 },
      { label: "wh/directional-sun", verdict: "PASS", rawStatus: "OK", tier: null, minLuminance: 0.005 },
    ],
  },
  {
    path: "tools/behavioral-gate/behavioral-gate-dzn-hdri-status.json",
    command: "npm run behavioral-gate:dzn -- --filter hdri --require-full-tier",
    filter: "hdri",
    goldenVariant: "dzn-full",
    verdict: "PASS",
    exitStatus: 0,
    totalConfigs: 3,
    failures: 0,
    configs: [
      { label: "pt/hdri-env", verdict: "PASS", rawStatus: "OK", tier: "full", minLuminance: 0.005 },
      { label: "pt/lite+hdri", verdict: "PASS", rawStatus: "OK", tier: "lite", minLuminance: 0.005 },
      { label: "wh/hdri-env", verdict: "PASS", rawStatus: "OK", tier: null, minLuminance: 0.005 },
    ],
  },
  {
    path: "tools/behavioral-gate/behavioral-gate-dzn-procedural-sky-status.json",
    command: "npm run behavioral-gate:dzn -- --filter procedural-sky --require-full-tier",
    filter: "procedural-sky",
    goldenVariant: "dzn-full",
    verdict: "PASS",
    exitStatus: 0,
    totalConfigs: 1,
    failures: 0,
    configs: [
      { label: "pt/procedural-sky", verdict: "PASS", rawStatus: "OK", tier: "full", minLuminance: 0.005 },
    ],
  },
  {
    path: "tools/behavioral-gate/behavioral-gate-dzn-material-lobes-status.json",
    command: "npm run behavioral-gate:dzn -- --filter material-lobes --require-full-tier",
    filter: "material-lobes",
    goldenVariant: "dzn-full",
    verdict: "PASS",
    exitStatus: 0,
    totalConfigs: 1,
    failures: 0,
    configs: [
      {
        label: "pt/material-lobes",
        verdict: "PASS",
        rawStatus: "OK",
        tier: "full",
        goldenStatus: "ok",
        goldenVariant: "dzn-full",
        maxRmse: 8,
        maxMeanAbs: 4,
        maxAbs: 48,
      },
    ],
  },
  {
    path: "tools/behavioral-gate/behavioral-gate-dzn-material-lobe-maps-status.json",
    command: "npm run behavioral-gate:dzn -- --filter material-lobe-maps --require-full-tier",
    filter: "material-lobe-maps",
    goldenVariant: "dzn-full",
    verdict: "PASS",
    exitStatus: 0,
    totalConfigs: 1,
    failures: 0,
    configs: [
      {
        label: "pt/material-lobe-maps",
        verdict: "PASS",
        rawStatus: "OK",
        tier: "full",
        goldenStatus: "ok",
        goldenVariant: "dzn-full",
        maxRmse: 8,
        maxMeanAbs: 4,
        maxAbs: 48,
      },
    ],
  },
  ...walkaroundShardStatuses(),
  {
    path: "tools/behavioral-gate/behavioral-gate-dzn-gltf-status.json",
    command: "npm run behavioral-gate:dzn -- --filter gltf --require-full-tier",
    filter: "gltf",
    goldenVariant: "dzn-full",
    verdict: "PASS",
    exitStatus: 0,
    totalConfigs: 11,
    failures: 0,
    configs: [
      { label: "pt/gltf-unlit", verdict: "PASS", rawStatus: "OK", tier: "full" },
      { label: "pt/gltf-textured-pbr", verdict: "PASS", rawStatus: "OK", tier: "full" },
      { label: "pt/gltf-transmission", verdict: "PASS", rawStatus: "OK", tier: "full" },
      { label: "pt/gltf-skinned-animation", verdict: "PASS", rawStatus: "OK", tier: "full" },
      { label: "pt/gltf-draco-mock", verdict: "PASS", rawStatus: "OK", tier: "full" },
      { label: "pt/gltf-point-line-fallback", verdict: "PASS", rawStatus: "OK", tier: "full", goldenStatus: "ok", maxRmse: 8, maxMeanAbs: 4, maxAbs: 48 },
      { label: "pt/gltf-triangle-strip-fan", verdict: "PASS", rawStatus: "OK", tier: "full", goldenStatus: "ok", maxRmse: 8, maxMeanAbs: 4, maxAbs: 48 },
      { label: "pt/gltf-material-sweep", verdict: "PASS", rawStatus: "OK", tier: "full", goldenStatus: "ok", maxRmse: 8, maxMeanAbs: 4, maxAbs: 48 },
      { label: "pt/gltf-real-box-textured", verdict: "PASS", rawStatus: "OK", tier: "full", goldenStatus: "ok", goldenVariant: "dzn-full", maxRmse: 8, maxMeanAbs: 4, maxAbs: 48 },
      { label: "pt/gltf-real-draco", verdict: "PASS", rawStatus: "OK", tier: "full", goldenStatus: "ok", goldenVariant: "dzn-full", maxRmse: 8, maxMeanAbs: 4, maxAbs: 48 },
      { label: "pt/gltf-real-meshopt", verdict: "PASS", rawStatus: "OK", tier: "full", goldenStatus: "ok", goldenVariant: "dzn-full", maxRmse: 8, maxMeanAbs: 4, maxAbs: 48 },
    ],
  },
]);

/** @returns {ExpectedStatus[]} */
function walkaroundShardStatuses() {
  const labels = [
    "wh/default",
    "wh/rcEnabled",
    "wh/ppgEnabled",
    "wh/gtao-off",
    "wh/checkerboard",
    "wh/skinned-mesh",
    "wh/hdri-env",
    "wh/rect-area-emitter",
    "wh/directional-sun",
    "wh/glass-gi",
    "wh/transparent-oit",
  ];
  return labels.map((label) => ({
    path: `tools/behavioral-gate/behavioral-gate-dzn-${slug(label)}-status.json`,
    command: `npm run behavioral-gate:dzn -- --filter ${label} --require-full-tier`,
    filter: label,
    goldenVariant: "dzn-full",
    verdict: "PASS",
    exitStatus: 0,
    totalConfigs: 1,
    failures: 0,
    configs: [
      walkaroundExpectedConfig(label),
    ],
  }));
}

/**
 * @param {string} label
 * @returns {ExpectedConfig}
 */
function walkaroundExpectedConfig(label) {
  return {
    label,
    verdict: "PASS",
    rawStatus: "OK",
    tier: null,
    minLuminance: 0.005,
    goldenStatus: "ok",
    goldenVariant: "dzn-full",
    maxRmse: 8,
    maxMeanAbs: 4,
    maxAbs: 48,
  };
}

/** @param {string} value */
function slug(value) {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "") || "filtered";
}

/**
 * @param {string} message
 * @returns {never}
 */
function fail(message) {
  throw new Error(`[behavioral-gate-dzn-status-check] ${message}`);
}

/**
 * @param {unknown} a
 * @param {unknown} b
 */
function sameJson(a, b) {
  return JSON.stringify(a) === JSON.stringify(b);
}

/** @type {Set<string>} */
const coveredLabels = new Set();

for (const expected of EXPECTED) {
  const url = new URL(`../../${expected.path}`, import.meta.url);
  const status = JSON.parse(await Deno.readTextFile(url));

  if (status.harness !== "behavioral-gate:dzn") fail(`${expected.path}: harness mismatch`);
  if (status.verdict !== expected.verdict) fail(`${expected.path}: verdict must be ${expected.verdict}, got ${status.verdict}`);
  if (status.command !== expected.command) fail(`${expected.path}: command mismatch`);
  if (status.filter !== expected.filter) fail(`${expected.path}: filter mismatch`);
  if (expected.goldenVariant !== undefined && (status.goldenVariant ?? null) !== expected.goldenVariant) {
    fail(`${expected.path}: goldenVariant mismatch`);
  }
  if (status.exitStatus !== expected.exitStatus) fail(`${expected.path}: exitStatus must be ${expected.exitStatus}`);
  if (status.signal != null) fail(`${expected.path}: signal must be null`);
  if (status.summary?.totalConfigs !== expected.totalConfigs) fail(`${expected.path}: totalConfigs mismatch`);
  if (status.summary?.failures !== expected.failures) fail(`${expected.path}: failures mismatch`);
  if (status.summary?.knownResiduals !== 0) fail(`${expected.path}: knownResiduals must be 0`);

  if (!Array.isArray(status.configs)) fail(`${expected.path}: configs must be an array`);
  const statusConfigs = /** @type {JsonRecord[]} */ (status.configs);
  if (statusConfigs.length !== expected.configs.length) {
    fail(`${expected.path}: configs length mismatch`);
  }
  const actualLabels = statusConfigs.map((config) => String(config.label)).sort();
  if (new Set(actualLabels).size !== actualLabels.length) {
    fail(`${expected.path}: duplicate config labels are not allowed`);
  }
  const expectedLabels = expected.configs.map((config) => config.label).sort();
  if (!sameJson(actualLabels, expectedLabels)) {
    fail(`${expected.path}: exact config labels mismatch`);
  }

  /** @type {Map<string, JsonRecord>} */
  const byLabel = new Map();
  for (const config of statusConfigs) {
    coveredLabels.add(String(config.label));
    byLabel.set(String(config.label), config);
  }
  for (const expectedConfig of expected.configs) {
    const config = byLabel.get(expectedConfig.label);
    if (config == null) fail(`${expected.path}: missing ${expectedConfig.label}`);
    if (config.verdict !== expectedConfig.verdict) fail(`${expected.path}: ${expectedConfig.label} verdict mismatch`);
    if (config.rawStatus !== expectedConfig.rawStatus) fail(`${expected.path}: ${expectedConfig.label} rawStatus mismatch`);
    if (config.tier !== expectedConfig.tier) fail(`${expected.path}: ${expectedConfig.label} tier mismatch`);
    if (config.gpuErrors !== 0) fail(`${expected.path}: ${expectedConfig.label} gpuErrors must be 0`);
    if (config.nan !== false) fail(`${expected.path}: ${expectedConfig.label} nan must be false`);
    if (expectedConfig.minLuminance != null && !(config.luminance >= expectedConfig.minLuminance)) {
      fail(`${expected.path}: ${expectedConfig.label} luminance below bound`);
    }

    if (expectedConfig.goldenStatus != null) {
      if (config.goldenStatus !== expectedConfig.goldenStatus) {
        fail(`${expected.path}: ${expectedConfig.label} goldenStatus mismatch`);
      }
      if (expectedConfig.goldenVariant !== undefined && (config.goldenVariant ?? null) !== expectedConfig.goldenVariant) {
        fail(`${expected.path}: ${expectedConfig.label} goldenVariant mismatch`);
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

    if (expectedConfig.cwbvhParityKind != null) {
      if (config.cwbvhParityKind !== expectedConfig.cwbvhParityKind) {
        fail(`${expected.path}: ${expectedConfig.label} cwbvhParityKind mismatch`);
      }
      if (config.cwbvhParityRmse > expectedConfig.maxCwbvhParityRmse) {
        fail(`${expected.path}: ${expectedConfig.label} cwbvhParityRmse exceeds bound`);
      }
      if (config.cwbvhParityMeanAbs > expectedConfig.maxCwbvhParityMeanAbs) {
        fail(`${expected.path}: ${expectedConfig.label} cwbvhParityMeanAbs exceeds bound`);
      }
      if (config.cwbvhParityMaxAbs > expectedConfig.maxCwbvhParityMaxAbs) {
        fail(`${expected.path}: ${expectedConfig.label} cwbvhParityMaxAbs exceeds bound`);
      }
      if (!sameJson(config.cwbvhParityThresholds, {
        maxRmse: expectedConfig.maxCwbvhParityRmse,
        maxMeanAbs: expectedConfig.maxCwbvhParityMeanAbs,
        maxAbs: expectedConfig.maxCwbvhParityMaxAbs,
      })) {
        fail(`${expected.path}: ${expectedConfig.label} cwbvhParityThresholds mismatch`);
      }
      if (config.cwbvhPerfKind !== "same-scene") {
        fail(`${expected.path}: ${expectedConfig.label} cwbvhPerfKind mismatch`);
      }
      for (const [key, value] of Object.entries({
        cwbvhBinaryRenderMs: config.cwbvhBinaryRenderMs,
        cwbvhRenderMs: config.cwbvhRenderMs,
        cwbvhRenderMsRatio: config.cwbvhRenderMsRatio,
        cwbvhBinaryMemoryBytes: config.cwbvhBinaryMemoryBytes,
        cwbvhMemoryBytes: config.cwbvhMemoryBytes,
        cwbvhBinarySceneBytes: config.cwbvhBinarySceneBytes,
        cwbvhSceneBytes: config.cwbvhSceneBytes,
      })) {
        if (!(Number.isFinite(value) && value > 0)) {
          fail(`${expected.path}: ${expectedConfig.label} ${key} must be positive`);
        }
      }
      for (const [key, value] of Object.entries({
        cwbvhMemoryBytesDelta: config.cwbvhMemoryBytesDelta,
        cwbvhSceneBytesDelta: config.cwbvhSceneBytesDelta,
      })) {
        if (!Number.isFinite(value)) {
          fail(`${expected.path}: ${expectedConfig.label} ${key} must be finite`);
        }
      }
      if (
        expectedConfig.minCwbvhRenderMsRatio != null &&
        !(config.cwbvhRenderMsRatio >= expectedConfig.minCwbvhRenderMsRatio)
      ) {
        fail(`${expected.path}: ${expectedConfig.label} cwbvhRenderMsRatio should preserve the no-default-promotion finding`);
      }
    }
  }
}

const gateSource = await Deno.readTextFile(new URL("./gate.mjs", import.meta.url));
/** @type {Set<string>} */
const labelsCoveredByFocusedProofs = new Set([]);
const gateLabels = [...gateSource.matchAll(/label:\s*"([^"]+)"/g)]
  .map((match) => match[1])
  .filter((label) => label.includes("/") && !label.startsWith("__self-test/"));
const missingLabels = [...new Set(gateLabels)]
  .filter((label) => !coveredLabels.has(label) && !labelsCoveredByFocusedProofs.has(label))
  .sort();
if (missingLabels.length > 0) {
  fail(`missing committed dzn status coverage for ${missingLabels.join(", ")}`);
}

console.log(
  `[behavioral-gate-dzn-status-check] PASS (${EXPECTED.length} committed dzn status artifacts; ` +
  "all regular and focused dzn-covered gate labels verified)",
);
