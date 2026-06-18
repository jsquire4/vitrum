#!/usr/bin/env -S deno run --sloppy-imports --allow-read
// @ts-check
// Verifies learned-system production posture without pretending quality A/B exists.

import {
  WALKAROUND_DENOISER_UNET_SPEC,
  deriveParamCount,
} from "../../packages/walkaround-hybrid/src/neural/unetArchitecture.ts";
import {
  loadWeightsFromArrayBuffer,
  validateWeightsForSpec,
} from "../../packages/walkaround-hybrid/src/neural/weights.ts";

/** @typedef {import("../../packages/walkaround-hybrid/src/neural/weights.ts").ModelWeights} ModelWeights */

const EXPECTED_PARAM_COUNT = 535107;
const TRACKED_RESEARCH_CHECKPOINTS = [
  "starter-v1.vitrum-model",
  "v2-random.vitrum-model",
];

/** @param {string} path */
function repoUrl(path) {
  return new URL(`../../${path}`, import.meta.url);
}

/** @param {string} message */
function fail(message) {
  throw new Error(`[learned-systems-proof-check] ${message}`);
}

/** @param {string} path */
async function readText(path) {
  return await Deno.readTextFile(repoUrl(path));
}

/** @param {string} path */
async function statFile(path) {
  const stat = await Deno.stat(repoUrl(path));
  if (!stat.isFile) fail(`${path} is missing or is not a file`);
  return stat;
}

/** @param {ModelWeights} weights */
function checkpointParamCount(weights) {
  return weights.layers.reduce(
    (acc, layer) => acc + layer.weights.length + layer.biases.length,
    0,
  );
}

/** @param {string} path */
async function loadCheckpoint(path) {
  const bytes = await Deno.readFile(repoUrl(path));
  const owned = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  return loadWeightsFromArrayBuffer(owned);
}

async function assertTrackedResearchCheckpoints() {
  if (deriveParamCount(WALKAROUND_DENOISER_UNET_SPEC.layers) !== EXPECTED_PARAM_COUNT) {
    fail(`canonical U-Net param count changed from ${EXPECTED_PARAM_COUNT}`);
  }

  for (const name of TRACKED_RESEARCH_CHECKPOINTS) {
    const path = `tools/neural-denoiser-training/checkpoints/${name}`;
    const stat = await statFile(path);
    if (stat.size <= 12) fail(`${path} is too small to contain a vitrum-model checkpoint`);

    const checkpoint = await loadCheckpoint(path);
    validateWeightsForSpec(WALKAROUND_DENOISER_UNET_SPEC, checkpoint);
    const paramCount = checkpointParamCount(checkpoint);
    if (paramCount !== EXPECTED_PARAM_COUNT) {
      fail(`${path} has ${paramCount} params, expected ${EXPECTED_PARAM_COUNT}`);
    }
  }
}

async function assertNoSilentProductionCheckpoint() {
  const dirUrl = repoUrl("tools/neural-denoiser-training/checkpoints/");
  /** @type {string[]} */
  const names = [];
  for await (const entry of Deno.readDir(dirUrl)) {
    if (entry.isFile) names.push(entry.name);
  }
  names.sort();

  const missing = TRACKED_RESEARCH_CHECKPOINTS.filter((name) => !names.includes(name));
  if (missing.length > 0) fail(`tracked research checkpoints are missing: ${missing.join(", ")}`);

  const unexpectedVitrumModels = names.filter((name) =>
    name.endsWith(".vitrum-model") && !TRACKED_RESEARCH_CHECKPOINTS.includes(name)
  );
  if (unexpectedVitrumModels.length > 0) {
    fail(
      `unregistered .vitrum-model checkpoint(s): ${unexpectedVitrumModels.join(", ")}. ` +
      `Add explicit proof metadata before committing new weights.`,
    );
  }

  const productionLike = names.filter((name) => /(^|[-_.])(prod|production|release|ga|default|blessed)([-_.]|$)/i.test(name));
  if (productionLike.length === 0) return;

  let manifest;
  try {
    manifest = JSON.parse(await readText("tools/neural-denoiser-training/quality-ab-production.json"));
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    fail(
      `production-like checkpoint(s) ${productionLike.join(", ")} require ` +
      `tools/neural-denoiser-training/quality-ab-production.json (${message})`,
    );
  }
  if (manifest.verdict !== "PASS" || manifest.mode !== "production-neural-denoiser") {
    fail("production neural checkpoint manifest must have mode='production-neural-denoiser' and verdict='PASS'");
  }
}

async function assertRuntimeTruthfulnessGuards() {
  const config = await readText("packages/walkaround-hybrid/src/HybridEngineConfig.ts");
  const engine = await readText("packages/walkaround-hybrid/src/HybridEngine.ts");
  const options = await readText("packages/walkaround-hybrid/src/HybridEngineOptions.ts");
  const pipeline = await readText("packages/walkaround-hybrid/src/pipeline/WalkaroundGPUPipeline.ts");
  const rootReadme = await readText("README.md");
  const architecture = await readText("plan/library-architecture.md");
  const packageReadme = await readText("packages/walkaround-hybrid/README.md");
  const trainingReadme = await readText("tools/neural-denoiser-training/README.md");
  const weights = await readText("packages/walkaround-hybrid/src/neural/weights.ts");
  const hardwareNeeds = await readText("HARDWARE-VALIDATION-NEEDS.md");

  const requiredFragments = [
    [config, "opts.denoiser === 'neural' && !opts.neuralWeights", "neural weights construction guard"],
    [config, "opts.denoiser ?? preset.denoiser ?? 'atrous-variance'", "non-neural default denoiser"],
    [config, "opts.nrcEnabled === true ? 1 : 0", "NRC opt-in config bit"],
    [engine, "walkaround-hybrid.nrc-experimental-biased", "NRC experimental warning code"],
    [engine, "defaultEnabled: false", "NRC warning defaultEnabled=false"],
    [engine, "estimator: 'biased'", "NRC warning estimator=biased"],
    [engine, "walkaround-hybrid.neural-host-weights-required", "neural host-weights warning code"],
    [engine, "packageProvidesProductionWeights: false", "neural warning production-weight disclosure"],
    [engine, "walkaround-hybrid-gris-unbiased-reuse", "GRIS opt-in experimental feature"],
    [engine, "walkaround-hybrid-ppg-guided-gi", "PPG opt-in experimental feature"],
    [engine, "walkaround-hybrid-nrc-biased-cache", "NRC opt-in experimental feature"],
    [engine, "walkaround-hybrid-neural-denoiser-host-weights", "neural opt-in experimental feature"],
    [options, "NRC is a BIASED estimator", "NRC option bias disclosure"],
    [pipeline, "export function assertNrcDeviceCapable", "NRC device capability gate"],
    [pipeline, "maxBindGroups < NRC_REQUIRED_MAX_BIND_GROUPS", "NRC maxBindGroups guard"],
    [pipeline, "maxWgStorage < NRC_REQUIRED_WORKGROUP_STORAGE_BYTES", "NRC workgroup-storage guard"],
    [pipeline, "assertNrcDeviceCapable(d.limits)", "NRC gate called before pipeline init"],
    [rootReadme, "opt-in PPG/NRC/neural", "root README learned-system opt-in disclosure"],
    [architecture, "SHIPPED as opt-in host-weight path", "architecture neural opt-in disclosure"],
    [architecture, "no production checkpoint is bundled", "architecture production checkpoint disclosure"],
    [packageReadme, "does **not** ship production neural weights", "package README production-weight disclosure"],
    [trainingReadme, "It does NOT produce useful denoiser weights", "training README smoke caveat"],
    [weights, "Repo-only research checkpoints", "weights.ts research checkpoint disclosure"],
    [weights, "does not ship production neural weights", "weights.ts production-weight disclosure"],
    [hardwareNeeds, "default-tier quality/convergence A/B before any default-on decision", "NRC hardware-validation default-tier caveat"],
    [hardwareNeeds, "biased-cache QUALITY A/B", "NRC hardware-validation quality A/B tail"],
    [hardwareNeeds, "not mean-preserving", "NRC hardware-validation biased-cache caveat"],
  ];

  for (const [text, fragment, label] of requiredFragments) {
    if (!text.includes(fragment)) fail(`missing ${label}: ${fragment}`);
  }

  const forbiddenFragments = [
    "biased-cache quality/perf A/B PASSED in WAVE8",
    "Remaining: the product decision to flip the default on",
    "GPU-confirm pending per #1",
  ];
  for (const fragment of forbiddenFragments) {
    if (hardwareNeeds.includes(fragment)) fail(`stale NRC hardware-validation claim remains: ${fragment}`);
  }
}

await assertTrackedResearchCheckpoints();
await assertNoSilentProductionCheckpoint();
await assertRuntimeTruthfulnessGuards();

console.log(
  `[learned-systems-proof-check] PASS ` +
  `(${TRACKED_RESEARCH_CHECKPOINTS.length} research checkpoints validate; neural/NRC remain opt-in and non-default)`,
);
