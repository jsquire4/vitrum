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
const CHECKPOINT_MANIFEST_PATH = "tools/neural-denoiser-training/checkpoints/manifest.json";

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

/** @param {Uint8Array} bytes */
async function sha256Hex(bytes) {
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
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
async function readCheckpointBytes(path) {
  const bytes = await Deno.readFile(repoUrl(path));
  return bytes;
}

/** @param {Uint8Array} bytes */
function loadCheckpointFromBytes(bytes) {
  const owned = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  return loadWeightsFromArrayBuffer(owned);
}

async function loadCheckpointManifest() {
  let manifest;
  try {
    manifest = JSON.parse(await readText(CHECKPOINT_MANIFEST_PATH));
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    fail(`checkpoint manifest is missing or invalid JSON: ${message}`);
  }
  if (manifest.schema !== "vitrum.neural-denoiser.checkpoints.v1") {
    fail(`checkpoint manifest schema is ${String(manifest.schema)}`);
  }
  if (!Array.isArray(manifest.checkpoints)) {
    fail("checkpoint manifest must contain a checkpoints array");
  }
  if (manifest.productionCheckpoint !== null) {
    const names = new Set(manifest.checkpoints.map((entry) => entry?.name));
    if (!names.has(manifest.productionCheckpoint)) {
      fail(`productionCheckpoint ${manifest.productionCheckpoint} is not listed in checkpoints`);
    }
  }
  return manifest;
}

/** @param {Awaited<ReturnType<typeof loadCheckpointManifest>>} manifest */
async function assertTrackedResearchCheckpoints(manifest) {
  if (deriveParamCount(WALKAROUND_DENOISER_UNET_SPEC.layers) !== EXPECTED_PARAM_COUNT) {
    fail(`canonical U-Net param count changed from ${EXPECTED_PARAM_COUNT}`);
  }

  const seen = new Set();
  for (const entry of manifest.checkpoints) {
    if (entry == null || typeof entry !== "object") fail("checkpoint manifest entry must be an object");
    const name = entry.name;
    if (typeof name !== "string" || !name.endsWith(".vitrum-model")) {
      fail(`checkpoint manifest entry has invalid name: ${String(name)}`);
    }
    if (seen.has(name)) fail(`checkpoint manifest lists ${name} more than once`);
    seen.add(name);
    if (entry.role !== "research" && entry.role !== "production") {
      fail(`${name} has invalid role ${String(entry.role)}`);
    }
    if (entry.role === "research" && entry.productionDefaultEligible !== false) {
      fail(`${name} is a research checkpoint but productionDefaultEligible is not false`);
    }
    if (typeof entry.qualityPosture !== "string" || !/not production/i.test(entry.qualityPosture)) {
      if (entry.productionDefaultEligible !== true) {
        fail(`${name} must state a non-production quality posture or be productionDefaultEligible:true`);
      }
    }

    const path = `tools/neural-denoiser-training/checkpoints/${name}`;
    const stat = await statFile(path);
    if (stat.size <= 12) fail(`${path} is too small to contain a vitrum-model checkpoint`);
    if (stat.size !== entry.sizeBytes) fail(`${path} size ${stat.size} differs from manifest ${entry.sizeBytes}`);

    const bytes = await readCheckpointBytes(path);
    const hash = await sha256Hex(bytes);
    if (hash !== entry.sha256) fail(`${path} sha256 ${hash} differs from manifest ${entry.sha256}`);

    const checkpoint = loadCheckpointFromBytes(bytes);
    validateWeightsForSpec(WALKAROUND_DENOISER_UNET_SPEC, checkpoint);
    const paramCount = checkpointParamCount(checkpoint);
    const expectedParams = entry.paramCount ?? EXPECTED_PARAM_COUNT;
    if (paramCount !== expectedParams) {
      fail(`${path} has ${paramCount} params, expected ${expectedParams}`);
    }
  }
}

/** @param {Awaited<ReturnType<typeof loadCheckpointManifest>>} manifest */
async function assertNoSilentProductionCheckpoint(manifest) {
  const dirUrl = repoUrl("tools/neural-denoiser-training/checkpoints/");
  /** @type {string[]} */
  const names = [];
  for await (const entry of Deno.readDir(dirUrl)) {
    if (entry.isFile) names.push(entry.name);
  }
  names.sort();

  const manifestNames = manifest.checkpoints.map((entry) => entry.name).sort();
  const vitrumModels = names.filter((name) => name.endsWith(".vitrum-model")).sort();
  const missing = manifestNames.filter((name) => !vitrumModels.includes(name));
  if (missing.length > 0) fail(`manifest-listed checkpoints are missing: ${missing.join(", ")}`);
  const unexpectedVitrumModels = vitrumModels.filter((name) =>
    !manifestNames.includes(name)
  );
  if (unexpectedVitrumModels.length > 0) {
    fail(
      `unregistered .vitrum-model checkpoint(s): ${unexpectedVitrumModels.join(", ")}. ` +
      `Add explicit proof metadata before committing new weights.`,
    );
  }

  const productionEntries = manifest.checkpoints.filter((entry) =>
    entry.role === "production" ||
    entry.productionDefaultEligible === true ||
    entry.name === manifest.productionCheckpoint
  );
  const productionLike = names.filter((name) => /(^|[-_.])(prod|production|release|ga|default|blessed)([-_.]|$)/i.test(name));
  const needsQualityManifest = productionEntries.length > 0 || productionLike.length > 0;
  if (!needsQualityManifest) return;

  let qualityManifest;
  try {
    qualityManifest = JSON.parse(await readText("tools/neural-denoiser-training/quality-ab-production.json"));
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    fail(
      `production-like checkpoint(s) ${productionLike.join(", ")} require ` +
      `tools/neural-denoiser-training/quality-ab-production.json (${message})`,
    );
  }
  if (qualityManifest.verdict !== "PASS" || qualityManifest.mode !== "production-neural-denoiser") {
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
    [config, "function resolveHybridDenoiser", "denoiser auto/default resolver"],
    [config, "opts.denoiser !== 'auto'", "non-auto denoiser path"],
    [config, "reason = 'host-neural-weights'", "auto denoiser host neural route"],
    [config, "reason = 'host-oidn-model-url'", "auto denoiser host OIDN route"],
    [config, "let reason: DenoiserAutoResolutionReason = 'no-host-model-assets'", "auto denoiser fallback disclosure"],
    [config, "opts.nrcEnabled === true ? 1 : 0", "NRC opt-in config bit"],
    [engine, "walkaround-hybrid.nrc-experimental-biased", "NRC experimental warning code"],
    [engine, "defaultEnabled: false", "NRC warning defaultEnabled=false"],
    [engine, "estimator: 'biased'", "NRC warning estimator=biased"],
    [engine, "walkaround-hybrid.neural-host-weights-required", "neural host-weights warning code"],
    [engine, "walkaround-hybrid.denoiser-auto-resolved", "denoiser auto resolution warning code"],
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
    [packageReadme, "checkpoints/manifest.json", "package README checkpoint manifest disclosure"],
    [trainingReadme, "It does NOT produce useful denoiser weights", "training README smoke caveat"],
    [trainingReadme, "checkpoint classification", "training README checkpoint manifest disclosure"],
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

const checkpointManifest = await loadCheckpointManifest();
await assertTrackedResearchCheckpoints(checkpointManifest);
await assertNoSilentProductionCheckpoint(checkpointManifest);
await assertRuntimeTruthfulnessGuards();

const researchCount = checkpointManifest.checkpoints.filter((entry) => entry.role === "research").length;
const productionCount = checkpointManifest.checkpoints.filter((entry) => entry.role === "production").length;
console.log(
  `[learned-systems-proof-check] PASS ` +
  `(${researchCount} research checkpoints, ${productionCount} production checkpoints validate; neural/NRC remain opt-in and non-default)`,
);
