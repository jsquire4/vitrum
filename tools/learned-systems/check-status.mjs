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
import {
  MIN_PRODUCTION_NEURAL_CLEAN_REFERENCE_SPP,
  MIN_PRODUCTION_NEURAL_SAMPLE_COUNT,
  validateProductionQualityManifest,
} from "./qualityManifestValidator.mjs";

/** @typedef {import("../../packages/walkaround-hybrid/src/neural/weights.ts").ModelWeights} ModelWeights */
/**
 * @typedef {{
 *   name: string,
 *   role: "research" | "production",
 *   productionDefaultEligible?: boolean,
 *   qualityPosture?: string,
 *   sizeBytes: number,
 *   sha256: string,
 *   paramCount?: number,
 * }} CheckpointManifestEntry
 */
/**
 * @typedef {{
 *   schema: string,
 *   productionCheckpoint: string | null,
 *   checkpoints: CheckpointManifestEntry[],
 * }} CheckpointManifest
 */

const EXPECTED_PARAM_COUNT = 535107;
const CHECKPOINT_MANIFEST_PATH = "tools/neural-denoiser-training/checkpoints/manifest.json";
const QUALITY_MANIFEST_PATH = "tools/neural-denoiser-training/quality-ab-production.json";
const STATUS_PATH = "tools/learned-systems/learned-systems-status.json";
const WRITE_STATUS = Deno.args.includes("--write-status");
const REQUIRED_RESEARCH_CHECKPOINTS = [
  {
    name: "starter-v1.vitrum-model",
    role: "research",
    productionDefaultEligible: false,
    paramCount: EXPECTED_PARAM_COUNT,
    sizeBytes: 2140724,
    sha256: "9fbf951ac6d0960436243f9326339108b96410afc4b5d9efcd39c8784161f13f",
  },
  {
    name: "v2-random.vitrum-model",
    role: "research",
    productionDefaultEligible: false,
    paramCount: EXPECTED_PARAM_COUNT,
    sizeBytes: 2140724,
    sha256: "6f59e32b8f84f05e90f4afdfa025a98ef97ae60f163a1a4a9f7703ac4fa3d9cb",
  },
];

function productionQualityRequirements() {
  return {
    manifestPath: QUALITY_MANIFEST_PATH,
    requiredVerdict: "PASS",
    requiredMode: "production-neural-denoiser",
    minSampleCount: MIN_PRODUCTION_NEURAL_SAMPLE_COUNT,
    noisySpp: 1,
    minCleanReferenceSpp: MIN_PRODUCTION_NEURAL_CLEAN_REFERENCE_SPP,
    requiresAlbedo: true,
    requiresNormals: true,
    requiresCaptureSource: true,
    requiresTonemap: true,
    requiresArtifacts: true,
    requiresArtifactFiles: true,
    requiresHardware: true,
    requiresGeneratedAt: true,
    requiresCheckpointIdentity: true,
    requiresComparison: true,
    requiresThresholds: true,
  };
}

function learnedPromotionQualityRequirements() {
  return {
    nrc: { manifestPath: "tools/learned-systems/nrc-quality-convergence.json", requiredVerdict: "PASS", requiredMode: "nrc-quality-convergence", requiresQualityComparison: true, requiresConvergenceComparison: true, requiresDefaultTierDecision: true, requiresHardware: true, requiresGeneratedAt: true, requiresArtifacts: true, requiresArtifactFiles: true, requiresBiasedEstimatorDisclosure: true },
    gris: { manifestPath: "tools/learned-systems/gris-unbiasedness-ab.json", requiredVerdict: "PASS", requiredMode: "gris-unbiasedness-ab", requiresUnbiasednessAB: true, requiresBiasedDefaultErrorQuantification: true, requiresReferenceEstimator: true, requiresHardware: true, requiresGeneratedAt: true, requiresArtifacts: true, requiresArtifactFiles: true },
    ppg: { manifestPath: "tools/learned-systems/ppg-favorable-scene-ab.json", requiredVerdict: "PASS", requiredMode: "ppg-favorable-scene-ab", requiresFavorableSceneAB: true, requiresConvergenceComparison: true, requiresInstabilityChecks: true, requiresHardware: true, requiresGeneratedAt: true, requiresArtifacts: true, requiresArtifactFiles: true },
  };
}

/** @param {string} path */
function repoUrl(path) {
  return new URL(`../../${path}`, import.meta.url);
}

/**
 * @param {string} message
 * @returns {never}
 */
function fail(message) {
  throw new Error(`[learned-systems-proof-check] ${message}`);
}

/** @param {string} path */
async function readText(path) {
  return await Deno.readTextFile(repoUrl(path));
}

/** @param {Uint8Array} bytes */
async function sha256Hex(bytes) {
  const owned = new ArrayBuffer(bytes.byteLength);
  new Uint8Array(owned).set(bytes);
  const digest = await crypto.subtle.digest("SHA-256", owned);
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
  const owned = new ArrayBuffer(bytes.byteLength);
  new Uint8Array(owned).set(bytes);
  return loadWeightsFromArrayBuffer(owned);
}

/** @returns {Promise<CheckpointManifest>} */
async function loadCheckpointManifest() {
  let manifest;
  try {
    manifest = JSON.parse(await readText(CHECKPOINT_MANIFEST_PATH));
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    fail(`checkpoint manifest is missing or invalid JSON: ${message}`);
  }
  if (manifest == null || typeof manifest !== "object") {
    fail("checkpoint manifest must be an object");
  }
  const candidate = /** @type {Record<string, any>} */ (manifest);
  if (candidate.schema !== "vitrum.neural-denoiser.checkpoints.v1") {
    fail(`checkpoint manifest schema is ${String(candidate.schema)}`);
  }
  if (!Array.isArray(candidate.checkpoints)) {
    fail("checkpoint manifest must contain a checkpoints array");
  }
  if (candidate.productionCheckpoint !== null && typeof candidate.productionCheckpoint !== "string") {
    fail("productionCheckpoint must be null or a checkpoint name string");
  }
  if (typeof candidate.productionCheckpoint === "string") {
    const productionEntry = candidate.checkpoints.find((entry) => entry?.name === candidate.productionCheckpoint);
    if (productionEntry == null) {
      fail(`productionCheckpoint ${candidate.productionCheckpoint} is not listed in checkpoints`);
    }
    if (productionEntry.role !== "production") {
      fail(`productionCheckpoint ${candidate.productionCheckpoint} must point at a role:"production" entry`);
    }
    if (productionEntry.productionDefaultEligible !== true) {
      fail(`productionCheckpoint ${candidate.productionCheckpoint} must be productionDefaultEligible:true`);
    }
  }
  return /** @type {CheckpointManifest} */ (candidate);
}

/** @param {CheckpointManifest} manifest */
async function assertTrackedResearchCheckpoints(manifest) {
  if (deriveParamCount(WALKAROUND_DENOISER_UNET_SPEC.layers) !== EXPECTED_PARAM_COUNT) {
    fail(`canonical U-Net param count changed from ${EXPECTED_PARAM_COUNT}`);
  }

  for (const required of REQUIRED_RESEARCH_CHECKPOINTS) {
    const entry = manifest.checkpoints.find((checkpoint) => checkpoint?.name === required.name);
    if (entry == null) fail(`missing required research checkpoint ${required.name}`);
    if (entry.role !== required.role) fail(`${required.name} role differs from required research checkpoint contract`);
    if (entry.productionDefaultEligible !== required.productionDefaultEligible) {
      fail(`${required.name} productionDefaultEligible differs from required research checkpoint contract`);
    }
    if (entry.paramCount !== required.paramCount) fail(`${required.name} paramCount differs from required research checkpoint contract`);
    if (entry.sizeBytes !== required.sizeBytes) fail(`${required.name} sizeBytes differs from required research checkpoint contract`);
    if (entry.sha256 !== required.sha256) fail(`${required.name} sha256 differs from required research checkpoint contract`);
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

/** @param {CheckpointManifest} manifest */
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
  if (productionLike.length > 0 && productionEntries.length === 0) {
    fail(
      `production-like checkpoint filename(s) ${productionLike.join(", ")} must be registered ` +
      `as role:"production" with productionDefaultEligible:true before quality evidence can count`,
    );
  }

  let qualityManifest;
  try {
    qualityManifest = JSON.parse(await readText(QUALITY_MANIFEST_PATH));
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    fail(
      `production-like checkpoint(s) ${productionLike.join(", ")} require ` +
      `${QUALITY_MANIFEST_PATH} (${message})`,
    );
  }
  if (qualityManifest.verdict !== "PASS" || qualityManifest.mode !== "production-neural-denoiser") {
    fail("production neural checkpoint manifest must have mode='production-neural-denoiser' and verdict='PASS'");
  }

  if (productionEntries.length > 0) {
    validateProductionQualityManifest({
      qualityManifest,
      productionEntries,
      productionCheckpoint: manifest.productionCheckpoint,
      productionLike,
      expectedParamCount: EXPECTED_PARAM_COUNT,
      artifactExists: artifactPathExists,
      artifactText: artifactPathText,
      fail,
    });
  }
}

/** @param {string} path */
function artifactPathExists(path) {
  try {
    const stat = Deno.statSync(repoUrl(path));
    return stat.isFile || stat.isDirectory;
  } catch {
    return false;
  }
}

/** @param {string} path */
function artifactPathText(path) {
  return Deno.readTextFileSync(repoUrl(path));
}

/**
 * @param {CheckpointManifest} checkpointManifest
 * @param {number} researchCount
 * @param {number} productionCount
 */
async function maybeWriteStatus(checkpointManifest, researchCount, productionCount) {
  if (!WRITE_STATUS) return;
  const productionCheckpoint = checkpointManifest.productionCheckpoint ?? null;
  const hasProductionCheckpoint = productionCheckpoint !== null && productionCount > 0;
  const learnedRequirements = learnedPromotionQualityRequirements();
  const status = {
    schema: "vitrum.learned-systems.status.v1",
    generatedAt: new Date().toISOString(),
    verdict: "PASS",
    productionPosture: hasProductionCheckpoint ? "quality-gated" : "provisioning-needed",
    neuralDenoiser: {
      productionCheckpoint,
      researchCheckpointCount: researchCount,
      productionCheckpointCount: productionCount,
      productionDefaultEligible: hasProductionCheckpoint,
      packageProvidesProductionWeights: hasProductionCheckpoint,
      qualityManifest: hasProductionCheckpoint
        ? QUALITY_MANIFEST_PATH
        : null,
      qualityManifestRequirements: productionQualityRequirements(),
      runtimePolicy: {
        autoSelectsNeuralOnlyWithProductionCheckpoint: true,
        explicitNeuralWithNonProductionWeights: "allowed-as-approximate",
        nonProductionCapabilitySupport: "approximate",
        requiredMetadataContract: "NeuralCheckpointMetadata",
      },
      remaining: hasProductionCheckpoint
        ? "Keep production default eligibility tied to the validated quality manifest."
        : "Provision a production neural checkpoint and passing quality A/B manifest before default or production claims.",
    },
    nrc: {
      defaultEnabled: false,
      estimator: "biased",
      productionDefaultEligible: false,
      qualityManifest: null,
      qualityManifestRequirements: learnedRequirements.nrc,
      remaining:
        "Run quality/convergence A/B and make a default-tier decision before promoting NRC beyond opt-in.",
    },
    gris: {
      defaultEnabled: false,
      optInFlag: "restirPtReuse",
      estimator: "unbiased-when-enabled",
      productionDefaultEligible: false,
      qualityManifest: null,
      qualityManifestRequirements: learnedRequirements.gris,
      remaining:
        "Run GRIS-on unbiasedness A/B and biased-default error quantification before any default-tier promotion.",
    },
    ppg: {
      defaultEnabled: false,
      productionDefaultEligible: false,
      qualityManifest: null,
      qualityManifestRequirements: learnedRequirements.ppg,
      remaining:
        "Run favorable-scene A/B and convergence/instability checks before production promotion.",
    },
    guardrails: [
      "Research checkpoints are loader/runtime validation assets only.",
      "Neural, NRC, PPG, and GRIS learned/reuse paths remain opt-in unless future quality evidence promotes them.",
      "Do not treat this PASS as production model quality.",
    ],
  };
  await Deno.writeTextFile(STATUS_PATH, `${JSON.stringify(status, null, 2)}\n`);
}

/**
 * @param {CheckpointManifest} checkpointManifest
 * @param {number} researchCount
 * @param {number} productionCount
 */
async function assertCommittedStatusArtifact(checkpointManifest, researchCount, productionCount) {
  let status;
  try {
    status = JSON.parse(await readText(STATUS_PATH));
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    fail(`${STATUS_PATH} is missing or invalid JSON: ${message}`);
  }
  if (status == null || typeof status !== "object") {
    fail(`${STATUS_PATH} must contain a status object`);
  }
  const record = /** @type {Record<string, any>} */ (status);
  const productionCheckpoint = checkpointManifest.productionCheckpoint ?? null;
  const hasProductionCheckpoint = productionCheckpoint !== null && productionCount > 0;
  const expectedPosture = hasProductionCheckpoint ? "quality-gated" : "provisioning-needed";
  if (record.schema !== "vitrum.learned-systems.status.v1") fail(`${STATUS_PATH} schema mismatch`);
  if (record.verdict !== "PASS") fail(`${STATUS_PATH} verdict must be PASS`);
  if (record.productionPosture !== expectedPosture) {
    fail(`${STATUS_PATH} productionPosture must be ${expectedPosture}`);
  }
  const neural = record.neuralDenoiser;
  if (neural == null || typeof neural !== "object") {
    fail(`${STATUS_PATH} neuralDenoiser must be an object`);
  }
  const neuralRecord = /** @type {Record<string, any>} */ (neural);
  if (neuralRecord.productionCheckpoint !== productionCheckpoint) {
    fail(`${STATUS_PATH} neuralDenoiser.productionCheckpoint must match checkpoint manifest`);
  }
  if (neuralRecord.researchCheckpointCount !== researchCount) {
    fail(`${STATUS_PATH} neuralDenoiser.researchCheckpointCount must match checkpoint manifest`);
  }
  if (neuralRecord.productionCheckpointCount !== productionCount) {
    fail(`${STATUS_PATH} neuralDenoiser.productionCheckpointCount must match checkpoint manifest`);
  }
  if (neuralRecord.productionDefaultEligible !== hasProductionCheckpoint) {
    fail(`${STATUS_PATH} neuralDenoiser.productionDefaultEligible must track production checkpoint availability`);
  }
  if (neuralRecord.packageProvidesProductionWeights !== hasProductionCheckpoint) {
    fail(`${STATUS_PATH} neuralDenoiser.packageProvidesProductionWeights must track bundled production checkpoint availability`);
  }
  const expectedQualityManifest = hasProductionCheckpoint ? QUALITY_MANIFEST_PATH : null;
  if (neuralRecord.qualityManifest !== expectedQualityManifest) {
    fail(`${STATUS_PATH} neuralDenoiser.qualityManifest must be ${String(expectedQualityManifest)}`);
  }
  const requirements = neuralRecord.qualityManifestRequirements;
  const expectedRequirements = productionQualityRequirements();
  if (JSON.stringify(requirements) !== JSON.stringify(expectedRequirements)) {
    fail(`${STATUS_PATH} neuralDenoiser.qualityManifestRequirements must match the production quality validator thresholds`);
  }
  const runtimePolicy = neuralRecord.runtimePolicy;
  if (runtimePolicy == null || typeof runtimePolicy !== "object") {
    fail(`${STATUS_PATH} neuralDenoiser.runtimePolicy must be an object`);
  }
  if (runtimePolicy.autoSelectsNeuralOnlyWithProductionCheckpoint !== true) {
    fail(`${STATUS_PATH} must pin neural auto-selection to production checkpoints only`);
  }
  if (runtimePolicy.explicitNeuralWithNonProductionWeights !== "allowed-as-approximate") {
    fail(`${STATUS_PATH} must pin non-production explicit neural as approximate`);
  }
  if (runtimePolicy.nonProductionCapabilitySupport !== "approximate") {
    fail(`${STATUS_PATH} must pin non-production neural supportDetails as approximate`);
  }
  if (runtimePolicy.requiredMetadataContract !== "NeuralCheckpointMetadata") {
    fail(STATUS_PATH + " must cite NeuralCheckpointMetadata as the production contract");
  }

  const learnedRequirements = learnedPromotionQualityRequirements();
  for (const [key, expectedRequirements] of Object.entries(learnedRequirements)) {
    const section = record[key];
    if (section == null || typeof section !== "object") {
      fail(STATUS_PATH + " " + key + " must be an object");
    }
    const sectionRecord = /** @type {Record<string, any>} */ (section);
    if (sectionRecord.defaultEnabled !== false) {
      fail(STATUS_PATH + " " + key + ".defaultEnabled must be false");
    }
    if (sectionRecord.productionDefaultEligible !== false) {
      fail(STATUS_PATH + " " + key + ".productionDefaultEligible must be false");
    }
    if (sectionRecord.qualityManifest !== null) {
      fail(STATUS_PATH + " " + key + ".qualityManifest must remain null until proof exists");
    }
    if (JSON.stringify(sectionRecord.qualityManifestRequirements) !== JSON.stringify(expectedRequirements)) {
      fail(STATUS_PATH + " " + key + ".qualityManifestRequirements must match the learned-system promotion evidence contract");
    }
    if (artifactPathExists(expectedRequirements.manifestPath)) {
      fail(
        STATUS_PATH + " " + key + ".qualityManifest is null while " +
          expectedRequirements.manifestPath +
          " exists; validate the manifest and update the status artifact before it can count",
      );
    }
  }
  if (record.nrc?.estimator !== "biased") {
    fail(STATUS_PATH + " nrc.estimator must be biased");
  }
  if (record.gris?.optInFlag !== "restirPtReuse") {
    fail(STATUS_PATH + " gris.optInFlag must be restirPtReuse");
  }
  if (record.gris?.estimator !== "unbiased-when-enabled") {
    fail(STATUS_PATH + " gris.estimator must be unbiased-when-enabled");
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
    [config, "assessNeuralCheckpointProductionReadiness", "neural checkpoint production-readiness resolver"],
    [config, "reason = 'host-neural-weights'", "auto denoiser host neural route"],
    [config, "reason = 'host-neural-weights-not-production-ready'", "auto denoiser research-checkpoint fallback disclosure"],
    [config, "reason = 'host-oidn-model-url'", "auto denoiser host OIDN route"],
    [config, "let reason: DenoiserAutoResolutionReason = 'no-host-model-assets'", "auto denoiser fallback disclosure"],
    [config, "validateWeightsForSpec(WALKAROUND_DENOISER_UNET_SPEC", "neural checkpoint shape validation before construction"],
    [config, "opts.nrcEnabled === true ? 1 : 0", "NRC opt-in config bit"],
    [config, "restirPtReuse: opts.restirPtReuse === true ? 1 : 0", "GRIS opt-in config bit"],
    [engine, "walkaround-hybrid.nrc-experimental-biased", "NRC experimental warning code"],
    [engine, "defaultEnabled: false", "NRC warning defaultEnabled=false"],
    [engine, "estimator: 'biased'", "NRC warning estimator=biased"],
    [engine, "walkaround-hybrid.neural-host-weights-required", "neural host-weights warning code"],
    [engine, "walkaround-hybrid.denoiser-auto-resolved", "denoiser auto resolution warning code"],
    [engine, "packageProvidesProductionWeights: false", "neural warning production-weight disclosure"],
    [engine, "neuralCheckpointAssessment.productionReady", "neural production-readiness capability gate"],
    [engine, "neural: fullTierWeights ? 'approximate' : 'unsupported'", "non-production neural supportDetails downgrade"],
    [engine, "walkaround-hybrid-gris-unbiased-reuse", "GRIS opt-in experimental feature"],
    [engine, "walkaround-hybrid-ppg-guided-gi", "PPG opt-in experimental feature"],
    [engine, "walkaround-hybrid-nrc-biased-cache", "NRC opt-in experimental feature"],
    [engine, "walkaround-hybrid-neural-denoiser-host-weights", "neural opt-in experimental feature"],
    [options, "NRC is a BIASED estimator", "NRC option bias disclosure"],
    [options, "readonly restirPtReuse?: boolean", "GRIS restirPtReuse public option"],
    [options, "V19 GPU unbiasedness validation", "GRIS validation caveat"],
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
    [trainingReadme, "Metric-only manifests are rejected", "training README production artifact disclosure"],
    [weights, "Repo-only research checkpoints", "weights.ts research checkpoint disclosure"],
    [weights, "does not ship production neural weights", "weights.ts production-weight disclosure"],
    [weights, "export interface NeuralCheckpointMetadata", "weights.ts production metadata contract"],
    [weights, "NEURAL_PRODUCTION_CHECKPOINT_REQUIREMENTS", "weights.ts production readiness thresholds"],
    [weights, "assessNeuralCheckpointProductionReadiness", "weights.ts production readiness assessment"],
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

async function assertTrainingPipelineEvidence() {
  const trainScript = await readText("tools/neural-denoiser-training/train.py");
  const exportScript = await readText("tools/neural-denoiser-training/export_weights.py");
  const captureDataset = await readText("tools/neural-denoiser-training/capture-dataset.mjs");
  const datasetSpec = await readText("tools/neural-denoiser-training/dataset_spec.md");
  const qualityValidator = await readText("tools/learned-systems/qualityManifestValidator.mjs");
  const roundTripTest = await readText("packages/walkaround-hybrid/__tests__/neuralWeightsRoundTrip.test.ts");

  const requiredFragments = [
    [trainScript, "NOTE: torch is imported lazily inside train()", "train.py lazy torch import guard"],
    [trainScript, "Do NOT add a top-level `import torch`", "train.py dry-run import boundary"],
    [trainScript, "class UNetDenoiser", "train.py canonical U-Net class"],
    [trainScript, "CANONICAL_PARAM_COUNT = 535107", "train.py canonical param count"],
    [trainScript, "def write_vitrum_binary", "train.py vitrum binary writer"],
    [trainScript, "def export_vitrum_weights", "train.py export path"],
    [trainScript, "def dry_run", "train.py numpy-only dry-run"],
    [trainScript, "def combined_loss", "train.py training loss"],
    [exportScript, "VITRUM_MODEL_MAGIC", "export_weights.py model magic"],
    [exportScript, "VITRUM_MODEL_VERSION = 1", "export_weights.py model version"],
    [exportScript, "LAYER_NAMES = [", "export_weights.py canonical layer list"],
    [exportScript, "torch.load(pth_path, map_location='cpu', weights_only=True)", "export_weights.py safe checkpoint load"],
    [exportScript, "state_dict", "export_weights.py state_dict handling"],
    [exportScript, "Total parameters", "export_weights.py param-count reporting"],
    [captureDataset, "It is NOT a real training", "capture-dataset smoke caveat"],
    [captureDataset, "Output layout matches dataset_spec.md exactly", "capture-dataset output-layout contract"],
    [captureDataset, "frame_NNNN_albedo.png", "capture-dataset albedo output"],
    [captureDataset, "frame_NNNN_normal.png", "capture-dataset normal output"],
    [captureDataset, "--scene random", "capture-dataset diversity mode"],
    [datasetSpec, "frame_0001.png           # 1 spp noisy color", "dataset spec noisy image layout"],
    [datasetSpec, "frame_0001_albedo.png", "dataset spec albedo layout"],
    [datasetSpec, "frame_0001_normal.png", "dataset spec normal layout"],
    [datasetSpec, "frame_0001.png           # 4096 spp reference", "dataset spec clean image layout"],
    [datasetSpec, "vitrum.neural-denoiser.dataset.v1", "dataset spec production manifest schema"],
    [datasetSpec, "scenes[].sampleCount", "dataset spec production manifest sample sum"],
    [datasetSpec, "This repo does NOT currently ship a batched G-buffer capture script", "dataset spec production capture gap"],
    [datasetSpec, "5000 pairs recommended for production quality", "dataset spec production dataset sizing"],
    [qualityValidator, "PRODUCTION_NEURAL_DATASET_MANIFEST_SCHEMA", "quality validator dataset manifest schema constant"],
    [qualityValidator, "validateProductionDatasetManifest", "quality validator dataset manifest content guard"],
    [qualityValidator, "artifactText(artifactRecord.datasetManifestPath)", "quality validator reads dataset manifest artifact JSON"],
    [roundTripTest, "neuralWeightsRoundTrip.test.ts — capture → train → export → load round-trip", "round-trip proof header"],
    [roundTripTest, "CANONICAL_PARAM_COUNT = 535107", "round-trip canonical param count"],
    [roundTripTest, "loadWeightsFromArrayBuffer", "round-trip runtime loader"],
    [roundTripTest, "validateWeightsForSpec", "round-trip runtime shape validator"],
    [roundTripTest, "InferenceGraph", "round-trip inference graph allocation"],
    [roundTripTest, "TRACKED_CHECKPOINTS", "round-trip tracked checkpoint coverage"],
  ];

  for (const [text, fragment, label] of requiredFragments) {
    if (!text.includes(fragment)) fail(`missing ${label}: ${fragment}`);
  }
}

async function assertBehavioralProofCoverage() {
  const proofFiles = [
    {
      path: "packages/walkaround-hybrid/src/__tests__/learnedSystemConfig.test.ts",
      needles: [
        "keeps denoiser:'auto' on the non-learned default when no host model assets exist",
        "resolves denoiser:'auto' to neural only when full-tier host production weights exist",
        "does not auto-select neural for shape-valid non-production weights",
        "requires production checkpoint thresholds before auto-selecting neural",
        "trainingSamples>=500",
        "cleanSpp>=4096",
        "qualityReport.reportPath",
        "does not auto-select neural on tier:'lite', even when host weights exist",
        "rejects denoiser:'auto' host weights that do not match the U-Net checkpoint contract",
        "rejects denoiser:'neural' host weights that do not match the U-Net checkpoint contract",
        "resolves denoiser:'auto' to OIDN when a host model URL is supplied",
        "clamps learned-system cadence and mixture knobs into the effective config",
        "packageProvidesProductionWeights: false",
        "defaultEnabled: false",
        "expect(cfg.ppgEnabled).toBe(1)",
        "expect(cfg.nrcEnabled).toBe(1)",
      ],
    },
    {
      path: "packages/walkaround-hybrid/src/__tests__/capabilitiesPartition.test.ts",
      needles: [
        "keeps learned/research paths out of experimentalFeatures until explicitly enabled",
        "resolves denoiser:'auto' to the default when no host model assets exist",
        "resolves denoiser:'auto' to neural only when full-tier production host weights are supplied",
        "keeps denoiser:'auto' off neural for shape-valid non-production weights",
        "expect(engine.capabilities.supportDetails?.denoisers.neural).toBe('approximate')",
        "resolves denoiser:'auto' away from neural on lite even if weights are present",
        "declares opt-in learned/research paths as experimental features",
        "walkaround-hybrid.nrc-experimental-biased",
        "walkaround-hybrid.neural-host-weights-required",
        "packageProvidesProductionWeights: false",
      ],
    },
    {
      path: "packages/walkaround-hybrid/src/__tests__/hybridLiteTier.test.ts",
      needles: [
        "throws on tier:lite + ppgEnabled",
        "throws on tier:lite + nrcEnabled",
        "reports neural support details from the runtime provisioning state",
        "default (no nrcEnabled) stores the gate as 0 (OFF)",
        "nrcEnabled:true on the full tier stores the gate as 1",
      ],
    },
    {
      path: "packages/walkaround-hybrid/src/pipeline/__tests__/nrcStructuralGate.test.ts",
      needles: [
        "keeps the default gi-ris pass on the 4-group non-NRC module",
        "adds the 5th NRC bind group and NRC shader symbols only when nrcConfig is provided",
        "expect(risGiGroupCount(stub.computePipelines)).toBe(4)",
        "expect(risGiGroupCount(stub.computePipelines)).toBe(5)",
      ],
    },
    {
      path: "packages/walkaround-hybrid/src/pipeline/__tests__/nrcDeviceCapability.test.ts",
      needles: [
        "exact required NRC limits pass",
        "default WebGPU limits fail with an actionable nrcEnabled error",
        "maxBindGroups",
        "maxStorageBuffersPerShaderStage",
        "maxComputeWorkgroupStorageSize",
        "NRC_REQUIRED_WORKGROUP_STORAGE_BYTES",
      ],
    },
    {
      path: "packages/walkaround-hybrid/src/neural/nrc/__tests__/nrcGateBitIdentity.test.ts",
      needles: [
        "omitting nrcEnabled is byte-identical to nrcEnabled: 0",
        "turning NRC on flips ONLY u32[91]",
        "const NRC_GATE_U32_INDEX = 91",
      ],
    },
    {
      path: "packages/walkaround-hybrid/src/__tests__/grisVariantPin.test.ts",
      needles: [
        "default (no restirPtReuse) stores 0 (OFF",
        "restirPtReuse: true stores 1 (ON",
        "restirPtReuse: true is compatible with the full tier",
        "restirPtReuse: true is compatible with the lite tier",
        "default OFF is stable across qualityTier values",
      ],
    },
    {
      path: "packages/walkaround-hybrid/src/pipeline/__tests__/ppgCoordinatorDiagnostics.test.ts",
      needles: [
        "routes maxSpatialCells import mismatch through structured warnings",
        "routes scene-bounds import mismatch through structured warnings",
        "reports training readback failures as deduped non-fatal EngineErrors",
      ],
    },
    {
      path: "packages/walkaround-hybrid/src/pipeline/__tests__/ppgCompilerGate.test.ts",
      needles: [
        "omits the PPG update pipeline by default",
        "compiles the PPG update pipeline only when ppgEnabled is true",
        "threads the GRIS reservoir stride into the PPG update shader when ReSTIR-PT reuse is enabled",
        "PPGUpdatePass gates training on ppgEnabled and ppgTrainThisFrame",
        "MAX_DTREE_NODES_PER_CELL",
        "RESERVOIR_GI_STRIDE_LOCAL",
      ],
    },
    {
      path: "scripts/__tests__/learned-systems-quality-manifest.test.mjs",
      needles: [
        "passes when identity, dataset, hardware, and thresholds are complete",
        "rejects mismatched checkpoint identity",
        "rejects missing hardware metadata",
        "rejects missing finite metrics",
        "rejects incomplete dataset metadata",
        "rejects failed higher-is-better thresholds",
        "rejects failed lower-is-better thresholds",
      ],
    },
  ];

  for (const proof of proofFiles) {
    const text = await readText(proof.path);
    if (/\b(?:describe|it|test)\.(?:skip|todo|only)\s*\(/.test(text)) {
      fail(`${proof.path} must not contain skipped, todo, or only-marked tests while used as learned-system proof`);
    }
    for (const needle of proof.needles) {
      if (!text.includes(needle)) {
        fail(`${proof.path} missing behavioral proof needle: ${needle}`);
      }
    }
  }
}

const checkpointManifest = await loadCheckpointManifest();
await assertTrackedResearchCheckpoints(checkpointManifest);
await assertNoSilentProductionCheckpoint(checkpointManifest);
await assertRuntimeTruthfulnessGuards();
await assertTrainingPipelineEvidence();
await assertBehavioralProofCoverage();

const researchCount = checkpointManifest.checkpoints.filter((entry) => entry.role === "research").length;
const productionCount = checkpointManifest.checkpoints.filter((entry) => entry.role === "production").length;
await maybeWriteStatus(checkpointManifest, researchCount, productionCount);
await assertCommittedStatusArtifact(checkpointManifest, researchCount, productionCount);
console.log(
  `[learned-systems-proof-check] PASS ` +
  `(${researchCount} research checkpoints, ${productionCount} production checkpoints validate; neural/GRIS/NRC/PPG remain opt-in and non-default; behavioral proof coverage pinned)`,
);
