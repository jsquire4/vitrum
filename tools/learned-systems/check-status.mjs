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

  if (productionEntries.length > 0) {
    assertProductionQualityManifest(
      qualityManifest,
      productionEntries,
      manifest.productionCheckpoint,
      productionLike,
    );
  }
}

/**
 * @param {Record<string, any>} qualityManifest
 * @param {CheckpointManifestEntry[]} productionEntries
 * @param {string | null} productionCheckpoint
 * @param {string[]} productionLike
 */
function assertProductionQualityManifest(
  qualityManifest,
  productionEntries,
  productionCheckpoint,
  productionLike,
) {
  if (productionCheckpoint == null) {
    fail(
      `production-like checkpoint(s) ${productionLike.join(", ")} require manifest.productionCheckpoint ` +
      "to identify the production default",
    );
  }
  const productionEntry = productionEntries.find((entry) => entry.name === productionCheckpoint);
  if (productionEntry == null) {
    fail(`quality manifest productionCheckpoint ${productionCheckpoint} is not a production checkpoint entry`);
  }

  const checkpointProof = qualityManifest.checkpoint;
  if (checkpointProof == null || typeof checkpointProof !== "object") {
    fail("production neural quality manifest must include a checkpoint identity object");
  }
  const proof = /** @type {Record<string, any>} */ (checkpointProof);
  const expectedIdentity = {
    name: productionEntry.name,
    sha256: productionEntry.sha256,
    sizeBytes: productionEntry.sizeBytes,
    paramCount: productionEntry.paramCount ?? EXPECTED_PARAM_COUNT,
  };
  for (const [key, expectedValue] of Object.entries(expectedIdentity)) {
    if (proof[key] !== expectedValue) {
      fail(`production neural quality manifest checkpoint.${key} must be ${String(expectedValue)}`);
    }
  }

  const metrics = qualityManifest.metrics;
  if (metrics == null || typeof metrics !== "object") {
    fail("production neural quality manifest must include bounded quality metrics");
  }
  const metricRecord = /** @type {Record<string, any>} */ (metrics);
  const hasBoundedMetric =
    finiteMetric(metricRecord.psnrDb) ||
    finiteMetric(metricRecord.ssim) ||
    finiteMetric(metricRecord.meanAbs) ||
    finiteMetric(metricRecord.rmse);
  if (!hasBoundedMetric) {
    fail("production neural quality manifest metrics must include at least one finite numeric quality bound");
  }
  if (typeof qualityManifest.hardware !== "string" || qualityManifest.hardware.length === 0) {
    fail("production neural quality manifest must name the validation hardware/backend");
  }
  if (typeof qualityManifest.generatedAt !== "string" || qualityManifest.generatedAt.length === 0) {
    fail("production neural quality manifest must include generatedAt");
  }
  const dataset = qualityManifest.dataset;
  if (dataset == null || typeof dataset !== "object") {
    fail("production neural quality manifest must identify the validation dataset");
  }
  const datasetRecord = /** @type {Record<string, any>} */ (dataset);
  if (typeof datasetRecord.id !== "string" || datasetRecord.id.length === 0) {
    fail("production neural quality manifest dataset.id must be a non-empty string");
  }
  if (!Number.isInteger(datasetRecord.sceneCount) || datasetRecord.sceneCount <= 0) {
    fail("production neural quality manifest dataset.sceneCount must be positive");
  }
  if (!Number.isInteger(datasetRecord.sampleCount) || datasetRecord.sampleCount <= 0) {
    fail("production neural quality manifest dataset.sampleCount must be positive");
  }
  const comparison = qualityManifest.comparison;
  if (comparison == null || typeof comparison !== "object") {
    fail("production neural quality manifest must include an A/B comparison descriptor");
  }
  const comparisonRecord = /** @type {Record<string, any>} */ (comparison);
  if (typeof comparisonRecord.baseline !== "string" || comparisonRecord.baseline.length === 0) {
    fail("production neural quality manifest comparison.baseline must be a non-empty string");
  }
  if (typeof comparisonRecord.candidate !== "string" || comparisonRecord.candidate.length === 0) {
    fail("production neural quality manifest comparison.candidate must be a non-empty string");
  }
  const thresholds = qualityManifest.thresholds;
  if (thresholds == null || typeof thresholds !== "object") {
    fail("production neural quality manifest must include metric thresholds");
  }
  const thresholdRecord = /** @type {Record<string, any>} */ (thresholds);
  const boundedMetricNames = ["psnrDb", "ssim", "meanAbs", "rmse"].filter((name) =>
    finiteMetric(metricRecord[name])
  );
  for (const name of boundedMetricNames) {
    if (!finiteMetric(thresholdRecord[name])) {
      fail(`production neural quality manifest threshold.${name} must be finite when metrics.${name} is reported`);
    }
  }
}

/** @param {unknown} value */
function finiteMetric(value) {
  return typeof value === "number" && Number.isFinite(value);
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

async function assertBehavioralProofCoverage() {
  const proofFiles = [
    {
      path: "packages/walkaround-hybrid/src/__tests__/learnedSystemConfig.test.ts",
      needles: [
        "keeps denoiser:'auto' on the non-learned default when no host model assets exist",
        "resolves denoiser:'auto' to neural only when full-tier host weights exist",
        "does not auto-select neural on tier:'lite', even when host weights exist",
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
        "resolves denoiser:'auto' to neural only when full-tier host weights are supplied",
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
await assertBehavioralProofCoverage();

const researchCount = checkpointManifest.checkpoints.filter((entry) => entry.role === "research").length;
const productionCount = checkpointManifest.checkpoints.filter((entry) => entry.role === "production").length;
console.log(
  `[learned-systems-proof-check] PASS ` +
  `(${researchCount} research checkpoints, ${productionCount} production checkpoints validate; neural/NRC remain opt-in and non-default; behavioral proof coverage pinned)`,
);
