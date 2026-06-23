// @ts-check
// Shared production neural quality manifest guard for learned-system promotion.

export const MIN_PRODUCTION_NEURAL_SAMPLE_COUNT = 500;
export const MIN_PRODUCTION_NEURAL_CLEAN_REFERENCE_SPP = 4096;

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
 *   qualityManifest: Record<string, any>,
 *   productionEntries: CheckpointManifestEntry[],
 *   productionCheckpoint: string | null,
 *   productionLike?: string[],
 *   expectedParamCount: number,
 *   fail?: (message: string) => void,
 * }} ProductionQualityValidationInput
 */

/**
 * @param {ProductionQualityValidationInput} input
 */
export function validateProductionQualityManifest(input) {
  const fail = input.fail ?? defaultFail;
  const {
    qualityManifest,
    productionEntries,
    productionCheckpoint,
    productionLike = [],
    expectedParamCount,
  } = input;

  if (productionCheckpoint == null) {
    const subject =
      productionLike.length > 0
        ? `production-like checkpoint(s) ${productionLike.join(", ")}`
        : `production checkpoint entry ${productionEntries.map((entry) => entry.name).join(", ")}`;
    fail(
      `${subject} require manifest.productionCheckpoint ` +
        "to identify the production default",
    );
  }
  const productionEntry = productionEntries.find((entry) => entry.name === productionCheckpoint);
  if (productionEntry == null) {
    fail(`quality manifest productionCheckpoint ${productionCheckpoint} is not a production checkpoint entry`);
  }
  const checkedProductionEntry = /** @type {CheckpointManifestEntry} */ (productionEntry);

  const checkpointProof = qualityManifest.checkpoint;
  if (checkpointProof == null || typeof checkpointProof !== "object") {
    fail("production neural quality manifest must include a checkpoint identity object");
  }
  const proof = /** @type {Record<string, any>} */ (checkpointProof);
  const expectedIdentity = {
    name: checkedProductionEntry.name,
    sha256: checkedProductionEntry.sha256,
    sizeBytes: checkedProductionEntry.sizeBytes,
    paramCount: checkedProductionEntry.paramCount ?? expectedParamCount,
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
  const artifacts = qualityManifest.artifacts;
  if (artifacts == null || typeof artifacts !== "object") {
    fail("production neural quality manifest must include reproducibility artifact paths");
  }
  const artifactRecord = /** @type {Record<string, any>} */ (artifacts);
  for (const field of [
    "datasetManifestPath",
    "resultSummaryPath",
    "candidateOutputsPath",
    "referenceOutputsPath",
  ]) {
    if (typeof artifactRecord[field] !== "string" || artifactRecord[field].length === 0) {
      fail(`production neural quality manifest artifacts.${field} must be a non-empty string`);
    }
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
  if (
    !Number.isInteger(datasetRecord.sampleCount) ||
    datasetRecord.sampleCount < MIN_PRODUCTION_NEURAL_SAMPLE_COUNT
  ) {
    fail(
      "production neural quality manifest dataset.sampleCount must be " +
        `>= ${MIN_PRODUCTION_NEURAL_SAMPLE_COUNT}`,
    );
  }
  if (datasetRecord.noisySpp !== 1) {
    fail("production neural quality manifest dataset.noisySpp must be 1");
  }
  if (
    !Number.isInteger(datasetRecord.cleanReferenceSpp) ||
    datasetRecord.cleanReferenceSpp < MIN_PRODUCTION_NEURAL_CLEAN_REFERENCE_SPP
  ) {
    fail(
      "production neural quality manifest dataset.cleanReferenceSpp must be " +
        `>= ${MIN_PRODUCTION_NEURAL_CLEAN_REFERENCE_SPP}`,
    );
  }
  if (datasetRecord.includesAlbedo !== true) {
    fail("production neural quality manifest dataset.includesAlbedo must be true");
  }
  if (datasetRecord.includesNormals !== true) {
    fail("production neural quality manifest dataset.includesNormals must be true");
  }
  if (typeof datasetRecord.captureSource !== "string" || datasetRecord.captureSource.length === 0) {
    fail("production neural quality manifest dataset.captureSource must be a non-empty string");
  }
  if (typeof datasetRecord.tonemap !== "string" || datasetRecord.tonemap.length === 0) {
    fail("production neural quality manifest dataset.tonemap must be a non-empty string");
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
    if (!qualityMetricPassesThreshold(name, metricRecord[name], thresholdRecord[name], fail)) {
      fail(
        `production neural quality manifest metric ${name}=${metricRecord[name]} ` +
          `does not satisfy threshold ${thresholdRecord[name]}`,
      );
    }
  }
}

/** @param {unknown} value */
export function finiteMetric(value) {
  return typeof value === "number" && Number.isFinite(value);
}

/**
 * @param {string} name
 * @param {number} metric
 * @param {number} threshold
 * @param {(message: string) => void} fail
 */
export function qualityMetricPassesThreshold(name, metric, threshold, fail = defaultFail) {
  switch (name) {
    case "psnrDb":
    case "ssim":
      return metric >= threshold;
    case "meanAbs":
    case "rmse":
      return metric <= threshold;
    default:
      fail(`unknown production neural quality metric ${name}`);
  }
}

/**
 * @param {string} message
 * @returns {never}
 */
function defaultFail(message) {
  throw new Error(message);
}
