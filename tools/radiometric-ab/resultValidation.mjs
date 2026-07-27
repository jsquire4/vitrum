// @ts-check
// Pure, runtime-neutral validators shared by the Deno regression checker and
// the Node recapture wrapper. Every boolean and verdict is re-derived from
// finite measurements; fresh recaptures additionally require current-source
// provenance.

export const RADIOMETRIC_RESULT_SCHEMA = 'vitrum.radiometric-ab.result.v1';

export class RadiometricResultValidationError extends Error {
  /** @param {string} message */
  constructor(message) {
    super(`[radiometric-result-validation] ${message}`);
    this.name = 'RadiometricResultValidationError';
  }
}

/** @param {string} id @param {string} message @returns {never} */
function fail(id, message) {
  throw new RadiometricResultValidationError(`${id}: ${message}`);
}

/** @param {unknown} value @returns {value is Record<string, any>} */
function isRecord(value) {
  return value != null && typeof value === 'object' && !Array.isArray(value);
}

/** @param {unknown} a @param {unknown} b */
function sameJson(a, b) {
  return JSON.stringify(a) === JSON.stringify(b);
}

/** @param {string} id @param {unknown} value @param {string} label @param {number} [minimum] */
function finite(id, value, label, minimum = -Infinity) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < minimum) {
    fail(id, `${label} must be a finite number >= ${minimum}`);
  }
  return value;
}

/** @param {string} id @param {unknown} value @param {string} label */
function positive(id, value, label) {
  const number = finite(id, value, label, 0);
  if (number <= 0) fail(id, `${label} must be positive`);
  return number;
}

/** @param {string} id @param {unknown} value @param {string} label */
function assertDeepFinite(id, value, label) {
  if (typeof value === 'number') {
    finite(id, value, label);
    return;
  }
  if (Array.isArray(value)) {
    value.forEach((entry, index) => assertDeepFinite(id, entry, `${label}[${index}]`));
    return;
  }
  if (isRecord(value)) {
    for (const [key, entry] of Object.entries(value)) {
      assertDeepFinite(id, entry, `${label}.${key}`);
    }
  }
}

/** @param {string} id @param {number} actual @param {number} expected @param {string} label */
function near(id, actual, expected, label) {
  if (!Number.isFinite(actual) || !Number.isFinite(expected)) {
    fail(id, `${label} and its derived value must both be finite`);
  }
  const tolerance = Math.max(1e-12, Math.abs(expected) * 1e-9);
  if (Math.abs(actual - expected) > tolerance) {
    fail(id, `${label} ${actual} differs from derived value ${expected}`);
  }
}

/** @param {number} a @param {number} b */
function relativeError(a, b) {
  if (b === 0) return a === 0 ? 0 : Infinity;
  return Math.abs(a - b) / b;
}

/** @param {number} numerator @param {number} denominator */
function ratioOrInfinity(numerator, denominator) {
  if (Math.abs(denominator) <= 1e-9) {
    return Math.abs(numerator) <= 1e-9 ? 1 : Infinity;
  }
  return numerator / denominator;
}

/** @param {string} id @param {unknown} value @param {string} label */
function assertTimestamp(id, value, label) {
  if (typeof value !== 'string' || value.length === 0 || !Number.isFinite(Date.parse(value))) {
    fail(id, `${label} must be a valid ISO timestamp`);
  }
  return Date.parse(value);
}

/** @param {string} id @param {any} roi @param {any} resolution @param {string} label */
function assertRoi(id, roi, resolution, label) {
  if (!isRecord(roi)) fail(id, `${label} must be an object`);
  for (const key of ['x0', 'y0', 'x1', 'y1']) {
    const value = roi[key];
    if (!Number.isSafeInteger(value) || value < 0) {
      fail(id, `${label}.${key} must be a non-negative safe integer`);
    }
  }
  if (roi.x0 > roi.x1 || roi.y0 > roi.y1) fail(id, `${label} is inverted`);
  const width = resolution.W ?? resolution.width;
  const height = resolution.H ?? resolution.height;
  if (roi.x1 >= width || roi.y1 >= height) fail(id, `${label} exceeds the result resolution`);
}

/** @param {string} id @param {any} proof @param {any} manifest */
function validateSourceManifest(id, proof, manifest) {
  if (!isRecord(manifest) ||
      manifest.schema !== 'vitrum.radiometric-ab.source-manifest.v1') {
    fail(id, 'provenance.sourceManifest is missing or has the wrong schema');
  }
  if (!sameJson(manifest.roots, proof.sourceRoots)) {
    fail(id, 'provenance.sourceManifest.roots differ from proof');
  }
  if (!Array.isArray(manifest.files) || manifest.files.length === 0) {
    fail(id, 'provenance.sourceManifest.files must be non-empty');
  }
  const paths = manifest.files.map((entry) => entry?.path);
  if (!sameJson(paths, [...new Set(paths)].sort())) {
    fail(id, 'provenance.sourceManifest.files must be sorted and unique');
  }
  for (const entry of manifest.files) {
    if (!isRecord(entry) ||
        typeof entry.path !== 'string' ||
        !proof.sourceRoots.some((/** @type {string} */ root) => entry.path.startsWith(`${root}/`)) ||
        !/^[0-9a-f]{64}$/.test(entry.sha256)) {
      fail(id, 'provenance.sourceManifest contains an invalid path or SHA-256 row');
    }
  }
  if (!/^[0-9a-f]{64}$/.test(manifest.digestSha256)) {
    fail(id, 'provenance.sourceManifest.digestSha256 must be a SHA-256 digest');
  }
}

/**
 * @param {any} proof
 * @param {any} result
 * @param {{ expectedProvenance?: unknown, historicalBaseline?: boolean }} [options]
 */
function validateCommon(proof, result, options = {}) {
  const id = String(proof?.id ?? 'unknown');
  if (!isRecord(result)) fail(id, 'result must be an object');
  if (proof.schema !== RADIOMETRIC_RESULT_SCHEMA) fail(id, 'proof schema is not the current result schema');
  if (result.schema !== proof.schema) fail(id, `schema ${String(result.schema)} differs from proof`);
  if (result.ab !== proof.ab) fail(id, `ab ${String(result.ab)} differs from proof`);
  if (!sameJson(result.resolution, proof.resolution)) fail(id, 'resolution differs from proof');
  if (result.verdict !== 'PASS') fail(id, `verdict must be PASS, got ${String(result.verdict)}`);
  if (!isRecord(result.provenance)) fail(id, 'provenance must be an object');
  if (result.provenance.scriptPath !== proof.scriptPath || result.provenance.resultPath !== proof.resultPath) {
    fail(id, 'provenance script/result paths differ from proof');
  }
  if (options.historicalBaseline === true) {
    if (options.expectedProvenance !== undefined) {
      fail(id, 'historicalBaseline cannot be combined with expectedProvenance');
    }
    if (result.provenance.sourceManifest !== undefined) {
      validateSourceManifest(id, proof, result.provenance.sourceManifest);
    }
  } else {
    validateSourceManifest(id, proof, result.provenance.sourceManifest);
  }
  if (options.expectedProvenance !== undefined && !sameJson(result.provenance, options.expectedProvenance)) {
    fail(id, 'provenance differs from the current script/helper/source identity');
  }
  const capturedAtMs = assertTimestamp(id, result.date ?? result.generatedAt, result.date != null ? 'date' : 'generatedAt');
  assertRoi(id, result.roi, result.resolution, 'roi');
  assertDeepFinite(id, result, 'result');
  return { id, capturedAtMs };
}

/** @param {any} proof @param {any} result */
function validateSppm(proof, result) {
  const id = proof.id;
  if (result.reference?.strategy !== proof.reference.strategy || result.reference?.frames !== proof.reference.frames) {
    fail(id, 'reference strategy/frame count differs from proof');
  }
  const refLum = positive(id, result.reference.roiLum, 'reference.roiLum');
  positive(id, result.reference.globalLum, 'reference.globalLum');
  if (!Array.isArray(result.sppm) || result.sppm.length !== proof.checkpoints.length) {
    fail(id, 'sppm checkpoints must match the proof count');
  }
  const frames = result.sppm.map((/** @type {any} */ entry) => entry.frames);
  if (!sameJson(frames, proof.checkpoints)) fail(id, 'checkpoint frame sequence differs from proof');
  let previousRelError = null;
  let derivedConverging = true;
  for (const entry of result.sppm) {
    const lum = positive(id, entry.lum, `checkpoint ${entry.frames}.lum`);
    positive(id, entry.globalLum, `checkpoint ${entry.frames}.globalLum`);
    finite(id, entry.rmse, `checkpoint ${entry.frames}.rmse`, 0);
    const relErr = finite(id, entry.relErr, `checkpoint ${entry.frames}.relErr`, 0);
    near(id, relErr, relativeError(lum, refLum), `checkpoint ${entry.frames}.relErr`);
    if (previousRelError != null && relErr > previousRelError * proof.thresholds.monotonicRelErrSlack) {
      derivedConverging = false;
    }
    previousRelError = relErr;
  }
  const finalRelErr = result.sppm.at(-1).relErr;
  const derivedInBallpark = finalRelErr < proof.thresholds.finalRelErrMax;
  if (result.converging !== derivedConverging) fail(id, 'converging does not match checkpoint measurements');
  if (result.inBallpark !== derivedInBallpark) fail(id, 'inBallpark does not match the final threshold');
  if (!(derivedConverging && derivedInBallpark)) fail(id, 'PASS verdict is not supported by SPPM measurements');
  return { finalRelErr, converging: derivedConverging, inBallpark: derivedInBallpark };
}

/** @param {string} id @param {any} arm @param {string} label */
function validateMeanVarianceArm(id, arm, label) {
  return {
    globalLum: positive(id, arm?.globalLum, `${label}.globalLum`),
    roiLum: positive(id, arm?.roiLum, `${label}.roiLum`),
    variance: finite(id, arm?.variance, `${label}.variance`, 0),
  };
}

/** @param {string} id @param {any} identity */
function validatePtDeviceIdentity(id, identity) {
  if (!isRecord(identity) ||
      identity.schema !== 'vitrum.radiometric-ab.device-identity.v1') {
    fail(id, 'deviceIdentity is missing or has the wrong schema');
  }
  if (!isRecord(identity.adapter)) fail(id, 'deviceIdentity.adapter must be an object');
  const adapterIdentityValues = Object.values(identity.adapter).filter(
    (value) => typeof value === 'string' && value.trim().length > 0,
  );
  if (adapterIdentityValues.length === 0) {
    fail(id, 'deviceIdentity.adapter must record at least one non-empty adapter identity field');
  }
  for (const label of ['adapterFeatures', 'deviceFeatures']) {
    const values = identity[label];
    if (!Array.isArray(values) ||
        values.some((value) => typeof value !== 'string') ||
        !sameJson(values, [...new Set(values)].sort())) {
      fail(id, `deviceIdentity.${label} must be a sorted unique string array`);
    }
  }
  for (const limitsLabel of ['advertisedLimits', 'requestedLimits']) {
    if (!isRecord(identity[limitsLabel])) {
      fail(id, `deviceIdentity.${limitsLabel} must be an object`);
    }
    for (const [name, value] of Object.entries(identity[limitsLabel])) {
      positive(id, value, `deviceIdentity.${limitsLabel}.${name}`);
    }
  }
  if (identity.advertisedLimits.maxStorageBuffersPerShaderStage < 28 ||
      identity.advertisedLimits.maxStorageTexturesPerShaderStage < 5 ||
      identity.advertisedLimits.maxBindGroups < 5) {
    fail(id, 'deviceIdentity does not prove the full-tier ReSTIR-PT adapter limits');
  }
  if (!isRecord(identity.runtime) || identity.runtime.kind !== 'deno') {
    fail(id, 'deviceIdentity.runtime must identify the native Deno capture runtime');
  }
  if (!isRecord(identity.runtime.version) ||
      !isRecord(identity.runtime.build) ||
      typeof identity.runtime.version.deno !== 'string' ||
      typeof identity.runtime.build.target !== 'string') {
    fail(id, 'deviceIdentity.runtime is incomplete');
  }
  if (identity.vulkanIcdFilenames !== null &&
      (typeof identity.vulkanIcdFilenames !== 'string' ||
       identity.vulkanIcdFilenames.length === 0)) {
    fail(id, 'deviceIdentity.vulkanIcdFilenames must be null or a non-empty string');
  }
}

/** @param {any} proof */
function expectedRestirCaptureConfig(proof) {
  const schedule = proof.capture.seedSchedule;
  const multiplier = BigInt(schedule.multiplier);
  const increment = BigInt(schedule.increment);
  const runStride = BigInt(schedule.runStride);
  /** @param {bigint} value */
  const u32 = (value) => Number(BigInt.asUintN(32, value));
  const meanFrameSeeds = Array.from(
    { length: proof.meanFrames },
    (_, frame) => u32(BigInt(frame) * multiplier + increment),
  );
  const varianceRunFrameSeeds = Array.from(
    { length: proof.varianceRuns },
    (_, run) => Array.from(
      { length: proof.varianceFramesPerRun },
      (_, frame) => {
        const globalFrame = BigInt(run * proof.varianceFramesPerRun + frame);
        return u32(globalFrame * multiplier + increment + BigInt(run) * runStride);
      },
    ),
  );
  return {
    scene: proof.capture.scene,
    traceTier: proof.capture.traceTier,
    colorSpace: proof.capture.colorSpace,
    requireFullTier: proof.capture.requireFullTier,
    requireRadiometricSignal: proof.capture.requireRadiometricSignal,
    maxBounces: proof.capture.maxBounces,
    resolution: proof.resolution,
    roi: proof.roi,
    meanFrames: proof.meanFrames,
    varianceRuns: proof.varianceRuns,
    varianceFramesPerRun: proof.varianceFramesPerRun,
    arms: {
      base: { restirPtReuse: false },
      candidate: {
        restirPtReuse: true,
        effectiveMClamp: proof.capture.effectiveMClamp,
        effectivePackedWCap: Math.fround(proof.professionalDefaultWeightCeiling),
      },
    },
    seeds: {
      ...schedule,
      meanFrameSeeds,
      varianceRunFrameSeeds,
    },
  };
}

/** @param {any} proof @param {any} result */
function validateBdpt(proof, result) {
  const id = proof.id;
  for (const field of ['meanFrames', 'varianceRuns', 'varianceFramesPerRun']) {
    if (result[field] !== proof[field]) fail(id, `${field} differs from proof`);
  }
  const uni = validateMeanVarianceArm(id, result.uni, 'uni');
  const bdpt = validateMeanVarianceArm(id, result.bdpt, 'bdpt');
  const globalRelErr = finite(id, result.globalRelErr, 'globalRelErr', 0);
  const roiRelErr = finite(id, result.roiRelErr, 'roiRelErr', 0);
  const varRatio = finite(id, result.varRatio, 'varRatio', 0);
  near(id, globalRelErr, relativeError(bdpt.globalLum, uni.globalLum), 'globalRelErr');
  near(id, roiRelErr, relativeError(bdpt.roiLum, uni.roiLum), 'roiRelErr');
  near(id, varRatio, ratioOrInfinity(bdpt.variance, uni.variance), 'varRatio');
  const meanAgreement = globalRelErr < proof.thresholds.globalRelErrMax;
  const varianceImproved = varRatio <= proof.thresholds.varRatioMax;
  if (result.meanAgreement !== meanAgreement) fail(id, 'meanAgreement does not match globalRelErr');
  if (result.varianceImproved !== varianceImproved) fail(id, 'varianceImproved does not match varRatio');

  if (result.controls?.meanFrames !== proof.meanFrames) fail(id, 'controls.meanFrames differs from proof');
  if (!sameJson(result.controls?.supportedDepths, proof.controls.depths)) fail(id, 'supportedDepths differs from proof');
  const controls = result.controls?.byMaxLightBounces;
  if (!Array.isArray(controls) || controls.length !== proof.controls.depths.length) {
    fail(id, 'depth controls must contain exactly the proved depths');
  }
  if (!sameJson(controls.map((entry) => entry.maxLightBounces), proof.controls.depths)) {
    fail(id, 'depth controls are missing, duplicated, or out of order');
  }
  for (const entry of controls) {
    const globalLum = positive(id, entry.globalLum, `control ${entry.maxLightBounces}.globalLum`);
    const roiLum = positive(id, entry.roiLum, `control ${entry.maxLightBounces}.roiLum`);
    const controlGlobalRelErr = finite(id, entry.globalRelErr, `control ${entry.maxLightBounces}.globalRelErr`, 0);
    const controlRoiRelErr = finite(id, entry.roiRelErr, `control ${entry.maxLightBounces}.roiRelErr`, 0);
    near(id, controlGlobalRelErr, relativeError(globalLum, uni.globalLum), `control ${entry.maxLightBounces}.globalRelErr`);
    near(id, controlRoiRelErr, relativeError(roiLum, uni.roiLum), `control ${entry.maxLightBounces}.roiRelErr`);
  }
  const endpoint = controls[proof.controls.depths.indexOf(1)];
  if (endpoint == null) fail(id, 'proof must include the endpoint-only depth 1 control');
  const endpointOnlyCarriesSignal = endpoint.globalLum > 1e-5 && endpoint.roiLum > 1e-5;
  const endpointOnlyMeanAgreement =
    endpoint.globalRelErr < proof.controls.endpointGlobalRelErrMax &&
    endpoint.roiRelErr < proof.controls.endpointRoiRelErrMax;
  if (result.controls.endpointOnlyCarriesSignal !== endpointOnlyCarriesSignal) {
    fail(id, 'endpointOnlyCarriesSignal does not match the endpoint measurements');
  }
  if (result.controls.endpointOnlyMeanAgreement !== endpointOnlyMeanAgreement) {
    fail(id, 'endpointOnlyMeanAgreement does not match the endpoint thresholds');
  }
  const defaultControl = controls[proof.controls.depths.indexOf(2)];
  if (defaultControl == null) fail(id, 'proof must include the default depth 2 control');
  near(id, defaultControl.globalLum, bdpt.globalLum, 'default depth control globalLum');
  near(id, defaultControl.roiLum, bdpt.roiLum, 'default depth control roiLum');
  if (!(meanAgreement && varianceImproved && endpointOnlyCarriesSignal && endpointOnlyMeanAgreement)) {
    fail(id, 'PASS verdict is not supported by BDPT measurements and endpoint controls');
  }
  return {
    globalRelErr,
    roiRelErr,
    varRatio,
    meanAgreement,
    varianceImproved,
    endpointOnlyCarriesSignal,
    endpointOnlyMeanAgreement,
    defaultMatchesDepth2: true,
    supportedDepths: [...proof.controls.depths],
  };
}

/** @param {any} proof @param {any} result */
function validateRestirPt(proof, result) {
  const id = proof.id;
  for (const field of ['meanFrames', 'varianceRuns', 'varianceFramesPerRun']) {
    if (result[field] !== proof[field]) fail(id, `${field} differs from proof`);
  }
  const base = validateMeanVarianceArm(id, result.base, 'base');
  const rpt = validateMeanVarianceArm(id, result.rpt, 'rpt');
  const globalRelErr = finite(id, result.globalRelErr, 'globalRelErr', 0);
  const roiRelErr = finite(id, result.roiRelErr, 'roiRelErr', 0);
  const varRatio = finite(id, result.varRatio, 'varRatio', 0);
  near(id, globalRelErr, relativeError(rpt.globalLum, base.globalLum), 'globalRelErr');
  near(id, roiRelErr, relativeError(rpt.roiLum, base.roiLum), 'roiRelErr');
  near(id, varRatio, ratioOrInfinity(rpt.variance, base.variance), 'varRatio');
  const highFrameMeanAgreement = globalRelErr < proof.thresholds.globalRelErrMax;
  const varianceNotWorse = varRatio <= proof.thresholds.varRatioMax;
  if (result.highFrameMeanAgreement !== highFrameMeanAgreement) {
    fail(id, 'highFrameMeanAgreement does not match globalRelErr');
  }
  if (result.varianceNotWorse !== varianceNotWorse) fail(id, 'varianceNotWorse does not match varRatio');
  if (result.defaultWeightMode !== 'effectively-unclamped-f32-max') {
    fail(id, 'defaultWeightMode must identify the professional unclamped estimator');
  }
  if (!sameJson(result.roi, proof.roi)) fail(id, 'roi differs from proof');
  const expectedCaptureConfig = expectedRestirCaptureConfig(proof);
  if (!sameJson(result.captureConfig, expectedCaptureConfig)) {
    fail(id, 'captureConfig or exact frame-seed schedule differs from proof');
  }
  validatePtDeviceIdentity(id, result.deviceIdentity);

  const stats = result.reservoirWeightStats;
  if (!isRecord(stats)) fail(id, 'reservoirWeightStats must be an object');
  const expectedPixels = proof.resolution.W * proof.resolution.H;
  if (stats.pixelCount !== expectedPixels) fail(id, 'reservoirWeightStats.pixelCount differs from resolution');
  if (!Number.isSafeInteger(stats.nonEmptyCount) || stats.nonEmptyCount <= 0 || stats.nonEmptyCount > expectedPixels) {
    fail(id, 'reservoirWeightStats.nonEmptyCount must be in 1..pixelCount');
  }
  near(id, stats.nonEmptyFraction, stats.nonEmptyCount / expectedPixels, 'reservoirWeightStats.nonEmptyFraction');
  if (stats.diagnosticClamp !== proof.diagnosticClamp) fail(id, 'reservoir diagnostic clamp differs from proof');
  if (!Number.isSafeInteger(stats.aboveDiagnosticClampCount) ||
      stats.aboveDiagnosticClampCount <= 0 ||
      stats.aboveDiagnosticClampCount > stats.nonEmptyCount) {
    fail(id, 'uncapped reservoir must contain a non-zero valid population above the legacy clamp');
  }
  near(
    id,
    stats.aboveDiagnosticClampFraction,
    stats.aboveDiagnosticClampCount / stats.nonEmptyCount,
    'reservoirWeightStats.aboveDiagnosticClampFraction',
  );
  const totalWeight = positive(id, stats.totalWeight, 'reservoirWeightStats.totalWeight');
  const clippedWeightMass = positive(id, stats.clippedWeightMass, 'reservoirWeightStats.clippedWeightMass');
  if (clippedWeightMass >= totalWeight) fail(id, 'legacy clamp cannot remove all resolved weight mass');
  near(
    id,
    stats.clippedWeightMassFraction,
    clippedWeightMass / totalWeight,
    'reservoirWeightStats.clippedWeightMassFraction',
  );
  const orderedWeights = ['min', 'p50', 'p90', 'p95', 'p99', 'max'].map((key) =>
    finite(id, stats.weight?.[key], `reservoirWeightStats.weight.${key}`, 0));
  for (let index = 1; index < orderedWeights.length; index += 1) {
    if (orderedWeights[index] < orderedWeights[index - 1]) {
      fail(id, 'reservoir weight quantiles are not monotone');
    }
  }
  finite(id, stats.weight?.mean, 'reservoirWeightStats.weight.mean', 0);
  if (!(stats.weight.max < proof.professionalDefaultWeightCeiling)) {
    fail(id, 'professional default clipped or saturated resolved reservoir weights');
  }

  if (!Array.isArray(result.clampControls) || result.clampControls.length !== proof.clampControls.length) {
    fail(id, 'clampControls must contain every proved cap exactly once');
  }
  if (!sameJson(result.clampControls.map((/** @type {any} */ entry) => entry.wCap), proof.clampControls)) {
    fail(id, 'clampControls caps are missing, duplicated, or out of order');
  }
  for (const entry of result.clampControls) {
    if (entry.effectivePackedWCap !== Math.fround(entry.wCap)) {
      fail(id, `clamp ${entry.wCap}.effectivePackedWCap differs from f32 packing`);
    }
    if (entry.promotionEligible !== false ||
        entry.estimatorMode !== 'intentionally-biased-finite-weight-clamp') {
      fail(id, `clamp ${entry.wCap} must be explicitly non-promotable and biased`);
    }
    const globalLum = positive(id, entry.globalLum, `clamp ${entry.wCap}.globalLum`);
    const roiLum = positive(id, entry.roiLum, `clamp ${entry.wCap}.roiLum`);
    near(id, entry.globalRelErr, relativeError(globalLum, base.globalLum), `clamp ${entry.wCap}.globalRelErr`);
    near(id, entry.roiRelErr, relativeError(roiLum, base.roiLum), `clamp ${entry.wCap}.roiRelErr`);
  }
  const legacyClampControl = result.clampControls[proof.clampControls.indexOf(proof.diagnosticClamp)];
  if (legacyClampControl == null || !(legacyClampControl.globalRelErr > globalRelErr)) {
    fail(id, 'legacy W clamp does not empirically move the mean farther from baseline than the default');
  }

  const paired = result.pairedSeedAnalysis;
  if (!isRecord(paired)) fail(id, 'pairedSeedAnalysis must be an object');
  if (paired.confidenceLevel !== proof.pairedSeedAnalysis.confidenceLevel ||
      paired.tCritical !== proof.pairedSeedAnalysis.tCritical ||
      paired.equivalenceMargin !== proof.thresholds.pairedEquivalenceMargin) {
    fail(id, 'paired-seed confidence configuration differs from proof');
  }
  if (!Array.isArray(paired.runs) || paired.runs.length !== proof.pairedSeedAnalysis.runs) {
    fail(id, 'pairedSeedAnalysis.runs count differs from proof');
  }
  /** @type {number[]} */
  const globalDiffs = [];
  /** @type {number[]} */
  const roiDiffs = [];
  paired.runs.forEach((entry, index) => {
    if (entry.run !== index) fail(id, 'paired run indices are missing, duplicated, or out of order');
    const baseGlobalLum = positive(id, entry.baseGlobalLum, `paired run ${index}.baseGlobalLum`);
    const rptGlobalLum = positive(id, entry.rptGlobalLum, `paired run ${index}.rptGlobalLum`);
    const baseRoiLum = positive(id, entry.baseRoiLum, `paired run ${index}.baseRoiLum`);
    const rptRoiLum = positive(id, entry.rptRoiLum, `paired run ${index}.rptRoiLum`);
    const globalDiff = (rptGlobalLum - baseGlobalLum) / baseGlobalLum;
    const roiDiff = (rptRoiLum - baseRoiLum) / baseRoiLum;
    near(id, entry.signedGlobalRelativeDifference, globalDiff, `paired run ${index}.signedGlobalRelativeDifference`);
    near(id, entry.signedRoiRelativeDifference, roiDiff, `paired run ${index}.signedRoiRelativeDifference`);
    globalDiffs.push(globalDiff);
    roiDiffs.push(roiDiff);
  });
  /** @param {number[]} values */
  const deriveConfidence = (values) => {
    const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
    const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (values.length - 1);
    const standardDeviation = Math.sqrt(variance);
    const halfWidth = proof.pairedSeedAnalysis.tCritical * standardDeviation / Math.sqrt(values.length);
    return { mean, standardDeviation, lower: mean - halfWidth, upper: mean + halfWidth };
  };
  const derivedGlobal95 = deriveConfidence(globalDiffs);
  const derivedRoi95 = deriveConfidence(roiDiffs);
  for (const [label, actual, expected] of [
    ['global', paired.global, derivedGlobal95],
    ['roi', paired.roi, derivedRoi95],
  ]) {
    for (const field of ['mean', 'standardDeviation', 'lower', 'upper']) {
      near(id, actual?.[field], expected[field], `pairedSeedAnalysis.${label}.${field}`);
    }
  }
  const globalEquivalent =
    derivedGlobal95.lower > -proof.thresholds.pairedEquivalenceMargin &&
    derivedGlobal95.upper < proof.thresholds.pairedEquivalenceMargin;
  const roiEquivalent =
    derivedRoi95.lower > -proof.thresholds.pairedEquivalenceMargin &&
    derivedRoi95.upper < proof.thresholds.pairedEquivalenceMargin;
  const pairedEquivalent = globalEquivalent && roiEquivalent;
  if (paired.globalEquivalent !== globalEquivalent) {
    fail(id, 'pairedSeedAnalysis.globalEquivalent does not match its CI');
  }
  if (paired.roiEquivalent !== roiEquivalent) {
    fail(id, 'pairedSeedAnalysis.roiEquivalent does not match its CI');
  }
  if (paired.equivalent !== pairedEquivalent) fail(id, 'pairedSeedAnalysis.equivalent does not match its CI');
  const meanAgreement = highFrameMeanAgreement && pairedEquivalent;
  if (result.meanAgreement !== meanAgreement) fail(id, 'meanAgreement does not match high-frame + paired-seed evidence');
  if (!(meanAgreement && varianceNotWorse)) fail(id, 'PASS verdict is not supported by ReSTIR-PT measurements');
  return {
    globalRelErr,
    roiRelErr,
    varRatio,
    meanAgreement,
    highFrameMeanAgreement,
    pairedMeanEquivalent: pairedEquivalent,
    pairedGlobal95: derivedGlobal95,
    pairedRoi95: derivedRoi95,
    varianceNotWorse,
    legacyClampWeightMassFraction: stats.clippedWeightMassFraction,
    professionalDefaultClippedWeightMass: 0,
  };
}

/** @param {any} proof @param {any} result */
function validateSobol(proof, result) {
  const id = proof.id;
  if (result.traceTier !== proof.traceTier) fail(id, 'traceTier differs from proof');
  if (result.reference?.sampling !== proof.reference.sampling || result.reference?.frames !== proof.reference.frames) {
    fail(id, 'reference sampling/frame count differs from proof');
  }
  if (result.candidateFrames !== proof.candidateFrames) fail(id, 'candidateFrames differs from proof');
  if (!sameJson(result.thresholds, proof.thresholds)) fail(id, 'thresholds differ from proof');
  if (!Array.isArray(result.scenes) || result.scenes.length !== proof.sceneIds.length) {
    fail(id, 'scene count differs from proof');
  }
  if (!sameJson(result.scenes.map((/** @type {any} */ scene) => scene.id), proof.sceneIds)) {
    fail(id, 'scene ids are missing, duplicated, or out of order');
  }
  const maxRatios = { globalRmse: 0, roiRmse: 0, elapsedMs: 0 };
  for (const scene of result.scenes) {
    if (scene.referenceFrames !== proof.reference.frames || scene.candidateFrames !== proof.candidateFrames) {
      fail(id, `${scene.id} frame budgets differ from proof`);
    }
    assertRoi(id, scene.roi, result.resolution, `${scene.id}.roi`);
    for (const armName of ['pcg', 'sobol']) {
      const arm = scene[armName];
      positive(id, arm?.elapsedMs, `${scene.id}.${armName}.elapsedMs`);
      finite(id, arm?.globalRmse, `${scene.id}.${armName}.globalRmse`, 0);
      finite(id, arm?.roiRmse, `${scene.id}.${armName}.roiRmse`, 0);
      finite(id, arm?.roiSpatialVariance, `${scene.id}.${armName}.roiSpatialVariance`, 0);
    }
    for (const field of ['globalRmse', 'roiRmse', 'roiSpatialVariance', 'elapsedMs']) {
      const ratio = finite(id, scene.ratios?.[field], `${scene.id}.ratios.${field}`, 0);
      near(id, ratio, ratioOrInfinity(scene.sobol[field], scene.pcg[field]), `${scene.id}.ratios.${field}`);
      if (scene.relativeErrors != null) {
        const rel = finite(id, scene.relativeErrors[field], `${scene.id}.relativeErrors.${field}`, 0);
        near(id, rel, relativeError(scene.sobol[field], scene.pcg[field]), `${scene.id}.relativeErrors.${field}`);
      }
    }
    const pass =
      scene.ratios.globalRmse <= proof.thresholds.maxGlobalRmseRatio &&
      scene.ratios.roiRmse <= proof.thresholds.maxRoiRmseRatio &&
      scene.ratios.elapsedMs <= proof.thresholds.maxElapsedMsRatio;
    if (scene.pass !== pass) fail(id, `${scene.id}.pass does not match measured ratios`);
    if (!pass) fail(id, `${scene.id} exceeds a promotion threshold`);
    maxRatios.globalRmse = Math.max(maxRatios.globalRmse, scene.ratios.globalRmse);
    maxRatios.roiRmse = Math.max(maxRatios.roiRmse, scene.ratios.roiRmse);
    maxRatios.elapsedMs = Math.max(maxRatios.elapsedMs, scene.ratios.elapsedMs);
  }
  for (const field of ['selected', 'evidenceClass', 'reason', 'requiredEvidence']) {
    if (result.defaultSelection?.[field] !== proof.defaultSelection[field]) {
      fail(id, `defaultSelection.${field} differs from proof`);
    }
  }
  if (result.defaultSelection?.evidencePath !== proof.resultPath) fail(id, 'defaultSelection.evidencePath differs from proof');
  near(id, result.defaultSelection.maxGlobalRmseRatio, maxRatios.globalRmse, 'defaultSelection.maxGlobalRmseRatio');
  near(id, result.defaultSelection.maxRoiRmseRatio, maxRatios.roiRmse, 'defaultSelection.maxRoiRmseRatio');
  near(id, result.defaultSelection.maxElapsedMsRatio, maxRatios.elapsedMs, 'defaultSelection.maxElapsedMsRatio');
  return {
    maxGlobalRmseRatio: maxRatios.globalRmse,
    maxRoiRmseRatio: maxRatios.roiRmse,
    maxElapsedMsRatio: maxRatios.elapsedMs,
    defaultSelected: result.defaultSelection.selected,
    supportState: result.defaultSelection.selected ? 'stable-default' : 'stable-opt-in',
  };
}

/**
 * Validate one GPU radiometric result and return only re-derived regression metrics.
 * @param {any} proof
 * @param {any} result
 * @param {{ expectedProvenance?: unknown, historicalBaseline?: boolean }} [options]
 */
export function validateRadiometricResult(proof, result, options = {}) {
  const common = validateCommon(proof, result, options);
  let metrics;
  if (proof.id === 'sppm') metrics = validateSppm(proof, result);
  else if (proof.id === 'bdpt') metrics = validateBdpt(proof, result);
  else if (proof.id === 'restir-pt') metrics = validateRestirPt(proof, result);
  else if (proof.id === 'sobol') metrics = validateSobol(proof, result);
  else fail(String(proof.id), 'unknown radiometric proof id');
  return { id: proof.id, capturedAtMs: common.capturedAtMs, metrics };
}

/**
 * Validate the deterministic specialty fixture and derive coverage rather than
 * trusting a caller-supplied glossy-coverage boolean.
 * @param {any} proof
 * @param {any} result
 */
export function validateRestirPtSpecialtyResult(proof, result) {
  const id = 'restir-pt-specialty';
  if (!isRecord(result)) fail(id, 'result must be an object');
  assertDeepFinite(id, result, 'result');
  if (result.schema !== proof.schema || result.mode !== proof.mode) fail(id, 'schema or mode differs from proof');
  if (!sameJson(result.coverage, proof.coverage)) fail(id, 'coverage differs from proof');
  if (!Array.isArray(result.cases) || result.cases.length !== proof.cases.length) fail(id, 'case count differs from proof');
  if (!sameJson(
    result.cases.map((/** @type {any} */ entry) => entry.id),
    proof.cases.map((/** @type {any} */ entry) => entry.id),
  )) {
    fail(id, 'case ids are missing, duplicated, or out of order');
  }
  let luminanceChecksum = 0;
  let pdfChecksum = 0;
  let maxAbsoluteError = 0;
  let maxRelativeError = 0;
  for (let index = 0; index < proof.cases.length; index += 1) {
    const expected = proof.cases[index];
    const entry = result.cases[index];
    if (entry.materialSource !== expected.materialSource || !sameJson(entry.activeLobes, expected.activeLobes)) {
      fail(id, `${expected.id} source/lobe coverage differs from proof`);
    }
    const pdfSrc = positive(id, entry.reference?.pdfSrc, `${expected.id}.reference.pdfSrc`);
    const pHat = positive(id, entry.reference?.pHat, `${expected.id}.reference.pHat`);
    const wCandidate = positive(id, entry.reference?.wCandidate, `${expected.id}.reference.wCandidate`);
    const reservoirW = positive(id, entry.reference?.W, `${expected.id}.reference.W`);
    near(id, wCandidate, pHat / pdfSrc, `${expected.id}.reference.wCandidate`);
    near(id, reservoirW, 1 / pdfSrc, `${expected.id}.reference.W`);
    const baseLum = positive(id, entry.basePath?.luminance, `${expected.id}.basePath.luminance`);
    const restirLum = positive(id, entry.restirPt?.luminance, `${expected.id}.restirPt.luminance`);
    near(id, restirLum, baseLum, `${expected.id}.restirPt.luminance`);
    const absDiff = finite(id, entry.ab?.absDiff, `${expected.id}.ab.absDiff`, 0);
    const relErr = finite(id, entry.ab?.relativeError, `${expected.id}.ab.relativeError`, 0);
    if (absDiff !== 0 || relErr !== 0) fail(id, `${expected.id} one-sample identity must be exact`);
    if (Math.abs(finite(id, entry.ab?.lobeDeltaFromNeutral, `${expected.id}.ab.lobeDeltaFromNeutral`)) < expected.minAbsLobeDeltaFromNeutral) {
      fail(id, `${expected.id} lobe delta is below the proof threshold`);
    }
    luminanceChecksum += (index + 1) * restirLum;
    pdfChecksum += (index + 1) * pdfSrc;
    maxAbsoluteError = Math.max(maxAbsoluteError, absDiff);
    maxRelativeError = Math.max(maxRelativeError, relErr);
  }
  if (result.summary?.caseCount !== result.cases.length) fail(id, 'summary.caseCount is not derived from cases');
  near(id, result.summary.maxAbsoluteError, maxAbsoluteError, 'summary.maxAbsoluteError');
  near(id, result.summary.maxRelativeError, maxRelativeError, 'summary.maxRelativeError');
  near(id, result.summary.luminanceChecksum, luminanceChecksum, 'summary.luminanceChecksum');
  near(id, result.summary.pdfChecksum, pdfChecksum, 'summary.pdfChecksum');
  if (!sameJson(result.summary, proof.summary)) fail(id, 'summary differs from pinned proof');
  const requiredGlossyLobes = ['anisotropy', 'clearcoat', 'iridescence', 'sheen', 'specular'];
  const glossyFixtureCovered = requiredGlossyLobes.every((lobe) => result.coverage.specialtyLobes.includes(lobe)) &&
    result.coverage.materialSources.includes('scalar') &&
    result.coverage.materialSources.includes('map-backed-effective-values');
  if (!glossyFixtureCovered) fail(id, 'fixture does not cover every required glossy specialty lobe/source');
  return {
    mode: result.mode,
    caseCount: result.cases.length,
    maxAbsoluteError,
    maxRelativeError,
    glossyFixtureCovered,
  };
}
