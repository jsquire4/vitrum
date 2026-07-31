// @ts-check

/** @typedef {{
 *   complete: ReadonlySet<string>,
 *   partial: ReadonlySet<string>,
 *   failure?: ReadonlySet<string>,
 * }} CaseVerdictContract */

/** @type {Readonly<Record<string, CaseVerdictContract>>} */
const CASE_VERDICT_CONTRACT = Object.freeze({
  a8: Object.freeze({
    complete: new Set(['NEGLIGIBLE']),
    partial: new Set(['SMALL', 'MODERATE']),
    failure: new Set(['SIGNIFICANT']),
  }),
  sun: Object.freeze({
    complete: new Set(['PASS']),
    partial: new Set(['PASS-PARTIAL']),
  }),
  glass: Object.freeze({
    complete: new Set(['PASS']),
    partial: new Set(['FINDING', 'SMOKE']),
  }),
  glossy: Object.freeze({
    complete: new Set(['PASS']),
    partial: new Set(['FINDING', 'PASS-WEAK']),
  }),
});

export class WalkaroundRunValidationError extends Error {
  /** @param {string} code @param {string} message */
  constructor(code, message) {
    super(message);
    this.name = 'WalkaroundRunValidationError';
    this.code = code;
  }
}

/** @param {unknown} value @returns {value is Record<string, any>} */
function isRecord(value) {
  return value != null && typeof value === 'object' && !Array.isArray(value);
}

/**
 * @param {{ mtimeMs: number, size: number, text: string } | null} beforeState
 * @param {{ mtimeMs: number, size: number, text: string } | null} afterState
 * @param {number} startedAtMs
 */
function assertFreshResult(beforeState, afterState, startedAtMs) {
  if (afterState == null) {
    throw new WalkaroundRunValidationError(
      'walkaround-ab-missing-result',
      'The native WebGPU harness exited 0 but did not write a readable result artifact.',
    );
  }
  if (!Number.isFinite(afterState.mtimeMs) || !Number.isSafeInteger(afterState.size)) {
    throw new WalkaroundRunValidationError(
      'walkaround-ab-invalid-result',
      'The native WebGPU harness wrote a result artifact with invalid file metadata.',
    );
  }
  const changed =
    beforeState == null ||
    afterState.mtimeMs !== beforeState.mtimeMs ||
    afterState.size !== beforeState.size ||
    afterState.text !== beforeState.text;
  // Leave room for filesystems whose timestamp precision rounds to whole seconds.
  const writtenDuringRun = afterState.mtimeMs >= startedAtMs - 2_000;
  if (!changed || !writtenDuringRun) {
    throw new WalkaroundRunValidationError(
      'walkaround-ab-stale-result',
      'The native WebGPU harness exited 0 without replacing the result artifact during this run.',
    );
  }
}

/** @param {string} path @param {string} expectation @returns {never} */
function invalidEvidence(path, expectation) {
  throw new WalkaroundRunValidationError(
    'walkaround-ab-invalid-result-evidence',
    `${path} ${expectation}`,
  );
}

/** @param {unknown} value @param {string} path @returns {Record<string, any>} */
function requireRecord(value, path) {
  if (!isRecord(value)) invalidEvidence(path, 'must be an object.');
  return value;
}

/**
 * @param {unknown} value
 * @param {string} path
 * @param {{ minimum?: number, integer?: boolean, positive?: boolean }} [options]
 * @returns {number}
 */
function requireFiniteNumber(value, path, options = {}) {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    invalidEvidence(path, 'must be a number-typed finite value.');
  }
  if (options.integer === true && !Number.isSafeInteger(value)) {
    invalidEvidence(path, 'must be a safe integer.');
  }
  if (options.positive === true && !(value > 0)) {
    invalidEvidence(path, 'must be positive.');
  }
  if (options.minimum != null && value < options.minimum) {
    invalidEvidence(path, `must be >= ${options.minimum}.`);
  }
  return value;
}

/** @param {unknown} value @param {string} path @returns {boolean} */
function requireBoolean(value, path) {
  if (typeof value !== 'boolean') invalidEvidence(path, 'must be a boolean.');
  return value;
}

/** @param {unknown} value @param {string} path @returns {string} */
function requireNonemptyString(value, path) {
  if (typeof value !== 'string' || value.length === 0) {
    invalidEvidence(path, 'must be a non-empty string.');
  }
  return value;
}

/**
 * @param {Record<string, any>} record
 * @param {string} key
 * @param {string} path
 * @param {{ minimum?: number, integer?: boolean, positive?: boolean }} [options]
 * @returns {number}
 */
function finiteField(record, key, path, options) {
  return requireFiniteNumber(record[key], `${path}.${key}`, options);
}

/** @param {number} actual @param {number} expected @param {string} path */
function requireNearlyEqual(actual, expected, path) {
  const tolerance = 1e-9 * Math.max(1, Math.abs(actual), Math.abs(expected));
  if (Math.abs(actual - expected) > tolerance) {
    invalidEvidence(path, `must match the derived value ${expected}.`);
  }
}

/** @param {unknown} value @param {readonly number[]} expected @param {string} path */
function requireFiniteVector(value, expected, path) {
  if (!Array.isArray(value) || value.length !== expected.length) {
    invalidEvidence(path, `must be a ${expected.length}-component array.`);
  }
  for (let i = 0; i < expected.length; i += 1) {
    const component = requireFiniteNumber(value[i], `${path}[${i}]`);
    requireNearlyEqual(component, expected[i], `${path}[${i}]`);
  }
}

/**
 * @param {unknown} value
 * @param {string} path
 * @param {number} width
 * @param {number} height
 */
function requirePixelRegion(value, path, width, height) {
  const region = requireRecord(value, path);
  for (const key of ['x0', 'y0', 'x1', 'y1']) {
    finiteField(region, key, path, { integer: true, minimum: 0 });
  }
  if (region.x0 >= region.x1 || region.y0 >= region.y1 || region.x1 > width || region.y1 > height) {
    invalidEvidence(path, 'must have positive area within the captured dimensions.');
  }
}

/**
 * @param {unknown} value
 * @param {string} path
 * @param {string} blocker
 */
function requirePromotion(value, path, blocker) {
  const promotion = requireRecord(value, path);
  const defaultReady = requireBoolean(promotion.defaultReady, `${path}.defaultReady`);
  if (defaultReady !== false) invalidEvidence(`${path}.defaultReady`, 'must be false.');
  if (promotion.blocker !== blocker) {
    invalidEvidence(`${path}.blocker`, `must be ${JSON.stringify(blocker)}.`);
  }
  requireNonemptyString(promotion.requiredEvidence, `${path}.requiredEvidence`);
}

/** @param {string} id @param {Record<string, any>} row */
function validateCommonCaseEvidence(id, row) {
  const expectedId = {
    a8: 'A8',
    sun: 'SUN',
    glass: 'GLASS',
    glossy: 'GLOSSY',
  }[id];
  if (row.id !== expectedId) {
    invalidEvidence(`${id}.id`, `must be ${JSON.stringify(expectedId)}.`);
  }
  requireNonemptyString(row.description, `${id}.description`);
  requireNonemptyString(row.qualityProfile, `${id}.qualityProfile`);
  const spp = requireFiniteNumber(row.spp, `${id}.spp`, {
    integer: true,
    positive: true,
  });
  const renderConfig = requireRecord(row.renderConfig, `${id}.renderConfig`);
  const width = finiteField(renderConfig, 'width', `${id}.renderConfig`, {
    integer: true,
    positive: true,
  });
  const height = finiteField(renderConfig, 'height', `${id}.renderConfig`, {
    integer: true,
    positive: true,
  });
  const configSpp = finiteField(renderConfig, 'spp', `${id}.renderConfig`, {
    integer: true,
    positive: true,
  });
  if (configSpp !== spp) invalidEvidence(`${id}.renderConfig.spp`, `must equal ${id}.spp.`);
  if (row.resolution !== `${width}x${height}`) {
    invalidEvidence(`${id}.resolution`, 'must match renderConfig width and height.');
  }
  requireFiniteNumber(row.renderTimeSec, `${id}.renderTimeSec`, { minimum: 0 });
  if (
    !Array.isArray(row.notes) ||
    row.notes.length === 0 ||
    row.notes.some((note) => typeof note !== 'string' || note.length === 0)
  ) {
    invalidEvidence(`${id}.notes`, 'must be a non-empty array of non-empty strings.');
  }
}

/** @param {Record<string, any>} row @param {string} verdict */
function validateA8Evidence(row, verdict) {
  const biased = requireRecord(row.biased, 'a8.biased');
  const unbiased = requireRecord(row.unbiased, 'a8.unbiased');
  const delta = requireRecord(row.delta, 'a8.delta');
  const regions = requireRecord(row.regions, 'a8.regions');
  const metricKeys = ['overall', 'floor', 'ceiling', 'leftWall', 'rightWall'];
  for (const key of metricKeys) {
    const biasedValue = finiteField(biased, key, 'a8.biased', { minimum: 0 });
    const unbiasedValue = finiteField(unbiased, key, 'a8.unbiased', { minimum: 0 });
    const deltaValue = finiteField(delta, key, 'a8.delta');
    requireNearlyEqual(deltaValue, unbiasedValue - biasedValue, `a8.delta.${key}`);
  }
  for (const key of ['floor', 'ceiling', 'leftWall', 'rightWall']) {
    requirePixelRegion(
      regions[key],
      `a8.regions.${key}`,
      row.renderConfig.width,
      row.renderConfig.height,
    );
  }
  const absoluteOverallDelta = Math.abs(delta.overall);
  const derivedVerdict =
    absoluteOverallDelta < 0.005
      ? 'NEGLIGIBLE'
      : absoluteOverallDelta < 0.03
        ? 'SMALL'
        : absoluteOverallDelta < 0.06
          ? 'MODERATE'
          : 'SIGNIFICANT';
  if (verdict !== derivedVerdict) {
    invalidEvidence('a8.verdict', `must be ${derivedVerdict} for the measured overall delta.`);
  }
}

/** @param {Record<string, any>} row @param {string} verdict */
function validateSunEvidence(row, verdict) {
  requireFiniteVector(row.sunTravelDirection, [0, 0, -1], 'sun.sunTravelDirection');
  requireFiniteVector(row.primaryLightDir, [0, 0, 1], 'sun.primaryLightDir');
  const intensity = requireFiniteNumber(row.sunIntensity, 'sun.sunIntensity', { minimum: 0 });
  const receiverAlbedo = requireFiniteNumber(row.receiverAlbedo, 'sun.receiverAlbedo', {
    minimum: 0,
  });
  const floorAlbedo = requireFiniteNumber(row.floorAlbedo, 'sun.floorAlbedo', { minimum: 0 });
  const cosTheta = requireFiniteNumber(row.cosTheta, 'sun.cosTheta', { minimum: 0 });
  const expectedReceiver = requireFiniteNumber(
    row.analyticExpectedReceiverLum,
    'sun.analyticExpectedReceiverLum',
    { minimum: 0 },
  );
  const expectedFloor = requireFiniteNumber(
    row.analyticExpectedFloorLum,
    'sun.analyticExpectedFloorLum',
    { minimum: 0 },
  );
  const rendered = requireRecord(row.rendered, 'sun.rendered');
  for (const key of ['receiverLum', 'sideDiagnosticLum', 'floorLum', 'leftWallLum', 'overall']) {
    finiteField(rendered, key, 'sun.rendered', { minimum: 0 });
  }
  const receiverRatio = requireFiniteNumber(
    row.receiverRatioToAnalytic,
    'sun.receiverRatioToAnalytic',
    { minimum: 0 },
  );
  const floorRatio = requireFiniteNumber(row.floorRatioToAnalytic, 'sun.floorRatioToAnalytic', {
    minimum: 0,
  });
  const analyticAgreement = requireBoolean(row.analyticAgreement, 'sun.analyticAgreement');
  const diffuseOnly = requireBoolean(row.diffuseOnly, 'sun.diffuseOnly');
  const shadowAssertionAuthored = requireBoolean(
    row.shadowAssertionAuthored,
    'sun.shadowAssertionAuthored',
  );
  if (!diffuseOnly) invalidEvidence('sun.diffuseOnly', 'must be true.');
  if (shadowAssertionAuthored) {
    invalidEvidence('sun.shadowAssertionAuthored', 'must be false for this direct-light proof.');
  }
  requireNearlyEqual(intensity, 0.3, 'sun.sunIntensity');
  requireNearlyEqual(receiverAlbedo, 0.8, 'sun.receiverAlbedo');
  requireNearlyEqual(floorAlbedo, receiverAlbedo, 'sun.floorAlbedo');
  requireNearlyEqual(cosTheta, 1, 'sun.cosTheta');
  const derivedExpected = (intensity * cosTheta * receiverAlbedo) / Math.PI;
  requireNearlyEqual(expectedReceiver, derivedExpected, 'sun.analyticExpectedReceiverLum');
  requireNearlyEqual(expectedFloor, derivedExpected, 'sun.analyticExpectedFloorLum');
  requireNearlyEqual(rendered.floorLum, rendered.receiverLum, 'sun.rendered.floorLum');
  requireNearlyEqual(rendered.leftWallLum, rendered.sideDiagnosticLum, 'sun.rendered.leftWallLum');
  const derivedRatio = derivedExpected > 0 ? rendered.receiverLum / derivedExpected : 0;
  requireNearlyEqual(receiverRatio, derivedRatio, 'sun.receiverRatioToAnalytic');
  requireNearlyEqual(floorRatio, derivedRatio, 'sun.floorRatioToAnalytic');
  const derivedAgreement = derivedRatio >= 0.5 && derivedRatio <= 1.5;
  if (analyticAgreement !== derivedAgreement) {
    invalidEvidence('sun.analyticAgreement', 'must match the measured analytic ratio.');
  }
  const derivedVerdict =
    rendered.receiverLum > 0.01 && derivedAgreement
      ? 'PASS'
      : rendered.receiverLum > 0.01
        ? 'PASS-PARTIAL'
        : 'FAIL';
  if (verdict !== derivedVerdict) {
    invalidEvidence('sun.verdict', `must be ${derivedVerdict} for the measured receiver signal.`);
  }
  const regions = requireRecord(row.regions, 'sun.regions');
  requirePixelRegion(
    regions.receiver,
    'sun.regions.receiver',
    row.renderConfig.width,
    row.renderConfig.height,
  );
  requirePixelRegion(
    regions.sideDiagnostic,
    'sun.regions.sideDiagnostic',
    row.renderConfig.width,
    row.renderConfig.height,
  );
}

/** @param {Record<string, any>} row @param {string} verdict */
function validateGlassEvidence(row, verdict) {
  const glass = requireRecord(row.glass, 'glass.glass');
  const noGlass = requireRecord(row.noGlass, 'glass.noGlass');
  const delta = requireRecord(row.delta, 'glass.delta');
  const glassCenter = finiteField(glass, 'centreRegionLum', 'glass.glass', { minimum: 0 });
  const glassOverall = finiteField(glass, 'overall', 'glass.glass', { minimum: 0 });
  const noGlassCenter = finiteField(noGlass, 'centreRegionLum', 'glass.noGlass', {
    minimum: 0,
  });
  const noGlassOverall = finiteField(noGlass, 'overall', 'glass.noGlass', { minimum: 0 });
  const centreRatio = requireFiniteNumber(row.centreRatio, 'glass.centreRatio', { minimum: 0 });
  const overallRatio = requireFiniteNumber(row.overallRatio, 'glass.overallRatio', { minimum: 0 });
  const centreDelta = finiteField(delta, 'centreRegionLum', 'glass.delta');
  const overallDelta = finiteField(delta, 'overall', 'glass.delta');
  const minRatio = requireFiniteNumber(row.expectedMinCentreRatio, 'glass.expectedMinCentreRatio', {
    minimum: 0,
  });
  const maxCentreRatio = requireFiniteNumber(
    row.expectedMaxCentreRatio,
    'glass.expectedMaxCentreRatio',
    { minimum: 0 },
  );
  const maxOverallRatio = requireFiniteNumber(
    row.expectedMaxOverallRatio,
    'glass.expectedMaxOverallRatio',
    { minimum: 0 },
  );
  const minSignalDelta = requireFiniteNumber(row.minSignalDelta, 'glass.minSignalDelta', {
    minimum: 0,
  });
  const effectObserved = requireBoolean(row.materialEffectObserved, 'glass.materialEffectObserved');
  const withinBounds = requireBoolean(
    row.ratioWithinPromotionBounds,
    'glass.ratioWithinPromotionBounds',
  );
  const fresnelTransmission = requireFiniteNumber(
    row.fresnelT_normal_incidence_n1p5,
    'glass.fresnelT_normal_incidence_n1p5',
    { minimum: 0 },
  );
  requireNearlyEqual(fresnelTransmission, 0.92, 'glass.fresnelT_normal_incidence_n1p5');
  requireNearlyEqual(minRatio, 0.5, 'glass.expectedMinCentreRatio');
  requireNearlyEqual(maxCentreRatio, 4, 'glass.expectedMaxCentreRatio');
  requireNearlyEqual(maxOverallRatio, 8, 'glass.expectedMaxOverallRatio');
  requireNearlyEqual(minSignalDelta, 1e-4, 'glass.minSignalDelta');
  const derivedCentreRatio = noGlassCenter > 0.01 ? glassCenter / noGlassCenter : 0;
  const derivedOverallRatio = noGlassOverall > 0.01 ? glassOverall / noGlassOverall : 0;
  requireNearlyEqual(centreRatio, derivedCentreRatio, 'glass.centreRatio');
  requireNearlyEqual(overallRatio, derivedOverallRatio, 'glass.overallRatio');
  requireNearlyEqual(centreDelta, glassCenter - noGlassCenter, 'glass.delta.centreRegionLum');
  requireNearlyEqual(overallDelta, glassOverall - noGlassOverall, 'glass.delta.overall');
  const derivedEffect = Math.max(Math.abs(centreDelta), Math.abs(overallDelta)) >= minSignalDelta;
  if (effectObserved !== derivedEffect) {
    invalidEvidence('glass.materialEffectObserved', 'must match the measured deltas.');
  }
  const derivedBounds = centreRatio <= maxCentreRatio && overallRatio <= maxOverallRatio;
  if (withinBounds !== derivedBounds) {
    invalidEvidence('glass.ratioWithinPromotionBounds', 'must match the measured ratios.');
  }
  const notBlack = glassCenter > 0.01;
  const ratioPass = centreRatio >= minRatio;
  const derivedVerdict =
    notBlack && ratioPass && derivedEffect && derivedBounds
      ? 'PASS'
      : notBlack && ratioPass && derivedEffect
        ? 'FINDING'
        : notBlack
          ? 'SMOKE'
          : 'FAIL';
  if (verdict !== derivedVerdict) {
    invalidEvidence('glass.verdict', `must be ${derivedVerdict} for the measured transport.`);
  }
  const regions = requireRecord(row.regions, 'glass.regions');
  requirePixelRegion(
    regions.center,
    'glass.regions.center',
    row.renderConfig.width,
    row.renderConfig.height,
  );
  if (verdict === 'FINDING') {
    requirePromotion(row.promotion, 'glass.promotion', 'glass-transport-radiance-blowout');
  }
}

/** @param {Record<string, any>} row @param {string} verdict */
function validateGlossyEvidence(row, verdict) {
  if (row.sampleRegion !== 'visible-back-wall-center-crop') {
    invalidEvidence('glossy.sampleRegion', 'must be "visible-back-wall-center-crop".');
  }
  const metal = requireRecord(row.metal, 'glossy.metal');
  const diffuse = requireRecord(row.diffuse, 'glossy.diffuse');
  const delta = requireRecord(row.delta, 'glossy.delta');
  const metalSample = finiteField(metal, 'sampleRegionLum', 'glossy.metal', { minimum: 0 });
  const metalFloor = finiteField(metal, 'floorLum', 'glossy.metal', { minimum: 0 });
  const metalOverall = finiteField(metal, 'overall', 'glossy.metal', { minimum: 0 });
  const diffuseSample = finiteField(diffuse, 'sampleRegionLum', 'glossy.diffuse', {
    minimum: 0,
  });
  const diffuseFloor = finiteField(diffuse, 'floorLum', 'glossy.diffuse', { minimum: 0 });
  const diffuseOverall = finiteField(diffuse, 'overall', 'glossy.diffuse', {
    minimum: 0,
  });
  const sampleRatio = requireFiniteNumber(row.sampleRatio, 'glossy.sampleRatio', {
    minimum: 0,
  });
  const floorRatio = requireFiniteNumber(row.floorRatio, 'glossy.floorRatio', { minimum: 0 });
  const sampleDelta = finiteField(delta, 'sampleRegionLum', 'glossy.delta');
  const floorDelta = finiteField(delta, 'floorLum', 'glossy.delta');
  const overallDelta = finiteField(delta, 'overall', 'glossy.delta');
  const expectedMinRatio = requireFiniteNumber(
    row.expectedMinSampleRatio,
    'glossy.expectedMinSampleRatio',
    { minimum: 0 },
  );
  const expectedMinFloorRatio = requireFiniteNumber(
    row.expectedMinFloorRatio,
    'glossy.expectedMinFloorRatio',
    { minimum: 0 },
  );
  const minSignalDelta = requireFiniteNumber(row.minSignalDelta, 'glossy.minSignalDelta', {
    minimum: 0,
  });
  const effectObserved = requireBoolean(
    row.materialEffectObserved,
    'glossy.materialEffectObserved',
  );
  requireNearlyEqual(metalFloor, metalSample, 'glossy.metal.floorLum');
  requireNearlyEqual(diffuseFloor, diffuseSample, 'glossy.diffuse.floorLum');
  const derivedRatio = diffuseSample > 0.01 ? metalSample / diffuseSample : 0;
  requireNearlyEqual(sampleRatio, derivedRatio, 'glossy.sampleRatio');
  requireNearlyEqual(floorRatio, derivedRatio, 'glossy.floorRatio');
  requireNearlyEqual(sampleDelta, metalSample - diffuseSample, 'glossy.delta.sampleRegionLum');
  requireNearlyEqual(floorDelta, sampleDelta, 'glossy.delta.floorLum');
  requireNearlyEqual(overallDelta, metalOverall - diffuseOverall, 'glossy.delta.overall');
  requireNearlyEqual(expectedMinRatio, 0.8, 'glossy.expectedMinSampleRatio');
  requireNearlyEqual(expectedMinFloorRatio, 0.8, 'glossy.expectedMinFloorRatio');
  requireNearlyEqual(minSignalDelta, 1e-4, 'glossy.minSignalDelta');
  const derivedEffect = Math.max(Math.abs(sampleDelta), Math.abs(overallDelta)) >= minSignalDelta;
  if (effectObserved !== derivedEffect) {
    invalidEvidence('glossy.materialEffectObserved', 'must match the measured deltas.');
  }
  const notBlack = metalSample > 1e-4 || metalOverall > 1e-3;
  const plausible = sampleRatio >= expectedMinRatio;
  const derivedVerdict =
    notBlack && plausible && derivedEffect
      ? 'PASS'
      : notBlack && derivedEffect
        ? 'FINDING'
        : notBlack
          ? 'PASS-WEAK'
          : 'FAIL';
  if (verdict !== derivedVerdict) {
    invalidEvidence(
      'glossy.verdict',
      `must be ${derivedVerdict} for the measured material response.`,
    );
  }
  const regions = requireRecord(row.regions, 'glossy.regions');
  requirePixelRegion(
    regions.sample,
    'glossy.regions.sample',
    row.renderConfig.width,
    row.renderConfig.height,
  );
  requirePromotion(
    row.promotion,
    'glossy.promotion',
    'ddgi-irradiance-cache-not-ggx-filtered-radiance',
  );
}

/** @param {string} id @param {Record<string, any>} row @param {string} verdict */
function validateCaseEvidence(id, row, verdict) {
  validateCommonCaseEvidence(id, row);
  if (id === 'a8') validateA8Evidence(row, verdict);
  else if (id === 'sun') validateSunEvidence(row, verdict);
  else if (id === 'glass') validateGlassEvidence(row, verdict);
  else validateGlossyEvidence(row, verdict);
}

/**
 * @param {unknown} payload
 * @param {{
 *   expectedCaseIds: string[],
 *   beforeState: { mtimeMs: number, size: number, text: string } | null,
 *   afterState: { mtimeMs: number, size: number, text: string } | null,
 *   startedAtMs: number,
 * }} options
 */
export function validateWalkaroundRunResult(payload, options) {
  assertFreshResult(options.beforeState, options.afterState, options.startedAtMs);
  if (!isRecord(payload)) {
    throw new WalkaroundRunValidationError(
      'walkaround-ab-invalid-result',
      'The native WebGPU harness exited 0 but wrote a non-object result artifact.',
    );
  }

  const expectedCaseIds = [...new Set(options.expectedCaseIds)];
  if (expectedCaseIds.length === 0 || expectedCaseIds.length !== options.expectedCaseIds.length) {
    throw new WalkaroundRunValidationError(
      'walkaround-ab-invalid-case-selection',
      'The walkaround result validator requires a non-empty, unique case selection.',
    );
  }

  /** @type {Record<string, string>} */
  const caseVerdicts = {};
  /** @type {string[]} */
  const partialCaseIds = [];
  for (const id of expectedCaseIds) {
    const contract = CASE_VERDICT_CONTRACT[id];
    if (contract == null) {
      throw new WalkaroundRunValidationError(
        'walkaround-ab-invalid-case-selection',
        `Unknown walkaround regression case: ${id}`,
      );
    }
    const row = payload[id];
    if (!isRecord(row) || typeof row.verdict !== 'string') {
      throw new WalkaroundRunValidationError(
        'walkaround-ab-incomplete-result',
        `The result artifact is missing a verdict for: ${id}`,
      );
    }
    const verdict = row.verdict;
    if (verdict === 'FAIL' || verdict === 'ERROR') {
      throw new WalkaroundRunValidationError(
        'walkaround-ab-nonpass-result',
        `The ${id} case returned ${verdict}.`,
      );
    }
    const isComplete = contract.complete.has(verdict);
    const isPartial = contract.partial.has(verdict);
    const isFailure = contract.failure?.has(verdict) === true;
    if (!isComplete && !isPartial && !isFailure) {
      throw new WalkaroundRunValidationError(
        'walkaround-ab-invalid-verdict',
        `The ${id} case returned an unknown verdict: ${verdict}`,
      );
    }
    validateCaseEvidence(id, row, verdict);
    if (isFailure) {
      throw new WalkaroundRunValidationError(
        'walkaround-ab-nonpass-result',
        `The ${id} case returned ${verdict}, which exceeds its regression threshold.`,
      );
    }
    if (isComplete) {
      caseVerdicts[id] = verdict;
      continue;
    }
    if (isPartial) {
      caseVerdicts[id] = verdict;
      partialCaseIds.push(`${id}:${verdict}`);
      continue;
    }
  }

  return {
    caseVerdicts,
    partialCaseIds,
    partial: partialCaseIds.length > 0,
  };
}
