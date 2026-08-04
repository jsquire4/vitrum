import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import test from 'node:test';

import {
  RADIOMETRIC_AB_PROOFS,
  RESTIR_PT_SPECIALTY_PROOF,
} from '../../tools/radiometric-ab/proofs.mjs';
import {
  validateRadiometricResult,
  validateRestirPtSpecialtyResult,
} from '../../tools/radiometric-ab/resultValidation.mjs';
import {
  buildRestirPtCaptureConfig,
  buildRestirPtResult,
  RESTIR_PT_WEIGHT_MODE,
} from '../../tools/radiometric-ab/restirPtResultContract.mjs';
import { validateWalkaroundRunResult } from '../../tools/radiometric-ab/walkaroundRunValidation.mjs';

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..');

async function readJson(path) {
  return JSON.parse(await readFile(resolve(repoRoot, path), 'utf8'));
}

function validSourceManifestFixture(proof) {
  const files = proof.sourceRoots
    .map((root, index) => ({
      path: `${root}/fixture-${index}.ts`,
      sha256: index.toString(16).padStart(64, '0'),
    }))
    .sort((a, b) => a.path.localeCompare(b.path));
  return {
    schema: 'vitrum.radiometric-ab.source-manifest.v1',
    roots: [...proof.sourceRoots],
    files,
    digestSha256: 'f'.repeat(64),
  };
}

function validRestirCaptureConfigFixture(proof) {
  const schedule = proof.capture.seedSchedule;
  const multiplier = BigInt(schedule.multiplier);
  const increment = BigInt(schedule.increment);
  const runStride = BigInt(schedule.runStride);
  const u32 = (value) => Number(BigInt.asUintN(32, value));
  return buildRestirPtCaptureConfig({
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
    effectiveMClamp: proof.capture.effectiveMClamp,
    seeds: {
      ...schedule,
      meanFrameSeeds: Array.from({ length: proof.meanFrames }, (_, frame) =>
        u32(BigInt(frame) * multiplier + increment),
      ),
      varianceRunFrameSeeds: Array.from({ length: proof.varianceRuns }, (_, run) =>
        Array.from({ length: proof.varianceFramesPerRun }, (_, frame) => {
          const globalFrame = BigInt(run * proof.varianceFramesPerRun + frame);
          return u32(globalFrame * multiplier + increment + BigInt(run) * runStride);
        }),
      ),
    },
  });
}

function validRestirDeviceIdentityFixture() {
  return {
    schema: 'vitrum.radiometric-ab.device-identity.v1',
    adapter: { description: 'test adapter' },
    adapterFeatures: [],
    deviceFeatures: [],
    advertisedLimits: {
      maxStorageBuffersPerShaderStage: 28,
      maxStorageTexturesPerShaderStage: 5,
      maxBindGroups: 5,
    },
    requestedLimits: {},
    runtime: {
      kind: 'deno',
      version: { deno: 'test' },
      build: { target: 'test-target' },
    },
    vulkanIcdFilenames: null,
  };
}

const WALKAROUND_REGIONS = {
  a8: {
    floor: { x0: 10, y0: 85, x1: 118, y1: 127 },
    ceiling: { x0: 10, y0: 0, x1: 118, y1: 20 },
    leftWall: { x0: 0, y0: 20, x1: 20, y1: 108 },
    rightWall: { x0: 108, y0: 20, x1: 128, y1: 108 },
  },
  sun: {
    receiver: { x0: 30, y0: 42, x1: 98, y1: 86 },
    sideDiagnostic: { x0: 0, y0: 30, x1: 15, y1: 98 },
  },
  glass: {
    center: { x0: 48, y0: 48, x1: 80, y1: 80 },
  },
  glossy: {
    sample: { x0: 32, y0: 32, x1: 96, y1: 96 },
  },
};

function walkaroundCommon(id, description) {
  return {
    id,
    description,
    spp: 16,
    resolution: '128x128',
    renderTimeSec: 1,
    qualityProfile: 'fixture',
    renderConfig: { width: 128, height: 128, spp: 16 },
    notes: ['fixture evidence'],
  };
}

function validA8WalkaroundFixture(overallDelta) {
  const biased = {
    overall: 0.1,
    floor: 0.1,
    ceiling: 0.1,
    leftWall: 0.1,
    rightWall: 0.1,
  };
  const unbiased = Object.fromEntries(
    Object.entries(biased).map(([key, value]) => [
      key,
      value + (key === 'overall' ? overallDelta : 0.001),
    ]),
  );
  const delta = Object.fromEntries(
    Object.keys(biased).map((key) => [key, unbiased[key] - biased[key]]),
  );
  const absoluteDelta = Math.abs(overallDelta);
  const verdict =
    absoluteDelta < 0.005
      ? 'NEGLIGIBLE'
      : absoluteDelta < 0.03
        ? 'SMALL'
        : absoluteDelta < 0.06
          ? 'MODERATE'
          : 'SIGNIFICANT';
  return {
    ...walkaroundCommon('A8', 'A8 fixture'),
    regions: structuredClone(WALKAROUND_REGIONS.a8),
    biased,
    unbiased,
    delta,
    verdict,
  };
}

function validSunWalkaroundFixture(receiverRatio) {
  const intensity = 0.3;
  const albedo = 0.8;
  const analytic = (intensity * albedo) / Math.PI;
  const receiverLum = analytic * receiverRatio;
  const agreement = receiverRatio >= 0.5 && receiverRatio <= 1.5;
  return {
    ...walkaroundCommon('SUN', 'SUN fixture'),
    regions: structuredClone(WALKAROUND_REGIONS.sun),
    sunTravelDirection: [0, 0, -1],
    primaryLightDir: [0, 0, 1],
    sunIntensity: intensity,
    receiverAlbedo: albedo,
    floorAlbedo: albedo,
    diffuseOnly: true,
    cosTheta: 1,
    analyticExpectedReceiverLum: analytic,
    analyticExpectedFloorLum: analytic,
    rendered: {
      receiverLum,
      sideDiagnosticLum: analytic,
      floorLum: receiverLum,
      leftWallLum: analytic,
      overall: receiverLum,
    },
    receiverRatioToAnalytic: receiverRatio,
    floorRatioToAnalytic: receiverRatio,
    analyticAgreement: agreement,
    shadowAssertionAuthored: false,
    verdict: receiverLum > 0.01 && agreement ? 'PASS' : 'PASS-PARTIAL',
  };
}

function validGlassWalkaroundFixture({ glassCenter, noGlassCenter, glassOverall, noGlassOverall }) {
  const centreRatio = noGlassCenter > 0.01 ? glassCenter / noGlassCenter : 0;
  const overallRatio = noGlassOverall > 0.01 ? glassOverall / noGlassOverall : 0;
  const centreDelta = glassCenter - noGlassCenter;
  const overallDelta = glassOverall - noGlassOverall;
  const materialEffectObserved = Math.max(Math.abs(centreDelta), Math.abs(overallDelta)) >= 1e-4;
  const ratioWithinPromotionBounds = centreRatio <= 4 && overallRatio <= 8;
  const notBlack = glassCenter > 0.01;
  const ratioPass = centreRatio >= 0.5;
  const verdict =
    notBlack && ratioPass && materialEffectObserved && ratioWithinPromotionBounds
      ? 'PASS'
      : notBlack && ratioPass && materialEffectObserved
        ? 'FINDING'
        : notBlack
          ? 'SMOKE'
          : 'FAIL';
  return {
    ...walkaroundCommon('GLASS', 'GLASS fixture'),
    regions: structuredClone(WALKAROUND_REGIONS.glass),
    fresnelT_normal_incidence_n1p5: 0.92,
    expectedMinCentreRatio: 0.5,
    expectedMaxCentreRatio: 4,
    expectedMaxOverallRatio: 8,
    glass: { centreRegionLum: glassCenter, overall: glassOverall },
    noGlass: { centreRegionLum: noGlassCenter, overall: noGlassOverall },
    centreRatio,
    overallRatio,
    delta: { centreRegionLum: centreDelta, overall: overallDelta },
    minSignalDelta: 1e-4,
    materialEffectObserved,
    ratioWithinPromotionBounds,
    ...(verdict === 'FINDING'
      ? {
          promotion: {
            defaultReady: false,
            blocker: 'glass-transport-radiance-blowout',
            requiredEvidence: 'reference A/B',
          },
        }
      : {}),
    verdict,
  };
}

function validGlossyWalkaroundFixture({
  metalSample,
  diffuseSample,
  metalOverall,
  diffuseOverall,
}) {
  const sampleRatio = diffuseSample > 0.01 ? metalSample / diffuseSample : 0;
  const sampleDelta = metalSample - diffuseSample;
  const overallDelta = metalOverall - diffuseOverall;
  const materialEffectObserved = Math.max(Math.abs(sampleDelta), Math.abs(overallDelta)) >= 1e-4;
  const notBlack = metalSample > 1e-4 || metalOverall > 1e-3;
  const plausible = sampleRatio >= 0.8;
  const verdict =
    notBlack && plausible && materialEffectObserved
      ? 'PASS'
      : notBlack && materialEffectObserved
        ? 'FINDING'
        : notBlack
          ? 'PASS-WEAK'
          : 'FAIL';
  return {
    ...walkaroundCommon('GLOSSY', 'GLOSSY fixture'),
    regions: structuredClone(WALKAROUND_REGIONS.glossy),
    sampleRegion: 'visible-back-wall-center-crop',
    metal: {
      sampleRegionLum: metalSample,
      floorLum: metalSample,
      overall: metalOverall,
    },
    diffuse: {
      sampleRegionLum: diffuseSample,
      floorLum: diffuseSample,
      overall: diffuseOverall,
    },
    sampleRatio,
    floorRatio: sampleRatio,
    delta: {
      sampleRegionLum: sampleDelta,
      floorLum: sampleDelta,
      overall: overallDelta,
    },
    expectedMinSampleRatio: 0.8,
    expectedMinFloorRatio: 0.8,
    minSignalDelta: 1e-4,
    materialEffectObserved,
    promotion: {
      defaultReady: false,
      blocker: 'ddgi-irradiance-cache-not-ggx-filtered-radiance',
      requiredEvidence: 'material furnace A/B',
    },
    verdict,
  };
}

function validateWalkaroundFixture(id, row) {
  const payload = { [id]: row };
  const text = JSON.stringify(payload);
  const now = Date.now();
  return validateWalkaroundRunResult(payload, {
    expectedCaseIds: [id],
    beforeState: null,
    afterState: { mtimeMs: now, size: text.length, text },
    startedAtMs: now - 1,
  });
}

async function validBdptFixture() {
  const result = await readJson('tools/radiometric-ab/results-bdpt.json');
  result.schema = 'vitrum.radiometric-ab.result.v1';
  result.provenance.sourceManifest = validSourceManifestFixture(bdptProof);
  return result;
}

async function validRestirPtFixture() {
  const retainedMeasurements = await readJson('tools/radiometric-ab/results-restir-pt.json');
  const pairedSeedAnalysis = structuredClone(retainedMeasurements.pairedSeedAnalysis);
  pairedSeedAnalysis.globalEquivalent = true;
  pairedSeedAnalysis.roiEquivalent = true;
  return buildRestirPtResult({
    provenance: {
      ...retainedMeasurements.provenance,
      sourceManifest: validSourceManifestFixture(restirPtProof),
    },
    date: retainedMeasurements.date,
    resolution: restirPtProof.resolution,
    roi: restirPtProof.roi,
    meanFrames: restirPtProof.meanFrames,
    varianceRuns: restirPtProof.varianceRuns,
    varianceFramesPerRun: restirPtProof.varianceFramesPerRun,
    captureConfig: validRestirCaptureConfigFixture(restirPtProof),
    deviceIdentity: validRestirDeviceIdentityFixture(),
    base: retainedMeasurements.base,
    rpt: retainedMeasurements.rpt,
    globalRelErr: retainedMeasurements.globalRelErr,
    roiRelErr: retainedMeasurements.roiRelErr,
    varRatio: retainedMeasurements.varRatio,
    reservoirWeightStats: retainedMeasurements.reservoirWeightStats,
    pairedSeedAnalysis,
    highFrameMeanAgreement: retainedMeasurements.highFrameMeanAgreement,
    meanAgreement: retainedMeasurements.meanAgreement,
    varianceNotWorse: retainedMeasurements.varianceNotWorse,
    verdict: retainedMeasurements.verdict,
  });
}

const bdptProof = RADIOMETRIC_AB_PROOFS.find((proof) => proof.id === 'bdpt');
assert.ok(bdptProof);
const restirPtProof = RADIOMETRIC_AB_PROOFS.find((proof) => proof.id === 'restir-pt');
assert.ok(restirPtProof);

test('walkaround result validation accepts complete evidence for every non-failure verdict', () => {
  const fixtures = [
    ['a8', validA8WalkaroundFixture(0.001), false],
    ['a8', validA8WalkaroundFixture(0.01), true],
    ['a8', validA8WalkaroundFixture(0.04), true],
    ['sun', validSunWalkaroundFixture(1), false],
    ['sun', validSunWalkaroundFixture(2), true],
    [
      'glass',
      validGlassWalkaroundFixture({
        glassCenter: 0.12,
        noGlassCenter: 0.1,
        glassOverall: 0.12,
        noGlassOverall: 0.1,
      }),
      false,
    ],
    [
      'glass',
      validGlassWalkaroundFixture({
        glassCenter: 0.5,
        noGlassCenter: 0.1,
        glassOverall: 0.9,
        noGlassOverall: 0.1,
      }),
      true,
    ],
    [
      'glass',
      validGlassWalkaroundFixture({
        glassCenter: 0.04,
        noGlassCenter: 0.1,
        glassOverall: 0.04,
        noGlassOverall: 0.1,
      }),
      true,
    ],
    [
      'glossy',
      validGlossyWalkaroundFixture({
        metalSample: 0.1,
        diffuseSample: 0.05,
        metalOverall: 0.1,
        diffuseOverall: 0.05,
      }),
      false,
    ],
    [
      'glossy',
      validGlossyWalkaroundFixture({
        metalSample: 0.01,
        diffuseSample: 0.1,
        metalOverall: 0.01,
        diffuseOverall: 0.1,
      }),
      true,
    ],
    [
      'glossy',
      validGlossyWalkaroundFixture({
        metalSample: 0.01,
        diffuseSample: 0.01,
        metalOverall: 0.01,
        diffuseOverall: 0.01,
      }),
      true,
    ],
  ];
  for (const [id, row, partial] of fixtures) {
    const validation = validateWalkaroundFixture(id, row);
    assert.equal(validation.caseVerdicts[id], row.verdict);
    assert.equal(validation.partial, partial, `${id}:${row.verdict}`);
  }
});

test('serialized null evidence cannot support any recognized walkaround verdict', () => {
  const fixtures = [
    ['a8', validA8WalkaroundFixture(0.001)],
    ['a8', validA8WalkaroundFixture(0.01)],
    ['a8', validA8WalkaroundFixture(0.04)],
    ['a8', validA8WalkaroundFixture(0.07)],
    ['sun', validSunWalkaroundFixture(1)],
    ['sun', validSunWalkaroundFixture(2)],
    [
      'glass',
      validGlassWalkaroundFixture({
        glassCenter: 0.12,
        noGlassCenter: 0.1,
        glassOverall: 0.12,
        noGlassOverall: 0.1,
      }),
    ],
    [
      'glass',
      validGlassWalkaroundFixture({
        glassCenter: 0.5,
        noGlassCenter: 0.1,
        glassOverall: 0.9,
        noGlassOverall: 0.1,
      }),
    ],
    [
      'glass',
      validGlassWalkaroundFixture({
        glassCenter: 0.04,
        noGlassCenter: 0.1,
        glassOverall: 0.04,
        noGlassOverall: 0.1,
      }),
    ],
    [
      'glossy',
      validGlossyWalkaroundFixture({
        metalSample: 0.1,
        diffuseSample: 0.05,
        metalOverall: 0.1,
        diffuseOverall: 0.05,
      }),
    ],
    [
      'glossy',
      validGlossyWalkaroundFixture({
        metalSample: 0.01,
        diffuseSample: 0.1,
        metalOverall: 0.01,
        diffuseOverall: 0.1,
      }),
    ],
    [
      'glossy',
      validGlossyWalkaroundFixture({
        metalSample: 0.01,
        diffuseSample: 0.01,
        metalOverall: 0.01,
        diffuseOverall: 0.01,
      }),
    ],
  ];
  for (const [id, fixture] of fixtures) {
    fixture.renderTimeSec = Number.NaN;
    const serialized = JSON.parse(JSON.stringify(fixture));
    assert.equal(serialized.renderTimeSec, null);
    assert.throws(
      () => validateWalkaroundFixture(id, serialized),
      /renderTimeSec must be a number-typed finite value/,
      `${id}:${fixture.verdict}`,
    );
  }
});

test('walkaround case-specific numeric and boolean evidence fails closed', () => {
  const fixtures = [
    [
      'a8',
      validA8WalkaroundFixture(0.001),
      (row) => {
        row.biased.floor = null;
      },
    ],
    [
      'sun',
      validSunWalkaroundFixture(1),
      (row) => {
        row.rendered.overall = null;
      },
    ],
    [
      'glass',
      validGlassWalkaroundFixture({
        glassCenter: 0.12,
        noGlassCenter: 0.1,
        glassOverall: 0.12,
        noGlassOverall: 0.1,
      }),
      (row) => {
        row.centreRatio = '1.2';
      },
    ],
    [
      'glossy',
      validGlossyWalkaroundFixture({
        metalSample: 0.1,
        diffuseSample: 0.05,
        metalOverall: 0.1,
        diffuseOverall: 0.05,
      }),
      (row) => {
        row.materialEffectObserved = 'true';
      },
    ],
  ];
  for (const [id, row, mutate] of fixtures) {
    mutate(row);
    assert.throws(
      () => validateWalkaroundFixture(id, row),
      /must be a number-typed finite value|must be a boolean/,
      id,
    );
  }
});

test('walkaround result verdicts are re-derived from finite measurements', () => {
  const forged = [
    ['a8', validA8WalkaroundFixture(0.01), 'NEGLIGIBLE'],
    ['sun', validSunWalkaroundFixture(2), 'PASS'],
    [
      'glass',
      validGlassWalkaroundFixture({
        glassCenter: 0.5,
        noGlassCenter: 0.1,
        glassOverall: 0.9,
        noGlassOverall: 0.1,
      }),
      'PASS',
    ],
    [
      'glossy',
      validGlossyWalkaroundFixture({
        metalSample: 0.01,
        diffuseSample: 0.1,
        metalOverall: 0.01,
        diffuseOverall: 0.1,
      }),
      'PASS',
    ],
  ];
  for (const [id, row, forgedVerdict] of forged) {
    row.verdict = forgedVerdict;
    assert.throws(() => validateWalkaroundFixture(id, row), /verdict must be/, id);
  }
});

test('complete BDPT validator re-derives every bounded-depth regression condition', async () => {
  const validation = validateRadiometricResult(bdptProof, await validBdptFixture());
  assert.deepEqual(validation.metrics.supportedDepths, [1, 2, 3, 8]);
  assert.equal(validation.metrics.defaultMatchesDepth2, true);
  assert.equal(validation.metrics.endpointOnlyCarriesSignal, true);
  assert.equal(validation.metrics.endpointOnlyMeanAgreement, true);
});

test('minimal PASS-shaped radiometric JSON fails closed', () => {
  assert.throws(
    () =>
      validateRadiometricResult(bdptProof, {
        schema: bdptProof.schema,
        ab: bdptProof.ab,
        verdict: 'PASS',
      }),
    /resolution differs from proof|provenance must be an object/,
  );
});

test('non-finite measurements cannot support PASS', async () => {
  const result = await validBdptFixture();
  result.bdpt.variance = Number.NaN;
  assert.throws(() => validateRadiometricResult(bdptProof, result), /must be a finite number/);
});

test('semantic BDPT booleans cannot contradict their measurements', async () => {
  const result = await validBdptFixture();
  result.controls.endpointOnlyMeanAgreement = false;
  assert.throws(
    () => validateRadiometricResult(bdptProof, result),
    /endpointOnlyMeanAgreement does not match/,
  );
});

test('stale provenance fails even when every measurement passes', async () => {
  const result = await validBdptFixture();
  assert.throws(
    () =>
      validateRadiometricResult(bdptProof, result, {
        expectedProvenance: { ...result.provenance, scriptSha256: '0'.repeat(64) },
      }),
    /provenance differs from the current script\/helper\/source identity/,
  );
});

test('pre-manifest numerical baselines require the explicit historical mode', async () => {
  const result = await readJson('tools/radiometric-ab/results-bdpt.json');
  assert.equal(result.provenance.sourceManifest, undefined);
  assert.throws(
    () => validateRadiometricResult(bdptProof, result),
    /provenance\.sourceManifest is missing/,
  );
  const validation = validateRadiometricResult(bdptProof, result, {
    historicalBaseline: true,
  });
  assert.deepEqual(validation.metrics.supportedDepths, [1, 2, 3, 8]);
});

test('historical baseline mode cannot impersonate current-source provenance', async () => {
  const result = await validBdptFixture();
  assert.throws(
    () =>
      validateRadiometricResult(bdptProof, result, {
        historicalBaseline: true,
        expectedProvenance: result.provenance,
      }),
    /historicalBaseline cannot be combined with expectedProvenance/,
  );
});

test('specialty coverage is derived from the deterministic fixture', async () => {
  const result = await readJson('tools/radiometric-ab/results-restir-pt-specialty.json');
  const metrics = validateRestirPtSpecialtyResult(RESTIR_PT_SPECIALTY_PROOF, result);
  assert.equal(metrics.glossyFixtureCovered, true);

  const truncated = structuredClone(result);
  truncated.coverage.specialtyLobes.pop();
  assert.throws(
    () => validateRestirPtSpecialtyResult(RESTIR_PT_SPECIALTY_PROOF, truncated),
    /coverage differs from proof/,
  );
});

test('ReSTIR-PT producer and validator share the current estimator and arm schema', async () => {
  const result = await validRestirPtFixture();
  assert.equal(result.weightMode, RESTIR_PT_WEIGHT_MODE);
  assert.deepEqual(result.captureConfig.arms, {
    base: { oneEdgeReconnectionReuse: false },
    candidate: {
      oneEdgeReconnectionReuse: true,
      effectiveMClamp: restirPtProof.capture.effectiveMClamp,
    },
  });
  const validation = validateRadiometricResult(restirPtProof, await validRestirPtFixture());
  assert.equal(validation.metrics.weightMode, RESTIR_PT_WEIGHT_MODE);
  assert.ok(validation.metrics.diagnosticClampWeightMassFraction > 0.3);
  assert.equal(validation.metrics.pairedMeanEquivalent, true);
});

test('ReSTIR-PT validator rejects deprecated or relabelled capture arms', async () => {
  const result = await validRestirPtFixture();
  result.captureConfig.arms.base = { restirPtReuse: false };
  assert.throws(
    () => validateRadiometricResult(restirPtProof, result),
    /captureConfig or exact frame-seed schedule differs from proof/,
  );
});

test('ReSTIR-PT validator re-derives clipped weight mass', async () => {
  const result = await validRestirPtFixture();
  result.reservoirWeightStats.clippedWeightMassFraction = 0;
  assert.throws(
    () => validateRadiometricResult(restirPtProof, result),
    /clippedWeightMassFraction .* differs from derived value/,
  );
});

test('ReSTIR-PT validator re-derives the paired-seed confidence interval', async () => {
  const result = await validRestirPtFixture();
  result.pairedSeedAnalysis.global.lower = -0.5;
  assert.throws(
    () => validateRadiometricResult(restirPtProof, result),
    /pairedSeedAnalysis.global.lower .* differs from derived value/,
  );
});

test('ReSTIR-PT validator rejects an obsolete estimator label', async () => {
  const result = await validRestirPtFixture();
  result.weightMode = 'effectively-unclamped-f32-max';
  assert.throws(
    () => validateRadiometricResult(restirPtProof, result),
    /weightMode must identify the shared-max-log estimator/,
  );
});
