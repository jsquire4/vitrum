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
      meanFrameSeeds: Array.from(
        { length: proof.meanFrames },
        (_, frame) => u32(BigInt(frame) * multiplier + increment),
      ),
      varianceRunFrameSeeds: Array.from(
        { length: proof.varianceRuns },
        (_, run) => Array.from(
          { length: proof.varianceFramesPerRun },
          (_, frame) => {
            const globalFrame = BigInt(run * proof.varianceFramesPerRun + frame);
            return u32(globalFrame * multiplier + increment + BigInt(run) * runStride);
          },
        ),
      ),
    },
  };
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

async function validBdptFixture() {
  const result = await readJson('tools/radiometric-ab/results-bdpt.json');
  result.schema = 'vitrum.radiometric-ab.result.v1';
  result.provenance.sourceManifest = validSourceManifestFixture(bdptProof);
  return result;
}

async function validRestirPtFixture() {
  const result = await readJson('tools/radiometric-ab/results-restir-pt.json');
  result.provenance.sourceManifest = validSourceManifestFixture(restirPtProof);
  result.captureConfig = validRestirCaptureConfigFixture(restirPtProof);
  result.deviceIdentity = validRestirDeviceIdentityFixture();
  result.clampControls = result.clampControls.map((entry) => ({
    ...entry,
    effectivePackedWCap: Math.fround(entry.wCap),
    promotionEligible: false,
    estimatorMode: 'intentionally-biased-finite-weight-clamp',
  }));
  result.pairedSeedAnalysis.globalEquivalent = true;
  result.pairedSeedAnalysis.roiEquivalent = true;
  return result;
}

const bdptProof = RADIOMETRIC_AB_PROOFS.find((proof) => proof.id === 'bdpt');
assert.ok(bdptProof);
const restirPtProof = RADIOMETRIC_AB_PROOFS.find((proof) => proof.id === 'restir-pt');
assert.ok(restirPtProof);

test('complete BDPT validator re-derives every bounded-depth regression condition', async () => {
  const validation = validateRadiometricResult(bdptProof, await validBdptFixture());
  assert.deepEqual(validation.metrics.supportedDepths, [1, 2, 3, 8]);
  assert.equal(validation.metrics.defaultMatchesDepth2, true);
  assert.equal(validation.metrics.endpointOnlyCarriesSignal, true);
  assert.equal(validation.metrics.endpointOnlyMeanAgreement, true);
});

test('minimal PASS-shaped radiometric JSON fails closed', () => {
  assert.throws(
    () => validateRadiometricResult(bdptProof, {
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
  assert.throws(
    () => validateRadiometricResult(bdptProof, result),
    /must be a finite number/,
  );
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
    () => validateRadiometricResult(bdptProof, result, {
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
    () => validateRadiometricResult(bdptProof, result, {
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

test('ReSTIR-PT validator proves an unclipped professional default and a biased legacy control', async () => {
  const validation = validateRadiometricResult(restirPtProof, await validRestirPtFixture());
  assert.equal(validation.metrics.professionalDefaultClippedWeightMass, 0);
  assert.ok(validation.metrics.legacyClampWeightMassFraction > 0.3);
  assert.equal(validation.metrics.pairedMeanEquivalent, true);
});

test('ReSTIR-PT validator rejects missing or relabelled clamp controls', async () => {
  const result = await validRestirPtFixture();
  result.clampControls[0].wCap = 11;
  assert.throws(
    () => validateRadiometricResult(restirPtProof, result),
    /clampControls caps are missing, duplicated, or out of order/,
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

test('ReSTIR-PT validator rejects saturation at the professional default ceiling', async () => {
  const result = await validRestirPtFixture();
  result.reservoirWeightStats.weight.max = restirPtProof.professionalDefaultWeightCeiling;
  assert.throws(
    () => validateRadiometricResult(restirPtProof, result),
    /professional default clipped or saturated resolved reservoir weights/,
  );
});
