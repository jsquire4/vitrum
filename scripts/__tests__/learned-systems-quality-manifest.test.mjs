import assert from 'node:assert/strict';
import test from 'node:test';

import {
  MIN_PRODUCTION_NEURAL_CLEAN_REFERENCE_SPP,
  MIN_PRODUCTION_NEURAL_SAMPLE_COUNT,
  validateProductionQualityManifest,
} from '../../tools/learned-systems/qualityManifestValidator.mjs';

const EXPECTED_PARAM_COUNT = 42;
const PRODUCTION_ENTRY = Object.freeze({
  name: 'prod-v1.vitrum-model',
  role: 'production',
  productionDefaultEligible: true,
  sizeBytes: 1024,
  sha256: 'abc123',
  paramCount: EXPECTED_PARAM_COUNT,
});

function validQualityManifest() {
  return {
    checkpoint: {
      name: PRODUCTION_ENTRY.name,
      sha256: PRODUCTION_ENTRY.sha256,
      sizeBytes: PRODUCTION_ENTRY.sizeBytes,
      paramCount: PRODUCTION_ENTRY.paramCount,
    },
    metrics: {
      psnrDb: 34.5,
      ssim: 0.96,
      meanAbs: 0.012,
      rmse: 0.018,
    },
    thresholds: {
      psnrDb: 32,
      ssim: 0.94,
      meanAbs: 0.02,
      rmse: 0.025,
    },
    hardware: 'fixture-browser-webgpu',
    generatedAt: '2026-06-22T00:00:00.000Z',
    dataset: {
      id: 'fixture-production-ab',
      sceneCount: 4,
      sampleCount: MIN_PRODUCTION_NEURAL_SAMPLE_COUNT,
      noisySpp: 1,
      cleanReferenceSpp: MIN_PRODUCTION_NEURAL_CLEAN_REFERENCE_SPP,
      includesAlbedo: true,
      includesNormals: true,
      captureSource: 'browser-webgpu-batched-capture',
      tonemap: 'reinhard',
    },
    comparison: {
      baseline: 'pt-webgpu-4096spp',
      candidate: 'walkaround-neural-production-checkpoint',
    },
  };
}

function validate(qualityManifest) {
  validateProductionQualityManifest({
    qualityManifest,
    productionEntries: [PRODUCTION_ENTRY],
    productionCheckpoint: PRODUCTION_ENTRY.name,
    productionLike: [PRODUCTION_ENTRY.name],
    expectedParamCount: EXPECTED_PARAM_COUNT,
  });
}

function cloneManifest() {
  return structuredClone(validQualityManifest());
}

test('production neural quality manifest fixture passes when identity, dataset, hardware, and thresholds are complete', () => {
  assert.doesNotThrow(() => validate(validQualityManifest()));
});

test('production neural quality manifest rejects mismatched checkpoint identity', () => {
  const manifest = cloneManifest();
  manifest.checkpoint.sha256 = 'wrong';
  assert.throws(
    () => validate(manifest),
    /checkpoint\.sha256 must be abc123/,
  );
});

test('production neural quality manifest rejects missing hardware metadata', () => {
  const manifest = cloneManifest();
  delete manifest.hardware;
  assert.throws(
    () => validate(manifest),
    /must name the validation hardware/,
  );
});

test('production neural quality manifest rejects missing finite metrics', () => {
  const manifest = cloneManifest();
  manifest.metrics = { psnrDb: Number.NaN };
  assert.throws(
    () => validate(manifest),
    /must include at least one finite numeric quality bound/,
  );
});

test('production neural quality manifest rejects incomplete dataset metadata', () => {
  const manifest = cloneManifest();
  delete manifest.dataset.sampleCount;
  assert.throws(
    () => validate(manifest),
    /dataset\.sampleCount must be >= 500/,
  );
});

test('production neural quality manifest rejects tiny dataset samples', () => {
  const manifest = cloneManifest();
  manifest.dataset.sampleCount = MIN_PRODUCTION_NEURAL_SAMPLE_COUNT - 1;
  assert.throws(
    () => validate(manifest),
    /dataset\.sampleCount must be >= 500/,
  );
});

test('production neural quality manifest rejects non-production reference spp', () => {
  const manifest = cloneManifest();
  manifest.dataset.cleanReferenceSpp = MIN_PRODUCTION_NEURAL_CLEAN_REFERENCE_SPP - 1;
  assert.throws(
    () => validate(manifest),
    /dataset\.cleanReferenceSpp must be >= 4096/,
  );
});

test('production neural quality manifest rejects missing auxiliary buffers', () => {
  const manifest = cloneManifest();
  manifest.dataset.includesNormals = false;
  assert.throws(
    () => validate(manifest),
    /dataset\.includesNormals must be true/,
  );
});

test('production neural quality manifest rejects missing capture source and tonemap', () => {
  const manifest = cloneManifest();
  manifest.dataset.captureSource = '';
  assert.throws(
    () => validate(manifest),
    /dataset\.captureSource must be a non-empty string/,
  );

  const manifestWithoutTonemap = cloneManifest();
  delete manifestWithoutTonemap.dataset.tonemap;
  assert.throws(
    () => validate(manifestWithoutTonemap),
    /dataset\.tonemap must be a non-empty string/,
  );
});

test('production neural quality manifest rejects failed higher-is-better thresholds', () => {
  const manifest = cloneManifest();
  manifest.metrics.psnrDb = 28;
  assert.throws(
    () => validate(manifest),
    /metric psnrDb=28 does not satisfy threshold 32/,
  );
});

test('production neural quality manifest rejects failed lower-is-better thresholds', () => {
  const manifest = cloneManifest();
  manifest.metrics.rmse = 0.05;
  assert.throws(
    () => validate(manifest),
    /metric rmse=0.05 does not satisfy threshold 0.025/,
  );
});
