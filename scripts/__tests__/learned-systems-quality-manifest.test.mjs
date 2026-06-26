import assert from 'node:assert/strict';
import test from 'node:test';

import {
  MIN_PRODUCTION_NEURAL_CLEAN_REFERENCE_SPP,
  MIN_PRODUCTION_NEURAL_SAMPLE_COUNT,
  PRODUCTION_NEURAL_DATASET_MANIFEST_SCHEMA,
  validateProductionDatasetManifest,
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
    artifacts: {
      datasetManifestPath: 'tools/neural-denoiser-training/datasets/fixture-production-ab/manifest.json',
      resultSummaryPath: 'tools/neural-denoiser-training/quality/fixture-production-ab/results.json',
      candidateOutputsPath: 'tools/neural-denoiser-training/quality/fixture-production-ab/candidate/',
      referenceOutputsPath: 'tools/neural-denoiser-training/quality/fixture-production-ab/reference/',
    },
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

function validDatasetManifest(qualityManifest = validQualityManifest()) {
  return {
    schema: PRODUCTION_NEURAL_DATASET_MANIFEST_SCHEMA,
    id: qualityManifest.dataset.id,
    sceneCount: qualityManifest.dataset.sceneCount,
    sampleCount: qualityManifest.dataset.sampleCount,
    noisySpp: qualityManifest.dataset.noisySpp,
    cleanReferenceSpp: qualityManifest.dataset.cleanReferenceSpp,
    includesAlbedo: qualityManifest.dataset.includesAlbedo,
    includesNormals: qualityManifest.dataset.includesNormals,
    captureSource: qualityManifest.dataset.captureSource,
    tonemap: qualityManifest.dataset.tonemap,
    scenes: [
      {
        id: 'cornell-box',
        sampleCount: 125,
        noisyPath: 'datasets/fixture-production-ab/cornell-box/noisy/',
        cleanPath: 'datasets/fixture-production-ab/cornell-box/clean/',
        albedoPath: 'datasets/fixture-production-ab/cornell-box/noisy/*_albedo.png',
        normalPath: 'datasets/fixture-production-ab/cornell-box/noisy/*_normal.png',
      },
      {
        id: 'multi-material',
        sampleCount: 125,
        noisyPath: 'datasets/fixture-production-ab/multi-material/noisy/',
        cleanPath: 'datasets/fixture-production-ab/multi-material/clean/',
        albedoPath: 'datasets/fixture-production-ab/multi-material/noisy/*_albedo.png',
        normalPath: 'datasets/fixture-production-ab/multi-material/noisy/*_normal.png',
      },
      {
        id: 'glass-emitter',
        sampleCount: 125,
        noisyPath: 'datasets/fixture-production-ab/glass-emitter/noisy/',
        cleanPath: 'datasets/fixture-production-ab/glass-emitter/clean/',
        albedoPath: 'datasets/fixture-production-ab/glass-emitter/noisy/*_albedo.png',
        normalPath: 'datasets/fixture-production-ab/glass-emitter/noisy/*_normal.png',
      },
      {
        id: 'foliage-cards',
        sampleCount: 125,
        noisyPath: 'datasets/fixture-production-ab/foliage-cards/noisy/',
        cleanPath: 'datasets/fixture-production-ab/foliage-cards/clean/',
        albedoPath: 'datasets/fixture-production-ab/foliage-cards/noisy/*_albedo.png',
        normalPath: 'datasets/fixture-production-ab/foliage-cards/noisy/*_normal.png',
      },
    ],
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

function validateWithDatasetManifest(qualityManifest, datasetManifest) {
  validateProductionQualityManifest({
    qualityManifest,
    productionEntries: [PRODUCTION_ENTRY],
    productionCheckpoint: PRODUCTION_ENTRY.name,
    productionLike: [PRODUCTION_ENTRY.name],
    expectedParamCount: EXPECTED_PARAM_COUNT,
    artifactExists: () => true,
    artifactText: (artifactPath) => {
      assert.equal(artifactPath, qualityManifest.artifacts.datasetManifestPath);
      return JSON.stringify(datasetManifest);
    },
  });
}

function validateWithArtifactSet(qualityManifest, existingPaths) {
  validateProductionQualityManifest({
    qualityManifest,
    productionEntries: [PRODUCTION_ENTRY],
    productionCheckpoint: PRODUCTION_ENTRY.name,
    productionLike: [PRODUCTION_ENTRY.name],
    expectedParamCount: EXPECTED_PARAM_COUNT,
    artifactExists: (artifactPath) => existingPaths.has(artifactPath),
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

test('production neural quality manifest rejects missing reproducibility artifact paths', () => {
  const manifest = cloneManifest();
  delete manifest.artifacts;
  assert.throws(
    () => validate(manifest),
    /must include reproducibility artifact paths/,
  );

  const manifestWithoutReferenceOutputs = cloneManifest();
  manifestWithoutReferenceOutputs.artifacts.referenceOutputsPath = '';
  assert.throws(
    () => validate(manifestWithoutReferenceOutputs),
    /artifacts\.referenceOutputsPath must be a non-empty string/,
  );
});

test('production neural quality manifest rejects nonexistent reproducibility artifacts when an existence checker is supplied', () => {
  const manifest = cloneManifest();
  const existingPaths = new Set([
    manifest.artifacts.datasetManifestPath,
    manifest.artifacts.resultSummaryPath,
    manifest.artifacts.candidateOutputsPath,
  ]);

  assert.throws(
    () => validateWithArtifactSet(manifest, existingPaths),
    /artifacts\.referenceOutputsPath must point at an existing artifact/,
  );

  existingPaths.add(manifest.artifacts.referenceOutputsPath);
  assert.doesNotThrow(() => validateWithArtifactSet(manifest, existingPaths));
});

test('production neural quality manifest validates dataset manifest artifact contents when available', () => {
  const manifest = validQualityManifest();
  assert.doesNotThrow(() => validateWithDatasetManifest(manifest, validDatasetManifest(manifest)));
});

test('production neural quality manifest rejects missing per-scene dataset artifacts', () => {
  const manifest = validQualityManifest();
  const datasetManifest = validDatasetManifest(manifest);
  const existingPaths = new Set([
    ...Object.values(manifest.artifacts),
    ...datasetManifest.scenes.flatMap((scene) => [
      scene.noisyPath,
      scene.cleanPath,
      scene.albedoPath,
      scene.normalPath,
    ]),
  ]);
  existingPaths.delete(datasetManifest.scenes[0].normalPath);

  assert.throws(
    () => validateProductionQualityManifest({
      qualityManifest: manifest,
      productionEntries: [PRODUCTION_ENTRY],
      productionCheckpoint: PRODUCTION_ENTRY.name,
      productionLike: [PRODUCTION_ENTRY.name],
      expectedParamCount: EXPECTED_PARAM_COUNT,
      artifactExists: (artifactPath) => existingPaths.has(artifactPath),
      artifactText: (artifactPath) => {
        assert.equal(artifactPath, manifest.artifacts.datasetManifestPath);
        return JSON.stringify(datasetManifest);
      },
    }),
    /scene cornell-box normalPath must point at an existing artifact/,
  );

  existingPaths.add(datasetManifest.scenes[0].normalPath);
  assert.doesNotThrow(() => validateProductionQualityManifest({
    qualityManifest: manifest,
    productionEntries: [PRODUCTION_ENTRY],
    productionCheckpoint: PRODUCTION_ENTRY.name,
    productionLike: [PRODUCTION_ENTRY.name],
    expectedParamCount: EXPECTED_PARAM_COUNT,
    artifactExists: (artifactPath) => existingPaths.has(artifactPath),
    artifactText: () => JSON.stringify(datasetManifest),
  }));
});

test('production neural quality manifest rejects invalid dataset manifest JSON', () => {
  const manifest = validQualityManifest();
  assert.throws(
    () => validateProductionQualityManifest({
      qualityManifest: manifest,
      productionEntries: [PRODUCTION_ENTRY],
      productionCheckpoint: PRODUCTION_ENTRY.name,
      productionLike: [PRODUCTION_ENTRY.name],
      expectedParamCount: EXPECTED_PARAM_COUNT,
      artifactExists: () => true,
      artifactText: () => '{not-json',
    }),
    /must point at valid dataset manifest JSON/,
  );
});

test('production neural dataset manifest rejects quality-manifest mismatches', () => {
  const qualityManifest = validQualityManifest();
  const datasetManifest = validDatasetManifest(qualityManifest);
  datasetManifest.sampleCount = qualityManifest.dataset.sampleCount + 1;
  assert.throws(
    () => validateWithDatasetManifest(qualityManifest, datasetManifest),
    /dataset manifest sampleCount must match qualityManifest\.dataset\.sampleCount/,
  );
});

test('production neural dataset manifest rejects incomplete per-scene artifact records', () => {
  const datasetManifest = validDatasetManifest();
  delete datasetManifest.scenes[0].normalPath;
  assert.throws(
    () => validateProductionDatasetManifest(datasetManifest, validQualityManifest().dataset),
    /normalPath must be a non-empty string/,
  );
});

test('production neural dataset manifest rejects inconsistent scene sample totals', () => {
  const qualityManifest = validQualityManifest();
  const datasetManifest = validDatasetManifest(qualityManifest);
  datasetManifest.scenes[0].sampleCount -= 1;
  assert.throws(
    () => validateWithDatasetManifest(qualityManifest, datasetManifest),
    /scene sampleCount total 499 must equal qualityManifest\.dataset\.sampleCount 500/,
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
