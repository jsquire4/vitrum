import { describe, expect, it } from 'vitest';

import { neuralCheckpointPayloadSha256 } from '../checkpointDigest.js';
import { NEURAL_PREPROCESSING_CONTRACT } from '../preprocessing.js';
import { sha256Hex } from '../sha256.js';
import {
  loadWeightsFromArrayBuffer,
  serializeWeightsToArrayBuffer,
  type ModelWeights,
  type NeuralCheckpointMetadata,
} from '../weights.js';

function certifiedWeights(): ModelWeights {
  const layers = [
    {
      name: 'conv_a',
      weights: new Float32Array([0.25, -0.5]),
      biases: new Float32Array([0.125]),
    },
    {
      name: 'conv_b',
      weights: new Float32Array([1, 2, 3]),
      biases: new Float32Array([-0.25]),
    },
  ];
  const checkpoint: NeuralCheckpointMetadata = {
    id: 'certificate-test',
    trainingSamples: 500,
    noisySpp: 1,
    cleanSpp: 4096,
    auxiliaryInputs: ['albedo', 'normal'],
    captureSource: 'unit',
    captureBackend: 'walkaround-hybrid',
    tonemap: 'linear-hdr',
    hardware: 'unit',
    preprocessing: NEURAL_PREPROCESSING_CONTRACT,
    tensorStorage: 'f16-compatible',
    mixedPrecision: {
      checkpointSha256: neuralCheckpointPayloadSha256(layers),
      architecture: 'vitrum-unet-9x3-v1',
      preprocessing: NEURAL_PREPROCESSING_CONTRACT,
      quantization: 'f16-storage-per-logical-layer-f32-weight-bias-accumulation',
      metricDomain: 'postprocessed-linear-hdr',
      validationCorpusSha256: 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
      status: 'pass',
      validationScenes: 8,
      maxAbsError: 0.01,
      meanAbsError: 0.001,
      psnrDb: 48,
      finiteOutputs: true,
      outputMin: 0,
      outputMax: 64,
      accumulation: 'f32',
      weights: 'f32',
    },
    qualityReport: { status: 'pass', reportPath: 'quality.json' },
  };
  return { layers, formatVersion: 2, checkpoint };
}

describe('neural mixed-precision certificate binding', () => {
  it('implements standard SHA-256', () => {
    expect(sha256Hex(new TextEncoder().encode('abc')))
      .toBe('ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad');
  });

  it('round-trips a certificate bound to exact ordered tensors', () => {
    const weights = certifiedWeights();
    const loaded = loadWeightsFromArrayBuffer(serializeWeightsToArrayBuffer(weights));
    expect(loaded.checkpoint).toEqual(weights.checkpoint);
  });

  it('rejects reordered tensors and a stale certificate before serialization', () => {
    const weights = certifiedWeights();
    expect(() => serializeWeightsToArrayBuffer({
      ...weights,
      layers: [...weights.layers].reverse(),
    })).toThrow(/checkpointSha256 does not match the ordered tensor payload/);

    const stale = certifiedWeights();
    stale.layers[0]!.weights[0] = 0.5;
    expect(() => serializeWeightsToArrayBuffer(stale))
      .toThrow(/checkpointSha256 does not match the ordered tensor payload/);
  });

  it('rejects tensor tampering when loading a previously certified binary', () => {
    const binary = serializeWeightsToArrayBuffer(certifiedWeights());
    const tampered = binary.slice(0);
    const view = new DataView(tampered);
    const metadataLength = view.getUint32(12, true);
    const firstRecord = 16 + metadataLength;
    const nameLength = view.getUint32(firstRecord, true);
    const firstWeight = firstRecord + 4 + nameLength + 4;
    view.setFloat32(firstWeight, 0.5, true);
    expect(() => loadWeightsFromArrayBuffer(tampered))
      .toThrow(/checkpointSha256 does not match the ordered tensor payload/);
  });
});
