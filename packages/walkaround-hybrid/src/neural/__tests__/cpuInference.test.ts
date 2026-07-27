import { describe, expect, it } from 'vitest';

import { executeNeuralInferenceCpu } from '../cpuInference.js';
import { NEURAL_PREPROCESSING_CONTRACT } from '../preprocessing.js';
import {
  buildUNetSpec,
  deriveParamCount,
  type UNetSpec,
} from '../unetArchitecture.js';
import {
  NEURAL_F16_VALIDATION_BOUNDS,
  type ModelWeights,
} from '../weights.js';

function productionCheckpoint() {
  return {
    id: 'cpu-oracle',
    trainingSamples: 500,
    noisySpp: 1,
    cleanSpp: 4096,
    auxiliaryInputs: ['albedo', 'normal'] as const,
    captureSource: 'unit',
    captureBackend: 'cpu',
    tonemap: 'linear-hdr',
    hardware: 'unit',
    preprocessing: NEURAL_PREPROCESSING_CONTRACT,
    qualityReport: { status: 'pass' as const, reportPath: 'unit.json' },
  };
}

function projectionSpec(): UNetSpec {
  const layers = [
    {
      name: 'pack',
      kind: 'inputPack',
      inputs: ['noisyColor', 'albedo', 'normals'],
      output: 'enc_input',
      params: { inC: 9, outC: 9 },
      weightLayout: 'none',
    },
    {
      name: 'proj',
      kind: 'conv2d',
      inputs: ['enc_input'],
      output: 'denoised',
      params: { inC: 9, outC: 3, kH: 1, kW: 1, stride: 1, padding: 0 },
      weightLayout: 'OIKW',
    },
  ] as const;
  return {
    name: 'projection-oracle',
    inputChannels: 9,
    outputChannels: 3,
    layers,
    paramCount: deriveParamCount(layers),
  };
}

function projectionWeights(): ModelWeights {
  const weights = new Float32Array(27);
  weights[0] = 1;
  weights[10] = 1;
  weights[20] = 1;
  return {
    formatVersion: 2,
    checkpoint: productionCheckpoint(),
    layers: [{
      name: 'proj',
      weights,
      biases: new Float32Array(3),
    }],
  };
}

describe('executeNeuralInferenceCpu', () => {
  it('mirrors preprocessing, OIKW projection, and inverse output scaling', () => {
    const result = executeNeuralInferenceCpu(
      projectionSpec(),
      projectionWeights(),
      2,
      1,
      {
        noisyColor: new Float32Array([16, 32, 64, Number.NaN, 100, -1]),
        albedo: new Float32Array([2, -1, Number.NaN, 0.1, 0.2, 0.3]),
        normals: new Float32Array([0, 0, 0, Number.NaN, 1, 0]),
      },
    );

    expect(Array.from(result.modelOutput)).toEqual([1, 2, 4, 0, 4, 0]);
    expect(Array.from(result.denoised)).toEqual([16, 32, 64, 0, 64, 0]);
  });

  it('executes the full canonical graph at its minimum shape with finite output', () => {
    const spec = buildUNetSpec();
    const layers = spec.layers
      .filter(layer => layer.kind === 'conv2d' || layer.kind === 'transposedConv2d')
      .map(layer => ({
        name: layer.name,
        weights: new Float32Array(
          layer.params.inC *
          layer.params.outC *
          (layer.params.kH ?? 1) *
          (layer.params.kW ?? 1),
        ),
        biases: new Float32Array(layer.params.outC),
      }));
    const weights: ModelWeights = {
      formatVersion: 2,
      checkpoint: productionCheckpoint(),
      layers,
    };
    const rgb = 8 * 8 * 3;
    const result = executeNeuralInferenceCpu(spec, weights, 8, 8, {
      noisyColor: new Float32Array(rgb).fill(8),
      albedo: new Float32Array(rgb).fill(0.5),
      normals: new Float32Array(rgb).fill(0),
    });

    expect(result.modelOutput).toHaveLength(rgb);
    expect(result.denoised).toHaveLength(rgb);
    expect(result.modelOutput.every(Number.isFinite)).toBe(true);
    expect(result.denoised.every(Number.isFinite)).toBe(true);
    expect(result.denoised.every(value => value === 0)).toBe(true);
  });

  it('runs all 25 logical layers with deterministic binary16 rounding inside certified HDR bounds', () => {
    const spec = buildUNetSpec();
    const layers = spec.layers
      .filter(layer => layer.kind === 'conv2d' || layer.kind === 'transposedConv2d')
      .map((layer, layerIndex) => {
        const { inC, outC, kH = 1, kW = 1 } = layer.params;
        return {
          name: layer.name,
          weights: Float32Array.from(
            { length: inC * outC * kH * kW },
            (_, index) => (((index * 13 + layerIndex * 7) % 19) - 9) * 0.0001,
          ),
          biases: Float32Array.from(
            { length: outC },
            (_, channel) => (((channel * 5 + layerIndex * 3) % 11) - 5) * 0.0002,
          ),
        };
      });
    const weights: ModelWeights = {
      formatVersion: 2,
      checkpoint: productionCheckpoint(),
      layers,
    };
    const rgbLength = 8 * 8 * 3;
    const inputs = {
      noisyColor: Float32Array.from(
        { length: rgbLength },
        (_, index) => ((index * 7) % 29) * 0.5,
      ),
      albedo: Float32Array.from(
        { length: rgbLength },
        (_, index) => (index % 5) / 4,
      ),
      normals: Float32Array.from(
        { length: rgbLength },
        (_, index) => (index % 3 === 1 ? 0.5 : 0.25),
      ),
    };

    expect(spec.layers).toHaveLength(25);
    const f32 = executeNeuralInferenceCpu(spec, weights, 8, 8, inputs, 'f32');
    const f16 = executeNeuralInferenceCpu(spec, weights, 8, 8, inputs, 'f16');
    const f16Repeat = executeNeuralInferenceCpu(spec, weights, 8, 8, inputs, 'f16');
    expect(Array.from(f16Repeat.modelOutput)).toEqual(Array.from(f16.modelOutput));
    expect(Array.from(f16Repeat.denoised)).toEqual(Array.from(f16.denoised));

    let maxAbsError = 0;
    let totalAbsError = 0;
    for (let i = 0; i < f16.denoised.length; i++) {
      const value = f16.denoised[i]!;
      expect(Number.isFinite(value), `denoised element ${i}`).toBe(true);
      expect(value).toBeGreaterThanOrEqual(0);
      expect(value).toBeLessThanOrEqual(NEURAL_PREPROCESSING_CONTRACT.radianceClamp);
      const error = Math.abs(value - f32.denoised[i]!);
      maxAbsError = Math.max(maxAbsError, error);
      totalAbsError += error;
    }
    expect(maxAbsError).toBeLessThanOrEqual(NEURAL_F16_VALIDATION_BOUNDS.maxAbsError);
    expect(totalAbsError / f16.denoised.length)
      .toBeLessThanOrEqual(NEURAL_F16_VALIDATION_BOUNDS.meanAbsError);
  });


  it('rejects malformed input lengths before executing layers', () => {
    expect(() => executeNeuralInferenceCpu(
      projectionSpec(),
      projectionWeights(),
      1,
      1,
      {
        noisyColor: new Float32Array(2),
        albedo: new Float32Array(3),
        normals: new Float32Array(3),
      },
    )).toThrow(/noisyColor length 2 != 3/);
  });
});
