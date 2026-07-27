import { describe, expect, it } from 'vitest';

import { executeNeuralInferenceCpu } from '../cpuInference.js';
import { packLayerUniform, preflightTensorDims } from '../tensorDimSolver.js';
import { deriveParamCount, type UNetSpec } from '../unetArchitecture.js';
import type { ModelWeights } from '../weights.js';
import { BILINEAR_UPSAMPLE_WGSL } from '../wgsl/bilinearUpsample.wgsl.js';
import { TRANSPOSED_CONV2D_WGSL } from '../wgsl/transposedConv2d.wgsl.js';

function bilinearSpec(): UNetSpec {
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
      name: 'down',
      kind: 'conv2d',
      inputs: ['enc_input'],
      output: 'downsampled',
      params: { inC: 9, outC: 9, kH: 1, kW: 1, stride: 2, padding: 0 },
      weightLayout: 'OIKW',
    },
    {
      name: 'upsample',
      kind: 'bilinearUpsample',
      inputs: ['downsampled'],
      output: 'upsampled',
      params: { inC: 9, outC: 9 },
      weightLayout: 'none',
    },
    {
      name: 'proj',
      kind: 'conv2d',
      inputs: ['upsampled'],
      output: 'denoised',
      params: { inC: 9, outC: 3, kH: 1, kW: 1, stride: 1, padding: 0 },
      weightLayout: 'OIKW',
    },
  ] as const;
  return {
    name: 'bilinear-edge-oracle',
    inputChannels: 9,
    outputChannels: 3,
    layers,
    paramCount: deriveParamCount(layers),
  };
}

function bilinearWeights(): ModelWeights {
  const down = new Float32Array(9 * 9);
  for (let channel = 0; channel < 9; channel++) {
    down[channel * 9 + channel] = 1;
  }
  const projection = new Float32Array(3 * 9);
  projection[0] = 1;
  projection[10] = 1;
  projection[20] = 1;
  return {
    layers: [
      { name: 'down', weights: down, biases: new Float32Array(9) },
      { name: 'proj', weights: projection, biases: new Float32Array(3) },
    ],
  };
}

function transposedConvSpec(
  outputPadding = 1,
  dilation = 2,
): UNetSpec {
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
      name: 'down',
      kind: 'conv2d',
      inputs: ['enc_input'],
      output: 'latent',
      params: { inC: 9, outC: 1, kH: 1, kW: 1, stride: 2, padding: 0 },
      weightLayout: 'OIKW',
    },
    {
      name: 'up',
      kind: 'transposedConv2d',
      inputs: ['latent'],
      output: 'denoised',
      params: {
        inC: 1,
        outC: 3,
        kH: 2,
        kW: 2,
        stride: 2,
        padding: 1,
        dilation,
        outputPadding,
      },
      weightLayout: 'IOKW',
    },
  ] as const;
  return {
    name: 'transposed-conv-extension-oracle',
    inputChannels: 9,
    outputChannels: 3,
    layers,
    paramCount: deriveParamCount(layers),
  };
}

const TCONV_WEIGHTS = Float32Array.from([
  // IOKW: one input channel, then one 2x2 kernel per output channel.
  1, 2, 3, 4,
  0.5, 1, 1.5, 2,
  -1, 0.25, 0.5, -0.5,
]);
const TCONV_BIASES = Float32Array.from([0.5, 1, 1.5]);

function transposedConvWeights(): ModelWeights {
  const down = new Float32Array(9);
  down[0] = 1;
  return {
    layers: [
      { name: 'down', weights: down, biases: new Float32Array(1) },
      { name: 'up', weights: TCONV_WEIGHTS, biases: TCONV_BIASES },
    ],
  };
}

/**
 * Independent scatter oracle for ConvTranspose2d. Production CPU/WGSL execute
 * a gather by solving the inverse mapping, so this deliberately starts from
 * each input pixel and scatters every dilated kernel tap into the output.
 */
function scatterTransposedConvReference(input: readonly number[]): Float32Array {
  const inputH = 2;
  const inputW = 2;
  const outputH = 4;
  const outputW = 4;
  const outputC = 3;
  const stride = 2;
  const padding = 1;
  const dilation = 2;
  const output = new Float32Array(outputH * outputW * outputC);
  for (let pixel = 0; pixel < outputH * outputW; pixel++) {
    output.set(TCONV_BIASES, pixel * outputC);
  }
  for (let iy = 0; iy < inputH; iy++) {
    for (let ix = 0; ix < inputW; ix++) {
      const value = input[iy * inputW + ix]!;
      for (let kh = 0; kh < 2; kh++) {
        const oy = iy * stride - padding + kh * dilation;
        if (oy < 0 || oy >= outputH) continue;
        for (let kw = 0; kw < 2; kw++) {
          const ox = ix * stride - padding + kw * dilation;
          if (ox < 0 || ox >= outputW) continue;
          for (let oc = 0; oc < outputC; oc++) {
            const outputIndex = (oy * outputW + ox) * outputC + oc;
            const weightIndex = oc * 4 + kh * 2 + kw;
            output[outputIndex] = Math.fround(
              output[outputIndex]! +
              Math.fround(value * TCONV_WEIGHTS[weightIndex]!),
            );
          }
        }
      }
    }
  }
  return output;
}

describe('neural extension kernels', () => {
  it.each(['f32', 'f16'] as const)(
    'uses align_corners=false edge replication in the %s CPU oracle',
    precision => {
      const width = 4;
      const height = 4;
      const noisy = new Float32Array(width * height * 3);
      const setPixel = (x: number, y: number, value: number): void => {
        const base = (y * width + x) * 3;
        noisy[base] = value;
        noisy[base + 1] = value;
        noisy[base + 2] = value;
      };
      setPixel(0, 0, 1);
      setPixel(2, 0, 2);
      setPixel(0, 2, 3);
      setPixel(2, 2, 4);

      const result = executeNeuralInferenceCpu(
        bilinearSpec(),
        bilinearWeights(),
        width,
        height,
        {
          noisyColor: noisy,
          albedo: new Float32Array(noisy.length),
          normals: new Float32Array(noisy.length),
        },
        precision,
      );
      const expectedPixels = [
        // Analytic half-pixel coordinates with each of the four signed taps
        // independently edge-clamped. These constants are not produced by the
        // production CPU sampler.
        1, 1.25, 1.75, 2,
        1.5, 1.75, 2.25, 2.5,
        2.5, 2.75, 3.25, 3.5,
        3, 3.25, 3.75, 4,
      ];
      expect(Array.from(result.modelOutput)).toEqual(
        expectedPixels.flatMap(value => [value, value, value]),
      );
    },
  );

  it.each(['f32', 'f16'] as const)(
    'matches an independent scatter oracle for custom transposed-conv shape parameters in %s',
    precision => {
      const width = 4;
      const height = 4;
      const noisy = new Float32Array(width * height * 3);
      const sampled = [1, 2, 3, 4];
      for (let y = 0; y < 2; y++) {
        for (let x = 0; x < 2; x++) {
          noisy[((y * 2) * width + x * 2) * 3] = sampled[y * 2 + x]!;
        }
      }
      const spec = transposedConvSpec();
      const dims = preflightTensorDims(spec, width, height);
      expect(dims.get('latent')).toEqual({ H: 2, W: 2, C: 1 });
      expect(dims.get('denoised')).toEqual({ H: 4, W: 4, C: 3 });

      const result = executeNeuralInferenceCpu(
        spec,
        transposedConvWeights(),
        width,
        height,
        {
          noisyColor: noisy,
          albedo: new Float32Array(noisy.length),
          normals: new Float32Array(noisy.length),
        },
        precision,
      );
      expect(Array.from(result.modelOutput)).toEqual(
        Array.from(scatterTransposedConvReference(sampled)),
      );

      const up = spec.layers[2]!;
      const uniform = new Uint32Array(packLayerUniform(up, dims, height, width));
      expect(uniform).toHaveLength(12);
      expect(Array.from(uniform.slice(0, 10))).toEqual([
        2, 2, 1, 3, 2, 2, 2, 1, 2, 1,
      ]);
    },
  );

  it('rejects invalid or misplaced transposed-convolution shape parameters', () => {
    expect(() => preflightTensorDims(transposedConvSpec(2), 4, 4))
      .toThrow(/outputPadding < stride/);
    const invalidConv = bilinearSpec();
    const layers = invalidConv.layers.map(layer => layer.name === 'down'
      ? { ...layer, params: { ...layer.params, dilation: 2 } }
      : layer);
    const invalid: UNetSpec = { ...invalidConv, layers };
    expect(() => preflightTensorDims(invalid, 4, 4))
      .toThrow(/transposed-convolution-only parameters/);
  });

  it('keeps WGSL transposed-convolution dimensions and inverse mapping complete', () => {
    expect(TRANSPOSED_CONV2D_WGSL).toContain(
      '+ effectiveKH + params.outputPadding - 2u * params.padding',
    );
    expect(TRANSPOSED_CONV2D_WGSL).toContain('params.dilation * (params.kH - 1u)');
    expect(TRANSPOSED_CONV2D_WGSL).toContain('let oyPadded = oy + params.padding;');
    expect(TRANSPOSED_CONV2D_WGSL).toContain('let khOffset = kh * params.dilation;');
    expect(TRANSPOSED_CONV2D_WGSL).toContain('let oxPadded = ox + params.padding;');
    expect(TRANSPOSED_CONV2D_WGSL).toContain('if (ix_r % params.stride != 0u)');
  });

  it('keeps the bilinear WGSL footprint signed until per-sample edge clamping', () => {
    expect(BILINEAR_UPSAMPLE_WGSL).toContain('fn sampleInput(iy: i32, ix: i32');
    expect(BILINEAR_UPSAMPLE_WGSL).toContain('let iy0 = i32(floor(fy));');
    expect(BILINEAR_UPSAMPLE_WGSL).toContain('let ix0 = i32(floor(fx));');
  });
});
