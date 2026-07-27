import {
  postprocessNeuralRadiance,
  preprocessNeuralInputs,
  preprocessingContractForCheckpoint,
  sanitizeNeuralSigned,
} from './preprocessing.js';
import { preflightTensorDims, type TensorDims } from './tensorDimSolver.js';
import type { LayerSpec, UNetSpec } from './unetArchitecture.js';
import {
  validateWeightsForSpec,
  type LayerWeights,
  type ModelWeights,
} from './weights.js';
import { roundNeuralTensorScalar } from './float16.js';
import type { NeuralTensorPrecision } from './tensorPrecision.js';

export interface NeuralCpuInputs {
  readonly noisyColor: Float32Array;
  readonly albedo: Float32Array;
  readonly normals: Float32Array;
}

export interface NeuralCpuInferenceResult {
  /** HWC output exactly as it exists in the graph before texture unpack. */
  readonly modelOutput: Float32Array;
  /** Finite linear-HDR RGB after the checkpoint's inverse output transform. */
  readonly denoised: Float32Array;
}

/**
 * Deterministic CPU oracle for the neural graph. It mirrors the WGSL indexing,
 * f32 storage rounding, per-accumulation finite clamp, and checkpoint-specific
 * preprocessing contract. This is intentionally correctness-first, not fast.
 */
export function executeNeuralInferenceCpu(
  spec: UNetSpec,
  weights: ModelWeights,
  width: number,
  height: number,
  inputs: NeuralCpuInputs,
  tensorPrecision: NeuralTensorPrecision = 'f32',
): NeuralCpuInferenceResult {
  validateWeightsForSpec(spec, weights);
  const dims = preflightTensorDims(spec, width, height);
  const rgbLength = width * height * 3;
  const inputChannels = [
    ['noisyColor', inputs.noisyColor],
    ['albedo', inputs.albedo],
    ['normals', inputs.normals],
  ] as const;
  for (const [name, values] of inputChannels) {
    if (values.length !== rgbLength) {
      throw new RangeError(
        `[executeNeuralInferenceCpu] ${name} length ${values.length} != ${rgbLength}`,
      );
    }
  }

  const contract = preprocessingContractForCheckpoint(weights.checkpoint);
  const tensors = new Map<string, Float32Array>([
    ['noisyColor', roundTensor(inputs.noisyColor, tensorPrecision)],
    ['albedo', roundTensor(inputs.albedo, tensorPrecision)],
    ['normals', roundTensor(inputs.normals, tensorPrecision)],
  ]);
  const weightsByName = new Map<string, LayerWeights>(
    weights.layers.map(layer => [layer.name, layer]),
  );

  for (const layer of spec.layers) {
    const outputDims = requireDims(dims, layer.output);
    const output = executeLayer(
      layer,
      outputDims,
      dims,
      tensors,
      weightsByName.get(layer.name),
      contract,
    );
    tensors.set(layer.output, output);
    roundTensorInPlace(output, tensorPrecision);
  }

  const modelOutput = tensors.get('denoised');
  if (modelOutput == null) {
    throw new Error("[executeNeuralInferenceCpu] graph did not produce 'denoised'");
  }
  const denoised = new Float32Array(modelOutput.length);
  for (let i = 0; i < modelOutput.length; i++) {
    denoised[i] = postprocessNeuralRadiance(modelOutput[i]!, contract);
  }
  return { modelOutput: modelOutput.slice(), denoised };
}


function roundTensor(values: Float32Array, precision: NeuralTensorPrecision): Float32Array {
  const rounded = values.slice();
  roundTensorInPlace(rounded, precision);
  return rounded;
}

function roundTensorInPlace(
  values: Float32Array,
  precision: NeuralTensorPrecision,
): void {
  for (let i = 0; i < values.length; i++) {
    values[i] = roundNeuralTensorScalar(values[i]!, precision);
  }
}
function executeLayer(
  layer: LayerSpec,
  outputDims: TensorDims,
  allDims: ReadonlyMap<string, TensorDims>,
  tensors: ReadonlyMap<string, Float32Array>,
  layerWeights: LayerWeights | undefined,
  contract: Parameters<typeof preprocessNeuralInputs>[3],
): Float32Array {
  switch (layer.kind) {
    case 'inputPack': {
      return preprocessNeuralInputs(
        requireTensor(tensors, layer.inputs[0]!),
        requireTensor(tensors, layer.inputs[1]!),
        requireTensor(tensors, layer.inputs[2]!),
        contract,
      );
    }
    case 'conv2d':
      return conv2d(
        requireTensor(tensors, layer.inputs[0]!),
        requireDims(allDims, layer.inputs[0]!),
        outputDims,
        requireLayerWeights(layer, layerWeights),
        layer,
      );
    case 'transposedConv2d':
      return transposedConv2d(
        requireTensor(tensors, layer.inputs[0]!),
        requireDims(allDims, layer.inputs[0]!),
        outputDims,
        requireLayerWeights(layer, layerWeights),
        layer,
      );
    case 'relu': {
      const input = requireTensor(tensors, layer.inputs[0]!);
      const output = new Float32Array(input.length);
      for (let i = 0; i < input.length; i++) {
        output[i] = sanitizeNeuralSigned(Math.max(0, sanitizeNeuralSigned(input[i]!)));
      }
      return output;
    }
    case 'skipAdd': {
      const a = requireTensor(tensors, layer.inputs[0]!);
      const b = requireTensor(tensors, layer.inputs[1]!);
      const output = new Float32Array(a.length);
      for (let i = 0; i < a.length; i++) {
        output[i] = sanitizeNeuralSigned(
          sanitizeNeuralSigned(a[i]!) + sanitizeNeuralSigned(b[i]!),
        );
      }
      return output;
    }
    case 'bilinearUpsample':
      return bilinearUpsample(
        requireTensor(tensors, layer.inputs[0]!),
        requireDims(allDims, layer.inputs[0]!),
        outputDims,
      );
  }
}

/**
 * WGSL convolution values are f32 even when activation storage is f16. Keep
 * the CPU oracle in the same numeric domain by rounding the multiply and the
 * accumulated sum after every logical MAC; JavaScript numbers would otherwise
 * retain binary64 precision until the final Float32Array store.
 */
function accumulateF32(acc: number, input: number, weight: number): number {
  const product = Math.fround(
    Math.fround(sanitizeNeuralSigned(input)) *
    Math.fround(sanitizeNeuralSigned(weight)),
  );
  return sanitizeNeuralSigned(Math.fround(Math.fround(acc) + product));
}

function conv2d(
  input: Float32Array,
  inputDims: TensorDims,
  outputDims: TensorDims,
  layerWeights: LayerWeights,
  layer: LayerSpec,
): Float32Array {
  const output = new Float32Array(outputDims.H * outputDims.W * outputDims.C);
  const kH = layer.params.kH ?? 3;
  const kW = layer.params.kW ?? 3;
  const stride = layer.params.stride ?? 1;
  const padding = layer.params.padding ?? (kH === 3 && kW === 3 ? 1 : 0);

  for (let oy = 0; oy < outputDims.H; oy++) {
    for (let ox = 0; ox < outputDims.W; ox++) {
      for (let oc = 0; oc < outputDims.C; oc++) {
        let acc = sanitizeNeuralSigned(layerWeights.biases[oc]!);
        const iyBase = oy * stride - padding;
        const ixBase = ox * stride - padding;
        for (let kh = 0; kh < kH; kh++) {
          const iy = iyBase + kh;
          if (iy < 0 || iy >= inputDims.H) continue;
          for (let kw = 0; kw < kW; kw++) {
            const ix = ixBase + kw;
            if (ix < 0 || ix >= inputDims.W) continue;
            for (let ic = 0; ic < inputDims.C; ic++) {
              const inputIndex = (iy * inputDims.W + ix) * inputDims.C + ic;
              const weightIndex =
                ((oc * inputDims.C + ic) * kH + kh) * kW + kw;
              acc = accumulateF32(
                acc,
                input[inputIndex]!,
                layerWeights.weights[weightIndex]!,
              );
            }
          }
        }
        output[(oy * outputDims.W + ox) * outputDims.C + oc] =
          sanitizeNeuralSigned(acc);
      }
    }
  }
  return output;
}

function transposedConv2d(
  input: Float32Array,
  inputDims: TensorDims,
  outputDims: TensorDims,
  layerWeights: LayerWeights,
  layer: LayerSpec,
): Float32Array {
  const output = new Float32Array(outputDims.H * outputDims.W * outputDims.C);
  const kH = layer.params.kH ?? 2;
  const kW = layer.params.kW ?? 2;
  const stride = layer.params.stride ?? 2;
  const padding = layer.params.padding ?? 0;
  const dilation = layer.params.dilation ?? 1;

  for (let oy = 0; oy < outputDims.H; oy++) {
    for (let ox = 0; ox < outputDims.W; ox++) {
      for (let oc = 0; oc < outputDims.C; oc++) {
        let acc = sanitizeNeuralSigned(layerWeights.biases[oc]!);
        for (let kh = 0; kh < kH; kh++) {
          const iyNumerator = oy + padding - kh * dilation;
          if (iyNumerator < 0 || iyNumerator % stride !== 0) continue;
          const iy = iyNumerator / stride;
          if (iy >= inputDims.H) continue;
          for (let kw = 0; kw < kW; kw++) {
            const ixNumerator = ox + padding - kw * dilation;
            if (ixNumerator < 0 || ixNumerator % stride !== 0) continue;
            const ix = ixNumerator / stride;
            if (ix >= inputDims.W) continue;
            for (let ic = 0; ic < inputDims.C; ic++) {
              const inputIndex = (iy * inputDims.W + ix) * inputDims.C + ic;
              const weightIndex =
                ((ic * outputDims.C + oc) * kH + kh) * kW + kw;
              acc = accumulateF32(
                acc,
                input[inputIndex]!,
                layerWeights.weights[weightIndex]!,
              );
            }
          }
        }
        output[(oy * outputDims.W + ox) * outputDims.C + oc] =
          sanitizeNeuralSigned(acc);
      }
    }
  }
  return output;
}

function bilinearUpsample(
  input: Float32Array,
  inputDims: TensorDims,
  outputDims: TensorDims,
): Float32Array {
  const output = new Float32Array(outputDims.H * outputDims.W * outputDims.C);
  const sample = (y: number, x: number, channel: number): number => {
    const cy = Math.max(0, Math.min(y, inputDims.H - 1));
    const cx = Math.max(0, Math.min(x, inputDims.W - 1));
    return sanitizeNeuralSigned(input[(cy * inputDims.W + cx) * inputDims.C + channel]!);
  };

  for (let oy = 0; oy < outputDims.H; oy++) {
    for (let ox = 0; ox < outputDims.W; ox++) {
      const fy = (oy + 0.5) / 2 - 0.5;
      const fx = (ox + 0.5) / 2 - 0.5;
      const y0 = Math.floor(fy);
      const x0 = Math.floor(fx);
      const ty = fy - y0;
      const tx = fx - x0;
      for (let channel = 0; channel < outputDims.C; channel++) {
        const value =
          sample(y0, x0, channel) * (1 - ty) * (1 - tx) +
          sample(y0, x0 + 1, channel) * (1 - ty) * tx +
          sample(y0 + 1, x0, channel) * ty * (1 - tx) +
          sample(y0 + 1, x0 + 1, channel) * ty * tx;
        output[(oy * outputDims.W + ox) * outputDims.C + channel] =
          sanitizeNeuralSigned(value);
      }
    }
  }
  return output;
}

function requireTensor(
  tensors: ReadonlyMap<string, Float32Array>,
  name: string,
): Float32Array {
  const tensor = tensors.get(name);
  if (tensor == null) throw new Error(`[executeNeuralInferenceCpu] missing tensor '${name}'`);
  return tensor;
}

function requireDims(
  dims: ReadonlyMap<string, TensorDims>,
  name: string,
): TensorDims {
  const value = dims.get(name);
  if (value == null) throw new Error(`[executeNeuralInferenceCpu] missing dims for '${name}'`);
  return value;
}

function requireLayerWeights(
  layer: LayerSpec,
  weights: LayerWeights | undefined,
): LayerWeights {
  if (weights == null) {
    throw new Error(`[executeNeuralInferenceCpu] missing weights for '${layer.name}'`);
  }
  return weights;
}
