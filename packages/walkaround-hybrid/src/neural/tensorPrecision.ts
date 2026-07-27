import {
  isNeuralCheckpointF16Compatible,
  type ModelWeights,
} from './weights.js';

/** Storage representation for neural inputs, activations, and outputs. */
export type NeuralTensorPrecision = 'f32' | 'f16';
export type NeuralTensorStoragePreference = 'auto' | NeuralTensorPrecision;
export type NeuralTensorStorageResolutionReason =
  | 'explicit-f32'
  | 'explicit-f16'
  | 'auto-certified-shader-f16'
  | 'auto-checkpoint-not-certified'
  | 'auto-shader-f16-not-enabled';

export interface NeuralTensorStorageResolution {
  readonly storage: NeuralTensorStorageContract;
  readonly reason: NeuralTensorStorageResolutionReason;
}

export interface NeuralTensorStorageContract {
  readonly precision: NeuralTensorPrecision;
  readonly bytesPerScalar: 4 | 2;
  readonly requiresShaderF16: boolean;
  /** Learned weights and biases intentionally remain f32. */
  readonly weightPrecision: 'f32';
}

export const NEURAL_F32_TENSOR_STORAGE: NeuralTensorStorageContract = Object.freeze({
  precision: 'f32',
  bytesPerScalar: 4,
  requiresShaderF16: false,
  weightPrecision: 'f32',
});

export const NEURAL_F16_TENSOR_STORAGE: NeuralTensorStorageContract = Object.freeze({
  precision: 'f16',
  bytesPerScalar: 2,
  requiresShaderF16: true,
  weightPrecision: 'f32',
});

/**
 * Resolve activation storage from both checkpoint compatibility and the
 * features enabled on the actual device. Older v2 metadata is deliberately
 * f32-only; new checkpoints must explicitly opt into f16 tensor rounding.
 * We never convert learned weights, so their f32 checkpoint payload is exact.
 */
export function resolveNeuralTensorStorage(
  device: GPUDevice,
  weights: ModelWeights,
  preference: NeuralTensorStoragePreference = 'auto',
): NeuralTensorStorageContract {
  return resolveNeuralTensorStorageDecision(device, weights, preference).storage;
}

export function resolveNeuralTensorStorageDecision(
  device: GPUDevice,
  weights: ModelWeights,
  preference: NeuralTensorStoragePreference = 'auto',
): NeuralTensorStorageResolution {
  const checkpointCompatible = isNeuralCheckpointF16Compatible(weights);
  const shaderF16Enabled = device.features?.has('shader-f16') === true;
  if (preference === 'f32') {
    return { storage: NEURAL_F32_TENSOR_STORAGE, reason: 'explicit-f32' };
  }
  if (preference === 'f16') {
    if (!checkpointCompatible) {
      throw new TypeError(
        "[neural] tensorStorage:'f16' requires a v2 checkpoint with passing mixed-precision certification",
      );
    }
    if (!shaderF16Enabled) {
      throw new TypeError(
        "[neural] tensorStorage:'f16' requires 'shader-f16' to be enabled on the GPUDevice",
      );
    }
    return { storage: NEURAL_F16_TENSOR_STORAGE, reason: 'explicit-f16' };
  }
  if (checkpointCompatible && shaderF16Enabled) {
    return { storage: NEURAL_F16_TENSOR_STORAGE, reason: 'auto-certified-shader-f16' };
  }
  return checkpointCompatible
    ? { storage: NEURAL_F32_TENSOR_STORAGE, reason: 'auto-shader-f16-not-enabled' }
    : { storage: NEURAL_F32_TENSOR_STORAGE, reason: 'auto-checkpoint-not-certified' };
}

export function neuralTensorWgslPreamble(
  storage: NeuralTensorStorageContract,
): string {
  return storage.precision === 'f16' ? 'enable f16;\n' : '';
}

export function neuralTensorWgslType(
  storage: NeuralTensorStorageContract,
): 'f32' | 'f16' {
  return storage.precision;
}
