import type { UNetSpec } from './unetArchitecture.js';
import type { ModelWeights } from './weights.js';
import {
  dispatchWorkgroupsFor,
  type TensorDims,
  preflightTensorDims,
} from './tensorDimSolver.js';
import {
  NEURAL_F32_TENSOR_STORAGE,
  type NeuralTensorPrecision,
  type NeuralTensorStorageContract,
} from './tensorPrecision.js';

export interface TensorAllocationSlot {
  readonly index: number;
  readonly byteSize: number;
  readonly logicalTensors: readonly string[];
}

export interface TensorAllocationPlan {
  readonly tensorToSlot: ReadonlyMap<string, number>;
  readonly slots: readonly TensorAllocationSlot[];
  readonly logicalTensorBytes: number;
  readonly physicalTensorBytes: number;
  readonly reuseSavingsBytes: number;
  readonly peakLiveTensorBytes: number;
}

export interface NeuralMemoryTelemetry {
  readonly tensorPrecision: NeuralTensorPrecision;
  readonly tensorBytesPerScalar: 4 | 2;
  readonly logicalTensorBytes: number;
  readonly physicalTensorBytes: number;
  readonly reuseSavingsBytes: number;
  readonly peakLiveTensorBytes: number;
  readonly tensorBufferCount: number;
  readonly logicalTensorCount: number;
  readonly parameterBytes: number;
  readonly uniformAndPlaceholderBytes: number;
  readonly totalAllocatedBytes: number;
}

export interface NeuralDeviceResolutionLimit {
  readonly width: number;
  readonly height: number;
}

interface MutableSlot {
  index: number;
  byteSize: number;
  releaseAfterLayer: number;
  logicalTensors: string[];
}

export function tensorByteSize(
  dims: TensorDims,
  label = 'tensor',
  storage: NeuralTensorStorageContract = NEURAL_F32_TENSOR_STORAGE,
): number {
  const elements = safeMultiply([dims.H, dims.W, dims.C], label + ' element count');
  return safeMultiply([elements, storage.bytesPerScalar], label + ' byte size');
}

function safeMultiply(values: readonly number[], label: string): number {
  let product = 1;
  for (const value of values) {
    if (!Number.isSafeInteger(value) || value < 0) {
      throw new RangeError('[neural] ' + label + ' contains invalid factor ' + value);
    }
    product *= value;
    if (!Number.isSafeInteger(product)) {
      throw new RangeError('[neural] ' + label + ' exceeds Number.MAX_SAFE_INTEGER');
    }
  }
  return product;
}

/**
 * Assign logical tensors to reusable storage slots using exact last-use
 * intervals. A slot is reused only when its previous value's final consumer is
 * in an earlier layer, so no dispatch ever binds one buffer as both a live input
 * and a new output.
 */
export function buildTensorAllocationPlan(
  spec: UNetSpec,
  dims: ReadonlyMap<string, TensorDims>,
  storage: NeuralTensorStorageContract = NEURAL_F32_TENSOR_STORAGE,
): TensorAllocationPlan {
  const lastUse = new Map<string, number>();
  for (let i = 0; i < spec.layers.length; i++) {
    for (const input of spec.layers[i]!.inputs) {
      if (dims.has(input)) lastUse.set(input, Math.max(lastUse.get(input) ?? -1, i));
    }
  }
  lastUse.set('denoised', spec.layers.length);

  const slots: MutableSlot[] = [];
  const tensorToSlot = new Map<string, number>();
  let logicalTensorBytes = 0;
  let peakLiveTensorBytes = 0;

  for (let i = 0; i < spec.layers.length; i++) {
    const output = spec.layers[i]!.output;
    const outputDims = dims.get(output);
    if (outputDims == null) {
      throw new Error("[neural memory planner] missing dimensions for '" + output + "'");
    }
    const bytes = tensorByteSize(outputDims, output, storage);
    logicalTensorBytes += bytes;
    const releaseAfterLayer = lastUse.get(output) ?? i;

    let selected: MutableSlot | undefined;
    for (const slot of slots) {
      if (slot.releaseAfterLayer >= i || slot.byteSize < bytes) continue;
      if (selected == null || slot.byteSize < selected.byteSize) selected = slot;
    }
    if (selected == null) {
      selected = {
        index: slots.length,
        byteSize: bytes,
        releaseAfterLayer,
        logicalTensors: [],
      };
      slots.push(selected);
    } else {
      selected.releaseAfterLayer = releaseAfterLayer;
    }
    selected.logicalTensors.push(output);
    tensorToSlot.set(output, selected.index);

    let liveBytes = 0;
    for (const slot of slots) {
      if (slot.releaseAfterLayer >= i) liveBytes += slot.byteSize;
    }
    peakLiveTensorBytes = Math.max(peakLiveTensorBytes, liveBytes);
  }

  const physicalTensorBytes = slots.reduce((sum, slot) => sum + slot.byteSize, 0);
  return {
    tensorToSlot,
    slots: slots.map(slot => Object.freeze({
      index: slot.index,
      byteSize: slot.byteSize,
      logicalTensors: Object.freeze([...slot.logicalTensors]),
    })),
    logicalTensorBytes,
    physicalTensorBytes,
    reuseSavingsBytes: logicalTensorBytes - physicalTensorBytes,
    peakLiveTensorBytes,
  };
}

export function estimateNeuralMemory(
  spec: UNetSpec,
  weights: ModelWeights,
  dims: ReadonlyMap<string, TensorDims>,
  storage: NeuralTensorStorageContract = NEURAL_F32_TENSOR_STORAGE,
): NeuralMemoryTelemetry {
  const plan = buildTensorAllocationPlan(spec, dims, storage);
  let parameterBytes = 0;
  for (const layer of weights.layers) {
    parameterBytes += layer.weights.byteLength + layer.biases.byteLength;
  }
  const computeLayerCount = spec.layers.filter(layer => layer.kind !== 'inputPack').length;
  const uniformAndPlaceholderBytes = 4 + 16 + computeLayerCount * 48;
  return {
    logicalTensorBytes: plan.logicalTensorBytes,
    tensorPrecision: storage.precision,
    tensorBytesPerScalar: storage.bytesPerScalar,
    physicalTensorBytes: plan.physicalTensorBytes,
    reuseSavingsBytes: plan.reuseSavingsBytes,
    peakLiveTensorBytes: plan.peakLiveTensorBytes,
    tensorBufferCount: plan.slots.length,
    logicalTensorCount: plan.tensorToSlot.size,
    parameterBytes,
    uniformAndPlaceholderBytes,
    totalAllocatedBytes: plan.physicalTensorBytes + parameterBytes + uniformAndPlaceholderBytes,
  };
}

function deviceLimit(device: GPUDevice, name: keyof GPUSupportedLimits, fallback: number): number {
  const value = device.limits?.[name];
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

export function neuralDeviceLimitFailures(
  device: GPUDevice,
  spec: UNetSpec,
  weights: ModelWeights,
  dims: ReadonlyMap<string, TensorDims>,
  storage: NeuralTensorStorageContract = NEURAL_F32_TENSOR_STORAGE,
): string[] {
  const failures: string[] = [];
  const maxBufferSize = deviceLimit(device, 'maxBufferSize', Number.POSITIVE_INFINITY);
  const maxStorage = deviceLimit(device, 'maxStorageBufferBindingSize', Number.POSITIVE_INFINITY);
  const maxUniform = deviceLimit(device, 'maxUniformBufferBindingSize', Number.POSITIVE_INFINITY);
  const maxWorkgroups = deviceLimit(device, 'maxComputeWorkgroupsPerDimension', 65_535);
  const maxTextureDimension = deviceLimit(device, 'maxTextureDimension2D', Number.POSITIVE_INFINITY);
  const maxStorageBindings = deviceLimit(device, 'maxStorageBuffersPerShaderStage', Number.POSITIVE_INFINITY);
  const maxInvocations = deviceLimit(device, 'maxComputeInvocationsPerWorkgroup', Number.POSITIVE_INFINITY);
  const maxWorkgroupX = deviceLimit(device, 'maxComputeWorkgroupSizeX', Number.POSITIVE_INFINITY);
  const maxWorkgroupY = deviceLimit(device, 'maxComputeWorkgroupSizeY', Number.POSITIVE_INFINITY);

  const denoised = dims.get('denoised');
  if (denoised != null && (denoised.W > maxTextureDimension || denoised.H > maxTextureDimension)) {
    failures.push(
      'output texture ' + denoised.W + 'x' + denoised.H +
      ' exceeds maxTextureDimension2D=' + maxTextureDimension,
    );
  }
  if (maxStorageBindings < 4) failures.push('maxStorageBuffersPerShaderStage=' + maxStorageBindings + ' < 4');
  if (maxInvocations < 256) failures.push('maxComputeInvocationsPerWorkgroup=' + maxInvocations + ' < 256');
  if (maxWorkgroupX < 256) failures.push('maxComputeWorkgroupSizeX=' + maxWorkgroupX + ' < 256');
  if (maxWorkgroupY < 8) failures.push('maxComputeWorkgroupSizeY=' + maxWorkgroupY + ' < 8');
  if (maxUniform < 48) failures.push('maxUniformBufferBindingSize=' + maxUniform + ' < 48');

  const plan = buildTensorAllocationPlan(spec, dims, storage);
  for (const slot of plan.slots) {
    if (slot.byteSize > maxBufferSize) {
      failures.push('tensor slot ' + slot.index + ' requires ' + slot.byteSize + ' bytes > maxBufferSize=' + maxBufferSize);
    }
    if (slot.byteSize > maxStorage) {
      failures.push('tensor slot ' + slot.index + ' requires ' + slot.byteSize + ' bytes > maxStorageBufferBindingSize=' + maxStorage);
    }
  }
  for (const layer of weights.layers) {
    for (const [kind, bytes] of [
      ['weights', layer.weights.byteLength],
      ['biases', layer.biases.byteLength],
    ] as const) {
      if (bytes > maxBufferSize || bytes > maxStorage) {
        failures.push(
          "layer '" + layer.name + "' " + kind + ' requires ' + bytes +
          ' bytes; limits are maxBufferSize=' + maxBufferSize +
          ', maxStorageBufferBindingSize=' + maxStorage,
        );
      }
    }
  }

  for (const layer of spec.layers) {
    if (layer.kind === 'inputPack') continue;
    const output = dims.get(layer.output);
    if (output == null) continue;
    const groups = dispatchWorkgroupsFor(layer.kind, output, maxWorkgroups);
    groups.forEach((count, axis) => {
      if (!Number.isSafeInteger(count) || count <= 0) {
        failures.push("layer '" + layer.name + "' has invalid workgroup count " + count);
      } else if (count > maxWorkgroups) {
        failures.push(
          "layer '" + layer.name + "' workgroup axis " + axis + '=' + count +
          ' exceeds maxComputeWorkgroupsPerDimension=' + maxWorkgroups,
        );
      }
    });
  }
  return failures;
}

export function maxSupportedNeuralResolutionForAspect(
  device: GPUDevice,
  spec: UNetSpec,
  weights: ModelWeights,
  aspectWidth: number,
  aspectHeight: number,
  storage: NeuralTensorStorageContract = NEURAL_F32_TENSOR_STORAGE,
): NeuralDeviceResolutionLimit {
  if (!(aspectWidth > 0) || !(aspectHeight > 0)) return { width: 0, height: 0 };
  const maxDimension = Math.min(
    deviceLimit(device, 'maxTextureDimension2D', 16_384),
    deviceLimit(device, 'maxComputeWorkgroupsPerDimension', 65_535) * 8,
  );
  const aspect = aspectWidth / aspectHeight;
  let lo = 1;
  let hi = Math.max(1, Math.floor(maxDimension / 8));
  let best = { width: 0, height: 0 };

  while (lo <= hi) {
    const mid = Math.floor((lo + hi) / 2);
    const height = mid * 8;
    const width = Math.max(8, Math.floor((height * aspect) / 8) * 8);
    let supported = width <= maxDimension && height <= maxDimension;
    if (supported) {
      try {
        const dims = preflightTensorDims(spec, width, height);
        supported = neuralDeviceLimitFailures(device, spec, weights, dims, storage).length === 0;
      } catch {
        supported = false;
      }
    }
    if (supported) {
      best = { width, height };
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  return best;
}

export function assertNeuralDeviceSupportsGraph(
  device: GPUDevice,
  spec: UNetSpec,
  weights: ModelWeights,
  dims: ReadonlyMap<string, TensorDims>,
  width: number,
  height: number,
  storage: NeuralTensorStorageContract = NEURAL_F32_TENSOR_STORAGE,
): void {
  const failures = neuralDeviceLimitFailures(device, spec, weights, dims, storage);
  if (failures.length === 0) return;
  const maximum = maxSupportedNeuralResolutionForAspect(device, spec, weights, width, height, storage);
  throw new RangeError(
    '[InferenceGraph] neural graph ' + width + 'x' + height +
    ' using ' + storage.precision + ' tensor storage exceeds this adapter: ' + failures.join('; ') +
    '. Maximum supported resolution at this aspect is ' + maximum.width + 'x' + maximum.height + '.',
  );
}
