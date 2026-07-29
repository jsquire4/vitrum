/**
 * layerResourceAllocator.ts — GPU-resource allocation for the U-Net inference
 * graph: intermediate tensor buffers, per-layer compute pipelines, weight/bias
 * buffers, uniform buffers, and bind groups.
 *
 * The executor delegates resource construction here. Tensor slots follow exact
 * liveness intervals, and bind groups contain only the bindings declared by
 * each WGSL kernel's auto-layout. Real-WebGPU canonical-graph coverage pins
 * those layouts in cpuGpuParity.gpu.test.ts.
 */

import type { UNetSpec, LayerSpec, LayerKind } from './unetArchitecture.js';
import type { ModelWeights, LayerWeights } from './weights.js';
import { CONV2D_WGSL } from './wgsl/conv2d.wgsl.js';
import { TRANSPOSED_CONV2D_WGSL } from './wgsl/transposedConv2d.wgsl.js';
import { RELU_WGSL } from './wgsl/relu.wgsl.js';
import { SKIP_CONNECTION_WGSL } from './wgsl/skipConnection.wgsl.js';
import { buildInputPackerWgsl, INPUT_PACKER_ENTRY } from './inputPacker.js';
import { buildOutputCropWgsl, OUTPUT_CROP_ENTRY } from './outputCrop.js';
import { preprocessingContractForCheckpoint } from './preprocessing.js';
import {
  type TensorDims,
  packLayerUniform,
} from './tensorDimSolver.js';
import {
  buildTensorAllocationPlan,
  estimateNeuralMemory,
  type NeuralMemoryTelemetry,
} from './tensorMemoryPlanner.js';
import { neuralLayerWgslForStorage } from './mixedPrecisionWgsl.js';
import {
  NEURAL_F32_TENSOR_STORAGE,
  type NeuralTensorStorageContract,
} from './tensorPrecision.js';


// ── Types ─────────────────────────────────────────────────────────────────────

export interface TensorBuffer {
  buf:    GPUBuffer;
  dims:   TensorDims;
  label:  string;
}

/** Per-layer GPU state: pipeline + buffers + cached bind group. */
export interface LayerGPUState {
  readonly layerName: string;
  readonly pipeline:  GPUComputePipeline;
  readonly uniformBuf: GPUBuffer;
  /** Cached bind group — invalidated (set to null) if any buffer identity changes. */
  cachedBindGroup: GPUBindGroup | null;
  /** Buffer identity keys used to check cache validity. */
  cachedBufKeys: readonly string[];
}

// ── WGSL entry points per layer kind ─────────────────────────────────────────

const WGSL_ENTRY: Record<LayerKind, string> = {
  conv2d:          'conv2dMain',
  transposedConv2d:'transposedConv2dMain',
  relu:            'reluMain',
  skipAdd:         'skipConnectionMain',
  inputPack:       '',  // handled CPU-side by packing pass
};

const WGSL_SOURCE: Partial<Record<LayerKind, string>> = {
  conv2d:          CONV2D_WGSL,
  transposedConv2d:TRANSPOSED_CONV2D_WGSL,
  relu:            RELU_WGSL,
  skipAdd:         SKIP_CONNECTION_WGSL,
};

// Largest layer uniform is TConv2DParams: 10 fields, padded to 48 bytes.
const UNIFORM_BUF_BYTES = 48;

// Placeholder buffer size for defensive missing-buffer fallbacks.
const PLACEHOLDER_BYTES = 4;

/** Everything the InferenceGraph executor needs after allocation. */
export interface AllocatedGraph {
  tensors: Map<string, TensorBuffer>;
  layerStates: (LayerGPUState | null)[];
  placeholderBuf: GPUBuffer;
  inputPackPipeline: GPUComputePipeline;
  inputPackUniformBuf: GPUBuffer;
  outputCropPipeline: GPUComputePipeline;
  outputCropUniformBuf: GPUBuffer;
  /** All non-tensor buffers (weights/biases/uniforms/placeholder/input-pack
   *  uniform) tracked so dispose() can destroy them. */
  allocatedBuffers: GPUBuffer[];
  /** Weight lookup retained for bind-group rebuild on resize. */
  weightsByName: Map<string, LayerWeights>;
  /** Uniform-write count incurred during allocation (init-time writes). */
  uniformWriteCount: number;
  memoryTelemetry: NeuralMemoryTelemetry;
}

/**
 * Allocate every GPU resource the inference graph needs for one resolution.
 * Writes uniform buffers with actual shape params and builds the per-layer bind
 * groups (cached by buffer identity).
 */
export async function allocateGraph(
  device: GPUDevice,
  spec: UNetSpec,
  weights: ModelWeights,
  W: number,
  H: number,
  tensorDimsMap: Map<string, TensorDims>,
  storage: NeuralTensorStorageContract = NEURAL_F32_TENSOR_STORAGE,
  inferenceW: number = W,
  inferenceH: number = H,
): Promise<AllocatedGraph> {
  const allocatedBuffers: GPUBuffer[] = [];
  let uniformWriteCount = 0;
  const tensors = new Map<string, TensorBuffer>();
  const tensorSlotBuffers: GPUBuffer[] = [];

  try {

  // Build the weight lookup by name (retained for bind-group rebuild in run()).
  const weightsByName = new Map<string, LayerWeights>(
    weights.layers.map(lw => [lw.name, lw]),
  );

  // Placeholder buffer for defensive missing-buffer fallbacks.
  const placeholderBuf = device.createBuffer({
    label: 'neural/placeholder',
    size: PLACEHOLDER_BYTES,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.UNIFORM,
  });
  allocatedBuffers.push(placeholderBuf);

  // ── Compile the input-packer compute pipeline once.
  const packModule = device.createShaderModule({
    label: 'neural/inputPacker',
    code: buildInputPackerWgsl(preprocessingContractForCheckpoint(weights.checkpoint), storage),
  });
  const inputPackPipeline = await device.createComputePipelineAsync({
    label: 'neural-pipeline-inputPack',
    layout: 'auto',
    compute: { module: packModule, entryPoint: INPUT_PACKER_ENTRY },
  });

  const cropModule = device.createShaderModule({
    label: 'neural/outputCrop',
    code: buildOutputCropWgsl(storage),
  });
  const outputCropPipeline = await device.createComputePipelineAsync({
    label: 'neural-pipeline-outputCrop',
    layout: 'auto',
    compute: { module: cropModule, entryPoint: OUTPUT_CROP_ENTRY },
  });

  // Logical and padded extents used to zero-fill the private inference lattice.
  const inputPackUniformBuf = device.createBuffer({
    label: 'neural-uniform-inputPack',
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  allocatedBuffers.push(inputPackUniformBuf);
  {
    const u32 = new Uint32Array([W, H, inferenceW, inferenceH]);
    device.queue.writeBuffer(inputPackUniformBuf, 0, u32.buffer);
  }

  const outputCropUniformBuf = device.createBuffer({
    label: 'neural-uniform-outputCrop',
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  allocatedBuffers.push(outputCropUniformBuf);
  device.queue.writeBuffer(
    outputCropUniformBuf,
    0,
    new Uint32Array([W, H, inferenceW, 0]).buffer,
  );

  // Allocate the liveness-planned physical tensor slots. Multiple logical
  // tensors may share a slot only when the earlier value's final consumer is
  // strictly before the later producer.
  const tensorPlan = buildTensorAllocationPlan(spec, tensorDimsMap, storage);
  for (const slot of tensorPlan.slots) {
    const buf = device.createBuffer({
      label: `neural/slot-${slot.index}:${slot.logicalTensors.join(',')}`,
      size: slot.byteSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    tensorSlotBuffers.push(buf);
  }
  for (const [name, slotIndex] of tensorPlan.tensorToSlot) {
    const dims = tensorDimsMap.get(name);
    const buf = tensorSlotBuffers[slotIndex];
    if (dims == null || buf == null) {
      throw new Error(`[InferenceGraph] incomplete tensor allocation plan for '${name}'`);
    }
    tensors.set(name, { buf, dims, label: `neural/${name}` });
  }

  // ── Build pipelines + layer states ────────────────────────────────────
  const N = spec.layers.length;
  const layerStates = new Array(N).fill(null) as (LayerGPUState | null)[];

  for (let i = 0; i < N; i++) {
    const layer = spec.layers[i]!;

    if (layer.kind === 'inputPack') {
      layerStates[i] = null;
      continue;
    }

    const wgsl = WGSL_SOURCE[layer.kind];
    if (!wgsl) {
      layerStates[i] = null;
      continue;
    }

    const sm = device.createShaderModule({
      label: `neural-${layer.name}`,
      code: neuralLayerWgslForStorage(layer.kind, wgsl, storage),
    });
    const pipeline = await device.createComputePipelineAsync({
      label: `neural-pipeline-${layer.name}`,
      layout: 'auto',
      compute: { module: sm, entryPoint: WGSL_ENTRY[layer.kind] },
    });

    const uniformBuf = device.createBuffer({
      label: `neural-uniform-${layer.name}`,
      size: UNIFORM_BUF_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    allocatedBuffers.push(uniformBuf);

    // Write the uniform buffer now with actual shape params.
    device.queue.writeBuffer(
      uniformBuf,
      0,
      packLayerUniform(
        layer,
        tensorDimsMap,
        inferenceH,
        inferenceW,
        device.limits.maxComputeWorkgroupsPerDimension,
      ),
    );
    uniformWriteCount++;

    const { bindGroup, bufKeys } = buildBindGroup(
      device, pipeline, layer, weightsByName, uniformBuf,
      tensors, placeholderBuf, allocatedBuffers,
    );

    layerStates[i] = {
      layerName:      layer.name,
      pipeline,
      uniformBuf,
      cachedBindGroup: bindGroup,
      cachedBufKeys:   bufKeys,
    };
  }

  return {
    tensors,
    layerStates,
    placeholderBuf,
    inputPackPipeline,
    inputPackUniformBuf,
    outputCropPipeline,
    outputCropUniformBuf,
    allocatedBuffers,
    weightsByName,
    uniformWriteCount,
    memoryTelemetry: estimateNeuralMemory(spec, weights, tensorDimsMap, storage),
  };
  } catch (err) {
    const buffersToDestroy = new Set<GPUBuffer>(allocatedBuffers);
    for (const buffer of tensorSlotBuffers) {
      buffersToDestroy.add(buffer);
    }
    for (const tensor of tensors.values()) {
      buffersToDestroy.add(tensor.buf);
    }
    for (const buffer of buffersToDestroy) {
      try { buffer.destroy(); } catch { /* tolerate partial cleanup errors */ }
    }
    throw err;
  }
}

/**
 * Build a bind group for a layer.
 * Binding entries match each WGSL auto-layout exactly: conv layers use all
 * bindings 0..4, skip-add omits binding 2, and element-wise layers use only
 * bindings 0, 3, and 4.
 *
 * For conv2d/transposedConv2d this UPLOADS weight + bias buffers (appending them
 * to `allocatedBuffers` for dispose tracking). Returns the bind group + the
 * buffer-identity keys (input + output labels) for cache invalidation.
 */
export function buildBindGroup(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  layer: LayerSpec,
  weightsByName: Map<string, LayerWeights>,
  uniformBuf: GPUBuffer,
  tensors: Map<string, TensorBuffer>,
  placeholderBuf: GPUBuffer,
  allocatedBuffers: GPUBuffer[],
): { bindGroup: GPUBindGroup; bufKeys: readonly string[] } {
  // Input buffer — always binding 0.
  const inputName = layer.inputs[0] ?? 'enc_input';
  const inputTensor = tensors.get(inputName);
  const inputBuf = inputTensor?.buf ?? placeholderBuf;

  // Weights buffer — binding 1 (or inputB for skipAdd).
  let weightsBuf = placeholderBuf;
  if (layer.kind === 'skipAdd') {
    const skipName = layer.inputs[1] ?? 'enc_input';
    weightsBuf = tensors.get(skipName)?.buf ?? placeholderBuf;
  } else if (layer.kind === 'conv2d' || layer.kind === 'transposedConv2d') {
    const lw = weightsByName.get(layer.name);
    if (lw && lw.weights.length > 0) {
      weightsBuf = device.createBuffer({
        label: `neural-weights-${layer.name}`,
        size: Math.max(PLACEHOLDER_BYTES, lw.weights.byteLength),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
      });
      allocatedBuffers.push(weightsBuf);
      new Float32Array(weightsBuf.getMappedRange()).set(lw.weights);
      weightsBuf.unmap();
    }
  }

  // Biases buffer — binding 2.
  let biasesBuf = placeholderBuf;
  if (layer.kind === 'conv2d' || layer.kind === 'transposedConv2d') {
    const lw = weightsByName.get(layer.name);
    if (lw && lw.biases.length > 0) {
      biasesBuf = device.createBuffer({
        label: `neural-biases-${layer.name}`,
        size: Math.max(PLACEHOLDER_BYTES, lw.biases.byteLength),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
      });
      allocatedBuffers.push(biasesBuf);
      new Float32Array(biasesBuf.getMappedRange()).set(lw.biases);
      biasesBuf.unmap();
    }
  }

  // Output buffer — binding 3.
  const outputName = layer.output;
  const outputTensor = tensors.get(outputName);
  const outputBuf = outputTensor?.buf ?? placeholderBuf;

  const entries: GPUBindGroupEntry[] = [
    { binding: 0, resource: { buffer: inputBuf } },
  ];
  if (layer.kind === 'conv2d' || layer.kind === 'transposedConv2d') {
    entries.push(
      { binding: 1, resource: { buffer: weightsBuf } },
      { binding: 2, resource: { buffer: biasesBuf } },
    );
  } else if (layer.kind === 'skipAdd') {
    entries.push({ binding: 1, resource: { buffer: weightsBuf } });
  }
  entries.push(
    { binding: 3, resource: { buffer: outputBuf } },
    { binding: 4, resource: { buffer: uniformBuf } },
  );

  const bindGroup = device.createBindGroup({
    label: `neural-bg-${layer.name}`,
    layout: pipeline.getBindGroupLayout(0),
    entries,
  });

  // Only the dynamic buffers (input + output) need cache-keying — weights and
  // biases are static after allocation.
  const bufKeys = [
    inputBuf.label ?? '',
    outputBuf.label ?? '',
  ] as const;

  return { bindGroup, bufKeys };
}

/** The input + output buffer-identity keys for a layer (cache validity check). */
export function currentBufKeys(
  layer: LayerSpec,
  tensors: Map<string, TensorBuffer>,
  placeholderBuf: GPUBuffer,
): readonly string[] {
  const inputName  = layer.inputs[0] ?? 'enc_input';
  const outputName = layer.output;
  const inputBuf   = tensors.get(inputName)?.buf ?? placeholderBuf;
  const outputBuf  = tensors.get(outputName)?.buf ?? placeholderBuf;
  return [inputBuf.label ?? '', outputBuf.label ?? ''];
}
