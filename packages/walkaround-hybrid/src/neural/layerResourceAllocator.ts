/**
 * layerResourceAllocator.ts — GPU-resource allocation for the U-Net inference
 * graph: intermediate tensor buffers, per-layer compute pipelines, weight/bias
 * buffers, uniform buffers, and bind groups.
 *
 * Extracted from InferenceGraph (Task 4.5 Theme I) so the executor itself is
 * just the dispatch orchestrator. All `device.createBuffer` /
 * `createComputePipelineAsync` / `createBindGroup` calls for the graph layers
 * live here. Behaviorally identical to the pre-extraction inline code — the
 * buffer labels, sizes, usages, bind-group entry order, and pipeline entry
 * points are unchanged (pinned by `__tests__/inferenceGraphDimsOnce.test.ts`
 * and the existing neural tests).
 */

import type { UNetSpec, LayerSpec, LayerKind } from './unetArchitecture.js';
import type { ModelWeights, LayerWeights } from './weights.js';
import { CONV2D_WGSL } from './wgsl/conv2d.wgsl.js';
import { TRANSPOSED_CONV2D_WGSL } from './wgsl/transposedConv2d.wgsl.js';
import { RELU_WGSL } from './wgsl/relu.wgsl.js';
import { SKIP_CONNECTION_WGSL } from './wgsl/skipConnection.wgsl.js';
import { BILINEAR_UPSAMPLE_WGSL } from './wgsl/bilinearUpsample.wgsl.js';
import { INPUT_PACKER_WGSL, INPUT_PACKER_ENTRY } from './inputPacker.js';
import {
  type TensorDims,
  validateSkipShapes,
  packLayerUniform,
} from './tensorDimSolver.js';

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

// NOTE — bilinearUpsample is a fully-plumbed EXTENSION POINT (Task 4.5 D2). No
// canonical UNetSpec currently emits a bilinearUpsample layer (the decoder uses
// transposedConv2d), but the kind is wired end-to-end (entry point, WGSL source,
// dim solver, dispatch layout) so an alternative decoder upsampler can be slotted
// in by emitting the layer in a custom spec. Do NOT delete it as "dead".
const WGSL_ENTRY: Record<LayerKind, string> = {
  conv2d:          'conv2dMain',
  transposedConv2d:'transposedConv2dMain',
  relu:            'reluMain',
  skipAdd:         'skipConnectionMain',
  bilinearUpsample:'bilinearUpsampleMain',  // extension point — see note above
  inputPack:       '',  // handled CPU-side by packing pass
};

const WGSL_SOURCE: Partial<Record<LayerKind, string>> = {
  conv2d:          CONV2D_WGSL,
  transposedConv2d:TRANSPOSED_CONV2D_WGSL,
  relu:            RELU_WGSL,
  skipAdd:         SKIP_CONNECTION_WGSL,
  bilinearUpsample:BILINEAR_UPSAMPLE_WGSL,  // extension point — see note above
};

// Uniform buffer size: 5 u32 fields, padded to 32 bytes (8×u32).
const UNIFORM_BUF_BYTES = 32;

// Placeholder buffer size for unused bindings (weights/biases on relu/skip).
const PLACEHOLDER_BYTES = 4;

/** Everything the InferenceGraph executor needs after allocation. */
export interface AllocatedGraph {
  tensors: Map<string, TensorBuffer>;
  layerStates: (LayerGPUState | null)[];
  placeholderBuf: GPUBuffer;
  inputPackPipeline: GPUComputePipeline;
  inputPackUniformBuf: GPUBuffer;
  /** All non-tensor buffers (weights/biases/uniforms/placeholder/input-pack
   *  uniform) tracked so dispose() can destroy them. */
  allocatedBuffers: GPUBuffer[];
  /** Weight lookup retained for bind-group rebuild on resize. */
  weightsByName: Map<string, LayerWeights>;
  /** Uniform-write count incurred during allocation (init-time writes). */
  uniformWriteCount: number;
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
): Promise<AllocatedGraph> {
  const allocatedBuffers: GPUBuffer[] = [];
  let uniformWriteCount = 0;

  // Build the weight lookup by name (retained for bind-group rebuild in run()).
  const weightsByName = new Map<string, LayerWeights>(
    weights.layers.map(lw => [lw.name, lw]),
  );

  // Placeholder buffer (for unused binding slots).
  const placeholderBuf = device.createBuffer({
    label: 'neural/placeholder',
    size: PLACEHOLDER_BYTES,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.UNIFORM,
  });
  allocatedBuffers.push(placeholderBuf);

  // ── Compile the input-packer compute pipeline once.
  const packModule = device.createShaderModule({
    label: 'neural/inputPacker',
    code: INPUT_PACKER_WGSL,
  });
  const inputPackPipeline = await device.createComputePipelineAsync({
    label: 'neural-pipeline-inputPack',
    layout: 'auto',
    compute: { module: packModule, entryPoint: INPUT_PACKER_ENTRY },
  });

  // Uniform buffer for input packer: holds the per-frame pixelCount (H*W).
  const inputPackUniformBuf = device.createBuffer({
    label: 'neural-uniform-inputPack',
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  allocatedBuffers.push(inputPackUniformBuf);
  {
    const u32 = new Uint32Array(4);
    u32[0] = H * W;
    device.queue.writeBuffer(inputPackUniformBuf, 0, u32.buffer);
  }

  // ── Allocate intermediate tensors ─────────────────────────────────────
  const tensors = new Map<string, TensorBuffer>();
  for (const [name, dims] of tensorDimsMap) {
    const floatCount = dims.H * dims.W * dims.C;
    const buf = device.createBuffer({
      label: `neural/${name}`,
      size: floatCount * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    tensors.set(name, { buf, dims, label: `neural/${name}` });
  }

  // Verify skip-connection shapes (both operands must match H×W×C).
  validateSkipShapes(spec, tensorDimsMap);

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

    const sm = device.createShaderModule({ label: `neural-${layer.name}`, code: wgsl });
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
    device.queue.writeBuffer(uniformBuf, 0, packLayerUniform(layer, tensorDimsMap, H, W));
    uniformWriteCount++;

    // H28 — ReLU in-place aliasing fix (see patchReLUInPlaceAliasing): a relu
    // layer whose input name == output name would alias the same GPU buffer at
    // binding 0 (read) and binding 3 (read_write). Falls through to the normal
    // path when the input tensor is missing (returns null).
    if (layer.kind === 'relu' && layer.inputs[0] === layer.output) {
      const patched = patchReLUInPlaceAliasing(
        device, pipeline, layer, weightsByName, uniformBuf,
        tensors, placeholderBuf, allocatedBuffers,
      );
      if (patched) {
        layerStates[i] = patched;
        continue;
      }
    }

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
    allocatedBuffers,
    weightsByName,
    uniformWriteCount,
  };
}

/**
 * H28 — ReLU in-place aliasing fix (D7.9: extracted from the allocateGraph loop).
 * When a relu layer lists the same name as both input and output (e.g.
 * `inputs: ['enc1_feat'], output: 'enc1_feat'`), `buildBindGroup` would
 * assign the SAME GPU buffer to binding 0 (read) AND binding 3 (read_write),
 * which is undefined behavior in WebGPU (aliased storage bindings with
 * mixed access modes). Fix: allocate a distinct `${layer.name}_out` buffer
 * for the relu output, build the bind group with that as binding 3, then
 * remap `tensors` so downstream layers reading `layer.output` see the
 * relu-written buffer. This is host-only; no WGSL changes are required.
 *
 * Tensor-map mutations are identical to the previous inline block:
 *   • `${layer.name}_out`     → the new distinct relu output buffer
 *   • `${layer.name}_in_orig` → the original input tensor (leak guard: the
 *     remap below overwrites the only reference to the relu's binding-0 INPUT
 *     buffer — the upstream conv's output. Without preserving it, dispose()'s
 *     tensor-map loop would never destroy it → GPU memory leak on engine
 *     teardown, 7 such buffers in the default UNet spec. The key is never read
 *     by any layer's bind-group build.)
 *   • `inName`                → remapped to the relu-written buffer so
 *     downstream readers of `layer.output` see it.
 * The new output buffer is deliberately NOT pushed to allocatedBuffers — it
 * lives in `tensors` and is destroyed via the tensors cleanup, consistent with
 * other tensors.
 *
 * Returns the layer's GPU state, or null when the input tensor is missing
 * (caller falls through to the unpatched buildBindGroup path).
 */
function patchReLUInPlaceAliasing(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  layer: LayerSpec,
  weightsByName: Map<string, LayerWeights>,
  uniformBuf: GPUBuffer,
  tensors: Map<string, TensorBuffer>,
  placeholderBuf: GPUBuffer,
  allocatedBuffers: GPUBuffer[],
): LayerGPUState | null {
  const inName = layer.inputs[0]!;
  const outKey = `${layer.name}_out`;
  const srcTb  = tensors.get(inName);
  if (!srcTb) return null;

  const floatCount = srcTb.dims.H * srcTb.dims.W * srcTb.dims.C;
  const outBuf = device.createBuffer({
    label: `neural/${outKey}`,
    size: floatCount * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const outTb: TensorBuffer = { buf: outBuf, dims: srcTb.dims, label: `neural/${outKey}` };
  // Temporarily inject the distinct output tensor so buildBindGroup picks
  // it up as binding 3 while keeping the original for binding 0.
  tensors.set(outKey, outTb);
  // Swap the output name → outKey for this layer's bind-group build.
  const patchedLayer = { ...layer, output: outKey };
  const { bindGroup, bufKeys } = buildBindGroup(
    device, pipeline, patchedLayer, weightsByName, uniformBuf,
    tensors, placeholderBuf, allocatedBuffers,
  );
  tensors.set(`${layer.name}_in_orig`, srcTb); // leak guard (see doc comment)
  tensors.set(inName, outTb);                  // downstream remap
  return {
    layerName:      layer.name,
    pipeline,
    uniformBuf,
    cachedBindGroup: bindGroup,
    cachedBufKeys:   bufKeys,
  };
}

/**
 * Build a bind group for a layer.
 * Binding layout matches WGSL declarations exactly:
 *   0=input, 1=weights (or inputB for skip), 2=biases, 3=output, 4=params
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
      new Float32Array(weightsBuf.getMappedRange()).set(lw.weights);
      weightsBuf.unmap();
      allocatedBuffers.push(weightsBuf);
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
      new Float32Array(biasesBuf.getMappedRange()).set(lw.biases);
      biasesBuf.unmap();
      allocatedBuffers.push(biasesBuf);
    }
  }

  // Output buffer — binding 3.
  const outputName = layer.output;
  const outputTensor = tensors.get(outputName);
  const outputBuf = outputTensor?.buf ?? placeholderBuf;

  const bindGroup = device.createBindGroup({
    label: `neural-bg-${layer.name}`,
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: inputBuf  } },
      { binding: 1, resource: { buffer: weightsBuf } },
      { binding: 2, resource: { buffer: biasesBuf  } },
      { binding: 3, resource: { buffer: outputBuf  } },
      { binding: 4, resource: { buffer: uniformBuf } },
    ],
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
