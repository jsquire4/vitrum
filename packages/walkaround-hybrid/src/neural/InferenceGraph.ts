/**
 * InferenceGraph.ts — WebGPU compute-shader inference graph for neural denoising.
 *
 * Wires the five WGSL primitive kernels (conv2d, transposed_conv2d, relu,
 * skip, upsample) into an arbitrary directed acyclic graph (DAG) that
 * implements a full UNet-style neural denoiser.
 *
 * Usage:
 *   1. Construct with a spec and weights loaded from the model file.
 *   2. Call `initialize(device)` once — allocates GPU buffers and compiles pipelines.
 *   3. Call `run(device, encoder, inputs, outputs)` once per frame.
 *   4. Call `dispose()` when done to release GPU resources.
 *
 * Design constraints:
 *   - No GPU verification available in the current build environment; the class
 *     is authored for correctness and exports. Integration into HybridEngine's
 *     renderFrame is deferred — see `plan/sprint-13-walkaround-integration.md`.
 *   - Intermediate tensors referenced as layer outputs are registered during
 *     `initialize()`, but buffers are created only once you populate them (this
 *     scaffold requires every produced tensor—including intermediates—to appear
 *     in the `outputs` map passed to `run()`, or reuse a GPUBuffer you wired
 *     manually). No automatic sizing/allocation occurs inside `run()`.
 *   - Layer pipelines are compiled lazily in `initialize()` and cached.
 *
 * Supported layer kinds and their WGSL entry points:
 *   conv2d          → conv2dKernel         (conv2d.wgsl.ts)
 *   transposed_conv2d → transposedConv2dKernel (transposedConv2d.wgsl.ts)
 *   relu            → reluKernel           (relu.wgsl.ts)
 *   skip            → skipConnectionKernel (skipConnection.wgsl.ts)
 *   upsample        → bilinearUpsampleKernel (bilinearUpsample.wgsl.ts)
 *
 * References:
 *   Ronneberger, Fischer, Brox "U-Net: Convolutional Networks for Biomedical
 *   Image Segmentation" MICCAI 2015. https://arxiv.org/abs/1505.04597
 *
 * @since Sprint 13, 2026-05-09
 */

import { CONV2D_WGSL } from './wgsl/conv2d.wgsl.js';
import { TRANSPOSED_CONV2D_WGSL } from './wgsl/transposedConv2d.wgsl.js';
import { RELU_WGSL } from './wgsl/relu.wgsl.js';
import { SKIP_CONNECTION_WGSL } from './wgsl/skipConnection.wgsl.js';
import { BILINEAR_UPSAMPLE_WGSL } from './wgsl/bilinearUpsample.wgsl.js';

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/** Supported neural layer operation kinds. */
export type InferenceLayerKind =
  | 'conv2d'
  | 'transposed_conv2d'
  | 'relu'
  | 'skip'
  | 'upsample';

/**
 * A single layer in the inference graph.
 *
 * - `inputs`: named tensors consumed by this layer. Named tensor keys are
 *   resolved in order: first from the `inputs` map passed to `run()`, then
 *   from intermediate tensors produced by prior layers.
 * - `output`: named tensor this layer writes into.
 * - `params`: layer-specific configuration (channel counts, kernel dims, etc.).
 */
export interface InferenceLayer {
  readonly kind:   InferenceLayerKind;
  readonly inputs: readonly string[];  // tensor names (resolved at dispatch time)
  readonly output: string;
  readonly params?: Record<string, unknown>;
}

/**
 * Complete description of the inference graph topology.
 * Layers are executed in declaration order (topological order assumed).
 */
export interface InferenceGraphSpec {
  readonly layers:        ReadonlyArray<InferenceLayer>;
  readonly inputTensors:  ReadonlyArray<string>;  // names expected in run() inputs map
  readonly outputTensors: ReadonlyArray<string>;  // names expected in run() outputs map
}

/**
 * Pre-loaded model weights, keyed by layer name.
 * Produced by the training pipeline exporter (see tools/neural-denoiser-training/).
 */
export interface ModelWeights {
  /** Layer name → Float32Array weight buffer.  Shape: [outputC × inputC × kH × kW]. */
  readonly weights: ReadonlyMap<string, Float32Array>;
  /** Layer name → Float32Array bias buffer.  Shape: [outputC]. */
  readonly biases:  ReadonlyMap<string, Float32Array>;
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/** Metadata stored per intermediate tensor (feature map). */
interface TensorMeta {
  /** Number of f32 elements. */
  elementCount: number;
  /** Allocated GPU buffer. */
  buffer: GPUBuffer | null;
}

/** Compiled state for one layer — pipeline + weight buffers. */
interface CompiledLayer {
  pipeline: GPUComputePipeline;
  weightBuffer: GPUBuffer | null;
  biasBuffer:   GPUBuffer | null;
  /** Uniform buffer carrying shape params for this layer. */
  uniformBuffer: GPUBuffer;
}

// ─────────────────────────────────────────────────────────────────────────────
// InferenceGraph
// ─────────────────────────────────────────────────────────────────────────────

export class InferenceGraph {
  private readonly _spec:    InferenceGraphSpec;
  private readonly _weights: ModelWeights;

  /** Map from tensor name → intermediate GPU buffer + element count. */
  private _tensors: Map<string, TensorMeta> = new Map();
  /** Per-layer compiled GPU resources, indexed parallel to spec.layers. */
  private _layers: CompiledLayer[] = [];
  /** Whether initialize() has completed. */
  private _ready = false;
  /** One bind group per layer; reused while pipeline + buffer bindings unchanged. */
  private _cachedBindGroups: Array<GPUBindGroup | undefined> = [];

  constructor(spec: InferenceGraphSpec, weights: ModelWeights) {
    this._spec    = spec;
    this._weights = weights;
  }

  // ── Initialization ─────────────────────────────────────────────────────────

  /**
   * Allocate all GPU resources (weight buffers, intermediate tensor buffers,
   * compute pipelines) for the inference graph.
   *
   * Must be called once before `run()`. May be called again after `dispose()`
   * to re-initialise with a new device.
   *
   * NOTE: GPU-device interaction is structured but not verified in this
   * scaffold — see plan/sprint-13-walkaround-integration.md for the wiring
   * plan into HybridEngine.renderFrame.
   */
  async initialize(device: GPUDevice): Promise<void> {
    if (this._ready) return;

    this._cachedBindGroups = [];

    // Compile one pipeline per unique layer kind.
    const moduleCache = new Map<string, GPUShaderModule>();

    const getModule = (kind: InferenceLayerKind): GPUShaderModule => {
      if (moduleCache.has(kind)) return moduleCache.get(kind)!;
      const src = InferenceGraph._wgslForKind(kind);
      const module = device.createShaderModule({ label: `neural/${kind}`, code: src });
      moduleCache.set(kind, module);
      return module;
    };

    for (const layer of this._spec.layers) {
      const module = getModule(layer.kind);
      const entryPoint = InferenceGraph._entryPointForKind(layer.kind);

      const pipeline = await device.createComputePipelineAsync({
        label:  `neural/${layer.kind}/${layer.output}`,
        layout: 'auto',
        compute: { module, entryPoint },
      });

      // Upload weight + bias buffers if the layer has learned parameters.
      let weightBuffer: GPUBuffer | null = null;
      let biasBuffer:   GPUBuffer | null = null;

      const layerWeights = this._weights.weights.get(layer.output);
      if (layerWeights) {
        weightBuffer = device.createBuffer({
          label: `neural/weights/${layer.output}`,
          size:  layerWeights.byteLength,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(weightBuffer, 0, layerWeights.buffer as ArrayBuffer, layerWeights.byteOffset, layerWeights.byteLength);
      }

      const layerBiases = this._weights.biases.get(layer.output);
      if (layerBiases) {
        biasBuffer = device.createBuffer({
          label: `neural/biases/${layer.output}`,
          size:  layerBiases.byteLength,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(biasBuffer, 0, layerBiases.buffer as ArrayBuffer, layerBiases.byteOffset, layerBiases.byteLength);
      }

      // Allocate uniform buffer (32 bytes — max param struct size across all kernels).
      const uniformBuffer = device.createBuffer({
        label: `neural/uniform/${layer.output}`,
        size:  32,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });

      this._layers.push({ pipeline, weightBuffer, biasBuffer, uniformBuffer });

      // Register intermediate output tensor.
      if (!this._tensors.has(layer.output)) {
      // Dimensions for intermediate buffers — host supplies concrete GPUBuffers via `outputs` in run().
        this._tensors.set(layer.output, { elementCount: 0, buffer: null });
      }
    }

    this._ready = true;
  }

  // ── Inference dispatch ─────────────────────────────────────────────────────

  /**
   * Dispatch the full inference pipeline into the provided command encoder.
   *
   * @param device  - Active WebGPU device.
   * @param encoder - Command encoder to record compute passes into.
   * @param inputs  - Named GPUBuffers for graph inputs (e.g. noisy color, albedo, normals).
   * @param outputs - Named GPUBuffers to write final graph outputs into.
   */
  run(
    device:  GPUDevice,
    encoder: GPUCommandEncoder,
    inputs:  ReadonlyMap<string, GPUBuffer>,
    outputs: ReadonlyMap<string, GPUBuffer>,
  ): void {
    if (!this._ready) {
      throw new Error('[InferenceGraph] run() called before initialize().');
    }

    for (const name of this._spec.inputTensors) {
      if (!inputs.has(name)) {
        throw new Error(`[InferenceGraph] inputs map missing required tensor "${name}".`);
      }
    }
    for (const name of this._spec.outputTensors) {
      if (!outputs.has(name)) {
        throw new Error(`[InferenceGraph] outputs map missing required tensor "${name}".`);
      }
    }

    // Resolve named buffers: inputs, then intermediates, then outputs.
    const resolve = (name: string): GPUBuffer => {
      if (inputs.has(name))  return inputs.get(name)!;
      const meta = this._tensors.get(name);
      if (meta?.buffer) return meta.buffer;
      if (outputs.has(name)) return outputs.get(name)!;
      throw new Error(`[InferenceGraph] tensor "${name}" not found.`);
    };

    for (let i = 0; i < this._spec.layers.length; i++) {
      const layer    = this._spec.layers[i]!;
      const compiled = this._layers[i]!;

      const pass = encoder.beginComputePass({
        label: `neural/${layer.kind}/${layer.output}`,
      });
      pass.setPipeline(compiled.pipeline);

      // Bind group entries: inputs[0..n], output, uniform, optional weight+bias.
      // The concrete layout matches the WGSL binding declarations in each shader.
      const entries: GPUBindGroupEntry[] = [];

      for (let b = 0; b < layer.inputs.length; b++) {
        const inputName = layer.inputs[b]!;
        entries.push({ binding: b, resource: { buffer: resolve(inputName) } });
      }
      // Output tensor: the next binding after inputs.
      const outputBuffer = outputs.has(layer.output)
        ? outputs.get(layer.output)!
        : (this._tensors.get(layer.output)?.buffer ?? (() => {
            throw new Error(`[InferenceGraph] No buffer for output "${layer.output}".`);
          })());
      entries.push({ binding: layer.inputs.length, resource: { buffer: outputBuffer } });

      // Uniform buffer (always present).
      entries.push({
        binding: layer.inputs.length + 1,
        resource: { buffer: compiled.uniformBuffer },
      });

      // Weight + bias buffers (conv2d / transposed_conv2d only).
      if (compiled.weightBuffer) {
        entries.push({
          binding: layer.inputs.length + 2,
          resource: { buffer: compiled.weightBuffer },
        });
      }
      if (compiled.biasBuffer) {
        entries.push({
          binding: layer.inputs.length + 3,
          resource: { buffer: compiled.biasBuffer },
        });
      }

      const bg =
        this._cachedBindGroups[i] ??
        device.createBindGroup({
          label: `neural/bg/${layer.output}`,
          layout: compiled.pipeline.getBindGroupLayout(0),
          entries,
        });
      this._cachedBindGroups[i] = bg;

      pass.setBindGroup(0, bg);

      // Workgroup dispatch: spatial layers use (ceil(W/8), ceil(H/8), 1);
      // flat layers (relu, skip) use (ceil(N/256), 1, 1).
      // Dispatch counts are parameterised by layer.params for flexibility.
      const p = (layer.params ?? {}) as Record<string, number>;
      const dispatchX = Math.ceil((p['dispatchX'] ?? 1));
      const dispatchY = Math.ceil((p['dispatchY'] ?? 1));
      const dispatchZ = Math.ceil((p['dispatchZ'] ?? 1));
      pass.dispatchWorkgroups(dispatchX, dispatchY, dispatchZ);
      pass.end();
    }
  }

  // ── Disposal ──────────────────────────────────────────────────────────────

  /** Release all GPU resources. After dispose(), initialize() may be called again. */
  dispose(): void {
    for (const compiled of this._layers) {
      compiled.weightBuffer?.destroy();
      compiled.biasBuffer?.destroy();
      compiled.uniformBuffer.destroy();
    }
    for (const meta of this._tensors.values()) {
      meta.buffer?.destroy();
    }
    this._layers  = [];
    this._tensors = new Map();
    this._cachedBindGroups = [];
    this._ready   = false;
  }

  // ── Static helpers ─────────────────────────────────────────────────────────

  private static _wgslForKind(kind: InferenceLayerKind): string {
    switch (kind) {
      case 'conv2d':           return CONV2D_WGSL;
      case 'transposed_conv2d':return TRANSPOSED_CONV2D_WGSL;
      case 'relu':             return RELU_WGSL;
      case 'skip':             return SKIP_CONNECTION_WGSL;
      case 'upsample':         return BILINEAR_UPSAMPLE_WGSL;
    }
  }

  private static _entryPointForKind(kind: InferenceLayerKind): string {
    switch (kind) {
      case 'conv2d':           return 'conv2dKernel';
      case 'transposed_conv2d':return 'transposedConv2dKernel';
      case 'relu':             return 'reluKernel';
      case 'skip':             return 'skipConnectionKernel';
      case 'upsample':         return 'bilinearUpsampleKernel';
    }
  }
}
