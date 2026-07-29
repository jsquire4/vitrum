/**
 * InferenceGraph.ts — WebGPU compute graph executor for the vitrum neural denoiser.
 *
 * Implements the U-Net inference pass specified by `unetArchitecture.ts`.
 *
 * Task 4.5 (Theme I) decomposition: this class is now the DISPATCH ORCHESTRATOR.
 * Two collaborators carry the heavy lifting:
 *   - `tensorDimSolver.ts` — PURE per-layer (H,W,C) solver + uniform/dispatch
 *     packing. The dim map is computed ONCE in initialize() and stored (the perf
 *     fix: the old code recomputed it per-layer per-frame inside run()'s loop).
 *   - `layerResourceAllocator.ts` — all GPU buffer / pipeline / bind-group
 *     allocation.
 *
 * Invariants this executor maintains:
 *
 *   - Skip-connection spatial shapes are verified at initialize() time.
 *   - The inputPacker layer assembles noisyColor+albedo+normals into enc_input.
 *   - Binding order in dispatch matches WGSL declarations
 *     (input=0, weights=1, bias=2, output=3, params=4).
 *   - Uniform buffers are written with actual shape params in initialize() AND
 *     re-written per layer in run() to handle resize. (run() now re-uses the
 *     stored dim map; resize is handled by re-initialize(), which recomputes it.)
 *   - Bind groups capture actual buffer identities. Tensor replacement is
 *     candidate-first through initialize(); an in-place identity change fails
 *     closed instead of silently re-uploading weights or leaking buffers.
 *   - dispose() clears cached bind groups slot-by-slot before destroying buffers,
 *     and 'neural' mode is wired into HybridEngine (see HybridEngine.ts).
 *
 * GPU resource lifecycle:
 *   - initialize(device, weights, W, H): allocate all intermediate tensors +
 *     pipeline objects + bind groups. Write uniform buffers with shape params.
 *   - run(noisyColor, albedo, normals, output): dispatch the inference graph.
 *   - dispose(): destroy all GPU resources.
 *
 * Canonical 1920×1080 f32 telemetry is executable and pinned: 2,326,579,200
 * logical bytes without reuse, 945,561,600 physical tensor bytes across eight
 * slots, and 622,080,000 peak live tensor bytes. Adapter limits are preflighted
 * before allocation and reported with an aspect-correct maximum resolution.
 */

import type { UNetSpec } from './unetArchitecture.js';
import { validateWeightsForSpec, type ModelWeights } from './weights.js';
import {
  type TensorDims,
  preflightTensorDims,
  packLayerUniform,
  dispatchWorkgroupsFor,
} from './tensorDimSolver.js';
import {
  type TensorBuffer,
  type LayerGPUState,
  allocateGraph,
  type AllocatedGraph,
  currentBufKeys,
} from './layerResourceAllocator.js';
import {
  assertNeuralDeviceSupportsGraph,
  type NeuralMemoryTelemetry,
} from './tensorMemoryPlanner.js';
import { withNeuralGpuErrorScopes } from './gpuValidation.js';
import {
  NEURAL_F32_TENSOR_STORAGE,
  resolveNeuralTensorStorage,
  type NeuralTensorStorageContract,
  type NeuralTensorStoragePreference,
} from './tensorPrecision.js';
import { walkaroundNeuralInferenceExtent } from './shapeContract.js';

export type InferenceGraphState = 'idle' | 'initializing' | 'ready' | 'failed' | 'disposed';
interface GraphResourceSnapshot {
  readonly tensors: ReadonlyMap<string, TensorBuffer>;
  readonly layerStates: readonly (LayerGPUState | null)[];
  readonly allocatedBuffers: readonly GPUBuffer[];
}


// ── InferenceGraph ────────────────────────────────────────────────────────────

export class InferenceGraph {

  private _device:       GPUDevice | null = null;
  private _spec:         UNetSpec;
  private _W:            number = 0;
  private _H:            number = 0;
  /** Private zero-padded extent used by the canonical U-Net tensors. */
  private _inferenceW:   number = 0;
  private _inferenceH:   number = 0;

  /** Named tensor buffers allocated during initialize(). */
  private _tensors: Map<string, TensorBuffer> = new Map();

  /** Per-layer GPU state: pipeline + uniform + bind group. */
  private _layerStates: (LayerGPUState | null)[] = [];

  /**
   * Per-layer tensor dimensions, computed ONCE at initialize() and re-used every
   * frame. (Perf fix — the previous code recomputed this map per-layer per-frame
   * inside run()'s dispatch loop, which is O(layers²) work per frame for an
   * identical result. Resize re-runs initialize() which recomputes it.)
   */
  private _tensorDimsMap: Map<string, TensorDims> = new Map();

  /** Placeholder buffer for unused bindings (weights/biases on parameterless layers). */
  private _placeholderBuf: GPUBuffer | null = null;

  /**
   * All GPU buffers allocated by initialize() — tracked so dispose() can
   * destroy them. Includes weights/biases/uniforms/placeholder, plus the
   * input-packer uniform buffer. Tensor buffers live in `_tensors` and are
   * destroyed separately.
   */
  private _allocatedBuffers: GPUBuffer[] = [];

  /** Input-packer compute pipeline (compiled once at initialize). */
  private _inputPackPipeline: GPUComputePipeline | null = null;
  /** Uniform buffer holding the pixelCount for the input packer. */
  private _inputPackUniformBuf: GPUBuffer | null = null;
  /** Exact padded-output crop, used only when logical and inference extents differ. */
  private _outputCropPipeline: GPUComputePipeline | null = null;
  private _outputCropUniformBuf: GPUBuffer | null = null;

  /** Whether initialize() has completed successfully. */
  private _ready = false;

  private _state: InferenceGraphState = 'idle';
  private _generation = 0;
  private _lastFailure: string | null = null;
  private _memoryTelemetry: NeuralMemoryTelemetry | null = null;
  private _tensorStorage: NeuralTensorStorageContract = NEURAL_F32_TENSOR_STORAGE;
  private _tensorStoragePreference: NeuralTensorStoragePreference = 'auto';
  /** Uniform-write call count — exposed for test instrumentation. */
  _uniformWriteCount = 0;

  constructor(spec: UNetSpec) {
    this._spec = spec;
  }

  // ── Public API ─────────────────────────────────────────────────────────────

  get ready(): boolean { return this._ready; }

  get state(): InferenceGraphState { return this._state; }
  get width(): number { return this._W; }
  get height(): number { return this._H; }
  get inferenceWidth(): number { return this._inferenceW; }
  get inferenceHeight(): number { return this._inferenceH; }
  get device(): GPUDevice | null { return this._device; }
  get lastFailure(): string | null { return this._lastFailure; }
  get memoryTelemetry(): NeuralMemoryTelemetry | null { return this._memoryTelemetry; }
  get tensorStorage(): NeuralTensorStorageContract { return this._tensorStorage; }

  owns(device: GPUDevice, width: number, height: number): boolean {
    return this._ready && this._device === device && this._W === width && this._H === height;
  }

  /**
   * Initialize GPU resources for the given resolution.
   *
   * Uniform buffers are written HERE with actual shape params, not just
   * allocated. Must be called before run().
   *
   * Bind groups are built here with current buffer references. If initialize()
   * is called again (resize), all bind groups are rebuilt.
   */
  async initialize(
    device: GPUDevice,
    weights: ModelWeights,
    W: number,
    H: number,
    tensorStoragePreference: NeuralTensorStoragePreference = this._tensorStoragePreference,
  ): Promise<void> {
    if (this._state === 'disposed') {
      throw new Error('[InferenceGraph] cannot initialize a disposed graph');
    }
    validateWeightsForSpec(this._spec, weights);
    const extent = walkaroundNeuralInferenceExtent(W, H);
    const tensorDimsMap = preflightTensorDims(
      this._spec,
      extent.inferenceWidth,
      extent.inferenceHeight,
    );
    const tensorStorage = resolveNeuralTensorStorage(device, weights, tensorStoragePreference);
    assertNeuralDeviceSupportsGraph(
      device,
      this._spec,
      weights,
      tensorDimsMap,
      extent.inferenceWidth,
      extent.inferenceHeight,
      tensorStorage,
    );

    const generation = ++this._generation;
    const hadReadyGeneration = this._ready;
    this._state = 'initializing';
    this._lastFailure = null;

    try {
      const alloc = await withNeuralGpuErrorScopes(
        device,
        `InferenceGraph ${W}x${H} (padded ${extent.inferenceWidth}x${extent.inferenceHeight}) generation ${generation}`,
        () => allocateGraph(
          device,
          this._spec,
          weights,
          W,
          H,
          tensorDimsMap,
          tensorStorage,
          extent.inferenceWidth,
          extent.inferenceHeight,
        ),
        disposeAllocatedGraph,
      );

      if (this._isDisposed() || generation !== this._generation) {
        disposeAllocatedGraph(alloc);
        throw new Error(
          `[InferenceGraph] generation ${generation} was superseded before publication`,
        );
      }

      const previous: GraphResourceSnapshot = {
        tensors: this._tensors,
        layerStates: this._layerStates,
        allocatedBuffers: this._allocatedBuffers,
      };

      this._device               = device;
      this._W                    = W;
      this._H                    = H;
      this._inferenceW           = extent.inferenceWidth;
      this._inferenceH           = extent.inferenceHeight;
      this._tensorDimsMap        = tensorDimsMap;
      this._tensors              = alloc.tensors;
      this._layerStates          = alloc.layerStates;
      this._placeholderBuf       = alloc.placeholderBuf;
      this._inputPackPipeline    = alloc.inputPackPipeline;
      this._inputPackUniformBuf  = alloc.inputPackUniformBuf;
      this._outputCropPipeline   = alloc.outputCropPipeline;
      this._outputCropUniformBuf = alloc.outputCropUniformBuf;
      this._allocatedBuffers     = alloc.allocatedBuffers;
      this._uniformWriteCount    = alloc.uniformWriteCount;
      this._memoryTelemetry      = alloc.memoryTelemetry;
      this._ready                = true;
      this._tensorStorage          = tensorStorage;
      this._tensorStoragePreference = tensorStoragePreference;
      this._state                = 'ready';
      this._lastFailure          = null;

      disposeGraphResourceSnapshot(previous);
    } catch (error) {
      if (generation === this._generation && !this._isDisposed()) {
        this._lastFailure = errorMessage(error);
        if (hadReadyGeneration && this._ready) {
          this._state = 'ready';
        } else {
          this._ready = false;
          this._state = 'failed';
        }
      }
      throw error;
    }
  }

  /**
   * Run the inference graph.
   *
   * @param noisyColorBuf  GPU buffer containing noisy RGB (H×W×3, f32).
   * @param albedoBuf      GPU buffer containing albedo (H×W×3, f32).
   * @param normalsBuf     GPU buffer containing world normals (H×W×3, f32).
   * @param outputBuf      GPU buffer to receive denoised RGB (H×W×3, f32).
   *
   * The three input buffers are packed into enc_input on GPU via the
   * inputPacker dispatch. Then the graph is dispatched layer by layer.
   *
   * Resize publishes a complete candidate graph through initialize(). A tensor
   * identity change inside one published graph is an invariant violation.
   */
  run(
    noisyColorBuf: GPUBuffer,
    albedoBuf:     GPUBuffer,
    normalsBuf:    GPUBuffer,
    outputBuf:     GPUBuffer,
    commandEncoder?: GPUCommandEncoder,
  ): void {
    if (!this._ready || !this._device) {
      throw new Error('[InferenceGraph] not initialized — call initialize() first');
    }

    if (this._state !== 'ready') {
      throw new Error(`[InferenceGraph] cannot dispatch while state=${this._state}`);
    }
    const rgbBytes = this._W * this._H * 3 * this._tensorStorage.bytesPerScalar;
    assertBufferCapacity(noisyColorBuf, rgbBytes, 'noisyColor');
    assertBufferCapacity(albedoBuf, rgbBytes, 'albedo');
    assertBufferCapacity(normalsBuf, rgbBytes, 'normals');
    assertBufferCapacity(outputBuf, rgbBytes, 'output');
    if (this._placeholderBuf == null) {
      throw new Error('[InferenceGraph] ready graph is missing its placeholder buffer');
    }
    const device = this._device;
    const enc = commandEncoder ?? device.createCommandEncoder({ label: 'neural-inference' });
    const tensorDimsMap = this._tensorDimsMap;

    // Pack noisyColor (3ch) + albedo (3ch) + normals (3ch) into enc_input (9ch)
    // via the inputPacker compute pass. INTERLEAVED layout: for each pixel p,
    //   enc_input[p*9..p*9+3] = noisyColor[p*3..p*3+3],
    //   enc_input[p*9+3..p*9+6] = albedo[p*3..p*3+3],
    //   enc_input[p*9+6..p*9+9] = normals[p*3..p*3+3].
    this._runInputPack(enc, noisyColorBuf, albedoBuf, normalsBuf);

    // ── Dispatch each layer ───────────────────────────────────────────────
    const N = this._spec.layers.length;
    for (let i = 0; i < N; i++) {
      const layer = this._spec.layers[i]!;
      const state = this._layerStates[i];

      if (!state) continue; // inputPack is dispatched separately

      // Re-validate bind group buffer identity.
      const curKeys = currentBufKeys(layer, this._tensors, this._placeholderBuf);
      if (!keysEqual(state.cachedBufKeys, curKeys)) {
        throw new Error(
          `[InferenceGraph] layer '${layer.name}' tensor identity changed ` +
            'outside candidate-first initialize(); reinitialize the graph.',
        );
      }

      // Re-write uniform with current (stored) dims (handles resize).
      device.queue.writeBuffer(
        state.uniformBuf, 0,
        packLayerUniform(
          layer,
          tensorDimsMap,
          this._inferenceH,
          this._inferenceW,
          device.limits.maxComputeWorkgroupsPerDimension,
        ),
      );
      this._uniformWriteCount++;

      // Dispatch.
      const bindGroup = state.cachedBindGroup;
      if (bindGroup == null) {
        throw new Error(`[InferenceGraph] layer '${layer.name}' has no bind group`);
      }
      const pass = enc.beginComputePass({ label: `neural-${layer.name}` });
      pass.setPipeline(state.pipeline);
      pass.setBindGroup(0, bindGroup);

      const outDims = tensorDimsMap.get(layer.output);
      if (outDims == null) throw new Error(`[InferenceGraph] missing dimensions for '${layer.output}'`);
      const [gx, gy, gz] = dispatchWorkgroupsFor(
        layer.kind,
        outDims,
        device.limits.maxComputeWorkgroupsPerDimension,
      );
      pass.dispatchWorkgroups(gx, gy, gz);
      pass.end();
    }

    // Copy final 'denoised' tensor to the output buffer.
    const denoisedTensor = this._tensors.get('denoised');
    if (denoisedTensor == null) throw new Error("[InferenceGraph] ready graph is missing 'denoised'");
    if (this._W === this._inferenceW && this._H === this._inferenceH) {
      enc.copyBufferToBuffer(
        denoisedTensor.buf, 0,
        outputBuf, 0,
        denoisedTensor.dims.H * denoisedTensor.dims.W * denoisedTensor.dims.C * this._tensorStorage.bytesPerScalar,
      );
    } else {
      this._runOutputCrop(enc, denoisedTensor.buf, outputBuf);
    }

    if (!commandEncoder) {
      device.queue.submit([enc.finish()]);
    }
  }

  /**
   * Dispose all GPU resources.
   *
   * Cached bind groups are cleared slot-by-slot before destroying underlying
   * buffers, ensuring GPUBindGroups release their buffer references before the
   * buffers are destroyed.
   *
   * Every buffer tracked in `_allocatedBuffers` (weights, biases, layer
   * uniforms, input-packer uniform, placeholder) is destroyed here.
   */
  dispose(): void {
    if (this._state === 'disposed') return;
    this._generation++;
    this._state = 'disposed';
    disposeGraphResourceSnapshot({
      tensors: this._tensors,
      layerStates: this._layerStates,
      allocatedBuffers: this._allocatedBuffers,
    });
    this._layerStates = new Array(this._spec.layers.length).fill(null) as null[];
    this._tensors = new Map();
    this._tensorDimsMap = new Map();
    this._allocatedBuffers = [];
    this._placeholderBuf = null;
    this._inputPackPipeline = null;
    this._inputPackUniformBuf = null;
    this._outputCropPipeline = null;
    this._outputCropUniformBuf = null;
    this._inferenceW = 0;
    this._inferenceH = 0;
    this._device = null;
    this._ready = false;
    this._memoryTelemetry = null;
    this._tensorStorage = NEURAL_F32_TENSOR_STORAGE;
    this._tensorStoragePreference = 'auto';
  }

  // ── Private helpers ────────────────────────────────────────────────────────


  private _isDisposed(): boolean {
    return this._state === 'disposed';
  }
  /**
   * GPU-side input packing pass.
   *
   * Packs noisyColor (H×W×3) + albedo (H×W×3) + normals (H×W×3) into
   * enc_input (H×W×9) with the per-pixel INTERLEAVED layout that
   * `unetArchitecture.ts` and the downstream conv2d kernels expect:
   *
   *   enc_input[p*9+0..2] = noisyColor[p*3+0..2]
   *   enc_input[p*9+3..5] = albedo[p*3+0..2]
   *   enc_input[p*9+6..8] = normals[p*3+0..2]
   *
   * The compute pipeline is compiled once at `initialize()`. The bind group is
   * rebuilt each frame because the three input buffers are supplied per-call and
   * their identities are not stable across frames.
   */
  private _runInputPack(
    enc: GPUCommandEncoder,
    noisyColorBuf: GPUBuffer,
    albedoBuf:     GPUBuffer,
    normalsBuf:    GPUBuffer,
  ): void {
    const encInputTensor = this._tensors.get('enc_input');
    if (encInputTensor == null) {
      throw new Error("[InferenceGraph] ready graph is missing 'enc_input'");
    }
    if (this._inputPackPipeline == null || this._inputPackUniformBuf == null) {
      throw new Error('[InferenceGraph] ready graph is missing input-pack resources');
    }
    if (this._device == null) {
      throw new Error('[InferenceGraph] ready graph lost its device');
    }

    const device = this._device;
    const pixelCount = this._inferenceH * this._inferenceW;

    const bindGroup = device.createBindGroup({
      label: 'neural-bg-inputPack',
      layout: this._inputPackPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: noisyColorBuf } },
        { binding: 1, resource: { buffer: albedoBuf } },
        { binding: 2, resource: { buffer: normalsBuf } },
        { binding: 3, resource: { buffer: encInputTensor.buf } },
        { binding: 4, resource: { buffer: this._inputPackUniformBuf } },
      ],
    });

    const pass = enc.beginComputePass({ label: 'neural-inputPack' });
    pass.setPipeline(this._inputPackPipeline);
    pass.setBindGroup(0, bindGroup);
    // Workgroup size in INPUT_PACKER_WGSL is 256×1×1.
    pass.dispatchWorkgroups(Math.ceil(pixelCount / 256), 1, 1);
    pass.end();
  }

  private _runOutputCrop(
    enc: GPUCommandEncoder,
    paddedInput: GPUBuffer,
    logicalOutput: GPUBuffer,
  ): void {
    if (this._device == null
        || this._outputCropPipeline == null
        || this._outputCropUniformBuf == null) {
      throw new Error('[InferenceGraph] ready graph is missing output-crop resources');
    }
    const bindGroup = this._device.createBindGroup({
      label: 'neural-bg-outputCrop',
      layout: this._outputCropPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: paddedInput } },
        { binding: 1, resource: { buffer: logicalOutput } },
        { binding: 2, resource: { buffer: this._outputCropUniformBuf } },
      ],
    });
    const pass = enc.beginComputePass({ label: 'neural-outputCrop' });
    pass.setPipeline(this._outputCropPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil((this._W * this._H) / 256), 1, 1);
    pass.end();
  }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function keysEqual(a: readonly GPUBuffer[], b: readonly GPUBuffer[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}
function disposeAllocatedGraph(graph: AllocatedGraph): void {
  disposeGraphResourceSnapshot(graph);
}

function disposeGraphResourceSnapshot(snapshot: GraphResourceSnapshot): void {
  const buffers = new Set<GPUBuffer>(snapshot.allocatedBuffers);
  for (const state of snapshot.layerStates) {
    if (state != null) state.cachedBindGroup = null;
  }
  for (const tensor of snapshot.tensors.values()) buffers.add(tensor.buf);
  for (const buffer of buffers) {
    try { buffer.destroy(); } catch { /* teardown is best-effort and idempotent */ }
  }
}

function assertBufferCapacity(buffer: GPUBuffer, expectedBytes: number, label: string): void {
  const size = (buffer as GPUBuffer & { readonly size?: number }).size;
  if (typeof size === 'number' && size < expectedBytes) {
    throw new RangeError(
      `[InferenceGraph] ${label} buffer size ${size} is smaller than required ${expectedBytes}`,
    );
  }
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
