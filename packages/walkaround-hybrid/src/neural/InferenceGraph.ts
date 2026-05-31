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
 *   - Bind-group cache is keyed by buffer identity (label) and invalidated on any
 *     buffer swap (device resize).
 *   - dispose() clears cached bind groups slot-by-slot before destroying buffers,
 *     and 'neural' mode is wired into HybridEngine (see HybridEngine.ts).
 *
 * GPU resource lifecycle:
 *   - initialize(device, weights, W, H): allocate all intermediate tensors +
 *     pipeline objects + bind groups. Write uniform buffers with shape params.
 *   - run(noisyColor, albedo, normals, output): dispatch the inference graph.
 *   - dispose(): destroy all GPU resources.
 *
 * Memory budget (f32, 1080×1920):
 *   enc_input (H×W×9):      ~71 MB
 *   enc1_feat (H×W×24):    ~199 MB
 *   enc1_out (H/2×W/2×24):  ~50 MB
 *   ...total intermediates:  ~461 MB (f32) → use f16 in production.
 *
 * For a hosted GPU test this is large; the smoke test uses 32×32 which is ~4 MB total.
 */

import type { UNetSpec } from './unetArchitecture.js';
import type { ModelWeights, LayerWeights } from './weights.js';
import {
  type TensorDims,
  computeTensorDims,
  packLayerUniform,
  dispatchWorkgroupsFor,
} from './tensorDimSolver.js';
import {
  type TensorBuffer,
  type LayerGPUState,
  allocateGraph,
  buildBindGroup,
  currentBufKeys,
} from './layerResourceAllocator.js';

// ── InferenceGraph ────────────────────────────────────────────────────────────

export class InferenceGraph {

  private _device:       GPUDevice | null = null;
  private _spec:         UNetSpec;
  private _W:            number = 0;
  private _H:            number = 0;

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

  /** Whether initialize() has completed successfully. */
  private _ready = false;

  /** Uploaded layer weights — retained for bind-group rebuild on buffer resize. */
  private _weightsByName: Map<string, LayerWeights> = new Map();

  /** Uniform-write call count — exposed for test instrumentation. */
  _uniformWriteCount = 0;

  constructor(spec: UNetSpec) {
    this._spec = spec;
  }

  // ── Public API ─────────────────────────────────────────────────────────────

  get ready(): boolean { return this._ready; }

  /**
   * Initialize GPU resources for the given resolution.
   *
   * Uniform buffers are written HERE with actual shape params, not just
   * allocated. Must be called before run().
   *
   * Bind groups are built here with current buffer references. If initialize()
   * is called again (resize), all bind groups are rebuilt.
   */
  async initialize(device: GPUDevice, weights: ModelWeights, W: number, H: number): Promise<void> {
    this._device = device;
    this._W      = W;
    this._H      = H;
    this._ready  = false;

    // Derive tensor dimensions ONCE by simulating the forward pass; store for
    // re-use every frame in run().
    this._tensorDimsMap = computeTensorDims(this._spec, W, H);

    // Allocate all GPU resources (buffers, pipelines, bind groups, uniforms).
    this._tensors = new Map();
    const alloc = await allocateGraph(device, this._spec, weights, W, H, this._tensorDimsMap);

    this._tensors             = alloc.tensors;
    this._layerStates         = alloc.layerStates;
    this._placeholderBuf      = alloc.placeholderBuf;
    this._inputPackPipeline   = alloc.inputPackPipeline;
    this._inputPackUniformBuf = alloc.inputPackUniformBuf;
    this._allocatedBuffers    = alloc.allocatedBuffers;
    this._weightsByName       = alloc.weightsByName;
    // One uniform write per compute layer (the input-pack uniform write at init
    // is NOT counted — it goes direct to the queue, matching the pre-refactor
    // `_writeUniform`-only counting).
    this._uniformWriteCount   = alloc.uniformWriteCount;

    this._ready = true;
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
   * If buffer identities have changed (e.g. after a resize via re-initialize),
   * bind groups are rebuilt before dispatch.
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

      if (!state) continue; // inputPack or unsupported kind

      // Re-validate bind group buffer identity.
      const curKeys = currentBufKeys(layer, this._tensors, this._placeholderBuf!);
      if (!keysEqual(state.cachedBufKeys, curKeys)) {
        // Rebuild bind group with fresh buffer references (keep trained weights).
        const { bindGroup, bufKeys } = buildBindGroup(
          device, state.pipeline, layer, this._weightsByName, state.uniformBuf,
          this._tensors, this._placeholderBuf!, this._allocatedBuffers,
        );
        state.cachedBindGroup = bindGroup;
        state.cachedBufKeys = bufKeys;
      }

      // Re-write uniform with current (stored) dims (handles resize).
      device.queue.writeBuffer(
        state.uniformBuf, 0,
        packLayerUniform(layer, tensorDimsMap, this._H, this._W),
      );
      this._uniformWriteCount++;

      // Dispatch.
      const pass = enc.beginComputePass({ label: `neural-${layer.name}` });
      pass.setPipeline(state.pipeline);
      pass.setBindGroup(0, state.cachedBindGroup);

      const outDims = tensorDimsMap.get(layer.output);
      if (outDims) {
        const [gx, gy, gz] = dispatchWorkgroupsFor(layer.kind, outDims);
        pass.dispatchWorkgroups(gx, gy, gz);
      }
      pass.end();
    }

    // Copy final 'denoised' tensor to the output buffer.
    const denoisedTensor = this._tensors.get('denoised');
    if (denoisedTensor) {
      enc.copyBufferToBuffer(
        denoisedTensor.buf, 0,
        outputBuf, 0,
        denoisedTensor.dims.H * denoisedTensor.dims.W * denoisedTensor.dims.C * 4,
      );
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
    // Null out cached bind groups slot-by-slot BEFORE destroying buffers.
    for (let i = 0; i < this._layerStates.length; i++) {
      const s = this._layerStates[i];
      if (s) {
        s.cachedBindGroup = null;
      }
    }
    this._layerStates = new Array(this._spec.layers.length).fill(null) as null[];

    // Destroy intermediate tensor buffers.
    for (const tb of this._tensors.values()) {
      tb.buf.destroy();
    }
    this._tensors.clear();
    this._tensorDimsMap = new Map();

    // Destroy all tracked allocations (weights, biases, layer uniforms,
    // input-packer uniform, placeholder buffer). Each buffer.destroy() is
    // idempotent on a fresh handle, but guard with try/catch to tolerate
    // double-destroy in error paths.
    for (const buf of this._allocatedBuffers) {
      try { buf.destroy(); } catch { /* tolerate already-destroyed */ }
    }
    this._allocatedBuffers = [];
    this._placeholderBuf = null;
    this._inputPackPipeline = null;
    this._inputPackUniformBuf = null;

    this._device = null;
    this._ready  = false;
    this._weightsByName.clear();
  }

  // ── Private helpers ────────────────────────────────────────────────────────

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
    if (!encInputTensor) return;
    if (!this._inputPackPipeline || !this._inputPackUniformBuf) return;
    if (!this._device) return;

    const device = this._device;
    const pixelCount = this._H * this._W;

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
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function keysEqual(a: readonly string[], b: readonly string[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}
