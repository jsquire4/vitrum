/**
 * WalkaroundGPUPipeline — manages all WebGPU resources + compute passes for
 * the ReSTIR DI/GI pipeline.
 *
 * Uses the fully-manual `device.createShaderModule()` path since wgslFn
 * composition with three's TSL compute() is unvalidated. Web-RTRT confirms
 * this approach works in browser. Exposes a simple `renderFrame()` method
 * that the WalkaroundStage calls per-frame.
 *
 * Pipeline shape per frame:
 *   1. RIS: primary-ray-cast primary visibility + initial candidate sampling
 *   2. Temporal reuse: merge with previous-frame reservoir
 *   3. Spatial reuse (2 separable passes)
 *   4. Shade + GI: compute DI + one indirect bounce, write HDR color
 *   5. Denoise:
 *        • default **SVGF** (Sprint 10a): temporal Welford + variance + 5 à-trous
 *        • optional legacy **à-trous** (3 iters)
 *   6. Temporal accumulation: EMA blend with previous frame's HDR
 *   7. Composite render pass: blit accumulated HDR to the swap-chain texture
 *
 * Note: we use primary-ray-casting mode instead of a G-buffer raster pass.
 * The G-buffer bind group slots are filled with 1×1 placeholder textures for
 * layout compatibility; the RIS + shade passes generate their own primary
 * visibility by casting rays through the BVH.
 */

import type { SceneBVHBuffers } from '../restir/bvhCompute.js';
import { updateUBO } from './uboUpdater.js';
import { compilePipelines } from './pipelineCompiler.js';
import {
  uploadBuffer,
  buildDDGIPlaceholderUBO,
  createFrameResources,
  destroyFrameResources,
  writePpgKdTree,
  type FrameResources,
} from './resourceManager.js';
import {
  encodePpgCellGpuBytes,
  type PpgCellPosition,
} from '../ppg/ppgCellUpload.js';
import {
  type BGLCache,
} from './bindGroupLayouts.js';
import {
  buildFrameBindGroup,
  buildSceneBindGroup,
  buildUboBindGroup,
  buildAtrousBindGroup,
  buildAccumBindGroup,
  buildHybridLayersBindGroup,
  buildCompositeBindGroup,
  buildWelfordBindGroup,
  buildSVGFVarianceBindGroup,
  buildSVGFAtrousBindGroup,
  buildSampleBudgetBindGroup,
  buildResolveBindGroup,
  buildGTAOBindGroup,
  buildGTAOUpsampleBindGroup,
  buildTemporalGiBindGroup,
  buildSpatialGiBindGroup,
  buildIndirectCombineBindGroup,
  type UboRef,
} from './bindGroupBuilders.js';
// Note: we deliberately do NOT import `runSvgfWebGPU` from shared-denoisers.
// That entry point is a one-shot CPU-backed path that allocates and frees
// transient GPU textures per call. This pipeline owns persistent GPU
// textures across frames (accumA/B, variance ping-pong, denoise pings),
// so the one-shot API would churn texture allocations every frame and
// invalidate the bind-group cache. We import only the host-side packing
// helpers and pipeline constants.
import {
  packSVGFUniforms,
  packSVGFVarianceUniforms,
  SVGF_DEFAULT_ATROUS_ITERATIONS,
  SVGF_DEFAULT_UNIFORMS,
  SVGF_UNIFORMS_SIZE_BYTES,
  SVGF_VARIANCE_UNIFORMS_SIZE_BYTES,
} from '@vitrum/shared-denoisers';
import {
  tsWrites,
  initTimestampQueries,
  kickTimestampReadback,
  resolveTimestamps,
  disposeTimestampState,
  makeTimestampState,
  buildPassLayout,
  readTimestampsOnce,
  type TimestampState,
  type PassLabel,
  type PassLayout,
} from './timestampQueries.js';

/**
 * WebGPU device limits required by the layered-hybrid shade pipeline.
 *
 * The ReSTIR shade pass binds 13 storage buffers in the live path:
 *   - 5 frame buffers (current + previous + spatial reservoirs + 2 G-buffer placeholders)
 *   - 5 RC cascade buffers (one per cascade level)
 *   - 3 BVH buffers (nodes / index / position)
 *
 * WebGPU's default `maxStorageBuffersPerShaderStage` is 8. Library
 * consumers must pass these `requiredLimits` when calling
 * `navigator.gpu.requestAdapter().requestDevice({...})` or via the
 * three.js WebGPURenderer constructor's `requiredLimits` option.
 *
 * Caller pattern:
 *   const renderer = new WebGPURenderer({
 *     ...,
 *     requiredLimits: HYBRID_WEBGPU_REQUIRED_LIMITS,
 *   });
 */
export const HYBRID_WEBGPU_REQUIRED_LIMITS: Record<string, number> = {
  maxStorageBuffersPerShaderStage: 16,
};

/**
 * Camera squared-distance threshold for temporal accumulator reset.
 * Tuned to OrbitControls damping: after a drag release, damping continues
 * to move the camera by ~0.1–0.5" per frame for ~30 frames before settling.
 * Threshold 1.0 lets damping ride through to α=0.1 while still resetting
 * history on actual pan/orbit. (WARM-4 fix: was inline magic literal `1.0`.)
 */
const CAMERA_MOVE_RESET_THRESHOLD_SQ = 1.0;

export interface PipelineFrameInputs {
  /** Camera view matrix (column-major mat4x4f, 16 floats). The pipeline
   *  composes VP = projMatrix * viewMatrix internally; do NOT pre-multiply. */
  viewMatrix: Float32Array;
  /** Camera projection matrix (column-major mat4x4f, 16 floats). */
  projMatrix: Float32Array;
  /** Previous-frame view matrix — drives temporal reservoir reuse. Pass
   *  the same matrix as viewMatrix on the first frame to avoid a one-frame
   *  ghost from uninitialized previous-frame state. */
  prevViewMatrix: Float32Array;
  /** Previous-frame projection matrix; same first-frame note as prevViewMatrix. */
  prevProjMatrix: Float32Array;
  /** World-space camera position [x, y, z]. */
  cameraPos: [number, number, number];
  /** Render-target dimensions in pixels. Used by all compute kernels for
   *  workgroup dispatch sizing — must match the swap chain's actual size. */
  screenWidth: number;
  screenHeight: number;
  /** u32 frame counter / per-frame randomness seed. Drives PCG hash inits
   *  for ray jitter, RIS candidate sampling, and temporal reservoir update.
   *  Caller may use a frame index, performance.now()|0, or any monotone u32. */
  frameSeed: number;
  /** Sum of (Le * area) over all emitter triangles, computed at BVH build
   *  time. Used by RIS importance-sampling weight normalization. Must match
   *  the value baked into the emitter CDF in SceneBVHBuffers. */
  totalEmissivePower: number;
  /** Number of entries in the emitter list (length of EmitterTri[] array
   *  in SceneBVHBuffers.emitters). Used by RIS to bound candidate selection. */
  emitterCount: number;
  /** Primary directional light direction [x, y, z] in world space, normalized.
   *  Today this is the sun for cathedral-window glass tracing; the field is
   *  named generically because the path tracer is light-source-agnostic. */
  primaryLightDir: [number, number, number];
  /** Primary directional light irradiance multiplier (linear, unitless).
   *  Must match the value passed to buildSceneBVH({primaryLightIntensity})
   *  at BVH-build time so the shader's self-emission Le for primary panel
   *  hits reproduces exactly the Le baked into the emitter list. */
  primaryLightIntensity: number;
  /** Diffuse-sky-dome RGB tint, derived from computeLightingState. Replaces
   *  four formerly-hardcoded sky tints in WGSL. Consumed by sky-aperture
   *  probe + second-bounce sky-miss paths. */
  skyTint: [number, number, number];
  /** Sky-dome irradiance scalar paired with skyTint. ~0.5×sun at noon. */
  skyIrradiance: number;
  /** The WebGPU swap-chain texture view to render into for this frame.
   *  Caller must obtain via context.getCurrentTexture().createView()
   *  inside the same animation-frame callback that calls renderFrame. */
  swapChainView: GPUTextureView;
  /** The format of swapChainView. The composite pass's render-pipeline
   *  is recompiled if this changes (rare — usually fixed at canvas mount). */
  swapChainFormat: GPUTextureFormat;
}

export class WalkaroundGPUPipeline {
  // Private fields use the `_field` underscore prefix, matching HybridEngine.
  private _device: GPUDevice;
  private _width: number;
  private _height: number;

  // Static BVH + emitter buffers (uploaded once at initialize time)
  private _bvhNodesBuffer!: GPUBuffer;
  private _bvhIndexBuffer!: GPUBuffer;
  private _bvhBeerBuffer!: GPUBuffer;
  private _bvhPositionBuffer!: GPUBuffer;
  private _emitterBuffer!: GPUBuffer;
  private _emitterCdfBuffer!: GPUBuffer;

  // Per-frame GPU resources (created by resourceManager.createFrameResources)
  private _res!: FrameResources;

  // Temporal accumulator ping-pong state
  private _accumPingPongIndex = 0;       // 0 = read A, write B; 1 = swap
  private _accumFrameIndex = 0;
  private _lastCameraPos: [number, number, number] = [0, 0, 0];

  // DDGI inputs (layered hybrid). Null → placeholder textures.
  private _ddgiIrrTex: GPUTexture | null = null;
  private _ddgiVisTex: GPUTexture | null = null;

  // Cached DDGI placeholder UBO — reused by setDDGIInputs(null) so we don't
  // allocate a fresh Float32Array(16) every frame when DDGI is disabled.
  // Populated lazily on first setDDGIInputs(null) call.
  private _ddgiPlaceholderUBO: Float32Array | null = null;

  // Compiled compute + render pipelines
  private _risPipeline!: GPUComputePipeline;
  private _temporalPipeline!: GPUComputePipeline;
  private _spatialPipeline!: GPUComputePipeline;
  private _shadePipeline!: GPUComputePipeline;
  private _atrousPipeline!: GPUComputePipeline;
  private _accumPipeline!: GPUComputePipeline;
  private _compositePipeline!: GPURenderPipeline;
  /** `svgf` — variance-guided (default). `atrous` — legacy three-pass à-trous only. */
  private _denoiserMode: 'atrous' | 'svgf' = 'svgf';
  private _welfordPipeline: GPUComputePipeline | undefined = undefined;
  private _svgfVariancePipeline: GPUComputePipeline | undefined = undefined;
  private _svgfAtrousPipeline: GPUComputePipeline | undefined = undefined;
  // Sprint 9 — adaptive sampling pipelines (always populated).
  private _sampleBudgetPipeline!: GPUComputePipeline;
  private _resolvePipeline!: GPUComputePipeline;
  private _gtaoPipeline!: GPUComputePipeline;
  private _gtaoUpsamplePipeline!: GPUComputePipeline;
  private _risGiPipeline!: GPUComputePipeline;
  private _temporalGiPipeline!: GPUComputePipeline;
  private _spatialGiPipeline!: GPUComputePipeline;
  private _indirectCombinePipeline!: GPUComputePipeline;
  /** Sprint 11 — shade records training samples; {@link ppgUpdatePipeline} consumes them. */
  private _ppgEnabled = false;
  private _ppgUpdatePipeline: GPUComputePipeline | undefined = undefined;
  /** Valid cell indices are [0 .. {@link _ppgActiveCellCount} - 1] after the last {@link uploadPpgCells}. */
  private _ppgActiveCellCount = 0;
  private _swapChainFormat: GPUTextureFormat = 'bgra8unorm';

  /** Ping-pong read index for Welford textures (0 = read varianceBuffer). */
  private _welfordPing = 0;

  // Bind group layout memoisation cache
  private _bglCache: BGLCache = {};

  // Per-pass UBO buffers. Two access patterns coexist:
  //  - Builder-managed (lazy): _atrousUboRef and _accumUboRef are passed
  //    by reference into buildAtrousBindGroup / buildAccumBindGroup, which
  //    lazy-allocate on first call so each builder owns its UBO lifetime.
  //  - Eager: the SVGF and PPG UBOs are allocated in initialize() (gated
  //    by denoiserMode / ppgEnabled) so renderFrame() can write straight
  //    into them without first-frame branching.
  // dispose() walks all seven via the `_perPassUboRefs` array below so
  // adding a new UBO only requires registering it there.
  private _atrousUboRef: UboRef = { buf: undefined };
  private _accumUboRef: UboRef  = { buf: undefined };
  private _welfordUboRef: UboRef = { buf: undefined };
  private _svgfVarianceUboRef: UboRef = { buf: undefined };
  private _svgfAtrousUboRef: UboRef = { buf: undefined };
  private _ppgUpdateUboRef: UboRef = { buf: undefined };
  private _ppgShadeMetaUboRef: UboRef = { buf: undefined };
  // Sprint 9 — adaptive sampling UBOs.
  private _sampleBudgetUboRef: UboRef = { buf: undefined };
  private _sampleCountUboRef:  UboRef = { buf: undefined };
  private _resolveUboRef:      UboRef = { buf: undefined };
  private get _perPassUboRefs(): readonly UboRef[] {
    return [
      this._atrousUboRef,
      this._accumUboRef,
      this._welfordUboRef,
      this._svgfVarianceUboRef,
      this._svgfAtrousUboRef,
      this._ppgUpdateUboRef,
      this._ppgShadeMetaUboRef,
      this._sampleBudgetUboRef,
      this._sampleCountUboRef,
      this._resolveUboRef,
    ];
  }

  // GPU timestamp query state (DEV-only, feature-gated)
  private _tsState: TimestampState = makeTimestampState();
  /** Last successfully-read timestamp values, ms per pass. */
  public lastGpuTimings: Record<string, number> = {};
  /** Frame index of the last completed timestamp read. */
  public lastGpuTimingsFrame: number = -1;

  private _frameCount = 0;
  private _initialized = false;

  constructor(device: GPUDevice, width: number, height: number) {
    this._device = device;
    this._width  = width;
    this._height = height;
  }

  /**
   * Diagnostic one-shot timestamp readback (P3-Vδ). Bypasses the ping-pong
   * fire-and-forget infrastructure and synchronously awaits a fresh staging
   * buffer's mapAsync. Returns per-pass timings + the raw BigInt pairs so
   * dev panels and validation harnesses can confirm whether timestamps are
   * landing in the queryset. Cheap to call (one extra encoder + buffer per
   * invocation) — fine to drive from a 1-Hz telemetry probe.
   */
  async readGpuTimingsOnce(): Promise<{ perPass: Record<string, number>; rawBigints: string[] }> {
    if (!this._initialized) return { perPass: {}, rawBigints: [] };
    const layout = buildPassLayout({
      ppgEnabled: this._ppgEnabled,
      denoiserMode: this._denoiserMode === 'svgf' ? 'svgf' : 'atrous',
    });
    return readTimestampsOnce(this._device, this._tsState, layout);
  }

  /**
   * When PPG buffers are allocated, `ppgBuffers.maxCells`; otherwise `0`.
   * Used by hosts to size CPU cell grids before {@link uploadPpgCells}.
   */
  get ppgAllocatedMaxCells(): number {
    if (!this._initialized) return 0;
    return this._res.ppgBuffers?.maxCells ?? 0;
  }

  /** Upload BVH data + compile shaders. Must be called once before renderFrame. */
  async initialize(
    bvhBuffers: SceneBVHBuffers,
    swapChainFormat: GPUTextureFormat = 'bgra8unorm',
    options?: { ppgEnabled?: boolean; verbose?: boolean; denoiser?: 'atrous' | 'svgf' },
  ): Promise<void> {
    const d = this._device;
    const { _width: W, _height: H } = this;
    this._swapChainFormat = swapChainFormat;
    this._ppgActiveCellCount = 0;

    // ── Upload BVH buffers ────────────────────────────────────────────────
    this._bvhNodesBuffer    = uploadBuffer(d, bvhBuffers.bvhNodes.cpuData,     GPUBufferUsage.STORAGE);
    this._bvhIndexBuffer    = uploadBuffer(d, bvhBuffers.bvhIndex.cpuData,     GPUBufferUsage.STORAGE);
    this._bvhBeerBuffer     = uploadBuffer(d, bvhBuffers.bvhBeerColors.cpuData, GPUBufferUsage.STORAGE);
    this._bvhPositionBuffer = uploadBuffer(d, bvhBuffers.bvhPositions.cpuData, GPUBufferUsage.STORAGE);
    // bvhNormals + bvhUvs are CPU-only on the walkaround path: UVs are packed
    // into bvhPosition[*].w (see restir/packingHelpers.packUVIntoPositionW)
    // and face normals are reconstructed in shader from the BVH-resolved
    // primary hit. No need to upload them.
    this._emitterBuffer     = uploadBuffer(d, bvhBuffers.emitters.cpuData,     GPUBufferUsage.STORAGE);
    this._emitterCdfBuffer  = uploadBuffer(d, bvhBuffers.emitterCdf.cpuData,   GPUBufferUsage.STORAGE);
    // triangleMatIds are packed into bvhIndex[*].w — no separate GPU buffer.

    // ── Per-frame GPU resources ───────────────────────────────────────────
    this._res = createFrameResources(d, W, H, { ppgEnabled: options?.ppgEnabled ?? false });

    // ── Compile shaders ───────────────────────────────────────────────────
    const compiled = await compilePipelines(d, this._bglCache, swapChainFormat, {
      verbose: options?.verbose ?? false,
      denoiser: options?.denoiser ?? 'svgf',
      ppgEnabled: options?.ppgEnabled === true,
    });
    if (options?.ppgEnabled === true) {
      if (!compiled.ppgUpdatePipeline) {
        throw new Error('[ReSTIR] PPG enabled but ppgUpdate pipeline is missing.');
      }
      if (!('ppgBuffers' in this._res)) {
        throw new Error('[ReSTIR] PPG enabled but frame resources omit ppgBuffers.');
      }
    }
    this._risPipeline       = compiled.risPipeline;
    this._temporalPipeline  = compiled.temporalPipeline;
    this._spatialPipeline   = compiled.spatialPipeline;
    this._shadePipeline     = compiled.shadePipeline;
    this._atrousPipeline    = compiled.atrousPipeline;
    this._accumPipeline     = compiled.accumPipeline;
    this._compositePipeline = compiled.compositePipeline;
    this._denoiserMode      = compiled.denoiserMode;
    this._welfordPipeline   = compiled.welfordPipeline;
    this._svgfVariancePipeline = compiled.svgfVariancePipeline;
    this._svgfAtrousPipeline   = compiled.svgfAtrousPipeline;
    this._ppgEnabled           = compiled.ppgEnabled;
    this._ppgUpdatePipeline    = compiled.ppgUpdatePipeline;
    this._sampleBudgetPipeline = compiled.sampleBudgetPipeline;
    this._resolvePipeline      = compiled.resolvePipeline;
    this._gtaoPipeline         = compiled.gtaoPipeline;
    this._gtaoUpsamplePipeline = compiled.gtaoUpsamplePipeline;
    this._risGiPipeline        = compiled.risGiPipeline;
    this._temporalGiPipeline   = compiled.temporalGiPipeline;
    this._spatialGiPipeline    = compiled.spatialGiPipeline;
    this._indirectCombinePipeline = compiled.indirectCombinePipeline;
    if (this._denoiserMode === 'svgf' && (
      !this._welfordPipeline || !this._svgfVariancePipeline || !this._svgfAtrousPipeline
    )) {
      throw new Error('[ReSTIR] SVGF denoiser requested but pipelines are missing.');
    }

    // ── Timestamp queries (DEV-only, feature-gated) ──────────────────────
    initTimestampQueries(d, this._tsState);

    // ── Eager UBO allocation ─────────────────────────────────────────────
    // Allocate all per-frame UBOs upfront so renderFrame() never blocks on
    // first-frame buffer creation and dispose() never has to guard. Each
    // UBO is small (≤32B); the total fixed cost is ~200B regardless of
    // denoiser/PPG mode.
    const U = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;
    this._atrousUboRef.buf = d.createBuffer({ label: 'atrous-ubo', size: 16, usage: U });
    this._accumUboRef.buf  = d.createBuffer({ label: 'accum-ubo',  size: 16, usage: U });
    // Sprint 9 — adaptive sampling UBOs (always allocated; passes always run).
    this._sampleBudgetUboRef.buf = d.createBuffer({ label: 'sample-budget-ubo', size: 16, usage: U });
    this._sampleCountUboRef.buf  = d.createBuffer({ label: 'sample-count-ubo',  size: 16, usage: U });
    this._resolveUboRef.buf      = d.createBuffer({ label: 'resolve-ubo',       size: 16, usage: U });
    if (this._denoiserMode === 'svgf') {
      this._welfordUboRef.buf       = d.createBuffer({ label: 'welford-ubo',        size: 16, usage: U });
      this._svgfVarianceUboRef.buf  = d.createBuffer({ label: 'svgf-variance-ubo',  size: SVGF_VARIANCE_UNIFORMS_SIZE_BYTES, usage: U });
      this._svgfAtrousUboRef.buf    = d.createBuffer({ label: 'svgf-atrous-ubo',    size: SVGF_UNIFORMS_SIZE_BYTES, usage: U });
    }
    if (this._ppgEnabled) {
      this._ppgUpdateUboRef.buf    = d.createBuffer({ label: 'ppg-update-ubo',    size: 16, usage: U });
      this._ppgShadeMetaUboRef.buf = d.createBuffer({ label: 'ppg-shade-meta',    size: 16, usage: U });
    }

    this._initialized = true;
    if (options?.verbose) {
      console.log('[ReSTIR] Pipeline initialized', { W, H, bvhNodes: bvhBuffers.bvhNodes.count, emitters: bvhBuffers.emitterCount });
    }
  }

  /**
   * Upload PPG spatial cell centroids and rebuild the kd-tree over
   * `[0 .. activeCellCount-1]`. Clears directional leaf statistics so training
   * restarts from a clean slate.
   */
  uploadPpgCells(
    cells: ReadonlyArray<PpgCellPosition>,
    activeCellCount: number,
  ): void {
    if (!this._initialized) {
      throw new Error('[ReSTIR] uploadPpgCells: pipeline not initialized');
    }
    if (!this._ppgEnabled || !this._res.ppgBuffers) {
      throw new Error('[ReSTIR] uploadPpgCells: PPG is not enabled on this pipeline');
    }
    const ppg = this._res.ppgBuffers;
    if (activeCellCount > ppg.maxCells) {
      throw new RangeError(
        `[ReSTIR] uploadPpgCells: activeCellCount ${activeCellCount} > maxCells ${ppg.maxCells}`,
      );
    }
    if (activeCellCount > cells.length) {
      throw new RangeError(
        `[ReSTIR] uploadPpgCells: activeCellCount ${activeCellCount} > cells.length ${cells.length}`,
      );
    }
    const enc = encodePpgCellGpuBytes(cells, activeCellCount, ppg.cellBuffer.size);
    this._device.queue.writeBuffer(
      ppg.cellBuffer,
      0,
      enc.buffer as ArrayBuffer,
      enc.byteOffset,
      enc.byteLength,
    );
    writePpgKdTree(this._device.queue, ppg.kdBuffer, cells, activeCellCount);
    const zeros = new Uint8Array(ppg.leafBuffer.size);
    this._device.queue.writeBuffer(ppg.leafBuffer, 0, zeros);
    this._ppgActiveCellCount = activeCellCount;
  }

  /** Re-upload emitter data (called on light/panel change).
   *
   * Re-uploads emitter triangles + power CDF only. CPU-side {@link SceneBVHBuffers.cellPower}
   * is regenerated when callers rebuild the scene BVH via {@link HybridEngine.setScene} /
   * `reset()`.
   */
  updateEmitters(bvhBuffers: Pick<SceneBVHBuffers, 'emitters' | 'emitterCdf'>): void {
    this._emitterBuffer.destroy();
    this._emitterCdfBuffer.destroy();
    this._emitterBuffer    = uploadBuffer(this._device, bvhBuffers.emitters.cpuData,    GPUBufferUsage.STORAGE);
    this._emitterCdfBuffer = uploadBuffer(this._device, bvhBuffers.emitterCdf.cpuData,  GPUBufferUsage.STORAGE);
  }

  /**
   * Blit the most recent resolvedTexture to the host's swap chain WITHOUT
   * running the compute pipeline. Used when HybridEngine's 60-FPS throttle
   * skips a frame — without this, on >60Hz displays the alternate frames'
   * swap-chain textures would never be written and would present as cleared
   * black, producing visible dark flashes.
   */
  presentLastFrame(swapChainView: GPUTextureView): void {
    if (!this._initialized) return;
    const d = this._device;
    const finalTex = this._res.resolvedTexture;
    const bgComposite = buildCompositeBindGroup(
      d, this._bglCache, finalTex.createView(), this._res.compositeLinearSampler,
    );
    const encoder = d.createCommandEncoder({ label: 'composite-only' });
    const pass = encoder.beginRenderPass({
      label: 'composite-only',
      colorAttachments: [{
        view: swapChainView,
        loadOp: 'clear',
        storeOp: 'store',
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      }],
    });
    pass.setPipeline(this._compositePipeline);
    pass.setBindGroup(0, bgComposite);
    pass.draw(3, 1, 0, 0);
    pass.end();
    d.queue.submit([encoder.finish()]);
  }

  /**
   * Run one frame of the ReSTIR compute pipeline + composite render pass.
   * Returns true on success, false if pipeline not ready.
   */
  renderFrame(inputs: PipelineFrameInputs): boolean {
    if (!this._initialized) return false;

    const d = this._device;
    const { _width: W, _height: H } = this;

    // ── Update UBO ────────────────────────────────────────────────────────
    updateUBO(d, this._res.uboBuffer, inputs);

    // ── Build placeholder texture view ────────────────────────────────────
    const placeholderView = this._res.placeholderTexture.createView();

    // ── Build bind groups ─────────────────────────────────────────────────
    const bgFrame = buildFrameBindGroup(d, this._bglCache, {
      placeholderView,
      reservoirCurrentBuffer:  this._res.reservoirCurrentBuffer,
      reservoirPreviousBuffer: this._res.reservoirPreviousBuffer,
      reservoirSpatialBuffer:  this._res.reservoirSpatialBuffer,
      hdrColorTexture:         this._res.hdrColorTexture,
      nearestSampler:          this._res.nearestSampler,
      gNormalDepthTexture:     this._res.gNormalDepthTexture,
      reservoirGiCurrentBuffer: this._res.reservoirGiCurrentBuffer,
      hdrIndirectTexture:      this._res.hdrIndirectTexture,
    });
    const bgScene = buildSceneBindGroup(d, this._bglCache, {
      bvhNodesBuffer:    this._bvhNodesBuffer,
      bvhIndexBuffer:    this._bvhIndexBuffer,
      bvhPositionBuffer: this._bvhPositionBuffer,
      emitterBuffer:     this._emitterBuffer,
      emitterCdfBuffer:  this._emitterCdfBuffer,
      bvhBeerBuffer:     this._bvhBeerBuffer,
    });
    const bgUbo   = buildUboBindGroup(
      d, this._bglCache, this._res.uboBuffer,
      this._res.aoFullTexture.createView(),
    );

    // ── Dispatch compute passes ───────────────────────────────────────────
    const passLayout = buildPassLayout({
      ppgEnabled: this._ppgEnabled,
      denoiserMode: this._denoiserMode === 'svgf' ? 'svgf' : 'atrous',
    });

    const encoder = d.createCommandEncoder({ label: 'walkaround-restir' });

    if (this._ppgEnabled && this._res.ppgBuffers) {
      const ppg = this._res.ppgBuffers;
      encoder.clearBuffer(ppg.sampleBuffer, 0, ppg.sampleBuffer.size);
      encoder.clearBuffer(ppg.sampleHeadBuffer, 0, ppg.sampleHeadBuffer.size);
    }

    const wgX = Math.ceil(W / 8);
    const wgY = Math.ceil(H / 8);

    // Helper: build a GPUComputePassDescriptor without an undefined timestampWrites
    // property — required by exactOptionalPropertyTypes. We spread the optional
    // timestampWrites field only when it has a value. The label is the pass's
    // PassLabel; slot index is resolved through passLayout so it stays in sync
    // with the GPU timing readback labels.
    const computeDesc = (label: PassLabel): GPUComputePassDescriptor => {
      const ts = tsWrites(this._tsState.querySet, passLayout, label);
      return ts ? { label, timestampWrites: ts } : { label };
    };

    // Pass 0: Sample budget (Sprint 9). Reads previous-frame Welford variance,
    // writes per-pixel tier texture (1=converged, 2=med, 4=high-noise). On the
    // first few frames (variance unconverged) every pixel classifies to tier 1
    // because raw bytes read 0 — that's harmless: the tier output is not yet
    // consumed by RIS or shade, so the pass is currently informational only.
    // (Tier-aware RIS / shade is the next Sprint 9 step; this dispatch is the
    // wire-in that lets the downstream consumers be added without re-doing the
    // plumbing.)
    {
      // Budget uniforms: f32 threshold_low, f32 threshold_high, u32 screenW, u32 screenH (16 bytes).
      const budgetBytes = new ArrayBuffer(16);
      const budgetF32 = new Float32Array(budgetBytes);
      const budgetU32 = new Uint32Array(budgetBytes);
      budgetF32[0] = 0.01;
      budgetF32[1] = 0.10;
      budgetU32[2] = W;
      budgetU32[3] = H;
      d.queue.writeBuffer(this._sampleBudgetUboRef.buf!, 0, budgetBytes);
      // Sample count uniforms: u32 sampleCount + 3 pad u32 (16 bytes).
      d.queue.writeBuffer(
        this._sampleCountUboRef.buf!,
        0,
        new Uint32Array([Math.max(this._accumFrameIndex + 1, 1), 0, 0, 0]),
      );
      const bgBudget = buildSampleBudgetBindGroup(
        d, this._bglCache,
        this._res.varianceBuffer.createView(),
        this._res.tierTexture.createView(),
        this._sampleBudgetUboRef.buf!,
        this._sampleCountUboRef.buf!,
      );
      const pass = encoder.beginComputePass(computeDesc('sample-budget'));
      pass.setPipeline(this._sampleBudgetPipeline);
      pass.setBindGroup(0, bgBudget);
      pass.dispatchWorkgroups(wgX, wgY, 1);
      pass.end();
    }

    // Pass 1: RIS (primary ray cast + reservoir sampling)
    {
      const pass = encoder.beginComputePass(computeDesc('ris'));
      pass.setPipeline(this._risPipeline);
      pass.setBindGroup(0, bgFrame);
      pass.setBindGroup(1, bgScene);
      pass.setBindGroup(2, bgUbo);
      pass.dispatchWorkgroups(wgX, wgY, 1);
      pass.end();
    }

    // Pass 2: Temporal reuse
    {
      const pass = encoder.beginComputePass(computeDesc('temporal'));
      pass.setPipeline(this._temporalPipeline);
      pass.setBindGroup(0, bgFrame);
      pass.setBindGroup(1, bgScene);
      pass.setBindGroup(2, bgUbo);
      pass.dispatchWorkgroups(wgX, wgY, 1);
      pass.end();
    }

    // Pass 3a + 3b: Spatial reuse — two passes for fidelity. Per-pass cost
    // on Lovelace was ~22ms each. With NEIGHBORS=5, each pass is heavier
    // but the visual win is the dominant variance reducer in the pipeline.
    {
      const pass = encoder.beginComputePass(computeDesc('spatial-1'));
      pass.setPipeline(this._spatialPipeline);
      pass.setBindGroup(0, bgFrame);
      pass.setBindGroup(1, bgScene);
      pass.setBindGroup(2, bgUbo);
      pass.dispatchWorkgroups(wgX, wgY, 1);
      pass.end();
    }
    {
      const pass = encoder.beginComputePass(computeDesc('spatial-2'));
      pass.setPipeline(this._spatialPipeline);
      pass.setBindGroup(0, bgFrame);
      pass.setBindGroup(1, bgScene);
      pass.setBindGroup(2, bgUbo);
      pass.dispatchWorkgroups(wgX, wgY, 1);
      pass.end();
    }

    // Pass 4: Shading + GI (re-traces primary ray, evaluates ReSTIR).
    // Combined hybrid-layers bind group at slot 3 holds DDGI inputs
    // (Lovelace caps maxBindGroups=4 so we can't use slot 4).
    // shade.wgsl gates on isDDGIWired().
    if (this._ppgEnabled && this._res.ppgBuffers && this._ppgShadeMetaUboRef.buf) {
      d.queue.writeBuffer(
        this._ppgShadeMetaUboRef.buf,
        0,
        new Uint32Array([this._ppgActiveCellCount, 0, 0, 0]),
      );
    }
    const bgHybrid = buildHybridLayersBindGroup(d, this._bglCache, {
      ddgiIrrTex:              this._ddgiIrrTex,
      ddgiVisTex:              this._ddgiVisTex,
      ddgiPlaceholderRgba16f:  this._res.ddgiPlaceholderRgba16f,
      ddgiPlaceholderRg16f:    this._res.ddgiPlaceholderRg16f,
      nearestSampler:          this._res.nearestSampler,
      ddgiUboBuffer:           this._res.ddgiUboBuffer,
      ...(this._ppgEnabled && this._res.ppgBuffers && this._ppgShadeMetaUboRef.buf
        ? {
            ppgTrainBuffers: {
              sampleBuffer:   this._res.ppgBuffers.sampleBuffer,
              headBuffer:     this._res.ppgBuffers.sampleHeadBuffer,
              cellBuffer:     this._res.ppgBuffers.cellBuffer,
              leafBuffer:     this._res.ppgBuffers.leafBuffer,
              kdBuffer:       this._res.ppgBuffers.kdBuffer,
              shadeMetaBuffer: this._ppgShadeMetaUboRef.buf,
            },
          }
        : {}),
    });

    // Sprint 16 — ReSTIR-GI RIS pass. Half-res dispatch (W/2 × H/2).
    // Reuses bgFrame (gNormalDepth + reservoirGiCurrent), bgScene (BVH),
    // bgUbo (camera + ao), bgHybrid (DDGI atlas).
    {
      const halfWg = Math.ceil((Math.floor(W / 2)) / 8);
      const halfWgY = Math.ceil((Math.floor(H / 2)) / 8);
      const pass = encoder.beginComputePass(computeDesc('gi-ris'));
      pass.setPipeline(this._risGiPipeline);
      pass.setBindGroup(0, bgFrame);
      pass.setBindGroup(1, bgScene);
      pass.setBindGroup(2, bgUbo);
      pass.setBindGroup(3, bgHybrid);
      pass.dispatchWorkgroups(halfWg, halfWgY, 1);
      pass.end();
    }

    // Sprint 17 — GI temporal reuse + two spatial passes (ping-pong).
    // All three run half-res (W/2 × H/2) using dedicated single-group BGLs.
    {
      const halfWg  = Math.ceil((Math.floor(W / 2)) / 8);
      const halfWgY = Math.ceil((Math.floor(H / 2)) / 8);

      const bgTemporalGi = buildTemporalGiBindGroup(
        d, this._bglCache,
        this._res.reservoirGiCurrentBuffer,
        this._res.reservoirGiPreviousBuffer,
        this._res.uboBuffer,
      );
      const tPass = encoder.beginComputePass(computeDesc('gi-temporal'));
      tPass.setPipeline(this._temporalGiPipeline);
      tPass.setBindGroup(0, bgTemporalGi);
      tPass.dispatchWorkgroups(halfWg, halfWgY, 1);
      tPass.end();

      // Spatial pass 1: current → spatial.
      const bgSpatial1 = buildSpatialGiBindGroup(
        d, this._bglCache,
        this._res.reservoirGiCurrentBuffer,
        this._res.reservoirGiSpatialBuffer,
        this._res.uboBuffer,
        'spatial-gi-bg-1',
      );
      const s1 = encoder.beginComputePass(computeDesc('gi-spatial-1'));
      s1.setPipeline(this._spatialGiPipeline);
      s1.setBindGroup(0, bgSpatial1);
      s1.dispatchWorkgroups(halfWg, halfWgY, 1);
      s1.end();

      // Spatial pass 2: spatial → current.
      const bgSpatial2 = buildSpatialGiBindGroup(
        d, this._bglCache,
        this._res.reservoirGiSpatialBuffer,
        this._res.reservoirGiCurrentBuffer,
        this._res.uboBuffer,
        'spatial-gi-bg-2',
      );
      const s2 = encoder.beginComputePass(computeDesc('gi-spatial-2'));
      s2.setPipeline(this._spatialGiPipeline);
      s2.setBindGroup(0, bgSpatial2);
      s2.dispatchWorkgroups(halfWg, halfWgY, 1);
      s2.end();
    }

    {
      const pass = encoder.beginComputePass(computeDesc('shade'));
      pass.setPipeline(this._shadePipeline);
      pass.setBindGroup(0, bgFrame);
      pass.setBindGroup(1, bgScene);
      pass.setBindGroup(2, bgUbo);
      pass.setBindGroup(3, bgHybrid);
      pass.dispatchWorkgroups(wgX, wgY, 1);
      pass.end();
    }

    // ── Sprint 15 — GTAO half-res + bilateral upsample ────────────────────
    // Runs after shade (consumes gNormalDepth) and before the denoiser
    // passes. The aoFullTexture is sampled by shade for the *next* frame
    // (so there's a 1-frame lag on AO, invisible for static cameras and
    // ReSTIR-DI's temporal accumulator absorbs slow camera changes).
    {
      // Pack GTAOUniforms (tanFovHalf, radiusPx, intensity, depthThresh).
      const camY = (inputs.projMatrix[5] ?? 1.0); // (1/tan(fov/2)) at the y-FOV
      const tanFovHalf = camY > 1e-6 ? 1.0 / camY : 0.5;
      const gtaoUboBytes = new Float32Array([
        tanFovHalf, // 0
        32.0,       // 1: radiusPx (32px contact radius)
        2.0,        // 2: intensity exponent
        2.0,        // 3: world-unit depth threshold (scene scale ~2 units)
      ]);
      d.queue.writeBuffer(this._res.gtaoUboBuffer, 0, gtaoUboBytes);
      const halfW = Math.max(1, Math.floor(W / 2));
      const halfH = Math.max(1, Math.floor(H / 2));
      const wgGtaoX = Math.ceil(halfW / 8);
      const wgGtaoY = Math.ceil(halfH / 8);
      {
        const bg = buildGTAOBindGroup(
          d, this._bglCache,
          this._res.gNormalDepthTexture.createView(),
          this._res.aoHalfTexture.createView(),
          this._res.gtaoUboBuffer,
        );
        const pass = encoder.beginComputePass(computeDesc('gtao'));
        pass.setPipeline(this._gtaoPipeline);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(wgGtaoX, wgGtaoY, 1);
        pass.end();
      }
      {
        const bg = buildGTAOUpsampleBindGroup(
          d, this._bglCache,
          this._res.aoHalfTexture.createView(),
          this._res.gNormalDepthTexture.createView(),
          this._res.aoFullTexture.createView(),
        );
        const pass = encoder.beginComputePass(computeDesc('gtao-upsample'));
        pass.setPipeline(this._gtaoUpsamplePipeline);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(wgX, wgY, 1);
        pass.end();
      }
    }

    if (this._ppgEnabled && this._ppgUpdatePipeline && this._res.ppgBuffers && this._ppgUpdateUboRef.buf) {
      const ppg = this._res.ppgBuffers;
      const sampleCapacity = Math.floor(ppg.sampleBuffer.size / 48);
      const uPpg = new Uint32Array([
        sampleCapacity,
        this._frameCount % 2,
        this._ppgActiveCellCount,
        0,
      ]);
      d.queue.writeBuffer(this._ppgUpdateUboRef.buf, 0, uPpg);
      const bgPpgUpdate = d.createBindGroup({
        layout: this._ppgUpdatePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this._ppgUpdateUboRef.buf } },
          { binding: 1, resource: { buffer: ppg.sampleBuffer } },
          { binding: 2, resource: { buffer: ppg.cellBuffer } },
          { binding: 3, resource: { buffer: ppg.leafBuffer } },
          { binding: 4, resource: { buffer: ppg.kdBuffer } },
        ],
      });
      const wgPpg = Math.max(1, Math.ceil(sampleCapacity / 64));
      {
        const pass = encoder.beginComputePass(computeDesc('ppg-update'));
        pass.setPipeline(this._ppgUpdatePipeline);
        pass.setBindGroup(0, bgPpgUpdate);
        pass.dispatchWorkgroups(wgPpg, 1, 1);
        pass.end();
      }
    }

    // ── Camera motion: reset temporal index before denoise / accum ────────
    const dx = inputs.cameraPos[0] - this._lastCameraPos[0];
    const dy = inputs.cameraPos[1] - this._lastCameraPos[1];
    const dz = inputs.cameraPos[2] - this._lastCameraPos[2];
    const camMoveSq = dx * dx + dy * dy + dz * dz;
    const isMoving = camMoveSq > CAMERA_MOVE_RESET_THRESHOLD_SQ;
    if (isMoving) {
      this._accumFrameIndex = 0;
    }

    const wgX16 = Math.ceil(W / 16);
    const wgY16 = Math.ceil(H / 16);
    const gNormalDepthView = this._res.gNormalDepthTexture.createView();

    const readAccum  = this._accumPingPongIndex === 0 ? this._res.accumTextureA : this._res.accumTextureB;
    const writeAccum = this._accumPingPongIndex === 0 ? this._res.accumTextureB : this._res.accumTextureA;

    let denoisedOut: GPUTexture;

    if (this._denoiserMode === 'svgf') {
      denoisedOut = this._dispatchSVGF(
        encoder, gNormalDepthView, readAccum, isMoving, wgX16, wgY16, computeDesc,
      );
      this._welfordPing = 1 - this._welfordPing;
    } else {
      denoisedOut = this._dispatchAtrousLegacy(
        encoder, gNormalDepthView, wgX16, wgY16, computeDesc,
      );
    }

    // Sprint 18 — per-channel combine. The SVGF/atrous chain above ran on
    // hdrColorTexture (now direct-only). Bilaterally blur hdrIndirectTexture
    // with broader sigmas, sum the two, write to combinedDenoisedTexture,
    // and feed that into temporalAccum below.
    const combinedTex = this._res.combinedDenoisedTexture;
    {
      const bgCombine = buildIndirectCombineBindGroup(
        d, this._bglCache,
        denoisedOut.createView(),
        this._res.hdrIndirectTexture.createView(),
        gNormalDepthView,
        combinedTex.createView(),
      );
      const pass = encoder.beginComputePass(computeDesc('indirect-combine'));
      pass.setPipeline(this._indirectCombinePipeline);
      pass.setBindGroup(0, bgCombine);
      pass.dispatchWorkgroups(wgX16, wgY16, 1);
      pass.end();
    }
    denoisedOut = combinedTex;

    // alpha=0.02 (was 0.1, then 0.05) gives ~98% history weight per frame.
    // Slower convergence to legitimate changes, but the camera-motion path
    // resets accumFrameIndex and forces alpha=1.0, so motion responsiveness
    // is unchanged. Aggressive history blending is the cheapest knob to
    // reduce ReSTIR-DI variance on static-camera frames, especially on
    // bright surfaces with partial light-source occlusion (floor near the
    // Cornell boxes), where stochastic light-point selection introduces
    // binary visibility signal per frame.
    const alpha = this._accumFrameIndex === 0 ? 1.0 : 0.02;
    this._lastCameraPos = [...inputs.cameraPos];

    {
      const bgAccum = buildAccumBindGroup(
        d, this._bglCache, this._accumUboRef,
        denoisedOut.createView(),
        readAccum.createView(),
        writeAccum.createView(),
        alpha,
      );
      const pass = encoder.beginComputePass(computeDesc('temporalAccum'));
      pass.setPipeline(this._accumPipeline);
      pass.setBindGroup(0, bgAccum);
      pass.dispatchWorkgroups(wgX16, wgY16, 1);
      pass.end();
    }
    this._accumPingPongIndex = 1 - this._accumPingPongIndex;
    this._accumFrameIndex++;

    // Pass: Resolve (Sprint 9). Currently runs in passthrough mode
    // (checkerboardOn=0) — every pixel copies through from writeAccum to
    // resolvedTexture. When shade.wgsl is upgraded to write sparsely
    // (checkerboard pattern), flip checkerboardOn=1 in the resolve UBO and
    // the gap-fill branch becomes active. Until then this pass costs one
    // extra texture copy per frame but produces identical output.
    {
      // ResolveUniforms: u32 W, u32 H, u32 frameParity, u32 checkerboardOn (16 bytes).
      d.queue.writeBuffer(
        this._resolveUboRef.buf!,
        0,
        new Uint32Array([W, H, this._frameCount & 1, 0]),
      );
      const bgResolve = buildResolveBindGroup(
        d, this._bglCache,
        this._resolveUboRef.buf!,
        writeAccum.createView(),                            // current radiance (post-accum)
        readAccum.createView(),                             // prev radiance (other ping-pong slot)
        this._res.motionVectorTexture.createView(),         // motion vectors (zero-filled until a motion-vector pass exists)
        this._res.resolvedTexture.createView(),
      );
      const pass = encoder.beginComputePass(computeDesc('resolve'));
      pass.setPipeline(this._resolvePipeline);
      pass.setBindGroup(0, bgResolve);
      // resolve.wgsl uses @workgroup_size(8, 8, 1) — dispatch with wgX/wgY
      // (ceil(W/8), ceil(H/8)) NOT the 16×16-sized wgX16/wgY16.
      pass.dispatchWorkgroups(wgX, wgY, 1);
      pass.end();
    }

    // Pass: Composite render pass — blit resolved HDR to swap-chain.
    const finalTex = this._res.resolvedTexture;
    const bgComposite = buildCompositeBindGroup(d, this._bglCache, finalTex.createView(), this._res.compositeLinearSampler);
    {
      const tsComp = tsWrites(this._tsState.querySet, passLayout, 'composite');
      const pass = encoder.beginRenderPass({
        label: 'composite',
        colorAttachments: [{
          view: inputs.swapChainView,
          loadOp: 'clear',
          storeOp: 'store',
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
        }],
        ...(tsComp ? { timestampWrites: tsComp } : {}),
      });
      pass.setPipeline(this._compositePipeline);
      pass.setBindGroup(0, bgComposite);
      pass.draw(3, 1, 0, 0);  // 3 vertices, fullscreen triangle
      pass.end();
    }

    // Resolve timestamps + copy into the inactive readback buffer.
    resolveTimestamps(encoder, this._tsState, this._frameCount, passLayout.slotCount);

    d.queue.submit([encoder.finish()]);

    // Swap reservoir ping-pong for next frame (copy current → previous).
    // Sprint 17 — also copy GI reservoir current → previous for next-frame
    // temporal reuse. Both copies submitted in the same command encoder.
    const enc2 = d.createCommandEncoder({ label: 'reservoir-swap' });
    enc2.copyBufferToBuffer(
      this._res.reservoirCurrentBuffer, 0,
      this._res.reservoirPreviousBuffer, 0,
      this._res.reservoirCurrentBuffer.size,
    );
    enc2.copyBufferToBuffer(
      this._res.reservoirGiCurrentBuffer, 0,
      this._res.reservoirGiPreviousBuffer, 0,
      this._res.reservoirGiCurrentBuffer.size,
    );
    d.queue.submit([enc2.finish()]);

    // Kick async readback of the timestamp buffer we just copied into.
    // Pass the layout labels so the async callback labels each slot
    // correctly even if the pipeline reconfigures between frames.
    kickTimestampReadback(this._tsState, this._frameCount, passLayout.labels);
    // Mirror public telemetry fields from the state object so callers
    // can read them as before.
    this.lastGpuTimings      = this._tsState.lastGpuTimings;
    this.lastGpuTimingsFrame = this._tsState.lastGpuTimingsFrame;

    this._frameCount++;
    return true;
  }

  /**
   * SVGF denoise dispatch — welford-temporal → svgf-variance → N × svgf-atrous.
   * Returns the final atrous output texture to feed into the accumulator.
   */
  private _dispatchSVGF(
    encoder: GPUCommandEncoder,
    gNormalDepthView: GPUTextureView,
    readAccum: GPUTexture,
    isMoving: boolean,
    wgX16: number,
    wgY16: number,
    computeDesc: (label: PassLabel) => GPUComputePassDescriptor,
  ): GPUTexture {
    const d = this._device;
    const wf = this._welfordPipeline!;
    const sv = this._svgfVariancePipeline!;
    const sa = this._svgfAtrousPipeline!;

    const welfordRead  = this._welfordPing === 0 ? this._res.varianceBuffer : this._res.varianceBufferAux;
    const welfordWrite = this._welfordPing === 0 ? this._res.varianceBufferAux : this._res.varianceBuffer;

    // welfordUboRef.buf is allocated eagerly in initialize() when denoiserMode === 'svgf'.
    const wU32 = new Uint32Array([this._accumFrameIndex + 1, isMoving ? 1 : 0, 0, 0]);
    d.queue.writeBuffer(this._welfordUboRef.buf!, 0, wU32);

    const hdrColorView = this._res.hdrColorTexture.createView();
    {
      const pass = encoder.beginComputePass(computeDesc('welford-temporal'));
      pass.setPipeline(wf);
      pass.setBindGroup(0, buildWelfordBindGroup(
        d, wf, hdrColorView, welfordRead.createView(), welfordWrite.createView(),
        this._welfordUboRef.buf!,
      ));
      pass.dispatchWorkgroups(wgX16, wgY16, 1);
      pass.end();
    }

    // svgfVarianceUboRef.buf is allocated eagerly in initialize() when denoiserMode === 'svgf'.
    const varUboBytes = new ArrayBuffer(SVGF_VARIANCE_UNIFORMS_SIZE_BYTES);
    packSVGFVarianceUniforms({ frameCount: this._accumFrameIndex }, varUboBytes, 0);
    d.queue.writeBuffer(this._svgfVarianceUboRef.buf!, 0, varUboBytes);

    {
      const pass = encoder.beginComputePass(computeDesc('svgf-variance'));
      pass.setPipeline(sv);
      pass.setBindGroup(0, buildSVGFVarianceBindGroup(
        d, sv,
        hdrColorView,
        readAccum.createView(),
        gNormalDepthView,
        this._res.motionVectorTexture.createView(),
        welfordWrite.createView(),
        this._res.svgfVarianceEstimateTexture.createView(),
        this._svgfVarianceUboRef.buf!,
      ));
      pass.dispatchWorkgroups(wgX16, wgY16, 1);
      pass.end();
    }

    // svgfAtrousUboRef.buf is allocated eagerly in initialize() when denoiserMode === 'svgf'.
    const atrousUboBytes = new ArrayBuffer(SVGF_UNIFORMS_SIZE_BYTES);
    let inputTex: GPUTexture = this._res.hdrColorTexture;
    const varView = this._res.svgfVarianceEstimateTexture.createView();
    for (let iter = 0; iter < SVGF_DEFAULT_ATROUS_ITERATIONS; iter++) {
      packSVGFUniforms(
        { iteration: iter, ...SVGF_DEFAULT_UNIFORMS },
        atrousUboBytes,
        0,
      );
      d.queue.writeBuffer(this._svgfAtrousUboRef.buf!, 0, atrousUboBytes);
      const outTex = iter % 2 === 0 ? this._res.denoisedPingTexture : this._res.denoisedPongTexture;
      const pass = encoder.beginComputePass(computeDesc(`svgf-atrous-${iter}` as PassLabel));
      pass.setPipeline(sa);
      pass.setBindGroup(0, buildSVGFAtrousBindGroup(
        d, sa,
        inputTex.createView(), outTex.createView(),
        gNormalDepthView, varView,
        this._svgfAtrousUboRef.buf!,
      ));
      pass.dispatchWorkgroups(wgX16, wgY16, 1);
      pass.end();
      inputTex = outTex;
    }
    return inputTex;
  }

  /**
   * Legacy à-trous denoise dispatch — 3 iterations with widening step.
   */
  private _dispatchAtrousLegacy(
    encoder: GPUCommandEncoder,
    gNormalDepthView: GPUTextureView,
    wgX16: number,
    wgY16: number,
    computeDesc: (label: PassLabel) => GPUComputePassDescriptor,
  ): GPUTexture {
    const d = this._device;
    let inputTex = this._res.hdrColorTexture;
    for (let iter = 0; iter < 3; iter++) {
      const stepWidth = 1 << iter;
      const outputTex = iter % 2 === 0 ? this._res.denoisedPingTexture : this._res.denoisedPongTexture;
      const bgAtrous = buildAtrousBindGroup(
        d, this._bglCache, this._atrousUboRef,
        inputTex.createView(), outputTex.createView(),
        gNormalDepthView, gNormalDepthView, stepWidth,
      );
      const pass = encoder.beginComputePass(computeDesc(`atrous-${iter}` as PassLabel));
      pass.setPipeline(this._atrousPipeline);
      pass.setBindGroup(0, bgAtrous);
      pass.dispatchWorkgroups(wgX16, wgY16, 1);
      pass.end();
      inputTex = outputTex;
    }
    return inputTex;
  }

  dispose(): void {
    this._bvhNodesBuffer?.destroy();
    this._bvhIndexBuffer?.destroy();
    this._bvhBeerBuffer?.destroy();
    this._bvhPositionBuffer?.destroy();
    this._emitterBuffer?.destroy();
    this._emitterCdfBuffer?.destroy();
    if (this._res) destroyFrameResources(this._res);
    for (const ref of this._perPassUboRefs) ref.buf?.destroy();
    disposeTimestampState(this._tsState);
  }

  /**
   * Bind a DDGI atlas. Pass `null` to revert to placeholder (no-DDGI
   * fallback). The shade pass continues to render with hardcoded sky color
   * when isDDGIWired() returns false (dimsX ≤ 1 in the placeholder UBO).
   *
   * Caller (HybridLayeredStage) provides:
   *  - irradianceTex / visibilityTex: GPUTexture with TEXTURE_BINDING usage.
   *  - gridParams: 64-byte ArrayBuffer matching the WGSL DDGIGridUniform
   *    layout (origin vec3 + spacing f32 + dims vec3u + pad u32 +
   *    irradianceAtlasW/H + visibilityAtlasW/H).
   */
  setDDGIInputs(inputs: {
    irradianceTex: GPUTexture;
    visibilityTex: GPUTexture;
    gridParams: ArrayBuffer;
  } | null): void {
    if (inputs === null) {
      this._ddgiIrrTex = null;
      this._ddgiVisTex = null;
      // Restore placeholder UBO (dims=1×1×1) so shade.wgsl's
      // isDDGIWired() check returns false and Lo_ddgi drops to zero.
      // Cache the placeholder to avoid allocating Float32Array(16) every
      // frame when DDGI is disabled (HOT-1 fix).
      if (this._ddgiPlaceholderUBO === null) {
        this._ddgiPlaceholderUBO = buildDDGIPlaceholderUBO();
      }
      this._device.queue.writeBuffer(this._res.ddgiUboBuffer, 0, this._ddgiPlaceholderUBO.buffer);
    } else {
      this._ddgiIrrTex = inputs.irradianceTex;
      this._ddgiVisTex = inputs.visibilityTex;
      if (inputs.gridParams.byteLength > 0) {
        this._device.queue.writeBuffer(this._res.ddgiUboBuffer, 0, inputs.gridParams);
      }
    }
  }
}
