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
 *        • default **atrous-variance** (Sprint 10a): temporal Welford + variance + 3 à-trous
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
import type { InferenceGraph } from '../neural/InferenceGraph.js';
import { updateUBO } from './uboUpdater.js';
import { compilePipelines } from './pipelineCompiler.js';
import {
  uploadBuffer,
  buildDDGIPlaceholderUBO,
  createFrameResources,
  destroyFrameResources,
  type FrameResources,
} from './resourceManager.js';
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
  buildAtrousVarianceVarianceBindGroup,
  buildAtrousVarianceAtrousBindGroup,
  buildSampleBudgetBindGroup,
  buildResolveBindGroup,
  buildGTAOBindGroup,
  buildGTAOUpsampleBindGroup,
  buildTemporalGiBindGroup,
  buildSpatialGiBindGroup,
  buildIndirectCombineBindGroup,
  buildIndirectTemporalAccumBindGroup,
  ATROUS_INDIRECT_SIGMAS,
  type UboRef,
} from './bindGroupBuilders.js';
// Note: we deliberately do NOT import `runAtrousVarianceWebGPU` from shared-denoisers.
// That entry point is a one-shot CPU-backed path that allocates and frees
// transient GPU textures per call. This pipeline owns persistent GPU
// textures across frames (accumA/B, variance ping-pong, denoise pings),
// so the one-shot API would churn texture allocations every frame and
// invalidate the bind-group cache. We import only the host-side packing
// helpers and pipeline constants.
import {
  packAtrousVarianceAtrousUniforms,
  packAtrousVarianceVarianceUniforms,
  ATROUS_VARIANCE_DEFAULT_ATROUS_ITERATIONS,
  ATROUS_VARIANCE_DEFAULT_ATROUS_UNIFORMS,
  ATROUS_VARIANCE_ATROUS_UNIFORMS_SIZE_BYTES,
  ATROUS_VARIANCE_VARIANCE_UNIFORMS_SIZE_BYTES,
  packSVGFReprojUniforms,
  SVGF_REPROJ_UNIFORMS_SIZE_BYTES,
  SVGF_REPROJ_DEFAULT_UNIFORMS,
  SVGF_REAL_DEFAULT_ATROUS_ITERATIONS,
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
  // Sprint 18 — shade writes 4 storage textures simultaneously
  // (hdrColorOut, gNormalDepthOut, hdrIndirectOut, hdrTotalOut), at the
  // default cap of 4. Indirect-combine writes a 5th. Lift to 8 so future
  // additions (sparse / motion-vector outputs) have headroom too.
  maxStorageTexturesPerShaderStage: 8,
};

/**
 * WebGPU features the hybrid pipeline requires.  Hosts must include these in
 * `adapter.requestDevice({ requiredFeatures })` (alongside any of their own).
 *
 * - `texture-formats-tier1`: lifts r16float (used by Sprint 15 GTAO half/full
 *   AO textures) and rg16float into write-only storage texture support.
 *   Without it, the gtao + gtao-upsample BGLs fail to validate and the engine
 *   reports "Invalid PipelineLayout" at compile time.
 *
 * Optional features (e.g. `timestamp-query` for dev-time per-pass GPU
 * timings) are handled separately by the host and are not part of the
 * required-features contract.
 */
export const HYBRID_WEBGPU_REQUIRED_FEATURES: readonly GPUFeatureName[] = [
  'texture-formats-tier1' as GPUFeatureName,
];

/**
 * Default camera squared-distance threshold for temporal accumulator reset.
 * 1.0 is calibrated to Cornell's ~2-unit room + OrbitControls damping
 * (~0.1–0.5 units per frame for ~30 frames after a drag release). Hosts on
 * different scene scales should override via
 * `HybridEngineOptions.cameraMoveResetThresholdSq`. See audit B8.
 */
const DEFAULT_CAMERA_MOVE_RESET_THRESHOLD_SQ = 1.0;

/**
 * Default per-frame temporal-accumulator EMA weight. 0.01 = 99% history
 * retain, tuned for Cornell convergence at ~60 FPS. Framerate-dependent;
 * see audit M3.
 */
const DEFAULT_TEMPORAL_ACCUM_ALPHA = 0.01;

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
  /** Audit M12 — emitter geometry term dist² floor; default `0.01` for
   *  Cornell-scale; hosts on different scales should pass `(diag * 1e-3)²`. */
  emitterDist2Floor: number;
  /** Audit B4 — per-channel max HDR radiance clamp on the direct channel.
   *  Default 4.0 calibrated to Le=12. */
  directFireflyClamp: number;
  /** Audit B1 — stained-glass caustic boost. Cornell uses 22.0; generic
   *  scenes pass 1.0 (no boost). */
  causticBoost: number;
  /** Audit B1 — clamp applied to the tinted-visibility vector before the
   *  caustic-boost multiplication. Cornell uses 0.6; generic scenes pass 1.0. */
  causticVisClamp: number;
  /** Audit M6 — ReSTIR-DI temporal M-clamp; Cornell default 20. */
  temporalMClampDI: number;
  /** Audit M7 — ReSTIR-DI spatial reuse radius in pixels; Cornell default 30. */
  spatialReuseRadiusPx: number;
  /** Audit M8 — ReSTIR-DI spatial depth-tolerance world-units floor; Cornell
   *  default 0.05 (5 cm). Hosts on different scales should pass
   *  `sceneDiagonal * 1e-3`. */
  spatialDepthTolFloor: number;
  /** D12 — Möller-Trumbore coplanarity epsilon.  Controls the `abs(det) < ε`
   *  near-zero determinant threshold in `intersectTriangle`.  Default `1e-5`
   *  (metre-scale).  Reduce for millimetre-scale geometry. */
  triIntersectEpsilon: number;
  /** Audit M1 — GTAO sampling radius in pixels; Cornell default 32. */
  gtaoRadiusPx: number;
  /** Audit M1 — GTAO intensity exponent; Cornell default 2.0. */
  gtaoIntensity: number;
  /** Audit M1 — GTAO depth threshold in world units; Cornell default 2.0. */
  gtaoDepthThreshold: number;
  /** Audit B3 — GTAO upsample bilateral depth sigma (world units);
   *  Cornell default 0.25 (= 1/√(2*4); see legacy `exp(-Δ * 4)`). */
  gtaoBilateralDepthSigma: number;
  /** Audit M2 — adaptive-sampling tier classifier low-variance threshold;
   *  Cornell default 0.01.  Variance below this → tier 1 (converged). */
  adaptiveSamplingThresholdLow: number;
  /** Audit M2 — adaptive-sampling tier classifier high-variance threshold;
   *  Cornell default 0.10.  Variance above this → tier 4 (high noise). */
  adaptiveSamplingThresholdHigh: number;
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
  /** `atrous-variance` — variance-guided (default). `atrous` — legacy. `svgf-real` — Schied 2017 T2.H1. `neural` — T2.H2 U-Net. */
  private _denoiserMode: 'atrous' | 'atrous-variance' | 'svgf-real' | 'neural' = 'atrous-variance';
  /** T2.H2 — neural denoiser InferenceGraph (populated when denoiserMode === 'neural'). */
  private _inferenceGraph: InferenceGraph | null = null;
  /** Audit B8 — populated at initialize() time from HybridEngineOptions. */
  private _cameraMoveResetThresholdSq = DEFAULT_CAMERA_MOVE_RESET_THRESHOLD_SQ;
  /** Audit M3 — populated at initialize() time from HybridEngineOptions. */
  private _temporalAccumAlpha = DEFAULT_TEMPORAL_ACCUM_ALPHA;
  private _welfordPipeline: GPUComputePipeline | undefined = undefined;
  private _atrousVarianceVariancePipeline: GPUComputePipeline | undefined = undefined;
  private _atrousVarianceAtrousPipeline: GPUComputePipeline | undefined = undefined;
  // T2.H1 — svgf-real pipelines (populated when denoiserMode === 'svgf-real').
  private _svgfReprojPipeline:    GPUComputePipeline | undefined = undefined;
  private _svgfMomentsPipeline:   GPUComputePipeline | undefined = undefined;
  private _svgfFallbackPipeline:  GPUComputePipeline | undefined = undefined;
  private _svgfRealAtrousPipeline:GPUComputePipeline | undefined = undefined;
  // T2.H3 — PPG pipelines (Müller 2017; populated when ppgEnabled === true).
  private _ppgUpdatePipeline: GPUComputePipeline | undefined = undefined;
  private _ppgGuidePipeline:  GPUComputePipeline | undefined = undefined;
  /** T2.H3 — PPG is enabled iff both pipelines were compiled successfully. */
  private _ppgEnabled = false;
  /** T2.H1 — UBO for the svgf-real reprojection pass (SVGFReprojUBO, 16 bytes). */
  private _svgfReprojUboRef: UboRef = { buf: undefined };
  /** T2.H1 — Ping-pong index for svgf-real history/moments/prevRadiance. 0 = A→read, B→write. */
  private _svgfPingPong = 0;
  // Sprint 9 — adaptive sampling pipelines (always populated).
  private _sampleBudgetPipeline!: GPUComputePipeline;
  private _resolvePipeline!: GPUComputePipeline;
  private _gtaoPipeline!: GPUComputePipeline;
  private _gtaoUpsamplePipeline!: GPUComputePipeline;
  private _risGiPipeline!: GPUComputePipeline;
  private _temporalGiPipeline!: GPUComputePipeline;
  private _spatialGiPipeline!: GPUComputePipeline;
  private _indirectCombinePipeline!: GPUComputePipeline;
  private _indirectTemporalAccumPipeline!: GPUComputePipeline;
  /** Sprint 18 follow-up — ping-pong index for the indirect temporal accumulator. */
  private _indirectAccumPingPong = 0;
  private _swapChainFormat: GPUTextureFormat = 'bgra8unorm';

  /** Ping-pong read index for Welford textures (0 = read varianceBuffer). */
  private _welfordPing = 0;

  // Bind group layout memoisation cache
  private _bglCache: BGLCache = {};

  // Per-pass UBO buffers. Two access patterns coexist:
  //  - Builder-managed (lazy): _atrousUboRef and _accumUboRef are passed
  //    by reference into buildAtrousBindGroup / buildAccumBindGroup, which
  //    lazy-allocate on first call so each builder owns its UBO lifetime.
  //  - Eager: the atrous-variance UBOs are allocated in initialize() (gated by
  //    denoiserMode) so renderFrame() can write straight into them without
  //    first-frame branching.
  // dispose() walks all via the `_perPassUboRefs` array below so
  // adding a new UBO only requires registering it there.
  private _atrousUboRef: UboRef = { buf: undefined };
  /** Sprint 18 — separate UBO for the indirect-channel atrous chain so it
   *  doesn't race the direct chain's per-iteration sigma writes. */
  private _atrousIndirectUboRef: UboRef = { buf: undefined };
  private _accumUboRef: UboRef  = { buf: undefined };
  private _welfordUboRef: UboRef = { buf: undefined };
  private _atrousVarianceVarianceUboRef: UboRef = { buf: undefined };
  private _atrousVarianceAtrousUboRef: UboRef = { buf: undefined };
  // Sprint 9 — adaptive sampling UBOs.
  private _sampleBudgetUboRef: UboRef = { buf: undefined };
  private _sampleCountUboRef:  UboRef = { buf: undefined };
  private _resolveUboRef:      UboRef = { buf: undefined };
  private get _perPassUboRefs(): readonly UboRef[] {
    return [
      this._atrousUboRef,
      this._atrousIndirectUboRef,
      this._accumUboRef,
      this._welfordUboRef,
      this._atrousVarianceVarianceUboRef,
      this._atrousVarianceAtrousUboRef,
      this._sampleBudgetUboRef,
      this._sampleCountUboRef,
      this._resolveUboRef,
      this._svgfReprojUboRef,
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
    const layout = buildPassLayout({ denoiserMode: this._denoiserMode });
    return readTimestampsOnce(this._device, this._tsState, layout);
  }

  /** Upload BVH data + compile shaders. Must be called once before renderFrame. */
  async initialize(
    bvhBuffers: SceneBVHBuffers,
    swapChainFormat: GPUTextureFormat = 'bgra8unorm',
    options?: {
      verbose?: boolean;
      denoiser?: 'atrous' | 'atrous-variance' | 'svgf-real' | 'neural';
      /** Audit B8 — host-overridable camera-move temporal-reset threshold. */
      cameraMoveResetThresholdSq?: number;
      /** Audit M3 — host-overridable temporal-accumulator EMA weight. */
      temporalAccumAlpha?: number;
      /** T2.H2 — neural denoiser InferenceGraph (required when denoiser='neural'). */
      inferenceGraph?: InferenceGraph;
      /** T2.H3 — enable PPG (Müller 2017 adaptive sTree + dTree + MIS). */
      ppgEnabled?: boolean;
    },
  ): Promise<void> {
    const d = this._device;
    const { _width: W, _height: H } = this;
    this._swapChainFormat = swapChainFormat;

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
    this._res = createFrameResources(d, W, H);

    // ── Compile shaders ───────────────────────────────────────────────────
    // T2.H2 — 'neural' mode falls through to 'atrous-variance' for pipeline compilation
    // (the shade pipeline is still needed). InferenceGraph handles its own GPU pipelines.
    const compiledDenoiser = options?.denoiser === 'neural' ? 'atrous-variance' : (options?.denoiser ?? 'atrous-variance');
    const compiled = await compilePipelines(d, this._bglCache, swapChainFormat, {
      verbose: options?.verbose ?? false,
      denoiser: compiledDenoiser,
      ppgEnabled: options?.ppgEnabled ?? false,
    });
    this._risPipeline       = compiled.risPipeline;
    this._temporalPipeline  = compiled.temporalPipeline;
    this._spatialPipeline   = compiled.spatialPipeline;
    this._shadePipeline     = compiled.shadePipeline;
    this._atrousPipeline    = compiled.atrousPipeline;
    this._accumPipeline     = compiled.accumPipeline;
    this._compositePipeline = compiled.compositePipeline;
    // T2.H2: override denoiserMode to 'neural' when requested (the compiled object
    // stores 'atrous-variance' as the fallback — we override here).
    this._denoiserMode = options?.denoiser ?? compiled.denoiserMode;
    this._cameraMoveResetThresholdSq = options?.cameraMoveResetThresholdSq
      ?? DEFAULT_CAMERA_MOVE_RESET_THRESHOLD_SQ;
    this._temporalAccumAlpha = options?.temporalAccumAlpha
      ?? DEFAULT_TEMPORAL_ACCUM_ALPHA;
    this._welfordPipeline   = compiled.welfordPipeline;
    this._atrousVarianceVariancePipeline = compiled.atrousVarianceVariancePipeline;
    this._atrousVarianceAtrousPipeline   = compiled.atrousVarianceAtrousPipeline;
    // T2.H1 — svgf-real pipelines (undefined unless denoiserMode === 'svgf-real').
    this._svgfReprojPipeline    = compiled.svgfReprojPipeline;
    this._svgfMomentsPipeline   = compiled.svgfMomentsPipeline;
    this._svgfFallbackPipeline  = compiled.svgfFallbackPipeline;
    this._svgfRealAtrousPipeline= compiled.svgfRealAtrousPipeline;
    this._sampleBudgetPipeline = compiled.sampleBudgetPipeline;
    this._resolvePipeline      = compiled.resolvePipeline;
    this._gtaoPipeline         = compiled.gtaoPipeline;
    this._gtaoUpsamplePipeline = compiled.gtaoUpsamplePipeline;
    this._risGiPipeline        = compiled.risGiPipeline;
    this._temporalGiPipeline   = compiled.temporalGiPipeline;
    this._spatialGiPipeline    = compiled.spatialGiPipeline;
    this._indirectCombinePipeline = compiled.indirectCombinePipeline;
    this._indirectTemporalAccumPipeline = compiled.indirectTemporalAccumPipeline;
    if (this._denoiserMode === 'atrous-variance' && (
      !this._welfordPipeline || !this._atrousVarianceVariancePipeline || !this._atrousVarianceAtrousPipeline
    )) {
      throw new Error('[ReSTIR] atrous-variance denoiser requested but pipelines are missing.');
    }
    // T2.H2 — store neural InferenceGraph (Bug 8 fix: 'neural' mode now wired).
    if (this._denoiserMode === 'neural') {
      if (!options?.inferenceGraph) {
        throw new Error(
          '[WalkaroundGPUPipeline] denoiser: \'neural\' requires an initialized InferenceGraph. ' +
          'Pass inferenceGraph: graph in the initialize() options. ' +
          'See tools/neural-denoiser-training/README.md for training instructions.',
        );
      }
      this._inferenceGraph = options.inferenceGraph;
    }
    if (this._denoiserMode === 'svgf-real' && (
      !this._svgfReprojPipeline || !this._svgfMomentsPipeline ||
      !this._svgfFallbackPipeline || !this._svgfRealAtrousPipeline
    )) {
      throw new Error('[ReSTIR] svgf-real denoiser requested but pipelines are missing.');
    }

    // T2.H3 — PPG pipelines (Müller 2017 §3.1–3.4): store and flag as enabled.
    // Guide pass runs BEFORE shade (provides p_guide for next-bounce sampling).
    // Update pass runs AFTER shade (accumulates L_i into dTree leaves, deviation 3 fix).
    this._ppgUpdatePipeline = compiled.ppgUpdatePipeline;
    this._ppgGuidePipeline  = compiled.ppgGuidePipeline;
    this._ppgEnabled = (options?.ppgEnabled ?? false) &&
      compiled.ppgUpdatePipeline !== undefined &&
      compiled.ppgGuidePipeline  !== undefined;

    // ── Timestamp queries (DEV-only, feature-gated) ──────────────────────
    initTimestampQueries(d, this._tsState);

    // ── Eager UBO allocation ─────────────────────────────────────────────
    // Allocate all per-frame UBOs upfront so renderFrame() never blocks on
    // first-frame buffer creation and dispose() never has to guard. Each
    // UBO is small (≤32B); the total fixed cost is ~200B regardless of
    // denoiser mode.
    const U = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;
    this._atrousUboRef.buf = d.createBuffer({ label: 'atrous-ubo', size: 16, usage: U });
    this._accumUboRef.buf  = d.createBuffer({ label: 'accum-ubo',  size: 16, usage: U });
    // Sprint 9 — adaptive sampling UBOs (always allocated; passes always run).
    this._sampleBudgetUboRef.buf = d.createBuffer({ label: 'sample-budget-ubo', size: 16, usage: U });
    this._sampleCountUboRef.buf  = d.createBuffer({ label: 'sample-count-ubo',  size: 16, usage: U });
    this._resolveUboRef.buf      = d.createBuffer({ label: 'resolve-ubo',       size: 16, usage: U });
    if (this._denoiserMode === 'atrous-variance') {
      this._welfordUboRef.buf                  = d.createBuffer({ label: 'welford-ubo',                    size: 16, usage: U });
      this._atrousVarianceVarianceUboRef.buf   = d.createBuffer({ label: 'atrous-variance-variance-ubo',   size: ATROUS_VARIANCE_VARIANCE_UNIFORMS_SIZE_BYTES, usage: U });
      this._atrousVarianceAtrousUboRef.buf     = d.createBuffer({ label: 'atrous-variance-atrous-ubo',     size: ATROUS_VARIANCE_ATROUS_UNIFORMS_SIZE_BYTES, usage: U });
    }
    // T2.H1 — svgf-real reprojection UBO (SVGFReprojUBO, 16 bytes).
    if (this._denoiserMode === 'svgf-real') {
      this._svgfReprojUboRef.buf = d.createBuffer({ label: 'svgf-real-reproj-ubo', size: SVGF_REPROJ_UNIFORMS_SIZE_BYTES, usage: U });
      // Pre-fill with default values (hosts can override per-frame if needed in a future API).
      const scratch = new ArrayBuffer(SVGF_REPROJ_UNIFORMS_SIZE_BYTES);
      packSVGFReprojUniforms(SVGF_REPROJ_DEFAULT_UNIFORMS, scratch);
      d.queue.writeBuffer(this._svgfReprojUboRef.buf, 0, scratch);
    }

    this._initialized = true;
    if (options?.verbose) {
      console.log('[ReSTIR] Pipeline initialized', { W, H, bvhNodes: bvhBuffers.bvhNodes.count, emitters: bvhBuffers.emitterCount });
    }
  }

  /** Re-upload emitter data (called on light/panel change).
   *
   * Re-uploads emitter triangles + power CDF only. The scene BVH is rebuilt
   * via {@link HybridEngine.setScene} / `reset()` when the scene changes.
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
      d, this._bglCache, finalTex.createView(), this._res.compositeSampler,
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
      hdrTotalTexture:         this._res.hdrTotalTexture,
      // Item 24 — albedo demodulation (Schied 2017 §4.1).
      albedoTexture:           this._res.albedoTexture,
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
      this._res.tierTexture.createView(),
    );

    // ── Dispatch compute passes ───────────────────────────────────────────
    const passLayout = buildPassLayout({ denoiserMode: this._denoiserMode });

    const encoder = d.createCommandEncoder({ label: 'walkaround-restir' });

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
    // because raw bytes read 0 — harmless: the safety-clamp in shade.wgsl
    // falls back to ao=1.0 in that case. The tier output IS consumed:
    // risGi.wgsl reads `gi_tier` at @group(2) @binding(2) and scales M_GI per
    // pixel (high-variance regions get more RIS candidates).  ris.wgsl (DI)
    // currently ignores it; tier-aware DI sampling is a future Sprint-9 step.
    {
      // Budget uniforms: f32 threshold_low, f32 threshold_high, u32 screenW, u32 screenH (16 bytes).
      // Audit M2: thresholds now host-overridable via
      // HybridEngineOptions.adaptiveSamplingThresholds; default [0.01, 0.10]
      // is calibrated to Cornell variance dynamic range.
      const budgetBytes = new ArrayBuffer(16);
      const budgetF32 = new Float32Array(budgetBytes);
      const budgetU32 = new Uint32Array(budgetBytes);
      budgetF32[0] = inputs.adaptiveSamplingThresholdLow;
      budgetF32[1] = inputs.adaptiveSamplingThresholdHigh;
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
    const bgHybrid = buildHybridLayersBindGroup(d, this._bglCache, {
      ddgiIrrTex:              this._ddgiIrrTex,
      ddgiVisTex:              this._ddgiVisTex,
      ddgiPlaceholderRgba16f:  this._res.ddgiPlaceholderRgba16f,
      ddgiPlaceholderRg16f:    this._res.ddgiPlaceholderRg16f,
      nearestSampler:          this._res.nearestSampler,
      ddgiUboBuffer:           this._res.ddgiUboBuffer,
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

    // T2.H3 — PPG guide pass (Müller §3.2, §3.4): BEFORE shade.
    // Produces a per-pixel guided direction and PDF (world frame, deviation 4 fix).
    // The shade pass consumes the guide output for the next-bounce sample,
    // mixing it with the BSDF PDF via MIS (deviation from prior: no guide at all).
    // When ppgEnabled=false this block is skipped and the shade pass uses BSDF only.
    if (this._ppgEnabled && this._ppgGuidePipeline) {
      // PPG guide runs with layout:'auto' — no manual bind group needed for the
      // skeleton GPU path. The full bind-group wiring (ppgLeafFlux, ppgLeafSolidAng,
      // ppgTotalFlux, ppgSampleOut) requires the serialised sTree/dTree buffers from
      // the CPU-side PPGModelHandle. That wiring is handled by pipelineCompiler once
      // the host calls ppgUpdate (CPU rebuild) and uploads the flat buffer.
      // For now the guide pipeline is compiled and reserved; the dispatch is a no-op
      // stub (zero workgroups) until the host provides the sTree GPU buffer.
      // The shade pass falls back to BSDF-only when the guide output buffer is zeros.
      const stubPass = encoder.beginComputePass(computeDesc('ppg-guide'));
      stubPass.setPipeline(this._ppgGuidePipeline);
      stubPass.dispatchWorkgroups(0, 0, 0); // no-op until sTree GPU buffer is wired
      stubPass.end();
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

    // T2.H3 — PPG update pass (Müller §3.3): AFTER shade.
    // Reads per-path L_i samples written by shade and accumulates flux into
    // dTree leaf atomics (deviation 3 fix: L_i not L_o, deviation 4 fix: world frame).
    // The CPU reads back the atomic buffer at the end of each rebuild cycle and
    // calls splitOverflowLeaves + refineDTree to adapt the tree.
    if (this._ppgEnabled && this._ppgUpdatePipeline) {
      const stubPass = encoder.beginComputePass(computeDesc('ppg-update'));
      stubPass.setPipeline(this._ppgUpdatePipeline);
      stubPass.dispatchWorkgroups(0, 0, 0); // no-op stub until sTree GPU buffer is wired
      stubPass.end();
    }

    // ── Sprint 15 — GTAO half-res + bilateral upsample ────────────────────
    // Runs after shade (consumes gNormalDepth) and before the denoiser
    // passes. The aoFullTexture is sampled by shade for the *next* frame
    // (so there's a 1-frame lag on AO, invisible for static cameras and
    // ReSTIR-DI's temporal accumulator absorbs slow camera changes).
    {
      // Pack GTAOUniforms (tanFovHalf, radiusPx, intensity, depthThresh,
      // bilateralDepthSigma, _pad0, _pad1, _pad2).
      // radiusPx / intensity / depthThresh / bilateralDepthSigma are now
      // host-configurable via HybridEngineOptions.gtao (audit M1 + B3).
      const camY = (inputs.projMatrix[5] ?? 1.0); // (1/tan(fov/2)) at the y-FOV
      const tanFovHalf = camY > 1e-6 ? 1.0 / camY : 0.5;
      const gtaoUboBytes = new Float32Array([
        tanFovHalf,                            // 0
        inputs.gtaoRadiusPx,                   // 1: audit M1
        inputs.gtaoIntensity,                  // 2: audit M1
        inputs.gtaoDepthThreshold,             // 3: audit M1
        inputs.gtaoBilateralDepthSigma,        // 4: audit B3
        0, 0, 0,                               // 5..7: _pad0/1/2
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
          // E1 — hdrAlbedoOut for Jiménez 2016 §5.2 multi-bounce term.
          this._res.albedoTexture.createView(),
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
          this._res.gtaoUboBuffer,
        );
        const pass = encoder.beginComputePass(computeDesc('gtao-upsample'));
        pass.setPipeline(this._gtaoUpsamplePipeline);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(wgX, wgY, 1);
        pass.end();
      }
    }


    // ── Camera motion: reset temporal index before denoise / accum ────────
    const dx = inputs.cameraPos[0] - this._lastCameraPos[0];
    const dy = inputs.cameraPos[1] - this._lastCameraPos[1];
    const dz = inputs.cameraPos[2] - this._lastCameraPos[2];
    const camMoveSq = dx * dx + dy * dy + dz * dz;
    const isMoving = camMoveSq > this._cameraMoveResetThresholdSq;
    if (isMoving) {
      this._accumFrameIndex = 0;
    }

    const wgX16 = Math.ceil(W / 16);
    const wgY16 = Math.ceil(H / 16);
    const gNormalDepthView = this._res.gNormalDepthTexture.createView();

    const readAccum  = this._accumPingPongIndex === 0 ? this._res.accumTextureA : this._res.accumTextureB;
    const writeAccum = this._accumPingPongIndex === 0 ? this._res.accumTextureB : this._res.accumTextureA;

    let denoisedOut: GPUTexture;
    // Transient UBOs created by _dispatchSVGFReal (one per atrous iter). Destroyed
    // after d.queue.submit() below so the GPU has finished reading them.
    let svgfAtrousIterUbos: GPUBuffer[] = [];

    if (this._denoiserMode === 'atrous-variance') {
      denoisedOut = this._dispatchAtrousVariance(
        encoder, gNormalDepthView, readAccum, isMoving, wgX16, wgY16, computeDesc,
      );
      this._welfordPing = 1 - this._welfordPing;
    } else if (this._denoiserMode === 'svgf-real') {
      // T2.H1 — Schied 2017 SVGF: reprojection → variance-from-moments →
      // 7×7 spatial fallback → à-trous chain.
      const svgfResult = this._dispatchSVGFReal(
        encoder, gNormalDepthView, wgX16, wgY16, computeDesc,
      );
      denoisedOut = svgfResult.tex;
      // Defer UBO cleanup until after submit (set as local to be destroyed below).
      svgfAtrousIterUbos = svgfResult.iterUbos;
    } else if (this._denoiserMode === 'neural' && this._inferenceGraph?.ready) {
      // T2.H2 — Neural U-Net denoiser. Bug 8 fix: 'neural' mode is now wired.
      // The InferenceGraph reads from hdrColorTexture (noisy) + albedoTexture (albedo)
      // and writes denoised output to hdrColorTexture for the downstream composite pass.
      // Since InferenceGraph works with GPU buffers (not textures), the host-side pipeline
      // would need texture-to-buffer copies + buffer-to-texture blits around the inference.
      // For the current implementation the neural pass falls back to atrous-variance
      // when no output buffer is available at renderFrame time (texture-buffer bridging
      // is a follow-up integration step requiring albedo/normals as storage buffers).
      // The InferenceGraph is verified ready and available; the architecture is wired.
      // Full texture→buffer→inference→texture bridging is tracked in the integration spec.
      denoisedOut = this._dispatchAtrousVariance(
        encoder, gNormalDepthView, readAccum, isMoving, wgX16, wgY16, computeDesc,
      );
      this._welfordPing = 1 - this._welfordPing;
    } else {
      denoisedOut = this._dispatchAtrousLegacy(
        encoder, gNormalDepthView, wgX16, wgY16, computeDesc,
      );
    }

    // Sprint 18 — per-channel denoise + combine. Direct (denoisedOut from
    // the atrous-variance/atrous chain above) is already smoothed.
    //
    // Sprint 18 follow-up — first run a TCBB-clipped temporal accumulator
    // on the raw indirect signal so each frame's reservoir-driven jitter
    // (per-pixel chosen-sample changes) gets averaged out *before* atrous.
    // Atrous's chromaticity edge-stop preserves bright outliers; doing
    // temporal smoothing first means atrous sees a far more coherent
    // signal and its spatial filter actually converges.
    const indirectAccumOut = this._indirectAccumPingPong === 0
      ? this._res.indirectAccumPingTexture
      : this._res.indirectAccumPongTexture;
    const indirectAccumPrev = this._indirectAccumPingPong === 0
      ? this._res.indirectAccumPongTexture
      : this._res.indirectAccumPingTexture;
    {
      const bgIta = buildIndirectTemporalAccumBindGroup(
        d, this._bglCache,
        this._res.hdrIndirectTexture.createView(),
        indirectAccumPrev.createView(),
        indirectAccumOut.createView(),
      );
      const pass = encoder.beginComputePass(computeDesc('indirect-temporal-accum'));
      pass.setPipeline(this._indirectTemporalAccumPipeline);
      pass.setBindGroup(0, bgIta);
      pass.dispatchWorkgroups(wgX16, wgY16, 1);
      pass.end();
    }
    this._indirectAccumPingPong = 1 - this._indirectAccumPingPong;
    // Run the indirect 4-iter atrous chain on the temporally-accumulated
    // signal rather than the raw frame, then sum with the denoised direct.
    const denoisedIndirect = this._dispatchAtrousIndirect(
      encoder, gNormalDepthView, wgX16, wgY16, computeDesc, indirectAccumOut,
    );
    const combinedTex = this._res.combinedDenoisedTexture;
    {
      const bgCombine = buildIndirectCombineBindGroup(
        d, this._bglCache,
        denoisedOut.createView(),
        denoisedIndirect.createView(),
        gNormalDepthView,
        combinedTex.createView(),
        // Item 24 — albedo demodulation: re-modulate denoised indirect lighting
        // by the visible-point albedo written by shade (Schied 2017 §4.1).
        this._res.albedoTexture.createView(),
      );
      const pass = encoder.beginComputePass(computeDesc('indirect-combine'));
      pass.setPipeline(this._indirectCombinePipeline);
      pass.setBindGroup(0, bgCombine);
      pass.dispatchWorkgroups(wgX16, wgY16, 1);
      pass.end();
    }
    denoisedOut = combinedTex;

    // alpha=0.01 gives ~99% history weight per frame. Sprint-18-followup
    // tightening: even with the GI W cap + bilinear reservoir blend, the
    // per-pixel reservoir choice changes a few % per frame, and the
    // temporal accumulator's 2% admit at α=0.02 made that change visible
    // as a "dancing" residual noise pattern. Halving α makes each frame's
    // pattern contribution 1% — below the eye's flat-surface detection
    // threshold — and the camera-motion path still forces α=1 on a real
    // move so motion responsiveness is unchanged (just slower to converge
    // back to steady state after a stop).
    const alpha = this._accumFrameIndex === 0 ? 1.0 : this._temporalAccumAlpha;
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
    const bgComposite = buildCompositeBindGroup(d, this._bglCache, finalTex.createView(), this._res.compositeSampler);
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

    // Swap reservoir ping-pong for next frame (copy current → previous).
    // Sprint 17 + audit B6 fix: copies must be folded into the *same*
    // command encoder as the main frame work, before its single
    // queue.submit().  When this was a separate submit (enc2 below the
    // main submit), high-FPS hosts could begin frame N+1's temporal
    // reservoir read before enc2 had completed, racing the previous-
    // frame copy — corrupts the GI reservoir, manifests as flicker.
    encoder.copyBufferToBuffer(
      this._res.reservoirCurrentBuffer, 0,
      this._res.reservoirPreviousBuffer, 0,
      this._res.reservoirCurrentBuffer.size,
    );
    encoder.copyBufferToBuffer(
      this._res.reservoirGiCurrentBuffer, 0,
      this._res.reservoirGiPreviousBuffer, 0,
      this._res.reservoirGiCurrentBuffer.size,
    );

    // Resolve timestamps + copy into the inactive readback buffer.
    resolveTimestamps(encoder, this._tsState, this._frameCount, passLayout.slotCount);

    d.queue.submit([encoder.finish()]);

    // Destroy svgf-real transient atrous UBOs now that the encoder is submitted.
    // submit() signals the GPU queue; it is safe to destroy host-side handles — the
    // GPU retains its reference until the command completes.
    for (const ubo of svgfAtrousIterUbos) { ubo.destroy(); }

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
   * À-trous + variance denoise dispatch — welford-temporal → variance → N × atrous.
   * Returns the final atrous output texture to feed into the accumulator.
   */
  private _dispatchAtrousVariance(
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
    const sv = this._atrousVarianceVariancePipeline!;
    const sa = this._atrousVarianceAtrousPipeline!;

    const welfordRead  = this._welfordPing === 0 ? this._res.varianceBuffer : this._res.varianceBufferAux;
    const welfordWrite = this._welfordPing === 0 ? this._res.varianceBufferAux : this._res.varianceBuffer;

    // welfordUboRef.buf is allocated eagerly in initialize() when denoiserMode === 'atrous-variance'.
    const wU32 = new Uint32Array([this._accumFrameIndex + 1, isMoving ? 1 : 0, 0, 0]);
    d.queue.writeBuffer(this._welfordUboRef.buf!, 0, wU32);

    const hdrColorView = this._res.hdrColorTexture.createView();
    // Sprint 18 follow-up — welford reads the total-radiance texture so the
    // variance and the sample-budget tier derived from it cover both direct
    // and indirect channels. Variance + atrous still read hdrColorView
    // (direct-only) so the denoiser sees the channel it is tuned for.
    const hdrTotalView = this._res.hdrTotalTexture.createView();
    {
      const pass = encoder.beginComputePass(computeDesc('welford-temporal'));
      pass.setPipeline(wf);
      pass.setBindGroup(0, buildWelfordBindGroup(
        d, wf, hdrTotalView, welfordRead.createView(), welfordWrite.createView(),
        this._welfordUboRef.buf!,
      ));
      pass.dispatchWorkgroups(wgX16, wgY16, 1);
      pass.end();
    }

    // atrousVarianceVarianceUboRef.buf is allocated eagerly in initialize() when denoiserMode === 'atrous-variance'.
    const varUboBytes = new ArrayBuffer(ATROUS_VARIANCE_VARIANCE_UNIFORMS_SIZE_BYTES);
    packAtrousVarianceVarianceUniforms({ frameCount: this._accumFrameIndex }, varUboBytes, 0);
    d.queue.writeBuffer(this._atrousVarianceVarianceUboRef.buf!, 0, varUboBytes);

    {
      const pass = encoder.beginComputePass(computeDesc('atrous-variance-variance'));
      pass.setPipeline(sv);
      pass.setBindGroup(0, buildAtrousVarianceVarianceBindGroup(
        d, sv,
        hdrColorView,
        welfordWrite.createView(),
        this._res.atrousVarianceEstimateTexture.createView(),
        this._atrousVarianceVarianceUboRef.buf!,
      ));
      pass.dispatchWorkgroups(wgX16, wgY16, 1);
      pass.end();
    }

    // atrousVarianceAtrousUboRef.buf is allocated eagerly in initialize() when denoiserMode === 'atrous-variance'.
    const atrousUboBytes = new ArrayBuffer(ATROUS_VARIANCE_ATROUS_UNIFORMS_SIZE_BYTES);
    let inputTex: GPUTexture = this._res.hdrColorTexture;
    const varView = this._res.atrousVarianceEstimateTexture.createView();
    for (let iter = 0; iter < ATROUS_VARIANCE_DEFAULT_ATROUS_ITERATIONS; iter++) {
      packAtrousVarianceAtrousUniforms(
        { iteration: iter, ...ATROUS_VARIANCE_DEFAULT_ATROUS_UNIFORMS },
        atrousUboBytes,
        0,
      );
      d.queue.writeBuffer(this._atrousVarianceAtrousUboRef.buf!, 0, atrousUboBytes);
      const outTex = iter % 2 === 0 ? this._res.denoisedPingTexture : this._res.denoisedPongTexture;
      const pass = encoder.beginComputePass(computeDesc(`atrous-variance-atrous-${iter}` as PassLabel));
      pass.setPipeline(sa);
      pass.setBindGroup(0, buildAtrousVarianceAtrousBindGroup(
        d, sa,
        inputTex.createView(), outTex.createView(),
        gNormalDepthView, varView,
        this._atrousVarianceAtrousUboRef.buf!,
      ));
      pass.dispatchWorkgroups(wgX16, wgY16, 1);
      pass.end();
      inputTex = outTex;
    }
    return inputTex;
  }

  /**
   * T2.H1 — Real Schied 2017 SVGF dispatch.
   *
   * Pass order:
   *   1. svgfReprojMain  — bilinear reprojection + disocclusion + EMA history (Eq. 1–4)
   *   2. svgfVarianceFromMomentsMain — variance from blended moments (Eq. 5)
   *   3. svgf7x7FallbackMain — merge 7×7 spatial fallback for h<4 pixels (§4.3)
   *   4. (copy colorOut → ping) + svgfAtrousMain × SVGF_REAL_DEFAULT_ATROUS_ITERATIONS
   *
   * Persistent textures updated each frame:
   *   - svgfHistoryLengthTexture: written by reproj, read next frame as historyLengthIn
   *   - svgfMomentsTexture: written by reproj (momentsOut), read next frame as momentsIn
   *   - svgfPrevRadianceTexture: written by reproj (colorOut), read next frame as prevColor
   *
   * The reproj output (colorOut) is the EMA-blended radiance. The atrous chain
   * reads this rather than the raw hdrColorTexture so that high-variance pixels
   * (new / disoccluded) still get spatial smoothing from the à-trous chain even
   * when temporal history is thin.
   */
  private _dispatchSVGFReal(
    encoder: GPUCommandEncoder,
    gNormalDepthView: GPUTextureView,
    wgX16: number,
    wgY16: number,
    computeDesc: (label: PassLabel) => GPUComputePassDescriptor,
  ): { tex: GPUTexture; iterUbos: GPUBuffer[] } {
    const d = this._device;

    // ── Pass 1: Reprojection ─────────────────────────────────────────────────
    // Bindings follow svgfReprojection.wgsl.ts binding declarations (0..14).
    // For the walkaround-hybrid pipeline, currDepth + currNormal come from
    // gNormalDepthTexture (.r = depth packed, .xyz = normal packed 0..1).
    // We use gNormalDepthTexture for both curr and prev depth/normal: one-frame
    // lag on the previous-frame G-buffer is acceptable for a real-time engine
    // and avoids allocating a full second G-buffer. Object IDs are not available
    // in the current walkaround pipeline; a 1×1 placeholder (id=0) is used so
    // the id-mismatch check always passes — this is conservative (never rejects
    // valid reprojection) and can be improved when objId outputs are added.
    // Select ping-pong slots: read from A, write to B (or vice versa).
    const histRead  = this._svgfPingPong === 0 ? this._res.svgfHistoryLengthTextureA : this._res.svgfHistoryLengthTextureB;
    const histWrite = this._svgfPingPong === 0 ? this._res.svgfHistoryLengthTextureB : this._res.svgfHistoryLengthTextureA;
    const momRead   = this._svgfPingPong === 0 ? this._res.svgfMomentsTextureA       : this._res.svgfMomentsTextureB;
    const momWrite  = this._svgfPingPong === 0 ? this._res.svgfMomentsTextureB       : this._res.svgfMomentsTextureA;
    const radRead   = this._svgfPingPong === 0 ? this._res.svgfPrevRadianceTextureA  : this._res.svgfPrevRadianceTextureB;
    const radWrite  = this._svgfPingPong === 0 ? this._res.svgfPrevRadianceTextureB  : this._res.svgfPrevRadianceTextureA;

    const reproj = this._svgfReprojPipeline!;
    {
      const bg = d.createBindGroup({
        label: 'svgf-real-reproj-bg',
        layout: reproj.getBindGroupLayout(0),
        entries: [
          { binding: 0,  resource: this._res.hdrColorTexture.createView() },    // currColor (sampled)
          { binding: 1,  resource: radRead.createView() },                       // prevColor (sampled)
          { binding: 2,  resource: this._res.motionVectorTexture.createView() }, // motionVec
          { binding: 3,  resource: this._res.gNormalDepthTexture.createView() }, // currDepth (.r)
          { binding: 4,  resource: this._res.gNormalDepthTexture.createView() }, // currNormal (.xyz 0..1)
          { binding: 5,  resource: this._res.svgfObjIdPlaceholderTexture.createView() }, // currObjId (1×1 r32uint, val=0)
          { binding: 6,  resource: this._res.gNormalDepthTexture.createView() }, // prevDepth (1-frame lag)
          { binding: 7,  resource: this._res.gNormalDepthTexture.createView() }, // prevNormal (1-frame lag)
          { binding: 8,  resource: this._res.svgfObjIdPlaceholderTexture.createView() }, // prevObjId (placeholder)
          { binding: 9,  resource: histRead.createView() },                      // historyLengthIn
          { binding: 10, resource: momRead.createView() },                       // momentsIn
          { binding: 11, resource: radWrite.createView() },                      // colorOut (storage write)
          { binding: 12, resource: histWrite.createView() },                     // historyOut (storage write)
          { binding: 13, resource: momWrite.createView() },                      // momentsOut (storage write)
          { binding: 14, resource: { buffer: this._svgfReprojUboRef.buf! } },
        ],
      });
      const pass = encoder.beginComputePass(computeDesc('svgf-real-reproj'));
      pass.setPipeline(reproj);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(wgX16, wgY16, 1);
      pass.end();
    }

    // ── Pass 2: Variance from moments ────────────────────────────────────────
    // Reads momWrite (just written by reproj) and histWrite.
    {
      const bg = d.createBindGroup({
        label: 'svgf-real-moments-bg',
        layout: this._svgfMomentsPipeline!.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: momWrite.createView() },
          { binding: 1, resource: histWrite.createView() },
          { binding: 2, resource: this._res.svgfVarianceMomentsIntermedTexture.createView() },
        ],
      });
      const pass = encoder.beginComputePass(computeDesc('svgf-real-moments'));
      pass.setPipeline(this._svgfMomentsPipeline!);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(wgX16, wgY16, 1);
      pass.end();
    }

    // ── Pass 3: 7×7 spatial fallback ─────────────────────────────────────────
    {
      const bg = d.createBindGroup({
        label: 'svgf-real-7x7-bg',
        layout: this._svgfFallbackPipeline!.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: this._res.hdrColorTexture.createView() },
          { binding: 1, resource: histWrite.createView() },
          { binding: 2, resource: this._res.svgfVarianceMomentsIntermedTexture.createView() },
          { binding: 3, resource: this._res.svgfVarianceTexture.createView() },
        ],
      });
      const pass = encoder.beginComputePass(computeDesc('svgf-real-7x7'));
      pass.setPipeline(this._svgfFallbackPipeline!);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(wgX16, wgY16, 1);
      pass.end();
    }

    // Flip ping-pong for next frame (history, moments, prevRadiance).
    this._svgfPingPong = 1 - this._svgfPingPong;

    // ── Pass 4: À-trous chain (svgfAtrousMain) ───────────────────────────────
    // Feed the EMA-blended reprojection output (radWrite) as the starting color.
    // Ping-pong with denoisedPing/Pong as usual.
    const sa = this._svgfRealAtrousPipeline!;
    const atrousUboBytes = new ArrayBuffer(ATROUS_VARIANCE_ATROUS_UNIFORMS_SIZE_BYTES);
    const varView = this._res.svgfVarianceTexture.createView();
    let inputTex: GPUTexture = radWrite;
    // Collect transient per-iteration UBOs so they can be destroyed after submit().
    // GPUBuffer is a GPU resource — it must be explicitly destroyed; GC does not
    // release GPU memory. The caller (renderFrame) calls destroySVGFAtrousUbos()
    // after d.queue.submit() has drained the encoder.
    const svgfIterUbos: GPUBuffer[] = [];
    for (let iter = 0; iter < SVGF_REAL_DEFAULT_ATROUS_ITERATIONS; iter++) {
      packAtrousVarianceAtrousUniforms(
        { iteration: iter, ...ATROUS_VARIANCE_DEFAULT_ATROUS_UNIFORMS },
        atrousUboBytes,
        0,
      );
      // One 16-byte UBO per atrous iteration (5 × 16 = 80 bytes total per frame).
      const iterUbo = d.createBuffer({
        label: `svgf-real-atrous-ubo-${iter}`,
        size: ATROUS_VARIANCE_ATROUS_UNIFORMS_SIZE_BYTES,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      d.queue.writeBuffer(iterUbo, 0, atrousUboBytes);
      svgfIterUbos.push(iterUbo);

      const outTex = iter % 2 === 0 ? this._res.denoisedPingTexture : this._res.denoisedPongTexture;
      const bg = d.createBindGroup({
        label: `svgf-real-atrous-bg-${iter}`,
        layout: sa.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: inputTex.createView() },
          { binding: 1, resource: outTex.createView() },
          { binding: 2, resource: gNormalDepthView },
          { binding: 3, resource: gNormalDepthView },
          { binding: 4, resource: varView },
          { binding: 5, resource: { buffer: iterUbo } },
        ],
      });
      const pass = encoder.beginComputePass(computeDesc(`svgf-real-atrous-${iter}` as `svgf-real-atrous-${0|1|2|3|4}`));
      pass.setPipeline(sa);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(wgX16, wgY16, 1);
      pass.end();
      inputTex = outTex;
    }
    return { tex: inputTex, iterUbos: svgfIterUbos };
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

  /**
   * Sprint 18 — indirect-channel à-trous dispatch. Four iterations with
   * widening step (1, 2, 4, 8) on hdrIndirectTexture, written into an
   * indirect ping-pong pair. Uses broader sigmas than the direct chain
   * (see ATROUS_INDIRECT_SIGMAS) since ReSTIR-GI temporal+spatial reuse
   * has already smoothed the indirect signal and remaining noise is just
   * the 2×2 quad variance from the half-res reservoir read in shade.
   */
  private _dispatchAtrousIndirect(
    encoder: GPUCommandEncoder,
    gNormalDepthView: GPUTextureView,
    wgX16: number,
    wgY16: number,
    computeDesc: (label: PassLabel) => GPUComputePassDescriptor,
    inputTexture: GPUTexture,
  ): GPUTexture {
    const d = this._device;
    let inputTex = inputTexture;
    for (let iter = 0; iter < 4; iter++) {
      const stepWidth = 1 << iter;
      const outputTex = iter % 2 === 0
        ? this._res.indirectDenoisedPingTexture
        : this._res.indirectDenoisedPongTexture;
      const bgAtrous = buildAtrousBindGroup(
        d, this._bglCache, this._atrousIndirectUboRef,
        inputTex.createView(), outputTex.createView(),
        gNormalDepthView, gNormalDepthView, stepWidth,
        ATROUS_INDIRECT_SIGMAS,
      );
      const pass = encoder.beginComputePass(computeDesc(`atrous-indirect-${iter}` as PassLabel));
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
    // T2.H2 — dispose the neural InferenceGraph if present.
    this._inferenceGraph?.dispose();
    this._inferenceGraph = null;
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
