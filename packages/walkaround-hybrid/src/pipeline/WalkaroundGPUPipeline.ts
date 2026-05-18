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
 * **W1-R5 — declarative pass order.** All 18 non-denoiser stages above are
 * implemented as self-contained {@link Pass} entries under
 * `pipeline/passes/` and registered with a {@link PassRegistry}. Frame
 * dispatch is now a pass-loop sandwiching the polymorphic denoiser
 * dispatch (which lives in {@link DenoiserRegistry} from W1-R3/R4). Adding
 * a non-denoiser pass is a single new file + one `register()` call below.
 * The position-encoded ordering inside this file was the largest single
 * source of integration complexity per the 2026-05-17 sweep (Theme B / B2
 * / B5 / B6).
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
  allocatePPGResources,
  type FrameResources,
} from './resourceManager.js';
import { buildSTree } from '../ppg/sTree.js';
import type { AABB, STree } from '../ppg/types.js';
import { serialiseSTree } from '../ppg/serialise.js';
import { PPG_CELL_SPLIT_THRESHOLD, PPG_MAX_SPATIAL_CELLS, PPG_MIS_ALPHA } from '../ppg/ppgConstants.js';
import { STreeRefinementScheduler } from '../ppg/refinementScheduler.js';
import {
  type BGLCache,
} from './bindGroupLayouts.js';
import {
  buildFrameBindGroup,
  buildSceneBindGroup,
  buildUboBindGroup,
  buildHybridLayersBindGroup,
  buildCompositeBindGroup,
  type UboRef,
} from './bindGroupBuilders.js';
import {
  DenoiserRegistry,
  type Denoiser,
  type DenoiserId,
} from './denoisers/index.js';
import { registerBuiltinDenoisers } from './denoisers/registerBuiltinDenoisers.js';
import { PassRegistry } from './PassRegistry.js';
import type {
  Pass,
  PassDispatchContext,
  PassFrameState,
  PassGateOptions,
} from './Pass.js';
import {
  AtrousIndirectPass,
  CompositePass,
  GTAOPass,
  GTAOUpsamplePass,
  IndirectCombinePass,
  IndirectTemporalAccumPass,
  PPGGuidePass,
  PPGUpdatePass,
  ResolvePass,
  RISGIPass,
  RISPass,
  SampleBudgetPass,
  ShadePass,
  SpatialGIReservoirPass,
  SpatialReservoirPass,
  TemporalAccumPass,
  TemporalGIReservoirPass,
  TemporalReservoirPass,
} from './passes/index.js';
import type { PingPongRef } from './passes/passRefs.js';
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

/**
 * W9 — derive a world-space AABB for the PPG sTree from the uploaded BVH data.
 *
 * The walkaround pipeline doesn't surface its scene bounds as a first-class
 * field; we recover them by scanning the BVH position buffer (which the host
 * always uploads, per `restir/bvhCompute.ts`). If the buffer is empty we
 * fall back to a generous default that contains any plausible scene.
 *
 * Phase 1: this AABB is used for two things — the sTree root cell extents
 * (so adaptive splits subdivide the actual scene volume), and the placeholder
 * "scene centre" query position uploaded into the guide UBO (until Phase 2
 * wires a per-pixel surface-position buffer).
 */
function derivePPGSceneAABB(bvh: { bvhPositions: { cpuData: ArrayBuffer; count: number } }): AABB {
  const view = new Float32Array(bvh.bvhPositions.cpuData);
  if (view.length < 4) {
    return { min: [-10, -10, -10], max: [10, 10, 10] };
  }
  // BVH position layout: vec4f per vertex (xyz + packed UV in w). Stride 4.
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (let i = 0; i + 3 <= view.length; i += 4) {
    const x = view[i]!, y = view[i + 1]!, z = view[i + 2]!;
    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (z < minZ) minZ = z;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
    if (z > maxZ) maxZ = z;
  }
  if (!Number.isFinite(minX) || !Number.isFinite(maxX)) {
    return { min: [-10, -10, -10], max: [10, 10, 10] };
  }
  // Pad by 1% to avoid edge-case boundary queries.
  const padX = (maxX - minX) * 0.01 + 1e-3;
  const padY = (maxY - minY) * 0.01 + 1e-3;
  const padZ = (maxZ - minZ) * 0.01 + 1e-3;
  return {
    min: [minX - padX, minY - padY, minZ - padZ],
    max: [maxX + padX, maxY + padY, maxZ + padZ],
  };
}

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

/** Index in `sortedPasses` AFTER which the orchestrator runs the
 *  polymorphic denoiser dispatch. Resolved at registry construction time
 *  by finding `gtao-upsample`. Honest layering note: the denoiser is a
 *  separate concept from pass scheduling — see
 *  `denoisers/index.ts::Denoiser`. Pre-W1-R5 this split lived as a
 *  position-encoded line in renderFrame; W1-R5 keeps the manual
 *  `_activeDenoiser.dispatch()` call here rather than wrapping the
 *  denoiser as a virtual Pass, so the layering distinction stays
 *  visible. */
const DENOISER_AFTER_PASS_ID = 'gtao-upsample';

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

  /** Shared à-trous pipeline. Used by the legacy `AtrousDenoiser` (passed
   *  in via the dispatch context) AND by the always-on
   *  {@link AtrousIndirectPass}. Compiled once in pipelineCompiler and
   *  shared by both consumers (rationale: identical shader / BGL — forking
   *  a private compile per consumer would double the boot cost for zero
   *  functional benefit). */
  private _atrousPipeline!: GPUComputePipeline;
  /** Active denoiser (looked up from `_denoiserRegistry` after init). */
  private _denoiserMode: DenoiserId = 'atrous-variance';
  /** Registry of all built-in denoisers; populated once at boot. */
  private _denoiserRegistry: DenoiserRegistry | null = null;
  /** The active denoiser instance for this pipeline (set in initialize). */
  private _activeDenoiser: Denoiser | null = null;
  /** Registry of non-denoiser passes; populated once at boot. */
  private _passRegistry: PassRegistry | null = null;
  /** Sorted pass list cached at boot; reused across frames. */
  private _sortedPasses: readonly Pass[] = [];
  /** Index of the pass after which the denoiser dispatch fires. */
  private _denoiserSplitIndex = -1;
  /** T2.H2 — neural denoiser InferenceGraph; kept for future W10 wiring. */
  private _inferenceGraph: InferenceGraph | null = null;
  /** Audit B8 — populated at initialize() time from HybridEngineOptions. */
  private _cameraMoveResetThresholdSq = DEFAULT_CAMERA_MOVE_RESET_THRESHOLD_SQ;
  /** Audit M3 — populated at initialize() time from HybridEngineOptions. */
  private _temporalAccumAlpha = DEFAULT_TEMPORAL_ACCUM_ALPHA;
  /** T2.H3 — PPG is enabled iff both pipelines were compiled successfully. */
  private _ppgEnabled = false;
  /** W9 — CPU-side PPG model (sTree + per-cell dTrees). Allocated at
   *  initialize() when ppgEnabled is true; serialised to GPU buffers per
   *  frame (Phase 1: static empty tree uploaded once). */
  private _ppgSTree: STree | null = null;
  /** Scene-bounds AABB carried in the guide UBO so the kernel can map flat
   *  pixel indices to the (placeholder) world-space query position for sTree
   *  descent. Set from the BVH bounds at initialize() time. */
  private _ppgSceneAABB: AABB = { min: [-10, -10, -10], max: [10, 10, 10] };
  /**
   * W9 — sTree refinement scheduler. Reads back `fluxAtomicsBuf` every
   * `intervalFrames` frames, decides whether refinement is warranted via
   * the samples-increasing OR loss-decreasing heuristic, calls
   * `splitOverflowLeaves` on the CPU mirror sTree, and re-uploads the
   * serialised buffers. See `refinementScheduler.ts`. Null when
   * `_ppgEnabled === false`.
   */
  private _ppgRefinement: STreeRefinementScheduler | null = null;
  /** Cached `dTreeOffsets` from the last `serialiseSTree` call. The
   *  refinement readback needs this to map atomic slots back to dTree
   *  leaves; re-serialising just to recover it would be wasteful. */
  private _ppgDTreeOffsets: Uint32Array | null = null;
  /** Sprint 18 follow-up — ping-pong index for the indirect temporal
   *  accumulator. Lives on the pipeline because the value persists across
   *  frames; the {@link IndirectTemporalAccumPass} reads + advances it
   *  through a {@link PingPongRef} wrapper. */
  private _indirectAccumPingPongRef: PingPongRef = { value: 0 };
  private _swapChainFormat: GPUTextureFormat = 'bgra8unorm';

  // Bind group layout memoisation cache
  private _bglCache: BGLCache = {};

  // Per-pass UBO buffers owned by the pipeline (i.e. NOT owned by a
  // denoiser — denoiser-private UBOs are field-owned by each Denoiser
  // implementation under `denoisers/`). Two access patterns coexist:
  //  - Builder-managed (lazy): _atrousIndirectUboRef and _accumUboRef
  //    are passed by reference into buildAtrousBindGroup /
  //    buildAccumBindGroup, which lazy-allocate on first call so each
  //    builder owns its UBO lifetime.
  //  - Eager: the adaptive-sampling UBOs are allocated in initialize().
  // dispose() walks all via the `_perPassUboRefs` array below so adding
  // a new UBO only requires registering it there.
  /** Sprint 18 — separate UBO for the indirect-channel atrous chain so it
   *  doesn't race the legacy denoiser's per-iteration sigma writes. */
  private _atrousIndirectUboRef: UboRef = { buf: undefined };
  private _accumUboRef: UboRef  = { buf: undefined };
  // Sprint 9 — adaptive sampling UBOs.
  private _sampleBudgetUboRef: UboRef = { buf: undefined };
  private _sampleCountUboRef:  UboRef = { buf: undefined };
  private _resolveUboRef:      UboRef = { buf: undefined };
  private get _perPassUboRefs(): readonly UboRef[] {
    return [
      this._atrousIndirectUboRef,
      this._accumUboRef,
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
    const layout = buildPassLayout({ denoiserMode: this._denoiserMode });
    return readTimestampsOnce(this._device, this._tsState, layout);
  }

  /** Upload BVH data + compile shaders. Must be called once before renderFrame. */
  async initialize(
    bvhBuffers: SceneBVHBuffers,
    swapChainFormat: GPUTextureFormat = 'bgra8unorm',
    options?: {
      verbose?: boolean;
      denoiser?: DenoiserId;
      /** Audit B8 — host-overridable camera-move temporal-reset threshold. */
      cameraMoveResetThresholdSq?: number;
      /** Audit M3 — host-overridable temporal-accumulator EMA weight. */
      temporalAccumAlpha?: number;
      /** T2.H2 — neural denoiser InferenceGraph (required when denoiser='neural').
       *  Kept on the options surface for forward compatibility with W10. */
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

    // ── Compile shaders (denoiser-agnostic) ───────────────────────────────
    const compiled = await compilePipelines(d, this._bglCache, swapChainFormat, {
      verbose: options?.verbose ?? false,
      ppgEnabled: options?.ppgEnabled ?? false,
    });
    // Shared à-trous pipeline — fed into the AtrousDenoiser context AND
    // the always-on AtrousIndirectPass.
    this._atrousPipeline = compiled.atrousPipeline;
    this._denoiserMode = options?.denoiser ?? 'atrous-variance';
    this._cameraMoveResetThresholdSq = options?.cameraMoveResetThresholdSq
      ?? DEFAULT_CAMERA_MOVE_RESET_THRESHOLD_SQ;
    this._temporalAccumAlpha = options?.temporalAccumAlpha
      ?? DEFAULT_TEMPORAL_ACCUM_ALPHA;

    // T2.H3 — PPG is enabled iff host opted-in AND both pipelines compiled.
    this._ppgEnabled = (options?.ppgEnabled ?? false) &&
      compiled.ppgUpdatePipeline !== undefined &&
      compiled.ppgGuidePipeline  !== undefined;

    // ── Denoiser registry: build, register builtins, look up + initialise
    //    the active denoiser. Disabled placeholders (neural / oidn-final)
    //    are registered but never reach `initialize()` — the registry
    //    rejects them at `lookup()` time with a clear error pointing at
    //    the workstream that will land the real implementation.
    this._denoiserRegistry = new DenoiserRegistry();
    registerBuiltinDenoisers(this._denoiserRegistry);
    this._activeDenoiser = this._denoiserRegistry.lookup(this._denoiserMode);
    await this._activeDenoiser.initialize({
      device: d,
      width: W,
      height: H,
      bglCache: this._bglCache,
      frameResources: this._res,
    });

    // Forward-compat: a host may still supply an InferenceGraph via options
    // even though the `neural` denoiser is `disabled: true` and lookup
    // would have already thrown above. We store the handle (no error
    // raised here) so a future test exercising W10 wiring can read it
    // back. The walkaround path no longer silently substitutes
    // atrous-variance for neural — that fallback was removed in W1-R3.
    if (options?.inferenceGraph) {
      this._inferenceGraph = options.inferenceGraph;
    }

    // ── Timestamp queries (DEV-only, feature-gated) ──────────────────────
    initTimestampQueries(d, this._tsState);

    // ── Eager UBO allocation ─────────────────────────────────────────────
    // Allocate the pipeline-owned per-frame UBOs upfront so renderFrame()
    // never blocks on first-frame buffer creation. Denoiser-owned UBOs
    // are allocated inside each `Denoiser.initialize()`.
    const U = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;
    this._accumUboRef.buf  = d.createBuffer({ label: 'accum-ubo',  size: 16, usage: U });
    // Sprint 9 — adaptive sampling UBOs (always allocated; passes always run).
    this._sampleBudgetUboRef.buf = d.createBuffer({ label: 'sample-budget-ubo', size: 16, usage: U });
    this._sampleCountUboRef.buf  = d.createBuffer({ label: 'sample-count-ubo',  size: 16, usage: U });
    this._resolveUboRef.buf      = d.createBuffer({ label: 'resolve-ubo',       size: 16, usage: U });

    // ── Pass registry: instantiate + register all non-denoiser passes ────
    // Order of registration is irrelevant; the registry topologically sorts.
    const registry = new PassRegistry();
    registry.register(new SampleBudgetPass(
      compiled.sampleBudgetPipeline,
      this._sampleBudgetUboRef,
      this._sampleCountUboRef,
    ));
    registry.register(new RISPass(compiled.risPipeline));
    registry.register(new TemporalReservoirPass(compiled.temporalPipeline));
    registry.register(new SpatialReservoirPass(compiled.spatialPipeline));
    registry.register(new RISGIPass(compiled.risGiPipeline));
    registry.register(new TemporalGIReservoirPass(compiled.temporalGiPipeline));
    registry.register(new SpatialGIReservoirPass(compiled.spatialGiPipeline));
    registry.register(new ShadePass(compiled.shadePipeline));
    registry.register(new GTAOPass(compiled.gtaoPipeline));
    registry.register(new GTAOUpsamplePass(compiled.gtaoUpsamplePipeline));
    registry.register(new IndirectTemporalAccumPass(
      compiled.indirectTemporalAccumPipeline,
      this._indirectAccumPingPongRef,
    ));
    registry.register(new AtrousIndirectPass(
      compiled.atrousPipeline,
      this._atrousIndirectUboRef,
    ));
    registry.register(new IndirectCombinePass(compiled.indirectCombinePipeline));
    registry.register(new TemporalAccumPass(compiled.accumPipeline, this._accumUboRef));
    registry.register(new ResolvePass(compiled.resolvePipeline, this._resolveUboRef));
    registry.register(new CompositePass(compiled.compositePipeline));
    // PPG passes — only register when the pipelines compiled successfully.
    // The `gates()` predicate gates dispatch on `opts.ppgEnabled` so they
    // can be registered unconditionally here, but skipping registration
    // when the pipeline is undefined avoids holding a stale field.
    if (compiled.ppgGuidePipeline) {
      registry.register(new PPGGuidePass(compiled.ppgGuidePipeline));
    }
    if (compiled.ppgUpdatePipeline) {
      registry.register(new PPGUpdatePass(compiled.ppgUpdatePipeline));
    }

    // ── Initialize all passes in parallel ────────────────────────────────
    this._passRegistry = registry;
    this._sortedPasses = registry.sortedPasses();
    this._denoiserSplitIndex = this._sortedPasses.findIndex(
      (p) => p.id === DENOISER_AFTER_PASS_ID,
    );
    if (this._denoiserSplitIndex < 0) {
      throw new Error(
        `WalkaroundGPUPipeline: pass "${DENOISER_AFTER_PASS_ID}" not registered`,
      );
    }
    await Promise.all(this._sortedPasses.map((p) => p.initialize({
      device: d, width: W, height: H, bglCache: this._bglCache, frameResources: this._res,
    })));

    // ── W9 — PPG GPU buffer init (opt-in) ────────────────────────────────
    // When ppgEnabled, allocate the static PPG storage buffers, build a fresh
    // sTree (single-cell at scene bounds), serialise it, and upload to the
    // GPU. The kernels descend the serialised buffers each frame; the CPU
    // refines + re-uploads on rebuild cycles (Phase 2 follow-up).
    if (this._ppgEnabled) {
      // Derive scene bounds from the uploaded BVH if available — for Phase 1
      // we use a generous default that contains any plausible walkaround
      // scene. The world-space query position is currently the scene centre
      // (see ppgGuide.wgsl.ts), so the exact bound doesn't drive correctness
      // until Phase 2 wires a per-pixel position buffer.
      this._ppgSceneAABB = derivePPGSceneAABB(bvhBuffers);
      this._ppgSTree = buildSTree(this._ppgSceneAABB);
      allocatePPGResources(d, this._res, W, H);
      this._uploadPPGTree();
      this._writePPGGuideUBO();
      this._writePPGUpdateUBO();
      // W9 — refinement scheduler: every N frames, read back the GPU's
      // ppgFluxAtomics, decide whether to split sTree leaves via the
      // samples-increasing OR loss-decreasing heuristic, then re-upload.
      // The scheduler's staging buffers are sized to the live atomics
      // buffer; if a future resize changes that size, `_ensurePPGRefinementStaging`
      // re-allocates them.
      this._ppgRefinement = new STreeRefinementScheduler();
      this._ensurePPGRefinementStaging();
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
   * Resize all per-frame GPU resources to a new render-surface size WITHOUT
   * rebuilding the BVH or recompiling pipelines. Destroys the current
   * `_res: FrameResources` (every full-res rgba16float texture, reservoir
   * buffer, variance buffer, GTAO half/full, SVGF persistent textures, …)
   * and reallocates them at the new dimensions. Resets ping-pong indices
   * and frame counters because the new textures contain garbage.
   *
   * Cost: O(W·H) GPU memory churn (~1 GB at 4K), no shader recompilation,
   * no BVH rebuild. Call this from the host (via `HybridEngine.setSize`)
   * when the canvas resizes — much cheaper than full engine teardown +
   * re-init.
   *
   * Bind groups are NOT cached at the pipeline level (they're built per
   * frame in `renderFrame()` from the live texture handles), so the
   * resize automatically picks up next frame.
   *
   * No-op when called before `initialize()`.
   */
  resize(width: number, height: number): void {
    if (!this._initialized) {
      // Update stored size so a later initialize() picks up the new value.
      this._width = width;
      this._height = height;
      return;
    }
    if (width === this._width && height === this._height) return;
    this._width = width;
    this._height = height;
    // Destroy + reallocate per-frame resources at the new size.
    destroyFrameResources(this._res);
    this._res = createFrameResources(this._device, width, height);
    // W9 — re-allocate PPG resolution-dependent buffers + re-upload the
    // (unchanged) sTree topology so the new bind groups have valid GPU
    // buffers to bind. The CPU sTree itself isn't size-dependent and
    // survives the resize unchanged.
    if (this._ppgEnabled) {
      allocatePPGResources(this._device, this._res, width, height);
      this._uploadPPGTree();
      this._writePPGGuideUBO();
      this._writePPGUpdateUBO();
      // Resize may have changed the atomics buffer size — re-size the
      // refinement staging buffers + reset gating history (new resolution
      // is effectively a fresh sample stream).
      this._ensurePPGRefinementStaging();
      this._ppgRefinement?.resetHistory();
    }
    // Reset transient per-frame state — ping-pong reads from the previous
    // frame's texture, but the new textures are blank, so we must restart
    // the accumulator at α=1 and re-seed history.
    this._accumPingPongIndex = 0;
    this._accumFrameIndex = 0;
    this._indirectAccumPingPongRef.value = 0;
    this._lastCameraPos = [0, 0, 0];
    // Denoiser-private ping-pong indices (Welford / SVGF) reset inside
    // each Denoiser.resize implementation.
    this._activeDenoiser?.resize(width, height);
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
    const finalTex = this._res.common.resolvedTexture;
    const bgComposite = buildCompositeBindGroup(
      d, this._bglCache, finalTex.createView(), this._res.common.compositeSampler,
    );
    // The CompositePass instance owns the compiled render pipeline; reuse
    // it here so a single source of truth for the composite shader stays
    // intact. Located via registry lookup — the orchestrator does not
    // hold a separate compiled-pipeline handle for composite.
    const compositePass = this._passRegistry!.get('composite') as CompositePass;
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
    pass.setPipeline(compositePass.pipeline);
    pass.setBindGroup(0, bgComposite);
    pass.draw(3, 1, 0, 0);
    pass.end();
    d.queue.submit([encoder.finish()]);
  }

  /**
   * Schedule a temporal-accumulator reset on the next rendered frame.
   *
   * Sets `_accumFrameIndex` to 0 so the accumulation shader selects α=1
   * (history discarded) on the very next `renderFrame` call. After that
   * single frame the index increments normally and the accumulator resumes
   * blending at the configured `_temporalAccumAlpha`.
   *
   * Cost: one JS field write. No GPU work is dispatched. Called by
   * `HybridEngine.updateLighting()` whenever lighting parameters change.
   */
  requestAccumReset(): void {
    this._accumFrameIndex = 0;
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
    updateUBO(d, this._res.common.uboBuffer, inputs);

    // W9 — refresh the PPG guide UBO so the kernel's per-frame RNG salt
    // (and any future per-frame inputs) stay current. The update UBO is
    // static-per-resolution and need not be re-uploaded each frame.
    if (this._ppgEnabled) {
      this._writePPGGuideUBO();
    }

    // ── Build placeholder texture view ────────────────────────────────────
    const placeholderView = this._res.common.placeholderTexture.createView();

    // ── Build shared bind groups (frame/scene/ubo/hybrid-layers) ─────────
    const bgFrame = buildFrameBindGroup(d, this._bglCache, {
      placeholderView,
      reservoirCurrentBuffer:  this._res.restirDI.reservoirCurrentBuffer,
      reservoirPreviousBuffer: this._res.restirDI.reservoirPreviousBuffer,
      reservoirSpatialBuffer:  this._res.restirDI.reservoirSpatialBuffer,
      hdrColorTexture:         this._res.common.hdrColorTexture,
      nearestSampler:          this._res.common.nearestSampler,
      gNormalDepthTexture:     this._res.common.gNormalDepthTexture,
      reservoirGiCurrentBuffer: this._res.restirGI.reservoirGiCurrentBuffer,
      hdrIndirectTexture:      this._res.common.hdrIndirectTexture,
      hdrTotalTexture:         this._res.common.hdrTotalTexture,
      // Item 24 — albedo demodulation (Schied 2017 §4.1).
      albedoTexture:           this._res.common.albedoTexture,
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
      d, this._bglCache, this._res.common.uboBuffer,
      this._res.gtao.aoFullTexture.createView(),
      this._res.common.tierTexture.createView(),
    );
    // Sprint 16 — DDGI hybrid layers slot 3 — shared by gi-ris and shade.
    // W9 Phase 2 — also carries the ppgGuidance buffer at binding 4. shade
    // consumes it for the indirect-path MIS combine; risGi shares the BGL
    // but does not declare binding 4 (WGSL permits unused BGL entries on
    // a given consumer shader).
    //
    // W9 Phase 2 wire — when PPG is enabled, point slot 4 at the real
    // kernel-output buffer (`ppg.sampleOutBuf`, written by the ppg-guide
    // pass earlier in this same encoder) instead of the zero-filled
    // placeholder. shade.wgsl's PDF-sentinel fallback (pdf <= 0) still
    // applies on pixels the guide kernel chose not to emit, so the
    // ReSTIR-GI-only branch remains intact for those pixels. When PPG is
    // disabled we keep the placeholder so the bind-group LAYOUT (and
    // therefore the compiled shade pipeline) is identical in both modes.
    //
    // Shape contract: both buffers carry `array<vec4<f32>>` of length
    // (W × H), `xyz`=world-space direction, `w`=solid-angle pdf — the
    // exact contract declared in `ppgGuide.wgsl.ts`'s `ppgSampleOut`
    // binding and `shade.wgsl.ts`'s `ppgGuidance` consumer binding.
    const shadePPGBuffer: GPUBuffer =
      this._ppgEnabled && this._res.ppg.sampleOutBuf !== undefined
        ? this._res.ppg.sampleOutBuf
        : this._res.ppg.ppgGuidanceBuffer;
    const bgHybrid = buildHybridLayersBindGroup(d, this._bglCache, {
      ddgiIrrTex:              this._ddgiIrrTex,
      ddgiVisTex:              this._ddgiVisTex,
      ddgiPlaceholderRgba16f:  this._res.ddgi.ddgiPlaceholderRgba16f,
      ddgiPlaceholderRg16f:    this._res.ddgi.ddgiPlaceholderRg16f,
      nearestSampler:          this._res.common.nearestSampler,
      ddgiUboBuffer:           this._res.ddgi.ddgiUboBuffer,
      ppgGuidanceBuffer:       shadePPGBuffer,
    });

    // ── Per-frame pre-computed scalars ───────────────────────────────────
    const passLayout = buildPassLayout({ denoiserMode: this._denoiserMode });

    const encoder = d.createCommandEncoder({ label: 'walkaround-restir' });

    const wgX  = Math.ceil(W / 8);
    const wgY  = Math.ceil(H / 8);
    const wgX16 = Math.ceil(W / 16);
    const wgY16 = Math.ceil(H / 16);
    const halfWgX = Math.ceil(Math.floor(W / 2) / 8);
    const halfWgY = Math.ceil(Math.floor(H / 2) / 8);

    // Helper: build a GPUComputePassDescriptor without an undefined timestampWrites
    // property — required by exactOptionalPropertyTypes. We spread the optional
    // timestampWrites field only when it has a value. The label is the pass's
    // PassLabel; slot index is resolved through passLayout so it stays in sync
    // with the GPU timing readback labels.
    const computeDesc = (label: PassLabel): GPUComputePassDescriptor => {
      const ts = tsWrites(this._tsState.querySet, passLayout, label);
      return ts ? { label, timestampWrites: ts } : { label };
    };
    const renderTimestampWrites = (label: PassLabel): GPURenderPassTimestampWrites | undefined => {
      return tsWrites(this._tsState.querySet, passLayout, label);
    };

    // ── Camera motion: reset temporal index before denoise / accum ────────
    const dx = inputs.cameraPos[0] - this._lastCameraPos[0];
    const dy = inputs.cameraPos[1] - this._lastCameraPos[1];
    const dz = inputs.cameraPos[2] - this._lastCameraPos[2];
    const camMoveSq = dx * dx + dy * dy + dz * dz;
    const isMoving = camMoveSq > this._cameraMoveResetThresholdSq;
    if (isMoving) {
      this._accumFrameIndex = 0;
    }

    // Resolve the temporal-accumulator ping-pong slots for this frame.
    const readAccum  = this._accumPingPongIndex === 0
      ? this._res.common.accumTextureA : this._res.common.accumTextureB;
    const writeAccum = this._accumPingPongIndex === 0
      ? this._res.common.accumTextureB : this._res.common.accumTextureA;

    const gNormalDepthView = this._res.common.gNormalDepthTexture.createView();

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

    // ── Build the shared per-pass dispatch context ───────────────────────
    const frameState: PassFrameState = {
      denoisedDirect: this._res.common.hdrColorTexture,   // overwritten by denoiser dispatch
      indirectAccumOut: this._res.common.indirectAccumPingTexture,  // overwritten by indirect-temporal-accum
      denoisedIndirect: this._res.common.indirectDenoisedPingTexture, // overwritten by atrous-indirect
      combinedDenoised: this._res.common.combinedDenoisedTexture,
      writeAccum,
      readAccum,
      alpha,
      isMoving,
    };
    const passCtx: PassDispatchContext = {
      device: d,
      encoder,
      width: W,
      height: H,
      frameIndex: this._accumFrameIndex,
      frameCount: this._frameCount,
      bglCache: this._bglCache,
      resources: this._res,
      inputs,
      frameBindGroup: bgFrame,
      sceneBindGroup: bgScene,
      uboBindGroup: bgUbo,
      hybridLayersBindGroup: bgHybrid,
      wgX, wgY, wgX16, wgY16, halfWgX, halfWgY,
      gNormalDepthView,
      computeDesc,
      renderTimestampWrites,
      frameState,
    };

    const gateOpts: PassGateOptions = {
      denoiserMode: this._denoiserMode,
      ppgEnabled: this._ppgEnabled,
    };

    // ── Pass loop, part 1 — up to and including gtao-upsample ────────────
    // Manual `Pass.gates` filtering inline so we can iterate the cached
    // `_sortedPasses` array; equivalent to `_passRegistry.activePasses(...)`
    // but avoids re-sorting per frame.
    for (let i = 0; i <= this._denoiserSplitIndex; i++) {
      const pass = this._sortedPasses[i]!;
      if (!pass.gates(gateOpts)) continue;
      pass.dispatch(passCtx);
    }

    // ── Polymorphic denoiser dispatch ────────────────────────────────────
    // Honest layering: denoising is a separate concept from pass
    // scheduling (denoisers have a return value — the resolved-radiance
    // texture downstream composition samples — and a different lifecycle
    // shape). Wrapping the denoiser as a virtual Pass would obscure that
    // distinction. The orchestrator threads the result back into
    // `frameState.denoisedDirect` for IndirectCombinePass to read.
    const denoiserResult = this._activeDenoiser!.dispatch({
      device: d,
      encoder,
      width: W,
      height: H,
      frameIndex: this._accumFrameIndex,
      resources: this._res,
      sharedAtrousPipeline: this._atrousPipeline,
      bglCache: this._bglCache,
      gNormalDepthView,
      readAccum,
      isMoving,
      wgX16,
      wgY16,
      computeDesc,
    });
    // NoneDenoiser returns null → source the raw HDR target directly.
    frameState.denoisedDirect = denoiserResult ?? this._res.common.hdrColorTexture;

    // ── Pass loop, part 2 — indirect-temporal-accum … composite ──────────
    for (let i = this._denoiserSplitIndex + 1; i < this._sortedPasses.length; i++) {
      const pass = this._sortedPasses[i]!;
      if (!pass.gates(gateOpts)) continue;
      pass.dispatch(passCtx);
    }

    // ── End-of-frame: swap-chain present sentinel + reservoir housekeeping ─
    this._accumPingPongIndex = 1 - this._accumPingPongIndex;
    this._accumFrameIndex++;
    this._lastCameraPos = [...inputs.cameraPos];

    // Swap reservoir ping-pong for next frame (copy current → previous).
    // Sprint 17 + audit B6 fix: copies must be folded into the *same*
    // command encoder as the main frame work, before its single
    // queue.submit().  When this was a separate submit (enc2 below the
    // main submit), high-FPS hosts could begin frame N+1's temporal
    // reservoir read before enc2 had completed, racing the previous-
    // frame copy — corrupts the GI reservoir, manifests as flicker.
    //
    // Kept as inline `encoder.copyBufferToBuffer` rather than a `FinalizePass`
    // because the housekeeping touches non-Pass state (the orchestrator-
    // owned reservoir ping-pong) and runs AFTER `composite` regardless of
    // gating; an extra Pass would have empty `passLabels` and force the
    // orchestrator to know about it specially. Two lines here vs a 30-line
    // file is the right trade.
    encoder.copyBufferToBuffer(
      this._res.restirDI.reservoirCurrentBuffer, 0,
      this._res.restirDI.reservoirPreviousBuffer, 0,
      this._res.restirDI.reservoirCurrentBuffer.size,
    );
    encoder.copyBufferToBuffer(
      this._res.restirGI.reservoirGiCurrentBuffer, 0,
      this._res.restirGI.reservoirGiPreviousBuffer, 0,
      this._res.restirGI.reservoirGiCurrentBuffer.size,
    );

    // Resolve timestamps + copy into the inactive readback buffer.
    resolveTimestamps(encoder, this._tsState, this._frameCount, passLayout.slotCount);

    // W9 — schedule a copyBufferToBuffer from the live ppgFluxAtomics to
    // our refinement staging buffer iff this frame matches the cadence.
    // The copy must be encoded BEFORE queue.submit; the mapAsync that
    // backs the readback happens AFTER submit (next block).
    const refinementStaging = this._maybeEncodePPGAtomicsCopy(encoder);

    d.queue.submit([encoder.finish()]);

    // Per-frame denoiser cleanup. Runs after `queue.submit()` — the GPU
    // queue holds its own reference to the encoded command buffer, so
    // it is safe to release host-side handles to anything the denoiser
    // wrote into this frame's encoder. SVGFRealDenoiser uses this hook
    // to destroy the 5 transient per-iter UBOs it allocates each frame;
    // other denoisers implement it as a no-op (or omit it entirely).
    this._activeDenoiser?.cleanupAfterSubmit?.();

    // Kick async readback of the timestamp buffer we just copied into.
    // Pass the layout labels so the async callback labels each slot
    // correctly even if the pipeline reconfigures between frames.
    kickTimestampReadback(this._tsState, this._frameCount, passLayout.labels);

    // W9 — if we encoded a flux-atomics copy above, kick its mapAsync
    // here (post-submit). The async resolution runs `_consumePPGRefinementReadback`
    // which decodes the atomics, applies the heuristic gate, calls
    // splitOverflowLeaves on the CPU mirror, and (on accept) re-uploads
    // the serialised tree + clears the atomics on the GPU.
    if (refinementStaging !== null) {
      this._kickPPGRefinementReadback(refinementStaging);
    }
    // Mirror public telemetry fields from the state object so callers
    // can read them as before.
    this.lastGpuTimings      = this._tsState.lastGpuTimings;
    this.lastGpuTimingsFrame = this._tsState.lastGpuTimingsFrame;

    this._frameCount++;
    return true;
  }

  /**
   * W9 — Serialise the CPU sTree + per-cell dTrees and upload to the GPU
   * storage buffers. Called once at init; the refinement scheduler calls
   * this again after each successful `splitOverflowLeaves` cycle. No-op
   * when PPG is disabled.
   *
   * Caches the returned `dTreeOffsets` on the pipeline so the refinement
   * scheduler can decode atomic readbacks without re-serialising.
   */
  private _uploadPPGTree(): void {
    if (!this._ppgEnabled || !this._ppgSTree) return;
    const ppg = this._res.ppg;
    if (!ppg.sTreeBuf || !ppg.dTreeBuf || !ppg.dTreeOffsetsBuf) return;
    const { sTreeBuf, dTreeBuf, dTreeOffsets } = serialiseSTree(this._ppgSTree);
    this._device.queue.writeBuffer(ppg.sTreeBuf, 0, sTreeBuf.buffer, sTreeBuf.byteOffset, sTreeBuf.byteLength);
    this._device.queue.writeBuffer(ppg.dTreeBuf, 0, dTreeBuf.buffer, dTreeBuf.byteOffset, dTreeBuf.byteLength);
    this._device.queue.writeBuffer(ppg.dTreeOffsetsBuf, 0, dTreeOffsets.buffer, dTreeOffsets.byteOffset, dTreeOffsets.byteLength);
    this._ppgDTreeOffsets = dTreeOffsets;
  }

  /**
   * W9 — Pack and upload the guide-kernel UBO. Layout matches `PPGGuideUBO`
   * in `ppgGuide.wgsl.ts` (12 × f32 = 48 bytes):
   *   [0]    pixelCount      (u32)
   *   [1]    imgWidth        (u32)
   *   [2]    alpha           (f32, MIS mixing weight)
   *   [3]    frameSeed       (u32)
   *   [4..6] sceneMin xyz    (f32)
   *   [7..9] sceneMax xyz    (f32)
   *   [10..11] padding
   */
  private _writePPGGuideUBO(): void {
    if (!this._ppgEnabled) return;
    const buf = this._res.ppg.guideUboBuffer;
    if (!buf) return;
    const data = new ArrayBuffer(48);
    const u32 = new Uint32Array(data);
    const f32 = new Float32Array(data);
    const pixelCount = this._width * this._height;
    u32[0] = pixelCount;
    u32[1] = this._width;
    f32[2] = PPG_MIS_ALPHA;
    u32[3] = this._frameCount >>> 0;
    f32[4] = this._ppgSceneAABB.min[0];
    f32[5] = this._ppgSceneAABB.min[1];
    f32[6] = this._ppgSceneAABB.min[2];
    f32[7] = this._ppgSceneAABB.max[0];
    f32[8] = this._ppgSceneAABB.max[1];
    f32[9] = this._ppgSceneAABB.max[2];
    u32[10] = 0;
    u32[11] = 0;
    this._device.queue.writeBuffer(buf, 0, data);
  }

  /**
   * W9 — Pack and upload the update-kernel UBO. Layout (16 bytes):
   *   [0] sampleCount  (u32)
   *   [1] fluxBudget   (u32) — total atomic slots
   *   [2..3] padding
   */
  private _writePPGUpdateUBO(): void {
    if (!this._ppgEnabled) return;
    const buf = this._res.ppg.updateUboBuffer;
    if (!buf) return;
    const fluxAtomics = this._res.ppg.fluxAtomicsBuf;
    const fluxBudget = fluxAtomics ? Math.floor(fluxAtomics.size / 4) : 0;
    const data = new ArrayBuffer(16);
    const u32 = new Uint32Array(data);
    u32[0] = this._width * this._height;
    u32[1] = fluxBudget;
    u32[2] = 0;
    u32[3] = 0;
    this._device.queue.writeBuffer(buf, 0, data);
  }

  /**
   * Public W9 metric — total sTree refinement cycles run since pipeline
   * init (i.e. how many times `splitOverflowLeaves` mutated the CPU
   * mirror sTree). Surfaced by `HybridEngine.debug.ppgRefinementCount()`.
   * Returns 0 when PPG is disabled.
   */
  getPPGRefinementCount(): number {
    return this._ppgRefinement?.refinementCount ?? 0;
  }

  /** Ensure the refinement scheduler's staging buffers are sized to the
   *  live `fluxAtomicsBuf`. No-op when PPG is disabled. */
  private _ensurePPGRefinementStaging(): void {
    if (!this._ppgEnabled || !this._ppgRefinement) return;
    const fluxBuf = this._res.ppg.fluxAtomicsBuf;
    if (!fluxBuf) return;
    this._ppgRefinement.ensureStaging(this._device, fluxBuf.size);
  }

  /**
   * If this frame matches the refinement cadence, encode a
   * `copyBufferToBuffer` from the live `fluxAtomicsBuf` into one of the
   * scheduler's ping-pong staging buffers and return that buffer. The
   * caller is expected to kick `mapAsync` on it AFTER `queue.submit`.
   * Returns null when no copy was encoded (cadence miss, PPG disabled,
   * or a readback is already in flight).
   */
  private _maybeEncodePPGAtomicsCopy(encoder: GPUCommandEncoder): GPUBuffer | null {
    if (!this._ppgEnabled || !this._ppgRefinement) return null;
    const fluxBuf = this._res.ppg.fluxAtomicsBuf;
    if (!fluxBuf) return null;
    if (!this._ppgRefinement.shouldReadback(this._frameCount)) return null;

    const staging = this._ppgRefinement.acquireStaging();
    if (staging === null) return null;
    encoder.copyBufferToBuffer(fluxBuf, 0, staging, 0, fluxBuf.size);
    return staging;
  }

  /**
   * Kick a `mapAsync(READ)` on the given staging buffer and, once it
   * resolves, run the heuristic gate + (on accept) `splitOverflowLeaves`
   * + re-upload the serialised tree + clear the GPU atomics.
   *
   * Fire-and-forget — we deliberately do NOT await this; if the
   * resolution lands during a later frame's `renderFrame`, the
   * scheduler's `_readbackInFlight` flag prevents a second concurrent
   * readback from racing. The async callback never touches GPU state
   * during another encoder's recording (only via post-completion
   * `device.queue.writeBuffer` calls, which are queue-serialised).
   */
  private _kickPPGRefinementReadback(staging: GPUBuffer): void {
    if (!this._ppgRefinement || !this._ppgSTree || !this._ppgDTreeOffsets) return;
    const sched = this._ppgRefinement;
    const sTree = this._ppgSTree;
    const dTreeOffsets = this._ppgDTreeOffsets;
    const device = this._device;
    const fluxBuf = this._res.ppg.fluxAtomicsBuf;

    staging.mapAsync(GPUMapMode.READ).then(() => {
      try {
        const range = staging.getMappedRange();
        // Copy out of the mapped range before unmapping — decoding
        // touches the buffer in-place but the snapshot stores derived
        // scalars only, so we can unmap immediately after.
        const snap = sched.consumeReadback(sTree, dTreeOffsets, range);
        staging.unmap();
        if (snap === null) return; // gate rejected — nothing to do
        // Apply the split + re-upload + atomic clear. The CPU sTree's
        // sampleCount fields were just populated by consumeReadback;
        // splitOverflowLeaves reads them directly.
        const grew = sched.applySplit(sTree, PPG_CELL_SPLIT_THRESHOLD, PPG_MAX_SPATIAL_CELLS);
        if (grew) {
          // Re-upload + cache new offsets via the existing helper.
          this._uploadPPGTree();
        }
        // Always clear the atomics after a readback (whether or not we
        // split). The next training cycle must start from zero, otherwise
        // the gating heuristic would see ever-increasing totals and split
        // forever. `writeBuffer` with a zero-filled view is the cheapest
        // option here — the buffer is at most a few MB.
        if (fluxBuf) {
          const zero = new Uint8Array(fluxBuf.size);
          device.queue.writeBuffer(fluxBuf, 0, zero);
        }
      } catch (e) {
        // Robustness: a mapAsync error (lost device, validation failure)
        // must NOT crash the render loop. Surface to console and reset
        // the scheduler's in-flight flag so the next cadence tick can
        // retry. `consumeReadback` already clears the flag on success.
        try { staging.unmap(); } catch { /* may not be mapped */ }
        sched.resetHistory();
        // eslint-disable-next-line no-console
        console.warn('[PPG] refinement readback failed:', e);
      }
    }).catch((e) => {
      // mapAsync rejection — same recovery as above. WebGPU spec: the
      // promise rejects on device-loss or buffer-destroy races.
      sched.resetHistory();
      // eslint-disable-next-line no-console
      console.warn('[PPG] refinement mapAsync rejected:', e);
    });
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
    // Denoiser owns its own pipelines + UBOs; let it release them.
    this._activeDenoiser?.dispose();
    this._activeDenoiser = null;
    this._denoiserRegistry = null;
    // Each Pass releases any pass-private GPU resources it owns.
    for (const pass of this._sortedPasses) pass.dispose();
    this._sortedPasses = [];
    this._passRegistry = null;
    // T2.H2 — dispose the neural InferenceGraph if present (reserved for W10).
    this._inferenceGraph?.dispose();
    this._inferenceGraph = null;
    // W9 — release refinement-scheduler staging buffers.
    this._ppgRefinement?.dispose();
    this._ppgRefinement = null;
    this._ppgDTreeOffsets = null;
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
      this._device.queue.writeBuffer(this._res.ddgi.ddgiUboBuffer, 0, this._ddgiPlaceholderUBO.buffer);
    } else {
      this._ddgiIrrTex = inputs.irradianceTex;
      this._ddgiVisTex = inputs.visibilityTex;
      if (inputs.gridParams.byteLength > 0) {
        this._device.queue.writeBuffer(this._res.ddgi.ddgiUboBuffer, 0, inputs.gridParams);
      }
    }
  }
}
