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
 * **W1-R5 — declarative pass order.** All non-denoiser stages (registered below) are
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

import { deriveSceneAABBFromBvhPositions } from '@vitrum/shared-bvh';
import type { EngineError, EngineWarning, Scene } from '@vitrum/core';
import type { SceneBVHBuffers } from '../restir/bvhTypes.js';
import type { MaterialTextureAtlasPayload } from './materialTextureAtlas.js';
import type { BvhUpdateSink } from './BvhUpdateSink.js';
import type { PipelineDebugTextures } from './PipelineDebugTextures.js';
import type { InferenceGraph } from '../neural/InferenceGraph.js';
import type { ModelWeights } from '../neural/weights.js';
import { updateUBO } from './uboUpdater.js';
import { compilePipelines } from './pipelineCompiler.js';
import type { CollectedBvhMutation } from './CollectingBvhUpdateSink.js';
import { BvhBufferHost } from './BvhBufferHost.js';
import {
  prepareSceneMutations,
  rethrowWithSceneMutationCleanup,
  runSceneMutationCleanups,
  type PreparedSceneMutation,
  type SceneMutationCleanup,
} from '../SceneMutationTransaction.js';
import {
  estimateFrameResourcesMemory,
  type GpuMemoryExternalSections,
} from './gpuMemoryEstimate.js';
import type { GpuMemoryBreakdown } from '@vitrum/core';
import {
  createFrameResources,
  destroyFrameResources,
  type FrameResources,
} from './resourceManager.js';
import type { BGLCache } from './bindGroupLayouts.js';
import type { UboRef, PassOwnedUboRef } from './bindGroupBuilders.js';
import { buildLightTreeBindGroup } from './bindGroupBuilders.js';
import { FrameCaptureHelper } from './frameCapture.js';
import {
  buildCompositePresentBindGroup,
  buildPerFrameBindGroups,
} from './pipelineBindGroupFactory.js';
import { PipelineResourceCache } from './PipelineResourceCache.js';
import { PPGCoordinator } from './PPGCoordinator.js';
import {
  DEFAULT_NRC_CONFIG,
  NrcSubsystem,
  resolveNrcConfig,
  type NrcConfig,
} from '../neural/nrc/nrcSubsystem.js';
import { computeNrcResourceFootprint, preflightNrcResources } from '../neural/nrc/nrcPreflight.js';
import type { NrcDiagnostics } from '../neural/nrc/nrcDiagnostics.js';
import { fusedMlpWorkgroupStorageBytes } from '../neural/nrc/wgsl/fusedMlp.wgsl.js';
import { ReGIRCoordinator, resolveReGIRConfig, type ReGIRConfig } from './ReGIRCoordinator.js';
import { OptionalSubsystemBindingState } from './OptionalSubsystemBindingState.js';
import { DenoiserRegistry, type Denoiser, type DenoiserId } from './denoisers/index.js';
import { registerBuiltinDenoisers } from './denoisers/registerBuiltinDenoisers.js';
import { PassRegistry } from './PassRegistry.js';
import type { Pass, PassDispatchContext, PassFrameState, PassGateOptions } from './Pass.js';
import {
  AtrousIndirectPass,
  CheckerboardPrefillPass,
  CompositePass,
  DenoiserAdapterPass,
  GTAOPass,
  GTAOUpsamplePass,
  IndirectCombinePass,
  IndirectTemporalAccumPass,
  MotionVectorsPass,
  PPGUpdatePass,
  ReGIRBuildPass,
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
  TransparentOitPass,
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
} from './timestampQueries.js';
import { reservoirGiStrideU32ForGrisReuse } from '../gi/giLayout.js';
import {
  isValidRestirGISnapshot,
  type RestirGISnapshot,
} from '../giStateSnapshot.js';

/**
 * Prepared ReSTIR-GI restore. Candidate buffers are fully populated before
 * this handle is returned; commit only publishes their identities, while abort
 * retires them without touching the live reservoir cohort.
 */
export interface RestirGIReservoirImportTransaction {
  commit(): void;
  abort(): void;
}

// D3.5 — PipelineFrame* interfaces extracted to pipelineFrameInputs.ts.
// Re-exported here for back-compat (test harnesses import PipelineFrameInputs
// and PipelineFrameFilter from this path).
export type { PipelineFrameFilter, PipelineFrameInputs } from './pipelineFrameInputs.js';
import type { PipelineFrameInputs } from './pipelineFrameInputs.js';
import {
  FramePublicationTransaction,
  finishSubmitAndPublishFrame,
  type FramePublication,
} from './FramePublication.js';

export function resolvePpgDispatchInterval(interval: number): number {
  if (!Number.isFinite(interval)
      || interval <= 0
      || interval > Number.MAX_SAFE_INTEGER) {
    throw new RangeError(
      `PPG dispatch interval must be finite and positive; got ${interval}`,
    );
  }
  return Math.max(1, Math.floor(interval));
}

/**
 * Dependencies the {@link registerPasses} free function needs to construct +
 * register the non-denoiser Pass set. Bundled so the orchestrator's
 * `initialize()` hands the registration step its inputs explicitly instead of
 * inlining ~75 LOC of `new XxxPass(...)` calls. Thunks (`getActiveDenoiser`
 * etc.) preserve the late-`this`-binding the original closures had — they are
 * read at dispatch time, never captured eagerly.
 */
interface RegisterPassesDeps {
  diSpatialPasses: 1 | 2;
  giSpatialPasses: 1 | 2;
  grisReuseStructural: boolean;
  /** Checkerboard half-res shading flag (host opt-in). Threaded into the
   *  ResolvePass + CheckerboardPrefillPass ctors; OFF ⇒ passthrough
   *  (byte-identity). */
  checkerboard: boolean;
  cbPrefillUboRef: UboRef;
  sampleBudgetUboRef: UboRef;
  sampleCountUboRef: UboRef;
  accumUboRef: UboRef;
  resolveUboRef: UboRef;
  /** D4.3 — pass-owned: excluded from `_perPassUboRefs` disposal. */
  atrousIndirectUboRef: PassOwnedUboRef;
  compositeUboRef: UboRef;
  indirectAccumPingPongRef: PingPongRef;
  regir: ReGIRCoordinator;
  bglCache: BGLCache;
  bvhBuffers: SceneBVHBuffers;
  /** Per-frame NRC slot-claim clear encoder, or `undefined` when NRC is off. */
  nrcClearSlotClaims: ((encoder: GPUCommandEncoder) => void) | undefined;
  getActiveDenoiser: () => Denoiser;
  getAtrousPipeline: () => GPUComputePipeline;
  isDenoiserPassEnabled: () => boolean;
  getRegirResources: () => {
    combinedLightTreeBuffer: GPUBuffer;
    uboBuffer: GPUBuffer;
  };
}

/**
 * Instantiate + register every non-denoiser {@link Pass} into a fresh
 * {@link PassRegistry}, in the canonical source order. Extracted verbatim from
 * `WalkaroundGPUPipeline.initialize()` (Task 4.1 god-orchestrator reduction) —
 * the registered set, registration order, and the resulting topologically-
 * sorted PASS_ORDER are byte-identical to the inlined block. The
 * `regir.initialize(...)` side-effect is preserved at its original position
 * (before the conditional ReGIRBuildPass registration). Returns the populated
 * registry plus the CompositePass instance the caller caches for
 * `presentLastFrame`.
 */
function registerPasses(
  compiled: Awaited<ReturnType<typeof compilePipelines>>,
  deps: RegisterPassesDeps,
): { registry: PassRegistry; compositePass: CompositePass } {
  const registry = new PassRegistry();
  registry.register(
    new SampleBudgetPass(
      compiled.sampleBudgetPipeline,
      deps.sampleBudgetUboRef,
      deps.sampleCountUboRef,
    ),
  );
  registry.register(new RISPass(compiled.risPipeline));
  registry.register(new TemporalReservoirPass(compiled.temporalPipeline));
  // Phase-0 — spatial pass count is preset-driven (1 or 2 ping-pong passes).
  registry.register(new SpatialReservoirPass(compiled.spatialPipeline, deps.diSpatialPasses));
  // NRC ON ⇒ supply the @group(4) bind group getter so the gi-ris pass binds
  // slot 4 (the inline-MLP variant was compiled). OFF ⇒ no getter, verbatim
  // 4-group dispatch (the default-path structure is unchanged).
  registry.register(
    new RISGIPass(compiled.risGiPipeline, deps.nrcClearSlotClaims),
  );
  // Train from the initial RIS reservoir. Temporal reuse depends on this pass
  // when PPG is compiled so the trainer sees the proposal with known q.
  if (compiled.ppgUpdatePipeline) {
    registry.register(new PPGUpdatePass(compiled.ppgUpdatePipeline));
  }
  registry.register(
    new TemporalGIReservoirPass(
      compiled.temporalGiPipeline,
      deps.grisReuseStructural,
      compiled.ppgUpdatePipeline !== undefined,
    ),
  );
  registry.register(
    new SpatialGIReservoirPass(
      compiled.spatialGiPipeline,
      deps.giSpatialPasses,
      deps.grisReuseStructural,
    ),
  );
  registry.register(new ShadePass(compiled.shadePipeline));
  registry.register(new MotionVectorsPass(compiled.motionVectorsPipeline));
  registry.register(new GTAOPass(compiled.gtaoPipeline));
  registry.register(new GTAOUpsamplePass(compiled.gtaoUpsamplePipeline));
  // Checkerboard pre-denoiser gap-fill — fills hdrColorTexture gap pixels
  // before the denoiser-adapter reads it.  Gated: only runs when
  // checkerboardOn AND the active denoiser is one of the four real denoisers.
  // Byte-identical to today when checkerboard is off or a default denoiser
  // is active.
  registry.register(
    new CheckerboardPrefillPass(
      compiled.cbPrefillPipeline,
      deps.cbPrefillUboRef,
      deps.checkerboard,
    ),
  );
  // Virtual pass — promotes the polymorphic denoiser dispatch into the
  // regular pass loop. Reads the active Denoiser through a getter so
  // the adapter stays valid across `_activeDenoiser` reassignment in
  // `dispose()` (where it is set to null AFTER the pass-list dispose
  // walk, so the getter never sees the null transition).
  registry.register(
    new DenoiserAdapterPass(
      deps.getActiveDenoiser,
      deps.getAtrousPipeline,
      deps.isDenoiserPassEnabled,
    ),
  );
  registry.register(
    new IndirectTemporalAccumPass(
      compiled.indirectTemporalAccumPipeline,
      deps.indirectAccumPingPongRef,
    ),
  );
  registry.register(new AtrousIndirectPass(compiled.atrousPipeline, deps.atrousIndirectUboRef));
  registry.register(new IndirectCombinePass(compiled.indirectCombinePipeline));
  registry.register(new TransparentOitPass(compiled.transparentOitPipeline));
  registry.register(new TemporalAccumPass(compiled.accumPipeline, deps.accumUboRef));
  registry.register(
    new ResolvePass(compiled.resolvePipeline, deps.resolveUboRef, deps.checkerboard),
  );
  const compositePass = new CompositePass(compiled.compositePipeline, deps.compositeUboRef);
  registry.register(compositePass);
  // ReGIR grid-build (Boksansky 2021) — register only when the pipeline
  // compiled (opt-in). Topo-sort runs it FIRST (no deps; `regir-build` <
  // `sample-budget` lexically) so the grid is filled before RIS reads it.
  // `gates()` further requires the coordinator be live (light tree live).
  // Initialise the coordinator's grid geometry first so `live` is correct.
  deps.regir.initialize(deps.bvhBuffers, compiled.regirBuildPipeline !== undefined);
  if (compiled.regirBuildPipeline) {
    registry.register(
      new ReGIRBuildPass(
        compiled.regirBuildPipeline,
        deps.regir,
        deps.bglCache,
        deps.getRegirResources,
      ),
    );
  }
  return { registry, compositePass };
}

/**
 * Device limits required by the actual explicit pipeline layouts created in
 * pipelineCompiler.ts. These are hard minima, not headroom:
 *
 * - 8 storage buffers: RIS/shade/GI-RIS (packed scene/PPG/NRC arenas)
 * - 7 storage textures: transparent-OIT (frame's six + OIT output)
 * - 16 sampled textures: NRC GI-RIS (frame + scene + UBO + hybrid layers)
 *
 * The companion layout-derivation test records the real bind-group descriptors
 * and recomputes all three peaks. If a layout changes, the test fails until this
 * public request-device contract changes with it.
 */
export const HYBRID_WEBGPU_REQUIRED_LIMITS: Readonly<Record<string, number>> = Object.freeze({
  maxStorageBuffersPerShaderStage: 8,
  maxStorageTexturesPerShaderStage: 7,
  maxSampledTexturesPerShaderStage: 16,
});

/**
 * Lite currently compiles the same explicit bind-group layouts as full. Runtime
 * gates and merged BVH reduce memory/work, but WebGPU validates every entry in a
 * pipeline layout, including bindings unused by a selected shader path. Lite
 * therefore has the same structural device floor until it gains a real layout
 * fork. Advertising 10/6 made compliant hosts request a device that could not
 * create scene-bgl (11 storage buffers) or transparentOitLayout (7 storage
 * textures).
 */
export const HYBRID_LITE_LIMITS: Readonly<Record<string, number>> = HYBRID_WEBGPU_REQUIRED_LIMITS;

const WEBGPU_DEFAULT_LIMITS = {
  maxStorageBuffersPerShaderStage: 8,
  maxStorageTexturesPerShaderStage: 4,
  maxSampledTexturesPerShaderStage: 16,
  maxBindGroups: 4,
  maxComputeWorkgroupStorageSize: 16_384,
} as const;

function limitValue(limits: GPUSupportedLimits, key: string, fallback: number): number {
  const value = (limits as unknown as Record<string, number | undefined>)[key];
  return typeof value === 'number' ? value : fallback;
}

/** Strict public checker for hosts and tests with a real GPUSupportedLimits. */
export function assertHybridDeviceCapable(limits: GPUSupportedLimits): void {
  const failed = Object.entries(HYBRID_WEBGPU_REQUIRED_LIMITS)
    .map(([key, required]) => ({
      key,
      required,
      actual: limitValue(
        limits,
        key,
        WEBGPU_DEFAULT_LIMITS[key as keyof typeof WEBGPU_DEFAULT_LIMITS] ?? 0,
      ),
    }))
    .filter(({ actual, required }) => actual < required);
  if (failed.length === 0) return;

  throw new TypeError(
    `[HybridEngine] device limits cannot create the walkaround-hybrid pipeline layouts: ` +
      failed
        .map(({ key, required, actual }) => `${key}=${actual} (requires >= ${required})`)
        .join(', ') +
      `. The host owns device creation — pass HYBRID_WEBGPU_REQUIRED_LIMITS as ` +
      `requiredLimits to GPUAdapter.requestDevice(), or select another backend.`,
  );
}

/**
 * Production GPUDevice.limits always reports every standard key. Some unit-test
 * doubles intentionally expose only the one limit under test; do not reinterpret
 * those partial objects as real adapters.
 */
export function assertHybridDeviceCapableIfReported(
  limits: GPUSupportedLimits | null | undefined,
): void {
  if (limits == null) return;
  const reported = limits as unknown as Record<string, unknown>;
  if (!Object.keys(HYBRID_WEBGPU_REQUIRED_LIMITS).every((key) => typeof reported[key] === 'number'))
    return;
  assertHybridDeviceCapable(limits);
}

/**
 * NRC extends hybrid group(3) with two packed storage arenas. The resulting
 * GI-RIS layout remains at the portable four-group / eight-storage floor.
 * Its default f32 fused trainer emits exactly two 32x64 workgroup arrays:
 * 2 * 32 * 64 * 4 = 16384 bytes.
 */
export const NRC_REQUIRED_MAX_BIND_GROUPS = 4;
export const NRC_REQUIRED_MAX_STORAGE_BUFFERS_PER_SHADER_STAGE = 8;
export const NRC_REQUIRED_WORKGROUP_STORAGE_BYTES = fusedMlpWorkgroupStorageBytes({
  useF16: DEFAULT_NRC_CONFIG.useF16,
  W: DEFAULT_NRC_CONFIG.width,
  TILE_B: DEFAULT_NRC_CONFIG.tileB,
});

/** Device limits for the actual resolved NRC shape rather than the historical
 * default trainer tile. */
export function nrcWebGpuRequiredLimitsForConfig(
  config: Partial<NrcConfig> = {},
): Readonly<Record<string, number>> {
  const resolved = resolveNrcConfig(config);
  return Object.freeze({
    ...HYBRID_WEBGPU_REQUIRED_LIMITS,
    maxBindGroups: NRC_REQUIRED_MAX_BIND_GROUPS,
    maxStorageBuffersPerShaderStage: NRC_REQUIRED_MAX_STORAGE_BUFFERS_PER_SHADER_STAGE,
    maxComputeWorkgroupStorageSize:
      computeNrcResourceFootprint(resolved).workgroupStorageBytes,
  });
}

/** Optional WebGPU features implied by the resolved trainer precision. */
export function nrcWebGpuRequiredFeaturesForConfig(
  config: Partial<NrcConfig> = {},
): readonly GPUFeatureName[] {
  return resolveNrcConfig(config).useF16 ? ['shader-f16'] : [];
}

export const NRC_WEBGPU_REQUIRED_LIMITS: Readonly<Record<string, number>> =
  nrcWebGpuRequiredLimitsForConfig(DEFAULT_NRC_CONFIG);

export function assertNrcDeviceCapable(
  limits: GPUSupportedLimits,
  config: Partial<NrcConfig> = {},
): void {
  const resolved = resolveNrcConfig(config);
  const required = {
    maxBindGroups: NRC_REQUIRED_MAX_BIND_GROUPS,
    maxStorageBuffersPerShaderStage: NRC_REQUIRED_MAX_STORAGE_BUFFERS_PER_SHADER_STAGE,
    maxComputeWorkgroupStorageSize:
      computeNrcResourceFootprint(resolved).workgroupStorageBytes,
  } as const;
  const failed = Object.entries(required)
    .map(([key, minimum]) => ({
      key,
      minimum,
      actual: limitValue(
        limits,
        key,
        WEBGPU_DEFAULT_LIMITS[key as keyof typeof WEBGPU_DEFAULT_LIMITS] ?? 0,
      ),
    }))
    .filter(({ actual, minimum }) => actual < minimum);
  if (failed.length === 0) return;

  throw new TypeError(
    `[HybridEngine] nrcEnabled requires additional device limits: ` +
      failed
        .map(({ key, minimum, actual }) => `${key}=${actual} (requires >= ${minimum})`)
        .join(', ') +
      `. The host owns device creation — request these together with ` +
      `HYBRID_WEBGPU_REQUIRED_LIMITS, or omit nrcEnabled.`,
  );
}

export function assertNrcDeviceCapableIfReported(
  limits: GPUSupportedLimits,
  config: Partial<NrcConfig> = {},
): void {
  const reported = limits as unknown as Record<string, unknown>;
  const keys = [
    'maxBindGroups',
    'maxStorageBuffersPerShaderStage',
    'maxComputeWorkgroupStorageSize',
  ] as const;
  if (!keys.every((key) => typeof reported[key] === 'number')) return;
  assertNrcDeviceCapable(limits, config);
}

/**
 * WebGPU features the hybrid pipeline requires.  Currently none — the
 * pipeline allocates every storage texture using base-spec-storage-capable
 * formats (rgba16float, rgba32float, rg32float, r32uint, rgba32uint). The
 * "downgraded from r16float / r16uint" notes in resourceManager.ts and the
 * GTAO/SVGF shaders are historical: those textures are intentionally
 * allocated in their wider, base-spec form so the engine runs on adapters
 * that lack the optional `texture-formats-tier1` feature (notably any host
 * driving us through three.js's WebGPURenderer, which omits tier1 from its
 * hardcoded feature enum).
 *
 * Kept exported as an empty readonly array for API stability — hosts that
 * already spread it into `requiredFeatures` continue to work. New code
 * should not depend on this export.
 *
 * Optional features (e.g. `timestamp-query` for dev-time per-pass GPU
 * timings) are handled separately by the host and are not part of the
 * required-features contract.
 */
export const HYBRID_WEBGPU_REQUIRED_FEATURES: readonly GPUFeatureName[] = [];

/**
 * Default camera squared-distance threshold for temporal accumulator reset.
 * 1.0 is calibrated to Cornell's ~2-unit room + OrbitControls damping
 * (~0.1–0.5 units per frame for ~30 frames after a drag release). Hosts on
 * different scene scales should override via
 * `HybridEngineOptions.cameraMoveResetThresholdSq`. See audit B8.
 */
const DEFAULT_CAMERA_MOVE_RESET_THRESHOLD_SQ = 1.0;

/**
 * Default squared world-space camera move above which the checkerboard sparse
 * path is forced FULL-RATE for that frame. 0.004 = 0.063² — much finer than the
 * temporal-accumulator reset above (1.0), because the checkerboard's half-rate
 * reservoir/radiance lag becomes visible at much smaller motion than a full
 * history discard. GPU-tuned (dzn motion A/B): checkerboard stays sparse through
 * a SLOW drag (~0.04 units/frame, where the reconstruction held ~50 dB) and
 * flips to full-rate by a faster pan (~0.07 units/frame, where it had fallen to
 * ~29 dB) — full-rate frames are then bit-identical to checkerboard-off.
 * Cornell-scale; hosts override via
 * `HybridEngineOptions.checkerboardMotionThresholdSq`. Only consulted when
 * checkerboard rendering is on.
 */
const DEFAULT_CHECKERBOARD_MOTION_THRESHOLD_SQ = 0.004;

/**
 * Default per-frame temporal-accumulator EMA weight. 0.01 = 99% history
 * retain, tuned for Cornell convergence at ~60 FPS. Framerate-dependent;
 * see audit M3.
 */
const DEFAULT_TEMPORAL_ACCUM_ALPHA = 0.01;

export class WalkaroundGPUPipeline implements BvhUpdateSink {
  // Private fields use the `_field` underscore prefix, matching HybridEngine.
  private _device: GPUDevice;
  private _width: number;
  private _height: number;

  /** Static BVH + TLAS + emitter GPU buffers (W4b — BvhBufferHost). */
  private readonly _bvhHost = new BvhBufferHost();
  /** Cached composite pass — avoids registry lookup in presentLastFrame. */
  private _compositePass: CompositePass | null = null;
  /** Swap-chain format this pipeline was compiled for (stored at initialize()). */
  private _swapChainFormat: GPUTextureFormat = 'bgra8unorm';
  /**
   * D3.3 — capture helper owning the lazily-compiled rgba8unorm render
   * pipeline and the offscreen capture texture.  Extracted from the inline
   * fields `_captureRenderPipeline` / `_captureOffscreenTex` to
   * {@link FrameCaptureHelper}; `captureOutputFrame` delegates to it.
   * `dispose()` calls `_captureHelper.dispose()` in the same order the
   * original inline destroy calls occupied.
   */
  private readonly _captureHelper = new FrameCaptureHelper();

  // Per-frame GPU resources (created by resourceManager.createFrameResources)
  private _res!: FrameResources;

  // Temporal accumulator ping-pong state
  private _accumPingPongIndex = 0; // 0 = read A, write B; 1 = swap
  private _accumFrameIndex = 0;
  private _grisHistoryEpoch = 1;
  private _grisHistoryClearPending = false;
  private _lastCameraPos: [number, number, number] = [0, 0, 0];

  // DDGI atlas binding state (layered hybrid). Owns the optional host-
  // supplied irradiance + visibility atlases and the cached placeholder UBO
  // for the no-DDGI fallback. Constructed once in the pipeline ctor; the
  // device handle is captured at that point so renderFrame / setDDGIInputs
  // can stay device-agnostic on this side.
  private readonly _ddgi: OptionalSubsystemBindingState;

  // PPG (Müller 2017) state — enabled flag, scene-bounds AABB, CPU sTree,
  // and the three serialise/upload writers used to be loose private members
  // (`_ppgEnabled`, `_ppgSTree`, `_ppgSceneAABB`, plus three methods).
  // Concentrated here so the orchestrator can stay focused on pass
  // scheduling. T2.H3 — `PPGCoordinator.enabled` mirrors the gate forwarded
  // into `PassGateOptions.ppgEnabled`.
  private readonly _ppg: PPGCoordinator;
  /** CPU shadow of the merged BVH position buffer used only to cold-restart
   * scene-bounds-dependent learned subsystems after geometry mutation. */
  private _learningBvhPositionsCpuData: ArrayBuffer | null = null;

  /** ReGIR (Boksansky 2021) grid coordinator. Constructed at `initialize`
   *  (config arrives in the init options). Off by default ⇒ a no-op coordinator
   *  whose `live` is false, so RIS uses the light-tree path. */
  private _regir: ReGIRCoordinator = new ReGIRCoordinator(resolveReGIRConfig());

  /** Phase-0 productization — PPG train-pass dispatch cadence (roadmap §5.3).
   *  The ppg-update pass dispatches only on frames where
   *  `_frameCount % _ppgDispatchInterval === 0`; the learned sTree/dTree GPU
   *  buffers persist between cycles and gi-ris guided sampling reads them every
   *  frame, so a higher interval trades training freshness for a lower per-frame
   *  cost. `1` = train every frame (no behaviour change — ultra/high). Always
   *  clamped to ≥ 1 at `initialize()` so a `0`/negative never skips forever.
   *  Set once per engine instance; mirrors the quality preset's
   *  `ppgDispatchInterval`. */
  private _ppgDispatchInterval = 1;

  /** Shared à-trous pipeline. Used by the legacy `AtrousDenoiser` (passed
   *  in via the dispatch context) AND by the always-on
   *  {@link AtrousIndirectPass}. Compiled once in pipelineCompiler and
   *  shared by both consumers (rationale: identical shader / BGL — forking
   *  a private compile per consumer would double the boot cost for zero
   *  functional benefit). */
  private _atrousPipeline!: GPUComputePipeline;
  /** Active denoiser (looked up from `_denoiserRegistry` after init). */
  private _denoiserMode: DenoiserId = 'atrous-variance';
  /** Phase-0 productization — quality-preset structural gating, fixed per
   *  engine instance (set at initialize()). `_gtaoEnabled` feeds the per-frame
   *  `gateOpts`; `_diSpatialPasses`/`_giSpatialPasses` size both the spatial
   *  Pass instances AND every `buildPassLayout` call so the timestamp slot
   *  layout matches the dispatched labels (Risk R2). Defaults preserve the
   *  pre-Phase-0 full layout (GTAO on, 2 spatial passes each). */
  private _gtaoEnabled = true;
  /** GTAO AO-compute downscale factor, fixed per engine instance at
   *  `initialize()`. `2` ⇒ half-res (`gtaoMode:'on'`, default); `4` ⇒
   *  quarter-res (`gtaoMode:'quarter'`). Sizes `gtao.aoHalfTexture`
   *  (`createFrameResources`) AND drives the per-frame GTAO dispatch +
   *  UBO field; the upsample reconstructs full-res AO at any factor. `'off'`
   *  leaves `_gtaoEnabled=false` so the downscale is moot (no AO dispatch). */
  private _gtaoDownscale: 2 | 4 = 2;
  private _diSpatialPasses: 1 | 2 = 2;
  private _giSpatialPasses: 1 | 2 = 2;
  /** GRIS DDGI-proxy reconnection-shift reuse (grisReuse). COMPILE-TIME
   *  gate: when true, the GI spatial + temporal pipelines are built with the
   *  two-group layout + GRIS shader and the passes bind the scene group at
   *  @group(1). When false (default) the GI passes are the verbatim Sprint-17
   *  single-group pipeline (the known-good default). Resolved once in
   *  initialize() from the host flag — NOT a per-frame UBO decision. */
  private _grisReuseStructural = false;
  /** Checkerboard half-res shading (HybridEngineOptions.checkerboardRendering).
   *  OFF by default ⇒ the RIS pass, the two DI spatial passes, and shade.wgsl all
   *  run full-res + ResolvePass passes through (byte-identity); ON ⇒ RIS, both
   *  spatial passes, AND shade compact their dispatch to the active-parity half
   *  (genuinely skipping the gap-parity BVH casts + candidate sampling) and
   *  ResolvePass reprojects the gap. The FULL-RATE temporal pass stays full-res:
   *  it reads each gap pixel's carried-forward reservoir (RIS seeds it on the
   *  frame that pixel is active; the parity flips each frame) and keeps refining
   *  it against the reprojected history, so every pixel always has a VALID
   *  reservoir for spatial/shade to consume. Resolved once in initialize() from
   *  the host flag; consumed per-frame as the motion-gated `cbActiveThisFrame`
   *  (forced full-rate above `_checkerboardMotionThresholdSq`) threaded into the
   *  UBO (frameParity / checkerboardOn), the ris/spatial/shade dispatch
   *  compaction, and the ResolvePass gap-fill. GPU-validated (dzn):
   *  spatial+shade ~1.28× whole-frame speedup at static/slow-motion, bit-identical
   *  to full-rate under faster motion. */
  private _checkerboard = false;
  /** NRC (Müller et al. 2021) live cache subsystem. Non-null ONLY when the
   *  engine was created with `nrcEnabled` (full-tier). When null (default) the
   *  gi-ris pipeline is the verbatim 4-group DDGI pass and no NRC GPU resources
   *  are allocated — the default pipeline is provably untouched. Owns the MLP
   *  trainer + the @group(4) NRC resources; host-owns-cadence train per frame. */
  private _nrc: NrcSubsystem | null = null;
  private readonly _onError: ((error: EngineError) => void) | null;
  private readonly _onWarning: ((warning: EngineWarning) => void) | null;
  private _lastNrcTrainingErrorMessage: string | null = null;
  /** Bundled layout config passed to every `buildPassLayout` call so the four
   *  call sites can't drift. */
  private get _passLayoutConfig(): {
    diSpatialPasses: 1 | 2;
    giSpatialPasses: 1 | 2;
    gtaoEnabled: boolean;
  } {
    return {
      diSpatialPasses: this._diSpatialPasses,
      giSpatialPasses: this._giSpatialPasses,
      gtaoEnabled: this._gtaoEnabled,
    };
  }
  /** Registry of all built-in denoisers; populated once at boot. */
  private _denoiserRegistry: DenoiserRegistry | null = null;
  /** The active denoiser instance for this pipeline (set in initialize). */
  private _activeDenoiser: Denoiser | null = null;
  /** Runtime bypass for dev A/B toggles (`engine.debug.setDenoiserEnabled`). */
  private _denoiserPassEnabled = true;
  /** Registry of non-denoiser passes; populated once at boot. */
  private _passRegistry: PassRegistry | null = null;
  /** Sorted pass list cached at boot; reused across frames. */
  private _sortedPasses: readonly Pass[] = [];
  /** Active neural graph handle; pipeline disposal releases its GPU resources. */
  private _inferenceGraph: InferenceGraph | null = null;
  /** Audit B8 — populated at initialize() time from HybridEngineOptions. */
  private _cameraMoveResetThresholdSq = DEFAULT_CAMERA_MOVE_RESET_THRESHOLD_SQ;
  /** Audit M3 — populated at initialize() time from HybridEngineOptions. */
  private _temporalAccumAlpha = DEFAULT_TEMPORAL_ACCUM_ALPHA;
  /** Checkerboard motion fallback threshold (squared world-space camera move
   *  above which checkerboard is forced full-rate for the frame). Populated at
   *  initialize() time; only consulted when `_checkerboard` is true. */
  private _checkerboardMotionThresholdSq = DEFAULT_CHECKERBOARD_MOTION_THRESHOLD_SQ;
  /** Sprint 18 follow-up — ping-pong index for the indirect temporal
   *  accumulator. Lives on the pipeline because the value persists across
   *  frames; the {@link IndirectTemporalAccumPass} reads + advances it
   *  through a {@link PingPongRef} wrapper. */
  private _indirectAccumPingPongRef: PingPongRef = { value: 0 };

  // Bind group layout memoisation cache
  private _bglCache: BGLCache = {};
  private _resourceCache = new PipelineResourceCache();

  // Per-pass UBO buffers owned by the pipeline (i.e. NOT owned by a
  // denoiser; denoiser-private UBOs are field-owned by each Denoiser
  // implementation under `denoisers/`). The indirect atrous UBO is passed by
  // reference into AtrousIndirectPass, lazy-allocated by buildAtrousBindGroup,
  // and destroyed by AtrousIndirectPass.dispose(); it is intentionally not in
  // the generic pipeline-owned list below.
  //
  // Eager pipeline-owned refs are allocated upfront in initialize().
  // dispose() walks those via `_perPassUboRefs` so adding a new eager UBO only
  // requires registering it there.
  /**
   * Sprint 18 — separate UBO for the indirect-channel atrous chain so it
   * doesn't race the legacy denoiser's per-iteration sigma writes.
   *
   * D4.3 — typed as {@link PassOwnedUboRef}: this UBO is passed by reference
   * into {@link AtrousIndirectPass}, which owns its GPU lifetime (lazy-allocates
   * in `buildAtrousBindGroup` and destroys in `AtrousIndirectPass.dispose()`).
   * It is intentionally NOT included in `_perPassUboRefs` so the orchestrator's
   * `dispose()` loop does not double-destroy it.
   */
  private _atrousIndirectUboRef: PassOwnedUboRef = { buf: undefined, __passOwned: true };
  private _accumUboRef: UboRef = { buf: undefined };
  // Sprint 9 — adaptive sampling UBOs.
  private _sampleBudgetUboRef: UboRef = { buf: undefined };
  private _sampleCountUboRef: UboRef = { buf: undefined };
  private _resolveUboRef: UboRef = { buf: undefined };
  /** Checkerboard pre-denoiser gap-fill UBO (16 bytes: screenW/H, frameParity, _pad). */
  private _cbPrefillUboRef: UboRef = { buf: undefined };
  /** Tonemap / exposure / outputColorSpace per-frame UBO for the composite pass
   *  (2026-06-10: FrameQualitySettings.tonemap / .exposure / .outputColorSpace). */
  private _compositeUboRef: UboRef = { buf: undefined };
  private get _perPassUboRefs(): readonly UboRef[] {
    return [
      this._accumUboRef,
      this._sampleBudgetUboRef,
      this._sampleCountUboRef,
      this._resolveUboRef,
      this._cbPrefillUboRef,
      this._compositeUboRef,
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

  /**
   * Live per-frame GPU resources, or `null` before `initialize()` resolves.
   *
   * @internal Package-internal use only. External debug consumers MUST use
   * {@link getDebugTextures} instead. Only `HybridEngine` may call this getter
   * directly (for `estimatedGpuMemoryBytes`, which needs the full struct).
   */
  get frameResources(): FrameResources | null {
    return this._initialized ? this._res : null;
  }

  get gpuMemoryExternalSections(): GpuMemoryExternalSections {
    return this._initialized
      ? { ...this._bvhHost.gpuMemorySections(), ...this._ddgi.gpuMemorySections() }
      : {};
  }

  /**
   * I3.2 — convenience wrapper that computes the full GPU memory breakdown for
   * the current frame resources. Returns `null` before `initialize()` resolves.
   *
   * Replaces the two-step dance (`pipeline.frameResources` +
   * `pipeline.gpuMemoryExternalSections`) that debug consumers previously had
   * to perform. The `frameResources` getter remains for callers that need the
   * raw struct, but debug-surface thunks should prefer this method.
   */
  getMemoryBreakdown(): GpuMemoryBreakdown | null {
    if (!this._initialized) return null;
    return estimateFrameResourcesMemory(this._res, this.gpuMemoryExternalSections);
  }

  /**
   * Read the ReSTIR-GI temporal reservoir buffers back to CPU (the reservoir
   * half of the "cached light field" export). Returns the three half-res
   * reservoir buffers (current / previous / spatial) as raw u32 + the grid
   * metadata, or null before `initialize()` resolves. Async (mapAsync).
   *
   * Models `ProbeUpdatePass.exportAtlasData` (the DDGI-atlas half) but for
   * storage buffers: copyBufferToBuffer → MAP_READ staging → slice out.
   * The cross-frame temporal history lives in `previous` (and `current`, which
   * equals it right after the end-of-frame copy); `spatial` is within-frame
   * scratch, persisted for a complete byte-identity round-trip.
   */
  async exportRestirGIReservoirs(device: GPUDevice): Promise<RestirGISnapshot | null> {
    if (!this._initialized) return null;
    const r = this._res.restirGI;
    const halfW = Math.max(1, Math.floor(this._width / 2));
    const halfH = Math.max(1, Math.floor(this._height / 2));
    const [current, previous, spatial] = await Promise.all([
      this.#readbackReservoir(device, r.reservoirGiCurrentBuffer),
      this.#readbackReservoir(device, r.reservoirGiPreviousBuffer),
      this.#readbackReservoir(device, r.reservoirGiSpatialBuffer),
    ]);
    return {
      halfW,
      halfH,
      strideU32: reservoirGiStrideU32ForGrisReuse(this._grisReuseStructural),
      current,
      previous,
      spatial,
    };
  }

  /**
   * Non-mutating compatibility check for a ReSTIR-GI reservoir restore.
   * HybridEngine runs this before publishing any other required GI section.
   */
  canImportRestirGIReservoirs(snap: RestirGISnapshot): boolean {
    if (
      !this._initialized ||
      snap == null ||
      typeof snap !== 'object' ||
      !isValidRestirGISnapshot(snap)
    ) {
      return false;
    }
    const r = this._res.restirGI;
    const halfW = Math.max(1, Math.floor(this._width / 2));
    const halfH = Math.max(1, Math.floor(this._height / 2));
    if (
      !(snap.current instanceof Uint32Array) ||
      !(snap.previous instanceof Uint32Array) ||
      !(snap.spatial instanceof Uint32Array) ||
      snap.halfW !== halfW ||
      snap.halfH !== halfH ||
      snap.strideU32 !== reservoirGiStrideU32ForGrisReuse(this._grisReuseStructural)
    ) {
      return false; // grid / stride mismatch — cannot restore into a different reservoir layout
    }
    if (
      r.reservoirGiCurrentBuffer === r.reservoirGiPreviousBuffer ||
      r.reservoirGiCurrentBuffer === r.reservoirGiSpatialBuffer ||
      r.reservoirGiPreviousBuffer === r.reservoirGiSpatialBuffer ||
      r.reservoirGiCurrentBuffer.size !== r.reservoirGiPreviousBuffer.size ||
      r.reservoirGiCurrentBuffer.size !== r.reservoirGiSpatialBuffer.size ||
      r.reservoirGiCurrentBuffer.size <= 0 ||
      r.reservoirGiCurrentBuffer.size % Uint32Array.BYTES_PER_ELEMENT !== 0
    ) {
      return false;
    }
    const expectU32 = r.reservoirGiCurrentBuffer.size / 4;
    return (
      snap.current.length === expectU32 &&
      snap.previous.length === expectU32 &&
      snap.spatial.length === expectU32
    );
  }

  /**
   * Populate a complete replacement reservoir cohort without publishing it.
   * Every fallible allocation/map/upload step happens before commit, so a
   * synchronous failure cannot leave one or two live buffers overwritten.
   */
  prepareRestirGIReservoirImport(
    device: GPUDevice,
    snap: RestirGISnapshot,
  ): RestirGIReservoirImportTransaction | null {
    if (
      device !== this._device ||
      !this.canImportRestirGIReservoirs(snap)
    ) {
      return null;
    }
    const live = this._res.restirGI;
    const size = live.reservoirGiCurrentBuffer.size;
    const usage =
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST;
    const candidates: GPUBuffer[] = [];
    const liveBuffers = new Set<GPUBuffer>([
      live.reservoirGiCurrentBuffer,
      live.reservoirGiPreviousBuffer,
      live.reservoirGiSpatialBuffer,
    ]);
    try {
      const createCandidate = (
        label: string,
        data: Uint32Array,
      ): GPUBuffer => {
        const candidate = device.createBuffer({
          label,
          size,
          usage,
          mappedAtCreation: true,
        });
        candidates.push(candidate);
        if (
          candidate.size !== size ||
          liveBuffers.has(candidate) ||
          candidates.slice(0, -1).includes(candidate)
        ) {
          throw new Error(
            'ReSTIR-GI import candidate aliases a live or sibling buffer.',
          );
        }
        const mapped = candidate.getMappedRange();
        if (mapped.byteLength !== size) {
          throw new Error(
            'ReSTIR-GI import candidate exposed an unexpected mapped size.',
          );
        }
        new Uint32Array(mapped).set(data);
        candidate.unmap();
        return candidate;
      };
      const current = createCandidate(
        'reservoir-gi-current.import-candidate',
        snap.current,
      );
      const previous = createCandidate(
        'reservoir-gi-previous.import-candidate',
        snap.previous,
      );
      const spatial = createCandidate(
        'reservoir-gi-spatial.import-candidate',
        snap.spatial,
      );
      const replacement = Object.freeze({
        reservoirGiCurrentBuffer: current,
        reservoirGiPreviousBuffer: previous,
        reservoirGiSpatialBuffer: spatial,
      });
      // Cache invalidation is deliberately staged before DDGI publication by
      // HybridEngine. If a future cache implementation can throw, the required
      // atlas cohort is still untouched and these candidates are retired.
      this._resourceCache.clear();

      let state: 'prepared' | 'committed' | 'aborted' = 'prepared';
      return {
        commit: () => {
          if (state !== 'prepared') return;
          this._res.restirGI = replacement;
          state = 'committed';
          candidates.length = 0;
          for (const buffer of liveBuffers) {
            try {
              buffer.destroy();
            } catch {
              // The replacement cohort is already live; retirement is best-effort.
            }
          }
        },
        abort: () => {
          if (state !== 'prepared') return;
          state = 'aborted';
          for (const candidate of new Set(candidates)) {
            if (liveBuffers.has(candidate)) continue;
            try {
              candidate.destroy();
            } catch {
              // Preserve the original prepare/DDGI failure.
            }
          }
          candidates.length = 0;
        },
      };
    } catch (error) {
      for (const candidate of new Set(candidates)) {
        if (liveBuffers.has(candidate)) continue;
        try {
          candidate.destroy();
        } catch {
          // Preserve the allocation/map/upload failure.
        }
      }
      throw error;
    }
  }

  /**
   * Transactional one-shot reservoir restore for callers that do not need to
   * coordinate it with another GI section.
   */
  importRestirGIReservoirs(device: GPUDevice, snap: RestirGISnapshot): boolean {
    const transaction = this.prepareRestirGIReservoirImport(device, snap);
    if (transaction == null) return false;
    transaction.commit();
    return true;
  }

  /**
   * Export the current PPG sTree + per-cell dTree guiding distribution as flat
   * buffers. Returns null when PPG is disabled or not yet initialised.
   *
   * Delegates to {@link PPGCoordinator.exportSTree}; exists here so HybridEngine
   * can call it through the pipeline reference without needing to reach inside
   * the coordinator directly.
   */
  exportPPGSTree(): ReturnType<PPGCoordinator['exportSTree']> {
    if (!this._initialized) return null;
    return this._ppg.exportSTree();
  }

  /**
   * Restore a PPG snapshot into the live coordinator + GPU buffers. Returns
   * false when PPG is disabled, not initialised, or the snapshot is incompatible.
   *
   * Delegates to {@link PPGCoordinator.importSTree} with the live FrameResources.
   */
  importPPGSTree(snapshot: Parameters<PPGCoordinator['importSTree']>[0]): boolean {
    if (!this._initialized) return false;
    return this._ppg.importSTree(snapshot, this._res);
  }

  /** copyBufferToBuffer (size is a multiple of 4) → MAP_READ → unpadded Uint32Array. */
  async #readbackReservoir(device: GPUDevice, src: GPUBuffer): Promise<Uint32Array> {
    const bytes = src.size; // already 4-aligned (stride is 28 u32; floor is 256)
    const staging = device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(src, 0, staging, 0, bytes);
    device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const out = new Uint32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    return out;
  }

  /**
   * Narrow debug-texture handle set consumed by dev overlays
   * (`HybridEngineDebug.giSignalTextures`). Returns `null` before
   * `initialize()` resolves.
   *
   * Textures are owned by the pipeline — callers MUST NOT destroy them.
   * Handles are invalidated on the next `setScene()` / `dispose()` /
   * `resize()`.
   */
  getDebugTextures(): PipelineDebugTextures | null {
    if (!this._initialized) return null;
    return {
      hdrColorTexture: this._res.common.hdrColorTexture ?? null,
      hdrIndirectTexture: this._res.common.hdrIndirectTexture ?? null,
      aoFullTexture: this._res.gtao.aoFullTexture ?? null,
    };
  }

  /**
   * Aux G-buffer views surfaced through `FrameRendered.{normalDepth, albedo,
   * variance, motionVectors}` (the `EngineCapabilities.supportsAuxBuffers`
   * contract). All four are full-res + always allocated (they are core GI /
   * denoiser inputs),
   * so this never partially returns:
   *   normalDepth   — rgba16float, xyz = normal * 0.5 + 0.5,
   *                   w = signed linear depth.
   *   albedo        — rgba16float, demodulated visible-point diffuse albedo
   *                   (Schied 2017 §4.1) — lighting × albedo = final colour.
   *   variance      — rgba16float, freshest full-resolution Welford estimate.
   *   motionVectors — rgba32float, (dx, dy) screen-space pixels in .xy.
   * Descriptor-free views are cached while resources are stable; owned by the
   * pipeline — callers MUST NOT destroy them, and the handles are invalidated on
   * the next setScene / resize / dispose. Null before initialize() resolves.
   */
  getAuxBufferTextures(): {
    normalDepth: GPUTextureView;
    albedo: GPUTextureView;
    variance: GPUTextureView;
    motionVectors: GPUTextureView;
  } | null {
    if (!this._initialized) return null;
    const c = this._res.common;
    return {
      normalDepth: this._resourceCache.textureView(c.gNormalDepthTexture),
      albedo: this._resourceCache.textureView(c.albedoTexture),
      variance: this._resourceCache.textureView(
        this._activeDenoiser?.welfordPing === 1
          ? c.varianceBufferAux
          : c.varianceBuffer,
      ),
      motionVectors: this._resourceCache.textureView(c.motionVectorTexture),
    };
  }

  /** The most recent post-denoise/composite-input HDR radiance texture
   *  (`resolvedTexture`: rgba16float, internal render resolution, linear pre-tonemap;
   *  `TEXTURE_BINDING | COPY_SRC`). This is the seed source for the progressive
   *  walkaround→PT handoff (P8 increment 2) — a host samples/copies it into a
   *  converged PT engine's accumulator. Null before the pipeline is initialised; the
   *  caller must consume it SYNCHRONOUSLY in the handoff frame (it is recycled each
   *  frame and destroyed on resize/dispose). */
  getProgressiveSeedTexture(): GPUTexture | null {
    if (!this._initialized) return null;
    return this._res.common.resolvedTexture;
  }

  /**
   * Run the composite pass into an engine-owned offscreen `rgba8unorm` texture
   * and read it back to a Float32Array (RGBA, unorm → [0,1], row-major,
   * top-left origin).
   *
   * D3.3 — delegates to {@link FrameCaptureHelper.captureFrame}. All GPU
   * resource management (lazy pipeline compile, offscreen texture create/resize,
   * unorm decode) lives in the helper. The pipeline holds an instance at
   * `_captureHelper`; `dispose()` calls `_captureHelper.dispose()` in the same
   * position the original inline destroy calls occupied.
   *
   * Returns `null` when:
   *   - the pipeline is not yet initialised (no frame rendered), OR
   *   - the `_compositeUboRef` buffer is absent (UBO not yet allocated).
   *
   * Pipeline stall: submits copyTextureToBuffer + mapAsync.  Use for
   * debugging / screenshot export, not per-frame readback.
   */
  async captureOutputFrame(): Promise<Float32Array | null> {
    if (!this._initialized) return null;
    const compositeUbo = this._compositeUboRef.buf;
    if (compositeUbo == null) return null;
    const compositePass = this._compositePass;
    if (compositePass == null) return null;
    return this._captureHelper.captureFrame(
      this._device,
      this._width,
      this._height,
      this._swapChainFormat,
      compositePass,
      compositeUbo,
      this._bglCache,
      this._res,
    );
  }

  /** Temporal-accumulator history depth: frames accumulated since the last
   *  α=1 reset (camera motion, `requestAccumReset`, or `resize`). Increments
   *  once per rendered frame; reset to 0 on each of those events. Read by
   *  `HybridEngine.onProgress` for the `'denoiser-converge'` fraction. */
  get accumFrameIndex(): number {
    return this._accumFrameIndex;
  }

  /**
   * Return the current {@link DenoiserState} from the active denoiser.
   * Returns `null` before the pipeline is initialised (no active denoiser).
   * Consumed by the frame telemetry path to populate
   * `FrameStats.denoiserState` without exposing the private `_activeDenoiser`.
   */
  getActiveDenoiserState(): import('./denoisers/index.js').DenoiserState | null {
    return this._activeDenoiser?.state() ?? null;
  }

  /** Temporal-accumulator EMA weight α (history blend `1-α` per frame).
   *  The effective convergence window is ≈ `1/α` frames (α=0.01 ⇒ ~100).
   *  Target denominator for the `'denoiser-converge'` progress metric. */
  get temporalAccumAlpha(): number {
    return this._temporalAccumAlpha;
  }

  constructor(
    device: GPUDevice,
    width: number,
    height: number,
    diagnostics: {
      onError?: (error: EngineError) => void;
      onWarning?: (warning: EngineWarning) => void;
    } = {},
  ) {
    this._device = device;
    this._width = width;
    this._height = height;
    this._onError = diagnostics.onError ?? null;
    this._onWarning = diagnostics.onWarning ?? null;
    this._ddgi = new OptionalSubsystemBindingState(device);
    const ppgDiagnostics: {
      onError?: (error: EngineError) => void;
      onWarning?: (warning: EngineWarning) => void;
    } = {};
    if (diagnostics.onError) ppgDiagnostics.onError = diagnostics.onError;
    if (diagnostics.onWarning) ppgDiagnostics.onWarning = diagnostics.onWarning;
    this._ppg = new PPGCoordinator(device, ppgDiagnostics);
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
    const layout = buildPassLayout({ denoiserMode: this._denoiserMode, ...this._passLayoutConfig });
    return readTimestampsOnce(this._device, this._tsState, layout);
  }

  /** Snapshot the active NRC subsystem's cumulative diagnostics. */
  getNrcDiagnostics(): NrcDiagnostics | null {
    return this._nrc?.diagnostics() ?? null;
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
      /** Checkerboard motion fallback — squared world-space camera-move above
       *  which the checkerboard sparse path is forced full-rate for that frame
       *  (only consulted when `checkerboard` is on). Default 0.0009 (0.03²). */
      checkerboardMotionThresholdSq?: number;
      /** Neural denoiser graph, required when the resolved mode is `'neural'`.
       *  Its device/dimensions are validated before wrapper publication. */
      inferenceGraph?: InferenceGraph;
      /** T2.H2 — host-provided neural weights retained for graph reinitialize on resize. */
      neuralWeights?: ModelWeights;
      /** T2.H3 — enable PPG (Müller 2017 adaptive sTree + dTree + MIS). */
      ppgEnabled?: boolean;
      /** H47 — maximum PPG sTree spatial cells forwarded to allocatePPGResources.
       *  undefined ⇒ use allocatePPGResources default (1 024). */
      ppgMaxSpatialCells?: number;
      /** H29 — maximum per-cell PPG dTree nodes. Threaded to both
       *  buildPpgUpdateWgsl and allocatePPGResources so the shader stride and
       *  buffers agree. undefined ⇒ default 341-node stride. */
      ppgMaxDTreeNodesPerCell?: number;
      /** PPG guide/cosine MIS mixture alpha. Defaults to PPG_MIS_ALPHA. */
      ppgMixAlpha?: number;
      /** W11 — OIDN final-pass denoiser config (required when denoiser='oidn-final').
       *  Threaded into `registerBuiltinDenoisers` so the OIDN entry registers as a
       *  real (non-disabled) denoiser; missing on a 'oidn-final' selection causes
       *  the registry to reject lookup with a clear remediation message. */
      oidn?: {
        modelUrl: string;
        executionProviders?: ReadonlyArray<'webnn' | 'webgpu' | 'wasm'>;
      };
      /** Phase-0 productization — GTAO dispatch mode. `'off'` gates GTAO +
       *  its upsample out (the AO target keeps its 1.0 init = no occlusion).
       *  `'on'` runs AO at half-res (W/2 × H/2 target + bilateral upsample).
       *  `'quarter'` runs AO at a true quarter-res (W/4 × H/4 target — 1/16
       *  the AO compute footprint), upsampled to full-res through the same
       *  depth/normal-aware bilateral machinery (the downscale factor is a
       *  UBO field consumed by both gtao + gtaoUpsample shaders). */
      gtaoMode?: 'on' | 'quarter' | 'off';
      /** Phase-0 — ReSTIR-DI spatial ping-pong pass count (1 or 2). Default 2. */
      diSpatialPasses?: 1 | 2;
      /** Phase-0 — ReSTIR-GI spatial ping-pong pass count (1 or 2). Default 2. */
      giSpatialPasses?: 1 | 2;
      /** GRIS DDGI-proxy reconnection-shift reuse (Lin et al. 2022) — opt-in.
       *  COMPILE-TIME structural gate: when true, the GI spatial + temporal
       *  pipelines are built with a `@group(1)` scene BVH/TLAS group (for the
       *  reconnection-visibility ray) + the GRIS combine shader; when false
       *  (default) they are the verbatim Sprint-17 single-group pipeline. This
       *  MUST be a compile-time decision — a runtime UBO flag that bound an
       *  extra group on the default path regressed the default render to an
       *  all-black frame (f8df9a4). Host opt-in via
       *  `HybridEngineOptions.grisReuse`. */
      grisReuse?: boolean;
      /** Checkerboard half-res shading (HybridEngineOptions.checkerboardRendering).
       *  OFF (default) ⇒ shade.wgsl + both DI spatial passes shade every pixel and
       *  ResolvePass passes through (byte-identical to the pre-checkerboard
       *  pipeline). ON ⇒ shade AND the two spatial passes compact their dispatch to
       *  the active-parity half and ResolvePass reprojects the gap. Stored on
       *  `_checkerboard`; consumed per-frame as the motion-gated `cbActiveThisFrame`
       *  (frameParity / checkerboardOn) across the UBO, the dispatch compaction, and
       *  the ResolvePass gap-fill. GPU-validated on dzn. */
      checkerboard?: boolean;
      /** NRC (Müller et al. 2021) live cache — COMPILE-TIME structural gate.
       *  When true, the gi-ris pipeline is built with a 5th `@group(4)` NRC bind
       *  group + the inline-MLP-forward shader variant, and a per-engine
       *  {@link NrcSubsystem} owns the MLP trainer + query resources; when false
       *  (default) the gi-ris pipeline is the verbatim 4-group DDGI-estimate pass
       *  and no NRC resources are allocated. MUST be a compile-time decision — a
       *  runtime UBO flag binding a fifth group on the default path is the
       *  GRIS-class regression (f8df9a4). Full-tier-only (the ctor forbids
       *  `tier:'lite' + nrcEnabled`). */
      nrcEnabled?: boolean;
      /** Complete NRC query/trainer/allocation contract. */
      nrcConfig?: Partial<NrcConfig>;
      /** Phase-0 — PPG train-pass dispatch cadence. The update pass dispatches
       *  only on frames where `frameCount % N === 0`. `1`
       *  (default) trains every frame; `N > 1` skips off-interval frames. The
       *  learned tree persists between cycles, so gi-ris guided sampling is
       *  unaffected. Clamped to ≥ 1. Only meaningful when `ppgEnabled` is true. */
      ppgDispatchInterval?: number;
      /** ReGIR (Boksansky 2021) grid-based DI light selection. When
       *  `enabled`, a per-frame grid-build pass pre-resamples lights into a
       *  world-space grid of reservoirs and RIS samples the containing cell
       *  instead of traversing the light tree per pixel (O(1) per pixel
       *  regardless of light count). The grid is seeded by the light tree, so
       *  it only goes live when the tree is live (≥ 2 emitters). When off, RIS
       *  uses the light-tree path bit-identically. */
      regirConfig?: Partial<ReGIRConfig>;
    },
  ): Promise<void> {
    assertHybridDeviceCapableIfReported(this._device.limits);
    let effectiveOptions = options;
    if (options?.nrcEnabled === true) {
      const resolvedNrcConfig = resolveNrcConfig(options.nrcConfig ?? {});
      assertNrcDeviceCapableIfReported(this._device.limits, resolvedNrcConfig);
      const aabb = deriveSceneAABBFromBvhPositions(bvhBuffers);
      // Exact buffer/binding/workgroup/feature preflight runs before any
      // pipeline-owned allocation. In particular, useF16 checks the actually
      // enabled shader-f16 device feature rather than assuming the default f32
      // trainer shape.
      preflightNrcResources(
        this._device,
        resolvedNrcConfig,
        aabb.min,
        aabb.max,
      );
      effectiveOptions = { ...options, nrcConfig: resolvedNrcConfig };
    }
    const { _width: W, _height: H } = this;

    // D3.1 — initialize() is now a sequencer of three private phases.
    // Exact statement order is preserved across the extraction boundary —
    // init order is load-bearing (each phase's outputs feed the next).
    const { compiled, ppgEnabled } = await this.#initGpuResources(
      bvhBuffers,
      swapChainFormat,
      effectiveOptions,
    );
    await this.#initPasses(compiled, bvhBuffers, effectiveOptions);
    this.#initSubsystems(bvhBuffers, ppgEnabled, W, H, effectiveOptions);

    this._initialized = true;
    if (effectiveOptions?.verbose) {
      console.log('[ReSTIR] Pipeline initialized', {
        W,
        H,
        bvhNodes: bvhBuffers.bvhNodes.count,
        emitters: bvhBuffers.emitterCount,
      });
    }
  }

  /**
   * D3.1 — Phase 1: GPU resource allocation + shader compilation.
   *
   * Covers (in exact original order):
   *   - ReGIR coordinator construction + BVH buffer upload
   *   - GTAO mode + denoiser mode resolution (must precede frame-resource alloc)
   *   - Per-frame GPU resource allocation (`createFrameResources`)
   *   - GRIS / checkerboard / NRC structural gate resolution (must precede compile)
   *   - NRC subsystem construction + initialization (compile-time gate)
   *   - Shader pipeline compilation (`compilePipelines`)
   *   - Scalar tuning field assignments
   *   - PPG enabled-flag derivation (used by both #initPasses and #initSubsystems)
   *
   * Returns the compiled pipeline bundle and the resolved ppgEnabled flag so
   * the subsequent phases can consume them without re-deriving.
   */
  async #initGpuResources(
    bvhBuffers: SceneBVHBuffers,
    swapChainFormat: GPUTextureFormat,
    options: Parameters<WalkaroundGPUPipeline['initialize']>[2],
  ): Promise<{ compiled: Awaited<ReturnType<typeof compilePipelines>>; ppgEnabled: boolean }> {
    const d = this._device;
    const { _width: W, _height: H } = this;

    // ── ReGIR coordinator (Boksansky 2021) ────────────────────────────────
    // Construct from the resolved config so the grid byte count is known
    // BEFORE the light-tree buffer is uploaded (the grid is co-located in the
    // SAME buffer — see ReGIRCoordinator / regir.wgsl). `gridRegionBytes()` is
    // 0 when ReGIR is off ⇒ the light-tree buffer is byte-identical to before.
    this._regir = new ReGIRCoordinator(resolveReGIRConfig(options?.regirConfig));
    this._regir.assertDeviceLimits(d);
    this._bvhHost.setRegirGridBytes(this._regir.gridRegionBytes());

    // ── Upload BVH buffers ────────────────────────────────────────────────
    this._bvhHost.uploadInitial(d, bvhBuffers);
    this._learningBvhPositionsCpuData = bvhBuffers.bvhPositions.cpuData.slice(0);

    // Phase-0 productization — resolve the GTAO mode BEFORE allocating frame
    // resources, since `'quarter'` sizes the AO target at a smaller resolution.
    //   'off'     → AO gated out (aoFullTexture keeps its 1.0 init).
    //   'on'      → half-res AO target (downscale 2) — Sprint-15 default.
    //   'quarter' → quarter-res AO target (downscale 4) — a real step below
    //               'on': 1/4 each axis, 1/16 the AO compute footprint.
    const gtaoMode = options?.gtaoMode ?? 'on';
    this._gtaoEnabled = gtaoMode !== 'off';
    this._gtaoDownscale = gtaoMode === 'quarter' ? 4 : 2;

    // Resolve the active denoiser mode BEFORE allocating frame resources so the
    // SVGF-real ~80-90 MB @1080p persistent-texture fleet is only allocated when
    // svgf-real is actually the active denoiser (G-P2.6). The id is fixed for the
    // pipeline's lifetime; resize() reuses the stored `_denoiserMode`.
    this._denoiserMode = options?.denoiser ?? 'atrous-variance';

    // ── Resolve the GRIS structural gate BEFORE allocating frame resources ─
    // grisReuse widens the GI reservoir buffers from the compact 20-u32
    // Sprint-16/17 layout to the 30-u32 GRIS cache layout, and also selects the
    // GI spatial + temporal shader/layout variants below.
    this._grisReuseStructural = options?.grisReuse ?? false;

    // ── Per-frame GPU resources ───────────────────────────────────────────
    this._res = createFrameResources(d, W, H, {
      gtaoDownscale: this._gtaoDownscale,
      svgfEnabled: this._denoiserMode === 'svgf-real',
      grisReuse: this._grisReuseStructural,
      welfordPingPong: this._denoiserMode === 'atrous-variance',
      checkerboard: options?.checkerboard === true,
    });

    // ── Resolve the checkerboard half-res-shading flag ─────────────────────
    // OFF (default) ⇒ shade shades every pixel + ResolvePass passes through
    // (byte-identity). Not a compile-time structural decision — no extra bind
    // groups; it only flips two already-present UBO fields + the ResolvePass
    // gate — so it is resolved here and consumed at construction + per frame.
    this._checkerboard = options?.checkerboard ?? false;

    // ── Resolve the NRC structural gate BEFORE compiling pipelines ─────────
    // nrcEnabled is a COMPILE-TIME decision (mirrors grisReuse): it selects
    // the gi-ris pipeline layout (4-group DDGI default vs 5-group inline-MLP
    // variant) + shader variant. When ON we construct + initialize the
    // per-engine NrcSubsystem (which owns the MLP trainer + the @group(4) query
    // resources) and pass its WGSL config to compilePipelines so the shader's
    // baked encoding sizes match the host buffers; the gi-ris pass then binds
    // the subsystem's bind group at slot 4. When OFF (default) `_nrc` stays null
    // and gi-ris compiles + dispatches the verbatim 4-group DDGI pass — the
    // default pipeline structure is provably untouched (the GRIS-class
    // regression discipline, f8df9a4).
    if (options?.nrcEnabled === true) {
      // NRC-ON capability gate (host-owns-lifecycle) — fail early + legibly if
      // the device lacks the @group(4) 5th bind group / the fused-MLP workgroup
      // storage, rather than cryptically in createComputePipeline.
      const nrcConfig = options.nrcConfig ?? {};
      assertNrcDeviceCapableIfReported(d.limits, nrcConfig);
      this._nrc = new NrcSubsystem(d, this._bglCache, nrcConfig);
      const aabb = deriveSceneAABBFromBvhPositions(bvhBuffers);
      await this._nrc.initialize(aabb.min, aabb.max);
    }

    // Store the swap-chain format so captureOutputFrame can decide whether to
    // reuse the existing composite pipeline or compile a capture variant.
    this._swapChainFormat = swapChainFormat;

    // ── Compile shaders (denoiser-agnostic) ───────────────────────────────
    const compiled = await compilePipelines(d, this._bglCache, swapChainFormat, {
      verbose: options?.verbose ?? false,
      ...(this._onWarning !== null ? { onWarning: this._onWarning } : {}),
      ppgEnabled: options?.ppgEnabled ?? false,
      ...(options?.ppgMaxDTreeNodesPerCell !== undefined
        ? { ppgMaxDTreeNodesPerCell: options.ppgMaxDTreeNodesPerCell }
        : {}),
      regirEnabled: this._regir.config.enabled,
      grisReuse: this._grisReuseStructural,
      // NRC ON ⇒ pass the subsystem's WGSL config so the gi-ris pipeline builds
      // the 5-group inline-MLP variant with byte-matching encoding sizes.
      ...(this._nrc !== null ? { nrcConfig: this._nrc.wgslConfig() } : {}),
    });
    // Shared à-trous pipeline — fed into the AtrousDenoiser context AND
    // the always-on AtrousIndirectPass.
    this._atrousPipeline = compiled.atrousPipeline;
    // `_denoiserMode` was already resolved before createFrameResources (so the
    // SVGF allocation could be gated on it) — no need to re-derive it here.
    this._diSpatialPasses = options?.diSpatialPasses ?? 2;
    this._giSpatialPasses = options?.giSpatialPasses ?? 2;
    // PPG train-pass cadence. Clamp to ≥ 1 (a 0/negative interval would make
    // `frameCount % N` undefined / skip the train passes forever). Floor any
    // fractional host value so the modulo is integer-clean.
    this._ppgDispatchInterval = resolvePpgDispatchInterval(
      options?.ppgDispatchInterval ?? 1,
    );
    this._cameraMoveResetThresholdSq =
      options?.cameraMoveResetThresholdSq ?? DEFAULT_CAMERA_MOVE_RESET_THRESHOLD_SQ;
    this._temporalAccumAlpha = options?.temporalAccumAlpha ?? DEFAULT_TEMPORAL_ACCUM_ALPHA;
    this._checkerboardMotionThresholdSq =
      options?.checkerboardMotionThresholdSq ?? DEFAULT_CHECKERBOARD_MOTION_THRESHOLD_SQ;

    // T2.H3 — PPG is enabled iff host opted-in AND both pipelines compiled.
    // The flag itself is computed here; `_ppg.initialize()` below acts on it
    // (allocates resources, builds sTree, uploads UBOs) once the pass
    // registry is wired.
    const ppgEnabled = (options?.ppgEnabled ?? false) && compiled.ppgUpdatePipeline !== undefined;

    return { compiled, ppgEnabled };
  }

  /**
   * D3.1 — Phase 2: denoiser registry, timestamp init, UBO allocation, and
   * pass registration + parallel initialization.
   *
   * Covers (in exact original order):
   *   - Denoiser registry construction, builtin registration, active-denoiser
   *     lookup + initialization
   *   - InferenceGraph ownership handle store
   *   - Timestamp query init
   *   - Eager pipeline-owned UBO allocation
   *   - Pass registry construction via `registerPasses` + sorted-pass cache
   *   - Parallel pass initialization
   */
  async #initPasses(
    compiled: Awaited<ReturnType<typeof compilePipelines>>,
    bvhBuffers: SceneBVHBuffers,
    options: Parameters<WalkaroundGPUPipeline['initialize']>[2],
  ): Promise<void> {
    const d = this._device;
    const { _width: W, _height: H } = this;

    // ── Denoiser registry: build, register builtins, look up + initialise
    //    the active denoiser. `neural` / `oidn-final` are REAL denoisers; they
    //    register as DISABLED only when their host config is absent (no
    //    InferenceGraph / no OIDN modelUrl), in which case lookup() rejects them
    //    with a clear error. When configured they pass lookup() and initialise.
    this._denoiserRegistry = new DenoiserRegistry();
    registerBuiltinDenoisers(this._denoiserRegistry, {
      ...(options?.inferenceGraph !== undefined
        ? { neuralInferenceGraph: options.inferenceGraph }
        : {}),
      ...(options?.neuralWeights !== undefined ? { neuralWeights: options.neuralWeights } : {}),
      ...(this._onWarning !== null ? { onWarning: this._onWarning } : {}),
      // exactOptionalPropertyTypes-safe: only forward `oidn` when supplied.
      ...(options?.oidn !== undefined ? { oidn: options.oidn } : {}),
    });
    this._activeDenoiser = this._denoiserRegistry.lookup(this._denoiserMode);
    await this._activeDenoiser.initialize({
      device: d,
      width: W,
      height: H,
      bglCache: this._bglCache,
      frameResources: this._res,
    });

    // Retain the configured graph so the pipeline owns and releases it. Registry
    // lookup above has already rejected explicit neural mode when it is absent.
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
    this._accumUboRef.buf = d.createBuffer({ label: 'accum-ubo', size: 16, usage: U });
    // Sprint 9 — adaptive sampling UBOs (always allocated; passes always run).
    this._sampleBudgetUboRef.buf = d.createBuffer({
      label: 'sample-budget-ubo',
      size: 16,
      usage: U,
    });
    this._sampleCountUboRef.buf = d.createBuffer({ label: 'sample-count-ubo', size: 16, usage: U });
    this._resolveUboRef.buf = d.createBuffer({ label: 'resolve-ubo', size: 16, usage: U });
    this._cbPrefillUboRef.buf = d.createBuffer({ label: 'cb-prefill-ubo', size: 16, usage: U });
    // 2026-06-10 — per-frame composite UBO (tonemap/exposure/outputColorSpace).
    this._compositeUboRef.buf = d.createBuffer({ label: 'composite-ubo', size: 16, usage: U });

    // ── Pass registry: instantiate + register all non-denoiser passes ────
    // Order of registration is irrelevant; the registry topologically sorts.
    // Extracted to the module-scope `registerPasses` free function (Task 4.1).
    // The thunks below preserve the original closures' late-`this` binding.
    const { registry, compositePass } = registerPasses(compiled, {
      diSpatialPasses: this._diSpatialPasses,
      giSpatialPasses: this._giSpatialPasses,
      grisReuseStructural: this._grisReuseStructural,
      checkerboard: this._checkerboard,
      cbPrefillUboRef: this._cbPrefillUboRef,
      sampleBudgetUboRef: this._sampleBudgetUboRef,
      sampleCountUboRef: this._sampleCountUboRef,
      accumUboRef: this._accumUboRef,
      resolveUboRef: this._resolveUboRef,
      atrousIndirectUboRef: this._atrousIndirectUboRef,
      compositeUboRef: this._compositeUboRef,
      indirectAccumPingPongRef: this._indirectAccumPingPongRef,
      regir: this._regir,
      bglCache: this._bglCache,
      bvhBuffers,
      nrcClearSlotClaims:
        this._nrc !== null ? (encoder) => this._nrc!.clearSlotClaims(encoder) : undefined,
      getActiveDenoiser: () => this._activeDenoiser!,
      getAtrousPipeline: () => this._atrousPipeline,
      isDenoiserPassEnabled: () => this._denoiserPassEnabled,
      getRegirResources: () => ({
        combinedLightTreeBuffer: this._bvhHost.lightTreeBuffer(),
        uboBuffer: this._res.common.uboBuffer,
      }),
    });
    this._compositePass = compositePass;

    // ── Initialize all passes in parallel ────────────────────────────────
    this._passRegistry = registry;
    this._sortedPasses = registry.sortedPasses();
    await Promise.all(
      this._sortedPasses.map((p) =>
        p.initialize({
          device: d,
          width: W,
          height: H,
          bglCache: this._bglCache,
          frameResources: this._res,
        }),
      ),
    );
  }

  /**
   * D3.1 — Phase 3: subsystem initialization (PPG).
   *
   * Delegated to {@link PPGCoordinator.initialize}: derives scene-bounds AABB
   * from the BVH, builds a fresh single-cell sTree, allocates PPG GPU storage
   * buffers, and uploads the serialised tree + both UBOs. No-op when ppgEnabled
   * is false. The kernels descend the serialised buffers each frame; the
   * CPU refines + re-uploads on rebuild cycles (Phase 2 follow-up).
   */
  #initSubsystems(
    bvhBuffers: SceneBVHBuffers,
    ppgEnabled: boolean,
    W: number,
    H: number,
    options: Parameters<WalkaroundGPUPipeline['initialize']>[2],
  ): void {
    this._ppg.initialize(
      bvhBuffers,
      this._res,
      W,
      H,
      ppgEnabled,
      this._frameCount,
      options?.ppgMaxSpatialCells,
      options?.ppgMaxDTreeNodesPerCell,
      options?.ppgMixAlpha,
    );
  }

  /** Re-upload emitter data (called on light/panel change).
   *
   * Re-uploads emitter triangles + power CDF + the light-selection tree (the
   * tree's leaf→emitter mapping and powers move with the emitters). The scene
   * BVH is rebuilt via {@link HybridEngine.setScene} / `reset()` when the scene
   * changes.
   */
  updateEmitters(
    bvhBuffers: Pick<
      SceneBVHBuffers,
      'emitters' | 'emitterCdf' | 'emitterAlias' | 'lightTree' | 'lightTreeNodeCount' | 'lightTreeEnabled'
    >,
  ): void {
    this._bvhHost.updateEmitters(this._device, bvhBuffers);
    // ReGIR: the light-tree node count may have changed (emitters moved), which
    // shifts the grid region's float offset within the combined buffer. Refresh
    // the coordinator so the next frame's UBO carries the right offset (and drop
    // ReGIR if the tree went degenerate). No-op when ReGIR is off.
    this._regir.refreshAfterEmitterRebuild(bvhBuffers);
  }

  /**
   * H41 — Re-upload the analytic point/spot lights buffer when the scene's
   * emitters change (setScene / updateEmitter). Called by HybridEngine
   * after each emitter rebuild, in both the init path and the fast-update path.
   * No-op before `initialize` (safe to call early; `_bvhHost.initialized`
   * guards the upload).
   */
  updateAnalyticLights(scene: Scene): void {
    if (!this._bvhHost.initialized) return;
    this._bvhHost.updateAnalyticLights(this._device, scene);
  }

  /**
   * Stage the complete pipeline-owned emitter-lighting publication.
   *
   * Candidate emitter/CDF/light-tree, camera-visible emissive, and analytic
   * resources remain private until commit. ReGIR is prepared against the same
   * candidate BVH, and the accumulator reset is reversible until the outer
   * engine transaction finalizes. No queue submission occurs here.
   */
  prepareEmitterLightingMutation(
    nextBvh: SceneBVHBuffers,
    nextRenderScene: Scene,
  ): PreparedSceneMutation {
    if (!this._initialized) {
      return {
        commit: () => undefined,
        rollback: () => undefined,
        finalize: () => undefined,
      };
    }

    const previousAccumFrame = this._accumFrameIndex;
    const previousGrisEpoch = this._grisHistoryEpoch;
    const previousGrisClearPending = this._grisHistoryClearPending;
    let stateCommitted = false;
    const accumulatorMutation: PreparedSceneMutation = {
      commit: () => {
        if (stateCommitted) return;
        this._accumFrameIndex = 0;
        this._invalidateGrisHistory();
        stateCommitted = true;
      },
      rollback: () => {
        if (!stateCommitted) return;
        this._accumFrameIndex = previousAccumFrame;
        this._grisHistoryEpoch = previousGrisEpoch;
        this._grisHistoryClearPending = previousGrisClearPending;
        stateCommitted = false;
      },
      finalize: () => undefined,
    };
    const prepared = prepareSceneMutations([
      () => this._bvhHost.prepareEmitterLightingReplacement(
        this._device,
        nextBvh,
        nextRenderScene,
      ),
      () => this._regir.prepareForSceneBvh(nextBvh),
      () => accumulatorMutation,
    ]);
    let published = false;
    let closed = false;
    return {
      commit: () => {
        if (closed || published) return;
        let committed = 0;
        try {
          for (; committed < prepared.length; committed += 1) {
            prepared[committed]!.commit();
          }
          published = true;
        } catch (error) {
          const rollbacks = [...prepared]
            .reverse()
            .map((participant): SceneMutationCleanup => () => participant.rollback());
          closed = true;
          rethrowWithSceneMutationCleanup(
            error,
            rollbacks,
            'emitter-lighting publication failed and rollback also failed',
          );
        }
      },
      rollback: () => {
        if (closed) return;
        closed = true;
        runSceneMutationCleanups(
          [...prepared].reverse().map(
            (participant) => () => participant.rollback(),
          ),
          'emitter-lighting rollback failed',
        );
      },
      finalize: () => {
        if (closed) return;
        closed = true;
        runSceneMutationCleanups(
          // BvhBufferHost is the resource provider; ReGIR and accumulator state
          // are consumers. Retire consumers before the provider's old buffers.
          [...prepared].reverse().map(
            (participant) => () => participant.finalize(),
          ),
          'emitter-lighting retirement failed',
        );
      },
    };
  }

  /**
   * B3 — swap the directional IBL resources (scene-group bindings 15-19). The
   * scene bind group is rebuilt every frame from `sceneBindGroupResources()`,
   * so the next frame picks up the new env textures with no explicit bind-group
   * invalidation. `data == null` resets to the no-HDRI placeholder (hasEnv=0 →
   * the WGSL scalar-sky fallback). No-op before initialize (the placeholder is
   * created in `uploadInitial`; an early call lazily creates it).
   */
  updateDirectionalEnvironment(
    data: import('../environment/equirectDirectional.js').DirectionalEnvData | null,
    rotationY: number,
    intensity: number,
  ): void {
    if (!this._bvhHost.initialized) return;
    this._bvhHost.updateEnvironment(this._device, data, rotationY, intensity);
    this.requestAccumReset();
  }

  /**
   * BVH-refit fast path — overwrite the bvhNodes + bvhPositions GPU
   * buffers in place via `device.queue.writeBuffer`. The buffer handles,
   * sizes, and bind groups are preserved (no pipeline rebind, no bind-
   * group invalidation).
   *
   * Used by `HybridEngine.updatePrimitive`'s transform-only fast path
   * after CPU-side refit. Caller passes the already-refit BVH node bytes
   * and the affected position slice (byte-offset relative to the start
   * of `bvhPositions`).
   *
   * `bvhNodesBytes` is uploaded whole because parent bounds bubble
   * upward; even a single-mesh transform can dirty the root's AABB.
   * `positions` is uploaded byte-by-byte using `byteOffset` + `data` so
   * a small primitive only pays for its own slice.
   */
  /**
   * Prepare all pipeline-owned pieces of an incremental scene mutation.
   *
   * The returned participant must be the final participant committed by the
   * whole-engine coordinator: its commit publishes reversible pointer/CPU
   * state first and calls queue.submit exactly once as the final irreversible
   * operation. No fallible work is performed after an accepted submit.
   */
  prepareSceneMutation(
    mutation: CollectedBvhMutation,
    nextBvh: SceneBVHBuffers,
    prefixCommandBuffers: readonly GPUCommandBuffer[] = [],
  ): PreparedSceneMutation {
    if (!this._initialized) {
      return { commit: () => undefined, rollback: () => undefined, finalize: () => undefined };
    }
    const encoder = this._device.createCommandEncoder({ label: 'scene-mutation-transaction' });
    const geometryChanged =
      mutation.nodes != null ||
      mutation.tlas != null ||
      mutation.replacement != null;
    const previousLearningShadow = this._learningBvhPositionsCpuData;
    const positionSlices = [
      ...(mutation.positions ?? []),
      ...(mutation.learningPositions ?? []),
    ];
    if (mutation.replacement == null && positionSlices.length > 0) {
      if (previousLearningShadow == null) {
        throw new Error('Learning BVH position shadow is not initialized.');
      }
      for (const slice of positionSlices) {
        if (slice.byteOffset < 0 ||
            slice.byteOffset + slice.data.byteLength > previousLearningShadow.byteLength) {
          throw new RangeError('Learning BVH position slice is outside the live shadow.');
        }
      }
    }
    const previousShadowSlices =
      mutation.replacement == null && previousLearningShadow != null
        ? positionSlices.map((slice) => ({
            byteOffset: slice.byteOffset,
            data: previousLearningShadow.slice(
              slice.byteOffset,
              slice.byteOffset + slice.data.byteLength,
            ),
          }))
        : [];
    const replacementLearningShadow = mutation.replacement
      ? nextBvh.bvhPositions.cpuData.slice(0)
      : null;
    const applyShadowSlices = (
      slices: ReadonlyArray<{ readonly byteOffset: number; readonly data: ArrayBuffer }>,
    ): void => {
      const shadow = this._learningBvhPositionsCpuData;
      if (shadow == null) return;
      const target = new Uint8Array(shadow);
      for (const slice of slices) {
        const source = new Uint8Array(slice.data);
        target.set(source, slice.byteOffset);
      }
    };
    const previousAccumFrame = this._accumFrameIndex;
    const previousGrisEpoch = this._grisHistoryEpoch;
    const previousGrisClearPending = this._grisHistoryClearPending;
    if (mutation.resetAccumulator && this._grisReuseStructural) {
      encoder.clearBuffer(this._res.restirGI.reservoirGiCurrentBuffer);
      encoder.clearBuffer(this._res.restirGI.reservoirGiPreviousBuffer);
      encoder.clearBuffer(this._res.restirGI.reservoirGiSpatialBuffer);
    }
    let stateCommitted = false;
    const stateMutation: PreparedSceneMutation = {
      commit: () => {
        if (stateCommitted) return;
        if (replacementLearningShadow != null) {
          this._learningBvhPositionsCpuData = replacementLearningShadow;
        } else {
          applyShadowSlices(positionSlices);
        }
        if (mutation.resetAccumulator) {
          this._accumFrameIndex = 0;
          this._invalidateGrisHistory();
          this._grisHistoryClearPending = false;
        }
        stateCommitted = true;
      },
      rollback: () => {
        if (!stateCommitted) return;
        if (replacementLearningShadow != null) {
          this._learningBvhPositionsCpuData = previousLearningShadow;
        } else {
          applyShadowSlices(previousShadowSlices);
        }
        this._accumFrameIndex = previousAccumFrame;
        this._grisHistoryEpoch = previousGrisEpoch;
        this._grisHistoryClearPending = previousGrisClearPending;
        stateCommitted = false;
      },
      finalize: () => undefined,
    };

    const factories: Array<() => PreparedSceneMutation> = [
      () => this._bvhHost.prepareMutation(this._device, encoder, mutation),
    ];
    if (geometryChanged) {
      factories.push(
        () => this._ppg.prepareResetForSceneBvh(
          nextBvh,
          this._res,
          this._width,
          this._height,
          encoder,
        ),
      );
      if (this._nrc != null) {
        const aabb = deriveSceneAABBFromBvhPositions(nextBvh);
        factories.push(() => this._nrc!.prepareSceneReset(encoder, aabb.min, aabb.max));
      }
    }
    factories.push(
      () => this._regir.prepareForSceneBvh(nextBvh),
      () => stateMutation,
    );

    const prepared = prepareSceneMutations(factories);
    let submitted = false;
    let closed = false;
    return {
      commit: () => {
        if (closed || submitted) return;
        let committed = 0;
        try {
          for (; committed < prepared.length; committed += 1) {
            prepared[committed]!.commit();
          }
          const commandBuffer = encoder.finish();
          this._device.queue.submit([...prefixCommandBuffers, commandBuffer]);
          submitted = true;
        } catch (error) {
          const rollbacks = [...prepared]
            .reverse()
            .map((participant): SceneMutationCleanup => () => participant.rollback());
          closed = true;
          rethrowWithSceneMutationCleanup(
            error,
            rollbacks,
            'GPU scene publication failed and rollback also failed',
          );
        }
      },
      rollback: () => {
        if (closed) return;
        // An accepted WebGPU submit cannot be synchronously undone. The
        // whole-engine coordinator therefore commits this participant last and
        // never executes a fallible participant after it.
        closed = true;
        if (!submitted) {
          runSceneMutationCleanups(
            [...prepared].reverse().map(
              (participant) => () => participant.rollback(),
            ),
            'GPU scene rollback failed',
          );
        }
      },
      finalize: () => {
        if (closed) return;
        closed = true;
        runSceneMutationCleanups(
          // PPG/NRC/ReGIR and CPU publication state depend on the BVH provider.
          // Reverse retirement keeps old provider resources alive until every
          // dependent has finished its own retirement step.
          [...prepared].reverse().map(
            (participant) => () => participant.finalize(),
          ),
          'GPU scene retirement failed',
        );
      },
    };
  }
  refreshBvhRefit(
    bvhNodesBytes: ArrayBuffer,
    positionsSlice: { byteOffset: number; data: ArrayBuffer },
    bvhNodesByteOffset = 0,
  ): void {
    if (!this._initialized) return;
    this._bvhHost.refreshBvhRefit(
      this._device,
      bvhNodesBytes,
      positionsSlice,
      bvhNodesByteOffset,
    );
    this.#patchLearningBvhPositions(positionsSlice);
  }

  /** PR-7 — upload refit BVH nodes only (positions already on GPU). */
  refreshBvhNodesOnly(
    bvhNodesBytes: ArrayBuffer,
    bvhNodesByteOffset = 0,
  ): void {
    if (!this._initialized) return;
    this._bvhHost.refreshBvhNodesOnly(
      this._device,
      bvhNodesBytes,
      bvhNodesByteOffset,
    );
    this.#resetLearnedSceneStateFromShadow();
  }

  /** H19 — upload a per-vertex normals slice after a transform/positions refit. */
  refreshBvhNormalsSlice(normalsSlice: { byteOffset: number; data: ArrayBuffer }): void {
    if (!this._initialized) return;
    this._bvhHost.refreshBvhNormalsSlice(this._device, normalsSlice);
  }

  /** Live merged vertex buffer for GPU skinning writes. */
  getBvhPositionBuffer(): GPUBuffer | null {
    return this._initialized ? this._bvhHost.getBvhPositionBuffer() : null;
  }

  getBvhPositionBinding(): GPUBufferBinding | null {
    return this._initialized ? this._bvhHost.getBvhPositionBinding() : null;
  }

  /** Shared rect-area emitter buffer + tri count for RC emitter NEE. Null
   *  before init. Emitters are world-space triangles, so the same buffer the
   *  shade/ReSTIR-DI path uses is valid for the RC probe cast — no re-upload. */
  getEmitterBufferAndCount(): (GPUBufferBinding & { count: number }) | null {
    return this._initialized ? this._bvhHost.emitterBufferAndCount() : null;
  }

  getEmitterSamplingBufferAndCount(): (GPUBufferBinding & {
    count: number;
    emitterDataOffset: number;
    emitterAliasOffset: number;
  }) | null {
    return this._initialized ? this._bvhHost.emitterSamplingBufferAndCount() : null;
  }

  /** A7 (2026-06-10): equirectangular env map texture view + sampler for RC
   *  probe-ray env sampling. Null before init. The view is always available
   *  after init (a 1×1 black placeholder backs it until updateEnvironment
   *  is called with a real HDRI). Forward both to `RCSubsystem.dispatchFrame`
   *  so the last-cascade env sample reads the scene environment. */
  getEnvBindings(): {
    textureView: GPUTextureView;
    sampler: GPUSampler;
    rotationY: number;
    intensity: number;
    hasDirectionalEnvironment: boolean;
  } | null {
    return this._initialized ? this._bvhHost.envBindings() : null;
  }

  /** Material atlas views for RC material-backed emitter NEE. Null before init. */
  getMaterialAtlasBindings(): {
    materialTextureAtlasView: GPUTextureView;
    materialMapMetaTextureView: GPUTextureView;
    bvhTangentTextureView: GPUTextureView;
    bvhVertexColorTextureView: GPUTextureView;
  } | null {
    return this._initialized ? this._bvhHost.materialAtlasBindings() : null;
  }

  /** WS1 — live merged per-vertex normal buffer for GPU skinning writes. */
  getBvhNormalBuffer(): GPUBuffer | null {
    return this._initialized ? this._bvhHost.getBvhNormalBuffer() : null;
  }

  getBvhNormalBinding(): GPUBufferBinding | null {
    return this._initialized ? this._bvhHost.getBvhNormalBinding() : null;
  }

  /** Live canonical scene-arena ranges consumed directly by RC. */
  getSceneGeometryBufferBindings(): import('./BvhBufferHost.js').SceneGeometryBufferBindings | null {
    return this._initialized ? this._bvhHost.sceneGeometryBufferBindings() : null;
  }

  /** PR-4 — upload refit TLAS nodes + instance transforms (topology unchanged). */
  refreshTlasRefit(
    mutation: import('./BvhUpdateSink.js').TlasRefitMutation,
  ): void {
    if (!this._initialized) return;
    this._bvhHost.refreshTlasRefit(this._device, mutation);
    this.#resetLearnedSceneStateFromShadow();
  }

  /**
   * Material-only fast path — partial upload of the packed `bvhIndex` slice
   * after CPU re-pack (PR-1). WS1: `bvh_beer` is a texture now, so the whole
   * (small) beer texture is re-uploaded from the full beer data + triCount.
   */
  refreshBvhMaterialSlice(
    indexSlice: { byteOffset: number; data: ArrayBuffer },
    beerFull: { data: ArrayBuffer; triCount: number },
    /** Camera-visible emitters — FULL per-tri emissive Le re-upload (same
     *  wholesale rationale as beer). */
    emissiveFull: { data: ArrayBuffer; triCount: number },
    /** B1 — FULL per-tri roughness+metalness re-upload (same wholesale rationale). */
    roughMetalFull?: { data: ArrayBuffer; triCount: number },
  ): void {
    if (!this._initialized) return;
    this._bvhHost.refreshBvhMaterialSlice(
      this._device,
      indexSlice,
      beerFull,
      emissiveFull,
      roughMetalFull,
    );
  }

  refreshBvhEmissiveLe(emissiveFull: { data: ArrayBuffer; triCount: number }): void {
    if (!this._initialized) return;
    this._bvhHost.refreshBvhEmissiveLe(this._device, emissiveFull);
  }

  refreshMaterialTextureAtlas(materialTextureAtlas: MaterialTextureAtlasPayload): void {
    if (!this._initialized) return;
    this._bvhHost.refreshMaterialTextureAtlas(this._device, materialTextureAtlas);
  }

  replaceBvhAndEmitters(bvhBuffers: SceneBVHBuffers): void {
    if (!this._initialized) return;
    const learningPositionsCandidate = bvhBuffers.bvhPositions.cpuData.slice(0);
    this._bvhHost.replaceBvhAndEmitters(this._device, bvhBuffers);
    this._learningBvhPositionsCpuData = learningPositionsCandidate;
    this.#resetLearnedSceneStateFromShadow();
    this._regir.refreshAfterEmitterRebuild(bvhBuffers);
  }

  /**
   * Full BVH-buffer reupload — allocate candidates, then atomically replace BVH GPU
   * buffers/textures (nodes, index, beer, positions, normals, tangents, colors)
   * from a freshly-built
   * `SceneBVHBuffers`. Used by `HybridEngine.updatePrimitive`'s topology-
   * change path after a `buildReSTIRSceneBVH` rebuild. Emitter buffers
   * are NOT touched here — call `updateEmitters` separately if the
   * emitter list also changed.
   *
   * The pipeline shaders and bind-group layouts stay intact because
   * `buildSceneBindGroup` is re-invoked per-frame in `renderFrame()`
   * from the live buffer handles, so the committed replacement is picked
   * up automatically next frame.
   */
  refreshBvhFullRebuild(
    bvhBuffers: Pick<
      SceneBVHBuffers,
      | 'bvhNodes'
      | 'bvhIndex'
      | 'bvhBeerColors'
      | 'bvhEmissiveLe'
      | 'materialTextureAtlas'
      | 'bvhRoughMetal'
      | 'bvhNormals'
      | 'bvhTangents'
      | 'bvhColors'
      | 'bvhPositions'
      | 'bvhMode'
      | 'tlas'
    >,
  ): void {
    if (!this._initialized) return;
    this._bvhHost.refreshBvhFullRebuild(this._device, bvhBuffers);
    this.#replaceLearningBvhPositions(bvhBuffers.bvhPositions.cpuData);
  }

  #replaceLearningBvhPositions(cpuData: ArrayBuffer): void {
    this._learningBvhPositionsCpuData = cpuData.slice(0);
    this.#resetLearnedSceneStateFromShadow();
  }

  #patchLearningBvhPositions(slice: { byteOffset: number; data: ArrayBuffer }): void {
    if (this._learningBvhPositionsCpuData == null) return;
    const target = new Uint8Array(this._learningBvhPositionsCpuData);
    const source = new Uint8Array(slice.data);
    if (slice.byteOffset < 0 || slice.byteOffset + source.byteLength > target.byteLength) return;
    target.set(source, slice.byteOffset);
    this.#resetLearnedSceneStateFromShadow();
  }

  #resetLearnedSceneStateFromShadow(): void {
    const cpuData = this._learningBvhPositionsCpuData;
    if (cpuData == null) return;
    const bvhPositions = { bvhPositions: { cpuData } };
    this._ppg.resetForSceneBvh(bvhPositions, this._res, this._width, this._height);
    const aabb = deriveSceneAABBFromBvhPositions(bvhPositions);
    this._nrc?.resetForSceneBounds(aabb.min, aabb.max);
  }

  /**
   * Resize all per-frame GPU resources to a new render-surface size WITHOUT
   * rebuilding the BVH or recompiling pipelines. It prepares a complete
   * replacement resource set first, publishes it only after every optional
   * subsystem accepts the new dimensions, then retires the old resources.
   * Failure leaves the live size and resources unchanged. A successful commit
   * resets ping-pong indices and frame counters.
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
    // Stage the complete replacement—including optional PPG and denoiser
    // resources—before publishing dimensions or destroying the live frame set.
    const replacement = createFrameResources(this._device, width, height, {
      gtaoDownscale: this._gtaoDownscale,
      svgfEnabled: this._denoiserMode === 'svgf-real',
      grisReuse: this._grisReuseStructural,
      welfordPingPong: this._denoiserMode === 'atrous-variance',
      checkerboard: this._checkerboard,
    });
    try {
      this._ppg.onResize(replacement, width, height, this._frameCount);
      this._activeDenoiser?.resize(width, height);
    } catch (error) {
      destroyFrameResources(replacement);
      throw error;
    }

    const previous = this._res;
    this._res = replacement;
    this._width = width;
    this._height = height;
    this._resourceCache.clear();

    // The new textures are blank, so every temporal index restarts only after
    // the resource transaction has committed.
    this._accumPingPongIndex = 0;
    this._accumFrameIndex = 0;
    this._invalidateGrisHistory();
    this._grisHistoryClearPending = false;
    this._indirectAccumPingPongRef.value = 0;
    this._lastCameraPos = [0, 0, 0];

    destroyFrameResources(previous);
  }

  /** Dev A/B — when false, {@link DenoiserAdapterPass} is gated off (raw HDR). */
  setDenoiserPassEnabled(enabled: boolean): void {
    this._denoiserPassEnabled = enabled;
  }

  isDenoiserPassEnabled(): boolean {
    return this._denoiserPassEnabled;
  }

  /**
   * Runtime PPG train-pass cadence. The pass graph and buffers are unchanged;
   * this only changes the modulo gate checked in {@link renderFrame}. Clamped
   * to >= 1 so a bad host value cannot disable PPG training forever.
   */
  setPpgDispatchInterval(interval: number): void {
    this._ppgDispatchInterval = resolvePpgDispatchInterval(interval);
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
    const compositeUbo = this._compositeUboRef.buf;
    if (compositeUbo == null) return;
    const bgComposite = buildCompositePresentBindGroup(
      d,
      this._bglCache,
      this._res.common.resolvedTexture,
      this._res.common.compositeSampler,
      compositeUbo,
      this._resourceCache,
    );
    const compositePass = this._compositePass;
    if (compositePass == null) return;
    const encoder = d.createCommandEncoder({ label: 'composite-only' });
    const pass = encoder.beginRenderPass({
      label: 'composite-only',
      colorAttachments: [
        {
          view: swapChainView,
          loadOp: 'clear',
          storeOp: 'store',
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
        },
      ],
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
  private _invalidateGrisHistory(): void {
    if (!this._grisReuseStructural) return;
    this._grisHistoryEpoch = (this._grisHistoryEpoch + 1) >>> 0;
    if (this._grisHistoryEpoch === 0) this._grisHistoryEpoch = 1;
    this._grisHistoryClearPending = true;
  }

  requestAccumReset(): void {
    this._accumFrameIndex = 0;
    this._invalidateGrisHistory();
  }

  /**
   * Run one frame of the ReSTIR compute pipeline + composite render pass.
   * Returns true on success, false if pipeline not ready.
   */
  renderFrame(inputs: PipelineFrameInputs): boolean {
    if (!this._initialized) return false;
    const publication = new FramePublicationTransaction();

    try {
      // D3.2 — renderFrame is now a sequencer of three private phases.
      // Exact GPU-command order is preserved across the extraction boundary —
      // the gpu-call-trace goldens pin the sequence.

    // ── Camera motion (computed up-front — shared by UBO packing and the
    //    checkerboard motion fallback).
    //
    // The checkerboard fallback uses its OWN, finer threshold
    // (`_checkerboardMotionThresholdSq`, default 0.004 = 0.063²) — NOT the much
    // coarser temporal-accumulator reset (`_cameraMoveResetThresholdSq`, 1.0):
    // half-rate reservoir lag shows at far smaller motion than a full history
    // discard. The `cbActive`/`parity` record is threaded identically into
    // updateUBO (UBO fields) and passCtx (dispatch compaction) so all three
    // consumers agree every frame.
    const mdx = inputs.camera.cameraPos[0] - this._lastCameraPos[0];
    const mdy = inputs.camera.cameraPos[1] - this._lastCameraPos[1];
    const mdz = inputs.camera.cameraPos[2] - this._lastCameraPos[2];
    const camMoveSqUpfront = mdx * mdx + mdy * mdy + mdz * mdz;
    /** Single checkerboard state record consumed by updateUBO AND #buildFrameContext. */
    const checkerboardState = {
      active: this._checkerboard && !(camMoveSqUpfront > this._checkerboardMotionThresholdSq),
      parity: this._frameCount & 1,
    } as const;

      const { passCtx, gateOpts, passLayout, encoder } = this.#buildFrameContext(
        inputs,
        camMoveSqUpfront,
        checkerboardState,
        publication,
      );

      this.#dispatchPasses(
        passCtx,
        gateOpts,
        passLayout,
        encoder,
        publication,
      );
      const ppgTrainingDispatched =
        gateOpts.ppgEnabled && (gateOpts.ppgTrainThisFrame ?? true);
      this._tickSubsystemTraining(passLayout, ppgTrainingDispatched);

      return true;
    } catch (error) {
      // If submit was accepted, accept() has already closed the transaction and
      // abort is intentionally a no-op. Encode/finish/submit failures release
      // staging resources and leave every persistent history source unchanged.
      publication.abort();
      throw error;
    }
  }

  /**
   * D3.2 — Per-frame Phase 1: UBO update, bind-group assembly, and
   * PassDispatchContext construction.
   *
   * Covers (in exact original order):
   *   - UBO write (updateUBO + NRC camera-pdf update)
   *   - Bind-group builds (frame/scene/ubo/hybridLayers/lightTree)
   *   - passLayout + encoder creation
   *   - workgroup count derivation
   *   - computeDesc / renderTimestampWrites helpers
   *   - isMoving accumulator-reset (mutates `_accumFrameIndex`)
   *   - ping-pong texture resolution
   *   - alpha derivation
   *   - PassFrameState + PassDispatchContext construction
   *   - PassGateOptions construction
   *
   * Returns everything the caller needs to dispatch passes and finalize.
   */
  #buildFrameContext(
    inputs: PipelineFrameInputs,
    camMoveSqUpfront: number,
    checkerboardState: { readonly active: boolean; readonly parity: number },
    publication: FramePublication,
  ): {
    passCtx: PassDispatchContext;
    gateOpts: PassGateOptions;
    passLayout: ReturnType<typeof buildPassLayout>;
    encoder: GPUCommandEncoder;
  } {
    const d = this._device;
    const { _width: W, _height: H } = this;
    const cbActiveThisFrame = checkerboardState.active;

    // ── Update UBO ────────────────────────────────────────────────────────
    // Inject the LIVE PPG gate (PPGCoordinator.enabled is true only when the
    // host opted in AND both PPG compute pipelines compiled) + the MIS mixing
    // weight α. gi-ris reads ppgEnabled/ppgMixAlpha from the UBO to decide
    // whether to guide candidate sampling; when off, α collapses to 0 and the
    // gi-ris RIS source pdf reduces to cosθ/π exactly (ppg-OFF bit-identity).
    // Checkerboard half-res shading state (host opt-in; default OFF, AND forced
    // OFF on a fast-motion frame via cbActiveThisFrame). When OFF both UBO fields
    // pack as 0 ⇒ byte-identical layout and shadeMain/spatialMain shade every
    // pixel. frameParity is `_frameCount & 1` — the SAME phase ResolvePass packs
    // into ResolveUniforms within this frame (it reads `passCtx.checkerboardOn`
    // set below), so the shade/spatial gap-out pixels match the resolve gap-fill
    // pixels exactly.
    updateUBO(
      d,
      this._res.common.uboBuffer,
      inputs,
      {
        enabled: this._ppg.enabled,
        mixAlpha: this._ppg.mixAlpha,
      },
      this._regir.uboState(),
      {
        enabled: cbActiveThisFrame,
        frameParity: checkerboardState.parity,
      },
      this._grisHistoryEpoch,
    );

    // H26 — update the NRC camera pdf every frame so the a0 primary footprint
    // reflects the current projection + internal render resolution.  No-op when
    // NRC is off (_nrc is null in the default pipeline). The write is a cheap
    // 4-byte queue.writeBuffer; it must happen before the gi-ris NRC pass runs.
    this._nrc?.updateCameraPixelPdf(inputs.camera.projMatrix, this._width, this._height);

    // ── Build placeholder texture view ────────────────────────────────────
    const placeholderView = this._resourceCache.textureView(this._res.common.placeholderTexture);

    const {
      frame: bgFrame,
      risGiFrame: bgRisGiFrame,
      scene: bgScene,
      ubo: bgUbo,
      hybridLayers: bgHybrid,
      shadeHybridLayers: bgShadeHybrid,
    } = buildPerFrameBindGroups(
      d,
      this._bglCache,
      this._res,
      this._bvhHost.sceneBindGroupResources(),
      this._ddgi,
      placeholderView,
      this._resourceCache,
      this._nrc?.queryBindings(),
    );

    // RIS-only light-tree bind group (group 3). Always built (a 1-node
    // placeholder backs the buffer when the tree is disabled); the RIS kernel
    // dereferences it only when ubo.lightTreeEnabled == 1.
    const lightTreeBuffer = this._bvhHost.lightTreeBuffer();
    const bgLightTree = this._resourceCache.bindGroup(
      'per-frame:light-tree',
      [lightTreeBuffer],
      () => buildLightTreeBindGroup(d, this._bglCache, lightTreeBuffer),
    );

    // ── Per-frame pre-computed scalars ───────────────────────────────────
    const passLayout = buildPassLayout({
      denoiserMode: this._denoiserMode,
      ...this._passLayoutConfig,
    });

    const encoder = d.createCommandEncoder({ label: 'walkaround-restir' });
    if (this._grisHistoryClearPending) {
      encoder.clearBuffer(this._res.restirGI.reservoirGiCurrentBuffer);
      encoder.clearBuffer(this._res.restirGI.reservoirGiPreviousBuffer);
      encoder.clearBuffer(this._res.restirGI.reservoirGiSpatialBuffer);
    }

    const wgX = Math.ceil(W / 8);
    const wgY = Math.ceil(H / 8);
    const wgX16 = Math.ceil(W / 16);
    const wgY16 = Math.ceil(H / 16);
    const halfWgX = Math.ceil(Math.floor(W / 2) / 8);
    const halfWgY = Math.ceil(Math.floor(H / 2) / 8);
    // Checkerboard sparse-dispatch workgroup counts (ceil-based, NOT the
    // floor-based half-res `halfWgX/halfWgY`): each row has at most ceil(W/2)
    // active-parity pixels compacted into 8-wide workgroups; Y stays full-res
    // (one compacted thread per row). Shared by RIS/Shade/SpatialReservoir when
    // `checkerboardOn` — single source of truth for the compaction math.
    const checkerboardWgX = Math.ceil(Math.ceil(W / 2) / 8);
    const checkerboardWgY = Math.ceil(H / 8);

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
    // The temporal-accumulator reset keeps its OWN (coarser) threshold
    // `_cameraMoveResetThresholdSq` (default 1.0) — a full history discard is a
    // bigger event than the checkerboard full-rate fallback (which trips at the
    // finer `_checkerboardMotionThresholdSq` computed above). Reuses the same
    // `camMoveSqUpfront` delta so there is one motion magnitude, two thresholds.
    const isMoving = camMoveSqUpfront > this._cameraMoveResetThresholdSq;
    const frameAccumIndex = isMoving ? 0 : this._accumFrameIndex;

    // Resolve the temporal-accumulator ping-pong slots for this frame.
    const readAccum =
      this._accumPingPongIndex === 0
        ? this._res.common.accumTextureA
        : this._res.common.accumTextureB;
    const writeAccum =
      this._accumPingPongIndex === 0
        ? this._res.common.accumTextureB
        : this._res.common.accumTextureA;

    const gNormalDepthView = this._resourceCache.textureView(this._res.common.gNormalDepthTexture);

    // alpha=0.01 gives ~99% history weight per frame. Sprint-18-followup
    // tightening: even with the GI W cap + bilinear reservoir blend, the
    // per-pixel reservoir choice changes a few % per frame, and the
    // temporal accumulator's 2% admit at α=0.02 made that change visible
    // as a "dancing" residual noise pattern. Halving α makes each frame's
    // pattern contribution 1% — below the eye's flat-surface detection
    // threshold — and the camera-motion path still forces α=1 on a real
    // move so motion responsiveness is unchanged (just slower to converge
    // back to steady state after a stop).
    const alpha = frameAccumIndex === 0 ? 1.0 : this._temporalAccumAlpha;

    // ── Build the shared per-pass dispatch context ───────────────────────
    const frameState: PassFrameState = {
      denoisedDirect: this._res.common.hdrColorTexture, // overwritten by denoiser dispatch
      indirectAccumOut: this._res.common.indirectAccumPingTexture, // overwritten by indirect-temporal-accum
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
      frameIndex: frameAccumIndex,
      frameCount: this._frameCount,
      publication,
      bglCache: this._bglCache,
      resources: this._res,
      inputs,
      frameBindGroup: bgFrame,
      risGiFrameBindGroup: bgRisGiFrame,
      sceneBindGroup: bgScene,
      uboBindGroup: bgUbo,
      hybridLayersBindGroup: bgHybrid,
      shadeHybridLayersBindGroup: bgShadeHybrid,
      buildTransparentOitBindGroup: (background, output) =>
        this._ddgi.buildTransparentOitBindGroup(
          d,
          this._bglCache,
          this._res,
          background,
          output,
          this._resourceCache,
        ),
      lightTreeBindGroup: bgLightTree,
      wgX,
      wgY,
      wgX16,
      wgY16,
      halfWgX,
      halfWgY,
      checkerboardWgX,
      checkerboardWgY,
      // Checkerboard sparse dispatch state. When ON, ShadePass + SpatialReservoirPass
      // each compact their dispatch to ~half the threads (one per active-parity
      // pixel), and ResolvePass gap-fills the complementary half. This is the
      // SAME `cbActiveThisFrame` (= `_checkerboard && !isMoving` — fast-motion
      // forces full-rate) / frameParity (`frameCount & 1`) updateUBO packs into
      // the WalkaroundUBO above, so the shade/spatial shaders' compacted-gid
      // decode, their UBO reads, and the resolve gap-fill all agree this frame.
      checkerboardOn: cbActiveThisFrame,
      frameParity: checkerboardState.parity,
      // Welford ping-pong state at the START of this frame (before the
      // denoiser's dispatch() flips it). SampleBudgetPass reads this to bind
      // the freshest variance side — ping===0 means varianceBuffer holds the
      // previous frame's write; ping===1 means varianceBufferAux does.
      // For denoisers that don't ping-pong variance, welfordPing is undefined
      // and we fall back to 0 (D3.4 — Denoiser interface property).
      welfordPing: this._activeDenoiser?.welfordPing ?? 0,
      gtaoDownscale: this._gtaoDownscale,
      gNormalDepthView,
      computeDesc,
      renderTimestampWrites,
      frameState,
      // D6 — pass the shared resource cache so per-pass bind groups memoize
      // through the same identity-keyed cache as the frame/scene/ubo groups.
      // Cleared on resize() + dispose(); resource identity changes (scene
      // rebuild, emitter/env update, ping-pong reallocation) auto-invalidate.
      resourceCache: this._resourceCache,
    };

    const gateOpts: PassGateOptions = {
      denoiserMode: this._denoiserMode,
      ppgEnabled: this._ppg.enabled,
      // Phase-0 — PPG train-pass modulo gate. The ppg-update pass dispatches
      // only on multiples of `_ppgDispatchInterval`. interval=1 ⇒ always true
      // (every frame). The persisted tree + gi-ris guided sampling are
      // unaffected — this only skips flux accumulation on off-interval frames.
      // (`_ppgDispatchInterval` is clamped ≥ 1 in initialize().)
      ppgTrainThisFrame:
        this._ppg.trainingDispatchAllowed &&
        this._frameCount % this._ppgDispatchInterval === 0,
      // Phase-0 — gate GTAO + its upsample when the preset disabled it.
      gtaoEnabled: this._gtaoEnabled,
      // Checkerboard pre-denoiser gap-fill gate. Mirrors `cbActiveThisFrame`
      // (= `_checkerboard && !cbMotionExceeded`) so CheckerboardPrefillPass
      // skips on full-rate (non-checkerboard) frames exactly as intended.
      checkerboardOn: cbActiveThisFrame,
    };

    return { passCtx, gateOpts, passLayout, encoder };
  }

  /**
   * D3.2 — Per-frame Phase 2: sorted-pass dispatch + end-of-frame
   * housekeeping (reservoir ping-pong copies, NRC record fold, timestamp
   * resolve, queue submit).
   *
   * Covers (in exact original order):
   *   - Sorted-pass loop with gate filtering
   *   - Accumulator ping-pong advance + lastCameraPos write
   *   - Reservoir current→previous copies (DI + GI)
   *   - NRC self-training-record copy
   *   - Timestamp resolve + queue submit
   *   - Timestamp readback kick + public telemetry mirror
   *
   * The two `copyBufferToBuffer` calls MUST be in the SAME encoder as the
   * pass dispatch — see the B6 race-condition note on the original site.
   */
  #dispatchPasses(
    passCtx: PassDispatchContext,
    gateOpts: PassGateOptions,
    passLayout: ReturnType<typeof buildPassLayout>,
    encoder: GPUCommandEncoder,
    publication: FramePublicationTransaction,
  ): void {
    // ── Unified pass loop ────────────────────────────────────────────────
    // Polymorphic denoiser dispatch is one of the sorted passes
    // ({@link DenoiserAdapterPass}); its dependency on `gtao-upsample` +
    // `indirect-temporal-accum`'s dependency on `denoiser-adapter` place
    // it in the slot the manual two-half split previously bracketed.
    // `frameState.denoisedDirect` is seeded above with the raw HDR
    // target, so when the adapter gates off (NoneDenoiser) downstream
    // passes see the legacy fallback handle.
    //
    // Manual `Pass.gates` filtering inline rather than calling
    // `_passRegistry.activePasses(...)` so we iterate the cached
    // `_sortedPasses` array and avoid re-sorting per frame.
    for (const pass of this._sortedPasses) {
      if (!pass.gates(gateOpts)) continue;
      pass.dispatch(passCtx);
    }

    // ── End-of-frame: swap-chain present sentinel + reservoir housekeeping ─
    // Stage all persistent CPU history state. The selected GPU sources/targets
    // remain unchanged until finish + submit have both succeeded.
    const nextAccumPingPongIndex = 1 - this._accumPingPongIndex;
    const nextAccumFrameIndex = passCtx.frameIndex + 1;
    const nextCameraPos = [...passCtx.inputs.camera.cameraPos] as [
      number,
      number,
      number,
    ];
    const nextFrameCount = passCtx.frameCount + 1;
    publication.stage(() => {
      this._accumPingPongIndex = nextAccumPingPongIndex;
      this._accumFrameIndex = nextAccumFrameIndex;
      this._lastCameraPos = nextCameraPos;
      this._frameCount = nextFrameCount;
      this._grisHistoryClearPending = false;
    });

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
      this._res.restirDI.reservoirCurrentBuffer,
      0,
      this._res.restirDI.reservoirPreviousBuffer,
      0,
      this._res.restirDI.reservoirCurrentBuffer.size,
    );
    encoder.copyBufferToBuffer(
      this._res.restirGI.reservoirGiCurrentBuffer,
      0,
      this._res.restirGI.reservoirGiPreviousBuffer,
      0,
      this._res.restirGI.reservoirGiCurrentBuffer.size,
    );

    // NRC ON ⇒ fold the self-training-record copy into THIS encoder (after the
    // gi-ris pass wrote the records, before submit) so the host gather sees the
    // current frame's records. No-op buffer when NRC is off (`_nrc` null).
    this._nrc?.recordCopyForReadback(encoder, publication);

    // Resolve timestamps + copy into the inactive readback buffer.
    resolveTimestamps(encoder, this._tsState, this._frameCount, passLayout.slotCount);

    this._submitAndRunPostSubmitHooks(
      encoder,
      publication,
      passCtx.frameCount,
      passLayout.labels,
    );
  }

  /**
   * Submit is the irreversible frame boundary. Failures before acceptance are
   * rethrown so the host may retry; failures after acceptance are contained and
   * reported as non-fatal diagnostics so an already-submitted frame is never
   * misreported as retryable. Each post-submit hook is isolated from its peers.
   */
  private _submitAndRunPostSubmitHooks(
    encoder: GPUCommandEncoder,
    publication: FramePublicationTransaction,
    frameCount: number,
    labels: readonly PassLabel[],
  ): void {
    try {
      finishSubmitAndPublishFrame(encoder, this._device.queue, publication);
    } catch (error) {
      if (publication.state !== 'accepted') throw error;
      this._reportAcceptedFrameFailure('frame publication', error);
    }

    try {
      this._activeDenoiser?.afterFrameSubmit?.();
    } catch (error) {
      this._reportAcceptedFrameFailure('denoiser afterFrameSubmit', error);
    }

    try {
      // Pass the labels captured for this frame so an async completion remains
      // correctly attributed even if the pipeline is reconfigured meanwhile.
      kickTimestampReadback(this._tsState, frameCount, labels);
    } catch (error) {
      this._reportAcceptedFrameFailure('timestamp readback kickoff', error);
    }

    this.lastGpuTimings = this._tsState.lastGpuTimings;
    this.lastGpuTimingsFrame = this._tsState.lastGpuTimingsFrame;
  }

  private _reportAcceptedFrameFailure(stage: string, raw: unknown): void {
    if (!this._initialized || this._onError == null) return;
    const detail = raw instanceof Error ? raw.message : String(raw);
    try {
      this._onError({
        kind: 'render',
        message:
          `[WalkaroundGPUPipeline] accepted frame post-submit hook '${stage}' failed; ` +
          `GPU submission remains accepted. ${detail}`,
        fatal: false,
        raw,
      });
    } catch {
      // A host diagnostics callback cannot retroactively fail an accepted frame.
    }
  }

  /**
   * D3.2 — Per-frame Phase 3: post-submit subsystem training ticks.
   *
   * Covers (in exact original order):
   *   - PPG `maybeRunTrainingRefine` (W9 follow-up, fire-and-forget CPU async)
   *   - NRC `trainFromRecords` (Müller §5, fire-and-forget async, one step/frame)
   *
   * Both calls happen AFTER `d.queue.submit()` so they do not block the
   * GPU timeline. PPG is CPU-side; NRC maps a staging buffer async — both
   * are re-entrant-guarded inside their subsystems.
   */
  private _reportNrcTrainingFailure(raw: unknown): void {
    if (!this._initialized || this._onError == null) return;
    const detail = raw instanceof Error ? raw.message : String(raw);
    if (this._lastNrcTrainingErrorMessage === detail) return;
    this._lastNrcTrainingErrorMessage = detail;
    try {
      this._onError({
        kind: 'render',
        message:
          `[WalkaroundGPUPipeline] NRC training transaction failed; the last ` +
          `committed NRC generation remains valid. ${detail}`,
        fatal: false,
        raw,
      });
    } catch {
      // Training is post-submit; host diagnostics must never make it retryable.
    }
  }

  private _tickSubsystemTraining(
    _passLayout: ReturnType<typeof buildPassLayout>,
    ppgTrainingDispatched: boolean,
  ): void {
    // W9 follow-up — periodic training/refine cycle:
    // fluxAtomics GPU readback -> CPU dTree/sTree refinement -> re-upload.
    try {
      this._ppg.maybeRunTrainingRefine(this._res, ppgTrainingDispatched);
    } catch (error) {
      this._reportAcceptedFrameFailure('PPG training/refine', error);
    }

    // NRC ON ⇒ read back this frame's self-training records and run ONE train
    // step (Müller §5 self-training; HOST-OWNS-CADENCE — one step per frame).
    // Fire-and-forget: the readback maps async; a still-pending readback skips
    // this frame and picks up fresh records next frame (re-entrancy guarded in
    // the subsystem). No-op when NRC is off. Rejections are reported as
    // non-fatal diagnostics while the pipeline is live; post-dispose failures
    // remain suppressed because they are expected during teardown/device loss.
    if (this._nrc !== null) {
      let training: Promise<void>;
      try {
        training = this._nrc.trainFromRecords();
      } catch (error) {
        this._reportNrcTrainingFailure(error);
        return;
      }
      void training
        .then(() => {
          this._lastNrcTrainingErrorMessage = null;
        })
        .catch((err: unknown) => {
          this._reportNrcTrainingFailure(err);
        });
    }
  }

  dispose(): void {
    this._bvhHost.dispose();
    this._compositePass = null;
    // D3.3 — delegate capture-resource teardown to the helper (same order as
    // the original inline _captureRenderPipeline=null / _captureOffscreenTex destroy).
    this._captureHelper.dispose();
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
    // Release the configured neural graph after the wrapper has stopped dispatching.
    this._inferenceGraph?.dispose();
    this._inferenceGraph = null;
    // Per-feature state objects (PPG / DDGI binding / ReGIR). None own
    // destroy()-able GPU buffers of their own — PPG buffers live in
    // FrameResources.ppg (released above); DDGI atlases are host-owned;
    // ReGIR grid data lives in BvhBufferHost's combined light-tree buffer
    // (released above via _bvhHost.dispose()). These calls simply drop
    // held references and reset CPU-side geometry mirrors.
    this._ppg.dispose();
    this._ddgi.dispose();
    this._regir.dispose();
    // NRC subsystem (only allocated when nrcEnabled). Releases the @group(4)
    // query buffers; the MLP trainer's buffers go with the device.
    this._nrc?.dispose();
    this._nrc = null;
    this._resourceCache.clear();
    // Guard all post-dispose entry points (renderFrame / presentLastFrame /
    // getDebugTextures / resize) — they all check this flag first.
    this._initialized = false;
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
  setDDGIInputs(
    inputs: {
      irradianceTex: GPUTexture;
      visibilityTex: GPUTexture;
      gridParams: ArrayBuffer;
    } | null,
  ): void {
    this._ddgi.setInputs(inputs, this._res);
  }

  /**
   * W8 Phase 3 (2026-05-18) — bind RC cascade-0 inputs for the shade pass.
   * Called per frame from `HybridEngine.renderFrame` when `rcEnabled`.
   * Pass `null` to revert to placeholder (RC contribution becomes 0).
   *
   * `paramsBytes` is the packed {@link RCParams} struct from
   * `HybridEngineRC.packRCParams(...)` — see that file for the WGSL-aligned
   * 64-byte layout.
   */
  setRCInputs(
    inputs: {
      cascade0Buffer: GPUBuffer;
      paramsBytes: ArrayBuffer;
    } | null,
  ): void {
    this._ddgi.setRCInputs(inputs);
  }
}
