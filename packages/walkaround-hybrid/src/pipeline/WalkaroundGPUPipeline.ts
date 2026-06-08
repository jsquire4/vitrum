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
import type { SceneBVHBuffers } from '../restir/bvhTypes.js';
import type { BvhUpdateSink } from './BvhUpdateSink.js';
import type { PipelineDebugTextures } from './PipelineDebugTextures.js';
import type { InferenceGraph } from '../neural/InferenceGraph.js';
import { updateUBO } from './uboUpdater.js';
import { compilePipelines } from './pipelineCompiler.js';
import { BvhBufferHost } from './BvhBufferHost.js';
import type { GpuMemoryExternalSections } from './gpuMemoryEstimate.js';
import {
  createFrameResources,
  destroyFrameResources,
  type FrameResources,
} from './resourceManager.js';
import {
  type BGLCache,
} from './bindGroupLayouts.js';
import type { UboRef } from './bindGroupBuilders.js';
import { buildLightTreeBindGroup } from './bindGroupBuilders.js';
import {
  buildCompositePresentBindGroup,
  buildPerFrameBindGroups,
} from './pipelineBindGroupFactory.js';
import { PipelineResourceCache } from './PipelineResourceCache.js';
import { PPGCoordinator } from './PPGCoordinator.js';
import { NrcSubsystem } from '../neural/nrc/nrcSubsystem.js';
import { ReGIRCoordinator, resolveReGIRConfig, type ReGIRConfig } from './ReGIRCoordinator.js';
import { DDGIBindingState } from './DDGIBindingState.js';
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
import { RESERVOIR_GI_STRIDE } from '../ppg/ppgConstants.js';
import type { RestirGISnapshot } from '../giStateSnapshot.js';

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
  restirPtReuseStructural: boolean;
  /** Checkerboard half-res shading flag (host opt-in). Threaded into the
   *  ResolvePass ctor; OFF ⇒ passthrough (byte-identity). */
  checkerboard: boolean;
  sampleBudgetUboRef: UboRef;
  sampleCountUboRef: UboRef;
  accumUboRef: UboRef;
  resolveUboRef: UboRef;
  atrousIndirectUboRef: UboRef;
  indirectAccumPingPongRef: PingPongRef;
  regir: ReGIRCoordinator;
  bglCache: BGLCache;
  bvhBuffers: SceneBVHBuffers;
  /** @group(4) NRC bind-group getter, or `undefined` when NRC is off. */
  nrcBindGroup: (() => GPUBindGroup) | undefined;
  getActiveDenoiser: () => Denoiser;
  getAtrousPipeline: () => GPUComputePipeline;
  isDenoiserPassEnabled: () => boolean;
  getRegirResources: () => {
    combinedLightTreeBuffer: GPUBuffer;
    emitterBuffer: GPUBuffer;
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
  registry.register(new SampleBudgetPass(
    compiled.sampleBudgetPipeline,
    deps.sampleBudgetUboRef,
    deps.sampleCountUboRef,
  ));
  registry.register(new RISPass(compiled.risPipeline));
  registry.register(new TemporalReservoirPass(compiled.temporalPipeline));
  // Phase-0 — spatial pass count is preset-driven (1 or 2 ping-pong passes).
  registry.register(new SpatialReservoirPass(compiled.spatialPipeline, deps.diSpatialPasses));
  // NRC ON ⇒ supply the @group(4) bind group getter so the gi-ris pass binds
  // slot 4 (the inline-MLP variant was compiled). OFF ⇒ no getter, verbatim
  // 4-group dispatch (the default-path structure is unchanged).
  registry.register(new RISGIPass(
    compiled.risGiPipeline,
    deps.nrcBindGroup,
  ));
  registry.register(new TemporalGIReservoirPass(compiled.temporalGiPipeline, deps.restirPtReuseStructural));
  registry.register(new SpatialGIReservoirPass(compiled.spatialGiPipeline, deps.giSpatialPasses, deps.restirPtReuseStructural));
  registry.register(new ShadePass(compiled.shadePipeline));
  registry.register(new MotionVectorsPass(compiled.motionVectorsPipeline));
  registry.register(new GTAOPass(compiled.gtaoPipeline));
  registry.register(new GTAOUpsamplePass(compiled.gtaoUpsamplePipeline));
  // Virtual pass — promotes the polymorphic denoiser dispatch into the
  // regular pass loop. Reads the active Denoiser through a getter so
  // the adapter stays valid across `_activeDenoiser` reassignment in
  // `dispose()` (where it is set to null AFTER the pass-list dispose
  // walk, so the getter never sees the null transition).
  registry.register(new DenoiserAdapterPass(
    deps.getActiveDenoiser,
    deps.getAtrousPipeline,
    deps.isDenoiserPassEnabled,
  ));
  registry.register(new IndirectTemporalAccumPass(
    compiled.indirectTemporalAccumPipeline,
    deps.indirectAccumPingPongRef,
  ));
  registry.register(new AtrousIndirectPass(
    compiled.atrousPipeline,
    deps.atrousIndirectUboRef,
  ));
  registry.register(new IndirectCombinePass(compiled.indirectCombinePipeline));
  registry.register(new TemporalAccumPass(compiled.accumPipeline, deps.accumUboRef));
  registry.register(new ResolvePass(compiled.resolvePipeline, deps.resolveUboRef, deps.checkerboard));
  const compositePass = new CompositePass(compiled.compositePipeline);
  registry.register(compositePass);
  // PPG update pass — only register when the pipeline compiled successfully.
  // The `gates()` predicate gates dispatch on `opts.ppgEnabled` so they
  // can be registered unconditionally here, but skipping registration
  // when the pipeline is undefined avoids holding a stale field.
  if (compiled.ppgUpdatePipeline) {
    registry.register(new PPGUpdatePass(compiled.ppgUpdatePipeline));
  }
  // ReGIR grid-build (Boksansky 2021) — register only when the pipeline
  // compiled (opt-in). Topo-sort runs it FIRST (no deps; `regir-build` <
  // `sample-budget` lexically) so the grid is filled before RIS reads it.
  // `gates()` further requires the coordinator be live (light tree live).
  // Initialise the coordinator's grid geometry first so `live` is correct.
  deps.regir.initialize(deps.bvhBuffers, compiled.regirBuildPipeline !== undefined);
  if (compiled.regirBuildPipeline) {
    registry.register(new ReGIRBuildPass(
      compiled.regirBuildPipeline,
      deps.regir,
      deps.bglCache,
      deps.getRegirResources,
    ));
  }
  return { registry, compositePass };
}

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
 * Phase-0 productization — reduced device limits for the hybrid **lite** tier
 * (Class B/C adapters that cannot satisfy the full 16-buffer / 8-texture
 * floor but can still run a degraded realtime path).
 *
 * The lite tier runs the SAME shade pipeline as full — there is no WGSL fork
 * (Deliverable 3 decision: runtime UBO/pass gating over an N× pipeline
 * permutation). The win is on the **storage-buffer axis**: lite forces the
 * merged-BVH path (`bvhMode:'merged'`), which removes the 5 TLAS scene-group
 * buffers, dropping the peak storage-buffer count from the full path's 16 to
 * the merged path's ~10. The **texture** floor stays at 5 because the shade
 * pass structurally writes 4 storage textures simultaneously + 1 — that cannot
 * drop without forking shade.wgsl, which the lite-tier decision explicitly
 * avoids.
 *
 * NOTE: these numbers are a *hypothesis* until a device requested with these
 * limits actually compiles + binds the merged-path shade pipeline on real
 * hardware (Risk R3 in `plan/phase0-productization.md`). The full 16/8 were
 * themselves "lift for headroom" choices, not hard minima, so the lite values
 * are a conservative reduction. The GPU-validation harness confirms them; see
 * the HARDWARE-VALIDATION lite-tier entry. Both axes MUST stay strictly below
 * `HYBRID_WEBGPU_REQUIRED_LIMITS` (asserted by a unit test) so the adapter
 * profile's lite-vs-full verdict is monotone.
 */
export const HYBRID_LITE_LIMITS: Record<string, number> = {
  // Merged-path peak (no 5 TLAS scene-group buffers vs the full 16).
  maxStorageBuffersPerShaderStage: 10,
  // Shade's 4 simultaneous storage-texture writes + 1; cannot go lower
  // without a shade.wgsl fork (the lite decision avoids that fork).
  maxStorageTexturesPerShaderStage: 5,
};

/**
 * Extra device limits required ONLY when `nrcEnabled` (opt-in NRC). These exceed
 * the WebGPU defaults (maxBindGroups 4, maxComputeWorkgroupStorageSize 16384), so
 * a host that opts into NRC must create the device with these in `requiredLimits`.
 * GPU validation (2026-05-29, V20) confirmed dzn/RTX-4090 exposes 8 / 32768 and
 * renders NRC-ON. The pipeline throws a clear error at init if the device is
 * under-spec (host-owns-lifecycle), rather than failing cryptically in
 * createComputePipeline. NRC is full-tier only — these are additive to
 * {@link HYBRID_WEBGPU_REQUIRED_LIMITS}, never applicable on the lite tier.
 */
export const NRC_REQUIRED_MAX_BIND_GROUPS = 5; // the @group(4) NRC bind group
export const NRC_REQUIRED_WORKGROUP_STORAGE_BYTES = 24576; // fused-MLP workgroup tiles

/**
 * Throw a clear, host-actionable error if a device cannot satisfy NRC's extra
 * limits. Called at pipeline init when `nrcEnabled` (host-owns-lifecycle): the
 * host owns device creation, so the remedy is to add these to `requiredLimits`.
 * Mirrors the lite-tier forbid — fail early + legibly, not cryptically inside
 * createComputePipeline.
 */
export function assertNrcDeviceCapable(limits: GPUSupportedLimits): void {
  const maxBindGroups = limits.maxBindGroups ?? 4;
  const maxWgStorage = limits.maxComputeWorkgroupStorageSize ?? 16384;
  if (maxBindGroups < NRC_REQUIRED_MAX_BIND_GROUPS
    || maxWgStorage < NRC_REQUIRED_WORKGROUP_STORAGE_BYTES) {
    throw new TypeError(
      `[HybridEngine] nrcEnabled requires a device created with `
      + `maxBindGroups >= ${NRC_REQUIRED_MAX_BIND_GROUPS} (the @group(4) NRC group; `
      + `default is 4) and maxComputeWorkgroupStorageSize >= `
      + `${NRC_REQUIRED_WORKGROUP_STORAGE_BYTES} (the fused-MLP workgroup tiles; `
      + `default is 16384), but this device reports maxBindGroups=${maxBindGroups}, `
      + `maxComputeWorkgroupStorageSize=${maxWgStorage}. The host owns device `
      + `creation — request these in requiredLimits (real full-tier GPUs expose `
      + `them, e.g. 8 / 32768), or omit nrcEnabled.`,
    );
  }
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

/** Camera matrices + position for one frame. */
export interface PipelineFrameCamera {
  /** Camera view matrix (column-major mat4x4f, 16 floats). The pipeline
   *  composes VP = projMatrix * viewMatrix internally; do NOT pre-multiply. */
  viewMatrix: Float32Array;
  /** Camera projection matrix (column-major mat4x4f, 16 floats). */
  projMatrix: Float32Array;
  /** Previous-frame view matrix — drives temporal reservoir reuse. Pass
   *  the same matrix as viewMatrix on the first frame to avoid a one-frame
   *  ghost from uninitialized previous-frame state. */
  prevViewMatrix: Float32Array;
  /** World-space camera position [x, y, z]. */
  cameraPos: [number, number, number];
}

/** Swap-chain + frame-seed for one frame. */
export interface PipelineFrameScreen {
  /** Render-target dimensions in pixels. Used by all compute kernels for
   *  workgroup dispatch sizing — must match the swap chain's actual size. */
  screenWidth: number;
  screenHeight: number;
  /** u32 frame counter / per-frame randomness seed. Drives PCG hash inits
   *  for ray jitter, RIS candidate sampling, and temporal reservoir update.
   *  Caller may use a frame index, performance.now()|0, or any monotone u32. */
  frameSeed: number;
  /** The WebGPU swap-chain texture view to render into for this frame.
   *  Caller must obtain via context.getCurrentTexture().createView()
   *  inside the same animation-frame callback that calls renderFrame. */
  swapChainView: GPUTextureView;
  /** The format of swapChainView. The composite pass's render-pipeline
   *  is recompiled if this changes (rare — usually fixed at canvas mount). */
  swapChainFormat: GPUTextureFormat;
}

/** Lighting scalars, emitter budget, and light-tree configuration. */
export interface PipelineFrameLighting {
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
  /** Light-tree gate for ReSTIR-DI initial-candidate light SELECTION (UBO
   *  offset 356). `1` ⇒ the RIS candidate loop draws lights via the
   *  spatially-aware light tree (`sampleLightTree`) and divides the WRS weight
   *  by the tree selection pdf; `0` ⇒ RIS uses the flat power-CDF path
   *  (`sampleEmitterIdx` + the `emitterPmf` weight) verbatim. Built `1` only
   *  when the tree has ≥ 2 emitters AND the host left light-tree selection on;
   *  see `SceneBVHBuffers.lightTreeEnabled`. The estimator is unbiased in BOTH
   *  states because the WRS weight always divides p̂ by the EXACT pdf the
   *  selection used. */
  lightTreeEnabled: number;
  /** Number of nodes in the packed light tree (UBO offset 360). Bounds the
   *  `sampleLightTree` descent loop. `0` when the tree is disabled. */
  lightTreeNodeCount: number;
}

/** ReSTIR-DI temporal + spatial reuse tuning knobs. */
export interface PipelineFrameRestirDI {
  /** Audit M6 — ReSTIR-DI temporal M-clamp; Cornell default 20. */
  temporalMClampDI: number;
  /** Audit M7 — ReSTIR-DI spatial reuse radius in pixels; Cornell default 30. */
  spatialReuseRadiusPx: number;
  /** Audit M8 — ReSTIR-DI spatial depth-tolerance world-units floor; Cornell
   *  default 0.05 (5 cm). Hosts on different scales should pass
   *  `sceneDiagonal * 1e-3`. */
  spatialDepthTolFloor: number;
}

/** ReSTIR-GI / GRIS tuning + reuse gate. */
export interface PipelineFrameRestirGI {
  /** 2026-05-18 sweep — ReSTIR-GI per-pixel unbiased weight cap (risGi,
   *  spatialGi). Cornell default 16.0. */
  restirGiWCap: number;
  /** 2026-05-18 sweep — DDGI irradiance clamp applied at the ReSTIR-GI
   *  reconnection vertex (risGi). Cornell default 5.0. */
  restirGiIrrClamp: number;
  /** 2026-05-18 sweep — ReSTIR-GI temporal previous-frame M clamp
   *  (temporalGi). Cornell default 50. */
  restirGiMClamp: number;
  /** 2026-05-18 sweep — ReSTIR-GI spatial-reuse disc radius (half-res
   *  pixels). Cornell default 12.0. */
  restirGiSpatialRadiusPx: number;
  /** 2026-05-18 sweep — ReSTIR-GI spatial-reuse normal-alignment cosine
   *  minimum (spatialGi). Cornell default 0.906 ≈ cos(25°). */
  restirGiSpatialNormalDotMin: number;
  /** 2026-05-18 sweep — ReSTIR-GI spatial-reuse tangent-plane distance
   *  tolerance (spatialGi). Cornell default 0.05 (5 cm world units). */
  restirGiSpatialCoplanarTol: number;
  /** GRIS / ReSTIR-PT reconnection-shift reuse gate (UBO offset 412). `1` ⇒
   *  the GI spatial + temporal reuse passes apply the unbiased GRIS
   *  reconnection shift, its change-of-variables Jacobian, a reconnection-
   *  visibility ray, and the pairwise generalized-balance MIS (Lin et al.
   *  2022). `0`/omitted ⇒ the reuse runs the legacy clamped-Jacobian path
   *  BIT-FOR-BIT (the GRIS branch is gated behind `ubo.restirPtReuse == 1`).
   *  Host opt-in via HybridEngineOptions.restirPtReuse — the same
   *  OFF-is-bit-identical pattern as RC/PPG/ReGIR. */
  restirPtReuse?: number;
}

/** GTAO + adaptive-sampling tuning knobs. */
export interface PipelineFrameGtao {
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
}

/** Denoiser filter parameters, firefly clamps, and stained-glass gate. */
export interface PipelineFrameFilter {
  /** D12 — Möller-Trumbore coplanarity epsilon.  Controls the `abs(det) < ε`
   *  near-zero determinant threshold in `intersectTriangle`.  Default `1e-5`
   *  (metre-scale).  Reduce for millimetre-scale geometry. */
  triIntersectEpsilon: number;
  /** 2026-05-18 sweep — probe-side glass-transmission perceptual mix scale.
   *  Cornell default 0.7. */
  glassMixScale: number;
  /** 2026-05-18 sweep — per-channel HDR clamp on the indirect channel
   *  (shade.wgsl). Cornell default [1.0, 1.0, 1.0]. */
  indirectFireflyClamp: readonly [number, number, number];
  /** 2026-05-19 B3a — atrous DIRECT-channel sigmas [sigmaN, sigmaZ, sigmaC].
   *  Cornell default `[128.0, 5.0, 0.05]`. Consumed by the AtrousDenoiser
   *  direct-path chain. */
  atrousDirectSigmas: readonly [number, number, number];
  /** 2026-05-19 B3a — atrous INDIRECT-channel sigmas [sigmaN, sigmaZ, sigmaC].
   *  Cornell default `[32.0, 20.0, 0.5]`. Consumed by AtrousIndirectPass. */
  atrousIndirectSigmas: readonly [number, number, number];
  /** T5 — stained-glass opt-in flag bitfield (lands at UBO offset 344).
   *  Bit 0 = sun-caustic enabled, bit 1 = sky-aperture enabled. Default 0
   *  (both OFF) → lo_sg_caustic / lo_sg_aperture early-return vec3f(0), so a
   *  generic scene gets zero stained-glass physics. Hosts opt in via
   *  HybridEngineOptions.stainedGlass. See pipeline/uboUpdater.ts
   *  `packStainedGlassFlags`. */
  stainedGlassFlags: number;
}

/** BVH traversal mode + TLAS configuration. */
export interface PipelineFrameBvh {
  /** PR-3 — 0 = merged world BVH, 1 = TLAS + local BLAS traversal. */
  bvhMode: number;
  /** PR-3 — TLAS node count from CPU pack (0 forces merged path in WGSL). */
  tlasNodeCount: number;
}

/** Optional per-frame NRC gate. */
export interface PipelineFrameNrc {
  /** NRC (Müller et al. 2021) cache gate (UBO offset 364 — the former
   *  `_ppgPad2` slot). `1` ⇒ the GI suffix may TERMINATE into the learned
   *  neural radiance cache (spread heuristic + MLP query) and radiance records
   *  self-train it. `0`/omitted ⇒ the gi-ris suffix runs the verbatim
   *  DDGI-atlas estimate and the UBO bytes are unchanged — **OFF is
   *  bit-identical**. Host opt-in via HybridEngineOptions.nrcEnabled; FORBIDDEN
   *  on tier:'lite'. NRC is a BIASED cache (not a converged-mean-preserving
   *  reuse) — see HARDWARE-VALIDATION-NEEDS.md V20. WIRED (2026-05-29): the gi-ris
   *  NRC variant runs the MLP query + writes self-training records;
   *  `NrcSubsystem.trainFromRecords` runs an MLP `trainStep` AND the hash-grid
   *  encode-backward + table Adam each frame — so with the gate at 1 the suffix
   *  uses the (biased) learned MLP prediction when the spread heuristic fires. */
  nrcEnabled?: number;
}

/**
 * Per-frame inputs to {@link WalkaroundGPUPipeline.renderFrame}.
 *
 * Fields are grouped into named sub-objects so each sprint's new field
 * lands in the right semantic bucket rather than growing a flat list.
 * UBO byte layout is unchanged — the sub-objects are TypeScript-only;
 * `uboUpdater.ts` unpacks them field-by-field as before.
 */
export interface PipelineFrameInputs {
  /** Camera matrices and world-space position. */
  camera: PipelineFrameCamera;
  /** Swap-chain targets and per-frame seed. */
  screen: PipelineFrameScreen;
  /** Lighting scalars, emitter budget, and light-tree gate. */
  lighting: PipelineFrameLighting;
  /** ReSTIR-DI temporal + spatial reuse tuning. */
  restirDI: PipelineFrameRestirDI;
  /** ReSTIR-GI / GRIS tuning and reconnection gate. */
  restirGI: PipelineFrameRestirGI;
  /** GTAO + adaptive-sampling tuning. */
  gtao: PipelineFrameGtao;
  /** Denoiser filter parameters, firefly clamps, and stained-glass gate. */
  filter: PipelineFrameFilter;
  /** BVH traversal mode and TLAS configuration. */
  bvh: PipelineFrameBvh;
  /** NRC cache gate (optional; absent ⇒ OFF, bit-identical). */
  nrc: PipelineFrameNrc;
}

export class WalkaroundGPUPipeline implements BvhUpdateSink {
  // Private fields use the `_field` underscore prefix, matching HybridEngine.
  private _device: GPUDevice;
  private _width: number;
  private _height: number;

  /** Static BVH + TLAS + emitter GPU buffers (W4b — BvhBufferHost). */
  private readonly _bvhHost = new BvhBufferHost();
  /** Cached composite pass — avoids registry lookup in presentLastFrame. */
  private _compositePass: CompositePass | null = null;

  // Per-frame GPU resources (created by resourceManager.createFrameResources)
  private _res!: FrameResources;

  // Temporal accumulator ping-pong state
  private _accumPingPongIndex = 0;       // 0 = read A, write B; 1 = swap
  private _accumFrameIndex = 0;
  private _lastCameraPos: [number, number, number] = [0, 0, 0];

  // DDGI atlas binding state (layered hybrid). Owns the optional host-
  // supplied irradiance + visibility atlases and the cached placeholder UBO
  // for the no-DDGI fallback. Constructed once in the pipeline ctor; the
  // device handle is captured at that point so renderFrame / setDDGIInputs
  // can stay device-agnostic on this side.
  private readonly _ddgi: DDGIBindingState;

  // PPG (Müller 2017) state — enabled flag, scene-bounds AABB, CPU sTree,
  // and the three serialise/upload writers used to be loose private members
  // (`_ppgEnabled`, `_ppgSTree`, `_ppgSceneAABB`, plus three methods).
  // Concentrated here so the orchestrator can stay focused on pass
  // scheduling. T2.H3 — `PPGCoordinator.enabled` mirrors the gate forwarded
  // into `PassGateOptions.ppgEnabled`.
  private readonly _ppg: PPGCoordinator;

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
  /** GRIS / ReSTIR-PT reconnection-shift reuse (restirPtReuse). COMPILE-TIME
   *  gate: when true, the GI spatial + temporal pipelines are built with the
   *  two-group layout + GRIS shader and the passes bind the scene group at
   *  @group(1). When false (default) the GI passes are the verbatim Sprint-17
   *  single-group pipeline (the known-good default). Resolved once in
   *  initialize() from the host flag — NOT a per-frame UBO decision. */
  private _restirPtReuseStructural = false;
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
  /** T2.H2 — neural denoiser InferenceGraph; kept for future W10 wiring. */
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
    return { halfW, halfH, strideU32: RESERVOIR_GI_STRIDE, current, previous, spatial };
  }

  /**
   * Upload previously-exported ReSTIR-GI reservoir buffers into the live
   * reservoirs (the reservoir half of the restore). Returns false (no-op) when
   * not yet initialized, when the snapshot's half-res grid / stride don't match
   * the current pipeline, or when the buffer length doesn't match the live
   * reservoir buffers (a different render size).
   *
   * Restoring `previous` seeds the next frame's temporal reuse; restoring
   * `current` keeps the immediate shade read consistent until `gi-ris` overwrites
   * it. Uses `writeBuffer` (no 256-row alignment needed for buffer uploads).
   */
  importRestirGIReservoirs(device: GPUDevice, snap: RestirGISnapshot): boolean {
    if (!this._initialized) return false;
    const r = this._res.restirGI;
    const halfW = Math.max(1, Math.floor(this._width / 2));
    const halfH = Math.max(1, Math.floor(this._height / 2));
    if (snap.halfW !== halfW || snap.halfH !== halfH || snap.strideU32 !== RESERVOIR_GI_STRIDE) {
      return false; // grid / stride mismatch — cannot restore into a different reservoir layout
    }
    const expectU32 = r.reservoirGiCurrentBuffer.size / 4;
    if (snap.current.length !== expectU32 || snap.previous.length !== expectU32 || snap.spatial.length !== expectU32) {
      return false; // buffer-size mismatch (different render size)
    }
    this.#uploadReservoir(device, r.reservoirGiCurrentBuffer, snap.current);
    this.#uploadReservoir(device, r.reservoirGiPreviousBuffer, snap.previous);
    this.#uploadReservoir(device, r.reservoirGiSpatialBuffer, snap.spatial);
    return true;
  }

  /** copyBufferToBuffer (size is a multiple of 4) → MAP_READ → unpadded Uint32Array. */
  async #readbackReservoir(device: GPUDevice, src: GPUBuffer): Promise<Uint32Array> {
    const bytes = src.size; // already 4-aligned (stride is 30 u32; floor is 256)
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

  /** writeBuffer from a tightly-packed u32 array (buffer-to-buffer needs no row alignment). */
  #uploadReservoir(device: GPUDevice, dst: GPUBuffer, data: Uint32Array): void {
    device.queue.writeBuffer(dst, 0, data as Uint32Array<ArrayBuffer>, 0, data.length);
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
      hdrColorTexture:   this._res.common.hdrColorTexture ?? null,
      hdrIndirectTexture: this._res.common.hdrIndirectTexture ?? null,
      aoFullTexture:     this._res.gtao.aoFullTexture ?? null,
    };
  }

  /**
   * Aux G-buffer views surfaced through `FrameRendered.{normalDepth, albedo,
   * motionVectors}` (the `EngineCapabilities.supportsAuxBuffers` contract). All
   * three are full-res + ALWAYS allocated (they are core GI / denoiser inputs),
   * so this never partially returns:
   *   normalDepth   — rgba16float, xyz = world-space normal, w = linear depth.
   *   albedo        — rgba16float, demodulated visible-point diffuse albedo
   *                   (Schied 2017 §4.1) — lighting × albedo = final colour.
   *   motionVectors — rg32float, (dx, dy) screen-space pixels.
   * Fresh views per call (cheap); owned by the pipeline — callers MUST NOT
   * destroy them, and the handles are invalidated on the next setScene / resize
   * / dispose. Null before initialize() resolves.
   */
  getAuxBufferTextures(): {
    normalDepth: GPUTextureView;
    albedo: GPUTextureView;
    motionVectors: GPUTextureView;
  } | null {
    if (!this._initialized) return null;
    const c = this._res.common;
    return {
      normalDepth: c.gNormalDepthTexture.createView(),
      albedo: c.albedoTexture.createView(),
      motionVectors: c.motionVectorTexture.createView(),
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

  /** Temporal-accumulator history depth: frames accumulated since the last
   *  α=1 reset (camera motion, `requestAccumReset`, or `resize`). Increments
   *  once per rendered frame; reset to 0 on each of those events. Read by
   *  `HybridEngine.onProgress` for the `'denoiser-converge'` fraction. */
  get accumFrameIndex(): number {
    return this._accumFrameIndex;
  }

  /** Temporal-accumulator EMA weight α (history blend `1-α` per frame).
   *  The effective convergence window is ≈ `1/α` frames (α=0.01 ⇒ ~100).
   *  Target denominator for the `'denoiser-converge'` progress metric. */
  get temporalAccumAlpha(): number {
    return this._temporalAccumAlpha;
  }

  constructor(device: GPUDevice, width: number, height: number) {
    this._device = device;
    this._width  = width;
    this._height = height;
    this._ddgi = new DDGIBindingState(device);
    this._ppg  = new PPGCoordinator(device);
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
      /** T2.H2 — neural denoiser InferenceGraph (required when denoiser='neural').
       *  Kept on the options surface for forward compatibility with W10. */
      inferenceGraph?: InferenceGraph;
      /** T2.H3 — enable PPG (Müller 2017 adaptive sTree + dTree + MIS). */
      ppgEnabled?: boolean;
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
      /** GRIS / ReSTIR-PT reconnection-shift reuse (Lin et al. 2022) — opt-in.
       *  COMPILE-TIME structural gate: when true, the GI spatial + temporal
       *  pipelines are built with a `@group(1)` scene BVH/TLAS group (for the
       *  reconnection-visibility ray) + the GRIS combine shader; when false
       *  (default) they are the verbatim Sprint-17 single-group pipeline. This
       *  MUST be a compile-time decision — a runtime UBO flag that bound an
       *  extra group on the default path regressed the default render to an
       *  all-black frame (f8df9a4). Host opt-in via
       *  `HybridEngineOptions.restirPtReuse`. */
      restirPtReuse?: boolean;
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
    const d = this._device;
    const { _width: W, _height: H } = this;

    // ── ReGIR coordinator (Boksansky 2021) ────────────────────────────────
    // Construct from the resolved config so the grid byte count is known
    // BEFORE the light-tree buffer is uploaded (the grid is co-located in the
    // SAME buffer — see ReGIRCoordinator / regir.wgsl). `gridRegionBytes()` is
    // 0 when ReGIR is off ⇒ the light-tree buffer is byte-identical to before.
    this._regir = new ReGIRCoordinator(resolveReGIRConfig(options?.regirConfig));
    this._bvhHost.setRegirGridBytes(this._regir.gridRegionBytes());

    // ── Upload BVH buffers ────────────────────────────────────────────────
    this._bvhHost.uploadInitial(d, bvhBuffers);

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

    // ── Per-frame GPU resources ───────────────────────────────────────────
    this._res = createFrameResources(d, W, H, {
      gtaoDownscale: this._gtaoDownscale,
      svgfEnabled: this._denoiserMode === 'svgf-real',
    });

    // ── Resolve the GRIS structural gate BEFORE compiling pipelines ────────
    // restirPtReuse is a COMPILE-TIME decision: it selects the GI spatial +
    // temporal pipeline layouts (single-group vs two-group) + shader variants.
    // Stored so the pass constructors below bind the scene group at @group(1)
    // iff the GRIS pipeline variant was built.
    this._restirPtReuseStructural = options?.restirPtReuse ?? false;

    // ── Resolve the checkerboard half-res-shading flag ─────────────────────
    // OFF (default) ⇒ shade shades every pixel + ResolvePass passes through
    // (byte-identity). Not a compile-time structural decision — no extra bind
    // groups; it only flips two already-present UBO fields + the ResolvePass
    // gate — so it is resolved here and consumed at construction + per frame.
    this._checkerboard = options?.checkerboard ?? false;

    // ── Resolve the NRC structural gate BEFORE compiling pipelines ─────────
    // nrcEnabled is a COMPILE-TIME decision (mirrors restirPtReuse): it selects
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
      assertNrcDeviceCapable(d.limits);
      this._nrc = new NrcSubsystem(d, this._bglCache);
      const aabb = deriveSceneAABBFromBvhPositions(bvhBuffers);
      await this._nrc.initialize(aabb.min, aabb.max);
    }

    // ── Compile shaders (denoiser-agnostic) ───────────────────────────────
    const compiled = await compilePipelines(d, this._bglCache, swapChainFormat, {
      verbose: options?.verbose ?? false,
      ppgEnabled: options?.ppgEnabled ?? false,
      regirEnabled: this._regir.config.enabled,
      restirPtReuse: this._restirPtReuseStructural,
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
    this._ppgDispatchInterval = Math.max(1, Math.floor(options?.ppgDispatchInterval ?? 1));
    this._cameraMoveResetThresholdSq = options?.cameraMoveResetThresholdSq
      ?? DEFAULT_CAMERA_MOVE_RESET_THRESHOLD_SQ;
    this._temporalAccumAlpha = options?.temporalAccumAlpha
      ?? DEFAULT_TEMPORAL_ACCUM_ALPHA;
    this._checkerboardMotionThresholdSq = options?.checkerboardMotionThresholdSq
      ?? DEFAULT_CHECKERBOARD_MOTION_THRESHOLD_SQ;

    // T2.H3 — PPG is enabled iff host opted-in AND both pipelines compiled.
    // The flag itself is computed here; `_ppg.initialize()` below acts on it
    // (allocates resources, builds sTree, uploads UBOs) once the pass
    // registry is wired.
    const ppgEnabled = (options?.ppgEnabled ?? false) &&
      compiled.ppgUpdatePipeline !== undefined;

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
    // Extracted to the module-scope `registerPasses` free function (Task 4.1).
    // The thunks below preserve the original closures' late-`this` binding.
    const { registry, compositePass } = registerPasses(compiled, {
      diSpatialPasses: this._diSpatialPasses,
      giSpatialPasses: this._giSpatialPasses,
      restirPtReuseStructural: this._restirPtReuseStructural,
      checkerboard: this._checkerboard,
      sampleBudgetUboRef: this._sampleBudgetUboRef,
      sampleCountUboRef: this._sampleCountUboRef,
      accumUboRef: this._accumUboRef,
      resolveUboRef: this._resolveUboRef,
      atrousIndirectUboRef: this._atrousIndirectUboRef,
      indirectAccumPingPongRef: this._indirectAccumPingPongRef,
      regir: this._regir,
      bglCache: this._bglCache,
      bvhBuffers,
      nrcBindGroup: this._nrc !== null ? () => this._nrc!.bindGroup() : undefined,
      getActiveDenoiser: () => this._activeDenoiser!,
      getAtrousPipeline: () => this._atrousPipeline,
      isDenoiserPassEnabled: () => this._denoiserPassEnabled,
      getRegirResources: () => ({
        combinedLightTreeBuffer: this._bvhHost.lightTreeBuffer(),
        emitterBuffer: this._bvhHost.sceneBindGroupResources().emitterBuffer,
        uboBuffer: this._res.common.uboBuffer,
      }),
    });
    this._compositePass = compositePass;

    // ── Initialize all passes in parallel ────────────────────────────────
    this._passRegistry = registry;
    this._sortedPasses = registry.sortedPasses();
    await Promise.all(this._sortedPasses.map((p) => p.initialize({
      device: d, width: W, height: H, bglCache: this._bglCache, frameResources: this._res,
    })));

    // ── W9 — PPG GPU buffer init (opt-in) ────────────────────────────────
    // Delegated to PPGCoordinator: derives scene-bounds AABB from the BVH,
    // builds a fresh single-cell sTree, allocates PPG GPU storage buffers,
    // and uploads the serialised tree + both UBOs. No-op when ppgEnabled
    // is false. The kernels descend the serialised buffers each frame; the
    // CPU refines + re-uploads on rebuild cycles (Phase 2 follow-up).
    this._ppg.initialize(bvhBuffers, this._res, W, H, ppgEnabled, this._frameCount);

    this._initialized = true;
    if (options?.verbose) {
      console.log('[ReSTIR] Pipeline initialized', { W, H, bvhNodes: bvhBuffers.bvhNodes.count, emitters: bvhBuffers.emitterCount });
    }
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
      'emitters' | 'emitterCdf' | 'lightTree' | 'lightTreeNodeCount' | 'lightTreeEnabled'
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
  refreshBvhRefit(
    bvhNodesBytes: ArrayBuffer,
    positionsSlice: { byteOffset: number; data: ArrayBuffer },
  ): void {
    if (!this._initialized) return;
    this._bvhHost.refreshBvhRefit(this._device, bvhNodesBytes, positionsSlice);
  }

  /** PR-7 — upload refit BVH nodes only (positions already on GPU). */
  refreshBvhNodesOnly(bvhNodesBytes: ArrayBuffer): void {
    if (!this._initialized) return;
    this._bvhHost.refreshBvhNodesOnly(this._device, bvhNodesBytes);
  }

  /** Live merged vertex buffer for GPU skinning writes. */
  getBvhPositionBuffer(): GPUBuffer | null {
    return this._initialized ? this._bvhHost.getBvhPositionBuffer() : null;
  }

  /** Shared rect-area emitter buffer + tri count for RC emitter NEE. Null
   *  before init. Emitters are world-space triangles, so the same buffer the
   *  shade/ReSTIR-DI path uses is valid for the RC probe cast — no re-upload. */
  getEmitterBufferAndCount(): { buffer: GPUBuffer; count: number } | null {
    return this._initialized ? this._bvhHost.emitterBufferAndCount() : null;
  }

  /** WS1 — live merged per-vertex normal buffer for GPU skinning writes. */
  getBvhNormalBuffer(): GPUBuffer | null {
    return this._initialized ? this._bvhHost.getBvhNormalBuffer() : null;
  }

  /** PR-4 — upload refit TLAS nodes + instance transforms (topology unchanged). */
  refreshTlasRefit(
    tlasNodes: ArrayBuffer,
    worldToLocal: ArrayBuffer,
    localToWorld: ArrayBuffer,
  ): void {
    if (!this._initialized) return;
    this._bvhHost.refreshTlasRefit(this._device, tlasNodes, worldToLocal, localToWorld);
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
  ): void {
    if (!this._initialized) return;
    this._bvhHost.refreshBvhMaterialSlice(this._device, indexSlice, beerFull, emissiveFull);
  }

  /**
   * Full BVH-buffer reupload — destroy + recreate the four BVH GPU
   * buffers (nodes, index, beer, positions) from a freshly-built
   * `SceneBVHBuffers`. Used by `HybridEngine.updatePrimitive`'s topology-
   * change path after a `buildReSTIRSceneBVH` rebuild. Emitter buffers
   * are NOT touched here — call `updateEmitters` separately if the
   * emitter list also changed.
   *
   * The pipeline shaders and bind-group layouts stay intact because
   * `buildSceneBindGroup` is re-invoked per-frame in `renderFrame()`
   * from the live buffer handles, so the destroy + recreate is picked
   * up automatically next frame.
   */
  refreshBvhFullRebuild(
    bvhBuffers: Pick<
      SceneBVHBuffers,
      'bvhNodes' | 'bvhIndex' | 'bvhBeerColors' | 'bvhEmissiveLe' | 'bvhNormals' | 'bvhPositions' | 'bvhMode' | 'tlas'
    >,
  ): void {
    if (!this._initialized) return;
    this._bvhHost.refreshBvhFullRebuild(this._device, bvhBuffers);
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
    // Destroy + reallocate per-frame resources at the new size. Preserve the
    // GTAO downscale resolved at initialize() so a resize keeps the AO target
    // at the same quarter/half-res tier the host selected.
    destroyFrameResources(this._res);
    this._resourceCache.clear();
    this._res = createFrameResources(this._device, width, height, {
      gtaoDownscale: this._gtaoDownscale,
      // Preserve the init-time SVGF gating (G-P2.6) — the active denoiser is
      // fixed for the pipeline's lifetime, so a resize keeps the same policy.
      svgfEnabled: this._denoiserMode === 'svgf-real',
    });
    // W9 — re-allocate PPG resolution-dependent buffers + re-upload the
    // (unchanged) sTree topology so the new bind groups have valid GPU
    // buffers to bind. The CPU sTree itself isn't size-dependent and
    // survives the resize unchanged. No-op inside the coordinator when
    // PPG is disabled.
    this._ppg.onResize(this._res, width, height, this._frameCount);
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
    this._ppgDispatchInterval = Math.max(1, Math.floor(interval));
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
    const bgComposite = buildCompositePresentBindGroup(
      d,
      this._bglCache,
      this._res.common.resolvedTexture,
      this._res.common.compositeSampler,
      this._resourceCache,
    );
    const compositePass = this._compositePass;
    if (compositePass == null) return;
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

    // ── Camera motion (computed up-front for the checkerboard motion fallback) ─
    // The checkerboard sparse path (shade + spatial compaction + resolve
    // gap-fill) reuses last-frame reservoirs/radiance for the gap-parity half.
    // Under camera motion that lag is exposed as softening, so we force
    // FULL-RATE shading on a moving frame: the per-frame `cbActiveThisFrame` flag
    // is `_checkerboard && !cbMotionExceeded`, threaded identically into the UBO
    // (shade/spatial read checkerboardOn/frameParity), the ShadePass/
    // SpatialReservoirPass dispatch compaction (passCtx.checkerboardOn), and the
    // ResolvePass gap-fill — so all three agree every frame.
    //
    // The checkerboard fallback uses its OWN, finer threshold
    // (`_checkerboardMotionThresholdSq`, default 0.004 = 0.063²) — NOT the much
    // coarser temporal-accumulator reset (`_cameraMoveResetThresholdSq`, 1.0):
    // half-rate reservoir lag shows at far smaller motion than a full history
    // discard, so checkerboard must drop to full-rate earlier (GPU-validated on
    // dzn — at the coarse 1.0 threshold the motion A/B fell to ~24 dB; the 0.004
    // threshold keeps checkerboard sparse through a slow drag (~50 dB) and flips
    // a faster pan to full-rate, where ON==OFF bit-identically). `_lastCameraPos`
    // is written only at end-of-frame, so reading it here matches the post-UBO
    // motion calc exactly.
    const mdx = inputs.camera.cameraPos[0] - this._lastCameraPos[0];
    const mdy = inputs.camera.cameraPos[1] - this._lastCameraPos[1];
    const mdz = inputs.camera.cameraPos[2] - this._lastCameraPos[2];
    const camMoveSqUpfront = mdx * mdx + mdy * mdy + mdz * mdz;
    const cbMotionExceeded = camMoveSqUpfront > this._checkerboardMotionThresholdSq;
    const cbActiveThisFrame = this._checkerboard && !cbMotionExceeded;

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
    updateUBO(d, this._res.common.uboBuffer, inputs, {
      enabled: this._ppg.enabled,
      mixAlpha: this._ppg.mixAlpha,
    }, this._regir.uboState(), {
      enabled: cbActiveThisFrame,
      frameParity: this._frameCount & 1,
    });

    // ── Build placeholder texture view ────────────────────────────────────
    const placeholderView = this._resourceCache.textureView(this._res.common.placeholderTexture);

    const {
      frame: bgFrame,
      scene: bgScene,
      ubo: bgUbo,
      hybridLayers: bgHybrid,
    } = buildPerFrameBindGroups(
      d,
      this._bglCache,
      this._res,
      this._bvhHost.sceneBindGroupResources(),
      this._ddgi,
      placeholderView,
      this._resourceCache,
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
    const passLayout = buildPassLayout({ denoiserMode: this._denoiserMode, ...this._passLayoutConfig });

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
    // The temporal-accumulator reset keeps its OWN (coarser) threshold
    // `_cameraMoveResetThresholdSq` (default 1.0) — a full history discard is a
    // bigger event than the checkerboard full-rate fallback (which trips at the
    // finer `_checkerboardMotionThresholdSq` computed above). Reuses the same
    // `camMoveSqUpfront` delta so there is one motion magnitude, two thresholds.
    const isMoving = camMoveSqUpfront > this._cameraMoveResetThresholdSq;
    if (isMoving) {
      this._accumFrameIndex = 0;
    }

    // Resolve the temporal-accumulator ping-pong slots for this frame.
    const readAccum  = this._accumPingPongIndex === 0
      ? this._res.common.accumTextureA : this._res.common.accumTextureB;
    const writeAccum = this._accumPingPongIndex === 0
      ? this._res.common.accumTextureB : this._res.common.accumTextureA;

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
      lightTreeBindGroup: bgLightTree,
      wgX, wgY, wgX16, wgY16, halfWgX, halfWgY,
      // Checkerboard sparse dispatch state. When ON, ShadePass + SpatialReservoirPass
      // each compact their dispatch to ~half the threads (one per active-parity
      // pixel), and ResolvePass gap-fills the complementary half. This is the
      // SAME `cbActiveThisFrame` (= `_checkerboard && !isMoving` — fast-motion
      // forces full-rate) / frameParity (`frameCount & 1`) updateUBO packs into
      // the WalkaroundUBO above, so the shade/spatial shaders' compacted-gid
      // decode, their UBO reads, and the resolve gap-fill all agree this frame.
      checkerboardOn: cbActiveThisFrame,
      frameParity: this._frameCount & 1,
      gtaoDownscale: this._gtaoDownscale,
      gNormalDepthView,
      computeDesc,
      renderTimestampWrites,
      frameState,
    };

    const gateOpts: PassGateOptions = {
      denoiserMode: this._denoiserMode,
      ppgEnabled: this._ppg.enabled,
      // Phase-0 — PPG train-pass modulo gate. The ppg-update pass dispatches
      // only on multiples of `_ppgDispatchInterval`. interval=1 ⇒ always true
      // (every frame). The persisted tree + gi-ris guided sampling are
      // unaffected — this only skips flux accumulation on off-interval frames.
      // (`_ppgDispatchInterval` is clamped ≥ 1 in initialize().)
      ppgTrainThisFrame: this._frameCount % this._ppgDispatchInterval === 0,
      // Phase-0 — gate GTAO + its upsample when the preset disabled it.
      gtaoEnabled: this._gtaoEnabled,
    };

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
    this._accumPingPongIndex = 1 - this._accumPingPongIndex;
    this._accumFrameIndex++;
    this._lastCameraPos = [...inputs.camera.cameraPos];

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

    // NRC ON ⇒ fold the self-training-record copy into THIS encoder (after the
    // gi-ris pass wrote the records, before submit) so the host gather sees the
    // current frame's records. No-op buffer when NRC is off (`_nrc` null).
    this._nrc?.recordCopyForReadback(encoder);

    // Resolve timestamps + copy into the inactive readback buffer.
    resolveTimestamps(encoder, this._tsState, this._frameCount, passLayout.slotCount);

    d.queue.submit([encoder.finish()]);

    // W9 follow-up — periodic training/refine cycle:
    // fluxAtomics GPU readback -> CPU dTree/sTree refinement -> re-upload.
    this._ppg.maybeRunTrainingRefine(this._res, this._frameCount);

    // NRC ON ⇒ read back this frame's self-training records and run ONE train
    // step (Müller §5 self-training; HOST-OWNS-CADENCE — one step per frame).
    // Fire-and-forget: the readback maps async; a still-pending readback skips
    // this frame and picks up fresh records next frame (re-entrancy guarded in
    // the subsystem). No-op when NRC is off. The promise rejection is swallowed
    // so a transient device-lost during teardown never surfaces on the render
    // hot path.
    if (this._nrc !== null) {
      void this._nrc.trainFromRecords().catch(() => { /* device lost / disposed */ });
    }

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

  dispose(): void {
    this._bvhHost.dispose();
    this._compositePass = null;
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
  setRCInputs(inputs: {
    cascade0Buffer: GPUBuffer;
    paramsBytes: ArrayBuffer;
  } | null): void {
    this._ddgi.setRCInputs(inputs);
  }
}
