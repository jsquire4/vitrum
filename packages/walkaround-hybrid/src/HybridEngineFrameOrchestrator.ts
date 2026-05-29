/**
 * Per-frame render orchestration for {@link HybridEngine} (W4e).
 */
import type {
  FrameInput,
  FrameOutput,
  FrameStats,
  ProgressStats,
  EngineDebugSurface,
  EngineState,
  Scene,
} from '@vitrum/core';
import { asBackendTexture } from '@vitrum/core';
import type * as THREE from 'three';
import type { DDGI } from './ddgi/DDGI.js';
import { packDDGIGridParams } from './ddgi/ddgiGridUbo.js';
import { propagateBvhToGiSubsystems } from './HybridEngineGiPropagation.js';
import type { RCSubsystem } from './HybridEngineRC.js';
import type { Tunables } from './HybridEngineTuning.js';
import type { WalkaroundGPUPipeline } from './pipeline/WalkaroundGPUPipeline.js';
import type { SceneBVHBuffers } from './restir/bvhCompute.js';
import type { GpuSkinningSubsystem } from './skin/GpuSkinningSubsystem.js';

export const HYBRID_FRAME_SKIP_OUTPUT: FrameOutput = {
  kind: 'skipped',
  samplesAccumulated: 0,
  isConverged: false,
};

export interface HybridEngineFrameDiag {
  initStart: number;
  initCount: number;
  disposeCount: number;
  skipNoPipeline: number;
  skipNoBvh: number;
  skipNoSwapView: number;
  skipFrameInterval: number;
  framesDispatched: number;
  lastReportTs: number;
}

/** Runtime-mutable lighting cluster (the engine's `updateLighting()` mutates
 *  these). Grouped so a new lighting field is one edit in the engine's
 *  `_lightingSnapshot()` rather than one per dependency builder. */
export interface HybridLightingDeps {
  primaryLightDir: [number, number, number];
  primaryLightIntensity: number;
  skyTint: [number, number, number];
  skyIrradiance: number;
}

/** Tuple-typed denoiser-filter cluster (firefly clamp + per-channel atrous
 *  sigmas). Lives outside the number-only {@link Tunables} table because the
 *  values are tuples; grouped for the same one-edit reason as the lighting
 *  cluster. */
export interface HybridDenoiserFilterDeps {
  indirectFireflyClamp: readonly [number, number, number];
  atrousDirectSigmas: readonly [number, number, number];
  atrousIndirectSigmas: readonly [number, number, number];
  /** T5 — stained-glass opt-in flag bitfield (bit 0 = sun-caustic, bit 1 =
   *  sky-aperture). Default 0 (both OFF). Splatted into pipeline.renderFrame
   *  as `stainedGlassFlags`. Lives in this cluster (not the number-only
   *  {@link Tunables} table) because it is a derived bitfield, not a
   *  host-overridable scalar tunable. */
  stainedGlassFlags: number;
}

export interface HybridEngineFrameDeps extends HybridLightingDeps, HybridDenoiserFilterDeps {
  state: EngineState;
  debug: boolean;
  dbg: HybridEngineFrameDiag | null;
  pipeline: WalkaroundGPUPipeline | null;
  bvhBuffers: SceneBVHBuffers | null;
  /** When true, caller already called reset() and frame should skip. */
  consumeRebuildKeyChange: () => boolean;
  targetFrameIntervalMs: number | null;
  getLastFrameTs: () => number;
  setLastFrameTs: (ts: number) => void;
  /** Canvas (swap-chain) width — what the composite pass blits TO. */
  width: number;
  /** Canvas (swap-chain) height. */
  height: number;
  /** Internal render width (= canvas × resolutionFactor) — what the compute
   *  kernels dispatch over and the UBO `screenSize` carries. Equals `width`
   *  when no resolutionFactor downscale is active. */
  internalWidth: number;
  /** Internal render height (= canvas × resolutionFactor). */
  internalHeight: number;
  /** Honour `FrameInput.quality.resolutionFactor` by (debounced) resizing the
   *  internal render resolution. Returns the internal dims to dispatch at this
   *  frame (which may be unchanged if the resize was debounced or the factor
   *  was unchanged). Pure-ish: the only side effect is `pipeline.resize` +
   *  engine internal-dim bookkeeping, both owned by the engine. */
  applyResolutionFactor: (factor: number | undefined, nowMs: number) => { width: number; height: number };
  skinning: GpuSkinningSubsystem | null;
  lastScene: Scene | null;
  runSkinning: () => void;
  ddgiOn: boolean;
  isLayerEnabled: (layer: string) => boolean;
  ddgi: DDGI;
  ddgiTraversalScene: THREE.Scene | null;
  ensureThreeSceneRoot: () => THREE.Scene | null;
  device: GPUDevice;
  tunables: Tunables;
  rc: RCSubsystem | null;
  rcWeight: number;
  frameSubs: ReadonlyArray<(stats: FrameStats) => void>;
  /** T3.E — long-running progress subscribers (`'ddgi-warmup'` +
   *  `'denoiser-converge'`). Fired at the end of each dispatched frame,
   *  once per still-converging kind. Empty array ⇒ zero progress work. */
  progressSubs: ReadonlyArray<(progress: ProgressStats) => void>;
  verbose: boolean;
  debugTimings: Array<{ t: number; ms: number }>;
  debugSurface: EngineDebugSurface;
  presentLastFrame: (view: GPUTextureView) => void;
}

/** Minimum interval (ms) between internal-resolution reallocations driven by
 *  `quality.resolutionFactor`. Each reallocation destroys + recreates all
 *  FrameResources and resets the temporal accumulator (~5-30 ms + a 1-frame
 *  history reset), so a host ramping the factor continuously must not thrash
 *  it (Risk R5). 250 ms ≈ 15 frames at 60 FPS. */
export const RESOLUTION_FACTOR_DEBOUNCE_MS = 250;

/** Resolve the per-frame internal render size from a host `resolutionFactor`,
 *  debouncing the actual reallocation.
 *
 *  Pure function — no GPU, no side effects. The caller (`applyResolutionFactor`
 *  on the engine deps) owns the `pipeline.resize` + bookkeeping when this
 *  returns `shouldResize: true`.
 *
 *  @param swapW/swapH       Canvas (swap-chain) dimensions.
 *  @param factor            Host `FrameInput.quality.resolutionFactor` (or undefined).
 *  @param currentW/currentH Current internal render dimensions.
 *  @param nowMs             `performance.now()` for the debounce clock.
 *  @param lastResizeTs      Timestamp of the last accepted resolution resize.
 *  @returns target internal dims, whether to actually resize this frame, and
 *           whether the change was debounced (so the caller does NOT update the
 *           debounce timestamp on a no-op). */
export function resolveInternalRenderSize(args: {
  swapW: number;
  swapH: number;
  factor: number | undefined;
  currentW: number;
  currentH: number;
  nowMs: number;
  lastResizeTs: number;
}): { targetW: number; targetH: number; shouldResize: boolean } {
  // Clamp factor to (0, 1]. Undefined / out-of-range / non-finite ⇒ 1.0 (full
  // resolution; regression-safe default identical to pre-Phase-0 behaviour).
  const f =
    typeof args.factor === 'number' && Number.isFinite(args.factor) && args.factor > 0
      ? Math.min(1, args.factor)
      : 1;
  const targetW = Math.max(1, Math.round(args.swapW * f));
  const targetH = Math.max(1, Math.round(args.swapH * f));

  // No meaningful change (within 2 px on either axis) ⇒ never resize. This
  // catches both "factor unchanged" and "factor changed by a sub-pixel amount"
  // (the omitted-factor regression-guard path lands here when target == swap).
  const dw = Math.abs(targetW - args.currentW);
  const dh = Math.abs(targetH - args.currentH);
  if (dw < 2 && dh < 2) {
    return { targetW, targetH, shouldResize: false };
  }

  // Changed beyond threshold — but debounce the reallocation: at most one
  // accepted resize per RESOLUTION_FACTOR_DEBOUNCE_MS. First-ever resize
  // (lastResizeTs === 0) is always allowed.
  const debounced = args.lastResizeTs !== 0 && args.nowMs - args.lastResizeTs < RESOLUTION_FACTOR_DEBOUNCE_MS;
  return { targetW, targetH, shouldResize: !debounced };
}

/**
 * DDGI warm-up progress (`'ddgi-warmup'`).
 *
 * The probe round-robin updates one stratum of `1/stride` probes per enabled
 * `updateFrame` tick (see {@link DDGI.updateFrame}). A freshly-built /
 * invalidated grid therefore needs `stride` ticks for *every* probe to receive
 * its first update — which is exactly when {@link DDGI.ready} flips true. So
 * the honest metric is `frame / stride`, clamped to [0,1]:
 *
 *   - `frame` (= {@link DDGI.warmupFrame}) is reset to 0 by
 *     `invalidateProbeCache()`, which `setScene()` (→ teardown → fresh DDGI is
 *     constructed at `frame=0`) and `updateLighting()` both trigger.
 *   - `stride` (= {@link DDGI.warmupStride}) is the round-robin divisor.
 *   - `ready` short-circuits emission: once the grid is warm we return `null`
 *     so `onProgress` stops spamming a steady `fraction:1`.
 *
 * Returns `null` when there is nothing to report (already warm, or stride is
 * not yet meaningful), so the caller emits ONLY while ramping 0→1.
 */
export function computeDdgiWarmupProgress(args: {
  frame: number;
  stride: number;
  ready: boolean;
}): ProgressStats | null {
  // Stop emitting once warm — `ready` flips true at `frame >= stride`.
  if (args.ready) return null;
  // A stride of <1 (or a not-yet-sized grid) has no meaningful warm-up window.
  const target = Math.max(1, Math.floor(args.stride));
  const current = Math.max(0, args.frame);
  const fraction = Math.min(1, Math.max(0, current / target));
  return { kind: 'ddgi-warmup', current, target, fraction };
}

/**
 * Denoiser temporal-convergence progress (`'denoiser-converge'`).
 *
 * Every realtime denoiser here feeds the final temporal accumulator, which
 * blends `alpha` of the new frame with `1-alpha` of history each frame
 * (α=0.01 ⇒ ~99% history retained ⇒ a ~`1/α`-frame effective averaging
 * window). The accumulator's history depth is {@link WalkaroundGPUPipeline.accumFrameIndex},
 * which:
 *   - increments once per rendered frame,
 *   - is reset to 0 on camera motion (`isMoving`), `requestAccumReset()`
 *     (lighting change / emitter edit), and `resize()`.
 *
 * So the honest convergence fraction is `accumFrameIndex / window` where
 * `window = round(1/alpha)`. The reset on motion is inherited for free:
 * `accumFrameIndex` drops to 0, so `fraction` snaps back to 0 and ramps again.
 *
 * Returns `null` once converged (`accumFrameIndex >= window`) so emission
 * stops at steady state instead of pinning `fraction:1` every frame.
 */
export function computeDenoiserConvergeProgress(args: {
  accumFrameIndex: number;
  alpha: number;
}): ProgressStats | null {
  // alpha ≤ 0 (or non-finite) ⇒ pure history hold, no finite convergence
  // window to report. alpha ≥ 1 ⇒ no temporal accumulation (each frame is
  // fully fresh) ⇒ "converged" every frame ⇒ nothing to report.
  if (!Number.isFinite(args.alpha) || args.alpha <= 0 || args.alpha >= 1) {
    return null;
  }
  const target = Math.max(1, Math.round(1 / args.alpha));
  const current = Math.max(0, args.accumFrameIndex);
  // Stop emitting once the effective window is full (steady state reached).
  if (current >= target) return null;
  const fraction = Math.min(1, Math.max(0, current / target));
  return { kind: 'denoiser-converge', current, target, fraction };
}

export function fingerprintHybridPipelineRebuildKey(
  key: string | number | null | undefined,
): string {
  if (key === null || key === undefined) return '__null';
  if (typeof key === 'number') return Number.isNaN(key) ? '__n:NaN' : `__n:${key}`;
  return `__s:${key}`;
}

export function getPreferredSwapChainFormat(): GPUTextureFormat {
  return (typeof navigator !== 'undefined' && 'gpu' in navigator
    ? (navigator.gpu as { getPreferredCanvasFormat?: () => GPUTextureFormat })
        .getPreferredCanvasFormat?.() ?? 'bgra8unorm'
    : 'bgra8unorm');
}

function runDdgiAndRc(deps: HybridEngineFrameDeps, input: FrameInput): void {
  const ddgiLayerOn = deps.ddgiOn && deps.isLayerEnabled('ddgi');
  const ddgiScene = ddgiLayerOn
    ? (deps.ddgiTraversalScene ?? deps.ensureThreeSceneRoot())
    : null;

  // Per-frame BVH ⇒ GI-subsystem cascade — same owner the post-update /
  // post-publish paths use, but tlas-sync-only (allowRcSceneRebuild=false so
  // merged-mode RC is NOT rebuilt every frame), and the DDGI sync gated on the
  // resolved ddgi-layer scene exactly as before.
  propagateBvhToGiSubsystems({
    ddgi: deps.ddgi,
    rc: deps.rc,
    bvhBuffers: deps.bvhBuffers,
    lastScene: deps.lastScene,
    syncDdgi: ddgiLayerOn && ddgiScene != null,
    allowRcSceneRebuild: false,
    ensureThreeSceneRoot: deps.ensureThreeSceneRoot,
  });

  if (ddgiLayerOn && ddgiScene != null) {
    void deps.ddgi.updateFrame({
      scene: ddgiScene,
      device: deps.device,
      enabled: true,
    });
    if (deps.ddgi.ready) {
      deps.ddgi.pass.setGlassMixScale(deps.tunables.glassMixScale);
    }
  }

  const pipeline = deps.pipeline!;
  if (deps.rc) {
    deps.rc.dispatchFrame({
      sunDirection: deps.primaryLightDir,
      sunColor: [
        deps.primaryLightIntensity,
        deps.primaryLightIntensity,
        deps.primaryLightIntensity,
      ],
      frameSeed: input.frameSeed,
      triIntersectEpsilon: deps.tunables.triIntersectEpsilon,
    });
    pipeline.setRCInputs(deps.rc.buildRCInputs(deps.rcWeight));
  } else {
    pipeline.setRCInputs(null);
  }

  if (!ddgiLayerOn) {
    pipeline.setDDGIInputs(null);
  } else if (deps.ddgi.ready) {
    const atlas = deps.ddgi.pass.getReadAtlasGPUTextures?.();
    if (atlas) {
      pipeline.setDDGIInputs({
        irradianceTex: atlas.irradiance,
        visibilityTex: atlas.visibility,
        gridParams: packDDGIGridParams(deps.ddgi.probeGrid.params),
      });
    }
  }
}

/**
 * T3.E — emit long-running progress events for the two walkaround warm-up
 * signals: DDGI probe convergence (`'ddgi-warmup'`) and temporal-accumulator
 * denoiser convergence (`'denoiser-converge'`). Both metrics are derived
 * PURELY from live subsystem state (the DDGI round-robin counter + the
 * pipeline's accumulator history depth) via the pure
 * {@link computeDdgiWarmupProgress} / {@link computeDenoiserConvergeProgress}
 * functions, each of which returns `null` once its signal is converged so we
 * stop spamming a steady `fraction:1`.
 *
 * No-op when no `onProgress` subscriber is registered — zero work, zero state
 * reads — so an unobserved engine pays nothing.
 */
function emitProgressTelemetry(
  deps: HybridEngineFrameDeps,
  pipeline: WalkaroundGPUPipeline,
): void {
  if (deps.progressSubs.length === 0) return;

  const events: ProgressStats[] = [];

  // DDGI warm-up — only meaningful while the ddgi layer is active.
  if (deps.ddgiOn && deps.isLayerEnabled('ddgi')) {
    const warm = computeDdgiWarmupProgress({
      frame: deps.ddgi.warmupFrame,
      stride: deps.ddgi.warmupStride,
      ready: deps.ddgi.ready,
    });
    if (warm) events.push(warm);
  }

  // Denoiser temporal convergence — anchored to the real accumulator depth.
  const converge = computeDenoiserConvergeProgress({
    accumFrameIndex: pipeline.accumFrameIndex,
    alpha: pipeline.temporalAccumAlpha,
  });
  if (converge) events.push(converge);

  for (const event of events) {
    for (const sub of deps.progressSubs) {
      try {
        sub(event);
      } catch (err) {
        if (deps.verbose) console.warn('[HybridEngine] onProgress subscriber threw', err);
      }
    }
  }
}

function emitFrameTelemetry(
  deps: HybridEngineFrameDeps,
  pipeline: WalkaroundGPUPipeline,
  dt: number,
  now: number,
): void {
  if (deps.frameSubs.length > 0) {
    const gpu = pipeline.lastGpuTimings;
    const gpuTotal = gpu?.['total'];
    const memBreakdown = deps.debugSurface.estimatedGpuMemoryBytes?.() ?? undefined;
    const stats: FrameStats = {
      frameTimeMs: dt,
      ...(gpuTotal !== undefined ? { gpuTimeMs: gpuTotal } : {}),
      ...(gpu ? { passTimings: gpu } : {}),
      spp: 1,
      ...(memBreakdown
        ? { gpuMemoryBytes: memBreakdown, estimatedGpuMemoryBytes: memBreakdown.total }
        : {}),
    };
    for (const sub of deps.frameSubs) {
      try {
        sub(stats);
      } catch (err) {
        if (deps.verbose) console.warn('[HybridEngine] onFrame subscriber threw', err);
      }
    }
  }

  emitProgressTelemetry(deps, pipeline);

  if (deps.debug) {
    deps.debugTimings.push({ t: now, ms: dt });
    if (deps.debugTimings.length > 240) deps.debugTimings.shift();

    if (typeof window !== 'undefined') {
      const w = window as unknown as { __WGPU__?: { walkaround?: { frameTimings: unknown } } };
      if (w.__WGPU__?.walkaround) {
        const ft = w.__WGPU__.walkaround.frameTimings as Array<{ t: number; ms: number }>;
        if (Array.isArray(ft)) {
          ft.push({ t: now, ms: dt });
          if (ft.length > 240) ft.shift();
        }
      }
    }
  }
}

export function runHybridEngineFrame(deps: HybridEngineFrameDeps, input: FrameInput): FrameOutput {
  if (deps.state === 'paused' || deps.state === 'disposed' || deps.state === 'error') {
    return HYBRID_FRAME_SKIP_OUTPUT;
  }

  if (deps.consumeRebuildKeyChange()) {
    return HYBRID_FRAME_SKIP_OUTPUT;
  }

  const dbg = deps.dbg;
  const pipeline = deps.pipeline;
  const bvh = deps.bvhBuffers;
  if (!pipeline) {
    if (dbg) dbg.skipNoPipeline++;
    return HYBRID_FRAME_SKIP_OUTPUT;
  }
  if (!bvh) {
    if (dbg) dbg.skipNoBvh++;
    return HYBRID_FRAME_SKIP_OUTPUT;
  }

  const now = performance.now();
  if (
    deps.targetFrameIntervalMs !== null &&
    deps.getLastFrameTs() !== 0 &&
    now - deps.getLastFrameTs() < deps.targetFrameIntervalMs
  ) {
    if (dbg) dbg.skipFrameInterval++;
    const skipSwapView = input.swapChainView as GPUTextureView | undefined;
    if (skipSwapView) {
      deps.presentLastFrame(skipSwapView);
    }
    return HYBRID_FRAME_SKIP_OUTPUT;
  }
  deps.setLastFrameTs(now);

  const t0 = now;
  const swapView = input.swapChainView as GPUTextureView | undefined;
  const swapFmt =
    (input.swapChainFormat as GPUTextureFormat | undefined) ?? getPreferredSwapChainFormat();

  if (!swapView) {
    if (dbg) dbg.skipNoSwapView++;
    return HYBRID_FRAME_SKIP_OUTPUT;
  }

  if (dbg) {
    dbg.framesDispatched++;
    if (dbg.lastReportTs === 0) dbg.lastReportTs = now;
    if (now - dbg.lastReportTs > 5_000) {
      const elapsed = (now - dbg.lastReportTs) / 1_000;
      const gpu = pipeline.lastGpuTimings ?? {};
      const gpuTotal = gpu['total'];
      console.log('[hybrid:debug] rate (5s window)', {
        framesDispatched: dbg.framesDispatched,
        fps: (dbg.framesDispatched / elapsed).toFixed(2),
        skipReasons: {
          noPipeline: dbg.skipNoPipeline,
          noBvh: dbg.skipNoBvh,
          noSwapView: dbg.skipNoSwapView,
          frameInterval: dbg.skipFrameInterval,
        },
        gpuTotalMs: gpuTotal !== undefined ? +gpuTotal.toFixed(2) : 'n/a',
        gpuPerPassMs: gpu,
      });
      dbg.framesDispatched = 0;
      dbg.skipNoPipeline = 0;
      dbg.skipNoBvh = 0;
      dbg.skipNoSwapView = 0;
      dbg.skipFrameInterval = 0;
      dbg.lastReportTs = now;
    }
  }

  deps.runSkinning();
  runDdgiAndRc(deps, input);

  // §5.1 — honour per-frame `quality.resolutionFactor` by (debounced) scaling
  // the internal render resolution. The composite pass upscales the
  // internal-sized resolvedTexture to the full swap-chain view, so the
  // compute kernels + UBO `screenSize` use the internal dims (not the canvas
  // dims). Canvas resizes still require `setSize()` (see renderFrame JSDoc).
  const internal = deps.applyResolutionFactor(input.quality?.resolutionFactor, now);

  pipeline.renderFrame({
    viewMatrix: new Float32Array(input.viewMatrix),
    projMatrix: new Float32Array(input.projMatrix),
    prevViewMatrix: new Float32Array(input.prevViewMatrix ?? input.viewMatrix),
    prevProjMatrix: new Float32Array(input.prevProjMatrix ?? input.projMatrix),
    cameraPos: input.cameraPosition as [number, number, number],
    screenWidth: internal.width,
    screenHeight: internal.height,
    frameSeed: input.frameSeed,
    totalEmissivePower: bvh.totalEmissivePower ?? 1.0,
    emitterCount: bvh.emitters?.count ?? 0,
    primaryLightDir: deps.primaryLightDir,
    primaryLightIntensity: deps.primaryLightIntensity,
    skyTint: deps.skyTint,
    skyIrradiance: deps.skyIrradiance,
    ...deps.tunables,
    indirectFireflyClamp: deps.indirectFireflyClamp,
    bvhMode: bvh.bvhMode === 'tlas' ? 1 : 0,
    tlasNodeCount: bvh.tlas?.nodeCount ?? 0,
    // Light-tree DI light SELECTION gate. Default ON whenever the scene has
    // ≥ 2 emitters (set at BVH build); RIS falls back to the flat power-CDF
    // path otherwise. Unbiased in both states (the WRS weight divides p̂ by
    // the exact selection pdf).
    lightTreeEnabled: bvh.lightTreeEnabled ? 1 : 0,
    lightTreeNodeCount: bvh.lightTreeNodeCount ?? 0,
    atrousDirectSigmas: deps.atrousDirectSigmas,
    atrousIndirectSigmas: deps.atrousIndirectSigmas,
    stainedGlassFlags: deps.stainedGlassFlags,
    swapChainView: swapView,
    swapChainFormat: swapFmt,
  });

  emitFrameTelemetry(deps, pipeline, performance.now() - t0, now);

  return {
    kind: 'rendered',
    primaryRadiance: asBackendTexture<'webgpu', GPUTextureView>(swapView),
    samplesAccumulated: 1,
    isConverged: false,
  };
}
