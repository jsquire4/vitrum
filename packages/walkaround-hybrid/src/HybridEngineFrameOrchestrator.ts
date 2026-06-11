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
import { TONEMAP_MODE_INDEX } from '@vitrum/shared-samplers';
import type { DDGI } from './ddgi/DDGI.js';
import { packDDGIGridParams } from './ddgi/ddgiGridUbo.js';
import { coreEmittersToDDGILights } from './coreEmittersToDDGILights.js';
import { propagateBvhToGiSubsystems } from './HybridEngineGiPropagation.js';
import type { RCSubsystem } from './HybridEngineRC.js';
import type { Tunables } from './HybridEngineTuning.js';
import type { WalkaroundGPUPipeline } from './pipeline/WalkaroundGPUPipeline.js';
import type { SceneBVHBuffers } from './restir/bvhCore.js';
import type { GpuSkinningSubsystem } from './skin/GpuSkinningSubsystem.js';

export const HYBRID_FRAME_SKIP_OUTPUT: FrameOutput = {
  kind: 'skipped',
  samplesAccumulated: 0,
  isConverged: false,
};

/**
 * H20-A — sky-only present for the empty-scene-ready state.
 *
 * When the scene has zero mesh primitives the engine reaches `'ready'` WITHOUT
 * a pipeline or BVH (see HybridEngineLifecycle `_runInitChain` — "empty scene;
 * ready without BVH/pipeline"). Before this path such frames returned SKIP on
 * every call and the host's swap chain was never written (a clean canvas mount
 * would present whatever the browser cleared it to). This is the minimal honest
 * v1: clear the swap-chain view to the flat sky colour (`skyTint *
 * skyIrradiance`) via a single device-level clear render pass — no compute, no
 * pipeline, no BVH. The walkaround sky is a scalar tint today (no directional
 * IBL on this stack), so a flat sky fill is the radiometrically-faithful
 * empty-scene background. Returns a genuine `kind:'rendered'` FrameOutput so a
 * host observing renderFrame sees a presented frame, not a skip.
 *
 * `clearValue` is the linear-sRGB sky radiance the swap chain expects; the
 * composite path elsewhere writes linear values into the same (typically
 * `*-unorm`/`*-srgb`) swap format, so we match that convention by writing the
 * linear tint directly. The alpha is 1 to fully cover the target.
 */
function presentSkyOnly(
  device: GPUDevice,
  swapView: GPUTextureView,
  skyTint: readonly [number, number, number],
  skyIrradiance: number,
): FrameOutput {
  // Flat sky radiance = tint × irradiance, clamped non-negative. Values may
  // exceed 1 (HDR sky); the swap-chain attachment clamps on write as usual.
  const r = Math.max(0, skyTint[0] * skyIrradiance);
  const g = Math.max(0, skyTint[1] * skyIrradiance);
  const b = Math.max(0, skyTint[2] * skyIrradiance);
  const encoder = device.createCommandEncoder({ label: 'hybrid-sky-only-present' });
  const pass = encoder.beginRenderPass({
    label: 'hybrid-sky-only-present',
    colorAttachments: [{
      view: swapView,
      loadOp: 'clear',
      storeOp: 'store',
      clearValue: { r, g, b, a: 1 },
    }],
  });
  pass.end();
  device.queue.submit([encoder.finish()]);
  return {
    kind: 'rendered',
    primaryRadiance: asBackendTexture<'webgpu', GPUTextureView>(swapView),
    samplesAccumulated: 1,
    isConverged: false,
  };
}

// Column-major mat4 multiply, matching WGSL `a * b` for the camera matrices.
function multiplyMat4ColumnMajor(a: Float32Array, b: Float32Array): Float32Array {
  const out = new Float32Array(16);
  for (let col = 0; col < 4; col++) {
    for (let row = 0; row < 4; row++) {
      let sum = 0;
      for (let k = 0; k < 4; k++) {
        sum += (a[k * 4 + row] ?? 0) * (b[col * 4 + k] ?? 0);
      }
      out[col * 4 + row] = sum;
    }
  }
  return out;
}

interface HybridEngineFrameDiag {
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
  /** GRIS / ReSTIR-PT reconnection-shift reuse gate (0 = legacy reuse, 1 =
   *  unbiased GRIS shift + visibility + pairwise MIS). Splatted into
   *  pipeline.renderFrame as `restirPtReuse` for the UBO. NOTE: the GI pipeline
   *  STRUCTURE (the @group(1) scene group + GRIS shader) is gated at COMPILE
   *  time in `pipeline.initialize` — this per-frame number only drives the UBO
   *  field (telemetry/consistency). Lives in this cluster for the same
   *  derived-gate (not scalar tunable) reason as `stainedGlassFlags`. */
  restirPtReuse: number;
  /** NRC (Müller et al. 2021) cache flag (0 = off / verbatim DDGI suffix, 1 =
   *  on). Splatted into pipeline.renderFrame as `nrcEnabled`. Same derived-gate
   *  cluster rationale as `restirPtReuse`. The load-bearing gate is compile-time
   *  (the risGiNrc variant); when ON the suffix cache-query + training are live. */
  nrcEnabled: number;
}

/** Subsystem handles (pipeline, BVH, GI, skinning). */
export interface HybridEngineFrameSubsystems {
  pipeline: WalkaroundGPUPipeline | null;
  bvhBuffers: SceneBVHBuffers | null;
  ddgi: DDGI;
  rc: RCSubsystem | null;
  skinning: GpuSkinningSubsystem | null;
  lastScene: Scene | null;
}

/** Canvas + internal render dimensions for this frame. */
export interface HybridEngineFrameDims {
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
}

/** Write-back closures and frame-rate control. */
export interface HybridEngineFrameControl {
  targetFrameIntervalMs: number | null;
  getLastFrameTs: () => number;
  setLastFrameTs: (ts: number) => void;
  /** Honour `FrameInput.quality.resolutionFactor` by (debounced) resizing the
   *  internal render resolution. Returns the internal dims to dispatch at this
   *  frame (which may be unchanged if the resize was debounced or the factor
   *  was unchanged). Pure-ish: the only side effect is `pipeline.resize` +
   *  engine internal-dim bookkeeping, both owned by the engine. */
  applyResolutionFactor: (factor: number | undefined, nowMs: number) => { width: number; height: number };
  runSkinning: () => void;
  presentLastFrame: (view: GPUTextureView) => void;
}

/** Telemetry subscribers, debug timings, and debug surface. */
export interface HybridEngineFrameTelemetry {
  frameSubs: ReadonlyArray<(stats: FrameStats) => void>;
  /** T3.E — long-running progress subscribers (`'ddgi-warmup'` +
   *  `'denoiser-converge'`). Fired at the end of each dispatched frame,
   *  once per still-converging kind. Empty array ⇒ zero progress work. */
  progressSubs: ReadonlyArray<(progress: ProgressStats) => void>;
  verbose: boolean;
  debugTimings: Array<{ t: number; ms: number }>;
  debugSurface: EngineDebugSurface;
  dbg: HybridEngineFrameDiag | null;
  /**
   * Returns the current active-denoiser state for `FrameStats.denoiserState`
   * population. Called once per frame only when there are `frameSubs`
   * subscribers. The pipeline exposes this via
   * `WalkaroundGPUPipeline.getActiveDenoiserState()`; null = pipeline not yet
   * initialised.
   */
  getDenoiserState: () => import('./pipeline/denoisers/index.js').DenoiserState | null;
}

/** Engine state flags, device handle, and per-frame tuning. */
export interface HybridEngineFrameFlags {
  state: EngineState;
  debug: boolean;
  ddgiOn: boolean;
  isLayerEnabled: (layer: string) => boolean;
  device: GPUDevice;
  tunables: Tunables;
  rcWeight: number;
}

/**
 * Per-frame dependency bundle for {@link runHybridEngineFrame}.
 *
 * Fields are grouped into named sub-objects so each sprint's new dependency
 * lands in the right semantic bucket. All values carry IDENTICALLY through the
 * frame loop — this is TypeScript grouping only, no semantic changes.
 */
export interface HybridEngineFrameDeps {
  /** Core subsystem handles (pipeline, BVH, GI, skinning). */
  subsystems: HybridEngineFrameSubsystems;
  /** Runtime-mutable lighting cluster (mutated by `updateLighting()`). */
  lighting: HybridLightingDeps;
  /** Denoiser filter parameters and stained-glass/NRC/GRIS gates. */
  filter: HybridDenoiserFilterDeps;
  /** Telemetry subscribers, debug timings, and debug surface. */
  telemetry: HybridEngineFrameTelemetry;
  /** Canvas + internal render dimensions. */
  dims: HybridEngineFrameDims;
  /** Write-back closures and frame-rate control. */
  control: HybridEngineFrameControl;
  /** Engine state, device handle, tuning knobs, and per-frame flags. */
  flags: HybridEngineFrameFlags;
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

/**
 * D2.9 — dispatch one RC frame and push the resulting inputs into the
 * pipeline. Extracted from `runDdgiAndRc` to isolate the 55-line RC assembly.
 *
 * Responsibilities:
 *  - syncs analytic lights into RC (A7, idempotent)
 *  - resolves chromatic sun color from the scene directional emitter
 *  - forwards rect-area emitters + env bindings from the main pipeline
 *  - calls `rc.dispatchFrame` then `pipeline.setRCInputs`
 *  - sets `pipeline.setRCInputs(null)` when RC is absent
 */
function dispatchRcAndSetInputs(
  deps: HybridEngineFrameDeps,
  input: FrameInput,
  pipeline: WalkaroundGPUPipeline,
): void {
  if (deps.subsystems.rc) {
    // Share the main pipeline's rect-area emitter buffer into RC so its probe
    // cast can NEE-sample the emitter list (closes the RC out-of-model regime
    // gap — RC otherwise saw only sun + emissive geometry + env). World-space
    // triangles ⇒ the same buffer is valid for RC's BVH; null ⇒ RC keeps its
    // prior light model.
    const rcEmitters = pipeline.getEmitterBufferAndCount();

    // A7 (2026-06-10): sync analytic point/spot lights into RC.
    // `updateLights` is idempotent and cheap (only re-uploads when the lights
    // array changes). Forward the same DDGILight list that DDGI uses so RC and
    // DDGI always agree on the fixture set. Null scene → empty list.
    const scene = deps.subsystems.lastScene;
    if (scene != null) {
      const ddgiLights = coreEmittersToDDGILights(scene);
      deps.subsystems.rc.updateLights(ddgiLights);
    }

    // A7 (2026-06-10): chromatic sun color from the scene directional emitter.
    // The prior code packed [I, I, I] (scalar intensity → achromatic grey),
    // losing the emitter's real RGB. Mirror DDGI's path (packDDGIProbeLights
    // packs col.r/g/b from the scene directional, defaulting to warm-white
    // (1, 0.95, 0.85) when absent). Extract from the core scene's first
    // `directional` emitter when available; fall back to the scalar tint.
    const dirEmitter = scene?.emitters.find((e) => e.kind === 'directional');
    const I = deps.lighting.primaryLightIntensity;
    const sunColor: [number, number, number] = dirEmitter != null
      ? [
          dirEmitter.color[0] * I,
          dirEmitter.color[1] * I,
          dirEmitter.color[2] * I,
        ]
      : [I, I, I];  // legacy fallback: achromatic (no scene directional)

    // A7: env texture forwarded from the main pipeline so the last-cascade
    // env sample reads the real HDRI (or the 1×1 black placeholder when
    // no HDRI is active — byte-identical env-less).
    const rcEnvBindings = pipeline.getEnvBindings();

    deps.subsystems.rc.dispatchFrame({
      sunDirection: deps.lighting.primaryLightDir,
      sunColor,
      frameSeed: input.frameSeed,
      triIntersectEpsilon: deps.flags.tunables.triIntersectEpsilon,
      ...(rcEmitters != null
        ? { emittersBuf: rcEmitters.buffer, emitterCount: rcEmitters.count }
        : {}),
      // A7: forward env texture so RC env sampling is live (placeholder if null).
      ...(rcEnvBindings != null
        ? { envTextureView: rcEnvBindings.textureView, envSampler: rcEnvBindings.sampler }
        : {}),
    });
    pipeline.setRCInputs(deps.subsystems.rc.buildRCInputs(deps.flags.rcWeight));
  } else {
    pipeline.setRCInputs(null);
  }
}

function runDdgiAndRc(deps: HybridEngineFrameDeps, input: FrameInput): void {
  const ddgiLayerOn = deps.flags.ddgiOn && deps.flags.isLayerEnabled('ddgi');
  const coreScene = deps.subsystems.lastScene;

  // Per-frame BVH ⇒ GI-subsystem cascade — same owner the post-update /
  // post-publish paths use, but tlas-sync-only (allowRcSceneRebuild=false so
  // merged-mode RC is NOT rebuilt every frame), and the DDGI sync gated on the
  // resolved DDGI layer exactly as before.
  propagateBvhToGiSubsystems({
    ddgi: deps.subsystems.ddgi,
    rc: deps.subsystems.rc,
    bvhBuffers: deps.subsystems.bvhBuffers,
    lastScene: coreScene,
    syncDdgi: ddgiLayerOn,
    allowRcSceneRebuild: false,
  });

  if (ddgiLayerOn) {
    deps.subsystems.ddgi.setSkyParams?.(deps.lighting.skyTint, deps.lighting.skyIrradiance);
    void deps.subsystems.ddgi.updateFrame({
      ...(coreScene != null ? { coreScene } : {}),
      device: deps.flags.device,
      enabled: true,
    });
    if (deps.subsystems.ddgi.ready) {
      deps.subsystems.ddgi.setGlassMixScale(deps.flags.tunables.glassMixScale);
    }
  }

  const pipeline = deps.subsystems.pipeline!;
  dispatchRcAndSetInputs(deps, input, pipeline);

  if (!ddgiLayerOn) {
    pipeline.setDDGIInputs(null);
  } else if (deps.subsystems.ddgi.ready) {
    const atlas = deps.subsystems.ddgi.getReadAtlasGPUTextures();
    if (atlas) {
      pipeline.setDDGIInputs({
        irradianceTex: atlas.irradiance,
        visibilityTex: atlas.visibility,
        gridParams: packDDGIGridParams(deps.subsystems.ddgi.gridParams),
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
  if (deps.telemetry.progressSubs.length === 0) return;

  const events: ProgressStats[] = [];

  // DDGI warm-up — only meaningful while the ddgi layer is active.
  if (deps.flags.ddgiOn && deps.flags.isLayerEnabled('ddgi')) {
    const warm = computeDdgiWarmupProgress({
      frame: deps.subsystems.ddgi.warmupFrame,
      stride: deps.subsystems.ddgi.warmupStride,
      ready: deps.subsystems.ddgi.ready,
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
    for (const sub of deps.telemetry.progressSubs) {
      try {
        sub(event);
      } catch (err) {
        if (deps.telemetry.verbose) console.warn('[HybridEngine] onProgress subscriber threw', err);
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
  if (deps.telemetry.frameSubs.length > 0) {
    const gpu = pipeline.lastGpuTimings;
    const gpuTotal = gpu?.['total'];
    const memBreakdown = deps.telemetry.debugSurface.estimatedGpuMemoryBytes?.() ?? undefined;
    const denoiserState = deps.telemetry.getDenoiserState();
    const stats: FrameStats = {
      frameTimeMs: dt,
      ...(gpuTotal !== undefined ? { gpuTimeMs: gpuTotal } : {}),
      ...(gpu ? { passTimings: gpu } : {}),
      spp: 1,
      ...(memBreakdown
        ? { gpuMemoryBytes: memBreakdown, estimatedGpuMemoryBytes: memBreakdown.total }
        : {}),
      // Populate denoiserState when the active denoiser has a state.
      // `reason` is normalised to null (the core contract field) when absent.
      ...(denoiserState != null
        ? {
            denoiserState: {
              status: denoiserState.status,
              reason: denoiserState.reason ?? null,
              ...(denoiserState.retryable !== undefined
                ? { retryable: denoiserState.retryable }
                : {}),
            },
          }
        : {}),
    };
    for (const sub of deps.telemetry.frameSubs) {
      try {
        sub(stats);
      } catch (err) {
        if (deps.telemetry.verbose) console.warn('[HybridEngine] onFrame subscriber threw', err);
      }
    }
  }

  emitProgressTelemetry(deps, pipeline);

  if (deps.flags.debug) {
    deps.telemetry.debugTimings.push({ t: now, ms: dt });
    if (deps.telemetry.debugTimings.length > 240) deps.telemetry.debugTimings.shift();

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
  if (deps.flags.state === 'paused' || deps.flags.state === 'disposed' || deps.flags.state === 'error') {
    return HYBRID_FRAME_SKIP_OUTPUT;
  }

  // Rebuild-key check moved to HybridEngine.renderFrame() (D2.5 — R3 B-chain
  // step 5). Engine-state mutation (reset()) no longer lives in the deps bundle.

  const dbg = deps.telemetry.dbg;
  const pipeline = deps.subsystems.pipeline;
  const bvh = deps.subsystems.bvhBuffers;
  // H20-A — empty-scene-ready: the engine is 'ready' with a retained Scene that
  // carries zero primitives, so the init chain skipped BVH/pipeline build (see
  // HybridEngineLifecycle "empty scene; ready without BVH/pipeline"). Present a
  // flat sky-only frame rather than skipping forever. Distinguished from
  // "pipeline not yet built" (a non-empty scene mid-init) by the retained
  // scene's primitive count: only a genuinely empty scene takes this path.
  const isEmptySceneReady =
    pipeline == null &&
    bvh == null &&
    deps.flags.state === 'ready' &&
    deps.subsystems.lastScene != null &&
    deps.subsystems.lastScene.primitives.length === 0;
  if (isEmptySceneReady) {
    const skyView = input.swapChainView as GPUTextureView | undefined;
    if (!skyView) {
      if (dbg) dbg.skipNoSwapView++;
      return HYBRID_FRAME_SKIP_OUTPUT;
    }
    if (dbg) dbg.framesDispatched++;
    return presentSkyOnly(
      deps.flags.device,
      skyView,
      deps.lighting.skyTint,
      deps.lighting.skyIrradiance,
    );
  }
  if (!pipeline) {
    if (dbg) dbg.skipNoPipeline++;
    return HYBRID_FRAME_SKIP_OUTPUT;
  }
  if (!bvh) {
    // H20 — bvh is null when the engine is in a transient no-geometry state
    // (e.g. mid-init after a topology change). The engine is 'ready' but
    // presents nothing: renderFrame returns SKIP every call until the BVH
    // publishes. (The genuinely-empty-scene case is handled above by the
    // sky-only present.)
    if (dbg) dbg.skipNoBvh++;
    return HYBRID_FRAME_SKIP_OUTPUT;
  }

  const now = performance.now();
  if (
    deps.control.targetFrameIntervalMs !== null &&
    deps.control.getLastFrameTs() !== 0 &&
    now - deps.control.getLastFrameTs() < deps.control.targetFrameIntervalMs
  ) {
    if (dbg) dbg.skipFrameInterval++;
    const skipSwapView = input.swapChainView as GPUTextureView | undefined;
    if (skipSwapView) {
      deps.control.presentLastFrame(skipSwapView);
    }
    return HYBRID_FRAME_SKIP_OUTPUT;
  }
  deps.control.setLastFrameTs(now);

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

  deps.control.runSkinning();
  runDdgiAndRc(deps, input);

  // §5.1 — honour per-frame `quality.resolutionFactor` by (debounced) scaling
  // the internal render resolution. The composite pass upscales the
  // internal-sized resolvedTexture to the full swap-chain view, so the
  // compute kernels + UBO `screenSize` use the internal dims (not the canvas
  // dims). Canvas resizes still require `setSize()` (see renderFrame JSDoc).
  const internal = deps.control.applyResolutionFactor(input.quality?.resolutionFactor, now);
  const viewMatrix = new Float32Array(input.viewMatrix);
  const projMatrix = new Float32Array(input.projMatrix);
  const prevViewMatrix = new Float32Array(input.prevViewMatrix ?? input.viewMatrix);
  const prevProjMatrix = new Float32Array(input.prevProjMatrix ?? input.projMatrix);

  pipeline.renderFrame({
    camera: {
      viewMatrix,
      projMatrix,
      prevViewProjMatrix: multiplyMat4ColumnMajor(prevProjMatrix, prevViewMatrix),
      cameraPos: input.cameraPosition as [number, number, number],
    },
    screen: {
      screenWidth:    internal.width,
      screenHeight:   internal.height,
      frameSeed:      input.frameSeed,
      swapChainView:  swapView,
      swapChainFormat: swapFmt,
    },
    lighting: {
      totalEmissivePower:  bvh.totalEmissivePower ?? 1.0,
      emitterCount:        bvh.emitters?.count ?? 0,
      primaryLightDir:     deps.lighting.primaryLightDir,
      primaryLightIntensity: deps.lighting.primaryLightIntensity,
      skyTint:             deps.lighting.skyTint,
      skyIrradiance:       deps.lighting.skyIrradiance,
      emitterDist2Floor:   deps.flags.tunables.emitterDist2Floor,
      directFireflyClamp:  deps.flags.tunables.directFireflyClamp,
      causticBoost:        deps.flags.tunables.causticBoost,
      causticVisClamp:     deps.flags.tunables.causticVisClamp,
      // Light-tree DI light SELECTION gate. Default ON whenever the scene has
      // ≥ 2 emitters (set at BVH build); RIS falls back to the flat power-CDF
      // path otherwise. Unbiased in both states (the WRS weight divides p̂ by
      // the exact selection pdf).
      lightTreeEnabled:    bvh.lightTreeEnabled ? 1 : 0,
      lightTreeNodeCount:  bvh.lightTreeNodeCount ?? 0,
    },
    restirDI: {
      temporalMClampDI:    deps.flags.tunables.temporalMClampDI,
      spatialReuseRadiusPx: deps.flags.tunables.spatialReuseRadiusPx,
      spatialDepthTolFloor: deps.flags.tunables.spatialDepthTolFloor,
    },
    restirGI: {
      restirGiWCap:              deps.flags.tunables.restirGiWCap,
      restirGiIrrClamp:          deps.flags.tunables.restirGiIrrClamp,
      restirGiMClamp:            deps.flags.tunables.restirGiMClamp,
      restirGiSpatialRadiusPx:   deps.flags.tunables.restirGiSpatialRadiusPx,
      restirGiSpatialNormalDotMin: deps.flags.tunables.restirGiSpatialNormalDotMin,
      restirGiSpatialCoplanarTol: deps.flags.tunables.restirGiSpatialCoplanarTol,
      restirPtReuse:             deps.filter.restirPtReuse,
    },
    gtao: {
      gtaoRadiusPx:                deps.flags.tunables.gtaoRadiusPx,
      gtaoIntensity:               deps.flags.tunables.gtaoIntensity,
      gtaoDepthThreshold:          deps.flags.tunables.gtaoDepthThreshold,
      gtaoBilateralDepthSigma:     deps.flags.tunables.gtaoBilateralDepthSigma,
      adaptiveSamplingThresholdLow:  deps.flags.tunables.adaptiveSamplingThresholdLow,
      adaptiveSamplingThresholdHigh: deps.flags.tunables.adaptiveSamplingThresholdHigh,
    },
    filter: {
      triIntersectEpsilon:  deps.flags.tunables.triIntersectEpsilon,
      glassMixScale:        deps.flags.tunables.glassMixScale,
      indirectFireflyClamp: deps.filter.indirectFireflyClamp,
      atrousDirectSigmas:   deps.filter.atrousDirectSigmas,
      atrousIndirectSigmas: deps.filter.atrousIndirectSigmas,
      stainedGlassFlags:    deps.filter.stainedGlassFlags,
    },
    bvh: {
      bvhMode:      bvh.bvhMode === 'tlas' ? 1 : 0,
      tlasNodeCount: bvh.tlas?.nodeCount ?? 0,
    },
    nrc: {
      nrcEnabled: deps.filter.nrcEnabled,
    },
    // 2026-06-10 — FrameQualitySettings.tonemap / .exposure / .outputColorSpace.
    // Defaults: 'aces' (mode 0), exposure 1.0, 'srgb' (colorSpace 0) — preserving
    // the historical hardcoded behavior bit-for-bit when quality fields are unset.
    //
    // Default-vs-contract audit: frame.ts defaults are 'aces' + 1.0 + 'srgb' — all
    // matching the historical composite behavior, so there is NO tension between the
    // contract default and the prior hardcoded path.
    //
    // outputColorSpace: 'display-p3' is not in the contract ('srgb' | 'linear' only),
    // so no console.warn needed here. If a future extension adds 'display-p3', wire it
    // to a warn-once + fallback to 'srgb' at this boundary.
    composite: {
      tonemapMode:      TONEMAP_MODE_INDEX[input.quality?.tonemap ?? 'aces'],
      exposure:         input.quality?.exposure ?? 1.0,
      outputColorSpace: input.quality?.outputColorSpace === 'linear' ? 1 : 0,
    },
  });

  emitFrameTelemetry(deps, pipeline, performance.now() - t0, now);

  // Aux G-buffers (EngineCapabilities.supportsAuxBuffers): expose the always-
  // allocated normal-depth / demodulated-albedo / motion-vector views so hosts
  // can feed an external denoiser (e.g. OIDN) or post chain. Fresh views owned
  // by the pipeline — invalidated on the next setScene / resize / dispose.
  const aux = pipeline?.getAuxBufferTextures?.() ?? null;
  return {
    kind: 'rendered',
    primaryRadiance: asBackendTexture<'webgpu', GPUTextureView>(swapView),
    ...(aux != null
      ? {
          normalDepth: asBackendTexture<'webgpu', GPUTextureView>(aux.normalDepth),
          albedo: asBackendTexture<'webgpu', GPUTextureView>(aux.albedo),
          motionVectors: asBackendTexture<'webgpu', GPUTextureView>(aux.motionVectors),
        }
      : {}),
    samplesAccumulated: 1,
    isConverged: false,
  };
}
