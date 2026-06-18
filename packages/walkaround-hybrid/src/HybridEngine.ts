/**
 * HybridEngine — WebGPU layered DDGI + ReSTIR DI engine.
 *
 * Class-based extraction of `useHybridLayeredGI.ts`. All React hooks stripped:
 *   - useRef      → private class fields
 *   - useState    → private class fields
 *   - useEffect   → initialize() + dispose() + setScene()
 *   - useFrame    → renderFrame()
 *   - useThree    → GPUDevice + canvas dimensions passed via factory opts
 *
 * Implements `@vitrum/core`'s `Engine` interface so a host can swap this
 * backend interchangeably with the native path-tracing backends.
 *
 * RC subsystem: cascade dispatch + shade-pass balance-heuristic MIS shipped
 * (W8 Phase 3). See plan/w8-rc-mis-composition.md.
 *
 * Debug globals:
 *   The original hook wrote to `window.__WGPU__.walkaround` and
 *   `window.__HYBRID_LAYERS__`. Those are host-bridge responsibilities.
 *   This class exposes `setLayerEnabled()` so the host can wire layer
 *   toggles; it calls `window.__WGPU__` only inside a debug branch
 *   guarded by `typeof window !== 'undefined'` and the `debug` option.
 *
 * Decomposition (refactor sweep 2026-05-18):
 *   - {@link PipelineInitCoordinator} (HybridEngineLifecycle.ts) owns the
 *     async pipeline-init race coordination + the multi-phase init chain.
 *   - {@link transformRefit} / {@link topologyRebuild}
 *     (HybridEnginePrimitiveUpdates.ts) implement the `updatePrimitive`
 *     fast / rebuild paths.
 *   - {@link TUNABLE_DEFINITIONS} (HybridEngineTuning.ts) is the single
 *     source of truth for audit-driven tuning knobs.
 *   - This file owns: public Engine API impl, construction-time options
 *     validation, debug surface, engine-state machine and reset
 *     coordination, scene synthesis + ownership, per-frame DDGI
 *     orchestration + frame throttle + telemetry mirror.
 */

import type {
  AnalyticShape,
  CapturedFrame,
  CaptureFrameOptions,
  Engine,
  EngineCapabilities,
  EngineDebugSurface,
  EngineError,
  EngineFactory,
  EngineState,
  EngineWarning,
  FrameStats,
  ProgressStats,
} from '@vitrum/core';
import type { Scene, ScenePrimitive, SceneEmitter, SceneEnvironment } from '@vitrum/core';
import { BACKEND_PROMISE_LEDGER, analyticPrimitiveToMesh, partitionSceneBySupport } from '@vitrum/core';
import type { FrameInput, FrameOutput } from '@vitrum/core';
import { asBackendTexture } from '@vitrum/core';
import type { BackendTexture } from '@vitrum/core';
import { DDGI } from './ddgi/DDGI.js';
import type { DDGILight } from './ddgi/types.js';
import { WalkaroundGPUPipeline } from './pipeline/WalkaroundGPUPipeline.js';
import {
  parseHybridEngineOptions,
  validateHybridEngineOptions,
  deriveHybridEngineConfig,
  type ParsedHybridEngineConfig,
} from './HybridEngineConfig.js';
import { createHybridEngineDebugSurface } from './HybridEngineDebug.js';
import type { PickCamera } from '@vitrum/shared-bvh';
import {
  fingerprintHybridPipelineRebuildKey,
  getPreferredSwapChainFormat,
  resolveInternalRenderSize,
  runHybridEngineFrame,
  HYBRID_FRAME_SKIP_OUTPUT,
  type HybridEngineFrameDeps,
  type HybridLightingDeps,
  type HybridDenoiserFilterDeps,
} from './HybridEngineFrameOrchestrator.js';
import {
  rebuildEmitterBuffersFromCoreScene,
  disposeSceneBVH,
  type ReSTIRBvhMode,
  type SceneBVHBuffers,
} from './restir/bvhCore.js';
import type { MaterialTextureAtlasDiagnostic } from './pipeline/materialTextureAtlas.js';
import { applyEmitterPatchToScene, applyPrimitivePatchToScene } from './scenePatch.js';
import { solveSkin } from '@vitrum/core';
import { readRgba16fWalkaround } from './util/gpuReadback.js';
import { SHADE_FLAG_DIRECT_SUN_SHADOW_DISABLED } from './pipeline/uboUpdater.js';
import {
  transformRefit,
  positionsRefit,
  topologyRebuild,
  materialPatch,
  skinnedPosePatch,
  refitSkinnedMeshAfterGpuWrite,
  SKIN_POSE_PATCH_FIELDS,
  TOPOLOGY_PATCH_FIELDS,
  TOPOLOGY_PATCH_WHOLESALE_FIELDS,
  type PrimitiveUpdateContext,
  type PrimitiveUpdateResult,
} from './HybridEnginePrimitiveUpdates.js';
import {
  PipelineInitCoordinator,
  type PipelineInitHost,
  type HybridInitStaticConfig,
} from './HybridEngineLifecycle.js';
import type { Tunables } from './HybridEngineTuning.js';
import {
  deriveScaleAwareClamps,
  type ScaleAwareHostExplicit,
} from './HybridEngineScaleAwareClamps.js';
import { FrameBudgetController } from './FrameBudgetController.js';
import type { FrameBudgetControllerConfig, FrameBudgetDecision } from './FrameBudgetController.js';
import type { HybridEngineOptions, LightingOptions } from './HybridEngineOptions.js';
import { assertKnownLightingKeys } from './HybridEngineOptions.js';
import {
  collectApproximateAlphaBlendPrimitiveIds,
  collectApproximateEmissiveMapTexelPdfPrimitiveIds,
  collectUnconsumedMaterialFields,
  collectUnconsumedMaterialFieldsForMaterial,
} from './restir/consumedMaterialFields.js';
import { RCSubsystem } from './HybridEngineRC.js';
import { propagateBvhToGiSubsystems } from './HybridEngineGiPropagation.js';
import {
  directionalSunMultiplier,
  orientDdgiSunLights,
} from './coreEmittersToDDGILights.js';
import { GpuSkinningSubsystem } from './skin/GpuSkinningSubsystem.js';
import type { GIStateSnapshot } from './giStateSnapshot.js';
import type { HybridEngineGISurface } from './HybridEnginePublic.js';
import {
  resolveHybridEnvironment,
  type HybridEnvironmentResolverExtensions,
} from './environment/resolveHybridEnvironment.js';
import {
  exportGIStateImpl,
  importGIStateImpl,
} from './HybridEngineGIState.js';
import { syncDdgiFromCoreScene } from './HybridEngineDdgiSync.js';

function sceneWithAnalyticMeshFallback(scene: Scene): Scene {
  let changed = false;
  const primitives = scene.primitives.map((primitive) => {
    if (primitive.kind !== 'analytic') return primitive;
    changed = true;
    return analyticPrimitiveToMesh(primitive);
  });
  return changed ? { ...scene, primitives } : scene;
}

function collectAuthoredDirectionalAngularDiameters(scene: Scene): Array<{
  id: string;
  angularDiameter: number;
  sunAngularRadius: number;
}> {
  const out: Array<{ id: string; angularDiameter: number; sunAngularRadius: number }> = [];
  for (const emitter of scene.emitters) {
    if (emitter.kind !== 'directional') continue;
    const angularDiameter = emitter.angularDiameter;
    if (typeof angularDiameter !== 'number' || !Number.isFinite(angularDiameter)) continue;
    out.push({
      id: String(emitter.id),
      angularDiameter,
      sunAngularRadius: Math.max(0, angularDiameter) * 0.5,
    });
  }
  return out;
}

// Re-export the option / lighting interfaces from their dedicated module so
// the package's public surface (`./HybridEngine.js` import path) stays
// unchanged after the type split (refactor sweep 2026-05-18).
export type { HybridEngineOptions, LightingOptions } from './HybridEngineOptions.js';

// ────────────────────────────────────────────────────────────────────────────
// Option parsing + validation
//
// `ParsedHybridEngineConfig`, `validateHybridEngineOptions`,
// `deriveHybridEngineConfig`, and `parseHybridEngineOptions` have been moved to
// `HybridEngineConfig.ts` (R3 B-chain decomposition sweep). Re-imported above.
//
// `HybridEngineOptions` + `LightingOptions` interface bodies live in
// `HybridEngineOptions.ts` (~340 LOC of pure JSDoc, extracted refactor sweep
// 2026-05-18). Re-exported above so the package surface is unchanged.
// ────────────────────────────────────────────────────────────────────────────

// Unused import guard: `validateHybridEngineOptions` and `deriveHybridEngineConfig`
// are exported by HybridEngineConfig.ts and re-exported from this file for
// back-compat (any external caller that imported them from HybridEngine.ts directly).
export { validateHybridEngineOptions, deriveHybridEngineConfig };

// `readRgba16fWalkaround` moved to `src/util/gpuReadback.ts` (R3 B-chain step 2).
// Re-imported above.

function buildWalkaroundExperimentalFeatures(cfg: ParsedHybridEngineConfig): ReadonlySet<string> {
  const features = new Set<string>(['svgf-real-conservative-objid']);
  if (cfg.restirPtReuse === 1) features.add('walkaround-hybrid-gris-unbiased-reuse');
  if (cfg.ppgEnabled === 1) features.add('walkaround-hybrid-ppg-guided-gi');
  if (cfg.nrcEnabled === 1) features.add('walkaround-hybrid-nrc-biased-cache');
  if (cfg.denoiser === 'neural') features.add('walkaround-hybrid-neural-denoiser-host-weights');
  return features;
}

// ────────────────────────────────────────────────────────────────────────────
// HybridEngine
// ────────────────────────────────────────────────────────────────────────────

export class HybridEngine implements Engine {

  // ── Engine contract fields ─────────────────────────────────────────────
  private _state: EngineState = 'uninitialized';
  readonly capabilities: EngineCapabilities;

  get state(): EngineState {
    return this._state;
  }

  // ── Creation-time options (immutable after construction) ───────────────
  private readonly _device:               GPUDevice;
  // Mutable since T-resize: the host calls `setSize()` whenever the
  // canvas resizes; the pipeline reallocates its FrameResources without
  // a full engine teardown. See `setSize()` for the resize contract.
  //
  // Phase-0 productization — `_width/_height` are the CANVAS (swap-chain)
  // dimensions (what the composite pass blits TO + what `setSize` sets). The
  // INTERNAL render resolution (what the compute kernels dispatch over) is
  // `_internalWidth/_internalHeight = canvas × _resolutionFactor`; it equals
  // the canvas dims when no `quality.resolutionFactor` downscale is active.
  // The two are kept in sync by `setSize` (recomputes internal from the last
  // factor) and by the per-frame `quality.resolutionFactor` path in
  // `HybridEngineFrameOrchestrator` (debounced internal resize).
  private _width:                number;
  private _height:               number;
  /** Fires the one-time `FrameInput.viewport`-ignored dev warning at most once
   *  per engine instance (see {@link renderFrame}). */
  private _viewportMismatchWarned = false;
  /** Fires environment-resolution warnings at most once per engine instance
   *  (see {@link _skyScalarsFromEnvironment}). */
  private _proceduralSkyWarned = false;
  /** Tracks which unconsumed-material-field sets have already been warned about
   *  (keyed by sorted join of the field names). Prevents duplicate console.warn
   *  calls across incremental `setScene` calls with the same ignored fields. */
  private _warnedMaterialFields = new Set<string>();
  /** Tracks which fractional alpha-blend primitive sets have already warned. */
  private _warnedAlphaBlendApproximationIds = new Set<string>();
  /** Tracks which emissive-map texel-PDF approximation primitive sets have warned. */
  private _warnedEmissiveMapTexelPdfApproximationIds = new Set<string>();
  /** Tracks atlas-backed material texture drops already reported to hosts. */
  private _warnedMaterialTextureAtlasDiagnostics = new Set<string>();
  /** Tracks authored directional angular-diameter partial-support warnings. */
  private _warnedDirectionalAngularDiameterApproximationIds = new Set<string>();
  /** Internal render width = `_width × _resolutionFactor`. Drives compute
   *  dispatch + UBO `screenSize`; the composite upscales to `_width`. */
  private _internalWidth:        number;
  /** Internal render height = `_height × _resolutionFactor`. */
  private _internalHeight:       number;
  /** Last-seen `FrameInput.quality.resolutionFactor` (clamped to (0,1]).
   *  Default 1.0 (internal == canvas). */
  private _resolutionFactor:     number = 1.0;
  /** `performance.now()` of the last accepted resolution-factor resize, for
   *  the debounce that prevents accumulator thrash (Risk R5). */
  private _lastResolutionResizeTs: number = 0;
  /** Optional host environment resolver extension. Used only by
   *  updateEnvironment() to reduce opaque HDRI handles into the diffuse
   *  sky-dome scalars this backend consumes. */
  private readonly _environmentResolverExtensions: HybridEnvironmentResolverExtensions | null;
  /** Optional host readiness predicate. Core mesh presence remains required. */
  private readonly _isSceneReady:         () => boolean;
  // Lighting fields are NOT readonly — updateLighting() mutates them at runtime.
  private _primaryLightDir:               [number, number, number];
  private _primaryLightIntensity:         number;
  private _skyTint:                       [number, number, number];
  private _skyIrradiance:                 number;
  private readonly _ctorLights:           readonly DDGILight[];

  /** Rolling window of per-frame timings (newest last, cap 240 entries).
   *  Only populated when `debug === true`. Hosts that want a UI gauge
   *  should poll {@link debugTimings} instead of reaching into globals. */
  private readonly _debugTimings: Array<{ t: number; ms: number }> = [];

  /** T3.E — telemetry subscribers fired at end of each successful renderFrame. */
  private readonly _frameSubs: Array<(s: FrameStats) => void> = [];

  /** T3.E — long-running progress subscribers fired at the end of each
   *  dispatched frame, once per still-converging signal (`'ddgi-warmup'`
   *  while the probe grid warms, `'denoiser-converge'` while the temporal
   *  accumulator fills). Empty until a host calls {@link onProgress}. */
  private readonly _progressSubs: Array<(p: ProgressStats) => void> = [];

  /** GPU error subscribers (item 28). */
  private readonly _errorSubs: Array<(e: EngineError) => void> = [];

  /** Structured non-fatal warning subscribers. */
  private readonly _warningSubs: Array<(w: EngineWarning) => void> = [];
  /** Dedup-throttle: message → frame-index of last emission. */
  private _errorThrottleMap = new Map<string, number>();
  /** Frame counter for error throttle (incremented in renderFrame). */
  private _errorFrameCount = 0;
  /** Bound uncapturederror handler — stored so it can be removed on dispose. */
  private _onUncapturedError: ((e: Event) => void) | null = null;

  /** Read-only snapshot of recent frame timings collected when the engine
   *  was constructed with `debug: true`. Returns an empty array when debug
   *  is off (no allocation cost is paid in production). */
  get debugTimings(): readonly { t: number; ms: number }[] {
    return this._debugTimings;
  }

  /** Per-pass GPU timings in milliseconds, keyed by the same `PassLabel`
   *  set the timestamp-query subsystem uses. Empty record when the active
   *  adapter doesn't expose `timestamp-query` or when the engine is not yet
   *  initialised. Useful for dev panels + telemetry harnesses. */
  get lastGpuTimings(): Record<string, number> {
    return this._pipeline?.lastGpuTimings ?? {};
  }

  /** Frame index that produced {@link lastGpuTimings}; -1 if no readback
   *  has resolved yet. */
  get lastGpuTimingsFrame(): number {
    return this._pipeline?.lastGpuTimingsFrame ?? -1;
  }

  /**
   * Diagnostic one-shot GPU timing readback (P3-Vδ). Bypasses the
   * production fire-and-forget ping-pong path and synchronously awaits a
   * fresh staging-buffer mapAsync. Use this from telemetry probes that
   * need a confirmed-fresh per-pass timing snapshot. Returns empty
   * objects when the engine isn't ready or the device lacks the
   * `timestamp-query` feature.
   */
  async readGpuTimingsOnce(): Promise<{ perPass: Record<string, number>; rawBigints: string[] }> {
    if (!this._pipeline) return { perPass: {}, rawBigints: [] };
    return this._pipeline.readGpuTimingsOnce();
  }

  /**
   * Runtime DDGI probe-update divisor (round-robin stride) setter. The
   * construction-time value comes from the quality preset / `ddgiUpdateDivisor`
   * option; this lets a host (or the adaptive frame-budget controller) retune
   * the GI-refresh cadence per frame without recreating the engine. Clamped to
   * ≥ 1 by the DDGI subsystem. Cheap (a couple of JS field writes + a UBO field
   * the next probe-update pass picks up); no atlas teardown.
   */
  setDdgiUpdateDivisor(divisor: number): void {
    this._ddgi.setProbeUpdateDivisor(divisor);
  }

  /**
   * Opt-in: turn ON the closed-loop adaptive frame-budget controller (Phase
   * IV.1 / review gap D1). Idempotent — re-calling with a new config replaces
   * the controller's config but the engine still owns no cadence: NOTHING
   * happens until the host feeds measured ms via {@link tickFrameBudget}.
   *
   * Seeds the controller's initial knobs from the engine's current state (the
   * preset/option `resolutionFactor` + `ddgiUpdateDivisor`) so the loop starts
   * from where the static path left off, then backs off / climbs from there.
   *
   * The controller defaults target ~60 fps; pass `{ targetMs: 33.3 }` for 30
   * fps, etc. See {@link FrameBudgetControllerConfig}.
   */
  enableFrameBudget(config: Partial<FrameBudgetControllerConfig> = {}): void {
    this._frameBudget = new FrameBudgetController({
      adaptPpgDispatchInterval: this._cfg.ppgEnabled === 1,
      ...config,
    }, {
      resolutionFactor: this._resolutionFactor,
      ddgiStride: this._cfg.ddgiUpdateDivisor,
      ppgDispatchInterval: this._cfg.ppgDispatchInterval,
    });
  }

  /** Opt-out: turn OFF the adaptive frame-budget controller. The knobs are left
   *  wherever the loop last set them (the host may restore them explicitly). */
  disableFrameBudget(): void {
    this._frameBudget = null;
  }

  /** True when the adaptive frame-budget loop is enabled. */
  get frameBudgetEnabled(): boolean {
    return this._frameBudget !== null;
  }

  /**
   * Opt-in adaptive-quality tick. The host calls this ONCE PER FRAME with a
   * measured frame time (ms) — from the wall-clock `FrameStats.frameTimeMs` of
   * an `onFrame` subscriber, or from {@link readGpuTimingsOnce}'s confirmed GPU
   * `total` — and the controller nudges the engine's quality knobs toward the
   * configured budget.
   *
   * Consistent with "the host owns cadence", this does NOT schedule itself and
   * does NOT read frame time on its own; the host drives it. It applies the
   * engine-owned knobs (DDGI stride, and PPG train cadence when PPG is enabled)
   * directly, and returns the decision so the host can feed the PRIMARY knob
   * back as the next frame's `FrameInput.quality.resolutionFactor` (the
   * resolution lever is a per-frame host input by contract — `renderFrame`
   * consumes `quality.resolutionFactor`, debounced).
   *
   * No-op (returns `null`) when the controller is not enabled — so a host can
   * call it unconditionally in its render loop and pay nothing until it opts in
   * via {@link enableFrameBudget}.
   *
   * @returns the {@link FrameBudgetDecision} (whose `resolutionFactor` the host
   *          should apply next frame), or `null` if the loop is disabled.
   */
  tickFrameBudget(measuredMs: number): FrameBudgetDecision | null {
    if (!this._frameBudget) return null;
    const decision = this._frameBudget.update(measuredMs);
    // Apply engine-owned runtime knobs immediately; the primary lever
    // (resolutionFactor) is a host per-frame input by the FrameInput contract,
    // so the host applies it via the returned decision.
    this.setDdgiUpdateDivisor(decision.ddgiStride);
    this.setPpgDispatchInterval(decision.ppgDispatchInterval);
    return decision;
  }

  /**
   * Runtime PPG train-pass dispatch interval. Meaningful only when the engine
   * was constructed with `ppgEnabled:true`; otherwise the pipeline has no PPG
   * update pass to gate, so this is harmless. Exposed mainly for the adaptive
   * frame-budget controller and host A/B knobs.
   */
  setPpgDispatchInterval(interval: number): void {
    this._pipeline?.setPpgDispatchInterval(interval);
  }

  /**
   * The construction-time-immutable derived config (Task 4.2 / Theme A).
   * `parseHybridEngineOptions(opts)` produces this once in the constructor;
   * every tunable-cluster value the engine used to splat onto an individual
   * `this._x` field is now read from `this._cfg.x`. Holding the parsed record
   * directly collapses ~25 one-by-one ctor assignments + their field
   * declarations, so a single tunable no longer hops a private-field layer.
   *
   * ONLY construction-immutable values live here. Genuinely per-instance
   * MUTABLE runtime state (lighting, size/internal-size, accumulators, the
   * rebuild-key fingerprint, BVH/pipeline/scene handles) stays in its own
   * field below because it changes after construction.
   */
  private readonly _cfg: ParsedHybridEngineConfig;
  /** Dev A/B — mirrors `engine.debug.setDenoiserEnabled` (default on). */
  private _denoiserPassEnabled = true;

  // ── B15 — scene-scale-aware radiometric clamp defaults ──────────────────
  /** Per-knob flags: did the HOST explicitly set this clamp? Host overrides are
   *  NEVER scaled (an explicit tuning value passes through verbatim). Captured
   *  once at construction from the options. */
  private readonly _clampHostExplicit: ScaleAwareHostExplicit;
  /** Scene-scale-derived per-frame tunables (B15). Computed at `setScene` from
   *  the scene's world diagonal; `null` before the first scene (falls back to
   *  the Cornell-baseline `_cfg.tunables`). At Cornell scale this is
   *  byte-identical to `_cfg.tunables` (the law short-circuits at ratio 1). */
  private _scaledTunables: Tunables | null = null;
  /** Scene-scale-derived `indirectFireflyClamp` (B15). `null` ⇒ use the
   *  `_cfg.indirectFireflyClamp` baseline. */
  private _scaledIndirectFireflyClamp: readonly [number, number, number] | null = null;

  // ── Pipeline state ─────────────────────────────────────────────────────
  private _pipeline:    WalkaroundGPUPipeline | null = null;
  private _bvhBuffers:  SceneBVHBuffers | null       = null;

  // ── Scene (from @vitrum/core contract) ────────────────────────────────
  /** Last authored scene accepted via `setScene()`. Incremental updates patch
   *  this snapshot so analytic primitives retain their `shape` / `params`
   *  semantics even when the renderer consumes generated mesh fallbacks. */
  private _lastScene: Scene | null = null;

  /** Render-ingestion view derived from {@link _lastScene}: analytic primitives
   *  become deterministic MeshPrimitive fallbacks with the same id/material/
   *  transform. BVH, DDGI, ReSTIR, RC, and THREE conversion consume this view. */
  private _renderScene: Scene | null = null;

  /** Last-frame camera (copied) for debug click-to-pick (`pickPrimitive`, T3.G).
   *  Captured each `renderFrame`; null until the first frame. */
  private _lastFrameCamera: PickCamera | null = null;

  // ── DDGI subsystem ─────────────────────────────────────────────────────
  private _ddgi:    DDGI;
  private _ddgiOn:  boolean = true;

  // ── Adaptive frame-budget controller (opt-in; Phase IV.1 / review gap D1) ─
  /** Lazily-created on the first {@link enableFrameBudget}/{@link tickFrameBudget}
   *  call. Null ⇒ the closed loop is OFF and the static quality path is
   *  untouched (the engine never reads frame time on its own). */
  private _frameBudget: FrameBudgetController | null = null;

  // ── RC subsystem (W8 Phase 2 — opt-in via opts.rcEnabled) ───────────────
  private _rc: RCSubsystem | null = null;
  /** W8 Phase 3 — balance-heuristic MIS weight for RC in Lo_indirect.
   *  Effective only when _rc != null. Default 0.5 when rcEnabled is true. */
  private _rcWeight: number = 0;

  // ── Per-frame throttle ─────────────────────────────────────────────────
  private _lastFrameTs = 0;

  // ── Diagnostic counters (debug only) ──────────────────────────────────
  private _dbg = {
    initStart:          0,
    initCount:          0,
    disposeCount:       0,
    skipNoPipeline:     0,
    skipNoBvh:          0,
    skipNoSwapView:     0,
    skipFrameInterval:  0,
    framesDispatched:   0,
    lastReportTs:       0,
  };

  // ── Layer toggles (debug console interface) ────────────────────────────
  private _layerEnabled: Map<string, boolean> = new Map([
    ['ddgi', true],
  ]);

  // ── Pipeline init coordinator (see HybridEngineLifecycle.ts) ──────────
  //
  // Owns the monotonic init-sequence + dispose race coordination + multi-
  // phase async init chain. The engine delegates init/dispose flow to it
  // via the {@link PipelineInitHost} surface built in `_buildInitHost()`.
  private readonly _initCoordinator: PipelineInitCoordinator;

  // Task 4.2 / Theme A — the construction-immutable tunable-cluster values live
  // on `_cfg` (one parsed record instead of ~25 splatted `_x` fields). Consumers
  // read `this._cfg.x` directly; tests pin resolved knobs via the `_cfg` seam.

  /** Monotonic fingerprint of {@link HybridEngineOptions.pipelineRebuildKey} /
   *  {@link HybridEngineOptions.getPipelineRebuildKey} — changes trigger `reset()`. */
  private _rebuildKeyFingerprintSeen: string;

  private readonly _skinning: GpuSkinningSubsystem | null;

  readonly debug: EngineDebugSurface;

  constructor(opts: HybridEngineOptions) {
    // Pure option parsing + validation (defaults, denoiser/neural/OIDN
    // validation throws) lives in `parseHybridEngineOptions` so the
    // constructor body stays focused on `this`-dependent wiring (subsystems,
    // capabilities, init coordinator, debug surface). Behaviour-preserving:
    // same throws in the same order, same defaults. (WD decomposition sweep.)
    // Task 4.2 / Theme A — hold the parsed config in one `_cfg` field rather
    // than splatting its ~25 construction-immutable values onto individual
    // `this._x` fields. Consumers read `this._cfg.x`; only genuinely-mutable
    // runtime state (device handle, size, lighting, accumulators, the rebuild-
    // key fingerprint) gets its own field.
    const cfg = parseHybridEngineOptions(opts);
    this._cfg = cfg;
    if (opts.onWarning != null) {
      this._warningSubs.push(opts.onWarning);
    }

    // B15 — capture which radiometric clamps the HOST set explicitly. These
    // bypass scene-scale scaling (an explicit override is absolute). None of
    // the three scalar clamps has a subsystem sub-object, so `opts.tuning` is
    // the only override path; `indirectFireflyClamp` is its own top-level field.
    this._clampHostExplicit = {
      restirGiIrrClamp:     opts.tuning?.restirGiIrrClamp !== undefined,
      directFireflyClamp:   opts.tuning?.directFireflyClamp !== undefined,
      emitterDist2Floor:    opts.tuning?.emitterDist2Floor !== undefined,
      indirectFireflyClamp: opts.indirectFireflyClamp !== undefined,
    };

    // H46-A — maxBounces now drives a REAL control surface on this realtime
    // stack: the DDGI indirect-feedback gate. The walkaround engine does NOT
    // path-trace, so maxBounces is NOT a per-ray bounce cap here — it has exactly
    // two regimes: 1 ⇒ DIRECT-ONLY probes (the DDGI atlas folds in one bounce of
    // direct light per probe and the infinite-bounce diffuse EMA is disabled);
    // >= 2 ⇒ the full multi-bounce diffuse equilibrium (the default; the atlas
    // EMA converges to infinite diffuse bounces). Intermediate/large values
    // (2, 3, 4, …) are all identical to the default >= 2 regime because the EMA
    // converges to the bounce limit regardless of the integer value — only the
    // 1-vs-many distinction is meaningful. Warn only when the value cannot be
    // honoured as authored (== 0 or negative is treated as direct-only).
    if (cfg.maxBounces < 1) {
      this._warn({
        code: 'walkaround-hybrid.max-bounces-clamped',
        backend: 'walkaround-hybrid',
        phase: 'construction',
        method: 'createWalkaroundEngine_Hybrid',
        message:
          `[HybridEngine] maxBounces=${cfg.maxBounces} is < 1 and is treated as ` +
        `1 (direct-only DDGI probes). The walkaround engine is not a path tracer; ` +
        `maxBounces gates the DDGI diffuse multi-bounce feedback (1 = direct-only, ` +
        `>= 2 = infinite-bounce diffuse equilibrium), not a per-ray bounce count.`,
        details: { requested: cfg.maxBounces, clampedTo: 1 },
      });
    }
    // H46 — causticStrategy: walkaround always reports 'none' in capabilities.
    // Non-'none' strategies are not implemented for this backend.
    if ((opts as { causticStrategy?: string }).causticStrategy != null &&
        (opts as { causticStrategy?: string }).causticStrategy !== 'none') {
      const strategy = (opts as { causticStrategy?: string }).causticStrategy;
      this._warn({
        code: 'walkaround-hybrid.unsupported-caustic-strategy',
        backend: 'walkaround-hybrid',
        phase: 'construction',
        method: 'createWalkaroundEngine_Hybrid',
        message:
          `[HybridEngine] causticStrategy='${strategy}' ` +
        `is not supported by the walkaround-hybrid engine. ` +
        `This engine always reports causticStrategy:'none' in capabilities. ` +
        `Use pt-webgpu or pt-webgl2 for manifold-nee/photon-map caustics.`,
        details: { requested: strategy },
      });
    }
    // H46 — maxSamplesPerPixel is a converged-PT structural cap. walkaround is a
    // realtime single-frame GI engine (capabilities.accumulates=false), so there
    // is no SPP accumulator to size or clamp. Warn when a host supplies it rather
    // than silently accepting a knob this backend cannot honour.
    if (opts.maxSamplesPerPixel !== undefined) {
      this._warn({
        code: 'walkaround-hybrid.max-samples-per-pixel-ignored',
        backend: 'walkaround-hybrid',
        phase: 'construction',
        method: 'createWalkaroundEngine_Hybrid',
        message:
          `[HybridEngine] maxSamplesPerPixel=${opts.maxSamplesPerPixel} is ignored by ` +
        `walkaround-hybrid. This backend does not progressively accumulate samples ` +
        `(capabilities.accumulates=false); use FrameInput quality knobs or a path-tracing ` +
        `backend for SPP caps.`,
        details: { requested: opts.maxSamplesPerPixel },
      });
    }
    if (opts.causticOptions !== undefined) {
      this._warn({
        code: 'walkaround-hybrid.unsupported-caustic-options',
        backend: 'walkaround-hybrid',
        phase: 'construction',
        method: 'createWalkaroundEngine_Hybrid',
        message:
          `[HybridEngine] causticOptions were provided but walkaround-hybrid does ` +
        `not implement causticStrategy modes. The causticOptions object is ignored.`,
        details: { keys: Object.keys(opts.causticOptions) },
      });
    }
    if (opts.nrcEnabled === true) {
      this._warn({
        code: 'walkaround-hybrid.nrc-experimental-biased',
        backend: 'walkaround-hybrid',
        phase: 'construction',
        method: 'createWalkaroundEngine_Hybrid',
        message:
          `[HybridEngine] nrcEnabled:true enables the opt-in Neural Radiance ` +
          `Cache. NRC is a biased experimental radiance cache for realtime GI; ` +
          `the default remains disabled and hosts should validate quality before ` +
          `using it in production scenes.`,
        details: {
          nrcEnabled: true,
          defaultEnabled: false,
          estimator: 'biased',
        },
      });
    }
    if (cfg.denoiser === 'neural') {
      this._warn({
        code: 'walkaround-hybrid.neural-host-weights-required',
        backend: 'walkaround-hybrid',
        phase: 'construction',
        method: 'createWalkaroundEngine_Hybrid',
        message:
          `[HybridEngine] denoiser:'neural' is an opt-in GPU U-Net path that ` +
          `requires host-provided, scene-validated weights. The package does ` +
          `not ship production neural weights or enable neural by default.`,
        details: {
          denoiser: 'neural',
          weightsRequired: true,
          packageProvidesProductionWeights: false,
          defaultEnabled: false,
        },
      });
    }

    this._device                = opts.device;
    this._width                 = opts.width;
    this._height                = opts.height;
    // Phase-0 — the quality preset supplies the INITIAL internal-resolution
    // factor. A per-frame `quality.resolutionFactor` still overrides at runtime
    // (`_applyResolutionFactor`); this is just the starting point so a
    // `qualityTier:'low'` engine boots at 0.5 internal res.
    this._resolutionFactor      = cfg.resolutionFactor;
    this._internalWidth         = Math.max(1, Math.round(opts.width * cfg.resolutionFactor));
    this._internalHeight        = Math.max(1, Math.round(opts.height * cfg.resolutionFactor));
    this._skinning              = opts.gpuSkinning
      ? new GpuSkinningSubsystem(opts.device, true)
      : null;
    this._environmentResolverExtensions = opts.extensions ?? null;
    this._primaryLightDir       = opts.primaryLightDir;
    this._primaryLightIntensity = opts.primaryLightIntensity;
    this._skyTint               = opts.skyTint;
    this._skyIrradiance         = opts.skyIrradiance;
    this._isSceneReady          = opts.isSceneReady ?? (() => true);

    // `_rebuildKeyFingerprintSeen` is the one rebuild-key value that MUTATES
    // post-construction (`consumeRebuildKeyChange` rewrites it), so it stays a
    // mutable field seeded from the parsed config. The static key + getter live
    // on `_cfg` (immutable).
    this._rebuildKeyFingerprintSeen = cfg.rebuildKeyFingerprintSeen;

    this._ddgi = new DDGI({
      debug: this._cfg.debug,
      ...(opts.ddgiMaxMaterials !== undefined ? { maxMaterials: opts.ddgiMaxMaterials } : {}),
      onError: (error) => this._emitError(error),
      onWarning: (warning) => this._warn(warning),
    });
    this._ddgi.setSkyParams?.(this._skyTint, this._skyIrradiance);
    // Phase-0 — apply the quality-preset DDGI probe-update divisor (default 4).
    this._ddgi.setProbeUpdateDivisor(this._cfg.ddgiUpdateDivisor);
    // H46-A — gate the DDGI indirect-feedback (multi-bounce diffuse EMA) on the
    // engine's maxBounces. maxBounces == 1 ⇒ direct-only probes; >= 2 ⇒ the
    // infinite-bounce equilibrium (default). Construction-immutable, and the
    // ProbeUpdatePass is created once (never recreated), so one call persists.
    this._ddgi.setIndirectFeedback(this._cfg.maxBounces >= 2);
    this._ctorLights = opts.lights ?? [];
    if (this._ctorLights.length > 0) {
      this._ddgi.setLights(orientDdgiSunLights(this._ctorLights, this._primaryLightDir));
    }

    // W8 Phase 2 — opt-in RC subsystem. RCSubsystem owns its own BVH +
    // cascade GPUBuffers. setScene() rebuilds them when the source scene
    // changes; dispatch happens per-frame in renderFrame() below.
    if (opts.rcEnabled === true) {
      // B3b — Cornell-tuned CASCADE_DIMS default lives in walkaround-rc;
      // hosts override via opts.cascadeDims for non-Cornell aspect ratios
      // or scene scales.
      this._rc = opts.cascadeDims !== undefined
        ? new RCSubsystem(this._device, opts.cascadeDims)
        : new RCSubsystem(this._device);
      // W8 Phase 3 — host-overridable MIS weight (default 0.5 = equal
      // mix with ReSTIR-GI). When rcEnabled is false the weight stays 0
      // and pipeline.setRCInputs(null) routes the bind group to the
      // placeholder buffers (rcParams.enabled = 0u short-circuits the
      // shader's sample to vec3f(0)).
      this._rcWeight = Math.max(0, Math.min(1, opts.rcWeight ?? 0.5));
    }

    this.capabilities = {
      // Incremental scene patches are implemented: transform/positions fast
      // paths plus full-rebuild fallbacks for material/topology edits, and
      // emitter patching via scene-level rebuild.
      supportsIncrementalScene:  true,
      incrementalPatchSupport: {
        transform: true,
        positions: true,
        material: true,
        emitter: true,
        topology: true,
      },
      // Explicit whole-primitive add/remove IS implemented: addPrimitive appends
      // a new primitive and removePrimitive evicts one, each by routing a fresh
      // mutated `Scene` copy through this engine's existing `setScene` spine —
      // the same `partitionSceneBySupport` → `_teardownPipeline` → init-chain
      // (full BVH / DDGI / ReSTIR rebuild + temporal-accumulator reset) the
      // initial scene build runs. A geometry change invalidates every cached GI
      // signal, so on this realtime stack the work is a rebuild either way; the
      // value is API consistency with pt-webgl / pt-webgpu, not a perf win. The
      // DDGI / ReSTIR / RC subsystems index off the packed scene, so reusing the
      // setScene packing path re-syncs them all correct-by-construction — no
      // fragile per-array index remap. Distinct from
      // incrementalPatchSupport.topology (COUNT-change patches on an EXISTING
      // primitive). Kept in sync with the walkaround-hybrid row in
      // @vitrum/core's BACKEND_PROMISE_LEDGER.
      supportsAddRemovePrimitive: true,
      // supportsAuxBuffers is false: the contract flag means variance AND
      // motionVectors are exposed; walkaround never exposes variance from its
      // FrameOutput wiring (the Welford buffer is internal, not in FrameOutput).
      // Kept in sync with the walkaround-hybrid ledger row in @vitrum/core's
      // BACKEND_PROMISE_LEDGER (plan/v1-closure-plan-2026-06-10.md).
      supportsAuxBuffers:        false,
      // The post-denoise resolvedTexture is exposed via getProgressiveSeedTexture()
      // as the seed source for progressive walkaround→PT handoff (P8).
      supportsProgressiveSeedSource: true,
      accumulates:               false,
      maxSamplesPerPixel:        Infinity,
      // H46-A — echoes the authored value. SEMANTICS for this realtime stack:
      // this is NOT a path-tracer per-ray bounce cap. 1 ⇒ direct-only DDGI
      // probes; >= 2 ⇒ infinite-bounce diffuse equilibrium (the atlas EMA). All
      // values >= 2 behave identically (the EMA converges regardless of the
      // integer). See the construction-site gate `setIndirectFeedback`.
      maxBounces:                this._cfg.maxBounces,
      supportedAnalyticShapes:   new Set<AnalyticShape>(['sphere', 'box', 'capsule', 'cylinder', 'h-channel-came']),
      // BVH + DDGI ingest via a render-scene view. Mesh/skinned/instanced-mesh
      // flow through directly; analytic primitives are accepted in the authored
      // scene and converted to deterministic MeshPrimitive fallbacks before
      // ReSTIR / DDGI / RC consume them.
      supportedPrimitiveKinds:   new Set<ScenePrimitive['kind']>(['mesh', 'skinned-mesh', 'instanced-mesh', 'analytic']),
      // Emitter kinds that genuinely reach a renderable state:
      //   - rect-area / disc-area → harvested as ReSTIR-DI direct emitter tris
      //     from core emitter data AND projected
      //     to DDGI fixture lights (coreEmittersToDDGILights) for indirect bounce.
      //   - mesh-area → folded into the referenced mesh's emissive material, so
      //     it reaches both the ReSTIR-DI emissive-triangle path and DDGI as
      //     emissive geometry.
      //   - point / spot → analytic direct-light terms in shade/OIT plus
      //     DDGI/RC fixture-light uploads via coreEmittersToDDGILights. Spots
      //     include real cone data (spotAxis + cosInner/cosOuter);
      //     evalPointLight in the probe shaders applies the smoothstep falloff
      //     when the axis is non-zero (axisLen² > 0.25). Points carry a zero
      //     axis and remain omnidirectional.
      //   - directional → projected to a DDGI `sun` light by
      //     coreEmittersToDDGILights, carrying the emitter's REAL direction
      //     (negated to a travel direction), intensity, and colour into the
      //     probe pass's sun path (replacing the packer's former hardcoded
      //     straight-down warm-white sun). Single-counted: the host sets the
      //     sun-intensity multiplier to 1 when a scene directional is present,
      //     so the emitter intensity is not double-applied. The directional
      //     still drives the shade-side Lo_emit via the WalkaroundUBO config
      //     path (primaryLightDir/Intensity) — those remain host config, not
      //     derived from the emitter, so there is no shade-side double-count.
      supportedEmitterKinds:     new Set<SceneEmitter['kind']>([
        'directional',
        'rect-area',
        'disc-area',
        'point',
        'spot',
        'mesh-area',
      ]),
      // procedural-sky bakes through resolveHybridEnvironment into a finite
      // Preetham equirect + CDF. It remains approximate due model/resolution
      // limits, not because turbidity/rayleigh/mie are dropped.
      supportedEnvironmentKinds: new Set<SceneEnvironment['kind']>(['none', 'hdri', 'procedural-sky']),
      presentationMode:          'swapchain-required',
      supportDetails:            BACKEND_PROMISE_LEDGER['walkaround-hybrid'].supportDetails,
      experimentalFeatures:      buildWalkaroundExperimentalFeatures(this._cfg),
      // RFE-05: Real-time caustic strategies (MNEE / photon-map) are not
      // compatible with the walkaround engine's frame cadence; the walkaround
      // engine always reports 'none'. Track via
      // external_requests/05-manifold-nee.md §4 for the approved approximation
      // path if real-time caustic approximation is added.
      causticStrategy: 'none',
      // W3-D8 — this engine ships a `debug` surface (DDGI atlases, BVH nodes,
      // GI signal textures, and now estimatedGpuMemoryBytes). Hosts can
      // structurally opt-in to the dev-overlay panel without typeof-checking
      // every method.
      debugSurface: true,
    };

    // Pipeline-init coordinator: own the async init race state machine and
    // dispose coordination. The host adapter built below grants the
    // coordinator the small surface it needs without exposing private
    // engine fields directly.
    this._initCoordinator = new PipelineInitCoordinator(this._buildInitHost());

    this.debug = createHybridEngineDebugSurface({
      device: () => this._device,
      readAtlas: () => this._ddgi?.getReadAtlasGPUTextures() ?? null,
      bvhNodesCpu: () => this._bvhBuffers?.bvhNodes?.cpuData,
      debugTextures: () => this._pipeline?.getDebugTextures() ?? null,
      getMemoryBreakdown: () => this._pipeline?.getMemoryBreakdown() ?? null,
      // T3.G click-to-pick: the retained core scene + last-frame camera + canvas
      // size feed the CPU ray-cast in createHybridEngineDebugSurface.
      pickScene: () => this._lastScene,
      pickCamera: () => this._lastFrameCamera,
      pickSize: () => ({ width: this._width, height: this._height }),
      denoiserPassEnabled: () => this._denoiserPassEnabled,
      setDenoiserPassEnabled: (enabled) => {
        this._denoiserPassEnabled = enabled;
      },
      setPipelineDenoiserPassEnabled: (enabled) => {
        this._pipeline?.setDenoiserPassEnabled(enabled);
      },
    });

    // ── GPU error wiring (item 28) ─────────────────────────────────────────
    // Attach an `uncapturederror` listener on the WebGPU device to route
    // validation/internal errors to the host via onError subscribers.
    // Throttled: one report per distinct message per 32 frames to avoid spam.
    // Listener is removed on dispose (engine does not own the device).
    this._onUncapturedError = (event: Event): void => {
      if (this._state === 'disposed') return;
      const gpuEvent = event as { error?: { message?: string } };
      const rawError = gpuEvent.error;
      // eslint-disable-next-line @typescript-eslint/no-base-to-string -- fallback stringification of GPUUncapturedErrorEvent; acceptable for diagnostic messages
      const message = rawError?.message ?? String(event);
      const kind: EngineError['kind'] = rawError != null &&
        rawError.constructor?.name === 'GPUInternalError'
          ? 'gpu-internal'
          : 'gpu-validation';
      const lastFrame = this._errorThrottleMap.get(message) ?? -Infinity;
      if (this._errorFrameCount - lastFrame >= 32) {
        this._errorThrottleMap.set(message, this._errorFrameCount);
        this._emitError({ kind, message, fatal: false, raw: rawError });
      }
    };
    opts.device.addEventListener('uncapturederror', this._onUncapturedError);

    // device.lost: fatal transition to 'error' state.
    opts.device.lost.then((info: { reason?: string; message?: string }) => {
      if (this._state === 'disposed') return;
      this._state = 'error';
      this._emitError({
        kind: 'device-lost',
        message: info.message ?? `GPUDevice lost (reason: ${info.reason ?? 'unknown'})`,
        fatal: true,
        raw: info,
      });
    }).catch(() => { /* spec says it shouldn't reject; guard defensively */ });
  }

  // ── Scene management ───────────────────────────────────────────────────

  private _warnUnconsumedMaterialFields(
    fields: readonly string[],
    method: 'setScene' | 'updatePrimitive',
  ): void {
    if (fields.length === 0) return;
    const sortedFields = Array.from(fields).sort();
    const key = sortedFields.join(',');
    if (this._warnedMaterialFields.has(key)) return;
    this._warnedMaterialFields.add(key);
    this._warn({
      code: 'walkaround-hybrid.unconsumed-material-fields',
      backend: 'walkaround-hybrid',
      phase: method,
      method,
      message:
        `[vitrum/walkaround-hybrid] ${method}: the following material fields are ` +
        `supplied but not consumed by this backend: ${sortedFields.join(', ')}. ` +
        `See consumedMaterialFields.ts for the full allowlist.`,
      details: { fields: sortedFields },
    });
  }

  private _warnApproximateAlphaBlendPrimitiveIds(
    primitiveIds: readonly string[],
    method: 'setScene' | 'updatePrimitive',
  ): void {
    if (primitiveIds.length === 0) return;
    const key = primitiveIds.join(',');
    if (this._warnedAlphaBlendApproximationIds.has(key)) return;
    this._warnedAlphaBlendApproximationIds.add(key);
    this._warn({
      code: 'walkaround-hybrid.alpha-blend-approximation',
      backend: 'walkaround-hybrid',
      phase: method,
      method,
      message:
        `[vitrum/walkaround-hybrid] ${method}: fractional or texture-driven alphaMode:'blend' ` +
        `is camera-composited by the transparent OIT pass, but transparent-layer ` +
        `ReSTIR/GI participation remains approximate; finite emitters are ` +
        `camera-visible fixed-stratified direct lights, not reservoir participants; ` +
        `primitives: ${primitiveIds.join(', ')}.`,
      details: { primitiveIds },
    });
  }

  private _warnApproximateEmissiveMapTexelPdfPrimitiveIds(
    primitiveIds: readonly string[],
    method: 'setScene' | 'updatePrimitive' | 'updateEmitter',
  ): void {
    if (primitiveIds.length === 0) return;
    const key = primitiveIds.join(',');
    if (this._warnedEmissiveMapTexelPdfApproximationIds.has(key)) return;
    this._warnedEmissiveMapTexelPdfApproximationIds.add(key);
    this._warn({
      code: 'walkaround-hybrid.emissive-map-texel-pdf-approximation',
      backend: 'walkaround-hybrid',
      phase: method,
      method,
      message:
        `[vitrum/walkaround-hybrid] ${method}: material-backed emissiveMap ` +
        `surfaces are rendered and localized with UV-aware micro-emitter ` +
        `selection, but exact texel-space emitter alias tables/PDFs are not ` +
        `guaranteed across direct, GI, RC, DDGI, and fallback sampling paths; ` +
        `primitives: ${primitiveIds.join(', ')}.`,
      details: {
        primitiveIds,
        approximation: 'uv-local-micro-emitter-selection',
        missing: 'exact-texel-alias-pdf',
      },
    });
  }

  private _warnReservedReceiveShadowPrimitiveIds(
    primitiveIds: readonly string[],
    method: 'setScene' | 'updatePrimitive',
  ): void {
    if (primitiveIds.length === 0) return;
    this._warn({
      code: 'walkaround-hybrid.reserved-receive-shadow',
      backend: 'walkaround-hybrid',
      phase: method,
      method,
      message:
        `[vitrum/walkaround-hybrid] ${method}: receiveShadow:false is reserved and not ` +
        `consumed by any backend (non-physical for GI); primitives: ${primitiveIds.join(', ')}.`,
      details: { primitiveIds },
    });
  }

  private _warnMaterialTextureAtlasDiagnostics(
    diagnostics: readonly MaterialTextureAtlasDiagnostic[],
    method: 'setScene' | 'updatePrimitive',
  ): void {
    for (const diagnostic of diagnostics) {
      const sourcePath = diagnostic.sourcePath;
      const key =
        `${method}:${diagnostic.code}:${diagnostic.materialIndex}:${diagnostic.field}:` +
        `${diagnostic.colorSpace}:${sourcePath ?? ''}:${diagnostic.texCoord ?? ''}:` +
        `${diagnostic.pixelStride ?? ''}:${diagnostic.valueCount ?? ''}`;
      if (this._warnedMaterialTextureAtlasDiagnostics.has(key)) continue;
      this._warnedMaterialTextureAtlasDiagnostics.add(key);
      const unsupportedTexCoord = diagnostic.code === 'unsupported-material-texture-texcoord';
      const ambiguousStride = diagnostic.code === 'ambiguous-material-texture-stride';
      this._warn({
        code: unsupportedTexCoord
          ? 'walkaround-hybrid.unsupported-material-texture-texcoord'
          : ambiguousStride
            ? 'walkaround-hybrid.ambiguous-material-texture-stride'
            : 'walkaround-hybrid.unreadable-material-texture-map',
        backend: 'walkaround-hybrid',
        phase: method,
        method,
        message: unsupportedTexCoord
          ? `[vitrum/walkaround-hybrid] ${method}: ${diagnostic.field} on material slot ` +
            `${diagnostic.materialIndex}${sourcePath !== undefined ? ` at ${sourcePath}` : ''} ` +
            `uses texCoord ${diagnostic.texCoord}; the material atlas only supports UV sets 0 and 1, so the map is ignored.`
          : ambiguousStride
            ? `[vitrum/walkaround-hybrid] ${method}: ${diagnostic.field} on material slot ` +
              `${diagnostic.materialIndex}${sourcePath !== undefined ? ` at ${sourcePath}` : ''} ` +
              `has ambiguous raw pixel stride ${diagnostic.pixelStride} ` +
              `(${diagnostic.valueCount} values / ${diagnostic.width}x${diagnostic.height} pixels); ` +
              `the atlas decoded it heuristically. Attach __vitrum_hint__ = { channels: N } ` +
              `to make texture ingestion deterministic.`
          : `[vitrum/walkaround-hybrid] ${method}: ${diagnostic.field} on material slot ` +
            `${diagnostic.materialIndex}${sourcePath !== undefined ? ` at ${sourcePath}` : ''} ` +
            `has a texture handle that is not CPU-readable; ` +
            `the map is ignored by the material atlas. Provide a raw {width,height,data} ` +
            `or DataTexture-shaped handle before setScene/updatePrimitive for native map sampling.`,
        details: {
          materialIndex: diagnostic.materialIndex,
          field: diagnostic.field,
          colorSpace: diagnostic.colorSpace,
          ...(diagnostic.texCoord !== undefined ? { texCoord: diagnostic.texCoord } : {}),
          ...(diagnostic.pixelStride !== undefined ? { pixelStride: diagnostic.pixelStride } : {}),
          ...(diagnostic.valueCount !== undefined ? { valueCount: diagnostic.valueCount } : {}),
          ...(diagnostic.width !== undefined ? { width: diagnostic.width } : {}),
          ...(diagnostic.height !== undefined ? { height: diagnostic.height } : {}),
          ...(sourcePath !== undefined ? { sourcePath } : {}),
          ...(diagnostic.textureIndex !== undefined ? { textureIndex: diagnostic.textureIndex } : {}),
          ...(diagnostic.imageIndex !== undefined ? { imageIndex: diagnostic.imageIndex } : {}),
          ...(diagnostic.samplerIndex !== undefined ? { samplerIndex: diagnostic.samplerIndex } : {}),
          ...(diagnostic.imageUri !== undefined ? { imageUri: diagnostic.imageUri } : {}),
          ...(diagnostic.imageMimeType !== undefined ? { imageMimeType: diagnostic.imageMimeType } : {}),
          ...(diagnostic.textureSourceExtension !== undefined
            ? { textureSourceExtension: diagnostic.textureSourceExtension }
            : {}),
          fallback: ambiguousStride ? 'heuristic pixel stride' : 'map ignored',
        },
      });
    }
  }

  private _warnDirectionalAngularDiameterPartialSupport(
    scene: Scene,
    method: 'setScene' | 'updateEmitter',
  ): void {
    const authored = collectAuthoredDirectionalAngularDiameters(scene);
    if (authored.length === 0) return;
    const key = `${method}:${authored.map((e) => `${e.id}:${e.angularDiameter}`).join('|')}`;
    if (this._warnedDirectionalAngularDiameterApproximationIds.has(key)) return;
    this._warnedDirectionalAngularDiameterApproximationIds.add(key);
    this._warn({
      code: 'walkaround-hybrid.directional-angular-diameter-partial-support',
      backend: 'walkaround-hybrid',
      phase: method === 'setScene' ? 'setScene' : 'mutation',
      method,
      message:
        `[vitrum/walkaround-hybrid] ${method}: directional emitter angularDiameter ` +
        `is consumed by visible direct-sun cone sampling, transparent OIT, and ` +
        `stained-glass caustics, but DDGI/RC probe-cache sun transport remains ` +
        `directional; emitters: ${authored.map((e) => e.id).join(', ')}.`,
      details: {
        emitters: authored,
        support: 'direct-sun-cone-only',
        unsupported: [
          'ddgi-sun-probe-soft-shadow',
          'rc-sun-probe-soft-shadow',
        ],
      },
    });
  }

  /**
   * Replace the scene. Triggers a full pipeline reinitialisation
   * (BVH rebuild + ReSTIR pipeline re-init).
   *
   * **BVH + DDGI geometry:** ReSTIR, DDGI, and RC consume the core scene's
   * mesh/skinned/instanced primitives directly.
   *
   * **Host guidance:** pass a canonical `@vitrum/core` Scene. Host-specific
   * scene adapters live outside this package.
   *
   * **Capability filter + analytic fallback:** the scene is first partitioned
   * against this engine's declared `supported*Kinds` (warn + skip). Supported
   * analytic primitives stay in the authored `_lastScene`, then `_renderScene`
   * replaces them with generated MeshPrimitive fallbacks before the BVH/GI
   * ingestion path runs.
   *
   * @param inputScene - The `@vitrum/core` scene.
   */
  setScene(inputScene: Scene): void {
    if (this._state === 'disposed') {
      throw new Error('HybridEngine.setScene: engine is disposed.');
    }
    // Capability filter (warn + skip) consumes this engine's OWN declared
    // support sets. The supported authored scene remains the mutation source of
    // truth; `_renderScene` is the mesh-like ingestion view.
    const { supported: scene, warnings } = partitionSceneBySupport(inputScene, this.capabilities);
    for (const warning of warnings) {
      this._warn({
        code: 'walkaround-hybrid.scene-support-warning',
        backend: 'walkaround-hybrid',
        phase: 'setScene',
        method: 'setScene',
        message: `[vitrum/walkaround-hybrid] ${warning}`,
        details: { warning },
      });
    }

    // Warn once per distinct set of unconsumed material fields so hosts know
    // which Material properties this backend silently ignores. The warn-once
    // key is the sorted field list so incremental scene/material changes with
    // the same ignored set don't spam.
    this._warnUnconsumedMaterialFields(
      collectUnconsumedMaterialFields(
        scene.primitives as unknown as ReadonlyArray<{ readonly kind: string; readonly material?: Record<string, unknown> }>,
      ),
      'setScene',
    );

    const alphaBlendApproxIds = collectApproximateAlphaBlendPrimitiveIds(
      scene.primitives as unknown as ReadonlyArray<{
        readonly id?: string;
        readonly kind: string;
        readonly material?: Record<string, unknown>;
      }>,
    );
    this._warnApproximateAlphaBlendPrimitiveIds(alphaBlendApproxIds, 'setScene');

    const emissiveMapTexelPdfApproxIds = collectApproximateEmissiveMapTexelPdfPrimitiveIds(
      scene.primitives as unknown as ReadonlyArray<{
        readonly id?: string;
        readonly kind: string;
        readonly material?: Record<string, unknown>;
      }>,
      scene.emitters,
    );
    this._warnApproximateEmissiveMapTexelPdfPrimitiveIds(emissiveMapTexelPdfApproxIds, 'setScene');
    this._warnDirectionalAngularDiameterPartialSupport(scene, 'setScene');

    // SHADOW-01 — receiveShadow is reserved/unsupported: a "receiver ignores
    // occlusion" toggle is non-physical for this GI renderer. castShadow rows
    // have native support and should not warn here.
    const receiveShadowIds = scene.primitives
      .filter((p) => (p as { receiveShadow?: boolean }).receiveShadow === false)
      .map((p) => p.id);
    this._warnReservedReceiveShadowPrimitiveIds(receiveShadowIds, 'setScene');

    this._lastScene = scene;
    this._renderScene = sceneWithAnalyticMeshFallback(scene);

    // B15 — derive scene-scale-aware radiometric clamp DEFAULTS from the new
    // scene's world diagonal. Uses the render-scene view (analytic primitives
    // already meshed) so the AABB is computed over real positions. At Cornell
    // scale the law short-circuits to byte-identical defaults; host-explicit
    // clamps pass through un-scaled. These feed the per-frame deps below (NOT
    // the UBO layout — the UBO stays frozen; only the host-computed VALUES move).
    this._applyScaleAwareClamps();

    // W8 Phase 2: rebuild the RC BVH + cascade buffers after async ReSTIR
    // BVH publish via the core-native path.

    // Tear down the existing pipeline, reinitialise asynchronously.
    this._teardownPipeline();
    this._initCoordinator.startInit();
  }

  /** Read back the retained canonical core {@link Scene} (`_lastScene` — the
   *  capability-filtered `supported` authored scene), or null before the
   *  first `setScene`. Implements the optional `Engine.getScene` contract — see
   *  its JSDoc for the no-defensive-copy / frozen-by-contract semantics. The
   *  reference survives {@link dispose}; the facade wrapper gates the
   *  post-dispose read. */
  getScene(): Scene | null {
    return this._lastScene;
  }

  /**
   * B15 — recompute the scene-scale-aware radiometric clamp defaults from the
   * current `_renderScene`'s world diagonal and cache them on
   * `_scaledTunables` / `_scaledIndirectFireflyClamp` (read by the per-frame
   * deps builders). Pure host-side arithmetic — no GPU, no UBO-layout change.
   *
   * INVARIANTS:
   *   • At Cornell scale (diagonal ≈ {@link CORNELL_DIAGONAL}) the result is
   *     byte-identical to the `_cfg` baselines (the law short-circuits at
   *     scaleRatio == 1).
   *   • Host-explicit clamps (`_clampHostExplicit[knob]`) pass through un-scaled.
   *
   * Called from `setScene` (the only diagonal-changing entry — add/remove/
   * topology-patch routes all funnel through `setScene`, so a single hook here
   * keeps the scaled defaults in sync with the live geometry).
   */
  private _applyScaleAwareClamps(): void {
    const result = deriveScaleAwareClamps(this._renderScene, {
      baseTunables: this._cfg.tunables,
      baseIndirectFireflyClamp: this._cfg.indirectFireflyClamp,
      hostExplicit: this._clampHostExplicit,
    });
    this._scaledTunables = result.tunables;
    this._scaledIndirectFireflyClamp = result.indirectFireflyClamp;
    if (this._cfg.verbose && Math.abs(result.scaleRatio - 1) > 1e-6) {
      console.log(
        `[HybridEngine] B15 scale-aware clamps: sceneDiagonal=${result.sceneDiagonal.toFixed(3)} ` +
        `(×${result.scaleRatio.toFixed(3)} vs Cornell) → ` +
        `restirGiIrrClamp=${result.tunables.restirGiIrrClamp.toExponential(3)}, ` +
        `directFireflyClamp=${result.tunables.directFireflyClamp.toExponential(3)}, ` +
        `emitterDist2Floor=${result.tunables.emitterDist2Floor.toExponential(3)} ` +
        `(host-explicit knobs un-scaled).`,
      );
    }
  }

  // ── updatePrimitive — geometry-change path ─────────────────────────────
  //
  // **Routing rules**:
  //  - `patch.transform` present AND no topology fields → fast-path (c):
  //     refit the BVH bounds in-place (no SAH rebuild, no pipeline
  //     recompile), rewrite the affected primitive's vertex slice in
  //     `bvhPositions` for merged BVH, reset the accumulator, and invalidate
  //     DDGI probes so cached irradiance follows the moved object.
  //  - vertex/index-count topology field present (`positions` / `normals` /
  //     `uvs` / `tangents` / `indices`) → full-rebuild path (a): re-run
  //     `buildReSTIRSceneBVH`, destroy + reupload all four BVH GPU
  //     buffers, reset the accumulator.
  //  - `instances` / `params` / `shape` / `fallbackMesh` / `kind` → route
  //     through a full `setScene` rebuild (P5 contract-honesty; see the
  //     `TOPOLOGY_PATCH_WHOLESALE_FIELDS` branch below). NOT a throw — a
  //     geometry/instance change invalidates every cached GI signal on this
  //     realtime stack anyway, so honoring `incrementalPatchSupport.topology`
  //     beats throwing "call setScene()" and matches pt-webgl/pt-webgpu.
  //  - skinned pose fields (`bones` / `boneInverses` / `morphWeights`) →
  //     solve the pose through `solveSkin` and reuse the positions/normals
  //     refit path while preserving the pose fields in scene state.
  //  - material-only patches → `materialPatch` fast path (A3): re-pack the
  //     affected `bvhIndex` / `bvhBeerColors` triangle slices + partial GPU
  //     upload — NO `setScene`, no pipeline recompile.
  //
  // Implementations live in `HybridEnginePrimitiveUpdates.ts`; this method
  // is the routing dispatcher.
  //
  // Implements `Engine.updatePrimitive(id, patch)` from `@vitrum/core`.
  updatePrimitive(id: string, patch: Partial<ScenePrimitive>): void {
    if (this._state === 'disposed') {
      throw new Error('HybridEngine.updatePrimitive: engine is disposed.');
    }
    if (this._state === 'initializing') {
      throw new Error(
        `HybridEngine.updatePrimitive("${id}"): engine is initializing. ` +
        `Wait for setScene init to finish before applying primitive patches.`,
      );
    }
    if (this._lastScene == null) {
      throw new Error(
        `HybridEngine.updatePrimitive("${id}"): no scene set. ` +
        `Call setScene(scene) before updatePrimitive.`,
      );
    }
    const primIndex = this._lastScene.primitives.findIndex((p) => String(p.id) === id);
    if (primIndex < 0) {
      throw new Error(
        `HybridEngine.updatePrimitive("${id}"): primitive id not found in current scene.`,
      );
    }

    // Wholesale-replacement patches — `instances` (instanced-mesh instance-COUNT
    // change), `params` / `shape` (analytic), `fallbackMesh`, `kind` — can't be
    // expressed as an in-place packed-buffer edit, so route them through a
    // full setScene rebuild (the same mutate-Scene → setScene spine addPrimitive /
    // removePrimitive use). A geometry/instance change invalidates every cached GI
    // signal anyway, so on this realtime stack the work is a rebuild either way;
    // the value is honoring incrementalPatchSupport.topology + matching
    // pt-webgl/pt-webgpu (which absorb the instance-COUNT case) instead of
    // throwing "call setScene()". P5 contract-honesty.
    if (TOPOLOGY_PATCH_WHOLESALE_FIELDS.some((f) => (patch as Record<string, unknown>)[f] !== undefined)) {
      this._warnPrimitiveUpdatePatchTruthfulness(id, patch);
      this.setScene(applyPrimitivePatchToScene(this._lastScene, id, patch));
      return;
    }

    // Three.js BVH refit (fast path) preserves topology when only AABB
    // bounds need updating. Three flavours:
    //   - Transform only       → transformRefit  (~1 ms / 30k tris)
    //   - Positions only       → positionsRefit  (A3, ~1 ms / 30k tris)
    //   - True topology change → topologyRebuild (~50 ms / 30k tris)
    // "Positions only" means new vertex data on the SAME index buffer +
    // SAME vertex count. The positionsRefit path falls through to
    // topologyRebuild internally if the count doesn't match.
    const result = this._routePrimitiveUpdate(id, patch);
    if (result == null) {
      // No recognised patch field — treat as a no-op rather than throw so
      // hosts can pass through optional patches without checking each
      // field's presence.
      return;
    }
    this._applyUpdateResult(result);
  }

  /**
   * Select the fast/full path for an `updatePrimitive` patch and run it.
   * Returns `null` for an unrecognised patch (no-op). Branch order is
   * load-bearing — mixed material+geometry patches rebuild so the whole patch is
   * applied; otherwise skinned pose beats structural topology beats positions beats
   * transform beats material:
   *  - structural topology fields (`indices` / UVs / tangents / instances /
   *    analytic shape data / kind) → full SAH `topologyRebuild` (Option (a)).
   *  - `positions` with optional same-count `normals` → A3/H19
   *    `positionsRefit` (same topology, new verts/normals).
   *  - skinned pose fields (`bones` / `boneInverses` / `morphWeights`) →
   *    solve + route through the positions/normals refit path.
   *  - `normals` without `positions` → full rebuild until a normals-only
   *    upload path exists.
   *  - `transform` only → `transformRefit` (refit AABB bounds in place).
   *  - `material` only → `materialPatch` (re-pack slices, NO GI propagation;
   *    the result carries `applySubsystems: false`).
   */
  private _routePrimitiveUpdate(
    id: string,
    patch: Partial<ScenePrimitive>,
  ): PrimitiveUpdateResult | null {
    const has = (f: string): boolean => (patch as Record<string, unknown>)[f] !== undefined;
    const ctx = this._buildPrimitiveUpdateContext();
    this._warnPrimitiveUpdatePatchTruthfulness(id, patch);
    const hasMaterial = has('material');
    const hasSkinnedPose = SKIN_POSE_PATCH_FIELDS.some((f) => has(f));
    if (hasMaterial && hasSkinnedPose) return topologyRebuild(id, patch, ctx);
    if (hasSkinnedPose) return skinnedPosePatch(id, patch, ctx);
    const hasStructuralTopology = TOPOLOGY_PATCH_FIELDS.some((f) => f !== 'normals' && has(f));
    if (hasStructuralTopology) return topologyRebuild(id, patch, ctx);
    if (hasMaterial && (has('positions') || has('normals') || has('transform'))) {
      return topologyRebuild(id, patch, ctx);
    }
    if (has('positions')) return positionsRefit(id, patch, ctx);
    if (has('normals')) return topologyRebuild(id, patch, ctx);
    if (has('transform')) return transformRefit(id, patch, ctx);
    if (has('material')) return materialPatch(id, patch, ctx);
    return null;
  }

  private _warnPrimitiveUpdatePatchTruthfulness(
    id: string,
    patch: Partial<ScenePrimitive>,
  ): void {
    if ((patch as { receiveShadow?: boolean }).receiveShadow === false) {
      this._warnReservedReceiveShadowPrimitiveIds([id], 'updatePrimitive');
    }
    const material = (patch as unknown as { material?: Record<string, unknown> }).material;
    if (material == null) return;
    this._warnUnconsumedMaterialFields(
      collectUnconsumedMaterialFieldsForMaterial(material),
      'updatePrimitive',
    );
    this._warnApproximateAlphaBlendPrimitiveIds(
      collectApproximateAlphaBlendPrimitiveIds([{
        id,
        kind: patch.kind ?? 'mesh',
        material,
      }]),
      'updatePrimitive',
    );
    this._warnApproximateEmissiveMapTexelPdfPrimitiveIds(
      collectApproximateEmissiveMapTexelPdfPrimitiveIds([{
        id,
        kind: patch.kind ?? 'mesh',
        material,
      }], this._lastScene?.emitters ?? []),
      'updatePrimitive',
    );
  }

  /**
   * Uniform epilogue for every primitive-update path: swap the freshly-built
   * BVH buffers + patched scene into engine state, then — unless the path
   * opted out (`applySubsystems === false`, the material-only fast path) —
   * re-sync the GI subsystems against the new BVH. Material-only paths can
   * still request a DDGI material-snapshot refresh without RC geometry
   * propagation.
   */
  private _applyUpdateResult(result: PrimitiveUpdateResult): void {
    this._bvhBuffers = result.bvhBuffers;
    this._lastScene = result.updatedScene;
    this._renderScene = sceneWithAnalyticMeshFallback(result.updatedScene);
    this._warnMaterialTextureAtlasDiagnostics(
      result.bvhBuffers.materialTextureAtlas.diagnostics,
      'updatePrimitive',
    );
    if (result.refreshRcMaterials === true) {
      this._rc?.refreshMaterialsFromCore(result.bvhBuffers.coreMaterials);
    }
    if (result.applySubsystems !== false) {
      this._applyPrimitiveUpdateSubsystems(result);
    } else if (result.refreshDdgiMaterialSnapshot === true) {
      this._ddgi.syncRestirBvhBuffers(result.bvhBuffers, this._renderScene ?? undefined);
    }
  }

  /**
   * PR-7 — GPU LBS wrote world positions into the live `bvhPositions` buffer;
   * refit BVH nodes and sync scene without re-uploading the position slice.
   */
  applyGpuSkinnedRefit(
    id: string,
    localPositions?: Float32Array,
    localNormals?: Float32Array,
  ): void {
    if (this._state === 'disposed') {
      throw new Error('HybridEngine.applyGpuSkinnedRefit: engine is disposed.');
    }
    if (this._state === 'initializing') {
      throw new Error(
        `HybridEngine.applyGpuSkinnedRefit("${id}"): engine is initializing. ` +
        `Wait for setScene init to finish before applying skinned-mesh refits.`,
      );
    }
    let positions = localPositions;
    let normals = localNormals;
    if (positions == null) {
      const prim = this._lastScene?.primitives.find(
        (p) => String(p.id) === id && p.kind === 'skinned-mesh',
      );
      if (prim?.kind !== 'skinned-mesh') {
        throw new Error(`applyGpuSkinnedRefit("${id}"): skinned-mesh primitive not found.`);
      }
      const solved = solveSkin(prim);
      positions = solved.positions;
      normals = solved.normals;
    }
    const result = refitSkinnedMeshAfterGpuWrite(
      id,
      positions,
      normals,
      this._buildPrimitiveUpdateContext(),
    );
    this._applyUpdateResult(result);
  }

  /** Merged BVH position SSBO for GPU skinning (null before pipeline init). */
  getGpuSkinningBvhBuffer(): GPUBuffer | null {
    return this._pipeline?.getBvhPositionBuffer() ?? null;
  }

  /** WS1 — merged BVH normal SSBO for GPU skinning (null before pipeline init).
   *  The skin kernel writes inverse-transpose skinned normals here at
   *  `baseVertex+vi` so the smooth shading-normal blend reads deformed normals. */
  getGpuSkinningNormalBuffer(): GPUBuffer | null {
    return this._pipeline?.getBvhNormalBuffer() ?? null;
  }

  /** Per-mesh vertex ranges in the merged BVH (for GPU skinning). */
  getMeshVertexRanges(): SceneBVHBuffers['meshVertexRanges'] | null {
    return this._bvhBuffers?.meshVertexRanges ?? null;
  }

  /** Active ReSTIR BVH layout (`merged` world positions vs `tlas` local BLAS). */
  getBvhMode(): ReSTIRBvhMode | null {
    return this._bvhBuffers?.bvhMode ?? null;
  }

  getPrimitiveTlasBindings(): SceneBVHBuffers['primitiveTlasBindings'] | null {
    return this._bvhBuffers?.primitiveTlasBindings ?? null;
  }

  /**
   * After geometry BVH updates: sync DDGI probe rays + RC cascades to the live
   * ReSTIR buffers without waiting for the next `renderFrame` tick.
   */
  private _applyPrimitiveUpdateSubsystems(result: PrimitiveUpdateResult): void {
    propagateBvhToGiSubsystems({
      ddgi: this._ddgi,
      rc: this._rc,
      bvhBuffers: this._bvhBuffers,
      lastScene: this._renderScene,
      syncDdgi: true,
      allowRcSceneRebuild: true,
      rcRefitBounds: result.rcRefitBounds,
    });
  }

  /** Build the per-call resource context the primitive-update helpers consume. */
  private _buildPrimitiveUpdateContext(): PrimitiveUpdateContext {
    if (this._lastScene == null) {
      throw new Error(
        'HybridEngine.updatePrimitive: no scene set. Call setScene(scene) first.',
      );
    }
    if (this._renderScene == null) {
      this._renderScene = sceneWithAnalyticMeshFallback(this._lastScene);
    }
    const ctx: PrimitiveUpdateContext = {
      bvhBuffers:            this._bvhBuffers,
      pipeline:              this._pipeline,
      ddgi:                  this._ddgi,
      primaryLightDir:       this._primaryLightDir,
      primaryLightIntensity: this._primaryLightIntensity,
      lastScene:             this._lastScene,
      renderScene:           this._renderScene,
      coreSceneSuppliesMeshes: this._coreSceneSuppliesMeshes(),
      warnUnconsumedMaterialFields: (fields) => {
        this._warnUnconsumedMaterialFields(fields, 'updatePrimitive');
      },
      warnApproximateAlphaBlendPrimitiveIds: (primitiveIds) => {
        this._warnApproximateAlphaBlendPrimitiveIds(primitiveIds, 'updatePrimitive');
      },
      warnApproximateEmissiveMapTexelPdfPrimitiveIds: (primitiveIds) => {
        this._warnApproximateEmissiveMapTexelPdfPrimitiveIds(primitiveIds, 'updatePrimitive');
      },
      onWarning: (warning) => {
        this._warn(warning);
      },
    };
    if (this._cfg.restirBvhModeOverride !== undefined) {
      return { ...ctx, restirBvhModeOverride: this._cfg.restirBvhModeOverride };
    }
    return ctx;
  }

  /**
   * Add one whole primitive to the live scene (contract:
   * {@link Engine.addPrimitive}).
   *
   * Design choice — full `setScene`-equivalent rebuild. A whole-primitive add
   * almost always introduces NEW geometry + a NEW material, and on this realtime
   * stack the geometry change invalidates EVERY cached GI signal — the DDGI
   * irradiance/visibility atlases, the ReSTIR-DI/GI reservoir history, the RC
   * cascade buffers, and the temporal accumulator all index off the packed
   * scene and must be rebuilt/reset. Rather than bolt a brittle "incremental BVH
   * splice + per-subsystem re-sync" onto the add path, we append the primitive
   * to a fresh `Scene` copy and route it through `setScene` — the same
   * already-correct `partitionSceneBySupport` → `_teardownPipeline` →
   * `_initCoordinator.startInit()` spine the initial scene build runs. That
   * spine rebuilds the BVH, re-synthesises the DDGI traversal scene, re-derives
   * DDGI lights, rebuilds the RC BVH (via `publishBvh` →
   * `propagateBvhToGiSubsystems`), and — because the pipeline (and its temporal
   * accumulator + reservoirs + history textures) is torn down and rebuilt blank
   * — resets all temporal/accumulation state. Mirrors pt-webgl / pt-webgpu's
   * full-repack choice: correct-by-construction, no fragile per-array index
   * remap. On a realtime engine the work is a rebuild either way; the value is
   * API consistency, not a perf win.
   *
   * Contract semantics honored:
   *   • Duplicate `id` throws BEFORE any mutation — the dup check runs against
   *     the live `_lastScene`, and `nextScene` is only built (and `setScene`
   *     only called) once it passes, so the scene is unchanged on throw.
   *   • Unsupported primitive kinds (or future analytic shapes outside the
   *     capability set) are warn-skipped by the `partitionSceneBySupport` filter
   *     inside `setScene` (they do not throw).
   *   • Accumulation / temporal history resets — `setScene` tears down the
   *     pipeline (blank accumulator + reservoirs + DDGI/ReSTIR/RC rebuild) and
   *     reinitialises.
   */
  addPrimitive(primitive: ScenePrimitive): void {
    if (this._state === 'disposed') {
      throw new Error('HybridEngine.addPrimitive: engine is disposed.');
    }
    if (this._lastScene == null) {
      throw new Error(
        `HybridEngine.addPrimitive("${String(primitive.id)}"): no scene set. ` +
        `Call setScene(scene) before addPrimitive.`,
      );
    }
    if (this._lastScene.primitives.some((p) => String(p.id) === String(primitive.id))) {
      throw new Error(
        `HybridEngine.addPrimitive: a primitive with id "${String(primitive.id)}" ` +
        `already exists; use updatePrimitive to mutate an existing primitive.`,
      );
    }
    const nextScene: Scene = {
      ...this._lastScene,
      primitives: [...this._lastScene.primitives, primitive],
    };
    this.setScene(nextScene);
  }

  /**
   * Remove one whole primitive from the live scene by `id` (contract:
   * {@link Engine.removePrimitive}). The inverse of {@link addPrimitive}.
   *
   * Implementation: drop the primitive from a fresh `Scene` copy and route
   * through `setScene` (same full-rebuild approach as {@link addPrimitive}).
   * Reusing the `setScene` packing path re-packs the dense BVH / DDGI-light /
   * ReSTIR-emitter arrays correctly by construction rather than hand-rolling a
   * multi-array compaction.
   *
   * **Empty-scene behaviour (H20 / H20-A):** Removing the LAST primitive routes
   * through `setScene` with an empty primitives array. The engine transitions to
   * `'ready'` state (no pipeline / BVH allocated) and `renderFrame` now presents
   * a flat SKY-ONLY frame (`skyTint × skyIrradiance`) via a single device-level
   * clear render pass, returning a genuine `kind:'rendered'` FrameOutput rather
   * than skipping (H20-A). The walkaround sky is a scalar tint on this stack, so
   * a flat fill is the radiometrically-faithful empty-scene background. The
   * dispatched-frame counter (`window.__WALKAROUND__.dbg.framesDispatched`)
   * advances for sky-only frames; `skipNoSwapView` still increments when the host
   * provides no swap-chain view.
   *
   * Contract semantics honored:
   *   • A missing `id` throws BEFORE any mutation — the membership check runs
   *     against the live `_lastScene`; `setScene` is only called once it passes,
   *     so the scene is unchanged on throw.
   *   • Accumulation / temporal history resets exactly as for
   *     {@link addPrimitive}.
   */
  removePrimitive(id: ScenePrimitive['id']): void {
    if (this._state === 'disposed') {
      throw new Error('HybridEngine.removePrimitive: engine is disposed.');
    }
    if (this._lastScene == null) {
      throw new Error(
        `HybridEngine.removePrimitive("${String(id)}"): no scene set. ` +
        `Call setScene(scene) before removePrimitive.`,
      );
    }
    const nextPrimitives = this._lastScene.primitives.filter((p) => String(p.id) !== String(id));
    if (nextPrimitives.length === this._lastScene.primitives.length) {
      throw new Error(
        `HybridEngine.removePrimitive: no primitive with id "${String(id)}" ` +
        `in the live scene.`,
      );
    }
    const nextScene: Scene = {
      ...this._lastScene,
      primitives: nextPrimitives,
    };
    this.setScene(nextScene);
  }

  updateEmitter(id: string, patch: Partial<SceneEmitter>): void {
    if (this._lastScene == null) {
      throw new Error(
        `HybridEngine.updateEmitter("${id}"): no scene set. ` +
        `Call setScene(scene) before updateEmitter.`,
      );
    }
    const idx = this._lastScene.emitters.findIndex((e) => String(e.id) === id);
    if (idx < 0) {
      throw new Error(
        `HybridEngine.updateEmitter("${id}"): emitter id not found in current scene.`,
      );
    }
    if (this._bvhBuffers == null) {
      throw new Error(
        `HybridEngine.updateEmitter("${id}"): BVH not ready. Wait for setScene init to finish.`,
      );
    }
    if (this._renderScene == null || !this._coreSceneSuppliesMeshes()) {
      throw new Error(
        `HybridEngine.updateEmitter("${id}"): current scene has no core mesh primitives.`,
      );
    }

    this._lastScene = applyEmitterPatchToScene(this._lastScene, id, patch);
    this._renderScene = sceneWithAnalyticMeshFallback(this._lastScene);
    this._warnDirectionalAngularDiameterPartialSupport(this._lastScene, 'updateEmitter');
    this._warnApproximateEmissiveMapTexelPdfPrimitiveIds(
      collectApproximateEmissiveMapTexelPdfPrimitiveIds(
        this._lastScene.primitives as unknown as ReadonlyArray<{
          readonly id?: string;
          readonly kind: string;
          readonly material?: Record<string, unknown>;
        }>,
        this._lastScene.emitters,
      ),
      'updateEmitter',
    );

    const emitterOptions = {
      primaryLightDir: {
        x: this._primaryLightDir[0],
        y: this._primaryLightDir[1],
        z: this._primaryLightDir[2],
      },
      primaryLightIntensity: this._primaryLightIntensity,
      packSourceTriIndex: true,
      ...(this._bvhBuffers.bvhMode === 'tlas'
        ? { tlasPrimitiveBindings: this._bvhBuffers.primitiveTlasBindings }
        : {}),
      onWarning: (warning: EngineWarning) => this._warn(warning),
      warningPhase: 'mutation' as const,
      warningMethod: 'updateEmitter',
    };
    const emitterSlice = rebuildEmitterBuffersFromCoreScene(this._renderScene, emitterOptions);

    this._bvhBuffers = {
      ...this._bvhBuffers,
      emitters: emitterSlice.emitters,
      emitterCdf: emitterSlice.emitterCdf,
      emitterCount: emitterSlice.emitterCount,
      totalEmissivePower: emitterSlice.totalEmissivePower,
      lightTree: emitterSlice.lightTree,
      lightTreeNodeCount: emitterSlice.lightTreeNodeCount,
      lightTreeEnabled: emitterSlice.lightTreeEnabled,
    };

    this._pipeline?.updateEmitters(this._bvhBuffers);
    this._rc?.invalidateBindings();
    this._syncDdgiLightsFromCoreScene();
    this._pipeline?.requestAccumReset();
  }

  private _syncDdgiLightsFromCoreScene(): void {
    if (this._renderScene == null || !this._coreSceneSuppliesMeshes()) return;
    // Steps 1–4 (sun intensity, lights merge, emitter tris H18, analytic lights H41)
    // delegated to the shared helper (R3 B-chain step 4). Engine path always
    // merges lights (setLightsConditional: false = default).
    syncDdgiFromCoreScene({
      ddgi: this._ddgi,
      pipeline: this._pipeline,
      ctorLights: this._ctorLights,
      primaryLightIntensity: this._primaryLightIntensity,
      primaryLightDir: this._primaryLightDir,
      onWarning: (warning) => this._warn(warning),
      ...(this._bvhBuffers?.bvhMode === 'tlas'
        ? { tlasPrimitiveBindings: this._bvhBuffers.primitiveTlasBindings }
        : {}),
    }, this._renderScene);
    // B3 — push the scene's directional IBL map+CDFs to the pipeline (or reset to
    // the no-HDRI placeholder). Called here so both the initial scene load and any
    // emitter/scene fast-update re-resolve the env; no-op before pipeline init.
    this._applyDirectionalEnvironment(this._renderScene.environment ?? { kind: 'none' });
    this._ddgi.invalidateProbeCache();
  }

  // ── GI state persistence ────────────────────────────────────────────────
  // Implementation bodies moved to HybridEngineGIState.ts (R3 B-chain step 3).
  // These thin delegates preserve the public API contract unchanged.

  /**
   * Export the converged DDGI global-illumination state (the "cached light
   * field") so the host can persist it (e.g. to IndexedDB via
   * {@link serializeGIState}) and restore it next session without re-converging.
   * Returns null if the probe atlases aren't allocated yet (call after the GI has
   * run at least one frame). Async (atlas readback uses mapAsync).
   */
  async exportGIState(): Promise<GIStateSnapshot | null> {
    return exportGIStateImpl({ device: this._device, ddgi: this._ddgi, pipeline: this._pipeline });
  }

  /**
   * Restore a previously {@link exportGIState}-ed snapshot into the live GI state
   * (seeds the temporal blend, so rendering continues from it instead of
   * re-converging). Restores the DDGI probe atlases AND — when the snapshot
   * carries them — the ReSTIR-GI temporal reservoirs (v2+) and the PPG
   * sTree/dTree guiding distribution (v4+).
   *
   * Returns false (no-op) if the atlases aren't allocated or the snapshot's atlas
   * dims don't match the current grid. When a reservoir section is present, the
   * restore also fails (returns false) if the reservoir grid/size doesn't match
   * the live pipeline — so a partial (atlas-only) restore is never silently
   * reported as a full success. A v3 snapshot (no PPG section) restores the
   * atlases + reservoirs and returns the atlas+reservoir result unchanged; PPG
   * starts cold without error.
   */
  importGIState(snapshot: GIStateSnapshot): boolean {
    return importGIStateImpl({
      device: this._device,
      ddgi: this._ddgi,
      pipeline: this._pipeline,
      onWarning: (warning) => this._warn(warning),
    }, snapshot);
  }

  /**
   * Runtime update of the primary directional light + sky parameters.
   *
   * Re-uploads the WalkaroundUBO at the next frame start. Invalidates the DDGI
   * probe atlas so it re-converges over the next ~8 frames, and resets the
   * temporal accumulator so stale lighting does not bleed through history.
   *
   * No pipelines or GPU buffers are recreated; calling with an empty object is
   * a safe no-op.
   *
   * @param opts - Partial lighting overrides. Omitted fields are unchanged.
   */
  updateLighting(opts: Partial<LightingOptions>): void {
    // `Engine.updateLighting` is contractually opaque (Record<string, unknown>),
    // so hosts can pass any key without a type error at the core-contract call
    // site. Warn (don't throw) on keys outside LightingOptions so silent drops
    // become visible; the field-by-field application below is unchanged.
    assertKnownLightingKeys(opts, (warning, ...args) => this._warn(warning, ...args));

    let changed = false;

    if (opts.primaryLightDir !== undefined) {
      this._primaryLightDir = opts.primaryLightDir;
      changed = true;
      // Republish DDGI sun lights so the probe-update pass follows the same
      // runtime direction that renderFrame() passes to the shade UBO. With a
      // core mesh scene this re-merges scene emitters; without one (lights-only
      // host, or before setScene) fall back to re-orienting the ctor lights —
      // mirroring the init path (line ~836) so the sun follows primaryLightDir
      // regardless of whether a mesh scene is present.
      if (this._renderScene != null && this._coreSceneSuppliesMeshes()) {
        this._syncDdgiLightsFromCoreScene();
      } else {
        this._ddgi.setLights(orientDdgiSunLights(this._ctorLights, this._primaryLightDir));
      }
    }
    if (opts.primaryLightIntensity !== undefined) {
      this._primaryLightIntensity = opts.primaryLightIntensity;
      changed = true;
      // Keep the DDGI ProbeUpdatePass sun-intensity multiplier in sync so the
      // irradiance atlas re-converges at the correct brightness. Single-count:
      // when a scene `directional` drives the sun, its `sun` DDGILight already
      // carries the emitter intensity, so the multiplier stays 1 and config
      // primaryLightIntensity does NOT additionally scale the DDGI sun (it
      // still drives the shade-side Lo_emit via the WalkaroundUBO). Absent a
      // scene directional, the config intensity is the multiplier as before.
      const sceneForSun =
        this._coreSceneSuppliesMeshes() && this._renderScene != null
          ? this._renderScene
          : null;
      this._ddgi.setSunIntensityMultiplier(
        directionalSunMultiplier(sceneForSun, opts.primaryLightIntensity),
      );
    }
    if (opts.skyTint !== undefined) {
      this._skyTint = opts.skyTint;
      changed = true;
    }
    if (opts.skyIrradiance !== undefined) {
      this._skyIrradiance = opts.skyIrradiance;
      changed = true;
    }

    if (!changed) return;

    this._ddgi.setSkyParams?.(this._skyTint, this._skyIrradiance);

    // Invalidate the DDGI probe atlas — re-converges from scratch over the
    // next STRIDE frames (~8 frames, ~133 ms at 60 FPS).
    this._ddgi.invalidateProbeCache();

    // Reset the temporal accumulator — history discarded, α=1 for next frame.
    // _pipeline may be null if the engine is still initialising; the flag is
    // applied as soon as the pipeline exists (set before any renderFrame call).
    this._pipeline?.requestAccumReset();
  }

  /**
   * Apply an environment-only update at runtime (HDRI intensity swap, or a
   * transition to `kind: 'none'`) WITHOUT rebuilding the BVH or re-uploading
   * geometry / materials. Implements the optional `Engine.updateEnvironment`
   * contract; the sibling of {@link updateLighting} for the env-map / sky-config
   * dimension. `attachVitrum` / a host can call this for time-of-day env scrubs
   * the way `pt-webgl` does, with no engine recreation.
   *
   * **What this backend's "environment" actually is.** The walkaround-hybrid
   * realtime stack has NO IBL baker / equirect sampler (unlike `pt-webgl`'s
   * `IblBakerCache`). Its environment lighting is the diffuse sky-dome pair
   * {@link _skyTint} (RGB) + {@link _skyIrradiance} (scalar) consumed by the
   * DDGI ProbeUpdate UBO + the shade pass's sky-aperture / sky-miss paths.
   * So a runtime env swap MAPS the `SceneEnvironment` onto those scalars
   * (see {@link _skyScalarsFromEnvironment}); there is no real HDRI texture to
   * re-bake, so this is intentionally cheap.
   *
   * **What it does (the minimal correct update):**
   *  - Maps the env → sky scalars and mutates `_skyTint` / `_skyIrradiance`
   *    (same fields `updateLighting` touches — one source of truth for the
   *    per-frame {@link _lightingSnapshot}).
   *  - Caches the env on `_lastScene.environment` so the engine's scene-state
   *    read reflects the swap (parallels pt-webgl/pt-webgpu caching the env on
   *    their scene record). A `null` env collapses to `{ kind: 'none' }` (the
   *    Scene.environment field is non-nullable).
   *  - Invalidates the DDGI probe cache (re-converges the world-space irradiance
   *    atlas over the next STRIDE frames) and resets the temporal accumulator —
   *    exactly the sky-portion of `updateLighting`, because the sky-dome term
   *    feeds both the probe rays and the shade pass.
   *
   * **What it deliberately does NOT re-do:** no BVH rebuild, no pipeline
   * recompile, no FrameResources reallocation, no scene re-partition — geometry
   * and materials are untouched. That is the whole point of an env-only fast
   * path (cf. `setScene`, which tears the pipeline down).
   *
   * **Known limitation (opaque HDRI directionality).** Opaque `hdri` handles are
   * not directionally sampled unless a host resolver supplies CPU-visible data.
   * Raw numeric RGB/RGBA HDRI payloads and procedural skies do feed the same
   * directional equirect/CDF path. Procedural skies are still graded
   * 'approximate' because they use a finite Preetham bake rather than an analytic
   * infinite-resolution sky model.
   *
   * After {@link dispose} this is a safe no-op (matches the runtime-update
   * siblings + the `@vitrum/engine` facade's `'noop'` disposed-behaviour for
   * `updateEnvironment`) — the DDGI subsystem is torn down on dispose, so
   * touching it would be unsafe.
   *
   * @param env the new scene environment, or `null` to clear it (≡ `{ kind:
   *   'none' }`).
   */
  updateEnvironment(env: SceneEnvironment | null): void {
    // Disposed → no-op. The runtime-update contract for `updateEnvironment` is
    // `'noop'` after dispose (see @vitrum/engine idempotentDispose), and
    // `dispose()` already called `this._ddgi.dispose()`, so the
    // invalidateProbeCache() below would touch a torn-down subsystem. Guard
    // here so the direct (non-facade) call is also safe.
    if (this._state === 'disposed') return;

    const nextEnv: SceneEnvironment = env ?? { kind: 'none' };
    // Cache on the live scene so a later scene-state read / debug surface sees
    // the current env (parallels pt-webgl/pt-webgpu). `_lastScene` may be null
    // if no setScene() ran yet — still record the sky-scalar change; the next
    // setScene() carries its own environment.
    if (this._lastScene != null) {
      this._lastScene = { ...this._lastScene, environment: nextEnv };
      this._renderScene = this._renderScene != null
        ? { ...this._renderScene, environment: nextEnv }
        : sceneWithAnalyticMeshFallback(this._lastScene);
    }

    // Map the env onto this backend's sky-dome scalars (the only env channel it
    // consumes — there is no IBL baker here). Omitted fields leave the
    // corresponding scalar unchanged.
    const sky = this._skyScalarsFromEnvironment(nextEnv);
    if (sky.skyTint !== undefined) this._skyTint = sky.skyTint;
    if (sky.skyIrradiance !== undefined) this._skyIrradiance = sky.skyIrradiance;
    this._ddgi.setSkyParams?.(this._skyTint, this._skyIrradiance);

    // Re-converge the world-space DDGI irradiance atlas (the sky-dome term feeds
    // the probe rays) and discard temporal history so the new sky energy shows
    // immediately rather than bleeding in over the accumulation window. Same
    // invalidation `updateLighting` does for the sky portion; `_pipeline` may be
    // null mid-init (the accum-reset flag is applied once the pipeline exists).
    this._ddgi.invalidateProbeCache();
    this._pipeline?.requestAccumReset();

    // B3 — push the directional IBL map+CDFs to the pipeline (or reset to the
    // no-HDRI placeholder). Independent of the sky scalars above, which remain
    // the WGSL fallback when no directional data is present.
    this._applyDirectionalEnvironment(nextEnv);
  }

  /**
   * Map a `SceneEnvironment` onto this backend's diffuse sky-dome scalars
   * ({@link _skyTint} / {@link _skyIrradiance}). No GPU work and no engine-state
   * mutation beyond flipping the one-time {@link _proceduralSkyWarned} flag, so
   * the mapping is straightforward to unit-test. Returns only the fields that
   * should change — an omitted field leaves the engine's current scalar in
   * place (so e.g. an `hdri` swap that carries no tint preserves a host-supplied
   * `skyTint`).
   *
   * Mapping (see {@link updateEnvironment} for the full rationale):
   *  - `none` → `skyIrradiance: 0` (sky contributes no light; matches
   *    `applyEnvironment`'s black-background `none`). Tint left unchanged.
   *  - `hdri` → raw numeric payloads / host extension resolvers can provide
   *    diffuse `skyTint` + `skyIrradiance`; opaque handles without a resolver
   *    fall back to intensity-only.
   *  - `procedural-sky` → 'approximate' grade: resolveHybridEnvironment bakes
   *    turbidity/rayleigh/mie/sunDirection into a finite Preetham equirect and
   *    returns scalar skyTint/skyIrradiance as the no-directional fallback.
   */
  private _skyScalarsFromEnvironment(
    env: SceneEnvironment,
  ): { skyTint?: [number, number, number]; skyIrradiance?: number } {
    const resolved = resolveHybridEnvironment(env, {
      extensions: this._environmentResolverExtensions,
    });
    if (resolved.warnings.length > 0 && !this._proceduralSkyWarned) {
      this._proceduralSkyWarned = true;
      for (const warning of resolved.warnings) {
        this._warn({
          code: 'walkaround-hybrid.environment-approximation',
          backend: 'walkaround-hybrid',
          phase: 'mutation',
          method: 'updateEnvironment',
          message: `[HybridEngine] updateEnvironment: ${warning}`,
          details: { warning },
        });
      }
    }
    return {
      ...(resolved.skyTint !== undefined ? { skyTint: resolved.skyTint } : {}),
      ...(resolved.skyIrradiance !== undefined ? { skyIrradiance: resolved.skyIrradiance } : {}),
    };
  }

  /**
   * B3 — resolve the directional IBL payload (PBRT 2D distribution) from the
   * environment and push it to the pipeline's scene-group env resources. A
   * raw pixel-backed HDRI or procedural sky yields the directional map+CDFs;
   * everything else (opaque handle, none, all-black map) resets the pipeline to
   * the no-HDRI placeholder so the WGSL scalar-sky fallback runs (no-HDRI
   * byte-identity). No-op when the pipeline is not yet initialized — setScene's
   * init path calls this AFTER the pipeline exists.
   */
  private _applyDirectionalEnvironment(env: SceneEnvironment): void {
    if (this._pipeline == null) return;
    const resolved = resolveHybridEnvironment(env, {
      extensions: this._environmentResolverExtensions,
    });
    if (resolved.directional !== undefined) {
      this._pipeline.updateDirectionalEnvironment(
        resolved.directional,
        resolved.rotationY ?? 0,
        resolved.directionalIntensity ?? 1,
      );
      // Wave 4 (2026-06-10) — HDRI into DDGI probe misses: hand the equirect
      // radiance view to the probe-update pass so probe-ray misses sample the
      // real map / finite procedural-sky bake when a directional env is bound.
      const envBindings = this._pipeline.getEnvBindings();
      if (envBindings != null) {
        this._ddgi.setEnvironment(
          envBindings.textureView,
          envBindings.sampler,
          resolved.rotationY ?? 0,
          resolved.directionalIntensity ?? 1,
          true,
        );
      }
    } else {
      this._pipeline.updateDirectionalEnvironment(null, 0, 0);
      this._ddgi.setEnvironment(null, null, 0, 0, false);
    }
  }

  // ── Progressive handoff seed source ──────────────────────────────────────
  /**
   * Progressive walkaround→PT seed source (P8 increment 2). The last frame's
   * post-denoise HDR radiance (linear, pre-tonemap — same space as a PT
   * accumulator) as a `BackendTexture` + its internal render dimensions, so a host
   * coordinator can seed a converged PT engine's accumulator
   * (`engine.seedAccumulator(texture, { weight, width, height })`). Null before the
   * first rendered frame. The texture is recycled each frame — consume it
   * SYNCHRONOUSLY within the handoff frame (do not cache the handle).
   *
   * Available only when `capabilities.supportsProgressiveSeedSource === true`.
   */
  getProgressiveSeedTexture(): {
    texture: BackendTexture<'webgpu', GPUTexture>;
    width: number;
    height: number;
  } | null {
    const tex = this._pipeline?.getProgressiveSeedTexture();
    if (tex == null) return null;
    return {
      texture: asBackendTexture<'webgpu', GPUTexture>(tex),
      width: this._internalWidth,
      height: this._internalHeight,
    };
  }

  /**
   * Capture the engine's rendered output as a host-side CPU Float32 RGBA image,
   * row-major, top-left origin.
   *
   * `colorSpace:'linear'` (default) reads `resolvedTexture` — the post-denoiser,
   * pre-tonemap rgba16float output (the same texture exposed by
   * `getProgressiveSeedTexture()`). This is linear-light HDR radiance in scene
   * units, suitable for tone-mapping, EXR export, or luminance checks.
   *
   * `colorSpace:'output'` runs the SAME composite pass (tonemap + OETF +
   * exposure) into an engine-owned offscreen `rgba8unorm` texture and reads it
   * back as display-encoded, post-OETF values in [0, 1].  Unlike 'linear', this
   * path produces the display-referred image a viewer would see on screen.  The
   * composite UBO settings (tonemap operator, exposure, output color space) from
   * the most recent rendered frame are reused verbatim — there is no need to call
   * `renderFrame` again.
   *
   * Returns `null` before the first frame (no pipeline or resolvedTexture not yet
   * allocated).  Pipeline stall: submits copyTextureToBuffer + mapAsync; use for
   * debugging/export, not per-frame readback.
   */
  async captureFrame(opts?: CaptureFrameOptions): Promise<CapturedFrame | null> {
    const colorSpace = opts?.colorSpace ?? 'linear';
    if (colorSpace === 'output') {
      const rgba = await this._pipeline?.captureOutputFrame() ?? null;
      if (rgba == null) return null;
      return { width: this._internalWidth, height: this._internalHeight, rgba };
    }
    const seedResult = this.getProgressiveSeedTexture();
    if (seedResult == null) return null;
    const { width, height } = seedResult;
    if (width <= 0 || height <= 0) return null;
    const texture = seedResult.texture as unknown as GPUTexture;
    const rgba = await readRgba16fWalkaround(this._device, texture, width, height);
    if (rgba == null) return null;
    return { width, height, rgba };
  }

  // ── Resize ─────────────────────────────────────────────────────────────

  /**
   * Resize the render surface WITHOUT rebuilding the BVH or recompiling
   * pipelines. The host calls this whenever the canvas (or device-pixel
   * ratio) changes — much cheaper than the previous resize-storm pattern
   * (engine teardown → recreate engine → poll for ready), which on a
   * single resize tick churned every BVH buffer + every pipeline shader
   * + every DDGI atlas + ~1 GB of FrameResources textures.
   *
   * Behaviour:
   *   - Updates `_width` / `_height` on the engine.
   *   - Calls `WalkaroundGPUPipeline.resize(W, H)` if the pipeline is
   *     live, which destroys + recreates per-frame GPU resources only
   *     (FrameResources textures + reservoir buffers + variance buffers
   *     + GTAO half/full + SVGF persistent textures) at the new size.
   *     The BVH, pipeline shaders, bind-group layouts, DDGI atlases,
   *     and per-pass UBOs are preserved.
   *   - Resets the temporal accumulator + ping-pong indices on the
   *     pipeline (the new textures are blank, so reusing prior history
   *     would sample undefined memory).
   *
   * No-op when called with the current size, or when the pipeline isn't
   * yet live (the new size is stored on the engine; `_initPipeline` will
   * use it when it constructs the pipeline).
   *
   * Cost: O(W·H) GPU memory churn for the FrameResources reallocation;
   * no shader recompile, no BVH rebuild. Typical resize tick: 5-30 ms
   * for the GPU allocations on a 4K surface, vs 500-2000 ms for a full
   * engine teardown + re-init.
   */
  setSize(width: number, height: number): void {
    if (this._state === 'disposed') {
      throw new Error('HybridEngine.setSize: engine is disposed.');
    }
    if (width === this._width && height === this._height) return;
    if (width <= 0 || height <= 0) {
      // Defensive: WebGPU createTexture rejects zero-sized textures.
      // Hosts sometimes feed in transient 0-pixel sizes during resize
      // animations — silently ignore.
      return;
    }
    this._width = width;
    this._height = height;
    // Recompute the internal render size from the last-seen resolution factor
    // so a canvas resize preserves the host's chosen downscale. The pipeline
    // renders at the INTERNAL size; the composite upscales to the canvas.
    this._internalWidth = Math.max(1, Math.round(width * this._resolutionFactor));
    this._internalHeight = Math.max(1, Math.round(height * this._resolutionFactor));
    if (this._pipeline) {
      this._pipeline.resize(this._internalWidth, this._internalHeight);
    }
    // No DDGI invalidation — the irradiance atlas is world-space, not
    // screen-space, so it survives a resize unchanged.
  }

  /**
   * §5.1 — apply a per-frame `quality.resolutionFactor`. Computes the target
   * internal render size (= canvas × clamped factor), and — when it changes
   * beyond a 2-px threshold and the debounce window has elapsed — resizes the
   * pipeline's per-frame resources to the internal size. The composite pass
   * upscales the internal-sized resolvedTexture to the full canvas swap-chain
   * view, so no swap-chain reconfigure is needed.
   *
   * Returns the internal dims to dispatch at this frame (unchanged when the
   * resize was debounced). Called once per frame from the orchestrator before
   * `pipeline.renderFrame`.
   */
  private _applyResolutionFactor(
    factor: number | undefined,
    nowMs: number,
  ): { width: number; height: number } {
    // Record the clamped factor so a subsequent setSize() preserves the host's
    // chosen downscale.
    this._resolutionFactor =
      typeof factor === 'number' && Number.isFinite(factor) && factor > 0
        ? Math.min(1, factor)
        : 1;

    const decision = resolveInternalRenderSize({
      swapW: this._width,
      swapH: this._height,
      factor,
      currentW: this._internalWidth,
      currentH: this._internalHeight,
      nowMs,
      lastResizeTs: this._lastResolutionResizeTs,
    });

    if (decision.shouldResize) {
      this._internalWidth = decision.targetW;
      this._internalHeight = decision.targetH;
      this._lastResolutionResizeTs = nowMs;
      this._pipeline?.resize(decision.targetW, decision.targetH);
    }
    return { width: this._internalWidth, height: this._internalHeight };
  }

  // ── Frame rendering ────────────────────────────────────────────────────

  /**
   * Render one walkaround frame. Drives:
   *   1. DDGI per-frame compute (fire-and-forget) — GPU command queueing is
   *      synchronous from JS's perspective; the actual GPU work runs after the
   *      JS tick returns. The atlas DDGI writes is double-buffered: this frame
   *      reads from the previous tick's write target while next tick will read
   *      from this one.
   *   2. DDGI atlas wire into the ReSTIR shade pass.
   *   3. ReSTIR pipeline.renderFrame().
   *
   * The host calls `engine.renderFrame(input)` and receives a complete frame.
   * The host does NOT separately call `ddgi.updateFrame()`.
   *
   * Returns a FrameOutput immediately — the pipeline writes directly into the
   * swap chain texture provided via `input.swapChainView`.
   *
   * The 60 FPS internal throttle is enforced here: on high-refresh-rate
   * displays, frames arriving faster than ~16.67 ms apart are skipped and
   * this returns a "skip" FrameOutput (`kind: 'skipped'`,
   * `samplesAccumulated: 0`, `isConverged: false`).
   *
   * Note: `input.viewport` (the CANVAS size) is ignored by HybridEngine — its
   * WebGPU render targets (DDGI atlas, ReSTIR reservoirs, history textures,
   * accumulation buffer) are sized to the canvas at construction and the
   * canvas size is changed only via {@link setSize}. Hosts MUST call
   * `engine.setSize(w, h)` when the canvas dimensions change; pushing a new
   * `viewport` per frame is silently dropped.
   *
   * However, `input.quality.resolutionFactor` IS honoured per-frame
   * (Phase-0 productization): it scales the INTERNAL render resolution
   * (= canvas × factor) via a debounced {@link _applyResolutionFactor}; the
   * composite pass upscales to the full canvas. See the `@vitrum/core`
   * FrameInput.viewport JSDoc for the cross-backend contract.
   */
  renderFrame(input: FrameInput): FrameOutput {
    // Advance the error-throttle frame counter (see _onUncapturedError).
    this._errorFrameCount++;
    // Ergonomics guard: HybridEngine sizes its render targets at construction /
    // `setSize()` and does NOT honour `FrameInput.viewport` per-frame (unlike the
    // converged PT backends — see the FrameInput.viewport contract note). A host
    // that resizes its canvas and pushes a new viewport, expecting the engine to
    // follow, would silently render at the stale size. Warn once so the misuse is
    // visible instead of mysterious. (attachVitrum wires setSize for you.)
    if (
      !this._viewportMismatchWarned &&
      (input.viewport.width !== this._width || input.viewport.height !== this._height)
    ) {
      this._viewportMismatchWarned = true;
      this._warn({
        code: 'walkaround-hybrid.viewport-ignored',
        backend: 'walkaround-hybrid',
        phase: 'renderFrame',
        method: 'renderFrame',
        message:
          `[HybridEngine] FrameInput.viewport (${input.viewport.width}×${input.viewport.height}) ` +
          `differs from the engine canvas size (${this._width}×${this._height}) and is IGNORED. ` +
          'HybridEngine sizes render targets at construction; call engine.setSize(width, height) ' +
          'on canvas resize (attachVitrum does this automatically). For per-frame internal-' +
          'resolution scaling use FrameInput.quality.resolutionFactor instead.',
        details: {
          viewportWidth: input.viewport.width,
          viewportHeight: input.viewport.height,
          engineWidth: this._width,
          engineHeight: this._height,
        },
      });
    }
    // Rebuild-key check (D2.5 — moved out of the orchestrator dep bundle so
    // engine-state mutations don't live inside the FrameDeps closure). Must
    // run BEFORE _buildFrameDeps / runHybridEngineFrame — same position as the
    // former orchestrator-side check, so skip-output semantics are preserved.
    const fp = fingerprintHybridPipelineRebuildKey(
      this._cfg.getPipelineRebuildKey?.() ?? this._cfg.staticPipelineRebuildKey,
    );
    if (fp !== this._rebuildKeyFingerprintSeen) {
      this._rebuildKeyFingerprintSeen = fp;
      this.reset();
      return HYBRID_FRAME_SKIP_OUTPUT;
    }

    // Retain the camera (copied — decoupled from host mutation) for debug
    // click-to-pick (EngineDebugSurface.pickPrimitive, T3.G).
    this._lastFrameCamera = {
      viewMatrix: new Float32Array(input.viewMatrix),
      projMatrix: new Float32Array(input.projMatrix),
      cameraPosition: [input.cameraPosition[0], input.cameraPosition[1], input.cameraPosition[2]],
    };
    return runHybridEngineFrame(this._buildFrameDeps(), input);
  }

  /** Live lighting snapshot — the four runtime-mutable lighting fields
   *  (`updateLighting()` mutates them). Grouped so both DI builders and any
   *  future lighting consumer share one source of truth: adding a lighting
   *  field is a single edit here, not one per builder. Read at call time
   *  (per-frame snapshot semantics — see {@link _buildFrameDeps}). */
  private _lightingSnapshot(): HybridLightingDeps {
    return {
      primaryLightDir: this._primaryLightDir,
      primaryLightIntensity: this._primaryLightIntensity,
      skyTint: this._skyTint,
      skyIrradiance: this._skyIrradiance,
    };
  }

  /** Tuple-typed denoiser-filter cluster (firefly clamp + per-channel atrous
   *  sigmas). These live outside the number-only {@link Tunables} table
   *  because they are tuple-valued; grouping them keeps {@link _buildFrameDeps}
   *  compact and makes a new tuple knob a single edit. */
  private _denoiserFilterDeps(): HybridDenoiserFilterDeps {
    const directionalSunShadowDisabled =
      this._renderScene?.emitters.some((e) => e.kind === 'directional' && e.castShadow === false) === true;
    return {
      // B15 — scene-scale-aware default (falls back to the Cornell baseline
      // before the first setScene). Host overrides already pass through verbatim
      // (deriveScaleAwareClamps leaves host-explicit knobs un-scaled).
      indirectFireflyClamp: this._scaledIndirectFireflyClamp ?? this._cfg.indirectFireflyClamp,
      atrousDirectSigmas: this._cfg.atrousDirectSigmas,
      atrousIndirectSigmas: this._cfg.atrousIndirectSigmas,
      stainedGlassFlags: directionalSunShadowDisabled
        ? (this._cfg.stainedGlassFlags | SHADE_FLAG_DIRECT_SUN_SHADOW_DISABLED) >>> 0
        : this._cfg.stainedGlassFlags,
      restirPtReuse: this._cfg.restirPtReuse,
      nrcEnabled: this._cfg.nrcEnabled,
    };
  }

  private _buildFrameDeps(): HybridEngineFrameDeps {
    // eslint-disable-next-line @typescript-eslint/no-this-alias -- `self` required for the `get state()` getter inside the returned object literal where `this` would refer to the object, not the class instance
    const self = this;
    return {
      subsystems: {
        pipeline: self._pipeline,
        bvhBuffers: self._bvhBuffers,
        ddgi: self._ddgi,
        rc: self._rc,
        skinning: self._skinning,
        lastScene: self._renderScene,
      },
      lighting: self._lightingSnapshot(),
      filter: self._denoiserFilterDeps(),
      telemetry: {
        frameSubs: self._frameSubs,
        progressSubs: self._progressSubs,
        verbose: self._cfg.verbose,
        debugTimings: self._debugTimings,
        debugSurface: self.debug,
        dbg: self._cfg.debug ? self._dbg : null,
        getDenoiserState: () => self._pipeline?.getActiveDenoiserState() ?? null,
      },
      dims: {
        width: self._width,
        height: self._height,
        internalWidth: self._internalWidth,
        internalHeight: self._internalHeight,
      },
      control: {
        targetFrameIntervalMs: self._cfg.targetFrameIntervalMs,
        getLastFrameTs: () => self._lastFrameTs,
        setLastFrameTs: (ts) => {
          self._lastFrameTs = ts;
        },
        applyResolutionFactor: (factor, nowMs) => self._applyResolutionFactor(factor, nowMs),
        runSkinning: () => {
          if (self._skinning != null && self._lastScene != null) {
            self._skinning.run(self, self._lastScene);
          }
        },
        presentLastFrame: (view) => {
          self._pipeline?.presentLastFrame(view);
        },
      },
      flags: {
        get state() {
          return self._state;
        },
        debug: self._cfg.debug,
        ddgiOn: self._ddgiOn,
        isLayerEnabled: (layer) => self._layerEnabled.get(layer) ?? true,
        device: self._device,
        // B15 — scene-scale-aware tunables (falls back to the Cornell baseline
        // before the first setScene; byte-identical at Cornell scale).
        tunables: self._scaledTunables ?? self._cfg.tunables,
        rcWeight: self._rcWeight,
      },
    };
  }

  // ── Reset ──────────────────────────────────────────────────────────────

  /**
   * Tear down the pipeline and reinitialise from scratch.
   * Hosts call this when the scene changes significantly.
   */
  reset(): void {
    if (this._state === 'disposed') {
      throw new Error('HybridEngine.reset: engine is disposed.');
    }
    this._teardownPipeline();
    this._initCoordinator.startInit();
  }

  // ── Pause / resume ─────────────────────────────────────────────────────

  /**
   * Pause per-frame compute. Engine state transitions from `'ready'` →
   * `'paused'`. Calls in any other live state (e.g. `'initializing'`,
   * `'uninitialized'`) are no-ops; calls after `'disposed'` throw.
   *
   * Aligns with `PTEngineWebGL2.pause()`: both throw on disposed, both
   * no-op when the state transition doesn't apply.
   */
  pause(): void {
    if (this._state === 'disposed' || this._state === 'error') {
      throw new Error('pause: engine is disposed or in error state');
    }
    if (this._state === 'ready') {
      this._state = 'paused';
    }
    // no-op in 'uninitialized' | 'initializing' | 'paused'
  }

  /**
   * Resume per-frame compute. Engine state transitions from `'paused'` →
   * `'ready'`. Calls in any other live state are no-ops; calls after
   * `'disposed'` throw.
   *
   * Aligns with `PTEngineWebGL2.resume()`: both throw on disposed, both
   * no-op when the state transition doesn't apply.
   */
  resume(): void {
    if (this._state === 'disposed' || this._state === 'error') {
      throw new Error('resume: engine is disposed or in error state');
    }
    if (this._state === 'paused') {
      this._state = 'ready';
    }
    // no-op in 'uninitialized' | 'initializing' | 'ready'
  }

  // ── Layer toggles (host-accessible debug interface) ────────────────────

  /**
   * Enable or disable a named render layer. Currently recognised layers:
   *   - 'ddgi' — DDGI probe atlas wiring into the shade pass.
   *
   * Replaces `window.__HYBRID_LAYERS__` from the original host bridge.
   * The host sets up `window.__HYBRID_LAYERS__` → `engine.setLayerEnabled()`
   * forwarding if it wants console-accessible toggles.
   */
  setLayerEnabled(layer: string, enabled: boolean): void {
    this._layerEnabled.set(layer, enabled);
  }

  // ── Telemetry (T3.E) ───────────────────────────────────────────────────

  /** Subscribe to per-frame stats. Fired at the end of each successful
   *  renderFrame() call. Returns an unsubscribe function. Subscribers
   *  that throw are swallowed so the render loop stays alive. */
  onFrame(cb: (stats: FrameStats) => void): () => void {
    this._frameSubs.push(cb);
    return () => {
      const i = this._frameSubs.indexOf(cb);
      if (i >= 0) this._frameSubs.splice(i, 1);
    };
  }

  /**
   * Subscribe to long-running progress events. Returns an unsubscribe
   * function. Subscribers that throw are swallowed so the render loop stays
   * alive (mirrors {@link onFrame}).
   *
   * Walkaround engines don't accumulate samples, so there is no `'pt-spp'`
   * signal — but the two REAL warm-up signals this engine has ARE surfaced
   * (closing the contract's `'ddgi-warmup'` / `'denoiser-converge'`
   * zero-producer gap):
   *
   *   - `'ddgi-warmup'` — the DDGI probe round-robin updates `1/stride` of the
   *     grid per frame, so a freshly built / invalidated grid takes `stride`
   *     frames for every probe to receive its first update. `fraction` ramps
   *     `frame / stride` from 0→1 and emission STOPS once the grid is warm
   *     (`DDGI.ready`). Reset to 0 on `setScene()` (fresh DDGI) and
   *     `updateLighting()` (`invalidateProbeCache()`).
   *
   *   - `'denoiser-converge'` — the temporal accumulator blends `α` of the new
   *     frame with `1-α` of history (α≈0.01 ⇒ ~100-frame window). `fraction`
   *     ramps `accumFrameIndex / round(1/α)` and emission STOPS once the
   *     window is full. Reset to 0 on camera motion, `updateLighting()` /
   *     `updateEmitter()` (`requestAccumReset()`), and `setSize()` / resize.
   *
   * Both events fire at most once per dispatched frame, only while their
   * signal is still converging. A no-op when the host registered no callback.
   */
  onProgress(cb: (progress: ProgressStats) => void): () => void {
    this._progressSubs.push(cb);
    return () => {
      const i = this._progressSubs.indexOf(cb);
      if (i >= 0) this._progressSubs.splice(i, 1);
    };
  }

  /** Subscribe to GPU/runtime errors. Returns an unsubscribe function.
   *  Wired events: device `uncapturederror` (throttled, non-fatal) and
   *  `device.lost` (fatal, transitions engine to `'error'`). */
  onError(cb: (error: EngineError) => void): () => void {
    this._errorSubs.push(cb);
    return () => {
      const i = this._errorSubs.indexOf(cb);
      if (i >= 0) this._errorSubs.splice(i, 1);
    };
  }

  /** Subscribe to non-fatal contract warnings. Returns an unsubscribe function. */
  onWarning(cb: (warning: EngineWarning) => void): () => void {
    this._warningSubs.push(cb);
    return () => {
      const i = this._warningSubs.indexOf(cb);
      if (i >= 0) this._warningSubs.splice(i, 1);
    };
  }

  /** Internal: emit an error to all subscribers. Catches subscriber throws. */
  private _emitError(error: EngineError): void {
    for (const cb of this._errorSubs) {
      try { cb(error); } catch { /* must not break rendering */ }
    }
  }

  /** Internal: emit a warning to all subscribers. Catches subscriber throws. */
  private _emitWarning(warning: EngineWarning): void {
    for (const cb of this._warningSubs) {
      try { cb(warning); } catch { /* must not break rendering */ }
    }
  }

  private _warn(warning: EngineWarning, ...consoleArgs: readonly unknown[]): void {
    console.warn(...(consoleArgs.length > 0 ? consoleArgs : [warning.message]));
    this._emitWarning(warning);
  }

  // ── Dispose ────────────────────────────────────────────────────────────

  /**
   * Synchronous dispose — releases all engine-owned GPU resources.
   *
   * The contract intentionally remains synchronous (so hosts can call it
   * from React cleanup effects, finalizers, etc. without an async
   * paradigm shift). When the {@link PipelineInitCoordinator} has an init
   * chain in flight, the actual GPU-resource release for any work that
   * chain hasn't yet published is deferred to the chain's own `finally`
   * block — the chain checks the coordinator's `_pendingTeardown` after
   * every await boundary and, if set, disposes its locals AND finalises
   * teardown of whatever did make it to shared state.
   *
   * Idempotent: a second `dispose()` call is a no-op.
   */
  dispose(): void {
    if (this._state === 'disposed' && !this._initCoordinator.initRunning) {
      // Already disposed and no in-flight chain to coordinate with — no-op.
      return;
    }

    // Remove GPU error listener before any teardown so the handler can't fire
    // after dispose (the device is no longer ours to observe).
    if (this._onUncapturedError != null) {
      this._device.removeEventListener('uncapturederror', this._onUncapturedError);
      this._onUncapturedError = null;
    }
    this._errorSubs.length = 0;
    this._warningSubs.length = 0;
    this._errorThrottleMap.clear();

    // requestTeardown returns true when the coordinator has no chain in
    // flight (we tear down here and now), false when it has one in flight
    // and its finally block will tear down. The coordinator records the
    // dispose intent regardless so its phase checkpoints bail.
    const teardownNow = this._initCoordinator.requestTeardown();
    if (teardownNow) {
      // No in-flight init; tear down here and now.
      this._teardownPipeline();
      this._skinning?.dispose();
      this._ddgi.dispose();
      // W8 Phase 2 — also tear down RC subsystem when active.
      if (this._rc) {
        this._rc.dispose();
        this._rc = null;
      }
      this._state = 'disposed';
    } else {
      // An init is mid-flight. Defer teardown to that chain's finally
      // block — it will dispose its locals AND tear down whatever's
      // currently in shared state. We can't safely call
      // _teardownPipeline() here because the in-flight chain's
      // `await pipeline.initialize()` may still be holding a live
      // reference to a half-built pipeline.
      this._state = 'disposed';
      // Note: _ddgi.dispose() is deferred too; the in-flight chain may
      // still call _ddgi.setSunIntensityMultiplier() after the
      // post-pipeline checkpoint, and we don't want a torn-down DDGI
      // under it. The chain's finally calls disposeDdgi() when it sees
      // pending teardown.
    }

    if (this._cfg.debug && typeof window !== 'undefined') {
      const dbg = this._dbg;
      dbg.disposeCount++;
      const liveMs = dbg.initStart > 0 ? performance.now() - dbg.initStart : 0;
      console.log(`[hybrid:debug] dispose #${dbg.disposeCount}`, {
        ranForMs: liveMs.toFixed(1),
        framesDispatched: dbg.framesDispatched,
        deferredTeardown: !teardownNow,
        skipReasons: {
          noPipeline: dbg.skipNoPipeline, noBvh: dbg.skipNoBvh,
          noSwapView: dbg.skipNoSwapView, frameInterval: dbg.skipFrameInterval,
        },
      });
    }
  }

  // ── Private helpers ────────────────────────────────────────────────────

  /** True when the render-ingestion scene supplies at least one triangle-backed
   *  primitive. Rest-pose skinned meshes count — host pushes deformed positions
   *  via `updatePrimitive`, but the BVH still needs a non-empty scene to build.
   *  Instanced meshes count as well because the walkaround TLAS path consumes
   *  their instance matrices directly. */
  private _coreSceneSuppliesMeshes(): boolean {
    const s = this._renderScene;
    return s != null && s.primitives.some(
      (p) => p.kind === 'mesh' || p.kind === 'skinned-mesh' || p.kind === 'instanced-mesh',
    );
  }

  /** Scene-readiness for BVH build: core mesh payload plus optional host gate. */
  private _sceneReadyForBvh(): boolean {
    return this._coreSceneSuppliesMeshes() && this._isSceneReady();
  }

  private _teardownPipeline(): void {
    if (this._pipeline) {
      this._pipeline.dispose();
      this._pipeline = null;
    }
    if (this._bvhBuffers) {
      disposeSceneBVH(this._bvhBuffers);
      this._bvhBuffers = null;
    }
    if (this._state !== 'disposed') {
      this._state = 'initializing';
    }
  }

  /** Construction-time immutable config consumed by the init coordinator.
   *  Every field here is assigned once in the constructor and never mutated,
   *  so plain values are behaviorally identical to live getters — grouping
   *  them collapses ~11 one-line getters into one spread and makes adding a
   *  ctor-immutable init input a single edit. The MUTABLE fields (width,
   *  height, lastScene, lighting, current BVH/traversal-scene) stay as live
   *  getters in {@link _buildInitHost} because the coordinator reads them
   *  across async phases. */
  private _initStaticConfig(): HybridInitStaticConfig {
    return {
      device: this._device,
      restirBvhModeOverride: this._cfg.restirBvhModeOverride,
      denoiser: this._cfg.denoiser,
      neuralWeights: this._cfg.neuralWeights,
      oidnModelUrl: this._cfg.oidnModelUrl,
      oidnExecutionProviders: this._cfg.oidnExecutionProviders,
      verbose: this._cfg.verbose,
      debug: this._cfg.debug,
      cameraMoveResetThresholdSq: this._cfg.initTunables.cameraMoveResetThresholdSq,
      temporalAccumAlpha: this._cfg.initTunables.temporalAccumAlpha,
      checkerboardMotionThresholdSq: this._cfg.initTunables.checkerboardMotionThresholdSq,
      ctorLights: this._ctorLights,
      ddgi: this._ddgi,
      gtaoMode: this._cfg.gtaoMode,
      diSpatialPasses: this._cfg.diSpatialPasses,
      giSpatialPasses: this._cfg.giSpatialPasses,
      // GRIS / ReSTIR-PT reuse is a COMPILE-TIME structural gate at the pipeline
      // level (selects the GI pipeline layout + shader variant). The `restirPtReuse`
      // number (0/1) also drives the UBO flag; here we forward the boolean so the
      // pipeline builds the matching layout.
      restirPtReuse: this._cfg.restirPtReuse === 1,
      // NRC live cache — same COMPILE-TIME structural gate discipline as
      // restirPtReuse. `nrcEnabled` (0/1) also drives the UBO flag; here we
      // forward the boolean so the pipeline builds the matching gi-ris layout
      // (4-group DDGI default vs 5-group inline-MLP variant).
      nrcEnabled: this._cfg.nrcEnabled === 1,
      // PPG guided sampling — builds the ppg-update pipeline + UBO gate.
      ppgEnabled: this._cfg.ppgEnabled === 1,
      // NRC substitution warmup gate.
      nrcWarmupSteps: this._cfg.nrcWarmupSteps,
      ppgDispatchInterval: this._cfg.ppgDispatchInterval,
      // H47 — PPG max spatial cells. undefined ⇒ allocatePPGResources default (1 024).
      ppgMaxSpatialCells: this._cfg.ppgMaxSpatialCells,
      // H29 — PPG max per-cell dTree nodes. undefined ⇒ default 341-node stride.
      ppgMaxDTreeNodesPerCell: this._cfg.ppgMaxDTreeNodesPerCell,
      // PPG guide/cosine MIS mixture alpha.
      ppgMixAlpha: this._cfg.ppgMixAlpha,
      // Checkerboard half-res shading — flips the ResolvePass gate + the
      // per-frame shade UBO fields. OFF (default) is bit-identical.
      checkerboard: this._cfg.checkerboard,
      regirConfig: this._cfg.regirConfig,
    };
  }

  /** Build the back-reference the {@link PipelineInitCoordinator} consumes.
   *  Live-mutable inputs are getters closing over `this`; construction-time
   *  immutables are spread from {@link _initStaticConfig}. The coordinator
   *  never sees raw field references, only the small documented surface in
   *  `HybridEngineLifecycle.ts`. */
  private _buildInitHost(): PipelineInitHost {
    // eslint-disable-next-line @typescript-eslint/no-this-alias -- `self` required for the `get width()` / `get height()` etc. getters inside the returned object literal where `this` would refer to the object, not the class instance
    const self = this;
    return {
      ...this._initStaticConfig(),
      // Pipeline initializes at the INTERNAL render size (= canvas ×
      // resolutionFactor). Equal to the canvas size on first init (factor
      // 1.0); after a factor was applied, a reset() re-inits at the live
      // internal size so the composite upscale stays correct.
      get width() { return self._internalWidth; },
      get height() { return self._internalHeight; },
      get lastScene() { return self._renderScene; },
      get primaryLightDir() { return self._primaryLightDir; },
      get primaryLightIntensity() { return self._primaryLightIntensity; },
      get preferredSwapChainFormat() { return getPreferredSwapChainFormat(); },
      get currentBvhBuffers() { return self._bvhBuffers; },

      isSceneReadyForBvh: () => self._sceneReadyForBvh(),
      coreSceneSuppliesMeshes: () => self._coreSceneSuppliesMeshes(),

      publishBvh:             (bvh) => {
        self._bvhBuffers = bvh;
        self._warnMaterialTextureAtlasDiagnostics(bvh.materialTextureAtlas.diagnostics, 'setScene');
        propagateBvhToGiSubsystems({
          ddgi: self._ddgi,
          rc: self._rc,
          bvhBuffers: bvh,
          lastScene: self._renderScene,
          syncDdgi: true,
          allowRcSceneRebuild: true,
        });
      },
      publishPipeline:        (p)   => { self._pipeline = p; },
      rollbackBvh:            ()    => { self._bvhBuffers = null; },
      setState:               (s)   => { self._state = s; },
      reportError:            (e)   => { self._emitError(e); },
      reportWarning:          (w)   => { self._warn(w); },
      teardownPipeline:       ()    => { self._teardownPipeline(); },
      disposeDdgi:            ()    => { self._ddgi.dispose(); },

      recordInitStart: () => {
        const d = self._dbg;
        d.initCount++;
        d.initStart = performance.now();
        console.log(`[hybrid:debug] init #${d.initCount} START`, {
          W: self._width, H: self._height, device: !!self._device,
          t: d.initStart.toFixed(0),
        });
      },
      recordInitComplete: (pipelineMs, totalMs) => {
        const d = self._dbg;
        console.log(`[hybrid:debug] init #${d.initCount} COMPLETE`, {
          pipelineMs: pipelineMs.toFixed(1), totalMs: totalMs.toFixed(1),
        });
      },
    };
  }

}

// ────────────────────────────────────────────────────────────────────────────
// Factory
// ────────────────────────────────────────────────────────────────────────────

/**
 * The walkaround-hybrid backend's STABLE, public-facing surface BEYOND the
 * host-agnostic {@link Engine} contract: DDGI GI-state persistence (the "cached
 * light field"). {@link createWalkaroundEngine_Hybrid} returns
 * `Promise<Engine & HybridEngineGISurface>` so a host that deliberately picks
 * this backend by name gets the export/import methods typed — without the
 * universal contract baking in a backend-specific feature. (`@vitrum/engine`'s
 * `createEngine` facade forwards these same methods via its internal
 * `GIStatePersistable` shape; this is the backend-package-level peer of that.)
 */
/**
 * Create a HybridEngine instance and begin asynchronous pipeline initialisation.
 *
 * The engine is returned immediately in `'initializing'` state. The host
 * should poll `engine.state` or listen for the `'ready'` transition before
 * calling `renderFrame`.
 *
 * @param opts  Creation-time options. `opts.device` must be a live GPUDevice.
 */
export const createWalkaroundEngine_Hybrid: EngineFactory<
  HybridEngineOptions,
  Engine & HybridEngineGISurface
> = async (
  opts: HybridEngineOptions,
// eslint-disable-next-line @typescript-eslint/require-await -- factory signature is async to match EngineFactory<…> contract; no async setup needed in this code path
): Promise<Engine & HybridEngineGISurface> => {
  // Duck-type GPUDevice validation — `instanceof GPUDevice` is not reliable
  // across realms; checking for a known required method is more robust.
  if (
    !opts.device ||
    typeof (opts.device as { createCommandEncoder?: unknown }).createCommandEncoder !== 'function'
  ) {
    throw new TypeError(
      '[createWalkaroundEngine_Hybrid] opts.device must be a live GPUDevice. ' +
      // eslint-disable-next-line @typescript-eslint/no-base-to-string -- diagnostic; [object Object] output is acceptable in a TypeError message
      'Received: ' + String(opts.device),
    );
  }

  const engine = new HybridEngine(opts);
  // Bootstrap setScene with an empty vitrum Scene. Two callers depend on
  // this:
  //   1. Hosts that DO call setScene afterwards (e.g. @vitrum/engine.createEngine).
  //      The host's setScene fires init-B which races init-A. The init-flight
  //      guard inside PipelineInitCoordinator (mySeq === _initSeq) ensures the
  //      loser bootstrap chain disposes its locals — no GPU resource leak.
  //      The bootstrap is wasted work but safe.
  engine.setScene({ primitives: [], emitters: [], environment: { kind: 'none' } });
  return engine;
}
