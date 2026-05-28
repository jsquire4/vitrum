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
 * backend interchangeably with `@vitrum/pt-webgl`.
 *
 * RC subsystem: shade pass does not sample Lo_rc — see
 * `plan/walkaround-without-three.md` for the re-integration plan.
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

import * as THREE from 'three';
import type {
  Engine,
  EngineCapabilities,
  EngineDebugSurface,
  EngineFactory,
  EngineState,
  FrameStats,
} from '@vitrum/core';
import { asBackendTexture } from '@vitrum/core';
import type { Scene, ScenePrimitive, SceneEmitter, SceneEnvironment } from '@vitrum/core';
import type { FrameInput, FrameOutput } from '@vitrum/core';
import { DDGI } from './ddgi/DDGI.js';
import type { DDGILight } from './ddgi/types.js';
import { WalkaroundGPUPipeline } from './pipeline/WalkaroundGPUPipeline.js';
import { packDDGIGridParams, type FrameResources } from './pipeline/resourceManager.js';
import { ATROUS_DIRECT_SIGMAS, ATROUS_INDIRECT_SIGMAS } from './pipeline/bindGroupBuilders.js';
import { createHybridEngineDebugSurface } from './HybridEngineDebug.js';
import { disposeSceneBVH } from './restir/bvhCompute.js';
import {
  rebuildEmitterBuffersFromSceneRoots,
  type ReSTIRBvhMode,
  type SceneBVHBuffers,
} from './restir/bvhCompute.js';
import { applyEmitterPatchToScene } from './scenePatch.js';
import {
  disposeVitrumThreeSceneRoot,
  solveSkin,
  vitrumSceneToThree,
} from '@vitrum/three-bindings';
import type { ModelWeights } from './neural/weights.js';
import {
  transformRefit,
  positionsRefit,
  topologyRebuild,
  materialPatch,
  refitSkinnedMeshAfterGpuWrite,
  type PrimitiveUpdateContext,
  type PrimitiveUpdateResult,
} from './HybridEnginePrimitiveUpdates.js';
import {
  PipelineInitCoordinator,
  collectDDGILightsFromThreeRoot,
  type PipelineInitHost,
} from './HybridEngineLifecycle.js';
import { readTunables, readInitTunables, type Tunables, type InitTunables } from './HybridEngineTuning.js';
import type { HybridEngineOptions, LightingOptions } from './HybridEngineOptions.js';
import { RCSubsystem } from './HybridEngineRC.js';
import { GpuSkinningSubsystem } from './skin/GpuSkinningSubsystem.js';

// Re-export the option / lighting interfaces from their dedicated module so
// the package's public surface (`./HybridEngine.js` import path) stays
// unchanged after the type split (refactor sweep 2026-05-18).
export type { HybridEngineOptions, LightingOptions } from './HybridEngineOptions.js';

/** Default per-frame target interval (~60 FPS soft-cap). */
const DEFAULT_TARGET_FRAME_INTERVAL_MS = 1000 / 60 - 1;

// `HybridEngineOptions` + `LightingOptions` interface bodies live in
// `HybridEngineOptions.ts` (~340 LOC of pure JSDoc, extracted refactor sweep
// 2026-05-18). Re-exported above so the package surface is unchanged.

// ────────────────────────────────────────────────────────────────────────────
// Scene-readiness helper
// ────────────────────────────────────────────────────────────────────────────

/** Default scene-readiness predicate: counts triangles via THREE.Scene.traverse. */
function defaultIsSceneReady(scene: THREE.Scene): boolean {
  let total = 0;
  scene.traverse((obj) => {
    const mesh = obj as THREE.Mesh;
    if (!mesh.isMesh || !mesh.geometry) return;
    const idx = mesh.geometry.index;
    total += idx ? idx.count / 3 : (mesh.geometry.attributes['position']?.count ?? 0) / 3;
  });
  // Audit M5: was `total >= 200`, calibrated to Cornell-scale scenes.
  // Procedurally-generated terrain or sparse-geometry scenes may have far
  // fewer triangles and a perfectly valid BVH; the 200 floor silently
  // blocked them. Hosts with stricter readiness signals supply their own
  // `isSceneReady` callback.
  return total > 0;
}

/** Get the preferred swap-chain format from the browser GPU. */
function getPreferredSwapChainFormat(): GPUTextureFormat {
  return (typeof navigator !== 'undefined' && 'gpu' in navigator
    ? (navigator.gpu as { getPreferredCanvasFormat?: () => GPUTextureFormat })
        .getPreferredCanvasFormat?.() ?? 'bgra8unorm'
    : 'bgra8unorm');
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
  private _width:                number;
  private _height:               number;
  /** Optional escape-hatch THREE.Scene from ctor opts. Null when the host
   *  goes through the canonical setScene(vitrumScene) path (T3.H removal). */
  private readonly _threeScene:           THREE.Scene | null;
  /** Lazily-synthesized THREE.Scene root from the most recent vitrum
   *  setScene() — caches `vitrumSceneToThree(_lastScene)` so DDGI updateFrame
   *  doesn't re-traverse on every frame. Reset on every setScene(). */
  private _synthesizedThreeScene:         THREE.Scene | null = null;
  private readonly _isSceneReady:         () => boolean;
  // Lighting fields are NOT readonly — updateLighting() mutates them at runtime.
  private _primaryLightDir:               [number, number, number];
  private _primaryLightIntensity:         number;
  private _skyTint:                       [number, number, number];
  private _skyIrradiance:                 number;
  private readonly _ctorLights:           readonly DDGILight[];
  private readonly _debug:                boolean;
  private readonly _verbose:             boolean;
  private readonly _maxBounces:           number;

  /** Rolling window of per-frame timings (newest last, cap 240 entries).
   *  Only populated when `debug === true`. Hosts that want a UI gauge
   *  should poll {@link debugTimings} instead of reaching into globals. */
  private readonly _debugTimings: Array<{ t: number; ms: number }> = [];

  /** T3.E — telemetry subscribers fired at end of each successful renderFrame. */
  private readonly _frameSubs: Array<(s: FrameStats) => void> = [];

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
    const p = this._pipeline as unknown as {
      lastGpuTimings?: Record<string, number>;
    } | null;
    return p?.lastGpuTimings ?? {};
  }

  /** Frame index that produced {@link lastGpuTimings}; -1 if no readback
   *  has resolved yet. */
  get lastGpuTimingsFrame(): number {
    const p = this._pipeline as unknown as {
      lastGpuTimingsFrame?: number;
    } | null;
    return p?.lastGpuTimingsFrame ?? -1;
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
    const p = this._pipeline as unknown as {
      readGpuTimingsOnce?: () => Promise<{ perPass: Record<string, number>; rawBigints: string[] }>;
    } | null;
    if (!p?.readGpuTimingsOnce) return { perPass: {}, rawBigints: [] };
    return p.readGpuTimingsOnce();
  }

  private readonly _denoiser: 'atrous' | 'atrous-variance' | 'svgf-real' | 'neural' | 'oidn-final';
  /** Dev A/B — mirrors `engine.debug.setDenoiserEnabled` (default on). */
  private _denoiserPassEnabled = true;
  /** T2.H2 — neural denoiser weights (populated when _denoiser === 'neural'). */
  private readonly _neuralWeights: ModelWeights | undefined;
  /** W11 — OIDN config (populated when _denoiser === 'oidn-final'). */
  private readonly _oidnModelUrl: string | undefined;
  private readonly _oidnExecutionProviders: ReadonlyArray<'webnn' | 'webgpu' | 'wasm'> | undefined;
  /** PR-2 — optional ReSTIR CPU pack mode override (`extensions['walkaround-hybrid'].bvhMode`). */
  private readonly _restirBvhModeOverride: ReSTIRBvhMode | undefined;
  /** Audit M4 — null disables the FPS cap; configured at construction. */
  private readonly _targetFrameIntervalMs: number | null;
  /** Per-frame audit tunable record (frozen; spread into pipeline.renderFrame).
   *  See `HybridEngineTuning.ts`. */
  private readonly _tunables: Tunables;
  /** Init-time audit tunable record (frozen; passed into pipeline.initialize). */
  private readonly _initTunables: InitTunables;
  /** 2026-05-18 sweep — per-channel HDR clamp on indirect-radiance.  Tuple-typed,
   *  so it lives here instead of the number-typed `Tunables` table. */
  private readonly _indirectFireflyClamp: readonly [number, number, number];
  /** 2026-05-19 B3a — atrous DIRECT-channel sigmas [sigmaN, sigmaZ, sigmaC].
   *  Cornell default `[128.0, 5.0, 0.05]`. Tuple-typed so it lives on the
   *  engine directly rather than in the number-only `Tunables` table. */
  private readonly _atrousDirectSigmas: readonly [number, number, number];
  /** 2026-05-19 B3a — atrous INDIRECT-channel sigmas [sigmaN, sigmaZ, sigmaC].
   *  Cornell default `[32.0, 20.0, 0.5]`. Broader on every axis since
   *  ReSTIR-GI already smooths the indirect signal. */
  private readonly _atrousIndirectSigmas: readonly [number, number, number];

  // ── Pipeline state ─────────────────────────────────────────────────────
  private _pipeline:    WalkaroundGPUPipeline | null = null;
  private _bvhBuffers:  SceneBVHBuffers | null       = null;

  // ── Scene (from @vitrum/core contract) ────────────────────────────────
  /** Last scene passed via `setScene()`. When it contains `mesh` primitives,
   *  ReSTIR BVH + DDGI probe walks use `vitrumSceneToThree(this._lastScene)`. */
  private _lastScene: Scene | null = null;

  /** Owned `THREE.Scene` from `vitrumSceneToThree` when BVH/DDGI follow the core
   *  contract; disposed on pipeline teardown. Null when falling back to ctor `threeScene`. */
  private _ddgiTraversalScene: THREE.Scene | null = null;

  // ── DDGI subsystem ─────────────────────────────────────────────────────
  private _ddgi:    DDGI;
  private _ddgiOn:  boolean = true;

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

  // Underscore-prefixed forwarders for the coordinator's race-tracking
  // state. These exist so existing dispose / init-race tests that reach
  // in via `engine['_initSeq']` / `engine['_pendingTeardown']` /
  // `engine['_initRunning']` continue to see the live values without
  // duplicating state on the engine. They're test seams, not part of any
  // documented engine surface.
  private get _initSeq(): number { return this._initCoordinator.initSeq; }
  private get _initRunning(): boolean { return this._initCoordinator.initRunning; }
  private get _pendingTeardown(): boolean { return this._initCoordinator.pendingTeardown; }
  private get _disposed(): boolean { return this._initCoordinator.disposed; }

  /** Monotonic fingerprint of {@link HybridEngineOptions.pipelineRebuildKey} /
   *  {@link HybridEngineOptions.getPipelineRebuildKey} — changes trigger `reset()`. */
  private _rebuildKeyFingerprintSeen: string;

  private readonly _staticPipelineRebuildKey: string | number | null;
  private readonly _getPipelineRebuildKey: (() => string | number | null | undefined) | undefined;
  private readonly _skinning: GpuSkinningSubsystem | null;

  readonly debug: EngineDebugSurface;

  constructor(opts: HybridEngineOptions) {
    this._device                = opts.device;
    this._width                 = opts.width;
    this._height                = opts.height;
    this._skinning              = opts.gpuSkinning
      ? new GpuSkinningSubsystem(opts.device, true)
      : null;
    this._threeScene            = opts.threeScene ?? null;
    this._primaryLightDir       = opts.primaryLightDir;
    this._primaryLightIntensity = opts.primaryLightIntensity;
    this._skyTint               = opts.skyTint;
    this._skyIrradiance         = opts.skyIrradiance;
    this._debug                 = opts.debug ?? false;
    this._verbose               = opts.verbose ?? false;
    this._maxBounces            = opts.maxBounces ?? 4;
    // Audit B7: validate the denoiser option at construction so an unsupported
    // value (e.g. `'none'`, `'bmfr'` from the @vitrum/core EngineOptions
    // contract) does not silently coerce to atrous-variance and produce
    // wrong output. Supported values are explicitly enumerated here.
    if (
      opts.denoiser !== undefined &&
      opts.denoiser !== 'atrous' &&
      opts.denoiser !== 'atrous-variance' &&
      opts.denoiser !== 'svgf-real' &&
      opts.denoiser !== 'neural' &&
      opts.denoiser !== 'oidn-final'
    ) {
      throw new TypeError(
        `[HybridEngine] unsupported denoiser '${opts.denoiser}'. ` +
        `walkaround-hybrid supports: 'atrous' | 'atrous-variance' | 'svgf-real' | 'neural' | 'oidn-final'. ` +
        `If you need 'none' / 'bmfr' from @vitrum/core, pick a backend that implements those modes.`,
      );
    }
    // T2.H2 — 'neural' requires neuralWeights to be provided.
    if (opts.denoiser === 'neural' && !opts.neuralWeights) {
      throw new TypeError(
        `[HybridEngine] denoiser: 'neural' requires neuralWeights to be provided. ` +
        `Load weights via loadWeightsFromArrayBuffer() from a .vitrum-model file, ` +
        `or train one with tools/neural-denoiser-training/train.py. ` +
        `See tools/neural-denoiser-training/README.md for instructions.`,
      );
    }
    // W11 — 'oidn-final' requires extensions['walkaround-hybrid'].oidnModelUrl.
    const _whExt = (opts.extensions as undefined | {
      'walkaround-hybrid'?: {
        oidnModelUrl?: string;
        oidnExecutionProviders?: ReadonlyArray<'webnn' | 'webgpu' | 'wasm'>;
        bvhMode?: 'merged' | 'tlas';
      };
    })?.['walkaround-hybrid'];
    const _oidnModelUrl = _whExt?.oidnModelUrl;
    if (opts.denoiser === 'oidn-final' &&
        (typeof _oidnModelUrl !== 'string' || _oidnModelUrl.length === 0)) {
      throw new TypeError(
        `[HybridEngine] denoiser: 'oidn-final' requires ` +
        `extensions['walkaround-hybrid'].oidnModelUrl (non-empty string) ` +
        `pointing at the bundled OIDN ONNX model file ` +
        `(e.g. '/models/oidn_rt_hdr_alb_nrm.onnx'). ` +
        `See plan/premium-grade-refactor-20260517.md §W11 + ` +
        `packages/shared-denoisers/src/oidnBridge.ts for the model-URL convention.`,
      );
    }
    this._denoiser = opts.denoiser ?? 'atrous-variance';
    this._neuralWeights = opts.neuralWeights;
    this._oidnModelUrl = _oidnModelUrl;
    this._oidnExecutionProviders = _whExt?.oidnExecutionProviders;
    this._restirBvhModeOverride = _whExt?.bvhMode;
    this._targetFrameIntervalMs = opts.targetFrameIntervalMs !== undefined
      ? opts.targetFrameIntervalMs
      : DEFAULT_TARGET_FRAME_INTERVAL_MS;
    // Library-generality tunables — table-driven; defaults preserve Cornell
    // behaviour, hosts override via HybridEngineOptions.
    this._tunables     = readTunables(opts);
    this._initTunables = readInitTunables(opts);
    // 2026-05-18 sweep — `indirectFireflyClamp` is tuple-typed so it lives
    // outside the number-typed Tunables table; default preserves Cornell.
    this._indirectFireflyClamp = opts.indirectFireflyClamp ?? [1.0, 1.0, 1.0];
    // 2026-05-19 B3a — atrous DIRECT/INDIRECT sigmas; tuple-typed same as
    // indirectFireflyClamp. Defaults sourced from the single-source-of-truth
    // constants in bindGroupBuilders.ts (no duplicated literals).
    this._atrousDirectSigmas   = opts.atrousDirectSigmas
      ?? [ATROUS_DIRECT_SIGMAS.sigmaN, ATROUS_DIRECT_SIGMAS.sigmaZ, ATROUS_DIRECT_SIGMAS.sigmaC];
    this._atrousIndirectSigmas = opts.atrousIndirectSigmas
      ?? [ATROUS_INDIRECT_SIGMAS.sigmaN, ATROUS_INDIRECT_SIGMAS.sigmaZ, ATROUS_INDIRECT_SIGMAS.sigmaC];
    // Default predicate: ready when EITHER the vitrum Scene supplies any mesh
    // primitive OR the optional escape-hatch THREE.Scene contains triangles.
    // Hosts override via opts.isSceneReady when they need a scene-specific
    // signal (e.g. wait for an async asset).
    this._isSceneReady          = opts.isSceneReady ?? (() => {
      if (this._coreSceneSuppliesMeshes()) return true;
      return this._threeScene != null && defaultIsSceneReady(this._threeScene);
    });

    this._staticPipelineRebuildKey = opts.pipelineRebuildKey ?? null;
    this._getPipelineRebuildKey     = opts.getPipelineRebuildKey;
    this._rebuildKeyFingerprintSeen = HybridEngine._fingerprintRebuildKey(
      opts.getPipelineRebuildKey?.() ?? opts.pipelineRebuildKey ?? null,
    );

    this._ddgi = new DDGI({ debug: this._debug });
    this._ctorLights = opts.lights ?? [];
    if (this._ctorLights.length > 0) {
      this._ddgi.setLights(this._ctorLights as DDGILight[]);
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
      supportsAuxBuffers:        false,
      accumulates:               false,
      maxSamplesPerPixel:        Infinity,
      maxBounces:                this._maxBounces,
      supportedAnalyticShapes:   new Set(),
      supportedPrimitiveKinds:   new Set<ScenePrimitive['kind']>(['mesh', 'skinned-mesh']),
      // Emitter kinds handled by DDGI _uploadLights: sun, fixture, teaLight
      // mapped to core taxonomy: rect-area/disc-area/mesh-area (disc is
      // converted to rect in the adapter path).
      supportedEmitterKinds:     new Set<SceneEmitter['kind']>(['rect-area', 'disc-area', 'mesh-area']),
      supportedEnvironmentKinds: new Set<SceneEnvironment['kind']>(['none', 'hdri']),
      presentationMode:          'swapchain-required',
      experimentalFeatures:      new Set(['svgf-real-conservative-objid']),
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
      readAtlas: () => this._ddgi?.pass?.getReadAtlasGPUTextures?.() ?? null,
      bvhNodesCpu: () => this._bvhBuffers?.bvhNodes?.cpuData,
      pipelineResources: () =>
        (this._pipeline as { _res?: FrameResources } | null)?._res ?? null,
      denoiserPassEnabled: () => this._denoiserPassEnabled,
      setDenoiserPassEnabled: (enabled) => {
        this._denoiserPassEnabled = enabled;
      },
      setPipelineDenoiserPassEnabled: (enabled) => {
        this._pipeline?.setDenoiserPassEnabled(enabled);
      },
    });
  }

  // ── Scene management ───────────────────────────────────────────────────

  /**
   * Replace the scene. Triggers a full pipeline reinitialisation
   * (BVH rebuild + ReSTIR pipeline re-init).
   *
   * **BVH + DDGI geometry:** When `setScene` receives a `Scene` with at least
   * one `mesh` primitive, ReSTIR `buildSceneBVH` and DDGI probe traversal use
   * `vitrumSceneToThree` (same path as `pt-webgl`). Otherwise the ctor
   * `threeScene` is the BVH + DDGI source (hosts with only Three.js graphs).
   *
   * **Host guidance:** For one `Scene` driving both `pt-webgl` and this engine,
   * pass `setScene(sceneFromThreeJS(yourThreeScene))` and keep ctor `threeScene`
   * in sync for the non-mesh case and for auxiliary Object3D state you have not
   * serialized into the core `Scene`.
   *
   * @param scene - The `@vitrum/core` scene (e.g. from `sceneFromThreeJS`).
   */
  setScene(scene: Scene): void {
    this._lastScene = scene;
    // T3.H removal: drop the cached synthesized THREE.Scene; the next BVH
    // build / DDGI updateFrame will re-derive it from the new vitrum Scene.
    if (this._synthesizedThreeScene != null) {
      try { disposeVitrumThreeSceneRoot(this._synthesizedThreeScene); } catch {}
      this._synthesizedThreeScene = null;
    }

    // W8 Phase 2 — rebuild the RC BVH + cascade buffers against the new
    // scene. Synthesise a fresh THREE root if needed (lazy via
    // `_ensureThreeSceneRoot`). When the host supplied no source, the RC
    // dispatcher stays idle until the next setScene.
    // RC BVH is wired after async ReSTIR BVH publish (see publishBvh).

    // Tear down the existing pipeline, reinitialise asynchronously.
    this._teardownPipeline();
    this._initCoordinator.startInit();
  }

  /** T3.H removal: lazily synthesize a THREE.Scene from the most recent
   *  vitrum Scene if (a) the host did not pass `threeScene` at construction,
   *  and (b) the vitrum Scene supplies meshes. Caller is responsible for
   *  null-checking — if both threeScene and synthesizable are null, the
   *  pipeline will throw at BVH build with a clear message. */
  private _ensureThreeSceneRoot(): THREE.Scene | null {
    if (this._threeScene != null) return this._threeScene;
    if (this._synthesizedThreeScene != null) return this._synthesizedThreeScene;
    if (this._lastScene != null && this._coreSceneSuppliesMeshes()) {
      this._synthesizedThreeScene = vitrumSceneToThree(this._lastScene);
      return this._synthesizedThreeScene;
    }
    return null;
  }

  // ── updatePrimitive — geometry-change path ─────────────────────────────
  //
  // **Scope of this branch (feat/a3-geometry-change-bvh-leaf-rebuild):**
  // implements the **transform-only fast path** + **topology-change full
  // rebuild path** of `Engine.updatePrimitive`. The **material-only fast
  // path** ships separately on `feat/a3-hybridengine-incremental-updates`
  // (commit `d0d22b0`); the merger combines both code paths into a single
  // dispatcher.
  //
  // **Routing rules (this branch alone)**:
  //  - `patch.transform` present AND no topology fields → fast-path (c):
  //     refit the BVH bounds in-place (no SAH rebuild, no pipeline
  //     recompile, no DDGI atlas invalidation), rewrite the affected
  //     primitive's vertex slice in `bvhPositions`, reset the accumulator.
  //  - any topology field present (`positions` / `normals` / `uvs` /
  //     `tangents` / `indices` / `instances` / `params` / `shape` /
  //     `fallbackMesh` / `kind`) → full-rebuild path (a): re-run
  //     `buildReSTIRSceneBVH`, destroy + reupload all four BVH GPU
  //     buffers, reset the accumulator.
  //  - material-only patches → patch scene primitive + rebuild via setScene()
  //     for correctness (material-byte upload optimization can still be
  //     layered later without changing host contract behavior).
  //
  // Implementations live in `HybridEnginePrimitiveUpdates.ts`; this method
  // is the routing dispatcher.
  //
  // Implements `Engine.updatePrimitive(id, patch)` from `@vitrum/core`.
  updatePrimitive(id: string, patch: Partial<ScenePrimitive>): void {
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
    const prim = this._lastScene.primitives[primIndex]!;

    // Three.js BVH refit (fast path) preserves topology when only AABB
    // bounds need updating. Three flavours:
    //   - Transform only       → transformRefit  (~1 ms / 30k tris)
    //   - Positions only       → positionsRefit  (A3, ~1 ms / 30k tris)
    //   - True topology change → topologyRebuild (~50 ms / 30k tris)
    // "Positions only" means new vertex data on the SAME index buffer +
    // SAME vertex count. The positionsRefit path falls through to
    // topologyRebuild internally if the count doesn't match.
    const topologyFields = [
      'normals', 'uvs', 'tangents', 'indices',
      'instances', 'params', 'shape', 'fallbackMesh', 'kind',
    ] as const;
    const hasTopologyChange = topologyFields.some(
      (f) => (patch as Record<string, unknown>)[f] !== undefined,
    );
    const hasPositionsChange = (patch as Record<string, unknown>)['positions'] !== undefined;
    const hasTransformChange = (patch as Record<string, unknown>)['transform'] !== undefined;
    const hasMaterialChange  = (patch as Record<string, unknown>)['material']  !== undefined;

    if (hasTopologyChange) {
      // True topology change — even if `positions` is also in the patch
      // it has to round-trip through the full SAH rebuild because the
      // index buffer / vertex layout changed.
      const result = topologyRebuild(id, patch, this._buildPrimitiveUpdateContext());
      this._bvhBuffers = result.bvhBuffers;
      this._lastScene = result.updatedScene;
      this._applyPrimitiveUpdateSubsystems(result);
      return;
    }
    if (hasPositionsChange) {
      // A3 fast path — same topology, new vertex positions.
      const result = positionsRefit(id, patch, this._buildPrimitiveUpdateContext());
      this._bvhBuffers = result.bvhBuffers;
      this._lastScene = result.updatedScene;
      this._applyPrimitiveUpdateSubsystems(result);
      return;
    }
    if (hasTransformChange) {
      const result = transformRefit(id, patch, this._buildPrimitiveUpdateContext());
      this._bvhBuffers = result.bvhBuffers;
      this._lastScene = result.updatedScene;
      this._applyPrimitiveUpdateSubsystems(result);
      return;
    }
    if (hasMaterialChange) {
      const result = materialPatch(id, patch, this._buildPrimitiveUpdateContext());
      this._bvhBuffers = result.bvhBuffers;
      this._lastScene = result.updatedScene;
      return;
    }

    // No recognised patch field — treat as a no-op rather than throw so
    // hosts can pass through optional patches without checking each
    // field's presence.
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
    this._bvhBuffers = result.bvhBuffers;
    this._lastScene = result.updatedScene;
    this._applyPrimitiveUpdateSubsystems(result);
  }

  /** Merged BVH position SSBO for GPU skinning (null before pipeline init). */
  getGpuSkinningBvhBuffer(): GPUBuffer | null {
    return this._pipeline?.getBvhPositionBuffer() ?? null;
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
    if (this._bvhBuffers != null) {
      this._ddgi.syncRestirBvhBuffers(
        this._bvhBuffers,
        this._lastScene ?? undefined,
      );
    }
    if (!this._rc) return;
    if (result.rcRefitBounds != null) {
      this._rc.refitCascadeBounds(result.rcRefitBounds.min, result.rcRefitBounds.max);
      if (this._bvhBuffers?.bvhMode === 'tlas') {
        this._rc.syncRestirBvhBuffers(this._bvhBuffers);
      }
      return;
    }
    if (this._bvhBuffers?.bvhMode === 'tlas') {
      this._rc.syncRestirBvhBuffers(this._bvhBuffers);
      return;
    }
    const rcRoot = this._ensureThreeSceneRoot();
    if (rcRoot != null) this._rc.setScene(rcRoot);
  }

  /** Build the per-call resource context the primitive-update helpers consume. */
  private _buildPrimitiveUpdateContext(): PrimitiveUpdateContext {
    if (this._lastScene == null) {
      throw new Error(
        'HybridEngine.updatePrimitive: no scene set. Call setScene(scene) first.',
      );
    }
    const ctx: PrimitiveUpdateContext = {
      bvhBuffers:            this._bvhBuffers,
      threeRoot:             this._ensureThreeSceneRoot(),
      pipeline:              this._pipeline,
      ddgi:                  this._ddgi,
      primaryLightDir:       this._primaryLightDir,
      primaryLightIntensity: this._primaryLightIntensity,
      lastScene:             this._lastScene,
    };
    if (this._restirBvhModeOverride !== undefined) {
      return { ...ctx, restirBvhModeOverride: this._restirBvhModeOverride };
    }
    return ctx;
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
    const threeRoot = this._ensureThreeSceneRoot();
    if (threeRoot == null) {
      throw new Error(
        `HybridEngine.updateEmitter("${id}"): no THREE scene available.`,
      );
    }

    this._lastScene = applyEmitterPatchToScene(this._lastScene, id, patch);

    const emitterSlice = rebuildEmitterBuffersFromSceneRoots(
      [threeRoot],
      this._bvhBuffers,
      {
        primaryLightDir: new THREE.Vector3(...this._primaryLightDir),
        primaryLightIntensity: this._primaryLightIntensity,
      },
    );

    this._bvhBuffers = {
      ...this._bvhBuffers,
      emitters: emitterSlice.emitters,
      emitterCdf: emitterSlice.emitterCdf,
      emitterCount: emitterSlice.emitterCount,
      totalEmissivePower: emitterSlice.totalEmissivePower,
    };

    this._pipeline?.updateEmitters(this._bvhBuffers);
    this._syncDdgiLightsFromThreeRoot();
    this._pipeline?.requestAccumReset();
  }

  /** Re-upload DDGI point/rect lights from the live THREE scene (no `setScene`). */
  refreshDdgiLightsFromThreeScene(): void {
    this._syncDdgiLightsFromThreeRoot();
    this._pipeline?.requestAccumReset();
  }

  private _syncDdgiLightsFromThreeRoot(): void {
    const root = this._ensureThreeSceneRoot();
    if (root == null) return;
    const sceneLights = collectDDGILightsFromThreeRoot(root);
    this._ddgi.setLights([...this._ctorLights, ...sceneLights]);
    this._ddgi.invalidateProbeCache();
  }

  // ── Runtime lighting update ────────────────────────────────────────────

  /**
   * Runtime update of the primary directional light + sky parameters.
   *
   * Re-uploads the WalkaroundUBO at the next frame start (the existing UBO
   * uploader reads live engine fields each frame — no frozen snapshot to
   * bust). Invalidates the DDGI probe atlas so it re-converges over the
   * next ~8 frames. Resets the temporal accumulator (history discarded;
   * α=1 for the very next frame).
   *
   * **No pipelines or GPU buffers are recreated.** The only cost is:
   *   - 2 JS field writes per changed field (field + DDGI sun-intensity mirror)
   *   - DDGI re-convergence over the default `STRIDE` frames (~133 ms at
   *     60 FPS)
   *   - 1 frame of temporal-accumulator reset overhead
   *
   * **Use case:** time-of-day scrubbing in stainedGlass without engine
   * teardown. Eliminates the engine-recreation workaround documented at
   * `useVitrumWalkaroundEngine.ts:34` (stainedGlass audit Gap 1).
   *
   * Calling with an empty object (`{}`) is a safe no-op.
   *
   * @param opts - Partial lighting overrides. Omitted fields are unchanged.
   */
  updateLighting(opts: Partial<LightingOptions>): void {
    let changed = false;

    if (opts.primaryLightDir !== undefined) {
      this._primaryLightDir = opts.primaryLightDir;
      changed = true;
      // Mirror into DDGI's sun-intensity multiplier path on the ProbeUpdatePass.
      // The pass uses the sun direction implicitly via the light list; updating
      // the field here ensures renderFrame() passes the new value to the UBO.
    }
    if (opts.primaryLightIntensity !== undefined) {
      this._primaryLightIntensity = opts.primaryLightIntensity;
      changed = true;
      // Keep the DDGI ProbeUpdatePass sun-intensity multiplier in sync so the
      // irradiance atlas re-converges at the correct brightness.
      this._ddgi.pass.setSunIntensityMultiplier(opts.primaryLightIntensity);
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

    // Invalidate the DDGI probe atlas — re-converges from scratch over the
    // next STRIDE frames (~8 frames, ~133 ms at 60 FPS).
    this._ddgi.invalidateProbeCache();

    // Reset the temporal accumulator — history discarded, α=1 for next frame.
    // _pipeline may be null if the engine is still initialising; the flag is
    // applied as soon as the pipeline exists (set before any renderFrame call).
    this._pipeline?.requestAccumReset();
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
    if (width === this._width && height === this._height) return;
    if (width <= 0 || height <= 0) {
      // Defensive: WebGPU createTexture rejects zero-sized textures.
      // Hosts sometimes feed in transient 0-pixel sizes during resize
      // animations — silently ignore.
      return;
    }
    this._width = width;
    this._height = height;
    if (this._pipeline) {
      this._pipeline.resize(width, height);
    }
    // No DDGI invalidation — the irradiance atlas is world-space, not
    // screen-space, so it survives a resize unchanged.
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
   * Note: `input.viewport` is ignored by HybridEngine — its WebGPU render
   * targets (DDGI atlas, ReSTIR reservoirs, history textures, accumulation
   * buffer) are sized at construction and resized only via {@link setSize}.
   * Hosts MUST call `engine.setSize(w, h)` when the canvas dimensions change;
   * pushing a new `viewport` per frame is silently dropped. See the
   * `@vitrum/core` FrameInput.viewport JSDoc for the cross-backend contract.
   */
  renderFrame(input: FrameInput): FrameOutput {
    const skipOutput: FrameOutput = {
      kind: 'skipped',
      samplesAccumulated: 0,
      isConverged: false,
    };

    if (this._state === 'paused' || this._state === 'disposed' || this._state === 'error') {
      return skipOutput;
    }

    const fp = HybridEngine._fingerprintRebuildKey(
      this._getPipelineRebuildKey?.() ?? this._staticPipelineRebuildKey,
    );
    if (fp !== this._rebuildKeyFingerprintSeen) {
      this._rebuildKeyFingerprintSeen = fp;
      this.reset();
      return skipOutput;
    }

    const dbg = this._debug ? this._dbg : null;

    const pipeline = this._pipeline;
    const bvh = this._bvhBuffers;
    if (!pipeline) { if (dbg) dbg.skipNoPipeline++; return skipOutput; }
    if (!bvh)      { if (dbg) dbg.skipNoBvh++;      return skipOutput; }

    const now = performance.now();
    // Audit M4: configurable FPS cap. `null` disables the throttle so VR /
    // 90+ Hz displays get every frame. Default preserves Cornell's 60-FPS
    // soft-cap.
    if (this._targetFrameIntervalMs !== null &&
        this._lastFrameTs !== 0 &&
        now - this._lastFrameTs < this._targetFrameIntervalMs) {
      if (dbg) dbg.skipFrameInterval++;
      // CRITICAL on >60Hz displays: even though the heavy pipeline is
      // throttled to 60 FPS, the host's rAF still acquires a fresh swap-
      // chain texture each tick. Without writing to it, the texture is
      // presented as cleared black → visible dark-flash on every other
      // frame at 120Hz. Blit the most recent resolvedTexture so the
      // canvas keeps showing the previous frame.
      const skipSwapView = input.swapChainView as GPUTextureView | undefined;
      if (skipSwapView && this._pipeline) {
        this._pipeline.presentLastFrame(skipSwapView);
      }
      return skipOutput;
    }
    this._lastFrameTs = now;

    const t0 = now;

    // Core's FrameInput types swap-chain fields opaquely (BackendTexture /
    // BackendTextureFormat). The walkaround backend requires WebGPU; cast at
    // this boundary so the rest of HybridEngine works with concrete types.
    const swapView   = input.swapChainView as GPUTextureView | undefined;
    const swapFmt    = (input.swapChainFormat as GPUTextureFormat | undefined) ?? getPreferredSwapChainFormat();

    if (!swapView) {
      if (dbg) dbg.skipNoSwapView++;
      return skipOutput;
    }

    // Periodic 5s rate report.
    if (dbg) {
      dbg.framesDispatched++;
      if (dbg.lastReportTs === 0) dbg.lastReportTs = now;
      if (now - dbg.lastReportTs > 5_000) {
        const elapsed = (now - dbg.lastReportTs) / 1_000;
        const gpu = (pipeline as unknown as { lastGpuTimings?: Record<string, number> })
          .lastGpuTimings ?? {};
        const gpuTotal = gpu['total'];
        console.log('[hybrid:debug] rate (5s window)', {
          framesDispatched: dbg.framesDispatched,
          fps: (dbg.framesDispatched / elapsed).toFixed(2),
          skipReasons: {
            noPipeline: dbg.skipNoPipeline, noBvh: dbg.skipNoBvh,
            noSwapView: dbg.skipNoSwapView, frameInterval: dbg.skipFrameInterval,
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

    if (this._skinning != null && this._lastScene != null) {
      this._skinning.run(this, this._lastScene);
    }

    // ── DDGI per-frame compute ──────────────────────────────────────────
    // Drive DDGI probe updates as part of this frame tick (fire-and-forget).
    // GPU command queueing (writeBuffer / dispatchWorkgroups / queue.submit) is
    // synchronous from JS — the GPU executes the work asynchronously after the
    // JS tick returns. The atlas is double-buffered, so this frame reads the
    // previous tick's write target while the GPU processes this tick's update.
    // The host MUST NOT call ddgi.updateFrame() separately.
    const ddgiLayerOn = this._ddgiOn && (this._layerEnabled.get('ddgi') ?? true);
    if (ddgiLayerOn) {
      // DDGIFrameInputs now accepts a DDGIDeviceHandle (`device` or a
      // Three.js renderer adapter). HybridEngine owns the device directly.
      // Source order: traversal scene set during BVH init (vitrum-derived),
      // host-provided threeScene escape hatch, lazily-synthesized fallback.
      const ddgiScene = this._ddgiTraversalScene ?? this._ensureThreeSceneRoot();
      if (ddgiScene != null) {
        if (this._bvhBuffers != null) {
          this._ddgi.syncRestirBvhBuffers(this._bvhBuffers, this._lastScene ?? undefined);
        }
        void this._ddgi.updateFrame({
          scene:   ddgiScene,
          device:  this._device,
          enabled: true,
        });
        // 2026-05-18 sweep — forward `glassMixScale` so probeUpdateRays.wgsl
        // reads the host-overridable value instead of a hardcoded const.
        if (this._ddgi.ready) {
          this._ddgi.pass.setGlassMixScale(this._tunables.glassMixScale);
        }
      }
    }

    // ── RC per-frame dispatch (W8 Phase 2) ──────────────────────────────
    // RCSubsystem owns its own BVH + cascade buffers. Dispatch each frame
    // when rcEnabled. Cascade-0 buffer is the indirect-diffuse source for
    // Phase 3's shade.wgsl wiring; Phase 2 just exercises the dispatch path.
    if (this._rc) {
      if (this._bvhBuffers?.bvhMode === 'tlas') {
        this._rc.syncRestirBvhBuffers(this._bvhBuffers);
      }
      this._rc.dispatchFrame({
        sunDirection:        this._primaryLightDir,
        // Multiply sun direction by intensity into a Color-ish [r,g,b].
        // Cornell-default Le for the directional sun is the radiance after
        // primaryLightIntensity scaling; reproduce that here.
        sunColor:            [
          this._primaryLightIntensity,
          this._primaryLightIntensity,
          this._primaryLightIntensity,
        ],
        frameSeed:           input.frameSeed,
        triIntersectEpsilon: this._tunables.triIntersectEpsilon,
      });
      // W8 Phase 3 — surface cascade-0 + rcParams to the shade pass so
      // sampleCascadeC0(...) integrates them into Lo_indirect. The
      // shader's `rcParams.enabled = 1u` activates the MIS mix with
      // weight `_rcWeight`.
      const rcInputs = this._rc.buildRCInputs(this._rcWeight);
      pipeline.setRCInputs(rcInputs);
    } else {
      pipeline.setRCInputs(null);
    }

    // ── DDGI atlas wire ─────────────────────────────────────────────────
    if (!ddgiLayerOn) {
      pipeline.setDDGIInputs(null);
    } else if (this._ddgi.ready) {
      const atlas = this._ddgi.pass.getReadAtlasGPUTextures?.();
      if (atlas) {
        const gridParams = packDDGIGridParams(this._ddgi.probeGrid.params);
        pipeline.setDDGIInputs({
          irradianceTex: atlas.irradiance,
          visibilityTex: atlas.visibility,
          gridParams,
        });
      }
    }

    // ── ReSTIR pipeline renderFrame ─────────────────────────────────────
    const W = this._width;
    const H = this._height;

    const viewMatrix  = input.viewMatrix;
    const projMatrix  = input.projMatrix;
    const prevView    = (input.prevViewMatrix ?? input.viewMatrix);
    const prevProj    = (input.prevProjMatrix ?? input.projMatrix);
    const camPos      = input.cameraPosition as [number, number, number];

    // `FrameInput.quality.bounces` is ignored: ReSTIR + shade WGSL use a fixed
    // path depth baked at shader compile time (see `capabilities.maxBounces`).

    pipeline.renderFrame({
      viewMatrix:            new Float32Array(viewMatrix),
      projMatrix:            new Float32Array(projMatrix),
      prevViewMatrix:        new Float32Array(prevView),
      prevProjMatrix:        new Float32Array(prevProj),
      cameraPos:             camPos,
      screenWidth:           W,
      screenHeight:          H,
      frameSeed:             input.frameSeed,
      totalEmissivePower:    bvh.totalEmissivePower ?? 1.0,
      emitterCount:          bvh.emitters?.count ?? 0,
      primaryLightDir:       this._primaryLightDir,
      primaryLightIntensity: this._primaryLightIntensity,
      skyTint:               this._skyTint,
      skyIrradiance:         this._skyIrradiance,
      // Library-generality audit tunables (HybridEngineTuning.ts). Defaults
      // preserve Cornell behaviour; hosts override via HybridEngineOptions.
      ...this._tunables,
      // 2026-05-18 sweep — tuple-typed clamp lives outside the number-only
      // Tunables table; same library-generality intent.
      indirectFireflyClamp:  this._indirectFireflyClamp,
      bvhMode:               bvh.bvhMode === 'tlas' ? 1 : 0,
      tlasNodeCount:         bvh.tlas?.nodeCount ?? 0,
      // 2026-05-19 B3a — atrous DIRECT/INDIRECT sigmas. Per-frame splat so
      // the atrous denoisers (direct + indirect chains) read overrides
      // without round-tripping through WalkaroundUBO.
      atrousDirectSigmas:    this._atrousDirectSigmas,
      atrousIndirectSigmas:  this._atrousIndirectSigmas,
      swapChainView:         swapView,
      swapChainFormat:       swapFmt,
    });

    const dt = performance.now() - t0;

    // T3.E telemetry. Fired before the legacy debug mirror so subscribers
    // see every frame, not just debug-mode ones. We pull GPU timings from
    // the pipeline's lastGpuTimings if it exposed any (they're populated
    // by the same timestamp-query infrastructure the debug log uses).
    if (this._frameSubs.length > 0) {
      const gpu = (pipeline as unknown as { lastGpuTimings?: Record<string, number> })
        .lastGpuTimings;
      const passTimings = gpu;
      const gpuTotal = gpu?.['total'];
      // GPU memory breakdown — call through the same debug-surface helper
      // so a single source of truth governs the numbers a dev overlay sees
      // via `engine.debug.estimatedGpuMemoryBytes()` vs the streaming
      // FrameStats hook. Cheap (O(field count)), pure host-side arithmetic.
      const memBreakdown = this.debug.estimatedGpuMemoryBytes?.() ?? undefined;
      const stats: FrameStats = {
        frameTimeMs: dt,
        ...(gpuTotal !== undefined ? { gpuTimeMs: gpuTotal } : {}),
        ...(passTimings ? { passTimings } : {}),
        spp: 1,
        ...(memBreakdown ? {
          gpuMemoryBytes: memBreakdown,
          estimatedGpuMemoryBytes: memBreakdown.total,
        } : {}),
      };
      for (const sub of this._frameSubs) {
        try { sub(stats); } catch (err) {
          if (this._verbose) console.warn('[HybridEngine] onFrame subscriber threw', err);
        }
      }
    }

    if (this._debug) {
      this._debugTimings.push({ t: now, ms: dt });
      if (this._debugTimings.length > 240) this._debugTimings.shift();

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

    return {
      kind:               'rendered',
      primaryRadiance:    asBackendTexture<'webgpu', GPUTextureView>(swapView), // swap chain is the output surface
      samplesAccumulated: 1,
      isConverged:        false,      // walkaround never converges; resamples every frame
    };
  }

  // ── Reset ──────────────────────────────────────────────────────────────

  /**
   * Tear down the pipeline and reinitialise from scratch.
   * Hosts call this when the scene changes significantly.
   */
  reset(): void {
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

  // Walkaround engines don't accumulate, so onProgress doesn't have a
  // meaningful 'pt-spp' to report. DDGI warm-up could surface here once
  // we expose probe-update progress; for now the optional method is
  // intentionally absent (consumers must typeof-check per the contract).

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
      // still call _ddgi.pass.setSunIntensityMultiplier() after the
      // post-pipeline checkpoint, and we don't want a torn-down DDGI
      // under it. The chain's finally calls disposeDdgi() when it sees
      // pending teardown.
    }

    if (this._debug && typeof window !== 'undefined') {
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

  /** True when `_lastScene` supplies at least one triangle mesh primitive
   *  (rest-pose skinned meshes count — host pushes deformed positions via
   *  `updatePrimitive`, but the BVH still needs a non-empty scene to build). */
  private _coreSceneSuppliesMeshes(): boolean {
    const s = this._lastScene;
    return s != null && s.primitives.some((p) => p.kind === 'mesh' || p.kind === 'skinned-mesh');
  }

  /**
   * Scene-readiness for BVH build: core mesh payload **or** ctor `isSceneReady`
   * heuristic on the host `threeScene` (proxy-heavy hosts may rely on the latter).
   */
  private _sceneReadyForBvh(): boolean {
    if (this._coreSceneSuppliesMeshes()) return true;
    return this._isSceneReady();
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
    if (this._ddgiTraversalScene) {
      disposeVitrumThreeSceneRoot(this._ddgiTraversalScene);
      this._ddgiTraversalScene = null;
    }
    if (this._state !== 'disposed') {
      this._state = 'initializing';
    }
  }

  /** Build the back-reference the {@link PipelineInitCoordinator} consumes.
   *  All getters close over `this`; the coordinator never sees raw field
   *  references, only the small documented surface in
   *  `HybridEngineLifecycle.ts`. */
  private _buildInitHost(): PipelineInitHost {
    const self = this;
    return {
      get device() { return self._device; },
      get width() { return self._width; },
      get height() { return self._height; },
      get threeScene() { return self._threeScene; },
      get lastScene() { return self._lastScene; },
      get primaryLightDir() { return self._primaryLightDir; },
      get primaryLightIntensity() { return self._primaryLightIntensity; },
      get restirBvhModeOverride() { return self._restirBvhModeOverride; },
      get denoiser() { return self._denoiser; },
      get neuralWeights() { return self._neuralWeights; },
      get oidnModelUrl() { return self._oidnModelUrl; },
      get oidnExecutionProviders() { return self._oidnExecutionProviders; },
      get verbose() { return self._verbose; },
      get debug() { return self._debug; },
      get cameraMoveResetThresholdSq() { return self._initTunables.cameraMoveResetThresholdSq; },
      get temporalAccumAlpha() { return self._initTunables.temporalAccumAlpha; },
      get ctorLights() { return self._ctorLights; },
      get ddgi() { return self._ddgi; },
      get preferredSwapChainFormat() { return getPreferredSwapChainFormat(); },
      get currentBvhBuffers() { return self._bvhBuffers; },
      get currentTraversalScene() { return self._ddgiTraversalScene; },

      isSceneReadyForBvh: () => self._sceneReadyForBvh(),
      coreSceneSuppliesMeshes: () => self._coreSceneSuppliesMeshes(),

      publishBvh:             (bvh) => {
        self._bvhBuffers = bvh;
        self._ddgi.syncRestirBvhBuffers(bvh, self._lastScene ?? undefined);
        if (self._rc == null) return;
        if (bvh.bvhMode === 'tlas') {
          self._rc.syncRestirBvhBuffers(bvh);
        } else {
          const rcRoot = self._ensureThreeSceneRoot();
          if (rcRoot != null) self._rc.setScene(rcRoot);
        }
      },
      publishTraversalScene:  (s)   => { self._ddgiTraversalScene = s; },
      publishPipeline:        (p)   => { self._pipeline = p; },
      rollbackBvh:            ()    => { self._bvhBuffers = null; },
      rollbackTraversalScene: ()    => { self._ddgiTraversalScene = null; },
      setState:               (s)   => { self._state = s; },
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

  // ── Private static helpers ─────────────────────────────────────────────

  private static _fingerprintRebuildKey(key: string | number | null | undefined): string {
    if (key === null || key === undefined) return '__null';
    if (typeof key === 'number') return Number.isNaN(key) ? '__n:NaN' : `__n:${key}`;
    return `__s:${key}`;
  }
}

// ────────────────────────────────────────────────────────────────────────────
// Factory
// ────────────────────────────────────────────────────────────────────────────

/**
 * Create a HybridEngine instance and begin asynchronous pipeline initialisation.
 *
 * The engine is returned immediately in `'initializing'` state. The host
 * should poll `engine.state` or listen for the `'ready'` transition before
 * calling `renderFrame`.
 *
 * @param opts  Creation-time options. `opts.device` must be a live GPUDevice.
 */
export const createWalkaroundEngine_Hybrid: EngineFactory<HybridEngineOptions> = async (
  opts: HybridEngineOptions,
): Promise<Engine> => {
  // Duck-type GPUDevice validation — `instanceof GPUDevice` is not reliable
  // across realms; checking for a known required method is more robust.
  if (
    !opts.device ||
    typeof (opts.device as { createCommandEncoder?: unknown }).createCommandEncoder !== 'function'
  ) {
    throw new TypeError(
      '[createWalkaroundEngine_Hybrid] opts.device must be a live GPUDevice. ' +
      'Received: ' + String(opts.device),
    );
  }

  const engine = new HybridEngine(opts);
  // Bootstrap setScene with an empty vitrum Scene. Two callers depend on
  // this:
  //   1. Hosts that pass `threeScene` at construction and never call setScene
  //      themselves (e.g. examples/two-engines-one-scene). Without the
  //      bootstrap they'd never trigger _initPipeline → engine stays
  //      'uninitialized' → renderFrame returns skip output forever.
  //   2. Hosts that DO call setScene afterwards (e.g. @vitrum/engine.createEngine).
  //      The host's setScene fires init-B which races init-A. The init-flight
  //      guard inside PipelineInitCoordinator (mySeq === _initSeq) ensures the
  //      loser bootstrap chain disposes its locals — no GPU resource leak.
  //      The bootstrap is wasted work but safe.
  //
  // We could remove the bootstrap and require all hosts to call setScene
  // explicitly, but that would silently break case 1 and offer no safety
  // benefit (the init-flight guard already eliminates the race-leak class).
  engine.setScene({ primitives: [], emitters: [], environment: { kind: 'none' } });
  return engine;
}
