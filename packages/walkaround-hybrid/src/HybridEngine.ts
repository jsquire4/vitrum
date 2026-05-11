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
 * RC subsystem note:
 *   The RC cascade compute was removed from the hybrid engine in 2026-05-08
 *   (shade pass no longer samples Lo_rc). Standalone RC still uses
 *   `rc/cascadeDispatch` and friends. Re-integrating RC into this class is
 *   **tracked design work** — see [plan/walkaround-without-three.md](../../../plan/walkaround-without-three.md)
 *   (“RC re-composition”) for steps and constraints; it is not a one-line TODO.
 *
 * Debug globals:
 *   The original hook wrote to `window.__WGPU__.walkaround` and
 *   `window.__HYBRID_LAYERS__`. Those are host-bridge responsibilities.
 *   This class exposes `setLayerEnabled()` so the host can wire layer
 *   toggles; it calls `window.__WGPU__` only inside a debug branch
 *   guarded by `typeof window !== 'undefined'` and the `debug` option.
 */

import * as THREE from 'three';
import type {
  Engine,
  EngineCapabilities,
  EngineFactory,
  EngineOptions,
  EngineState,
} from '@vitrum/core';
import type { Scene } from '@vitrum/core';
import type { FrameInput, FrameOutput } from '@vitrum/core';
import { DDGI } from './ddgi/DDGI.js';
import type { DDGILight } from './ddgi/types.js';
import { WalkaroundGPUPipeline } from './pipeline/WalkaroundGPUPipeline.js';
import { buildSceneBVH, disposeSceneBVH } from './restir/bvhCompute.js';
import type { SceneBVHBuffers } from './restir/bvhCompute.js';
import { vitrumSceneToThree, disposeVitrumThreeSceneRoot } from '@vitrum/three-bindings';
import { aabbFromBvhPositions, buildPpgUniformGridCells } from './ppg/ppgCellUpload.js';

/** Per-frame target interval (60 FPS soft-cap). */
const TARGET_FRAME_INTERVAL_MS = 1000 / 60 - 1;

// ────────────────────────────────────────────────────────────────────────────
// Public types
// ────────────────────────────────────────────────────────────────────────────

export interface HybridEngineOptions extends EngineOptions {
  /** WebGPU device (narrowed from the opaque `device: unknown` on EngineOptions). */
  readonly device: GPUDevice;

  /** Physical pixel width of the render surface. */
  readonly width: number;

  /** Physical pixel height of the render surface. */
  readonly height: number;

  /**
   * Predicate the engine polls before kicking off ReSTIR pipeline init.
   * Returns true when the scene has enough geometry to build a meaningful BVH.
   * Defaults to the `defaultIsSceneReady` heuristic (>= 200 triangles).
   */
  readonly isSceneReady?: () => boolean;

  /**
   * Stable signal sampled at ctor (`pipelineRebuildKey`) and/or dynamically
   * via {@link getPipelineRebuildKey}. When the effective value changes compared
   * to the previous frame's sample, {@link HybridEngine.reset} runs so the GPU
   * pipeline is recreated (same `_lastScene` / `THREE` graph).
   */
  readonly pipelineRebuildKey?: string | number | null;

  /**
   * Optional callback polled at the **start** of each {@link HybridEngine.renderFrame}
   * (after state guards). Takes precedence over {@link pipelineRebuildKey} when
   * supplied. Enables hosts to invalidate the pipeline without `setScene()`.
   */
  readonly getPipelineRebuildKey?: () => string | number | null | undefined;

  /**
   * Primary directional light direction (world-space, normalised).
   * Used for both BVH-build-time emitter list construction AND per-frame
   * sun-shadow casting. The two MUST match exactly for self-emission Le
   * to reproduce correctly.
   */
  readonly primaryLightDir: [number, number, number];

  /** Primary directional light intensity (linear, unitless). */
  readonly primaryLightIntensity: number;

  /**
   * Diffuse-sky-dome RGB tint. Consumed by the sky-aperture probe and
   * second-bounce sky-miss paths.
   */
  readonly skyTint: [number, number, number];

  /** Sky-dome irradiance scalar paired with skyTint. */
  readonly skyIrradiance: number;

  /** Host `THREE.Scene` — BVH / DDGI fallback when `setScene` has no mesh primitives. */
  readonly threeScene: THREE.Scene;

  /** Light list for DDGI probe update pass. */
  readonly lights?: DDGILight[];

  /** When true, enables informational ReSTIR pipeline logs (initialization / shader compile). */
  readonly verbose?: boolean;

  /**
   * When true, enables debug logging and exposes
   * `window.__DDGI__` inside `typeof window !== 'undefined'` guards.
   */
  readonly debug?: boolean;

  /**
   * Sprint 11 — Enable PPG (path guiding) buffer allocation + training dispatch.
   *
   * When true, the engine allocates PPG storage buffers
   * (cellBuffer, leafBuffer, sampleBuffer, sampleHeadBuffer, kdBuffer) during
   * pipeline initialisation, injects training writes into the shade pass, and
   * dispatches `ppgUpdate` after shade. Sampling guided paths from the learned
   * distribution remains future work.
   *
   * Defaults to false — no behavioural change for existing consumers.
   *
   * @since Sprint 11, 2026-05-09
   */
  readonly ppgEnabled?: boolean;

  /**
   * Post-shade denoiser: `svgf` (default) — temporal Welford + SVGF à-trous;
   * `atrous` — legacy three-pass edge-stopping à-trous only.
   */
  readonly denoiser?: 'atrous' | 'svgf';
}

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
  return total >= 200;
}

/** Get the preferred swap-chain format from the browser GPU. */
function getPreferredSwapChainFormat(): GPUTextureFormat {
  return (typeof navigator !== 'undefined' && 'gpu' in navigator
    ? (navigator.gpu as { getPreferredCanvasFormat?: () => GPUTextureFormat })
        .getPreferredCanvasFormat?.() ?? 'bgra8unorm'
    : 'bgra8unorm') as GPUTextureFormat;
}

// ────────────────────────────────────────────────────────────────────────────
// HybridEngine
// ────────────────────────────────────────────────────────────────────────────

export class HybridEngine implements Engine {

  private static _fingerprintRebuildKey(key: string | number | null | undefined): string {
    if (key === null || key === undefined) return '__null';
    if (typeof key === 'number') return Number.isNaN(key) ? '__n:NaN' : `__n:${key}`;
    return `__s:${key}`;
  }

  // ── Engine contract fields ─────────────────────────────────────────────
  private _state: EngineState = 'uninitialized';
  readonly capabilities: EngineCapabilities;

  get state(): EngineState {
    return this._state;
  }

  // ── Creation-time options (immutable after construction) ───────────────
  private readonly _device:               GPUDevice;
  private readonly _width:                number;
  private readonly _height:               number;
  private readonly _threeScene:           THREE.Scene;
  private readonly _isSceneReady:         () => boolean;
  private readonly _primaryLightDir:      [number, number, number];
  private readonly _primaryLightIntensity:number;
  private readonly _skyTint:              [number, number, number];
  private readonly _skyIrradiance:        number;
  private readonly _debug:                boolean;
  private readonly _verbose:             boolean;
  private readonly _maxBounces:           number;

  // ── Sprint 11 — PPG state ──────────────────────────────────────────────
  /**
   * Whether PPG buffers are allocated. Set at construction from
   * `HybridEngineOptions.ppgEnabled`. May be toggled post-construction
   * via `setPPGEnabled()`, but the buffer allocation change only takes
   * effect on the next `reset()` / `setScene()` cycle (reinitialisation
   * required to resize GPU allocations).
   *
   * Default: false — no behavioural change for existing consumers.
   */
  private _ppgEnabled: boolean;
  private readonly _denoiser: 'atrous' | 'svgf';

  // ── Pipeline state ─────────────────────────────────────────────────────
  private _pipeline:    WalkaroundGPUPipeline | null = null;
  private _bvhBuffers:  SceneBVHBuffers | null       = null;
  private _pipelineReady                             = false;

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

  // ── Initialisation cancellation ───────────────────────────────────────
  /** Set to true by dispose() to cancel an in-flight async init. */
  private _disposed = false;

  /** Monotonic fingerprint of {@link HybridEngineOptions.pipelineRebuildKey} /
   *  {@link HybridEngineOptions.getPipelineRebuildKey} — changes trigger `reset()`. */
  private _rebuildKeyFingerprintSeen: string;

  private readonly _staticPipelineRebuildKey: string | number | null;
  private readonly _getPipelineRebuildKey: (() => string | number | null | undefined) | undefined;

  constructor(opts: HybridEngineOptions) {
    this._device                = opts.device;
    this._width                 = opts.width;
    this._height                = opts.height;
    this._threeScene            = opts.threeScene;
    this._primaryLightDir       = opts.primaryLightDir;
    this._primaryLightIntensity = opts.primaryLightIntensity;
    this._skyTint               = opts.skyTint;
    this._skyIrradiance         = opts.skyIrradiance;
    this._debug                 = opts.debug ?? false;
    this._verbose               = opts.verbose ?? false;
    this._maxBounces            = opts.maxBounces ?? 4;
    this._ppgEnabled            = opts.ppgEnabled ?? false;
    this._denoiser             = opts.denoiser ?? 'svgf';
    this._isSceneReady          = opts.isSceneReady ?? (() => defaultIsSceneReady(this._threeScene));

    this._staticPipelineRebuildKey = opts.pipelineRebuildKey ?? null;
    this._getPipelineRebuildKey     = opts.getPipelineRebuildKey;
    this._rebuildKeyFingerprintSeen = HybridEngine._fingerprintRebuildKey(
      opts.getPipelineRebuildKey?.() ?? opts.pipelineRebuildKey ?? null,
    );

    this._ddgi = new DDGI({ debug: this._debug });
    if (opts.lights && opts.lights.length > 0) {
      this._ddgi.setLights(opts.lights);
    }

    this.capabilities = {
      supportsIncrementalScene:  false,
      supportsMotionBlur:        false,
      supportsAuxBuffers:        false,
      accumulates:               false,
      maxSamplesPerPixel:        Infinity,
      maxBounces:                this._maxBounces,
      supportedAnalyticShapes:   new Set<string>(),
      // Emitter kinds handled by DDGI _uploadLights: sun, fixture, teaLight
      // mapped to core taxonomy: directional, point
      supportedEmitterKinds:     new Set<string>(['directional', 'point']),
      // RFE-05: Real-time caustic strategies (MNEE / photon-map) are not
      // compatible with the walkaround engine's frame cadence. The walkaround
      // engine always reports 'none'; see external_requests/05-manifold-nee.md
      // §4 ("walkaround-hybrid" backend guidance) for the approved approximation
      // path when real-time caustic approximations are added in a future sprint.
      causticStrategy: 'none',
    };
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

    // Tear down the existing pipeline, reinitialise asynchronously.
    this._teardownPipeline();
    void this._initPipeline();
  }

  updatePrimitive?: never;
  updateEmitter?: never;

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
   * this returns a "skip" FrameOutput (samplesAccumulated: 0, isConverged:
   * false, primaryRadiance: null).
   */
  renderFrame(input: FrameInput): FrameOutput {
    const skipOutput: FrameOutput = {
      primaryRadiance: null,
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
    if (this._lastFrameTs !== 0 &&
        now - this._lastFrameTs < TARGET_FRAME_INTERVAL_MS) {
      if (dbg) dbg.skipFrameInterval++;
      return skipOutput;
    }
    this._lastFrameTs = now;

    const t0 = now;

    const swapView   = input.swapChainView;
    const swapFmt    = input.swapChainFormat ?? getPreferredSwapChainFormat();

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

    // ── DDGI per-frame compute ──────────────────────────────────────────
    // Drive DDGI probe updates as part of this frame tick (fire-and-forget).
    // GPU command queueing (writeBuffer / dispatchWorkgroups / queue.submit) is
    // synchronous from JS — the GPU executes the work asynchronously after the
    // JS tick returns. The atlas is double-buffered, so this frame reads the
    // previous tick's write target while the GPU processes this tick's update.
    // The host MUST NOT call ddgi.updateFrame() separately.
    const ddgiLayerOn = this._ddgiOn && (this._layerEnabled.get('ddgi') ?? true);
    if (ddgiLayerOn) {
      // Construct a minimal renderer-adapter so ProbeUpdatePass can access the
      // device on its lazy-init path. HybridEngine owns the device directly,
      // so we synthesise the Three.js WebGPU backend shape it expects.
      const rendererAdapter = {
        backend: { device: this._device, isWebGPUBackend: true as const },
      };
      void this._ddgi.updateFrame({
        scene:   this._ddgiTraversalScene ?? this._threeScene,
        renderer: rendererAdapter,
        enabled: true,
      });
    }

    // ── DDGI atlas wire ─────────────────────────────────────────────────
    if (!ddgiLayerOn) {
      pipeline.setDDGIInputs(null);
    } else if (this._ddgi.ready) {
      const atlas = this._ddgi.pass.getReadAtlasGPUTextures?.();
      if (atlas) {
        const p = this._ddgi.probeGrid.params;
        const gridParams = new ArrayBuffer(64);
        const f32 = new Float32Array(gridParams);
        const u32 = new Uint32Array(gridParams);
        f32[0] = p.origin.x;
        f32[1] = p.origin.y;
        f32[2] = p.origin.z;
        f32[3] = p.spacing;
        u32[4] = p.dims.x;
        u32[5] = p.dims.y;
        u32[6] = p.dims.z;
        u32[7] = 0;
        f32[8]  = p.irradianceAtlasW;
        f32[9]  = p.irradianceAtlasH;
        f32[10] = p.visibilityAtlasW;
        f32[11] = p.visibilityAtlasH;
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

    const viewMatrix  = input.viewMatrix  as Float32Array;
    const projMatrix  = input.projMatrix  as Float32Array;
    const prevView    = (input.prevViewMatrix ?? input.viewMatrix) as Float32Array;
    const prevProj    = (input.prevProjMatrix ?? input.projMatrix) as Float32Array;
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
      swapChainView:         swapView,
      swapChainFormat:       swapFmt,
    });

    const dt = performance.now() - t0;

    // Write debug frame timing to window.__WGPU__.walkaround if host exposed it.
    if (this._debug && typeof window !== 'undefined') {
      const w = window as unknown as { __WGPU__?: { walkaround?: { frameTimings: unknown } } };
      if (w.__WGPU__?.walkaround) {
        const ft = w.__WGPU__.walkaround.frameTimings as Array<{ t: number; ms: number }>;
        if (Array.isArray(ft)) {
          ft.push({ t: now, ms: dt });
          if (ft.length > 240) ft.shift();
        }
      }
    }

    return {
      primaryRadiance:    swapView,   // swap chain is the output surface
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
    void this._initPipeline();
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

  // ── Sprint 11 — PPG toggle ─────────────────────────────────────────────

  /**
   * Enable or disable PPG (path guiding) for subsequent frames.
   *
   * The setting takes effect on reinitialisation (`reset()`): frame resources
   * and pipelines are rebuilt with or without PPG training buffers and dispatch.
   *
   * The no-op guarantee: calling `setPPGEnabled(false)` when PPG was never
   * enabled has zero cost and no side effects. Existing consumers that never
   * call this method are unaffected.
   *
   * @param on - true to enable PPG training path on next reinit; false to disable.
   *
   * @since Sprint 11, 2026-05-09
   */
  setPPGEnabled(on: boolean): void {
    if (this._ppgEnabled === on) return;
    this._ppgEnabled = on;
    // Reinitialize so createFrameResources receives the new ppgEnabled flag.
    this.reset();
    if (this._debug) {
      console.log('[hybrid:debug] setPPGEnabled', { on, reinitialized: true });
    }
  }

  /**
   * Returns whether PPG is currently enabled.
   * Reflects the last call to `setPPGEnabled` or the `ppgEnabled` constructor option.
   *
   * @since Sprint 11, 2026-05-09
   */
  get ppgEnabled(): boolean {
    return this._ppgEnabled;
  }

  // ── Dispose ────────────────────────────────────────────────────────────

  dispose(): void {
    this._disposed = true;
    this._teardownPipeline();
    this._ddgi.dispose();
    this._state = 'disposed';

    if (this._debug && typeof window !== 'undefined') {
      const dbg = this._dbg;
      dbg.disposeCount++;
      const liveMs = dbg.initStart > 0 ? performance.now() - dbg.initStart : 0;
      console.log(`[hybrid:debug] dispose #${dbg.disposeCount}`, {
        ranForMs: liveMs.toFixed(1),
        framesDispatched: dbg.framesDispatched,
        skipReasons: {
          noPipeline: dbg.skipNoPipeline, noBvh: dbg.skipNoBvh,
          noSwapView: dbg.skipNoSwapView, frameInterval: dbg.skipFrameInterval,
        },
      });
    }
  }

  // ── Private helpers ────────────────────────────────────────────────────

  /** True when `_lastScene` supplies at least one triangle mesh primitive. */
  private _coreSceneSuppliesMeshes(): boolean {
    const s = this._lastScene;
    return s != null && s.primitives.some((p) => p.kind === 'mesh');
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
    this._pipelineReady = false;
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

  private async _initPipeline(): Promise<void> {
    if (this._disposed) return;

    const device = this._device;

    if (this._debug) {
      const dbg = this._dbg;
      dbg.initCount++;
      dbg.initStart = performance.now();
      console.log(`[hybrid:debug] init #${dbg.initCount} START`, {
        W: this._width, H: this._height, device: !!device,
        t: dbg.initStart.toFixed(0),
      });
    }

    const buildBVHWhenReady = async () => {
      // Poll until scene has enough geometry (or 5s timeout).
      const pollStart = Date.now();
      let pollIters = 0;
      while (!this._disposed) {
        const elapsed = Date.now() - pollStart;
        if (elapsed >= 5_000) break;
        if (this._sceneReadyForBvh()) break;
        await new Promise<void>((r) => setTimeout(r, 50));
        pollIters++;
      }

      if (this._disposed) {
        if (this._debug) {
          console.log('[hybrid:debug] init aborted during scene-readiness poll', { pollIters });
        }
        return;
      }

      if (this._debug) {
        console.log('[hybrid:debug] scene-ready', { pollIters, elapsed: Date.now() - pollStart });
      }

      try {
        const bvhStart = performance.now();
        const bvhRoot: THREE.Object3D = this._coreSceneSuppliesMeshes()
          ? vitrumSceneToThree(this._lastScene!)
          : this._threeScene;
        if (bvhRoot !== this._threeScene) {
          this._ddgiTraversalScene = bvhRoot as THREE.Scene;
        }
        const bvh = buildSceneBVH([bvhRoot], {
          primaryLightDir:       new THREE.Vector3(...this._primaryLightDir),
          primaryLightIntensity: this._primaryLightIntensity,
        });
        const bvhMs = performance.now() - bvhStart;
        this._bvhBuffers = bvh;

        if (this._debug) {
          console.log('[hybrid:debug] BVH built', {
            bvhMs: bvhMs.toFixed(1),
            triCount: bvh.bvhNodes?.count,
            emitterCount: bvh.emitters?.count,
          });
        }

        const pipeline = new WalkaroundGPUPipeline(device, this._width, this._height);
        const pipelineStart = performance.now();
        await pipeline.initialize(
          bvh,
          getPreferredSwapChainFormat(),
          { ppgEnabled: this._ppgEnabled, verbose: this._verbose || this._debug, denoiser: this._denoiser },
        );
        const pipelineMs = performance.now() - pipelineStart;

        if (this._disposed) {
          if (this._debug) {
            console.log('[hybrid:debug] init aborted post-pipeline.initialize', {
              pipelineMs: pipelineMs.toFixed(1),
            });
          }
          pipeline.dispose();
          return;
        }

        if (this._ppgEnabled) {
          const maxC = pipeline.ppgAllocatedMaxCells;
          if (maxC > 0) {
            const box = aabbFromBvhPositions(bvh.bvhPositions.cpuData, bvh.bvhPositions.count);
            const cells = buildPpgUniformGridCells(box.min, box.max, maxC);
            pipeline.uploadPpgCells(cells, cells.length);
          }
        }

        // Wire the sun intensity multiplier into DDGI so its Le bake
        // matches shade.wgsl's Lo_emit. `setSunIntensityMultiplier` is a
        // public method on ProbeUpdatePass — no cast needed.
        this._ddgi.pass.setSunIntensityMultiplier(this._primaryLightIntensity);

        this._pipeline     = pipeline;
        this._pipelineReady = true;
        this._state        = 'ready';

        if (this._debug) {
          const dbg = this._dbg;
          const totalMs = performance.now() - dbg.initStart;
          console.log(`[hybrid:debug] init #${dbg.initCount} COMPLETE`, {
            pipelineMs: pipelineMs.toFixed(1), totalMs: totalMs.toFixed(1),
          });
        }
      } catch (err) {
        if (!this._disposed) {
          this._state = 'error';
        }
        console.error(
          '[HybridEngine] init failed — engine state set to error. Call dispose() and recreate the engine to retry.',
          err,
        );
      }
    };

    this._state = 'initializing';
    void buildBVHWhenReady();
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
  // _initPipeline is fire-and-forget; the engine transitions to 'ready'
  // when the BVH poll + pipeline compile finishes. The host polls
  // engine.state or observes renderFrame returning samplesAccumulated=0.
  // Bootstrap with a valid empty scene (no primitives, no emitters) so the
  // scene-readiness poll falls through to the host threeScene heuristic.
  engine.setScene({ primitives: [], emitters: [], environment: { kind: 'none' } });
  return engine;
}
