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
 *   as part of the render-mode-hierarchy restructure (the hybrid shade pass
 *   discarded Lo_rc). The standalone 'rc' walkaround engine still uses the
 *   cascade subsystem; those modules (cascadeDispatch, cascadePyramid,
 *   useCascadeBuffers) remain live for it.
 *   TODO Step 4: re-wire RC composition here when the RC cascade is extracted.
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
   * Stable signal that triggers pipeline reinitialisation when it changes
   * (e.g. room loaded/unloaded, fixture swapped). Pass a string/number/null.
   * The engine calls `reset()` internally when the value changes.
   */
  readonly pipelineRebuildKey?: string | number | null;

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

  /** THREE.Scene reference, used for BVH construction. */
  readonly threeScene: THREE.Scene;

  /** Light list for DDGI probe update pass. */
  readonly lights?: DDGILight[];

  /**
   * When true, enables debug logging and exposes
   * `window.__DDGI__` inside `typeof window !== 'undefined'` guards.
   */
  readonly debug?: boolean;
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
  private readonly _maxBounces:           number;

  // ── Pipeline state ─────────────────────────────────────────────────────
  private _pipeline:    WalkaroundGPUPipeline | null = null;
  private _bvhBuffers:  SceneBVHBuffers | null       = null;
  private _pipelineReady                             = false;

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

  // ── Constructor ────────────────────────────────────────────────────────

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
    this._maxBounces            = opts.maxBounces ?? 4;
    this._isSceneReady          = opts.isSceneReady ?? (() => defaultIsSceneReady(this._threeScene));

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
    };
  }

  // ── Scene management ───────────────────────────────────────────────────

  /**
   * Replace the scene. Triggers a full pipeline reinitialisation
   * (BVH rebuild + ReSTIR pipeline re-init).
   *
   * Per the Engine contract: `supportsIncrementalScene = false`, so the host
   * must call `setScene` for every topology change. This engine ignores the
   * @vitrum/core Scene argument for the BVH build — it traverses the
   * THREE.Scene passed at construction time (the live Three.js graph is the
   * source of truth for geometry). The `_scene` parameter is accepted to
   * satisfy the interface but is not inspected.
   */
  setScene(_scene: Scene): void {
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

    if (this._state === 'paused' || this._state === 'disposed') {
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
        scene:   this._threeScene,
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

    // Clamp bounces to structural cap.
    // (Walkaround ignores samplesTarget — it resamples every frame.)
    const _bounces = Math.min(
      input.quality?.bounces ?? this._maxBounces,
      this._maxBounces,
    );
    void _bounces; // Walkaround pipeline currently uses a fixed bounce count baked at compile time.

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

  pause(): void {
    if (this._state === 'ready') {
      this._state = 'paused';
    }
  }

  resume(): void {
    if (this._state === 'paused') {
      this._state = 'ready';
    }
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
        if (this._isSceneReady()) break;
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
        const bvh = buildSceneBVH([this._threeScene], {
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
        await pipeline.initialize(bvh, getPreferredSwapChainFormat());
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

        // Wire the sun intensity multiplier into DDGI so its Le bake
        // matches shade.wgsl's Lo_emit.
        if ('setSunIntensityMultiplier' in this._ddgi.pass) {
          (this._ddgi.pass as unknown as { setSunIntensityMultiplier: (m: number) => void })
            .setSunIntensityMultiplier(this._primaryLightIntensity);
        }

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
        console.error('[HybridEngine] init failed:', err);
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
export async function createWalkaroundEngine_Hybrid(
  opts: HybridEngineOptions,
): Promise<Engine> {
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
  engine.setScene({} as Scene);
  return engine;
}
