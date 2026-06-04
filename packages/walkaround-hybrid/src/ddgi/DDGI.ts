/**
 * DDGI — class-based DDGI probe grid lifecycle.
 *
 * De-React-ified from `useDDGI.ts`. All React hooks stripped:
 *   - useRef       → private class fields
 *   - useEffect    → constructor + dispose()
 *   - useCallback  → plain method
 *   - useFrame     → updateFrame()
 *   - useThree     → renderer / scene passed as arguments
 *   - useSelector  → lights passed explicitly to setLights()
 *
 * Normal usage via HybridEngine:
 *   1. `new DDGI(opts)` — creates BVH, ProbeGrid, ProbeUpdatePass in memory.
 *   2. `setLights(lights)` whenever the light list changes.
 *   3. `HybridEngine.renderFrame()` drives `updateFrame()` internally.
 *   4. `dispose()` when the canvas is torn down.
 *
 * Standalone (advanced) usage — DDGI without the full HybridEngine:
 *   Call `updateFrame(inputs)` directly once per animation frame.
 *
 * Debug global: when `opts.debug === true` the class writes
 * `window.__DDGI__` after each updateFrame, guarded by
 * `typeof window !== 'undefined'`. The host is responsible for exposing
 * the `window.__WALKAROUND__.layers.ddgi` and `window.__SET_CAMERA__`
 * globals that the original hook published — these are host-side bridge
 * concerns, not library concerns.
 */

import * as THREE from 'three';
import { SceneBvh } from '@vitrum/shared-bvh';
import type { Scene } from '@vitrum/core';
import { ProbeGrid } from './probeGrid.js';
import type { ProbeGridParams } from './probeGrid.js';
import { ProbeUpdatePass } from './probeUpdatePass.js';
import type { DDGILight } from './types.js';
import { makeDdgiRestirBvhSnapshot, type DdgiRestirBvhSnapshot } from './ddgiRestirBvh.js';
import type { SceneBVHBuffers } from '../restir/bvhCompute.js';

// Default probe round-robin stride. STRIDE=8 means each probe updates every
// 8th frame (~133ms at 60fps). This is the cadence the engine has always
// actually run — it is the divisor used to build the per-frame `activeProbes`
// set (see updateFrame). `setProbeUpdateDivisor` overrides it at runtime; when
// no divisor is set the round-robin falls back to this value.
const DEFAULT_STRIDE = 8;

// Target frame interval for the 60 FPS cap.
const TARGET_FRAME_INTERVAL_MS = 1000 / 60 - 1;

export interface DDGIOptions {
  /**
   * When true, writes debug state to `window.__DDGI__` after each frame.
   * Guarded by `typeof window !== 'undefined'`.
   */
  debug?: boolean;
  /**
   * Maximum number of distinct materials supported by the DDGI probe pass.
   * Forwarded to {@link ProbeUpdatePassOptions.maxMaterials}. Defaults to 64.
   * @since Sprint 16 (M9 audit remediation)
   */
  maxMaterials?: number;
  /**
   * Probe spacing (world units). Passed to `ProbeGrid.computeFromBounds`.
   * `undefined` = auto-derived from scene AABB (`maxSize / 12`).
   * @since Sprint 16 (M11 audit remediation)
   */
  probeSpacing?: number;
  /**
   * Hard cap on probes per axis. Defaults to 16.
   * @since Sprint 16 (M11 audit remediation)
   */
  maxProbesPerAxis?: number;
}

/** Per-frame inputs supplied by the host for a DDGI update tick. */
export interface DDGIFrameInputs {
  /**
   * The THREE.Scene to traverse for the standalone (no-ReSTIR-snapshot) BVH
   * update via {@link SceneBvh.update}. Required for THREE-only standalone DDGI
   * consumers. When {@link coreScene} is also supplied, the core-first path
   * ({@link SceneBvh.updateFromCore}) is preferred and this is unused for the
   * BVH build (it may still be a host-managed throwaway root).
   */
  scene: THREE.Scene;
  /**
   * Optional `@vitrum/core` `Scene` — when present (and no ReSTIR snapshot is
   * active), the standalone DDGI BVH is built core-first via
   * {@link SceneBvh.updateFromCore} (`mergeWorldSpaceFromCore` + THREE-free
   * materials) instead of the THREE `buildSceneBVH` path. The THREE-decouple of
   * the DDGI merged-BVH ingestion (mirrors the ReSTIR-DI emitter decouple). When
   * absent, the THREE {@link scene} path is used (existing behaviour).
   */
  coreScene?: Scene;
  /**
   * Raw WebGPU device. Supply this when the host owns the device directly
   * (e.g. HybridEngine). Either `device` or `renderer` must be present.
   */
  device?: GPUDevice;
  /**
   * Legacy Three.js WebGPURenderer-shaped object. Supported for standalone
   * DDGI consumers that wrap a Three.js renderer. Either `device` or
   * `renderer` must be present — when both are supplied, `renderer.backend.
   * device` takes precedence (matches the original behaviour).
   */
  renderer?: { backend?: { device?: GPUDevice; isWebGPUBackend?: boolean } };
  /**
   * Whether DDGI compute is enabled this frame. When false, updateFrame
   * returns immediately without dispatching any GPU work.
   */
  enabled: boolean;
}

export class DDGI {
  private _bvh:         SceneBvh;
  private _grid:        ProbeGrid;
  private _pass:        ProbeUpdatePass;
  /** PR-5.1 — when set, skips SceneBvh rebuild and uses ReSTIR GPU buffers. */
  private _restirSnapshot: DdgiRestirBvhSnapshot | null = null;
  private _ready:       boolean  = false;
  private _lastFrameMs: number   = 0;
  private _frame:       number   = 0;
  private _inited:      boolean  = false;
  private _gpuOk:       boolean  = false;
  private _lastFrameTs: number   = 0;
  private _debug:       boolean;
  // M11: probe grid parameters forwarded to computeFromBounds each frame.
  private _probeSpacing:      number | undefined;
  private _maxProbesPerAxis:  number;
  // Phase-0 productization (H1) — round-robin probe-update stride. THIS is the
  // load-bearing cadence knob: `updateFrame` builds the per-frame active-probe
  // set from `offset = _frame % _stride; stride = _stride`, and ProbeUpdatePass
  // constructs `activeProbes` from that stride. Defaults to DEFAULT_STRIDE (8);
  // `setProbeUpdateDivisor` overrides it so the quality preset's divisor
  // actually changes how many probes update per frame (previously the divisor
  // only fed a UBO field that no shader reads).
  private _stride:            number = DEFAULT_STRIDE;

  constructor(opts: DDGIOptions = {}) {
    this._debug = opts.debug ?? false;
    this._probeSpacing     = opts.probeSpacing;
    this._maxProbesPerAxis = opts.maxProbesPerAxis ?? 16;
    this._bvh   = new SceneBvh();
    this._grid  = new ProbeGrid();
    this._pass  = new ProbeUpdatePass(this._bvh, this._grid, {
      debug: this._debug,
      ...(opts.maxMaterials !== undefined ? { maxMaterials: opts.maxMaterials } : {}),
    });
  }

  // ── Read-only accessors matching the old DDGIHandle shape ─────────────────

  get bvh():        SceneBvh    { return this._bvh; }
  get probeGrid():  ProbeGrid   { return this._grid; }
  get pass():       ProbeUpdatePass { return this._pass; }
  get ready():      boolean     { return this._ready; }
  get lastFrameMs(): number     { return this._lastFrameMs; }
  get probeCount(): number      { return this._grid.probeCount; }

  /** Number of probe-update passes dispatched since the last
   *  `invalidateProbeCache()` (or construction). Increments once per enabled
   *  `updateFrame` tick; reset to 0 by `invalidateProbeCache()`. Read by
   *  `HybridEngine.onProgress` to compute the `'ddgi-warmup'` fraction —
   *  after `warmupStride` passes the round-robin has touched every probe at
   *  least once (one stratum of `1/stride` probes per pass). */
  get warmupFrame(): number     { return this._frame; }

  /** Round-robin probe-update stride (= the divisor). After `warmupStride`
   *  enabled passes every probe has received ≥1 update, which is exactly when
   *  `ready` flips true. Target for the `'ddgi-warmup'` progress metric. */
  get warmupStride(): number    { return this._stride; }

  // ── Light configuration ───────────────────────────────────────────────────

  /** Replace the current light list. Forwarded to ProbeUpdatePass. */
  setLights(lights: DDGILight[]): void {
    this._pass.setLights(lights);
  }

  // ── Forwarding façade — callers go through DDGI, not DDGI.pass/probeGrid ──

  /**
   * Set the sun-intensity multiplier on the underlying ProbeUpdatePass.
   * Forwarded from `HybridEngine` / `HybridEngineLifecycle` so they don't
   * reach through to `DDGI.pass` directly.
   */
  setSunIntensityMultiplier(m: number): void {
    this._pass.setSunIntensityMultiplier(m);
  }

  /**
   * Set the glass mix scale on the underlying ProbeUpdatePass.
   * Forwarded from `HybridEngineFrameOrchestrator` so it doesn't reach
   * through to `DDGI.pass` directly.
   */
  setGlassMixScale(s: number): void {
    this._pass.setGlassMixScale(s);
  }

  /**
   * Return the read-side atlas GPU textures from the underlying
   * ProbeUpdatePass. Forwarded from `HybridEngineFrameOrchestrator`.
   */
  getReadAtlasGPUTextures(): { irradiance: GPUTexture; visibility: GPUTexture } | null {
    return this._pass.getReadAtlasGPUTextures();
  }

  /**
   * Probe-grid parameters (origin, spacing, dims, atlas sizes).
   * Forwarded from `HybridEngineFrameOrchestrator` so it doesn't reach
   * through to `DDGI.probeGrid` directly.
   */
  get gridParams(): ProbeGridParams {
    return this._grid.params;
  }

  /** Phase-0 productization (H1) — set the round-robin probe-update divisor.
   *  This is now the LOAD-BEARING cadence knob: it sets the round-robin
   *  `_stride`, which directly determines how many probes update each frame
   *  (`ceil(totalProbes / stride)` probes, one stratum of the grid). A higher
   *  divisor ⇒ more strata ⇒ fewer probes per frame ⇒ cheaper but slower GI
   *  response.
   *
   *  Also forwarded to ProbeUpdatePass so its blend/ray UBO `probesPerFrame`
   *  coverage field stays consistent with the actual active set (kept in
   *  lockstep even though no shader currently branches on it).
   *
   *  Clamped to ≥ 1. The default (no call) is DEFAULT_STRIDE = 8. The quality
   *  presets thread an explicit divisor across a 2→32 spread: ultra=2 (fastest
   *  GI cadence), high=4, medium=8 (= the default), low=32 (cheapest). Because
   *  this knob is load-bearing, those preset values directly set how many probes
   *  update per frame. See HybridEngineQualityPreset.ts. */
  setProbeUpdateDivisor(divisor: number): void {
    this._stride = Math.max(1, Math.floor(divisor));
    this._pass.setProbeUpdateDivisor(divisor);
  }

  // ── Probe cache invalidation ──────────────────────────────────────────────

  /**
   * Invalidate the DDGI probe atlas so it re-converges from scratch.
   *
   * Resets `_frame` to 0 and `_ready` to false. On the next `updateFrame`
   * call the blend kernel will fire with `alpha=1` for every texel (history
   * weight = 0), effectively clearing the irradiance + visibility atlases
   * and letting the DDGI update kernel re-converge over the next `_stride`
   * frame window (the probe-update divisor; default 8).
   *
   * Does NOT deallocate GPU textures or touch the BVH — cost is two JS
   * field writes only. Called by `HybridEngine.updateLighting()` when
   * lighting parameters change at runtime.
   */
  invalidateProbeCache(): void {
    this._frame = 0;
    this._ready = false;
  }

  /**
   * PR-5 — TLAS transform-only refit: nudge probe temporal blend without a
   * full atlas wipe (geometry unchanged; instance matrices moved).
   */
  markInstancesDirty(): void {
    this._frame = Math.max(0, this._frame - Math.floor(this._stride / 2));
  }

  /**
   * PR-5.1 — point DDGI probe rays at the same BLAS/TLAS as ReSTIR.
   * Call each frame from HybridEngine when `_bvhBuffers` is ready.
   */
  syncRestirBvhBuffers(buffers: SceneBVHBuffers | null, scene?: Scene): void {
    if (buffers == null) {
      this._restirSnapshot = null;
      this._pass.setRestirBvhSnapshot(null);
      return;
    }
    this._restirSnapshot = makeDdgiRestirBvhSnapshot(buffers, scene);
    this._pass.setRestirBvhSnapshot(this._restirSnapshot);
  }

  // ── Per-frame update ──────────────────────────────────────────────────────

  /**
   * Run one frame of DDGI compute.
   *
   * **Called internally by `HybridEngine.renderFrame` once per frame.**
   * The host does NOT need to call this when using `HybridEngine` — the
   * engine drives it as part of the normal `renderFrame` tick.
   *
   * Advanced standalone DDGI consumers (e.g. a DDGI-only host without the
   * full ReSTIR pipeline) may call this directly. The 60 FPS cap is enforced
   * internally — callers on high-refresh-rate displays will get a no-op on
   * frames that arrive too quickly.
   */
  async updateFrame(inputs: DDGIFrameInputs): Promise<void> {
    if (!inputs.enabled) return;

    // 60 FPS frame cap.
    const now = performance.now();
    if (this._lastFrameTs !== 0 &&
        now - this._lastFrameTs < TARGET_FRAME_INTERVAL_MS) {
      return;
    }
    this._lastFrameTs = now;

    const t0 = now;

    // Resolve renderer-adapter shape from either `device` or `renderer`.
    // Existing renderer wins for back-compat with three.js standalone hosts.
    const rendererAdapter =
      inputs.renderer ??
      (inputs.device
        ? { backend: { device: inputs.device, isWebGPUBackend: true as const } }
        : undefined);
    if (!rendererAdapter) {
      console.warn('[DDGI] updateFrame called without device or renderer; skipping.');
      return;
    }

    // Initialize GPU on first enabled frame (only try once).
    if (!this._inited) {
      this._inited = true;
      const ok = await this._pass.init(rendererAdapter);
      this._gpuOk = ok;
      if (!ok) {
        console.warn('[DDGI] GPU init failed — DDGI compute disabled (scene still renders without indirect).');
        // Don't return — still update BVH + mark ready so waitForFunction
        // gates in tests don't hang.
      }
    }

    if (this._restirSnapshot == null) {
      try {
        // Core-first when a @vitrum/core Scene is supplied (the THREE-decoupled
        // standalone path); else the legacy THREE buildSceneBVH path. Both
        // populate `_bvh.buffers`; the core path additionally fills
        // `coreMaterials` so the probe-material packer skips the THREE round-trip.
        if (inputs.coreScene != null) {
          this._bvh.updateFromCore(inputs.coreScene);
        } else {
          this._bvh.update(inputs.scene);
        }
      } catch (e) {
        console.error('[DDGI] BVH update failed:', e);
      }
    }

    const boundsBox = this._restirSnapshot?.boundingBox ?? this._bvh.buffers?.boundingBox;
    if (boundsBox) {
      this._grid.computeFromBounds(boundsBox, this._probeSpacing, this._maxProbesPerAxis);
      if (this._grid.dirty || !this._grid.irradianceA) {
        this._grid.allocateAtlases();
      }
    }

    // Round-robin: update 1/_stride of probes this frame. `_stride` is the
    // probe-update divisor set via setProbeUpdateDivisor (default 8).
    const stride = this._stride;
    if (this._gpuOk) {
      const offset = this._frame % stride;
      this._frame++;
      try {
        await this._pass.runFrame(rendererAdapter, offset, stride);
      } catch (e) {
        console.error('[DDGI] runFrame error:', e);
      }
    } else {
      this._frame++;
    }

    // Mark ready after the first full cycle (`_stride` frames).
    if (this._frame >= stride) {
      this._ready = true;
    }

    this._lastFrameMs = performance.now() - t0;

    // Debug window global (guarded by typeof + debug flag).
    if (this._debug && typeof window !== 'undefined') {
      (window as unknown as Record<string, unknown>)['__DDGI__'] = {
        ready: this._ready,
        lastFrameMs: this._lastFrameMs,
        probeCount: this._grid.probeCount,
      };
    }
  }

  // ── Dispose ───────────────────────────────────────────────────────────────

  /** Free all GPU resources. Safe to call even before updateFrame. */
  dispose(): void {
    this._pass.dispose();
    this._grid.dispose();
    this._bvh.dispose();
    this._ready   = false;
    this._inited  = false;
    this._gpuOk   = false;
  }
}
