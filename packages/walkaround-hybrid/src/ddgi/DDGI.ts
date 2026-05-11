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
import { ProbeGrid } from './probeGrid.js';
import { ProbeUpdatePass } from './probeUpdatePass.js';
import type { DDGILight } from './types.js';

// Probe round-robin stride. STRIDE=8 means each probe updates every
// 8th frame (~133ms at 60fps). See useDDGI.ts for full commentary.
const STRIDE = 8;

// Target frame interval for the 60 FPS cap (see useDDGI.ts for rationale).
const TARGET_FRAME_INTERVAL_MS = 1000 / 60 - 1;

export interface DDGIOptions {
  /**
   * When true, writes debug state to `window.__DDGI__` after each frame.
   * Guarded by `typeof window !== 'undefined'`.
   */
  debug?: boolean;
}

/** Per-frame inputs supplied by the host for a DDGI update tick. */
export interface DDGIFrameInputs {
  /**
   * The THREE.Scene to traverse for BVH update.
   */
  scene: THREE.Scene;
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
  private _ready:       boolean  = false;
  private _lastFrameMs: number   = 0;
  private _frame:       number   = 0;
  private _inited:      boolean  = false;
  private _gpuOk:       boolean  = false;
  private _lastFrameTs: number   = 0;
  private _debug:       boolean;

  constructor(opts: DDGIOptions = {}) {
    this._debug = opts.debug ?? false;
    this._bvh   = new SceneBvh();
    this._grid  = new ProbeGrid();
    this._pass  = new ProbeUpdatePass(this._bvh, this._grid, { debug: this._debug });
  }

  // ── Read-only accessors matching the old DDGIHandle shape ─────────────────

  get bvh():        SceneBvh    { return this._bvh; }
  get probeGrid():  ProbeGrid   { return this._grid; }
  get pass():       ProbeUpdatePass { return this._pass; }
  get ready():      boolean     { return this._ready; }
  get lastFrameMs(): number     { return this._lastFrameMs; }
  get probeCount(): number      { return this._grid.probeCount; }

  // ── Light configuration ───────────────────────────────────────────────────

  /** Replace the current light list. Forwarded to ProbeUpdatePass. */
  setLights(lights: DDGILight[]): void {
    this._pass.setLights(lights);
  }

  // ── Per-frame update ──────────────────────────────────────────────────────

  /**
   * Run one frame of DDGI compute. Mirrors the body of the original
   * `useDDGI` useFrame callback exactly.
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

    // 60 FPS frame cap (preserves useDDGI behaviour).
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

    // Update BVH from current scene.
    try {
      this._bvh.update(inputs.scene);
    } catch (e) {
      console.error('[DDGI] BVH update failed:', e);
    }

    // Compute probe grid dims from BVH bounds.
    const bufs = this._bvh.buffers;
    if (bufs) {
      this._grid.computeFromBounds(bufs.boundingBox);
      if (this._grid.dirty || !this._grid.irradianceA) {
        this._grid.allocateAtlases();
      }
    }

    // Round-robin: update 1/STRIDE of probes this frame.
    if (this._gpuOk) {
      const offset = this._frame % STRIDE;
      this._frame++;
      try {
        await this._pass.runFrame(rendererAdapter, offset, STRIDE);
      } catch (e) {
        console.error('[DDGI] runFrame error:', e);
      }
    } else {
      this._frame++;
    }

    // Mark ready after the first full cycle (STRIDE frames).
    if (this._frame >= STRIDE) {
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
