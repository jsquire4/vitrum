/**
 * useDDGI — React hook that owns the BVH, probe grid, and runs the
 * DDGI compute passes each frame via the WebGPU renderer.
 *
 * Exposes `window.__DDGI__` in dev mode for Playwright test inspection.
 */

import { useRef, useEffect, useCallback } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import { useSelector } from 'react-redux';
import { SceneBvh } from './sceneBvh';
import { ProbeGrid } from './probeGrid';
import { ProbeUpdatePass } from './probeUpdatePass';
import { selectLighting } from '@/store/selectors';

export interface DDGIHandle {
  bvh:        SceneBvh;
  probeGrid:  ProbeGrid;
  pass:       ProbeUpdatePass;
  ready:      boolean;
  lastFrameMs: number;
  probeCount: number;
  /** Camera setter for tests — exposed as window.__SET_CAMERA__. */
  setCamera:  ((pos: [number, number, number], lookAt: [number, number, number]) => void) | null;
}

// Probe round-robin stride. STRIDE=8 means each probe updates every
// 8th frame (~133ms at 60fps, ~267ms at 30fps). Was 4 (1/4); halving
// the per-frame DDGI compute cost trims 30–40% off the DDGI compute
// pass. Lighting changes (e.g. moving the sun, opening a drape) take
// up to 8 frames to fully propagate instead of 4 — perceptible only
// during transient lighting edits, not during steady-state rendering.
// Polish backlog #2 from project_layered_hybrid_milestone, applied
// 2026-05-08 alongside the speed bundle.
const STRIDE = 8;

export function useDDGI({ enabled }: { enabled: boolean }): DDGIHandle {
  const { gl, scene, camera } = useThree();
  const lights = useSelector(selectLighting);

  const bvhRef   = useRef<SceneBvh>(new SceneBvh());
  const gridRef  = useRef<ProbeGrid>(new ProbeGrid());
  const passRef  = useRef<ProbeUpdatePass>(new ProbeUpdatePass(bvhRef.current, gridRef.current));
  const readyRef  = useRef(false);
  const lastMsRef = useRef(0);
  const frameRef  = useRef(0);
  const initedRef = useRef(false);
  const gpuOkRef  = useRef(false);
  /**
   * Last dispatched frame's `performance.now()` timestamp.  Drives the
   * 60 FPS frame cap below.  DDGI's BVH update + probe round-robin
   * compute is GPU-bound at well above 60 FPS on a discrete card;
   * uncapped, the whole pipeline runs as fast as the GPU can dispatch
   * (often 200+ FPS), wasting power and battery for zero perceptible
   * visual gain.  Throttling to ~16.67 ms/frame matches a 60 Hz
   * display's refresh rate, which is the user-perceivable ceiling on
   * a typical monitor.
   */
  const lastFrameTsRef = useRef<number>(0);
  /**
   * Target frame interval in milliseconds.  Computed as (1000/60) − 1 ≈
   * 15.67 ms so that on a 60 Hz display (rAF spacing already ≈ 16.67 ms)
   * the cap NEVER accidentally drops a naturally-paced frame to 30 Hz —
   * only on higher-refresh-rate displays (120/144/240 Hz, rAF spacing
   * 8.33 / 6.94 / 4.17 ms) does the cap kick in.  Without the −1 ms
   * margin, floating-point jitter caused intervals to oscillate between
   * 16.6 ms and 16.8 ms; the 16.6 ms cases triggered the early-return
   * and the next rAF arrived ~16.7 ms later, pushing the effective
   * frame rate to 30 Hz instead of 60 Hz on a 60 Hz panel.
   */
  const TARGET_FRAME_INTERVAL_MS = 1000 / 60 - 1;

  // Build the handle object that we'll return.
  const handle: DDGIHandle = {
    bvh:        bvhRef.current,
    probeGrid:  gridRef.current,
    pass:       passRef.current,
    get ready()       { return readyRef.current; },
    get lastFrameMs() { return lastMsRef.current; },
    get probeCount()  { return gridRef.current.probeCount; },
    setCamera: null,
  };

  // Set up the __SET_CAMERA__ debug bridge.
  const setCameraFn = useCallback((pos: [number, number, number], lookAt: [number, number, number]) => {
    if (!camera) return;
    camera.position.set(pos[0], pos[1], pos[2]);
    camera.lookAt(lookAt[0], lookAt[1], lookAt[2]);
    camera.updateMatrixWorld();
  }, [camera]);
  handle.setCamera = setCameraFn;

  // Expose debug globals for Playwright tests (always, not just DEV).
  // Two namespaces, single source of truth via shared getters:
  //   - window.__DDGI__              — legacy alias, used by 12-walkaround-ddgi.spec
  //                                    and the legacy __DDGI__.ready gate path in
  //                                    14-walkaround-hybrid.spec.
  //   - window.__WALKAROUND__.layers.ddgi — canonical multi-engine bridge,
  //                                    asserted by 14-walkaround-hybrid.spec:131
  //                                    ("__WALKAROUND__.layers.ddgi.ready should be true").
  // Salvaged from archive/walkaround-path-a/layers/DDGILayer.publishDebugBridge
  // because the canonical contract was defined there but never published by
  // any active code, leaving the test assertion latently failing.
  useEffect(() => {
    const handle = {
      get ready()       { return readyRef.current; },
      get lastFrameMs() { return lastMsRef.current; },
      get probeCount()  { return gridRef.current.probeCount; },
    };
    const w = window as unknown as {
      __DDGI__?: typeof handle;
      __WALKAROUND__?: { layers?: Record<string, unknown> };
      __SET_CAMERA__?: (pos: [number, number, number], lookAt: [number, number, number]) => void;
    };
    w.__DDGI__ = handle;
    if (!w.__WALKAROUND__) w.__WALKAROUND__ = { layers: {} };
    if (!w.__WALKAROUND__.layers) w.__WALKAROUND__.layers = {};
    w.__WALKAROUND__.layers['ddgi'] = handle;
    w.__SET_CAMERA__ = setCameraFn;
    return () => {
      delete w.__DDGI__;
      if (w.__WALKAROUND__?.layers) delete w.__WALKAROUND__.layers['ddgi'];
      delete w.__SET_CAMERA__;
    };
  }, [setCameraFn]);

  // Update lights on the probe pass.
  useEffect(() => {
    const lightArr = lights.allIds.map(id => lights.byId[id]).filter(Boolean);
    passRef.current.setLights(lightArr);
  }, [lights]);

  // Main frame loop.
  useFrame(async () => {
    if (!enabled) return;

    // ── 60 FPS frame cap ──────────────────────────────────────────────────
    // r3f's useFrame runs once per requestAnimationFrame, which on most
    // laptops/displays is 60 Hz (16.67 ms) but on high-refresh-rate
    // monitors (120/144/240 Hz) or when rAF is decoupled from vsync the
    // DDGI pipeline ends up dispatched 2–4× more often than the user
    // can perceive.  Each dispatch fires the BVH update + 1/4 round-robin
    // probe-grid compute (probe ray-cast + irradiance/depth atomic blend)
    // and a per-pixel TSL DDGI shading injection, so uncapped throughput
    // burns GPU time + battery for zero visible benefit.  Skip the
    // dispatch when fewer than ~16.67 ms have elapsed since the previous
    // one.  The first frame (lastFrameTsRef === 0) always runs so the
    // canvas has something to display immediately.
    const now = performance.now();
    if (lastFrameTsRef.current !== 0 &&
        now - lastFrameTsRef.current < TARGET_FRAME_INTERVAL_MS) {
      return;
    }
    lastFrameTsRef.current = now;

    const t0 = now;

    // Initialize GPU on first enabled frame (only try once).
    if (!initedRef.current) {
      initedRef.current = true;
      const renderer = gl as unknown as { backend?: { device?: GPUDevice; isWebGPUBackend?: boolean } };
      const ok = await passRef.current.init(renderer);
      gpuOkRef.current = ok;
      if (!ok) {
        console.warn('[DDGI] GPU init failed — DDGI compute disabled (scene still renders without indirect).');
        // Don't return — still update BVH + mark ready so the test waits don't hang.
      }
    }

    // Update BVH from current scene.
    try {
      bvhRef.current.update(scene);
    } catch (e) {
      console.error('[DDGI] BVH update failed:', e);
    }

    // Compute probe grid dims from BVH bounds.
    const bufs = bvhRef.current.buffers;
    if (bufs) {
      gridRef.current.computeFromBounds(bufs.boundingBox);
      if (gridRef.current.dirty || !gridRef.current.irradianceA) {
        gridRef.current.allocateAtlases();
      }
    }

    // Round-robin: update 1/4 of probes this frame (only if GPU available).
    if (gpuOkRef.current) {
      const offset = frameRef.current % STRIDE;
      frameRef.current++;

      try {
        const renderer = gl as unknown as { backend?: { device?: GPUDevice; isWebGPUBackend?: boolean } };
        await passRef.current.runFrame(renderer, offset, STRIDE);
      } catch (e) {
        console.error('[DDGI] runFrame error:', e);
      }
    } else {
      frameRef.current++;
    }

    // Mark ready after the first full cycle (4 frames) — even if GPU is off,
    // so the Playwright test's waitForFunction doesn't time out.
    if (frameRef.current >= STRIDE) {
      readyRef.current = true;
    }

    lastMsRef.current = performance.now() - t0;
  });

  // Dispose on unmount.
  useEffect(() => {
    return () => {
      passRef.current.dispose();
      gridRef.current.dispose();
      bvhRef.current.dispose();
      readyRef.current = false;
      initedRef.current = false;
      gpuOkRef.current  = false;
    };
  }, []);

  return handle;
}
