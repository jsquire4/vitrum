/**
 * useHybridLayeredGI — pure-logic hook owning the layered hybrid path
 * tracer's compute + render lifecycle.
 *
 * Lifts every piece of GI plumbing out of HybridLayeredStage so the
 * stage component can be a thin app-specific shell (Redux selectors,
 * scene primitives, JSX) and the GI pipeline can drop into any other
 * R3F scene with the right inputs.
 *
 * Owned by this hook:
 *  - DDGI probe-grid compute (via the standalone `useDDGI` hook)
 *  - RC cascade BVH build + per-frame dispatch (priority=0 useFrame)
 *  - ReSTIR pipeline init/dispose lifecycle
 *  - ReSTIR per-frame renderFrame (priority=1 useFrame), wiring DDGI
 *    atlas + RC cascade GPUBuffers into the shade pass each tick
 *
 * NOT owned by this hook (caller's responsibility):
 *  - Scene mounting (FaceRenderer, EdgeLines, RoomLoader, MountDispatch, etc.)
 *  - Light source selection (caller passes primaryLightDir/Intensity)
 *  - Scene-readiness predicate (caller injects via opts.isSceneReady)
 *  - Camera framing (caller positions camera before calling this hook)
 *  - SwiftShader refusal banner (caller renders banner when !opts.enabled)
 */

import { useEffect, useRef, useState } from 'react';
import type * as THREE from 'three';
import { useFrame, useThree } from '@react-three/fiber';
import { useDDGI, type DDGIHandle } from './useDDGI';
import { WalkaroundGPUPipeline } from './engines/restir/WalkaroundGPUPipeline';
import { buildSceneBVH, disposeSceneBVH, type SceneBVHBuffers } from './engines/restir/bvhCompute';

/** Per-frame target interval (60 FPS soft-cap). */
const TARGET_FRAME_INTERVAL_MS = 1000 / 60 - 1;

export interface HybridLayeredGIOpts {
  /** Master enable. When false the hook short-circuits every effect/frame
   *  so a SwiftShader-refusal banner can render in place of the GI canvas. */
  enabled: boolean;
  /** WebGPU device, typically read from window.__WGPU__.device. Hook waits
   *  for this to be non-null before initializing the ReSTIR pipeline. */
  device: GPUDevice | null;
  /** Primary directional light direction (world-space, normalized). Used
   *  for both BVH-build-time emitter list construction AND per-frame
   *  sun-shadow casting. The two MUST match exactly for self-emission Le
   *  to reproduce — see WalkaroundGPUPipeline.PipelineFrameInputs. */
  primaryLightDir: THREE.Vector3;
  /** Primary directional light intensity (linear, unitless). Same MUST-match
   *  invariant as primaryLightDir. */
  primaryLightIntensity: number;
  /** Diffuse-sky-dome RGB tint, derived from skyParams turbidity by
   *  computeLightingState. Consumed by walkaround's sky-aperture probe
   *  and second-bounce-sky-miss paths. Replaces four formerly-hardcoded
   *  tints scattered across WGSL. */
  skyTint: [number, number, number];
  /** Sky-dome irradiance scalar paired with skyTint. ~0.5×sun at noon. */
  skyIrradiance: number;
  /** DDGI compute enable toggle (typically wired from window.__HYBRID_LAYERS__.ddgi). */
  ddgiOn: boolean;
  /** Predicate the hook polls before kicking off ReSTIR pipeline init.
   *  Returns true when the scene has enough geometry mounted to build a
   *  meaningful BVH. Defaults to "scene has at least 200 triangles" if
   *  not supplied — library consumers should supply their own scene-
   *  appropriate predicate. */
  isSceneReady?: () => boolean;
  /** Stable signal that changes when the ReSTIR pipeline must reinitialize
   *  (e.g., scene topology changed: room loaded/unloaded, fixture swapped).
   *  Pass a string/number/null. Caller should use a stable identifier like
   *  roomKey rather than a fresh-object useMemo. */
  pipelineRebuildKey: string | number | null;
}

export interface HybridLayeredGIState {
  /** The DDGI handle returned by useDDGI. Caller can debug-bridge from
   *  this (handle.ready, handle.probeCount, handle.lastFrameMs). */
  ddgi: DDGIHandle;
  /** True once the ReSTIR pipeline has finished initialization. */
  ready: boolean;
  /** Pipeline + BVH for downstream debug bridges (e.g. WalkaroundDebugBridge).
   *  Null until ready=true. */
  passes: { pipeline: WalkaroundGPUPipeline; bvh: SceneBVHBuffers } | null;
}

/** Default scene-readiness predicate for non-stained-glass library consumers
 *  that don't supply their own. Counts triangles via scene.traverse. */
function defaultIsSceneReady(scene: THREE.Scene, _enabled: boolean): boolean {
  let total = 0;
  scene.traverse((obj) => {
    const mesh = obj as THREE.Mesh;
    if (!mesh.isMesh || !mesh.geometry) return;
    const idx = mesh.geometry.index;
    total += idx ? idx.count / 3 : (mesh.geometry.attributes.position?.count ?? 0) / 3;
    if (total >= 200) return;
  });
  return total >= 200;
}

export function useHybridLayeredGI(opts: HybridLayeredGIOpts): HybridLayeredGIState {
  const { enabled, device, primaryLightDir, primaryLightIntensity, skyTint, skyIrradiance, ddgiOn, isSceneReady, pipelineRebuildKey } = opts;
  const { scene, camera, gl, size } = useThree();

  // ── DDGI compute ────────────────────────────────────────────────────
  const ddgi = useDDGI({ enabled: enabled && ddgiOn });

  // Propagate primaryLightIntensity to DDGI's probe-update light UBO so
  // its bake of the sun's Le matches shade.wgsl's Lo_emit. Without
  // this, DDGI stores Le at Redux's intensity=1.0 (5× dimmer than
  // shade) → walls render dark even with all GI terms enabled.
  // Updates whenever the value changes (intensity is per-render but
  // ProbeUpdatePass's _sunIntensityMul is static-set, so re-applying
  // is cheap).
  useEffect(() => {
    if ('setSunIntensityMultiplier' in ddgi.pass) {
      (ddgi.pass as unknown as { setSunIntensityMultiplier: (m: number) => void })
        .setSunIntensityMultiplier(primaryLightIntensity);
    }
  }, [ddgi.pass, primaryLightIntensity]);

  // RC cascade compute removed step 4 of the render-mode-hierarchy
  // restructure (2026-05-08): hybrid's shade pass discarded Lo_rc, so
  // dispatching the cascade compute every frame was pure GPU waste.
  // The standalone 'rc' walkaround engine still uses the cascade
  // subsystem; the shared modules (useSceneBVH, useCascadeBuffers,
  // dispatchCascadePasses, cascadePyramid) remain live for it.

  // ── ReSTIR pipeline lifecycle ───────────────────────────────────────
  const pipelineRef    = useRef<WalkaroundGPUPipeline | null>(null);
  const bvhBuffersRef  = useRef<SceneBVHBuffers | null>(null);
  const lastFrameTsRef = useRef<number>(0);

  // ── DEV-only diagnostic instrumentation ──────────────────────────────
  // All `[hybrid:debug]`-prefixed logs are stripped from production by Vite
  // (gated on import.meta.env.DEV). Tracks pipeline init/dispose events,
  // per-frame skip reasons, and rebuild churn so a live session can be
  // grep-debugged without re-instrumenting code.
  const debugRef = useRef({
    initStart: 0,
    initCount: 0,
    disposeCount: 0,
    skipNoEnabled: 0,
    skipNoPipeline: 0,
    skipNoBvh: 0,
    skipNoDevice: 0,
    skipNoSwapView: 0,
    skipFrameInterval: 0,
    framesDispatched: 0,
    lastReportTs: 0,
    prevPipelineRebuildKey: null as string | number | null,
  });
  const [ready, setReady] = useState(false);
  const [passes, setPasses] = useState<HybridLayeredGIState['passes']>(null);

  useEffect(() => {
    if (!enabled) return;
    if (!device) return;

    let cancelled = false;
    const sceneReadyFn = isSceneReady ?? (() => defaultIsSceneReady(scene, enabled));

    if (import.meta.env.DEV) {
      const dbg = debugRef.current;
      dbg.initCount++;
      dbg.initStart = performance.now();
      const prev = dbg.prevPipelineRebuildKey;
      dbg.prevPipelineRebuildKey = pipelineRebuildKey;
      console.log(
        `[hybrid:debug] init #${dbg.initCount} START`,
        { pipelineRebuildKey, prev, deltaPrev: prev !== pipelineRebuildKey,
          sceneRef: !!scene, sizeW: size.width, sizeH: size.height,
          deviceRef: !!device, enabled, t: dbg.initStart.toFixed(0) },
      );
    }

    const buildBVHWhenReady = async () => {
      const pollStart = Date.now();
      let pollIters = 0;
      while (!cancelled) {
        const elapsed = Date.now() - pollStart;
        if (elapsed >= 5_000) break;
        if (sceneReadyFn()) break;
        await new Promise<void>((r) => setTimeout(r, 50));
        pollIters++;
      }
      if (cancelled) {
        if (import.meta.env.DEV) {
          console.log('[hybrid:debug] init aborted during scene-readiness poll',
            { pollIters, elapsed: Date.now() - pollStart });
        }
        return;
      }
      if (import.meta.env.DEV) {
        console.log('[hybrid:debug] scene-ready', { pollIters, elapsed: Date.now() - pollStart });
      }

      try {
        const bvhStart = performance.now();
        const bvh = buildSceneBVH([scene], {
          primaryLightDir,
          primaryLightIntensity,
        });
        const bvhMs = performance.now() - bvhStart;
        bvhBuffersRef.current = bvh;

        const canvas = (gl as unknown as { domElement: HTMLCanvasElement }).domElement;
        const W = canvas.width  || size.width;
        const H = canvas.height || size.height;

        if (import.meta.env.DEV) {
          console.log('[hybrid:debug] BVH built', {
            bvhMs: bvhMs.toFixed(1),
            triCount: bvh.bvhNodes?.count, emitterCount: bvh.emitters?.count,
            canvasW: W, canvasH: H,
          });
        }

        const pipeline = new WalkaroundGPUPipeline(device, W, H);
        const pipelineStart = performance.now();
        await pipeline.initialize(bvh, getSwapChainFormat());
        const pipelineMs = performance.now() - pipelineStart;

        if (cancelled) {
          if (import.meta.env.DEV) {
            console.log('[hybrid:debug] init aborted post-pipeline.initialize (leak risk)',
              { pipelineMs: pipelineMs.toFixed(1) });
          }
          pipeline.dispose();
          return;
        }

        pipelineRef.current = pipeline;
        setReady(true);
        setPasses({ pipeline, bvh });

        const w = window as Window & typeof globalThis;
        if (w.__WGPU__) {
          w.__WGPU__.walkaround = {
            passes: { pipeline, bvh },
            frameTimings: [],
          } as unknown as typeof w.__WGPU__.walkaround;
        }

        if (import.meta.env.DEV) {
          const dbg = debugRef.current;
          const totalMs = performance.now() - dbg.initStart;
          console.log(`[hybrid:debug] init #${dbg.initCount} COMPLETE`, {
            pipelineMs: pipelineMs.toFixed(1),
            totalMs: totalMs.toFixed(1),
          });
        }
      } catch (err) {
        console.error('[useHybridLayeredGI] init failed:', err);
      }
    };

    void buildBVHWhenReady();

    return () => {
      cancelled = true;
      setReady(false);
      setPasses(null);
      if (pipelineRef.current) {
        pipelineRef.current.dispose();
        pipelineRef.current = null;
      }
      if (bvhBuffersRef.current) {
        disposeSceneBVH(bvhBuffersRef.current);
        bvhBuffersRef.current = null;
      }
      const w = window as Window & typeof globalThis;
      if (w.__WGPU__?.walkaround) {
        w.__WGPU__.walkaround = undefined;
      }
      if (import.meta.env.DEV) {
        const dbg = debugRef.current;
        dbg.disposeCount++;
        const liveMs = dbg.initStart > 0 ? performance.now() - dbg.initStart : 0;
        console.log(`[hybrid:debug] dispose #${dbg.disposeCount}`, {
          ranForMs: liveMs.toFixed(1),
          framesDispatched: dbg.framesDispatched,
          skipReasons: {
            noEnabled: dbg.skipNoEnabled, noPipeline: dbg.skipNoPipeline,
            noBvh: dbg.skipNoBvh, noDevice: dbg.skipNoDevice,
            noSwapView: dbg.skipNoSwapView, frameInterval: dbg.skipFrameInterval,
          },
        });
        // Reset per-pipeline counters; init/dispose counts are cumulative.
        dbg.framesDispatched = 0;
        dbg.skipNoEnabled = 0;
        dbg.skipNoPipeline = 0;
        dbg.skipNoBvh = 0;
        dbg.skipNoDevice = 0;
        dbg.skipNoSwapView = 0;
        dbg.skipFrameInterval = 0;
      }
    };
  // pipelineRebuildKey is the explicit caller-controlled signal for
  // scene-topology changes that warrant pipeline reinit (e.g. roomKey
  // change when entering/exiting room mode). Including a fresh-object
  // dep like graphFaces would make the ReSTIR pipeline reinitialize on
  // every edit — which never reaches steady-state and the canvas
  // stays black.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [device, scene, size.width, size.height, enabled, pipelineRebuildKey]);

  // ── Per-frame ReSTIR dispatch (priority=1) ──────────────────────────
  useFrame(() => {
    const dbg = import.meta.env.DEV ? debugRef.current : null;
    if (!enabled) {
      if (dbg) dbg.skipNoEnabled++;
      return;
    }
    const pipeline = pipelineRef.current;
    const bvh = bvhBuffersRef.current;
    if (!pipeline) { if (dbg) dbg.skipNoPipeline++; return; }
    if (!bvh)      { if (dbg) dbg.skipNoBvh++;      return; }
    if (!device)   { if (dbg) dbg.skipNoDevice++;   return; }

    const now = performance.now();
    if (lastFrameTsRef.current !== 0 &&
        now - lastFrameTsRef.current < TARGET_FRAME_INTERVAL_MS) {
      if (dbg) dbg.skipFrameInterval++;
      return;
    }
    lastFrameTsRef.current = now;

    const t0 = now;
    const canvasEl = (gl as unknown as { domElement: HTMLCanvasElement }).domElement;
    const W = canvasEl.width  || size.width;
    const H = canvasEl.height || size.height;

    const swapView = getSwapChainView(gl);
    if (!swapView) {
      if (dbg) dbg.skipNoSwapView++;
      return;
    }
    const swapFmt = getSwapChainFormat();

    // Periodic per-5s rate report — fires once every 5s of wall clock,
    // logs frames-dispatched + skip-reason histogram so a quiet pipeline
    // is immediately distinguishable from a healthy 30fps one.
    if (dbg) {
      dbg.framesDispatched++;
      if (dbg.lastReportTs === 0) dbg.lastReportTs = now;
      if (now - dbg.lastReportTs > 5_000) {
        const elapsedSec = (now - dbg.lastReportTs) / 1_000;
        // Per-pass GPU timings (ms) from the pipeline's timestamp queries.
        // Empty when 'timestamp-query' isn't available; populated once
        // readback completes (~2 frames after submit).
        const gpu = (pipeline as unknown as { lastGpuTimings?: Record<string, number> })
          .lastGpuTimings ?? {};
        const gpuTotal = gpu['total'];
        console.log('[hybrid:debug] rate (5s window)', {
          framesDispatched: dbg.framesDispatched,
          fps: (dbg.framesDispatched / elapsedSec).toFixed(2),
          skipReasons: {
            noEnabled: dbg.skipNoEnabled, noPipeline: dbg.skipNoPipeline,
            noBvh: dbg.skipNoBvh, noDevice: dbg.skipNoDevice,
            noSwapView: dbg.skipNoSwapView, frameInterval: dbg.skipFrameInterval,
          },
          gpuTotalMs: gpuTotal !== undefined ? +gpuTotal.toFixed(2) : 'n/a',
          gpuPerPassMs: gpu,
        });
        dbg.framesDispatched = 0;
        dbg.skipNoEnabled = 0;
        dbg.skipNoPipeline = 0;
        dbg.skipNoBvh = 0;
        dbg.skipNoDevice = 0;
        dbg.skipNoSwapView = 0;
        dbg.skipFrameInterval = 0;
        dbg.lastReportTs = now;
      }
    }

    // ── DDGI atlas wire ───────────────────────────────────────────────
    if (!ddgiOn) {
      pipeline.setDDGIInputs(null);
    } else if (ddgi.ready) {
      const atlas = ddgi.pass.getReadAtlasGPUTextures?.();
      if (atlas) {
        const p = ddgi.probeGrid.params;
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

    const view  = camera.matrixWorldInverse.elements;
    const proj  = (camera as THREE.PerspectiveCamera).projectionMatrix.elements;

    pipeline.renderFrame({
      viewMatrix: new Float32Array(view),
      projMatrix: new Float32Array(proj),
      prevViewMatrix: new Float32Array(view),
      prevProjMatrix: new Float32Array(proj),
      cameraPos: [camera.position.x, camera.position.y, camera.position.z],
      screenWidth: W,
      screenHeight: H,
      frameSeed: Math.floor(now * 1000) & 0xFFFFFFFF,
      totalEmissivePower: bvh.totalEmissivePower ?? 1.0,
      emitterCount: bvh.emitters?.count ?? 0,
      primaryLightDir: [primaryLightDir.x, primaryLightDir.y, primaryLightDir.z],
      primaryLightIntensity,
      skyTint,
      skyIrradiance,
      swapChainView: swapView,
      swapChainFormat: swapFmt,
    });

    const dt = performance.now() - t0;
    const w = window as Window & typeof globalThis;
    if (w.__WGPU__?.walkaround) {
      const ft = w.__WGPU__.walkaround.frameTimings as Array<{ t: number; ms: number }>;
      if (Array.isArray(ft)) {
        ft.push({ t: now, ms: dt });
        if (ft.length > 240) ft.shift();
      }
    }
  }, 1);

  return { ddgi, ready, passes };
}

// ──────────────────────────────────────────────────────────────────────
// Local helpers (kept here rather than re-exported from HybridLayeredStage
// so the hook is self-contained for library extraction).
// ──────────────────────────────────────────────────────────────────────

function getSwapChainView(gl: unknown): GPUTextureView | null {
  try {
    const ctx = (gl as { backend?: { getContext?: () => GPUCanvasContext | null } })
      ?.backend?.getContext?.();
    if (!ctx) return null;
    return ctx.getCurrentTexture().createView();
  } catch {
    return null;
  }
}

function getSwapChainFormat(): GPUTextureFormat {
  return (typeof navigator !== 'undefined' && 'gpu' in navigator
    ? (navigator.gpu as { getPreferredCanvasFormat?: () => GPUTextureFormat })
        .getPreferredCanvasFormat?.() ?? 'bgra8unorm'
    : 'bgra8unorm') as GPUTextureFormat;
}
