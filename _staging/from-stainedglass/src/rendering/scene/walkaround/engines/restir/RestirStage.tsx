/**
 * WalkaroundStage — sibling of RasterStage / PTStage. Mounts only when
 * `exploreEnabled && walkaroundEngine === 'restir'` AND the renderer is
 * WebGPURenderer.  See StudioScene's walkaroundActive gate.
 *
 * Renders entirely via ReSTIR DI/GI compute + à-trous denoiser on the GPU,
 * and composites the result directly to the WebGPU swap-chain texture.
 * Three.js raster rendering is bypassed (we pass priority=1 to useFrame so
 * r3f does NOT auto-call gl.render() after our frame callback).
 *
 * §8.3 + §10.7 of the walkaround plan (primary-ray-cast mode).
 */
import { useEffect, useMemo, useRef, useState } from 'react';
import { useThree, useFrame } from '@react-three/fiber';
import { OrbitControls, Sky, Environment, Html } from '@react-three/drei';
import { useSelector } from 'react-redux';
import * as THREE from 'three';
import { MountDispatch } from '../../../mounts/MountDispatch';
import { lightboxDimsFor } from '../../../mounts/lightboxDims';
import { LightSourceList } from '../../../lighting/renderLightSource';
import { RoomLoader } from '../../../../assets/RoomLoader';
import { FaceRenderer } from '../../../../glass/FaceRenderer';
import { GLASS_FACE_MESH_NAME } from '../../../../glass/GlassMesh';
import { EdgeLines } from '../../../../edges/EdgeLines';
import { buildSceneBVH, disposeSceneBVH } from './bvhCompute';
import { FLOOR_Y } from '../../../../assets/livingRoomShellGeometry';
import { computeLightingState } from '../../../lightingState';
import type { SceneBVHBuffers } from './bvhCompute';
import { WalkaroundGPUPipeline } from './WalkaroundGPUPipeline';
import { WalkaroundDebugBridge } from './WalkaroundDebugBridge';
import type { WgWalkaroundBridge } from './walkaroundBridgeTypes';
import type { BackdropMode } from '@/store/uiSlice';
import type { SkyParams } from '../../../skyParams';
import type { RootState } from '@/store';

interface RestirStageProps {
  backdropMode: BackdropMode;
  skyParams: SkyParams;
  nightSkyParams: SkyParams;
  frameLayout: { cx: number; cy: number; w: number; h: number };
  orbitTarget: [number, number, number];
}

/**
 * Get the WebGPU swap-chain (canvas) texture view for the current frame.
 * The three.js WebGPUBackend stores a GPUCanvasContext as `backend.context`,
 * accessible via `backend.getContext()`. `getCurrentTexture()` returns the
 * same GPUTexture for the entire animation frame.
 */
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

/**
 * Returns the preferred swap-chain format for this device/platform.
 * Matches what three.js WebGPUBackend configures the canvas with.
 */
function getSwapChainFormat(): GPUTextureFormat {
  return navigator.gpu?.getPreferredCanvasFormat?.() ?? 'bgra8unorm';
}

export function RestirStage({
  backdropMode,
  skyParams,
  nightSkyParams,
  frameLayout,
  orbitTarget,
}: RestirStageProps) {
  const { gl, scene, camera, size } = useThree();
  const mount = useSelector((s: RootState) => s.scene.mount);
  const space = useSelector((s: RootState) => s.scene.space);
  const roomOrbitRadius = useSelector((s: RootState) => s.scene.roomOrbitRadius);
  // Graph faces map drives panel geometry — when faces arrive after an
  // initial BVH build (race-recovery: scene-readiness gate fired on a
  // partial scene), we need to retrigger the build effect so the panel
  // is actually in the BVH. Subscribing to s.graph.faces gives a stable
  // identity for "set of faces" — per-face property edits (color, opacity,
  // etc.) live in state.properties and don't change this reference, so
  // we don't pay a BVH rebuild for paint-bucket clicks. graphSlice only
  // mutates via setGraph / loadProject (full replace), so faces +
  // vertices + halfEdges all change in lockstep — tracking faces alone
  // covers all geometry-meaningful changes.
  const graphFaces = useSelector((s: RootState) => s.graph.faces);
  // Slice 3.3 — moved from state.ui to state.viewport.
  const timeOfDay = useSelector((s: RootState) => s.viewport.timeOfDay);

  const lightboxDims = lightboxDimsFor(mount, frameLayout.w, frameLayout.h);
  const roomKey = space.kind === 'room' ? space.roomKey : null;
  const suppressSun = backdropMode === 'studio' || backdropMode === 'sunset';
  const skyBackdrop = backdropMode === 'sky' || backdropMode === 'night';
  const isNight = backdropMode === 'night';

  // Sun + sky lighting derived from current timeOfDay. computeLightingState
  // is the single source of truth shared with PT and HybridLayeredStage so
  // RestirStage can't drift from the rest of the renderer (sweep finding
  // bugs 1+2: previously RestirStage hardcoded sun=π and omitted skyTint
  // from the WGSL UBO inputs entirely → TS2345 + sky pixels reading
  // garbage at runtime).
  const lightingState = useMemo(
    () => computeLightingState({
      timeOfDay,
      skyParams: isNight ? nightSkyParams : skyParams,
      isNight,
    }),
    [timeOfDay, skyParams, nightSkyParams, isNight],
  );
  const showBackboard = !skyBackdrop;

  // Pipeline state.
  const pipelineRef = useRef<WalkaroundGPUPipeline | null>(null);
  const bvhBuffersRef = useRef<SceneBVHBuffers | null>(null);
  const prevViewMatRef = useRef(new THREE.Matrix4());
  const frameCountRef = useRef(0);
  /**
   * Live sun parameters captured at BVH-build time and reused by every
   * renderFrame call.  Both the (normalized) sunDirection and sunIntensity
   * MUST match the values passed to buildSceneBVH so the shade shader's
   * self-emission Le for primary glass hits reproduces exactly the Le
   * baked into the emitter list.  Pre-fix the pipeline received a
   * hardcoded `sunDirection=[0.5,1.0,0.5]` (un-normalized, never matched
   * the actual sun) and no sunIntensity at all — so the shader had no
   * way to recover Le for self-emissive glass primary hits, leaving the
   * panel cells rendering as pure black against the lit room.
   */
  const sunParamsRef = useRef<{ direction: [number, number, number]; intensity: number }>({
    direction: [0, 1, 0],
    intensity: Math.PI,
  });
  /**
   * Last dispatched frame's `performance.now()` timestamp.  Drives the
   * 60 FPS frame cap below.  ReSTIR + atrous + composite at 3840×1902
   * is GPU-bound at well above 60 FPS on a discrete card; uncapped, the
   * whole pipeline runs as fast as the GPU can dispatch (often 200+
   * FPS), wasting power and battery for zero perceptible visual gain.
   * Throttling to ~16.67 ms/frame matches a 60 Hz display's refresh
   * rate, which is the user-perceivable ceiling on a typical monitor.
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
  const [passes, setPasses] = useState<unknown>(null);

  // ── Hardware-GPU gate (Option F) ──────────────────────────────────────
  // The R3F gl factory in StudioScene publishes `window.__WG__` AFTER
  // `renderer.init()` resolves (which is what makes useThree() return,
  // which is what mounts this component).  By the time this effect fires,
  // __WG__ is populated.  We poll defensively for a tick in case the
  // factory's bridge write somehow races behind WalkaroundStage mount —
  // the e2e gate logic also accepts __WGPU__ presence as "settled" so the
  // test continues to work even if __WG__ briefly lags.
  //
  // Pre-Tier-2-migration this gate lived inside WebGPUCanvas.tsx as a
  // pre-mount probe; the migration consolidates the canvas mount into
  // StudioScene's R3F factory and moves the refuse-to-mount overlay here
  // (per-branch, since the SwiftShader copy differs across DDGI / RC /
  // ReSTIR).
  const [wgGate, setWgGate] = useState<WgWalkaroundBridge | null>(null);
  useEffect(() => {
    let cancelled = false;
    const check = () => {
      if (cancelled) return;
      const g = window.__WG__;
      if (g) {
        setWgGate(g);
        return;
      }
      // Factory hasn't published yet — try again on the next microtask.
      Promise.resolve().then(check);
    };
    check();
    return () => { cancelled = true; };
  }, []);

  const refuseToMount = wgGate !== null && wgGate.isWebGPU && wgGate.isHardwareGpu === false;

  // WebGPU device detection.
  const wgpuDevice: GPUDevice | null = (() => {
    const w = window as Window & typeof globalThis;
    return w.__WGPU__?.device ?? null;
  })();

  // Panel-framing default camera.  StudioScene's StudioCameraRig is
  // skipped under walkaroundActive (the rig would clobber the calibrated
  // walkaround pose), so the camera arrives at R3F's default origin pose.
  // The 13-walkaround-restir e2e spec sets `window.__CHROMA_TEST_CAMERA__`
  // BEFORE the explore toggles fire, then reads the canvas at the
  // calibrated chroma-test pose ([0,0,80] camera, target [0,FLOOR_Y,-40]).
  // For the dev / .sglass-loaded path no test bridge fires, so the camera
  // would otherwise stay at the origin and the panel would sit outside
  // the frustum.  This effect frames the panel head-on or stages a
  // viewer-in-the-room pose on first mount only, leaving subsequent test
  // overrides authoritative.  Setting `window.__CHROMA_TEST_CAMERA__ =
  // true` opts out of the framing setup entirely (the chroma test owns
  // the pose via OrbitControls' target — see below).
  const framedRef = useRef(false);
  useEffect(() => {
    if (framedRef.current) return;
    if (refuseToMount) return;
    const w = window as Window & { __CHROMA_TEST_CAMERA__?: boolean };
    if (w.__CHROMA_TEST_CAMERA__ === true) {
      // Chroma-test pose calibrated by commit a37355c — preserve.
      camera.position.set(0, 0, 80);
      camera.lookAt(0, FLOOR_Y, -40);
      camera.updateMatrixWorld();
      framedRef.current = true;
      return;
    }
    // Default pose — same as the pre-migration StudioScene walkaround
    // branch: viewer standing in the living room interior, looking at
    // the suncatcher.  Calibrated to make the through-glass sun caustic
    // on the floor visible AND keep the panel + room interior in frame.
    const isRoomSpace = space.kind === 'room';
    if (isRoomSpace) {
      camera.position.set(frameLayout.cx, 0, 160);
      camera.lookAt(frameLayout.cx, -30, 0);
    } else {
      camera.position.set(frameLayout.cx, frameLayout.cy, 60);
      camera.lookAt(frameLayout.cx, frameLayout.cy, 0);
    }
    camera.updateMatrixWorld();
    framedRef.current = true;
  }, [camera, frameLayout.cx, frameLayout.cy, refuseToMount, space.kind]);

  // Build BVH + initialize pipeline on mount / scene change.
  useEffect(() => {
    if (refuseToMount) return;
    if (!wgpuDevice) return;

    let cancelled = false;

    /**
     * Poll for scene mesh readiness before building the BVH.
     * Instead of a fixed-duration timeout (which may expire before the honeycomb
     * geometry is loaded), we poll `scene.children` for at least one Mesh with
     * a non-degenerate geometry index (>0 triangles). This guarantees the BVH
     * includes the honeycomb panels, room shell, etc.
     *
     * We cap at 10 seconds; if no geometry appears the scene is probably void-space
     * and we build with whatever is present.
     */
    /**
     * Returns true when the scene has enough geometry to build a useful BVH.
     * Conditions:
     *  - Room mode (roomKey set): require BOTH the named room floor mesh AND
     *    glass-face geometry (transmissive panels). The earlier "room floor
     *    sufficient" check raced FaceRenderer commit and produced a BVH with
     *    only the room shell, dropping the emitter list to the dummy 1-tri
     *    fallback. Post-roommesh-merge, the room shell tessellation arrives
     *    very quickly (~32×32 = 2048 tris), making the race much more likely.
     *  - Void mode: glass faces alone suffice (triCount ≥ 200).
     *
     * Glass-face detection uses the explicit GLASS_FACE_MESH_NAME tag
     * GlassMesh writes on every body/flash mesh. The earlier "untagged
     * mesh" heuristic counted ANY anonymous mesh — including the four
     * unnamed ExtrudeGeometry bars FrameSimulation renders for the wood
     * frame, which together carry hundreds of tris and could satisfy the
     * 60-tri threshold before any panel face committed. The fix forces
     * the gate to wait for actual GlassMesh meshes.
     */
    const isSceneReady = (): boolean => {
      let hasFaceGeo = false;
      let hasRoomFloor = false;
      let glassTriCount = 0;
      let totalTriCount = 0;
      scene.traverse((obj) => {
        const mesh = obj as THREE.Mesh;
        if (!mesh.isMesh || !mesh.geometry) return;
        const idx = mesh.geometry.index;
        const tris = idx ? idx.count / 3 : (mesh.geometry.attributes.position?.count ?? 0) / 3;
        totalTriCount += tris;
        if (obj.name === 'surface_floor_living') hasRoomFloor = true;
        // Glass panel faces — explicit name set by GlassMesh.
        if (obj.name === GLASS_FACE_MESH_NAME && mesh.geometry.attributes.position) {
          hasFaceGeo = true;
          glassTriCount += tris;
        }
      });
      if (roomKey) {
        // Room mode: need BOTH room floor AND glass geometry.
        return hasRoomFloor && hasFaceGeo && glassTriCount >= 60;
      }
      // Void mode: glass faces alone — total-tri threshold reflects the
      // scene-graph as a whole (panel + ancillary fixtures) which is fine
      // when there's no room-shell wood-frame race to worry about.
      return hasFaceGeo && totalTriCount >= 200;
    };

    const buildBVHWhenReady = async () => {
      // Poll every 50ms for up to 5s for scene geometry to be ready.
      // Instead of counting triangles (which can oscillate during baking),
      // we look for the named room floor mesh — its presence guarantees
      // LivingRoomShell has fully committed all 5 surface planes.
      //
      // If the scene has no room (void space), fall back to glass-face detection.
      const pollStart = Date.now();
      while (!cancelled) {
        const elapsed = Date.now() - pollStart;
        if (elapsed >= 5_000) {
          break;
        }
        if (isSceneReady()) {
          break;
        }
        await new Promise<void>((r) => setTimeout(r, 50));
      }
      if (cancelled) return;

      try {
        // Sweep finding Bug 2: BVH builds with the LIVE sunDirection +
        // sunIntensity from computeLightingState (single source of truth)
        // so emitter Le tracks timeOfDay. Previously hardcoded to π here
        // → emitters stayed at noon-radiance regardless of time-of-day,
        // 3.3× too bright at sunset. The skyParams-derived direction
        // also replaces the scene-graph DirectionalLight lookup, which
        // was off when SunRenderer dispatched suppressSun=true.
        const sunDirection = lightingState.sunDirection;

        const bvh = buildSceneBVH([scene], {
          primaryLightDir: sunDirection,
          primaryLightIntensity: lightingState.sunIntensity,
        });
        bvhBuffersRef.current = bvh;
        // Capture sun params for renderFrame.  The shader's self-emission
        // Le must reproduce the Le baked into the emitter list — that
        // requires the SAME sunDirection (normalized world-to-sun vector)
        // and the SAME sunIntensity used during BVH build.
        sunParamsRef.current = {
          direction: [sunDirection.x, sunDirection.y, sunDirection.z] as [number, number, number],
          intensity: lightingState.sunIntensity,
        };

        // Use the canvas's physical (DPR-scaled) pixel dimensions so the pipeline
        // renders to the full swap-chain texture rather than the CSS-pixel-sized
        // sub-region. size.width/height are in CSS logical pixels; the actual WebGPU
        // texture is gl.domElement.width × gl.domElement.height (DPR-scaled).
        const canvas = (gl as unknown as { domElement: HTMLCanvasElement }).domElement;
        const W = canvas.width  || size.width;
        const H = canvas.height || size.height;

        // Determine swap-chain format before initializing pipeline.
        const swapChainFmt = getSwapChainFormat();

        // Create + initialize pipeline.
        const pipeline = new WalkaroundGPUPipeline(wgpuDevice, W, H);
        await pipeline.initialize(bvh, swapChainFmt);
        pipelineRef.current = pipeline;

        // Publish to the debug bridge ONLY after pipeline is fully online.
        // This ensures window.__WGPU__.walkaround is set only when ReSTIR
        // is actually ready to render (never from init heartbeats).
        setPasses({ pipeline, bvh });
        const w = window as Window & typeof globalThis;
        if (w.__WGPU__) {
          const device = wgpuDevice;

          /**
           * renderOneFrame — manually drives one full ReSTIR frame without RAF.
           *
           * RAF is throttled to 0 Hz in hidden tabs (MCP-controlled Chrome is
           * "hidden"). This bypass calls the pipeline directly, awaits GPU
           * completion via queue.onSubmittedWorkDone(), and returns the wall-clock
           * milliseconds for that frame. Used by the validation harness.
           */
          const renderOneFrame = async (): Promise<number> => {
            const p = pipelineRef.current;
            const b = bvhBuffersRef.current;
            if (!p || !b) throw new Error('Pipeline not initialized');

            // Get the swap-chain texture view. We call getCurrentTexture()
            // directly on the WebGPU canvas context, bypassing three.js RAF.
            // This works outside RAF in Chrome's WebGPU implementation.
            const ctx = (gl as { backend?: { getContext?: () => GPUCanvasContext | null } })
              ?.backend?.getContext?.();
            if (!ctx) throw new Error('WebGPU canvas context not available');

            const swapChainView = ctx.getCurrentTexture().createView();
            const swapChainFormat = getSwapChainFormat();

            // Build frame inputs from the current three.js camera.
            const cam3 = camera as THREE.PerspectiveCamera;
            cam3.updateMatrixWorld(true);
            cam3.updateProjectionMatrix();

            const viewMatrix = new Float32Array(cam3.matrixWorldInverse.elements);
            const projMatrix = new Float32Array(cam3.projectionMatrix.elements);
            const prevViewMatrix = new Float32Array(prevViewMatRef.current.elements);

            // Use DPR-scaled canvas dimensions (matches pipeline initialization).
            const glCanvas2 = (gl as unknown as { domElement: HTMLCanvasElement }).domElement;
            const rof_W = glCanvas2.width  || size.width;
            const rof_H = glCanvas2.height || size.height;
            const inputs = {
              viewMatrix,
              projMatrix,
              prevViewMatrix,
              prevProjMatrix: projMatrix,
              cameraPos: cam3.getWorldPosition(new THREE.Vector3()).toArray() as [number, number, number],
              screenWidth: rof_W,
              screenHeight: rof_H,
              frameSeed: (frameCountRef.current * 1664525 + 1013904223) & 0xFFFFFFFF,
              totalEmissivePower: b.totalEmissivePower,
              emitterCount: b.emitterCount,
              // Use the live primary-light params captured at BVH-build
              // time so the shader's self-emission Le for primary glass
              // hits reproduces exactly the Le baked into the emitter list.
              primaryLightDir: sunParamsRef.current.direction,
              primaryLightIntensity: sunParamsRef.current.intensity,
              skyTint: lightingState.skyTint,
              skyIrradiance: lightingState.skyIrradiance,
              swapChainView,
              swapChainFormat,
            };

            const t0 = performance.now();
            p.renderFrame(inputs);
            // Await GPU completion so our timing includes actual GPU work.
            await device.queue.onSubmittedWorkDone();
            const ms = performance.now() - t0;

            prevViewMatRef.current.copy(cam3.matrixWorldInverse);
            frameCountRef.current++;

            // Record timing on bridge.
            const wb = window as Window & typeof globalThis;
            if (wb.__WGPU__?.walkaround) {
              const timings = wb.__WGPU__.walkaround.frameTimings;
              timings.push({ t: t0, ms });
              if (timings.length > 240) timings.shift();
            }

            return ms;
          };

          /**
           * captureFrame — render one frame and return pixel data from the
           * final denoised texture (denoisedPingTexture) as a Float32Array.
           * Pixel layout: [R, G, B, A, R, G, B, A, ...] in row-major order.
           * Used by the caustic validation harness to read GPU pixels directly,
           * bypassing the WebGPU canvas toDataURL limitation.
           */
          const captureFrame = async (): Promise<{ pixels: Float32Array; width: number; height: number }> => {
            const ms = await renderOneFrame();
            const p = pipelineRef.current;
            if (!p) throw new Error('Pipeline not initialized');

            const glCanvas = (gl as unknown as { domElement: HTMLCanvasElement }).domElement;
            const W = glCanvas.width || size.width;
            const H = glCanvas.height || size.height;

            // After atrous pass 4 (0-indexed), result is in denoisedPingTexture.
            // We read via GPU buffer copy (COPY_SRC usage added to the texture).
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            const pip = (p as any);
            const finalTex: GPUTexture = pip.denoisedPingTexture;
            const bytesPerPixel = 8; // rgba16float = 4 × f16 = 8 bytes
            const bytesPerRow = Math.ceil((W * bytesPerPixel) / 256) * 256;
            const staging = wgpuDevice!.createBuffer({
              size: bytesPerRow * H,
              usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
            });

            const enc = wgpuDevice!.createCommandEncoder({ label: 'captureFrame-readback' });
            enc.copyTextureToBuffer(
              { texture: finalTex, mipLevel: 0, origin: [0, 0, 0] },
              { buffer: staging, bytesPerRow, rowsPerImage: H },
              { width: W, height: H, depthOrArrayLayers: 1 },
            );
            wgpuDevice!.queue.submit([enc.finish()]);
            await wgpuDevice!.queue.onSubmittedWorkDone();
            await staging.mapAsync(GPUMapMode.READ);

            // Convert Float16 → Float32 row-by-row (skip padding bytes per row).
            // Float16Array is an ES2024 proposal; TS lib=ES2023 doesn't know
            // about it yet, but Chrome 126+ ships it. Cast through a typed
            // proxy to keep tsc quiet.
            const f16PerRow = bytesPerRow / 2;   // number of f16 scalars per padded row
            const F16 = (globalThis as { Float16Array?: { new (b: ArrayBuffer): { [i: number]: number; length: number } } }).Float16Array;
            if (!F16) {
              throw new Error('Float16Array not available — requires Chrome 126+ for ReSTIR readback');
            }
            const raw = new F16(staging.getMappedRange());
            const out = new Float32Array(W * H * 4);
            for (let y = 0; y < H; y++) {
              const srcBase = y * f16PerRow;
              const dstBase = y * W * 4;
              for (let x = 0; x < W; x++) {
                out[dstBase + x * 4 + 0] = raw[srcBase + x * 4 + 0];
                out[dstBase + x * 4 + 1] = raw[srcBase + x * 4 + 1];
                out[dstBase + x * 4 + 2] = raw[srcBase + x * 4 + 2];
                out[dstBase + x * 4 + 3] = raw[srcBase + x * 4 + 3];
              }
            }
            staging.unmap();
            staging.destroy();
            // Return timing alongside so callers can chain timing checks.
            void ms;
            return { pixels: out, width: W, height: H };
          };

          // walkaround sub-object is set here, post-init — never pre-init.
          w.__WGPU__.walkaround = {
            passes: { pipeline, bvh },
            camera,
            frameTimings: [],
            renderOneFrame,
            captureFrame,
          } as unknown as typeof w.__WGPU__.walkaround;
        }
      } catch (err) {
        console.error('[WalkaroundStage] Pipeline initialization failed:', err);
      }
    };  // end buildBVHWhenReady

    buildBVHWhenReady();

    return () => {
      cancelled = true;
      if (pipelineRef.current) {
        pipelineRef.current.dispose();
        pipelineRef.current = null;
      }
      if (bvhBuffersRef.current) {
        disposeSceneBVH(bvhBuffersRef.current);
        bvhBuffersRef.current = null;
      }
      // Clear the bridge on unmount. Sweep finding (correctness 2026-05-08
      // Bug 14.1): clearing only the `walkaround` slot left the device,
      // adapter and frameTimings pointing at the now-disposed
      // WebGPURenderer. On a rapid walkaround off→on toggle, a fresh
      // RestirStage that reads w.__WGPU__.device synchronously (before the
      // gl factory's async init writes fresh handles) would consume the
      // stale device. Null everything; consumers must wait for the
      // factory's post-init write.
      const w = window as Window & typeof globalThis;
      if (w.__WGPU__) {
        w.__WGPU__.walkaround = undefined;
        w.__WGPU__.device = null;
        w.__WGPU__.adapter = null;
        w.__WGPU__.frameTimings = [];
      }
    };
  // Sweep finding Bug 2: timeOfDay added to deps so the BVH (with its
  // baked emitter Le) rebuilds when the user scrubs time-of-day. Without
  // this, the directional sun visibly tracked t-of-d while emitters
  // stayed at noon-radiance — visible as a 3.3× over-bright panel at
  // sunset. If scrubbing causes BVH-rebuild thrash for users, gate this
  // behind a 600ms debounce in a follow-up.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [wgpuDevice, scene, size.width, size.height, roomKey, graphFaces, refuseToMount, timeOfDay]);

  // Per-frame rendering — priority=1 tells r3f NOT to auto-call gl.render().
  // We own the entire frame: ReSTIR compute passes + composite render pass
  // write directly to the WebGPU swap-chain texture.
  useFrame(() => {
    if (refuseToMount) return;
    const pipeline = pipelineRef.current;
    const bvh = bvhBuffersRef.current;
    if (!pipeline || !bvh || !wgpuDevice) return;

    // ── 60 FPS frame cap ──────────────────────────────────────────────────
    // r3f's useFrame runs once per requestAnimationFrame, which on most
    // laptops/displays is 60 Hz (16.67 ms) but on high-refresh-rate
    // monitors (120/144/240 Hz) or when rAF is decoupled from vsync the
    // ReSTIR pipeline ends up dispatched 2–4× more often than the user
    // can perceive.  Each dispatch fires 5 atrous passes + RIS/temporal/
    // spatial reservoir compute + a per-pixel BVH walk per primary ray,
    // so uncapped throughput burns GPU time + battery for zero visible
    // benefit.  Skip the dispatch when fewer than ~16.67 ms have elapsed
    // since the previous one.  The first frame (lastFrameTsRef === 0)
    // always runs so the canvas has something to display immediately.
    const now = performance.now();
    if (lastFrameTsRef.current !== 0 &&
        now - lastFrameTsRef.current < TARGET_FRAME_INTERVAL_MS) {
      return;
    }
    lastFrameTsRef.current = now;

    const t0 = now;
    // Use DPR-scaled canvas dimensions (matches pipeline initialization).
    const canvasEl = (gl as unknown as { domElement: HTMLCanvasElement }).domElement;
    const W = canvasEl.width  || size.width;
    const H = canvasEl.height || size.height;

    // ── 1. Get swap-chain texture view ────────────────────────────────────
    // Must be obtained before submitting any GPU work this frame.
    const swapChainView = getSwapChainView(gl);
    if (!swapChainView) {
      // Swap chain not yet available — skip this frame.
      return;
    }
    const swapChainFormat = getSwapChainFormat();

    // ── 2. Prepare frame inputs ───────────────────────────────────────────
    const viewMatrix = new Float32Array(16);
    const projMatrix = new Float32Array(16);
    const prevViewMatrix = new Float32Array(16);

    // Extract matrices from three.js camera.
    const cam3 = camera as THREE.PerspectiveCamera;
    cam3.updateMatrixWorld(true);
    cam3.updateProjectionMatrix();

    // three.js stores matrices in column-major order; copy directly.
    viewMatrix.set(cam3.matrixWorldInverse.elements);
    projMatrix.set(cam3.projectionMatrix.elements);
    prevViewMatRef.current.toArray(prevViewMatrix);

    const inputs = {
      viewMatrix,
      projMatrix,
      prevViewMatrix,
      prevProjMatrix: projMatrix,
      cameraPos: cam3.getWorldPosition(new THREE.Vector3()).toArray() as [number, number, number],
      screenWidth: W,
      screenHeight: H,
      frameSeed: (frameCountRef.current * 1664525 + 1013904223) & 0xFFFFFFFF,
      totalEmissivePower: bvh.totalEmissivePower,
      emitterCount: bvh.emitterCount,
      // Use the live primary-light params captured at BVH-build time so
      // the shader's self-emission Le for primary glass hits reproduces
      // exactly the Le baked into the emitter list.
      primaryLightDir: sunParamsRef.current.direction,
      primaryLightIntensity: sunParamsRef.current.intensity,
      skyTint: lightingState.skyTint,
      skyIrradiance: lightingState.skyIrradiance,
      swapChainView,
      swapChainFormat,
    };

    // ── 3. Run ReSTIR compute pipeline + composite to swap-chain ─────────
    try {
      pipeline.renderFrame(inputs);
    } catch (err) {
      // Log once per 60 frames to avoid spam.
      if (frameCountRef.current % 60 === 0) {
        console.error('[WalkaroundStage] renderFrame error:', err);
      }
    }

    // Save view matrix for temporal reuse next frame.
    prevViewMatRef.current.copy(cam3.matrixWorldInverse);
    frameCountRef.current++;

    const ms = performance.now() - t0;

    // Record frame timing on the bridge.
    const w = window as Window & typeof globalThis;
    if (w.__WGPU__?.walkaround) {
      const timings = w.__WGPU__.walkaround.frameTimings;
      timings.push({ t: t0, ms });
      if (timings.length > 240) timings.shift();
    }
  }, 1);  // priority=1 — r3f skips its auto gl.render() when any subscriber has priority != 0

  // Refuse-to-mount path: render only the SwiftShader warning so the
  // canvas area shows an unambiguous error instead of a deceptive scene.
  // Copy preserved from the pre-Tier-2 WebGPUCanvas SwiftShaderRefusal
  // overlay — ReSTIR's tone is a yellow-on-dark "walkaround mode disabled"
  // banner, not the red warning DDGI uses; this is per-branch styling.
  if (refuseToMount) {
    const vendor = wgGate?.adapterVendor || '(unknown)';
    const architecture = wgGate?.adapterArchitecture || '(unknown)';
    return (
      <Html center style={{ pointerEvents: 'none' }}>
        <div
          role="alert"
          style={{
            background: '#1a1a2e',
            color: '#ffe066',
            border: '2px solid #ffe066',
            padding: '32px',
            borderRadius: 8,
            font: '14px/1.5 system-ui, sans-serif',
            maxWidth: 640,
            textAlign: 'center',
            boxShadow: '0 8px 24px rgba(0,0,0,0.6)',
            boxSizing: 'border-box',
          }}
        >
          <div style={{ fontSize: '20px', fontWeight: 600, marginBottom: '12px' }}>
            Walkaround mode disabled — software rasterizer detected
          </div>
          <div style={{ fontSize: '14px', opacity: 0.85, lineHeight: 1.5 }}>
            WebGPU requested a hardware adapter but Chromium returned the SwiftShader
            software fallback (<code>vendor=&quot;{vendor}&quot;</code>,{' '}
            <code>architecture=&quot;{architecture}&quot;</code>).  ReSTIR walkaround
            requires a real GPU (NVIDIA / AMD / Intel / Apple) to produce honest
            performance and visual results.  Open this page in a browser with
            hardware WebGPU enabled and reload.
          </div>
        </div>
      </Html>
    );
  }

  return (
    <>
      {/* Backdrop (same as RasterStage — walkaround inherits raster backdrop). */}
      {backdropMode === 'sky' && (
        <Sky
          distance={450000}
          sunPosition={skyParams.sunPosition}
          turbidity={skyParams.turbidity}
          rayleigh={skyParams.rayleigh}
          mieCoefficient={skyParams.mieCoefficient}
          mieDirectionalG={skyParams.mieDirectionalG}
        />
      )}
      {backdropMode === 'night' && (
        <Sky
          distance={450000}
          sunPosition={nightSkyParams.sunPosition}
          turbidity={nightSkyParams.turbidity}
          rayleigh={nightSkyParams.rayleigh}
          mieCoefficient={nightSkyParams.mieCoefficient}
          mieDirectionalG={nightSkyParams.mieDirectionalG}
        />
      )}
      {backdropMode === 'studio' && <Environment preset="studio" background />}
      {backdropMode === 'sunset' && <Environment preset="sunset" background />}

      <LightSourceList ctx={{ lightbox: lightboxDims, suppressSun }} />

      {roomKey && <RoomLoader roomKey={roomKey} mode="pt" />}

      {/* Glass panel faces — rendered as MeshPhysicalMaterial meshes so the
          BVH builder can pick up per-face transmission + color for emitter
          construction (§4.4) and material-color packing into bvhIndex[*].w. */}
      <FaceRenderer />

      {/* Lead/solder came geometry between cells. The bead meshes are real
          BoxGeometry/TubeGeometry boxes with opaque MeshPhysicalMaterial
          (transmission=0); StaticGeometryGenerator picks them up via
          traverseVisible and the BVH packs their dark metallic color into
          bvhIndex[*].w. Without this the panel had no visible tessellation
          outline; primary rays passed through the gaps between cells without
          hitting anything until reaching the room shell behind. */}
      <EdgeLines />

      <MountDispatch
        centerX={frameLayout.cx}
        centerY={frameLayout.cy}
        width={frameLayout.w}
        height={frameLayout.h}
        mode="pt"
        showBackboard={showBackboard}
      />

      {/*
        OrbitControls target — drives where the camera "looks at".

        Two regimes:

        • Chroma-test path (window.__CHROMA_TEST_CAMERA__ === true) — only
          set by the 13-walkaround-restir Playwright spec. In room mode this
          aims the camera DOWN at the floor (target [0, FLOOR_Y, -40]) so
          primary rays in the bottom 40 % of the screen hit floor geometry,
          which is where the chroma assertion samples. Geometry calibrated
          in commit a37355c:
            Camera at [0, 0, 80], FOV 50°. Target [0, -64, -40] ⇒ pitch
            θ = atan2(64, 120) = 28.07° below horizontal. Floor pixels at
            world z ∈ [-10, +30] land in screen-Y [0.638, 0.976] — the
            test samples screen-Y 0.60-1.00. The colored caustic from the
            panel (sun [0.5,1,0.5] through glass at z≈0) lands on the
            floor at z ≈ 17-47, so its centre (z≈30) sits at screen-Y
            0.976 (mid-bottom of test region, well within sample window).

        • Default / scene-load path (flag absent) — aim at the loaded
          panel's centre so the user sees a head-on framed view of their
          suncatcher in its room. With the camera positioned by
          StudioScene at [frameLayout.cx, frameLayout.cy, 60], target
          [frameLayout.cx, frameLayout.cy, 0] gives a horizontal,
          panel-centred look axis — panel + casing + some back wall
          margin all visible.

        In non-room (void-space) mode the chroma test doesn't apply, so we
        keep the panel-centre target unconditionally for normal frame
        inspection (matches the previous behaviour).
      */}
      {(() => {
        const chromaTestCamera =
          typeof window !== 'undefined' &&
          (window as Window & { __CHROMA_TEST_CAMERA__?: boolean }).__CHROMA_TEST_CAMERA__ === true;
        // Default room-mode target: 30″ BELOW panel centre.  Calibrated
        // (alongside the camera at (panelCenter.cx, 0, 160) in
        // StudioScene.tsx) so the sun-caustic patch on the floor at
        // z∈[61, 73] (where the sun shadow ray through the panel cells
        // lands; see StudioScene comment for the geometry derivation)
        // projects to ~76-96% screen-Y — visible at the bottom of the
        // frame, where the user expects coloured light splashes from
        // the suncatcher to land.
        const target: [number, number, number] = roomKey
          ? (chromaTestCamera
              ? [orbitTarget[0], FLOOR_Y, -40]
              : [frameLayout.cx, frameLayout.cy - 30, 0])
          : orbitTarget;
        return (
          <OrbitControls
            makeDefault
            enableDamping
            dampingFactor={0.08}
            target={target}
            minPolarAngle={Math.PI * 0.05}
            maxPolarAngle={Math.PI * 0.95}
            minAzimuthAngle={-Math.PI}
            maxAzimuthAngle={ Math.PI}
            minDistance={1}
            maxDistance={roomKey ? (roomOrbitRadius ?? 200) : Math.max(frameLayout.w, frameLayout.h) * 6}
            enablePan
          />
        );
      })()}

      <WalkaroundDebugBridge passes={passes} />
    </>
  );
}
