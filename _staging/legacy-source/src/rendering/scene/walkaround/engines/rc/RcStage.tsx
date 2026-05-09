/**
 * WalkaroundStage — Radiance Cascades 3D walk-around mode (§9.1).
 *
 * Mounted when `viewport.exploreEnabled === true` and
 * `viewport.walkaroundEngine === 'rc'` (locked-decision #3, branch-parity
 * layered Redux shape — replaces the legacy `walkaroundEnabled` boolean
 * and `window.__WALKAROUND_RC__` ad-hoc flag).
 *
 * Rendering loop:
 *   1. Standard raster primary pass (three.js default pipeline).
 *   2. Probe ray-cast compute (C0–C4).
 *   3. Cascade merge compute (C3→C0 bottom-up).
 *   4. GI receiver materials sample C0 for indirect lighting.
 *
 * Walk camera: useWalkCamera (WASD + scripted-path for tests).
 * GI: useGIReceiverConverter wraps wall/floor materials.
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import { useThree, useFrame } from '@react-three/fiber';
import { useSelector } from 'react-redux';
import { WebGPURenderer, RectAreaLightNode } from 'three/webgpu';
import * as THREE from 'three';
import { RectAreaLightTexturesLib } from 'three/examples/jsm/lights/RectAreaLightTexturesLib.js';
import { MountDispatch } from '../../../mounts/MountDispatch';
import { lightboxDimsFor } from '../../../mounts/lightboxDims';
import { LightSourceList } from '../../../lighting/renderLightSource';
import { RoomLoader } from '../../../../assets/RoomLoader';
import type { SkyParams } from '../../../skyParams';
import type { BackdropMode } from '@/store/uiSlice';
import type { RootState } from '@/store';
import { selectGraph } from '@/store/selectors';
import { useSceneBVH } from '../../useSceneBVH';
import { useCascadeBuffers } from '../../useCascadeBuffers';
import { useGIReceiverConverter } from '../../giReceiver';
import { useWalkCamera } from '@/hooks/useWalkCamera';
import { dispatchCascadePasses } from '../../cascadeDispatch';
import { skyParamsFor } from '../../../skyParams';

// Initialize RectAreaLight LTC lookup textures at module load time, before any
// WebGPU material setup can occur. RectAreaLightNode.setup() reads the LTC
// textures synchronously during the first frame render — if setLTC() hasn't
// been called yet (e.g. useEffect hasn't fired) it throws a null crash.
// Module-level init guarantees it runs before the first JSX render.
RectAreaLightTexturesLib.init();
// three.js typedef inconsistency: setLTC expects an instance type, but the
// runtime expects the static object (the lib uses a static-only pattern).
// eslint-disable-next-line @typescript-eslint/no-explicit-any
RectAreaLightNode.setLTC(RectAreaLightTexturesLib as any);

interface RcStageProps {
  backdropMode: BackdropMode;
  skyParams: SkyParams;
  nightSkyParams: SkyParams;
  frameLayout: { cx: number; cy: number; w: number; h: number };
  /** Accepted for prop-shape parity with DDGI/ReSTIR stages; RC drives
   *  its camera through useWalkCamera and ignores this. */
  orbitTarget: [number, number, number];
}

/** Current sun direction from the sky params. Approximate — good enough for cascade. */
function sunDirFromSkyParams(sp: SkyParams): THREE.Vector3 {
  const [sx, sy, sz] = sp.sunPosition;
  return new THREE.Vector3(sx, sy, sz).normalize();
}

/** Approximate sun color from time-of-day (warm at dawn/dusk, white at noon). */
function sunColorFromSkyParams(sp: SkyParams): THREE.Color {
  const elev = sp.sunPosition[1];
  if (elev < 0.05) return new THREE.Color(0.0, 0.0, 0.0); // sun below horizon
  const t = THREE.MathUtils.clamp(elev / 0.5, 0, 1);
  // Warm yellow → white
  return new THREE.Color(
    THREE.MathUtils.lerp(1.0, 1.0, t),
    THREE.MathUtils.lerp(0.6, 1.0, t),
    THREE.MathUtils.lerp(0.2, 0.95, t),
  ).multiplyScalar(2.0);
}

export function RcStage({
  backdropMode,
  // skyParams, nightSkyParams: walkaround derives sky internally via
  // skyParamsFor(timeOfDay) for the cascade compute pass; the prop-passed
  // sky params are unused here. Kept on the prop signature for API parity
  // with RasterStage / PTStage.
  skyParams: _skyParams,
  nightSkyParams: _nightSkyParams,
  frameLayout,
}: RcStageProps) {
  const { gl, scene, camera } = useThree();
  const mount  = useSelector((s: RootState) => s.scene.mount);
  const space  = useSelector((s: RootState) => s.scene.space);
  const timeOfDay = useSelector((s: RootState) => s.viewport.timeOfDay);
  // Graph slice is a BVH-rebuild trigger — when the user adds/removes/edits
  // panel faces the glass meshes change but the THREE.Scene reference stays
  // stable (R3F mutates it in place). Without this dep the BVH would only
  // ever build with whatever faces were present at first commit, leaving
  // later edits invisible to ray-cast / GI.
  const graphState = useSelector(selectGraph);
  const lightboxDims = lightboxDimsFor(mount, frameLayout.w, frameLayout.h);
  const roomKey = space.kind === 'room' ? space.roomKey : null;
  const suppressSun = backdropMode === 'studio' || backdropMode === 'sunset';

  // Refs that track the latest pipeline state for use inside useEffect closures
  // (where React state variables are stale after the first render).
  const sceneBVHRef      = useRef<ReturnType<typeof useSceneBVH>>(null);
  const cascadeBuffersRef = useRef<ReturnType<typeof useCascadeBuffers>>(null);
  const timeOfDayRef      = useRef(timeOfDay);
  /**
   * Last dispatched frame's `performance.now()` timestamp.  Drives the
   * 60 FPS frame cap below.  Cascade compute (probe ray-cast + merge) at
   * canvas resolution is GPU-bound at well above 60 FPS on a discrete
   * card; uncapped, the whole pipeline runs as fast as the GPU can
   * dispatch (often 200+ FPS), wasting power and battery for zero
   * perceptible visual gain.  Throttling to ~16.67 ms/frame matches a
   * 60 Hz display's refresh rate, which is the user-perceivable ceiling
   * on a typical monitor.
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

  // Option F (hardware-gpu-validation-spec.md §3): fail-fast SwiftShader gate.
  // When SwiftShader is detected we refuse to mount the walkaround pipeline.
  // Software-rasterized output has historically been mistaken for hardware-GPU
  // output during validation; refusing to mount makes that mistake impossible.
  const [swiftShaderDetected, setSwiftShaderDetected] = useState(false);

  // ── Format-mismatch fix: disable Three.js's internal HDR framebuffer target ──
  //
  // R3F sets `gl.toneMapping = ACESFilmicToneMapping` and
  // `gl.outputColorSpace = SRGBColorSpace` by default. With WebGPURenderer this
  // triggers Three.js's internal `_frameBufferTarget` — an HDR (rgba16float) RT
  // that the scene is rendered into, then tone-mapped + color-space-converted
  // into the swap-chain in a final blit pass.
  //
  // The HDR `_frameBufferTarget` collides with the WebGPU transmission
  // compositor (`MeshPhysicalMaterial.transmission > 0` → PhysicalLightingModel
  // → `viewportMipTexture()`), which copies the framebuffer into a default
  // `FramebufferTexture` to sample for refraction. The destination texture's
  // format is not reliably inferred from the active render target, and on
  // hardware GPU it locks in at `rgba8unorm` while the source is `rgba16float`.
  // Result: `WebGPUBackend: copyFramebufferToTexture: Source and destination
  // formats do not match` fires once per glass triangle per frame (~5060/frame
  // on the honeycomb-room scene), which forces an early return inside
  // `copyFramebufferToTexture` and short-circuits the cascade pipeline init.
  //
  // Setting `toneMapping = NoToneMapping` + `outputColorSpace = LinearSRGBColorSpace`
  // returns null from `Renderer._getFrameBufferTarget()` so the renderer renders
  // directly into the canvas swap-chain — both source and destination of the
  // transmission compositor's copy resolve to the same canvas format
  // (`bgra8unorm` on Chrome desktop), eliminating the mismatch.
  //
  // Linear output is the physically-correct space for the cascade GI pipeline
  // (radiance values are already calibrated). A separate tone-map post-pass
  // can be layered in later if final-output color grading is required.
  useEffect(() => {
    const renderer = gl as unknown as THREE.WebGLRenderer;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const anyR = renderer as any;
    if (anyR.backend?.isWebGPUBackend === true) {
      renderer.toneMapping = THREE.NoToneMapping;
      renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
    }
  }, [gl]);

  // Expose WebGPU backend info for test assertion.
  // Uses navigator.gpu.requestAdapter to read adapter.info.vendor — the only
  // reliable way to distinguish hardware from SwiftShader. device.label is always
  // empty (it's a user-set field, not the GPU driver name).
  useEffect(() => {
    const renderer = gl as unknown as WebGPURenderer;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const anyR = renderer as any;
    const isWebGPU = anyR.backend?.isWebGPUBackend === true;

    void (async () => {
      let vendor = '';
      let architecture = '';
      let adapterName = 'WebGPU-software';
      if (isWebGPU && navigator.gpu) {
        const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' }).catch(() => null);
        if (adapter) {
          vendor       = adapter.info?.vendor       ?? '';
          architecture = adapter.info?.architecture ?? '';
          // Hardware if vendor is not 'google' (SwiftShader) and not empty.
          if (vendor && vendor !== 'google') {
            adapterName = `${vendor}-${architecture}`;
          }
        }
      }
      // hardware = WebGPU backend + real driver (not SwiftShader / WARP / llvmpipe)
      const isHardwareGpu = isWebGPU && vendor.length > 0 && vendor !== 'google';
      // Set the namespace BEFORE triggering the React state update, so any
      // test or downstream observer that reacts to the unmount sees the
      // adapter info immediately.
      window.__WG__ = {
        isWebGPU,
        isHardwareGpu,
        adapterVendor: vendor,
        adapterArchitecture: architecture,
        adapter: isWebGPU ? { name: adapterName } : undefined,
      };
      // Option F: SwiftShader = `vendor === 'google' && architecture === 'swiftshader'`.
      // Anything else (including empty vendor on adapter-request failure) is left
      // alone — only the explicit SwiftShader signature triggers the gate.
      if (isWebGPU && vendor === 'google' && architecture === 'swiftshader') {
        console.error(
          '[WalkaroundStage] SwiftShader (software rasterizer) detected — refusing to mount. ' +
          'Validation requires a hardware WebGPU adapter (NVIDIA / AMD / Intel / Apple).',
        );
        setSwiftShaderDetected(true);
      }
    })();
  }, [gl]);

  // Option F: when SwiftShader is detected, render a DOM-level error overlay
  // outside the R3F canvas tree (since the canvas itself is already mounted
  // with the WebGPU renderer). The overlay is positioned over the canvas and
  // explains why the walkaround pipeline did not start.
  useEffect(() => {
    if (!swiftShaderDetected) return;
    const overlay = document.createElement('div');
    overlay.setAttribute('data-walkaround-swiftshader-error', 'true');
    overlay.style.cssText = [
      'position: absolute',
      'inset: 0',
      'z-index: 9999',
      'display: flex',
      'flex-direction: column',
      'align-items: center',
      'justify-content: center',
      'background: rgba(20, 0, 0, 0.92)',
      'color: #ffd9d9',
      'font-family: system-ui, sans-serif',
      'padding: 24px',
      'text-align: center',
      'pointer-events: auto',
    ].join('; ');
    overlay.innerHTML = [
      '<h2 style="font-size: 22px; margin: 0 0 12px 0; color: #ff6b6b;">',
      'Walk-around mode unavailable — software rasterizer detected',
      '</h2>',
      '<p style="font-size: 15px; margin: 0 0 8px 0; max-width: 520px;">',
      'WebGPU is running on Google SwiftShader (CPU). The Radiance Cascades ',
      'pipeline requires a hardware GPU adapter (NVIDIA, AMD, Intel, or Apple). ',
      '</p>',
      '<p style="font-size: 13px; opacity: 0.8; margin: 8px 0 0 0;">',
      'Open the app in a Chrome window with hardware GPU enabled, ',
      'or disable walk-around mode to continue in the standard raster view.',
      '</p>',
    ].join('');
    document.body.appendChild(overlay);
    return () => {
      overlay.remove();
    };
  }, [swiftShaderDetected]);

  // Walk camera shim (§9.2).
  // Pass frameLayout so the hook can pose the camera head-on at the panel
  // center as the default exploration starting view (when no `__WALK_PATH__`
  // scripted path is installed). Scripted paths still take precedence — the
  // chroma-test e2e installs its keyframes BEFORE measurement so its
  // panel-wall view continues to drive caustic-channel readings unchanged.
  useWalkCamera({ camera, sceneRef: scene });

  // BVH (rebuilds on scene topology changes, debounced). Trigger composes
  // roomKey (room mesh swap) with graphState (panel-face mutations) so any
  // change that adds/removes mesh triangles forces a rebuild. Memoised so
  // the trigger object is reference-stable across renders that don't mutate
  // either input — otherwise useSceneBVH's `[scene, trigger]` dep would fire
  // on every render and the 100ms debounce would never elapse to completion.
  const bvhTrigger = useMemo(() => ({ roomKey, graph: graphState }), [roomKey, graphState]);
  const sceneBVH = useSceneBVH(scene, bvhTrigger);

  // Cascade buffers (allocated when BVH bounds available).
  const cascadeBuffers = useCascadeBuffers(sceneBVH?.bounds);

  // Keep refs in sync so the manual-render hook always sees fresh pipeline state.
  sceneBVHRef.current      = sceneBVH;
  cascadeBuffersRef.current = cascadeBuffers;
  timeOfDayRef.current      = timeOfDay;

  // GI receiver material wrap.
  useGIReceiverConverter(cascadeBuffers);

  // Per-frame frame-timing bridge for e2e tests.
  // requestAnimationFrame is throttled in headless Chromium when a WebGPU
  // canvas is active (the GPU canvas compositor runs on a different schedule).
  // Inject timing samples from within R3F's own render loop instead.
  useFrame((_, delta) => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const w = window as any;
    // Always increment total frame counter for diagnostics.
    w.__R3F_FRAMES__ = (w.__R3F_FRAMES__ ?? 0) + 1;
    // Push to fps samples if the test counter is installed.
    if (!w.__FPS__) return;
    const ms = delta * 1000;
    w.__FPS__.samples.push(ms);
  });

  // Per-frame cascade compute dispatch (runs before render at priority -1,
  // ensuring cascade data is ready before the geometry renders).
  useFrame(async (state) => {
    if (!sceneBVH || !cascadeBuffers) return;

    // ── 60 FPS frame cap ──────────────────────────────────────────────────
    // r3f's useFrame runs once per requestAnimationFrame, which on most
    // laptops/displays is 60 Hz (16.67 ms) but on high-refresh-rate
    // monitors (120/144/240 Hz) or when rAF is decoupled from vsync the
    // cascade pipeline ends up dispatched 2–4× more often than the user
    // can perceive.  Each dispatch fires the probe ray-cast + cascade
    // merge compute (per-probe BVH walks across C0–C4) and a primary
    // render pass, so uncapped throughput burns GPU time + battery for
    // zero visible benefit.  Skip the dispatch when fewer than ~16.67 ms
    // have elapsed since the previous one.  The first frame
    // (lastFrameTsRef === 0) always runs so the canvas has something to
    // display immediately.
    const now = performance.now();
    if (lastFrameTsRef.current !== 0 &&
        now - lastFrameTsRef.current < TARGET_FRAME_INTERVAL_MS) {
      return;
    }
    lastFrameTsRef.current = now;

    const skyP = skyParamsFor(timeOfDay);
    const sunDir   = sunDirFromSkyParams(skyP);
    const sunColor = sunColorFromSkyParams(skyP);

    await dispatchCascadePasses({
      gl:             gl as unknown as WebGPURenderer,
      sceneBVH,
      cascadeBuffers,
      sunDirection:   sunDir,
      sunColor,
      envEquirect:    scene.environment,
      frameSeed:      Math.floor(state.clock.elapsedTime * 100) & 0xFF,
      debugFill:      false,
    });

  });

  // Canvas readback bridge for e2e tests.
  // WebGPU discards the swap-chain texture after present(), so canvas.toDataURL()
  // always returns transparent black. Fix: render the scene into an explicit
  // THREE.RenderTarget inside useFrame, async-read the pixels to CPU, encode a
  // data URL, and patch canvas.toDataURL() to return it.
  //
  // This runs every 3rd R3F frame (throttled) to avoid stalling the pipeline.
  // Note: fps measurement is via window.__FPS__ populated by the useFrame fps bridge
  // above — NOT via requestAnimationFrame, which is throttled for WebGPU canvases.
  const rtRef    = useRef<THREE.RenderTarget | null>(null);
  const mirrorRef = useRef<{ canvas: HTMLCanvasElement; ctx: CanvasRenderingContext2D } | null>(null);
  const capturingRef   = useRef(false);
  const frameCountRef  = useRef(0);
  /** True when the readback RT's backing GPU format stores channels as BGRA. */
  const rtIsBgraRef    = useRef(false);

  useEffect(() => {
    const gpuCanvas = (gl as unknown as THREE.WebGLRenderer).domElement;
    if (!gpuCanvas) return;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const glAny = gl as any;

    const W = gpuCanvas.width  || 1280;
    const H = gpuCanvas.height || 720;

    // Render target compatible with WebGPURenderer.
    //
    // Format must match the canvas swap-chain format. When the readback
    // render's RT format differs from the canvas, the WebGPU transmission
    // compositor's FramebufferTexture singleton (in PhysicalLightingModel)
    // gets cached at one format and then mismatches the other on the next
    // render — `WebGPUBackend.copyFramebufferToTexture` then logs `Source and
    // destination formats do not match` and short-circuits, so transmission
    // samples become noise.
    //
    // Three.js's WebGPU canvas is configured with
    // `navigator.gpu.getPreferredCanvasFormat()` — `bgra8unorm` on Chrome
    // desktop, `rgba8unorm` on Safari/Quest. We mirror that exact format on
    // the readback RT via `internalFormat`, so the FramebufferTexture
    // singleton stays stable across the readback render → main render
    // boundary. The CPU-side `format`/`type` stay RGBA/UnsignedByte so
    // `readRenderTargetPixelsAsync` returns 4 bytes/pixel as expected; if the
    // canvas chose BGRA, we swap channels below before PNG encoding.
    const preferredFmt = (navigator.gpu?.getPreferredCanvasFormat?.() ?? 'rgba8unorm') as 'bgra8unorm' | 'rgba8unorm';
    const rt = new THREE.RenderTarget(W, H, {
      format: THREE.RGBAFormat,
      type: THREE.UnsignedByteType,
    });
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (rt.texture as any).internalFormat = preferredFmt;
    rtIsBgraRef.current = preferredFmt === 'bgra8unorm';
    rtRef.current = rt;

    const mirrorCanvas = document.createElement('canvas');
    mirrorCanvas.width  = W;
    mirrorCanvas.height = H;
    mirrorRef.current = { canvas: mirrorCanvas, ctx: mirrorCanvas.getContext('2d')! };

    let cachedDataUrl = '';

    const _original = gpuCanvas.toDataURL.bind(gpuCanvas);
    gpuCanvas.toDataURL = (type?: string, quality?: number): string => {
      if (cachedDataUrl) return cachedDataUrl;
      return _original(type, quality);
    };
    glAny.__setMirrorUrl = (url: string) => { cachedDataUrl = url; };

    return () => {
      gpuCanvas.toDataURL = _original;
      glAny.__setMirrorUrl = undefined;
      rt.dispose();
      rtRef.current = null;
      mirrorRef.current = null;
      capturingRef.current = false;
    };
  }, [gl]);

  // Per-frame readback (every 3rd frame).
  useFrame(() => {
    if (capturingRef.current) return;
    frameCountRef.current += 1;
    if (frameCountRef.current % 3 !== 0) return;

    const rt     = rtRef.current;
    const mirror = mirrorRef.current;
    if (!rt || !mirror) return;

    const renderer  = gl as unknown as THREE.WebGLRenderer;
    const gpuCanvas = renderer.domElement;
    const cw = gpuCanvas.width  || rt.width;
    const ch = gpuCanvas.height || rt.height;

    if (rt.width !== cw || rt.height !== ch) {
      rt.setSize(cw, ch);
      mirror.canvas.width  = cw;
      mirror.canvas.height = ch;
    }

    // WebGPURenderer.setRenderTarget accepts THREE.RenderTarget; the WebGL
    // type signature on the cast type is narrower (WebGLRenderTarget). Cast
    // through `any` for the off-cycle render-target type.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (renderer as any).setRenderTarget(rt);
    renderer.render(scene, camera);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (renderer as any).setRenderTarget(null);

    capturingRef.current = true;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const readbackResult: Promise<Uint8Array | null> = (renderer as any).readRenderTargetPixelsAsync
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      ? (renderer as any).readRenderTargetPixelsAsync(rt, 0, 0, cw, ch)
      : Promise.resolve(null);

    readbackResult.then((buf) => {
      if (!buf) {
        const syncBuf = new Uint8Array(cw * ch * 4);
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        (renderer as any).readRenderTargetPixels?.(rt, 0, 0, cw, ch, syncBuf);
        buf = syncBuf;
      }
      // Copy to a freshly-allocated ArrayBuffer-backed Uint8ClampedArray so
      // the ImageData ctor sees a non-shared buffer (TS lib types reject
      // SharedArrayBuffer-backed views even though runtime accepts them).
      // When the RT's GPU format is bgra8unorm (Chrome desktop) we swap B↔R
      // while copying so PNG encoding sees RGBA byte order. When it's
      // rgba8unorm (Safari/Quest) the byte order already matches and we just
      // copy through.
      const clamped = new Uint8ClampedArray(buf.length);
      if (rtIsBgraRef.current) {
        for (let i = 0; i < buf.length; i += 4) {
          clamped[i + 0] = buf[i + 2]!;  // R ← B
          clamped[i + 1] = buf[i + 1]!;  // G ← G
          clamped[i + 2] = buf[i + 0]!;  // B ← R
          clamped[i + 3] = buf[i + 3]!;  // A ← A
        }
      } else {
        clamped.set(buf);
      }
      const imageData = new ImageData(clamped, cw, ch);
      mirror.ctx.putImageData(imageData, 0, 0);
      const url = mirror.canvas.toDataURL('image/png');
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const glAny = gl as any;
      if (glAny.__setMirrorUrl) glAny.__setMirrorUrl(url);
      capturingRef.current = false;
    }).catch(() => { capturingRef.current = false; });
  });

  // ── manual-render hook (§10): RAF-independent single-frame trigger ──────
  //
  // MCP-controlled Chrome tabs report document.visibilityState === 'hidden',
  // which causes RAF to throttle to ~0 Hz. R3F's useFrame callbacks stop
  // firing in hidden tabs — no rendering happens via the normal path.
  //
  // Solution: expose window.__WGPU__.walkaround.renderOneFrame() so external
  // test code can drive one full RC pipeline frame (probe ray-cast → cascade
  // merge → GI shading → composite) without depending on requestAnimationFrame.
  //
  // The hook uses refs (sceneBVHRef, cascadeBuffersRef, timeOfDayRef) that are
  // always current — see the assignment sites above.
  useEffect(() => {
    const renderer  = gl as unknown as WebGPURenderer;

    const renderOneFrame = async (
      cameraOverride?: { position: [number, number, number]; lookAt: [number, number, number] },
    ): Promise<void> => {
      // Apply optional camera override.
      if (cameraOverride) {
        const cam = camera as unknown as THREE.PerspectiveCamera;
        cam.position.set(...cameraOverride.position);
        cam.lookAt(...cameraOverride.lookAt);
        cam.updateMatrixWorld(true);
      }

      // Run the cascade compute pipeline (probe ray-cast → cascade merge).
      const bvh  = sceneBVHRef.current;
      const bufs = cascadeBuffersRef.current;
      if (bvh && bufs) {
        const skyP     = skyParamsFor(timeOfDayRef.current);
        const sunDir   = sunDirFromSkyParams(skyP);
        const sunColor = sunColorFromSkyParams(skyP);

        await dispatchCascadePasses({
          gl:             renderer,
          sceneBVH:       bvh,
          cascadeBuffers: bufs,
          sunDirection:   sunDir,
          sunColor,
          envEquirect:    scene.environment,
          frameSeed:      Math.floor(performance.now() / 10) & 0xFF,
          debugFill:      false,
        });
      }

      // Render the scene to the primary swap-chain surface (GI-shaded composite).
      (renderer as unknown as THREE.WebGLRenderer).render(scene, camera);
    };

    // Namespace: window.__WGPU__.walkaround — single-frame render hook for
    // tests that drive Chrome tabs where requestAnimationFrame is throttled
    // (visibilityState === 'hidden' under MCP-Chrome).
    const w = window as unknown as {
      __WGPU__?: {
        walkaround?: {
          renderOneFrame: typeof renderOneFrame;
          isInitialized: () => boolean;
        };
      };
    };
    w.__WGPU__ = w.__WGPU__ ?? {};
    w.__WGPU__.walkaround = {
      renderOneFrame,
      /** Returns true once the BVH + cascade buffers are ready. */
      isInitialized: () => sceneBVHRef.current !== null && cascadeBuffersRef.current !== null,
    };

    return () => {
      // Clean up on unmount.
      if (w.__WGPU__?.walkaround) {
        delete w.__WGPU__.walkaround;
      }
    };
  }, [gl, camera, scene]);

  // Option F: refuse to mount on SwiftShader. All hooks above must keep running
  // in stable order, so the gate is placed AFTER hook calls. The DOM overlay
  // (mounted via the useEffect above) reports the failure to the user; here we
  // simply skip rendering the walkaround scene tree.
  if (swiftShaderDetected) {
    return null;
  }

  return (
    <>
      {/* Sky backdrop: drei's <Sky> uses THREE.ShaderMaterial which is not
          compatible with WebGPURenderer's NodeMaterial pipeline and would
          log a console error every frame. Use a plain scene background colour
          instead; the cascade env-map handles sky-light contribution. */}
      {backdropMode === 'sky'  && <color attach="background" args={['#87ceeb']} />}
      {backdropMode === 'night' && <color attach="background" args={['#0a0a1a']} />}

      {/* Light sources (direct + fixtures) */}
      <LightSourceList ctx={{ lightbox: lightboxDims, suppressSun }} />

      {/* Room geometry */}
      {roomKey && <RoomLoader roomKey={roomKey} mode="raster" />}

      {/* Panel geometry */}
      <MountDispatch
        centerX={frameLayout.cx}
        centerY={frameLayout.cy}
        width={frameLayout.w}
        height={frameLayout.h}
        mode="raster"
        showBackboard={false}
      />

      {/* Orientation gizmo: GizmoHelper/GizmoViewport use gl.capabilities which is
          WebGL-only and not available on WebGPURenderer. Omitted in walkaround mode. */}
    </>
  );
}
