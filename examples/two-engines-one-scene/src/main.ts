/**
 * Gate G2: identical @vitrum/core Scene drives pt-webgl (WebGL2) and walkaround-hybrid (WebGPU).
 */

import type { Engine, FrameInput, Mat4, Scene } from '@vitrum/core';
import { buildComplexThreeScene, buildCornellBoxThreeScene } from '@vitrum-examples/shared';
import { createPTEngine_WebGL2 } from '@vitrum/pt-webgl';
// pt-webgpu drives the bottom canvas headlessly (no swap-chain present;
// pt-webgpu accumulates to an internal HDR texture, leaving the host to
// implement display). The engine drive verifies WGSL params layout,
// scene buffer uploads, and the path-trace dispatch on real hardware.
import { createPTEngine_WebGPU } from '@vitrum/pt-webgpu';
import { sceneFromThreeJS } from '@vitrum/three-bindings';
import {
  createWalkaroundEngine_Hybrid,
  HYBRID_WEBGPU_REQUIRED_LIMITS,
} from '@vitrum/walkaround-hybrid';

if (typeof console !== 'undefined' && console.debug) {
  console.debug('[two-engines] pt-webgpu factory loaded:', typeof createPTEngine_WebGPU);
}
import * as THREE from 'three';

function mat4FromThree(m: THREE.Matrix4): Mat4 {
  return new Float32Array(m.elements);
}

function resizeCanvasToDisplaySize(canvas: HTMLCanvasElement): void {
  const dpr = Math.min(window.devicePixelRatio, 2);
  const w = Math.floor(canvas.clientWidth * dpr);
  const h = Math.floor(canvas.clientHeight * dpr);
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
}

async function waitEngineReady(engine: Engine, timeoutMs: number): Promise<boolean> {
  const t0 = performance.now();
  while (performance.now() - t0 < timeoutMs) {
    if (engine.state === 'ready') return true;
    if (engine.state === 'disposed') return false;
    await new Promise<void>((r) => setTimeout(r, 40));
  }
  return false;
}

function nonNull<T>(v: T | null, name: string): T {
  if (v == null) throw new Error(`Missing DOM node: ${name}`);
  return v;
}

async function main(): Promise<void> {
  // Per-engine pages omit some canvases. nonNullIf returns the canvas if
  // present and required by mode; throws if required-but-missing; returns
  // a 1×1 placeholder if not-required (so loops can still reference the
  // variable without conditional plumbing throughout).
  const canvasPt    = document.querySelector<HTMLCanvasElement>('#c-pt');
  const canvasWgpu  = document.querySelector<HTMLCanvasElement>('#c-wgpu');
  const canvasPtGpu = document.querySelector<HTMLCanvasElement>('#c-ptgpu');
  const statusEl    = nonNull(document.querySelector<HTMLDivElement>('#status'), '#status');

  const lines: [string, string, string] = ['', '', ''];

  // URL params drive validation-harness toggles. Pre-defined keys:
  //   ?cameraMotion=1            — slow orbit of the camera (drives #4 motion path)
  //   ?ppgEnabled=1              — enable PPG (#6 path)
  //   ?denoiser=atrous           — use legacy 3-iter atrous instead of SVGF (#7)
  //   ?quality=interactive|final|capture|safe — pt-webgl scheduler mode (#9)
  //   ?ptWebgpuBounces=N         — pt-webgpu bounce-depth override (default 4)
  //   ?samplesTarget=N           — pt-webgl + pt-webgpu convergence target
  //   ?scene=cornell|complex     — scene selector (#11)
  const params = new URLSearchParams(window.location.search);
  // `mode` selects which engine(s) to initialize. Each per-engine page
  // (pt-webgl.html / walkaround.html / pt-webgpu.html) sets the
  // body[data-engine-mode] attribute, which we read here as a fallback.
  const bodyMode = (document.body.getAttribute('data-engine-mode') ?? 'all') as
    | 'all' | 'ptwebgl' | 'walkaround' | 'ptwebgpu';
  const mode = (params.get('mode') ?? bodyMode) as typeof bodyMode;
  // `walkaround-webgl2.html` sets data-walkaround-mode="1" to force pt-webgl
  // into interactive + camera-motion defaults so the page behaves like a
  // WebGL2 walkaround (cross-engine fallback for users without WebGPU).
  // URL params still override (e.g. ?cameraMotion=0 freezes the orbit).
  const walkaroundFallback = document.body.getAttribute('data-walkaround-mode') === '1';
  const FLAGS = {
    mode,
    cameraMotion: params.has('cameraMotion')
      ? params.get('cameraMotion') === '1'
      : walkaroundFallback,
    ppgEnabled: params.get('ppgEnabled') === '1',
    denoiser: (params.get('denoiser') ?? 'svgf') as 'svgf' | 'atrous',
    quality: (params.get('quality')
      ?? (walkaroundFallback ? 'interactive' : 'capture')) as 'interactive' | 'final' | 'capture' | 'safe',
    ptWebgpuBounces: parseInt(params.get('ptWebgpuBounces') ?? '4', 10) || 4,
    samplesTarget: parseInt(params.get('samplesTarget')
      ?? (walkaroundFallback ? '256' : '32'), 10) || 32,
    scene: (params.get('scene') ?? 'cornell') as 'cornell' | 'complex',
  };
  const RUN = {
    ptWebgl: mode === 'all' || mode === 'ptwebgl',
    walkaround: mode === 'all' || mode === 'walkaround',
    ptWebgpu: mode === 'all' || mode === 'ptwebgpu',
  };

  // Expose engine telemetry as window.__vitrum so the validation harness
  // (driven via Claude-in-Chrome or DevTools) can poll without touching DOM.
  const telemetry: {
    flags: typeof FLAGS;
    ptWebgl?: { state: string; spp: number; target: number; converged: boolean };
    walkaround?: {
      state: string;
      frame: number;
      debugTimingsLen: number;
      lastGpuTimings?: Record<string, number>;
      lastGpuTimingsFrame?: number;
    };
    ptWebgpu?: { state: string; spp: number; target: number; converged: boolean };
    memory?: { usedJSHeapMB: number; totalJSHeapMB: number; jsHeapLimitMB: number; sampledAtFrame: number };
    cameraMotion?: { angleDeg: number; moving: boolean };
  } = { flags: FLAGS };
  (globalThis as unknown as { __vitrum: typeof telemetry }).__vitrum = telemetry;

  // ── Camera motion (#4) ────────────────────────────────────────────────
  // When cameraMotion=1, slowly orbit the camera around the scene origin.
  // Each animation tick advances the orbit angle by a small delta. The
  // walkaround engine's CAMERA_MOVE_RESET_THRESHOLD_SQ should fire and
  // reset the temporal accumulator — confirm via debugTimingsLen / frame
  // count behaviour.
  let cameraAngle = 0;
  let motionStartTs = 0;

  // ── Memory probe (#8) ─────────────────────────────────────────────────
  // Poll performance.memory every ~30 frames. Surfaces gradual heap growth
  // and JS-side leak risk. Real GPU-memory usage is not exposed by the
  // browser; this catches JS-side growth (closures, event listeners, etc.).
  function pollMemory(frameIdx: number): void {
    if (frameIdx % 30 !== 0) return;
    const m = (performance as unknown as { memory?: { usedJSHeapSize: number; totalJSHeapSize: number; jsHeapSizeLimit: number } }).memory;
    if (!m) return;
    telemetry.memory = {
      usedJSHeapMB: +(m.usedJSHeapSize / (1024 * 1024)).toFixed(2),
      totalJSHeapMB: +(m.totalJSHeapSize / (1024 * 1024)).toFixed(2),
      jsHeapLimitMB: +(m.jsHeapSizeLimit / (1024 * 1024)).toFixed(2),
      sampledAtFrame: frameIdx,
    };
  }

  function refreshStatus(): void {
    statusEl.textContent = lines.join('\n').trim() || '…';
  }

  const threeScene = FLAGS.scene === 'complex'
    ? buildComplexThreeScene()
    : buildCornellBoxThreeScene();
  const vitrumScene: Scene = sceneFromThreeJS(threeScene);

  const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 50);
  camera.position.set(-0.05, 0, 2.75);
  camera.lookAt(-0.05, -0.15, 0);

  // ── WebGL2 path trace ─────────────────────────────────────────────────
  // pt-webgl skipped entirely when mode doesn't include it or canvas absent.
  if (!RUN.ptWebgl || !canvasPt) {
    lines[0] = '';
  }
  const renderer = (RUN.ptWebgl && canvasPt)
    ? new THREE.WebGLRenderer({ canvas: canvasPt, antialias: false, alpha: false })
    : null;
  if (renderer) {
    renderer.setPixelRatio(1);
    renderer.setClearColor(0x111111, 1);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
  }

  const ptEngine = renderer
    ? await createPTEngine_WebGL2({
        device: renderer,
        extensions: {
          'vitrum.ptWebgl.qualityMode': FLAGS.quality,
        },
      })
    : null;
  if (ptEngine) {
    ptEngine.setScene(vitrumScene);
    (globalThis as unknown as { __vitrumPtWebgl: typeof ptEngine }).__vitrumPtWebgl = ptEngine;
  }

  let ptFrame = 0;
  const samplesTarget = FLAGS.samplesTarget;

  function resizePt(): void {
    if (!renderer || !canvasPt) return;
    resizeCanvasToDisplaySize(canvasPt);
    camera.aspect = canvasPt.width / Math.max(canvasPt.height, 1);
    camera.updateProjectionMatrix();
    renderer.setSize(canvasPt.width, canvasPt.height, false);
  }

  resizePt();

  // Orbit step computed against motionStart so the path is deterministic
  // independent of frame-rate jitter.
  function applyCameraMotion(): void {
    if (!FLAGS.cameraMotion) return;
    const now = performance.now();
    if (motionStartTs === 0) motionStartTs = now;
    // 360° in 24 seconds ⇒ ~0.067 deg/ms. Slow enough to hit
    // CAMERA_MOVE_RESET_THRESHOLD_SQ every frame so we can observe the
    // walkaround temporal reset path.
    cameraAngle = ((now - motionStartTs) * 0.0667) % 360;
    const rad = (cameraAngle * Math.PI) / 180;
    const radius = 2.75;
    camera.position.set(Math.sin(rad) * radius, 0, Math.cos(rad) * radius);
    camera.lookAt(-0.05, -0.15, 0);
    telemetry.cameraMotion = { angleDeg: +cameraAngle.toFixed(2), moving: true };
  }

  function ptLoop(): void {
    if (!ptEngine || !canvasPt) return;
    resizePt();
    applyCameraMotion();
    camera.updateMatrixWorld();
    const input: FrameInput = {
      viewMatrix: mat4FromThree(camera.matrixWorldInverse),
      projMatrix: mat4FromThree(camera.projectionMatrix),
      cameraPosition: [camera.position.x, camera.position.y, camera.position.z],
      viewport: {
        width: canvasPt.width,
        height: canvasPt.height,
        devicePixelRatio: window.devicePixelRatio,
      },
      frameIndex: ptFrame,
      frameSeed: (ptFrame * 9973 + 12345) >>> 0,
      quality: {
        samplesTarget,
        bounces: 8,
        resolutionFactor: 1,
        filteredGlossyFactor: 0.5,
      },
    };

    const out = ptEngine.renderFrame(input);
    ptFrame++;
    telemetry.ptWebgl = {
      state: ptEngine.state,
      spp: out.samplesAccumulated,
      target: samplesTarget,
      converged: out.isConverged,
    };
    lines[0] = `PT: SPP ${out.samplesAccumulated}/${samplesTarget}${out.isConverged ? ' ✓' : ''}${FLAGS.cameraMotion ? ' · motion' : ''}`;
    refreshStatus();
    // Keep RAFing when cameraMotion is on; the fork resets accumulation on
    // camera changes internally so each frame redrives a fresh accumulator.
    if (!out.isConverged || FLAGS.cameraMotion) requestAnimationFrame(ptLoop);
  }

  if (ptEngine) requestAnimationFrame(ptLoop);

  // ── WebGPU walkaround (optional) ───────────────────────────────────────
  // Flatten the three nested guards into early returns inside an IIFE so the
  // happy path (configure → init engine → animation loop) lives at the
  // outer indentation level. The IIFE keeps the late-binding `lines[1]`
  // updates in scope.
  await (async (): Promise<void> => {
    if (!RUN.walkaround && !RUN.ptWebgpu) return;
    if (!canvasWgpu && !canvasPtGpu) return;
    if (!navigator.gpu) {
      lines[1] = 'Walkaround: no WebGPU — skipped (PT demonstrates shared Scene).';
      refreshStatus();
      return;
    }
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      lines[1] = 'Walkaround: no GPU adapter.';
      refreshStatus();
      return;
    }
    // The walkaround-hybrid shade pass binds 13+ storage buffers; default
    // WebGPU device limit is 8. Pass HYBRID_WEBGPU_REQUIRED_LIMITS so the
    // pipeline layouts validate. The pipeline itself requires no optional
    // features (all storage textures use base-spec formats). Opt into
    // 'timestamp-query' when the adapter exposes it so the engine can
    // record per-pass GPU timings for telemetry / dev-panel readback.
    const tsAvailable = adapter.features.has('timestamp-query');
    const requiredFeatures: GPUFeatureName[] = tsAvailable
      ? ['timestamp-query' as GPUFeatureName]
      : [];
    const device = await adapter.requestDevice({
      requiredLimits: HYBRID_WEBGPU_REQUIRED_LIMITS,
      ...(requiredFeatures.length > 0 ? { requiredFeatures } : {}),
    });
    const format = navigator.gpu.getPreferredCanvasFormat();
    const ctx = (RUN.walkaround && canvasWgpu) ? canvasWgpu.getContext('webgpu') : null;
    if (RUN.walkaround && canvasWgpu && !ctx) {
      lines[1] = 'Walkaround: getContext("webgpu") failed.';
      refreshStatus();
      return;
    }
    const gpuCtx = ctx;

    function configureWgpu(): void {
      if (!gpuCtx || !canvasWgpu) return;
      resizeCanvasToDisplaySize(canvasWgpu);
      gpuCtx.configure({
        device,
        format,
        alphaMode: 'premultiplied',
      });
    }
    configureWgpu();

    const raw: [number, number, number] = [0.35, 1, 0.2];
    const len = Math.hypot(raw[0], raw[1], raw[2]);
    const primaryLightDir: [number, number, number] = [raw[0] / len, raw[1] / len, raw[2] / len];

    // Expose the device + engine on window for the diagnostic probe.
    (globalThis as unknown as { __vitrumDevice: GPUDevice; __vitrumWalkaround?: unknown }).__vitrumDevice = device;
    const hybrid = (RUN.walkaround && canvasWgpu && gpuCtx)
      ? await createWalkaroundEngine_Hybrid({
          device,
          width: canvasWgpu.width,
          height: canvasWgpu.height,
          threeScene,
          primaryLightDir,
          // Cornell box is an indoor scene with one area light at the top.
          // Sun (`primaryLightIntensity`) and sky (`skyIrradiance`) belong
          // to the original stained-glass-studio context the engine was
          // built around; firing them for Cornell adds spurious extra
          // illumination (notably a bright peach stripe at the red-wall
          // edge from grazing-angle sun shading + skyAperture probes).
          primaryLightIntensity: 0,
          skyTint: [0.55, 0.72, 1.0],
          skyIrradiance: 0,
          isSceneReady: () => true,
          denoiser: FLAGS.denoiser,
          ppgEnabled: FLAGS.ppgEnabled,
        })
      : null;
    if (hybrid) {
      (globalThis as unknown as { __vitrumWalkaround: typeof hybrid }).__vitrumWalkaround = hybrid;
      hybrid.setScene(vitrumScene);
      // Scene lifecycle (#5) — expose a callable that rerun setScene with the
      // current vitrumScene. Cheap correctness probe: makes sure the engine
      // re-uploads BVH + reservoir state without leaking.
      (globalThis as unknown as { __vitrumResetScene: () => void }).__vitrumResetScene = () => {
        hybrid.setScene(vitrumScene);
      };

      const ready = await waitEngineReady(hybrid, 45_000);
      if (!ready) {
        lines[1] = `Walkaround: still ${hybrid.state} after timeout (try smaller scene or check GPU).`;
        refreshStatus();
        return;
      }
      lines[1] = 'Walkaround: rendering';
      refreshStatus();
    }

    if (hybrid && canvasWgpu && gpuCtx) {
      let wFrame = 0;
      // Track last-seen canvas dimensions so we only reconfigure on resize,
      // not every frame. Spam-reconfiguring `gpuCtx.configure(...)` every
      // frame reallocates the swap chain — if the GPU's composite write
      // lands after the new swap chain is acquired but before this frame's
      // composite runs, the canvas presents an uninitialised (black) image
      // for one frame, producing visible dark flashes.
      let lastW = canvasWgpu.width, lastH = canvasWgpu.height;
      function wgpuLoop(): void {
        if (!hybrid || !canvasWgpu || !gpuCtx) return;
        if (canvasWgpu.clientWidth !== lastW / window.devicePixelRatio ||
            canvasWgpu.clientHeight !== lastH / window.devicePixelRatio) {
          configureWgpu();
          lastW = canvasWgpu.width;
          lastH = canvasWgpu.height;
        }
        applyCameraMotion();
        camera.aspect = canvasWgpu.width / Math.max(canvasWgpu.height, 1);
        camera.updateProjectionMatrix();
        camera.updateMatrixWorld();
        const view = gpuCtx.getCurrentTexture().createView();
        const input: FrameInput = {
          viewMatrix: mat4FromThree(camera.matrixWorldInverse),
          projMatrix: mat4FromThree(camera.projectionMatrix),
          cameraPosition: [camera.position.x, camera.position.y, camera.position.z],
          viewport: {
            width: canvasWgpu.width,
            height: canvasWgpu.height,
            devicePixelRatio: window.devicePixelRatio,
          },
          frameIndex: wFrame,
          frameSeed: (wFrame * 1664525 + 1013904223) >>> 0,
          swapChainView: view,
          swapChainFormat: format,
          quality: { bounces: 4 },
        };
        hybrid.renderFrame(input);
        wFrame++;
        const hyb = hybrid as unknown as {
          debugTimings?: ReadonlyArray<{ t: number; ms: number }>;
          lastGpuTimings?: Record<string, number>;
          lastGpuTimingsFrame?: number;
        };
        telemetry.walkaround = {
          state: hybrid.state,
          frame: wFrame,
          debugTimingsLen: hyb.debugTimings?.length ?? 0,
          ...(hyb.lastGpuTimings ? { lastGpuTimings: hyb.lastGpuTimings } : {}),
          ...(hyb.lastGpuTimingsFrame != null ? { lastGpuTimingsFrame: hyb.lastGpuTimingsFrame } : {}),
        };
        pollMemory(wFrame);
        requestAnimationFrame(wgpuLoop);
      }
      requestAnimationFrame(wgpuLoop);

      window.addEventListener('resize', () => {
        configureWgpu();
      });
    }

    // ── pt-webgpu (headless validation drive) ──────────────────────────────
    // Shares the same GPUDevice as the walkaround pipeline. Runs in its own
    // RAF loop until samplesTarget is reached. The bottom canvas is left
    // unconfigured — pt-webgpu writes to its internal HDR accum and does
    // not present to a swap chain.
    if (!RUN.ptWebgpu || !canvasPtGpu) return;
    try {
      resizeCanvasToDisplaySize(canvasPtGpu);
      const ptGpuEngine = await createPTEngine_WebGPU({ device });
      ptGpuEngine.setScene(vitrumScene);
      (globalThis as unknown as { __vitrumPtWebgpu: typeof ptGpuEngine }).__vitrumPtWebgpu = ptGpuEngine;
      const ptGpuSamplesTarget = FLAGS.samplesTarget;
      let ptGpuFrame = 0;
      let lastCameraPos: [number, number, number] = [camera.position.x, camera.position.y, camera.position.z];
      function ptGpuLoop(): void {
        if (ptGpuEngine.state !== 'ready') {
          lines[2] = `pt-webgpu: state=${ptGpuEngine.state}`;
          refreshStatus();
          requestAnimationFrame(ptGpuLoop);
          return;
        }
        applyCameraMotion();
        camera.updateMatrixWorld();
        // Camera-move reset — pt-webgpu accumulates monotonically, so on
        // camera change we have to reset() to flush the accum.
        const dx = camera.position.x - lastCameraPos[0];
        const dy = camera.position.y - lastCameraPos[1];
        const dz = camera.position.z - lastCameraPos[2];
        if (dx * dx + dy * dy + dz * dz > 1e-6) {
          ptGpuEngine.reset();
          lastCameraPos = [camera.position.x, camera.position.y, camera.position.z];
        }
        const input: FrameInput = {
          viewMatrix: mat4FromThree(camera.matrixWorldInverse),
          projMatrix: mat4FromThree(camera.projectionMatrix),
          cameraPosition: [camera.position.x, camera.position.y, camera.position.z],
          viewport: {
            width: canvasPtGpu.width,
            height: canvasPtGpu.height,
            devicePixelRatio: window.devicePixelRatio,
          },
          frameIndex: ptGpuFrame,
          frameSeed: (ptGpuFrame * 6364136223846793005 + 1442695040888963407) >>> 0,
          quality: { samplesTarget: ptGpuSamplesTarget, bounces: FLAGS.ptWebgpuBounces, resolutionFactor: 1 },
        };
        const out = ptGpuEngine.renderFrame(input);
        ptGpuFrame++;
        telemetry.ptWebgpu = {
          state: ptGpuEngine.state,
          spp: out.samplesAccumulated,
          target: ptGpuSamplesTarget,
          converged: out.isConverged,
        };
        lines[2] = `pt-webgpu: SPP ${out.samplesAccumulated}/${ptGpuSamplesTarget}${out.isConverged ? ' ✓' : ''}${FLAGS.cameraMotion ? ' · motion' : ''}`;
        refreshStatus();
        if (!out.isConverged || FLAGS.cameraMotion) requestAnimationFrame(ptGpuLoop);
      }
      requestAnimationFrame(ptGpuLoop);
    } catch (e) {
      lines[2] = `pt-webgpu: init failed — ${String(e)}`;
      console.error('[two-engines] pt-webgpu init failed', e);
      refreshStatus();
    }
  })();

  if (renderer) window.addEventListener('resize', resizePt);
}

main().catch((e) => {
  console.error(e);
  const statusEl = document.querySelector('#status');
  if (statusEl) statusEl.textContent = String(e);
});
