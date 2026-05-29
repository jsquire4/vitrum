/**
 * Gate G2: identical @vitrum/core Scene drives pt-webgl (WebGL2) and walkaround-hybrid (WebGPU).
 */

import {
  asBackendTexture,
  asBackendTextureFormat,
  type Engine,
  type FrameInput,
  type Scene,
} from '@vitrum/core';
import {
  buildBenchmark200kThreeScene,
  buildComplexThreeScene,
  buildCornellBoxThreeScene,
  buildGapClosureCornellThreeScene,
  buildTlas10InstThreeScene,
  defaultCausticForGapScenario,
  mat4FromThree,
  spectralForGapScenario,
  resizeCanvasToDisplaySize,
} from '@vitrum-examples/shared';
import { createPTEngine_WebGL2 } from '@vitrum/pt-webgl';
// pt-webgpu drives the bottom canvas headlessly (no swap-chain present;
// pt-webgpu accumulates to an internal HDR texture, leaving the host to
// implement display). The engine drive verifies WGSL params layout,
// scene buffer uploads, and the path-trace dispatch on real hardware.
import {
  createPTEngine_WebGPU,
  mergeAdapterRequiredLimits,
  ptWebgpuRequiredLimitsForAdapter,
} from '@vitrum/pt-webgpu';
import { sceneFromThreeJS } from '@vitrum/three-bindings';
import {
  createWalkaroundEngine_Hybrid,
  HYBRID_WEBGPU_REQUIRED_LIMITS,
  type HybridEngine,
} from '@vitrum/walkaround-hybrid';
import {
  installHybridSoakApi,
  maybeAutoRunHybridSoak,
} from './hybridSoakHarness.js';
import {
  installPrBenchApi,
  maybeAutoRunPrBench,
  type PrBenchMode,
} from './prBenchHarness.js';

if (typeof console !== 'undefined' && console.debug) {
  console.debug('[two-engines] pt-webgpu factory loaded:', typeof createPTEngine_WebGPU);
}
import * as THREE from 'three';

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
  //   ?denoiser=atrous           — use legacy 3-iter atrous instead of atrous-variance (#7)
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
    denoiser: (params.get('denoiser') ?? 'atrous-variance') as 'atrous-variance' | 'atrous',
    quality: (params.get('quality')
      ?? (walkaroundFallback ? 'interactive' : 'capture')) as 'interactive' | 'final' | 'capture' | 'safe',
    ptWebgpuBounces: parseInt(params.get('ptWebgpuBounces') ?? '4', 10) || 4,
    samplesTarget: parseInt(params.get('samplesTarget')
      ?? (walkaroundFallback ? '256' : '32'), 10) || 32,
    scene: (params.get('scene') ?? 'cornell') as
      | 'cornell'
      | 'complex'
      | 'bench200k'
      | 'tlas10inst',
    targetTriangles: parseInt(params.get('targetTriangles') ?? '200000', 10) || 200_000,
    prBenchScenario: params.get('prBenchScenario') ?? '',
    vitrumSeed: parseInt(params.get('vitrumSeed') ?? '0', 10) || 0,
    vitrumGapScenario: params.get('vitrumGapScenario') ?? '',
    vitrumCaustic: (params.get('vitrumCaustic') ?? '') as
      | ''
      | 'none'
      | 'manifold-nee'
      | 'photon-map',
    vitrumPtWebgpuSpectral: params.get('vitrumPtWebgpuSpectral') === '1',
    bvhMode: (params.get('bvhMode') ?? '') as '' | 'merged' | 'tlas',
    rcEnabled: params.get('rcEnabled') === '1',
    rcWeight: Math.max(0, Math.min(1, parseFloat(params.get('rcWeight') ?? '1') || 1)),
    prBench: (params.get('prBench') ?? null) as PrBenchMode | null,
    prBenchIters: parseInt(params.get('prBenchIters') ?? '100', 10) || 100,
    prBenchFrames: parseInt(params.get('prBenchFrames') ?? '120', 10) || 120,
    prBenchAuto: params.get('prBenchAuto') === '1',
    hybridSoakAuto: params.get('hybridSoakAuto') === '1',
    hybridSoakFrames: parseInt(params.get('hybridSoakFrames') ?? '120', 10) || 120,
    hybridSoakMaterialEvery: parseInt(params.get('hybridSoakMaterialEvery') ?? '10', 10) || 10,
    hybridSoakEmitterEvery: parseInt(params.get('hybridSoakEmitterEvery') ?? '0', 10) || 0,
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

  const threeScene =
    FLAGS.vitrumGapScenario.length > 0
      ? buildGapClosureCornellThreeScene(FLAGS.vitrumGapScenario)
      : FLAGS.scene === 'bench200k'
        ? buildBenchmark200kThreeScene(FLAGS.targetTriangles)
        : FLAGS.scene === 'tlas10inst'
          ? buildTlas10InstThreeScene()
          : FLAGS.scene === 'complex'
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
      frameSeed: (FLAGS.vitrumSeed + ptFrame * 9973 + 12345) >>> 0,
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
    lines[1] = 'WebGPU: requesting adapter…';
    refreshStatus();
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
    const deviceRequiredLimits = RUN.walkaround
      ? HYBRID_WEBGPU_REQUIRED_LIMITS
      : ptWebgpuRequiredLimitsForAdapter(adapter);
    let device: GPUDevice;
    try {
      device = await adapter.requestDevice({
        requiredLimits: mergeAdapterRequiredLimits(adapter, deviceRequiredLimits),
        ...(requiredFeatures.length > 0 ? { requiredFeatures } : {}),
      });
    } catch (deviceError) {
      const msg = deviceError instanceof Error ? deviceError.message : String(deviceError);
      lines[1] = `WebGPU device: ${msg}`;
      refreshStatus();
      return;
    }
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
          ...(FLAGS.bvhMode !== ''
            ? { extensions: { 'walkaround-hybrid': { bvhMode: FLAGS.bvhMode } } }
            : {}),
          ...(FLAGS.rcEnabled
            ? { rcEnabled: true, rcWeight: FLAGS.rcWeight }
            : {}),
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
      telemetry.walkaround = {
        state: hybrid.state,
        frame: 0,
        debugTimingsLen: 0,
      };
      lines[1] = 'Walkaround: rendering';
      refreshStatus();

      const hybridEngine = hybrid as HybridEngine;
      const prBenchApi = installPrBenchApi(hybridEngine, vitrumScene, {
        prBench: FLAGS.prBench,
        prBenchIters: FLAGS.prBenchIters,
        prBenchFrames: FLAGS.prBenchFrames,
        prBenchAuto: FLAGS.prBenchAuto,
        ...(FLAGS.prBenchScenario ? { prBenchScenario: FLAGS.prBenchScenario } : {}),
      });
      const hybridSoakApi = installHybridSoakApi(hybridEngine, vitrumScene);
      if (FLAGS.prBenchAuto && FLAGS.prBench != null) {
        void maybeAutoRunPrBench(hybridEngine, vitrumScene, {
          prBench: FLAGS.prBench,
          prBenchIters: FLAGS.prBenchIters,
          prBenchFrames: FLAGS.prBenchFrames,
          prBenchAuto: true,
          ...(FLAGS.prBenchScenario ? { prBenchScenario: FLAGS.prBenchScenario } : {}),
        }, prBenchApi);
      }
      if (FLAGS.hybridSoakAuto) {
        void maybeAutoRunHybridSoak(hybridEngine, vitrumScene, {
          hybridSoakAuto: true,
          hybridSoakFrames: FLAGS.hybridSoakFrames,
          hybridSoakMaterialEvery: FLAGS.hybridSoakMaterialEvery,
          hybridSoakEmitterEvery: FLAGS.hybridSoakEmitterEvery,
        }, hybridSoakApi);
      }
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
          frameSeed: (FLAGS.vitrumSeed + wFrame * 1664525 + 1013904223) >>> 0,
          swapChainView: asBackendTexture<'webgpu', GPUTextureView>(view),
          swapChainFormat: asBackendTextureFormat<'webgpu', GPUTextureFormat>(format),
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
    // Re-bind to a non-nullable local so the rAF closure below inherits the
    // narrowed type rather than `HTMLCanvasElement | null` from the outer scope.
    const ptGpuCanvas: HTMLCanvasElement = canvasPtGpu;
    try {
      const resizePtGpu = (): void => {
        resizeCanvasToDisplaySize(ptGpuCanvas);
      };
      resizePtGpu();
      const gapId = FLAGS.vitrumGapScenario;
      const causticStrategy =
        FLAGS.vitrumCaustic !== ''
          ? FLAGS.vitrumCaustic
          : defaultCausticForGapScenario(gapId, null);
      const spectral =
        FLAGS.vitrumPtWebgpuSpectral || gapId.includes('spectral') || spectralForGapScenario(gapId);
      const ptGpuEngine = await createPTEngine_WebGPU({
        device,
        causticStrategy,
        spectral,
      });
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
            width: ptGpuCanvas.width,
            height: ptGpuCanvas.height,
            devicePixelRatio: window.devicePixelRatio,
          },
          frameIndex: ptGpuFrame,
          frameSeed: (FLAGS.vitrumSeed + ptGpuFrame * 6364136223846793005 + 1442695040888963407) >>> 0,
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
      window.addEventListener('resize', resizePtGpu);
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
