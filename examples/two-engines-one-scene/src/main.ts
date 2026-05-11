/**
 * Gate G2: identical @vitrum/core Scene drives pt-webgl (WebGL2) and walkaround-hybrid (WebGPU).
 */

import type { Engine, FrameInput, Mat4, Scene } from '@vitrum/core';
import { buildCornellBoxThreeScene } from '@vitrum-examples/shared';
import { createPTEngine_WebGL2 } from '@vitrum/pt-webgl';
// pt-webgpu drives the bottom canvas headlessly (no swap-chain present;
// pt-webgpu accumulates to an internal HDR texture, leaving the host to
// implement display). The engine drive verifies WGSL params layout,
// scene buffer uploads, and the path-trace dispatch on real hardware.
import { createPTEngine_WebGPU } from '@vitrum/pt-webgpu';
import { sceneFromThreeJS } from '@vitrum/three-bindings';
import { createWalkaroundEngine_Hybrid, HYBRID_WEBGPU_REQUIRED_LIMITS } from '@vitrum/walkaround-hybrid';

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
  const canvasPt    = nonNull(document.querySelector<HTMLCanvasElement>('#c-pt'), '#c-pt');
  const canvasWgpu  = nonNull(document.querySelector<HTMLCanvasElement>('#c-wgpu'), '#c-wgpu');
  const canvasPtGpu = nonNull(document.querySelector<HTMLCanvasElement>('#c-ptgpu'), '#c-ptgpu');
  const statusEl    = nonNull(document.querySelector<HTMLDivElement>('#status'), '#status');

  const lines: [string, string, string] = ['', '', ''];

  // Expose engine telemetry as window.__vitrum so the validation harness
  // (driven via Claude-in-Chrome or DevTools) can poll without touching DOM.
  const telemetry: {
    ptWebgl?: { state: string; spp: number; target: number; converged: boolean };
    walkaround?: {
      state: string;
      frame: number;
      debugTimingsLen: number;
      lastGpuTimings?: Record<string, number>;
      lastGpuTimingsFrame?: number;
    };
    ptWebgpu?: { state: string; spp: number; target: number; converged: boolean };
  } = {};
  (globalThis as unknown as { __vitrum: typeof telemetry }).__vitrum = telemetry;

  function refreshStatus(): void {
    statusEl.textContent = lines.join('\n').trim() || '…';
  }

  const threeScene = buildCornellBoxThreeScene();
  const vitrumScene: Scene = sceneFromThreeJS(threeScene);

  const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 50);
  camera.position.set(-0.05, 0, 2.75);
  camera.lookAt(-0.05, -0.15, 0);

  // ── WebGL2 path trace ─────────────────────────────────────────────────
  const renderer = new THREE.WebGLRenderer({ canvas: canvasPt, antialias: false, alpha: false });
  renderer.setPixelRatio(1);
  renderer.setClearColor(0x111111, 1);
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  const ptEngine = await createPTEngine_WebGL2({ device: renderer });
  ptEngine.setScene(vitrumScene);

  let ptFrame = 0;
  const samplesTarget = 32;

  function resizePt(): void {
    resizeCanvasToDisplaySize(canvasPt);
    camera.aspect = canvasPt.width / Math.max(canvasPt.height, 1);
    camera.updateProjectionMatrix();
    renderer.setSize(canvasPt.width, canvasPt.height, false);
  }

  resizePt();

  function ptLoop(): void {
    resizePt();
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
    lines[0] = `PT: SPP ${out.samplesAccumulated}/${samplesTarget}${out.isConverged ? ' ✓' : ''}`;
    refreshStatus();
    if (!out.isConverged) requestAnimationFrame(ptLoop);
  }

  requestAnimationFrame(ptLoop);

  // ── WebGPU walkaround (optional) ───────────────────────────────────────
  // Flatten the three nested guards into early returns inside an IIFE so the
  // happy path (configure → init engine → animation loop) lives at the
  // outer indentation level. The IIFE keeps the late-binding `lines[1]`
  // updates in scope.
  await (async (): Promise<void> => {
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
    // pipeline layouts validate. Also opt into 'timestamp-query' when the
    // adapter exposes it so the engine can record per-pass GPU timings
    // for telemetry / dev-panel readback.
    const tsAvailable = adapter.features.has('timestamp-query');
    const device = await adapter.requestDevice({
      requiredLimits: HYBRID_WEBGPU_REQUIRED_LIMITS,
      ...(tsAvailable ? { requiredFeatures: ['timestamp-query' as GPUFeatureName] } : {}),
    });
    const format = navigator.gpu.getPreferredCanvasFormat();
    const ctx = canvasWgpu.getContext('webgpu');
    if (!ctx) {
      lines[1] = 'Walkaround: getContext("webgpu") failed.';
      refreshStatus();
      return;
    }
    const gpuCtx = ctx;

    function configureWgpu(): void {
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

    const hybrid = await createWalkaroundEngine_Hybrid({
      device,
      width: canvasWgpu.width,
      height: canvasWgpu.height,
      threeScene,
      primaryLightDir,
      primaryLightIntensity: 5,
      skyTint: [0.55, 0.72, 1.0],
      skyIrradiance: 0.35,
      isSceneReady: () => true,
    });
    hybrid.setScene(vitrumScene);

    const ready = await waitEngineReady(hybrid, 45_000);
    if (!ready) {
      lines[1] = `Walkaround: still ${hybrid.state} after timeout (try smaller scene or check GPU).`;
      refreshStatus();
      return;
    }
    lines[1] = 'Walkaround: rendering';
    refreshStatus();

    let wFrame = 0;
    function wgpuLoop(): void {
      configureWgpu();
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
      requestAnimationFrame(wgpuLoop);
    }
    requestAnimationFrame(wgpuLoop);

    window.addEventListener('resize', () => {
      configureWgpu();
    });

    // ── pt-webgpu (headless validation drive) ──────────────────────────────
    // Shares the same GPUDevice as the walkaround pipeline. Runs in its own
    // RAF loop until samplesTarget is reached. The bottom canvas is left
    // unconfigured — pt-webgpu writes to its internal HDR accum and does
    // not present to a swap chain.
    try {
      resizeCanvasToDisplaySize(canvasPtGpu);
      const ptGpuEngine = await createPTEngine_WebGPU({ device });
      ptGpuEngine.setScene(vitrumScene);
      const ptGpuSamplesTarget = 32;
      let ptGpuFrame = 0;
      function ptGpuLoop(): void {
        if (ptGpuEngine.state !== 'ready') {
          lines[2] = `pt-webgpu: state=${ptGpuEngine.state}`;
          refreshStatus();
          requestAnimationFrame(ptGpuLoop);
          return;
        }
        camera.updateMatrixWorld();
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
          quality: { samplesTarget: ptGpuSamplesTarget, bounces: 4, resolutionFactor: 1 },
        };
        const out = ptGpuEngine.renderFrame(input);
        ptGpuFrame++;
        telemetry.ptWebgpu = {
          state: ptGpuEngine.state,
          spp: out.samplesAccumulated,
          target: ptGpuSamplesTarget,
          converged: out.isConverged,
        };
        lines[2] = `pt-webgpu: SPP ${out.samplesAccumulated}/${ptGpuSamplesTarget}${out.isConverged ? ' ✓' : ''}`;
        refreshStatus();
        if (!out.isConverged) requestAnimationFrame(ptGpuLoop);
      }
      requestAnimationFrame(ptGpuLoop);
    } catch (e) {
      lines[2] = `pt-webgpu: init failed — ${String(e)}`;
      console.error('[two-engines] pt-webgpu init failed', e);
      refreshStatus();
    }
  })();

  window.addEventListener('resize', resizePt);
}

main().catch((e) => {
  console.error(e);
  const statusEl = document.querySelector('#status');
  if (statusEl) statusEl.textContent = String(e);
});
