/**
 * Gate G2: identical @vitrum/core Scene drives pt-webgl (WebGL2) and walkaround-hybrid (WebGPU).
 */

import type { Engine, FrameInput, Mat4, Scene } from '@vitrum/core';
import { buildCornellBoxThreeScene } from '@vitrum-examples/shared';
import { createPTEngine_WebGL2 } from '@vitrum/pt-webgl';
import { sceneFromThreeJS } from '@vitrum/three-bindings';
import { createWalkaroundEngine_Hybrid } from '@vitrum/walkaround-hybrid';
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

async function main(): Promise<void> {
  const canvasPt = document.querySelector<HTMLCanvasElement>('#c-pt');
  const canvasWgpu = document.querySelector<HTMLCanvasElement>('#c-wgpu');
  const statusEl = document.querySelector<HTMLDivElement>('#status');
  if (!canvasPt || !canvasWgpu || !statusEl) throw new Error('missing DOM nodes');

  const ptCanvas = canvasPt;
  const wgpuCanvas = canvasWgpu;
  const st = statusEl;

  const lines: [string, string] = ['', ''];

  function refreshStatus(): void {
    st.textContent = lines.join('\n').trim() || '…';
  }

  const threeScene = buildCornellBoxThreeScene();
  const vitrumScene: Scene = sceneFromThreeJS(threeScene);

  const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 50);
  camera.position.set(-0.05, 0, 2.75);
  camera.lookAt(-0.05, -0.15, 0);

  // ── WebGL2 path trace ─────────────────────────────────────────────────
  const renderer = new THREE.WebGLRenderer({ canvas: ptCanvas, antialias: false, alpha: false });
  renderer.setPixelRatio(1);
  renderer.setClearColor(0x111111, 1);
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  const ptEngine = await createPTEngine_WebGL2({ device: renderer });
  ptEngine.setScene(vitrumScene);

  let ptFrame = 0;
  const samplesTarget = 32;

  function resizePt(): void {
    resizeCanvasToDisplaySize(ptCanvas);
    camera.aspect = ptCanvas.width / Math.max(ptCanvas.height, 1);
    camera.updateProjectionMatrix();
    renderer.setSize(ptCanvas.width, ptCanvas.height, false);
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
        width: ptCanvas.width,
        height: ptCanvas.height,
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
    lines[0] = `PT: SPP ${out.samplesAccumulated}/${samplesTarget}${out.isConverged ? ' ✓' : ''}`;
    refreshStatus();
    if (!out.isConverged) requestAnimationFrame(ptLoop);
  }

  requestAnimationFrame(ptLoop);

  // ── WebGPU walkaround (optional) ───────────────────────────────────────
  if (!navigator.gpu) {
    lines[1] = 'Walkaround: no WebGPU — skipped (PT demonstrates shared Scene).';
    refreshStatus();
  } else {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      lines[1] = 'Walkaround: no GPU adapter.';
      refreshStatus();
    } else {
      const device = await adapter.requestDevice();
      const format = navigator.gpu.getPreferredCanvasFormat();
      const ctx = wgpuCanvas.getContext('webgpu');
      if (!ctx) {
        lines[1] = 'Walkaround: getContext("webgpu") failed.';
        refreshStatus();
      } else {
        const gpuCtx = ctx;
        function configureWgpu(): void {
          resizeCanvasToDisplaySize(wgpuCanvas);
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
          width: wgpuCanvas.width,
          height: wgpuCanvas.height,
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
        } else {
          lines[1] = 'Walkaround: rendering';
          refreshStatus();
          let wFrame = 0;
          function wgpuLoop(): void {
            configureWgpu();
            camera.aspect = wgpuCanvas.width / Math.max(wgpuCanvas.height, 1);
            camera.updateProjectionMatrix();
            camera.updateMatrixWorld();
            const view = gpuCtx.getCurrentTexture().createView();
            const input: FrameInput = {
              viewMatrix: mat4FromThree(camera.matrixWorldInverse),
              projMatrix: mat4FromThree(camera.projectionMatrix),
              cameraPosition: [camera.position.x, camera.position.y, camera.position.z],
              viewport: {
                width: wgpuCanvas.width,
                height: wgpuCanvas.height,
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
            requestAnimationFrame(wgpuLoop);
          }
          requestAnimationFrame(wgpuLoop);
        }

        window.addEventListener('resize', () => {
          configureWgpu();
        });
      }
    }
  }

  window.addEventListener('resize', resizePt);
}

main().catch((e) => {
  console.error(e);
  const statusEl = document.querySelector('#status');
  if (statusEl) statusEl.textContent = String(e);
});
