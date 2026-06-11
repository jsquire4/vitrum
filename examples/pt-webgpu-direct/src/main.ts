/**
 * pt-webgpu-direct — negotiateWebGPUDevice + createPTEngine_WebGPU.
 *
 * Demonstrates the backend-direct WebGPU usage pattern:
 *   1. Acquire a host-owned GPUDevice via negotiateWebGPUDevice().
 *   2. Configure the canvas's WebGPU swap chain manually.
 *   3. Pass the device to createPTEngine_WebGPU().
 *   4. Run the render loop, acquiring a fresh GPUTextureView each frame
 *      via canvas.getContext('webgpu').getCurrentTexture().createView()
 *      and passing it as FrameInput.swapChainView.
 *   5. Destroy the device on dispose (host owns lifecycle).
 *
 * The factory signature:
 *   createPTEngine_WebGPU(opts: PTEngineWebGPUOptions): Promise<Engine & PTEngineWebGPUSurface>
 *
 * Required opts: device (GPUDevice).
 * Optional: maxBounces, maxSamplesPerPixel, spectral, bdpt, causticStrategy.
 *
 * Capture protocol: sets VITRUM_CAPTURE_READY after targetSpp samples.
 *
 * API sharp edges observed while writing this example:
 * - swapChainView MUST be a fresh per-frame GPUTextureView from
 *   getCurrentTexture().createView(). Caching it across frames is a WebGPU
 *   spec violation and produces a validation error.
 * - The engine does NOT configure the WebGPU canvas context — the host must
 *   call ctx.configure({ device, format }) before the first renderFrame().
 *   negotiateWebGPUDevice() returns the preferred format for this purpose.
 * - FrameInput.swapChainView must be provided every frame (via asBackendTexture).
 *   If undefined, the engine returns kind:'skipped' without rendering.
 * - FrameStats (frameTimeMs, spp) come from engine.onFrame, NOT from the
 *   renderFrame() return value. renderFrame returns samplesAccumulated only.
 * - Swap-chain resizes require ctx.configure() re-issued with the new canvas
 *   size after changing canvas.width / canvas.height.
 * - FrameInput.swapChainFormat expects BackendTextureFormat (branded), not a
 *   bare GPUTextureFormat string. Wrap via asBackendTextureFormat<'webgpu', GPUTextureFormat>(format).
 * - TypeScript doesn't narrow a const through nested function closures for the
 *   canvas context null check. Rebind to a new const after the guard.
 */

import { negotiateWebGPUDevice } from '@vitrum/engine';
import { createPTEngine_WebGPU } from '@vitrum/pt-webgpu';
import type { FrameStats } from '@vitrum/core';
import { asMat4, asBackendTexture, asBackendTextureFormat } from '@vitrum/core';
import { createCornellScene } from '@vitrum-examples/cornell-scene';

// ── URL params ────────────────────────────────────────────────────────────────
const params = new URLSearchParams(location.search);
const targetSpp  = Number(params.get('vitrumSpp'))    || 128;
const maxBounces = Number(params.get('vitrumBounces')) || 8;

// ── Canvas ────────────────────────────────────────────────────────────────────
const canvas   = document.getElementById('vitrum-canvas') as HTMLCanvasElement;
const sppLabel = document.getElementById('spp') as HTMLDivElement;

// ── Camera (static Cornell-box view) ──────────────────────────────────────────
const viewMatrix = asMat4(new Float32Array([
  1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 1, 0,
  0,-1,-4, 1,
]));

const aspect = canvas.clientWidth / Math.max(canvas.clientHeight, 1);
const fovY   = Math.PI / 3;
const near   = 0.1;
const far    = 100;
const f = 1 / Math.tan(fovY / 2);
const projMatrix = asMat4(new Float32Array([
  f / aspect, 0, 0,                               0,
  0,          f, 0,                               0,
  0,          0, (far + near) / (near - far),   -1,
  0,          0, (2 * far * near) / (near - far), 0,
]));

// ── Scene ─────────────────────────────────────────────────────────────────────
const scene = createCornellScene();

// ── Engine ────────────────────────────────────────────────────────────────────
async function main(): Promise<void> {
  // 1. Acquire HOST-OWNED device (host must device.destroy() on cleanup).
  let negotiated: Awaited<ReturnType<typeof negotiateWebGPUDevice>>;
  try {
    negotiated = await negotiateWebGPUDevice({ target: 'pt-webgpu' });
  } catch (err) {
    console.error('[pt-webgpu-direct] WebGPU unavailable:', err);
    sppLabel.textContent = 'WebGPU unavailable';
    return;
  }
  const { device, format } = negotiated;

  // 2. Configure the WebGPU canvas context (host responsibility — the engine
  //    does NOT call ctx.configure()).
  // Non-null assertion after guard: TypeScript cannot narrow through RAF closures.
  const ctx = canvas.getContext('webgpu');
  if (ctx == null) {
    sppLabel.textContent = 'Canvas getContext("webgpu") returned null';
    device.destroy();
    return;
  }
   
  const gpuCtx = ctx!;
  canvas.width  = Math.max(1, canvas.clientWidth);
  canvas.height = Math.max(1, canvas.clientHeight);
  ctx.configure({ device, format, alphaMode: 'opaque' });

  // 3. Build the engine on the host-owned device.
  const engine = await createPTEngine_WebGPU({
    device,
    maxBounces,
    maxSamplesPerPixel: targetSpp,
  });

  await engine.setScene(scene);

  let frameIndex       = 0;
  let captureSignalled = false;

  // FrameStats (spp, frameTimeMs) come from engine.onFrame, not renderFrame().
  engine.onFrame?.((stats: FrameStats) => {
    const spp = stats.spp ?? 0;
    sppLabel.textContent = `spp: ${spp}`;
    (globalThis as Record<string, unknown>).VITRUM_MS_PER_SAMPLE =
      stats.frameTimeMs > 0 && spp > 0 ? stats.frameTimeMs / spp : 0;
  });

  // 4. Render loop — acquire a fresh swap-chain view each frame.
  // Use gpuCtx (the post-null-check alias) so TypeScript doesn't widen back to
  // GPUCanvasContext | null inside the nested tick closure.
  function tick(): void {
    const width  = Math.max(1, canvas.clientWidth);
    const height = Math.max(1, canvas.clientHeight);

    // Resize: re-configure swap chain when dimensions change.
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width  = width;
      canvas.height = height;
      gpuCtx.configure({ device, format, alphaMode: 'opaque' });
    }

    // MUST call getCurrentTexture() inside the RAF tick — WebGPU spec requirement.
    let swapChainView: GPUTextureView;
    try {
      swapChainView = gpuCtx.getCurrentTexture().createView();
    } catch {
      requestAnimationFrame(tick);
      return;
    }

    const output = engine.renderFrame({
      viewMatrix,
      projMatrix,
      cameraPosition: [0, -1, -4],
      viewport:  { width, height, devicePixelRatio: 1 },
      frameIndex,
      frameSeed: (frameIndex * 1664525 + 1013904223) >>> 0,
      swapChainView: asBackendTexture(swapChainView),
      // swapChainFormat must be a branded BackendTextureFormat, not a bare string.
      swapChainFormat: asBackendTextureFormat<'webgpu', GPUTextureFormat>(format),
    });

    if (output.kind === 'rendered') {
      frameIndex++;
      const spp = output.samplesAccumulated;
      if (!captureSignalled && spp >= targetSpp) {
        (globalThis as Record<string, unknown>).VITRUM_CAPTURE_READY = true;
        captureSignalled = true;
      }
    }

    requestAnimationFrame(tick);
  }

  requestAnimationFrame(tick);

  // 5. Cleanup (exposed for DevTools / E2E tests).
  (globalThis as Record<string, unknown>).VITRUM_DISPOSE = () => {
    engine.dispose();
    device.destroy();   // host destroys the device after engine disposal
  };
}

main().catch((err: unknown) => {
  console.error('[pt-webgpu-direct] fatal:', err);
  sppLabel.textContent = `error: ${String(err)}`;
});
