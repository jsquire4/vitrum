/**
 * pt-webgpu-direct — negotiateWebGPUDevice + createPTEngine_WebGPU.
 *
 * Demonstrates the backend-direct WebGPU usage pattern:
 *   1. Acquire a host-owned GPUDevice via negotiateWebGPUDevice().
 *   2. Acquire the canvas context and construct an offscreen presenter.
 *   3. Pass the device to createPTEngine_WebGPU().
 *   4. Run the render loop and blit the backend's offscreen presentation
 *      texture with createOffscreenPresenter().
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
 * - pt-webgpu is an offscreen-texture backend. renderFrame() does not present
 *   to the canvas; the host must blit engine.getPresentationSource().texture.
 * - createOffscreenPresenter() configures the canvas context and owns only its
 *   blit resources. The host still owns and eventually destroys the GPUDevice.
 * - FrameStats (frameTimeMs, spp) come from engine.onFrame, NOT from the
 *   renderFrame() return value. renderFrame returns samplesAccumulated only.
 * - Canvas resizes require both physical backing-store synchronization and an
 *   engine.setSize() call for size-dependent offscreen resources.
 * - TypeScript doesn't narrow a const through nested function closures for the
 *   canvas context null check. Rebind to a new const after the guard.
 */

import { createOffscreenPresenter, negotiateWebGPUDevice } from '@vitrum/engine';
import { createPTEngine_WebGPU } from '@vitrum/pt-webgpu';
import type { FrameStats, ProgressStats } from '@vitrum/core';
import { asMat4 } from '@vitrum/core';
import { createCornellScene } from '@vitrum-examples/cornell-scene';
import {
  CORNELL_CAMERA_POSITION,
  createAxisAlignedView,
  createPerspectiveProjection,
  syncCanvasToDisplaySize,
  writePerspectiveProjection,
} from '../../shared/exampleHost.js';

// ── URL params ────────────────────────────────────────────────────────────────
const params = new URLSearchParams(location.search);
const targetSpp  = Number(params.get('vitrumSpp'))    || 128;
const maxBounces = Number(params.get('vitrumBounces')) || 8;

// ── Canvas ────────────────────────────────────────────────────────────────────
const canvas   = document.getElementById('vitrum-canvas') as HTMLCanvasElement;
const sppLabel = document.getElementById('spp') as HTMLDivElement;
const initialViewport = syncCanvasToDisplaySize(canvas);

// ── Camera (static Cornell-box view) ──────────────────────────────────────────
const viewMatrix = asMat4(createAxisAlignedView(CORNELL_CAMERA_POSITION));
const projMatrix = asMat4(
  createPerspectiveProjection(initialViewport.width, initialViewport.height),
);

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
    (globalThis as Record<string, unknown>).VITRUM_CAPTURE_ERROR = String(
      err instanceof Error ? err.stack ?? err.message : err,
    );
    return;
  }
  const { device, format } = negotiated;

  // 2. Acquire the WebGPU canvas context. The presenter configures it; the
  //    path-tracing engine itself never owns or configures this context.
  // Non-null assertion after guard: TypeScript cannot narrow through RAF closures.
  const ctx = canvas.getContext('webgpu');
  if (ctx == null) {
    sppLabel.textContent = 'Canvas getContext("webgpu") returned null';
    (globalThis as Record<string, unknown>).VITRUM_CAPTURE_ERROR =
      'Canvas getContext("webgpu") returned null';
    device.destroy();
    return;
  }
  const gpuCtx = ctx!;

  // 3. Build the engine on the host-owned device.
  let partialPresenter: ReturnType<typeof createOffscreenPresenter> | undefined;
  let partialEngine: Awaited<ReturnType<typeof createPTEngine_WebGPU>> | undefined;
  try {
    partialPresenter = createOffscreenPresenter({ device, context: gpuCtx, format });
    partialEngine = await createPTEngine_WebGPU({
      device,
      maxBounces,
      maxSamplesPerPixel: targetSpp,
    });
    await partialEngine.setScene(scene);
    partialEngine.setSize?.(initialViewport.width, initialViewport.height);
  } catch (err) {
    try { partialEngine?.dispose(); } catch { /* initialization rollback is best-effort */ }
    try { partialPresenter?.dispose(); } catch { /* initialization rollback is best-effort */ }
    device.destroy();
    throw err;
  }
  const engine = partialEngine;
  const presenter = partialPresenter;
  if (engine == null || presenter == null) {
    device.destroy();
    throw new Error('[pt-webgpu-direct] initialization completed without engine resources');
  }

  let frameIndex       = 0;
  let captureSignalled = false;
  let disposed          = false;
  let rafHandle         = 0;

  const dispose = (): void => {
    if (disposed) return;
    disposed = true;
    if (rafHandle !== 0) {
      cancelAnimationFrame(rafHandle);
      rafHandle = 0;
    }
    try {
      presenter.dispose();
    } finally {
      try {
        engine.dispose();
      } finally {
        device.destroy(); // host destroys the device after engine disposal
      }
    }
  };
  (globalThis as Record<string, unknown>).VITRUM_DISPOSE = dispose;

  // FrameStats (spp, frameTimeMs) come from engine.onFrame, not renderFrame().
  engine.onFrame?.((stats: FrameStats) => {
    const spp = stats.spp ?? 0;
    (globalThis as Record<string, unknown>).VITRUM_MS_PER_SAMPLE =
      stats.frameTimeMs > 0 && spp > 0 ? stats.frameTimeMs / spp : 0;
  });
  engine.onProgress?.((progress: ProgressStats) => {
    if (progress.kind === 'pt-spp') sppLabel.textContent = `spp: ${progress.current}`;
  });

  const reportTickError = (error: unknown): void => {
    console.error('[pt-webgpu-direct] render loop failed:', error);
    sppLabel.textContent = `error: ${String(error)}`;
    (globalThis as Record<string, unknown>).VITRUM_CAPTURE_ERROR = String(
      error instanceof Error ? error.stack ?? error.message : error,
    );
    dispose();
  };

  const requestNextFrame = (): void => {
    if (disposed) return;
    rafHandle = requestAnimationFrame(() => {
      rafHandle = 0;
      try {
        tick();
      } catch (error) {
        reportTickError(error);
      }
    });
  };

  // 4. Render loop — render offscreen, then blit the presentation source.
  // Use gpuCtx (the post-null-check alias) so TypeScript doesn't widen back to
  // GPUCanvasContext | null inside the nested tick closure.
  function tick(): void {
    if (disposed) return;
    const viewport = syncCanvasToDisplaySize(canvas);
    const { width, height, devicePixelRatio } = viewport;
    writePerspectiveProjection(projMatrix, width, height);

    // Resize: re-configure swap chain when dimensions change.
    if (viewport.resized) {
      gpuCtx.configure({ device, format, alphaMode: 'opaque' });
      engine.setSize?.(width, height);
    }

    const output = engine.renderFrame({
      viewMatrix,
      projMatrix,
      cameraPosition: CORNELL_CAMERA_POSITION,
      viewport: { width, height, devicePixelRatio },
      quality: { samplesTarget: targetSpp, bounces: maxBounces },
      frameIndex,
      frameSeed: (frameIndex * 1664525 + 1013904223) >>> 0,
    });

    if (output.kind === 'rendered') {
      const source = engine.getPresentationSource?.();
      if (source == null || source.device !== device) {
        throw new Error('[pt-webgpu-direct] rendered frame has no device-local presentation source');
      }
      presenter.present(source.texture as unknown as GPUTexture);
      frameIndex++;
      const spp = output.samplesAccumulated;
      if (!captureSignalled && spp >= targetSpp) {
        (globalThis as Record<string, unknown>).VITRUM_CAPTURE_READY = true;
        captureSignalled = true;
      }
    }

    requestNextFrame();
  }

  requestNextFrame();
}

main().catch((err: unknown) => {
  console.error('[pt-webgpu-direct] fatal:', err);
  sppLabel.textContent = `error: ${String(err)}`;
  (globalThis as Record<string, unknown>).VITRUM_CAPTURE_ERROR = String(
    err instanceof Error ? err.stack ?? err.message : err,
  );
});
