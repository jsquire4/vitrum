// Shared WebGPU canvas-context configuration.
//
// Both the walkaround-hybrid and pt-webgpu constructors in createEngine need to
// configure the canvas's WebGPU context so the attachVitrum RAF tick can acquire
// a fresh GPUTextureView per frame and pass it as FrameInput.swapChainView.
// HybridEngine.renderFrame skips the WebGPU path when input.swapChainView is
// undefined; without this configure step a host using attachVitrum() against a
// WebGPU backend gets a black canvas. We configure here (not in attachVitrum)
// because createEngine owns the GPUDevice handle.

/**
 * Configure the canvas's WebGPU context with the given device.
 *
 * Best-effort: hosts that pre-configure their own context (or use a headless
 * test canvas with no getContext('webgpu') support) are fine — attachVitrum
 * will simply not plumb swapChainView and the engine will skip frames cleanly.
 *
 * Uses OPAQUE compositing — matches the engines' resolve pass (which writes RGB
 * with alpha = 1.0). PREMULTIPLIED would double-composite the canvas over the
 * page background.
 */
export function configureWebGpuCanvas(
  canvas: HTMLCanvasElement,
  device: GPUDevice,
  onError?: (error: unknown) => void,
): void {
  try {
    const ctx = canvas.getContext('webgpu');
    if (ctx != null) {
      const format = (typeof navigator !== 'undefined' && 'gpu' in navigator
        ? (navigator.gpu as { getPreferredCanvasFormat?: () => GPUTextureFormat })
            .getPreferredCanvasFormat?.() ?? ('bgra8unorm')
        : ('bgra8unorm' as GPUTextureFormat));
      ctx.configure({
        device,
        format,
        alphaMode: 'opaque',
      });
    }
  } catch (err) {
    // Best-effort canvas configure for attachVitrum swap-chain plumbing.
    try {
      onError?.(err);
    } catch {
      // Host error callbacks must not break best-effort canvas configuration.
    }
  }
}
