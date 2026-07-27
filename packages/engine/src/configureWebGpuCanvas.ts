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
 * Best-effort by default: hosts that pre-configure their own context (or use a
 * headless test canvas with no getContext('webgpu') support) are fine —
 * attachVitrum will simply not plumb swapChainView and OFFSCREEN backends
 * (pt-webgpu) skip frames cleanly.
 *
 * V1-7 — the "skip frames cleanly" contract is ONLY valid for offscreen
 * backends. A swap-chain-required backend (walkaround-hybrid) that fails to
 * configure its context has no `swapChainView` for every subsequent frame and
 * renders permanently black. For that path the caller passes `required: true`,
 * which re-throws the configure failure so createEngine fails fast (a
 * recoverable auto-recreate / a clear construction error) instead of silently
 * producing a black canvas forever. `onError` is still invoked first so the
 * host observes the underlying failure either way.
 *
 * Uses OPAQUE compositing — matches the engines' resolve pass (which writes RGB
 * with alpha = 1.0). PREMULTIPLIED would double-composite the canvas over the
 * page background.
 */
export interface ConfigureWebGpuCanvasOptions {
  /** When true, a configure failure is re-thrown after `onError` runs, so a
   *  swap-chain-required backend fails fast instead of rendering a permanently
   *  black canvas. Default false (offscreen backends swallow the failure). */
  readonly required?: boolean;
}

export function configureWebGpuCanvas(
  canvas: HTMLCanvasElement,
  device: GPUDevice,
  onError?: (error: unknown) => void,
  options?: ConfigureWebGpuCanvasOptions,
): void {
  try {
    const ctx = canvas.getContext('webgpu');
    if (ctx == null) {
      if (options?.required === true) {
        throw new Error(
          'configureWebGpuCanvas: canvas.getContext("webgpu") returned null',
        );
      }
      return;
    }

    const format = (typeof navigator !== 'undefined' && 'gpu' in navigator
      ? (navigator.gpu as { getPreferredCanvasFormat?: () => GPUTextureFormat })
          .getPreferredCanvasFormat?.() ?? ('bgra8unorm')
      : ('bgra8unorm' as GPUTextureFormat));
    ctx.configure({
      device,
      format,
      alphaMode: 'opaque',
    });
  } catch (err) {
    // Best-effort canvas configure for attachVitrum swap-chain plumbing.
    try {
      onError?.(err);
    } catch {
      // Host error callbacks must not break best-effort canvas configuration.
    }
    // V1-7 — swap-chain-required backends must surface a configure failure
    // rather than silently render black forever. Offscreen backends keep the
    // historical swallow behaviour (required defaults to false).
    if (options?.required === true) throw err;
  }
}
