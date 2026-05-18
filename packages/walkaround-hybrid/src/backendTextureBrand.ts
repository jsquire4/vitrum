// W3-D19 — WebGPU brand constructors / narrowers for the @vitrum/core
// `BackendTexture<'webgpu'>` and `BackendTextureFormat<'webgpu'>` slots.
//
// `BackendTexture` is a nominally-branded opaque type in `@vitrum/core` —
// hosts cannot pass a raw `GPUTextureView` into `FrameInput.swapChainView`
// without going through `asWebGPUBackendTexture`, and HybridEngine cannot
// pull the underlying view out without going through
// `narrowToWebGPUTextureView`. This is zero-cost at runtime — the same
// `GPUTextureView` reference flows through both directions; the brand only
// exists in TS's type system.
//
// The point of the brand is to make mismatched-backend wiring (passing a
// `WebGLTexture` to a WebGPU engine, or pulling a `GPUTextureView` out of a
// WebGL `primaryRadiance` slot) a compile-time error instead of a runtime
// cast failure deep inside renderFrame.

import type { BackendTexture, BackendTextureFormat } from '@vitrum/core';

// ────────────────────────────────────────────────────────────────────────────
// Construction (host-side: raw WebGPU handle → branded BackendTexture)
// ────────────────────────────────────────────────────────────────────────────

/**
 * Brand a raw `GPUTextureView` (or `GPUTexture` for FrameOutput slots) as a
 * `BackendTexture<'webgpu'>` suitable for `FrameInput.swapChainView` or
 * `FrameOutput.primaryRadiance` etc. Zero-cost at runtime — returns the
 * same reference; the brand is type-system-only.
 *
 * Usage (host, inside rAF):
 * ```ts
 * const view = gpuCtx.getCurrentTexture().createView();
 * engine.renderFrame({
 *   ...,
 *   swapChainView: asWebGPUBackendTexture(view),
 *   swapChainFormat: asWebGPUBackendTextureFormat(format),
 * });
 * ```
 *
 * Engines also use this to brand their `FrameOutput.primaryRadiance` /
 * G-buffer slots so consumers can statically distinguish WebGPU handles
 * from WebGL ones downstream. `@vitrum/pt-webgpu` exports a parallel
 * constructor with identical semantics (the two packages don't depend on
 * each other, so each owns a local copy of the brand bridge).
 */
export function asWebGPUBackendTexture(
  view: GPUTextureView | GPUTexture,
): BackendTexture<'webgpu'> {
  return view as unknown as BackendTexture<'webgpu'>;
}

/**
 * Brand a `GPUTextureFormat` string literal as a `BackendTextureFormat<'webgpu'>`
 * suitable for `FrameInput.swapChainFormat`. Zero-cost at runtime — returns
 * the same string.
 */
export function asWebGPUBackendTextureFormat(
  format: GPUTextureFormat,
): BackendTextureFormat<'webgpu'> {
  return format as unknown as BackendTextureFormat<'webgpu'>;
}

// ────────────────────────────────────────────────────────────────────────────
// Narrowing (engine-side: branded BackendTexture → raw WebGPU handle)
// ────────────────────────────────────────────────────────────────────────────

/**
 * Recover the underlying `GPUTextureView` from a `BackendTexture<'webgpu'>`.
 * The brand guarantees (at the type level) that the value was constructed
 * by `asWebGPUBackendTexture`, so this is a safe unbrand. Zero-cost at
 * runtime — returns the same reference.
 *
 * Used inside HybridEngine.renderFrame to bridge the branded
 * `FrameInput.swapChainView` slot back to the concrete `GPUTextureView`
 * that the internal pipeline expects.
 */
export function narrowToWebGPUTextureView(tex: BackendTexture<'webgpu'>): GPUTextureView {
  return tex as unknown as GPUTextureView;
}

/**
 * Recover the underlying `GPUTextureFormat` from a `BackendTextureFormat<'webgpu'>`.
 * Mirror of `narrowToWebGPUTextureView` for the format slot.
 */
export function narrowToWebGPUTextureFormat(
  fmt: BackendTextureFormat<'webgpu'>,
): GPUTextureFormat {
  return fmt as unknown as GPUTextureFormat;
}
