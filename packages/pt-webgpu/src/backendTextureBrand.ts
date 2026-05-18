// W3-D19 — WebGPU brand constructors / narrowers for the @vitrum/core
// `BackendTexture<'webgpu'>` and `BackendTextureFormat<'webgpu'>` slots,
// scoped to @vitrum/pt-webgpu. Parallel to the constructors in
// @vitrum/walkaround-hybrid; pt-webgpu does not depend on walkaround-hybrid
// so each WebGPU-backed package owns a local copy of these zero-cost
// brand bridges.
//
// At runtime the brand is purely type-system — the underlying `GPUTexture`,
// `GPUTextureView`, or `GPUTextureFormat` reference flows through unchanged.

import type { BackendTexture, BackendTextureFormat } from '@vitrum/core';

/**
 * Brand a raw `GPUTexture` or `GPUTextureView` as a `BackendTexture<'webgpu'>`.
 * Zero-cost at runtime — returns the same reference.
 */
export function asWebGPUBackendTexture(
  view: GPUTextureView | GPUTexture,
): BackendTexture<'webgpu'> {
  return view as unknown as BackendTexture<'webgpu'>;
}

/**
 * Brand a `GPUTextureFormat` string as a `BackendTextureFormat<'webgpu'>`.
 * Zero-cost at runtime — returns the same string.
 */
export function asWebGPUBackendTextureFormat(
  format: GPUTextureFormat,
): BackendTextureFormat<'webgpu'> {
  return format as unknown as BackendTextureFormat<'webgpu'>;
}

/**
 * Recover the underlying `GPUTextureView` from a `BackendTexture<'webgpu'>`.
 * The brand guarantees the value was constructed by `asWebGPUBackendTexture`.
 * Zero-cost at runtime — returns the same reference.
 */
export function narrowToWebGPUTextureView(
  tex: BackendTexture<'webgpu'>,
): GPUTextureView {
  return tex as unknown as GPUTextureView;
}

/**
 * Recover the underlying `GPUTextureFormat` string from a branded format.
 * Zero-cost at runtime.
 */
export function narrowToWebGPUTextureFormat(
  fmt: BackendTextureFormat<'webgpu'>,
): GPUTextureFormat {
  return fmt as unknown as GPUTextureFormat;
}
