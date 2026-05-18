// W3-D19 — WebGL brand constructors / narrowers for the @vitrum/core
// `BackendTexture<'webgl'>` slot.
//
// `BackendTexture` is a nominally-branded opaque type in `@vitrum/core` —
// pt-webgl returns its accumulation `WebGLTexture` via
// `FrameOutput.primaryRadiance`, branded as `BackendTexture<'webgl'>` so
// downstream consumers (readback paths, host save pipelines) can statically
// distinguish it from a WebGPU-branded handle. Zero-cost at runtime — the
// `WebGLTexture` reference flows through unchanged.

import type { BackendTexture } from '@vitrum/core';

// ────────────────────────────────────────────────────────────────────────────
// Construction (engine-side: raw WebGL handle → branded BackendTexture)
// ────────────────────────────────────────────────────────────────────────────

/**
 * Brand a raw `WebGLTexture` as a `BackendTexture<'webgl'>` suitable for
 * `FrameOutput.primaryRadiance`. Zero-cost at runtime — returns the same
 * `WebGLTexture` reference; the brand is type-system-only.
 */
export function asWebGLBackendTexture(tex: WebGLTexture): BackendTexture<'webgl'> {
  return tex as unknown as BackendTexture<'webgl'>;
}

// ────────────────────────────────────────────────────────────────────────────
// Narrowing (consumer-side: branded BackendTexture → raw WebGL handle)
// ────────────────────────────────────────────────────────────────────────────

/**
 * Recover the underlying `WebGLTexture` from a `BackendTexture<'webgl'>`.
 * The brand guarantees (at the type level) that the value was constructed
 * by `asWebGLBackendTexture`. Zero-cost at runtime — returns the same
 * reference.
 */
export function narrowToWebGLTexture(tex: BackendTexture<'webgl'>): WebGLTexture {
  return tex as unknown as WebGLTexture;
}
