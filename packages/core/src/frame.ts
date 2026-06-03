// Per-frame I/O.
//
// Design principle: anything that changes every frame lives here, not in
// `Scene`. The camera is per-frame because it scrubs continuously during
// orbit; the frame seed is per-frame because the QMC sequence advances; the
// shutter time is per-frame because motion blur samples within an interval.
//
// Quality dials (samplesTarget, bounces, resolutionFactor, etc.) are ALSO
// per-frame. The host owns quality — it changes quality by passing a different
// `FrameInput.quality` payload, not by calling a mutation on the engine.
// This is what makes PT_PREVIEW and PT_FINAL two different payloads to the
// same engine instance, not a mode-switch event.
//
// This split is what makes `engine.setScene(scene)` cheap — the scene only
// changes when geometry/materials/lights change. Frame state is hot.

import type { Mat4, Vec3 } from './scene/index.js';

// ────────────────────────────────────────────────────────────────────────────
// Per-frame quality settings (host → engine, every frame)
// ────────────────────────────────────────────────────────────────────────────

/** Per-frame quality dials. The host owns these — they are NOT engine state.
 *  PT preview, PT final, walkaround real-time, and offline hero render all use
 *  the same engine instance with different quality payloads per frame.
 *
 *  Engines clamp values against `EngineCapabilities.maxBounces` and
 *  `EngineCapabilities.maxSamplesPerPixel` (structural caps fixed at engine
 *  creation). Out-of-range values are clamped, not errors. */
export interface FrameQualitySettings {
  /** Convergence target. PT engines accumulate until samplesAccumulated >=
   *  samplesTarget, then flip `FrameOutput.isConverged = true`. Walkaround
   *  engines ignore (they resample every frame). Default: engine-specific. */
  readonly samplesTarget?: number;

  /** Per-frame bounce count. Must be <= EngineCapabilities.maxBounces.
   *  Default: engine-specific (typically the cap). */
  readonly bounces?: number;

  /** Glossy filtering strength (three-gpu-pathtracer fork concept). 0 = off
   *  (physically correct), 1 = aggressive firefly suppression. Backends that
   *  don't support glossy filtering ignore this. */
  readonly filteredGlossyFactor?: number;

  /** Internal render resolution factor in (0, 1]. Engines render at
   *  `viewport.width * resolutionFactor` and upscale. Default: 1.0. */
  readonly resolutionFactor?: number;
}

// ────────────────────────────────────────────────────────────────────────────
// Frame inputs (host → engine, every frame)
// ────────────────────────────────────────────────────────────────────────────

export interface FrameInput {
  // ── Camera (column-major matrices, three.js convention) ────────────────
  readonly viewMatrix: Mat4;
  readonly projMatrix: Mat4;
  readonly cameraPosition: Vec3;

  /** Previous-frame matrices for temporal accumulation, reprojection, and
   *  motion-vector computation. Hosts that don't track these can pass the
   *  current matrices (degrades to "no temporal reuse"). */
  readonly prevViewMatrix?: Mat4;
  readonly prevProjMatrix?: Mat4;

  // ── Viewport ────────────────────────────────────────────────────────────
  /**
   * Per-frame viewport (physical pixel dimensions + DPR).
   *
   * **Engine-honour contract (A4):**
   * - Generic PT engines (e.g. `@vitrum/pt-webgl`) honour `viewport.width`
   *   and `viewport.height` every frame; passing different values triggers
   *   an internal render-target resize transparently.
   * - **`HybridEngine` (`@vitrum/walkaround-hybrid`) does NOT honour
   *   `viewport`.** Its WebGPU render targets (DDGI atlas, ReSTIR reservoirs,
   *   history textures, accumulation buffer) are sized to the *canvas* at
   *   construction via `HybridEngineOptions.{width,height}` and the canvas
   *   size can only be changed explicitly via `HybridEngine.setSize(width,
   *   height)`. Pushing a new `FrameInput.viewport` is silently ignored —
   *   there is no per-frame canvas-resize-detection branch in
   *   `HybridEngine.renderFrame`.
   *
   *   **However, `FrameInput.quality.resolutionFactor` IS honoured per-frame
   *   by HybridEngine** (Phase-0 productization): it scales the *internal*
   *   render resolution (= canvas × factor) and the composite pass upscales
   *   to the full canvas. The internal reallocation is debounced so a host
   *   ramping the factor continuously does not thrash the temporal
   *   accumulator. So: changing the *canvas* size still requires `setSize`;
   *   changing only the internal-resolution scale is per-frame via
   *   `quality.resolutionFactor`.
   *
   * Hosts driving `HybridEngine` directly MUST call `engine.setSize()`
   * when their canvas dimensions change. Hosts using `attachVitrum()`
   * get this for free: its `ResizeObserver` duck-types `engine.setSize`
   * and calls it when the underlying engine exposes the method (i.e.
   * when the backend is walkaround-hybrid).
   */
  readonly viewport: Viewport;

  // ── Frame indexing ──────────────────────────────────────────────────────
  /** Monotonically-increasing frame index. Engines use this for temporal
   *  stability (Hammersley sequence, golden-ratio rotation, sub-grid
   *  stratification). */
  readonly frameIndex: number;

  /** Random seed for this frame. Hosts should derive from frameIndex
   *  (e.g., `frameIndex * 1664525 + 1013904223`) so frames are reproducible
   *  but uncorrelated. */
  readonly frameSeed: number;

  // ── Optional: per-frame quality dials ──────────────────────────────────
  /** Quality settings for this frame. The host passes different payloads to
   *  switch between PT preview, PT final, hero render, and walkaround modes —
   *  without recreating the engine. When omitted, each dial falls back to its
   *  engine-specific default (typically the structural cap for bounces, 1.0
   *  for resolutionFactor). */
  readonly quality?: FrameQualitySettings;

  // ── Optional: backend-specific output target ────────────────────────────
  /** Backend-opaque swap-chain target. Some WebGPU backends (notably
   *  `@vitrum/walkaround-hybrid`) require a fresh `GPUTextureView` each frame.
   *  Others (`@vitrum/pt-webgpu`) currently render to internal textures and
   *  ignore this field. WebGL backends ignore both fields and write to their
   *  own framebuffer; the host then
   *  reads via `FrameOutput.primaryRadiance`. Typed as opaque so the
   *  backend-agnostic core does not pull in WebGPU type declarations.
   *  Backends document what they require and cast at the boundary. */
  readonly swapChainView?: BackendTexture;
  /** Backend-opaque swap-chain format. WebGPU backends expect a
   *  `GPUTextureFormat` string literal; WebGL backends ignore. */
  readonly swapChainFormat?: BackendTextureFormat;
}

export interface Viewport {
  /** Physical pixel dimensions (DPR-applied). Engines render at this
   *  resolution; engines that downscale internally do so via
   *  `FrameInput.quality.resolutionFactor`, not by the host pre-applying DPR. */
  readonly width: number;
  readonly height: number;
  readonly devicePixelRatio: number;
}

// ────────────────────────────────────────────────────────────────────────────
// Frame outputs (engine → host, every frame)
// ────────────────────────────────────────────────────────────────────────────

interface FrameOutputBase {
  /** Number of accumulated samples-per-pixel. Increments each `renderFrame`
   *  call until target reached. Resets to 0 on `engine.reset()`. */
  readonly samplesAccumulated: number;

  /** True when the engine considers the image converged enough to display
   *  the post-processing pipeline. PT engines flip this at sample target;
   *  real-time walkaround engines typically keep this false because they
   *  resample every frame instead of converging to a terminal image. */
  readonly isConverged: boolean;
}

/** Engine skipped rendering this frame (paused, throttled, or not ready). */
export interface FrameSkipped extends FrameOutputBase {
  readonly kind: 'skipped';
  readonly samplesAccumulated: 0;
  readonly isConverged: false;
}

/** Engine produced render targets for this frame. */
export interface FrameRendered extends FrameOutputBase {
  readonly kind: 'rendered';

  /** Primary radiance buffer — the final converged-or-converging color image.
   *  Format depends on backend: WebGPU returns a `GPUTexture`, WebGL2 returns
   *  the renderer's framebuffer or a `WebGLTexture` handle. */
  readonly primaryRadiance: BackendTexture;

  // ── Optional G-buffer ──────────────────────────────────────────────────
  /** Encoded normal + linear depth. RGBA16F: xyz = world-space normal,
   *  w = linear-depth (camera-space, always positive). */
  readonly normalDepth?: BackendTexture;

  /** Demodulated albedo (base color × occlusion, no lighting). Used by OIDN
   *  and atrous-variance to denoise lighting independently from texture detail. */
  readonly albedo?: BackendTexture;

  // ── Optional metadata ──────────────────────────────────────────────────
  /** Per-pixel variance estimate (Welford running variance). Used by adaptive
   *  sampling and by some denoisers. RGBA32F format. */
  readonly variance?: BackendTexture;

  /** Motion vectors for temporal reprojection (svgf-real denoiser) and
   *  checkerboard upsampling. RG32F: (dx, dy) in pixels. */
  readonly motionVectors?: BackendTexture;
}

export type FrameOutput = FrameSkipped | FrameRendered;

/** Opaque texture handle. The shape varies per backend; hosts pass it back
 *  through `engine.renderFrame` outputs into post-processing chains, save
 *  pipelines, etc. without inspecting it. */
declare const BACKEND_TEXTURE_BRAND: unique symbol;
export type BackendTexture<
  TBackend extends string = string,
  THandle = unknown,
> = THandle & { readonly [BACKEND_TEXTURE_BRAND]: TBackend };

/** Opaque texture-format token. Backend-specific (e.g. WebGPU uses
 *  `GPUTextureFormat` string literals); the core contract treats it as
 *  opaque so backend types don't bleed in here. */
declare const BACKEND_TEXTURE_FORMAT_BRAND: unique symbol;
export type BackendTextureFormat<
  TBackend extends string = string,
  TFormat = unknown,
> = TFormat & { readonly [BACKEND_TEXTURE_FORMAT_BRAND]: TBackend };

/** Brand a backend texture handle at the boundary where backend identity is known. */
export function asBackendTexture<TBackend extends string, THandle>(
  value: THandle,
): BackendTexture<TBackend, THandle> {
  return value as BackendTexture<TBackend, THandle>;
}

/** Brand a backend texture format token at the boundary where backend identity is known. */
export function asBackendTextureFormat<TBackend extends string, TFormat>(
  value: TFormat,
): BackendTextureFormat<TBackend, TFormat> {
  return value as BackendTextureFormat<TBackend, TFormat>;
}

/** Narrow any backend texture handle to a specific backend identity. */
export function narrowToBackendTexture<TBackend extends string, THandle = unknown>(
  value: BackendTexture | null | undefined,
): BackendTexture<TBackend, THandle> | null {
  if (value == null) return null;
  return value as BackendTexture<TBackend, THandle>;
}

/** Narrow any backend texture format token to a specific backend identity. */
export function narrowToBackendTextureFormat<TBackend extends string, TFormat = unknown>(
  value: BackendTextureFormat | null | undefined,
): BackendTextureFormat<TBackend, TFormat> | null {
  if (value == null) return null;
  return value as BackendTextureFormat<TBackend, TFormat>;
}
