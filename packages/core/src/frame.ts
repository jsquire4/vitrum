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

import type { Mat4, Vec3 } from './scene.js';

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

  // ── Optional: motion blur ───────────────────────────────────────────────
  /** Shutter time in [0, 1], representing a position within a single shutter
   *  interval. Engines that report `capabilities.supportsMotionBlur = true`
   *  use this to sample the time-varying scene. Engines that don't ignore. */
  readonly shutterTime?: number;

  // ── Optional: backend-specific output target ────────────────────────────
  /** Backend-opaque swap-chain target. WebGPU backends (`@vitrum/pt-webgpu`,
   *  `@vitrum/walkaround-hybrid`) expect a `GPUTextureView`. WebGL backends
   *  ignore both fields and write to their own framebuffer; the host then
   *  reads via `FrameOutput.primaryRadiance`. Typed as opaque so the
   *  backend-agnostic core does not pull in WebGPU type declarations.
   *  Backends document what they require and recover the underlying handle
   *  at the boundary via their `as*BackendTexture` constructor / narrow
   *  helpers (e.g. `asWebGPUBackendTexture` in `@vitrum/walkaround-hybrid`).
   *  Both swap-chain fields are WebGPU-specific by today's contract — the
   *  brand parameter `'webgpu'` documents this and lets TS reject e.g.
   *  passing a WebGL handle here without going through an explicit
   *  constructor. */
  readonly swapChainView?: BackendTexture<'webgpu'>;
  /** Backend-opaque swap-chain format. WebGPU backends expect a
   *  `GPUTextureFormat` string literal; WebGL backends ignore. */
  readonly swapChainFormat?: BackendTextureFormat<'webgpu'>;
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

export interface FrameOutput {
  /**
   * Primary radiance buffer — the final converged-or-converging color image.
   * Format depends on backend: WebGPU returns a `GPUTexture`, WebGL2 returns
   * the renderer's framebuffer or a `WebGLTexture` handle.
   *
   * **Skip frames:** When `samplesAccumulated === 0`, the engine elected to
   * skip this frame (e.g. frame-rate throttle, pipeline not yet ready, paused
   * state). On skip frames `primaryRadiance` is `null` — hosts MUST check
   * `samplesAccumulated > 0` before treating `primaryRadiance` as a valid
   * texture handle. Example guard:
   *
   * ```ts
   * const out = engine.renderFrame(input);
   * if (out.samplesAccumulated > 0 && out.primaryRadiance != null) {
   *   // safe to use out.primaryRadiance
   * }
   * ```
   *
   * Walkaround engines set `samplesAccumulated = 1` on every rendered frame
   * (they resample rather than accumulate). `samplesAccumulated = 0` is the
   * universal "skip frame" sentinel.
   */
  readonly primaryRadiance: BackendTexture | null;

  // ── Optional G-buffer (Phase 6 sprint 5 introduces these) ──────────────
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

  /** Per-pixel motion vectors in screen space. RG32F: (dx, dy) in pixels.
   *  Reserved for future temporal-reprojection denoiser (real Schied 2017
   *  SVGF, see plan/sprint-svgf-real-future.md) + checkerboard upsampling. */
  readonly motionVectors?: BackendTexture;

  // ── Convergence stats ──────────────────────────────────────────────────
  /** Number of accumulated samples-per-pixel. Increments each `renderFrame`
   *  call until target reached. Resets to 0 on `engine.reset()`. */
  readonly samplesAccumulated: number;

  /** True when the engine considers the image converged enough to display
   *  the post-processing pipeline. PT engines flip this at sample target;
   *  walkaround engines flip it once temporal accumulation has stabilized. */
  readonly isConverged: boolean;
}

// ────────────────────────────────────────────────────────────────────────────
// Backend texture branding (W3-D19)
// ────────────────────────────────────────────────────────────────────────────
//
// `BackendTexture` and `BackendTextureFormat` are nominally branded — values
// CANNOT be assigned to them without going through a backend's `as*Backend*`
// constructor. The brand encodes the backend identity (`'webgpu'`, `'webgl'`,
// or `unknown` for backend-agnostic slots) so TS catches mismatched-backend
// assignments at compile time instead of failing at runtime cast.
//
// At runtime the brand is zero-cost — the underlying value (`GPUTextureView`,
// `WebGLTexture`, etc.) flows through unchanged. The brand exists only in
// types; `unique symbol`-keyed properties on type-only `declare const`
// declarations never exist on real values, so the structural type system
// treats them as nominal markers.
//
// Construction is the only way in: backends export `asWebGPUBackendTexture`
// (in `@vitrum/walkaround-hybrid` / `@vitrum/pt-webgpu`) and
// `asWebGLBackendTexture` (in `@vitrum/pt-webgl`). On the read side they
// expose a matching `narrowTo*` that asserts the brand and returns the
// underlying handle. The host code never sees these brand symbols directly.

declare const __backendTextureBrand: unique symbol;
/** Opaque, backend-branded texture handle. `TBackend` is `'webgpu'`,
 *  `'webgl'`, or `unknown` (when the slot is backend-agnostic, e.g.
 *  `FrameOutput.primaryRadiance` which the host may forward to either a
 *  WebGPU composite or a WebGL readback). Construct via a backend's
 *  `as*BackendTexture` helper; never via raw assignment. */
export type BackendTexture<TBackend = unknown> = {
  readonly [__backendTextureBrand]: TBackend;
};

declare const __backendTextureFormatBrand: unique symbol;
/** Opaque, backend-branded texture-format token. Backed by `string` at
 *  runtime (e.g. WebGPU's `GPUTextureFormat` is a string literal union).
 *  Construct via a backend's `as*BackendTextureFormat` helper. */
export type BackendTextureFormat<TBackend = unknown> = string & {
  readonly [__backendTextureFormatBrand]: TBackend;
};
