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

/**
 * Tag on the skipped variant of `FrameOutput`. The engine elected NOT to
 * render this frame and produced no usable texture.
 *
 * Known reasons (string-literal union so consumers can switch exhaustively):
 *   - `'fps-throttle'`   — internal frame-rate cap (walkaround targets ~60 Hz).
 *   - `'paused'`         — engine is in the `'paused'` state.
 *   - `'no-scene'`       — `setScene()` has not been called yet.
 *   - `'pending-init'`   — backend pipeline still warming up (BVH compile,
 *                          WebGPU shader module ready, etc.).
 *   - `'no-swap-view'`   — host did not supply `swapChainView` for a backend
 *                          that draws directly into the swap chain.
 *   - `'rebuild'`        — a pipeline rebuild was triggered this tick; this
 *                          frame is dropped while the new pipeline initialises.
 *   - `'disposed'`       — engine was disposed; caller likely racing dispose.
 *
 * Forward-compat: `string` is also accepted so backends can introduce new
 * reasons without bumping the contract. Hosts that switch on `reason` should
 * supply a default branch.
 */
export type FrameSkipReason =
  | 'fps-throttle'
  | 'paused'
  | 'no-scene'
  | 'pending-init'
  | 'no-swap-view'
  | 'rebuild'
  | 'disposed'
  | (string & {});

/**
 * The engine elected to skip this frame. No texture was produced.
 * Hosts should leave the previous frame on screen (engines that need to keep
 * the swap-chain alive — e.g. walkaround's FPS throttle — blit the last
 * presented frame internally before returning).
 */
export interface FrameSkipped {
  readonly kind: 'skipped';
  /** Why the frame was skipped. See `FrameSkipReason` for known values. */
  readonly reason: FrameSkipReason;
}

/**
 * The engine produced a rendered frame. `primaryRadiance` is a valid
 * backend-texture handle; optional G-buffer aux textures may also be present.
 */
export interface FrameRendered {
  readonly kind: 'rendered';

  /**
   * Primary radiance buffer — the final converged-or-converging color image.
   * Format depends on backend: WebGPU returns a `GPUTexture` (or the
   * `GPUTextureView` of the swap chain for surface-drawing backends like
   * walkaround-hybrid); WebGL2 returns the renderer's framebuffer or a
   * `WebGLTexture` handle.
   */
  readonly primaryRadiance: BackendTexture;

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
   *  call until target reached. Resets to 0 on `engine.reset()`. Walkaround
   *  engines return `1` per rendered frame (they resample rather than
   *  accumulate). */
  readonly samplesAccumulated: number;

  /** True when the engine considers the image converged enough to display
   *  the post-processing pipeline. PT engines flip this at sample target;
   *  walkaround engines flip it once temporal accumulation has stabilized. */
  readonly isConverged: boolean;
}

/**
 * Discriminated union returned by `Engine.renderFrame`. The `kind` field
 * tells the host whether a frame was produced. This replaces the previous
 * `{ primaryRadiance: BackendTexture | null; samplesAccumulated: number }`
 * shape, where `samplesAccumulated === 0` was an in-band sentinel for
 * "skipped frame" — easy to violate and impossible for the type system
 * to enforce.
 *
 * Recommended consumer pattern:
 *
 * ```ts
 * const out = engine.renderFrame(input);
 * if (out.kind === 'skipped') {
 *   // diagnostics / FPS counter / leave previous frame on screen
 *   return;
 * }
 * // TypeScript now knows out.primaryRadiance is a real BackendTexture and
 * // out.samplesAccumulated / out.isConverged are present.
 * postProcess(out.primaryRadiance);
 * ```
 */
export type FrameOutput = FrameSkipped | FrameRendered;

/** Opaque texture handle. The shape varies per backend; hosts pass it back
 *  through `engine.renderFrame` outputs into post-processing chains, save
 *  pipelines, etc. without inspecting it. */
export type BackendTexture = unknown;

/** Opaque texture-format token. Backend-specific (e.g. WebGPU uses
 *  `GPUTextureFormat` string literals); the core contract treats it as
 *  opaque so backend types don't bleed in here. */
export type BackendTextureFormat = unknown;
