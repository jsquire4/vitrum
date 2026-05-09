// Per-frame I/O.
//
// Design principle: anything that changes every frame lives here, not in
// `Scene`. The camera is per-frame because it scrubs continuously during
// orbit; the frame seed is per-frame because the QMC sequence advances; the
// shutter time is per-frame because motion blur samples within an interval.
//
// This split is what makes `engine.setScene(scene)` cheap — the scene only
// changes when geometry/materials/lights change. Frame state is hot.

import type { Mat4, Vec3 } from './scene.js';

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

  // ── Optional: motion blur ───────────────────────────────────────────────
  /** Shutter time in [0, 1], representing a position within a single shutter
   *  interval. Engines that report `capabilities.supportsMotionBlur = true`
   *  use this to sample the time-varying scene. Engines that don't ignore. */
  readonly shutterTime?: number;

  // ── Optional: backend-specific output target ────────────────────────────
  /** WebGPU swap-chain texture view to render into. Required for WebGPU
   *  backends; ignored by WebGL backends (which write to their own
   *  framebuffer and the host reads via `FrameOutput.primaryRadiance`). */
  readonly swapChainView?: GPUTextureView;
  readonly swapChainFormat?: GPUTextureFormat;
}

export interface Viewport {
  /** Physical pixel dimensions (DPR-applied). Engines render at this
   *  resolution; engines that downscale internally do so via their own
   *  `resolutionFactor` config, not by the host pre-applying DPR. */
  readonly width: number;
  readonly height: number;
  readonly devicePixelRatio: number;
}

// ────────────────────────────────────────────────────────────────────────────
// Frame outputs (engine → host, every frame)
// ────────────────────────────────────────────────────────────────────────────

export interface FrameOutput {
  /** Primary radiance buffer — the final converged-or-converging color image.
   *  Format depends on backend: WebGPU returns a `GPUTexture`, WebGL2 returns
   *  the renderer's framebuffer or a `WebGLTexture` handle. */
  readonly primaryRadiance: BackendTexture;

  // ── Optional G-buffer (Phase 6 sprint 5 introduces these) ──────────────
  /** Encoded normal + linear depth. RGBA16F: xyz = world-space normal,
   *  w = linear-depth (camera-space, always positive). */
  readonly normalDepth?: BackendTexture;

  /** Demodulated albedo (base color × occlusion, no lighting). Used by OIDN
   *  and SVGF to denoise lighting independently from texture detail. */
  readonly albedo?: BackendTexture;

  // ── Optional metadata ──────────────────────────────────────────────────
  /** Per-pixel variance estimate (Welford running variance). Used by adaptive
   *  sampling and by some denoisers. RGBA32F format. */
  readonly variance?: BackendTexture;

  /** Per-pixel motion vectors in screen space. RG32F: (dx, dy) in pixels.
   *  Used by SVGF + checkerboard upsampling. */
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

/** Opaque texture handle. The shape varies per backend; hosts pass it back
 *  through `engine.renderFrame` outputs into post-processing chains, save
 *  pipelines, etc. without inspecting it. */
export type BackendTexture = unknown;
