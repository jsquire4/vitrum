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

  /** Per-frame renderer-depth control. Progressive path tracers interpret this
   *  as a finite path-bounce count; other renderer families may expose a
   *  bounded quality regime instead. See
   *  `EngineCapabilities.supportDetails.bounceSemantics` for the live
   *  interpretation. Must be <= EngineCapabilities.maxBounces.
   *  Default: engine-specific (typically the cap). */
  readonly bounces?: number;

  /** Glossy filtering strength for converged path tracers. 0 = off
   *  (physically correct), 1 = aggressive firefly suppression. Backends that
   *  don't support glossy filtering ignore this. */
  readonly filteredGlossyFactor?: number;

  /** Internal render resolution factor in (0, 1]. Engines render at
   *  `viewport.width * resolutionFactor` and upscale. Default: 1.0. */
  readonly resolutionFactor?: number;

  /** Linear-exposure multiplier applied before tonemapping. Default: 1.0. */
  readonly exposure?: number;

  /** Output tonemap operator. Default: 'aces' (the historical hardcoded curve).
   *  'none' = raw linear HDR (no curve); 'linear' = exposure + clamp only.
   *  Operators live in `@vitrum/shared-samplers` (`applyTonemap` / `vitrumTonemap`). */
  readonly tonemap?: 'aces' | 'agx' | 'reinhard' | 'linear' | 'none';

  /** Output color space for the presented frame. Default: 'srgb'. */
  readonly outputColorSpace?: 'srgb' | 'linear';
}

/**
 * Shared stills / viewer-final quality payload. Path tracers use `bounces` as
 * path depth and `samplesTarget` as the accumulation goal. Walkaround ignores
 * `samplesTarget` and clamps `bounces` to its DDGI feedback regimes (1 or 2).
 */
export const QUALITY_FINAL: FrameQualitySettings = Object.freeze({
  samplesTarget: 16,
  bounces: 8,
  exposure: 1,
  tonemap: 'aces',
  outputColorSpace: 'srgb',
});

/**
 * Shared viewer-preview quality payload. Same bag as {@link QUALITY_FINAL} so
 * one `FrameInput.quality` can ride both progressive phases without backend-
 * specific stripping. Preview engines ignore fields they do not accumulate.
 */
export const QUALITY_PREVIEW: FrameQualitySettings = QUALITY_FINAL;

// ────────────────────────────────────────────────────────────────────────────
// Frame inputs (host → engine, every frame)
// ────────────────────────────────────────────────────────────────────────────

export interface FrameInput {
  // ── Camera (column-major matrices, three.js convention) ────────────────
  readonly viewMatrix: Mat4;
  readonly projMatrix: Mat4;
  /**
   * @deprecated The camera position is derived canonically from
   * `inverse(viewMatrix)`. When supplied for source compatibility, every
   * backend must reject a meaningful disagreement with that derived value.
   * Omit this field in new integrations.
   */
  readonly cameraPosition?: Vec3;

  /** Previous-frame matrices for temporal accumulation, reprojection, and
   *  motion-vector computation. Hosts that don't track these can pass the
   *  current matrices (degrades to "no temporal reuse"). */
  readonly prevViewMatrix?: Mat4;
  readonly prevProjMatrix?: Mat4;

  // ── Viewport ────────────────────────────────────────────────────────────
  /**
   * Per-frame viewport (physical pixel dimensions + DPR).
   *
   * Every shipped rendering backend honours `viewport.width` and
   * `viewport.height` on each `renderFrame` call. Changed physical dimensions
   * transparently resize persistent targets before rendering. Backends with an
   * explicit `setSize(width, height)` method expose the same operation as an
   * eager host hook; calling it is optional when the next frame carries the
   * correct viewport.
   *
   * `FrameInput.quality.resolutionFactor` remains independent: viewport controls
   * physical output dimensions, while the factor controls internal working
   * resolution. `attachVitrum()` calls `setSize` from its `ResizeObserver` when
   * available so allocation can complete before the next frame.
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

/** A frame whose redundant camera-position input has been resolved. */
export type CanonicalCameraFrameInput = FrameInput & {
  readonly cameraPosition: Vec3;
};

/**
 * Absolute component tolerance used when checking a legacy
 * `FrameInput.cameraPosition` against the position derived from `viewMatrix`.
 */
export const CAMERA_POSITION_ABSOLUTE_TOLERANCE = 1e-5;

/**
 * Relative component tolerance for legacy camera-position checks: eight f32
 * epsilons, enough to cover a Float32 matrix paired with a host's f64 position
 * at large world coordinates without accepting a materially different eye.
 */
export const CAMERA_POSITION_RELATIVE_TOLERANCE =
  8 * 1.1920928955078125e-7;

function finiteMatrixElement(
  matrix: ArrayLike<number>,
  index: number,
  label: string,
): number {
  const value = matrix[index];
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new RangeError(
      `${label}[${index}] must be finite (got ${String(value)}).`,
    );
  }
  return value;
}

/** Collapse IEEE -0 so canonical camera payloads have one stable zero encoding. */
function canonicalCameraComponent(value: number): number {
  return value === 0 ? 0 : value;
}

/**
 * Derive the world-space eye position from a column-major camera view matrix.
 *
 * A camera view is affine: its upper-left 3×3 maps world-space directions into
 * view space and its fourth column is translation. Solving `A·eye + t = 0`
 * yields the translation column of `inverse(viewMatrix)` without coupling core
 * to a renderer package or allocating a full inverse matrix.
 *
 * @throws when the matrix is malformed, non-finite, or has a singular 3×3
 * camera transform.
 */
export function deriveCameraPositionFromViewMatrix(
  viewMatrix: ArrayLike<number>,
  label = 'FrameInput.viewMatrix',
): Vec3 {
  if (viewMatrix.length !== 16) {
    throw new RangeError(
      `${label} must contain exactly 16 values (got ${viewMatrix.length}).`,
    );
  }
  const m = Array.from(
    { length: 16 },
    (_, index) => finiteMatrixElement(viewMatrix, index, label),
  );
  // The homogeneous scale is fixed by m[15] = 1. Translation can be
  // arbitrarily large, but it cannot introduce round-off into the structurally
  // zero projective row, so it must not relax this affine-shape check.
  const affineTolerance = CAMERA_POSITION_RELATIVE_TOLERANCE;
  if (
    Math.abs(m[3]!) > affineTolerance ||
    Math.abs(m[7]!) > affineTolerance ||
    Math.abs(m[11]!) > affineTolerance ||
    Math.abs(m[15]! - 1) > affineTolerance
  ) {
    throw new RangeError(
      `${label} must be an affine camera view matrix with last row [0, 0, 0, 1].`,
    );
  }

  // Column-major upper-left 3×3 and translation column.
  const a00 = m[0]!, a01 = m[4]!, a02 = m[8]!;
  const a10 = m[1]!, a11 = m[5]!, a12 = m[9]!;
  const a20 = m[2]!, a21 = m[6]!, a22 = m[10]!;
  const tx = m[12]!, ty = m[13]!, tz = m[14]!;

  const c00 = a11 * a22 - a12 * a21;
  const c01 = a02 * a21 - a01 * a22;
  const c02 = a01 * a12 - a02 * a11;
  const c10 = a12 * a20 - a10 * a22;
  const c11 = a00 * a22 - a02 * a20;
  const c12 = a02 * a10 - a00 * a12;
  const c20 = a10 * a21 - a11 * a20;
  const c21 = a01 * a20 - a00 * a21;
  const c22 = a00 * a11 - a01 * a10;
  const determinant = a00 * c00 + a01 * c10 + a02 * c20;
  const matrixScale = Math.max(
    Math.abs(a00), Math.abs(a01), Math.abs(a02),
    Math.abs(a10), Math.abs(a11), Math.abs(a12),
    Math.abs(a20), Math.abs(a21), Math.abs(a22),
  );
  const determinantFloor = Math.max(
    Number.MIN_VALUE,
    matrixScale ** 3 * 32 * 1.1920928955078125e-7,
  );
  if (
    !Number.isFinite(determinant) ||
    Math.abs(determinant) <= determinantFloor
  ) {
    throw new RangeError(
      `${label} has a singular camera transform and cannot define an eye position.`,
    );
  }

  const inverseDeterminant = 1 / determinant;
  const eye: Vec3 = [
    canonicalCameraComponent(
      -(c00 * tx + c01 * ty + c02 * tz) * inverseDeterminant,
    ),
    canonicalCameraComponent(
      -(c10 * tx + c11 * ty + c12 * tz) * inverseDeterminant,
    ),
    canonicalCameraComponent(
      -(c20 * tx + c21 * ty + c22 * tz) * inverseDeterminant,
    ),
  ];
  if (!eye.every(Number.isFinite)) {
    throw new RangeError(
      `${label} derives a non-finite camera position.`,
    );
  }
  return eye;
}

/**
 * Return the canonical eye position for a frame and validate the deprecated
 * redundant host value when present.
 */
export function resolveFrameCameraPosition(
  input: Pick<FrameInput, 'viewMatrix' | 'cameraPosition'>,
  label = 'FrameInput',
): Vec3 {
  const derived = deriveCameraPositionFromViewMatrix(
    input.viewMatrix,
    `${label}.viewMatrix`,
  );
  const provided = input.cameraPosition;
  if (provided === undefined) return derived;
  const providedLength = (provided as ArrayLike<number>).length;
  if (providedLength !== 3) {
    throw new RangeError(
      `${label}.cameraPosition must contain exactly 3 values (got ${providedLength}).`,
    );
  }
  for (let component = 0; component < 3; component += 1) {
    const actual = provided[component]!;
    if (typeof actual !== 'number' || !Number.isFinite(actual)) {
      throw new RangeError(
        `${label}.cameraPosition[${component}] must be finite (got ${String(actual)}).`,
      );
    }
    const expected = derived[component]!;
    const tolerance = Math.max(
      CAMERA_POSITION_ABSOLUTE_TOLERANCE,
      CAMERA_POSITION_RELATIVE_TOLERANCE *
        Math.max(1, Math.abs(actual), Math.abs(expected)),
    );
    if (Math.abs(actual - expected) > tolerance) {
      throw new RangeError(
        `${label}.cameraPosition disagrees with inverse(viewMatrix) at component ` +
          `${component}: provided ${actual}, derived ${expected}, tolerance ${tolerance}. ` +
          'Projection jitter and off-axis projection do not change the camera origin.',
      );
    }
  }
  return derived;
}

/**
 * Normalize a public frame payload to the canonical camera position while
 * preserving every other host-owned field.
 */
export function canonicalizeFrameCamera(
  input: FrameInput,
  label = 'FrameInput',
): CanonicalCameraFrameInput {
  return {
    ...input,
    cameraPosition: resolveFrameCameraPosition(input, label),
  };
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
  /**
   * Most-recent primary-hit normal + depth auxiliary.
   *
   * Float-RGBA precision is backend-defined. xyz stores a world-space unit
   * normal affine-encoded to [0,1] as `normal * 0.5 + 0.5`; decode with
   * `encoded * 2 - 1`. `abs(w)` stores linear distance from the primary camera
   * ray origin to its first accepted hit, in scene units; alpha-test
   * pass-through surfaces are not hits. 0 denotes no primary hit.
   * Backends may reserve the sign of w for surface classification, so consumers
   * needing geometric depth must use `abs(w)`. The conventional no-hit value
   * is `(0.5, 1.0, 0.5, 0.0)` (encoded world-up plus zero depth).
   */
  readonly normalDepth?: BackendTexture;

  /** Demodulated albedo (base color × occlusion, no lighting). Used by OIDN
   *  and atrous-variance to denoise lighting independently from texture detail. */
  readonly albedo?: BackendTexture;

  // ── Optional metadata ──────────────────────────────────────────────────
  /** Per-pixel scalar variance estimate (Welford running variance). Used by
   *  adaptive sampling and by some denoisers. The value is stored in `.r`;
   *  texture precision/format is backend-defined. */
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
