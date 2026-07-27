/** Default peak budget for pt-webgl2 frame-sized GPU render targets. */
export const DEFAULT_RENDER_TARGET_BUDGET_BYTES = 512 * 1024 * 1024;

/**
 * Steady-state base layout: accumulator (16 B), NEE candidates (64 B),
 * running-mean blend pair (32 B), and present target (16 B).
 */
export const BASE_RENDER_TARGET_BYTES_PER_PIXEL = 128;

/** Base steady state plus normal/depth and albedo RGBA32F targets (32 B). */
export const AUX_RENDER_TARGET_BYTES_PER_PIXEL = 160;

/** New resize transaction before its lazy blend pair exists. */
export const BASE_ALLOCATION_BYTES_PER_PIXEL = 96;

/** New full-tier resize transaction before its lazy blend pair exists. */
export const AUX_ALLOCATION_BYTES_PER_PIXEL = 128;

/** Two RGBA32F running-mean targets allocated on the first draw. */
export const BLEND_RENDER_TARGET_BYTES_PER_PIXEL = 32;

/** One RGBA32F accepted OIDN result target. */
export const DENOISED_RENDER_TARGET_BYTES_PER_PIXEL = 16;

function assertPositiveSafeInteger(label: string, value: number): void {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new RangeError(`${label} must be a positive safe integer (got ${String(value)})`);
  }
}

function assertNonNegativeSafeInteger(label: string, value: number): void {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new RangeError(
      `${label} must be a non-negative safe integer (got ${String(value)})`,
    );
  }
}

function estimatePixelBytes(
  width: number,
  height: number,
  bytesPerPixel: number,
): number {
  assertPositiveSafeInteger('pt-webgl2: render-target width', width);
  assertPositiveSafeInteger('pt-webgl2: render-target height', height);
  if (width > Math.floor(Number.MAX_SAFE_INTEGER / height)) {
    throw new RangeError(
      `pt-webgl2: render-target pixel count overflows Number.MAX_SAFE_INTEGER ` +
        `(${width}×${height})`,
    );
  }
  const pixels = width * height;
  if (pixels > Math.floor(Number.MAX_SAFE_INTEGER / bytesPerPixel)) {
    throw new RangeError(
      `pt-webgl2: render-target byte estimate overflows Number.MAX_SAFE_INTEGER ` +
        `(${width}×${height} at ${bytesPerPixel} bytes/pixel)`,
    );
  }
  return pixels * bytesPerPixel;
}

/** Full steady-state bytes after the first draw has allocated the blend pair. */
export function estimateWebGl2RenderTargetBytes(
  width: number,
  height: number,
  withAuxBuffers: boolean,
): number {
  return estimatePixelBytes(
    width,
    height,
    withAuxBuffers
      ? AUX_RENDER_TARGET_BYTES_PER_PIXEL
      : BASE_RENDER_TARGET_BYTES_PER_PIXEL,
  );
}

/** Bytes allocated by a new accumulation/candidate/present resize transaction. */
export function estimateWebGl2AllocationBytes(
  width: number,
  height: number,
  withAuxBuffers: boolean,
): number {
  return estimatePixelBytes(
    width,
    height,
    withAuxBuffers
      ? AUX_ALLOCATION_BYTES_PER_PIXEL
      : BASE_ALLOCATION_BYTES_PER_PIXEL,
  );
}

export interface WebGl2ResidentTargetState {
  readonly blend: boolean;
  readonly denoised: boolean;
}

/** Exact currently-owned frame-target bytes for one published size. */
export function estimateWebGl2ResidentBytes(
  width: number,
  height: number,
  withAuxBuffers: boolean,
  state: WebGl2ResidentTargetState,
): number {
  const base = estimateWebGl2AllocationBytes(width, height, withAuxBuffers);
  const optionalBytesPerPixel =
    (state.blend ? BLEND_RENDER_TARGET_BYTES_PER_PIXEL : 0) +
    (state.denoised ? DENOISED_RENDER_TARGET_BYTES_PER_PIXEL : 0);
  if (optionalBytesPerPixel === 0) return base;
  const optional = estimatePixelBytes(width, height, optionalBytesPerPixel);
  if (base > Number.MAX_SAFE_INTEGER - optional) {
    throw new RangeError(
      'pt-webgl2: resident render-target byte estimate overflows Number.MAX_SAFE_INTEGER',
    );
  }
  return base + optional;
}

export function estimateWebGl2DenoisedTargetBytes(width: number, height: number): number {
  return estimatePixelBytes(width, height, DENOISED_RENDER_TARGET_BYTES_PER_PIXEL);
}

/**
 * Validate both the requested steady state and, on resize, the transaction
 * peak while the previous published targets remain alive.
 */
export function assertWebGl2RenderTargetRequest(
  width: number,
  height: number,
  withAuxBuffers: boolean,
  maxBytes: number,
  maxTextureSize: number,
  concurrentResidentBytes = 0,
): number {
  assertPositiveSafeInteger('pt-webgl2: maxRenderTargetBytes', maxBytes);
  assertPositiveSafeInteger('pt-webgl2: MAX_TEXTURE_SIZE', maxTextureSize);
  assertNonNegativeSafeInteger(
    'pt-webgl2: concurrent resident render-target bytes',
    concurrentResidentBytes,
  );
  const requiredBytes = estimateWebGl2RenderTargetBytes(width, height, withAuxBuffers);
  if (width > maxTextureSize || height > maxTextureSize) {
    throw new RangeError(
      `pt-webgl2: render target ${width}×${height} exceeds MAX_TEXTURE_SIZE=${maxTextureSize}`,
    );
  }
  if (requiredBytes > maxBytes) {
    const bytesPerPixel = withAuxBuffers
      ? AUX_RENDER_TARGET_BYTES_PER_PIXEL
      : BASE_RENDER_TARGET_BYTES_PER_PIXEL;
    throw new RangeError(
      `pt-webgl2: render target ${width}×${height} requires ${requiredBytes} bytes ` +
        `(${bytesPerPixel} bytes/pixel), exceeding maxRenderTargetBytes=${maxBytes}. ` +
        `Reduce viewport dimensions/resolutionFactor or raise the explicit budget.`,
    );
  }

  if (concurrentResidentBytes > 0) {
    const nextAllocationBytes = estimateWebGl2AllocationBytes(
      width,
      height,
      withAuxBuffers,
    );
    if (concurrentResidentBytes > Number.MAX_SAFE_INTEGER - nextAllocationBytes) {
      throw new RangeError(
        'pt-webgl2: render-target replacement peak overflows Number.MAX_SAFE_INTEGER',
      );
    }
    const replacementPeakBytes = concurrentResidentBytes + nextAllocationBytes;
    if (replacementPeakBytes > maxBytes) {
      throw new RangeError(
        `pt-webgl2: render-target replacement peak requires ${replacementPeakBytes} bytes ` +
          `(${concurrentResidentBytes} resident + ${nextAllocationBytes} new), exceeding ` +
          `maxRenderTargetBytes=${maxBytes}. The previous complete frame remains active.`,
      );
    }
  }
  return requiredBytes;
}
