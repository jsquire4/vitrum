/** Default peak budget for pt-webgl2 frame-sized GPU render targets. */
export const DEFAULT_RENDER_TARGET_BUDGET_BYTES = 512 * 1024 * 1024;

/** Two RGBA32F progressive-history colors. */
export const BLEND_RENDER_TARGET_BYTES_PER_PIXEL = 32;
/** One RGBA16F tonemapped presentation target. */
export const PRESENT_RENDER_TARGET_BYTES_PER_PIXEL = 8;
/** Full tier adds RGBA32F normal/depth plus RGBA16F albedo. */
export const AUXILIARY_RENDER_TARGET_BYTES_PER_PIXEL = 24;
/** Four RGBA32F NEE candidate attachments, allocated as a bounded row tile. */
export const NEE_CANDIDATE_BYTES_PER_PIXEL = 64;

/** Compact lite profile: two histories plus the presentation target. */
export const BASE_RENDER_TARGET_BYTES_PER_PIXEL =
  BLEND_RENDER_TARGET_BYTES_PER_PIXEL + PRESENT_RENDER_TARGET_BYTES_PER_PIXEL;
/** Compact full profile: lite plus the shared last-sample auxiliary pair. */
export const AUX_RENDER_TARGET_BYTES_PER_PIXEL =
  BASE_RENDER_TARGET_BYTES_PER_PIXEL + AUXILIARY_RENDER_TARGET_BYTES_PER_PIXEL;

/** New resize transaction before its lazy NEE candidate tile exists. */
export const BASE_ALLOCATION_BYTES_PER_PIXEL = BASE_RENDER_TARGET_BYTES_PER_PIXEL;
/** New full-tier resize transaction before its lazy NEE candidate tile exists. */
export const AUX_ALLOCATION_BYTES_PER_PIXEL = AUX_RENDER_TARGET_BYTES_PER_PIXEL;

/**
 * OIDN reuses the inactive RGBA32F progressive-history texture. It does not add
 * another frame-sized allocation.
 */
export const DENOISED_RENDER_TARGET_BYTES_PER_PIXEL = 0;

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

function checkedAdd(label: string, a: number, b: number): number {
  if (a > Number.MAX_SAFE_INTEGER - b) {
    throw new RangeError(`pt-webgl2: ${label} overflows Number.MAX_SAFE_INTEGER`);
  }
  return a + b;
}

function estimatePixelBytes(
  width: number,
  height: number,
  bytesPerPixel: number,
): number {
  assertPositiveSafeInteger('pt-webgl2: render-target width', width);
  assertPositiveSafeInteger('pt-webgl2: render-target height', height);
  assertNonNegativeSafeInteger('pt-webgl2: render-target bytes/pixel', bytesPerPixel);
  if (width > Math.floor(Number.MAX_SAFE_INTEGER / height)) {
    throw new RangeError(
      `pt-webgl2: render-target pixel count overflows Number.MAX_SAFE_INTEGER ` +
        `(${width}×${height})`,
    );
  }
  const pixels = width * height;
  if (bytesPerPixel !== 0 && pixels > Math.floor(Number.MAX_SAFE_INTEGER / bytesPerPixel)) {
    throw new RangeError(
      `pt-webgl2: render-target byte estimate overflows Number.MAX_SAFE_INTEGER ` +
        `(${width}×${height} at ${bytesPerPixel} bytes/pixel)`,
    );
  }
  return pixels * bytesPerPixel;
}

/** Persistent compact layout, excluding the bounded lazy NEE scratch tile. */
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

/** Bytes allocated by a new resize transaction before lazy NEE scratch. */
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

export function estimateWebGl2NeeCandidateBytes(width: number, rows: number): number {
  return estimatePixelBytes(width, rows, NEE_CANDIDATE_BYTES_PER_PIXEL);
}

/**
 * Choose the largest candidate tile that stays inside the exact active budget.
 * At least one row is required; callers receive an actionable rejection before
 * any GL allocation when even that minimum cannot coexist with the base layout.
 */
export function selectWebGl2NeeCandidateRows(
  width: number,
  height: number,
  withAuxBuffers: boolean,
  maxBytes: number,
): number {
  assertPositiveSafeInteger('pt-webgl2: maxRenderTargetBytes', maxBytes);
  const base = estimateWebGl2RenderTargetBytes(width, height, withAuxBuffers);
  const rowBytes = estimateWebGl2NeeCandidateBytes(width, 1);
  if (base > maxBytes || maxBytes - base < rowBytes) {
    throw new RangeError(
      `pt-webgl2: render target ${width}×${height} requires ${base} persistent bytes plus ` +
        `${rowBytes} bytes for one NEE candidate row, exceeding ` +
        `maxRenderTargetBytes=${maxBytes}. Reduce viewport dimensions/resolutionFactor ` +
        `or raise the explicit budget.`,
    );
  }
  return Math.min(height, Math.floor((maxBytes - base) / rowBytes));
}

export interface WebGl2ResidentTargetState {
  /** Number of resident NEE scratch rows; zero before the first draw. */
  readonly candidateRows?: number;
  /** @deprecated Historical state key; no additional blend allocation exists. */
  readonly blend?: boolean;
  /** OIDN aliases an inactive history color and consumes no additional bytes. */
  readonly denoised?: boolean;
}

/** Exact currently-owned frame-target bytes for one published size. */
export function estimateWebGl2ResidentBytes(
  width: number,
  height: number,
  withAuxBuffers: boolean,
  state: WebGl2ResidentTargetState,
): number {
  const base = estimateWebGl2RenderTargetBytes(width, height, withAuxBuffers);
  const rows = state.candidateRows ?? 0;
  assertNonNegativeSafeInteger('pt-webgl2: resident NEE candidate rows', rows);
  if (rows > height) {
    throw new RangeError(
      `pt-webgl2: resident NEE candidate rows ${rows} exceed target height ${height}`,
    );
  }
  return checkedAdd(
    'resident render-target byte estimate',
    base,
    rows === 0 ? 0 : estimateWebGl2NeeCandidateBytes(width, rows),
  );
}

/** OIDN aliases an inactive history texture, so this is always zero. */
export function estimateWebGl2DenoisedTargetBytes(width: number, height: number): number {
  // Still validate dimensions so the public estimator remains fail-closed.
  estimatePixelBytes(width, height, 0);
  return 0;
}

/**
 * Validate requested persistent storage, the minimum one-row NEE scratch, and,
 * on resize, the exact overlap while the previous published targets remain
 * alive. The candidate scratch from the previous size is retired before this
 * overlap and is therefore deliberately absent from concurrentResidentBytes.
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
  if (width > maxTextureSize || height > maxTextureSize) {
    throw new RangeError(
      `pt-webgl2: render target ${width}×${height} exceeds MAX_TEXTURE_SIZE=${maxTextureSize}`,
    );
  }

  const requiredBytes = estimateWebGl2RenderTargetBytes(width, height, withAuxBuffers);
  const candidateRows = selectWebGl2NeeCandidateRows(
    width,
    height,
    withAuxBuffers,
    maxBytes,
  );
  const minimumActiveBytes = checkedAdd(
    'minimum active render-target byte estimate',
    requiredBytes,
    estimateWebGl2NeeCandidateBytes(width, Math.min(candidateRows, 1)),
  );

  if (concurrentResidentBytes > 0) {
    const replacementPeakBytes = checkedAdd(
      'render-target replacement peak',
      concurrentResidentBytes,
      requiredBytes,
    );
    if (replacementPeakBytes > maxBytes) {
      throw new RangeError(
        `pt-webgl2: render-target replacement peak requires ${replacementPeakBytes} bytes ` +
          `(${concurrentResidentBytes} resident + ${requiredBytes} new), exceeding ` +
          `maxRenderTargetBytes=${maxBytes}. The previous complete frame remains active.`,
      );
    }
  }
  return minimumActiveBytes;
}
