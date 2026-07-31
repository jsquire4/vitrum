import { alignedTextureCopyBytesPerRow } from './webGpuTextureCopy.js';
import { float32ToFloat16Bits } from './halfFloat.js';

export interface FiniteNumberBounds {
  readonly min?: number;
  readonly max?: number;
  readonly integer?: boolean;
}

/** Validate dimensions before device acquisition or any GPU allocation. */
export function assertOneShotDimensions(
  label: string,
  width: number,
  height: number,
): number {
  if (!Number.isSafeInteger(width) || width <= 0) {
    throw new Error(`${label}: width must be a positive safe integer; received ${String(width)}`);
  }
  if (!Number.isSafeInteger(height) || height <= 0) {
    throw new Error(`${label}: height must be a positive safe integer; received ${String(height)}`);
  }
  const pixelCount = width * height;
  if (
    !Number.isSafeInteger(pixelCount) ||
    pixelCount > Math.floor(Number.MAX_SAFE_INTEGER / 4)
  ) {
    throw new Error(`${label}: width * height * channelCount exceeds the safe integer range`);
  }
  return pixelCount;
}

/** Validate an input slice before allocating resources or uploading any bytes. */
export function assertOneShotArrayLength(
  label: string,
  name: string,
  value: ArrayLike<unknown>,
  requiredLength: number,
): void {
  if (value.length < requiredLength) {
    throw new Error(
      `${label}: ${name} length must be >= ${requiredLength}; received ${value.length}`,
    );
  }
}

/** Reject NaN/Infinity in caller-controlled float data before GPU upload. */
export function assertFiniteFloatSlice(
  label: string,
  name: string,
  value: ArrayLike<number>,
  usedLength: number,
): void {
  for (let i = 0; i < usedLength; i += 1) {
    if (!Number.isFinite(value[i])) {
      throw new Error(`${label}: ${name}[${i}] must be finite; received ${String(value[i])}`);
    }
  }
}

/**
 * Reject values that are finite in float32 but become NaN or infinity when
 * stored in a binary16 texture.
 */
export function assertFiniteFloat16Slice(
  label: string,
  name: string,
  value: ArrayLike<number>,
  usedLength: number,
): void {
  assertFiniteFloatSlice(label, name, value, usedLength);
  for (let i = 0; i < usedLength; i += 1) {
    const halfBits = float32ToFloat16Bits(value[i]!);
    if ((halfBits & 0x7c00) === 0x7c00) {
      throw new Error(
        `${label}: ${name}[${i}] must be representable as finite float16; ` +
        `received ${String(value[i])}`,
      );
    }
  }
}

/** Validate a scalar tuning value with optional inclusive bounds. */
export function assertFiniteNumber(
  label: string,
  name: string,
  value: number,
  bounds: FiniteNumberBounds = {},
): void {
  if (!Number.isFinite(value)) {
    throw new Error(`${label}: ${name} must be finite; received ${String(value)}`);
  }
  if (bounds.integer === true && !Number.isInteger(value)) {
    throw new Error(`${label}: ${name} must be an integer; received ${String(value)}`);
  }
  if (bounds.min !== undefined && value < bounds.min) {
    throw new Error(`${label}: ${name} must be >= ${bounds.min}; received ${value}`);
  }
  if (bounds.max !== undefined && value > bounds.max) {
    throw new Error(`${label}: ${name} must be <= ${bounds.max}; received ${value}`);
  }
}

/**
 * Validate limits known only after selecting the actual GPU device. The caller
 * must place this inside the acquired-device try/finally so an ephemeral device
 * is still destroyed when preflight fails.
 */
export function assertOneShotDeviceLimits(
  device: GPUDevice,
  label: string,
  width: number,
  height: number,
  largestReadbackBytesPerPixel: number,
): void {
  const limits = device.limits;
  const maxTextureDimension2D = limits?.maxTextureDimension2D;
  if (
    typeof maxTextureDimension2D === 'number' &&
    (width > maxTextureDimension2D || height > maxTextureDimension2D)
  ) {
    throw new Error(
      `${label}: ${width}x${height} exceeds device maxTextureDimension2D ` +
      `(${maxTextureDimension2D})`,
    );
  }

  const bytesPerRow = alignedTextureCopyBytesPerRow(width, largestReadbackBytesPerPixel);
  const readbackBytes = bytesPerRow * height;
  const maxBufferSize = limits?.maxBufferSize;
  if (typeof maxBufferSize === 'number' && readbackBytes > maxBufferSize) {
    throw new Error(
      `${label}: readback buffer requires ${readbackBytes} bytes, exceeding device ` +
      `maxBufferSize (${maxBufferSize})`,
    );
  }
}
