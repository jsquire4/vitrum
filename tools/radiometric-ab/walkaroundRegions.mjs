// @ts-check

const REFERENCE_WIDTH = 128;
const REFERENCE_HEIGHT = 128;
const REGION_KEYS = /** @type {const} */ (['x0', 'y0', 'x1', 'y1']);

/** @param {[number, number, number, number]} bounds */
function normalizedReferenceRegion(bounds) {
  return Object.freeze({
    x0: bounds[0] / REFERENCE_WIDTH,
    y0: bounds[1] / REFERENCE_HEIGHT,
    x1: bounds[2] / REFERENCE_WIDTH,
    y1: bounds[3] / REFERENCE_HEIGHT,
  });
}

export const WALKAROUND_NORMALIZED_REGIONS = Object.freeze({
  a8Floor: normalizedReferenceRegion([10, 85, 118, 127]),
  a8Ceiling: normalizedReferenceRegion([10, 0, 118, 20]),
  a8LeftWall: normalizedReferenceRegion([0, 20, 20, 108]),
  a8RightWall: normalizedReferenceRegion([108, 20, 128, 108]),
  sunReceiver: normalizedReferenceRegion([30, 42, 98, 86]),
  sunSideDiagnostic: normalizedReferenceRegion([0, 30, 15, 98]),
  glassCenter: normalizedReferenceRegion([48, 48, 80, 80]),
  glossyBackWall: normalizedReferenceRegion([32, 32, 96, 96]),
});

/**
 * Resolve a normalized half-open region against the dimensions of the actual
 * captured texture.
 *
 * @param {{ x0: number, y0: number, x1: number, y1: number }} normalized
 * @param {number} width
 * @param {number} height
 */
export function resolveNormalizedRegion(normalized, width, height) {
  if (!Number.isSafeInteger(width) || width <= 0 || !Number.isSafeInteger(height) || height <= 0) {
    throw new RangeError('walkaround region dimensions must be positive safe integers');
  }
  for (const key of REGION_KEYS) {
    const value = normalized?.[key];
    if (typeof value !== 'number' || !Number.isFinite(value) || value < 0 || value > 1) {
      throw new RangeError(`walkaround normalized region ${key} must be finite and in [0, 1]`);
    }
  }
  if (normalized.x0 >= normalized.x1 || normalized.y0 >= normalized.y1) {
    throw new RangeError('walkaround normalized region must have positive area');
  }

  const x0 = Math.min(width - 1, Math.floor(normalized.x0 * width));
  const y0 = Math.min(height - 1, Math.floor(normalized.y0 * height));
  const x1 = Math.max(x0 + 1, Math.min(width, Math.ceil(normalized.x1 * width)));
  const y1 = Math.max(y0 + 1, Math.min(height, Math.ceil(normalized.y1 * height)));
  return { x0, y0, x1, y1 };
}

/** @param {number} width @param {number} height */
export function resolveWalkaroundRegions(width, height) {
  return Object.fromEntries(
    Object.entries(WALKAROUND_NORMALIZED_REGIONS).map(([name, region]) => [
      name,
      resolveNormalizedRegion(region, width, height),
    ]),
  );
}

/**
 * @param {number} referenceX
 * @param {number} referenceY
 * @param {number} width
 * @param {number} height
 */
export function resolveReferencePoint(referenceX, referenceY, width, height) {
  if (!Number.isFinite(referenceX) || !Number.isFinite(referenceY)) {
    throw new RangeError('walkaround reference point must be finite');
  }
  if (!Number.isSafeInteger(width) || width <= 0 || !Number.isSafeInteger(height) || height <= 0) {
    throw new RangeError('walkaround point dimensions must be positive safe integers');
  }
  return {
    x: Math.max(0, Math.min(width - 1, Math.floor((referenceX / REFERENCE_WIDTH) * width))),
    y: Math.max(0, Math.min(height - 1, Math.floor((referenceY / REFERENCE_HEIGHT) * height))),
  };
}

/**
 * Validate the complete linear RGBA capture before any regional statistic is
 * allowed to classify it. Checking all four components matters: a non-finite
 * value outside the selected ROI, or in alpha, still means the renderer emitted
 * an invalid frame and must not be hidden by a finite local average.
 *
 * @param {ArrayLike<number>} pixels
 * @param {number} width
 * @param {number} height
 * @param {string} [label]
 */
export function validateWalkaroundPixelBuffer(pixels, width, height, label = 'walkaround capture') {
  if (!Number.isSafeInteger(width) || width <= 0 || !Number.isSafeInteger(height) || height <= 0) {
    throw new RangeError(`${label} dimensions must be positive safe integers`);
  }
  const expectedLength = width * height * 4;
  if (!Number.isSafeInteger(expectedLength)) {
    throw new RangeError(`${label} dimensions exceed the safe RGBA buffer range`);
  }
  if (pixels == null || pixels.length !== expectedLength) {
    throw new RangeError(`${label} pixel buffer length must be exactly ${expectedLength}`);
  }

  const componentNames = ['r', 'g', 'b', 'a'];
  for (let i = 0; i < expectedLength; i += 1) {
    const value = pixels[i];
    if (typeof value !== 'number' || !Number.isFinite(value)) {
      const pixelIndex = Math.floor(i / 4);
      const x = pixelIndex % width;
      const y = Math.floor(pixelIndex / width);
      throw new Error(
        `${label} contains a non-finite ${componentNames[i % 4]} component at (${x}, ${y})`,
      );
    }
  }
}

/**
 * @param {ArrayLike<number>} pixels
 * @param {number} width
 * @param {number} height
 * @param {{ x0: number, y0: number, x1: number, y1: number }} region
 */
export function regionLuminance(pixels, width, height, region) {
  if (pixels == null || pixels.length !== width * height * 4) {
    throw new RangeError(`walkaround pixel buffer length must be exactly ${width * height * 4}`);
  }
  for (const key of REGION_KEYS) {
    if (!Number.isSafeInteger(region?.[key])) {
      throw new RangeError(`walkaround region ${key} must be a safe integer`);
    }
  }
  if (
    region.x0 < 0 ||
    region.y0 < 0 ||
    region.x0 >= region.x1 ||
    region.y0 >= region.y1 ||
    region.x1 > width ||
    region.y1 > height
  ) {
    throw new RangeError('walkaround region exceeds the actual captured texture');
  }

  let sum = 0;
  let count = 0;
  for (let y = region.y0; y < region.y1; y += 1) {
    for (let x = region.x0; x < region.x1; x += 1) {
      const i = (y * width + x) * 4;
      const luminance = 0.2126 * pixels[i] + 0.7152 * pixels[i + 1] + 0.0722 * pixels[i + 2];
      if (!Number.isFinite(luminance)) {
        throw new Error(`walkaround region contains a non-finite pixel at (${x}, ${y})`);
      }
      sum += luminance;
      count += 1;
    }
  }
  return sum / count;
}
