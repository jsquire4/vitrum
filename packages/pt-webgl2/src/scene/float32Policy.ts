/** Largest finite IEEE-754 binary32 value. */
export const WEBGL2_F32_MAX = 3.4028234663852886e38;

/** Smallest positive normal IEEE-754 binary32 value. */
export const WEBGL2_F32_MIN_NORMAL = 1.1754943508222875e-38;

/**
 * Quantize one host number exactly as a WebGL `float`/RGBA32F upload does.
 * Finite JavaScript numbers outside binary32 are rejected instead of entering
 * the shader as infinity.
 */
export function requireFiniteFloat32(value: number, context: string): number {
  if (!Number.isFinite(value)) {
    throw new RangeError(`${context} must be finite.`);
  }
  const stored = Math.fround(value);
  if (!Number.isFinite(stored)) {
    throw new RangeError(`${context} overflows WebGL float32 storage.`);
  }
  return stored;
}

/**
 * Quantize a non-negative scalar while preserving authored positive support.
 * A positive value that rounds to zero would silently disable the associated
 * radiometric or attenuation term, so that case is an explicit boundary error.
 */
export function requireNonNegativeFloat32(
  value: number,
  context: string,
): number {
  if (!Number.isFinite(value) || value < 0) {
    throw new RangeError(`${context} must be finite and non-negative.`);
  }
  const stored = Math.fround(value);
  if (!Number.isFinite(stored)) {
    throw new RangeError(`${context} overflows WebGL float32 storage.`);
  }
  if (value > 0 && stored === 0) {
    throw new RangeError(`${context} underflows WebGL float32 storage.`);
  }
  return stored;
}

/**
 * Mirror a shader multiplication whose inputs have already crossed the f32
 * storage boundary. This catches a finite pair producing infinity or a
 * positive pair collapsing to zero before the packed light is accepted.
 */
export function multiplyNonNegativeFloat32(
  a: number,
  b: number,
  context: string,
): number {
  if (
    !Number.isFinite(a) ||
    !Number.isFinite(b) ||
    a < 0 ||
    b < 0
  ) {
    throw new RangeError(`${context} requires finite non-negative f32 operands.`);
  }
  const exact = a * b;
  const stored = Math.fround(exact);
  if (!Number.isFinite(exact) || !Number.isFinite(stored)) {
    throw new RangeError(`${context} overflows shader float32 multiplication.`);
  }
  if (a > 0 && b > 0 && stored === 0) {
    throw new RangeError(`${context} underflows shader float32 multiplication.`);
  }
  return stored;
}
