/**
 * IEEE-754 binary16 helpers used by the RGBA16F material-radiance atlas.
 *
 * Conversion rounds through binary32 and then uses round-to-nearest,
 * ties-to-even, matching WebGL's f32 -> f16 upload semantics.
 */

const FLOAT32_SCRATCH = new Float32Array(1);
const UINT32_SCRATCH = new Uint32Array(FLOAT32_SCRATCH.buffer);

export const FLOAT16_MAX_FINITE = 65_504;
export const FLOAT16_MIN_SUBNORMAL = 2 ** -24;
export const FLOAT16_HALF_MIN_SUBNORMAL = 2 ** -25;

export function float32ToFloat16Bits(value: number): number {
  FLOAT32_SCRATCH[0] = value;
  const bits = UINT32_SCRATCH[0]!;
  const sign = (bits >>> 16) & 0x8000;
  const exponent = (bits >>> 23) & 0xff;
  const mantissa = bits & 0x007f_ffff;

  if (exponent === 0xff) {
    return sign | (mantissa === 0 ? 0x7c00 : 0x7e00);
  }

  let halfExponent = exponent - 127 + 15;
  if (halfExponent >= 31) return sign | 0x7c00;

  if (halfExponent <= 0) {
    if (halfExponent < -10) return sign;
    const significand = mantissa | 0x0080_0000;
    const shift = 14 - halfExponent;
    const divisor = 2 ** shift;
    let halfMantissa = Math.floor(significand / divisor);
    const remainder = significand - halfMantissa * divisor;
    const halfway = divisor / 2;
    if (
      remainder > halfway ||
      (remainder === halfway && (halfMantissa & 1) !== 0)
    ) {
      halfMantissa += 1;
    }
    return sign | halfMantissa;
  }

  let halfMantissa = mantissa >>> 13;
  const remainder = mantissa & 0x1fff;
  if (
    remainder > 0x1000 ||
    (remainder === 0x1000 && (halfMantissa & 1) !== 0)
  ) {
    halfMantissa += 1;
    if (halfMantissa === 0x0400) {
      halfMantissa = 0;
      halfExponent += 1;
      if (halfExponent >= 31) return sign | 0x7c00;
    }
  }

  return sign | (halfExponent << 10) | halfMantissa;
}

export function float16BitsToFloat32(halfBits: number): number {
  const h = halfBits & 0xffff;
  const sign = (h & 0x8000) !== 0 ? -1 : 1;
  const exponent = (h >>> 10) & 0x1f;
  const mantissa = h & 0x03ff;
  if (exponent === 0) return sign * 2 ** -14 * (mantissa / 1024);
  if (exponent === 0x1f) {
    return mantissa !== 0 ? Number.NaN : sign * Number.POSITIVE_INFINITY;
  }
  return sign * 2 ** (exponent - 15) * (1 + mantissa / 1024);
}

/**
 * Encode a finite value that must remain finite and nonzero when stored as
 * binary16. Exact zero (including authored -0) is valid; a finite nonzero value
 * that rounds to signed zero is rejected so CPU-side light distributions cannot
 * retain energy that the RGBA16F shader texture loses.
 */
export function finiteFloat16Bits(
  value: number,
  context = 'value',
): number {
  if (!Number.isFinite(value)) {
    throw new RangeError(
      `${context} must be finite for RGBA16F storage (received ${String(value)}).`,
    );
  }
  if (Math.abs(value) > FLOAT16_MAX_FINITE) {
    throw new RangeError(
      `${context} ${String(value)} exceeds the finite RGBA16F range ` +
        `(±${FLOAT16_MAX_FINITE}).`,
    );
  }
  const bits = float32ToFloat16Bits(value);
  const quantized = float16BitsToFloat32(bits);
  if (!Number.isFinite(quantized)) {
    throw new RangeError(
      `${context} ${String(value)} overflows finite RGBA16F storage.`,
    );
  }
  if (value !== 0 && quantized === 0) {
    const signedZero = Object.is(quantized, -0) ? '-0' : '+0';
    throw new RangeError(
      `${context} ${String(value)} is finite and nonzero but underflows to ` +
        `${signedZero} in RGBA16F storage.`,
    );
  }
  return bits;
}

/** Promote the exact checked binary16 encoding back to a JavaScript number. */
export function quantizeFiniteFloat16(
  value: number,
  context = 'value',
): number {
  return float16BitsToFloat32(finiteFloat16Bits(value, context));
}
