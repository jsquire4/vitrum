/**
 * IEEE 754 binary16 ↔ binary32 conversion (no deps).
 * Used to upload/read rgba16float atrous-variance ping-pong textures from CPU.
 */

const FLOAT32_SCRATCH = new Float32Array(1);
const UINT32_SCRATCH = new Uint32Array(FLOAT32_SCRATCH.buffer);

/**
 * Pack a JavaScript number into IEEE-754 binary16 bits.
 *
 * The input is rounded to binary32 first, matching a GPU `f32 -> f16`
 * conversion, then rounded to binary16 with round-to-nearest, ties-to-even.
 * Finite overflow maps to signed infinity. NaNs are canonicalised to a quiet
 * half NaN while preserving the binary32 sign lane.
 */
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
    // Values below half the least positive binary16 subnormal round to zero.
    // At exactly half, ties-to-even also selects zero.
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

    // A rounded subnormal can carry into the minimum normal (0x0400).
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

/** Expand binary16 bits to float32. */
export function float16BitsToFloat32(halfBits: number): number {
  const h = halfBits & 0xffff;
  const s = (h >> 15) & 0x1;
  const e = (h >> 10) & 0x1f;
  const m = h & 0x03ff;
  const sign = s !== 0 ? -1 : 1;
  if (e === 0) {
    return sign * Math.pow(2, -14) * (m / 1024);
  }
  if (e === 31) {
    return m !== 0 ? Number.NaN : sign * Number.POSITIVE_INFINITY;
  }
  return sign * Math.pow(2, e - 15) * (1 + m / 1024);
}
