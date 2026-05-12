/**
 * IEEE 754 binary16 ↔ binary32 conversion (no deps).
 * Used to upload/read rgba16float SVGF ping-pong textures from CPU.
 */

/** Pack float32 into binary16 bits (uint16). */
export function float32ToFloat16Bits(value: number): number {
  const floatView = new Float32Array(1);
  const int32View = new Int32Array(floatView.buffer);
  floatView[0] = value;
  const x = int32View[0]!;

  if ((x & 0x7fff_ffff) === 0) {
    return ((x >> 16) & 0x8000) >>> 0;
  }

  const sign = (x >> 16) & 0x8000;
  let exponent = ((x >> 23) & 0xff) - 127 + 15;
  let mantissa = x & 0x007f_ffff;

  if (exponent <= 0) {
    if (exponent < -10) {
      return (sign >>> 0);
    }
    mantissa |= 0x0080_0000;
    const shift = 14 - exponent;
    const halfMantissa = mantissa >> (shift + 13);
    const roundBit = (mantissa >> (shift + 12)) & 1;
    return ((sign | (halfMantissa + roundBit)) & 0xffff) >>> 0;
  }

  if (exponent >= 31) {
    // Distinguish: was the f32 input already Inf/NaN (exp bits all-ones),
    // or is this a finite f32 that overflows the fp16 range?
    const expBits = (x & 0x7f80_0000) >>> 23; // biased f32 exponent
    if (expBits === 0xff) {
      // Input was Inf or NaN — preserve the NaN-vs-Inf distinction.
      return ((sign | 0x7c00 | (mantissa !== 0 ? 0x0200 : 0)) & 0xffff) >>> 0;
    }
    // Input was finite but > 65504 — IEEE specifies saturation to ±Inf.
    return ((sign | 0x7c00) & 0xffff) >>> 0;
  }

  return ((sign | (exponent << 10) | (mantissa >> 13)) & 0xffff) >>> 0;
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
