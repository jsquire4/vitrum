import { sanitizeNeuralSigned } from './preprocessing.js';
import type { NeuralTensorPrecision } from './tensorPrecision.js';

const f32 = new Float32Array(1);
const u32 = new Uint32Array(f32.buffer);

/** IEEE-754 round-to-nearest-even binary32 -> binary16 conversion. */
export function float32ToFloat16Bits(value: number): number {
  f32[0] = sanitizeNeuralSigned(value);
  const bits = u32[0]!;
  const sign = (bits >>> 16) & 0x8000;
  const exponent = (bits >>> 23) & 0xff;
  const mantissa = bits & 0x7fffff;

  if (exponent === 0xff) return sign | (mantissa === 0 ? 0x7c00 : 0x7e00);
  const halfExponent = exponent - 127 + 15;
  if (halfExponent >= 31) return sign | 0x7bff;
  if (halfExponent <= 0) {
    if (halfExponent < -10) return sign;
    const normalizedMantissa = mantissa | 0x800000;
    const shift = 14 - halfExponent;
    let halfMantissa = normalizedMantissa >>> shift;
    const remainder = normalizedMantissa & ((1 << shift) - 1);
    const halfway = 1 << (shift - 1);
    if (remainder > halfway || (remainder === halfway && (halfMantissa & 1) !== 0)) {
      halfMantissa++;
    }
    return sign | halfMantissa;
  }

  let halfMantissa = mantissa >>> 13;
  const remainder = mantissa & 0x1fff;
  let roundedExponent = halfExponent;
  if (remainder > 0x1000 || (remainder === 0x1000 && (halfMantissa & 1) !== 0)) {
    halfMantissa++;
    if (halfMantissa === 0x400) {
      halfMantissa = 0;
      roundedExponent++;
      if (roundedExponent >= 31) return sign | 0x7bff;
    }
  }
  return sign | (roundedExponent << 10) | halfMantissa;
}

export function float16BitsToFloat32(bits: number): number {
  const sign = (bits & 0x8000) !== 0 ? -1 : 1;
  const exponent = (bits >>> 10) & 0x1f;
  const mantissa = bits & 0x3ff;
  if (exponent === 0) {
    return mantissa === 0 ? sign * 0 : sign * 2 ** -14 * (mantissa / 1024);
  }
  if (exponent === 0x1f) return mantissa === 0 ? sign * Infinity : NaN;
  return sign * 2 ** (exponent - 15) * (1 + mantissa / 1024);
}

export function roundNeuralTensorScalar(
  value: number,
  precision: NeuralTensorPrecision,
): number {
  const finite = sanitizeNeuralSigned(value);
  return precision === 'f16'
    ? float16BitsToFloat32(float32ToFloat16Bits(finite))
    : Math.fround(finite);
}

export function encodeNeuralTensor(
  values: Float32Array,
  precision: NeuralTensorPrecision,
): Float32Array | Uint16Array {
  if (precision === 'f32') return values;
  const encoded = new Uint16Array(values.length);
  for (let i = 0; i < values.length; i++) encoded[i] = float32ToFloat16Bits(values[i]!);
  return encoded;
}

export function decodeNeuralTensor(
  values: Float32Array | Uint16Array,
  precision: NeuralTensorPrecision,
): Float32Array {
  if (precision === 'f32') return values instanceof Float32Array ? values : new Float32Array(values);
  const decoded = new Float32Array(values.length);
  for (let i = 0; i < values.length; i++) decoded[i] = float16BitsToFloat32(values[i]!);
  return decoded;
}
