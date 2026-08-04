export const MATERIAL_ATLAS_ENCODING_RGBA8_UNORM = 0;
export const MATERIAL_ATLAS_ENCODING_RGBA8_SNORM = 1;
export const MATERIAL_ATLAS_ENCODING_RGBA16_FLOAT = 2;
export const MATERIAL_ATLAS_ENCODING_RGBA16_UNORM = 3;
export const MATERIAL_ATLAS_ENCODING_RGBA16_SNORM = 4;
export const MATERIAL_ATLAS_ENCODING_RGBA32_FLOAT = 5;

export type MaterialTextureAtlasEncoding =
  | typeof MATERIAL_ATLAS_ENCODING_RGBA8_UNORM
  | typeof MATERIAL_ATLAS_ENCODING_RGBA8_SNORM
  | typeof MATERIAL_ATLAS_ENCODING_RGBA16_FLOAT
  | typeof MATERIAL_ATLAS_ENCODING_RGBA16_UNORM
  | typeof MATERIAL_ATLAS_ENCODING_RGBA16_SNORM
  | typeof MATERIAL_ATLAS_ENCODING_RGBA32_FLOAT;

export function materialTextureAtlasEncodingPlaneCount(
  encoding: MaterialTextureAtlasEncoding,
): 1 | 2 | 4 {
  if (
    encoding === MATERIAL_ATLAS_ENCODING_RGBA8_UNORM ||
    encoding === MATERIAL_ATLAS_ENCODING_RGBA8_SNORM
  ) {
    return 1;
  }
  if (
    encoding === MATERIAL_ATLAS_ENCODING_RGBA16_FLOAT ||
    encoding === MATERIAL_ATLAS_ENCODING_RGBA16_UNORM ||
    encoding === MATERIAL_ATLAS_ENCODING_RGBA16_SNORM
  ) {
    return 2;
  }
  return 4;
}

export function materialTextureAtlasEncodingForDataType(
  dataType: 'uint8' | 'uint16' | 'float16' | 'half-float' | 'float32',
): MaterialTextureAtlasEncoding {
  switch (dataType) {
    case 'uint8':
      return MATERIAL_ATLAS_ENCODING_RGBA8_UNORM;
    case 'uint16':
      return MATERIAL_ATLAS_ENCODING_RGBA16_UNORM;
    case 'float16':
    case 'half-float':
      return MATERIAL_ATLAS_ENCODING_RGBA16_FLOAT;
    case 'float32':
      return MATERIAL_ATLAS_ENCODING_RGBA32_FLOAT;
  }
}

/**
 * Optional `texture-formats-tier1` formats are implemented by current WebGPU
 * runtimes even though Deno's bundled WebGPU declaration can lag the spec and
 * omit them from `GPUTextureFormat`.
 */
export type MaterialTextureAtlasGpuFormat =
  | GPUTextureFormat
  | 'rgba16unorm'
  | 'rgba16snorm';

export function materialTextureAtlasEncodingForGpuFormat(
  format: MaterialTextureAtlasGpuFormat,
): MaterialTextureAtlasEncoding {
  switch (format) {
    case 'r8unorm':
    case 'rg8unorm':
    case 'rgba8unorm':
    case 'rgba8unorm-srgb':
    case 'bgra8unorm':
    case 'bgra8unorm-srgb':
      return MATERIAL_ATLAS_ENCODING_RGBA8_UNORM;
    case 'r8snorm':
    case 'rg8snorm':
    case 'rgba8snorm':
      return MATERIAL_ATLAS_ENCODING_RGBA8_SNORM;
    case 'r16float':
    case 'rg16float':
    case 'rgba16float':
      return MATERIAL_ATLAS_ENCODING_RGBA16_FLOAT;
    case 'rgba16unorm':
      return MATERIAL_ATLAS_ENCODING_RGBA16_UNORM;
    case 'rgba16snorm':
      return MATERIAL_ATLAS_ENCODING_RGBA16_SNORM;
    case 'r32float':
    case 'rg32float':
    case 'rgba32float':
    case 'rg11b10ufloat':
    case 'rgb9e5ufloat':
    case 'rgb10a2unorm':
      // Packed-float and rgb10a2 sources expose decoded RGBA values. Keeping
      // those values in full-float planes avoids an unproven re-quantisation
      // step and preserves the current sampled-value contract exactly.
      return MATERIAL_ATLAS_ENCODING_RGBA32_FLOAT;
    default:
      throw new RangeError(`Unsupported material texture GPU format ${format}.`);
  }
}

const FLOAT32_BITS = new Float32Array(1);
const UINT32_BITS = new Uint32Array(FLOAT32_BITS.buffer);

function floatBits(value: number): number {
  FLOAT32_BITS[0] = value;
  return UINT32_BITS[0]!;
}

function bitsFloat(value: number): number {
  UINT32_BITS[0] = value >>> 0;
  return FLOAT32_BITS[0]!;
}

export function materialTextureAtlasFloatToHalf(value: number): number {
  FLOAT32_BITS[0] = value;
  const bits = UINT32_BITS[0]!;
  const sign = (bits >>> 16) & 0x8000;
  const exponent = (bits >>> 23) & 0xff;
  const mantissa = bits & 0x7fffff;
  if (exponent === 0xff) {
    return sign | (mantissa === 0 ? 0x7c00 : 0x7e00);
  }
  const unbiased = exponent - 127;
  if (unbiased > 15) return sign | 0x7c00;
  if (unbiased < -25) return sign;
  if (unbiased === -25) {
    // 2^-25 is exactly halfway between zero and the minimum half subnormal,
    // so ties-to-even selects zero. Every larger f32 in this exponent bin
    // rounds to 0x0001; flushing the whole bin loses a valid half interval.
    return sign | (mantissa === 0 ? 0 : 1);
  }
  if (unbiased < -14) {
    const shift = -unbiased - 14;
    const significand = mantissa | 0x800000;
    const round = (1 << (shift + 12)) - 1;
    return sign | ((significand + round + ((significand >>> (shift + 13)) & 1)) >>> (shift + 13));
  }
  let halfExponent = unbiased + 15;
  let halfMantissa = mantissa + 0xfff + ((mantissa >>> 13) & 1);
  if ((halfMantissa & 0x800000) !== 0) {
    halfMantissa = 0;
    halfExponent += 1;
    if (halfExponent >= 31) return sign | 0x7c00;
  }
  return sign | (halfExponent << 10) | (halfMantissa >>> 13);
}

export function materialTextureAtlasHalfToFloat(value: number): number {
  const sign = (value & 0x8000) !== 0 ? -1 : 1;
  const exponent = (value >>> 10) & 0x1f;
  const fraction = value & 0x03ff;
  if (exponent === 0) {
    return sign * 2 ** -14 * (fraction / 1024);
  }
  if (exponent === 0x1f) {
    return fraction === 0 ? sign * Number.POSITIVE_INFINITY : Number.NaN;
  }
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

function clamp(value: number, low: number, high: number): number {
  return Math.min(high, Math.max(low, value));
}

function packUnorm8(value: number): number {
  return Math.round(clamp(value, 0, 1) * 255) & 0xff;
}

function packSnorm8(value: number): number {
  return Math.round(clamp(value, -1, 1) * 127) & 0xff;
}

function packUnorm16(value: number): number {
  return Math.round(clamp(value, 0, 1) * 65535) & 0xffff;
}

function packSnorm16(value: number): number {
  return Math.round(clamp(value, -1, 1) * 32767) & 0xffff;
}

function signed8(value: number): number {
  const byte = value & 0xff;
  return byte >= 0x80 ? byte - 0x100 : byte;
}

function signed16(value: number): number {
  const word = value & 0xffff;
  return word >= 0x8000 ? word - 0x10000 : word;
}

function decodeMaterialTextureAtlasPixel(
  packed: Uint32Array,
  encoding: MaterialTextureAtlasEncoding,
  pixelCount: number,
  pixel: number,
  out: Float32Array,
): void {
  const p0 = packed[pixel]!;
  if (encoding === MATERIAL_ATLAS_ENCODING_RGBA8_UNORM) {
    out[0] = (p0 & 0xff) / 255;
    out[1] = ((p0 >>> 8) & 0xff) / 255;
    out[2] = ((p0 >>> 16) & 0xff) / 255;
    out[3] = ((p0 >>> 24) & 0xff) / 255;
    return;
  }
  if (encoding === MATERIAL_ATLAS_ENCODING_RGBA8_SNORM) {
    out[0] = Math.max(-1, signed8(p0) / 127);
    out[1] = Math.max(-1, signed8(p0 >>> 8) / 127);
    out[2] = Math.max(-1, signed8(p0 >>> 16) / 127);
    out[3] = Math.max(-1, signed8(p0 >>> 24) / 127);
    return;
  }
  const p1 = packed[pixelCount + pixel]!;
  if (encoding === MATERIAL_ATLAS_ENCODING_RGBA16_FLOAT) {
    out[0] = materialTextureAtlasHalfToFloat(p0 & 0xffff);
    out[1] = materialTextureAtlasHalfToFloat(p0 >>> 16);
    out[2] = materialTextureAtlasHalfToFloat(p1 & 0xffff);
    out[3] = materialTextureAtlasHalfToFloat(p1 >>> 16);
  } else if (encoding === MATERIAL_ATLAS_ENCODING_RGBA16_UNORM) {
    out[0] = (p0 & 0xffff) / 65535;
    out[1] = (p0 >>> 16) / 65535;
    out[2] = (p1 & 0xffff) / 65535;
    out[3] = (p1 >>> 16) / 65535;
  } else if (encoding === MATERIAL_ATLAS_ENCODING_RGBA16_SNORM) {
    out[0] = Math.max(-1, signed16(p0) / 32767);
    out[1] = Math.max(-1, signed16(p0 >>> 16) / 32767);
    out[2] = Math.max(-1, signed16(p1) / 32767);
    out[3] = Math.max(-1, signed16(p1 >>> 16) / 32767);
  } else {
    out[0] = bitsFloat(p0);
    out[1] = bitsFloat(p1);
    out[2] = bitsFloat(packed[pixelCount * 2 + pixel]!);
    out[3] = bitsFloat(packed[pixelCount * 3 + pixel]!);
  }
}

export function packMaterialTextureAtlasPixels(
  rgba: Float32Array,
  encoding: MaterialTextureAtlasEncoding,
): Uint32Array {
  if (rgba.length % 4 !== 0) {
    throw new RangeError('Material texture RGBA data length must be divisible by four.');
  }
  const pixelCount = rgba.length / 4;
  const planes = materialTextureAtlasEncodingPlaneCount(encoding);
  const packed = new Uint32Array(pixelCount * planes);
  for (let pixel = 0; pixel < pixelCount; pixel += 1) {
    const source = pixel * 4;
    const r = rgba[source]!;
    const g = rgba[source + 1]!;
    const b = rgba[source + 2]!;
    const a = rgba[source + 3]!;
    if (encoding === MATERIAL_ATLAS_ENCODING_RGBA8_UNORM) {
      packed[pixel] =
        packUnorm8(r) |
        (packUnorm8(g) << 8) |
        (packUnorm8(b) << 16) |
        (packUnorm8(a) << 24);
    } else if (encoding === MATERIAL_ATLAS_ENCODING_RGBA8_SNORM) {
      packed[pixel] =
        packSnorm8(r) |
        (packSnorm8(g) << 8) |
        (packSnorm8(b) << 16) |
        (packSnorm8(a) << 24);
    } else if (encoding === MATERIAL_ATLAS_ENCODING_RGBA16_FLOAT) {
      packed[pixel] =
        materialTextureAtlasFloatToHalf(r) |
        (materialTextureAtlasFloatToHalf(g) << 16);
      packed[pixelCount + pixel] =
        materialTextureAtlasFloatToHalf(b) |
        (materialTextureAtlasFloatToHalf(a) << 16);
    } else if (encoding === MATERIAL_ATLAS_ENCODING_RGBA16_UNORM) {
      packed[pixel] = packUnorm16(r) | (packUnorm16(g) << 16);
      packed[pixelCount + pixel] = packUnorm16(b) | (packUnorm16(a) << 16);
    } else if (encoding === MATERIAL_ATLAS_ENCODING_RGBA16_SNORM) {
      packed[pixel] = packSnorm16(r) | (packSnorm16(g) << 16);
      packed[pixelCount + pixel] = packSnorm16(b) | (packSnorm16(a) << 16);
    } else {
      packed[pixel] = floatBits(r);
      packed[pixelCount + pixel] = floatBits(g);
      packed[pixelCount * 2 + pixel] = floatBits(b);
      packed[pixelCount * 3 + pixel] = floatBits(a);
    }
  }
  return packed;
}

export function unpackMaterialTextureAtlasPixels(
  packed: Uint32Array,
  encoding: MaterialTextureAtlasEncoding,
): Float32Array {
  const planes = materialTextureAtlasEncodingPlaneCount(encoding);
  if (packed.length % planes !== 0) {
    throw new RangeError('Packed material texture data does not contain complete codec planes.');
  }
  const pixelCount = packed.length / planes;
  const rgba = new Float32Array(pixelCount * 4);
  const decoded = new Float32Array(4);
  for (let pixel = 0; pixel < pixelCount; pixel += 1) {
    const target = pixel * 4;
    decodeMaterialTextureAtlasPixel(
      packed,
      encoding,
      pixelCount,
      pixel,
      decoded,
    );
    rgba.set(decoded, target);
  }
  return rgba;
}

function srgbToLinear(value: number): number {
  const c = clamp(value, 0, 1);
  return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
}

function linearToSrgb(value: number): number {
  const c = clamp(value, 0, 1);
  return c <= 0.0031308 ? c * 12.92 : 1.055 * c ** (1 / 2.4) - 0.055;
}

export function generateMaterialTextureAtlasMip(
  source: Uint32Array,
  width: number,
  height: number,
  encoding: MaterialTextureAtlasEncoding,
  decodeSrgb: boolean,
): { readonly width: number; readonly height: number; readonly data: Uint32Array } {
  if (
    !Number.isSafeInteger(width) ||
    !Number.isSafeInteger(height) ||
    width < 1 ||
    height < 1
  ) {
    throw new RangeError('Material texture mip dimensions must be positive safe integers.');
  }
  const sourcePixelCount = width * height;
  const expectedWords =
    sourcePixelCount * materialTextureAtlasEncodingPlaneCount(encoding);
  if (
    !Number.isSafeInteger(sourcePixelCount) ||
    !Number.isSafeInteger(expectedWords) ||
    source.length !== expectedWords
  ) {
    throw new RangeError(
      `Packed material texture mip has ${source.length} words; expected ` +
      `${String(expectedWords)} for ${width}x${height}.`,
    );
  }
  const targetWidth = Math.max(1, Math.floor(width / 2));
  const targetHeight = Math.max(1, Math.floor(height / 2));
  const target = new Float32Array(targetWidth * targetHeight * 4);
  const decoded = new Float32Array(4);
  for (let y = 0; y < targetHeight; y += 1) {
    for (let x = 0; x < targetWidth; x += 1) {
      const targetBase = (y * targetWidth + x) * 4;
      const sourceX0 = x * width / targetWidth;
      const sourceX1 = (x + 1) * width / targetWidth;
      const sourceY0 = y * height / targetHeight;
      const sourceY1 = (y + 1) * height / targetHeight;
      const firstX = Math.floor(sourceX0);
      const lastX = Math.ceil(sourceX1) - 1;
      const firstY = Math.floor(sourceY0);
      const lastY = Math.ceil(sourceY1) - 1;
      const mean = [0, 0, 0, 0];
      let weightSum = 0;
      for (let sy = firstY; sy <= lastY; sy += 1) {
        const wy = Math.max(
          0,
          Math.min(sourceY1, sy + 1) - Math.max(sourceY0, sy),
        );
        for (let sx = firstX; sx <= lastX; sx += 1) {
          const wx = Math.max(
            0,
            Math.min(sourceX1, sx + 1) - Math.max(sourceX0, sx),
          );
          const weight = wx * wy;
          decodeMaterialTextureAtlasPixel(
            source,
            encoding,
            sourcePixelCount,
            sy * width + sx,
            decoded,
          );
          const nextWeightSum = weightSum + weight;
          const blend = weight / Math.max(nextWeightSum, Number.EPSILON);
          for (let channel = 0; channel < 4; channel += 1) {
            const value = decodeSrgb && channel < 3
              ? srgbToLinear(decoded[channel]!)
              : decoded[channel]!;
            // Online convex averaging avoids overflowing an intermediate sum
            // when several finite HDR texels sit near the f32 maximum. This is
            // the CPU mirror of the GPU mip kernel below.
            mean[channel] =
              mean[channel]! * (1 - blend) + value * blend;
          }
          weightSum = nextWeightSum;
        }
      }
      for (let channel = 0; channel < 4; channel += 1) {
        let value = mean[channel]!;
        if (decodeSrgb && channel < 3) value = linearToSrgb(value);
        target[targetBase + channel] = value;
      }
    }
  }
  return {
    width: targetWidth,
    height: targetHeight,
    data: packMaterialTextureAtlasPixels(target, encoding),
  };
}
