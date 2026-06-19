export const SOBOL_TEXTURE_SIZE = 256;
export const SOBOL_TEXTURE_POINTS = SOBOL_TEXTURE_SIZE * SOBOL_TEXTURE_SIZE;
export const SOBOL_TEXTURE_CHANNELS = 4;

const SOBOL_FACTOR = 1 / 16_777_216; // 2^-24, matching the GLSL Sobol texture path.

const DIRECTIONS_1 = [
  0x80000000, 0xc0000000, 0xa0000000, 0xf0000000,
  0x88000000, 0xcc000000, 0xaa000000, 0xff000000,
  0x80800000, 0xc0c00000, 0xa0a00000, 0xf0f00000,
  0x88880000, 0xcccc0000, 0xaaaa0000, 0xffff0000,
  0x80008000, 0xc000c000, 0xa000a000, 0xf000f000,
  0x88008800, 0xcc00cc00, 0xaa00aa00, 0xff00ff00,
  0x80808080, 0xc0c0c0c0, 0xa0a0a0a0, 0xf0f0f0f0,
  0x88888888, 0xcccccccc, 0xaaaaaaaa, 0xffffffff,
] as const;

const DIRECTIONS_2 = [
  0x80000000, 0xc0000000, 0x60000000, 0x90000000,
  0xe8000000, 0x5c000000, 0x8e000000, 0xc5000000,
  0x68800000, 0x9cc00000, 0xee600000, 0x55900000,
  0x80680000, 0xc09c0000, 0x60ee0000, 0x90550000,
  0xe8808000, 0x5cc0c000, 0x8e606000, 0xc5909000,
  0x6868e800, 0x9c9c5c00, 0xeeee8e00, 0x5555c500,
  0x8000e880, 0xc0005cc0, 0x60008e60, 0x9000c590,
  0xe8006868, 0x5c009c9c, 0x8e00eeee, 0xc5005555,
] as const;

const DIRECTIONS_3 = [
  0x80000000, 0xc0000000, 0x20000000, 0x50000000,
  0xf8000000, 0x74000000, 0xa2000000, 0x93000000,
  0xd8800000, 0x25400000, 0x59e00000, 0xe6d00000,
  0x78080000, 0xb40c0000, 0x82020000, 0xc3050000,
  0x208f8000, 0x51474000, 0xfbea2000, 0x75d93000,
  0xa0858800, 0x914e5400, 0xdbe79e00, 0x25db6d00,
  0x58800080, 0xe54000c0, 0x79e00020, 0xb6d00050,
  0x800800f8, 0xc00c0074, 0x200200a2, 0x50050093,
] as const;

const DIRECTIONS_4 = [
  0x80000000, 0x40000000, 0x20000000, 0xb0000000,
  0xf8000000, 0xdc000000, 0x7a000000, 0x9d000000,
  0x5a800000, 0x2fc00000, 0xa1600000, 0xf0b00000,
  0xda880000, 0x6fc40000, 0x81620000, 0x40bb0000,
  0x22878000, 0xb3c9c000, 0xfb65a000, 0xddb2d000,
  0x78022800, 0x9c0b3c00, 0x5a0fb600, 0x2d0ddb00,
  0xa2878080, 0xf3c9c040, 0xdb65a020, 0x6db2d0b0,
  0x800228f8, 0x400b3cdc, 0x200fb67a, 0xb00ddb9d,
] as const;

export function reverseBits32(value: number): number {
  let x = value >>> 0;
  x = (((x & 0xaaaaaaaa) >>> 1) | ((x & 0x55555555) << 1)) >>> 0;
  x = (((x & 0xcccccccc) >>> 2) | ((x & 0x33333333) << 2)) >>> 0;
  x = (((x & 0xf0f0f0f0) >>> 4) | ((x & 0x0f0f0f0f) << 4)) >>> 0;
  x = (((x & 0xff00ff00) >>> 8) | ((x & 0x00ff00ff) << 8)) >>> 0;
  return ((x >>> 16) | (x << 16)) >>> 0;
}

export function maskedSobol(index: number, directions: readonly number[]): number {
  let out = 0;
  const i = index >>> 0;
  for (let bit = 0; bit < 32; bit += 1) {
    if (((i >>> bit) & 1) !== 0) out = (out ^ (directions[bit] ?? 0)) >>> 0;
  }
  return out >>> 0;
}

function sobolComponent(index: number, directions: readonly number[]): number {
  return (reverseBits32(maskedSobol(index, directions)) & 0x00ffffff) * SOBOL_FACTOR;
}

export function sobolTexturePoint(index: number): readonly [number, number, number, number] {
  const i = Math.floor(index);
  if (!Number.isFinite(i) || i < 0 || i >= SOBOL_TEXTURE_POINTS) return [0, 0, 0, 0];
  return [
    sobolComponent(i, DIRECTIONS_1),
    sobolComponent(i, DIRECTIONS_2),
    sobolComponent(i, DIRECTIONS_3),
    sobolComponent(i, DIRECTIONS_4),
  ];
}

export function generateSobolTextureData(pointCount = SOBOL_TEXTURE_POINTS): Float32Array {
  if (!Number.isInteger(pointCount) || pointCount < 0 || pointCount > SOBOL_TEXTURE_POINTS) {
    throw new RangeError(`pointCount must be an integer in [0, ${SOBOL_TEXTURE_POINTS}]`);
  }
  const out = new Float32Array(pointCount * SOBOL_TEXTURE_CHANNELS);
  for (let i = 0; i < pointCount; i += 1) {
    const p = sobolTexturePoint(i);
    const o = i * SOBOL_TEXTURE_CHANNELS;
    out[o + 0] = p[0];
    out[o + 1] = p[1];
    out[o + 2] = p[2];
    out[o + 3] = p[3];
  }
  return out;
}
