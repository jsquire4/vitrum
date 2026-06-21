import type { MaterialSpec, TextureRef, TextureWrapMode, UvTransform } from '@vitrum/core';

interface RawHeightPixels {
  readonly width: number;
  readonly height: number;
  readonly data: ArrayLike<number>;
  readonly channels: 1 | 2 | 3 | 4;
  readonly decode: (value: number) => number;
}

type DisplacementWarningSink = (warning: string) => void;

function halfToFloat(h: number): number {
  const s = (h & 0x8000) >> 15;
  const e = (h & 0x7c00) >> 10;
  const f = h & 0x03ff;
  if (e === 0) return (s ? -1 : 1) * 2 ** -14 * (f / 1024);
  if (e === 0x1f) return f ? NaN : (s ? -1 : 1) * Infinity;
  return (s ? -1 : 1) * 2 ** (e - 15) * (1 + f / 1024);
}

function finiteOr(value: number | undefined, fallback: number): number {
  return value != null && Number.isFinite(value) ? value : fallback;
}

function textureHandleType(handle: unknown): string {
  return handle == null ? 'null' : Object.prototype.toString.call(handle);
}

function warningPrefix(primitiveId: string): string {
  return `Primitive "${primitiveId}" displacementMap`;
}

function handlePayload(handle: unknown): {
  readonly width: number;
  readonly height: number;
  readonly data: ArrayLike<number> | undefined;
  readonly hint?: { readonly channels?: number; readonly dataType?: string };
} | null {
  if (handle == null || typeof handle !== 'object') return null;
  const h = handle as {
    readonly width?: number;
    readonly height?: number;
    readonly data?: ArrayLike<number>;
    readonly image?: {
      readonly width?: number;
      readonly height?: number;
      readonly data?: ArrayLike<number>;
    };
    readonly __vitrum_hint__?: { readonly channels?: number; readonly dataType?: string };
    readonly channels?: number;
    readonly dataType?: string;
  };
  const image = h.image;
  const hint = h.__vitrum_hint__ ?? (
    h.channels != null || h.dataType != null
      ? {
          ...(h.channels != null ? { channels: h.channels } : {}),
          ...(h.dataType != null ? { dataType: h.dataType } : {}),
        }
      : undefined
  );
  return {
    width: Number(h.width ?? image?.width ?? 0),
    height: Number(h.height ?? image?.height ?? 0),
    data: h.data ?? image?.data,
    ...(hint != null ? { hint } : {}),
  };
}

function decoderFor(data: ArrayLike<number>, dataType: string | undefined): (value: number) => number {
  if (dataType === 'float32' || data instanceof Float32Array || data instanceof Float64Array) {
    return (value) => value;
  }
  if (dataType === 'float16' || dataType === 'half-float') {
    return (value) => halfToFloat(value);
  }
  if (dataType === 'uint16') {
    return (value) => Math.min(1, Math.max(0, value / 65535));
  }
  if (data instanceof Uint16Array) {
    return (value) => Math.min(1, Math.max(0, value / 65535));
  }
  if (
    dataType === 'uint8' ||
    data instanceof Uint8Array ||
    data instanceof Uint8ClampedArray ||
    data instanceof Int8Array
  ) {
    return (value) => Math.min(1, Math.max(0, value / 255));
  }
  const bytesPerElement = (data as { readonly BYTES_PER_ELEMENT?: number }).BYTES_PER_ELEMENT;
  if (typeof bytesPerElement === 'number' && bytesPerElement > 0) {
    const max = 2 ** (8 * bytesPerElement) - 1;
    return (value) => Math.min(1, Math.max(0, value / max));
  }
  return (value) => value;
}

function readHeightPixels(
  ref: TextureRef,
  primitiveId: string,
  warn: DisplacementWarningSink,
): RawHeightPixels | null {
  const payload = handlePayload(ref.handle);
  if (payload == null) {
    warn(`${warningPrefix(primitiveId)} handle is not CPU-readable (${textureHandleType(ref.handle)}); vertex displacement skipped.`);
    return null;
  }
  const { width, height, data, hint } = payload;
  if (width <= 0 || height <= 0 || data == null || typeof data.length !== 'number') {
    warn(`${warningPrefix(primitiveId)} handle is missing positive width/height/data; vertex displacement skipped.`);
    return null;
  }
  const pixelCount = width * height;
  const inferredChannels = data.length / pixelCount;
  const channels = hint?.channels ?? inferredChannels;
  if (![1, 2, 3, 4].includes(channels) || !Number.isInteger(channels)) {
    warn(
      `${warningPrefix(primitiveId)} has ${data.length} values for ${width}x${height} pixels; ` +
      'expected 1, 2, 3, or 4 channels. Vertex displacement skipped.',
    );
    return null;
  }
  return {
    width,
    height,
    data,
    channels: channels as 1 | 2 | 3 | 4,
    decode: decoderFor(data, hint?.dataType),
  };
}

function wrapCoord(value: number, mode: TextureWrapMode | undefined): number {
  if (!Number.isFinite(value)) return 0;
  switch (mode ?? 'repeat') {
    case 'clamp-to-edge':
      return Math.min(1, Math.max(0, value));
    case 'mirrored-repeat': {
      const period = Math.floor(value);
      const frac = value - period;
      return Math.abs(period % 2) === 1 ? 1 - frac : frac;
    }
    case 'repeat':
    default:
      return value - Math.floor(value);
  }
}

function applyUvTransform(u: number, v: number, transform: UvTransform | undefined): readonly [number, number] {
  const sx = finiteOr(transform?.scale?.[0], 1);
  const sy = finiteOr(transform?.scale?.[1], 1);
  const rotation = finiteOr(transform?.rotation, 0);
  const x = u * sx;
  const y = v * sy;
  const c = Math.cos(rotation);
  const s = Math.sin(rotation);
  return [
    x * c - y * s + finiteOr(transform?.offset?.[0], 0),
    x * s + y * c + finiteOr(transform?.offset?.[1], 0),
  ];
}

function redAt(pixels: RawHeightPixels, x: number, y: number): number {
  const clampedX = Math.min(pixels.width - 1, Math.max(0, x));
  const clampedY = Math.min(pixels.height - 1, Math.max(0, y));
  const offset = (clampedY * pixels.width + clampedX) * pixels.channels;
  return pixels.decode(Number(pixels.data[offset] ?? 0));
}

function sampleHeight(pixels: RawHeightPixels, u: number, v: number, ref: TextureRef): number {
  const [tu, tv] = applyUvTransform(u, v, ref.transform);
  const wu = wrapCoord(tu, ref.wrapS);
  const wv = wrapCoord(tv, ref.wrapT);
  if (pixels.width === 1 && pixels.height === 1) {
    return redAt(pixels, 0, 0);
  }
  const x = wu * Math.max(0, pixels.width - 1);
  const y = wv * Math.max(0, pixels.height - 1);
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const x1 = Math.min(pixels.width - 1, x0 + 1);
  const y1 = Math.min(pixels.height - 1, y0 + 1);
  const fx = x - x0;
  const fy = y - y0;
  const h00 = redAt(pixels, x0, y0);
  const h10 = redAt(pixels, x1, y0);
  const h01 = redAt(pixels, x0, y1);
  const h11 = redAt(pixels, x1, y1);
  const hx0 = h00 * (1 - fx) + h10 * fx;
  const hx1 = h01 * (1 - fx) + h11 * fx;
  return hx0 * (1 - fy) + hx1 * fy;
}

function normalize3(x: number, y: number, z: number): readonly [number, number, number] {
  const len = Math.hypot(x, y, z);
  if (len > 1e-8 && Number.isFinite(len)) return [x / len, y / len, z / len];
  return [0, 1, 0];
}

/**
 * Apply CPU-readable vertex displacement to a mesh-like primitive's local
 * positions. This is real geometry/BVH displacement, but it is intentionally
 * vertex-only: no tessellation/microdisplacement is synthesized.
 */
export function maybeDisplaceMeshPositions(
  input: {
    readonly primitiveId: string;
    readonly material: MaterialSpec;
    readonly positions: Float32Array;
    readonly normals: Float32Array;
    readonly uvs?: Float32Array;
    readonly uv1?: Float32Array;
    readonly onWarning?: DisplacementWarningSink;
  },
): Float32Array | null {
  const ref = input.material.displacementMap;
  if (ref == null) return null;
  const warn = input.onWarning ?? (() => {});
  const texCoord = ref.texCoord ?? 0;
  if (texCoord !== 0 && texCoord !== 1) {
    warn(`${warningPrefix(input.primitiveId)} requests TEXCOORD_${texCoord}; vertex displacement supports TEXCOORD_0/1 only. Skipped.`);
    return null;
  }
  const uvSource = texCoord === 1 ? input.uv1 : input.uvs;
  if (uvSource == null || uvSource.length === 0) {
    warn(`${warningPrefix(input.primitiveId)} requests TEXCOORD_${texCoord}, but that UV channel is absent; vertex displacement skipped.`);
    return null;
  }
  const pixels = readHeightPixels(ref, input.primitiveId, warn);
  if (pixels == null) return null;

  const vertexCount = Math.floor(input.positions.length / 3);
  const out = new Float32Array(input.positions.length);
  const scale = finiteOr(input.material.displacementScale, 1);
  const bias = finiteOr(input.material.displacementBias, 0);
  for (let i = 0; i < vertexCount; i += 1) {
    const px = input.positions[i * 3] ?? 0;
    const py = input.positions[i * 3 + 1] ?? 0;
    const pz = input.positions[i * 3 + 2] ?? 0;
    const [nx, ny, nz] = normalize3(
      input.normals[i * 3] ?? 0,
      input.normals[i * 3 + 1] ?? 1,
      input.normals[i * 3 + 2] ?? 0,
    );
    const u = uvSource[i * 2] ?? 0;
    const v = uvSource[i * 2 + 1] ?? 0;
    const amount = sampleHeight(pixels, u, v, ref) * scale + bias;
    out[i * 3] = px + nx * amount;
    out[i * 3 + 1] = py + ny * amount;
    out[i * 3 + 2] = pz + nz * amount;
  }
  return out;
}
