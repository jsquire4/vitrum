import { srgbToLinear } from './tonemap.js';

export type EnvironmentMapColorSpace = 'linear' | 'srgb';

export interface EnvironmentMapHandleHint {
  readonly channels?: 1 | 2 | 3 | 4;
  readonly dataType?: 'uint8' | 'uint16' | 'float32';
  readonly colorSpace?: EnvironmentMapColorSpace;
}

export interface EnvironmentMapPixels {
  readonly width: number;
  readonly height: number;
  /** RGBA, row-major, linear-light float pixels. */
  readonly data: Float32Array;
  readonly sourceChannels: 1 | 2 | 3 | 4;
  readonly explicitChannels: boolean;
  readonly sourceColorSpace: EnvironmentMapColorSpace;
}

type EnvironmentMapHandleLike = {
  readonly width?: unknown;
  readonly height?: unknown;
  readonly data?: ArrayLike<number>;
  readonly image?: {
    readonly width?: unknown;
    readonly height?: unknown;
    readonly data?: ArrayLike<number>;
  };
  readonly __vitrum_hint__?: EnvironmentMapHandleHint;
  readonly channels?: unknown;
  readonly dataType?: unknown;
  readonly colorSpace?: unknown;
};

function halfToFloat(h: number): number {
  const s = (h & 0x8000) >> 15;
  const e = (h & 0x7c00) >> 10;
  const f = h & 0x03ff;
  if (e === 0) return (s ? -1 : 1) * 2 ** -14 * (f / 1024);
  if (e === 0x1f) return f ? NaN : (s ? -1 : 1) * Infinity;
  return (s ? -1 : 1) * 2 ** (e - 15) * (1 + f / 1024);
}

function normalizedHint(handle: EnvironmentMapHandleLike): EnvironmentMapHandleHint | undefined {
  const direct = handle.__vitrum_hint__;
  if (direct != null) return direct;
  const channels = handle.channels;
  const dataType = handle.dataType;
  const colorSpace = handle.colorSpace;
  if (channels == null && dataType == null && colorSpace == null) return undefined;
  return {
    ...(channels === 1 || channels === 2 || channels === 3 || channels === 4 ? { channels } : {}),
    ...(dataType === 'uint8' || dataType === 'uint16' || dataType === 'float32' ? { dataType } : {}),
    ...(colorSpace === 'linear' || colorSpace === 'srgb' ? { colorSpace } : {}),
  };
}

function inferChannels(valueCount: number, pixelCount: number, hint: EnvironmentMapHandleHint | undefined): 1 | 2 | 3 | 4 {
  if (hint?.channels === 1 || hint?.channels === 2 || hint?.channels === 3 || hint?.channels === 4) {
    return hint.channels;
  }
  const perPixel = valueCount / Math.max(1, pixelCount);
  if (perPixel >= 4) return 4;
  if (perPixel >= 3) return 3;
  if (perPixel >= 2) return 2;
  return 1;
}

function decodeScalar(src: ArrayLike<number>, index: number, hint: EnvironmentMapHandleHint | undefined): number {
  const raw = Number(src[index] ?? 0);
  const view = src as { readonly BYTES_PER_ELEMENT?: number };
  const dataType = hint?.dataType;
  if (dataType === 'uint16') return halfToFloat(raw);
  if (dataType === 'float32') return raw;
  if (dataType === 'uint8') return raw / 255;
  if (src instanceof Uint16Array) return halfToFloat(raw);
  if (src instanceof Float32Array || src instanceof Float64Array) return raw;
  if (Array.isArray(src)) return raw;
  const bpe = view.BYTES_PER_ELEMENT ?? 1;
  const maxValue = 2 ** (8 * bpe) - 1;
  return maxValue > 0 ? raw / maxValue : raw;
}

/**
 * Read an environment-map handle into linear RGBA float pixels.
 *
 * Supported host shapes intentionally mirror the material-texture bridge:
 * - raw `{ width, height, data }`
 * - DataTexture-shaped `{ image: { width, height, data } }`
 * - optional direct or `__vitrum_hint__` fields for channels/dataType/colorSpace
 *
 * Opaque image/file handles still return `null`; the host or glTF adapter must
 * decode those into CPU-readable pixels before a renderer can build HDRI CDFs.
 */
export function readEnvironmentMapPixels(handle: unknown): EnvironmentMapPixels | null {
  const h = handle as EnvironmentMapHandleLike | null;
  if (h == null || typeof h !== 'object') return null;
  const src = h.data ?? h.image?.data;
  const width = Number(h.width ?? h.image?.width ?? 0);
  const height = Number(h.height ?? h.image?.height ?? 0);
  if (
    src == null ||
    typeof src.length !== 'number' ||
    !Number.isFinite(width) ||
    !Number.isFinite(height) ||
    width <= 0 ||
    height <= 0
  ) {
    return null;
  }
  const pixelCount = width * height;
  if (src.length < pixelCount) return null;
  const hint = normalizedHint(h);
  const channels = inferChannels(src.length, pixelCount, hint);
  if (src.length < pixelCount * channels) return null;
  const colorSpace = hint?.colorSpace === 'srgb' ? 'srgb' : 'linear';
  const out = new Float32Array(pixelCount * 4);
  for (let p = 0; p < pixelCount; p += 1) {
    const base = p * channels;
    let r = decodeScalar(src, base, hint);
    let g = channels > 1 ? decodeScalar(src, base + 1, hint) : r;
    let b = channels > 2 ? decodeScalar(src, base + 2, hint) : r;
    if (colorSpace === 'srgb') {
      r = srgbToLinear(r);
      g = srgbToLinear(g);
      b = srgbToLinear(b);
    }
    out[p * 4] = r;
    out[p * 4 + 1] = g;
    out[p * 4 + 2] = b;
    out[p * 4 + 3] = channels > 3 ? decodeScalar(src, base + 3, hint) : 1;
  }
  return {
    width,
    height,
    data: out,
    sourceChannels: channels,
    explicitChannels: hint?.channels != null,
    sourceColorSpace: colorSpace,
  };
}
