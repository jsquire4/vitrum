import type { MaterialSpec, TextureRef, TextureWrapMode } from '@vitrum/core';

export const BASE_COLOR_MAP_META_TEX_WIDTH = 4096;

export interface MaterialTextureAtlasPayload {
  readonly atlasData: Float32Array;
  readonly atlasDim: number;
  readonly atlasLayerCount: number;
  readonly baseColorMetaData: Float32Array;
  readonly baseColorMetaWidth: number;
  readonly baseColorMetaHeight: number;
  readonly readableBaseColorLayerCount: number;
}

export interface MaterialTextureAtlasGpu {
  readonly atlasTexture: GPUTexture;
  readonly atlasTextureView: GPUTextureView;
  readonly baseColorMetaTexture: GPUTexture;
  readonly baseColorMetaTextureView: GPUTextureView;
  readonly atlasDim: number;
  readonly atlasLayerCount: number;
  readonly baseColorMetaWidth: number;
  readonly baseColorMetaHeight: number;
}

interface RawPixels {
  readonly width: number;
  readonly height: number;
  readonly data: Float32Array;
  readonly sourceColorSpace?: 'srgb' | 'linear';
}

interface TextureHandleHint {
  readonly channels?: 1 | 2 | 3 | 4;
  readonly dataType?: 'uint8' | 'uint16' | 'float32';
  readonly colorSpace?: 'srgb' | 'linear';
}

const TEX_BINDING = 0x04;
const COPY_DST = 0x02;

function halfToFloat(h: number): number {
  const s = (h & 0x8000) >> 15;
  const e = (h & 0x7c00) >> 10;
  const f = h & 0x03ff;
  if (e === 0) return (s ? -1 : 1) * 2 ** -14 * (f / 1024);
  if (e === 0x1f) return f ? NaN : (s ? -1 : 1) * Infinity;
  return (s ? -1 : 1) * 2 ** (e - 15) * (1 + f / 1024);
}

function srgbToLinear(v: number): number {
  const c = Math.max(0, Math.min(1, v));
  return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
}

function asTextureRef(value: unknown): TextureRef | null {
  if (value == null || typeof value !== 'object') return null;
  if ('handle' in value) return value as TextureRef;
  return { handle: value };
}

function readHandlePixels(handle: unknown): RawPixels | null {
  const h = handle as {
    width?: number;
    height?: number;
    data?: ArrayLike<number>;
    image?: { width?: number; height?: number; data?: ArrayLike<number> };
    __vitrum_hint__?: TextureHandleHint;
    channels?: number;
    dataType?: string;
    colorSpace?: string;
  } | null;
  if (h == null) return null;
  const src = h.data ?? h.image?.data;
  const width = Number(h.width ?? h.image?.width ?? 0);
  const height = Number(h.height ?? h.image?.height ?? 0);
  if (src == null || typeof src.length !== 'number' || width <= 0 || height <= 0) return null;

  const hint: TextureHandleHint | undefined = h.__vitrum_hint__ ?? (
    (h.channels != null || h.dataType != null || h.colorSpace != null)
      ? Object.assign(
          {} as TextureHandleHint,
          h.channels != null ? { channels: h.channels as TextureHandleHint['channels'] } : {},
          h.dataType != null ? { dataType: h.dataType as TextureHandleHint['dataType'] } : {},
          h.colorSpace != null ? { colorSpace: h.colorSpace as TextureHandleHint['colorSpace'] } : {},
        )
      : undefined
  );

  const heuristicStride = Math.max(1, Math.round(src.length / (width * height)));
  const stride = hint?.channels ?? heuristicStride;
  if (hint == null && stride !== 1 && stride !== 4) {
    console.warn(
      `[walkaround-hybrid] baseColorMap texture handle has ambiguous pixel stride ${stride} ` +
        `(${src.length} values / ${width}x${height} pixels). Attach ` +
        '__vitrum_hint__ = { channels: N } to decode it deterministically.',
    );
  }

  const isHalf = src instanceof Uint16Array;
  const isFloat = src instanceof Float32Array;
  const useHalf = hint?.dataType != null ? hint.dataType === 'uint16' : isHalf;
  const useFloat = hint?.dataType != null ? hint.dataType === 'float32' : isFloat;
  const bpe = (src as { BYTES_PER_ELEMENT?: number }).BYTES_PER_ELEMENT ?? 1;
  const intMax = useHalf || useFloat ? 0 : 2 ** (8 * bpe) - 1;
  const dec = (v: number): number => (useHalf ? halfToFloat(v) : useFloat ? v : intMax > 0 ? v / intMax : v);

  const out = new Float32Array(width * height * 4);
  for (let p = 0; p < width * height; p += 1) {
    const s = p * stride;
    out[p * 4] = dec(Number(src[s] ?? 0));
    out[p * 4 + 1] = dec(Number(src[s + (stride > 1 ? 1 : 0)] ?? 0));
    out[p * 4 + 2] = dec(Number(src[s + (stride > 2 ? 2 : 0)] ?? 0));
    out[p * 4 + 3] = stride >= 4 ? dec(Number(src[s + 3] ?? 1)) : 1;
  }

  const sourceColorSpace =
    hint?.colorSpace === 'srgb' || hint?.colorSpace === 'linear'
      ? hint.colorSpace
      : undefined;
  return { width, height, data: out, ...(sourceColorSpace ? { sourceColorSpace } : {}) };
}

function blitBaseColorLayer(px: RawPixels, dim: number, data: Float32Array, layer: number): void {
  const base = layer * dim * dim * 4;
  const decodeSrgb = px.sourceColorSpace !== 'linear';
  for (let y = 0; y < dim; y += 1) {
    const sy = Math.min(px.height - 1, (y * px.height / dim) | 0);
    for (let x = 0; x < dim; x += 1) {
      const sx = Math.min(px.width - 1, (x * px.width / dim) | 0);
      const s = (sy * px.width + sx) * 4;
      const d = base + (y * dim + x) * 4;
      const r = px.data[s]!;
      const g = px.data[s + 1]!;
      const b = px.data[s + 2]!;
      data[d] = decodeSrgb ? srgbToLinear(r) : r;
      data[d + 1] = decodeSrgb ? srgbToLinear(g) : g;
      data[d + 2] = decodeSrgb ? srgbToLinear(b) : b;
      data[d + 3] = px.data[s + 3]!;
    }
  }
}

function wrapIndex(mode: TextureWrapMode | undefined): number {
  switch (mode ?? 'repeat') {
    case 'repeat':
      return 0;
    case 'clamp-to-edge':
      return 1;
    case 'mirrored-repeat':
      return 2;
  }
}

function writeDisabledMeta(meta: Float32Array, texel: number): void {
  const b = texel * 4;
  meta[b] = -1;
  meta[b + 1] = 0;
  meta[b + 2] = 0;
  meta[b + 3] = 0;
  meta[b + 4] = 1;
  meta[b + 5] = 1;
  meta[b + 6] = 1;
  meta[b + 7] = 0;
}

export function packMaterialTextureAtlas(
  materials: readonly MaterialSpec[],
  triMaterialIds: Uint32Array,
  triCount: number,
): MaterialTextureAtlasPayload {
  const readable = new Map<unknown, { readonly layer: number; readonly pixels: RawPixels }>();
  const ordered: { readonly handle: unknown; readonly pixels: RawPixels }[] = [];

  for (const material of materials) {
    const ref = asTextureRef(material.baseColorMap);
    if (ref?.handle == null) continue;
    if (readable.has(ref.handle)) continue;
    const pixels = readHandlePixels(ref.handle);
    if (pixels == null) {
      console.warn(
        '[walkaround-hybrid] baseColorMap texture handle is not readable ' +
          '(expected raw {width,height,data} or DataTexture-shaped image); the map is ignored.',
      );
      continue;
    }
    const layer = ordered.length;
    ordered.push({ handle: ref.handle, pixels });
    readable.set(ref.handle, { layer, pixels });
  }

  const atlasDim = Math.max(1, ...ordered.map((entry) => Math.max(entry.pixels.width, entry.pixels.height)));
  const atlasLayerCount = Math.max(1, ordered.length);
  const atlasData = new Float32Array(atlasDim * atlasDim * 4 * atlasLayerCount);
  if (ordered.length === 0) {
    atlasData.set([1, 1, 1, 1]);
  } else {
    ordered.forEach((entry, layer) => blitBaseColorLayer(entry.pixels, atlasDim, atlasData, layer));
  }

  const metaTexels = Math.max(1, triCount * 2);
  const baseColorMetaWidth = Math.min(BASE_COLOR_MAP_META_TEX_WIDTH, metaTexels);
  const baseColorMetaHeight = Math.ceil(metaTexels / baseColorMetaWidth);
  const baseColorMetaData = new Float32Array(baseColorMetaWidth * baseColorMetaHeight * 4);
  for (let tri = 0; tri < triCount; tri += 1) {
    const texel = tri * 2;
    const mat = materials[triMaterialIds[tri] ?? 0];
    const ref = asTextureRef(mat?.baseColorMap);
    if (ref?.handle == null) {
      writeDisabledMeta(baseColorMetaData, texel);
      continue;
    }
    const texCoord = ref.texCoord ?? 0;
    const layer = texCoord === 0 || texCoord === 1 ? readable.get(ref.handle)?.layer : undefined;
    if (layer == null) {
      writeDisabledMeta(baseColorMetaData, texel);
      continue;
    }
    const t = ref.transform;
    const rotation = t?.rotation ?? 0;
    const b0 = texel * 4;
    const b1 = b0 + 4;
    baseColorMetaData[b0] = layer;
    baseColorMetaData[b0 + 1] = wrapIndex(ref.wrapS) + wrapIndex(ref.wrapT) * 4 + texCoord * 16;
    baseColorMetaData[b0 + 2] = t?.offset?.[0] ?? 0;
    baseColorMetaData[b0 + 3] = t?.offset?.[1] ?? 0;
    baseColorMetaData[b1] = t?.scale?.[0] ?? 1;
    baseColorMetaData[b1 + 1] = t?.scale?.[1] ?? 1;
    baseColorMetaData[b1 + 2] = Math.cos(rotation);
    baseColorMetaData[b1 + 3] = Math.sin(rotation);
  }

  return {
    atlasData,
    atlasDim,
    atlasLayerCount,
    baseColorMetaData,
    baseColorMetaWidth,
    baseColorMetaHeight,
    readableBaseColorLayerCount: ordered.length,
  };
}

export function uploadMaterialTextureAtlas(
  device: GPUDevice,
  payload: MaterialTextureAtlasPayload,
): MaterialTextureAtlasGpu {
  const atlasTexture = device.createTexture({
    label: 'vitrum.materialTextureAtlas.baseColor.rgba32float-array',
    size: {
      width: payload.atlasDim,
      height: payload.atlasDim,
      depthOrArrayLayers: payload.atlasLayerCount,
    },
    format: 'rgba32float',
    usage: TEX_BINDING | COPY_DST,
  });
  writeRgba32FloatTexture(
    device,
    atlasTexture,
    payload.atlasData,
    payload.atlasDim,
    payload.atlasDim,
    payload.atlasLayerCount,
  );

  const baseColorMetaTexture = device.createTexture({
    label: 'vitrum.materialTextureAtlas.baseColorMeta.rgba32float',
    size: {
      width: payload.baseColorMetaWidth,
      height: payload.baseColorMetaHeight,
      depthOrArrayLayers: 1,
    },
    format: 'rgba32float',
    usage: TEX_BINDING | COPY_DST,
  });
  writeRgba32FloatTexture(
    device,
    baseColorMetaTexture,
    payload.baseColorMetaData,
    payload.baseColorMetaWidth,
    payload.baseColorMetaHeight,
    1,
  );

  return {
    atlasTexture,
    atlasTextureView: atlasTexture.createView({ dimension: '2d-array' }),
    baseColorMetaTexture,
    baseColorMetaTextureView: baseColorMetaTexture.createView(),
    atlasDim: payload.atlasDim,
    atlasLayerCount: payload.atlasLayerCount,
    baseColorMetaWidth: payload.baseColorMetaWidth,
    baseColorMetaHeight: payload.baseColorMetaHeight,
  };
}

function alignTo(value: number, alignment: number): number {
  return Math.ceil(value / alignment) * alignment;
}

function writeRgba32FloatTexture(
  device: GPUDevice,
  texture: GPUTexture,
  data: Float32Array,
  width: number,
  height: number,
  depthOrArrayLayers: number,
): void {
  const rowBytes = width * 4 * 4;
  const bytesPerRow = alignTo(rowBytes, 256);
  const source = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  let upload: Uint8Array;
  if (bytesPerRow === rowBytes) {
    upload = source;
  } else {
    upload = new Uint8Array(bytesPerRow * height * depthOrArrayLayers);
    for (let layer = 0; layer < depthOrArrayLayers; layer += 1) {
      for (let y = 0; y < height; y += 1) {
        const srcOffset = (layer * height + y) * rowBytes;
        const dstOffset = (layer * height + y) * bytesPerRow;
        upload.set(source.subarray(srcOffset, srcOffset + rowBytes), dstOffset);
      }
    }
  }
  const uploadBuffer = upload.buffer.slice(upload.byteOffset, upload.byteOffset + upload.byteLength) as ArrayBuffer;
  device.queue.writeTexture(
    { texture },
    uploadBuffer,
    { bytesPerRow, rowsPerImage: height },
    { width, height, depthOrArrayLayers },
  );
}
