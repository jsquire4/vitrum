// texturesArray — the material-map texture atlas (`sampler2DArray textures`) the
// fork GLSL samples via `texture(textures, vec3(uv, material.<map>))`, where the
// per-map float is a LAYER index into this array (materialsTexture assigns them).
//
// THREE-free: a `TextureRef.handle` is opaque (`EnvironmentMapRef = unknown`). For
// the THREE-free path tracer the handle must expose CPU pixels — either a raw
// `{ width, height, data }` payload (the on-ramp form, like the env G2 bridge) or a
// THREE `DataTexture`-shaped `{ image: { data, width, height } }`. Image/ImageBitmap
// sources (canvas readback) are a documented host-side follow-up; this packer reads
// the DataTexture/raw forms that cover procedural + baked textures.
//
// All layers share one storage dimension (sampler2DArray requirement), but each
// source keeps its authored width/height and independent mip pyramid in the
// top-left source rectangle. GLSL samples with those native extents, so mixed
// resolutions do not change filtering footprints or bleed padded texels.

import type { EngineWarning, MaterialSpec } from '@vitrum/core';

/** The material-map fields the fork GLSL samples (others are inert until wired).
 *  D3 (Wave C) added the clearcoat/sheen/iridescence/specular maps — the fork
 *  `get_surface_record` ALREADY samples each (see clearcoatMap…specularIntensityMap
 *  in get_surface_record_function.glsl.js), only the packer wired them as NO_TEXTURE.
 *  aoMap/lightMap/bumpMap are NEW GLSL (added to material_struct + get_surface_record)
 *  so they are gathered here too. */
const SAMPLED_MAP_KEYS = [
  'baseColorMap',
  'metallicMap',
  'roughnessMap',
  'transmissionMap',
  'emissiveMap',
  'normalMap',
  'alphaMap',
  // D3 — clearcoat / sheen / iridescence / specular maps (GLSL already samples).
  'clearcoatMap',
  'clearcoatRoughnessMap',
  'clearcoatNormalMap',
  'sheenColorMap',
  'sheenRoughnessMap',
  'iridescenceMap',
  'iridescenceThicknessMap',
  'specularColorMap',
  'specularIntensityMap',
  // D3 — aoMap / lightMap / bumpMap (new GLSL consumption sites).
  'aoMap',
  'lightMap',
  'bumpMap',
  // KHR_materials_anisotropy: RG = tangent direction, B = strength.
  'anisotropyMap',
  // KHR_materials_volume: G = scalar multiplier for thicknessFactor.
  'thicknessMap',
] as const satisfies ReadonlyArray<keyof MaterialSpec>;

// ── D10.12: TextureHandleHint ─────────────────────────────────────────────────
// Optional hints that a TextureRef handle can expose to make readHandlePixels
// unambiguous without relying on the stride heuristic (which can mis-classify
// 3-channel RGB data, for example). A host that provides these hints avoids the
// ambiguous-stride console.warn and gets deterministic decoding.
//
// Usage: attach a `__vitrum_hint__` property to the texture handle object, OR
// pass a wrapper that implements this interface as the handle.
//
// channels: 1 | 2 | 3 | 4 — number of channels per pixel in `data`.
//   If omitted, only exact one- or four-channel payloads can be inferred. RG/RGB
//   payloads require the hint so authored channel layouts are never guessed.
// dataType: 'uint8' | 'uint16' | 'float16' | 'half-float' | 'float32' —
//   encoding of each channel value. Uint16Array without a dataType hint is
//   treated as normalized uint16, matching glTF decode handles.
//   If omitted, inferred from the ArrayLike type (Uint8Array→uint8 etc.).

export interface TextureHandleHint {
  readonly channels?: 1 | 2 | 3 | 4;
  readonly dataType?: 'uint8' | 'uint16' | 'float16' | 'half-float' | 'float32';
  /**
   * Source encoding hint. By default, color/tint map roles are treated as sRGB
   * sources and converted into the atlas' linear RGBA32F payload; scalar/data
   * map roles stay linear. Set `colorSpace:'linear'` for a color map handle
   * that is already linear-light.
   */
  readonly colorSpace?: TextureSampleColorSpace;
}

export type TextureSampleColorSpace = 'srgb' | 'linear';

export interface TextureAtlasLayerMap {
  readonly srgb: ReadonlyMap<unknown, number>;
  readonly linear: ReadonlyMap<unknown, number>;
  /** Native level-0 width/height for each handle. */
  readonly dimensions?: ReadonlyMap<unknown, readonly [number, number]>;
}

export interface TextureAtlas {
  /** RGBA32F, `layerCount` layers each `dim × dim`. */
  readonly data: Float32Array;
  readonly dim: number;
  /** Complete RGBA32F mip chain; level 0 aliases `data`. */
  readonly mipLevels: readonly TextureAtlasMipLevel[];
  readonly layerCount: number;
  /** Back-compat default map: first layer for a handle, regardless of role. */
  readonly layerOf: Map<unknown, number>;
  /** Role-aware layer maps; color/tint maps and data maps can share a handle safely. */
  readonly layerOfByColorSpace: TextureAtlasLayerMap;
  /** Native width/height by layer, in the same order as atlas layer ids. */
  readonly sourceDimensions: readonly (readonly [number, number])[];
}

export interface TextureAtlasMipLevel {
  readonly data: Float32Array;
  readonly dim: number;
}

export function textureAtlasLayerCapacity(layerCount: number, maxLayers: number): number {
  const count = Math.max(0, Math.floor(layerCount));
  const limit = Math.max(0, Math.floor(maxLayers));
  if (count === 0 || limit === 0) return 0;
  let capacity = 1;
  while (capacity < count + 1) capacity *= 2;
  return Math.min(limit, Math.max(count, capacity));
}

export interface TextureAtlasBuildOptions {
  readonly onWarning?: (warning: EngineWarning) => void;
  readonly warningPhase?: string;
  readonly warningMethod?: string;
  /** Device array-layer limit used to reject an impossible atlas before source validation. */
  readonly maxArrayTextureLayers?: number;
}

function handleType(handle: unknown): string {
  return handle == null ? 'null' : Object.prototype.toString.call(handle);
}

const SRGB_MAP_KEYS = new Set<keyof MaterialSpec>([
  'baseColorMap',
  'emissiveMap',
  'sheenColorMap',
  'specularColorMap',
]);

export function textureColorSpaceForMapKey(key: keyof MaterialSpec): TextureSampleColorSpace {
  return SRGB_MAP_KEYS.has(key) ? 'srgb' : 'linear';
}

function collectHandle(
  handles: { handle: unknown; colorSpace: TextureSampleColorSpace }[],
  seen: Map<unknown, Set<TextureSampleColorSpace>>,
  ref: { readonly handle?: unknown } | undefined,
  colorSpace: TextureSampleColorSpace,
): void {
  const handle = ref?.handle;
  if (handle == null) return;
  let seenSpaces = seen.get(handle);
  if (seenSpaces == null) {
    seenSpaces = new Set<TextureSampleColorSpace>();
    seen.set(handle, seenSpaces);
  }
  if (!seenSpaces.has(colorSpace)) {
    seenSpaces.add(colorSpace);
    handles.push({ handle, colorSpace });
  }
}

/** IEEE-754 half (uint16) → float32 (DataTextures may ship HalfFloat). */
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

function texturePayloadError(
  handle: unknown,
  options: TextureAtlasBuildOptions | undefined,
  reason: string,
): Error {
  const operation = options?.warningMethod != null ? ` during ${options.warningMethod}` : '';
  return new Error(
    `[pt-webgl2] authored material texture${operation} is not CPU-readable: ${reason}. ` +
      'Provide one coherent raw { width, height, data } payload, a DataTexture-shaped image, ' +
      'or an immutable cpuMirror with an explicit channels hint for RG/RGB data. ' +
      `Handle type: ${handleType(handle)}`,
  );
}

const VALID_CHANNEL_COUNTS = new Set<number>([1, 2, 3, 4]);
const VALID_DATA_TYPES = new Set<string>(['uint8', 'uint16', 'float16', 'half-float', 'float32']);

const FLOAT32_BYTES = 4;
const MAX_TEXTURE_ATLAS_CPU_BYTES = 512 * 1024 * 1024;
const MAX_TEXTURE_ATLAS_FLOATS = MAX_TEXTURE_ATLAS_CPU_BYTES / FLOAT32_BYTES;

interface TexturePixelPayload {
  readonly width?: number;
  readonly height?: number;
  readonly data?: ArrayLike<number>;
  readonly channels?: number;
  readonly dataType?: string;
  readonly colorSpace?: string;
}

type ResolvedTextureDataType = NonNullable<TextureHandleHint['dataType']>;

interface InspectedTexturePixels {
  readonly width: number;
  readonly height: number;
  readonly pixelCount: number;
  readonly data: ArrayLike<number>;
  readonly stride: number;
  readonly dataType: ResolvedTextureDataType;
  readonly sourceColorSpace?: TextureSampleColorSpace;
}

function decodeTextureChannel(dataType: ResolvedTextureDataType, value: number): number {
  switch (dataType) {
    case 'uint8':
      return value / 255;
    case 'uint16':
      return value / 65535;
    case 'float16':
    case 'half-float':
      return halfToFloat(value);
    case 'float32':
      return value;
  }
}

/** Read RGBA float pixels from a TextureRef handle (raw payload or DataTexture-shaped).
 *  D10.12: optional TextureHandleHint (`__vitrum_hint__` property on the handle, or the
 *  handle itself implementing TextureHandleHint) provides explicit channels/dataType.
 *  Every authored payload is validated exactly and failure is synchronous: no texture
 *  may enter the retained scene after a guessed layout or partial CPU decode. */
function inspectHandlePixels(
  handle: unknown,
  options?: TextureAtlasBuildOptions,
): InspectedTexturePixels {
  const h = handle as {
    width?: number;
    height?: number;
    data?: ArrayLike<number>;
    image?: TexturePixelPayload;
    cpuMirror?: TexturePixelPayload;
    // D10.12: optional hint as a direct property on the handle
    __vitrum_hint__?: TextureHandleHint;
    channels?: number;
    dataType?: string;
    colorSpace?: string;
  } | null;
  if (h == null) {
    throw texturePayloadError(handle, options, 'the handle is null');
  }
  // Select one coherent payload. In particular, a partial cpuMirror must not be
  // completed with dimensions or data from a separate mutable image object.
  const source: TexturePixelPayload | undefined = h.cpuMirror ?? (h.data != null ? h : h.image);
  const src = source?.data;
  const rawWidth = source?.width;
  const rawHeight = source?.height;
  if (src == null || typeof src.length !== 'number') {
    throw texturePayloadError(
      handle,
      options,
      'no cpuMirror, raw data, or DataTexture-shaped image data was supplied',
    );
  }
  if (
    typeof rawWidth !== 'number' ||
    typeof rawHeight !== 'number' ||
    !Number.isSafeInteger(rawWidth) ||
    !Number.isSafeInteger(rawHeight) ||
    rawWidth <= 0 ||
    rawHeight <= 0
  ) {
    throw texturePayloadError(
      handle,
      options,
      `width and height must be positive safe integers (received ${String(rawWidth)}×${String(rawHeight)})`,
    );
  }
  const width = rawWidth;
  const height = rawHeight;
  const pixelCount = width * height;
  if (!Number.isSafeInteger(pixelCount) || pixelCount * 4 > MAX_TEXTURE_ATLAS_FLOATS) {
    throw texturePayloadError(
      handle,
      options,
      `the ${width}×${height} decoded RGBA payload exceeds the 512 MiB CPU staging budget`,
    );
  }
  const valueCount = src.length;
  if (!Number.isSafeInteger(valueCount) || valueCount < 0) {
    throw texturePayloadError(
      handle,
      options,
      `data.length must be a finite non-negative safe integer (received ${String(valueCount)})`,
    );
  }

  // D10.12: resolve hint from __vitrum_hint__ property, or direct channels/dataType on handle.
  // Use type assertions at the object-literal level to satisfy exactOptionalPropertyTypes:
  // only include a property in the literal when the source value is non-null.
  const inlineChannels = source?.channels ?? h.channels;
  const inlineDataType = source?.dataType ?? h.dataType;
  const inlineColorSpace = source?.colorSpace ?? h.colorSpace;
  const hint: TextureHandleHint | undefined =
    h.__vitrum_hint__ ??
    (inlineChannels != null || inlineDataType != null || inlineColorSpace != null
      ? Object.assign(
          {} as TextureHandleHint,
          inlineChannels != null
            ? { channels: inlineChannels as TextureHandleHint['channels'] }
            : {},
          inlineDataType != null
            ? { dataType: inlineDataType as TextureHandleHint['dataType'] }
            : {},
          inlineColorSpace != null
            ? { colorSpace: inlineColorSpace as TextureHandleHint['colorSpace'] }
            : {},
        )
      : undefined);

  const hintedChannels = hint?.channels;
  if (hintedChannels != null && !VALID_CHANNEL_COUNTS.has(hintedChannels)) {
    throw texturePayloadError(
      handle,
      options,
      `channels must be one of 1, 2, 3, or 4 (received ${String(hintedChannels)})`,
    );
  }
  if (hint?.dataType != null && !VALID_DATA_TYPES.has(hint.dataType)) {
    throw texturePayloadError(
      handle,
      options,
      `dataType "${String(hint.dataType)}" is unsupported`,
    );
  }
  if (hint?.colorSpace != null && hint.colorSpace !== 'srgb' && hint.colorSpace !== 'linear') {
    throw texturePayloadError(
      handle,
      options,
      `colorSpace "${String(hint.colorSpace)}" is unsupported`,
    );
  }

  let stride: number;
  if (hintedChannels != null) {
    stride = hintedChannels;
  } else if (valueCount === pixelCount) {
    stride = 1;
  } else if (valueCount === pixelCount * 4) {
    stride = 4;
  } else if (valueCount === pixelCount * 2 || valueCount === pixelCount * 3) {
    const inferred = valueCount / pixelCount;
    throw texturePayloadError(
      handle,
      options,
      `the ${inferred}-channel layout is ambiguous without __vitrum_hint__.channels`,
    );
  } else {
    throw texturePayloadError(
      handle,
      options,
      `data length ${valueCount} does not describe exact 1- or 4-channel pixels for ${width}×${height}`,
    );
  }
  const expectedValueCount = pixelCount * stride;
  if (valueCount !== expectedValueCount) {
    throw texturePayloadError(
      handle,
      options,
      `data length ${valueCount} does not equal width×height×channels (${expectedValueCount})`,
    );
  }

  const backingType = Object.prototype.toString.call(src);
  let dataType = hint?.dataType;
  if (dataType == null) {
    if (backingType === '[object Uint8Array]' || backingType === '[object Uint8ClampedArray]') {
      dataType = 'uint8';
    } else if (backingType === '[object Uint16Array]') {
      dataType = 'uint16';
    } else if (backingType === '[object Float32Array]') {
      dataType = 'float32';
    } else {
      throw texturePayloadError(
        handle,
        options,
        `pixel backing ${backingType} cannot be inferred; use Uint8Array, Uint16Array, or Float32Array`,
      );
    }
  }
  const compatibleBacking =
    (dataType === 'uint8' &&
      (backingType === '[object Uint8Array]' || backingType === '[object Uint8ClampedArray]')) ||
    ((dataType === 'uint16' || dataType === 'float16' || dataType === 'half-float') &&
      backingType === '[object Uint16Array]') ||
    (dataType === 'float32' && backingType === '[object Float32Array]');
  if (!compatibleBacking) {
    const expected =
      dataType === 'uint8'
        ? 'Uint8Array or Uint8ClampedArray'
        : dataType === 'float32'
          ? 'Float32Array'
          : 'Uint16Array';
    throw texturePayloadError(
      handle,
      options,
      `dataType "${dataType}" requires ${expected}, received ${backingType}`,
    );
  }
  const sourceColorSpace =
    hint?.colorSpace === 'srgb' || hint?.colorSpace === 'linear' ? hint.colorSpace : undefined;
  return {
    width,
    height,
    pixelCount,
    data: src,
    stride,
    dataType,
    ...(sourceColorSpace ? { sourceColorSpace } : {}),
  };
}

function decodedChannel(
  inspected: InspectedTexturePixels,
  index: number,
  handle: unknown,
  options: TextureAtlasBuildOptions | undefined,
): number {
  const value = decodeTextureChannel(inspected.dataType, Number(inspected.data[index]));
  if (!Number.isFinite(value)) {
    throw texturePayloadError(
      handle,
      options,
      `decoded pixel data must be finite (value ${index} decoded to ${String(value)})`,
    );
  }
  return value;
}

/** Validate mutable source values before any decoded-layer allocation. */
function validateInspectedPixels(
  inspected: InspectedTexturePixels,
  handle: unknown,
  options?: TextureAtlasBuildOptions,
): void {
  const valueCount = inspected.pixelCount * inspected.stride;
  for (let index = 0; index < valueCount; index += 1) {
    decodedChannel(inspected, index, handle, options);
  }
}

/** Decode directly into one atlas layer without retaining an RGBA source copy. */
function blitInspectedLayer(
  inspected: InspectedTexturePixels,
  handle: unknown,
  dim: number,
  data: Float32Array,
  base: number,
  colorSpace: TextureSampleColorSpace,
  options?: TextureAtlasBuildOptions,
): void {
  const decodeSrgb = colorSpace === 'srgb' && inspected.sourceColorSpace !== 'linear';
  for (let y = 0; y < inspected.height; y += 1) {
    for (let x = 0; x < inspected.width; x += 1) {
      const sourcePixel = y * inspected.width + x;
      const s = sourcePixel * inspected.stride;
      const d = base + (y * dim + x) * 4;
      const r = decodedChannel(inspected, s, handle, options);
      const g = inspected.stride > 1 ? decodedChannel(inspected, s + 1, handle, options) : r;
      const b = inspected.stride > 2 ? decodedChannel(inspected, s + 2, handle, options) : r;
      const a = inspected.stride >= 4 ? decodedChannel(inspected, s + 3, handle, options) : 1;
      data[d] = decodeSrgb ? srgbToLinear(r) : r;
      data[d + 1] = decodeSrgb ? srgbToLinear(g) : g;
      data[d + 2] = decodeSrgb ? srgbToLinear(b) : b;
      data[d + 3] = a;
    }
  }
}

/**
 * Preflight every allocation in an RGBA32F atlas mip chain. The returned counts
 * are the exact Float32Array lengths used while packing; hostile dimensions or
 * layer counts fail deterministically instead of reaching RangeError/OOM.
 */
export function textureAtlasMipElementCounts(dim: number, layerCount: number): readonly number[] {
  if (
    !Number.isSafeInteger(dim) ||
    !Number.isSafeInteger(layerCount) ||
    dim <= 0 ||
    layerCount <= 0
  ) {
    throw new Error(
      `[pt-webgl2] material texture atlas dimensions and layer count must be positive safe integers ` +
        `(received dim=${String(dim)}, layers=${String(layerCount)})`,
    );
  }
  const maxFloats = BigInt(MAX_TEXTURE_ATLAS_FLOATS);
  const counts: number[] = [];
  let totalFloats = 0n;
  let levelDim = dim;
  let lod = 0;
  while (true) {
    const levelFloats = BigInt(levelDim) * BigInt(levelDim) * 4n * BigInt(layerCount);
    const nextTotal = totalFloats + levelFloats;
    if (levelFloats > maxFloats || nextTotal > maxFloats) {
      throw new Error(
        `[pt-webgl2] material texture atlas ${dim}²×${layerCount} mip chain exceeds the ` +
          `512 MiB CPU staging budget at mip ${lod} (${nextTotal.toString()} float values). ` +
          'Reduce material texture resolution or unique texture count.',
      );
    }
    counts.push(Number(levelFloats));
    totalFloats = nextTotal;
    if (levelDim === 1) break;
    levelDim = Math.max(1, Math.floor(levelDim / 2));
    lod += 1;
  }
  return counts;
}

/**
 * Bound the complete JS-side atlas construction before its first Float32Array.
 * Source values decode directly into level zero and immutable GPU storage
 * reserves spare layers, so the retained mip chain is the only atlas staging
 * allocation.
 */
function preflightTextureAtlasCpuAllocations(
  dim: number,
  layerCount: number,
  options?: TextureAtlasBuildOptions,
): readonly number[] {
  const mipElementCounts = textureAtlasMipElementCounts(dim, layerCount);
  const configuredMaxLayers = options?.maxArrayTextureLayers;
  if (
    configuredMaxLayers != null &&
    (!Number.isSafeInteger(configuredMaxLayers) || configuredMaxLayers <= 0)
  ) {
    throw new Error(
      `[pt-webgl2] MAX_ARRAY_TEXTURE_LAYERS must be a positive safe integer ` +
        `(received ${String(configuredMaxLayers)})`,
    );
  }
  if (configuredMaxLayers != null && layerCount > configuredMaxLayers) {
    throw new Error(
      `pt-webgl2: material texture atlas needs ${layerCount} layers but this device only supports ` +
        `${configuredMaxLayers} — reduce the number of unique material textures in the scene.`,
    );
  }
  return mipElementCounts;
}

function buildAtlasMipLevels(
  data: Float32Array,
  dim: number,
  layerCount: number,
  mipElementCounts: readonly number[],
  sourceDimensions: readonly (readonly [number, number])[],
): readonly TextureAtlasMipLevel[] {
  const levels: TextureAtlasMipLevel[] = [{ data, dim }];
  let src = data;
  let srcDim = dim;
  let lod = 1;
  while (srcDim > 1) {
    const dstDim = Math.max(1, Math.floor(srcDim / 2));
    const dstElementCount = mipElementCounts[lod];
    if (dstElementCount == null) {
      throw new Error('[pt-webgl2] internal material texture atlas mip allocation plan mismatch');
    }
    const dst = new Float32Array(dstElementCount);
    for (let layer = 0; layer < layerCount; layer += 1) {
      const srcLayerBase = layer * srcDim * srcDim * 4;
      const dstLayerBase = layer * dstDim * dstDim * 4;
      const sourceBase = sourceDimensions[layer];
      if (sourceBase == null) {
        throw new Error('[pt-webgl2] internal material texture source-dimension mismatch');
      }
      const sourceWidth = Math.max(1, Math.floor(sourceBase[0] / 2 ** (lod - 1)));
      const sourceHeight = Math.max(1, Math.floor(sourceBase[1] / 2 ** (lod - 1)));
      const targetWidth = Math.max(1, Math.floor(sourceBase[0] / 2 ** lod));
      const targetHeight = Math.max(1, Math.floor(sourceBase[1] / 2 ** lod));
      for (let y = 0; y < targetHeight; y += 1) {
        for (let x = 0; x < targetWidth; x += 1) {
          const dstOffset = dstLayerBase + (y * dstDim + x) * 4;
          const sx0 = Math.floor((x * sourceWidth) / targetWidth);
          const sx1 = Math.max(sx0 + 1, Math.floor(((x + 1) * sourceWidth) / targetWidth));
          const sy0 = Math.floor((y * sourceHeight) / targetHeight);
          const sy1 = Math.max(sy0 + 1, Math.floor(((y + 1) * sourceHeight) / targetHeight));
          const sums: [number, number, number, number] = [0, 0, 0, 0];
          let count = 0;
          for (let sy = sy0; sy < sy1; sy += 1) {
            for (let sx = sx0; sx < sx1; sx += 1) {
              const srcOffset =
                srcLayerBase +
                (Math.min(sourceHeight - 1, sy) * srcDim + Math.min(sourceWidth - 1, sx)) * 4;
              sums[0] += src[srcOffset] ?? 0;
              sums[1] += src[srcOffset + 1] ?? 0;
              sums[2] += src[srcOffset + 2] ?? 0;
              sums[3] += src[srcOffset + 3] ?? 0;
              count += 1;
            }
          }
          for (let c = 0; c < 4; c += 1) {
            dst[dstOffset + c] = sums[c]! / Math.max(1, count);
          }
        }
      }
    }
    levels.push({ data: dst, dim: dstDim });
    src = dst;
    srcDim = dstDim;
    lod += 1;
  }
  return levels;
}

/**
 * Build the material-map atlas: gather every unique authored map handle across the
 * scene materials, place each native extent in a common-dimension layer, and assign a layer index. Returns
 * `null` only when no material texture is authored. An authored map with no exact,
 * finite CPU payload throws before the candidate scene can be published.
 */
export function packTextureAtlas(
  materials: readonly MaterialSpec[],
  options?: TextureAtlasBuildOptions,
): TextureAtlas | null {
  // unique (handle, color-space role) pairs in first-seen order
  const handles: { handle: unknown; colorSpace: TextureSampleColorSpace }[] = [];
  const seen = new Map<unknown, Set<TextureSampleColorSpace>>();
  for (const m of materials) {
    for (const key of SAMPLED_MAP_KEYS) {
      const ref = m[key] as { readonly handle?: unknown } | undefined;
      collectHandle(handles, seen, ref, textureColorSpaceForMapKey(key));
    }
    collectHandle(handles, seen, m.frontLayer?.normalMap, 'linear');
    collectHandle(handles, seen, m.backLayer?.normalMap, 'linear');
  }
  if (handles.length === 0) return null;

  // Inspect every layer without allocating decoded RGBA. Only the retained mip
  // chain is staged on the CPU; source decode writes directly into level zero.
  const inspected: {
    handle: unknown;
    colorSpace: TextureSampleColorSpace;
    pixels: InspectedTexturePixels;
  }[] = [];
  let dim = 0;
  for (const { handle, colorSpace } of handles) {
    const source = inspectHandlePixels(handle, options);
    inspected.push({ handle, colorSpace, pixels: source });
    dim = Math.max(dim, source.width, source.height);
  }

  const layerCount = inspected.length;
  const mipElementCounts = preflightTextureAtlasCpuAllocations(dim, layerCount, options);

  // Validate source values before allocating level zero. Re-read through the
  // checked decoder while copying so a mutable payload still fails closed.
  for (const { handle, pixels: source } of inspected) {
    validateInspectedPixels(source, handle, options);
  }

  const data = new Float32Array(mipElementCounts[0]!);
  const layerOf = new Map<unknown, number>();
  const layerOfByColorSpace: TextureAtlasLayerMap = {
    srgb: new Map<unknown, number>(),
    linear: new Map<unknown, number>(),
    dimensions: new Map<unknown, readonly [number, number]>(),
  };
  const sourceDimensions: Array<readonly [number, number]> = [];
  inspected.forEach(({ handle, colorSpace, pixels: source }, layer) => {
    if (!layerOf.has(handle)) layerOf.set(handle, layer);
    (layerOfByColorSpace[colorSpace] as Map<unknown, number>).set(handle, layer);
    (layerOfByColorSpace.dimensions as Map<unknown, readonly [number, number]>).set(handle, [
      source.width,
      source.height,
    ]);
    sourceDimensions.push([source.width, source.height]);
    blitInspectedLayer(source, handle, dim, data, layer * dim * dim * 4, colorSpace, options);
  });
  return {
    data,
    dim,
    mipLevels: buildAtlasMipLevels(data, dim, layerCount, mipElementCounts, sourceDimensions),
    layerCount,
    layerOf,
    layerOfByColorSpace,
    sourceDimensions,
  };
}

function checkedTextureAtlasLayerCapacity(
  gl: WebGL2RenderingContext,
  atlas: TextureAtlas,
  opts?: { readonly layerCapacity?: number },
): number {
  // Size guards: exceed MAX_TEXTURE_SIZE or MAX_ARRAY_TEXTURE_LAYERS and the
  // GPU allocation silently fails on many drivers — throw an actionable error first.
  if (gl.isContextLost()) {
    throw new Error('pt-webgl2: WebGL context lost — cannot create material texture atlas');
  }
  const maxSize = gl.getParameter(gl.MAX_TEXTURE_SIZE) as number;
  if (atlas.dim > maxSize) {
    throw new Error(
      `pt-webgl2: material texture atlas needs a ${atlas.dim}² layer but this device only supports ` +
        `${maxSize}² — reduce the resolution of material textures in the scene.`,
    );
  }
  const maxLayers = gl.getParameter(gl.MAX_ARRAY_TEXTURE_LAYERS) as number;
  if (atlas.layerCount > maxLayers) {
    throw new Error(
      `pt-webgl2: material texture atlas needs ${atlas.layerCount} layers but this device only supports ` +
        `${maxLayers} — reduce the number of unique material textures in the scene.`,
    );
  }
  const layerCapacity =
    opts?.layerCapacity ?? textureAtlasLayerCapacity(atlas.layerCount, maxLayers);
  if (layerCapacity < atlas.layerCount || layerCapacity > maxLayers) {
    throw new Error(
      `pt-webgl2: material texture atlas allocation requested ${layerCapacity} layers for ` +
        `${atlas.layerCount} live layers on a device with ${maxLayers} maximum layers.`,
    );
  }
  return layerCapacity;
}

function configureTextureAtlasParameters(gl: WebGL2RenderingContext): void {
  gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
}

function uploadTextureAtlasStorage(
  gl: WebGL2RenderingContext,
  atlas: TextureAtlas,
  layerCapacity: number,
): void {
  // Reserve spare GPU layers without manufacturing capacity-sized CPU arrays.
  // texStorage3D is core WebGL2 and makes every mip available for later
  // texSubImage3D growth updates.
  gl.texStorage3D(
    gl.TEXTURE_2D_ARRAY,
    atlas.mipLevels.length,
    gl.RGBA32F,
    atlas.dim,
    atlas.dim,
    layerCapacity,
  );
  atlas.mipLevels.forEach((level, lod) => {
    gl.texSubImage3D(
      gl.TEXTURE_2D_ARRAY,
      lod,
      0,
      0,
      0,
      level.dim,
      level.dim,
      atlas.layerCount,
      gl.RGBA,
      gl.FLOAT,
      level.data,
    );
  });
  gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_BASE_LEVEL, 0);
  gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MAX_LEVEL, atlas.mipLevels.length - 1);
}

/** Upload the atlas as an RGBA32F TEXTURE_2D_ARRAY (NEAREST, ClampToEdge). */
export function uploadTextureAtlas(
  gl: WebGL2RenderingContext,
  atlas: TextureAtlas,
  opts?: { readonly layerCapacity?: number },
): WebGLTexture {
  const layerCapacity = checkedTextureAtlasLayerCapacity(gl, atlas, opts);
  const tex = gl.createTexture();
  if (tex == null)
    throw new Error('pt-webgl2: WebGL context lost — cannot create material texture atlas');
  try {
    gl.bindTexture(gl.TEXTURE_2D_ARRAY, tex);
    configureTextureAtlasParameters(gl);
    uploadTextureAtlasStorage(gl, atlas, layerCapacity);
    if (typeof gl.getError === 'function' && typeof gl.NO_ERROR === 'number') {
      const error = gl.getError();
      if (error !== gl.NO_ERROR) {
        throw new Error(
          `pt-webgl2: failed to upload material texture atlas (WebGL error 0x${error.toString(16)})`,
        );
      }
    }
    return tex;
  } catch (error) {
    gl.deleteTexture(tex);
    throw error;
  } finally {
    gl.bindTexture(gl.TEXTURE_2D_ARRAY, null);
  }
}

export function updateTextureAtlasLayers(
  gl: WebGL2RenderingContext,
  texture: WebGLTexture,
  atlas: TextureAtlas,
): void {
  gl.bindTexture(gl.TEXTURE_2D_ARRAY, texture);
  atlas.mipLevels.forEach((level, lod) => {
    gl.texSubImage3D(
      gl.TEXTURE_2D_ARRAY,
      lod,
      0,
      0,
      0,
      level.dim,
      level.dim,
      atlas.layerCount,
      gl.RGBA,
      gl.FLOAT,
      level.data,
    );
  });
}
