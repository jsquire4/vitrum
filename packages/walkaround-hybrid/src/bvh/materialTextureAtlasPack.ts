import type {
  MaterialSpec,
  TextureFilterMode,
  TextureMipFilterMode,
  TextureRef,
  TextureWrapMode,
} from '@vitrum/core';
import {
  MATERIAL_OPTICAL_META_TEXELS,
  packMaterialOpticalMeta,
} from './materialOptics.js';
import {
  isWalkaroundWebGpuTextureSource,
  type WalkaroundWebGpuTextureSource,
} from '../materialTextureSource.js';

/**
 * Pure-CPU material-texture-atlas packing (I3-2).
 *
 * This is the CPU pack half of the former `pipeline/materialTextureAtlas.ts`.
 * It imports only `@vitrum/core` types — no `GPUDevice`, no `GPUTexture` — so
 * subsystem BVH builders (`restir/bvhCore.ts`, `HybridEnginePrimitiveUpdates`)
 * can call it without an upward `subsystem → pipeline` value edge. The
 * GPU-upload half (`uploadMaterialTextureAtlas`, `writeRgba32FloatTexture`,
 * `MaterialTextureAtlasGpu`) stays in `pipeline/materialTextureAtlas.ts`, which
 * re-exports the CPU types + `packMaterialTextureAtlas` from here for back-compat.
 */

export const BASE_COLOR_MAP_META_TEX_WIDTH = 4096;
export const MATERIAL_UV_AFFINE_LANE_BUDGET = 14;
export const MATERIAL_MAP_META_TEXELS_PER_TRI = 157;
const MATERIAL_TEXTURE_ATLAS_CPU_ALLOCATION_BUDGET_BYTES = 256 * 1024 * 1024;
const MATERIAL_TEXTURE_ATLAS_CPU_AGGREGATE_BUDGET_BYTES = 512 * 1024 * 1024;
const FLOAT32_BYTES = Float32Array.BYTES_PER_ELEMENT;
const MAX_SAFE_INTEGER_BIGINT = BigInt(Number.MAX_SAFE_INTEGER);
const MATERIAL_TEXTURE_ATLAS_CPU_ALLOCATION_BUDGET_BIGINT =
  BigInt(MATERIAL_TEXTURE_ATLAS_CPU_ALLOCATION_BUDGET_BYTES);
const MATERIAL_TEXTURE_ATLAS_CPU_AGGREGATE_BUDGET_BIGINT =
  BigInt(MATERIAL_TEXTURE_ATLAS_CPU_AGGREGATE_BUDGET_BYTES);

interface MaterialAtlasCpuBudget {
  usedBytes: bigint;
}

interface MaterialAtlasProjection {
  dim: number;
  layers: number;
  elementCount: number;
  reservedBytes: bigint;
}

function checkedAllocationProduct(
  label: string,
  factors: readonly (number | bigint)[],
): bigint {
  let product = 1n;
  for (const factor of factors) {
    if (
      (typeof factor === 'number' &&
        (!Number.isSafeInteger(factor) || factor < 0)) ||
      (typeof factor === 'bigint' && factor < 0n)
    ) {
      throw new RangeError(
        `packMaterialTextureAtlas: ${label} dimensions must be non-negative safe integers.`,
      );
    }
    product *= typeof factor === 'bigint' ? factor : BigInt(factor);
  }
  return product;
}

/**
 * Reserve one Float32Array before constructing it. BigInt multiplication keeps
 * adversarial dimensions exact; the aggregate cap reflects simultaneous CPU
 * staging retained while the atlas and metadata are assembled.
 */
function reserveFloat32Allocation(
  budget: MaterialAtlasCpuBudget,
  label: string,
  factors: readonly (number | bigint)[],
): number {
  const elementCount = checkedAllocationProduct(label, factors);
  const byteCount = elementCount * BigInt(FLOAT32_BYTES);
  if (byteCount > MAX_SAFE_INTEGER_BIGINT) {
    throw new RangeError(
      `packMaterialTextureAtlas: ${label} byte length exceeds the safe integer range.`,
    );
  }
  if (byteCount > MATERIAL_TEXTURE_ATLAS_CPU_ALLOCATION_BUDGET_BIGINT) {
    throw new RangeError(
      `packMaterialTextureAtlas: ${label} requires ${byteCount.toString()} CPU bytes, ` +
      `above the ${MATERIAL_TEXTURE_ATLAS_CPU_ALLOCATION_BUDGET_BYTES}-byte ` +
      'per-allocation staging budget.',
    );
  }
  const aggregateBytes = budget.usedBytes + byteCount;
  if (aggregateBytes > MATERIAL_TEXTURE_ATLAS_CPU_AGGREGATE_BUDGET_BIGINT) {
    throw new RangeError(
      `packMaterialTextureAtlas: ${label} requires ${byteCount.toString()} CPU bytes ` +
      `(aggregate ${aggregateBytes.toString()}), above the ` +
      `${MATERIAL_TEXTURE_ATLAS_CPU_AGGREGATE_BUDGET_BYTES}-byte aggregate ` +
      'staging budget.',
    );
  }
  budget.usedBytes = aggregateBytes;
  return Number(elementCount);
}

function updateAtlasProjection(
  budget: MaterialAtlasCpuBudget,
  projection: MaterialAtlasProjection,
  width: number,
  height: number,
  layerCount: number,
): void {
  assertPositiveAtlasDimension(width, 'atlas layer width');
  assertPositiveAtlasDimension(height, 'atlas layer height');
  assertPositiveAtlasDimension(layerCount, 'atlas layer count');
  const dim = Math.max(projection.dim, width, height);
  const layers = Math.max(projection.layers, layerCount);
  const elementCount = checkedAllocationProduct(
    'material texture atlas',
    [dim, dim, 4, layers],
  );
  const byteCount = elementCount * BigInt(FLOAT32_BYTES);
  if (byteCount > MAX_SAFE_INTEGER_BIGINT) {
    throw new RangeError(
      'packMaterialTextureAtlas: material texture atlas byte length exceeds ' +
      'the safe integer range.',
    );
  }
  if (byteCount > MATERIAL_TEXTURE_ATLAS_CPU_ALLOCATION_BUDGET_BIGINT) {
    throw new RangeError(
      `packMaterialTextureAtlas: material texture atlas requires ` +
      `${byteCount.toString()} CPU bytes, above the ` +
      `${MATERIAL_TEXTURE_ATLAS_CPU_ALLOCATION_BUDGET_BYTES}-byte ` +
      'per-allocation staging budget.',
    );
  }
  const aggregateBytes =
    budget.usedBytes - projection.reservedBytes + byteCount;
  if (aggregateBytes > MATERIAL_TEXTURE_ATLAS_CPU_AGGREGATE_BUDGET_BIGINT) {
    throw new RangeError(
      `packMaterialTextureAtlas: material texture atlas requires ` +
      `${byteCount.toString()} CPU bytes (aggregate ` +
      `${aggregateBytes.toString()}), above the ` +
      `${MATERIAL_TEXTURE_ATLAS_CPU_AGGREGATE_BUDGET_BYTES}-byte aggregate ` +
      'staging budget.',
    );
  }
  budget.usedBytes = aggregateBytes;
  projection.dim = dim;
  projection.layers = layers;
  projection.elementCount = Number(elementCount);
  projection.reservedBytes = byteCount;
}

function assertPositiveAtlasDimension(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value < 1) {
    throw new RangeError(
      `packMaterialTextureAtlas: ${label} must be a positive safe integer; ` +
      `received ${String(value)}.`,
    );
  }
}

export const MATERIAL_MAP_META_TEXEL_OFFSETS = {
  BASE_COLOR: 0,
  ROUGHNESS: 2,
  METALLIC: 4,
  AO: 6,
  ALPHA: 8,
  ALPHA_COVERAGE: 10,
  EMISSIVE: 11,
  TRANSMISSION: 13,
  NORMAL: 15,
  NORMAL_SCALE: 17,
  LIGHT: 18,
  LIGHT_INTENSITY: 20,
  SPECULAR: 21,
  CLEARCOAT: 22,
  SHEEN_COLOR: 23,
  SPECULAR_COLOR: 24,
  SPECULAR_INTENSITY: 26,
  CLEARCOAT_FACTOR: 28,
  CLEARCOAT_ROUGHNESS: 30,
  SHEEN_COLOR_MAP: 32,
  SHEEN_ROUGHNESS: 34,
  CLEARCOAT_NORMAL: 36,
  CLEARCOAT_NORMAL_SCALE: 38,
  ANISOTROPY: 39,
  ANISOTROPY_SCALAR: 41,
  IRIDESCENCE: 42,
  IRIDESCENCE_THICKNESS: 44,
  IRIDESCENCE_SCALAR: 46,
  THICKNESS: 47,
  BUMP: 49,
  BUMP_SCALE: 51,
  ENV_INTENSITY: 52,
  FRONT_LAYER: 53,
  BACK_LAYER: 54,
  VOLUME_SCATTERING: 55,
  FRONT_LAYER_NORMAL: 56,
  FRONT_LAYER_NORMAL_SCALE: 58,
  BACK_LAYER_NORMAL: 59,
  BACK_LAYER_NORMAL_SCALE: 61,
  OPTICAL_HEADER: 62,
  DISPERSION_IOR_RGB: 63,
  SPECTRAL_SAMPLES: 64,
  THIN_FILM_FRONT_REFLECTANCE: 96,
  THIN_FILM_FRONT_TRANSMITTANCE: 104,
  THIN_FILM_BACK_REFLECTANCE: 112,
  THIN_FILM_BACK_TRANSMITTANCE: 120,
  UV_AFFINE_BASE: 128,
  SIDE_FLAGS: 156,
} as const;

export interface MaterialTriangleUvData {
  /** Final BVH-reordered triangle indices, stride 3. */
  readonly indices: Uint32Array;
  /** Vertex-aligned legacy UV0/UV1 streams used as affine source charts. */
  readonly uv0: Float32Array;
  readonly uv1?: Float32Array;
  /** Vertex-aligned arbitrary source streams keyed by actual TextureRef.texCoord. */
  readonly uvSets?: ReadonlyMap<number, Float32Array>;
}

export type AtlasMapField =
  | 'baseColorMap'
  | 'normalMap'
  | 'roughnessMap'
  | 'metallicMap'
  | 'aoMap'
  | 'alphaMap'
  | 'emissiveMap'
  | 'transmissionMap'
  | 'lightMap'
  | 'specularColorMap'
  | 'specularIntensityMap'
  | 'clearcoatMap'
  | 'clearcoatRoughnessMap'
  | 'clearcoatNormalMap'
  | 'sheenColorMap'
  | 'sheenRoughnessMap'
  | 'anisotropyMap'
  | 'iridescenceMap'
  | 'iridescenceThicknessMap'
  | 'thicknessMap'
  | 'bumpMap'
  | 'frontLayer.normalMap'
  | 'backLayer.normalMap';
export type AtlasColorSpace = 'srgb' | 'linear';

export interface MaterialTextureAtlasDiagnostic {
  readonly code:
    | 'unreadable-material-texture-map'
    | 'ambiguous-material-texture-stride'
    | 'invalid-material-texture-transform';
  readonly materialIndex: number;
  readonly field: AtlasMapField;
  readonly colorSpace: AtlasColorSpace;
  readonly texCoord?: number;
  readonly transformComponents?: readonly string[];
  readonly magFilter?: TextureFilterMode;
  readonly minFilter?: TextureFilterMode;
  readonly mipFilter?: TextureMipFilterMode;
  readonly pixelStride?: number;
  readonly valueCount?: number;
  readonly width?: number;
  readonly height?: number;
  readonly sourcePath?: string;
  readonly textureIndex?: number;
  readonly imageIndex?: number;
  readonly samplerIndex?: number;
  readonly imageUri?: string;
  readonly imageMimeType?: string;
  readonly textureSourceExtension?: string;
  readonly message: string;
}

const ATLAS_MAP_FIELDS: readonly { readonly field: AtlasMapField; readonly colorSpace: AtlasColorSpace }[] = [
  { field: 'baseColorMap', colorSpace: 'srgb' },
  { field: 'normalMap', colorSpace: 'linear' },
  { field: 'roughnessMap', colorSpace: 'linear' },
  { field: 'metallicMap', colorSpace: 'linear' },
  { field: 'aoMap', colorSpace: 'linear' },
  { field: 'alphaMap', colorSpace: 'linear' },
  { field: 'emissiveMap', colorSpace: 'srgb' },
  { field: 'transmissionMap', colorSpace: 'linear' },
  { field: 'lightMap', colorSpace: 'linear' },
  { field: 'specularColorMap', colorSpace: 'srgb' },
  { field: 'specularIntensityMap', colorSpace: 'linear' },
  { field: 'clearcoatMap', colorSpace: 'linear' },
  { field: 'clearcoatRoughnessMap', colorSpace: 'linear' },
  { field: 'clearcoatNormalMap', colorSpace: 'linear' },
  { field: 'sheenColorMap', colorSpace: 'srgb' },
  { field: 'sheenRoughnessMap', colorSpace: 'linear' },
  { field: 'anisotropyMap', colorSpace: 'linear' },
  { field: 'iridescenceMap', colorSpace: 'linear' },
  { field: 'iridescenceThicknessMap', colorSpace: 'linear' },
  { field: 'thicknessMap', colorSpace: 'linear' },
  { field: 'bumpMap', colorSpace: 'linear' },
  { field: 'frontLayer.normalMap', colorSpace: 'linear' },
  { field: 'backLayer.normalMap', colorSpace: 'linear' },
];

const FILTER_MODE_INDEX: Readonly<Record<TextureFilterMode, number>> = {
  nearest: 0,
  linear: 1,
};

const MIP_FILTER_INDEX: Readonly<Record<TextureMipFilterMode, number>> = {
  none: 0,
  nearest: 1,
  linear: 2,
};

export interface MaterialTextureAtlasPayload {
  readonly atlasData: Float32Array;
  readonly atlasDim: number;
  readonly atlasLayerCount: number;
  readonly atlasMipLevelCount: number;
  /** Host-owned GPU subresources copied into their assigned atlas layers. */
  readonly gpuSourceLayers: readonly MaterialTextureAtlasGpuSourceLayer[];
  readonly baseColorMetaData: Float32Array;
  readonly baseColorMetaWidth: number;
  readonly baseColorMetaHeight: number;
  readonly readableBaseColorLayerCount: number;
  readonly readableNormalLayerCount: number;
  readonly readableRoughnessLayerCount: number;
  readonly readableMetallicLayerCount: number;
  readonly readableAoLayerCount: number;
  readonly readableAlphaLayerCount: number;
  readonly readableEmissiveLayerCount: number;
  readonly readableTransmissionLayerCount: number;
  readonly readableLightLayerCount: number;
  readonly readableSpecularColorLayerCount: number;
  readonly readableSpecularIntensityLayerCount: number;
  readonly readableClearcoatLayerCount: number;
  readonly readableClearcoatRoughnessLayerCount: number;
  readonly readableClearcoatNormalLayerCount: number;
  readonly readableSheenColorLayerCount: number;
  readonly readableSheenRoughnessLayerCount: number;
  readonly readableAnisotropyLayerCount: number;
  readonly readableIridescenceLayerCount: number;
  readonly readableIridescenceThicknessLayerCount: number;
  readonly readableThicknessLayerCount: number;
  readonly readableBumpLayerCount: number;
  /**
   * Retained CPU geometry needed to rebuild per-triangle high-UV affine lanes
   * during material-only incremental updates.
   */
  readonly triangleUvs?: MaterialTriangleUvData;
  readonly diagnostics: readonly MaterialTextureAtlasDiagnostic[];
}

export interface MaterialTextureAtlasGpuSourceLayer {
  readonly layer: number;
  readonly source: WalkaroundWebGpuTextureSource;
  /**
   * True only for encoded-sRGB values in a non-sRGB GPU format. Native
   * *-srgb formats are decoded by textureLoad and therefore set this false.
   */
  readonly decodeSrgb: boolean;
}

interface RawPixels {
  readonly width: number;
  readonly height: number;
  readonly data: Float32Array;
  readonly sourceColorSpace?: 'srgb' | 'linear';
}

interface AtlasLayerRecord {
  readonly layer: number;
  readonly width: number;
  readonly height: number;
  readonly source:
    | { readonly kind: 'cpu'; readonly pixels: RawPixels }
    | {
        readonly kind: 'gpu';
        readonly descriptor: WalkaroundWebGpuTextureSource;
        readonly cpuMirror?: RawPixels;
      };
}

interface OrderedAtlasLayer {
  readonly handle: unknown;
  readonly colorSpace: AtlasColorSpace;
  readonly record: Omit<AtlasLayerRecord, 'layer'>;
}

interface ReadHandlePixelsResult {
  readonly pixels: RawPixels;
  readonly ambiguousStride?: {
    readonly pixelStride: number;
    readonly valueCount: number;
    readonly width: number;
    readonly height: number;
  };
}

interface TextureHandleHint {
  readonly channels?: 1 | 2 | 3 | 4;
  readonly dataType?: 'uint8' | 'uint16' | 'float16' | 'half-float' | 'float32';
  readonly colorSpace?: 'srgb' | 'linear';
}

interface TextureRefSourceMetadata {
  readonly path?: string;
  readonly textureIndex?: number;
  readonly imageIndex?: number;
  readonly samplerIndex?: number;
  readonly imageUri?: string;
  readonly imageMimeType?: string;
  readonly textureSourceExtension?: string;
}

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
  if ('handle' in value) return value;
  return { handle: value };
}

function finiteOrFallback(value: number | undefined, fallback: number): number {
  return Number.isFinite(value) ? (value as number) : fallback;
}

function invalidTextureTransformComponents(ref: TextureRef): string[] {
  const transform = ref.transform;
  if (transform == null) return [];
  const invalid: string[] = [];
  if (transform.offset != null) {
    if (!Number.isFinite(transform.offset[0])) invalid.push('offset.x');
    if (!Number.isFinite(transform.offset[1])) invalid.push('offset.y');
  }
  if (transform.scale != null) {
    if (!Number.isFinite(transform.scale[0])) invalid.push('scale.x');
    if (!Number.isFinite(transform.scale[1])) invalid.push('scale.y');
  }
  if (transform.rotation !== undefined && !Number.isFinite(transform.rotation)) {
    invalid.push('rotation');
  }
  return invalid;
}

function textureRefSourceMetadata(ref: TextureRef): TextureRefSourceMetadata | undefined {
  for (const symbol of Object.getOwnPropertySymbols(ref)) {
    if (symbol.description !== 'vitrum.gltf.textureRefSource') continue;
    const value = (ref as unknown as Record<symbol, unknown>)[symbol];
    if (value == null || typeof value !== 'object') return undefined;
    const record = value as Record<string, unknown>;
    return {
      ...(typeof record.path === 'string' ? { path: record.path } : {}),
      ...(typeof record.textureIndex === 'number' ? { textureIndex: record.textureIndex } : {}),
      ...(typeof record.imageIndex === 'number' ? { imageIndex: record.imageIndex } : {}),
      ...(typeof record.samplerIndex === 'number' ? { samplerIndex: record.samplerIndex } : {}),
      ...(typeof record.imageUri === 'string' ? { imageUri: record.imageUri } : {}),
      ...(typeof record.imageMimeType === 'string' ? { imageMimeType: record.imageMimeType } : {}),
      ...(typeof record.textureSourceExtension === 'string'
        ? { textureSourceExtension: record.textureSourceExtension }
        : {}),
    };
  }
  return undefined;
}

/**
 * Diagnostic source-provenance fields, spread verbatim into every
 * material-texture-atlas diagnostic. Emits ONLY the fields present on
 * `source` (so a diagnostic never carries `undefined` provenance keys —
 * matching the historical `...(source?.x !== undefined ? {x} : {})` chain
 * that was copy-pasted at every diagnostic site). Note the `path → sourcePath`
 * rename mirrors the original per-site spread exactly.
 */
function sourceMetaFields(source: TextureRefSourceMetadata | undefined): {
  sourcePath?: string;
  textureIndex?: number;
  imageIndex?: number;
  samplerIndex?: number;
  imageUri?: string;
  imageMimeType?: string;
  textureSourceExtension?: string;
} {
  return {
    ...(source?.path !== undefined ? { sourcePath: source.path } : {}),
    ...(source?.textureIndex !== undefined ? { textureIndex: source.textureIndex } : {}),
    ...(source?.imageIndex !== undefined ? { imageIndex: source.imageIndex } : {}),
    ...(source?.samplerIndex !== undefined ? { samplerIndex: source.samplerIndex } : {}),
    ...(source?.imageUri !== undefined ? { imageUri: source.imageUri } : {}),
    ...(source?.imageMimeType !== undefined ? { imageMimeType: source.imageMimeType } : {}),
    ...(source?.textureSourceExtension !== undefined
      ? { textureSourceExtension: source.textureSourceExtension }
      : {}),
  };
}

function readHandlePixels(
  handle: unknown,
  budget: MaterialAtlasCpuBudget,
  label: string,
  beforeDecode?: (width: number, height: number) => void,
): ReadHandlePixelsResult | null {
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
  if (src == null || typeof src.length !== 'number') return null;
  assertPositiveAtlasDimension(width, `${label}.width`);
  assertPositiveAtlasDimension(height, `${label}.height`);
  if (!Number.isSafeInteger(src.length) || src.length < 0) {
    throw new RangeError(
      `packMaterialTextureAtlas: ${label}.data.length must be a non-negative ` +
      `safe integer; received ${String(src.length)}.`,
    );
  }
  beforeDecode?.(width, height);
  const outputElementCount = reserveFloat32Allocation(
    budget,
    `${label} RGBA decode`,
    [width, height, 4],
  );
  const pixelCount = outputElementCount / 4;

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

  const heuristicStride = Math.max(1, Math.round(src.length / pixelCount));
  const stride = hint?.channels ?? heuristicStride;
  const ambiguousStride = hint == null && stride !== 1 && stride !== 4
    ? { pixelStride: stride, valueCount: src.length, width, height }
    : undefined;

  const isFloat = src instanceof Float32Array;
  const useHalf = hint?.dataType != null
    ? hint.dataType === 'float16' || hint.dataType === 'half-float'
    : false;
  const useUint16 = hint?.dataType != null ? hint.dataType === 'uint16' : src instanceof Uint16Array;
  const useFloat = hint?.dataType != null ? hint.dataType === 'float32' : isFloat;
  const bpe = (src as { BYTES_PER_ELEMENT?: number }).BYTES_PER_ELEMENT ?? 1;
  const intMax = useHalf || useUint16 || useFloat ? 0 : 2 ** (8 * bpe) - 1;
  const dec = (v: number): number => (
    useHalf ? halfToFloat(v) :
      useFloat ? v :
      useUint16 ? Math.min(1, Math.max(0, v / 65535)) :
      intMax > 0 ? v / intMax : v
  );

  const out = new Float32Array(outputElementCount);
  for (let p = 0; p < pixelCount; p += 1) {
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
  return {
    pixels: { width, height, data: out, ...(sourceColorSpace ? { sourceColorSpace } : {}) },
    ...(ambiguousStride ? { ambiguousStride } : {}),
  };
}

function looksLikeUnwrappedGpuTexture(handle: unknown): boolean {
  if (handle == null || typeof handle !== 'object') return false;
  const candidate = handle as Record<string, unknown>;
  return (
    typeof candidate.createView === 'function' &&
    typeof candidate.destroy === 'function' &&
    typeof candidate.format === 'string' &&
    typeof candidate.usage === 'number' &&
    typeof candidate.width === 'number' &&
    typeof candidate.height === 'number'
  );
}

function blitAtlasLayer(
  px: RawPixels,
  dim: number,
  data: Float32Array,
  layer: number,
  decodeSrgb: boolean,
): void {
  const base = layer * dim * dim * 4;
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

function samplerPolicyPacked(ref: TextureRef, texCoordCode: number): number {
  const wrapS = wrapIndex(ref.wrapS);
  const wrapT = wrapIndex(ref.wrapT);
  const mipFilter = MIP_FILTER_INDEX[ref.mipFilter ?? 'none'];
  const magFilter = FILTER_MODE_INDEX[ref.magFilter ?? 'nearest'];
  const minFilter = FILTER_MODE_INDEX[ref.minFilter ?? 'nearest'];
  return (
    wrapS +
    wrapT * 4 +
    texCoordCode * 16 +
    mipFilter * 256 +
    magFilter * 1024 +
    minFilter * 2048
  );
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

function alphaModeIndex(mode: MaterialSpec['alphaMode'] | undefined): number {
  switch (mode ?? 'opaque') {
    case 'mask':
      return 1;
    case 'blend':
      return 2;
    case 'opaque':
      return 0;
  }
}

function clampedUnit(value: number | undefined, fallback: number): number {
  return Number.isFinite(value) ? Math.min(1, Math.max(0, value ?? fallback)) : fallback;
}

function clampedNonNegative(value: number | undefined, fallback: number): number {
  return Number.isFinite(value) ? Math.max(0, value ?? fallback) : fallback;
}

function clampedSignedUnit(value: number | undefined, fallback: number): number {
  return Number.isFinite(value) ? Math.min(0.99, Math.max(-0.99, value ?? fallback)) : fallback;
}

function nonNegativeVec3Component(
  color: readonly [number, number, number] | undefined,
  index: 0 | 1 | 2,
  fallback: number,
): number {
  const value = color?.[index];
  return Number.isFinite(value) ? Math.max(0, value ?? fallback) : fallback;
}

function dielectricF0FromIor(ior: number | undefined): number {
  // KHR_materials_ior defaults to 1.5. The core contract also preserves the
  // finite IOR=0 endpoint; ((0 - 1) / (0 + 1))² is unit reflectance.
  const resolvedIor = Number.isFinite(ior) ? Math.max(0, ior ?? 1.5) : 1.5;
  const denominator = resolvedIor + 1;
  const ratio = denominator > 1e-8 ? (resolvedIor - 1) / denominator : 1;
  return Math.min(1, Math.max(0, ratio * ratio));
}

function clampedColorComponent(
  color: readonly [number, number, number] | undefined,
  index: 0 | 1 | 2,
  fallback = 1,
): number {
  const value = color?.[index];
  return Number.isFinite(value) ? Math.min(1, Math.max(0, value ?? fallback)) : fallback;
}

export function packMaterialTextureAtlas(
  materials: readonly MaterialSpec[],
  triMaterialIds: Uint32Array,
  triCount: number,
  triangleUvs?: MaterialTriangleUvData,
): MaterialTextureAtlasPayload {
  if (!Number.isSafeInteger(triCount) || triCount < 0) {
    throw new RangeError(
      'packMaterialTextureAtlas: triCount must be a non-negative safe integer; ' +
      `received ${String(triCount)}.`,
    );
  }
  const allocationBudget: MaterialAtlasCpuBudget = { usedBytes: 0n };
  const rawMetaTexels = checkedAllocationProduct(
    'material metadata texels',
    [triCount, MATERIAL_MAP_META_TEXELS_PER_TRI],
  );
  const metaTexelsBig = rawMetaTexels > 0n ? rawMetaTexels : 1n;
  const baseColorMetaWidthBig =
    metaTexelsBig < BigInt(BASE_COLOR_MAP_META_TEX_WIDTH)
      ? metaTexelsBig
      : BigInt(BASE_COLOR_MAP_META_TEX_WIDTH);
  const baseColorMetaHeightBig =
    (metaTexelsBig + baseColorMetaWidthBig - 1n) / baseColorMetaWidthBig;
  const baseColorMetaElementCount = reserveFloat32Allocation(
    allocationBudget,
    'material metadata atlas',
    [baseColorMetaWidthBig, baseColorMetaHeightBig, 4n],
  );
  const baseColorMetaWidth = Number(baseColorMetaWidthBig);
  const baseColorMetaHeight = Number(baseColorMetaHeightBig);
  const opticalMetaElementCount = MATERIAL_OPTICAL_META_TEXELS * 4;
  reserveFloat32Allocation(
    allocationBudget,
    'material optical metadata',
    [BigInt(materials.length) + 1n, opticalMetaElementCount],
  );
  const atlasProjection: MaterialAtlasProjection = {
    dim: 1,
    layers: 1,
    elementCount: 0,
    reservedBytes: 0n,
  };
  updateAtlasProjection(allocationBudget, atlasProjection, 1, 1, 1);

  const readable = new Map<unknown, Partial<Record<AtlasColorSpace, AtlasLayerRecord>>>();
  const ordered: OrderedAtlasLayer[] = [];
  const diagnostics: MaterialTextureAtlasDiagnostic[] = [];
  const fieldLayers: Record<AtlasMapField, Set<number>> = {
    baseColorMap: new Set<number>(),
    normalMap: new Set<number>(),
    roughnessMap: new Set<number>(),
    metallicMap: new Set<number>(),
    aoMap: new Set<number>(),
    alphaMap: new Set<number>(),
    emissiveMap: new Set<number>(),
    transmissionMap: new Set<number>(),
    lightMap: new Set<number>(),
    specularColorMap: new Set<number>(),
    specularIntensityMap: new Set<number>(),
    clearcoatMap: new Set<number>(),
    clearcoatRoughnessMap: new Set<number>(),
    clearcoatNormalMap: new Set<number>(),
    sheenColorMap: new Set<number>(),
    sheenRoughnessMap: new Set<number>(),
    anisotropyMap: new Set<number>(),
    iridescenceMap: new Set<number>(),
    iridescenceThicknessMap: new Set<number>(),
    thicknessMap: new Set<number>(),
    bumpMap: new Set<number>(),
    'frontLayer.normalMap': new Set<number>(),
    'backLayer.normalMap': new Set<number>(),
  };

  const materialTextureRefForField = (
    material: MaterialSpec | undefined,
    field: AtlasMapField,
  ): TextureRef | null => {
    if (field === 'frontLayer.normalMap') {
      return asTextureRef(material?.frontLayer?.normalMap);
    }
    if (field === 'backLayer.normalMap') {
      return asTextureRef(material?.backLayer?.normalMap);
    }
    const topLevelField = field;
    return asTextureRef(material?.[topLevelField]);
  };

  const collect = (
    material: MaterialSpec,
    materialIndex: number,
    field: AtlasMapField,
    colorSpace: AtlasColorSpace,
  ): void => {
    const ref = materialTextureRefForField(material, field);
    if (ref?.handle == null) return;
    const texCoord = ref.texCoord ?? 0;
    const pushInvalidTransformDiagnostic = (): void => {
      const transformComponents = invalidTextureTransformComponents(ref);
      if (transformComponents.length === 0) return;
      const source = textureRefSourceMetadata(ref);
      diagnostics.push({
        code: 'invalid-material-texture-transform',
        materialIndex,
        field,
        colorSpace,
        texCoord,
        transformComponents,
        ...sourceMetaFields(source),
        message:
          `${field} texture transform contains non-finite component(s) ` +
          `${transformComponents.join(', ')}` +
          `${source?.path !== undefined ? ` at ${source.path}` : ''}; ` +
          'invalid components are replaced with the identity transform fallback.',
      });
    };
    let perHandle = readable.get(ref.handle);
    const existing = perHandle?.[colorSpace];
    if (existing != null) {
      pushInvalidTransformDiagnostic();
      fieldLayers[field].add(existing.layer);
      return;
    }

    if (isWalkaroundWebGpuTextureSource(ref.handle)) {
      if (colorSpace === 'linear' && ref.handle.colorSpace !== 'linear') {
        const source = textureRefSourceMetadata(ref);
        throw new RangeError(
          `packMaterialTextureAtlas: ${field} is a linear-data map, but its WebGPU ` +
          `source declares ${ref.handle.colorSpace} values` +
          `${source?.path !== undefined ? ` at ${source.path}` : ''}.`,
        );
      }
      assertPositiveAtlasDimension(
        ref.handle.width,
        `material ${materialIndex} ${field} GPU source width`,
      );
      assertPositiveAtlasDimension(
        ref.handle.height,
        `material ${materialIndex} ${field} GPU source height`,
      );
      updateAtlasProjection(
        allocationBudget,
        atlasProjection,
        ref.handle.width,
        ref.handle.height,
        ordered.length + 1,
      );
      let cpuMirror: RawPixels | undefined;
      if (ref.handle.cpuMirror != null) {
        cpuMirror = readHandlePixels(
          ref.handle.cpuMirror,
          allocationBudget,
          `GPU source layer ${ordered.length} CPU mirror`,
        )?.pixels;
        if (cpuMirror == null) {
          throw new TypeError(
            `packMaterialTextureAtlas: GPU source layer ${ordered.length} has ` +
            'an invalid CPU mirror.',
          );
        }
      }
      pushInvalidTransformDiagnostic();
      const layer = ordered.length;
      const record: AtlasLayerRecord = {
        layer,
        width: ref.handle.width,
        height: ref.handle.height,
        source: {
          kind: 'gpu',
          descriptor: ref.handle,
          ...(cpuMirror != null ? { cpuMirror } : {}),
        },
      };
      ordered.push({
        handle: ref.handle,
        colorSpace,
        record: {
          width: record.width,
          height: record.height,
          source: record.source,
        },
      });
      perHandle ??= {};
      perHandle[colorSpace] = record;
      readable.set(ref.handle, perHandle);
      fieldLayers[field].add(layer);
      return;
    }
    if (looksLikeUnwrappedGpuTexture(ref.handle)) {
      const source = textureRefSourceMetadata(ref);
      throw new TypeError(
        `packMaterialTextureAtlas: ${field} received a raw GPUTexture` +
        `${source?.path !== undefined ? ` at ${source.path}` : ''}. ` +
        'Wrap it with createWalkaroundWebGpuTextureSource so device, format, ' +
        'ownership, selected subresource, and color space are explicit.',
      );
    }

    const read = readHandlePixels(
      ref.handle,
      allocationBudget,
      `material ${materialIndex} ${field}`,
      (width, height) => updateAtlasProjection(
        allocationBudget,
        atlasProjection,
        width,
        height,
        ordered.length + 1,
      ),
    );
    if (read == null) {
      const source = textureRefSourceMetadata(ref);
      diagnostics.push({
        code: 'unreadable-material-texture-map',
        materialIndex,
        field,
        colorSpace,
        ...sourceMetaFields(source),
        message:
          `${field} texture handle is not an atlas-supported source ` +
          `(expected raw {width,height,data}, a DataTexture-shaped image, or ` +
          `createWalkaroundWebGpuTextureSource(...))` +
          `${source?.path !== undefined ? ` at ${source.path}` : ''}; the map is ignored.`,
      });
      return;
    }
    if (read.ambiguousStride != null) {
      const source = textureRefSourceMetadata(ref);
      diagnostics.push({
        code: 'ambiguous-material-texture-stride',
        materialIndex,
        field,
        colorSpace,
        pixelStride: read.ambiguousStride.pixelStride,
        valueCount: read.ambiguousStride.valueCount,
        width: read.ambiguousStride.width,
        height: read.ambiguousStride.height,
        ...sourceMetaFields(source),
        message:
          `${field} texture handle has ambiguous pixel stride ${read.ambiguousStride.pixelStride} ` +
          `(${read.ambiguousStride.valueCount} values / ` +
          `${read.ambiguousStride.width}x${read.ambiguousStride.height} pixels)` +
          `${source?.path !== undefined ? ` at ${source.path}` : ''}; ` +
          'attach __vitrum_hint__ = { channels: N } to decode it deterministically.',
      });
    }
    pushInvalidTransformDiagnostic();
    const pixels = read.pixels;
    const layer = ordered.length;
    const record: AtlasLayerRecord = {
      layer,
      width: pixels.width,
      height: pixels.height,
      source: { kind: 'cpu', pixels },
    };
    ordered.push({
      handle: ref.handle,
      colorSpace,
      record: {
        width: record.width,
        height: record.height,
        source: record.source,
      },
    });
    perHandle ??= {};
    perHandle[colorSpace] = record;
    readable.set(ref.handle, perHandle);
    fieldLayers[field].add(layer);
  };

  materials.forEach((material, materialIndex) => {
    for (const { field, colorSpace } of ATLAS_MAP_FIELDS) {
      collect(material, materialIndex, field, colorSpace);
    }
  });

  // Compact arbitrary authored texCoord indices into the 4-bit shader lane.
  // UV0/UV1 keep codes 0/1; the remaining fourteen codes are assigned only to
  // Atlas-backed maps that can actually be sampled by this renderer.
  const activeHighTexCoords = new Set<number>();
  const materialHighTexCoords = materials.map(() => new Set<number>());
  materials.forEach((material, materialIndex) => {
    for (const { field, colorSpace } of ATLAS_MAP_FIELDS) {
      const ref = materialTextureRefForField(material, field);
      const texCoord = ref?.texCoord ?? 0;
      if (
        texCoord > 1 && ref?.handle != null &&
        readable.get(ref.handle)?.[colorSpace]?.layer != null
      ) {
        activeHighTexCoords.add(texCoord);
        materialHighTexCoords[materialIndex]!.add(texCoord);
      }
    }
  });
  const sortedHighTexCoords = [...activeHighTexCoords].sort((a, b) => a - b);
  if (sortedHighTexCoords.length > MATERIAL_UV_AFFINE_LANE_BUDGET) {
    throw new RangeError(
      `packMaterialTextureAtlas: scene references ${sortedHighTexCoords.length} atlas-backed high UV sets ` +
      `(${sortedHighTexCoords.join(', ')}), exceeding the ${MATERIAL_UV_AFFINE_LANE_BUDGET}-lane material UV budget.`,
    );
  }
  const compactTexCoordCode = new Map<number, number>([[0, 0], [1, 1]]);
  sortedHighTexCoords.forEach((texCoord, lane) => compactTexCoordCode.set(texCoord, lane + 2));
  if (sortedHighTexCoords.length > 0 && triangleUvs == null) {
    throw new RangeError(
      'packMaterialTextureAtlas: atlas-backed material maps reference UV sets above 1, but no triangle UV streams were supplied.',
    );
  }

  let atlasDim = 1;
  for (const entry of ordered) {
    assertPositiveAtlasDimension(entry.record.width, 'atlas layer width');
    assertPositiveAtlasDimension(entry.record.height, 'atlas layer height');
    atlasDim = Math.max(atlasDim, entry.record.width, entry.record.height);
  }
  const atlasLayerCount = Math.max(1, ordered.length);
  if (
    atlasDim !== atlasProjection.dim ||
    atlasLayerCount !== atlasProjection.layers
  ) {
    throw new Error(
      'packMaterialTextureAtlas: internal atlas allocation projection drifted.',
    );
  }
  const atlasMipLevelCount = Math.floor(Math.log2(atlasDim)) + 1;
  const atlasData = new Float32Array(atlasProjection.elementCount);
  const gpuSourceLayers: MaterialTextureAtlasGpuSourceLayer[] = [];
  if (ordered.length === 0) {
    atlasData.set([1, 1, 1, 1]);
  } else {
    ordered.forEach((entry, layer) => {
      if (entry.record.source.kind === 'gpu') {
        const source = entry.record.source.descriptor;
        gpuSourceLayers.push({
          layer,
          source,
          decodeSrgb: (
            entry.colorSpace === 'srgb' &&
            source.colorSpace === 'srgb' &&
            !source.format.endsWith('-srgb')
          ),
        });
        if (entry.record.source.cpuMirror != null) {
          const mirror = entry.record.source.cpuMirror;
          blitAtlasLayer(
            mirror,
            atlasDim,
            atlasData,
            layer,
            entry.colorSpace === 'srgb' && mirror.sourceColorSpace !== 'linear',
          );
        }
        return;
      }
      const pixels = entry.record.source.pixels;
      blitAtlasLayer(
        pixels,
        atlasDim,
        atlasData,
        layer,
        entry.colorSpace === 'srgb' && pixels.sourceColorSpace !== 'linear',
      );
    });
  }

  const baseColorMetaData = new Float32Array(baseColorMetaElementCount);
  const opticalMetaByMaterial = materials.map((material) => packMaterialOpticalMeta(material));
  const emptyOpticalMeta = new Float32Array(opticalMetaElementCount);

  const writeMapMeta = (
    mat: MaterialSpec | undefined,
    field: AtlasMapField,
    colorSpace: AtlasColorSpace,
    texel: number,
  ): void => {
    const ref = materialTextureRefForField(mat, field);
    if (ref?.handle == null) {
      writeDisabledMeta(baseColorMetaData, texel);
      return;
    }
    const texCoord = ref.texCoord ?? 0;
    const texCoordCode = compactTexCoordCode.get(texCoord);
    const layer = texCoordCode == null ? undefined : readable.get(ref.handle)?.[colorSpace]?.layer;
    if (texCoordCode == null || layer == null) {
      writeDisabledMeta(baseColorMetaData, texel);
      return;
    }
    const t = ref.transform;
    const rotation = finiteOrFallback(t?.rotation, 0);
    const b0 = texel * 4;
    const b1 = b0 + 4;
    baseColorMetaData[b0] = layer;
    baseColorMetaData[b0 + 1] = samplerPolicyPacked(ref, texCoordCode);
    baseColorMetaData[b0 + 2] = finiteOrFallback(t?.offset?.[0], 0);
    baseColorMetaData[b0 + 3] = finiteOrFallback(t?.offset?.[1], 0);
    baseColorMetaData[b1] = finiteOrFallback(t?.scale?.[0], 1);
    baseColorMetaData[b1 + 1] = finiteOrFallback(t?.scale?.[1], 1);
    baseColorMetaData[b1 + 2] = Math.cos(rotation);
    baseColorMetaData[b1 + 3] = Math.sin(rotation);
  };

  // Single-scalar meta texel: value in .x, .yzw zeroed. Shared by the
  // normal-scale / clearcoat-normal-scale / face-layer-normal-scale /
  // light-map-intensity / env-map-intensity closures (identical 4-line body).
  const writeScalarMeta = (texel: number, value: number): void => {
    const b = texel * 4;
    baseColorMetaData[b] = value;
    baseColorMetaData[b + 1] = 0;
    baseColorMetaData[b + 2] = 0;
    baseColorMetaData[b + 3] = 0;
  };

  const writeAlphaCoverageMeta = (mat: MaterialSpec | undefined, texel: number): void => {
    const b = texel * 4;
    baseColorMetaData[b] = alphaModeIndex(mat?.alphaMode);
    baseColorMetaData[b + 1] = clampedUnit(mat?.opacity, 1);
    baseColorMetaData[b + 2] = clampedUnit(mat?.alphaCutoff, 0.5);
    baseColorMetaData[b + 3] = 0;
  };

  const writeNormalScaleMeta = (mat: MaterialSpec | undefined, texel: number): void => {
    writeScalarMeta(texel, Number.isFinite(mat?.normalScale)
      ? Math.max(0, mat?.normalScale ?? 1)
      : 1);
  };

  const writeFaceLayerNormalScaleMeta = (
    layer: MaterialSpec['frontLayer']   | undefined,
    texel: number,
  ): void => {
    writeScalarMeta(texel, Number.isFinite(layer?.normalScale)
      ? Math.max(0, layer?.normalScale ?? 1)
      : 1);
  };

  const writeClearcoatNormalScaleMeta = (mat: MaterialSpec | undefined, texel: number): void => {
    writeScalarMeta(texel, Number.isFinite(mat?.clearcoatNormalScale)
      ? Math.max(0, mat?.clearcoatNormalScale ?? 1)
      : 1);
  };

  const writeLightMapIntensityMeta = (mat: MaterialSpec | undefined, texel: number): void => {
    writeScalarMeta(texel, Number.isFinite(mat?.lightMapIntensity)
      ? Math.max(0, mat?.lightMapIntensity ?? 1)
      : 1);
  };

  const writeSpecularMeta = (mat: MaterialSpec | undefined, texel: number): void => {
    const b = texel * 4;
    const dielectricF0 = dielectricF0FromIor(mat?.ior);
    // Store absolute dielectric F0. Nonnegative KHR specularColor factors are
    // intentionally unbounded, so authored values above one remain observable.
    baseColorMetaData[b] =
      dielectricF0 * nonNegativeVec3Component(mat?.specularColor, 0, 1);
    baseColorMetaData[b + 1] =
      dielectricF0 * nonNegativeVec3Component(mat?.specularColor, 1, 1);
    baseColorMetaData[b + 2] =
      dielectricF0 * nonNegativeVec3Component(mat?.specularColor, 2, 1);
    baseColorMetaData[b + 3] = clampedUnit(mat?.specularIntensity, 1);
  };

  const writeClearcoatMeta = (mat: MaterialSpec | undefined, texel: number): void => {
    const b = texel * 4;
    baseColorMetaData[b] = clampedUnit(mat?.clearcoat, 0);
    baseColorMetaData[b + 1] = clampedUnit(mat?.clearcoatRoughness, 0);
    baseColorMetaData[b + 2] = clampedUnit(mat?.sheen, 0);
    baseColorMetaData[b + 3] = clampedUnit(mat?.sheenRoughness, 0);
  };

  const writeSheenColorMeta = (mat: MaterialSpec | undefined, texel: number): void => {
    const b = texel * 4;
    baseColorMetaData[b] = clampedColorComponent(mat?.sheenColor, 0, 0);
    baseColorMetaData[b + 1] = clampedColorComponent(mat?.sheenColor, 1, 0);
    baseColorMetaData[b + 2] = clampedColorComponent(mat?.sheenColor, 2, 0);
    baseColorMetaData[b + 3] = 0;
  };

  const writeAnisotropyMeta = (mat: MaterialSpec | undefined, texel: number): void => {
    const b = texel * 4;
    baseColorMetaData[b] = clampedUnit(mat?.anisotropy, 0);
    baseColorMetaData[b + 1] = Number.isFinite(mat?.anisotropyRotation)
      ? mat?.anisotropyRotation ?? 0
      : 0;
    baseColorMetaData[b + 2] = 0;
    baseColorMetaData[b + 3] = 0;
  };

  const writeIridescenceMeta = (mat: MaterialSpec | undefined, texel: number): void => {
    const b = texel * 4;
    const range = mat?.iridescenceThicknessRange;
    baseColorMetaData[b] = clampedUnit(mat?.iridescence, 0);
    baseColorMetaData[b + 1] = Number.isFinite(mat?.iridescenceIor)
      ? Math.max(1, mat?.iridescenceIor ?? 1.3)
      : 1.3;
    baseColorMetaData[b + 2] = Number.isFinite(range?.[0])
      ? Math.max(0, range?.[0] ?? 100)
      : 100;
    baseColorMetaData[b + 3] = Number.isFinite(range?.[1])
      ? Math.max(0, range?.[1] ?? 400)
      : 400;
  };

  const writeBumpScaleMeta = (mat: MaterialSpec | undefined, texel: number): void => {
    const b = texel * 4;
    const ref = asTextureRef(mat?.bumpMap);
    const texCoord = ref?.texCoord ?? 0;
    const record = ref?.handle != null && compactTexCoordCode.has(texCoord)
      ? readable.get(ref.handle)?.linear
      : undefined;
    baseColorMetaData[b] = Number.isFinite(mat?.bumpScale)
      ? mat?.bumpScale ?? 1
      : 1;
    baseColorMetaData[b + 1] = record?.width ?? 0;
    baseColorMetaData[b + 2] = record?.height ?? 0;
    baseColorMetaData[b + 3] = 0;
  };

  const writeEnvMapIntensityMeta = (mat: MaterialSpec | undefined, texel: number): void => {
    writeScalarMeta(texel, clampedNonNegative(mat?.envMapIntensity, 1));
  };

  const writeFaceLayerMeta = (
    layer: MaterialSpec['frontLayer']   | undefined,
    texel: number,
  ): void => {
    const b = texel * 4;
    baseColorMetaData[b] = clampedColorComponent(layer?.transmission, 0);
    baseColorMetaData[b + 1] = clampedColorComponent(layer?.transmission, 1);
    baseColorMetaData[b + 2] = clampedColorComponent(layer?.transmission, 2);
    baseColorMetaData[b + 3] = Number.isFinite(layer?.roughness)
      ? clampedUnit(layer?.roughness, 0)
      : -1;
  };

  const writeVolumeScatteringMeta = (mat: MaterialSpec | undefined, texel: number): void => {
    const b = texel * 4;
    const scalar = clampedNonNegative(mat?.scatteringCoefficient, 0);
    baseColorMetaData[b] = nonNegativeVec3Component(mat?.scatteringCoefficientRGB, 0, scalar);
    baseColorMetaData[b + 1] = nonNegativeVec3Component(mat?.scatteringCoefficientRGB, 1, scalar);
    baseColorMetaData[b + 2] = nonNegativeVec3Component(mat?.scatteringCoefficientRGB, 2, scalar);
    baseColorMetaData[b + 3] = clampedSignedUnit(mat?.scatteringAnisotropy, 0);
  };

  const uvAt = (stream: Float32Array, vertexIndex: number): readonly [number, number] => [
    stream[vertexIndex * 2] ?? 0,
    stream[vertexIndex * 2 + 1] ?? 0,
  ];
  const determinant = (
    a: readonly [number, number], b: readonly [number, number], c: readonly [number, number],
  ): number => (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
  const writeUvAffine = (tri: number, texCoord: number, compactCode: number): void => {
    const geometry = triangleUvs!;
    const target = geometry.uvSets?.get(texCoord);
    if (target == null) {
      throw new RangeError(
        `packMaterialTextureAtlas: triangle ${tri} material references texCoord ${texCoord}, but that UV stream is missing.`,
      );
    }
    const i0 = geometry.indices[tri * 3] ?? 0;
    const i1 = geometry.indices[tri * 3 + 1] ?? 0;
    const i2 = geometry.indices[tri * 3 + 2] ?? 0;
    const ta = uvAt(target, i0); const tb = uvAt(target, i1); const tc = uvAt(target, i2);
    if (![...ta, ...tb, ...tc].every(Number.isFinite)) {
      throw new RangeError(
        `packMaterialTextureAtlas: triangle ${tri} material references texCoord ${texCoord}, but that primitive does not supply the UV stream.`,
      );
    }
    const constantTarget =
      Math.abs(ta[0] - tb[0]) + Math.abs(ta[1] - tb[1]) +
      Math.abs(ta[0] - tc[0]) + Math.abs(ta[1] - tc[1]) <= 1e-12;
    let source = geometry.uv0;
    let sourceSelector = 0;
    let sa = uvAt(source, i0); let sb = uvAt(source, i1); let sc = uvAt(source, i2);
    let det = determinant(sa, sb, sc);
    if (Math.abs(det) <= 1e-12 && geometry.uv1 != null) {
      source = geometry.uv1;
      sourceSelector = 1;
      sa = uvAt(source, i0); sb = uvAt(source, i1); sc = uvAt(source, i2);
      det = determinant(sa, sb, sc);
    }
    let m00 = 0; let m01 = 0; let b0 = ta[0];
    let m10 = 0; let m11 = 0; let b1 = ta[1];
    if (!constantTarget) {
      if (Math.abs(det) <= 1e-12) {
        throw new RangeError(
          `packMaterialTextureAtlas: triangle ${tri} texCoord ${texCoord} cannot be reconstructed because both UV0 and UV1 source charts are degenerate.`,
        );
      }
      const inv00 = (sc[1] - sa[1]) / det;
      const inv01 = -(sc[0] - sa[0]) / det;
      const inv10 = -(sb[1] - sa[1]) / det;
      const inv11 = (sb[0] - sa[0]) / det;
      const du0 = tb[0] - ta[0]; const du1 = tc[0] - ta[0];
      const dv0 = tb[1] - ta[1]; const dv1 = tc[1] - ta[1];
      m00 = du0 * inv00 + du1 * inv10;
      m01 = du0 * inv01 + du1 * inv11;
      m10 = dv0 * inv00 + dv1 * inv10;
      m11 = dv0 * inv01 + dv1 * inv11;
      b0 = ta[0] - m00 * sa[0] - m01 * sa[1];
      b1 = ta[1] - m10 * sa[0] - m11 * sa[1];
    }
    const lane = compactCode - 2;
    const texel0 = tri * MATERIAL_MAP_META_TEXELS_PER_TRI + MATERIAL_MAP_META_TEXEL_OFFSETS.UV_AFFINE_BASE + lane * 2;
    const base = texel0 * 4;
    baseColorMetaData.set([m00, m01, b0, sourceSelector, m10, m11, b1, 1], base);
  };

  for (let tri = 0; tri < triCount; tri += 1) {
    const baseTexel = tri * MATERIAL_MAP_META_TEXELS_PER_TRI;
    const mat = materials[triMaterialIds[tri] ?? 0];
    const offsets = MATERIAL_MAP_META_TEXEL_OFFSETS;
    writeMapMeta(mat, 'baseColorMap', 'srgb', baseTexel + offsets.BASE_COLOR);
    writeMapMeta(mat, 'roughnessMap', 'linear', baseTexel + offsets.ROUGHNESS);
    writeMapMeta(mat, 'metallicMap', 'linear', baseTexel + offsets.METALLIC);
    writeMapMeta(mat, 'aoMap', 'linear', baseTexel + offsets.AO);
    writeMapMeta(mat, 'alphaMap', 'linear', baseTexel + offsets.ALPHA);
    writeAlphaCoverageMeta(mat, baseTexel + offsets.ALPHA_COVERAGE);
    writeScalarMeta(baseTexel + offsets.SIDE_FLAGS, mat?.doubleSided === true ? 1 : 0);
    writeMapMeta(mat, 'emissiveMap', 'srgb', baseTexel + offsets.EMISSIVE);
    writeMapMeta(mat, 'transmissionMap', 'linear', baseTexel + offsets.TRANSMISSION);
    writeMapMeta(mat, 'normalMap', 'linear', baseTexel + offsets.NORMAL);
    writeNormalScaleMeta(mat, baseTexel + offsets.NORMAL_SCALE);
    writeMapMeta(mat, 'lightMap', 'linear', baseTexel + offsets.LIGHT);
    writeLightMapIntensityMeta(mat, baseTexel + offsets.LIGHT_INTENSITY);
    writeSpecularMeta(mat, baseTexel + offsets.SPECULAR);
    writeClearcoatMeta(mat, baseTexel + offsets.CLEARCOAT);
    writeSheenColorMeta(mat, baseTexel + offsets.SHEEN_COLOR);
    writeMapMeta(mat, 'specularColorMap', 'srgb', baseTexel + offsets.SPECULAR_COLOR);
    writeMapMeta(mat, 'specularIntensityMap', 'linear', baseTexel + offsets.SPECULAR_INTENSITY);
    writeMapMeta(mat, 'clearcoatMap', 'linear', baseTexel + offsets.CLEARCOAT_FACTOR);
    writeMapMeta(mat, 'clearcoatRoughnessMap', 'linear', baseTexel + offsets.CLEARCOAT_ROUGHNESS);
    writeMapMeta(mat, 'sheenColorMap', 'srgb', baseTexel + offsets.SHEEN_COLOR_MAP);
    writeMapMeta(mat, 'sheenRoughnessMap', 'linear', baseTexel + offsets.SHEEN_ROUGHNESS);
    writeMapMeta(mat, 'clearcoatNormalMap', 'linear', baseTexel + offsets.CLEARCOAT_NORMAL);
    writeClearcoatNormalScaleMeta(mat, baseTexel + offsets.CLEARCOAT_NORMAL_SCALE);
    writeMapMeta(mat, 'anisotropyMap', 'linear', baseTexel + offsets.ANISOTROPY);
    writeAnisotropyMeta(mat, baseTexel + offsets.ANISOTROPY_SCALAR);
    writeMapMeta(mat, 'iridescenceMap', 'linear', baseTexel + offsets.IRIDESCENCE);
    writeMapMeta(mat, 'iridescenceThicknessMap', 'linear', baseTexel + offsets.IRIDESCENCE_THICKNESS);
    writeIridescenceMeta(mat, baseTexel + offsets.IRIDESCENCE_SCALAR);
    writeMapMeta(mat, 'thicknessMap', 'linear', baseTexel + offsets.THICKNESS);
    writeMapMeta(mat, 'bumpMap', 'linear', baseTexel + offsets.BUMP);
    writeBumpScaleMeta(mat, baseTexel + offsets.BUMP_SCALE);
    writeEnvMapIntensityMeta(mat, baseTexel + offsets.ENV_INTENSITY);
    writeFaceLayerMeta(mat?.frontLayer, baseTexel + offsets.FRONT_LAYER);
    writeFaceLayerMeta(mat?.backLayer, baseTexel + offsets.BACK_LAYER);
    writeVolumeScatteringMeta(mat, baseTexel + offsets.VOLUME_SCATTERING);
    writeMapMeta(mat, 'frontLayer.normalMap', 'linear', baseTexel + offsets.FRONT_LAYER_NORMAL);
    writeFaceLayerNormalScaleMeta(mat?.frontLayer, baseTexel + offsets.FRONT_LAYER_NORMAL_SCALE);
    writeMapMeta(mat, 'backLayer.normalMap', 'linear', baseTexel + offsets.BACK_LAYER_NORMAL);
    writeFaceLayerNormalScaleMeta(mat?.backLayer, baseTexel + offsets.BACK_LAYER_NORMAL_SCALE);
    baseColorMetaData.set(
      opticalMetaByMaterial[triMaterialIds[tri] ?? 0] ?? emptyOpticalMeta,
      (baseTexel + offsets.OPTICAL_HEADER) * 4,
    );
    const materialIndex = triMaterialIds[tri] ?? 0;
    for (const texCoord of materialHighTexCoords[materialIndex] ?? []) {
      writeUvAffine(tri, texCoord, compactTexCoordCode.get(texCoord)!);
    }
  }

  return {
    atlasData,
    atlasDim,
    atlasLayerCount,
    atlasMipLevelCount,
    gpuSourceLayers,
    baseColorMetaData,
    baseColorMetaWidth,
    baseColorMetaHeight,
    readableBaseColorLayerCount: fieldLayers.baseColorMap.size,
    readableNormalLayerCount: new Set<number>([
      ...fieldLayers.normalMap,
      ...fieldLayers['frontLayer.normalMap'],
      ...fieldLayers['backLayer.normalMap'],
    ]).size,
    readableRoughnessLayerCount: fieldLayers.roughnessMap.size,
    readableMetallicLayerCount: fieldLayers.metallicMap.size,
    readableAoLayerCount: fieldLayers.aoMap.size,
    readableAlphaLayerCount: fieldLayers.alphaMap.size,
    readableEmissiveLayerCount: fieldLayers.emissiveMap.size,
    readableTransmissionLayerCount: fieldLayers.transmissionMap.size,
    readableLightLayerCount: fieldLayers.lightMap.size,
    readableSpecularColorLayerCount: fieldLayers.specularColorMap.size,
    readableSpecularIntensityLayerCount: fieldLayers.specularIntensityMap.size,
    readableClearcoatLayerCount: fieldLayers.clearcoatMap.size,
    readableClearcoatRoughnessLayerCount: fieldLayers.clearcoatRoughnessMap.size,
    readableClearcoatNormalLayerCount: fieldLayers.clearcoatNormalMap.size,
    readableSheenColorLayerCount: fieldLayers.sheenColorMap.size,
    readableSheenRoughnessLayerCount: fieldLayers.sheenRoughnessMap.size,
    readableAnisotropyLayerCount: fieldLayers.anisotropyMap.size,
    readableIridescenceLayerCount: fieldLayers.iridescenceMap.size,
    readableIridescenceThicknessLayerCount: fieldLayers.iridescenceThicknessMap.size,
    readableThicknessLayerCount: fieldLayers.thicknessMap.size,
    readableBumpLayerCount: fieldLayers.bumpMap.size,
    ...(triangleUvs != null ? { triangleUvs } : {}),
    diagnostics,
  };
}
