import type {
  MaterialSpec,
  TextureFilterMode,
  TextureMipFilterMode,
  TextureRef,
  TextureWrapMode,
} from '@vitrum/core';
import {
  materialSpecScalarEmissiveLe,
  packNonNegativeRadianceScalarF32,
  packRadianceRgbProductF32,
  srgbToLinear,
} from '@vitrum/shared-bvh';
import {
  MATERIAL_OPTICAL_META_TEXELS,
  packMaterialOpticalMeta,
} from './materialOptics.js';
import {
  isWalkaroundWebGpuTextureSource,
  walkaroundTextureFormatChannelCount,
  type WalkaroundWebGpuTextureSource,
} from '../materialTextureSource.js';
import {
  materialTextureAtlasEncodingForDataType,
  materialTextureAtlasEncodingForGpuFormat,
  materialTextureAtlasEncodingPlaneCount,
  generateMaterialTextureAtlasMip,
  packMaterialTextureAtlasPixels,
  unpackMaterialTextureAtlasPixels,
  type MaterialTextureAtlasEncoding,
} from './materialTextureAtlasCodec.js';
import { assertWalkaroundEnvironmentScaleF32 } from '../environment/environmentRadianceScale.js';

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
export const MATERIAL_META_HEADER_TEXELS = 4;
export const MATERIAL_ATLAS_MAX_MIP_LEVELS = 16;
export const MATERIAL_ATLAS_ADDRESS_TEXELS_PER_LAYER =
  2 + MATERIAL_ATLAS_MAX_MIP_LEVELS;
/** Largest integer that remains exact when stored numerically in binary32. */
export const MATERIAL_META_MAX_EXACT_UINT = 2 ** 24;
const MATERIAL_TEXTURE_ATLAS_CPU_ALLOCATION_BUDGET_BYTES = 256 * 1024 * 1024;
export const MATERIAL_TEXTURE_ATLAS_CPU_TRANSACTION_BUDGET_BYTES =
  512 * 1024 * 1024;
const MAX_FINITE_F32 = 3.4028234663852886e38;
const MIN_NORMAL_F32 = 2 ** -126;
// Shader pow/codec arithmetic and GPU-generated mip reductions are f32 and may
// differ from JavaScript's binary64 transcendental result by a small number of
// representable values. Keeping 1/64 of the top range unused is deliberately
// wider than atlas codec quantization and prevents an accepted CPU product from
// crossing into infinity on a conforming shader implementation.
const RADIOMETRIC_F32_MAX_WITH_DECODE_HEADROOM =
  MAX_FINITE_F32 * (1 - 1 / 64);
const FLOAT32_BYTES = Float32Array.BYTES_PER_ELEMENT;
const MAX_SAFE_INTEGER_BIGINT = BigInt(Number.MAX_SAFE_INTEGER);
const MATERIAL_TEXTURE_ATLAS_CPU_ALLOCATION_BUDGET_BIGINT =
  BigInt(MATERIAL_TEXTURE_ATLAS_CPU_ALLOCATION_BUDGET_BYTES);
const MATERIAL_TEXTURE_ATLAS_CPU_AGGREGATE_BUDGET_BIGINT =
  BigInt(MATERIAL_TEXTURE_ATLAS_CPU_TRANSACTION_BUDGET_BYTES);

interface MaterialAtlasCpuBudget {
  usedBytes: bigint;
  aggregateLimitBytes: bigint;
}

interface RadiometricPositiveSupport {
  /** Three low bits identify source-positive R/G/B lanes per texel. */
  readonly mask: Uint8Array;
  readonly reservedBytes: number;
}

type RadiometricAtlasUse =
  | {
      readonly kind: 'emissive';
      readonly materialIndex: number;
      readonly scalar: readonly [number, number, number];
    }
  | {
      readonly kind: 'light-map';
      readonly materialIndex: number;
      readonly intensity: number;
    };

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
  if (aggregateBytes > budget.aggregateLimitBytes) {
    throw new RangeError(
      `packMaterialTextureAtlas: ${label} requires ${byteCount.toString()} CPU bytes ` +
      `(aggregate ${aggregateBytes.toString()}), above the ` +
      `${budget.aggregateLimitBytes.toString()}-byte aggregate ` +
      'staging budget.',
    );
  }
  budget.usedBytes = aggregateBytes;
  return Number(elementCount);
}

function reserveByteAllocation(
  budget: MaterialAtlasCpuBudget,
  label: string,
  factors: readonly (number | bigint)[],
): number {
  const byteCount = checkedAllocationProduct(label, factors);
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
  if (aggregateBytes > budget.aggregateLimitBytes) {
    throw new RangeError(
      `packMaterialTextureAtlas: ${label} requires ${byteCount.toString()} CPU bytes ` +
      `(aggregate ${aggregateBytes.toString()}), above the ` +
      `${budget.aggregateLimitBytes.toString()}-byte aggregate ` +
      'staging budget.',
    );
  }
  budget.usedBytes = aggregateBytes;
  return Number(byteCount);
}

function releaseStagingAllocation(
  budget: MaterialAtlasCpuBudget,
  byteLength: number,
): void {
  const bytes = BigInt(byteLength);
  if (bytes > budget.usedBytes) {
    throw new Error('packMaterialTextureAtlas: internal staging-budget underflow.');
  }
  budget.usedBytes -= bytes;
}

function buildRadiometricPositiveSupport(
  rgba: Float32Array,
  budget: MaterialAtlasCpuBudget,
  label: string,
): RadiometricPositiveSupport {
  if (rgba.length % 4 !== 0) {
    throw new Error(
      `packMaterialTextureAtlas: ${label} requires complete RGBA texels.`,
    );
  }
  const pixelCount = rgba.length / 4;
  const reservedBytes = reserveByteAllocation(
    budget,
    `${label} positive-support mask`,
    [pixelCount],
  );
  const mask = new Uint8Array(pixelCount);
  for (let pixel = 0; pixel < pixelCount; pixel += 1) {
    const base = pixel * 4;
    mask[pixel] =
      (rgba[base]! > 0 ? 1 : 0) |
      (rgba[base + 1]! > 0 ? 2 : 0) |
      (rgba[base + 2]! > 0 ? 4 : 0);
  }
  return { mask, reservedBytes };
}

function pushRadiometricUse(
  usesByLayer: Map<number, RadiometricAtlasUse[]>,
  layer: number,
  use: RadiometricAtlasUse,
): void {
  const uses = usesByLayer.get(layer);
  if (uses == null) {
    usesByLayer.set(layer, [use]);
  } else {
    uses.push(use);
  }
}

/**
 * Validate a prospective standalone Float32Array without reserving it in the
 * aggregate staging ledger. This is used for the metadata lower-bound
 * preflight that must run before any material property is traversed; the exact
 * allocation (including active high-UV lanes) is reserved after collection.
 */
function preflightFloat32Allocation(
  label: string,
  factors: readonly (number | bigint)[],
): void {
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
    | 'invalid-material-texture-payload'
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
  readonly expectedValueCount?: number;
  readonly reason?: string;
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
  /**
   * Logical material maps in stable metadata-layer order. CPU sources retain
   * native, non-square dimensions in a fidelity-matched packed codec; the
   * device-aware uploader rectangle-packs these maps and their required mips.
   */
  readonly atlasLayers: readonly MaterialTextureAtlasLayer[];
  /** Compatibility projection of the GPU-backed subset of {@link atlasLayers}. */
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
  readonly encoding: MaterialTextureAtlasEncoding;
  readonly mipLevelCount: number;
  /**
   * True when atlas codec words contain encoded-sRGB RGB values. Native
   * *-srgb sources are re-encoded by the conversion pass after textureLoad so
   * their original 8-bit transfer-function precision is not collapsed into
   * linear 8-bit storage.
   */
  readonly decodeSrgb: boolean;
}

export interface MaterialTextureAtlasCpuSourceLayer {
  readonly kind: 'cpu';
  readonly layer: number;
  readonly width: number;
  readonly height: number;
  readonly encoding: MaterialTextureAtlasEncoding;
  readonly mipLevelCount: number;
  readonly decodeSrgb: boolean;
  /** Codec planes are contiguous: plane 0, then plane 1, etc. */
  readonly data: Uint32Array;
}

export interface MaterialTextureAtlasGpuBackedLayer {
  readonly kind: 'gpu';
  readonly layer: number;
  readonly width: number;
  readonly height: number;
  readonly encoding: MaterialTextureAtlasEncoding;
  readonly mipLevelCount: number;
  readonly decodeSrgb: boolean;
  readonly source: WalkaroundWebGpuTextureSource;
}

export type MaterialTextureAtlasLayer =
  | MaterialTextureAtlasCpuSourceLayer
  | MaterialTextureAtlasGpuBackedLayer;

/**
 * Canonical logical atlas bytes for scene/state fingerprints.
 *
 * Device-specific rectangle placements deliberately do not participate: they
 * are a deterministic upload detail derived from these logical dimensions,
 * codecs, mip requirements, source identities, and texels. Returning views
 * avoids rebuilding a monolithic compatibility atlas solely for hashing.
 */
export function materialTextureAtlasFingerprintParts(
  atlas: MaterialTextureAtlasPayload,
): readonly ArrayBufferView[] {
  const descriptors = new Uint32Array(1 + atlas.atlasLayers.length * 10);
  descriptors[0] = atlas.atlasLayers.length;
  atlas.atlasLayers.forEach((layer, index) => {
    const base = 1 + index * 10;
    descriptors[base] = layer.kind === 'gpu' ? 1 : 0;
    descriptors[base + 1] = layer.layer;
    descriptors[base + 2] = layer.width;
    descriptors[base + 3] = layer.height;
    descriptors[base + 4] = layer.encoding;
    descriptors[base + 5] = layer.mipLevelCount;
    descriptors[base + 6] = layer.decodeSrgb ? 1 : 0;
    if (layer.kind === 'gpu') {
      // Unversioned sources carry a session-salted descriptor identity, so
      // persisted GI fails closed across reloads. A host-supplied
      // contentRevision produces stable words instead.
      descriptors[base + 7] = layer.source.compatibilityKeyLo;
      descriptors[base + 8] = layer.source.compatibilityKeyHi;
      descriptors[base + 9] =
        ((layer.source.baseMipLevel & 0xffff) |
          ((layer.source.arrayLayer & 0xffff) << 16)) >>> 0;
    }
  });
  return [
    descriptors,
    ...atlas.atlasLayers.flatMap((layer) =>
      layer.kind === 'cpu' ? [layer.data] : []),
  ];
}

interface RawPixels {
  readonly width: number;
  readonly height: number;
  readonly data: Float32Array;
  readonly encoding: MaterialTextureAtlasEncoding;
  readonly channelCount: 1 | 2 | 3 | 4;
  readonly sourceColorSpace?: 'srgb' | 'linear';
}

interface AtlasLayerRecord {
  readonly layer: number;
  readonly width: number;
  readonly height: number;
  readonly source:
    | {
        readonly kind: 'cpu';
        readonly data: Uint32Array;
        readonly encoding: MaterialTextureAtlasEncoding;
        readonly decodeSrgb: boolean;
      }
    | {
        readonly kind: 'gpu';
        readonly descriptor: WalkaroundWebGpuTextureSource;
        readonly encoding: MaterialTextureAtlasEncoding;
        readonly decodeSrgb: boolean;
      };
}

interface OrderedAtlasLayer {
  readonly handle: unknown;
  readonly colorSpace: AtlasColorSpace;
  readonly record: Omit<AtlasLayerRecord, 'layer'>;
}

interface ValidReadHandlePixelsResult {
  readonly pixels: RawPixels;
  readonly ambiguousStride?: {
    readonly pixelStride: number;
    readonly valueCount: number;
    readonly width: number;
    readonly height: number;
  };
}

interface InvalidReadHandlePixelsResult {
  readonly invalidPayload: {
    readonly pixelStride?: number;
    readonly valueCount: number;
    readonly expectedValueCount?: number;
    readonly width: number;
    readonly height: number;
    readonly reason: string;
  };
}

type ReadHandlePixelsResult =
  | ValidReadHandlePixelsResult
  | InvalidReadHandlePixelsResult;

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

function asTextureRef(value: unknown): TextureRef | null {
  if (value == null || typeof value !== 'object') return null;
  if ('handle' in value) return value;
  return { handle: value };
}

function isFiniteNonUnderflowingFloat32(value: number | undefined): value is number {
  if (!Number.isFinite(value)) return false;
  const packed = Math.fround(value as number);
  return Number.isFinite(packed) && ((value as number) === 0 || packed !== 0);
}

function finiteOrFallback(value: number | undefined, fallback: number): number {
  return isFiniteNonUnderflowingFloat32(value) ? Math.fround(value) : Math.fround(fallback);
}

function invalidTextureTransformComponents(ref: TextureRef): string[] {
  const transform = ref.transform;
  if (transform == null) return [];
  const invalid: string[] = [];
  if (transform.offset != null) {
    if (!isFiniteNonUnderflowingFloat32(transform.offset[0])) invalid.push('offset.x');
    if (!isFiniteNonUnderflowingFloat32(transform.offset[1])) invalid.push('offset.y');
  }
  if (transform.scale != null) {
    if (!isFiniteNonUnderflowingFloat32(transform.scale[0])) invalid.push('scale.x');
    if (!isFiniteNonUnderflowingFloat32(transform.scale[1])) invalid.push('scale.y');
  }
  if (
    transform.rotation !== undefined &&
    !isFiniteNonUnderflowingFloat32(transform.rotation)
  ) {
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

function describeTextureHintValue(value: unknown): string {
  if (value === null) return 'null';
  switch (typeof value) {
    case 'string':
      return JSON.stringify(value);
    case 'number':
      return String(value);
    case 'boolean':
      return value ? 'true' : 'false';
    case 'undefined':
      return 'undefined';
    case 'bigint':
      return `${value.toString()}n`;
    case 'symbol':
      return value.description == null
        ? 'Symbol()'
        : `Symbol(${value.description})`;
    case 'function':
      return '[function]';
    case 'object':
      try {
        const serialized = JSON.stringify(value);
        if (serialized !== undefined) return serialized;
      } catch {
        // Cyclic objects still receive a deterministic, non-coercive label.
      }
      return Object.prototype.toString.call(value);
  }
  return 'unknown';
}

function readHandlePixels(
  handle: unknown,
  budget: MaterialAtlasCpuBudget,
  label: string,
  beforeDecode?: (
    width: number,
    height: number,
    channelCount: 1 | 2 | 3 | 4,
  ) => void,
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
  // Snapshot every user-controlled descriptor property once. Validation,
  // role preflight, allocation, and decode below consume only these locals.
  const image = h.image;
  const directData = h.data;
  const imageData = image?.data;
  const directWidth = h.width;
  const imageWidth = image?.width;
  const directHeight = h.height;
  const imageHeight = image?.height;
  const explicitHint = h.__vitrum_hint__;
  const looseChannels = h.channels;
  const looseDataType = h.dataType;
  const looseColorSpace = h.colorSpace;
  const src = directData ?? imageData;
  const width = Number(directWidth ?? imageWidth ?? 0);
  const height = Number(directHeight ?? imageHeight ?? 0);
  if (src == null) return null;
  const sourceLength = src.length;
  if (typeof sourceLength !== 'number') return null;
  assertPositiveAtlasDimension(width, `${label}.width`);
  assertPositiveAtlasDimension(height, `${label}.height`);
  if (!Number.isSafeInteger(sourceLength) || sourceLength < 0) {
    throw new RangeError(
      `packMaterialTextureAtlas: ${label}.data.length must be a non-negative ` +
      `safe integer; received ${String(sourceLength)}.`,
    );
  }
  const pixelCount = width * height;
  if (!Number.isSafeInteger(pixelCount)) {
    throw new RangeError(
      `packMaterialTextureAtlas: ${label} pixel count exceeds the safe integer range.`,
    );
  }

  const rawHint: unknown = explicitHint ?? (
    (looseChannels != null || looseDataType != null || looseColorSpace != null)
      ? {
          ...(looseChannels != null ? { channels: looseChannels } : {}),
          ...(looseDataType != null ? { dataType: looseDataType } : {}),
          ...(looseColorSpace != null ? { colorSpace: looseColorSpace } : {}),
        }
      : undefined
  );
  const invalidPayload = (
    reason: string,
    extra: {
      readonly pixelStride?: number;
      readonly expectedValueCount?: number;
    } = {},
  ): InvalidReadHandlePixelsResult => ({
    invalidPayload: {
      ...extra,
      valueCount: sourceLength,
      width,
      height,
      reason,
    },
  });
  if (
    rawHint != null &&
    (typeof rawHint !== 'object' || Array.isArray(rawHint))
  ) {
    return invalidPayload('__vitrum_hint__ must be an object');
  }
  const hintRecord = rawHint as {
    readonly channels?: unknown;
    readonly dataType?: unknown;
    readonly colorSpace?: unknown;
  } | undefined;

  const hintChannels = hintRecord?.channels;
  const hintDataType = hintRecord?.dataType;
  const hintColorSpace = hintRecord?.colorSpace;
  const hintedStride = Number(hintChannels);
  if (
    hintChannels !== undefined &&
    (!Number.isSafeInteger(hintedStride) || hintedStride < 1 || hintedStride > 4)
  ) {
    return invalidPayload(
      `declared channel count ${describeTextureHintValue(hintChannels)} is outside 1..4`,
    );
  }
  const validDataTypes = new Set([
    'uint8',
    'uint16',
    'float16',
    'half-float',
    'float32',
  ]);
  if (
    hintDataType !== undefined &&
    (
      typeof hintDataType !== 'string' ||
      !validDataTypes.has(hintDataType)
    )
  ) {
    return invalidPayload(
      `declared dataType ${describeTextureHintValue(hintDataType)} is unsupported`,
    );
  }
  if (
    hintColorSpace !== undefined &&
    hintColorSpace !== 'srgb' &&
    hintColorSpace !== 'linear'
  ) {
    return invalidPayload(
      `declared colorSpace ${describeTextureHintValue(hintColorSpace)} must be "srgb" or "linear"`,
    );
  }
  const hint: TextureHandleHint | undefined = hintRecord == null
    ? undefined
    : {
        ...(hintChannels !== undefined
          ? { channels: hintedStride as 1 | 2 | 3 | 4 }
          : {}),
        ...(hintDataType !== undefined
          ? { dataType: hintDataType as NonNullable<TextureHandleHint['dataType']> }
          : {}),
        ...(hintColorSpace !== undefined
          ? { colorSpace: hintColorSpace }
          : {}),
      };
  const inferredStride = sourceLength / pixelCount;
  const stride = hint?.channels ?? inferredStride;
  if (!Number.isSafeInteger(stride) || stride < 1 || stride > 4) {
    return {
      invalidPayload: {
        valueCount: sourceLength,
        width,
        height,
        reason:
          `value count ${sourceLength} does not resolve to an exact supported ` +
          `1..4 channel stride for ${width}x${height} pixels`,
      },
    };
  }
  const expectedValueCount = pixelCount * stride;
  if (!Number.isSafeInteger(expectedValueCount) || sourceLength !== expectedValueCount) {
    return invalidPayload(
      `declared ${stride}-channel payload has ${sourceLength} values; ` +
      `expected exactly ${expectedValueCount}`,
      { pixelStride: stride, expectedValueCount },
    );
  }
  const ambiguousStride = hint == null && stride !== 1 && stride !== 4
    ? { pixelStride: stride, valueCount: sourceLength, width, height }
    : undefined;

  // Source-role requirements must fail before the RGBA staging allocation or
  // any indexed traversal of an adversarial array-like payload.
  beforeDecode?.(width, height, stride as 1 | 2 | 3 | 4);

  const isFloat = src instanceof Float32Array;
  const useHalf = hint?.dataType != null
    ? hint.dataType === 'float16' || hint.dataType === 'half-float'
    : false;
  const useUint16 = hint?.dataType != null ? hint.dataType === 'uint16' : src instanceof Uint16Array;
  const useFloat = hint?.dataType != null ? hint.dataType === 'float32' : isFloat;
  const resolvedDataType: TextureHandleHint['dataType'] = useHalf
    ? 'float16'
    : useFloat
      ? 'float32'
      : useUint16
        ? 'uint16'
        : 'uint8';
  // Reject impossible decode allocations before traversing an adversarial
  // array-like payload or invoking any of its indexed getters.
  preflightFloat32Allocation(
    `${label} RGBA decode`,
    [width, height, 4],
  );
  const integerMax = resolvedDataType === 'uint8'
    ? 255
    : (
        resolvedDataType === 'uint16' ||
        resolvedDataType === 'float16'
      )
      ? 65535
      : null;
  const outputElementCount = reserveFloat32Allocation(
    budget,
    `${label} RGBA decode`,
    [width, height, 4],
  );
  const out = new Float32Array(outputElementCount);
  for (let p = 0; p < pixelCount; p += 1) {
    const s = p * stride;
    let r = 0;
    let g = 0;
    let b = 0;
    let a = 1;
    for (let lane = 0; lane < stride; lane += 1) {
      const index = s + lane;
      // Read each user-controlled lane exactly once. Validation, f32
      // canonicalization, and publication all consume this same value.
      const value = Number(src[index]);
      if (!Number.isFinite(value)) {
        releaseStagingAllocation(budget, out.byteLength);
        return invalidPayload(
          `payload value ${index} must be finite for ${resolvedDataType}`,
          { pixelStride: stride, expectedValueCount },
        );
      }
      if (
        integerMax != null &&
        (!Number.isInteger(value) || value < 0 || value > integerMax)
      ) {
        releaseStagingAllocation(budget, out.byteLength);
        return invalidPayload(
          `payload value ${index} must be an integer in [0, ${integerMax}] for ${resolvedDataType}`,
          { pixelStride: stride, expectedValueCount },
        );
      }

      let decoded = value;
      if (resolvedDataType === 'float32') {
        decoded = Math.fround(value);
        if (!Number.isFinite(decoded)) {
          releaseStagingAllocation(budget, out.byteLength);
          return invalidPayload(
            `payload value ${index} is outside the finite float32 range`,
            { pixelStride: stride, expectedValueCount },
          );
        }
        if (value !== 0 && decoded === 0) {
          releaseStagingAllocation(budget, out.byteLength);
          return invalidPayload(
            `payload value ${index} is nonzero but underflows to zero in float32`,
            { pixelStride: stride, expectedValueCount },
          );
        }
      } else if (useHalf) {
        decoded = halfToFloat(value);
      } else if (useUint16) {
        decoded = value / 65535;
      } else {
        decoded = value / 255;
      }
      if (!Number.isFinite(decoded)) {
        releaseStagingAllocation(budget, out.byteLength);
        return invalidPayload(
          `decoded pixel ${p} contains a non-finite component`,
          { pixelStride: stride, expectedValueCount },
        );
      }

      if (lane === 0) r = decoded;
      else if (lane === 1) g = decoded;
      else if (lane === 2) b = decoded;
      else a = decoded;
    }
    if (stride === 1) {
      g = r;
      b = r;
    }
    // Public raw-channel convention shared by every backend:
    //   R → [R,R,R,1], RG → [R,G,0,1], RGB → [R,G,B,1].
    // Four-channel payloads retain authored RGBA.
    out[p * 4] = r;
    out[p * 4 + 1] = g;
    out[p * 4 + 2] = b;
    out[p * 4 + 3] = a;
  }

  const sourceColorSpace =
    hint?.colorSpace === 'srgb' || hint?.colorSpace === 'linear'
      ? hint.colorSpace
      : resolvedDataType === 'float32'
        ? 'linear'
        : undefined;
  return {
    pixels: {
      width,
      height,
      data: out,
      encoding: materialTextureAtlasEncodingForDataType(resolvedDataType),
      channelCount: stride as 1 | 2 | 3 | 4,
      ...(sourceColorSpace ? { sourceColorSpace } : {}),
    },
    ...(ambiguousStride ? { ambiguousStride } : {}),
  };
}

function packGpuCpuMirrorForRadiometricAtlas(
  layer: Pick<
    MaterialTextureAtlasGpuBackedLayer,
    'layer' | 'width' | 'height' | 'encoding' | 'source'
  >,
  budget: MaterialAtlasCpuBudget,
): {
  readonly data: Uint32Array;
  readonly positiveSupport: RadiometricPositiveSupport;
} {
  const mirror = layer.source.cpuMirror;
  if (mirror == null) {
    throw new TypeError(
      `packMaterialTextureAtlas: GPU atlas layer ${layer.layer} has no CPU mirror.`,
    );
  }
  const label = `GPU atlas layer ${layer.layer} CPU mirror`;
  const read = readHandlePixels(
    {
      width: mirror.width,
      height: mirror.height,
      data: mirror.data,
      channels: mirror.channels,
      dataType: mirror.dataType,
      colorSpace: mirror.colorSpace,
    },
    budget,
    label,
  );
  if (read == null || 'invalidPayload' in read) {
    const reason = read != null && 'invalidPayload' in read
      ? `: ${read.invalidPayload.reason}`
      : '';
    throw new TypeError(
      `packMaterialTextureAtlas: ${label} is invalid${reason}.`,
    );
  }
  const pixels = read.pixels;
  if (pixels.width !== layer.width || pixels.height !== layer.height) {
    releaseStagingAllocation(budget, pixels.data.byteLength);
    throw new RangeError(
      `packMaterialTextureAtlas: ${label} dimensions do not match the selected GPU subresource.`,
    );
  }

  // `readHandlePixels` applies the public raw-channel convention. Radiometric
  // GPU sources publish this immutable snapshot as their canonical atlas
  // layer, so classification, preflight, mip generation, and shader sampling
  // all consume these exact packed bytes. A separately converted GPU copy can
  // differ at native-sRGB decode/re-encode and f32 mip-rounding boundaries and
  // therefore cannot be used after the mirror has been accepted as exact.
  for (let pixel = 0; pixel < pixels.width * pixels.height; pixel += 1) {
    const base = pixel * 4;
    if (mirror.channels < 4) {
      pixels.data[base + 3] = 1;
    }
  }

  const positiveSupport = buildRadiometricPositiveSupport(
    pixels.data,
    budget,
    `${label}`,
  );
  reserveFloat32Allocation(
    budget,
    `${label} packed codec planes`,
    [
      layer.width,
      layer.height,
      materialTextureAtlasEncodingPlaneCount(layer.encoding),
    ],
  );
  try {
    const data = packMaterialTextureAtlasPixels(pixels.data, layer.encoding);
    return {
      data,
      positiveSupport,
    };
  } finally {
    releaseStagingAllocation(budget, pixels.data.byteLength);
  }
}

function generateRadiometricPositiveSupportMip(
  source: RadiometricPositiveSupport,
  width: number,
  height: number,
  budget: MaterialAtlasCpuBudget,
  label: string,
): {
  readonly width: number;
  readonly height: number;
  readonly support: RadiometricPositiveSupport;
} {
  if (source.mask.length !== width * height) {
    throw new Error(
      `packMaterialTextureAtlas: ${label} positive-support size drifted.`,
    );
  }
  const targetWidth = Math.max(1, Math.floor(width / 2));
  const targetHeight = Math.max(1, Math.floor(height / 2));
  const reservedBytes = reserveByteAllocation(
    budget,
    `${label} positive-support mip`,
    [targetWidth, targetHeight],
  );
  const mask = new Uint8Array(targetWidth * targetHeight);
  for (let y = 0; y < targetHeight; y += 1) {
    for (let x = 0; x < targetWidth; x += 1) {
      const sourceX0 = x * width / targetWidth;
      const sourceX1 = (x + 1) * width / targetWidth;
      const sourceY0 = y * height / targetHeight;
      const sourceY1 = (y + 1) * height / targetHeight;
      const firstX = Math.floor(sourceX0);
      const lastX = Math.ceil(sourceX1) - 1;
      const firstY = Math.floor(sourceY0);
      const lastY = Math.ceil(sourceY1) - 1;
      let support = 0;
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
          if (wx * wy > 0) support |= source.mask[sy * width + sx]!;
        }
      }
      mask[y * targetWidth + x] = support;
    }
  }
  return {
    width: targetWidth,
    height: targetHeight,
    support: { mask, reservedBytes },
  };
}

interface RadiometricAtlasValidationCodec {
  readonly unpack: typeof unpackMaterialTextureAtlasPixels;
  readonly generateMip: typeof generateMaterialTextureAtlasMip;
}

const DEFAULT_RADIOMETRIC_ATLAS_VALIDATION_CODEC: RadiometricAtlasValidationCodec = {
  unpack: unpackMaterialTextureAtlasPixels,
  generateMip: generateMaterialTextureAtlasMip,
};

function generateRadiometricAtlasMipWithBudget(
  layer: MaterialTextureAtlasLayer,
  packed: Uint32Array,
  width: number,
  height: number,
  mipLevel: number,
  budget: MaterialAtlasCpuBudget,
  codec: RadiometricAtlasValidationCodec,
): {
  readonly width: number;
  readonly height: number;
  readonly data: Uint32Array;
  readonly packedReservedBytes: number;
} {
  const targetWidth = Math.max(1, Math.floor(width / 2));
  const targetHeight = Math.max(1, Math.floor(height / 2));
  const label = `radiometric layer ${layer.layer} mip ${mipLevel + 1}`;
  // The codec retains its target RGBA staging array until the packed target is
  // complete. Reserve both arrays before entering it; keep only the returned
  // packed reservation after the call.
  const targetRgbaElements = reserveFloat32Allocation(
    budget,
    `${label} generated RGBA staging`,
    [targetWidth, targetHeight, 4],
  );
  const targetRgbaReservedBytes = targetRgbaElements * FLOAT32_BYTES;
  let packedReservedBytes = 0;
  try {
    const packedWords = reserveFloat32Allocation(
      budget,
      `${label} generated packed codec planes`,
      [
        targetWidth,
        targetHeight,
        materialTextureAtlasEncodingPlaneCount(layer.encoding),
      ],
    );
    packedReservedBytes = packedWords * FLOAT32_BYTES;
    let next: ReturnType<typeof generateMaterialTextureAtlasMip>;
    try {
      next = codec.generateMip(
        packed,
        width,
        height,
        layer.encoding,
        layer.decodeSrgb,
      );
    } catch (error) {
      releaseStagingAllocation(budget, packedReservedBytes);
      packedReservedBytes = 0;
      throw error;
    }
    if (
      next.width !== targetWidth ||
      next.height !== targetHeight ||
      next.data.length !== packedWords
    ) {
      releaseStagingAllocation(budget, packedReservedBytes);
      packedReservedBytes = 0;
      throw new Error(
        `packMaterialTextureAtlas: ${label} generated an unexpected codec payload.`,
      );
    }
    return { ...next, packedReservedBytes };
  } finally {
    releaseStagingAllocation(budget, targetRgbaReservedBytes);
  }
}

function validateRadiometricAtlasLayer(
  layer: MaterialTextureAtlasLayer,
  baseLevel: Uint32Array,
  uses: readonly RadiometricAtlasUse[],
  basePositiveSupport: RadiometricPositiveSupport,
  budget: MaterialAtlasCpuBudget,
  codec: RadiometricAtlasValidationCodec =
    DEFAULT_RADIOMETRIC_ATLAS_VALIDATION_CODEC,
): void {
  let width = layer.width;
  let height = layer.height;
  let packed = baseLevel;
  let positiveSupport = basePositiveSupport;
  // The base packed level was reserved when the atlas layer was collected.
  // Generated levels are transient and keep their own reservation only while
  // they are the current validation source.
  let generatedPackedReservedBytes = 0;
  try {
    for (let mipLevel = 0; mipLevel < layer.mipLevelCount; mipLevel += 1) {
      const decodedElements = reserveFloat32Allocation(
        budget,
        `radiometric layer ${layer.layer} mip ${mipLevel} decoded RGBA staging`,
        [width, height, 4],
      );
      const decodedReservedBytes = decodedElements * FLOAT32_BYTES;
      try {
        const decoded = codec.unpack(packed, layer.encoding);
        const pixelCount = width * height;
        if (
          decoded.length !== pixelCount * 4 ||
          positiveSupport.mask.length !== pixelCount
        ) {
          throw new Error(
            `packMaterialTextureAtlas: radiometric layer ${layer.layer} mip ${mipLevel} ` +
            'decoded to an unexpected texel count.',
          );
        }
        for (let pixel = 0; pixel < pixelCount; pixel += 1) {
          const base = pixel * 4;
          const storedRgb = [
            decoded[base]!,
            decoded[base + 1]!,
            decoded[base + 2]!,
          ] as const;
          for (let channel = 0; channel < 3; channel += 1) {
            const value = storedRgb[channel]!;
            if (!Number.isFinite(value) || value < 0) {
              throw new RangeError(
                `packMaterialTextureAtlas: radiometric layer ${layer.layer} mip ` +
                `${mipLevel} texel ${pixel} channel ${channel} must decode to finite ` +
                `non-negative radiance; received ${String(value)}.`,
              );
            }
          }
          const linearRgb = storedRgb.map((value) =>
            layer.decodeSrgb
              ? Math.fround(srgbToLinear(value))
              : Math.fround(value)
          ) as [number, number, number];
          if (linearRgb.some((value) => !Number.isFinite(value))) {
            throw new RangeError(
              `packMaterialTextureAtlas: radiometric layer ${layer.layer} mip ` +
              `${mipLevel} texel ${pixel} is non-finite after color-space decoding.`,
            );
          }

          for (const use of uses) {
            const label =
              `material ${use.materialIndex} ${use.kind} layer ${layer.layer} ` +
              `mip ${mipLevel} texel ${pixel}`;
            const scalar = use.kind === 'emissive'
              ? use.scalar
              : [use.intensity, use.intensity, use.intensity] as const;
            const sourceSupport = positiveSupport.mask[pixel]!;
            const expectedPositive = scalar.some(
              (component, channel) =>
                component > 0 && (sourceSupport & (1 << channel)) !== 0,
            );
            const radiance = packRadianceRgbProductF32(
              scalar,
              linearRgb,
              `${label} radiance`,
            );
            const positiveProducts = linearRgb
              .map((component, channel) => component * scalar[channel]!)
              .filter((component) => component > 0);
            if (
              expectedPositive &&
              positiveProducts.length === 0
            ) {
              throw new RangeError(
                `packMaterialTextureAtlas: positive ${label} radiance must not ` +
                'collapse completely through atlas codec or mip generation.',
              );
            }
            if (
              positiveProducts.some(
                (component) =>
                  component > RADIOMETRIC_F32_MAX_WITH_DECODE_HEADROOM,
              )
            ) {
              throw new RangeError(
                `packMaterialTextureAtlas: ${label} radiance must retain finite ` +
                'Float32 headroom across shader color-space decode and mip arithmetic.',
              );
            }
            if (
              expectedPositive &&
              positiveProducts.every((component) => component < MIN_NORMAL_F32)
            ) {
              throw new RangeError(
                `packMaterialTextureAtlas: positive ${label} radiance must retain ` +
                'normal Float32 magnitude across shader decode and mip arithmetic.',
              );
            }
            if (
              expectedPositive &&
              radiance.every((component) => component === 0)
            ) {
              throw new RangeError(
                `packMaterialTextureAtlas: positive ${label} radiance must not ` +
                'collapse completely to zero after atlas decoding and Float32 multiplication.',
              );
            }
          }
        }

        if (mipLevel + 1 < layer.mipLevelCount) {
          const next = generateRadiometricAtlasMipWithBudget(
            layer,
            packed,
            width,
            height,
            mipLevel,
            budget,
            codec,
          );
          let nextPositiveSupport: ReturnType<
            typeof generateRadiometricPositiveSupportMip
          >;
          try {
            nextPositiveSupport = generateRadiometricPositiveSupportMip(
              positiveSupport,
              width,
              height,
              budget,
              `radiometric layer ${layer.layer} mip ${mipLevel + 1}`,
            );
          } catch (error) {
            releaseStagingAllocation(budget, next.packedReservedBytes);
            throw error;
          }
          const previousPositiveSupportBytes = positiveSupport.reservedBytes;
          const previousGeneratedPackedBytes = generatedPackedReservedBytes;
          positiveSupport = nextPositiveSupport.support;
          generatedPackedReservedBytes = next.packedReservedBytes;
          width = next.width;
          height = next.height;
          packed = next.data;
          releaseStagingAllocation(budget, previousPositiveSupportBytes);
          if (previousGeneratedPackedBytes > 0) {
            releaseStagingAllocation(budget, previousGeneratedPackedBytes);
          }
        }
      } finally {
        releaseStagingAllocation(budget, decodedReservedBytes);
      }
    }
  } finally {
    releaseStagingAllocation(budget, positiveSupport.reservedBytes);
    if (generatedPackedReservedBytes > 0) {
      releaseStagingAllocation(budget, generatedPackedReservedBytes);
    }
  }
}

/**
 * @internal Test seam for adversarial transaction-budget ordering. This is not
 * re-exported from the package: it drives the real validator with one logical
 * rgba32float texel while callbacks make codec entry observable.
 */
export function _probeRadiometricAtlasValidationBudgetForTest(options: {
  readonly usedBytes: number;
  readonly aggregateLimitBytes: number;
  readonly mipLevelCount: 1 | 2;
  readonly onDecode: () => void;
  readonly onGenerateMip: () => void;
}): void {
  if (
    !Number.isSafeInteger(options.usedBytes) ||
    options.usedBytes < 0 ||
    !Number.isSafeInteger(options.aggregateLimitBytes) ||
    options.aggregateLimitBytes < options.usedBytes
  ) {
    throw new RangeError('radiometric atlas budget probe requires valid byte limits.');
  }
  const encoding = materialTextureAtlasEncodingForDataType('float32');
  const data = new Uint32Array(4);
  validateRadiometricAtlasLayer(
    {
      kind: 'cpu',
      layer: 0,
      width: 1,
      height: 1,
      encoding,
      mipLevelCount: options.mipLevelCount,
      decodeSrgb: false,
      data,
    },
    data,
    [],
    { mask: new Uint8Array(1), reservedBytes: 0 },
    {
      usedBytes: BigInt(options.usedBytes),
      aggregateLimitBytes: BigInt(options.aggregateLimitBytes),
    },
    {
      unpack: () => {
        options.onDecode();
        return new Float32Array(4);
      },
      generateMip: () => {
        options.onGenerateMip();
        return { width: 1, height: 1, data: new Uint32Array(4) };
      },
    },
  );
}

function validateRadiometricAtlasLayers(
  layers: readonly MaterialTextureAtlasLayer[],
  usesByLayer: ReadonlyMap<number, readonly RadiometricAtlasUse[]>,
  cpuPositiveSupport: ReadonlyMap<number, RadiometricPositiveSupport>,
  budget: MaterialAtlasCpuBudget,
): void {
  for (const layer of layers) {
    const uses = usesByLayer.get(layer.layer);
    if (uses == null || uses.length === 0) continue;
    if (layer.kind === 'cpu') {
      const positiveSupport = cpuPositiveSupport.get(layer.layer);
      if (positiveSupport == null) {
        throw new Error(
          `packMaterialTextureAtlas: radiometric CPU layer ${layer.layer} ` +
          'lost its source positive-support mask.',
        );
      }
      validateRadiometricAtlasLayer(
        layer,
        layer.data,
        uses,
        positiveSupport,
        budget,
      );
      continue;
    }
    // Every radiometric GPU source is required to carry an exact CPU mirror and
    // is canonicalised into a CPU atlas layer during collection. Reaching a GPU
    // layer here means collection violated the pre-publication invariant.
    throw new Error(
      `packMaterialTextureAtlas: radiometric GPU layer ${layer.layer} ` +
      'reached validation without its required exact CPU mirror.',
    );
  }
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
  const mipFilter = MIP_FILTER_INDEX[ref.mipFilter ?? 'linear'];
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
  const clamped = Number.isFinite(value) ? Math.max(0, value ?? fallback) : fallback;
  return finiteOrFallback(clamped, fallback);
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
  const clamped = Number.isFinite(value) ? Math.max(0, value ?? fallback) : fallback;
  return finiteOrFallback(clamped, fallback);
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
  emissiveRadianceMaterials: readonly MaterialSpec[] = materials,
): MaterialTextureAtlasPayload {
  if (!Number.isSafeInteger(triCount) || triCount < 0) {
    throw new RangeError(
      'packMaterialTextureAtlas: triCount must be a non-negative safe integer; ' +
      `received ${String(triCount)}.`,
    );
  }
  if (emissiveRadianceMaterials.length !== materials.length) {
    throw new RangeError(
      'packMaterialTextureAtlas: emissiveRadianceMaterials must preserve the ' +
      `material-slot count (${materials.length}); received ` +
      `${emissiveRadianceMaterials.length}.`,
    );
  }
  const lightMapIntensities = materials.map((material, materialIndex) =>
    packNonNegativeRadianceScalarF32(
      material.lightMapIntensity ?? 1,
      `material ${materialIndex} lightMapIntensity`,
    )
  );
  const materialRecordCount = Math.max(1, materials.length);
  // The material records, packed triangle→material table, and ABI header are
  // mandatory regardless of which material properties/maps are present.
  // Reject impossible geometry dimensions before invoking even one material
  // getter; active high-UV lanes are added to the exact reservation later.
  const mandatoryMetaTexels =
    BigInt(materialRecordCount) * BigInt(MATERIAL_MAP_META_TEXELS_PER_TRI) +
    (BigInt(triCount) + 3n) / 4n +
    BigInt(MATERIAL_META_HEADER_TEXELS);
  const mandatoryMetaWidth = mandatoryMetaTexels < BigInt(BASE_COLOR_MAP_META_TEX_WIDTH)
    ? mandatoryMetaTexels
    : BigInt(BASE_COLOR_MAP_META_TEX_WIDTH);
  const mandatoryMetaHeight =
    (mandatoryMetaTexels + mandatoryMetaWidth - 1n) / mandatoryMetaWidth;
  preflightFloat32Allocation(
    'material metadata atlas',
    [mandatoryMetaWidth, mandatoryMetaHeight, 4n],
  );

  const allocationBudget: MaterialAtlasCpuBudget = {
    usedBytes: 0n,
    aggregateLimitBytes: MATERIAL_TEXTURE_ATLAS_CPU_AGGREGATE_BUDGET_BIGINT,
  };
  const opticalMetaElementCount = MATERIAL_OPTICAL_META_TEXELS * 4;
  reserveFloat32Allocation(
    allocationBudget,
    'material optical metadata',
    [BigInt(materials.length) + 1n, opticalMetaElementCount],
  );
  const readable = new Map<unknown, Partial<Record<AtlasColorSpace, AtlasLayerRecord>>>();
  const sourceChannelCounts = new Map<unknown, 1 | 2 | 3 | 4>();
  const ordered: OrderedAtlasLayer[] = [];
  const mipmappedLayers = new Set<number>();
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

  // Know which logical handle/color-space pairs will carry radiance before
  // the first deduplicated atlas layer is collected. A map may first appear in
  // a non-radiometric field and only later be reused as emissive/light data;
  // pre-marking it lets the single authoritative decode retain a compact
  // positive-support mask before its Float32 staging array is released.
  const radiometricHandleColorSpaces =
    new Map<unknown, Set<AtlasColorSpace>>();
  const markRadiometricHandle = (
    ref: TextureRef | null,
    colorSpace: AtlasColorSpace,
  ): void => {
    if (ref?.handle == null) return;
    const spaces = radiometricHandleColorSpaces.get(ref.handle);
    if (spaces == null) {
      radiometricHandleColorSpaces.set(ref.handle, new Set([colorSpace]));
    } else {
      spaces.add(colorSpace);
    }
  };
  materials.forEach((material) => {
    markRadiometricHandle(
      materialTextureRefForField(material, 'emissiveMap'),
      'srgb',
    );
    markRadiometricHandle(
      materialTextureRefForField(material, 'lightMap'),
      'linear',
    );
  });
  const cpuRadiometricPositiveSupport =
    new Map<number, RadiometricPositiveSupport>();

  const minimumAuthoredChannelCount = (field: AtlasMapField): 1 | 3 | 4 => {
    if (
      field === 'normalMap' ||
      field === 'clearcoatNormalMap' ||
      field === 'frontLayer.normalMap' ||
      field === 'backLayer.normalMap' ||
      field === 'anisotropyMap'
    ) {
      return 3;
    }
    if (
      field === 'specularIntensityMap' ||
      field === 'sheenRoughnessMap'
    ) {
      return 4;
    }
    return 1;
  };
  const assertMaterialMapChannels = (
    field: AtlasMapField,
    channelCount: 1 | 2 | 3 | 4,
    materialIndex: number,
  ): void => {
    if (field === 'metallicMap' && channelCount === 2) {
      throw new RangeError(
        `packMaterialTextureAtlas: material ${materialIndex} metallicMap uses a ` +
        '2-channel source. This role requires at least 3 authored channels ' +
        'containing B, unless exactly 1 channel is supplied (R is replicated ' +
        'to the consumed B lane).',
      );
    }
    const minimum = minimumAuthoredChannelCount(field);
    if (channelCount >= minimum) return;
    const requirement = minimum === 4
      ? 'the authored alpha channel consumed by this map role'
      : field === 'anisotropyMap'
        ? 'authored RG direction and B strength channels'
        : 'authored XYZ/RGB normal channels; one/two-channel Z reconstruction is not part of the material-atlas source contract';
    throw new RangeError(
      `packMaterialTextureAtlas: material ${materialIndex} ${field} uses a ` +
      `${channelCount}-channel source. This role requires at least ${minimum} ` +
      `channels for ${requirement}.`,
    );
  };

  const collect = (
    material: MaterialSpec,
    materialIndex: number,
    field: AtlasMapField,
    colorSpace: AtlasColorSpace,
  ): void => {
    const ref = materialTextureRefForField(material, field);
    if (ref?.handle == null) return;
    if (
      (field === 'emissiveMap' || field === 'lightMap') &&
      isWalkaroundWebGpuTextureSource(ref.handle) &&
      ref.handle.cpuMirror == null
    ) {
      const source = textureRefSourceMetadata(ref);
      throw new TypeError(
        `packMaterialTextureAtlas: material ${materialIndex} ${field} uses a ` +
        'GPU source without the exact cpuMirror required for emitter ' +
        'classification or atlas-radiance preflight' +
        `${source?.path !== undefined ? ` at ${source.path}` : ''}.`,
      );
    }
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
          `${field} texture transform contains component(s) outside the finite, ` +
          'non-underflowing float32 domain ' +
          `${transformComponents.join(', ')}` +
          `${source?.path !== undefined ? ` at ${source.path}` : ''}; ` +
          'invalid components are replaced with the identity transform fallback.',
      });
    };
    let perHandle = readable.get(ref.handle);
    const existing = perHandle?.[colorSpace];
    if (existing != null) {
      const channelCount = sourceChannelCounts.get(ref.handle);
      if (channelCount == null) {
        throw new Error(
          'packMaterialTextureAtlas: internal source-channel metadata drifted.',
        );
      }
      assertMaterialMapChannels(field, channelCount, materialIndex);
      pushInvalidTransformDiagnostic();
      fieldLayers[field].add(existing.layer);
      if (ref.mipFilter !== 'none') {
        mipmappedLayers.add(existing.layer);
      }
      return;
    }

    if (isWalkaroundWebGpuTextureSource(ref.handle)) {
      const channelCount =
        walkaroundTextureFormatChannelCount(ref.handle.format);
      assertMaterialMapChannels(field, channelCount, materialIndex);
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
      pushInvalidTransformDiagnostic();
      const layer = ordered.length;
      const encoding = materialTextureAtlasEncodingForGpuFormat(ref.handle.format);
      const decodeSrgb =
        colorSpace === 'srgb' && ref.handle.colorSpace === 'srgb';
      const radiometric =
        radiometricHandleColorSpaces.get(ref.handle)?.has(colorSpace) === true;
      if (radiometric && ref.handle.cpuMirror != null) {
        const mirror = packGpuCpuMirrorForRadiometricAtlas(
          {
            layer,
            width: ref.handle.width,
            height: ref.handle.height,
            encoding,
            source: ref.handle,
          },
          allocationBudget,
        );
        cpuRadiometricPositiveSupport.set(layer, mirror.positiveSupport);
        const record: AtlasLayerRecord = {
          layer,
          width: ref.handle.width,
          height: ref.handle.height,
          source: {
            kind: 'cpu',
            data: mirror.data,
            encoding,
            decodeSrgb,
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
        sourceChannelCounts.set(ref.handle, channelCount);
        fieldLayers[field].add(layer);
        if (ref.mipFilter !== 'none') {
          mipmappedLayers.add(layer);
        }
        return;
      }
      const record: AtlasLayerRecord = {
        layer,
        width: ref.handle.width,
        height: ref.handle.height,
        source: {
          kind: 'gpu',
          descriptor: ref.handle,
          encoding,
          decodeSrgb,
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
      sourceChannelCounts.set(ref.handle, channelCount);
      fieldLayers[field].add(layer);
      if (ref.mipFilter !== 'none') {
        mipmappedLayers.add(layer);
      }
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
      (_width, _height, channelCount) => {
        assertMaterialMapChannels(field, channelCount, materialIndex);
      },
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
    if ('invalidPayload' in read) {
      const source = textureRefSourceMetadata(ref);
      diagnostics.push({
        code: 'invalid-material-texture-payload',
        materialIndex,
        field,
        colorSpace,
        ...(read.invalidPayload.pixelStride !== undefined
          ? { pixelStride: read.invalidPayload.pixelStride }
          : {}),
        valueCount: read.invalidPayload.valueCount,
        ...(read.invalidPayload.expectedValueCount !== undefined
          ? { expectedValueCount: read.invalidPayload.expectedValueCount }
          : {}),
        width: read.invalidPayload.width,
        height: read.invalidPayload.height,
        reason: read.invalidPayload.reason,
        ...sourceMetaFields(source),
        message:
          `${field} texture payload is invalid: ${read.invalidPayload.reason}` +
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
    assertMaterialMapChannels(field, pixels.channelCount, materialIndex);
    if (colorSpace === 'linear' && pixels.sourceColorSpace === 'srgb') {
      const source = textureRefSourceMetadata(ref);
      throw new RangeError(
        `packMaterialTextureAtlas: ${field} is a linear-data map, but its CPU ` +
        `source declares srgb values` +
        `${source?.path !== undefined ? ` at ${source.path}` : ''}.`,
      );
    }
    const decodeSrgb =
      colorSpace === 'srgb' && pixels.sourceColorSpace !== 'linear';
    reserveFloat32Allocation(
      allocationBudget,
      `material ${materialIndex} ${field} packed codec planes`,
      [
        pixels.width,
        pixels.height,
        materialTextureAtlasEncodingPlaneCount(pixels.encoding),
      ],
    );
    const packedData = packMaterialTextureAtlasPixels(pixels.data, pixels.encoding);
    const layer = ordered.length;
    if (radiometricHandleColorSpaces.get(ref.handle)?.has(colorSpace) === true) {
      cpuRadiometricPositiveSupport.set(
        layer,
        buildRadiometricPositiveSupport(
          pixels.data,
          allocationBudget,
          `material atlas layer ${layer}`,
        ),
      );
    }
    releaseStagingAllocation(allocationBudget, pixels.data.byteLength);
    const record: AtlasLayerRecord = {
      layer,
      width: pixels.width,
      height: pixels.height,
      source: {
        kind: 'cpu',
        data: packedData,
        encoding: pixels.encoding,
        decodeSrgb,
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
    sourceChannelCounts.set(ref.handle, pixels.channelCount);
    fieldLayers[field].add(layer);
    if (ref.mipFilter !== 'none') {
      mipmappedLayers.add(layer);
    }
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

  const atlasLayers: MaterialTextureAtlasLayer[] = [];
  const gpuSourceLayers: MaterialTextureAtlasGpuSourceLayer[] = [];
  ordered.forEach((entry, layer) => {
    assertPositiveAtlasDimension(entry.record.width, 'atlas layer width');
    assertPositiveAtlasDimension(entry.record.height, 'atlas layer height');
    const mipLevelCount = mipmappedLayers.has(layer)
      ? Math.floor(Math.log2(Math.max(entry.record.width, entry.record.height))) + 1
      : 1;
    if (mipLevelCount > MATERIAL_ATLAS_MAX_MIP_LEVELS) {
      throw new RangeError(
        `packMaterialTextureAtlas: layer ${layer} requires ${mipLevelCount} mip levels, ` +
        `above the ${MATERIAL_ATLAS_MAX_MIP_LEVELS}-level packed-atlas ABI.`,
      );
    }
    if (entry.record.source.kind === 'gpu') {
      const source = entry.record.source.descriptor;
      const gpuLayer: MaterialTextureAtlasGpuBackedLayer = {
        kind: 'gpu',
        layer,
        width: entry.record.width,
        height: entry.record.height,
        encoding: entry.record.source.encoding,
        mipLevelCount,
        decodeSrgb: entry.record.source.decodeSrgb,
        source,
      };
      atlasLayers.push(gpuLayer);
      gpuSourceLayers.push({
        layer,
        source,
        encoding: gpuLayer.encoding,
        mipLevelCount,
        decodeSrgb: gpuLayer.decodeSrgb,
      });
      return;
    }
    const expectedPackedWords =
      entry.record.width *
      entry.record.height *
      materialTextureAtlasEncodingPlaneCount(entry.record.source.encoding);
    if (
      !Number.isSafeInteger(expectedPackedWords) ||
      entry.record.source.data.length !== expectedPackedWords
    ) {
      throw new Error('packMaterialTextureAtlas: internal packed-layer size drifted.');
    }
    atlasLayers.push({
      kind: 'cpu',
      layer,
      width: entry.record.width,
      height: entry.record.height,
      encoding: entry.record.source.encoding,
      mipLevelCount,
      decodeSrgb: entry.record.source.decodeSrgb,
      data: entry.record.source.data,
    });
  });

  const radiometricUsesByLayer = new Map<number, RadiometricAtlasUse[]>();
  materials.forEach((material, materialIndex) => {
    const emissiveRef = materialTextureRefForField(material, 'emissiveMap');
    const emissiveLayer = emissiveRef?.handle == null
      ? undefined
      : readable.get(emissiveRef.handle)?.srgb?.layer;
    if (emissiveLayer != null) {
      pushRadiometricUse(radiometricUsesByLayer, emissiveLayer, {
        kind: 'emissive',
        materialIndex,
        scalar:
          materialSpecScalarEmissiveLe(
            emissiveRadianceMaterials[materialIndex]!,
          ) ?? [0, 0, 0],
      });
    }

    const lightRef = materialTextureRefForField(material, 'lightMap');
    const lightLayer = lightRef?.handle == null
      ? undefined
      : readable.get(lightRef.handle)?.linear?.layer;
    if (lightLayer != null) {
      pushRadiometricUse(radiometricUsesByLayer, lightLayer, {
        kind: 'light-map',
        materialIndex,
        intensity: lightMapIntensities[materialIndex]!,
      });
    }
  });
  validateRadiometricAtlasLayers(
    atlasLayers,
    radiometricUsesByLayer,
    cpuRadiometricPositiveSupport,
    allocationBudget,
  );
  cpuRadiometricPositiveSupport.clear();

  const materialBaseTexel = 0;
  const triangleMaterialBaseTexel =
    materialBaseTexel + materialRecordCount * MATERIAL_MAP_META_TEXELS_PER_TRI;
  const triangleMaterialTexels = Math.ceil(triCount / 4);
  const uvAffineBaseTexel = triangleMaterialBaseTexel + triangleMaterialTexels;
  const uvAffineLaneCount = sortedHighTexCoords.length;
  const atlasAddressBaseTexel = Number(
    checkedAllocationProduct(
      'material UV affine metadata texels',
      [triCount, uvAffineLaneCount, 2],
    ) + BigInt(uvAffineBaseTexel),
  );
  const atlasAddressTexelCount =
    atlasLayers.length * MATERIAL_ATLAS_ADDRESS_TEXELS_PER_LAYER;
  const rawMetaTexels =
    BigInt(atlasAddressBaseTexel) +
    BigInt(atlasAddressTexelCount) +
    BigInt(MATERIAL_META_HEADER_TEXELS);
  if (rawMetaTexels > BigInt(MATERIAL_META_MAX_EXACT_UINT)) {
    throw new RangeError(
      `packMaterialTextureAtlas: metadata requires ${rawMetaTexels.toString()} texels; ` +
      `the rgba32float numeric-integer ABI supports at most ` +
      `${MATERIAL_META_MAX_EXACT_UINT} exactly addressable texels.`,
    );
  }
  if (
    materialRecordCount > MATERIAL_META_MAX_EXACT_UINT ||
    triCount > MATERIAL_META_MAX_EXACT_UINT ||
    atlasLayers.length > MATERIAL_META_MAX_EXACT_UINT
  ) {
    throw new RangeError(
      'packMaterialTextureAtlas: metadata counts exceed the exact binary32 integer domain.',
    );
  }
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
  const baseColorMetaData = new Float32Array(baseColorMetaElementCount);
  // Versioned address header. Metadata is per material; only a packed
  // triangle→material table and active high-UV affine lanes scale with geometry.
  // Integer address words are stored as exact numeric f32 values. Raw u32 bit
  // patterns such as 0x00000003 are f32 subnormals and may be flushed to zero
  // by conforming WebGPU implementations during texel copies. The allocation
  // guard above makes the <2^24 exact-integer domain an explicit ABI limit.
  // ABI v3
  // reserves header texel 2 for the device-populated packed-atlas directory:
  // { atlasAddressBaseTexel, logicalLayerCount, reserved, reserved }.
  const headerBaseTexel =
    baseColorMetaWidth * baseColorMetaHeight - MATERIAL_META_HEADER_TEXELS;
  baseColorMetaData.set([
    3, materialRecordCount, triCount, MATERIAL_MAP_META_TEXELS_PER_TRI,
    materialBaseTexel, triangleMaterialBaseTexel, uvAffineBaseTexel, uvAffineLaneCount,
    atlasAddressBaseTexel, atlasLayers.length, 0, 0,
  ], headerBaseTexel * 4);
  for (let tri = 0; tri < triCount; tri += 1) {
    const materialId = triMaterialIds[tri] ?? 0;
    const safeMaterialId = materialId < materials.length ? materialId : 0;
    baseColorMetaData[
      (triangleMaterialBaseTexel + Math.floor(tri / 4)) * 4 + tri % 4
    ] = safeMaterialId;
  }
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
    const clamped = Number.isFinite(mat?.normalScale)
      ? Math.max(0, mat?.normalScale ?? 1)
      : 1;
    writeScalarMeta(texel, finiteOrFallback(clamped, 1));
  };

  const writeFaceLayerNormalScaleMeta = (
    layer: MaterialSpec['frontLayer']   | undefined,
    texel: number,
  ): void => {
    const clamped = Number.isFinite(layer?.normalScale)
      ? Math.max(0, layer?.normalScale ?? 1)
      : 1;
    writeScalarMeta(texel, finiteOrFallback(clamped, 1));
  };

  const writeClearcoatNormalScaleMeta = (mat: MaterialSpec | undefined, texel: number): void => {
    const clamped = Number.isFinite(mat?.clearcoatNormalScale)
      ? Math.max(0, mat?.clearcoatNormalScale ?? 1)
      : 1;
    writeScalarMeta(texel, finiteOrFallback(clamped, 1));
  };

  const writeLightMapIntensityMeta = (
    materialIndex: number,
    texel: number,
  ): void => {
    writeScalarMeta(
      texel,
      lightMapIntensities[materialIndex] ??
        packNonNegativeRadianceScalarF32(
          1,
          `material ${materialIndex} lightMapIntensity`,
        ),
    );
  };

  const writeSpecularMeta = (mat: MaterialSpec | undefined, texel: number): void => {
    const b = texel * 4;
    const dielectricF0 = dielectricF0FromIor(mat?.ior);
    // Store absolute dielectric F0. Nonnegative KHR specularColor factors are
    // intentionally unbounded, so authored values above one remain observable.
    baseColorMetaData[b] = finiteOrFallback(
      dielectricF0 * nonNegativeVec3Component(mat?.specularColor, 0, 1),
      dielectricF0,
    );
    baseColorMetaData[b + 1] = finiteOrFallback(
      dielectricF0 * nonNegativeVec3Component(mat?.specularColor, 1, 1),
      dielectricF0,
    );
    baseColorMetaData[b + 2] = finiteOrFallback(
      dielectricF0 * nonNegativeVec3Component(mat?.specularColor, 2, 1),
      dielectricF0,
    );
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
    baseColorMetaData[b + 1] = finiteOrFallback(mat?.anisotropyRotation, 0);
    baseColorMetaData[b + 2] = 0;
    baseColorMetaData[b + 3] = 0;
  };

  const writeIridescenceMeta = (mat: MaterialSpec | undefined, texel: number): void => {
    const b = texel * 4;
    const range = mat?.iridescenceThicknessRange;
    baseColorMetaData[b] = clampedUnit(mat?.iridescence, 0);
    baseColorMetaData[b + 1] = finiteOrFallback(
      Number.isFinite(mat?.iridescenceIor)
        ? Math.max(1, mat?.iridescenceIor ?? 1.3)
        : 1.3,
      1.3,
    );
    baseColorMetaData[b + 2] = finiteOrFallback(
      Number.isFinite(range?.[0]) ? Math.max(0, range?.[0] ?? 100) : 100,
      100,
    );
    baseColorMetaData[b + 3] = finiteOrFallback(
      Number.isFinite(range?.[1]) ? Math.max(0, range?.[1] ?? 400) : 400,
      400,
    );
  };

  const writeBumpScaleMeta = (mat: MaterialSpec | undefined, texel: number): void => {
    const b = texel * 4;
    const ref = asTextureRef(mat?.bumpMap);
    const texCoord = ref?.texCoord ?? 0;
    const record = ref?.handle != null && compactTexCoordCode.has(texCoord)
      ? readable.get(ref.handle)?.linear
      : undefined;
    baseColorMetaData[b] = finiteOrFallback(mat?.bumpScale, 1);
    baseColorMetaData[b + 1] = record?.width ?? 0;
    baseColorMetaData[b + 2] = record?.height ?? 0;
    baseColorMetaData[b + 3] = 0;
  };

  const writeEnvMapIntensityMeta = (
    mat: MaterialSpec | undefined,
    texel: number,
    materialIndex: number,
  ): void => {
    writeScalarMeta(
      texel,
      assertWalkaroundEnvironmentScaleF32(
        mat?.envMapIntensity ?? 1,
        `material ${materialIndex} envMapIntensity`,
      ),
    );
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

  const uvAt = (
    stream: Float32Array,
    vertexIndex: number,
  ): readonly [number, number] => [
    stream[vertexIndex * 2] ?? Number.NaN,
    stream[vertexIndex * 2 + 1] ?? Number.NaN,
  ];
  const uvTriangleBasis = (
    a: readonly [number, number],
    b: readonly [number, number],
    c: readonly [number, number],
  ): {
    readonly scale: number;
    readonly e1x: number;
    readonly e1y: number;
    readonly e2x: number;
    readonly e2y: number;
    readonly determinant: number;
  } => {
    const rawE1x = b[0] - a[0];
    const rawE1y = b[1] - a[1];
    const rawE2x = c[0] - a[0];
    const rawE2y = c[1] - a[1];
    const scale = Math.max(
      Math.abs(rawE1x),
      Math.abs(rawE1y),
      Math.abs(rawE2x),
      Math.abs(rawE2y),
    );
    if (!(scale > 0) || !Number.isFinite(scale)) {
      return {
        scale,
        e1x: 0,
        e1y: 0,
        e2x: 0,
        e2y: 0,
        determinant: 0,
      };
    }
    const e1x = rawE1x / scale;
    const e1y = rawE1y / scale;
    const e2x = rawE2x / scale;
    const e2y = rawE2y / scale;
    return {
      scale,
      e1x,
      e1y,
      e2x,
      e2y,
      determinant: e1x * e2y - e1y * e2x,
    };
  };
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
    const targetBasis = uvTriangleBasis(ta, tb, tc);
    const constantTarget = targetBasis.scale === 0;
    let source = geometry.uv0;
    let sourceSelector = 0;
    let sa = uvAt(source, i0); let sb = uvAt(source, i1); let sc = uvAt(source, i2);
    let sourceBasis = uvTriangleBasis(sa, sb, sc);
    const sourceBasisUsable = (): boolean => (
      [...sa, ...sb, ...sc].every(Number.isFinite) &&
      sourceBasis.scale > 0 &&
      Math.abs(sourceBasis.determinant) > 1e-12
    );
    if (!sourceBasisUsable() && geometry.uv1 != null) {
      source = geometry.uv1;
      sourceSelector = 1;
      sa = uvAt(source, i0); sb = uvAt(source, i1); sc = uvAt(source, i2);
      sourceBasis = uvTriangleBasis(sa, sb, sc);
    }
    let m00 = 0; let m01 = 0; let b0 = ta[0];
    let m10 = 0; let m11 = 0; let b1 = ta[1];
    if (!constantTarget) {
      if (!sourceBasisUsable()) {
        throw new RangeError(
          `packMaterialTextureAtlas: triangle ${tri} texCoord ${texCoord} cannot be ` +
          'reconstructed because neither UV0 nor UV1 supplies a finite, ' +
          'non-degenerate source chart.',
        );
      }
      const inv00 = sourceBasis.e2y / sourceBasis.determinant;
      const inv01 = -sourceBasis.e2x / sourceBasis.determinant;
      const inv10 = -sourceBasis.e1y / sourceBasis.determinant;
      const inv11 = sourceBasis.e1x / sourceBasis.determinant;
      const du0 = tb[0] - ta[0]; const du1 = tc[0] - ta[0];
      const dv0 = tb[1] - ta[1]; const dv1 = tc[1] - ta[1];
      m00 = (du0 * inv00 + du1 * inv10) / sourceBasis.scale;
      m01 = (du0 * inv01 + du1 * inv11) / sourceBasis.scale;
      m10 = (dv0 * inv00 + dv1 * inv10) / sourceBasis.scale;
      m11 = (dv0 * inv01 + dv1 * inv11) / sourceBasis.scale;
      b0 = ta[0] - m00 * sa[0] - m01 * sa[1];
      b1 = ta[1] - m10 * sa[0] - m11 * sa[1];
    }
    const affine = [m00, m01, b0, m10, m11, b1].map(Math.fround);
    if (
      affine.some((value) => !Number.isFinite(value)) ||
      affine.some((value, index) => (
        value === 0 &&
        [m00, m01, b0, m10, m11, b1][index] !== 0
      ))
    ) {
      throw new RangeError(
        `packMaterialTextureAtlas: triangle ${tri} texCoord ${texCoord} ` +
        'requires affine coefficients outside the finite f32 domain.',
      );
    }
    [m00, m01, b0, m10, m11, b1] = affine as [
      number, number, number, number, number, number,
    ];
    const lane = compactCode - 2;
    const texel0 = uvAffineBaseTexel
      + tri * uvAffineLaneCount * 2
      + lane * 2;
    const base = texel0 * 4;
    baseColorMetaData.set([m00, m01, b0, sourceSelector, m10, m11, b1, 1], base);
  };

  for (let materialIndex = 0; materialIndex < materialRecordCount; materialIndex += 1) {
    const baseTexel = materialBaseTexel
      + materialIndex * MATERIAL_MAP_META_TEXELS_PER_TRI;
    const mat = materials[materialIndex];
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
    writeLightMapIntensityMeta(materialIndex, baseTexel + offsets.LIGHT_INTENSITY);
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
    writeEnvMapIntensityMeta(
      mat,
      baseTexel + offsets.ENV_INTENSITY,
      materialIndex,
    );
    writeFaceLayerMeta(mat?.frontLayer, baseTexel + offsets.FRONT_LAYER);
    writeFaceLayerMeta(mat?.backLayer, baseTexel + offsets.BACK_LAYER);
    writeVolumeScatteringMeta(mat, baseTexel + offsets.VOLUME_SCATTERING);
    writeMapMeta(mat, 'frontLayer.normalMap', 'linear', baseTexel + offsets.FRONT_LAYER_NORMAL);
    writeFaceLayerNormalScaleMeta(mat?.frontLayer, baseTexel + offsets.FRONT_LAYER_NORMAL_SCALE);
    writeMapMeta(mat, 'backLayer.normalMap', 'linear', baseTexel + offsets.BACK_LAYER_NORMAL);
    writeFaceLayerNormalScaleMeta(mat?.backLayer, baseTexel + offsets.BACK_LAYER_NORMAL_SCALE);
    baseColorMetaData.set(
      opticalMetaByMaterial[materialIndex] ?? emptyOpticalMeta,
      (baseTexel + offsets.OPTICAL_HEADER) * 4,
    );
  }
  for (let tri = 0; tri < triCount; tri += 1) {
    const materialIndex = triMaterialIds[tri] ?? 0;
    for (const texCoord of materialHighTexCoords[materialIndex] ?? []) {
      writeUvAffine(tri, texCoord, compactTexCoordCode.get(texCoord)!);
    }
  }

  return {
    atlasLayers,
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
