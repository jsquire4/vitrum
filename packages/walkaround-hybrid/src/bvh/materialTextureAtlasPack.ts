import type {
  MaterialSpec,
  TextureFilterMode,
  TextureMipFilterMode,
  TextureRef,
  TextureWrapMode,
} from '@vitrum/core';

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
export const MATERIAL_MAP_META_TEXELS_PER_TRI = 62;
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
} as const;

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
    | 'unsupported-material-texture-texcoord'
    | 'ambiguous-material-texture-stride'
    | 'invalid-material-texture-transform'
    | 'material-texture-sampler-policy-approximation';
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
  readonly diagnostics: readonly MaterialTextureAtlasDiagnostic[];
}

interface RawPixels {
  readonly width: number;
  readonly height: number;
  readonly data: Float32Array;
  readonly sourceColorSpace?: 'srgb' | 'linear';
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
  if ('handle' in value) return value as TextureRef;
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

function hasAuthoredSamplerPolicy(ref: TextureRef): boolean {
  const magFilter = ref.magFilter ?? 'nearest';
  const minFilter = ref.minFilter ?? 'nearest';
  const mipFilter = ref.mipFilter ?? 'none';
  return magFilter !== minFilter || mipFilter !== 'none';
}

function readHandlePixels(handle: unknown): ReadHandlePixelsResult | null {
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
  return {
    pixels: { width, height, data: out, ...(sourceColorSpace ? { sourceColorSpace } : {}) },
    ...(ambiguousStride ? { ambiguousStride } : {}),
  };
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

function samplerPolicyPacked(ref: TextureRef, texCoord: number): number {
  const wrapS = wrapIndex(ref.wrapS);
  const wrapT = wrapIndex(ref.wrapT);
  const mipFilter = MIP_FILTER_INDEX[ref.mipFilter ?? 'none'];
  const magFilter = FILTER_MODE_INDEX[ref.magFilter ?? 'nearest'];
  const minFilter = FILTER_MODE_INDEX[ref.minFilter ?? 'nearest'];
  return (
    wrapS +
    wrapT * 4 +
    texCoord * 16 +
    mipFilter * 64 +
    magFilter * 256 +
    minFilter * 512
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
): MaterialTextureAtlasPayload {
  const readable = new Map<unknown, Partial<Record<AtlasColorSpace, { readonly layer: number; readonly pixels: RawPixels }>>>();
  const ordered: { readonly handle: unknown; readonly pixels: RawPixels; readonly colorSpace: AtlasColorSpace }[] = [];
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
    const topLevelField = field as Exclude<AtlasMapField, 'frontLayer.normalMap' | 'backLayer.normalMap'>;
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
    if (texCoord !== 0 && texCoord !== 1) {
      const source = textureRefSourceMetadata(ref);
      diagnostics.push({
        code: 'unsupported-material-texture-texcoord',
        materialIndex,
        field,
        colorSpace,
        texCoord,
        ...(source?.path !== undefined ? { sourcePath: source.path } : {}),
        ...(source?.textureIndex !== undefined ? { textureIndex: source.textureIndex } : {}),
        ...(source?.imageIndex !== undefined ? { imageIndex: source.imageIndex } : {}),
        ...(source?.samplerIndex !== undefined ? { samplerIndex: source.samplerIndex } : {}),
        ...(source?.imageUri !== undefined ? { imageUri: source.imageUri } : {}),
        ...(source?.imageMimeType !== undefined ? { imageMimeType: source.imageMimeType } : {}),
        ...(source?.textureSourceExtension !== undefined
          ? { textureSourceExtension: source.textureSourceExtension }
          : {}),
        message:
          `${field} texture uses unsupported texCoord ${texCoord} ` +
          `(walkaround material atlas supports UV sets 0 and 1 only)` +
          `${source?.path !== undefined ? ` at ${source.path}` : ''}; the map is ignored.`,
      });
      return;
    }
    if (hasAuthoredSamplerPolicy(ref)) {
      const source = textureRefSourceMetadata(ref);
      diagnostics.push({
        code: 'material-texture-sampler-policy-approximation',
        materialIndex,
        field,
        colorSpace,
        texCoord,
        ...(ref.magFilter !== undefined ? { magFilter: ref.magFilter } : {}),
        ...(ref.minFilter !== undefined ? { minFilter: ref.minFilter } : {}),
        ...(ref.mipFilter !== undefined ? { mipFilter: ref.mipFilter } : {}),
        ...(source?.path !== undefined ? { sourcePath: source.path } : {}),
        ...(source?.textureIndex !== undefined ? { textureIndex: source.textureIndex } : {}),
        ...(source?.imageIndex !== undefined ? { imageIndex: source.imageIndex } : {}),
        ...(source?.samplerIndex !== undefined ? { samplerIndex: source.samplerIndex } : {}),
        ...(source?.imageUri !== undefined ? { imageUri: source.imageUri } : {}),
        ...(source?.imageMimeType !== undefined ? { imageMimeType: source.imageMimeType } : {}),
        ...(source?.textureSourceExtension !== undefined
          ? { textureSourceExtension: source.textureSourceExtension }
          : {}),
        message:
          `${field} texture authors sampler filter/mip policy ` +
          `(mag=${ref.magFilter ?? 'default'}, min=${ref.minFilter ?? 'default'}, mip=${ref.mipFilter ?? 'default'})` +
          `${source?.path !== undefined ? ` at ${source.path}` : ''}; ` +
          'walkaround material atlas honors footprint-independent nearest/linear filtering, but ' +
          'this policy needs implicit LOD or min/mag footprint selection in compute passes; ' +
          'the map remains atlas-backed with approximate mip/footprint filtering.',
      });
    }
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
        ...(source?.path !== undefined ? { sourcePath: source.path } : {}),
        ...(source?.textureIndex !== undefined ? { textureIndex: source.textureIndex } : {}),
        ...(source?.imageIndex !== undefined ? { imageIndex: source.imageIndex } : {}),
        ...(source?.samplerIndex !== undefined ? { samplerIndex: source.samplerIndex } : {}),
        ...(source?.imageUri !== undefined ? { imageUri: source.imageUri } : {}),
        ...(source?.imageMimeType !== undefined ? { imageMimeType: source.imageMimeType } : {}),
        ...(source?.textureSourceExtension !== undefined
          ? { textureSourceExtension: source.textureSourceExtension }
          : {}),
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
    const read = readHandlePixels(ref.handle);
    if (read == null) {
      const source = textureRefSourceMetadata(ref);
      diagnostics.push({
        code: 'unreadable-material-texture-map',
        materialIndex,
        field,
        colorSpace,
        ...(source?.path !== undefined ? { sourcePath: source.path } : {}),
        ...(source?.textureIndex !== undefined ? { textureIndex: source.textureIndex } : {}),
        ...(source?.imageIndex !== undefined ? { imageIndex: source.imageIndex } : {}),
        ...(source?.samplerIndex !== undefined ? { samplerIndex: source.samplerIndex } : {}),
        ...(source?.imageUri !== undefined ? { imageUri: source.imageUri } : {}),
        ...(source?.imageMimeType !== undefined ? { imageMimeType: source.imageMimeType } : {}),
        ...(source?.textureSourceExtension !== undefined
          ? { textureSourceExtension: source.textureSourceExtension }
          : {}),
        message:
          `${field} texture handle is not CPU-readable ` +
          `(expected raw {width,height,data} or DataTexture-shaped image)` +
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
        ...(source?.path !== undefined ? { sourcePath: source.path } : {}),
        ...(source?.textureIndex !== undefined ? { textureIndex: source.textureIndex } : {}),
        ...(source?.imageIndex !== undefined ? { imageIndex: source.imageIndex } : {}),
        ...(source?.samplerIndex !== undefined ? { samplerIndex: source.samplerIndex } : {}),
        ...(source?.imageUri !== undefined ? { imageUri: source.imageUri } : {}),
        ...(source?.imageMimeType !== undefined ? { imageMimeType: source.imageMimeType } : {}),
        ...(source?.textureSourceExtension !== undefined
          ? { textureSourceExtension: source.textureSourceExtension }
          : {}),
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
    ordered.push({ handle: ref.handle, pixels, colorSpace });
    perHandle ??= {};
    perHandle[colorSpace] = { layer, pixels };
    readable.set(ref.handle, perHandle);
    fieldLayers[field].add(layer);
  };

  materials.forEach((material, materialIndex) => {
    for (const { field, colorSpace } of ATLAS_MAP_FIELDS) {
      collect(material, materialIndex, field, colorSpace);
    }
  });

  const atlasDim = Math.max(1, ...ordered.map((entry) => Math.max(entry.pixels.width, entry.pixels.height)));
  const atlasLayerCount = Math.max(1, ordered.length);
  const atlasData = new Float32Array(atlasDim * atlasDim * 4 * atlasLayerCount);
  if (ordered.length === 0) {
    atlasData.set([1, 1, 1, 1]);
  } else {
    ordered.forEach((entry, layer) => {
      blitAtlasLayer(
        entry.pixels,
        atlasDim,
        atlasData,
        layer,
        entry.colorSpace === 'srgb' && entry.pixels.sourceColorSpace !== 'linear',
      );
    });
  }

  const metaTexels = Math.max(1, triCount * MATERIAL_MAP_META_TEXELS_PER_TRI);
  const baseColorMetaWidth = Math.min(BASE_COLOR_MAP_META_TEX_WIDTH, metaTexels);
  const baseColorMetaHeight = Math.ceil(metaTexels / baseColorMetaWidth);
  const baseColorMetaData = new Float32Array(baseColorMetaWidth * baseColorMetaHeight * 4);

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
    const layer = texCoord === 0 || texCoord === 1 ? readable.get(ref.handle)?.[colorSpace]?.layer : undefined;
    if (layer == null) {
      writeDisabledMeta(baseColorMetaData, texel);
      return;
    }
    const t = ref.transform;
    const rotation = finiteOrFallback(t?.rotation, 0);
    const b0 = texel * 4;
    const b1 = b0 + 4;
    baseColorMetaData[b0] = layer;
    baseColorMetaData[b0 + 1] = samplerPolicyPacked(ref, texCoord);
    baseColorMetaData[b0 + 2] = finiteOrFallback(t?.offset?.[0], 0);
    baseColorMetaData[b0 + 3] = finiteOrFallback(t?.offset?.[1], 0);
    baseColorMetaData[b1] = finiteOrFallback(t?.scale?.[0], 1);
    baseColorMetaData[b1 + 1] = finiteOrFallback(t?.scale?.[1], 1);
    baseColorMetaData[b1 + 2] = Math.cos(rotation);
    baseColorMetaData[b1 + 3] = Math.sin(rotation);
  };

  const writeAlphaCoverageMeta = (mat: MaterialSpec | undefined, texel: number): void => {
    const b = texel * 4;
    baseColorMetaData[b] = alphaModeIndex(mat?.alphaMode);
    baseColorMetaData[b + 1] = clampedUnit(mat?.opacity, 1);
    baseColorMetaData[b + 2] = clampedUnit(mat?.alphaCutoff, 0.5);
    baseColorMetaData[b + 3] = 0;
  };

  const writeNormalScaleMeta = (mat: MaterialSpec | undefined, texel: number): void => {
    const b = texel * 4;
    baseColorMetaData[b] = Number.isFinite(mat?.normalScale)
      ? Math.max(0, mat?.normalScale ?? 1)
      : 1;
    baseColorMetaData[b + 1] = 0;
    baseColorMetaData[b + 2] = 0;
    baseColorMetaData[b + 3] = 0;
  };

  const writeFaceLayerNormalScaleMeta = (
    layer: MaterialSpec['frontLayer'] | MaterialSpec['backLayer'] | undefined,
    texel: number,
  ): void => {
    const b = texel * 4;
    baseColorMetaData[b] = Number.isFinite(layer?.normalScale)
      ? Math.max(0, layer?.normalScale ?? 1)
      : 1;
    baseColorMetaData[b + 1] = 0;
    baseColorMetaData[b + 2] = 0;
    baseColorMetaData[b + 3] = 0;
  };

  const writeClearcoatNormalScaleMeta = (mat: MaterialSpec | undefined, texel: number): void => {
    const b = texel * 4;
    baseColorMetaData[b] = Number.isFinite(mat?.clearcoatNormalScale)
      ? Math.max(0, mat?.clearcoatNormalScale ?? 1)
      : 1;
    baseColorMetaData[b + 1] = 0;
    baseColorMetaData[b + 2] = 0;
    baseColorMetaData[b + 3] = 0;
  };

  const writeLightMapIntensityMeta = (mat: MaterialSpec | undefined, texel: number): void => {
    const b = texel * 4;
    baseColorMetaData[b] = Number.isFinite(mat?.lightMapIntensity)
      ? Math.max(0, mat?.lightMapIntensity ?? 1)
      : 1;
    baseColorMetaData[b + 1] = 0;
    baseColorMetaData[b + 2] = 0;
    baseColorMetaData[b + 3] = 0;
  };

  const writeSpecularMeta = (mat: MaterialSpec | undefined, texel: number): void => {
    const b = texel * 4;
    baseColorMetaData[b] = clampedColorComponent(mat?.specularColor, 0);
    baseColorMetaData[b + 1] = clampedColorComponent(mat?.specularColor, 1);
    baseColorMetaData[b + 2] = clampedColorComponent(mat?.specularColor, 2);
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
    const pixels = ref?.handle != null && (texCoord === 0 || texCoord === 1)
      ? readable.get(ref.handle)?.linear?.pixels
      : undefined;
    baseColorMetaData[b] = Number.isFinite(mat?.bumpScale)
      ? mat?.bumpScale ?? 1
      : 1;
    baseColorMetaData[b + 1] = pixels?.width ?? 0;
    baseColorMetaData[b + 2] = pixels?.height ?? 0;
    baseColorMetaData[b + 3] = 0;
  };

  const writeEnvMapIntensityMeta = (mat: MaterialSpec | undefined, texel: number): void => {
    const b = texel * 4;
    baseColorMetaData[b] = clampedNonNegative(mat?.envMapIntensity, 1);
    baseColorMetaData[b + 1] = 0;
    baseColorMetaData[b + 2] = 0;
    baseColorMetaData[b + 3] = 0;
  };

  const writeFaceLayerMeta = (
    layer: MaterialSpec['frontLayer'] | MaterialSpec['backLayer'] | undefined,
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
  }

  return {
    atlasData,
    atlasDim,
    atlasLayerCount,
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
    diagnostics,
  };
}
