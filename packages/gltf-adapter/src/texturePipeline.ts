// texturePipeline.ts — scene-level diagnostics for decoded glTF texture handles.

import type {
  MaterialSpec,
  Scene,
  ScenePrimitive,
  TextureFilterMode,
  TextureMipFilterMode,
  TextureRef,
  TextureWrapMode,
} from '@vitrum/core';
import { gltfTextureRefSource, type RawImageHandle } from './textures.js';

export type GltfMaterialTextureField =
  | 'baseColorMap'
  | 'normalMap'
  | 'roughnessMap'
  | 'metallicMap'
  | 'transmissionMap'
  | 'thicknessMap'
  | 'emissiveMap'
  | 'alphaMap'
  | 'aoMap'
  | 'clearcoatMap'
  | 'clearcoatRoughnessMap'
  | 'clearcoatNormalMap'
  | 'sheenColorMap'
  | 'sheenRoughnessMap'
  | 'iridescenceMap'
  | 'iridescenceThicknessMap'
  | 'anisotropyMap'
  | 'specularColorMap'
  | 'specularIntensityMap'
  | 'bumpMap'
  | 'displacementMap'
  | 'lightMap';

export type GltfTextureHandleKind =
  | 'raw-image'
  | 'pixel-data'
  | 'data-texture'
  | 'image-bitmap'
  | 'opaque';

export type GltfTextureColorSpace = 'srgb' | 'linear';

export type GltfBackendTextureStatus = 'ready' | 'opaque' | 'ignored';

export interface GltfTextureDecodeReportEntry {
  readonly primitiveId: string;
  readonly primitiveKind: ScenePrimitive['kind'];
  readonly primitiveIndex: number;
  readonly materialField: GltfMaterialTextureField;
  readonly path: string;
  readonly texCoord: number;
  readonly hasTransform: boolean;
  readonly wrapS: TextureWrapMode;
  readonly wrapT: TextureWrapMode;
  readonly magFilter?: TextureFilterMode;
  readonly minFilter?: TextureFilterMode;
  readonly mipFilter?: TextureMipFilterMode;
  readonly usesMipmaps?: boolean;
  readonly colorSpace: GltfTextureColorSpace;
  readonly handleKind: GltfTextureHandleKind;
  readonly backendReadiness: {
    readonly ptWebgl2: GltfBackendTextureStatus;
    readonly ptWebgpu: GltfBackendTextureStatus;
    readonly walkaroundHybrid: GltfBackendTextureStatus;
  };
}

export interface GltfTextureDecodeReport {
  readonly mapCount: number;
  readonly uniqueHandleCount: number;
  readonly rawImageCount: number;
  readonly opaqueHandleCount: number;
  readonly cpuReadableCount: number;
  readonly rawImageRefs: readonly GltfTextureDecodeReportEntry[];
  readonly entries: readonly GltfTextureDecodeReportEntry[];
}

export interface GltfDecodedTexturePixels {
  readonly width: number;
  readonly height: number;
  readonly data: ArrayLike<number>;
  readonly channels?: 1 | 2 | 3 | 4;
  readonly dataType?: 'uint8' | 'uint16' | 'float32';
  readonly colorSpace?: GltfTextureColorSpace;
}

export interface GltfCpuLinearTextureHandle {
  readonly width: number;
  readonly height: number;
  readonly data: Float32Array;
  readonly __vitrum_hint__: {
    readonly channels: 4;
    readonly dataType: 'float32';
    readonly colorSpace: 'linear';
  };
}

export interface GltfCpuTextureHandle {
  readonly width: number;
  readonly height: number;
  readonly data: Float32Array;
  readonly __vitrum_hint__: {
    readonly channels: 4;
    readonly dataType: 'float32';
    readonly colorSpace: GltfTextureColorSpace;
  };
}

export type DecodeGltfTexturePixelsFn = (
  handle: RawImageHandle,
  context: {
    readonly materialField: GltfMaterialTextureField;
    readonly path: string;
    readonly colorSpace: GltfTextureColorSpace;
    readonly primitiveId: string;
    readonly primitiveIndex: number;
  },
) => Promise<GltfDecodedTexturePixels> | GltfDecodedTexturePixels;

export type DecodeSceneTextureDiagnosticCode =
  | 'unsupported-handle-kind'
  | 'raw-image-decoder-missing'
  | 'decoded-texture-exceeds-max-size'
  | 'decoded-texture-npot-repeat-wrap'
  | 'spec-gloss-alpha-bake-unavailable';

export interface DecodeSceneTextureDiagnostic {
  readonly severity: 'warning';
  readonly code: DecodeSceneTextureDiagnosticCode;
  readonly path: string;
  readonly materialField: GltfMaterialTextureField;
  readonly primitiveId: string;
  readonly primitiveIndex: number;
  readonly message: string;
  readonly handleKind?: GltfTextureHandleKind;
  readonly width?: number;
  readonly height?: number;
  readonly maxTextureSize?: number;
  readonly resizedWidth?: number;
  readonly resizedHeight?: number;
  readonly wrapS?: TextureWrapMode;
  readonly wrapT?: TextureWrapMode;
}

export interface DecodeSceneTexturesOptions {
  readonly target: 'cpu-linear' | 'webgpu';
  readonly decodePixels?: DecodeGltfTexturePixelsFn;
  readonly maxTextureSize?: number;
  readonly warnOnNpotRepeatWrap?: boolean;
  readonly onDiagnostic?: (diagnostic: DecodeSceneTextureDiagnostic) => void;
  readonly onWarning?: (message: string) => void;
}

export interface DecodeSceneTexturesResult {
  readonly scene: Scene;
  readonly report: GltfTextureDecodeReport;
  readonly decodedCount: number;
  readonly unchangedCount: number;
  readonly diagnostics: readonly DecodeSceneTextureDiagnostic[];
  readonly warnings: readonly string[];
}

interface DecodedTextureCacheEntry {
  readonly handle: GltfCpuTextureHandle;
  readonly originalWidth: number;
  readonly originalHeight: number;
}

type SpecGlossRoughnessBakeCache = Map<unknown, Map<number, GltfCpuLinearTextureHandle>>;

const MATERIAL_TEXTURE_FIELDS: readonly GltfMaterialTextureField[] = [
  'baseColorMap',
  'normalMap',
  'roughnessMap',
  'metallicMap',
  'transmissionMap',
  'thicknessMap',
  'emissiveMap',
  'alphaMap',
  'aoMap',
  'clearcoatMap',
  'clearcoatRoughnessMap',
  'clearcoatNormalMap',
  'sheenColorMap',
  'sheenRoughnessMap',
  'iridescenceMap',
  'iridescenceThicknessMap',
  'anisotropyMap',
  'specularColorMap',
  'specularIntensityMap',
  'bumpMap',
  'displacementMap',
  'lightMap',
];

const SRGB_TEXTURE_FIELDS = new Set<GltfMaterialTextureField>([
  'baseColorMap',
  'emissiveMap',
  'sheenColorMap',
  'specularColorMap',
]);

const WALKAROUND_ATLAS_TEXTURE_FIELDS = new Set<GltfMaterialTextureField>([
  'baseColorMap',
  'normalMap',
  'roughnessMap',
  'metallicMap',
  'aoMap',
  'alphaMap',
  'emissiveMap',
  'transmissionMap',
  'thicknessMap',
  'lightMap',
  'specularColorMap',
  'specularIntensityMap',
  'clearcoatMap',
  'clearcoatRoughnessMap',
  'clearcoatNormalMap',
  'bumpMap',
  'sheenColorMap',
  'sheenRoughnessMap',
  'anisotropyMap',
  'iridescenceMap',
  'iridescenceThicknessMap',
]);

export function gltfTextureColorSpaceForField(field: GltfMaterialTextureField): GltfTextureColorSpace {
  return SRGB_TEXTURE_FIELDS.has(field) ? 'srgb' : 'linear';
}

export function buildTextureDecodeReport(scene: Scene): GltfTextureDecodeReport {
  const entries: GltfTextureDecodeReportEntry[] = [];
  const uniqueHandles = new Set<unknown>();
  for (const [primitiveIndex, primitive] of scene.primitives.entries()) {
    const material = materialForPrimitive(primitive);
    for (const field of MATERIAL_TEXTURE_FIELDS) {
      const ref = material[field] as TextureRef | undefined;
      if (!ref) continue;
      uniqueHandles.add(ref.handle);
      const handleKind = classifyTextureHandle(ref.handle);
      const samplerFields = textureSamplerReportFields(ref);
      const scenePath = `scene.primitives[${primitiveIndex}].material.${field}`;
      entries.push({
        primitiveId: String(primitive.id),
        primitiveKind: primitive.kind,
        primitiveIndex,
        materialField: field,
        path: gltfTextureRefSource(ref)?.path ?? scenePath,
        texCoord: ref.texCoord ?? 0,
        hasTransform: ref.transform !== undefined,
        wrapS: ref.wrapS ?? 'repeat',
        wrapT: ref.wrapT ?? 'repeat',
        ...samplerFields,
        colorSpace: gltfTextureColorSpaceForField(field),
        handleKind,
        backendReadiness: backendReadinessForHandle(field, handleKind),
      });
    }
  }

  const rawImageRefs = entries.filter((entry) => entry.handleKind === 'raw-image');
  return {
    mapCount: entries.length,
    uniqueHandleCount: uniqueHandles.size,
    rawImageCount: rawImageRefs.length,
    opaqueHandleCount: entries.filter((entry) => entry.handleKind === 'opaque').length,
    cpuReadableCount: entries.filter((entry) =>
      entry.handleKind === 'pixel-data' || entry.handleKind === 'data-texture',
    ).length,
    rawImageRefs,
    entries,
  };
}

function textureSamplerReportFields(
  ref: TextureRef,
): Pick<GltfTextureDecodeReportEntry, 'magFilter' | 'minFilter' | 'mipFilter' | 'usesMipmaps'> {
  type SamplerReportFields = Pick<
    GltfTextureDecodeReportEntry,
    'magFilter' | 'minFilter' | 'mipFilter' | 'usesMipmaps'
  >;
  const fields: { -readonly [K in keyof SamplerReportFields]?: SamplerReportFields[K] } = {};
  if (ref.magFilter !== undefined) fields.magFilter = ref.magFilter;
  if (ref.minFilter !== undefined) fields.minFilter = ref.minFilter;
  if (ref.mipFilter !== undefined) {
    fields.mipFilter = ref.mipFilter;
    fields.usesMipmaps = ref.mipFilter !== 'none';
  }
  return fields;
}

export async function decodeSceneTextures(
  scene: Scene,
  options: DecodeSceneTexturesOptions,
): Promise<DecodeSceneTexturesResult> {
  const warnings: string[] = [];
  const diagnostics: DecodeSceneTextureDiagnostic[] = [];
  const decoded = new Map<unknown, Map<GltfTextureColorSpace, DecodedTextureCacheEntry>>();
  const specGlossBakes: SpecGlossRoughnessBakeCache = new Map();
  let decodedCount = 0;
  let unchangedCount = 0;

  const diagnostic = (entry: DecodeSceneTextureDiagnostic): void => {
    diagnostics.push(entry);
    warnings.push(entry.message);
    options.onDiagnostic?.(entry);
    options.onWarning?.(entry.message);
  };

  const primitives = await Promise.all(scene.primitives.map(async (primitive, primitiveIndex) => {
    const material = materialForPrimitive(primitive);
    let nextMaterial: MaterialSpec | null = null;
    for (const field of MATERIAL_TEXTURE_FIELDS) {
      const ref = material[field] as TextureRef | undefined;
      if (!ref) continue;
      const scenePath = `scene.primitives[${primitiveIndex}].material.${field}`;
      const path = gltfTextureRefSource(ref)?.path ?? scenePath;
      const nextRef = await decodeTextureRef(ref, {
        field,
        path,
        primitiveId: String(primitive.id),
        primitiveIndex,
        options,
        decoded,
        diagnostic,
      });
      if (nextRef === ref) {
        unchangedCount += 1;
        continue;
      }
      decodedCount += 1;
      if (nextMaterial == null) nextMaterial = { ...material };
      (nextMaterial as unknown as Record<string, unknown>)[field] = nextRef;
    }
    const baked = maybeBakeSpecGlossRoughnessMap(nextMaterial ?? material, {
      primitiveId: String(primitive.id),
      primitiveIndex,
      options,
      diagnostic,
      specGlossBakes,
    });
    if (baked !== null) {
      if (nextMaterial == null) nextMaterial = { ...material };
      (nextMaterial as unknown as Record<string, unknown>).roughnessMap = baked;
      decodedCount += 1;
    }
    return nextMaterial == null ? primitive : { ...primitive, material: nextMaterial };
  }));

  const nextScene = { ...scene, primitives } as Scene;
  return {
    scene: nextScene,
    report: buildTextureDecodeReport(nextScene),
    decodedCount,
    unchangedCount,
    diagnostics,
    warnings,
  };
}

export function classifyTextureHandle(handle: unknown): GltfTextureHandleKind {
  if (isRawImageHandle(handle)) return 'raw-image';
  if (isDataTextureLike(handle)) return 'data-texture';
  if (isPixelDataLike(handle)) return 'pixel-data';
  if (isImageBitmapLike(handle)) return 'image-bitmap';
  return 'opaque';
}

function materialForPrimitive(primitive: ScenePrimitive): MaterialSpec {
  return primitive.material;
}

function backendReadinessForHandle(
  field: GltfMaterialTextureField,
  handleKind: GltfTextureHandleKind,
): GltfTextureDecodeReportEntry['backendReadiness'] {
  const cpuReady = handleKind === 'pixel-data' || handleKind === 'data-texture';
  return {
    ptWebgl2: cpuReady ? 'ready' : 'opaque',
    ptWebgpu: handleKind === 'opaque' || handleKind === 'raw-image' ? 'opaque' : 'ready',
    walkaroundHybrid: WALKAROUND_ATLAS_TEXTURE_FIELDS.has(field)
      ? (cpuReady ? 'ready' : 'opaque')
      : 'ignored',
  };
}

async function decodeTextureRef(
  ref: TextureRef,
  context: {
    readonly field: GltfMaterialTextureField;
    readonly path: string;
    readonly primitiveId: string;
    readonly primitiveIndex: number;
    readonly options: DecodeSceneTexturesOptions;
    readonly decoded: Map<unknown, Map<GltfTextureColorSpace, DecodedTextureCacheEntry>>;
    readonly diagnostic: (diagnostic: DecodeSceneTextureDiagnostic) => void;
  },
): Promise<TextureRef> {
  const handleKind = classifyTextureHandle(ref.handle);
  if (handleKind === 'pixel-data' || handleKind === 'data-texture') return ref;
  if (handleKind !== 'raw-image') {
    context.diagnostic({
      severity: 'warning',
      code: 'unsupported-handle-kind',
      path: context.path,
      materialField: context.field,
      primitiveId: context.primitiveId,
      primitiveIndex: context.primitiveIndex,
      handleKind,
      message: `[vitrum/gltf-adapter] ${context.path} has ${handleKind} texture handle; ` +
        'decodeSceneTextures(target:"cpu-linear") can only normalize raw-image handles. Texture left unchanged.',
    });
    return ref;
  }
  if (context.options.decodePixels == null) {
    context.diagnostic({
      severity: 'warning',
      code: 'raw-image-decoder-missing',
      path: context.path,
      materialField: context.field,
      primitiveId: context.primitiveId,
      primitiveIndex: context.primitiveIndex,
      handleKind,
      message: `[vitrum/gltf-adapter] ${context.path} is a raw-image texture but no decodePixels hook was supplied. ` +
        'Texture left unchanged.',
    });
    return ref;
  }

  const colorSpace = gltfTextureColorSpaceForField(context.field);
  const outputColorSpace: GltfTextureColorSpace =
    context.options.target === 'webgpu' ? colorSpace : 'linear';
  let perSpace = context.decoded.get(ref.handle);
  if (perSpace == null) {
    perSpace = new Map();
    context.decoded.set(ref.handle, perSpace);
  }
  let entry = perSpace.get(colorSpace);
  if (entry == null) {
    const pixels = await context.options.decodePixels(ref.handle as RawImageHandle, {
      materialField: context.field,
      path: context.path,
      colorSpace,
      primitiveId: context.primitiveId,
      primitiveIndex: context.primitiveIndex,
    });
    const normalized = normalizeDecodedPixels(pixels, colorSpace, outputColorSpace);
    entry = {
      handle: resizeDecodedTextureToMaxSize(normalized, context.options.maxTextureSize),
      originalWidth: normalized.width,
      originalHeight: normalized.height,
    };
    perSpace.set(colorSpace, entry);
  }
  emitDecodedTextureDiagnostics(entry, ref, context);
  return { ...ref, handle: entry.handle };
}

function maybeBakeSpecGlossRoughnessMap(
  material: MaterialSpec,
  context: {
    readonly primitiveId: string;
    readonly primitiveIndex: number;
    readonly options: DecodeSceneTexturesOptions;
    readonly diagnostic: (diagnostic: DecodeSceneTextureDiagnostic) => void;
    readonly specGlossBakes: SpecGlossRoughnessBakeCache;
  },
): TextureRef | null {
  if (context.options.target !== 'cpu-linear') return null;
  const specGloss = material.extensions?.KHR_materials_pbrSpecularGlossiness;
  if (!isRecord(specGloss) || !isRecord(specGloss.specularGlossinessTexture)) return null;
  const sourceRef = material.specularColorMap;
  if (sourceRef == null) return null;
  const path = gltfTextureRefSource(sourceRef)?.path ??
    `scene.primitives[${context.primitiveIndex}].material.roughnessMap`;
  const glossinessFactor = clamp01Number(specGloss.glossinessFactor, 1);
  const sourceHandle = cpuLinearTextureHandleForSpecGlossBake(sourceRef.handle);
  if (sourceHandle !== null) {
    return {
      ...sourceRef,
      handle: getOrBakeSpecGlossRoughnessHandle(
        sourceRef.handle,
        sourceHandle,
        glossinessFactor,
        context.specGlossBakes,
      ),
    };
  }

  context.diagnostic({
    severity: 'warning',
    code: 'spec-gloss-alpha-bake-unavailable',
    path,
    materialField: 'roughnessMap',
    primitiveId: context.primitiveId,
    primitiveIndex: context.primitiveIndex,
    handleKind: classifyTextureHandle(sourceRef.handle),
    message:
      `[vitrum/gltf-adapter] ${path} uses ` +
      'KHR_materials_pbrSpecularGlossiness.specularGlossinessTexture alpha for glossiness, ' +
      'but the texture was not decoded to CPU-linear pixels, so no roughnessMap could be baked. ' +
      'Supply decodePixels through decodeSceneTextures() or loadGltfAndDecodeTextures() to derive roughness per pixel.',
  });
  return null;
}

function getOrBakeSpecGlossRoughnessHandle(
  cacheKey: unknown,
  source: GltfCpuLinearTextureHandle,
  glossinessFactor: number,
  cache: SpecGlossRoughnessBakeCache,
): GltfCpuLinearTextureHandle {
  let perSource = cache.get(cacheKey);
  if (perSource == null) {
    perSource = new Map();
    cache.set(cacheKey, perSource);
  }
  const key = Math.round(glossinessFactor * 1_000_000);
  const cached = perSource.get(key);
  if (cached !== undefined) return cached;

  const data = new Float32Array(source.width * source.height * 4);
  for (let p = 0; p < source.width * source.height; p += 1) {
    const alpha = clamp01Number(source.data[p * 4 + 3], 1);
    const roughness = 1 - clamp01Number(glossinessFactor * alpha, 1);
    const dst = p * 4;
    data[dst] = roughness;
    data[dst + 1] = roughness;
    data[dst + 2] = roughness;
    data[dst + 3] = 1;
  }

  const baked: GltfCpuLinearTextureHandle = {
    width: source.width,
    height: source.height,
    data,
    __vitrum_hint__: { channels: 4, dataType: 'float32', colorSpace: 'linear' },
  };
  perSource.set(key, baked);
  return baked;
}

function cpuLinearTextureHandleForSpecGlossBake(handle: unknown): GltfCpuLinearTextureHandle | null {
  if (isCpuLinearTextureHandle(handle)) return handle;
  const pixels = decodedPixelsFromCpuReadableHandle(handle);
  return pixels === null ? null : normalizeDecodedPixels(pixels, 'srgb');
}

function decodedPixelsFromCpuReadableHandle(handle: unknown): GltfDecodedTexturePixels | null {
  if (isRecord(handle)) {
    const direct = decodedPixelsFromRecord(handle);
    if (direct !== null) return direct;
    if (isRecord(handle.image)) return decodedPixelsFromRecord(handle.image, handle);
  }
  return null;
}

function decodedPixelsFromRecord(
  record: Record<string, unknown>,
  metadata: Record<string, unknown> = record,
): GltfDecodedTexturePixels | null {
  if (typeof record.width !== 'number' || typeof record.height !== 'number' || !isArrayLikeData(record.data)) {
    return null;
  }
  const base = {
    width: record.width,
    height: record.height,
    data: record.data as ArrayLike<number>,
  };
  const channels = metadata.channels;
  const dataType = metadata.dataType;
  const colorSpace = metadata.colorSpace;
  return {
    ...base,
    ...(channels === 1 || channels === 2 || channels === 3 || channels === 4 ? { channels } : {}),
    ...(dataType === 'uint8' || dataType === 'uint16' || dataType === 'float32' ? { dataType } : {}),
    ...(colorSpace === 'srgb' || colorSpace === 'linear' ? { colorSpace } : {}),
  };
}

function emitDecodedTextureDiagnostics(
  entry: DecodedTextureCacheEntry,
  ref: TextureRef,
  context: {
    readonly field: GltfMaterialTextureField;
    readonly path: string;
    readonly primitiveId: string;
    readonly primitiveIndex: number;
    readonly options: DecodeSceneTexturesOptions;
    readonly diagnostic: (diagnostic: DecodeSceneTextureDiagnostic) => void;
  },
): void {
  const handle = entry.handle;
  const maxTextureSize = context.options.maxTextureSize;
  if (typeof maxTextureSize === 'number' && maxTextureSize > 0 &&
      (entry.originalWidth > maxTextureSize || entry.originalHeight > maxTextureSize)) {
    context.diagnostic({
      severity: 'warning',
      code: 'decoded-texture-exceeds-max-size',
      path: context.path,
      materialField: context.field,
      primitiveId: context.primitiveId,
      primitiveIndex: context.primitiveIndex,
      width: entry.originalWidth,
      height: entry.originalHeight,
      maxTextureSize,
      resizedWidth: handle.width,
      resizedHeight: handle.height,
      message: `[vitrum/gltf-adapter] ${context.path} decoded to ${entry.originalWidth}x${entry.originalHeight}, ` +
        `which exceeds maxTextureSize=${maxTextureSize}. Texture was resized to ${handle.width}x${handle.height} ` +
        'during CPU-linear decode before backend upload.',
    });
  }

  if (context.options.warnOnNpotRepeatWrap === true && !isPowerOfTwo(handle.width, handle.height) &&
      usesRepeatWrap(ref)) {
    const wrapS = ref.wrapS ?? 'repeat';
    const wrapT = ref.wrapT ?? 'repeat';
    context.diagnostic({
      severity: 'warning',
      code: 'decoded-texture-npot-repeat-wrap',
      path: context.path,
      materialField: context.field,
      primitiveId: context.primitiveId,
      primitiveIndex: context.primitiveIndex,
      width: handle.width,
      height: handle.height,
      wrapS,
      wrapT,
      message: `[vitrum/gltf-adapter] ${context.path} decoded to NPOT ${handle.width}x${handle.height} ` +
        `with wrapS=${wrapS} wrapT=${wrapT}. WebGL2/WebGPU can sample NPOT textures, but exact mip/border ` +
        'parity depends on backend upload policy; pre-resize to power-of-two if this asset needs strict parity.',
    });
  }
}

function normalizeDecodedPixels(
  pixels: GltfDecodedTexturePixels,
  fieldColorSpace: GltfTextureColorSpace,
): GltfCpuLinearTextureHandle;
function normalizeDecodedPixels(
  pixels: GltfDecodedTexturePixels,
  fieldColorSpace: GltfTextureColorSpace,
  outputColorSpace: 'linear',
): GltfCpuLinearTextureHandle;
function normalizeDecodedPixels(
  pixels: GltfDecodedTexturePixels,
  fieldColorSpace: GltfTextureColorSpace,
  outputColorSpace: 'srgb',
): GltfCpuTextureHandle;
function normalizeDecodedPixels(
  pixels: GltfDecodedTexturePixels,
  fieldColorSpace: GltfTextureColorSpace,
  outputColorSpace: GltfTextureColorSpace,
): GltfCpuTextureHandle;
function normalizeDecodedPixels(
  pixels: GltfDecodedTexturePixels,
  fieldColorSpace: GltfTextureColorSpace,
  outputColorSpace: GltfTextureColorSpace = 'linear',
): GltfCpuTextureHandle {
  const width = Math.max(0, Math.floor(pixels.width));
  const height = Math.max(0, Math.floor(pixels.height));
  if (width <= 0 || height <= 0) {
    throw new Error(`[vitrum/gltf-adapter] decodePixels returned invalid texture dimensions ${pixels.width}x${pixels.height}.`);
  }
  const channels = pixels.channels ?? inferDecodedChannels(pixels.data, width, height);
  const dataType = pixels.dataType ?? inferDecodedDataType(pixels.data);
  const sourceColorSpace = pixels.colorSpace ?? fieldColorSpace;
  const out = new Float32Array(width * height * 4);
  for (let p = 0; p < width * height; p += 1) {
    const s = p * channels;
    const r = decodeChannel(pixels.data[s] ?? 0, dataType);
    const g = decodeChannel(pixels.data[s + (channels > 1 ? 1 : 0)] ?? 0, dataType);
    const b = decodeChannel(pixels.data[s + (channels > 2 ? 2 : 0)] ?? 0, dataType);
    const a = channels >= 4 ? decodeChannel(pixels.data[s + 3] ?? 1, dataType) : 1;
    out[p * 4] = convertColorChannel(r, sourceColorSpace, outputColorSpace);
    out[p * 4 + 1] = convertColorChannel(g, sourceColorSpace, outputColorSpace);
    out[p * 4 + 2] = convertColorChannel(b, sourceColorSpace, outputColorSpace);
    out[p * 4 + 3] = a;
  }
  return {
    width,
    height,
    data: out,
    __vitrum_hint__: { channels: 4, dataType: 'float32', colorSpace: outputColorSpace },
  };
}

function resizeDecodedTextureToMaxSize(
  handle: GltfCpuTextureHandle,
  maxTextureSize: number | undefined,
): GltfCpuTextureHandle {
  if (typeof maxTextureSize !== 'number' || maxTextureSize <= 0) return handle;
  if (handle.width <= maxTextureSize && handle.height <= maxTextureSize) return handle;

  const scale = maxTextureSize / Math.max(handle.width, handle.height);
  const width = Math.max(1, Math.round(handle.width * scale));
  const height = Math.max(1, Math.round(handle.height * scale));
  const data = new Float32Array(width * height * 4);

  for (let y = 0; y < height; y += 1) {
    const sy = Math.min(handle.height - 1, Math.floor((y * handle.height) / height));
    for (let x = 0; x < width; x += 1) {
      const sx = Math.min(handle.width - 1, Math.floor((x * handle.width) / width));
      const src = (sy * handle.width + sx) * 4;
      const dst = (y * width + x) * 4;
      data[dst] = handle.data[src]!;
      data[dst + 1] = handle.data[src + 1]!;
      data[dst + 2] = handle.data[src + 2]!;
      data[dst + 3] = handle.data[src + 3]!;
    }
  }

  return {
    width,
    height,
    data,
    __vitrum_hint__: {
      channels: 4,
      dataType: 'float32',
      colorSpace: handle.__vitrum_hint__.colorSpace,
    },
  };
}

function inferDecodedChannels(data: ArrayLike<number>, width: number, height: number): 1 | 2 | 3 | 4 {
  const stride = Math.max(1, Math.round(data.length / Math.max(1, width * height)));
  if (stride <= 1) return 1;
  if (stride === 2) return 2;
  if (stride === 3) return 3;
  return 4;
}

function inferDecodedDataType(data: ArrayLike<number>): 'uint8' | 'uint16' | 'float32' {
  if (data instanceof Uint8Array || data instanceof Uint8ClampedArray) return 'uint8';
  if (data instanceof Uint16Array) return 'uint16';
  return 'float32';
}

function isPowerOfTwo(width: number, height: number): boolean {
  return isSinglePowerOfTwo(width) && isSinglePowerOfTwo(height);
}

function isSinglePowerOfTwo(value: number): boolean {
  return Number.isInteger(value) && value > 0 && (value & (value - 1)) === 0;
}

function usesRepeatWrap(ref: TextureRef): boolean {
  return (ref.wrapS ?? 'repeat') !== 'clamp-to-edge' || (ref.wrapT ?? 'repeat') !== 'clamp-to-edge';
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

function isCpuLinearTextureHandle(handle: unknown): handle is GltfCpuLinearTextureHandle {
  if (!isRecord(handle)) return false;
  const hint = handle.__vitrum_hint__;
  return typeof handle.width === 'number' &&
    typeof handle.height === 'number' &&
    handle.data instanceof Float32Array &&
    isRecord(hint) &&
    hint.channels === 4 &&
    hint.dataType === 'float32' &&
    hint.colorSpace === 'linear';
}

function clamp01Number(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value)
    ? Math.min(1, Math.max(0, value))
    : fallback;
}

function decodeChannel(value: number, dataType: 'uint8' | 'uint16' | 'float32'): number {
  if (dataType === 'uint8') return Math.max(0, Math.min(1, value / 255));
  if (dataType === 'uint16') return Math.max(0, Math.min(1, value / 65535));
  return Number(value);
}

function srgbToLinear(v: number): number {
  const c = Math.max(0, Math.min(1, v));
  return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
}

function linearToSrgb(v: number): number {
  const c = Math.max(0, Math.min(1, v));
  return c <= 0.0031308 ? c * 12.92 : 1.055 * (c ** (1 / 2.4)) - 0.055;
}

function convertColorChannel(
  value: number,
  sourceColorSpace: GltfTextureColorSpace,
  outputColorSpace: GltfTextureColorSpace,
): number {
  if (sourceColorSpace === outputColorSpace) return value;
  if (sourceColorSpace === 'srgb' && outputColorSpace === 'linear') return srgbToLinear(value);
  return linearToSrgb(value);
}

function isRawImageHandle(handle: unknown): handle is RawImageHandle {
  return typeof handle === 'object' && handle !== null &&
    (handle as { kind?: unknown }).kind === 'raw-image';
}

function isPixelDataLike(handle: unknown): boolean {
  if (typeof handle !== 'object' || handle === null) return false;
  const h = handle as { width?: unknown; height?: unknown; data?: unknown };
  return typeof h.width === 'number' && typeof h.height === 'number' && isArrayLikeData(h.data);
}

function isDataTextureLike(handle: unknown): boolean {
  if (typeof handle !== 'object' || handle === null) return false;
  const image = (handle as { image?: unknown }).image;
  if (typeof image !== 'object' || image === null) return false;
  const h = image as { width?: unknown; height?: unknown; data?: unknown };
  return typeof h.width === 'number' && typeof h.height === 'number' && isArrayLikeData(h.data);
}

function isImageBitmapLike(handle: unknown): boolean {
  if (typeof ImageBitmap !== 'undefined' && handle instanceof ImageBitmap) return true;
  if (typeof handle !== 'object' || handle === null) return false;
  const h = handle as { width?: unknown; height?: unknown; close?: unknown };
  return typeof h.width === 'number' && typeof h.height === 'number' && typeof h.close === 'function';
}

function isArrayLikeData(data: unknown): boolean {
  return typeof data === 'object' && data !== null && typeof (data as { length?: unknown }).length === 'number';
}
