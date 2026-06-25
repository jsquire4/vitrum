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
import {
  gltfTextureRefSource,
  type GltfTextureRefSource,
  type GltfTextureSourceExtension,
  type RawImageHandle,
} from './textures.js';

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
export type GltfNpotRepeatWrapPolicy = 'warn' | 'resize-to-pot' | 'clamp-sampler';

export interface GltfTextureDecodeReportEntry {
  readonly primitiveId: string;
  readonly primitiveKind: ScenePrimitive['kind'];
  readonly primitiveIndex: number;
  readonly materialField: GltfMaterialTextureField;
  readonly path: string;
  readonly imageSourcePath?: string;
  readonly texCoord: number;
  readonly hasTransform: boolean;
  readonly wrapS: TextureWrapMode;
  readonly wrapT: TextureWrapMode;
  readonly magFilter?: TextureFilterMode;
  readonly minFilter?: TextureFilterMode;
  readonly mipFilter?: TextureMipFilterMode;
  readonly usesMipmaps?: boolean;
  readonly width?: number;
  readonly height?: number;
  readonly isPowerOfTwo?: boolean;
  readonly originalWidth?: number;
  readonly originalHeight?: number;
  readonly wasResized?: boolean;
  readonly maxTextureSize?: number;
  readonly textureIndex?: number;
  readonly imageIndex?: number;
  readonly samplerIndex?: number;
  readonly imageUri?: string;
  readonly imageMimeType?: string;
  readonly textureSourceExtension?: GltfTextureSourceExtension;
  readonly handleChannels?: 1 | 2 | 3 | 4;
  readonly handleDataType?: 'uint8' | 'uint16' | 'float32';
  /**
   * The decoded payload's own color-space hint when the handle exposes one.
   * This is intentionally separate from `colorSpace`, which describes the
   * glTF/material role. Example: `baseColorMap` has `colorSpace:'srgb'`; after
   * `target:'cpu-linear'` its handleColorSpace is `'linear'`, while after
   * `target:'webgpu'` it remains `'srgb'` for backend sRGB texture upload.
   */
  readonly handleColorSpace?: GltfTextureColorSpace;
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
  readonly imageBitmapCount: number;
  readonly opaqueHandleCount: number;
  readonly cpuReadableCount: number;
  readonly rawImageRefs: readonly GltfTextureDecodeReportEntry[];
  readonly imageBitmapRefs: readonly GltfTextureDecodeReportEntry[];
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
    readonly originalWidth?: number;
    readonly originalHeight?: number;
    readonly maxTextureSize?: number;
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
    readonly originalWidth?: number;
    readonly originalHeight?: number;
    readonly maxTextureSize?: number;
  };
}

export type DecodeGltfTexturePixelsFn = (
  handle: RawImageHandle,
  context: {
    readonly materialField: GltfMaterialTextureField;
    readonly path: string;
    readonly imageSourcePath?: string;
    readonly colorSpace: GltfTextureColorSpace;
    readonly primitiveId: string;
    readonly primitiveIndex: number;
    readonly textureIndex?: number;
    readonly imageIndex?: number;
    readonly samplerIndex?: number;
    readonly imageUri?: string;
    readonly imageMimeType?: string;
    readonly textureSourceExtension?: GltfTextureSourceExtension;
  },
) => Promise<GltfDecodedTexturePixels> | GltfDecodedTexturePixels;

export type DecodeSceneTextureDiagnosticCode =
  | 'unsupported-handle-kind'
  | 'raw-image-decoder-missing'
  | 'decode-pixels-failed'
  | 'decode-pixels-invalid'
  | 'platform-image-decode-failed'
  | 'platform-image-readback-unavailable'
  | 'platform-image-readback-failed'
  | 'decoded-texture-exceeds-max-size'
  | 'decoded-texture-npot-repeat-wrap'
  | 'decoded-texture-npot-repeat-wrap-resized'
  | 'decoded-texture-npot-repeat-wrap-clamped'
  | 'spec-gloss-alpha-bake-unavailable';

export interface DecodeSceneTextureDiagnostic {
  readonly severity: 'warning';
  readonly code: DecodeSceneTextureDiagnosticCode;
  readonly path: string;
  readonly imageSourcePath?: string;
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
  readonly npotRepeatWrapPolicy?: GltfNpotRepeatWrapPolicy;
  readonly textureIndex?: number;
  readonly imageIndex?: number;
  readonly samplerIndex?: number;
  readonly imageUri?: string;
  readonly imageMimeType?: string;
  readonly textureSourceExtension?: GltfTextureSourceExtension;
  readonly causeMessage?: string;
}

export interface DecodeSceneTexturesOptions {
  readonly target: 'cpu-linear' | 'webgpu';
  readonly decodePixels?: DecodeGltfTexturePixelsFn;
  readonly maxTextureSize?: number;
  readonly npotRepeatWrapPolicy?: GltfNpotRepeatWrapPolicy;
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

class PlatformTextureDecodeError extends Error {
  readonly code: Extract<
    DecodeSceneTextureDiagnosticCode,
    'platform-image-decode-failed' | 'platform-image-readback-unavailable' | 'platform-image-readback-failed'
  >;

  constructor(
    code: PlatformTextureDecodeError['code'],
    message: string,
  ) {
    super(message);
    this.name = 'PlatformTextureDecodeError';
    this.code = code;
  }
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
      const handleColorSpace = textureHandleColorSpace(ref.handle);
      const payloadFields = textureHandlePayloadReportFields(ref.handle);
      const samplerFields = textureSamplerReportFields(ref);
      const source = gltfTextureRefSource(ref);
      const dimensionFields = textureDimensionReportFields(ref.handle);
      const scenePath = `scene.primitives[${primitiveIndex}].material.${field}`;
      entries.push({
        primitiveId: String(primitive.id),
        primitiveKind: primitive.kind,
        primitiveIndex,
        materialField: field,
        path: source?.path ?? scenePath,
        ...(source?.imageSourcePath !== undefined ? { imageSourcePath: source.imageSourcePath } : {}),
        texCoord: ref.texCoord ?? 0,
        hasTransform: ref.transform !== undefined,
        wrapS: ref.wrapS ?? 'repeat',
        wrapT: ref.wrapT ?? 'repeat',
        ...samplerFields,
        ...dimensionFields,
        ...(source?.textureIndex !== undefined ? { textureIndex: source.textureIndex } : {}),
        ...(source?.imageIndex !== undefined ? { imageIndex: source.imageIndex } : {}),
        ...(source?.samplerIndex !== undefined ? { samplerIndex: source.samplerIndex } : {}),
        ...(source?.imageUri !== undefined ? { imageUri: source.imageUri } : {}),
        ...(source?.imageMimeType !== undefined ? { imageMimeType: source.imageMimeType } : {}),
        ...(source?.textureSourceExtension !== undefined
          ? { textureSourceExtension: source.textureSourceExtension }
          : {}),
        ...payloadFields,
        ...(handleColorSpace !== undefined ? { handleColorSpace } : {}),
        colorSpace: gltfTextureColorSpaceForField(field),
        handleKind,
        backendReadiness: backendReadinessForHandle(field, handleKind),
      });
    }
  }

  const rawImageRefs = entries.filter((entry) => entry.handleKind === 'raw-image');
  const imageBitmapRefs = entries.filter((entry) => entry.handleKind === 'image-bitmap');
  return {
    mapCount: entries.length,
    uniqueHandleCount: uniqueHandles.size,
    rawImageCount: rawImageRefs.length,
    imageBitmapCount: imageBitmapRefs.length,
    opaqueHandleCount: entries.filter((entry) => entry.handleKind === 'opaque').length,
    cpuReadableCount: entries.filter((entry) =>
      entry.handleKind === 'pixel-data' || entry.handleKind === 'data-texture',
    ).length,
    rawImageRefs,
    imageBitmapRefs,
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

function textureDimensionReportFields(
  handle: unknown,
): Pick<
  GltfTextureDecodeReportEntry,
  'width' | 'height' | 'isPowerOfTwo' | 'originalWidth' | 'originalHeight' | 'wasResized' | 'maxTextureSize'
> {
  const dims = textureHandleDimensions(handle);
  if (dims === null) return {};
  const hint = textureDecodeHint(handle);
  const originalWidth = hint?.originalWidth ?? dims.width;
  const originalHeight = hint?.originalHeight ?? dims.height;
  return {
    width: dims.width,
    height: dims.height,
    isPowerOfTwo: isPowerOfTwo(dims.width, dims.height),
    originalWidth,
    originalHeight,
    wasResized: originalWidth !== dims.width || originalHeight !== dims.height,
    ...(hint?.maxTextureSize !== undefined ? { maxTextureSize: hint.maxTextureSize } : {}),
  };
}

function textureHandleDimensions(handle: unknown): { readonly width: number; readonly height: number } | null {
  if (!isRecord(handle)) return null;
  if (typeof handle.width === 'number' && typeof handle.height === 'number') {
    return { width: handle.width, height: handle.height };
  }
  const image = handle.image;
  if (isRecord(image) && typeof image.width === 'number' && typeof image.height === 'number') {
    return { width: image.width, height: image.height };
  }
  return null;
}

function textureHandlePayloadReportFields(
  handle: unknown,
): Pick<GltfTextureDecodeReportEntry, 'handleChannels' | 'handleDataType'> {
  const hint = textureHandlePayloadHint(handle);
  if (hint !== null) return hint;
  const pixels = decodedPixelsFromCpuReadableHandle(handle);
  if (pixels === null) return {};
  const width = Math.max(0, Math.floor(pixels.width));
  const height = Math.max(0, Math.floor(pixels.height));
  if (width <= 0 || height <= 0 || !isArrayLikeData(pixels.data)) return {};
  return {
    handleChannels: pixels.channels ?? inferDecodedChannels(pixels.data, width, height),
    handleDataType: pixels.dataType ?? inferDecodedDataType(pixels.data),
  };
}

function textureHandlePayloadHint(
  handle: unknown,
): Pick<GltfTextureDecodeReportEntry, 'handleChannels' | 'handleDataType'> | null {
  if (!isRecord(handle)) return null;
  const direct = texturePayloadHintFromRecord(handle);
  if (direct !== null) return direct;
  return isRecord(handle.image) ? texturePayloadHintFromRecord(handle.image, handle) : null;
}

function texturePayloadHintFromRecord(
  record: Record<string, unknown>,
  metadata: Record<string, unknown> = record,
): Pick<GltfTextureDecodeReportEntry, 'handleChannels' | 'handleDataType'> | null {
  const hint = isRecord(metadata.__vitrum_hint__) ? metadata.__vitrum_hint__ : metadata;
  const channels = hint.channels;
  const dataType = hint.dataType;
  const out: { handleChannels?: 1 | 2 | 3 | 4; handleDataType?: 'uint8' | 'uint16' | 'float32' } = {};
  if (channels === 1 || channels === 2 || channels === 3 || channels === 4) out.handleChannels = channels;
  if (dataType === 'uint8' || dataType === 'uint16' || dataType === 'float32') out.handleDataType = dataType;
  return out.handleChannels !== undefined || out.handleDataType !== undefined ? out : null;
}

function textureDecodeHint(handle: unknown): {
  readonly originalWidth?: number;
  readonly originalHeight?: number;
  readonly maxTextureSize?: number;
} | null {
  if (!isRecord(handle) || !isRecord(handle.__vitrum_hint__)) return null;
  const hint = handle.__vitrum_hint__;
  return {
    ...(typeof hint.originalWidth === 'number' ? { originalWidth: hint.originalWidth } : {}),
    ...(typeof hint.originalHeight === 'number' ? { originalHeight: hint.originalHeight } : {}),
    ...(typeof hint.maxTextureSize === 'number' ? { maxTextureSize: hint.maxTextureSize } : {}),
  };
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
    try {
      options.onDiagnostic?.(entry);
    } catch {
      // Host diagnostic callbacks must not abort texture normalization.
    }
    try {
      options.onWarning?.(entry.message);
    } catch {
      // Host warning callbacks must not abort texture normalization.
    }
  };

  const primitives = await Promise.all(scene.primitives.map(async (primitive, primitiveIndex) => {
    const material = materialForPrimitive(primitive);
    let nextMaterial: MaterialSpec | null = null;
    for (const field of MATERIAL_TEXTURE_FIELDS) {
      const ref = material[field] as TextureRef | undefined;
      if (!ref) continue;
      const scenePath = `scene.primitives[${primitiveIndex}].material.${field}`;
      const source = gltfTextureRefSource(ref);
      const path = source?.path ?? scenePath;
      const nextRef = await decodeTextureRef(ref, {
        field,
        path,
        ...(source !== undefined ? { source } : {}),
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

function textureSourceDiagnosticFields(
  source: GltfTextureRefSource | undefined,
): Pick<
  DecodeSceneTextureDiagnostic,
  'imageSourcePath' | 'textureIndex' | 'imageIndex' | 'samplerIndex' | 'imageUri' | 'imageMimeType' | 'textureSourceExtension'
> {
  return {
    ...(source?.imageSourcePath !== undefined ? { imageSourcePath: source.imageSourcePath } : {}),
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

function rawImageDecoderMissingMessage(context: {
  readonly path: string;
  readonly source?: GltfTextureRefSource;
  readonly options: DecodeSceneTexturesOptions;
}): string {
  const extension = context.source?.textureSourceExtension;
  if (extension === 'KHR_texture_basisu' || extension === 'MSFT_texture_dds') {
    return `[vitrum/gltf-adapter] ${context.path} selects ${extension}` +
      `${context.source?.imageMimeType ? ` (${context.source.imageMimeType})` : ''}, ` +
      'but this compressed texture-source extension has no built-in pixel decoder. Supply decodePixels ' +
      `for decodeSceneTextures(target:"${context.options.target}") or choose an asset fallback. Texture left unchanged.`;
  }
  return `[vitrum/gltf-adapter] ${context.path} is a raw-image texture but no decodePixels hook was supplied ` +
    `and this host has no browser image/canvas readback path for decodeSceneTextures(target:"${context.options.target}"). ` +
    'Texture left unchanged.';
}

async function decodeTextureRef(
  ref: TextureRef,
  context: {
    readonly field: GltfMaterialTextureField;
    readonly path: string;
    readonly source?: GltfTextureRefSource;
    readonly primitiveId: string;
    readonly primitiveIndex: number;
    readonly options: DecodeSceneTexturesOptions;
    readonly decoded: Map<unknown, Map<GltfTextureColorSpace, DecodedTextureCacheEntry>>;
    readonly diagnostic: (diagnostic: DecodeSceneTextureDiagnostic) => void;
  },
): Promise<TextureRef> {
  const handleKind = classifyTextureHandle(ref.handle);
  const colorSpace = gltfTextureColorSpaceForField(context.field);
  const outputColorSpace: GltfTextureColorSpace =
    context.options.target === 'webgpu' ? colorSpace : 'linear';
  if (handleKind === 'pixel-data' || handleKind === 'data-texture') {
    const pixels = decodedPixelsFromCpuReadableHandle(ref.handle);
    if (pixels === null) return ref;
    let perSpace = context.decoded.get(ref.handle);
    if (perSpace == null) {
      perSpace = new Map();
      context.decoded.set(ref.handle, perSpace);
    }
    let entry = perSpace.get(colorSpace);
    if (entry == null) {
      entry = cacheEntryFromDecodedPixels(
        pixels,
        colorSpace,
        outputColorSpace,
        context.options.maxTextureSize,
      );
      perSpace.set(colorSpace, entry);
    }
    emitDecodedTextureDiagnostics(entry, ref, context);
    return applyNpotRepeatWrapPolicy(ref, entry, context);
  }
  if (handleKind !== 'raw-image') {
    context.diagnostic({
      severity: 'warning',
      code: 'unsupported-handle-kind',
      path: context.path,
      materialField: context.field,
      primitiveId: context.primitiveId,
      primitiveIndex: context.primitiveIndex,
      handleKind,
      ...textureSourceDiagnosticFields(context.source),
      message: `[vitrum/gltf-adapter] ${context.path} has ${handleKind} texture handle; ` +
        `decodeSceneTextures(target:"${context.options.target}") can only normalize raw-image handles. ` +
        'Texture left unchanged.',
    });
    return ref;
  }
  let decodePixels = context.options.decodePixels;
  if (decodePixels == null) {
    if (canDecodeRawPngPixelsWithNode(ref.handle as RawImageHandle)) {
      decodePixels = decodeRawPngPixelsWithNode;
    } else if (canDecodeRawJpegPixelsWithNode(ref.handle as RawImageHandle)) {
      decodePixels = decodeRawJpegPixelsWithNode;
    } else if (canDecodeRawWebpPixelsWithNode(ref.handle as RawImageHandle)) {
      decodePixels = decodeRawWebpPixelsWithNode;
    } else if (canDecodeRawImagePixelsWithPlatform()) {
      decodePixels = decodeRawImagePixelsWithPlatform;
    }
  }
  if (decodePixels == null) {
    context.diagnostic({
      severity: 'warning',
      code: 'raw-image-decoder-missing',
      path: context.path,
      materialField: context.field,
      primitiveId: context.primitiveId,
      primitiveIndex: context.primitiveIndex,
      handleKind,
      ...textureSourceDiagnosticFields(context.source),
      message: rawImageDecoderMissingMessage(context),
    });
    return ref;
  }

  let perSpace = context.decoded.get(ref.handle);
  if (perSpace == null) {
    perSpace = new Map();
    context.decoded.set(ref.handle, perSpace);
  }
  let entry = perSpace.get(colorSpace);
  if (entry == null) {
    let pixels: GltfDecodedTexturePixels;
    try {
      pixels = await decodePixels(ref.handle as RawImageHandle, {
        materialField: context.field,
        path: context.path,
        colorSpace,
        primitiveId: context.primitiveId,
        primitiveIndex: context.primitiveIndex,
        ...textureSourceDiagnosticFields(context.source),
      });
    } catch (err) {
      if (err instanceof PlatformTextureDecodeError) {
        context.diagnostic({
          severity: 'warning',
          code: err.code,
          path: context.path,
          materialField: context.field,
          primitiveId: context.primitiveId,
          primitiveIndex: context.primitiveIndex,
          handleKind,
          ...textureSourceDiagnosticFields(context.source),
          message: err.message,
        });
        return ref;
      }
      const causeMessage = err instanceof Error ? err.message : String(err);
      context.diagnostic({
        severity: 'warning',
        code: 'decode-pixels-failed',
        path: context.path,
        materialField: context.field,
        primitiveId: context.primitiveId,
        primitiveIndex: context.primitiveIndex,
        handleKind,
        ...textureSourceDiagnosticFields(context.source),
        causeMessage,
        message:
          `[vitrum/gltf-adapter] ${context.path} decodePixels hook failed: ` +
          `${causeMessage}. Texture left unchanged.`,
      });
      return ref;
    }
    const invalid = validateDecodedTexturePixels(pixels);
    if (invalid != null) {
      context.diagnostic({
        severity: 'warning',
        code: 'decode-pixels-invalid',
        path: context.path,
        materialField: context.field,
        primitiveId: context.primitiveId,
        primitiveIndex: context.primitiveIndex,
        handleKind,
        ...(typeof pixels.width === 'number' && Number.isFinite(pixels.width) ? { width: pixels.width } : {}),
        ...(typeof pixels.height === 'number' && Number.isFinite(pixels.height) ? { height: pixels.height } : {}),
        ...textureSourceDiagnosticFields(context.source),
        message:
          `[vitrum/gltf-adapter] ${context.path} decodePixels hook returned invalid pixels: ` +
          `${invalid}. Texture left unchanged.`,
      });
      return ref;
    }
    entry = cacheEntryFromDecodedPixels(
      pixels,
      colorSpace,
      outputColorSpace,
      context.options.maxTextureSize,
    );
    perSpace.set(colorSpace, entry);
  }
  emitDecodedTextureDiagnostics(entry, ref, context);
  return applyNpotRepeatWrapPolicy(ref, entry, context);
}

function validateDecodedTexturePixels(pixels: GltfDecodedTexturePixels): string | null {
  if (typeof pixels.width !== 'number' || !Number.isFinite(pixels.width)) {
    return `width must be a finite number, got ${String(pixels.width)}`;
  }
  if (typeof pixels.height !== 'number' || !Number.isFinite(pixels.height)) {
    return `height must be a finite number, got ${String(pixels.height)}`;
  }
  const width = Math.floor(pixels.width);
  const height = Math.floor(pixels.height);
  if (!Number.isInteger(pixels.width) || !Number.isInteger(pixels.height)) {
    return `dimensions must be integers, got ${pixels.width}x${pixels.height}`;
  }
  if (width <= 0 || height <= 0) {
    return `dimensions must be positive, got ${pixels.width}x${pixels.height}`;
  }
  if (!isArrayLikeData(pixels.data)) {
    return 'data must be an array-like pixel payload';
  }
  if (
    pixels.channels !== undefined &&
    pixels.channels !== 1 &&
    pixels.channels !== 2 &&
    pixels.channels !== 3 &&
    pixels.channels !== 4
  ) {
    return `channels must be 1, 2, 3, or 4, got ${String(pixels.channels)}`;
  }
  if (
    pixels.dataType !== undefined &&
    pixels.dataType !== 'uint8' &&
    pixels.dataType !== 'uint16' &&
    pixels.dataType !== 'float32'
  ) {
    return `dataType must be uint8, uint16, or float32, got ${String(pixels.dataType)}`;
  }
  const channels = pixels.channels ?? inferDecodedChannels(pixels.data, width, height);
  const requiredLength = width * height * channels;
  if (pixels.data.length < requiredLength) {
    return `data length ${pixels.data.length} is too short for ${width}x${height}x${channels}; ` +
      `expected at least ${requiredLength}`;
  }
  if (
    pixels.colorSpace !== undefined &&
    pixels.colorSpace !== 'srgb' &&
    pixels.colorSpace !== 'linear'
  ) {
    return `colorSpace must be srgb or linear, got ${String(pixels.colorSpace)}`;
  }
  return null;
}

function cacheEntryFromDecodedPixels(
  pixels: GltfDecodedTexturePixels,
  colorSpace: GltfTextureColorSpace,
  outputColorSpace: GltfTextureColorSpace,
  maxTextureSize: number | undefined,
): DecodedTextureCacheEntry {
  const normalized = normalizeDecodedPixels(pixels, colorSpace, outputColorSpace);
  const resized = resizeDecodedTextureToMaxSize(normalized, maxTextureSize);
  const shouldAnnotate =
    resized.width !== normalized.width ||
    resized.height !== normalized.height ||
    (typeof maxTextureSize === 'number' && maxTextureSize > 0);
  return {
    handle: shouldAnnotate
      ? withDecodedTextureMetadata(
          resized,
          normalized.width,
          normalized.height,
          maxTextureSize,
        )
      : resized,
    originalWidth: normalized.width,
    originalHeight: normalized.height,
  };
}

const decodeRawImagePixelsWithPlatform: DecodeGltfTexturePixelsFn = async (handle, context) => {
  const bitmap = await createBitmapFromRawImage(handle, context.path);
  try {
    const width = Math.max(0, Math.floor(numberProp(bitmap, 'width') ?? 0));
    const height = Math.max(0, Math.floor(numberProp(bitmap, 'height') ?? 0));
    if (width <= 0 || height <= 0) {
      throw new PlatformTextureDecodeError(
        'platform-image-decode-failed',
        `[vitrum/gltf-adapter] ${context.path} decoded to invalid image dimensions ${width}x${height}. Texture left unchanged.`,
      );
    }
    const ctx = createReadback2dContext(width, height, context.path);
    try {
      ctx.drawImage(bitmap, 0, 0, width, height);
      const imageData = ctx.getImageData(0, 0, width, height);
      return {
        width,
        height,
        data: imageData.data,
        channels: 4,
        dataType: 'uint8',
        colorSpace: context.colorSpace,
      };
    } catch (err) {
      throw new PlatformTextureDecodeError(
        'platform-image-readback-failed',
        `[vitrum/gltf-adapter] ${context.path} decoded through browser image APIs, but canvas pixel readback failed: ` +
          `${err instanceof Error ? err.message : String(err)}. Texture left unchanged.`,
      );
    }
  } finally {
    closeBitmap(bitmap);
  }
};

function canDecodeRawImagePixelsWithPlatform(): boolean {
  return typeof globalThis.createImageBitmap === 'function' && typeof globalThis.Blob === 'function';
}

function canDecodeRawPngPixelsWithNode(handle: RawImageHandle): boolean {
  return isNodeLikeHost() && isPngRawImageHandle(handle);
}

function canDecodeRawJpegPixelsWithNode(handle: RawImageHandle): boolean {
  return isNodeLikeHost() && isJpegRawImageHandle(handle);
}

function canDecodeRawWebpPixelsWithNode(handle: RawImageHandle): boolean {
  return isNodeLikeHost() && isWebpRawImageHandle(handle);
}

function isNodeLikeHost(): boolean {
  const host = globalThis as typeof globalThis & {
    process?: { versions?: { node?: unknown } };
  };
  return typeof host.process?.versions?.node === 'string';
}

function isPngRawImageHandle(handle: RawImageHandle): boolean {
  const data = handle.data;
  return data.length >= 8 &&
    data[0] === 0x89 &&
    data[1] === 0x50 &&
    data[2] === 0x4e &&
    data[3] === 0x47 &&
    data[4] === 0x0d &&
    data[5] === 0x0a &&
    data[6] === 0x1a &&
    data[7] === 0x0a;
}

function isJpegRawImageHandle(handle: RawImageHandle): boolean {
  const data = handle.data;
  const mimeType = handle.mimeType.toLowerCase();
  return (mimeType === 'image/jpeg' || mimeType === 'image/jpg') ||
    (data.length >= 3 && data[0] === 0xff && data[1] === 0xd8 && data[2] === 0xff);
}

function isWebpRawImageHandle(handle: RawImageHandle): boolean {
  const data = handle.data;
  const mimeType = handle.mimeType.toLowerCase();
  return mimeType === 'image/webp' ||
    (data.length >= 12 &&
      data[0] === 0x52 &&
      data[1] === 0x49 &&
      data[2] === 0x46 &&
      data[3] === 0x46 &&
      data[8] === 0x57 &&
      data[9] === 0x45 &&
      data[10] === 0x42 &&
      data[11] === 0x50);
}

const decodeRawPngPixelsWithNode: DecodeGltfTexturePixelsFn = async (handle, context) => {
  try {
    const { PNG } = await importPngJs();
    const bytes = handle.data;
    const decoded = PNG.sync.read(nodeBufferFromUint8Array(bytes));
    return {
      width: decoded.width,
      height: decoded.height,
      data: decoded.data,
      channels: 4,
      dataType: 'uint8',
      colorSpace: context.colorSpace,
    };
  } catch (err) {
    throw new PlatformTextureDecodeError(
      'platform-image-decode-failed',
      `[vitrum/gltf-adapter] ${context.path} could not be decoded as PNG through the built-in Node decoder: ` +
        `${err instanceof Error ? err.message : String(err)}. Texture left unchanged.`,
    );
  }
};

const decodeRawJpegPixelsWithNode: DecodeGltfTexturePixelsFn = async (handle, context) => {
  try {
    const jpeg = await importJpegJs();
    const decode = jpegDecodeFn(jpeg);
    const decoded = decode(nodeBufferFromUint8Array(handle.data), { useTArray: true });
    return {
      width: decoded.width,
      height: decoded.height,
      data: decoded.data,
      channels: 4,
      dataType: 'uint8',
      colorSpace: context.colorSpace,
    };
  } catch (err) {
    throw new PlatformTextureDecodeError(
      'platform-image-decode-failed',
      `[vitrum/gltf-adapter] ${context.path} could not be decoded as JPEG through the built-in Node decoder: ` +
        `${err instanceof Error ? err.message : String(err)}. Texture left unchanged.`,
    );
  }
};

const decodeRawWebpPixelsWithNode: DecodeGltfTexturePixelsFn = async (handle, context) => {
  try {
    const webp = await importWebpWasm();
    const decode = webpDecodeFn(webp);
    const decoded = await decode(arrayBufferFromUint8Array(handle.data));
    return {
      width: decoded.width,
      height: decoded.height,
      data: decoded.data,
      channels: 4,
      dataType: 'uint8',
      colorSpace: context.colorSpace,
    };
  } catch (err) {
    throw new PlatformTextureDecodeError(
      'platform-image-decode-failed',
      `[vitrum/gltf-adapter] ${context.path} could not be decoded as WebP through the built-in Node decoder: ` +
        `${err instanceof Error ? err.message : String(err)}. Texture left unchanged.`,
    );
  }
};

interface PngJsSyncReader {
  read(data: unknown): {
    readonly width: number;
    readonly height: number;
    readonly data: Uint8Array;
  };
}

interface PngJsModule {
  readonly PNG: {
    readonly sync: PngJsSyncReader;
  };
}

async function importPngJs(): Promise<PngJsModule> {
  const specifier = 'pngjs';
  return await import(/* @vite-ignore */ specifier) as PngJsModule;
}

interface JpegJsDecodedImage {
  readonly width: number;
  readonly height: number;
  readonly data: Uint8Array | Uint8ClampedArray;
}

type JpegJsDecodeFn = (
  data: unknown,
  options?: { readonly useTArray?: boolean },
) => JpegJsDecodedImage;

interface JpegJsModule {
  readonly decode?: JpegJsDecodeFn;
  readonly default?: {
    readonly decode?: JpegJsDecodeFn;
  };
}

async function importJpegJs(): Promise<JpegJsModule> {
  const specifier = 'jpeg-js';
  return await import(/* @vite-ignore */ specifier) as JpegJsModule;
}

function jpegDecodeFn(module: JpegJsModule): JpegJsDecodeFn {
  const decode = module.decode ?? module.default?.decode;
  if (typeof decode !== 'function') {
    throw new Error('jpeg-js decode export is unavailable');
  }
  return decode;
}

interface WebpWasmDecodedImage {
  readonly width: number;
  readonly height: number;
  readonly data: Uint8Array | Uint8ClampedArray;
}

type WebpWasmDecodeFn = (data: unknown) => Promise<WebpWasmDecodedImage> | WebpWasmDecodedImage;

interface WebpWasmModule {
  readonly decode?: WebpWasmDecodeFn;
  readonly default?: {
    readonly decode?: WebpWasmDecodeFn;
  };
}

async function importWebpWasm(): Promise<WebpWasmModule> {
  const specifier = 'webp-wasm';
  return await import(/* @vite-ignore */ specifier) as WebpWasmModule;
}

function webpDecodeFn(module: WebpWasmModule): WebpWasmDecodeFn {
  const owner = module.decode !== undefined ? module : module.default;
  const decode = owner?.decode;
  if (typeof decode !== 'function') {
    throw new Error('webp-wasm decode export is unavailable');
  }
  return (data) => decode.call(owner, data);
}

function nodeBufferFromUint8Array(bytes: Uint8Array): unknown {
  const host = globalThis as typeof globalThis & {
    Buffer?: {
      from(buffer: ArrayBufferLike, byteOffset?: number, length?: number): unknown;
    };
  };
  if (host.Buffer == null) {
    throw new Error('Node Buffer is unavailable');
  }
  return host.Buffer.from(bytes.buffer, bytes.byteOffset, bytes.byteLength);
}

function arrayBufferFromUint8Array(bytes: Uint8Array): ArrayBuffer {
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer;
}

async function createBitmapFromRawImage(handle: RawImageHandle, path: string): Promise<unknown> {
  try {
    const bytes = handle.data;
    const slice = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
    const blob = new Blob([slice as ArrayBuffer], { type: handle.mimeType });
    return await createImageBitmap(blob);
  } catch (err) {
    throw new PlatformTextureDecodeError(
      'platform-image-decode-failed',
      `[vitrum/gltf-adapter] ${path} could not be decoded through browser image APIs: ` +
        `${err instanceof Error ? err.message : String(err)}. Texture left unchanged.`,
    );
  }
}

interface Canvas2dReadbackContext {
  drawImage(image: unknown, dx: number, dy: number, dw: number, dh: number): void;
  getImageData(sx: number, sy: number, sw: number, sh: number): { readonly data: Uint8ClampedArray };
}

function createReadback2dContext(
  width: number,
  height: number,
  path: string,
): Canvas2dReadbackContext {
  const host = globalThis as typeof globalThis & {
    OffscreenCanvas?: new (width: number, height: number) => { getContext(type: '2d'): unknown };
    document?: { createElement(tag: 'canvas'): { width: number; height: number; getContext(type: '2d'): unknown } };
  };
  const canvas = typeof host.OffscreenCanvas === 'function'
    ? new host.OffscreenCanvas(width, height)
    : host.document?.createElement('canvas');
  if (canvas == null) {
    throw new PlatformTextureDecodeError(
      'platform-image-readback-unavailable',
      `[vitrum/gltf-adapter] ${path} decoded through browser image APIs, but no OffscreenCanvas/document canvas ` +
        'is available for pixel readback. Texture left unchanged.',
    );
  }
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  if (!isCanvas2dReadbackContext(ctx)) {
    throw new PlatformTextureDecodeError(
      'platform-image-readback-unavailable',
      `[vitrum/gltf-adapter] ${path} decoded through browser image APIs, but a 2D canvas readback context ` +
        'could not be created. Texture left unchanged.',
    );
  }
  return ctx;
}

function isCanvas2dReadbackContext(value: unknown): value is Canvas2dReadbackContext {
  return isRecord(value) &&
    typeof value.drawImage === 'function' &&
    typeof value.getImageData === 'function';
}

function numberProp(value: unknown, key: string): number | undefined {
  return isRecord(value) && typeof value[key] === 'number' ? value[key] : undefined;
}

function closeBitmap(bitmap: unknown): void {
  if (isRecord(bitmap) && typeof bitmap.close === 'function') bitmap.close();
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
  const specGloss = material.extensions?.KHR_materials_pbrSpecularGlossiness;
  if (!isRecord(specGloss) || !isRecord(specGloss.specularGlossinessTexture)) return null;
  const sourceRef = material.specularColorMap;
  if (sourceRef == null) return null;
  const source = gltfTextureRefSource(sourceRef);
  const path = source?.path ??
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
    ...textureSourceDiagnosticFields(source),
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
    __vitrum_hint__: {
      channels: 4,
      dataType: 'float32',
      colorSpace: 'linear',
      ...(textureDecodeHint(cacheKey) ?? {}),
      ...(textureDecodeHint(source) ?? {}),
    },
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
    readonly source?: GltfTextureRefSource;
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
      ...textureSourceDiagnosticFields(context.source),
      message: `[vitrum/gltf-adapter] ${context.path} decoded to ${entry.originalWidth}x${entry.originalHeight}, ` +
        `which exceeds maxTextureSize=${maxTextureSize}. Texture was resized to ${handle.width}x${handle.height} ` +
        `during ${context.options.target} decode before backend upload.`,
    });
  }

  const npotPolicy = effectiveNpotRepeatWrapPolicy(context.options);
  if (npotPolicy === 'warn' && !isPowerOfTwo(handle.width, handle.height) && usesRepeatWrap(ref)) {
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
      npotRepeatWrapPolicy: npotPolicy,
      ...textureSourceDiagnosticFields(context.source),
      message: `[vitrum/gltf-adapter] ${context.path} decoded to NPOT ${handle.width}x${handle.height} ` +
        `with wrapS=${wrapS} wrapT=${wrapT}. WebGL2/WebGPU can sample NPOT textures, but exact mip/border ` +
        'parity depends on backend upload policy; pre-resize to power-of-two if this asset needs strict parity.',
    });
  }
}

function effectiveNpotRepeatWrapPolicy(
  options: DecodeSceneTexturesOptions,
): GltfNpotRepeatWrapPolicy | 'none' {
  if (options.npotRepeatWrapPolicy !== undefined) return options.npotRepeatWrapPolicy;
  return options.warnOnNpotRepeatWrap === true ? 'warn' : 'none';
}

function applyNpotRepeatWrapPolicy(
  ref: TextureRef,
  entry: DecodedTextureCacheEntry,
  context: {
    readonly field: GltfMaterialTextureField;
    readonly path: string;
    readonly source?: GltfTextureRefSource;
    readonly primitiveId: string;
    readonly primitiveIndex: number;
    readonly options: DecodeSceneTexturesOptions;
    readonly diagnostic: (diagnostic: DecodeSceneTextureDiagnostic) => void;
  },
): TextureRef {
  const policy = effectiveNpotRepeatWrapPolicy(context.options);
  const handle = entry.handle;
  if (policy === 'none' || policy === 'warn' || isPowerOfTwo(handle.width, handle.height) || !usesRepeatWrap(ref)) {
    return { ...ref, handle };
  }

  const wrapS = ref.wrapS ?? 'repeat';
  const wrapT = ref.wrapT ?? 'repeat';
  if (policy === 'clamp-sampler') {
    context.diagnostic({
      severity: 'warning',
      code: 'decoded-texture-npot-repeat-wrap-clamped',
      path: context.path,
      materialField: context.field,
      primitiveId: context.primitiveId,
      primitiveIndex: context.primitiveIndex,
      width: handle.width,
      height: handle.height,
      wrapS,
      wrapT,
      npotRepeatWrapPolicy: policy,
      ...textureSourceDiagnosticFields(context.source),
      message: `[vitrum/gltf-adapter] ${context.path} decoded to NPOT ${handle.width}x${handle.height} ` +
        `with wrapS=${wrapS} wrapT=${wrapT}. Sampler wrap was clamped to clamp-to-edge by ` +
        `npotRepeatWrapPolicy:"${policy}".`,
    });
    return {
      ...ref,
      handle,
      wrapS: 'clamp-to-edge',
      wrapT: 'clamp-to-edge',
    };
  }

  const resized = resizeDecodedTextureToPowerOfTwo(handle, context.options.maxTextureSize);
  context.diagnostic({
    severity: 'warning',
    code: 'decoded-texture-npot-repeat-wrap-resized',
    path: context.path,
    materialField: context.field,
    primitiveId: context.primitiveId,
    primitiveIndex: context.primitiveIndex,
    width: handle.width,
    height: handle.height,
    resizedWidth: resized.width,
    resizedHeight: resized.height,
    wrapS,
    wrapT,
    npotRepeatWrapPolicy: policy,
    ...textureSourceDiagnosticFields(context.source),
    message: `[vitrum/gltf-adapter] ${context.path} decoded to NPOT ${handle.width}x${handle.height} ` +
      `with wrapS=${wrapS} wrapT=${wrapT}. Texture was resized to ${resized.width}x${resized.height} ` +
      `by npotRepeatWrapPolicy:"${policy}" for deterministic repeat-wrap sampling.`,
  });
  return { ...ref, handle: resized };
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
  if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
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
  return resizeDecodedTextureNearest(handle, width, height);
}

function resizeDecodedTextureToPowerOfTwo(
  handle: GltfCpuTextureHandle,
  maxTextureSize: number | undefined,
): GltfCpuTextureHandle {
  const width = powerOfTwoTarget(handle.width, maxTextureSize);
  const height = powerOfTwoTarget(handle.height, maxTextureSize);
  if (width === handle.width && height === handle.height) return handle;
  const resized = resizeDecodedTextureNearest(handle, width, height);
  return withDecodedTextureMetadata(
    resized,
    textureDecodeHint(handle)?.originalWidth ?? handle.width,
    textureDecodeHint(handle)?.originalHeight ?? handle.height,
    maxTextureSize,
  );
}

function powerOfTwoTarget(value: number, maxTextureSize: number | undefined): number {
  if (isSinglePowerOfTwo(value)) return value;
  const ceil = 2 ** Math.ceil(Math.log2(Math.max(1, value)));
  if (typeof maxTextureSize !== 'number' || maxTextureSize <= 0 || ceil <= maxTextureSize) {
    return Math.max(1, ceil);
  }
  return Math.max(1, 2 ** Math.floor(Math.log2(maxTextureSize)));
}

function resizeDecodedTextureNearest(
  handle: GltfCpuTextureHandle,
  width: number,
  height: number,
): GltfCpuTextureHandle {
  if (width === handle.width && height === handle.height) return handle;
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

function withDecodedTextureMetadata<T extends GltfCpuTextureHandle>(
  handle: T,
  originalWidth: number,
  originalHeight: number,
  maxTextureSize: number | undefined,
): T {
  return {
    ...handle,
    __vitrum_hint__: {
      ...handle.__vitrum_hint__,
      originalWidth,
      originalHeight,
      ...(typeof maxTextureSize === 'number' && maxTextureSize > 0 ? { maxTextureSize } : {}),
    },
  } as T;
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

function textureHandleColorSpace(handle: unknown): GltfTextureColorSpace | undefined {
  if (!isRecord(handle)) return undefined;
  const direct = handle.colorSpace;
  if (direct === 'srgb' || direct === 'linear') return direct;
  const hint = handle.__vitrum_hint__;
  if (isRecord(hint) && (hint.colorSpace === 'srgb' || hint.colorSpace === 'linear')) {
    return hint.colorSpace;
  }
  const image = handle.image;
  if (isRecord(image)) {
    const imageColorSpace = image.colorSpace;
    if (imageColorSpace === 'srgb' || imageColorSpace === 'linear') return imageColorSpace;
    const imageHint = image.__vitrum_hint__;
    if (isRecord(imageHint) && (imageHint.colorSpace === 'srgb' || imageHint.colorSpace === 'linear')) {
      return imageHint.colorSpace;
    }
  }
  return undefined;
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
