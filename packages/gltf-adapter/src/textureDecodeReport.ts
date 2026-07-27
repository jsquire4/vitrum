// textureDecodeReport.ts — builds the scene-level GltfTextureDecodeReport (D15-6).
// Extracted verbatim from texturePipeline.ts. Reads decoded texture handles off a
// core Scene and classifies them; the low-level handle predicates + dimension/
// payload inference helpers stay in texturePipeline.ts and are imported here (the
// texturePipeline→buildTextureDecodeReport back-reference is a function-level ESM
// cycle only, never evaluated at module init, so it is safe).

import type { Scene, TextureRef } from '@vitrum/core';
import { gltfTextureRefSource } from './textures.js';
import {
  MATERIAL_TEXTURE_FIELDS,
  backendReadinessForHandle,
  classifyTextureHandle,
  decodedPixelsFromCpuReadableHandle,
  gltfTextureColorSpaceForField,
  inferDecodedChannels,
  inferDecodedDataType,
  isArrayLikeData,
  isCpuReadableTexturePayloadValid,
  isPowerOfTwo,
  isRecord,
  materialForPrimitive,
  textureDecodeHint,
  textureHandleColorSpace,
  type GltfTextureDecodeReport,
  type GltfTextureDecodeReportEntry,
} from './texturePipeline.js';

export function buildTextureDecodeReport(scene: Scene): GltfTextureDecodeReport {
  const entries: GltfTextureDecodeReportEntry[] = [];
  const uniqueHandles = new Set<unknown>();
  for (const [primitiveIndex, primitive] of scene.primitives.entries()) {
    const material = materialForPrimitive(primitive);
    for (const field of MATERIAL_TEXTURE_FIELDS) {
      const ref = material[field];
      if (!ref) continue;
      uniqueHandles.add(ref.handle);
      const classifiedHandleKind = classifyTextureHandle(ref.handle);
      const handleKind =
        (classifiedHandleKind === 'pixel-data' || classifiedHandleKind === 'data-texture') &&
        !isCpuReadableTexturePayloadValid(ref.handle)
          ? 'opaque'
          : classifiedHandleKind;
      const handleColorSpace = textureHandleColorSpace(ref.handle);
      const payloadFields = handleKind === 'opaque'
        ? {}
        : textureHandlePayloadReportFields(ref.handle);
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
  try {
    if (!isRecord(handle)) return null;
    const width = handle.width;
    const height = handle.height;
    if (isPositiveSafeDimension(width) && isPositiveSafeDimension(height)) {
      return { width, height };
    }
    const image = handle.image;
    if (isRecord(image)) {
      const imageWidth = image.width;
      const imageHeight = image.height;
      if (isPositiveSafeDimension(imageWidth) && isPositiveSafeDimension(imageHeight)) {
        return { width: imageWidth, height: imageHeight };
      }
    }
  } catch {
    return null;
  }
  return null;
}

function isPositiveSafeDimension(value: unknown): value is number {
  return Number.isSafeInteger(value) && (value as number) > 0;
}

function textureHandlePayloadReportFields(
  handle: unknown,
): Pick<GltfTextureDecodeReportEntry, 'handleChannels' | 'handleDataType'> {
  try {
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
  } catch {
    return {};
  }
}

function textureHandlePayloadHint(
  handle: unknown,
): Pick<GltfTextureDecodeReportEntry, 'handleChannels' | 'handleDataType'> | null {
  try {
    if (!isRecord(handle)) return null;
    const direct = texturePayloadHintFromRecord(handle);
    if (direct !== null) return direct;
    const image = handle.image;
    return isRecord(image) ? texturePayloadHintFromRecord(image, handle) : null;
  } catch {
    return null;
  }
}

function texturePayloadHintFromRecord(
  record: Record<string, unknown>,
  metadata: Record<string, unknown> = record,
): Pick<GltfTextureDecodeReportEntry, 'handleChannels' | 'handleDataType'> | null {
  try {
    const metadataHint = metadata.__vitrum_hint__;
    const hint = isRecord(metadataHint) ? metadataHint : metadata;
    const channels = hint.channels;
    const dataType = hint.dataType;
    const out: { handleChannels?: 1 | 2 | 3 | 4; handleDataType?: 'uint8' | 'uint16' | 'float32' } = {};
    if (channels === 1 || channels === 2 || channels === 3 || channels === 4) out.handleChannels = channels;
    if (dataType === 'uint8' || dataType === 'uint16' || dataType === 'float32') out.handleDataType = dataType;
    return out.handleChannels !== undefined || out.handleDataType !== undefined ? out : null;
  } catch {
    return null;
  }
}
