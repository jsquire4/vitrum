// texturePipeline.ts — scene-level diagnostics for decoded glTF texture handles.

import type { MaterialSpec, Scene, ScenePrimitive, TextureRef, TextureWrapMode } from '@vitrum/core';
import type { RawImageHandle } from './textures.js';

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
      entries.push({
        primitiveId: String(primitive.id),
        primitiveKind: primitive.kind,
        primitiveIndex,
        materialField: field,
        path: `scene.primitives[${primitiveIndex}].material.${field}`,
        texCoord: ref.texCoord ?? 0,
        hasTransform: ref.transform !== undefined,
        wrapS: ref.wrapS ?? 'repeat',
        wrapT: ref.wrapT ?? 'repeat',
        colorSpace: gltfTextureColorSpaceForField(field),
        handleKind,
        backendReadiness: backendReadinessForHandle(handleKind),
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
  handleKind: GltfTextureHandleKind,
): GltfTextureDecodeReportEntry['backendReadiness'] {
  const cpuReady = handleKind === 'pixel-data' || handleKind === 'data-texture';
  return {
    ptWebgl2: cpuReady ? 'ready' : 'opaque',
    ptWebgpu: handleKind === 'opaque' || handleKind === 'raw-image' ? 'opaque' : 'ready',
    walkaroundHybrid: 'ignored',
  };
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
