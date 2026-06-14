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
  | 'decoded-texture-npot-repeat-wrap';

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
  'roughnessMap',
  'metallicMap',
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

export async function decodeSceneTextures(
  scene: Scene,
  options: DecodeSceneTexturesOptions,
): Promise<DecodeSceneTexturesResult> {
  const warnings: string[] = [];
  const diagnostics: DecodeSceneTextureDiagnostic[] = [];
  const decoded = new Map<unknown, Map<GltfTextureColorSpace, GltfCpuLinearTextureHandle>>();
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
      const path = `scene.primitives[${primitiveIndex}].material.${field}`;
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
    readonly decoded: Map<unknown, Map<GltfTextureColorSpace, GltfCpuLinearTextureHandle>>;
    readonly diagnostic: (diagnostic: DecodeSceneTextureDiagnostic) => void;
  },
): Promise<TextureRef> {
  if (context.options.target === 'webgpu') return ref;
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
  let perSpace = context.decoded.get(ref.handle);
  if (perSpace == null) {
    perSpace = new Map();
    context.decoded.set(ref.handle, perSpace);
  }
  let handle = perSpace.get(colorSpace);
  if (handle == null) {
    const pixels = await context.options.decodePixels(ref.handle as RawImageHandle, {
      materialField: context.field,
      path: context.path,
      colorSpace,
      primitiveId: context.primitiveId,
      primitiveIndex: context.primitiveIndex,
    });
    handle = normalizeDecodedPixels(pixels, colorSpace);
    perSpace.set(colorSpace, handle);
  }
  emitDecodedTextureDiagnostics(handle, ref, context);
  return { ...ref, handle };
}

function emitDecodedTextureDiagnostics(
  handle: GltfCpuLinearTextureHandle,
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
  const maxTextureSize = context.options.maxTextureSize;
  if (typeof maxTextureSize === 'number' && maxTextureSize > 0 &&
      (handle.width > maxTextureSize || handle.height > maxTextureSize)) {
    context.diagnostic({
      severity: 'warning',
      code: 'decoded-texture-exceeds-max-size',
      path: context.path,
      materialField: context.field,
      primitiveId: context.primitiveId,
      primitiveIndex: context.primitiveIndex,
      width: handle.width,
      height: handle.height,
      maxTextureSize,
      message: `[vitrum/gltf-adapter] ${context.path} decoded to ${handle.width}x${handle.height}, ` +
        `which exceeds maxTextureSize=${maxTextureSize}. Texture left at decoded size; downsample before upload ` +
        'or choose a backend/device that can accept it.',
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
): GltfCpuLinearTextureHandle {
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
    const convert = fieldColorSpace === 'srgb' && sourceColorSpace !== 'linear';
    out[p * 4] = convert ? srgbToLinear(r) : r;
    out[p * 4 + 1] = convert ? srgbToLinear(g) : g;
    out[p * 4 + 2] = convert ? srgbToLinear(b) : b;
    out[p * 4 + 3] = a;
  }
  return {
    width,
    height,
    data: out,
    __vitrum_hint__: { channels: 4, dataType: 'float32', colorSpace: 'linear' },
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

function decodeChannel(value: number, dataType: 'uint8' | 'uint16' | 'float32'): number {
  if (dataType === 'uint8') return Math.max(0, Math.min(1, value / 255));
  if (dataType === 'uint16') return Math.max(0, Math.min(1, value / 65535));
  return Number(value);
}

function srgbToLinear(v: number): number {
  const c = Math.max(0, Math.min(1, v));
  return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
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
