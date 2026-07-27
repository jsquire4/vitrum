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
  attachGltfTextureRefSource,
  gltfTextureRefSource,
  type GltfTextureRefSource,
  type GltfTextureSourceExtension,
  type RawImageHandle,
} from './textures.js';
import {
  PlatformTextureDecodeError,
  canDecodeRawImagePixelsWithPlatform,
  canDecodeRawJpegPixelsDeterministically,
  canDecodeRawPngPixelsDeterministically,
  canDecodeRawWebpPixelsWithNode,
  decodeRawImagePixelsWithPlatform,
  decodeRawJpegPixelsDeterministically,
  decodeRawPngPixelsDeterministically,
  decodeRawWebpPixelsWithNode,
  normalizeCapturedRawImageForDecode,
} from './textureCodecs.js';
import { buildTextureDecodeReport } from './textureDecodeReport.js';
import {
  ImportResourceLedger,
  GltfResourceLimitError,
  createAsyncResourceLimiter,
  gltfImageResourceKey,
  normalizeGltfImportResourceLimits,
  type AsyncResourceLimiter,
  type GltfImportResourceContext,
  type GltfImportResourceLimits,
  type NormalizedGltfImportResourceLimits,
} from './importResourceBudget.js';
import { inspectIntrinsicTypedArray } from './intrinsicTypedArrays.js';

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
  /** Pixel channel layout. One channel is luminance, two channels are
   * luminance-alpha, three are RGB, and four are RGBA. */
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
    /**
     * Effective per-image decoded-pixel ceiling. Built-in decoders enforce it
     * from encoded headers before allocating; custom decoders should do the
     * same. `0` explicitly disables the ceiling.
     */
    readonly maxDecodedTexturePixels: number;
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
  | 'encoded-texture-exceeds-byte-budget'
  | 'encoded-textures-exceed-total-byte-budget'
  | 'decoded-texture-exceeds-max-size'
  | 'decoded-texture-exceeds-pixel-budget'
  | 'decoded-texture-exceeds-total-pixel-budget'
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
  readonly encodedTextureBytes?: number;
  readonly maxEncodedTextureBytes?: number;
  readonly totalEncodedTextureBytes?: number;
  readonly maxTotalEncodedTextureBytes?: number;
  readonly maxDecodedTexturePixels?: number;
  readonly maxTotalDecodedTexturePixels?: number;
  readonly totalDecodedTexturePixels?: number;
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
  /** Shared import resource limits. Flat texture options below take precedence. */
  readonly resourceLimits?: GltfImportResourceLimits;
  /**
   * Hard ceiling on the number of pixels (width × height) a decoded texture may
   * have. A decoded texture exceeding this budget is REJECTED before the
   * full-resolution Float32Array is allocated (the texture is left unchanged and
   * a `decoded-texture-exceeds-pixel-budget` diagnostic is emitted), guarding
   * against unbounded allocation from a hostile/huge asset. `maxTextureSize`
   * clamps dimensions of accepted textures; this budget rejects outright.
   * `0` explicitly disables this ceiling. Undefined uses the safe default.
   */
  readonly maxDecodedTexturePixels?: number;
  /**
   * Aggregate pixel budget for one import. A shared high-level import ledger
   * already includes decoded surfaces accepted during image acquisition; this
   * stage additionally charges every adapter-owned normalization, spec-gloss
   * bake, and NPOT/POT resize output. A standalone call has no upstream
   * acquisition charge and therefore accounts for the outputs it creates.
   * `0` explicitly disables this ceiling.
   */
  readonly maxTotalDecodedTexturePixels?: number;
  /**
   * Maximum simultaneously active raw-image decode hooks. Undefined uses
   * `resourceLimits.maxConcurrentResourceOperations`, then the safe default (4).
   */
  readonly maxImageDecodeConcurrency?: number;
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

type SpecGlossRoughnessBakeCache = Map<unknown, Map<number, GltfCpuLinearTextureHandle>>;

type DecodedTextureCacheOutcome =
  | {
    readonly kind: 'decoded';
    readonly entry: DecodedTextureCacheEntry;
  }
  | {
    readonly kind: 'platform-error';
    readonly error: PlatformTextureDecodeError;
  }
  | {
    readonly kind: 'decode-error' | 'normalization-error';
    readonly causeMessage: string;
  }
  | {
    readonly kind: 'decoder-missing';
  }
  | {
    readonly kind: 'invalid';
    readonly reason: string;
    readonly width?: number;
    readonly height?: number;
  }
  | {
    readonly kind: 'resource-limit';
    readonly error: GltfResourceLimitError;
    readonly width?: number;
    readonly height?: number;
  };

type DecodedTexturePromiseCache = Map<
  unknown,
  Map<GltfTextureColorSpace, Promise<DecodedTextureCacheOutcome>>
>;

type RawImageSnapshotOutcome =
  | { readonly kind: 'ready'; readonly handle: RawImageHandle }
  | { readonly kind: 'platform-error'; readonly error: PlatformTextureDecodeError }
  | { readonly kind: 'resource-limit'; readonly error: GltfResourceLimitError };

/**
 * Stateful decode scope for one import. Reuse the same context for the active
 * scene and any converted/inactive material pass so caches, aggregate budgets,
 * and concurrency limits cover the whole import.
 *
 * @internal
 */
export interface DecodeSceneTexturesContext {
  readonly options: DecodeSceneTexturesOptions;
  readonly limits: Readonly<NormalizedGltfImportResourceLimits>;
  /** One import-wide ledger, including any post-policy limit overrides. */
  readonly resourceLedger: ImportResourceLedger;
  /** Alias retained for the raw-image snapshot path. */
  readonly encodedResourceLedger: ImportResourceLedger;
  readonly imageDecodeLimiter: AsyncResourceLimiter;
  readonly decoded: DecodedTexturePromiseCache;
  readonly specGlossBakes: SpecGlossRoughnessBakeCache;
  readonly rawImageSnapshots: Map<unknown, Promise<RawImageSnapshotOutcome>>;
  readonly rawImageResourceKeys: Map<unknown, string>;
}

export const MATERIAL_TEXTURE_FIELDS: readonly GltfMaterialTextureField[] = [
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

/** @internal */
export function createDecodeSceneTexturesContext(
  options: DecodeSceneTexturesOptions,
  resourceContext?: GltfImportResourceContext,
): DecodeSceneTexturesContext {
  validateDecodeSceneTexturesOptions(options);
  const limits = effectiveTextureResourceLimits(options, resourceContext);
  const resourceLedger = resourceContext?.ledger ?? new ImportResourceLedger(limits);
  resourceLedger.reconfigureLimits(limits);
  const imageDecodeLimiter =
    resourceContext?.limiter.maxConcurrency === limits.maxConcurrentResourceOperations
      ? resourceContext.limiter
      : createAsyncResourceLimiter(limits.maxConcurrentResourceOperations);
  return {
    options,
    limits,
    resourceLedger,
    encodedResourceLedger: resourceLedger,
    imageDecodeLimiter,
    decoded: new Map(),
    specGlossBakes: new Map(),
    rawImageSnapshots: new Map(),
    rawImageResourceKeys: new Map(),
  };
}

function validateDecodeSceneTexturesOptions(
  options: DecodeSceneTexturesOptions,
): void {
  if (typeof options !== 'object' || options === null || Array.isArray(options)) {
    throw new TypeError('[vitrum/gltf-adapter] decode texture options must be an object.');
  }
  if (options.target !== 'cpu-linear' && options.target !== 'webgpu') {
    throw new TypeError(
      `[vitrum/gltf-adapter] target must be "cpu-linear" or "webgpu"; received ${safeString(options.target)}.`,
    );
  }
  if (
    options.npotRepeatWrapPolicy !== undefined &&
    options.npotRepeatWrapPolicy !== 'warn' &&
    options.npotRepeatWrapPolicy !== 'resize-to-pot' &&
    options.npotRepeatWrapPolicy !== 'clamp-sampler'
  ) {
    throw new TypeError(
      '[vitrum/gltf-adapter] npotRepeatWrapPolicy must be "warn", "resize-to-pot", or "clamp-sampler".',
    );
  }
  if (
    options.maxTextureSize !== undefined &&
    (!Number.isSafeInteger(options.maxTextureSize) || options.maxTextureSize < 0)
  ) {
    throw new RangeError(
      `[vitrum/gltf-adapter] maxTextureSize must be a non-negative safe integer; ` +
        `received ${safeString(options.maxTextureSize)}.`,
    );
  }
  validateOptionalFunction(options.decodePixels, 'decodePixels');
  validateOptionalFunction(options.onDiagnostic, 'onDiagnostic');
  validateOptionalFunction(options.onWarning, 'onWarning');
  if (
    options.warnOnNpotRepeatWrap !== undefined &&
    typeof options.warnOnNpotRepeatWrap !== 'boolean'
  ) {
    throw new TypeError('[vitrum/gltf-adapter] warnOnNpotRepeatWrap must be a boolean.');
  }
  if (
    options.resourceLimits !== undefined &&
    (
      options.resourceLimits === null ||
      typeof options.resourceLimits !== 'object' ||
      Array.isArray(options.resourceLimits)
    )
  ) {
    throw new TypeError('[vitrum/gltf-adapter] resourceLimits must be an object when supplied.');
  }
}

function validateOptionalFunction(value: unknown, name: string): void {
  if (value !== undefined && typeof value !== 'function') {
    throw new TypeError(`[vitrum/gltf-adapter] ${name} must be a function when supplied.`);
  }
}

function effectiveTextureResourceLimits(
  options: DecodeSceneTexturesOptions,
  resourceContext: GltfImportResourceContext | undefined,
): Readonly<NormalizedGltfImportResourceLimits> {
  const inherited = resourceContext?.ledger.limits;
  const structuredOverrides = validatedDefinedResourceLimitOverrides(
    options.resourceLimits,
  );
  return normalizeGltfImportResourceLimits({
    ...(inherited ?? {}),
    ...structuredOverrides,
    ...(options.maxDecodedTexturePixels !== undefined
      ? { maxDecodedTexturePixels: options.maxDecodedTexturePixels }
      : {}),
    ...(options.maxTotalDecodedTexturePixels !== undefined
      ? { maxTotalDecodedTexturePixels: options.maxTotalDecodedTexturePixels }
      : {}),
    ...(options.maxImageDecodeConcurrency !== undefined
      ? { maxConcurrentResourceOperations: options.maxImageDecodeConcurrency }
      : {}),
  });
}

function validatedDefinedResourceLimitOverrides(
  limits: GltfImportResourceLimits | undefined,
): GltfImportResourceLimits {
  if (limits === undefined) return {};
  const maxDecodedGeometryBytes = limits.maxDecodedGeometryBytes;
  const maxEncodedResourceBytes = limits.maxEncodedResourceBytes;
  const maxTotalEncodedBytes = limits.maxTotalEncodedBytes;
  const maxDecodedTexturePixels = limits.maxDecodedTexturePixels;
  const maxTotalDecodedTexturePixels = limits.maxTotalDecodedTexturePixels;
  const maxConcurrentResourceOperations = limits.maxConcurrentResourceOperations;
  const defined: GltfImportResourceLimits = {
    ...(maxDecodedGeometryBytes !== undefined
      ? { maxDecodedGeometryBytes }
      : {}),
    ...(maxEncodedResourceBytes !== undefined
      ? { maxEncodedResourceBytes }
      : {}),
    ...(maxTotalEncodedBytes !== undefined
      ? { maxTotalEncodedBytes }
      : {}),
    ...(maxDecodedTexturePixels !== undefined
      ? { maxDecodedTexturePixels }
      : {}),
    ...(maxTotalDecodedTexturePixels !== undefined
      ? { maxTotalDecodedTexturePixels }
      : {}),
    ...(maxConcurrentResourceOperations !== undefined
      ? { maxConcurrentResourceOperations }
      : {}),
  };
  // Validate the structured object before applying higher-precedence flat
  // aliases so an invalid nested value cannot be hidden by an alias.
  normalizeGltfImportResourceLimits(defined);
  return defined;
}

export async function decodeSceneTextures(
  scene: Scene,
  options: DecodeSceneTexturesOptions,
): Promise<DecodeSceneTexturesResult> {
  return decodeSceneTexturesWithContext(
    scene,
    createDecodeSceneTexturesContext(options),
  );
}

/** @internal */
export async function decodeSceneTexturesWithContext(
  scene: Scene,
  decodeContext: DecodeSceneTexturesContext,
): Promise<DecodeSceneTexturesResult> {
  const options = decodeContext.options;
  const warnings: string[] = [];
  const diagnostics: DecodeSceneTextureDiagnostic[] = [];
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
      const ref = material[field];
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
        decodeContext,
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
      decodeContext,
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

export function materialForPrimitive(primitive: ScenePrimitive): MaterialSpec {
  return primitive.material;
}

export function backendReadinessForHandle(
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

interface TextureRefDecodeContext {
  readonly field: GltfMaterialTextureField;
  readonly path: string;
  readonly source?: GltfTextureRefSource;
  readonly primitiveId: string;
  readonly primitiveIndex: number;
  readonly options: DecodeSceneTexturesOptions;
  readonly decodeContext: DecodeSceneTexturesContext;
  readonly diagnostic: (diagnostic: DecodeSceneTextureDiagnostic) => void;
}

interface ValidatedDecodedTexturePixels extends GltfDecodedTexturePixels {
  readonly width: number;
  readonly height: number;
  readonly data: ArrayLike<number>;
  readonly channels: 1 | 2 | 3 | 4;
  readonly dataType: 'uint8' | 'uint16' | 'float32';
  readonly pixelCount: number;
  readonly dataLength: number;
}

type DecodedPixelsValidation =
  | { readonly valid: true; readonly pixels: ValidatedDecodedTexturePixels }
  | {
    readonly valid: false;
    readonly reason: string;
    readonly width?: number;
    readonly height?: number;
  };

async function decodeTextureRef(
  ref: TextureRef,
  context: TextureRefDecodeContext,
): Promise<TextureRef> {
  const handleKind = classifyTextureHandle(ref.handle);
  const colorSpace = gltfTextureColorSpaceForField(context.field);
  const outputColorSpace: GltfTextureColorSpace =
    context.options.target === 'webgpu' ? colorSpace : 'linear';

  let operation: (() => Promise<DecodedTextureCacheOutcome>) | undefined;
  if (handleKind === 'pixel-data' || handleKind === 'data-texture') {
    operation = () => Promise.resolve().then(() => {
      let pixels: GltfDecodedTexturePixels | null;
      try {
        pixels = decodedPixelsFromCpuReadableHandle(ref.handle);
      } catch (err) {
        return {
          kind: 'invalid',
          reason: `CPU-readable texture property access failed: ${safeErrorMessage(err)}`,
        };
      }
      if (pixels === null) {
        return {
          kind: 'invalid',
          reason: 'CPU-readable texture handle did not expose stable width, height, and data properties',
        };
      }
      return outcomeFromDecodedPixels(
        pixels,
        colorSpace,
        outputColorSpace,
        context,
      );
    });
  } else if (handleKind === 'raw-image') {
    operation = () => decodeRawTextureOutcome(
      ref.handle as RawImageHandle,
      context.options.decodePixels,
      colorSpace,
      outputColorSpace,
      context,
    );
  } else {
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

  let perSpace = context.decodeContext.decoded.get(ref.handle);
  if (perSpace === undefined) {
    perSpace = new Map();
    context.decodeContext.decoded.set(ref.handle, perSpace);
  }
  let pending = perSpace.get(colorSpace);
  if (pending === undefined) {
    // Install the promise before awaiting it. Both raw and CPU-readable handles
    // therefore deduplicate concurrent references, including failures.
    const created = operation().catch((err): DecodedTextureCacheOutcome => ({
      kind: 'normalization-error',
      causeMessage: safeErrorMessage(err),
    }));
    pending = created;
    perSpace.set(colorSpace, created);
    void created.then((settled) => {
      if (
        settled.kind !== 'decoded' &&
        perSpace?.get(colorSpace) === created
      ) {
        perSpace.delete(colorSpace);
        if (perSpace.size === 0) {
          context.decodeContext.decoded.delete(ref.handle);
        }
      }
    });
  }
  const outcome = await pending;
  if (outcome.kind !== 'decoded') {
    emitTextureDecodeFailure(outcome, handleKind, context);
    return ref;
  }

  emitDecodedTextureDiagnostics(outcome.entry, ref, context);
  try {
    return applyNpotRepeatWrapPolicy(ref, outcome.entry, context);
  } catch (err) {
    emitDerivedTextureFailure(err, context);
    return attachGltfTextureRefSource(
      { ...ref, handle: outcome.entry.handle },
      context.source,
    );
  }
}

function builtInRawImageDecoder(
  handle: RawImageHandle,
): DecodeGltfTexturePixelsFn | undefined {
  // PNG and JPEG are intentionally routed through the same pure-JS decoder
  // in browser and Node. Browser image/canvas decoding remains a fallback for
  // WebP and other platform-supported formats; hosts that require exact
  // cross-platform WebP pixels can provide the already-active decodePixels
  // hook to select one decoder implementation everywhere.
  if (canDecodeRawPngPixelsDeterministically(handle)) {
    return decodeRawPngPixelsDeterministically;
  }
  if (canDecodeRawJpegPixelsDeterministically(handle)) {
    return decodeRawJpegPixelsDeterministically;
  }
  if (canDecodeRawImagePixelsWithPlatform()) return decodeRawImagePixelsWithPlatform;
  if (canDecodeRawWebpPixelsWithNode(handle)) return decodeRawWebpPixelsWithNode;
  return undefined;
}

async function decodeRawTextureOutcome(
  handle: RawImageHandle,
  decodePixels: DecodeGltfTexturePixelsFn | undefined,
  colorSpace: GltfTextureColorSpace,
  outputColorSpace: GltfTextureColorSpace,
  context: TextureRefDecodeContext,
): Promise<DecodedTextureCacheOutcome> {
  const snapshot = await rawImageSnapshot(handle, context);
  if (snapshot.kind === 'platform-error') {
    return { kind: 'platform-error', error: snapshot.error };
  }
  if (snapshot.kind === 'resource-limit') {
    return { kind: 'resource-limit', error: snapshot.error };
  }
  const selectedDecoder = decodePixels ?? builtInRawImageDecoder(snapshot.handle);
  if (selectedDecoder === undefined) return { kind: 'decoder-missing' };
  let pixels: GltfDecodedTexturePixels;
  try {
    pixels = await context.decodeContext.imageDecodeLimiter.run(() => selectedDecoder(snapshot.handle, {
      materialField: context.field,
      path: context.path,
      colorSpace,
      primitiveId: context.primitiveId,
      primitiveIndex: context.primitiveIndex,
      ...textureSourceDiagnosticFields(context.source),
      maxDecodedTexturePixels: context.decodeContext.limits.maxDecodedTexturePixels,
    }));
  } catch (err) {
    if (err instanceof PlatformTextureDecodeError) {
      return { kind: 'platform-error', error: err };
    }
    return { kind: 'decode-error', causeMessage: safeErrorMessage(err) };
  }
  return outcomeFromDecodedPixels(
    pixels,
    colorSpace,
    outputColorSpace,
    context,
  );
}

async function rawImageSnapshot(
  handle: RawImageHandle,
  context: TextureRefDecodeContext,
): Promise<RawImageSnapshotOutcome> {
  const snapshots = context.decodeContext.rawImageSnapshots;
  const cached = snapshots.get(handle);
  if (cached !== undefined) return cached;

  let resourceKey = context.decodeContext.rawImageResourceKeys.get(handle);
  if (resourceKey === undefined) {
    resourceKey = context.source?.imageIndex === undefined
      ? `texture-raw:${context.decodeContext.rawImageResourceKeys.size}`
      : gltfImageResourceKey(context.source.imageIndex);
    context.decodeContext.rawImageResourceKeys.set(handle, resourceKey);
  }
  const stableResourceKey = resourceKey;
  const created = Promise.resolve().then((): RawImageSnapshotOutcome => {
    let data: unknown;
    let mimeType: unknown;
    try {
      const candidate = handle as unknown as {
        readonly data?: unknown;
        readonly mimeType?: unknown;
      };
      data = candidate.data;
      mimeType = candidate.mimeType;
    } catch (err) {
      return {
        kind: 'platform-error',
        error: new PlatformTextureDecodeError(
          'platform-image-decode-failed',
          `[vitrum/gltf-adapter] ${context.path} raw-image data access failed: ` +
            `${safeErrorMessage(err)}. Texture left unchanged.`,
        ),
      };
    }
    const info = inspectIntrinsicTypedArray(data);
    if (info === null || info.brand !== 'Uint8Array' || info.isShared) {
      return {
        kind: 'platform-error',
        error: new PlatformTextureDecodeError(
          'platform-image-decode-failed',
          `[vitrum/gltf-adapter] ${context.path} raw-image data must be a non-shared intrinsic ` +
            'Uint8Array. Texture left unchanged.',
        ),
      };
    }
    try {
      context.decodeContext.encodedResourceLedger.chargeEncodedBytes(
        stableResourceKey,
        info.byteLength,
        context.path,
      );
      return {
        kind: 'ready',
        handle: normalizeCapturedRawImageForDecode(
          data,
          mimeType,
          context.path,
        ),
      };
    } catch (err) {
      if (err instanceof GltfResourceLimitError) {
        return { kind: 'resource-limit', error: err };
      }
      if (err instanceof PlatformTextureDecodeError) {
        return { kind: 'platform-error', error: err };
      }
      return {
        kind: 'platform-error',
        error: new PlatformTextureDecodeError(
          'platform-image-decode-failed',
          `[vitrum/gltf-adapter] ${context.path} raw-image snapshot failed: ` +
            `${safeErrorMessage(err)}. Texture left unchanged.`,
        ),
      };
    }
  });
  snapshots.set(handle, created);
  void created.then((settled) => {
    if (settled.kind !== 'ready' && snapshots.get(handle) === created) {
      snapshots.delete(handle);
    }
  });
  return created;
}

function outcomeFromDecodedPixels(
  value: unknown,
  colorSpace: GltfTextureColorSpace,
  outputColorSpace: GltfTextureColorSpace,
  context: TextureRefDecodeContext,
): DecodedTextureCacheOutcome {
  const validation = validateDecodedTexturePixels(value);
  if (!validation.valid) {
    return {
      kind: 'invalid',
      reason: validation.reason,
      ...(validation.width !== undefined ? { width: validation.width } : {}),
      ...(validation.height !== undefined ? { height: validation.height } : {}),
    };
  }
  const pixels = validation.pixels;
  const maxPixels = context.decodeContext.limits.maxDecodedTexturePixels;
  if (maxPixels !== 0 && pixels.pixelCount > maxPixels) {
    return {
      kind: 'resource-limit',
      error: new GltfResourceLimitError({
        limitKind: 'decoded-texture-pixels',
        limit: maxPixels,
        actual: pixels.pixelCount,
        path: context.path,
      }),
      width: pixels.width,
      height: pixels.height,
    };
  }
  try {
    return {
      kind: 'decoded',
      entry: cacheEntryFromDecodedPixels(
        pixels,
        colorSpace,
        outputColorSpace,
        context.options.maxTextureSize,
        context.decodeContext,
        context.path,
      ),
    };
  } catch (err) {
    if (err instanceof GltfResourceLimitError) {
      return {
        kind: 'resource-limit',
        error: err,
        width: pixels.width,
        height: pixels.height,
      };
    }
    return {
      kind: 'normalization-error',
      causeMessage: safeErrorMessage(err),
    };
  }
}

function emitTextureDecodeFailure(
  outcome: Exclude<DecodedTextureCacheOutcome, { readonly kind: 'decoded' }>,
  handleKind: GltfTextureHandleKind,
  context: TextureRefDecodeContext,
): void {
  const common = {
    severity: 'warning' as const,
    path: context.path,
    materialField: context.field,
    primitiveId: context.primitiveId,
    primitiveIndex: context.primitiveIndex,
    handleKind,
    ...textureSourceDiagnosticFields(context.source),
  };
  if (outcome.kind === 'platform-error') {
    context.diagnostic({
      ...common,
      code: outcome.error.code,
      ...(outcome.error.width !== undefined ? { width: outcome.error.width } : {}),
      ...(outcome.error.height !== undefined ? { height: outcome.error.height } : {}),
      ...(outcome.error.maxDecodedTexturePixels !== undefined
        ? { maxDecodedTexturePixels: outcome.error.maxDecodedTexturePixels }
        : {}),
      message: outcome.error.message,
    });
    return;
  }
  if (outcome.kind === 'resource-limit') {
    emitResourceLimitDiagnostic(
      outcome.error,
      context,
      handleKind,
      outcome.width,
      outcome.height,
    );
    return;
  }
  if (outcome.kind === 'invalid') {
    context.diagnostic({
      ...common,
      code: 'decode-pixels-invalid',
      ...(outcome.width !== undefined ? { width: outcome.width } : {}),
      ...(outcome.height !== undefined ? { height: outcome.height } : {}),
      message: `[vitrum/gltf-adapter] ${context.path} decodePixels hook returned invalid pixels: ` +
        `${outcome.reason}. Texture left unchanged.`,
    });
    return;
  }
  if (outcome.kind === 'decoder-missing') {
    context.diagnostic({
      ...common,
      code: 'raw-image-decoder-missing',
      message: rawImageDecoderMissingMessage(context),
    });
    return;
  }
  const failed = outcome.kind === 'decode-error';
  context.diagnostic({
    ...common,
    code: failed ? 'decode-pixels-failed' : 'decode-pixels-invalid',
    causeMessage: outcome.causeMessage,
    message: `[vitrum/gltf-adapter] ${context.path} ` +
      `${failed ? 'decodePixels hook failed' : 'could not normalize decoded pixels'}: ` +
      `${outcome.causeMessage}. Texture left unchanged.`,
  });
}

function emitResourceLimitDiagnostic(
  error: GltfResourceLimitError,
  context: TextureRefDecodeContext,
  handleKind?: GltfTextureHandleKind,
  width?: number,
  height?: number,
): void {
  const totalPixels = error.limitKind === 'total-decoded-texture-pixels';
  const encoded = error.limitKind === 'encoded-resource-bytes';
  const totalEncoded = error.limitKind === 'total-encoded-bytes';
  const code: DecodeSceneTextureDiagnosticCode = encoded
    ? 'encoded-texture-exceeds-byte-budget'
    : totalEncoded
      ? 'encoded-textures-exceed-total-byte-budget'
      : totalPixels
        ? 'decoded-texture-exceeds-total-pixel-budget'
        : 'decoded-texture-exceeds-pixel-budget';
  context.diagnostic({
    severity: 'warning',
    code,
    path: context.path,
    materialField: context.field,
    primitiveId: context.primitiveId,
    primitiveIndex: context.primitiveIndex,
    ...(handleKind !== undefined ? { handleKind } : {}),
    ...(width !== undefined ? { width } : {}),
    ...(height !== undefined ? { height } : {}),
    ...(encoded
      ? {
          maxEncodedTextureBytes: error.limit,
          encodedTextureBytes: error.actual,
        }
      : totalEncoded
        ? {
            maxTotalEncodedTextureBytes: error.limit,
            totalEncodedTextureBytes: error.actual,
          }
        : totalPixels
      ? {
          maxTotalDecodedTexturePixels: error.limit,
          totalDecodedTexturePixels: error.actual,
        }
      : {
          maxDecodedTexturePixels: error.limit,
        }),
    ...textureSourceDiagnosticFields(context.source),
    message: `${error.message} Texture left unchanged before ` +
      `${encoded || totalEncoded ? 'raw-image snapshot allocation' : 'adapter output allocation'}.`,
  });
}

function emitDerivedTextureFailure(
  error: unknown,
  context: TextureRefDecodeContext,
): void {
  if (error instanceof GltfResourceLimitError) {
    emitResourceLimitDiagnostic(error, context);
    return;
  }
  const causeMessage = safeErrorMessage(error);
  context.diagnostic({
    severity: 'warning',
    code: 'decode-pixels-invalid',
    path: context.path,
    materialField: context.field,
    primitiveId: context.primitiveId,
    primitiveIndex: context.primitiveIndex,
    ...textureSourceDiagnosticFields(context.source),
    causeMessage,
    message: `[vitrum/gltf-adapter] ${context.path} could not allocate a derived texture: ` +
      `${causeMessage}. The normalized texture was retained.`,
  });
}

function validateDecodedTexturePixels(value: unknown): DecodedPixelsValidation {
  let widthValue: unknown;
  let heightValue: unknown;
  let data: unknown;
  let channelsValue: unknown;
  let dataTypeValue: unknown;
  let colorSpaceValue: unknown;
  try {
    if (typeof value !== 'object' || value === null) {
      return { valid: false, reason: 'result must be a non-null object' };
    }
    const pixels = value as Record<string, unknown>;
    widthValue = pixels.width;
    heightValue = pixels.height;
    data = pixels.data;
    channelsValue = pixels.channels;
    dataTypeValue = pixels.dataType;
    colorSpaceValue = pixels.colorSpace;
  } catch (err) {
    return {
      valid: false,
      reason: `pixel result property access failed: ${safeErrorMessage(err)}`,
    };
  }
  const diagnosticDimensions = {
    ...(typeof widthValue === 'number' && Number.isFinite(widthValue)
      ? { width: widthValue }
      : {}),
    ...(typeof heightValue === 'number' && Number.isFinite(heightValue)
      ? { height: heightValue }
      : {}),
  };
  if (typeof widthValue !== 'number' || !Number.isFinite(widthValue)) {
    return {
      valid: false,
      reason: `width must be a finite number, got ${safeString(widthValue)}`,
      ...diagnosticDimensions,
    };
  }
  if (typeof heightValue !== 'number' || !Number.isFinite(heightValue)) {
    return {
      valid: false,
      reason: `height must be a finite number, got ${safeString(heightValue)}`,
      ...diagnosticDimensions,
    };
  }
  if (!Number.isSafeInteger(widthValue) || !Number.isSafeInteger(heightValue)) {
    return {
      valid: false,
      reason: `dimensions must be safe integers, got ${widthValue}x${heightValue}`,
      ...diagnosticDimensions,
    };
  }
  if (widthValue <= 0 || heightValue <= 0) {
    return {
      valid: false,
      reason: `dimensions must be positive, got ${widthValue}x${heightValue}`,
      ...diagnosticDimensions,
    };
  }

  const dataInfo = inspectPixelData(data);
  if (!dataInfo.valid) {
    return {
      valid: false,
      reason: dataInfo.reason,
      ...diagnosticDimensions,
    };
  }
  if (
    channelsValue !== undefined &&
    channelsValue !== 1 &&
    channelsValue !== 2 &&
    channelsValue !== 3 &&
    channelsValue !== 4
  ) {
    return {
      valid: false,
      reason: `channels must be 1, 2, 3, or 4, got ${safeString(channelsValue)}`,
      ...diagnosticDimensions,
    };
  }
  if (
    dataTypeValue !== undefined &&
    dataTypeValue !== 'uint8' &&
    dataTypeValue !== 'uint16' &&
    dataTypeValue !== 'float32'
  ) {
    return {
      valid: false,
      reason: `dataType must be uint8, uint16, or float32, got ${safeString(dataTypeValue)}`,
      ...diagnosticDimensions,
    };
  }
  if (
    colorSpaceValue !== undefined &&
    colorSpaceValue !== 'srgb' &&
    colorSpaceValue !== 'linear'
  ) {
    return {
      valid: false,
      reason: `colorSpace must be srgb or linear, got ${safeString(colorSpaceValue)}`,
      ...diagnosticDimensions,
    };
  }

  const pixelCount = checkedSafeProduct(
    [widthValue, heightValue],
    'decoded texture pixel count',
  );
  if (pixelCount === null) {
    return {
      valid: false,
      reason: `${widthValue}x${heightValue} pixel count exceeds the safe integer range`,
      ...diagnosticDimensions,
    };
  }
  const channels = channelsValue ??
    inferDecodedChannelsFromLength(dataInfo.length, pixelCount);
  const requiredLength = checkedSafeProduct(
    [pixelCount, channels],
    'decoded texture component count',
  );
  if (requiredLength === null) {
    return {
      valid: false,
      reason: `${widthValue}x${heightValue}x${channels} component count exceeds the safe integer range`,
      ...diagnosticDimensions,
    };
  }
  if (dataInfo.length < requiredLength) {
    return {
      valid: false,
      reason: `data length ${dataInfo.length} is too short for ` +
        `${widthValue}x${heightValue}x${channels}; expected at least ${requiredLength}`,
      ...diagnosticDimensions,
    };
  }
  return {
    valid: true,
    pixels: {
      width: widthValue,
      height: heightValue,
      data: data as ArrayLike<number>,
      channels,
      dataType: dataTypeValue ?? dataInfo.inferredDataType,
      pixelCount,
      dataLength: dataInfo.length,
      ...(colorSpaceValue !== undefined ? { colorSpace: colorSpaceValue } : {}),
    },
  };
}

function cacheEntryFromDecodedPixels(
  pixels: ValidatedDecodedTexturePixels,
  colorSpace: GltfTextureColorSpace,
  outputColorSpace: GltfTextureColorSpace,
  maxTextureSize: number | undefined,
  decodeContext: DecodeSceneTexturesContext,
  path: string,
): DecodedTextureCacheEntry {
  // Clamp target dimensions to maxTextureSize BEFORE allocation: the
  // decode+resize is fused so only the clamped Float32Array is ever created,
  // never the full-resolution buffer for an over-`maxTextureSize` source.
  const sourceWidth = pixels.width;
  const sourceHeight = pixels.height;
  const normalized = normalizeDecodedPixels(
    pixels,
    colorSpace,
    outputColorSpace,
    maxTextureSize,
    decodeContext,
    path,
  );
  const wasClamped = normalized.width !== sourceWidth || normalized.height !== sourceHeight;
  const shouldAnnotate =
    wasClamped || (typeof maxTextureSize === 'number' && maxTextureSize > 0);
  return {
    handle: shouldAnnotate
      ? withDecodedTextureMetadata(
          normalized,
          sourceWidth,
          sourceHeight,
          maxTextureSize,
        )
      : normalized,
    originalWidth: sourceWidth,
    originalHeight: sourceHeight,
  };
}


function maybeBakeSpecGlossRoughnessMap(
  material: MaterialSpec,
  context: {
    readonly primitiveId: string;
    readonly primitiveIndex: number;
    readonly options: DecodeSceneTexturesOptions;
    readonly diagnostic: (diagnostic: DecodeSceneTextureDiagnostic) => void;
    readonly decodeContext: DecodeSceneTexturesContext;
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
  const textureContext: TextureRefDecodeContext = {
    field: 'roughnessMap',
    path,
    ...(source !== undefined ? { source } : {}),
    primitiveId: context.primitiveId,
    primitiveIndex: context.primitiveIndex,
    options: context.options,
    decodeContext: context.decodeContext,
    diagnostic: context.diagnostic,
  };
  try {
    const sourceHandle = cpuLinearTextureHandleForSpecGlossBake(
      sourceRef.handle,
      context.options.maxTextureSize,
      context.decodeContext,
      path,
    );
    if (sourceHandle !== null) {
      return attachGltfTextureRefSource({
        ...sourceRef,
        handle: getOrBakeSpecGlossRoughnessHandle(
          sourceRef.handle,
          sourceHandle,
          glossinessFactor,
          context.decodeContext.specGlossBakes,
          context.decodeContext,
          path,
        ),
      }, source);
    }
  } catch (err) {
    if (err instanceof GltfResourceLimitError) {
      emitResourceLimitDiagnostic(err, textureContext);
    } else {
      const causeMessage = safeErrorMessage(err);
      context.diagnostic({
        severity: 'warning',
        code: 'decode-pixels-invalid',
        path,
        materialField: 'roughnessMap',
        primitiveId: context.primitiveId,
        primitiveIndex: context.primitiveIndex,
        ...textureSourceDiagnosticFields(source),
        causeMessage,
        message: `[vitrum/gltf-adapter] ${path} could not bake specular-glossiness alpha into roughness: ` +
          `${causeMessage}. No derived roughnessMap was installed.`,
      });
    }
    return null;
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
  decodeContext: DecodeSceneTexturesContext,
  path: string,
): GltfCpuLinearTextureHandle {
  let perSource = cache.get(cacheKey);
  if (perSource == null) {
    perSource = new Map();
    cache.set(cacheKey, perSource);
  }
  const key = Math.round(glossinessFactor * 1_000_000);
  const cached = perSource.get(key);
  if (cached !== undefined) return cached;

  const pixelCount = checkedPixelCount(source.width, source.height, path);
  const data = allocateFloat32Rgba(pixelCount, decodeContext, path);
  for (let p = 0; p < pixelCount; p += 1) {
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

function cpuLinearTextureHandleForSpecGlossBake(
  handle: unknown,
  maxTextureSize: number | undefined,
  decodeContext: DecodeSceneTexturesContext,
  path: string,
): GltfCpuLinearTextureHandle | null {
  const alreadyLinear = isCpuLinearTextureHandle(handle);
  const pixels: GltfDecodedTexturePixels | null = alreadyLinear
    ? {
        width: handle.width,
        height: handle.height,
        data: handle.data,
        channels: 4,
        dataType: 'float32',
        colorSpace: 'linear',
      }
    : decodedPixelsFromCpuReadableHandle(handle);
  if (pixels === null) return null;
  const validation = validateDecodedTexturePixels(pixels);
  if (!validation.valid) return null;
  if (
    decodeContext.limits.maxDecodedTexturePixels !== 0 &&
    validation.pixels.pixelCount > decodeContext.limits.maxDecodedTexturePixels
  ) {
    throw new GltfResourceLimitError({
      limitKind: 'decoded-texture-pixels',
      limit: decodeContext.limits.maxDecodedTexturePixels,
      actual: validation.pixels.pixelCount,
      path,
    });
  }
  if (alreadyLinear) return handle;
  return normalizeDecodedPixels(
    validation.pixels,
    'srgb',
    'linear',
    maxTextureSize,
    decodeContext,
    path,
  );
}

export function decodedPixelsFromCpuReadableHandle(handle: unknown): GltfDecodedTexturePixels | null {
  if (isRecord(handle)) {
    const direct = decodedPixelsFromRecord(handle);
    if (direct !== null) return direct;
    if (isRecord(handle.image)) return decodedPixelsFromRecord(handle.image, handle);
  }
  return null;
}

/** Non-allocating readiness validation used by texture reports. @internal */
export function isCpuReadableTexturePayloadValid(handle: unknown): boolean {
  try {
    const pixels = decodedPixelsFromCpuReadableHandle(handle);
    return pixels !== null && validateDecodedTexturePixels(pixels).valid;
  } catch {
    return false;
  }
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
  const metadataHint = metadata.__vitrum_hint__;
  const descriptor = isRecord(metadataHint) ? metadataHint : metadata;
  const channels = descriptor.channels;
  const dataType = descriptor.dataType;
  const colorSpace = descriptor.colorSpace;
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
  context: TextureRefDecodeContext,
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
  context: TextureRefDecodeContext,
): TextureRef {
  const policy = effectiveNpotRepeatWrapPolicy(context.options);
  const handle = entry.handle;
  if (policy === 'none' || policy === 'warn' || isPowerOfTwo(handle.width, handle.height) || !usesRepeatWrap(ref)) {
    return attachGltfTextureRefSource({ ...ref, handle }, context.source);
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
    return attachGltfTextureRefSource({
      ...ref,
      handle,
      wrapS: 'clamp-to-edge',
      wrapT: 'clamp-to-edge',
    }, context.source);
  }

  const resized = resizeDecodedTextureToPowerOfTwo(
    handle,
    context.options.maxTextureSize,
    context.decodeContext,
    context.path,
  );
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
  return attachGltfTextureRefSource({ ...ref, handle: resized }, context.source);
}

function normalizeDecodedPixels(
  pixels: ValidatedDecodedTexturePixels,
  fieldColorSpace: GltfTextureColorSpace,
  outputColorSpace: 'linear',
  maxTextureSize: number | undefined,
  decodeContext: DecodeSceneTexturesContext,
  path: string,
): GltfCpuLinearTextureHandle;
function normalizeDecodedPixels(
  pixels: ValidatedDecodedTexturePixels,
  fieldColorSpace: GltfTextureColorSpace,
  outputColorSpace: 'srgb',
  maxTextureSize: number | undefined,
  decodeContext: DecodeSceneTexturesContext,
  path: string,
): GltfCpuTextureHandle;
function normalizeDecodedPixels(
  pixels: ValidatedDecodedTexturePixels,
  fieldColorSpace: GltfTextureColorSpace,
  outputColorSpace: GltfTextureColorSpace,
  maxTextureSize: number | undefined,
  decodeContext: DecodeSceneTexturesContext,
  path: string,
): GltfCpuTextureHandle;
function normalizeDecodedPixels(
  pixels: ValidatedDecodedTexturePixels,
  fieldColorSpace: GltfTextureColorSpace,
  outputColorSpace: GltfTextureColorSpace,
  maxTextureSize: number | undefined,
  decodeContext: DecodeSceneTexturesContext,
  path: string,
): GltfCpuTextureHandle {
  const sourceWidth = pixels.width;
  const sourceHeight = pixels.height;
  // Clamp the OUTPUT dimensions to maxTextureSize BEFORE allocating the
  // destination buffer, then filter the source directly into it. This fuses
  // normalization and resizing so the full-resolution Float32Array is never
  // allocated for an over-`maxTextureSize` source.
  const { width, height } = clampToMaxTextureSize(sourceWidth, sourceHeight, maxTextureSize);
  const channels = pixels.channels;
  const dataType = pixels.dataType;
  const sourceColorSpace = pixels.colorSpace ?? fieldColorSpace;
  const pixelCount = checkedPixelCount(width, height, path);
  const out = allocateFloat32Rgba(pixelCount, decodeContext, path);

  if (width !== sourceWidth || height !== sourceHeight) {
    resampleDecodedTexture(
      pixels.data,
      channels,
      dataType,
      sourceWidth,
      sourceHeight,
      sourceColorSpace,
      outputColorSpace,
      width,
      height,
      out,
    );
    return {
      width,
      height,
      data: out,
      __vitrum_hint__: { channels: 4, dataType: 'float32', colorSpace: outputColorSpace },
    };
  }

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const s = (y * sourceWidth + x) * channels;
      const dst = (y * width + x) * 4;
      const r = decodeChannel(pixels.data[s] ?? 0, dataType);
      const g = decodeChannel(pixels.data[s + (channels >= 3 ? 1 : 0)] ?? 0, dataType);
      const b = decodeChannel(pixels.data[s + (channels > 2 ? 2 : 0)] ?? 0, dataType);
      const a =
        channels === 2
          ? decodeChannel(pixels.data[s + 1] ?? 1, dataType)
          : channels === 4
            ? decodeChannel(pixels.data[s + 3] ?? 1, dataType)
            : 1;
      out[dst] = convertColorChannel(r, sourceColorSpace, outputColorSpace);
      out[dst + 1] = convertColorChannel(g, sourceColorSpace, outputColorSpace);
      out[dst + 2] = convertColorChannel(b, sourceColorSpace, outputColorSpace);
      out[dst + 3] = a;
    }
  }
  return {
    width,
    height,
    data: out,
    __vitrum_hint__: { channels: 4, dataType: 'float32', colorSpace: outputColorSpace },
  };
}

/**
 * Compute the clamped output dimensions for a source of `width`×`height`,
 * scaled down to fit within `maxTextureSize` on its longest edge (the same
 * scale the former `resizeDecodedTextureToMaxSize` used). Returns the source
 * dimensions unchanged when no clamp applies.
 */
function clampToMaxTextureSize(
  width: number,
  height: number,
  maxTextureSize: number | undefined,
): { readonly width: number; readonly height: number } {
  if (typeof maxTextureSize !== 'number' || maxTextureSize <= 0) return { width, height };
  if (width <= maxTextureSize && height <= maxTextureSize) return { width, height };
  const scale = maxTextureSize / Math.max(width, height);
  return {
    width: Math.max(1, Math.round(width * scale)),
    height: Math.max(1, Math.round(height * scale)),
  };
}

function resizeDecodedTextureToPowerOfTwo(
  handle: GltfCpuTextureHandle,
  maxTextureSize: number | undefined,
  decodeContext: DecodeSceneTexturesContext,
  path: string,
): GltfCpuTextureHandle {
  const width = powerOfTwoTarget(handle.width, maxTextureSize);
  const height = powerOfTwoTarget(handle.height, maxTextureSize);
  if (width === handle.width && height === handle.height) return handle;
  const resized = resizeDecodedTextureFiltered(
    handle,
    width,
    height,
    decodeContext,
    path,
  );
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

function resizeDecodedTextureFiltered(
  handle: GltfCpuTextureHandle,
  width: number,
  height: number,
  decodeContext: DecodeSceneTexturesContext,
  path: string,
): GltfCpuTextureHandle {
  if (width === handle.width && height === handle.height) return handle;
  const pixelCount = checkedPixelCount(width, height, path);
  const data = allocateFloat32Rgba(pixelCount, decodeContext, path);
  resampleDecodedTexture(
    handle.data,
    4,
    'float32',
    handle.width,
    handle.height,
    handle.__vitrum_hint__.colorSpace,
    handle.__vitrum_hint__.colorSpace,
    width,
    height,
    data,
  );

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

interface AxisFilterTap {
  readonly index: number;
  readonly weight: number;
}

/**
 * Build one-dimensional reconstruction weights for a resize axis.
 *
 * Downsampling integrates the exact overlap of the destination footprint with
 * each source texel (a box/area filter), so high-frequency energy is not
 * discarded by point sampling. Upsampling uses pixel-centred bilinear
 * reconstruction. The two axis tables are combined as a separable 2D filter.
 */
function buildAxisFilterTable(
  sourceSize: number,
  targetSize: number,
): readonly (readonly AxisFilterTap[])[] {
  if (sourceSize === targetSize) {
    return Array.from({ length: targetSize }, (_, index) => [{ index, weight: 1 }]);
  }

  if (targetSize < sourceSize) {
    const scale = sourceSize / targetSize;
    return Array.from({ length: targetSize }, (_, targetIndex) => {
      const start = targetIndex * scale;
      const end = (targetIndex + 1) * scale;
      const first = Math.floor(start);
      const last = Math.min(sourceSize - 1, Math.ceil(end) - 1);
      const taps: AxisFilterTap[] = [];
      for (let sourceIndex = first; sourceIndex <= last; sourceIndex += 1) {
        const overlap = Math.max(
          0,
          Math.min(end, sourceIndex + 1) - Math.max(start, sourceIndex),
        );
        if (overlap > 0) {
          taps.push({ index: sourceIndex, weight: overlap / scale });
        }
      }
      return taps;
    });
  }

  const scale = sourceSize / targetSize;
  return Array.from({ length: targetSize }, (_, targetIndex) => {
    const sourcePosition = (targetIndex + 0.5) * scale - 0.5;
    const lowerUnclamped = Math.floor(sourcePosition);
    const upperUnclamped = lowerUnclamped + 1;
    const upperWeight = sourcePosition - lowerUnclamped;
    const lower = Math.max(0, Math.min(sourceSize - 1, lowerUnclamped));
    const upper = Math.max(0, Math.min(sourceSize - 1, upperUnclamped));
    if (lower === upper) return [{ index: lower, weight: 1 }];
    return [
      { index: lower, weight: 1 - upperWeight },
      { index: upper, weight: upperWeight },
    ];
  });
}

/**
 * Filter arbitrary decoded channel layouts directly into an RGBA Float32
 * destination. RGB values are reconstructed in linear light and encoded only
 * after filtering; alpha remains an independent linear channel because glTF
 * data textures also use it for non-opacity payloads.
 */
function resampleDecodedTexture(
  source: ArrayLike<number>,
  channels: 1 | 2 | 3 | 4,
  dataType: 'uint8' | 'uint16' | 'float32',
  sourceWidth: number,
  sourceHeight: number,
  sourceColorSpace: GltfTextureColorSpace,
  outputColorSpace: GltfTextureColorSpace,
  targetWidth: number,
  targetHeight: number,
  destination: Float32Array,
): void {
  const xFilters = buildAxisFilterTable(sourceWidth, targetWidth);
  const yFilters = buildAxisFilterTable(sourceHeight, targetHeight);

  for (let y = 0; y < targetHeight; y += 1) {
    const yTaps = yFilters[y]!;
    for (let x = 0; x < targetWidth; x += 1) {
      const xTaps = xFilters[x]!;
      let r = 0;
      let g = 0;
      let b = 0;
      let a = 0;

      for (const yTap of yTaps) {
        for (const xTap of xTaps) {
          const weight = xTap.weight * yTap.weight;
          const offset = (yTap.index * sourceWidth + xTap.index) * channels;
          const sourceR = decodeChannel(source[offset] ?? 0, dataType);
          const sourceG = decodeChannel(
            source[offset + (channels >= 3 ? 1 : 0)] ?? 0,
            dataType,
          );
          const sourceB = decodeChannel(
            source[offset + (channels >= 3 ? 2 : 0)] ?? 0,
            dataType,
          );
          const sourceA = channels === 2
            ? decodeChannel(source[offset + 1] ?? 1, dataType)
            : channels === 4
              ? decodeChannel(source[offset + 3] ?? 1, dataType)
              : 1;
          r += (sourceColorSpace === 'srgb' ? srgbToLinear(sourceR) : sourceR) * weight;
          g += (sourceColorSpace === 'srgb' ? srgbToLinear(sourceG) : sourceG) * weight;
          b += (sourceColorSpace === 'srgb' ? srgbToLinear(sourceB) : sourceB) * weight;
          a += sourceA * weight;
        }
      }

      const destinationOffset = (y * targetWidth + x) * 4;
      destination[destinationOffset] = outputColorSpace === 'srgb' ? linearToSrgb(r) : r;
      destination[destinationOffset + 1] = outputColorSpace === 'srgb' ? linearToSrgb(g) : g;
      destination[destinationOffset + 2] = outputColorSpace === 'srgb' ? linearToSrgb(b) : b;
      destination[destinationOffset + 3] = a;
    }
  }
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
  };
}

type PixelDataInspection =
  | {
    readonly valid: true;
    readonly length: number;
    readonly inferredDataType: 'uint8' | 'uint16' | 'float32';
  }
  | {
    readonly valid: false;
    readonly reason: string;
  };

function inspectPixelData(data: unknown): PixelDataInspection {
  const typed = inspectIntrinsicTypedArray(data);
  if (typed !== null) {
    if (typed.isShared) {
      return {
        valid: false,
        reason: 'data must not be backed by SharedArrayBuffer',
      };
    }
    if (typed.brand === 'BigInt64Array' || typed.brand === 'BigUint64Array') {
      return {
        valid: false,
        reason: `${typed.brand} pixel data is unsupported`,
      };
    }
    return {
      valid: true,
      length: typed.length,
      inferredDataType:
        typed.brand === 'Uint8Array' || typed.brand === 'Uint8ClampedArray'
          ? 'uint8'
          : typed.brand === 'Uint16Array'
            ? 'uint16'
            : 'float32',
    };
  }
  if (typeof data !== 'object' || data === null) {
    return { valid: false, reason: 'data must be an array-like pixel payload' };
  }
  try {
    const length = (data as { readonly length?: unknown }).length;
    if (!Number.isSafeInteger(length) || (length as number) < 0) {
      return {
        valid: false,
        reason: `data length must be a non-negative safe integer, got ${safeString(length)}`,
      };
    }
    return {
      valid: true,
      length: length as number,
      inferredDataType: 'float32',
    };
  } catch (err) {
    return {
      valid: false,
      reason: `data length access failed: ${safeErrorMessage(err)}`,
    };
  }
}

function checkedSafeProduct(
  factors: readonly number[],
  _label: string,
): number | null {
  let product = 1;
  for (const factor of factors) {
    if (!Number.isSafeInteger(factor) || factor < 0) return null;
    if (factor !== 0 && product > Math.floor(Number.MAX_SAFE_INTEGER / factor)) {
      return null;
    }
    product *= factor;
  }
  return product;
}

function safeString(value: unknown): string {
  try {
    return String(value);
  } catch {
    return '<unprintable value>';
  }
}

function safeErrorMessage(error: unknown): string {
  try {
    if (
      typeof error === 'object' &&
      error !== null &&
      typeof (error as { readonly message?: unknown }).message === 'string'
    ) {
      return (error as { readonly message: string }).message;
    }
  } catch {
    return '<error with inaccessible message>';
  }
  return safeString(error);
}

function checkedPixelCount(width: number, height: number, path: string): number {
  const pixelCount = checkedSafeProduct([width, height], `${path} pixel count`);
  if (pixelCount === null || pixelCount <= 0) {
    throw new RangeError(
      `[vitrum/gltf-adapter] ${path} has unsafe texture dimensions ${width}x${height}.`,
    );
  }
  return pixelCount;
}

function allocateFloat32Rgba(
  pixelCount: number,
  context: DecodeSceneTexturesContext,
  path: string,
): Float32Array {
  const componentCount = checkedSafeProduct(
    [pixelCount, 4],
    `${path} Float32 RGBA component count`,
  );
  if (componentCount === null) {
    throw new RangeError(
      `[vitrum/gltf-adapter] ${path} Float32 RGBA component count exceeds the safe integer range.`,
    );
  }
  context.resourceLedger.chargeDecodedTexturePixels(pixelCount, path);
  return new Float32Array(componentCount);
}

function inferDecodedChannelsFromLength(
  dataLength: number,
  pixelCount: number,
): 1 | 2 | 3 | 4 {
  const stride = Math.max(1, Math.round(dataLength / Math.max(1, pixelCount)));
  if (stride <= 1) return 1;
  if (stride === 2) return 2;
  if (stride === 3) return 3;
  return 4;
}

export function inferDecodedChannels(data: ArrayLike<number>, width: number, height: number): 1 | 2 | 3 | 4 {
  const dataInfo = inspectPixelData(data);
  if (!dataInfo.valid) {
    throw new TypeError(`[vitrum/gltf-adapter] ${dataInfo.reason}.`);
  }
  const pixelCount = checkedSafeProduct([width, height], 'decoded texture pixel count');
  if (pixelCount === null || pixelCount <= 0) {
    throw new RangeError(
      `[vitrum/gltf-adapter] Invalid decoded texture dimensions ${width}x${height}.`,
    );
  }
  return inferDecodedChannelsFromLength(dataInfo.length, pixelCount);
}

export function inferDecodedDataType(data: ArrayLike<number>): 'uint8' | 'uint16' | 'float32' {
  const dataInfo = inspectPixelData(data);
  if (!dataInfo.valid) {
    throw new TypeError(`[vitrum/gltf-adapter] ${dataInfo.reason}.`);
  }
  return dataInfo.inferredDataType;
}

export function isPowerOfTwo(width: number, height: number): boolean {
  return isSinglePowerOfTwo(width) && isSinglePowerOfTwo(height);
}

function isSinglePowerOfTwo(value: number): boolean {
  return Number.isInteger(value) && value > 0 && (value & (value - 1)) === 0;
}

function usesRepeatWrap(ref: TextureRef): boolean {
  return (ref.wrapS ?? 'repeat') !== 'clamp-to-edge' || (ref.wrapT ?? 'repeat') !== 'clamp-to-edge';
}

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

function isCpuLinearTextureHandle(handle: unknown): handle is GltfCpuLinearTextureHandle {
  if (!isRecord(handle)) return false;
  try {
    const hint = handle.__vitrum_hint__;
    const data = inspectIntrinsicTypedArray(handle.data);
    return typeof handle.width === 'number' &&
      typeof handle.height === 'number' &&
      data?.brand === 'Float32Array' &&
      !data.isShared &&
      isRecord(hint) &&
      hint.channels === 4 &&
      hint.dataType === 'float32' &&
      hint.colorSpace === 'linear';
  } catch {
    return false;
  }
}

export function textureHandleColorSpace(handle: unknown): GltfTextureColorSpace | undefined {
  try {
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
  } catch {
    return undefined;
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
  try {
    return typeof handle === 'object' && handle !== null &&
      (handle as { kind?: unknown }).kind === 'raw-image';
  } catch {
    return false;
  }
}

function isPixelDataLike(handle: unknown): boolean {
  if (typeof handle !== 'object' || handle === null) return false;
  try {
    const h = handle as { width?: unknown; height?: unknown; data?: unknown };
    return typeof h.width === 'number' && typeof h.height === 'number' && isArrayLikeData(h.data);
  } catch {
    return false;
  }
}

function isDataTextureLike(handle: unknown): boolean {
  if (typeof handle !== 'object' || handle === null) return false;
  try {
    const image = (handle as { image?: unknown }).image;
    if (typeof image !== 'object' || image === null) return false;
    const h = image as { width?: unknown; height?: unknown; data?: unknown };
    return typeof h.width === 'number' && typeof h.height === 'number' && isArrayLikeData(h.data);
  } catch {
    return false;
  }
}

function isImageBitmapLike(handle: unknown): boolean {
  try {
    if (typeof handle !== 'object' || handle === null) return false;
    const h = handle as { width?: unknown; height?: unknown; close?: unknown };
    const width = h.width;
    const height = h.height;
    const hasValidDimensions =
      Number.isSafeInteger(width) &&
      (width as number) > 0 &&
      Number.isSafeInteger(height) &&
      (height as number) > 0;
    if (!hasValidDimensions) return false;
    if (typeof ImageBitmap !== 'undefined' && handle instanceof ImageBitmap) return true;
    return typeof h.close === 'function';
  } catch {
    return false;
  }
}

export function isArrayLikeData(data: unknown): boolean {
  return inspectPixelData(data).valid;
}

export function textureDecodeHint(handle: unknown): {
  readonly originalWidth?: number;
  readonly originalHeight?: number;
  readonly maxTextureSize?: number;
} | null {
  try {
    if (!isRecord(handle) || !isRecord(handle.__vitrum_hint__)) return null;
    const hint = handle.__vitrum_hint__;
    return {
      ...(typeof hint.originalWidth === 'number' ? { originalWidth: hint.originalWidth } : {}),
      ...(typeof hint.originalHeight === 'number' ? { originalHeight: hint.originalHeight } : {}),
      ...(typeof hint.maxTextureSize === 'number' ? { maxTextureSize: hint.maxTextureSize } : {}),
    };
  } catch {
    return null;
  }
}

// D15-6: buildTextureDecodeReport now lives in textureDecodeReport.ts; re-export it
// from here so the historical `./texturePipeline.js` import path keeps working.
export { buildTextureDecodeReport } from './textureDecodeReport.js';
