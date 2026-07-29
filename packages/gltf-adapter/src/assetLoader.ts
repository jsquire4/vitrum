// assetLoader.ts — turnkey glTF/GLB loading into Vitrum's core Scene contract.
//
// `gltfToScene()` remains the low-level converter for hosts that already own
// resource resolution. `loadGltfAsset()` is the predictable higher-level path:
// it accepts URL/string/ArrayBuffer/GltfJson, resolves external buffers/images,
// returns a structured feature report, and ranks the shipping backends against
// the asset's actual feature use.

import type { MaterialSpec, MeshPrimitive, Scene, ScenePrimitive } from '@vitrum/core';
import type { GltfJson } from './gltfTypes.js';
import {
  assertSupportedGltfVersion,
  gltfToSceneWithResourceContext,
  type GltfToSceneOptions,
  type GltfToSceneResult,
} from './gltfToScene.js';
import { decodeGltfUtf8, parseGlb } from './glbParser.js';
import {
  effectiveGltfTextureSourceExtensions,
  type DecodeImageFn,
  type GltfImageBytes,
  type RawImageHandle,
} from './textures.js';
import {
  analyzeGltfAsset,
  rankGltfBackends,
  type GltfBackendCompatibility,
  type GltfBackendPolicy,
  type GltfFeatureReport,
} from './featureReport.js';
import {
  buildTextureDecodeReport,
  createDecodeSceneTexturesContext,
  decodeSceneTexturesWithContext,
  MATERIAL_TEXTURE_FIELDS,
  type DecodeGltfTexturePixelsFn,
  type DecodeSceneTexturesContext,
  type DecodeSceneTextureDiagnostic,
  type DecodeSceneTexturesOptions,
  type GltfTextureDecodeReport,
} from './texturePipeline.js';
import {
  GltfAdapterError,
  GltfFetchFailed,
  GltfParseFailed,
  GltfResourceDecodeFailed,
  GltfResourceNotFound,
  type GltfAssetResourceKind,
  type GltfResourceDecodeFailureReason,
} from './errors.js';
import {
  attachGltfResourceOwner,
  DecodedImageHandleOwner,
  GLTF_INPUT_RESOURCE_KEY,
  ImportResourceLedger,
  createAsyncResourceLimiter,
  gltfArrayBufferByteLength,
  gltfBufferResourceKey,
  gltfImageResourceKey,
  GltfResourceLimitError,
  normalizeGltfImportResourceLimits,
  type GltfImportResourceContext,
} from './importResourceBudget.js';
import { localUint8ArrayView } from './intrinsicTypedArrays.js';
import { collectGltfSceneReachability } from './sceneScope.js';
import {
  bakePtWebgpuLiteCompatibleVertexColors,
  hasReachableMaterialPointerAnimationForColoredPrimitive,
  reconcileBackendCompatibilityAfterSceneImport,
  reconcileBackendCompatibilityAfterTextureDecode,
  rerankBackendCompatibility,
} from './backendCompatibilityReconcile.js';

const MAX_RESOURCE_STREAM_CHUNKS = 65_536;

export type GltfAssetInput = string | URL | ArrayBuffer | GltfJson;

export interface GltfAssetFetchResponse {
  readonly ok?: boolean;
  readonly status?: number;
  readonly statusText?: string;
  readonly headers?: { get(name: string): string | null };
  readonly body?: GltfAssetReadableStream | null;
  arrayBuffer(): Promise<ArrayBuffer>;
}

export interface GltfAssetReadableStreamReadResult {
  readonly done: boolean;
  readonly value?: Uint8Array;
}

export interface GltfAssetReadableStreamReader {
  read(): Promise<GltfAssetReadableStreamReadResult>;
  cancel?(reason?: unknown): void | Promise<void>;
  releaseLock?(): void;
}

export interface GltfAssetReadableStream {
  getReader(): GltfAssetReadableStreamReader;
  cancel?(reason?: unknown): void | Promise<void>;
}

export type GltfAssetFetch = (
  url: string,
  init?: { readonly signal?: AbortSignal },
) => Promise<GltfAssetFetchResponse>;

export interface GltfAssetCacheKey {
  readonly url: string;
  readonly kind: GltfAssetResourceKind;
}

/**
 * Complete reusable fetch result. Image content type is part of the cache
 * identity because it participates in decoder selection when a glTF image has
 * neither an authored mimeType nor an identifying URI suffix.
 */
export interface GltfAssetCacheEntry {
  readonly data: ArrayBuffer;
  readonly mimeType?: string;
}

export interface GltfAssetCache {
  /**
   * Legacy byte-only cache hooks. These remain source-compatible, but cannot
   * preserve an HTTP Content-Type for an extensionless image. On a legacy hit,
   * image MIME falls back to the authored glTF mimeType or URI suffix exactly
   * as it did before metadata-aware caching.
   */
  get(key: GltfAssetCacheKey): ArrayBuffer | undefined | Promise<ArrayBuffer | undefined>;
  set(key: GltfAssetCacheKey, data: ArrayBuffer): void | Promise<void>;
  /**
   * Optional metadata-aware hooks. Implement both methods to preserve decoder
   * selection exactly across cold and warm loads. When both are present the
   * loader uses them instead of the legacy byte-only hooks.
   */
  getEntry?(
    key: GltfAssetCacheKey,
  ): GltfAssetCacheEntry | undefined | Promise<GltfAssetCacheEntry | undefined>;
  setEntry?(key: GltfAssetCacheKey, entry: GltfAssetCacheEntry): void | Promise<void>;
}

export interface LoadGltfAssetOptions extends GltfToSceneOptions {
  readonly baseUri?: string | URL;
  readonly fetch?: GltfAssetFetch;
  readonly signal?: AbortSignal;
  readonly backendPolicy?: GltfBackendPolicy;
  readonly cache?: GltfAssetCache;
  /**
   * Optional static compatibility preflight, invoked after parsing/resource
   * resolution and before scene conversion. The high-level engine bridge uses
   * this to reject known strict-mode blockers with a canonical compatibility
   * error before lower-level import code can throw a raw conversion failure.
   */
  readonly compatibilityPreflight?: (preflight: GltfAssetCompatibilityPreflight) => void;
}

export interface GltfAssetResult extends GltfToSceneResult {
  readonly gltf: GltfJson;
  readonly sceneIndex: number;
  readonly featureReport: GltfFeatureReport;
  readonly backendCompatibility: readonly GltfBackendCompatibility[];
  readonly recommendedBackend: GltfBackendCompatibility;
  readonly textureDecodeReport: GltfTextureDecodeReport;
}

export interface GltfAssetCompatibilityPreflight {
  readonly gltf: GltfJson;
  readonly sceneIndex: number;
  readonly featureReport: GltfFeatureReport;
  readonly backendCompatibility: readonly GltfBackendCompatibility[];
  readonly recommendedBackend: GltfBackendCompatibility;
}

export interface LoadGltfAndDecodeTexturesOptions extends LoadGltfAssetOptions {
  readonly textureTarget?: DecodeSceneTexturesOptions['target'];
  readonly decodePixels?: DecodeGltfTexturePixelsFn;
  readonly maxTextureSize?: number;
  readonly maxDecodedTexturePixels?: number;
  readonly maxTotalDecodedTexturePixels?: number;
  readonly maxImageDecodeConcurrency?: number;
  readonly warnOnNpotRepeatWrap?: boolean;
  readonly npotRepeatWrapPolicy?: DecodeSceneTexturesOptions['npotRepeatWrapPolicy'];
  /**
   * Optional policy hook invoked after glTF preflight/import has built the
   * initial asset report, but before scene texture handles are decoded. Engine
   * bridges use this to add runtime-adapter decode limits once the selected or
   * recommended backend is known, without doing a second load.
   */
  readonly configureTextureDecode?: ConfigureGltfTextureDecodeOptions;
  readonly onTextureDiagnostic?: (diagnostic: DecodeSceneTextureDiagnostic) => void;
  readonly onTextureWarning?: (message: string) => void;
}

export interface GltfTextureDecodePolicyContext {
  readonly asset: GltfAssetResult;
  readonly decodeOptions: DecodeSceneTexturesOptions;
}

export type ConfigureGltfTextureDecodeOptions = (
  context: GltfTextureDecodePolicyContext,
) =>
  | Partial<DecodeSceneTexturesOptions>
  | void
  | Promise<Partial<DecodeSceneTexturesOptions> | void>;

export interface GltfDecodedAssetResult extends GltfAssetResult {
  readonly decodedTextureCount: number;
  readonly unchangedTextureCount: number;
  readonly textureDecodeDiagnostics: readonly DecodeSceneTextureDiagnostic[];
  readonly textureDecodeWarnings: readonly string[];
}

interface ParsedInput {
  readonly gltf: GltfJson;
  readonly baseUri?: string;
  readonly buffers: Map<number, ArrayBuffer>;
}

export async function loadGltfAsset(
  input: GltfAssetInput,
  options: LoadGltfAssetOptions = {},
): Promise<GltfAssetResult> {
  const resourceContext = createImportResourceContext(options);
  try {
    const asset = await loadGltfAssetWithResourceContext(
      input,
      options,
      resourceContext,
    );
    return attachGltfResourceOwner(asset, resourceContext.decodedImageHandles);
  } catch (error) {
    resourceContext.decodedImageHandles.rollback();
    throw error;
  }
}

async function loadGltfAssetWithResourceContext(
  input: GltfAssetInput,
  options: LoadGltfAssetOptions,
  resourceContext: GltfImportResourceContext,
): Promise<GltfAssetResult> {
  const buffers = normalizeBufferMap(options.buffers, resourceContext);
  const parsed = await parseInput(
    input,
    options,
    resourceContext,
    buffers.has(0),
  );
  // Reject incompatible JSON before scene traversal, compatibility callbacks,
  // or any dependent external-resource fetch. The low-level converter repeats
  // this check at its own public boundary.
  assertSupportedGltfVersion(parsed.gltf);
  for (const [index, buffer] of parsed.buffers) {
    chargeArrayBuffer(
      resourceContext,
      buffers.has(index)
        ? `${gltfBufferResourceKey(index)}:unused-glb-copy`
        : gltfBufferResourceKey(index),
      buffer,
      `buffers[${index}]`,
    );
    if (!buffers.has(index)) buffers.set(index, buffer);
  }

  const backendPolicy = options.backendPolicy ?? 'fidelity';
  const sceneIndex = options.sceneIndex ?? parsed.gltf.scene ?? 0;
  const textureSourceExtensions =
    effectiveGltfTextureSourceExtensions(options.textureSourceExtensions);
  const sceneReachability = collectGltfSceneReachability(
    parsed.gltf,
    sceneIndex,
    textureSourceExtensions,
  );

  await resolveExternalBuffers(
    parsed.gltf,
    buffers,
    parsed.baseUri,
    options,
    resourceContext,
    sceneReachability.bufferIndices,
  );
  const imageBytes = await resolveExternalImages(
    parsed.gltf,
    parsed.baseUri,
    options,
    resourceContext,
    sceneReachability.imageIndices,
  );

  const featureReport = analyzeGltfAsset(parsed.gltf, {
    textureSourceExtensions,
    sceneIndex,
  });
  const staticBackendCompatibility = rankGltfBackends(featureReport, backendPolicy);
  options.compatibilityPreflight?.({
    gltf: parsed.gltf,
    sceneIndex,
    featureReport,
    backendCompatibility: staticBackendCompatibility,
    recommendedBackend: staticBackendCompatibility[0]!,
  });
  const sceneOptions: GltfToSceneOptions = {
    buffers,
    imageBytes,
    ...(options.decodeImage ? { decodeImage: options.decodeImage } : {}),
    sceneIndex,
    ...(options.dracoDecode ? { dracoDecode: options.dracoDecode } : {}),
    ...(options.meshoptDecode ? { meshoptDecode: options.meshoptDecode } : {}),
    ...(options.compressionDecoderPolicy !== undefined
      ? { compressionDecoderPolicy: options.compressionDecoderPolicy }
      : {}),
    textureSourceExtensions,
    ...(options.materialVariant !== undefined ? { materialVariant: options.materialVariant } : {}),
    ...(options.pointLineFallbackRadius !== undefined ? { pointLineFallbackRadius: options.pointLineFallbackRadius } : {}),
  };
  const sceneResult = await gltfToSceneWithResourceContext(
    parsed.gltf,
    sceneOptions,
    resourceContext,
  );
  const canBakeLiteVertexColors =
    (sceneResult.materialVariantBindings?.length ?? 0) === 0 &&
    !hasReachableMaterialPointerAnimationForColoredPrimitive(parsed.gltf, sceneReachability);
  const scene = canBakeLiteVertexColors
    ? bakePtWebgpuLiteCompatibleVertexColors(sceneResult.scene)
    : sceneResult.scene;
  const nodeVisibilityScene = sceneResult.nodeVisibilityPrimitives === undefined
    ? undefined
    : {
        ...sceneResult.scene,
        primitives: sceneResult.nodeVisibilityPrimitives,
        emitters: sceneResult.nodeVisibilityEmitters ?? sceneResult.scene.emitters,
      };
  const bakedNodeVisibilityScene = nodeVisibilityScene === undefined
    ? undefined
    : canBakeLiteVertexColors
      ? bakePtWebgpuLiteCompatibleVertexColors(nodeVisibilityScene)
      : nodeVisibilityScene;
  // Compatibility must cover the retained visibility inventory, not merely
  // the currently visible subset. A controller may restore any of these
  // primitives after backend selection without re-running the planner.
  const compatibilityScene = bakedNodeVisibilityScene ?? scene;
  const convertedMaterials = sceneResult.convertedMaterials;
  const sceneWithMaterialTable = convertedMaterials === undefined
    ? scene
    : appendInactiveMaterialPrimitives(
      scene,
      convertedMaterials,
      materialIndicesForSelectedScene(sceneResult.materialVariantBindings),
    );
  const textureDecodeReport = buildTextureDecodeReport(sceneWithMaterialTable);
  const backendCompatibility = rerankBackendCompatibility(
    reconcileBackendCompatibilityAfterSceneImport(
      staticBackendCompatibility,
      compatibilityScene,
      textureDecodeReport,
      canBakeLiteVertexColors,
    ),
    backendPolicy,
  );

  return {
    ...sceneResult,
    scene,
    ...(bakedNodeVisibilityScene !== undefined
      ? { nodeVisibilityPrimitives: bakedNodeVisibilityScene.primitives }
      : {}),
    ...(convertedMaterials !== undefined ? { convertedMaterials } : {}),
    gltf: parsed.gltf,
    sceneIndex,
    featureReport,
    backendCompatibility,
    recommendedBackend: backendCompatibility[0]!,
    textureDecodeReport,
  };
}

export async function loadGltfAndDecodeTextures(
  input: GltfAssetInput,
  options: LoadGltfAndDecodeTexturesOptions = {},
): Promise<GltfDecodedAssetResult> {
  const resourceContext = createImportResourceContext(options);
  try {
    const asset = await loadGltfAndDecodeTexturesWithResourceContext(
      input,
      options,
      resourceContext,
    );
    resourceContext.decodedImageHandles.retainOnly(
      decodedImageHandlesReachableFromAsset(asset),
    );
    return attachGltfResourceOwner(asset, resourceContext.decodedImageHandles);
  } catch (error) {
    resourceContext.decodedImageHandles.rollback();
    throw error;
  }
}

async function loadGltfAndDecodeTexturesWithResourceContext(
  input: GltfAssetInput,
  options: LoadGltfAndDecodeTexturesOptions,
  resourceContext: GltfImportResourceContext,
): Promise<GltfDecodedAssetResult> {
  const asset = await loadGltfAssetWithResourceContext(
    input,
    loadOptionsForTextureDecode(options),
    resourceContext,
  );
  let decodeOptions: DecodeSceneTexturesOptions = {
    target: options.textureTarget ?? 'cpu-linear',
    ...(options.decodePixels ? { decodePixels: options.decodePixels } : {}),
    ...(options.maxTextureSize !== undefined ? { maxTextureSize: options.maxTextureSize } : {}),
    ...(options.resourceLimits !== undefined ? { resourceLimits: options.resourceLimits } : {}),
    ...(options.maxDecodedTexturePixels !== undefined
      ? { maxDecodedTexturePixels: options.maxDecodedTexturePixels }
      : {}),
    ...(options.maxTotalDecodedTexturePixels !== undefined
      ? { maxTotalDecodedTexturePixels: options.maxTotalDecodedTexturePixels }
      : {}),
    ...(options.maxImageDecodeConcurrency !== undefined
      ? { maxImageDecodeConcurrency: options.maxImageDecodeConcurrency }
      : {}),
    ...(options.warnOnNpotRepeatWrap !== undefined ? { warnOnNpotRepeatWrap: options.warnOnNpotRepeatWrap } : {}),
    ...(options.npotRepeatWrapPolicy !== undefined ? { npotRepeatWrapPolicy: options.npotRepeatWrapPolicy } : {}),
    ...(options.onTextureDiagnostic ? { onDiagnostic: options.onTextureDiagnostic } : {}),
    ...(options.onTextureWarning ? { onWarning: options.onTextureWarning } : {}),
  };
  const configuredDecodeOptions = await options.configureTextureDecode?.({ asset, decodeOptions });
  validateConfiguredTextureDecodeOptions(configuredDecodeOptions);
  if (configuredDecodeOptions !== undefined) {
    decodeOptions = { ...decodeOptions, ...configuredDecodeOptions };
  }
  const decodeContext = createDecodeSceneTexturesContext(
    decodeOptions,
    resourceContext,
  );
  const decodeSourceScene =
    asset.nodeVisibilityPrimitives === undefined
      ? asset.scene
      : {
        ...asset.scene,
        primitives: asset.nodeVisibilityPrimitives,
        emitters: asset.nodeVisibilityEmitters ?? asset.scene.emitters,
      };
  const decoded = await decodeSceneTexturesWithContext(decodeSourceScene, decodeContext);
  const decodedScene = decodedVisibleScene(asset.scene, decodeSourceScene, decoded.scene);
  const convertedMaterials = asset.convertedMaterials === undefined
    ? undefined
    : await decodeConvertedMaterials(
      asset.convertedMaterials,
      decodeSourceScene,
      decoded.scene,
      decodeContext,
      materialIndicesForSelectedScene(asset.materialVariantBindings),
    );
  const sceneWithMaterialTable = convertedMaterials === undefined
    ? decodedScene
    : appendInactiveMaterialPrimitives(
      decodedScene,
      convertedMaterials.materials,
      materialIndicesForSelectedScene(asset.materialVariantBindings),
    );
  const textureDecodeReport = buildTextureDecodeReport(sceneWithMaterialTable);
  const textureDecodeDiagnostics = [
    ...decoded.diagnostics,
    ...(convertedMaterials?.diagnostics ?? []),
  ];
  const textureDecodeWarnings = [
    ...decoded.warnings,
    ...(convertedMaterials?.warnings ?? []),
  ];
  const backendCompatibility = reconcileBackendCompatibilityAfterTextureDecode(
    asset.backendCompatibility,
    textureDecodeReport,
    textureDecodeDiagnostics,
    asset.featureReport,
    options.backendPolicy ?? 'fidelity',
  );
  return {
    ...asset,
    scene: decodedScene,
    ...(asset.nodeVisibilityPrimitives !== undefined
      ? { nodeVisibilityPrimitives: decoded.scene.primitives }
      : {}),
    ...(convertedMaterials !== undefined ? { convertedMaterials: convertedMaterials.materials } : {}),
    backendCompatibility,
    recommendedBackend: backendCompatibility[0]!,
    textureDecodeReport,
    decodedTextureCount: decoded.decodedCount + (convertedMaterials?.decodedCount ?? 0),
    unchangedTextureCount: decoded.unchangedCount + (convertedMaterials?.unchangedCount ?? 0),
    textureDecodeDiagnostics,
    textureDecodeWarnings,
  };
}

function decodedVisibleScene(
  visibleScene: Scene,
  sourceInventory: Scene,
  decodedInventory: Scene,
): Scene {
  const sourceById = new Map(
    sourceInventory.primitives.map((primitive) => [String(primitive.id), primitive]),
  );
  const decodedById = new Map(
    decodedInventory.primitives.map((primitive) => [String(primitive.id), primitive]),
  );
  return {
    ...visibleScene,
    primitives: visibleScene.primitives.map((primitive) => {
      const source = sourceById.get(String(primitive.id));
      const decoded = decodedById.get(String(primitive.id));
      if (source === undefined || decoded === undefined) return primitive;
      let material: MaterialSpec | undefined;
      for (const field of MATERIAL_TEXTURE_FIELDS) {
        if (decoded.material[field] === source.material[field]) continue;
        material ??= { ...primitive.material };
        (material as unknown as Record<string, unknown>)[field] = decoded.material[field];
      }
      return material === undefined ? primitive : { ...primitive, material };
    }),
  };
}

function* decodedImageHandlesReachableFromAsset(
  asset: GltfDecodedAssetResult,
): Iterable<unknown> {
  const materials = new Set<MaterialSpec>();
  for (const primitive of asset.scene.primitives) materials.add(primitive.material);
  for (const primitive of asset.nodeVisibilityPrimitives ?? []) {
    materials.add(primitive.material);
  }
  for (const material of asset.convertedMaterials ?? []) materials.add(material);
  for (const material of materials) {
    for (const field of MATERIAL_TEXTURE_FIELDS) {
      const ref = material[field];
      if (ref !== undefined) yield ref.handle;
    }
    if (material.frontLayer?.normalMap !== undefined) {
      yield material.frontLayer.normalMap.handle;
    }
    if (material.backLayer?.normalMap !== undefined) {
      yield material.backLayer.normalMap.handle;
    }
  }
}

function loadOptionsForTextureDecode(
  options: LoadGltfAndDecodeTexturesOptions,
): LoadGltfAssetOptions {
  if (options.decodeImage !== undefined) {
    return options;
  }

  return {
    ...options,
    decodeImage: preserveRawImageForTextureDecode,
  };
}

const preserveRawImageForTextureDecode: DecodeImageFn = (
  data,
  mimeType,
): Promise<RawImageHandle> => Promise.resolve({
  kind: 'raw-image',
  mimeType,
  data,
});


interface DecodeConvertedMaterialsResult {
  readonly materials: readonly MaterialSpec[];
  readonly decodedCount: number;
  readonly unchangedCount: number;
  readonly diagnostics: readonly DecodeSceneTextureDiagnostic[];
  readonly warnings: readonly string[];
}

async function decodeConvertedMaterials(
  materials: readonly MaterialSpec[],
  originalScene: Scene,
  decodedScene: Scene,
  decodeContext: DecodeSceneTexturesContext,
  materialIndices: ReadonlySet<number>,
): Promise<DecodeConvertedMaterialsResult> {
  const activeMaterialMap = new Map<MaterialSpec, MaterialSpec>();
  for (let i = 0; i < originalScene.primitives.length; i += 1) {
    const original = originalScene.primitives[i];
    const decoded = decodedScene.primitives[i];
    if (original !== undefined && decoded !== undefined) {
      activeMaterialMap.set(materialForPrimitive(original), materialForPrimitive(decoded));
    }
  }

  const decodedMaterials = materials.slice();
  const pending: Array<{ index: number; material: MaterialSpec }> = [];
  for (let i = 0; i < materials.length; i += 1) {
    const material = materials[i]!;
    const activeDecoded = activeMaterialMap.get(material);
    if (activeDecoded !== undefined) {
      decodedMaterials[i] = activeDecoded;
    } else if (materialIndices.has(i)) {
      pending.push({ index: i, material });
    }
  }

  if (pending.length === 0) {
    return { materials: decodedMaterials, decodedCount: 0, unchangedCount: 0, diagnostics: [], warnings: [] };
  }

  const decoded = await decodeSceneTexturesWithContext(
    materialsToSyntheticScene(pending),
    decodeContext,
  );
  for (let i = 0; i < pending.length; i += 1) {
    const primitive = decoded.scene.primitives[i];
    if (primitive !== undefined) {
      decodedMaterials[pending[i]!.index] = materialForPrimitive(primitive);
    }
  }

  return {
    materials: decodedMaterials,
    decodedCount: decoded.decodedCount,
    unchangedCount: decoded.unchangedCount,
    diagnostics: decoded.diagnostics,
    warnings: decoded.warnings,
  };
}

function appendInactiveMaterialPrimitives(
  scene: Scene,
  materials: readonly MaterialSpec[],
  materialIndices: ReadonlySet<number> | undefined,
): Scene {
  const activeMaterials = new Set(scene.primitives.map((primitive) => materialForPrimitive(primitive)));
  const inactive = materials
    .map((material, index) => ({ index, material }))
    .filter(({ index }) => materialIndices === undefined || materialIndices.has(index))
    .filter(({ material }) => !activeMaterials.has(material));
  if (inactive.length === 0) return scene;
  const synthetic = materialsToSyntheticScene(inactive);
  return {
    ...scene,
    primitives: [...scene.primitives, ...synthetic.primitives],
  };
}

function materialIndicesForSelectedScene(
  bindings: GltfAssetResult['materialVariantBindings'],
): ReadonlySet<number> {
  if (bindings === undefined) return new Set();
  const indices = new Set<number>();
  for (const binding of bindings) {
    if (binding.baseMaterialIndex !== undefined) indices.add(binding.baseMaterialIndex);
    if (binding.basePatch?.materialIndex !== undefined) indices.add(binding.basePatch.materialIndex);
    for (const variantPatch of binding.variantPatches ?? []) {
      if (variantPatch.patch.materialIndex !== undefined) indices.add(variantPatch.patch.materialIndex);
    }
  }
  return indices;
}

function materialsToSyntheticScene(
  materials: readonly { readonly index: number; readonly material: MaterialSpec }[],
): Scene {
  return {
    primitives: materials.map(({ index, material }) => ({
      kind: 'mesh',
      id: `gltf-material-${index}`,
      positions: new Float32Array(),
      normals: new Float32Array(),
      material,
    } satisfies MeshPrimitive)),
    emitters: [],
    environment: { kind: 'none' },
  };
}

function materialForPrimitive(primitive: ScenePrimitive): MaterialSpec {
  return primitive.material;
}

function createImportResourceContext(
  options: Pick<GltfToSceneOptions, 'resourceLimits'> & {
    readonly maxDecodedTexturePixels?: number;
    readonly maxTotalDecodedTexturePixels?: number;
    readonly maxImageDecodeConcurrency?: number;
  },
): GltfImportResourceContext {
  const inheritedLimits = normalizeGltfImportResourceLimits(
    options.resourceLimits,
  );
  const ledger = new ImportResourceLedger({
    ...inheritedLimits,
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
  return {
    ledger,
    limiter: createAsyncResourceLimiter(
      ledger.limits.maxConcurrentResourceOperations,
    ),
    decodedImageHandles: new DecodedImageHandleOwner(),
  };
}

function validateConfiguredTextureDecodeOptions(
  value: unknown,
): void {
  if (
    value !== undefined &&
    (
      value === null ||
      typeof value !== 'object' ||
      Array.isArray(value)
    )
  ) {
    throw new TypeError(
      '[vitrum/gltf-adapter] configureTextureDecode must return an options object or undefined.',
    );
  }
}

async function parseInput(
  input: GltfAssetInput,
  options: LoadGltfAssetOptions,
  resourceContext: GltfImportResourceContext,
  hasBufferZeroOverride: boolean,
): Promise<ParsedInput> {
  const inputUrl = gltfAssetInputUrl(input);
  if (inputUrl !== undefined) {
    const url = resolveUri(inputUrl, options.baseUri, 'asset');
    const bytes = await fetchArrayBuffer(
      url,
      options,
      'asset',
      'input',
      GLTF_INPUT_RESOURCE_KEY,
      resourceContext,
    );
    return parseArrayBufferInput(
      bytes,
      directoryUrl(url),
      resourceContext,
      hasBufferZeroOverride,
    );
  }
  const inputByteLength = gltfArrayBufferByteLength(input);
  if (inputByteLength !== undefined) {
    resourceContext.ledger.chargeEncodedBytes(
      GLTF_INPUT_RESOURCE_KEY,
      inputByteLength,
      'input',
    );
    return parseArrayBufferInput(
      input as ArrayBuffer,
      options.baseUri ? String(options.baseUri) : undefined,
      resourceContext,
      hasBufferZeroOverride,
    );
  }
  return {
    gltf: input as GltfJson,
    ...(options.baseUri ? { baseUri: String(options.baseUri) } : {}),
    buffers: new Map(),
  };
}

function gltfAssetInputUrl(input: GltfAssetInput): string | undefined {
  if (typeof input === 'string') return input;
  if (typeof URL === 'undefined') return undefined;
  try {
    return URL.prototype.toString.call(input);
  } catch {
    return undefined;
  }
}

function parseArrayBufferInput(
  buffer: ArrayBuffer,
  baseUri: string | undefined,
  resourceContext: GltfImportResourceContext,
  hasBufferZeroOverride: boolean,
): ParsedInput {
  if (isGlb(buffer)) {
    const binResourceKey = hasBufferZeroOverride
      ? `${gltfBufferResourceKey(0)}:unused-glb-copy`
      : gltfBufferResourceKey(0);
    const glb = parseGlb(buffer, {
      beforeBinChunkCopy: ({ byteLength }) => {
        resourceContext.ledger.ensureEncodedBytes(
          binResourceKey,
          byteLength,
          'buffers[0]',
        );
      },
    });
    const buffers = new Map<number, ArrayBuffer>();
    if (glb.binChunk) buffers.set(0, glb.binChunk);
    return { gltf: glb.json, ...(baseUri ? { baseUri } : {}), buffers };
  }
  try {
    const text = decodeGltfUtf8(buffer);
    return { gltf: JSON.parse(text) as GltfJson, ...(baseUri ? { baseUri } : {}), buffers: new Map() };
  } catch (cause) {
    throw new GltfParseFailed({
      format: 'gltf-json',
      reason: 'json-parse-failed',
      message: '[vitrum/gltf-adapter] glTF JSON asset is not valid JSON.',
      cause,
    });
  }
}

async function resolveExternalBuffers(
  gltf: GltfJson,
  buffers: Map<number, ArrayBuffer>,
  baseUri: string | undefined,
  options: LoadGltfAssetOptions,
  resourceContext: GltfImportResourceContext,
  bufferIndices?: ReadonlySet<number>,
): Promise<void> {
  const pending: Array<() => Promise<void>> = [];
  for (const [index, buffer] of (gltf.buffers ?? []).entries()) {
    if (bufferIndices !== undefined && !bufferIndices.has(index)) continue;
    if (buffers.has(index)) continue;
    if (buffer.uri == null) continue;
    const sourcePath = `buffers[${index}].uri`;
    const resourceKey = gltfBufferResourceKey(index);
    if (buffer.uri.startsWith('data:')) {
      buffers.set(
        index,
        uint8ToArrayBuffer(
          decodeDataUri(
            buffer.uri,
            'buffer',
            `buffer ${index}`,
            resourceKey,
            resourceContext,
            sourcePath,
          ),
        ),
      );
      continue;
    }
    const url = resolveUri(buffer.uri, baseUri, 'buffer', sourcePath);
    pending.push(async () => {
      const data = await fetchArrayBuffer(
        url,
        options,
        'buffer',
        sourcePath,
        resourceKey,
        resourceContext,
      );
      buffers.set(index, data);
    });
  }
  await awaitAllResourceOperations(pending.map((operation) => operation()));
}

async function resolveExternalImages(
  gltf: GltfJson,
  baseUri: string | undefined,
  options: LoadGltfAssetOptions,
  resourceContext: GltfImportResourceContext,
  imageIndices?: ReadonlySet<number>,
): Promise<Map<number, GltfImageBytes>> {
  const out = normalizePreloadedImageBytes(
    options.imageBytes,
    resourceContext,
    imageIndices,
  );
  const pending: Array<() => Promise<void>> = [];
  for (const [index, image] of (gltf.images ?? []).entries()) {
    if (imageIndices !== undefined && !imageIndices.has(index)) continue;
    if (image.bufferView !== undefined) continue;
    if (image.uri == null || image.uri.startsWith('data:')) continue;
    if (out.has(index)) continue;
    const sourcePath = `images[${index}].uri`;
    const url = resolveUri(image.uri, baseUri, 'image', sourcePath);
    pending.push(async () => {
      const fetched = await fetchImageBytes(
        url,
        options,
        gltfImageResourceKey(index),
        resourceContext,
        sourcePath,
        image.mimeType,
      );
      out.set(index, {
        bytes: fetched.bytes,
        mimeType: image.mimeType ?? fetched.mimeType ?? inferMimeType(image.uri!),
      });
    });
  }
  await awaitAllResourceOperations(pending.map((operation) => operation()));
  return out;
}

async function fetchImageBytes(
  url: string,
  options: LoadGltfAssetOptions,
  resourceKey: string,
  resourceContext: GltfImportResourceContext,
  sourcePath: string,
  authoredMimeType?: string,
): Promise<{ readonly bytes: Uint8Array; readonly mimeType?: string }> {
  const fetched = await acquireResourceBytes(
    url,
    options,
    'image',
    resourceKey,
    resourceContext,
    sourcePath,
    authoredMimeType,
  );
  const bytes = new Uint8Array(fetched.data);
  const mimeType = fetched.mimeType;
  return mimeType === undefined
    ? { bytes }
    : { bytes, mimeType };
}

async function fetchArrayBuffer(
  url: string,
  options: LoadGltfAssetOptions,
  kind: GltfAssetResourceKind,
  sourcePath: string,
  resourceKey: string,
  resourceContext: GltfImportResourceContext,
): Promise<ArrayBuffer> {
  return (
    await acquireResourceBytes(
      url,
      options,
      kind,
      resourceKey,
      resourceContext,
      sourcePath,
    )
  ).data;
}

interface AcquiredResourceBytes {
  readonly data: ArrayBuffer;
  readonly mimeType?: string;
}

async function awaitAllResourceOperations(
  operations: readonly Promise<void>[],
): Promise<void> {
  const settled = await Promise.allSettled(operations);
  for (const result of settled) {
    if (result.status === 'rejected') throw result.reason;
  }
}

async function acquireResourceBytes(
  url: string,
  options: LoadGltfAssetOptions,
  kind: GltfAssetResourceKind,
  resourceKey: string,
  resourceContext: GltfImportResourceContext,
  sourcePath: string,
  authoredImageMimeType?: string,
): Promise<AcquiredResourceBytes> {
  return resourceContext.limiter.run(async () => {
    const cacheKey = { url, kind } satisfies GltfAssetCacheKey;
    const cache = options.cache;
    const metadataAwareCache =
      typeof cache?.getEntry === 'function' &&
      typeof cache.setEntry === 'function'
        ? cache as GltfAssetCache & Required<Pick<GltfAssetCache, 'getEntry' | 'setEntry'>>
        : undefined;
    if (metadataAwareCache !== undefined) {
      const cachedEntry = await metadataAwareCache.getEntry(cacheKey);
      if (cachedEntry !== undefined) {
        const entry = normalizeAssetCacheEntry(cachedEntry, sourcePath);
        chargeArrayBuffer(resourceContext, resourceKey, entry.data, sourcePath);
        return entry;
      }
    } else {
      const cachedData = await cache?.get(cacheKey);
      if (cachedData !== undefined) {
        chargeArrayBuffer(resourceContext, resourceKey, cachedData, sourcePath);
        if (kind !== 'image') return { data: cachedData };

        // A byte-only cache has no response Content-Type. Recover the MIME
        // deterministically from the authored glTF declaration, a supported
        // container signature, or the URL suffix. If none is available, bypass
        // this ambiguous image hit and fetch again: reusing the bytes with the
        // old image/png fallback could select a different decoder than the cold
        // request.
        const cachedMimeType =
          authoredImageMimeType ?? inferCachedImageMimeType(cachedData, url);
        if (cachedMimeType !== undefined) {
          return { data: cachedData, mimeType: cachedMimeType };
        }
      }
    }

    const response = await fetchResource(url, options, kind, sourcePath);
    const data = await readBoundedResponseBody(
      response,
      url,
      kind,
      resourceKey,
      sourcePath,
      resourceContext,
    );
    const mimeType = safeResponseHeader(response, 'content-type');
    const entry = mimeType === undefined ? { data } : { data, mimeType };
    if (metadataAwareCache !== undefined) {
      await metadataAwareCache.setEntry(cacheKey, entry);
    } else {
      await cache?.set(cacheKey, data);
    }
    return entry;
  });
}

function normalizeAssetCacheEntry(
  value: unknown,
  sourcePath: string,
): GltfAssetCacheEntry {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) {
    throw new TypeError(
      `[vitrum/gltf-adapter] Cache entry for ${sourcePath} must be an object with an ArrayBuffer data field.`,
    );
  }
  let data: unknown;
  let mimeType: unknown;
  try {
    const candidate = value as {
      readonly data?: unknown;
      readonly mimeType?: unknown;
    };
    data = candidate.data;
    mimeType = candidate.mimeType;
  } catch {
    throw new TypeError(
      `[vitrum/gltf-adapter] Cache entry for ${sourcePath} could not be inspected.`,
    );
  }
  if (gltfArrayBufferByteLength(data) === undefined) {
    throw new TypeError(
      `[vitrum/gltf-adapter] Cache entry for ${sourcePath}.data must be a non-shared ArrayBuffer.`,
    );
  }
  if (mimeType !== undefined && typeof mimeType !== 'string') {
    throw new TypeError(
      `[vitrum/gltf-adapter] Cache entry for ${sourcePath}.mimeType must be a string when supplied.`,
    );
  }
  return mimeType === undefined
    ? { data: data as ArrayBuffer }
    : { data: data as ArrayBuffer, mimeType };
}

async function readBoundedResponseBody(
  response: GltfAssetFetchResponse,
  url: string,
  kind: GltfAssetResourceKind,
  resourceKey: string,
  sourcePath: string,
  resourceContext: GltfImportResourceContext,
): Promise<ArrayBuffer> {
  const body = responseBody(response, url, kind, sourcePath);
  const contentLength = responseContentLength(response);
  if (contentLength !== undefined) {
    try {
      resourceContext.ledger.ensureEncodedBytes(
        resourceKey,
        contentLength,
        sourcePath,
      );
    } catch (cause) {
      await cancelResponseBody(body, cause);
      throw cause;
    }
  }

  if (body !== undefined && body !== null) {
    return readBoundedStream(
      body,
      url,
      kind,
      resourceKey,
      sourcePath,
      resourceContext,
    );
  }

  let data: ArrayBuffer;
  try {
    data = await response.arrayBuffer();
  } catch (cause) {
    throw new GltfFetchFailed({
      url,
      kind,
      sourcePath,
      message:
        `[vitrum/gltf-adapter] Failed to read ${kind} resource "${url}" as an ArrayBuffer.`,
      cause,
    });
  }
  chargeArrayBuffer(resourceContext, resourceKey, data, sourcePath);
  return data;
}

function responseBody(
  response: GltfAssetFetchResponse,
  url: string,
  kind: GltfAssetResourceKind,
  sourcePath: string,
): GltfAssetReadableStream | null | undefined {
  try {
    return response.body;
  } catch (cause) {
    throw new GltfFetchFailed({
      url,
      kind,
      sourcePath,
      message:
        `[vitrum/gltf-adapter] Failed to inspect the response body for ${kind} resource "${url}".`,
      cause,
    });
  }
}

async function readBoundedStream(
  body: GltfAssetReadableStream,
  url: string,
  kind: GltfAssetResourceKind,
  resourceKey: string,
  sourcePath: string,
  resourceContext: GltfImportResourceContext,
): Promise<ArrayBuffer> {
  let reader: GltfAssetReadableStreamReader;
  try {
    reader = body.getReader();
  } catch (cause) {
    throw new GltfFetchFailed({
      url,
      kind,
      sourcePath,
      message:
        `[vitrum/gltf-adapter] Failed to acquire the response stream for ${kind} resource "${url}".`,
      cause,
    });
  }

  const chunks: Uint8Array[] = [];
  let total = 0;
  let chunkCount = 0;
  try {
    while (true) {
      const result = await reader.read();
      if (result.done) break;
      chunkCount += 1;
      if (chunkCount > MAX_RESOURCE_STREAM_CHUNKS) {
        throw new RangeError(
          `[vitrum/gltf-adapter] Streamed ${kind} resource "${url}" exceeded ` +
          `${MAX_RESOURCE_STREAM_CHUNKS} response chunks.`,
        );
      }
      const chunk = localUint8ArrayView(result.value);
      if (chunk === null) {
        throw new TypeError(
          '[vitrum/gltf-adapter] Resource response streams must yield non-shared Uint8Array chunks.',
        );
      }
      if (chunk.byteLength > Number.MAX_SAFE_INTEGER - total) {
        throw new RangeError(
          `[vitrum/gltf-adapter] Streamed ${kind} resource "${url}" exceeds the safe integer range.`,
        );
      }
      const nextTotal = total + chunk.byteLength;
      resourceContext.ledger.ensureEncodedBytes(
        resourceKey,
        nextTotal,
        sourcePath,
      );
      const copy = new Uint8Array(chunk.byteLength);
      copy.set(chunk);
      chunks.push(copy);
      resourceContext.ledger.chargeEncodedBytes(
        resourceKey,
        nextTotal,
        sourcePath,
      );
      total = nextTotal;
    }
  } catch (cause) {
    await cancelReader(reader, cause);
    if (cause instanceof GltfAdapterError) throw cause;
    throw new GltfFetchFailed({
      url,
      kind,
      sourcePath,
      message:
        `[vitrum/gltf-adapter] Failed while streaming ${kind} resource "${url}".`,
      cause,
    });
  } finally {
    try {
      reader.releaseLock?.();
    } catch {
      // Releasing a host reader is best-effort and must not mask the load result.
    }
  }

  const joined = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    joined.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return joined.buffer;
}

function responseContentLength(
  response: GltfAssetFetchResponse,
): number | undefined {
  const raw = safeResponseHeader(response, 'content-length');
  if (raw === undefined) return undefined;
  const trimmed = raw.trim();
  if (!/^[0-9]+$/.test(trimmed)) return undefined;
  const value = Number(trimmed);
  return Number.isSafeInteger(value) && value >= 0 ? value : undefined;
}

function safeResponseHeader(
  response: GltfAssetFetchResponse,
  name: string,
): string | undefined {
  try {
    const value = response.headers?.get(name);
    return typeof value === 'string' ? value : undefined;
  } catch {
    // Content headers are advisory; body validation remains authoritative.
    return undefined;
  }
}

async function cancelResponseBody(
  body: GltfAssetReadableStream | null | undefined,
  reason: unknown,
): Promise<void> {
  if (body === undefined || body === null) return;
  try {
    if (typeof body.cancel === 'function') {
      await body.cancel(reason);
      return;
    }
    const reader = body.getReader();
    await reader.cancel?.(reason);
    try {
      reader.releaseLock?.();
    } catch {
      // Best-effort cleanup only.
    }
  } catch {
    // Cancellation must not replace the resource-limit failure.
  }
}

async function cancelReader(
  reader: GltfAssetReadableStreamReader,
  reason: unknown,
): Promise<void> {
  try {
    await reader.cancel?.(reason);
  } catch {
    // Cancellation must not replace the original stream failure.
  }
}

async function fetchResource(
  url: string,
  options: LoadGltfAssetOptions,
  kind: GltfAssetResourceKind,
  sourcePath?: string,
): Promise<GltfAssetFetchResponse> {
  const fetchFn = (options.fetch ?? globalThis.fetch) as GltfAssetFetch | undefined;
  if (typeof fetchFn !== 'function') {
    throw new GltfResourceNotFound({
      url,
      kind,
      ...(sourcePath !== undefined ? { sourcePath } : {}),
      message:
        `[vitrum/gltf-adapter] loadGltfAsset requires a fetch implementation ` +
        `for ${kind} resource "${url}".`,
    });
  }
  const init = options.signal ? { signal: options.signal } : undefined;
  let response: GltfAssetFetchResponse;
  try {
    response = await fetchFn(url, init);
  } catch (cause) {
    throw new GltfFetchFailed({ url, kind, ...(sourcePath !== undefined ? { sourcePath } : {}), cause });
  }
  if (
    response === null ||
    (typeof response !== 'object' && typeof response !== 'function')
  ) {
    throw new GltfFetchFailed({
      url,
      kind,
      ...(sourcePath !== undefined ? { sourcePath } : {}),
      message:
        `[vitrum/gltf-adapter] Fetch for ${kind} resource "${url}" returned an invalid response object.`,
    });
  }

  let ok: boolean | undefined;
  let status: number | undefined;
  let statusText: string | undefined;
  try {
    ok = response.ok;
    if (ok === false) {
      status = response.status;
      statusText = response.statusText;
    }
  } catch (cause) {
    throw new GltfFetchFailed({
      url,
      kind,
      ...(sourcePath !== undefined ? { sourcePath } : {}),
      message:
        `[vitrum/gltf-adapter] Failed to inspect fetch metadata for ${kind} resource "${url}".`,
      cause,
    });
  }
  if (
    (ok !== undefined && typeof ok !== 'boolean') ||
    (status !== undefined && (typeof status !== 'number' || !Number.isFinite(status))) ||
    (statusText !== undefined && typeof statusText !== 'string')
  ) {
    throw new GltfFetchFailed({
      url,
      kind,
      ...(sourcePath !== undefined ? { sourcePath } : {}),
      message:
        `[vitrum/gltf-adapter] Fetch for ${kind} resource "${url}" returned invalid response metadata.`,
    });
  }
  if (ok === false) {
    throw new GltfFetchFailed(Object.assign(
      { url, kind },
      sourcePath === undefined ? {} : { sourcePath },
      status === undefined ? {} : { status },
      statusText === undefined ? {} : { statusText },
    ));
  }
  return response;
}

function normalizeBufferMap(
  buffers: GltfToSceneOptions['buffers'] | undefined,
  resourceContext: GltfImportResourceContext,
): Map<number, ArrayBuffer> {
  const out = new Map<number, ArrayBuffer>();
  if (buffers == null) return out;
  if (isBufferMapLike(buffers)) {
    const bufferMap = buffers;
    for (const [k, v] of bufferMap) {
      assertResourceIndex(k, `options.buffers key "${String(k)}"`);
      chargeArrayBuffer(
        resourceContext,
        gltfBufferResourceKey(k),
        v,
        `options.buffers[${k}]`,
      );
      out.set(k, v);
    }
  } else {
    const bufferRecord = buffers as Record<string, ArrayBuffer>;
    for (const [k, v] of Object.entries(bufferRecord)) {
      const bufferIndex = Number(k);
      assertResourceIndex(bufferIndex, `options.buffers key "${k}"`);
      chargeArrayBuffer(
        resourceContext,
        gltfBufferResourceKey(bufferIndex),
        v,
        `options.buffers[${k}]`,
      );
      out.set(bufferIndex, v);
    }
  }
  return out;
}

function isBufferMapLike(
  value: NonNullable<GltfToSceneOptions['buffers']>,
): value is ReadonlyMap<number, ArrayBuffer> {
  const candidate = value as unknown as {
    readonly entries?: unknown;
    readonly get?: unknown;
    readonly size?: unknown;
    readonly [Symbol.iterator]?: unknown;
  };
  return typeof candidate.entries === 'function' &&
    typeof candidate.get === 'function' &&
    typeof candidate.size === 'number' &&
    typeof candidate[Symbol.iterator] === 'function';
}

function assertResourceIndex(value: unknown, path: string): asserts value is number {
  if (!Number.isSafeInteger(value) || (value as number) < 0) {
    throw new TypeError(
      `[vitrum/gltf-adapter] ${path} must be a non-negative safe integer.`,
    );
  }
}

function chargeArrayBuffer(
  resourceContext: GltfImportResourceContext,
  resourceKey: string,
  value: unknown,
  path: string,
): asserts value is ArrayBuffer {
  const byteLength = gltfArrayBufferByteLength(value);
  if (byteLength === undefined) {
    throw new TypeError(
      `[vitrum/gltf-adapter] ${path} must be a non-shared ArrayBuffer.`,
    );
  }
  resourceContext.ledger.chargeEncodedBytes(resourceKey, byteLength, path);
}

function normalizePreloadedImageBytes(
  images: GltfToSceneOptions['imageBytes'] | undefined,
  resourceContext: GltfImportResourceContext,
  imageIndices?: ReadonlySet<number>,
): Map<number, GltfImageBytes> {
  const out = new Map<number, GltfImageBytes>();
  if (images == null) return out;

  const entries = isImageBytesMapLike(images)
    ? images.entries()
    : Object.entries(images).map(([key, value]) => [Number(key), value] as const);
  for (const [rawIndex, image] of entries) {
    const imageIndex = Number(rawIndex);
    assertResourceIndex(imageIndex, `imageBytes key "${String(rawIndex)}"`);
    if (imageIndices !== undefined && !imageIndices.has(imageIndex)) continue;
    if (image === null || typeof image !== 'object') {
      throw new TypeError(
        `[vitrum/gltf-adapter] imageBytes[${imageIndex}] must be an image byte record.`,
      );
    }
    const bytes = localUint8ArrayView(image.bytes);
    if (bytes === null) {
      throw new TypeError(
        `[vitrum/gltf-adapter] imageBytes[${imageIndex}].bytes must be a non-shared Uint8Array.`,
      );
    }
    if (typeof image.mimeType !== 'string' || image.mimeType.length === 0) {
      throw new TypeError(
        `[vitrum/gltf-adapter] imageBytes[${imageIndex}].mimeType must be a non-empty string.`,
      );
    }
    resourceContext.ledger.chargeEncodedBytes(
      gltfImageResourceKey(imageIndex),
      bytes.byteLength,
      `options.imageBytes[${imageIndex}].bytes`,
    );
    out.set(imageIndex, { bytes, mimeType: image.mimeType });
  }
  return out;
}

function isImageBytesMapLike(
  value: NonNullable<GltfToSceneOptions['imageBytes']>,
): value is ReadonlyMap<number, GltfImageBytes> {
  const candidate = value as unknown as {
    readonly entries?: unknown;
    readonly get?: unknown;
    readonly size?: unknown;
  };
  return typeof candidate.entries === 'function' &&
    typeof candidate.get === 'function' &&
    typeof candidate.size === 'number';
}

function isGlb(buffer: ArrayBuffer): boolean {
  const byteLength = gltfArrayBufferByteLength(buffer);
  return byteLength !== undefined &&
    byteLength >= 4 &&
    new DataView(buffer).getUint32(0, true) === 0x46546c67;
}

function resolveUri(
  uri: string,
  baseUri: string | URL | undefined,
  kind: GltfAssetResourceKind,
  sourcePath?: string,
): string {
  if (uri.startsWith('data:')) return uri;
  try {
    return new URL(uri).toString();
  } catch {
    if (baseUri == null) {
      throw new GltfResourceNotFound({
        url: uri,
        kind,
        ...(sourcePath !== undefined ? { sourcePath } : {}),
        message:
          `[vitrum/gltf-adapter] Cannot resolve relative ${kind} URI "${uri}" without a baseUri.`,
      });
    }
    return new URL(uri, baseUri).toString();
  }
}

function directoryUrl(url: string): string | undefined {
  try {
    return new URL('.', url).toString();
  } catch {
    // Non-hierarchical URLs (for example data:) have no directory base.
    return undefined;
  }
}

function decodeDataUri(
  uri: string,
  kind: GltfAssetResourceKind,
  label: string,
  resourceKey: string,
  resourceContext: GltfImportResourceContext,
  sourcePath?: string,
): Uint8Array {
  const comma = uri.indexOf(',');
  if (comma < 0) {
    throw dataUriError(
      uri,
      kind,
      'malformed-data-uri',
      `[vitrum/gltf-adapter] ${label} has a malformed data: URI.`,
      sourcePath,
    );
  }
  const meta = uri.slice(5, comma);
  const payload = uri.slice(comma + 1);
  const isBase64 = meta.split(';').some((p) => p.toLowerCase() === 'base64');
  const path = sourcePath ?? label;
  resourceContext.ledger.ensureEncodedBytes(
    resourceKey,
    dataUriDecodedByteUpperBound(payload, isBase64),
    path,
  );
  let bytes: Uint8Array;
  try {
    if (isBase64) {
      if (typeof globalThis.atob !== 'function') {
        throw dataUriError(
          uri,
          kind,
          'data-uri-atob-unavailable',
          `[vitrum/gltf-adapter] ${label} uses base64 data URI but atob() is unavailable.`,
          sourcePath,
        );
      }
      const bin = globalThis.atob(payload.replace(/\s+/g, ''));
      bytes = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i += 1) bytes[i] = bin.charCodeAt(i);
    } else {
      bytes = new TextEncoder().encode(decodeURIComponent(payload));
    }
  } catch (cause) {
    if (cause instanceof GltfResourceDecodeFailed) throw cause;
    if (cause instanceof GltfResourceLimitError) throw cause;
    throw dataUriError(
      uri,
      kind,
      'data-uri-decode-failed',
      `[vitrum/gltf-adapter] ${label} data: URI could not be decoded: ` +
        `${cause instanceof Error ? cause.message : String(cause)}.`,
      sourcePath,
      cause,
    );
  }
  resourceContext.ledger.chargeEncodedBytes(
    resourceKey,
    bytes.byteLength,
    path,
  );
  return bytes;
}

function dataUriDecodedByteUpperBound(
  payload: string,
  isBase64: boolean,
): number {
  return isBase64
    ? base64DecodedByteUpperBound(payload)
    : percentDecodedByteUpperBound(payload);
}

function base64DecodedByteUpperBound(payload: string): number {
  let encodedLength = 0;
  let previous = -1;
  let last = -1;
  for (let i = 0; i < payload.length; i += 1) {
    const code = payload.charCodeAt(i);
    if (isDataUriWhitespaceCodeUnit(code)) continue;
    encodedLength += 1;
    previous = last;
    last = code;
  }
  const padding = last === 0x3d ? (previous === 0x3d ? 2 : 1) : 0;
  const completeGroups = Math.floor(encodedLength / 4);
  const remainder = encodedLength % 4;
  const upperBound =
    completeGroups * 3 +
    Math.floor((remainder * 6) / 8) -
    padding;
  return Math.max(0, upperBound);
}

function isDataUriWhitespaceCodeUnit(code: number): boolean {
  return (
    (code >= 0x09 && code <= 0x0d) ||
    code === 0x20 ||
    code === 0xa0 ||
    code === 0x1680 ||
    (code >= 0x2000 && code <= 0x200a) ||
    code === 0x2028 ||
    code === 0x2029 ||
    code === 0x202f ||
    code === 0x205f ||
    code === 0x3000 ||
    code === 0xfeff
  );
}

function percentDecodedByteUpperBound(payload: string): number {
  let byteLength = 0;
  for (let i = 0; i < payload.length; i += 1) {
    if (
      payload.charCodeAt(i) === 0x25 &&
      i + 2 < payload.length &&
      isHexCodeUnit(payload.charCodeAt(i + 1)) &&
      isHexCodeUnit(payload.charCodeAt(i + 2))
    ) {
      byteLength = checkedByteLengthAdd(byteLength, 1);
      i += 2;
      continue;
    }
    const codePoint = payload.codePointAt(i)!;
    const width =
      codePoint <= 0x7f ? 1 :
      codePoint <= 0x7ff ? 2 :
      codePoint <= 0xffff ? 3 :
      4;
    byteLength = checkedByteLengthAdd(byteLength, width);
    if (codePoint > 0xffff) i += 1;
  }
  return byteLength;
}

function checkedByteLengthAdd(total: number, additional: number): number {
  if (additional > Number.MAX_SAFE_INTEGER - total) {
    throw new RangeError(
      '[vitrum/gltf-adapter] data: URI decoded byte length exceeds the safe integer range.',
    );
  }
  return total + additional;
}

function isHexCodeUnit(code: number): boolean {
  return (
    (code >= 0x30 && code <= 0x39) ||
    (code >= 0x41 && code <= 0x46) ||
    (code >= 0x61 && code <= 0x66)
  );
}

function dataUriError(
  uri: string,
  kind: GltfAssetResourceKind,
  reason: GltfResourceDecodeFailureReason,
  message: string,
  sourcePath?: string,
  cause?: unknown,
): GltfResourceDecodeFailed {
  return new GltfResourceDecodeFailed({
    url: uri,
    kind,
    reason,
    message,
    ...(sourcePath !== undefined ? { sourcePath } : {}),
    ...(cause === undefined ? {} : { cause }),
  });
}

function uint8ToArrayBuffer(bytes: Uint8Array): ArrayBuffer {
  const out = new ArrayBuffer(bytes.byteLength);
  new Uint8Array(out).set(bytes);
  return out;
}

function inferMimeType(uri: string): string {
  return inferMimeTypeFromUri(uri) ?? 'image/png';
}

function inferMimeTypeFromUri(uri: string): string | undefined {
  const lower = uri.toLowerCase().split(/[?#]/u)[0] ?? uri.toLowerCase();
  if (lower.endsWith('.png')) return 'image/png';
  if (lower.endsWith('.jpg') || lower.endsWith('.jpeg')) return 'image/jpeg';
  if (lower.endsWith('.webp')) return 'image/webp';
  if (lower.endsWith('.ktx2')) return 'image/ktx2';
  if (lower.endsWith('.dds')) return 'image/vnd-ms.dds';
  return undefined;
}

function inferCachedImageMimeType(
  data: ArrayBuffer,
  uri: string,
): string | undefined {
  const bytes = new Uint8Array(data);
  if (
    bytes.length >= 8 &&
    bytes[0] === 0x89 &&
    bytes[1] === 0x50 &&
    bytes[2] === 0x4e &&
    bytes[3] === 0x47 &&
    bytes[4] === 0x0d &&
    bytes[5] === 0x0a &&
    bytes[6] === 0x1a &&
    bytes[7] === 0x0a
  ) {
    return 'image/png';
  }
  if (
    bytes.length >= 3 &&
    bytes[0] === 0xff &&
    bytes[1] === 0xd8 &&
    bytes[2] === 0xff
  ) {
    return 'image/jpeg';
  }
  if (
    bytes.length >= 12 &&
    bytes[0] === 0x52 &&
    bytes[1] === 0x49 &&
    bytes[2] === 0x46 &&
    bytes[3] === 0x46 &&
    bytes[8] === 0x57 &&
    bytes[9] === 0x45 &&
    bytes[10] === 0x42 &&
    bytes[11] === 0x50
  ) {
    return 'image/webp';
  }
  if (
    bytes.length >= 12 &&
    bytes[0] === 0xab &&
    bytes[1] === 0x4b &&
    bytes[2] === 0x54 &&
    bytes[3] === 0x58 &&
    bytes[4] === 0x20 &&
    bytes[5] === 0x32 &&
    bytes[6] === 0x30 &&
    bytes[7] === 0xbb &&
    bytes[8] === 0x0d &&
    bytes[9] === 0x0a &&
    bytes[10] === 0x1a &&
    bytes[11] === 0x0a
  ) {
    return 'image/ktx2';
  }
  if (
    bytes.length >= 4 &&
    bytes[0] === 0x44 &&
    bytes[1] === 0x44 &&
    bytes[2] === 0x53 &&
    bytes[3] === 0x20
  ) {
    return 'image/vnd-ms.dds';
  }
  return inferMimeTypeFromUri(uri);
}
