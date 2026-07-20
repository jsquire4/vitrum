// assetLoader.ts — turnkey glTF/GLB loading into Vitrum's core Scene contract.
//
// `gltfToScene()` remains the low-level converter for hosts that already own
// resource resolution. `loadGltfAsset()` is the predictable higher-level path:
// it accepts URL/string/ArrayBuffer/GltfJson, resolves external buffers/images,
// returns a structured feature report, and ranks the shipping backends against
// the asset's actual feature use.

import type { MaterialSpec, MeshPrimitive, Scene, ScenePrimitive } from '@vitrum/core';
import type { GltfJson } from './gltfTypes.js';
import { gltfToScene, type GltfToSceneOptions, type GltfToSceneResult } from './gltfToScene.js';
import { parseGlb } from './glbParser.js';
import type { DecodeImageFn, GltfImageBytes, RawImageHandle } from './textures.js';
import {
  analyzeGltfAsset,
  rankGltfBackends,
  type GltfBackendCompatibility,
  type GltfBackendPolicy,
  type GltfFeatureReport,
} from './featureReport.js';
import {
  buildTextureDecodeReport,
  decodeSceneTextures,
  type DecodeGltfTexturePixelsFn,
  type DecodeSceneTextureDiagnostic,
  type DecodeSceneTexturesOptions,
  type GltfTextureDecodeReport,
} from './texturePipeline.js';
import {
  GltfFetchFailed,
  GltfParseFailed,
  GltfResourceDecodeFailed,
  GltfResourceNotFound,
  type GltfAssetResourceKind,
  type GltfResourceDecodeFailureReason,
} from './errors.js';
import { collectGltfSceneReachability } from './sceneScope.js';
import {
  bakePtWebgpuLiteCompatibleVertexColors,
  hasReachableMaterialPointerAnimationForColoredPrimitive,
  reconcileBackendCompatibilityAfterSceneImport,
  reconcileBackendCompatibilityAfterTextureDecode,
  rerankBackendCompatibility,
} from './backendCompatibilityReconcile.js';

export type GltfAssetInput = string | URL | ArrayBuffer | GltfJson;

export interface GltfAssetFetchResponse {
  readonly ok?: boolean;
  readonly status?: number;
  readonly statusText?: string;
  readonly headers?: { get(name: string): string | null };
  arrayBuffer(): Promise<ArrayBuffer>;
}

export type GltfAssetFetch = (
  url: string,
  init?: { readonly signal?: AbortSignal },
) => Promise<GltfAssetFetchResponse>;

export interface GltfAssetCacheKey {
  readonly url: string;
  readonly kind: GltfAssetResourceKind;
}

export interface GltfAssetCache {
  get(key: GltfAssetCacheKey): ArrayBuffer | undefined | Promise<ArrayBuffer | undefined>;
  set(key: GltfAssetCacheKey, data: ArrayBuffer): void | Promise<void>;
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
  const parsed = await parseInput(input, options);
  const buffers = normalizeBufferMap(options.buffers);
  for (const [index, buffer] of parsed.buffers) {
    if (!buffers.has(index)) buffers.set(index, buffer);
  }

  const backendPolicy = options.backendPolicy ?? 'fidelity';
  const sceneIndex = options.sceneIndex ?? parsed.gltf.scene ?? 0;
  const sceneReachability = collectGltfSceneReachability(
    parsed.gltf,
    sceneIndex,
    options.textureSourceExtensions ?? [],
  );

  await resolveExternalBuffers(parsed.gltf, buffers, parsed.baseUri, options, sceneReachability.bufferIndices);
  const imageBytes = await resolveExternalImages(
    parsed.gltf,
    parsed.baseUri,
    options,
    sceneReachability.imageIndices,
  );

  const featureReport = analyzeGltfAsset(parsed.gltf, {
    ...(options.textureSourceExtensions ? { textureSourceExtensions: options.textureSourceExtensions } : {}),
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
    ...(options.textureSourceExtensions ? { textureSourceExtensions: options.textureSourceExtensions } : {}),
    ...(options.materialVariant !== undefined ? { materialVariant: options.materialVariant } : {}),
    ...(options.pointLineFallbackRadius !== undefined ? { pointLineFallbackRadius: options.pointLineFallbackRadius } : {}),
  };
  const sceneResult = await gltfToScene(parsed.gltf, sceneOptions);
  const canBakeLiteVertexColors =
    (sceneResult.materialVariantBindings?.length ?? 0) === 0 &&
    !hasReachableMaterialPointerAnimationForColoredPrimitive(parsed.gltf, sceneReachability);
  const scene = canBakeLiteVertexColors
    ? bakePtWebgpuLiteCompatibleVertexColors(sceneResult.scene)
    : sceneResult.scene;
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
      scene,
      textureDecodeReport,
      canBakeLiteVertexColors,
    ),
    backendPolicy,
  );

  return {
    ...sceneResult,
    scene,
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
  const asset = await loadGltfAsset(input, loadOptionsForTextureDecode(options));
  let decodeOptions: DecodeSceneTexturesOptions = {
    target: options.textureTarget ?? 'cpu-linear',
    ...(options.decodePixels ? { decodePixels: options.decodePixels } : {}),
    ...(options.maxTextureSize !== undefined ? { maxTextureSize: options.maxTextureSize } : {}),
    ...(options.warnOnNpotRepeatWrap !== undefined ? { warnOnNpotRepeatWrap: options.warnOnNpotRepeatWrap } : {}),
    ...(options.npotRepeatWrapPolicy !== undefined ? { npotRepeatWrapPolicy: options.npotRepeatWrapPolicy } : {}),
    ...(options.onTextureDiagnostic ? { onDiagnostic: options.onTextureDiagnostic } : {}),
    ...(options.onTextureWarning ? { onWarning: options.onTextureWarning } : {}),
  };
  const configuredDecodeOptions = await options.configureTextureDecode?.({ asset, decodeOptions });
  if (configuredDecodeOptions !== undefined) {
    decodeOptions = { ...decodeOptions, ...configuredDecodeOptions };
  }
  const decoded = await decodeSceneTextures(asset.scene, decodeOptions);
  const convertedMaterials = asset.convertedMaterials === undefined
    ? undefined
    : await decodeConvertedMaterials(
      asset.convertedMaterials,
      asset.scene,
      decoded.scene,
      decodeOptions,
      materialIndicesForSelectedScene(asset.materialVariantBindings),
    );
  const sceneWithMaterialTable = convertedMaterials === undefined
    ? decoded.scene
    : appendInactiveMaterialPrimitives(
      decoded.scene,
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
    scene: decoded.scene,
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
  options: DecodeSceneTexturesOptions,
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

  const decoded = await decodeSceneTextures(materialsToSyntheticScene(pending), options);
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

async function parseInput(
  input: GltfAssetInput,
  options: LoadGltfAssetOptions,
): Promise<ParsedInput> {
  if (typeof input === 'string' || input instanceof URL) {
    const url = resolveUri(String(input), options.baseUri, 'asset');
    const bytes = await fetchArrayBuffer(url, options, 'asset');
    return parseArrayBufferInput(bytes, directoryUrl(url));
  }
  if (input instanceof ArrayBuffer) {
    return parseArrayBufferInput(input, options.baseUri ? String(options.baseUri) : undefined);
  }
  return {
    gltf: input,
    ...(options.baseUri ? { baseUri: String(options.baseUri) } : {}),
    buffers: new Map(),
  };
}

function parseArrayBufferInput(buffer: ArrayBuffer, baseUri?: string): ParsedInput {
  if (isGlb(buffer)) {
    const glb = parseGlb(buffer);
    const buffers = new Map<number, ArrayBuffer>();
    if (glb.binChunk) buffers.set(0, glb.binChunk);
    return { gltf: glb.json, ...(baseUri ? { baseUri } : {}), buffers };
  }
  const text = new TextDecoder().decode(buffer);
  try {
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
  bufferIndices?: ReadonlySet<number>,
): Promise<void> {
  for (const [index, buffer] of (gltf.buffers ?? []).entries()) {
    if (bufferIndices !== undefined && !bufferIndices.has(index)) continue;
    if (buffers.has(index)) continue;
    if (buffer.uri == null) continue;
    const sourcePath = `buffers[${index}].uri`;
    if (buffer.uri.startsWith('data:')) {
      buffers.set(index, uint8ToArrayBuffer(decodeDataUri(buffer.uri, 'buffer', `buffer ${index}`, sourcePath)));
      continue;
    }
    const url = resolveUri(buffer.uri, baseUri, 'buffer', sourcePath);
    const data = await fetchArrayBuffer(url, options, 'buffer', sourcePath);
    buffers.set(index, data);
  }
}

async function resolveExternalImages(
  gltf: GltfJson,
  baseUri: string | undefined,
  options: LoadGltfAssetOptions,
  imageIndices?: ReadonlySet<number>,
): Promise<Map<number, GltfImageBytes>> {
  const out = new Map<number, GltfImageBytes>();
  for (const [index, image] of (gltf.images ?? []).entries()) {
    if (imageIndices !== undefined && !imageIndices.has(index)) continue;
    if (image.bufferView !== undefined) continue;
    if (image.uri == null || image.uri.startsWith('data:')) continue;
    const sourcePath = `images[${index}].uri`;
    const url = resolveUri(image.uri, baseUri, 'image', sourcePath);
    const fetched = await fetchImageBytes(url, options, sourcePath);
    out.set(index, {
      bytes: fetched.bytes,
      mimeType: image.mimeType ?? fetched.mimeType ?? inferMimeType(image.uri),
    });
  }
  return out;
}

async function fetchImageBytes(
  url: string,
  options: LoadGltfAssetOptions,
  sourcePath?: string,
): Promise<{ readonly bytes: Uint8Array; readonly mimeType?: string }> {
  const cacheKey = { url, kind: 'image' } satisfies GltfAssetCacheKey;
  const cached = await options.cache?.get(cacheKey);
  if (cached !== undefined) return { bytes: new Uint8Array(cached) };
  const response = await fetchResource(url, options, 'image', sourcePath);
  const data = await response.arrayBuffer();
  await options.cache?.set(cacheKey, data);
  const mimeType = response.headers?.get('content-type') ?? undefined;
  return mimeType === undefined
    ? { bytes: new Uint8Array(data) }
    : { bytes: new Uint8Array(data), mimeType };
}

async function fetchArrayBuffer(
  url: string,
  options: LoadGltfAssetOptions,
  kind: GltfAssetResourceKind,
  sourcePath?: string,
): Promise<ArrayBuffer> {
  const cacheKey = { url, kind } satisfies GltfAssetCacheKey;
  const cached = await options.cache?.get(cacheKey);
  if (cached !== undefined) return cached;
  const response = await fetchResource(url, options, kind, sourcePath);
  const data = await response.arrayBuffer();
  await options.cache?.set(cacheKey, data);
  return data;
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
  if (response.ok === false) {
    throw new GltfFetchFailed(Object.assign(
      { url, kind },
      sourcePath === undefined ? {} : { sourcePath },
      response.status === undefined ? {} : { status: response.status },
      response.statusText === undefined ? {} : { statusText: response.statusText },
    ));
  }
  return response;
}

function normalizeBufferMap(
  buffers: GltfToSceneOptions['buffers'] | undefined,
): Map<number, ArrayBuffer> {
  const out = new Map<number, ArrayBuffer>();
  if (buffers == null) return out;
  if (buffers instanceof Map) {
    const bufferMap = buffers as ReadonlyMap<number, ArrayBuffer>;
    for (const [k, v] of bufferMap) out.set(k, v);
  } else {
    const bufferRecord = buffers as Record<string, ArrayBuffer>;
    for (const [k, v] of Object.entries(bufferRecord)) {
      out.set(Number(k), v);
    }
  }
  return out;
}

function isGlb(buffer: ArrayBuffer): boolean {
  return buffer.byteLength >= 4 && new DataView(buffer).getUint32(0, true) === 0x46546c67;
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

function directoryUrl(url: string): string {
  return new URL('.', url).toString();
}

function decodeDataUri(uri: string, kind: GltfAssetResourceKind, label: string, sourcePath?: string): Uint8Array {
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
      const bytes = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i += 1) bytes[i] = bin.charCodeAt(i);
      return bytes;
    }
    return new TextEncoder().encode(decodeURIComponent(payload));
  } catch (cause) {
    if (cause instanceof GltfResourceDecodeFailed) throw cause;
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
  const lower = uri.toLowerCase().split('?')[0] ?? uri.toLowerCase();
  if (lower.endsWith('.jpg') || lower.endsWith('.jpeg')) return 'image/jpeg';
  if (lower.endsWith('.webp')) return 'image/webp';
  if (lower.endsWith('.ktx2')) return 'image/ktx2';
  if (lower.endsWith('.dds')) return 'image/vnd-ms.dds';
  return 'image/png';
}
