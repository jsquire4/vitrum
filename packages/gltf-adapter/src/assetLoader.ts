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
  type GltfCompatibilityIssue,
  type GltfFeatureReport,
} from './featureReport.js';
import {
  buildTextureDecodeReport,
  decodeSceneTextures,
  type DecodeGltfTexturePixelsFn,
  type DecodeSceneTextureDiagnostic,
  type DecodeSceneTexturesOptions,
  type GltfBackendTextureStatus,
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
}

export interface GltfAssetResult extends GltfToSceneResult {
  readonly gltf: GltfJson;
  readonly sceneIndex: number;
  readonly featureReport: GltfFeatureReport;
  readonly backendCompatibility: readonly GltfBackendCompatibility[];
  readonly recommendedBackend: GltfBackendCompatibility;
  readonly textureDecodeReport: GltfTextureDecodeReport;
}

export interface LoadGltfAndDecodeTexturesOptions extends LoadGltfAssetOptions {
  readonly textureTarget?: DecodeSceneTexturesOptions['target'];
  readonly decodePixels?: DecodeGltfTexturePixelsFn;
  readonly maxTextureSize?: number;
  readonly warnOnNpotRepeatWrap?: boolean;
  readonly onTextureDiagnostic?: (diagnostic: DecodeSceneTextureDiagnostic) => void;
  readonly onTextureWarning?: (message: string) => void;
}

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

  await resolveExternalBuffers(parsed.gltf, buffers, parsed.baseUri, options);
  const imageBytes = await resolveExternalImages(parsed.gltf, parsed.baseUri, options);

  const backendPolicy = options.backendPolicy ?? 'fidelity';
  const featureReport = analyzeGltfAsset(parsed.gltf, {
    ...(options.textureSourceExtensions ? { textureSourceExtensions: options.textureSourceExtensions } : {}),
  });
  const staticBackendCompatibility = rankGltfBackends(featureReport, backendPolicy);
  const sceneIndex = options.sceneIndex ?? parsed.gltf.scene ?? 0;
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
  const textureDecodeReport = buildTextureDecodeReport(sceneResult.scene);
  const backendCompatibility = rerankBackendCompatibility(
    reconcileBackendCompatibilityAfterSceneImport(
      staticBackendCompatibility,
      sceneResult.scene,
      textureDecodeReport,
    ),
    backendPolicy,
  );

  return {
    ...sceneResult,
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
  const decodeOptions: DecodeSceneTexturesOptions = {
    target: options.textureTarget ?? 'cpu-linear',
    ...(options.decodePixels ? { decodePixels: options.decodePixels } : {}),
    ...(options.maxTextureSize !== undefined ? { maxTextureSize: options.maxTextureSize } : {}),
    ...(options.warnOnNpotRepeatWrap !== undefined ? { warnOnNpotRepeatWrap: options.warnOnNpotRepeatWrap } : {}),
    ...(options.onTextureDiagnostic ? { onDiagnostic: options.onTextureDiagnostic } : {}),
    ...(options.onTextureWarning ? { onWarning: options.onTextureWarning } : {}),
  };
  const decoded = await decodeSceneTextures(asset.scene, decodeOptions);
  const convertedMaterials = asset.convertedMaterials === undefined
    ? undefined
    : await decodeConvertedMaterials(
      asset.convertedMaterials,
      asset.scene,
      decoded.scene,
      decodeOptions,
    );
  const sceneWithMaterialTable = convertedMaterials === undefined
    ? decoded.scene
    : appendInactiveMaterialPrimitives(decoded.scene, convertedMaterials.materials);
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

const SPEC_GLOSS_ALPHA_ISSUE =
  'KHR_materials_pbrSpecularGlossiness.specularGlossinessTexture.glossinessAlpha';
const EMISSIVE_MAP_TEXEL_PDF_ISSUE = 'emissiveMap.texelPdf';
const TEXTURE_READINESS_ISSUE_PREFIX = 'texture-readiness:';
const TEXTURE_SOURCE_EXTENSION_HOOK_ISSUES = new Set([
  'KHR_texture_basisu',
  'EXT_texture_webp',
  'MSFT_texture_dds',
]);

function reconcileBackendCompatibilityAfterSceneImport(
  compatibility: readonly GltfBackendCompatibility[],
  scene: Scene,
  textureDecodeReport: GltfTextureDecodeReport,
): readonly GltfBackendCompatibility[] {
  const liteVertexColorsBakeable = sceneHasOnlyPtWebgpuLiteBakeableVertexColors(scene);

  return compatibility.map((candidate) => {
    if (candidate.profileId !== 'pt-webgpu-lite' || !liteVertexColorsBakeable) {
      return compatibilityWithTextureReadiness(candidate, textureDecodeReport);
    }
    const issues = candidate.issues.filter((issue) =>
      !(issue.category === 'primitive' && issue.name === 'vertexColors')
    );
    return compatibilityWithTextureReadiness(
      issues.length === candidate.issues.length ? candidate : compatibilityWithIssues(candidate, issues),
      textureDecodeReport,
    );
  });
}

function sceneHasOnlyPtWebgpuLiteBakeableVertexColors(scene: Scene): boolean {
  for (const primitive of scene.primitives) {
    const colors = (primitive as { readonly colors?: Float32Array }).colors;
    if (colors != null && colors.length > 0 && !ptWebgpuLiteCanBakeVertexColors(primitive)) {
      return false;
    }
  }
  return true;
}

function ptWebgpuLiteCanBakeVertexColors(primitive: ScenePrimitive): boolean {
  const colors = (primitive as { readonly colors?: Float32Array }).colors;
  const positions = (primitive as { readonly positions?: Float32Array }).positions;
  if (colors == null || colors.length === 0) return true;
  if (positions == null || positions.length === 0) return false;
  const vertexCount = Math.floor(positions.length / 3);
  const stride = colors.length >= vertexCount * 4
    ? 4
    : colors.length >= vertexCount * 3
      ? 3
      : 0;
  if (vertexCount === 0 || stride === 0) return false;
  const r = colors[0] ?? 1;
  const g = colors[1] ?? 1;
  const b = colors[2] ?? 1;
  const eps = 1e-6;
  for (let i = 0; i < vertexCount; i += 1) {
    const o = i * stride;
    if (
      Math.abs((colors[o] ?? 1) - r) > eps ||
      Math.abs((colors[o + 1] ?? 1) - g) > eps ||
      Math.abs((colors[o + 2] ?? 1) - b) > eps
    ) {
      return false;
    }
    if (stride === 4 && Math.abs((colors[o + 3] ?? 1) - 1) > eps) {
      return false;
    }
  }
  return true;
}

function reconcileBackendCompatibilityAfterTextureDecode(
  compatibility: readonly GltfBackendCompatibility[],
  report: GltfTextureDecodeReport,
  diagnostics: readonly DecodeSceneTextureDiagnostic[],
  backendPolicy: GltfBackendPolicy,
): readonly GltfBackendCompatibility[] {
  const bakeUnavailable = diagnostics.some((diagnostic) =>
    diagnostic.code === 'spec-gloss-alpha-bake-unavailable'
  );
  const bakedRoughnessMap = report.entries.some((entry) =>
    entry.materialField === 'roughnessMap' && entry.handleKind === 'pixel-data'
  );

  const reconciled = compatibility.map((candidate) => {
    const issues = candidate.issues.filter((issue) => {
      if (isTextureReadinessIssue(issue)) return false;
      if (
        issue.name === SPEC_GLOSS_ALPHA_ISSUE &&
        bakedRoughnessMap &&
        !bakeUnavailable
      ) {
        return false;
      }
      if (emissiveTexelPdfIssueSatisfiedByDecode(issue, candidate, report)) return false;
      if (textureSourceHookIssueSatisfiedByDecode(issue, report)) return false;
      return true;
    });
    return compatibilityWithTextureReadiness(
      issues.length === candidate.issues.length ? candidate : compatibilityWithIssues(candidate, issues),
      report,
    );
  });
  return rerankBackendCompatibility(reconciled, backendPolicy);
}

function emissiveTexelPdfIssueSatisfiedByDecode(
  issue: GltfCompatibilityIssue,
  candidate: GltfBackendCompatibility,
  report: GltfTextureDecodeReport,
): boolean {
  if (
    issue.name !== EMISSIVE_MAP_TEXEL_PDF_ISSUE ||
    issue.support !== 'approximate' ||
    (candidate.profileId !== 'pt-webgl2' && candidate.profileId !== 'pt-webgpu')
  ) {
    return false;
  }
  const entries = report.entries.filter((entry) => entry.materialField === 'emissiveMap');
  if (entries.length === 0) return false;
  const key = textureReadinessKey(candidate);
  if (key == null) return false;
  return entries.every((entry) =>
    (entry.handleKind === 'pixel-data' || entry.handleKind === 'data-texture') &&
    entry.backendReadiness[key] === 'ready'
  );
}

function textureSourceHookIssueSatisfiedByDecode(
  issue: GltfCompatibilityIssue,
  report: GltfTextureDecodeReport,
): boolean {
  if (issue.support !== 'requires-hook' || !TEXTURE_SOURCE_EXTENSION_HOOK_ISSUES.has(issue.name)) {
    return false;
  }
  const entries = report.entries.filter((entry) => entry.textureSourceExtension === issue.name);
  return entries.length > 0 && entries.every((entry) =>
    entry.handleKind === 'pixel-data' || entry.handleKind === 'data-texture'
  );
}

function compatibilityWithTextureReadiness(
  candidate: GltfBackendCompatibility,
  report: GltfTextureDecodeReport,
): GltfBackendCompatibility {
  const baseIssues = candidate.issues.filter((issue) => !isTextureReadinessIssue(issue));
  const textureIssues = textureReadinessIssuesForCandidate(
    issuesMatch(candidate.issues, baseIssues) ? candidate : compatibilityWithIssues(candidate, baseIssues),
    report,
  );
  if (textureIssues.length === 0 && issuesMatch(candidate.issues, baseIssues)) return candidate;
  return compatibilityWithIssues(candidate, [...baseIssues, ...textureIssues]);
}

function textureReadinessIssuesForCandidate(
  candidate: GltfBackendCompatibility,
  report: GltfTextureDecodeReport,
): GltfCompatibilityIssue[] {
  const key = textureReadinessKey(candidate);
  if (key == null) return [];

  const unsupportedMaterialFields = new Set(
    candidate.issues
      .filter((issue) => issue.category === 'material' && issue.support === 'unsupported')
      .map((issue) => issue.name),
  );

  const issues: GltfCompatibilityIssue[] = [];
  for (const entry of report.entries) {
    if (unsupportedMaterialFields.has(entry.materialField)) continue;
    const status = entry.backendReadiness[key];
    const support = textureReadinessSupport(status);
    if (support == null) continue;
    issues.push({
      category: 'texture',
      name: `${TEXTURE_READINESS_ISSUE_PREFIX}${entry.materialField}`,
      support,
      path: entry.path,
      message:
        support === 'requires-hook'
          ? `Backend profile ${candidate.profileId} needs a decoded or backend-native texture handle for ` +
            `"${entry.materialField}" at ${entry.path}; current handle is ${entry.handleKind}.`
          : `Backend profile ${candidate.profileId} does not consume "${entry.materialField}" texture data ` +
            `at ${entry.path}; current handle is ${entry.handleKind}.`,
    });
  }
  return issues;
}

function textureReadinessKey(
  candidate: GltfBackendCompatibility,
): keyof GltfTextureDecodeReport['entries'][number]['backendReadiness'] | undefined {
  if (candidate.backend === 'pt-webgl2') return 'ptWebgl2';
  if (candidate.backend === 'pt-webgpu') return 'ptWebgpu';
  if (candidate.backend === 'walkaround-hybrid') return 'walkaroundHybrid';
  return undefined;
}

function textureReadinessSupport(status: GltfBackendTextureStatus): GltfCompatibilityIssue['support'] | null {
  if (status === 'ready') return null;
  if (status === 'opaque') return 'requires-hook';
  return 'unsupported';
}

function isTextureReadinessIssue(issue: GltfCompatibilityIssue): boolean {
  return issue.category === 'texture' && issue.name.startsWith(TEXTURE_READINESS_ISSUE_PREFIX);
}

function issuesMatch(
  a: readonly GltfCompatibilityIssue[],
  b: readonly GltfCompatibilityIssue[],
): boolean {
  return a.length === b.length && a.every((issue, index) => issue === b[index]);
}

function rerankBackendCompatibility(
  compatibility: readonly GltfBackendCompatibility[],
  policy: GltfBackendPolicy,
): readonly GltfBackendCompatibility[] {
  const preferred = policy === 'realtime'
    ? ['walkaround-hybrid', 'pt-webgpu', 'pt-webgpu-lite', 'pt-webgl2']
    : ['pt-webgl2', 'pt-webgpu', 'pt-webgpu-lite', 'walkaround-hybrid'];
  const order = new Map(preferred.map((profileId, index) => [profileId, index]));
  return [...compatibility].sort((a, b) => {
    if (policy === 'strict') {
      const aBad = a.unsupportedCount + a.approximateCount + a.requiresHookCount;
      const bBad = b.unsupportedCount + b.approximateCount + b.requiresHookCount;
      if (aBad !== bBad) return aBad - bBad;
    } else {
      if (a.unsupportedCount !== b.unsupportedCount) return a.unsupportedCount - b.unsupportedCount;
      if (a.requiresHookCount !== b.requiresHookCount) return a.requiresHookCount - b.requiresHookCount;
      if (a.approximateCount !== b.approximateCount) return a.approximateCount - b.approximateCount;
    }
    return (order.get(a.profileId) ?? 99) - (order.get(b.profileId) ?? 99);
  });
}

function compatibilityWithIssues(
  candidate: GltfBackendCompatibility,
  issues: readonly GltfCompatibilityIssue[],
): GltfBackendCompatibility {
  let unsupportedCount = 0;
  let approximateCount = 0;
  let requiresHookCount = 0;

  for (const issue of issues) {
    if (issue.support === 'unsupported') {
      unsupportedCount += 1;
    } else if (
      issue.support === 'approximate' ||
      issue.support === 'fallback-generated-mesh' ||
      issue.support === 'fallback-rebuild'
    ) {
      approximateCount += 1;
    } else if (issue.support === 'requires-hook') {
      requiresHookCount += 1;
    }
  }

  return {
    ...candidate,
    unsupportedCount,
    approximateCount,
    nativeCount: candidate.nativeCount,
    requiresHookCount,
    issues,
    isCompatible: unsupportedCount === 0,
  };
}

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
    } else {
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
): Scene {
  const activeMaterials = new Set(scene.primitives.map((primitive) => materialForPrimitive(primitive)));
  const inactive = materials
    .map((material, index) => ({ index, material }))
    .filter(({ material }) => !activeMaterials.has(material));
  if (inactive.length === 0) return scene;
  const synthetic = materialsToSyntheticScene(inactive);
  return {
    ...scene,
    primitives: [...scene.primitives, ...synthetic.primitives],
  };
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
): Promise<void> {
  for (const [index, buffer] of (gltf.buffers ?? []).entries()) {
    if (buffers.has(index)) continue;
    if (buffer.uri == null) continue;
    if (buffer.uri.startsWith('data:')) {
      buffers.set(index, uint8ToArrayBuffer(decodeDataUri(buffer.uri, 'buffer', `buffer ${index}`)));
      continue;
    }
    const url = resolveUri(buffer.uri, baseUri, 'buffer');
    const data = await fetchArrayBuffer(url, options, 'buffer');
    buffers.set(index, data);
  }
}

async function resolveExternalImages(
  gltf: GltfJson,
  baseUri: string | undefined,
  options: LoadGltfAssetOptions,
): Promise<Map<number, GltfImageBytes>> {
  const out = new Map<number, GltfImageBytes>();
  for (const [index, image] of (gltf.images ?? []).entries()) {
    if (image.bufferView !== undefined) continue;
    if (image.uri == null || image.uri.startsWith('data:')) continue;
    const url = resolveUri(image.uri, baseUri, 'image');
    const fetched = await fetchImageBytes(url, options);
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
): Promise<{ readonly bytes: Uint8Array; readonly mimeType?: string }> {
  const cacheKey = { url, kind: 'image' } satisfies GltfAssetCacheKey;
  const cached = await options.cache?.get(cacheKey);
  if (cached !== undefined) return { bytes: new Uint8Array(cached) };
  const response = await fetchResource(url, options, 'image');
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
): Promise<ArrayBuffer> {
  const cacheKey = { url, kind } satisfies GltfAssetCacheKey;
  const cached = await options.cache?.get(cacheKey);
  if (cached !== undefined) return cached;
  const response = await fetchResource(url, options, kind);
  const data = await response.arrayBuffer();
  await options.cache?.set(cacheKey, data);
  return data;
}

async function fetchResource(
  url: string,
  options: LoadGltfAssetOptions,
  kind: GltfAssetResourceKind,
): Promise<GltfAssetFetchResponse> {
  const fetchFn = (options.fetch ?? globalThis.fetch) as GltfAssetFetch | undefined;
  if (typeof fetchFn !== 'function') {
    throw new GltfResourceNotFound({
      url,
      kind,
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
    throw new GltfFetchFailed({ url, kind, cause });
  }
  if (response.ok === false) {
    throw new GltfFetchFailed(Object.assign(
      { url, kind },
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

function resolveUri(uri: string, baseUri: string | URL | undefined, kind: GltfAssetResourceKind): string {
  if (uri.startsWith('data:')) return uri;
  try {
    return new URL(uri).toString();
  } catch {
    if (baseUri == null) {
      throw new GltfResourceNotFound({
        url: uri,
        kind,
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

function decodeDataUri(uri: string, kind: GltfAssetResourceKind, label: string): Uint8Array {
  const comma = uri.indexOf(',');
  if (comma < 0) {
    throw dataUriError(uri, kind, 'malformed-data-uri', `[vitrum/gltf-adapter] ${label} has a malformed data: URI.`);
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
      cause,
    );
  }
}

function dataUriError(
  uri: string,
  kind: GltfAssetResourceKind,
  reason: GltfResourceDecodeFailureReason,
  message: string,
  cause?: unknown,
): GltfResourceDecodeFailed {
  return new GltfResourceDecodeFailed({
    url: uri,
    kind,
    reason,
    message,
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
