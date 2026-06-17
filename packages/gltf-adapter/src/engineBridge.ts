// engineBridge.ts — one-call glTF asset preparation for Vitrum engines.
//
// This module intentionally lives in @vitrum/gltf-adapter instead of making
// @vitrum/engine depend on the adapter. Hosts can inject whichever engine
// factory they use, while the adapter owns asset loading, compatibility
// ranking, and controller construction.

import type { BackendId, Scene } from '@vitrum/core';
import {
  loadGltfAndDecodeTextures,
  loadGltfAsset,
  type GltfAssetInput,
  type GltfAssetResult,
  type GltfDecodedAssetResult,
  type LoadGltfAndDecodeTexturesOptions,
} from './assetLoader.js';
import type {
  DecodeSceneTextureDiagnostic,
  GltfBackendTextureStatus,
  GltfTextureDecodeReport,
  GltfTextureDecodeReportEntry,
} from './texturePipeline.js';
import type { GltfImportDiagnostic, GltfImportDiagnosticCode } from './gltfToScene.js';
import {
  createGltfSceneController,
  type GltfSceneController,
  type GltfScenePatchTarget,
} from './sceneController.js';
import type { GltfBackendProfileId, GltfCompatibilityIssue } from './featureReport.js';

export type GltfEngineSelection = GltfBackendProfileId | 'recommended';
export type GltfCompatibilityMode = 'best-effort' | 'reject-unsupported' | 'reject-degraded';

export interface GltfEngineFactoryInput<TFactoryOptions extends object = Record<string, never>> {
  readonly scene: Scene;
  readonly backend: BackendId;
  readonly asset: GltfAssetResult;
  readonly options: TFactoryOptions;
}

export type GltfEngineFactory<
  TEngine extends GltfScenePatchTarget = GltfScenePatchTarget,
  TFactoryOptions extends object = Record<string, never>,
> = (input: GltfEngineFactoryInput<TFactoryOptions>) => Promise<TEngine> | TEngine;

export interface LoadGltfForEngineOptions<
  TEngine extends GltfScenePatchTarget = GltfScenePatchTarget,
  TFactoryOptions extends object = Record<string, never>,
> extends LoadGltfAndDecodeTexturesOptions {
  /**
   * Run the CPU texture decode bridge before engine construction/attachment.
   * This promotes raw-image `TextureRef` handles into backend-ready CPU-linear
   * handles and surfaces `textureDecodeDiagnostics` / `textureDecodeWarnings` on
   * the returned bridge result. It defaults to false unless a decode-specific
   * option such as `decodePixels`, `textureTarget`, `maxTextureSize`, or
   * `warnOnNpotRepeatWrap` is supplied.
   */
  readonly decodeTextures?: boolean;

  /**
   * Existing engine to attach the loaded scene/controller to. Mutually useful
   * with `createEngine`: pass one or the other. If both are supplied,
   * `engine` wins and the factory is ignored.
   */
  readonly engine?: TEngine;

  /**
   * Host-injected engine factory. The adapter passes the imported scene, the
   * selected backend id, the full asset result, and the opaque `engineOptions`
   * bag. This avoids a package dependency from @vitrum/gltf-adapter to
   * @vitrum/engine while still giving hosts a one-call path.
   */
  readonly createEngine?: GltfEngineFactory<TEngine, TFactoryOptions>;
  readonly engineOptions?: TFactoryOptions;

  /**
   * Backend/profile to target. Defaults to the compatibility planner's top
   * pick. Passing `'pt-webgpu-lite'` validates against the constrained lite
   * profile while still passing the real backend id (`'pt-webgpu'`) to the
   * injected engine factory.
   */
  readonly backend?: GltfEngineSelection;

  /**
   * Concrete runtime profile when the host already knows the backend tier.
   * This is most useful for adapter-only hosts that construct `pt-webgpu`
   * themselves and know that the negotiated device is lite-tier. The profile
   * must belong to the selected backend family; for example, `backend:'pt-webgpu'`
   * may be validated as `runtimeProfile:'pt-webgpu-lite'`, but it cannot be
   * redirected to a WebGL or walkaround profile.
   */
  readonly runtimeProfile?: GltfBackendProfileId;

  /**
   * Compatibility gate for the selected backend.
   * - best-effort: return diagnostics, never reject for optional degradation.
   * - reject-unsupported: throw when the selected backend has unsupported rows.
   * - reject-degraded: throw on unsupported, approximate/fallback, or required
   *   host-hook rows.
   */
  readonly compatibilityMode?: GltfCompatibilityMode;

  /**
   * Strict texture-readiness escape hatch for host-owned opaque handles.
   * By default, `compatibilityMode:'reject-degraded'` rejects texture refs whose
   * decode report says the selected backend sees only an opaque handle. Set this
   * to true (or an explicit backend list) only when the host/factory guarantees
   * those opaque handles are already uploadable by that backend.
   */
  readonly opaqueTextureHandlesReady?: boolean | readonly BackendId[];

  /** Whether attaching/constructing an engine should call setScene(scene). */
  readonly attachScene?: boolean;
}

export interface GltfForEngineResult<TEngine extends GltfScenePatchTarget = GltfScenePatchTarget> {
  readonly asset: GltfAssetResult;
  readonly backend: BackendId;
  readonly profileId: GltfBackendProfileId;
  readonly engine?: TEngine;
  readonly controller: GltfSceneController;
  readonly attached: boolean;
  readonly textureDecodeReport: GltfTextureDecodeReport;
  readonly decodedTextureCount: number;
  readonly unchangedTextureCount: number;
  readonly textureDecodeDiagnostics: readonly DecodeSceneTextureDiagnostic[];
  readonly textureDecodeWarnings: readonly string[];
  readonly warnings: readonly string[];
  readonly diagnostics: readonly GltfImportDiagnostic[];
}

export async function loadGltfForEngine<
  TEngine extends GltfScenePatchTarget = GltfScenePatchTarget,
  TFactoryOptions extends object = Record<string, never>,
>(
  input: GltfAssetInput,
  options: LoadGltfForEngineOptions<TEngine, TFactoryOptions> = {},
): Promise<GltfForEngineResult<TEngine>> {
  const asset = shouldDecodeTextures(options)
    ? await loadGltfAndDecodeTextures(input, options)
    : await loadGltfAsset(input, options);
  const selected = selectBackendTarget(asset, options.backend ?? 'recommended');
  const selectedBackend = selected.backend;
  let selectedProfileId = resolveRuntimeProfile(selectedBackend, selected.profileId, options.runtimeProfile);
  const compatibilityMode = options.compatibilityMode ?? 'best-effort';
  enforceCompatibility(asset, selectedBackend, selectedProfileId, compatibilityMode, options);

  const controller = createGltfSceneController({
    gltf: asset.gltf,
    sceneIndex: asset.sceneIndex,
    scene: asset.scene,
    warnings: asset.warnings,
    diagnostics: asset.diagnostics,
    animations: asset.animations,
    animationTargets: asset.animationTargets,
    ...(asset.convertedMaterials !== undefined ? { convertedMaterials: asset.convertedMaterials } : {}),
    ...(asset.materialVariantBindings !== undefined ? { materialVariantBindings: asset.materialVariantBindings } : {}),
    ...(asset.instancingBindings !== undefined ? { instancingBindings: asset.instancingBindings } : {}),
  });

  let engine = options.engine;
  if (!engine && options.createEngine) {
    engine = await options.createEngine({
      scene: asset.scene,
      backend: selectedBackend,
      asset,
      options: options.engineOptions ?? ({} as TFactoryOptions),
    });
  }

  const backend = backendIdFromEngine(engine) ?? selectedBackend;
  if (backend !== selectedBackend) {
    selectedProfileId = backend;
    enforceCompatibility(asset, backend, backend, compatibilityMode, options, 'Actual engine backend');
  }

  const attachScene = options.attachScene ?? true;
  let attached = false;
  if (engine) {
    controller.attachEngine(engine, { setScene: attachScene });
    attached = true;
  }

  return {
    asset,
    backend,
    profileId: selectedProfileId,
    ...(engine ? { engine } : {}),
    controller,
    attached,
    textureDecodeReport: asset.textureDecodeReport,
    decodedTextureCount: decodedTextureCount(asset),
    unchangedTextureCount: unchangedTextureCount(asset),
    textureDecodeDiagnostics: textureDecodeDiagnostics(asset),
    textureDecodeWarnings: textureDecodeWarnings(asset),
    warnings: [...asset.warnings, ...textureDecodeWarnings(asset), ...controller.warnings],
    diagnostics: asset.diagnostics,
  };
}

function shouldDecodeTextures<
  TEngine extends GltfScenePatchTarget,
  TFactoryOptions extends object,
>(options: LoadGltfForEngineOptions<TEngine, TFactoryOptions>): boolean {
  return options.decodeTextures === true ||
    options.textureTarget !== undefined ||
    options.decodePixels !== undefined ||
    options.maxTextureSize !== undefined ||
    options.warnOnNpotRepeatWrap !== undefined ||
    options.onTextureDiagnostic !== undefined ||
    options.onTextureWarning !== undefined;
}

function isDecodedAsset(asset: GltfAssetResult | GltfDecodedAssetResult): asset is GltfDecodedAssetResult {
  return 'textureDecodeDiagnostics' in asset;
}

function decodedTextureCount(asset: GltfAssetResult | GltfDecodedAssetResult): number {
  return isDecodedAsset(asset) ? asset.decodedTextureCount : 0;
}

function unchangedTextureCount(asset: GltfAssetResult | GltfDecodedAssetResult): number {
  return isDecodedAsset(asset) ? asset.unchangedTextureCount : 0;
}

function textureDecodeDiagnostics(
  asset: GltfAssetResult | GltfDecodedAssetResult,
): readonly DecodeSceneTextureDiagnostic[] {
  return isDecodedAsset(asset) ? asset.textureDecodeDiagnostics : [];
}

function textureDecodeWarnings(asset: GltfAssetResult | GltfDecodedAssetResult): readonly string[] {
  return isDecodedAsset(asset) ? asset.textureDecodeWarnings : [];
}

function selectBackendTarget(
  asset: GltfAssetResult,
  selection: GltfEngineSelection,
): { readonly backend: BackendId; readonly profileId: GltfBackendProfileId } {
  if (selection === 'recommended') {
    return {
      backend: asset.recommendedBackend.backend,
      profileId: asset.recommendedBackend.profileId,
    };
  }
  if (selection === 'pt-webgpu-lite') {
    return { backend: 'pt-webgpu', profileId: 'pt-webgpu-lite' };
  }
  return { backend: selection, profileId: selection };
}

function resolveRuntimeProfile(
  selectedBackend: BackendId,
  selectedProfileId: GltfBackendProfileId,
  runtimeProfile: GltfBackendProfileId | undefined,
): GltfBackendProfileId {
  if (runtimeProfile === undefined) return selectedProfileId;
  const runtimeBackend = backendFromProfileId(runtimeProfile);
  if (runtimeBackend !== selectedBackend) {
    throw new Error(
      `[vitrum/gltf-adapter] runtimeProfile ${formatBackendProfile(runtimeBackend, runtimeProfile)} ` +
        `does not match selected backend ${formatBackendProfile(selectedBackend, selectedProfileId)}.`,
    );
  }
  return runtimeProfile;
}

function backendFromProfileId(profileId: GltfBackendProfileId): BackendId {
  return profileId === 'pt-webgpu-lite' ? 'pt-webgpu' : profileId;
}

function enforceCompatibility<
  TEngine extends GltfScenePatchTarget,
  TFactoryOptions extends object,
>(
  asset: GltfAssetResult,
  backend: BackendId,
  profileId: GltfBackendProfileId,
  mode: GltfCompatibilityMode,
  options: LoadGltfForEngineOptions<TEngine, TFactoryOptions>,
  label = 'Selected backend',
): void {
  if (mode === 'best-effort') return;
  const selected = asset.backendCompatibility.find((entry) =>
    entry.backend === backend && entry.profileId === profileId,
  );
  if (!selected) {
    throw new Error(
      `[vitrum/gltf-adapter] No compatibility entry found for ${formatBackendProfile(backend, profileId)}.`,
    );
  }
  const effectiveIssues = selected.issues.filter((issue) => !isSatisfiedCompatibilityIssue(issue, options, asset));
  const rejectedIssues = effectiveIssues
    .filter((issue) => {
      if (mode === 'reject-unsupported') return issue.support === 'unsupported';
      return issue.support !== 'native';
    })
    .map(formatCompatibilityIssue);
  const rejectedImportDiagnostics = importDiagnosticFailures(asset.diagnostics, mode);
  const rejectedTextureReadiness = textureReadinessFailures(asset.textureDecodeReport, backend, mode, options);
  const failures = [
    ...rejectedIssues,
    ...rejectedImportDiagnostics,
    ...rejectedTextureReadiness,
  ];
  if (failures.length === 0) return;

  const issues = failures
    .join(', ');
  throw new Error(
    `[vitrum/gltf-adapter] ${label} ${formatBackendProfile(backend, profileId)} does not satisfy ` +
      `${mode}: ${issues || 'unknown compatibility issue'}.`,
  );
}

function formatBackendProfile(backend: BackendId, profileId: GltfBackendProfileId): string {
  return profileId === backend
    ? `"${backend}"`
    : `"${backend}" profile "${profileId}"`;
}

const UNSUPPORTED_IMPORT_DIAGNOSTICS: ReadonlySet<GltfImportDiagnosticCode> = new Set([
  'scene-not-found',
  'unsupported-required-extension',
  'unsupported-primitive-mode',
  'unresolved-compression',
  'draco-buffer-view-unavailable',
  'draco-decode-hook-failed',
  'draco-decode-hook-missing',
  'draco-geometry-unusable',
  'meshopt-buffer-unavailable',
  'meshopt-decode-hook-failed',
  'meshopt-decode-hook-missing',
  'meshopt-decoded-byte-length-mismatch',
  'missing-position',
  'unreadable-position',
  'unreadable-indices',
  'empty-triangulated-primitive',
  'ignored-skin-attributes',
  'incomplete-skin-attributes',
  'missing-animation-sampler',
  'unreadable-animation-sampler',
  'unsupported-animation-target-path',
  'missing-animation-target-node',
  'animation-target-node-not-found',
  'invalid-animation-output-count',
  'dropped-animation',
  'ignored-vertex-color-set',
  'ignored-morph-target-texcoord',
]);

const DEGRADED_IMPORT_DIAGNOSTICS: ReadonlySet<GltfImportDiagnosticCode> = new Set([
  'unsupported-version',
  'ignored-camera',
  'double-sided-material',
  'ignored-gpu-instancing',
  'fallback-generated-primitive-mode',
  'generated-tangents',
  'missing-tangent-texcoord',
  'tangent-generation-failed',
  'skin-rest-pose',
  'ignored-material-texcoord',
  'unknown-animation-interpolation',
  'external-image-uri',
  'malformed-data-uri',
  'data-uri-atob-unavailable',
  'data-uri-decode-failed',
  'image-decoder-missing',
  'disabled-texture-source-extension',
  'sparse-indices-buffer-view-not-found',
  'sparse-indices-buffer-unavailable',
  'sparse-values-buffer-view-not-found',
  'sparse-values-buffer-unavailable',
  'invalid-sparse-indices-component-type',
  'sparse-index-out-of-range',
  'invalid-material-dispersion',
  'unsupported-material-extension',
  'unknown-material-extension',
]);

function importDiagnosticFailures(
  diagnostics: readonly GltfImportDiagnostic[],
  mode: GltfCompatibilityMode,
): string[] {
  if (mode === 'best-effort') return [];
  return diagnostics
    .map((diagnostic) => {
      const support = importDiagnosticSupport(diagnostic.code);
      if (support == null) return undefined;
      if (mode === 'reject-unsupported' && support !== 'unsupported') return undefined;
      return `import:${diagnostic.code}=${support} at ${diagnostic.path}`;
    })
    .filter((message): message is string => message !== undefined);
}

function importDiagnosticSupport(code: GltfImportDiagnosticCode): 'unsupported' | 'approximate' | undefined {
  if (UNSUPPORTED_IMPORT_DIAGNOSTICS.has(code)) return 'unsupported';
  if (DEGRADED_IMPORT_DIAGNOSTICS.has(code)) return 'approximate';
  return undefined;
}

function textureReadinessFailures<
  TEngine extends GltfScenePatchTarget,
  TFactoryOptions extends object,
>(
  report: GltfTextureDecodeReport,
  backend: BackendId,
  mode: GltfCompatibilityMode,
  options: LoadGltfForEngineOptions<TEngine, TFactoryOptions>,
): string[] {
  if (mode === 'best-effort') return [];
  const key = textureReadinessKey(backend);
  return report.entries
    .map((entry) => {
      const status = entry.backendReadiness[key];
      const support = textureReadinessSupport(status, options, backend);
      if (support == null) return undefined;
      if (mode === 'reject-unsupported' && support !== 'unsupported') return undefined;
      return formatTextureReadinessFailure(entry, status, support);
    })
    .filter((message): message is string => message !== undefined);
}

function textureReadinessSupport<
  TEngine extends GltfScenePatchTarget,
  TFactoryOptions extends object,
>(
  status: GltfBackendTextureStatus,
  options: LoadGltfForEngineOptions<TEngine, TFactoryOptions>,
  backend: BackendId,
): 'unsupported' | 'requires-hook' | undefined {
  if (status === 'ready') return undefined;
  if (status === 'opaque' && opaqueTextureHandlesReadyForBackend(options, backend)) return undefined;
  return status === 'ignored' ? 'unsupported' : 'requires-hook';
}

function formatTextureReadinessFailure(
  entry: GltfTextureDecodeReportEntry,
  status: GltfBackendTextureStatus,
  support: 'unsupported' | 'requires-hook',
): string {
  return `texture:${entry.materialField}=${support} at ${entry.path} (${status})`;
}

function textureReadinessKey(backend: BackendId): keyof GltfTextureDecodeReportEntry['backendReadiness'] {
  if (backend === 'pt-webgl2') return 'ptWebgl2';
  if (backend === 'pt-webgpu') return 'ptWebgpu';
  return 'walkaroundHybrid';
}

function opaqueTextureHandlesReadyForBackend<
  TEngine extends GltfScenePatchTarget,
  TFactoryOptions extends object,
>(
  options: LoadGltfForEngineOptions<TEngine, TFactoryOptions>,
  backend: BackendId,
): boolean {
  const policy = options.opaqueTextureHandlesReady;
  return policy === true || (Array.isArray(policy) && policy.includes(backend));
}

function backendIdFromEngine(engine: GltfScenePatchTarget | undefined): BackendId | undefined {
  if (engine == null) return undefined;
  const backendId = (engine as { readonly backendId?: unknown }).backendId;
  return isBackendId(backendId) ? backendId : undefined;
}

function isBackendId(value: unknown): value is BackendId {
  return value === 'walkaround-hybrid' || value === 'pt-webgl2' || value === 'pt-webgpu';
}

function formatCompatibilityIssue(issue: GltfCompatibilityIssue): string {
  return `${issue.category}:${issue.name}=${issue.support} at ${issue.path}`;
}

function isSatisfiedCompatibilityIssue<
  TEngine extends GltfScenePatchTarget,
  TFactoryOptions extends object,
>(
  issue: GltfCompatibilityIssue,
  options: LoadGltfForEngineOptions<TEngine, TFactoryOptions>,
  asset: GltfAssetResult | GltfDecodedAssetResult,
): boolean {
  if (issue.support === 'requires-hook') {
    if (issue.name === 'KHR_draco_mesh_compression') return typeof options.dracoDecode === 'function';
    if (issue.name === 'EXT_meshopt_compression' || issue.name === 'KHR_meshopt_compression') {
      return typeof options.meshoptDecode === 'function';
    }
    if (issue.name === 'KHR_texture_basisu' || issue.name === 'EXT_texture_webp' || issue.name === 'MSFT_texture_dds') {
      return (options.textureSourceExtensions ?? []).includes(issue.name) && typeof options.decodeImage === 'function';
    }
    return false;
  }
  if (
    issue.name === 'KHR_materials_pbrSpecularGlossiness.specularGlossinessTexture.glossinessAlpha' &&
    isDecodedAsset(asset)
  ) {
    const bakeUnavailable = asset.textureDecodeDiagnostics.some((diagnostic) =>
      diagnostic.code === 'spec-gloss-alpha-bake-unavailable'
    );
    const bakedRoughnessMap = asset.textureDecodeReport.entries.some((entry) =>
      entry.materialField === 'roughnessMap' && entry.handleKind === 'pixel-data'
    );
    return bakedRoughnessMap && !bakeUnavailable;
  }
  return false;
}
