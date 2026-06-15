// engineBridge.ts — one-call glTF asset preparation for Vitrum engines.
//
// This module intentionally lives in @vitrum/gltf-adapter instead of making
// @vitrum/engine depend on the adapter. Hosts can inject whichever engine
// factory they use, while the adapter owns asset loading, compatibility
// ranking, and controller construction.

import type { BackendId, Scene } from '@vitrum/core';
import {
  loadGltfAsset,
  type GltfAssetInput,
  type GltfAssetResult,
  type LoadGltfAssetOptions,
} from './assetLoader.js';
import type { GltfTextureDecodeReport } from './texturePipeline.js';
import type { GltfImportDiagnostic } from './gltfToScene.js';
import {
  createGltfSceneController,
  type GltfSceneController,
  type GltfScenePatchTarget,
} from './sceneController.js';
import type { GltfCompatibilityIssue } from './featureReport.js';

export type GltfEngineSelection = BackendId | 'recommended';
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
> extends LoadGltfAssetOptions {
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

  /** Backend to target. Defaults to the compatibility planner's top pick. */
  readonly backend?: GltfEngineSelection;

  /**
   * Compatibility gate for the selected backend.
   * - best-effort: return diagnostics, never reject for optional degradation.
   * - reject-unsupported: throw when the selected backend has unsupported rows.
   * - reject-degraded: throw on unsupported, approximate/fallback, or required
   *   host-hook rows.
   */
  readonly compatibilityMode?: GltfCompatibilityMode;

  /** Whether attaching/constructing an engine should call setScene(scene). */
  readonly attachScene?: boolean;
}

export interface GltfForEngineResult<TEngine extends GltfScenePatchTarget = GltfScenePatchTarget> {
  readonly asset: GltfAssetResult;
  readonly backend: BackendId;
  readonly engine?: TEngine;
  readonly controller: GltfSceneController;
  readonly attached: boolean;
  readonly textureDecodeReport: GltfTextureDecodeReport;
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
  const asset = await loadGltfAsset(input, options);
  const backend = selectBackend(asset, options.backend ?? 'recommended');
  enforceCompatibility(asset, backend, options.compatibilityMode ?? 'best-effort', options);

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
  });

  let engine = options.engine;
  if (!engine && options.createEngine) {
    engine = await options.createEngine({
      scene: asset.scene,
      backend,
      asset,
      options: options.engineOptions ?? ({} as TFactoryOptions),
    });
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
    ...(engine ? { engine } : {}),
    controller,
    attached,
    textureDecodeReport: asset.textureDecodeReport,
    warnings: [...asset.warnings, ...controller.warnings],
    diagnostics: asset.diagnostics,
  };
}

function selectBackend(asset: GltfAssetResult, selection: GltfEngineSelection): BackendId {
  if (selection === 'recommended') return asset.recommendedBackend.backend;
  return selection;
}

function enforceCompatibility<
  TEngine extends GltfScenePatchTarget,
  TFactoryOptions extends object,
>(
  asset: GltfAssetResult,
  backend: BackendId,
  mode: GltfCompatibilityMode,
  options: LoadGltfForEngineOptions<TEngine, TFactoryOptions>,
): void {
  if (mode === 'best-effort') return;
  const selected = asset.backendCompatibility.find((entry) =>
    entry.backend === backend && entry.profileId === backend,
  ) ?? asset.backendCompatibility.find((entry) => entry.backend === backend);
  if (!selected) {
    throw new Error(`[vitrum/gltf-adapter] No compatibility entry found for backend "${backend}".`);
  }
  const effectiveIssues = selected.issues.filter((issue) => !isSatisfiedHostHook(issue, options));
  const hardFailures = effectiveIssues.filter((issue) => issue.support === 'unsupported').length;
  const degradedFailures = effectiveIssues.filter((issue) =>
    issue.support !== 'native' && issue.support !== 'unsupported',
  ).length;
  const shouldReject = mode === 'reject-unsupported'
    ? hardFailures > 0
    : hardFailures + degradedFailures > 0;
  if (!shouldReject) return;

  const issues = effectiveIssues
    .filter((issue) => {
      if (mode === 'reject-unsupported') return issue.support === 'unsupported';
      return issue.support !== 'native';
    })
    .map(formatCompatibilityIssue)
    .join(', ');
  throw new Error(
    `[vitrum/gltf-adapter] Selected backend "${backend}" does not satisfy ` +
      `${mode}: ${issues || 'unknown compatibility issue'}.`,
  );
}

function formatCompatibilityIssue(issue: GltfCompatibilityIssue): string {
  return `${issue.category}:${issue.name}=${issue.support} at ${issue.path}`;
}

function isSatisfiedHostHook<
  TEngine extends GltfScenePatchTarget,
  TFactoryOptions extends object,
>(
  issue: GltfCompatibilityIssue,
  options: LoadGltfForEngineOptions<TEngine, TFactoryOptions>,
): boolean {
  if (issue.support !== 'requires-hook') return false;
  if (issue.name === 'KHR_draco_mesh_compression') return typeof options.dracoDecode === 'function';
  if (issue.name === 'EXT_meshopt_compression') return typeof options.meshoptDecode === 'function';
  if (issue.name === 'KHR_texture_basisu' || issue.name === 'EXT_texture_webp' || issue.name === 'MSFT_texture_dds') {
    return (options.textureSourceExtensions ?? []).includes(issue.name);
  }
  return false;
}
