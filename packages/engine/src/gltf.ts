// @vitrum/engine/gltf — one-import glTF loading helpers.
//
// The glTF adapter remains the owner of asset loading, feature reporting,
// compatibility checks, texture diagnostics, and controller construction. This
// subpath only injects @vitrum/engine's createEngine facade for hosts that want
// a single import path.

import { loadGltfForEngine } from '@vitrum/gltf-adapter';
import type {
  GltfAssetInput,
  GltfAssetResult,
  GltfEngineSelection,
  GltfCompatibilityMode,
  GltfCompatibilityIssue,
  GltfBackendProfileId,
  GltfForEngineResult,
  GltfImportDiagnostic,
  GltfSceneController,
  DecodeSceneTextureDiagnostic,
  GltfTextureDecodeReport,
  LoadGltfForEngineOptions,
} from '@vitrum/gltf-adapter';
import { probeAdapterProfile } from './adapterProfile.js';
import { createEngine } from './createEngine.js';
import {
  createProgressiveEngine,
  type CreateProgressiveEngineOptions,
  type ProgressiveEngineHandle,
} from './createProgressiveEngine.js';
import type {
  CreateEngineBackendId,
  CreateEngineOptions,
  EnginePreference,
  EngineWithBackendId,
} from './createEngine.js';

export { loadGltfForEngine } from '@vitrum/gltf-adapter';
export type {
  DecodeSceneTextureDiagnostic,
  DecodeSceneTextureDiagnosticCode,
  GltfBackendTextureStatus,
  GltfAssetResult,
  GltfCompatibilityMode,
  GltfEngineSelection,
  GltfForEngineResult,
  GltfImportDiagnostic,
  GltfMaterialTextureField,
  GltfTextureColorSpace,
  GltfTextureDecodeReport,
  GltfTextureDecodeReportEntry,
  GltfTextureHandleKind,
  LoadGltfForEngineOptions,
} from '@vitrum/gltf-adapter';

export type GltfCreateEngineOptions =
  Omit<CreateEngineOptions, 'scene' | 'prefer' | 'gltfAsset'> & {
    readonly prefer?: EnginePreference;
  };

export type LoadGltfWithEngineOptions = Omit<
  LoadGltfForEngineOptions<EngineWithBackendId, GltfCreateEngineOptions>,
  'createEngine' | 'engineOptions'
> & {
  readonly engineOptions?: GltfCreateEngineOptions;
};

export type GltfCreateProgressiveEngineOptions =
  Omit<CreateProgressiveEngineOptions, 'scene' | 'controller'>;

export type LoadGltfWithProgressiveEngineOptions = Omit<
  LoadGltfForEngineOptions,
  'engine' | 'createEngine' | 'engineOptions' | 'attachScene' | 'backend' | 'runtimeProfile'
> & {
  readonly engineOptions: GltfCreateProgressiveEngineOptions;
};

export interface GltfProgressiveEngineResult {
  readonly asset: GltfAssetResult;
  readonly backend: 'pt-webgpu';
  readonly profileId: GltfBackendProfileId;
  readonly engine: ProgressiveEngineHandle;
  readonly controller: GltfSceneController;
  readonly attached: true;
  readonly textureDecodeReport: GltfTextureDecodeReport;
  readonly decodedTextureCount: number;
  readonly unchangedTextureCount: number;
  readonly textureDecodeDiagnostics: readonly DecodeSceneTextureDiagnostic[];
  readonly textureDecodeWarnings: readonly string[];
  readonly warnings: readonly string[];
  readonly diagnostics: readonly GltfImportDiagnostic[];
}

export async function loadGltfWithEngine(
  input: Parameters<typeof loadGltfForEngine>[0],
  options: LoadGltfWithEngineOptions = {},
): Promise<GltfForEngineResult<EngineWithBackendId>> {
  const { engineOptions, ...adapterOptions } = options;
  if (adapterOptions.engine != null) {
    const { engine, attachScene, ...loadOptions } = adapterOptions;
    const loaded = await loadGltfForEngine<EngineWithBackendId, GltfCreateEngineOptions>(input, {
      ...loadOptions,
      backend: engine.backendId,
      attachScene: false,
      engineOptions: engineOptions ?? ({} as GltfCreateEngineOptions),
    });
    const runtimeProfileId = await resolvePtWebgpuRuntimeProfile(
      engine.backendId,
      adapterOptions.compatibilityMode ?? 'best-effort',
      loaded.asset,
      adapterOptions,
    );
    loaded.controller.attachEngine(engine, { setScene: attachScene ?? true });
    return {
      ...loaded,
      backend: engine.backendId,
      profileId: runtimeProfileId ?? loaded.profileId,
      engine,
      attached: true,
      warnings: [...loaded.asset.warnings, ...loaded.textureDecodeWarnings, ...loaded.controller.warnings],
    };
  }

  let runtimeProfileId: GltfBackendProfileId | undefined;
  const loaded = await loadGltfForEngine<EngineWithBackendId, GltfCreateEngineOptions>(input, {
    ...adapterOptions,
    engineOptions: engineOptions ?? ({} as GltfCreateEngineOptions),
    createEngine: async ({ scene, backend, asset, options: createOptions }) => {
      runtimeProfileId = await resolvePtWebgpuRuntimeProfile(
        backend,
        adapterOptions.compatibilityMode ?? 'best-effort',
        asset,
        adapterOptions,
      );
      return await createEngine({
        ...createOptions,
        scene,
        gltfAsset: asset,
        prefer: preferForSelectedBackend(backend, createOptions.prefer),
      });
    },
  });
  return loaded.backend === 'pt-webgpu' && runtimeProfileId != null
    ? { ...loaded, profileId: runtimeProfileId }
    : loaded;
}

export async function loadGltfWithProgressiveEngine(
  input: GltfAssetInput,
  options: LoadGltfWithProgressiveEngineOptions,
): Promise<GltfProgressiveEngineResult> {
  const { engineOptions, ...adapterOptions } = options;
  const loaded = await loadGltfForEngine(input, {
    ...adapterOptions,
    backend: 'pt-webgpu',
    runtimeProfile: 'pt-webgpu',
    attachScene: false,
  });
  const engine = await createProgressiveEngine({
    ...engineOptions,
    scene: loaded.asset.scene,
    controller: loaded.controller,
  });

  return {
    asset: loaded.asset,
    backend: 'pt-webgpu',
    profileId: loaded.profileId,
    engine,
    controller: loaded.controller,
    attached: true,
    textureDecodeReport: loaded.textureDecodeReport,
    decodedTextureCount: loaded.decodedTextureCount,
    unchangedTextureCount: loaded.unchangedTextureCount,
    textureDecodeDiagnostics: loaded.textureDecodeDiagnostics,
    textureDecodeWarnings: loaded.textureDecodeWarnings,
    warnings: loaded.warnings,
    diagnostics: loaded.diagnostics,
  };
}

async function resolvePtWebgpuRuntimeProfile(
  backend: CreateEngineBackendId | GltfEngineSelection,
  compatibilityMode: GltfCompatibilityMode,
  asset: GltfAssetResult,
  options: LoadGltfWithEngineOptions,
): Promise<GltfBackendProfileId | undefined> {
  if (backend !== 'pt-webgpu') return undefined;

  const profile = await probeAdapterProfile();
  if (profile.ptWebgpuTier === 'full') return 'pt-webgpu';

  const profileId = profile.ptWebgpuTier === 'lite' ? 'pt-webgpu-lite' : 'pt-webgpu';
  if (compatibilityMode === 'best-effort') return profileId;
  const selected = asset.backendCompatibility.find((entry) => entry.profileId === profileId);
  if (selected != null) {
    const effectiveIssues = selected.issues.filter((issue) =>
      !isSatisfiedRuntimeCompatibilityIssue(issue, options, asset)
    );
    const rejectedIssues = rejectedIssuesForMode(effectiveIssues, compatibilityMode);
    if (rejectedIssues.length === 0) return profileId;

    throw new Error(
      `[vitrum/engine/gltf] Selected backend "pt-webgpu" resolves to ` +
        `"${profile.ptWebgpuTier}" trace tier, which does not satisfy ` +
        `${compatibilityMode}: ${formatRuntimeCompatibilityIssues(rejectedIssues)}. ` +
        `Use compatibilityMode:"best-effort", select "pt-webgl2", or run on a full-tier WebGPU adapter.`,
    );
  }

  throw new Error(
    `[vitrum/engine/gltf] Selected backend "pt-webgpu" resolves to ` +
      `"${profile.ptWebgpuTier}" trace tier, but the glTF asset has no compatibility row ` +
      `for runtime profile "${profileId}". ` +
      `Use compatibilityMode:"best-effort", select "pt-webgl2", or run on a full-tier WebGPU adapter.`,
  );
}

function rejectedIssuesForMode(
  issues: readonly GltfCompatibilityIssue[],
  compatibilityMode: Exclude<GltfCompatibilityMode, 'best-effort'>,
): readonly GltfCompatibilityIssue[] {
  if (compatibilityMode === 'reject-unsupported') {
    return issues.filter((issue) => issue.support === 'unsupported');
  }
  return issues.filter((issue) => issue.support !== 'native');
}

function formatRuntimeCompatibilityIssues(issues: readonly GltfCompatibilityIssue[]): string {
  return issues
    .map((issue) => `${issue.category}:${issue.name}=${issue.support} at ${issue.path}`)
    .join(', ');
}

function isSatisfiedRuntimeCompatibilityIssue(
  issue: GltfCompatibilityIssue,
  options: LoadGltfWithEngineOptions,
  asset: GltfAssetResult,
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

  if (issue.name === 'KHR_materials_pbrSpecularGlossiness.specularGlossinessTexture.glossinessAlpha') {
    const decoded = decodedAssetView(asset);
    if (decoded == null) return false;
    const bakeUnavailable = decoded.textureDecodeDiagnostics.some((diagnostic) =>
      diagnostic.code === 'spec-gloss-alpha-bake-unavailable'
    );
    const bakedRoughnessMap = asset.textureDecodeReport.entries.some((entry) =>
      entry.materialField === 'roughnessMap' && entry.handleKind === 'pixel-data'
    );
    return bakedRoughnessMap && !bakeUnavailable;
  }

  return false;
}

function decodedAssetView(
  asset: GltfAssetResult,
): { readonly textureDecodeDiagnostics: readonly DecodeSceneTextureDiagnostic[] } | null {
  const maybe = asset as GltfAssetResult & {
    readonly textureDecodeDiagnostics?: readonly DecodeSceneTextureDiagnostic[];
  };
  return Array.isArray(maybe.textureDecodeDiagnostics)
    ? { textureDecodeDiagnostics: maybe.textureDecodeDiagnostics }
    : null;
}

function preferForSelectedBackend(
  backend: CreateEngineBackendId | GltfEngineSelection,
  fallback: EnginePreference | undefined,
): EnginePreference {
  if (backend === 'walkaround-hybrid') return 'realtime';
  if (backend === 'pt-webgpu') return 'quality-webgpu';
  if (backend === 'pt-webgl2') return 'quality';
  return fallback ?? 'auto';
}
