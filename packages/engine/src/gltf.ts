// @vitrum/engine/gltf — one-import glTF loading helpers.
//
// The glTF adapter remains the owner of asset loading, feature reporting,
// compatibility checks, texture diagnostics, and controller construction. This
// subpath only injects @vitrum/engine's createEngine facade for hosts that want
// a single import path.

import { GltfCompatibilityError, loadGltfForEngine } from '@vitrum/gltf-adapter';
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

export { GltfCompatibilityError, loadGltfForEngine } from '@vitrum/gltf-adapter';
export type {
  DecodeSceneTextureDiagnostic,
  DecodeSceneTextureDiagnosticCode,
  GltfCompatibilityErrorCode,
  GltfCompatibilityErrorInit,
  GltfBackendTextureStatus,
  GltfAssetResult,
  GltfCompatibilityMode,
  GltfEngineSelection,
  GltfForEngineResult,
  GltfImportDiagnostic,
  GltfMaterialTextureField,
  GltfNpotRepeatWrapPolicy,
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
  const preferredAdapterBackend =
    adapterOptions.backend ?? backendSelectionForExplicitPrefer(engineOptions?.prefer);
  if (adapterOptions.engine != null) {
    const { engine, attachScene, ...baseLoadOptions } = adapterOptions;
    const runtimeProfile = await maybeProbePtWebgpuRuntimeProfile(engine.backendId, baseLoadOptions);
    const loadOptions = withRuntimeTextureCap(baseLoadOptions, runtimeProfile);
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
      loadOptions,
      runtimeProfile,
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
  const runtimeProfile = await maybeProbePtWebgpuRuntimeProfile(preferredAdapterBackend, adapterOptions);
  const loadOptions = withRuntimeTextureCap(adapterOptions, runtimeProfile);
  const loaded = await loadGltfForEngine<EngineWithBackendId, GltfCreateEngineOptions>(input, {
    ...loadOptions,
    ...(preferredAdapterBackend !== undefined ? { backend: preferredAdapterBackend } : {}),
    engineOptions: engineOptions ?? ({} as GltfCreateEngineOptions),
    createEngine: async ({ scene, backend, asset, options: createOptions }) => {
      runtimeProfileId = await resolvePtWebgpuRuntimeProfile(
        backend,
        loadOptions.compatibilityMode ?? 'best-effort',
        asset,
        loadOptions,
        runtimeProfile,
      );
      const engine = await createEngine({
        ...createOptions,
        scene,
        gltfAsset: asset,
        prefer: preferForSelectedBackend(backend, createOptions.prefer),
      });
      if (backend !== 'pt-webgpu' && engine.backendId === 'pt-webgpu') {
        try {
          runtimeProfileId = await resolvePtWebgpuRuntimeProfile(
            engine.backendId,
            loadOptions.compatibilityMode ?? 'best-effort',
            asset,
            loadOptions,
            runtimeProfile,
          );
        } catch (err) {
          disposeEngineAfterRejectedGltfRuntimeProfile(engine);
          throw err;
        }
      }
      return engine;
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
  loaded.controller.attachEngine(engine.coordinator, { setScene: false });

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
  runtimeProfile: PtWebgpuRuntimeProfile | null = null,
): Promise<GltfBackendProfileId | undefined> {
  if (backend !== 'pt-webgpu') return undefined;

  if (options.runtimeProfile !== undefined) {
    const runtimeBackend = backendFromProfileId(options.runtimeProfile);
    if (runtimeBackend !== 'pt-webgpu') {
      throw new GltfCompatibilityError({
        code: 'GLTF_RUNTIME_PROFILE_MISMATCH',
        message:
          `[vitrum/engine/gltf] runtimeProfile ${formatBackendProfile(runtimeBackend, options.runtimeProfile)} ` +
          `does not match selected backend "pt-webgpu".`,
        backend: 'pt-webgpu',
        profileId: 'pt-webgpu',
        runtimeProfile: options.runtimeProfile,
      });
    }
    validatePtWebgpuRuntimeProfile(
      options.runtimeProfile,
      'explicit',
      compatibilityMode,
      asset,
      options,
    );
    return options.runtimeProfile;
  }

  const profile = runtimeProfile ?? await probeAdapterProfile();
  if (profile.ptWebgpuTier === 'full') return 'pt-webgpu';
  if (profile.ptWebgpuTier === 'none') {
    validatePtWebgpuRuntimeUnavailable(compatibilityMode);
    return undefined;
  }

  const profileId = 'pt-webgpu-lite';
  validatePtWebgpuRuntimeProfile(
    profileId,
    profile.ptWebgpuTier,
    compatibilityMode,
    asset,
    options,
  );
  return profileId;
}

type PtWebgpuRuntimeProfile = Awaited<ReturnType<typeof probeAdapterProfile>>;

async function maybeProbePtWebgpuRuntimeProfile(
  backend: CreateEngineBackendId | GltfEngineSelection | undefined,
  options: Pick<LoadGltfWithEngineOptions, 'runtimeProfile' | 'maxTextureSize'>,
): Promise<PtWebgpuRuntimeProfile | null> {
  if (options.runtimeProfile !== undefined) return null;
  if (backend !== 'pt-webgpu' && backend !== 'pt-webgpu-lite') return null;
  return probeAdapterProfile();
}

function withRuntimeTextureCap<TOptions extends Pick<LoadGltfWithEngineOptions, 'maxTextureSize'>>(
  options: TOptions,
  runtimeProfile: PtWebgpuRuntimeProfile | null,
): TOptions {
  if (options.maxTextureSize !== undefined || runtimeProfile == null) return options;
  const maxTextureSize = runtimeMaxTextureSize(runtimeProfile);
  return maxTextureSize === undefined
    ? options
    : { ...options, maxTextureSize };
}

function runtimeMaxTextureSize(profile: PtWebgpuRuntimeProfile): number | undefined {
  const limit = profile.limits.maxTextureDimension2D;
  if (typeof limit !== 'number' || !Number.isFinite(limit) || limit <= 0) return undefined;
  return Math.floor(limit);
}

function validatePtWebgpuRuntimeUnavailable(
  compatibilityMode: GltfCompatibilityMode,
): void {
  if (compatibilityMode === 'best-effort') return;
  throw new GltfCompatibilityError({
    code: 'GLTF_COMPATIBILITY_REJECTED',
    message:
      `[vitrum/engine/gltf] Selected backend "pt-webgpu" resolves to "none" trace tier, ` +
      `which does not satisfy ${compatibilityMode}. ` +
      `Use compatibilityMode:"best-effort", select "pt-webgl2", or run on a WebGPU adapter that satisfies pt-webgpu-lite.`,
    backend: 'pt-webgpu',
    profileId: 'pt-webgpu-lite',
    runtimeProfile: 'pt-webgpu-lite',
    compatibilityMode,
    label: 'Selected backend',
    failures: ['runtime:pt-webgpu=unsupported at adapterProfile.ptWebgpuTier'],
  });
}

function validatePtWebgpuRuntimeProfile(
  profileId: GltfBackendProfileId,
  traceTier: string,
  compatibilityMode: GltfCompatibilityMode,
  asset: GltfAssetResult,
  options: LoadGltfWithEngineOptions,
): void {
  if (compatibilityMode === 'best-effort') return;
  const selected = asset.backendCompatibility.find((entry) => entry.profileId === profileId);
  if (selected != null) {
    const effectiveIssues = selected.issues.filter((issue) =>
      !isSatisfiedRuntimeCompatibilityIssue(issue, options, asset)
    );
    const rejectedIssues = rejectedIssuesForMode(effectiveIssues, compatibilityMode);
    if (rejectedIssues.length === 0) return;
    const failures = rejectedIssues.map(formatRuntimeCompatibilityIssue);

    throw new GltfCompatibilityError({
      code: 'GLTF_COMPATIBILITY_REJECTED',
      message:
        `[vitrum/engine/gltf] Selected backend "pt-webgpu" resolves to ` +
        `"${traceTier}" trace tier, which does not satisfy ` +
        `${compatibilityMode}: ${failures.join(', ')}. ` +
        `Use compatibilityMode:"best-effort", select "pt-webgl2", or run on a full-tier WebGPU adapter.`,
      backend: 'pt-webgpu',
      profileId,
      runtimeProfile: profileId,
      compatibilityMode,
      label: 'Selected backend',
      failures,
    });
  }

  throw new GltfCompatibilityError({
    code: 'GLTF_COMPATIBILITY_PROFILE_MISSING',
    message:
      `[vitrum/engine/gltf] Selected backend "pt-webgpu" resolves to ` +
      `"${traceTier}" trace tier, but the glTF asset has no compatibility row ` +
      `for runtime profile "${profileId}". ` +
      `Use compatibilityMode:"best-effort", select "pt-webgl2", or run on a full-tier WebGPU adapter.`,
    backend: 'pt-webgpu',
    profileId,
    runtimeProfile: profileId,
    compatibilityMode,
    label: 'Selected backend',
  });
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

function formatRuntimeCompatibilityIssue(issue: GltfCompatibilityIssue): string {
  return `${issue.category}:${issue.name}=${issue.support} at ${issue.path}`;
}

function backendFromProfileId(profileId: GltfBackendProfileId): CreateEngineBackendId {
  return profileId === 'pt-webgpu-lite' ? 'pt-webgpu' : profileId;
}

function formatBackendProfile(backend: CreateEngineBackendId, profileId: GltfBackendProfileId): string {
  return profileId === backend
    ? `"${backend}"`
    : `"${backend}" profile "${profileId}"`;
}

function disposeEngineAfterRejectedGltfRuntimeProfile(engine: EngineWithBackendId): void {
  try {
    engine.dispose();
  } catch {
    // Strict glTF validation is already failing; dispose is best-effort cleanup.
  }
}

function isSatisfiedRuntimeCompatibilityIssue(
  issue: GltfCompatibilityIssue,
  options: LoadGltfWithEngineOptions,
  asset: GltfAssetResult,
): boolean {
  if (
    issue.category === 'texture' &&
    issue.name.startsWith('texture-readiness:') &&
    issue.support === 'requires-hook'
  ) {
    return opaqueTextureHandlesReadyForBackend(options, 'pt-webgpu');
  }

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
    const paths = decoded.featureReport.materials.issuePaths.specGlossGlossinessAlpha;
    const requiredPaths = paths !== undefined && paths.length > 0 ? paths : [issue.path];
    return requiredPaths.every((path) =>
      !decoded.textureDecodeDiagnostics.some((diagnostic) =>
        diagnostic.code === 'spec-gloss-alpha-bake-unavailable' &&
        diagnostic.path === path
      ) &&
      decoded.textureDecodeReport.entries.some((entry) =>
        entry.materialField === 'roughnessMap' &&
        entry.path === path &&
        entry.handleKind === 'pixel-data' &&
        entry.handleColorSpace === 'linear'
      )
    );
  }

  return false;
}

function opaqueTextureHandlesReadyForBackend(
  options: LoadGltfWithEngineOptions,
  backend: CreateEngineBackendId,
): boolean {
  const policy = options.opaqueTextureHandlesReady;
  return policy === true || (Array.isArray(policy) && policy.includes(backend));
}

function decodedAssetView(
  asset: GltfAssetResult,
): {
  readonly featureReport: GltfAssetResult['featureReport'];
  readonly textureDecodeDiagnostics: readonly DecodeSceneTextureDiagnostic[];
  readonly textureDecodeReport: GltfTextureDecodeReport;
} | null {
  const maybe = asset as GltfAssetResult & {
    readonly textureDecodeDiagnostics?: readonly DecodeSceneTextureDiagnostic[];
  };
  return Array.isArray(maybe.textureDecodeDiagnostics)
    ? {
        featureReport: asset.featureReport,
        textureDecodeDiagnostics: maybe.textureDecodeDiagnostics,
        textureDecodeReport: asset.textureDecodeReport,
      }
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

function backendSelectionForExplicitPrefer(
  prefer: EnginePreference | undefined,
): GltfEngineSelection | undefined {
  switch (prefer) {
    case 'quality':
      return 'pt-webgl2';
    case 'quality-webgpu':
      return 'pt-webgpu';
    case 'realtime':
      return 'walkaround-hybrid';
    case 'auto':
    case undefined:
      return undefined;
  }
}
