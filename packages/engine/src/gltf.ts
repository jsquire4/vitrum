// @vitrum/engine/gltf — one-import glTF loading helpers.
//
// The glTF adapter remains the owner of asset loading, feature reporting,
// compatibility checks, texture diagnostics, and controller construction. This
// subpath only injects @vitrum/engine's createEngine facade for hosts that want
// a single import path.

import {
  GltfCompatibilityError,
  isTextureReadinessIssue,
  loadGltfForEngine,
  releaseGltfResources,
} from '@vitrum/gltf-adapter';
import type {
  GltfAssetInput,
  GltfAssetResult,
  GltfEngineSelection,
  GltfCompatibilityMode,
  GltfCompatibilityIssue,
  GltfBackendProfileId,
  GltfForEngineResult,
  GltfImportDiagnostic,
  GltfTextureDecodePolicyContext,
  GltfSceneController,
  DecodeSceneTextureDiagnostic,
  GltfCompatibilityFailureDetail,
  GltfTextureDecodeReport,
  LoadGltfForEngineOptions,
} from '@vitrum/gltf-adapter';
import { probeAdapterProfile } from './adapterProfile.js';
import { createEngine } from './createEngine.js';
import { createEngineGltfAssetHint } from './gltfAssetHint.js';
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

export {
  DEFAULT_GLTF_IMPORT_RESOURCE_LIMITS,
  GltfCompatibilityError,
  GltfResourceLimitError,
  loadGltfForEngine,
  normalizeGltfImportResourceLimits,
} from '@vitrum/gltf-adapter';
export { releaseGltfResources };
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
  GltfImportResourceLimits,
  GltfImportDiagnostic,
  GltfMaterialTextureField,
  GltfNpotRepeatWrapPolicy,
  GltfResourceLimitErrorInit,
  GltfResourceLimitKind,
  GltfTextureColorSpace,
  GltfTextureDecodeReport,
  GltfTextureDecodeReportEntry,
  GltfTextureHandleKind,
  LoadGltfForEngineOptions,
  NormalizedGltfImportResourceLimits,
} from '@vitrum/gltf-adapter';

export type GltfCreateEngineOptions = Omit<
  CreateEngineOptions,
  'scene' | 'prefer' | 'gltfAsset'
> & {
  readonly prefer?: EnginePreference;
};

export type LoadGltfWithEngineOptions = Omit<
  LoadGltfForEngineOptions<EngineWithBackendId, GltfCreateEngineOptions>,
  'createEngine' | 'engineOptions'
> & {
  readonly engineOptions?: GltfCreateEngineOptions;
};

export type GltfCreateProgressiveEngineOptions = Omit<
  CreateProgressiveEngineOptions,
  'scene' | 'controller'
>;

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
    const engineRuntimeProfile = ptWebgpuRuntimeProfileFromEngine(engine);
    const profileAwareLoadOptions =
      engineRuntimeProfile !== undefined && baseLoadOptions.runtimeProfile === undefined
        ? { ...baseLoadOptions, runtimeProfile: engineRuntimeProfile }
        : baseLoadOptions;
    const runtimeProfile =
      engineRuntimeProfile === undefined
        ? await maybeProbePtWebgpuRuntimeProfile(engine.backendId, profileAwareLoadOptions)
        : null;
    const loadOptions = withRuntimeTextureCap(profileAwareLoadOptions, runtimeProfile);
    const loaded = await loadGltfForEngine<EngineWithBackendId, GltfCreateEngineOptions>(input, {
      ...loadOptions,
      backend: engine.backendId,
      attachScene: false,
      engineOptions: engineOptions ?? ({} as GltfCreateEngineOptions),
    });
    try {
      let runtimeProfileId = await resolvePtWebgpuRuntimeProfile(
        engine.backendId,
        adapterOptions.compatibilityMode ?? 'best-effort',
        loaded.asset,
        loadOptions,
        runtimeProfile,
      );
      if (engineRuntimeProfile !== undefined && engineRuntimeProfile !== runtimeProfileId) {
        validatePtWebgpuRuntimeProfile(
          engineRuntimeProfile,
          traceTierForPtWebgpuProfile(engineRuntimeProfile),
          adapterOptions.compatibilityMode ?? 'best-effort',
          loaded.asset,
          loadOptions,
        );
        runtimeProfileId = engineRuntimeProfile;
      }
      loaded.controller.attachEngine(engine, { setScene: attachScene ?? true });
      return {
        ...loaded,
        backend: engine.backendId,
        profileId: runtimeProfileId ?? loaded.profileId,
        engine,
        attached: true,
        warnings: [
          ...loaded.asset.warnings,
          ...loaded.textureDecodeWarnings,
          ...loaded.controller.warnings,
        ],
      };
    } catch (error) {
      releaseGltfResources(loaded);
      throw error;
    }
  }

  let runtimeProfileId: GltfBackendProfileId | undefined;
  let runtimeProfile = await maybeProbePtWebgpuRuntimeProfile(
    preferredAdapterBackend,
    adapterOptions,
  );
  const loadOptions = withRecommendedRuntimeTextureCap(
    withRuntimeTextureCap(adapterOptions, runtimeProfile),
    preferredAdapterBackend,
    async (backend, context) => {
      runtimeProfile = await maybeProbePtWebgpuRuntimeProfile(backend, {
        ...adapterOptions,
        ...context.decodeOptions,
      });
      return runtimeProfile;
    },
  );
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
        gltfAsset: createEngineGltfAssetHint(asset),
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
  let engine: ProgressiveEngineHandle | undefined;
  try {
    engine = await createProgressiveEngine({
      ...engineOptions,
      scene: loaded.asset.scene,
      controller: loaded.controller,
    });
    const profiledEngine = Object.defineProperty(engine, 'profileId', {
      value: loaded.profileId,
      writable: false,
      enumerable: true,
      configurable: false,
    });
    loaded.controller.attachEngine(profiledEngine.coordinator, { setScene: false });

    return {
      asset: loaded.asset,
      backend: 'pt-webgpu',
      profileId: loaded.profileId,
      engine: profiledEngine,
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
  } catch (error) {
    disposeProgressiveEngineAfterRejectedGltfLoad(engine);
    releaseGltfResources(loaded);
    throw error;
  }
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

  const profile = runtimeProfile ?? (await probeAdapterProfile());
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
  return maxTextureSize === undefined ? options : { ...options, maxTextureSize };
}

function withRecommendedRuntimeTextureCap<TOptions extends LoadGltfWithEngineOptions>(
  options: TOptions,
  selectedBackend: CreateEngineBackendId | GltfEngineSelection | undefined,
  resolveRuntimeProfile: (
    backend: CreateEngineBackendId | GltfEngineSelection,
    context: GltfTextureDecodePolicyContext,
  ) => Promise<PtWebgpuRuntimeProfile | null>,
): TOptions {
  if (selectedBackend !== undefined) return options;
  return {
    ...options,
    configureTextureDecode: async (context) => {
      const hostPatch = await options.configureTextureDecode?.(context);
      validateConfiguredTextureDecodeOptions(hostPatch);
      const decodeOptions =
        hostPatch === undefined
          ? context.decodeOptions
          : { ...context.decodeOptions, ...hostPatch };
      const recommendedBackend = context.asset.recommendedBackend.backend;
      const runtimeProfile =
        recommendedBackend === 'pt-webgpu'
          ? await resolveRuntimeProfile(recommendedBackend, {
              ...context,
              decodeOptions,
            })
          : null;
      const capPatch = runtimeTextureCapPatch(decodeOptions, runtimeProfile);
      return hostPatch === undefined && capPatch === undefined
        ? undefined
        : { ...(hostPatch ?? {}), ...(capPatch ?? {}) };
    },
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
      '[vitrum/engine/gltf] configureTextureDecode must return an options object or undefined.',
    );
  }
}

function runtimeTextureCapPatch(
  options: Pick<LoadGltfWithEngineOptions, 'maxTextureSize'>,
  runtimeProfile: PtWebgpuRuntimeProfile | null,
): Pick<LoadGltfWithEngineOptions, 'maxTextureSize'> | undefined {
  if (options.maxTextureSize !== undefined || runtimeProfile == null) return undefined;
  const maxTextureSize = runtimeMaxTextureSize(runtimeProfile);
  return maxTextureSize === undefined ? undefined : { maxTextureSize };
}

function runtimeMaxTextureSize(profile: PtWebgpuRuntimeProfile): number | undefined {
  const limit = profile.limits.maxTextureDimension2D;
  if (typeof limit !== 'number' || !Number.isFinite(limit) || limit <= 0) return undefined;
  return Math.floor(limit);
}

function validatePtWebgpuRuntimeUnavailable(compatibilityMode: GltfCompatibilityMode): void {
  if (compatibilityMode === 'best-effort') return;
  const failure = runtimeCompatibilityFailure(
    'pt-webgpu',
    'unsupported',
    'adapterProfile.ptWebgpuTier',
    'Selected pt-webgpu runtime profile resolved to no supported trace tier.',
  );
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
    failures: [failure.message],
    failureDetails: [failure.detail],
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
    const effectiveIssues = selected.issues.filter(
      (issue) => !isSatisfiedRuntimeCompatibilityIssue(issue, options, asset),
    );
    const rejectedIssues = rejectedIssuesForMode(effectiveIssues, compatibilityMode);
    if (rejectedIssues.length === 0) return;
    const failures = rejectedIssues.map(compatibilityIssueFailure);

    throw new GltfCompatibilityError({
      code: 'GLTF_COMPATIBILITY_REJECTED',
      message:
        `[vitrum/engine/gltf] Selected backend "pt-webgpu" resolves to ` +
        `"${traceTier}" trace tier, which does not satisfy ` +
        `${compatibilityMode}: ${failures.map((failure) => failure.message).join(', ')}. ` +
        `Use compatibilityMode:"best-effort", select "pt-webgl2", or run on a full-tier WebGPU adapter.`,
      backend: 'pt-webgpu',
      profileId,
      runtimeProfile: profileId,
      compatibilityMode,
      label: 'Selected backend',
      failures: failures.map((failure) => failure.message),
      failureDetails: failures.map((failure) => failure.detail),
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

interface RuntimeCompatibilityFailure {
  readonly message: string;
  readonly detail: GltfCompatibilityFailureDetail;
}

function compatibilityIssueFailure(issue: GltfCompatibilityIssue): RuntimeCompatibilityFailure {
  return {
    message: `${issue.category}:${issue.name}=${issue.support} at ${issue.path}`,
    detail: {
      source: 'compatibility-issue',
      category: issue.category,
      name: issue.name,
      support: issue.support,
      path: issue.path,
      message: issue.message,
    },
  };
}

function runtimeCompatibilityFailure(
  name: string,
  support: string,
  path: string,
  message: string,
): RuntimeCompatibilityFailure {
  return {
    message: `runtime:${name}=${support} at ${path}`,
    detail: {
      source: 'compatibility-issue',
      category: 'runtime',
      name,
      support,
      path,
      message,
    },
  };
}

function backendFromProfileId(profileId: GltfBackendProfileId): CreateEngineBackendId {
  return profileId === 'pt-webgpu-lite' ? 'pt-webgpu' : profileId;
}

function formatBackendProfile(
  backend: CreateEngineBackendId,
  profileId: GltfBackendProfileId,
): string {
  return profileId === backend ? `"${backend}"` : `"${backend}" profile "${profileId}"`;
}

function disposeEngineAfterRejectedGltfRuntimeProfile(engine: EngineWithBackendId): void {
  try {
    engine.dispose();
  } catch {
    // Strict glTF validation is already failing; dispose is best-effort cleanup.
  }
}

function disposeProgressiveEngineAfterRejectedGltfLoad(
  engine: ProgressiveEngineHandle | undefined,
): void {
  try {
    engine?.dispose();
  } catch {
    // The glTF wrapper is already rejecting; progressive cleanup is best effort.
  }
}

function isSatisfiedRuntimeCompatibilityIssue(
  issue: GltfCompatibilityIssue,
  options: LoadGltfWithEngineOptions,
  asset: GltfAssetResult,
): boolean {
  if (isTextureReadinessIssue(issue) && issue.support === 'requires-hook') {
    return opaqueTextureHandlesReadyForBackend(options, 'pt-webgpu');
  }

  if (issue.support === 'requires-hook') {
    if (issue.name === 'KHR_draco_mesh_compression')
      return typeof options.dracoDecode === 'function';
    if (issue.name === 'EXT_meshopt_compression' || issue.name === 'KHR_meshopt_compression') {
      return typeof options.meshoptDecode === 'function';
    }
    if (
      issue.name === 'KHR_texture_basisu' ||
      issue.name === 'EXT_texture_webp' ||
      issue.name === 'MSFT_texture_dds'
    ) {
      return (
        (options.textureSourceExtensions ?? []).includes(issue.name) &&
        typeof options.decodeImage === 'function'
      );
    }
    return false;
  }

  if (
    issue.name === 'KHR_materials_pbrSpecularGlossiness.specularGlossinessTexture.glossinessAlpha'
  ) {
    const decoded = decodedAssetView(asset);
    if (decoded == null) return false;
    const paths = decoded.featureReport.materials.issuePaths.specGlossGlossinessAlpha;
    const requiredPaths = paths !== undefined && paths.length > 0 ? paths : [issue.path];
    return requiredPaths.every(
      (path) =>
        !decoded.textureDecodeDiagnostics.some(
          (diagnostic) =>
            diagnostic.code === 'spec-gloss-alpha-bake-unavailable' && diagnostic.path === path,
        ) &&
        decoded.textureDecodeReport.entries.some(
          (entry) =>
            entry.materialField === 'roughnessMap' &&
            entry.path === path &&
            entry.handleKind === 'pixel-data' &&
            entry.handleColorSpace === 'linear',
        ),
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

function decodedAssetView(asset: GltfAssetResult): {
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

function ptWebgpuRuntimeProfileFromEngine(
  engine: EngineWithBackendId,
): GltfBackendProfileId | undefined {
  if (engine.backendId !== 'pt-webgpu') return undefined;
  const profileId = engine.backendProfileId ?? engine.profileId;
  if (profileId === 'pt-webgpu' || profileId === 'pt-webgpu-lite') return profileId;
  return undefined;
}

function traceTierForPtWebgpuProfile(profileId: GltfBackendProfileId): string {
  return profileId === 'pt-webgpu-lite' ? 'lite' : 'full';
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
