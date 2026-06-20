// featureReport.ts — structured glTF asset inventory + backend compatibility.
//
// This is intentionally a pure JSON walk. It does not decode buffers, fetch
// resources, or mutate the asset. The goal is to classify what an asset asks
// for before a backend is selected, so hosts can choose strict/fidelity/realtime
// policy with data instead of guessing from triangle count alone.

import {
  BACKEND_PROMISE_LEDGER,
  type BackendId,
  type BackendSupportMode,
  type MaterialSpec,
} from '@vitrum/core';
import { typeComponentCount } from './accessors.js';
import type {
  GltfAnimationChannel,
  GltfAnimationSampler,
  GltfImage,
  GltfJson,
  GltfMaterial,
  GltfPrimitive,
  GltfTextureInfo,
} from './gltfTypes.js';
import {
  collectGltfSceneReachability,
  collectPrimitiveMaterialIndices,
  gltfPrimitiveKey,
  type GltfSceneReachability,
} from './sceneScope.js';

export type GltfResourceKind = 'embedded' | 'bufferView' | 'data-uri' | 'external-uri' | 'missing';

export interface GltfResourceUse {
  readonly index: number;
  readonly kind: GltfResourceKind;
  readonly uri?: string;
  readonly mimeType?: string;
}

export interface GltfExtensionReport {
  readonly used: readonly string[];
  readonly required: readonly string[];
  readonly supported: readonly string[];
  readonly requiresHook: readonly string[];
  readonly unsupportedOptional: readonly string[];
  readonly unsupportedRequired: readonly string[];
  readonly textureSourceUses: readonly GltfTextureSourceExtensionUse[];
  readonly sourcePaths: Readonly<Record<string, readonly string[]>>;
}

export type GltfTextureSourceExtensionName =
  | 'KHR_texture_basisu'
  | 'EXT_texture_webp'
  | 'MSFT_texture_dds';

export interface GltfTextureSourceExtensionUse {
  readonly extension: GltfTextureSourceExtensionName;
  readonly textureIndex: number;
  readonly sourceImageIndex: number;
  readonly path: string;
  readonly selected: boolean;
  readonly required: boolean;
  readonly hasBaseSource: boolean;
  readonly requiresHook: boolean;
  readonly mimeType?: string;
}

export interface GltfPrimitiveFeatureReport {
  readonly total: number;
  readonly byMode: Readonly<Record<string, number>>;
  readonly unsupportedModes: readonly string[];
  readonly fallbackGeneratedModes: readonly string[];
  readonly attributeSemantics: readonly string[];
  readonly expectedPrimitiveKinds: readonly ('mesh' | 'skinned-mesh' | 'instanced-mesh')[];
  readonly usesDraco: boolean;
  readonly usesMeshopt: boolean;
  readonly hasTangents: boolean;
  readonly hasMorphTargets: boolean;
  readonly hasMorphTargetTangents: boolean;
  readonly hasMorphTargetTexcoords: boolean;
  readonly hasUnsupportedMorphTargetTexcoords: boolean;
  readonly hasSkins: boolean;
  readonly hasInstancedSkinnedOrMorphed: boolean;
  readonly hasIgnoredSkinAttributes: boolean;
  readonly hasIncompleteSkinAttributes: boolean;
  readonly malformedPrimitives: readonly GltfMalformedPrimitiveIssue[];
  readonly accessorStorageIssues: readonly GltfPrimitiveAccessorStorageIssue[];
  readonly accessorImportIssues: readonly GltfPrimitiveAccessorImportIssue[];
  readonly instancingIssues: readonly GltfPrimitiveInstancingIssue[];
  readonly hasVertexColors: boolean;
  readonly ignoredVertexColorSets: readonly string[];
  readonly hasUv1: boolean;
  readonly issuePaths: Readonly<Record<string, readonly string[]>>;
}

export type GltfMalformedPrimitiveKind =
  | 'missing-position'
  | 'missing-position-accessor'
  | 'invalid-position-accessor-type'
  | 'invalid-position-accessor-component-type'
  | 'missing-position-buffer-view'
  | 'missing-position-buffer'
  | 'invalid-position-sparse-accessor'
  | 'missing-index-accessor'
  | 'invalid-index-accessor'
  | 'missing-index-buffer-view'
  | 'missing-index-buffer'
  | 'invalid-index-sparse-accessor'
  | 'empty-triangulated-primitive';

export type GltfSparseAccessorStorageIssueKind =
  | 'missing-sparse-indices-buffer-view'
  | 'missing-sparse-indices-buffer'
  | 'missing-sparse-values-buffer-view'
  | 'missing-sparse-values-buffer'
  | 'invalid-sparse-indices-component-type';

export interface GltfMalformedPrimitiveIssue {
  readonly kind: GltfMalformedPrimitiveKind;
  readonly path: string;
  readonly meshIndex: number;
  readonly primitiveIndex: number;
  readonly accessorIndex?: number;
  readonly bufferViewIndex?: number;
  readonly bufferIndex?: number;
  readonly accessorType?: string;
  readonly componentType?: number;
  readonly sparseIssueKind?: GltfSparseAccessorStorageIssueKind;
  readonly mode?: number;
}

export interface GltfPrimitiveAccessorStorageIssue {
  readonly path: string;
  readonly semantic: string;
  readonly accessorIndex: number;
  readonly sparseIssueKind: GltfSparseAccessorStorageIssueKind;
  readonly meshIndex?: number;
  readonly primitiveIndex?: number;
  readonly targetIndex?: number;
  readonly nodeIndex?: number;
  readonly skinIndex?: number;
  readonly bufferViewIndex?: number;
  readonly bufferIndex?: number;
  readonly componentType?: number;
}

export type GltfPrimitiveAccessorImportIssueKind =
  | 'missing-accessor'
  | 'invalid-accessor-type'
  | 'invalid-accessor-count'
  | 'invalid-accessor-component-type'
  | 'missing-buffer-view'
  | 'missing-buffer';

export interface GltfPrimitiveAccessorImportIssue {
  readonly kind: GltfPrimitiveAccessorImportIssueKind;
  readonly support: 'approximate' | 'unsupported';
  readonly path: string;
  readonly semantic: string;
  readonly accessorIndex: number;
  readonly meshIndex?: number;
  readonly primitiveIndex?: number;
  readonly targetIndex?: number;
  readonly nodeIndex?: number;
  readonly skinIndex?: number;
  readonly expectedTypes?: readonly string[];
  readonly accessorType?: string;
  readonly expectedCount?: number;
  readonly actualCount?: number;
  readonly bufferViewIndex?: number;
  readonly bufferIndex?: number;
  readonly componentType?: number;
}

export type GltfPrimitiveInstancingIssueKind =
  | 'missing-attributes'
  | 'missing-transform-attributes'
  | 'invalid-attribute-accessor-index';

export interface GltfPrimitiveInstancingIssue {
  readonly kind: GltfPrimitiveInstancingIssueKind;
  readonly path: string;
  readonly nodeIndex: number;
  readonly attribute?: string;
  readonly value?: unknown;
}

export interface GltfMaterialFeatureReport {
  readonly count: number;
  readonly materialFields: readonly (keyof MaterialSpec)[];
  readonly textureFields: readonly (keyof MaterialSpec)[];
  readonly samplerPolicies: readonly GltfTextureSamplerPolicyUse[];
  readonly malformedSamplerPolicies: readonly GltfMalformedTextureSamplerPolicyUse[];
  readonly textureReferenceIssues: readonly GltfMaterialTextureReferenceIssue[];
  readonly primitiveMaterialReferenceIssues: readonly GltfPrimitiveMaterialReferenceIssue[];
  readonly variantMappingIssues: readonly GltfMaterialVariantMappingIssue[];
  readonly extensions: readonly string[];
  readonly unsupportedKnownExtensions: readonly string[];
  readonly alphaModes: readonly string[];
  readonly uvSets: readonly number[];
  readonly unrepresentableUvSets: readonly number[];
  readonly textureTransformCount: number;
  readonly volumeThicknessTextureCount: number;
  readonly specularGlossinessMaterialCount: number;
  readonly specularGlossinessTextureCount: number;
  readonly doubleSidedCount: number;
  readonly issuePaths: Readonly<Record<string, readonly string[]>>;
}

export type GltfTextureSamplerFilterMode = 'nearest' | 'linear';
export type GltfTextureSamplerMipMode = 'none' | 'nearest' | 'linear';

export interface GltfTextureSamplerPolicyUse {
  readonly materialField: keyof MaterialSpec;
  readonly textureIndex: number;
  readonly samplerIndex: number;
  readonly materialPath: string;
  readonly path: string;
  readonly magFilter?: GltfTextureSamplerFilterMode;
  readonly minFilter?: GltfTextureSamplerFilterMode;
  readonly mipFilter?: GltfTextureSamplerMipMode;
  readonly usesMipmaps?: boolean;
}

export type GltfMalformedTextureSamplerKind =
  | 'missing-sampler'
  | 'invalid-wrap-s'
  | 'invalid-wrap-t'
  | 'invalid-mag-filter'
  | 'invalid-min-filter';

export interface GltfMalformedTextureSamplerPolicyUse {
  readonly kind: GltfMalformedTextureSamplerKind;
  readonly materialField: keyof MaterialSpec;
  readonly textureIndex: number;
  readonly samplerIndex: number;
  readonly materialPath: string;
  readonly path: string;
  readonly value?: number;
}

export type GltfMaterialTextureReferenceIssueKind =
  | 'missing-texture'
  | 'disabled-texture-source-extension'
  | 'missing-texture-source'
  | 'missing-image'
  | 'missing-image-source'
  | 'missing-image-buffer-view'
  | 'image-buffer-unavailable';

export interface GltfMaterialTextureReferenceIssue {
  readonly kind: GltfMaterialTextureReferenceIssueKind;
  readonly materialField: keyof MaterialSpec;
  readonly textureIndex: number;
  readonly materialPath: string;
  readonly path: string;
  readonly imageIndex?: number;
  readonly bufferViewIndex?: number;
  readonly bufferIndex?: number;
  readonly textureSourceExtensions?: readonly GltfTextureSourceExtensionName[];
}

export interface GltfPrimitiveMaterialReferenceIssue {
  readonly kind: 'missing-material';
  readonly path: string;
  readonly meshIndex: number;
  readonly primitiveIndex: number;
  readonly materialIndex: number;
}

export type GltfMaterialVariantMappingIssueKind =
  | 'malformed-root-variant-list'
  | 'missing-material'
  | 'missing-variant-list'
  | 'missing-variant';

export interface GltfMaterialVariantMappingIssue {
  readonly kind: GltfMaterialVariantMappingIssueKind;
  readonly path: string;
  readonly meshIndex?: number;
  readonly primitiveIndex?: number;
  readonly mappingIndex?: number;
  readonly materialIndex?: number;
  readonly variantIndex?: number;
}

export interface GltfAnimationFeatureReport {
  readonly count: number;
  readonly channelCount: number;
  readonly paths: readonly string[];
  readonly unsupportedTargetPaths: readonly string[];
  readonly interpolations: readonly string[];
  readonly degradedInterpolations: readonly string[];
  readonly malformedChannels: readonly GltfAnimationMalformedChannelIssue[];
  readonly targetNodeCount: number;
  readonly issuePaths: Readonly<Record<string, readonly string[]>>;
}

export type GltfAnimationMalformedChannelKind =
  | 'missing-sampler'
  | 'missing-input-accessor'
  | 'missing-output-accessor'
  | 'invalid-input-accessor-type'
  | 'invalid-output-accessor-type'
  | 'invalid-input-accessor-component-type'
  | 'invalid-output-accessor-component-type'
  | 'missing-input-buffer-view'
  | 'missing-output-buffer-view'
  | 'missing-input-buffer'
  | 'missing-output-buffer'
  | 'invalid-input-sparse-accessor'
  | 'invalid-output-sparse-accessor'
  | 'missing-target-node'
  | 'target-node-not-found'
  | 'invalid-output-count';

export interface GltfAnimationMalformedChannelIssue {
  readonly kind: GltfAnimationMalformedChannelKind;
  readonly path: string;
  readonly animationIndex: number;
  readonly channelIndex: number;
  readonly targetPath?: string;
  readonly samplerIndex?: number;
  readonly nodeIndex?: number;
  readonly accessorIndex?: number;
  readonly accessorRole?: 'input' | 'output';
  readonly accessorType?: string;
  readonly componentType?: number;
  readonly bufferViewIndex?: number;
  readonly bufferIndex?: number;
  readonly sparseIssueKind?: GltfSparseAccessorStorageIssueKind;
  readonly expectedOutputFloats?: number;
  readonly actualOutputFloats?: number;
}

export interface GltfSceneGraphFeatureReport {
  readonly scenes: number;
  readonly nodes: number;
  readonly cameras: number;
  readonly cameraPaths: readonly string[];
  readonly punctualLights: number;
  readonly punctualLightIssues: readonly GltfPunctualLightIssue[];
}

export type GltfPunctualLightIssueKind =
  | 'missing-light'
  | 'unsupported-light-type';

export interface GltfPunctualLightIssue {
  readonly kind: GltfPunctualLightIssueKind;
  readonly path: string;
  readonly nodeIndex?: number;
  readonly lightIndex?: number;
  readonly lightType?: string;
}

export interface GltfResourceFeatureReport {
  readonly buffers: readonly GltfResourceUse[];
  readonly images: readonly GltfResourceUse[];
  readonly textureCount: number;
  readonly externalBufferCount: number;
  readonly externalImageCount: number;
}

export interface GltfFeatureReport {
  readonly assetVersion?: string;
  readonly generator?: string;
  readonly extensions: GltfExtensionReport;
  readonly resources: GltfResourceFeatureReport;
  readonly primitives: GltfPrimitiveFeatureReport;
  readonly materials: GltfMaterialFeatureReport;
  readonly animations: GltfAnimationFeatureReport;
  readonly sceneGraph: GltfSceneGraphFeatureReport;
}

export interface GltfCompatibilityIssue {
  readonly category: 'extension' | 'primitive' | 'material' | 'scene' | 'texture' | 'animation';
  readonly name: string;
  readonly support: BackendSupportMode | 'requires-hook' | 'unknown';
  readonly path: string;
  readonly message: string;
}

export interface GltfBackendCompatibility {
  readonly backend: BackendId;
  /**
   * Concrete planner profile. Most rows are one-to-one with `backend`; pt-webgpu
   * has both a full profile and a constrained lite profile selected at engine
   * creation from adapter limits.
   */
  readonly profileId: GltfBackendProfileId;
  readonly traceTier?: GltfBackendTraceTier;
  readonly unsupportedCount: number;
  readonly approximateCount: number;
  readonly nativeCount: number;
  readonly requiresHookCount: number;
  readonly issues: readonly GltfCompatibilityIssue[];
  readonly isCompatible: boolean;
}

export type GltfBackendProfileId = BackendId | 'pt-webgpu-lite';
export type GltfBackendTraceTier = 'full' | 'lite';
export type GltfBackendPolicy = 'fidelity' | 'realtime' | 'strict' | 'best-effort';

export interface AnalyzeGltfAssetOptions {
  readonly textureSourceExtensions?: readonly GltfTextureSourceExtensionName[];
  /**
   * When supplied, compatibility-affecting primitive/material/scene rows are
   * scoped to the graph reachable from this scene. Omit to keep the historical
   * whole-asset inventory behavior.
   */
  readonly sceneIndex?: number;
}

const REQUIRED_EXTENSION_SUPPORT = new Set([
  'KHR_draco_mesh_compression',
  'EXT_meshopt_compression',
  'KHR_meshopt_compression',
  'KHR_lights_punctual',
  'KHR_materials_unlit',
  'KHR_materials_transmission',
  'KHR_materials_ior',
  'KHR_materials_volume',
  'KHR_materials_specular',
  'KHR_materials_sheen',
  'KHR_materials_clearcoat',
  'KHR_materials_iridescence',
  'KHR_materials_anisotropy',
  'KHR_materials_dispersion',
  'KHR_materials_emissive_strength',
  'KHR_materials_variants',
  'KHR_materials_pbrSpecularGlossiness',
  'KHR_mesh_quantization',
  'KHR_texture_transform',
  'KHR_texture_basisu',
  'EXT_texture_webp',
  'MSFT_texture_dds',
  'EXT_mesh_gpu_instancing',
]);

const EXTENSIONS_REQUIRING_HOST_HOOK = new Set([
  'KHR_draco_mesh_compression',
  'EXT_meshopt_compression',
  'KHR_meshopt_compression',
  'KHR_texture_basisu',
  'EXT_texture_webp',
  'MSFT_texture_dds',
]);

const TEXTURE_SOURCE_EXTENSION_NAMES = [
  'KHR_texture_basisu',
  'EXT_texture_webp',
  'MSFT_texture_dds',
] as const satisfies readonly GltfTextureSourceExtensionName[];

const TEXTURE_SOURCE_EXTENSIONS = new Set<string>(TEXTURE_SOURCE_EXTENSION_NAMES);
const MESHOPT_COMPRESSION_EXTENSIONS = new Set(['EXT_meshopt_compression', 'KHR_meshopt_compression']);

const COMMON_UNSUPPORTED_EXTENSIONS = new Set<string>();

const FALLBACK_GENERATED_PRIMITIVE_MODES = new Set([0, 1, 2, 3]);
const SUPPORTED_GLTF_PRIMITIVE_MODES = new Set([0, 1, 2, 3, 4, 5, 6]);
const VERTEX_COLOR_SUPPORT: Readonly<Record<GltfBackendProfileId, BackendSupportMode>> = Object.freeze({
  'pt-webgl2': 'native',
  'pt-webgpu': 'native',
  'pt-webgpu-lite': 'unsupported',
  'walkaround-hybrid': 'approximate',
});

const PT_WEBGPU_LITE_UNSUPPORTED_MATERIAL_FIELDS = [
  // Lite composes no full-tier group-3 material texture bindings; every
  // texture-backed MaterialSpec field is therefore unsupported on that profile.
  'baseColorMap',
  'normalMap',
  'normalScale',
  'roughnessMap',
  'metallicMap',
  'transmissionMap',
  'emissiveMap',
  'alphaMap',
  'aoMap',
  'aoMapIntensity',
  'clearcoatMap',
  'clearcoatRoughnessMap',
  'clearcoatNormalMap',
  'clearcoatNormalScale',
  'sheenColorMap',
  'sheenRoughnessMap',
  'iridescenceMap',
  'iridescenceThicknessMap',
  'anisotropyMap',
  'specularColorMap',
  'specularIntensityMap',
  'bumpMap',
  'bumpScale',
  'lightMap',
  'lightMapIntensity',
  // Lite also omits the full-tier alpha-test and per-material env/aniso paths.
  'alphaMode',
  'alphaCutoff',
  'opacity',
  'envMapIntensity',
  'anisotropy',
  'anisotropyRotation',
] as const satisfies readonly (keyof MaterialSpec)[];

interface GltfBackendProfile {
  readonly id: GltfBackendProfileId;
  readonly backend: BackendId;
  readonly traceTier?: GltfBackendTraceTier;
  readonly materialOverrides?: Readonly<Partial<Record<keyof MaterialSpec, BackendSupportMode>>>;
}

const PT_WEBGPU_LITE_MATERIAL_OVERRIDES: Readonly<Partial<Record<keyof MaterialSpec, BackendSupportMode>>> =
  Object.freeze(Object.fromEntries(
    PT_WEBGPU_LITE_UNSUPPORTED_MATERIAL_FIELDS.map((field) => [field, 'unsupported' as const]),
  ));

const BACKEND_PROFILES: Readonly<Record<GltfBackendProfileId, GltfBackendProfile>> = Object.freeze({
  'pt-webgl2': Object.freeze({ id: 'pt-webgl2', backend: 'pt-webgl2' }),
  'pt-webgpu': Object.freeze({
    id: 'pt-webgpu',
    backend: 'pt-webgpu',
    traceTier: 'full',
  }),
  'pt-webgpu-lite': Object.freeze({
    id: 'pt-webgpu-lite',
    backend: 'pt-webgpu',
    traceTier: 'lite',
    materialOverrides: PT_WEBGPU_LITE_MATERIAL_OVERRIDES,
  }),
  'walkaround-hybrid': Object.freeze({
    id: 'walkaround-hybrid',
    backend: 'walkaround-hybrid',
  }),
});

const CORE_ANIMATION_TARGET_PATHS: ReadonlySet<string> = new Set([
  'translation',
  'rotation',
  'scale',
  'weights',
]);
const CORE_ANIMATION_INTERPOLATIONS: ReadonlySet<string> = new Set([
  'LINEAR',
  'STEP',
  'CUBICSPLINE',
]);

type SourcePathMap = Map<string, string[]>;

function addSourcePath(paths: SourcePathMap, key: string, path: string): void {
  const current = paths.get(key);
  if (current !== undefined) {
    if (!current.includes(path)) current.push(path);
    return;
  }
  paths.set(key, [path]);
}

function sourcePathRecord(paths: SourcePathMap): Readonly<Record<string, readonly string[]>> {
  return Object.fromEntries(
    [...paths.entries()]
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([key, values]) => [key, values.slice().sort()]),
  );
}

function firstSourcePath(
  paths: Readonly<Record<string, readonly string[]>>,
  key: string,
  fallback: string,
): string {
  return paths[key]?.[0] ?? fallback;
}

function requiresHookIssuePath(report: GltfFeatureReport, ext: string): string {
  const fallback = firstSourcePath(report.extensions.sourcePaths, ext, 'extensionsUsed');
  const textureSourcePath = TEXTURE_SOURCE_EXTENSIONS.has(ext)
    ? report.extensions.sourcePaths[ext]?.find((path) =>
      path.startsWith('textures[') && path.endsWith(`extensions.${ext}`),
    )
    : undefined;
  if (textureSourcePath !== undefined) return textureSourcePath;
  if (report.extensions.required.includes(ext) || !TEXTURE_SOURCE_EXTENSIONS.has(ext)) return fallback;
  return fallback;
}

export function analyzeGltfAsset(
  gltf: GltfJson,
  options: AnalyzeGltfAssetOptions = {},
): GltfFeatureReport {
  const selectedTextureSourceExtensions = new Set<string>(options.textureSourceExtensions ?? []);
  const sceneScope = options.sceneIndex === undefined
    ? undefined
    : collectGltfSceneReachability(gltf, options.sceneIndex);
  const extensions = analyzeExtensions(gltf, selectedTextureSourceExtensions, sceneScope);
  const resources = analyzeResources(gltf);
  const primitives = analyzePrimitives(gltf, sceneScope);
  const materials = analyzeMaterials(gltf, sceneScope, selectedTextureSourceExtensions);
  const animations = analyzeAnimations(gltf, sceneScope);
  const punctualLights = sceneScope?.punctualLightIndices.size ?? extractPunctualLightCount(gltf);
  const punctualLightIssues = analyzePunctualLightIssues(gltf, sceneScope);
  const cameraPaths = sceneScope === undefined
    ? (gltf.cameras ?? []).map((_, index) => `cameras[${index}]`)
    : [...sceneScope.cameraIndices].sort((a, b) => a - b).map((index) => `cameras[${index}]`);

  return {
    ...(gltf.asset?.version !== undefined ? { assetVersion: gltf.asset.version } : {}),
    ...(gltf.asset?.generator !== undefined ? { generator: gltf.asset.generator } : {}),
    extensions,
    resources,
    primitives,
    materials,
    animations,
    sceneGraph: {
      scenes: gltf.scenes?.length ?? 0,
      nodes: sceneScope?.nodeIndices.size ?? gltf.nodes?.length ?? 0,
      cameras: sceneScope?.cameraIndices.size ?? gltf.cameras?.length ?? 0,
      cameraPaths,
      punctualLights,
      punctualLightIssues,
    },
  };
}

export function evaluateGltfBackendCompatibility(
  report: GltfFeatureReport,
  backend: BackendId,
): GltfBackendCompatibility {
  return evaluateGltfBackendProfileCompatibility(report, backend);
}

export function evaluateGltfBackendProfileCompatibility(
  report: GltfFeatureReport,
  profileId: GltfBackendProfileId,
): GltfBackendCompatibility {
  const profile = BACKEND_PROFILES[profileId];
  const { backend } = profile;
  const ledger = BACKEND_PROMISE_LEDGER[backend];
  const issues: GltfCompatibilityIssue[] = [];
  let unsupportedCount = 0;
  let approximateCount = 0;
  let nativeCount = 0;
  let requiresHookCount = 0;

  const addIssue = (issue: GltfCompatibilityIssue): void => {
    issues.push(issue);
    if (issue.support === 'unsupported') unsupportedCount += 1;
    else if (issue.support === 'approximate' || issue.support === 'fallback-generated-mesh' || issue.support === 'fallback-rebuild') {
      approximateCount += 1;
    } else if (issue.support === 'native') {
      nativeCount += 1;
    } else if (issue.support === 'requires-hook') {
      requiresHookCount += 1;
    }
  };

  for (const ext of report.extensions.unsupportedRequired) {
    addIssue({
      category: 'extension',
      name: ext,
      support: 'unsupported',
      path: firstSourcePath(report.extensions.sourcePaths, ext, 'extensionsRequired'),
      message: `Required glTF extension "${ext}" is not supported by the adapter.`,
    });
  }
  for (const ext of report.extensions.requiresHook) {
    addIssue({
      category: 'extension',
      name: ext,
      support: 'requires-hook',
      path: requiresHookIssuePath(report, ext),
      message: `glTF extension "${ext}" requires host-supplied decode support.`,
    });
  }
  for (const ext of report.extensions.unsupportedOptional) {
    addIssue({
      category: 'extension',
      name: ext,
      support: 'unsupported',
      path: firstSourcePath(report.extensions.sourcePaths, ext, 'extensionsUsed'),
      message: `Optional glTF extension "${ext}" has no Vitrum mapping today.`,
    });
  }

  for (const kind of report.primitives.expectedPrimitiveKinds) {
    const support = ledger.supportDetails.primitives[kind] ?? 'unknown';
    if (support !== 'native') {
      addIssue({
        category: 'primitive',
        name: kind,
        support,
        path: firstSourcePath(report.primitives.issuePaths, `kind:${kind}`, 'meshes'),
        message: `Backend ${backend} reports primitive kind "${kind}" as ${support}.`,
      });
    } else {
      nativeCount += 1;
    }
  }

  for (const mode of report.primitives.unsupportedModes) {
    addIssue({
      category: 'primitive',
      name: `mode:${mode}`,
      support: 'unsupported',
      path: firstSourcePath(report.primitives.issuePaths, `mode:${mode}`, 'meshes'),
      message: `glTF primitive mode ${mode} has no Vitrum adapter representation.`,
    });
  }

  for (const mode of report.primitives.fallbackGeneratedModes) {
    addIssue({
      category: 'primitive',
      name: `mode:${mode}`,
      support: 'fallback-generated-mesh',
      path: firstSourcePath(report.primitives.issuePaths, `mode:${mode}`, 'meshes'),
      message:
        `glTF primitive mode ${mode} is imported as generated triangle mesh fallback geometry ` +
        'because @vitrum/core has no native point/line primitive contract.',
    });
  }

  for (const malformed of report.primitives.malformedPrimitives) {
    addIssue({
      category: 'primitive',
      name: `malformed.${malformed.kind}`,
      support: 'unsupported',
      path: malformed.path,
      message: malformedPrimitiveMessage(malformed),
    });
  }

  for (const storageIssue of report.primitives.accessorStorageIssues) {
    addIssue({
      category: 'primitive',
      name: `accessor.${storageIssue.semantic}.${storageIssue.sparseIssueKind}`,
      support: 'approximate',
      path: storageIssue.path,
      message: primitiveAccessorStorageIssueMessage(storageIssue),
    });
  }

  for (const importIssue of report.primitives.accessorImportIssues) {
    addIssue({
      category: 'primitive',
      name: `accessor.${importIssue.semantic}.${importIssue.kind}`,
      support: importIssue.support,
      path: importIssue.path,
      message: primitiveAccessorImportIssueMessage(importIssue),
    });
  }

  for (const instancingIssue of report.primitives.instancingIssues) {
    addIssue({
      category: 'primitive',
      name: `EXT_mesh_gpu_instancing.${instancingIssue.kind}`,
      support: 'unsupported',
      path: instancingIssue.path,
      message: primitiveInstancingIssueMessage(instancingIssue),
    });
  }

  if (report.primitives.hasMorphTargetTangents) {
    addIssue({
      category: 'primitive',
      name: 'morphTargetTangents',
      support: 'approximate',
      path: firstSourcePath(report.primitives.issuePaths, 'morphTargetTangents', 'meshes'),
      message:
        'glTF morph-target TANGENT deltas are preserved on SkinnedMeshPrimitive.morphTargetTangents, ' +
        'and CPU-solved skinned paths apply them to posed tangent-space shading when rest tangents exist; ' +
        'GPU-native tangent skinning is still a fallback-to-CPU path.',
    });
  }

  if (report.primitives.hasUnsupportedMorphTargetTexcoords) {
    addIssue({
      category: 'primitive',
      name: 'morphTargetTexcoords',
      support: 'unsupported',
      path: firstSourcePath(report.primitives.issuePaths, 'unsupportedMorphTargetTexcoords', 'meshes'),
      message:
        'glTF morph-target UV deltas require the matching base UV stream assigned to core uv/uv1. ' +
        'TEXCOORD_2+ morph deltas are supported only when that high UV set is losslessly remapped ' +
        'into core uv1 for this primitive; unsupported lanes are reported explicitly.',
    });
  }

  if (report.primitives.hasInstancedSkinnedOrMorphed) {
    addIssue({
      category: 'primitive',
      name: 'EXT_mesh_gpu_instancing.skinnedOrMorphed',
      support: 'unsupported',
      path: firstSourcePath(
        report.primitives.issuePaths,
        'instancedSkinnedOrMorphed',
        'nodes',
      ),
      message:
        'glTF EXT_mesh_gpu_instancing on skinned or morphed meshes is not representable in the core Scene contract yet; ' +
        'the importer falls back to one skinned/morphed primitive and ignores the instance transforms.',
    });
  }

  if (report.primitives.hasIgnoredSkinAttributes) {
    addIssue({
      category: 'primitive',
      name: 'skinAttributesWithoutNodeSkin',
      support: 'unsupported',
      path: firstSourcePath(report.primitives.issuePaths, 'ignoredSkinAttributes', 'meshes'),
      message:
        'glTF JOINTS_0/WEIGHTS_0 attributes are present on a mesh instance whose node does not bind a skin; ' +
        'the importer ignores those attributes and imports the primitive as an ordinary mesh.',
    });
  }

  if (report.primitives.hasIncompleteSkinAttributes) {
    addIssue({
      category: 'primitive',
      name: 'incompleteSkinAttributes',
      support: 'unsupported',
      path: firstSourcePath(report.primitives.issuePaths, 'incompleteSkinAttributes', 'meshes'),
      message:
        'glTF skinned primitives must provide both JOINTS_0 and WEIGHTS_0; ' +
        'the importer falls back to a static mesh when either stream is missing.',
    });
  }

  if (report.primitives.hasVertexColors) {
    const support = VERTEX_COLOR_SUPPORT[profileId];
    if (support === 'native') {
      nativeCount += 1;
    } else {
      addIssue({
        category: 'primitive',
        name: 'vertexColors',
        support,
        path: firstSourcePath(report.primitives.issuePaths, 'vertexColors', 'meshes'),
        message: `Backend profile ${profileId} reports glTF COLOR_0 vertex colors as ${support}.`,
      });
    }
  }

  for (const semantic of report.primitives.ignoredVertexColorSets) {
    addIssue({
      category: 'primitive',
      name: semantic,
      support: 'unsupported',
      path: firstSourcePath(report.primitives.issuePaths, `ignoredVertexColorSet:${semantic}`, 'meshes'),
      message:
        `glTF ${semantic} secondary vertex-color sets are not imported; ` +
        'the core Scene contract currently preserves COLOR_0 only.',
    });
  }

  if (report.sceneGraph.cameras > 0) {
    addIssue({
      category: 'scene',
      name: 'cameras',
      support: 'approximate',
      path: report.sceneGraph.cameraPaths[0] ?? 'cameras',
      message:
        'glTF cameras are reported for host inspection but are not imported into the core Scene contract; ' +
        'Vitrum cameras are supplied per frame through FrameInput.',
    });
  }

  for (const punctualIssue of report.sceneGraph.punctualLightIssues) {
    addIssue({
      category: 'scene',
      name: `KHR_lights_punctual.${punctualIssue.kind}`,
      support: 'approximate',
      path: punctualIssue.path,
      message: punctualLightIssueMessage(punctualIssue),
    });
  }

  for (const targetPath of report.animations.unsupportedTargetPaths) {
    addIssue({
      category: 'animation',
      name: `target.path:${targetPath}`,
      support: 'unsupported',
      path: firstSourcePath(
        report.animations.issuePaths,
        `unsupportedTargetPath:${targetPath}`,
        'animations',
      ),
      message:
        `glTF animation target path "${targetPath}" is not imported into the core animation controller; ` +
        'supported target paths are translation, rotation, scale, and weights.',
    });
  }

  for (const interpolation of report.animations.degradedInterpolations) {
    addIssue({
      category: 'animation',
      name: `sampler.interpolation:${interpolation}`,
      support: 'approximate',
      path: firstSourcePath(
        report.animations.issuePaths,
        `degradedInterpolation:${interpolation}`,
        'animations',
      ),
      message:
        `glTF animation sampler interpolation "${interpolation}" is not part of the core glTF interpolation set; ` +
        'the importer falls back to LINEAR for those samplers.',
    });
  }

  for (const malformed of report.animations.malformedChannels) {
    addIssue({
      category: 'animation',
      name: `channel.${malformed.kind}`,
      support: 'unsupported',
      path: malformed.path,
      message: animationMalformedChannelMessage(malformed),
    });
  }

  if (report.materials.specularGlossinessMaterialCount > 0) {
    addIssue({
      category: 'material',
      name: 'KHR_materials_pbrSpecularGlossiness',
      support: 'approximate',
      path: firstSourcePath(report.materials.issuePaths, 'extension:KHR_materials_pbrSpecularGlossiness', 'materials'),
      message:
        'Archived specular-glossiness materials are converted approximately to metallic-roughness plus specular fields.',
    });
  }

  if (report.materials.specularGlossinessTextureCount > 0) {
    addIssue({
      category: 'material',
      name: 'KHR_materials_pbrSpecularGlossiness.specularGlossinessTexture.glossinessAlpha',
      support: 'approximate',
      path: firstSourcePath(
        report.materials.issuePaths,
        'specGlossGlossinessAlpha',
        'materials',
      ),
      message:
        'Archived specular-glossiness texture RGB is imported as specularColorMap, ' +
        'and raw import uses scalar glossinessFactor for roughness until the texture ' +
        'decode bridge can bake glossiness-in-alpha into a CPU-linear roughnessMap.',
    });
  }

  if (report.materials.doubleSidedCount > 0) {
    addIssue({
      category: 'material',
      name: 'doubleSided',
      support: 'approximate',
      path: firstSourcePath(report.materials.issuePaths, 'doubleSided', 'materials'),
      message:
        'glTF doubleSided is preserved in MaterialSpec.extensions for host inspection, ' +
        'but Vitrum has no first-class double-sided/backface-normal contract yet.',
    });
  }

  for (const uvSet of report.materials.uvSets) {
    if (uvSet <= 1) continue;
    if (!report.materials.unrepresentableUvSets.includes(uvSet)) continue;
    addIssue({
      category: 'material',
      name: `TEXCOORD_${uvSet}`,
      support: 'unsupported',
      path: firstSourcePath(report.materials.issuePaths, `uvSet:${uvSet}`, 'materials'),
      message:
        `glTF material textures reference TEXCOORD_${uvSet}, but the core Scene ` +
        'contract carries only UV sets 0 and 1 (`uvs` / `uv1`) and this asset ' +
        'cannot be losslessly remapped into the uv1 lane.',
    });
  }

  for (const samplerPolicy of report.materials.samplerPolicies) {
    const field = samplerPolicy.materialField;
    const fieldSupport = profile.materialOverrides?.[field] ?? ledger.supportDetails.materials[field] ?? 'unknown';
    if (fieldSupport === 'unsupported') continue;
    const samplerSupport = samplerPolicySupport(profile.id, samplerPolicy);
    if (samplerSupport !== 'native') {
      addIssue({
        category: 'material',
        name: `${String(field)}.samplerPolicy`,
        support: samplerSupport,
        path: samplerPolicy.path,
        message:
          `glTF material texture "${String(field)}" authors sampler filtering at ${samplerPolicy.path}; ` +
          `backend profile ${profile.id} imports the texture but does not guarantee exact per-texture ` +
          `filter/mipmap policy for that material path (${samplerPolicy.materialPath}).`,
      });
    }
  }

  for (const malformedSampler of report.materials.malformedSamplerPolicies) {
    const field = malformedSampler.materialField;
    const fieldSupport = profile.materialOverrides?.[field] ?? ledger.supportDetails.materials[field] ?? 'unknown';
    if (fieldSupport === 'unsupported') continue;
    addIssue({
      category: 'material',
      name: `${String(field)}.samplerPolicy.${malformedSampler.kind}`,
      support: 'approximate',
      path: malformedSampler.path,
      message: malformedSamplerPolicyMessage(profile.id, malformedSampler),
    });
  }

  for (const textureIssue of report.materials.textureReferenceIssues) {
    const field = textureIssue.materialField;
    const fieldSupport = profile.materialOverrides?.[field] ?? ledger.supportDetails.materials[field] ?? 'unknown';
    if (fieldSupport === 'unsupported') continue;
    addIssue({
      category: 'material',
      name: `${String(field)}.textureRef.${textureIssue.kind}`,
      support: materialTextureReferenceIssueSupport(textureIssue),
      path: textureIssue.path,
      message: materialTextureReferenceIssueMessage(profile.id, textureIssue),
    });
  }

  for (const materialIssue of report.materials.primitiveMaterialReferenceIssues) {
    addIssue({
      category: 'material',
      name: `primitive.material.${materialIssue.kind}`,
      support: 'approximate',
      path: materialIssue.path,
      message: primitiveMaterialReferenceIssueMessage(materialIssue),
    });
  }

  for (const variantIssue of report.materials.variantMappingIssues) {
    addIssue({
      category: 'material',
      name: variantIssue.kind === 'malformed-root-variant-list'
        ? 'KHR_materials_variants.variants.malformed-list'
        : `KHR_materials_variants.mapping.${variantIssue.kind}`,
      support: 'unsupported',
      path: variantIssue.path,
      message: materialVariantMappingIssueMessage(variantIssue),
    });
  }

  if (report.materials.textureFields.includes('emissiveMap')) {
    const support = profile.materialOverrides?.emissiveMap ?? ledger.supportDetails.materials.emissiveMap ?? 'unknown';
    if (support === 'native' || support === 'approximate') {
      addIssue({
        category: 'material',
        name: 'emissiveMap.texelPdf',
        support: 'approximate',
        path: firstSourcePath(report.materials.issuePaths, 'field:emissiveMap', 'materials'),
        message:
          `Backend profile ${profile.id} imports glTF emissiveTexture for visible emission, ` +
          'and CPU-readable maps may be subdivided into UV-local mesh emitter records, ' +
          'while GI/probe hit shading samples readable texels at the hit UV. Exact ' +
          'global texel-space emitter alias tables/PDFs are still not guaranteed across ' +
          'every GI, RC, DDGI, and fallback sampling path.',
      });
    }
  }

  for (const field of report.materials.materialFields) {
    const support = profile.materialOverrides?.[field] ?? ledger.supportDetails.materials[field] ?? 'unknown';
    if (support === 'native') {
      nativeCount += 1;
    } else {
      addIssue({
        category: 'material',
        name: String(field),
        support,
        path: firstSourcePath(report.materials.issuePaths, `field:${String(field)}`, 'materials'),
        message: `Backend profile ${profile.id} reports material field "${String(field)}" as ${support}.`,
      });
    }
  }

  return {
    backend,
    profileId: profile.id,
    ...(profile.traceTier !== undefined ? { traceTier: profile.traceTier } : {}),
    unsupportedCount,
    approximateCount,
    nativeCount,
    requiresHookCount,
    issues,
    isCompatible: unsupportedCount === 0,
  };
}

export function rankGltfBackends(
  report: GltfFeatureReport,
  policy: GltfBackendPolicy = 'fidelity',
): readonly GltfBackendCompatibility[] {
  const preferred: readonly GltfBackendProfileId[] = policy === 'realtime'
    ? ['walkaround-hybrid', 'pt-webgpu', 'pt-webgpu-lite', 'pt-webgl2']
    : ['pt-webgl2', 'pt-webgpu', 'pt-webgpu-lite', 'walkaround-hybrid'];
  const order = new Map(preferred.map((b, i) => [b, i]));
  return preferred
    .map((profileId) => evaluateGltfBackendProfileCompatibility(report, profileId))
    .sort((a, b) => {
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

export function recommendGltfBackend(
  report: GltfFeatureReport,
  policy: GltfBackendPolicy = 'fidelity',
): GltfBackendCompatibility {
  return rankGltfBackends(report, policy)[0]!;
}

function samplerPolicySupport(
  profileId: GltfBackendProfileId,
  policy: GltfTextureSamplerPolicyUse,
): BackendSupportMode {
  if (profileId === 'walkaround-hybrid') return 'approximate';
  if (profileId === 'pt-webgl2') {
    if ((policy.magFilter ?? 'nearest') !== 'nearest') return 'approximate';
    if ((policy.minFilter ?? 'nearest') !== 'nearest') return 'approximate';
    if ((policy.mipFilter ?? 'none') !== 'none') return 'approximate';
    return 'native';
  }
  if (policy.materialField === 'bumpMap') {
    if ((policy.magFilter ?? 'linear') !== 'linear') return 'approximate';
    if ((policy.minFilter ?? 'linear') !== 'linear') return 'approximate';
    if (policy.mipFilter !== undefined && policy.mipFilter !== 'none') {
      return 'approximate';
    }
  }
  return 'native';
}

function malformedSamplerPolicyMessage(
  profileId: GltfBackendProfileId,
  issue: GltfMalformedTextureSamplerPolicyUse,
): string {
  if (issue.kind === 'missing-sampler') {
    return (
      `glTF material texture "${String(issue.materialField)}" references sampler ${issue.samplerIndex} at ` +
      `${issue.path}, but that sampler is missing; backend profile ${profileId} imports the texture with default sampler settings.`
    );
  }
  return (
    `glTF material texture "${String(issue.materialField)}" has malformed sampler value ` +
    `${String(issue.value)} at ${issue.path}; backend profile ${profileId} imports the texture with default/fallback ` +
    `sampler settings for that material path (${issue.materialPath}).`
  );
}

function materialTextureReferenceIssueSupport(
  issue: GltfMaterialTextureReferenceIssue,
): BackendSupportMode | 'requires-hook' {
  return issue.kind === 'disabled-texture-source-extension' ? 'requires-hook' : 'approximate';
}

function materialTextureReferenceIssueMessage(
  profileId: GltfBackendProfileId,
  issue: GltfMaterialTextureReferenceIssue,
): string {
  if (issue.kind === 'missing-texture') {
    return (
      `glTF material texture "${String(issue.materialField)}" references missing texture index ` +
      `${issue.textureIndex} at ${issue.path}; backend profile ${profileId} imports the material without that texture.`
    );
  }
  if (issue.kind === 'disabled-texture-source-extension') {
    return (
      `glTF material texture "${String(issue.materialField)}" at ${issue.materialPath} has no base texture.source ` +
      `and only provides ${issue.textureSourceExtensions?.join(', ') ?? 'texture-source extension'} image sources; ` +
      `backend profile ${profileId} needs a host texture-source decode hook or it imports the material without that texture.`
    );
  }
  if (issue.kind === 'missing-texture-source') {
    return (
      `glTF material texture "${String(issue.materialField)}" references textures[${issue.textureIndex}], ` +
      `but that texture has no image source at ${issue.path}; backend profile ${profileId} imports the material without that texture.`
    );
  }
  if (issue.kind === 'missing-image') {
    return (
      `glTF material texture "${String(issue.materialField)}" references image ${String(issue.imageIndex)} at ` +
      `${issue.path}, but that image is missing; backend profile ${profileId} imports the material without that texture.`
    );
  }
  if (issue.kind === 'missing-image-source') {
    return (
      `glTF material texture "${String(issue.materialField)}" resolves to image ${String(issue.imageIndex)}, ` +
      `but the image has neither uri nor bufferView at ${issue.path}; backend profile ${profileId} imports the material without that texture.`
    );
  }
  if (issue.kind === 'missing-image-buffer-view') {
    return (
      `glTF material texture "${String(issue.materialField)}" resolves to image ${String(issue.imageIndex)}, ` +
      `but image bufferView ${String(issue.bufferViewIndex)} is missing at ${issue.path}; backend profile ${profileId} imports the material without that texture.`
    );
  }
  return (
    `glTF material texture "${String(issue.materialField)}" resolves to image ${String(issue.imageIndex)}, ` +
    `but image bufferView ${String(issue.bufferViewIndex)} references missing buffer ${String(issue.bufferIndex)} at ${issue.path}; ` +
    `backend profile ${profileId} imports the material without that texture.`
  );
}

function materialVariantMappingIssueMessage(issue: GltfMaterialVariantMappingIssue): string {
  if (issue.kind === 'malformed-root-variant-list') {
    return (
      'glTF KHR_materials_variants declares a missing or malformed root variants array; ' +
      'variant selection cannot be matched safely.'
    );
  }
  const label = `glTF mesh ${issue.meshIndex} primitive ${issue.primitiveIndex}`;
  if (issue.kind === 'missing-material') {
    return (
      `${label} KHR_materials_variants mapping ${issue.mappingIndex} references ` +
      `missing material ${String(issue.materialIndex)}; selecting that variant falls back to the base material.`
    );
  }
  return (
    `${label} KHR_materials_variants mapping ${issue.mappingIndex} references ` +
    (issue.kind === 'missing-variant-list'
      ? 'a missing or malformed variants array'
      : `missing variant ${String(issue.variantIndex)}`) +
    '; strict backend selection rejects this broken variant route.'
  );
}

function primitiveMaterialReferenceIssueMessage(issue: GltfPrimitiveMaterialReferenceIssue): string {
  return (
    `glTF mesh ${issue.meshIndex} primitive ${issue.primitiveIndex} references missing ` +
    `material ${String(issue.materialIndex)}; the importer falls back to the default material.`
  );
}

function animationMalformedChannelMessage(issue: GltfAnimationMalformedChannelIssue): string {
  if (issue.kind === 'missing-sampler') {
    return `glTF animation channel ${issue.animationIndex}:${issue.channelIndex} references sampler ${String(issue.samplerIndex)} which does not exist; the importer skips the channel.`;
  }
  if (issue.kind === 'missing-input-accessor' || issue.kind === 'missing-output-accessor') {
    return (
      `glTF animation channel ${issue.animationIndex}:${issue.channelIndex} references sampler ` +
      `${String(issue.samplerIndex)} ${String(issue.accessorRole)} accessor ${String(issue.accessorIndex)} ` +
      'which does not exist; the importer skips the channel.'
    );
  }
  if (issue.kind === 'invalid-input-accessor-type' || issue.kind === 'invalid-output-accessor-type') {
    return (
      `glTF animation channel ${issue.animationIndex}:${issue.channelIndex} references sampler ` +
      `${String(issue.samplerIndex)} ${String(issue.accessorRole)} accessor ${String(issue.accessorIndex)} ` +
      `with invalid accessor type "${String(issue.accessorType)}"; the importer skips the channel.`
    );
  }
  if (issue.kind === 'invalid-input-accessor-component-type' || issue.kind === 'invalid-output-accessor-component-type') {
    return (
      `glTF animation channel ${issue.animationIndex}:${issue.channelIndex} references sampler ` +
      `${String(issue.samplerIndex)} ${String(issue.accessorRole)} accessor ${String(issue.accessorIndex)} ` +
      `with unsupported componentType ${String(issue.componentType)}; the importer skips the channel.`
    );
  }
  if (issue.kind === 'missing-input-buffer-view' || issue.kind === 'missing-output-buffer-view') {
    return (
      `glTF animation channel ${issue.animationIndex}:${issue.channelIndex} references sampler ` +
      `${String(issue.samplerIndex)} ${String(issue.accessorRole)} accessor ${String(issue.accessorIndex)} ` +
      `whose bufferView ${String(issue.bufferViewIndex)} is missing; the importer skips the channel.`
    );
  }
  if (issue.kind === 'missing-input-buffer' || issue.kind === 'missing-output-buffer') {
    return (
      `glTF animation channel ${issue.animationIndex}:${issue.channelIndex} references sampler ` +
      `${String(issue.samplerIndex)} ${String(issue.accessorRole)} accessor ${String(issue.accessorIndex)} ` +
      `whose bufferView ${String(issue.bufferViewIndex)} references missing buffer ${String(issue.bufferIndex)}; ` +
      'the importer skips the channel.'
    );
  }
  if (issue.kind === 'invalid-input-sparse-accessor' || issue.kind === 'invalid-output-sparse-accessor') {
    return (
      `glTF animation channel ${issue.animationIndex}:${issue.channelIndex} references sampler ` +
      `${String(issue.samplerIndex)} ${String(issue.accessorRole)} accessor ${String(issue.accessorIndex)} ` +
      `with malformed sparse storage (${sparseAccessorStorageIssueMessage(issue)}); ` +
      'the importer skips that sparse patch and the sampled animation data is incomplete.'
    );
  }
  if (issue.kind === 'missing-target-node') {
    return `glTF animation channel ${issue.animationIndex}:${issue.channelIndex} has no target node; extension-targeted animation channels are not imported.`;
  }
  if (issue.kind === 'target-node-not-found') {
    return `glTF animation channel ${issue.animationIndex}:${issue.channelIndex} targets node ${String(issue.nodeIndex)} which does not exist; the importer skips the channel.`;
  }
  return (
    `glTF animation channel ${issue.animationIndex}:${issue.channelIndex} has ` +
    `${String(issue.actualOutputFloats)} output floats but the target path expects ` +
    `${String(issue.expectedOutputFloats)} for the authored keyframes; the importer skips the channel.`
  );
}

function malformedPrimitiveMessage(issue: GltfMalformedPrimitiveIssue): string {
  const label = `glTF mesh ${issue.meshIndex} primitive ${issue.primitiveIndex}`;
  if (issue.kind === 'missing-position') {
    return `${label} has no POSITION attribute; the importer skips the primitive.`;
  }
  if (issue.kind === 'missing-position-accessor') {
    return `${label} references POSITION accessor ${String(issue.accessorIndex)} which does not exist; the importer skips the primitive.`;
  }
  if (issue.kind === 'invalid-position-accessor-type') {
    return `${label} references POSITION accessor ${String(issue.accessorIndex)} with invalid accessor type "${String(issue.accessorType)}"; the importer skips the primitive.`;
  }
  if (issue.kind === 'invalid-position-accessor-component-type') {
    return (
      `${label} references POSITION accessor ${String(issue.accessorIndex)} with unsupported ` +
      `componentType ${String(issue.componentType)}; the importer skips the primitive.`
    );
  }
  if (issue.kind === 'missing-position-buffer-view') {
    return `${label} references POSITION accessor ${String(issue.accessorIndex)} whose bufferView is missing; the importer skips the primitive.`;
  }
  if (issue.kind === 'missing-position-buffer') {
    return (
      `${label} references POSITION accessor ${String(issue.accessorIndex)} whose bufferView ` +
      `${String(issue.bufferViewIndex)} references missing buffer ${String(issue.bufferIndex)}; the importer skips the primitive.`
    );
  }
  if (issue.kind === 'invalid-position-sparse-accessor') {
    return (
      `${label} references POSITION accessor ${String(issue.accessorIndex)} with malformed sparse storage ` +
      `(${sparseAccessorStorageIssueMessage(issue)}); the importer skips that sparse patch and geometry is incomplete.`
    );
  }
  if (issue.kind === 'missing-index-accessor') {
    return `${label} references index accessor ${String(issue.accessorIndex)} which does not exist; the importer skips the primitive.`;
  }
  if (issue.kind === 'invalid-index-accessor') {
    return (
      `${label} references index accessor ${String(issue.accessorIndex)} with type ` +
      `${String(issue.accessorType)} / componentType ${String(issue.componentType)}; the importer skips the primitive.`
    );
  }
  if (issue.kind === 'missing-index-buffer-view') {
    return `${label} references index accessor ${String(issue.accessorIndex)} whose bufferView is missing; the importer skips the primitive.`;
  }
  if (issue.kind === 'missing-index-buffer') {
    return (
      `${label} references index accessor ${String(issue.accessorIndex)} whose bufferView ` +
      `${String(issue.bufferViewIndex)} references missing buffer ${String(issue.bufferIndex)}; the importer skips the primitive.`
    );
  }
  if (issue.kind === 'invalid-index-sparse-accessor') {
    return (
      `${label} references index accessor ${String(issue.accessorIndex)} with malformed sparse storage ` +
      `(${sparseAccessorStorageIssueMessage(issue)}); the importer cannot faithfully reconstruct the index stream.`
    );
  }
  return `${label} uses topology mode ${String(issue.mode)} but the authored accessor counts cannot produce any triangles; the importer skips the primitive.`;
}

function sparseAccessorStorageIssueMessage(
  issue: {
    readonly sparseIssueKind?: GltfSparseAccessorStorageIssueKind;
    readonly bufferViewIndex?: number;
    readonly bufferIndex?: number;
    readonly componentType?: number;
  },
): string {
  if (issue.sparseIssueKind === 'missing-sparse-indices-buffer-view') {
    return `sparse indices bufferView ${String(issue.bufferViewIndex)} is missing`;
  }
  if (issue.sparseIssueKind === 'missing-sparse-indices-buffer') {
    return (
      `sparse indices bufferView ${String(issue.bufferViewIndex)} references missing ` +
      `buffer ${String(issue.bufferIndex)}`
    );
  }
  if (issue.sparseIssueKind === 'missing-sparse-values-buffer-view') {
    return `sparse values bufferView ${String(issue.bufferViewIndex)} is missing`;
  }
  if (issue.sparseIssueKind === 'missing-sparse-values-buffer') {
    return (
      `sparse values bufferView ${String(issue.bufferViewIndex)} references missing ` +
      `buffer ${String(issue.bufferIndex)}`
    );
  }
  if (issue.sparseIssueKind === 'invalid-sparse-indices-component-type') {
    return (
      `sparse indices componentType ${String(issue.componentType)} is invalid; ` +
      'expected UNSIGNED_BYTE, UNSIGNED_SHORT, or UNSIGNED_INT'
    );
  }
  return 'unknown sparse accessor storage issue';
}

function primitiveAccessorStorageIssueMessage(issue: GltfPrimitiveAccessorStorageIssue): string {
  const location = issue.meshIndex !== undefined && issue.primitiveIndex !== undefined
    ? `glTF mesh ${issue.meshIndex} primitive ${issue.primitiveIndex}`
    : issue.nodeIndex !== undefined
      ? `glTF node ${issue.nodeIndex}`
      : issue.skinIndex !== undefined
        ? `glTF skin ${issue.skinIndex}`
        : 'glTF primitive input';
  return (
    `${location} ${issue.semantic} accessor ${issue.accessorIndex} has malformed sparse storage ` +
    `(${sparseAccessorStorageIssueMessage(issue)}); best-effort import skips that sparse patch ` +
    'and imports degraded attribute data.'
  );
}

function primitiveAccessorImportIssueMessage(issue: GltfPrimitiveAccessorImportIssue): string {
  const location = issue.meshIndex !== undefined && issue.primitiveIndex !== undefined
    ? `glTF mesh ${issue.meshIndex} primitive ${issue.primitiveIndex}`
    : issue.nodeIndex !== undefined
      ? `glTF node ${issue.nodeIndex}`
      : issue.skinIndex !== undefined
        ? `glTF skin ${issue.skinIndex}`
        : 'glTF primitive input';
  const consequence = issue.support === 'unsupported'
    ? 'strict backend selection rejects this because the importer falls back to a less capable primitive representation.'
    : 'backend selection reports this as an approximate import because the importer drops or substitutes that auxiliary stream.';
  if (issue.kind === 'missing-accessor') {
    return `${location} ${issue.semantic} references missing accessor ${issue.accessorIndex}; ${consequence}`;
  }
  if (issue.kind === 'invalid-accessor-type') {
    return (
      `${location} ${issue.semantic} accessor ${issue.accessorIndex} has type ` +
      `${String(issue.accessorType)}, expected ${issue.expectedTypes?.join(' or ') ?? 'a supported type'}; ` +
      consequence
    );
  }
  if (issue.kind === 'invalid-accessor-count') {
    return (
      `${location} ${issue.semantic} accessor ${issue.accessorIndex} has count ` +
      `${String(issue.actualCount)}, expected ${String(issue.expectedCount)}; ${consequence}`
    );
  }
  if (issue.kind === 'invalid-accessor-component-type') {
    return (
      `${location} ${issue.semantic} accessor ${issue.accessorIndex} has unsupported ` +
      `componentType ${String(issue.componentType)}; ${consequence}`
    );
  }
  if (issue.kind === 'missing-buffer-view') {
    return (
      `${location} ${issue.semantic} accessor ${issue.accessorIndex} references missing ` +
      `bufferView ${String(issue.bufferViewIndex)}; ${consequence}`
    );
  }
  return (
    `${location} ${issue.semantic} accessor ${issue.accessorIndex} uses bufferView ` +
    `${String(issue.bufferViewIndex)} referencing missing buffer ${String(issue.bufferIndex)}; ${consequence}`
  );
}

function primitiveInstancingIssueMessage(issue: GltfPrimitiveInstancingIssue): string {
  if (issue.kind === 'missing-attributes') {
    return (
      `glTF node ${issue.nodeIndex} uses EXT_mesh_gpu_instancing without an attributes object; ` +
      'the importer falls back to a single base mesh.'
    );
  }
  if (issue.kind === 'missing-transform-attributes') {
    return (
      `glTF node ${issue.nodeIndex} uses EXT_mesh_gpu_instancing without TRANSLATION, ROTATION, or SCALE accessors; ` +
      'the importer falls back to a single base mesh.'
    );
  }
  return (
    `glTF node ${issue.nodeIndex} EXT_mesh_gpu_instancing attribute ${String(issue.attribute)} ` +
    `must reference a non-negative accessor index, got ${String(issue.value)}; ` +
    'the importer falls back to a single base mesh.'
  );
}

function punctualLightIssueMessage(issue: GltfPunctualLightIssue): string {
  if (issue.kind === 'missing-light') {
    return (
      `glTF node ${String(issue.nodeIndex)} references KHR_lights_punctual light ` +
      `${String(issue.lightIndex)} which does not exist; the importer skips that emitter.`
    );
  }
  return (
    `glTF KHR_lights_punctual light ${String(issue.lightIndex)} has unsupported type ` +
    `"${String(issue.lightType)}"; the importer skips that emitter.`
  );
}

function analyzeExtensions(
  gltf: GltfJson,
  selectedTextureSourceExtensions: ReadonlySet<string>,
  sceneScope: GltfSceneReachability | undefined,
): GltfExtensionReport {
  const sourcePaths: SourcePathMap = new Map();
  const all = new Set<string>();
  if (sceneScope === undefined) {
    for (const ext of gltf.extensionsUsed ?? []) all.add(ext);
    for (const ext of gltf.extensionsRequired ?? []) all.add(ext);
    collectNestedExtensionNames(gltf, all, sourcePaths);
  } else {
    collectScopedNestedExtensionNames(gltf, sceneScope, all, sourcePaths);
  }
  (gltf.extensionsUsed ?? []).forEach((ext, index) => {
    if (sceneScope === undefined || all.has(ext)) addSourcePath(sourcePaths, ext, `extensionsUsed[${index}]`);
  });
  (gltf.extensionsRequired ?? []).forEach((ext, index) => {
    if (!requiredExtensionAppliesToScope(ext, sceneScope, all)) return;
    addSourcePath(sourcePaths, ext, `extensionsRequired[${index}]`);
    if (sceneScope !== undefined && !all.has(ext)) all.add(ext);
  });
  const used = sorted((gltf.extensionsUsed ?? []).filter((ext) => sceneScope === undefined || all.has(ext)));
  const required = sorted((gltf.extensionsRequired ?? []).filter((ext) => requiredExtensionAppliesToScope(ext, sceneScope, all)));

  const supported: string[] = [];
  const requiresHook: string[] = [];
  const unsupportedOptional: string[] = [];
  const unsupportedRequired: string[] = [];

  for (const ext of sorted(all)) {
    if (extensionRequiresHostHook(gltf, ext, required, selectedTextureSourceExtensions, sceneScope)) requiresHook.push(ext);
    if (REQUIRED_EXTENSION_SUPPORT.has(ext)) {
      supported.push(ext);
      continue;
    }
    if (required.includes(ext)) unsupportedRequired.push(ext);
    else if (COMMON_UNSUPPORTED_EXTENSIONS.has(ext) || ext.startsWith('KHR_') || ext.startsWith('EXT_') || ext.startsWith('MSFT_')) {
      unsupportedOptional.push(ext);
    }
  }

  return {
    used,
    required,
    supported: sorted(supported),
    requiresHook: sorted(requiresHook),
    unsupportedOptional: sorted(unsupportedOptional),
    unsupportedRequired: sorted(unsupportedRequired),
    textureSourceUses: collectTextureSourceExtensionUses(gltf, required, selectedTextureSourceExtensions, sceneScope),
    sourcePaths: sourcePathRecord(sourcePaths),
  };
}

function extensionRequiresHostHook(
  gltf: GltfJson,
  ext: string,
  required: readonly string[],
  selectedTextureSourceExtensions: ReadonlySet<string>,
  sceneScope: GltfSceneReachability | undefined,
): boolean {
  if (!EXTENSIONS_REQUIRING_HOST_HOOK.has(ext)) return false;
  if (ext === 'KHR_draco_mesh_compression') {
    return dracoCompressionRequiresHostHook(gltf, required.includes(ext), sceneScope);
  }
  if (MESHOPT_COMPRESSION_EXTENSIONS.has(ext)) {
    return meshoptCompressionRequiresHostHook(gltf, required.includes(ext), sceneScope);
  }
  if (!TEXTURE_SOURCE_EXTENSIONS.has(ext)) return true;

  // Required texture-source extensions cannot be safely ignored. Optional
  // texture-source extensions only need a host hook when they are the sole
  // available image source for at least one texture; otherwise the loader uses
  // the base `texture.source` fallback until the host opts into the extension.
  if (required.includes(ext)) return true;
  if (selectedTextureSourceExtensions.has(ext)) {
    return textureEntriesForScope(gltf, sceneScope).some(([, texture]) =>
      textureSourceExtensionHasImageSource(texture.extensions?.[ext]),
    );
  }
  return textureEntriesForScope(gltf, sceneScope).some(([, texture]) =>
    texture.source === undefined &&
    textureSourceExtensionHasImageSource(texture.extensions?.[ext]),
  );
}

function dracoCompressionRequiresHostHook(
  gltf: GltfJson,
  isRequired: boolean,
  sceneScope: GltfSceneReachability | undefined,
): boolean {
  if (isRequired) return true;

  for (const [meshIndex, mesh] of (gltf.meshes ?? []).entries()) {
    for (const [primitiveIndex, primitive] of mesh.primitives.entries()) {
      if (sceneScope !== undefined && !sceneScope.primitiveKeys.has(gltfPrimitiveKey(meshIndex, primitiveIndex))) continue;
      if (primitive.extensions?.KHR_draco_mesh_compression === undefined) continue;
      if (!dracoPrimitiveHasRealFallback(gltf, primitive)) return true;
    }
  }

  return false;
}

function dracoPrimitiveHasRealFallback(
  gltf: GltfJson,
  primitive: NonNullable<NonNullable<GltfJson['meshes']>[number]['primitives']>[number],
): boolean {
  const accessorIndices = [
    ...Object.values(primitive.attributes ?? {}).filter((value): value is number => typeof value === 'number'),
    ...(primitive.indices !== undefined ? [primitive.indices] : []),
  ];
  return accessorIndices.length > 0 &&
    accessorIndices.every((accessorIndex) => gltf.accessors?.[accessorIndex]?.bufferView !== undefined);
}

function meshoptCompressionRequiresHostHook(
  gltf: GltfJson,
  isRequired: boolean,
  sceneScope: GltfSceneReachability | undefined,
): boolean {
  if (isRequired) return true;
  const scopedBufferViews = sceneScope?.bufferViewIndices;

  for (const [bufferViewIndex, bufferView] of (gltf.bufferViews ?? []).entries()) {
    if (scopedBufferViews !== undefined && !scopedBufferViews.has(bufferViewIndex)) continue;
    if (!hasMeshoptCompressionExtension(bufferView.extensions)) continue;
    if (!meshoptBufferViewHasRealFallback(gltf, bufferView.buffer)) return true;
  }

  return false;
}

function requiredExtensionAppliesToScope(
  ext: string,
  sceneScope: GltfSceneReachability | undefined,
  scopedExtensionNames: ReadonlySet<string>,
): boolean {
  if (sceneScope === undefined || scopedExtensionNames.has(ext)) return true;
  // Supported required extensions are resource-local in vitrum's scoped import
  // policy: if the selected scene does not reach the extension-bearing resource,
  // it should not reject a clean selected-scene load. Unknown required extensions
  // stay asset-level blockers because their semantics are unknowable.
  return !REQUIRED_EXTENSION_SUPPORT.has(ext);
}

function meshoptBufferViewHasRealFallback(gltf: GltfJson, bufferIndex: number): boolean {
  const buffer = gltf.buffers?.[bufferIndex];
  if (!buffer) return false;
  const meshoptExt = meshoptCompressionExtensionValue(buffer.extensions);
  if (meshoptExt == null || typeof meshoptExt !== 'object' || Array.isArray(meshoptExt)) {
    return true;
  }
  return (meshoptExt as { readonly fallback?: unknown }).fallback !== true;
}

function hasMeshoptCompressionExtension(extensions: Record<string, unknown> | undefined): boolean {
  if (!extensions) return false;
  return [...MESHOPT_COMPRESSION_EXTENSIONS].some((name) => extensions[name] !== undefined);
}

function meshoptCompressionExtensionValue(extensions: Record<string, unknown> | undefined): unknown {
  if (!extensions) return undefined;
  for (const name of MESHOPT_COMPRESSION_EXTENSIONS) {
    const value = extensions[name];
    if (value !== undefined) return value;
  }
  return undefined;
}

function textureSourceExtensionHasImageSource(value: unknown): boolean {
  if (value == null || typeof value !== 'object' || Array.isArray(value)) return false;
  return typeof (value as { readonly source?: unknown }).source === 'number';
}

function collectTextureSourceExtensionUses(
  gltf: GltfJson,
  required: readonly string[],
  selectedTextureSourceExtensions: ReadonlySet<string>,
  sceneScope: GltfSceneReachability | undefined,
): readonly GltfTextureSourceExtensionUse[] {
  const uses: GltfTextureSourceExtensionUse[] = [];
  for (const [textureIndex, texture] of textureEntriesForScope(gltf, sceneScope)) {
    for (const extension of TEXTURE_SOURCE_EXTENSION_NAMES) {
      const source = texture.extensions?.[extension]?.source;
      if (typeof source !== 'number') continue;
      const requiredUse = required.includes(extension);
      const selectedUse = selectedTextureSourceExtensions.has(extension);
      const hasBaseSource = texture.source !== undefined;
      uses.push({
        extension,
        textureIndex,
        sourceImageIndex: source,
        path: `textures[${textureIndex}].extensions.${extension}`,
        selected: selectedUse,
        required: requiredUse,
        hasBaseSource,
        requiresHook: requiredUse || selectedUse || !hasBaseSource,
        ...(gltf.images?.[source]?.mimeType !== undefined ? { mimeType: gltf.images[source]!.mimeType } : {}),
      });
    }
  }
  return uses;
}

function analyzeResources(gltf: GltfJson): GltfResourceFeatureReport {
  const buffers = (gltf.buffers ?? []).map((buffer, index): GltfResourceUse => {
    if (buffer.uri == null) return { index, kind: 'embedded' };
    if (buffer.uri.startsWith('data:')) return { index, kind: 'data-uri', uri: buffer.uri };
    return { index, kind: 'external-uri', uri: buffer.uri };
  });
  const images = (gltf.images ?? []).map((image, index) => classifyImage(image, index));
  return {
    buffers,
    images,
    textureCount: gltf.textures?.length ?? 0,
    externalBufferCount: buffers.filter((b) => b.kind === 'external-uri').length,
    externalImageCount: images.filter((i) => i.kind === 'external-uri').length,
  };
}

function classifyImage(image: GltfImage, index: number): GltfResourceUse {
  const mime = image.mimeType !== undefined ? { mimeType: image.mimeType } : {};
  if (image.bufferView !== undefined) return { index, kind: 'bufferView', ...mime };
  if (image.uri == null) return { index, kind: 'missing', ...mime };
  if (image.uri.startsWith('data:')) return { index, kind: 'data-uri', uri: image.uri, ...mime };
  return { index, kind: 'external-uri', uri: image.uri, ...mime };
}

function analyzePrimitives(
  gltf: GltfJson,
  sceneScope: GltfSceneReachability | undefined,
): GltfPrimitiveFeatureReport {
  const byMode = new Map<string, number>();
  const unsupportedModes = new Set<string>();
  const fallbackGeneratedModes = new Set<string>();
  const attributeSemantics = new Set<string>();
  const issuePaths: SourcePathMap = new Map();
  let total = 0;
  let usesDraco = false;
  let usesMeshopt = false;
  let hasTangents = false;
  let hasMorphTargets = false;
  let hasMorphTargetTangents = false;
  let hasMorphTargetTexcoords = false;
  let hasUnsupportedMorphTargetTexcoords = false;
  let hasVertexColors = false;
  const ignoredVertexColorSets = new Set<string>();
  let hasUv1 = false;
  let hasBoundSkinAttrs = false;
  let hasInstancing = false;
  let hasInstancedSkinnedOrMorphed = false;
  let hasIgnoredSkinAttributes = false;
  let hasIncompleteSkinAttributes = false;
  const malformedPrimitives: GltfMalformedPrimitiveIssue[] = [];
  const accessorStorageIssues: GltfPrimitiveAccessorStorageIssue[] = [];
  const accessorImportIssues: GltfPrimitiveAccessorImportIssue[] = [];
  const instancingIssues: GltfPrimitiveInstancingIssue[] = [];
  const meshNodesWithSkin = new Map<number, string[]>();
  const meshNodesWithoutSkin = new Map<number, string[]>();
  for (const [nodeIndex, node] of (gltf.nodes ?? []).entries()) {
    if (sceneScope !== undefined && !sceneScope.nodeIndices.has(nodeIndex)) continue;
    if (node.mesh === undefined) continue;
    const target = node.skin !== undefined ? meshNodesWithSkin : meshNodesWithoutSkin;
    const paths = target.get(node.mesh) ?? [];
    paths.push(`nodes[${nodeIndex}]`);
    target.set(node.mesh, paths);
  }

  for (const [meshIndex, mesh] of (gltf.meshes ?? []).entries()) {
    for (const [primitiveIndex, primitive] of (mesh.primitives ?? []).entries()) {
      if (sceneScope !== undefined && !sceneScope.primitiveKeys.has(gltfPrimitiveKey(meshIndex, primitiveIndex))) continue;
      const primitivePath = `meshes[${meshIndex}].primitives[${primitiveIndex}]`;
      total += 1;
      addSourcePath(issuePaths, 'kind:mesh', primitivePath);
      const mode = primitive.mode ?? 4;
      const modeKey = String(mode);
      byMode.set(modeKey, (byMode.get(modeKey) ?? 0) + 1);
      const supportedOrFallbackMode = SUPPORTED_GLTF_PRIMITIVE_MODES.has(mode);
      if (FALLBACK_GENERATED_PRIMITIVE_MODES.has(mode)) {
        fallbackGeneratedModes.add(modeKey);
        addSourcePath(issuePaths, `mode:${modeKey}`, `${primitivePath}.mode`);
      } else if (!SUPPORTED_GLTF_PRIMITIVE_MODES.has(mode)) {
        unsupportedModes.add(modeKey);
        addSourcePath(issuePaths, `mode:${modeKey}`, `${primitivePath}.mode`);
      }
      if (supportedOrFallbackMode) {
        malformedPrimitives.push(
          ...primitiveImportBlockers(gltf, meshIndex, primitiveIndex, primitive, primitivePath, mode),
        );
      }
      const primitiveVertexCount = primitive.attributes?.POSITION !== undefined
        ? gltf.accessors?.[primitive.attributes.POSITION]?.count
        : undefined;
      const meshHasSkinnedNode = meshNodesWithSkin.has(meshIndex);
      const meshHasUnskinnedNode = meshNodesWithoutSkin.has(meshIndex);
      for (const [semantic, accessorIndex] of Object.entries(primitive.attributes ?? {})) {
        attributeSemantics.add(semantic);
        if (semantic !== 'POSITION' && accessorIndex !== undefined) {
          addPrimitiveAccessorStorageIssue(gltf, accessorStorageIssues, {
            semantic: `attributes.${semantic}`,
            accessorIndex,
            meshIndex,
            primitiveIndex,
          });
          addPrimitiveAccessorImportIssueForAttribute(gltf, accessorImportIssues, {
            semantic,
            accessorIndex,
            expectedCount: primitiveVertexCount,
            support: meshHasSkinnedNode && (semantic === 'JOINTS_0' || semantic === 'WEIGHTS_0')
              ? 'unsupported'
              : 'approximate',
            meshIndex,
            primitiveIndex,
          });
        }
        if (semantic === 'TANGENT') hasTangents = true;
        if (semantic === 'COLOR_0') {
          hasVertexColors = true;
          addSourcePath(issuePaths, 'vertexColors', `${primitivePath}.attributes.COLOR_0`);
        } else if (/^COLOR_[1-9][0-9]*$/.test(semantic)) {
          ignoredVertexColorSets.add(semantic);
          addSourcePath(issuePaths, `ignoredVertexColorSet:${semantic}`, `${primitivePath}.attributes.${semantic}`);
        }
        if (semantic === 'TEXCOORD_1') hasUv1 = true;
      }
      const hasJoints = primitive.attributes?.JOINTS_0 !== undefined;
      const hasWeights = primitive.attributes?.WEIGHTS_0 !== undefined;
      if (meshHasSkinnedNode) {
        if (hasJoints && hasWeights) {
          hasBoundSkinAttrs = true;
          addSourcePath(issuePaths, 'kind:skinned-mesh', `${primitivePath}.attributes.JOINTS_0`);
          addSourcePath(issuePaths, 'kind:skinned-mesh', `${primitivePath}.attributes.WEIGHTS_0`);
        } else {
          hasIncompleteSkinAttributes = true;
          addSourcePath(issuePaths, 'incompleteSkinAttributes', primitivePath);
        }
      }
      if (hasJoints || hasWeights) {
        if (!meshHasSkinnedNode || meshHasUnskinnedNode) {
          hasIgnoredSkinAttributes = true;
          if (hasJoints) addSourcePath(issuePaths, 'ignoredSkinAttributes', `${primitivePath}.attributes.JOINTS_0`);
          if (hasWeights) addSourcePath(issuePaths, 'ignoredSkinAttributes', `${primitivePath}.attributes.WEIGHTS_0`);
        }
      }
      const ext = primitive.extensions ?? {};
      if (ext['KHR_draco_mesh_compression']) usesDraco = true;
      if (hasMeshoptCompressionExtension(ext)) usesMeshopt = true;
      if ((primitive.targets?.length ?? 0) > 0) {
        hasMorphTargets = true;
        addSourcePath(issuePaths, 'kind:skinned-mesh', `${primitivePath}.targets`);
        for (const [targetIndex, target] of (primitive.targets ?? []).entries()) {
          if (target['TANGENT'] !== undefined) {
            hasMorphTargetTangents = true;
            addSourcePath(issuePaths, 'morphTargetTangents', `${primitivePath}.targets[${targetIndex}].TANGENT`);
          }
          for (const [attr, accessorIndex] of Object.entries(target)) {
            addPrimitiveAccessorStorageIssue(gltf, accessorStorageIssues, {
              semantic: `targets.${attr}`,
              accessorIndex,
              meshIndex,
              primitiveIndex,
              targetIndex,
            });
            addPrimitiveAccessorImportIssueForMorphTarget(gltf, accessorImportIssues, {
              semantic: attr,
              accessorIndex,
              expectedCount: primitiveVertexCount,
              meshIndex,
              primitiveIndex,
              targetIndex,
            });
            if (/^TEXCOORD_\d+$/.test(attr)) {
              hasMorphTargetTexcoords = true;
              addSourcePath(issuePaths, 'morphTargetTexcoords', `${primitivePath}.targets[${targetIndex}].${attr}`);
              const uvIndex = Number(attr.slice('TEXCOORD_'.length));
              if (!primitiveMorphTexcoordIsRepresentable(gltf, primitive, uvIndex)) {
                hasUnsupportedMorphTargetTexcoords = true;
                addSourcePath(issuePaths, 'unsupportedMorphTargetTexcoords', `${primitivePath}.targets[${targetIndex}].${attr}`);
              }
            }
          }
        }
      }
    }
  }
  for (const [nodeIndex, node] of (gltf.nodes ?? []).entries()) {
    if (sceneScope !== undefined && !sceneScope.nodeIndices.has(nodeIndex)) continue;
    if (node.mesh === undefined || node.extensions?.EXT_mesh_gpu_instancing === undefined) continue;
    hasInstancing = true;
    const instancingPath = `nodes[${nodeIndex}].extensions.EXT_mesh_gpu_instancing`;
    addSourcePath(issuePaths, 'kind:instanced-mesh', instancingPath);
    const instancingExtension = node.extensions.EXT_mesh_gpu_instancing as unknown;
    const instancingAttributes = isJsonRecord(instancingExtension)
      ? instancingExtension.attributes
      : undefined;
    if (!isJsonRecord(instancingExtension) || !isJsonRecord(instancingAttributes)) {
      instancingIssues.push({
        kind: 'missing-attributes',
        path: instancingPath,
        nodeIndex,
      });
      continue;
    }
    let instancingCount: number | undefined;
    let hasTransformAttribute = false;
    for (const [attr, rawAccessorIndex] of Object.entries(instancingAttributes)) {
      if (!INSTANCING_TRANSFORM_ATTRIBUTES.has(attr)) continue;
      hasTransformAttribute = true;
      if (
        typeof rawAccessorIndex !== 'number' ||
        !Number.isInteger(rawAccessorIndex) ||
        rawAccessorIndex < 0
      ) {
        instancingIssues.push({
          kind: 'invalid-attribute-accessor-index',
          path: `${instancingPath}.attributes.${attr}`,
          nodeIndex,
          attribute: attr,
          value: rawAccessorIndex,
        });
        continue;
      }
      const accessorIndex = rawAccessorIndex;
      addPrimitiveAccessorStorageIssue(gltf, accessorStorageIssues, {
        semantic: `instancing.${attr}`,
        accessorIndex,
        nodeIndex,
      });
      addPrimitiveAccessorImportIssueForInstancing(gltf, accessorImportIssues, {
        semantic: attr,
        accessorIndex,
        expectedCount: instancingCount,
        nodeIndex,
      });
      const accessor = gltf.accessors?.[accessorIndex];
      if (accessor !== undefined && instancingCount === undefined) instancingCount = accessor.count;
    }
    if (!hasTransformAttribute) {
      instancingIssues.push({
        kind: 'missing-transform-attributes',
        path: `${instancingPath}.attributes`,
        nodeIndex,
      });
    }
    const mesh = gltf.meshes?.[node.mesh];
    const meshHasMorphTargets = (mesh?.primitives ?? []).some((primitive) =>
      (primitive.targets?.length ?? 0) > 0,
    );
    if (node.skin !== undefined || meshHasMorphTargets) {
      hasInstancedSkinnedOrMorphed = true;
      addSourcePath(issuePaths, 'instancedSkinnedOrMorphed', instancingPath);
    }
  }
  for (const [skinIndex, skin] of (gltf.skins ?? []).entries()) {
    if (sceneScope !== undefined && !sceneScope.skinIndices.has(skinIndex)) continue;
    if (skin.inverseBindMatrices === undefined) continue;
    addPrimitiveAccessorStorageIssue(gltf, accessorStorageIssues, {
      semantic: 'skin.inverseBindMatrices',
      accessorIndex: skin.inverseBindMatrices,
      skinIndex,
    });
    addPrimitiveAccessorImportIssue(gltf, accessorImportIssues, {
      semantic: 'skin.inverseBindMatrices',
      accessorIndex: skin.inverseBindMatrices,
      sourcePath: `skins[${skinIndex}].inverseBindMatrices`,
      expectedTypes: ['MAT4'],
      expectedCount: skin.joints.length,
      support: 'approximate',
      skinIndex,
    });
  }
  for (const bv of bufferViewsForScope(gltf, sceneScope)) {
    if (hasMeshoptCompressionExtension(bv.extensions)) usesMeshopt = true;
  }
  const hasSkins = hasBoundSkinAttrs;
  const expectedPrimitiveKinds = new Set<'mesh' | 'skinned-mesh' | 'instanced-mesh'>();
  expectedPrimitiveKinds.add('mesh');
  if (hasBoundSkinAttrs || hasMorphTargets) expectedPrimitiveKinds.add('skinned-mesh');
  if (hasInstancing) expectedPrimitiveKinds.add('instanced-mesh');
  return {
    total,
    byMode: Object.fromEntries([...byMode.entries()].sort()),
    unsupportedModes: sorted(unsupportedModes),
    fallbackGeneratedModes: sorted(fallbackGeneratedModes),
    attributeSemantics: sorted(attributeSemantics),
    expectedPrimitiveKinds: sorted(expectedPrimitiveKinds),
    usesDraco,
    usesMeshopt,
    hasTangents,
    hasMorphTargets,
    hasMorphTargetTangents,
    hasMorphTargetTexcoords,
    hasUnsupportedMorphTargetTexcoords,
    hasSkins,
    hasInstancedSkinnedOrMorphed,
    hasIgnoredSkinAttributes,
    hasIncompleteSkinAttributes,
    malformedPrimitives: malformedPrimitives.sort((a, b) =>
      a.path.localeCompare(b.path) || a.kind.localeCompare(b.kind)
    ),
    accessorStorageIssues: accessorStorageIssues.sort((a, b) =>
      a.path.localeCompare(b.path) ||
      a.semantic.localeCompare(b.semantic) ||
      a.accessorIndex - b.accessorIndex
    ),
    accessorImportIssues: accessorImportIssues.sort((a, b) =>
      a.path.localeCompare(b.path) ||
      a.semantic.localeCompare(b.semantic) ||
      a.accessorIndex - b.accessorIndex
    ),
    instancingIssues: instancingIssues.sort((a, b) =>
      a.path.localeCompare(b.path) || a.kind.localeCompare(b.kind)
    ),
    hasVertexColors,
    ignoredVertexColorSets: sorted(ignoredVertexColorSets),
    hasUv1,
    issuePaths: sourcePathRecord(issuePaths),
  };
}

function primitiveImportBlockers(
  gltf: GltfJson,
  meshIndex: number,
  primitiveIndex: number,
  primitive: GltfPrimitive,
  primitivePath: string,
  mode: number,
): GltfMalformedPrimitiveIssue[] {
  const issues: GltfMalformedPrimitiveIssue[] = [];
  const positionAccessorIndex = primitive.attributes?.POSITION;
  if (positionAccessorIndex === undefined) {
    issues.push({
      kind: 'missing-position',
      path: `${primitivePath}.attributes.POSITION`,
      meshIndex,
      primitiveIndex,
      mode,
    });
    return issues;
  }
  const positionAccessor = gltf.accessors?.[positionAccessorIndex];
  if (positionAccessor == null) {
    issues.push({
      kind: 'missing-position-accessor',
      path: `${primitivePath}.attributes.POSITION`,
      meshIndex,
      primitiveIndex,
      accessorIndex: positionAccessorIndex,
      mode,
    });
    return issues;
  }
  if (accessorComponentCount(positionAccessor.type) === undefined) {
    issues.push({
      kind: 'invalid-position-accessor-type',
      path: `${primitivePath}.attributes.POSITION`,
      meshIndex,
      primitiveIndex,
      accessorIndex: positionAccessorIndex,
      accessorType: String(positionAccessor.type),
      mode,
    });
    return issues;
  }
  if (!floatAccessorComponentTypeIsReadableByImporter(positionAccessor.componentType)) {
    issues.push({
      kind: 'invalid-position-accessor-component-type',
      path: `${primitivePath}.attributes.POSITION`,
      meshIndex,
      primitiveIndex,
      accessorIndex: positionAccessorIndex,
      componentType: positionAccessor.componentType,
      mode,
    });
    return issues;
  }
  if (
    positionAccessor.bufferView !== undefined &&
    gltf.bufferViews?.[positionAccessor.bufferView] == null
  ) {
    issues.push({
      kind: 'missing-position-buffer-view',
      path: `${primitivePath}.attributes.POSITION`,
      meshIndex,
      primitiveIndex,
      accessorIndex: positionAccessorIndex,
      mode,
    });
    return issues;
  }
  if (positionAccessor.bufferView !== undefined) {
    const bufferView = gltf.bufferViews?.[positionAccessor.bufferView];
    if (bufferView !== undefined && gltf.buffers?.[bufferView.buffer] == null) {
      issues.push({
        kind: 'missing-position-buffer',
        path: `bufferViews[${positionAccessor.bufferView}].buffer`,
        meshIndex,
        primitiveIndex,
        accessorIndex: positionAccessorIndex,
        bufferViewIndex: positionAccessor.bufferView,
        bufferIndex: bufferView.buffer,
        mode,
      });
      return issues;
    }
  }
  const positionSparseIssue = sparseAccessorStorageIssue(gltf, positionAccessorIndex, positionAccessor);
  if (positionSparseIssue !== undefined) {
    issues.push({
      kind: 'invalid-position-sparse-accessor',
      path: positionSparseIssue.path,
      meshIndex,
      primitiveIndex,
      accessorIndex: positionAccessorIndex,
      mode,
      ...sparseIssueDetails(positionSparseIssue),
    });
    return issues;
  }

  const indexAccessorIndex = primitive.indices;
  let indexCount: number | undefined;
  if (indexAccessorIndex !== undefined) {
    const indexAccessor = gltf.accessors?.[indexAccessorIndex];
    if (indexAccessor == null) {
      issues.push({
        kind: 'missing-index-accessor',
        path: `${primitivePath}.indices`,
        meshIndex,
        primitiveIndex,
        accessorIndex: indexAccessorIndex,
        mode,
      });
      return issues;
    }
    if (!indexAccessorIsReadableByImporter(indexAccessor)) {
      issues.push({
        kind: 'invalid-index-accessor',
        path: `${primitivePath}.indices`,
        meshIndex,
        primitiveIndex,
        accessorIndex: indexAccessorIndex,
        accessorType: String(indexAccessor.type),
        componentType: indexAccessor.componentType,
        mode,
      });
      return issues;
    }
    if (
      indexAccessor.bufferView !== undefined &&
      gltf.bufferViews?.[indexAccessor.bufferView] == null
    ) {
      issues.push({
        kind: 'missing-index-buffer-view',
        path: `${primitivePath}.indices`,
        meshIndex,
        primitiveIndex,
        accessorIndex: indexAccessorIndex,
        mode,
      });
      return issues;
    }
    if (indexAccessor.bufferView !== undefined) {
      const bufferView = gltf.bufferViews?.[indexAccessor.bufferView];
      if (bufferView !== undefined && gltf.buffers?.[bufferView.buffer] == null) {
        issues.push({
          kind: 'missing-index-buffer',
          path: `bufferViews[${indexAccessor.bufferView}].buffer`,
          meshIndex,
          primitiveIndex,
          accessorIndex: indexAccessorIndex,
          bufferViewIndex: indexAccessor.bufferView,
          bufferIndex: bufferView.buffer,
          mode,
        });
        return issues;
      }
    }
    const indexSparseIssue = sparseAccessorStorageIssue(gltf, indexAccessorIndex, indexAccessor);
    if (indexSparseIssue !== undefined) {
      issues.push({
        kind: 'invalid-index-sparse-accessor',
        path: indexSparseIssue.path,
        meshIndex,
        primitiveIndex,
        accessorIndex: indexAccessorIndex,
        mode,
        ...sparseIssueDetails(indexSparseIssue),
      });
      return issues;
    }
    indexCount = indexAccessor.count;
  }

  if ((mode === 5 || mode === 6) && (indexCount ?? positionAccessor.count) < 3) {
    issues.push({
      kind: 'empty-triangulated-primitive',
      path: primitivePath,
      meshIndex,
      primitiveIndex,
      mode,
    });
  }

  return issues;
}

function indexAccessorIsReadableByImporter(accessor: NonNullable<GltfJson['accessors']>[number]): boolean {
  if (accessor.type !== 'SCALAR') return false;
  return accessor.componentType === 5121 ||
    accessor.componentType === 5123 ||
    accessor.componentType === 5125;
}

function floatAccessorComponentTypeIsReadableByImporter(componentType: number): boolean {
  return componentType === 5120 ||
    componentType === 5121 ||
    componentType === 5122 ||
    componentType === 5123 ||
    componentType === 5125 ||
    componentType === 5126;
}

interface SparseAccessorStorageIssue {
  readonly kind: GltfSparseAccessorStorageIssueKind;
  readonly path: string;
  readonly bufferViewIndex?: number;
  readonly bufferIndex?: number;
  readonly componentType?: number;
}

function sparseAccessorStorageIssue(
  gltf: GltfJson,
  accessorIndex: number,
  accessor: NonNullable<GltfJson['accessors']>[number],
): SparseAccessorStorageIssue | undefined {
  const sparse = accessor.sparse;
  if (sparse === undefined) return undefined;

  const indicesBufferViewIndex = sparse.indices.bufferView;
  const indicesBufferView = gltf.bufferViews?.[indicesBufferViewIndex];
  if (indicesBufferView == null) {
    return {
      kind: 'missing-sparse-indices-buffer-view',
      path: `accessors[${accessorIndex}].sparse.indices.bufferView`,
      bufferViewIndex: indicesBufferViewIndex,
    };
  }
  if (gltf.buffers?.[indicesBufferView.buffer] == null) {
    return {
      kind: 'missing-sparse-indices-buffer',
      path: `bufferViews[${indicesBufferViewIndex}].buffer`,
      bufferViewIndex: indicesBufferViewIndex,
      bufferIndex: indicesBufferView.buffer,
    };
  }

  const valuesBufferViewIndex = sparse.values.bufferView;
  const valuesBufferView = gltf.bufferViews?.[valuesBufferViewIndex];
  if (valuesBufferView == null) {
    return {
      kind: 'missing-sparse-values-buffer-view',
      path: `accessors[${accessorIndex}].sparse.values.bufferView`,
      bufferViewIndex: valuesBufferViewIndex,
    };
  }
  if (gltf.buffers?.[valuesBufferView.buffer] == null) {
    return {
      kind: 'missing-sparse-values-buffer',
      path: `bufferViews[${valuesBufferViewIndex}].buffer`,
      bufferViewIndex: valuesBufferViewIndex,
      bufferIndex: valuesBufferView.buffer,
    };
  }

  if (!sparseIndexComponentTypeIsReadableByImporter(sparse.indices.componentType)) {
    return {
      kind: 'invalid-sparse-indices-component-type',
      path: `accessors[${accessorIndex}].sparse.indices.componentType`,
      componentType: sparse.indices.componentType,
    };
  }

  return undefined;
}

function sparseIssueDetails(issue: SparseAccessorStorageIssue): {
  readonly sparseIssueKind: GltfSparseAccessorStorageIssueKind;
  readonly bufferViewIndex?: number;
  readonly bufferIndex?: number;
  readonly componentType?: number;
} {
  return {
    sparseIssueKind: issue.kind,
    ...(issue.bufferViewIndex !== undefined ? { bufferViewIndex: issue.bufferViewIndex } : {}),
    ...(issue.bufferIndex !== undefined ? { bufferIndex: issue.bufferIndex } : {}),
    ...(issue.componentType !== undefined ? { componentType: issue.componentType } : {}),
  };
}

function addPrimitiveAccessorStorageIssue(
  gltf: GltfJson,
  issues: GltfPrimitiveAccessorStorageIssue[],
  input: {
    readonly semantic: string;
    readonly accessorIndex: number;
    readonly meshIndex?: number;
    readonly primitiveIndex?: number;
    readonly targetIndex?: number;
    readonly nodeIndex?: number;
    readonly skinIndex?: number;
  },
): void {
  const accessor = gltf.accessors?.[input.accessorIndex];
  if (accessor == null) return;
  const sparseIssue = sparseAccessorStorageIssue(gltf, input.accessorIndex, accessor);
  if (sparseIssue === undefined) return;
  issues.push({
    semantic: input.semantic,
    accessorIndex: input.accessorIndex,
    path: sparseIssue.path,
    ...(input.meshIndex !== undefined ? { meshIndex: input.meshIndex } : {}),
    ...(input.primitiveIndex !== undefined ? { primitiveIndex: input.primitiveIndex } : {}),
    ...(input.targetIndex !== undefined ? { targetIndex: input.targetIndex } : {}),
    ...(input.nodeIndex !== undefined ? { nodeIndex: input.nodeIndex } : {}),
    ...(input.skinIndex !== undefined ? { skinIndex: input.skinIndex } : {}),
    ...sparseIssueDetails(sparseIssue),
  });
}

function addPrimitiveAccessorImportIssueForAttribute(
  gltf: GltfJson,
  issues: GltfPrimitiveAccessorImportIssue[],
  input: {
    readonly semantic: string;
    readonly accessorIndex: number;
    readonly expectedCount?: number | undefined;
    readonly support: 'approximate' | 'unsupported';
    readonly meshIndex: number;
    readonly primitiveIndex: number;
  },
): void {
  const expectation = primitiveAttributeAccessorExpectation(input.semantic);
  if (expectation === undefined) return;
  addPrimitiveAccessorImportIssue(gltf, issues, {
    semantic: `attributes.${input.semantic}`,
    accessorIndex: input.accessorIndex,
    sourcePath: `meshes[${input.meshIndex}].primitives[${input.primitiveIndex}].attributes.${input.semantic}`,
    expectedTypes: expectation.expectedTypes,
    expectedCount: input.expectedCount,
    support: input.support,
    meshIndex: input.meshIndex,
    primitiveIndex: input.primitiveIndex,
    componentTypes: expectation.componentTypes,
  });
}

function addPrimitiveAccessorImportIssueForMorphTarget(
  gltf: GltfJson,
  issues: GltfPrimitiveAccessorImportIssue[],
  input: {
    readonly semantic: string;
    readonly accessorIndex: number;
    readonly expectedCount?: number | undefined;
    readonly meshIndex: number;
    readonly primitiveIndex: number;
    readonly targetIndex: number;
  },
): void {
  const expectedTypes = morphTargetAccessorTypes(input.semantic);
  if (expectedTypes === undefined) return;
  addPrimitiveAccessorImportIssue(gltf, issues, {
    semantic: `targets.${input.semantic}`,
    accessorIndex: input.accessorIndex,
    sourcePath: `meshes[${input.meshIndex}].primitives[${input.primitiveIndex}].targets[${input.targetIndex}].${input.semantic}`,
    expectedTypes,
    expectedCount: input.expectedCount,
    support: 'approximate',
    meshIndex: input.meshIndex,
    primitiveIndex: input.primitiveIndex,
    targetIndex: input.targetIndex,
  });
}

function addPrimitiveAccessorImportIssueForInstancing(
  gltf: GltfJson,
  issues: GltfPrimitiveAccessorImportIssue[],
  input: {
    readonly semantic: string;
    readonly accessorIndex: number;
    readonly expectedCount?: number | undefined;
    readonly nodeIndex: number;
  },
): void {
  const expectedTypes = instancingAccessorTypes(input.semantic);
  if (expectedTypes === undefined) return;
  addPrimitiveAccessorImportIssue(gltf, issues, {
    semantic: `instancing.${input.semantic}`,
    accessorIndex: input.accessorIndex,
    sourcePath: `nodes[${input.nodeIndex}].extensions.EXT_mesh_gpu_instancing.attributes.${input.semantic}`,
    expectedTypes,
    expectedCount: input.expectedCount,
    support: 'unsupported',
    nodeIndex: input.nodeIndex,
  });
}

function addPrimitiveAccessorImportIssue(
  gltf: GltfJson,
  issues: GltfPrimitiveAccessorImportIssue[],
  input: {
    readonly semantic: string;
    readonly accessorIndex: number;
    readonly sourcePath: string;
    readonly expectedTypes: readonly string[];
    readonly expectedCount?: number | undefined;
    readonly support: 'approximate' | 'unsupported';
    readonly meshIndex?: number;
    readonly primitiveIndex?: number;
    readonly targetIndex?: number;
    readonly nodeIndex?: number;
    readonly skinIndex?: number;
    readonly componentTypes?: readonly number[] | undefined;
  },
): void {
  const base = {
    semantic: input.semantic,
    accessorIndex: input.accessorIndex,
    support: input.support,
    ...(input.meshIndex !== undefined ? { meshIndex: input.meshIndex } : {}),
    ...(input.primitiveIndex !== undefined ? { primitiveIndex: input.primitiveIndex } : {}),
    ...(input.targetIndex !== undefined ? { targetIndex: input.targetIndex } : {}),
    ...(input.nodeIndex !== undefined ? { nodeIndex: input.nodeIndex } : {}),
    ...(input.skinIndex !== undefined ? { skinIndex: input.skinIndex } : {}),
  } satisfies Omit<GltfPrimitiveAccessorImportIssue, 'kind' | 'path'>;

  const accessor = gltf.accessors?.[input.accessorIndex];
  if (accessor == null) {
    issues.push({
      ...base,
      kind: 'missing-accessor',
      path: input.sourcePath,
    });
    return;
  }
  if (!input.expectedTypes.includes(accessor.type)) {
    issues.push({
      ...base,
      kind: 'invalid-accessor-type',
      path: `accessors[${input.accessorIndex}].type`,
      expectedTypes: input.expectedTypes,
      accessorType: String(accessor.type),
    });
    return;
  }
  const componentTypes = input.componentTypes ?? FLOAT_ACCESSOR_COMPONENT_TYPES;
  if (!componentTypes.includes(accessor.componentType)) {
    issues.push({
      ...base,
      kind: 'invalid-accessor-component-type',
      path: `accessors[${input.accessorIndex}].componentType`,
      componentType: accessor.componentType,
    });
    return;
  }
  if (input.expectedCount !== undefined && accessor.count !== input.expectedCount) {
    issues.push({
      ...base,
      kind: 'invalid-accessor-count',
      path: `accessors[${input.accessorIndex}].count`,
      expectedCount: input.expectedCount,
      actualCount: accessor.count,
    });
    return;
  }
  if (accessor.bufferView === undefined) return;
  const bufferView = gltf.bufferViews?.[accessor.bufferView];
  if (bufferView == null) {
    issues.push({
      ...base,
      kind: 'missing-buffer-view',
      path: `accessors[${input.accessorIndex}].bufferView`,
      bufferViewIndex: accessor.bufferView,
    });
    return;
  }
  if (gltf.buffers?.[bufferView.buffer] == null) {
    issues.push({
      ...base,
      kind: 'missing-buffer',
      path: `bufferViews[${accessor.bufferView}].buffer`,
      bufferViewIndex: accessor.bufferView,
      bufferIndex: bufferView.buffer,
    });
  }
}

const FLOAT_ACCESSOR_COMPONENT_TYPES = [5120, 5121, 5122, 5123, 5125, 5126] as const;
const JOINTS_ACCESSOR_COMPONENT_TYPES = [5121, 5123] as const;
const INSTANCING_TRANSFORM_ATTRIBUTES = new Set(['TRANSLATION', 'ROTATION', 'SCALE']);

function isJsonRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function primitiveAttributeAccessorExpectation(semantic: string): {
  readonly expectedTypes: readonly string[];
  readonly componentTypes?: readonly number[];
} | undefined {
  if (semantic === 'NORMAL') return { expectedTypes: ['VEC3'] };
  if (semantic === 'TANGENT') return { expectedTypes: ['VEC4'] };
  if (semantic === 'COLOR_0') return { expectedTypes: ['VEC3', 'VEC4'] };
  if (semantic === 'JOINTS_0') {
    return { expectedTypes: ['VEC4'], componentTypes: JOINTS_ACCESSOR_COMPONENT_TYPES };
  }
  if (semantic === 'WEIGHTS_0') return { expectedTypes: ['VEC4'] };
  if (/^TEXCOORD_\d+$/.test(semantic)) return { expectedTypes: ['VEC2'] };
  return undefined;
}

function morphTargetAccessorTypes(semantic: string): readonly string[] | undefined {
  if (semantic === 'POSITION' || semantic === 'NORMAL' || semantic === 'TANGENT') return ['VEC3'];
  if (/^TEXCOORD_\d+$/.test(semantic)) return ['VEC2'];
  return undefined;
}

function instancingAccessorTypes(semantic: string): readonly string[] | undefined {
  if (semantic === 'TRANSLATION' || semantic === 'SCALE') return ['VEC3'];
  if (semantic === 'ROTATION') return ['VEC4'];
  return undefined;
}

function sparseIndexComponentTypeIsReadableByImporter(componentType: number): boolean {
  return componentType === 5121 ||
    componentType === 5123 ||
    componentType === 5125;
}

function analyzeUnrepresentableMaterialUvSets(
  gltf: GltfJson,
  materialUvSets: ReadonlyMap<number, ReadonlySet<number>>,
  sceneScope: GltfSceneReachability | undefined,
): readonly number[] {
  const unrepresentable = new Set<number>();
  const usedMaterials = new Set<number>();

  for (const [meshIndex, mesh] of (gltf.meshes ?? []).entries()) {
    for (const [primitiveIndex, primitive] of mesh.primitives.entries()) {
      if (sceneScope !== undefined && !sceneScope.primitiveKeys.has(gltfPrimitiveKey(meshIndex, primitiveIndex))) continue;
      for (const materialIndex of collectPrimitiveMaterialIndices(primitive)) {
        usedMaterials.add(materialIndex);
        const uvSets = materialUvSets.get(materialIndex);
        if (uvSets === undefined) continue;
        const highUvSets = [...uvSets].filter((uvSet) => uvSet > 1);
        if (highUvSets.length === 0) continue;
        const canRemap =
          highUvSets.length === 1 &&
          !uvSets.has(1) &&
          primitive.attributes?.[`TEXCOORD_${highUvSets[0]}`] !== undefined;
        if (!canRemap) {
          for (const uvSet of highUvSets) unrepresentable.add(uvSet);
        }
      }
    }
  }

  for (const [materialIndex, uvSets] of materialUvSets) {
    if (usedMaterials.has(materialIndex)) continue;
    for (const uvSet of uvSets) {
      if (uvSet > 1) unrepresentable.add(uvSet);
    }
  }

  return [...unrepresentable].sort((a, b) => a - b);
}

function primitiveMorphTexcoordIsRepresentable(
  gltf: GltfJson,
  primitive: GltfPrimitive,
  uvIndex: number,
): boolean {
  const baseSemantic = `TEXCOORD_${uvIndex}`;
  if (primitive.attributes?.[baseSemantic] === undefined) return false;
  if (uvIndex <= 1) return true;

  for (const materialIndex of collectPrimitiveMaterialIndices(primitive)) {
    const material = gltf.materials?.[materialIndex];
    if (material == null) continue;
    const uvSets = materialTextureUvSets(material);
    const highUvSets = [...uvSets].filter((uvSet) => uvSet > 1);
    if (highUvSets.length === 1 && highUvSets[0] === uvIndex && !uvSets.has(1)) {
      return true;
    }
  }
  return false;
}

function materialTextureUvSets(material: GltfMaterial): ReadonlySet<number> {
  const uvSets = new Set<number>();
  const visit = (value: unknown): void => {
    if (value == null || typeof value !== 'object' || Array.isArray(value)) return;
    const object = value as Record<string, unknown>;
    if (typeof object.index === 'number') {
      uvSets.add(textureInfoUvSet(object as unknown as GltfTextureInfo));
    }
    for (const child of Object.values(object)) visit(child);
  };
  visit(material);
  return uvSets;
}

function analyzeMaterials(
  gltf: GltfJson,
  sceneScope: GltfSceneReachability | undefined,
  selectedTextureSourceExtensions: ReadonlySet<string>,
): GltfMaterialFeatureReport {
  const materials = gltf.materials ?? [];
  const materialEntries = [...materials.entries()].filter(([materialIndex]) =>
    sceneScope === undefined || sceneScope.materialIndices.has(materialIndex)
  );
  const fields = new Set<keyof MaterialSpec>();
  const textureFields = new Set<keyof MaterialSpec>();
  const samplerPolicies: GltfTextureSamplerPolicyUse[] = [];
  const malformedSamplerPolicies: GltfMalformedTextureSamplerPolicyUse[] = [];
  const textureReferenceIssues: GltfMaterialTextureReferenceIssue[] = [];
  const primitiveMaterialReferenceIssues = collectPrimitiveMaterialReferenceIssues(gltf, sceneScope);
  const variantMappingIssues = materialVariantMappingIssues(gltf, sceneScope);
  const extensions = new Set<string>();
  const unsupportedKnownExtensions = new Set<string>();
  const alphaModes = new Set<string>();
  const uvSets = new Set<number>();
  const materialUvSets = new Map<number, Set<number>>();
  const issuePaths: SourcePathMap = new Map();
  let textureTransformCount = 0;
  let volumeThicknessTextureCount = 0;
  let specularGlossinessMaterialCount = 0;
  let specularGlossinessTextureCount = 0;
  let doubleSidedCount = 0;
  let currentMaterialIndex = -1;

  const addField = (field: keyof MaterialSpec, path: string): void => {
    fields.add(field);
    addSourcePath(issuePaths, `field:${String(field)}`, path);
  };
  const addTexture = (field: keyof MaterialSpec, info: GltfTextureInfo | undefined, path: string): void => {
    if (info == null) return;
    fields.add(field);
    textureFields.add(field);
    addSourcePath(issuePaths, `field:${String(field)}`, path);
    const samplerPolicy = textureSamplerPolicyUse(gltf, field, info, path);
    if (samplerPolicy !== null) samplerPolicies.push(samplerPolicy);
    malformedSamplerPolicies.push(...textureMalformedSamplerPolicyUses(gltf, field, info, path));
    textureReferenceIssues.push(...materialTextureReferenceIssues(
      gltf,
      field,
      info,
      path,
      selectedTextureSourceExtensions,
    ));
    const uvSet = textureInfoUvSet(info);
    uvSets.add(uvSet);
    if (currentMaterialIndex >= 0) {
      let perMaterial = materialUvSets.get(currentMaterialIndex);
      if (perMaterial === undefined) {
        perMaterial = new Set<number>();
        materialUvSets.set(currentMaterialIndex, perMaterial);
      }
      perMaterial.add(uvSet);
    }
    if (uvSet > 1) addSourcePath(issuePaths, `uvSet:${uvSet}`, textureInfoUvSetPath(info, path));
    if (info.extensions?.KHR_texture_transform) textureTransformCount += 1;
  };

  for (const [materialIndex, mat] of materialEntries) {
    currentMaterialIndex = materialIndex;
    const matPath = `materials[${materialIndex}]`;
    const pbr = mat.pbrMetallicRoughness;
    const effectiveAlphaMode = mat.alphaMode ?? 'OPAQUE';
    const usesMaterialAlpha = effectiveAlphaMode === 'MASK' || effectiveAlphaMode === 'BLEND';
    if (pbr?.baseColorFactor) {
      addField('baseColor', `${matPath}.pbrMetallicRoughness.baseColorFactor`);
      if (usesMaterialAlpha && (pbr.baseColorFactor[3] ?? 1) < 1) {
        addField('opacity', `${matPath}.pbrMetallicRoughness.baseColorFactor[3]`);
      }
    }
    if (pbr?.metallicFactor !== undefined) addField('metallic', `${matPath}.pbrMetallicRoughness.metallicFactor`);
    if (pbr?.roughnessFactor !== undefined) addField('roughness', `${matPath}.pbrMetallicRoughness.roughnessFactor`);
    addTexture('baseColorMap', pbr?.baseColorTexture, `${matPath}.pbrMetallicRoughness.baseColorTexture`);
    if (pbr?.metallicRoughnessTexture) {
      addTexture('roughnessMap', pbr.metallicRoughnessTexture, `${matPath}.pbrMetallicRoughness.metallicRoughnessTexture`);
      addTexture('metallicMap', pbr.metallicRoughnessTexture, `${matPath}.pbrMetallicRoughness.metallicRoughnessTexture`);
    }
    addTexture('normalMap', mat.normalTexture, `${matPath}.normalTexture`);
    if (mat.normalTexture?.scale !== undefined) addField('normalScale', `${matPath}.normalTexture.scale`);
    addTexture('aoMap', mat.occlusionTexture, `${matPath}.occlusionTexture`);
    if (mat.occlusionTexture?.strength !== undefined) addField('aoMapIntensity', `${matPath}.occlusionTexture.strength`);
    if (mat.emissiveFactor) addField('emissive', `${matPath}.emissiveFactor`);
    addTexture('emissiveMap', mat.emissiveTexture, `${matPath}.emissiveTexture`);
    if (mat.alphaMode !== undefined) {
      addField('alphaMode', `${matPath}.alphaMode`);
      alphaModes.add(mat.alphaMode);
    }
    if (mat.alphaCutoff !== undefined) addField('alphaCutoff', `${matPath}.alphaCutoff`);
    if (mat.doubleSided) {
      doubleSidedCount += 1;
      addSourcePath(issuePaths, 'doubleSided', `${matPath}.doubleSided`);
    }

    const ext = mat.extensions ?? {};
    for (const key of Object.keys(ext)) {
      extensions.add(key);
      addSourcePath(issuePaths, `extension:${key}`, `${matPath}.extensions.${key}`);
      if (COMMON_UNSUPPORTED_EXTENSIONS.has(key)) unsupportedKnownExtensions.add(key);
    }
    if (ext.KHR_materials_unlit) addField('shadingModel', `${matPath}.extensions.KHR_materials_unlit`);
    const transmission = ext.KHR_materials_transmission;
    if (transmission) {
      if (transmission.transmissionFactor !== undefined) {
        addField('transmission', `${matPath}.extensions.KHR_materials_transmission.transmissionFactor`);
      }
      addTexture('transmissionMap', transmission.transmissionTexture, `${matPath}.extensions.KHR_materials_transmission.transmissionTexture`);
    }
    const ior = ext.KHR_materials_ior;
    if (ior?.ior !== undefined) addField('ior', `${matPath}.extensions.KHR_materials_ior.ior`);
    const volume = ext.KHR_materials_volume;
    if (volume) {
      if (volume.thicknessFactor !== undefined) addField('thickness', `${matPath}.extensions.KHR_materials_volume.thicknessFactor`);
      addTexture('thicknessMap', volume.thicknessTexture, `${matPath}.extensions.KHR_materials_volume.thicknessTexture`);
      if (volume.attenuationDistance !== undefined) {
        addField('attenuationDistance', `${matPath}.extensions.KHR_materials_volume.attenuationDistance`);
      }
      if (volume.attenuationColor !== undefined) {
        addField('attenuationColor', `${matPath}.extensions.KHR_materials_volume.attenuationColor`);
      }
      if (volume.thicknessTexture) volumeThicknessTextureCount += 1;
    }
    const specular = ext.KHR_materials_specular;
    if (specular) {
      if (specular.specularFactor !== undefined) addField('specularIntensity', `${matPath}.extensions.KHR_materials_specular.specularFactor`);
      if (specular.specularColorFactor !== undefined) addField('specularColor', `${matPath}.extensions.KHR_materials_specular.specularColorFactor`);
      addTexture('specularIntensityMap', specular.specularTexture, `${matPath}.extensions.KHR_materials_specular.specularTexture`);
      addTexture('specularColorMap', specular.specularColorTexture, `${matPath}.extensions.KHR_materials_specular.specularColorTexture`);
    }
    const sheen = ext.KHR_materials_sheen;
    if (sheen) {
      addField('sheen', `${matPath}.extensions.KHR_materials_sheen`);
      if (sheen.sheenColorFactor !== undefined) addField('sheenColor', `${matPath}.extensions.KHR_materials_sheen.sheenColorFactor`);
      if (sheen.sheenRoughnessFactor !== undefined) addField('sheenRoughness', `${matPath}.extensions.KHR_materials_sheen.sheenRoughnessFactor`);
      addTexture('sheenColorMap', sheen.sheenColorTexture, `${matPath}.extensions.KHR_materials_sheen.sheenColorTexture`);
      addTexture('sheenRoughnessMap', sheen.sheenRoughnessTexture, `${matPath}.extensions.KHR_materials_sheen.sheenRoughnessTexture`);
    }
    const clearcoat = ext.KHR_materials_clearcoat;
    if (clearcoat) {
      if (clearcoat.clearcoatFactor !== undefined) addField('clearcoat', `${matPath}.extensions.KHR_materials_clearcoat.clearcoatFactor`);
      if (clearcoat.clearcoatRoughnessFactor !== undefined) addField('clearcoatRoughness', `${matPath}.extensions.KHR_materials_clearcoat.clearcoatRoughnessFactor`);
      addTexture('clearcoatMap', clearcoat.clearcoatTexture, `${matPath}.extensions.KHR_materials_clearcoat.clearcoatTexture`);
      addTexture('clearcoatRoughnessMap', clearcoat.clearcoatRoughnessTexture, `${matPath}.extensions.KHR_materials_clearcoat.clearcoatRoughnessTexture`);
      addTexture('clearcoatNormalMap', clearcoat.clearcoatNormalTexture, `${matPath}.extensions.KHR_materials_clearcoat.clearcoatNormalTexture`);
      if (clearcoat.clearcoatNormalTexture?.scale !== undefined) {
        addField('clearcoatNormalScale', `${matPath}.extensions.KHR_materials_clearcoat.clearcoatNormalTexture.scale`);
      }
    }
    const iridescence = ext.KHR_materials_iridescence;
    if (iridescence) {
      if (iridescence.iridescenceFactor !== undefined) addField('iridescence', `${matPath}.extensions.KHR_materials_iridescence.iridescenceFactor`);
      if (iridescence.iridescenceIor !== undefined) addField('iridescenceIor', `${matPath}.extensions.KHR_materials_iridescence.iridescenceIor`);
      if (
        iridescence.iridescenceThicknessMinimum !== undefined ||
        iridescence.iridescenceThicknessMaximum !== undefined
      ) {
        addField('iridescenceThicknessRange', `${matPath}.extensions.KHR_materials_iridescence.iridescenceThicknessMinimum`);
      }
      addTexture('iridescenceMap', iridescence.iridescenceTexture, `${matPath}.extensions.KHR_materials_iridescence.iridescenceTexture`);
      addTexture('iridescenceThicknessMap', iridescence.iridescenceThicknessTexture, `${matPath}.extensions.KHR_materials_iridescence.iridescenceThicknessTexture`);
    }
    const anisotropy = ext.KHR_materials_anisotropy;
    if (anisotropy) {
      if (anisotropy.anisotropyStrength !== undefined) addField('anisotropy', `${matPath}.extensions.KHR_materials_anisotropy.anisotropyStrength`);
      if (anisotropy.anisotropyRotation !== undefined) addField('anisotropyRotation', `${matPath}.extensions.KHR_materials_anisotropy.anisotropyRotation`);
      addTexture('anisotropyMap', anisotropy.anisotropyTexture, `${matPath}.extensions.KHR_materials_anisotropy.anisotropyTexture`);
    }
    const dispersion = ext.KHR_materials_dispersion;
    if (dispersion?.dispersion !== undefined && dispersion.dispersion > 0) {
      addField('dispersionAbbeNumber', `${matPath}.extensions.KHR_materials_dispersion.dispersion`);
    }
    const emissiveStrength = ext.KHR_materials_emissive_strength;
    if (emissiveStrength?.emissiveStrength !== undefined) {
      addField('emissiveIntensity', `${matPath}.extensions.KHR_materials_emissive_strength.emissiveStrength`);
    }
    const specGloss = ext.KHR_materials_pbrSpecularGlossiness;
    if (specGloss) {
      specularGlossinessMaterialCount += 1;
      const specGlossPath = `${matPath}.extensions.KHR_materials_pbrSpecularGlossiness`;
      addField('baseColor', specGloss.diffuseFactor !== undefined ? `${specGlossPath}.diffuseFactor` : specGlossPath);
      addField('roughness', specGloss.glossinessFactor !== undefined ? `${specGlossPath}.glossinessFactor` : specGlossPath);
      addField('metallic', specGlossPath);
      if (usesMaterialAlpha && specGloss.diffuseFactor?.[3] !== undefined && specGloss.diffuseFactor[3] < 1) {
        addField('opacity', `${specGlossPath}.diffuseFactor[3]`);
      }
      if (specGloss.specularFactor !== undefined) addField('specularColor', `${specGlossPath}.specularFactor`);
      addTexture('baseColorMap', specGloss.diffuseTexture, `${specGlossPath}.diffuseTexture`);
      addTexture('specularColorMap', specGloss.specularGlossinessTexture, `${specGlossPath}.specularGlossinessTexture`);
      if (specGloss.specularGlossinessTexture) {
        specularGlossinessTextureCount += 1;
        addSourcePath(issuePaths, 'specGlossGlossinessAlpha', `${specGlossPath}.specularGlossinessTexture`);
      }
    }
  }

  return {
    count: materialEntries.length,
    materialFields: sorted(fields) as (keyof MaterialSpec)[],
    textureFields: sorted(textureFields) as (keyof MaterialSpec)[],
    samplerPolicies: samplerPolicies.sort((a, b) =>
      a.path.localeCompare(b.path) || String(a.materialField).localeCompare(String(b.materialField)),
    ),
    malformedSamplerPolicies: malformedSamplerPolicies.sort((a, b) =>
      a.path.localeCompare(b.path) || String(a.materialField).localeCompare(String(b.materialField)),
    ),
    textureReferenceIssues: textureReferenceIssues.sort((a, b) =>
      a.path.localeCompare(b.path) || String(a.materialField).localeCompare(String(b.materialField)),
    ),
    primitiveMaterialReferenceIssues: primitiveMaterialReferenceIssues.sort((a, b) =>
      a.path.localeCompare(b.path),
    ),
    variantMappingIssues: variantMappingIssues.sort((a, b) => a.path.localeCompare(b.path)),
    extensions: sorted(extensions),
    unsupportedKnownExtensions: sorted(unsupportedKnownExtensions),
    alphaModes: sorted(alphaModes),
    uvSets: [...uvSets].sort((a, b) => a - b),
    unrepresentableUvSets: analyzeUnrepresentableMaterialUvSets(gltf, materialUvSets, sceneScope),
    textureTransformCount,
    volumeThicknessTextureCount,
    specularGlossinessMaterialCount,
    specularGlossinessTextureCount,
    doubleSidedCount,
    issuePaths: sourcePathRecord(issuePaths),
  };
}

function collectPrimitiveMaterialReferenceIssues(
  gltf: GltfJson,
  sceneScope: GltfSceneReachability | undefined,
): GltfPrimitiveMaterialReferenceIssue[] {
  const issues: GltfPrimitiveMaterialReferenceIssue[] = [];
  const materialCount = gltf.materials?.length ?? 0;
  for (const [meshIndex, mesh] of (gltf.meshes ?? []).entries()) {
    for (const [primitiveIndex, primitive] of mesh.primitives.entries()) {
      if (sceneScope !== undefined && !sceneScope.primitiveKeys.has(gltfPrimitiveKey(meshIndex, primitiveIndex))) continue;
      const materialIndex = primitive.material;
      if (materialIndex === undefined) continue;
      if (!Number.isInteger(materialIndex) || materialIndex < 0 || materialIndex >= materialCount) {
        issues.push({
          kind: 'missing-material',
          path: `meshes[${meshIndex}].primitives[${primitiveIndex}].material`,
          meshIndex,
          primitiveIndex,
          materialIndex,
        });
      }
    }
  }
  return issues;
}

function materialVariantMappingIssues(
  gltf: GltfJson,
  sceneScope: GltfSceneReachability | undefined,
): GltfMaterialVariantMappingIssue[] {
  const issues: GltfMaterialVariantMappingIssue[] = [];
  const materialCount = gltf.materials?.length ?? 0;
  const rootVariants = gltf.extensions?.KHR_materials_variants?.variants;
  const rootVariantListMalformed = rootVariants !== undefined && !Array.isArray(rootVariants);
  const variantCount = Array.isArray(rootVariants) ? rootVariants.length : 0;

  if (rootVariantListMalformed) {
    issues.push({
      kind: 'malformed-root-variant-list',
      path: 'extensions.KHR_materials_variants.variants',
    });
  }

  for (const [meshIndex, mesh] of (gltf.meshes ?? []).entries()) {
    for (const [primitiveIndex, primitive] of mesh.primitives.entries()) {
      if (sceneScope !== undefined && !sceneScope.primitiveKeys.has(gltfPrimitiveKey(meshIndex, primitiveIndex))) continue;
      const mappings = primitive.extensions?.KHR_materials_variants?.mappings ?? [];
      for (const [mappingIndex, mapping] of mappings.entries()) {
        const mappingPath = `meshes[${meshIndex}].primitives[${primitiveIndex}].extensions.KHR_materials_variants.mappings[${mappingIndex}]`;
        if (!Number.isInteger(mapping.material) || mapping.material < 0 || mapping.material >= materialCount) {
          issues.push({
            kind: 'missing-material',
            path: `${mappingPath}.material`,
            meshIndex,
            primitiveIndex,
            mappingIndex,
            materialIndex: mapping.material,
          });
        }
        if (!Array.isArray(mapping.variants)) {
          issues.push({
            kind: 'missing-variant-list',
            path: `${mappingPath}.variants`,
            meshIndex,
            primitiveIndex,
            mappingIndex,
          });
          continue;
        }
        if (rootVariantListMalformed) continue;
        for (const [variantSlot, variantIndex] of mapping.variants.entries()) {
          if (!Number.isInteger(variantIndex) || variantIndex < 0 || variantIndex >= variantCount) {
            issues.push({
              kind: 'missing-variant',
              path: `${mappingPath}.variants[${variantSlot}]`,
              meshIndex,
              primitiveIndex,
              mappingIndex,
              variantIndex,
            });
          }
        }
      }
    }
  }

  return issues;
}

function textureSamplerPolicyUse(
  gltf: GltfJson,
  materialField: keyof MaterialSpec,
  info: GltfTextureInfo,
  materialPath: string,
): GltfTextureSamplerPolicyUse | null {
  const texture = gltf.textures?.[info.index];
  const samplerIndex = texture?.sampler;
  if (samplerIndex === undefined) return null;
  const sampler = gltf.samplers?.[samplerIndex];
  if (sampler == null) return null;
  const magFilter = textureMagFilterMode(sampler.magFilter);
  const minFilter = textureMinFilterModes(sampler.minFilter);
  if (magFilter === undefined && minFilter === null) return null;
  const path = minFilter !== null
    ? `samplers[${samplerIndex}].minFilter`
    : `samplers[${samplerIndex}].magFilter`;
  return {
    materialField,
    textureIndex: info.index,
    samplerIndex,
    materialPath,
    path,
    ...(magFilter !== undefined ? { magFilter } : {}),
    ...(minFilter !== null ? {
      minFilter: minFilter.minFilter,
      mipFilter: minFilter.mipFilter,
      usesMipmaps: minFilter.mipFilter !== 'none',
    } : {}),
  };
}

function textureMalformedSamplerPolicyUses(
  gltf: GltfJson,
  materialField: keyof MaterialSpec,
  info: GltfTextureInfo,
  materialPath: string,
): GltfMalformedTextureSamplerPolicyUse[] {
  const texture = gltf.textures?.[info.index];
  const samplerIndex = texture?.sampler;
  if (samplerIndex === undefined) return [];
  const sampler = gltf.samplers?.[samplerIndex];
  if (sampler == null) {
    return [{
      kind: 'missing-sampler',
      materialField,
      textureIndex: info.index,
      samplerIndex,
      materialPath,
      path: `textures[${info.index}].sampler`,
    }];
  }

  const issues: GltfMalformedTextureSamplerPolicyUse[] = [];
  if (sampler.wrapS !== undefined && !isGltfWrapMode(sampler.wrapS)) {
    issues.push({
      kind: 'invalid-wrap-s',
      materialField,
      textureIndex: info.index,
      samplerIndex,
      materialPath,
      path: `samplers[${samplerIndex}].wrapS`,
      value: sampler.wrapS,
    });
  }
  if (sampler.wrapT !== undefined && !isGltfWrapMode(sampler.wrapT)) {
    issues.push({
      kind: 'invalid-wrap-t',
      materialField,
      textureIndex: info.index,
      samplerIndex,
      materialPath,
      path: `samplers[${samplerIndex}].wrapT`,
      value: sampler.wrapT,
    });
  }
  if (sampler.magFilter !== undefined && textureMagFilterMode(sampler.magFilter) === undefined) {
    issues.push({
      kind: 'invalid-mag-filter',
      materialField,
      textureIndex: info.index,
      samplerIndex,
      materialPath,
      path: `samplers[${samplerIndex}].magFilter`,
      value: sampler.magFilter,
    });
  }
  if (sampler.minFilter !== undefined && textureMinFilterModes(sampler.minFilter) === null) {
    issues.push({
      kind: 'invalid-min-filter',
      materialField,
      textureIndex: info.index,
      samplerIndex,
      materialPath,
      path: `samplers[${samplerIndex}].minFilter`,
      value: sampler.minFilter,
    });
  }
  return issues;
}

function materialTextureReferenceIssues(
  gltf: GltfJson,
  materialField: keyof MaterialSpec,
  info: GltfTextureInfo,
  materialPath: string,
  selectedTextureSourceExtensions: ReadonlySet<string>,
): GltfMaterialTextureReferenceIssue[] {
  const texture = gltf.textures?.[info.index];
  if (texture == null) {
    return [{
      kind: 'missing-texture',
      materialField,
      textureIndex: info.index,
      materialPath,
      path: `${materialPath}.index`,
    }];
  }

  const selectedSource = selectMaterialTextureImageSourceForReport(texture, info.index, selectedTextureSourceExtensions);
  if (selectedSource === undefined) {
    const availableExtensions = availableTextureSourceExtensions(texture);
    if (availableExtensions.length > 0) {
      return [{
        kind: 'disabled-texture-source-extension',
        materialField,
        textureIndex: info.index,
        materialPath,
        path: availableExtensions.length === 1
          ? `textures[${info.index}].extensions.${availableExtensions[0]}`
          : `textures[${info.index}].extensions`,
        textureSourceExtensions: availableExtensions,
      }];
    }
    return [{
      kind: 'missing-texture-source',
      materialField,
      textureIndex: info.index,
      materialPath,
      path: `textures[${info.index}].source`,
    }];
  }

  const image = gltf.images?.[selectedSource.imageIndex];
  if (image == null) {
    return [{
      kind: 'missing-image',
      materialField,
      textureIndex: info.index,
      materialPath,
      path: selectedSource.path,
      imageIndex: selectedSource.imageIndex,
    }];
  }
  if (image.bufferView !== undefined && gltf.bufferViews?.[image.bufferView] == null) {
    return [{
      kind: 'missing-image-buffer-view',
      materialField,
      textureIndex: info.index,
      materialPath,
      path: `images[${selectedSource.imageIndex}].bufferView`,
      imageIndex: selectedSource.imageIndex,
      bufferViewIndex: image.bufferView,
    }];
  }
  if (image.bufferView !== undefined) {
    const bufferView = gltf.bufferViews?.[image.bufferView];
    if (bufferView !== undefined && gltf.buffers?.[bufferView.buffer] == null) {
      return [{
        kind: 'image-buffer-unavailable',
        materialField,
        textureIndex: info.index,
        materialPath,
        path: `bufferViews[${image.bufferView}].buffer`,
        imageIndex: selectedSource.imageIndex,
        bufferViewIndex: image.bufferView,
        bufferIndex: bufferView.buffer,
      }];
    }
  }
  if (image.uri === undefined && image.bufferView === undefined) {
    return [{
      kind: 'missing-image-source',
      materialField,
      textureIndex: info.index,
      materialPath,
      path: `images[${selectedSource.imageIndex}]`,
      imageIndex: selectedSource.imageIndex,
    }];
  }
  return [];
}

function selectMaterialTextureImageSourceForReport(
  texture: NonNullable<GltfJson['textures']>[number],
  textureIndex: number,
  selectedTextureSourceExtensions: ReadonlySet<string>,
): { readonly imageIndex: number; readonly path: string } | undefined {
  for (const extension of TEXTURE_SOURCE_EXTENSION_NAMES) {
    if (!selectedTextureSourceExtensions.has(extension)) continue;
    const source = texture.extensions?.[extension]?.source;
    if (source !== undefined) {
      return {
        imageIndex: source,
        path: `textures[${textureIndex}].extensions.${extension}.source`,
      };
    }
  }
  return texture.source !== undefined
    ? { imageIndex: texture.source, path: `textures[${textureIndex}].source` }
    : undefined;
}

function availableTextureSourceExtensions(
  texture: NonNullable<GltfJson['textures']>[number],
): readonly GltfTextureSourceExtensionName[] {
  return TEXTURE_SOURCE_EXTENSION_NAMES.filter((extension) =>
    texture.extensions?.[extension]?.source !== undefined
  );
}

function isGltfWrapMode(value: number): boolean {
  return value === 33071 || value === 33648 || value === 10497;
}

function textureMagFilterMode(value: number | undefined): GltfTextureSamplerFilterMode | undefined {
  if (value === 9728) return 'nearest';
  if (value === 9729) return 'linear';
  return undefined;
}

function textureMinFilterModes(value: number | undefined): {
  readonly minFilter: GltfTextureSamplerFilterMode;
  readonly mipFilter: GltfTextureSamplerMipMode;
} | null {
  switch (value) {
    case 9728:
      return { minFilter: 'nearest', mipFilter: 'none' };
    case 9729:
      return { minFilter: 'linear', mipFilter: 'none' };
    case 9984:
      return { minFilter: 'nearest', mipFilter: 'nearest' };
    case 9985:
      return { minFilter: 'linear', mipFilter: 'nearest' };
    case 9986:
      return { minFilter: 'nearest', mipFilter: 'linear' };
    case 9987:
      return { minFilter: 'linear', mipFilter: 'linear' };
    default:
      return null;
  }
}

function analyzeAnimations(
  gltf: GltfJson,
  sceneScope: GltfSceneReachability | undefined,
): GltfAnimationFeatureReport {
  const paths = new Set<string>();
  const unsupportedTargetPaths = new Set<string>();
  const interpolations = new Set<string>();
  const degradedInterpolations = new Set<string>();
  const malformedChannels: GltfAnimationMalformedChannelIssue[] = [];
  const targetNodes = new Set<number>();
  const animationIndices = new Set<number>();
  const issuePaths: SourcePathMap = new Map();
  let channelCount = 0;
  for (const [animationIndex, animation] of (gltf.animations ?? []).entries()) {
    let animationHasReachableChannel = sceneScope === undefined;
    for (const [channelIndex, channel] of (animation.channels ?? []).entries()) {
      if (
        sceneScope !== undefined &&
        (channel.target.node === undefined || !sceneScope.nodeIndices.has(channel.target.node))
      ) {
        continue;
      }
      animationHasReachableChannel = true;
      channelCount += 1;
      const targetPath = channel.target.path;
      paths.add(targetPath);
      if (!CORE_ANIMATION_TARGET_PATHS.has(targetPath)) {
        unsupportedTargetPaths.add(targetPath);
        addSourcePath(
          issuePaths,
          `unsupportedTargetPath:${targetPath}`,
          `animations[${animationIndex}].channels[${channelIndex}].target.path`,
        );
        if (channel.target.node !== undefined) targetNodes.add(channel.target.node);
        continue;
      }
      if (channel.target.node === undefined) {
        malformedChannels.push({
          kind: 'missing-target-node',
          path: `animations[${animationIndex}].channels[${channelIndex}].target.node`,
          animationIndex,
          channelIndex,
          targetPath,
        });
        continue;
      }
      if (!gltf.nodes?.[channel.target.node]) {
        malformedChannels.push({
          kind: 'target-node-not-found',
          path: `animations[${animationIndex}].channels[${channelIndex}].target.node`,
          animationIndex,
          channelIndex,
          targetPath,
          nodeIndex: channel.target.node,
        });
        continue;
      }
      const sampler = animation.samplers?.[channel.sampler];
      if (sampler == null) {
        malformedChannels.push({
          kind: 'missing-sampler',
          path: `animations[${animationIndex}].samplers[${channel.sampler}]`,
          animationIndex,
          channelIndex,
          targetPath,
          samplerIndex: channel.sampler,
          nodeIndex: channel.target.node,
        });
        continue;
      }
      const samplerAccessorIssues = animationSamplerAccessorIssues(
        gltf,
        animationIndex,
        channelIndex,
        channel,
        sampler,
      );
      if (samplerAccessorIssues.length > 0) {
        malformedChannels.push(...samplerAccessorIssues);
        continue;
      }
      const outputCountIssue = animationOutputCountIssue(
        gltf,
        animationIndex,
        channelIndex,
        channel,
        sampler,
      );
      if (outputCountIssue !== undefined) malformedChannels.push(outputCountIssue);
      targetNodes.add(channel.target.node);
    }
    if (!animationHasReachableChannel) continue;
    animationIndices.add(animationIndex);
    for (const [samplerIndex, sampler] of (animation.samplers ?? []).entries()) {
      const interpolation = sampler.interpolation ?? 'LINEAR';
      interpolations.add(interpolation);
      if (!CORE_ANIMATION_INTERPOLATIONS.has(interpolation)) {
        degradedInterpolations.add(interpolation);
        addSourcePath(
          issuePaths,
          `degradedInterpolation:${interpolation}`,
          `animations[${animationIndex}].samplers[${samplerIndex}].interpolation`,
        );
      }
    }
  }
  return {
    count: sceneScope === undefined ? gltf.animations?.length ?? 0 : animationIndices.size,
    channelCount,
    paths: sorted(paths),
    unsupportedTargetPaths: sorted(unsupportedTargetPaths),
    interpolations: sorted(interpolations),
    degradedInterpolations: sorted(degradedInterpolations),
    malformedChannels: malformedChannels.sort((a, b) =>
      a.path.localeCompare(b.path) || a.kind.localeCompare(b.kind)
    ),
    targetNodeCount: targetNodes.size,
    issuePaths: sourcePathRecord(issuePaths),
  };
}

function animationSamplerAccessorIssues(
  gltf: GltfJson,
  animationIndex: number,
  channelIndex: number,
  channel: GltfAnimationChannel,
  sampler: GltfAnimationSampler,
): GltfAnimationMalformedChannelIssue[] {
  const issues: GltfAnimationMalformedChannelIssue[] = [];
  const addIssue = (
    kind: Extract<
      GltfAnimationMalformedChannelKind,
      | 'missing-input-accessor'
      | 'missing-output-accessor'
      | 'invalid-input-accessor-type'
      | 'invalid-output-accessor-type'
      | 'invalid-input-accessor-component-type'
      | 'invalid-output-accessor-component-type'
      | 'missing-input-buffer-view'
      | 'missing-output-buffer-view'
      | 'missing-input-buffer'
      | 'missing-output-buffer'
    >,
    accessorRole: 'input' | 'output',
    accessorIndex: number,
    extra: {
      readonly accessorType?: string;
      readonly componentType?: number;
      readonly bufferViewIndex?: number;
      readonly bufferIndex?: number;
      readonly path?: string;
    } = {},
  ): void => {
    issues.push({
      kind,
      path: extra.path ?? `animations[${animationIndex}].samplers[${channel.sampler}].${accessorRole}`,
      animationIndex,
      channelIndex,
      targetPath: channel.target.path,
      samplerIndex: channel.sampler,
      ...(channel.target.node !== undefined ? { nodeIndex: channel.target.node } : {}),
      accessorIndex,
      accessorRole,
      ...(extra.accessorType !== undefined ? { accessorType: extra.accessorType } : {}),
      ...(extra.componentType !== undefined ? { componentType: extra.componentType } : {}),
      ...(extra.bufferViewIndex !== undefined ? { bufferViewIndex: extra.bufferViewIndex } : {}),
      ...(extra.bufferIndex !== undefined ? { bufferIndex: extra.bufferIndex } : {}),
    });
  };

  const inputAccessor = gltf.accessors?.[sampler.input];
  if (inputAccessor == null) {
    addIssue('missing-input-accessor', 'input', sampler.input);
  } else if (animationAccessorFloatCount(inputAccessor) === undefined) {
    addIssue('invalid-input-accessor-type', 'input', sampler.input, { accessorType: String(inputAccessor.type) });
  } else if (!floatAccessorComponentTypeIsReadableByImporter(inputAccessor.componentType)) {
    addIssue('invalid-input-accessor-component-type', 'input', sampler.input, {
      componentType: inputAccessor.componentType,
      path: `accessors[${sampler.input}].componentType`,
    });
  } else {
    addAnimationAccessorStorageIssues(gltf, issues, {
      animationIndex,
      channelIndex,
      channel,
      sampler,
      accessorRole: 'input',
      accessorIndex: sampler.input,
      accessor: inputAccessor,
    });
  }

  const outputAccessor = gltf.accessors?.[sampler.output];
  if (outputAccessor == null) {
    addIssue('missing-output-accessor', 'output', sampler.output);
  } else if (animationAccessorFloatCount(outputAccessor) === undefined) {
    addIssue('invalid-output-accessor-type', 'output', sampler.output, { accessorType: String(outputAccessor.type) });
  } else if (!floatAccessorComponentTypeIsReadableByImporter(outputAccessor.componentType)) {
    addIssue('invalid-output-accessor-component-type', 'output', sampler.output, {
      componentType: outputAccessor.componentType,
      path: `accessors[${sampler.output}].componentType`,
    });
  } else {
    addAnimationAccessorStorageIssues(gltf, issues, {
      animationIndex,
      channelIndex,
      channel,
      sampler,
      accessorRole: 'output',
      accessorIndex: sampler.output,
      accessor: outputAccessor,
    });
  }

  return issues;
}

function addAnimationAccessorStorageIssues(
  gltf: GltfJson,
  issues: GltfAnimationMalformedChannelIssue[],
  input: {
    readonly animationIndex: number;
    readonly channelIndex: number;
    readonly channel: GltfAnimationChannel;
    readonly sampler: GltfAnimationSampler;
    readonly accessorRole: 'input' | 'output';
    readonly accessorIndex: number;
    readonly accessor: NonNullable<GltfJson['accessors']>[number];
  },
): void {
  const { animationIndex, channelIndex, channel, sampler, accessorRole, accessorIndex, accessor } = input;
  const kindPrefix = accessorRole === 'input' ? 'input' : 'output';
  if (accessor.bufferView !== undefined) {
    const bufferViewIndex = accessor.bufferView;
    const bufferView = gltf.bufferViews?.[bufferViewIndex];
    if (bufferView == null) {
      issues.push({
        kind: kindPrefix === 'input' ? 'missing-input-buffer-view' : 'missing-output-buffer-view',
        path: `accessors[${accessorIndex}].bufferView`,
        animationIndex,
        channelIndex,
        targetPath: channel.target.path,
        samplerIndex: channel.sampler,
        ...(channel.target.node !== undefined ? { nodeIndex: channel.target.node } : {}),
        accessorIndex,
        accessorRole,
        bufferViewIndex,
      });
      return;
    }
    if (gltf.buffers?.[bufferView.buffer] == null) {
      issues.push({
        kind: kindPrefix === 'input' ? 'missing-input-buffer' : 'missing-output-buffer',
        path: `bufferViews[${bufferViewIndex}].buffer`,
        animationIndex,
        channelIndex,
        targetPath: channel.target.path,
        samplerIndex: channel.sampler,
        ...(channel.target.node !== undefined ? { nodeIndex: channel.target.node } : {}),
        accessorIndex,
        accessorRole,
        bufferViewIndex,
        bufferIndex: bufferView.buffer,
      });
      return;
    }
  }

  const sparseIssue = sparseAccessorStorageIssue(gltf, accessorIndex, accessor);
  if (sparseIssue !== undefined) {
    issues.push({
      kind: kindPrefix === 'input' ? 'invalid-input-sparse-accessor' : 'invalid-output-sparse-accessor',
      path: sparseIssue.path,
      animationIndex,
      channelIndex,
      targetPath: channel.target.path,
      samplerIndex: channel.sampler,
      ...(channel.target.node !== undefined ? { nodeIndex: channel.target.node } : {}),
      accessorIndex,
      accessorRole,
      ...sparseIssueDetails(sparseIssue),
    });
  }
}

function animationOutputCountIssue(
  gltf: GltfJson,
  animationIndex: number,
  channelIndex: number,
  channel: GltfAnimationChannel,
  sampler: GltfAnimationSampler,
): GltfAnimationMalformedChannelIssue | undefined {
  const inputAccessor = gltf.accessors?.[sampler.input];
  const outputAccessor = gltf.accessors?.[sampler.output];
  if (inputAccessor == null || outputAccessor == null) return undefined;
  const cubicFactor = sampler.interpolation === 'CUBICSPLINE' ? 3 : 1;
  const actualOutputFloats = animationAccessorFloatCount(outputAccessor);
  if (actualOutputFloats === undefined) return undefined;
  const targetPath = channel.target.path;
  const trsCount = animationTargetTrsComponentCount(targetPath);
  if (trsCount !== undefined) {
    const expectedOutputFloats = inputAccessor.count * trsCount * cubicFactor;
    if (actualOutputFloats === expectedOutputFloats) return undefined;
    return {
      kind: 'invalid-output-count',
      path: `animations[${animationIndex}].channels[${channelIndex}].sampler`,
      animationIndex,
      channelIndex,
      targetPath,
      samplerIndex: channel.sampler,
      ...(channel.target.node !== undefined ? { nodeIndex: channel.target.node } : {}),
      expectedOutputFloats,
      actualOutputFloats,
    };
  }
  const expectedStride = inputAccessor.count * cubicFactor;
  if (expectedStride <= 0 || actualOutputFloats % expectedStride === 0) return undefined;
  return {
    kind: 'invalid-output-count',
    path: `animations[${animationIndex}].channels[${channelIndex}].sampler`,
    animationIndex,
    channelIndex,
    targetPath,
    samplerIndex: channel.sampler,
    ...(channel.target.node !== undefined ? { nodeIndex: channel.target.node } : {}),
    expectedOutputFloats: expectedStride,
    actualOutputFloats,
  };
}

function animationTargetTrsComponentCount(path: string): number | undefined {
  if (path === 'rotation') return 4;
  if (path === 'translation' || path === 'scale') return 3;
  return undefined;
}

function animationAccessorFloatCount(accessor: NonNullable<GltfJson['accessors']>[number]): number | undefined {
  try {
    const componentCount = accessorComponentCount(accessor.type);
    return componentCount === undefined ? undefined : accessor.count * componentCount;
  } catch {
    return undefined;
  }
}

function accessorComponentCount(type: string): number | undefined {
  try {
    return typeComponentCount(type);
  } catch {
    return undefined;
  }
}

function textureInfoUvSet(info: GltfTextureInfo): number {
  return info.extensions?.KHR_texture_transform?.texCoord ?? info.texCoord ?? 0;
}

function textureInfoUvSetPath(info: GltfTextureInfo, path: string): string {
  return info.extensions?.KHR_texture_transform?.texCoord !== undefined
    ? `${path}.extensions.KHR_texture_transform.texCoord`
    : `${path}.texCoord`;
}

function textureEntriesForScope(
  gltf: GltfJson,
  sceneScope: GltfSceneReachability | undefined,
): Array<readonly [number, NonNullable<GltfJson['textures']>[number]]> {
  const textures = gltf.textures ?? [];
  if (sceneScope === undefined) return [...textures.entries()];
  const textureIndices = new Set<number>();
  for (const materialIndex of sceneScope.materialIndices) {
    const material = gltf.materials?.[materialIndex];
    if (material == null) continue;
    for (const textureIndex of materialTextureIndices(material)) {
      textureIndices.add(textureIndex);
    }
  }
  return [...textureIndices]
    .sort((a, b) => a - b)
    .map((textureIndex) => [textureIndex, textures[textureIndex]] as const)
    .filter((entry): entry is readonly [number, NonNullable<GltfJson['textures']>[number]] => entry[1] !== undefined);
}

function materialTextureIndices(material: GltfMaterial): readonly number[] {
  const indices = new Set<number>();
  const visit = (value: unknown): void => {
    if (value == null || typeof value !== 'object' || Array.isArray(value)) return;
    const object = value as Record<string, unknown>;
    if (typeof object.index === 'number') indices.add(object.index);
    for (const child of Object.values(object)) visit(child);
  };
  visit(material);
  return [...indices].sort((a, b) => a - b);
}

function bufferViewsForScope(
  gltf: GltfJson,
  sceneScope: GltfSceneReachability | undefined,
): readonly NonNullable<GltfJson['bufferViews']>[number][] {
  const bufferViews = gltf.bufferViews ?? [];
  if (sceneScope === undefined) return bufferViews;
  return [...sceneScope.bufferViewIndices]
    .sort((a, b) => a - b)
    .map((index) => bufferViews[index])
    .filter((bufferView): bufferView is NonNullable<GltfJson['bufferViews']>[number] => bufferView !== undefined);
}

function collectScopedNestedExtensionNames(
  gltf: GltfJson,
  sceneScope: GltfSceneReachability,
  out: Set<string>,
  sourcePaths: SourcePathMap,
): void {
  for (const nodeIndex of sceneScope.nodeIndices) {
    const node = gltf.nodes?.[nodeIndex];
    if (node !== undefined) collectNestedExtensionNames(node, out, sourcePaths, `nodes[${nodeIndex}]`);
  }
  for (const [meshIndex, mesh] of (gltf.meshes ?? []).entries()) {
    for (const [primitiveIndex, primitive] of mesh.primitives.entries()) {
      if (!sceneScope.primitiveKeys.has(gltfPrimitiveKey(meshIndex, primitiveIndex))) continue;
      collectNestedExtensionNames(
        primitive,
        out,
        sourcePaths,
        `meshes[${meshIndex}].primitives[${primitiveIndex}]`,
      );
    }
  }
  for (const materialIndex of sceneScope.materialIndices) {
    const material = gltf.materials?.[materialIndex];
    if (material !== undefined) collectNestedExtensionNames(material, out, sourcePaths, `materials[${materialIndex}]`);
  }
  for (const [textureIndex, texture] of textureEntriesForScope(gltf, sceneScope)) {
    collectNestedExtensionNames(texture, out, sourcePaths, `textures[${textureIndex}]`);
  }
  for (const bufferViewIndex of sceneScope.bufferViewIndices) {
    const bufferView = gltf.bufferViews?.[bufferViewIndex];
    if (bufferView !== undefined) collectNestedExtensionNames(bufferView, out, sourcePaths, `bufferViews[${bufferViewIndex}]`);
  }
  if (sceneUsesMeshQuantization(gltf, sceneScope)) {
    out.add('KHR_mesh_quantization');
  }
  if (sceneScope.punctualLightIndices.size > 0) {
    out.add('KHR_lights_punctual');
    addSourcePath(sourcePaths, 'KHR_lights_punctual', 'extensions.KHR_lights_punctual');
  }
  if (out.has('KHR_materials_variants')) {
    addSourcePath(sourcePaths, 'KHR_materials_variants', 'extensions.KHR_materials_variants');
  }
}

function sceneUsesMeshQuantization(gltf: GltfJson, sceneScope: GltfSceneReachability): boolean {
  const declared = (gltf.extensionsUsed ?? []).includes('KHR_mesh_quantization') ||
    (gltf.extensionsRequired ?? []).includes('KHR_mesh_quantization');
  if (!declared) return false;

  for (const [meshIndex, mesh] of (gltf.meshes ?? []).entries()) {
    for (const [primitiveIndex, primitive] of mesh.primitives.entries()) {
      if (!sceneScope.primitiveKeys.has(gltfPrimitiveKey(meshIndex, primitiveIndex))) continue;
      if (primitiveAttributesUseQuantizedAccessors(gltf, primitive)) return true;
    }
  }

  return false;
}

function primitiveAttributesUseQuantizedAccessors(gltf: GltfJson, primitive: GltfPrimitive): boolean {
  const usesQuantizedAccessor = (accessorIndex: number | undefined): boolean => {
    if (accessorIndex === undefined) return false;
    const componentType = gltf.accessors?.[accessorIndex]?.componentType;
    return componentType !== undefined && componentType !== 5126;
  };

  for (const accessorIndex of Object.values(primitive.attributes ?? {})) {
    if (usesQuantizedAccessor(accessorIndex)) return true;
  }
  for (const target of primitive.targets ?? []) {
    for (const accessorIndex of Object.values(target)) {
      if (usesQuantizedAccessor(accessorIndex)) return true;
    }
  }
  return false;
}

function collectNestedExtensionNames(
  value: unknown,
  out: Set<string>,
  sourcePaths: SourcePathMap,
  path = '',
): void {
  if (value == null || typeof value !== 'object') return;
  if (Array.isArray(value)) {
    value.forEach((item, index) => collectNestedExtensionNames(item, out, sourcePaths, `${path}[${index}]`));
    return;
  }
  const obj = value as Record<string, unknown>;
  const ext = obj['extensions'];
  if (ext && typeof ext === 'object' && !Array.isArray(ext)) {
    for (const key of Object.keys(ext as Record<string, unknown>)) {
      out.add(key);
      addSourcePath(sourcePaths, key, path === '' ? `extensions.${key}` : `${path}.extensions.${key}`);
    }
  }
  for (const [key, nested] of Object.entries(obj)) {
    collectNestedExtensionNames(nested, out, sourcePaths, path === '' ? key : `${path}.${key}`);
  }
}

function extractPunctualLightCount(gltf: GltfJson): number {
  const ext = gltf.extensions?.['KHR_lights_punctual'];
  if (ext && typeof ext === 'object' && !Array.isArray(ext)) {
    const lights = (ext as { lights?: unknown }).lights;
    return Array.isArray(lights) ? lights.length : 0;
  }
  return 0;
}

function analyzePunctualLightIssues(
  gltf: GltfJson,
  sceneScope: GltfSceneReachability | undefined,
): readonly GltfPunctualLightIssue[] {
  const ext = gltf.extensions?.['KHR_lights_punctual'];
  const lights = ext && typeof ext === 'object' && !Array.isArray(ext)
    ? (ext as { lights?: unknown }).lights
    : undefined;
  const lightEntries = Array.isArray(lights) ? lights : [];
  const issues: GltfPunctualLightIssue[] = [];

  for (const [nodeIndex, node] of (gltf.nodes ?? []).entries()) {
    if (sceneScope !== undefined && !sceneScope.nodeIndices.has(nodeIndex)) continue;
    const lightRef = node.extensions?.KHR_lights_punctual;
    if (
      lightRef == null ||
      typeof lightRef !== 'object' ||
      Array.isArray(lightRef) ||
      typeof (lightRef as { readonly light?: unknown }).light !== 'number'
    ) {
      continue;
    }
    const lightIndex = (lightRef as { readonly light: number }).light;
    if (lightEntries[lightIndex] === undefined) {
      issues.push({
        kind: 'missing-light',
        path: `nodes[${nodeIndex}].extensions.KHR_lights_punctual.light`,
        nodeIndex,
        lightIndex,
      });
    }
  }

  const scopedLightIndices = sceneScope?.punctualLightIndices;
  for (const [lightIndex, light] of lightEntries.entries()) {
    if (scopedLightIndices !== undefined && !scopedLightIndices.has(lightIndex)) continue;
    const type = light != null &&
      typeof light === 'object' &&
      !Array.isArray(light) &&
      typeof (light as { readonly type?: unknown }).type === 'string'
      ? (light as { readonly type: string }).type
      : undefined;
    if (type === 'point' || type === 'spot' || type === 'directional') continue;
    issues.push({
      kind: 'unsupported-light-type',
      path: `extensions.KHR_lights_punctual.lights[${lightIndex}].type`,
      lightIndex,
      ...(type !== undefined ? { lightType: type } : {}),
    });
  }

  return issues.sort((a, b) => a.path.localeCompare(b.path) || a.kind.localeCompare(b.kind));
}

function sorted<T extends string>(values: Iterable<T>): T[] {
  return [...new Set(values)].sort();
}
