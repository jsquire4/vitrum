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
import type {
  GltfImage,
  GltfJson,
  GltfMaterial,
  GltfTextureInfo,
} from './gltfTypes.js';

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
  readonly sourcePaths: Readonly<Record<string, readonly string[]>>;
}

export interface GltfPrimitiveFeatureReport {
  readonly total: number;
  readonly byMode: Readonly<Record<string, number>>;
  readonly unsupportedModes: readonly string[];
  readonly attributeSemantics: readonly string[];
  readonly expectedPrimitiveKinds: readonly ('mesh' | 'skinned-mesh')[];
  readonly usesDraco: boolean;
  readonly usesMeshopt: boolean;
  readonly hasTangents: boolean;
  readonly hasMorphTargets: boolean;
  readonly hasMorphTargetTangents: boolean;
  readonly hasSkins: boolean;
  readonly hasVertexColors: boolean;
  readonly hasUv1: boolean;
  readonly issuePaths: Readonly<Record<string, readonly string[]>>;
}

export interface GltfMaterialFeatureReport {
  readonly count: number;
  readonly materialFields: readonly (keyof MaterialSpec)[];
  readonly textureFields: readonly (keyof MaterialSpec)[];
  readonly extensions: readonly string[];
  readonly unsupportedKnownExtensions: readonly string[];
  readonly alphaModes: readonly string[];
  readonly uvSets: readonly number[];
  readonly textureTransformCount: number;
  readonly volumeThicknessTextureCount: number;
  readonly specularGlossinessMaterialCount: number;
  readonly specularGlossinessTextureCount: number;
  readonly doubleSidedCount: number;
  readonly issuePaths: Readonly<Record<string, readonly string[]>>;
}

export interface GltfAnimationFeatureReport {
  readonly count: number;
  readonly channelCount: number;
  readonly paths: readonly string[];
  readonly interpolations: readonly string[];
  readonly targetNodeCount: number;
}

export interface GltfSceneGraphFeatureReport {
  readonly scenes: number;
  readonly nodes: number;
  readonly cameras: number;
  readonly cameraPaths: readonly string[];
  readonly punctualLights: number;
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
  readonly category: 'extension' | 'primitive' | 'material' | 'scene';
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

const REQUIRED_EXTENSION_SUPPORT = new Set([
  'KHR_draco_mesh_compression',
  'EXT_meshopt_compression',
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
  'KHR_texture_transform',
  'KHR_texture_basisu',
  'EXT_texture_webp',
  'MSFT_texture_dds',
]);

const EXTENSIONS_REQUIRING_HOST_HOOK = new Set([
  'KHR_draco_mesh_compression',
  'EXT_meshopt_compression',
  'KHR_texture_basisu',
  'EXT_texture_webp',
  'MSFT_texture_dds',
]);

const TEXTURE_SOURCE_EXTENSIONS = new Set([
  'KHR_texture_basisu',
  'EXT_texture_webp',
  'MSFT_texture_dds',
]);

const COMMON_UNSUPPORTED_EXTENSIONS = new Set<string>();

const UNSUPPORTED_PRIMITIVE_MODES = new Set([0, 1, 2, 3]);
const VERTEX_COLOR_SUPPORT: Readonly<Record<GltfBackendProfileId, BackendSupportMode>> = Object.freeze({
  'pt-webgl2': 'native',
  'pt-webgpu': 'native',
  'pt-webgpu-lite': 'unsupported',
  'walkaround-hybrid': 'unsupported',
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
  if (report.extensions.required.includes(ext) || !TEXTURE_SOURCE_EXTENSIONS.has(ext)) return fallback;
  return report.extensions.sourcePaths[ext]?.find((path) =>
    path.startsWith('textures[') && path.endsWith(`extensions.${ext}`),
  ) ?? fallback;
}

export function analyzeGltfAsset(gltf: GltfJson): GltfFeatureReport {
  const extensions = analyzeExtensions(gltf);
  const resources = analyzeResources(gltf);
  const primitives = analyzePrimitives(gltf);
  const materials = analyzeMaterials(gltf.materials ?? []);
  const animations = analyzeAnimations(gltf);
  const punctualLights = extractPunctualLightCount(gltf);

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
      nodes: gltf.nodes?.length ?? 0,
      cameras: gltf.cameras?.length ?? 0,
      cameraPaths: (gltf.cameras ?? []).map((_, index) => `cameras[${index}]`),
      punctualLights,
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
      message: `glTF primitive mode ${mode} has no core primitive representation.`,
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
        'but shipping backends do not yet apply those deltas to posed tangent-space shading.',
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

  if (report.sceneGraph.cameras > 0) {
    addIssue({
      category: 'scene',
      name: 'cameras',
      support: 'unsupported',
      path: report.sceneGraph.cameraPaths[0] ?? 'cameras',
      message: 'glTF cameras are reported for host inspection but are not imported into the core Scene contract.',
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
        'but glossiness-in-alpha is not baked into roughnessMap; scalar glossinessFactor drives roughness.',
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
    addIssue({
      category: 'material',
      name: `TEXCOORD_${uvSet}`,
      support: 'unsupported',
      path: firstSourcePath(report.materials.issuePaths, `uvSet:${uvSet}`, 'materials'),
      message:
        `glTF material textures reference TEXCOORD_${uvSet}, but the core Scene ` +
        'contract currently carries only UV sets 0 and 1 (`uvs` / `uv1`).',
    });
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

function analyzeExtensions(gltf: GltfJson): GltfExtensionReport {
  const used = sorted(gltf.extensionsUsed ?? []);
  const required = sorted(gltf.extensionsRequired ?? []);
  const all = new Set([...used, ...required]);
  const sourcePaths: SourcePathMap = new Map();
  (gltf.extensionsUsed ?? []).forEach((ext, index) => addSourcePath(sourcePaths, ext, `extensionsUsed[${index}]`));
  (gltf.extensionsRequired ?? []).forEach((ext, index) => addSourcePath(sourcePaths, ext, `extensionsRequired[${index}]`));
  collectNestedExtensionNames(gltf, all, sourcePaths);

  const supported: string[] = [];
  const requiresHook: string[] = [];
  const unsupportedOptional: string[] = [];
  const unsupportedRequired: string[] = [];

  for (const ext of sorted(all)) {
    if (extensionRequiresHostHook(gltf, ext, required)) requiresHook.push(ext);
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
    sourcePaths: sourcePathRecord(sourcePaths),
  };
}

function extensionRequiresHostHook(
  gltf: GltfJson,
  ext: string,
  required: readonly string[],
): boolean {
  if (!EXTENSIONS_REQUIRING_HOST_HOOK.has(ext)) return false;
  if (!TEXTURE_SOURCE_EXTENSIONS.has(ext)) return true;

  // Required texture-source extensions cannot be safely ignored. Optional
  // texture-source extensions only need a host hook when they are the sole
  // available image source for at least one texture; otherwise the loader uses
  // the base `texture.source` fallback until the host opts into the extension.
  if (required.includes(ext)) return true;
  return (gltf.textures ?? []).some((texture) =>
    texture.source === undefined &&
    textureSourceExtensionHasImageSource(texture.extensions?.[ext]),
  );
}

function textureSourceExtensionHasImageSource(value: unknown): boolean {
  if (value == null || typeof value !== 'object' || Array.isArray(value)) return false;
  return typeof (value as { readonly source?: unknown }).source === 'number';
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

function analyzePrimitives(gltf: GltfJson): GltfPrimitiveFeatureReport {
  const byMode = new Map<string, number>();
  const unsupportedModes = new Set<string>();
  const attributeSemantics = new Set<string>();
  const issuePaths: SourcePathMap = new Map();
  let total = 0;
  let usesDraco = false;
  let usesMeshopt = false;
  let hasTangents = false;
  let hasMorphTargets = false;
  let hasMorphTargetTangents = false;
  let hasVertexColors = false;
  let hasUv1 = false;
  let hasJointAttrs = false;

  for (const [meshIndex, mesh] of (gltf.meshes ?? []).entries()) {
    for (const [primitiveIndex, primitive] of (mesh.primitives ?? []).entries()) {
      const primitivePath = `meshes[${meshIndex}].primitives[${primitiveIndex}]`;
      total += 1;
      addSourcePath(issuePaths, 'kind:mesh', primitivePath);
      const mode = primitive.mode ?? 4;
      const modeKey = String(mode);
      byMode.set(modeKey, (byMode.get(modeKey) ?? 0) + 1);
      if (UNSUPPORTED_PRIMITIVE_MODES.has(mode)) {
        unsupportedModes.add(modeKey);
        addSourcePath(issuePaths, `mode:${modeKey}`, `${primitivePath}.mode`);
      }
      for (const semantic of Object.keys(primitive.attributes ?? {})) {
        attributeSemantics.add(semantic);
        if (semantic === 'TANGENT') hasTangents = true;
        if (semantic === 'COLOR_0') {
          hasVertexColors = true;
          addSourcePath(issuePaths, 'vertexColors', `${primitivePath}.attributes.COLOR_0`);
        }
        if (semantic === 'TEXCOORD_1') hasUv1 = true;
        if (semantic === 'JOINTS_0' || semantic === 'WEIGHTS_0') {
          hasJointAttrs = true;
          addSourcePath(issuePaths, 'kind:skinned-mesh', `${primitivePath}.attributes.${semantic}`);
        }
      }
      const ext = primitive.extensions ?? {};
      if (ext['KHR_draco_mesh_compression']) usesDraco = true;
      if (Object.keys(ext).includes('EXT_meshopt_compression')) usesMeshopt = true;
      if ((primitive.targets?.length ?? 0) > 0) {
        hasMorphTargets = true;
        addSourcePath(issuePaths, 'kind:skinned-mesh', `${primitivePath}.targets`);
        for (const [targetIndex, target] of (primitive.targets ?? []).entries()) {
          if (target['TANGENT'] !== undefined) {
            hasMorphTargetTangents = true;
            addSourcePath(issuePaths, 'morphTargetTangents', `${primitivePath}.targets[${targetIndex}].TANGENT`);
          }
        }
      }
    }
  }
  (gltf.skins ?? []).forEach((_, index) => addSourcePath(issuePaths, 'kind:skinned-mesh', `skins[${index}]`));
  for (const bv of gltf.bufferViews ?? []) {
    if (bv.extensions?.['EXT_meshopt_compression']) usesMeshopt = true;
  }
  const hasSkins = (gltf.skins?.length ?? 0) > 0 || hasJointAttrs;
  const expectedPrimitiveKinds = new Set<'mesh' | 'skinned-mesh'>();
  expectedPrimitiveKinds.add('mesh');
  if (hasSkins || hasMorphTargets) expectedPrimitiveKinds.add('skinned-mesh');
  return {
    total,
    byMode: Object.fromEntries([...byMode.entries()].sort()),
    unsupportedModes: sorted(unsupportedModes),
    attributeSemantics: sorted(attributeSemantics),
    expectedPrimitiveKinds: sorted(expectedPrimitiveKinds) as ('mesh' | 'skinned-mesh')[],
    usesDraco,
    usesMeshopt,
    hasTangents,
    hasMorphTargets,
    hasMorphTargetTangents,
    hasSkins,
    hasVertexColors,
    hasUv1,
    issuePaths: sourcePathRecord(issuePaths),
  };
}

function analyzeMaterials(materials: readonly GltfMaterial[]): GltfMaterialFeatureReport {
  const fields = new Set<keyof MaterialSpec>();
  const textureFields = new Set<keyof MaterialSpec>();
  const extensions = new Set<string>();
  const unsupportedKnownExtensions = new Set<string>();
  const alphaModes = new Set<string>();
  const uvSets = new Set<number>();
  const issuePaths: SourcePathMap = new Map();
  let textureTransformCount = 0;
  let volumeThicknessTextureCount = 0;
  let specularGlossinessMaterialCount = 0;
  let specularGlossinessTextureCount = 0;
  let doubleSidedCount = 0;

  const addField = (field: keyof MaterialSpec, path: string): void => {
    fields.add(field);
    addSourcePath(issuePaths, `field:${String(field)}`, path);
  };
  const addTexture = (field: keyof MaterialSpec, info: GltfTextureInfo | undefined, path: string): void => {
    if (info == null) return;
    fields.add(field);
    textureFields.add(field);
    addSourcePath(issuePaths, `field:${String(field)}`, path);
    const uvSet = textureInfoUvSet(info);
    uvSets.add(uvSet);
    if (uvSet > 1) addSourcePath(issuePaths, `uvSet:${uvSet}`, textureInfoUvSetPath(info, path));
    if (info.extensions?.KHR_texture_transform) textureTransformCount += 1;
  };

  for (const [materialIndex, mat] of materials.entries()) {
    const matPath = `materials[${materialIndex}]`;
    const pbr = mat.pbrMetallicRoughness;
    if (pbr?.baseColorFactor) {
      addField('baseColor', `${matPath}.pbrMetallicRoughness.baseColorFactor`);
      if ((pbr.baseColorFactor[3] ?? 1) < 1) {
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
      if (specGloss.diffuseFactor?.[3] !== undefined && specGloss.diffuseFactor[3] < 1) addField('opacity', `${specGlossPath}.diffuseFactor[3]`);
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
    count: materials.length,
    materialFields: sorted(fields) as (keyof MaterialSpec)[],
    textureFields: sorted(textureFields) as (keyof MaterialSpec)[],
    extensions: sorted(extensions),
    unsupportedKnownExtensions: sorted(unsupportedKnownExtensions),
    alphaModes: sorted(alphaModes),
    uvSets: [...uvSets].sort((a, b) => a - b),
    textureTransformCount,
    volumeThicknessTextureCount,
    specularGlossinessMaterialCount,
    specularGlossinessTextureCount,
    doubleSidedCount,
    issuePaths: sourcePathRecord(issuePaths),
  };
}

function analyzeAnimations(gltf: GltfJson): GltfAnimationFeatureReport {
  const paths = new Set<string>();
  const interpolations = new Set<string>();
  const targetNodes = new Set<number>();
  let channelCount = 0;
  for (const animation of gltf.animations ?? []) {
    for (const sampler of animation.samplers ?? []) {
      interpolations.add(sampler.interpolation ?? 'LINEAR');
    }
    for (const channel of animation.channels ?? []) {
      channelCount += 1;
      paths.add(channel.target.path);
      if (channel.target.node !== undefined) targetNodes.add(channel.target.node);
    }
  }
  return {
    count: gltf.animations?.length ?? 0,
    channelCount,
    paths: sorted(paths),
    interpolations: sorted(interpolations),
    targetNodeCount: targetNodes.size,
  };
}

function textureInfoUvSet(info: GltfTextureInfo): number {
  return info.extensions?.KHR_texture_transform?.texCoord ?? info.texCoord ?? 0;
}

function textureInfoUvSetPath(info: GltfTextureInfo, path: string): string {
  return info.extensions?.KHR_texture_transform?.texCoord !== undefined
    ? `${path}.extensions.KHR_texture_transform.texCoord`
    : `${path}.texCoord`;
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

function sorted<T extends string>(values: Iterable<T>): T[] {
  return [...new Set(values)].sort();
}
