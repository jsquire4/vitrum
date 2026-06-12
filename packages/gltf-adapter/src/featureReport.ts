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
  readonly category: 'extension' | 'primitive' | 'material';
  readonly name: string;
  readonly support: BackendSupportMode | 'requires-hook' | 'unknown';
  readonly path?: string;
  readonly message: string;
}

export interface GltfBackendCompatibility {
  readonly backend: BackendId;
  readonly unsupportedCount: number;
  readonly approximateCount: number;
  readonly nativeCount: number;
  readonly requiresHookCount: number;
  readonly issues: readonly GltfCompatibilityIssue[];
  readonly isCompatible: boolean;
}

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

const COMMON_UNSUPPORTED_EXTENSIONS = new Set<string>();

const UNSUPPORTED_PRIMITIVE_MODES = new Set([0, 1, 2, 3]);

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
      punctualLights,
    },
  };
}

export function evaluateGltfBackendCompatibility(
  report: GltfFeatureReport,
  backend: BackendId,
): GltfBackendCompatibility {
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
      message: `Required glTF extension "${ext}" is not supported by the adapter.`,
    });
  }
  for (const ext of report.extensions.requiresHook) {
    addIssue({
      category: 'extension',
      name: ext,
      support: 'requires-hook',
      message: `glTF extension "${ext}" requires host-supplied decode support.`,
    });
  }
  for (const ext of report.extensions.unsupportedOptional) {
    addIssue({
      category: 'extension',
      name: ext,
      support: 'unsupported',
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
      message: `glTF primitive mode ${mode} has no core primitive representation.`,
    });
  }

  if (report.primitives.hasMorphTargetTangents) {
    addIssue({
      category: 'primitive',
      name: 'morphTargetTangents',
      support: 'unsupported',
      message: 'glTF morph-target TANGENT deltas have no core primitive field and are ignored by the adapter.',
    });
  }

  if (report.materials.specularGlossinessMaterialCount > 0) {
    addIssue({
      category: 'material',
      name: 'KHR_materials_pbrSpecularGlossiness',
      support: 'approximate',
      message:
        'Archived specular-glossiness materials are converted approximately to metallic-roughness plus specular fields.',
    });
  }

  if (report.materials.specularGlossinessTextureCount > 0) {
    addIssue({
      category: 'material',
      name: 'KHR_materials_pbrSpecularGlossiness.specularGlossinessTexture.glossinessAlpha',
      support: 'approximate',
      message:
        'Archived specular-glossiness texture RGB is imported as specularColorMap, ' +
        'but glossiness-in-alpha is not baked into roughnessMap; scalar glossinessFactor drives roughness.',
    });
  }

  for (const field of report.materials.materialFields) {
    const support = ledger.supportDetails.materials[field] ?? 'unknown';
    if (support === 'native') {
      nativeCount += 1;
    } else {
      addIssue({
        category: 'material',
        name: String(field),
        support,
        message: `Backend ${backend} reports material field "${String(field)}" as ${support}.`,
      });
    }
  }

  return {
    backend,
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
  const preferred: readonly BackendId[] = policy === 'realtime'
    ? ['walkaround-hybrid', 'pt-webgpu', 'pt-webgl2']
    : ['pt-webgl2', 'pt-webgpu', 'walkaround-hybrid'];
  const order = new Map(preferred.map((b, i) => [b, i]));
  return preferred
    .map((backend) => evaluateGltfBackendCompatibility(report, backend))
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
      return (order.get(a.backend) ?? 99) - (order.get(b.backend) ?? 99);
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
  collectNestedExtensionNames(gltf, all);

  const supported: string[] = [];
  const requiresHook: string[] = [];
  const unsupportedOptional: string[] = [];
  const unsupportedRequired: string[] = [];

  for (const ext of sorted(all)) {
    if (EXTENSIONS_REQUIRING_HOST_HOOK.has(ext)) requiresHook.push(ext);
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
  };
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
  let total = 0;
  let usesDraco = false;
  let usesMeshopt = false;
  let hasTangents = false;
  let hasMorphTargets = false;
  let hasMorphTargetTangents = false;
  let hasVertexColors = false;
  let hasUv1 = false;
  let hasJointAttrs = false;

  for (const mesh of gltf.meshes ?? []) {
    for (const primitive of mesh.primitives ?? []) {
      total += 1;
      const mode = primitive.mode ?? 4;
      const modeKey = String(mode);
      byMode.set(modeKey, (byMode.get(modeKey) ?? 0) + 1);
      if (UNSUPPORTED_PRIMITIVE_MODES.has(mode)) unsupportedModes.add(modeKey);
      for (const semantic of Object.keys(primitive.attributes ?? {})) {
        attributeSemantics.add(semantic);
        if (semantic === 'TANGENT') hasTangents = true;
        if (semantic === 'COLOR_0') hasVertexColors = true;
        if (semantic === 'TEXCOORD_1') hasUv1 = true;
        if (semantic === 'JOINTS_0' || semantic === 'WEIGHTS_0') hasJointAttrs = true;
      }
      const ext = primitive.extensions ?? {};
      if (ext['KHR_draco_mesh_compression']) usesDraco = true;
      if (Object.keys(ext).includes('EXT_meshopt_compression')) usesMeshopt = true;
      if ((primitive.targets?.length ?? 0) > 0) {
        hasMorphTargets = true;
        if (primitive.targets?.some((target) => target['TANGENT'] !== undefined)) {
          hasMorphTargetTangents = true;
        }
      }
    }
  }
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
  };
}

function analyzeMaterials(materials: readonly GltfMaterial[]): GltfMaterialFeatureReport {
  const fields = new Set<keyof MaterialSpec>();
  const textureFields = new Set<keyof MaterialSpec>();
  const extensions = new Set<string>();
  const unsupportedKnownExtensions = new Set<string>();
  const alphaModes = new Set<string>();
  const uvSets = new Set<number>();
  let textureTransformCount = 0;
  let volumeThicknessTextureCount = 0;
  let specularGlossinessMaterialCount = 0;
  let specularGlossinessTextureCount = 0;
  let doubleSidedCount = 0;

  const addField = (field: keyof MaterialSpec): void => {
    fields.add(field);
  };
  const addTexture = (field: keyof MaterialSpec, info?: GltfTextureInfo): void => {
    if (info == null) return;
    fields.add(field);
    textureFields.add(field);
    uvSets.add(textureInfoUvSet(info));
    if (info.extensions?.KHR_texture_transform) textureTransformCount += 1;
  };

  for (const mat of materials) {
    const pbr = mat.pbrMetallicRoughness;
    if (pbr?.baseColorFactor) {
      addField('baseColor');
      if ((pbr.baseColorFactor[3] ?? 1) < 1) addField('opacity');
    }
    if (pbr?.metallicFactor !== undefined) addField('metallic');
    if (pbr?.roughnessFactor !== undefined) addField('roughness');
    addTexture('baseColorMap', pbr?.baseColorTexture);
    if (pbr?.metallicRoughnessTexture) {
      addTexture('roughnessMap', pbr.metallicRoughnessTexture);
      addTexture('metallicMap', pbr.metallicRoughnessTexture);
    }
    addTexture('normalMap', mat.normalTexture);
    if (mat.normalTexture?.scale !== undefined) addField('normalScale');
    addTexture('aoMap', mat.occlusionTexture);
    if (mat.occlusionTexture?.strength !== undefined) addField('aoMapIntensity');
    if (mat.emissiveFactor) addField('emissive');
    addTexture('emissiveMap', mat.emissiveTexture);
    if (mat.alphaMode !== undefined) {
      addField('alphaMode');
      alphaModes.add(mat.alphaMode);
    }
    if (mat.alphaCutoff !== undefined) addField('alphaCutoff');
    if (mat.doubleSided) doubleSidedCount += 1;

    const ext = mat.extensions ?? {};
    for (const key of Object.keys(ext)) {
      extensions.add(key);
      if (COMMON_UNSUPPORTED_EXTENSIONS.has(key)) unsupportedKnownExtensions.add(key);
    }
    if (ext.KHR_materials_unlit) addField('shadingModel');
    const transmission = ext.KHR_materials_transmission;
    if (transmission) {
      if (transmission.transmissionFactor !== undefined) addField('transmission');
      addTexture('transmissionMap', transmission.transmissionTexture);
    }
    const ior = ext.KHR_materials_ior;
    if (ior?.ior !== undefined) addField('ior');
    const volume = ext.KHR_materials_volume;
    if (volume) {
      if (volume.thicknessFactor !== undefined) addField('thickness');
      addTexture('thicknessMap', volume.thicknessTexture);
      if (volume.attenuationDistance !== undefined) addField('attenuationDistance');
      if (volume.attenuationColor !== undefined) addField('attenuationColor');
      if (volume.thicknessTexture) volumeThicknessTextureCount += 1;
    }
    const specular = ext.KHR_materials_specular;
    if (specular) {
      if (specular.specularFactor !== undefined) addField('specularIntensity');
      if (specular.specularColorFactor !== undefined) addField('specularColor');
      addTexture('specularIntensityMap', specular.specularTexture);
      addTexture('specularColorMap', specular.specularColorTexture);
    }
    const sheen = ext.KHR_materials_sheen;
    if (sheen) {
      addField('sheen');
      if (sheen.sheenColorFactor !== undefined) addField('sheenColor');
      if (sheen.sheenRoughnessFactor !== undefined) addField('sheenRoughness');
      addTexture('sheenColorMap', sheen.sheenColorTexture);
      addTexture('sheenRoughnessMap', sheen.sheenRoughnessTexture);
    }
    const clearcoat = ext.KHR_materials_clearcoat;
    if (clearcoat) {
      if (clearcoat.clearcoatFactor !== undefined) addField('clearcoat');
      if (clearcoat.clearcoatRoughnessFactor !== undefined) addField('clearcoatRoughness');
      addTexture('clearcoatMap', clearcoat.clearcoatTexture);
      addTexture('clearcoatRoughnessMap', clearcoat.clearcoatRoughnessTexture);
      addTexture('clearcoatNormalMap', clearcoat.clearcoatNormalTexture);
      if (clearcoat.clearcoatNormalTexture?.scale !== undefined) addField('clearcoatNormalScale');
    }
    const iridescence = ext.KHR_materials_iridescence;
    if (iridescence) {
      if (iridescence.iridescenceFactor !== undefined) addField('iridescence');
      if (iridescence.iridescenceIor !== undefined) addField('iridescenceIor');
      if (
        iridescence.iridescenceThicknessMinimum !== undefined ||
        iridescence.iridescenceThicknessMaximum !== undefined
      ) {
        addField('iridescenceThicknessRange');
      }
      addTexture('iridescenceMap', iridescence.iridescenceTexture);
      addTexture('iridescenceThicknessMap', iridescence.iridescenceThicknessTexture);
    }
    const anisotropy = ext.KHR_materials_anisotropy;
    if (anisotropy) {
      if (anisotropy.anisotropyStrength !== undefined) addField('anisotropy');
      if (anisotropy.anisotropyRotation !== undefined) addField('anisotropyRotation');
      addTexture('anisotropyMap', anisotropy.anisotropyTexture);
    }
    const dispersion = ext.KHR_materials_dispersion;
    if (dispersion?.dispersion !== undefined && dispersion.dispersion > 0) {
      addField('dispersionAbbeNumber');
    }
    const emissiveStrength = ext.KHR_materials_emissive_strength;
    if (emissiveStrength?.emissiveStrength !== undefined) addField('emissiveIntensity');
    const specGloss = ext.KHR_materials_pbrSpecularGlossiness;
    if (specGloss) {
      specularGlossinessMaterialCount += 1;
      addField('baseColor');
      addField('roughness');
      addField('metallic');
      if (specGloss.diffuseFactor?.[3] !== undefined && specGloss.diffuseFactor[3] < 1) addField('opacity');
      if (specGloss.specularFactor !== undefined) addField('specularColor');
      addTexture('baseColorMap', specGloss.diffuseTexture);
      addTexture('specularColorMap', specGloss.specularGlossinessTexture);
      if (specGloss.specularGlossinessTexture) specularGlossinessTextureCount += 1;
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

function collectNestedExtensionNames(value: unknown, out: Set<string>): void {
  if (value == null || typeof value !== 'object') return;
  if (Array.isArray(value)) {
    for (const item of value) collectNestedExtensionNames(item, out);
    return;
  }
  const obj = value as Record<string, unknown>;
  const ext = obj['extensions'];
  if (ext && typeof ext === 'object' && !Array.isArray(ext)) {
    for (const key of Object.keys(ext as Record<string, unknown>)) out.add(key);
  }
  for (const nested of Object.values(obj)) collectNestedExtensionNames(nested, out);
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
