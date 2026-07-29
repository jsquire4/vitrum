// assetInventory.ts — the pure-JSON glTF asset inventory walk (analyze* family).
// Extracted verbatim from featureReport.ts (T3-E / D15-1, D15-3). Does not decode
// buffers, fetch resources, or mutate the asset. featureReport.ts re-exports
// `analyzeGltfAsset` from here; the extension/texture-source sets are exported for
// backendCompatibility.ts (which imports TEXTURE_SOURCE_EXTENSIONS).

import { type MaterialSpec } from '@vitrum/core';
import {
  accessorBufferViewRange,
  componentByteSize,
  typeComponentCount,
} from './accessors.js';
import { GltfComponentType } from './gltfTypes.js';
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
import { effectiveGltfTextureSourceExtensions } from './textures.js';
import {
  gltfAnimationPointerInterpolationError,
  gltfAnimationPointerOutputAccessorError,
  gltfAnimationPointerTargetComponentCount,
  gltfAnimationPointerTargetDefinitionError,
  gltfAnimationPointerTargetIdentity,
  gltfAnimationTargetsConflict,
  gltfNativeAnimationTargetIdentity,
  isGltfAnimationPointerTargetReachable,
  resolveGltfAnimationPointer,
  type GltfAnimationTargetIdentity,
} from './animationPointer.js';
import type {
  AnalyzeGltfAssetOptions,
  GltfAnimationFeatureReport,
  GltfAnimationMalformedChannelIssue,
  GltfAnimationMalformedChannelKind,
  GltfExtensionReport,
  GltfFeatureReport,
  GltfMalformedPrimitiveIssue,
  GltfMalformedTextureSamplerPolicyUse,
  GltfMaterialFeatureReport,
  GltfMaterialTextureReferenceIssue,
  GltfMaterialVariantMappingIssue,
  GltfPrimitiveAccessorImportIssue,
  GltfPrimitiveAccessorStorageIssue,
  GltfPrimitiveFeatureReport,
  GltfPrimitiveInstancingIssue,
  GltfPrimitiveMaterialReferenceIssue,
  GltfPunctualLightIssue,
  GltfResourceFeatureReport,
  GltfResourceUse,
  GltfSparseAccessorStorageIssueKind,
  GltfTextureSamplerFilterMode,
  GltfTextureSamplerMipMode,
  GltfTextureSamplerPolicyUse,
  GltfTextureSourceExtensionName,
  GltfTextureSourceExtensionUse,
} from './featureReport.types.js';

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
  'KHR_animation_pointer',
  'KHR_node_visibility',
  'KHR_mesh_quantization',
  'KHR_texture_transform',
  'KHR_texture_basisu',
  'EXT_texture_webp',
  'MSFT_texture_dds',
  'EXT_mesh_gpu_instancing',
]);

export const TEXTURE_SOURCE_EXTENSION_NAMES = [
  'KHR_texture_basisu',
  'EXT_texture_webp',
  'MSFT_texture_dds',
] as const satisfies readonly GltfTextureSourceExtensionName[];

export const TEXTURE_SOURCE_EXTENSIONS = new Set<string>(TEXTURE_SOURCE_EXTENSION_NAMES);
const MESHOPT_COMPRESSION_EXTENSIONS = new Set(['EXT_meshopt_compression', 'KHR_meshopt_compression']);

const FALLBACK_GENERATED_PRIMITIVE_MODES = new Set([0, 1, 2, 3]);
const SUPPORTED_GLTF_PRIMITIVE_MODES = new Set([0, 1, 2, 3, 4, 5, 6]);

const CORE_ANIMATION_TARGET_PATHS: ReadonlySet<string> = new Set([
  'translation',
  'rotation',
  'scale',
  'weights',
  'pointer',
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

export function analyzeGltfAsset(
  gltf: GltfJson,
  options: AnalyzeGltfAssetOptions = {},
): GltfFeatureReport {
  const selectedTextureSourceExtensions = new Set<string>(
    effectiveGltfTextureSourceExtensions(options.textureSourceExtensions),
  );
  const sceneScope = options.sceneIndex === undefined
    ? undefined
    : collectGltfSceneReachability(gltf, options.sceneIndex);
  validateAssetPrimitiveSemantics(gltf, sceneScope);
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
    const selectedTextureSource =
      !TEXTURE_SOURCE_EXTENSIONS.has(ext) ||
      selectedTextureSourceExtensions.has(ext);
    if (REQUIRED_EXTENSION_SUPPORT.has(ext) && selectedTextureSource) {
      supported.push(ext);
      continue;
    }
    if (required.includes(ext)) unsupportedRequired.push(ext);
    else {
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

function hasMeshoptCompressionExtension(extensions: Record<string, unknown> | undefined): boolean {
  if (!extensions) return false;
  return [...MESHOPT_COMPRESSION_EXTENSIONS].some((name) => extensions[name] !== undefined);
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
        requiresHook: false,
        ...(gltf.images?.[source]?.mimeType !== undefined ? { mimeType: gltf.images[source].mimeType } : {}),
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

const PRIMITIVE_ATTRIBUTE_SEMANTICS = new Set(['POSITION', 'NORMAL', 'TANGENT']);
const MORPH_TARGET_ATTRIBUTE_SEMANTICS = new Set(['POSITION', 'NORMAL', 'TANGENT']);
const INDEXED_ATTRIBUTE_SEMANTIC_PREFIXES = [
  'TEXCOORD',
  'COLOR',
  'JOINTS',
  'WEIGHTS',
] as const;

function parseCanonicalIndexedAttributeSemantic(
  semantic: string,
  path: string,
): {
  readonly prefix: (typeof INDEXED_ATTRIBUTE_SEMANTIC_PREFIXES)[number];
  readonly setIndex: number;
} | undefined {
  const prefix = INDEXED_ATTRIBUTE_SEMANTIC_PREFIXES.find(
    (candidate) => semantic === candidate || semantic.startsWith(`${candidate}_`),
  );
  if (prefix === undefined) return undefined;
  const match = new RegExp(`^${prefix}_(\\d+)$`, 'u').exec(semantic);
  if (match == null) {
    throw new TypeError(
      `[vitrum/gltf-adapter] ${path} must use the ` +
        `${prefix}_<non-negative canonical integer> form.`,
    );
  }
  const setIndex = Number(match[1]);
  if (!Number.isSafeInteger(setIndex)) {
    throw new RangeError(
      `[vitrum/gltf-adapter] ${path} exceeds the supported non-negative ` +
        'safe-integer semantic range.',
    );
  }
  const canonical = `${prefix}_${setIndex}`;
  if (semantic !== canonical) {
    throw new TypeError(
      `[vitrum/gltf-adapter] ${path} is not canonical; use "${canonical}".`,
    );
  }
  return { prefix, setIndex };
}

function validatePrimitiveAttributeSemantic(semantic: string, path: string): void {
  if (
    PRIMITIVE_ATTRIBUTE_SEMANTICS.has(semantic) ||
    semantic.startsWith('_') ||
    parseCanonicalIndexedAttributeSemantic(semantic, path) !== undefined
  ) {
    return;
  }
  throw new TypeError(
    `[vitrum/gltf-adapter] ${path} uses unknown non-application primitive ` +
      `semantic "${semantic}".`,
  );
}

function validateMorphTargetAttributeSemantic(semantic: string, path: string): void {
  if (MORPH_TARGET_ATTRIBUTE_SEMANTICS.has(semantic) || semantic.startsWith('_')) return;
  const indexed = parseCanonicalIndexedAttributeSemantic(semantic, path);
  if (indexed?.prefix === 'TEXCOORD' || indexed?.prefix === 'COLOR') return;
  throw new TypeError(
    `[vitrum/gltf-adapter] ${path} uses unknown non-application morph-target ` +
      `semantic "${semantic}".`,
  );
}

function validateAssetPrimitiveSemantics(
  gltf: GltfJson,
  sceneScope: GltfSceneReachability | undefined,
): void {
  for (const [meshIndex, mesh] of (gltf.meshes ?? []).entries()) {
    for (const [primitiveIndex, primitive] of mesh.primitives.entries()) {
      if (
        sceneScope !== undefined &&
        !sceneScope.primitiveKeys.has(gltfPrimitiveKey(meshIndex, primitiveIndex))
      ) {
        continue;
      }
      const primitivePath = `meshes[${meshIndex}].primitives[${primitiveIndex}]`;
      for (const semantic of Object.keys(primitive.attributes ?? {})) {
        validatePrimitiveAttributeSemantic(
          semantic,
          `${primitivePath}.attributes.${semantic}`,
        );
      }
      for (const [targetIndex, target] of (primitive.targets ?? []).entries()) {
        for (const semantic of Object.keys(target)) {
          validateMorphTargetAttributeSemantic(
            semantic,
            `${primitivePath}.targets[${targetIndex}].${semantic}`,
          );
        }
      }
    }
  }
}

function collectSkinInfluenceSetIndices(
  attributes: GltfPrimitive['attributes'] | undefined,
  primitivePath: string,
): number[] {
  const sets = new Set<number>();
  for (const attrName of Object.keys(attributes ?? {})) {
    const match = /^(JOINTS|WEIGHTS)_([0-9]+)$/u.exec(attrName);
    if (!match) continue;
    const suffix = match[2]!;
    const setIndex = Number(suffix);
    if (!Number.isSafeInteger(setIndex)) {
      throw new RangeError(
        `[vitrum/gltf-adapter] ${primitivePath}.attributes.${attrName} ` +
          'exceeds the supported non-negative safe-integer semantic range.',
      );
    }
    const canonical = `${match[1]}_${setIndex}`;
    if (attrName !== canonical) {
      throw new TypeError(
        `[vitrum/gltf-adapter] ${primitivePath}.attributes.${attrName} is not ` +
          `canonical; use "${canonical}".`,
      );
    }
    sets.add(setIndex);
  }
  return [...sets].sort((a, b) => a - b);
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
  // Retained in the report shape for source compatibility. The scalable core
  // color contract now preserves every valid COLOR_n stream.
  const ignoredVertexColorSets = new Set<string>();
  let hasUv1 = false;
  let hasBoundSkinAttrs = false;
  let hasInstancing = false;
  let hasInstancedSkinnedOrMorphed = false;
  // Retained in the report shape for compatibility. The scalable core skin
  // contract now preserves all complete JOINTS_N/WEIGHTS_N sets.
  const hasCollapsedSkinInfluenceSets = false;
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
      const skinInfluenceSetIndices = collectSkinInfluenceSetIndices(
        primitive.attributes,
        primitivePath,
      );
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
          hasVertexColors = true;
          addSourcePath(issuePaths, 'vertexColors', `${primitivePath}.attributes.${semantic}`);
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
          for (const setIndex of skinInfluenceSetIndices) {
            if (setIndex === 0) continue;
            const hasSetJoints = primitive.attributes?.[`JOINTS_${setIndex}`] !== undefined;
            const hasSetWeights = primitive.attributes?.[`WEIGHTS_${setIndex}`] !== undefined;
            if (!(hasSetJoints && hasSetWeights) && (hasSetJoints || hasSetWeights)) {
              hasIncompleteSkinAttributes = true;
              addSourcePath(issuePaths, 'incompleteSkinAttributes', primitivePath);
            }
          }
        } else {
          hasIncompleteSkinAttributes = true;
          addSourcePath(issuePaths, 'incompleteSkinAttributes', primitivePath);
        }
      }
      if (skinInfluenceSetIndices.length > 0) {
        if (!meshHasSkinnedNode || meshHasUnskinnedNode) {
          hasIgnoredSkinAttributes = true;
          for (const setIndex of skinInfluenceSetIndices) {
            if (primitive.attributes?.[`JOINTS_${setIndex}`] !== undefined) {
              addSourcePath(issuePaths, 'ignoredSkinAttributes', `${primitivePath}.attributes.JOINTS_${setIndex}`);
            }
            if (primitive.attributes?.[`WEIGHTS_${setIndex}`] !== undefined) {
              addSourcePath(issuePaths, 'ignoredSkinAttributes', `${primitivePath}.attributes.WEIGHTS_${setIndex}`);
            }
          }
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
              if (!primitiveMorphTexcoordIsRepresentable(primitive, uvIndex)) {
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
      const accessor = gltf.accessors?.[accessorIndex];
      const expectedTypes = instancingAccessorTypes(attr);
      const accessorCanDefineInstanceCount =
        accessor !== undefined &&
        expectedTypes?.includes(accessor.type) === true &&
        FLOAT_ACCESSOR_COMPONENT_TYPES.includes(accessor.componentType) &&
        Number.isSafeInteger(accessor.count) &&
        accessor.count > 0;
      if (instancingCount === undefined && accessorCanDefineInstanceCount) {
        instancingCount = accessor.count;
      }
      addPrimitiveAccessorImportIssueForInstancing(gltf, accessorImportIssues, {
        semantic: attr,
        accessorIndex,
        expectedCount: instancingCount,
        nodeIndex,
      });
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
    hasCollapsedSkinInfluenceSets,
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
  const positionStorageIssue = accessorDeclaredStorageIssue(gltf, positionAccessorIndex, positionAccessor);
  if (positionStorageIssue !== undefined) {
    issues.push({
      kind: 'truncated-position-buffer-view',
      path: positionStorageIssue.path,
      meshIndex,
      primitiveIndex,
      accessorIndex: positionAccessorIndex,
      bufferViewIndex: positionStorageIssue.bufferViewIndex,
      mode,
      requiredByteLength: positionStorageIssue.requiredByteLength,
      byteLength: positionStorageIssue.byteLength,
    });
    return issues;
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
    const indexStorageIssue = accessorDeclaredStorageIssue(gltf, indexAccessorIndex, indexAccessor);
    if (indexStorageIssue !== undefined) {
      issues.push({
        kind: 'truncated-index-buffer-view',
        path: indexStorageIssue.path,
        meshIndex,
        primitiveIndex,
        accessorIndex: indexAccessorIndex,
        bufferViewIndex: indexStorageIssue.bufferViewIndex,
        mode,
        requiredByteLength: indexStorageIssue.requiredByteLength,
        byteLength: indexStorageIssue.byteLength,
      });
      return issues;
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
  return accessor.componentType === GltfComponentType.UNSIGNED_BYTE ||
    accessor.componentType === GltfComponentType.UNSIGNED_SHORT ||
    accessor.componentType === GltfComponentType.UNSIGNED_INT;
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
  readonly requiredByteLength?: number;
  readonly byteLength?: number;
}

interface AccessorDeclaredStorageIssue {
  readonly path: string;
  readonly bufferViewIndex: number;
  readonly requiredByteLength: number;
  readonly byteLength: number;
}

function accessorDeclaredStorageIssue(
  gltf: GltfJson,
  accessorIndex: number,
  accessor: NonNullable<GltfJson['accessors']>[number],
): AccessorDeclaredStorageIssue | undefined {
  if (accessor.bufferView === undefined) return undefined;
  const bufferView = gltf.bufferViews?.[accessor.bufferView];
  if (bufferView === undefined) return undefined;
  let range: ReturnType<typeof accessorBufferViewRange>;
  try {
    range = accessorBufferViewRange(accessor, bufferView);
  } catch {
    return undefined;
  }
  if (range.requiredByteLength <= bufferView.byteLength) return undefined;
  return {
    path: `accessors[${accessorIndex}].bufferView`,
    bufferViewIndex: accessor.bufferView,
    requiredByteLength: range.requiredByteLength,
    byteLength: bufferView.byteLength,
  };
}

function componentByteSizeOrUndefined(componentType: number): number | undefined {
  try {
    return componentByteSize(componentType);
  } catch {
    return undefined;
  }
}

function typeComponentCountOrUndefined(type: string): number | undefined {
  try {
    return typeComponentCount(type);
  } catch {
    return undefined;
  }
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
  const indicesByteOffset = sparse.indices.byteOffset ?? 0;
  const indicesRequiredByteLength = indicesByteOffset + sparse.count * componentByteSize(sparse.indices.componentType);
  if (indicesRequiredByteLength > indicesBufferView.byteLength) {
    return {
      kind: 'truncated-sparse-indices-buffer-view',
      path: `accessors[${accessorIndex}].sparse.indices.bufferView`,
      bufferViewIndex: indicesBufferViewIndex,
      requiredByteLength: indicesRequiredByteLength,
      byteLength: indicesBufferView.byteLength,
    };
  }
  const valuesByteOffset = sparse.values.byteOffset ?? 0;
  const valuesComponentSize = componentByteSizeOrUndefined(accessor.componentType);
  const valueComponentCount = typeComponentCountOrUndefined(accessor.type);
  if (valuesComponentSize === undefined || valueComponentCount === undefined) return undefined;
  const valuesRequiredByteLength = valuesByteOffset + sparse.count * valueComponentCount * valuesComponentSize;
  if (valuesRequiredByteLength > valuesBufferView.byteLength) {
    return {
      kind: 'truncated-sparse-values-buffer-view',
      path: `accessors[${accessorIndex}].sparse.values.bufferView`,
      bufferViewIndex: valuesBufferViewIndex,
      requiredByteLength: valuesRequiredByteLength,
      byteLength: valuesBufferView.byteLength,
    };
  }

  return undefined;
}

function sparseIssueDetails(issue: SparseAccessorStorageIssue): {
  readonly sparseIssueKind: GltfSparseAccessorStorageIssueKind;
  readonly bufferViewIndex?: number;
  readonly bufferIndex?: number;
  readonly componentType?: number;
  readonly requiredByteLength?: number;
  readonly byteLength?: number;
} {
  return {
    sparseIssueKind: issue.kind,
    ...(issue.bufferViewIndex !== undefined ? { bufferViewIndex: issue.bufferViewIndex } : {}),
    ...(issue.bufferIndex !== undefined ? { bufferIndex: issue.bufferIndex } : {}),
    ...(issue.componentType !== undefined ? { componentType: issue.componentType } : {}),
    ...(issue.requiredByteLength !== undefined ? { requiredByteLength: issue.requiredByteLength } : {}),
    ...(issue.byteLength !== undefined ? { byteLength: issue.byteLength } : {}),
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
  if (!Number.isSafeInteger(accessor.count) || accessor.count < 1) {
    issues.push({
      ...base,
      kind: 'invalid-accessor-count',
      path: `accessors[${input.accessorIndex}].count`,
      expectedCount: input.expectedCount ?? 1,
      actualCount: accessor.count,
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
    return;
  }
  const storageIssue = accessorDeclaredStorageIssue(gltf, input.accessorIndex, accessor);
  if (storageIssue !== undefined) {
    issues.push({
      ...base,
      kind: 'truncated-buffer-view',
      path: storageIssue.path,
      bufferViewIndex: storageIssue.bufferViewIndex,
      requiredByteLength: storageIssue.requiredByteLength,
      byteLength: storageIssue.byteLength,
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
        const sortedUvSets = [...uvSets].sort((a, b) => a - b);
        const highUvSets = sortedUvSets.filter((uvSet) => uvSet > 1);
        if (highUvSets.length === 0) continue;
        for (const uvSet of highUvSets) {
          if (primitive.attributes?.[`TEXCOORD_${uvSet}`] === undefined) {
            unrepresentable.add(uvSet);
          }
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
  primitive: GltfPrimitive,
  uvIndex: number,
): boolean {
  const baseSemantic = `TEXCOORD_${uvIndex}`;
  return primitive.attributes?.[baseSemantic] !== undefined;
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
  const materialProfiles =
    new Set<NonNullable<GltfMaterialFeatureReport['materialProfiles']>[number]>();
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
  const addProfile = (
    profile: NonNullable<GltfMaterialFeatureReport['materialProfiles']>[number],
    path: string,
  ): void => {
    materialProfiles.add(profile);
    addSourcePath(issuePaths, `profile:${profile}`, path);
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
      addField('doubleSided', `${matPath}.doubleSided`);
      addSourcePath(issuePaths, 'doubleSided', `${matPath}.doubleSided`);
    }

    const ext = mat.extensions ?? {};
    for (const key of Object.keys(ext)) {
      extensions.add(key);
      addSourcePath(issuePaths, `extension:${key}`, `${matPath}.extensions.${key}`);
      if (!REQUIRED_EXTENSION_SUPPORT.has(key)) unsupportedKnownExtensions.add(key);
    }
    if (ext.KHR_materials_unlit) addField('shadingModel', `${matPath}.extensions.KHR_materials_unlit`);
    const transmission = ext.KHR_materials_transmission;
    if (transmission) {
      if (transmission.transmissionFactor !== undefined) {
        addField('transmission', `${matPath}.extensions.KHR_materials_transmission.transmissionFactor`);
      }
      addTexture('transmissionMap', transmission.transmissionTexture, `${matPath}.extensions.KHR_materials_transmission.transmissionTexture`);
      const transmissionFactor = transmission.transmissionFactor ?? 0;
      const specGloss = ext.KHR_materials_pbrSpecularGlossiness;
      const glossinessFactor = specGloss?.glossinessFactor ?? 1;
      const roughnessFactor = specGloss == null
        ? (pbr?.roughnessFactor ?? 1)
        : 1 - (
          Number.isFinite(glossinessFactor)
            ? Math.min(1, Math.max(0, glossinessFactor))
            : 1
        );
      if (transmissionFactor > 0) {
        const transmissionPath = `${matPath}.extensions.KHR_materials_transmission`;
        addProfile(
          roughnessFactor > 0 ? 'roughTransmission' : 'deltaTransmission',
          transmissionPath,
        );
        const clearcoatActive =
          (ext.KHR_materials_clearcoat?.clearcoatFactor ?? 0) > 0;
        const sheenActive =
          ext.KHR_materials_sheen?.sheenColorFactor?.some(
            (component) => Number.isFinite(component) && component > 0,
          ) === true;
        const iridescenceActive =
          (ext.KHR_materials_iridescence?.iridescenceFactor ?? 0) > 0;
        if (clearcoatActive || sheenActive || iridescenceActive) {
          addProfile('layeredTransmission', transmissionPath);
        }
        if (
          mat.normalTexture !== undefined ||
          (clearcoatActive &&
            ext.KHR_materials_clearcoat?.clearcoatNormalTexture !== undefined)
        ) {
          addProfile('normalMappedTransmission', transmissionPath);
        }
        const volume = ext.KHR_materials_volume;
        if (
          (volume?.thicknessFactor ?? 0) > 0 ||
          volume?.thicknessTexture !== undefined
        ) {
          addProfile('participatingMedia', transmissionPath);
        }
      }
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
    materialFields: sorted(fields),
    materialProfiles: sorted(materialProfiles),
    textureFields: sorted(textureFields),
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
        if (!Number.isSafeInteger(mapping.material) || mapping.material < 0 || mapping.material >= materialCount) {
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
          if (!Number.isSafeInteger(variantIndex) || variantIndex < 0 || variantIndex >= variantCount) {
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
    const claimedTargets: GltfAnimationTargetIdentity[] = [];
    for (const [channelIndex, channel] of (animation.channels ?? []).entries()) {
      const targetPath = channel.target.path;
      if (targetPath === 'pointer') {
        const pointer = channel.target.extensions?.KHR_animation_pointer?.pointer;
        const pointerTarget = resolveGltfAnimationPointer(pointer);
        if (
          sceneScope !== undefined &&
          pointerTarget !== undefined &&
          !isGltfAnimationPointerTargetReachable(pointerTarget, sceneScope)
        ) {
          continue;
        }
        animationHasReachableChannel = true;
        channelCount += 1;
        paths.add(targetPath);
        if (pointerTarget === undefined) {
          unsupportedTargetPaths.add(targetPath);
          addSourcePath(
            issuePaths,
            `unsupportedTargetPath:${targetPath}`,
            pointer === undefined
              ? `animations[${animationIndex}].channels[${channelIndex}].target.path`
              : `animations[${animationIndex}].channels[${channelIndex}].target.extensions.KHR_animation_pointer.pointer`,
          );
          continue;
        }
        const definitionError = gltfAnimationPointerTargetDefinitionError(gltf, pointerTarget);
        if (definitionError !== undefined) {
          malformedChannels.push({
            kind: 'pointer-target-undefined',
            path: `animations[${animationIndex}].channels[${channelIndex}].target.extensions.KHR_animation_pointer.pointer`,
            animationIndex,
            channelIndex,
            targetPath,
            pointer: pointerTarget.pointer,
            reason: definitionError,
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
        const pointerAccessorError = gltfAnimationPointerOutputAccessorError(
          gltf,
          pointerTarget,
          gltf.accessors?.[sampler.output],
        );
        if (pointerAccessorError !== undefined) {
          malformedChannels.push({
            kind: 'invalid-pointer-output-accessor',
            path: `animations[${animationIndex}].samplers[${channel.sampler}].output`,
            animationIndex,
            channelIndex,
            targetPath,
            samplerIndex: channel.sampler,
            accessorIndex: sampler.output,
            accessorRole: 'output',
            pointer: pointerTarget.pointer,
            reason: pointerAccessorError,
          });
          continue;
        }
        const pointerInterpolationError = gltfAnimationPointerInterpolationError(
          pointerTarget,
          (sampler.interpolation ?? 'LINEAR') as 'LINEAR' | 'STEP' | 'CUBICSPLINE',
        );
        if (pointerInterpolationError !== undefined) {
          malformedChannels.push({
            kind: 'invalid-pointer-interpolation',
            path: `animations[${animationIndex}].samplers[${channel.sampler}].interpolation`,
            animationIndex,
            channelIndex,
            targetPath,
            samplerIndex: channel.sampler,
            pointer: pointerTarget.pointer,
            reason: pointerInterpolationError,
          });
          continue;
        }
        const outputCountIssue = animationOutputCountIssue(
          gltf,
          animationIndex,
          channelIndex,
          channel,
          sampler,
        );
        if (outputCountIssue !== undefined) {
          malformedChannels.push(outputCountIssue);
          continue;
        }
        const pointerIdentity = gltfAnimationPointerTargetIdentity(pointerTarget);
        if (claimedTargets.some((claimed) => gltfAnimationTargetsConflict(claimed, pointerIdentity))) {
          malformedChannels.push({
            kind: 'duplicate-animation-target',
            path: `animations[${animationIndex}].channels[${channelIndex}].target`,
            animationIndex,
            channelIndex,
            targetPath,
            pointer: pointerTarget.pointer,
          });
          continue;
        }
        claimedTargets.push(pointerIdentity);
        if (
          pointerTarget.kind === 'node' ||
          pointerTarget.kind === 'node-weight' ||
          pointerTarget.kind === 'node-visibility'
        ) {
          targetNodes.add(pointerTarget.nodeIndex);
        }
        continue;
      }
      if (
        sceneScope !== undefined &&
        (channel.target.node === undefined || !sceneScope.nodeIndices.has(channel.target.node))
      ) {
        continue;
      }
      animationHasReachableChannel = true;
      channelCount += 1;
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
      if (outputCountIssue !== undefined) {
        malformedChannels.push(outputCountIssue);
        continue;
      }
      const nativeIdentity = gltfNativeAnimationTargetIdentity(channel.target.node, targetPath);
      if (claimedTargets.some((claimed) => gltfAnimationTargetsConflict(claimed, nativeIdentity))) {
        malformedChannels.push({
          kind: 'duplicate-animation-target',
          path: `animations[${animationIndex}].channels[${channelIndex}].target`,
          animationIndex,
          channelIndex,
          targetPath,
          nodeIndex: channel.target.node,
        });
        continue;
      }
      claimedTargets.push(nativeIdentity);
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
  const { animationIndex, channelIndex, channel, accessorRole, accessorIndex, accessor } = input;
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
    const storageIssue = accessorDeclaredStorageIssue(gltf, accessorIndex, accessor);
    if (storageIssue !== undefined) {
      issues.push({
        kind: kindPrefix === 'input' ? 'truncated-input-buffer-view' : 'truncated-output-buffer-view',
        path: storageIssue.path,
        animationIndex,
        channelIndex,
        targetPath: channel.target.path,
        samplerIndex: channel.sampler,
        ...(channel.target.node !== undefined ? { nodeIndex: channel.target.node } : {}),
        accessorIndex,
        accessorRole,
        bufferViewIndex: storageIssue.bufferViewIndex,
        requiredByteLength: storageIssue.requiredByteLength,
        byteLength: storageIssue.byteLength,
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
  if (targetPath === 'pointer') {
    const pointerTarget = resolveGltfAnimationPointer(
      channel.target.extensions?.KHR_animation_pointer?.pointer,
    );
    if (pointerTarget !== undefined) {
      const expectedStride = inputAccessor.count * cubicFactor;
      const components = gltfAnimationPointerTargetComponentCount(gltf, pointerTarget);
      if (components === undefined) return undefined;
      const expectedOutputFloats = expectedStride * components;
      const validOutputCount = actualOutputFloats === expectedOutputFloats;
      if (validOutputCount) return undefined;
      return {
        kind: 'invalid-output-count',
        path: `animations[${animationIndex}].channels[${channelIndex}].sampler`,
        animationIndex,
        channelIndex,
        targetPath,
        samplerIndex: channel.sampler,
        expectedOutputFloats,
        actualOutputFloats,
      };
    }
  }
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
  for (const [animationIndex, animation] of (gltf.animations ?? []).entries()) {
    for (const [channelIndex, channel] of (animation.channels ?? []).entries()) {
      if (channel.target.path !== 'pointer') continue;
      const target = resolveGltfAnimationPointer(
        channel.target.extensions?.KHR_animation_pointer?.pointer,
      );
      if (target === undefined || !isGltfAnimationPointerTargetReachable(target, sceneScope)) continue;
      out.add('KHR_animation_pointer');
      addSourcePath(
        sourcePaths,
        'KHR_animation_pointer',
        `animations[${animationIndex}].channels[${channelIndex}].target.extensions.KHR_animation_pointer`,
      );
    }
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
    return componentType !== undefined && componentType !== GltfComponentType.FLOAT;
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
    for (const key of Object.keys(ext)) {
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
