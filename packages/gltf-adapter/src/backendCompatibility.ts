// backendCompatibility.ts — evaluate/rank/recommend a glTF feature report against
// each backend profile. Extracted from featureReport.ts (T3-E / D15-1). The
// ~470-line flat evaluator was refactored into per-category emitter functions
// (D15-2) that share a mutable CompatibilityEmitContext — SAME issues, SAME order.
// featureReport.ts re-exports evaluateGltfBackendCompatibility /
// evaluateGltfBackendProfileCompatibility / rankGltfBackends / recommendGltfBackend
// from here.

import {
  BACKEND_PROMISE_LEDGER,
  type BackendId,
  type BackendSupportMode,
  type MaterialSpec,
} from '@vitrum/core';
import { TEXTURE_SOURCE_EXTENSIONS } from './assetInventory.js';
import {
  animationMalformedChannelMessage,
  malformedPrimitiveMessage,
  malformedSamplerPolicyMessage,
  materialTextureReferenceIssueMessage,
  materialTextureReferenceIssueSupport,
  materialVariantMappingIssueMessage,
  primitiveAccessorImportIssueMessage,
  primitiveAccessorStorageIssueMessage,
  primitiveInstancingIssueMessage,
  primitiveMaterialReferenceIssueMessage,
  punctualLightIssueMessage,
  samplerPolicySupport,
} from './compatibilityMessages.js';
import type {
  GltfBackendCompatibility,
  GltfBackendPolicy,
  GltfBackendProfileId,
  GltfBackendTraceTier,
  GltfCompatibilityIssue,
  GltfFeatureReport,
} from './featureReport.types.js';

export const VERTEX_COLOR_SUPPORT: Readonly<Record<GltfBackendProfileId, BackendSupportMode>> = Object.freeze({
  'pt-webgl2': 'native',
  'pt-webgpu': 'native',
  'pt-webgpu-lite': 'unsupported',
  'walkaround-hybrid': 'approximate',
});

/** Exported for the capability-drift pin-test. The lite trace tier composes no
 *  full-tier group-3 material texture bindings, so every entry here must be a
 *  valid `keyof MaterialSpec` AND must be one that the FULL pt-webgpu profile does
 *  NOT mark unsupported (the lite tier is what restricts them). See the pin. */
export const PT_WEBGPU_LITE_UNSUPPORTED_MATERIAL_FIELDS = [
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

export function evaluateGltfBackendCompatibility(
  report: GltfFeatureReport,
  backend: BackendId,
): GltfBackendCompatibility {
  return evaluateGltfBackendProfileCompatibility(report, backend);
}

interface CompatibilityEmitContext {
  readonly report: GltfFeatureReport;
  readonly profile: GltfBackendProfile;
  readonly backend: BackendId;
  readonly ledger: (typeof BACKEND_PROMISE_LEDGER)[BackendId];
  readonly issues: GltfCompatibilityIssue[];
  readonly counts: { unsupported: number; approximate: number; native: number; requiresHook: number };
  readonly addIssue: (issue: GltfCompatibilityIssue) => void;
}

function emitExtensionIssues(ctx: CompatibilityEmitContext): void {
  const { report, addIssue } = ctx;
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
      message: `glTF extension "${ext}" requires built-in or host-supplied decode support.`,
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
}

function emitPrimitiveIssues(ctx: CompatibilityEmitContext): void {
  const { report, profile, ledger, addIssue, counts } = ctx;
  const { backend } = profile;
  const profileId = profile.id;
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
      counts.native += 1;
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
        'glTF morph-target UV deltas require a matching primitive TEXCOORD_N base stream. ' +
        'All authored UV-set indices are representable; only deltas whose base stream is absent ' +
        'remain unsupported and are reported explicitly.',
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
      counts.native += 1;
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

}

function emitSceneIssues(ctx: CompatibilityEmitContext): void {
  const { report, addIssue } = ctx;
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
}

function emitAnimationIssues(ctx: CompatibilityEmitContext): void {
  const { report, addIssue } = ctx;
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
      message: targetPath === 'pointer'
        ? 'glTF KHR_animation_pointer channel targets an unsupported JSON pointer; supported material-field pointers are imported, other mutable asset properties remain unsupported.'
        : `glTF animation target path "${targetPath}" is not imported into the core animation controller; ` +
          'supported target paths are translation, rotation, scale, weights, and supported KHR_animation_pointer material-field pointers.',
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
}

function emitMaterialIssues(ctx: CompatibilityEmitContext): void {
  const { report, profile, ledger, addIssue, counts } = ctx;
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

  for (const uvSet of report.materials.uvSets) {
    if (uvSet <= 1) continue;
    if (!report.materials.unrepresentableUvSets.includes(uvSet)) continue;
    addIssue({
      category: 'material',
      name: `TEXCOORD_${uvSet}`,
      support: 'unsupported',
      path: firstSourcePath(report.materials.issuePaths, `uvSet:${uvSet}`, 'materials'),
      message:
        `glTF material textures reference TEXCOORD_${uvSet}, but a primitive using the material ` +
        `does not provide the matching TEXCOORD_${uvSet} base stream.`,
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
      counts.native += 1;
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
}

export function evaluateGltfBackendProfileCompatibility(
  report: GltfFeatureReport,
  profileId: GltfBackendProfileId,
): GltfBackendCompatibility {
  const profile = BACKEND_PROFILES[profileId];
  const { backend } = profile;
  const ledger = BACKEND_PROMISE_LEDGER[backend];
  const issues: GltfCompatibilityIssue[] = [];
  const counts = { unsupported: 0, approximate: 0, native: 0, requiresHook: 0 };

  const addIssue = (issue: GltfCompatibilityIssue): void => {
    issues.push(issue);
    if (issue.support === 'unsupported') counts.unsupported += 1;
    else if (issue.support === 'approximate' || issue.support === 'fallback-generated-mesh' || issue.support === 'fallback-rebuild') {
      counts.approximate += 1;
    } else if (issue.support === 'native') {
      counts.native += 1;
    } else if (issue.support === 'requires-hook') {
      counts.requiresHook += 1;
    }
  };

  const ctx: CompatibilityEmitContext = { report, profile, backend, ledger, issues, counts, addIssue };

  emitExtensionIssues(ctx);
  emitPrimitiveIssues(ctx);
  emitSceneIssues(ctx);
  emitAnimationIssues(ctx);
  emitMaterialIssues(ctx);

  return {
    backend,
    profileId: profile.id,
    ...(profile.traceTier !== undefined ? { traceTier: profile.traceTier } : {}),
    unsupportedCount: counts.unsupported,
    approximateCount: counts.approximate,
    nativeCount: counts.native,
    requiresHookCount: counts.requiresHook,
    issues,
    isCompatible: counts.unsupported === 0,
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
