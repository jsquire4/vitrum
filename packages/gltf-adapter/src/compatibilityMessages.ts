// compatibilityMessages.ts — per-issue human-readable message + support
// formatters for the backend-compatibility evaluator. Extracted verbatim from
// featureReport.ts (T3-E / D15-1); featureReport.ts re-exports nothing from here
// (these are internal helpers consumed by backendCompatibility.ts).

import type { BackendSupportMode } from '@vitrum/core';
import type {
  GltfAnimationMalformedChannelIssue,
  GltfBackendProfileId,
  GltfMalformedPrimitiveIssue,
  GltfMalformedTextureSamplerPolicyUse,
  GltfMaterialTextureReferenceIssue,
  GltfMaterialVariantMappingIssue,
  GltfPrimitiveAccessorImportIssue,
  GltfPrimitiveAccessorStorageIssue,
  GltfPrimitiveInstancingIssue,
  GltfPrimitiveMaterialReferenceIssue,
  GltfPunctualLightIssue,
  GltfSparseAccessorStorageIssueKind,
  GltfTextureSamplerPolicyUse,
} from './featureReport.types.js';

export function samplerPolicySupport(
  profileId: GltfBackendProfileId,
  policy: GltfTextureSamplerPolicyUse,
): BackendSupportMode {
  if (profileId === 'walkaround-hybrid') {
    // The material atlas implements independent nearest/linear mag/min modes
    // plus none/nearest/linear mip selection. Omitted mip policy resolves to
    // the same library-wide linear default as the runtime packer.
    return 'native';
  }
  if (profileId === 'pt-webgl2') return 'native';
  if (policy.materialField === 'bumpMap') {
    if ((policy.magFilter ?? 'linear') !== 'linear') return 'approximate';
    if ((policy.minFilter ?? 'linear') !== 'linear') return 'approximate';
    if (policy.mipFilter !== undefined && policy.mipFilter !== 'none') {
      return 'approximate';
    }
  }
  return 'native';
}

export function malformedSamplerPolicyMessage(
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

export function materialTextureReferenceIssueSupport(
  issue: GltfMaterialTextureReferenceIssue,
): BackendSupportMode | 'requires-hook' {
  return issue.kind === 'disabled-texture-source-extension' ? 'requires-hook' : 'approximate';
}

export function materialTextureReferenceIssueMessage(
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

export function materialVariantMappingIssueMessage(issue: GltfMaterialVariantMappingIssue): string {
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

export function primitiveMaterialReferenceIssueMessage(issue: GltfPrimitiveMaterialReferenceIssue): string {
  return (
    `glTF mesh ${issue.meshIndex} primitive ${issue.primitiveIndex} references missing ` +
    `material ${String(issue.materialIndex)}; the importer falls back to the default material.`
  );
}

export function animationMalformedChannelMessage(issue: GltfAnimationMalformedChannelIssue): string {
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
  if (issue.kind === 'truncated-input-buffer-view' || issue.kind === 'truncated-output-buffer-view') {
    return (
      `glTF animation channel ${issue.animationIndex}:${issue.channelIndex} references sampler ` +
      `${String(issue.samplerIndex)} ${String(issue.accessorRole)} accessor ${String(issue.accessorIndex)} ` +
      `whose bufferView ${String(issue.bufferViewIndex)} declares ${String(issue.byteLength)} bytes ` +
      `but ${String(issue.requiredByteLength)} are required; the importer skips the channel.`
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
  if (issue.kind === 'pointer-target-undefined') {
    return (
      `glTF animation channel ${issue.animationIndex}:${issue.channelIndex} pointer ` +
      `"${String(issue.pointer)}" does not resolve to a defined asset property (${String(issue.reason)}); ` +
      'the importer skips the channel.'
    );
  }
  if (issue.kind === 'invalid-pointer-output-accessor') {
    return (
      `glTF animation channel ${issue.animationIndex}:${issue.channelIndex} pointer ` +
      `"${String(issue.pointer)}" has an incompatible output accessor (${String(issue.reason)}); ` +
      'the importer skips the channel.'
    );
  }
  if (issue.kind === 'invalid-pointer-interpolation') {
    return (
      `glTF animation channel ${issue.animationIndex}:${issue.channelIndex} pointer ` +
      `"${String(issue.pointer)}" uses invalid interpolation (${String(issue.reason)}); ` +
      'the importer skips the channel.'
    );
  }
  if (issue.kind === 'duplicate-animation-target') {
    return (
      `glTF animation channel ${issue.animationIndex}:${issue.channelIndex} targets a property already ` +
      'targeted by an earlier channel in the same animation; the importer deterministically keeps the first channel.'
    );
  }
  return (
    `glTF animation channel ${issue.animationIndex}:${issue.channelIndex} has ` +
    `${String(issue.actualOutputFloats)} output floats but the target path expects ` +
    `${String(issue.expectedOutputFloats)} for the authored keyframes; the importer skips the channel.`
  );
}

export function malformedPrimitiveMessage(issue: GltfMalformedPrimitiveIssue): string {
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
  if (issue.kind === 'truncated-position-buffer-view') {
    return (
      `${label} references POSITION accessor ${String(issue.accessorIndex)} whose bufferView ` +
      `${String(issue.bufferViewIndex)} declares ${String(issue.byteLength)} bytes but ` +
      `${String(issue.requiredByteLength)} are required; the importer skips the primitive.`
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
  if (issue.kind === 'truncated-index-buffer-view') {
    return (
      `${label} references index accessor ${String(issue.accessorIndex)} whose bufferView ` +
      `${String(issue.bufferViewIndex)} declares ${String(issue.byteLength)} bytes but ` +
      `${String(issue.requiredByteLength)} are required; the importer skips the primitive.`
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
    readonly requiredByteLength?: number;
    readonly byteLength?: number;
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
  if (issue.sparseIssueKind === 'truncated-sparse-indices-buffer-view') {
    return (
      `sparse indices bufferView ${String(issue.bufferViewIndex)} declares ` +
      `${String(issue.byteLength)} bytes but ${String(issue.requiredByteLength)} are required`
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
  if (issue.sparseIssueKind === 'truncated-sparse-values-buffer-view') {
    return (
      `sparse values bufferView ${String(issue.bufferViewIndex)} declares ` +
      `${String(issue.byteLength)} bytes but ${String(issue.requiredByteLength)} are required`
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

export function primitiveAccessorStorageIssueMessage(issue: GltfPrimitiveAccessorStorageIssue): string {
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

export function primitiveAccessorImportIssueMessage(issue: GltfPrimitiveAccessorImportIssue): string {
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
  if (issue.kind === 'truncated-buffer-view') {
    return (
      `${location} ${issue.semantic} accessor ${issue.accessorIndex} uses bufferView ` +
      `${String(issue.bufferViewIndex)} declaring ${String(issue.byteLength)} bytes, but ` +
      `${String(issue.requiredByteLength)} are required; ${consequence}`
    );
  }
  return (
    `${location} ${issue.semantic} accessor ${issue.accessorIndex} uses bufferView ` +
    `${String(issue.bufferViewIndex)} referencing missing buffer ${String(issue.bufferIndex)}; ${consequence}`
  );
}

export function primitiveInstancingIssueMessage(issue: GltfPrimitiveInstancingIssue): string {
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

export function punctualLightIssueMessage(issue: GltfPunctualLightIssue): string {
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
