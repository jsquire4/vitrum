// featureReport.types.ts — structured glTF asset inventory + backend
// compatibility type surface. Extracted from featureReport.ts (T3-E / D15-1).
// These are pure type declarations (no runtime code); featureReport.ts re-exports
// them so all existing `./featureReport.js` imports keep working.

import type {
  BackendId,
  BackendSupportMode,
  MaterialSpec,
} from '@vitrum/core';

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
  readonly hasCollapsedSkinInfluenceSets: boolean;
  readonly hasIgnoredSkinAttributes: boolean;
  readonly hasIncompleteSkinAttributes: boolean;
  readonly malformedPrimitives: readonly GltfMalformedPrimitiveIssue[];
  readonly accessorStorageIssues: readonly GltfPrimitiveAccessorStorageIssue[];
  readonly accessorImportIssues: readonly GltfPrimitiveAccessorImportIssue[];
  readonly instancingIssues: readonly GltfPrimitiveInstancingIssue[];
  readonly hasVertexColors: boolean;
  /** @deprecated Always empty: valid COLOR_n streams are preserved losslessly. */
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
  | 'truncated-position-buffer-view'
  | 'invalid-position-sparse-accessor'
  | 'missing-index-accessor'
  | 'invalid-index-accessor'
  | 'missing-index-buffer-view'
  | 'missing-index-buffer'
  | 'truncated-index-buffer-view'
  | 'invalid-index-sparse-accessor'
  | 'empty-triangulated-primitive';

export type GltfSparseAccessorStorageIssueKind =
  | 'missing-sparse-indices-buffer-view'
  | 'missing-sparse-indices-buffer'
  | 'truncated-sparse-indices-buffer-view'
  | 'missing-sparse-values-buffer-view'
  | 'missing-sparse-values-buffer'
  | 'truncated-sparse-values-buffer-view'
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
  readonly requiredByteLength?: number;
  readonly byteLength?: number;
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
  readonly requiredByteLength?: number;
  readonly byteLength?: number;
}

export type GltfPrimitiveAccessorImportIssueKind =
  | 'missing-accessor'
  | 'invalid-accessor-type'
  | 'invalid-accessor-count'
  | 'invalid-accessor-component-type'
  | 'missing-buffer-view'
  | 'missing-buffer'
  | 'truncated-buffer-view';

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
  readonly requiredByteLength?: number;
  readonly byteLength?: number;
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

export type GltfMaterialProfile =
  | 'deltaTransmission'
  | 'roughTransmission'
  | 'layeredTransmission'
  | 'normalMappedTransmission'
  | 'participatingMedia';

export interface GltfMaterialFeatureReport {
  readonly count: number;
  readonly materialFields: readonly (keyof MaterialSpec)[];
  /**
   * Conditional material combinations present in the selected asset scope.
   * These names index `BackendSupportDetails.materialProfiles` during backend
   * compatibility evaluation.
   */
  readonly materialProfiles?: readonly GltfMaterialProfile[];
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
  | 'truncated-input-buffer-view'
  | 'truncated-output-buffer-view'
  | 'invalid-input-sparse-accessor'
  | 'invalid-output-sparse-accessor'
  | 'missing-target-node'
  | 'target-node-not-found'
  | 'pointer-target-undefined'
  | 'invalid-pointer-output-accessor'
  | 'invalid-pointer-interpolation'
  | 'duplicate-animation-target'
  | 'invalid-output-count';

export interface GltfAnimationMalformedChannelIssue {
  readonly kind: GltfAnimationMalformedChannelKind;
  readonly path: string;
  readonly animationIndex: number;
  readonly channelIndex: number;
  readonly targetPath?: string;
  readonly samplerIndex?: number;
  readonly nodeIndex?: number;
  readonly pointer?: string;
  readonly reason?: string;
  readonly accessorIndex?: number;
  readonly accessorRole?: 'input' | 'output';
  readonly accessorType?: string;
  readonly componentType?: number;
  readonly bufferViewIndex?: number;
  readonly bufferIndex?: number;
  readonly sparseIssueKind?: GltfSparseAccessorStorageIssueKind;
  readonly expectedOutputFloats?: number;
  readonly actualOutputFloats?: number;
  readonly requiredByteLength?: number;
  readonly byteLength?: number;
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
export type GltfBackendPolicy = 'fidelity' | 'realtime' | 'strict';

export interface AnalyzeGltfAssetOptions {
  readonly textureSourceExtensions?: readonly GltfTextureSourceExtensionName[];
  /**
   * When supplied, compatibility-affecting primitive/material/scene rows are
   * scoped to the graph reachable from this scene. Omit to keep the historical
   * whole-asset inventory behavior.
   */
  readonly sceneIndex?: number;
}
