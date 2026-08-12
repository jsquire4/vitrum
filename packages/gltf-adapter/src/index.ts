// @vitrum/gltf-adapter — glTF 2.0 → @vitrum/core Scene adapter.
//
// Primary entry point: gltfToScene.
// animationNodeId builds the stable channel-target node id (`gltf-node-<i>`)
// used by result.animations / result.animationTargets.
// Re-exports the GltfJson type for hosts that parse glTF JSON before passing it in.
// RawImageHandle is exported so hosts can type-narrow in non-browser environments.

export { GltfImportError, gltfToScene } from './gltfToScene.js';
export type {
  GltfImportDiagnostic,
  GltfImportDiagnosticCode,
  GltfCompressionDecoderPolicy,
  GltfInstancingBinding,
  GltfMaterialBinding,
  GltfMaterialVariantBinding,
  GltfPunctualEmitterBinding,
  GltfSceneCamera,
  GltfToSceneOptions,
  GltfToSceneResult,
} from './gltfToScene.js';
export { gltfSceneCameraToDescriptor } from './cameraMetadata.js';
export {
  DEFAULT_GLTF_IMPORT_RESOURCE_LIMITS,
  GltfResourceLimitError,
  normalizeGltfImportResourceLimits,
  releaseGltfResources,
} from './importResourceBudget.js';
export type {
  GltfImportResourceLimits,
  GltfResourceLimitErrorInit,
  GltfResourceLimitKind,
  NormalizedGltfImportResourceLimits,
} from './importResourceBudget.js';
export { loadGltfAndDecodeTextures, loadGltfAsset } from './assetLoader.js';
export type {
  GltfAssetCache,
  GltfAssetCacheEntry,
  GltfAssetCacheKey,
  GltfAssetCompatibilityPreflight,
  GltfAssetFetch,
  GltfAssetFetchResponse,
  GltfAssetInput,
  GltfAssetReadableStream,
  GltfAssetReadableStreamReader,
  GltfAssetReadableStreamReadResult,
  GltfAssetResult,
  ConfigureGltfTextureDecodeOptions,
  GltfDecodedAssetResult,
  GltfTextureDecodePolicyContext,
  LoadGltfAndDecodeTexturesOptions,
  LoadGltfAssetOptions,
} from './assetLoader.js';
export {
  GltfAdapterError,
  GltfCompatibilityError,
  GltfFetchFailed,
  GltfParseFailed,
  GltfResourceDecodeFailed,
  GltfResourceNotFound,
} from './errors.js';
export type {
  GltfAssetResourceKind,
  GltfCompatibilityErrorCode,
  GltfCompatibilityErrorInit,
  GltfCompatibilityFailureDetail,
  GltfParseFailedInit,
  GltfParseFailureReason,
  GltfParseFormat,
  GltfResourceDecodeFailedInit,
  GltfResourceDecodeFailureReason,
} from './errors.js';
export {
  buildTextureDecodeReport,
  classifyTextureHandle,
  decodeSceneTextures,
} from './texturePipeline.js';
export {
  resolveGltfMaterialAnimationPointer,
  supportedGltfMaterialAnimationPointers,
} from './materialPointerAnimation.js';
export type {
  GltfMaterialPointerField,
  GltfMaterialPointerTarget,
} from './materialPointerAnimation.js';
export {
  applyGltfMaterialAnimationPointerValue,
  applyGltfMaterialTextureTransformPointerValue,
  gltfAnimationPointerInterpolationError,
  gltfAnimationPointerOutputAccessorError,
  gltfAnimationPointerSampleValueError,
  gltfAnimationPointerTargetComponentCount,
  gltfAnimationPointerTargetDefinitionError,
  gltfAnimationPointerTargetIdentity,
  gltfAnimationPointerValuesError,
  gltfAnimationTargetsConflict,
  gltfNativeAnimationTargetIdentity,
  isGltfAnimationPointerTargetReachable,
  resolveGltfAnimationPointer,
  supportedGltfAnimationPointers,
} from './animationPointer.js';
export type {
  GltfAnimationPointerComponents,
  GltfAnimationPointerReachability,
  GltfAnimationPointerTarget,
  GltfAnimationPointerValueType,
  GltfAnimationTargetIdentity,
  GltfCameraAnimationPointerTarget,
  GltfMaterialPropertyAnimationPointerTarget,
  GltfMaterialTextureRefField,
  GltfMaterialTextureTransformAnimationPointerTarget,
  GltfNodeAnimationPointerTarget,
  GltfNodeVisibilityAnimationPointerTarget,
  GltfNodeWeightElementAnimationPointerTarget,
  GltfOrthographicCameraAnimationField,
  GltfPerspectiveCameraAnimationField,
  GltfPunctualLightAnimationField,
  GltfPunctualLightAnimationPointerTarget,
  GltfTextureTransformAnimationField,
} from './animationPointer.js';
export type {
  DecodeGltfTexturePixelsFn,
  DecodeSceneTextureDiagnostic,
  DecodeSceneTextureDiagnosticCode,
  DecodeSceneTexturesOptions,
  DecodeSceneTexturesResult,
  GltfCpuTextureHandle,
  GltfCpuLinearTextureHandle,
  GltfDecodedTexturePixels,
  GltfNpotRepeatWrapPolicy,
  GltfBackendTextureStatus,
  GltfMaterialTextureField,
  GltfTextureColorSpace,
  GltfTextureDecodeReport,
  GltfTextureDecodeReportEntry,
  GltfTextureHandleKind,
} from './texturePipeline.js';
export { GltfSceneController, createGltfSceneController } from './sceneController.js';
export type {
  GltfAnimationApplyResult,
  GltfApplyAnimationOptions,
  GltfApplyVariantOptions,
  GltfBlendAnimationOptions,
  GltfBlendApplyResult,
  GltfClipSelector,
  GltfPlaybackOptions,
  GltfPrimitivePatchRecord,
  GltfResetPoseOptions,
  GltfSceneControllerDiagnostic,
  GltfSceneControllerDiagnosticCode,
  GltfSceneControllerInput,
  GltfSceneControllerOptions,
  GltfScenePatchTarget,
  GltfVariantApplyResult,
} from './sceneController.js';
export { loadGltfForEngine } from './engineBridge.js';
export type {
  GltfCompatibilityMode,
  GltfEngineFactory,
  GltfEngineFactoryInput,
  GltfEngineSelection,
  GltfForEngineResult,
  LoadGltfForEngineOptions,
} from './engineBridge.js';
export {
  analyzeGltfAsset,
  evaluateGltfBackendCompatibility,
  evaluateGltfBackendProfileCompatibility,
  rankGltfBackends,
  recommendGltfBackend,
} from './featureReport.js';
export type {
  GltfAnimationFeatureReport,
  GltfBackendCompatibility,
  GltfBackendProfileId,
  GltfBackendPolicy,
  GltfBackendTraceTier,
  GltfCompatibilityIssue,
  GltfExtensionReport,
  GltfFeatureReport,
  GltfMaterialFeatureReport,
  GltfPrimitiveFeatureReport,
  GltfResourceFeatureReport,
  GltfResourceKind,
  GltfResourceUse,
  GltfSceneGraphFeatureReport,
  GltfTextureSamplerFilterMode,
  GltfTextureSamplerMipMode,
  GltfTextureSamplerPolicyUse,
  GltfTextureSourceExtensionName,
  GltfTextureSourceExtensionUse,
} from './featureReport.js';
export { animationNodeId } from './animations.js';
export type {
  GltfAnimationImportDiagnostic,
  GltfAnimationImportDiagnosticCode,
} from './animations.js';
export type {
  GltfAccessorDiagnostic,
  GltfAccessorDiagnosticCode,
} from './accessors.js';
export type {
  GltfMaterialDiagnostic,
  GltfMaterialDiagnosticCode,
} from './materials.js';
export type { GltfJson } from './gltfTypes.js';
export { GLTF_TEXTURE_SOURCE_EXTENSIONS } from './textures.js';
export type {
  DecodeImageFn,
  GltfImageBytes,
  GltfImageBytesMap,
  GltfTextureSourceExtension,
  RawImageHandle,
} from './textures.js';
// Compressed-geometry override contracts (GLTF-02). Lazy built-in Draco and
// meshoptimizer decoders are the default; host hooks take precedence.
export type {
  GltfDecodeHooks,
  GltfCompressionDiagnostic,
  GltfCompressionDiagnosticCode,
  DracoDecodeFn,
  DracoDecodeContext,
  DracoAttributeDecodeSchema,
  DracoAccessorComponentType,
  DracoDecodeResult,
  DracoTypedArray,
  MeshoptDecodeFn,
  MeshoptMode,
  MeshoptFilter,
} from './compression.js';

// Shared compatibility-issue vocabulary (I4-2 / D15-8): single source of truth for
// the `texture-readiness:` / `texture-decode:` issue-name prefixes + predicates,
// consumed by engineBridge, @vitrum/engine's gltf.ts, and the reconciler.
export {
  TEXTURE_DECODE_DIAGNOSTIC_ISSUE_PREFIX,
  TEXTURE_READINESS_ISSUE_PREFIX,
  isTextureDecodeDiagnosticIssue,
  isTextureReadinessIssue,
} from './compatibilityIssuePredicates.js';
