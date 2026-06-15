// @vitrum/gltf-adapter — glTF 2.0 → @vitrum/core Scene adapter.
//
// Primary entry point: gltfToScene.
// animationNodeId builds the stable channel-target node id (`gltf-node-<i>`)
// used by result.animations / result.animationTargets.
// Re-exports the GltfJson type for hosts that parse glTF JSON before passing it in.
// RawImageHandle is exported so hosts can type-narrow in non-browser environments.

export { gltfToScene } from './gltfToScene.js';
export type {
  GltfMaterialVariantBinding,
  GltfToSceneOptions,
  GltfToSceneResult,
} from './gltfToScene.js';
export { loadGltfAndDecodeTextures, loadGltfAsset } from './assetLoader.js';
export type {
  GltfAssetCache,
  GltfAssetCacheKey,
  GltfAssetFetch,
  GltfAssetFetchResponse,
  GltfAssetInput,
  GltfAssetResult,
  GltfDecodedAssetResult,
  LoadGltfAndDecodeTexturesOptions,
  LoadGltfAssetOptions,
} from './assetLoader.js';
export {
  GltfAdapterError,
  GltfFetchFailed,
  GltfResourceNotFound,
} from './errors.js';
export type { GltfAssetResourceKind } from './errors.js';
export {
  buildTextureDecodeReport,
  classifyTextureHandle,
  decodeSceneTextures,
} from './texturePipeline.js';
export type {
  DecodeGltfTexturePixelsFn,
  DecodeSceneTextureDiagnostic,
  DecodeSceneTextureDiagnosticCode,
  DecodeSceneTexturesOptions,
  DecodeSceneTexturesResult,
  GltfCpuLinearTextureHandle,
  GltfDecodedTexturePixels,
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
  GltfPrimitivePatchRecord,
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
  GltfTextureSourceExtensionName,
  GltfTextureSourceExtensionUse,
} from './featureReport.js';
export { animationNodeId } from './animations.js';
export type { GltfJson } from './gltfTypes.js';
export { GLTF_TEXTURE_SOURCE_EXTENSIONS } from './textures.js';
export type {
  DecodeImageFn,
  GltfImageBytes,
  GltfImageBytesMap,
  GltfTextureSourceExtension,
  RawImageHandle,
} from './textures.js';
// Compressed-geometry decoder hook contract (GLTF-02): the host injects
// KHR_draco_mesh_compression / EXT_meshopt_compression decoders; the package
// itself bundles none. See README "Compressed geometry".
export type {
  GltfDecodeHooks,
  DracoDecodeFn,
  DracoDecodeResult,
  DracoTypedArray,
  MeshoptDecodeFn,
  MeshoptMode,
  MeshoptFilter,
} from './compression.js';
