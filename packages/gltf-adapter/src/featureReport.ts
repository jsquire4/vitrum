// featureReport.ts — structured glTF asset inventory + backend compatibility.
//
// This is intentionally a pure JSON walk. It does not decode buffers, fetch
// resources, or mutate the asset. The goal is to classify what an asset asks
// for before a backend is selected, so hosts can choose strict/fidelity/realtime
// policy with data instead of guessing from triangle count alone.
//
// T3-E / D15-1: the implementation was split into four modules for concern
// separation. THIS FILE IS NOW A BACK-COMPAT RE-EXPORT BARREL — every symbol that
// was previously importable from `./featureReport.js` is re-exported here so all
// existing imports keep working unchanged:
//   - `./featureReport.types.js`      — all interface/union type declarations
//   - `./assetInventory.js`           — the analyze* asset-inventory walk
//   - `./backendCompatibility.js`     — evaluate / rank / recommend + per-category emitters
//   - `./compatibilityMessages.js`    — per-issue message + support formatters (internal)

export type {
  AnalyzeGltfAssetOptions,
  GltfAnimationFeatureReport,
  GltfAnimationMalformedChannelIssue,
  GltfAnimationMalformedChannelKind,
  GltfBackendCompatibility,
  GltfBackendPolicy,
  GltfBackendProfileId,
  GltfBackendTraceTier,
  GltfCompatibilityIssue,
  GltfExtensionReport,
  GltfFeatureReport,
  GltfMalformedPrimitiveIssue,
  GltfMalformedPrimitiveKind,
  GltfMalformedTextureSamplerKind,
  GltfMalformedTextureSamplerPolicyUse,
  GltfMaterialFeatureReport,
  GltfMaterialTextureReferenceIssue,
  GltfMaterialTextureReferenceIssueKind,
  GltfMaterialVariantMappingIssue,
  GltfMaterialVariantMappingIssueKind,
  GltfPrimitiveAccessorImportIssue,
  GltfPrimitiveAccessorImportIssueKind,
  GltfPrimitiveAccessorStorageIssue,
  GltfPrimitiveFeatureReport,
  GltfPrimitiveInstancingIssue,
  GltfPrimitiveInstancingIssueKind,
  GltfPrimitiveMaterialReferenceIssue,
  GltfPunctualLightIssue,
  GltfPunctualLightIssueKind,
  GltfResourceFeatureReport,
  GltfResourceKind,
  GltfResourceUse,
  GltfSceneGraphFeatureReport,
  GltfSparseAccessorStorageIssueKind,
  GltfTextureSamplerFilterMode,
  GltfTextureSamplerMipMode,
  GltfTextureSamplerPolicyUse,
  GltfTextureSourceExtensionName,
  GltfTextureSourceExtensionUse,
} from './featureReport.types.js';

export { analyzeGltfAsset } from './assetInventory.js';

export {
  PT_WEBGPU_LITE_UNSUPPORTED_MATERIAL_FIELDS,
  VERTEX_COLOR_SUPPORT,
  evaluateGltfBackendCompatibility,
  evaluateGltfBackendProfileCompatibility,
  rankGltfBackends,
  recommendGltfBackend,
} from './backendCompatibility.js';
