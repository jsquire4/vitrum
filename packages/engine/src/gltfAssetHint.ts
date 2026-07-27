import type { CreateEngineBackendId, CreateEngineGltfAssetHint } from './createEngineInternals.js';

interface GltfAssetBackendRecommendation {
  readonly recommendedBackend: {
    readonly backend: CreateEngineBackendId;
  };
}

/**
 * Narrow the adapter's rich asset result to the plain recommendation hint
 * accepted by createEngine/attachVitrum's strict option validator.
 */
export function createEngineGltfAssetHint(
  asset: GltfAssetBackendRecommendation,
): CreateEngineGltfAssetHint {
  return {
    recommendedBackend: {
      backend: asset.recommendedBackend.backend,
    },
  };
}
