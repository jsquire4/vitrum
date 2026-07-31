/**
 * Exact live-input compatibility key for persisted renderer state.
 *
 * Temporal reservoirs, DDGI probes, PPG trees, and NRC weights are meaningful
 * only for the scene/light/environment that trained them. Dimensions and an
 * AABB are not sufficient: two structurally different scenes can share both.
 * This key hashes every CPU mirror that feeds the realtime estimator into
 * independent words, so import can fail closed before publishing any state.
 *
 * This is an accidental-mismatch guard, not an authenticity primitive. Hosts
 * loading untrusted snapshot bytes must apply their own integrity/authentication
 * layer before deserialization.
 */

import { fingerprintBuffersExact } from '@vitrum/shared-bvh';
import type { DirectionalEnvData } from './environment/equirectDirectional.js';
import type { SceneBVHBuffers } from './restir/bvhTypes.js';
import {
  materialTextureAtlasFingerprintParts,
  type MaterialTextureAtlasPayload,
} from './bvh/materialTextureAtlasPack.js';

export const GI_STATE_COMPATIBILITY_WORDS = 16;
export const GI_STATE_COMPATIBILITY_SCHEMA = 0x47534331; // "GSC1"
const EMPTY_WORD = new Uint32Array([0]);

export interface GIStateCompatibilityInputs {
  readonly bvh: SceneBVHBuffers;
  readonly directionalEnvironment?: DirectionalEnvData;
  readonly environmentRotationY: number;
  readonly environmentIntensity: number;
  readonly primaryLightDirection: readonly [number, number, number];
  readonly primaryLightIntensity: number;
  readonly skyTint: readonly [number, number, number];
  readonly skyIrradiance: number;
  /**
   * Canonical bytes for construction/runtime estimator configuration not
   * already represented by scene buffers (PPG/NRC/RC/DDGI/reuse knobs).
   */
  readonly estimatorConfiguration: ArrayBuffer | ArrayBufferView;
}

/** Build the fixed-width compatibility key for the currently published scene. */
export function makeGIStateCompatibility(
  inputs: GIStateCompatibilityInputs,
): Uint32Array {
  const { bvh } = inputs;
  const atlas = bvh.materialTextureAtlas;
  const tlas = bvh.tlas;
  const directional = inputs.directionalEnvironment;
  const result = new Uint32Array(GI_STATE_COMPATIBILITY_WORDS);
  result[0] = GI_STATE_COMPATIBILITY_SCHEMA;
  result[1] = fingerprintBuffersExact(
    new Uint32Array([
      bvh.bvhMode === 'tlas' ? 1 : 0,
      bvh.bvhNodes.count,
      bvh.bvhIndex.count,
      bvh.bvhPositions.count,
      bvh.bvhNormals.count,
      bvh.emitters.count,
      bvh.emitterCount,
      bvh.lightTreeNodeCount,
      bvh.lightTreeEnabled ? 1 : 0,
      tlas?.nodeCount ?? 0,
      atlas.atlasLayers.length,
      atlas.atlasLayers.reduce((sum, layer) => sum + layer.width, 0),
      atlas.atlasLayers.reduce((sum, layer) => sum + layer.height, 0),
    ]),
  );
  result[2] = fingerprintBuffersExact(bvh.bvhNodes.cpuData);
  result[3] = fingerprintBuffersExact(bvh.bvhIndex.cpuData);
  result[4] = fingerprintBuffersExact(bvh.bvhPositions.cpuData);
  result[5] = fingerprintBuffersExact(
    bvh.bvhNormals.cpuData,
    bvh.bvhTangents.cpuData,
    bvh.bvhColors.cpuData,
  );
  result[6] = fingerprintBuffersExact(
    bvh.triangleMaterialIds.cpuData,
    bvh.bvhBeerColors.cpuData,
    bvh.bvhRoughMetal.cpuData,
    bvh.bvhEmissiveLe.cpuData,
  );
  result[7] = fingerprintBuffersExact(
    atlas.baseColorMetaData,
    materialAtlasMetadata(atlas),
    ...materialTextureAtlasFingerprintParts(atlas),
  );
  result[8] = tlas == null
    ? 0
    : fingerprintBuffersExact(
        tlas.nodes.cpuData,
        tlas.instanceIndices.cpuData,
        tlas.blasRoots.cpuData,
        tlas.worldToLocal.cpuData,
        tlas.localToWorld.cpuData,
      );
  result[9] = fingerprintBuffersExact(bvh.emitters.cpuData);
  result[10] = fingerprintBuffersExact(
    bvh.emitterCdf.cpuData,
    bvh.emitterAlias.cpuData,
  );
  result[11] = fingerprintBuffersExact(bvh.lightTree.cpuData);
  result[12] = directional == null
    ? 0
    : fingerprintBuffersExact(directional.map);
  result[13] = directional == null
    ? 0
    : fingerprintBuffersExact(
        directional.pdf,
        directional.marginal,
        directional.conditional,
      );
  result[14] = fingerprintBuffersExact(
    directional == null
      ? EMPTY_WORD
      : new Uint32Array([directional.width, directional.height]),
    new Float32Array([
      directional?.totalWeight ?? 0,
      inputs.environmentRotationY,
      directional == null ? 0 : inputs.environmentIntensity,
    ]),
  );
  result[15] = fingerprintBuffersExact(
    new Float32Array([
      inputs.primaryLightDirection[0],
      inputs.primaryLightDirection[1],
      inputs.primaryLightDirection[2],
      inputs.primaryLightIntensity,
      inputs.skyTint[0],
      inputs.skyTint[1],
      inputs.skyTint[2],
      inputs.skyIrradiance,
    ]),
    inputs.estimatorConfiguration,
  );
  return result;
}

export function isValidGIStateCompatibility(
  value: unknown,
): value is Uint32Array {
  return (
    value instanceof Uint32Array &&
    value.length === GI_STATE_COMPATIBILITY_WORDS &&
    value[0] === GI_STATE_COMPATIBILITY_SCHEMA
  );
}

export function giStateCompatibilityMatches(
  snapshot: unknown,
  live: unknown,
): boolean {
  if (
    !isValidGIStateCompatibility(snapshot) ||
    !isValidGIStateCompatibility(live)
  ) {
    return false;
  }
  for (let index = 0; index < GI_STATE_COMPATIBILITY_WORDS; index += 1) {
    if (snapshot[index] !== live[index]) return false;
  }
  return true;
}

function materialAtlasMetadata(
  atlas: MaterialTextureAtlasPayload,
): Uint32Array {
  return new Uint32Array([
    atlas.atlasLayers.length,
    atlas.atlasLayers.reduce(
      (maximum, layer) => Math.max(maximum, layer.mipLevelCount),
      0,
    ),
    atlas.atlasLayers.reduce(
      (sum, layer) => sum + layer.width * layer.height,
      0,
    ),
    atlas.gpuSourceLayers.length,
    atlas.baseColorMetaWidth,
    atlas.baseColorMetaHeight,
    atlas.readableBaseColorLayerCount,
    atlas.readableNormalLayerCount,
    atlas.readableRoughnessLayerCount,
    atlas.readableMetallicLayerCount,
    atlas.readableAoLayerCount,
    atlas.readableAlphaLayerCount,
    atlas.readableEmissiveLayerCount,
    atlas.readableTransmissionLayerCount,
    atlas.readableLightLayerCount,
    atlas.readableSpecularColorLayerCount,
    atlas.readableSpecularIntensityLayerCount,
    atlas.readableClearcoatLayerCount,
    atlas.readableClearcoatRoughnessLayerCount,
    atlas.readableClearcoatNormalLayerCount,
    atlas.readableSheenColorLayerCount,
    atlas.readableSheenRoughnessLayerCount,
    atlas.readableAnisotropyLayerCount,
    atlas.readableIridescenceLayerCount,
    atlas.readableIridescenceThicknessLayerCount,
    atlas.readableThicknessLayerCount,
    atlas.readableBumpLayerCount,
  ]);
}
