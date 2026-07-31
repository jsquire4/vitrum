/**
 * C2 — CPU mirror of ReSTIR `SceneBVHBuffers` for subsystems that ray-cast
 * the same BLAS/TLAS (DDGI probe update, RC cascades).
 */

import type { MaterialSpec, Scene } from '@vitrum/core';
import {
  computeWorldAabbForBindings,
  fingerprintBuffersExact,
  isTlasOnlyVersionBump,
} from '@vitrum/shared-bvh';
import type { SceneBVHBuffers } from './bvhTypes.js';
import {
  materialTextureAtlasFingerprintParts,
  type MaterialTextureAtlasPayload,
} from '../pipeline/materialTextureAtlas.js';
import {
  packDDGIMaterialsFromCoreN,
  packDDGIMaterialsN,
} from '../ddgi/probeUpdateMaterials.js';
import type { PbrScalarSource } from '../pbrScalars.js';
import {
  makeEmptyAabb,
  setAabb,
  copyBoxLike,
  isAabbEmpty,
} from './aabbHelpers.js';

interface RestirBvhVector3 {
  readonly x: number;
  readonly y: number;
  readonly z: number;
}

interface RestirBvhAabb {
  readonly min: RestirBvhVector3;
  readonly max: RestirBvhVector3;
}

export interface RestirBvhSnapshot {
  readonly bvhMode: 'merged' | 'tlas';
  readonly tlasNodeCount: number;
  readonly bvhNodes: ArrayBuffer;
  readonly positions: ArrayBuffer;
  /** Stride-4 triangle index buffer (ReSTIR `bvhIndex`). */
  readonly bvhIndex: ArrayBuffer;
  readonly normals: ArrayBuffer;
  /** Per-vertex authored/generated tangent.xyzw stream. Zero means derive TBN from UVs. */
  readonly tangents: ArrayBuffer;
  /** Per-vertex COLOR_0 rgba stream. Missing authored colors are white-filled upstream. */
  readonly vertexColors: ArrayBuffer;
  readonly triMaterialIds: ArrayBuffer;
  readonly materials: readonly unknown[];
  /**
   * THREE-DECOUPLE of the production ReSTIR MATERIAL path. The deduped core
   * `MaterialSpec[]`, slot-aligned with {@link materials} (THREE) and
   * `triMaterialIds`. Mirrors `SceneBVHBuffers.coreMaterials`. Production DDGI
   * (`probeUpdatePass.ts`) prefers this — packing its per-material struct from
   * core `MaterialSpec`s via `coreMaterialToMaterialEntry`, NO THREE round-trip —
   * and falls back to {@link materials} (THREE) when it is empty (the legacy
   * THREE-only merged build). RC's cascade-material packer is unaffected: it
   * keeps reading {@link materials} (THREE).
   */
  readonly coreMaterials: readonly MaterialSpec[];
  /** Readable material-map atlas payload, slot-aligned with the active BVH triangles. */
  readonly materialTextureAtlas: MaterialTextureAtlasPayload;
  readonly boundingBox: RestirBvhAabb;
  readonly tlas?: {
    readonly nodes: ArrayBuffer;
    readonly instanceIndices: ArrayBuffer;
    readonly blasRoots: ArrayBuffer;
    readonly worldToLocal: ArrayBuffer;
    readonly localToWorld: ArrayBuffer;
  };
  /** Bumps when any mirrored buffer payload changes (not just lengths). */
  readonly contentVersion: number;
  /** BLAS concat buffers only — stable across TLAS transform-only refit. */
  readonly blasContentVersion: number;
  /** TLAS nodes + instance transforms — bumps on transform refit. */
  readonly tlasContentVersion: number;
  /** Scalar material bytes + texture-atlas bytes consumed by DDGI/RC. */
  readonly materialContentVersion: number;
}

function materialAtlasMetadata(atlas: MaterialTextureAtlasPayload): Uint32Array {
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

export function makeRestirBvhSnapshot(
  buffers: SceneBVHBuffers,
  scene?: Scene,
): RestirBvhSnapshot {
  const bbox = makeEmptyAabb();
  if (
    buffers.bvhMode === 'tlas' &&
    scene != null &&
    buffers.primitiveTlasBindings.length > 0
  ) {
    const world = computeWorldAabbForBindings(scene, buffers.primitiveTlasBindings);
    if (world != null) {
      setAabb(
        bbox,
        world.min[0], world.min[1], world.min[2],
        world.max[0], world.max[1], world.max[2],
      );
    }
  }
  if (isAabbEmpty(bbox)) {
    if (buffers.mergedGeometry.boundingBox != null) {
      copyBoxLike(bbox, buffers.mergedGeometry.boundingBox);
    } else {
      buffers.mergedGeometry.computeBoundingBox();
      if (buffers.mergedGeometry.boundingBox != null) {
        copyBoxLike(bbox, buffers.mergedGeometry.boundingBox);
      }
    }
  }

  const tlas = buffers.tlas;
  const blasContentVersion = fingerprintBuffersExact(
    buffers.bvhNodes.cpuData,
    buffers.bvhPositions.cpuData,
    buffers.bvhIndex.cpuData,
    buffers.bvhNormals.cpuData,
    buffers.bvhTangents.cpuData,
    buffers.bvhColors.cpuData,
    buffers.triangleMaterialIds.cpuData,
  );
  const tlasContentVersion = tlas != null
    ? fingerprintBuffersExact(
        tlas.nodes.cpuData,
        tlas.instanceIndices.cpuData,
        tlas.blasRoots.cpuData,
        tlas.worldToLocal.cpuData,
        tlas.localToWorld.cpuData,
      )
    : 0;
  const materialSlots = Math.max(
    1,
    buffers.coreMaterials.length,
    buffers.buildMaterials.length,
  );
  const materialBytes = buffers.coreMaterials.length > 0
    ? packDDGIMaterialsFromCoreN(buffers.coreMaterials, materialSlots)
    : packDDGIMaterialsN(
        buffers.buildMaterials as readonly PbrScalarSource[],
        materialSlots,
      );
  const atlas = buffers.materialTextureAtlas;
  const materialContentVersion = fingerprintBuffersExact(
    materialBytes,
    atlas.baseColorMetaData,
    materialAtlasMetadata(atlas),
    ...materialTextureAtlasFingerprintParts(atlas),
  );

  return {
    bvhMode: buffers.bvhMode,
    tlasNodeCount: tlas?.nodeCount ?? 0,
    bvhNodes: buffers.bvhNodes.cpuData,
    positions: buffers.bvhPositions.cpuData,
    bvhIndex: buffers.bvhIndex.cpuData,
    normals: buffers.bvhNormals.cpuData,
    tangents: buffers.bvhTangents.cpuData,
    vertexColors: buffers.bvhColors.cpuData,
    triMaterialIds: buffers.triangleMaterialIds.cpuData,
    materials: buffers.buildMaterials,
    coreMaterials: buffers.coreMaterials,
    materialTextureAtlas: buffers.materialTextureAtlas,
    boundingBox: bbox,
    ...(tlas != null
      ? {
          tlas: {
            nodes: tlas.nodes.cpuData,
            instanceIndices: tlas.instanceIndices.cpuData,
            blasRoots: tlas.blasRoots.cpuData,
            worldToLocal: tlas.worldToLocal.cpuData,
            localToWorld: tlas.localToWorld.cpuData,
          },
        }
      : {}),
    contentVersion: fingerprintBuffersExact(
      new Uint32Array([
        blasContentVersion,
        tlasContentVersion,
        materialContentVersion,
      ]),
    ),
    blasContentVersion,
    tlasContentVersion,
    materialContentVersion,
  };
}

/** True when only TLAS nodes / instance transforms changed (transform-only refit). */
export function isRestirTlasOnlyRefit(
  snap: RestirBvhSnapshot,
  prev: {
    readonly blasContentVersion: number;
    readonly tlasContentVersion: number;
    readonly materialContentVersion: number;
  },
): boolean {
  return (
    snap.tlas != null &&
    snap.materialContentVersion === prev.materialContentVersion &&
    isTlasOnlyVersionBump(snap.blasContentVersion, snap.tlasContentVersion, prev)
  );
}
