/**
 * C2 — CPU mirror of ReSTIR `SceneBVHBuffers` for subsystems that ray-cast
 * the same BLAS/TLAS (DDGI probe update, RC cascades).
 */

import type { MaterialSpec, Scene } from '@vitrum/core';
import {
  computeWorldAabbForBindings,
  fingerprintBuffers,
  isTlasOnlyVersionBump,
} from '@vitrum/shared-bvh';
import type { SceneBVHBuffers } from './bvhTypes.js';
import type { MaterialTextureAtlasPayload } from '../pipeline/materialTextureAtlas.js';
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
  const blasContentVersion = fingerprintBuffers(
    buffers.bvhNodes.cpuData,
    buffers.bvhPositions.cpuData,
    buffers.bvhIndex.cpuData,
    buffers.emitterNormals.buffer as ArrayBuffer,
    buffers.bvhTangents.cpuData,
    buffers.triangleMaterialIds.cpuData,
  );
  const tlasContentVersion = tlas != null
    ? fingerprintBuffers(
        tlas.nodes.cpuData,
        tlas.instanceIndices.cpuData,
        tlas.blasRoots.cpuData,
        tlas.worldToLocal.cpuData,
        tlas.localToWorld.cpuData,
      )
    : 0;

  return {
    bvhMode: buffers.bvhMode,
    tlasNodeCount: tlas?.nodeCount ?? 0,
    bvhNodes: buffers.bvhNodes.cpuData,
    positions: buffers.bvhPositions.cpuData,
    bvhIndex: buffers.bvhIndex.cpuData,
    normals: buffers.emitterNormals.buffer as ArrayBuffer,
    tangents: buffers.bvhTangents.cpuData,
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
    contentVersion: fingerprintBuffers(
      new Uint32Array([blasContentVersion, tlasContentVersion]).buffer,
    ),
    blasContentVersion,
    tlasContentVersion,
  };
}

/** True when only TLAS nodes / instance transforms changed (transform-only refit). */
export function isRestirTlasOnlyRefit(
  snap: RestirBvhSnapshot,
  prev: { readonly blasContentVersion: number; readonly tlasContentVersion: number },
): boolean {
  return (
    snap.tlas != null &&
    isTlasOnlyVersionBump(snap.blasContentVersion, snap.tlasContentVersion, prev)
  );
}
