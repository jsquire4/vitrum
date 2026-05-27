/**
 * C2 — CPU mirror of ReSTIR `SceneBVHBuffers` for subsystems that ray-cast
 * the same BLAS/TLAS (DDGI probe update, RC cascades).
 */

import * as THREE from 'three';
import type { Scene } from '@vitrum/core';
import {
  computeWorldAabbForBindings,
  fingerprintBuffers,
  isTlasOnlyVersionBump,
} from '@vitrum/shared-bvh';
import type { SceneBVHBuffers } from './bvhCompute.js';

export interface RestirBvhSnapshot {
  readonly bvhMode: 'merged' | 'tlas';
  readonly tlasNodeCount: number;
  readonly bvhNodes: ArrayBuffer;
  readonly positions: ArrayBuffer;
  /** Stride-4 triangle index buffer (ReSTIR `bvhIndex`). */
  readonly bvhIndex: ArrayBuffer;
  readonly normals: ArrayBuffer;
  readonly triMaterialIds: ArrayBuffer;
  readonly materials: readonly THREE.Material[];
  readonly boundingBox: THREE.Box3;
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
  const bbox = new THREE.Box3();
  if (
    buffers.bvhMode === 'tlas' &&
    scene != null &&
    buffers.primitiveTlasBindings.length > 0
  ) {
    const world = computeWorldAabbForBindings(scene, buffers.primitiveTlasBindings);
    if (world != null) {
      bbox.min.set(world.min[0], world.min[1], world.min[2]);
      bbox.max.set(world.max[0], world.max[1], world.max[2]);
    }
  }
  if (bbox.isEmpty()) {
    if (buffers.mergedGeometry.boundingBox != null) {
      bbox.copy(buffers.mergedGeometry.boundingBox);
    } else {
      buffers.mergedGeometry.computeBoundingBox();
      if (buffers.mergedGeometry.boundingBox != null) {
        bbox.copy(buffers.mergedGeometry.boundingBox);
      }
    }
  }

  const tlas = buffers.tlas;
  const blasContentVersion = fingerprintBuffers(
    buffers.bvhNodes.cpuData,
    buffers.bvhPositions.cpuData,
    buffers.bvhIndex.cpuData,
    buffers.emitterNormals.buffer as ArrayBuffer,
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
    triMaterialIds: buffers.triangleMaterialIds.cpuData,
    materials: buffers.buildMaterials,
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
