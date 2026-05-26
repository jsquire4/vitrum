/**
 * PR-5.1 — DDGI probe rays share ReSTIR `SceneBVHBuffers` (no separate SceneBvh build).
 */

import * as THREE from 'three';
import type { Scene } from '@vitrum/core';
import { computeWorldAabbForBindings } from '@vitrum/shared-bvh';
import type { SceneBVHBuffers } from '../restir/bvhCompute.js';

export interface DdgiRestirBvhSnapshot {
  readonly bvhMode: 'merged' | 'tlas';
  readonly tlasNodeCount: number;
  readonly bvhNodes: ArrayBuffer;
  readonly positions: ArrayBuffer;
  /** Stride-4 triangle index buffer (same as ReSTIR `bvhIndex`). */
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
  /** Bumps when BLAS/TLAS CPU payloads change (used to skip redundant GPU uploads). */
  readonly contentVersion: number;
}

function hashBuffers(...parts: ArrayBuffer[]): number {
  let h = 2166136261;
  for (const buf of parts) {
    h ^= buf.byteLength;
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

export function makeDdgiRestirBvhSnapshot(
  buffers: SceneBVHBuffers,
  scene?: Scene,
): DdgiRestirBvhSnapshot {
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
  const tlasParts = tlas != null
    ? [tlas.nodes.cpuData, tlas.instanceIndices.cpuData, tlas.blasRoots.cpuData, tlas.worldToLocal.cpuData, tlas.localToWorld.cpuData]
    : [];

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
    contentVersion: hashBuffers(
      buffers.bvhNodes.cpuData,
      buffers.bvhPositions.cpuData,
      buffers.bvhIndex.cpuData,
      buffers.emitterNormals.buffer as ArrayBuffer,
      buffers.triangleMaterialIds.cpuData,
      ...tlasParts,
    ),
  };
}
