import * as THREE from 'three';
import { buildSceneBVH as buildCommonSceneBVH } from '@vitrum/shared-bvh';

export interface SceneBVHBuffers {
  bvhNodes: Float32Array;
  positions: Float32Array;
  indices: Uint32Array;
  normals: Float32Array;
  totalEmissivePower: number;
  emitterCount: number;
}

/**
 * Compatibility shim for legacy staging host code.
 * The canonical BVH implementation now lives in @vitrum/shared-bvh.
 */
export function buildSceneBVH(
  scenes: THREE.Scene[],
  _opts?: { primaryLightDir?: THREE.Vector3; primaryLightIntensity?: number },
): SceneBVHBuffers {
  const merged = new THREE.Scene();
  for (const s of scenes) merged.add(s);
  const built = buildCommonSceneBVH(merged, { positionStride: 4 });
  return {
    bvhNodes: built.bvhNodes,
    positions: built.positions,
    indices: built.indices,
    normals: built.normals,
    totalEmissivePower: 0,
    emitterCount: 0,
  };
}

export function disposeSceneBVH(_buffers: SceneBVHBuffers): void {
  // No-op for staging shim.
}
