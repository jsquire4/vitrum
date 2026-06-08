import * as THREE from 'three';
import { buildSceneBVH } from './bvhCommon.js';
import { SceneBvh, type SceneBvhBuffers, type SceneBvhOptions } from '../sceneBvh.js';
import type { PlainAabb } from '../aabb.js';

export { buildSceneBVH } from './bvhCommon.js';
export type { SceneBVHCommonOpts, SceneBVHCommonResult } from './bvhCommon.js';

export interface LegacyThreeSceneBvhBuffers extends SceneBvhBuffers {
  readonly sourceMaterials: readonly THREE.Material[];
  /** @deprecated Use {@link sourceMaterials}. */
  readonly legacyThreeMaterials: readonly THREE.Material[];
}

/**
 * Legacy THREE scene ingestion for DDGI standalone callers.
 *
 * Root `SceneBvh` is core-native; this wrapper keeps the old `update(scene)`
 * path available without making the shared-bvh root import THREE.
 */
export class LegacyThreeSceneBvh extends SceneBvh {
  private _lastGeometryVersion = -1;

  constructor(opts: SceneBvhOptions = {}) {
    super(opts);
  }

  override get buffers(): LegacyThreeSceneBvhBuffers | null {
    return this._buffers as LegacyThreeSceneBvhBuffers | null;
  }

  update(scene: THREE.Scene): void {
    const meshes: THREE.Mesh[] = [];
    scene.traverseVisible((obj) => {
      if (DDGI_MESH_FILTER(obj)) meshes.push(obj as THREE.Mesh);
    });

    if (meshes.length === 0) {
      this._buffers = null;
      this._lastGeometryVersion = -1;
      return;
    }

    let version = meshes.length * 1000003;
    for (const m of meshes) {
      const posAttr = m.geometry.attributes['position'] as THREE.BufferAttribute;
      version += posAttr.version ?? 0;
    }
    if (version === this._lastGeometryVersion && this._buffers !== null) return;
    this._lastGeometryVersion = version;

    const t0 = performance.now();
    const result = buildSceneBVH(scene, {
      filter: DDGI_MESH_FILTER,
      positionStride: 4,
    });

    const elapsed = performance.now() - t0;
    if (elapsed > 50) {
      if (this.opts.onSlowRebuild) {
        this.opts.onSlowRebuild(elapsed);
      } else {
        console.warn(`[DDGI SceneBvh] BVH rebuild took ${elapsed.toFixed(0)}ms (>50ms threshold)`);
      }
    }

    const buffers: LegacyThreeSceneBvhBuffers = {
      bvhNodes: result.bvhNodes,
      positions: result.positions,
      indices: result.indices,
      normals: result.normals,
      triMaterialId: result.triMaterialId,
      materials: [],
      sourceMaterials: result.materials,
      legacyThreeMaterials: result.materials,
      boundingBox: box3ToPlainAabb(result.boundingBox),
    };
    this._buffers = buffers;
  }

  override dispose(): void {
    super.dispose();
    this._lastGeometryVersion = -1;
  }
}

const DDGI_MESH_FILTER = (obj: THREE.Object3D): boolean => {
  if (!(obj instanceof THREE.Mesh)) return false;
  if (!obj.geometry) return false;
  if (!obj.geometry.attributes['position']) return false;
  return true;
};

function box3ToPlainAabb(box: THREE.Box3): PlainAabb {
  return {
    min: [box.min.x, box.min.y, box.min.z],
    max: [box.max.x, box.max.y, box.max.z],
  };
}
