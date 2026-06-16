/**
 * Core-scene BVH build + material packing for the RC subsystem.
 *
 * This module intentionally has no runtime dependency on `three` or
 * `three/webgpu`. `HybridEngineRC` consumes only raw typed-array payloads before
 * uploading them to GPU buffers, so a small structural attribute object is
 * enough.
 */

import type { MaterialSpec, Scene, ScenePrimitive } from '@vitrum/core';
import {
  mergeUv1FromCore,
  mergeWorldSpaceFromCore,
  coreMaterialToMaterialEntry,
  packMaterials,
  MATERIAL_ENTRY_FLOATS,
  type MaterialEntryInput,
} from '@vitrum/shared-bvh';
import { packUVIntoVec4W } from '../restir/packingHelpers.js';

export interface StorageAttributeLike<T extends Float32Array | Uint32Array = Float32Array | Uint32Array> {
  readonly array: T;
  readonly itemSize: number;
}

export interface SceneBVH {
  bvhNodes: StorageAttributeLike<Float32Array>;
  positions: StorageAttributeLike<Float32Array>;
  normals: StorageAttributeLike<Float32Array>;
  indices: StorageAttributeLike<Uint32Array>;
  coreMaterials: readonly MaterialSpec[];
  materials: StorageAttributeLike<Float32Array>;
  triMaterialId: StorageAttributeLike<Uint32Array>;
  bounds: {
    readonly min: { readonly x: number; readonly y: number; readonly z: number };
    readonly max: { readonly x: number; readonly y: number; readonly z: number };
  };
}

const RC_CORE_MESH_FILTER = (p: ScenePrimitive): boolean =>
  p.kind === 'mesh' || p.kind === 'skinned-mesh' || p.kind === 'instanced-mesh';

function attr<T extends Float32Array | Uint32Array>(array: T, itemSize: number): StorageAttributeLike<T> {
  return { array, itemSize };
}

function toProductionEmissiveRadiance(m: MaterialSpec): MaterialSpec {
  if (m.emissive === undefined) return m;
  if (m.emissiveIntensity === 1) return m;
  return { ...m, emissiveIntensity: 1 };
}

function coreToCascadeMaterialEntryInput(mat: MaterialSpec): MaterialEntryInput {
  const entry = coreMaterialToMaterialEntry(toProductionEmissiveRadiance(mat));
  return {
    ...entry,
    thickness: entry.thickness !== undefined && entry.thickness > 0 ? entry.thickness : 0.1,
  };
}

export function packCascadeMaterialsFromCore(materials: readonly MaterialSpec[]): Float32Array {
  if (materials.length === 0) {
    return new Float32Array(MATERIAL_ENTRY_FLOATS);
  }
  return packMaterials(materials.map(coreToCascadeMaterialEntryInput));
}

export function buildRCSceneBVHFromCore(
  scene: Scene,
  opts: { filter?: (p: ScenePrimitive) => boolean } = {},
): SceneBVH {
  const merged = mergeWorldSpaceFromCore(scene, {
    positionStride: 4,
    filter: opts.filter ?? RC_CORE_MESH_FILTER,
    splitMaterialsByCastShadow: true,
  });
  const materialFloats = packCascadeMaterialsFromCore(merged.materials);
  const vertCount = merged.vertexCount;
  const mergedUv1 = mergeUv1FromCore(scene, merged.meshVertexRanges, vertCount);
  const normalsWithUV1 = packUVIntoVec4W(
    merged.normals,
    mergedUv1 == null ? undefined : { array: mergedUv1 },
    vertCount,
  );
  return {
    bvhNodes: attr(merged.bvhNodes, 8),
    positions: attr(merged.positions, 4),
    normals: attr(normalsWithUV1, 4),
    indices: attr(merged.indices, 3),
    coreMaterials: merged.materials,
    materials: attr(materialFloats, 16),
    triMaterialId: attr(merged.triMaterialId, 1),
    bounds: {
      min: {
        x: merged.boundingBox.min[0],
        y: merged.boundingBox.min[1],
        z: merged.boundingBox.min[2],
      },
      max: {
        x: merged.boundingBox.max[0],
        y: merged.boundingBox.max[1],
        z: merged.boundingBox.max[2],
      },
    },
  };
}
