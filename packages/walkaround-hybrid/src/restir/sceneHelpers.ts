/**
 * Core-native scene-graph utilities for ReSTIR BVH construction.
 *
 * Functions here work directly with `@vitrum/core` Scene/Primitive types
 * and carry no emitter-packing or GPU-buffer concerns.
 */

import type { Mat4, Scene } from '@vitrum/core';

const IDENTITY_MAT4 = new Float32Array([
  1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 1, 0,
  0, 0, 0, 1,
]);

type RawMeshVertexRange = {
  name: string;
  vertexStart: number;
  vertexCount: number;
  triStart: number;
  triCount: number;
};

type MeshVertexRangeWithMatrix = RawMeshVertexRange & {
  matrixWorldAtBuild: Float32Array;
};

function cloneMat4(m: Mat4 | Float32Array | undefined): Float32Array {
  return m != null ? new Float32Array(m) : new Float32Array(IDENTITY_MAT4);
}

/**
 * Derive the build-time world matrix snapshots from the core primitive
 * transforms directly, so core BVH/update paths do not need a synthesized
 * scene graph solely to populate `matrixWorldAtBuild`.
 */
export function enrichMeshVertexRangesWithCoreMatrix(
  scene: Scene,
  rawRanges: ReadonlyArray<RawMeshVertexRange>,
): ReadonlyArray<MeshVertexRangeWithMatrix> {
  const primitiveById = new Map<string, Scene['primitives'][number]>();
  for (const primitive of scene.primitives) {
    primitiveById.set(String(primitive.id), primitive);
  }
  const instancedOccurrences = new Map<string, number>();
  return rawRanges.map((r) => {
    const primitive = primitiveById.get(r.name);
    let matrix: Float32Array;
    if (primitive?.kind === 'instanced-mesh') {
      const occurrence = instancedOccurrences.get(r.name) ?? 0;
      instancedOccurrences.set(r.name, occurrence + 1);
      matrix = cloneMat4(primitive.instances[occurrence]);
    } else if (primitive?.kind === 'mesh' || primitive?.kind === 'skinned-mesh') {
      matrix = cloneMat4(primitive.transform);
    } else {
      matrix = cloneMat4(undefined);
    }
    return {
      name: r.name,
      vertexStart: r.vertexStart,
      vertexCount: r.vertexCount,
      triStart: r.triStart,
      triCount: r.triCount,
      matrixWorldAtBuild: matrix,
    };
  });
}
