import type { Scene } from '@vitrum/core';

export interface SceneSummary {
  readonly primitiveCount: number;
  readonly emitterCount: number;
  readonly meshPrimitiveCount: number;
  readonly instancedMeshPrimitiveCount: number;
  readonly analyticPrimitiveCount: number;
  readonly skinnedMeshPrimitiveCount: number;
  readonly vertexCountEstimate: number;
  readonly instanceCountEstimate: number;
  readonly environmentKind: Scene['environment']['kind'];
}

/**
 * Lightweight scene summary used to size buffers and debug unsupported content.
 */
export function summarizeScene(scene: Scene): SceneSummary {
  let meshPrimitiveCount = 0;
  let instancedMeshPrimitiveCount = 0;
  let analyticPrimitiveCount = 0;
  let skinnedMeshPrimitiveCount = 0;
  let vertexCountEstimate = 0;
  let instanceCountEstimate = 0;

  for (const primitive of scene.primitives) {
    if (primitive.kind === 'mesh') {
      meshPrimitiveCount += 1;
      vertexCountEstimate += Math.floor(primitive.positions.length / 3);
      continue;
    }
    if (primitive.kind === 'instanced-mesh') {
      instancedMeshPrimitiveCount += 1;
      vertexCountEstimate += Math.floor(primitive.positions.length / 3);
      instanceCountEstimate += primitive.instances.length;
      continue;
    }
    if (primitive.kind === 'skinned-mesh') {
      // pt-webgpu treats a skinned mesh as a regular mesh at upload time;
      // hosts pre-solve the pose and re-submit positions/normals via
      // engine.updatePrimitive each frame. C1 (2026-05-19).
      skinnedMeshPrimitiveCount += 1;
      vertexCountEstimate += Math.floor(primitive.positions.length / 3);
      continue;
    }
    analyticPrimitiveCount += 1;
  }

  return {
    primitiveCount: scene.primitives.length,
    emitterCount: scene.emitters.length,
    meshPrimitiveCount,
    instancedMeshPrimitiveCount,
    analyticPrimitiveCount,
    skinnedMeshPrimitiveCount,
    vertexCountEstimate,
    instanceCountEstimate,
    environmentKind: scene.environment.kind,
  };
}
