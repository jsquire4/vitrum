import type { Scene } from '@vitrum/core';

export interface SceneSummary {
  readonly primitiveCount: number;
  readonly emitterCount: number;
  readonly meshPrimitiveCount: number;
  readonly instancedMeshPrimitiveCount: number;
  readonly analyticPrimitiveCount: number;
  readonly vertexCountEstimate: number;
  readonly instanceCountEstimate: number;
  readonly environmentKind: Scene['environment']['kind'];
}

/**
 * Lightweight scene summary used by the early pt-webgpu port.
 * This captures enough structural information to size buffers and debug
 * unsupported content while the full BVH/material upload path is implemented.
 */
export function summarizeScene(scene: Scene): SceneSummary {
  let meshPrimitiveCount = 0;
  let instancedMeshPrimitiveCount = 0;
  let analyticPrimitiveCount = 0;
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
    analyticPrimitiveCount += 1;
  }

  return {
    primitiveCount: scene.primitives.length,
    emitterCount: scene.emitters.length,
    meshPrimitiveCount,
    instancedMeshPrimitiveCount,
    analyticPrimitiveCount,
    vertexCountEstimate,
    instanceCountEstimate,
    environmentKind: scene.environment.kind,
  };
}
