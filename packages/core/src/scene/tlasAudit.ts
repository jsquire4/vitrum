/**
 * C2 — classify whether a scene structurally needs TLAS (multi-mesh / instancing).
 * Used by createEngine() backend selection and pt-webgl2 setScene warnings.
 */

import type { ScenePrimitive } from './primitives.js';

export interface SceneTlasAudit {
  /** True when the scene has multiple mesh roots or instancing (TLAS territory). */
  readonly needsTlas: boolean;
  readonly meshLikePrimitiveCount: number;
  readonly totalInstanceCount: number;
  readonly recommendation: 'merged-bvh-ok' | 'prefer-tlas-backend';
  readonly detail: string;
}

/** Classify whether merged single-BVH backends are structurally sufficient. */
export function auditSceneNeedsTlas(scene: {
  readonly primitives: ReadonlyArray<ScenePrimitive>;
}): SceneTlasAudit {
  const primitives = scene.primitives;
  if (primitives == null || typeof primitives[Symbol.iterator] !== 'function') {
    return {
      needsTlas: false,
      meshLikePrimitiveCount: 0,
      totalInstanceCount: 0,
      recommendation: 'merged-bvh-ok',
      detail: 'Scene primitives unavailable; skipping TLAS audit.',
    };
  }
  let meshLike = 0;
  let instances = 0;
  for (const p of primitives) {
    if (p.kind === 'mesh' || p.kind === 'skinned-mesh') meshLike += 1;
    if (p.kind === 'instanced-mesh') {
      meshLike += 1;
      instances += p.instances.length;
    }
  }
  const needsTlas = meshLike > 1 || instances > 1;
  const recommendation = needsTlas ? 'prefer-tlas-backend' : 'merged-bvh-ok';
  const detail = needsTlas
    ? `Scene has ${meshLike} mesh-like primitive(s) and ${instances} TLAS instance(s); ` +
      'prefer @vitrum/walkaround-hybrid or @vitrum/pt-webgpu over merged-BVH pt-webgl2.'
    : 'Single merged BVH is sufficient for this scene layout.';
  return {
    needsTlas,
    meshLikePrimitiveCount: meshLike,
    totalInstanceCount: instances,
    recommendation,
    detail,
  };
}
