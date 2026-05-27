/**
 * C2 follow-up — pt-webgl still uses the fork's merged world-space BVH.
 * Hosts with multi-mesh / instanced scenes should prefer walkaround-hybrid or
 * pt-webgpu (TLAS). This module reports when a Vitrum scene would need TLAS so
 * hosts can route backends without silent single-BVH merge.
 */

import type { Scene } from '@vitrum/core';

export interface PtWebglTlasAudit {
  /** True when the scene has multiple mesh roots or instancing (TLAS territory). */
  readonly needsTlas: boolean;
  readonly meshLikePrimitiveCount: number;
  readonly totalInstanceCount: number;
  readonly recommendation: 'merged-bvh-ok' | 'prefer-tlas-backend';
  readonly detail: string;
}

/** Classify whether pt-webgl's merged-BVH path is structurally sufficient. */
export function auditPtWebglSceneForTlas(scene: Scene): PtWebglTlasAudit {
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
      'pt-webgl merges into one BVH — use @vitrum/walkaround-hybrid or @vitrum/pt-webgpu for per-mesh BLAS.'
    : 'Single merged BVH is sufficient for this scene layout.';
  return {
    needsTlas,
    meshLikePrimitiveCount: meshLike,
    totalInstanceCount: instances,
    recommendation,
    detail,
  };
}
