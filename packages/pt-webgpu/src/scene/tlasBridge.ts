/**
 * TLAS integration bridge (C2 follow-up).
 *
 * CPU TLAS build/refit/traverse is implemented in `@vitrum/shared-bvh`.
 * This module is the pt-webgpu-side adapter entry for future multi-BLAS
 * scene uploads; today scenes still use a single merged BLAS.
 */

import { buildTlas, type TlasData, type TlasInstance } from '@vitrum/shared-bvh';

export type { TlasInstance, TlasData };

/** Build a TLAS over per-mesh instance world AABBs (CPU-only until WGSL traverse lands). */
export function buildSceneTlas(instances: readonly TlasInstance[]): TlasData {
  return buildTlas(instances);
}
