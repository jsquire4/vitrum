/**
 * TLAS integration bridge (C2 follow-up).
 *
 * CPU TLAS build/refit/traverse is implemented in `@vitrum/shared-bvh`.
 * This module is the pt-webgpu-side adapter entry for multi-BLAS scene
 * uploads (BLAS nodes are packed into one storage buffer with per-instance
 * root offsets, and TLAS traversal selects the appropriate root).
 */

import { buildTlas, refitTlas, type TlasData, type TlasInstance } from '@vitrum/shared-bvh';

export type { TlasInstance, TlasData };

/** Build a TLAS over per-mesh instance world AABBs (CPU-only until WGSL traverse lands). */
export function buildSceneTlas(instances: readonly TlasInstance[]): TlasData {
  return buildTlas(instances);
}

/** Refit an existing TLAS with updated per-instance world AABBs. */
export function refitSceneTlas(
  data: TlasData,
  newAabbs: ReadonlyArray<{ min: readonly [number, number, number]; max: readonly [number, number, number] }>,
): void {
  refitTlas(data, newAabbs);
}
