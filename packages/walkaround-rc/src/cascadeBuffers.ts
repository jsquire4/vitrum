import {
  CASCADE_DIMS,
  validateCascadeDims,
  type CascadeAABB,
  type CascadeDim,
} from './cascadePyramid.js';

export interface CascadeBuffers {
  /** Cascade dimensions used for allocation. */
  dims: readonly CascadeDim[];
  /** One Float32Array per cascade, length = probeX * probeY * probeZ * rays * 4 floats. */
  cascades: Float32Array[];
  /** Room AABB min corner in world space. */
  probeOriginWorld: readonly [number, number, number];
  /** Room AABB size in world space. */
  roomSize: readonly [number, number, number];
}

function cascadeTotalRays(c: CascadeDim): number {
  return c.probes[0] * c.probes[1] * c.probes[2] * c.rays;
}

function cascadeBufferSize(c: CascadeDim): number {
  return cascadeTotalRays(c) * 4;
}

/**
 * Allocate CPU cascade storage for raw Radiance Cascades validation and hosts
 * that want the package's canonical cascade dimensions without constructing a
 * dispatcher.
 */
export function allocateCascades(
  bounds: CascadeAABB,
  dims: readonly CascadeDim[] = CASCADE_DIMS,
): CascadeBuffers {
  validateCascadeDims(dims, 'allocateCascades cascadeDims');
  const sx = Math.max(bounds.max[0] - bounds.min[0], 1e-6);
  const sy = Math.max(bounds.max[1] - bounds.min[1], 1e-6);
  const sz = Math.max(bounds.max[2] - bounds.min[2], 1e-6);
  const origin: readonly [number, number, number] = [bounds.min[0], bounds.min[1], bounds.min[2]];
  const size: readonly [number, number, number] = [sx, sy, sz];
  const cascades = dims.map((c) => new Float32Array(cascadeBufferSize(c)));
  return { dims, cascades, probeOriginWorld: origin, roomSize: size };
}

/** Drop CPU-side cascade references. */
export function disposeCascades(b: CascadeBuffers): void {
  b.cascades.length = 0;
}
