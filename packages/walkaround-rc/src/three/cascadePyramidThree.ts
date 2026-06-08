/**
 * THREE/WebGPU bridge allocation for Radiance Cascades.
 *
 * This module owns the `StorageBufferAttribute` wrappers consumed by the TSL
 * receiver path. Keep it off the package root so raw RC runtime imports do not
 * load `three/webgpu`.
 */

import { StorageBufferAttribute } from 'three/webgpu';
import { CASCADE_DIMS, type CascadeAABB, type CascadeDim } from '../cascadePyramid.js';

export type { CascadeAABB, CascadeDim } from '../cascadePyramid.js';

export interface CascadeBuffers {
  /** Cascade dimensions used for allocation. */
  dims: readonly CascadeDim[];
  /** One Float32Array per cascade, length = probeX*probeY*probeZ*rays*4 floats. */
  cascades: Float32Array[];
  /**
   * GPU-side StorageBufferAttribute wrappers, one per cascade.
   * Never create a second StorageBufferAttribute from the same Float32Array;
   * the WebGPU runtime allocates a separate GPU buffer per wrapper instance.
   */
  gpuCascades: StorageBufferAttribute[];
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
 * Allocate cascade storage for the legacy THREE/TSL receiver bridge.
 *
 * StorageBufferAttribute GPU-side buffers are allocated by the Three.js WebGPU
 * renderer the first time they are uploaded.
 */
export function allocateCascades(
  bounds: CascadeAABB,
  dims: readonly CascadeDim[] = CASCADE_DIMS,
): CascadeBuffers {
  // Floor each axis at 1e-6 so degenerate scenes never feed zero divisors into
  // the cascade merge UV mapping or probe-ray slab step.
  const sx = Math.max(bounds.max[0] - bounds.min[0], 1e-6);
  const sy = Math.max(bounds.max[1] - bounds.min[1], 1e-6);
  const sz = Math.max(bounds.max[2] - bounds.min[2], 1e-6);
  const origin: readonly [number, number, number] = [bounds.min[0], bounds.min[1], bounds.min[2]];
  const size: readonly [number, number, number] = [sx, sy, sz];
  const cascades = dims.map((c) => new Float32Array(cascadeBufferSize(c)));
  const gpuCascades = cascades.map((arr) => new StorageBufferAttribute(arr, 4));
  return { dims, cascades, gpuCascades, probeOriginWorld: origin, roomSize: size };
}

/** Dispose CPU-side cascade references. */
export function disposeCascades(b: CascadeBuffers): void {
  b.cascades.length = 0;
}

const DEBUG_COLORS: [number, number, number][] = [
  [0.8, 0.1, 0.1],
  [0.8, 0.4, 0.1],
  [0.8, 0.8, 0.1],
  [0.1, 0.7, 0.2],
  [0.1, 0.3, 0.9],
];

/** Fill all cascades with constant test colors. */
export function fillCascadeDebug(b: CascadeBuffers): void {
  b.dims.forEach((c, k) => {
    const buf = b.cascades[k];
    if (!buf) return;
    const color = DEBUG_COLORS[k] ?? [0.5, 0.5, 0.5];
    const total = cascadeTotalRays(c);
    for (let i = 0; i < total; i++) {
      buf[i * 4 + 0] = color[0];
      buf[i * 4 + 1] = color[1];
      buf[i * 4 + 2] = color[2];
      buf[i * 4 + 3] = 0;
    }
  });
}
