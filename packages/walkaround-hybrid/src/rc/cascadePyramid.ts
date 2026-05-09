/**
 * Cascade pyramid storage layout.
 *
 * Cascade dimensions derived from Sannikov's conservation law:
 *   R_k ∝ (L_k / s_k)²  (ray count tracks solid-angle resolution at interval distance)
 *
 * C0 anchor: s_0 = 3" (Nyquist from ~3" panel pieces).
 * Probes: ceil(192/3) × ceil(108/3) × ceil(168/3) = 64 × 36 × 56.
 * Interval base b = 3 (room-diagonal fit in 4 finite cascades).
 * Conservation-law check: b=3 means (L_k/s_k) grows by 1.5×, so strict conservation
 * wants R_k × 9 per cascade; we keep × 4 for perf (documented angular under-resolution).
 *
 * Storage: each cascade is a flat Float32Array (4 floats per direction-bin: rgba).
 * Packed layout: [probeX * probeY * probeZ * raysPerProbe] × 4 floats.
 * Probe index: probeIdx = px + py*PX + pz*PX*PY.
 * Direction bin index: probeIdx * raysPerProbe + rayIdx.
 *
 * Note: `gpuCascades` uses `StorageBufferAttribute` from `three/webgpu` because the
 * C0 buffer is consumed by the TSL `walkaroundDiffuseLighting.ts` node via `storage()`.
 * The `RCDispatcher` (cascadeDispatch.ts) accesses the same GPU buffers through the
 * Three.js WebGPU renderer backend.  See TSL_TO_RAW_MAPPING.md for rationale.
 */

import * as THREE from 'three';
import { StorageBufferAttribute } from 'three/webgpu';

// Performance budget: total compute invocations ≤ 200K for 30fps on RTX-class hardware
// (assuming ~3 BVH traversals per probe-ray + merge overhead).
// Previous (paper-derived) dimensions: C0=64×36×56×16 = 2.06M rays (too slow: ~4fps measured).
// Reduced by 4× in each probe dimension to bring within budget while retaining spatial coverage.
export const CASCADE_DIMS = [
  { probes: [16,  9, 14] as [number, number, number], rays: 16,   intervalNear: 0,    intervalFar: 12   },
  { probes: [ 8,  5,  7] as [number, number, number], rays: 64,   intervalNear: 12,   intervalFar: 36   },
  { probes: [ 4,  3,  4] as [number, number, number], rays: 256,  intervalNear: 36,   intervalFar: 96   },
  { probes: [ 3,  2,  3] as [number, number, number], rays: 1024, intervalNear: 96,   intervalFar: 240  },
  { probes: [ 2,  2,  2] as [number, number, number], rays: 4096, intervalNear: 240,  intervalFar: 1e9  },
] as const;

export type CascadeDim = (typeof CASCADE_DIMS)[number];
export const CASCADE_COUNT = CASCADE_DIMS.length;

/** Total rays per cascade (probeX * probeY * probeZ * raysPerProbe). */
function cascadeTotalRays(c: CascadeDim): number {
  return c.probes[0] * c.probes[1] * c.probes[2] * c.rays;
}

/** Number of float32 values per cascade (4 per ray: rgba radiance). */
function cascadeBufferSize(c: CascadeDim): number {
  return cascadeTotalRays(c) * 4;
}

export interface CascadeBuffers {
  /** One Float32Array per cascade, length = probeX*probeY*probeZ*rays*4 floats. */
  cascades: Float32Array[];
  /**
   * GPU-side StorageBufferAttribute wrappers — one per cascade, same order as `cascades`.
   * Created alongside the CPU arrays so both the compute dispatch (writer) and the
   * GI lighting node (reader) share the exact same GPU buffer object.
   * IMPORTANT: never create a second StorageBufferAttribute from the same Float32Array;
   * the WebGPU runtime allocates a separate GPU buffer per StorageBufferAttribute instance.
   */
  gpuCascades: StorageBufferAttribute[];
  /** Room AABB min corner in world space (probes start here). */
  probeOriginWorld: THREE.Vector3;
  /** Room AABB size in world space. */
  roomSize: THREE.Vector3;
}

/**
 * Allocate cascade storage on the CPU.
 * StorageBufferAttribute GPU-side buffers are allocated by the Three.js WebGPU
 * renderer the first time they are uploaded.
 */
export function allocateCascades(bounds: THREE.Box3): CascadeBuffers {
  const size   = bounds.getSize(new THREE.Vector3());
  const origin = bounds.min.clone();
  const cascades = CASCADE_DIMS.map((c) => {
    const len = cascadeBufferSize(c);
    return new Float32Array(len);
  });
  // itemSize=4 (vec4f: r,g,b,a per ray). count = total rays per cascade.
  const gpuCascades = cascades.map((arr) => new StorageBufferAttribute(arr, 4));
  return { cascades, gpuCascades, probeOriginWorld: origin, roomSize: size };
}

/** Dispose: clear references (Float32Arrays are GC'd). */
export function disposeCascades(b: CascadeBuffers): void {
  b.cascades.length = 0;
}

/**
 * Fill all cascades with constant test colours.
 * Smoke-test path — verifies the data path without trusting ray-cast math.
 * Cascade 0 = warm red, 1 = orange, 2 = yellow, 3 = green, 4 = blue.
 */
const DEBUG_COLORS: [number, number, number][] = [
  [0.8, 0.1, 0.1],
  [0.8, 0.4, 0.1],
  [0.8, 0.8, 0.1],
  [0.1, 0.7, 0.2],
  [0.1, 0.3, 0.9],
];

export function fillCascadeDebug(b: CascadeBuffers): void {
  CASCADE_DIMS.forEach((c, k) => {
    const buf   = b.cascades[k];
    if (!buf) return;
    const color = DEBUG_COLORS[k] ?? [0.5, 0.5, 0.5];
    const total = cascadeTotalRays(c);
    for (let i = 0; i < total; i++) {
      buf[i * 4 + 0] = color[0];
      buf[i * 4 + 1] = color[1];
      buf[i * 4 + 2] = color[2];
      buf[i * 4 + 3] = 0;  // alpha=0 means "escaped" (for merge pass to merge from upper)
    }
  });
}
