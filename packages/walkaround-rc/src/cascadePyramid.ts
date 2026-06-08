/**
 * Cascade pyramid geometry and raw storage contract.
 *
 * Cascade dimensions are derived from Sannikov 2023, "Radiance Cascades: A
 * Novel Approach to Calculating Global Illumination", with deliberate
 * Cornell/WebGPU performance reductions. Runtime buffer allocation for the
 * legacy THREE/TSL receiver path lives in `src/three/cascadePyramidThree.ts`;
 * this module stays free of `three/webgpu` so package-root imports remain
 * raw-runtime safe.
 */

// Performance budget: total compute invocations <= 200K for 30fps on RTX-class hardware
// (assuming ~3 BVH traversals per probe-ray + merge overhead).
// Previous paper-derived dimensions: C0=64x36x56x16 = 2.06M rays.
// Reduced by 4x in each probe dimension while retaining spatial coverage.
export const CASCADE_DIMS = [
  { probes: [16, 9, 14] as [number, number, number], rays: 16, intervalNear: 0, intervalFar: 12 },
  { probes: [8, 5, 7] as [number, number, number], rays: 64, intervalNear: 12, intervalFar: 36 },
  { probes: [4, 3, 4] as [number, number, number], rays: 256, intervalNear: 36, intervalFar: 96 },
  { probes: [3, 2, 3] as [number, number, number], rays: 1024, intervalNear: 96, intervalFar: 240 },
  {
    probes: [2, 2, 2] as [number, number, number],
    rays: 4096,
    intervalNear: 240,
    intervalFar: 1e9,
  },
] as const;

/**
 * One cascade level's geometry. Structural type so hosts can pass custom
 * dimensions via `HybridEngineOptions.cascadeDims` without their tuple
 * literals being narrowed to the Cornell defaults' union of literal types.
 */
export interface CascadeDim {
  readonly probes: readonly [number, number, number];
  readonly rays: number;
  readonly intervalNear: number;
  readonly intervalFar: number;
}

export const CASCADE_COUNT = CASCADE_DIMS.length;

/** Plain AABB ({min,max} in world space), used instead of `THREE.Box3`. */
export interface CascadeAABB {
  readonly min: readonly [number, number, number];
  readonly max: readonly [number, number, number];
}
