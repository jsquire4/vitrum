/**
 * Cascade pyramid geometry and raw storage contract.
 *
 * Cascade dimensions are derived from Sannikov 2023, "Radiance Cascades: A
 * Novel Approach to Calculating Global Illumination", with deliberate
 * Cornell/WebGPU performance reductions. This module is free of `three/webgpu`
 * so package-root imports remain raw-runtime safe. The old TSL receiver wrappers
 * are not shipped; `@vitrum/walkaround-hybrid` composes the raw cascade output
 * into its shade pass.
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

/** Number of levels in the exported default pyramid; always derived from its layout. */
export const CASCADE_COUNT = CASCADE_DIMS.length;

const UINT32_MAX = 0xffff_ffff;
const MAX_CASCADE_RAYS_FOR_VEC4_INDEXING = Math.floor(UINT32_MAX / 4);

function isArrayValue(value: unknown): boolean {
  return Array.isArray(value);
}

function assertPositiveInteger(value: unknown, path: string): asserts value is number {
  if (!Number.isSafeInteger(value) || (value as number) <= 0 || (value as number) > UINT32_MAX) {
    throw new Error(`${path} must be a positive integer; received ${String(value)}`);
  }
}

function assertFiniteNumber(value: unknown, path: string): asserts value is number {
  if (typeof value !== 'number' || !Number.isFinite(value) || !Number.isFinite(Math.fround(value))) {
    throw new Error(`${path} must be a finite number; received ${String(value)}`);
  }
}

function checkedProduct(values: readonly number[], path: string): number {
  let product = 1;
  for (const value of values) {
    if (product > Math.floor(Number.MAX_SAFE_INTEGER / value)) {
      throw new Error(`${path} exceeds JavaScript's safe-integer range`);
    }
    product *= value;
  }
  return product;
}

/**
 * Validate cascade dimensions before allocating buffers or compiling dispatch
 * state. The merge shader assumes each cascade's ray grid is square and each
 * higher cascade doubles the lower cascade's ray-grid width.
 */
export function validateCascadeDims(
  dims: readonly CascadeDim[],
  label = 'cascadeDims',
): readonly CascadeDim[] {
  if (!isArrayValue(dims) || dims.length === 0) {
    throw new Error(`${label} must contain at least one cascade`);
  }
  if (!Number.isSafeInteger(dims.length) || dims.length > UINT32_MAX) {
    throw new Error(`${label}.length must fit in a u32`);
  }

  let previousRayGrid = 0;
  for (let i = 0; i < dims.length; i += 1) {
    const dim = dims[i];
    if (dim == null || !Array.isArray(dim.probes) || dim.probes.length !== 3) {
      throw new Error(`${label}[${i}].probes must be a [x, y, z] tuple`);
    }
    assertPositiveInteger(dim.probes[0], `${label}[${i}].probes[0]`);
    assertPositiveInteger(dim.probes[1], `${label}[${i}].probes[1]`);
    assertPositiveInteger(dim.probes[2], `${label}[${i}].probes[2]`);
    assertPositiveInteger(dim.rays, `${label}[${i}].rays`);

    const totalRays = checkedProduct(
      [dim.probes[0], dim.probes[1], dim.probes[2], dim.rays],
      `${label}[${i}] total ray count`,
    );
    if (totalRays > MAX_CASCADE_RAYS_FOR_VEC4_INDEXING) {
      throw new Error(
        `${label}[${i}] total ray count ${totalRays} exceeds the u32 vec4 indexing limit ` +
        `${MAX_CASCADE_RAYS_FOR_VEC4_INDEXING}`,
      );
    }

    const rayGrid = Math.sqrt(dim.rays);
    if (!Number.isInteger(rayGrid)) {
      throw new Error(`${label}[${i}].rays must be a perfect square; received ${dim.rays}`);
    }
    if (i > 0 && rayGrid !== previousRayGrid * 2) {
      throw new Error(
        `${label}[${i}].rays must double the previous cascade ray-grid width; ` +
        `sqrt(rays)=${rayGrid}, expected ${previousRayGrid * 2}`,
      );
    }
    previousRayGrid = rayGrid;

    assertFiniteNumber(dim.intervalNear, `${label}[${i}].intervalNear`);
    assertFiniteNumber(dim.intervalFar, `${label}[${i}].intervalFar`);
    if (dim.intervalNear < 0 || dim.intervalFar <= dim.intervalNear) {
      throw new Error(
        `${label}[${i}] interval must satisfy 0 <= intervalNear < intervalFar; ` +
        `received ${dim.intervalNear}..${dim.intervalFar}`,
      );
    }
  }

  return dims;
}
