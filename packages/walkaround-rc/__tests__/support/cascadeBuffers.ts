import {
  CASCADE_DIMS,
  validateCascadeDims,
  type CascadeDim,
} from '../../src/cascadePyramid.js';

interface CascadeAABB {
  readonly min: readonly [number, number, number];
  readonly max: readonly [number, number, number];
}

interface CascadeBuffers {
  /** Cascade dimensions used for allocation. */
  dims: readonly CascadeDim[];
  /** One Float32Array per cascade, length = probeX * probeY * probeZ * rays * 4 floats. */
  cascades: Float32Array[];
  /** Room AABB min corner in world space. */
  probeOriginWorld: readonly [number, number, number];
  /** Room AABB size in world space. */
  roomSize: readonly [number, number, number];
}

/** Refuse accidental multi-gigabyte CPU allocations from untrusted custom dimensions. */
const MAX_CASCADE_CPU_ALLOCATION_BYTES = 512 * 1024 * 1024;

function cascadeTotalRays(c: CascadeDim): number {
  return c.probes[0] * c.probes[1] * c.probes[2] * c.rays;
}

function cascadeBufferSize(c: CascadeDim): number {
  return cascadeTotalRays(c) * 4;
}

function isArrayValue(value: unknown): boolean {
  return Array.isArray(value);
}

function validateBounds(bounds: CascadeAABB): {
  origin: readonly [number, number, number];
  size: readonly [number, number, number];
} {
  if (
    bounds == null ||
    !isArrayValue(bounds.min) || bounds.min.length !== 3 ||
    !isArrayValue(bounds.max) || bounds.max.length !== 3
  ) {
    throw new Error('allocateCascades bounds must contain min/max [x, y, z] tuples');
  }

  const origin = [...bounds.min] as [number, number, number];
  const size: [number, number, number] = [0, 0, 0];
  for (let axis = 0; axis < 3; axis += 1) {
    const min = origin[axis];
    const max = bounds.max[axis];
    if (
      typeof min !== 'number' || !Number.isFinite(min) || !Number.isFinite(Math.fround(min)) ||
      typeof max !== 'number' || !Number.isFinite(max) || !Number.isFinite(Math.fround(max))
    ) {
      throw new Error(`allocateCascades bounds axis ${axis} must contain finite f32 values`);
    }
    const extent = max - min;
    if (!(extent > 0) || !Number.isFinite(Math.fround(extent))) {
      throw new Error(`allocateCascades bounds axis ${axis} must have a positive finite f32 extent`);
    }
    size[axis] = extent;
  }
  return { origin, size };
}

/**
 * Allocate CPU cascade storage for deterministic dispatcher-layout tests.
 *
 * This is deliberately test support rather than a package runtime API: the
 * production RC subsystem allocates GPUBuffer storage in HybridEngineRC.
 */
export function allocateCascades(
  bounds: CascadeAABB,
  dims: readonly CascadeDim[] = CASCADE_DIMS,
): CascadeBuffers {
  validateCascadeDims(dims, 'allocateCascades cascadeDims');
  const { origin, size } = validateBounds(bounds);
  const floatLengths = dims.map(cascadeBufferSize);
  const totalBytes = floatLengths.reduce((sum, length, index) => {
    if (!Number.isSafeInteger(length) || length <= 0) {
      throw new Error(`allocateCascades cascadeDims[${index}] has an unsafe buffer length`);
    }
    const bytes = length * Float32Array.BYTES_PER_ELEMENT;
    if (!Number.isSafeInteger(bytes) || sum > MAX_CASCADE_CPU_ALLOCATION_BYTES - bytes) {
      throw new Error(
        `allocateCascades requires more than ${MAX_CASCADE_CPU_ALLOCATION_BYTES} bytes of CPU storage`,
      );
    }
    return sum + bytes;
  }, 0);
  if (totalBytes <= 0) {
    throw new Error('allocateCascades requires a non-empty allocation');
  }
  const cascades = floatLengths.map(length => new Float32Array(length));
  return { dims, cascades, probeOriginWorld: origin, roomSize: size };
}
