export type AreaEmitterVec3 = readonly [number, number, number];

export interface AreaVectorMeasure {
  /** Unit direction of u × v after Float32 publication semantics. */
  readonly normal: readonly [number, number, number];
  /** coefficient · |u × v|, rounded exactly as the shader-facing Float32 value. */
  readonly area: number;
  /** Shared max-component scale used to equilibrate both input vectors. */
  readonly edgeScale: number;
}

export type AreaVectorMeasureResult =
  | ({ readonly valid: true } & AreaVectorMeasure)
  | {
      readonly valid: false;
      readonly reason: 'non-finite-input' | 'degenerate' | 'unrepresentable-area';
    };

const F32_MAX = 3.4028234663852886e38;

function finiteF32(value: number): number | null {
  const rounded = Math.fround(value);
  return Number.isFinite(rounded) ? rounded : null;
}

function f32Product(a: number, b: number): number {
  return Math.fround(Math.fround(a) * Math.fround(b));
}

function f32Difference(a: number, b: number): number {
  return Math.fround(Math.fround(a) - Math.fround(b));
}

/**
 * Measure an area vector using the same scale-equilibrated arithmetic used by
 * pt-webgpu's production WGSL.
 *
 * Both input vectors are first rounded to the values that can actually be
 * published in a Float32 storage buffer, then divided by one shared
 * max-component scale. The cross direction is normalized in that O(1) domain;
 * its magnitude is recovered as
 *
 *   coefficient · |cross(u / s, v / s)| · s · s
 *
 * with one multiplication by `s` at a time. This avoids both the raw
 * `dot(cross, cross)` underflow for tiny emitters and the raw cross overflow for
 * huge emitters. Exact collinearity remains invalid. A null result means the
 * inputs, normal, or requested area cannot be represented as finite positive
 * f32 shader values.
 */
export function classifyAreaVectorF32(
  uInput: AreaEmitterVec3,
  vInput: AreaEmitterVec3,
  coefficient: number,
): AreaVectorMeasureResult {
  const ux = finiteF32(uInput[0]);
  const uy = finiteF32(uInput[1]);
  const uz = finiteF32(uInput[2]);
  const vx = finiteF32(vInput[0]);
  const vy = finiteF32(vInput[1]);
  const vz = finiteF32(vInput[2]);
  const areaCoefficient = finiteF32(coefficient);
  if (
    ux == null || uy == null || uz == null ||
    vx == null || vy == null || vz == null ||
    areaCoefficient == null || !(areaCoefficient > 0)
  ) {
    return { valid: false, reason: 'non-finite-input' };
  }

  const edgeScale = Math.max(
    Math.abs(ux), Math.abs(uy), Math.abs(uz),
    Math.abs(vx), Math.abs(vy), Math.abs(vz),
  );
  if (!(edgeScale > 0)) return { valid: false, reason: 'degenerate' };
  if (edgeScale > F32_MAX) return { valid: false, reason: 'non-finite-input' };

  const sux = Math.fround(ux / edgeScale);
  const suy = Math.fround(uy / edgeScale);
  const suz = Math.fround(uz / edgeScale);
  const svx = Math.fround(vx / edgeScale);
  const svy = Math.fround(vy / edgeScale);
  const svz = Math.fround(vz / edgeScale);
  const cx = f32Difference(f32Product(suy, svz), f32Product(suz, svy));
  const cy = f32Difference(f32Product(suz, svx), f32Product(sux, svz));
  const cz = f32Difference(f32Product(sux, svy), f32Product(suy, svx));

  const crossScale = Math.max(Math.abs(cx), Math.abs(cy), Math.abs(cz));
  if (!(crossScale > 0)) return { valid: false, reason: 'degenerate' };
  if (!Number.isFinite(crossScale)) {
    return { valid: false, reason: 'non-finite-input' };
  }

  const dcx = Math.fround(cx / crossScale);
  const dcy = Math.fround(cy / crossScale);
  const dcz = Math.fround(cz / crossScale);
  const directionLength = Math.fround(Math.hypot(dcx, dcy, dcz));
  if (!(directionLength > 0)) return { valid: false, reason: 'degenerate' };
  if (!Number.isFinite(directionLength)) {
    return { valid: false, reason: 'non-finite-input' };
  }

  const nx = Math.fround(dcx / directionLength);
  const ny = Math.fround(dcy / directionLength);
  const nz = Math.fround(dcz / directionLength);
  if (![nx, ny, nz].every(Number.isFinite)) {
    return { valid: false, reason: 'non-finite-input' };
  }

  const scaledCrossLength = f32Product(crossScale, directionLength);
  const weightedCrossLength = f32Product(areaCoefficient, scaledCrossLength);
  const onceRescaled = f32Product(weightedCrossLength, edgeScale);
  const area = f32Product(onceRescaled, edgeScale);
  const inverseArea = Math.fround(1 / area);
  if (
    !(area > 0) || !Number.isFinite(area) ||
    !(inverseArea > 0) || !Number.isFinite(inverseArea)
  ) {
    return { valid: false, reason: 'unrepresentable-area' };
  }

  return {
    valid: true,
    normal: [nx, ny, nz],
    area,
    edgeScale,
  };
}

export function measureAreaVectorF32(
  uInput: AreaEmitterVec3,
  vInput: AreaEmitterVec3,
  coefficient: number,
): AreaVectorMeasure | null {
  const result = classifyAreaVectorF32(uInput, vInput, coefficient);
  if (!result.valid) return null;
  return result;
}

/** Float32-published triangle edge vectors, including the subtraction rounding. */
export function classifyTriangleAreaF32(
  a: AreaEmitterVec3,
  b: AreaEmitterVec3,
  c: AreaEmitterVec3,
): AreaVectorMeasureResult {
  const qa = [finiteF32(a[0]), finiteF32(a[1]), finiteF32(a[2])] as const;
  const qb = [finiteF32(b[0]), finiteF32(b[1]), finiteF32(b[2])] as const;
  const qc = [finiteF32(c[0]), finiteF32(c[1]), finiteF32(c[2])] as const;
  if ([...qa, ...qb, ...qc].some((value) => value == null)) {
    return { valid: false, reason: 'non-finite-input' };
  }
  const ab: AreaEmitterVec3 = [
    f32Difference(qb[0]!, qa[0]!),
    f32Difference(qb[1]!, qa[1]!),
    f32Difference(qb[2]!, qa[2]!),
  ];
  const ac: AreaEmitterVec3 = [
    f32Difference(qc[0]!, qa[0]!),
    f32Difference(qc[1]!, qa[1]!),
    f32Difference(qc[2]!, qa[2]!),
  ];
  return classifyAreaVectorF32(ab, ac, 0.5);
}

export function measureTriangleAreaF32(
  a: AreaEmitterVec3,
  b: AreaEmitterVec3,
  c: AreaEmitterVec3,
): AreaVectorMeasure | null {
  const result = classifyTriangleAreaF32(a, b, c);
  if (!result.valid) return null;
  return result;
}

/**
 * Normalize a direction before Float32 publication without first quantizing its
 * arbitrary source magnitude. Directions are scale-free: every finite nonzero
 * JS vector has a representable unit direction even when its authored
 * components are outside the finite-normal f32 magnitude range.
 */
export function normalizeDirectionF32(
  value: AreaEmitterVec3,
): [number, number, number] | null {
  if (!value.every(Number.isFinite)) return null;
  const scale = Math.max(Math.abs(value[0]), Math.abs(value[1]), Math.abs(value[2]));
  if (!(scale > 0) || !Number.isFinite(scale)) return null;
  const sx = value[0] / scale;
  const sy = value[1] / scale;
  const sz = value[2] / scale;
  const length = Math.hypot(sx, sy, sz);
  if (!(length > 0) || !Number.isFinite(length)) return null;
  const result: [number, number, number] = [
    Math.fround(sx / length),
    Math.fround(sy / length),
    Math.fround(sz / length),
  ];
  return result.every(Number.isFinite) ? result : null;
}
