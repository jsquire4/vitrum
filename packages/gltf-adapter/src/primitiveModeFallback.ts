// primitiveModeFallback.ts — analytic sphere/capsule import for glTF point/line
// topologies.
//
// glTF modes POINTS/LINES/LINE_LOOP/LINE_STRIP have no first-class @vitrum/core
// primitive kind. The adapter maps them onto existing analytic shapes: each
// point becomes a sphere and each non-degenerate line segment becomes a
// capsule. Engines with native analytic intersect keep those shapes; engines
// that only pack triangles tessellate them later. Compatibility reports that
// backend grade instead of claiming a missing point/line contract.

import type { ImportResourceLedger } from './importResourceBudget.js';

export const GLTF_MODE_POINTS = 0;
export const GLTF_MODE_LINES = 1;
export const GLTF_MODE_LINE_LOOP = 2;
export const GLTF_MODE_LINE_STRIP = 3;

export type GltfPointLineMode =
  | typeof GLTF_MODE_POINTS
  | typeof GLTF_MODE_LINES
  | typeof GLTF_MODE_LINE_LOOP
  | typeof GLTF_MODE_LINE_STRIP;

export interface PointLineAnalytic {
  readonly shape: 'sphere' | 'capsule';
  readonly params: Float32Array;
}

export interface PointLineAnalytics {
  readonly analytics: readonly PointLineAnalytic[];
  readonly radius: number;
}

type V3 = readonly [number, number, number];

const FALLBACK_RADIUS_FRACTION = 0.0025;
const MIN_FALLBACK_RADIUS = 1e-4;
const DEFAULT_DEGENERATE_RADIUS = 0.01;
const SPHERE_PARAM_COUNT = 4;
const CAPSULE_PARAM_COUNT = 7;
const FLOAT32_BYTES = Float32Array.BYTES_PER_ELEMENT;

export function isPointLineMode(mode: number): mode is GltfPointLineMode {
  return (
    mode === GLTF_MODE_POINTS ||
    mode === GLTF_MODE_LINES ||
    mode === GLTF_MODE_LINE_LOOP ||
    mode === GLTF_MODE_LINE_STRIP
  );
}

export function pointLineModeName(mode: number): string {
  switch (mode) {
    case GLTF_MODE_POINTS: return 'POINTS';
    case GLTF_MODE_LINES: return 'LINES';
    case GLTF_MODE_LINE_LOOP: return 'LINE_LOOP';
    case GLTF_MODE_LINE_STRIP: return 'LINE_STRIP';
    default: return 'UNKNOWN';
  }
}

/**
 * Authored radius from `extras.vitrum.radius`. Returns undefined when the
 * extras bag is missing or the value is not a finite positive number.
 */
export function readVitrumPointLineRadius(extras: unknown): number | undefined {
  if (!isRecord(extras)) return undefined;
  const vitrum = extras.vitrum;
  if (!isRecord(vitrum)) return undefined;
  const radius = vitrum.radius;
  if (typeof radius !== 'number' || !Number.isFinite(radius) || radius <= 0) {
    return undefined;
  }
  return radius;
}

/**
 * Radius resolution: primitive/mesh/node `extras.vitrum.radius`, then the host
 * `pointLineFallbackRadius` override, then a deterministic AABB-diagonal
 * default.
 */
export function resolvePointLineRadius(
  positions: Float32Array,
  extrasSources: readonly unknown[],
  hostOverride?: number,
): number {
  for (const extras of extrasSources) {
    const radius = readVitrumPointLineRadius(extras);
    if (radius != null) return radius;
  }
  return sanitizeRadius(hostOverride, estimateRadius(positions));
}

export function buildPointLineAnalytics(
  positions: Float32Array,
  indices: Uint32Array | undefined,
  mode: GltfPointLineMode,
  radius: number,
  resourceLedger?: ImportResourceLedger,
  allocationPath = 'point/line analytics',
): PointLineAnalytics | null {
  const vertexCount = Math.floor(positions.length / 3);
  if (vertexCount <= 0) return null;
  const sanitizedRadius = sanitizeRadius(radius, estimateRadius(positions));

  const pending: Array<{ shape: 'sphere' | 'capsule'; values: number[] }> = [];
  if (mode === GLTF_MODE_POINTS) {
    for (const index of sourceIndices(indices, vertexCount)) {
      if (index >= vertexCount) continue;
      const [x, y, z] = readPoint(positions, index);
      pending.push({ shape: 'sphere', values: [x, y, z, sanitizedRadius] });
    }
  } else {
    for (const [a, b] of lineSegments(indices, vertexCount, mode)) {
      if (a >= vertexCount || b >= vertexCount || a === b) continue;
      const start = readPoint(positions, a);
      const end = readPoint(positions, b);
      if (!isFinitePoint(start) || !isFinitePoint(end) || pointsEqual(start, end)) continue;
      pending.push({
        shape: 'capsule',
        values: [start[0], start[1], start[2], end[0], end[1], end[2], sanitizedRadius],
      });
    }
  }

  if (pending.length === 0) return null;

  let paramFloats = 0;
  for (const item of pending) {
    paramFloats = checkedSum(
      paramFloats,
      item.shape === 'sphere' ? SPHERE_PARAM_COUNT : CAPSULE_PARAM_COUNT,
      `${allocationPath} param count`,
    );
  }
  const outputByteLength = checkedProduct(
    paramFloats,
    FLOAT32_BYTES,
    `${allocationPath} param bytes`,
  );
  resourceLedger?.chargeDecodedGeometryBytes(outputByteLength, allocationPath);

  const analytics: PointLineAnalytic[] = pending.map((item) => ({
    shape: item.shape,
    params: Float32Array.from(item.values),
  }));
  return { analytics, radius: sanitizedRadius };
}

function checkedProduct(left: number, right: number, path: string): number {
  if (
    !Number.isSafeInteger(left) ||
    left < 0 ||
    !Number.isSafeInteger(right) ||
    right < 0 ||
    (left !== 0 && right > Math.floor(Number.MAX_SAFE_INTEGER / left))
  ) {
    throw new RangeError(`[vitrum/gltf-adapter] ${path} exceeds the safe integer range.`);
  }
  return left * right;
}

function checkedSum(left: number, right: number, path: string): number {
  if (
    !Number.isSafeInteger(left) ||
    left < 0 ||
    !Number.isSafeInteger(right) ||
    right < 0 ||
    left > Number.MAX_SAFE_INTEGER - right
  ) {
    throw new RangeError(`[vitrum/gltf-adapter] ${path} exceeds the safe integer range.`);
  }
  return left + right;
}

function sanitizeRadius(value: number | undefined, fallback: number): number {
  if (value == null || !Number.isFinite(value) || value <= 0) return fallback;
  return value;
}

function estimateRadius(positions: Float32Array): number {
  let minX = Infinity; let minY = Infinity; let minZ = Infinity;
  let maxX = -Infinity; let maxY = -Infinity; let maxZ = -Infinity;
  for (let i = 0; i + 2 < positions.length; i += 3) {
    const x = positions[i]!;
    const y = positions[i + 1]!;
    const z = positions[i + 2]!;
    minX = Math.min(minX, x); minY = Math.min(minY, y); minZ = Math.min(minZ, z);
    maxX = Math.max(maxX, x); maxY = Math.max(maxY, y); maxZ = Math.max(maxZ, z);
  }
  const dx = maxX - minX;
  const dy = maxY - minY;
  const dz = maxZ - minZ;
  const diag = Math.hypot(dx, dy, dz);
  if (!Number.isFinite(diag) || diag <= 0) return DEFAULT_DEGENERATE_RADIUS;
  return Math.max(MIN_FALLBACK_RADIUS, diag * FALLBACK_RADIUS_FRACTION);
}

function* sourceIndices(
  indices: Uint32Array | undefined,
  vertexCount: number,
): Generator<number> {
  if (indices !== undefined) {
    for (let index = 0; index < indices.length; index += 1) {
      yield indices[index]!;
    }
    return;
  }
  for (let index = 0; index < vertexCount; index += 1) yield index;
}

function readPoint(positions: Float32Array, index: number): V3 {
  const i = index * 3;
  return [positions[i] ?? 0, positions[i + 1] ?? 0, positions[i + 2] ?? 0];
}

function isFinitePoint(point: V3): boolean {
  return Number.isFinite(point[0]) && Number.isFinite(point[1]) && Number.isFinite(point[2]);
}

function pointsEqual(a: V3, b: V3): boolean {
  return a[0] === b[0] && a[1] === b[1] && a[2] === b[2];
}

function* lineSegments(
  indices: Uint32Array | undefined,
  vertexCount: number,
  mode: Exclude<GltfPointLineMode, typeof GLTF_MODE_POINTS>,
): Generator<readonly [number, number]> {
  const length = indices?.length ?? vertexCount;
  const at = (index: number): number => indices?.[index] ?? index;
  if (mode === GLTF_MODE_LINES) {
    for (let i = 0; i + 1 < length; i += 2) yield [at(i), at(i + 1)];
    return;
  }
  for (let i = 0; i + 1 < length; i += 1) yield [at(i), at(i + 1)];
  if (mode === GLTF_MODE_LINE_LOOP && length > 2) {
    yield [at(length - 1), at(0)];
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value != null && typeof value === 'object' && !Array.isArray(value);
}
