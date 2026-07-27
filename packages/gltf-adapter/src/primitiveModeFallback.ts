// primitiveModeFallback.ts — deterministic mesh fallback for glTF point/line topologies.
//
// glTF modes POINTS/LINES/LINE_LOOP/LINE_STRIP have no first-class @vitrum/core
// primitive kind.  Rather than dropping those assets, the adapter expands them
// into small triangle meshes: points become tiny cubes and line segments become
// thin rectangular prisms.  This is intentionally reported as
// fallback-generated-mesh, not native topology support.

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

export interface PointLineFallbackGeometry {
  readonly positions: Float32Array;
  readonly normals: Float32Array;
  readonly indices: Uint32Array;
  /** Original source-vertex index for each generated vertex. */
  readonly sourceVertices: Uint32Array;
  /** Half-width used for generated cubes/prisms, in source asset units. */
  readonly radius: number;
}

interface Builder {
  readonly positions: Float32Array;
  readonly normals: Float32Array;
  readonly indices: Uint32Array;
  readonly sourceVertices: Uint32Array;
  vertexOffset: number;
  indexOffset: number;
}

type V3 = readonly [number, number, number];

const FALLBACK_RADIUS_FRACTION = 0.0025;
const MIN_FALLBACK_RADIUS = 1e-4;
const DEFAULT_DEGENERATE_RADIUS = 0.01;

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

export function buildPointLineFallbackGeometry(
  positions: Float32Array,
  indices: Uint32Array | undefined,
  mode: GltfPointLineMode,
  radiusOverride?: number,
  resourceLedger?: ImportResourceLedger,
  allocationPath = 'point/line fallback geometry',
): PointLineFallbackGeometry | null {
  const vertexCount = Math.floor(positions.length / 3);
  if (vertexCount <= 0) return null;
  const radius = sanitizeRadius(radiusOverride, estimateRadius(positions));
  let generatedElementCount = 0;

  if (mode === GLTF_MODE_POINTS) {
    for (const index of sourceIndices(indices, vertexCount)) {
      if (index < vertexCount) generatedElementCount += 1;
    }
  } else {
    for (const [a, b] of lineSegments(indices, vertexCount, mode)) {
      if (a >= vertexCount || b >= vertexCount || a === b) continue;
      if (linePrismFrame(readPoint(positions, a), readPoint(positions, b)) !== null) {
        generatedElementCount += 1;
      }
    }
  }

  if (generatedElementCount === 0) return null;
  const generatedVertexCount = checkedProduct(
    generatedElementCount,
    24,
    `${allocationPath} vertex count`,
  );
  const generatedIndexCount = checkedProduct(
    generatedElementCount,
    36,
    `${allocationPath} index count`,
  );
  const positionElementCount = checkedProduct(
    generatedVertexCount,
    3,
    `${allocationPath} position component count`,
  );
  const outputByteLength =
    checkedProduct(
      positionElementCount,
      Float32Array.BYTES_PER_ELEMENT * 2,
      `${allocationPath} position/normal bytes`,
    ) +
    checkedProduct(
      generatedIndexCount,
      Uint32Array.BYTES_PER_ELEMENT,
      `${allocationPath} index bytes`,
    ) +
    checkedProduct(
      generatedVertexCount,
      Uint32Array.BYTES_PER_ELEMENT,
      `${allocationPath} source-vertex bytes`,
    );
  if (!Number.isSafeInteger(outputByteLength)) {
    throw new RangeError(
      `[vitrum/gltf-adapter] ${allocationPath} byte length exceeds the safe integer range.`,
    );
  }
  resourceLedger?.chargeDecodedGeometryBytes(outputByteLength, allocationPath);
  const builder: Builder = {
    positions: new Float32Array(positionElementCount),
    normals: new Float32Array(positionElementCount),
    indices: new Uint32Array(generatedIndexCount),
    sourceVertices: new Uint32Array(generatedVertexCount),
    vertexOffset: 0,
    indexOffset: 0,
  };

  if (mode === GLTF_MODE_POINTS) {
    for (const index of sourceIndices(indices, vertexCount)) {
      if (index >= vertexCount) continue;
      pushPointCube(builder, readPoint(positions, index), radius, index);
    }
  } else {
    for (const [a, b] of lineSegments(indices, vertexCount, mode)) {
      if (a >= vertexCount || b >= vertexCount || a === b) continue;
      pushLinePrism(builder, readPoint(positions, a), readPoint(positions, b), radius, a, b);
    }
  }

  if (
    builder.vertexOffset !== generatedVertexCount ||
    builder.indexOffset !== generatedIndexCount
  ) {
    throw new Error(
      '[vitrum/gltf-adapter] Internal point/line fallback allocation count mismatch.',
    );
  }
  return {
    positions: builder.positions,
    normals: builder.normals,
    indices: builder.indices,
    sourceVertices: builder.sourceVertices,
    radius,
  };
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

function pushPointCube(builder: Builder, center: V3, radius: number, sourceIndex: number): void {
  const [x, y, z] = center;
  const r = radius;
  const p000: V3 = [x - r, y - r, z - r];
  const p100: V3 = [x + r, y - r, z - r];
  const p110: V3 = [x + r, y + r, z - r];
  const p010: V3 = [x - r, y + r, z - r];
  const p001: V3 = [x - r, y - r, z + r];
  const p101: V3 = [x + r, y - r, z + r];
  const p111: V3 = [x + r, y + r, z + r];
  const p011: V3 = [x - r, y + r, z + r];

  pushQuad(builder, p100, p101, p111, p110, [1, 0, 0], [sourceIndex, sourceIndex, sourceIndex, sourceIndex]);
  pushQuad(builder, p001, p000, p010, p011, [-1, 0, 0], [sourceIndex, sourceIndex, sourceIndex, sourceIndex]);
  pushQuad(builder, p010, p110, p111, p011, [0, 1, 0], [sourceIndex, sourceIndex, sourceIndex, sourceIndex]);
  pushQuad(builder, p000, p001, p101, p100, [0, -1, 0], [sourceIndex, sourceIndex, sourceIndex, sourceIndex]);
  pushQuad(builder, p001, p011, p111, p101, [0, 0, 1], [sourceIndex, sourceIndex, sourceIndex, sourceIndex]);
  pushQuad(builder, p000, p100, p110, p010, [0, 0, -1], [sourceIndex, sourceIndex, sourceIndex, sourceIndex]);
}

function pushLinePrism(
  builder: Builder,
  a: V3,
  b: V3,
  radius: number,
  sourceA: number,
  sourceB: number,
): void {
  const frame = linePrismFrame(a, b);
  if (frame == null) return;
  const { dir, u, v } = frame;
  const ru = scale(u, radius);
  const rv = scale(v, radius);

  const a00 = sub(sub(a, ru), rv);
  const a10 = sub(add(a, ru), rv);
  const a11 = add(add(a, ru), rv);
  const a01 = add(sub(a, ru), rv);
  const b00 = sub(sub(b, ru), rv);
  const b10 = sub(add(b, ru), rv);
  const b11 = add(add(b, ru), rv);
  const b01 = add(sub(b, ru), rv);

  pushQuad(builder, a00, b00, b10, a10, scale(v, -1), [sourceA, sourceB, sourceB, sourceA]);
  pushQuad(builder, a10, b10, b11, a11, u, [sourceA, sourceB, sourceB, sourceA]);
  pushQuad(builder, a11, b11, b01, a01, v, [sourceA, sourceB, sourceB, sourceA]);
  pushQuad(builder, a01, b01, b00, a00, scale(u, -1), [sourceA, sourceB, sourceB, sourceA]);
  pushQuad(builder, a01, a00, a10, a11, scale(dir, -1), [sourceA, sourceA, sourceA, sourceA]);
  pushQuad(builder, b00, b01, b11, b10, dir, [sourceB, sourceB, sourceB, sourceB]);
}

interface LinePrismFrame {
  readonly dir: V3;
  readonly u: V3;
  readonly v: V3;
}

function linePrismFrame(a: V3, b: V3): LinePrismFrame | null {
  const dir = normalize(sub(b, a));
  if (dir == null) return null;
  const helper: V3 = Math.abs(dir[2]) < 0.9 ? [0, 0, 1] : [0, 1, 0];
  const u = normalize(cross(dir, helper));
  if (u == null) return null;
  const v = normalize(cross(u, dir));
  if (v == null) return null;
  return { dir, u, v };
}

function pushQuad(
  builder: Builder,
  p0: V3,
  p1: V3,
  p2: V3,
  p3: V3,
  normal: V3,
  source: readonly [number, number, number, number],
): void {
  const base = builder.vertexOffset;
  pushVertex(builder, p0, normal, source[0]);
  pushVertex(builder, p1, normal, source[1]);
  pushVertex(builder, p2, normal, source[2]);
  pushVertex(builder, p3, normal, source[3]);
  const indexOffset = builder.indexOffset;
  builder.indices[indexOffset] = base;
  builder.indices[indexOffset + 1] = base + 1;
  builder.indices[indexOffset + 2] = base + 2;
  builder.indices[indexOffset + 3] = base;
  builder.indices[indexOffset + 4] = base + 2;
  builder.indices[indexOffset + 5] = base + 3;
  builder.indexOffset += 6;
}

function pushVertex(builder: Builder, position: V3, normal: V3, sourceIndex: number): void {
  const vertex = builder.vertexOffset;
  const offset = vertex * 3;
  builder.positions[offset] = position[0];
  builder.positions[offset + 1] = position[1];
  builder.positions[offset + 2] = position[2];
  builder.normals[offset] = normal[0];
  builder.normals[offset + 1] = normal[1];
  builder.normals[offset + 2] = normal[2];
  builder.sourceVertices[vertex] = sourceIndex;
  builder.vertexOffset += 1;
}

function sub(a: V3, b: V3): V3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function add(a: V3, b: V3): V3 {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

function scale(a: V3, s: number): V3 {
  return [a[0] * s, a[1] * s, a[2] * s];
}

function cross(a: V3, b: V3): V3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function normalize(a: V3): V3 | null {
  const len = Math.hypot(a[0], a[1], a[2]);
  if (!Number.isFinite(len) || len <= 1e-12) return null;
  return [a[0] / len, a[1] / len, a[2] / len];
}
