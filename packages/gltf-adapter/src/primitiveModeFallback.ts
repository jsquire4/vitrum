// primitiveModeFallback.ts — deterministic mesh fallback for glTF point/line topologies.
//
// glTF modes POINTS/LINES/LINE_LOOP/LINE_STRIP have no first-class @vitrum/core
// primitive kind.  Rather than dropping those assets, the adapter expands them
// into small triangle meshes: points become tiny cubes and line segments become
// thin rectangular prisms.  This is intentionally reported as
// fallback-generated-mesh, not native topology support.

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
  readonly positions: number[];
  readonly normals: number[];
  readonly indices: number[];
  readonly sourceVertices: number[];
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
): PointLineFallbackGeometry | null {
  const vertexCount = Math.floor(positions.length / 3);
  if (vertexCount <= 0) return null;
  const source = indices ?? sequential(vertexCount);
  const radius = sanitizeRadius(radiusOverride, estimateRadius(positions));
  const builder: Builder = { positions: [], normals: [], indices: [], sourceVertices: [] };

  if (mode === GLTF_MODE_POINTS) {
    for (const index of source) {
      if (index >= vertexCount) continue;
      pushPointCube(builder, readPoint(positions, index), radius, index);
    }
  } else {
    for (const [a, b] of lineSegments(source, mode)) {
      if (a >= vertexCount || b >= vertexCount || a === b) continue;
      pushLinePrism(builder, readPoint(positions, a), readPoint(positions, b), radius, a, b);
    }
  }

  if (builder.indices.length === 0) return null;
  return {
    positions: new Float32Array(builder.positions),
    normals: new Float32Array(builder.normals),
    indices: new Uint32Array(builder.indices),
    sourceVertices: new Uint32Array(builder.sourceVertices),
    radius,
  };
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

function sequential(count: number): Uint32Array {
  const out = new Uint32Array(count);
  for (let i = 0; i < count; i += 1) out[i] = i;
  return out;
}

function readPoint(positions: Float32Array, index: number): V3 {
  const i = index * 3;
  return [positions[i] ?? 0, positions[i + 1] ?? 0, positions[i + 2] ?? 0];
}

function* lineSegments(
  source: Uint32Array,
  mode: Exclude<GltfPointLineMode, typeof GLTF_MODE_POINTS>,
): Generator<readonly [number, number]> {
  if (mode === GLTF_MODE_LINES) {
    for (let i = 0; i + 1 < source.length; i += 2) yield [source[i]!, source[i + 1]!];
    return;
  }
  for (let i = 0; i + 1 < source.length; i += 1) yield [source[i]!, source[i + 1]!];
  if (mode === GLTF_MODE_LINE_LOOP && source.length > 2) {
    yield [source[source.length - 1]!, source[0]!];
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
  const dir = normalize(sub(b, a));
  if (dir == null) return;
  const helper: V3 = Math.abs(dir[2]) < 0.9 ? [0, 0, 1] : [0, 1, 0];
  const u = normalize(cross(dir, helper));
  if (u == null) return;
  const v = normalize(cross(u, dir));
  if (v == null) return;
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

function pushQuad(
  builder: Builder,
  p0: V3,
  p1: V3,
  p2: V3,
  p3: V3,
  normal: V3,
  source: readonly [number, number, number, number],
): void {
  const base = builder.positions.length / 3;
  pushVertex(builder, p0, normal, source[0]);
  pushVertex(builder, p1, normal, source[1]);
  pushVertex(builder, p2, normal, source[2]);
  pushVertex(builder, p3, normal, source[3]);
  builder.indices.push(base, base + 1, base + 2, base, base + 2, base + 3);
}

function pushVertex(builder: Builder, position: V3, normal: V3, sourceIndex: number): void {
  builder.positions.push(position[0], position[1], position[2]);
  builder.normals.push(normal[0], normal[1], normal[2]);
  builder.sourceVertices.push(sourceIndex);
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
