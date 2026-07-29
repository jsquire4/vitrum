import { decodeAnalyticParams } from './analyticParams.js';
import { cloneSparseArray, sparseArrayOwnIndices } from './primitives.js';
import type {
  AnalyticPrimitive,
  MeshPrimitive,
  PrimitiveColorSets,
  PrimitiveUvSets,
} from './primitives.js';

export interface AnalyticPrimitiveToMeshOptions {
  /** Circumferential segments for curved shapes. Clamped to at least 3. */
  readonly segments?: number;
  /** Latitude bands for spheres. Capsules use half this count per hemisphere. */
  readonly rings?: number;
  /** Use primitive.fallbackMesh when supplied. Defaults to true. */
  readonly preferFallbackMesh?: boolean;
}

interface GeometryData {
  readonly positions: Float32Array;
  readonly normals: Float32Array;
  readonly uvs: Float32Array;
  readonly indices: Uint32Array;
}

interface GeometryBuilder {
  readonly positions: number[];
  readonly normals: number[];
  readonly uvs: number[];
  readonly indices: number[];
}

type V3 = readonly [number, number, number];

const DEFAULT_SEGMENTS = 32;
const EPS_DIM = 1e-4;
const TAU = Math.PI * 2;

export function analyticPrimitiveToMesh(
  primitive: AnalyticPrimitive,
  options: AnalyticPrimitiveToMeshOptions = {},
): MeshPrimitive {
  if (options.preferFallbackMesh !== false && primitive.fallbackMesh != null) {
    return meshFromFallback(primitive);
  }

  const segments = sanitizeSegments(options.segments);
  const rings = sanitizeRings(options.rings, segments);
  const geometry = (() => {
    switch (primitive.shape) {
      case 'sphere':
        return buildSphere(decodeAnalyticParams('sphere', primitive.params), segments, rings);
      case 'box':
        return buildBox(decodeAnalyticParams('box', primitive.params));
      case 'capsule':
        return buildCapsule(decodeAnalyticParams('capsule', primitive.params), segments, rings);
      case 'cylinder':
        return buildCylinder(decodeAnalyticParams('cylinder', primitive.params), segments);
      case 'h-channel-came':
        return buildHChannelCame(decodeAnalyticParams('h-channel-came', primitive.params));
    }
  })();

  return meshFromGeometry(primitive, geometry);
}

function meshFromGeometry(primitive: AnalyticPrimitive, geometry: GeometryData): MeshPrimitive {
  return {
    kind: 'mesh',
    id: primitive.id,
    positions: geometry.positions,
    normals: geometry.normals,
    uvs: geometry.uvs,
    indices: geometry.indices,
    material: primitive.material,
    ...(primitive.transform != null ? { transform: primitive.transform } : {}),
    ...(primitive.castShadow != null ? { castShadow: primitive.castShadow } : {}),
  };
}

function meshFromFallback(primitive: AnalyticPrimitive): MeshPrimitive {
  const fallback = primitive.fallbackMesh!;
  const castShadow = primitive.castShadow ?? fallback.castShadow;
  return {
    kind: 'mesh',
    id: primitive.id,
    positions: new Float32Array(fallback.positions),
    normals: new Float32Array(fallback.normals),
    ...(fallback.uvs != null ? { uvs: new Float32Array(fallback.uvs) } : {}),
    ...(fallback.uv1 != null ? { uv1: new Float32Array(fallback.uv1) } : {}),
    ...(fallback.uvSets != null ? { uvSets: cloneSparseStreams(fallback.uvSets) } : {}),
    ...(fallback.tangents != null ? { tangents: new Float32Array(fallback.tangents) } : {}),
    ...(fallback.colors != null ? { colors: new Float32Array(fallback.colors) } : {}),
    ...(fallback.colorSets != null ? { colorSets: cloneSparseStreams(fallback.colorSets) } : {}),
    ...(fallback.vertexColorSet !== undefined ? { vertexColorSet: fallback.vertexColorSet } : {}),
    ...(fallback.indices != null ? { indices: cloneIndices(fallback.indices) } : {}),
    material: primitive.material,
    ...(primitive.transform != null ? { transform: primitive.transform } : {}),
    ...(castShadow != null ? { castShadow } : {}),
  };
}

/**
 * Clone only the authored own lanes. Iterating numeric indices up to
 * `streams.length` turns a valid TEXCOORD_1000000/COLOR_1000000 sparse array
 * into a linear-time denial of service. The canonical sparse-key helper is
 * proportional to the number of present streams; assigning those same numeric
 * keys preserves their semantic indices, including keys beyond array-index
 * range.
 */
function cloneSparseStreams(
  streams: PrimitiveUvSets | PrimitiveColorSets,
): Array<Float32Array | undefined> {
  const clone = cloneSparseArray(streams);
  for (const index of sparseArrayOwnIndices(clone)) {
    const stream = clone[index];
    if (stream !== undefined) clone[index] = new Float32Array(stream);
  }
  return clone;
}

function cloneIndices(indices: Uint32Array | Uint16Array): Uint32Array | Uint16Array {
  return indices instanceof Uint16Array ? new Uint16Array(indices) : new Uint32Array(indices);
}

function sanitizeSegments(value: number | undefined): number {
  if (value == null || !Number.isFinite(value)) return DEFAULT_SEGMENTS;
  return Math.max(3, Math.floor(value));
}

function sanitizeRings(value: number | undefined, segments: number): number {
  if (value == null || !Number.isFinite(value)) {
    return Math.max(4, Math.floor(segments / 2));
  }
  return Math.max(2, Math.floor(value));
}

function positiveDimension(value: number): number {
  return Number.isFinite(value) ? Math.max(value, EPS_DIM) : EPS_DIM;
}

function createBuilder(): GeometryBuilder {
  return { positions: [], normals: [], uvs: [], indices: [] };
}

function pushVertex(builder: GeometryBuilder, position: V3, normal: V3, uv: readonly [number, number] = [0, 0]): number {
  const index = builder.positions.length / 3;
  builder.positions.push(position[0], position[1], position[2]);
  builder.normals.push(normal[0], normal[1], normal[2]);
  builder.uvs.push(uv[0], uv[1]);
  return index;
}

function pushTriangle(builder: GeometryBuilder, a: number, b: number, c: number): void {
  builder.indices.push(a, b, c);
}

function finish(builder: GeometryBuilder): GeometryData {
  return {
    positions: new Float32Array(builder.positions),
    normals: new Float32Array(builder.normals),
    uvs: new Float32Array(builder.uvs),
    indices: new Uint32Array(builder.indices),
  };
}

function addQuad(
  builder: GeometryBuilder,
  a: V3,
  b: V3,
  c: V3,
  d: V3,
  normal: V3,
  uvA: readonly [number, number] = [0, 0],
  uvB: readonly [number, number] = [1, 0],
  uvC: readonly [number, number] = [0, 1],
  uvD: readonly [number, number] = [1, 1],
): void {
  const i0 = pushVertex(builder, a, normal, uvA);
  const i1 = pushVertex(builder, b, normal, uvB);
  const i2 = pushVertex(builder, c, normal, uvC);
  const i3 = pushVertex(builder, d, normal, uvD);
  pushTriangle(builder, i0, i1, i2);
  pushTriangle(builder, i2, i1, i3);
}

function buildBox(params: readonly [number, number, number, number, number, number]): GeometryData {
  const [cx, cy, cz, rawHx, rawHy, rawHz] = params;
  const hx = positiveDimension(rawHx);
  const hy = positiveDimension(rawHy);
  const hz = positiveDimension(rawHz);
  const x0 = cx - hx, x1 = cx + hx;
  const y0 = cy - hy, y1 = cy + hy;
  const z0 = cz - hz, z1 = cz + hz;
  const builder = createBuilder();

  addQuad(builder, [x1, y0, z0], [x1, y1, z0], [x1, y0, z1], [x1, y1, z1], [1, 0, 0]);
  addQuad(builder, [x0, y0, z0], [x0, y0, z1], [x0, y1, z0], [x0, y1, z1], [-1, 0, 0]);
  addQuad(builder, [x0, y1, z0], [x0, y1, z1], [x1, y1, z0], [x1, y1, z1], [0, 1, 0]);
  addQuad(builder, [x0, y0, z0], [x1, y0, z0], [x0, y0, z1], [x1, y0, z1], [0, -1, 0]);
  addQuad(builder, [x0, y0, z1], [x1, y0, z1], [x0, y1, z1], [x1, y1, z1], [0, 0, 1]);
  addQuad(builder, [x0, y0, z0], [x0, y1, z0], [x1, y0, z0], [x1, y1, z0], [0, 0, -1]);

  return finish(builder);
}

interface SurfaceRow {
  readonly center: V3;
  readonly radius: number;
  readonly normalAxis: number;
  readonly normalRadial: number;
}

function buildSphere(
  params: readonly [number, number, number, number],
  segments: number,
  rings: number,
): GeometryData {
  const [cx, cy, cz, rawRadius] = params;
  const radius = positiveDimension(rawRadius);
  const center: V3 = [cx, cy, cz];
  const axis: V3 = [0, 1, 0];
  const u: V3 = [1, 0, 0];
  const v: V3 = [0, 0, 1];
  const rows: SurfaceRow[] = [
    { center: add(center, scale(axis, -radius)), radius: 0, normalAxis: -1, normalRadial: 0 },
  ];

  for (let j = 1; j < rings; j++) {
    const theta = -Math.PI / 2 + (j * Math.PI) / rings;
    const sinTheta = Math.sin(theta);
    const cosTheta = Math.cos(theta);
    rows.push({
      center: add(center, scale(axis, sinTheta * radius)),
      radius: cosTheta * radius,
      normalAxis: sinTheta,
      normalRadial: cosTheta,
    });
  }

  rows.push({ center: add(center, scale(axis, radius)), radius: 0, normalAxis: 1, normalRadial: 0 });
  return buildSurfaceRows(rows, axis, u, v, segments);
}

function buildCylinder(params: readonly [number, number, number, number, number], segments: number): GeometryData {
  const [cx, cy, cz, rawRadius, rawHalfHeight] = params;
  const radius = positiveDimension(rawRadius);
  const halfHeight = positiveDimension(rawHalfHeight);
  const center: V3 = [cx, cy, cz];
  const axis: V3 = [0, 1, 0];
  const u: V3 = [1, 0, 0];
  const v: V3 = [0, 0, 1];
  const builder = createBuilder();
  const bottomCenter = add(center, scale(axis, -halfHeight));
  const topCenter = add(center, scale(axis, halfHeight));

  appendRingSide(builder, bottomCenter, topCenter, radius, u, v, segments);
  appendCylinderCap(builder, bottomCenter, radius, -1, u, v, segments);
  appendCylinderCap(builder, topCenter, radius, 1, u, v, segments);

  return finish(builder);
}

function buildCapsule(
  params: readonly [number, number, number, number, number, number, number],
  segments: number,
  rings: number,
): GeometryData {
  const [ax, ay, az, bx, by, bz, rawRadius] = params;
  const radius = positiveDimension(rawRadius);
  const pa: V3 = [ax, ay, az];
  const pb: V3 = [bx, by, bz];
  const axisDelta = sub(pb, pa);
  const length = len(axisDelta);
  if (length <= EPS_DIM) {
    return buildSphere([ax, ay, az, radius], segments, rings);
  }

  const axis = scale(axisDelta, 1 / length);
  const basis = makeBasis(axis);
  const hemiRings = Math.max(1, Math.floor(rings / 2));
  const rows: SurfaceRow[] = [
    { center: add(pa, scale(axis, -radius)), radius: 0, normalAxis: -1, normalRadial: 0 },
  ];

  for (let j = 1; j <= hemiRings; j++) {
    const theta = -Math.PI / 2 + (j * Math.PI) / (2 * hemiRings);
    const sinTheta = Math.sin(theta);
    const cosTheta = Math.cos(theta);
    rows.push({
      center: add(pa, scale(axis, sinTheta * radius)),
      radius: cosTheta * radius,
      normalAxis: sinTheta,
      normalRadial: cosTheta,
    });
  }

  rows.push({ center: pb, radius, normalAxis: 0, normalRadial: 1 });

  for (let j = 1; j < hemiRings; j++) {
    const theta = (j * Math.PI) / (2 * hemiRings);
    const sinTheta = Math.sin(theta);
    const cosTheta = Math.cos(theta);
    rows.push({
      center: add(pb, scale(axis, sinTheta * radius)),
      radius: cosTheta * radius,
      normalAxis: sinTheta,
      normalRadial: cosTheta,
    });
  }

  rows.push({ center: add(pb, scale(axis, radius)), radius: 0, normalAxis: 1, normalRadial: 0 });
  return buildSurfaceRows(rows, axis, basis.u, basis.v, segments);
}

function buildHChannelCame(params: readonly [number, number, number, number]): GeometryData {
  const [rawLength, rawRailWidth, rawBlockHeight, rawWebThickness] = params;
  const hx = positiveDimension(rawLength * 0.5);
  const hy = positiveDimension(rawBlockHeight * 0.5);
  const hz = positiveDimension(rawRailWidth * 0.5);
  const rawT = Number.isFinite(rawWebThickness) ? rawWebThickness * 0.5 : EPS_DIM;
  const t = Math.max(Math.min(rawT, hy, hz), EPS_DIM);
  const builder = createBuilder();

  appendXCapRect(builder, hx, 1, hy - t, hy, -hz, hz);
  appendXCapRect(builder, hx, 1, -hy, -hy + t, -hz, hz);
  appendXCapRect(builder, hx, 1, -hy + t, hy - t, -t, t);
  appendXCapRect(builder, -hx, -1, hy - t, hy, -hz, hz);
  appendXCapRect(builder, -hx, -1, -hy, -hy + t, -hz, hz);
  appendXCapRect(builder, -hx, -1, -hy + t, hy - t, -t, t);

  const profile: ReadonlyArray<readonly [z: number, y: number]> = [
    [-hz, -hy],
    [hz, -hy],
    [hz, -hy + t],
    [t, -hy + t],
    [t, hy - t],
    [hz, hy - t],
    [hz, hy],
    [-hz, hy],
    [-hz, hy - t],
    [-t, hy - t],
    [-t, -hy + t],
    [-hz, -hy + t],
  ];

  for (let i = 0; i < profile.length; i++) {
    const a = profile[i]!;
    const b = profile[(i + 1) % profile.length]!;
    appendExtrudedProfileEdge(builder, hx, a, b);
  }

  return finish(builder);
}

function appendRingSide(
  builder: GeometryBuilder,
  bottomCenter: V3,
  topCenter: V3,
  radius: number,
  u: V3,
  v: V3,
  segments: number,
): void {
  const bottom: number[] = [];
  const top: number[] = [];
  // Duplicate the azimuth-zero vertices at u=1. Sharing a single vertex
  // between u=0 and the final segment interpolates almost the full texture
  // width across that segment instead of crossing the wrap seam.
  for (let i = 0; i <= segments; i++) {
    // u = azimuth [0,1], v = 0 at bottom, 1 at top.
    const uCoord = i / segments;
    const radial = circleDirection(u, v, i, segments);
    const normal = radial;
    bottom.push(pushVertex(builder, add(bottomCenter, scale(radial, radius)), normal, [uCoord, 0]));
    top.push(pushVertex(builder, add(topCenter, scale(radial, radius)), normal, [uCoord, 1]));
  }

  for (let i = 0; i < segments; i++) {
    const next = i + 1;
    pushTriangle(builder, bottom[i]!, top[i]!, bottom[next]!);
    pushTriangle(builder, bottom[next]!, top[i]!, top[next]!);
  }
}

function appendCylinderCap(
  builder: GeometryBuilder,
  center: V3,
  radius: number,
  axisSign: -1 | 1,
  u: V3,
  v: V3,
  segments: number,
): void {
  const normal: V3 = [0, axisSign, 0];
  // Cap center at UV (0.5, 0.5); ring vertices at (cos+1)/2, (sin+1)/2.
  const centerIndex = pushVertex(builder, center, normal, [0.5, 0.5]);
  const ring: number[] = [];
  for (let i = 0; i < segments; i++) {
    const radial = circleDirection(u, v, i, segments);
    // radial is already normalized [cos,0,sin]; map to [0,1]² cap UV.
    const uCap = (radial[0] + 1) * 0.5;
    const vCap = (radial[2] + 1) * 0.5;
    ring.push(pushVertex(builder, add(center, scale(radial, radius)), normal, [uCap, vCap]));
  }
  for (let i = 0; i < segments; i++) {
    const next = (i + 1) % segments;
    if (axisSign > 0) {
      pushTriangle(builder, centerIndex, ring[next]!, ring[i]!);
    } else {
      pushTriangle(builder, centerIndex, ring[i]!, ring[next]!);
    }
  }
}

function buildSurfaceRows(
  rows: readonly SurfaceRow[],
  axis: V3,
  u: V3,
  v: V3,
  segments: number,
): GeometryData {
  const builder = createBuilder();
  const rowIndices: number[][] = [];
  const totalRows = rows.length;

  for (let rowIdx = 0; rowIdx < totalRows; rowIdx++) {
    const row = rows[rowIdx]!;
    // v-coordinate: 0 at south pole (first row), 1 at north pole (last row).
    const vCoord = rowIdx / (totalRows - 1);

    if (row.radius <= EPS_DIM) {
      // Pole vertex — single vertex at UV (0.5, vCoord). The standard
      // lat/long convention uses the column-average u for the pole (0.5).
      // This keeps vertex counts unchanged from the pre-UV implementation.
      rowIndices.push([pushVertex(builder, row.center, normalize(scale(axis, row.normalAxis)), [0.5, vCoord])]);
      continue;
    }

    const indices: number[] = [];
    // Non-pole rows carry a duplicate u=1 endpoint so no indexed triangle
    // interpolates from (segments - 1) / segments directly back to u=0.
    for (let i = 0; i <= segments; i++) {
      const uCoord = i / segments;
      const radial = circleDirection(u, v, i, segments);
      const position = add(row.center, scale(radial, row.radius));
      const normal = normalize(add(scale(radial, row.normalRadial), scale(axis, row.normalAxis)));
      indices.push(pushVertex(builder, position, normal, [uCoord, vCoord]));
    }
    rowIndices.push(indices);
  }

  for (let row = 0; row < rowIndices.length - 1; row++) {
    const lower = rowIndices[row]!;
    const upper = rowIndices[row + 1]!;
    if (lower.length === 1 && upper.length > 1) {
      for (let i = 0; i < segments; i++) {
        pushTriangle(builder, lower[0]!, upper[i]!, upper[i + 1]!);
      }
    } else if (lower.length > 1 && upper.length === 1) {
      for (let i = 0; i < segments; i++) {
        pushTriangle(builder, lower[i]!, upper[0]!, lower[i + 1]!);
      }
    } else {
      for (let i = 0; i < segments; i++) {
        const next = i + 1;
        pushTriangle(builder, lower[i]!, upper[i]!, lower[next]!);
        pushTriangle(builder, lower[next]!, upper[i]!, upper[next]!);
      }
    }
  }

  return finish(builder);
}

function appendXCapRect(
  builder: GeometryBuilder,
  x: number,
  normalX: -1 | 1,
  y0: number,
  y1: number,
  z0: number,
  z1: number,
): void {
  if (normalX > 0) {
    addQuad(builder, [x, y0, z0], [x, y1, z0], [x, y0, z1], [x, y1, z1], [1, 0, 0]);
  } else {
    addQuad(builder, [x, y0, z0], [x, y0, z1], [x, y1, z0], [x, y1, z1], [-1, 0, 0]);
  }
}

function appendExtrudedProfileEdge(
  builder: GeometryBuilder,
  hx: number,
  a: readonly [z: number, y: number],
  b: readonly [z: number, y: number],
): void {
  const dz = b[0] - a[0];
  const dy = b[1] - a[1];
  const normal = normalize([0, -dz, dy]);
  addQuad(
    builder,
    [-hx, a[1], a[0]],
    [hx, a[1], a[0]],
    [-hx, b[1], b[0]],
    [hx, b[1], b[0]],
    normal,
  );
}

function circleDirection(u: V3, v: V3, i: number, segments: number): V3 {
  const phi = (i * TAU) / segments;
  return add(scale(u, Math.cos(phi)), scale(v, Math.sin(phi)));
}

function makeBasis(axis: V3): { readonly u: V3; readonly v: V3 } {
  const helper: V3 = Math.abs(axis[1]) < 0.999 ? [0, 1, 0] : [1, 0, 0];
  const u = normalize(cross(helper, axis));
  return { u, v: normalize(cross(axis, u)) };
}

function add(a: V3, b: V3): V3 {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

function sub(a: V3, b: V3): V3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
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

function len(a: V3): number {
  return Math.hypot(a[0], a[1], a[2]);
}

function normalize(a: V3): V3 {
  const l = len(a);
  return l > 0 ? [a[0] / l, a[1] / l, a[2] / l] : [0, 1, 0];
}
