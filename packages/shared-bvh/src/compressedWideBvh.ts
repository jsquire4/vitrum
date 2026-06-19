import {
  buildArrayBvh,
  isLeafSplit,
  type BuildArrayBvhOpts,
  type CpuBvhBuildResult,
} from './buildArrayBvh.js';
import {
  CWBVH_CHILDREN,
  CWBVH_CHILD_BOUNDS_PACKED_U32,
  CWBVH_CHILD_BOUNDS_U16,
  CWBVH_CHILD_META_WORDS,
} from './strides.js';

export const CWBVH_CHILD_EMPTY = 0 as const;
export const CWBVH_CHILD_NODE = 1 as const;
export const CWBVH_CHILD_LEAF = 2 as const;

export type CwbvhChildKind =
  | typeof CWBVH_CHILD_EMPTY
  | typeof CWBVH_CHILD_NODE
  | typeof CWBVH_CHILD_LEAF;

export interface CompressedWideBvhBuildResult extends CpuBvhBuildResult {
  /** Parent-node bounds, 6 f32 words per CWBVH node: min xyz, max xyz. */
  readonly cwbvhNodeBounds: Float32Array;
  /**
   * Child bounds, quantized u16 relative to each parent node's bounds.
   * Layout: node * 8 * 6 + child * 6 + boundWord.
   */
  readonly cwbvhChildBounds: Uint16Array;
  /**
   * Child metadata. Layout: node * 8 * 3 + child * 3:
   *   word0 kind: 0 empty, 1 child wide-node, 2 leaf range
   *   word1 child wide-node index or leaf triangle offset
   *   word2 leaf triangle count, zero for child wide-nodes
   */
  readonly cwbvhChildMeta: Uint32Array;
  /** Number of non-empty child slots per wide node. */
  readonly cwbvhChildCount: Uint32Array;
  readonly cwbvhNodeCount: number;
}

export interface CwbvhIntersection {
  readonly didHit: boolean;
  readonly dist: number;
  /** Triangle index in BVH-reordered order. */
  readonly triangleIndex: number;
  /** Original source triangle index before the binary BVH reorder. */
  readonly sourceTriangleIndex: number;
  readonly bary: readonly [number, number, number];
}

export interface CwbvhRay {
  readonly origin: readonly [number, number, number];
  readonly direction: readonly [number, number, number];
}

export interface CwbvhTraverseOptions {
  readonly positionStride?: number;
  readonly indexStride?: number;
  readonly triEps?: number;
  readonly tMin?: number;
  readonly tMax?: number;
}

interface BinaryNodeInfo {
  readonly min: readonly [number, number, number];
  readonly max: readonly [number, number, number];
  readonly rightChildOrTriOffset: number;
  readonly splitAxisOrTriCount: number;
}

function readBinaryNode(bvhNodes: Float32Array, nodeIndex: number): BinaryNodeInfo {
  const dv = new DataView(bvhNodes.buffer, bvhNodes.byteOffset, bvhNodes.byteLength);
  const off = nodeIndex * 32;
  return {
    min: [
      dv.getFloat32(off + 0, true),
      dv.getFloat32(off + 4, true),
      dv.getFloat32(off + 8, true),
    ],
    max: [
      dv.getFloat32(off + 12, true),
      dv.getFloat32(off + 16, true),
      dv.getFloat32(off + 20, true),
    ],
    rightChildOrTriOffset: dv.getUint32(off + 24, true),
    splitAxisOrTriCount: dv.getUint32(off + 28, true),
  };
}

function surfaceArea(min: readonly [number, number, number], max: readonly [number, number, number]): number {
  const dx = Math.max(0, max[0] - min[0]);
  const dy = Math.max(0, max[1] - min[1]);
  const dz = Math.max(0, max[2] - min[2]);
  return dx * dy + dy * dz + dz * dx;
}

function leftChildOf(nodeIndex: number): number {
  return nodeIndex + 1;
}

function rightChildOf(bvhNodes: Float32Array, nodeIndex: number): number {
  const node = readBinaryNode(bvhNodes, nodeIndex);
  return nodeIndex + node.rightChildOrTriOffset;
}

function clampU16(v: number): number {
  if (!Number.isFinite(v)) return 0;
  return Math.max(0, Math.min(65535, Math.round(v)));
}

function quantizeBound(value: number, min: number, max: number, mode: 'floor' | 'ceil'): number {
  const extent = max - min;
  if (!(extent > 0)) return mode === 'floor' ? 0 : 65535;
  const q = ((value - min) / extent) * 65535;
  return clampU16(mode === 'floor' ? Math.floor(q) : Math.ceil(q));
}

function dequantizeBound(q: number, min: number, max: number): number {
  const extent = max - min;
  if (!(extent > 0)) return min;
  return min + (q / 65535) * extent;
}

function collectWideChildren(bvhNodes: Float32Array, binaryRoot: number): number[] {
  const root = readBinaryNode(bvhNodes, binaryRoot);
  if (isLeafSplit(root.splitAxisOrTriCount)) return [binaryRoot];

  const children = [leftChildOf(binaryRoot), rightChildOf(bvhNodes, binaryRoot)];
  while (children.length < CWBVH_CHILDREN) {
    let expandAt = -1;
    let bestArea = -1;
    for (let i = 0; i < children.length; i += 1) {
      const n = readBinaryNode(bvhNodes, children[i]!);
      if (isLeafSplit(n.splitAxisOrTriCount)) continue;
      const area = surfaceArea(n.min, n.max);
      if (area > bestArea) {
        bestArea = area;
        expandAt = i;
      }
    }
    if (expandAt < 0) break;
    const nodeToExpand = children[expandAt]!;
    children.splice(
      expandAt,
      1,
      leftChildOf(nodeToExpand),
      rightChildOf(bvhNodes, nodeToExpand),
    );
  }
  return children;
}

/**
 * Collapse the canonical binary BVH into a packed 8-wide, quantized-bounds CPU
 * representation. This is deliberately a build/oracle primitive: renderer
 * backends remain on the binary BVH until a WGSL traversal and A/B proof land.
 */
export function buildCompressedWideBvhFromArrayBvh(
  binary: CpuBvhBuildResult,
): CompressedWideBvhBuildResult {
  const binaryNodeCount = Math.floor(binary.bvhNodes.length / 8);
  const emptyWideRoot = (): CompressedWideBvhBuildResult => ({
    ...binary,
    cwbvhNodeBounds: new Float32Array(6),
    cwbvhChildBounds: new Uint16Array(CWBVH_CHILDREN * CWBVH_CHILD_BOUNDS_U16),
    cwbvhChildMeta: new Uint32Array(CWBVH_CHILDREN * CWBVH_CHILD_META_WORDS),
    cwbvhChildCount: new Uint32Array(1),
    cwbvhNodeCount: 1,
  });
  if (binaryNodeCount === 0) {
    return {
      ...binary,
      cwbvhNodeBounds: new Float32Array(0),
      cwbvhChildBounds: new Uint16Array(0),
      cwbvhChildMeta: new Uint32Array(0),
      cwbvhChildCount: new Uint32Array(0),
      cwbvhNodeCount: 0,
    };
  }
  if (binary.reorderedIndices.length === 0) {
    return emptyWideRoot();
  }

  const root = readBinaryNode(binary.bvhNodes, 0);
  if (binaryNodeCount === 1 && isLeafSplit(root.splitAxisOrTriCount) && (root.splitAxisOrTriCount & 0xffff) === 0) {
    return emptyWideRoot();
  }

  const nodeBounds: number[] = [];
  const childBounds: number[] = [];
  const childMeta: number[] = [];
  const childCount: number[] = [];

  const allocNode = (node: BinaryNodeInfo): number => {
    const nodeIndex = nodeBounds.length / 6;
    nodeBounds.push(node.min[0], node.min[1], node.min[2], node.max[0], node.max[1], node.max[2]);
    childCount.push(0);
    for (let i = 0; i < CWBVH_CHILDREN * CWBVH_CHILD_BOUNDS_U16; i += 1) childBounds.push(0);
    for (let i = 0; i < CWBVH_CHILDREN * CWBVH_CHILD_META_WORDS; i += 1) childMeta.push(0);
    return nodeIndex;
  };

  const fillChildBounds = (
    wideNodeIndex: number,
    slot: number,
    parent: BinaryNodeInfo,
    child: BinaryNodeInfo,
  ): void => {
    const base = wideNodeIndex * CWBVH_CHILDREN * CWBVH_CHILD_BOUNDS_U16 + slot * CWBVH_CHILD_BOUNDS_U16;
    childBounds[base + 0] = quantizeBound(child.min[0], parent.min[0], parent.max[0], 'floor');
    childBounds[base + 1] = quantizeBound(child.min[1], parent.min[1], parent.max[1], 'floor');
    childBounds[base + 2] = quantizeBound(child.min[2], parent.min[2], parent.max[2], 'floor');
    childBounds[base + 3] = quantizeBound(child.max[0], parent.min[0], parent.max[0], 'ceil');
    childBounds[base + 4] = quantizeBound(child.max[1], parent.min[1], parent.max[1], 'ceil');
    childBounds[base + 5] = quantizeBound(child.max[2], parent.min[2], parent.max[2], 'ceil');
  };

  const fillChildMeta = (
    wideNodeIndex: number,
    slot: number,
    kind: CwbvhChildKind,
    indexOrOffset: number,
    triCount: number,
  ): void => {
    const base = wideNodeIndex * CWBVH_CHILDREN * CWBVH_CHILD_META_WORDS + slot * CWBVH_CHILD_META_WORDS;
    childMeta[base + 0] = kind;
    childMeta[base + 1] = indexOrOffset >>> 0;
    childMeta[base + 2] = triCount >>> 0;
  };

  const buildWide = (binaryNodeIndex: number): number => {
    const parent = readBinaryNode(binary.bvhNodes, binaryNodeIndex);
    const wideNodeIndex = allocNode(parent);
    const candidates = collectWideChildren(binary.bvhNodes, binaryNodeIndex);
    childCount[wideNodeIndex] = candidates.length;

    for (let slot = 0; slot < candidates.length; slot += 1) {
      const childBinaryIndex = candidates[slot]!;
      const child = readBinaryNode(binary.bvhNodes, childBinaryIndex);
      fillChildBounds(wideNodeIndex, slot, parent, child);
      if (isLeafSplit(child.splitAxisOrTriCount)) {
        fillChildMeta(
          wideNodeIndex,
          slot,
          CWBVH_CHILD_LEAF,
          child.rightChildOrTriOffset,
          child.splitAxisOrTriCount & 0xffff,
        );
      } else {
        const childWideIndex = buildWide(childBinaryIndex);
        fillChildMeta(wideNodeIndex, slot, CWBVH_CHILD_NODE, childWideIndex, 0);
      }
    }
    return wideNodeIndex;
  };

  buildWide(0);

  return {
    ...binary,
    cwbvhNodeBounds: new Float32Array(nodeBounds),
    cwbvhChildBounds: new Uint16Array(childBounds),
    cwbvhChildMeta: new Uint32Array(childMeta),
    cwbvhChildCount: new Uint32Array(childCount),
    cwbvhNodeCount: childCount.length,
  };
}

export function buildCompressedWideBvh(
  positions: Float32Array,
  indices: Uint32Array,
  triMaterialIds: Uint32Array,
  opts: BuildArrayBvhOpts = {},
): CompressedWideBvhBuildResult {
  return buildCompressedWideBvhFromArrayBvh(
    buildArrayBvh(positions, indices, triMaterialIds, opts),
  );
}

/**
 * Pack the CPU oracle's six-u16 child bounds into the three-u32 layout consumed
 * by WGSL storage buffers. Each output word is `lo16 | (hi16 << 16)`.
 */
export function packCwbvhChildBoundsForWgsl(childBounds: Uint16Array): Uint32Array {
  if ((childBounds.length & 1) !== 0) {
    throw new Error(`CWBVH child bounds length must be even; got ${childBounds.length}`);
  }
  const packed = new Uint32Array(childBounds.length / 2);
  for (let i = 0; i < packed.length; i += 1) {
    const lo = childBounds[i * 2] ?? 0;
    const hi = childBounds[i * 2 + 1] ?? 0;
    packed[i] = (lo | (hi << 16)) >>> 0;
  }
  return packed;
}

export function packCwbvhBuildBoundsForWgsl(
  cwbvh: Pick<CompressedWideBvhBuildResult, 'cwbvhChildBounds' | 'cwbvhNodeCount'>,
): Uint32Array {
  const expected = cwbvh.cwbvhNodeCount * CWBVH_CHILDREN * CWBVH_CHILD_BOUNDS_PACKED_U32;
  const packed = packCwbvhChildBoundsForWgsl(cwbvh.cwbvhChildBounds);
  if (packed.length !== expected) {
    throw new Error(`CWBVH packed child bounds length mismatch: got ${packed.length}, expected ${expected}`);
  }
  return packed;
}

export function cwbvhChildBounds(
  cwbvh: Pick<CompressedWideBvhBuildResult, 'cwbvhNodeBounds' | 'cwbvhChildBounds'>,
  nodeIndex: number,
  slot: number,
): { min: readonly [number, number, number]; max: readonly [number, number, number] } {
  const nb = nodeIndex * 6;
  const parentMin: [number, number, number] = [
    cwbvh.cwbvhNodeBounds[nb + 0] ?? 0,
    cwbvh.cwbvhNodeBounds[nb + 1] ?? 0,
    cwbvh.cwbvhNodeBounds[nb + 2] ?? 0,
  ];
  const parentMax: [number, number, number] = [
    cwbvh.cwbvhNodeBounds[nb + 3] ?? 0,
    cwbvh.cwbvhNodeBounds[nb + 4] ?? 0,
    cwbvh.cwbvhNodeBounds[nb + 5] ?? 0,
  ];
  const cb = nodeIndex * CWBVH_CHILDREN * CWBVH_CHILD_BOUNDS_U16 + slot * CWBVH_CHILD_BOUNDS_U16;
  return {
    min: [
      dequantizeBound(cwbvh.cwbvhChildBounds[cb + 0] ?? 0, parentMin[0], parentMax[0]),
      dequantizeBound(cwbvh.cwbvhChildBounds[cb + 1] ?? 0, parentMin[1], parentMax[1]),
      dequantizeBound(cwbvh.cwbvhChildBounds[cb + 2] ?? 0, parentMin[2], parentMax[2]),
    ],
    max: [
      dequantizeBound(cwbvh.cwbvhChildBounds[cb + 3] ?? 0, parentMin[0], parentMax[0]),
      dequantizeBound(cwbvh.cwbvhChildBounds[cb + 4] ?? 0, parentMin[1], parentMax[1]),
      dequantizeBound(cwbvh.cwbvhChildBounds[cb + 5] ?? 0, parentMin[2], parentMax[2]),
    ],
  };
}

function v3Sub(a: readonly [number, number, number], b: readonly [number, number, number]): [number, number, number] {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function v3Cross(a: readonly [number, number, number], b: readonly [number, number, number]): [number, number, number] {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function v3Dot(a: readonly [number, number, number], b: readonly [number, number, number]): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function positionAt(
  positions: Float32Array,
  index: number,
  stride: number,
): [number, number, number] {
  const o = index * stride;
  return [positions[o] ?? 0, positions[o + 1] ?? 0, positions[o + 2] ?? 0];
}

function intersectTriangle(
  ray: CwbvhRay,
  a: readonly [number, number, number],
  b: readonly [number, number, number],
  c: readonly [number, number, number],
  triEps: number,
): { t: number; bary: readonly [number, number, number] } | null {
  const e1 = v3Sub(b, a);
  const e2 = v3Sub(c, a);
  const n = v3Cross(e1, e2);
  const det = -v3Dot(ray.direction, n);
  if (Math.abs(det) < triEps) return null;
  const invDet = 1 / det;
  const ao = v3Sub(ray.origin, a);
  const dao = v3Cross(ao, ray.direction);
  const u = v3Dot(e2, dao) * invDet;
  const v = -v3Dot(e1, dao) * invDet;
  const t = v3Dot(ao, n) * invDet;
  const w = 1 - u - v;
  if (u < -triEps || v < -triEps || w < -triEps || t < triEps) return null;
  return { t, bary: [w, u, v] };
}

function intersectAabb(
  ray: CwbvhRay,
  min: readonly [number, number, number],
  max: readonly [number, number, number],
  tMinIn: number,
  tMaxIn: number,
): number | null {
  let tMin = tMinIn;
  let tMax = tMaxIn;
  for (let axis = 0; axis < 3; axis += 1) {
    const o = ray.origin[axis] ?? 0;
    const d = ray.direction[axis] ?? 0;
    if (Math.abs(d) < 1e-30) {
      if (o < (min[axis] ?? 0) || o > (max[axis] ?? 0)) return null;
      continue;
    }
    const invD = 1 / d;
    let t0 = ((min[axis] ?? 0) - o) * invD;
    let t1 = ((max[axis] ?? 0) - o) * invD;
    if (t0 > t1) {
      const tmp = t0;
      t0 = t1;
      t1 = tmp;
    }
    tMin = Math.max(tMin, t0);
    tMax = Math.min(tMax, t1);
    if (tMax < tMin) return null;
  }
  return tMin;
}

export function intersectCompressedWideBvhFirstHit(
  cwbvh: CompressedWideBvhBuildResult,
  positions: Float32Array,
  ray: CwbvhRay,
  opts: CwbvhTraverseOptions = {},
): CwbvhIntersection {
  const positionStride = opts.positionStride ?? 4;
  const indexStride = opts.indexStride ?? 4;
  const triEps = opts.triEps ?? 1e-5;
  const tMin = opts.tMin ?? triEps;
  let closest = opts.tMax ?? Number.POSITIVE_INFINITY;
  let hitTriangle = -1;
  let hitSourceTriangle = -1;
  let hitBary: readonly [number, number, number] = [0, 0, 0];

  if (cwbvh.cwbvhNodeCount === 0) {
    return { didHit: false, dist: closest, triangleIndex: -1, sourceTriangleIndex: -1, bary: hitBary };
  }

  const rootMin: [number, number, number] = [
    cwbvh.cwbvhNodeBounds[0] ?? 0,
    cwbvh.cwbvhNodeBounds[1] ?? 0,
    cwbvh.cwbvhNodeBounds[2] ?? 0,
  ];
  const rootMax: [number, number, number] = [
    cwbvh.cwbvhNodeBounds[3] ?? 0,
    cwbvh.cwbvhNodeBounds[4] ?? 0,
    cwbvh.cwbvhNodeBounds[5] ?? 0,
  ];
  if (intersectAabb(ray, rootMin, rootMax, tMin, closest) == null) {
    return { didHit: false, dist: closest, triangleIndex: -1, sourceTriangleIndex: -1, bary: hitBary };
  }

  const stack: number[] = [0];
  while (stack.length > 0) {
    const nodeIndex = stack.pop()!;
    const count = Math.min(CWBVH_CHILDREN, cwbvh.cwbvhChildCount[nodeIndex] ?? 0);
    for (let slot = 0; slot < count; slot += 1) {
      const mb = nodeIndex * CWBVH_CHILDREN * CWBVH_CHILD_META_WORDS + slot * CWBVH_CHILD_META_WORDS;
      const kind = cwbvh.cwbvhChildMeta[mb] ?? CWBVH_CHILD_EMPTY;
      if (kind === CWBVH_CHILD_EMPTY) continue;
      const bounds = cwbvhChildBounds(cwbvh, nodeIndex, slot);
      if (intersectAabb(ray, bounds.min, bounds.max, tMin, closest) == null) continue;

      if (kind === CWBVH_CHILD_NODE) {
        stack.push(cwbvh.cwbvhChildMeta[mb + 1] ?? 0);
        continue;
      }

      if (kind === CWBVH_CHILD_LEAF) {
        const triOffset = cwbvh.cwbvhChildMeta[mb + 1] ?? 0;
        const triCount = cwbvh.cwbvhChildMeta[mb + 2] ?? 0;
        for (let i = 0; i < triCount; i += 1) {
          const tri = triOffset + i;
          const ib = tri * indexStride;
          const i0 = cwbvh.reorderedIndices[ib] ?? 0;
          const i1 = cwbvh.reorderedIndices[ib + 1] ?? 0;
          const i2 = cwbvh.reorderedIndices[ib + 2] ?? 0;
          const a = positionAt(positions, i0, positionStride);
          const b = positionAt(positions, i1, positionStride);
          const c = positionAt(positions, i2, positionStride);
          const hit = intersectTriangle(ray, a, b, c, triEps);
          if (hit != null && hit.t >= tMin && hit.t < closest) {
            closest = hit.t;
            hitTriangle = tri;
            hitSourceTriangle = cwbvh.reorderedToSourceTriangle[tri] ?? tri;
            hitBary = hit.bary;
          }
        }
      }
    }
  }

  return {
    didHit: hitTriangle >= 0,
    dist: closest,
    triangleIndex: hitTriangle,
    sourceTriangleIndex: hitSourceTriangle,
    bary: hitBary,
  };
}
