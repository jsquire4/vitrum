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
import { packedMaterialHasTransmission } from './wgsl/materialTransmission.wgsl.js';

export const CWBVH_CHILD_EMPTY = 0 as const;
export const CWBVH_CHILD_NODE = 1 as const;
export const CWBVH_CHILD_LEAF = 2 as const;
/** Host-side invalid-layout sentinel; valid wide nodes always have 0..8 children. */
export const CWBVH_CHILD_COUNT_INVALID = 0xffffffff as const;

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
  /**
   * Wide-node root to traverse. Defaults to 0 for single-BLAS CWBVH trees.
   * Renderer-shaped forests concatenate one CWBVH tree per BLAS and use this to
   * mirror TLAS BLAS-root remapping.
   */
  readonly root?: number;
  readonly triEps?: number;
  readonly tMin?: number;
  readonly tMax?: number;
  /**
   * Mirror the WGSL glass-skip filter: when true, stride-4 triangle payloads
   * whose physical-transmission nibble is nonzero do not occlude.
   */
  /**
   * CPU-oracle stack bound. Defaults to unbounded; tests can lower it to force
   * the same explicit fallback condition as the fixed-depth WGSL traversal.
   */
  readonly maxStackDepth?: number;
  readonly skipGlass?: boolean;
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

function stableUnitOffset(value: number, min: number, max: number): number | null {
  if (![value, min, max].every(Number.isFinite) || min > max) {
    return null;
  }
  if (max === min) return 0;
  const extent = max - min;
  if (Number.isFinite(extent)) return (value - min) / extent;
  const halfExtent = max * 0.5 - min * 0.5;
  if (!(halfExtent > 0) || !Number.isFinite(halfExtent)) return null;
  return (value * 0.5 - min * 0.5) / halfExtent;
}

function quantizeBound(value: number, min: number, max: number, mode: 'floor' | 'ceil'): number {
  const normalized = stableUnitOffset(value, min, max);
  if (normalized == null) return 0;
  if (max === min) return mode === 'floor' ? 0 : 65535;
  const q = normalized * 65535;
  const rounded = mode === 'floor' ? Math.floor(q) : Math.ceil(q);
  return clampU16(mode === 'floor' ? rounded - 1 : rounded + 1);
}

function dequantizeBound(q: number, min: number, max: number): number {
  if (![min, max].every(Number.isFinite) || min > max) return Number.NaN;
  if (max === min) return min;
  if (q <= 0) return min;
  if (q >= 65535) return max;
  const t = q / 65535;
  return min * (1 - t) + max * t;
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
 * representation shared by CPU validation and renderer traversal.
 */
export function buildCompressedWideBvhFromArrayBvh(
  binary: CpuBvhBuildResult,
): CompressedWideBvhBuildResult {
  const binaryNodeCount = Math.floor(binary.bvhNodes.length / 8);
  const emptyWideRoot = (invalid = false): CompressedWideBvhBuildResult => ({
    ...binary,
    cwbvhNodeBounds: new Float32Array(6),
    cwbvhChildBounds: new Uint16Array(CWBVH_CHILDREN * CWBVH_CHILD_BOUNDS_U16),
    cwbvhChildMeta: new Uint32Array(CWBVH_CHILDREN * CWBVH_CHILD_META_WORDS),
    cwbvhChildCount: new Uint32Array([invalid ? CWBVH_CHILD_COUNT_INVALID : 0]),
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

  if (binary.bvhNodes.length % 8 !== 0) return emptyWideRoot(true);
  for (let nodeIndex = 0; nodeIndex < binaryNodeCount; nodeIndex += 1) {
    const node = readBinaryNode(binary.bvhNodes, nodeIndex);
    for (let axis = 0; axis < 3; axis += 1) {
      const min = node.min[axis]!;
      const max = node.max[axis]!;
      if (stableUnitOffset(min, min, max) == null) {
        return emptyWideRoot(true);
      }
    }
  }

  const root = readBinaryNode(binary.bvhNodes, 0);
  if (binaryNodeCount === 1 && isLeafSplit(root.splitAxisOrTriCount) && (root.splitAxisOrTriCount & 0xffff) === 0) {
    return emptyWideRoot();
  }

  const nodeBounds: number[] = [];
  const childBounds: number[] = [];
  const childMeta: number[] = [];
  const childCount: number[] = [];
  let traversalValid = true;

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
    for (let axis = 0; axis < 3; axis += 1) {
      const parentMin = parent.min[axis]!;
      const parentMax = parent.max[axis]!;
      const childMin = child.min[axis]!;
      const childMax = child.max[axis]!;
      if (
        childMin < parentMin || childMax > parentMax || childMin > childMax ||
        stableUnitOffset(childMin, parentMin, parentMax) == null ||
        stableUnitOffset(childMax, parentMin, parentMax) == null
      ) {
        traversalValid = false;
      }
    }
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
  // A public caller may traverse a nonzero root (for example a renderer-shaped
  // concatenated forest). Mark every root candidate invalid when conversion
  // discovered a global parent/child layout violation; a node-0-only sentinel
  // let an explicitly selected nonzero root bypass the corruption gate.
  if (!traversalValid) childCount.fill(CWBVH_CHILD_COUNT_INVALID);

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

/**
 * Overlay per-source-triangle payload words onto a BVH-reordered stride-4 index
 * buffer. `buildArrayBvh` deliberately zero-fills padding lanes; consumers that
 * need a material/visibility payload in `.w` can call this with the source
 * triangle payload stream after the BVH reorder is known.
 */
export function reorderCwbvhTrianglePayloads(
  cwbvh: Pick<CompressedWideBvhBuildResult, 'reorderedIndices' | 'reorderedToSourceTriangle'>,
  sourcePayloads: Uint32Array,
  indexStride = 4,
): Uint32Array {
  if (indexStride < 4) {
    throw new Error(`CWBVH triangle payloads require indexStride >= 4; got ${indexStride}`);
  }
  const withPayloads = new Uint32Array(cwbvh.reorderedIndices);
  for (let tri = 0; tri < cwbvh.reorderedToSourceTriangle.length; tri += 1) {
    const sourceTri = cwbvh.reorderedToSourceTriangle[tri] ?? tri;
    withPayloads[tri * indexStride + 3] = sourcePayloads[sourceTri] ?? 0;
  }
  return withPayloads;
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
  if (!Number.isInteger(index) || index < 0 || o + 2 >= positions.length) {
    throw new Error(`CWBVH position index ${index} is out of range`);
  }
  const value: [number, number, number] = [positions[o]!, positions[o + 1]!, positions[o + 2]!];
  if (!value.every(Number.isFinite)) {
    throw new Error(`CWBVH position ${index} is non-finite`);
  }
  return value;
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

function shouldSkipGlassTriangle(
  reorderedIndices: Uint32Array,
  tri: number,
  indexStride: number,
  skipGlass: boolean,
): boolean {
  if (!skipGlass || indexStride < 4) return false;
  const payload = reorderedIndices[tri * indexStride + 3] ?? 0;
  return packedMaterialHasTransmission(payload);
}

function resolveCwbvhRoot(cwbvh: CompressedWideBvhBuildResult, root: number | undefined): number {
  const resolved = root ?? 0;
  if (!Number.isInteger(resolved) || resolved < 0 || resolved >= cwbvh.cwbvhNodeCount) {
    throw new RangeError(`CWBVH root ${resolved} is outside 0..${Math.max(0, cwbvh.cwbvhNodeCount - 1)}`);
  }
  return resolved;
}

function validatedCwbvhChildCount(cwbvh: CompressedWideBvhBuildResult, nodeIndex: number): number {
  if (!Number.isInteger(nodeIndex) || nodeIndex < 0 || nodeIndex >= cwbvh.cwbvhNodeCount) {
    throw new Error(`CWBVH node reference ${nodeIndex} is out of range`);
  }
  const count = cwbvh.cwbvhChildCount[nodeIndex] ?? CWBVH_CHILD_COUNT_INVALID;
  if (count > CWBVH_CHILDREN) {
    throw new Error(`CWBVH node ${nodeIndex} has invalid child count ${count}`);
  }
  return count;
}

function validateCwbvhTraversalInputs(
  cwbvh: CompressedWideBvhBuildResult,
  positions: Float32Array,
  ray: CwbvhRay,
  positionStride: number,
  indexStride: number,
  triEps: number,
  tMin: number,
  tMax: number,
): void {
  if (!Number.isInteger(cwbvh.cwbvhNodeCount) || cwbvh.cwbvhNodeCount < 0) {
    throw new Error(`CWBVH node count ${cwbvh.cwbvhNodeCount} is invalid`);
  }
  const nodeCount = cwbvh.cwbvhNodeCount;
  if (
    cwbvh.cwbvhNodeBounds.length < nodeCount * 6 ||
    cwbvh.cwbvhChildBounds.length < nodeCount * CWBVH_CHILDREN * CWBVH_CHILD_BOUNDS_U16 ||
    cwbvh.cwbvhChildMeta.length < nodeCount * CWBVH_CHILDREN * CWBVH_CHILD_META_WORDS ||
    cwbvh.cwbvhChildCount.length < nodeCount
  ) {
    throw new Error('CWBVH traversal buffers are shorter than cwbvhNodeCount');
  }
  if (!Number.isInteger(positionStride) || positionStride < 3 || positions.length % positionStride !== 0) {
    throw new Error(`CWBVH position stride ${positionStride} is invalid for ${positions.length} words`);
  }
  if (
    !Number.isInteger(indexStride) ||
    indexStride < 3 ||
    cwbvh.reorderedIndices.length % indexStride !== 0
  ) {
    throw new Error(`CWBVH index stride ${indexStride} is invalid for ${cwbvh.reorderedIndices.length} words`);
  }
  if (!Number.isFinite(triEps) || triEps <= 0) {
    throw new Error(`CWBVH triangle epsilon ${triEps} is invalid`);
  }
  if (!Number.isFinite(tMin) || tMin < 0 || Number.isNaN(tMax) || tMax < tMin) {
    throw new Error(`CWBVH traversal interval [${tMin}, ${tMax}) is invalid`);
  }
  if (!ray.origin.every(Number.isFinite) || !ray.direction.every(Number.isFinite)) {
    throw new Error('CWBVH ray contains non-finite components');
  }
}

function validateCwbvhBounds(
  bounds: { min: readonly [number, number, number]; max: readonly [number, number, number] },
  label: string,
): void {
  if (
    !bounds.min.every(Number.isFinite) ||
    !bounds.max.every(Number.isFinite) ||
    bounds.min[0] > bounds.max[0] ||
    bounds.min[1] > bounds.max[1] ||
    bounds.min[2] > bounds.max[2]
  ) {
    throw new Error(`${label} has invalid bounds`);
  }
}

function cwbvhNodeBounds(
  cwbvh: Pick<CompressedWideBvhBuildResult, 'cwbvhNodeBounds'>,
  nodeIndex: number,
): { min: readonly [number, number, number]; max: readonly [number, number, number] } {
  const nb = nodeIndex * 6;
  return {
    min: [
      cwbvh.cwbvhNodeBounds[nb + 0] ?? 0,
      cwbvh.cwbvhNodeBounds[nb + 1] ?? 0,
      cwbvh.cwbvhNodeBounds[nb + 2] ?? 0,
    ],
    max: [
      cwbvh.cwbvhNodeBounds[nb + 3] ?? 0,
      cwbvh.cwbvhNodeBounds[nb + 4] ?? 0,
      cwbvh.cwbvhNodeBounds[nb + 5] ?? 0,
    ],
  };
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
  const skipGlass = opts.skipGlass ?? false;
  const maxStackDepth = opts.maxStackDepth ?? Number.POSITIVE_INFINITY;
  validateCwbvhTraversalInputs(cwbvh, positions, ray, positionStride, indexStride, triEps, tMin, closest);
  if (!Number.isInteger(maxStackDepth) && maxStackDepth !== Number.POSITIVE_INFINITY || maxStackDepth < 1) {
    throw new RangeError('CWBVH maxStackDepth must be a positive integer or Infinity');
  }
  let hitTriangle = -1;
  let hitSourceTriangle = -1;
  let hitBary: readonly [number, number, number] = [0, 0, 0];

  if (cwbvh.cwbvhNodeCount === 0) {
    return { didHit: false, dist: closest, triangleIndex: -1, sourceTriangleIndex: -1, bary: hitBary };
  }

  const root = resolveCwbvhRoot(cwbvh, opts.root);
  validatedCwbvhChildCount(cwbvh, root);
  const rootBounds = cwbvhNodeBounds(cwbvh, root);
  validateCwbvhBounds(rootBounds, `CWBVH node ${root}`);
  if (intersectAabb(ray, rootBounds.min, rootBounds.max, tMin, closest) == null) {
    return { didHit: false, dist: closest, triangleIndex: -1, sourceTriangleIndex: -1, bary: hitBary };
  }

  const stack: number[] = [root];
  while (stack.length > 0) {
    const nodeIndex = stack.pop()!;
    const count = validatedCwbvhChildCount(cwbvh, nodeIndex);
    for (let slot = 0; slot < count; slot += 1) {
      const mb = nodeIndex * CWBVH_CHILDREN * CWBVH_CHILD_META_WORDS + slot * CWBVH_CHILD_META_WORDS;
      const kind = cwbvh.cwbvhChildMeta[mb] ?? CWBVH_CHILD_EMPTY;
      if (kind === CWBVH_CHILD_EMPTY) {
        throw new Error(`CWBVH node ${nodeIndex} has an empty live child at slot ${slot}`);
      }
      const bounds = cwbvhChildBounds(cwbvh, nodeIndex, slot);
      validateCwbvhBounds(bounds, `CWBVH node ${nodeIndex} child ${slot}`);
      if (intersectAabb(ray, bounds.min, bounds.max, tMin, closest) == null) continue;

      if (kind === CWBVH_CHILD_NODE) {
        const childNode = cwbvh.cwbvhChildMeta[mb + 1]!;
        if (childNode >= cwbvh.cwbvhNodeCount) {
          throw new Error(`CWBVH child node reference ${childNode} is out of range`);
        }
        if (stack.length >= maxStackDepth) {
          throw new Error(`CWBVH traversal stack overflow at capacity ${maxStackDepth}`);
        }
        stack.push(childNode);
        continue;
      }

      if (kind === CWBVH_CHILD_LEAF) {
        const triOffset = cwbvh.cwbvhChildMeta[mb + 1] ?? 0;
        const triCount = cwbvh.cwbvhChildMeta[mb + 2] ?? 0;
        const triangleCount = cwbvh.reorderedIndices.length / indexStride;
        if (triCount === 0 || triOffset > triangleCount || triCount > triangleCount - triOffset) {
          throw new Error(`CWBVH leaf range ${triOffset}+${triCount} is out of range`);
        }
        for (let i = 0; i < triCount; i += 1) {
          const tri = triOffset + i;
          if (shouldSkipGlassTriangle(cwbvh.reorderedIndices, tri, indexStride, skipGlass)) continue;
          const ib = tri * indexStride;
          const i0 = cwbvh.reorderedIndices[ib] ?? 0;
          const i1 = cwbvh.reorderedIndices[ib + 1] ?? 0;
          const i2 = cwbvh.reorderedIndices[ib + 2] ?? 0;
          const a = positionAt(positions, i0, positionStride);
          const b = positionAt(positions, i1, positionStride);
          const c = positionAt(positions, i2, positionStride);
          const hit = intersectTriangle(ray, a, b, c, triEps);
          if (hit != null && hit.t > tMin && hit.t < closest) {
            closest = hit.t;
            hitTriangle = tri;
            hitSourceTriangle = cwbvh.reorderedToSourceTriangle[tri] ?? tri;
            hitBary = hit.bary;
          }
        }
      } else {
        throw new Error(`CWBVH child kind ${kind} is invalid`);
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

export function intersectCompressedWideBvhAnyHit(
  cwbvh: CompressedWideBvhBuildResult,
  positions: Float32Array,
  ray: CwbvhRay,
  opts: CwbvhTraverseOptions = {},
): boolean {
  const positionStride = opts.positionStride ?? 4;
  const indexStride = opts.indexStride ?? 4;
  const triEps = opts.triEps ?? 1e-5;
  // Reconciled with intersectCompressedWideBvhFirstHit: same default tMin
  // (triEps) and the same strict open-interval `>` acceptance below. anyHit must not
  // report a *miss* where firstHit reports a *hit* at the same epsilon boundary,
  // or shadow/occlusion rays would under-occlude relative to the closest-hit
  // geometry.
  const tMin = opts.tMin ?? triEps;
  const tMax = opts.tMax ?? Number.POSITIVE_INFINITY;
  const skipGlass = opts.skipGlass ?? false;

  const maxStackDepth = opts.maxStackDepth ?? Number.POSITIVE_INFINITY;
  validateCwbvhTraversalInputs(cwbvh, positions, ray, positionStride, indexStride, triEps, tMin, tMax);
  if (!Number.isInteger(maxStackDepth) && maxStackDepth !== Number.POSITIVE_INFINITY || maxStackDepth < 1) {
    throw new RangeError('CWBVH maxStackDepth must be a positive integer or Infinity');
  }
  if (cwbvh.cwbvhNodeCount === 0) return false;

  const root = resolveCwbvhRoot(cwbvh, opts.root);
  validatedCwbvhChildCount(cwbvh, root);
  const rootBounds = cwbvhNodeBounds(cwbvh, root);
  validateCwbvhBounds(rootBounds, `CWBVH node ${root}`);
  if (intersectAabb(ray, rootBounds.min, rootBounds.max, tMin, tMax) == null) return false;

  const stack: number[] = [root];
  while (stack.length > 0) {
    const nodeIndex = stack.pop()!;
    const count = validatedCwbvhChildCount(cwbvh, nodeIndex);
    for (let slot = 0; slot < count; slot += 1) {
      const mb = nodeIndex * CWBVH_CHILDREN * CWBVH_CHILD_META_WORDS + slot * CWBVH_CHILD_META_WORDS;
      const kind = cwbvh.cwbvhChildMeta[mb] ?? CWBVH_CHILD_EMPTY;
      if (kind === CWBVH_CHILD_EMPTY) {
        throw new Error(`CWBVH node ${nodeIndex} has an empty live child at slot ${slot}`);
      }
      const bounds = cwbvhChildBounds(cwbvh, nodeIndex, slot);
      validateCwbvhBounds(bounds, `CWBVH node ${nodeIndex} child ${slot}`);
      if (intersectAabb(ray, bounds.min, bounds.max, tMin, tMax) == null) continue;

      if (kind === CWBVH_CHILD_NODE) {
        const childNode = cwbvh.cwbvhChildMeta[mb + 1]!;
        if (childNode >= cwbvh.cwbvhNodeCount) {
          throw new Error(`CWBVH child node reference ${childNode} is out of range`);
        }
        if (stack.length >= maxStackDepth) {
          throw new Error(`CWBVH traversal stack overflow at capacity ${maxStackDepth}`);
        }
        stack.push(childNode);
        continue;
      }
      if (kind === CWBVH_CHILD_LEAF) {
        const triOffset = cwbvh.cwbvhChildMeta[mb + 1] ?? 0;
        const triCount = cwbvh.cwbvhChildMeta[mb + 2] ?? 0;
        const triangleCount = cwbvh.reorderedIndices.length / indexStride;
        if (triCount === 0 || triOffset > triangleCount || triCount > triangleCount - triOffset) {
          throw new Error(`CWBVH leaf range ${triOffset}+${triCount} is out of range`);
        }
        for (let i = 0; i < triCount; i += 1) {
          const tri = triOffset + i;
          if (shouldSkipGlassTriangle(cwbvh.reorderedIndices, tri, indexStride, skipGlass)) continue;
          const ib = tri * indexStride;
          const i0 = cwbvh.reorderedIndices[ib] ?? 0;
          const i1 = cwbvh.reorderedIndices[ib + 1] ?? 0;
          const i2 = cwbvh.reorderedIndices[ib + 2] ?? 0;
          const a = positionAt(positions, i0, positionStride);
          const b = positionAt(positions, i1, positionStride);
          const c = positionAt(positions, i2, positionStride);
          const hit = intersectTriangle(ray, a, b, c, triEps);
          if (hit != null && hit.t > tMin && hit.t < tMax) return true;
        }
      } else {
        throw new Error(`CWBVH child kind ${kind} is invalid`);
      }
    }
  }

  return false;
}
