/**
 * tlas.ts — Top-level acceleration structure (TLAS) builder.
 *
 * A TLAS is a BVH over **instances** rather than triangles. Each leaf
 * references one instance carrying:
 *   - a BLAS root index (which mesh's BLAS to enter on traversal),
 *   - a 4x4 world-to-local transform (for ray transformation),
 *   - the instance's world AABB (for build + refit).
 *
 * Pipeline shape (when wired up downstream): ray enters TLAS → BVH
 * traversal finds candidate instance leaves → for each candidate, ray
 * is transformed into instance-local space → BLAS traversal walks that
 * mesh's local BVH → closest-hit returns to TLAS frame for shading.
 *
 * Build algorithm: binned SAH, K=16 bins per axis, mirroring
 * `buildArrayBvh.ts` for consistency with the BLAS builder. The
 * difference is the **primitive** type: instance AABBs here, triangle
 * AABBs in the BLAS builder. Leaves carry instance indices into a
 * reordered `instanceIndices` array (analogous to `reorderedIndices`
 * for triangles).
 *
 * Node layout (32 bytes, 8 × u32) — identical to the BLAS layout so the
 * WGSL traversal can share intersection primitives (`bvhIntersect.wgsl`).
 * The semantic difference lives entirely in the *leaf payload*: instead
 * of a triangle offset/count, a TLAS leaf yields an instance offset +
 * count, and traversal looks up `(blasRoot, worldToLocal)` from the
 * instance table.
 *
 *   f32[0..2]  boundsMin xyz       — world-space AABB of the subtree
 *   f32[3..5]  boundsMax xyz
 *   u32[6]     rightChildOrInstanceOffset
 *               - interior: RELATIVE offset to right child node
 *                 (rightChildIndex − thisNodeIndex). Left child = +1.
 *               - leaf:     absolute offset into `instanceIndices` for
 *                           the first instance of the leaf.
 *   u32[7]     splitAxisOrInstanceCount
 *               - interior: split axis (0=X, 1=Y, 2=Z)
 *               - leaf:     0xFFFF0000 | instanceCount
 *
 * Refit (no topology change): call `refitTlas(data, instances)` after
 * any instance transform changes; this walks the tree bottom-up,
 * recomputing each interior node's AABB as the union of its children's.
 * Cost: O(nodes), no rebalancing — appropriate when topology (which
 * instance is where, how many instances) is unchanged.
 */

import {
  BINARY_BVH_MAX_BUILD_DEPTH,
  BVH_NODE_FLOATS,
  TLAS_TRAVERSAL_STACK_DEPTH,
} from './strides.js';

const TLAS_LEAFNODE_FLAG = 0xffff0000;
const TLAS_DEFAULT_MAX_LEAF_INSTANCES = 1;       // typically 1 instance/leaf
const TLAS_DEFAULT_NUM_BINS = 16;
/** Packed TLAS leaves store their instance count in 16 bits. */
export const TLAS_MAX_LEAF_INSTANCES = 0xffff;
/**
 * Upper bound for binned-SAH scratch allocation. More bins have sharply
 * diminishing value for an instance TLAS while allocating six O(K) arrays
 * per axis; 256 is deliberately generous and keeps malformed options bounded.
 */
export const TLAS_MAX_NUM_BINS = 256;
/** Deepest interior level accepted by every live binary TLAS traversal. */
export const TLAS_MAX_BUILD_DEPTH = BINARY_BVH_MAX_BUILD_DEPTH;
/** Alias for {@link BVH_NODE_FLOATS}: TLAS shares the same 32-byte node layout as BLAS. */
const TLAS_NODE_STRIDE_U32 = BVH_NODE_FLOATS;

export interface TlasInstance {
  /** Index into the caller's BLAS table; passed through verbatim. */
  readonly blasId: number;
  /** World AABB of this instance, used for build + refit. */
  readonly aabbMin: readonly [number, number, number];
  readonly aabbMax: readonly [number, number, number];
  /**
   * 4x4 world-to-local transform, column-major (16 floats). Used by the
   * traversal to bring the ray into the BLAS-local frame. Carried verbatim
   * in the returned `instanceTransforms` Float32Array (16 floats per
   * instance), in the order that the *original* instance list was passed.
   */
  readonly worldToLocal: Float32Array;
}

export interface TlasBuildOptions {
  /** Maximum instances in any leaf; finite safe integer in [1, 65535]. */
  readonly maxLeafInstances?: number;
  /** SAH bin count; finite safe integer in [2, 256]. */
  readonly numBins?: number;
  /** Maximum node depth; finite safe integer in [0, 58]. May only lower the live-safe cap. */
  readonly maxDepth?: number;
}

export type TlasBalancedFallbackReason =
  | 'degenerate-centroids'
  | 'non-improving-sah'
  | 'degenerate-partition'
  | 'depth-safety';

export interface TlasValidationReport {
  readonly status: 'valid';
  readonly nodeCount: number;
  readonly interiorNodeCount: number;
  readonly leafNodeCount: number;
  readonly maxDepth: number;
  readonly maxLeafInstances: number;
  readonly maxTraversalStackEntries: number;
  readonly traversalStackCapacity: number;
}

export interface TlasBuildStatus {
  readonly status: 'valid';
  readonly balancedFallbackCount: number;
  readonly balancedFallbackReasons: Readonly<Record<TlasBalancedFallbackReason, number>>;
  readonly validation: TlasValidationReport;
}

/** Packed TLAS buffers accepted by validators, refit helpers, and CPU traversal. */
export interface TlasBufferView {
  /** 8 × u32 per node, packed in preorder/depth-first layout; node 0 is the root. */
  readonly nodes: Uint32Array;
  readonly nodeCount: number;
  /**
   * Permutation of instance indices into the original input list. Leaves
   * reference contiguous spans here. `instanceIndices[i]` = original
   * input index of the i-th instance in build order.
   */
  readonly instanceIndices: Uint32Array;
  /**
   * Per-instance bookkeeping kept in **original-input order** (not build
   * order). Indexed by the original instance id; downstream traversal
   * looks up `blasRoots[origIndex]` after dereferencing a leaf.
   */
  readonly blasRoots: Uint32Array;
  readonly instanceTransforms: Float32Array;
}

/** Internally built TLAS payload — designed for direct DMA to WebGPU storage buffers. */
export interface TlasData extends TlasBufferView {
  /**
   * Build-time proof that the packed tree fits the live WGSL traversal stack.
   * Serialized/foreign buffers remain valid {@link TlasBufferView} values and
   * can obtain the same report with {@link validateTlasBuild}.
   */
  readonly buildStatus: TlasBuildStatus;
}

interface InstanceRecord {
  readonly origIndex: number;
  readonly centroid: readonly [number, number, number];
  readonly min: readonly [number, number, number];
  readonly max: readonly [number, number, number];
}

interface TlasNodeBuild {
  min: [number, number, number];
  max: [number, number, number];
  rightChildOrInstanceOffset: number;
  splitAxisOrInstanceCount: number;
}

interface AabbBin {
  min: [number, number, number];
  max: [number, number, number];
  count: number;
}

interface TlasBuildStats {
  balancedFallbackCount: number;
  readonly balancedFallbackReasons: Record<TlasBalancedFallbackReason, number>;
}

function aabbCentroid(min: readonly [number, number, number], max: readonly [number, number, number])
  : readonly [number, number, number] {
  return [
    (min[0] + max[0]) * 0.5,
    (min[1] + max[1]) * 0.5,
    (min[2] + max[2]) * 0.5,
  ];
}

function isFiniteVec3(v: readonly [number, number, number]): boolean {
  return Number.isFinite(v[0]) && Number.isFinite(v[1]) && Number.isFinite(v[2]);
}

function isFiniteMat4(m: Float32Array): boolean {
  // The caller has already established the intrinsic 16-element length. Loop
  // the contract size directly so an own spoofed `length` property cannot turn
  // this validation into a zero-iteration pass.
  for (let i = 0; i < 16; i += 1) {
    if (!Number.isFinite(m[i])) return false;
  }
  return true;
}

const TYPED_ARRAY_BRAND_DESCRIPTOR = Object.getOwnPropertyDescriptor(
  Object.getPrototypeOf(Float32Array.prototype) as object,
  Symbol.toStringTag,
);
const TYPED_ARRAY_LENGTH_DESCRIPTOR = Object.getOwnPropertyDescriptor(
  Object.getPrototypeOf(Float32Array.prototype) as object,
  'length',
);

function intrinsicFloat32Length(value: unknown): number | undefined {
  if (
    !ArrayBuffer.isView(value) ||
    TYPED_ARRAY_BRAND_DESCRIPTOR?.get === undefined ||
    TYPED_ARRAY_LENGTH_DESCRIPTOR?.get === undefined
  ) {
    return undefined;
  }
  try {
    if (TYPED_ARRAY_BRAND_DESCRIPTOR.get.call(value) !== 'Float32Array') return undefined;
    const length = TYPED_ARRAY_LENGTH_DESCRIPTOR.get.call(value) as unknown;
    return Number.isSafeInteger(length) && (length as number) >= 0 ? length as number : undefined;
  } catch {
    return undefined;
  }
}

function assertDenseArray(value: unknown, label: string): asserts value is unknown[] {
  if (!Array.isArray(value)) throw new TypeError(`buildTlas: ${label} must be an array.`);
  const length = value.length;
  let count = 0;
  for (const key of Reflect.ownKeys(value)) {
    if (key === 'length') continue;
    if (typeof key !== 'string' || !/^(0|[1-9][0-9]*)$/.test(key) || Number(key) >= length) {
      throw new RangeError(`buildTlas: ${label} may only contain indexed elements.`);
    }
    const descriptor = Object.getOwnPropertyDescriptor(value, key)!;
    if (!('value' in descriptor) || descriptor.enumerable !== true) {
      throw new RangeError(`buildTlas: ${label}[${key}] must be an own enumerable data property.`);
    }
    count += 1;
  }
  if (count !== length) throw new RangeError(`buildTlas: ${label} must be dense.`);
}

function assertVec3Tuple(value: unknown, label: string): asserts value is [number, number, number] {
  assertDenseArray(value, label);
  if (value.length !== 3) throw new RangeError(`buildTlas: ${label} must have exactly 3 elements.`);
  for (let lane = 0; lane < 3; lane += 1) {
    const component = value[lane];
    if (typeof component !== 'number' || !Number.isFinite(component) || !Number.isFinite(Math.fround(component))) {
      throw new RangeError(`buildTlas: ${label}[${lane}] must be finite and representable as float32.`);
    }
  }
}

function isInvertedAabb(
  min: readonly [number, number, number],
  max: readonly [number, number, number],
): boolean {
  return max[0] < min[0] || max[1] < min[1] || max[2] < min[2];
}

function aabbSurfaceArea(
  min: readonly [number, number, number],
  max: readonly [number, number, number],
): number {
  const dx = Math.max(0, max[0] - min[0]);
  const dy = Math.max(0, max[1] - min[1]);
  const dz = Math.max(0, max[2] - min[2]);
  return 2 * (dx * dy + dy * dz + dz * dx);
}

function emptyBin(): AabbBin {
  return {
    min: [Infinity, Infinity, Infinity],
    max: [-Infinity, -Infinity, -Infinity],
    count: 0,
  };
}

function unionBinInto(target: AabbBin, src: AabbBin): void {
  if (src.count === 0) return;
  if (src.min[0] < target.min[0]) target.min[0] = src.min[0];
  if (src.min[1] < target.min[1]) target.min[1] = src.min[1];
  if (src.min[2] < target.min[2]) target.min[2] = src.min[2];
  if (src.max[0] > target.max[0]) target.max[0] = src.max[0];
  if (src.max[1] > target.max[1]) target.max[1] = src.max[1];
  if (src.max[2] > target.max[2]) target.max[2] = src.max[2];
  target.count += src.count;
}

function buildLeaf(records: ReadonlyArray<InstanceRecord>): TlasNodeBuild {
  if (records.length > 0xffff) {
    throw new Error(
      `[@vitrum/shared-bvh/tlas] Leaf instance count ${records.length} exceeds the ` +
      `16-bit limit (0xFFFF = 65535). Reduce the number of instances per leaf.`,
    );
  }
  let nMinX = Infinity, nMinY = Infinity, nMinZ = Infinity;
  let nMaxX = -Infinity, nMaxY = -Infinity, nMaxZ = -Infinity;
  for (const r of records) {
    if (r.min[0] < nMinX) nMinX = r.min[0];
    if (r.min[1] < nMinY) nMinY = r.min[1];
    if (r.min[2] < nMinZ) nMinZ = r.min[2];
    if (r.max[0] > nMaxX) nMaxX = r.max[0];
    if (r.max[1] > nMaxY) nMaxY = r.max[1];
    if (r.max[2] > nMaxZ) nMaxZ = r.max[2];
  }
  return {
    min: [nMinX, nMinY, nMinZ],
    max: [nMaxX, nMaxY, nMaxZ],
    rightChildOrInstanceOffset: 0,                     // patched at flatten time
    splitAxisOrInstanceCount: TLAS_LEAFNODE_FLAG | records.length,
  };
}

interface CentroidBounds {
  readonly cMin: readonly [number, number, number];
  readonly cMax: readonly [number, number, number];
}

function computeCentroidBounds(records: ReadonlyArray<InstanceRecord>): CentroidBounds {
  let cMinX = Infinity, cMinY = Infinity, cMinZ = Infinity;
  let cMaxX = -Infinity, cMaxY = -Infinity, cMaxZ = -Infinity;
  for (const r of records) {
    if (r.centroid[0] < cMinX) cMinX = r.centroid[0];
    if (r.centroid[1] < cMinY) cMinY = r.centroid[1];
    if (r.centroid[2] < cMinZ) cMinZ = r.centroid[2];
    if (r.centroid[0] > cMaxX) cMaxX = r.centroid[0];
    if (r.centroid[1] > cMaxY) cMaxY = r.centroid[1];
    if (r.centroid[2] > cMaxZ) cMaxZ = r.centroid[2];
  }
  return { cMin: [cMinX, cMinY, cMinZ], cMax: [cMaxX, cMaxY, cMaxZ] };
}

function pickSplit(records: ReadonlyArray<InstanceRecord>, cb: CentroidBounds, numBins: number)
  : { axis: number; binIdx: number; cost: number } | null
{
  const extents: readonly [number, number, number] = [
    cb.cMax[0] - cb.cMin[0], cb.cMax[1] - cb.cMin[1], cb.cMax[2] - cb.cMin[2],
  ];
  // Degenerate: all centroids coincide → cannot split.
  if (extents[0] <= 0 && extents[1] <= 0 && extents[2] <= 0) return null;

  let best: { axis: number; binIdx: number; cost: number } | null = null;

  for (let axis = 0; axis < 3; axis++) {
    const ext = extents[axis]!;
    if (ext <= 0) continue;
    const cMin = cb.cMin[axis]!;
    const bins: AabbBin[] = [];
    for (let b = 0; b < numBins; b++) bins.push(emptyBin());

    for (const r of records) {
      const cAxis = r.centroid[axis]!;
      const tBin = Math.floor((cAxis - cMin) / ext * numBins);
      const bIdx = Math.max(0, Math.min(numBins - 1, tBin));
      const bin = bins[bIdx]!;
      if (r.min[0] < bin.min[0]) bin.min[0] = r.min[0];
      if (r.min[1] < bin.min[1]) bin.min[1] = r.min[1];
      if (r.min[2] < bin.min[2]) bin.min[2] = r.min[2];
      if (r.max[0] > bin.max[0]) bin.max[0] = r.max[0];
      if (r.max[1] > bin.max[1]) bin.max[1] = r.max[1];
      if (r.max[2] > bin.max[2]) bin.max[2] = r.max[2];
      bin.count += 1;
    }

    // Prefix / suffix sweep for SAH cost.
    const leftBin = emptyBin();
    const leftSA: number[] = new Array<number>(numBins);
    const leftCount: number[] = new Array<number>(numBins);
    for (let b = 0; b < numBins; b++) {
      unionBinInto(leftBin, bins[b]!);
      leftSA[b] = aabbSurfaceArea(leftBin.min, leftBin.max);
      leftCount[b] = leftBin.count;
    }
    const rightBin = emptyBin();
    const rightSA: number[] = new Array<number>(numBins);
    const rightCount: number[] = new Array<number>(numBins);
    for (let b = numBins - 1; b >= 0; b--) {
      unionBinInto(rightBin, bins[b]!);
      rightSA[b] = aabbSurfaceArea(rightBin.min, rightBin.max);
      rightCount[b] = rightBin.count;
    }
    const totalSA = leftSA[numBins - 1]!;
    if (totalSA <= 0) continue;

    for (let b = 0; b < numBins - 1; b++) {
      const lCount = leftCount[b]!;
      const rCount = rightCount[b + 1]!;
      if (lCount === 0 || rCount === 0) continue;
      const sah = (lCount * leftSA[b]! + rCount * rightSA[b + 1]!) / totalSA;
      if (best == null || sah < best.cost) {
        best = { axis, binIdx: b, cost: sah };
      }
    }
  }
  return best;
}

function partitionByBin(
  records: ReadonlyArray<InstanceRecord>,
  cb: CentroidBounds,
  axis: number,
  binIdx: number,
  numBins: number,
): { left: InstanceRecord[]; right: InstanceRecord[] } {
  const cMin = cb.cMin[axis]!;
  const ext = cb.cMax[axis]! - cMin;
  const left: InstanceRecord[] = [];
  const right: InstanceRecord[] = [];
  for (const r of records) {
    const c = r.centroid[axis]!;
    const tBin = Math.floor((c - cMin) / ext * numBins);
    const bIdx = Math.max(0, Math.min(numBins - 1, tBin));
    if (bIdx <= binIdx) left.push(r);
    else right.push(r);
  }
  return { left, right };
}

function minimumBalancedDepth(recordCount: number, maxLeaf: number): number {
  const leavesNeeded = Math.ceil(recordCount / maxLeaf);
  return leavesNeeded <= 1 ? 0 : Math.ceil(Math.log2(leavesNeeded));
}

function noteBalancedFallback(
  stats: TlasBuildStats,
  reason: TlasBalancedFallbackReason,
): void {
  stats.balancedFallbackCount += 1;
  stats.balancedFallbackReasons[reason] += 1;
}

function balancedPartition(
  records: ReadonlyArray<InstanceRecord>,
  cb: CentroidBounds,
): { left: InstanceRecord[]; right: InstanceRecord[]; axis: number } {
  const extents: readonly [number, number, number] = [
    cb.cMax[0] - cb.cMin[0],
    cb.cMax[1] - cb.cMin[1],
    cb.cMax[2] - cb.cMin[2],
  ];
  let axis = 0;
  if (extents[1] > extents[axis]!) axis = 1;
  if (extents[2] > extents[axis]!) axis = 2;
  const ordered = [...records].sort((a, b) => {
    const centroidDelta = a.centroid[axis]! - b.centroid[axis]!;
    return centroidDelta !== 0 ? centroidDelta : a.origIndex - b.origIndex;
  });
  const midpoint = Math.floor(ordered.length / 2);
  return {
    left: ordered.slice(0, midpoint),
    right: ordered.slice(midpoint),
    axis,
  };
}

function splitFitsDepth(
  leftCount: number,
  rightCount: number,
  childDepth: number,
  maxDepth: number,
  maxLeaf: number,
): boolean {
  const remainingDepth = maxDepth - childDepth;
  return (
    minimumBalancedDepth(leftCount, maxLeaf) <= remainingDepth &&
    minimumBalancedDepth(rightCount, maxLeaf) <= remainingDepth
  );
}

function buildRecursive(
  records: ReadonlyArray<InstanceRecord>,
  nodes: TlasNodeBuild[],
  permutation: number[],
  maxLeaf: number,
  numBins: number,
  depth: number,
  maxDepth: number,
  stats: TlasBuildStats,
): number {
  const thisIdx = nodes.length;
  if (records.length <= maxLeaf) {
    const leaf = buildLeaf(records);
    leaf.rightChildOrInstanceOffset = permutation.length;
    nodes.push(leaf);
    for (const r of records) permutation.push(r.origIndex);
    return thisIdx;
  }

  if (
    depth >= maxDepth ||
    minimumBalancedDepth(records.length, maxLeaf) > maxDepth - depth
  ) {
    throw new Error(
      `[@vitrum/shared-bvh/tlas] Cannot encode ${records.length} instances at depth ` +
      `${depth} within the traversal-safe maximum depth ${maxDepth}. ` +
      'Split the scene into fewer top-level instances.',
    );
  }

  const cb = computeCentroidBounds(records);
  const split = pickSplit(records, cb, numBins);
  let axis = split?.axis ?? 0;
  let left: InstanceRecord[] = [];
  let right: InstanceRecord[] = [];
  let fallbackReason: TlasBalancedFallbackReason | null = null;

  if (split == null) {
    fallbackReason = 'degenerate-centroids';
  } else if (split.cost >= records.length) {
    fallbackReason = 'non-improving-sah';
  } else {
    ({ left, right } = partitionByBin(records, cb, split.axis, split.binIdx, numBins));
    if (left.length === 0 || right.length === 0) {
      fallbackReason = 'degenerate-partition';
    } else if (
      !splitFitsDepth(left.length, right.length, depth + 1, maxDepth, maxLeaf)
    ) {
      fallbackReason = 'depth-safety';
    }
  }

  if (fallbackReason != null) {
    ({ left, right, axis } = balancedPartition(records, cb));
    noteBalancedFallback(stats, fallbackReason);
  }

  if (
    left.length === 0 ||
    right.length === 0 ||
    !splitFitsDepth(left.length, right.length, depth + 1, maxDepth, maxLeaf)
  ) {
    throw new Error(
      '[@vitrum/shared-bvh/tlas] Internal balanced fallback failed to make a ' +
      'non-empty traversal-safe partition.',
    );
  }

  nodes.push({
    min: [0, 0, 0],
    max: [0, 0, 0],
    rightChildOrInstanceOffset: 0,
    splitAxisOrInstanceCount: axis,
  });

  buildRecursive(
    left,
    nodes,
    permutation,
    maxLeaf,
    numBins,
    depth + 1,
    maxDepth,
    stats,
  );
  const rightIdx = buildRecursive(
    right,
    nodes,
    permutation,
    maxLeaf,
    numBins,
    depth + 1,
    maxDepth,
    stats,
  );

  const thisNode = nodes[thisIdx]!;
  thisNode.rightChildOrInstanceOffset = rightIdx - thisIdx;
  const leftNode = nodes[thisIdx + 1]!;
  const rightNode = nodes[rightIdx]!;
  thisNode.min = [
    Math.min(leftNode.min[0], rightNode.min[0]),
    Math.min(leftNode.min[1], rightNode.min[1]),
    Math.min(leftNode.min[2], rightNode.min[2]),
  ];
  thisNode.max = [
    Math.max(leftNode.max[0], rightNode.max[0]),
    Math.max(leftNode.max[1], rightNode.max[1]),
    Math.max(leftNode.max[2], rightNode.max[2]),
  ];

  return thisIdx;
}

function flattenNodes(nodes: ReadonlyArray<TlasNodeBuild>): Uint32Array {
  const out = new Uint32Array(nodes.length * TLAS_NODE_STRIDE_U32);
  // Reinterpret a 4-byte aligned view for the bounds.
  const f32 = new Float32Array(out.buffer);
  for (let i = 0; i < nodes.length; i++) {
    const n = nodes[i]!;
    const base = i * TLAS_NODE_STRIDE_U32;
    f32[base + 0] = n.min[0];
    f32[base + 1] = n.min[1];
    f32[base + 2] = n.min[2];
    f32[base + 3] = n.max[0];
    f32[base + 4] = n.max[1];
    f32[base + 5] = n.max[2];
    out[base + 6] = n.rightChildOrInstanceOffset >>> 0;
    out[base + 7] = n.splitAxisOrInstanceCount >>> 0;
  }
  return out;
}

export interface TlasValidationOptions {
  /** Reject leaves larger than this value. Defaults to the packed 16-bit limit. */
  readonly maxLeafInstances?: number;
  /** Reject nodes deeper than this value. May only lower the canonical limit. */
  readonly maxDepth?: number;
}

function finiteSafeIntegerInRange(
  value: number,
  name: string,
  minimum: number,
  maximum: number,
): number {
  if (!Number.isSafeInteger(value) || value < minimum || value > maximum) {
    throw new RangeError(
      `buildTlas: ${name} must be a finite safe integer in [${minimum}, ${maximum}], ` +
      `got ${String(value)}.`,
    );
  }
  return value;
}

/**
 * Validate a packed TLAS and prove that its exact depth-first traversal stack
 * requirement fits the private stack compiled into every live WGSL consumer.
 *
 * This is deliberately exported for deserializers and foreign buffer
 * producers. {@link buildTlas} runs it unconditionally before publishing a
 * result, so a successful build cannot rely on the shader's overflow policy.
 */
export function validateTlasBuild(
  data: TlasBufferView,
  opts: TlasValidationOptions = {},
): TlasValidationReport {
  const maxLeafInstances = finiteSafeIntegerInRange(
    opts.maxLeafInstances ?? TLAS_MAX_LEAF_INSTANCES,
    'validation maxLeafInstances',
    1,
    TLAS_MAX_LEAF_INSTANCES,
  );
  const maxDepth = finiteSafeIntegerInRange(
    opts.maxDepth ?? TLAS_MAX_BUILD_DEPTH,
    'validation maxDepth',
    0,
    TLAS_MAX_BUILD_DEPTH,
  );
  if (!Number.isSafeInteger(data.nodeCount) || data.nodeCount <= 0) {
    throw new Error('validateTlasBuild: nodeCount must be a positive safe integer.');
  }
  if (data.nodes.length !== data.nodeCount * TLAS_NODE_STRIDE_U32) {
    throw new Error(
      `validateTlasBuild: node buffer has ${data.nodes.length} words for ` +
      `${data.nodeCount} nodes; expected exactly ${data.nodeCount * TLAS_NODE_STRIDE_U32}.`,
    );
  }
  if (data.instanceTransforms.length !== data.blasRoots.length * 16) {
    throw new Error(
      'validateTlasBuild: instanceTransforms length must equal blasRoots.length * 16.',
    );
  }

  const nodeFloats = new Float32Array(
    data.nodes.buffer,
    data.nodes.byteOffset,
    data.nodes.length,
  );
  const visitedNodes = new Uint8Array(data.nodeCount);
  const visitedPermutation = new Uint8Array(data.instanceIndices.length);
  const visitedInstances = new Uint8Array(data.blasRoots.length);
  const stack: Array<{ readonly nodeIndex: number; readonly depth: number }> = [
    { nodeIndex: 0, depth: 0 },
  ];
  let maxTraversalStackEntries = 1;
  let maxObservedDepth = 0;
  let interiorNodeCount = 0;
  let leafNodeCount = 0;
  let maxObservedLeafInstances = 0;
  let visitedNodeCount = 0;

  while (stack.length > 0) {
    const current = stack.pop()!;
    const { nodeIndex, depth } = current;
    if (nodeIndex < 0 || nodeIndex >= data.nodeCount) {
      throw new Error(`validateTlasBuild: node reference ${nodeIndex} is out of range.`);
    }
    if (visitedNodes[nodeIndex] !== 0) {
      throw new Error(
        `validateTlasBuild: node ${nodeIndex} is referenced more than once (cycle or DAG).`,
      );
    }
    if (depth > maxDepth) {
      throw new Error(
        `validateTlasBuild: node ${nodeIndex} depth ${depth} exceeds maximum ${maxDepth}.`,
      );
    }
    visitedNodes[nodeIndex] = 1;
    visitedNodeCount += 1;
    maxObservedDepth = Math.max(maxObservedDepth, depth);
    // Child visit order is ray-dependent. The order-independent worst case is
    // the deepest root-to-node path plus one pending sibling per ancestor.
    // Measuring this validator's fixed left-first stack would underestimate a
    // deep-right/shallow-left tree even though a live ray may choose the
    // opposite order.
    maxTraversalStackEntries = Math.max(
      maxTraversalStackEntries,
      depth + 1,
    );
    if (maxTraversalStackEntries > TLAS_TRAVERSAL_STACK_DEPTH) {
      throw new Error(
        `validateTlasBuild: traversal requires ${maxTraversalStackEntries} stack entries, ` +
        `exceeding the live WGSL capacity ${TLAS_TRAVERSAL_STACK_DEPTH}.`,
      );
    }

    const base = nodeIndex * TLAS_NODE_STRIDE_U32;
    const minX = nodeFloats[base]!;
    const minY = nodeFloats[base + 1]!;
    const minZ = nodeFloats[base + 2]!;
    const maxX = nodeFloats[base + 3]!;
    const maxY = nodeFloats[base + 4]!;
    const maxZ = nodeFloats[base + 5]!;
    if (
      !Number.isFinite(minX) ||
      !Number.isFinite(minY) ||
      !Number.isFinite(minZ) ||
      !Number.isFinite(maxX) ||
      !Number.isFinite(maxY) ||
      !Number.isFinite(maxZ) ||
      minX > maxX ||
      minY > maxY ||
      minZ > maxZ
    ) {
      throw new Error(`validateTlasBuild: node ${nodeIndex} has invalid bounds.`);
    }

    const splitOrCount = data.nodes[base + 7]!;
    const isLeaf = (splitOrCount >>> 16) === 0xffff;
    if (isLeaf) {
      leafNodeCount += 1;
      const count = splitOrCount & 0x0000ffff;
      const offset = data.nodes[base + 6]!;
      if (
        count === 0 ||
        count > maxLeafInstances ||
        offset > data.instanceIndices.length ||
        count > data.instanceIndices.length - offset
      ) {
        throw new Error(
          `validateTlasBuild: leaf ${nodeIndex} has invalid range offset=${offset}, count=${count}.`,
        );
      }
      maxObservedLeafInstances = Math.max(maxObservedLeafInstances, count);
      for (let i = 0; i < count; i += 1) {
        const permutationIndex = offset + i;
        if (visitedPermutation[permutationIndex] !== 0) {
          throw new Error(
            `validateTlasBuild: permutation slot ${permutationIndex} is referenced more than once.`,
          );
        }
        visitedPermutation[permutationIndex] = 1;
        const instanceIndex = data.instanceIndices[permutationIndex]!;
        if (instanceIndex >= data.blasRoots.length) {
          throw new Error(
            `validateTlasBuild: instance reference ${instanceIndex} is out of range.`,
          );
        }
        if (visitedInstances[instanceIndex] !== 0) {
          throw new Error(
            `validateTlasBuild: original instance ${instanceIndex} is referenced more than once ` +
            'by the instance permutation.',
          );
        }
        visitedInstances[instanceIndex] = 1;
        const transformOffset = instanceIndex * 16;
        for (let lane = 0; lane < 16; lane += 1) {
          if (!Number.isFinite(data.instanceTransforms[transformOffset + lane])) {
            throw new Error(
              `validateTlasBuild: included instance ${instanceIndex} has a non-finite transform.`,
            );
          }
        }
      }
      continue;
    }

    interiorNodeCount += 1;
    if (depth >= maxDepth) {
      throw new Error(
        `validateTlasBuild: interior node ${nodeIndex} reaches maximum depth ${maxDepth}.`,
      );
    }
    if (splitOrCount > 2) {
      throw new Error(
        `validateTlasBuild: interior node ${nodeIndex} has invalid split axis ${splitOrCount}.`,
      );
    }
    const rightOffset = data.nodes[base + 6]!;
    const leftChild = nodeIndex + 1;
    const rightChild = nodeIndex + rightOffset;
    if (
      rightOffset <= 1 ||
      leftChild >= data.nodeCount ||
      rightChild >= data.nodeCount
    ) {
      throw new Error(
        `validateTlasBuild: interior node ${nodeIndex} has invalid child references.`,
      );
    }
    for (const childIndex of [leftChild, rightChild]) {
      const childBase = childIndex * TLAS_NODE_STRIDE_U32;
      const childMinX = nodeFloats[childBase]!;
      const childMinY = nodeFloats[childBase + 1]!;
      const childMinZ = nodeFloats[childBase + 2]!;
      const childMaxX = nodeFloats[childBase + 3]!;
      const childMaxY = nodeFloats[childBase + 4]!;
      const childMaxZ = nodeFloats[childBase + 5]!;
      if (
        childMinX < minX || childMinY < minY || childMinZ < minZ ||
        childMaxX > maxX || childMaxY > maxY || childMaxZ > maxZ
      ) {
        throw new Error(
          `validateTlasBuild: interior node ${nodeIndex} bounds do not enclose child ${childIndex}.`,
        );
      }
    }
    stack.push({ nodeIndex: rightChild, depth: depth + 1 });
    stack.push({ nodeIndex: leftChild, depth: depth + 1 });
  }

  if (visitedNodeCount !== data.nodeCount) {
    throw new Error(
      `validateTlasBuild: ${data.nodeCount - visitedNodeCount} node(s) are unreachable from root.`,
    );
  }
  for (let i = 0; i < visitedPermutation.length; i += 1) {
    if (visitedPermutation[i] === 0) {
      throw new Error(`validateTlasBuild: permutation slot ${i} is not referenced by a leaf.`);
    }
  }

  return {
    status: 'valid',
    nodeCount: data.nodeCount,
    interiorNodeCount,
    leafNodeCount,
    maxDepth: maxObservedDepth,
    maxLeafInstances: maxObservedLeafInstances,
    maxTraversalStackEntries,
    traversalStackCapacity: TLAS_TRAVERSAL_STACK_DEPTH,
  };
}

/**
 * Build a TLAS from a list of instances.
 *
 * Throws if `instances` is empty. (Empty TLAS has no defined root; the
 * caller should skip TLAS dispatch when no instances are present rather
 * than constructing an empty buffer.)
 */
export function buildTlas(
  instances: ReadonlyArray<TlasInstance>,
  opts: TlasBuildOptions = {},
): TlasData {
  assertDenseArray(instances, 'instances');
  if (opts == null || typeof opts !== 'object' || Array.isArray(opts)) {
    throw new TypeError('buildTlas: opts must be an object.');
  }
  const optionKeys = new Set(['maxLeafInstances', 'numBins', 'maxDepth']);
  for (const key of Reflect.ownKeys(opts)) {
    if (typeof key !== 'string' || !optionKeys.has(key)) {
      throw new RangeError(`buildTlas: unknown option ${String(key)}.`);
    }
  }
  if (instances.length === 0) {
    throw new Error('buildTlas: instances list is empty.');
  }
  const maxLeaf = finiteSafeIntegerInRange(
    opts.maxLeafInstances ?? TLAS_DEFAULT_MAX_LEAF_INSTANCES,
    'maxLeafInstances',
    1,
    TLAS_MAX_LEAF_INSTANCES,
  );
  const numBins = finiteSafeIntegerInRange(
    opts.numBins ?? TLAS_DEFAULT_NUM_BINS,
    'numBins',
    2,
    TLAS_MAX_NUM_BINS,
  );
  const maxDepth = finiteSafeIntegerInRange(
    opts.maxDepth ?? TLAS_MAX_BUILD_DEPTH,
    'maxDepth',
    0,
    TLAS_MAX_BUILD_DEPTH,
  );

  const records: InstanceRecord[] = [];
  for (let i = 0; i < instances.length; i += 1) {
    const inst = instances[i]!;
    if (inst == null || typeof inst !== 'object' || Array.isArray(inst)) {
      throw new TypeError(`buildTlas: instance ${i} must be an object.`);
    }
    const instanceKeys = new Set(['blasId', 'aabbMin', 'aabbMax', 'worldToLocal']);
    for (const key of Reflect.ownKeys(inst)) {
      if (typeof key !== 'string' || !instanceKeys.has(key)) {
        throw new RangeError(`buildTlas: instance ${i} has unknown field ${String(key)}.`);
      }
    }
    if (!Number.isSafeInteger(inst.blasId) || inst.blasId < 0 || inst.blasId > 0xffffffff) {
      throw new RangeError(
        `buildTlas: instance ${i} blasId must be an unsigned 32-bit safe integer, ` +
        `got ${String(inst.blasId)}.`,
      );
    }
    const matrixLength = intrinsicFloat32Length(inst.worldToLocal);
    if (matrixLength !== 16) {
      throw new TypeError(`buildTlas: instance ${i} worldToLocal must be an exact 16-element Float32Array.`);
    }
    assertVec3Tuple(inst.aabbMin, `instance ${i} aabbMin`);
    assertVec3Tuple(inst.aabbMax, `instance ${i} aabbMax`);
    if (isInvertedAabb(inst.aabbMin, inst.aabbMax)) {
      throw new Error(`buildTlas: instance ${i} has inverted AABB.`);
    }
    if (!isFiniteVec3(inst.aabbMin) || !isFiniteVec3(inst.aabbMax) || !isFiniteMat4(inst.worldToLocal)) {
      throw new RangeError(`buildTlas: instance ${i} has non-finite AABB or worldToLocal data.`);
    }
    records.push({
      origIndex: i,
      centroid: aabbCentroid(inst.aabbMin, inst.aabbMax),
      min: inst.aabbMin,
      max: inst.aabbMax,
    });
  }

  const nodes: TlasNodeBuild[] = [];
  const permutation: number[] = [];
  if (minimumBalancedDepth(records.length, maxLeaf) > maxDepth) {
    throw new RangeError(
      `buildTlas: ${records.length} valid instances cannot fit maxLeafInstances=${maxLeaf} ` +
      `within requested traversal-safe depth ${maxDepth}.`,
    );
  }
  const stats: TlasBuildStats = {
    balancedFallbackCount: 0,
    balancedFallbackReasons: {
      'degenerate-centroids': 0,
      'non-improving-sah': 0,
      'degenerate-partition': 0,
      'depth-safety': 0,
    },
  };
  buildRecursive(
    records,
    nodes,
    permutation,
    maxLeaf,
    numBins,
    0,
    maxDepth,
    stats,
  );

  const flatNodes = flattenNodes(nodes);
  const instanceIndices = new Uint32Array(permutation);
  const blasRoots = new Uint32Array(instances.length);
  const instanceTransforms = new Float32Array(instances.length * 16);
  for (let i = 0; i < instances.length; i++) {
    blasRoots[i] = instances[i]!.blasId >>> 0;
    instanceTransforms.set(instances[i]!.worldToLocal, i * 16);
  }
  const buffers: TlasBufferView = {
    nodes: flatNodes,
    nodeCount: nodes.length,
    instanceIndices,
    blasRoots,
    instanceTransforms,
  };
  const validation = validateTlasBuild(buffers, {
    maxLeafInstances: maxLeaf,
    maxDepth,
  });
  return {
    ...buffers,
    buildStatus: {
      status: 'valid',
      balancedFallbackCount: stats.balancedFallbackCount,
      balancedFallbackReasons: { ...stats.balancedFallbackReasons },
      validation,
    },
  };
}

/**
 * Refit (no topology change). Recomputes interior node AABBs as the
 * union of their children's bounds, after the caller has supplied
 * updated per-instance world AABBs. Useful when instances move but
 * count + leaf assignment is unchanged.
 *
 * The caller must pass AABBs in **original input order** (the same
 * order as the `instances` argument to the original `buildTlas` call);
 * the function looks up the build-order permutation from
 * `data.instanceIndices`.
 */
export function refitTlas(
  data: TlasBufferView,
  newAabbs: ReadonlyArray<{ min: readonly [number, number, number]; max: readonly [number, number, number]; }>,
): void {
  refitTlasInstances(
    data,
    newAabbs,
    data.instanceIndices,
  );
}

function validateTlasRefitAabbs(
  data: TlasBufferView,
  newAabbs: ReadonlyArray<{
    min: readonly [number, number, number];
    max: readonly [number, number, number];
  }>,
): void {
  if (newAabbs.length !== data.blasRoots.length) {
    throw new Error(
      `refitTlas: expected ${data.blasRoots.length} AABBs, got ${newAabbs.length}.`,
    );
  }
  const included = new Uint8Array(data.blasRoots.length);
  for (let permutationIndex = 0; permutationIndex < data.instanceIndices.length; permutationIndex += 1) {
    const i = data.instanceIndices[permutationIndex]!;
    if (i >= included.length) {
      throw new Error(`refitTlas: instance reference ${i} is out of range.`);
    }
    if (included[i] !== 0) continue;
    included[i] = 1;
    const aabb = newAabbs[i]!;
    if (isInvertedAabb(aabb.min, aabb.max)) {
      throw new Error(`refitTlas: instance ${i} has inverted AABB.`);
    }
    if (!isFiniteVec3(aabb.min) || !isFiniteVec3(aabb.max)) {
      throw new Error(`refitTlas: instance ${i} has non-finite AABB.`);
    }
  }
}

/**
 * Return the exact leaf-and-ancestor node set affected by original-input
 * instance indices. Indices are child-before-parent so callers may snapshot,
 * refit, and upload only these 32-byte node records.
 */
export function tlasRefitNodeIndices(
  data: TlasBufferView,
  changedInstanceIndices: ArrayLike<number>,
): Uint32Array {
  if (data.nodes.length !== data.nodeCount * TLAS_NODE_STRIDE_U32) {
    throw new Error('tlasRefitNodeIndices: node count does not match packed nodes.');
  }
  const changed = new Uint8Array(data.blasRoots.length);
  for (let i = 0; i < changedInstanceIndices.length; i += 1) {
    const instance = changedInstanceIndices[i]!;
    if (
      !Number.isSafeInteger(instance) ||
      instance < 0 ||
      instance >= changed.length
    ) {
      throw new RangeError(
        `tlasRefitNodeIndices: instance index ${instance} is out of bounds.`,
      );
    }
    changed[instance] = 1;
  }
  if (changedInstanceIndices.length === 0) return new Uint32Array(0);

  const parents = new Int32Array(data.nodeCount);
  parents.fill(-1);
  const directlyAffected = new Uint8Array(data.nodeCount);
  const referencedInstances = new Uint8Array(data.blasRoots.length);
  for (let node = 0; node < data.nodeCount; node += 1) {
    const base = node * TLAS_NODE_STRIDE_U32;
    const split = data.nodes[base + 7]!;
    const isLeaf = (split >>> 16) === 0xffff;
    if (isLeaf) {
      const offset = data.nodes[base + 6]!;
      const count = split & 0x0000ffff;
      if (count === 0 || offset + count > data.instanceIndices.length) {
        throw new Error(
          `tlasRefitNodeIndices: leaf ${node} has invalid instance span.`,
        );
      }
      for (let k = 0; k < count; k += 1) {
        const original = data.instanceIndices[offset + k]!;
        if (original >= changed.length) {
          throw new Error(
            `tlasRefitNodeIndices: leaf ${node} references invalid instance ${original}.`,
          );
        }
        referencedInstances[original] = 1;
        if (changed[original] !== 0) directlyAffected[node] = 1;
      }
      continue;
    }

    const left = node + 1;
    const right = node + data.nodes[base + 6]!;
    if (
      left >= data.nodeCount ||
      right <= node ||
      right >= data.nodeCount ||
      parents[left] !== -1 ||
      parents[right] !== -1
    ) {
      throw new Error(
        `tlasRefitNodeIndices: interior node ${node} has invalid children.`,
      );
    }
    parents[left] = node;
    parents[right] = node;
  }

  for (let instance = 0; instance < changed.length; instance += 1) {
    if (changed[instance] !== 0 && referencedInstances[instance] === 0) {
      throw new Error(
        `tlasRefitNodeIndices: changed instance ${instance} has no TLAS leaf.`,
      );
    }
  }

  const affected = new Uint8Array(data.nodeCount);
  for (let node = 0; node < data.nodeCount; node += 1) {
    if (directlyAffected[node] === 0) continue;
    for (let cursor = node; cursor >= 0; cursor = parents[cursor]!) {
      if (affected[cursor] !== 0) break;
      affected[cursor] = 1;
    }
  }
  const result: number[] = [];
  for (let node = data.nodeCount - 1; node >= 0; node -= 1) {
    if (affected[node] !== 0) result.push(node);
  }
  if (result.length === 0) {
    throw new Error('tlasRefitNodeIndices: changed instance has no TLAS leaf.');
  }
  return Uint32Array.from(result);
}

/**
 * Refit only leaves containing the changed instances and their ancestors.
 * `newAabbs` remains complete because a TLAS leaf may contain multiple
 * instances; unrelated nodes are neither read-modify-written nor returned.
 */
export function refitTlasInstances(
  data: TlasBufferView,
  newAabbs: ReadonlyArray<{
    min: readonly [number, number, number];
    max: readonly [number, number, number];
  }>,
  changedInstanceIndices: ArrayLike<number>,
): Uint32Array {
  validateTlasRefitAabbs(data, newAabbs);
  const affectedNodes = tlasRefitNodeIndices(data, changedInstanceIndices);
  const f32 = new Float32Array(
    data.nodes.buffer,
    data.nodes.byteOffset,
    data.nodes.length,
  );

  for (const i of affectedNodes) {
    const base = i * TLAS_NODE_STRIDE_U32;
    const split = data.nodes[base + 7]!;
    // Extract upper 16 bits as unsigned; mirrors the BLAS leaf-flag check
    // in buildArrayBvh.ts and avoids the int32-sign trap of `& 0xFFFF0000`.
    const isLeaf = (split >>> 16) === 0xffff;
    if (isLeaf) {
      const offset = data.nodes[base + 6]!;
      const count = split & 0x0000ffff;
      let mnX = Infinity, mnY = Infinity, mnZ = Infinity;
      let mxX = -Infinity, mxY = -Infinity, mxZ = -Infinity;
      for (let k = 0; k < count; k++) {
        const origIdx = data.instanceIndices[offset + k]!;
        const a = newAabbs[origIdx]!;
        if (a.min[0] < mnX) mnX = a.min[0];
        if (a.min[1] < mnY) mnY = a.min[1];
        if (a.min[2] < mnZ) mnZ = a.min[2];
        if (a.max[0] > mxX) mxX = a.max[0];
        if (a.max[1] > mxY) mxY = a.max[1];
        if (a.max[2] > mxZ) mxZ = a.max[2];
      }
      f32[base + 0] = mnX;
      f32[base + 1] = mnY;
      f32[base + 2] = mnZ;
      f32[base + 3] = mxX;
      f32[base + 4] = mxY;
      f32[base + 5] = mxZ;
    } else {
      // Interior: union of left (i+1) and right (i + rightOffset).
      const leftBase = (i + 1) * TLAS_NODE_STRIDE_U32;
      const rightOffset = data.nodes[base + 6]!;
      const rightBase = (i + rightOffset) * TLAS_NODE_STRIDE_U32;
      const lMnX = f32[leftBase + 0]!, lMnY = f32[leftBase + 1]!, lMnZ = f32[leftBase + 2]!;
      const lMxX = f32[leftBase + 3]!, lMxY = f32[leftBase + 4]!, lMxZ = f32[leftBase + 5]!;
      const rMnX = f32[rightBase + 0]!, rMnY = f32[rightBase + 1]!, rMnZ = f32[rightBase + 2]!;
      const rMxX = f32[rightBase + 3]!, rMxY = f32[rightBase + 4]!, rMxZ = f32[rightBase + 5]!;
      f32[base + 0] = Math.min(lMnX, rMnX);
      f32[base + 1] = Math.min(lMnY, rMnY);
      f32[base + 2] = Math.min(lMnZ, rMnZ);
      f32[base + 3] = Math.max(lMxX, rMxX);
      f32[base + 4] = Math.max(lMxY, rMxY);
      f32[base + 5] = Math.max(lMxZ, rMxZ);
    }
  }
  return affectedNodes;
}

/**
 * CPU reference traversal: returns the original-input indices of
 * **candidate** instances whose enclosing leaf AABB the ray intersects
 * (in unspecified order). When `maxLeafInstances === 1` (the default)
 * the candidate set equals the precise set. With larger leaves the
 * caller must per-instance re-test — production traversal dereferences
 * each candidate's BLAS and does the exact ray-vs-mesh test, so the
 * leaf-level conservative answer is the contract.
 *
 * Intended for unit tests + correctness pinning. The GPU traversal in
 * WGSL is the production path.
 */
export function tlasIntersect(
  data: TlasBufferView,
  origin: readonly [number, number, number],
  direction: readonly [number, number, number],
  tMax: number = Infinity,
): number[] {
  const hits: number[] = [];
  if (data.nodeCount === 0) return hits;
  if (
    !origin.every(Number.isFinite) ||
    !direction.every(Number.isFinite) ||
    Math.hypot(direction[0], direction[1], direction[2]) === 0 ||
    Number.isNaN(tMax) ||
    tMax < 0
  ) {
    return hits;
  }
  if (data.nodes.length < data.nodeCount * TLAS_NODE_STRIDE_U32) {
    throw new Error('tlasIntersect: node buffer is shorter than nodeCount.');
  }
  const f32 = new Float32Array(
    data.nodes.buffer,
    data.nodes.byteOffset,
    data.nodes.length,
  );

  function nodeAabbIntersects(nodeIdx: number): boolean {
    const base = nodeIdx * TLAS_NODE_STRIDE_U32;
    const mnX = f32[base + 0]!, mnY = f32[base + 1]!, mnZ = f32[base + 2]!;
    const mxX = f32[base + 3]!, mxY = f32[base + 4]!, mxZ = f32[base + 5]!;
    if (
      !Number.isFinite(mnX) || !Number.isFinite(mnY) || !Number.isFinite(mnZ) ||
      !Number.isFinite(mxX) || !Number.isFinite(mxY) || !Number.isFinite(mxZ) ||
      mnX > mxX || mnY > mxY || mnZ > mxZ
    ) {
      throw new Error(`tlasIntersect: node ${nodeIdx} has invalid bounds.`);
    }
    let tEnter = 0;
    let tExit = tMax;
    const clipAxis = (o: number, d: number, mn: number, mx: number): boolean => {
      // Match WGSL safeInvDir: magnitudes below 1e-30 are effectively parallel.
      // Handle the slab explicitly so an on-boundary 0 * Infinity cannot become NaN.
      if (Math.abs(d) < 1e-30) return o >= mn && o <= mx;
      const t0 = (mn - o) / d;
      const t1 = (mx - o) / d;
      tEnter = Math.max(tEnter, Math.min(t0, t1));
      tExit = Math.min(tExit, Math.max(t0, t1));
      return tEnter <= tExit;
    };
    if (!clipAxis(origin[0], direction[0], mnX, mxX)) return false;
    if (!clipAxis(origin[1], direction[1], mnY, mxY)) return false;
    if (!clipAxis(origin[2], direction[2], mnZ, mxZ)) return false;
    return tEnter <= tExit && tExit >= 0;
  }

  const stack: number[] = [0];
  while (stack.length > 0) {
    const nodeIdx = stack.pop()!;
    if (nodeIdx >= data.nodeCount) {
      throw new Error(`tlasIntersect: node reference ${nodeIdx} is out of range.`);
    }
    if (!nodeAabbIntersects(nodeIdx)) continue;
    const base = nodeIdx * TLAS_NODE_STRIDE_U32;
    const split = data.nodes[base + 7]!;
    const isLeaf = (split >>> 16) === 0xffff;
    if (isLeaf) {
      const offset = data.nodes[base + 6]!;
      const count = split & 0x0000ffff;
      if (count === 0 || offset + count > data.instanceIndices.length) {
        throw new Error(`tlasIntersect: node ${nodeIdx} has an invalid leaf range.`);
      }
      for (let k = 0; k < count; k++) {
        const instanceIdx = data.instanceIndices[offset + k]!;
        if (instanceIdx >= data.blasRoots.length) {
          throw new Error(`tlasIntersect: instance reference ${instanceIdx} is out of range.`);
        }
        hits.push(instanceIdx);
      }
    } else {
      const rightOff = data.nodes[base + 6]!;
      const left = nodeIdx + 1;
      const right = nodeIdx + rightOff;
      if (rightOff <= 1 || left >= data.nodeCount || right >= data.nodeCount) {
        throw new Error(`tlasIntersect: node ${nodeIdx} has invalid child references.`);
      }
      stack.push(left);
      stack.push(right);
    }
  }
  return hits;
}
