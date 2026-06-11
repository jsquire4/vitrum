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

import { BVH_NODE_FLOATS } from './strides.js';

const TLAS_LEAFNODE_FLAG = 0xffff0000;
const TLAS_DEFAULT_MAX_LEAF_INSTANCES = 1;       // typically 1 instance/leaf
const TLAS_DEFAULT_NUM_BINS = 16;
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
  readonly maxLeafInstances?: number;
  readonly numBins?: number;
}

/** Built TLAS payload — designed for direct DMA to a WebGPU storage buffer. */
export interface TlasData {
  /** 8 × u32 per node, packed. Tree is breadth-first; node 0 is the root. */
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

function aabbCentroid(min: readonly [number, number, number], max: readonly [number, number, number])
  : readonly [number, number, number] {
  return [
    (min[0] + max[0]) * 0.5,
    (min[1] + max[1]) * 0.5,
    (min[2] + max[2]) * 0.5,
  ];
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

function buildRecursive(
  records: ReadonlyArray<InstanceRecord>,
  nodes: TlasNodeBuild[],
  permutation: number[],
  maxLeaf: number,
  numBins: number,
): number {
  const thisIdx = nodes.length;
  if (records.length <= maxLeaf) {
    const leaf = buildLeaf(records);
    leaf.rightChildOrInstanceOffset = permutation.length;     // patched: instance offset
    nodes.push(leaf);
    for (const r of records) permutation.push(r.origIndex);
    return thisIdx;
  }

  // Compute centroid bounds ONCE; both SAH evaluation and partitioning
  // consume them. Mirrors the BLAS builder pattern in `buildArrayBvh.ts`
  // (avoids the redundant rescan partitionByBin used to do).
  const cb = computeCentroidBounds(records);
  const split = pickSplit(records, cb, numBins);
  if (split == null || split.cost >= records.length) {
    // No improving split — emit a leaf even if larger than maxLeaf.
    const leaf = buildLeaf(records);
    leaf.rightChildOrInstanceOffset = permutation.length;
    nodes.push(leaf);
    for (const r of records) permutation.push(r.origIndex);
    return thisIdx;
  }

  // Reserve this node slot; we'll fill it after recursing.
  nodes.push({
    min: [0, 0, 0],
    max: [0, 0, 0],
    rightChildOrInstanceOffset: 0,
    splitAxisOrInstanceCount: split.axis,
  });

  const { left, right } = partitionByBin(records, cb, split.axis, split.binIdx, numBins);
  if (left.length === 0 || right.length === 0) {
    // Degenerate split → fall back to a leaf.
    if (records.length > 0xffff) {
      throw new Error(
        `[@vitrum/shared-bvh/tlas] Degenerate-partition leaf instance count ${records.length} exceeds the ` +
        `16-bit limit (0xFFFF = 65535). Reduce the number of instances per leaf.`,
      );
    }
    const leaf = buildLeaf(records);
    leaf.rightChildOrInstanceOffset = permutation.length;
    leaf.splitAxisOrInstanceCount = TLAS_LEAFNODE_FLAG | records.length;
    nodes[thisIdx] = leaf;
    for (const r of records) permutation.push(r.origIndex);
    return thisIdx;
  }

  buildRecursive(left, nodes, permutation, maxLeaf, numBins);
  const rightIdx = buildRecursive(right, nodes, permutation, maxLeaf, numBins);

  const thisNode = nodes[thisIdx]!;
  thisNode.rightChildOrInstanceOffset = rightIdx - thisIdx;
  // Union of children's bounds.
  const ln = nodes[thisIdx + 1]!;
  const rn = nodes[rightIdx]!;
  thisNode.min = [
    Math.min(ln.min[0], rn.min[0]),
    Math.min(ln.min[1], rn.min[1]),
    Math.min(ln.min[2], rn.min[2]),
  ];
  thisNode.max = [
    Math.max(ln.max[0], rn.max[0]),
    Math.max(ln.max[1], rn.max[1]),
    Math.max(ln.max[2], rn.max[2]),
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
  if (instances.length === 0) {
    throw new Error('buildTlas: instances list is empty.');
  }
  const maxLeaf = Math.max(1, opts.maxLeafInstances ?? TLAS_DEFAULT_MAX_LEAF_INSTANCES);
  const numBins = Math.max(2, opts.numBins ?? TLAS_DEFAULT_NUM_BINS);

  const records: InstanceRecord[] = instances.map((inst, i) => {
    if (inst.worldToLocal.length !== 16) {
      throw new Error(`buildTlas: instance ${i} worldToLocal length ${inst.worldToLocal.length} != 16.`);
    }
    if (inst.aabbMax[0] < inst.aabbMin[0]
     || inst.aabbMax[1] < inst.aabbMin[1]
     || inst.aabbMax[2] < inst.aabbMin[2]) {
      throw new Error(`buildTlas: instance ${i} has inverted AABB.`);
    }
    return {
      origIndex: i,
      centroid: aabbCentroid(inst.aabbMin, inst.aabbMax),
      min: inst.aabbMin,
      max: inst.aabbMax,
    };
  });

  const nodes: TlasNodeBuild[] = [];
  const permutation: number[] = [];
  buildRecursive(records, nodes, permutation, maxLeaf, numBins);

  const flatNodes = flattenNodes(nodes);
  const instanceIndices = new Uint32Array(permutation);
  const blasRoots = new Uint32Array(instances.length);
  const instanceTransforms = new Float32Array(instances.length * 16);
  for (let i = 0; i < instances.length; i++) {
    blasRoots[i] = instances[i]!.blasId >>> 0;
    instanceTransforms.set(instances[i]!.worldToLocal, i * 16);
  }
  return {
    nodes: flatNodes,
    nodeCount: nodes.length,
    instanceIndices,
    blasRoots,
    instanceTransforms,
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
  data: TlasData,
  newAabbs: ReadonlyArray<{ min: readonly [number, number, number]; max: readonly [number, number, number]; }>,
): void {
  if (newAabbs.length !== data.blasRoots.length) {
    throw new Error(
      `refitTlas: expected ${data.blasRoots.length} AABBs, got ${newAabbs.length}.`,
    );
  }
  const f32 = new Float32Array(data.nodes.buffer);

  // Walk nodes back-to-front; leaves get their AABBs from the instance
  // table, interiors get them from their already-refit children.
  for (let i = data.nodeCount - 1; i >= 0; i--) {
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
  data: TlasData,
  origin: readonly [number, number, number],
  direction: readonly [number, number, number],
  tMax: number = Infinity,
): number[] {
  const hits: number[] = [];
  if (data.nodeCount === 0) return hits;
  const f32 = new Float32Array(data.nodes.buffer);

  // Pre-compute inverse direction for slab tests. Treat zero components
  // as ±∞ to keep the algorithm branchless (parallel slab).
  const idx = direction[0] !== 0 ? 1 / direction[0] : direction[0] >= 0 ? Infinity : -Infinity;
  const idy = direction[1] !== 0 ? 1 / direction[1] : direction[1] >= 0 ? Infinity : -Infinity;
  const idz = direction[2] !== 0 ? 1 / direction[2] : direction[2] >= 0 ? Infinity : -Infinity;

  function nodeAabbIntersects(nodeIdx: number): boolean {
    const base = nodeIdx * TLAS_NODE_STRIDE_U32;
    const mnX = f32[base + 0]!, mnY = f32[base + 1]!, mnZ = f32[base + 2]!;
    const mxX = f32[base + 3]!, mxY = f32[base + 4]!, mxZ = f32[base + 5]!;
    let tEnter = 0;
    let tExit = tMax;
    const t0x = (mnX - origin[0]) * idx;
    const t1x = (mxX - origin[0]) * idx;
    tEnter = Math.max(tEnter, Math.min(t0x, t1x));
    tExit = Math.min(tExit, Math.max(t0x, t1x));
    const t0y = (mnY - origin[1]) * idy;
    const t1y = (mxY - origin[1]) * idy;
    tEnter = Math.max(tEnter, Math.min(t0y, t1y));
    tExit = Math.min(tExit, Math.max(t0y, t1y));
    const t0z = (mnZ - origin[2]) * idz;
    const t1z = (mxZ - origin[2]) * idz;
    tEnter = Math.max(tEnter, Math.min(t0z, t1z));
    tExit = Math.min(tExit, Math.max(t0z, t1z));
    return tEnter <= tExit && tExit >= 0;
  }

  const stack: number[] = [0];
  while (stack.length > 0) {
    const nodeIdx = stack.pop()!;
    if (!nodeAabbIntersects(nodeIdx)) continue;
    const base = nodeIdx * TLAS_NODE_STRIDE_U32;
    const split = data.nodes[base + 7]!;
    const isLeaf = (split >>> 16) === 0xffff;
    if (isLeaf) {
      const offset = data.nodes[base + 6]!;
      const count = split & 0x0000ffff;
      for (let k = 0; k < count; k++) hits.push(data.instanceIndices[offset + k]!);
    } else {
      const rightOff = data.nodes[base + 6]!;
      stack.push(nodeIdx + 1);
      stack.push(nodeIdx + rightOff);
    }
  }
  return hits;
}
