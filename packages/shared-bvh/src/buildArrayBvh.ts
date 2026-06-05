/**
 * THREE-independent CPU BVH builder — binned SAH per Wald 2007 "On fast
 * Construction of SAH-based Bounding Volume Hierarchies" (IEEE Symposium
 * on Interactive Ray Tracing, 2007). K=16 bins per axis.
 *
 * Canonical hoist of pt-webgpu's hand-rolled `buildCpuBvh` into the shared
 * library (W2-C2 — see `plan/premium-grade-refactor-20260517.md`).
 * Input is raw typed-array geometry — NO `three` or `three-mesh-bvh`
 * dependency at the import boundary, by construction. Callers that have
 * THREE scenes use `buildSceneBVH` (sibling export in `bvhCommon.ts`),
 * which adapts THREE objects to the raw arrays this builder accepts.
 *
 * Node layout (32 bytes, 8 × u32) — matches the rest of the vitrum stack
 * (shared-bvh/normalizeBvhInteriorOffsets, walkaround-hybrid/common.wgsl,
 * pt-webgpu's WGSL traversal):
 *   f32[0..2]  boundsMin xyz
 *   f32[3..5]  boundsMax xyz
 *   u32[6]     rightChildOrTriOffset
 *              - interior node: RELATIVE offset to right child node
 *                (rightChildIndex − thisNodeIndex). Left child is always
 *                nodeIndex + 1. Invariant: 1 ≤ offset < totalNodes.
 *              - leaf node:     absolute triangle offset into the
 *                reorderedIndices array (first triangle of the leaf).
 *   u32[7]     splitAxisOrTriCount
 *              - interior: split axis (0=X, 1=Y, 2=Z)
 *              - leaf:     0xFFFF0000 | triangleCount
 *
 * Index buffer stride (caller-configurable via `indexStride`):
 *   3 — `array<vec3u>` form (RC, DDGI traversal shaders).
 *   4 — `array<vec4u>` form (pt-webgpu; ReSTIR with payload in .w).
 *       For stride 4 the three vertex indices occupy .x .y .z; .w is
 *       always zero (zero-fill contract).
 *
 * Position buffer stride (caller-configurable via `positionStride`):
 *   3 — packed (12 bytes/vertex), TSL / raster path.
 *   4 — vec3f-aligned (16 bytes/vertex), the WGSL `array<vec3f>` layout.
 */

const LEAFNODE_FLAG = 0xffff0000;

/**
 * Leaf-node test for the packed `splitAxisOrTriCount` word (node slot 7).
 *
 * Use `(splitWord >>> 16) === 0xffff`, NOT `(splitWord & 0xffff0000) === 0xffff0000`:
 * a real uint32 leaf word is ≥ 0x80000000, so JS coerces it to a negative
 * int32 under `&`, which can never `===` the positive literal `0xffff0000`
 * (4294901760) — that comparison is a dead branch. `>>> 16` stays unsigned.
 */
export function isLeafSplit(splitWord: number): boolean {
  return (splitWord >>> 16) === 0xffff;
}

/** Default upper bound on triangles per leaf (Wald 2007 §3 sweet spot). */
const DEFAULT_MAX_LEAF_TRIANGLES = 4;

/** Default number of SAH bins per axis (Wald 2007 recommends 16). */
const DEFAULT_NUM_BINS = 16;

interface TriangleRecord {
  readonly triIndex: number;
  readonly centroid: readonly [number, number, number];
  readonly min: readonly [number, number, number];
  readonly max: readonly [number, number, number];
}

interface NodeBuild {
  min: [number, number, number];
  max: [number, number, number];
  rightChildOrTriOffset: number;
  splitAxisOrTriCount: number;
}

/** Per-axis bin accumulator for surface area heuristic. */
interface BinData {
  min: [number, number, number];
  max: [number, number, number];
  count: number;
}

function getPosition(
  positions: Float32Array,
  vertexIndex: number,
  positionStride: number,
): readonly [number, number, number] {
  const base = vertexIndex * positionStride;
  return [positions[base] ?? 0, positions[base + 1] ?? 0, positions[base + 2] ?? 0];
}

function min3(
  a: readonly [number, number, number],
  b: readonly [number, number, number],
  c: readonly [number, number, number],
): [number, number, number] {
  return [
    Math.min(a[0], b[0], c[0]),
    Math.min(a[1], b[1], c[1]),
    Math.min(a[2], b[2], c[2]),
  ];
}

function max3(
  a: readonly [number, number, number],
  b: readonly [number, number, number],
  c: readonly [number, number, number],
): [number, number, number] {
  return [
    Math.max(a[0], b[0], c[0]),
    Math.max(a[1], b[1], c[1]),
    Math.max(a[2], b[2], c[2]),
  ];
}

function triCentroid(
  min: readonly [number, number, number],
  max: readonly [number, number, number],
): [number, number, number] {
  return [
    0.5 * (min[0] + max[0]),
    0.5 * (min[1] + max[1]),
    0.5 * (min[2] + max[2]),
  ];
}

/** Surface area of an AABB. Returns 0 for degenerate (empty) boxes. */
function surfaceArea(
  minX: number, minY: number, minZ: number,
  maxX: number, maxY: number, maxZ: number,
): number {
  const dx = maxX - minX;
  const dy = maxY - minY;
  const dz = maxZ - minZ;
  if (dx <= 0 || dy <= 0 || dz <= 0) return 0;
  return 2 * (dx * dy + dy * dz + dz * dx);
}

function makeEmptyBin(): BinData {
  return {
    min: [Infinity, Infinity, Infinity],
    max: [-Infinity, -Infinity, -Infinity],
    count: 0,
  };
}

function resetBin(bin: BinData): void {
  bin.min[0] = Infinity;
  bin.min[1] = Infinity;
  bin.min[2] = Infinity;
  bin.max[0] = -Infinity;
  bin.max[1] = -Infinity;
  bin.max[2] = -Infinity;
  bin.count = 0;
}

function growBin(bin: BinData, r: TriangleRecord): void {
  bin.min[0] = Math.min(bin.min[0], r.min[0]);
  bin.min[1] = Math.min(bin.min[1], r.min[1]);
  bin.min[2] = Math.min(bin.min[2], r.min[2]);
  bin.max[0] = Math.max(bin.max[0], r.max[0]);
  bin.max[1] = Math.max(bin.max[1], r.max[1]);
  bin.max[2] = Math.max(bin.max[2], r.max[2]);
  bin.count += 1;
}

export interface CpuBvhBuildResult {
  /** Packed 32-byte node buffer (see file-header comment for layout). */
  readonly bvhNodes: Float32Array;
  /** Triangle indices in BVH-traversal order; same stride as the input. */
  readonly reorderedIndices: Uint32Array;
  /** Per-triangle material IDs, reordered to match `reorderedIndices`. */
  readonly reorderedTriMaterialIds: Uint32Array;
}

export interface BuildArrayBvhOpts {
  /**
   * Vertex stride in floats:
   *   3 → packed 12-byte layout (default raster/TSL path).
   *   4 → 16-byte vec3f-aligned layout used by pt-webgpu's WGSL.
   * Defaults to 4 for backward compatibility with the historical pt-webgpu
   * call site — that consumer always packed positions as `vec4f`.
   */
  positionStride?: number;

  /**
   * Index stride in u32s:
   *   3 → `array<vec3u>` form (no padding).
   *   4 → `array<vec4u>` form (three vertex indices in .x.y.z, `.w = 0`).
   * Defaults to 4 for backward compatibility with the historical pt-webgpu
   * call site. The output `reorderedIndices` always uses the same stride
   * as the input.
   */
  indexStride?: number;

  /**
   * Maximum triangles per leaf node. Defaults to 4 (Wald 2007 §3 sweet
   * spot for ~16-bin SAH; matches the pt-webgpu builder's historical
   * constant).
   */
  maxLeafTriangles?: number;

  /**
   * Number of SAH bins per axis. Defaults to 16 (Wald 2007's
   * recommended value; matches pt-webgpu's historical constant).
   */
  binCount?: number;
}

/**
 * Build a binned-SAH BVH over raw triangle data.
 *
 * @param positions       Vertex positions, stride = `opts.positionStride`
 *                        (default 4 floats/vertex).
 * @param indices         Triangle indices, stride = `opts.indexStride`
 *                        (default 4 u32/triangle, `.w` ignored).
 * @param triMaterialIds  Per-triangle material id (one u32 per triangle).
 * @param opts            See {@link BuildArrayBvhOpts}.
 * @returns               Packed node buffer + BVH-reordered indices +
 *                        BVH-reordered triMaterialIds (all three caller-
 *                        owned typed arrays).
 *
 * Behaviour notes:
 *  - No dependency on `three` or `three-mesh-bvh` at the import boundary.
 *  - Output bytes are stable for the same input + same opts.
 *  - The function preserves the input arrays — the inputs are read but
 *    never mutated.
 *  - Empty input (`indices.length / indexStride === 0`) returns a single
 *    zero-filled "empty leaf" node + the unmodified input arrays (so
 *    callers don't have to special-case the zero-triangle path).
 */
export function buildArrayBvh(
  positions: Float32Array,
  indices: Uint32Array,
  triMaterialIds: Uint32Array,
  opts: BuildArrayBvhOpts = {},
): CpuBvhBuildResult {
  const positionStride = opts.positionStride ?? 4;
  const indexStride = opts.indexStride ?? 4;
  const maxLeafTriangles = opts.maxLeafTriangles ?? DEFAULT_MAX_LEAF_TRIANGLES;
  const numBins = opts.binCount ?? DEFAULT_NUM_BINS;

  const triCount = Math.floor(indices.length / indexStride);
  if (triCount === 0) {
    const emptyNode = new Float32Array(8);
    return {
      bvhNodes: emptyNode,
      reorderedIndices: indices,
      reorderedTriMaterialIds: triMaterialIds,
    };
  }

  const records: TriangleRecord[] = [];
  for (let t = 0; t < triCount; t += 1) {
    const i0 = indices[t * indexStride] ?? 0;
    const i1 = indices[t * indexStride + 1] ?? 0;
    const i2 = indices[t * indexStride + 2] ?? 0;
    const a = getPosition(positions, i0, positionStride);
    const b = getPosition(positions, i1, positionStride);
    const c = getPosition(positions, i2, positionStride);
    const triMin = min3(a, b, c);
    const triMax = max3(a, b, c);
    records.push({
      triIndex: t,
      min: triMin,
      max: triMax,
      centroid: triCentroid(triMin, triMax),
    });
  }

  const nodes: NodeBuild[] = [];
  const orderedTriangles: number[] = [];
  const bins: BinData[] = Array.from({ length: numBins }, makeEmptyBin);
  const prefixMinX = new Float32Array(numBins);
  const prefixMinY = new Float32Array(numBins);
  const prefixMinZ = new Float32Array(numBins);
  const prefixMaxX = new Float32Array(numBins);
  const prefixMaxY = new Float32Array(numBins);
  const prefixMaxZ = new Float32Array(numBins);
  const prefixCount = new Int32Array(numBins);
  const suffixMinX = new Float32Array(numBins);
  const suffixMinY = new Float32Array(numBins);
  const suffixMinZ = new Float32Array(numBins);
  const suffixMaxX = new Float32Array(numBins);
  const suffixMaxY = new Float32Array(numBins);
  const suffixMaxZ = new Float32Array(numBins);
  const suffixCount = new Int32Array(numBins);

  /**
   * Recursively build a BVH subtree for `subset` starting at `nodes[nodes.length]`.
   *
   * Algorithm: binned SAH per Wald 2007 §4.
   *  1. Compute centroid AABB.
   *  2. For each axis, bin centroids into `numBins` uniform buckets.
   *  3. Sweep left→right and right→left to compute prefix surface areas + counts.
   *  4. Evaluate SAH cost at each of the `numBins − 1` split planes.
   *  5. If best SAH cost ≥ leaf cost (= N triangles), make a leaf.
   *  6. Otherwise partition and recurse.
   *
   * Returns the absolute node index of the subtree root.
   */
  const build = (subset: TriangleRecord[]): number => {
    const nodeIndex = nodes.length;
    const node: NodeBuild = {
      min: [Infinity, Infinity, Infinity],
      max: [-Infinity, -Infinity, -Infinity],
      rightChildOrTriOffset: 0,
      splitAxisOrTriCount: 0,
    };

    // Compute node AABB.
    for (const r of subset) {
      node.min[0] = Math.min(node.min[0], r.min[0]);
      node.min[1] = Math.min(node.min[1], r.min[1]);
      node.min[2] = Math.min(node.min[2], r.min[2]);
      node.max[0] = Math.max(node.max[0], r.max[0]);
      node.max[1] = Math.max(node.max[1], r.max[1]);
      node.max[2] = Math.max(node.max[2], r.max[2]);
    }
    nodes.push(node);

    // Leaf: too few triangles to split profitably.
    if (subset.length <= maxLeafTriangles) {
      const leafOffset = orderedTriangles.length;
      for (const r of subset) orderedTriangles.push(r.triIndex);
      node.rightChildOrTriOffset = leafOffset;
      node.splitAxisOrTriCount = LEAFNODE_FLAG | subset.length;
      return nodeIndex;
    }

    // Compute centroid AABB for bin placement.
    const cMin: [number, number, number] = [Infinity, Infinity, Infinity];
    const cMax: [number, number, number] = [-Infinity, -Infinity, -Infinity];
    for (const r of subset) {
      cMin[0] = Math.min(cMin[0], r.centroid[0]);
      cMin[1] = Math.min(cMin[1], r.centroid[1]);
      cMin[2] = Math.min(cMin[2], r.centroid[2]);
      cMax[0] = Math.max(cMax[0], r.centroid[0]);
      cMax[1] = Math.max(cMax[1], r.centroid[1]);
      cMax[2] = Math.max(cMax[2], r.centroid[2]);
    }

    // Parent surface area (for SAH cost normalisation).
    const parentSA = surfaceArea(
      node.min[0], node.min[1], node.min[2],
      node.max[0], node.max[1], node.max[2],
    );
    // Leaf cost: traversal terminates here, each triangle pays one test.
    // (In relative SAH units, the leaf cost is just N; the parent SA cancels
    // when comparing against parentSA * N.)
    const leafCost = subset.length;

    let bestCost = Infinity;
    let bestAxis = 0;
    let bestSplit = 0; // index of the last bin in the LEFT partition

    for (let axis = 0; axis < 3; axis++) {
      const span = cMax[axis]! - cMin[axis]!;
      if (span <= 1e-9) continue; // degenerate — all centroids co-planar on this axis

      // Bin triangles.
      for (let i = 0; i < numBins; i += 1) resetBin(bins[i]!);
      for (const r of subset) {
        const t = (r.centroid[axis]! - cMin[axis]!) / span;
        const binIdx = Math.min(numBins - 1, Math.floor(t * numBins));
        growBin(bins[binIdx]!, r);
      }

      // Prefix left sweep: prefixMin/Max[i] = merged AABB of bins [0..i].
      {
        let bminX = Infinity, bminY = Infinity, bminZ = Infinity;
        let bmaxX = -Infinity, bmaxY = -Infinity, bmaxZ = -Infinity;
        let cnt = 0;
        for (let i = 0; i < numBins; i++) {
          const b = bins[i]!;
          bminX = Math.min(bminX, b.min[0]);
          bminY = Math.min(bminY, b.min[1]);
          bminZ = Math.min(bminZ, b.min[2]);
          bmaxX = Math.max(bmaxX, b.max[0]);
          bmaxY = Math.max(bmaxY, b.max[1]);
          bmaxZ = Math.max(bmaxZ, b.max[2]);
          cnt += b.count;
          prefixMinX[i] = bminX; prefixMinY[i] = bminY; prefixMinZ[i] = bminZ;
          prefixMaxX[i] = bmaxX; prefixMaxY[i] = bmaxY; prefixMaxZ[i] = bmaxZ;
          prefixCount[i] = cnt;
        }
      }

      // Suffix right sweep: suffixMin/Max[i] = merged AABB of bins [i..numBins-1].
      {
        let bminX = Infinity, bminY = Infinity, bminZ = Infinity;
        let bmaxX = -Infinity, bmaxY = -Infinity, bmaxZ = -Infinity;
        let cnt = 0;
        for (let i = numBins - 1; i >= 0; i--) {
          const b = bins[i]!;
          bminX = Math.min(bminX, b.min[0]);
          bminY = Math.min(bminY, b.min[1]);
          bminZ = Math.min(bminZ, b.min[2]);
          bmaxX = Math.max(bmaxX, b.max[0]);
          bmaxY = Math.max(bmaxY, b.max[1]);
          bmaxZ = Math.max(bmaxZ, b.max[2]);
          cnt += b.count;
          suffixMinX[i] = bminX; suffixMinY[i] = bminY; suffixMinZ[i] = bminZ;
          suffixMaxX[i] = bmaxX; suffixMaxY[i] = bmaxY; suffixMaxZ[i] = bmaxZ;
          suffixCount[i] = cnt;
        }
      }

      // Evaluate numBins−1 split planes (split after bin i → left=[0..i], right=[i+1..K-1]).
      for (let split = 0; split < numBins - 1; split++) {
        const leftCount = prefixCount[split] ?? 0;
        const rightCount = suffixCount[split + 1] ?? 0;
        if (leftCount === 0 || rightCount === 0) continue;

        const leftSA = surfaceArea(
          prefixMinX[split]!, prefixMinY[split]!, prefixMinZ[split]!,
          prefixMaxX[split]!, prefixMaxY[split]!, prefixMaxZ[split]!,
        );
        const rightSA = surfaceArea(
          suffixMinX[split + 1]!, suffixMinY[split + 1]!, suffixMinZ[split + 1]!,
          suffixMaxX[split + 1]!, suffixMaxY[split + 1]!, suffixMaxZ[split + 1]!,
        );

        // SAH cost (unnormalised — parentSA cancels when comparing to leafCost).
        // cost = (leftSA * leftCount + rightSA * rightCount) / parentSA
        const cost = parentSA > 0
          ? (leftSA * leftCount + rightSA * rightCount) / parentSA
          : leftSA * leftCount + rightSA * rightCount;

        if (cost < bestCost) {
          bestCost = cost;
          bestAxis = axis;
          bestSplit = split;
        }
      }
    }

    // If SAH didn't find a split cheaper than making a leaf, create a leaf.
    // (This happens when all centroids are co-planar on every axis, or when
    // the SAH cost exceeds N.)
    if (bestCost >= leafCost || bestCost === Infinity) {
      const leafOffset = orderedTriangles.length;
      for (const r of subset) orderedTriangles.push(r.triIndex);
      node.rightChildOrTriOffset = leafOffset;
      node.splitAxisOrTriCount = LEAFNODE_FLAG | subset.length;
      return nodeIndex;
    }

    // Partition at the chosen bin boundary.
    const span = cMax[bestAxis]! - cMin[bestAxis]!;
    const left: TriangleRecord[] = [];
    const right: TriangleRecord[] = [];
    for (const r of subset) {
      const t = (r.centroid[bestAxis]! - cMin[bestAxis]!) / span;
      const binIdx = Math.min(numBins - 1, Math.floor(t * numBins));
      if (binIdx <= bestSplit) {
        left.push(r);
      } else {
        right.push(r);
      }
    }

    // Degenerate partition: SAH chose a split that put everything on one side.
    // Fall back to a leaf to avoid infinite recursion.
    if (left.length === 0 || right.length === 0) {
      const leafOffset = orderedTriangles.length;
      for (const r of subset) orderedTriangles.push(r.triIndex);
      node.rightChildOrTriOffset = leafOffset;
      node.splitAxisOrTriCount = LEAFNODE_FLAG | subset.length;
      return nodeIndex;
    }

    // Recurse. Left subtree is built first (its root is nodeIndex + 1).
    build(left);
    const rightChild = build(right);

    // Store RELATIVE right-child offset (rightChild − nodeIndex).
    // Left child is always nodeIndex + 1 (the immediately-following node).
    // Invariant: 1 ≤ rightChildOrTriOffset < totalNodes for all interior nodes.
    node.rightChildOrTriOffset = rightChild - nodeIndex;
    node.splitAxisOrTriCount = bestAxis;
    return nodeIndex;
  };

  build(records);

  // Dev/test mode: verify the relative-offset encoding invariant.
  // Every interior node's rightChildOrTriOffset must satisfy 1 ≤ offset < totalNodes.
  // Gated behind NODE_ENV so it is zero-cost in production bundles.
  //
  // Read `process` via `globalThis` so the check is TS-clean without
  // requiring `@types/node` in every downstream consumer (this module is
  // also dragged into the example apps, which have only DOM lib types).
  const proc = (globalThis as { process?: { env?: Record<string, string | undefined> } }).process;
  if (proc && proc.env?.['NODE_ENV'] !== 'production') {
    const n = nodes.length;
    for (let i = 0; i < n; i++) {
      const node = nodes[i];
      if (node == null) continue;
      if (isLeafSplit(node.splitAxisOrTriCount)) continue;
      const offset = node.rightChildOrTriOffset;
      if (offset < 1 || offset >= n) {
        throw new Error(
          `[@vitrum/shared-bvh/buildArrayBvh] Interior node ${i} has invalid relative right-child ` +
            `offset ${offset} (must be in [1, ${n - 1}]). Build logic is broken.`,
        );
      }
    }
  }

  const reorderedIndices = new Uint32Array(indices.length);
  const reorderedTriMaterialIds = new Uint32Array(triMaterialIds.length);
  for (let newTri = 0; newTri < orderedTriangles.length; newTri += 1) {
    const oldTri = orderedTriangles[newTri] ?? 0;
    for (let k = 0; k < 3; k += 1) {
      reorderedIndices[newTri * indexStride + k] = indices[oldTri * indexStride + k] ?? 0;
    }
    // Zero-fill any padding lanes beyond the three vertex indices
    // (stride 4 → .w = 0; other strides match the documented contract).
    for (let k = 3; k < indexStride; k += 1) {
      reorderedIndices[newTri * indexStride + k] = 0;
    }
    reorderedTriMaterialIds[newTri] = triMaterialIds[oldTri] ?? 0;
  }

  const nodeBuffer = new ArrayBuffer(nodes.length * 32);
  const dv = new DataView(nodeBuffer);
  for (let i = 0; i < nodes.length; i += 1) {
    const node = nodes[i];
    if (node == null) continue;
    const off = i * 32;
    dv.setFloat32(off + 0, node.min[0], true);
    dv.setFloat32(off + 4, node.min[1], true);
    dv.setFloat32(off + 8, node.min[2], true);
    dv.setFloat32(off + 12, node.max[0], true);
    dv.setFloat32(off + 16, node.max[1], true);
    dv.setFloat32(off + 20, node.max[2], true);
    dv.setUint32(off + 24, node.rightChildOrTriOffset >>> 0, true);
    dv.setUint32(off + 28, node.splitAxisOrTriCount >>> 0, true);
  }

  return {
    bvhNodes: new Float32Array(nodeBuffer),
    reorderedIndices,
    reorderedTriMaterialIds,
  };
}
