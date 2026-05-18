/**
 * CPU BVH builder — binned SAH per Wald 2007 "On fast Construction of
 * SAH-based Bounding Volume Hierarchies" (IEEE Symposium on Interactive
 * Ray Tracing, 2007). K=16 bins per axis.
 *
 * Node layout (32 bytes, 8 × u32):
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
 * The relative right-child encoding matches shared-bvh/normalizeBvhInteriorOffsets
 * and walkaround-hybrid/common.wgsl — all three packages share this canonical
 * convention.
 *
 * Index buffer: stride 4 (vec4u). The three vertex indices occupy .x .y .z;
 * .w is always zero (zero-fill contract). pt-webgpu's WGSL reads
 * `array<vec4u>` — stride 4 is required for correct WGSL alignment.
 */

const LEAFNODE_FLAG = 0xffff0000;
const MAX_LEAF_TRIANGLES = 4;

/** Number of SAH bins per axis (Wald 2007 recommends 16). */
const NUM_BINS = 16;

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

function getPosition(positions: Float32Array, vertexIndex: number): readonly [number, number, number] {
  const base = vertexIndex * 4;
  return [positions[base] ?? 0, positions[base + 1] ?? 0, positions[base + 2] ?? 0];
}

function min3(a: readonly [number, number, number], b: readonly [number, number, number], c: readonly [number, number, number]): [number, number, number] {
  return [
    Math.min(a[0], b[0], c[0]),
    Math.min(a[1], b[1], c[1]),
    Math.min(a[2], b[2], c[2]),
  ];
}

function max3(a: readonly [number, number, number], b: readonly [number, number, number], c: readonly [number, number, number]): [number, number, number] {
  return [
    Math.max(a[0], b[0], c[0]),
    Math.max(a[1], b[1], c[1]),
    Math.max(a[2], b[2], c[2]),
  ];
}

function triCentroid(min: readonly [number, number, number], max: readonly [number, number, number]): [number, number, number] {
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
  readonly bvhNodes: Float32Array;
  readonly reorderedIndices: Uint32Array;
  readonly reorderedTriMaterialIds: Uint32Array;
}

export function buildCpuBvh(
  positions: Float32Array,
  indices: Uint32Array,
  triMaterialIds: Uint32Array,
): CpuBvhBuildResult {
  const triCount = Math.floor(indices.length / 4);
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
    const i0 = indices[t * 4] ?? 0;
    const i1 = indices[t * 4 + 1] ?? 0;
    const i2 = indices[t * 4 + 2] ?? 0;
    const a = getPosition(positions, i0);
    const b = getPosition(positions, i1);
    const c = getPosition(positions, i2);
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

  /**
   * Recursively build a BVH subtree for `subset` starting at `nodes[nodes.length]`.
   *
   * Algorithm: binned SAH per Wald 2007 §4.
   *  1. Compute centroid AABB.
   *  2. For each axis, bin centroids into NUM_BINS uniform buckets.
   *  3. Sweep left→right and right→left to compute prefix surface areas + counts.
   *  4. Evaluate SAH cost at each of the NUM_BINS−1 split planes.
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
    if (subset.length <= MAX_LEAF_TRIANGLES) {
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
      const bins: BinData[] = Array.from({ length: NUM_BINS }, makeEmptyBin);
      for (const r of subset) {
        const t = (r.centroid[axis]! - cMin[axis]!) / span;
        const binIdx = Math.min(NUM_BINS - 1, Math.floor(t * NUM_BINS));
        growBin(bins[binIdx]!, r);
      }

      // Prefix left sweep: prefixMin/Max[i] = merged AABB of bins [0..i].
      const prefixMinX = new Float32Array(NUM_BINS);
      const prefixMinY = new Float32Array(NUM_BINS);
      const prefixMinZ = new Float32Array(NUM_BINS);
      const prefixMaxX = new Float32Array(NUM_BINS);
      const prefixMaxY = new Float32Array(NUM_BINS);
      const prefixMaxZ = new Float32Array(NUM_BINS);
      const prefixCount = new Int32Array(NUM_BINS);

      {
        let bminX = Infinity, bminY = Infinity, bminZ = Infinity;
        let bmaxX = -Infinity, bmaxY = -Infinity, bmaxZ = -Infinity;
        let cnt = 0;
        for (let i = 0; i < NUM_BINS; i++) {
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

      // Suffix right sweep: suffixMin/Max[i] = merged AABB of bins [i..NUM_BINS-1].
      const suffixMinX = new Float32Array(NUM_BINS);
      const suffixMinY = new Float32Array(NUM_BINS);
      const suffixMinZ = new Float32Array(NUM_BINS);
      const suffixMaxX = new Float32Array(NUM_BINS);
      const suffixMaxY = new Float32Array(NUM_BINS);
      const suffixMaxZ = new Float32Array(NUM_BINS);
      const suffixCount = new Int32Array(NUM_BINS);

      {
        let bminX = Infinity, bminY = Infinity, bminZ = Infinity;
        let bmaxX = -Infinity, bmaxY = -Infinity, bmaxZ = -Infinity;
        let cnt = 0;
        for (let i = NUM_BINS - 1; i >= 0; i--) {
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

      // Evaluate NUM_BINS−1 split planes (split after bin i → left=[0..i], right=[i+1..K-1]).
      for (let split = 0; split < NUM_BINS - 1; split++) {
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
      const binIdx = Math.min(NUM_BINS - 1, Math.floor(t * NUM_BINS));
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
  // Gated behind NODE_ENV so it is zero-cost in production bundles. Accessed via
  // `globalThis` so consumers without `@types/node` typecheck cleanly (the check
  // is dev-only and still skips in any environment lacking `process`).
  const globalProc = (globalThis as { process?: { env?: Record<string, string | undefined> } }).process;
  if (globalProc !== undefined && globalProc.env?.['NODE_ENV'] !== 'production') {
    const LEAFNODE_CHECK = 0xffff;
    const n = nodes.length;
    for (let i = 0; i < n; i++) {
      const node = nodes[i];
      if (node == null) continue;
      const isLeaf = (node.splitAxisOrTriCount >>> 16) === LEAFNODE_CHECK;
      if (isLeaf) continue;
      const offset = node.rightChildOrTriOffset;
      if (offset < 1 || offset >= n) {
        throw new Error(
          `[pt-webgpu/buildCpuBvh] Interior node ${i} has invalid relative right-child ` +
            `offset ${offset} (must be in [1, ${n - 1}]). Build logic is broken.`,
        );
      }
    }
  }

  const reorderedIndices = new Uint32Array(indices.length);
  const reorderedTriMaterialIds = new Uint32Array(triMaterialIds.length);
  for (let newTri = 0; newTri < orderedTriangles.length; newTri += 1) {
    const oldTri = orderedTriangles[newTri] ?? 0;
    reorderedIndices[newTri * 4] = indices[oldTri * 4] ?? 0;
    reorderedIndices[newTri * 4 + 1] = indices[oldTri * 4 + 1] ?? 0;
    reorderedIndices[newTri * 4 + 2] = indices[oldTri * 4 + 2] ?? 0;
    reorderedIndices[newTri * 4 + 3] = 0;
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
