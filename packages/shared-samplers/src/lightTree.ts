/**
 * lightTree.ts — CPU-side binary light tree with SAH-variant splitting.
 *
 * Builds a power-weighted binary tree over an emitter list. The tree is
 * used by the PT fork's GPU binary-search traversal to importance-sample
 * direct lighting with spatial proximity weighting.
 *
 * Algorithm: surface-area heuristic (SAH) variant for emitter splitting —
 * partition the emitter list along the longest axis of their collective
 * bounding AABB at each step. This is a classic approach (Shirley 1996,
 * "Monte Carlo Techniques for Direct Lighting Calculations") adapted with
 * power-as-cost in place of triangle count.
 *
 * Each internal node stores the union AABB and summed power of its subtree.
 * The GPU traversal descends probabilistically, choosing a child proportional
 * to its power, then applies a proximity correction once a leaf is reached.
 *
 * GPU texture layout: 10 floats per node, uploaded as RGBA32F texture
 * (4 components per texel × 3 texels, padded to 12 floats for alignment).
 * See `packLightTreeForGPU` for the exact layout.
 *
 * NOTE: The split policy is Shirley 1996 median-split with power-as-cost.
 * The *traversal* (`sampleLightTreeCPU` / `lightTreePdfCPU` below, mirrored by
 * the WGSL `sampleLightTree` in walkaround-hybrid) is the spatially-aware
 * descent: at each internal node a child is chosen with probability
 * proportional to its **importance** `power / max(dist²(x, childAABB), floor)`
 * for the shading point `x`. This is the distance-weighted importance metric of
 * Estévez & Kulla 2018 (without the orientation-cone term, which `buildLightTree`
 * does not yet store). The selection pmf of a leaf is the product of the branch
 * probabilities along the root→leaf path; it sums to 1 over the leaves (a proper
 * probability tree) so it is an unbiased light-selection pdf for RIS/MIS.
 *
 * Future SOTA upgrade to full Estévez-Kulla 2018 would add: (1) per-node
 * orientation cones; (2) SAH-like split with receiver-aware importance; (3)
 * adaptive (non-median) split. Tracked separately.
 *
 * References:
 *   - Shirley, Smits, Wang, Zimmerman 1996, "Monte Carlo Techniques for
 *     Direct Lighting Calculations", ACM TOG (median split, power-as-cost).
 *   - Estévez & Kulla 2018, "Importance Sampling of Many Lights with Adaptive
 *     Tree Splitting", Proc. ACM CGIT (distance-weighted importance descent).
 */

// ────────────────────────────────────────────────────────────────────────────
// Public types
// ────────────────────────────────────────────────────────────────────────────

export interface LightTreeNode {
  /** Index into emitter list (-1 for internal nodes) */
  readonly emitterIndex: number;
  /** Total power (sum of leaf powers in this subtree) */
  readonly totalPower: number;
  /** AABB min for spatial proximity heuristic */
  readonly aabbMin: readonly [number, number, number];
  /** AABB max */
  readonly aabbMax: readonly [number, number, number];
  /** Left child index (-1 if leaf) */
  readonly leftChild: number;
  /** Right child index (-1 if leaf) */
  readonly rightChild: number;
}

export interface LightTreeBuildInput {
  /** Per-emitter luminous power (output of Sprint 2 cellPower computation) */
  readonly powers: ReadonlyArray<number>;
  /** Per-emitter centroid in world space (for spatial split heuristic) */
  readonly centroids: ReadonlyArray<readonly [number, number, number]>;
  /** Per-emitter AABB */
  readonly aabbs: ReadonlyArray<{
    min: readonly [number, number, number];
    max: readonly [number, number, number];
  }>;
}

// ────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ────────────────────────────────────────────────────────────────────────────

interface BuildItem {
  emitterIndex: number;
  power: number;
  centroid: readonly [number, number, number];
  aabbMin: readonly [number, number, number];
  aabbMax: readonly [number, number, number];
}

/**
 * Compute the union AABB of a slice of build items.
 * Returns [min3, max3] world-space corners.
 */
function unionAabb(
  items: BuildItem[],
): { min: [number, number, number]; max: [number, number, number] } {
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (const item of items) {
    minX = Math.min(minX, item.aabbMin[0]);
    minY = Math.min(minY, item.aabbMin[1]);
    minZ = Math.min(minZ, item.aabbMin[2]);
    maxX = Math.max(maxX, item.aabbMax[0]);
    maxY = Math.max(maxY, item.aabbMax[1]);
    maxZ = Math.max(maxZ, item.aabbMax[2]);
  }
  return {
    min: [minX, minY, minZ],
    max: [maxX, maxY, maxZ],
  };
}

/**
 * Sum total power of an item slice.
 */
function sumPower(items: BuildItem[]): number {
  let total = 0;
  for (const item of items) total += item.power;
  return total;
}

/**
 * Recursive SAH-variant light tree builder.
 *
 * Splits along the longest AABB axis of the centroid cloud at each level.
 * Leaf: single emitter. Internal: left + right subtrees.
 *
 * @param items - the emitter set for this subtree
 * @param nodes - output flat node array (accumulated in pre-order)
 * @returns the index of the node just pushed into `nodes`
 */
function buildSubtree(items: BuildItem[], nodes: LightTreeNode[]): number {
  const nodeIndex = nodes.length;

  if (items.length === 1) {
    // Leaf node
    const item = items[0]!;
    nodes.push({
      emitterIndex: item.emitterIndex,
      totalPower: item.power,
      aabbMin: item.aabbMin,
      aabbMax: item.aabbMax,
      leftChild: -1,
      rightChild: -1,
    });
    return nodeIndex;
  }

  // Compute AABB of centroids to find longest axis
  let cMinX = Infinity, cMinY = Infinity, cMinZ = Infinity;
  let cMaxX = -Infinity, cMaxY = -Infinity, cMaxZ = -Infinity;
  for (const item of items) {
    cMinX = Math.min(cMinX, item.centroid[0]);
    cMinY = Math.min(cMinY, item.centroid[1]);
    cMinZ = Math.min(cMinZ, item.centroid[2]);
    cMaxX = Math.max(cMaxX, item.centroid[0]);
    cMaxY = Math.max(cMaxY, item.centroid[1]);
    cMaxZ = Math.max(cMaxZ, item.centroid[2]);
  }
  const spanX = cMaxX - cMinX;
  const spanY = cMaxY - cMinY;
  const spanZ = cMaxZ - cMinZ;

  let leftItems: BuildItem[];
  let rightItems: BuildItem[];

  if (Math.max(spanX, spanY, spanZ) < 1e-4) {
    // Degenerate case: all emitter centroids are co-located (e.g. a single stained-glass
    // panel with all triangles modelled at one point). Centroid-axis sorting is meaningless
    // here — fall back to a power-based median split so that the tree remains balanced and
    // descends correctly by power weighting. The spatial heuristic will be ineffective for
    // this subtree (no spatial separation), but the GPU traversal degrades gracefully to
    // pure power-weighted sampling.
    const sorted = items.slice().sort((a, b) => b.power - a.power); // descending power
    const mid = Math.floor(sorted.length / 2);
    leftItems = sorted.slice(0, mid);
    rightItems = sorted.slice(mid);
  } else {
    const axis = spanX >= spanY && spanX >= spanZ ? 0 : spanY >= spanZ ? 1 : 2;

    // Sort along longest axis
    const sorted = items.slice().sort((a, b) => a.centroid[axis] - b.centroid[axis]);

    // Split at median (equal partition — sufficient for typical emitter counts <100)
    const mid = Math.floor(sorted.length / 2);
    leftItems = sorted.slice(0, mid);
    rightItems = sorted.slice(mid);
  }

  // Compute union AABB of this node's entire item set
  const { min: aabbMin, max: aabbMax } = unionAabb(items);
  const totalPower = sumPower(items);

  // Push a placeholder — we need the node index before building children
  nodes.push({
    emitterIndex: -1,
    totalPower,
    aabbMin,
    aabbMax,
    leftChild: -1, // patched below
    rightChild: -1, // patched below
  });

  const leftChildIdx = buildSubtree(leftItems, nodes);
  const rightChildIdx = buildSubtree(rightItems, nodes);

  // Patch the internal node now that we know child indices
  // (nodes are readonly, so we replace the entry at nodeIndex)
  nodes[nodeIndex] = {
    emitterIndex: -1,
    totalPower,
    aabbMin,
    aabbMax,
    leftChild: leftChildIdx,
    rightChild: rightChildIdx,
  };

  return nodeIndex;
}

// ────────────────────────────────────────────────────────────────────────────
// Public API
// ────────────────────────────────────────────────────────────────────────────

/**
 * Build a binary light tree where each internal node sums child powers.
 *
 * GPU traversal: descends toward the heavier-power child with probability
 * proportional to child power, then corrects for spatial proximity at the
 * leaf. The GPU consumes the packed node array directly via `packLightTreeForGPU`.
 *
 * `_powerPrefixSumDebug` layout: length = nodeCount (one entry per node, pre-order).
 * Each entry is the running prefix sum of `totalPower` values across the
 * pre-order node array, normalised by `root.totalPower`. Because internal nodes
 * aggregate subtree power, their contribution is counted once in the running sum
 * before each leaf's power is also counted — so entries routinely exceed 1.0 for
 * trees with more than one leaf. This is NOT a true probability CDF (which would
 * be normalised to [0, 1] and built only over leaves). It is an unnormalised
 * prefix-sum provided for CPU-side structural verification and monotonicity checks.
 * The GPU does its own binary descent from the root and does not consume this array.
 *
 * @throws if powers/centroids/aabbs arrays have mismatched lengths
 * @throws if any array is empty
 */
export function buildLightTree(input: LightTreeBuildInput): {
  nodes: LightTreeNode[];
  /**
   * @internal
   *
   * **WARNING: Do NOT use for sampling — values exceed 1.0 because internal
   * nodes are counted before children. Use leaf-only power traversal on
   * `nodes` instead.**
   *
   * Unnormalised node-power prefix-sum for CPU-side structural verification
   * only. Length = nodeCount (pre-order). Values can exceed 1.0 because
   * internal nodes aggregate subtree power, so their power is counted before
   * each child's power is also counted. This is NOT a true CDF.
   */
  _powerPrefixSumDebug: Float32Array;
} {
  const { powers, centroids, aabbs } = input;
  const n = powers.length;
  if (n === 0) throw new Error('buildLightTree: at least one emitter required');
  if (centroids.length !== n || aabbs.length !== n) {
    throw new Error('buildLightTree: powers/centroids/aabbs length mismatch');
  }

  // Build item list
  const items: BuildItem[] = [];
  for (let i = 0; i < n; i++) {
    const power = powers[i]!;
    const centroid = centroids[i]!;
    const aabb = aabbs[i]!;
    items.push({
      emitterIndex: i,
      power,
      centroid,
      aabbMin: aabb.min,
      aabbMax: aabb.max,
    });
  }

  const nodes: LightTreeNode[] = [];
  buildSubtree(items, nodes);

  // Compute unnormalised node-power prefix-sum over pre-order node array.
  // Each entry is the running sum of totalPower values normalised by root.totalPower.
  // Because internal nodes aggregate subtree power, values exceed 1.0 for trees
  // with more than one leaf — this is intentional (see JSDoc above).
  // For sampling, use leaf-only power traversal on `nodes` instead.
  const rootPower = nodes[0]!.totalPower;
  const _powerPrefixSumDebug = new Float32Array(nodes.length);
  let running = 0;
  for (let i = 0; i < nodes.length; i++) {
    running += nodes[i]!.totalPower;
    _powerPrefixSumDebug[i] = rootPower > 0 ? running / rootPower : 0;
  }

  return { nodes, _powerPrefixSumDebug };
}

/**
 * Pack the node array into a Float32Array suitable for GPU texture upload.
 *
 * Layout per node (10 logical floats, padded to 12 for RGBA32F 3-texel alignment):
 *   [0]  emitterIndex (as float; -1.0 for internal)
 *   [1]  totalPower
 *   [2]  leftChild (as float; -1.0 for leaf)
 *   [3]  rightChild (as float; -1.0 for leaf)
 *   [4]  aabbMin.x
 *   [5]  aabbMin.y
 *   [6]  aabbMin.z
 *   [7]  aabbMax.x
 *   [8]  aabbMax.y
 *   [9]  aabbMax.z
 *   [10] padding (0)
 *   [11] padding (0)
 *
 * 3 texels × 4 components = 12 floats per node. This aligns to RGBA32F
 * texture uploads where the texture width = nodeCount and height = 3.
 * On the GPU, a given node at index `i` reads:
 *   texelFetch(lightTree, ivec2(i, 0), 0)  → [emitterIdx, power, leftChild, rightChild]
 *   texelFetch(lightTree, ivec2(i, 1), 0)  → [aabbMin.xyz, aabbMax.x]
 *   texelFetch(lightTree, ivec2(i, 2), 0)  → [aabbMax.yz, pad, pad]
 */
export function packLightTreeForGPU(nodes: ReadonlyArray<LightTreeNode>): Float32Array {
  const FLOATS_PER_NODE = LIGHT_TREE_FLOATS_PER_NODE;
  const out = new Float32Array(nodes.length * FLOATS_PER_NODE);
  for (let i = 0; i < nodes.length; i++) {
    const node = nodes[i]!;
    const base = i * FLOATS_PER_NODE;
    out[base + 0] = node.emitterIndex;
    out[base + 1] = node.totalPower;
    out[base + 2] = node.leftChild;
    out[base + 3] = node.rightChild;
    out[base + 4] = node.aabbMin[0];
    out[base + 5] = node.aabbMin[1];
    out[base + 6] = node.aabbMin[2];
    out[base + 7] = node.aabbMax[0];
    out[base + 8] = node.aabbMax[1];
    out[base + 9] = node.aabbMax[2];
    out[base + 10] = 0; // padding
    out[base + 11] = 0; // padding
  }
  return out;
}

/**
 * 12 floats per node in the packed flat layout consumed by `packLightTreeForGPU`
 * and the WGSL `sampleLightTree` traversal. The WGSL side reads the same stride
 * from a flat `array<f32>` storage buffer (NOT a texture — the walkaround ReSTIR
 * path is compute-only and consumes the tree as a storage buffer); see
 * `walkaround-hybrid/src/shaders/lightTree.wgsl.ts`.
 */
export const LIGHT_TREE_FLOATS_PER_NODE = 12;

// ────────────────────────────────────────────────────────────────────────────
// Spatially-aware traversal (CPU reference; mirrored 1:1 by the WGSL kernel)
// ────────────────────────────────────────────────────────────────────────────

/**
 * Squared distance from a point to an axis-aligned bounding box. Zero when the
 * point is inside the box. Matches the WGSL `lt_dist2ToAabb` exactly.
 */
function dist2ToAabb(
  px: number, py: number, pz: number,
  min: readonly [number, number, number],
  max: readonly [number, number, number],
): number {
  const dx = Math.max(min[0] - px, 0, px - max[0]);
  const dy = Math.max(min[1] - py, 0, py - max[1]);
  const dz = Math.max(min[2] - pz, 0, pz - max[2]);
  return dx * dx + dy * dy + dz * dz;
}

/**
 * Importance of a node for a shading point `x`: `power / max(dist², floor)`.
 * The `dist2Floor` clamp prevents a divide-by-zero / unbounded importance when
 * the shading point lies inside (or on) the node AABB. It is the SAME floor the
 * RIS geometry term uses (`ubo.emitterDist2Floor`) so near-light behaviour stays
 * consistent between selection and evaluation. Matches WGSL `lt_importance`.
 */
function nodeImportance(
  node: LightTreeNode,
  px: number, py: number, pz: number,
  dist2Floor: number,
): number {
  if (node.totalPower <= 0) return 0;
  const d2 = Math.max(dist2ToAabb(px, py, pz, node.aabbMin, node.aabbMax), dist2Floor);
  return node.totalPower / d2;
}

/**
 * Importance-sample a single emitter (leaf) from the light tree for a shading
 * point `x`, returning the chosen `emitterIndex` and the **selection pdf** — the
 * probability that this descent reaches that leaf. The pdf is the product of the
 * per-internal-node branch probabilities along the root→leaf path; it forms a
 * proper pmf over leaves (sums to 1), which is exactly what the RIS source-pdf
 * `w = p̂ / p_source` requires for an unbiased estimator.
 *
 * `rand01` is a 0-arg sampler returning a fresh uniform in [0, 1) per call (one
 * draw per internal node descended). The WGSL kernel draws from the PCG RNG.
 *
 * Degenerate cases (both children zero-importance) fall back to a 50/50 split so
 * the descent always terminates at a leaf with a well-defined, strictly-positive
 * pdf — never returning pdf 0 for a reachable leaf (which would create an
 * infinite RIS weight).
 *
 * Mirrors the WGSL `sampleLightTree` byte-for-byte in branch logic.
 */
export function sampleLightTreeCPU(
  nodes: ReadonlyArray<LightTreeNode>,
  x: readonly [number, number, number],
  dist2Floor: number,
  rand01: () => number,
): { emitterIndex: number; pdf: number } {
  if (nodes.length === 0) return { emitterIndex: -1, pdf: 0 };
  const [px, py, pz] = x;
  let nodeIdx = 0;
  let pdf = 1.0;
  // Bounded descent: a binary tree over N leaves has depth ≤ N; the explicit
  // cap mirrors the WGSL loop bound (a while-true is illegal there).
  for (let guard = 0; guard < nodes.length + 1; guard++) {
    const node = nodes[nodeIdx]!;
    if (node.leftChild < 0 || node.rightChild < 0) {
      // Leaf.
      return { emitterIndex: node.emitterIndex, pdf };
    }
    const left = nodes[node.leftChild]!;
    const right = nodes[node.rightChild]!;
    const impL = nodeImportance(left, px, py, pz, dist2Floor);
    const impR = nodeImportance(right, px, py, pz, dist2Floor);
    const sum = impL + impR;
    // Degenerate: both subtrees contribute zero importance (e.g. all-zero
    // power under this node). Fall back to a uniform 50/50 split so the
    // descent terminates with a positive pdf.
    const pL = sum > 0 ? impL / sum : 0.5;
    if (rand01() < pL) {
      pdf *= pL;
      nodeIdx = node.leftChild;
    } else {
      pdf *= 1.0 - pL;
      nodeIdx = node.rightChild;
    }
  }
  // Unreachable for a well-formed tree; return the current node as a leaf guard.
  const node = nodes[nodeIdx]!;
  return { emitterIndex: node.emitterIndex, pdf };
}

/**
 * Recompute the selection pdf the tree assigns to a given `emitterIndex` for a
 * shading point `x`, WITHOUT drawing random numbers — it walks the unique
 * root→leaf path to that emitter, multiplying the branch probabilities. This is
 * the deterministic inverse of `sampleLightTreeCPU`'s pdf and is used (a) in
 * tests to assert the pmf integrates to 1 over the emitter set, and (b) anywhere
 * a light's selection probability is needed independently of the sampling draw
 * (e.g. MIS with a BRDF strategy).
 *
 * Returns 0 if the emitter is not present in the tree.
 */
export function lightTreePdfCPU(
  nodes: ReadonlyArray<LightTreeNode>,
  x: readonly [number, number, number],
  dist2Floor: number,
  emitterIndex: number,
): number {
  if (nodes.length === 0) return 0;
  const [px, py, pz] = x;
  // Precompute, per node, whether the target emitter lies in its subtree.
  // (Leaf emitterIndex match, or one of its children's subtrees contains it.)
  const contains = new Array<boolean>(nodes.length).fill(false);
  // Pre-order array ⇒ a child always has a higher index than its parent, so a
  // reverse pass propagates "contains target" up to the root in one sweep.
  for (let i = nodes.length - 1; i >= 0; i--) {
    const n = nodes[i]!;
    if (n.emitterIndex === emitterIndex) {
      contains[i] = true;
    } else if (n.leftChild >= 0 && n.rightChild >= 0) {
      contains[i] = contains[n.leftChild]! || contains[n.rightChild]!;
    }
  }
  if (!contains[0]) return 0;

  let nodeIdx = 0;
  let pdf = 1.0;
  for (let guard = 0; guard < nodes.length + 1; guard++) {
    const node = nodes[nodeIdx]!;
    if (node.leftChild < 0 || node.rightChild < 0) {
      return node.emitterIndex === emitterIndex ? pdf : 0;
    }
    const left = nodes[node.leftChild]!;
    const right = nodes[node.rightChild]!;
    const impL = nodeImportance(left, px, py, pz, dist2Floor);
    const impR = nodeImportance(right, px, py, pz, dist2Floor);
    const sum = impL + impR;
    const pL = sum > 0 ? impL / sum : 0.5;
    if (contains[node.leftChild]) {
      pdf *= pL;
      nodeIdx = node.leftChild;
    } else {
      pdf *= 1.0 - pL;
      nodeIdx = node.rightChild;
    }
  }
  return 0;
}

// ────────────────────────────────────────────────────────────────────────────
// ReGIR — Reservoir-based Grid Importance Resampling (CPU reference core)
//
// ReGIR (Boksansky, Wyman, Benty 2021, "Rendering Many Lights with Grid-Based
// Reservoirs", Ray Tracing Gems II ch. 23) decouples the per-pixel light-
// selection cost from the light count by pre-resampling lights into a world-
// space grid of reservoirs ONCE per frame. ReSTIR-DI then draws its initial
// candidates from the grid cell containing the shading point instead of
// traversing the light tree per pixel.
//
// This module is the CPU reference for the per-cell WRS that the WGSL grid-
// build kernel mirrors 1:1. It is correctness-critical: the per-cell selection
// pmf computed here is the EXACT pmf the RIS source weight `w = p̂ / p_source`
// divides by, so it MUST be a valid pmf (sum to 1 over the emitter set in
// expectation) or the ReSTIR estimator is biased.
//
// Unbiasedness construction (matches the light-tree discipline already in this
// file):
//   - The cell target is `q̂_c(e) = power_e / max(dist²(x_c, e.aabb), floor)`
//     for the cell centroid `x_c` — power × spatial proximity, NO BRDF (the
//     per-pixel receiver BRDF is unknown at grid-build time; the BRDF enters
//     later, in the RIS p̂). `q̂_c` is exactly the *leaf importance* the light
//     tree already uses, so the grid is "seeded by the light tree" in the sense
//     that the per-cell candidates are drawn via `sampleLightTreeCPU` at `x_c`
//     and weighted to the same target the tree's descent approximates.
//   - One sub-reservoir runs WRS over `M` tree draws: candidate `e_i` is drawn
//     with source pdf `p_tree(e_i | x_c)`, RIS weight `w_i = q̂_c(e_i) /
//     p_tree(e_i | x_c)`. The survivor `e*` is (in expectation) distributed
//     ∝ `q̂_c`. The reservoir's running `wSum / M` is an unbiased estimate
//     `Ŝ` of the cell's total target mass `S_c = Σ_e q̂_c(e)`.
//   - The survivor's EFFECTIVE selection pmf is therefore
//         pSel(e*) = q̂_c(e*) / Ŝ  =  q̂_c(e*) · M / wSum,
//     the standard RIS relation (Bitterli 2020 §3): a WRS reservoir is an
//     importance sampler whose effective pdf is `target / normalisation-estimate`.
//     Σ_e pSel(e) → 1 in expectation (a valid pmf), which the tests assert.
//   - A cell stores `K` independent sub-reservoirs. RIS picks one uniformly and
//     uses ITS `pSel` as the source pmf; the `1/K` does not enter the weight
//     because RIS draws ONE candidate (it is not summing over the K reservoirs).
//     The K survivors give per-pixel candidate diversity without re-running the
//     tree descent per pixel.
// ────────────────────────────────────────────────────────────────────────────

/** Floats per ReGIR cell-reservoir survivor slot in the packed grid buffer:
 *  [0] emitterIndex (as f32; -1 ⇒ empty slot), [1] pSel (effective selection
 *  pmf of that emitter from this cell). The grid buffer is a flat `array<f32>`
 *  of `numCells × REGIR_SURVIVORS_PER_CELL × REGIR_FLOATS_PER_SURVIVOR`. */
export const REGIR_FLOATS_PER_SURVIVOR = 2;

/** A single ReGIR cell-reservoir survivor: a chosen emitter + its effective
 *  per-cell selection pmf (q̂_c(e*) / Ŝ). `pSel <= 0` marks an empty slot. */
export interface ReGIRSurvivor {
  readonly emitterIndex: number;
  /** Effective selection pmf of `emitterIndex` from this cell's reservoir. */
  readonly pSel: number;
}

/**
 * Run ONE per-cell WRS sub-reservoir over the light tree, seeded at the cell
 * centroid `x_c`, and return the survivor + its effective selection pmf.
 *
 * Mirrors the WGSL `regir_build_survivor` byte-for-byte. The `q̂_c` target is
 * `leafImportance(e) = power_e / max(dist²(x_c, e.aabb), floor)` — recovered
 * from the tree leaf for the chosen emitter — and the source pdf is the tree's
 * own `p_tree(e | x_c)`. The returned `pSel = q̂_c(e*) · M / wSum` is the
 * unbiased effective pmf (see module header).
 *
 * Degenerate cells (no positive-power emitter reachable, or `wSum == 0`) return
 * `{ emitterIndex: -1, pSel: 0 }` — an empty slot RIS skips (never an infinite
 * weight). `M` is the candidate count per sub-reservoir.
 *
 * @param leafImportanceOf - maps emitterIndex → q̂_c for this cell. Provided by
 *   the caller so the CPU port and the WGSL kernel share the SAME target
 *   (the WGSL recomputes it from the emitter list; the CPU port can pass a
 *   tree-leaf lookup). MUST equal the per-leaf `power / max(dist², floor)`.
 */
export function regirBuildSurvivorCPU(
  nodes: ReadonlyArray<LightTreeNode>,
  xc: readonly [number, number, number],
  dist2Floor: number,
  M: number,
  leafImportanceOf: (emitterIndex: number) => number,
  rand01: () => number,
): ReGIRSurvivor {
  let wSum = 0;
  let chosen = -1;
  let chosenQHat = 0;
  for (let i = 0; i < M; i++) {
    const draw = sampleLightTreeCPU(nodes, xc, dist2Floor, rand01);
    if (draw.emitterIndex < 0 || draw.pdf <= 0) continue;
    const qHat = leafImportanceOf(draw.emitterIndex);
    if (qHat <= 0) continue;
    // RIS source weight: target / source-pdf. Source is the tree's own
    // selection pmf at the cell centroid.
    const w = qHat / draw.pdf;
    wSum += w;
    // WRS: accept with probability w / wSum.
    if (rand01() * wSum < w) {
      chosen = draw.emitterIndex;
      chosenQHat = qHat;
    }
  }
  if (chosen < 0 || wSum <= 0) return { emitterIndex: -1, pSel: 0 };
  // Effective selection pmf = q̂(e*) / Ŝ, Ŝ = wSum / M (unbiased S_c estimate).
  const pSel = (chosenQHat * M) / wSum;
  return { emitterIndex: chosen, pSel };
}

/**
 * Deterministic cell target q̂_c(e) = power_e / max(dist²(x_c, e.aabb), floor)
 * for the leaf carrying `emitterIndex`. This is the SAME importance metric
 * `nodeImportance` applies to a leaf node, so the ReGIR cell target and the
 * light-tree descent agree exactly. Returns 0 if the emitter is not a leaf in
 * the tree.
 *
 * The WGSL kernel recomputes q̂_c directly from the emitter list
 * (`luminance(Le)·area / max(dist², floor)`); this CPU helper recovers it from
 * the tree leaves so the reference port needs only the tree + centroid.
 */
export function regirCellTargetFromTree(
  nodes: ReadonlyArray<LightTreeNode>,
  xc: readonly [number, number, number],
  dist2Floor: number,
): (emitterIndex: number) => number {
  // Map emitterIndex → leaf node for O(1) lookup.
  const leafByEmitter = new Map<number, LightTreeNode>();
  for (const n of nodes) {
    if (n.leftChild < 0 && n.rightChild < 0 && n.emitterIndex >= 0) {
      leafByEmitter.set(n.emitterIndex, n);
    }
  }
  const [px, py, pz] = xc;
  return (emitterIndex: number): number => {
    const leaf = leafByEmitter.get(emitterIndex);
    if (!leaf) return 0;
    return nodeImportance(leaf, px, py, pz, dist2Floor);
  };
}

/**
 * The EXACT normalized cell pmf `q̂_c(e) / S_c`, `S_c = Σ_e q̂_c(e)`, over all
 * emitters (tree leaves). This is the limiting distribution the per-cell WRS
 * (`regirBuildSurvivorCPU`) estimates: as `M → ∞` the survivor distribution
 * converges to this pmf, and the stored `pSel` is an unbiased estimate of the
 * per-emitter value `q̂_c(e)/S_c` evaluated at the survivor.
 *
 * Returns a `Map<emitterIndex, pmf>` that sums to 1 (a valid pmf), which is the
 * correctness invariant the ReGIR tests assert. Used by tests + as the
 * reference for the "concentrates on locally-important lights" assertion.
 */
export function regirCellPmfExact(
  nodes: ReadonlyArray<LightTreeNode>,
  xc: readonly [number, number, number],
  dist2Floor: number,
): Map<number, number> {
  const target = regirCellTargetFromTree(nodes, xc, dist2Floor);
  const emitters: number[] = [];
  for (const n of nodes) {
    if (n.leftChild < 0 && n.rightChild < 0 && n.emitterIndex >= 0) {
      emitters.push(n.emitterIndex);
    }
  }
  let S = 0;
  const qHat = new Map<number, number>();
  for (const e of emitters) {
    const q = target(e);
    qHat.set(e, q);
    S += q;
  }
  const pmf = new Map<number, number>();
  for (const e of emitters) {
    pmf.set(e, S > 0 ? qHat.get(e)! / S : 0);
  }
  return pmf;
}
