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

/**
 * B8 — Conty-Estévez orientation cone for a tree node.
 *
 * `axis` is the (unit) average emission direction of all emitters in the subtree.
 * `thetaO` is the half-angle that bounds the spread of the constituent emitters'
 * emission axes (the "normal cone"); `thetaE` is the additional half-angle of each
 * emitter's own emission lobe beyond its axis (a cosine-lobe area light emits over
 * a hemisphere ⇒ `thetaE = π/2`; an isotropic point light emits over the full
 * sphere). The pair `(thetaO, thetaE)` lets the importance function cull a node
 * whose emitters cannot illuminate the shading point (it lies outside the union of
 * their emission cones), per Conty Estévez & Kulla 2018 §5.
 *
 * The "full sphere" sentinel (`thetaO = π`, `thetaE = π`, `axis = (0,0,0)`) means
 * "no orientation information / emits everywhere" — the cone importance term is
 * then identically 1, recovering the pre-B8 spatial-only behaviour byte-for-byte.
 */
export interface OrientationCone {
  /** Unit average emission axis (or (0,0,0) for an unoriented / full-sphere node). */
  readonly axis: readonly [number, number, number];
  /** Half-angle bounding the spread of emitter axes in the subtree (radians). */
  readonly thetaO: number;
  /** Emission-lobe half-angle beyond the axis (radians; π/2 = hemisphere, π = sphere). */
  readonly thetaE: number;
}

/** Full-sphere cone: emits in every direction ⇒ cone importance term ≡ 1. */
const FULL_SPHERE_CONE: OrientationCone = { axis: [0, 0, 0], thetaO: Math.PI, thetaE: Math.PI };

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
  /**
   * B8 — merged orientation cone of this subtree (Conty-Estévez 2018). Defaults to
   * the full sphere when the build input omits per-emitter directions, so the cone
   * importance term is 1 and behaviour matches the pre-B8 spatial-only tree.
   */
  readonly cone: OrientationCone;
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
  /**
   * B8 (OPTIONAL) — per-emitter orientation cone. When omitted (or an entry is
   * omitted) the emitter is treated as a full-sphere emitter (no orientation
   * culling) — identical to the pre-B8 behaviour. Supply this for oriented
   * emitters (spotlights, single-sided area lights) so the importance function
   * can cull nodes whose emitters point away from the shading point.
   *
   * `axis` need not be normalised — the builder normalises it. `thetaE` defaults
   * to π/2 (a one-sided cosine lobe) when an entry is present but omits it;
   * `thetaO` defaults to 0 (a single sharp axis) for a leaf.
   */
  readonly cones?: ReadonlyArray<{
    readonly axis: readonly [number, number, number];
    readonly thetaO?: number;
    readonly thetaE?: number;
  } | undefined>;
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
  cone: OrientationCone;
}

function vlen(v: readonly [number, number, number]): number {
  return Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

/**
 * Merge two orientation cones into one that bounds both, per Conty Estévez & Kulla
 * 2018 §4.4 ("Combining lights into clusters"). The merged cone shares a single
 * axis (the wider cone's axis rotated toward the other so both are covered); its
 * `thetaO` is grown so both input cones' axes lie within it, and its `thetaE` is
 * the max of the two. A full-sphere input collapses the result to full-sphere.
 */
function mergeCones(a: OrientationCone, b: OrientationCone): OrientationCone {
  const la = vlen(a.axis);
  const lb = vlen(b.axis);
  // A degenerate (zero-axis) or already-full-sphere cone unions to full-sphere.
  if (la < 1e-8 || lb < 1e-8 || a.thetaO >= Math.PI || b.thetaO >= Math.PI) {
    const thetaE = Math.min(Math.PI, Math.max(a.thetaE, b.thetaE));
    return { axis: [0, 0, 0], thetaO: Math.PI, thetaE };
  }
  // Ensure `A` is the wider cone (larger thetaO) — EK keeps the wider as the base.
  let A = a;
  let B = b;
  if (B.thetaO > A.thetaO) { A = b; B = a; }
  const laA = vlen(A.axis);
  const laB = vlen(B.axis);
  const ax: readonly [number, number, number] = [A.axis[0] / laA, A.axis[1] / laA, A.axis[2] / laA];
  const bx: readonly [number, number, number] = [B.axis[0] / laB, B.axis[1] / laB, B.axis[2] / laB];
  const cosD = Math.max(-1, Math.min(1, ax[0] * bx[0] + ax[1] * bx[1] + ax[2] * bx[2]));
  const dTheta = Math.acos(cosD);
  const thetaE = Math.min(Math.PI, Math.max(A.thetaE, B.thetaE));
  // If B's cone is already enclosed by A's, A is the answer.
  if (Math.min(dTheta + B.thetaO, Math.PI) <= A.thetaO) {
    return { axis: ax, thetaO: A.thetaO, thetaE };
  }
  const thetaO = (A.thetaO + dTheta + B.thetaO) * 0.5;
  if (thetaO >= Math.PI) {
    return { axis: [0, 0, 0], thetaO: Math.PI, thetaE };
  }
  // New axis: rotate A's axis toward B's by (thetaO - A.thetaO) in the plane the
  // two axes span. Build that plane's tangent via Gram-Schmidt.
  const rot = thetaO - A.thetaO;
  let tx = bx[0] - cosD * ax[0];
  let ty = bx[1] - cosD * ax[1];
  let tz = bx[2] - cosD * ax[2];
  const tl = Math.sqrt(tx * tx + ty * ty + tz * tz);
  let axis: readonly [number, number, number];
  if (tl < 1e-8) {
    axis = ax; // axes parallel — no rotation needed
  } else {
    tx /= tl; ty /= tl; tz /= tl;
    const cr = Math.cos(rot);
    const sr = Math.sin(rot);
    axis = [ax[0] * cr + tx * sr, ax[1] * cr + ty * sr, ax[2] * cr + tz * sr];
  }
  return { axis, thetaO, thetaE };
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
      cone: item.cone,
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
    cone: FULL_SPHERE_CONE, // patched below
  });

  const leftChildIdx = buildSubtree(leftItems, nodes);
  const rightChildIdx = buildSubtree(rightItems, nodes);

  // B8 — merge the children's cones into this interior node's bounding cone.
  const mergedCone = mergeCones(nodes[leftChildIdx]!.cone, nodes[rightChildIdx]!.cone);

  // Patch the internal node now that we know child indices
  // (nodes are readonly, so we replace the entry at nodeIndex)
  nodes[nodeIndex] = {
    emitterIndex: -1,
    totalPower,
    aabbMin,
    aabbMax,
    leftChild: leftChildIdx,
    rightChild: rightChildIdx,
    cone: mergedCone,
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
  const { powers, centroids, aabbs, cones } = input;
  const n = powers.length;
  if (n === 0) throw new Error('buildLightTree: at least one emitter required');
  if (centroids.length !== n || aabbs.length !== n) {
    throw new Error('buildLightTree: powers/centroids/aabbs length mismatch');
  }
  if (cones != null && cones.length !== n) {
    throw new Error('buildLightTree: cones length mismatch (must equal powers.length when supplied)');
  }

  // Build item list
  const items: BuildItem[] = [];
  for (let i = 0; i < n; i++) {
    const power = powers[i]!;
    const centroid = centroids[i]!;
    const aabb = aabbs[i]!;
    // B8 — per-emitter orientation cone. Omitted ⇒ full sphere (no culling),
    // exactly the pre-B8 behaviour. A present entry defaults thetaE to π/2 (a
    // one-sided cosine emission lobe) and thetaO to 0 (a single sharp axis).
    const ci = cones?.[i];
    let cone: OrientationCone;
    if (ci == null || vlen(ci.axis) < 1e-8) {
      cone = FULL_SPHERE_CONE;
    } else {
      const l = vlen(ci.axis);
      cone = {
        axis: [ci.axis[0] / l, ci.axis[1] / l, ci.axis[2] / l],
        thetaO: ci.thetaO ?? 0,
        thetaE: Math.min(Math.PI, ci.thetaE ?? Math.PI / 2),
      };
    }
    items.push({
      emitterIndex: i,
      power,
      centroid,
      aabbMin: aabb.min,
      aabbMax: aabb.max,
      cone,
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
 * Pack the node array into a Float32Array suitable for GPU texture / storage
 * upload.
 *
 * Layout per node (15 logical floats, padded to 16 for RGBA32F 4-texel
 * alignment — B8 grew the stride from 12 to carry the orientation cone):
 *   [0]  emitterIndex (as float; -1.0 for internal)
 *   [1]  totalPower
 *   [2]  leftChild (as float; -1.0 for leaf)
 *   [3]  rightChild (as float; -1.0 for leaf)
 *   [4]  aabbMin.x        [5] aabbMin.y      [6] aabbMin.z
 *   [7]  aabbMax.x        [8] aabbMax.y      [9] aabbMax.z
 *   [10] cone.axis.x      [11] cone.axis.y   [12] cone.axis.z
 *   [13] cos(cone.thetaO) — cosine of the normal-cone half-angle
 *   [14] cos(min(π, thetaO + thetaE)) — cosine of the total emission half-angle
 *   [15] padding (0)
 *
 * Slots [13]/[14] store COSINES (not radians) so the GPU importance term avoids a
 * per-node `acos`. A full-sphere node has axis (0,0,0) and both cosines = −1
 * (cos π), which the importance term reads as "no orientation culling" ⇒ cone
 * factor 1, recovering the pre-B8 spatial-only descent byte-for-byte.
 *
 * 4 texels × 4 components = 16 floats per node. Aligns to RGBA32F uploads where
 * texture width = nodeCount and height = 4.
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
    out[base + 10] = node.cone.axis[0];
    out[base + 11] = node.cone.axis[1];
    out[base + 12] = node.cone.axis[2];
    out[base + 13] = Math.cos(Math.min(Math.PI, node.cone.thetaO));
    out[base + 14] = Math.cos(Math.min(Math.PI, node.cone.thetaO + node.cone.thetaE));
    out[base + 15] = 0; // padding
  }
  return out;
}

/**
 * 16 floats per node in the packed flat layout consumed by `packLightTreeForGPU`
 * and the WGSL `sampleLightTree` traversal (B8 grew this from 12 to carry the
 * orientation cone). The WGSL side reads the same stride from a flat
 * `array<f32>` storage buffer (NOT a texture — the walkaround ReSTIR path is
 * compute-only and consumes the tree as a storage buffer); see
 * `walkaround-hybrid/src/shaders/lightTree.wgsl.ts`.
 */
export const LIGHT_TREE_FLOATS_PER_NODE = 16;

// ────────────────────────────────────────────────────────────────────────────
// Spatially-aware traversal (CPU reference; mirrored 1:1 by the WGSL kernel)
// ────────────────────────────────────────────────────────────────────────────

/**
 * Squared distance from a point to an axis-aligned bounding box. Zero when the
 * point is inside the box. Matches the WGSL `lt_dist2ToAabb` exactly.
 */
export function dist2ToAabb(
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
 * Cone importance factor (B8): how much a node's emitters can illuminate the
 * shading point, given the merged orientation cone `(axis, thetaO, thetaE)`.
 *
 * Let `θ` be the angle between the emission axis and the direction from the node
 * (AABB centre) TO the point. The emitters can reach the point only if it lies
 * within the total emission cone of half-angle `thetaO + thetaE`. We return
 * `max(0, cos(max(0, θ − thetaO)))` clamped to 0 beyond `thetaO + thetaE`:
 *   - inside the normal cone (θ ≤ thetaO): factor 1;
 *   - in the lobe skirt (thetaO < θ ≤ thetaO+thetaE): a smooth cosine falloff;
 *   - outside (θ > thetaO+thetaE): 0 — the node is culled from selection.
 * This is the Conty-Estévez 2018 orientation term, in cosine space using the
 * packed `cosThetaO`/`cosThetaOE` so no `acos` is needed. A full-sphere node
 * (cosThetaO = cosThetaOE = −1, axis length 0) returns 1 identically.
 *
 * `cosThetaO` = cos(thetaO); `cosThetaOE` = cos(min(π, thetaO+thetaE)).
 */
function coneImportanceFactor(
  axis: readonly [number, number, number],
  cosThetaO: number,
  cosThetaOE: number,
  px: number, py: number, pz: number,
  cx: number, cy: number, cz: number,
): number {
  const al = axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2];
  if (al < 1e-12) return 1.0;             // full sphere / unoriented — no culling
  let dx = px - cx, dy = py - cy, dz = pz - cz;
  const dl = Math.sqrt(dx * dx + dy * dy + dz * dz);
  if (dl < 1e-12) return 1.0;             // point at the centre — cannot orient
  const inv = 1.0 / dl;
  dx *= inv; dy *= inv; dz *= inv;
  const aInv = 1.0 / Math.sqrt(al);
  const cosTheta = (axis[0] * aInv) * dx + (axis[1] * aInv) * dy + (axis[2] * aInv) * dz;
  if (cosTheta < cosThetaOE) return 0.0;  // outside the total emission cone — cull
  if (cosTheta >= cosThetaO) return 1.0;  // inside the normal cone — full factor
  // Lobe skirt: cos(θ − thetaO) = cosθ·cosθO + sinθ·sinθO.
  const sinTheta = Math.sqrt(Math.max(0, 1 - cosTheta * cosTheta));
  const sinThetaO = Math.sqrt(Math.max(0, 1 - cosThetaO * cosThetaO));
  return Math.max(0, cosTheta * cosThetaO + sinTheta * sinThetaO);
}

/**
 * Importance of a node for a shading point `x`:
 *   `(power / max(dist², floor)) · coneFactor`.
 * The `dist2Floor` clamp prevents a divide-by-zero / unbounded importance when
 * the shading point lies inside (or on) the node AABB. It is the SAME floor the
 * RIS geometry term uses (`ubo.emitterDist2Floor`) so near-light behaviour stays
 * consistent between selection and evaluation. The B8 cone factor (1 for
 * full-sphere nodes) culls oriented emitters that point away from the point.
 * Matches WGSL `lt_importance`.
 */
export function nodeImportance(
  node: LightTreeNode,
  px: number, py: number, pz: number,
  dist2Floor: number,
): number {
  if (node.totalPower <= 0) return 0;
  const d2 = Math.max(dist2ToAabb(px, py, pz, node.aabbMin, node.aabbMax), dist2Floor);
  const cx = 0.5 * (node.aabbMin[0] + node.aabbMax[0]);
  const cy = 0.5 * (node.aabbMin[1] + node.aabbMax[1]);
  const cz = 0.5 * (node.aabbMin[2] + node.aabbMax[2]);
  const cosThetaO = Math.cos(Math.min(Math.PI, node.cone.thetaO));
  const cosThetaOE = Math.cos(Math.min(Math.PI, node.cone.thetaO + node.cone.thetaE));
  const coneFactor = coneImportanceFactor(
    node.cone.axis, cosThetaO, cosThetaOE, px, py, pz, cx, cy, cz,
  );
  return (node.totalPower / d2) * coneFactor;
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
