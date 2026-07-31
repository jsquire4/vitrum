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
 * GPU flat-buffer layout: 16 floats per node. The final six lanes carry the
 * Conty–Estévez orientation cone and one padding word.
 * See `packLightTreeForGPU` for the exact layout.
 *
 * NOTE: The split policy is Shirley 1996 median-split with power-as-cost.
 * The *traversal* (`sampleLightTreeCPU` / `lightTreePdfCPU` below, mirrored by
 * the WGSL `sampleLightTree` in walkaround-hybrid) is the spatially-aware
 * descent: at each internal node a child is chosen with probability
 * proportional to its **importance** `power / max(dist²(x, childAABB), floor)`
 * for the shading point `x`. This is the distance-weighted importance metric of
 * Estévez & Kulla 2018; when callers supply `cones`, traversal also applies the
 * Conty-Estévez orientation-cone term, and omitted cones recover the old
 * distance-only behaviour. The selection pmf of a leaf is the product of the
 * branch probabilities along the root→leaf path; it sums to 1 over the leaves
 * (a proper probability tree) so it is an unbiased light-selection pdf for RIS/MIS.
 *
 * Future SOTA upgrade to full Estévez-Kulla 2018 would add: (1) SAH-like split
 * with receiver-aware importance; (2) adaptive (non-median) split. Tracked
 * separately.
 *
 * References:
 *   - Shirley, Smits, Wang, Zimmerman 1996, "Monte Carlo Techniques for
 *     Direct Lighting Calculations", ACM TOG (median split, power-as-cost).
 *   - Estévez & Kulla 2018, "Importance Sampling of Many Lights with Adaptive
 *     Tree Splitting", Proc. ACM CGIT (distance-weighted importance descent).
 */
import {
  normalizedPairFirst,
  requireFinite,
  requireFiniteVec3,
  requireInteger,
  requireNonNegative,
  requirePositive,
  requireUnitRandom,
} from './numericGuards.js';

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

/**
 * Full-sphere cone: emits in every direction ⇒ cone importance term ≡ 1.
 *
 * Use this as the default cone for unoriented emitters (point lights, env
 * domes, any emitter with no preferred direction). Exported so consumers can
 * reference the canonical constant rather than re-deriving
 * `{ axis:[0,0,0], thetaO:Math.PI, thetaE:Math.PI }`.
 */
export const FULL_SPHERE_CONE: OrientationCone = { axis: [0, 0, 0], thetaO: Math.PI, thetaE: Math.PI };

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
  const scale = Math.max(Math.abs(v[0]), Math.abs(v[1]), Math.abs(v[2]));
  if (scale === 0) return 0;
  return scale * Math.hypot(v[0] / scale, v[1] / scale, v[2] / scale);
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
  if (la === 0 || lb === 0 || a.thetaO >= Math.PI || b.thetaO >= Math.PI) {
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
  for (const item of items) {
    total += item.power;
    if (!Number.isFinite(total)) throw new RangeError('buildLightTree total power overflowed');
  }
  if (!Number.isFinite(Math.fround(total))) throw new RangeError('buildLightTree total power exceeds f32');
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
    const sorted = items.slice().sort((a, b) => (b.power - a.power) || (a.emitterIndex - b.emitterIndex));
    const mid = Math.floor(sorted.length / 2);
    leftItems = sorted.slice(0, mid);
    rightItems = sorted.slice(mid);
  } else {
    const axis = spanX >= spanY && spanX >= spanZ ? 0 : spanY >= spanZ ? 1 : 2;

    // Sort along longest axis
    const sorted = items.slice().sort((a, b) =>
      (a.centroid[axis] - b.centroid[axis]) || (a.emitterIndex - b.emitterIndex));

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
 * Optional debug output bag returned alongside `nodes` by {@link buildLightTree}.
 *
 * All fields are for CPU-side structural verification only. None are consumed
 * by the GPU traversal or the RIS estimator.
 */
export interface LightTreeDebugOutput {
  /**
   * Unnormalised node-power prefix-sum for CPU-side structural verification.
   *
   * Length = nodeCount (pre-order). Each entry is the running sum of
   * `totalPower` values normalised by `root.totalPower`. Because internal nodes
   * aggregate subtree power, their power is counted before each child's power
   * is also counted — so entries routinely exceed 1.0 for trees with more than
   * one leaf.
   *
   * This is NOT a true probability CDF (which would be normalised to [0, 1]
   * and built only over leaves). Use leaf-only power traversal on `nodes` for
   * sampling. The GPU does its own binary descent from the root and does not
   * consume this array.
   *
   * Useful checks: monotonically non-decreasing; first entry ≈ 1.0 (root
   * power / root power); all entries > 0 when all emitter powers > 0.
   */
  _powerPrefixSumDebug: Float32Array;
}

/**
 * Build a binary light tree where each internal node sums child powers.
 *
 * GPU traversal: descends toward the heavier-power child with probability
 * proportional to child power, then corrects for spatial proximity at the
 * leaf. The GPU consumes the packed node array directly via `packLightTreeForGPU`.
 *
 * The `debug` sub-object carries CPU-side verification data (see
 * {@link LightTreeDebugOutput}). It is always populated but intentionally
 * separated from the primary return so production call-sites can destructure
 * `{ nodes }` without pulling in debug allocations into their type surface.
 *
 * @throws if powers/centroids/aabbs arrays have mismatched lengths
 * @throws if any array is empty
 */
export function buildLightTree(input: LightTreeBuildInput): {
  nodes: LightTreeNode[];
  /**
   * CPU-side structural verification data. Never consumed by the GPU or the
   * RIS estimator. See {@link LightTreeDebugOutput} for field documentation.
   */
  debug: LightTreeDebugOutput;
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
    requireNonNegative(power, `buildLightTree.powers[${i}]`);
    if (!Number.isFinite(Math.fround(power))) throw new RangeError(`buildLightTree.powers[${i}] exceeds f32`);
    requireFiniteVec3(centroid, `buildLightTree.centroids[${i}]`);
    requireFiniteVec3(aabb.min, `buildLightTree.aabbs[${i}].min`);
    requireFiniteVec3(aabb.max, `buildLightTree.aabbs[${i}].max`);
    for (let axis = 0; axis < 3; axis++) {
      const minValue = aabb.min[axis]!;
      const maxValue = aabb.max[axis]!;
      const centroidValue = centroid[axis]!;
      if (minValue > maxValue) {
        throw new RangeError(`buildLightTree.aabbs[${i}] has min > max on axis ${axis}`);
      }
      for (const value of [centroidValue, minValue, maxValue]) {
        if (!Number.isFinite(Math.fround(value))) {
          throw new RangeError(`buildLightTree emitter ${i} geometry exceeds f32`);
        }
      }
    }
    // B8 — per-emitter orientation cone. Omitted ⇒ full sphere (no culling),
    // exactly the pre-B8 behaviour. A present entry defaults thetaE to π/2 (a
    // one-sided cosine emission lobe) and thetaO to 0 (a single sharp axis).
    const ci = cones?.[i];
    let cone: OrientationCone;
    if (ci != null) {
      requireFiniteVec3(ci.axis, `buildLightTree.cones[${i}].axis`);
      const thetaO = ci.thetaO ?? 0;
      const thetaE = ci.thetaE ?? Math.PI / 2;
      requireFinite(thetaO, `buildLightTree.cones[${i}].thetaO`);
      requireFinite(thetaE, `buildLightTree.cones[${i}].thetaE`);
      if (thetaO < 0 || thetaO > Math.PI || thetaE < 0 || thetaE > Math.PI) throw new RangeError(`buildLightTree.cones[${i}] angles must be in [0, PI]`);
    }
    if (ci == null || vlen(ci.axis) === 0) {
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

  return { nodes, debug: { _powerPrefixSumDebug } };
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
  const maxExactF32Integer = 0x01000000;
  const packedLength = nodes.length * FLOATS_PER_NODE;
  if (!Number.isSafeInteger(packedLength) ||
      packedLength > 0x7fffffff ||
      nodes.length - 1 > maxExactF32Integer) {
    throw new RangeError('packLightTreeForGPU node array is too large');
  }
  const parentCounts = new Uint32Array(nodes.length);
  const leafEmitterIndices = new Set<number>();
  const out = new Float32Array(nodes.length * FLOATS_PER_NODE);
  for (let i = 0; i < nodes.length; i++) {
    const node = nodes[i]!;
    requireInteger(node.emitterIndex, `packLightTreeForGPU.nodes[${i}].emitterIndex`, -1, maxExactF32Integer);
    requireInteger(node.leftChild, `packLightTreeForGPU.nodes[${i}].leftChild`, -1, nodes.length - 1);
    requireInteger(node.rightChild, `packLightTreeForGPU.nodes[${i}].rightChild`, -1, nodes.length - 1);
    requireNonNegative(node.totalPower, `packLightTreeForGPU.nodes[${i}].totalPower`);
    requireFiniteVec3(node.aabbMin, `packLightTreeForGPU.nodes[${i}].aabbMin`);
    requireFiniteVec3(node.aabbMax, `packLightTreeForGPU.nodes[${i}].aabbMax`);
    requireFiniteVec3(node.cone.axis, `packLightTreeForGPU.nodes[${i}].cone.axis`);
    requireFinite(node.cone.thetaO, `packLightTreeForGPU.nodes[${i}].cone.thetaO`);
    requireFinite(node.cone.thetaE, `packLightTreeForGPU.nodes[${i}].cone.thetaE`);
    if (node.cone.thetaO < 0 || node.cone.thetaO > Math.PI ||
        node.cone.thetaE < 0 || node.cone.thetaE > Math.PI) {
      throw new RangeError(`packLightTreeForGPU.nodes[${i}].cone angles must be in [0, PI]`);
    }
    for (let axis = 0; axis < 3; axis++) {
      if (node.aabbMin[axis]! > node.aabbMax[axis]!) {
        throw new RangeError(`packLightTreeForGPU.nodes[${i}] has min > max on axis ${axis}`);
      }
    }
    const coneAxisLength = vlen(node.cone.axis);
    if (coneAxisLength === 0 && node.cone.thetaO < Math.PI) {
      throw new RangeError(
        `packLightTreeForGPU.nodes[${i}] zero cone axis requires a full-sphere thetaO`,
      );
    }
    const isLeaf = node.leftChild === -1 && node.rightChild === -1;
    const isInternal = node.leftChild >= 0 && node.rightChild >= 0;
    if (!isLeaf && !isInternal) {
      throw new RangeError(`packLightTreeForGPU.nodes[${i}] must have zero or two children`);
    }
    if (isLeaf) {
      if (node.emitterIndex < 0) {
        throw new RangeError(`packLightTreeForGPU.nodes[${i}] leaf must name an emitter`);
      }
      if (leafEmitterIndices.has(node.emitterIndex)) {
        throw new RangeError(`packLightTreeForGPU duplicate emitter index ${node.emitterIndex}`);
      }
      leafEmitterIndices.add(node.emitterIndex);
    } else {
      if (node.emitterIndex !== -1 || node.leftChild <= i || node.rightChild <= i) {
        throw new RangeError(`packLightTreeForGPU.nodes[${i}] violates pre-order internal-node layout`);
      }
      parentCounts[node.leftChild]!++;
      parentCounts[node.rightChild]!++;
    }
    for (const value of [node.totalPower, ...node.aabbMin, ...node.aabbMax, ...node.cone.axis]) {
      if (!Number.isFinite(Math.fround(value))) throw new RangeError(`packLightTreeForGPU.nodes[${i}] exceeds f32`);
    }

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
    // This packer is public and may receive nodes not built by `buildLightTree`.
    // Normalize at the GPU boundary so a cone is independent of axis magnitude.
    const coneAxisInvLength = coneAxisLength > 0 ? 1 / coneAxisLength : 0;
    out[base + 10] = node.cone.axis[0] * coneAxisInvLength;
    out[base + 11] = node.cone.axis[1] * coneAxisInvLength;
    out[base + 12] = node.cone.axis[2] * coneAxisInvLength;
    out[base + 13] = Math.cos(Math.min(Math.PI, node.cone.thetaO));
    out[base + 14] = Math.cos(Math.min(Math.PI, node.cone.thetaO + node.cone.thetaE));
    out[base + 15] = 0; // padding
  }
  for (let i = 0; i < parentCounts.length; i++) {
    const expectedParents = i === 0 ? 0 : 1;
    if (parentCounts[i] !== expectedParents) {
      throw new RangeError(
        `packLightTreeForGPU.nodes[${i}] has ${parentCounts[i]} parents; expected ${expectedParents}`,
      );
    }
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
 * (AABB centre) TO the point, and `θu` the angular radius of the node's AABB
 * bounding sphere as seen from the point. Conty-Estévez uses
 * `θ' = max(0, θ − θu)`: the subtended-angle term is essential because
 * the centre direction can be behind an emitter plane while another point in
 * the same node is still visible. The factor is:
 *   - `1` when `θ' ≤ thetaO` (inside the normal cone);
 *   - `max(0, cos(θ' − thetaO))` when
 *     `thetaO < θ' ≤ thetaO + thetaE` (the emission-lobe skirt);
 *   - `0` when `θ' > thetaO + thetaE` (outside the bounded lobe).
 * This is the conservative Conty-Estévez 2018 orientation term, evaluated in
 * cosine space so hot traversal performs no inverse trigonometric operations.
 * A full-sphere node (cosThetaO = cosThetaOE = −1, axis length 0) returns 1
 * identically.
 *
 * `cosThetaO` = cos(thetaO); `cosThetaOE` = cos(min(π, thetaO+thetaE)).
 */
function coneImportanceFactor(
  axis: readonly [number, number, number],
  cosThetaO: number,
  cosThetaOE: number,
  px: number, py: number, pz: number,
  cx: number, cy: number, cz: number,
  radius: number,
): number {
  const axisLength = vlen(axis);
  if (axisLength === 0) return 1.0;        // full sphere / unoriented — no culling
  let dx = px - cx, dy = py - cy, dz = pz - cz;
  const dl = vlen([dx, dy, dz]);
  if (dl === 0 || dl <= radius) return 1.0; // inside bounding sphere: do not cull
  const inv = 1.0 / dl;
  dx *= inv; dy *= inv; dz *= inv;
  const aInv = 1.0 / axisLength;
  const cosTheta = Math.max(-1, Math.min(
    1,
    (axis[0] * aInv) * dx + (axis[1] * aInv) * dy + (axis[2] * aInv) * dz,
  ));
  // The AABB is wholly contained by this sphere, so its angular radius cannot
  // exceed asin(radius / distance). Compute cos(max(0, theta-thetaU))
  // directly from the cosine-difference identity.
  const sinThetaU = Math.max(0, Math.min(1, radius / dl));
  const cosThetaU = Math.sqrt(Math.max(0, 1 - sinThetaU * sinThetaU));
  let cosAdjusted = 1.0;
  if (cosTheta < cosThetaU) {
    const sinTheta = Math.sqrt(Math.max(0, 1 - cosTheta * cosTheta));
    cosAdjusted = Math.max(
      -1,
      Math.min(1, cosTheta * cosThetaU + sinTheta * sinThetaU),
    );
  }
  if (cosAdjusted < cosThetaOE) return 0.0;
  if (cosAdjusted >= cosThetaO) return 1.0;
  const sinAdjusted = Math.sqrt(Math.max(0, 1 - cosAdjusted * cosAdjusted));
  const sinThetaO = Math.sqrt(Math.max(0, 1 - cosThetaO * cosThetaO));
  return Math.max(0, cosAdjusted * cosThetaO + sinAdjusted * sinThetaO);
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
  const hx = 0.5 * (node.aabbMax[0] - node.aabbMin[0]);
  const hy = 0.5 * (node.aabbMax[1] - node.aabbMin[1]);
  const hz = 0.5 * (node.aabbMax[2] - node.aabbMin[2]);
  const radius = Math.sqrt(hx * hx + hy * hy + hz * hz);
  const cosThetaO = Math.cos(Math.min(Math.PI, node.cone.thetaO));
  const cosThetaOE = Math.cos(Math.min(Math.PI, node.cone.thetaO + node.cone.thetaE));
  const coneFactor = coneImportanceFactor(
    node.cone.axis, cosThetaO, cosThetaOE, px, py, pz, cx, cy, cz, radius,
  );
  const importance = (node.totalPower / d2) * coneFactor;
  if (!Number.isFinite(importance)) return Number.MAX_VALUE;
  return Math.min(importance, 3.4028234663852886e38);
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
  requireFiniteVec3(x, 'sampleLightTreeCPU.x');
  requirePositive(dist2Floor, 'sampleLightTreeCPU.dist2Floor');
  const [px, py, pz] = x;
  let nodeIdx = 0;
  let pdf = 1.0;
  // Bounded descent: a binary tree over N leaves has depth ≤ N; the explicit
  // cap mirrors the WGSL loop bound (a while-true is illegal there).
  for (let guard = 0; guard < nodes.length + 1; guard++) {
    const node = nodes[nodeIdx];
    if (node == null) throw new RangeError('sampleLightTreeCPU encountered an invalid node index');
    if (node.leftChild < 0 || node.rightChild < 0) {
      // Leaf.
      return { emitterIndex: node.emitterIndex, pdf };
    }
    if (node.leftChild >= nodes.length || node.rightChild >= nodes.length) {
      throw new RangeError('sampleLightTreeCPU encountered an out-of-range child index');
    }
    const left = nodes[node.leftChild]!;
    const right = nodes[node.rightChild]!;
    const impL = nodeImportance(left, px, py, pz, dist2Floor);
    const impR = nodeImportance(right, px, py, pz, dist2Floor);
    // Degenerate: both subtrees contribute zero importance (e.g. all-zero
    // power under this node). Fall back to a uniform 50/50 split so the
    // descent terminates with a positive pdf.
    const pL = normalizedPairFirst(impL, impR);
    const random = requireUnitRandom(rand01(), 'sampleLightTreeCPU.rand01()');
    if (random < pL) {
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
  if (!Number.isSafeInteger(emitterIndex) || emitterIndex < 0) return 0;
  requireFiniteVec3(x, 'lightTreePdfCPU.x');
  requirePositive(dist2Floor, 'lightTreePdfCPU.dist2Floor');
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
    const pL = normalizedPairFirst(impL, impR);
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
