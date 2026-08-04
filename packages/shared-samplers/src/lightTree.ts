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
 * The GPU traversal descends probabilistically, apportioning buckets from each
 * child's represented power, proximity, and orientation importance.
 *
 * GPU flat-buffer layout: 16 floats per node. The final six lanes carry the
 * Conty–Estévez orientation cone and exact subtree leaf count.
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
 * distance-only behaviour. A single 24-bit root bucket is partitioned through
 * the tree with one reserved bucket per descendant leaf. The selection PMF is
 * the leaf interval's exact bucket count / 2²⁴, so it sums to one and cannot
 * collapse with either an extreme power ratio or a deep tree.
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

/**
 * The shared PCG and pt-webgpu Sobol adapters both expose exactly 24 random
 * mantissa bits.  Light-tree selection therefore partitions this many root
 * buckets instead of multiplying f32 branch probabilities down the tree.
 * Every represented leaf owns at least one bucket and its returned PMF is the
 * exact size of its root-bucket interval divided by this constant.
 */
export const LIGHT_TREE_BUCKET_COUNT = 0x01000000;

const F32_MAX = 3.4028234663852886e38;
const F32_MIN_NORMAL = 1.1754943508222875e-38;
const F32_MIN_NORMAL_LOG2 = -126;
const F32_MAX_LOG2 = Math.log2(F32_MAX);

// The packed shader normalizes two f32 vectors and evaluates their dot product
// through several rounded operations. At an authored cone boundary, storing the
// nearest f32 cosine can therefore place that same direction outside the
// represented cone. Round both thresholds outward by at least 64 local f32
// values and by an absolute 2^-18 cosine pad. The absolute pad is required near
// cosine zero, where local ulps are far smaller than the accumulated error from
// axis packing plus differing legal dot/length evaluation orders. At cosine one
// the same pad widens a zero-width cone by only about 0.16 degrees.
const LIGHT_TREE_CONE_BOUND_OUTWARD_ULPS = 64;
const LIGHT_TREE_CONE_BOUND_ABSOLUTE_PAD = 2 ** -18;
const F32_STEP_BUFFER = new ArrayBuffer(Float32Array.BYTES_PER_ELEMENT);
const F32_STEP_VALUE = new Float32Array(F32_STEP_BUFFER);
const F32_STEP_BITS = new Uint32Array(F32_STEP_BUFFER);

function nextDownF32(value: number): number {
  F32_STEP_VALUE[0] = value;
  const rounded = F32_STEP_VALUE[0];
  if (Number.isNaN(rounded) || rounded === Number.NEGATIVE_INFINITY) return rounded;
  if (rounded === 0) return -(2 ** -149);
  const bits = F32_STEP_BITS[0]!;
  F32_STEP_BITS[0] = rounded > 0 ? bits - 1 : bits + 1;
  return F32_STEP_VALUE[0];
}

function conservativeConeCosineThreshold(angle: number): number {
  const roundedCosine = Math.fround(Math.cos(Math.min(Math.PI, angle)));
  let threshold = roundedCosine;
  for (let step = 0; step < LIGHT_TREE_CONE_BOUND_OUTWARD_ULPS; step += 1) {
    threshold = nextDownF32(threshold);
  }
  const absoluteThreshold = Math.fround(
    roundedCosine - LIGHT_TREE_CONE_BOUND_ABSOLUTE_PAD,
  );
  return Math.max(-1, Math.min(threshold, absoluteThreshold));
}

/**
 * Publish a raw positive subtree power into the scale-free f32 proposal
 * domain.  A common scale preserves all ordinary ratios.  Ratios below the
 * smallest normal f32 are deliberately floored rather than stored as zero:
 * WebGPU implementations may flush subnormals, and zero would silently remove
 * a physically positive emitter from the proposal's support.
 *
 * Production consumers use this through {@link packLightTreeForGPU}.
 */
function representedLightTreePower(
  totalPower: number,
  maximumNodePower: number,
): number {
  if (!(totalPower > 0) || !(maximumNodePower > 0)) return 0;
  const ratio = totalPower / maximumNodePower;
  if (!(ratio > 0)) return F32_MIN_NORMAL;
  const rounded = Math.fround(ratio);
  return rounded > F32_MIN_NORMAL ? rounded : F32_MIN_NORMAL;
}

/** Common scale used by GPU packing. */
function maximumLightTreeNodePower(
  nodes: ReadonlyArray<LightTreeNode>,
): number {
  let maximum = 0;
  for (const node of nodes) {
    requireNonNegative(node.totalPower, 'maximumLightTreeNodePower.totalPower');
    maximum = Math.max(maximum, node.totalPower);
  }
  return maximum;
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
  if (tl < 1e-8) {
    // The same-direction case was returned as enclosed above. Reaching this
    // branch therefore means the axes are antiparallel (or numerically
    // indistinguishable from antiparallel), where the minimal union has no
    // stable rotation plane. Choosing an arbitrary perpendicular plane can put
    // the real second axis just outside the returned cone after f32 packing,
    // making a supported descendant's importance exactly zero. The full-sphere
    // sentinel is the conservative union and cannot cull either descendant.
    return { axis: [0, 0, 0], thetaO: Math.PI, thetaE };
  }
  tx /= tl; ty /= tl; tz /= tl;
  const cr = Math.cos(rot);
  const sr = Math.sin(rot);
  const axis: readonly [number, number, number] = [
    ax[0] * cr + tx * sr,
    ax[1] * cr + ty * sr,
    ax[2] * cr + tz * sr,
  ];
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
 * GPU traversal apportions the root bucket domain at each branch from the
 * child's represented power, spatial proximity, and orientation cone. The GPU
 * consumes the packed node array directly via `packLightTreeForGPU`.
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
  let runningNormalised = 0;
  for (let i = 0; i < nodes.length; i++) {
    runningNormalised += rootPower > 0 ? nodes[i]!.totalPower / rootPower : 0;
    _powerPrefixSumDebug[i] = runningNormalised;
  }

  return { nodes, debug: { _powerPrefixSumDebug } };
}

/**
 * Pack the node array into a Float32Array suitable for GPU texture / storage
 * upload.
 *
 * Layout per node (16 logical floats, aligned for RGBA32F 4-texel
 * alignment — B8 grew the stride from 12 to carry the orientation cone):
 *   [0]  emitterIndex (as float; -1.0 for internal)
 *   [1]  represented proposal power: totalPower/commonMax, with every raw
 *        positive value floored to the smallest normal f32
 *   [2]  leftChild (as float; -1.0 for leaf)
 *   [3]  rightChild (as float; -1.0 for leaf)
 *   [4]  aabbMin.x        [5] aabbMin.y      [6] aabbMin.z
 *   [7]  aabbMax.x        [8] aabbMax.y      [9] aabbMax.z
 *   [10] cone.axis.x      [11] cone.axis.y   [12] cone.axis.z
 *   [13] conservatively outward-rounded cos(cone.thetaO)
 *   [14] conservatively outward-rounded cos(min(π, thetaO + thetaE))
 *   [15] subtree leaf count (exact positive integer encoded as f32)
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
  const maximumNodePower = maximumLightTreeNodePower(nodes);
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
    for (const value of [...node.aabbMin, ...node.aabbMax, ...node.cone.axis]) {
      if (!Number.isFinite(Math.fround(value))) throw new RangeError(`packLightTreeForGPU.nodes[${i}] exceeds f32`);
    }

    const base = i * FLOATS_PER_NODE;
    out[base + 0] = node.emitterIndex;
    out[base + 1] = representedLightTreePower(node.totalPower, maximumNodePower);
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
    out[base + 10] = coneAxisLength > 0 ? node.cone.axis[0] / coneAxisLength : 0;
    out[base + 11] = coneAxisLength > 0 ? node.cone.axis[1] / coneAxisLength : 0;
    out[base + 12] = coneAxisLength > 0 ? node.cone.axis[2] / coneAxisLength : 0;
    if (!Number.isFinite(out[base + 10]!) ||
        !Number.isFinite(out[base + 11]!) ||
        !Number.isFinite(out[base + 12]!)) {
      throw new RangeError(`packLightTreeForGPU.nodes[${i}].cone.axis cannot be represented as a finite unit f32 vector`);
    }
    out[base + 13] = conservativeConeCosineThreshold(node.cone.thetaO);
    out[base + 14] = conservativeConeCosineThreshold(
      node.cone.thetaO + node.cone.thetaE,
    );
    // Lane 15 is filled after structural validation by the reverse leaf-count
    // pass below. Children are later in the pre-order array.
    out[base + 15] = 0;
  }
  for (let i = 0; i < parentCounts.length; i++) {
    const expectedParents = i === 0 ? 0 : 1;
    if (parentCounts[i] !== expectedParents) {
      throw new RangeError(
        `packLightTreeForGPU.nodes[${i}] has ${parentCounts[i]} parents; expected ${expectedParents}`,
      );
    }
  }
  const subtreeLeafCounts = new Uint32Array(nodes.length);
  for (let i = nodes.length - 1; i >= 0; i--) {
    const node = nodes[i]!;
    let leafCount: number;
    if (node.leftChild < 0 && node.rightChild < 0) {
      leafCount = 1;
    } else {
      leafCount = subtreeLeafCounts[node.leftChild]! + subtreeLeafCounts[node.rightChild]!;
    }
    if (leafCount > LIGHT_TREE_BUCKET_COUNT) {
      throw new RangeError(
        `packLightTreeForGPU.nodes[${i}] has more leaves than the 24-bit proposal domain`,
      );
    }
    subtreeLeafCounts[i] = leafCount;
    out[i * FLOATS_PER_NODE + 15] = leafCount;
  }
  return out;
}

/**
 * 16 floats per node in the packed flat layout consumed by `packLightTreeForGPU`
 * and the WGSL `sampleLightTree` traversal (B8 grew this from 12 to carry the
 * orientation cone and exact subtree leaf count). The WGSL side reads the same stride from a flat
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
  const scale = Math.max(Math.abs(dx), Math.abs(dy), Math.abs(dz));
  if (scale === 0) return 0;
  const sx = dx / scale;
  const sy = dy / scale;
  const sz = dz / scale;
  const scaledSquared = sx * sx + sy * sy + sz * sz;
  if (scale >= Math.sqrt(F32_MAX / scaledSquared)) return F32_MAX;
  return Math.min(F32_MAX, scale * scale * scaledSquared);
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
 *   - a positive cosine-space taper when
 *     `thetaO < θ' ≤ thetaO + thetaE` (the emission-lobe skirt);
 *   - `0` when `θ' > thetaO + thetaE` (outside the bounded lobe).
 * The taper linearly maps the represented cosine interval from 1 at `thetaO`
 * to the smallest normal f32 at the inclusive support boundary. It equals the
 * ordinary cosine lobe for a sharp-axis hemisphere leaf and, unlike a clamped
 * `cos(θ' − thetaO)`, remains positive for the public `thetaE > π/2` contract.
 * Thus the Conty-Estévez support cone never removes a physically supported
 * emitter while hot traversal still performs no inverse trigonometric work.
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
  const cosineSpan = cosThetaO - cosThetaOE;
  if (!(cosineSpan > 0)) return 1.0;
  return Math.max(
    F32_MIN_NORMAL,
    Math.min(1, (cosAdjusted - cosThetaOE) / cosineSpan),
  );
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
  return nodeImportanceWithPower(
    node,
    node.totalPower,
    px,
    py,
    pz,
    dist2Floor,
  );
}

function nodeImportanceWithPower(
  node: LightTreeNode,
  power: number,
  px: number,
  py: number,
  pz: number,
  dist2Floor: number,
): number {
  if (!(power > 0)) return 0;
  const min = node.aabbMin;
  const max = node.aabbMax;
  const d2 = Math.max(
    dist2ToAabb(px, py, pz, min, max),
    dist2Floor,
    F32_MIN_NORMAL,
  );
  // Half-sum / half-difference forms remain finite for opposite-sign f32
  // extrema, unlike `(min + max) * 0.5` and `(max - min) * 0.5`.
  const cx = 0.5 * min[0] + 0.5 * max[0];
  const cy = 0.5 * min[1] + 0.5 * max[1];
  const cz = 0.5 * min[2] + 0.5 * max[2];
  const hx = 0.5 * max[0] - 0.5 * min[0];
  const hy = 0.5 * max[1] - 0.5 * min[1];
  const hz = 0.5 * max[2] - 0.5 * min[2];
  const radius = vlen([hx, hy, hz]);
  const cosThetaO = conservativeConeCosineThreshold(node.cone.thetaO);
  const cosThetaOE = conservativeConeCosineThreshold(
    node.cone.thetaO + node.cone.thetaE,
  );
  const coneFactor = coneImportanceFactor(
    node.cone.axis,
    cosThetaO,
    cosThetaOE,
    px,
    py,
    pz,
    cx,
    cy,
    cz,
    radius,
  );
  if (!(coneFactor > 0)) return 0;

  // Log-domain evaluation prevents both `(power / d2)` overflow and a
  // positive product collapsing to zero. The result is deliberately kept in
  // the normal f32 domain for the same FTZ-safe contract as packed powers.
  const logImportance =
    Math.log2(power) - Math.log2(d2) + Math.log2(coneFactor);
  if (logImportance >= F32_MAX_LOG2) return F32_MAX;
  if (logImportance <= Math.log2(Number.MIN_VALUE)) return 0;
  return Math.min(F32_MAX, 2 ** logImportance);
}

// Explicit f32 operations used by the packed CPU oracle. Each intermediate is
// rounded where the corresponding WGSL expression produces an f32 value.
const f32 = Math.fround;
const f32Add = (a: number, b: number): number => f32(f32(a) + f32(b));
const f32Sub = (a: number, b: number): number => f32(f32(a) - f32(b));
const f32Mul = (a: number, b: number): number => f32(f32(a) * f32(b));
const f32Div = (a: number, b: number): number => f32(f32(a) / f32(b));
const f32Sqrt = (a: number): number => f32(Math.sqrt(f32(a)));
const f32Log2 = (a: number): number => f32(Math.log2(f32(a)));
const f32Exp2 = (a: number): number => f32(2 ** f32(a));

function f32Dot3(
  ax: number,
  ay: number,
  az: number,
  bx: number,
  by: number,
  bz: number,
): number {
  return f32Add(
    f32Add(f32Mul(ax, bx), f32Mul(ay, by)),
    f32Mul(az, bz),
  );
}

function packedDist2ToAabbCPU(
  px: number,
  py: number,
  pz: number,
  minX: number,
  minY: number,
  minZ: number,
  maxX: number,
  maxY: number,
  maxZ: number,
): number {
  const halfDx = Math.max(
    f32Sub(f32Mul(0.5, minX), f32Mul(0.5, px)),
    0,
    f32Sub(f32Mul(0.5, px), f32Mul(0.5, maxX)),
  );
  const halfDy = Math.max(
    f32Sub(f32Mul(0.5, minY), f32Mul(0.5, py)),
    0,
    f32Sub(f32Mul(0.5, py), f32Mul(0.5, maxY)),
  );
  const halfDz = Math.max(
    f32Sub(f32Mul(0.5, minZ), f32Mul(0.5, pz)),
    0,
    f32Sub(f32Mul(0.5, pz), f32Mul(0.5, maxZ)),
  );
  const halfScale = Math.max(Math.abs(halfDx), Math.abs(halfDy), Math.abs(halfDz));
  if (!(halfScale >= F32_MIN_NORMAL)) return 0;
  if (halfScale > f32Mul(0.5, F32_MAX)) return F32_MAX;
  const sx = f32Div(halfDx, halfScale);
  const sy = f32Div(halfDy, halfScale);
  const sz = f32Div(halfDz, halfScale);
  const scaledSquared = f32Dot3(sx, sy, sz, sx, sy, sz);
  const actualScale = f32Mul(2, halfScale);
  const limit = f32Sqrt(f32Div(F32_MAX, scaledSquared));
  if (actualScale >= limit) return F32_MAX;
  return Math.min(
    F32_MAX,
    f32Mul(f32Mul(actualScale, actualScale), scaledSquared),
  );
}

function packedConeFactorCPU(
  axisX: number,
  axisY: number,
  axisZ: number,
  cosThetaO: number,
  cosThetaOE: number,
  px: number,
  py: number,
  pz: number,
  minX: number,
  minY: number,
  minZ: number,
  maxX: number,
  maxY: number,
  maxZ: number,
): number {
  const axisScale = Math.max(Math.abs(axisX), Math.abs(axisY), Math.abs(axisZ));
  if (!(axisScale > 0) || axisScale > F32_MAX) return 1;
  const scaledAxisX = f32Div(axisX, axisScale);
  const scaledAxisY = f32Div(axisY, axisScale);
  const scaledAxisZ = f32Div(axisZ, axisScale);
  const axisLength = f32Sqrt(f32Dot3(
    scaledAxisX,
    scaledAxisY,
    scaledAxisZ,
    scaledAxisX,
    scaledAxisY,
    scaledAxisZ,
  ));

  const dvX = f32Sub(
    f32Sub(f32Mul(0.5, px), f32Mul(0.25, minX)),
    f32Mul(0.25, maxX),
  );
  const dvY = f32Sub(
    f32Sub(f32Mul(0.5, py), f32Mul(0.25, minY)),
    f32Mul(0.25, maxY),
  );
  const dvZ = f32Sub(
    f32Sub(f32Mul(0.5, pz), f32Mul(0.25, minZ)),
    f32Mul(0.25, maxZ),
  );
  const radiusX = f32Sub(f32Mul(0.25, maxX), f32Mul(0.25, minX));
  const radiusY = f32Sub(f32Mul(0.25, maxY), f32Mul(0.25, minY));
  const radiusZ = f32Sub(f32Mul(0.25, maxZ), f32Mul(0.25, minZ));
  const distanceScale = Math.max(Math.abs(dvX), Math.abs(dvY), Math.abs(dvZ));
  if (!(distanceScale >= F32_MIN_NORMAL) || distanceScale > F32_MAX) return 1;
  const scaledDistanceX = f32Div(dvX, distanceScale);
  const scaledDistanceY = f32Div(dvY, distanceScale);
  const scaledDistanceZ = f32Div(dvZ, distanceScale);
  const scaledDistanceLength = f32Sqrt(f32Dot3(
    scaledDistanceX,
    scaledDistanceY,
    scaledDistanceZ,
    scaledDistanceX,
    scaledDistanceY,
    scaledDistanceZ,
  ));

  const radiusScale = Math.max(Math.abs(radiusX), Math.abs(radiusY), Math.abs(radiusZ));
  let radiusOverDistance = 0;
  if (radiusScale >= F32_MIN_NORMAL) {
    const scaledRadiusX = f32Div(radiusX, radiusScale);
    const scaledRadiusY = f32Div(radiusY, radiusScale);
    const scaledRadiusZ = f32Div(radiusZ, radiusScale);
    const scaledRadiusLength = f32Sqrt(f32Dot3(
      scaledRadiusX,
      scaledRadiusY,
      scaledRadiusZ,
      scaledRadiusX,
      scaledRadiusY,
      scaledRadiusZ,
    ));
    const logRatio = f32Sub(
      f32Sub(
        f32Add(f32Log2(radiusScale), f32Log2(scaledRadiusLength)),
        f32Log2(distanceScale),
      ),
      f32Log2(scaledDistanceLength),
    );
    if (logRatio >= 0) return 1;
    radiusOverDistance = f32Exp2(Math.max(logRatio, F32_MIN_NORMAL_LOG2));
  }

  const dirX = f32Div(scaledDistanceX, scaledDistanceLength);
  const dirY = f32Div(scaledDistanceY, scaledDistanceLength);
  const dirZ = f32Div(scaledDistanceZ, scaledDistanceLength);
  const normAxisX = f32Div(scaledAxisX, axisLength);
  const normAxisY = f32Div(scaledAxisY, axisLength);
  const normAxisZ = f32Div(scaledAxisZ, axisLength);
  const cosTheta = Math.max(-1, Math.min(1, f32Dot3(
    normAxisX,
    normAxisY,
    normAxisZ,
    dirX,
    dirY,
    dirZ,
  )));
  const sinThetaU = Math.max(0, Math.min(1, radiusOverDistance));
  const cosThetaU = f32Sqrt(Math.max(0, f32Sub(1, f32Mul(sinThetaU, sinThetaU))));
  let cosAdjusted = 1;
  if (cosTheta < cosThetaU) {
    const sinTheta = f32Sqrt(Math.max(0, f32Sub(1, f32Mul(cosTheta, cosTheta))));
    cosAdjusted = Math.max(-1, Math.min(1, f32Add(
      f32Mul(cosTheta, cosThetaU),
      f32Mul(sinTheta, sinThetaU),
    )));
  }
  if (cosAdjusted < cosThetaOE) return 0;
  if (cosAdjusted >= cosThetaO) return 1;
  const cosineSpan = f32Sub(cosThetaO, cosThetaOE);
  if (!(cosineSpan > 0)) return 1;
  return Math.max(
    F32_MIN_NORMAL,
    Math.min(1, f32Div(f32Sub(cosAdjusted, cosThetaOE), cosineSpan)),
  );
}

function packedNodeImportanceUnchecked(
  packed: Float32Array,
  nodeIndex: number,
  px: number,
  py: number,
  pz: number,
  dist2Floor: number,
): number {
  const base = nodeIndex * LIGHT_TREE_FLOATS_PER_NODE;
  const power = packed[base + 1]!;
  if (!(power > 0)) return 0;
  const minX = packed[base + 4]!;
  const minY = packed[base + 5]!;
  const minZ = packed[base + 6]!;
  const maxX = packed[base + 7]!;
  const maxY = packed[base + 8]!;
  const maxZ = packed[base + 9]!;
  const d2 = Math.max(
    packedDist2ToAabbCPU(px, py, pz, minX, minY, minZ, maxX, maxY, maxZ),
    f32(dist2Floor),
    F32_MIN_NORMAL,
  );
  const cone = packedConeFactorCPU(
    packed[base + 10]!,
    packed[base + 11]!,
    packed[base + 12]!,
    packed[base + 13]!,
    packed[base + 14]!,
    px,
    py,
    pz,
    minX,
    minY,
    minZ,
    maxX,
    maxY,
    maxZ,
  );
  if (!(cone > 0)) return 0;
  const logImportance = f32Add(
    f32Sub(f32Log2(power), f32Log2(d2)),
    f32Log2(Math.max(cone, F32_MIN_NORMAL)),
  );
  if (logImportance >= f32(127.99999)) return F32_MAX;
  if (logImportance <= F32_MIN_NORMAL_LOG2) return F32_MIN_NORMAL;
  return Math.max(
    F32_MIN_NORMAL,
    Math.min(F32_MAX, f32Exp2(logImportance)),
  );
}

/**
 * CPU oracle for the canonical WGSL importance evaluation, reading the exact
 * packed f32 lanes and rounding every arithmetic stage back to f32.
 */
export function packedLightTreeNodeImportanceCPU(
  packed: Float32Array,
  nodeIndex: number,
  x: readonly [number, number, number],
  dist2Floor: number,
): number {
  if (packed.length % LIGHT_TREE_FLOATS_PER_NODE !== 0) {
    throw new RangeError('packedLightTreeNodeImportanceCPU received a partial node');
  }
  const nodeCount = packed.length / LIGHT_TREE_FLOATS_PER_NODE;
  requireInteger(nodeIndex, 'packedLightTreeNodeImportanceCPU.nodeIndex', 0, nodeCount - 1);
  requireFiniteVec3(x, 'packedLightTreeNodeImportanceCPU.x');
  requirePositive(dist2Floor, 'packedLightTreeNodeImportanceCPU.dist2Floor');
  return packedNodeImportanceUnchecked(
    packed,
    nodeIndex,
    f32(x[0]),
    f32(x[1]),
    f32(x[2]),
    f32(dist2Floor),
  );
}

/** f32 branch target used only to apportion the buckets left after support. */
function representedPairFirst(a: number, b: number): number {
  if (a === 0 && b === 0) return 0.5;
  const scale = Math.max(a, b);
  const left = Math.fround(a / scale);
  const right = Math.fround(b / scale);
  return Math.fround(left / Math.fround(left + right));
}

function allocateLeftBuckets(
  currentBuckets: number,
  leftLeaves: number,
  rightLeaves: number,
  leftImportance: number,
  rightImportance: number,
): number {
  const leafCount = leftLeaves + rightLeaves;
  if (currentBuckets < leafCount) {
    throw new RangeError('light-tree bucket partition lost represented leaf support');
  }
  const remaining = currentBuckets - leafCount;
  const pLeft = representedPairFirst(leftImportance, rightImportance);
  const idealExtra = Math.fround(Math.fround(remaining) * pLeft);
  const leftExtra = Math.min(
    remaining,
    Math.max(0, Math.floor(Math.fround(idealExtra + 0.5))),
  );
  return leftLeaves + leftExtra;
}

function packedChildIndex(
  packed: Float32Array,
  base: number,
  lane: 2 | 3,
  nodeCount: number,
  caller: string,
): number {
  const value = packed[base + lane]!;
  if (!Number.isSafeInteger(value) || value < -1 || value >= nodeCount) {
    throw new RangeError(`${caller} encountered an invalid packed child index`);
  }
  return value;
}

function packedLeafCount(
  packed: Float32Array,
  base: number,
  caller: string,
): number {
  const count = packed[base + 15]!;
  if (!Number.isSafeInteger(count) || count < 1 || count > LIGHT_TREE_BUCKET_COUNT) {
    throw new RangeError(`${caller} encountered an invalid packed subtree leaf count`);
  }
  return count;
}

/**
 * Sample directly from the packed GPU representation. This is the CPU/WGSL
 * parity oracle: proposal powers, cone axes, cosines, AABBs, leaf counts, and
 * every importance arithmetic stage are the same f32 values the shader reads.
 */
export function samplePackedLightTreeCPU(
  packed: Float32Array,
  x: readonly [number, number, number],
  dist2Floor: number,
  rand01: () => number,
): { emitterIndex: number; pdf: number } {
  if (packed.length === 0) return { emitterIndex: -1, pdf: 0 };
  if (packed.length % LIGHT_TREE_FLOATS_PER_NODE !== 0) {
    throw new RangeError('samplePackedLightTreeCPU received a partial node');
  }
  requireFiniteVec3(x, 'samplePackedLightTreeCPU.x');
  requirePositive(dist2Floor, 'samplePackedLightTreeCPU.dist2Floor');
  const nodeCount = packed.length / LIGHT_TREE_FLOATS_PER_NODE;
  const pointX = f32(x[0]);
  const pointY = f32(x[1]);
  const pointZ = f32(x[2]);
  const floorValue = f32(dist2Floor);
  let nodeIndex = 0;
  let currentBuckets = LIGHT_TREE_BUCKET_COUNT;
  let localBucket = 0;
  const rootLeft = packedChildIndex(packed, 0, 2, nodeCount, 'samplePackedLightTreeCPU');
  const rootRight = packedChildIndex(packed, 0, 3, nodeCount, 'samplePackedLightTreeCPU');
  if ((rootLeft < 0) !== (rootRight < 0)) {
    throw new RangeError('samplePackedLightTreeCPU root has only one child');
  }
  if (rootLeft >= 0) {
    localBucket = Math.min(
      LIGHT_TREE_BUCKET_COUNT - 1,
      Math.floor(
        requireUnitRandom(rand01(), 'samplePackedLightTreeCPU.rand01()') *
          LIGHT_TREE_BUCKET_COUNT,
      ),
    );
  }

  for (let guard = 0; guard < nodeCount + 1; guard++) {
    const base = nodeIndex * LIGHT_TREE_FLOATS_PER_NODE;
    const left = packedChildIndex(packed, base, 2, nodeCount, 'samplePackedLightTreeCPU');
    const right = packedChildIndex(packed, base, 3, nodeCount, 'samplePackedLightTreeCPU');
    if ((left < 0) !== (right < 0)) {
      throw new RangeError('samplePackedLightTreeCPU encountered a node with only one child');
    }
    if (left < 0) {
      return {
        emitterIndex: packed[base + 0]!,
        pdf: currentBuckets / LIGHT_TREE_BUCKET_COUNT,
      };
    }
    const leftBase = left * LIGHT_TREE_FLOATS_PER_NODE;
    const rightBase = right * LIGHT_TREE_FLOATS_PER_NODE;
    const leftBuckets = allocateLeftBuckets(
      currentBuckets,
      packedLeafCount(packed, leftBase, 'samplePackedLightTreeCPU'),
      packedLeafCount(packed, rightBase, 'samplePackedLightTreeCPU'),
      packedNodeImportanceUnchecked(
        packed,
        left,
        pointX,
        pointY,
        pointZ,
        floorValue,
      ),
      packedNodeImportanceUnchecked(
        packed,
        right,
        pointX,
        pointY,
        pointZ,
        floorValue,
      ),
    );
    if (localBucket < leftBuckets) {
      currentBuckets = leftBuckets;
      nodeIndex = left;
    } else {
      localBucket -= leftBuckets;
      currentBuckets -= leftBuckets;
      nodeIndex = right;
    }
  }
  throw new RangeError('samplePackedLightTreeCPU exceeded the bounded tree descent');
}

/** Deterministic inverse PMF of {@link samplePackedLightTreeCPU}. */
export function packedLightTreePdfCPU(
  packed: Float32Array,
  x: readonly [number, number, number],
  dist2Floor: number,
  emitterIndex: number,
): number {
  if (packed.length === 0) return 0;
  if (packed.length % LIGHT_TREE_FLOATS_PER_NODE !== 0) {
    throw new RangeError('packedLightTreePdfCPU received a partial node');
  }
  if (!Number.isSafeInteger(emitterIndex) || emitterIndex < 0) return 0;
  requireFiniteVec3(x, 'packedLightTreePdfCPU.x');
  requirePositive(dist2Floor, 'packedLightTreePdfCPU.dist2Floor');
  const nodeCount = packed.length / LIGHT_TREE_FLOATS_PER_NODE;
  const contains = new Array<boolean>(nodeCount).fill(false);
  for (let i = nodeCount - 1; i >= 0; i--) {
    const base = i * LIGHT_TREE_FLOATS_PER_NODE;
    const left = packedChildIndex(packed, base, 2, nodeCount, 'packedLightTreePdfCPU');
    const right = packedChildIndex(packed, base, 3, nodeCount, 'packedLightTreePdfCPU');
    if ((left < 0) !== (right < 0)) {
      throw new RangeError('packedLightTreePdfCPU encountered a node with only one child');
    }
    contains[i] = left < 0
      ? packed[base + 0] === emitterIndex
      : contains[left]! || contains[right]!;
  }
  if (!contains[0]) return 0;

  const pointX = f32(x[0]);
  const pointY = f32(x[1]);
  const pointZ = f32(x[2]);
  const floorValue = f32(dist2Floor);
  let nodeIndex = 0;
  let currentBuckets = LIGHT_TREE_BUCKET_COUNT;
  for (let guard = 0; guard < nodeCount + 1; guard++) {
    const base = nodeIndex * LIGHT_TREE_FLOATS_PER_NODE;
    const left = packedChildIndex(packed, base, 2, nodeCount, 'packedLightTreePdfCPU');
    const right = packedChildIndex(packed, base, 3, nodeCount, 'packedLightTreePdfCPU');
    if (left < 0 || right < 0) {
      return packed[base + 0] === emitterIndex
        ? currentBuckets / LIGHT_TREE_BUCKET_COUNT
        : 0;
    }
    const leftBase = left * LIGHT_TREE_FLOATS_PER_NODE;
    const rightBase = right * LIGHT_TREE_FLOATS_PER_NODE;
    const leftBuckets = allocateLeftBuckets(
      currentBuckets,
      packedLeafCount(packed, leftBase, 'packedLightTreePdfCPU'),
      packedLeafCount(packed, rightBase, 'packedLightTreePdfCPU'),
      packedNodeImportanceUnchecked(
        packed,
        left,
        pointX,
        pointY,
        pointZ,
        floorValue,
      ),
      packedNodeImportanceUnchecked(
        packed,
        right,
        pointX,
        pointY,
        pointZ,
        floorValue,
      ),
    );
    if (contains[left]) {
      currentBuckets = leftBuckets;
      nodeIndex = left;
    } else {
      currentBuckets -= leftBuckets;
      nodeIndex = right;
    }
  }
  throw new RangeError('packedLightTreePdfCPU exceeded the bounded tree descent');
}

/**
 * Importance-sample a single emitter (leaf) from the light tree for a shading
 * point `x`, returning the chosen `emitterIndex` and the **selection pdf**.
 * One 24-bit root draw selects an integer bucket. At every branch, each leaf is
 * first reserved one bucket and the remaining buckets are apportioned by the
 * represented child importances. The returned PDF is the leaf's final bucket
 * count divided by 2²⁴: it is exactly the probability implemented by the draw,
 * cannot underflow with depth, and is positive for every represented leaf.
 *
 * `rand01` is a 0-arg sampler returning a uniform in [0, 1). Exactly one draw is
 * consumed for a non-trivial tree. The WGSL kernel's PCG/Sobol adapters expose
 * the same high 24-bit domain.
 *
 * Degenerate cases (both children zero-importance) apportion spare buckets
 * evenly; the one-bucket-per-leaf reservation remains the support guarantee.
 *
 * Mirrors the WGSL `sampleLightTree` byte-for-byte in branch logic.
 */
export function sampleLightTreeCPU(
  nodes: ReadonlyArray<LightTreeNode>,
  x: readonly [number, number, number],
  dist2Floor: number,
  rand01: () => number,
): { emitterIndex: number; pdf: number } {
  return samplePackedLightTreeCPU(
    packLightTreeForGPU(nodes),
    x,
    dist2Floor,
    rand01,
  );
}

/**
 * Recompute the selection pdf the tree assigns to a given `emitterIndex` for a
 * shading point `x`, WITHOUT drawing random numbers — it walks the unique
 * root→leaf path to that emitter, reproducing the exact integer bucket
 * partition. This is the deterministic inverse of `sampleLightTreeCPU`'s pdf and is used (a) in
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
  return packedLightTreePdfCPU(
    packLightTreeForGPU(nodes),
    x,
    dist2Floor,
    emitterIndex,
  );
}
