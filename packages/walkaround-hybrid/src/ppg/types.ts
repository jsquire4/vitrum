/**
 * PPG types — sTree + dTree node structs and host model handle.
 *
 * Reference: Müller et al. 2017 "Practical Path Guiding for Efficient
 * Light-Transport Simulation", §3.1 (sTree), §3.2 (dTree).
 */

// ────────────────────────────────────────────────────────────────────────────
// Axis-aligned bounding box (used by sTree for split decisions)
// ────────────────────────────────────────────────────────────────────────────

/** 3-D axis-aligned bounding box in world space. */
export interface AABB {
  min: [number, number, number];
  max: [number, number, number];
}

// ────────────────────────────────────────────────────────────────────────────
// sTree — Adaptive spatial binary tree (Müller §3.1)
// ────────────────────────────────────────────────────────────────────────────

/**
 * Internal node of the spatial binary tree.
 *
 * The tree is stored as a flat array of `STreeNode`. A node is a leaf when
 * `splitAxis === -1`; in that case `dTreeIndex` is the handle into the
 * per-leaf dTree array. Interior nodes have `splitAxis ∈ {0,1,2}` and
 * `leftChild` / `rightChild` pointing to child array indices.
 *
 * Müller §3.1: splits along the longest AABB axis when
 * `sampleCount > PPG_CELL_SPLIT_THRESHOLD`.
 */
export interface STreeNode {
  /** World-space bounds of this node's cell. */
  aabb: AABB;
  /**
   * Split axis: 0=X, 1=Y, 2=Z.
   * -1 indicates this node is a leaf.
   */
  splitAxis: 0 | 1 | 2 | -1;
  /** World-space position of the split plane (only valid for interior nodes). */
  splitValue: number;
  /** Index of the left child in the flat node array (only valid for interior nodes). */
  leftChild: number;
  /** Index of the right child in the flat node array (only valid for interior nodes). */
  rightChild: number;
  /**
   * Index into the `dTrees` array on the containing STree (only valid for leaves).
   * -1 for interior nodes.
   */
  dTreeIndex: number;
  /**
   * Number of training samples accumulated in this cell since the last rebuild.
   * Tracked to determine when to split (Müller §3.1).
   */
  sampleCount: number;
}

// ────────────────────────────────────────────────────────────────────────────
// dTree — Per-cell directional quadtree (Müller §3.2)
// ────────────────────────────────────────────────────────────────────────────

/**
 * A node in the per-cell directional quadtree.
 *
 * The dTree is a full quadtree over the octahedral unit-square [0,1]².
 * Each leaf covers a solid-angle patch on the sphere. Interior nodes split
 * their [u0,v0]×[u1,v1] patch into four equal children.
 *
 * Müller §3.2: a leaf is split when `flux > PPG_DTREE_FLUX_FRACTION × totalFlux`.
 * The leaf's solid angle is computed from its octahedral patch area at build time.
 *
 * Solid-angle formula (deviation 5 fix, Müller §3.2):
 *   For a leaf covering octahedral patch [u0,u1] × [v0,v1] in [0,1]²,
 *   solid angle = 4π × (u1−u0) × (v1−v0).
 *   This is exact for the octahedral parameterisation because the octahedral
 *   square has total area 1 covering the full sphere (4π sr).
 */
export interface DTreeNode {
  /** Whether this node is a leaf (has no children). */
  isLeaf: boolean;
  /** Octahedral patch: u ∈ [u0, u1], v ∈ [v0, v1] in [0,1]². */
  u0: number;
  v0: number;
  u1: number;
  v1: number;
  /**
   * Solid angle (steradians) of the spherical patch represented by this leaf.
   * Computed as 4π × (u1−u0) × (v1−v0) at tree build time.
   * Only meaningful for leaves; interior nodes carry -1.
   *
   * Addresses deviation 5: each leaf has its own solid-angle weight, not
   * the uniform 4π/N approximation used by the deleted implementation.
   */
  solidAngle: number;
  /**
   * Accumulated radiance flux (training signal) at this leaf.
   * Incremented during the training pass via atomic adds on the GPU;
   * summed over all leaves to derive `totalFlux`.
   *
   * Addresses deviation 3: flux is accumulated from L_i (incoming radiance)
   * at the sample point, NOT from post-BRDF outgoing radiance L_o.
   */
  flux: number;
  /** Index of the first child (quadrant NW) in the flat DTree.nodes array. Children are at [child, child+1, child+2, child+3]. -1 for leaves. */
  firstChild: number;
  /** Depth of this node in the tree (root = 0). */
  depth: number;
}

/** A per-spatial-cell directional quadtree. */
export interface DTree {
  /** Flat array of all nodes in this dTree (root at index 0). */
  nodes: DTreeNode[];
  /**
   * Total accumulated flux across all leaves.
   * Used as denominator in the flux-fraction split test.
   */
  totalFlux: number;
}

// ────────────────────────────────────────────────────────────────────────────
// STree — top-level spatial tree
// ────────────────────────────────────────────────────────────────────────────

/** The adaptive spatial binary tree: an array of nodes + per-leaf dTrees. */
export interface STree {
  /** Flat array of all sTree nodes (root at index 0). */
  nodes: STreeNode[];
  /**
   * Per-leaf directional trees.
   * `dTrees[node.dTreeIndex]` is the dTree for leaf `node`.
   */
  dTrees: DTree[];
  /** World-space bounding box of the entire scene. */
  sceneBounds: AABB;
}

// ────────────────────────────────────────────────────────────────────────────
// Host model handle (GPU↔CPU interface)
// ────────────────────────────────────────────────────────────────────────────

/**
 * Opaque handle returned by `buildSTree` / `ppgUpdateCPU` to the host.
 * The host passes this handle into `ppgGuide` (direction sampling) and
 * `ppgUpdateCPU` (rebuild after a training frame).
 */
export interface PPGModelHandle {
  /** The live spatial + directional tree (CPU-side). */
  sTree: STree;
  /**
   * Serialised flat representation of the tree ready to upload to a GPU
   * storage buffer. Updated each rebuild cycle.
   */
  serialised: Float32Array;
  /**
   * Current MIS mixing weight α ∈ [PPG_MIS_ALPHA_MIN, PPG_MIS_ALPHA_MAX].
   * Fixed at PPG_MIS_ALPHA = 0.5 in this implementation (variance-adaptive
   * update is a future enhancement — see plan/sprint-ppg-rebuild-future.md).
   */
  alpha: number;
}
