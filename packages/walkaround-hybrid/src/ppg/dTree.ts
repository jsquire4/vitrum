/**
 * dTree — Per-cell adaptive directional quadtree for Practical Path Guiding.
 *
 * Reference: Müller et al. 2017 "Practical Path Guiding for Efficient
 * Light-Transport Simulation", §3.2.
 *
 * Each sTree leaf has its own dTree — a full quadtree over the octahedral
 * unit-square [0,1]². Leaves split adaptively when their accumulated flux
 * fraction exceeds `PPG_DTREE_FLUX_FRACTION × totalFlux`.
 *
 * ADDRESSES DEVIATION 2 (from plan/archive/sweep-2026-05-11-fixes-engines.md Item 25):
 *   The deleted implementation used a fixed 4×4 octahedral grid (16 bins).
 *   This module implements an ADAPTIVE quadtree that refines high-flux
 *   directional bins, capable of representing sharp indirect caustics.
 *
 * ADDRESSES DEVIATION 5 (solid-angle weights):
 *   Each dTree leaf stores its exact solid angle computed as
 *     solidAngle = 4π × (u1−u0) × (v1−v0)
 *   because the octahedral map parameterises the full sphere in a 1×1 square
 *   (total area 1 ↔ 4π sr). This replaces the uniform 4π/N used by the
 *   deleted implementation.
 */

import {
  PPG_DTREE_FLUX_FRACTION,
  PPG_DTREE_MERGE_FRACTION,
  PPG_DTREE_MAX_DEPTH,
} from './ppgConstants.js';
import type { DTree, DTreeNode } from './types.js';

const FOUR_PI = 4 * Math.PI;
const MAX_FINITE_F32 = 3.402823466e38;

// ────────────────────────────────────────────────────────────────────────────
// Build
// ────────────────────────────────────────────────────────────────────────────

/**
 * Build a dTree with `initialDepth` levels, covering the full octahedral
 * square [0,1]². At depth 0 there is 1 leaf (the root). At depth d there
 * are 4^d leaves.
 *
 * Each leaf starts with zero flux and its exact solid angle computed from
 * the octahedral patch area (deviation 5 fix, Müller §3.2).
 *
 * @param initialDepth  Number of pre-split levels (0 = root only = 1 leaf,
 *                      1 = 4 leaves, 2 = 16 leaves, …).
 */
export function buildEmptyDTree(initialDepth: number): DTree {
  if (!Number.isSafeInteger(initialDepth)
      || initialDepth < 0
      || initialDepth > PPG_DTREE_MAX_DEPTH) {
    throw new RangeError(
      `initialDepth must be an integer in [0, ${PPG_DTREE_MAX_DEPTH}]; got ${initialDepth}`,
    );
  }
  const nodes: DTreeNode[] = [];
  buildSubtree(nodes, 0, 1, 0, 1, 0, initialDepth);
  return { nodes, totalFlux: 0 };
}

/**
 * Recursively build the quadtree subtree rooted at the current node.
 *
 * **Layout invariant (W9 fix):** for every interior node, its four children
 * are stored at consecutive indices `[firstChild, firstChild+1,
 * firstChild+2, firstChild+3]` (NW, NE, SW, SE). This matches the assumption
 * built into {@link findDTreeLeaf} and into the GPU traversal in
 * `ppgPdf.wgsl.ts`:
 *
 *     idx = node.firstChild + (goDown ? 2 : 0) + (goRight ? 1 : 0)
 *
 * The naive single-pass "push self then recurse" scheme interleaves
 * grandchildren between siblings (root→NW-subtree→NE-subtree means root's
 * NE child lands far past `firstChild+1`), breaking the invariant for any
 * tree with `maxDepth ≥ 2`. We instead use a **two-phase build per level**:
 * push all four children's slots consecutively, THEN recurse into each
 * non-leaf child. This guarantees children are always adjacent.
 *
 * Nodes are appended in BFS-by-level / DFS-by-subtree order: root is at 0,
 * its four children at 1..4, then for each non-leaf child its grandchildren
 * are appended next (sub-tree depth-first).
 */
function buildSubtree(
  nodes: DTreeNode[],
  u0: number, u1: number,
  v0: number, v1: number,
  depth: number,
  maxDepth: number,
): number {
  const idx = nodes.length;
  const isLeaf = depth >= maxDepth;

  // Push this node first (its index is `idx`).
  nodes.push({
    isLeaf,
    u0, v0, u1, v1,
    solidAngle: isLeaf ? FOUR_PI * (u1 - u0) * (v1 - v0) : -1,
    flux: 0,
    firstChild: -1,
    depth,
  });

  if (isLeaf) return idx;

  // Two-phase build: (1) push 4 consecutive children (shared helper),
  // (2) recurse into each non-leaf child to build its sub-tree.
  const { firstChild, childExtents, childIsLeaf } =
    pushFourChildren(nodes, u0, u1, v0, v1, depth, maxDepth);
  nodes[idx]!.firstChild = firstChild;

  if (!childIsLeaf) {
    for (let ci = 0; ci < 4; ci++) {
      const [cu0, cu1, cv0, cv1] = childExtents[ci]!;
      const childIdx = firstChild + ci;
      // The recursive call must build the grandchildren at the current
      // tail (i.e. `nodes.length`). Patch this child's firstChild to point
      // there, then push its 4 grandchildren consecutively.
      nodes[childIdx]!.firstChild = nodes.length;
      buildSubtreeChildrenOnly(nodes, cu0, cu1, cv0, cv1, depth + 1, maxDepth);
    }
  }

  return idx;
}

/**
 * Helper used by Phase 2 — given a parent whose own slot is already pushed,
 * push its four children consecutively AND recurse into each non-leaf child.
 *
 * Distinct from {@link buildSubtree} because the parent slot is NOT pushed
 * here; the caller already pushed it (Phase 1) and is patching its
 * `firstChild`. This avoids the double-push that would mis-align the layout.
 */
function buildSubtreeChildrenOnly(
  nodes: DTreeNode[],
  u0: number, u1: number,
  v0: number, v1: number,
  depth: number,
  maxDepth: number,
): void {
  const { firstChild, childExtents, childIsLeaf } =
    pushFourChildren(nodes, u0, u1, v0, v1, depth, maxDepth);
  if (!childIsLeaf) {
    for (let ci = 0; ci < 4; ci++) {
      const [cu0, cu1, cv0, cv1] = childExtents[ci]!;
      const childIdx = firstChild + ci;
      nodes[childIdx]!.firstChild = nodes.length;
      buildSubtreeChildrenOnly(nodes, cu0, cu1, cv0, cv1, depth + 1, maxDepth);
    }
  }
}

/**
 * Push the four quadrant children of a node consecutively (NW, NE, SW, SE) and
 * return their first index + extents + leaf-ness for the caller's Phase-2
 * recursion. Shared by {@link buildSubtree} (Phase 1) and
 * {@link buildSubtreeChildrenOnly} — both computed identical child extents +
 * pushed four nodes of identical shape; this collapses that duplication.
 *
 * Does NOT recurse and does NOT patch the parent's `firstChild` (the caller
 * does, since the two callers wire it differently). Output-identical to the two
 * inlined push loops it replaces (pinned by `dTreePushFourChildren.test.ts`).
 */
function pushFourChildren(
  nodes: DTreeNode[],
  u0: number, u1: number,
  v0: number, v1: number,
  depth: number,
  maxDepth: number,
): { firstChild: number; childExtents: Array<[number, number, number, number]>; childIsLeaf: boolean } {
  const uMid = (u0 + u1) * 0.5;
  const vMid = (v0 + v1) * 0.5;
  const childExtents: Array<[number, number, number, number]> = [
    [u0,   uMid, v0,   vMid], // NW (offset 0)
    [uMid, u1,   v0,   vMid], // NE (offset 1)
    [u0,   uMid, vMid, v1  ], // SW (offset 2)
    [uMid, u1,   vMid, v1  ], // SE (offset 3)
  ];
  const childIsLeaf = (depth + 1) >= maxDepth;
  const firstChild = nodes.length;
  for (let ci = 0; ci < 4; ci++) {
    const [cu0, cu1, cv0, cv1] = childExtents[ci]!;
    nodes.push({
      isLeaf: childIsLeaf,
      u0: cu0, u1: cu1, v0: cv0, v1: cv1,
      solidAngle: childIsLeaf ? FOUR_PI * (cu1 - cu0) * (cv1 - cv0) : -1,
      flux: 0,
      firstChild: -1,
      depth: depth + 1,
    });
  }
  return { firstChild, childExtents, childIsLeaf };
}

// ────────────────────────────────────────────────────────────────────────────
// Traversal
// ────────────────────────────────────────────────────────────────────────────

/**
 * Find the leaf node in `dTree` that covers the octahedral UV coordinate
 * `octUV ∈ [0,1]²`. Returns the leaf's node index.
 *
 * Direction encoding (deviation 4 fix, Müller §3.2):
 *   The octahedral UV is computed from the WORLD-SPACE direction ωi.
 *   No per-surface ONB transform is applied — all dTree lookups use a
 *   single canonical world frame.
 */
export function findDTreeLeaf(dTree: DTree, octUV: [number, number]): number {
  let idx = 0;
  while (true) {
    const node = dTree.nodes[idx]!;
    if (node.isLeaf) return idx;
    const uMid = (node.u0 + node.u1) * 0.5;
    const vMid = (node.v0 + node.v1) * 0.5;
    const goRight = octUV[0] >= uMid;
    const goDown  = octUV[1] >= vMid;
    idx = node.firstChild + (goDown ? 2 : 0) + (goRight ? 1 : 0);
  }
}

// ────────────────────────────────────────────────────────────────────────────
// Guide: sample a direction from the dTree proportional to leaf flux
// ────────────────────────────────────────────────────────────────────────────

/**
 * Sample an octahedral UV direction from `dTree` proportional to leaf flux.
 *
 * Returns `{ octUV, pdf }` where:
 *   - `octUV` is the sampled direction in octahedral UV space [0,1]²
 *   - `pdf`   is the solid-angle PDF for that direction:
 *       pdf(ω) = (flux_leaf / totalFlux) / solidAngle_leaf
 *
 * DEVIATION 5 FIX: the PDF uses the per-leaf `solidAngle` (exact octahedral
 * patch area × 4π), NOT the uniform 4π/N approximation.
 *
 * @param dTree     The directional tree to sample from.
 * @param u0, u1    Uniform random variables in [0, 1).
 * @param totalFlux Total flux across all leaves (pre-computed from dTree state).
 *
 * @internal CPU reference oracle for the WGSL D-tree sampler; not public API.
 */
export function dTreeSample(
  dTree: DTree,
  u0: number,
  u1: number,
): { octUV: [number, number]; pdf: number } {
  const totalFlux = dTree.totalFlux;

  // Degenerate case: no flux accumulated yet → uniform sample.
  if (totalFlux <= 0) {
    return { octUV: [u0, u1], pdf: 1 / FOUR_PI };
  }

  let idx = 0;
  let remaining = u0 * totalFlux;

  while (true) {
    const node = dTree.nodes[idx]!;
    if (node.isLeaf) {
      // Leaf jitter must be UNIFORM within the leaf rectangle AND INDEPENDENT
      // of the flux-proportional descent path. The GPU production sampler
      // (ppgPdf.wgsl.ts `ppgDTreeSampleLeafBase` -> `ppgSampleGuidedDir`)
      // draws two fresh randoms for the leaf u,v jitter after the descent, so
      // its jitter is fully decorrelated from which leaf was picked. The old
      // CPU oracle instead reused the SAME `u0` that had
      // already been consumed by the descent (`remaining = u0 * totalFlux`,
      // decremented through the tree) for `vSample`, correlating the leaf
      // v-position with the descent path — a divergence from the GPU.
      //
      // FIX (rescaled descent residual — standard hierarchical-sampling
      // decorrelation, Müller §3.2 / pbrt §13.3 inverse-CDF residual reuse):
      // after descent, `remaining` holds the leftover mass WITHIN the chosen
      // leaf's flux interval, i.e. `remaining ∈ [0, leafFlux)`. Rescaling it
      // to [0,1) yields a value that is uniform within the leaf and
      // statistically independent of the coarse path selection (which
      // consumed the high-order bits of `u0`). We use it for `vSample`, so a
      // single (u0,u1) pair still maps to ONE deterministic sample (oracle
      // determinism the tests rely on) while removing the correlation. This
      // matches the GPU's fresh-random leaf jitter in distribution.
      const leafFlux = node.flux;
      const uLeaf = leafFlux > 0
        ? Math.min(remaining / leafFlux, 1 - 1e-7)
        : u0; // cold leaf (zero flux) reached via uniform fallback: keep u0.
      // Sample uniformly within the leaf's octahedral patch. uSample uses the
      // independent `u1` (already correct); vSample uses the decorrelated
      // residual `uLeaf` instead of the descent-consumed `u0`.
      const uSample = node.u0 + u1 * (node.u1 - node.u0);
      const vSample = node.v0 + uLeaf * (node.v1 - node.v0);
      // PDF = (leafFlux / totalFlux) / solidAngle_leaf  (deviation 5 fix) —
      // unchanged by this fix; the jitter decorrelation does not alter the
      // per-leaf solid-angle PDF.
      const representedLeaf = node.flux > 0
        && totalFlux > 0
        && Number.isFinite(node.solidAngle)
        && node.solidAngle > 0;
      const pdf = representedLeaf
        ? (node.flux / totalFlux) / node.solidAngle
        : 1 / FOUR_PI;
      return { octUV: [uSample, vSample], pdf };
    }

    // Traverse children by selecting proportional to accumulated flux.
    const c0 = node.firstChild;
    let cumFlux = 0;
    let chosen = 3; // default to last child
    for (let ci = 0; ci < 4; ci++) {
      cumFlux += dTree.nodes[c0 + ci]!.flux;
      if (remaining < cumFlux) {
        chosen = ci;
        break;
      }
    }
    // Adjust remaining for the next level: subtract the preceding siblings'
    // cumulative flux so `remaining` becomes the residual within the chosen
    // child's flux interval. At the leaf this residual ∈ [0, leafFlux) is the
    // decorrelated jitter rescaled above.
    const chosenFlux = dTree.nodes[c0 + chosen]!.flux;
    remaining -= (cumFlux - chosenFlux);
    idx = c0 + chosen;
  }
}

/**
 * Evaluate the dTree PDF for a given octahedral UV direction.
 * Returns the solid-angle PDF: (leafFlux / totalFlux) / solidAngle_leaf.
 * Returns 1/(4π) if totalFlux ≤ 0 (uniform fall-back).
 *
 * Used for MIS weight computation (Müller §3.4).
 */
export function dTreePdf(dTree: DTree, octUV: [number, number]): number {
  if (dTree.totalFlux <= 0) return 1 / FOUR_PI;
  const leafIdx = findDTreeLeaf(dTree, octUV);
  const leaf = dTree.nodes[leafIdx]!;
  if (!(leaf.flux > 0) || !Number.isFinite(leaf.solidAngle) || !(leaf.solidAngle > 0)) {
    return 1 / FOUR_PI;
  }
  return (leaf.flux / dTree.totalFlux) / leaf.solidAngle;
}

// ────────────────────────────────────────────────────────────────────────────
// Adaptive refinement — split / merge
// ────────────────────────────────────────────────────────────────────────────

/**
 * Refine the dTree after a training frame:
 *   - Split leaves with `flux > PPG_DTREE_FLUX_FRACTION × totalFlux`.
 *   - Merge sibling leaves that are both below the merge threshold.
 *
 * DEVIATION 2 FIX: this is the adaptive refinement described in Müller §3.2,
 * replacing the fixed 4×4 grid of the deleted implementation.
 *
 * The split/merge loop runs on the CPU after reading back GPU atomic counters.
 * New nodes are appended to the flat array; the tree is re-serialised after.
 *
 * @param dTree      The dTree to refine (mutated in place).
 * @param fluxFrac   Split threshold fraction (default `PPG_DTREE_FLUX_FRACTION`).
 * @param mergeFrac  Merge threshold fraction (default `PPG_DTREE_MERGE_FRACTION`).
 * @param maxDepth   Maximum allowed node depth (default `PPG_DTREE_MAX_DEPTH`).
 */
export function refineDTree(
  dTree: DTree,
  fluxFrac: number = PPG_DTREE_FLUX_FRACTION,
  mergeFrac: number = PPG_DTREE_MERGE_FRACTION,
  maxDepth: number = PPG_DTREE_MAX_DEPTH,
  maxNodes: number = Number.POSITIVE_INFINITY,
): void {
  if (!Number.isFinite(fluxFrac) || fluxFrac < 0 || fluxFrac > 1) {
    throw new RangeError(`fluxFrac must be finite and inside [0,1]; got ${fluxFrac}`);
  }
  if (!Number.isFinite(mergeFrac) || mergeFrac < 0 || mergeFrac > 1) {
    throw new RangeError(`mergeFrac must be finite and inside [0,1]; got ${mergeFrac}`);
  }
  if (!Number.isSafeInteger(maxDepth)
      || maxDepth < 0
      || maxDepth > PPG_DTREE_MAX_DEPTH) {
    throw new RangeError(
      `maxDepth must be an integer in [0, ${PPG_DTREE_MAX_DEPTH}]; got ${maxDepth}`,
    );
  }
  if (maxNodes !== Number.POSITIVE_INFINITY
      && (!Number.isSafeInteger(maxNodes) || maxNodes < 1)) {
    throw new RangeError(`maxNodes must be a positive safe integer or Infinity; got ${maxNodes}`);
  }
  const totalFlux = dTree.totalFlux;
  if (!Number.isFinite(totalFlux) || totalFlux < 0 || totalFlux > MAX_FINITE_F32) {
    throw new RangeError(`dTree totalFlux must be finite, non-negative, and f32-representable; got ${totalFlux}`);
  }
  if (totalFlux <= 0) return; // nothing to refine

  const splitThreshold = fluxFrac * totalFlux;
  const mergeThreshold = mergeFrac * totalFlux;

  const initialLen = dTree.nodes.length;
  for (let i = 0; i < initialLen; i++) {
    const node = dTree.nodes[i]!;
    if (!node.isLeaf) continue;
    if (node.depth >= maxDepth) continue;
    if (node.flux <= splitThreshold) continue;
    if (dTree.nodes.length + 4 > maxNodes) break;

    // Split this leaf into 4 children (Müller §3.2).
    splitDTreeLeaf(dTree, i);
  }

  // Merge pass: merge sibling leaf-pairs whose combined flux < mergeThreshold.
  // (Simple pass: traverse interior nodes, merge their 4 children if all leaves
  // and each child flux < mergeThreshold.)
  for (let i = 0; i < dTree.nodes.length; i++) {
    const node = dTree.nodes[i]!;
    if (node.isLeaf) continue;
    const c = node.firstChild;
    if (c < 0) continue;
    let allLeaves = true;
    let maxChildFlux = 0;
    for (let ci = 0; ci < 4; ci++) {
      const child = dTree.nodes[c + ci];
      if (!child || !child.isLeaf) { allLeaves = false; break; }
      if (child.flux > maxChildFlux) maxChildFlux = child.flux;
    }
    if (!allLeaves) continue;
    if (maxChildFlux >= mergeThreshold) continue;

    // Merge: parent becomes a leaf, absorbs children's total flux. The merged
    // children become unreachable (the parent's `firstChild` is cleared); the
    // compaction pass below drops them from the flat array so it can't grow
    // unbounded across refine cycles (A5 fix). We do NOT mutate the children
    // here — `compactDTree` simply never re-emits anything not reachable from
    // the root, which naturally discards them.
    let mergedFlux = 0;
    for (let ci = 0; ci < 4; ci++) {
      mergedFlux += dTree.nodes[c + ci]!.flux;
    }
    node.isLeaf = true;
    node.flux = mergedFlux;
    node.solidAngle = FOUR_PI * (node.u1 - node.u0) * (node.v1 - node.v0);
    node.firstChild = -1;
  }

  // ── Compaction pass (A5 fix) ──────────────────────────────────────────────
  // The split pass appends to `dTree.nodes`; the merge pass orphans whole
  // child blocks. Without compaction the flat array grows unbounded across
  // refine cycles and accumulates orphan nodes — which the GPU flux readback
  // then clamps at MAX_DTREE_NODES_PER_CELL, silently truncating the live
  // tree. Rebuild the array from scratch, re-emitting ONLY nodes reachable
  // from the root and re-patching `firstChild` to the consecutive-children
  // invariant `buildSubtree` documents.
  compactDTree(dTree);
  recomputeDTreeInteriorFlux(dTree);
}

/**
 * Recompute all subtree masses and the total leaf mass after topology or leaf
 * changes. Invalid/non-f32-representable state is rejected instead of silently
 * serialising NaN/Infinity into the GPU guide.
 */
export function recomputeDTreeInteriorFlux(dTree: DTree): void {
  if (dTree.nodes.length === 0) {
    throw new RangeError('cannot recompute flux for an empty dTree');
  }

  // Stage every derived mass before committing. Validation failures therefore
  // leave both node flux and totalFlux byte-for-byte unchanged.
  const staged = new Float64Array(dTree.nodes.length);
  const visitState = new Uint8Array(dTree.nodes.length); // 0=unseen, 1=open, 2=closed
  const stack: Array<{ index: number; expanded: boolean }> = [
    { index: 0, expanded: false },
  ];
  while (stack.length > 0) {
    const { index, expanded } = stack.pop()!;
    const node = dTree.nodes[index];
    if (!node) throw new RangeError(`dTree traversal reached invalid node ${index}`);
    if (!expanded) {
      if (visitState[index] === 1) {
        throw new RangeError(`dTree node ${index} is cyclic`);
      }
      if (visitState[index] === 2) {
        throw new RangeError(`dTree node ${index} is multiply referenced`);
      }
      if (!Number.isFinite(node.flux) || node.flux < 0) {
        throw new RangeError(`dTree node ${index} has invalid flux ${node.flux}`);
      }
      visitState[index] = 1;
      stack.push({ index, expanded: true });
      if (!node.isLeaf) {
        if (
          !Number.isSafeInteger(node.firstChild) ||
          node.firstChild <= index ||
          node.firstChild + 3 >= dTree.nodes.length
        ) {
          throw new RangeError(`dTree interior node ${index} has invalid children`);
        }
        for (let child = 3; child >= 0; child--) {
          stack.push({ index: node.firstChild + child, expanded: false });
        }
      }
      continue;
    }

    let mass = node.flux;
    if (!node.isLeaf) {
      mass = 0;
      for (let child = 0; child < 4; child++) {
        mass += staged[node.firstChild + child]!;
      }
    }
    if (!Number.isFinite(mass) || mass > MAX_FINITE_F32) {
      throw new RangeError(`dTree node ${index} exceeds finite f32 range`);
    }
    staged[index] = mass;
    visitState[index] = 2;
  }
  if (visitState.some((state) => state === 0)) {
    throw new RangeError('dTree contains unreachable nodes');
  }

  for (let index = 0; index < dTree.nodes.length; index++) {
    dTree.nodes[index]!.flux = staged[index]!;
  }
  dTree.totalFlux = staged[0]!;
}

/**
 * Rebuild `dTree.nodes` into a fresh array containing only nodes reachable
 * from the root, with the consecutive-children layout invariant restored:
 * every interior node's four children occupy `[firstChild, firstChild+3]`.
 *
 * Live-leaf state (flux, solidAngle, octahedral extents, depth) is copied
 * verbatim, so sampling behaviour (`dTreeSample`, `dTreePdf`, `findDTreeLeaf`)
 * is unchanged — only orphaned / merged-away nodes are dropped and indices
 * are renumbered.
 *
 * BFS allocation guarantees consecutive children: we pop a node off a queue,
 * append it, and when it is interior we reserve its four child slots
 * consecutively (recording the old child indices to process next). This
 * mirrors the two-phase scheme in {@link buildSubtree}.
 */
function compactDTree(dTree: DTree): void {
  const old = dTree.nodes;
  if (old.length === 0) return;

  const fresh: DTreeNode[] = [];
  // Queue holds the OLD index of a node plus the NEW index it was placed at,
  // so when we expand an interior node we can patch its new `firstChild`.
  const queue: Array<{ oldIdx: number; newIdx: number }> = [];

  // Place the root first.
  fresh.push(copyNode(old[0]!));
  queue.push({ oldIdx: 0, newIdx: 0 });

  let head = 0;
  while (head < queue.length) {
    const { oldIdx, newIdx } = queue[head++]!;
    const oldNode = old[oldIdx]!;
    if (oldNode.isLeaf || oldNode.firstChild < 0) {
      // Leaf (or defensively-malformed) node — ensure it reads as a leaf in
      // the compacted tree and carries no dangling child pointer.
      fresh[newIdx]!.isLeaf = true;
      fresh[newIdx]!.firstChild = -1;
      continue;
    }
    // Interior node: reserve four consecutive child slots in the fresh array.
    const newFirstChild = fresh.length;
    fresh[newIdx]!.firstChild = newFirstChild;
    const oldFirstChild = oldNode.firstChild;
    for (let ci = 0; ci < 4; ci++) {
      const oldChildIdx = oldFirstChild + ci;
      fresh.push(copyNode(old[oldChildIdx]!));
      queue.push({ oldIdx: oldChildIdx, newIdx: newFirstChild + ci });
    }
  }

  dTree.nodes = fresh;
}

/** Shallow copy of a DTreeNode (all fields are primitives). */
function copyNode(n: DTreeNode): DTreeNode {
  return {
    isLeaf: n.isLeaf,
    u0: n.u0, v0: n.v0, u1: n.u1, v1: n.v1,
    solidAngle: n.solidAngle,
    flux: n.flux,
    firstChild: n.firstChild,
    depth: n.depth,
  };
}

/** Split a single dTree leaf into 4 children (Müller §3.2). */
function splitDTreeLeaf(dTree: DTree, nodeIdx: number): void {
  const node = dTree.nodes[nodeIdx]!;
  const uMid = (node.u0 + node.u1) * 0.5;
  const vMid = (node.v0 + node.v1) * 0.5;
  const childDepth = node.depth + 1;
  // Distribute parent flux equally across 4 children.
  const childFlux = node.flux * 0.25;

  const firstChild = dTree.nodes.length;
  // NW, NE, SW, SE
  for (const [cu0, cu1, cv0, cv1] of [
    [node.u0, uMid, node.v0, vMid],
    [uMid, node.u1, node.v0, vMid],
    [node.u0, uMid, vMid, node.v1],
    [uMid, node.u1, vMid, node.v1],
  ] as Array<[number, number, number, number]>) {
    dTree.nodes.push({
      isLeaf: true,
      u0: cu0, u1: cu1,
      v0: cv0, v1: cv1,
      solidAngle: FOUR_PI * (cu1 - cu0) * (cv1 - cv0),
      flux: childFlux,
      firstChild: -1,
      depth: childDepth,
    });
  }

  // Promote parent to interior.
  node.isLeaf = false;
  node.solidAngle = -1;
  node.flux = 0;
  node.firstChild = firstChild;
}

// ────────────────────────────────────────────────────────────────────────────
// Solid-angle invariant helper (used in tests)
// ────────────────────────────────────────────────────────────────────────────

/**
 * Sum the solid angles of all LEAF nodes in a dTree.
 * For a complete tree covering the full sphere this must equal 4π.
 *
 * Used by Test 3 (solid-angle invariant).
 */
/**
 * Descend `dTree` to the leaf covering `octUV` and accumulate `flux` there.
 *
 * Mirrors the descent in {@link findDTreeLeaf}. Extracted from `sTree.ts` so
 * the traversal lives once; if this changes, nothing else needs updating.
 */
export function dTreeAccumulateFlux(
  dTree: DTree,
  octUV: [number, number],
  flux: number,
): void {
  if (!Number.isFinite(flux) || flux < 0) {
    throw new RangeError(`flux must be finite and non-negative; got ${flux}`);
  }
  if (
    !Number.isFinite(octUV[0]) ||
    !Number.isFinite(octUV[1]) ||
    octUV[0] < 0 || octUV[0] > 1 || octUV[1] < 0 || octUV[1] > 1
  ) {
    throw new RangeError(
      `octUV must be finite and inside [0,1]^2; got ${octUV.join(',')}`,
    );
  }
  if (dTree.nodes.length === 0) {
    throw new RangeError('cannot accumulate flux into an empty dTree');
  }

  // Resolve and validate the complete path before mutating anything. This keeps
  // failed deposits transactional and prevents a malformed graph from leaving
  // only some ancestors updated.
  const path: number[] = [];
  const seen = new Set<number>();
  let idx = 0;
  while (true) {
    const node = dTree.nodes[idx];
    if (!node || seen.has(idx)) {
      throw new RangeError(`dTree traversal reached invalid or cyclic node ${idx}`);
    }
    seen.add(idx);
    path.push(idx);
    if (!Number.isFinite(node.flux) || node.flux < 0 || node.flux + flux > MAX_FINITE_F32) {
      throw new RangeError(`dTree node ${idx} flux would exceed finite f32 range`);
    }
    if (node.isLeaf) break;
    if (node.firstChild < 0 || node.firstChild + 3 >= dTree.nodes.length) {
      throw new RangeError(`dTree node ${idx} has invalid children`);
    }
    const uMid = (node.u0 + node.u1) * 0.5;
    const vMid = (node.v0 + node.v1) * 0.5;
    const goRight = octUV[0] >= uMid;
    const goDown = octUV[1] >= vMid;
    idx = node.firstChild + (goDown ? 2 : 0) + (goRight ? 1 : 0);
  }
  if (!Number.isFinite(dTree.totalFlux) || dTree.totalFlux < 0 || dTree.totalFlux + flux > MAX_FINITE_F32) {
    throw new RangeError('dTree total flux would exceed finite f32 range');
  }
  for (const nodeIndex of path) dTree.nodes[nodeIndex]!.flux += flux;
  dTree.totalFlux += flux;
}

export function sumLeafSolidAngles(dTree: DTree): number {
  let sum = 0;
  for (const node of dTree.nodes) {
    if (node.isLeaf && node.solidAngle > 0) {
      sum += node.solidAngle;
    }
  }
  return sum;
}

/**
 * Sum the PDFs of all leaf nodes evaluated at the centre of each leaf.
 * For a valid distribution this must equal 1 (over the sphere) when
 * each leaf PDF is `(flux/total) / solidAngle` and integrated over the leaf.
 *
 * Used by Test 4 (guide PDF normalization).
 * Computed as Σ (flux_i / total) = 1 when all flux is accounted for.
 */
export function sumLeafPdfIntegrals(dTree: DTree): number {
  const total = dTree.totalFlux;
  if (total <= 0) return 1; // degenerate: uniform PDF integrates to 1.
  let sum = 0;
  for (const node of dTree.nodes) {
    if (node.isLeaf && node.solidAngle > 0) {
      // pdf(ω) × solidAngle = flux_i / total (the probability mass of this leaf)
      sum += node.flux / total;
    }
  }
  return sum;
}
