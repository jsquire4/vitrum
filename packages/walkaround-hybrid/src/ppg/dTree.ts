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
 * ADDRESSES DEVIATION 2 (from plan/sweep-2026-05-11-fixes-engines.md Item 25):
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
  const nodes: DTreeNode[] = [];
  buildSubtree(nodes, 0, 1, 0, 1, 0, initialDepth);
  return { nodes, totalFlux: 0 };
}

/**
 * Recursively build the quadtree subtree rooted at the current node.
 * Nodes are appended in pre-order DFS (root, then NW, NE, SW, SE children).
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
  const uMid = (u0 + u1) * 0.5;
  const vMid = (v0 + v1) * 0.5;

  // Solid angle = 4π × patch area in octahedral square (deviation 5 fix).
  const solidAngle = isLeaf ? FOUR_PI * (u1 - u0) * (v1 - v0) : -1;

  // Reserve the slot, fill firstChild after children are built.
  nodes.push({
    isLeaf,
    u0, v0, u1, v1,
    solidAngle,
    flux: 0,
    firstChild: -1,
    depth,
  });

  if (!isLeaf) {
    const firstChild = nodes.length;
    nodes[idx]!.firstChild = firstChild;
    buildSubtree(nodes, u0, uMid, v0, vMid, depth + 1, maxDepth); // NW
    buildSubtree(nodes, uMid, u1, v0, vMid, depth + 1, maxDepth); // NE
    buildSubtree(nodes, u0, uMid, vMid, v1, depth + 1, maxDepth); // SW
    buildSubtree(nodes, uMid, u1, vMid, v1, depth + 1, maxDepth); // SE
  }

  return idx;
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
      // Sample uniformly within the leaf's octahedral patch.
      const uSample = node.u0 + u1 * (node.u1 - node.u0);
      const vSample = node.v0 + u0 * (node.v1 - node.v0);
      // PDF = (leafFlux / totalFlux) / solidAngle_leaf  (deviation 5 fix)
      const pdf = (node.flux > 0 && totalFlux > 0)
        ? (node.flux / totalFlux) / node.solidAngle
        : 1 / FOUR_PI;
      return { octUV: [uSample, vSample], pdf: Math.max(pdf, 1e-12) };
    }

    // Traverse children by selecting proportional to accumulated flux.
    const c0 = node.firstChild;
    let cumFlux = 0;
    let chosen = 3; // default to last child
    for (let ci = 0; ci < 4; ci++) {
      cumFlux += dTree.nodes[c0 + ci]!.flux;
      if (remaining <= cumFlux) {
        chosen = ci;
        break;
      }
    }
    // Adjust remaining for the next level.
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
): void {
  const totalFlux = dTree.totalFlux;
  if (totalFlux <= 0) return; // nothing to refine

  const splitThreshold = fluxFrac * totalFlux;
  const mergeThreshold = mergeFrac * totalFlux;

  const initialLen = dTree.nodes.length;
  for (let i = 0; i < initialLen; i++) {
    const node = dTree.nodes[i]!;
    if (!node.isLeaf) continue;
    if (node.depth >= maxDepth) continue;
    if (node.flux <= splitThreshold) continue;

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

    // Merge: parent becomes a leaf, absorbs children's total flux.
    let mergedFlux = 0;
    for (let ci = 0; ci < 4; ci++) {
      mergedFlux += dTree.nodes[c + ci]!.flux;
      // Mark children as defunct (we can't remove from flat array cheaply).
      dTree.nodes[c + ci]!.isLeaf = false;
      dTree.nodes[c + ci]!.firstChild = -1;
    }
    node.isLeaf = true;
    node.flux = mergedFlux;
    node.solidAngle = FOUR_PI * (node.u1 - node.u0) * (node.v1 - node.v0);
    node.firstChild = -1;
  }
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
