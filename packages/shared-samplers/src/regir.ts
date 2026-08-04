/**
 * regir.ts — ReGIR (Reservoir-based Grid Importance Resampling) CPU reference core.
 *
 * ReGIR (Boksansky, Wyman, Benty 2021, "Rendering Many Lights with Grid-Based
 * Reservoirs", Ray Tracing Gems II ch. 23) decouples the per-pixel light-
 * selection cost from the light count by pre-resampling lights into a world-
 * space grid of reservoirs ONCE per frame. ReSTIR-DI then draws its initial
 * candidates from the grid cell containing the shading point instead of
 * traversing the light tree per pixel.
 *
 * This module is the CPU reference for the per-cell WRS that the WGSL grid-
 * build kernel mirrors. It is correctness-critical: finite 24-bit branch
 * decisions do not realise the ideal WRS probability exactly. Each survivor
 * therefore stores the base-2 log of its represented effective source density.
 *
 * The grid is "seeded by the light tree" — its candidates are drawn via the
 * packed light-tree traversal at the cell centroid and weighted
 * to a target that matches the tree's per-leaf importance. The tree build +
 * traversal live in `lightTree.ts`; this module imports them.
 *
 * Represented unbiasedness construction:
 *   - The cell target is `q̂_c(e) = power_e / max(dist²(x_c, e.aabb), floor)`
 *     for the cell centroid `x_c` — power × spatial proximity, NO BRDF (the
 *     per-pixel receiver BRDF is unknown at grid-build time; the BRDF enters
 *     later, in the RIS p̂). `q̂_c` is the packed leaf importance, with a
 *     minimum-normal support floor when a positive-power leaf's cone target is
 *     zero at the centroid. The grid is "seeded by the light tree" in the sense
 *     that the per-cell candidates are drawn via `sampleLightTreeCPU` at `x_c`
 *     and weighted to the same target the tree's descent approximates.
 *   - One sub-reservoir runs WRS over `M` tree draws: candidate `e_i` is drawn
 *     with source pdf `p_tree(e_i | x_c)`, RIS weight `w_i = q̂_c(e_i) /
 *     p_tree(e_i | x_c)`. The ideal WRS ratio is represented by non-empty
 *     integer branch buckets; the actual occurrence probability is tracked.
 *   - Let `r_i` be the selected occurrence's actual conditional probability
 *     under the integer branch buckets, including every later keep decision.
 *     The stored effective density is `pSel_i = M · r_i · p_tree_i`.
 *     Therefore `Σ_i r_i F_i / pSel_i = (1/M) Σ_i F_i / p_tree_i` for
 *     every realised candidate batch. This is the finite-RNG
 *     Horvitz–Thompson correction, not the ideal `q̂ / Ŝ` approximation.
 *   - A cell stores `K` independent sub-reservoirs. RIS picks one uniformly and
 *     uses ITS `pSel` as the source density; the `1/K` does not enter the weight
 *     because RIS draws ONE candidate (it is not summing over the K reservoirs).
 *     The K survivors give per-pixel candidate diversity without re-running the
 *     tree descent per pixel.
 */

import {
  LIGHT_TREE_FLOATS_PER_NODE,
  packLightTreeForGPU,
  packedLightTreeNodeImportanceCPU,
  samplePackedLightTreeCPU,
  type LightTreeNode,
} from './lightTree.js';
import {
  createRepresentedWrsStateF32,
  representedWrsLogSelectionProbability,
  updateRepresentedWrsF32,
} from './representedWrs.js';
import {
  requireFiniteVec3,
  requireInteger,
  requireNonNegative,
  requirePositive,
} from './numericGuards.js';

/** Floats per ReGIR cell-reservoir survivor slot in the packed grid buffer:
 *  [0] emitterIndex (as f32; -1 ⇒ empty slot), [1] log2PSel (base-2 log of
 *  the represented effective source density). The grid buffer is a flat `array<f32>`
 *  of `numCells × REGIR_SURVIVORS_PER_CELL × REGIR_FLOATS_PER_SURVIVOR`. */
export const REGIR_FLOATS_PER_SURVIVOR = 2;

/** Maximum serial candidate loop shared by the CPU reference and GPU host. */
export const REGIR_MAX_CANDIDATES_PER_CELL = 4096;

/** Invalid/empty density-lane sentinel. Emptiness itself is identified only by
 * `emitterIndex === -1`, so valid negative log densities remain valid. */
export const REGIR_LOG2_PSEL_INVALID = Math.fround(-3.4028234663852886e38);

const F32_MIN_NORMAL = 1.1754943508222875e-38;
const F32_MAX = Math.fround(3.4028234663852886e38);

/** A ReGIR survivor plus its represented effective log density. Empty slots
 * are identified by `emitterIndex === -1`, not by the sign of the log lane. */
export interface ReGIRSurvivor {
  readonly emitterIndex: number;
  /** `log2(M * representedOccurrenceProbability * selectedTreePmf)`. */
  readonly log2PSel: number;
}

/**
 * Run ONE per-cell WRS sub-reservoir over the light tree, seeded at the cell
 * centroid `x_c`, and return the survivor + its represented log density.
 *
 * Mirrors the WGSL `regir_build_survivor` byte-for-byte. The `q̂_c` target is
 * `leafImportance(e) = power_e / max(dist²(x_c, e.aabb), floor)` — recovered
 * from the tree leaf for the chosen emitter — and the source pdf is the tree's
 * own `p_tree(e | x_c)`. The returned `log2PSel` is the finite-branch
 * correction described in the module header.
 *
 * Degenerate cells return `{ emitterIndex: -1, log2PSel:
 * REGIR_LOG2_PSEL_INVALID }`. `M` is the candidate count per sub-reservoir.
 *
 * @param leafImportanceOf - maps emitterIndex → q̂_c for this cell. Provided by
 *   the caller so the CPU port and the WGSL kernel share the SAME target
 *   (the WGSL recomputes it from the emitter list; the CPU port can pass a
 *   tree-leaf lookup). It must be finite and non-negative. The builder applies
 *   the minimum-normal support floor for every positive packed leaf.
 */
export function regirBuildSurvivorCPU(
  nodes: ReadonlyArray<LightTreeNode>,
  xc: readonly [number, number, number],
  dist2Floor: number,
  M: number,
  leafImportanceOf: (emitterIndex: number) => number,
  rand01: () => number,
): ReGIRSurvivor {
  requireFiniteVec3(xc, 'regirBuildSurvivorCPU.xc');
  requirePositive(dist2Floor, 'regirBuildSurvivorCPU.dist2Floor');
  requireInteger(M, 'regirBuildSurvivorCPU.M', 1, REGIR_MAX_CANDIDATES_PER_CELL);
  // Pack once for the complete candidate batch. The public nodes[] wrapper
  // intentionally repacks on every call because ReadonlyArray is not runtime
  // immutability and identity-caching it can publish stale sampling semantics.
  const packedTree = packLightTreeForGPU(nodes);
  const positivePowerEmitters = new Set<number>();
  for (let base = 0; base < packedTree.length; base += LIGHT_TREE_FLOATS_PER_NODE) {
    const emitterIndex = packedTree[base + 0]!;
    if (emitterIndex >= 0 && packedTree[base + 1]! > 0) {
      positivePowerEmitters.add(emitterIndex);
    }
  }
  const wrs = createRepresentedWrsStateF32();
  let chosen = -1;
  let chosenTreePmf = 0;
  for (let i = 0; i < M; i++) {
    const draw = samplePackedLightTreeCPU(packedTree, xc, dist2Floor, rand01);
    if (draw.emitterIndex < 0 || draw.pdf <= 0) continue;
    const rawQHat = leafImportanceOf(draw.emitterIndex);
    requireNonNegative(rawQHat, `regirBuildSurvivorCPU.qHat(${draw.emitterIndex})`);
    if (!positivePowerEmitters.has(draw.emitterIndex)) continue;
    // A represented tree leaf must remain a valid WRS candidate even when its
    // orientation-cone target is exactly zero at this cell centroid.
    const qHat = Math.max(F32_MIN_NORMAL, Math.min(F32_MAX, Math.fround(rawQHat)));
    const treePmf = Math.fround(draw.pdf);
    const logWeight = Math.fround(Math.log2(qHat) - Math.log2(treePmf));
    if (updateRepresentedWrsF32(wrs, logWeight, rand01)) {
      chosen = draw.emitterIndex;
      chosenTreePmf = treePmf;
    }
  }
  if (chosen < 0 || !wrs.hasSelection || !(chosenTreePmf > 0)) {
    return { emitterIndex: -1, log2PSel: REGIR_LOG2_PSEL_INVALID };
  }
  // Compose in binary64 from the canonical compensated occurrence log, then
  // round exactly once for the one-f32 persistent grid ABI.
  const log2PSel = Math.fround(
    Math.log2(M) + representedWrsLogSelectionProbability(wrs) + Math.log2(chosenTreePmf),
  );
  if (!Number.isFinite(log2PSel) || log2PSel <= REGIR_LOG2_PSEL_INVALID) {
    throw new RangeError('regirBuildSurvivorCPU represented log density is invalid');
  }
  return { emitterIndex: chosen, log2PSel };
}

/**
 * Deterministic represented cell target for the leaf carrying `emitterIndex`.
 * It is the packed light-tree importance when positive, and minimum-normal f32
 * when a positive-power leaf's cone is zero at the centroid. This ReGIR-only
 * floor prevents a reachable tree outcome from becoming a zero-weight WRS
 * candidate. Returns 0 for missing or zero-power leaves.
 *
 * The WGSL kernel recovers the same target from the selected packed leaf; this
 * CPU helper lets the reference port do so from only the tree + centroid.
 */
export function regirCellTargetFromTree(
  nodes: ReadonlyArray<LightTreeNode>,
  xc: readonly [number, number, number],
  dist2Floor: number,
): (emitterIndex: number) => number {
  requireFiniteVec3(xc, 'regirCellTargetFromTree.xc');
  requirePositive(dist2Floor, 'regirCellTargetFromTree.dist2Floor');
  // Map emitterIndex → leaf node for O(1) lookup.
  const leafByEmitter = new Map<number, number>();
  for (let nodeIndex = 0; nodeIndex < nodes.length; nodeIndex++) {
    const n = nodes[nodeIndex]!;
    if (n.leftChild < 0 && n.rightChild < 0 && n.emitterIndex >= 0) {
      leafByEmitter.set(n.emitterIndex, nodeIndex);
    }
  }
  const packed = packLightTreeForGPU(nodes);
  return (emitterIndex: number): number => {
    if (!Number.isSafeInteger(emitterIndex) || emitterIndex < 0) return 0;
    const leafNodeIndex = leafByEmitter.get(emitterIndex);
    if (leafNodeIndex == null) return 0;
    if (!(packed[leafNodeIndex * LIGHT_TREE_FLOATS_PER_NODE + 1]! > 0)) return 0;
    return Math.max(
      F32_MIN_NORMAL,
      packedLightTreeNodeImportanceCPU(packed, leafNodeIndex, xc, dist2Floor),
    );
  };
}

/**
 * Normalized represented cell-target diagnostic `q̂_c(e) / Σ q̂_c` over tree
 * leaves. It describes the target weights supplied to represented WRS; it is
 * not the exact finite-batch survivor distribution, and `log2PSel` is an
 * occurrence correction rather than an estimate of this map.
 *
 * When at least one leaf has positive represented power, the returned
 * `Map<emitterIndex, pmf>` sums to 1. An all-zero tree returns zero for every
 * leaf, matching the empty-survivor path.
 */
export function regirCellPmfExact(
  nodes: ReadonlyArray<LightTreeNode>,
  xc: readonly [number, number, number],
  dist2Floor: number,
): Map<number, number> {
  const target = regirCellTargetFromTree(nodes, xc, dist2Floor);
  const emitterSet = new Set<number>();
  for (const n of nodes) {
    if (n.leftChild < 0 && n.rightChild < 0 && n.emitterIndex >= 0) {
      emitterSet.add(n.emitterIndex);
    }
  }
  const emitters = [...emitterSet];
  let maxQ = 0;
  const qHat = new Map<number, number>();
  for (const e of emitters) {
    const q = target(e);
    qHat.set(e, q);
    maxQ = Math.max(maxQ, q);
  }
  let scaledSum = 0;
  if (maxQ > 0) {
    for (const e of emitters) scaledSum += qHat.get(e)! / maxQ;
  }
  const pmf = new Map<number, number>();
  for (const e of emitters) {
    pmf.set(e, maxQ > 0 ? qHat.get(e)! / maxQ / scaledSum : 0);
  }
  return pmf;
}
