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
 * build kernel mirrors 1:1. It is correctness-critical: the per-cell selection
 * pmf computed here is the EXACT pmf the RIS source weight `w = p̂ / p_source`
 * divides by, so it MUST be a valid pmf (sum to 1 over the emitter set in
 * expectation) or the ReSTIR estimator is biased.
 *
 * The grid is "seeded by the light tree" — its candidates are drawn via the
 * light-tree traversal (`sampleLightTreeCPU`) at the cell centroid and weighted
 * to a target that matches the tree's per-leaf importance. The tree build +
 * traversal live in `lightTree.ts`; this module imports them.
 *
 * Unbiasedness construction (matches the light-tree discipline in lightTree.ts):
 *   - The cell target is `q̂_c(e) = power_e / max(dist²(x_c, e.aabb), floor)`
 *     for the cell centroid `x_c` — power × spatial proximity, NO BRDF (the
 *     per-pixel receiver BRDF is unknown at grid-build time; the BRDF enters
 *     later, in the RIS p̂). `q̂_c` is exactly the *leaf importance* the light
 *     tree already uses, so the grid is "seeded by the light tree" in the sense
 *     that the per-cell candidates are drawn via `sampleLightTreeCPU` at `x_c`
 *     and weighted to the same target the tree's descent approximates.
 *   - One sub-reservoir runs WRS over `M` tree draws: candidate `e_i` is drawn
 *     with source pdf `p_tree(e_i | x_c)`, RIS weight `w_i = q̂_c(e_i) /
 *     p_tree(e_i | x_c)`. The survivor `e*` is (in expectation) distributed
 *     ∝ `q̂_c`. The reservoir's running `wSum / M` is an unbiased estimate
 *     `Ŝ` of the cell's total target mass `S_c = Σ_e q̂_c(e)`.
 *   - The survivor's EFFECTIVE selection pmf is therefore
 *         pSel(e*) = q̂_c(e*) / Ŝ  =  q̂_c(e*) · M / wSum,
 *     the standard RIS relation (Bitterli 2020 §3): a WRS reservoir is an
 *     importance sampler whose effective pdf is `target / normalisation-estimate`.
 *     Σ_e pSel(e) → 1 in expectation (a valid pmf), which the tests assert.
 *   - A cell stores `K` independent sub-reservoirs. RIS picks one uniformly and
 *     uses ITS `pSel` as the source pmf; the `1/K` does not enter the weight
 *     because RIS draws ONE candidate (it is not summing over the K reservoirs).
 *     The K survivors give per-pixel candidate diversity without re-running the
 *     tree descent per pixel.
 */

import { dist2ToAabb, nodeImportance, sampleLightTreeCPU, type LightTreeNode } from './lightTree.js';

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
