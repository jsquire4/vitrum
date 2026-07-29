/**
 * PPG constants — Müller et al. 2017 "Practical Path Guiding for Efficient
 * Light-Transport Simulation", Eurographics Symposium on Rendering 2017.
 * https://tom94.net/data/publications/mueller17practical/mueller17practical.pdf
 *
 * All values are exposed as named constants so hosts can tune them without
 * forking shader source.
 */

/**
 * sTree split threshold — number of samples accumulated in a leaf cell
 * before it is split along its longest AABB axis (Müller §3.1).
 *
 * Set to 12 000 samples (Müller §3.1 uses 4 000 as a reference point; vitrum
 * uses a higher threshold to favour fewer, better-populated cells).
 *
 * Scene-scale-sensitive: sparser scenes may need a lower threshold to get
 * meaningful splits; very dense scenes can afford higher values.
 */
export const PPG_CELL_SPLIT_THRESHOLD = 12_000;

/**
 * dTree flux fraction threshold — a quadtree leaf is split when its
 * accumulated flux exceeds this fraction of the total cell flux (Müller §3.2).
 *
 * Set to 0.01: any leaf carrying > 1% of total cell flux is split.
 */
export const PPG_DTREE_FLUX_FRACTION = 0.01;

/**
 * dTree merge threshold — a leaf is merged when its flux drops below this
 * fraction of total cell flux AND the leaf is at depth > 1 (Müller §3.2).
 */
export const PPG_DTREE_MERGE_FRACTION = 0.001;

/**
 * Maximum dTree depth. The directional quadtree subdivides into 4 children per
 * level, so a depth-`d` tree holds at most 4^d leaves — depth 4 ⇒ 256 leaves.
 * Configurable; deeper trees resolve sharper directional peaks at higher cost.
 */
export const PPG_DTREE_MAX_DEPTH = 4;

/**
 * Initial number of dTree depth levels at tree creation (2 levels → 4 leaves).
 * Müller §3.2: start with a small fixed number and refine adaptively.
 */
export const PPG_DTREE_INITIAL_DEPTH = 2;

/**
 * Default maximum number of adaptive sTree leaf cells allocated by the PPG
 * GPU buffers. This is the public `HybridEngineOptions.ppgMaxSpatialCells`
 * default and the compatibility value stored in GI snapshots when hosts do
 * not override the cap.
 */
export const PPG_DEFAULT_SPATIAL_CELLS = 1_024;

/** Public hard ceiling for the spatial-cell allocation. */
export const PPG_MAX_SPATIAL_CELLS = 16_384;

/**
 * Default maximum number of dTree nodes allocated per spatial cell.
 *
 * Depth-4 directional quadtree capacity:
 *   1 + 4 + 16 + 64 + 256 = 341 nodes.
 *
 * The same value must size `allocatePPGResources`, template the PPG update
 * WGSL stride, clamp CPU tree uploads, and gate GI snapshot restores.
 */
export const PPG_DEFAULT_MAX_DTREE_NODES_PER_CELL = 341;

/**
 * Maximum useful per-cell node count at the compiled depth-four limit.
 * Accepting a larger allocation only wastes storage: the CPU refiner cannot
 * make more than `1 + 4 + 16 + 64 + 256` reachable nodes.
 */
export const PPG_MAX_DTREE_NODES_PER_CELL = 341;

/**
 * MIS mixing weight α — fraction of the guiding PDF used in the source-pdf
 * mixture `p_src = α·p_guide + (1−α)·p_cos` (Müller §3.4). This is consumed
 * live by the gi-ris guided sampling path: `risGi.wgsl` draws candidates from
 * the learned dTree and weights them by this α-mixture (see PPGCoordinator's
 * `misAlpha` UBO field).
 *
 * Fixed at 0.5 (equal guide/cosine weighting). This stable defensive mixture
 * keeps cosine-proposal support wherever the diffuse target is non-zero while
 * still exploiting the learned guide; both proposal PDFs are evaluated for
 * every chosen direction before the importance weight is formed.
 */
export const PPG_MIS_ALPHA = 0.5;

/**
 * Per-window flux decay factor (Müller §5 — "keep the SD-tree across iterations").
 *
 * Applied to the persistent CPU flux accumulator at the start of each training
 * window: `flux ← decay·flux + newWindowFlux`. Under steady input this gives the
 * bounded geometric steady state `F/(1−decay)` (0.5 ⇒ 2F), retaining temporal
 * history (lower variance than a hard reset) while provably bounding total flux.
 * `0` reproduces the old full-reset; `1` is the divergent no-decay regime that
 * caused the filed refine-loop runaway. See `sTree.decayAccumulators` +
 * `__tests__/ppgSpatialSplitAndRunaway.test.ts`.
 */
export const PPG_FLUX_DECAY = 0.5;
