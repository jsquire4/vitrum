/**
 * PPG constants — Müller et al. 2017 "Practical Path Guiding for Efficient
 * Light-Transport Simulation", Eurographics Symposium on Rendering 2017.
 * https://tom94.net/data/publications/mueller17practical/mueller17practical.pdf
 *
 * These are the paper-default values. All are exposed as named constants so
 * hosts can tune them without forking shader source.
 */

/**
 * sTree split threshold — number of samples accumulated in a leaf cell
 * before it is split along its longest AABB axis (Müller §3.1).
 *
 * Paper default: 4 000 samples. The prompt specifies 12 000; we use 12 000
 * as the exported default while keeping 4 000 noted for reference.
 *
 * Scene-scale-sensitive: sparser scenes may need a lower threshold to get
 * meaningful splits; very dense scenes can afford higher values.
 */
export const PPG_CELL_SPLIT_THRESHOLD = 12_000;

/**
 * dTree flux fraction threshold — a quadtree leaf is split when its
 * accumulated flux exceeds this fraction of the total cell flux (Müller §3.2).
 *
 * Paper default: 0.01 (any leaf carrying > 1% of total cell flux is split).
 */
export const PPG_DTREE_FLUX_FRACTION = 0.01;

/**
 * dTree merge threshold — a leaf is merged when its flux drops below this
 * fraction of total cell flux AND the leaf is at depth > 1 (Müller §3.2).
 */
export const PPG_DTREE_MERGE_FRACTION = 0.001;

/**
 * Maximum dTree depth (leaf count = 4^depth = 256 at depth 4).
 * Configurable; paper suggests depth 8 (256 leaves) as a practical maximum.
 */
export const PPG_DTREE_MAX_DEPTH = 4;

/**
 * Initial number of dTree depth levels at tree creation (2 levels → 4 leaves).
 * Müller §3.2: start with a small fixed number and refine adaptively.
 */
export const PPG_DTREE_INITIAL_DEPTH = 2;

/**
 * MIS mixing weight α — fraction of the guiding PDF used in the
 * mixture `p_mixed = α·p_guide + (1−α)·p_bsdf` (Müller §3.4).
 *
 * Fixed at 0.5 for a first implementation. Paper describes a variance-
 * reduction-driven update; the variance-adaptive version is tracked in
 * plan/sprint-ppg-rebuild-future.md.
 */
export const PPG_MIS_ALPHA = 0.5;

/** Maximum number of sTree spatial cells (host-configurable default). */
export const PPG_MAX_SPATIAL_CELLS = 16_384;
