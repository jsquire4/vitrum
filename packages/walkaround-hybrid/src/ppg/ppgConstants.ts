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
 * MIS mixing weight α — fraction of the guiding PDF used in the source-pdf
 * mixture `p_src = α·p_guide + (1−α)·p_cos` (Müller §3.4). This is consumed
 * live by the gi-ris guided sampling path: `risGi.wgsl` draws candidates from
 * the learned dTree and weights them by this α-mixture (see PPGCoordinator's
 * `misAlpha` UBO field).
 *
 * Set to 0.5 (equal guide/cosine weighting). A variance-reduction-driven
 * adaptive update is described in the paper; that variant is tracked in
 * plan/sprint-ppg-rebuild-future.md.
 */
export const PPG_MIS_ALPHA = 0.5;

/** Maximum number of sTree spatial cells (host-configurable default). */
export const PPG_MAX_SPATIAL_CELLS = 16_384;

/**
 * ReSTIR-GI reservoir stride — number of u32 elements per reservoir in the
 * flat array<u32> buffer. Single source of truth shared between:
 *   - `ppgGuide.wgsl.ts` (interpolated as RESERVOIR_GI_STRIDE_LOCAL in WGSL)
 *   - `createRestirGIFrameResources.ts` (used to compute the buffer byte size)
 *
 * Must stay in lockstep with `const RESERVOIR_GI_STRIDE: u32 = 30u;` in
 * `shaders/reservoirGi.wgsl.ts` (the WGSL side of the same constant).
 * GRIS Phase-0 widened this from 20 → 30 (Sprint-16/17 fields + reconnection
 * shift cache at indices [20..29]). See `shaders/reservoirGi.wgsl.ts` header.
 */
export const RESERVOIR_GI_STRIDE = 30;
