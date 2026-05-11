/**
 * PPG (Path Guiding) data structure types — Sprint 11.
 *
 * Implements the Müller 2017 online path guiding scheme for the walkaround
 * WebGPU engine. Walkaround-only; WebGL2 PT has no compute shaders and
 * cannot maintain the kd-tree update pass.
 *
 * References:
 *   - Müller et al. 2017, "Practical Path Guiding for Efficient Light-Transport
 *     Simulation", Computer Graphics Forum 36(4).
 *     https://research.nvidia.com/publication/2017-07_practical-path-guiding-efficient-light-transport-simulation
 *
 * GPU layout decisions:
 *   - Spatial kd-tree: dense linear array, capped at PPG_MAX_SPATIAL_CELLS.
 *     Dense vs. sparse: the dense layout avoids pointer chasing on the GPU
 *     and lets binary-descent traversal index directly into the buffer.
 *     10K cells × 32 bytes = 320 KB — fits comfortably within WebGPU
 *     `maxStorageBufferBindingSize` on all target tiers.
 *   - Directional bins: 16 octahedral cells per leaf (4 × 4 grid in
 *     [−1,1]² octahedral space). 16 bins × 8 bytes (vec2f) = 128 bytes/leaf.
 *   - Each bin stores (radianceSum, sampleCount) as f32 pair for online
 *     CDF reconstruction. No log-domain compression in Sprint 11.
 *
 * @since Sprint 11, 2026-05-09
 */

/**
 * Single directional bin in a PPG quad-tree leaf.
 *
 * Stores cumulative radiance for online PDF estimation. 16-direction
 * discretisation per the Sprint 11 DoD (16 octahedral cells arranged
 * in a 4×4 grid over the unit hemisphere mapped to octahedral [0,1]²).
 *
 * GPU layout (vec2f per bin): x = radianceSum, y = sampleCount.
 * Total per leaf: 16 × 8 bytes = 128 bytes.
 */
export interface PPGDirectionalBin {
  /** Cumulative incident radiance accumulated into this directional bin. */
  readonly radianceSum: number;
  /** Total path samples contributing to this bin (for variance / PDF weight). */
  readonly sampleCount: number;
}

/**
 * PPG quad-tree node — one leaf per spatial cell.
 *
 * Sparse in concept, dense in GPU storage: every spatial cell allocates
 * exactly one leaf. The `aabb` is used for future sub-cell split decisions
 * (tracking radiance-variance per spatial region). In Sprint 11, the split
 * logic is deferred; all cells start as leaves.
 */
export interface PPGQuadTreeNode {
  /** 16 directional bins for this spatial cell. */
  readonly bins: ReadonlyArray<PPGDirectionalBin>; // length === PPG_DIRECTIONS
  /** World-space axis-aligned bounding box of this spatial cell. */
  readonly aabb: {
    min: readonly [number, number, number];
    max: readonly [number, number, number];
  };
}

/**
 * PPG kd-tree spatial cell.
 *
 * One cell per occupied spatial region. Capped at `PPG_MAX_SPATIAL_CELLS`
 * to bound GPU buffer size. `leafIndex` indexes into the directional
 * quad-tree leaf array; in Sprint 11 leafIndex === cell array index (1:1).
 *
 * GPU layout (PPGSpatialCell struct, 32 bytes):
 *   bytes  0-11: position xyz (vec3f, 12 bytes + 4 padding = 16 bytes aligned)
 *   bytes 12-15: padding (_pad: f32)
 *   bytes 16-19: leafIndex (u32)
 *   bytes 20-31: reserved / alignment padding
 *
 * Total: 32 bytes/cell × 10,000 cells = 320 KB.
 */
export interface PPGSpatialCell {
  /** World-space centroid of this spatial cell. Used for kd-tree descent. */
  readonly position: readonly [number, number, number];
  /** Index into the directional leaf array (ppgLeaves buffer). */
  readonly leafIndex: number;
}

/**
 * Maximum number of PPG spatial cells that will be allocated.
 *
 * Chosen to fit within a 320 KB buffer (32 bytes/cell × 10K) — well within
 * WebGPU's guaranteed `maxStorageBufferBindingSize` of 128 MB.
 * Increasing beyond 10K requires a corresponding `maxCells` override in
 * `createPPGBuffers`.
 */
export const PPG_MAX_SPATIAL_CELLS = 10_000;

/**
 * Number of directional bins per spatial cell.
 *
 * 16 = 4 × 4 octahedral grid cells over the upper hemisphere.
 * Each bin covers a solid angle of ~π/4 sr.
 *
 * Sprint 11 DoD pins this at 16. A future sprint may raise it to 64 for
 * finer directional resolution at the cost of 4× leaf buffer size.
 */
export const PPG_DIRECTIONS = 16;

/**
 * Byte stride per PPGSpatialCell in the GPU buffer.
 *
 * Layout (WGSL struct PPGSpatialCell):
 *   position: vec3f  → 12 bytes
 *   _pad:     f32    →  4 bytes
 *   leafIndex: u32   →  4 bytes
 *   _pad2:    vec3u  → 12 bytes (alignment to 32 bytes)
 * Total: 32 bytes.
 */
export const PPG_CELL_BYTE_STRIDE = 32;

/**
 * Byte stride per PPGDirectionalLeaf in the GPU buffer.
 *
 * Layout (WGSL struct PPGDirectionalLeaf):
 *   bins: array<vec2f, 16>  → 16 × 8 bytes = 128 bytes
 * Total: 128 bytes.
 *
 * Note: padded to 256 bytes in the GPU allocation so that each leaf starts
 * on a 256-byte boundary, which satisfies WebGPU's `minStorageBufferOffsetAlignment`
 * on all known implementations. The extra 128 bytes are reserved for
 * future split-tracking fields (variance, sample count sum, split axis).
 */
export const PPG_LEAF_BYTE_STRIDE = 256;

/**
 * Single 16-byte GPU node for the PPG spatial kd-tree (see `buildPpgKdTree.ts`).
 * Layout matches WGSL `struct PPGKdNode` in `ppgSample.wgsl.ts`.
 */
export const PPG_KD_NODE_BYTE_STRIDE = 16;

/**
 * High bit set ⇒ leaf in kd-tree `meta` field; low 31 bits = cell index.
 */
export const PPG_KD_LEAF_FLAG = 0x8000_0000;

/**
 * Upper bound on kd-tree node count for a full cell budget (complete binary-ish tree).
 */
export const PPG_KD_MAX_NODES = PPG_MAX_SPATIAL_CELLS * 2 + 8;

/**
 * Options for `createPPGBuffers`. Matches the Sprint 11 integration spec.
 *
 * All fields optional — defaults match the Sprint 11 DoD values.
 */
export interface PPGBufferOptions {
  /**
   * Maximum number of spatial cells to allocate.
   * Defaults to `PPG_MAX_SPATIAL_CELLS` (10,000).
   */
  readonly maxCells?: number;
}
