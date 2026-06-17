/**
 * sppmParams.ts — TypeScript host-side SPPM physics constants and helpers.
 *
 * These are CPU/host-side values that parameterize the SPPM photon-mapping
 * algorithm. Extracted from sppmBindings.wgsl.ts so the WGSL string file only
 * holds the GPU shader string exports.
 *
 * Provenance: Hachisuka & Jensen 2009 "Stochastic Progressive Photon Mapping"
 * (ACM SIGGRAPH Asia 2009).
 */

/** Number of spatial-hash cells.  Power-of-2 avoids the modulo's division for
 *  non-power-of-2 table sizes, but an odd prime-ish size gives better hash
 *  distribution; 65521 is the largest prime below 2^16. */
export const SPPM_MAX_CELLS = 65521;

/** Photons stored per cell (bounded reservoir).  Over-capacity cells retain a
 *  random subset of SPPM_CELL_CAPACITY photons and the gather estimator weights
 *  stored samples by totalInserted / storedCount, so the cap controls memory
 *  without silently treating the retained subset as the whole cell.  R7a
 *  behavioral-gate fix (2026-06-10): capacity 128 made the cells
 *  buffer 65521 × 128 × 48 B ≈ 402 MiB — EXCEEDING WebGPU's default
 *  maxBufferSize (256 MiB), so every photon-map render failed buffer validation
 *  on default-limit devices.  32 → ≈ 100 MiB, inside BOTH the default maxBufferSize (256 MiB) AND the default maxStorageBufferBindingSize (128 MiB — the binding limit binds first); the
 *  host additionally guards against the live device limit at allocation and
 *  degrades to manifold-nee with a warning. */
export const SPPM_CELL_CAPACITY = 32;

/** Bytes per PhotonRecord: 3 × vec4f = 48 bytes. */
export const SPPM_PHOTON_RECORD_BYTES = 48;

/** Bytes for the sppmPhotonCells buffer:
 *  SPPM_MAX_CELLS × SPPM_CELL_CAPACITY × SPPM_PHOTON_RECORD_BYTES. */
export const SPPM_PHOTON_CELLS_BYTES =
  SPPM_MAX_CELLS * SPPM_CELL_CAPACITY * SPPM_PHOTON_RECORD_BYTES;

/** Bytes for the sppmCellCounters buffer: SPPM_MAX_CELLS × 4 (atomic<u32>). */
export const SPPM_CELL_COUNTERS_BYTES = SPPM_MAX_CELLS * 4;

/** Bytes for the SppmStats UBO: 4 × f32 = 32 bytes (padded to 32 for alignment). */
export const SPPM_STATS_BYTES = 32;

/**
 * I4.5 — Structural pin descriptor for the `SppmStats` WGSL struct.
 *
 * Each entry mirrors a field in the `struct SppmStats { … }` definition in
 * sppmBindings.wgsl.ts. The host packer (`GpuResources.writeSppmStats`) writes
 * these slots in this order; this descriptor lets a test assert:
 *   (a) the WGSL struct fields parse out in this exact order, and
 *   (b) the total size equals SPPM_STATS_BYTES (= 32).
 */
export const SPPM_STATS_FIELDS = [
  { name: 'currentRadius',    byteOffset:  0, type: 'f32' },
  { name: 'r0',               byteOffset:  4, type: 'f32' },
  { name: 'frameAccumulated', byteOffset:  8, type: 'u32' },
  { name: 'photonCount',      byteOffset: 12, type: 'u32' },
  { name: 'sceneExtent',      byteOffset: 16, type: 'f32' },
  { name: '_pad0',            byteOffset: 20, type: 'f32' },
  { name: '_pad1',            byteOffset: 24, type: 'f32' },
  { name: '_pad2',            byteOffset: 28, type: 'f32' },
] as const;

/**
 * Bytes per per-pixel SPPM statistics record:
 *   tau.rgb (f32×3) + radius2 (f32) + N (f32) + _pad (f32×3) = 8 × f32 = 32 bytes.
 *
 * A4-progressive: each pixel accumulates τ, R², and N across frames so the
 * Hachisuka progressive update rule can run without re-visiting previous frames.
 * The buffer is sized W×H×32 bytes and reset whenever the PT accumulator resets
 * (camera move, setScene, reset()) — the same static-eye-point assumption that
 * makes progressive accumulation valid also gates the SPPM stats.
 */
export const SPPM_PIXEL_STATS_BYTES_PER_PIXEL = 32; // 8 × f32

/**
 * Safety ceiling for the photon-cells buffer.  ~512 MiB at default params;
 * the warn-once pattern mirrors the BDPT eye-stack ceiling.
 */
export const SPPM_PHOTON_CELLS_MAX_BYTES = 512 * 1024 * 1024; // 512 MiB

/**
 * SPPM progressive-radius decay constant α (Hachisuka & Jensen 2009, Eq. 4).
 *
 * After `n` accumulated frames:
 *   r(n) = r₀ × sqrt((n × α + α) / (n + 1))
 * with α = 2/3.  For n=0 (first frame) r(0) = r₀ × sqrt(α) ≈ r₀ × 0.8165,
 * converging to 0 as n → ∞.
 */
export const SPPM_ALPHA = 2.0 / 3.0; // Hachisuka & Jensen 2009 α

/**
 * Compute the SPPM radius for frame index `n` (0-based) given initial radius `r0`.
 * Used on the CPU host side (TypeScript) to write the UBO each frame.
 */
export function sppmRadiusAtFrame(r0: number, n: number): number {
  if (n <= 0) return r0 * Math.sqrt(SPPM_ALPHA);
  return r0 * Math.sqrt((n * SPPM_ALPHA + SPPM_ALPHA) / (n + 1));
}

/**
 * Compute the scale-aware initial radius `r₀` from the scene AABB diagonal.
 *
 * r₀ = max(diagonal / 100, 1e-3)
 *
 * The divisor 100 ensures that even on a 1 m Cornell box (diagonal ~1.7 m)
 * the initial radius is ~0.017 m (17 mm) — a reasonable first-frame footprint
 * that shrinks to sub-millimetre over thousands of frames.
 */
export function sppmInitialRadius(
  sceneMin: readonly [number, number, number],
  sceneMax: readonly [number, number, number],
): number {
  const dx = sceneMax[0] - sceneMin[0];
  const dy = sceneMax[1] - sceneMin[1];
  const dz = sceneMax[2] - sceneMin[2];
  const diagonal = Math.sqrt(dx * dx + dy * dy + dz * dz);
  return Math.max(diagonal / 100, 1e-3);
}
