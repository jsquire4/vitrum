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

/** Per-iteration photon count and unique-record capacity. Each invocation owns
 * one record; bucket publication is a single atomic head exchange. */
export const SPPM_PHOTON_COUNT = 65536;

/** Bytes per PhotonRecord: 3 × vec4f = 48 bytes. */
export const SPPM_PHOTON_RECORD_BYTES = 48;

/** Bytes for the per-iteration photon-record buffer. */
export const SPPM_PHOTON_CELLS_BYTES =
  SPPM_PHOTON_COUNT * SPPM_PHOTON_RECORD_BYTES;

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
  { name: 'sceneCenterX',      byteOffset: 20, type: 'f32' },
  { name: 'sceneCenterY',      byteOffset: 24, type: 'f32' },
  { name: 'sceneCenterZ',      byteOffset: 28, type: 'f32' },
] as const;

/**
 * Bytes per pixel for two independent SPPM statistics records:
 *   surface: tau.rgb + radius2 + N + pad = 32 bytes
 *   volume:  tau.rgb + radius2 + N + pad = 32 bytes
 *
 * A4-progressive: each pixel accumulates τ, R², and N across frames so the
 * Hachisuka progressive update rule can run without re-visiting previous frames.
 * The buffer is sized W×H×64 bytes and reset whenever the PT accumulator resets
 * (camera move, setScene, reset()) — the same static-eye-point assumption that
 * makes progressive accumulation valid also gates the SPPM stats.
 */
export const SPPM_PIXEL_STATS_BYTES_PER_PIXEL = 64; // 2 measures × 8 f32

/**
 * Exact fixed-capacity photon-record allocation ceiling (3 MiB). The grid has
 * one record per emitted lane and never allocates per hash bucket.
 */
export const SPPM_PHOTON_CELLS_MAX_BYTES = SPPM_PHOTON_CELLS_BYTES;

/**
 * SPPM progressive-radius decay constant α (Hachisuka & Jensen 2009, Eq. 4).
 *
 * Radius is updated by the exact Hachisuka N/R recurrence, not by a closed-form
 * shortcut that would incorrectly remain constant for α=2/3.
 */
export const SPPM_ALPHA = 2.0 / 3.0; // Hachisuka & Jensen 2009 α

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
export interface SppmSceneBounds {
  readonly initialRadius: number;
  readonly extent: number;
  readonly center: readonly [number, number, number];
}

/**
 * Convert the canonical packed-scene bounding sphere into SPPM launch/gather
 * parameters. `sceneRadius` is the root AABB half-diagonal, so the equivalent
 * full diagonal used by {@link sppmInitialRadius} is `2 * sceneRadius`.
 *
 * This is the mutation-safe path: TLAS refits update `sceneCenter` and
 * `sceneRadius` even though the retained BLAS vertex buffer remains local and
 * byte-identical.
 */
export function sppmSceneBoundsFromCenterRadius(
  sceneCenter: readonly [number, number, number],
  sceneRadius: number,
): SppmSceneBounds | null {
  if (
    sceneCenter.length !== 3 ||
    !sceneCenter.every(Number.isFinite) ||
    !Number.isFinite(sceneRadius) ||
    sceneRadius < 0
  ) {
    return null;
  }
  const extent = Math.max(sceneRadius, 1e-3);
  return {
    initialRadius: Math.max((sceneRadius * 2) / 100, 1e-3),
    extent,
    center: [sceneCenter[0], sceneCenter[1], sceneCenter[2]],
  };
}

/** Compute the AABB-derived launch disk and gather scale from packed xyz/w vertices. */
export function sppmSceneBoundsFromPackedPositions(
  positions: ArrayLike<number>,
): SppmSceneBounds | null {
  if (positions.length < 4) return null;
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (let i = 0; i + 2 < positions.length; i += 4) {
    const x = positions[i]!, y = positions[i + 1]!, z = positions[i + 2]!;
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (y < minY) minY = y; if (y > maxY) maxY = y;
    if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
  }
  if (!Number.isFinite(minX)) return null;
  const min = [minX, minY, minZ] as const;
  const max = [maxX, maxY, maxZ] as const;
  const initialRadius = sppmInitialRadius(min, max);
  const dx = maxX - minX, dy = maxY - minY, dz = maxZ - minZ;
  return {
    initialRadius,
    extent: Math.max(Math.sqrt(dx * dx + dy * dy + dz * dz) * 0.5, initialRadius),
    center: [
      (minX + maxX) * 0.5,
      (minY + maxY) * 0.5,
      (minZ + maxZ) * 0.5,
    ],
  };
}
