// recordUnpack.ts — pure CPU repack of the per-frame NRC self-training records
// (D7.6, extracted from NrcSubsystem.trainFromRecords so the gap-detection +
// dense-repack loop is unit-testable without a GPU).
//
// Record layout (written by nrcQuery.wgsl nrcWriteRecord; recordStride f32s):
//   [ inW encoded input | OUT_W radiance target | 3 query WORLD pos ]
//
// A6 empty-slot detection: an unfilled slot (NRC spread never fired for that
// pixel) has all-zero ENCODED INPUT — the GPU record buffer is zero-initialized
// and nrcWriteRecord only runs when nrcFired is set. Checking input[0] is
// sufficient because a genuine record has at least one non-zero hash-grid
// feature (slot races are resolved first-writer-wins by nrcSlotClaims, so the
// readback sees one fully-assembled record per claimed slot). Do NOT skip
// zero-TARGET records — r.Lo==0 is a valid training signal (occluded surface;
// the NRC should predict black).

/** Radiance target width (RGB). MUST match nrcSubsystem's OUT_W. */
const OUT_W = 3;

export interface UnpackedRecords {
  /** Densely packed encoded inputs, [cap × inW] (zero tail past `filled`). */
  x: Float32Array;
  /** Densely packed radiance targets, [cap × OUT_W]. */
  y: Float32Array;
  /** Densely packed query world positions, [cap × 3]. */
  pos: Float32Array;
  /** Number of non-empty records repacked into the front of x/y/pos. */
  filled: number;
}

/**
 * Detect record gaps and repack the non-empty records densely (sample index s
 * in x == sample index s in pos == dL/dX row s for the encode-backward).
 *
 * @param raw    the mapped record buffer, [cap × stride] f32s.
 * @param cap    record capacity (slots).
 * @param stride f32s per record (= inW + OUT_W + 3).
 * @param inW    encoded-input width.
 * @param out    optional pre-allocated destination arrays (re-used each frame by
 *               the subsystem to avoid per-frame allocation); zeroed here. When
 *               omitted, fresh arrays are allocated.
 */
export function unpackRecords(
  raw: Float32Array, cap: number, stride: number, inW: number,
  out?: { x: Float32Array; y: Float32Array; pos: Float32Array },
): UnpackedRecords {
  const x = out?.x ?? new Float32Array(cap * inW);
  const y = out?.y ?? new Float32Array(cap * OUT_W);
  const pos = out?.pos ?? new Float32Array(cap * 3);
  x.fill(0);
  y.fill(0);
  pos.fill(0);
  let filled = 0;
  for (let rIdx = 0; rIdx < cap; rIdx++) {
    const base = rIdx * stride;
    if (raw[base] === 0) continue; // empty slot: encoded input never written
    const tx = raw[base + inW + 0]!;
    const ty = raw[base + inW + 1]!;
    const tz = raw[base + inW + 2]!;
    for (let i = 0; i < inW; i++) x[filled * inW + i] = raw[base + i]!;
    y[filled * OUT_W + 0] = tx;
    y[filled * OUT_W + 1] = ty;
    y[filled * OUT_W + 2] = tz;
    pos[filled * 3 + 0] = raw[base + inW + OUT_W + 0]!;
    pos[filled * 3 + 1] = raw[base + inW + OUT_W + 1]!;
    pos[filled * 3 + 2] = raw[base + inW + OUT_W + 2]!;
    filled++;
  }
  return { x, y, pos, filled };
}
