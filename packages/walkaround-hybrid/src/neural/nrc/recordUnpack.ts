// recordUnpack.ts — pure CPU repack of the per-frame NRC self-training records
// (D7.6, extracted from NrcSubsystem.trainFromRecords so the gap-detection +
// dense-repack loop is unit-testable without a GPU).
//
// Record layout (written by nrcQuery.wgsl nrcWriteRecord; recordStride f32s):
//   [ inW encoded input | OUT_W radiance target | 3 query WORLD pos ]
//
// A6 empty-slot detection: an unfilled slot (NRC spread never fired for that
// pixel) has all-zero ENCODED INPUT — the GPU record buffer is zero-initialized
// and nrcWriteRecord only runs when nrcFired is set. A genuine record guarantees
// at least one non-zero encoded feature, but not specifically input[0], so the
// CPU repack must scan the whole encoded-input prefix. Do NOT skip zero-TARGET
// records — a zero suffix target is a valid dark-surface training signal; the NRC
// should learn black rather than treating that slot as padding.

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
  /** Filled slots rejected because any record lane was non-finite. */
  droppedNonFinite: number;
  /** Target scalar lanes clamped into the finite training range. */
  clampedTargets: number;
}

/**
 * Bound online targets before fixed-point gradient accumulation. This keeps
 * HDR headroom while preventing one firefly from dominating a sparse window.
 */
export const NRC_MAX_TRAINING_TARGET = 1024;

function assertPositiveSafeInteger(label: string, value: number): void {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new RangeError(`${label} must be a positive safe integer; got ${value}`);
  }
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
  assertPositiveSafeInteger('NRC record capacity', cap);
  assertPositiveSafeInteger('NRC record stride', stride);
  assertPositiveSafeInteger('NRC encoded input width', inW);
  const minimumStride = inW + OUT_W + 3;
  if (stride < minimumStride) {
    throw new RangeError(`NRC record stride ${stride} is smaller than required ${minimumStride}`);
  }
  const requiredRaw = cap * stride;
  if (!Number.isSafeInteger(requiredRaw) || raw.length < requiredRaw) {
    throw new RangeError(`NRC mapped record buffer has ${raw.length} f32s; requires ${requiredRaw}`);
  }
  const x = out?.x ?? new Float32Array(cap * inW);
  const y = out?.y ?? new Float32Array(cap * OUT_W);
  const pos = out?.pos ?? new Float32Array(cap * 3);
  if (x.length < cap * inW || y.length < cap * OUT_W || pos.length < cap * 3) {
    throw new RangeError('NRC unpack destination arrays are smaller than the configured record capacity');
  }
  x.fill(0);
  y.fill(0);
  pos.fill(0);
  let filled = 0;
  let droppedNonFinite = 0;
  let clampedTargets = 0;
  for (let rIdx = 0; rIdx < cap; rIdx++) {
    const base = rIdx * stride;
    let hasEncodedInput = false;
    for (let i = 0; i < inW; i++) {
      if (raw[base + i] !== 0) {
        hasEncodedInput = true;
        break;
      }
    }
    if (!hasEncodedInput) continue; // empty slot: encoded input never written
    let finite = true;
    for (let i = 0; i < minimumStride; i++) {
      if (!Number.isFinite(raw[base + i])) {
        finite = false;
        break;
      }
    }
    if (!finite) {
      droppedNonFinite++;
      continue;
    }
    const tx = raw[base + inW + 0]!;
    const ty = raw[base + inW + 1]!;
    const tz = raw[base + inW + 2]!;
    for (let i = 0; i < inW; i++) x[filled * inW + i] = raw[base + i]!;
    const targets = [tx, ty, tz] as const;
    for (let c = 0; c < OUT_W; c++) {
      const bounded = Math.min(NRC_MAX_TRAINING_TARGET, Math.max(0, targets[c]!));
      if (bounded !== targets[c]) clampedTargets++;
      y[filled * OUT_W + c] = bounded;
    }
    pos[filled * 3 + 0] = raw[base + inW + OUT_W + 0]!;
    pos[filled * 3 + 1] = raw[base + inW + OUT_W + 1]!;
    pos[filled * 3 + 2] = raw[base + inW + OUT_W + 2]!;
    filled++;
  }
  return { x, y, pos, filled, droppedNonFinite, clampedTargets };
}
