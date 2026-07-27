/**
 * Runtime-sized Walker/Vose alias table with the represented selection PMF
 * stored beside every entry. The packed ABI is four 32-bit words per entry:
 * threshold q (f32), alias index (u32), represented pmf (f32), pad (u32).
 */

export const ALIAS_TABLE_ENTRY_BYTES = 16;
const ALIAS_RNG_BUCKETS = 1 << 24;

export interface AliasTable {
  readonly data: ArrayBuffer;
  readonly count: number;
  readonly representedPmf: Float32Array;
}

/** CPU oracle for one accepted word of WGSL's exact modulo-rejection draw. */
export function aliasColumnFromU32(randomU32: number, count: number): number {
  if (!Number.isSafeInteger(randomU32) || randomU32 < 0 || randomU32 > 0xffff_ffff) {
    throw new RangeError('alias random value must fit in u32');
  }
  if (!Number.isSafeInteger(count) || count <= 0 || count > 0xffff_ffff) {
    throw new RangeError('alias column count must be a non-zero u32');
  }
  const threshold = Number((1n << 32n) % BigInt(count));
  if (randomU32 < threshold) {
    throw new RangeError(`alias random value ${randomU32} is below rejection threshold ${threshold}`);
  }
  return randomU32 % count;
}

function assertWeights(weights: ArrayLike<number>): void {
  if (weights === null || (typeof weights !== 'object' && typeof weights !== 'function')) {
    throw new TypeError('alias-table weights must be array-like');
  }
  if (!Number.isSafeInteger(weights.length) || weights.length < 0 || weights.length > 0xffff_ffff) {
    throw new RangeError('alias-table weight count must fit in u32');
  }
  for (let index = 0; index < weights.length; index += 1) {
    const weight = weights[index];
    if (typeof weight !== 'number' || !Number.isFinite(weight) || weight < 0) {
      throw new RangeError(`alias-table weight[${index}] must be finite and nonnegative`);
    }
  }
}

/**
 * Build an alias table. Positive source weights retain positive represented
 * support; an all-zero input intentionally becomes a uniform distribution.
 */
export function buildAliasTable(weights: ArrayLike<number>): AliasTable {
  assertWeights(weights);
  const count = weights.length;
  if (count === 0) {
    return { data: new ArrayBuffer(0), count: 0, representedPmf: new Float32Array(0) };
  }

  let maxWeight = 0;
  for (let index = 0; index < count; index += 1) {
    maxWeight = Math.max(maxWeight, weights[index]!);
  }

  const normalized = new Float64Array(count);
  if (maxWeight === 0) {
    normalized.fill(1 / count);
  } else {
    const relativeWeights = new Float64Array(count);
    let relativeSum = 0;
    let compensation = 0;
    for (let index = 0; index < count; index += 1) {
      const relativeWeight = weights[index]! / maxWeight;
      relativeWeights[index] = relativeWeight;
      const corrected = relativeWeight - compensation;
      const next = relativeSum + corrected;
      compensation = (next - relativeSum) - corrected;
      relativeSum = next;
    }
    for (let index = 0; index < count; index += 1) {
      normalized[index] = relativeWeights[index]! / relativeSum;
    }
  }

  const scaled = new Float64Array(count);
  const small: number[] = [];
  const large: number[] = [];
  for (let index = 0; index < count; index += 1) {
    scaled[index] = normalized[index]! * count;
    (scaled[index]! < 1 ? small : large).push(index);
  }

  const threshold = new Float32Array(count);
  const alias = new Uint32Array(count);
  while (small.length > 0 && large.length > 0) {
    const low = small.pop()!;
    const high = large.pop()!;
    const rawThreshold = Math.min(1, Math.max(0, scaled[low]!));
    let thresholdBuckets = Math.floor(rawThreshold * ALIAS_RNG_BUCKETS);
    if (weights[low]! > 0 && thresholdBuckets === 0) thresholdBuckets = 1;
    threshold[low] = Math.fround(thresholdBuckets / ALIAS_RNG_BUCKETS);
    alias[low] = high;
    scaled[high] = scaled[high]! - (1 - scaled[low]!);
    (scaled[high] < 1 ? small : large).push(high);
  }
  for (const index of small) {
    threshold[index] = 1;
    alias[index] = index;
  }
  for (const index of large) {
    threshold[index] = 1;
    alias[index] = index;
  }

  // Recompute the PMF represented by the quantized f32 thresholds rather than
  // retaining the ideal input PMF. Shader weighting therefore divides by the
  // actual table encoded on the wire.
  const represented64 = new Float64Array(count);
  const invCount = 1 / count;
  for (let column = 0; column < count; column += 1) {
    const q = threshold[column]!;
    represented64[column] = represented64[column]! + q * invCount;
    const aliasIndex = alias[column]!;
    represented64[aliasIndex] = represented64[aliasIndex]! + (1 - q) * invCount;
  }
  const representedPmf = new Float32Array(count);
  for (let index = 0; index < count; index += 1) {
    representedPmf[index] = Math.fround(represented64[index]!);
    if (weights[index]! > 0 && representedPmf[index] === 0) {
      throw new RangeError(
        `alias-table represented pmf underflowed for positive weight[${index}]`,
      );
    }
  }

  const data = new ArrayBuffer(count * ALIAS_TABLE_ENTRY_BYTES);
  const f32 = new Float32Array(data);
  const u32 = new Uint32Array(data);
  for (let index = 0; index < count; index += 1) {
    const base = index * 4;
    f32[base] = threshold[index]!;
    u32[base + 1] = alias[index]!;
    f32[base + 2] = representedPmf[index]!;
  }
  return { data, count, representedPmf };
}
