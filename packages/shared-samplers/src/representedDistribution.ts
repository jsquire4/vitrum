/**
 * A finite-precision categorical proposal whose published PMF is exactly the
 * distribution sampled by a `bucketBits`-wide uniform integer draw.
 *
 * Web shaders commonly obtain a uniform variate by retaining 24 random bits
 * and multiplying by 2^-24. Normalising arbitrary floating-point weights into
 * a Float32 CDF does not, by itself, preserve positive support or make the
 * published PMF agree with those 2^24 reachable variates. This allocator uses
 * Hamilton's largest-remainder method and reserves one bucket for every
 * positive source. Every returned PMF/CDF value is therefore an exact binary
 * fraction representable by Float32.
 */

export const REPRESENTED_PROPOSAL_BUCKET_BITS = 24;
export const REPRESENTED_PROPOSAL_BUCKET_COUNT = 2 ** REPRESENTED_PROPOSAL_BUCKET_BITS;

export type RepresentedZeroWeightFallback = 'empty' | 'uniform';

export interface RepresentedDistributionOptions {
  /** Defaults to 24, matching shader f32 mantissa precision. */
  readonly bucketBits?: number;
  /** Defaults to `empty`; `uniform` gives every entry support when all weights are zero. */
  readonly zeroWeightFallback?: RepresentedZeroWeightFallback;
}

export interface RepresentedDistributionF32 {
  /** One exact bucket count per source. */
  readonly bucketCounts: Uint32Array;
  /** Exact bucketCount / totalBucketCount values. */
  readonly pmf: Float32Array;
  /** Prefix CDF with cdf[0] = 0 and length pmf.length + 1. */
  readonly cdf: Float32Array;
  readonly bucketBits: number;
  readonly totalBucketCount: number;
}

function requireBucketBits(value: number | undefined): number {
  const bits = value ?? REPRESENTED_PROPOSAL_BUCKET_BITS;
  if (!Number.isSafeInteger(bits) || bits < 1 || bits > 24) {
    throw new RangeError('represented categorical bucketBits must be an integer in [1, 24].');
  }
  return bits;
}

function allocatePositiveWeights(
  weights: readonly number[],
  positiveIndices: readonly number[],
  totalBucketCount: number,
): Uint32Array {
  if (positiveIndices.length > totalBucketCount) {
    throw new RangeError(
      `represented categorical proposal has ${positiveIndices.length} positive sources but only ` +
        `${totalBucketCount} random buckets; exact positive support is impossible.`,
    );
  }

  const buckets = new Uint32Array(weights.length);
  if (positiveIndices.length === 0) return buckets;

  let maxWeight = 0;
  for (const index of positiveIndices) maxWeight = Math.max(maxWeight, weights[index]!);

  // Max-relative Kahan accumulation keeps finite binary64 inputs finite without
  // pre-quantising them to Float32 (which is the support loss this helper fixes).
  let scaledSum = 0;
  let compensation = 0;
  for (const index of positiveIndices) {
    const scaled = weights[index]! / maxWeight;
    const corrected = scaled - compensation;
    const next = scaledSum + corrected;
    compensation = (next - scaledSum) - corrected;
    scaledSum = next;
  }

  const remaining = totalBucketCount - positiveIndices.length;
  const remainders: Array<{ readonly index: number; readonly fraction: number }> = [];
  let allocated = positiveIndices.length;
  for (const index of positiveIndices) {
    const quota = remaining === 0
      ? 0
      : ((weights[index]! / maxWeight) / scaledSum) * remaining;
    const extra = Math.floor(quota);
    buckets[index] = 1 + extra;
    allocated += extra;
    remainders.push({ index, fraction: quota - extra });
  }

  // Floating-point quota sums can differ from `remaining` by a few ulps. The
  // ordinary Hamilton case distributes fewer than N residual buckets; the
  // cyclic form below also gives a deterministic, support-preserving result in
  // the face of a larger numerical residual.
  remainders.sort((a, b) =>
    b.fraction === a.fraction ? a.index - b.index : b.fraction - a.fraction,
  );
  let deficit = totalBucketCount - allocated;
  for (let cursor = 0; deficit > 0; cursor += 1, deficit -= 1) {
    const recipient = remainders[cursor % remainders.length]!;
    buckets[recipient.index] = (buckets[recipient.index] ?? 0) + 1;
  }

  if (deficit < 0) {
    const ascending = [...remainders].sort((a, b) =>
      a.fraction === b.fraction ? b.index - a.index : a.fraction - b.fraction,
    );
    let excess = -deficit;
    for (let cursor = 0; excess > 0; cursor += 1) {
      const donor = ascending[cursor % ascending.length]!;
      const donorCount = buckets[donor.index] ?? 0;
      if (donorCount <= 1) continue;
      buckets[donor.index] = donorCount - 1;
      excess -= 1;
    }
  }

  return buckets;
}

export function buildRepresentedDistributionF32(
  sourceWeights: ArrayLike<number>,
  options: RepresentedDistributionOptions = {},
): RepresentedDistributionF32 {
  if (options == null || typeof options !== 'object' || Array.isArray(options)) {
    throw new TypeError('represented categorical options must be an object.');
  }
  if (
    options.zeroWeightFallback !== undefined &&
    options.zeroWeightFallback !== 'empty' &&
    options.zeroWeightFallback !== 'uniform'
  ) {
    throw new RangeError(
      'represented categorical zeroWeightFallback must be "empty" or "uniform".',
    );
  }
  const bucketBits = requireBucketBits(options.bucketBits);
  const totalBucketCount = 2 ** bucketBits;
  const weights = Array.from(sourceWeights);
  const positiveIndices: number[] = [];

  for (let i = 0; i < weights.length; i += 1) {
    const weight = weights[i]!;
    if (!Number.isFinite(weight) || weight < 0) {
      throw new RangeError(
        `represented categorical weight[${i}] must be finite and non-negative (received ${String(weight)}).`,
      );
    }
    if (weight > 0) positiveIndices.push(i);
  }

  let allocationWeights = weights;
  let allocationIndices = positiveIndices;
  if (allocationIndices.length === 0 && options.zeroWeightFallback === 'uniform') {
    if (weights.length > totalBucketCount) {
      throw new RangeError(
        `represented uniform fallback has ${weights.length} sources but only ` +
          `${totalBucketCount} random buckets; exact support is impossible.`,
      );
    }
    allocationWeights = weights.map(() => 1);
    allocationIndices = weights.map((_, index) => index);
  }

  const bucketCounts = allocatePositiveWeights(
    allocationWeights,
    allocationIndices,
    totalBucketCount,
  );
  const pmf = new Float32Array(weights.length);
  const cdf = new Float32Array(weights.length + 1);
  let cumulative = 0;
  for (let i = 0; i < weights.length; i += 1) {
    const count = bucketCounts[i]!;
    cumulative += count;
    pmf[i] = Math.fround(count / totalBucketCount);
    cdf[i + 1] = Math.fround(cumulative / totalBucketCount);
  }

  const hasRepresentedDistribution = allocationIndices.length > 0;
  const expectedTerminal = hasRepresentedDistribution ? 1 : 0;
  if (cumulative !== (hasRepresentedDistribution ? totalBucketCount : 0)) {
    throw new Error(
      `represented categorical allocation invariant failed: assigned ${cumulative} of ` +
        `${hasRepresentedDistribution ? totalBucketCount : 0} expected buckets.`,
    );
  }
  if (cdf[cdf.length - 1] !== expectedTerminal) {
    throw new Error('represented categorical terminal CDF invariant failed.');
  }
  for (const index of positiveIndices) {
    if (!(bucketCounts[index]! > 0)) {
      throw new Error(`represented categorical support invariant failed at source ${index}.`);
    }
  }

  return { bucketCounts, pmf, cdf, bucketBits, totalBucketCount };
}

export function buildRepresentedPmfF32(
  sourceWeights: ArrayLike<number>,
  options: RepresentedDistributionOptions = {},
): Float32Array {
  return buildRepresentedDistributionF32(sourceWeights, options).pmf;
}

export function buildRepresentedCdfF32(
  sourceWeights: ArrayLike<number>,
  options: RepresentedDistributionOptions = {},
): Float32Array {
  return buildRepresentedDistributionF32(sourceWeights, options).cdf;
}

/**
 * Publish an authored open-unit Bernoulli probability as an exact member of
 * the same 24-bit random domain used by the shader RNGs.
 *
 * For a shader branch `rand24 < p`, returning exactly `k / 2^24` makes the
 * realized branch probability exactly `p`: precisely the integer buckets
 * `[0, k)` select the branch. Clamping to the two interior endpoint buckets
 * preserves both sides of an authored defensive mixture.
 */
export function quantizeOpenUnitProbabilityF32(
  value: number,
  label = 'probability',
): number {
  if (!Number.isFinite(value) || value <= 0 || value >= 1) {
    throw new RangeError(`${label} must be finite and strictly between 0 and 1.`);
  }
  return representBernoulliProbabilityF32(value, label);
}

/**
 * Represent a closed-unit Bernoulli probability on the shader RNG's exact
 * 24-bit lattice. Endpoints remain exact. Every interior probability keeps at
 * least one bucket on each side, then rounds to the nearest `k / 2^24` value.
 * A weighted branch must use the returned value for both its comparison and
 * its throughput/PDF correction.
 */
export function representBernoulliProbabilityF32(
  value: number,
  label = 'probability',
): number {
  if (!Number.isFinite(value) || value < 0 || value > 1) {
    throw new RangeError(`${label} must be finite and inside [0, 1].`);
  }
  if (value === 0 || value === 1) return value;
  const buckets = Math.min(
    REPRESENTED_PROPOSAL_BUCKET_COUNT - 1,
    Math.max(1, Math.round(value * REPRESENTED_PROPOSAL_BUCKET_COUNT)),
  );
  return Math.fround(buckets / REPRESENTED_PROPOSAL_BUCKET_COUNT);
}
