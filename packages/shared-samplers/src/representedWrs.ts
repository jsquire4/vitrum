import {
  REPRESENTED_PROPOSAL_BUCKET_BITS,
  REPRESENTED_PROPOSAL_BUCKET_COUNT,
} from './representedDistribution.js';

/**
 * Transient state for represented, finite-RNG weighted reservoir sampling.
 *
 * Candidate weights are supplied as base-2 logarithms. The running weight sum
 * is held in max-relative Float32 form, while `logSelectionProbability` tracks
 * the high component of the selected *occurrence* probability under the actual
 * 24-bit branch decisions. `logSelectionProbabilityLow` is its double-single
 * residual. Keeping occurrence probability (rather than the ideal w / sum)
 * lets downstream estimators correct the represented proposal without losing
 * one-bucket keep factors to ordinary Float32 cancellation.
 */
export interface RepresentedWrsStateF32 {
  maxLogWeight: number;
  scaledWeightSum: number;
  selectedLogWeight: number;
  logSelectionProbability: number;
  logSelectionProbabilityLow: number;
  hasSelection: boolean;
}

export type RepresentedWrsRandom01 = () => number;

const F32_MAX = Math.fround(3.4028234663852886e38);

function f32(value: number): number {
  return Math.fround(value);
}

function exp2F32(value: number): number {
  return f32(2 ** f32(value));
}

function logBucketProbabilityF32(bucketCount: number): number {
  // Evaluate the ratio before log2. Computing log2(bucketCount) - 24 loses
  // the one-bucket endpoint probability to cancellation when count = B - 1.
  return f32(Math.log2(f32(bucketCount / REPRESENTED_PROPOSAL_BUCKET_COUNT)));
}

/** Add one f32 term to a double-single f32 pair and renormalize it. */
function addF32ToDoubleSingle(hi: number, lo: number, term: number): readonly [number, number] {
  const sum = f32(f32(hi) + f32(term));
  const virtualTerm = f32(sum - f32(hi));
  const roundoff = f32(
    f32(f32(hi) - f32(sum - virtualTerm)) + f32(f32(term) - virtualTerm),
  );
  const tail = f32(f32(lo) + roundoff);
  const normalizedHi = f32(sum + tail);
  const normalizedLo = f32(tail - f32(normalizedHi - sum));
  return [normalizedHi, normalizedLo];
}

function requireLiveState(state: RepresentedWrsStateF32): void {
  if (
    !Number.isFinite(state.maxLogWeight) ||
    state.maxLogWeight <= -F32_MAX ||
    state.maxLogWeight > F32_MAX ||
    !Number.isFinite(state.scaledWeightSum) ||
    !(state.scaledWeightSum > 0) ||
    !Number.isFinite(state.selectedLogWeight) ||
    state.selectedLogWeight <= -F32_MAX ||
    state.selectedLogWeight > F32_MAX ||
    !Number.isFinite(state.logSelectionProbability) ||
    state.logSelectionProbability > 0 ||
    !Number.isFinite(state.logSelectionProbabilityLow) ||
    state.logSelectionProbability + state.logSelectionProbabilityLow > 0
  ) {
    throw new RangeError('represented WRS state is not a valid live Float32 state.');
  }
}

/** Create an empty transient represented-WRS state. */
export function createRepresentedWrsStateF32(): RepresentedWrsStateF32 {
  return {
    maxLogWeight: 0,
    scaledWeightSum: 0,
    selectedLogWeight: 0,
    logSelectionProbability: 0,
    logSelectionProbabilityLow: 0,
    hasSelection: false,
  };
}

/**
 * Add one positive-weight candidate expressed as a base-2 log weight.
 *
 * Returns `true` exactly when the caller must copy this candidate's payload
 * into its reservoir. The first finite candidate is selected deterministically
 * and does not call `random01`. Every later finite candidate consumes one draw.
 * Non-finite values (including -Infinity, the log of zero weight) are ignored
 * without changing state or consuming RNG.
 *
 * `random01` must return a finite value in [0, 1). An invalid RNG value throws
 * before state is changed.
 */
export function updateRepresentedWrsF32(
  state: RepresentedWrsStateF32,
  candidateLogWeight: number,
  random01: RepresentedWrsRandom01,
): boolean {
  const logWeight = f32(candidateLogWeight);
  // -F32_MAX is the cross-shader LOG_ZERO sentinel, not a positive weight.
  if (!Number.isFinite(logWeight) || logWeight <= -F32_MAX || logWeight > F32_MAX) {
    return false;
  }

  if (!state.hasSelection) {
    state.maxLogWeight = logWeight;
    state.scaledWeightSum = 1;
    state.selectedLogWeight = logWeight;
    state.logSelectionProbability = 0;
    state.logSelectionProbabilityLow = 0;
    state.hasSelection = true;
    return true;
  }

  requireLiveState(state);
  const previousMax = f32(state.maxLogWeight);
  const nextMax = f32(Math.max(previousMax, logWeight));
  const oldScaledWeight = f32(
    f32(state.scaledWeightSum) * exp2F32(f32(previousMax - nextMax)),
  );
  const newScaledWeight = exp2F32(f32(logWeight - nextMax));
  const nextScaledWeightSum = f32(oldScaledWeight + newScaledWeight);
  const replacementRatio = f32(newScaledWeight / nextScaledWeightSum);
  const unboundedBuckets = Math.ceil(
    f32(f32(REPRESENTED_PROPOSAL_BUCKET_COUNT) * replacementRatio),
  );
  const replacementBuckets = Math.min(
    REPRESENTED_PROPOSAL_BUCKET_COUNT - 1,
    Math.max(1, unboundedBuckets),
  );

  const randomValue = random01();
  if (!Number.isFinite(randomValue) || randomValue < 0 || randomValue >= 1) {
    throw new RangeError('represented WRS random01 must return a finite value in [0, 1).');
  }
  const ticket = Math.floor(randomValue * REPRESENTED_PROPOSAL_BUCKET_COUNT);

  state.maxLogWeight = nextMax;
  state.scaledWeightSum = nextScaledWeightSum;
  if (ticket < replacementBuckets) {
    state.selectedLogWeight = logWeight;
    state.logSelectionProbability = logBucketProbabilityF32(replacementBuckets);
    state.logSelectionProbabilityLow = 0;
    return true;
  }

  const keepBuckets = REPRESENTED_PROPOSAL_BUCKET_COUNT - replacementBuckets;
  const [nextLogProbability, nextLogProbabilityLow] = addF32ToDoubleSingle(
    state.logSelectionProbability,
    state.logSelectionProbabilityLow,
    logBucketProbabilityF32(keepBuckets),
  );
  state.logSelectionProbability = nextLogProbability;
  state.logSelectionProbabilityLow = nextLogProbabilityLow;
  return false;
}

/** Recover the f32-accumulated log2 weight sum from the transient scaled sum. */
export function representedWrsLogWeightSumF32(state: RepresentedWrsStateF32): number {
  if (!state.hasSelection) return Number.NEGATIVE_INFINITY;
  requireLiveState(state);
  return f32(f32(state.maxLogWeight) + f32(Math.log2(f32(state.scaledWeightSum))));
}

/** Recover the represented occurrence log probability in binary64 host arithmetic. */
export function representedWrsLogSelectionProbability(
  state: RepresentedWrsStateF32,
): number {
  if (!state.hasSelection) return Number.NEGATIVE_INFINITY;
  requireLiveState(state);
  return state.logSelectionProbability + state.logSelectionProbabilityLow;
}

/** Log-domain correction carried by the selected occurrence: log2(w_selected / r). */
export function representedWrsSelectedLogCorrection(
  state: RepresentedWrsStateF32,
): number {
  if (!state.hasSelection) return Number.NEGATIVE_INFINITY;
  requireLiveState(state);
  return (
    state.selectedLogWeight -
    state.logSelectionProbability -
    state.logSelectionProbabilityLow
  );
}

export {
  REPRESENTED_PROPOSAL_BUCKET_BITS as REPRESENTED_WRS_BUCKET_BITS,
  REPRESENTED_PROPOSAL_BUCKET_COUNT as REPRESENTED_WRS_BUCKET_COUNT,
};
