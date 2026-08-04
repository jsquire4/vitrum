import {
  REPRESENTED_PROPOSAL_BUCKET_BITS,
  REPRESENTED_PROPOSAL_BUCKET_COUNT,
} from '../representedDistribution.js';

/** Module name for include-graph consumers. */
export const REPRESENTED_WRS_MODULE_NAME = 'representedWrs';

/**
 * Generic represented finite-RNG weighted-reservoir primitive.
 *
 * `pcgNext(ptr<function, u32>)` must already be in scope. Payload storage is
 * deliberately caller-owned: `representedWrsUpdate` returns true when the
 * caller must copy the current candidate into its reservoir.
 */
export const REPRESENTED_WRS_WGSL = /* wgsl */ `
const REPRESENTED_WRS_BUCKET_BITS: u32 = ${REPRESENTED_PROPOSAL_BUCKET_BITS}u;
const REPRESENTED_WRS_BUCKET_COUNT: u32 = ${REPRESENTED_PROPOSAL_BUCKET_COUNT}u;

struct RepresentedWrsState {
  maxLogWeight: f32,
  scaledWeightSum: f32,
  selectedLogWeight: f32,
  logSelectionProbability: f32,
  logSelectionProbabilityLow: f32,
  hasSelection: bool,
};

fn representedWrsInit() -> RepresentedWrsState {
  var result: RepresentedWrsState;
  result.maxLogWeight = 0.0;
  result.scaledWeightSum = 0.0;
  result.selectedLogWeight = 0.0;
  result.logSelectionProbability = 0.0;
  result.logSelectionProbabilityLow = 0.0;
  result.hasSelection = false;
  return result;
}

fn representedWrsLogBucketProbability(bucketCount: u32) -> f32 {
  // Mathematically log2(bucketCount) - 24. Evaluating the exact f32 ratio
  // first avoids cancellation to zero for the B - 1 endpoint bucket count.
  return log2(f32(bucketCount) / f32(REPRESENTED_WRS_BUCKET_COUNT));
}

// Add one f32 term to a double-single f32 pair. The residual prevents repeated
// one-bucket keep factors from rounding away when the high component is large.
fn representedWrsAddLogTerm(hi: f32, lo: f32, term: f32) -> vec2f {
  let sum = hi + term;
  let virtualTerm = sum - hi;
  let roundoff = (hi - (sum - virtualTerm)) + (term - virtualTerm);
  let tail = lo + roundoff;
  let normalizedHi = sum + tail;
  let normalizedLo = tail - (normalizedHi - sum);
  return vec2f(normalizedHi, normalizedLo);
}

fn representedWrsUpdate(
  wrs: ptr<function, RepresentedWrsState>,
  candidateLogWeight: f32,
  rng: ptr<function, u32>,
) -> bool {
  // NaN, both infinities, and the shared -F32_MAX LOG_ZERO sentinel fail this
  // ordered finite-range test. They model no candidate and consume no RNG.
  if (!(
    candidateLogWeight > -3.402823466e38 &&
    candidateLogWeight <= 3.402823466e38
  )) {
    return false;
  }

  if (!(*wrs).hasSelection) {
    (*wrs).maxLogWeight = candidateLogWeight;
    (*wrs).scaledWeightSum = 1.0;
    (*wrs).selectedLogWeight = candidateLogWeight;
    (*wrs).logSelectionProbability = 0.0;
    (*wrs).logSelectionProbabilityLow = 0.0;
    (*wrs).hasSelection = true;
    return true;
  }

  let nextMaxLogWeight = max((*wrs).maxLogWeight, candidateLogWeight);
  let oldScaledWeight = (*wrs).scaledWeightSum * exp2(
    (*wrs).maxLogWeight - nextMaxLogWeight,
  );
  let newScaledWeight = exp2(candidateLogWeight - nextMaxLogWeight);
  let nextScaledWeightSum = oldScaledWeight + newScaledWeight;
  let replacementRatio = newScaledWeight / nextScaledWeightSum;
  let replacementBuckets = clamp(
    u32(ceil(f32(REPRESENTED_WRS_BUCKET_COUNT) * replacementRatio)),
    1u,
    REPRESENTED_WRS_BUCKET_COUNT - 1u,
  );

  // The high 24 PCG bits are one exact uniform integer ticket in [0, B).
  let ticket = pcgNext(rng) >> 8u;
  (*wrs).maxLogWeight = nextMaxLogWeight;
  (*wrs).scaledWeightSum = nextScaledWeightSum;
  if (ticket < replacementBuckets) {
    (*wrs).selectedLogWeight = candidateLogWeight;
    (*wrs).logSelectionProbability = representedWrsLogBucketProbability(
      replacementBuckets,
    );
    (*wrs).logSelectionProbabilityLow = 0.0;
    return true;
  }

  let keepBuckets = REPRESENTED_WRS_BUCKET_COUNT - replacementBuckets;
  let nextLogProbability = representedWrsAddLogTerm(
    (*wrs).logSelectionProbability,
    (*wrs).logSelectionProbabilityLow,
    representedWrsLogBucketProbability(keepBuckets),
  );
  (*wrs).logSelectionProbability = nextLogProbability.x;
  (*wrs).logSelectionProbabilityLow = nextLogProbability.y;
  return false;
}

fn representedWrsLogSelectionProbabilityParts(wrs: RepresentedWrsState) -> vec2f {
  return vec2f(wrs.logSelectionProbability, wrs.logSelectionProbabilityLow);
}

fn representedWrsSelectedLogCorrectionParts(wrs: RepresentedWrsState) -> vec2f {
  let highDifference = representedWrsAddLogTerm(
    wrs.selectedLogWeight,
    0.0,
    -wrs.logSelectionProbability,
  );
  return representedWrsAddLogTerm(
    highDifference.x,
    highDifference.y,
    -wrs.logSelectionProbabilityLow,
  );
}
`;
