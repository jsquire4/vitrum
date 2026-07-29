/**
 * CPU reference for the ReSTIR-DI generalized Talbot / GRIS reuse weights.
 *
 * This is intentionally independent of WGSL composition so numerical tests can
 * exercise the estimator itself. For candidate y_i represented by reservoir i:
 *
 *   m_i(y_i) = M_i pHat_i(y_i) / sum_j M_j pHat_j(y_i)
 *   w_i      = m_i(y_i) pHat_0(y_i) W_i
 *
 * DI keeps the exact finite-emitter area sample (or exact environment
 * direction), so the shift Jacobian is one. The output reservoir's unbiased
 * contribution weight is sum_i(w_i) / pHat_0(z), with no second M division.
 *
 * Reference: Lin et al., "Generalized Resampled Importance Sampling:
 * Foundations of ReSTIR", SIGGRAPH 2022, Eq. 19 and supplemental Eq. S.7.
 */

const MAX_FINITE = 3.402823466e38;

function positiveFinite(value: number): number {
  return Number.isFinite(value) && value > 0 && value <= MAX_FINITE
    ? value
    : 0;
}

function logWeightedDensity(attempts: number, density: number): number {
  if (!Number.isSafeInteger(attempts) || attempts <= 0) {
    return Number.NEGATIVE_INFINITY;
  }
  const validDensity = positiveFinite(density);
  return validDensity > 0
    ? Math.log2(attempts) + Math.log2(validDensity)
    : Number.NEGATIVE_INFINITY;
}

export interface DiGeneralizedReuseCandidate {
  /** Index of the surface domain that produced this reservoir. */
  readonly sourceIndex: number;
  /** Represented proposal-attempt count for every gathered reservoir. */
  readonly attempts: readonly number[];
  /** pHat_j(y_i) for this candidate under every gathered surface domain j. */
  readonly densitiesAtDomains: readonly number[];
  /** Canonical contribution weight stored by source reservoir i. */
  readonly sourceReservoirWeight: number;
  /** Canonical output domain; center/current is index zero by convention. */
  readonly canonicalIndex?: number;
}

export function generalizedTalbotMisWeight(
  candidate: Pick<
    DiGeneralizedReuseCandidate,
    'sourceIndex' | 'attempts' | 'densitiesAtDomains'
  >,
): number {
  const { sourceIndex, attempts, densitiesAtDomains } = candidate;
  if (
    sourceIndex < 0 ||
    !Number.isSafeInteger(sourceIndex) ||
    attempts.length !== densitiesAtDomains.length ||
    sourceIndex >= attempts.length
  ) {
    return 0;
  }
  const logDensities: number[] = [];
  for (let index = 0; index < attempts.length; index += 1) {
    logDensities.push(
      logWeightedDensity(
        attempts[index]!,
        densitiesAtDomains[index]!,
      ),
    );
  }
  const sourceLogDensity = logDensities[sourceIndex]!;
  if (!Number.isFinite(sourceLogDensity)) return 0;
  const maxLogDensity = Math.max(...logDensities);
  let scaledDenominator = 0;
  for (const logDensity of logDensities) {
    if (Number.isFinite(logDensity)) {
      scaledDenominator += 2 ** (logDensity - maxLogDensity);
    }
  }
  return positiveFinite(
    2 ** (sourceLogDensity - maxLogDensity) / scaledDenominator,
  );
}

export function generalizedDiReuseCandidateWeight(
  candidate: DiGeneralizedReuseCandidate,
): number {
  const logWeight = generalizedDiReuseCandidateLogWeight(candidate);
  if (!Number.isFinite(logWeight)) return 0;
  if (logWeight >= Math.log2(MAX_FINITE)) return MAX_FINITE;
  return positiveFinite(2 ** logWeight);
}

export function generalizedDiReuseCandidateLogWeight(
  candidate: DiGeneralizedReuseCandidate,
): number {
  const canonicalIndex = candidate.canonicalIndex ?? 0;
  if (
    canonicalIndex < 0 ||
    !Number.isSafeInteger(canonicalIndex) ||
    canonicalIndex >= candidate.densitiesAtDomains.length
  ) {
    return 0;
  }
  const misWeight = generalizedTalbotMisWeight(candidate);
  const canonicalDensity = positiveFinite(
    candidate.densitiesAtDomains[canonicalIndex]!,
  );
  const sourceReservoirWeight = positiveFinite(
    candidate.sourceReservoirWeight,
  );
  if (
    misWeight === 0 ||
    canonicalDensity === 0 ||
    sourceReservoirWeight === 0
  ) {
    return Number.NEGATIVE_INFINITY;
  }
  return (
    Math.log2(misWeight) +
    Math.log2(canonicalDensity) +
    Math.log2(sourceReservoirWeight)
  );
}

export function scaleGeneralizedDiCandidateLogWeights(
  logWeights: readonly number[],
): {
  readonly maxLogWeight: number;
  readonly scaledWeights: readonly number[];
  readonly scaledWeightSum: number;
} {
  const maxLogWeight = Math.max(...logWeights);
  if (!Number.isFinite(maxLogWeight)) {
    return {
      maxLogWeight: Number.NEGATIVE_INFINITY,
      scaledWeights: logWeights.map(() => 0),
      scaledWeightSum: 0,
    };
  }
  const scaledWeights = logWeights.map((logWeight) =>
    Number.isFinite(logWeight) ? 2 ** (logWeight - maxLogWeight) : 0
  );
  return {
    maxLogWeight,
    scaledWeights,
    scaledWeightSum: scaledWeights.reduce((sum, weight) => sum + weight, 0),
  };
}

export function finalizeGeneralizedDiReservoirWeight(
  scaledResamplingWeightSum: number,
  selectedCanonicalDensity: number,
  maxLogWeight = 0,
): number {
  const numerator = positiveFinite(scaledResamplingWeightSum);
  const denominator = positiveFinite(selectedCanonicalDensity);
  if (
    numerator === 0 ||
    denominator === 0 ||
    !Number.isFinite(maxLogWeight)
  ) {
    return 0;
  }
  const logWeight =
    maxLogWeight + Math.log2(numerator) - Math.log2(denominator);
  return logWeight >= Math.log2(MAX_FINITE)
    ? MAX_FINITE
    : positiveFinite(2 ** logWeight);
}
