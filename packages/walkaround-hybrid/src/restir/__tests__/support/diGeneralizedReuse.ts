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
 * direction), so the shift Jacobian is one. The outer represented WRS stores
 * H = log2(w_selected / pi_selected); generalized finalization persists
 * logW = H - log2(pHat_0(z)), with no second M division and no linear endpoint.
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
  /** Legacy linear source UCW, used only when no persisted H is supplied. */
  readonly sourceReservoirWeight?: number;
  /** Persisted H = log2(W_uncapped * pHat_i(selected)). */
  readonly sourceLogEstimatorNumerator?: number;
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
  if (
    candidate.sourceIndex < 0 ||
    !Number.isSafeInteger(candidate.sourceIndex) ||
    candidate.sourceIndex >= candidate.densitiesAtDomains.length
  ) {
    return Number.NEGATIVE_INFINITY;
  }
  const misWeight = generalizedTalbotMisWeight(candidate);
  const canonicalDensity = positiveFinite(
    candidate.densitiesAtDomains[canonicalIndex]!,
  );
  const sourceDensity = positiveFinite(
    candidate.densitiesAtDomains[candidate.sourceIndex]!,
  );
  let sourceLogWeight = Number.NEGATIVE_INFINITY;
  if (
    candidate.sourceLogEstimatorNumerator !== undefined &&
    Number.isFinite(candidate.sourceLogEstimatorNumerator) &&
    sourceDensity > 0
  ) {
    sourceLogWeight = Math.min(
      candidate.sourceLogEstimatorNumerator - Math.log2(sourceDensity),
      MAX_FINITE,
    );
  } else {
    const sourceReservoirWeight = positiveFinite(
      candidate.sourceReservoirWeight ?? 0,
    );
    if (sourceReservoirWeight > 0) {
      sourceLogWeight = Math.log2(sourceReservoirWeight);
    }
  }
  if (
    misWeight === 0 ||
    canonicalDensity === 0 ||
    !Number.isFinite(sourceLogWeight)
  ) {
    return Number.NEGATIVE_INFINITY;
  }
  return (
    Math.log2(misWeight) +
    Math.log2(canonicalDensity) +
    sourceLogWeight
  );
}

export function finalizeGeneralizedDiReservoirLogWeight(
  selectedRepresentedLogCorrection: number,
  selectedCanonicalDensity: number,
): { readonly H: number; readonly logW: number } {
  const denominator = positiveFinite(selectedCanonicalDensity);
  if (
    denominator === 0 ||
    !Number.isFinite(selectedRepresentedLogCorrection) ||
    selectedRepresentedLogCorrection <= -MAX_FINITE ||
    selectedRepresentedLogCorrection > MAX_FINITE
  ) {
    return {
      H: Number.NEGATIVE_INFINITY,
      logW: Number.NEGATIVE_INFINITY,
    };
  }
  const H = selectedRepresentedLogCorrection;
  return {
    H,
    logW: Math.min(H - Math.log2(denominator), MAX_FINITE),
  };
}
