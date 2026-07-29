/**
 * Test-only independent CPU oracle for the bounded GRIS reuse implemented by
 * the walkaround renderer.
 *
 * This is deliberately narrower than a general path-space GRIS implementation:
 * samples represent the renderer's one-bounce diffuse DDGI proxy. Surface
 * reconnection uses a geometry-term Jacobian; environment directions use an
 * identity mapping. Every proposal density is transformed into the canonical
 * receiver measure with pHat / |dT| before the generalized-balance weights are
 * formed. Invalid inverse mappings and occluded techniques have zero density.
 *
 * The renderer remains biased by its one-bounce proxy, reservoir clamps, and
 * irradiance clamps. These helpers validate reuse arithmetic; they do not claim
 * full-path or unbiased rendering.
 */

export type Vec3 = readonly [number, number, number];
export type GrisSampleKind = 'surface' | 'environment';

const INV_PI = 0.3183098861837907;
export const MAX_GRIS_TECHNIQUES = 6;
const MAX_U32 = 0xffff_ffff;
const MAX_FINITE_F32 = 3.402823466e38;
const LOG_MAX_FINITE_F32 = Math.log2(MAX_FINITE_F32);

function finiteVec3(value: Vec3): boolean {
  return value.every((component) => Number.isFinite(component));
}

function positiveU32Attempts(value: number, domainIndex: number): number {
  if (!Number.isInteger(value) || value <= 0 || value > MAX_U32) {
    throw new RangeError(
      `GRIS domain ${domainIndex} attempts must be a positive u32; got ${value}`,
    );
  }
  return value;
}

/** log2(M * pHat / J), without forming an overflow-prone quotient/product. */
export function logWeightedTransformedDensity(
  attempts: number,
  pHat: number,
  jacobian: number,
  inverseValid = true,
): number {
  if (
    !inverseValid ||
    !Number.isInteger(attempts) ||
    attempts <= 0 ||
    attempts > MAX_U32 ||
    !(pHat > 0) ||
    !Number.isFinite(pHat) ||
    !(jacobian > 0) ||
    !Number.isFinite(jacobian)
  ) {
    return Number.NEGATIVE_INFINITY;
  }
  return Math.log2(attempts) + Math.log2(pHat) - Math.log2(jacobian);
}

function sub(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function dot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function normalized(v: Vec3): Vec3 | undefined {
  const lengthSquared = dot(v, v);
  if (!(lengthSquared > 1e-12) || !Number.isFinite(lengthSquared)) return undefined;
  const inverseLength = 1 / Math.sqrt(lengthSquared);
  return [v[0] * inverseLength, v[1] * inverseLength, v[2] * inverseLength];
}

function luminance(c: Vec3): number {
  return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
}

/** Receiver technique participating in the bounded all-technique matrix. */
export interface GrisDomain {
  readonly xv: Vec3;
  readonly nv: Vec3;
  /** Reservoir attempts represented by this technique. */
  readonly attempts: number;
  /** Visibility of this fixed sample from this receiver. Defaults to one. */
  readonly visibility?: number;
  /** Whether this receiver has a valid inverse mapping for the sample. */
  readonly inverseValid?: boolean;
}

/** A fixed sample evaluated under every receiver technique. */
export interface GrisSample {
  readonly kind: GrisSampleKind;
  /** Surface reconnection position; required when kind is `surface`. */
  readonly xs?: Vec3;
  /** Surface reconnection normal; required when kind is `surface`. */
  readonly ns?: Vec3;
  /** Persistent direction; required when kind is `environment`. */
  readonly direction?: Vec3;
  readonly Lo: Vec3;
  /** Technique that originally produced this sample. */
  readonly nativeDomainIndex: number;
  /** Stored native pHat, including native visibility, when available. */
  readonly nativePHat?: number;
}

export interface GrisTechniqueMatrix {
  readonly pHats: readonly number[];
  readonly jacobians: readonly number[];
  readonly transformedDensities: readonly number[];
  /** Max-log-normalized masses in [0,1], preserving all representable ratios. */
  readonly numerators: readonly number[];
  /** Sum of the normalized numerators (at most MAX_GRIS_TECHNIQUES). */
  readonly denominator: number;
  readonly logNumerators: readonly number[];
  /** Common log2 scale removed from numerators/denominator. */
  readonly logScale: number;
  readonly weights: readonly number[];
}

/** Destination-cosine half-geometry term used by the reconnection map. */
export function reconnectionGeometryTerm(x1: Vec3, x2: Vec3, n2: Vec3): number {
  if (!finiteVec3(x1) || !finiteVec3(x2) || !finiteVec3(n2)) return 0;
  const edge = sub(x2, x1);
  const distanceSquared = dot(edge, edge);
  if (!(distanceSquared > 1e-12) || !Number.isFinite(distanceSquared)) return 0;
  const inverseDistance = 1 / Math.sqrt(distanceSquared);
  const result = Math.abs(dot(n2, edge) * inverseDistance) / distanceSquared;
  return Number.isFinite(result) && result > 0 ? result : 0;
}

/** |dT_domain->canonical|. Environment directions map identically. */
export function domainToCanonicalJacobian(
  domainXv: Vec3,
  canonicalXv: Vec3,
  sample: GrisSample,
): number {
  if (sample.kind === 'environment') return 1;
  if (sample.xs === undefined || sample.ns === undefined) return 0;
  const domainGeometry = reconnectionGeometryTerm(domainXv, sample.xs, sample.ns);
  const canonicalGeometry = reconnectionGeometryTerm(canonicalXv, sample.xs, sample.ns);
  if (!(domainGeometry > 0) || !(canonicalGeometry > 0)) return 0;
  const result = canonicalGeometry / domainGeometry;
  return Number.isFinite(result) && result > 0 && result <= MAX_FINITE_F32
    ? result
    : 0;
}

/** Diffuse one-bounce proxy target without visibility. */
export function proxyTargetAt(domain: GrisDomain, sample: GrisSample): number {
  const direction = sample.kind === 'environment'
    ? (sample.direction === undefined ? undefined : normalized(sample.direction))
    : (sample.xs === undefined ? undefined : normalized(sub(sample.xs, domain.xv)));
  if (direction === undefined) return 0;
  const receiverCosine = Math.max(0, dot(domain.nv, direction));
  const value = luminance(sample.Lo) * receiverCosine * INV_PI;
  return Number.isFinite(value) && value > 0 ? value : 0;
}

/** Proxy pHat including the technique's own visibility result. */
export function proxyPHatAt(domain: GrisDomain, sample: GrisSample): number {
  if (domain.inverseValid === false) return 0;
  const rawVisibility = domain.visibility ?? 1;
  if (!Number.isFinite(rawVisibility)) return 0;
  const visibility = Math.min(1, Math.max(0, rawVisibility));
  if (!(visibility > 0)) return 0;
  return proxyTargetAt(domain, sample) * visibility;
}

/** Convert a native-domain density into the canonical receiver measure. */
export function transformedDensity(
  pHatDomain: number,
  domainToCanonical: number,
  inverseValid = true,
): number {
  if (!inverseValid || !(pHatDomain > 0) || !(domainToCanonical > 0)) return 0;
  const result = pHatDomain / domainToCanonical;
  return Number.isFinite(result) && result > 0 && result <= MAX_FINITE_F32
    ? result
    : 0;
}

/**
 * Evaluate the complete bounded generalized-balance denominator for one fixed
 * sample. No pairwise streaming approximation is used.
 */
export function evaluateTechniqueMatrix(
  domains: readonly GrisDomain[],
  sample: GrisSample,
  canonicalDomainIndex: number,
): GrisTechniqueMatrix {
  if (sample.kind !== 'surface' && sample.kind !== 'environment') {
    throw new TypeError(
      `sample.kind must be surface or environment; got ${String(sample.kind)}`,
    );
  }
  if (domains.length > MAX_GRIS_TECHNIQUES) {
    throw new RangeError(`GRIS supports at most ${MAX_GRIS_TECHNIQUES} techniques`);
  }
  if (!Number.isInteger(canonicalDomainIndex)
      || canonicalDomainIndex < 0
      || canonicalDomainIndex >= domains.length) {
    throw new RangeError('canonicalDomainIndex is outside the domain matrix');
  }
  if (!Number.isInteger(sample.nativeDomainIndex)
      || sample.nativeDomainIndex < 0
      || sample.nativeDomainIndex >= domains.length) {
    throw new RangeError('sample.nativeDomainIndex is outside the domain matrix');
  }

  const canonical = domains[canonicalDomainIndex]!;
  const pHats = new Array<number>(domains.length).fill(0);
  const jacobians = new Array<number>(domains.length).fill(0);
  const densities = new Array<number>(domains.length).fill(0);
  const logNumerators = new Array<number>(domains.length).fill(
    Number.NEGATIVE_INFINITY,
  );

  for (let index = 0; index < domains.length; index += 1) {
    const domain = domains[index]!;
    const attempts = positiveU32Attempts(domain.attempts, index);
    const visibility = domain.visibility ?? 1;
    const inverseValid = domain.inverseValid !== false
      && Number.isFinite(visibility)
      && visibility > 0;
    let pHat = index === sample.nativeDomainIndex && sample.nativePHat !== undefined
      ? sample.nativePHat
      : proxyPHatAt(domain, sample);
    if (!inverseValid || !Number.isFinite(pHat) || !(pHat > 0)) pHat = 0;

    const jacobian = domainToCanonicalJacobian(domain.xv, canonical.xv, sample);
    const density = transformedDensity(pHat, jacobian, inverseValid);
    const logNumerator = logWeightedTransformedDensity(
      attempts,
      pHat,
      jacobian,
      inverseValid,
    );

    pHats[index] = pHat;
    jacobians[index] = jacobian;
    densities[index] = density;
    logNumerators[index] = logNumerator;
  }

  const logScale = Math.max(...logNumerators);
  const numerators = Number.isFinite(logScale)
    ? logNumerators.map((value) =>
        Number.isFinite(value) ? 2 ** (value - logScale) : 0,
      )
    : logNumerators.map(() => 0);
  const denominator = numerators.reduce((total, value) => total + value, 0);
  const weights = denominator > 0
    ? numerators.map((numerator) => numerator / denominator)
    : numerators.map(() => 0);

  return {
    pHats,
    jacobians,
    transformedDensities: densities,
    numerators,
    denominator,
    logNumerators,
    logScale,
    weights,
  };
}

/** Log-domain form used to batch-normalize canonical WRS contributions. */
export function logCanonicalResamplingWeight(
  misWeight: number,
  canonicalPHat: number,
  sourceReservoirW: number,
  sourceToCanonicalJacobian: number,
): number {
  if (!(misWeight > 0)
      || !Number.isFinite(misWeight)
      || !(canonicalPHat > 0)
      || !Number.isFinite(canonicalPHat)
      || !(sourceReservoirW > 0)
      || !Number.isFinite(sourceReservoirW)
      || !(sourceToCanonicalJacobian > 0)
      || !Number.isFinite(sourceToCanonicalJacobian)) {
    return Number.NEGATIVE_INFINITY;
  }
  return Math.log2(misWeight)
    + Math.log2(canonicalPHat)
    + Math.log2(sourceReservoirW)
    + Math.log2(sourceToCanonicalJacobian);
}

/** Canonical resampling weight accumulated for one source reservoir sample. */
export function canonicalResamplingWeight(
  misWeight: number,
  canonicalPHat: number,
  sourceReservoirW: number,
  sourceToCanonicalJacobian: number,
): number {
  const logWeight = logCanonicalResamplingWeight(
    misWeight,
    canonicalPHat,
    sourceReservoirW,
    sourceToCanonicalJacobian,
  );
  if (!Number.isFinite(logWeight)) return 0;
  if (logWeight >= LOG_MAX_FINITE_F32) return MAX_FINITE_F32;
  const weight = 2 ** logWeight;
  return Number.isFinite(weight) && weight > 0 ? weight : 0;
}

/**
 * Apply the shader's common max-log scale to a complete canonical candidate
 * set. Returned weights are safe for WRS and retain every f64-representable
 * ratio; `logScale` recovers the omitted common factor for final W.
 */
export function normaliseCanonicalResamplingWeights(
  logWeights: readonly number[],
): { readonly weights: readonly number[]; readonly logScale: number } {
  if (logWeights.length > MAX_GRIS_TECHNIQUES) {
    throw new RangeError(
      `GRIS supports at most ${MAX_GRIS_TECHNIQUES} canonical candidates`,
    );
  }
  const logScale = Math.max(...logWeights);
  if (!Number.isFinite(logScale)) {
    return {
      weights: logWeights.map(() => 0),
      logScale: Number.NEGATIVE_INFINITY,
    };
  }
  return {
    weights: logWeights.map((value) =>
      Number.isFinite(value) ? 2 ** (value - logScale) : 0,
    ),
    logScale,
  };
}

/**
 * Recover the unscaled reservoir weight after common-max-log WRS selection.
 * This mirrors `grisFinaliseLogScaledReservoir`: selection ratios are left
 * untouched and the configured production cap is applied only to the final
 * estimator multiplier.
 */
export function finaliseLogScaledReservoirWeight(
  maxLogWeight: number,
  scaledWeightSum: number,
  selectedPHat: number,
  weightCap: number,
): number {
  if (
    !Number.isFinite(maxLogWeight)
    || !(scaledWeightSum > 0)
    || !Number.isFinite(scaledWeightSum)
    || !(selectedPHat > 0)
    || !Number.isFinite(selectedPHat)
    || !(weightCap > 0)
    || !Number.isFinite(weightCap)
  ) {
    return 0;
  }
  const logWeight =
    maxLogWeight + Math.log2(scaledWeightSum) - Math.log2(selectedPHat);
  if (!Number.isFinite(logWeight)) return 0;
  if (logWeight >= Math.log2(weightCap)) return weightCap;
  const weight = 2 ** logWeight;
  return Number.isFinite(weight) && weight > 0 ? weight : 0;
}

/**
 * Fold represented attempts exactly, including techniques whose sample weight
 * is zero. Selecting a candidate does not add a synthetic extra attempt.
 */
export function foldAttemptCount(attemptCounts: readonly number[]): number {
  let total = 0;
  for (let index = 0; index < attemptCounts.length; index += 1) {
    const count = attemptCounts[index]!;
    if (!Number.isInteger(count) || count < 0 || count > MAX_U32) {
      throw new RangeError(
        `GRIS attempt count ${index} must be a u32; got ${count}`,
      );
    }
    total = Math.min(MAX_U32, total + count);
  }
  return total;
}

/**
 * Apply the per-source confidence clamp used before a reuse batch, then fold
 * the represented attempt counts with u32 saturation. The output is not
 * clamped again: it represents the complete batch and is clamped per-domain
 * when consumed by the next temporal/spatial batch.
 */
export function foldClampedAttemptCount(
  attemptCounts: readonly number[],
  perDomainCap: number,
): number {
  if (
    !Number.isInteger(perDomainCap)
    || perDomainCap <= 0
    || perDomainCap > MAX_U32
  ) {
    throw new RangeError(
      `GRIS per-domain attempt cap must be a positive u32; got ${perDomainCap}`,
    );
  }
  return foldAttemptCount(
    attemptCounts.map((count, index) => {
      if (!Number.isInteger(count) || count < 0 || count > MAX_U32) {
        throw new RangeError(
          `GRIS attempt count ${index} must be a u32; got ${count}`,
        );
      }
      return Math.min(count, perDomainCap);
    }),
  );
}
