/**
 * Independent CPU oracle for the bounded GRIS reuse implemented by the
 * walkaround renderer.
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
const MAX_WEIGHTED_DENSITY = MAX_FINITE_F32 / MAX_GRIS_TECHNIQUES;

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

function weightedDensity(attempts: number, density: number): number {
  if (!(density > 0) || !Number.isFinite(density)) return 0;
  // Six bounded terms form the shader denominator. Capping each term at one
  // sixth of max-f32 guarantees that neither multiplication nor accumulation
  // can become Infinity while retaining monotonic mass in the extreme tail.
  if (density > MAX_WEIGHTED_DENSITY / attempts) {
    return MAX_WEIGHTED_DENSITY;
  }
  return attempts * density;
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
  readonly numerators: readonly number[];
  readonly denominator: number;
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
  const numerators = new Array<number>(domains.length).fill(0);
  let denominator = 0;

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
    const numerator = weightedDensity(attempts, density);

    pHats[index] = pHat;
    jacobians[index] = jacobian;
    densities[index] = density;
    numerators[index] = numerator;
    denominator += numerator;
  }

  const weights = denominator > 0
    ? numerators.map((numerator) => numerator / denominator)
    : numerators.map(() => 0);

  return {
    pHats,
    jacobians,
    transformedDensities: densities,
    numerators,
    denominator,
    weights,
  };
}

/** Canonical resampling weight accumulated for one source reservoir sample. */
export function canonicalResamplingWeight(
  misWeight: number,
  canonicalPHat: number,
  sourceReservoirW: number,
  sourceToCanonicalJacobian: number,
): number {
  if (!(misWeight > 0)
      || !(canonicalPHat > 0)
      || !(sourceReservoirW > 0)
      || !(sourceToCanonicalJacobian > 0)) return 0;
  const weight = misWeight * canonicalPHat * sourceReservoirW * sourceToCanonicalJacobian;
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
