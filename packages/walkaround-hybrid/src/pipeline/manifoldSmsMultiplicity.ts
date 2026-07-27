/** Default and hard maximum for the deliberately bounded SMS recurrence. */
export const SMS_MULTIPLICITY_TRIALS_DEFAULT = 8;
export const SMS_MULTIPLICITY_TRIALS_MAX = 32;

function assertProbability(probability: number): void {
  if (!Number.isFinite(probability) || probability <= 0 || probability > 1) {
    throw new RangeError('SMS root probability must be finite and in (0, 1]');
  }
}

function assertTrialCap(cap: number): void {
  if (!Number.isSafeInteger(cap) || cap < 1 || cap > SMS_MULTIPLICITY_TRIALS_MAX) {
    throw new RangeError(
      `SMS multiplicity trial cap must be an integer in [1, ${SMS_MULTIPLICITY_TRIALS_MAX}]`,
    );
  }
}

/**
 * Conditional expectation of W_K=min(T,K), T~Geometric(p). This is the exact
 * scalar oracle for the shader's bounded recurrence, not an unbiased 1/p
 * estimator: E[W_K]=(1-(1-p)^K)/p and therefore E[W_K] <= 1/p.
 */
export function boundedSmsMultiplicityExpectation(
  rootProbability: number,
  trialCap: number,
): number {
  assertProbability(rootProbability);
  assertTrialCap(trialCap);
  if (rootProbability === 1) return 1;
  return -Math.expm1(trialCap * Math.log1p(-rootProbability)) / rootProbability;
}

/** One realized W_K: firstMatchTrial is one-based, or null after K misses. */
export function boundedSmsMultiplicityWeight(
  firstMatchTrial: number | null,
  trialCap: number,
): number {
  assertTrialCap(trialCap);
  if (firstMatchTrial === null) return trialCap;
  if (!Number.isSafeInteger(firstMatchTrial) || firstMatchTrial < 1 || firstMatchTrial > trialCap) {
    throw new RangeError('SMS first-match trial must be an integer in [1, trialCap]');
  }
  return firstMatchTrial;
}

export interface SmsFrozenFacetProposal {
  /** Discrete probability of the frozen triangle/instance identity. */
  readonly pairPmf: number;
  /** Area used only by the uniform root-initialization draw. */
  readonly seedArea: number;
  readonly offsetNormalPdf: number;
  readonly eventPmf: number;
}

export interface SmsFrozenTopologyProposal {
  readonly chainLengthPmf: number;
  readonly endpointSelectionPmf: number;
  readonly endpointPdf: number;
  readonly facets: readonly SmsFrozenFacetProposal[];
}

function assertPositiveFinite(label: string, value: number): void {
  if (!Number.isFinite(value) || value <= 0) {
    throw new RangeError(`${label} must be finite and positive`);
  }
}

/**
 * Joint density outside the Bernoulli recurrence. Facet seed areas are checked
 * because they define valid uniform initialization laws, but their 1/area
 * densities are intentionally absent: W estimates the inverse probability mass
 * of each root's whole convergence basin under exactly those laws.
 */
export function frozenSmsTopologyProposalDensity(
  input: SmsFrozenTopologyProposal,
): number {
  assertPositiveFinite('SMS chain-length PMF', input.chainLengthPmf);
  assertPositiveFinite('SMS endpoint-selection PMF', input.endpointSelectionPmf);
  assertPositiveFinite('SMS endpoint PDF', input.endpointPdf);
  if (input.facets.length < 1 || input.facets.length > 8) {
    throw new RangeError('SMS frozen topology must contain between 1 and 8 facets');
  }
  let density = input.chainLengthPmf * input.endpointSelectionPmf * input.endpointPdf;
  for (const facet of input.facets) {
    assertPositiveFinite('SMS facet pair PMF', facet.pairPmf);
    assertPositiveFinite('SMS facet seed area', facet.seedArea);
    assertPositiveFinite('SMS offset-normal PDF', facet.offsetNormalPdf);
    assertPositiveFinite('SMS event PMF', facet.eventPmf);
    density *= facet.pairPmf * facet.offsetNormalPdf * facet.eventPmf;
  }
  assertPositiveFinite('SMS frozen-topology proposal density', density);
  return density;
}

function pcgHash(input: number): number {
  const state = (Math.imul(input >>> 0, 747_796_405) + 2_891_336_453) >>> 0;
  const shift = ((state >>> 28) + 4) >>> 0;
  const word = Math.imul(((state >>> shift) ^ state) >>> 0, 277_803_737) >>> 0;
  return ((word >>> 22) ^ word) >>> 0;
}

/** Disjoint deterministic seed stream mirrored by smsMultiplicitySeed WGSL. */
export function smsMultiplicitySeed(
  frameSeed: number,
  pixelIndex: number,
  channel: number,
  proposal: number,
  oneBasedTrial: number,
): number {
  return pcgHash(
    (frameSeed >>> 0) ^
    Math.imul(pixelIndex >>> 0, 0x9e37_79b9) ^
    Math.imul(channel >>> 0, 0x85eb_ca6b) ^
    Math.imul(proposal >>> 0, 0xc2b2_ae35) ^
    Math.imul(oneBasedTrial >>> 0, 0x27d4_eb2d) ^
    0x534d_534d,
  );
}

export interface SmsRootPoint {
  readonly x: number;
  readonly y: number;
  readonly z: number;
}

/** Ordered-vertex identity test; near but distinct roots outside tolerance stay distinct. */
export function sameSmsRoot(
  a: readonly SmsRootPoint[],
  b: readonly SmsRootPoint[],
  tolerance: number,
): boolean {
  if (a.length !== b.length || !Number.isFinite(tolerance) || tolerance < 0) return false;
  for (let index = 0; index < a.length; index += 1) {
    const lhs = a[index]!;
    const rhs = b[index]!;
    const distance = Math.hypot(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z);
    if (!Number.isFinite(distance) || distance > tolerance) return false;
  }
  return true;
}

export interface PlanarUniqueRootPreconditions {
  readonly chainLength: number;
  readonly event: 'reflection' | 'transmission';
  readonly roughness: number;
  readonly constantShadingFrame: boolean;
  readonly hasNormalMap: boolean;
  readonly hasBumpMap: boolean;
  readonly hasLayerNormalMap: boolean;
  readonly etaIncident: number;
  readonly etaTransmitted: number;
  /** Absolute cosine of the incident direction against the interface normal. */
  readonly incidentCosine: number;
  /** Signed endpoint and receiver plane distances. */
  readonly endpointPlaneDistance: number;
  readonly receiverPlaneDistance: number;
}

/**
 * Conservative proof gate for the sole exact W=1 fast path. A single planar,
 * delta transmission event between homogeneous positive media has a strictly
 * convex weighted optical path on the plane when endpoints lie in opposite
 * open half-spaces. The no-TIR check guarantees that transmission event exists.
 */
export function provesPlanarUniqueSmsRoot(
  input: PlanarUniqueRootPreconditions,
): boolean {
  if (
    input.chainLength !== 1 ||
    input.event !== 'transmission' ||
    input.roughness !== 0 ||
    !input.constantShadingFrame ||
    input.hasNormalMap ||
    input.hasBumpMap ||
    input.hasLayerNormalMap ||
    !Number.isFinite(input.etaIncident) || input.etaIncident <= 0 ||
    !Number.isFinite(input.etaTransmitted) || input.etaTransmitted <= 0 ||
    !Number.isFinite(input.incidentCosine) || input.incidentCosine <= 0 || input.incidentCosine > 1 ||
    !Number.isFinite(input.endpointPlaneDistance) ||
    !Number.isFinite(input.receiverPlaneDistance) ||
    input.endpointPlaneDistance * input.receiverPlaneDistance >= 0
  ) return false;
  const sinSquaredIncident = Math.max(0, 1 - input.incidentCosine * input.incidentCosine);
  const eta = input.etaIncident / input.etaTransmitted;
  return eta * eta * sinSquaredIncident < 1;
}
