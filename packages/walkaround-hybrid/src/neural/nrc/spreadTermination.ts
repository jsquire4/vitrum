/**
 * spreadTermination.ts — NRC path-spread cache-termination CPU oracle.
 *
 * Müller, Rousselle, Novák, Keller 2021, "Real-time Neural Radiance Caching for
 * Path Tracing", ACM TOG 40(4) §5 ("Self-training & path termination"). The
 * renderer traces a short path; at each scattering vertex it accumulates a
 * "spread" footprint area a(x). The path is TERMINATED into the cache (the MLP
 * prediction of outgoing radiance becomes the suffix contribution) at the first
 * vertex where the accumulated spread exceeds a constant multiple of the spread
 * at the primary vertex:
 *
 *     a(x_k) > c · a_0          (Müller 2021 Eq. for the termination heuristic)
 *
 * where a_0 is the primary-vertex spread and c is the tunable constant (the
 * paper uses c ≈ 0.01 of the unit-sphere-projected area; we expose it).
 *
 * ════════════════════════════════════════════════════════════════════════════
 * The spread definition (Müller 2021 §5)
 * ════════════════════════════════════════════════════════════════════════════
 * The spread accumulated along a path is, per vertex i ≥ 1,
 *
 *     a(x_i) = ( Σ_{j=1}^{i}  sqrt( d_{j-1}² / ( p(ω_j | x_{j-1}) · |cos θ_j| ) ) )²
 *
 * i.e. the SQUARE of the running sum of per-segment terms, where for segment j:
 *   • d_{j-1}  = the distance from vertex j-1 to vertex j (the segment length),
 *   • p(ω_j…)  = the solid-angle pdf of the scattering direction sampled at the
 *                previous vertex (BSDF or guided pdf),
 *   • |cos θ_j| = |dot(n_j, ω_j)| at the arriving vertex.
 *
 * The first segment's contribution uses the PRIMARY vertex's spread a_0 as the
 * primary footprint (camera/pixel footprint); the paper sets
 *
 *     a_0 = sqrt( d_0² / ( p(ω_1 | x_0) · |cos θ_1| ) )
 *
 * squared — i.e. a_0 is the first-segment term squared, and the threshold
 * a(x_k) > c · a_0 compares the running squared sum against c times that first
 * footprint. Larger c → terminate later (longer paths, less bias, more cost);
 * smaller c → terminate sooner (more cache hits, more bias, cheaper).
 *
 * This module is the deterministic reference the WGSL predicate
 * (`spreadTermination.wgsl.ts`) is pinned against to ~1e-6 (f32). The predicate
 * is pure arithmetic (sqrt + sums + a comparison), no kinks, so the WGSL match
 * is exact to f32 epsilon.
 *
 * NOTE ON BIAS: terminating into the cache replaces the true (unbounded) path
 * suffix with the MLP's learned approximation — this is the SOURCE of NRC's
 * bounded bias. A larger c defers termination and reduces bias at higher cost;
 * the heuristic trades variance/cost for bounded bias. See the V-item.
 */

/** Per-segment inputs for one path edge (vertex j-1 → vertex j). */
export interface PathSegment {
  /** Segment length d_{j-1} = ‖x_j − x_{j-1}‖. */
  readonly dist: number;
  /** Solid-angle pdf p(ω_j | x_{j-1}) of the direction sampled at x_{j-1}. */
  readonly pdf: number;
  /** |cos θ_j| = |dot(n_j, ω_j)| at the arriving vertex x_j. */
  readonly cosTheta: number;
}

const SPREAD_MAX_FINITE_F32 = Math.fround(3.402823466e38);
const SPREAD_MAX_ROOT_F32 = Math.fround(1.844674297e19);

/**
 * GPU-f32 oracle for sqrt(d² / (p · |cosθ|)). Ordinary values use direct f32
 * arithmetic. If p·|cosθ| over/underflows, the same ratio is evaluated in log2
 * space. Saturation occurs only at sqrt(max-finite-f32), the representability
 * boundary required to keep the squared footprint finite.
 */
export function segmentSpreadTerm(seg: PathSegment): number {
  const dist = Math.fround(seg.dist);
  const pdf = Math.fround(seg.pdf);
  const cosTheta = Math.fround(seg.cosTheta);
  if (!Number.isFinite(dist) || !Number.isFinite(pdf) || !Number.isFinite(cosTheta)) {
    return SPREAD_MAX_ROOT_F32;
  }
  const absCos = Math.abs(cosTheta);
  if (!(pdf > 0) || !(absCos > 0)) return SPREAD_MAX_ROOT_F32;
  const absDist = Math.abs(dist);
  if (absDist === 0) return 0;

  const density = Math.fround(pdf * absCos);
  let term: number;
  if (Number.isFinite(density) && density > 0) {
    term = Math.fround(absDist / Math.fround(Math.sqrt(density)));
  } else {
    const logPdfCos = Math.fround(
      Math.fround(Math.log2(pdf)) + Math.fround(Math.log2(absCos)),
    );
    const logTerm = Math.fround(
      Math.fround(Math.log2(absDist)) - Math.fround(0.5 * logPdfCos),
    );
    if (!Number.isFinite(logTerm) || logTerm >= 64) return SPREAD_MAX_ROOT_F32;
    term = Math.fround(2 ** logTerm);
  }
  if (!Number.isFinite(term) || term > SPREAD_MAX_ROOT_F32) {
    return SPREAD_MAX_ROOT_F32;
  }
  return term;
}

/**
 * Accumulated spread a(x_k) after `k` segments: the SQUARE of the running sum of
 * per-segment terms (Müller 2021 §5). `segments[0]` is the primary→first-bounce
 * edge; the returned array element [i] is a(x_{i+1}) for i = 0..k-1.
 */
export function accumulatedSpread(segments: readonly PathSegment[]): Float32Array {
  const out = new Float32Array(segments.length);
  let runningSum = 0;
  for (let i = 0; i < segments.length; i++) {
    const next = Math.fround(runningSum + segmentSpreadTerm(segments[i]!));
    runningSum = Number.isFinite(next) && next <= SPREAD_MAX_ROOT_F32
      ? Math.max(next, 0)
      : SPREAD_MAX_ROOT_F32;
    out[i] = Math.min(Math.fround(runningSum * runningSum), SPREAD_MAX_FINITE_F32);
  }
  return out;
}

/**
 * Production termination spread, indexed like `segments`. `segments[0]` is the
 * camera-to-primary edge and defines a0 separately; it is deliberately excluded
 * from this running sum. Element 0 is therefore zero, and element k (k >= 1)
 * mirrors k calls to the WGSL bounce accumulator beginning from runningSum=0.
 */
export function accumulatedBounceSpread(
  segments: readonly PathSegment[],
): Float32Array {
  const out = new Float32Array(segments.length);
  let runningSum = 0;
  for (let i = 1; i < segments.length; i++) {
    const next = Math.fround(runningSum + segmentSpreadTerm(segments[i]!));
    runningSum = Number.isFinite(next) && next <= SPREAD_MAX_ROOT_F32
      ? Math.max(next, 0)
      : SPREAD_MAX_ROOT_F32;
    out[i] = Math.min(Math.fround(runningSum * runningSum), SPREAD_MAX_FINITE_F32);
  }
  return out;
}

/** The primary-vertex footprint a_0 = (first-segment term)². */
export function primarySpread(segments: readonly PathSegment[]): number {
  if (segments.length === 0) return 0;
  const t0 = segmentSpreadTerm(segments[0]!);
  return Math.min(Math.fround(t0 * t0), SPREAD_MAX_FINITE_F32);
}

export interface SpreadTerminationResult {
  /** True if the heuristic fired (a path vertex met the cache-termination test). */
  readonly terminate: boolean;
  /**
   * Index of the segment AFTER which termination occurs (the cache is queried at
   * vertex `terminateAtSegment + 1`), or -1 if the path never met the threshold
   * within the provided segments.
   */
  readonly terminateAtSegment: number;
  /** a_0 used for the comparison. */
  readonly a0: number;
}

/**
 * Evaluate the cache-termination heuristic over a path. Returns the first
 * segment index k (≥ 1, so the primary footprint is never itself terminated)
 * where a(x_{k+1}) > c · a_0. The primary vertex (k = 0) is never a cache-query
 * target — at least one bounce always happens before the cache can be queried
 * (Müller 2021: terminate the SUFFIX, never the primary hit).
 *
 * @param segments path edges in order (segments[0] = primary edge).
 * @param c        tunable termination constant (Müller §5; paper ≈ 0.01).
 */
export function evaluateSpreadTermination(
  segments: readonly PathSegment[],
  c: number,
): SpreadTerminationResult {
  const a0 = primarySpread(segments);
    const acc = accumulatedBounceSpread(segments);
  // Start at segment index 1: never terminate at the primary vertex (k=0).
  for (let k = 1; k < segments.length; k++) {
    if (acc[k]! > c * a0) {
      return { terminate: true, terminateAtSegment: k, a0 };
    }
  }
  return { terminate: false, terminateAtSegment: -1, a0 };
}
