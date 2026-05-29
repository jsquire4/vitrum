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

/**
 * The per-segment spread term sqrt( d² / (p · |cosθ|) ). Guards against a
 * zero/near-zero denominator (degenerate grazing or delta-pdf) by clamping to a
 * tiny epsilon — a degenerate segment yields a very LARGE spread term, which
 * makes the path terminate early (the correct conservative behaviour: we cannot
 * usefully extend a path whose footprint has blown up).
 */
export function segmentSpreadTerm(seg: PathSegment): number {
  const denom = Math.max(seg.pdf * Math.abs(seg.cosTheta), 1e-12);
  return Math.sqrt((seg.dist * seg.dist) / denom);
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
    runningSum += segmentSpreadTerm(segments[i]!);
    out[i] = runningSum * runningSum;
  }
  return out;
}

/** The primary-vertex footprint a_0 = (first-segment term)². */
export function primarySpread(segments: readonly PathSegment[]): number {
  if (segments.length === 0) return 0;
  const t0 = segmentSpreadTerm(segments[0]!);
  return t0 * t0;
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
  const acc = accumulatedSpread(segments);
  // Start at segment index 1: never terminate at the primary vertex (k=0).
  for (let k = 1; k < segments.length; k++) {
    if (acc[k]! > c * a0) {
      return { terminate: true, terminateAtSegment: k, a0 };
    }
  }
  return { terminate: false, terminateAtSegment: -1, a0 };
}
