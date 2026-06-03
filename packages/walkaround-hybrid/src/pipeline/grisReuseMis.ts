/**
 * grisReuseMis.ts — CPU mirror of the GRIS reconnection-shift + pairwise-MIS
 * arithmetic the GPU reuse passes run (`shaders/grisReuse.wgsl.ts`).
 *
 * This is the SINGLE SOURCE OF TRUTH for the GRIS reuse math as the WGSL
 * computes it, in the same role `pt-webgpu/src/bdpt/bdptConnectionMisFull.ts`
 * plays for BDPT MIS: an independent TS reimplementation that ports the *same
 * arithmetic* the shader runs, so one test can assert (a) it matches the
 * `@vitrum/shared-samplers/reconnectionShift.ts` oracle to machine ε, and (b)
 * the GRIS pairwise-MIS weights form a partition of unity (Σ m_i = 1).
 *
 * The shared-samplers oracle (`reconnectionShift.ts`) is the read-only first-
 * principles reference for the shift mapping + Jacobian; this file ports the
 * GPU-side arithmetic (geometry term, Jacobian-from-cached-half-G, GI target,
 * pairwise generalized-balance MIS denominators) the WGSL `grisReuse.wgsl.ts`
 * runs. The WGSL helpers and these functions are line-for-line equivalent.
 *
 * References:
 *   - Lin, Kettunen, Bitterli, Pantaleoni, Jakob, Nowrouzezahrai 2022,
 *     "Generalized Resampled Importance Sampling: Foundations of ReSTIR",
 *     SIGGRAPH 2022 — §5 (reconnection shift), Eq. 12 (Jacobian),
 *     §"pairwise MIS" (the practical generalized-balance form).
 *
 * @module grisReuseMis
 */

export type Vec3 = readonly [number, number, number];

function sub(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}
function dot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

const INV_PI = 0.31830988618;

/** Rec.709 luminance — mirrors the WGSL `luminance()` weights used by the
 *  GI target. Kept local so the mirror has no cross-package dependency. */
function luminance(c: Vec3): number {
  return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
}

/**
 * Destination-cosine reconnection-edge geometry term — the GPU
 * `grisReconnectionGeometryTerm`, identical to the oracle's
 * `reconnectionGeometryTerm`:
 *   G(x1 ↔ x2) = |cos θ_out(x2)| / ‖x1 − x2‖²
 * Returns 0 for a degenerate (coincident) edge or a tangent connection.
 */
export function reconnectionGeometryTerm(x1: Vec3, x2: Vec3, n2: Vec3): number {
  const d = sub(x2, x1);
  const dist2 = dot(d, d);
  if (dist2 <= 0) return 0;
  const dist = Math.sqrt(dist2);
  const cosOut = Math.abs(dot(n2, d) / dist);
  return cosOut / dist2;
}

/**
 * Shift Jacobian |∂T/∂·| = G(shifted) / G(base) computed the GPU way: the base
 * half-G is supplied directly (recovered from the Phase-0 cache
 * cosReconOut / distRecon²) and the shifted half-G is recomputed from the
 * offset primary vertex. Mirrors `grisShiftJacobian`. Returns 0 when either
 * half-G is 0 (degenerate / non-invertible shift).
 */
export function shiftJacobian(
  gBase: number,
  xvOffset: Vec3,
  xs: Vec3,
  ns: Vec3,
): number {
  if (gBase <= 0) return 0;
  const gShifted = reconnectionGeometryTerm(xvOffset, xs, ns);
  return gShifted / gBase;
}

/**
 * Recover the cached base half-G from the Phase-0 reservoir fields exactly as
 * the GPU does: `gBase = cosReconOut / distRecon²`. This is the value the
 * reservoir stores (cosReconOut = |cos θ_out| at xs; distRecon = ‖xv − xs‖)
 * and is algebraically identical to `reconnectionGeometryTerm(xv, xs, ns)`.
 */
export function cachedBaseHalfG(cosReconOut: number, distRecon: number): number {
  if (distRecon <= 0) return 0;
  return cosReconOut / (distRecon * distRecon);
}

/**
 * GI target p̂ in the domain rooted at `xv`:
 *   p̂(z) = luminance(Lo) · max(0, cos(nv, xv→xs)) · INV_PI
 * Mirrors `grisTargetAt`. Returns 0 for a degenerate edge.
 */
export function giTargetAt(xv: Vec3, nv: Vec3, xs: Vec3, Lo: Vec3): number {
  const d = sub(xs, xv);
  const dist2 = dot(d, d);
  if (dist2 < 1e-8) return 0;
  const inv = 1 / Math.sqrt(dist2);
  const wi: Vec3 = [d[0] * inv, d[1] * inv, d[2] * inv];
  const cosTheta = Math.max(0, dot(nv, wi));
  return luminance(Lo) * cosTheta * INV_PI;
}

/**
 * One domain participating in the GRIS pairwise / generalized-balance MIS. The
 * canonical pixel r and each neighbour q is a {@link GrisDomain}: a primary
 * vertex (xv/nv) and its per-domain confidence weight c (the reservoir M).
 */
export interface GrisDomain {
  /** Primary/visible vertex of this domain. */
  readonly xv: Vec3;
  /** Normal at the visible vertex. */
  readonly nv: Vec3;
  /** This domain's stored reconnection sample (held fixed by the shift). */
  readonly xs: Vec3;
  readonly ns: Vec3;
  readonly Lo: Vec3;
  /** Per-domain confidence weight (the reservoir's M count, as a number). */
  readonly c: number;
}

/** A reconnection sample (the world-space reconnection vertex + its suffix
 *  radiance) that the reconnection shift re-roots onto each domain's primary
 *  vertex. The shift keeps {xs, ns, Lo} fixed and swaps the primary vertex. */
export interface GrisSample {
  readonly xs: Vec3;
  readonly ns: Vec3;
  readonly Lo: Vec3;
}

/**
 * GRIS generalized balance heuristic (Lin 2022, generalized balance heuristic /
 * §pairwise MIS) for ONE fixed reconnection sample `y` evaluated across a set of
 * {@link domains}. The MIS weight of technique (domain) `i` for the sample `y`
 * is
 *
 *   m_i(y) = (c_i · p̂_i(T_{·→i} y)) / Σ_j ( c_j · p̂_j(T_{·→j} y) )
 *
 * where the reconnection shift T_{·→i} re-roots `y`'s reconnection vertex onto
 * domain i's primary vertex (xv_i), so the per-domain target is
 * `giTargetAt(domains[i].xv, domains[i].nv, y.xs, y.Lo)`. The shift keeps the
 * reconnection vertex shared — only the primary vertex changes per domain —
 * which is exactly the reconnection shift. The shift JACOBIAN enters the
 * resampling weight (applied by the GPU reuse loop), not these target ratios;
 * each target is evaluated in its domain's native measure so the densities are
 * commensurable.
 *
 * For a FIXED sample `y` these weights PARTITION UNITY: Σ_i m_i(y) = 1 — this is
 * the unbiasedness invariant of the generalized balance heuristic (a sample
 * could have been produced by ANY of the domains' techniques; the weights of
 * those techniques for that one sample sum to 1).
 *
 * @returns the MIS weights `m_i(y)`, one per domain, in input order. All-zero
 *   only when `y` shifts to a zero target in every domain (then the sample
 *   carries no contribution at all).
 */
export function grisGeneralizedBalanceWeights(
  domains: ReadonlyArray<GrisDomain>,
  y: GrisSample,
): number[] {
  const n = domains.length;
  const weights = new Array<number>(n).fill(0);
  // Denominator: Σ_j c_j · p̂_j(T_{·→j} y) — the same sample y shifted into
  // every domain j (xs/Lo fixed, primary vertex swapped to xv_j).
  let denom = 0;
  const numer = new Array<number>(n).fill(0);
  for (let j = 0; j < n; j += 1) {
    const dj = domains[j]!;
    const t = dj.c * giTargetAt(dj.xv, dj.nv, y.xs, y.Lo);
    numer[j] = t;
    denom += t;
  }
  if (denom <= 0) return weights;
  for (let i = 0; i < n; i += 1) weights[i] = numer[i]! / denom;
  return weights;
}

/**
 * Streaming pairwise denominator the GPU reuse loop uses when folding ONE
 * neighbour `q` into the canonical pixel `r` (mirrors
 * `grisPairwiseDenomNeighbor`):
 *   denom = c_r · p̂_r(shifted q sample) + c_q · p̂_q(native q sample)
 * The neighbour's MIS weight is then `m_q = (c_q · p̂_q_native) / denom`.
 */
export function pairwiseDenomNeighbor(
  cR: number,
  pHatR_atQsample: number,
  cQ: number,
  pHatQ_native: number,
): number {
  return cR * pHatR_atQsample + cQ * pHatQ_native;
}

/**
 * Streaming pairwise denominator for the CANONICAL sample against neighbour
 * `q` (mirrors `grisPairwiseDenomCanonical`):
 *   denom = c_r · p̂_r(native r sample) + c_q · p̂_q(shifted r sample)
 * The canonical MIS weight is `m_r = (c_r · p̂_r_native) / denom`.
 */
export function pairwiseDenomCanonical(
  cR: number,
  pHatR_native: number,
  cQ: number,
  pHatQ_atRsample: number,
): number {
  return cR * pHatR_native + cQ * pHatQ_atRsample;
}
