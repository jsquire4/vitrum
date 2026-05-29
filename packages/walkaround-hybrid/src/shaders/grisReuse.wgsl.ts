/**
 * grisReuse.wgsl.ts — GRIS / ReSTIR-PT reconnection-shift + pairwise-MIS WGSL.
 *
 * Phases 1+2 of evolving ReSTIR-GI toward GRIS (Lin, Kettunen, Bitterli,
 * Pantaleoni, Jakob, Nowrouzezahrai — "Generalized Resampled Importance
 * Sampling: Foundations of ReSTIR", SIGGRAPH 2022).
 *
 * This module is the GPU side of the CPU oracle
 * `@vitrum/shared-samplers/reconnectionShift.ts`. Every arithmetic helper here
 * mirrors the oracle EXACTLY (verified by the TS mirror `grisReuseMis.ts` +
 * its test, the same discipline `bdptConnectionMisFull.ts` follows for BDPT
 * MIS). The functions are pure (no buffer reads) so they can be unit-pinned
 * against the oracle on shared fixtures.
 *
 * ════════════════════════════════════════════════════════════════════════════
 * Phase 1 — reconnection shift T + Jacobian
 * ════════════════════════════════════════════════════════════════════════════
 * A neighbour pixel q's reservoir stores a reconnection sample rooted at q's
 * visible vertex (q.xv → q.xs, with cached suffix Lo at q.xs). To reuse it at
 * the current pixel r we apply the reconnection shift
 *
 *   T : (q.xv, q.xs, q.ns) ↦ (r.xv, q.xs, q.ns)
 *
 * which holds the reconnection vertex q.xs (and its suffix radiance) FIXED in
 * world space and re-roots the path on r.xv via a fresh edge r.xv → q.xs. The
 * change-of-variables Jacobian is the ratio of the destination-cosine half-G
 * geometry terms (oracle `reconnectionGeometryTerm` / `reconnectionJacobian`,
 * Lin 2022 Eq. 12):
 *
 *   G(x1 ↔ x2) = |cos θ_out(x2)| / ‖x1 − x2‖²        (destination cosine only)
 *   |∂T/∂·|    = G(r.xv ↔ q.xs) / G(q.xv ↔ q.xs)
 *
 * The base half-G denominator is recovered from the Phase-0 cache
 * (q.cosReconOut / q.distRecon²) so the shift needs no re-trace of the base
 * edge. The shifted numerator G(r.xv ↔ q.xs) is recomputed from r.xv, q.xs,
 * q.ns. The reused sample's resampling weight is multiplied by this Jacobian.
 *
 * Reconnection visibility (required for unbiasedness): the shift maps to ZERO
 * contribution if the edge r.xv → q.xs is occluded, degenerate, or backfacing,
 * or if the path prefixes are incompatible (prefixVertexCount mismatch). The
 * caller traces the visibility ray through the BVH; these helpers handle the
 * geometric rejection.
 *
 * ════════════════════════════════════════════════════════════════════════════
 * Phase 2 — GRIS generalized-balance-heuristic (pairwise) MIS
 * ════════════════════════════════════════════════════════════════════════════
 * The canonical sample (rooted at r) and each shifted neighbour are combined
 * with the GRIS generalized balance heuristic in its practical pairwise form
 * (Lin 2022 §pairwise MIS). For a domain set {r, q_1, …, q_K} the MIS weight of
 * a sample drawn in domain i, evaluated at the canonical domain r, is
 *
 *   m_i = (c_i · p̂_i(z_i))
 *       / Σ_j ( c_j · p̂_j(T_{i→j} z_i) · |∂T_{i→j}/∂·| )
 *
 * where p̂ is the unnormalised target (here the GI target luminance(Lo)·cosθ·INV_PI
 * in the SAME domain the sample lives), c are the per-domain confidence weights
 * (the reservoir M counts), and the shift Jacobian enters every cross term that
 * maps domain i's sample into domain j.
 *
 * The per-neighbour resampling weight the reuse loop accumulates is
 *
 *   w_i = m_i · p̂_r(T_{i→r} z_i) · W_i · |∂T_{i→r}/∂·|
 *
 * where W_i is the NEIGHBOUR's unbiased contribution weight (its reservoir UCW).
 * There is NO division by the source pdf here: the inputs are RESERVOIRS, not
 * fresh BSDF draws — q's source pdf was already consumed when its W_i was
 * finalised (W_i = w_sum/(M·p̂_i)), so W_i·p̂ is already the unbiased
 * contribution and the Jacobian alone carries the reconnection-edge measure
 * conversion (Lin 2022, Alg. 3 / Eq. 9). Re-dividing by p_src double-discounts
 * the pdf and diverges the temporal feedback loop; see the spatialGi/temporalGi
 * GRIS branches for the load-bearing note.
 *
 * `grisPairwiseMisCanonical` / `grisPairwiseMisNeighbor` below build the two
 * pairwise denominators the per-neighbour reuse loop needs (the spatial/temporal
 * passes accumulate the canonical-vs-this-neighbour pair as each neighbour is
 * folded in — the streaming pairwise form, Lin 2022 §"pairwise MIS").
 *
 * References:
 *   - Lin et al. 2022 (GRIS), §5 (reconnection shift), Eq. 12 (shift Jacobian),
 *     §"pairwise MIS" (the practical generalized-balance form).
 *   - Bitterli et al. 2020/2021 (ReSTIR DI/GI base reservoir + reconnection).
 *
 * @module grisReuse
 */

import type { WgslModule } from '../pipeline/wgslComposer.js';

export const GRIS_REUSE_WGSL = /* wgsl */ `// ============================================================
// GRIS reconnection shift + pairwise MIS (Lin et al. 2022)
// Mirrors @vitrum/shared-samplers/reconnectionShift.ts exactly.
// ============================================================

// G(x1 ↔ x2) = |cos θ_out(x2)| / ‖x1 − x2‖² — destination-cosine "half-G",
// the exact form of the oracle's reconnectionGeometryTerm. Returns 0 on a
// degenerate (coincident) edge or a tangent connection (cos θ_out = 0).
fn grisReconnectionGeometryTerm(x1: vec3f, x2: vec3f, n2: vec3f) -> f32 {
  let d = x2 - x1;
  let dist2 = dot(d, d);
  if (dist2 <= 0.0) { return 0.0; }
  let dist = sqrt(dist2);
  let cosOut = abs(dot(n2, d) / dist);
  return cosOut / dist2;
}

// Shift Jacobian |∂T/∂·| = G(shifted) / G(base) (oracle reconnectionJacobian,
// Lin 2022 Eq. 12). The base half-G is supplied directly (recovered from the
// Phase-0 cache cosReconOut / distRecon²) so no base re-trace is needed; the
// shifted half-G is recomputed from the offset primary vertex.
//   gBase = |cos θ_out(base)| / distRecon²   (the cached base half-G)
// Returns 0 when either half-G is 0 (degenerate / non-invertible shift).
fn grisShiftJacobian(
  gBase:      f32,    // base reconnection-edge half-G (cached)
  xvOffset:   vec3f,  // offset (current-pixel) primary/visible vertex
  xs:         vec3f,  // shared reconnection vertex (held fixed)
  ns:         vec3f,  // reconnection-vertex normal (held fixed)
) -> f32 {
  if (gBase <= 0.0) { return 0.0; }
  let gShifted = grisReconnectionGeometryTerm(xvOffset, xs, ns);
  return gShifted / gBase;
}

// The GI target function p̂ in the domain whose primary vertex is xv:
//   p̂(z) = luminance(Lo) · max(0, cos(nv, xv→xs)) · INV_PI
// This is identical to the per-pixel pHat the existing RIS reuse computes; it
// is the target the GRIS pairwise-MIS denominator weights each shifted sample
// by, evaluated in the domain the sample is being mapped INTO. Returns 0 for a
// degenerate edge (caller treats 0 target as "this term contributes nothing").
fn grisTargetAt(xv: vec3f, nv: vec3f, xs: vec3f, Lo: vec3f) -> f32 {
  let d = xs - xv;
  let dist2 = dot(d, d);
  if (dist2 < 1e-8) { return 0.0; }
  let wi = d * inverseSqrt(dist2);
  let cosTheta = max(0.0, dot(nv, wi));
  return luminance(Lo) * cosTheta * INV_PI;
}

// ── GRIS pairwise MIS (Lin 2022 §"pairwise MIS") ───────────────────────────
//
// The streaming spatial/temporal reuse folds one neighbour at a time. For the
// pair {canonical r, neighbour q} the generalized balance heuristic gives:
//
//   denomNeighbor = c_r · p̂_r(T_{q→r} z_q)              (q's sample shifted to r)
//                 + c_q · p̂_q(z_q)                       (q's sample in its own domain)
//   m_q           = (c_q · p̂_q(z_q)) / denomNeighbor
//
// The shift Jacobian re-weights the resampling weight (multiplied in by the
// caller after m_q), NOT the target ratio — the target is evaluated in each
// term's NATIVE domain so the densities are commensurable (the Jacobian
// converts the resampling measure, the targets stay per-domain). This matches
// the oracle's separation: reconnectionJacobian is the measure conversion,
// the BSDF/cosine target factors are carried by the target function.
//
// cR / cQ are the per-domain confidence weights (reservoir M counts as
// f32). pHatR_atQsample is p-hat evaluated at the canonical pixel r for q's
// reconnection sample (the shift target). pHatQ_native is p-hat at q for its
// own sample. The caller forms m_q = numer/denom and guards denom > 0.
fn grisPairwiseDenomNeighbor(
  cR: f32, pHatR_atQsample: f32,
  cQ: f32, pHatQ_native: f32,
) -> f32 {
  return cR * pHatR_atQsample + cQ * pHatQ_native;
}

// Canonical-sample MIS weight for the streaming pair {r, q}. The canonical
// reservoir's own sample lives in domain r; mapping it INTO q's domain uses the
// INVERSE shift (Jacobian reciprocal), but for the GI target the cross term is
// just p̂ evaluated at q for r's sample. m_r = (c_r·p̂_r) / (c_r·p̂_r + c_q·p̂_q_atRsample).
fn grisPairwiseDenomCanonical(
  cR: f32, pHatR_native: f32,
  cQ: f32, pHatQ_atRsample: f32,
) -> f32 {
  return cR * pHatR_native + cQ * pHatQ_atRsample;
}

`;

/** GRIS reconnection-shift + pairwise-MIS WGSL include-graph entry.
 *  `grisTargetAt` calls `luminance` (sharedPrimitives) and uses `INV_PI`
 *  (walkaroundUbo). Declared as `requires` so the topo-sort emits this module
 *  AFTER those definitions — the consuming GI passes also require them, but
 *  the explicit dependency keeps the emit order correct regardless of the root
 *  module's own `requires` ordering. (`inverseSqrt` is a WGSL builtin.) */
export const GRIS_REUSE_MODULE: WgslModule = {
  name: 'grisReuse',
  source: GRIS_REUSE_WGSL,
  requires: ['walkaroundUbo', 'sharedPrimitives'],
};
