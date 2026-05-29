/**
 * Sprint 17 — ReSTIR-GI spatial reuse.
 *
 * Per half-res pixel, pull K_SPATIAL = 5 random neighbours from within a
 * disc of radius SPATIAL_RADIUS px. For each accepted neighbour:
 *
 *   1. Geometric-consistency reject if normal mismatch or depth jump.
 *   2. Compute reconnection-shift jacobian J at the current pixel.
 *   3. Compute p̂(z_q) at the current pixel.
 *   4. Combine into RIS reservoir with weight  w_q = p̂(z_q) · W_q · M_q · J.
 *
 * Finalise W from the chosen-sample p̂. M clamps at 500 to bound variance.
 *
 * Run twice per frame (current → spatial, then spatial → current) with
 * different RNG seeds for full-resolution-equivalent coverage.
 *
 * Bindings (ping-pong via two distinct bind groups that swap
 * reservoirGiCurrent / reservoirGiSpatial between in and out):
 *   @group(0) @binding(0) input  reservoir (storage, read)
 *   @group(0) @binding(1) output reservoir (storage, read_write)
 *   @group(0) @binding(2) WalkaroundUBO    (uniform)
 *   @group(1)             scene BVH/TLAS   (read-only storage) — used by the
 *                         GRIS reconnection-visibility ray when
 *                         ubo.restirPtReuse == 1; inert when the gate is 0.
 *
 * ════════════════════════════════════════════════════════════════════════════
 * GRIS / ReSTIR-PT reconnection-shift reuse (Phases 1+2; Lin et al. 2022)
 * ════════════════════════════════════════════════════════════════════════════
 * When ubo.restirPtReuse == 1 the per-neighbour combine takes the UNBIASED
 * GRIS path instead of the legacy clamped-Jacobian reuse:
 *   1. Reconnection shift T: re-root q's reconnection vertex (q.xs/ns/Lo) onto
 *      the current pixel's primary vertex (rCenter.xv/nv) — a fresh edge
 *      rCenter.xv → q.xs.
 *   2. Shift Jacobian |∂T/∂·| = G(rCenter.xv ↔ q.xs)/G_base, where the base
 *      half-G is recovered from q's Phase-0 cache (cosReconOut/distRecon²) —
 *      mirrors @vitrum/shared-samplers reconnectionJacobian (Eq. 12).
 *   3. Reconnection VISIBILITY: trace rCenter.xv → q.xs through the BVH; if
 *      occluded / degenerate / backfacing / prefix-incompatible the shift maps
 *      to zero contribution (reject this neighbour). Required for unbiasedness.
 *   4. Pairwise generalized-balance MIS (§pairwise MIS): m_q weights q's sample
 *      by p̂ evaluated in each domain, the shift Jacobian entering the
 *      resampling weight; the combined reservoir is an unbiased RIS estimator.
 * The gate-OFF (== 0) branch is the verbatim Sprint-17 reuse, so the rendered
 * output is BIT-IDENTICAL when restirPtReuse is 0 (default).
 */

import type { WgslModule } from '../pipeline/wgslComposer.js';

export const SPATIAL_GI_WGSL = /* wgsl */ `

@group(0) @binding(0) var<storage, read>       sgi_resIn:  array<u32>;
@group(0) @binding(1) var<storage, read_write> sgi_resOut: array<u32>;
@group(0) @binding(2) var<uniform> ubo: WalkaroundUBO;

// Scene group (group 1) — BVH + TLAS buffers for the GRIS reconnection-
// visibility ray. Same binding layout as the shared scene bind group
// (bindGroupDescriptors 'scene'); bound to every spatial-GI dispatch for
// pipeline-layout compat. Read ONLY when ubo.restirPtReuse == 1 (the visibility
// ray is inside the GRIS branch), so the legacy path never traverses the BVH.
@group(1) @binding(0) var<storage, read> sgi_bvh:          array<BVHNode>;
@group(1) @binding(1) var<storage, read> sgi_bvh_index:    array<vec4u>;
@group(1) @binding(2) var<storage, read> sgi_bvh_position: array<vec4f>;
@group(1) @binding(6) var<storage, read> sgi_tlasNodes: array<BVHNode>;
@group(1) @binding(7) var<storage, read> sgi_tlasInstanceIndices: array<u32>;
@group(1) @binding(8) var<storage, read> sgi_tlasBlasRoots: array<u32>;
@group(1) @binding(9) var<storage, read> sgi_tlasInstanceWorldToLocal: array<vec4f>;
@group(1) @binding(10) var<storage, read> sgi_tlasInstanceLocalToWorld: array<vec4f>;

// Normal bias for the GRIS reconnection-visibility ray origin (lift off the
// surface so the ray does not self-intersect the visible point's triangle).
const GRIS_NORMAL_BIAS: f32 = 1e-3;

const K_SPATIAL_GI: u32 = 5u;
const M_CLAMP_SPATIAL: u32 = 500u;
// SPATIAL_RADIUS_GI / NORMAL_DOT_MIN_S / COPLANAR_TOL_S now live on the
// WalkaroundUBO so library consumers can override the Cornell-tuned defaults:
//   ubo.restirGiSpatialRadiusPx         (default 12.0 — half-res pixels)
//   ubo.restirGiSpatialNormalDotMin     (default 0.906 ≈ cos(25°))
//   ubo.restirGiSpatialCoplanarTol      (default 0.05 — 5 cm world units)
//
// Coplanar-distance tolerance rationale: neighbour must lie within this
// perpendicular distance of the centre pixel's tangent plane.  Replaces the
// older camera-distance ratio test (DEPTH_REL_TOL_S) which rejected
// neighbours in corner geometry where the same wall recedes from the camera
// at a steep angle — verified via reservoir probe that the camera-ratio
// test was rejecting essentially all 5 neighbours on left-wall-near-back-
// corner pixels, locking each pixel into its own initial-RIS sample.  The
// plane test instead asks "are these points on the same surface" which is
// what the spatial filter actually needs.

fn sampleDiscPx(rng: ptr<function, u32>) -> vec2f {
  let r = ubo.restirGiSpatialRadiusPx * sqrt(rand_f32(rng));
  let phi = 6.2831853 * rand_f32(rng);
  return vec2f(r * cos(phi), r * sin(phi));
}

// GRIS reconnection visibility — is the shifted edge xv → xs unoccluded? Trace
// a shadow ray through the scene BVH/TLAS (group 1). Returns true if the
// connection is clear (the suffix radiance Lo at xs is actually visible from
// the current pixel's primary vertex). skipGlass=true matches the risGi /
// ReSTIR shadow-ray convention (light passes through glass; tint handled
// elsewhere). Only called inside the restirPtReuse==1 branch.
fn grisReconnectionVisible(xv: vec3f, nv: vec3f, xs: vec3f) -> bool {
  let toS = xs - xv;
  let dist = length(toS);
  if (dist < 1e-4) { return false; }
  let wi = toS / dist;
  let orig = xv + nv * GRIS_NORMAL_BIAS;
  let occ = traceSceneAny(
    ubo.bvhMode, ubo.tlasNodeCount,
    &sgi_bvh_index, &sgi_bvh_position, &sgi_bvh,
    &sgi_tlasNodes, &sgi_tlasInstanceIndices, &sgi_tlasBlasRoots,
    &sgi_tlasInstanceWorldToLocal, &sgi_tlasInstanceLocalToWorld,
    orig, wi, dist - 2e-3, ubo.triIntersectEpsilon, true);
  return !occ;
}

@compute @workgroup_size(8, 8, 1)
fn spatialGiMain(@builtin(global_invocation_id) gid: vec3u) {
  let fullDims = ubo.screenSize;
  let halfDims = fullDims / 2u;
  if (any(gid.xy >= halfDims)) { return; }

  let pixelIdx = gid.y * halfDims.x + gid.x;
  var rCenter = loadReservoirGI_ro(&sgi_resIn, pixelIdx);

  // No surface here — skip reuse, copy through.
  if (rCenter.M == 0u) {
    storeReservoirGI_rw(&sgi_resOut, pixelIdx, rCenter);
    return;
  }

  var rng = pcgInit(
    gid.x ^ (ubo.frameSeed * 0xA127u),
    gid.y ^ (ubo.frameSeed * 0x271Au),
    ubo.frameSeed ^ 0xBCD3u,
  );

  // GRIS gate. When 1, the per-neighbour combine takes the unbiased
  // reconnection-shift + visibility + pairwise-MIS path; when 0 it runs the
  // verbatim Sprint-17 clamped-Jacobian reuse (bit-identical default).
  let grisOn = (ubo.restirPtReuse == 1u);

  // rOut: legacy reuse mutates the centre reservoir in place (rOut = rCenter);
  // GRIS reuse builds a FRESH reservoir so the canonical sample can be folded
  // in with its OWN pairwise-MIS weight (Lin 2022 §pairwise MIS — the canonical
  // technique is one of the resampling techniques, not an un-weighted base).
  var rOut: ReservoirPT;
  if (grisOn) {
    rOut = emptyReservoirGI();
    rOut.xv = rCenter.xv; rOut.nv = rCenter.nv;
    rOut.prefixVertexCount = rCenter.prefixVertexCount;
  } else {
    rOut = rCenter;
  }

  // Canonical (this-pixel) target p̂_r(z_r) for its OWN reservoir sample — the
  // numerator of the canonical pairwise-MIS weight. cR is the canonical
  // confidence (its M count). Only used when grisOn.
  let pHatCanonNative = grisTargetAt(rCenter.xv, rCenter.nv, rCenter.xs, rCenter.Lo);
  let cR = f32(rCenter.M);
  // Canonical-sample MIS weight accumulator + neighbour count for the
  // defensive 1/(K+1) pairwise-MIS normalisation (Bitterli 2020 / Lin 2022
  // §pairwise MIS): m_canon = (1/(K+1))·(1 + Σ_i pairwise-canonical-term_i).
  var canonMisAccum: f32 = 0.0;
  var acceptedNeighbors: u32 = 0u;

  for (var i: u32 = 0u; i < K_SPATIAL_GI; i = i + 1u) {
    let off = sampleDiscPx(&rng);
    let qx = i32(gid.x) + i32(round(off.x));
    let qy = i32(gid.y) + i32(round(off.y));
    if (qx < 0 || qy < 0
     || u32(qx) >= halfDims.x || u32(qy) >= halfDims.y) { continue; }
    if (qx == i32(gid.x) && qy == i32(gid.y)) { continue; }

    let qIdx = u32(qy) * halfDims.x + u32(qx);
    let rQ = loadReservoirGI_ro(&sgi_resIn, qIdx);
    if (rQ.M == 0u || rQ.W <= 0.0) { continue; }

    // Geometric-consistency: normal alignment + coplanarity to centre pixel.
    if (dot(rCenter.nv, rQ.nv) < ubo.restirGiSpatialNormalDotMin) { continue; }
    let planeDist = abs(dot(rQ.xv - rCenter.xv, rCenter.nv));
    if (planeDist > ubo.restirGiSpatialCoplanarTol) { continue; }

    if (grisOn) {
      // ── GRIS reconnection-shift reuse (Lin 2022 §5 + Eq. 12 + pairwise MIS) ──
      // Shift-compatibility gate: only reconnection samples with a matching
      // path prefix take the reconnection shift (single-bounce prefix = 1u).
      // A 0 prefix is an unpopulated Phase-0 cache → skip.
      if (rQ.prefixVertexCount != rCenter.prefixVertexCount
       || rQ.prefixVertexCount == 0u) { continue; }

      let Mq = min(rQ.M, M_CLAMP_SPATIAL);
      let cQ = f32(Mq);

      // ── Pairwise-canonical term: re-root the CANONICAL sample onto q (the
      // INVERSE shift T⁻¹) and balance it against q's own domain. This is the
      // m_canon partner of the m_q below; accumulated for the defensive
      // canonical MIS weight. The candidate-technique SET is every geometry- +
      // prefix-compatible neighbour (counted here, before the visibility /
      // Jacobian rejections): a neighbour that later fails visibility is a
      // technique that produced a ZERO-weight sample — it still counts in the
      // (K+1) defensive denominator, which keeps the estimator unbiased (the
      // defensive MIS never over-counts; empty techniques just shift weight to
      // the canonical). ──
      if (rCenter.M > 0u && pHatCanonNative > 1e-9) {
        let pHatCanon_atQ = grisTargetAt(rQ.xv, rQ.nv, rCenter.xs, rCenter.Lo);
        let denomC = grisPairwiseDenomCanonical(cR, pHatCanonNative, cQ, pHatCanon_atQ);
        canonMisAccum += select(0.0, (cR * pHatCanonNative) / denomC, denomC > 1e-12);
      }
      acceptedNeighbors += 1u;

      // Base half-G recovered from q's Phase-0 cache (cosReconOut/distRecon²) —
      // identical to reconnectionGeometryTerm(rQ.xv, rQ.xs, rQ.ns).
      let gBase = select(0.0, rQ.cosReconOut / (rQ.distRecon * rQ.distRecon),
                         rQ.distRecon > 1e-6);
      if (gBase <= 0.0) { continue; }

      // Shift Jacobian |∂T/∂·| = G(rCenter.xv ↔ q.xs) / gBase.
      let J = grisShiftJacobian(gBase, rCenter.xv, rQ.xs, rQ.ns);
      if (J <= 0.0) { continue; }

      // q's reconnection sample evaluated at the canonical pixel r (the shift
      // target p̂_r(T z_q)) and in q's own domain (p̂_q(z_q)).
      let pHatQ_atR = grisTargetAt(rCenter.xv, rCenter.nv, rQ.xs, rQ.Lo);
      if (pHatQ_atR < 1e-9) { continue; }
      let pHatQ_native = grisTargetAt(rQ.xv, rQ.nv, rQ.xs, rQ.Lo);
      if (pHatQ_native < 1e-9) { continue; }

      // Reconnection VISIBILITY — required for unbiasedness. If the shifted
      // edge rCenter.xv → q.xs is occluded the shift maps to zero contribution.
      if (!grisReconnectionVisible(rCenter.xv, rCenter.nv, rQ.xs)) { continue; }

      // Pairwise generalized-balance MIS weight for q's sample:
      //   m_q = (c_q·p̂_q_native) / (c_r·p̂_r(T z_q) + c_q·p̂_q_native)
      let denomQ = grisPairwiseDenomNeighbor(cR, pHatQ_atR, cQ, pHatQ_native);
      let m_q = select(0.0, (cQ * pHatQ_native) / denomQ, denomQ > 1e-12);

      // GRIS resampling weight: w_q = m_q · p̂_r(T z_q) · W_q · |∂T/∂·| / p_src.
      // p_src is q's source pdf (Phase-0 cache pdfReconBsdf, the SA pdf that
      // generated the base reconnection direction). The Jacobian carries the
      // reconnection-edge measure conversion (Lin 2022 Eq. 12).
      let pSrc = max(rQ.pdfReconBsdf, 1e-12);
      let w_q = m_q * pHatQ_atR * rQ.W * J / pSrc;
      let oldM = rOut.M;
      updateReservoirGI(&rOut, rQ.xs, rQ.ns, rQ.Lo, w_q, &rng);
      rOut.M = oldM + Mq;
    } else {
      // ── Legacy Sprint-17 reuse (bit-identical when restirPtReuse == 0) ──
      // Jacobian shift: rQ's reservoir holds (xs, ns, Lo) seen from rQ.xv;
      // re-weight it for evaluation at rCenter.xv.
      let J = jacobianReconnectionShift(rCenter.xv, rCenter.nv, rQ.xv, rQ.xs, rQ.ns);
      if (J <= 0.0) { continue; }

      // p̂ at center pixel.
      let toS = rQ.xs - rCenter.xv;
      let distS = length(toS);
      if (distS < 1e-4) { continue; }
      let wiZ = toS / distS;
      let cosThetaZ = max(0.0, dot(rCenter.nv, wiZ));
      let pHatZ = luminance(rQ.Lo) * cosThetaZ * INV_PI;
      if (pHatZ < 1e-9) { continue; }

      let Mq = min(rQ.M, M_CLAMP_SPATIAL);
      let w_q = pHatZ * rQ.W * f32(Mq) * J;
      let oldM = rOut.M;
      updateReservoirGI(&rOut, rQ.xs, rQ.ns, rQ.Lo, w_q, &rng);
      rOut.M = oldM + Mq;
    }
  }

  // GRIS — fold the CANONICAL sample into rOut with its defensive pairwise-MIS
  // weight (Lin 2022 §pairwise MIS / Bitterli 2020): the canonical technique is
  // one resampling technique among the K neighbours, so its sample carries
  //   m_canon = (1/(K+1)) · (1 + Σ_i pairwise-canonical-term_i)
  // (the leading 1 is the defensive self-term; the accumulated sum is the
  // canonical's balance against each accepted neighbour). With NO accepted
  // neighbours m_canon = 1 (the canonical sample stands alone, unchanged).
  if (grisOn && rCenter.M > 0u) {
    let denomCount = f32(acceptedNeighbors + 1u);
    let m_canon = (1.0 + canonMisAccum) / denomCount;
    // Canonical resampling weight: w_canon = m_canon · p̂_r(z_r) · W_r (no shift,
    // J = 1; the sample already lives at this pixel so no visibility re-test).
    let w_canon = m_canon * pHatCanonNative * rCenter.W;
    let oldM = rOut.M;
    updateReservoirGI(&rOut, rCenter.xs, rCenter.ns, rCenter.Lo, w_canon, &rng);
    rOut.M = oldM + rCenter.M;
  }

  // Finalise W from the chosen sample's p̂ at this pixel.
  //   - Legacy reuse: W = w_sum / (M · p̂) — the standard RIS estimator divides
  //     by M because each candidate carried an unweighted 1/M average.
  //   - GRIS reuse: W = w_sum / p̂ — the per-sample MIS weights m_i ALREADY sum
  //     to 1 (they replace the 1/M averaging), so dividing by M again would
  //     under-energise the estimate. This is the GRIS generalized-RIS contribution
  //     weight (Lin 2022 §generalized RIS: W = w_sum / p̂(z) with Σ m_i = 1).
  if (rOut.M > 0u) {
    let toSf = rOut.xs - rOut.xv;
    let distSf = length(toSf);
    if (distSf > 1e-4) {
      let wiF = toSf / distSf;
      let cosThetaF = max(0.0, dot(rOut.nv, wiF));
      let pHatF = luminance(rOut.Lo) * cosThetaF * INV_PI;
      let misNorm = select(f32(rOut.M), 1.0, grisOn);
      let W_raw = select(0.0, rOut.w_sum / (misNorm * pHatF), pHatF > 1e-9);
      rOut.W = min(W_raw, ubo.restirGiWCap);
    } else {
      rOut.W = 0.0;
    }
  }

  // GRIS — refresh the Phase-0 reconnection-shift cache on the chosen sample so
  // the NEXT reuse pass (the ping-pong second spatial dispatch, or the next
  // frame's temporal reuse) sees a base edge rooted at THIS pixel's visible
  // vertex. updateReservoirGI re-roots xs/ns/Lo but not the cache; when a
  // neighbour's sample was chosen the stale cache would otherwise describe the
  // neighbour's edge. Recompute it here from rOut.xv → rOut.xs. Inert when the
  // GRIS gate is off (the legacy path never reads the cache).
  if (grisOn && rOut.M > 0u) {
    let toRecon = rOut.xs - rOut.xv;
    let dRecon = length(toRecon);
    if (dRecon > 1e-6) {
      let wiR = toRecon / dRecon;
      rOut.wi_recon    = wiR;
      rOut.distRecon   = dRecon;
      rOut.cosReconOut = abs(dot(rOut.ns, -wiR));
      rOut.pdfReconBsdf = max(0.0, dot(rOut.nv, wiR)) * INV_PI;
      rOut.prefixVertexCount = 1u;
    }
  }

  storeReservoirGI_rw(&sgi_resOut, pixelIdx, rOut);
}
`;

/** W1-R6 — declarative include-graph entry.
 *  Modules this pass references:
 *    - `WalkaroundUBO` / `INV_PI`            → walkaroundUbo
 *    - `loadReservoirGI_*` / `storeReservoirGI_rw` / `updateReservoirGI`
 *                                            → reservoirGi
 *    - `rand_f32` / `pcgInit` / `luminance`  → sharedPrimitives
 *                                              (which needs PI/INV_PI from
 *                                              walkaroundUbo)
 *    - `jacobianReconnectionShift`           → jacobianShift (legacy reuse)
 *    - `BVHNode` / `traceSceneAny`           → sceneTraversal (GRIS
 *                                              reconnection-visibility ray)
 *    - `grisReconnectionGeometryTerm` / `grisShiftJacobian` / `grisTargetAt` /
 *      `grisPairwiseDenom*`                  → grisReuse (GRIS Phase 1+2)
 *  The GRIS path (BVH traversal + grisReuse) is gated at runtime behind
 *  `ubo.restirPtReuse == 1`; the legacy path stays bit-identical. */
export const SPATIAL_GI_MODULE: WgslModule = {
  name: 'spatialGi',
  source: SPATIAL_GI_WGSL,
  requires: ['walkaroundUbo', 'sceneTraversal', 'reservoirGi', 'sharedPrimitives', 'jacobianShift', 'grisReuse'],
};
