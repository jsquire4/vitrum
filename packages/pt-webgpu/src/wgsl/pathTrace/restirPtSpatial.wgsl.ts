/**
 * restirPtSpatial.wgsl.ts — the ReSTIR-PT SPATIAL reuse pass (GRIS, Lin 2022).
 *
 * A SEPARATE `@compute` entry point (`restirPtSpatial`) — a DIRECT PORT of the
 * SHIPPING walkaround-hybrid `SPATIAL_GI_GRIS_WGSL`
 * (`@vitrum/walkaround-hybrid/src/shaders/spatialGi.wgsl.ts`, the GRIS variant),
 * generalized to pt-webgpu's full-res hero reservoir (arbitrary visible-vertex
 * material; the hero target uses the real visible-vertex BRDF, not the GI cosine
 * proxy). It runs AFTER the temporal pass and BEFORE resolve.
 *
 * Per full-res pixel it pulls K_RPT_SPATIAL random neighbours from a disc of
 * radius RPT_SPATIAL_RADIUS px and folds the accepted ones into a FRESH GRIS
 * reservoir with the EXACT generalized balance heuristic (Σ m_i = 1):
 *
 *   1. Geometric-consistency reject (normal alignment + coplanarity to centre).
 *   2. Prefix-match + non-degenerate base half-G + positive shift Jacobian.
 *   3. Reconnection VISIBILITY ray (xv → q.xs through the scene) — required for
 *      unbiasedness; an occluded shifted edge maps to zero contribution.
 *   4. Pass-1 GATHER accepted neighbours into a fixed array (≤ K), then Pass-2
 *      FOLD each sample with its full-GBH weight
 *        m_i(z) = c_i·p̂_i(z) / Σ_j c_j·p̂_j(T_{·→j} z)
 *      where the per-domain target is restirPtTargetAt(xv_j, nv_j, wo_j, mat_j,
 *      z.xs, z.Lo) (the hero target re-rooted onto domain j's visible vertex).
 *   5. Finalise W = w_sum/p̂ (GRIS, NO /M — the m_i already sum to 1), then refresh
 *      the reconnection-shift cache from the chosen edge.
 *
 * ════════════════════════════════════════════════════════════════════════════
 * THE LOAD-BEARING LESSON (mirrored from temporalGi/spatialGi GRIS)
 * ════════════════════════════════════════════════════════════════════════════
 * The reused (neighbour) reservoir's resampling weight is
 *     w_q = m_q · p̂_r(T z_q) · W_q · J          (J = shift Jacobian)
 * with NO division by a source pdf. rQ is a RESERVOIR: its W_q already bakes in
 * the source pdf. An extra /p_src would over-energise the reservoir → divergence
 * in the feedback loop (the V19 grison-divergence). REPLICATED HERE EXACTLY.
 *
 * The per-domain eye direction wo for the target function: the scene is static
 * and all reservoirs are this-frame full-res, so the eye direction at each
 * domain's visible vertex is normalize(cameraPos − xv) (p̂ only sets resampling
 * variance, not the mean, so this is an unbiased choice for the resampling
 * heuristic — the resolve reconstructs with the real BRDF + the real wo).
 *
 * ── Bind groups (relocated to @group(0) high bindings by the compose step) ───
 *   @binding(1) rpt_resSpatialIn  (read)        — the temporal pass output (read-
 *                                                  only neighbour source: this pass
 *                                                  never writes the slot it samples)
 *   @binding(5) rpt_resSpatialOut (read_write)  — this pass's fresh-GRIS output
 *                                                  (resolve reads this slot)
 *   @binding(4) rptParams         (uniform)
 * The reconnection-visibility ray traverses the inherited @group(0..2) scene.
 */

export const RESTIR_PT_SPATIAL_WGSL = /* wgsl */ `
@group(4) @binding(1) var<storage, read>       rpt_resSpatialIn:  array<u32>;
@group(4) @binding(5) var<storage, read_write> rpt_resSpatialOut: array<u32>;
@group(4) @binding(4) var<uniform>             rptParams:         RestirPtParams;

const K_RPT_SPATIAL: u32 = 5u;            // neighbours sampled per pixel
const RPT_SPATIAL_RADIUS: f32 = 16.0;     // disc radius (px) — full-res
const RPT_SPATIAL_NORMAL_DOT_MIN: f32 = 0.906; // cos(25°) normal reject
const RPT_SPATIAL_COPLANAR_TOL: f32 = 0.05;    // plane-distance reject (world)
const RPT_SPATIAL_M_CLAMP: u32 = 500u;    // per-neighbour confidence clamp
const RPT_SPATIAL_NORMAL_BIAS: f32 = 1e-3;

// Uniform disc sample in pixels (radius RPT_SPATIAL_RADIUS).
fn rptSpatialDiscPx(rng: ptr<function, u32>) -> vec2f {
  let r = RPT_SPATIAL_RADIUS * sqrt(rand_f32(rng));
  let theta = 6.2831853 * rand_f32(rng);
  return vec2f(r * cos(theta), r * sin(theta));
}

// Reconnection-visibility for the shifted edge xv → xs (traceAny, group 0..2).
fn rptSpatialReconVisible(xv: vec3f, nv: vec3f, xs: vec3f) -> bool {
  let toS = xs - xv;
  let dist = length(toS);
  if (dist < 1e-4) { return false; }
  let wi = toS / dist;
  let orig = xv + nv * RPT_SPATIAL_NORMAL_BIAS;
  return !traceAny(Ray(orig, wi), 1e-4, max(dist - 2e-3, 1e-3));
}

@compute @workgroup_size(8, 8, 1)
fn restirPtSpatial(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= params.width || gid.y >= params.height) {
    return;
  }
  let pixelIdx = gid.y * params.width + gid.x;
  var rCenter = loadReservoirPTHero_ro(&rpt_resSpatialIn, pixelIdx);

  // No reusable surface here — copy through.
  if (rCenter.M == 0u) {
    storeReservoirPTHero_rw(&rpt_resSpatialOut, pixelIdx, rCenter);
    return;
  }

  var rng = pcgInit(
    gid.x ^ (params.frameSeed * 0xA127u),
    gid.y ^ (params.frameSeed * 0x271Au),
    params.frameSeed ^ 0xBCD3u,
  );

  let woCenter = restirpt_safe_normalize(params.cameraPos.xyz - rCenter.xv);
  let pHatCanonNative = restirPtTargetAt(
    rCenter.xv, rCenter.nv, woCenter,
    rCenter.albV, rCenter.roughnessV, rCenter.metalV,
    rCenter.xs, rCenter.Lo);
  let cR = f32(rCenter.M);

  // ── Pass-1 GATHER accepted neighbours (full-GBH needs the domain set up front)
  var nQ: u32 = 0u;
  var qXv:     array<vec3f, 5>;
  var qNv:     array<vec3f, 5>;
  var qWo:     array<vec3f, 5>;
  var qAlb:    array<vec3f, 5>;
  var qRough:  array<f32, 5>;
  var qMetal:  array<f32, 5>;
  var qXs:     array<vec3f, 5>;
  var qNs:     array<vec3f, 5>;
  var qLo:     array<vec3f, 5>;
  var qC:      array<f32, 5>;
  var qW:      array<f32, 5>;
  var qPdfSrc: array<f32, 5>;  // the REAL source BSDF pdf stored in each neighbour's reservoir
  var qJ:      array<f32, 5>;

  for (var i: u32 = 0u; i < K_RPT_SPATIAL; i = i + 1u) {
    let off = rptSpatialDiscPx(&rng);
    let qx = i32(gid.x) + i32(round(off.x));
    let qy = i32(gid.y) + i32(round(off.y));
    if (qx < 0 || qy < 0
     || u32(qx) >= params.width || u32(qy) >= params.height) { continue; }
    if (qx == i32(gid.x) && qy == i32(gid.y)) { continue; }

    let qIdx = u32(qy) * params.width + u32(qx);
    let rQ = loadReservoirPTHero_ro(&rpt_resSpatialIn, qIdx);
    if (rQ.M == 0u || rQ.W <= 0.0) { continue; }

    // Geometric-consistency: normal alignment + coplanarity to centre pixel.
    if (dot(rCenter.nv, rQ.nv) < RPT_SPATIAL_NORMAL_DOT_MIN) { continue; }
    let planeDist = abs(dot(rQ.xv - rCenter.xv, rCenter.nv));
    if (planeDist > RPT_SPATIAL_COPLANAR_TOL) { continue; }

    // Prefix-match + non-degenerate base half-G.
    if (rQ.prefixVertexCount != rCenter.prefixVertexCount
     || rQ.prefixVertexCount == 0u) { continue; }

    // Shift Jacobian |∂T/∂·| = G(rCenter.xv ↔ q.xs) / G(rQ.xv ↔ q.xs).
    let J = restirPtShiftJacobian(rQ.xv, rCenter.xv, rQ.xs, rQ.ns);
    if (J <= 0.0) { continue; }

    let woQ = restirpt_safe_normalize(params.cameraPos.xyz - rQ.xv);
    // Non-degenerate shifted + native targets, else q contributes nothing.
    let pHatQ_atR = restirPtTargetAt(
      rCenter.xv, rCenter.nv, woCenter,
      rCenter.albV, rCenter.roughnessV, rCenter.metalV,
      rQ.xs, rQ.Lo);
    if (pHatQ_atR < 1e-9) { continue; }
    let pHatQ_native = restirPtTargetAt(
      rQ.xv, rQ.nv, woQ, rQ.albV, rQ.roughnessV, rQ.metalV, rQ.xs, rQ.Lo);
    if (pHatQ_native < 1e-9) { continue; }

    // Reconnection VISIBILITY — required for unbiasedness.
    if (!rptSpatialReconVisible(rCenter.xv, rCenter.nv, rQ.xs)) { continue; }

    let Mq = min(rQ.M, RPT_SPATIAL_M_CLAMP);
    qXv[nQ] = rQ.xv; qNv[nQ] = rQ.nv; qWo[nQ] = woQ;
    qAlb[nQ] = rQ.albV; qRough[nQ] = rQ.roughnessV; qMetal[nQ] = rQ.metalV;
    qXs[nQ] = rQ.xs; qNs[nQ] = rQ.ns; qLo[nQ] = rQ.Lo;
    qC[nQ] = f32(Mq); qW[nQ] = rQ.W; qPdfSrc[nQ] = rQ.pdfSrc; qJ[nQ] = J;
    nQ = nQ + 1u;
  }

  var rOut = emptyReservoirPTHero();
  rOut.xv = rCenter.xv; rOut.nv = rCenter.nv;
  rOut.albV = rCenter.albV; rOut.roughnessV = rCenter.roughnessV; rOut.metalV = rCenter.metalV;
  rOut.prefixVertexCount = rCenter.prefixVertexCount;

  // ── Pass-2 FOLD: canonical sample with its full-GBH weight ──
  if (rCenter.M > 0u && pHatCanonNative > 1e-9) {
    var denomR = cR * pHatCanonNative; // canonical's own native term
    for (var j: u32 = 0u; j < nQ; j = j + 1u) {
      denomR += qC[j] * restirPtTargetAt(
        qXv[j], qNv[j], qWo[j], qAlb[j], qRough[j], qMetal[j],
        rCenter.xs, rCenter.Lo);
    }
    let m_canon = select(0.0, (cR * pHatCanonNative) / denomR, denomR > 1e-12);
    // Canonical: no shift (already at this pixel; J = 1).
    let w_canon = m_canon * pHatCanonNative * rCenter.W;
    let oldM = rOut.M;
    updateReservoirPT(&rOut, rCenter.xs, rCenter.ns, rCenter.Lo, rCenter.pdfSrc, w_canon, &rng);
    rOut.M = oldM + rCenter.M;
  }

  // ── Pass-2 FOLD: each neighbour's sample with its full-GBH weight ──
  for (var i: u32 = 0u; i < nQ; i = i + 1u) {
    let pHatQ_native = restirPtTargetAt(
      qXv[i], qNv[i], qWo[i], qAlb[i], qRough[i], qMetal[i], qXs[i], qLo[i]);
    // GBH denominator: canonical's target for z_q + every neighbour's target.
    var denomQ = cR * restirPtTargetAt(
      rCenter.xv, rCenter.nv, woCenter,
      rCenter.albV, rCenter.roughnessV, rCenter.metalV, qXs[i], qLo[i]);
    for (var j: u32 = 0u; j < nQ; j = j + 1u) {
      denomQ += qC[j] * restirPtTargetAt(
        qXv[j], qNv[j], qWo[j], qAlb[j], qRough[j], qMetal[j], qXs[i], qLo[i]);
    }
    let m_q = select(0.0, (qC[i] * pHatQ_native) / denomQ, denomQ > 1e-12);
    // p̂_r(T z_q): q's sample re-rooted onto the canonical visible vertex.
    let pHatQ_atR = restirPtTargetAt(
      rCenter.xv, rCenter.nv, woCenter,
      rCenter.albV, rCenter.roughnessV, rCenter.metalV, qXs[i], qLo[i]);
    let w_q = m_q * pHatQ_atR * qW[i] * qJ[i];
    let oldM = rOut.M;
    // Pass the neighbour's real source BSDF pdf (pdfSrc) — NOT the unbiased contribution
    // weight W. W already bakes in 1/pdfSrc (W = w_sum/p̂ after finalise); passing W here
    // would store an energy-scaled pdf that corrupts the reconstructed path when this
    // neighbour's sample wins. pdfSrc is the denominator for the resolve unbiased estimator.
    updateReservoirPT(&rOut, qXs[i], qNs[i], qLo[i], qPdfSrc[i], w_q, &rng);
    rOut.M = oldM + u32(qC[i]);
  }

  // GRIS finalise: W = w_sum / p̂ (the MIS weights already sum to 1 — no /M).
  finaliseReservoirPTWGris(&rOut, rptParams.wCap, params.cameraPos.xyz);
  refreshReconnectionCachePT(&rOut);

  storeReservoirPTHero_rw(&rpt_resSpatialOut, pixelIdx, rOut);
}
`;
