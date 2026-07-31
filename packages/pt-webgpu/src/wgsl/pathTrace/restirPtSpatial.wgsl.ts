/**
 * restirPtSpatial.wgsl.ts — the ReSTIR-PT SPATIAL reuse pass (GRIS, Lin 2022).
 *
 * A SEPARATE `@compute` entry point (`restirPtSpatial`) — a DIRECT PORT of the
 * SHIPPING walkaround-hybrid `SPATIAL_GI_WGSL`
 * (`@vitrum/walkaround-hybrid/src/shaders/spatialGi.wgsl.ts`),
 * generalized to pt-webgpu's full-res hero reservoir (arbitrary visible-vertex
 * material; the hero target uses the real visible-vertex BRDF, not the GI cosine
 * proxy). It runs AFTER the temporal pass and BEFORE resolve.
 *
 * Per full-res pixel it pulls K_RPT_SPATIAL random neighbours from a disc of
 * radius RPT_SPATIAL_RADIUS px and folds the accepted ones into a FRESH GRIS
 * reservoir with the EXACT generalized balance heuristic (Σ m_i = 1):
 *
 *   1. Geometric-consistency reject (normal alignment + coplanarity to centre).
 *   2. Prefix-match + non-degenerate half-G reconnection Jacobian.
 *   3. Reconnection VISIBILITY ray (xv → q.xs through the scene) — required for
 *      unbiasedness; an occluded shifted edge maps to zero contribution.
 *   4. Pass-1 GATHER accepted neighbours into a fixed array (≤ K), then Pass-2
 *      FOLD each sample with its full-GBH weight, expressed in candidate i's
 *      native measure:
 *        m_i(z) = c_i·p̂_i(z)
 *               / Σ_j c_j·p̂_j(T_{i→j} z)·|∂T_{i→j}/∂z|
 *      where the per-domain target is restirPtTargetAt(xv_j, nv_j, wo_j, mat_j,
 *      z.xs, z.Lo), and undefined/occluded inverse mappings contribute zero.
 *   5. Finalise log(W) = log(weight_sum)-log(p̂) (GRIS, NO /M), then validate
 *      the selected reconnection edge.
 *
 * ════════════════════════════════════════════════════════════════════════════
 * THE LOAD-BEARING LESSON (mirrored from temporalGi/spatialGi GRIS)
 * ════════════════════════════════════════════════════════════════════════════
 * The reused (neighbour) reservoir's resampling weight is
 *     w_q = m_q · p̂_r(T z_q) · W_q · J          (J = half-G ratio)
 * with NO division by a source pdf. rQ is a RESERVOIR: its W_q already carries
 * the complete prior GRIS normalization, potentially from many source domains.
 * An extra /p_src would over-energise the reservoir → divergence
 * in the feedback loop (the V19 grison-divergence). REPLICATED HERE EXACTLY.
 *
 * Each reservoir carries the native eye direction of its producer domain.
 * Re-evaluating every target with that wo keeps glossy targets correct during
 * camera motion.
 *
 * ── Bind groups (relocated to @group(0) high bindings by the compose step) ───
 *   @binding(1) rpt_resSpatialIn  (read)        — the temporal pass output (read-
 *                                                  only neighbour source: this pass
 *                                                  never writes the slot it samples)
 *   @binding(5) rpt_resSpatialOut (read_write)  — this pass's fresh-GRIS output
 *                                                  (resolve reads this slot)
 * The reconnection-visibility ray traverses the inherited @group(0..2) scene.
 */

export const RESTIR_PT_SPATIAL_WGSL = /* wgsl */ `
@group(4) @binding(1) var<storage, read>       rpt_resSpatialIn:  array<u32>;
@group(4) @binding(5) var<storage, read_write> rpt_resSpatialOut: array<u32>;

const K_RPT_SPATIAL: u32 = 5u;            // neighbours sampled per pixel
const RPT_SPATIAL_RADIUS: f32 = 16.0;     // disc radius (px) — full-res
const RPT_SPATIAL_NORMAL_DOT_MIN: f32 = 0.906; // cos(25°) normal reject
const RPT_SPATIAL_COPLANAR_REL_TOL: f32 = 0.05; // plane-distance / local-path scale
const RPT_SPATIAL_M_CLAMP: u32 = 500u;    // per-neighbour confidence clamp

// Uniform disc sample in pixels (radius RPT_SPATIAL_RADIUS).
fn rptSpatialDiscPx(rng: ptr<function, PtRngState>) -> vec2f {
  let r = RPT_SPATIAL_RADIUS * sqrt(rand_f32(rng));
  let theta = 6.2831853 * rand_f32(rng);
  return vec2f(r * cos(theta), r * sin(theta));
}

// Reconnection-visibility for the shifted edge xv → xs (traceAny, group 0..2).
fn rptSpatialReconVisible(
  xv: vec3f,
  nv: vec3f,
  xs: vec3f,
  rng: ptr<function, PtRngState>,
) -> bool {
  let toS = xs - xv;
  let dist = rptScaledLength(toS);
  if (!rptFinitePositive(dist)) { return false; }
  let startEpsilon = rptWorldRayEpsilon(xv, dist);
  if (!rptFinitePositive(startEpsilon)) { return false; }
  let orig = xv + safe_normalize(nv) * startEpsilon;
  if (!rptFiniteVec3(orig)) { return false; }
  let remaining = xs - orig;
  let remainingDistance = rptScaledLength(remaining);
  if (!rptFinitePositive(remainingDistance)) { return false; }
  let endEpsilon = rptWorldRayEpsilon(xs, remainingDistance);
  let tMax = remainingDistance - endEpsilon;
  if (!rptFinitePositive(endEpsilon) || !(tMax > startEpsilon)) { return false; }
  return !traceAny(
    Ray(orig, safe_normalize(remaining)), startEpsilon, tMax, rng,
  );
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
    ptRngFrameKey(params.frameSeed ^ 0xBCD3u, params.frameIndex),
  );

  let woCenter = rCenter.woV;
  let pHatCanonNative = restirPtTargetForDomainAtHero(
    rCenter, rCenter.heroLambdaV, woCenter, rCenter.xs, rCenter.Lo,
  );
  let cR = f32(rCenter.M);

  // ── Pass-1 GATHER accepted neighbours (full-GBH needs the domain set up front)
  var nQ: u32 = 0u;
  var qR:      array<ReservoirPTHero, 5>;
  var qWo:     array<vec3f, 5>;
  var qC:      array<f32, 5>;
  var qLogW:   array<f32, 5>;
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
    if (rQ.M == 0u
     || !rptFiniteScalar(rQ.logW)
     || rQ.logW == RPT_LOG_ZERO
     || rQ.logW == RPT_LOG_NUMERIC_FAILURE) { continue; }

    // Geometric-consistency: normal alignment + coplanarity to centre pixel.
    if (dot(rCenter.nv, rQ.nv) < RPT_SPATIAL_NORMAL_DOT_MIN) { continue; }
    let surfaceDelta = rQ.xv - rCenter.xv;
    let planeDist = rptScaledAbsProjection(surfaceDelta, rCenter.nv);
    let localPathScale = max(
      rptScaledLength(surfaceDelta),
      max(
        rptScaledLength(rCenter.xs - rCenter.xv),
        rptScaledLength(rQ.xs - rQ.xv),
      ),
    );
    if (!rptFinitePositive(localPathScale)
     || planeDist > localPathScale * RPT_SPATIAL_COPLANAR_REL_TOL) {
      continue;
    }

    let woQ = rQ.woV;
    // Cache this source sample's one-edge q→canonical determinant.  Do not
    // reject the whole technique when its selected sample is not shiftable:
    // its represented attempts still belong in other candidates' MIS matrices.
    let J = restirPtReconnectionJacobianForPair(rQ, rCenter);

    let Mq = min(rQ.M, RPT_SPATIAL_M_CLAMP);
    qR[nQ] = rQ; qWo[nQ] = woQ;
    qC[nQ] = f32(Mq); qLogW[nQ] = rQ.logW; qJ[nQ] = J;
    nQ = nQ + 1u;
  }

  var rOut = emptyReservoirPTHero();
  copyReservoirPTVisibleDomain(&rOut, rCenter);
  var representedM = rCenter.M;
  for (var i: u32 = 0u; i < nQ; i = i + 1u) {
    representedM = rptSaturatingAddU32(representedM, u32(qC[i]));
  }

  // ── Pass-2 FOLD: canonical sample with its full-GBH weight ──
  if (rCenter.M > 0u && rptFinitePositive(pHatCanonNative)) {
    var logDenomR = rptLogWeightedTarget(
      cR, pHatCanonNative,
    ); // canonical's own native term
    for (var j: u32 = 0u; j < nQ; j = j + 1u) {
      let pHatQ_atCanonicalSample = restirPtTargetForDomainAtHero(
        qR[j], rCenter.heroLambdaV, qWo[j], rCenter.xs, rCenter.Lo,
      );
      let JCanonicalToQ = restirPtReconnectionJacobianForPair(
        rCenter, qR[j],
      );
      let qCoversCanonical =
           rptFinitePositive(JCanonicalToQ)
        && rptFinitePositive(pHatQ_atCanonicalSample)
        && rptSpatialReconVisible(
          qR[j].xv, qR[j].nv, rCenter.xs, &rng,
        );
      if (qCoversCanonical) {
        logDenomR = rptLogAddExp(
          logDenomR,
          rptLogWeightedShiftedTarget(
            qC[j], pHatQ_atCanonicalSample, JCanonicalToQ,
          ),
        );
      }
    }
    let logMCanonical =
      rptLogWeightedTarget(cR, pHatCanonNative) - logDenomR;
    // Canonical: no shift (already at this pixel; J = 1).
    let logWeightCanonical =
      logMCanonical + log(pHatCanonNative) + rCenter.logW;
    let acceptedCanonical = updateReservoirPTLog(
      &rOut, rCenter.xs, rCenter.ns, rCenter.Lo, rCenter.heroLambdaV,
      rCenter.pdfSrc, logWeightCanonical, &rng,
    );
  }

  // ── Pass-2 FOLD: each neighbour's sample with its full-GBH weight ──
  for (var i: u32 = 0u; i < nQ; i = i + 1u) {
    let pHatQ_native = restirPtTargetForDomainAtHero(
      qR[i], qR[i].heroLambdaV, qWo[i], qR[i].xs, qR[i].Lo,
    );
    let pHatQ_atR = restirPtTargetForDomainAtHero(
      rCenter, qR[i].heroLambdaV, woCenter, qR[i].xs, qR[i].Lo,
    );
    let qCandidateValid =
         rptFinitePositive(pHatQ_native)
      && rptFinitePositive(pHatQ_atR)
      && rptFinitePositive(qJ[i])
      && rptSpatialReconVisible(
        rCenter.xv, rCenter.nv, qR[i].xs, &rng,
      );
    if (!qCandidateValid) { continue; }

    // GBH denominator in candidate i's native measure.  Every cross-domain
    // target carries |dT_i→j| and contributes only when that inverse mapping is
    // non-degenerate and visible.
    var logDenomQ = rptLogWeightedShiftedTarget(
      cR, pHatQ_atR, qJ[i],
    );
    for (var j: u32 = 0u; j < nQ; j = j + 1u) {
      if (j == i) {
        logDenomQ = rptLogAddExp(
          logDenomQ,
          rptLogWeightedTarget(qC[i], pHatQ_native),
        );
        continue;
      }
      let pHatJ_atQSample = restirPtTargetForDomainAtHero(
        qR[j], qR[i].heroLambdaV, qWo[j], qR[i].xs, qR[i].Lo,
      );
      let JQToJ = restirPtReconnectionJacobianForPair(qR[i], qR[j]);
      let jCoversQSample =
           rptFinitePositive(JQToJ)
        && rptFinitePositive(pHatJ_atQSample)
        && rptSpatialReconVisible(
          qR[j].xv, qR[j].nv, qR[i].xs, &rng,
        );
      if (jCoversQSample) {
        logDenomQ = rptLogAddExp(
          logDenomQ,
          rptLogWeightedShiftedTarget(qC[j], pHatJ_atQSample, JQToJ),
        );
      }
    }
    let logMNeighbor =
      rptLogWeightedTarget(qC[i], pHatQ_native) - logDenomQ;
    let logWeightNeighbor =
      logMNeighbor + log(pHatQ_atR) + qLogW[i] + log(qJ[i]);
    // Preserve the selected neighbour sample's source proposal metadata. The
    // reusable contribution weight is qR[i].W and is already folded into w_q;
    // pdfSrc is not a resolve denominator and must not be replaced by W.
    let acceptedNeighbor = updateReservoirPTLog(
      &rOut, qR[i].xs, qR[i].ns, qR[i].Lo, qR[i].heroLambdaV,
      qR[i].pdfSrc, logWeightNeighbor, &rng,
    );
  }

  // Represent every gathered source attempt, including techniques whose
  // selected sample had zero shifted weight.  Delay assignment until all WRS
  // candidates are folded so a saturated history cannot suppress selection.
  if (rOut.M > 0u) {
    rOut.M = representedM;
  }

  // GRIS finalise in log space (MIS weights already sum to 1 — no /M).
  finaliseReservoirPTWGris(&rOut);
  refreshReconnectionStatePT(&rOut);

  storeReservoirPTHero_rw(&rpt_resSpatialOut, pixelIdx, rOut);
}
`;
