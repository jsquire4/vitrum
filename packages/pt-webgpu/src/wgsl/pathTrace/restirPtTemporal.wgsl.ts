/**
 * restirPtTemporal.wgsl.ts — the ReSTIR-PT TEMPORAL reuse pass (GRIS, Lin 2022).
 *
 * A SEPARATE `@compute` entry point (`restirPtTemporal`) — a DIRECT PORT of the
 * SHIPPING walkaround-hybrid `TEMPORAL_GI_WGSL`
 * (`@vitrum/walkaround-hybrid/src/shaders/temporalGi.wgsl.ts`), generalized to
 * pt-webgpu's full-res hero reservoir.
 *
 * Per full-res pixel it:
 *   1. reprojects the current visible vertex xv through the PREVIOUS-frame camera
 *      (params.prevViewProj) to find the same world point in the previous
 *      reservoir,
 *   2. validates motion-stable primitive identity and finite unit normals —
 *      material swaps and disocclusions are rejected,
 *   3. M-clamps the previous history (rptParams.mClamp),
 *   4. forms the TWO-DOMAIN (current/previous) pairwise generalized-balance MIS,
 *   5. shifts the previous reservoir's reconnection sample onto THIS pixel's
 *      visible vertex via the reconnection shift + its Jacobian, gated by a
 *      reconnection-visibility ray (traceAny along xv → xs),
 *   6. folds both samples into a fresh log-domain GRIS reservoir and finalises
 *      log(W) = log(weight_sum)-log(p̂) (NO /M).
 *
 * ════════════════════════════════════════════════════════════════════════════
 * THE LOAD-BEARING LESSON (mirrored from temporalGi.wgsl.ts:358-367)
 * ════════════════════════════════════════════════════════════════════════════
 * The reused (previous) reservoir's resampling weight is
 *     w_prev = m_prev · p̂_cur(T z_prev) · W_prev · J          (J = shift Jacobian)
 * with NO division by a source pdf. rPrev is a RESERVOIR: its W_prev already
 * carries the complete prior GRIS normalization (W = w_sum/p̂), which may combine
 * multiple earlier source reservoirs. Reservoir reuse
 * re-weights by m·p̂·W·J only — the Jacobian alone carries the reconnection-edge
 * measure conversion. This is the temporal FEEDBACK loop (this pass's output
 * becomes next frame's rPrev). An extra /p_src would multiply the carried weight
 * by ≈π…∞ every frame, driving the recursion gain above 1 and making the GI
 * mean climb without bound (the V19 grison-divergence). REPLICATED HERE
 * EXACTLY — see the w_prev line; there is NO /p_src.
 *
 * For prefix length 1 the source/target pre-reconnection vertices ARE the visible
 * vertices (xPre == xv), so the complete change of variables is the half-G
 * geometry ratio. Source proposal information is already represented by W_prev;
 * pdfSrc is carried sample metadata, not an additional reuse denominator.
 *
 * ── Bind groups ─────────────────────────────────────────────────────────────
 * Composes the SHARED pt-webgpu modules (for traceAny / projectToNdc); the
 * ReSTIR-PT reservoirs + params live in @group(4):
 *   @binding(1) rpt_reservoirCur  (read_write) — this frame's producer output, fused in place
 *   @binding(2) rpt_reservoirPrev (read)        — last frame's temporal output
 *   @binding(4) rptParams         (uniform)
 * The reconnection-visibility ray traverses the inherited @group(0..2) scene
 * (traceAny). (maxBindGroups ≥ 5 — see the compose-module note.)
 */

export const RESTIR_PT_TEMPORAL_WGSL = /* wgsl */ `
@group(4) @binding(1) var<storage, read_write> rpt_resCurrent: array<u32>;
@group(4) @binding(2) var<storage, read>       rpt_resPrev:    array<u32>;
@group(4) @binding(4) var<uniform>             rptParams:      RestirPtParams;

const RPT_TEMPORAL_IDENTITY_RADIUS: i32 = 2;

fn rptTemporalNormalIsValid(n: vec3f) -> bool {
  let len2 = dot(n, n);
  return all(n == n)
      && all(abs(n) <= vec3f(1e6))
      && len2 >= 0.5
      && len2 <= 1.5;
}

fn rptTemporalSurfaceDistance(
  current: ReservoirPTHero,
  previous: ReservoirPTHero,
) -> f32 {
  let delta = current.surfaceParamV - previous.surfaceParamV;
  if (current.triangleIndexV < params.triangleCount) {
    return rptScaledLength(vec3f(delta.xy, 0.0));
  }
  // Analytic surface parameters are already normalized into [-1,1]^3.
  return rptScaledLength(delta);
}

fn rptTemporalSurfaceIdentityMatches(
  current: ReservoirPTHero,
  previous: ReservoirPTHero,
) -> bool {
  if (current.materialIdV != previous.materialIdV
   || current.instanceIndexV != previous.instanceIndexV
   || current.triangleIndexV != previous.triangleIndexV) {
    return false;
  }
  let surfaceDistance = rptTemporalSurfaceDistance(current, previous);
  if (current.triangleIndexV < params.triangleCount) {
    // Nearest-pixel reprojection can move slightly within a triangle.
    return surfaceDistance <= 0.08;
  }
  // Analytic shapes carry local-space hit positions, so rigid object motion
  // remains a valid correspondence while a disoccluded surface is rejected.
  return surfaceDistance <= 0.03;
}

// Reproject a world point through the PREVIOUS-frame camera to its full-res
// pixel. Mirrors the GI projectToPrevHalfPx, but full-res and via the VP matrix
// pt-webgpu carries (params.prevViewProj) rather than separate view/proj. Returns
// the integer pixel coords, or a negative sentinel when behind the prev camera.
fn rptProjectToPrevPx(worldPos: vec3f) -> vec2i {
  let clip = params.prevViewProj * vec4f(worldPos, 1.0);
  if (!all(clip == clip) || any(abs(clip) > vec4f(RPT_MAX_FINITE_F32))
   || clip.w <= 0.0) {
    return vec2i(-1, -1);
  }
  let ndc = clip.xy / clip.w;
  if (!all(ndc == ndc) || any(abs(ndc) > vec2f(RPT_MAX_FINITE_F32))) {
    return vec2i(-1, -1);
  }
  let uv = vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
  if (uv.x < 0.0 || uv.x >= 1.0 || uv.y < 0.0 || uv.y >= 1.0) { return vec2i(-1, -1); }
  let px = i32(uv.x * f32(params.width));
  let py = i32(uv.y * f32(params.height));
  return vec2i(px, py);
}

// GRIS reconnection visibility — clear edge xv → xs through the scene BVH/TLAS
// (traceAny over the inherited @group(0..2)). Mirrors GI tgiReconnectionVisible.
fn rptReconnectionVisible(
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
fn restirPtTemporal(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= params.width || gid.y >= params.height) {
    return;
  }
  let pixelIdx = gid.y * params.width + gid.x;
  var rCur = loadReservoirPTHero_rw(&rpt_resCurrent, pixelIdx);

  // Need a visible-surface point to reproject. If the producer wrote an empty
  // reservoir (primary miss / specular / transmissive xv), nothing to fuse.
  if (rCur.M == 0u) {
    storeReservoirPTHero_rw(&rpt_resCurrent, pixelIdx, rCur);
    return;
  }

  if (!rptTemporalNormalIsValid(rCur.nv)) {
    storeReservoirPTHero_rw(&rpt_resCurrent, pixelIdx, rCur);
    return;
  }

  // Camera reprojection supplies the fast-path location. For moving geometry,
  // the same primitive-local point need not occupy that exact previous pixel,
  // so a bounded identity search recovers nearby rigid/skinned motion.
  let prevPx = rptProjectToPrevPx(rCur.xv);
  if (prevPx.x < 0 || prevPx.y < 0
   || u32(prevPx.x) >= params.width || u32(prevPx.y) >= params.height) {
    storeReservoirPTHero_rw(&rpt_resCurrent, pixelIdx, rCur);
    return;
  }
  let prevIdx = u32(prevPx.y) * params.width + u32(prevPx.x);
  var rPrev = loadReservoirPTHero_ro(&rpt_resPrev, prevIdx);
  var prevFound = rPrev.M > 0u
               && rptFiniteScalar(rPrev.logW)
               && rPrev.logW != RPT_LOG_ZERO
               && rPrev.logW != RPT_LOG_NUMERIC_FAILURE
               && rptTemporalNormalIsValid(rPrev.nv)
               && rptTemporalSurfaceIdentityMatches(rCur, rPrev);
  var prevAmbiguous = false;
  if (!prevFound) {
    var bestSurfaceDistance = 1e30;
    for (var oy = -RPT_TEMPORAL_IDENTITY_RADIUS;
         oy <= RPT_TEMPORAL_IDENTITY_RADIUS;
         oy = oy + 1) {
      for (var ox = -RPT_TEMPORAL_IDENTITY_RADIUS;
           ox <= RPT_TEMPORAL_IDENTITY_RADIUS;
           ox = ox + 1) {
        if (ox == 0 && oy == 0) { continue; }
        let candidatePx = prevPx + vec2i(ox, oy);
        if (candidatePx.x < 0 || candidatePx.y < 0
         || u32(candidatePx.x) >= params.width
         || u32(candidatePx.y) >= params.height) {
          continue;
        }
        let candidateIdx = u32(candidatePx.y) * params.width + u32(candidatePx.x);
        let candidate = loadReservoirPTHero_ro(&rpt_resPrev, candidateIdx);
        if (candidate.M == 0u
         || !rptFiniteScalar(candidate.logW)
         || candidate.logW == RPT_LOG_ZERO
         || candidate.logW == RPT_LOG_NUMERIC_FAILURE
         || !rptTemporalNormalIsValid(candidate.nv)
         || !rptTemporalSurfaceIdentityMatches(rCur, candidate)) {
          continue;
        }
        let surfaceDistance = rptTemporalSurfaceDistance(rCur, candidate);
        if (!prevFound || surfaceDistance < bestSurfaceDistance) {
          rPrev = candidate;
          bestSurfaceDistance = surfaceDistance;
          prevFound = true;
          prevAmbiguous = false;
        } else if (surfaceDistance == bestSurfaceDistance) {
          // Duplicate primitive-local identities at the same search score are
          // not distinguishable without previous geometry. Reject history
          // instead of selecting stale reuse by scan order.
          prevAmbiguous = true;
        }
      }
    }
  }
  if (!prevFound || prevAmbiguous) {
    storeReservoirPTHero_rw(&rpt_resCurrent, pixelIdx, rCur);
    return;
  }

  // M-clamp: bound prev history before contributing.
  let prevM = min(rPrev.M, rptParams.mClamp);

  var rng = pcgInit(
    gid.x ^ (params.frameSeed * 0x71E5u),
    gid.y ^ (params.frameSeed * 0xE571u),
    ptRngFrameKey(params.frameSeed ^ 0x9B7Fu, params.frameIndex),
  );

  // ── GRIS reconnection-shift temporal reuse (Lin 2022 §5 + Eq. 12 + MIS) ──
  // Two-domain pairwise MIS: the CURRENT (canonical) sample and the reprojected
  // PREVIOUS sample, both folded into a FRESH reservoir with their own pairwise
  // weights (the canonical is a resampling technique, not an un-weighted base).
  let cCur = f32(rCur.M);
  let cPrev = f32(prevM);
  // Each domain carries its native producer eye direction. This remains exact
  // under camera motion; reconstructing both directions from the current camera
  // would evaluate the previous target in the wrong domain.
  let woCur  = rCur.woV;
  let woPrev = rPrev.woV;
  let pHatCur_native = restirPtTargetForDomainAtHero(
    rCur, rCur.heroLambdaV, woCur, rCur.xs, rCur.Lo,
  );

  // Decide whether prev is a VALID one-edge reconnection-shift candidate:
  // positive half-G Jacobian, non-zero shifted+native targets, and a visible
  // reconnection edge. Nonempty reservoirs already satisfy the sole supported
  // path topology; invalid selected edges are emptied before storage.
  let J = restirPtReconnectionJacobianForPair(rPrev, rCur);
  let pHatPrev_atCur = restirPtTargetForDomainAtHero(
    rCur, rPrev.heroLambdaV, woCur, rPrev.xs, rPrev.Lo,
  );
  let pHatPrev_native = restirPtTargetForDomainAtHero(
    rPrev, rPrev.heroLambdaV, woPrev, rPrev.xs, rPrev.Lo,
  );
  let prevValid = rptFinitePositive(J)
               && rptFinitePositive(pHatPrev_atCur)
               && rptFinitePositive(pHatPrev_native)
               && rptReconnectionVisible(rCur.xv, rCur.nv, rPrev.xs, &rng);

  // The canonical sample has a separate inverse-support question: could the
  // previous technique have produced this current sample?  Evaluate that
  // current→previous map independently.  Its determinant converts the previous
  // target into the current sample's native measure for generalized-balance MIS.
  let JCurrentToPrevious = restirPtReconnectionJacobianForPair(rCur, rPrev);
  let pHatPrev_atCurSample = restirPtTargetForDomainAtHero(
    rPrev, rCur.heroLambdaV, woPrev, rCur.xs, rCur.Lo,
  );
  let previousCoversCurrent =
       rptFinitePositive(JCurrentToPrevious)
    && rptFinitePositive(pHatPrev_atCurSample)
    && rptReconnectionVisible(rPrev.xv, rPrev.nv, rCur.xs, &rng);

  var rGris = emptyReservoirPTHero();
  copyReservoirPTVisibleDomain(&rGris, rCur);
  let representedM = rptSaturatingAddU32(rCur.M, prevM);

  // Canonical (current) sample, MIS-weighted against the prev pair.
  if (rCur.M > 0u && rptFinitePositive(pHatCur_native)) {
    var logMCurrent: f32 = 0.0;
    if (previousCoversCurrent) {
      // The previous proxy is evaluated at the inverse-shifted current sample.
      // J_current→previous converts it into the current sample's measure.
      let logDenomCur = restirPtPairwiseLogDenomCanonical(
        cCur, pHatCur_native,
        cPrev, pHatPrev_atCurSample, JCurrentToPrevious,
      );
      logMCurrent =
        rptLogWeightedTarget(cCur, pHatCur_native) - logDenomCur;
    }
    // Canonical: no shift (J = 1), already at this pixel (no visibility re-test).
    let logWeightCurrent =
      logMCurrent + log(pHatCur_native) + rCur.logW;
    let acceptedCurrent = updateReservoirPTLog(
      &rGris, rCur.xs, rCur.ns, rCur.Lo, rCur.heroLambdaV,
      rCur.pdfSrc, logWeightCurrent, &rng,
    );
  }

  // Previous (reprojected) sample, reconnection-shifted + MIS-weighted.
  if (prevValid) {
    let logDenomPrev = restirPtPairwiseLogDenomNeighbor(
      cCur, pHatPrev_atCur, J,
      cPrev, pHatPrev_native,
    );
    let logMPrevious =
      rptLogWeightedTarget(cPrev, pHatPrev_native) - logDenomPrev;
    // GRIS resampling weight for a REUSED reservoir sample (Lin 2022, Alg. 3 /
    // Eq. 9):  w_prev = m_prev · p̂_cur(T z_prev) · W_prev · |∂T/∂·|.
    // NO /p_src — rPrev is a reservoir; W_prev already bakes its source pdf in.
    // (See the load-bearing lesson in the file header; an extra /p_src diverges
    // the temporal feedback loop — V19 grison.)
    let logWeightPrevious =
      logMPrevious + log(pHatPrev_atCur) + rPrev.logW + log(J);
    // Carry the selected producer density as sample metadata. The reuse weight
    // does not divide by it again because rPrev.W already contains 1 / p_src.
    let acceptedPrevious = updateReservoirPTLog(
      &rGris, rPrev.xs, rPrev.ns, rPrev.Lo, rPrev.heroLambdaV,
      rPrev.pdfSrc, logWeightPrevious, &rng,
    );
  }

  // M is confidence/represented-attempt metadata, not the number of positive
  // WRS candidates.  Include zero-weight techniques and assign it only after
  // reservoir selection so a saturated history cannot suppress later folds.
  if (rGris.M > 0u) {
    rGris.M = representedM;
  }

  // GRIS finalise in log space (MIS weights already sum to 1 — no /M).
  finaliseReservoirPTWGris(&rGris);
  // Validate the selected reconnection edge before downstream spatial reuse and
  // next frame's temporal step.
  refreshReconnectionStatePT(&rGris);

  storeReservoirPTHero_rw(&rpt_resCurrent, pixelIdx, rGris);
}
`;
