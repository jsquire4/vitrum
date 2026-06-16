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
 * ════════════════════════════════════════════════════════════════════════════
 * COMPILE-TIME GRIS gate (restirPtReuse) — why this file has TWO shader bodies
 * ════════════════════════════════════════════════════════════════════════════
 * GRIS reconnection-shift reuse (Phases 1+2; Lin et al. 2022) is an OPT-IN
 * feature gated by `HybridEngineOptions.restirPtReuse`. The gate is resolved at
 * PIPELINE-COMPILE time (the flag is fixed at engine creation), NOT at runtime,
 * because turning it on STRUCTURALLY changes this pass's reservoir cache stride
 * and shader body:
 *   - the ON shader adds a GRIS combine branch and reconnection-visibility ray;
 *   - the OFF shader keeps standard spatial reuse.
 * A runtime `ubo.restirPtReuse` flag is NOT sufficient: binding a second group
 * or changing the cache layout only on some frames would alter the DEFAULT
 * pipeline structure, which regressed the default walkaround render to an
 * all-black frame (the original f8df9a4 bug). Both variants now bind the shared
 * `@group(1)` scene/material group because receiver-lobe p-hat recasts need it
 * even in the default path:
 *   - {@link SPATIAL_GI_MODULE}      — OFF (default). Standard spatial reuse
 *                                       with receiver-material p-hat recast.
 *   - {@link SPATIAL_GI_GRIS_MODULE} — ON. Adds the GRIS branch + the
 *                                       `sceneTraversal`/`grisReuse` deps.
 * {@link compilePipelines} composes whichever module matches the host flag and
 * builds the matching shader body; {@link SpatialGIReservoirPass} always binds
 * `@group(1)`.
 *
 * Bindings (ping-pong via two distinct bind groups that swap
 * reservoirGiCurrent / reservoirGiSpatial between in and out):
 *   @group(0) @binding(0) input  reservoir (storage, read)
 *   @group(0) @binding(1) output reservoir (storage, read_write)
 *   @group(0) @binding(2) WalkaroundUBO    (uniform)
 *   @group(1)             scene BVH/TLAS/material atlas (read-only) — both
 *                         variants use it for receiver-material p-hat recasts;
 *                         GRIS also traces reconnection visibility through it.
 *
 * GRIS combine (ON variant only): when reused, combining neighbour q's reservoir
 * into pixel r takes the UNBIASED GRIS path instead of the legacy clamped-Jacobian
 * reuse:
 *   1. Reconnection shift T: re-root q's reconnection vertex (q.xs/ns/Lo) onto
 *      the current pixel's primary vertex — a fresh edge rCenter.xv → q.xs.
 *   2. Shift Jacobian |∂T/∂·| = G(rCenter.xv ↔ q.xs)/G_base, where the base
 *      half-G is recovered from q's Phase-0 cache (cosReconOut/distRecon²) —
 *      mirrors @vitrum/shared-samplers reconnectionJacobian (Eq. 12).
 *   3. Reconnection VISIBILITY: trace rCenter.xv → q.xs through the BVH; if
 *      occluded / degenerate / backfacing / prefix-incompatible the shift maps
 *      to zero contribution (reject this neighbour). Required for unbiasedness.
 *   4. Pairwise generalized-balance MIS (§pairwise MIS): m_q weights q's sample
 *      by p̂ evaluated in each domain, the shift Jacobian entering the
 *      resampling weight; the combined reservoir is an unbiased RIS estimator.
 */

import type { WgslModule } from '../pipeline/wgslComposer.js';

// ════════════════════════════════════════════════════════════════════════════
// OFF (default) — Sprint-17 spatial reuse plus receiver-material p-hat recast.
// The GRIS branch stays absent, but @group(1) is bound so rich receivers use the
// same material-aware target as the RIS producer.
// ════════════════════════════════════════════════════════════════════════════
export const SPATIAL_GI_WGSL = /* wgsl */ `

@group(0) @binding(0) var<storage, read>       sgi_resIn:  array<u32>;
@group(0) @binding(1) var<storage, read_write> sgi_resOut: array<u32>;
@group(0) @binding(2) var<uniform> ubo: WalkaroundUBO;

@group(1) @binding(0) var<storage, read> bvh:          array<BVHNode>;
@group(1) @binding(1) var<storage, read> bvh_index:    array<vec4u>;
@group(1) @binding(2) var<storage, read> bvh_position: array<vec4f>;
@group(1) @binding(6) var<storage, read> tlasNodes: array<BVHNode>;
@group(1) @binding(7) var<storage, read> tlasInstanceIndices: array<u32>;
@group(1) @binding(8) var<storage, read> tlasBlasRoots: array<u32>;
@group(1) @binding(9) var<storage, read> tlasInstanceWorldToLocal: array<vec4f>;
@group(1) @binding(10) var<storage, read> tlasInstanceLocalToWorld: array<vec4f>;

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

  let centerFullPx = gid.xy * 2u + 1u;
  let vp = ubo.projMatrix * ubo.viewMatrix;
  let invVP = invertMat4_common(vp);
  let centerSurf = castPrimary(centerFullPx, fullDims, ubo.cameraPos, invVP);

  var rng = pcgInit(
    gid.x ^ (ubo.frameSeed * 0xA127u),
    gid.y ^ (ubo.frameSeed * 0x271Au),
    ubo.frameSeed ^ 0xBCD3u,
  );

  var rOut = rCenter;
  let centerDepth = max(1e-3, length(rCenter.xv - ubo.cameraPos));

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

    // Jacobian shift: rQ's reservoir holds (xs, ns, Lo) seen from rQ.xv;
    // re-weight it for evaluation at rCenter.xv.
    let J = jacobianReconnectionShift(rCenter.xv, rCenter.nv, rQ.xv, rQ.xs, rQ.ns);
    if (J <= 0.0) { continue; }

    // p̂ at center pixel.
    let toS = rQ.xs - rCenter.xv;
    let distS = length(toS);
    if (distS < 1e-4) { continue; }
    let wiZ = toS / distS;
    let pHatZ = restir_gi_receiver_phat_from_surface_or_geometry(
      centerSurf,
      rCenter.xv,
      rCenter.nv,
      rQ.xs,
      rQ.Lo,
    );
    if (pHatZ < 1e-9) { continue; }

    let Mq = min(rQ.M, M_CLAMP_SPATIAL);
    let w_q = pHatZ * rQ.W * f32(Mq) * J;
    let oldM = rOut.M;
    updateReservoirGI(&rOut, rQ.xs, rQ.ns, rQ.Lo, w_q, &rng);
    rOut.M = oldM + Mq;
  }

  // Finalise W from the chosen sample's p̂ at this pixel.
  // D5.3 (gris=false): standard RIS — divide by M (MIS weight 1 per candidate).
  let pHatOut = restir_gi_receiver_phat_from_surface_or_geometry(
    centerSurf,
    rOut.xv,
    rOut.nv,
    rOut.xs,
    rOut.Lo,
  );
  finaliseGIReservoirWFromPHat(&rOut, ubo.restirGiWCap, false, pHatOut);

  storeReservoirGI_rw(&sgi_resOut, pixelIdx, rOut);
}
`;

/** OFF (default) include-graph entry.
 *  T9-stepC — narrowed from `['common']` to the modules this pass uses:
 *    - `WalkaroundUBO` / `INV_PI`            → walkaroundUbo
 *    - `loadReservoirGI_*` / `storeReservoirGI_rw` / `updateReservoirGI`
 *                                            → reservoirGi
 *    - `rand_f32` / `pcgInit` / `luminance`  → sharedPrimitives
 *                                              (which needs PI/INV_PI from
 *                                              walkaroundUbo)
 *    - `jacobianReconnectionShift`           → jacobianShift
 *  No primary-ray cast → no cameraRays / sceneTraversal. No GRIS → no grisReuse.
 *  Verified complete by the static ident-resolution gate. */
export const SPATIAL_GI_MODULE: WgslModule = {
  name: 'spatialGi',
  source: SPATIAL_GI_WGSL,
  requires: ['walkaroundUbo', 'spatialGiCommon', 'reservoirGi', 'sharedPrimitives', 'jacobianShift', 'cameraRays', 'sceneTraversal', 'restirCastPrimary', 'restirGiMaterial'],
};

// ════════════════════════════════════════════════════════════════════════════
// ON (opt-in, restirPtReuse) — GRIS reconnection-shift reuse. Adds the
// @group(1) scene BVH/TLAS bindings + the reconnection-visibility ray + the
// GRIS combine branch + the pairwise-MIS canonical fold. Composed ONLY when
// restirPtReuse is set, with the two-group pipeline layout.
// ════════════════════════════════════════════════════════════════════════════
export const SPATIAL_GI_GRIS_WGSL = /* wgsl */ `

@group(0) @binding(0) var<storage, read>       sgi_resIn:  array<u32>;
@group(0) @binding(1) var<storage, read_write> sgi_resOut: array<u32>;
@group(0) @binding(2) var<uniform> ubo: WalkaroundUBO;

// Scene group (group 1) — BVH + TLAS + material atlas. The default variant
// binds the same group for receiver-material p-hat recasts; the GRIS variant
// additionally uses it for reconnection visibility.
@group(1) @binding(0) var<storage, read> bvh:          array<BVHNode>;
@group(1) @binding(1) var<storage, read> bvh_index:    array<vec4u>;
@group(1) @binding(2) var<storage, read> bvh_position: array<vec4f>;
@group(1) @binding(6) var<storage, read> tlasNodes: array<BVHNode>;
@group(1) @binding(7) var<storage, read> tlasInstanceIndices: array<u32>;
@group(1) @binding(8) var<storage, read> tlasBlasRoots: array<u32>;
@group(1) @binding(9) var<storage, read> tlasInstanceWorldToLocal: array<vec4f>;
@group(1) @binding(10) var<storage, read> tlasInstanceLocalToWorld: array<vec4f>;

// Normal bias for the GRIS reconnection-visibility ray origin (lift off the
// surface so the ray does not self-intersect the visible point's triangle).
const GRIS_NORMAL_BIAS: f32 = 1e-3;

// GRIS reconnection visibility — is the shifted edge xv → xs unoccluded? Trace
// a shadow ray through the scene BVH/TLAS (group 1). Returns true if the
// connection is clear (the suffix radiance Lo at xs is actually visible from
// the current pixel's primary vertex). skipGlass=true matches the risGi /
// ReSTIR shadow-ray convention (light passes through glass; tint handled
// elsewhere).
fn grisReconnectionVisible(xv: vec3f, nv: vec3f, xs: vec3f) -> bool {
  let toS = xs - xv;
  let dist = length(toS);
  if (dist < 1e-4) { return false; }
  let wi = toS / dist;
  let orig = xv + nv * GRIS_NORMAL_BIAS;
  let occ = traceSceneAnyCastMask(
    ubo.bvhMode, ubo.tlasNodeCount,
    &bvh_index, &bvh_position, &bvh,
    &tlasNodes, &tlasInstanceIndices, &tlasBlasRoots,
    &tlasInstanceWorldToLocal, &tlasInstanceLocalToWorld,
    orig, wi, dist - 2e-3, ubo.triIntersectEpsilon, true,
    bvh_material, BVH_MATERIAL_TEX_WIDTH);
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

  let centerFullPx = gid.xy * 2u + 1u;
  let vp = ubo.projMatrix * ubo.viewMatrix;
  let invVP = invertMat4_common(vp);
  let centerSurf = castPrimary(centerFullPx, fullDims, ubo.cameraPos, invVP);

  var rng = pcgInit(
    gid.x ^ (ubo.frameSeed * 0xA127u),
    gid.y ^ (ubo.frameSeed * 0x271Au),
    ubo.frameSeed ^ 0xBCD3u,
  );

  // GRIS reuse builds a FRESH reservoir so the canonical sample can be folded
  // in with its OWN MIS weight (the canonical technique is one of the resampling
  // techniques, not an un-weighted base).
  var rOut = emptyReservoirGI();
  rOut.xv = rCenter.xv; rOut.nv = rCenter.nv;
  rOut.prefixVertexCount = rCenter.prefixVertexCount;

  let pHatCanonNative = restir_gi_receiver_phat_from_surface_or_geometry(
    centerSurf,
    rCenter.xv,
    rCenter.nv,
    rCenter.xs,
    rCenter.Lo,
  );
  let cR = f32(rCenter.M);

  // ── GRIS combine via the EXACT generalized balance heuristic (Lin 2022;
  // mirrors @vitrum/shared-samplers grisGeneralizedBalanceWeights, the unit-
  // pinned Σ m_i = 1 oracle) ───────────────────────────────────────────────
  //
  // The streaming-pairwise approximation used previously does NOT partition
  // unity in general (its Σ m_i drifts well above 1 → over-energised reservoir
  // → divergence in the temporal feedback loop). The full GBH IS exact: for a
  // sample z held by domain i,
  //   m_i(z) = c_i·p̂_i(z) / Σ_j c_j·p̂_j(T_{·→j} z)
  // where the sum runs over the canonical AND every accepted neighbour, and the
  // shift T_{·→j} re-roots z's reconnection vertex onto domain j's primary
  // vertex (xs/Lo fixed; the per-domain target is the receiver-lobe p̂ evaluated
  // by restir_gi_receiver_phat_from_surface_or_geometry). This requires the full
  // domain set up front, so we GATHER accepted neighbours into a small fixed
  // array (≤ K_SPATIAL_GI) in pass 1, then fold each sample with its full-GBH
  // weight in pass 2. The reused-reservoir
  // resampling weight is  w_i = m_i · p̂_r(T_{i→r} z_i) · W_i · |∂T_{i→r}/∂·|
  // (no /p_src — W_i already bakes in the source pdf; the Jacobian alone carries
  // the reconnection-edge measure conversion).

  // Pass-1 gather: store each accepted neighbour's domain (xv/nv), its sample
  // (xs/ns/Lo), confidence c, UCW W, and shift Jacobian J at the canonical.
  var nQ: u32 = 0u;
  var qXv:  array<vec3f, 5>;
  var qNv:  array<vec3f, 5>;
  var qXs:  array<vec3f, 5>;
  var qNs:  array<vec3f, 5>;
  var qLo:  array<vec3f, 5>;
  var qSurf: array<PrimarySurface, 5>;
  var qC:   array<f32, 5>;
  var qW:   array<f32, 5>;
  var qJ:   array<f32, 5>;

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

    // Shift-compatibility gate: only reconnection samples with a matching path
    // prefix take the reconnection shift (single-bounce prefix = 1u). A 0 prefix
    // is an unpopulated Phase-0 cache → skip.
    if (rQ.prefixVertexCount != rCenter.prefixVertexCount
     || rQ.prefixVertexCount == 0u) { continue; }

    // Base half-G recovered from q's Phase-0 cache (cosReconOut/distRecon²) —
    // identical to reconnectionGeometryTerm(rQ.xv, rQ.xs, rQ.ns).
    let gBase = select(0.0, rQ.cosReconOut / (rQ.distRecon * rQ.distRecon),
                       rQ.distRecon > 1e-6);
    if (gBase <= 0.0) { continue; }

    // Shift Jacobian |∂T/∂·| = G(rCenter.xv ↔ q.xs) / gBase.
    let J = grisShiftJacobian(gBase, rCenter.xv, rQ.xs, rQ.ns);
    if (J <= 0.0) { continue; }

    // Non-degenerate shifted + native targets, else q contributes nothing.
    let qFullPx = vec2u(u32(qx), u32(qy)) * 2u + 1u;
    let surfQ = castPrimary(qFullPx, fullDims, ubo.cameraPos, invVP);

    let pHatQ_atR = restir_gi_receiver_phat_from_surface_or_geometry(
      centerSurf,
      rCenter.xv,
      rCenter.nv,
      rQ.xs,
      rQ.Lo,
    );
    if (pHatQ_atR < 1e-9) { continue; }
    let pHatQ_native = restir_gi_receiver_phat_from_surface_or_geometry(
      surfQ,
      rQ.xv,
      rQ.nv,
      rQ.xs,
      rQ.Lo,
    );
    if (pHatQ_native < 1e-9) { continue; }

    // Reconnection VISIBILITY — required for unbiasedness. If the shifted edge
    // rCenter.xv → q.xs is occluded the shift maps to zero contribution.
    if (!grisReconnectionVisible(rCenter.xv, rCenter.nv, rQ.xs)) { continue; }

    let Mq = min(rQ.M, M_CLAMP_SPATIAL);
    qXv[nQ] = rQ.xv; qNv[nQ] = rQ.nv;
    qXs[nQ] = rQ.xs; qNs[nQ] = rQ.ns; qLo[nQ] = rQ.Lo;
    qSurf[nQ] = surfQ;
    qC[nQ] = f32(Mq); qW[nQ] = rQ.W; qJ[nQ] = J;
    nQ = nQ + 1u;
  }

  // Pass-2 fold: each domain's sample carries its full-GBH weight m_i. The GBH
  // denominator for a fixed sample z = Σ_j c_j·p̂_j(T_{·→j} z) over canonical +
  // all gathered neighbours. fold(z, m_i·p̂_r(T z)·W_i·J_i) into rOut.

  // Canonical sample z_r — domain set evaluates p̂ at each xv for z_r's (xs,Lo).
  if (rCenter.M > 0u && pHatCanonNative > 1e-9) {
    var denomR = cR * pHatCanonNative;            // canonical's own term (J·target native)
    for (var j: u32 = 0u; j < nQ; j = j + 1u) {
      denomR += qC[j] * restir_gi_receiver_phat_from_surface_or_geometry(
        qSurf[j],
        qXv[j],
        qNv[j],
        rCenter.xs,
        rCenter.Lo,
      );
    }
    let m_canon = select(0.0, (cR * pHatCanonNative) / denomR, denomR > 1e-12);
    // No shift for the canonical's own sample (already at this pixel; J = 1).
    let w_canon = m_canon * pHatCanonNative * rCenter.W;
    updateReservoirGI(&rOut, rCenter.xs, rCenter.ns, rCenter.Lo, w_canon, &rng);
    rOut.M = rOut.M + rCenter.M;
  }

  // Each neighbour's sample z_q — same full-GBH denominator over all domains.
  for (var i: u32 = 0u; i < nQ; i = i + 1u) {
    let pHatQ_native = restir_gi_receiver_phat_from_surface_or_geometry(
      qSurf[i],
      qXv[i],
      qNv[i],
      qXs[i],
      qLo[i],
    );
    // GBH denominator: canonical's target for z_q + every neighbour's target.
    var denomQ = cR * restir_gi_receiver_phat_from_surface_or_geometry(
      centerSurf,
      rCenter.xv,
      rCenter.nv,
      qXs[i],
      qLo[i],
    );
    for (var j: u32 = 0u; j < nQ; j = j + 1u) {
      denomQ += qC[j] * restir_gi_receiver_phat_from_surface_or_geometry(
        qSurf[j],
        qXv[j],
        qNv[j],
        qXs[i],
        qLo[i],
      );
    }
    let m_q = select(0.0, (qC[i] * pHatQ_native) / denomQ, denomQ > 1e-12);
    // p̂_r(T z_q): q's sample re-rooted onto the canonical primary vertex.
    let pHatQ_atR = restir_gi_receiver_phat_from_surface_or_geometry(
      centerSurf,
      rCenter.xv,
      rCenter.nv,
      qXs[i],
      qLo[i],
    );
    let w_q = m_q * pHatQ_atR * qW[i] * qJ[i];
    updateReservoirGI(&rOut, qXs[i], qNs[i], qLo[i], w_q, &rng);
    rOut.M = rOut.M + u32(qC[i]);
  }

  // Finalise W from the chosen sample's p̂ at this pixel.
  // D5.3 (gris=true): GRIS reuse — W = w_sum / p̂ (the per-sample MIS weights
  // m_i already sum to 1; dividing by M again would under-energise the estimate).
  // Lin 2022 §generalised RIS: W = w_sum / p̂(z) with Σ m_i = 1.
  let pHatOut = restir_gi_receiver_phat_from_surface_or_geometry(
    centerSurf,
    rOut.xv,
    rOut.nv,
    rOut.xs,
    rOut.Lo,
  );
  finaliseGIReservoirWFromPHat(&rOut, ubo.restirGiWCap, true, pHatOut);

  // GRIS — refresh the Phase-0 reconnection-shift cache on the chosen sample so
  // the NEXT reuse pass (the ping-pong second spatial dispatch, or the next
  // frame's temporal reuse) sees a base edge rooted at THIS pixel's visible
  // vertex. updateReservoirGI re-roots xs/ns/Lo but not the cache; when a
  // neighbour's sample was chosen the stale cache would otherwise describe the
  // neighbour's edge. Recompute it here from rOut.xv → rOut.xs.
  if (rOut.M > 0u) {
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

/** ON (opt-in, restirPtReuse) include-graph entry.
 *  Over {@link SPATIAL_GI_MODULE} this variant DROPS `jacobianShift` (the
 *  legacy clamped-Jacobian reuse is gone — GRIS uses `grisShiftJacobian`) and
 *  ADDS:
 *    - `BVHNode` / `traceSceneAny`           → sceneTraversal (GRIS
 *                                              reconnection-visibility ray)
 *    - `grisReconnectionGeometryTerm` / `grisShiftJacobian` /
 *      `grisPairwiseDenom*`                  → grisReuse (GRIS Phase 1+2)
 *    - `castPrimary` + receiver-lobe p̂      → restirCastPrimary/restirGiMaterial
 *  Composed ONLY when the host opts into restirPtReuse; both variants emit the
 *  same @group(1) scene/material bindings, while only this variant emits the
 *  GRIS branch. */
export const SPATIAL_GI_GRIS_MODULE: WgslModule = {
  name: 'spatialGiGris',
  source: SPATIAL_GI_GRIS_WGSL,
  requires: ['walkaroundUbo', 'spatialGiCommon', 'sceneTraversal', 'reservoirGi', 'sharedPrimitives', 'grisReuse', 'materialDecode', 'cameraRays', 'restirCastPrimary', 'restirGiMaterial'],
};
