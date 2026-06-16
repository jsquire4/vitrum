/**
 * Sprint 17 — ReSTIR-GI temporal reuse.
 *
 * Reproject the current half-res pixel through the previous-frame camera
 * matrices to find the same world point in the previous reservoir, then
 * combine its sample with this frame's RIS-output reservoir under the
 * standard ReSTIR temporal-reuse rules:
 *
 *   - Geometric consistency: |Δdepth| < 0.1 × depth and |Δnormal| < 25°.
 *   - M-clamp at 20 to bound history accumulation (Bitterli 2020 §5.2).
 *   - Jacobian reconnection shift (common.wgsl jacobianReconnectionShift)
 *     re-weights the prev sample's contribution at this pixel's visible
 *     point so the integrand stays unbiased under shifted reconnections.
 *
 * Reads:  reservoirGiCurrent  (this frame's RIS output)
 *         reservoirGiPrevious (last frame's spatial output)
 *         gNormalDepth        (full-res, sampled at full-res pixel)
 *         WalkaroundUBO       (prevViewMatrix, viewMatrix, projMatrix, screenSize)
 * Writes: reservoirGiCurrent  (in-place — RIS output is consumed, temporally fused)
 *
 * Half-resolution: dispatches W/2 × H/2 invocations, like the RIS pass.
 *
 * ════════════════════════════════════════════════════════════════════════════
 * COMPILE-TIME GRIS gate (restirPtReuse) — why this file has TWO shader bodies
 * ════════════════════════════════════════════════════════════════════════════
 * See the matching header in `spatialGi.wgsl.ts`. GRIS reconnection-shift reuse
 * (Phases 1+2; Lin et al. 2022) is opt-in via `HybridEngineOptions.restirPtReuse`
 * and is gated at PIPELINE-COMPILE time because turning it on STRUCTURALLY
 * changes this pass (it swaps in the GRIS branch and Phase-0 cache stride). A
 * runtime UBO flag is NOT enough for those structural choices. Both variants now
 * bind the shared `@group(1)` scene/material group because receiver-lobe p-hat
 * recasts need material payloads in the default path too. So:
 *   - {@link TEMPORAL_GI_MODULE}      — OFF (default). Standard temporal reuse
 *                                        with receiver-material p-hat recast.
 *   - {@link TEMPORAL_GI_GRIS_MODULE} — ON. Adds the GRIS branch + `grisReuse`
 *                                        dep on the same group layout.
 * {@link compilePipelines} composes whichever module matches the host flag and
 * builds the matching pipeline layout; {@link TemporalGIReservoirPass} only
 * calls `setBindGroup(1, …)` when ON.
 *
 * Bindings:
 *   @group(0) @binding(0) reservoirGiCurrent  (storage, read_write)
 *   @group(0) @binding(1) reservoirGiPrevious (storage, read)
 *   @group(0) @binding(2) WalkaroundUBO       (uniform)
 *   @group(1)             scene BVH/TLAS/material atlas (read-only) — both
 *                         variants use it to recast the receiver material for
 *                         true GI receiver-lobe p-hat; GRIS also traces
 *                         reconnection visibility through it.
 *
 * GRIS combine (ON variant only): the previous-frame reservoir is combined via
 * the unbiased reconnection shift (re-root rPrev's xs onto rCur.xv), its
 * Jacobian (recovered from rPrev's Phase-0 cache), a reconnection-visibility
 * ray, and a pairwise generalized-balance MIS weight (§pairwise MIS).
 */

import type { WgslModule } from '../pipeline/wgslComposer.js';
import { TEMPORAL_GI_COMMON_WGSL } from './temporalGiCommon.wgsl.js';

// ════════════════════════════════════════════════════════════════════════════
// OFF (default) — Sprint-17 temporal reuse plus receiver-material p-hat recast.
// The GRIS branch stays absent, but @group(1) is bound so rich receivers use the
// same material-aware target as the RIS producer.
// ════════════════════════════════════════════════════════════════════════════
export const TEMPORAL_GI_WGSL = /* wgsl */ `

@group(0) @binding(0) var<storage, read_write> tgi_resCurrent: array<u32>;
@group(0) @binding(1) var<storage, read>       tgi_resPrev:    array<u32>;
@group(0) @binding(2) var<uniform> ubo: WalkaroundUBO;

@group(1) @binding(0) var<storage, read> bvh:          array<BVHNode>;
@group(1) @binding(1) var<storage, read> bvh_index:    array<vec4u>;
@group(1) @binding(2) var<storage, read> bvh_position: array<vec4f>;
@group(1) @binding(6) var<storage, read> tlasNodes: array<BVHNode>;
@group(1) @binding(7) var<storage, read> tlasInstanceIndices: array<u32>;
@group(1) @binding(8) var<storage, read> tlasBlasRoots: array<u32>;
@group(1) @binding(9) var<storage, read> tlasInstanceWorldToLocal: array<vec4f>;
@group(1) @binding(10) var<storage, read> tlasInstanceLocalToWorld: array<vec4f>;

// The temporal-GI M clamp (ubo.restirGiMClamp, Cornell default 50)
// controls how strongly the previous-frame reservoir dominates temporal
// reuse.  Higher = the chosen sample changes less often per-pixel → less
// per-frame pattern jitter (the temporal accumulator's per-frame
// contribution looks stabler).  Bitterli 2020 uses M=20 for ReSTIR-DI;
// Majercik 2021 §4.5 suggests ~30–100 for GI since the indirect signal
// varies less per pixel than DI light-source swaps.  Empirically 50 cuts
// visible pattern dance on Cornell static frames in half compared to 20
// without introducing motion lag (the camera-move reset path forces α=1
// and discards prev independently). Library consumers override via
// HybridEngineOptions.restirGiMClamp.
${TEMPORAL_GI_COMMON_WGSL}

@compute @workgroup_size(8, 8, 1)
fn temporalGiMain(@builtin(global_invocation_id) gid: vec3u) {
  let fullDims = ubo.screenSize;
  let halfDims = fullDims / 2u;
  if (any(gid.xy >= halfDims)) { return; }

  let pixelIdx = gid.y * halfDims.x + gid.x;
  var rCur = loadReservoirGI_rw(&tgi_resCurrent, pixelIdx);

  // Need a visible-surface point to reproject. If current's RIS pass wrote
  // an empty reservoir (primary miss / glass / metal), nothing to fuse.
  if (rCur.M == 0u) {
    storeReservoirGI_rw(&tgi_resCurrent, pixelIdx, rCur);
    return;
  }

  let curFullPx = gid.xy * 2u + 1u;
  let curVp = ubo.projMatrix * ubo.viewMatrix;
  let curInvVP = invertMat4_common(curVp);
  let curSurf = castPrimary(curFullPx, fullDims, ubo.cameraPos, curInvVP);

  // Reproject through prev camera. Use the current visible-point xv as the
  // world anchor — same world point in both frames (camera moves, not scene).
  let prevHalfPx = projectToPrevHalfPx(rCur.xv, halfDims, fullDims);
  if (prevHalfPx.x < 0 || prevHalfPx.y < 0
   || u32(prevHalfPx.x) >= halfDims.x || u32(prevHalfPx.y) >= halfDims.y) {
    storeReservoirGI_rw(&tgi_resCurrent, pixelIdx, rCur);
    return;
  }
  let prevIdx = u32(prevHalfPx.y) * halfDims.x + u32(prevHalfPx.x);
  let rPrev = loadReservoirGI_ro(&tgi_resPrev, prevIdx);

  if (rPrev.M == 0u || rPrev.W <= 0.0) {
    storeReservoirGI_rw(&tgi_resCurrent, pixelIdx, rCur);
    return;
  }

  // Geometric-consistency test: compare current vs prev visible-point depth
  // and normal. Reject under occlusion or material swap.
  let dDepth = abs(length(rCur.xv - ubo.cameraPos) - length(rPrev.xv - ubo.cameraPos));
  let depthRef = max(1e-3, length(rCur.xv - ubo.cameraPos));
  if (dDepth / depthRef > DEPTH_REL_TOL) {
    storeReservoirGI_rw(&tgi_resCurrent, pixelIdx, rCur);
    return;
  }
  if (dot(rCur.nv, rPrev.nv) < NORMAL_DOT_MIN) {
    storeReservoirGI_rw(&tgi_resCurrent, pixelIdx, rCur);
    return;
  }

  // M-clamp: bound prev history before contributing.
  let prevM = min(rPrev.M, ubo.restirGiMClamp);

  // Reconnection-shift jacobian: prev's reservoir holds the (xs, ns, Lo)
  // visible *from* rPrev.xv. We want to weight it as if observed from rCur.xv.
  let J = jacobianReconnectionShift(rCur.xv, rCur.nv, rPrev.xv, rPrev.xs, rPrev.ns);
  if (J <= 0.0) {
    storeReservoirGI_rw(&tgi_resCurrent, pixelIdx, rCur);
    return;
  }

  // Compute the prev sample's p̂ at the current pixel.
  let toS = rPrev.xs - rCur.xv;
  let distS2 = dot(toS, toS);
  if (distS2 < 1e-8) {
    storeReservoirGI_rw(&tgi_resCurrent, pixelIdx, rCur);
    return;
  }
  let wiZ = toS / sqrt(distS2);
  let pHatZ_prev = restir_gi_receiver_phat_from_surface_or_geometry(
    curSurf,
    rCur.xv,
    rCur.nv,
    rPrev.xs,
    rPrev.Lo,
  );
  if (pHatZ_prev < 1e-9) {
    storeReservoirGI_rw(&tgi_resCurrent, pixelIdx, rCur);
    return;
  }

  // Combined-RIS weight: prev contributes w = p̂_at_cur × W × M × J.
  let w_prev = pHatZ_prev * rPrev.W * f32(prevM) * J;
  // Combine. Mirror the standard ReSTIR temporal-reuse formula.
  var rng = pcgInit(
    gid.x ^ (ubo.frameSeed * 0x71E5u),
    gid.y ^ (ubo.frameSeed * 0xE571u),
    ubo.frameSeed ^ 0x9B7Fu,
  );
  let M_total = rCur.M + prevM;
  updateReservoirGI(&rCur, rPrev.xs, rPrev.ns, rPrev.Lo, w_prev, &rng);
  rCur.M = M_total;

  // Finalise W with the chosen sample's p̂ at this pixel.
  // D5.3 (gris=false): standard RIS — divide by M (MIS weight 1 per candidate).
  let pHatCurFinal = restir_gi_receiver_phat_from_surface_or_geometry(
    curSurf,
    rCur.xv,
    rCur.nv,
    rCur.xs,
    rCur.Lo,
  );
  finaliseGIReservoirWFromPHat(&rCur, ubo.restirGiWCap, false, pHatCurFinal);

  storeReservoirGI_rw(&tgi_resCurrent, pixelIdx, rCur);
}
`;

/** OFF (default) include-graph entry.
 *  T9-stepC — narrowed from `['common']` to the modules this pass uses:
 *    - `WalkaroundUBO` / `INV_PI`            → walkaroundUbo
 *    - `loadReservoirGI_*` / `storeReservoirGI_rw` / `updateReservoirGI`
 *                                            → reservoirGi
 *    - `pcgInit` / `luminance` / `safe_normalize` → sharedPrimitives
 *    - `jacobianReconnectionShift`           → jacobianShift
 *    - `invertMat4_common` (reprojection)    → cameraRays (uses `Ray` →
 *                                              sceneTraversal)
 *  No GRIS → no grisReuse. Verified complete by the static ident-resolution gate. */
export const TEMPORAL_GI_MODULE: WgslModule = {
  name: 'temporalGi',
  source: TEMPORAL_GI_WGSL,
  requires: ['walkaroundUbo', 'sceneTraversal', 'reservoirGi', 'sharedPrimitives', 'jacobianShift', 'cameraRays', 'restirCastPrimary', 'restirGiMaterial'],
};

// ════════════════════════════════════════════════════════════════════════════
// ON (opt-in, restirPtReuse) — GRIS reconnection-shift temporal reuse. Adds the
// @group(1) scene BVH/TLAS bindings + the reconnection-visibility ray + the
// two-domain (current/previous) pairwise-MIS GRIS combine. Composed ONLY when
// restirPtReuse is set, with the two-group pipeline layout.
// ════════════════════════════════════════════════════════════════════════════
export const TEMPORAL_GI_GRIS_WGSL = /* wgsl */ `

@group(0) @binding(0) var<storage, read_write> tgi_resCurrent: array<u32>;
@group(0) @binding(1) var<storage, read>       tgi_resPrev:    array<u32>;
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

const TGI_GRIS_NORMAL_BIAS: f32 = 1e-3;

// GRIS reconnection visibility — clear edge xv → xs through the scene BVH/TLAS.
fn tgiReconnectionVisible(xv: vec3f, nv: vec3f, xs: vec3f) -> bool {
  let toS = xs - xv;
  let dist = length(toS);
  if (dist < 1e-4) { return false; }
  let wi = toS / dist;
  let orig = xv + nv * TGI_GRIS_NORMAL_BIAS;
  let occ = traceSceneAnyCastMask(
    ubo.bvhMode, ubo.tlasNodeCount,
    &bvh_index, &bvh_position, &bvh,
    &tlasNodes, &tlasInstanceIndices, &tlasBlasRoots,
    &tlasInstanceWorldToLocal, &tlasInstanceLocalToWorld,
    orig, wi, dist - 2e-3, ubo.triIntersectEpsilon, true,
    bvh_material, BVH_MATERIAL_TEX_WIDTH);
  return !occ;
}

// The temporal-GI M clamp (ubo.restirGiMClamp, Cornell default 50)
// controls how strongly the previous-frame reservoir dominates temporal
// reuse.  Higher = the chosen sample changes less often per-pixel → less
// per-frame pattern jitter (the temporal accumulator's per-frame
// contribution looks stabler).  Bitterli 2020 uses M=20 for ReSTIR-DI;
// Majercik 2021 §4.5 suggests ~30–100 for GI since the indirect signal
// varies less per pixel than DI light-source swaps.  Empirically 50 cuts
// visible pattern dance on Cornell static frames in half compared to 20
// without introducing motion lag (the camera-move reset path forces α=1
// and discards prev independently). Library consumers override via
// HybridEngineOptions.restirGiMClamp.
${TEMPORAL_GI_COMMON_WGSL}

@compute @workgroup_size(8, 8, 1)
fn temporalGiMain(@builtin(global_invocation_id) gid: vec3u) {
  let fullDims = ubo.screenSize;
  let halfDims = fullDims / 2u;
  if (any(gid.xy >= halfDims)) { return; }

  let pixelIdx = gid.y * halfDims.x + gid.x;
  var rCur = loadReservoirGI_rw(&tgi_resCurrent, pixelIdx);

  // Need a visible-surface point to reproject. If current's RIS pass wrote
  // an empty reservoir (primary miss / glass / metal), nothing to fuse.
  if (rCur.M == 0u) {
    storeReservoirGI_rw(&tgi_resCurrent, pixelIdx, rCur);
    return;
  }

  let curFullPx = gid.xy * 2u + 1u;
  let curVp = ubo.projMatrix * ubo.viewMatrix;
  let curInvVP = invertMat4_common(curVp);
  let curSurf = castPrimary(curFullPx, fullDims, ubo.cameraPos, curInvVP);

  // Reproject through prev camera. Use the current visible-point xv as the
  // world anchor — same world point in both frames (camera moves, not scene).
  let prevHalfPx = projectToPrevHalfPx(rCur.xv, halfDims, fullDims);
  if (prevHalfPx.x < 0 || prevHalfPx.y < 0
   || u32(prevHalfPx.x) >= halfDims.x || u32(prevHalfPx.y) >= halfDims.y) {
    storeReservoirGI_rw(&tgi_resCurrent, pixelIdx, rCur);
    return;
  }
  let prevIdx = u32(prevHalfPx.y) * halfDims.x + u32(prevHalfPx.x);
  let rPrev = loadReservoirGI_ro(&tgi_resPrev, prevIdx);
  // We only pack prevViewProjMatrix today, not a previous inverse-VP/camera
  // origin. This best-effort recast uses the current camera and the
  // surface-or-geometry helper falls back to rPrev.xv/rPrev.nv when it does not
  // hit the stored previous receiver point.
  let prevFullPx = vec2u(u32(prevHalfPx.x), u32(prevHalfPx.y)) * 2u + 1u;
  let prevSurf = castPrimary(prevFullPx, fullDims, ubo.cameraPos, curInvVP);

  if (rPrev.M == 0u || rPrev.W <= 0.0) {
    storeReservoirGI_rw(&tgi_resCurrent, pixelIdx, rCur);
    return;
  }

  // Geometric-consistency test: compare current vs prev visible-point depth
  // and normal. Reject under occlusion or material swap.
  let dDepth = abs(length(rCur.xv - ubo.cameraPos) - length(rPrev.xv - ubo.cameraPos));
  let depthRef = max(1e-3, length(rCur.xv - ubo.cameraPos));
  if (dDepth / depthRef > DEPTH_REL_TOL) {
    storeReservoirGI_rw(&tgi_resCurrent, pixelIdx, rCur);
    return;
  }
  if (dot(rCur.nv, rPrev.nv) < NORMAL_DOT_MIN) {
    storeReservoirGI_rw(&tgi_resCurrent, pixelIdx, rCur);
    return;
  }

  // M-clamp: bound prev history before contributing.
  let prevM = min(rPrev.M, ubo.restirGiMClamp);

  var rng = pcgInit(
    gid.x ^ (ubo.frameSeed * 0x71E5u),
    gid.y ^ (ubo.frameSeed * 0xE571u),
    ubo.frameSeed ^ 0x9B7Fu,
  );

  // ── GRIS reconnection-shift temporal reuse (Lin 2022 §5 + Eq. 12 + MIS) ──
  // Two-domain pairwise MIS: the CURRENT (canonical) sample and the
  // reprojected PREVIOUS sample. Build a FRESH reservoir so BOTH are folded
  // in with their own pairwise-MIS weights (the canonical is a resampling
  // technique, not an un-weighted base).
  let cCur = f32(rCur.M);
  let cPrev = f32(prevM);
  let pHatCur_native = restir_gi_receiver_phat_from_surface_or_geometry(
    curSurf,
    rCur.xv,
    rCur.nv,
    rCur.xs,
    rCur.Lo,
  );

  // Decide whether the prev sample is a VALID reconnection-shift candidate.
  // (prefix match, non-degenerate base half-G, positive Jacobian, non-zero
  // shifted+native targets, AND reconnection-visible). If any fails the prev
  // contributes nothing and the canonical stands alone (m_cur = 1).
  var prevValid = (rPrev.prefixVertexCount == rCur.prefixVertexCount)
               && (rPrev.prefixVertexCount != 0u);
  var J: f32 = 0.0;
  var pHatPrev_atCur: f32 = 0.0;
  var pHatPrev_native: f32 = 0.0;
  if (prevValid) {
    let gBase = select(0.0, rPrev.cosReconOut / (rPrev.distRecon * rPrev.distRecon),
                       rPrev.distRecon > 1e-6);
    J = grisShiftJacobian(gBase, rCur.xv, rPrev.xs, rPrev.ns);
    pHatPrev_atCur = restir_gi_receiver_phat_from_surface_or_geometry(
      curSurf,
      rCur.xv,
      rCur.nv,
      rPrev.xs,
      rPrev.Lo,
    );
    pHatPrev_native = restir_gi_receiver_phat_from_surface_or_geometry(
      prevSurf,
      rPrev.xv,
      rPrev.nv,
      rPrev.xs,
      rPrev.Lo,
    );
    prevValid = (gBase > 0.0) && (J > 0.0)
             && (pHatPrev_atCur >= 1e-9) && (pHatPrev_native >= 1e-9)
             && tgiReconnectionVisible(rCur.xv, rCur.nv, rPrev.xs);
  }

  // Pairwise MIS weights (2-domain generalized balance):
  //   m_prev = c_prev·p̂_prev_native / (c_cur·p̂_cur(T z_prev) + c_prev·p̂_prev_native)
  //   m_cur  = c_cur·p̂_cur_native  / (c_cur·p̂_cur_native + c_prev·p̂_prev(T z_cur))
  // When prev is invalid m_cur collapses to 1 (canonical alone).
  var rGris = emptyReservoirGI();
  rGris.xv = rCur.xv; rGris.nv = rCur.nv;
  rGris.prefixVertexCount = rCur.prefixVertexCount;

  // Canonical (current) sample, MIS-weighted against the prev pair.
  if (rCur.M > 0u && pHatCur_native > 1e-9) {
    var m_cur: f32 = 1.0;
    if (prevValid) {
      // prev's sample re-rooted onto the CURRENT domain is just p̂_cur(T z_prev);
      // the canonical's own sample re-rooted onto prev is p̂_prev(T⁻¹ z_cur).
      let pHatPrev_atCurSample = restir_gi_receiver_phat_from_surface_or_geometry(
        prevSurf,
        rPrev.xv,
        rPrev.nv,
        rCur.xs,
        rCur.Lo,
      );
      let denomCur = grisPairwiseDenomCanonical(cCur, pHatCur_native, cPrev, pHatPrev_atCurSample);
      m_cur = select(1.0, (cCur * pHatCur_native) / denomCur, denomCur > 1e-12);
    }
    // Canonical: no shift (J = 1), already at this pixel (no visibility re-test).
    let w_cur = m_cur * pHatCur_native * rCur.W;
    let oldM = rGris.M;
    updateReservoirGI(&rGris, rCur.xs, rCur.ns, rCur.Lo, w_cur, &rng);
    rGris.M = oldM + rCur.M;
  }

  // Previous (reprojected) sample, reconnection-shifted + MIS-weighted.
  if (prevValid) {
    let denomPrev = grisPairwiseDenomNeighbor(cCur, pHatPrev_atCur, cPrev, pHatPrev_native);
    let m_prev = select(0.0, (cPrev * pHatPrev_native) / denomPrev, denomPrev > 1e-12);
    // GRIS resampling weight for a REUSED reservoir sample (Lin 2022, Alg. 3 /
    // Eq. 9):  w_prev = m_prev · p̂_cur(T z_prev) · W_prev · |∂T/∂·|.
    //
    // NO /p_src.  rPrev is a *reservoir*: its W_prev already bakes in the source
    // pdf (the producer finalised W = w_sum/(M·p̂)). Reservoir reuse re-weights
    // by m·p̂_canonical·W·J only — the SAME shape as the canonical fold above
    // (m_cur·p̂_cur·W). This pass is the temporal FEEDBACK loop: rGris becomes
    // next frame's rPrev. An extra /p_src (p_src = cosine-hemisphere pdf ∈ (0,
    // 1/π]) would multiply the carried weight by 1/p_src ≈ π…∞ every frame,
    // driving the recursion's gain above 1 → W pins at restirGiWCap and the GI
    // mean climbs without bound (the V19 grison / grison-r0 divergence). The
    // shift Jacobian J alone carries the reconnection-edge measure conversion.
    let w_prev = m_prev * pHatPrev_atCur * rPrev.W * J;
    let oldM = rGris.M;
    updateReservoirGI(&rGris, rPrev.xs, rPrev.ns, rPrev.Lo, w_prev, &rng);
    rGris.M = oldM + prevM;
  }

  // GRIS finalise: W = w_sum / p̂ (the MIS weights already sum to 1 — no /M).
  // D5.3 (gris=true): GRIS — divide by 1 (pairwise MIS weights Σ=1, no /M).
  let pHatGrisFinal = restir_gi_receiver_phat_from_surface_or_geometry(
    curSurf,
    rGris.xv,
    rGris.nv,
    rGris.xs,
    rGris.Lo,
  );
  finaliseGIReservoirWFromPHat(&rGris, ubo.restirGiWCap, true, pHatGrisFinal);
  if (rGris.M > 0u) {
    // Refresh the Phase-0 cache so downstream spatial reuse sees a base edge
    // rooted at THIS pixel's visible vertex.
    let toRecon = rGris.xs - rGris.xv;
    let dRecon = length(toRecon);
    if (dRecon > 1e-6) {
      let wiR = toRecon / dRecon;
      rGris.wi_recon = wiR;
      rGris.distRecon = dRecon;
      rGris.cosReconOut = abs(dot(rGris.ns, -wiR));
      rGris.pdfReconBsdf = max(0.0, dot(rGris.nv, wiR)) * INV_PI;
      rGris.prefixVertexCount = 1u;
    }
  }
  storeReservoirGI_rw(&tgi_resCurrent, pixelIdx, rGris);
}
`;

/** ON (opt-in, restirPtReuse) include-graph entry.
 *  Over {@link TEMPORAL_GI_MODULE} this variant DROPS `jacobianShift` (the
 *  legacy clamped-Jacobian reuse is gone — GRIS uses `grisShiftJacobian`) and
 *  ADDS:
 *    - `traceSceneAny` / `BVHNode`           → sceneTraversal (GRIS
 *                                              reconnection-visibility ray;
 *                                              already in the closure via
 *                                              cameraRays, but referenced here)
 *    - `grisReconnectionGeometryTerm` / `grisShiftJacobian` /
 *      `grisPairwiseDenom*`                  → grisReuse (GRIS Phase 1+2)
 *    - `castPrimary` + receiver-lobe p̂      → restirCastPrimary/restirGiMaterial
 *  Composed ONLY when the host opts into restirPtReuse; both variants emit the
 *  same @group(1) scene/material bindings, while only this variant emits the
 *  GRIS branch. */
export const TEMPORAL_GI_GRIS_MODULE: WgslModule = {
  name: 'temporalGiGris',
  source: TEMPORAL_GI_GRIS_WGSL,
  requires: ['walkaroundUbo', 'sceneTraversal', 'reservoirGi', 'sharedPrimitives', 'cameraRays', 'grisReuse', 'materialDecode', 'restirCastPrimary', 'restirGiMaterial'],
};
