/**
 * ReSTIR-GI temporal reuse with a construction-time layout variant.
 *
 * The compact default keeps the legacy 20-u32 reservoir. `grisReuse` selects a
 * 28-u32 shader/layout that combines the current native receiver and the prior
 * native receiver for the declared diffuse/geometric one-bounce DDGI proxy. The
 * GRIS variant uses the complete two-domain transformed-density matrix,
 * visibility support, exact attempt counts, and a mutation epoch. It is not
 * ReSTIR PT and makes no unbiased path-tracing claim.
 */

import type { WgslModule } from '../pipeline/wgslComposer.js';
import { TEMPORAL_GI_COMMON_WGSL, TEMPORAL_GI_MCLAMP_COMMENT_WGSL } from './temporalGiCommon.wgsl.js';
import { reservoirGiAccessorsWgsl } from './reservoirGi.wgsl.js';

// ════════════════════════════════════════════════════════════════════════════
// OFF (default) — Sprint-17 temporal reuse plus receiver-material p-hat recast.
// The GRIS branch stays absent, but @group(1) is bound so rich receivers use the
// same material-aware target as the RIS producer.
// ════════════════════════════════════════════════════════════════════════════
export const TEMPORAL_GI_WGSL = /* wgsl */ `

@group(0) @binding(0) var<storage, read_write> tgi_resCurrent: array<u32>;
@group(0) @binding(1) var<storage, read>       tgi_resPrev:    array<u32>;
@group(0) @binding(2) var<uniform> ubo: WalkaroundUBO;

${reservoirGiAccessorsWgsl({
  loadReadWriteBinding: 'tgi_resCurrent',
  loadReadBinding: 'tgi_resPrev',
  storeReadWriteBinding: 'tgi_resCurrent',
})}

${TEMPORAL_GI_MCLAMP_COMMENT_WGSL}
${TEMPORAL_GI_COMMON_WGSL}

@compute @workgroup_size(8, 8, 1)
fn temporalGiMain(@builtin(global_invocation_id) gid: vec3u) {
  let fullDims = ubo.screenSize;
  let halfDims = fullDims / 2u;
  if (any(gid.xy >= halfDims)) { return; }

  let pixelIdx = gid.y * halfDims.x + gid.x;
  var rCur = loadReservoirGI_rw(pixelIdx);

  // Need a visible-surface point to reproject. If current's RIS pass wrote
  // an empty reservoir (primary miss / glass / metal), nothing to fuse.
  if (rCur.M == 0u) {
    storeReservoirGI_rw(pixelIdx, rCur);
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
    storeReservoirGI_rw(pixelIdx, rCur);
    return;
  }
  let prevIdx = u32(prevHalfPx.y) * halfDims.x + u32(prevHalfPx.x);
  let rPrev = loadReservoirGI_ro(prevIdx);

  if (rPrev.M == 0u || rPrev.W <= 0.0) {
    storeReservoirGI_rw(pixelIdx, rCur);
    return;
  }

  // Geometric-consistency test: compare current vs prev visible-point depth
  // and normal. Reject under occlusion or material swap.
  let dDepth = abs(length(rCur.xv - ubo.cameraPos) - length(rPrev.xv - ubo.cameraPos));
  let depthRef = max(1e-3, length(rCur.xv - ubo.cameraPos));
  if (dDepth / depthRef > DEPTH_REL_TOL) {
    storeReservoirGI_rw(pixelIdx, rCur);
    return;
  }
  if (dot(rCur.nv, rPrev.nv) < NORMAL_DOT_MIN) {
    storeReservoirGI_rw(pixelIdx, rCur);
    return;
  }

  // M-clamp: bound prev history before contributing.
  let prevM = min(rPrev.M, ubo.restirGiMClamp);

  // Reconnection-shift jacobian: prev's reservoir holds the (xs, ns, Lo)
  // visible *from* rPrev.xv. We want to weight it as if observed from rCur.xv.
  let J = jacobianReconnectionShift(rCur.xv, rCur.nv, rPrev.xv, rPrev.xs, rPrev.ns);
  if (J <= 0.0) {
    storeReservoirGI_rw(pixelIdx, rCur);
    return;
  }

  // Compute the prev sample's p̂ at the current pixel.
  let toS = rPrev.xs - rCur.xv;
  let distS2 = dot(toS, toS);
  if (distS2 < 1e-8) {
    storeReservoirGI_rw(pixelIdx, rCur);
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
  if (!reservoirGiFinite(pHatZ_prev) || !(pHatZ_prev > 0.0)) {
    storeReservoirGI_rw(pixelIdx, rCur);
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

  storeReservoirGI_rw(pixelIdx, rCur);
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
// ON (opt-in, grisReuse) — GRIS reconnection-shift temporal reuse. Adds the
// @group(1) scene BVH/TLAS bindings + the reconnection-visibility ray + the
// two-domain (current/previous) all-technique transformed-density GRIS combine. Composed ONLY when
// grisReuse is set, with the two-group pipeline layout.
// ════════════════════════════════════════════════════════════════════════════
export const TEMPORAL_GI_GRIS_WGSL = /* wgsl */ `

@group(0) @binding(0) var<storage, read_write> tgi_resCurrent: array<u32>;
@group(0) @binding(1) var<storage, read>       tgi_resPrev:    array<u32>;
@group(0) @binding(2) var<uniform> ubo: WalkaroundUBO;

${reservoirGiAccessorsWgsl({
  loadReadWriteBinding: 'tgi_resCurrent',
  loadReadBinding: 'tgi_resPrev',
  storeReadWriteBinding: 'tgi_resCurrent',
})}

@group(1) @binding(5) var bvh_beer: texture_2d<u32>;
${TEMPORAL_GI_MCLAMP_COMMENT_WGSL}
${TEMPORAL_GI_COMMON_WGSL}

@compute @workgroup_size(8, 8, 1)
fn temporalGiMain(@builtin(global_invocation_id) gid: vec3u) {
  let fullDims = ubo.screenSize;
  let halfDims = fullDims / 2u;
  if (any(gid.xy >= halfDims)) { return; }
  let pixelIdx = gid.y * halfDims.x + gid.x;
  let epoch = grisHistoryEpoch();
  let current = loadReservoirGI_rw(pixelIdx);
  if (current.M == 0u
   || !reservoirGiFinite(current.W) || !(current.W > 0.0)
   || current.historyEpoch != epoch
   || !reservoirGiFinite(current.nativePHat) || !(current.nativePHat > 0.0)
   || !reservoirGiFinite(current.sampleVisibility) || !(current.sampleVisibility > 0.0)
   || !grisSampleKindValid(current.sampleKind)) {
    storeReservoirGI_rw(pixelIdx, emptyReservoirGI());
    return;
  }

  let prevHalfPx = projectToPrevHalfPx(current.xv, halfDims, fullDims);
  if (prevHalfPx.x < 0 || prevHalfPx.y < 0
   || u32(prevHalfPx.x) >= halfDims.x || u32(prevHalfPx.y) >= halfDims.y) {
    storeReservoirGI_rw(pixelIdx, current);
    return;
  }
  let prevIdx = u32(prevHalfPx.y) * halfDims.x + u32(prevHalfPx.x);
  let previous = loadReservoirGI_ro(prevIdx);
  if (previous.M == 0u
   || !reservoirGiFinite(previous.W) || !(previous.W > 0.0)
   || previous.historyEpoch != epoch
   || !reservoirGiFinite(previous.nativePHat) || !(previous.nativePHat > 0.0)
   || !reservoirGiFinite(previous.sampleVisibility) || !(previous.sampleVisibility > 0.0)
   || !grisSampleKindValid(previous.sampleKind)) {
    storeReservoirGI_rw(pixelIdx, current);
    return;
  }

  let dDepth = abs(length(current.xv - ubo.cameraPos) - length(previous.xv - ubo.cameraPos));
  let depthRef = max(1e-3, length(current.xv - ubo.cameraPos));
  if (dDepth / depthRef > DEPTH_REL_TOL || dot(current.nv, previous.nv) < NORMAL_DOT_MIN) {
    storeReservoirGI_rw(pixelIdx, current);
    return;
  }
  let previousJ = grisDomainToCanonicalJacobian(
    previous.xv, current.xv, previous.sampleKind, previous.xs, previous.ns,
  );
  let previousCanonicalPHat = grisProxyPHatAt(
    current.xv, current.nv,
    previous.sampleKind, previous.xs, previous.wi_recon, previous.Lo,
  );
  if (previousJ <= 0.0
   || !reservoirGiFinite(previousCanonicalPHat)
   || !(previousCanonicalPHat > 0.0)) {
    storeReservoirGI_rw(pixelIdx, current);
    return;
  }

  var rng = pcgInit(
    gid.x ^ (ubo.frameSeed * 0x71E5u),
    gid.y ^ (ubo.frameSeed * 0xE571u),
    ubo.frameSeed ^ 0x9B7Fu,
  );
  var domains: array<ReservoirPT, 2>;
  var domainM: array<u32, 2>;
  domains[0] = current;
  domains[1] = previous;
  domainM[0] = min(current.M, ubo.restirGiMClamp);
  domainM[1] = min(previous.M, ubo.restirGiMClamp);

  var out = emptyReservoirGI();
  out.xv = current.xv;
  out.nv = current.nv;
  for (var i: u32 = 0u; i < 2u; i = i + 1u) {
    let candidate = domains[i];
    let attempts = domainM[i];
    let sourceJ = grisDomainToCanonicalJacobian(
      candidate.xv, current.xv, candidate.sampleKind, candidate.xs, candidate.ns,
    );
    let sourceDensity = grisTransformedDensity(
      candidate.nativePHat,
      sourceJ,
      sourceJ > 0.0 && candidate.sampleVisibility > 0.0,
    );
    if (sourceDensity <= 0.0) {
      foldInvalidReservoirGICandidates(&out, attempts, candidate.sampleKind, epoch);
      continue;
    }

    var denominator = 0.0;
    for (var j: u32 = 0u; j < 2u; j = j + 1u) {
      let technique = domains[j];
      let Jj = grisDomainToCanonicalJacobian(
        technique.xv, current.xv, candidate.sampleKind, candidate.xs, candidate.ns,
      );
      var pHatJ = 0.0;
      if (j == i) {
        pHatJ = candidate.nativePHat;
      } else if (Jj > 0.0) {
        pHatJ = grisProxyPHatAt(
          technique.xv, technique.nv,
          candidate.sampleKind, candidate.xs, candidate.wi_recon, candidate.Lo,
        );
      }
      let transformed = grisTransformedDensity(
        pHatJ,
        Jj,
        Jj > 0.0 && pHatJ > 0.0,
      );
      denominator = denominator + grisWeightedDensity(domainM[j], transformed);
    }
    if (!reservoirGiFinite(denominator) || !(denominator > 0.0)) {
      foldInvalidReservoirGICandidates(&out, attempts, candidate.sampleKind, epoch);
      continue;
    }

    let sourceNumerator = grisWeightedDensity(attempts, sourceDensity);
    let m = sourceNumerator / denominator;
    if (!reservoirGiFinite(m) || !(m > 0.0)) {
      foldInvalidReservoirGICandidates(&out, attempts, candidate.sampleKind, epoch);
      continue;
    }
    let visibilityCanonical = grisProxyVisibilityAt(
      current.xv, current.nv,
      candidate.sampleKind, candidate.xs, candidate.wi_recon,
    );
    let pCanonical = grisProxyTargetAt(
      current.xv, current.nv,
      candidate.sampleKind, candidate.xs, candidate.wi_recon, candidate.Lo,
    ) * visibilityCanonical;
    if (!reservoirGiFinite(pCanonical) || !(pCanonical > 0.0)) {
      foldInvalidReservoirGICandidates(&out, attempts, candidate.sampleKind, epoch);
      continue;
    }
    let mappedXs = grisMappedXs(
      candidate.sampleKind, current.xv, candidate.xs, candidate.wi_recon,
    );
    let mappedLo = grisMappedLo(candidate.sampleKind, candidate.wi_recon, candidate.Lo);
    let weight = m * pCanonical * candidate.W * sourceJ;
    if (!reservoirGiFinite(weight) || !(weight > 0.0)) {
      foldInvalidReservoirGICandidates(&out, attempts, candidate.sampleKind, epoch);
      continue;
    }
    let oldM = out.M;
    updateReservoirGIWithMetadata(
      &out, mappedXs, candidate.ns, mappedLo,
      candidate.sampleKind, candidate.wi_recon,
      pCanonical, visibilityCanonical, epoch,
      weight, &rng,
    );
    out.M = reservoirGiSaturatingAddU32(oldM, attempts);
  }

  finaliseGIReservoirWFromPHat(&out, ubo.restirGiWCap, true, out.nativePHat);
  refreshGrisMetadata(&out);
  storeReservoirGI_rw(pixelIdx, out);
}
`;

/** ON (opt-in, grisReuse) include-graph entry.
 *  Over {@link TEMPORAL_GI_MODULE} this variant DROPS `jacobianShift` (the
 *  legacy clamped-Jacobian reuse is gone — GRIS uses `grisDomainToCanonicalJacobian`) and
 *  ADDS:
 *    - `traceSceneAny` / `BVHNode`           → sceneTraversal (GRIS
 *                                              reconnection-visibility ray;
 *                                              already in the closure via
 *                                              cameraRays, but referenced here)
 *    - `grisReconnectionGeometryTerm` / `grisDomainToCanonicalJacobian` /
 *      `grisTransformedDensity`                  → grisReuse (GRIS DDGI-proxy reuse)
 *    - `castPrimary` + receiver-lobe p̂      → restirCastPrimary/restirGiMaterial
 *  Composed ONLY when the host opts into grisReuse; both variants emit the
 *  same @group(1) scene/material bindings, while only this variant emits the
 *  GRIS branch. */
export const TEMPORAL_GI_GRIS_MODULE: WgslModule = {
  name: 'temporalGiGris',
  source: TEMPORAL_GI_GRIS_WGSL,
  requires: ['walkaroundUbo', 'sceneTraversal', 'reservoirGi', 'sharedPrimitives', 'cameraRays', 'grisReuse', 'materialDecode', 'materialAtlas', 'restirCastPrimary', 'restirGiMaterial'],
};
