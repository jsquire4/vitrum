/**
 * ReSTIR-GI spatial reuse with a construction-time layout variant.
 *
 * The compact default keeps the legacy 20-u32 reservoir. `grisReuse` selects a
 * 28-u32 reservoir and a distinct shader that reconnects only the declared
 * diffuse/geometric one-bounce DDGI proxy. The GRIS variant evaluates up to six
 * native receiver domains through the full bounded transformed-density matrix,
 * traces reconnection visibility, folds exact attempt counts, and rejects stale
 * epochs. It is not ReSTIR PT and makes no unbiased path-tracing claim.
 *
 * Both variants bind the scene group; only the selected fixed shader/layout is
 * compiled for an engine instance.
 */

import type { WgslModule } from '../pipeline/wgslComposer.js';
import { reservoirGiAccessorsWgsl } from './reservoirGi.wgsl.js';

// ════════════════════════════════════════════════════════════════════════════
// OFF (default) — Sprint-17 spatial reuse plus receiver-material p-hat recast.
// The GRIS branch stays absent, but @group(1) is bound so rich receivers use the
// same material-aware target as the RIS producer.
// ════════════════════════════════════════════════════════════════════════════
export const SPATIAL_GI_WGSL = /* wgsl */ `

@group(0) @binding(0) var<storage, read>       sgi_resIn:  array<u32>;
@group(0) @binding(1) var<storage, read_write> sgi_resOut: array<u32>;
@group(0) @binding(2) var<uniform> ubo: WalkaroundUBO;

${reservoirGiAccessorsWgsl({
  loadReadBinding: 'sgi_resIn',
  storeReadWriteBinding: 'sgi_resOut',
})}

@compute @workgroup_size(8, 8, 1)
fn spatialGiMain(@builtin(global_invocation_id) gid: vec3u) {
  let fullDims = ubo.screenSize;
  let halfDims = fullDims / 2u;
  if (any(gid.xy >= halfDims)) { return; }

  let pixelIdx = gid.y * halfDims.x + gid.x;
  var rCenter = loadReservoirGI_ro(pixelIdx);

  // No surface here — skip reuse, copy through.
  if (rCenter.M == 0u) {
    storeReservoirGI_rw(pixelIdx, rCenter);
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
    let rQ = loadReservoirGI_ro(qIdx);
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
    if (!reservoirGiFinite(pHatZ) || !(pHatZ > 0.0)) { continue; }

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

  storeReservoirGI_rw(pixelIdx, rOut);
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
// ON (opt-in, grisReuse) — GRIS reconnection-shift reuse. Adds the
// @group(1) scene BVH/TLAS bindings + the reconnection-visibility ray + the
// GRIS combine branch + the all-technique canonical fold. Composed ONLY when
// grisReuse is set, with the two-group pipeline layout.
// ════════════════════════════════════════════════════════════════════════════
export const SPATIAL_GI_GRIS_WGSL = /* wgsl */ `

@group(0) @binding(0) var<storage, read>       sgi_resIn:  array<u32>;
@group(0) @binding(1) var<storage, read_write> sgi_resOut: array<u32>;
@group(0) @binding(2) var<uniform> ubo: WalkaroundUBO;

${reservoirGiAccessorsWgsl({
  loadReadBinding: 'sgi_resIn',
  storeReadWriteBinding: 'sgi_resOut',
})}

@group(1) @binding(5) var bvh_beer: texture_2d<u32>;

@compute @workgroup_size(8, 8, 1)
fn spatialGiMain(@builtin(global_invocation_id) gid: vec3u) {
  let fullDims = ubo.screenSize;
  let halfDims = fullDims / 2u;
  if (any(gid.xy >= halfDims)) { return; }
  let pixelIdx = gid.y * halfDims.x + gid.x;
  let epoch = grisHistoryEpoch();
  let rCenter = loadReservoirGI_ro(pixelIdx);
  if (rCenter.M == 0u
   || !reservoirGiFinite(rCenter.W) || !(rCenter.W > 0.0)
   || rCenter.historyEpoch != epoch
   || !reservoirGiFinite(rCenter.nativePHat) || !(rCenter.nativePHat > 0.0)
   || !reservoirGiFinite(rCenter.sampleVisibility) || !(rCenter.sampleVisibility > 0.0)
   || !grisSampleKindValid(rCenter.sampleKind)) {
    storeReservoirGI_rw(pixelIdx, emptyReservoirGI());
    return;
  }

  var rng = pcgInit(
    gid.x ^ (ubo.frameSeed * 0xA127u),
    gid.y ^ (ubo.frameSeed * 0x271Au),
    ubo.frameSeed ^ 0xBCD3u,
  );
  // Canonical plus at most five neighbours: gather the complete bounded
  // technique set before evaluating the all-candidate/all-technique matrix.
  var domains: array<ReservoirPT, 6>;
  var domainM: array<u32, 6>;
  var domainPixel: array<u32, 6>;
  domains[0] = rCenter;
  domainM[0] = min(rCenter.M, M_CLAMP_SPATIAL);
  domainPixel[0] = pixelIdx;
  var domainCount: u32 = 1u;

  for (var gather: u32 = 0u; gather < K_SPATIAL_GI; gather = gather + 1u) {
    let off = sampleDiscPx(&rng);
    let qx = i32(gid.x) + i32(round(off.x));
    let qy = i32(gid.y) + i32(round(off.y));
    if (qx < 0 || qy < 0 || u32(qx) >= halfDims.x || u32(qy) >= halfDims.y) { continue; }
    if (qx == i32(gid.x) && qy == i32(gid.y)) { continue; }
    let qIdx = u32(qy) * halfDims.x + u32(qx);
    var duplicateDomain = false;
    for (var existing: u32 = 0u; existing < domainCount; existing = existing + 1u) {
      duplicateDomain = duplicateDomain || domainPixel[existing] == qIdx;
    }
    if (duplicateDomain) { continue; }
    let q = loadReservoirGI_ro(qIdx);
    if (q.M == 0u
     || !reservoirGiFinite(q.W) || !(q.W > 0.0)
     || q.historyEpoch != epoch
     || !reservoirGiFinite(q.nativePHat) || !(q.nativePHat > 0.0)
     || !reservoirGiFinite(q.sampleVisibility) || !(q.sampleVisibility > 0.0)
     || !grisSampleKindValid(q.sampleKind)) { continue; }
    if (q.prefixVertexCount != 1u) { continue; }
    if (dot(rCenter.nv, q.nv) < ubo.restirGiSpatialNormalDotMin) { continue; }
    if (abs(dot(q.xv - rCenter.xv, rCenter.nv)) > ubo.restirGiSpatialCoplanarTol) { continue; }
    let Jq = grisDomainToCanonicalJacobian(q.xv, rCenter.xv, q.sampleKind, q.xs, q.ns);
    if (Jq <= 0.0) { continue; }
    let pCanonical = grisProxyPHatAt(
      rCenter.xv, rCenter.nv, q.sampleKind, q.xs, q.wi_recon, q.Lo,
    );
    if (!reservoirGiFinite(pCanonical) || !(pCanonical > 0.0)) { continue; }
    domains[domainCount] = q;
    domainM[domainCount] = min(q.M, M_CLAMP_SPATIAL);
    domainPixel[domainCount] = qIdx;
    domainCount = domainCount + 1u;
  }

  var out = emptyReservoirGI();
  out.xv = rCenter.xv;
  out.nv = rCenter.nv;
  for (var i: u32 = 0u; i < domainCount; i = i + 1u) {
    let candidate = domains[i];
    let attempts = domainM[i];
    let sourceJ = grisDomainToCanonicalJacobian(
      candidate.xv, rCenter.xv, candidate.sampleKind, candidate.xs, candidate.ns,
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
    for (var j: u32 = 0u; j < domainCount; j = j + 1u) {
      let technique = domains[j];
      let Jj = grisDomainToCanonicalJacobian(
        technique.xv, rCenter.xv, candidate.sampleKind, candidate.xs, candidate.ns,
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
      rCenter.xv, rCenter.nv,
      candidate.sampleKind, candidate.xs, candidate.wi_recon,
    );
    let pCanonical = grisProxyTargetAt(
      rCenter.xv, rCenter.nv,
      candidate.sampleKind, candidate.xs, candidate.wi_recon, candidate.Lo,
    ) * visibilityCanonical;
    if (!reservoirGiFinite(pCanonical) || !(pCanonical > 0.0)) {
      foldInvalidReservoirGICandidates(&out, attempts, candidate.sampleKind, epoch);
      continue;
    }
    let mappedXs = grisMappedXs(
      candidate.sampleKind, rCenter.xv, candidate.xs, candidate.wi_recon,
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
 *  Over {@link SPATIAL_GI_MODULE} this variant DROPS `jacobianShift` (the
 *  legacy clamped-Jacobian reuse is gone — GRIS uses `grisDomainToCanonicalJacobian`) and
 *  ADDS:
 *    - `BVHNode` / `traceSceneAny`           → sceneTraversal (GRIS
 *                                              reconnection-visibility ray)
 *    - `grisReconnectionGeometryTerm` / `grisDomainToCanonicalJacobian` /
 *      `grisTransformedDensity`                  → grisReuse (GRIS DDGI-proxy reuse)
 *    - `castPrimary` + receiver-lobe p̂      → restirCastPrimary/restirGiMaterial
 *  Composed ONLY when the host opts into grisReuse; both variants emit the
 *  same @group(1) scene/material bindings, while only this variant emits the
 *  GRIS branch. */
export const SPATIAL_GI_GRIS_MODULE: WgslModule = {
  name: 'spatialGiGris',
  source: SPATIAL_GI_GRIS_WGSL,
  requires: ['walkaroundUbo', 'spatialGiCommon', 'sceneTraversal', 'reservoirGi', 'sharedPrimitives', 'grisReuse', 'materialDecode', 'materialAtlas', 'cameraRays', 'restirCastPrimary', 'restirGiMaterial'],
};
