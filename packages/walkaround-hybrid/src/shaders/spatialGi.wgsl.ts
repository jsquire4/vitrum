/**
 * ReSTIR-GI spatial reuse for the renderer's bounded one-bounce proxy.
 *
 * Generalized reconnection-shift reuse is the sole live implementation. The
 * canonical receiver plus at most five distinct compatible neighbours are
 * evaluated through the complete transformed-density matrix. The estimator
 * traces canonical visibility, preserves exact represented attempt counts,
 * rejects stale epochs, and uses ratio-preserving log-domain arithmetic.
 *
 * This is not full ReSTIR PT: the declared estimator remains the renderer's
 * one-bounce DDGI proxy. Camera-prefix dielectric samples remain local because
 * their full receiver-specific refractive prefix is not represented.
 */

import type { WgslModule } from '../pipeline/wgslComposer.js';
import { reservoirGiAccessorsWgsl } from './reservoirGi.wgsl.js';

export const SPATIAL_GI_WGSL = /* wgsl */ `

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
  let halfDims = restirGiDimensions();
  if (any(gid.xy >= halfDims)) { return; }
  let pixelIdx = gid.y * halfDims.x + gid.x;
  let epoch = grisHistoryEpoch();
  let rCenter = loadReservoirGI_ro(pixelIdx);
  if (rCenter.M == 0u
   || rCenter.historyEpoch != epoch
   || !reservoirGiFiniteReceiver(rCenter)) {
    storeReservoirGI_rw(pixelIdx, emptyReservoirGI());
    return;
  }
  if (!reservoirGiHasShiftableTechnique(rCenter)) {
    storeReservoirGI_rw(pixelIdx, rCenter);
    return;
  }

  var rng = pcgInit(
    gid.x ^ (ubo.frameSeed * 0xA127u),
    gid.y ^ (ubo.frameSeed * 0x271Au),
    ubo.frameSeed ^ 0xBCD3u,
  );

  // Canonical plus at most five distinct neighbours. Gather the complete
  // bounded technique set before evaluating any candidate MIS weight.
  var domains: array<ReservoirPT, 6>;
  var domainM: array<u32, 6>;
  var domainPixel: array<u32, 6>;
  var domainSurface: array<PrimarySurface, 6>;
  let invVP = invertMat4_common(ubo.projMatrix * ubo.viewMatrix);
  domains[0] = rCenter;
  domainM[0] = min(rCenter.M, M_CLAMP_SPATIAL);
  domainPixel[0] = pixelIdx;
  domainSurface[0] = castPrimary(
    restirGiFullPixel(gid.xy),
    fullDims,
    ubo.cameraPos,
    invVP,
  );
  var domainCount: u32 = 1u;

  for (var gather: u32 = 0u; gather < K_SPATIAL_GI; gather = gather + 1u) {
    let off = sampleDiscPx(&rng);
    let qx = i32(gid.x) + i32(round(off.x));
    let qy = i32(gid.y) + i32(round(off.y));
    if (qx < 0 || qy < 0
     || u32(qx) >= halfDims.x || u32(qy) >= halfDims.y) {
      continue;
    }
    if (qx == i32(gid.x) && qy == i32(gid.y)) { continue; }
    let qIdx = u32(qy) * halfDims.x + u32(qx);
    var duplicateDomain = false;
    for (var existing: u32 = 0u; existing < domainCount; existing = existing + 1u) {
      duplicateDomain = duplicateDomain || domainPixel[existing] == qIdx;
    }
    if (duplicateDomain) { continue; }

    let q = loadReservoirGI_ro(qIdx);
    if (q.M == 0u
     || q.historyEpoch != epoch
     || !reservoirGiHasShiftableTechnique(q)) {
      continue;
    }
    if (
      restirReservoirScaleValue() > 1u
      && q.receiverMaterialKey != rCenter.receiverMaterialKey
    ) {
      continue;
    }
    if (dot(rCenter.nv, q.nv) < ubo.restirGiSpatialNormalDotMin) { continue; }
    if (abs(dot(q.xv - rCenter.xv, rCenter.nv))
        > ubo.restirGiSpatialCoplanarTol) {
      continue;
    }
    // Technique membership depends only on receiver/domain compatibility, not
    // on whether this technique's selected sample happens to map to a valid
    // canonical contribution. The complete matrix below may assign that one
    // candidate a zero entry while this technique still contributes density to
    // another candidate's Eq. 7 denominator.
    let qSurface = castPrimary(
      restirGiFullPixel(vec2u(u32(qx), u32(qy))),
      fullDims,
      ubo.cameraPos,
      invVP,
    );
    domains[domainCount] = q;
    domainM[domainCount] = min(q.M, M_CLAMP_SPATIAL);
    domainPixel[domainCount] = qIdx;
    domainSurface[domainCount] = qSurface;
    domainCount = domainCount + 1u;
  }

  var candidateLogWeight: array<f32, 6>;
  var candidateLogPHat: array<f32, 6>;
  var candidateVisibility: array<f32, 6>;
  var totalAttempts = 0u;

  // Generalized balance heuristic (Lin et al. 2022, Eq. 7): M_j multiplies
  // the transformed target density for technique j. The source reservoir UCW
  // enters the canonical candidate weight once; it is not multiplied by M_i
  // a second time. Common max-log scales preserve unequal finite ratios.
  for (var i: u32 = 0u; i < domainCount; i = i + 1u) {
    candidateLogWeight[i] = GRIS_LOG_ZERO;
    candidateLogPHat[i] = GRIS_LOG_ZERO;
    candidateVisibility[i] = 0.0;
    let candidate = domains[i];
    totalAttempts = reservoirGiSaturatingAddU32(
      totalAttempts,
      domainM[i],
    );
    // The canonical receiver may retain its own angular/local selected suffix
    // as an identity-only row. Neighbour local representatives remain
    // denominator-only techniques; treating the canonical selection as an
    // early-return condition would make reuse depend on a WRS outcome and bias
    // away all simultaneously available safe rows.
    let canonicalLocal =
      i == 0u && reservoirGiHasLocalEstimator(candidate);
    if (canonicalLocal) {
      candidateLogWeight[i] = candidate.H;
      candidateLogPHat[i] = candidate.nativeLogPHat;
      candidateVisibility[i] = candidate.sampleVisibility;
      continue;
    }
    if (!reservoirGiHasShiftableCandidate(candidate)) { continue; }
    let logSourceJ = grisLogDomainToCanonicalJacobian(
      candidate.xv,
      rCenter.xv,
      candidate.sampleKind,
      candidate.xs,
      candidate.ns,
    );

    var techniqueLogMass: array<f32, 6>;
    var maxTechniqueLogMass = GRIS_LOG_ZERO;
    for (var j: u32 = 0u; j < domainCount; j = j + 1u) {
      let technique = domains[j];
      let logJj = grisLogDomainToCanonicalJacobian(
        technique.xv,
        rCenter.xv,
        candidate.sampleKind,
        candidate.xs,
        candidate.ns,
      );
      var logPHatJ = GRIS_LOG_ZERO;
      if (j == i) {
        logPHatJ = candidate.nativeLogPHat;
      } else if (reservoirGiValidLog(logJj)) {
        logPHatJ = grisLogMaterialPHatAt(
          domainSurface[j],
          technique.xv,
          technique.nv,
          candidate.sampleKind,
          candidate.xs,
          candidate.wi_recon,
          candidate.Lo,
        );
      }
      let logMass = grisLogWeightedTransformedDensity(
        domainM[j],
        logPHatJ,
        logJj,
        reservoirGiValidLog(logJj) && reservoirGiValidLog(logPHatJ),
      );
      techniqueLogMass[j] = logMass;
      maxTechniqueLogMass = max(maxTechniqueLogMass, logMass);
    }
    if (!reservoirGiValidLog(maxTechniqueLogMass)) { continue; }

    var scaledDenominator = 0.0;
    for (var j: u32 = 0u; j < domainCount; j = j + 1u) {
      scaledDenominator =
        scaledDenominator
        + grisScaledMass(techniqueLogMass[j], maxTechniqueLogMass);
    }
    let logDenominator = grisLogTechniqueDenominator(
      maxTechniqueLogMass,
      scaledDenominator,
    );
    if (!reservoirGiValidLog(logDenominator)
     || !reservoirGiValidLog(techniqueLogMass[i])) {
      continue;
    }
    let logMisWeight = techniqueLogMass[i] - logDenominator;
    let tintCanonical = grisProxyTintAt(
      rCenter.xv,
      rCenter.nv,
      domainSurface[0].geoNormal,
      candidate.sampleKind,
      candidate.xs,
      candidate.wi_recon,
    );
    let canonicalContribution = grisMaterialContributionAt(
      domainSurface[0],
      rCenter.xv,
      rCenter.nv,
      candidate.sampleKind,
      candidate.xs,
      candidate.wi_recon,
      candidate.Lo,
    );
    let logCanonicalPHat = reservoirGiLogPositive(
      luminance(canonicalContribution * tintCanonical),
    );
    let logWeight = grisLogCanonicalResamplingWeight(
      logMisWeight,
      logCanonicalPHat,
      candidate.H,
      candidate.nativeLogPHat,
      logSourceJ,
    );
    if (!reservoirGiValidLog(logWeight)) { continue; }
    candidateLogWeight[i] = logWeight;
    candidateLogPHat[i] = logCanonicalPHat;
    candidateVisibility[i] = luminance(tintCanonical);
  }

  var out = emptyReservoirGI();
  out.xv = rCenter.xv;
  out.nv = rCenter.nv;
  out.receiverMaterialKey = rCenter.receiverMaterialKey;
  out.historyEpoch = epoch;
  out.prefixVertexCount = GI_PREFIX_RECONNECTABLE;
  var reuseWrs = representedWrsInit();
  let canonicalLocalParticipated =
    reservoirGiHasLocalEstimator(domains[0]);
  for (var i: u32 = 0u; i < domainCount; i = i + 1u) {
      let candidate = domains[i];
      if (!reservoirGiValidLog(candidateLogWeight[i])) { continue; }
      let identityOnly = i == 0u && canonicalLocalParticipated;
      let mappedXs = select(
        grisMappedXs(
          candidate.sampleKind,
          rCenter.xv,
          candidate.xs,
          candidate.wi_recon,
        ),
        candidate.xs,
        identityOnly,
      );
      let mappedLo = select(
        grisMappedLo(
          candidate.sampleKind,
          candidate.wi_recon,
          candidate.Lo,
        ),
        candidate.Lo,
        identityOnly,
      );
      updateReservoirGIWithMetadata(
        &out, &reuseWrs,
        mappedXs,
        candidate.ns,
        mappedLo,
        candidate.sampleKind,
        candidate.wi_recon,
        select(
          GI_SAMPLE_FLAG_RECAST_TINT,
          candidate.sampleFlags,
          identityOnly,
        ),
        candidateLogPHat[i],
        candidateVisibility[i],
        epoch,
        candidateLogWeight[i],
        &rng,
      );
  }
  // Confidence is the saturating sum of the already per-domain-clamped source
  // attempt counts. Candidate selection adds no synthetic attempt.
  out.M = totalAttempts;
  grisFinaliseRepresentedReservoir(
    &out,
    reuseWrs,
    ubo.restirGiWCap,
  );
  refreshGrisMetadata(&out);
  storeReservoirGI_rw(pixelIdx, out);
}
`;

export const SPATIAL_GI_MODULE: WgslModule = {
  name: 'spatialGi',
  source: SPATIAL_GI_WGSL,
  requires: [
    'walkaroundUbo',
    'spatialGiCommon',
    'sceneTraversal',
    'reservoirGi',
    'sharedPrimitives',
    'grisReuse',
    'materialDecode',
    'materialAtlas',
    'cameraRays',
    'restirCastPrimary',
    'restirGiMaterial',
  ],
};
