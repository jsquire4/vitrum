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
  if (rCenter.prefixVertexCount != GI_PREFIX_RECONNECTABLE) {
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
    gid.xy * 2u + vec2u(1u),
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
     || !reservoirGiFinite(q.W) || !(q.W > 0.0)
     || q.historyEpoch != epoch
     || !reservoirGiFinite(q.nativePHat) || !(q.nativePHat > 0.0)
     || !reservoirGiFinite(q.sampleVisibility) || !(q.sampleVisibility > 0.0)
     || !grisSampleKindValid(q.sampleKind)
     || q.prefixVertexCount != GI_PREFIX_RECONNECTABLE) {
      continue;
    }
    if (dot(rCenter.nv, q.nv) < ubo.restirGiSpatialNormalDotMin) { continue; }
    if (abs(dot(q.xv - rCenter.xv, rCenter.nv))
        > ubo.restirGiSpatialCoplanarTol) {
      continue;
    }
    let Jq = grisDomainToCanonicalJacobian(
      q.xv,
      rCenter.xv,
      q.sampleKind,
      q.xs,
      q.ns,
    );
    if (Jq <= 0.0) { continue; }
    let qSurface = castPrimary(
      vec2u(u32(qx), u32(qy)) * 2u + vec2u(1u),
      fullDims,
      ubo.cameraPos,
      invVP,
    );
    let pCanonical = grisMaterialPHatAt(
      domainSurface[0],
      rCenter.xv,
      rCenter.nv,
      q.sampleKind,
      q.xs,
      q.wi_recon,
      q.Lo,
    );
    if (!reservoirGiFinite(pCanonical) || !(pCanonical > 0.0)) { continue; }
    domains[domainCount] = q;
    domainM[domainCount] = min(q.M, M_CLAMP_SPATIAL);
    domainPixel[domainCount] = qIdx;
    domainSurface[domainCount] = qSurface;
    domainCount = domainCount + 1u;
  }

  var candidateLogWeight: array<f32, 6>;
  var candidatePHat: array<f32, 6>;
  var candidateVisibility: array<f32, 6>;
  var maxCandidateLogWeight = GRIS_LOG_ZERO;
  var totalAttempts = 0u;

  // Generalized balance heuristic (Lin et al. 2022, Eq. 7): M_j multiplies
  // the transformed target density for technique j. The source reservoir UCW
  // enters the canonical candidate weight once; it is not multiplied by M_i
  // a second time. Common max-log scales preserve unequal finite ratios.
  for (var i: u32 = 0u; i < domainCount; i = i + 1u) {
    candidateLogWeight[i] = GRIS_LOG_ZERO;
    candidatePHat[i] = 0.0;
    candidateVisibility[i] = 0.0;
    let candidate = domains[i];
    totalAttempts = reservoirGiSaturatingAddU32(
      totalAttempts,
      domainM[i],
    );
    let sourceJ = grisDomainToCanonicalJacobian(
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
      let Jj = grisDomainToCanonicalJacobian(
        technique.xv,
        rCenter.xv,
        candidate.sampleKind,
        candidate.xs,
        candidate.ns,
      );
      var pHatJ = 0.0;
      if (j == i) {
        pHatJ = candidate.nativePHat;
      } else if (Jj > 0.0) {
        pHatJ = grisMaterialPHatAt(
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
        pHatJ,
        Jj,
        Jj > 0.0 && pHatJ > 0.0,
      );
      techniqueLogMass[j] = logMass;
      maxTechniqueLogMass = max(maxTechniqueLogMass, logMass);
    }
    if (maxTechniqueLogMass == GRIS_LOG_ZERO) { continue; }

    var scaledDenominator = 0.0;
    for (var j: u32 = 0u; j < domainCount; j = j + 1u) {
      scaledDenominator =
        scaledDenominator
        + grisScaledMass(techniqueLogMass[j], maxTechniqueLogMass);
    }
    let sourceScaledMass =
      grisScaledMass(techniqueLogMass[i], maxTechniqueLogMass);
    if (!reservoirGiFinite(scaledDenominator) || !(scaledDenominator > 0.0)
     || !(sourceScaledMass > 0.0)) {
      continue;
    }
    let m = sourceScaledMass / scaledDenominator;
    let visibilityCanonical = grisProxyVisibilityAt(
      rCenter.xv,
      rCenter.nv,
      candidate.sampleKind,
      candidate.xs,
      candidate.wi_recon,
    );
    let pCanonical = grisMaterialTargetAt(
      domainSurface[0],
      rCenter.xv,
      rCenter.nv,
      candidate.sampleKind,
      candidate.xs,
      candidate.wi_recon,
      candidate.Lo,
    ) * visibilityCanonical;
    let logWeight = grisLogCanonicalResamplingWeight(
      m,
      pCanonical,
      candidate.W,
      sourceJ,
    );
    if (logWeight == GRIS_LOG_ZERO) { continue; }
    candidateLogWeight[i] = logWeight;
    candidatePHat[i] = pCanonical;
    candidateVisibility[i] = visibilityCanonical;
    maxCandidateLogWeight = max(maxCandidateLogWeight, logWeight);
  }

  var out = emptyReservoirGI();
  out.xv = rCenter.xv;
  out.nv = rCenter.nv;
  if (maxCandidateLogWeight != GRIS_LOG_ZERO) {
    for (var i: u32 = 0u; i < domainCount; i = i + 1u) {
      let candidate = domains[i];
      let weight = grisScaledMass(
        candidateLogWeight[i],
        maxCandidateLogWeight,
      );
      if (!(weight > 0.0)) { continue; }
      let mappedXs = grisMappedXs(
        candidate.sampleKind,
        rCenter.xv,
        candidate.xs,
        candidate.wi_recon,
      );
      let mappedLo = grisMappedLo(
        candidate.sampleKind,
        candidate.wi_recon,
        candidate.Lo,
      );
      updateReservoirGIWithMetadata(
        &out,
        mappedXs,
        candidate.ns,
        mappedLo,
        candidate.sampleKind,
        candidate.wi_recon,
        candidatePHat[i],
        candidateVisibility[i],
        epoch,
        weight,
        &rng,
      );
    }
  }
  // Confidence is the saturating sum of the already per-domain-clamped source
  // attempt counts. Candidate selection adds no synthetic attempt.
  out.M = totalAttempts;
  grisFinaliseLogScaledReservoir(
    &out,
    maxCandidateLogWeight,
    out.w_sum,
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
