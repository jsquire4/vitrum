/**
 * ReSTIR-GI temporal reuse for the renderer's bounded one-bounce proxy.
 *
 * Generalized reconnection-shift reuse is the sole live implementation. The
 * current and reprojected native receiver techniques are evaluated through the
 * complete transformed-density matrix. Visibility support, exact represented
 * attempt counts, mutation epochs, and ratio-preserving log-domain MIS are
 * carried by the canonical 28-u32 reservoir ABI.
 *
 * Camera-prefix dielectric samples are deliberately local-only: their `Lo`
 * includes receiver-specific transmission throughput, and the full refractive
 * prefix required for a valid inverse shift is not represented.
 */

import type { WgslModule } from '../pipeline/wgslComposer.js';
import {
  TEMPORAL_GI_COMMON_WGSL,
  TEMPORAL_GI_MCLAMP_COMMENT_WGSL,
} from './temporalGiCommon.wgsl.js';
import { reservoirGiAccessorsWgsl } from './reservoirGi.wgsl.js';

export const TEMPORAL_GI_WGSL = /* wgsl */ `

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
  let halfDims = restirGiDimensions();
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
  if (current.prefixVertexCount != GI_PREFIX_RECONNECTABLE) {
    storeReservoirGI_rw(pixelIdx, current);
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
   || !grisSampleKindValid(previous.sampleKind)
   || previous.prefixVertexCount != GI_PREFIX_RECONNECTABLE) {
    storeReservoirGI_rw(pixelIdx, current);
    return;
  }
  if (
    restirReservoirScaleValue() > 1u
    && previous.receiverMaterialKey != current.receiverMaterialKey
  ) {
    storeReservoirGI_rw(pixelIdx, current);
    return;
  }

  let currentDepth = safe_length(current.xv - ubo.cameraPos);
  let previousDepth = safe_length(previous.xv - ubo.cameraPos);
  let dDepth = abs(currentDepth - previousDepth);
  let depthRef = max(walkaroundRayOriginBias(), currentDepth);
  if (dDepth / depthRef > DEPTH_REL_TOL || dot(current.nv, previous.nv) < NORMAL_DOT_MIN) {
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
  var domainSurface: array<PrimarySurface, 2>;
  domains[0] = current;
  domains[1] = previous;
  domainM[0] = min(current.M, ubo.restirGiMClamp);
  domainM[1] = min(previous.M, ubo.restirGiMClamp);
  let currentInvVP = invertMat4_common(ubo.projMatrix * ubo.viewMatrix);
  domainSurface[0] = castPrimary(
    restirGiFullPixel(gid.xy),
    fullDims,
    ubo.cameraPos,
    currentInvVP,
  );
  let previousInvVP = invertMat4_common(ubo.prevViewProjMatrix);
  domainSurface[1] = castPrimaryFromInvVP(
    restirGiFullPixel(vec2u(prevHalfPx)),
    fullDims,
    previousInvVP,
  );

  var candidateLogWeight: array<f32, 2>;
  var candidatePHat: array<f32, 2>;
  var candidateVisibility: array<f32, 2>;
  var maxCandidateLogWeight = GRIS_LOG_ZERO;
  var totalAttempts = 0u;

  // Generalized balance heuristic (Lin et al. 2022, Eq. 7): M_j multiplies
  // the transformed target density for technique j. The source reservoir UCW
  // enters the canonical candidate weight once; it is not multiplied by M_i
  // a second time.
  for (var i: u32 = 0u; i < 2u; i = i + 1u) {
    candidateLogWeight[i] = GRIS_LOG_ZERO;
    candidatePHat[i] = 0.0;
    candidateVisibility[i] = 0.0;
    totalAttempts = reservoirGiSaturatingAddU32(totalAttempts, domainM[i]);
    let candidate = domains[i];
    let sourceJ = grisDomainToCanonicalJacobian(
      candidate.xv, current.xv, candidate.sampleKind, candidate.xs, candidate.ns,
    );

    var techniqueLogMass: array<f32, 2>;
    var maxTechniqueLogMass = GRIS_LOG_ZERO;
    for (var j: u32 = 0u; j < 2u; j = j + 1u) {
      let technique = domains[j];
      let Jj = grisDomainToCanonicalJacobian(
        technique.xv, current.xv, candidate.sampleKind, candidate.xs, candidate.ns,
      );
      var pHatJ = 0.0;
      if (j == i) {
        pHatJ = candidate.nativePHat;
      } else if (Jj > 0.0) {
        pHatJ = grisMaterialPHatAt(
          domainSurface[j],
          technique.xv, technique.nv,
          candidate.sampleKind, candidate.xs, candidate.wi_recon, candidate.Lo,
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
    for (var j: u32 = 0u; j < 2u; j = j + 1u) {
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
      current.xv, current.nv,
      candidate.sampleKind, candidate.xs, candidate.wi_recon,
    );
    let pCanonical = grisMaterialTargetAt(
      domainSurface[0],
      current.xv, current.nv,
      candidate.sampleKind, candidate.xs, candidate.wi_recon, candidate.Lo,
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
  out.xv = current.xv;
  out.nv = current.nv;
  out.receiverMaterialKey = current.receiverMaterialKey;
  if (maxCandidateLogWeight != GRIS_LOG_ZERO) {
    for (var i: u32 = 0u; i < 2u; i = i + 1u) {
      let candidate = domains[i];
      let weight = grisScaledMass(
        candidateLogWeight[i],
        maxCandidateLogWeight,
      );
      if (!(weight > 0.0)) { continue; }
      let mappedXs = grisMappedXs(
        candidate.sampleKind, current.xv, candidate.xs, candidate.wi_recon,
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

export const TEMPORAL_GI_MODULE: WgslModule = {
  name: 'temporalGi',
  source: TEMPORAL_GI_WGSL,
  requires: [
    'walkaroundUbo',
    'sceneTraversal',
    'reservoirGi',
    'sharedPrimitives',
    'cameraRays',
    'grisReuse',
    'materialDecode',
    'materialAtlas',
    'restirCastPrimary',
    'restirGiMaterial',
  ],
};
