/**
 * ReSTIR-DI temporal reuse.
 *
 * The previous pixel is found by reprojection, then recast against the current
 * scene BVH using the previous camera. Reuse is allowed only when that recast
 * identifies the same visible surface. Accepted current/previous reservoirs are
 * combined with generalized Talbot MIS over their represented attempts.
 */

import type { WgslModule } from '../pipeline/wgslComposer.js';
import { reservoirDiAccessorsWgsl } from './reservoirDi.wgsl.js';

export const TEMPORAL_WGSL = /* wgsl */ `

@group(0) @binding(5) var<storage, read_write> currentReservoir:  array<u32>;
@group(0) @binding(6) var<storage, read>       previousReservoir: array<u32>;

${reservoirDiAccessorsWgsl({
  loadReadWriteBinding: 'currentReservoir',
  loadReadBinding: 'previousReservoir',
  storeReadWriteBinding: 'currentReservoir',
})}

@group(2) @binding(0) var<uniform> ubo: WalkaroundUBO;

fn reprojectToPrev(world: vec3f, dims: vec2u) -> vec2i {
  let clip = ubo.prevViewProjMatrix * vec4f(world, 1.0);
  if (clip.w <= 0.0) { return vec2i(-1, -1); }
  let ndc = clip.xyz / clip.w;
  if (
    ndc.x < -1.0 || ndc.x > 1.0 ||
    ndc.y < -1.0 || ndc.y > 1.0 ||
    ndc.z < -1.0 || ndc.z > 1.0
  ) {
    return vec2i(-1, -1);
  }
  let uv = vec2f(ndc.x * 0.5 + 0.5, 1.0 - (ndc.y * 0.5 + 0.5));
  return vec2i(i32(uv.x * f32(dims.x)), i32(uv.y * f32(dims.y)));
}

// Reprojection alone is not a disocclusion test. Recast the historical pixel
// against the CURRENT BVH and require exact primitive/material identity plus
// conservative geometric agreement with the current visible point. This
// rejects camera disocclusion and fails safe for changed geometry.
fn temporalSurfaceCorresponds(
  currentSurface: PrimarySurface,
  previousSurfaceNow: PrimarySurface,
  previousRay: Ray,
) -> bool {
  if (!previousSurfaceNow.hit) { return false; }
  if (
    previousSurfaceNow.instanceId != currentSurface.instanceId ||
    previousSurfaceNow.triangleId != currentSurface.triangleId ||
    previousSurfaceNow.materialKey != currentSurface.materialKey
  ) {
    return false;
  }

  let expectedPreviousDepth = length(currentSurface.pos - previousRay.origin);
  let depthDifference = abs(previousSurfaceNow.depth - expectedPreviousDepth);
  let worldDifference = length(previousSurfaceNow.pos - currentSurface.pos);
  let depthTolerance = max(
    ubo.spatialDepthTolFloor * 4.0,
    expectedPreviousDepth * 0.02,
  );
  let worldTolerance = max(
    ubo.spatialDepthTolFloor * 8.0,
    expectedPreviousDepth * 0.02,
  );
  return
    reservoirDiFinite(expectedPreviousDepth) &&
    reservoirDiFinite(depthDifference) &&
    reservoirDiFinite(worldDifference) &&
    depthDifference <= depthTolerance &&
    worldDifference <= worldTolerance &&
    dot(previousSurfaceNow.normal, currentSurface.normal) >= 0.9;
}

fn combineReceiverIndependentTemporalDI(
  current: ReservoirDI,
  previousInput: ReservoirDI,
  rng: ptr<function, u32>,
) -> ReservoirDI {
  var previous = previousInput;
  scaleReservoirDIToM(&previous, ubo.temporalMClampDI);
  var combined = emptyReservoirDI();
  var wrs = representedWrsInit();
  updateReservoirDI(
    &combined,
    &wrs,
    current.lightId,
    current.xi,
    reservoirDiCoarseReuseLogWeight(current),
    rng,
  );
  updateReservoirDI(
    &combined,
    &wrs,
    previous.lightId,
    previous.xi,
    reservoirDiCoarseReuseLogWeight(previous),
    rng,
  );
  combined.areaM = reservoirDiSaturatingAddU32(current.areaM, previous.areaM);
  combined.envM = reservoirDiSaturatingAddU32(current.envM, previous.envM);
  combined.M = reservoirDiSaturatingAddU32(current.M, previous.M);
  if (wrs.hasSelection) {
    let pHat = restir_di_coarse_proposal_phat(
      combined.lightId,
      combined.xi,
    );
    finaliseReservoirDIFromNativeWrs(&combined, wrs, pHat);
  }
  return combined;
}

@compute @workgroup_size(8, 8, 1)
fn temporalMain(@builtin(global_invocation_id) gid: vec3u) {
  let dims = ubo.screenSize;
  let reservoirDims = restirDiDimensions();
  if (any(gid.xy >= reservoirDims)) { return; }

  let pixelIdx = gid.y * reservoirDims.x + gid.x;
  var current = loadReservoirDI_rw(pixelIdx);
  // Sparse RIS does not write the complementary checkerboard parity. The bytes
  // left in current are therefore the same terminal reservoir copied into
  // previous at the end of the preceding frame, not a new proposal. Treating
  // both as independent inputs doubles M/support and overstates confidence.
  // Start gap pixels with no current technique; valid reprojected history is
  // still evaluated below against the current receiver and carried once.
  let checkerboardGap =
    ubo.checkerboardOn == 1u &&
    restirReservoirScaleValue() == 1u &&
    ((gid.x + gid.y) & 1u) != ubo.frameParity;
  if (checkerboardGap) {
    current = emptyReservoirDI();
  }
  if (restirReservoirScaleValue() > 1u) {
    let previous = loadReservoirDI_ro(pixelIdx);
    var coarseRng = pcgInit(
      gid.x ^ 12345u,
      gid.y ^ 67890u,
      ubo.frameSeed ^ 0xABCDu,
    );
    current = combineReceiverIndependentTemporalDI(
      current,
      previous,
      &coarseRng,
    );
    storeReservoirDI_rw(pixelIdx, current);
    return;
  }
  let vp = ubo.projMatrix * ubo.viewMatrix;
  let invVP = invertMat4_common(vp);
  let currentSurface = castPrimary(gid.xy, dims, ubo.cameraPos, invVP);
  if (!currentSurface.hit) {
    storeReservoirDI_rw(pixelIdx, current);
    return;
  }

  let previousPixel = reprojectToPrev(currentSurface.pos, dims);
  if (any(previousPixel < vec2i(0)) || any(previousPixel >= vec2i(dims))) {
    storeReservoirDI_rw(pixelIdx, current);
    return;
  }

  let previousIndex = u32(previousPixel.y) * dims.x + u32(previousPixel.x);
  var previous = loadReservoirDI_ro(previousIndex);
  if (previous.M == 0u) {
    storeReservoirDI_rw(pixelIdx, current);
    return;
  }

  let previousInvVP = invertMat4_common(ubo.prevViewProjMatrix);
  let previousRay = generatePrimaryRayFromInvVP_common(
    u32(previousPixel.x),
    u32(previousPixel.y),
    dims.x,
    dims.y,
    previousInvVP,
  );
  let previousSurfaceNow = castPrimaryFromInvVP(
    vec2u(previousPixel),
    dims,
    previousInvVP,
  );
  if (!temporalSurfaceCorresponds(currentSurface, previousSurfaceNow, previousRay)) {
    storeReservoirDI_rw(pixelIdx, current);
    return;
  }

  scaleReservoirDIToM(&previous, ubo.temporalMClampDI);

  // Full two-technique generalized Talbot matrix. For candidate y_i, every
  // domain evaluates pHat_j(y_i); M_j pHat_j(y_i) forms the denominator.
  let currentSupport =
    reservoirDiSupportForLight(current, current.lightId);
  var currentAtCurrent = 0.0;
  var currentAtPrevious = 0.0;
  if (currentSupport > 0u && reservoirDiHasEstimatorNumerator(current)) {
    currentAtCurrent = restir_di_compute_phat_xi(
      current.lightId,
      current.xi,
      currentSurface,
    );
    currentAtPrevious = restir_di_compute_phat_xi(
      current.lightId,
      current.xi,
      previousSurfaceNow,
    );
  }

  let previousSupport =
    reservoirDiSupportForLight(previous, previous.lightId);
  var previousAtCurrent = 0.0;
  var previousAtPrevious = 0.0;
  if (previousSupport > 0u && reservoirDiHasEstimatorNumerator(previous)) {
    previousAtCurrent = restir_di_compute_phat_xi(
      previous.lightId,
      previous.xi,
      currentSurface,
    );
    previousAtPrevious = restir_di_compute_phat_xi(
      previous.lightId,
      previous.xi,
      previousSurfaceNow,
    );
  }

  let currentSourceLogDensity = reservoirDiLogWeightedDensity(
    currentSupport,
    currentAtCurrent,
  );
  let currentPreviousLogDensity = reservoirDiLogWeightedDensity(
    reservoirDiSupportForLight(previous, current.lightId),
    currentAtPrevious,
  );
  let currentMaxLogDensity = max(
    currentSourceLogDensity,
    currentPreviousLogDensity,
  );
  let currentScaledDenominator =
    reservoirDiScaledDensityFromLog(
      currentSourceLogDensity,
      currentMaxLogDensity,
    ) +
    reservoirDiScaledDensityFromLog(
      currentPreviousLogDensity,
      currentMaxLogDensity,
    );
  let currentLogDenominator = reservoirDiLogSumExpFromMaxScale(
    currentMaxLogDensity,
    currentScaledDenominator,
  );

  let previousCurrentLogDensity = reservoirDiLogWeightedDensity(
    reservoirDiSupportForLight(current, previous.lightId),
    previousAtCurrent,
  );
  let previousSourceLogDensity = reservoirDiLogWeightedDensity(
    previousSupport,
    previousAtPrevious,
  );
  let previousMaxLogDensity = max(
    previousCurrentLogDensity,
    previousSourceLogDensity,
  );
  let previousScaledDenominator =
    reservoirDiScaledDensityFromLog(
      previousCurrentLogDensity,
      previousMaxLogDensity,
    ) +
    reservoirDiScaledDensityFromLog(
      previousSourceLogDensity,
      previousMaxLogDensity,
    );
  let previousLogDenominator = reservoirDiLogSumExpFromMaxScale(
    previousMaxLogDensity,
    previousScaledDenominator,
  );
  let currentLogWeight = reservoirDiGeneralizedReuseLogWeight(
    currentSourceLogDensity,
    currentLogDenominator,
    currentAtCurrent,
    currentAtCurrent,
    current.logEstimatorNumerator,
  );
  let previousLogWeight = reservoirDiGeneralizedReuseLogWeight(
    previousSourceLogDensity,
    previousLogDenominator,
    previousAtCurrent,
    previousAtPrevious,
    previous.logEstimatorNumerator,
  );

  var rng = pcgInit(gid.x ^ 12345u, gid.y ^ 67890u, ubo.frameSeed ^ 0xABCDu);
  var combined = emptyReservoirDI();
  var wrs = representedWrsInit();
  updateReservoirDI(
    &combined,
    &wrs,
    current.lightId,
    current.xi,
    currentLogWeight,
    &rng,
  );
  updateReservoirDI(
    &combined,
    &wrs,
    previous.lightId,
    previous.xi,
    previousLogWeight,
    &rng,
  );
  combined.areaM = reservoirDiSaturatingAddU32(current.areaM, previous.areaM);
  combined.envM = reservoirDiSaturatingAddU32(current.envM, previous.envM);
  combined.M = reservoirDiSaturatingAddU32(current.M, previous.M);
  var selectedCanonicalDensity = 0.0;
  if (wrs.hasSelection) {
    selectedCanonicalDensity = restir_di_compute_phat_xi(
      combined.lightId,
      combined.xi,
      currentSurface,
    );
  }
  finaliseReservoirDIFromGeneralizedReuse(
    &combined,
    wrs,
    selectedCanonicalDensity,
  );

  storeReservoirDI_rw(pixelIdx, combined);
}
`;

export const TEMPORAL_MODULE: WgslModule = {
  name: 'temporal',
  source: TEMPORAL_WGSL,
  requires: ['restirPHat', 'restirCastPrimary'],
};
