/**
 * ReSTIR-DI spatial reuse.
 *
 * Each round gathers the center and up to five geometrically compatible
 * neighbors. Their chosen samples are combined with generalized Talbot MIS,
 * evaluating every candidate under every gathered surface domain. The host
 * swaps bindings 5 and 7 for round two, making the rounds a real ping-pong.
 */

import type { WgslModule } from '../pipeline/wgslComposer.js';
import { reservoirDiAccessorsWgsl } from './reservoirDi.wgsl.js';

export const SPATIAL_WGSL = /* wgsl */ `

@group(0) @binding(5) var<storage, read_write> currentReservoir: array<u32>;
@group(0) @binding(7) var<storage, read_write> spatialReservoir: array<u32>;

${reservoirDiAccessorsWgsl({
  loadReadWriteBinding: 'currentReservoir',
  storeReadWriteBinding: 'spatialReservoir',
})}

@group(2) @binding(0) var<uniform> ubo: WalkaroundUBO;

const NEIGHBORS: u32 = 5u;
const REUSE_TECHNIQUES: u32 = NEIGHBORS + 1u;
const M_SCALE: u32 = 4u;

// Pipeline-specialized so otherwise-identical rounds cannot reuse the same
// neighbor rotation or WRS random stream.
override SPATIAL_ROUND_INDEX: u32 = 0u;

fn poissonDisk(i: u32, rotation: f32) -> vec2f {
  var offsets = array<vec2f, 8>(
    vec2f( 0.0,    1.0),
    vec2f( 0.866,  0.5),
    vec2f( 0.866, -0.5),
    vec2f( 0.0,   -1.0),
    vec2f(-0.866, -0.5),
    vec2f(-0.866,  0.5),
    vec2f( 0.354,  0.354),
    vec2f(-0.354, -0.354),
  );
  let offset = offsets[i % 8u];
  let sine = sin(rotation);
  let cosine = cos(rotation);
  return vec2f(
    offset.x * cosine - offset.y * sine,
    offset.x * sine + offset.y * cosine,
  );
}

fn spatialPixelFromInvocation(gid: vec3u) -> vec2u {
  if (ubo.checkerboardOn == 1u && restirReservoirScaleValue() == 1u) {
    let startColumn = (gid.y + ubo.frameParity) & 1u;
    return vec2u(gid.x * 2u + startColumn, gid.y);
  }
  return gid.xy;
}

@compute @workgroup_size(8, 8, 1)
fn spatialMain(@builtin(global_invocation_id) gid: vec3u) {
  let dims = ubo.screenSize;
  let reservoirDims = restirDiDimensions();
  let pixel = spatialPixelFromInvocation(gid);
  if (any(pixel >= reservoirDims)) { return; }

  let pixelIndex = pixel.y * reservoirDims.x + pixel.x;
  var centerReservoir = loadReservoirDI_rw(pixelIndex);
  scaleReservoirDIToM(&centerReservoir, M_SCALE);

  if (restirReservoirScaleValue() > 1u) {
    let roundSalt = SPATIAL_ROUND_INDEX * 0x9e3779b9u;
    var coarseRng = pcgInit(
      pixel.x ^ 54321u ^ roundSalt,
      pixel.y ^ 98765u ^ (roundSalt >> 1u),
      ubo.frameSeed ^ 0xCAFEu ^ (roundSalt * 0x85ebca6bu),
    );
    let rotation =
      rand_f32(&coarseRng) * 6.28318530718 +
      f32(SPATIAL_ROUND_INDEX) * 2.39996322973;
    var output = emptyReservoirDI();
    var coarseWrs = representedWrsInit();
    updateReservoirDI(
      &output,
      &coarseWrs,
      centerReservoir.lightId,
      centerReservoir.xi,
      reservoirDiCoarseReuseLogWeight(centerReservoir),
      &coarseRng,
    );
    var areaSupport = centerReservoir.areaM;
    var environmentSupport = centerReservoir.envM;
    var representedAttempts = centerReservoir.M;
    let coarseRadius = max(
      1.0,
      ubo.spatialReuseRadiusPx / f32(restirReservoirScaleValue()),
    );
    for (var neighborIndex = 0u; neighborIndex < NEIGHBORS; neighborIndex++) {
      let offset = poissonDisk(neighborIndex, rotation);
      let neighborPixel = vec2i(pixel) + vec2i(offset * coarseRadius);
      if (
        any(neighborPixel < vec2i(0))
        || any(neighborPixel >= vec2i(reservoirDims))
      ) {
        continue;
      }
      let neighborPixelIndex =
        u32(neighborPixel.y) * reservoirDims.x + u32(neighborPixel.x);
      var neighbor = loadReservoirDI_rw(neighborPixelIndex);
      let targetM = select(
        0u,
        max(1u, neighbor.M / M_SCALE),
        neighbor.M > 0u,
      );
      scaleReservoirDIToM(&neighbor, targetM);
      updateReservoirDI(
        &output,
        &coarseWrs,
        neighbor.lightId,
        neighbor.xi,
        reservoirDiCoarseReuseLogWeight(neighbor),
        &coarseRng,
      );
      areaSupport = reservoirDiSaturatingAddU32(
        areaSupport,
        neighbor.areaM,
      );
      environmentSupport = reservoirDiSaturatingAddU32(
        environmentSupport,
        neighbor.envM,
      );
      representedAttempts = reservoirDiSaturatingAddU32(
        representedAttempts,
        neighbor.M,
      );
    }
    output.areaM = areaSupport;
    output.envM = environmentSupport;
    output.M = representedAttempts;
    if (coarseWrs.hasSelection) {
      let pHat = restir_di_coarse_proposal_phat(
        output.lightId,
        output.xi,
      );
      finaliseReservoirDIFromNativeWrs(&output, coarseWrs, pHat);
    }
    storeReservoirDI_rw(pixelIndex, output);
    return;
  }

  let vp = ubo.projMatrix * ubo.viewMatrix;
  let invVP = invertMat4_common(vp);
  let centerSurface = castPrimary(
    restirDiFullPixel(pixel),
    dims,
    ubo.cameraPos,
    invVP,
  );
  if (!centerSurface.hit) {
    storeReservoirDI_rw(pixelIndex, centerReservoir);
    return;
  }

  var reservoirs: array<ReservoirDI, 6>;
  var surfaces: array<PrimarySurface, 6>;
  reservoirs[0u] = centerReservoir;
  surfaces[0u] = centerSurface;
  var techniqueCount = 1u;

  let roundSalt = SPATIAL_ROUND_INDEX * 0x9e3779b9u;
  var rng = pcgInit(
    pixel.x ^ 54321u ^ roundSalt,
    pixel.y ^ 98765u ^ (roundSalt >> 1u),
    ubo.frameSeed ^ 0xCAFEu ^ (roundSalt * 0x85ebca6bu),
  );
  let rotation =
    rand_f32(&rng) * 6.28318530718 +
    f32(SPATIAL_ROUND_INDEX) * 2.39996322973;

  for (var neighborIndex = 0u; neighborIndex < NEIGHBORS; neighborIndex++) {
    let offset = poissonDisk(neighborIndex, rotation);
    let neighborPixel = vec2i(pixel) + vec2i(
      vec2f(
        offset.x * ubo.spatialReuseRadiusPx,
        offset.y * ubo.spatialReuseRadiusPx,
      ),
    );
    if (any(neighborPixel < vec2i(0)) || any(neighborPixel >= vec2i(dims))) {
      continue;
    }

    let neighborSurface = castPrimary(
      vec2u(neighborPixel),
      dims,
      ubo.cameraPos,
      invVP,
    );
    if (!neighborSurface.hit) { continue; }
    let depthDifference = abs(centerSurface.depth - neighborSurface.depth);
    let depthTolerance = max(
      ubo.spatialDepthTolFloor,
      0.10 * centerSurface.depth,
    );
    if (
      depthDifference > depthTolerance ||
      dot(centerSurface.normal, neighborSurface.normal) < 0.9
    ) {
      continue;
    }

    let neighborPixelIndex =
      u32(neighborPixel.y) * dims.x + u32(neighborPixel.x);
    var neighborReservoir = loadReservoirDI_rw(neighborPixelIndex);
    let targetM = select(
      0u,
      max(1u, neighborReservoir.M / M_SCALE),
      neighborReservoir.M > 0u,
    );
    scaleReservoirDIToM(&neighborReservoir, targetM);
    reservoirs[techniqueCount] = neighborReservoir;
    surfaces[techniqueCount] = neighborSurface;
    techniqueCount++;
  }

  var output = emptyReservoirDI();
  var areaSupport = 0u;
  var environmentSupport = 0u;
  var representedAttempts = 0u;
  var wrs = representedWrsInit();

  for (var sourceIndex = 0u; sourceIndex < techniqueCount; sourceIndex++) {
    let source = reservoirs[sourceIndex];
    areaSupport = reservoirDiSaturatingAddU32(areaSupport, source.areaM);
    environmentSupport =
      reservoirDiSaturatingAddU32(environmentSupport, source.envM);
    representedAttempts =
      reservoirDiSaturatingAddU32(representedAttempts, source.M);

    let sourceSupport =
      reservoirDiSupportForLight(source, source.lightId);
    if (sourceSupport == 0u || !reservoirDiHasEstimatorNumerator(source)) {
      continue;
    }

    let sourceDensity = restir_di_compute_phat_xi(
      source.lightId,
      source.xi,
      surfaces[sourceIndex],
    );
    var techniqueLogDensities: array<f32, 6>;
    var maxLogDensity = RESERVOIR_DI_INVALID_LOG_DENSITY;
    for (var domainIndex = 0u; domainIndex < techniqueCount; domainIndex++) {
      let domainSupport = reservoirDiSupportForLight(
        reservoirs[domainIndex],
        source.lightId,
      );
      var domainDensity = 0.0;
      if (domainSupport > 0u) {
        domainDensity = restir_di_compute_phat_xi(
          source.lightId,
          source.xi,
          surfaces[domainIndex],
        );
      }
      let logDensity = reservoirDiLogWeightedDensity(
        domainSupport,
        domainDensity,
      );
      techniqueLogDensities[domainIndex] = logDensity;
      maxLogDensity = max(maxLogDensity, logDensity);
    }
    var scaledTechniqueDenominator = 0.0;
    for (var domainIndex = 0u; domainIndex < techniqueCount; domainIndex++) {
      scaledTechniqueDenominator += reservoirDiScaledDensityFromLog(
        techniqueLogDensities[domainIndex],
        maxLogDensity,
      );
    }
    let logTechniqueDenominator = reservoirDiLogSumExpFromMaxScale(
      maxLogDensity,
      scaledTechniqueDenominator,
    );
    let canonicalDensity = restir_di_compute_phat_xi(
      source.lightId,
      source.xi,
      centerSurface,
    );
    let candidateLogWeight = reservoirDiGeneralizedReuseLogWeight(
      reservoirDiLogWeightedDensity(sourceSupport, sourceDensity),
      logTechniqueDenominator,
      canonicalDensity,
      sourceDensity,
      source.logEstimatorNumerator,
    );
    updateReservoirDI(
      &output,
      &wrs,
      source.lightId,
      source.xi,
      candidateLogWeight,
      &rng,
    );
  }

  output.areaM = areaSupport;
  output.envM = environmentSupport;
  output.M = representedAttempts;
  var selectedCanonicalDensity = 0.0;
  if (wrs.hasSelection) {
    selectedCanonicalDensity = restir_di_compute_phat_xi(
      output.lightId,
      output.xi,
      centerSurface,
    );
  }
  finaliseReservoirDIFromGeneralizedReuse(
    &output,
    wrs,
    selectedCanonicalDensity,
  );
  storeReservoirDI_rw(pixelIndex, output);
}
`;

export const SPATIAL_MODULE: WgslModule = {
  name: 'spatial',
  source: SPATIAL_WGSL,
  requires: ['restirPHat', 'restirCastPrimary'],
};
