/**
 * DDGI probe classification + relocation compute pass.
 *
 * Original WebGPU adaptation of:
 * - Majercik et al., "Scaling Probe-Based Real-Time Dynamic Global
 *   Illumination for Production", JCGT 2021.
 * - NVIDIA RTXGI `ProbeClassificationCS.hlsl` and `ProbeRelocationCS.hlsl`:
 *   https://github.com/NVIDIAGameWorks/RTXGI-DDGI/tree/main/rtxgi-sdk/shaders/ddgi
 *
 * The current stratum's ray results and the prior irradiance atlas's reserved
 * rgba16float ring texel produce the next atlas state. Offsets are
 * hard-clamped to a 0.45-cell sphere and per-frame movement is capped at 0.20
 * cell, so malformed geometry cannot move a probe outside its interpolation
 * neighbourhood.
 */

import { RAYS_PER_PROBE } from '../ddgiConstants.js';
import {
  IRR_PROBE_STATE_LOCAL_X,
  IRR_PROBE_STATE_LOCAL_Y,
  IRR_STRIDE,
} from '../ddgiAtlasLayout.js';
import {
  DDGI_PROBE_BACKFACE_THRESHOLD,
  DDGI_PROBE_MAX_OFFSET_NORMALIZED,
  DDGI_PROBE_MAX_RELOCATION_STEP_NORMALIZED,
  DDGI_PROBE_MIN_HIT_DISTANCE,
} from '../probeState.js';

export const PROBE_CLASSIFY_RELOCATE_WGSL = /* wgsl */`
const RAYS_PER_PROBE: u32 = ${RAYS_PER_PROBE}u;
const BACKFACE_THRESHOLD: f32 = ${DDGI_PROBE_BACKFACE_THRESHOLD};
const MAX_OFFSET_NORMALIZED: f32 = ${DDGI_PROBE_MAX_OFFSET_NORMALIZED};
const MAX_RELOCATION_STEP_NORMALIZED: f32 = ${DDGI_PROBE_MAX_RELOCATION_STEP_NORMALIZED};
const MIN_HIT_DISTANCE: f32 = ${DDGI_PROBE_MIN_HIT_DISTANCE};
const IRRADIANCE_STRIDE: u32 = ${IRR_STRIDE}u;
const STATE_LOCAL_X: u32 = ${IRR_PROBE_STATE_LOCAL_X}u;
const STATE_LOCAL_Y: u32 = ${IRR_PROBE_STATE_LOCAL_Y}u;
// Keep the exponent form in WGSL. Interpolating the JS number emits an
// unrepresentable abstract-integer token (10000000000000000000).
const MISS_DISTANCE: f32 = 1.0e19;

struct ProbeRay {
  hitRadiance: vec3f,
  hitDistance: f32,
  direction: vec3f,
  _pad0: f32,
}

struct ProbeGridParams {
  origin: vec3f,
  spacing: f32,
  dims: vec3u,
  _pad0: u32,
  irradianceAtlasW: f32,
  irradianceAtlasH: f32,
  visibilityAtlasW: f32,
  visibilityAtlasH: f32,
}

@group(0) @binding(0) var<storage, read_write> rayResults: array<ProbeRay>;
@group(0) @binding(1) var<storage, read> activeProbes: array<u32>;
@group(0) @binding(2) var<uniform> gridParams: ProbeGridParams;
@group(0) @binding(3) var irradiancePrev: texture_2d<f32>;
@group(0) @binding(4) var irradianceOut: texture_storage_2d<rgba16float, write>;

fn stateCoord(probeIdx: u32) -> vec2u {
  let x = probeIdx % gridParams.dims.x;
  let yz = probeIdx / gridParams.dims.x;
  return vec2u(
    x * IRRADIANCE_STRIDE + STATE_LOCAL_X,
    yz * IRRADIANCE_STRIDE + STATE_LOCAL_Y,
  );
}

fn safeDirection(direction: vec3f) -> vec3f {
  let len2 = dot(direction, direction);
  if (!(len2 > 1.0e-12) || !(len2 < 1.0e20)) { return vec3f(0.0); }
  return direction * inverseSqrt(len2);
}

fn clampOffset(offset: vec3f) -> vec3f {
  let maxLength = MAX_OFFSET_NORMALIZED * gridParams.spacing;
  let guardedMaxLength =
    max(0.0, maxLength - gridParams.spacing * 1.0e-6);
  let len2 = dot(offset, offset);
  if (!(len2 >= 0.0) || !(len2 < 1.0e20)) { return vec3f(0.0); }
  if (len2 <= maxLength * maxLength) { return offset; }
  return offset * (guardedMaxLength * inverseSqrt(max(len2, 1.0e-12)));
}

@compute @workgroup_size(64, 1, 1)
fn probeClassifyRelocate(@builtin(global_invocation_id) gid: vec3u) {
  let activeIdx = gid.x;
  if (activeIdx >= arrayLength(&activeProbes)) { return; }
  let probeIdx = activeProbes[activeIdx];
  let totalProbes = gridParams.dims.x * gridParams.dims.y * gridParams.dims.z;
  if (probeIdx >= totalProbes) { return; }

  let coord = stateCoord(probeIdx);
  let previous = textureLoad(irradiancePrev, vec2i(coord), 0);
  // The packed rgba16float texel stores scale-independent normalized offsets.
  let currentOffset = clampOffset(previous.xyz * gridParams.spacing);
  let minFrontDistance = max(MIN_HIT_DISTANCE, gridParams.spacing * 0.05);
  let maxStep = gridParams.spacing * MAX_RELOCATION_STEP_NORMALIZED;

  var closestBackfaceDistance = 1.0e20;
  var closestBackfaceDirection = vec3f(0.0);
  var closestFrontfaceDistance = 1.0e20;
  var closestFrontfaceDirection = vec3f(0.0);
  var farthestFrontfaceDistance = 0.0;
  var farthestFrontfaceDirection = vec3f(0.0);
  var backfaceCount = 0u;
  var inspectedRayCount = 0u;
  var nearbyFrontface = false;

  let baseIdx = probeIdx * RAYS_PER_PROBE;
  let rayCount = arrayLength(&rayResults);
  if (baseIdx >= rayCount) { return; }
  for (var rayIdx = 0u; rayIdx < RAYS_PER_PROBE; rayIdx = rayIdx + 1u) {
    let resultIdx = baseIdx + rayIdx;
    if (resultIdx >= rayCount) { break; }
    let ray = rayResults[resultIdx];
    let direction = safeDirection(ray.direction);
    if (dot(direction, direction) == 0.0 || !(abs(ray.hitDistance) < 3.0e38)) {
      continue;
    }
    inspectedRayCount = inspectedRayCount + 1u;

    if (ray.hitDistance < 0.0) {
      backfaceCount = backfaceCount + 1u;
      let distance = abs(ray.hitDistance);
      if (distance < closestBackfaceDistance) {
        closestBackfaceDistance = distance;
        closestBackfaceDirection = direction;
      }
      continue;
    }
    // Every finite nonnegative ray participates in relocation: a very close
    // front hit identifies the wall and a miss identifies open space.
    if (ray.hitDistance < closestFrontfaceDistance) {
      closestFrontfaceDistance = ray.hitDistance;
      closestFrontfaceDirection = direction;
    }
    if (ray.hitDistance > farthestFrontfaceDistance) {
      farthestFrontfaceDistance = ray.hitDistance;
      farthestFrontfaceDirection = direction;
    }

    let maxAxis = max(abs(direction.x), max(abs(direction.y), abs(direction.z)));
    let voxelPlaneDistance = gridParams.spacing / max(maxAxis, 1.0e-6);
    if (
      ray.hitDistance < MISS_DISTANCE &&
      ray.hitDistance <= voxelPlaneDistance
    ) {
      nearbyFrontface = true;
    }
  }

  let backfaceFraction = select(
    1.0,
    f32(backfaceCount) / f32(max(inspectedRayCount, 1u)),
    inspectedRayCount > 0u,
  );
  var candidate = currentOffset;

  if (closestBackfaceDistance < 1.0e20 && backfaceFraction > BACKFACE_THRESHOLD) {
    let step = min(closestBackfaceDistance + minFrontDistance * 0.5, maxStep);
    candidate = currentOffset + closestBackfaceDirection * step;
  } else if (
    closestFrontfaceDistance < minFrontDistance &&
    farthestFrontfaceDistance > 0.0 &&
    dot(closestFrontfaceDirection, farthestFrontfaceDirection) <= 0.0
  ) {
    candidate =
      currentOffset +
      farthestFrontfaceDirection * min(farthestFrontfaceDistance, maxStep);
  } else {
    let offsetLength = length(currentOffset);
    if (offsetLength > 0.0) {
      let clearance = select(
        maxStep,
        max(0.0, closestFrontfaceDistance - minFrontDistance),
        closestFrontfaceDistance < 1.0e20,
      );
      let moveBack = min(min(clearance, offsetLength), maxStep);
      candidate = currentOffset - (currentOffset / offsetLength) * moveBack;
    }
  }

  let probeActive =
    inspectedRayCount > 0u &&
    backfaceFraction <= BACKFACE_THRESHOLD &&
    nearbyFrontface;
  let activeLane = select(0.0, 1.0, probeActive);
  // Blends process only this active stratum. Publish classification through the
  // existing ray record so they need no extra sampled/storage resource.
  rayResults[baseIdx]._pad0 = activeLane;
  textureStore(
    irradianceOut,
    coord,
    vec4f(clampOffset(candidate) / gridParams.spacing, activeLane),
  );
}
`;
