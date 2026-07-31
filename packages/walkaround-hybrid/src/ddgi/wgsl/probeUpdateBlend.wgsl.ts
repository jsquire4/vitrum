/**
 * DDGI Probe Update — Pass 2: Octahedral Atlas Blend.
 *
 * Two separate WGSL modules:
 *   makeProbeUpdateBlendIrrWGSL() — irradiance L2-SH blend (rgba16float, 3×3/probe)
 *   makeProbeUpdateBlendVisWGSL() — visibility atlas blend (rgba16float, 16×16/probe, rg used)
 *
 * Bindings are independent so the two pipelines don't share layouts.
 *
 * The cell sizes (IRR_CELL / VIS_CELL) are template-substituted from
 * `ddgiAtlasLayout.ts` — the single source of truth shared with the
 * producer (probeGrid.allocateAtlases) and the samplers
 * (ddgiSampleWgsl.ts + engines/restir/shaders/shade.wgsl.ts). The blend
 * shaders previously hardcoded `IRR_CELL=8u` / `VIS_CELL=16u` as WGSL
 * literals, which risked silent drift if the layout ever changed.
 *
 */

import { OCTAHEDRAL_WGSL } from '@vitrum/shared-bvh';
import { RAYS_PER_PROBE } from '../ddgiConstants.js';
import { IRR_CELL, VIS_CELL } from '../ddgiAtlasLayout.js';
import { DDGI_PROBE_MIN_HIT_DISTANCE_NORMALIZED } from '../probeState.js';
import { DDGI_SH_WGSL } from './ddgiSH.wgsl.js';

// Common header shared by both shaders. IRR_CELL / VIS_CELL are interpolated
// from ddgiAtlasLayout.ts so the blend pass cannot drift from the producer.
function makeCommonHeader(): string {
  return /* wgsl */`

${OCTAHEDRAL_WGSL}

const RAYS_PER_PROBE: u32 = ${RAYS_PER_PROBE}u;
const IRR_CELL:       u32 = ${IRR_CELL}u;
const VIS_CELL:       u32 = ${VIS_CELL}u;
const DDGI_MISS_DISTANCE: f32 = 1.0e19;
const DDGI_VISIBILITY_MAX_MOMENT_DISTANCE: f32 = 255.0;
const DDGI_PROBE_MIN_HIT_DISTANCE_NORMALIZED: f32 = ${DDGI_PROBE_MIN_HIT_DISTANCE_NORMALIZED};

struct ProbeRay {
  hitRadiance: vec3f,
  hitDistance: f32,
  direction:   vec3f,
  _pad0:       f32,
}

struct ProbeGridParams {
  origin:              vec3f,
  spacing:             f32,
  dims:                vec3u,
  _pad0:               u32,
  irradianceAtlasW:    f32,
  irradianceAtlasH:    f32,
  visibilityAtlasW:    f32,
  visibilityAtlasH:    f32,
}

struct FrameBlendParams {
  hysteresis: f32,
}

// @group(0) @binding(0-3) — shared layout

@group(0) @binding(0) var<storage, read> rayResults:   array<ProbeRay>;
@group(0) @binding(1) var<storage, read> activeProbes: array<u32>;
@group(0) @binding(2) var<uniform>       gridParams:   ProbeGridParams;
@group(0) @binding(3) var<uniform>       blendParams:  FrameBlendParams;

`;
}

// -----------------------------------------------------------------
// Irradiance blend shader
// -----------------------------------------------------------------
/** Build the DDGI irradiance-atlas blend WGSL with IRR_CELL substituted
 *  from {@link ddgiAtlasLayout}. */
export function makeProbeUpdateBlendIrrWGSL(): string {
  return /* wgsl */`

${makeCommonHeader()}

${DDGI_SH_WGSL}

@group(1) @binding(0) var irrPrev:   texture_2d<f32>;
@group(1) @binding(1) var irrSamp:   sampler;
@group(1) @binding(2) var irrOut:    texture_storage_2d<rgba16float, write>;

fn irrAtlasCoord(probeIdx: u32, pixel: vec2u) -> vec2u {
  let STRIDE = IRR_CELL + 2u;
  let px  = probeIdx % gridParams.dims.x;
  let tmp = probeIdx / gridParams.dims.x;
  let py  = tmp % gridParams.dims.y;
  let pz  = tmp / gridParams.dims.y;
  return vec2u(
    px * STRIDE + 1u + pixel.x,
    (py + pz * gridParams.dims.y) * STRIDE + 1u + pixel.y,
  );
}

// L2 SH irradiance blend (seam-free, replaces the octahedral cosine-mean atlas).
// Each probe's 9 RGB coefficients are projected from its RAYS_PER_PROBE rays and
// written into the first 3x3 interior texels (coeff k at (k%3, k/3)). Only those
// 9 threads do work; the rest of the 8x8 workgroup early-outs. The octahedral
// dir/octDecode is gone — SH needs the raw ray direction, not a per-texel bin.
@compute @workgroup_size(${IRR_CELL}, ${IRR_CELL}, 1)
fn probeUpdateBlendIrradiance(
  @builtin(global_invocation_id) gid: vec3u,
) {
  // One workgroup per probe; workgroup size == IRR_CELL so gid.x/IRR_CELL is the
  // probe index and (gid.x%IRR_CELL, gid.y%IRR_CELL) is the cell-local texel.
  // With IRR_CELL=3 the 3x3 workgroup maps exactly onto the 9 SH coeff texels.
  let lx       = gid.x % IRR_CELL;
  let ly       = gid.y % IRR_CELL;
  let groupIdx = gid.x / IRR_CELL;
  if (groupIdx >= arrayLength(&activeProbes)) { return; }
  let probeIdx = activeProbes[groupIdx];
  let totalProbes = gridParams.dims.x * gridParams.dims.y * gridParams.dims.z;
  if (probeIdx >= totalProbes) { return; }

  // Only the first 3x3 interior texels carry the 9 SH coefficients.
  if (lx >= 3u || ly >= 3u) { return; }
  let k = ly * 3u + lx;   // SH coefficient index 0..8
  let atlasCoord = irrAtlasCoord(probeIdx, vec2u(lx, ly));
  let baseIdx = probeIdx * RAYS_PER_PROBE;
  let numRays = arrayLength(&rayResults);
  if (baseIdx >= numRays) { return; }
  if (rayResults[baseIdx]._pad0 < 0.5) {
    // Inactive probes publish conservative zero, bypassing temporal history.
    textureStore(irrOut, atlasCoord, vec4f(0.0, 0.0, 0.0, 1.0));
    return;
  }

  // Project incoming radiance onto SH coeff k. Backface and self-intersection
  // records are invalid samples for a relocated probe and are excluded from
  // both numerator and denominator. This prevents an embedded probe from being
  // dimmed merely because many of its rays were rejected.
  var accum = vec3f(0.0);
  var validRayCount = 0u;
  let minHitDistance =
    gridParams.spacing * DDGI_PROBE_MIN_HIT_DISTANCE_NORMALIZED;
  for (var r = 0u; r < RAYS_PER_PROBE; r = r + 1u) {
    let rIdx = baseIdx + r;
    if (rIdx >= numRays) { break; }
    let ray = rayResults[rIdx];
    if (ray.hitDistance < minHitDistance) { continue; }
    validRayCount = validRayCount + 1u;
    let Y = ddgiShBasis(ray.direction);
    accum = accum + ray.hitRadiance * Y[k];
  }
  // Store the COSINE-CONVOLVED coefficient E_lm = A_l * c_k so the receiver eval
  // (sum_k E_lm * Y_k(n)) yields irradiance E directly. 4PI = 12.56637061436.
  let coeff = select(
    vec3f(0.0),
    accum * (12.56637061436 / f32(max(validRayCount, 1u))) * ddgiShCosineA(k),
    validRayCount > 0u,
  );

  // EMA read at the EXACT texel centre (bilinear collapses to the exact coeff)
  // so the sampler binding stays USED and the layout:"auto" blend pipeline does
  // not prune it (a pruned sampler desyncs the host bind group).
  let iUv = (vec2f(atlasCoord) + vec2f(0.5)) /
            vec2f(gridParams.irradianceAtlasW, gridParams.irradianceAtlasH);
  let prev = textureSampleLevel(irrPrev, irrSamp, iUv, 0.0).rgb;
  // An all-invalid update must not retain stale active-probe radiance.
  let hysteresis = select(0.0, blendParams.hysteresis, validRayCount > 0u);
  let blended = mix(coeff, prev, hysteresis);
  textureStore(irrOut, atlasCoord, vec4f(blended, 1.0));
}

`;
}

// -----------------------------------------------------------------
// Visibility blend shader
// -----------------------------------------------------------------
/** Build the DDGI visibility-atlas blend WGSL with VIS_CELL substituted
 *  from {@link ddgiAtlasLayout}. */
export function makeProbeUpdateBlendVisWGSL(): string {
  return /* wgsl */`

${makeCommonHeader()}

@group(1) @binding(0) var visPrev:   texture_2d<f32>;
@group(1) @binding(1) var visSamp:   sampler;
// rg16float is not writable as a storage texture in WebGPU; use rgba16float.
@group(1) @binding(2) var visOut:    texture_storage_2d<rgba16float, write>;

fn visAtlasCoord(probeIdx: u32, pixel: vec2u) -> vec2u {
  let STRIDE = VIS_CELL + 2u;
  let px  = probeIdx % gridParams.dims.x;
  let tmp = probeIdx / gridParams.dims.x;
  let py  = tmp % gridParams.dims.y;
  let pz  = tmp / gridParams.dims.y;
  return vec2u(
    px * STRIDE + 1u + pixel.x,
    (py + pz * gridParams.dims.y) * STRIDE + 1u + pixel.y,
  );
}

@compute @workgroup_size(16, 16, 1)
fn probeUpdateBlendVisibility(
  @builtin(global_invocation_id) gid: vec3u,
) {
  let pixel    = vec2u(gid.x % VIS_CELL, gid.y % VIS_CELL);
  let groupIdx = gid.x / VIS_CELL;
  if (groupIdx >= arrayLength(&activeProbes)) { return; }
  let probeIdx = activeProbes[groupIdx];
  let totalProbes = gridParams.dims.x * gridParams.dims.y * gridParams.dims.z;
  if (probeIdx >= totalProbes) { return; }

  let octUv = (vec2f(pixel) + vec2f(0.5)) / vec2f(f32(VIS_CELL));
  let dir   = octDecode(octUv * 2.0 - 1.0);
  let atlasCoord = visAtlasCoord(probeIdx, pixel);
  let baseIdx = probeIdx * RAYS_PER_PROBE;
  let numRays = arrayLength(&rayResults);
  if (baseIdx >= numRays) { return; }
  if (rayResults[baseIdx]._pad0 < 0.5) {
    // Inactive/embedded probes are conservatively occluded. Never publish the
    // historical far-open sentinel for a probe with no valid geometry sample.
    textureStore(visOut, atlasCoord, vec4f(0.0, 0.0, 0.0, 1.0));
    return;
  }

  var newDepth   = 0.0;
  var newDepthSq = 0.0;
  var totalWeight = 0.0;
  var validRayCount = 0u;
  // Receivers only interpolate the eight probes surrounding their cell. A
  // finite distance comfortably beyond that neighbourhood represents an open
  // ray without overflowing rgba16float's second-moment lane.
  let missDepth = min(
    DDGI_VISIBILITY_MAX_MOMENT_DISTANCE,
    gridParams.spacing * 16.0,
  );
  let minHitDistance =
    gridParams.spacing * DDGI_PROBE_MIN_HIT_DISTANCE_NORMALIZED;
  for (var r = 0u; r < RAYS_PER_PROBE; r = r + 1u) {
    let rIdx = baseIdx + r;
    if (rIdx >= numRays) { break; }
    let ray = rayResults[rIdx];
    // Backface hits encode negative distance (DDGI paper convention). Skip them in
    // the visibility depth estimate — they represent hits from inside geometry
    // and should not influence mean/depth² (they'd incorrectly tighten the
    // Chebyshev test and cause light leaking through back-faces).
    if (ray.hitDistance < 0.0) { continue; }
    // Self-intersection filter — match the irradiance blend pass. Probes
    // inside opaque meshes record distance near zero in every direction,
    // which pegs the mean and variance in the visibility atlas to ~0.
    // The shade-side ddgiSample uses these for Chebyshev visibility;
    // a 0-mean/0-variance entry propagates corrupt depth into trilinear
    // neighbours (the bordered atlas samples in zero corners of an
    // inside-box probe cell). Skip these the same way as the irradiance pass.
    if (ray.hitDistance < minHitDistance) { continue; }
    // Sky misses are valid open-direction observations. Excluding them
    // conditions mixed hit/miss angular cells only on blockers and
    // over-occludes silhouettes. Substitute the finite far-open depth above
    // so both the first and second moments remain representable in f16.
    let d = select(ray.hitDistance, missDepth, ray.hitDistance >= DDGI_MISS_DISTANCE);
    validRayCount = validRayCount + 1u;
    let w = max(0.0, dot(dir, ray.direction));
    if (w <= 0.0) { continue; }
    // Variance-shadow visibility kernel — Majercik 2019 §3 uses pow(2) for the
    // depth/depth² accumulation (Chebyshev shadow visibility). pow(50) was too
    // narrow for a 192-ray budget (most atlas pixels had zero aligned rays).
    let weight  = pow(w, 2.0);
    newDepth   = newDepth + d * weight;
    newDepthSq = newDepthSq + d * d * weight;
    totalWeight = totalWeight + weight;
  }
  if (totalWeight > 0.0) {
    newDepth   = newDepth / totalWeight;
    newDepthSq = newDepthSq / totalWeight;
  } else if (validRayCount > 0u) {
    // The probe has valid observations, but none align with this octahedral
    // cell. Preserve a finite open-direction moment pair.
    newDepth = missDepth;
    newDepthSq = missDepth * missDepth;
  } else {
    // Every ray was invalid/miss/backface. Conservative zero avoids turning an
    // embedded or unclassified probe into a permanently open visibility cell.
    newDepth = 0.0;
    newDepthSq = 0.0;
  }

  let vUv = (vec2f(atlasCoord) + vec2f(0.5)) /
            vec2f(gridParams.visibilityAtlasW, gridParams.visibilityAtlasH);
  let prev    = textureSampleLevel(visPrev, visSamp, vUv, 0.0).rg;
  let hysteresis = select(0.0, blendParams.hysteresis, validRayCount > 0u);
  let blended = mix(vec2f(newDepth, newDepthSq), prev, hysteresis);
  textureStore(visOut, atlasCoord, vec4f(blended, 0.0, 1.0));
}

`;
}
