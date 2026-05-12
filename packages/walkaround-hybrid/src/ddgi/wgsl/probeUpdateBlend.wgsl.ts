/**
 * DDGI Probe Update — Pass 2: Octahedral Atlas Blend.
 *
 * Two separate WGSL modules:
 *   PROBE_UPDATE_BLEND_IRR_WGSL  — irradiance atlas blend (rgba16float, 8×8/probe)
 *   PROBE_UPDATE_BLEND_VIS_WGSL  — visibility atlas blend (rgba16float, 16×16/probe, rg used)
 *
 * Bindings are independent so the two pipelines don't share layouts.
 */

import { OCTAHEDRAL_WGSL } from '@vitrum/shared-bvh';
import { RAYS_PER_PROBE } from '../ddgiConstants.js';

// Common header shared by both shaders
const COMMON = /* wgsl */`

${OCTAHEDRAL_WGSL}

const RAYS_PER_PROBE: u32 = ${RAYS_PER_PROBE}u;
const HYSTERESIS:     f32 = 0.97;
const IRR_CELL:       u32 = 8u;
const VIS_CELL:       u32 = 16u;

struct ProbeRay {
  hitPosition:   vec3f,
  hitDistance:   f32,
  hitNormal:     vec3f,
  hitMaterialId: u32,
  hitRadiance:   vec3f,
  isGlass:       u32,
  direction:     vec3f,
  _pad0:         f32,
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
  probesPerFrame: u32,
  hysteresis:     f32,
  _pad0:          u32,
  _pad1:          u32,
}

// @group(0) @binding(0-3) — shared layout

@group(0) @binding(0) var<storage, read> rayResults:   array<ProbeRay>;
@group(0) @binding(1) var<storage, read> activeProbes: array<u32>;
@group(0) @binding(2) var<uniform>       gridParams:   ProbeGridParams;
@group(0) @binding(3) var<uniform>       blendParams:  FrameBlendParams;

`;

// -----------------------------------------------------------------
// Irradiance blend shader
// -----------------------------------------------------------------
export const PROBE_UPDATE_BLEND_IRR_WGSL = /* wgsl */`

${COMMON}

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

@compute @workgroup_size(8, 8, 1)
fn probeUpdateBlendIrradiance(
  @builtin(global_invocation_id) gid: vec3u,
) {
  // Each workgroup covers one probe's 8×8 irradiance map.
  // global x: column = probe_x_in_atlas * 8 + pixel_x
  //           group_x = probe_idx, local_x = pixel.x (via built-in subgroup)
  let pixel    = vec2u(gid.x % IRR_CELL, gid.y % IRR_CELL);
  let groupIdx = gid.x / IRR_CELL;
  if (groupIdx >= arrayLength(&activeProbes)) { return; }
  let probeIdx = activeProbes[groupIdx];
  let totalProbes = gridParams.dims.x * gridParams.dims.y * gridParams.dims.z;
  if (probeIdx >= totalProbes) { return; }

  let octUv = (vec2f(pixel) + vec2f(0.5)) / vec2f(f32(IRR_CELL));
  let dir   = octDecode(octUv * 2.0 - 1.0);

  var newColor    = vec3f(0.0);
  var totalWeight = 0.0;
  let baseIdx = probeIdx * RAYS_PER_PROBE;
  let numRays = arrayLength(&rayResults);
  for (var r = 0u; r < RAYS_PER_PROBE; r = r + 1u) {
    let rIdx = baseIdx + r;
    if (rIdx >= numRays) { break; }
    let ray = rayResults[rIdx];
    // Skip backface hits (encoded as negative distance per DDGI paper).
    if (ray.hitDistance < 0.0) { continue; }
    // Skip self-intersection hits — a probe positioned inside an opaque
    // mesh (e.g. one of the Cornell inner boxes) sees every outgoing ray
    // hit the surrounding inner-wall surface at distance ≈ 0, contaminating
    // the atlas with gray near-zero-light surfaces and washing out the
    // actual wall colour bleed from outside-geometry probes. When all
    // valid rays for this direction get filtered, totalWeight stays 0 →
    // newColor=0 → blended decays via hysteresis to 0, and the trilinear
    // interpolation in shade.wgsl correctly drops these probes from the
    // 8-probe stencil.
    if (ray.hitDistance < 0.05) { continue; }
    // Paper Lambertian cosine kernel — Majercik 2019 §3 Algorithm 1.
    // Weight = max(0, n·d) where n is the atlas-texel outgoing direction and
    // d is the probe ray direction. This is the standard cosine weight for
    // irradiance accumulation; summing ray contributions with this weight and
    // then dividing by totalWeight produces the correct irradiance E at each
    // atlas texel (after Change 3 moves albedo/π to the receiver side).
    // Per-frame SO(3) rotation (Change 1 / Item 6) decorrelates ray samples
    // across frames so the temporal EMA can average the higher per-frame
    // variance produced by a true cosine kernel vs the narrower pow(8) lobe.
    let w = max(0.0, dot(dir, ray.direction));
    if (w < 1e-3) { continue; }
    let weight = w;
    newColor    = newColor + ray.hitRadiance * weight;
    totalWeight = totalWeight + weight;
  }
  if (totalWeight > 1e-5) {
    newColor = newColor / totalWeight;
  }

  let atlasCoord = irrAtlasCoord(probeIdx, pixel);
  let iUv    = (vec2f(atlasCoord) + vec2f(0.5)) /
               vec2f(gridParams.irradianceAtlasW, gridParams.irradianceAtlasH);
  let prev   = textureSampleLevel(irrPrev, irrSamp, iUv, 0.0).rgb;
  let blended = mix(newColor, prev, blendParams.hysteresis);
  textureStore(irrOut, atlasCoord, vec4f(blended, 1.0));
}

`;

// -----------------------------------------------------------------
// Visibility blend shader
// -----------------------------------------------------------------
export const PROBE_UPDATE_BLEND_VIS_WGSL = /* wgsl */`

${COMMON}

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

  var newDepth   = 0.0;
  var newDepthSq = 0.0;
  var totalWeight = 0.0;
  let baseIdx = probeIdx * RAYS_PER_PROBE;
  let numRays = arrayLength(&rayResults);
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
    if (ray.hitDistance < 0.05) { continue; }
    let w = max(0.0, dot(dir, ray.direction));
    if (w < 1e-3) { continue; }
    // Variance-shadow visibility kernel — Majercik 2019 §3 uses pow(2) for the
    // depth/depth² accumulation (Chebyshev shadow visibility). pow(50) was too
    // narrow for a 192-ray budget (most atlas pixels had zero aligned rays).
    let weight  = pow(w, 2.0);
    let d       = ray.hitDistance;
    newDepth   = newDepth + d * weight;
    newDepthSq = newDepthSq + d * d * weight;
    totalWeight = totalWeight + weight;
  }
  if (totalWeight > 1e-5) {
    newDepth   = newDepth / totalWeight;
    newDepthSq = newDepthSq / totalWeight;
  }

  let atlasCoord = visAtlasCoord(probeIdx, pixel);
  let vUv = (vec2f(atlasCoord) + vec2f(0.5)) /
            vec2f(gridParams.visibilityAtlasW, gridParams.visibilityAtlasH);
  let prev    = textureSampleLevel(visPrev, visSamp, vUv, 0.0).rg;
  let blended = mix(vec2f(newDepth, newDepthSq), prev, blendParams.hysteresis);
  textureStore(visOut, atlasCoord, vec4f(blended, 0.0, 1.0));
}

`;
