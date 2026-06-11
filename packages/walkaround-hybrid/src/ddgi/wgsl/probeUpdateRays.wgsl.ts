/**
 * DDGI Probe Update — Pass 1: Ray Casting.
 *
 * One workgroup per probe in the active set. Each thread handles
 * ceil(RAYS_PER_PROBE / 32) rays (192 rays / 32 threads today). Writes ray hit results to a storage buffer for
 * Pass 2 (the atlas blend pass).
 *
 * The factory `makeProbeUpdateRaysWGSL(maxMaterials)` composes three
 * sub-template functions:
 *   makeTraceSunVisibilityWGSL()   — glass-aware sun-visibility helper
 *   makeDirectLightingWGSL()       — analytic lights + area-emitter NEE
 *   makeProbeMainEntryWGSL()       — probe world pos, sky/env sampling,
 *                                    main compute entry point
 */

import { HAMMERSLEY_WGSL } from '@vitrum/shared-samplers';
import {
  OCTAHEDRAL_WGSL,
  MATERIAL_ENTRY_WGSL,
  BVH_INTERSECT_WGSL,
  TLAS_TRAVERSAL_WGSL,
} from '@vitrum/shared-bvh';
import { RAYS_PER_PROBE } from '../ddgiConstants.js';
import { IRR_STRIDE } from '../ddgiAtlasLayout.js';
import { DDGI_SH_WGSL } from './ddgiSH.wgsl.js';

const WG_SIZE = 32;
const RAYS_PER_THREAD = Math.ceil(RAYS_PER_PROBE / WG_SIZE);

/**
 * Generate the probeUpdateRays WGSL shader with a compile-time material
 * array size. Injecting this as a template literal avoids exceeding
 * WebGPU's uniform array size limits when the caller has fewer than 64
 * materials — and allows scenes with more materials to raise the cap.
 *
 * M9 audit remediation: `DDGI_MAX_MATERIALS` was previously hardcoded as
 * `array<MaterialEntry, 64>` in the WGSL. Now driven by
 * `HybridEngineOptions.ddgiMaxMaterials` (default 64).
 *
 * @param maxMaterials Maximum number of distinct materials. Must be >= 1.
 *        The host-side materialsBuf must be at least
 *        `maxMaterials × DDGI_MATERIAL_STRIDE_BYTES` bytes.
 */
export function makeProbeUpdateRaysWGSL(maxMaterials: number): string {
  if (maxMaterials < 1) throw new RangeError(`makeProbeUpdateRaysWGSL: maxMaterials must be >= 1, got ${maxMaterials}`);
  return (
    _makeProbeUpdateRaysHeader(maxMaterials) +
    makeTraceSunVisibilityWGSL() +
    makeDirectLightingWGSL() +
    makeProbeMainEntryWGSL()
  );
}

// @deprecated `PROBE_UPDATE_RAYS_WGSL` (bound to a 64-ray default)
// removed 2026-05-18 dead-code sweep — supplanted by
// `makeProbeUpdateRaysWGSL(64)`; zero non-self consumers.

// ─── Sub-template functions ────────────────────────────────────────────────

/**
 * Shared header: includes, struct declarations, constants, and bindings.
 * (Not exported — composed internally by makeProbeUpdateRaysWGSL.)
 */
function _makeProbeUpdateRaysHeader(maxMaterials: number): string { return /* wgsl */`

${HAMMERSLEY_WGSL}
${OCTAHEDRAL_WGSL}
${MATERIAL_ENTRY_WGSL}
${BVH_INTERSECT_WGSL}
${TLAS_TRAVERSAL_WGSL}
${DDGI_SH_WGSL}

const WG_SIZE: u32       = ${WG_SIZE}u;
const RAYS_PER_PROBE: u32  = ${RAYS_PER_PROBE}u;
const RAYS_PER_THREAD: u32 = ${RAYS_PER_THREAD}u;   // RAYS_PER_PROBE / WG_SIZE
// NORMAL_BIAS is derived per-frame from gridParams.spacing * 0.001 (see
// evalSunLight / evalPointLight below). M13 audit: the fixed 0.02 was
// Cornell-specific (probe spacing ~0.17 units → 0.17×0.001 = 0.00017, well
// below 0.02). Large scenes with spacing >20 units need a proportionally
// larger bias; tiny scenes (spacing <0.02) would over-bias with 0.02.
// Removed as a compile-time constant; computed inline where needed.
//
// sweep-20260518/moller-trumbore-canonical: the local INFINITY (= 1e20) and
// BVH_STACK_DEPTH (= 60) constants previously lived here. The canonical
// module declares BVH_INTERSECT_INFINITY / BVH_INTERSECT_STACK_DEPTH with the
// same values; the one remaining INFINITY reference below (out.hitDistance
// for sky misses) now reads BVH_INTERSECT_INFINITY.
const PI: f32              = 3.14159265359;

// Probe-side glass-transmission perceptual scale lives on the probe-side
// FrameParams UBO (frameParams.glassMixScale); the canonical Cornell-tuned
// default 0.7 is written by ProbeUpdatePass._uploadFrameParams from the
// HybridEngine option glassMixScale. When a probe ray hits glass we mix
// room radiance with sky-tinted transmitted radiance, weighted by
// mat.transmission * frameParams.glassMixScale. At 0.7 fully-transparent
// glass leaves 30% of the room radiance in the probe's irradiance estimate
// — preventing bright sky from drowning indirect-bounce contribution.
//
// The mirror field also lives on WalkaroundUBO (ubo.glassMixScale) as the
// canonical single source of truth: probeUpdateRays binds FrameParams, not
// WalkaroundUBO, so the host packs the same scalar into both UBOs.

// -----------------------------------------------------------------
// BVH structs come from @vitrum/shared-bvh BVH_INTERSECT_WGSL (injected
// at the top of this file). Pre-canonical DDGI declared its own:
//   - BVHBoundingBox { min: array<f32,3>, max: array<f32,3> }
//   - BVHNode (nested-bounds form)
//   - Ray, IntersectionResult
// The canonical struct uses flat boundsMin/boundsMax fields; rename refs
// from node.bounds.min[i] → node.boundsMin[i] etc. The canonical
// IntersectionResult is a superset with extra (matColorPacked, uv) slots
// that this consumer ignores.
// -----------------------------------------------------------------

// -----------------------------------------------------------------
// DDGI material table — uses the canonical MaterialEntry struct
// declared by @vitrum/shared-bvh/wgsl/materialEntry.wgsl.ts
// (injected above via MATERIAL_ENTRY_WGSL).
//
// Pre-W2-C5 this file declared a local DDGIMaterial struct with a
// different field order (no attenuationDistance / thickness). The
// canonical struct carries both, which lets future revisions of
// traceSunVisibility apply full Beer-Lambert tint (today still uses
// the simplified attenColor * transmission blend below).
// -----------------------------------------------------------------

// -----------------------------------------------------------------
// Light uniforms
// -----------------------------------------------------------------
const LIGHT_SUN:   u32 = 0u;
const LIGHT_POINT: u32 = 1u;
const LIGHT_SPOT:  u32 = 2u;
const MAX_LIGHTS:  u32 = 16u;

struct DDGILight {
  kind:       u32,
  _pad0: f32, _pad1: f32, _pad2: f32,
  position:   vec3f,
  intensity:  f32,
  direction:  vec3f,
  innerCone:  f32,
  color:      vec3f,
  outerCone:  f32,
}

struct DDGILightUniforms {
  count: u32,
  _pad0: u32, _pad1: u32, _pad2: u32,
  items: array<DDGILight, 16>,
}

// -----------------------------------------------------------------
// Probe grid parameters
// -----------------------------------------------------------------
struct ProbeGridParams {
  origin:   vec3f,
  spacing:  f32,
  dims:     vec3u,
  _pad0:    u32,
  irradianceAtlasW: f32,
  irradianceAtlasH: f32,
  visibilityAtlasW: f32,
  visibilityAtlasH: f32,
}

struct FrameParams {
  randomRotation: vec3f,
  frameIndex:     u32,
  totalProbes:    u32,
  probesPerFrame: u32,
  _pad0: u32, _pad1: u32,
  // Sky appearance for miss rays (B2 audit: previously hardcoded gradient).
  // Written by ProbeUpdatePass.setSkyParams(); defaults match the original
  // Cornell-tuned values so existing behaviour is unchanged.
  skyTint:        vec3f,
  skyIrradiance:  f32,
  // 2026-05-18 sweep — glass-transmission perceptual mix scale.  Written
  // by ProbeUpdatePass._uploadFrameParams from HybridEngineOptions.glassMixScale.
  glassMixScale:  f32,
  // H46-A — DDGI indirect-feedback gate (maxBounces semantics for this regime).
  // 1u = the previous-frame irradiance-atlas read is folded into the bounce
  // surface's outgoing radiance (the infinite-bounce diffuse EMA — the default
  // maxBounces >= 2 behaviour). 0u = DIRECT-ONLY probes: the indirect term is
  // dropped so each probe carries one bounce of direct light only (maxBounces
  // == 1). Written by ProbeUpdatePass._uploadFrameParams from
  // HybridEngine._cfg.maxBounces. NOTE: this is NOT a path-tracer bounce cap —
  // it gates the diffuse multi-bounce feedback loop of the DDGI atlas. (Was the
  // inert _pad2 slot; byte size unchanged.)
  indirectFeedback: u32,
  // Wave 4 (2026-06-10) — HDRI into DDGI probe misses.
  // hasEnv = 1u  → sample ddgiEnvMap by ray direction on probe-ray miss.
  // hasEnv = 0u  → existing procedural gradient (default; byte-identical).
  // envRotationY + envIntensity match the H6 convention in environmentSample.wgsl:
  //   lookupDir = RY(-envRotationY) · worldDir
  //   result    = texel.rgb * envIntensity
  hasEnv:       u32,
  envRotationY: f32,
  envIntensity: f32,
}

// -----------------------------------------------------------------
// Ray result written by pass 1, read by pass 2
// -----------------------------------------------------------------
struct ProbeRay {
  hitPosition:  vec3f,
  hitDistance:  f32,
  hitNormal:    vec3f,
  hitMaterialId: u32,
  hitRadiance:  vec3f,
  isGlass:      u32,
  direction:    vec3f,
  _pad0:        f32,
}

// -----------------------------------------------------------------
// Bindings — vec4 storage matches ReSTIR / shared scenePack (PR-5.2).
// -----------------------------------------------------------------
struct DdgiTraceParams {
  bvhMode: u32,
  tlasNodeCount: u32,
  // H18 Stage 2 — number of valid emitter triangles in ddgiEmitterTris (0 = sun-only
  // scene; guard gates the NEE loop so sun-only scenes are byte-identical with pre-H18).
  emitterTriCount: u32,
  _pad1: u32,
}

@group(0) @binding(0)  var<storage, read> bvh:                      array<BVHNode>;
@group(0) @binding(1)  var<storage, read> bvh_position:             array<vec4f>;
@group(0) @binding(2)  var<storage, read> bvh_index:                array<vec4u>;
@group(0) @binding(3)  var<storage, read> bvh_normal:                array<vec4f>;
@group(0) @binding(4)  var<storage, read> bvh_materialId:           array<u32>;
@group(0) @binding(5)  var<storage, read> tlasNodes:                array<BVHNode>;
@group(0) @binding(6)  var<storage, read> tlasInstanceIndices:     array<u32>;
@group(0) @binding(7)  var<storage, read> tlasBlasRoots:            array<u32>;
@group(0) @binding(8)  var<storage, read> tlasInstanceWorldToLocal: array<vec4f>;
@group(0) @binding(9)  var<storage, read> tlasInstanceLocalToWorld: array<vec4f>;
@group(0) @binding(10) var<uniform>       ddgiTrace:                DdgiTraceParams;

@group(1) @binding(0) var<uniform> materials:     array<MaterialEntry, ${maxMaterials}>;
@group(1) @binding(1) var<uniform> lights:        DDGILightUniforms;
// H18 Stage 2 — packed area-emitter triangles for per-probe NEE (same layout as
// the RC probeRayCast rc_emitters). Stride: 80 bytes / 20 f32 per tri.
//   [0..2]  vA.xyz + pad    [4..6]  vB.xyz + pad    [8..10] vC.xyz + pad
//   [12..14] normal.xyz + area (at [15])             [16..18] Le.rgb + pad
// emitterCount (uniform in lights) is reused for the area-emitter count. A
// dedicated u32 is cheaper than a second UBO; it lives in DdgiTraceParams.
@group(1) @binding(2) var<storage, read> ddgiEmitterTris: array<vec4f>;

@group(2) @binding(0) var<storage, read_write> rayResults:   array<ProbeRay>;
@group(2) @binding(1) var<storage, read>       activeProbes: array<u32>;
@group(2) @binding(2) var                      irradiancePrev: texture_2d<f32>;
@group(2) @binding(3) var                      irradianceSamp: sampler;
@group(2) @binding(4) var<uniform>             gridParams:   ProbeGridParams;
@group(2) @binding(5) var<uniform>             frameParams:  FrameParams;
// Wave 4 (2026-06-10) — HDRI into DDGI probe misses.
// ddgiEnvMap  : rgba16float equirect radiance (unit-intensity, .rgb; .a unused
//               by DDGI — pdf lane is for DI MIS). A 1×1 placeholder is bound
//               when hasEnv=0 so the bind group is always valid.
// NOTE: the env look-up uses textureLoad (not textureSample), so NO sampler
// binding exists here. Trust-audit F3 (2026-06-10): a declared-but-unused
// ddgiEnvSamp sampler at binding(7) was stripped by the layout:'auto'
// pipeline, while the dispatcher still passed an 8th bind-group entry; WebGPU
// rejected the probe-update bind group on EVERY frame (probe radiance silently
// never updated). Sampler removed on both sides.
@group(2) @binding(6) var                      ddgiEnvMap:   texture_2d<f32>;

// -----------------------------------------------------------------
// BVH traversal — merged world BLAS or TLAS+local BLAS (PR-5.2).
// -----------------------------------------------------------------
const DDGI_TRI_EPSILON: f32 = 1e-5;

fn safe_normalize(v: vec3f) -> vec3f {
  let len2 = dot(v, v);
  if (len2 < 1e-20) { return vec3f(0.0, 1.0, 0.0); }
  return v * inverseSqrt(len2);
}

fn traceSceneFirstHitDdgi(ray: Ray) -> IntersectionResult {
  if (ddgiTrace.bvhMode == 1u && ddgiTrace.tlasNodeCount > 0u) {
    return traceTlasFirstHit(
      &tlasNodes,
      &tlasInstanceIndices,
      &tlasBlasRoots,
      &tlasInstanceWorldToLocal,
      &tlasInstanceLocalToWorld,
      ddgiTrace.tlasNodeCount,
      &bvh_index,
      &bvh_position,
      &bvh,
      ray,
      DDGI_TRI_EPSILON,
    );
  }
  return bvhIntersectFirstHit(&bvh_index, &bvh_position, &bvh, ray, DDGI_TRI_EPSILON);
}

fn bvhTraceFirstHit(ray: Ray) -> IntersectionResult {
  return traceSceneFirstHitDdgi(ray);
}
`; }

/**
 * Glass-aware sun-visibility helper (traceSunVisibility function).
 *
 * Exported so test harnesses can compile + verify this section independently.
 * Composed first after the header in makeProbeUpdateRaysWGSL.
 */
export function makeTraceSunVisibilityWGSL(): string { return /* wgsl */`
// -----------------------------------------------------------------
// Direct lighting at a hit point
// -----------------------------------------------------------------

// Glass-aware sun visibility helper (caustic enabler).
//
// Returns a per-channel visibility multiplier from origin along sunDir:
//   - Unobstructed     -> vec3f(1.0)
//   - Hit opaque       -> vec3f(0.0)
//   - Hit glass        -> Beer-Lambert transmittance, then continue past the
//                        slab and recurse (bounded to 3 glass crossings).
//
// Beer-Lambert glass attenuation (B5, 2026-06-10). Per glass slab:
//   visibility *= transmission · exp(-attenuationColor · (t / attenuationDistance))
// where attenuationColor is the per-channel absorption coefficient σ and the
// dimensionless optical-depth ratio t/attenuationDistance is the path length in
// units of the medium e-fold (mean-free) distance — matching the canonical
// probeRayCast.wgsl in @vitrum/walkaround-rc (which uses the material thickness
// scalar). HERE we have the actual continuation ray, so we use the TRUE
// geometric path length through the slab: t = (entry->exit) distance found by
// continuing the ray inside the medium to its next surface, clamped to the
// material thickness as an upper bound so a probe ray that grazes a thin
// pane or misses the far face (open/non-watertight glass) cannot accumulate an
// unbounded optical depth. This is the documented path-length approximation:
//   t = clamp(distToExit, 0, thickness)   [exit = next hit along sunDir]
// Limits: σ→0 OR t→0  ⇒ exp(0)=1 (clear glass passes transmission only);
//         σ→∞ OR t→∞  ⇒ exp(-∞)=0 (opaque). Reduces to Beer-Lambert exactly.
// The previous linear-tint form (visibility *= attenuationColor · transmission)
// did NOT reduce to Beer-Lambert in any limit (no exponential, no thickness).
//
// MaterialEntry carries attenuationColor / attenuationDistance / thickness
// (W2-C5) so no buffer-layout change is needed.
fn traceSunVisibility(origin: vec3f, sunDir: vec3f) -> vec3f {
  var visibility = vec3f(1.0);
  var rayOrigin  = origin;
  // Bounded glass-crossing loop. Upper bound 3 handles single-slab and
  // edge-case double-slab paths; ≥4 stained-glass panels in series is not
  // a configuration the walkaround pipeline produces.
  for (var iter: u32 = 0u; iter < 3u; iter = iter + 1u) {
    var sRay: Ray;
    sRay.origin    = rayOrigin;
    sRay.direction = sunDir;
    let sHit = bvhTraceFirstHit(sRay);
    if (!sHit.didHit || sHit.dist >= 1e15) {
      // Reached sky — sun is unobstructed (modulo accumulated glass tint).
      return visibility;
    }
    let sMatId = bvh_materialId[sHit.indices.w];
    let sMat   = materials[sMatId];
    if ((sMat.flags & 1u) == 0u) {
      // Opaque occluder — sun is fully blocked.
      return vec3f(0.0);
    }
    // Glass slab — Beer-Lambert transmittance over the geometric path length.
    // Find the exit point by intersecting the continuation ray just past the
    // entry face; distToExit is the in-medium path length. Clamp to thickness
    // (upper bound — guards open/non-watertight glass where the ray would exit
    // far away or miss the far face entirely).
    let entryPos  = rayOrigin + sunDir * sHit.dist;
    var exitRay: Ray;
    exitRay.origin    = entryPos + sunDir * (gridParams.spacing * 1e-4);
    exitRay.direction = sunDir;
    let exitHit  = bvhTraceFirstHit(exitRay);
    let distToExit = select(sMat.thickness, exitHit.dist, exitHit.didHit && exitHit.dist < 1e15);
    let pathLen  = clamp(distToExit, 0.0, max(sMat.thickness, 1e-4));
    let beerAtten = exp(-sMat.attenuationColor * (pathLen / max(1e-4, sMat.attenuationDistance)));
    visibility = visibility * sMat.transmission * beerAtten;
    let hitPos = entryPos;
    // M14: step past the slab by 1% of probe spacing so the offset is
    // proportional to scene scale (replacing the Cornell-specific 0.5 units).
    // For Cornell spacing ~0.17 → step 0.0017; for a 100-unit building →
    // step ~3 units, ensuring the continuation ray clears the slab face.
    rayOrigin  = hitPos + sunDir * (gridParams.spacing * 0.01);
  }
  // Loop exhausted (more than 3 glass crossings) — treat as fully attenuated.
  return vec3f(0.0);
}
`; }

/**
 * Analytic-light + area-emitter NEE block (sun + point/spot + mesh-area NEE).
 *
 * Exported for independent compile/test. Composed after makeTraceSunVisibilityWGSL
 * in makeProbeUpdateRaysWGSL.
 */
export function makeDirectLightingWGSL(): string { return /* wgsl */`
fn evalSunLight(lightDir: vec3f, lightColor: vec3f, intensity: f32,
                hitPos: vec3f, hitNormal: vec3f) -> vec3f {
  let nDotL = max(0.0, dot(hitNormal, lightDir));
  if (nDotL < 1e-3) { return vec3f(0.0); }

  // Glass-aware multi-crossing visibility (replaces single-hit bvhTraceFirstHit
  // + binary glass-attenuation pre-fix).
  // M13: normal bias derived from probe spacing to stay scene-scale-agnostic.
  let normalBias = gridParams.spacing * 0.001;
  let visibility = traceSunVisibility(hitPos + hitNormal * normalBias, lightDir);
  return lightColor * intensity * nDotL * visibility;
}

fn evalPointLight(light: DDGILight, hitPos: vec3f, hitNormal: vec3f) -> vec3f {
  let toLight = light.position - hitPos;
  let dist    = length(toLight);
  // Guard against probe-light coincidence (point light embedded in or behind
  // geometry the probe ray hit). Without this, dist==0 yields toLight/dist
  // = NaN, and the downstream nDotL early-out does not catch NaN because any
  // NaN comparison is false, so the NaN propagates into the probe radiance.
  if (dist < 1e-6) { return vec3f(0.0); }
  let lightDir = toLight / dist;
  let nDotL = max(0.0, dot(hitNormal, lightDir));
  if (nDotL < 1e-3) { return vec3f(0.0); }

  // Spot cone falloff: light.direction is the toward-light cone axis (unit for a
  // spot, 0 for a point fixture -> no cone). cosToP = dot(toLightDir, axis) is 1
  // on the axis, cos(angle) at the cone edge. smoothstep(outer, inner, cosToP) is
  // 1 inside the inner cone, ramps to 0 at the outer edge (KHR_lights_punctual).
  // Cheap early-out: fully outside the cone contributes nothing, so skip the ray.
  let axisLen2 = dot(light.direction, light.direction);
  var coneFalloff = 1.0;
  if (axisLen2 > 0.25) {
    let cosToP = dot(lightDir, light.direction * inverseSqrt(axisLen2));
    coneFalloff = smoothstep(light.outerCone, light.innerCone, cosToP);
    if (coneFalloff <= 0.0) { return vec3f(0.0); }
  }

  // M13: normal bias proportional to probe spacing (scene-scale-agnostic).
  let normalBias_p = gridParams.spacing * 0.001;
  var shadowRay: Ray;
  shadowRay.origin    = hitPos + hitNormal * normalBias_p;
  shadowRay.direction = lightDir;
  let shadow = bvhTraceFirstHit(shadowRay);
  if (shadow.didHit && shadow.dist < dist - normalBias_p) {
    return vec3f(0.0);
  }
  let atten = light.intensity / (dist * dist + 1.0);
  return light.color * atten * nDotL * coneFalloff;
}

fn evalDirectLighting(hitPos: vec3f, hitNormal: vec3f) -> vec3f {
  var result = vec3f(0.0);
  for (var li = 0u; li < min(lights.count, MAX_LIGHTS); li = li + 1u) {
    let light = lights.items[li];
    if (light.kind == LIGHT_SUN) {
      let dir = normalize(-light.direction);
      result = result + evalSunLight(dir, light.color, light.intensity, hitPos, hitNormal);
    } else if (light.kind == LIGHT_POINT) {
      result = result + evalPointLight(light, hitPos, hitNormal);
    }
  }
  return result;
}

// -----------------------------------------------------------------
// H18 Stage 2 — Area-emitter NEE for probe rays
//
// One deterministic sample per emitter triangle (same "sum-all, weight by area"
// pattern as RC probeRayCast.wgsl rcEmitterNEE). Gated on
// ddgiTrace.emitterTriCount > 0 so sun-only scenes are byte-identical.
//
// Estimator (area form, pdf = 1/area ⇒ 1/pdf = area):
//   Lo += (albedo/π) · Le · (cosSurf · cosLight / dist²) · area · vis
// Shadow test: opaque first-hit only (glass tint ignored — DDGI is a coarse
// cache). Bias via the same gridParams.spacing-derived normal offset as the
// sun path.
// -----------------------------------------------------------------
fn pcgHashToF32Ddgi(seed: u32) -> f32 {
  // Avalanche hash → float in [0, 1)
  var s = seed;
  s = s ^ (s >> 17u);
  s = s * 0xBF324C81u;
  s = s ^ (s >> 13u);
  s = s * 0x9C7493ADu;
  s = s ^ (s >> 15u);
  return f32(s >> 8u) * (1.0 / 16777216.0);
}

fn ddgiEmitterNEE(hitPos: vec3f, n: vec3f, albedo: vec3f, seed0: u32) -> vec3f {
  let count = ddgiTrace.emitterTriCount;
  if (count == 0u) { return vec3f(0.0); }
  let normalBias = gridParams.spacing * 0.001;
  var Lo = vec3f(0.0);
  for (var ei: u32 = 0u; ei < count; ei = ei + 1u) {
    // Decode the 5-vec4f EmitterTri entry (80 bytes = 20 f32 = 5 vec4f).
    let base = ei * 5u;
    let vA  = ddgiEmitterTris[base + 0u].xyz;
    let vB  = ddgiEmitterTris[base + 1u].xyz;
    let vC  = ddgiEmitterTris[base + 2u].xyz;
    let nrm = ddgiEmitterTris[base + 3u].xyz;
    let area = ddgiEmitterTris[base + 3u].w;
    let Le   = ddgiEmitterTris[base + 4u].xyz;

    // Jittered uniform area sample (deterministic per emitter index).
    let s0 = pcgHashToF32Ddgi(seed0 ^ (ei * 0x9E3779B9u + 0x1u));
    let s1 = pcgHashToF32Ddgi(seed0 * 7919u ^ (ei * 0x85EBCA6Bu + 0x2u));
    let su = sqrt(s0);
    let pos = (1.0 - su) * vA + (su * (1.0 - s1)) * vB + (su * s1) * vC;

    let toL     = pos - hitPos;
    let dist2   = max(dot(toL, toL), 1e-8);
    let dist    = sqrt(dist2);
    let wi      = toL / dist;
    let cosSurf  = dot(n, wi);
    let cosLight = dot(nrm, -wi);   // front-face only (one-sided emitter)
    if (cosSurf <= 0.0 || cosLight <= 0.0) { continue; }

    // Opaque shadow test — stop just short of the light sample.
    var sRay: Ray;
    sRay.origin    = hitPos + n * normalBias;
    sRay.direction = wi;
    let sHit = bvhTraceFirstHit(sRay);
    if (sHit.didHit && sHit.dist < dist - normalBias) { continue; }

    let G = (cosSurf * cosLight) / dist2;
    Lo = Lo + albedo * 0.31831 * Le * G * area;   // 0.31831 = 1/π
  }
  return Lo;
}
`; }

/**
 * Probe world-position helper, sky/env sampling, and the main compute entry point.
 *
 * Exported for independent compile/test. Composed last in makeProbeUpdateRaysWGSL.
 */
export function makeProbeMainEntryWGSL(): string { return /* wgsl */`
// -----------------------------------------------------------------
// Probe world position from flat index
// -----------------------------------------------------------------
fn probeWorldPos(probeIdx: u32) -> vec3f {
  let x = f32(probeIdx % gridParams.dims.x);
  let tmp = probeIdx / gridParams.dims.x;
  let y = f32(tmp % gridParams.dims.y);
  let z = f32(tmp / gridParams.dims.y);
  return gridParams.origin + vec3f(x, y, z) * gridParams.spacing;
}

// -----------------------------------------------------------------
// Sky / environment colour sampling for probe miss-rays.
//
// Wave 4 (2026-06-10) — HDRI-aware: when frameParams.hasEnv == 1u the
// probe-ray miss samples the equirect env map (ddgiEnvMap) by the ray
// direction using the SAME UV + rotation convention as envRadiance in
// environmentSample.wgsl (H6):
//
//   lookupDir = ddgiEnvRotateYNeg(normalize(dir), envRotationY)
//               [= RY(-envRotationY)·dir — world → unrotated-map space]
//   phi   = atan2(lookupDir.z, lookupDir.x)
//   theta = acos(clamp(lookupDir.y, -1, 1))
//   u = fract(phi/(2π) + 0.5)        [same as: fract(phi*INV_PI*0.5 + 0.5)]
//   v = clamp(theta/π, 0, 0.999999)
//   texel = textureLoad(ddgiEnvMap, vec2i(floor(u*W), floor(v*H)), 0)
//   result = texel.rgb * max(envIntensity, 0)
//
// When hasEnv == 0u the existing procedural gradient is returned unchanged
// (byte-identical to the pre-Wave-4 path for scenes without an HDRI).
//
// Procedural path: uses frameParams.skyTint and frameParams.skyIrradiance
// written by ProbeUpdatePass.setSkyParams(). Default values
// (tint=(0.4,0.6,1.0), irradiance=2.0) reproduce the former hardcoded
// midpoint exactly — Cornell renders are unchanged.
// -----------------------------------------------------------------

// H6 — RY(-rotY)·d: world direction → unrotated-map lookup direction.
// Matches envRotateYNeg in environmentSample.wgsl exactly.
fn ddgiEnvRotateYNeg(d: vec3f, rotY: f32) -> vec3f {
  let c = cos(rotY);
  let s = sin(rotY);
  return vec3f(c * d.x - s * d.z, d.y, s * d.x + c * d.z);
}

fn sampleSkyColor(dir: vec3f) -> vec3f {
  // HDRI path: sample the equirect map by direction.
  if (frameParams.hasEnv == 1u) {
    let dims = textureDimensions(ddgiEnvMap, 0);
    let w = i32(dims.x);
    let h = i32(dims.y);
    if (w > 0 && h > 0) {
      let lookupDir = ddgiEnvRotateYNeg(safe_normalize(dir), frameParams.envRotationY);
      let phi   = atan2(lookupDir.z, lookupDir.x);
      let theta = acos(clamp(lookupDir.y, -1.0, 1.0));
      let u = fract(phi * (1.0 / (2.0 * PI)) + 0.5);
      let v = clamp(theta * (1.0 / PI), 0.0, 0.999999);
      let ix = clamp(i32(floor(u * f32(w))), 0, w - 1);
      let iy = clamp(i32(floor(v * f32(h))), 0, h - 1);
      let texel = textureLoad(ddgiEnvMap, vec2i(ix, iy), 0);
      return texel.rgb * max(frameParams.envIntensity, 0.0);
    }
  }
  // Procedural sky fallback (no HDRI, or degenerate map dims).
  let above = max(0.0, dir.y);   // 0..1 above horizon
  let below = max(0.0, -dir.y);  // 0..1 below horizon
  // Above-horizon: lerp from horizon (white/neutral) to zenith tint,
  // then scale by the scene's sky irradiance level.
  let horizon = vec3f(0.9, 0.85, 0.75);           // warm neutral horizon (fixed)
  let skyColor = mix(horizon, frameParams.skyTint, above) * frameParams.skyIrradiance;
  // Below-horizon: dark ground, attenuated so it doesn't dominate.
  let ground  = vec3f(0.1, 0.08, 0.06);           // dark earth (fixed)
  let groundColor = mix(horizon, ground, below) * 0.3;
  return mix(skyColor, groundColor, below);
}

// -----------------------------------------------------------------
// Compute entry point
// -----------------------------------------------------------------
@compute @workgroup_size(32, 1, 1)
fn probeUpdateRays(
  @builtin(workgroup_id)       wg:  vec3u,
  @builtin(local_invocation_id) lid: vec3u,
) {
  let activeIdx = wg.x;
  if (activeIdx >= arrayLength(&activeProbes)) { return; }
  let probeIdx = activeProbes[activeIdx];

  let totalProbes = gridParams.dims.x * gridParams.dims.y * gridParams.dims.z;
  if (probeIdx >= totalProbes) { return; }

  let probeOrigin = probeWorldPos(probeIdx);
  let threadId    = lid.x;

  for (var r = 0u; r < RAYS_PER_THREAD; r = r + 1u) {
    let rayIdx = threadId * RAYS_PER_THREAD + r;
    if (rayIdx >= RAYS_PER_PROBE) { break; }

    let dir = ddgiRayDirection(rayIdx, RAYS_PER_PROBE, frameParams.randomRotation);

    var ray: Ray;
    ray.origin    = probeOrigin;
    ray.direction = dir;

    let hit = bvhTraceFirstHit(ray);

    var out: ProbeRay;
    out.direction = dir;

    if (!hit.didHit) {
      out.hitRadiance  = sampleSkyColor(dir);
      out.hitDistance  = BVH_INTERSECT_INFINITY;
      out.hitNormal    = -dir;
      out.hitPosition  = probeOrigin + dir * 1e6;
      out.hitMaterialId = 0u;
      out.isGlass       = 0u;
    } else {
      let matId = bvh_materialId[hit.indices.w];
      let mat   = materials[matId];

      // Backface hit — encode as negative distance per DDGI paper convention.
      let backface = dot(-dir, hit.normal) < 0.0;
      if (backface) {
        out.hitRadiance  = vec3f(0.0);
        out.hitDistance  = -hit.dist;
        out.hitNormal    = hit.normal;
        out.hitPosition  = probeOrigin + dir * hit.dist;
        out.hitMaterialId = matId;
        out.isGlass       = (mat.flags & 1u);
      } else {
        let hitWorldPos = probeOrigin + dir * hit.dist;

        // Smooth normal from barycentric blend.
        let i0 = hit.indices.x;
        let i1 = hit.indices.y;
        let i2 = hit.indices.z;
        let n0 = bvh_normal[i0].xyz;
        let n1 = bvh_normal[i1].xyz;
        let n2 = bvh_normal[i2].xyz;
        let smoothNormal = normalize(
          hit.barycoord.x * n0 +
          hit.barycoord.y * n1 +
          hit.barycoord.z * n2
        ) * hit.side;

        // Direct lighting: analytic sun/fixture lights.
        let direct_analytic = evalDirectLighting(hitWorldPos, smoothNormal);
        // H18 Stage 2 — area-emitter NEE. Guard on emitterTriCount>0 is inside the
        // helper; emitter-less scenes get vec3f(0) at zero cost.
        let direct_emitter = ddgiEmitterNEE(
          hitWorldPos, smoothNormal, mat.baseColor,
          frameParams.frameIndex ^ (probeIdx * 0x9E3779B9u) ^ rayIdx,
        );
        let direct = direct_analytic + direct_emitter;

        // Previous-frame indirect feedback: sample the irradiance atlas at the
        // hit position so each frame folds in one more diffuse bounce; the
        // temporal EMA then converges to the multi-bounce equilibrium. TWO
        // fixes over the pre-2026-06-07 form, BOTH required and BOTH GPU-
        // validated against a CPU f64 multi-bounce path-trace anchor
        // (wsl-gpu/scripts/ddgi-indirect-pi-ab.ts, dzn RTX-4090: mean luminance
        // error vs ground truth 71% [base] -> 63% [clamp only] -> 14% [both]):
        //  1. CLAMP the cell index to [0, dims-1]. The old guard
        //     'baseProbeIdx3 + 1 < dims' returned indirect=0 for EVERY hit on
        //     enclosing geometry (room walls/floor sit on or just past the grid
        //     boundary), which disabled wall->wall->receiver multi-bounce
        //     entirely: the field was effectively SINGLE-bounce and the floor
        //     (lit almost only by wall bounce) came out ~0.58x of reference.
        //     The receiver ddgiSample already clamps to its available probes;
        //     this makes the producer feedback consistent with it. After the
        //     clamp the index is always valid, so the guard is gone.
        //  2. The atlas stores the cosine-weighted incoming-radiance MEAN = E/PI
        //     (the blend pass and ddgiSample both reconstruct E by multiplying
        //     by PI). 'direct' here is irradiance E, so the atlas read must also
        //     be multiplied by PI to add like-for-like BEFORE the (baseColor/PI)
        //     bounce factor below. The old code added E/PI to E, making the
        //     indirect feedback PI-times too weak at every bounce.
        let gridPos  = (hitWorldPos - gridParams.origin) / gridParams.spacing;
        let baseProbeIdx3 = clamp(vec3i(floor(gridPos)), vec3i(0), vec3i(gridParams.dims) - vec3i(1));
        let pi = u32(baseProbeIdx3.x) + u32(baseProbeIdx3.y) * gridParams.dims.x +
                 u32(baseProbeIdx3.z) * gridParams.dims.x * gridParams.dims.y;
        // L2 SH irradiance eval at the bounce normal (seam-free; replaces the
        // octahedral lookup + *PI). irradiancePrev holds the 9 cosine-convolved
        // SH coeffs per probe in the first 3x3 interior texels, so the eval
        // returns irradiance E directly.
        let shStride = ${IRR_STRIDE}u;
        let fpx = pi % gridParams.dims.x;
        let ftmp = pi / gridParams.dims.x;
        let fpy = ftmp % gridParams.dims.y;
        let fpz = ftmp / gridParams.dims.y;
        let fix = fpx * shStride + 1u;
        let fiy = (fpy + fpz * gridParams.dims.y) * shStride + 1u;
        let indirect = ddgiSampleSHProbe(
          irradiancePrev, irradianceSamp,
          gridParams.irradianceAtlasW, gridParams.irradianceAtlasH,
          fix, fiy, smoothNormal,
        );

        // Outgoing radiance from the BOUNCE surface toward the probe.
        //
        // direct (evalDirectLighting) and indirect (previous-frame atlas
        // lookup) are both IRRADIANCE E at the bounce surface — no BRDF. A
        // Lambertian bounce surface re-emits Lo = (baseColor/π)·E toward the
        // probe; that factor belongs to the BOUNCE surface and is applied
        // HERE. The receiver factor ((albedo/π)·E_atlas in
        // applyDDGIShading.ts / giReceiver.ts) is the SHADED point's BRDF —
        // a different surface. Both factors are required in a physically
        // correct chain: light → wall (ρ_wall/π) → probe atlas (E) →
        // receiver (ρ_recv/π).
        //
        // Post-fix math contract (Majercik 2019 §3 Algorithm 1):
        //   producer : stores Lo = (baseColor_hit/PI) * E_hit, with
        //              E_hit = direct + (atlas read)*PI  [atlas holds E/PI]
        //   blend    : cosine-weights rays -> atlas holds the MEAN E/PI (NOT E;
        //              a stale "atlas holds irradiance E" comment HERE is what
        //              made the feedback skip the *PI for so long, see above)
        //   receiver : reads atlas, reconstructs E (*PI), applies (albedo/PI)*E
        //
        // History: M7 (e66429d Change 3) removed the producer factor,
        // diagnosing producer·receiver albedo as the "double-albedo error"
        // behind the 3fb63e3 gain band-aid — but those are two DIFFERENT
        // surfaces' BRDFs, not double-counting. The real pre-M7 brightness
        // bug was the non-physical pow(8) blend basis + missing SO(3) ray
        // rotation, which M7 Changes 1–2 fixed in the same commit. The
        // white-bounce model M7 left behind stored E itself: perfectly
        // achromatic bounce (zero colour bleed — red/green walls produce
        // grey indirect) and ≈π/albedo (~4× at ρ=0.85) energy over-estimate.
        // GPU A/B vs a CPU f64 path-trace anchor (2026-06-07, dzn RTX-4090 +
        // lavapipe; wsl-gpu/captures/queue-2026-06-07/ddgi-white-bounce/
        // RESULTS.md) sides with this form: 23–34% residual (the octahedral
        // atlas border confound) vs 244–365% for white-bounce on coloured
        // walls.
        //
        // H46-A — maxBounces gate: when indirectFeedback is 0 (maxBounces == 1)
        // the probe carries direct-only light (one bounce: light -> bounce
        // surface -> probe), dropping the previous-frame atlas read that the
        // EMA otherwise converges into the infinite-bounce diffuse equilibrium.
        let indirectGated = select(vec3f(0.0), indirect, frameParams.indirectFeedback != 0u);

        // B2 — Glossy-aware probe bounce: specular complement via reflected
        // previous-frame field (2026-06-10, R8-B).
        //
        // DDGI probes are an irradiance cache — they cannot store a full 5D
        // radiance field. The honest one-bounce specular complement uses the same
        // previous-frame SH atlas evaluated at the REFLECTED probe-ray direction
        // (r = dir - 2·(n·dir)·n) to approximate the outgoing specular radiance.
        //
        // This is the split-sum-flavoured approximation: the atlas stores
        // irradiance E(n) = ∫L(ω)·max(0,n·ω)dω (cosine-weighted incoming
        // hemisphere). Evaluating it at the reflected direction gives the
        // irradiance that would illuminate a surface facing the reflection axis —
        // an approximation of the specular lobe radiance integral. NOT GGX-
        // filtered radiance (which would require a prefiltered radiance cube).
        // The error shrinks as the material approaches a perfect mirror (α→0).
        //
        // Energy discipline: blend, not add. The Lambertian indirect and the
        // specular indirect are alternative transport paths — adding them would
        // double-count. We lerp the indirect contribution between Lambertian
        // and specular by specularWeight; the direct term (analytic lights with
        // Lambertian response) is kept as-is for simplicity. The direct term
        // also uses a Lambertian formulation for analytic lights; the specular
        // improvement applies to the important multi-bounce indirect term where
        // the DDGI atlas is the only source of radiance.
        //
        //   specularWeight = metalness · (1 - roughness²)
        //     = 1 for a perfect mirror metal (metalness=1, roughness=0)
        //     = 0 for a dielectric OR a rough metal (roughness→1)
        //     ranges continuously between these extremes.
        //   roughness² is α² (GGX alpha-squared), so the specular weight
        //   vanishes quadratically with roughness — consistent with how GGX
        //   broadens from a mirror at α=0 to diffuse-equivalent at α=1.
        //
        // Gate: the reflected atlas lookup requires the previous-frame atlas to
        // be populated (indirectFeedback != 0). When direct-only probes are
        // requested (maxBounces == 1, indirectFeedback = 0) the specular
        // complement is also disabled — both paths fall through to the
        // Lambertian-direct-only formula, preserving byte-identity with the
        // pre-B2 path when indirectFeedback = 0.
        //
        // MaterialEntry carries mat.roughness (slot 3) and mat.metalness (slot 7)
        // from the canonical 64-byte struct — no new material threading required.
        // (DDGI material packing in probeUpdateMaterials.ts already fills these
        // fields via pbrToMaterialEntryInput → extractPbrScalars.)
        //
        // Cite: Karis (2013) "Real Shading in Unreal Engine 4" §4.4 (split-sum
        // approximation); McGuire et al. (2017) "Real-Time Global Illumination
        // using Precomputed Light Field Probes" (irradiance-cache specular via
        // reflected direction lookup).
        let specularWeight = mat.metalness * max(0.0, 1.0 - mat.roughness * mat.roughness);
        var indirectRadiance: vec3f;
        if (specularWeight > 1e-4 && frameParams.indirectFeedback != 0u) {
          // Reflected probe-ray direction: mirror dir about the hit normal.
          // dir points FROM the probe TO the hit surface — so -dir is the
          // incoming direction at the surface. reflect(-dir, n) gives the
          // outgoing specular direction, which is also the direction we use to
          // query the SH atlas for the radiance arriving from that hemisphere.
          let reflDir = safe_normalize(dir - 2.0 * dot(dir, smoothNormal) * smoothNormal);
          let specularIrr = ddgiSampleSHProbe(
            irradiancePrev, irradianceSamp,
            gridParams.irradianceAtlasW, gridParams.irradianceAtlasH,
            fix, fiy, reflDir,
          );
          // Specular indirect: atlas irradiance at reflected direction, tinted
          // by metallic baseColor (Fresnel ≈ F0 = baseColor for conductors).
          // Divide by PI for the same irradiance→radiance conversion the
          // Lambertian indirect uses (atlas stores cosine-weighted mean E/PI;
          // ddgiSampleSHProbe returns E — see comment above).
          let specularIndirectLo = mat.baseColor * (specularIrr * (1.0 / PI));
          // Lambertian indirect for the blend reference.
          let lambertianIndirectLo = indirectGated * mat.baseColor * (1.0 / PI);
          // Blend indirect contribution: lerp from Lambertian to specular.
          indirectRadiance = mix(lambertianIndirectLo, specularIndirectLo, specularWeight);
        } else {
          // Rough/dielectric or no feedback: pure Lambertian indirect.
          indirectRadiance = indirectGated * mat.baseColor * (1.0 / PI);
        }
        // Direct: Lambertian (analytic lights use nDotL-weighted eval, kept
        // Lambertian since per-probe direct uses the coarse probe-light model).
        let directRadiance = direct * mat.baseColor * (1.0 / PI);
        var radiance = directRadiance + indirectRadiance;

        if ((mat.flags & 1u) != 0u) {
          // Glass: add transmitted environment contribution.
          let transmitted = sampleSkyColor(dir) * mat.attenuationColor;
          radiance = mix(radiance, transmitted, mat.transmission * frameParams.glassMixScale);
        }

        out.hitRadiance  = radiance;
        out.hitDistance  = hit.dist;
        out.hitNormal    = smoothNormal;
        out.hitPosition  = hitWorldPos;
        out.hitMaterialId = matId;
        out.isGlass       = (mat.flags & 1u);
      }
    }

    let resultIdx = probeIdx * RAYS_PER_PROBE + rayIdx;
    if (resultIdx < arrayLength(&rayResults)) {
      rayResults[resultIdx] = out;
    }
  }
}
`; }
