/**
 * DDGI Probe Update — Pass 1: Ray Casting.
 *
 * One workgroup per probe in the active set. Each thread handles 3 rays
 * (96 rays / 32 threads). Writes ray hit results to a storage buffer for
 * Pass 2 (the atlas blend pass).
 */

import { HAMMERSLEY_WGSL } from '@vitrum/shared-samplers';
import { OCTAHEDRAL_WGSL, MATERIAL_ENTRY_WGSL, BVH_INTERSECT_WGSL } from '@vitrum/shared-bvh';
import { RAYS_PER_PROBE } from '../ddgiConstants.js';

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
  return makeProbeUpdateRaysWGSLImpl(maxMaterials);
}

/** @deprecated Use {@link makeProbeUpdateRaysWGSL}(64) instead. */
export const PROBE_UPDATE_RAYS_WGSL = /* wgsl */ makeProbeUpdateRaysWGSLImpl(64);

function makeProbeUpdateRaysWGSLImpl(maxMaterials: number): string { return /* wgsl */`

${HAMMERSLEY_WGSL}
${OCTAHEDRAL_WGSL}
${MATERIAL_ENTRY_WGSL}
${BVH_INTERSECT_WGSL}

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
  _pad2: u32, _pad3: u32, _pad4: u32,
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
// Bindings
// -----------------------------------------------------------------
@group(0) @binding(0) var<storage, read> bvh:             array<BVHNode>;
@group(0) @binding(1) var<storage, read> bvh_position:    array<vec3f>;
@group(0) @binding(2) var<storage, read> bvh_index:       array<vec3u>;
@group(0) @binding(3) var<storage, read> bvh_normal:      array<vec3f>;
@group(0) @binding(4) var<storage, read> bvh_materialId:  array<u32>;

@group(1) @binding(0) var<uniform> materials:     array<MaterialEntry, ${maxMaterials}>;
@group(1) @binding(1) var<uniform> lights:        DDGILightUniforms;

@group(2) @binding(0) var<storage, read_write> rayResults:   array<ProbeRay>;
@group(2) @binding(1) var<storage, read>       activeProbes: array<u32>;
@group(2) @binding(2) var                      irradiancePrev: texture_2d<f32>;
@group(2) @binding(3) var                      irradianceSamp: sampler;
@group(2) @binding(4) var<uniform>             gridParams:   ProbeGridParams;
@group(2) @binding(5) var<uniform>             frameParams:  FrameParams;

// -----------------------------------------------------------------
// BVH traversal — canonical helpers come from @vitrum/shared-bvh
// BVH_INTERSECT_WGSL (injected at the top of this file). The pre-canonical
// local helpers safeInvDir, intersectsAABBDist, intersectsTriangleBVH
// (hardcoded triEps = 1e-5), and bvhTraceFirstHit (60-LOC traversal loop)
// were here and have been removed. The four call sites below thread
// DDGI_TRI_EPSILON through canonical bvhIntersectFirstHitV3 — replacing
// the pre-canonical hardcoded constant with a single named local that a
// future revision can promote to a UBO field without touching the call
// sites.
// -----------------------------------------------------------------
const DDGI_TRI_EPSILON: f32 = 1e-5;

// Adapter shim — preserves the pre-canonical bvhTraceFirstHit(ray) call
// signature used at four sites below (traceSunVisibility, the point-light
// shadow query, the main probe-ray cast, etc.) by threading DDGI_TRI_EPSILON
// into the canonical vec3-storage entry point. Inlining the canonical
// directly at each call site would also work; the shim keeps the diff
// concentrated and the call sites readable.
fn bvhTraceFirstHit(ray: Ray) -> IntersectionResult {
  return bvhIntersectFirstHitV3(
    &bvh_index, &bvh_position, &bvh, ray,
    DDGI_TRI_EPSILON,
  );
}

// -----------------------------------------------------------------
// Direct lighting at a hit point
// -----------------------------------------------------------------

// Glass-aware sun visibility helper (caustic enabler).
//
// Returns a per-channel visibility multiplier from origin along sunDir:
//   - Unobstructed     -> vec3f(1.0)
//   - Hit opaque       -> vec3f(0.0)
//   - Hit glass        -> tint * transmission, then continue past the slab
//                        and recurse (bounded to 3 glass crossings).
//
// Linear-tint glass attenuation (NOT Beer-Lambert): this kernel applies
// visibility *= attenuationColor * transmission per glass slab — no
// exponential, no thickness. The canonical MaterialEntry now carries
// thickness and attenuationDistance (W2-C5), so a future revision can
// promote this to full Beer-Lambert exp(-attenColor * thickness /
// attenDist) (matching probeRayCast.wgsl in @vitrum/walkaround-rc)
// without changing the buffer layout. Earlier comments labelled this
// "Beer-Lambert simplification" — the linear formulation does NOT
// reduce to Beer-Lambert in any limit, so the label was misleading.
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
    // Glass slab — apply per-cell tint, then continue past the slab.
    visibility = visibility * sMat.attenuationColor * sMat.transmission;
    let hitPos = rayOrigin + sunDir * sHit.dist;
    // M14: step past the slab by 1% of probe spacing so the offset is
    // proportional to scene scale (replacing the Cornell-specific 0.5 units).
    // For Cornell spacing ~0.17 → step 0.0017; for a 100-unit building →
    // step ~3 units, ensuring the continuation ray clears the slab face.
    rayOrigin  = hitPos + sunDir * (gridParams.spacing * 0.01);
  }
  // Loop exhausted (more than 3 glass crossings) — treat as fully attenuated.
  return vec3f(0.0);
}

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

fn evalPointLight(lightPos: vec3f, lightColor: vec3f, intensity: f32,
                  hitPos: vec3f, hitNormal: vec3f) -> vec3f {
  let toLight = lightPos - hitPos;
  let dist    = length(toLight);
  // Guard against probe-light coincidence (point light embedded in or behind
  // geometry the probe ray hit). Without this, dist==0 yields toLight/dist
  // = NaN, and the downstream nDotL early-out does not catch NaN because any
  // NaN comparison is false, so the NaN propagates into the probe radiance.
  if (dist < 1e-6) { return vec3f(0.0); }
  let lightDir = toLight / dist;
  let nDotL = max(0.0, dot(hitNormal, lightDir));
  if (nDotL < 1e-3) { return vec3f(0.0); }

  // M13: normal bias proportional to probe spacing (scene-scale-agnostic).
  let normalBias_p = gridParams.spacing * 0.001;
  var shadowRay: Ray;
  shadowRay.origin    = hitPos + hitNormal * normalBias_p;
  shadowRay.direction = lightDir;
  let shadow = bvhTraceFirstHit(shadowRay);
  if (shadow.didHit && shadow.dist < dist - normalBias_p) {
    return vec3f(0.0);
  }
  let atten = intensity / (dist * dist + 1.0);
  return lightColor * atten * nDotL;
}

fn evalDirectLighting(hitPos: vec3f, hitNormal: vec3f) -> vec3f {
  var result = vec3f(0.0);
  for (var li = 0u; li < min(lights.count, MAX_LIGHTS); li = li + 1u) {
    let light = lights.items[li];
    if (light.kind == LIGHT_SUN) {
      let dir = normalize(-light.direction);
      result = result + evalSunLight(dir, light.color, light.intensity, hitPos, hitNormal);
    } else if (light.kind == LIGHT_POINT) {
      result = result + evalPointLight(light.position, light.color, light.intensity, hitPos, hitNormal);
    }
  }
  return result;
}

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
// Uses frameParams.skyTint and frameParams.skyIrradiance written by the
// host's ProbeUpdatePass.setSkyParams() (B2 audit remediation — replaced
// the Cornell-only hardcoded gradient). The default host values
// (tint=vec3f(0.4,0.6,1.0), irradiance=2.0) reproduce the former
// hardcoded midpoint exactly, so Cornell renders are unchanged.
//
// Geometry: above-horizon tinted by skyTint × cosine falloff from zenith;
// below-horizon attenuated to a neutral dark ground to avoid artificially
// brightening the indirect irradiance from probe rays that miss the floor.
// -----------------------------------------------------------------
fn sampleSkyColor(dir: vec3f) -> vec3f {
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
        let n0 = bvh_normal[i0];
        let n1 = bvh_normal[i1];
        let n2 = bvh_normal[i2];
        let smoothNormal = normalize(
          hit.barycoord.x * n0 +
          hit.barycoord.y * n1 +
          hit.barycoord.z * n2
        ) * hit.side;

        // Direct lighting.
        let direct = evalDirectLighting(hitWorldPos, smoothNormal);

        // Previous-frame indirect: sample the irradiance atlas at the
        // hit position. Simple atlas UV from probe-grid lookup.
        let gridPos  = (hitWorldPos - gridParams.origin) / gridParams.spacing;
        let baseProbeIdx3 = vec3i(floor(gridPos));
        var indirect = vec3f(0.0);
        if (all(baseProbeIdx3 >= vec3i(0)) &&
            all(baseProbeIdx3 + vec3i(1) < vec3i(gridParams.dims))) {
          let pi = u32(baseProbeIdx3.x) + u32(baseProbeIdx3.y) * gridParams.dims.x +
                   u32(baseProbeIdx3.z) * gridParams.dims.x * gridParams.dims.y;
          let octUv = (octEncode(smoothNormal) * 0.5 + 0.5);
          let iUv   = irradianceAtlasUv(
            pi, octUv,
            gridParams.irradianceAtlasW, gridParams.irradianceAtlasH,
            gridParams.dims,
          );
          indirect = textureSampleLevel(irradiancePrev, irradianceSamp, iUv, 0.0).rgb;
        }

        // Store raw incoming radiance L_i at the probe (direct + indirect
        // from the hit surface). Do NOT premultiply by albedo/π here.
        //
        // The blend kernel (probeUpdateBlend) applies a cosine weight and
        // averages over rays; the result stored in the atlas is irradiance E
        // (Majercik 2019 §3 Algorithm 1). The receiver (applyDDGIShading.ts
        // or giReceiver.ts) applies (albedo/π) · E to produce outgoing
        // Lambertian diffuse radiance at the shaded point.
        //
        // M7 DDGI Coherent Physical Model: albedo/π baking moved from
        // producer to receiver to eliminate the double-albedo error
        // (albedo²/π²) that required the 3fb63e3 band-aid gain reduction.
        var radiance = (direct + indirect);

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

