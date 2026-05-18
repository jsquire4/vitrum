/**
 * DDGI Probe Update — Pass 1: Ray Casting.
 *
 * One workgroup per probe in the active set. Each thread handles 3 rays
 * (96 rays / 32 threads). Writes ray hit results to a storage buffer for
 * Pass 2 (the atlas blend pass).
 */

import { HAMMERSLEY_WGSL } from '@vitrum/shared-samplers';
import { OCTAHEDRAL_WGSL } from '@vitrum/shared-bvh';
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
 * `array<DDGIMaterial, 64>` in the WGSL. Now driven by
 * `HybridEngineOptions.ddgiMaxMaterials` (default 64).
 *
 * @param maxMaterials Maximum number of distinct materials. Must be >= 1.
 *        The host-side materialsBuf must be at least
 *        `maxMaterials × DDGI_MATERIAL_STRIDE_BYTES` bytes.
 */
export function makeProbeUpdateRaysWGSL(maxMaterials: number): string {
  if (maxMaterials < 1)
    throw new RangeError(`makeProbeUpdateRaysWGSL: maxMaterials must be >= 1, got ${maxMaterials}`);
  return makeProbeUpdateRaysWGSLImpl(maxMaterials);
}

/** @deprecated Use {@link makeProbeUpdateRaysWGSL}(64) instead. */
export const PROBE_UPDATE_RAYS_WGSL = /* wgsl */ makeProbeUpdateRaysWGSLImpl(64);

function makeProbeUpdateRaysWGSLImpl(maxMaterials: number): string {
  return /* wgsl */ `

${HAMMERSLEY_WGSL}
${OCTAHEDRAL_WGSL}

const WG_SIZE: u32       = ${WG_SIZE}u;
const RAYS_PER_PROBE: u32  = ${RAYS_PER_PROBE}u;
const RAYS_PER_THREAD: u32 = ${RAYS_PER_THREAD}u;   // RAYS_PER_PROBE / WG_SIZE
// NORMAL_BIAS is derived per-frame from gridParams.spacing * 0.001 (see
// evalSunLight / evalPointLight below). M13 audit: the fixed 0.02 was
// Cornell-specific (probe spacing ~0.17 units → 0.17×0.001 = 0.00017, well
// below 0.02). Large scenes with spacing >20 units need a proportionally
// larger bias; tiny scenes (spacing <0.02) would over-bias with 0.02.
// Removed as a compile-time constant; computed inline where needed.
const INFINITY: f32        = 1e20;
const PI: f32              = 3.14159265359;
const BVH_STACK_DEPTH: u32 = 60u;

// Probe-side glass-transmission perceptual scale. When a probe ray hits
// glass we mix room radiance with sky-tinted transmitted radiance,
// weighted by mat.transmission * this constant. At 0.7, fully-transparent
// glass (transmission=1.0) leaves 30% of the original room radiance in
// the probe's irradiance estimate — this prevents bright sky from totally
// drowning out indirect-bounce contribution and matches the perceptual
// cell-vibrance balance dialed in during the 2026-05 caustic sweep.
const GLASS_TRANSMISSION_PROBE_SCALE: f32 = 0.7;

// -----------------------------------------------------------------
// BVH structures (mirrored from three-mesh-bvh)
// -----------------------------------------------------------------
struct BVHBoundingBox {
  min: array<f32, 3>,
  max: array<f32, 3>,
}

struct BVHNode {
  bounds: BVHBoundingBox,
  rightChildOrTriangleOffset: u32,
  splitAxisOrTriangleCount: u32,
}

struct Ray {
  origin: vec3f,
  direction: vec3f,
}

struct IntersectionResult {
  didHit: bool,
  indices: vec4u,
  normal: vec3f,
  barycoord: vec3f,
  side: f32,
  dist: f32,
}

// -----------------------------------------------------------------
// DDGI material table
// -----------------------------------------------------------------
struct DDGIMaterial {
  baseColor: vec3f,
  _pad0: f32,
  emissive: vec3f,
  roughness: f32,
  metalness: f32,
  ior: f32,
  transmission: f32,
  _pad1: f32,
  attenuationColor: vec3f,
  flags: u32,   // bit 0 = isGlass, bit 1 = isLight
}

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

@group(1) @binding(0) var<uniform> materials:     array<DDGIMaterial, ${maxMaterials}>;
@group(1) @binding(1) var<uniform> lights:        DDGILightUniforms;

@group(2) @binding(0) var<storage, read_write> rayResults:   array<ProbeRay>;
@group(2) @binding(1) var<storage, read>       activeProbes: array<u32>;
@group(2) @binding(2) var                      irradiancePrev: texture_2d<f32>;
@group(2) @binding(3) var                      irradianceSamp: sampler;
@group(2) @binding(4) var<uniform>             gridParams:   ProbeGridParams;
@group(2) @binding(5) var<uniform>             frameParams:  FrameParams;

// -----------------------------------------------------------------
// BVH traversal (inline — not using wgslFn wrapper here for robustness)
// -----------------------------------------------------------------

// Williams 2005 §4 IEEE-safe inverse-direction helper.
// Prevents NaN from 0 * ±Inf in slab tests when a ray direction component
// is zero.  WGSL sign(0)==0, so a zero component yields 0*1e30==0, which
// correctly contributes nothing to the tNear/tFar computation.
fn safeInvDir(d: vec3f) -> vec3f {
  return vec3f(
    select(1.0 / d.x, sign(d.x) * 1e30, abs(d.x) < 1e-30),
    select(1.0 / d.y, sign(d.y) * 1e30, abs(d.y) < 1e-30),
    select(1.0 / d.z, sign(d.z) * 1e30, abs(d.z) < 1e-30),
  );
}

fn intersectsAABBDist(ray: Ray, boundsMin: vec3f, boundsMax: vec3f) -> f32 {
  let invDir = safeInvDir(ray.direction);
  let t0 = (boundsMin - ray.origin) * invDir;
  let t1 = (boundsMax - ray.origin) * invDir;
  let tmin3 = min(t0, t1);
  let tmax3 = max(t0, t1);
  let tmin = max(max(tmin3.x, tmin3.y), tmin3.z);
  let tmax = min(min(tmax3.x, tmax3.y), tmax3.z);
  if (tmax < 0.0 || tmin > tmax) { return INFINITY; }
  return max(0.0, tmin);
}

fn intersectsTriangleBVH(ray: Ray, a: vec3f, b: vec3f, c: vec3f) -> IntersectionResult {
  var result: IntersectionResult;
  result.didHit = false;
  let e1 = b - a;
  let e2 = c - a;
  let n  = cross(e1, e2);
  let det = -dot(ray.direction, n);
  if (abs(det) < 1e-5) { return result; }
  let invDet = 1.0 / det;
  let AO = ray.origin - a;
  let DAO = cross(AO, ray.direction);
  let u = dot(e2, DAO) * invDet;
  let v = -dot(e1, DAO) * invDet;
  let t = dot(AO, n) * invDet;
  let w = 1.0 - u - v;
  if (u < -1e-5 || v < -1e-5 || w < -1e-5 || t < 1e-5) { return result; }
  result.didHit   = true;
  result.dist     = t;
  result.barycoord = vec3f(w, u, v);
  result.side     = sign(det);
  result.normal   = result.side * normalize(n);
  return result;
}

fn bvhTraceFirstHit(ray: Ray) -> IntersectionResult {
  var best: IntersectionResult;
  best.didHit = false;
  best.dist   = INFINITY;

  // BVH traversal matching three-mesh-bvh's bvhIntersectFirstHit format.
  // BVHNode layout: bounds (6 floats), rightChildOrTriangleOffset (u32),
  // splitAxisOrTriangleCount (u32).
  // isLeaf = (splitAxisOrTriangleCount & 0xffff0000u) != 0u
  // leaf: triCount = splitAxisOrTriangleCount & 0x0000ffffu,
  //        triOffset = rightChildOrTriangleOffset
  // internal: splitAxis = splitAxisOrTriangleCount & 0x0000ffffu,
  //            rightChild = currNodeIndex + rightChildOrTriangleOffset

  var pointer: i32 = 0;
  var stack: array<u32, 60>;
  stack[0] = 0u;

  loop {
    if (pointer < 0 || pointer >= i32(BVH_STACK_DEPTH)) { break; }
    let currNodeIdx = stack[pointer];
    let node        = bvh[currNodeIdx];
    pointer -= 1;

    let bmin    = vec3f(node.bounds.min[0], node.bounds.min[1], node.bounds.min[2]);
    let bmax    = vec3f(node.bounds.max[0], node.bounds.max[1], node.bounds.max[2]);
    let tHit    = intersectsAABBDist(ray, bmin, bmax);
    if (tHit > best.dist) { continue; }

    let boundsInfoX = node.splitAxisOrTriangleCount;
    let boundsInfoY = node.rightChildOrTriangleOffset;
    let isLeaf = (boundsInfoX & 0xffff0000u) != 0u;

    if (isLeaf) {
      let triCount  = boundsInfoX & 0x0000ffffu;
      let triOffset = boundsInfoY;
      for (var i = 0u; i < triCount; i = i + 1u) {
        let idx = bvh_index[triOffset + i];
        let a   = bvh_position[idx.x];
        let b   = bvh_position[idx.y];
        let c   = bvh_position[idx.z];
        let tri = intersectsTriangleBVH(ray, a, b, c);
        if (tri.didHit && tri.dist < best.dist) {
          best = tri;
          best.indices = vec4u(idx, triOffset + i);
        }
      }
    } else {
      let leftIdx  = currNodeIdx + 1u;
      let rightIdx = currNodeIdx + boundsInfoY;
      let splitAxis = boundsInfoX & 0x0000ffffu;
      let leftToRight = ray.direction[splitAxis] >= 0.0;
      let c1 = select(rightIdx, leftIdx, leftToRight);
      let c2 = select(leftIdx, rightIdx, leftToRight);
      pointer += 1;
      stack[pointer] = c2;
      pointer += 1;
      stack[pointer] = c1;
    }
  }

  return best;
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
// Beer-Lambert simplification: DDGI's DDGIMaterial struct does not carry
// the per-tri thickness / attenuationDistance needed for full
// exp(-attenColor * thickness / attenDist). We use a simplified per-cell
// tint (attenuationColor * transmission) for byte-compatibility with the
// legacy single-hit path. Full Beer-Lambert can be added later if
// DDGIMaterial gains thickness + attenDist fields.
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
      out.hitDistance  = INFINITY;
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
          radiance = mix(radiance, transmitted, mat.transmission * GLASS_TRANSMISSION_PROBE_SCALE);
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
`;
}
