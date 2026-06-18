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
  BVH_CAST_SHADOW_PREDICATE_WGSL,
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
const LIGHT_KIND_MASK: u32 = 0x7fffffffu;
const LIGHT_CAST_SHADOW_DISABLED: u32 = 0x80000000u;
const MAX_LIGHTS:  u32 = 16u;

struct DDGILight {
  kind:       u32,
  distance:   f32,
  decay:      f32,
  _pad2:      f32,
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

fn ddgiLightKind(light: DDGILight) -> u32 {
  return light.kind & LIGHT_KIND_MASK;
}

fn ddgiLightCastShadowDisabled(light: DDGILight) -> bool {
  return (light.kind & LIGHT_CAST_SHADOW_DISABLED) != 0u;
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
//   [0..2]  vA.xyz + sourceTriIndex (-1 = scalar fallback)
//   [4..6]  vB.xyz + sourceSubdivLevel
//   [8..10] vC.xyz + sourceSubdivOrdinal
//   [12..14] normal.xyz + area (at [15])             [16..18] Le.rgb + castShadowDisabled
// emitterCount (uniform in lights) is reused for the area-emitter count. A
// dedicated u32 is cheaper than a second UBO; it lives in DdgiTraceParams.
@group(1) @binding(2) var<storage, read> ddgiEmitterTris: array<vec4f>;
// DDGI probe-hit / emitter-NEE emission-map subset. These are a DDGI-local copy
// of the walkaround material atlas, used to modulate direct probe hits on
// emissive materials and mapped mesh-area emitter samples whose packed
// sourceTriIndex points back to a material-atlas triangle.
@group(1) @binding(3) var ddgiMaterialTextureAtlas: texture_2d_array<f32>;
@group(1) @binding(4) var ddgiMaterialMapMeta: texture_2d<f32>;
// DDGI-local copy of the authored/generated per-vertex tangent.xyzw stream.
// Zero tangents intentionally mean "derive the frame from UVs".
@group(1) @binding(5) var ddgiBvhTangent: texture_2d<f32>;

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

const DDGI_MATERIAL_MAP_META_TEXELS_PER_TRI: u32 = 53u;
const DDGI_MATERIAL_MAP_SLOT_BASE_COLOR: u32 = 0u;
const DDGI_MATERIAL_MAP_SLOT_ROUGHNESS: u32 = 1u;
const DDGI_MATERIAL_MAP_SLOT_METALLIC: u32 = 2u;
const DDGI_MATERIAL_MAP_SLOT_ALPHA: u32 = 4u;
const DDGI_MATERIAL_MAP_ALPHA_COVERAGE_TEXEL_OFFSET: u32 = 10u;
const DDGI_MATERIAL_MAP_EMISSIVE_TEXEL_OFFSET: u32 = 11u;
const DDGI_MATERIAL_MAP_NORMAL_TEXEL_OFFSET: u32 = 15u;
const DDGI_MATERIAL_MAP_NORMAL_SCALE_TEXEL_OFFSET: u32 = 17u;
const DDGI_MATERIAL_MAP_BUMP_TEXEL_OFFSET: u32 = 49u;
const DDGI_MATERIAL_MAP_BUMP_SCALE_TEXEL_OFFSET: u32 = 51u;

fn ddgiMaterialMetaCoord(texel: u32) -> vec2i {
  let dims = textureDimensions(ddgiMaterialMapMeta);
  let w = max(dims.x, 1u);
  return vec2i(i32(texel % w), i32(texel / w));
}

fn ddgiWrapMaterialUv1(v: f32, mode: u32) -> f32 {
  if (mode == 1u) {
    return clamp(v, 0.0, 1.0);
  }
  if (mode == 2u) {
    return 1.0 - abs(fract(v * 0.5) * 2.0 - 1.0);
  }
  return fract(v);
}

fn ddgiWrapMaterialUv(uv: vec2f, wrapPacked: u32) -> vec2f {
  let wrapS = wrapPacked & 0x3u;
  let wrapT = (wrapPacked >> 2u) & 0x3u;
  return vec2f(ddgiWrapMaterialUv1(uv.x, wrapS), ddgiWrapMaterialUv1(uv.y, wrapT));
}

fn ddgiPackedUvFromVec4(v: vec4f) -> vec2f {
  return unpack2x16unorm(bitcast<u32>(v.w));
}

struct DdgiHitMaterialUvs {
  valid: u32,
  uv0: vec2f,
  uv1: vec2f,
}

fn ddgiHitMaterialUvs(hit: IntersectionResult) -> DdgiHitMaterialUvs {
  var out: DdgiHitMaterialUvs;
  out.valid = 0u;
  out.uv0 = vec2f(0.0);
  out.uv1 = vec2f(0.0);

  let i0 = hit.indices.x;
  let i1 = hit.indices.y;
  let i2 = hit.indices.z;
  if (
    hit.indices.w >= arrayLength(&bvh_index) ||
    i0 >= arrayLength(&bvh_position) || i1 >= arrayLength(&bvh_position) || i2 >= arrayLength(&bvh_position) ||
    i0 >= arrayLength(&bvh_normal) || i1 >= arrayLength(&bvh_normal) || i2 >= arrayLength(&bvh_normal)
  ) {
    return out;
  }

  out.valid = 1u;
  out.uv0 =
    hit.barycoord.x * ddgiPackedUvFromVec4(bvh_position[i0]) +
    hit.barycoord.y * ddgiPackedUvFromVec4(bvh_position[i1]) +
    hit.barycoord.z * ddgiPackedUvFromVec4(bvh_position[i2]);
  out.uv1 =
    hit.barycoord.x * ddgiPackedUvFromVec4(bvh_normal[i0]) +
    hit.barycoord.y * ddgiPackedUvFromVec4(bvh_normal[i1]) +
    hit.barycoord.z * ddgiPackedUvFromVec4(bvh_normal[i2]);
  return out;
}

struct DdgiMaterialTangentFrame {
  tangent: vec3f,
  bitangent: vec3f,
}

fn ddgiFallbackBitangentForNormal(n: vec3f, t: vec3f) -> vec3f {
  let b = cross(n, t);
  let len2 = dot(b, b);
  if (len2 < 1e-8) {
    let up = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(n.y) > 0.95);
    return normalize(cross(n, up));
  }
  return b * inverseSqrt(len2);
}

fn ddgiBvhTangentTexel(vertexIndex: u32) -> vec4f {
  let dims = textureDimensions(ddgiBvhTangent);
  let width = max(dims.x, 1u);
  let height = max(dims.y, 1u);
  let y = vertexIndex / width;
  if (y >= height) {
    return vec4f(0.0);
  }
  return textureLoad(ddgiBvhTangent, vec2i(i32(vertexIndex % width), i32(y)), 0);
}

fn ddgiTransformDirectionCols(l2w0: vec4f, l2w1: vec4f, l2w2: vec4f, v: vec3f) -> vec3f {
  return l2w0.xyz * v.x + l2w1.xyz * v.y + l2w2.xyz * v.z;
}

fn ddgiTangentHandednessForLocalToWorld(l2w0: vec4f, l2w1: vec4f, l2w2: vec4f) -> f32 {
  let det = dot(l2w0.xyz, cross(l2w1.xyz, l2w2.xyz));
  return select(-1.0, 1.0, det >= 0.0);
}

fn ddgiPreferAuthoredTangentFrameForHit(
  hit: IntersectionResult,
  frameNormal: vec3f,
  fallbackTangent: vec3f,
  fallbackBitangent: vec3f,
) -> DdgiMaterialTangentFrame {
  var tangent = fallbackTangent;
  var bitangent = fallbackBitangent;

  let ta = ddgiBvhTangentTexel(hit.indices.x);
  let tb = ddgiBvhTangentTexel(hit.indices.y);
  let tc = ddgiBvhTangentTexel(hit.indices.z);
  var authoredTangent =
    hit.barycoord.x * ta.xyz +
    hit.barycoord.y * tb.xyz +
    hit.barycoord.z * tc.xyz;
  var authoredHandedness =
    hit.barycoord.x * ta.w +
    hit.barycoord.y * tb.w +
    hit.barycoord.z * tc.w;

  if (length(authoredTangent) > 1e-8 && abs(authoredHandedness) > 0.5) {
    let isTlas = ddgiTrace.bvhMode == 1u;
    let tBase = hit.instanceIndex * 4u;
    let tOk = isTlas && tBase + 2u < arrayLength(&tlasInstanceLocalToWorld);
    if (tOk) {
      authoredTangent = ddgiTransformDirectionCols(
        tlasInstanceLocalToWorld[tBase],
        tlasInstanceLocalToWorld[tBase + 1u],
        tlasInstanceLocalToWorld[tBase + 2u],
        authoredTangent,
      );
      authoredHandedness = authoredHandedness * ddgiTangentHandednessForLocalToWorld(
        tlasInstanceLocalToWorld[tBase],
        tlasInstanceLocalToWorld[tBase + 1u],
        tlasInstanceLocalToWorld[tBase + 2u],
      );
    }

    authoredTangent = authoredTangent - frameNormal * dot(frameNormal, authoredTangent);
    let tLen2 = dot(authoredTangent, authoredTangent);
    if (tLen2 > 1e-8) {
      tangent = authoredTangent * inverseSqrt(tLen2);
      bitangent = cross(frameNormal, tangent) * select(-1.0, 1.0, authoredHandedness >= 0.0);
    }
  }

  return DdgiMaterialTangentFrame(tangent, bitangent);
}

fn ddgiMaterialTangentFrameForHit(
  hit: IntersectionResult,
  frameNormal: vec3f,
  mapOffset: u32,
) -> DdgiMaterialTangentFrame {
  let triIndex = hit.indices.w;
  let metaTexel = triIndex * DDGI_MATERIAL_MAP_META_TEXELS_PER_TRI + mapOffset;
  let meta0 = textureLoad(ddgiMaterialMapMeta, ddgiMaterialMetaCoord(metaTexel), 0);
  let flags = u32(max(meta0.y, 0.0) + 0.5);
  let useUv1 = ((flags >> 4u) & 0x3u) == 1u;

  let p0 = bvh_position[hit.indices.x];
  let p1 = bvh_position[hit.indices.y];
  let p2 = bvh_position[hit.indices.z];
  let n0 = bvh_normal[hit.indices.x];
  let n1 = bvh_normal[hit.indices.y];
  let n2 = bvh_normal[hit.indices.z];
  let uv0a = ddgiPackedUvFromVec4(p0);
  let uv0b = ddgiPackedUvFromVec4(p1);
  let uv0c = ddgiPackedUvFromVec4(p2);
  let uv1a = ddgiPackedUvFromVec4(n0);
  let uv1b = ddgiPackedUvFromVec4(n1);
  let uv1c = ddgiPackedUvFromVec4(n2);
  let ta = select(uv0a, uv1a, useUv1);
  let tb = select(uv0b, uv1b, useUv1);
  let tc = select(uv0c, uv1c, useUv1);

  let dp1 = p1.xyz - p0.xyz;
  let dp2 = p2.xyz - p0.xyz;
  let duv1 = tb - ta;
  let duv2 = tc - ta;
  let det = duv1.x * duv2.y - duv1.y * duv2.x;
  var tangent = dp1;
  var bitangent = ddgiFallbackBitangentForNormal(frameNormal, tangent);
  if (abs(det) > 1e-8) {
    let invDet = 1.0 / det;
    tangent = (dp1 * duv2.y - dp2 * duv1.y) * invDet;
    bitangent = (dp2 * duv1.x - dp1 * duv2.x) * invDet;
  }

  tangent = tangent - frameNormal * dot(frameNormal, tangent);
  let tLen2 = dot(tangent, tangent);
  if (tLen2 < 1e-8) {
    let up = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(frameNormal.y) > 0.95);
    tangent = normalize(cross(up, frameNormal));
  } else {
    tangent = tangent * inverseSqrt(tLen2);
  }

  bitangent = bitangent - frameNormal * dot(frameNormal, bitangent) - tangent * dot(tangent, bitangent);
  let bLen2 = dot(bitangent, bitangent);
  if (bLen2 < 1e-8) {
    bitangent = ddgiFallbackBitangentForNormal(frameNormal, tangent);
  } else {
    bitangent = bitangent * inverseSqrt(bLen2);
  }

  return ddgiPreferAuthoredTangentFrameForHit(hit, frameNormal, tangent, bitangent);
}

fn ddgiSampleMaterialAtlasRawAtOffsetDelta(
  triIndex: u32,
  metaOffset: u32,
  uv0: vec2f,
  uv1: vec2f,
  transformedDelta: vec2f,
) -> vec4f {
  let metaDims = textureDimensions(ddgiMaterialMapMeta);
  let metaTexel = triIndex * DDGI_MATERIAL_MAP_META_TEXELS_PER_TRI + metaOffset;
  if (metaTexel + 1u >= metaDims.x * metaDims.y) {
    return vec4f(-1.0);
  }
  let meta0 = textureLoad(ddgiMaterialMapMeta, ddgiMaterialMetaCoord(metaTexel), 0);
  let layer = i32(meta0.x);
  if (layer < 0 || u32(layer) >= textureNumLayers(ddgiMaterialTextureAtlas)) {
    return vec4f(-1.0);
  }
  let wrapPacked = u32(max(meta0.y, 0.0) + 0.5);
  let texCoord = (wrapPacked >> 4u) & 0x3u;
  let uv = select(uv0, uv1, texCoord == 1u);
  let meta1 = textureLoad(ddgiMaterialMapMeta, ddgiMaterialMetaCoord(metaTexel + 1u), 0);
  let scaled = uv * meta1.xy;
  let transformed = vec2f(
    scaled.x * meta1.z - scaled.y * meta1.w,
    scaled.x * meta1.w + scaled.y * meta1.z,
  ) + meta0.zw + transformedDelta;
  let wrapped = ddgiWrapMaterialUv(transformed, wrapPacked);
  let dims = textureDimensions(ddgiMaterialTextureAtlas);
  let texel = vec2i(
    i32(min(u32(floor(wrapped.x * f32(dims.x))), dims.x - 1u)),
    i32(min(u32(floor(wrapped.y * f32(dims.y))), dims.y - 1u)),
  );
  return textureLoad(ddgiMaterialTextureAtlas, texel, layer, 0);
}

fn ddgiSampleMaterialAtlasRawAtOffset(triIndex: u32, metaOffset: u32, uv0: vec2f, uv1: vec2f) -> vec4f {
  return ddgiSampleMaterialAtlasRawAtOffsetDelta(triIndex, metaOffset, uv0, uv1, vec2f(0.0));
}

fn ddgiSampleMaterialAtlasRaw(triIndex: u32, slot: u32, uv0: vec2f, uv1: vec2f) -> vec4f {
  return ddgiSampleMaterialAtlasRawAtOffset(triIndex, slot * 2u, uv0, uv1);
}

fn ddgiMaterialMapChannel(v: vec4f, channel: u32) -> f32 {
  if (channel == 1u) { return v.g; }
  if (channel == 2u) { return v.b; }
  if (channel == 3u) { return v.a; }
  return v.r;
}

fn ddgiSampleMaterialScalarMap(
  triIndex: u32,
  slot: u32,
  channel: u32,
  uv0: vec2f,
  uv1: vec2f,
  fallback: f32,
) -> f32 {
  let texel = ddgiSampleMaterialAtlasRaw(triIndex, slot, uv0, uv1);
  if (texel.x < 0.0) {
    return fallback;
  }
  return clamp(ddgiMaterialMapChannel(texel, channel), 0.0, 1.0);
}

struct DdgiProbeHitMaterial {
  albedo: vec3f,
  roughness: f32,
  metalness: f32,
}

fn ddgiSampleProbeHitMaterial(
  hit: IntersectionResult,
  scalarBaseColor: vec3f,
  scalarRoughness: f32,
  scalarMetalness: f32,
) -> DdgiProbeHitMaterial {
  var out: DdgiProbeHitMaterial;
  out.albedo = scalarBaseColor;
  out.roughness = scalarRoughness;
  out.metalness = scalarMetalness;

  let uvs = ddgiHitMaterialUvs(hit);
  if (uvs.valid == 0u) {
    return out;
  }

  let baseColorTexel = ddgiSampleMaterialAtlasRaw(
    hit.indices.w,
    DDGI_MATERIAL_MAP_SLOT_BASE_COLOR,
    uvs.uv0,
    uvs.uv1,
  );
  if (baseColorTexel.x >= 0.0) {
    out.albedo = scalarBaseColor * baseColorTexel.rgb;
  }
  out.roughness = ddgiSampleMaterialScalarMap(
    hit.indices.w,
    DDGI_MATERIAL_MAP_SLOT_ROUGHNESS,
    1u,
    uvs.uv0,
    uvs.uv1,
    scalarRoughness,
  );
  out.metalness = ddgiSampleMaterialScalarMap(
    hit.indices.w,
    DDGI_MATERIAL_MAP_SLOT_METALLIC,
    2u,
    uvs.uv0,
    uvs.uv1,
    scalarMetalness,
  );
  return out;
}

fn ddgiApplyNormalMapAtOffsetForHit(
  hit: IntersectionResult,
  frameNormal: vec3f,
  fallbackNormal: vec3f,
  normalMapOffset: u32,
  normalScaleOffset: u32,
) -> vec3f {
  let triIndex = hit.indices.w;
  let metaTexel = triIndex * DDGI_MATERIAL_MAP_META_TEXELS_PER_TRI + normalMapOffset;
  let metaDims = textureDimensions(ddgiMaterialMapMeta);
  if (metaTexel + 1u >= metaDims.x * metaDims.y) {
    return fallbackNormal;
  }
  let meta0 = textureLoad(ddgiMaterialMapMeta, ddgiMaterialMetaCoord(metaTexel), 0);
  if (i32(meta0.x) < 0) {
    return fallbackNormal;
  }

  let uvs = ddgiHitMaterialUvs(hit);
  if (uvs.valid == 0u) {
    return fallbackNormal;
  }

  let texelColor = ddgiSampleMaterialAtlasRawAtOffset(
    triIndex,
    normalMapOffset,
    uvs.uv0,
    uvs.uv1,
  );
  if (texelColor.x < 0.0) {
    return fallbackNormal;
  }

  let scaleMeta = textureLoad(
    ddgiMaterialMapMeta,
    ddgiMaterialMetaCoord(triIndex * DDGI_MATERIAL_MAP_META_TEXELS_PER_TRI + normalScaleOffset),
    0,
  );
  let normalScale = max(scaleMeta.x, 0.0);
  let tangentSample = normalize(vec3f(
    (texelColor.r * 2.0 - 1.0) * normalScale,
    (texelColor.g * 2.0 - 1.0) * normalScale,
    texelColor.b * 2.0 - 1.0,
  ));

  let frame = ddgiMaterialTangentFrameForHit(hit, frameNormal, normalMapOffset);
  let perturbed = normalize(frame.tangent * tangentSample.x + frame.bitangent * tangentSample.y + frameNormal * tangentSample.z);
  return select(-perturbed, perturbed, dot(perturbed, frameNormal) >= 0.0);
}

fn ddgiApplyNormalMapForHit(hit: IntersectionResult, baseNormal: vec3f) -> vec3f {
  return ddgiApplyNormalMapAtOffsetForHit(
    hit,
    baseNormal,
    baseNormal,
    DDGI_MATERIAL_MAP_NORMAL_TEXEL_OFFSET,
    DDGI_MATERIAL_MAP_NORMAL_SCALE_TEXEL_OFFSET,
  );
}

fn ddgiApplyBumpMapForHit(hit: IntersectionResult, shadingNormal: vec3f) -> vec3f {
  let triIndex = hit.indices.w;
  let metaTexel = triIndex * DDGI_MATERIAL_MAP_META_TEXELS_PER_TRI + DDGI_MATERIAL_MAP_BUMP_TEXEL_OFFSET;
  let metaDims = textureDimensions(ddgiMaterialMapMeta);
  if (metaTexel + 1u >= metaDims.x * metaDims.y) {
    return shadingNormal;
  }
  let meta0 = textureLoad(ddgiMaterialMapMeta, ddgiMaterialMetaCoord(metaTexel), 0);
  if (i32(meta0.x) < 0) {
    return shadingNormal;
  }

  let scaleMeta = textureLoad(
    ddgiMaterialMapMeta,
    ddgiMaterialMetaCoord(triIndex * DDGI_MATERIAL_MAP_META_TEXELS_PER_TRI + DDGI_MATERIAL_MAP_BUMP_SCALE_TEXEL_OFFSET),
    0,
  );
  let bumpScale = scaleMeta.x;
  if (abs(bumpScale) < 1e-8) {
    return shadingNormal;
  }

  let uvs = ddgiHitMaterialUvs(hit);
  if (uvs.valid == 0u) {
    return shadingNormal;
  }
  let hC = ddgiSampleMaterialAtlasRawAtOffset(
    triIndex,
    DDGI_MATERIAL_MAP_BUMP_TEXEL_OFFSET,
    uvs.uv0,
    uvs.uv1,
  );
  if (hC.x < 0.0) {
    return shadingNormal;
  }

  let atlasDims = textureDimensions(ddgiMaterialTextureAtlas);
  let atlasTexelStep = vec2f(
    1.0 / f32(max(atlasDims.x, 1u)),
    1.0 / f32(max(atlasDims.y, 1u)),
  );
  let bumpTexelStep = vec2f(
    1.0 / max(scaleMeta.y, 1.0),
    1.0 / max(scaleMeta.z, 1.0),
  );
  let texelStep = select(atlasTexelStep, bumpTexelStep, scaleMeta.y > 0.0 && scaleMeta.z > 0.0);
  let hU = ddgiSampleMaterialAtlasRawAtOffsetDelta(
    triIndex,
    DDGI_MATERIAL_MAP_BUMP_TEXEL_OFFSET,
    uvs.uv0,
    uvs.uv1,
    vec2f(texelStep.x, 0.0),
  ).r;
  let hV = ddgiSampleMaterialAtlasRawAtOffsetDelta(
    triIndex,
    DDGI_MATERIAL_MAP_BUMP_TEXEL_OFFSET,
    uvs.uv0,
    uvs.uv1,
    vec2f(0.0, texelStep.y),
  ).r;
  let dhdu = (hU - hC.r) / texelStep.x;
  let dhdv = (hV - hC.r) / texelStep.y;
  let frame = ddgiMaterialTangentFrameForHit(hit, shadingNormal, DDGI_MATERIAL_MAP_BUMP_TEXEL_OFFSET);
  let perturbed = shadingNormal - bumpScale * (dhdu * frame.tangent + dhdv * frame.bitangent);
  let plen = length(perturbed);
  let n = select(shadingNormal, perturbed / plen, plen > 1e-6);
  return select(-n, n, dot(n, shadingNormal) >= 0.0);
}

fn ddgiSampleEmissiveMap(hit: IntersectionResult, scalarEmission: vec3f) -> vec3f {
  let triIndex = hit.indices.w;
  let uvs = ddgiHitMaterialUvs(hit);
  if (uvs.valid == 0u) {
    return scalarEmission;
  }
  let texel = ddgiSampleMaterialAtlasRawAtOffset(
    triIndex,
    DDGI_MATERIAL_MAP_EMISSIVE_TEXEL_OFFSET,
    uvs.uv0,
    uvs.uv1,
  );
  if (texel.x < 0.0) {
    return scalarEmission;
  }
  return scalarEmission * texel.rgb;
}

fn ddgiEmitterSubdivWeightAt(i: u32, j: u32, level: u32) -> vec3f {
  let invLevel = 1.0 / f32(max(level, 1u));
  let u = f32(i) * invLevel;
  let v = f32(j) * invLevel;
  return vec3f(1.0 - u - v, u, v);
}

fn ddgiEmitterParentBarycentricFromLocal(localBary: vec3f, levelF: f32, ordinalF: f32) -> vec3f {
  let level = min(16u, max(1u, u32(round(max(levelF, 1.0)))));
  if (level <= 1u) {
    return localBary;
  }

  let ordinal = u32(round(max(ordinalF, 0.0)));
  var cursor = 0u;
  for (var i = 0u; i < level; i = i + 1u) {
    for (var j = 0u; j < level - i; j = j + 1u) {
      let a = ddgiEmitterSubdivWeightAt(i, j, level);
      let b = ddgiEmitterSubdivWeightAt(i + 1u, j, level);
      let c = ddgiEmitterSubdivWeightAt(i, j + 1u, level);
      if (cursor == ordinal) {
        return localBary.x * a + localBary.y * b + localBary.z * c;
      }
      cursor = cursor + 1u;

      if (i + j < level - 1u) {
        let d = ddgiEmitterSubdivWeightAt(i + 1u, j + 1u, level);
        if (cursor == ordinal) {
          return localBary.x * b + localBary.y * d + localBary.z * c;
        }
        cursor = cursor + 1u;
      }
    }
  }

  return localBary;
}

fn ddgiSampleEmitterLeAtBary(base: u32, localBary: vec3f, scalarEmission: vec3f) -> vec3f {
  let encodedSourceTri = i32(round(ddgiEmitterTris[base + 0u].w));
  if (encodedSourceTri == -1) {
    return scalarEmission;
  }
  let mirroredSourceTri = encodedSourceTri < -1;
  let sourceTri = select(encodedSourceTri, -encodedSourceTri - 2, mirroredSourceTri);
  let triIndex = u32(sourceTri);
  if (triIndex >= arrayLength(&bvh_index)) {
    return scalarEmission;
  }
  let tri = bvh_index[triIndex].xyz;
  if (tri.x >= arrayLength(&bvh_position) || tri.y >= arrayLength(&bvh_position) || tri.z >= arrayLength(&bvh_position) ||
      tri.x >= arrayLength(&bvh_normal) || tri.y >= arrayLength(&bvh_normal) || tri.z >= arrayLength(&bvh_normal)) {
    return scalarEmission;
  }

  var bary = ddgiEmitterParentBarycentricFromLocal(
    localBary,
    ddgiEmitterTris[base + 1u].w,
    ddgiEmitterTris[base + 2u].w,
  );
  if (mirroredSourceTri) {
    bary = vec3f(bary.z, bary.y, bary.x);
  }

  let uv0a = ddgiPackedUvFromVec4(bvh_position[tri.x]);
  let uv0b = ddgiPackedUvFromVec4(bvh_position[tri.y]);
  let uv0c = ddgiPackedUvFromVec4(bvh_position[tri.z]);
  let uv1a = ddgiPackedUvFromVec4(bvh_normal[tri.x]);
  let uv1b = ddgiPackedUvFromVec4(bvh_normal[tri.y]);
  let uv1c = ddgiPackedUvFromVec4(bvh_normal[tri.z]);
  let uv0 = bary.x * uv0a + bary.y * uv0b + bary.z * uv0c;
  let uv1 = bary.x * uv1a + bary.y * uv1b + bary.z * uv1c;
  let texel = ddgiSampleMaterialAtlasRawAtOffset(
    triIndex,
    DDGI_MATERIAL_MAP_EMISSIVE_TEXEL_OFFSET,
    uv0,
    uv1,
  );
  if (texel.x < 0.0) {
    return scalarEmission;
  }
  return scalarEmission * texel.rgb;
}

struct DdgiAlphaCoverage {
  mode: u32,
  coverage: f32,
  cutoff: f32,
}

fn ddgiMaterialAlphaCoverageForHit(hit: IntersectionResult) -> DdgiAlphaCoverage {
  var out: DdgiAlphaCoverage;
  out.mode = 0u;
  out.coverage = 1.0;
  out.cutoff = 0.0;

  let metaDims = textureDimensions(ddgiMaterialMapMeta);
  let metaTexel = hit.indices.w * DDGI_MATERIAL_MAP_META_TEXELS_PER_TRI + DDGI_MATERIAL_MAP_ALPHA_COVERAGE_TEXEL_OFFSET;
  if (metaTexel >= metaDims.x * metaDims.y) {
    return out;
  }
  let coverageMeta = textureLoad(ddgiMaterialMapMeta, ddgiMaterialMetaCoord(metaTexel), 0);
  out.mode = u32(max(coverageMeta.x, 0.0) + 0.5);
  if (out.mode == 0u) {
    return out;
  }

  let uvs = ddgiHitMaterialUvs(hit);
  if (uvs.valid == 0u) {
    return out;
  }
  let baseColorTexel = ddgiSampleMaterialAtlasRaw(hit.indices.w, DDGI_MATERIAL_MAP_SLOT_BASE_COLOR, uvs.uv0, uvs.uv1);
  let baseColorAlpha = select(clamp(baseColorTexel.a, 0.0, 1.0), 1.0, baseColorTexel.x < 0.0);
  let alphaTexel = ddgiSampleMaterialAtlasRaw(hit.indices.w, DDGI_MATERIAL_MAP_SLOT_ALPHA, uvs.uv0, uvs.uv1);
  let alphaMapCoverage = select(clamp(alphaTexel.r, 0.0, 1.0), 1.0, alphaTexel.x < 0.0);
  let opacity = clamp(coverageMeta.y, 0.0, 1.0);
  out.cutoff = clamp(coverageMeta.z, 0.0, 1.0);
  out.coverage = clamp(opacity * baseColorAlpha * alphaMapCoverage, 0.0, 1.0);
  return out;
}

fn ddgiAlphaShadowTransmittanceForHit(hit: IntersectionResult) -> f32 {
  let alpha = ddgiMaterialAlphaCoverageForHit(hit);
  if (alpha.mode == 0u) {
    return 0.0;
  }
  if (alpha.mode == 1u) {
    return select(0.0, 1.0, alpha.coverage < alpha.cutoff);
  }
  if (alpha.mode == 2u) {
    return clamp(1.0 - alpha.coverage, 0.0, 1.0);
  }
  return select(0.0, 1.0, alpha.coverage <= 0.0);
}

// -----------------------------------------------------------------
// BVH traversal — merged world BLAS or TLAS+local BLAS (PR-5.2).
// -----------------------------------------------------------------
const DDGI_TRI_EPSILON: f32 = 1e-5;

fn safe_normalize(v: vec3f) -> vec3f {
  let len2 = dot(v, v);
  if (len2 < 1e-20) { return vec3f(0.0, 1.0, 0.0); }
  return v * inverseSqrt(len2);
}

fn bvhCastShadowDisabledForTri(triIdx: u32) -> bool {
  let matId = bvh_materialId[triIdx];
  return (materials[matId].flags & MATERIAL_FLAG_CAST_SHADOW_DISABLED) != 0u;
}

${BVH_CAST_SHADOW_PREDICATE_WGSL}

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

fn ddgiMaterialAlphaDiscardedForOpaqueProbeHit(hit: IntersectionResult) -> bool {
  let alpha = ddgiMaterialAlphaCoverageForHit(hit);
  if (alpha.mode == 0u) {
    return false;
  }
  if (alpha.mode == 1u) {
    return alpha.coverage < alpha.cutoff;
  }
  if (alpha.mode == 2u) {
    return alpha.coverage < 0.999;
  }
  return alpha.coverage <= 0.0;
}

fn ddgiTraceFirstHitAlphaMaskTextured(ray: Ray) -> IntersectionResult {
  var walkRay = ray;
  var traveled = 0.0;
  let step = max(1e-4, DDGI_TRI_EPSILON * 4.0);

  for (var layer = 0u; layer < 32u; layer = layer + 1u) {
    var hit = traceSceneFirstHitDdgi(walkRay);
    if (!hit.didHit) {
      return hit;
    }
    if (!ddgiMaterialAlphaDiscardedForOpaqueProbeHit(hit)) {
      hit.dist = hit.dist + traveled;
      return hit;
    }
    traveled = traveled + hit.dist + step;
    walkRay.origin = ray.origin + ray.direction * traveled;
  }

  var exhausted = traceSceneFirstHitDdgi(walkRay);
  if (exhausted.didHit && ddgiMaterialAlphaDiscardedForOpaqueProbeHit(exhausted)) {
    exhausted.didHit = false;
  }
  if (exhausted.didHit) {
    exhausted.dist = exhausted.dist + traveled;
  }
  return exhausted;
}

fn bvhTraceAnyCastShadow(origin: vec3f, dir: vec3f, tMax: f32, skipGlass: bool) -> bool {
  if (ddgiTrace.bvhMode == 1u && ddgiTrace.tlasNodeCount > 0u) {
    return traceTlasAnyCastPredicate(
      &tlasNodes,
      &tlasInstanceIndices,
      &tlasBlasRoots,
      &tlasInstanceWorldToLocal,
      &tlasInstanceLocalToWorld,
      ddgiTrace.tlasNodeCount,
      &bvh_index,
      &bvh_position,
      &bvh,
      origin,
      dir,
      tMax,
      DDGI_TRI_EPSILON,
      skipGlass,
    );
  }
  return bvhIntersectAnyAtRootCastPredicate(
    &bvh_index, &bvh_position, &bvh, origin, dir, tMax, DDGI_TRI_EPSILON, skipGlass, 0u,
  );
}

fn ddgiTraceShadowTransmittance(origin: vec3f, dir: vec3f, tMax: f32, skipGlass: bool) -> f32 {
  var walkRay: Ray;
  walkRay.origin = origin;
  walkRay.direction = dir;
  var traveled = 0.0;
  var tau = 1.0;
  let step = max(1e-4, DDGI_TRI_EPSILON * 4.0);

  for (var layer = 0u; layer < 32u; layer = layer + 1u) {
    let remaining = tMax - traveled;
    if (remaining <= step || tau <= 0.001) {
      return clamp(tau, 0.0, 1.0);
    }

    let hit = traceSceneFirstHitDdgi(walkRay);
    if (!hit.didHit || hit.dist >= remaining) {
      return clamp(tau, 0.0, 1.0);
    }

    let matId = bvh_materialId[hit.indices.w];
    let mat = materials[matId];
    if ((mat.flags & MATERIAL_FLAG_CAST_SHADOW_DISABLED) == 0u) {
      if (skipGlass && (mat.flags & MATERIAL_FLAG_IS_GLASS) != 0u) {
        // Let the walk continue through scalar glass, matching the predicate path.
      } else {
        tau = tau * ddgiAlphaShadowTransmittanceForHit(hit);
        if (tau <= 0.001) {
          return 0.0;
        }
      }
    }

    traveled = traveled + hit.dist + step;
    walkRay.origin = origin + dir * traveled;
  }

  if (bvhTraceAnyCastShadow(walkRay.origin, dir, max(0.0, tMax - traveled), skipGlass)) {
    return 0.0;
  }
  return clamp(tau, 0.0, 1.0);
}
`; }

/**
 * Glass-aware sun-visibility helper (traceSunVisibility function).
 *
 * Exported so test harnesses can compile + verify this section independently.
 * Composed first after the header in makeProbeUpdateRaysWGSL.
 */
function makeTraceSunVisibilityWGSL(): string { return /* wgsl */`
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
    let entryPos  = rayOrigin + sunDir * sHit.dist;
    let sMatId = bvh_materialId[sHit.indices.w];
    let sMat   = materials[sMatId];
    if ((sMat.flags & MATERIAL_FLAG_CAST_SHADOW_DISABLED) != 0u) {
      // Primitive castShadow:false: transparent to DDGI shadow/visibility rays.
      rayOrigin = entryPos + sunDir * (gridParams.spacing * 1e-4);
      continue;
    }
    let alphaT = ddgiAlphaShadowTransmittanceForHit(sHit);
    if (alphaT >= 0.999) {
      rayOrigin = entryPos + sunDir * (gridParams.spacing * 1e-4);
      continue;
    }
    if (alphaT > 0.001) {
      visibility = visibility * alphaT;
      if ((sMat.flags & MATERIAL_FLAG_IS_GLASS) == 0u) {
        rayOrigin = entryPos + sunDir * (gridParams.spacing * 1e-4);
        continue;
      }
    } else if ((sMat.flags & MATERIAL_FLAG_IS_GLASS) == 0u) {
      // Opaque occluder — sun is fully blocked.
      return vec3f(0.0);
    }
    // Glass slab — Beer-Lambert transmittance over the geometric path length.
    // Find the exit point by intersecting the continuation ray just past the
    // entry face; distToExit is the in-medium path length. Clamp to thickness
    // (upper bound — guards open/non-watertight glass where the ray would exit
    // far away or miss the far face entirely).
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
function makeDirectLightingWGSL(): string { return /* wgsl */`
fn evalSunLight(lightDir: vec3f, lightColor: vec3f, intensity: f32,
                hitPos: vec3f, hitNormal: vec3f, castShadowDisabled: bool) -> vec3f {
  let nDotL = max(0.0, dot(hitNormal, lightDir));
  if (nDotL < 1e-3) { return vec3f(0.0); }

  // Glass-aware multi-crossing visibility (replaces single-hit bvhTraceFirstHit
  // + binary glass-attenuation pre-fix).
  // M13: normal bias derived from probe spacing to stay scene-scale-agnostic.
  let normalBias = gridParams.spacing * 0.001;
  var visibility = vec3f(1.0);
  if (!castShadowDisabled) {
    visibility = traceSunVisibility(hitPos + hitNormal * normalBias, lightDir);
  }
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

  // Spot cone falloff: light.direction is the spot beam/travel axis (unit for a
  // spot, 0 for a point fixture -> no cone). cosToP = dot(-axis, toLightDir) is
  // 1 on the axis, cos(angle) at the cone edge. The hard-edge branch avoids
  // smoothstep(edge, edge, x) for penumbra=0.
  // Cheap early-out: fully outside the cone contributes nothing, so skip the ray.
  let axisLen2 = dot(light.direction, light.direction);
  var coneFalloff = 1.0;
  if (axisLen2 > 0.25) {
    let cosToP = dot(-light.direction * inverseSqrt(axisLen2), lightDir);
    if (cosToP < light.outerCone) { return vec3f(0.0); }
    if (abs(light.innerCone - light.outerCone) < 1e-5) {
      coneFalloff = 1.0;
    } else {
      coneFalloff = smoothstep(light.outerCone, light.innerCone, cosToP);
    }
    if (coneFalloff <= 0.0) { return vec3f(0.0); }
  }

  // M13: normal bias proportional to probe spacing (scene-scale-agnostic).
  let normalBias_p = gridParams.spacing * 0.001;
  if (!ddgiLightCastShadowDisabled(light)) {
    let shadowOrig = hitPos + hitNormal * normalBias_p;
    let shadowT = ddgiTraceShadowTransmittance(shadowOrig, lightDir, dist - normalBias_p, false);
    if (shadowT <= 0.001) {
      return vec3f(0.0);
    }
    coneFalloff = coneFalloff * shadowT;
  }
  var distanceAttenuation = 1.0;
  if (light.decay > 0.01) {
    if (abs(light.decay - 2.0) < 1e-5) {
      distanceAttenuation = 1.0 / (dist * dist + 1.0);
    } else {
      distanceAttenuation = 1.0 / max(pow(max(dist, 1.0), light.decay), 1.0);
    }
  }
  if (light.distance > 0.0) {
    let x = clamp(1.0 - pow(dist / light.distance, 4.0), 0.0, 1.0);
    distanceAttenuation = distanceAttenuation * x * x;
  }
  let atten = light.intensity * distanceAttenuation;
  return light.color * atten * nDotL * coneFalloff;
}

fn evalDirectLighting(hitPos: vec3f, hitNormal: vec3f) -> vec3f {
  var result = vec3f(0.0);
  for (var li = 0u; li < min(lights.count, MAX_LIGHTS); li = li + 1u) {
    let light = lights.items[li];
    let kind = ddgiLightKind(light);
    if (kind == LIGHT_SUN) {
      let dir = normalize(-light.direction);
      result = result + evalSunLight(
        dir, light.color, light.intensity, hitPos, hitNormal,
        ddgiLightCastShadowDisabled(light));
    } else if (kind == LIGHT_POINT) {
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
    let scalarLe = ddgiEmitterTris[base + 4u].xyz;
    let castShadowDisabled = ddgiEmitterTris[base + 4u].w > 0.5;

    // Jittered uniform area sample (deterministic per emitter index).
    let s0 = pcgHashToF32Ddgi(seed0 ^ (ei * 0x9E3779B9u + 0x1u));
    let s1 = pcgHashToF32Ddgi((seed0 * 7919u) ^ (ei * 0x85EBCA6Bu + 0x2u));
    let su = sqrt(s0);
    let localBary = vec3f(1.0 - su, su * (1.0 - s1), su * s1);
    let pos = localBary.x * vA + localBary.y * vB + localBary.z * vC;
    let Le = ddgiSampleEmitterLeAtBary(base, localBary, scalarLe);

    let toL     = pos - hitPos;
    let dist2   = max(dot(toL, toL), 1e-8);
    let dist    = sqrt(dist2);
    let wi      = toL / dist;
    let cosSurf  = dot(n, wi);
    let cosLight = dot(nrm, -wi);   // front-face only (one-sided emitter)
    if (cosSurf <= 0.0 || cosLight <= 0.0) { continue; }

    let G = (cosSurf * cosLight) / dist2;
    var shadowT = 1.0;
    if (!castShadowDisabled) {
      // Alpha-aware shadow walk — stop just short of the light sample.
      shadowT = ddgiTraceShadowTransmittance(hitPos + n * normalBias, wi, dist - normalBias, false);
      if (shadowT <= 0.001) { continue; }
    }

    Lo = Lo + albedo * 0.31831 * Le * G * area * shadowT;   // 0.31831 = 1/π
  }
  return Lo;
}
`; }

/**
 * Probe world-position helper, sky/env sampling, and the main compute entry point.
 *
 * Exported for independent compile/test. Composed last in makeProbeUpdateRaysWGSL.
 */
function makeProbeMainEntryWGSL(): string { return /* wgsl */`
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

    let hit = ddgiTraceFirstHitAlphaMaskTextured(ray);

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
        out.isGlass       = (mat.flags & MATERIAL_FLAG_IS_GLASS);
      } else {
        let hitWorldPos = probeOrigin + dir * hit.dist;
        let probeMat = ddgiSampleProbeHitMaterial(hit, mat.baseColor, mat.roughness, mat.metalness);

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
        let normalMapped = ddgiApplyNormalMapForHit(hit, smoothNormal);
        let probeNormal = ddgiApplyBumpMapForHit(hit, normalMapped);

        // Direct lighting: analytic sun/fixture lights.
        let direct_analytic = evalDirectLighting(hitWorldPos, probeNormal);
        // H18 Stage 2 — area-emitter NEE. Guard on emitterTriCount>0 is inside the
        // helper; emitter-less scenes get vec3f(0) at zero cost.
        let direct_emitter = ddgiEmitterNEE(
          hitWorldPos, probeNormal, probeMat.albedo,
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
          fix, fiy, probeNormal,
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
        // probeMat carries atlas-sampled roughness/metalness when readable maps
        // are present, falling back to MaterialEntry scalar slots 3/7 from the
        // canonical 64-byte struct. No new bind layout is required because DDGI
        // already binds the material atlas for alpha/emissive probe paths.
        //
        // Cite: Karis (2013) "Real Shading in Unreal Engine 4" §4.4 (split-sum
        // approximation); McGuire et al. (2017) "Real-Time Global Illumination
        // using Precomputed Light Field Probes" (irradiance-cache specular via
        // reflected direction lookup).
        let specularWeight = probeMat.metalness * max(0.0, 1.0 - probeMat.roughness * probeMat.roughness);
        var indirectRadiance: vec3f;
        if (specularWeight > 1e-4 && frameParams.indirectFeedback != 0u) {
          // Reflected probe-ray direction: mirror dir about the hit normal.
          // dir points FROM the probe TO the hit surface — so -dir is the
          // incoming direction at the surface. reflect(-dir, n) gives the
          // outgoing specular direction, which is also the direction we use to
          // query the SH atlas for the radiance arriving from that hemisphere.
          let reflDir = safe_normalize(dir - 2.0 * dot(dir, probeNormal) * probeNormal);
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
          let specularIndirectLo = probeMat.albedo * (specularIrr * (1.0 / PI));
          // Lambertian indirect for the blend reference.
          let lambertianIndirectLo = indirectGated * probeMat.albedo * (1.0 / PI);
          // Blend indirect contribution: lerp from Lambertian to specular.
          indirectRadiance = mix(lambertianIndirectLo, specularIndirectLo, specularWeight);
        } else {
          // Rough/dielectric or no feedback: pure Lambertian indirect.
          indirectRadiance = indirectGated * probeMat.albedo * (1.0 / PI);
        }
        // Direct: Lambertian (analytic lights use nDotL-weighted eval, kept
        // Lambertian since per-probe direct uses the coarse probe-light model).
        let directRadiance = direct * probeMat.albedo * (1.0 / PI);
        var radiance = directRadiance + indirectRadiance;

        if ((mat.flags & MATERIAL_FLAG_IS_GLASS) != 0u) {
          // Glass: add transmitted environment contribution.
          let transmitted = sampleSkyColor(dir) * mat.attenuationColor;
          radiance = mix(radiance, transmitted, mat.transmission * frameParams.glassMixScale);
        }

        // H18 — direct probe hits on plain emissive materials carry their
        // packed surface emission. Explicit rect/disc/mesh-area emitters are
        // handled by ddgiEmitterNEE above; this closes the non-emitter
        // material-emissive path without requiring hosts to author duplicate
        // mesh-area emitters for camera-visible glowing surfaces.
        let scalarSurfaceEmission = vec3f(
          max(mat.emissive.r, 0.0),
          max(mat.emissive.g, 0.0),
          max(mat.emissive.b, 0.0),
        );
        let surfaceEmission = ddgiSampleEmissiveMap(hit, scalarSurfaceEmission);
        radiance = radiance + surfaceEmission;

        out.hitRadiance  = radiance;
        out.hitDistance  = hit.dist;
        out.hitNormal    = probeNormal;
        out.hitPosition  = hitWorldPos;
        out.hitMaterialId = matId;
        out.isGlass       = (mat.flags & MATERIAL_FLAG_IS_GLASS);
      }
    }

    let resultIdx = probeIdx * RAYS_PER_PROBE + rayIdx;
    if (resultIdx < arrayLength(&rayResults)) {
      rayResults[resultIdx] = out;
    }
  }
}
`; }
