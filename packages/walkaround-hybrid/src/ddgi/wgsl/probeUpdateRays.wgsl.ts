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
  MATERIAL_OPTICS_WGSL,
  BEER_LAMBERT_WGSL,
  buildMaterialAtlasOffsetConstsWGSL,
} from '@vitrum/shared-bvh';
import { RAYS_PER_PROBE } from '../ddgiConstants.js';
import { WALKAROUND_ENVIRONMENT_RADIANCE_SCALE_WGSL } from '../../environment/environmentRadianceScale.js';
import {
  IRR_PROBE_STATE_LOCAL_X,
  IRR_PROBE_STATE_LOCAL_Y,
  IRR_STRIDE,
} from '../ddgiAtlasLayout.js';
import { DDGI_PROBE_MAX_OFFSET_NORMALIZED } from '../probeState.js';
import { DDGI_SH_WGSL } from './ddgiSH.wgsl.js';
import { DDGI_SAMPLE_WGSL } from '../ddgiSampleWgsl.js';
import { analyticLightFalloffWgsl } from '../../shaders/analyticLightFalloff.wgsl.js';

const WG_SIZE = 32;
const RAYS_PER_THREAD = Math.ceil(RAYS_PER_PROBE / WG_SIZE);

// 62-texel material-atlas offset ABI — single-sourced in @vitrum/shared-bvh
// (T4-2, 2026-07-20). Reproduces the historical hand-written
// `DDGI_MATERIAL_MAP_*` block byte-for-byte (pinned by the probeUpdateRays
// composed goldens). The ddgiSample* decode fns below stay DDGI-specific — they
// diverge semantically from the shade/RC copies (see materialAtlasOffsets.wgsl.ts).
const DDGI_MATERIAL_ATLAS_OFFSET_CONSTS = buildMaterialAtlasOffsetConstsWGSL({
  prefix: 'DDGI_',
  include: [
    'META_TEXELS_PER_TRI',
    'SLOT_BASE_COLOR',
    'SLOT_ROUGHNESS',
    'SLOT_METALLIC',
    'SLOT_ALPHA',
    'ALPHA_COVERAGE_TEXEL_OFFSET',
    'EMISSIVE_TEXEL_OFFSET',
    'TRANSMISSION_TEXEL_OFFSET',
    'NORMAL_TEXEL_OFFSET',
    'NORMAL_SCALE_TEXEL_OFFSET',
    'LIGHT_TEXEL_OFFSET',
    'LIGHT_INTENSITY_TEXEL_OFFSET',
    'SPECULAR_TEXEL_OFFSET',
    'CLEARCOAT_TEXEL_OFFSET',
    'SHEEN_COLOR_TEXEL_OFFSET',
    'SPECULAR_COLOR_TEXEL_OFFSET',
    'SPECULAR_INTENSITY_TEXEL_OFFSET',
    'CLEARCOAT_FACTOR_TEXEL_OFFSET',
    'CLEARCOAT_ROUGHNESS_TEXEL_OFFSET',
    'SHEEN_COLOR_MAP_TEXEL_OFFSET',
    'SHEEN_ROUGHNESS_TEXEL_OFFSET',
    'CLEARCOAT_NORMAL_TEXEL_OFFSET',
    'CLEARCOAT_NORMAL_SCALE_TEXEL_OFFSET',
    'ANISOTROPY_TEXEL_OFFSET',
    'ANISOTROPY_SCALAR_TEXEL_OFFSET',
    'IRIDESCENCE_TEXEL_OFFSET',
    'IRIDESCENCE_THICKNESS_TEXEL_OFFSET',
    'IRIDESCENCE_SCALAR_TEXEL_OFFSET',
    'THICKNESS_TEXEL_OFFSET',
    'BUMP_TEXEL_OFFSET',
    'BUMP_SCALE_TEXEL_OFFSET',
    'FRONT_LAYER_TEXEL_OFFSET',
    'BACK_LAYER_TEXEL_OFFSET',
    'VOLUME_SCATTERING_TEXEL_OFFSET',
    'FRONT_LAYER_NORMAL_TEXEL_OFFSET',
    'FRONT_LAYER_NORMAL_SCALE_TEXEL_OFFSET',
    'BACK_LAYER_NORMAL_TEXEL_OFFSET',
    'BACK_LAYER_NORMAL_SCALE_TEXEL_OFFSET',
  ],
});

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

${WALKAROUND_ENVIRONMENT_RADIANCE_SCALE_WGSL}
${HAMMERSLEY_WGSL}
${OCTAHEDRAL_WGSL}
${MATERIAL_ENTRY_WGSL}
${BVH_INTERSECT_WGSL}
${TLAS_TRAVERSAL_WGSL}
${DDGI_SH_WGSL}
${DDGI_SAMPLE_WGSL}
${BEER_LAMBERT_WGSL}

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
// All DDGI traversal offsets are expressed as fractions of the probe spacing.
// The host validates spacing as finite and positive before this shader runs.
const DDGI_TRACE_T_MIN_NORMALIZED: f32 = 1.0e-5;
const DDGI_SURFACE_STEP_NORMALIZED: f32 = 4.0e-5;
const DDGI_GLASS_BOUNDARY_STEP_NORMALIZED: f32 = 1.0e-4;

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
// canonical struct carries both, and DDGI now applies Beer-Lambert glass
// visibility plus atlas-backed transmission/thickness modulation when readable
// maps are present.
// -----------------------------------------------------------------

// -----------------------------------------------------------------
// Light uniforms
// -----------------------------------------------------------------
const LIGHT_SUN:   u32 = 0u;
const LIGHT_POINT: u32 = 1u;
const LIGHT_SPOT:  u32 = 2u;
const LIGHT_KIND_MASK: u32 = 0x7fffffffu;
const LIGHT_CAST_SHADOW_DISABLED: u32 = 0x80000000u;
const DDGI_LIGHTS_ABI_MAGIC: u32 = 0x444c4131u;

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

struct DDGIAliasDraw {
  index: u32,
  pmf: f32,
}

fn ddgiLightKind(light: DDGILight) -> u32 {
  return light.kind & LIGHT_KIND_MASK;
}

fn ddgiLightCastShadowDisabled(light: DDGILight) -> bool {
  return (light.kind & LIGHT_CAST_SHADOW_DISABLED) != 0u;
}

fn ddgiPcgHashU32(seed: u32) -> u32 {
  var state = seed * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

// Exact rejection-sampled modulo mapping: every alias-table column has equal
// probability even when count is not a power of two.
fn ddgiAliasColumn(seed: u32, count: u32) -> u32 {
  let threshold = ((0xffffffffu % count) + 1u) % count;
  var word = ddgiPcgHashU32(seed);
  loop {
    if (word >= threshold) { return word % count; }
    word = ddgiPcgHashU32(word ^ 0x27d4eb2du);
  }
  return 0u;
}

fn ddgiRuntimeLightCount() -> u32 {
  let words = arrayLength(&lights);
  if (words < 4u || lights[3u] != DDGI_LIGHTS_ABI_MAGIC) { return 0u; }
  let count = lights[0u];
  let dataEnd = lights[1u] + count * 16u;
  let aliasEnd = lights[2u] + count * 4u;
  return select(0u, count, dataEnd <= words && aliasEnd <= words);
}

fn ddgiLoadLight(index: u32) -> DDGILight {
  let base = lights[1u] + index * 16u;
  var light: DDGILight;
  light.kind = lights[base];
  light.distance = bitcast<f32>(lights[base + 1u]);
  light.decay = bitcast<f32>(lights[base + 2u]);
  light._pad2 = bitcast<f32>(lights[base + 3u]);
  light.position = bitcast<vec3f>(vec3u(lights[base + 4u], lights[base + 5u], lights[base + 6u]));
  light.intensity = bitcast<f32>(lights[base + 7u]);
  light.direction = bitcast<vec3f>(vec3u(lights[base + 8u], lights[base + 9u], lights[base + 10u]));
  light.innerCone = bitcast<f32>(lights[base + 11u]);
  light.color = bitcast<vec3f>(vec3u(lights[base + 12u], lights[base + 13u], lights[base + 14u]));
  light.outerCone = bitcast<f32>(lights[base + 15u]);
  return light;
}

fn ddgiLightAliasDraw(count: u32, seed: u32) -> DDGIAliasDraw {
  let column = ddgiAliasColumn(seed, count);
  let aliasOffset = lights[2u];
  let base = aliasOffset + column * 4u;
  let q = bitcast<f32>(lights[base]);
  let aliasEntry = lights[base + 1u];
  let selected = select(aliasEntry, column, pcgHashToF32Ddgi(seed ^ 0x85ebca6bu) < q);
  var draw: DDGIAliasDraw;
  draw.index = selected;
  draw.pmf = bitcast<f32>(lights[aliasOffset + selected * 4u + 2u]);
  return draw;
}

fn ddgiEmitterAliasDraw(count: u32, seed: u32) -> DDGIAliasDraw {
  let column = ddgiAliasColumn(seed, count);
  let aliasOffset = count * 5u;
  let entry = ddgiEmitterTris[aliasOffset + column];
  let aliasEntry = bitcast<u32>(entry.y);
  let selected = select(aliasEntry, column, pcgHashToF32Ddgi(seed ^ 0x9e3779b9u) < entry.x);
  var draw: DDGIAliasDraw;
  draw.index = selected;
  draw.pmf = ddgiEmitterTris[aliasOffset + selected].z;
  return draw;
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
  hitRadiance: vec3f,
  hitDistance: f32,
  direction:   vec3f,
  _pad0:       f32,
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
@group(1) @binding(1) var<storage, read> lights:   array<u32>;
// H18 Stage 2 — packed area-emitter triangles for per-probe NEE (same layout as
// the RC probeRayCast rc_emitters). Stride: 80 bytes / 20 f32 per tri.
//   [0..2]  vA.xyz + sourceTriIndex (-1 = scalar fallback)
//   [4..6]  vB.xyz + sourceSubdivLevel
//   [8..10] vC.xyz + sourceSubdivOrdinal
//   [12..14] normal.xyz + area (at [15])             [16..18] Le.rgb + emitterFlags
// emitterFlags: bit 0 = castShadowDisabled; bit 1 = twoSided.
// emitterCount (uniform in lights) is reused for the area-emitter count. A
// dedicated u32 is cheaper than a second UBO; it lives in DdgiTraceParams.
@group(1) @binding(2) var<storage, read> ddgiEmitterTris: array<vec4f>;
// DDGI probe-hit / emitter-NEE emission-map subset. These are a DDGI-local copy
// of the walkaround material atlas, used to modulate direct probe hits on
// emissive materials and mapped mesh-area emitter samples whose packed
// sourceTriIndex points back to a material-atlas triangle.
@group(1) @binding(3) var ddgiMaterialTextureAtlas: texture_2d_array<u32>;
@group(1) @binding(4) var ddgiMaterialMapMeta: texture_2d<f32>;
// DDGI-local copy of the authored/generated per-vertex tangent.xyzw stream.
// Zero tangents intentionally mean "derive the frame from UVs".
@group(1) @binding(5) var ddgiBvhTangent: texture_2d<f32>;
// DDGI-local copy of the per-vertex COLOR_0 rgba stream. Missing authored
// colors are white-filled upstream, matching the main material atlas path.
@group(1) @binding(6) var ddgiBvhVertexColor: texture_2d<f32>;

@group(2) @binding(0) var<storage, read_write> rayResults:   array<ProbeRay>;
@group(2) @binding(1) var<storage, read>       activeProbes: array<u32>;
@group(2) @binding(2) var                      irradiancePrev: texture_2d<f32>;
@group(2) @binding(3) var                      irradianceSamp: sampler;
@group(2) @binding(4) var<uniform>             gridParams:   ProbeGridParams;
@group(2) @binding(5) var<uniform>             frameParams:  FrameParams;
// Wave 4 (2026-06-10) — HDRI into DDGI probe misses.
// ddgiEnvMap  : rgba32float equirect radiance (unit-intensity, .rgb). A 1×1
//               placeholder is bound when hasEnv=0. Bilinear textureLoad
//               avoids requiring the optional float32-filterable feature.
@group(2) @binding(6) var                      ddgiEnvMap:   texture_2d<f32>;
@group(2) @binding(8) var                      visibilityPrev: texture_2d<f32>;

${DDGI_MATERIAL_ATLAS_OFFSET_CONSTS}

fn ddgiFeedbackAt(worldPos: vec3f, surfaceNormal: vec3f) -> vec3f {
  return ddgiSample(
    worldPos,
    surfaceNormal,
    irradiancePrev,
    visibilityPrev,
    irradianceSamp,
    gridParams.origin.x,
    gridParams.origin.y,
    gridParams.origin.z,
    gridParams.spacing,
    gridParams.dims.x,
    gridParams.dims.y,
    gridParams.dims.z,
    gridParams.irradianceAtlasW,
    gridParams.irradianceAtlasH,
    gridParams.visibilityAtlasW,
    gridParams.visibilityAtlasH,
  );
}

fn ddgiMaterialMetaRawCoord(texel: u32) -> vec2i {
  let dims = textureDimensions(ddgiMaterialMapMeta);
  let w = max(dims.x, 1u);
  return vec2i(i32(texel % w), i32(texel / w));
}

fn ddgiMaterialMetaExactU32(value: f32) -> u32 {
  if (!(value >= 0.0) || value > 16777216.0 || value != floor(value)) {
    return 0xffffffffu;
  }
  return u32(value);
}

fn ddgiMaterialMetaPhysicalTexel(triIndex: u32, metaOffset: u32) -> u32 {
  let metaDims = textureDimensions(ddgiMaterialMapMeta);
  let totalTexels = metaDims.x * metaDims.y;
  if (totalTexels < 4u) { return totalTexels; }
  let formatHeader = textureLoad(
    ddgiMaterialMapMeta,
    ddgiMaterialMetaRawCoord(totalTexels - 4u),
    0,
  );
  let addressHeader = textureLoad(
    ddgiMaterialMapMeta,
    ddgiMaterialMetaRawCoord(totalTexels - 3u),
    0,
  );
  let materialRecordCount = ddgiMaterialMetaExactU32(formatHeader.y);
  let triangleCount = ddgiMaterialMetaExactU32(formatHeader.z);
  if (
    ddgiMaterialMetaExactU32(formatHeader.x) != 3u ||
    materialRecordCount == 0u ||
    triIndex >= triangleCount ||
    ddgiMaterialMetaExactU32(formatHeader.w) != DDGI_MATERIAL_MAP_META_TEXELS_PER_TRI
  ) {
    return totalTexels;
  }
  let materialBase = ddgiMaterialMetaExactU32(addressHeader.x);
  let triangleMaterialBase = ddgiMaterialMetaExactU32(addressHeader.y);
  let uvAffineBase = ddgiMaterialMetaExactU32(addressHeader.z);
  let activeUvLaneCount = ddgiMaterialMetaExactU32(addressHeader.w);
  if (metaOffset >= 128u && metaOffset < 156u) {
    let laneWord = metaOffset - 128u;
    let lane = laneWord / 2u;
    if (lane >= activeUvLaneCount) { return totalTexels; }
    let physicalTexel = uvAffineBase
      + triIndex * activeUvLaneCount * 2u
      + laneWord;
    return select(totalTexels, physicalTexel, physicalTexel < totalTexels - 4u);
  }
  let idTableTexel = triangleMaterialBase + triIndex / 4u;
  if (idTableTexel >= totalTexels - 4u) { return totalTexels; }
  let ids = textureLoad(
    ddgiMaterialMapMeta,
    ddgiMaterialMetaRawCoord(idTableTexel),
    0,
  );
  let materialId = ddgiMaterialMetaExactU32(ids[triIndex & 3u]);
  if (materialId >= materialRecordCount) { return totalTexels; }
  let physicalTexel = materialBase
    + materialId * DDGI_MATERIAL_MAP_META_TEXELS_PER_TRI
    + metaOffset;
  return select(totalTexels, physicalTexel, physicalTexel < triangleMaterialBase);
}

fn ddgiMaterialMetaCoord(triIndex: u32, metaOffset: u32) -> vec2i {
  return ddgiMaterialMetaRawCoord(
    ddgiMaterialMetaPhysicalTexel(triIndex, metaOffset),
  );
}

fn ddgiMaterialMetaAvailable(triIndex: u32, metaOffset: u32) -> bool {
  let dims = textureDimensions(ddgiMaterialMapMeta);
  let texel = ddgiMaterialMetaPhysicalTexel(triIndex, metaOffset);
  return texel < dims.x * dims.y;
}

fn ddgiMaterialMetaLoadOrZero(triIndex: u32, metaOffset: u32) -> vec4f {
  if (!ddgiMaterialMetaAvailable(triIndex, metaOffset)) {
    return vec4f(0.0);
  }
  return textureLoad(ddgiMaterialMapMeta, ddgiMaterialMetaCoord(triIndex, metaOffset), 0);
}

fn materialOpticalLoad(triIndex: u32, metaOffset: u32) -> vec4f {
  return ddgiMaterialMetaLoadOrZero(triIndex, metaOffset);
}

${MATERIAL_OPTICS_WGSL}

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

fn ddgiWrapMaterialTexelIndex(index: i32, size: i32, mode: u32) -> i32 {
  if (size <= 1) { return 0; }
  if (mode == 1u) { return clamp(index, 0, size - 1); }
  if (mode == 2u) {
    let period = size * 2;
    var x = index % period;
    if (x < 0) { x = x + period; }
    return select(x, period - x - 1, x >= size);
  }
  var x = index % size;
  if (x < 0) { x = x + size; }
  return x;
}

fn ddgiMaterialAtlasFilterMode(samplerPacked: u32, lod: f32) -> u32 {
  let magFilter = (samplerPacked >> 10u) & 0x1u;
  let minFilter = (samplerPacked >> 11u) & 0x1u;
  return select(magFilter, minFilter, lod > 0.0);
}

const DDGI_MATERIAL_ATLAS_ADDRESS_TEXELS_PER_LAYER: u32 = 18u;
const DDGI_MATERIAL_ATLAS_ENCODING_RGBA8_UNORM: u32 = 0u;
const DDGI_MATERIAL_ATLAS_ENCODING_RGBA8_SNORM: u32 = 1u;
const DDGI_MATERIAL_ATLAS_ENCODING_RGBA16_FLOAT: u32 = 2u;
const DDGI_MATERIAL_ATLAS_ENCODING_RGBA16_UNORM: u32 = 3u;
const DDGI_MATERIAL_ATLAS_ENCODING_RGBA16_SNORM: u32 = 4u;
const DDGI_MATERIAL_ATLAS_ENCODING_RGBA32_FLOAT: u32 = 5u;

struct DdgiMaterialAtlasLayerAddress {
  encoding: u32,
  width: u32,
  height: u32,
  mipLevelCount: u32,
  decodeSrgb: u32,
  planeCount: u32,
  recordTexel: u32,
  valid: u32,
};

fn ddgiMaterialAtlasLayerAddress(layer: i32) -> DdgiMaterialAtlasLayerAddress {
  var out: DdgiMaterialAtlasLayerAddress;
  out.valid = 0u;
  if (layer < 0) { return out; }
  let metaDims = textureDimensions(ddgiMaterialMapMeta);
  let totalTexels = metaDims.x * metaDims.y;
  if (totalTexels < 4u) { return out; }
  let directoryHeader = textureLoad(
    ddgiMaterialMapMeta,
    ddgiMaterialMetaRawCoord(totalTexels - 2u),
    0,
  );
  let addressBase = ddgiMaterialMetaExactU32(directoryHeader.x);
  let layerCount = ddgiMaterialMetaExactU32(directoryHeader.y);
  let logicalLayer = u32(layer);
  if (logicalLayer >= layerCount) { return out; }
  let recordTexel = addressBase +
    logicalLayer * DDGI_MATERIAL_ATLAS_ADDRESS_TEXELS_PER_LAYER;
  if (
    recordTexel + DDGI_MATERIAL_ATLAS_ADDRESS_TEXELS_PER_LAYER >
    totalTexels - 4u
  ) {
    return out;
  }
  let info0 = textureLoad(
    ddgiMaterialMapMeta,
    ddgiMaterialMetaRawCoord(recordTexel),
    0,
  );
  let info1 = textureLoad(
    ddgiMaterialMapMeta,
    ddgiMaterialMetaRawCoord(recordTexel + 1u),
    0,
  );
  out.encoding = ddgiMaterialMetaExactU32(info0.x);
  out.width = ddgiMaterialMetaExactU32(info0.y);
  out.height = ddgiMaterialMetaExactU32(info0.z);
  out.mipLevelCount = ddgiMaterialMetaExactU32(info0.w);
  out.decodeSrgb = ddgiMaterialMetaExactU32(info1.x);
  out.planeCount = ddgiMaterialMetaExactU32(info1.y);
  out.recordTexel = recordTexel;
  out.valid = select(
    0u,
    1u,
    out.width > 0u &&
    out.height > 0u &&
    out.mipLevelCount > 0u &&
    (out.planeCount == 1u || out.planeCount == 2u || out.planeCount == 4u),
  );
  return out;
}

fn ddgiMaterialAtlasMapAvailableAtOffset(
  triIndex: u32,
  metaOffset: u32,
) -> bool {
  let metaDims = textureDimensions(ddgiMaterialMapMeta);
  let totalTexels = metaDims.x * metaDims.y;
  let physicalTexel = ddgiMaterialMetaPhysicalTexel(triIndex, metaOffset);
  if (physicalTexel >= totalTexels) { return false; }
  let meta0 = textureLoad(
    ddgiMaterialMapMeta,
    ddgiMaterialMetaRawCoord(physicalTexel),
    0,
  );
  return ddgiMaterialAtlasLayerAddress(i32(meta0.x)).valid != 0u;
}

fn ddgiMaterialAtlasLevelDimensions(
  address: DdgiMaterialAtlasLayerAddress,
  level: u32,
) -> vec2u {
  let divisor = 1u << min(level, 31u);
  return max(vec2u(1u), vec2u(address.width, address.height) / divisor);
}

fn ddgiMaterialAtlasSigned16(value: u32) -> i32 {
  let word = value & 0xffffu;
  return select(i32(word), i32(word) - 65536, word >= 32768u);
}

fn ddgiMaterialAtlasSrgbChannelToLinear(value: f32) -> f32 {
  let c = clamp(value, 0.0, 1.0);
  return select(c / 12.92, pow((c + 0.055) / 1.055, 2.4), c > 0.04045);
}

fn ddgiMaterialAtlasDecodeTexel(
  address: DdgiMaterialAtlasLayerAddress,
  logicalTexel: vec2i,
  level: u32,
) -> vec4f {
  if (address.valid == 0u || level >= address.mipLevelCount) {
    return vec4f(-1.0);
  }
  let mipRecord = textureLoad(
    ddgiMaterialMapMeta,
    ddgiMaterialMetaRawCoord(address.recordTexel + 2u + level),
    0,
  );
  let origin = vec2u(
    ddgiMaterialMetaExactU32(mipRecord.x),
    ddgiMaterialMetaExactU32(mipRecord.y),
  );
  let baseLayer = ddgiMaterialMetaExactU32(mipRecord.z);
  let atlasDims = textureDimensions(ddgiMaterialTextureAtlas);
  let atlasLayers = textureNumLayers(ddgiMaterialTextureAtlas);
  let coord = origin + vec2u(logicalTexel);
  if (
    coord.x >= atlasDims.x ||
    coord.y >= atlasDims.y ||
    baseLayer + address.planeCount > atlasLayers
  ) {
    return vec4f(-1.0);
  }
  let p0 = textureLoad(
    ddgiMaterialTextureAtlas,
    vec2i(coord),
    i32(baseLayer),
    0,
  ).r;
  var value = vec4f(0.0);
  if (address.encoding == DDGI_MATERIAL_ATLAS_ENCODING_RGBA8_UNORM) {
    value = unpack4x8unorm(p0);
  } else if (address.encoding == DDGI_MATERIAL_ATLAS_ENCODING_RGBA8_SNORM) {
    value = unpack4x8snorm(p0);
  } else {
    let p1 = textureLoad(
      ddgiMaterialTextureAtlas,
      vec2i(coord),
      i32(baseLayer + 1u),
      0,
    ).r;
    if (address.encoding == DDGI_MATERIAL_ATLAS_ENCODING_RGBA16_FLOAT) {
      value = vec4f(unpack2x16float(p0), unpack2x16float(p1));
    } else if (address.encoding == DDGI_MATERIAL_ATLAS_ENCODING_RGBA16_UNORM) {
      value = vec4f(
        f32(p0 & 0xffffu),
        f32(p0 >> 16u),
        f32(p1 & 0xffffu),
        f32(p1 >> 16u),
      ) / 65535.0;
    } else if (address.encoding == DDGI_MATERIAL_ATLAS_ENCODING_RGBA16_SNORM) {
      value = max(
        vec4f(
          f32(ddgiMaterialAtlasSigned16(p0)),
          f32(ddgiMaterialAtlasSigned16(p0 >> 16u)),
          f32(ddgiMaterialAtlasSigned16(p1)),
          f32(ddgiMaterialAtlasSigned16(p1 >> 16u)),
        ) / 32767.0,
        vec4f(-1.0),
      );
    } else if (
      address.encoding == DDGI_MATERIAL_ATLAS_ENCODING_RGBA32_FLOAT &&
      address.planeCount == 4u
    ) {
      let p2 = textureLoad(
        ddgiMaterialTextureAtlas,
        vec2i(coord),
        i32(baseLayer + 2u),
        0,
      ).r;
      let p3 = textureLoad(
        ddgiMaterialTextureAtlas,
        vec2i(coord),
        i32(baseLayer + 3u),
        0,
      ).r;
      value = vec4f(
        bitcast<f32>(p0),
        bitcast<f32>(p1),
        bitcast<f32>(p2),
        bitcast<f32>(p3),
      );
    } else {
      return vec4f(-1.0);
    }
  }
  if (address.decodeSrgb != 0u) {
    value = vec4f(
      ddgiMaterialAtlasSrgbChannelToLinear(value.r),
      ddgiMaterialAtlasSrgbChannelToLinear(value.g),
      ddgiMaterialAtlasSrgbChannelToLinear(value.b),
      value.a,
    );
  }
  return value;
}

fn ddgiSampleMaterialAtlasNearestLevel(
  wrapped: vec2f,
  layer: i32,
  level: u32,
) -> vec4f {
  let address = ddgiMaterialAtlasLayerAddress(layer);
  if (address.valid == 0u) { return vec4f(-1.0); }
  let dims = ddgiMaterialAtlasLevelDimensions(address, level);
  let texel = vec2i(
    i32(min(u32(floor(wrapped.x * f32(dims.x))), dims.x - 1u)),
    i32(min(u32(floor(wrapped.y * f32(dims.y))), dims.y - 1u)),
  );
  return ddgiMaterialAtlasDecodeTexel(address, texel, level);
}

fn ddgiSampleMaterialAtlasLinearLevel(
  wrapped: vec2f,
  layer: i32,
  samplerPacked: u32,
  level: u32,
) -> vec4f {
  let address = ddgiMaterialAtlasLayerAddress(layer);
  if (address.valid == 0u) { return vec4f(-1.0); }
  let dims = ddgiMaterialAtlasLevelDimensions(address, level);
  let size = vec2i(i32(dims.x), i32(dims.y));
  let coord = wrapped * vec2f(f32(dims.x), f32(dims.y)) - vec2f(0.5);
  let base = vec2i(i32(floor(coord.x)), i32(floor(coord.y)));
  let fraction = coord - floor(coord);
  let wrapS = samplerPacked & 0x3u;
  let wrapT = (samplerPacked >> 2u) & 0x3u;
  let x0 = ddgiWrapMaterialTexelIndex(base.x, size.x, wrapS);
  let x1 = ddgiWrapMaterialTexelIndex(base.x + 1, size.x, wrapS);
  let y0 = ddgiWrapMaterialTexelIndex(base.y, size.y, wrapT);
  let y1 = ddgiWrapMaterialTexelIndex(base.y + 1, size.y, wrapT);
  let c00 = ddgiMaterialAtlasDecodeTexel(address, vec2i(x0, y0), level);
  let c10 = ddgiMaterialAtlasDecodeTexel(address, vec2i(x1, y0), level);
  let c01 = ddgiMaterialAtlasDecodeTexel(address, vec2i(x0, y1), level);
  let c11 = ddgiMaterialAtlasDecodeTexel(address, vec2i(x1, y1), level);
  return mix(mix(c00, c10, fraction.x), mix(c01, c11, fraction.x), fraction.y);
}

fn ddgiSampleMaterialAtlasLevel(
  wrapped: vec2f,
  layer: i32,
  samplerPacked: u32,
  level: u32,
  lod: f32,
) -> vec4f {
  if (ddgiMaterialAtlasFilterMode(samplerPacked, lod) == 0u) {
    return ddgiSampleMaterialAtlasNearestLevel(wrapped, layer, level);
  }
  return ddgiSampleMaterialAtlasLinearLevel(
    wrapped,
    layer,
    samplerPacked,
    level,
  );
}

fn ddgiSampleMaterialAtlasAtLod(
  wrapped: vec2f,
  layer: i32,
  samplerPacked: u32,
  lod: f32,
) -> vec4f {
  let mipFilter = (samplerPacked >> 8u) & 0x3u;
  let address = ddgiMaterialAtlasLayerAddress(layer);
  if (address.valid == 0u) { return vec4f(-1.0); }
  let lastLevel = address.mipLevelCount - 1u;
  if (mipFilter == 0u || lastLevel == 0u) {
    return ddgiSampleMaterialAtlasLevel(
      wrapped,
      layer,
      samplerPacked,
      0u,
      lod,
    );
  }
  let clampedLod = clamp(lod, 0.0, f32(lastLevel));
  if (mipFilter == 1u) {
    let level = min(u32(floor(clampedLod + 0.5)), lastLevel);
    return ddgiSampleMaterialAtlasLevel(
      wrapped,
      layer,
      samplerPacked,
      level,
      lod,
    );
  }
  let level0 = min(u32(floor(clampedLod)), lastLevel);
  let level1 = min(level0 + 1u, lastLevel);
  let c0 = ddgiSampleMaterialAtlasLevel(
    wrapped,
    layer,
    samplerPacked,
    level0,
    lod,
  );
  let c1 = ddgiSampleMaterialAtlasLevel(
    wrapped,
    layer,
    samplerPacked,
    level1,
    lod,
  );
  return mix(c0, c1, clampedLod - floor(clampedLod));
}

fn ddgiPackedUvFromVec4(v: vec4f) -> vec2f {
  return unpack2x16float(bitcast<u32>(v.w));
}

fn ddgiBvhVertexColorTexel(vertexIndex: u32) -> vec4f {
  let dims = textureDimensions(ddgiBvhVertexColor);
  let width = u32(dims.x);
  let height = u32(dims.y);
  if (width == 0u || height == 0u) {
    return vec4f(1.0);
  }
  let y = vertexIndex / width;
  if (y >= height) {
    return vec4f(1.0);
  }
  return textureLoad(ddgiBvhVertexColor, vec2i(i32(vertexIndex % width), i32(y)), 0);
}

fn ddgiSampleVertexColorForHit(hit: IntersectionResult) -> vec4f {
  let ca = ddgiBvhVertexColorTexel(hit.indices.x);
  let cb = ddgiBvhVertexColorTexel(hit.indices.y);
  let cc = ddgiBvhVertexColorTexel(hit.indices.z);
  return clamp(
    hit.barycoord.x * ca +
    hit.barycoord.y * cb +
    hit.barycoord.z * cc,
    vec4f(0.0),
    vec4f(1.0)
  );
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
  if (!ddgiCanNormalize(b)) {
    let up = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(n.y) > 0.95);
    return ddgiNormalizeOr(cross(n, up), vec3f(0.0, 0.0, 1.0));
  }
  return ddgiNormalizeOr(b, vec3f(0.0, 0.0, 1.0));
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

fn ddgiMaxAbsVec2(v: vec2f) -> f32 {
  return max(abs(v.x), abs(v.y));
}

fn ddgiMaxAbsVec3(v: vec3f) -> f32 {
  return max(abs(v.x), max(abs(v.y), abs(v.z)));
}

fn ddgiTransformDirectionCols(l2w0: vec4f, l2w1: vec4f, l2w2: vec4f, v: vec3f) -> vec3f {
  let matrixScale = max(
    ddgiMaxAbsVec3(l2w0.xyz),
    max(ddgiMaxAbsVec3(l2w1.xyz), ddgiMaxAbsVec3(l2w2.xyz)),
  );
  let vectorScale = ddgiMaxAbsVec3(v);
  if (!(matrixScale > 0.0) || !(vectorScale > 0.0)) {
    return vec3f(0.0);
  }
  let scaledV = v / vectorScale;
  return
    (l2w0.xyz / matrixScale) * scaledV.x +
    (l2w1.xyz / matrixScale) * scaledV.y +
    (l2w2.xyz / matrixScale) * scaledV.z;
}

fn ddgiSmoothShadingNormalForHit(hit: IntersectionResult, geoNormal: vec3f) -> vec3f {
  let n0 = bvh_normal[hit.indices.x].xyz;
  let n1 = bvh_normal[hit.indices.y].xyz;
  let n2 = bvh_normal[hit.indices.z].xyz;
  let blended =
    hit.barycoord.x * n0 +
    hit.barycoord.y * n1 +
    hit.barycoord.z * n2;
  var n = ddgiNormalizeOr(blended, geoNormal);
  let isTlas = ddgiTrace.bvhMode == 1u;
  let tBase = hit.instanceIndex * 4u;
  let tOk = isTlas && tBase + 2u < arrayLength(&tlasInstanceWorldToLocal);
  if (tOk) {
    let w2l0 = tlasInstanceWorldToLocal[tBase];
    let w2l1 = tlasInstanceWorldToLocal[tBase + 1u];
    let w2l2 = tlasInstanceWorldToLocal[tBase + 2u];
    n = tlasTransformNormalFromLocalCols(
      w2l0,
      w2l1,
      w2l2,
      n,
    ) * tlasLinearOrientationSign(w2l0, w2l1, w2l2);
  }
  return n * hit.side;
}

fn ddgiTangentHandednessForLocalToWorld(l2w0: vec4f, l2w1: vec4f, l2w2: vec4f) -> f32 {
  let s0 = ddgiMaxAbsVec3(l2w0.xyz);
  let s1 = ddgiMaxAbsVec3(l2w1.xyz);
  let s2 = ddgiMaxAbsVec3(l2w2.xyz);
  if (!(s0 > 0.0) || !(s1 > 0.0) || !(s2 > 0.0)) {
    return 1.0;
  }
  let det = dot(
    l2w0.xyz / s0,
    cross(l2w1.xyz / s1, l2w2.xyz / s2),
  );
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

  if (ddgiCanNormalize(authoredTangent) && abs(authoredHandedness) > 0.5) {
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
    if (ddgiCanNormalize(authoredTangent)) {
      tangent = ddgiNormalizeOr(authoredTangent, fallbackTangent);
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
  let meta0 = textureLoad(ddgiMaterialMapMeta, ddgiMaterialMetaCoord(triIndex, mapOffset), 0);
  let flags = u32(max(meta0.y, 0.0) + 0.5);
  let texCoord = (flags >> 4u) & 0xFu;

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
  let ta = materialResolveUv(triIndex, texCoord, uv0a, uv1a);
  let tb = materialResolveUv(triIndex, texCoord, uv0b, uv1b);
  let tc = materialResolveUv(triIndex, texCoord, uv0c, uv1c);

  var dp1 = p1.xyz - p0.xyz;
  var dp2 = p2.xyz - p0.xyz;
  let positionScale = max(ddgiMaxAbsVec3(dp1), ddgiMaxAbsVec3(dp2));
  if (positionScale > 0.0) {
    dp1 = dp1 / positionScale;
    dp2 = dp2 / positionScale;
  }
  var duv1 = tb - ta;
  var duv2 = tc - ta;
  let uvScale = max(ddgiMaxAbsVec2(duv1), ddgiMaxAbsVec2(duv2));
  if (uvScale > 0.0) {
    duv1 = duv1 / uvScale;
    duv2 = duv2 / uvScale;
  }
  let det = duv1.x * duv2.y - duv1.y * duv2.x;
  var tangent = dp1;
  var bitangent = dp2;
  if (uvScale > 0.0 && abs(det) > 1e-12) {
    let invDet = 1.0 / det;
    tangent = (dp1 * duv2.y - dp2 * duv1.y) * invDet;
    bitangent = (dp2 * duv1.x - dp1 * duv2.x) * invDet;
  }
  let isTlas = ddgiTrace.bvhMode == 1u;
  let tBase = hit.instanceIndex * 4u;
  let tOk = isTlas && tBase + 2u < arrayLength(&tlasInstanceLocalToWorld);
  if (tOk) {
    tangent = ddgiTransformDirectionCols(
      tlasInstanceLocalToWorld[tBase],
      tlasInstanceLocalToWorld[tBase + 1u],
      tlasInstanceLocalToWorld[tBase + 2u],
      tangent,
    );
    bitangent = ddgiTransformDirectionCols(
      tlasInstanceLocalToWorld[tBase],
      tlasInstanceLocalToWorld[tBase + 1u],
      tlasInstanceLocalToWorld[tBase + 2u],
      bitangent,
    );
  }

  tangent = tangent - frameNormal * dot(frameNormal, tangent);
  if (!ddgiCanNormalize(tangent)) {
    let up = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(frameNormal.y) > 0.95);
    tangent = ddgiNormalizeOr(cross(up, frameNormal), vec3f(1.0, 0.0, 0.0));
  } else {
    tangent = ddgiNormalizeOr(tangent, vec3f(1.0, 0.0, 0.0));
  }

  bitangent = bitangent - frameNormal * dot(frameNormal, bitangent) - tangent * dot(tangent, bitangent);
  if (!ddgiCanNormalize(bitangent)) {
    bitangent = ddgiFallbackBitangentForNormal(frameNormal, tangent);
  } else {
    bitangent = ddgiNormalizeOr(bitangent, ddgiFallbackBitangentForNormal(frameNormal, tangent));
  }

  // Authored glTF tangents describe TEXCOORD_0 only. Keep the derivative
  // frame derived from the selected UV lane whenever a material map addresses
  // TEXCOORD_1 or a higher auxiliary lane.
  if (texCoord == 0u) {
    return ddgiPreferAuthoredTangentFrameForHit(hit, frameNormal, tangent, bitangent);
  }
  return DdgiMaterialTangentFrame(tangent, bitangent);
}

fn ddgiSampleMaterialAtlasRawAtOffsetDelta(
  triIndex: u32,
  metaOffset: u32,
  uv0: vec2f,
  uv1: vec2f,
  transformedDelta: vec2f,
) -> vec4f {
  if (
    !ddgiMaterialMetaAvailable(triIndex, metaOffset) ||
    !ddgiMaterialMetaAvailable(triIndex, metaOffset + 1u)
  ) {
    return vec4f(-1.0);
  }
  let meta0 = textureLoad(ddgiMaterialMapMeta, ddgiMaterialMetaCoord(triIndex, metaOffset), 0);
  let layer = i32(meta0.x);
  let address = ddgiMaterialAtlasLayerAddress(layer);
  if (address.valid == 0u) { return vec4f(-1.0); }
  let wrapPacked = u32(max(meta0.y, 0.0) + 0.5);
  let texCoord = (wrapPacked >> 4u) & 0xFu;
  let uv = materialResolveUv(triIndex, texCoord, uv0, uv1);
  let meta1 = textureLoad(ddgiMaterialMapMeta, ddgiMaterialMetaCoord(triIndex, metaOffset + 1u), 0);
  let scaled = uv * meta1.xy;
  let transformed = vec2f(
    scaled.x * meta1.z - scaled.y * meta1.w,
    scaled.x * meta1.w + scaled.y * meta1.z,
  ) + meta0.zw + transformedDelta;
  let wrapped = ddgiWrapMaterialUv(transformed, wrapPacked);
  // Probe rays have no screen-space derivatives. Use the logical source
  // footprint per angular probe-ray sample as the bounded minification model;
  // authored mip/nearest/linear policy still controls the actual lookup.
  let logicalSize = vec2f(f32(address.width), f32(address.height));
  let angularSamples = sqrt(f32(max(RAYS_PER_PROBE, 1u)));
  let footprint = abs(meta1.xy) * logicalSize / angularSamples;
  let lod = log2(max(max(footprint.x, footprint.y), 1e-8));
  return ddgiSampleMaterialAtlasAtLod(wrapped, layer, wrapPacked, lod);
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
  if (!ddgiMaterialAtlasMapAvailableAtOffset(triIndex, slot * 2u)) {
    return fallback;
  }
  return clamp(fallback * ddgiMaterialMapChannel(texel, channel), 0.0, 1.0);
}

fn ddgiSampleTransmissionMapForHit(hit: IntersectionResult, scalarTransmission: f32) -> f32 {
  let uvs = ddgiHitMaterialUvs(hit);
  if (uvs.valid == 0u) {
    return scalarTransmission;
  }
  let texel = ddgiSampleMaterialAtlasRawAtOffset(
    hit.indices.w,
    DDGI_MATERIAL_MAP_TRANSMISSION_TEXEL_OFFSET,
    uvs.uv0,
    uvs.uv1,
  );
  if (!ddgiMaterialAtlasMapAvailableAtOffset(
    hit.indices.w,
    DDGI_MATERIAL_MAP_TRANSMISSION_TEXEL_OFFSET,
  )) {
    return scalarTransmission;
  }
  return clamp(scalarTransmission * texel.r, 0.0, 1.0);
}

fn ddgiSampleThicknessMapFactorForHit(hit: IntersectionResult) -> vec2f {
  let uvs = ddgiHitMaterialUvs(hit);
  if (uvs.valid == 0u) {
    return vec2f(1.0, 0.0);
  }
  let texel = ddgiSampleMaterialAtlasRawAtOffset(
    hit.indices.w,
    DDGI_MATERIAL_MAP_THICKNESS_TEXEL_OFFSET,
    uvs.uv0,
    uvs.uv1,
  );
  if (!ddgiMaterialAtlasMapAvailableAtOffset(
    hit.indices.w,
    DDGI_MATERIAL_MAP_THICKNESS_TEXEL_OFFSET,
  )) {
    return vec2f(1.0, 0.0);
  }
  return vec2f(clamp(texel.g, 0.0, 1.0), 1.0);
}

fn ddgiApplyThicknessMapToBeerTint(hit: IntersectionResult, beerTint: vec3f) -> vec3f {
  let thickness = ddgiSampleThicknessMapFactorForHit(hit);
  if (thickness.y < 0.5) {
    return beerTint;
  }
  if (thickness.x <= 0.0) { return vec3f(1.0); }
  return pow(max(beerTint, vec3f(0.0)), vec3f(thickness.x));
}

fn ddgiSampleSpecularMeta(triIndex: u32) -> vec4f {
  var color = vec3f(0.04);
  var intensity = 1.0;
  if (ddgiMaterialMetaAvailable(triIndex, DDGI_MATERIAL_MAP_SPECULAR_TEXEL_OFFSET)) {
    let spec = ddgiMaterialMetaLoadOrZero(triIndex, DDGI_MATERIAL_MAP_SPECULAR_TEXEL_OFFSET);
    color = max(spec.rgb, vec3f(0.0));
    intensity = clamp(spec.a, 0.0, 1.0);
  }
  return vec4f(color, intensity);
}

fn ddgiSampleSpecularControls(triIndex: u32, uv0: vec2f, uv1: vec2f) -> vec4f {
  let scalar = ddgiSampleSpecularMeta(triIndex);
  var color = scalar.rgb;
  var intensity = scalar.a;
  let colorMap = ddgiSampleMaterialAtlasRawAtOffset(triIndex, DDGI_MATERIAL_MAP_SPECULAR_COLOR_TEXEL_OFFSET, uv0, uv1);
  if (ddgiMaterialAtlasMapAvailableAtOffset(
    triIndex,
    DDGI_MATERIAL_MAP_SPECULAR_COLOR_TEXEL_OFFSET,
  )) {
    color = max(color * colorMap.rgb, vec3f(0.0));
  }
  let intensityMap = ddgiSampleMaterialAtlasRawAtOffset(triIndex, DDGI_MATERIAL_MAP_SPECULAR_INTENSITY_TEXEL_OFFSET, uv0, uv1);
  if (ddgiMaterialAtlasMapAvailableAtOffset(
    triIndex,
    DDGI_MATERIAL_MAP_SPECULAR_INTENSITY_TEXEL_OFFSET,
  )) {
    intensity = clamp(intensity * intensityMap.a, 0.0, 1.0);
  }
  return vec4f(color, intensity);
}

fn ddgiSampleClearcoatControls(triIndex: u32, uv0: vec2f, uv1: vec2f) -> vec2f {
  let cc = ddgiMaterialMetaLoadOrZero(triIndex, DDGI_MATERIAL_MAP_CLEARCOAT_TEXEL_OFFSET);
  var factor = clamp(cc.x, 0.0, 1.0);
  var roughness = clamp(cc.y, 0.0, 1.0);

  let clearcoatMap = ddgiSampleMaterialAtlasRawAtOffset(triIndex, DDGI_MATERIAL_MAP_CLEARCOAT_FACTOR_TEXEL_OFFSET, uv0, uv1);
  if (ddgiMaterialAtlasMapAvailableAtOffset(
    triIndex,
    DDGI_MATERIAL_MAP_CLEARCOAT_FACTOR_TEXEL_OFFSET,
  )) {
    factor = clamp(factor * clearcoatMap.r, 0.0, 1.0);
  }
  let roughnessMap = ddgiSampleMaterialAtlasRawAtOffset(triIndex, DDGI_MATERIAL_MAP_CLEARCOAT_ROUGHNESS_TEXEL_OFFSET, uv0, uv1);
  if (ddgiMaterialAtlasMapAvailableAtOffset(
    triIndex,
    DDGI_MATERIAL_MAP_CLEARCOAT_ROUGHNESS_TEXEL_OFFSET,
  )) {
    roughness = clamp(roughness * roughnessMap.g, 0.0, 1.0);
  }
  return vec2f(factor, roughness);
}

fn ddgiSampleSheenControls(triIndex: u32, uv0: vec2f, uv1: vec2f) -> vec4f {
  let scalars = ddgiMaterialMetaLoadOrZero(triIndex, DDGI_MATERIAL_MAP_CLEARCOAT_TEXEL_OFFSET);
  let colorMeta = ddgiMaterialMetaLoadOrZero(triIndex, DDGI_MATERIAL_MAP_SHEEN_COLOR_TEXEL_OFFSET);
  var sheenColor = clamp(colorMeta.rgb, vec3f(0.0), vec3f(1.0));
  var sheen = clamp(scalars.z, 0.0, 1.0);

  let colorMap = ddgiSampleMaterialAtlasRawAtOffset(triIndex, DDGI_MATERIAL_MAP_SHEEN_COLOR_MAP_TEXEL_OFFSET, uv0, uv1);
  if (ddgiMaterialAtlasMapAvailableAtOffset(
    triIndex,
    DDGI_MATERIAL_MAP_SHEEN_COLOR_MAP_TEXEL_OFFSET,
  )) {
    sheenColor = clamp(sheenColor * colorMap.rgb, vec3f(0.0), vec3f(1.0));
  }
  return vec4f(sheenColor, sheen);
}

fn ddgiSampleSheenRoughness(triIndex: u32, uv0: vec2f, uv1: vec2f) -> f32 {
  let scalars = ddgiMaterialMetaLoadOrZero(triIndex, DDGI_MATERIAL_MAP_CLEARCOAT_TEXEL_OFFSET);
  var roughness = clamp(scalars.w, 0.0, 1.0);
  let roughnessMap = ddgiSampleMaterialAtlasRawAtOffset(triIndex, DDGI_MATERIAL_MAP_SHEEN_ROUGHNESS_TEXEL_OFFSET, uv0, uv1);
  if (ddgiMaterialAtlasMapAvailableAtOffset(
    triIndex,
    DDGI_MATERIAL_MAP_SHEEN_ROUGHNESS_TEXEL_OFFSET,
  )) {
    roughness = clamp(roughness * roughnessMap.a, 0.0, 1.0);
  }
  return roughness;
}

fn ddgiSampleAnisotropyControls(triIndex: u32, uv0: vec2f, uv1: vec2f) -> vec2f {
  let scalars = ddgiMaterialMetaLoadOrZero(triIndex, DDGI_MATERIAL_MAP_ANISOTROPY_SCALAR_TEXEL_OFFSET);
  var strength = clamp(scalars.x, 0.0, 1.0);
  var rotation = scalars.y;

  let anisoMap = ddgiSampleMaterialAtlasRawAtOffset(triIndex, DDGI_MATERIAL_MAP_ANISOTROPY_TEXEL_OFFSET, uv0, uv1);
  if (ddgiMaterialAtlasMapAvailableAtOffset(
    triIndex,
    DDGI_MATERIAL_MAP_ANISOTROPY_TEXEL_OFFSET,
  )) {
    strength = clamp(strength * anisoMap.b, 0.0, 1.0);
    let direction = anisoMap.rg * 2.0 - vec2f(1.0);
    if (dot(direction, direction) > 0.0) {
      rotation += atan2(direction.y, direction.x);
    }
  }
  return vec2f(strength, rotation);
}

fn ddgiSampleIridescenceControls(triIndex: u32, uv0: vec2f, uv1: vec2f) -> vec4f {
  let scalars = ddgiMaterialMetaLoadOrZero(triIndex, DDGI_MATERIAL_MAP_IRIDESCENCE_SCALAR_TEXEL_OFFSET);
  var factor = clamp(scalars.x, 0.0, 1.0);
  let ior = max(1.0, scalars.y);
  var thicknessMin = max(0.0, scalars.z);
  var thicknessMax = max(0.0, scalars.w);

  let iridescenceMap = ddgiSampleMaterialAtlasRawAtOffset(triIndex, DDGI_MATERIAL_MAP_IRIDESCENCE_TEXEL_OFFSET, uv0, uv1);
  if (ddgiMaterialAtlasMapAvailableAtOffset(
    triIndex,
    DDGI_MATERIAL_MAP_IRIDESCENCE_TEXEL_OFFSET,
  )) {
    factor = clamp(factor * iridescenceMap.r, 0.0, 1.0);
  }
  let thicknessMap = ddgiSampleMaterialAtlasRawAtOffset(triIndex, DDGI_MATERIAL_MAP_IRIDESCENCE_THICKNESS_TEXEL_OFFSET, uv0, uv1);
  if (ddgiMaterialAtlasMapAvailableAtOffset(
    triIndex,
    DDGI_MATERIAL_MAP_IRIDESCENCE_THICKNESS_TEXEL_OFFSET,
  )) {
    let thickness = mix(thicknessMin, thicknessMax, clamp(thicknessMap.g, 0.0, 1.0));
    thicknessMin = thickness;
    thicknessMax = thickness;
    if (thickness <= 0.0) {
      factor = 0.0;
    }
  }
  return vec4f(factor, ior, thicknessMin, thicknessMax);
}

fn ddgiSampleFaceLayerControls(triIndex: u32, isFrontFace: bool) -> vec4f {
  let front = ddgiMaterialMetaLoadOrZero(triIndex, DDGI_MATERIAL_MAP_FRONT_LAYER_TEXEL_OFFSET);
  let back = ddgiMaterialMetaLoadOrZero(triIndex, DDGI_MATERIAL_MAP_BACK_LAYER_TEXEL_OFFSET);
  return select(back, front, isFrontFace);
}

fn ddgiFaceLayerTransmission(layer: vec4f) -> vec3f {
  return clamp(layer.rgb, vec3f(0.0), vec3f(1.0));
}

fn ddgiFaceLayerRoughness(roughness: f32, layer: vec4f) -> f32 {
  return select(roughness, clamp(layer.a, 0.0, 1.0), layer.a >= 0.0);
}

fn ddgiSampleVolumeScatteringControls(triIndex: u32) -> vec4f {
  let scatter = ddgiMaterialMetaLoadOrZero(triIndex, DDGI_MATERIAL_MAP_VOLUME_SCATTERING_TEXEL_OFFSET);
  return vec4f(max(scatter.rgb, vec3f(0.0)), clamp(scatter.a, -0.99, 0.99));
}

fn ddgiHomogeneousBeerTransmittanceRgb(sigmaT: vec3f, distance: f32) -> vec3f {
  return exp(-max(sigmaT, vec3f(0.0)) * max(distance, 0.0));
}

fn ddgiHenyeyGreensteinPhase(cosTheta: f32, g: f32) -> f32 {
  let anisotropy = clamp(g, -0.99, 0.99);
  let denominator = 1.0 + anisotropy * anisotropy -
    2.0 * anisotropy * clamp(cosTheta, -1.0, 1.0);
  return (1.0 - anisotropy * anisotropy) /
    (4.0 * PI * denominator * sqrt(denominator));
}

fn ddgiApplyHomogeneousVolumeSingleScatter(
  radiance: vec3f,
  albedo: vec3f,
  scatter: vec4f,
  pathLength: f32,
  normal: vec3f,
  wo: vec3f,
) -> vec3f {
  let sigmaS = max(scatter.rgb, vec3f(0.0));
  if (all(sigmaS <= vec3f(0.0)) || pathLength <= 0.0) { return radiance; }
  let n = safe_normalize(normal);
  let v = safe_normalize(wo);
  // The incoming radiance is already directionally aggregated. There is no
  // incident direction left with which to evaluate anisotropic HG, so use its
  // isotropic angular average rather than inventing one from the surface normal.
  let phase = ddgiHenyeyGreensteinPhase(0.0, 0.0);
  let source = dot(max(radiance, vec3f(0.0)), vec3f(0.2126, 0.7152, 0.0722)) *
    max(albedo, vec3f(0.0)) * phase;
  let projectedCosine = abs(dot(n, v));
  if (projectedCosine <= 0.0) { return source; }
  let distance = pathLength / projectedCosine;
  let transmittance = ddgiHomogeneousBeerTransmittanceRgb(sigmaS, distance);
  return radiance * transmittance + source * (vec3f(1.0) - transmittance);
}

fn ddgiDistributionGGX(nDotH: f32, rough: f32) -> f32 {
  if (nDotH <= 0.0 || rough <= 0.0) { return 0.0; }
  let alpha = rough * rough;
  let alpha2 = alpha * alpha;
  let d = nDotH * nDotH * (alpha2 - 1.0) + 1.0;
  return alpha2 / (PI * d * d);
}

fn ddgiSmithG1GGX(nDotV: f32, alpha2: f32) -> f32 {
  if (nDotV <= 0.0) { return 0.0; }
  return (2.0 * nDotV) /
    (nDotV + sqrt(alpha2 + (1.0 - alpha2) * nDotV * nDotV));
}

fn ddgiDielectricFresnelExact(cosThetaI: f32, etaIncident: f32, etaTarget: f32) -> f32 {
  if (etaIncident == etaTarget) { return 0.0; }
  let ci = clamp(abs(cosThetaI), 0.0, 1.0);
  let eta = etaIncident / etaTarget;
  let sin2ThetaT = eta * eta * (1.0 - ci * ci);
  if (sin2ThetaT >= 1.0) { return 1.0; }
  let ct = sqrt(1.0 - sin2ThetaT);
  let rs = (etaIncident * ci - etaTarget * ct) /
    (etaIncident * ci + etaTarget * ct);
  let rp = (etaTarget * ci - etaIncident * ct) /
    (etaTarget * ci + etaIncident * ct);
  return 0.5 * (rs * rs + rp * rp);
}

fn ddgiSampleVisibleGgxNormal(n: vec3f, wo: vec3f, rough: f32, xi: vec2f) -> vec3f {
  if (rough <= 0.0) { return n; }
  var up = vec3f(0.0, 1.0, 0.0);
  if (abs(n.y) > 0.999) { up = vec3f(1.0, 0.0, 0.0); }
  let tangent = normalize(cross(up, n));
  let bitangent = cross(n, tangent);
  let woT = vec3f(dot(wo, tangent), dot(wo, bitangent), dot(wo, n));
  let alpha = rough * rough;
  let vh = safe_normalize(vec3f(alpha * woT.x, alpha * woT.y, woT.z));
  let lensq = vh.x * vh.x + vh.y * vh.y;
  var t1 = vec3f(1.0, 0.0, 0.0);
  if (lensq > 0.0) {
    t1 = vec3f(-vh.y, vh.x, 0.0) * inverseSqrt(lensq);
  }
  let t2 = cross(vh, t1);
  let radius = sqrt(clamp(xi.x, 0.0, 1.0));
  let phi = 2.0 * PI * clamp(xi.y, 0.0, 1.0);
  let diskX = radius * cos(phi);
  var diskY = radius * sin(phi);
  let blend = 0.5 * (1.0 + vh.z);
  diskY = (1.0 - blend) * sqrt(max(0.0, 1.0 - diskX * diskX)) +
    blend * diskY;
  let nh = diskX * t1 + diskY * t2 +
    sqrt(max(0.0, 1.0 - diskX * diskX - diskY * diskY)) * vh;
  let wmT = safe_normalize(vec3f(alpha * nh.x, alpha * nh.y, max(0.0, nh.z)));
  return safe_normalize(wmT.x * tangent + wmT.y * bitangent + wmT.z * n);
}

struct DdgiGgxDielectricTransmissionSample {
  direction: vec3f,
  weight: f32,
  transmission: f32,
  valid: u32,
};

fn ddgiSampleGgxDielectricTransmission(
  n: vec3f,
  wo: vec3f,
  rough: f32,
  etaIncident: f32,
  etaTarget: f32,
  xi: vec2f,
) -> DdgiGgxDielectricTransmissionSample {
  var out: DdgiGgxDielectricTransmissionSample;
  out.direction = vec3f(0.0);
  out.weight = 0.0;
  out.transmission = 0.0;
  out.valid = 0u;
  let nDotWo = dot(n, wo);
  if (nDotWo <= 0.0 || etaIncident <= 0.0 || etaTarget <= 0.0) { return out; }
  let eta = etaIncident / etaTarget;
  let etap = etaTarget / etaIncident;
  if (rough <= 0.0) {
    let wi = refract(-wo, n, eta);
    if (dot(wi, wi) <= 0.0) { return out; }
    let interfaceT = 1.0 - ddgiDielectricFresnelExact(nDotWo, etaIncident, etaTarget);
    if (interfaceT <= 0.0) { return out; }
    out.direction = safe_normalize(wi);
    out.weight = interfaceT / (etap * etap);
    out.transmission = interfaceT;
    out.valid = 1u;
    return out;
  }
  let authoredRoughness = clamp(rough, 0.0, 1.0);
  var wm = ddgiSampleVisibleGgxNormal(n, wo, authoredRoughness, xi);
  if (dot(wm, n) < 0.0) { wm = -wm; }
  let woDotM = dot(wo, wm);
  if (woDotM <= 0.0) { return out; }
  let wiRaw = refract(-wo, wm, eta);
  if (dot(wiRaw, wiRaw) <= 0.0) { return out; }
  let wi = safe_normalize(wiRaw);
  let nDotWiAbs = abs(dot(n, wi));
  let wiDotM = dot(wi, wm);
  let denom = wiDotM + woDotM / etap;
  if (dot(n, wi) >= 0.0 || nDotWiAbs <= 0.0 || wiDotM >= 0.0 || denom == 0.0) {
    return out;
  }
  let alpha = authoredRoughness * authoredRoughness;
  let alpha2 = alpha * alpha;
  let D = ddgiDistributionGGX(dot(n, wm), authoredRoughness);
  let G1o = ddgiSmithG1GGX(nDotWo, alpha2);
  let G = G1o * ddgiSmithG1GGX(nDotWiAbs, alpha2);
  let interfaceT = 1.0 - ddgiDielectricFresnelExact(woDotM, etaIncident, etaTarget);
  let denom2 = denom * denom;
  let pdf = (D * G1o * abs(woDotM) / nDotWo) * abs(wiDotM) / denom2;
  if (D <= 0.0 || G <= 0.0 || interfaceT <= 0.0 || pdf <= 0.0) { return out; }
  let ft = interfaceT * D * G *
    abs(wiDotM * woDotM / (nDotWiAbs * nDotWo * denom2)) /
    (etap * etap);
  out.direction = wi;
  out.weight = ft * nDotWiAbs / pdf;
  out.transmission = interfaceT;
  out.valid = 1u;
  return out;
}

struct DdgiProbeHitMaterial {
  albedo: vec3f,
  roughness: f32,
  metalness: f32,
  specular: vec4f,
  clearcoat: vec2f,
  clearcoatNormal: vec3f,
  sheen: vec4f,
  sheenRoughness: f32,
  anisotropy: vec2f,
  iridescence: vec4f,
  layerTransmission: vec3f,
  volumeScattering: vec4f,
  transmission: f32,
  beerTint: vec3f,
  opticalIor: vec3f,
  bulkThickness: f32,
}

fn ddgiSampleProbeHitMaterial(
  hit: IntersectionResult,
  scalarBaseColor: vec3f,
  scalarRoughness: f32,
  scalarMetalness: f32,
  scalarTransmission: f32,
  scalarIor: f32,
  scalarBeerTint: vec3f,
  frameNormal: vec3f,
  shadingNormal: vec3f,
  viewDirection: vec3f,
) -> DdgiProbeHitMaterial {
  var out: DdgiProbeHitMaterial;
  out.albedo = scalarBaseColor;
  out.roughness = scalarRoughness;
  out.metalness = scalarMetalness;
  out.specular = ddgiSampleSpecularMeta(hit.indices.w);
  out.clearcoat = vec2f(0.0);
  out.clearcoatNormal = shadingNormal;
  out.sheen = vec4f(0.0);
  out.sheenRoughness = 0.0;
  out.anisotropy = vec2f(0.0);
  out.iridescence = vec4f(0.0, 1.0, 0.0, 0.0);
  out.layerTransmission = vec3f(1.0);
  out.volumeScattering = vec4f(0.0);
  out.transmission = scalarTransmission;
  out.beerTint = scalarBeerTint;
  let transportIor = select(max(scalarIor, 1.0), 1e6, scalarIor == 0.0);
  out.opticalIor = vec3f(transportIor);
  out.bulkThickness = materialOpticalThickness(hit.indices.w);
  out.clearcoatNormal = ddgiApplyClearcoatNormalMapForHit(hit, frameNormal, shadingNormal);

  let uvs = ddgiHitMaterialUvs(hit);
  if (uvs.valid == 0u) {
    let layerControls = ddgiSampleFaceLayerControls(hit.indices.w, hit.side >= 0.0);
    out.roughness = ddgiFaceLayerRoughness(out.roughness, layerControls);
    out.layerTransmission = ddgiFaceLayerTransmission(layerControls);
    out.volumeScattering = ddgiSampleVolumeScatteringControls(hit.indices.w);
    out.opticalIor = materialDispersionIorRgb(hit.indices.w, transportIor);
    out.beerTint = materialSpectralAttenuation(hit.indices.w, out.bulkThickness, out.beerTint);
    let film = materialThinFilmResponse(
      hit.indices.w,
      hit.side >= 0.0,
      abs(dot(shadingNormal, safe_normalize(viewDirection))),
    );
    if (film.present != 0u) {
      out.specular = vec4f(film.reflectance, 1.0);
      out.iridescence = vec4f(0.0);
      out.layerTransmission = out.layerTransmission * film.transmittance;
    }
    return out;
  }

  let baseColorTexel = ddgiSampleMaterialAtlasRaw(
    hit.indices.w,
    DDGI_MATERIAL_MAP_SLOT_BASE_COLOR,
    uvs.uv0,
    uvs.uv1,
  );
  if (ddgiMaterialAtlasMapAvailableAtOffset(
    hit.indices.w,
    DDGI_MATERIAL_MAP_SLOT_BASE_COLOR * 2u,
  )) {
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
  out.specular = ddgiSampleSpecularControls(hit.indices.w, uvs.uv0, uvs.uv1);
  out.clearcoat = ddgiSampleClearcoatControls(hit.indices.w, uvs.uv0, uvs.uv1);
  out.sheen = ddgiSampleSheenControls(hit.indices.w, uvs.uv0, uvs.uv1);
  out.sheenRoughness = ddgiSampleSheenRoughness(hit.indices.w, uvs.uv0, uvs.uv1);
  out.anisotropy = ddgiSampleAnisotropyControls(hit.indices.w, uvs.uv0, uvs.uv1);
  out.iridescence = ddgiSampleIridescenceControls(hit.indices.w, uvs.uv0, uvs.uv1);
  let layerControls = ddgiSampleFaceLayerControls(hit.indices.w, hit.side >= 0.0);
  out.roughness = ddgiFaceLayerRoughness(out.roughness, layerControls);
  out.layerTransmission = ddgiFaceLayerTransmission(layerControls);
  out.volumeScattering = ddgiSampleVolumeScatteringControls(hit.indices.w);
  out.transmission = ddgiSampleTransmissionMapForHit(hit, scalarTransmission);
  let thicknessMap = ddgiSampleThicknessMapFactorForHit(hit);
  out.bulkThickness = out.bulkThickness * select(1.0, thicknessMap.x, thicknessMap.y >= 0.5);
  out.beerTint = materialSpectralAttenuation(
    hit.indices.w,
    out.bulkThickness,
    ddgiApplyThicknessMapToBeerTint(hit, scalarBeerTint),
  );
  out.opticalIor = materialDispersionIorRgb(hit.indices.w, transportIor);
  let film = materialThinFilmResponse(
    hit.indices.w,
    hit.side >= 0.0,
    abs(dot(shadingNormal, safe_normalize(viewDirection))),
  );
  if (film.present != 0u) {
    out.specular = vec4f(film.reflectance, 1.0);
    out.iridescence = vec4f(0.0);
    out.layerTransmission = out.layerTransmission * film.transmittance;
  }
  return out;
}

fn ddgiProbeDielectricF0(mat: DdgiProbeHitMaterial) -> vec3f {
  return max(mat.specular.rgb, vec3f(0.0)) *
    clamp(mat.specular.a, 0.0, 1.0);
}

fn ddgiProbeMaterialF0(mat: DdgiProbeHitMaterial) -> vec3f {
  let dielectricF0 = ddgiProbeDielectricF0(mat);
  return mix(dielectricF0, mat.albedo, clamp(mat.metalness, 0.0, 1.0));
}

fn ddgiIridescenceModifiedF0(baseF0: vec3f, iridescence: vec4f, vDotH: f32) -> vec3f {
  let factor = clamp(iridescence.x, 0.0, 1.0);
  if (factor <= 0.0) {
    return baseF0;
  }
  let thickness = max(0.0, (iridescence.z + iridescence.w) * 0.5);
  let iorShift = clamp(iridescence.y - 1.0, 0.0, 2.0) * 0.12;
  let phase = thickness * 0.012 + (1.0 - clamp(vDotH, 0.0, 1.0)) * PI;
  let filmTint = clamp(
    0.5 + 0.5 * cos(vec3f(phase, phase + 2.0943951, phase + 4.1887902)) + vec3f(iorShift),
    vec3f(0.0),
    vec3f(1.0),
  );
  let filmF0 = mix(vec3f(0.04), filmTint, clamp(thickness / 1200.0, 0.0, 1.0));
  return clamp(mix(baseF0, filmF0, factor), vec3f(0.0), vec3f(1.0));
}

fn ddgiProbeSpecularTint(mat: DdgiProbeHitMaterial, vDotH: f32) -> vec3f {
  return ddgiIridescenceModifiedF0(ddgiProbeMaterialF0(mat), mat.iridescence, vDotH);
}

fn ddgiProbeBaseSpecularWeight(mat: DdgiProbeHitMaterial) -> f32 {
  let roughFade = max(0.0, 1.0 - mat.roughness * mat.roughness);
  let metalness = clamp(mat.metalness, 0.0, 1.0);
  let metallic = metalness * roughFade;
  // Consume the explicit absolute dielectric F0 for lobe weighting.
  let dielectricF0 = ddgiProbeDielectricF0(mat);
  let dielectric = max(dielectricF0.r, max(dielectricF0.g, dielectricF0.b)) *
    (1.0 - metalness) * roughFade;
  let anisotropy = clamp(mat.anisotropy.x, 0.0, 1.0) * max(metallic, dielectric);
  let iridescence = clamp(mat.iridescence.x, 0.0, 1.0) * roughFade;
  return clamp(max(max(metallic, dielectric), max(anisotropy, iridescence)), 0.0, 1.0);
}

fn ddgiProbeClearcoatWeight(mat: DdgiProbeHitMaterial) -> f32 {
  return clamp(mat.clearcoat.x, 0.0, 1.0) * max(0.0, 1.0 - mat.clearcoat.y * mat.clearcoat.y);
}

fn ddgiProbeSheenWeight(mat: DdgiProbeHitMaterial) -> f32 {
  let colorPower = max(mat.sheen.r, max(mat.sheen.g, mat.sheen.b));
  return clamp(mat.sheen.a, 0.0, 1.0) * colorPower * max(0.0, 1.0 - mat.sheenRoughness * mat.sheenRoughness);
}

fn ddgiProbeExtensionSpecularWeight(mat: DdgiProbeHitMaterial) -> f32 {
  return clamp(
    max(ddgiProbeBaseSpecularWeight(mat), max(ddgiProbeClearcoatWeight(mat), ddgiProbeSheenWeight(mat))),
    0.0,
    1.0,
  );
}

fn ddgiApplyNormalMapAtOffsetForHit(
  hit: IntersectionResult,
  frameNormal: vec3f,
  fallbackNormal: vec3f,
  normalMapOffset: u32,
  normalScaleOffset: u32,
) -> vec3f {
  let triIndex = hit.indices.w;
  if (
    !ddgiMaterialMetaAvailable(triIndex, normalMapOffset) ||
    !ddgiMaterialMetaAvailable(triIndex, normalMapOffset + 1u)
  ) {
    return fallbackNormal;
  }
  if (!ddgiMaterialAtlasMapAvailableAtOffset(triIndex, normalMapOffset)) {
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
  let scaleMeta = textureLoad(
    ddgiMaterialMapMeta,
      ddgiMaterialMetaCoord(triIndex, normalScaleOffset),
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

fn ddgiApplyFaceLayerNormalMapForHit(hit: IntersectionResult, frameNormal: vec3f, fallbackNormal: vec3f) -> vec3f {
  let isFrontFace = hit.side >= 0.0;
  let normalMapOffset = select(
    DDGI_MATERIAL_MAP_BACK_LAYER_NORMAL_TEXEL_OFFSET,
    DDGI_MATERIAL_MAP_FRONT_LAYER_NORMAL_TEXEL_OFFSET,
    isFrontFace,
  );
  let normalScaleOffset = select(
    DDGI_MATERIAL_MAP_BACK_LAYER_NORMAL_SCALE_TEXEL_OFFSET,
    DDGI_MATERIAL_MAP_FRONT_LAYER_NORMAL_SCALE_TEXEL_OFFSET,
    isFrontFace,
  );
  return ddgiApplyNormalMapAtOffsetForHit(hit, frameNormal, fallbackNormal, normalMapOffset, normalScaleOffset);
}

fn ddgiApplyNormalMapForHit(hit: IntersectionResult, baseNormal: vec3f) -> vec3f {
  let baseMapped = ddgiApplyNormalMapAtOffsetForHit(
    hit,
    baseNormal,
    baseNormal,
    DDGI_MATERIAL_MAP_NORMAL_TEXEL_OFFSET,
    DDGI_MATERIAL_MAP_NORMAL_SCALE_TEXEL_OFFSET,
  );
  return ddgiApplyFaceLayerNormalMapForHit(hit, baseNormal, baseMapped);
}

fn ddgiApplyClearcoatNormalMapForHit(hit: IntersectionResult, frameNormal: vec3f, fallbackNormal: vec3f) -> vec3f {
  return ddgiApplyNormalMapAtOffsetForHit(
    hit,
    frameNormal,
    fallbackNormal,
    DDGI_MATERIAL_MAP_CLEARCOAT_NORMAL_TEXEL_OFFSET,
    DDGI_MATERIAL_MAP_CLEARCOAT_NORMAL_SCALE_TEXEL_OFFSET,
  );
}

fn ddgiApplyBumpMapForHit(hit: IntersectionResult, shadingNormal: vec3f) -> vec3f {
  let triIndex = hit.indices.w;
  if (
    !ddgiMaterialMetaAvailable(triIndex, DDGI_MATERIAL_MAP_BUMP_TEXEL_OFFSET) ||
    !ddgiMaterialMetaAvailable(triIndex, DDGI_MATERIAL_MAP_BUMP_TEXEL_OFFSET + 1u)
  ) {
    return shadingNormal;
  }
  if (!ddgiMaterialAtlasMapAvailableAtOffset(
    triIndex,
    DDGI_MATERIAL_MAP_BUMP_TEXEL_OFFSET,
  )) {
    return shadingNormal;
  }

  let scaleMeta = textureLoad(
    ddgiMaterialMapMeta,
      ddgiMaterialMetaCoord(triIndex, DDGI_MATERIAL_MAP_BUMP_SCALE_TEXEL_OFFSET),
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
  let bumpMeta = textureLoad(
    ddgiMaterialMapMeta,
    ddgiMaterialMetaCoord(triIndex, DDGI_MATERIAL_MAP_BUMP_TEXEL_OFFSET),
    0,
  );
  let bumpAddress = ddgiMaterialAtlasLayerAddress(i32(bumpMeta.x));
  let logicalTexelStep = vec2f(
    1.0 / f32(max(bumpAddress.width, 1u)),
    1.0 / f32(max(bumpAddress.height, 1u)),
  );
  let bumpTexelStep = vec2f(
    1.0 / max(scaleMeta.y, 1.0),
    1.0 / max(scaleMeta.z, 1.0),
  );
  let texelStep = select(
    logicalTexelStep,
    bumpTexelStep,
    scaleMeta.y > 0.0 && scaleMeta.z > 0.0,
  );
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
  let n = ddgiNormalizeOr(perturbed, shadingNormal);
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
  if (!ddgiMaterialAtlasMapAvailableAtOffset(
    triIndex,
    DDGI_MATERIAL_MAP_EMISSIVE_TEXEL_OFFSET,
  )) {
    return scalarEmission;
  }
  return scalarEmission * texel.rgb;
}

fn ddgiSampleLightMapIrradiance(hit: IntersectionResult) -> vec3f {
  let triIndex = hit.indices.w;
  let uvs = ddgiHitMaterialUvs(hit);
  if (uvs.valid == 0u) {
    return vec3f(0.0);
  }
  let texel = ddgiSampleMaterialAtlasRawAtOffset(
    triIndex,
    DDGI_MATERIAL_MAP_LIGHT_TEXEL_OFFSET,
    uvs.uv0,
    uvs.uv1,
  );
  if (!ddgiMaterialAtlasMapAvailableAtOffset(
    triIndex,
    DDGI_MATERIAL_MAP_LIGHT_TEXEL_OFFSET,
  )) {
    return vec3f(0.0);
  }
  let intensity = ddgiMaterialMetaLoadOrZero(
    triIndex,
    DDGI_MATERIAL_MAP_LIGHT_INTENSITY_TEXEL_OFFSET,
  ).x;
  return max(texel.rgb, vec3f(0.0)) * max(intensity, 0.0);
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
  if (!ddgiMaterialAtlasMapAvailableAtOffset(
    triIndex,
    DDGI_MATERIAL_MAP_EMISSIVE_TEXEL_OFFSET,
  )) {
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

  if (!ddgiMaterialMetaAvailable(
    hit.indices.w,
    DDGI_MATERIAL_MAP_ALPHA_COVERAGE_TEXEL_OFFSET,
  )) {
    return out;
  }
  let coverageMeta = textureLoad(
    ddgiMaterialMapMeta,
    ddgiMaterialMetaCoord(hit.indices.w, DDGI_MATERIAL_MAP_ALPHA_COVERAGE_TEXEL_OFFSET),
    0,
  );
  out.mode = u32(max(coverageMeta.x, 0.0) + 0.5);
  if (out.mode == 0u) {
    return out;
  }

  let uvs = ddgiHitMaterialUvs(hit);
  if (uvs.valid == 0u) {
    return out;
  }
  let baseColorTexel = ddgiSampleMaterialAtlasRaw(hit.indices.w, DDGI_MATERIAL_MAP_SLOT_BASE_COLOR, uvs.uv0, uvs.uv1);
  let baseColorAlpha = select(
    1.0,
    clamp(baseColorTexel.a, 0.0, 1.0),
    ddgiMaterialAtlasMapAvailableAtOffset(
      hit.indices.w,
      DDGI_MATERIAL_MAP_SLOT_BASE_COLOR * 2u,
    ),
  );
  let alphaTexel = ddgiSampleMaterialAtlasRaw(hit.indices.w, DDGI_MATERIAL_MAP_SLOT_ALPHA, uvs.uv0, uvs.uv1);
  let alphaMapCoverage = select(
    1.0,
    clamp(alphaTexel.r, 0.0, 1.0),
    ddgiMaterialAtlasMapAvailableAtOffset(
      hit.indices.w,
      DDGI_MATERIAL_MAP_SLOT_ALPHA * 2u,
    ),
  );
  let vertexColorAlpha = ddgiSampleVertexColorForHit(hit).a;
  let opacity = clamp(coverageMeta.y, 0.0, 1.0);
  out.cutoff = clamp(coverageMeta.z, 0.0, 1.0);
  out.coverage = clamp(opacity * vertexColorAlpha * baseColorAlpha * alphaMapCoverage, 0.0, 1.0);
  return out;
}

fn ddgiMaterialSideAdmittedForHit(hit: IntersectionResult) -> bool {
  if (hit.side >= 0.0) { return true; }
  let matId = bvh_materialId[hit.indices.w];
  let mat = materials[matId];
  return (mat.flags & MATERIAL_FLAG_DOUBLE_SIDED) != 0u ||
    (mat.flags & MATERIAL_FLAG_IS_GLASS) != 0u ||
    mat.transmission > 0.0;
}

fn ddgiAlphaShadowTransmittanceForHit(hit: IntersectionResult) -> f32 {
  if (!ddgiMaterialSideAdmittedForHit(hit)) {
    return 1.0;
  }
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
fn ddgiFiniteF32(v: f32) -> bool {
  return v == v && abs(v) <= 3.402823e+38;
}

fn ddgiCanNormalize(v: vec3f) -> bool {
  let maxComponent = max(abs(v.x), max(abs(v.y), abs(v.z)));
  if (!ddgiFiniteF32(maxComponent) || !(maxComponent > 0.0)) {
    return false;
  }
  let scaled = v / maxComponent;
  let scaledLen2 = dot(scaled, scaled);
  return ddgiFiniteF32(scaledLen2) && scaledLen2 > 0.0;
}

fn ddgiNormalizeOr(v: vec3f, fallback: vec3f) -> vec3f {
  if (ddgiCanNormalize(v)) {
    let scaled = v / max(abs(v.x), max(abs(v.y), abs(v.z)));
    return scaled * inverseSqrt(dot(scaled, scaled));
  }
  if (ddgiCanNormalize(fallback)) {
    let scaledFallback =
      fallback /
      max(abs(fallback.x), max(abs(fallback.y), abs(fallback.z)));
    return scaledFallback * inverseSqrt(dot(scaledFallback, scaledFallback));
  }
  return vec3f(0.0, 1.0, 0.0);
}

fn ddgiLengthOrZero(v: vec3f) -> f32 {
  let maxComponent = max(abs(v.x), max(abs(v.y), abs(v.z)));
  if (!ddgiFiniteF32(maxComponent) || !(maxComponent > 0.0)) {
    return 0.0;
  }
  let scaled = v / maxComponent;
  let result = maxComponent * sqrt(dot(scaled, scaled));
  return select(0.0, result, ddgiFiniteF32(result) && result > 0.0);
}

fn ddgiProbeDistance(normalizedDistance: f32) -> f32 {
  let distance = gridParams.spacing * normalizedDistance;
  return select(0.0, distance, ddgiFiniteF32(distance) && distance > 0.0);
}

fn safe_normalize(v: vec3f) -> vec3f {
  return ddgiNormalizeOr(v, vec3f(0.0, 1.0, 0.0));
}

fn traceSceneFirstHitDdgi(ray: Ray) -> IntersectionResult {
  let traceTMin = ddgiProbeDistance(DDGI_TRACE_T_MIN_NORMALIZED);
  if (ddgiTrace.bvhMode == 1u && ddgiTrace.tlasNodeCount > 0u) {
    return traceTlasFirstHit(
      ddgiTrace.tlasNodeCount,
      ray,
      traceTMin,
    );
  }
  return bvhIntersectFirstHit(ray, traceTMin);
}

fn ddgiAlphaBlendCoverageHash(hit: IntersectionResult, ray: Ray, layer: u32) -> f32 {
  let originBits = bitcast<vec3u>(ray.origin);
  let directionBits = bitcast<vec3u>(ray.direction);
  var seed =
    (hit.indices.w * 2654435761u) ^
    (hit.instanceIndex * 1597334677u) ^
    (originBits.x * 2246822519u) ^
    (originBits.y * 3266489917u) ^
    (originBits.z * 668265263u) ^
    (directionBits.x * 374761393u) ^
    (directionBits.y * 1274126177u) ^
    (directionBits.z * 1431374977u) ^
    (layer * 0x9e3779b9u) ^
    (frameParams.frameIndex * 0x85ebca6bu);
  seed = seed * 747796405u + 2891336453u;
  let word = ((seed >> ((seed >> 28u) + 4u)) ^ seed) * 277803737u;
  return f32((word >> 22u) ^ word) / 4294967296.0;
}

fn ddgiMaterialAlphaDiscardedForProbeHit(hit: IntersectionResult, ray: Ray, layer: u32) -> bool {
  if (!ddgiMaterialSideAdmittedForHit(hit)) {
    return true;
  }
  let alpha = ddgiMaterialAlphaCoverageForHit(hit);
  if (alpha.mode == 0u) {
    return false;
  }
  if (alpha.mode == 1u) {
    return alpha.coverage < alpha.cutoff;
  }
  if (alpha.mode == 2u) {
    return alpha.coverage < 1.0 && ddgiAlphaBlendCoverageHash(hit, ray, layer) >= alpha.coverage;
  }
  return alpha.coverage <= 0.0;
}

fn ddgiTraceFirstHitAlphaMaskTextured(ray: Ray) -> IntersectionResult {
  var walkRay = ray;
  var traveled = 0.0;
  let step = ddgiProbeDistance(DDGI_SURFACE_STEP_NORMALIZED);
  if (!(step > 0.0)) {
    // Invalid/unrepresentable spacing must not advance through alpha surfaces.
    return traceSceneFirstHitDdgi(ray);
  }
  let surfaceBudget = ddgiWorldSurfaceBudget();

  for (var layer = 0u; layer < surfaceBudget; layer = layer + 1u) {
    var hit = traceSceneFirstHitDdgi(walkRay);
    if (!hit.didHit) {
      return hit;
    }
    if (!ddgiMaterialAlphaDiscardedForProbeHit(hit, ray, layer)) {
      hit.dist = hit.dist + traveled;
      return hit;
    }
    traveled = traveled + hit.dist + step;
    walkRay.origin = ray.origin + ray.direction * traveled;
  }

  var exhausted = traceSceneFirstHitDdgi(walkRay);
  // Conservative world-surface-budget overflow blocks instead of leaking.
  if (exhausted.didHit) {
    exhausted.dist = exhausted.dist + traveled;
  }
  return exhausted;
}

fn ddgiWorldSurfaceBudget() -> u32 {
  let triangleCount = arrayLength(&bvh_materialId);
  if (triangleCount == 0u) { return 1u; }
  var instanceCount = 1u;
  if (ddgiTrace.bvhMode == 1u && ddgiTrace.tlasNodeCount > 0u) {
    instanceCount = max(tlasBlasRootCount(), 1u);
  }
  // A TLAS may place the same BLAS many times. A unique-triangle budget is
  // therefore not a valid upper bound on the number of world-space surfaces
  // crossed by a ray. Saturate rather than wrapping hostile scene sizes.
  if (triangleCount > 0xfffffffeu / instanceCount) {
    return 0xffffffffu;
  }
  return triangleCount * instanceCount + 1u;
}

fn ddgiTraceShadowVisibility(origin: vec3f, dir: vec3f, tMax: f32) -> vec3f {
  var walkRay: Ray;
  walkRay.origin = origin;
  walkRay.direction = dir;
  var traveled = 0.0;
  var visibility = vec3f(1.0);
  let step = ddgiProbeDistance(DDGI_SURFACE_STEP_NORMALIZED);
  if (!(step > 0.0)) { return vec3f(0.0); }
  let surfaceBudget = ddgiWorldSurfaceBudget();

  // Bulk glass is owned by an authored material in one TLAS instance. Beer is
  // accumulated over the geometric segment between paired boundaries; scalar
  // transmission is paid only when entering that medium.
  var mediumMaterial: array<u32, 16>;
  var mediumTri: array<u32, 16>;
  var mediumInstance: array<u32, 16>;
  var mediumColor: array<vec3f, 16>;
  var mediumDistance: array<f32, 16>;
  var mediumDepth = 0u;

  for (var layer = 0u; layer < surfaceBudget; layer = layer + 1u) {
    let remaining = tMax - traveled;
    if (remaining <= step || max(max(visibility.x, visibility.y), visibility.z) <= 0.0) {
      return clamp(visibility, vec3f(0.0), vec3f(1.0));
    }

    let hit = traceSceneFirstHitDdgi(walkRay);
    if (!hit.didHit || hit.dist >= remaining) {
      if (mediumDepth > 0u) {
        let top = mediumDepth - 1u;
        let rgbBeer = beerLambertTransmittanceRgb(
          mediumColor[top],
          mediumDistance[top],
          remaining,
        );
        visibility = visibility * materialSpectralAttenuation(
          mediumTri[top],
          remaining,
          rgbBeer,
        );
      }
      return clamp(visibility, vec3f(0.0), vec3f(1.0));
    }

    if (mediumDepth > 0u) {
      let top = mediumDepth - 1u;
      let rgbBeer = beerLambertTransmittanceRgb(
        mediumColor[top],
        mediumDistance[top],
        hit.dist,
      );
      visibility = visibility * materialSpectralAttenuation(
        mediumTri[top],
        hit.dist,
        rgbBeer,
      );
      if (max(max(visibility.x, visibility.y), visibility.z) <= 0.0) {
        return vec3f(0.0);
      }
    }

    let matId = bvh_materialId[hit.indices.w];
    let mat = materials[matId];
    let isGlass = (mat.flags & MATERIAL_FLAG_IS_GLASS) != 0u;
    if ((mat.flags & MATERIAL_FLAG_CAST_SHADOW_DISABLED) == 0u) {
      let alphaT = ddgiAlphaShadowTransmittanceForHit(hit);
      if (!isGlass) {
        if (alphaT <= 0.0) { return vec3f(0.0); }
        visibility = visibility * alphaT;
      } else if (
        hit.side < 0.0 &&
        mediumDepth > 0u &&
        mediumMaterial[mediumDepth - 1u] == matId &&
        mediumInstance[mediumDepth - 1u] == hit.instanceIndex
      ) {
        // An observed entry owns its paired exit even when the exit texel has
        // different alpha coverage. The bulk segment was already charged
        // above; only this interface's authored face layer remains.
        visibility = visibility * ddgiFaceLayerTransmission(
          ddgiSampleFaceLayerControls(hit.indices.w, false),
        );
        mediumDepth = mediumDepth - 1u;
      } else {
        let coverage = clamp(1.0 - alphaT, 0.0, 1.0);
        if (coverage > 0.0) {
          let layerTransmission = ddgiFaceLayerTransmission(
            ddgiSampleFaceLayerControls(hit.indices.w, hit.side >= 0.0),
          );
          let thicknessMap = ddgiSampleThicknessMapFactorForHit(hit);
          let bulkThickness = max(
            mat.thickness * select(1.0, thicknessMap.x, thicknessMap.y > 0.5),
            0.0,
          );
          let boundaryTransmission = clamp(
            ddgiSampleTransmissionMapForHit(hit, mat.transmission),
            0.0,
            1.0,
          );
          let interfaceTransmission =
            layerTransmission * vec3f(boundaryTransmission);

          if (coverage < 1.0 || bulkThickness <= 0.0) {
            // Deterministic alpha-blend visibility is the expected mixture of
            // an uncovered ray and a covered thin interface. Do not place the
            // entire ray inside a bulk medium for fractional coverage.
            visibility = visibility * mix(
              vec3f(1.0),
              interfaceTransmission,
              vec3f(coverage),
            );
          } else if (hit.side >= 0.0) {
            if (mediumDepth >= 16u) {
              return vec3f(0.0);
            }
            visibility = visibility * interfaceTransmission;
            mediumMaterial[mediumDepth] = matId;
            mediumTri[mediumDepth] = hit.indices.w;
            mediumInstance[mediumDepth] = hit.instanceIndex;
            mediumColor[mediumDepth] = clamp(mat.attenuationColor, vec3f(0.0), vec3f(1.0));
            mediumDistance[mediumDepth] = mat.attenuationDistance;
            mediumDepth = mediumDepth + 1u;
          } else if (mediumDepth > 0u) {
            // A back boundary that does not close the active top medium has
            // ambiguous ownership. Never invent an absorption segment.
            return vec3f(0.0);
          } else {
            // The shadow origin is already inside this medium. No entry was
            // observed, so pay scalar transmission exactly once at this exit.
            let rgbBeer = beerLambertTransmittanceRgb(
              clamp(mat.attenuationColor, vec3f(0.0), vec3f(1.0)),
              mat.attenuationDistance,
              hit.dist,
            );
            visibility = visibility * interfaceTransmission *
              materialSpectralAttenuation(
                hit.indices.w,
                hit.dist,
                rgbBeer,
              );
          }
        }
      }
    }

    if (mediumDepth > 0u) {
      let top = mediumDepth - 1u;
      let stepBeer = beerLambertTransmittanceRgb(
        mediumColor[top],
        mediumDistance[top],
        step,
      );
      visibility = visibility * materialSpectralAttenuation(
        mediumTri[top],
        step,
        stepBeer,
      );
    }
    traveled = traveled + hit.dist + step;
    walkRay.origin = origin + dir * traveled;
  }

  // Traversal-budget exhaustion means an unclassified surface remains; never
  // turn that ambiguity into a light leak.
  return vec3f(0.0);
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

// Sun rays use the same boundary/medium invariant as finite-light shadows.
fn traceSunVisibility(origin: vec3f, sunDir: vec3f) -> vec3f {
  return ddgiTraceShadowVisibility(origin, sunDir, 1e15);
}
`; }

/**
 * Analytic-light + area-emitter NEE block (sun + point/spot + mesh-area NEE).
 *
 * Exported for independent compile/test. Composed after makeTraceSunVisibilityWGSL
 * in makeProbeUpdateRaysWGSL.
 */
function makeDirectLightingWGSL(): string { return /* wgsl */`
${analyticLightFalloffWgsl('ddgi')}

fn ddgiSoftSunHashToF32(seedIn: u32) -> f32 {
  var seed = seedIn;
  seed = seed ^ (seed >> 17u);
  seed = seed * 0xBF324C81u;
  seed = seed ^ (seed >> 13u);
  seed = seed * 0x9C7493ADu;
  seed = seed ^ (seed >> 15u);
  return f32(seed >> 8u) * (1.0 / 16777216.0);
}

fn ddgiSoftSunDirection(sunBase: vec3f, angularRadius: f32, hitPos: vec3f) -> vec3f {
  let radius = max(angularRadius, 0.0);
  if (radius <= 0.0) {
    return sunBase;
  }
  let quant = vec3i(floor(hitPos / gridParams.spacing * 1024.0));
  let seed =
    bitcast<u32>(quant.x) * 0x9E3779B9u ^
    bitcast<u32>(quant.y) * 0x85EBCA6Bu ^
    bitcast<u32>(quant.z) * 0xC2B2AE35u ^
    0x44444749u;
  let xi = vec2f(
    ddgiSoftSunHashToF32(seed ^ 0x53474341u),
    ddgiSoftSunHashToF32(seed ^ 0x4f495431u),
  );
  let upRef = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), abs(sunBase.y) < 0.99);
  let tangent = normalize(cross(upRef, sunBase));
  let bitangent = cross(sunBase, tangent);
  let r = radius * sqrt(xi.x);
  let phi = 6.2831853 * xi.y;
  return normalize(sunBase + tangent * (r * cos(phi)) + bitangent * (r * sin(phi)));
}

fn evalSunLight(lightDir: vec3f, lightColor: vec3f, intensity: f32,
                hitPos: vec3f, hitNormal: vec3f, castShadowDisabled: bool) -> vec3f {
  let nDotL = max(0.0, dot(hitNormal, lightDir));
  if (nDotL <= 0.0) { return vec3f(0.0); }

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
  if (!(dist > 0.0)) { return vec3f(0.0); }
  let lightDir = toLight / dist;
  let nDotL = max(0.0, dot(hitNormal, lightDir));
  if (nDotL <= 0.0) { return vec3f(0.0); }

  // Spot cone falloff: light.direction is the spot beam/travel axis (unit for a
  // spot, 0 for a point fixture -> no cone). cosToP = dot(-axis, toLightDir) is
  // 1 on the axis, cos(angle) at the cone edge. The hard-edge branch avoids
  // smoothstep(edge, edge, x) for penumbra=0.
  // Cheap early-out: fully outside the cone contributes nothing, so skip the ray.
  var coneFalloff = 1.0;
  if (ddgiLightKind(light) == LIGHT_SPOT) {
    if (!ddgiCanNormalize(light.direction)) { return vec3f(0.0); }
    coneFalloff = ddgiSpotConeFalloff(
      light.direction, lightDir, light.innerCone, light.outerCone,
    );
    if (coneFalloff <= 0.0) { return vec3f(0.0); }
  }

  // M13: normal bias proportional to probe spacing (scene-scale-agnostic).
  let normalBias_p = gridParams.spacing * 0.001;
  var shadowVisibility = vec3f(1.0);
  if (!ddgiLightCastShadowDisabled(light)) {
    let shadowOrig = hitPos + hitNormal * normalBias_p;
    shadowVisibility = ddgiTraceShadowVisibility(shadowOrig, lightDir, dist - normalBias_p);
    if (max(max(shadowVisibility.x, shadowVisibility.y), shadowVisibility.z) <= 0.0) {
      return vec3f(0.0);
    }
  }
  let distanceAttenuation = ddgiPointSpotAttenuation(
    dist, light.distance, light.decay, normalBias_p * normalBias_p,
  );
  let atten = light.intensity * distanceAttenuation;
  return light.color * atten * nDotL * coneFalloff * shadowVisibility;
}

fn evalDirectLighting(hitPos: vec3f, hitNormal: vec3f, seed0: u32) -> vec3f {
  let lightCount = ddgiRuntimeLightCount();
  if (lightCount == 0u) { return vec3f(0.0); }
  let draw = ddgiLightAliasDraw(lightCount, seed0 ^ 0x3c6ef372u);
  if (draw.pmf <= 0.0) { return vec3f(0.0); }
  let light = ddgiLoadLight(draw.index);
  let kind = ddgiLightKind(light);
  var result = vec3f(0.0);
  if (kind == LIGHT_SUN) {
    if (!ddgiCanNormalize(light.direction)) { return vec3f(0.0); }
    let dir = ddgiSoftSunDirection(
      -ddgiNormalizeOr(light.direction, vec3f(0.0, -1.0, 0.0)),
      light.innerCone,
      hitPos,
    );
    result = evalSunLight(
      dir, light.color, light.intensity, hitPos, hitNormal,
      ddgiLightCastShadowDisabled(light));
  } else if (kind == LIGHT_POINT || kind == LIGHT_SPOT) {
    result = evalPointLight(light, hitPos, hitNormal);
  }
  return result / draw.pmf;
}

// -----------------------------------------------------------------
// H18 Stage 2 — Area-emitter NEE for probe rays
//
// One power-weighted alias sample from the emitter-triangle distribution.
// The appended alias table supplies threshold/alias/represented PMF. Gated on
// ddgiTrace.emitterTriCount > 0 so sun-only scenes are byte-identical.
//
// Incident-irradiance estimator (area form, q = p_select/area):
//   E = Le · (cosSurf · cosLight / dist²) · area/p_select · vis
// The bounce surface converts E to Lo with albedo/π exactly once after all
// direct-light estimators are combined.
// Shadow test: material-aware alpha plus Beer/transmission/thickness glass
// visibility. Bias via the same gridParams.spacing-derived normal offset as the
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

fn ddgiEmitterNEE(hitPos: vec3f, n: vec3f, seed0: u32) -> vec3f {
  let count = ddgiTrace.emitterTriCount;
  if (count == 0u) { return vec3f(0.0); }
  const sampleCount = 1u;
  let normalBias = gridParams.spacing * 0.001;
  var irradiance = vec3f(0.0);
  for (var sampleIndex = 0u; sampleIndex < sampleCount; sampleIndex = sampleIndex + 1u) {
    let sampleSeed = seed0 ^ (sampleIndex * 0x9e3779b9u);
    let draw = ddgiEmitterAliasDraw(count, sampleSeed ^ 0x61c88647u);
    if (draw.pmf <= 0.0) { continue; }
    // Decode the 5-vec4f EmitterTri entry (80 bytes = 20 f32 = 5 vec4f).
    let base = draw.index * 5u;
    let vA  = ddgiEmitterTris[base + 0u].xyz;
    let vB  = ddgiEmitterTris[base + 1u].xyz;
    let vC  = ddgiEmitterTris[base + 2u].xyz;
    let nrm = ddgiEmitterTris[base + 3u].xyz;
    let area = ddgiEmitterTris[base + 3u].w;
    let scalarLe = ddgiEmitterTris[base + 4u].xyz;
    let emitterFlags = u32(max(ddgiEmitterTris[base + 4u].w, 0.0));
    let castShadowDisabled = (emitterFlags & 1u) != 0u;
    let twoSidedEmitter = (emitterFlags & 2u) != 0u;

    let s0 = pcgHashToF32Ddgi(sampleSeed ^ 0x243f6a88u);
    let s1 = pcgHashToF32Ddgi(sampleSeed ^ 0xb7e15162u);
    let su = sqrt(s0);
    let localBary = vec3f(1.0 - su, su * (1.0 - s1), su * s1);
    let pos = localBary.x * vA + localBary.y * vB + localBary.z * vC;
    let Le = ddgiSampleEmitterLeAtBary(base, localBary, scalarLe);

    let toL = pos - hitPos;
    let dist = ddgiLengthOrZero(toL);
    if (!(dist > 0.0)) { continue; }
    let wi = ddgiNormalizeOr(toL, vec3f(0.0, 1.0, 0.0));
    let dist2 = dist * dist;
    if (!ddgiFiniteF32(dist2) || !(dist2 > 0.0)) { continue; }
    let cosSurf  = dot(n, wi);
    let signedCosLight = dot(nrm, -wi);
    let cosLight = select(
      max(signedCosLight, 0.0),
      abs(signedCosLight),
      twoSidedEmitter,
    );
    if (cosSurf <= 0.0 || cosLight <= 0.0) { continue; }

    let G = (cosSurf * cosLight) / dist2;
    var shadowT = vec3f(1.0);
    if (!castShadowDisabled) {
      // Material-aware shadow walk — stop just short of the light sample.
      shadowT = ddgiTraceShadowVisibility(hitPos + n * normalBias, wi, dist - normalBias);
      if (max(max(shadowT.x, shadowT.y), shadowT.z) <= 0.0) { continue; }
    }

    // Return incident irradiance. The bounce surface applies its Lambertian
    // albedo/π exactly once when direct_analytic and direct_emitter are combined.
    irradiance = irradiance + Le * G * area * shadowT /
      (f32(sampleCount) * draw.pmf);
  }
  return irradiance;
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
  let px = probeIdx % gridParams.dims.x;
  let tmp = probeIdx / gridParams.dims.x;
  let py = tmp % gridParams.dims.y;
  let pz = tmp / gridParams.dims.y;
  // Relocation/classification is packed in the otherwise-unused (4,4)
  // irradiance-cell ring texel. Offsets are normalized before f16 storage.
  let state = textureLoad(irradiancePrev, vec2i(
    i32(px * ${IRR_STRIDE}u + ${IRR_PROBE_STATE_LOCAL_X}u),
    i32((py + pz * gridParams.dims.y) * ${IRR_STRIDE}u +
      ${IRR_PROBE_STATE_LOCAL_Y}u),
  ), 0);
  var normalizedOffset = state.xyz;
  let maxOffset = ${DDGI_PROBE_MAX_OFFSET_NORMALIZED};
  let offsetMaxComponent = max(
    abs(normalizedOffset.x),
    max(abs(normalizedOffset.y), abs(normalizedOffset.z)),
  );
  if (!ddgiFiniteF32(offsetMaxComponent)) {
    normalizedOffset = vec3f(0.0);
  } else if (offsetMaxComponent > 0.0) {
    let scaledOffset = normalizedOffset / offsetMaxComponent;
    let scaledOffsetLength = length(scaledOffset);
    if (
      !ddgiFiniteF32(scaledOffsetLength) ||
      !(scaledOffsetLength > 0.0)
    ) {
      normalizedOffset = vec3f(0.0);
    } else if (offsetMaxComponent > maxOffset / scaledOffsetLength) {
      normalizedOffset =
        scaledOffset * ((maxOffset - 1.0e-6) / scaledOffsetLength);
    }
  }
  return
    gridParams.origin +
    vec3f(f32(px), f32(py), f32(pz)) * gridParams.spacing +
    normalizedOffset * gridParams.spacing;
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
//   texel = ddgiSampleEnvironmentBilinear(vec2f(u,v))
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

fn ddgiWrapEnvironmentX(x: i32, width: i32) -> i32 {
  var wrapped = x % width;
  if (wrapped < 0) { wrapped = wrapped + width; }
  return wrapped;
}

// Portable bilinear lookup for rgba32float. Equirect U repeats across the seam
// while V clamps at the poles. Avoiding a sampler binding keeps this path valid
// without the optional float32-filterable WebGPU feature.
fn ddgiSampleEnvironmentBilinear(uv: vec2f) -> vec4f {
  let dims = textureDimensions(ddgiEnvMap, 0);
  let width = i32(dims.x);
  let height = i32(dims.y);
  let coord = uv * vec2f(f32(width), f32(height)) - vec2f(0.5);
  let base = vec2i(i32(floor(coord.x)), i32(floor(coord.y)));
  let residual = coord - floor(coord);
  let x0 = ddgiWrapEnvironmentX(base.x, width);
  let x1 = ddgiWrapEnvironmentX(base.x + 1, width);
  let y0 = clamp(base.y, 0, height - 1);
  let y1 = clamp(base.y + 1, 0, height - 1);
  let c00 = textureLoad(ddgiEnvMap, vec2i(x0, y0), 0);
  let c10 = textureLoad(ddgiEnvMap, vec2i(x1, y0), 0);
  let c01 = textureLoad(ddgiEnvMap, vec2i(x0, y1), 0);
  let c11 = textureLoad(ddgiEnvMap, vec2i(x1, y1), 0);
  return mix(mix(c00, c10, residual.x), mix(c01, c11, residual.x), residual.y);
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
      let texel = ddgiSampleEnvironmentBilinear(vec2f(u, v));
      return walkaroundScaleEnvironmentRadiance(
        texel.rgb,
        frameParams.envIntensity,
      );
    }
  }
  // Procedural sky fallback (no HDRI, or degenerate map dims).
  let above = max(0.0, dir.y);   // 0..1 above horizon
  let below = max(0.0, -dir.y);  // 0..1 below horizon
  // Above-horizon: lerp from horizon (white/neutral) to zenith tint,
  // then scale by the scene's sky irradiance level.
  let horizon = vec3f(0.9, 0.85, 0.75);           // warm neutral horizon (fixed)
  let skyColor = walkaroundScaleEnvironmentRadiance(
    mix(horizon, frameParams.skyTint, above),
    frameParams.skyIrradiance,
  );
  // Below-horizon: dark ground, attenuated so it doesn't dominate.
  let ground  = vec3f(0.1, 0.08, 0.06);           // dark earth (fixed)
  let groundColor = mix(horizon, ground, below) * 0.3;
  return mix(skyColor, groundColor, below);
}

fn ddgiRgbChannel(value: vec3f, channel: u32) -> f32 {
  if (channel == 0u) { return value.r; }
  if (channel == 1u) { return value.g; }
  return value.b;
}

fn ddgiGlassChannelXi(seed: u32, channel: u32, exitInterface: bool) -> vec2f {
  let interfaceSalt = select(0x243f6a88u, 0x082efa98u, exitInterface);
  return vec2f(
    pcgHashToF32Ddgi(seed ^ interfaceSalt ^ (channel * 0x9e3779b9u)),
    pcgHashToF32Ddgi(seed ^ 0xb7e15162u ^ (channel * 0x85ebca6bu) ^
      select(0u, 0xc2b2ae35u, exitInterface)),
  );
}

// Trace one DDGI glass channel through a reciprocal thin sheet or through an
// actual, ownership-paired bulk exit. This is deliberately a one-slab suffix:
// DDGI stores one-bounce radiance, so encountering a different intervening
// boundary is ambiguous and fails closed instead of charging the entry medium
// to an unrelated surface.
fn ddgiTraceGlassChannel(
  entryHit: IntersectionResult,
  entryPos: vec3f,
  incidentDirection: vec3f,
  faceNormal: vec3f,
  entryMaterial: DdgiProbeHitMaterial,
  attenuationColor: vec3f,
  attenuationDistance: f32,
  materialId: u32,
  channel: u32,
  seed: u32,
) -> f32 {
  let etaTarget = ddgiRgbChannel(entryMaterial.opticalIor, channel);
  let entryBtdf = ddgiSampleGgxDielectricTransmission(
    faceNormal,
    -incidentDirection,
    entryMaterial.roughness,
    1.0,
    etaTarget,
    ddgiGlassChannelXi(seed, channel, false),
  );
  if (entryBtdf.valid == 0u) { return 0.0; }

  if (entryMaterial.bulkThickness <= 0.0) {
    let reverseLayer = ddgiSampleFaceLayerControls(
      entryHit.indices.w,
      entryHit.side < 0.0,
    );
    let exitBtdf = ddgiSampleGgxDielectricTransmission(
      faceNormal,
      -entryBtdf.direction,
      ddgiFaceLayerRoughness(entryMaterial.roughness, reverseLayer),
      etaTarget,
      1.0,
      ddgiGlassChannelXi(seed, channel, true),
    );
    if (exitBtdf.valid == 0u) { return 0.0; }
    return ddgiRgbChannel(sampleSkyColor(exitBtdf.direction), channel) *
      entryBtdf.weight *
      exitBtdf.weight *
      ddgiRgbChannel(ddgiFaceLayerTransmission(reverseLayer), channel);
  }

  // A closed bulk walk must begin at the authored front boundary. hit.normal
  // is face-forward by construction, so the signed side field is the only
  // reliable front/back classifier.
  if (entryHit.side < 0.0) { return 0.0; }

  let boundaryStep =
    ddgiProbeDistance(DDGI_GLASS_BOUNDARY_STEP_NORMALIZED);
  if (!(boundaryStep > 0.0)) { return 0.0; }
  var exitRay: Ray;
  exitRay.origin = entryPos + entryBtdf.direction * boundaryStep;
  exitRay.direction = entryBtdf.direction;
  let exitHit = ddgiTraceFirstHitAlphaMaskTextured(exitRay);
  if (!exitHit.didHit || exitHit.side >= 0.0) { return 0.0; }
  let exitMaterialId = bvh_materialId[exitHit.indices.w];
  if (
    exitMaterialId != materialId ||
    exitHit.instanceIndex != entryHit.instanceIndex
  ) {
    return 0.0;
  }

  let exitMat = materials[exitMaterialId];
  if ((exitMat.flags & MATERIAL_FLAG_IS_GLASS) == 0u) { return 0.0; }
  let exitSmoothNormal = ddgiSmoothShadingNormalForHit(exitHit, exitHit.normal);
  let exitMappedNormal = ddgiApplyNormalMapForHit(exitHit, exitSmoothNormal);
  let exitNormalRaw = ddgiApplyBumpMapForHit(exitHit, exitMappedNormal);
  let exitNormal = select(
    -exitNormalRaw,
    exitNormalRaw,
    dot(entryBtdf.direction, exitNormalRaw) < 0.0,
  );
  let exitMaterial = ddgiSampleProbeHitMaterial(
    exitHit,
    exitMat.baseColor,
    exitMat.roughness,
    exitMat.metalness,
    exitMat.transmission,
    exitMat.ior,
    exitMat.attenuationColor,
    exitSmoothNormal,
    exitNormal,
    -entryBtdf.direction,
  );
  let exitBtdf = ddgiSampleGgxDielectricTransmission(
    exitNormal,
    -entryBtdf.direction,
    exitMaterial.roughness,
    etaTarget,
    1.0,
    ddgiGlassChannelXi(seed, channel, true),
  );
  if (exitBtdf.valid == 0u) { return 0.0; }

  let pathLength = boundaryStep + exitHit.dist;
  let rgbBeer = beerLambertTransmittanceRgb(
    clamp(attenuationColor, vec3f(0.0), vec3f(1.0)),
    attenuationDistance,
    pathLength,
  );
  let spectralBeer = materialSpectralAttenuation(
    entryHit.indices.w,
    pathLength,
    rgbBeer,
  );
  return ddgiRgbChannel(sampleSkyColor(exitBtdf.direction), channel) *
    entryBtdf.weight *
    exitBtdf.weight *
    ddgiRgbChannel(exitMaterial.layerTransmission, channel) *
    ddgiRgbChannel(spectralBeer, channel);
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
    // Reserve a transient classification lane for the blend passes. The
    // classifier overwrites the first record for each active probe; initialize
    // every record so malformed/early classifier exits stay conservative.
    out._pad0 = 0.0;

    if (!hit.didHit) {
      out.hitRadiance  = sampleSkyColor(dir);
      out.hitDistance  = BVH_INTERSECT_INFINITY;
    } else {
      let matId = bvh_materialId[hit.indices.w];
      let mat   = materials[matId];

      // Backface hit — encode as negative distance per DDGI paper convention.
      let backface = hit.side < 0.0 &&
        (mat.flags & MATERIAL_FLAG_DOUBLE_SIDED) == 0u &&
        (mat.flags & MATERIAL_FLAG_IS_GLASS) == 0u &&
        mat.transmission <= 0.0;
      if (backface) {
        out.hitRadiance  = vec3f(0.0);
        out.hitDistance  = -hit.dist;
      } else {
        let hitWorldPos = probeOrigin + dir * hit.dist;

        // Smooth normal from barycentric blend, transformed to world for TLAS hits.
        let smoothNormal = ddgiSmoothShadingNormalForHit(hit, hit.normal);
        let normalMapped = ddgiApplyNormalMapForHit(hit, smoothNormal);
        let probeNormal = ddgiApplyBumpMapForHit(hit, normalMapped);
        let probeMat = ddgiSampleProbeHitMaterial(
          hit, mat.baseColor, mat.roughness, mat.metalness, mat.transmission,
          mat.ior, mat.attenuationColor, smoothNormal, probeNormal, -dir,
        );

        // Direct lighting: analytic sun/fixture lights.
        let directSeed = frameParams.frameIndex ^ (probeIdx * 0x9E3779B9u) ^ rayIdx;
        let direct_analytic = evalDirectLighting(
          hitWorldPos, probeNormal, directSeed ^ 0xa511e9b3u,
        );
        // H18 Stage 2 — area-emitter NEE. Guard on emitterTriCount>0 is inside the
        // helper; emitter-less scenes get vec3f(0) at zero cost.
        let direct_emitter = ddgiEmitterNEE(
          hitWorldPos, probeNormal,
          directSeed ^ 0x63d83595u,
        );
        let direct = direct_analytic + direct_emitter;

        // Previous-frame feedback uses the same relocated, active-probe-aware
        // 8-probe trilinear + Chebyshev visibility sampler as receivers. A
        // nearest-probe SH read leaked across walls and made producer/consumer
        // visibility disagree.
        let indirect = ddgiFeedbackAt(hitWorldPos, probeNormal);

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
        //              E_hit = direct + SH irradiance feedback.
        //   blend    : projects radiance and stores cosine-convolved SH
        //              irradiance coefficients.
        //   receiver : evaluates irradiance E and applies (albedo/PI)*E.
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

        // B2 / 3E — Glossy + extension-aware probe bounce: specular/clearcoat/
        // sheen complement via reflected previous-frame SH field.
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
        //   extensionSpecularWeight is bounded [0,1] and includes metallic F0,
        //   KHR_specular dielectric F0, anisotropy/iridescence visibility,
        //   clearcoat factor/roughness, and sheen colour/roughness. This is
        //   still a probe-cache approximation: lobes steer/tint the reflected
        //   SH lookup rather than evaluating a full directional radiance BRDF.
        //
        // Gate: the reflected atlas lookup requires the previous-frame atlas to
        // be populated (indirectFeedback != 0). When direct-only probes are
        // requested (maxBounces == 1, indirectFeedback = 0) the specular
        // complement is also disabled — both paths fall through to the
        // Lambertian-direct-only formula, preserving byte-identity with the
        // pre-B2 path when indirectFeedback = 0.
        //
        // probeMat carries atlas-sampled roughness/metalness/specular/
        // clearcoat/sheen/anisotropy/iridescence controls when atlas-backed maps
        // are present, falling back to MaterialEntry scalar slots and zeroed
        // extension lobes. No new bind layout is required because DDGI already
        // binds the material atlas and tangent texture for probe-hit maps.
        //
        // Cite: Karis (2013) "Real Shading in Unreal Engine 4" §4.4 (split-sum
        // approximation); McGuire et al. (2017) "Real-Time Global Illumination
        // using Precomputed Light Field Probes" (irradiance-cache specular via
        // reflected direction lookup).
        let baseSpecularWeight = ddgiProbeBaseSpecularWeight(probeMat);
        let clearcoatWeight = ddgiProbeClearcoatWeight(probeMat);
        let sheenWeight = ddgiProbeSheenWeight(probeMat);
        let extensionSpecularWeight = ddgiProbeExtensionSpecularWeight(probeMat);
        var indirectRadiance: vec3f;
        if (extensionSpecularWeight > 0.0 && frameParams.indirectFeedback != 0u) {
          // Reflected probe-ray direction: mirror dir about the hit normal.
          // dir points FROM the probe TO the hit surface — so -dir is the
          // incoming direction at the surface. reflect(-dir, n) gives the
          // outgoing specular direction, which is also the direction we use to
          // query the SH atlas for the radiance arriving from that hemisphere.
          let reflDir = safe_normalize(dir - 2.0 * dot(dir, probeNormal) * probeNormal);
          let specularIrr = ddgiFeedbackAt(hitWorldPos, reflDir);
          let clearcoatReflDir = safe_normalize(dir - 2.0 * dot(dir, probeMat.clearcoatNormal) * probeMat.clearcoatNormal);
          let clearcoatIrr = ddgiFeedbackAt(hitWorldPos, clearcoatReflDir);
          let f0Tint = ddgiProbeSpecularTint(probeMat, max(0.0, dot(-dir, probeNormal)));
          let baseSpecularLo = f0Tint * (specularIrr * (1.0 / PI));
          let clearcoatLo = vec3f(0.04) * (clearcoatIrr * (1.0 / PI));
          let sheenLo = clamp(probeMat.sheen.rgb, vec3f(0.0), vec3f(1.0)) * (specularIrr * (1.0 / PI));
          let lobeWeightSum = baseSpecularWeight + clearcoatWeight + sheenWeight;
          let specularIndirectLo =
            (baseSpecularLo * baseSpecularWeight +
             clearcoatLo * clearcoatWeight +
             sheenLo * sheenWeight) / lobeWeightSum;
          // Lambertian indirect for the blend reference.
          let lambertianIndirectLo = indirectGated * probeMat.albedo * (1.0 / PI);
          // Blend indirect contribution: lerp from Lambertian to specular.
          indirectRadiance = mix(lambertianIndirectLo, specularIndirectLo, extensionSpecularWeight);
        } else {
          // Rough/dielectric or no feedback: pure Lambertian indirect.
          indirectRadiance = indirectGated * probeMat.albedo * (1.0 / PI);
        }
        // Direct: Lambertian (analytic lights use nDotL-weighted eval, kept
        // Lambertian since per-probe direct uses the coarse probe-light model).
        let directRadiance = direct * probeMat.albedo * (1.0 / PI);
        var radiance = directRadiance + indirectRadiance;

        if ((mat.flags & MATERIAL_FLAG_IS_GLASS) != 0u) {
          // Thin sheets use a reciprocal virtual exit. Bulk glass traces and
          // validates the actual back boundary, so Beer is charged over the
          // geometric in-medium segment instead of authored thickness.
          let faceNormal = select(-probeNormal, probeNormal, dot(dir, probeNormal) < 0.0);
          let glassSeed = frameParams.frameIndex ^
            (probeIdx * 0x9e3779b9u) ^ (rayIdx * 0x85ebca6bu);
          let transmitted = vec3f(
            ddgiTraceGlassChannel(
              hit, hitWorldPos, dir, faceNormal, probeMat,
              mat.attenuationColor, mat.attenuationDistance,
              matId, 0u, glassSeed,
            ),
            ddgiTraceGlassChannel(
              hit, hitWorldPos, dir, faceNormal, probeMat,
              mat.attenuationColor, mat.attenuationDistance,
              matId, 1u, glassSeed,
            ),
            ddgiTraceGlassChannel(
              hit, hitWorldPos, dir, faceNormal, probeMat,
              mat.attenuationColor, mat.attenuationDistance,
              matId, 2u, glassSeed,
            ),
          );
          radiance = mix(
            radiance,
            transmitted,
            probeMat.transmission * frameParams.glassMixScale,
          );
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
        // Closed transmissive volumes admit their back interface so rays can
        // exit, but a one-sided emissive material still radiates only through
        // its authored front orientation.
        let surfaceEmission = select(
          vec3f(0.0),
          ddgiSampleEmissiveMap(hit, scalarSurfaceEmission),
          hit.side >= 0.0 ||
            (mat.flags & MATERIAL_FLAG_DOUBLE_SIDED) != 0u,
        );
        let bakedOutgoing = probeMat.albedo * (1.0 / PI) *
          ddgiSampleLightMapIrradiance(hit);
        radiance = radiance + surfaceEmission + bakedOutgoing;
        radiance = radiance * probeMat.layerTransmission;
        radiance = ddgiApplyHomogeneousVolumeSingleScatter(
          radiance,
          probeMat.albedo,
          probeMat.volumeScattering,
          probeMat.bulkThickness,
          probeNormal,
          -dir,
        );

        out.hitRadiance  = radiance;
        out.hitDistance  = hit.dist;
      }
    }

    let resultIdx = probeIdx * RAYS_PER_PROBE + rayIdx;
    if (resultIdx < arrayLength(&rayResults)) {
      rayResults[resultIdx] = out;
    }
  }
}
`; }
