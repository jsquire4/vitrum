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

import { HAMMERSLEY_WGSL, PCG_WGSL } from '@vitrum/shared-samplers';
import {
  OCTAHEDRAL_WGSL,
  MATERIAL_ENTRY_WGSL,
  BVH_INTERSECT_WGSL,
  TLAS_TRAVERSAL_WGSL,
  OPTICAL_WATERTIGHT_TRIANGLE_WGSL,
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
import { sceneOpticalTraversalWgslForBindings } from '../../shaders/sceneTraversal.wgsl.js';

const WG_SIZE = 32;
const RAYS_PER_THREAD = Math.ceil(RAYS_PER_PROBE / WG_SIZE);
const DDGI_OPTICAL_SCENE_TRAVERSAL_WGSL =
  sceneOpticalTraversalWgslForBindings();

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
${PCG_WGSL}
${OCTAHEDRAL_WGSL}
${MATERIAL_ENTRY_WGSL}
${BVH_INTERSECT_WGSL}
${TLAS_TRAVERSAL_WGSL}
${OPTICAL_WATERTIGHT_TRIANGLE_WGSL}
${DDGI_SH_WGSL}
${DDGI_SAMPLE_WGSL}
${BEER_LAMBERT_WGSL}

const WG_SIZE: u32       = ${WG_SIZE}u;
const RAYS_PER_PROBE: u32  = ${RAYS_PER_PROBE}u;
const RAYS_PER_THREAD: u32 = ${RAYS_PER_THREAD}u;   // RAYS_PER_PROBE / WG_SIZE
const DDGI_MAX_MATERIALS: u32 = ${maxMaterials}u;
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
// The initial probe-ray exclusive minimum is expressed as a fraction of probe
// spacing. Secondary optical walks keep their origin fixed and advance only
// this exclusive minimum to the exact represented hit distance.
const DDGI_TRACE_T_MIN_NORMALIZED: f32 = 1.0e-5;

// Probe-side glass-transmission perceptual scale lives on the probe-side
// FrameParams UBO (frameParams.glassMixScale); the canonical Cornell-tuned
// default 0.7 is written by ProbeUpdatePass._uploadFrameParams from the
// HybridEngine option glassMixScale. The dielectric estimator keeps exact
// Fresnel reflection, weights the local surface family by (1 - t), and weights
// transmitted transport by t, where
// t = mat.transmission * frameParams.glassMixScale. At the 0.7 default the
// local family therefore retains a 30% envelope without suppressing Fresnel.
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
  let dataOffset = lights[1u];
  let aliasOffset = lights[2u];
  if (
    dataOffset != 4u ||
    aliasOffset < dataOffset || aliasOffset > words ||
    count > (words - dataOffset) / 16u
  ) { return 0u; }
  // The count bound above proves this multiply/add cannot wrap and that the
  // complete record region fits. Require the packer's exact partition so an
  // in-bounds hostile header cannot redirect alias reads into record words.
  let expectedAliasOffset = dataOffset + count * 16u;
  if (
    aliasOffset != expectedAliasOffset ||
    count > (words - aliasOffset) / 4u
  ) { return 0u; }
  return count;
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
  var draw: DDGIAliasDraw;
  draw.index = 0u;
  draw.pmf = 0.0;
  if (count == 0u) { return draw; }
  let column = ddgiAliasColumn(seed, count);
  let aliasOffset = lights[2u];
  let base = aliasOffset + column * 4u;
  let q = bitcast<f32>(lights[base]);
  let aliasEntry = lights[base + 1u];
  if (aliasEntry >= count) { return draw; }
  let representedQ = represented_bernoulli_probability_f32(q);
  let selected = select(
    aliasEntry,
    column,
    pcgHashToF32Ddgi(seed ^ 0x85ebca6bu) < representedQ,
  );
  draw.index = selected;
  let representedPmf = bitcast<f32>(lights[aliasOffset + selected * 4u + 2u]);
  draw.pmf = select(
    0.0,
    representedPmf,
    ddgiFiniteF32(representedPmf) && representedPmf > 0.0,
  );
  return draw;
}

fn ddgiEmitterAliasDraw(count: u32, seed: u32) -> DDGIAliasDraw {
  var draw: DDGIAliasDraw;
  draw.index = 0u;
  draw.pmf = 0.0;
  let wordCount = arrayLength(&ddgiEmitterTris);
  // Each represented emitter owns five payload vec4f values and one alias
  // vec4f value. Divide before multiplying so a malformed count cannot wrap.
  if (count == 0u || count > wordCount / 6u) { return draw; }
  let column = ddgiAliasColumn(seed, count);
  let aliasOffset = count * 5u;
  let entry = ddgiEmitterTris[aliasOffset + column];
  let aliasEntry = bitcast<u32>(entry.y);
  if (aliasEntry >= count) { return draw; }
  let representedQ = represented_bernoulli_probability_f32(entry.x);
  let selected = select(
    aliasEntry,
    column,
    pcgHashToF32Ddgi(seed ^ 0x9e3779b9u) < representedQ,
  );
  draw.index = selected;
  let representedPmf = ddgiEmitterTris[aliasOffset + selected].z;
  draw.pmf = select(
    0.0,
    representedPmf,
    ddgiFiniteF32(representedPmf) && representedPmf > 0.0,
  );
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
@group(0) @binding(11) var<storage, read> sceneOpticalTriangleIdentity: array<vec2u>;
@group(0) @binding(12) var<storage, read> sceneOpticalInstanceBoundaryIdBasePlusOne: array<u32>;

// Binding adapters required by the canonical exact optical traversal. DDGI
// keeps the ordinary raw-buffer layout rather than the walkaround scene arena,
// but the traversal algorithm and all its identity/source-feature rules are
// byte-identical with the main renderer.
fn bvhNodeCapacity() -> u32 { return arrayLength(&bvh); }
fn bvhIndexCount() -> u32 { return arrayLength(&bvh_index); }
fn bvhPositionCount() -> u32 { return arrayLength(&bvh_position); }
fn sceneOpticalTriangleIdentityCount() -> u32 {
  return arrayLength(&sceneOpticalTriangleIdentity);
}
fn sceneLoadOpticalTriangleIdentity(index: u32) -> vec2u {
  return sceneOpticalTriangleIdentity[index];
}
fn sceneOpticalInstanceBoundaryIdBaseCount() -> u32 {
  return arrayLength(&sceneOpticalInstanceBoundaryIdBasePlusOne);
}
fn sceneLoadOpticalInstanceBoundaryIdBasePlusOne(index: u32) -> u32 {
  return sceneOpticalInstanceBoundaryIdBasePlusOne[index];
}

${DDGI_OPTICAL_SCENE_TRAVERSAL_WGSL}

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
  let version = ddgiMaterialMetaExactU32(formatHeader.x);
  let materialRecordCount = ddgiMaterialMetaExactU32(formatHeader.y);
  let triangleCount = ddgiMaterialMetaExactU32(formatHeader.z);
  let recordStride = ddgiMaterialMetaExactU32(formatHeader.w);
  if (
    version != 3u ||
    materialRecordCount == 0u ||
    materialRecordCount == 0xffffffffu ||
    triangleCount == 0xffffffffu ||
    triIndex >= triangleCount ||
    recordStride != DDGI_MATERIAL_MAP_META_TEXELS_PER_TRI
  ) {
    return totalTexels;
  }
  let materialBase = ddgiMaterialMetaExactU32(addressHeader.x);
  let triangleMaterialBase = ddgiMaterialMetaExactU32(addressHeader.y);
  let uvAffineBase = ddgiMaterialMetaExactU32(addressHeader.z);
  let activeUvLaneCount = ddgiMaterialMetaExactU32(addressHeader.w);
  let payloadEnd = totalTexels - 4u;
  if (
    materialBase == 0xffffffffu ||
    triangleMaterialBase == 0xffffffffu ||
    uvAffineBase == 0xffffffffu ||
    activeUvLaneCount == 0xffffffffu ||
    materialBase > triangleMaterialBase ||
    triangleMaterialBase > uvAffineBase ||
    triangleMaterialBase > payloadEnd ||
    uvAffineBase > payloadEnd ||
    activeUvLaneCount > 14u
  ) { return totalTexels; }
  let directoryHeader = textureLoad(
    ddgiMaterialMapMeta,
    ddgiMaterialMetaRawCoord(totalTexels - 2u),
    0,
  );
  let atlasAddressBase = ddgiMaterialMetaExactU32(directoryHeader.x);
  if (
    atlasAddressBase == 0xffffffffu ||
    uvAffineBase > atlasAddressBase ||
    atlasAddressBase > payloadEnd
  ) { return totalTexels; }
  let materialRegionTexels = triangleMaterialBase - materialBase;
  if (
    materialRecordCount >
      materialRegionTexels / DDGI_MATERIAL_MAP_META_TEXELS_PER_TRI ||
    materialRecordCount * DDGI_MATERIAL_MAP_META_TEXELS_PER_TRI !=
      materialRegionTexels
  ) { return totalTexels; }
  let triangleTableTexels = (triangleCount + 3u) / 4u;
  if (triangleTableTexels != uvAffineBase - triangleMaterialBase) {
    return totalTexels;
  }
  let uvStride = activeUvLaneCount * 2u;
  let uvRegionTexels = atlasAddressBase - uvAffineBase;
  if (
    (uvStride == 0u && uvRegionTexels != 0u) ||
    (uvStride != 0u &&
      (triangleCount > uvRegionTexels / uvStride ||
        triangleCount * uvStride != uvRegionTexels))
  ) { return totalTexels; }
  if (metaOffset >= 128u && metaOffset < 156u) {
    let laneWord = metaOffset - 128u;
    let lane = laneWord / 2u;
    if (lane >= activeUvLaneCount) { return totalTexels; }
    let availableUvTexels = atlasAddressBase - uvAffineBase;
    if (uvStride == 0u || triIndex > availableUvTexels / uvStride) {
      return totalTexels;
    }
    let triangleUvOffset = triIndex * uvStride;
    if (
      triangleUvOffset > availableUvTexels ||
      laneWord >= availableUvTexels - triangleUvOffset
    ) { return totalTexels; }
    return uvAffineBase + triangleUvOffset + laneWord;
  }
  let idTableOffset = triIndex / 4u;
  if (idTableOffset >= uvAffineBase - triangleMaterialBase) {
    return totalTexels;
  }
  let idTableTexel = triangleMaterialBase + idTableOffset;
  let ids = textureLoad(
    ddgiMaterialMapMeta,
    ddgiMaterialMetaRawCoord(idTableTexel),
    0,
  );
  let materialId = ddgiMaterialMetaExactU32(ids[triIndex & 3u]);
  if (materialId >= materialRecordCount) { return totalTexels; }
  if (
    metaOffset >= DDGI_MATERIAL_MAP_META_TEXELS_PER_TRI ||
    materialId > materialRegionTexels / DDGI_MATERIAL_MAP_META_TEXELS_PER_TRI
  ) { return totalTexels; }
  let materialOffset = materialId * DDGI_MATERIAL_MAP_META_TEXELS_PER_TRI;
  if (
    materialOffset >= materialRegionTexels ||
    metaOffset >= materialRegionTexels - materialOffset
  ) { return totalTexels; }
  return materialBase + materialOffset + metaOffset;
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
const DDGI_MATERIAL_ATLAS_MAX_MIP_LEVELS: u32 = 16u;
const DDGI_MATERIAL_ATLAS_MAX_EXACT_DIMENSION: u32 = 0x01000000u;
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

struct DdgiMaterialAtlasSample {
  value: vec4f,
  encoding: u32,
  valid: u32,
};

fn ddgiMaterialAtlasInvalidSample() -> DdgiMaterialAtlasSample {
  var out: DdgiMaterialAtlasSample;
  out.value = vec4f(0.0);
  out.encoding = 0u;
  out.valid = 0u;
  return out;
}

fn ddgiMaterialAtlasValidSample(value: vec4f, encoding: u32) -> DdgiMaterialAtlasSample {
  var out: DdgiMaterialAtlasSample;
  out.value = value;
  out.encoding = encoding;
  out.valid = 1u;
  return out;
}

fn ddgiMaterialAtlasSampleUsable(sample: DdgiMaterialAtlasSample) -> bool {
  let maxFiniteF32 = bitcast<f32>(0x7f7fffffu);
  return
    sample.valid != 0u &&
    all(sample.value == sample.value) &&
    all(abs(sample.value) <= vec4f(maxFiniteF32));
}

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
  let directoryReserved0 = ddgiMaterialMetaExactU32(directoryHeader.z);
  let directoryReserved1 = ddgiMaterialMetaExactU32(directoryHeader.w);
  let directoryEnd = totalTexels - 4u;
  if (
    addressBase == 0xffffffffu || layerCount == 0xffffffffu ||
    directoryReserved0 != 0u || directoryReserved1 != 0u ||
    addressBase > directoryEnd ||
    layerCount > (directoryEnd - addressBase) /
      DDGI_MATERIAL_ATLAS_ADDRESS_TEXELS_PER_LAYER
  ) { return out; }
  let logicalLayer = u32(layer);
  if (logicalLayer >= layerCount) { return out; }
  let recordTexel = addressBase +
    logicalLayer * DDGI_MATERIAL_ATLAS_ADDRESS_TEXELS_PER_LAYER;
  if (
    recordTexel > directoryEnd ||
    DDGI_MATERIAL_ATLAS_ADDRESS_TEXELS_PER_LAYER > directoryEnd - recordTexel
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
  let infoReserved0 = ddgiMaterialMetaExactU32(info1.z);
  let infoReserved1 = ddgiMaterialMetaExactU32(info1.w);
  let encodingPlaneCompatible =
    ((out.encoding == DDGI_MATERIAL_ATLAS_ENCODING_RGBA8_UNORM ||
      out.encoding == DDGI_MATERIAL_ATLAS_ENCODING_RGBA8_SNORM) &&
      out.planeCount == 1u) ||
    ((out.encoding == DDGI_MATERIAL_ATLAS_ENCODING_RGBA16_FLOAT ||
      out.encoding == DDGI_MATERIAL_ATLAS_ENCODING_RGBA16_UNORM ||
      out.encoding == DDGI_MATERIAL_ATLAS_ENCODING_RGBA16_SNORM) &&
      out.planeCount == 2u) ||
    (out.encoding == DDGI_MATERIAL_ATLAS_ENCODING_RGBA32_FLOAT &&
      out.planeCount == 4u);
  out.recordTexel = recordTexel;
  out.valid = select(
    0u,
    1u,
    out.width > 0u &&
    out.width <= DDGI_MATERIAL_ATLAS_MAX_EXACT_DIMENSION &&
    out.height > 0u &&
    out.height <= DDGI_MATERIAL_ATLAS_MAX_EXACT_DIMENSION &&
    out.mipLevelCount > 0u &&
    out.mipLevelCount <= DDGI_MATERIAL_ATLAS_MAX_MIP_LEVELS &&
    out.decodeSrgb <= 1u &&
    infoReserved0 == 0u && infoReserved1 == 0u &&
    encodingPlaneCompatible,
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
  let logicalLayer = ddgiMaterialMetaExactU32(meta0.x);
  if (logicalLayer > 0x7fffffffu) { return false; }
  return ddgiMaterialAtlasLayerAddress(i32(logicalLayer)).valid != 0u;
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
) -> DdgiMaterialAtlasSample {
  if (address.valid == 0u || level >= address.mipLevelCount) {
    return ddgiMaterialAtlasInvalidSample();
  }
  let mipRecord = textureLoad(
    ddgiMaterialMapMeta,
    ddgiMaterialMetaRawCoord(address.recordTexel + 2u + level),
    0,
  );
  let originX = ddgiMaterialMetaExactU32(mipRecord.x);
  let originY = ddgiMaterialMetaExactU32(mipRecord.y);
  let baseLayer = ddgiMaterialMetaExactU32(mipRecord.z);
  let mipReserved = ddgiMaterialMetaExactU32(mipRecord.w);
  let atlasDims = textureDimensions(ddgiMaterialTextureAtlas);
  let atlasLayers = textureNumLayers(ddgiMaterialTextureAtlas);
  if (
    originX == 0xffffffffu || originY == 0xffffffffu ||
    baseLayer == 0xffffffffu ||
    mipReserved != 0u ||
    any(logicalTexel < vec2i(0)) ||
    originX >= atlasDims.x || originY >= atlasDims.y ||
    baseLayer >= atlasLayers ||
    address.planeCount > atlasLayers - baseLayer
  ) {
    return ddgiMaterialAtlasInvalidSample();
  }
  let logical = vec2u(logicalTexel);
  let logicalDims = ddgiMaterialAtlasLevelDimensions(address, level);
  if (
    logicalDims.x > atlasDims.x - originX ||
    logicalDims.y > atlasDims.y - originY ||
    logical.x >= atlasDims.x - originX ||
    logical.y >= atlasDims.y - originY
  ) { return ddgiMaterialAtlasInvalidSample(); }
  let coord = vec2u(originX, originY) + logical;
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
      return ddgiMaterialAtlasInvalidSample();
    }
  }
  let maxFiniteF32 = bitcast<f32>(0x7f7fffffu);
  if (
    !all(value == value) ||
    any(abs(value) > vec4f(maxFiniteF32))
  ) { return ddgiMaterialAtlasInvalidSample(); }
  if (address.decodeSrgb != 0u) {
    value = vec4f(
      ddgiMaterialAtlasSrgbChannelToLinear(value.r),
      ddgiMaterialAtlasSrgbChannelToLinear(value.g),
      ddgiMaterialAtlasSrgbChannelToLinear(value.b),
      value.a,
    );
  }
  if (
    !all(value == value) ||
    any(abs(value) > vec4f(maxFiniteF32))
  ) { return ddgiMaterialAtlasInvalidSample(); }
  return ddgiMaterialAtlasValidSample(value, address.encoding);
}

fn ddgiSampleMaterialAtlasNearestLevel(
  wrapped: vec2f,
  layer: i32,
  level: u32,
) -> DdgiMaterialAtlasSample {
  if (
    !ddgiFiniteVec2(wrapped) ||
    any(wrapped < vec2f(0.0)) ||
    any(wrapped > vec2f(1.0))
  ) {
    return ddgiMaterialAtlasInvalidSample();
  }
  let address = ddgiMaterialAtlasLayerAddress(layer);
  if (address.valid == 0u) { return ddgiMaterialAtlasInvalidSample(); }
  let dims = ddgiMaterialAtlasLevelDimensions(address, level);
  let texelF = floor(wrapped * vec2f(dims));
  if (!ddgiFiniteVec2(texelF) || any(texelF < vec2f(0.0))) {
    return ddgiMaterialAtlasInvalidSample();
  }
  let texel = vec2i(
    i32(min(u32(texelF.x), dims.x - 1u)),
    i32(min(u32(texelF.y), dims.y - 1u)),
  );
  return ddgiMaterialAtlasDecodeTexel(address, texel, level);
}

fn ddgiSampleMaterialAtlasLinearLevel(
  wrapped: vec2f,
  layer: i32,
  samplerPacked: u32,
  level: u32,
) -> DdgiMaterialAtlasSample {
  if (
    !ddgiFiniteVec2(wrapped) ||
    any(wrapped < vec2f(0.0)) ||
    any(wrapped > vec2f(1.0))
  ) {
    return ddgiMaterialAtlasInvalidSample();
  }
  let address = ddgiMaterialAtlasLayerAddress(layer);
  if (address.valid == 0u) { return ddgiMaterialAtlasInvalidSample(); }
  let dims = ddgiMaterialAtlasLevelDimensions(address, level);
  let size = vec2i(i32(dims.x), i32(dims.y));
  let coord = wrapped * vec2f(f32(dims.x), f32(dims.y)) - vec2f(0.5);
  if (!ddgiFiniteVec2(coord)) { return ddgiMaterialAtlasInvalidSample(); }
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
  if (
    c00.valid == 0u || c10.valid == 0u ||
    c01.valid == 0u || c11.valid == 0u
  ) { return ddgiMaterialAtlasInvalidSample(); }
  return ddgiMaterialAtlasValidSample(mix(
    mix(c00.value, c10.value, fraction.x),
    mix(c01.value, c11.value, fraction.x),
    fraction.y,
  ), address.encoding);
}

fn ddgiSampleMaterialAtlasLevel(
  wrapped: vec2f,
  layer: i32,
  samplerPacked: u32,
  level: u32,
  lod: f32,
) -> DdgiMaterialAtlasSample {
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
) -> DdgiMaterialAtlasSample {
  if (
    !ddgiFiniteVec2(wrapped) ||
    any(wrapped < vec2f(0.0)) ||
    any(wrapped > vec2f(1.0)) ||
    !ddgiFiniteF32(lod)
  ) {
    return ddgiMaterialAtlasInvalidSample();
  }
  let mipFilter = (samplerPacked >> 8u) & 0x3u;
  let address = ddgiMaterialAtlasLayerAddress(layer);
  if (address.valid == 0u) { return ddgiMaterialAtlasInvalidSample(); }
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
  if (c0.valid == 0u || c1.valid == 0u) {
    return ddgiMaterialAtlasInvalidSample();
  }
  return ddgiMaterialAtlasValidSample(mix(
    c0.value,
    c1.value,
    clampedLod - floor(clampedLod),
  ), address.encoding);
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
  let meta0 = ddgiMaterialMetaLoadOrZero(triIndex, mapOffset);
  let packedFlags = ddgiMaterialMetaExactU32(meta0.y);
  let flags = select(0u, packedFlags, packedFlags != 0xffffffffu);
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
) -> DdgiMaterialAtlasSample {
  if (
    !ddgiMaterialMetaAvailable(triIndex, metaOffset) ||
    !ddgiMaterialMetaAvailable(triIndex, metaOffset + 1u)
  ) {
    return ddgiMaterialAtlasInvalidSample();
  }
  let meta0 = textureLoad(ddgiMaterialMapMeta, ddgiMaterialMetaCoord(triIndex, metaOffset), 0);
  let logicalLayer = ddgiMaterialMetaExactU32(meta0.x);
  let wrapPacked = ddgiMaterialMetaExactU32(meta0.y);
  if (logicalLayer > 0x7fffffffu || wrapPacked == 0xffffffffu) {
    return ddgiMaterialAtlasInvalidSample();
  }
  let layer = i32(logicalLayer);
  let address = ddgiMaterialAtlasLayerAddress(layer);
  if (address.valid == 0u) { return ddgiMaterialAtlasInvalidSample(); }
  let texCoord = (wrapPacked >> 4u) & 0xFu;
  let uv = materialResolveUv(triIndex, texCoord, uv0, uv1);
  if (!ddgiFiniteVec2(uv)) { return ddgiMaterialAtlasInvalidSample(); }
  let meta1 = textureLoad(ddgiMaterialMapMeta, ddgiMaterialMetaCoord(triIndex, metaOffset + 1u), 0);
  let fallbackCandidate = uv + transformedDelta;
  var transformed = select(
    uv,
    fallbackCandidate,
    ddgiFiniteVec2(fallbackCandidate),
  );
  var transformScale = vec2f(1.0);
  if (
    ddgiFiniteVec2(meta0.zw) &&
    ddgiFiniteVec2(meta1.xy) &&
    ddgiFiniteVec2(meta1.zw) &&
    ddgiFiniteVec2(transformedDelta)
  ) {
    let scaled = uv * meta1.xy;
    let candidate = vec2f(
      scaled.x * meta1.z - scaled.y * meta1.w,
      scaled.x * meta1.w + scaled.y * meta1.z,
    ) + meta0.zw + transformedDelta;
    // A malformed affine transform must not feed NaN/Inf into fract(),
    // floor(), or an integer conversion. Match the other walkaround atlas
    // consumers: retain a finite derivative delta, then fall back to raw UV.
    if (ddgiFiniteVec2(candidate)) {
      transformed = candidate;
      transformScale = abs(meta1.xy);
    }
  }
  let wrapped = ddgiWrapMaterialUv(transformed, wrapPacked);
  if (
    !ddgiFiniteVec2(wrapped) ||
    any(wrapped < vec2f(0.0)) ||
    any(wrapped > vec2f(1.0))
  ) {
    return ddgiMaterialAtlasInvalidSample();
  }
  // Probe rays have no screen-space derivatives. Use the logical source
  // footprint per angular probe-ray sample as the bounded minification model;
  // authored mip/nearest/linear policy still controls the actual lookup.
  let logicalSize = vec2f(f32(address.width), f32(address.height));
  let angularSamples = sqrt(f32(max(RAYS_PER_PROBE, 1u)));
  let footprint = transformScale * logicalSize / angularSamples;
  let footprintMax = max(footprint.x, footprint.y);
  let lodCandidate = log2(max(footprintMax, 1e-8));
  let lod = select(0.0, lodCandidate, ddgiFiniteF32(lodCandidate));
  return ddgiSampleMaterialAtlasAtLod(wrapped, layer, wrapPacked, lod);
}

fn ddgiSampleMaterialAtlasRawAtOffset(triIndex: u32, metaOffset: u32, uv0: vec2f, uv1: vec2f) -> DdgiMaterialAtlasSample {
  return ddgiSampleMaterialAtlasRawAtOffsetDelta(triIndex, metaOffset, uv0, uv1, vec2f(0.0));
}

fn ddgiSampleMaterialAtlasRaw(triIndex: u32, slot: u32, uv0: vec2f, uv1: vec2f) -> DdgiMaterialAtlasSample {
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
  if (
    !ddgiMaterialAtlasMapAvailableAtOffset(triIndex, slot * 2u) ||
    !ddgiMaterialAtlasSampleUsable(texel)
  ) {
    return fallback;
  }
  return clamp(fallback * ddgiMaterialMapChannel(texel.value, channel), 0.0, 1.0);
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
  ) || !ddgiMaterialAtlasSampleUsable(texel)) {
    return scalarTransmission;
  }
  return clamp(scalarTransmission * texel.value.r, 0.0, 1.0);
}

fn ddgiSampleThicknessMapFactorForHit(hit: IntersectionResult) -> vec2f {
  if (!materialOpticalHasAuthoredThickness(hit.indices.w)) {
    return vec2f(1.0, 0.0);
  }
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
  ) || !ddgiMaterialAtlasSampleUsable(texel)) {
    return vec2f(1.0, 0.0);
  }
  return vec2f(clamp(texel.value.g, 0.0, 1.0), 1.0);
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
  ) && ddgiMaterialAtlasSampleUsable(colorMap)) {
    color = max(color * colorMap.value.rgb, vec3f(0.0));
  }
  let intensityMap = ddgiSampleMaterialAtlasRawAtOffset(triIndex, DDGI_MATERIAL_MAP_SPECULAR_INTENSITY_TEXEL_OFFSET, uv0, uv1);
  if (ddgiMaterialAtlasMapAvailableAtOffset(
    triIndex,
    DDGI_MATERIAL_MAP_SPECULAR_INTENSITY_TEXEL_OFFSET,
  ) && ddgiMaterialAtlasSampleUsable(intensityMap)) {
    intensity = clamp(intensity * intensityMap.value.a, 0.0, 1.0);
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
  ) && ddgiMaterialAtlasSampleUsable(clearcoatMap)) {
    factor = clamp(factor * clearcoatMap.value.r, 0.0, 1.0);
  }
  let roughnessMap = ddgiSampleMaterialAtlasRawAtOffset(triIndex, DDGI_MATERIAL_MAP_CLEARCOAT_ROUGHNESS_TEXEL_OFFSET, uv0, uv1);
  if (ddgiMaterialAtlasMapAvailableAtOffset(
    triIndex,
    DDGI_MATERIAL_MAP_CLEARCOAT_ROUGHNESS_TEXEL_OFFSET,
  ) && ddgiMaterialAtlasSampleUsable(roughnessMap)) {
    roughness = clamp(roughness * roughnessMap.value.g, 0.0, 1.0);
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
  ) && ddgiMaterialAtlasSampleUsable(colorMap)) {
    sheenColor = clamp(sheenColor * colorMap.value.rgb, vec3f(0.0), vec3f(1.0));
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
  ) && ddgiMaterialAtlasSampleUsable(roughnessMap)) {
    roughness = clamp(roughness * roughnessMap.value.a, 0.0, 1.0);
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
  ) && ddgiMaterialAtlasSampleUsable(anisoMap)) {
    strength = clamp(strength * anisoMap.value.b, 0.0, 1.0);
    let isSnorm =
      anisoMap.encoding == DDGI_MATERIAL_ATLAS_ENCODING_RGBA8_SNORM ||
      anisoMap.encoding == DDGI_MATERIAL_ATLAS_ENCODING_RGBA16_SNORM;
    let direction = select(
      clamp(anisoMap.value.rg, vec2f(0.0), vec2f(1.0)) * 2.0 - vec2f(1.0),
      clamp(anisoMap.value.rg, vec2f(-1.0), vec2f(1.0)),
      isSnorm,
    );
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
  ) && ddgiMaterialAtlasSampleUsable(iridescenceMap)) {
    factor = clamp(factor * iridescenceMap.value.r, 0.0, 1.0);
  }
  let thicknessMap = ddgiSampleMaterialAtlasRawAtOffset(triIndex, DDGI_MATERIAL_MAP_IRIDESCENCE_THICKNESS_TEXEL_OFFSET, uv0, uv1);
  if (ddgiMaterialAtlasMapAvailableAtOffset(
    triIndex,
    DDGI_MATERIAL_MAP_IRIDESCENCE_THICKNESS_TEXEL_OFFSET,
  ) && ddgiMaterialAtlasSampleUsable(thicknessMap)) {
    let thickness = mix(thicknessMin, thicknessMax, clamp(thicknessMap.value.g, 0.0, 1.0));
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
  let tangent = ddgiNormalizeOr(cross(up, n), vec3f(1.0, 0.0, 0.0));
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

struct DdgiProbeHitMaterial {
  albedo: vec3f,
  baseRoughness: f32,
  roughness: f32,
  metalness: f32,
  specular: vec4f,
  clearcoat: vec2f,
  clearcoatNormal: vec3f,
  sheen: vec4f,
  sheenRoughness: f32,
  anisotropy: vec2f,
  iridescence: vec4f,
  // Face-layer transmission applies to every closure. Thin-film
  // transmittance applies only to the base/source side: the reflected lobe
  // already carries the film's absolute reflectance and must not pay T again.
  reflectionLayerTransmission: vec3f,
  layerTransmission: vec3f,
  volumeScattering: vec4f,
  transmission: f32,
  attenuationColor: vec3f,
  attenuationDistance: f32,
  opticalIor: vec3f,
  authoredThickness: f32,
  thicknessMapScale: f32,
}

fn ddgiSampleProbeHitMaterial(
  hit: IntersectionResult,
  scalarBaseColor: vec3f,
  scalarRoughness: f32,
  scalarMetalness: f32,
  scalarTransmission: f32,
  scalarIor: f32,
  scalarBeerTint: vec3f,
  scalarAttenuationDistance: f32,
  frameNormal: vec3f,
  shadingNormal: vec3f,
  viewDirection: vec3f,
) -> DdgiProbeHitMaterial {
  var out: DdgiProbeHitMaterial;
  out.albedo = scalarBaseColor;
  out.baseRoughness = scalarRoughness;
  out.roughness = scalarRoughness;
  out.metalness = scalarMetalness;
  out.specular = ddgiSampleSpecularMeta(hit.indices.w);
  out.clearcoat = vec2f(0.0);
  out.clearcoatNormal = shadingNormal;
  out.sheen = vec4f(0.0);
  out.sheenRoughness = 0.0;
  out.anisotropy = vec2f(0.0);
  out.iridescence = vec4f(0.0, 1.0, 0.0, 0.0);
  out.reflectionLayerTransmission = vec3f(1.0);
  out.layerTransmission = vec3f(1.0);
  out.volumeScattering = vec4f(0.0);
  out.transmission = scalarTransmission;
  out.attenuationColor = clamp(
    scalarBeerTint, vec3f(0.0), vec3f(1.0),
  );
  out.attenuationDistance = scalarAttenuationDistance;
  let transportIor = select(max(scalarIor, 1.0), 1e6, scalarIor == 0.0);
  out.opticalIor = vec3f(transportIor);
  // Topology comes only from the separately packed analyzed component lane.
  // Positive authored thickness is an optional path-length cap; -1 denotes a
  // synthetic zero-thickness bulk whose full closed-boundary segment is used.
  let headerThickness = max(materialOpticalThickness(hit.indices.w), 0.0);
  out.authoredThickness = select(
    -1.0,
    headerThickness,
    materialOpticalHasAuthoredThickness(hit.indices.w) &&
      headerThickness > 0.0,
  );
  out.thicknessMapScale = 1.0;
  out.clearcoatNormal = ddgiApplyClearcoatNormalMapForHit(hit, frameNormal, shadingNormal);

  let uvs = ddgiHitMaterialUvs(hit);
  if (uvs.valid == 0u) {
    let layerControls = ddgiSampleFaceLayerControls(hit.indices.w, hit.side >= 0.0);
    out.roughness = ddgiFaceLayerRoughness(out.roughness, layerControls);
    out.reflectionLayerTransmission = ddgiFaceLayerTransmission(layerControls);
    out.layerTransmission = out.reflectionLayerTransmission;
    out.volumeScattering = ddgiSampleVolumeScatteringControls(hit.indices.w);
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

  let baseColorTexel = ddgiSampleMaterialAtlasRaw(
    hit.indices.w,
    DDGI_MATERIAL_MAP_SLOT_BASE_COLOR,
    uvs.uv0,
    uvs.uv1,
  );
  if (ddgiMaterialAtlasMapAvailableAtOffset(
    hit.indices.w,
    DDGI_MATERIAL_MAP_SLOT_BASE_COLOR * 2u,
  ) && ddgiMaterialAtlasSampleUsable(baseColorTexel)) {
    out.albedo = scalarBaseColor * baseColorTexel.value.rgb;
  }
  out.roughness = ddgiSampleMaterialScalarMap(
    hit.indices.w,
    DDGI_MATERIAL_MAP_SLOT_ROUGHNESS,
    1u,
    uvs.uv0,
    uvs.uv1,
    scalarRoughness,
  );
  out.baseRoughness = out.roughness;
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
  out.reflectionLayerTransmission = ddgiFaceLayerTransmission(layerControls);
  out.layerTransmission = out.reflectionLayerTransmission;
  out.volumeScattering = ddgiSampleVolumeScatteringControls(hit.indices.w);
  out.transmission = ddgiSampleTransmissionMapForHit(hit, scalarTransmission);
  let thicknessMap = ddgiSampleThicknessMapFactorForHit(hit);
  out.thicknessMapScale = select(
    1.0,
    thicknessMap.x,
    thicknessMap.y >= 0.5 && out.authoredThickness > 0.0,
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
    !ddgiMaterialMetaAvailable(triIndex, normalMapOffset + 1u) ||
    !ddgiMaterialMetaAvailable(triIndex, normalScaleOffset)
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
  if (!ddgiMaterialAtlasSampleUsable(texelColor)) {
    return fallbackNormal;
  }
  let scaleMeta = textureLoad(
    ddgiMaterialMapMeta,
      ddgiMaterialMetaCoord(triIndex, normalScaleOffset),
    0,
  );
  if (!ddgiFiniteF32(scaleMeta.x)) { return fallbackNormal; }
  let normalScale = max(scaleMeta.x, 0.0);
  let isSnorm =
    texelColor.encoding == DDGI_MATERIAL_ATLAS_ENCODING_RGBA8_SNORM ||
    texelColor.encoding == DDGI_MATERIAL_ATLAS_ENCODING_RGBA16_SNORM;
  let decodedNormal = select(
    clamp(texelColor.value.rgb, vec3f(0.0), vec3f(1.0)) * 2.0 - vec3f(1.0),
    clamp(texelColor.value.rgb, vec3f(-1.0), vec3f(1.0)),
    isSnorm,
  );
  let tangentSample = ddgiNormalizeOr(vec3f(
    decodedNormal.x * normalScale,
    decodedNormal.y * normalScale,
    decodedNormal.z,
  ), vec3f(0.0, 0.0, 1.0));

  let frame = ddgiMaterialTangentFrameForHit(hit, frameNormal, normalMapOffset);
  let perturbed = ddgiNormalizeOr(
    frame.tangent * tangentSample.x +
      frame.bitangent * tangentSample.y +
      frameNormal * tangentSample.z,
    fallbackNormal,
  );
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
  if (
    !ddgiFiniteF32(scaleMeta.x) ||
    !ddgiFiniteF32(scaleMeta.y) ||
    !ddgiFiniteF32(scaleMeta.z)
  ) { return shadingNormal; }
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
  if (!ddgiMaterialAtlasSampleUsable(hC)) { return shadingNormal; }
  let bumpMeta = textureLoad(
    ddgiMaterialMapMeta,
    ddgiMaterialMetaCoord(triIndex, DDGI_MATERIAL_MAP_BUMP_TEXEL_OFFSET),
    0,
  );
  let bumpLayer = ddgiMaterialMetaExactU32(bumpMeta.x);
  if (bumpLayer > 0x7fffffffu) { return shadingNormal; }
  let bumpAddress = ddgiMaterialAtlasLayerAddress(i32(bumpLayer));
  if (bumpAddress.valid == 0u) { return shadingNormal; }
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
  let hUSample = ddgiSampleMaterialAtlasRawAtOffsetDelta(
    triIndex,
    DDGI_MATERIAL_MAP_BUMP_TEXEL_OFFSET,
    uvs.uv0,
    uvs.uv1,
    vec2f(texelStep.x, 0.0),
  );
  let hVSample = ddgiSampleMaterialAtlasRawAtOffsetDelta(
    triIndex,
    DDGI_MATERIAL_MAP_BUMP_TEXEL_OFFSET,
    uvs.uv0,
    uvs.uv1,
    vec2f(0.0, texelStep.y),
  );
  if (
    !ddgiMaterialAtlasSampleUsable(hUSample) ||
    !ddgiMaterialAtlasSampleUsable(hVSample) ||
    !ddgiFiniteVec2(texelStep) ||
    any(texelStep <= vec2f(0.0))
  ) { return shadingNormal; }
  let hU = hUSample.value.r;
  let hV = hVSample.value.r;
  let dhdu = (hU - hC.value.r) / texelStep.x;
  let dhdv = (hV - hC.value.r) / texelStep.y;
  let frame = ddgiMaterialTangentFrameForHit(hit, shadingNormal, DDGI_MATERIAL_MAP_BUMP_TEXEL_OFFSET);
  let perturbed = shadingNormal - bumpScale * (dhdu * frame.tangent + dhdv * frame.bitangent);
  let n = ddgiNormalizeOr(perturbed, shadingNormal);
  return select(-n, n, dot(n, shadingNormal) >= 0.0);
}

fn ddgiMaterialAtlasFiniteNonNegativeRadianceOrBlack(value: vec3f) -> vec3f {
  let maxFiniteF32 = bitcast<f32>(0x7f7fffffu);
  let valid =
    all(value == value) &&
    all(abs(value) <= vec3f(maxFiniteF32)) &&
    all(value >= vec3f(0.0));
  return select(vec3f(0.0), value, valid);
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
  ) || !ddgiMaterialAtlasSampleUsable(texel)) {
    return scalarEmission;
  }
  return ddgiMaterialAtlasFiniteNonNegativeRadianceOrBlack(
    scalarEmission * texel.value.rgb,
  );
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
  ) || !ddgiMaterialAtlasSampleUsable(texel)) {
    return vec3f(0.0);
  }
  let intensity = ddgiMaterialMetaLoadOrZero(
    triIndex,
    DDGI_MATERIAL_MAP_LIGHT_INTENSITY_TEXEL_OFFSET,
  ).x;
  return ddgiMaterialAtlasFiniteNonNegativeRadianceOrBlack(
    max(texel.value.rgb, vec3f(0.0)) * max(intensity, 0.0),
  );
}

fn ddgiEmitterSubdivWeightAt(i: u32, j: u32, level: u32) -> vec3f {
  let invLevel = 1.0 / f32(max(level, 1u));
  let u = f32(i) * invLevel;
  let v = f32(j) * invLevel;
  return vec3f(1.0 - u - v, u, v);
}

fn ddgiEmitterParentBarycentricFromLocal(localBary: vec3f, levelF: f32, ordinalF: f32) -> vec3f {
  if (
    !ddgiFiniteF32(levelF) || levelF != round(levelF) ||
    levelF < 0.0 || levelF > 16.0 ||
    !ddgiFiniteF32(ordinalF) || ordinalF != round(ordinalF) ||
    ordinalF < 0.0 || ordinalF > 255.0
  ) { return localBary; }
  let level = max(1u, u32(levelF));
  if (level <= 1u) {
    return localBary;
  }

  let ordinal = u32(ordinalF);
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
  let encodedSourceTriF = ddgiEmitterTris[base + 0u].w;
  if (
    !ddgiFiniteF32(encodedSourceTriF) ||
    encodedSourceTriF != round(encodedSourceTriF) ||
    abs(encodedSourceTriF) > 16777216.0
  ) { return scalarEmission; }
  let encodedSourceTri = i32(encodedSourceTriF);
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
  ) || !ddgiMaterialAtlasSampleUsable(texel)) {
    return scalarEmission;
  }
  return ddgiMaterialAtlasFiniteNonNegativeRadianceOrBlack(
    scalarEmission * texel.value.rgb,
  );
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
  let coverageMode = ddgiMaterialMetaExactU32(coverageMeta.x);
  if (coverageMode > 2u) { return out; }
  out.mode = coverageMode;
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
    clamp(baseColorTexel.value.a, 0.0, 1.0),
    ddgiMaterialAtlasMapAvailableAtOffset(
      hit.indices.w,
      DDGI_MATERIAL_MAP_SLOT_BASE_COLOR * 2u,
    ) && ddgiMaterialAtlasSampleUsable(baseColorTexel),
  );
  let alphaTexel = ddgiSampleMaterialAtlasRaw(hit.indices.w, DDGI_MATERIAL_MAP_SLOT_ALPHA, uvs.uv0, uvs.uv1);
  let alphaMapCoverage = select(
    1.0,
    clamp(alphaTexel.value.r, 0.0, 1.0),
    ddgiMaterialAtlasMapAvailableAtOffset(
      hit.indices.w,
      DDGI_MATERIAL_MAP_SLOT_ALPHA * 2u,
    ) && ddgiMaterialAtlasSampleUsable(alphaTexel),
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
    let representedCoverage = represented_bernoulli_probability_f32(
      alpha.coverage,
    );
    return 1.0 - representedCoverage;
  }
  return select(0.0, 1.0, alpha.coverage <= 0.0);
}

// -----------------------------------------------------------------
// BVH traversal — merged world BLAS or TLAS+local BLAS (PR-5.2).
// -----------------------------------------------------------------
fn ddgiFiniteF32(v: f32) -> bool {
  return v == v && abs(v) <= 3.402823e+38;
}

fn ddgiFiniteVec2(v: vec2f) -> bool {
  return ddgiFiniteF32(v.x) && ddgiFiniteF32(v.y);
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

fn traceSceneFirstHitDdgiAt(
  ray: Ray,
  exclusiveMinT: f32,
) -> IntersectionResult {
  if (ddgiTrace.bvhMode == 1u && ddgiTrace.tlasNodeCount > 0u) {
    return traceTlasFirstHit(
      ddgiTrace.tlasNodeCount,
      ray,
      exclusiveMinT,
    );
  }
  return bvhIntersectFirstHit(ray, exclusiveMinT);
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
  let hash = (word >> 22u) ^ word;
  return f32(hash >> 8u) / 16777216.0;
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
    let representedCoverage = represented_bernoulli_probability_f32(
      alpha.coverage,
    );
    return representedCoverage < 1.0 &&
      ddgiAlphaBlendCoverageHash(hit, ray, layer) >= representedCoverage;
  }
  return alpha.coverage <= 0.0;
}

fn ddgiTraceFirstHitAlphaMaskTextured(ray: Ray) -> IntersectionResult {
  var exclusiveMinT = ddgiProbeDistance(DDGI_TRACE_T_MIN_NORMALIZED);
  let surfaceBudget = ddgiWorldSurfaceBudget();

  for (var layer = 0u; layer < surfaceBudget; layer = layer + 1u) {
    let hit = traceSceneFirstHitDdgiAt(ray, exclusiveMinT);
    if (!hit.didHit) {
      return hit;
    }
    if (!ddgiMaterialAlphaDiscardedForProbeHit(hit, ray, layer)) {
      return hit;
    }
    if (!(hit.dist > exclusiveMinT)) { return hit; }
    exclusiveMinT = hit.dist;
  }

  let exhausted = traceSceneFirstHitDdgiAt(ray, exclusiveMinT);
  // Conservative world-surface-budget overflow blocks instead of leaking.
  return exhausted;
}

struct DdgiOpticalSourceAwareHit {
  valid: u32,
  hit: IntersectionResult,
};

// Alpha-aware continuation from an accepted optical feature. The represented
// origin stays fixed and the open interval begins at t=0; only the exact source
// face/edge/vertex fan is excluded by the canonical optical traversal.
fn ddgiTraceFirstHitAlphaMaskTexturedWithOpticalSource(
  ray: Ray,
  sourceFeature: OpticalSourceFeature,
) -> DdgiOpticalSourceAwareHit {
  var out: DdgiOpticalSourceAwareHit;
  out.valid = 0u;
  out.hit = tlasEmptyFirstHit();
  var exclusiveMinT = 0.0;
  let surfaceBudget = ddgiWorldSurfaceBudget();
  for (var layer = 0u; layer < surfaceBudget; layer = layer + 1u) {
    let candidate = traceSceneFirstHitWithOpticalSourceExclusion(
      ddgiTrace.bvhMode,
      ddgiTrace.tlasNodeCount,
      ray,
      exclusiveMinT,
      sourceFeature,
    );
    if (candidate.valid == 0u) { return out; }
    if (!candidate.hit.didHit) {
      out.valid = 1u;
      out.hit = candidate.hit;
      return out;
    }
    if (!ddgiMaterialAlphaDiscardedForProbeHit(
      candidate.hit, ray, layer,
    )) {
      out.valid = 1u;
      out.hit = candidate.hit;
      return out;
    }
    if (!(candidate.hit.dist > exclusiveMinT)) { return out; }
    exclusiveMinT = candidate.hit.dist;
  }
  return out;
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

const DDGI_OPTICAL_MEDIUM_CAPACITY: u32 = 8u;
const DDGI_MAX_FINITE_F32: f32 = 3.402823466e38;

struct DdgiOpticalMediumState {
  depth: u32,
  boundaryId: array<u32, 8>,
  representedId: array<u32, 8>,
  tri: array<u32, 8>,
  ior: array<vec3f, 8>,
  attenuationColor: array<vec3f, 8>,
  attenuationDistance: array<f32, 8>,
  authoredThickness: array<f32, 8>,
  thicknessMapScale: array<f32, 8>,
  scatter: array<vec4f, 8>,
  albedo: array<vec3f, 8>,
  distance: array<f32, 8>,
  transmissionPaid: array<u32, 8>,
};

struct DdgiContainingMedia {
  valid: u32,
  state: DdgiOpticalMediumState,
};

fn ddgiEmptyOpticalMediumState() -> DdgiOpticalMediumState {
  var state: DdgiOpticalMediumState;
  state.depth = 0u;
  return state;
}

fn ddgiOpticalCoverageForHit(hit: IntersectionResult) -> f32 {
  let alpha = ddgiMaterialAlphaCoverageForHit(hit);
  if (alpha.mode == 0u) { return 1.0; }
  if (alpha.mode == 1u) {
    return select(0.0, 1.0, alpha.coverage >= alpha.cutoff);
  }
  if (alpha.mode == 2u) {
    return represented_bernoulli_probability_f32(alpha.coverage);
  }
  return select(0.0, 1.0, alpha.coverage > 0.0);
}

fn ddgiOpticalStateIdentityEqual(
  a: DdgiOpticalMediumState,
  b: DdgiOpticalMediumState,
) -> bool {
  if (a.depth != b.depth) { return false; }
  for (var i = 0u; i < a.depth; i = i + 1u) {
    if (
      a.boundaryId[i] != b.boundaryId[i] ||
      a.representedId[i] != b.representedId[i]
    ) { return false; }
  }
  return true;
}

// Fixed-origin, actual-direction containment. Front crossings push a temporary
// LIFO and matching backs pop it. Unmatched backs are the real launch-inside
// exits (inner-to-outer); their exact exit-face optical payload is reversed into
// the live outer-to-inner state. Tangencies do not change containment.
fn ddgiClassifyContainingMedia(
  origin: vec3f,
  direction: vec3f,
  sourceFeature: OpticalSourceFeature,
  respectCastShadow: bool,
) -> DdgiContainingMedia {
  var out: DdgiContainingMedia;
  out.valid = 0u;
  out.state = ddgiEmptyOpticalMediumState();
  if (
    !all(origin == origin) || !all(direction == direction) ||
    any(abs(origin) > vec3f(DDGI_MAX_FINITE_F32)) ||
    any(abs(direction) > vec3f(DDGI_MAX_FINITE_F32)) ||
    !ddgiCanNormalize(direction)
  ) { return out; }

  let ray = Ray(origin, direction);
  var temporaryDepth = 0u;
  var temporaryBoundary: array<u32, 8>;
  var temporaryRepresented: array<u32, 8>;
  var innerToOuter = ddgiEmptyOpticalMediumState();
  var exclusiveMinT = 0.0;
  var complete = false;
  let surfaceBudget = ddgiWorldSurfaceBudget();
  for (var surface = 0u; surface < surfaceBudget; surface = surface + 1u) {
    let event = traceSceneOpticalBoundaryEvent(
      ddgiTrace.bvhMode,
      ddgiTrace.tlasNodeCount,
      ray,
      exclusiveMinT,
      sourceFeature,
    );
    if (event.status == OPTICAL_BOUNDARY_EVENT_INVALID) { return out; }
    if (event.status == OPTICAL_BOUNDARY_EVENT_NONE) {
      complete = true;
      break;
    }
    if (!(event.t > exclusiveMinT)) { return out; }
    exclusiveMinT = event.t;
    if (event.status == OPTICAL_BOUNDARY_EVENT_TANGENT) { continue; }
    if (
      event.status != OPTICAL_BOUNDARY_EVENT_CROSSING ||
      event.encodedBoundaryId == 0u ||
      event.representedPrimitiveInstanceId == 0u ||
      !event.hit.didHit ||
      event.hit.indices.w >= arrayLength(&bvh_materialId)
    ) { return out; }

    let materialId = bvh_materialId[event.hit.indices.w];
    if (materialId >= DDGI_MAX_MATERIALS) { return out; }
    let entry = materials[materialId];
    // A primitive which does not cast shadows is absent from the shadow ray's
    // topology as well as its opacity. Ignore both faces symmetrically so a
    // disabled outer shell cannot corrupt enabled nested-medium ownership.
    if (
      respectCastShadow &&
      (entry.flags & MATERIAL_FLAG_CAST_SHADOW_DISABLED) != 0u
    ) { continue; }
    if (
      !(entry.transmission > 0.0) ||
      ddgiOpticalCoverageForHit(event.hit) != 1.0
    ) { return out; }

    if (event.side > 0.0) {
      if (temporaryDepth >= DDGI_OPTICAL_MEDIUM_CAPACITY) { return out; }
      temporaryBoundary[temporaryDepth] = event.encodedBoundaryId;
      temporaryRepresented[temporaryDepth] =
        event.representedPrimitiveInstanceId;
      temporaryDepth = temporaryDepth + 1u;
      continue;
    }
    if (event.side >= 0.0) { return out; }
    if (temporaryDepth > 0u) {
      let top = temporaryDepth - 1u;
      if (
        temporaryBoundary[top] != event.encodedBoundaryId ||
        temporaryRepresented[top] !=
          event.representedPrimitiveInstanceId
      ) { return out; }
      temporaryDepth = top;
      continue;
    }
    if (innerToOuter.depth >= DDGI_OPTICAL_MEDIUM_CAPACITY) { return out; }
    let smoothNormal = ddgiSmoothShadingNormalForHit(
      event.hit, event.hit.normal,
    );
    let shadingNormal = ddgiApplyBumpMapForHit(
      event.hit,
      ddgiApplyNormalMapForHit(event.hit, smoothNormal),
    );
    let sampled = ddgiSampleProbeHitMaterial(
      event.hit,
      entry.baseColor,
      entry.roughness,
      entry.metalness,
      entry.transmission,
      entry.ior,
      entry.attenuationColor,
      entry.attenuationDistance,
      smoothNormal,
      shadingNormal,
      -direction,
    );
    let depth = innerToOuter.depth;
    innerToOuter.boundaryId[depth] = event.encodedBoundaryId;
    innerToOuter.representedId[depth] =
      event.representedPrimitiveInstanceId;
    innerToOuter.tri[depth] = event.hit.indices.w;
    innerToOuter.ior[depth] = sampled.opticalIor;
    innerToOuter.attenuationColor[depth] = sampled.attenuationColor;
    innerToOuter.attenuationDistance[depth] = sampled.attenuationDistance;
    innerToOuter.authoredThickness[depth] = sampled.authoredThickness;
    innerToOuter.thicknessMapScale[depth] = sampled.thicknessMapScale;
    innerToOuter.scatter[depth] = sampled.volumeScattering;
    innerToOuter.albedo[depth] = sampled.albedo;
    innerToOuter.distance[depth] = 0.0;
    innerToOuter.transmissionPaid[depth] = 0u;
    innerToOuter.depth = depth + 1u;
  }
  if (!complete || temporaryDepth != 0u) { return out; }

  out.state.depth = innerToOuter.depth;
  for (var destination = 0u; destination < innerToOuter.depth; destination += 1u) {
    let source = innerToOuter.depth - 1u - destination;
    out.state.boundaryId[destination] = innerToOuter.boundaryId[source];
    out.state.representedId[destination] = innerToOuter.representedId[source];
    out.state.tri[destination] = innerToOuter.tri[source];
    out.state.ior[destination] = innerToOuter.ior[source];
    out.state.attenuationColor[destination] =
      innerToOuter.attenuationColor[source];
    out.state.attenuationDistance[destination] =
      innerToOuter.attenuationDistance[source];
    out.state.authoredThickness[destination] =
      innerToOuter.authoredThickness[source];
    out.state.thicknessMapScale[destination] =
      innerToOuter.thicknessMapScale[source];
    out.state.scatter[destination] = innerToOuter.scatter[source];
    out.state.albedo[destination] = innerToOuter.albedo[source];
    out.state.distance[destination] = 0.0;
    out.state.transmissionPaid[destination] = 0u;
  }
  out.valid = 1u;
  return out;
}

fn ddgiMediumExtinctionForSegment(
  triIndex: u32,
  attenuationColor: vec3f,
  attenuationDistance: f32,
  authoredThickness: f32,
  thicknessMapScale: f32,
  scattering: vec3f,
  segmentLength: f32,
) -> vec3f {
  if (segmentLength <= 0.0) { return vec3f(1.0); }
  if (
    segmentLength != segmentLength ||
    authoredThickness != authoredThickness ||
    thicknessMapScale != thicknessMapScale ||
    any(attenuationColor != attenuationColor) ||
    any(scattering != scattering) ||
    segmentLength > DDGI_MAX_FINITE_F32
  ) { return vec3f(0.0); }
  let hasAuthoredThickness = authoredThickness > 0.0 &&
    materialOpticalHasAuthoredThickness(triIndex);
  let mappedCap = authoredThickness * clamp(thicknessMapScale, 0.0, 1.0);
  let transportDistance = select(
    segmentLength,
    min(segmentLength, mappedCap),
    hasAuthoredThickness,
  );
  if (transportDistance <= 0.0) { return vec3f(1.0); }
  let rgbBeer = beerLambertTransmittanceRgb(
    clamp(attenuationColor, vec3f(0.0), vec3f(1.0)),
    attenuationDistance,
    transportDistance,
  );
  let absorption = materialSpectralAttenuation(
    triIndex,
    transportDistance,
    rgbBeer,
  );
  return clamp(absorption, vec3f(0.0), vec3f(1.0)) *
    ddgiHomogeneousBeerTransmittanceRgb(
      max(scattering, vec3f(0.0)), transportDistance,
    );
}

fn ddgiAccumulateMediumDistance(
  state: ptr<function, DdgiOpticalMediumState>,
  segmentDistance: f32,
) -> bool {
  if (
    segmentDistance != segmentDistance || segmentDistance < 0.0 ||
    segmentDistance > DDGI_MAX_FINITE_F32
  ) { return false; }
  if ((*state).depth == 0u) { return true; }
  let top = (*state).depth - 1u;
  let distance = (*state).distance[top] + segmentDistance;
  if (
    distance != distance || distance < 0.0 ||
    distance > DDGI_MAX_FINITE_F32
  ) { return false; }
  (*state).distance[top] = distance;
  return true;
}

fn ddgiShadowFaceTransmission(
  hit: IntersectionResult,
  direction: vec3f,
  frontFacing: bool,
) -> vec3f {
  let layer = ddgiFaceLayerTransmission(
    ddgiSampleFaceLayerControls(hit.indices.w, frontFacing),
  );
  let film = materialThinFilmResponse(
    hit.indices.w,
    frontFacing,
    abs(dot(hit.normal, direction)),
  );
  return layer * select(
    vec3f(1.0), film.transmittance, film.present != 0u,
  );
}

fn ddgiShadowThinSheetTransmission(
  hit: IntersectionResult,
  direction: vec3f,
) -> vec3f {
  let frontFacing = hit.side >= 0.0;
  return ddgiShadowFaceTransmission(hit, direction, frontFacing) *
    ddgiFaceLayerTransmission(
      ddgiSampleFaceLayerControls(hit.indices.w, !frontFacing),
    );
}

fn ddgiShadowEndpointTransmission(
  origin: vec3f,
  direction: vec3f,
  tMax: f32,
  state: DdgiOpticalMediumState,
) -> vec3f {
  if (tMax != tMax || tMax < 0.0) { return vec3f(0.0); }
  if (tMax > DDGI_MAX_FINITE_F32) {
    return select(vec3f(0.0), vec3f(1.0), state.depth == 0u);
  }
  let endpoint = origin + direction * tMax;
  let classified = ddgiClassifyContainingMedia(
    endpoint,
    direction,
    opticalSourceFeatureInvalid(),
    true,
  );
  if (
    classified.valid == 0u ||
    !ddgiOpticalStateIdentityEqual(state, classified.state)
  ) { return vec3f(0.0); }
  var attenuation = vec3f(1.0);
  for (var depth = 0u; depth < state.depth; depth = depth + 1u) {
    attenuation = attenuation * ddgiMediumExtinctionForSegment(
      classified.state.tri[depth],
      classified.state.attenuationColor[depth],
      classified.state.attenuationDistance[depth],
      classified.state.authoredThickness[depth],
      classified.state.thicknessMapScale[depth],
      classified.state.scatter[depth].rgb,
      state.distance[depth],
    );
  }
  return clamp(attenuation, vec3f(0.0), vec3f(1.0));
}

fn ddgiTraceShadowVisibility(origin: vec3f, dir: vec3f, tMax: f32) -> vec3f {
  let containing = ddgiClassifyContainingMedia(
    origin,
    dir,
    opticalSourceFeatureInvalid(),
    true,
  );
  if (containing.valid == 0u) { return vec3f(0.0); }
  var mediumState = containing.state;
  let walkRay = Ray(origin, dir);
  var exclusiveMinT = 0.0;
  var visibility = vec3f(1.0);
  let surfaceBudget = ddgiWorldSurfaceBudget();
  let useTlas = ddgiTrace.bvhMode == 1u && ddgiTrace.tlasNodeCount > 0u;

  for (var layer = 0u; layer < surfaceBudget; layer = layer + 1u) {
    if (max(max(visibility.x, visibility.y), visibility.z) <= 0.0) {
      return vec3f(0.0);
    }
    if (tMax <= exclusiveMinT) {
      return clamp(
        visibility * ddgiShadowEndpointTransmission(
          origin, dir, tMax, mediumState,
        ),
        vec3f(0.0), vec3f(1.0),
      );
    }
    var hit = traceSceneFirstHitDdgiAt(walkRay, exclusiveMinT);
    if (!hit.didHit || hit.dist >= tMax) {
      if (tMax > DDGI_MAX_FINITE_F32) {
        return select(
          vec3f(0.0),
          clamp(visibility, vec3f(0.0), vec3f(1.0)),
          mediumState.depth == 0u,
        );
      }
      if (!ddgiAccumulateMediumDistance(
        &mediumState, max(tMax - exclusiveMinT, 0.0),
      )) { return vec3f(0.0); }
      return clamp(
        visibility * ddgiShadowEndpointTransmission(
          origin, dir, tMax, mediumState,
        ),
        vec3f(0.0), vec3f(1.0),
      );
    }
    if (hit.indices.w >= arrayLength(&bvh_materialId)) {
      return vec3f(0.0);
    }
    let materialId = bvh_materialId[hit.indices.w];
    if (materialId >= DDGI_MAX_MATERIALS) { return vec3f(0.0); }
    let entry = materials[materialId];
    let hasTransmission = entry.transmission > 0.0;
    var acceptedT = hit.dist;
    if (hasTransmission) {
      let exactHit = traceSceneRetraceOpticalHit(
        ddgiTrace.bvhMode,
        ddgiTrace.tlasNodeCount,
        walkRay,
        hit,
        exclusiveMinT,
      );
      if (!exactHit.hit || !(exactHit.t > exclusiveMinT)) {
        return vec3f(0.0);
      }
      acceptedT = exactHit.t;
      hit.normal = exactHit.normal;
      hit.barycoord = exactHit.bary;
      hit.side = exactHit.side;
      hit.dist = exactHit.t;
      let triangle = sceneLoadOpticalWorldTriangle(
        useTlas, hit.indices.w, hit.instanceIndex,
      );
      if (triangle.valid == 0u) { return vec3f(0.0); }
      hit.uv = exactHit.bary.x * triangle.uvA +
        exactHit.bary.y * triangle.uvB +
        exactHit.bary.z * triangle.uvC;
    }
    if (!ddgiAccumulateMediumDistance(
      &mediumState, acceptedT - exclusiveMinT,
    )) { return vec3f(0.0); }

    if ((entry.flags & MATERIAL_FLAG_CAST_SHADOW_DISABLED) != 0u) {
      // Native castShadow=false semantics: advance from the exact accepted
      // intersection without changing visibility or optical-medium topology.
    } else if (!hasTransmission) {
      let alphaT = ddgiAlphaShadowTransmittanceForHit(hit);
      if (alphaT <= 0.0) { return vec3f(0.0); }
      visibility = visibility * vec3f(alphaT);
    } else {
      let coverage = ddgiOpticalCoverageForHit(hit);
      let boundaryId = sceneOpticalEncodedBoundaryId(
        useTlas, hit.indices.w, hit.instanceIndex,
      );
      let representedId = sceneOpticalRepresentedPrimitiveInstanceId(
        useTlas, hit.indices.w, hit.instanceIndex,
      );
      if (representedId == 0u) { return vec3f(0.0); }
      let bulkMedium = boundaryId != 0u;
      if (bulkMedium) {
        let event = traceSceneOpticalBoundaryEvent(
          ddgiTrace.bvhMode,
          ddgiTrace.tlasNodeCount,
          walkRay,
          exclusiveMinT,
          opticalSourceFeatureInvalid(),
        );
        if (
          event.status != OPTICAL_BOUNDARY_EVENT_CROSSING ||
          event.t != acceptedT ||
          event.encodedBoundaryId != boundaryId ||
          event.representedPrimitiveInstanceId != representedId ||
          coverage != 1.0
        ) { return vec3f(0.0); }
      }
      let mappedTransmission = clamp(
        ddgiSampleTransmissionMapForHit(hit, entry.transmission),
        0.0, 1.0,
      );
      if (!bulkMedium) {
        visibility = visibility * mix(
          vec3f(1.0),
          ddgiShadowThinSheetTransmission(hit, dir) *
            vec3f(mappedTransmission),
          vec3f(coverage),
        );
      } else if (hit.side >= 0.0) {
        if (mediumState.depth >= DDGI_OPTICAL_MEDIUM_CAPACITY) {
          return vec3f(0.0);
        }
        visibility = visibility *
          ddgiShadowFaceTransmission(hit, dir, true) *
          vec3f(mappedTransmission);
        let depth = mediumState.depth;
        mediumState.boundaryId[depth] = boundaryId;
        mediumState.representedId[depth] = representedId;
        mediumState.distance[depth] = 0.0;
        mediumState.transmissionPaid[depth] = 1u;
        mediumState.depth = depth + 1u;
      } else {
        if (mediumState.depth == 0u) { return vec3f(0.0); }
        let top = mediumState.depth - 1u;
        if (
          mediumState.boundaryId[top] != boundaryId ||
          mediumState.representedId[top] != representedId
        ) { return vec3f(0.0); }
        visibility = visibility * ddgiMediumExtinctionForSegment(
          hit.indices.w,
          clamp(entry.attenuationColor, vec3f(0.0), vec3f(1.0)),
          entry.attenuationDistance,
          select(
            -1.0,
            max(materialOpticalThickness(hit.indices.w), 0.0),
            materialOpticalHasAuthoredThickness(hit.indices.w) &&
              materialOpticalThickness(hit.indices.w) > 0.0,
          ),
          select(
            1.0,
            ddgiSampleThicknessMapFactorForHit(hit).x,
            materialOpticalHasAuthoredThickness(hit.indices.w) &&
              materialOpticalThickness(hit.indices.w) > 0.0 &&
              ddgiSampleThicknessMapFactorForHit(hit).y >= 0.5,
          ),
          ddgiSampleVolumeScatteringControls(hit.indices.w).rgb,
          mediumState.distance[top],
        );
        visibility = visibility *
          ddgiShadowFaceTransmission(hit, dir, false);
        if (mediumState.transmissionPaid[top] == 0u) {
          visibility = visibility * vec3f(mappedTransmission);
        }
        mediumState.depth = top;
      }
    }
    exclusiveMinT = acceptedT;
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
  let unboundedShadowDistance = bitcast<f32>(0x7f800000u);
  return ddgiTraceShadowVisibility(origin, sunDir, unboundedShadowDistance);
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
  let base = ddgiNormalizeOr(sunBase, vec3f(0.0, 1.0, 0.0));
  if (!ddgiFiniteF32(angularRadius) || angularRadius > 3.1415927) {
    return base;
  }
  let radius = max(angularRadius, 0.0);
  if (radius <= 0.0) {
    return base;
  }
  let relative = (hitPos - gridParams.origin) / gridParams.spacing;
  if (!ddgiAtlasFiniteVec3(relative)) {
    return base;
  }
  // Hash finite f32 bit patterns directly. This preserves deterministic
  // grid-relative spatial jitter without an out-of-range float-to-i32 cast.
  let scaledRelative = relative * 1024.0;
  if (!ddgiAtlasFiniteVec3(scaledRelative)) { return base; }
  let quant = floor(scaledRelative);
  if (!ddgiAtlasFiniteVec3(quant)) { return base; }
  let seed =
    bitcast<u32>(quant.x) * 0x9E3779B9u ^
    bitcast<u32>(quant.y) * 0x85EBCA6Bu ^
    bitcast<u32>(quant.z) * 0xC2B2AE35u ^
    0x44444749u;
  let xi = vec2f(
    ddgiSoftSunHashToF32(seed ^ 0x53474341u),
    ddgiSoftSunHashToF32(seed ^ 0x4f495431u),
  );
  let upRef = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), abs(base.y) < 0.99);
  let tangent = ddgiNormalizeOr(cross(upRef, base), vec3f(1.0, 0.0, 0.0));
  let bitangent = cross(base, tangent);
  let r = radius * sqrt(xi.x);
  let phi = 6.2831853 * xi.y;
  return ddgiNormalizeOr(
    base + tangent * (r * cos(phi)) + bitangent * (r * sin(phi)),
    base,
  );
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
    let nrmRaw = ddgiEmitterTris[base + 3u].xyz;
    let area = ddgiEmitterTris[base + 3u].w;
    let scalarLe = ddgiEmitterTris[base + 4u].xyz;
    if (
      !ddgiAtlasFiniteVec3(vA) || !ddgiAtlasFiniteVec3(vB) ||
      !ddgiAtlasFiniteVec3(vC) || !ddgiCanNormalize(nrmRaw) ||
      !ddgiFiniteF32(area) || !(area > 0.0) ||
      !ddgiAtlasFiniteVec3(scalarLe) || any(scalarLe < vec3f(0.0))
    ) { continue; }
    let nrm = ddgiNormalizeOr(nrmRaw, vec3f(0.0, 1.0, 0.0));
    let encodedEmitterFlags = ddgiMaterialMetaExactU32(ddgiEmitterTris[base + 4u].w);
    let emitterFlags = select(
      encodedEmitterFlags,
      0u,
      encodedEmitterFlags == 0xffffffffu || encodedEmitterFlags > 3u,
    );
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

const DDGI_GLASS_MAX_INTERFACES: u32 = 8u;

fn ddgiGlassPcgNext(state: ptr<function, u32>) -> u32 {
  *state = *state * 747796405u + 2891336453u;
  let word = ((*state >> ((*state >> 28u) + 4u)) ^ *state) * 277803737u;
  return (word >> 22u) ^ word;
}

fn ddgiGlassRandF32(state: ptr<function, u32>) -> f32 {
  return f32(ddgiGlassPcgNext(state) >> 8u) / 16777216.0;
}

fn ddgiAnisotropyFrame(
  n: vec3f,
  tangentBasis: vec3f,
  bitangentBasis: vec3f,
  rotation: f32,
) -> DdgiMaterialTangentFrame {
  var tangent = tangentBasis - n * dot(n, tangentBasis);
  if (!ddgiCanNormalize(tangent)) {
    let up = select(
      vec3f(0.0, 1.0, 0.0),
      vec3f(1.0, 0.0, 0.0),
      abs(n.y) > 0.999,
    );
    tangent = ddgiNormalizeOr(cross(up, n), vec3f(1.0, 0.0, 0.0));
  } else {
    tangent = ddgiNormalizeOr(tangent, vec3f(1.0, 0.0, 0.0));
  }
  var bitangent = bitangentBasis - n * dot(n, bitangentBasis) -
    tangent * dot(tangent, bitangentBasis);
  bitangent = ddgiNormalizeOr(
    bitangent,
    ddgiFallbackBitangentForNormal(n, tangent),
  );
  let c = cos(rotation);
  let s = sin(rotation);
  return DdgiMaterialTangentFrame(
    ddgiNormalizeOr(tangent * c + bitangent * s, tangent),
    ddgiNormalizeOr(-tangent * s + bitangent * c, bitangent),
  );
}

fn ddgiDielectricAxes(roughness: f32, anisotropy: f32) -> vec2f {
  let alpha = max(clamp(roughness, 0.0, 1.0) *
    clamp(roughness, 0.0, 1.0), 1e-6);
  let aspect = sqrt(1.0 - 0.9 * clamp(anisotropy, 0.0, 1.0));
  return vec2f(alpha / aspect, alpha * aspect);
}

fn ddgiSampleVisibleGgxNormalAnisotropic(
  n: vec3f,
  tangentBasis: vec3f,
  bitangentBasis: vec3f,
  wo: vec3f,
  roughness: f32,
  anisotropy: f32,
  rotation: f32,
  xi: vec2f,
) -> vec3f {
  let frame = ddgiAnisotropyFrame(
    n, tangentBasis, bitangentBasis, rotation,
  );
  let axes = ddgiDielectricAxes(roughness, anisotropy);
  let woT = vec3f(
    dot(wo, frame.tangent),
    dot(wo, frame.bitangent),
    dot(wo, n),
  );
  let vh = ddgiNormalizeOr(
    vec3f(axes.x * woT.x, axes.y * woT.y, woT.z),
    vec3f(0.0, 0.0, 1.0),
  );
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
  diskY = (1.0 - blend) *
    sqrt(max(0.0, 1.0 - diskX * diskX)) + blend * diskY;
  let nh = diskX * t1 + diskY * t2 +
    sqrt(max(0.0, 1.0 - diskX * diskX - diskY * diskY)) * vh;
  let wmT = ddgiNormalizeOr(
    vec3f(axes.x * nh.x, axes.y * nh.y, max(0.0, nh.z)),
    vec3f(0.0, 0.0, 1.0),
  );
  return ddgiNormalizeOr(
    wmT.x * frame.tangent + wmT.y * frame.bitangent + wmT.z * n,
    n,
  );
}

fn ddgiDistributionGgxAnisotropic(
  n: vec3f,
  frame: DdgiMaterialTangentFrame,
  h: vec3f,
  axes: vec2f,
) -> f32 {
  let nDotH = dot(n, h);
  if (nDotH <= 0.0 || axes.x <= 0.0 || axes.y <= 0.0) {
    return 0.0;
  }
  let tx = dot(frame.tangent, h) / axes.x;
  let by = dot(frame.bitangent, h) / axes.y;
  let denominator = tx * tx + by * by + nDotH * nDotH;
  return 1.0 / (PI * axes.x * axes.y * denominator * denominator);
}

fn ddgiSmithG1GgxAnisotropic(
  n: vec3f,
  frame: DdgiMaterialTangentFrame,
  v: vec3f,
  axes: vec2f,
) -> f32 {
  let nDotV = dot(n, v);
  if (nDotV <= 0.0 || axes.x <= 0.0 || axes.y <= 0.0) {
    return 0.0;
  }
  let tDotV = dot(frame.tangent, v);
  let bDotV = dot(frame.bitangent, v);
  let root = sqrt(
    axes.x * axes.x * tDotV * tDotV +
    axes.y * axes.y * bDotV * bDotV +
    nDotV * nDotV,
  );
  return (2.0 * nDotV) / (nDotV + root);
}

struct DdgiDielectricInterfaceSample {
  reflectedDirection: vec3f,
  transmittedDirection: vec3f,
  reflectionWeight: vec3f,
  transmissionWeight: vec3f,
  transmissionValid: u32,
  valid: u32,
}

// Samples one visible microfacet and evaluates both reciprocal optical lobes.
// The caller selects the discrete reflection/transmission family after seeing
// whether refraction exists, so total internal reflection has probability one
// and cannot disappear into a preselected invalid transmission branch.
fn ddgiSampleDielectricInterface(
  hit: IntersectionResult,
  n: vec3f,
  tangentBasis: vec3f,
  bitangentBasis: vec3f,
  wo: vec3f,
  roughness: f32,
  anisotropy: f32,
  anisotropyRotation: f32,
  etaIncidentRgb: vec3f,
  etaTargetRgb: vec3f,
  frontFacing: bool,
  layer: vec4f,
  channel: u32,
  xi: vec2f,
) -> DdgiDielectricInterfaceSample {
  var out: DdgiDielectricInterfaceSample;
  out.reflectedDirection = vec3f(0.0);
  out.transmittedDirection = vec3f(0.0);
  out.reflectionWeight = vec3f(0.0);
  out.transmissionWeight = vec3f(0.0);
  out.transmissionValid = 0u;
  out.valid = 0u;
  let etaIncident = ddgiRgbChannel(etaIncidentRgb, channel);
  let etaTarget = ddgiRgbChannel(etaTargetRgb, channel);
  let nDotWo = dot(n, wo);
  if (!(nDotWo > 0.0) || etaIncident <= 0.0 || etaTarget <= 0.0) {
    return out;
  }

  let rough = clamp(roughness, 0.0, 1.0);
  let aniso = clamp(anisotropy, 0.0, 1.0);
  var wm = n;
  if (rough > 0.0) {
    if (aniso > 0.0) {
      wm = ddgiSampleVisibleGgxNormalAnisotropic(
        n, tangentBasis, bitangentBasis, wo,
        rough, aniso, anisotropyRotation, xi,
      );
    } else {
      wm = ddgiSampleVisibleGgxNormal(n, wo, rough, xi);
    }
    if (dot(wm, n) < 0.0) { wm = -wm; }
  }
  let woDotM = dot(wo, wm);
  if (!(woDotM > 0.0)) { return out; }
  let reflectedDirection = ddgiNormalizeOr(reflect(-wo, wm), n);
  let nDotWr = dot(n, reflectedDirection);
  if (!(nDotWr > 0.0)) { return out; }

  let refractedRaw = refract(-wo, wm, etaIncident / etaTarget);
  let tir = dot(refractedRaw, refractedRaw) <= 1e-12;
  var reflectanceRgb = vec3f(
    ddgiDielectricFresnelExact(
      woDotM, etaIncidentRgb.r, etaTargetRgb.r,
    ),
    ddgiDielectricFresnelExact(
      woDotM, etaIncidentRgb.g, etaTargetRgb.g,
    ),
    ddgiDielectricFresnelExact(
      woDotM, etaIncidentRgb.b, etaTargetRgb.b,
    ),
  );
  var transmittanceRgb = vec3f(1.0) - reflectanceRgb;
  let film = materialThinFilmResponse(
    hit.indices.w, frontFacing, woDotM,
  );
  if (film.present != 0u) {
    reflectanceRgb = film.reflectance;
    transmittanceRgb = film.transmittance;
  }
  let etaRatioRgb = max(etaIncidentRgb, vec3f(1e-6)) /
    max(etaTargetRgb, vec3f(1e-6));
  let sin2TargetRgb = etaRatioRgb * etaRatioRgb *
    (1.0 - woDotM * woDotM);
  let tirRgb = sin2TargetRgb >= vec3f(1.0);
  reflectanceRgb = select(reflectanceRgb, vec3f(1.0), tirRgb);
  transmittanceRgb = select(transmittanceRgb, vec3f(0.0), tirRgb);
  let layerTransferRgb = ddgiFaceLayerTransmission(layer);
  reflectanceRgb = clamp(
    reflectanceRgb, vec3f(0.0), vec3f(1.0),
  ) * layerTransferRgb;
  transmittanceRgb = clamp(
    transmittanceRgb, vec3f(0.0), vec3f(1.0),
  ) * layerTransferRgb;
  let reflectance = ddgiRgbChannel(reflectanceRgb, channel);
  let transmittance = ddgiRgbChannel(
    transmittanceRgb, channel,
  );

  var reflectionBaseWeight = 1.0;
  if (rough > 0.0 && reflectance > 0.0) {
    var distribution = 0.0;
    var g1o = 0.0;
    var g1i = 0.0;
    if (aniso > 0.0) {
      let frame = ddgiAnisotropyFrame(
        n, tangentBasis, bitangentBasis, anisotropyRotation,
      );
      let axes = ddgiDielectricAxes(rough, aniso);
      distribution = ddgiDistributionGgxAnisotropic(n, frame, wm, axes);
      g1o = ddgiSmithG1GgxAnisotropic(n, frame, wo, axes);
      g1i = ddgiSmithG1GgxAnisotropic(
        n, frame, reflectedDirection, axes,
      );
    } else {
      let alpha = rough * rough;
      let alpha2 = alpha * alpha;
      distribution = ddgiDistributionGGX(dot(n, wm), rough);
      g1o = ddgiSmithG1GGX(nDotWo, alpha2);
      g1i = ddgiSmithG1GGX(nDotWr, alpha2);
    }
    let directionPdf = distribution * g1o / (4.0 * nDotWo);
    let frBase = distribution * g1o * g1i /
      (4.0 * nDotWo * nDotWr);
    if (directionPdf > 0.0 && frBase > 0.0) {
      reflectionBaseWeight = frBase * nDotWr / directionPdf;
    } else {
      reflectionBaseWeight = 0.0;
    }
  }
  out.reflectedDirection = reflectedDirection;
  out.reflectionWeight = reflectanceRgb * reflectionBaseWeight;
  out.valid = 1u;

  if (!tir && transmittance > 0.0) {
    let transmittedDirection = ddgiNormalizeOr(refractedRaw, -n);
    let nDotWt = abs(dot(n, transmittedDirection));
    let wiDotM = dot(transmittedDirection, wm);
    let etap = etaTarget / etaIncident;
    let denominator = wiDotM + woDotM / etap;
    if (
      dot(n, transmittedDirection) < 0.0 && nDotWt > 0.0 &&
      wiDotM < 0.0 && abs(denominator) > 1e-8
    ) {
      var transmissionBaseWeight = 1.0 / (etap * etap);
      if (rough > 0.0) {
        var distribution = 0.0;
        var g1o = 0.0;
        var g1i = 0.0;
        if (aniso > 0.0) {
          let frame = ddgiAnisotropyFrame(
            n, tangentBasis, bitangentBasis, anisotropyRotation,
          );
          let axes = ddgiDielectricAxes(rough, aniso);
          distribution = ddgiDistributionGgxAnisotropic(
            n, frame, wm, axes,
          );
          g1o = ddgiSmithG1GgxAnisotropic(n, frame, wo, axes);
          g1i = ddgiSmithG1GgxAnisotropic(
            n, frame, -transmittedDirection, axes,
          );
        } else {
          let alpha = rough * rough;
          let alpha2 = alpha * alpha;
          distribution = ddgiDistributionGGX(dot(n, wm), rough);
          g1o = ddgiSmithG1GGX(nDotWo, alpha2);
          g1i = ddgiSmithG1GGX(nDotWt, alpha2);
        }
        let microfacetPdf = distribution * g1o * abs(woDotM) /
          nDotWo;
        let directionPdf = microfacetPdf * abs(wiDotM) /
          (denominator * denominator);
        let ftBase = distribution * g1o * g1i *
          abs(wiDotM * woDotM /
            (nDotWt * nDotWo * denominator * denominator)) /
          (etap * etap);
        if (directionPdf > 0.0 && ftBase > 0.0) {
          transmissionBaseWeight = ftBase * nDotWt / directionPdf;
        } else {
          transmissionBaseWeight = 0.0;
        }
      }
      let transmissionWeight =
        transmittanceRgb * transmissionBaseWeight;
      let maximumTransmissionWeight = max(
        transmissionWeight.x,
        max(transmissionWeight.y, transmissionWeight.z),
      );
      if (
        ddgiRgbChannel(transmissionWeight, channel) > 0.0 &&
        maximumTransmissionWeight < 1e30
      ) {
        out.transmittedDirection = transmittedDirection;
        out.transmissionWeight = transmissionWeight;
        out.transmissionValid = 1u;
      }
    }
  }
  let maximumReflectionWeight = max(
    out.reflectionWeight.x,
    max(out.reflectionWeight.y, out.reflectionWeight.z),
  );
  if (
    any(out.reflectionWeight != out.reflectionWeight) ||
    any(out.reflectionWeight < vec3f(0.0)) ||
    maximumReflectionWeight >= 1e30
  ) {
    out.reflectionWeight = vec3f(0.0);
  }
  return out;
}

struct DdgiProbeSurfaceSource {
  // Emission is a deterministic surface source. It is never part of the
  // authored opaque-vs-transmission material split.
  emissionLo: vec3f,
  // Direct, diffuse-feedback, and baked response before the unpaid (1-t)
  // material weight is applied by the dielectric suffix.
  opaqueLo: vec3f,
  // Clearcoat and sheen are not represented by the exact base-dielectric
  // interface walk. Keep only those same-side local-reflection shares here.
  persistentReflectionLo: vec3f,
  // Fast-path response for a non-glass terminal. This preserves the complete
  // pre-split base plus extension response exactly.
  terminalLo: vec3f,
}

fn ddgiEvaluateProbeSurfaceRadiance(
  hit: IntersectionResult,
  hitWorldPos: vec3f,
  rayDirection: vec3f,
  mat: MaterialEntry,
  probeMat: DdgiProbeHitMaterial,
  probeNormal: vec3f,
  seed: u32,
) -> DdgiProbeSurfaceSource {
  var out: DdgiProbeSurfaceSource;
  out.emissionLo = vec3f(0.0);
  out.opaqueLo = vec3f(0.0);
  out.persistentReflectionLo = vec3f(0.0);
  out.terminalLo = vec3f(0.0);
  let direct = evalDirectLighting(
    hitWorldPos, probeNormal, seed ^ 0xa511e9b3u,
  ) + ddgiEmitterNEE(
    hitWorldPos, probeNormal, seed ^ 0x63d83595u,
  );
  let indirect = ddgiFeedbackAt(hitWorldPos, probeNormal);
  let indirectGated = select(
    vec3f(0.0), indirect, frameParams.indirectFeedback != 0u,
  );
  let baseSpecularWeight = ddgiProbeBaseSpecularWeight(probeMat);
  let clearcoatWeight = ddgiProbeClearcoatWeight(probeMat);
  let sheenWeight = ddgiProbeSheenWeight(probeMat);
  let extensionSpecularWeight = ddgiProbeExtensionSpecularWeight(probeMat);
  var baseIndirectRadiance = indirectGated * probeMat.albedo * (1.0 / PI);
  var reflectionIndirectRadiance = vec3f(0.0);
  var persistentReflectionIndirectRadiance = vec3f(0.0);
  if (
    extensionSpecularWeight > 0.0 &&
    frameParams.indirectFeedback != 0u
  ) {
    let reflectedDirection = ddgiNormalizeOr(
      rayDirection - 2.0 * dot(rayDirection, probeNormal) * probeNormal,
      probeNormal,
    );
    let specularIrradiance = ddgiFeedbackAt(
      hitWorldPos, reflectedDirection,
    );
    let clearcoatReflectedDirection = ddgiNormalizeOr(
      rayDirection - 2.0 *
        dot(rayDirection, probeMat.clearcoatNormal) *
        probeMat.clearcoatNormal,
      probeMat.clearcoatNormal,
    );
    let clearcoatIrradiance = ddgiFeedbackAt(
      hitWorldPos, clearcoatReflectedDirection,
    );
    let f0Tint = ddgiProbeSpecularTint(
      probeMat, max(0.0, dot(-rayDirection, probeNormal)),
    );
    let baseSpecularLo = f0Tint * specularIrradiance * (1.0 / PI);
    let clearcoatLo = vec3f(0.04) * clearcoatIrradiance * (1.0 / PI);
    let sheenLo = clamp(
      probeMat.sheen.rgb, vec3f(0.0), vec3f(1.0),
    ) * specularIrradiance * (1.0 / PI);
    let lobeWeightSum = baseSpecularWeight + clearcoatWeight + sheenWeight;
    if (lobeWeightSum > 0.0) {
      let specularIndirectLo =
        (baseSpecularLo * baseSpecularWeight +
         clearcoatLo * clearcoatWeight +
         sheenLo * sheenWeight) / lobeWeightSum;
      // Preserve the former mix algebra while applying thin-film T only to the
      // base share. The reflected share already contains absolute film R in F0.
      baseIndirectRadiance = baseIndirectRadiance *
        (1.0 - extensionSpecularWeight);
      reflectionIndirectRadiance = specularIndirectLo *
        extensionSpecularWeight;
      // The exact dielectric interface sampler below owns the base-specular
      // share (including absolute thin-film R). Retain only clearcoat and sheen
      // in the glass-local reflection family so that share cannot be counted by
      // both the proxy and the sampled continuation.
      persistentReflectionIndirectRadiance =
        (clearcoatLo * clearcoatWeight + sheenLo * sheenWeight) /
        lobeWeightSum * extensionSpecularWeight;
    }
  }
  let directRadiance = direct * probeMat.albedo * (1.0 / PI);
  let bakedOutgoing = probeMat.albedo * (1.0 / PI) *
    ddgiSampleLightMapIrradiance(hit);
  let scalarSurfaceEmission = max(mat.emissive, vec3f(0.0));
  let surfaceEmission = select(
    vec3f(0.0),
    ddgiSampleEmissiveMap(hit, scalarSurfaceEmission),
    hit.side >= 0.0 ||
      (mat.flags & MATERIAL_FLAG_DOUBLE_SIDED) != 0u,
  );
  out.emissionLo = surfaceEmission * probeMat.layerTransmission;
  out.opaqueLo =
    (directRadiance + baseIndirectRadiance + bakedOutgoing) *
    probeMat.layerTransmission;
  out.persistentReflectionLo = persistentReflectionIndirectRadiance *
    probeMat.reflectionLayerTransmission;
  out.terminalLo = out.emissionLo + out.opaqueLo +
    reflectionIndirectRadiance * probeMat.reflectionLayerTransmission;
  if ((mat.flags & MATERIAL_FLAG_UNLIT) != 0u) {
    // Unlit is an authored outgoing-radiance source, even if the same material
    // also carries transmission/glass metadata. It terminates the suffix and is
    // never charged by (1-t).
    out.emissionLo = probeMat.albedo * probeMat.layerTransmission;
    out.opaqueLo = vec3f(0.0);
    out.persistentReflectionLo = vec3f(0.0);
    out.terminalLo = out.emissionLo;
  }
  return out;
}

fn ddgiDiagonalTransfer(value: vec3f) -> mat3x3f {
  return mat3x3f(
    vec3f(value.x, 0.0, 0.0),
    vec3f(0.0, value.y, 0.0),
    vec3f(0.0, 0.0, value.z),
  );
}

// Linear form of the directionally aggregated single-scatter contract. The
// source is driven by luminance, so a 3x3 operator is required to preserve its
// cross-channel redistribution while each dispersion lane traces independent
// geometry. Synthetic zero-thickness bulk consumes the full represented
// segment. Positive authored thickness supplies a transport-distance cap, and
// its readable thickness map scales that cap for both absorption and scatter.
fn ddgiMediumSegmentTransfer(
  triIndex: u32,
  attenuationColor: vec3f,
  attenuationDistance: f32,
  authoredThickness: f32,
  thicknessMapScale: f32,
  scatter: vec4f,
  mediumAlbedo: vec3f,
  segmentLength: f32,
) -> mat3x3f {
  if (segmentLength < 0.0 || segmentLength != segmentLength) {
    return ddgiDiagonalTransfer(vec3f(0.0));
  }
  let hasAuthoredThickness = authoredThickness > 0.0 &&
    materialOpticalHasAuthoredThickness(triIndex);
  let mappedCap = authoredThickness * clamp(thicknessMapScale, 0.0, 1.0);
  let transportDistance = select(
    segmentLength,
    min(segmentLength, mappedCap),
    hasAuthoredThickness,
  );
  let rgbBeer = beerLambertTransmittanceRgb(
    clamp(attenuationColor, vec3f(0.0), vec3f(1.0)),
    attenuationDistance,
    transportDistance,
  );
  let absorption = clamp(materialSpectralAttenuation(
    triIndex, transportDistance, rgbBeer,
  ), vec3f(0.0), vec3f(1.0));
  if (!(transportDistance > 0.0)) {
    return ddgiDiagonalTransfer(absorption);
  }
  let sigmaS = max(scatter.rgb, vec3f(0.0));
  let sigmaA = -log(max(absorption, vec3f(1e-30))) /
    transportDistance;
  let sigmaT = sigmaA + sigmaS;
  let transmittance = absorption *
    ddgiHomogeneousBeerTransmittanceRgb(sigmaS, transportDistance);
  var scatterAlbedo = vec3f(0.0);
  if (sigmaT.x > 0.0) { scatterAlbedo.x = sigmaS.x / sigmaT.x; }
  if (sigmaT.y > 0.0) { scatterAlbedo.y = sigmaS.y / sigmaT.y; }
  if (sigmaT.z > 0.0) { scatterAlbedo.z = sigmaS.z / sigmaT.z; }
  let sourceScale = max(mediumAlbedo, vec3f(0.0)) * scatterAlbedo *
    (vec3f(1.0) - transmittance) *
    ddgiHenyeyGreensteinPhase(0.0, scatter.a);
  return mat3x3f(
    vec3f(transmittance.x, 0.0, 0.0) + sourceScale * 0.2126,
    vec3f(0.0, transmittance.y, 0.0) + sourceScale * 0.7152,
    vec3f(0.0, 0.0, transmittance.z) + sourceScale * 0.0722,
  );
}

fn ddgiTransferFinite(transfer: mat3x3f) -> bool {
  let maximum = max(
    max(
      max(abs(transfer[0].x), abs(transfer[0].y)),
      abs(transfer[0].z),
    ),
    max(
      max(
        max(abs(transfer[1].x), abs(transfer[1].y)),
        abs(transfer[1].z),
      ),
      max(
        max(abs(transfer[2].x), abs(transfer[2].y)),
        abs(transfer[2].z),
      ),
    ),
  );
  return
    all(transfer[0] == transfer[0]) &&
    all(transfer[1] == transfer[1]) &&
    all(transfer[2] == transfer[2]) &&
    maximum < 1e30;
}

fn ddgiTransferredChannel(
  transfer: mat3x3f,
  radiance: vec3f,
  channel: u32,
) -> f32 {
  return ddgiRgbChannel(transfer * radiance, channel);
}

fn ddgiOpticalStatePrefixEqual(
  a: DdgiOpticalMediumState,
  b: DdgiOpticalMediumState,
  count: u32,
) -> bool {
  if (a.depth < count || b.depth < count) { return false; }
  for (var i = 0u; i < count; i = i + 1u) {
    if (
      a.boundaryId[i] != b.boundaryId[i] ||
      a.representedId[i] != b.representedId[i]
    ) { return false; }
  }
  return true;
}

fn ddgiConsumeTopMediumSegment(
  state: ptr<function, DdgiOpticalMediumState>,
  segmentLength: f32,
) -> mat3x3f {
  if (
    segmentLength != segmentLength || segmentLength < 0.0 ||
    segmentLength > DDGI_MAX_FINITE_F32
  ) { return ddgiDiagonalTransfer(vec3f(0.0)); }
  if ((*state).depth == 0u) {
    return ddgiDiagonalTransfer(vec3f(1.0));
  }
  let top = (*state).depth - 1u;
  var transportLength = segmentLength;
  let authoredThickness = (*state).authoredThickness[top];
  if (
    authoredThickness > 0.0 &&
    materialOpticalHasAuthoredThickness((*state).tri[top])
  ) {
    let cap = authoredThickness * clamp(
      (*state).thicknessMapScale[top], 0.0, 1.0,
    );
    transportLength = min(
      segmentLength,
      max(cap - (*state).distance[top], 0.0),
    );
  }
  let nextDistance = (*state).distance[top] + transportLength;
  if (
    nextDistance != nextDistance || nextDistance < 0.0 ||
    nextDistance > DDGI_MAX_FINITE_F32
  ) { return ddgiDiagonalTransfer(vec3f(0.0)); }
  (*state).distance[top] = nextDistance;
  return ddgiMediumSegmentTransfer(
    (*state).tri[top],
    (*state).attenuationColor[top],
    (*state).attenuationDistance[top],
    (*state).authoredThickness[top],
    (*state).thicknessMapScale[top],
    (*state).scatter[top],
    (*state).albedo[top],
    transportLength,
  );
}

struct DdgiAcceptedProbeHit {
  valid: u32,
  hasTransmission: u32,
  boundaryId: u32,
  representedId: u32,
  hit: IntersectionResult,
  sourceFeature: OpticalSourceFeature,
};

fn ddgiAcceptProbeHit(
  ray: Ray,
  ordinaryHit: IntersectionResult,
  exclusiveMinT: f32,
  launchSourceFeature: OpticalSourceFeature,
) -> DdgiAcceptedProbeHit {
  var out: DdgiAcceptedProbeHit;
  out.valid = 0u;
  out.hasTransmission = 0u;
  out.boundaryId = 0u;
  out.representedId = 0u;
  out.hit = ordinaryHit;
  out.sourceFeature = opticalSourceFeatureInvalid();
  if (
    !ordinaryHit.didHit ||
    ordinaryHit.indices.w >= arrayLength(&bvh_materialId)
  ) { return out; }
  let materialId = bvh_materialId[ordinaryHit.indices.w];
  if (materialId >= DDGI_MAX_MATERIALS) { return out; }
  if (!(materials[materialId].transmission > 0.0)) {
    out.valid = 1u;
    return out;
  }

  let exactHit = traceSceneRetraceOpticalHit(
    ddgiTrace.bvhMode,
    ddgiTrace.tlasNodeCount,
    ray,
    ordinaryHit,
    exclusiveMinT,
  );
  let sourceFeature = sceneOpticalSourceFeatureForExactHit(
    ddgiTrace.bvhMode,
    ddgiTrace.tlasNodeCount,
    ordinaryHit,
    exactHit,
  );
  if (
    !exactHit.hit || !(exactHit.t > exclusiveMinT) ||
    sourceFeature.kind == OPTICAL_SOURCE_FEATURE_INVALID
  ) { return out; }
  let useTlas = ddgiTrace.bvhMode == 1u && ddgiTrace.tlasNodeCount > 0u;
  let triangle = sceneLoadOpticalWorldTriangle(
    useTlas, ordinaryHit.indices.w, ordinaryHit.instanceIndex,
  );
  if (triangle.valid == 0u) { return out; }
  out.hit.normal = exactHit.normal;
  out.hit.barycoord = exactHit.bary;
  out.hit.side = exactHit.side;
  out.hit.dist = exactHit.t;
  out.hit.uv = exactHit.bary.x * triangle.uvA +
    exactHit.bary.y * triangle.uvB +
    exactHit.bary.z * triangle.uvC;
  out.boundaryId = sceneOpticalEncodedBoundaryId(
    useTlas, ordinaryHit.indices.w, ordinaryHit.instanceIndex,
  );
  out.representedId = sceneOpticalRepresentedPrimitiveInstanceId(
    useTlas, ordinaryHit.indices.w, ordinaryHit.instanceIndex,
  );
  if (out.representedId == 0u) { return out; }
  if (out.boundaryId != 0u) {
    let event = traceSceneOpticalBoundaryEvent(
      ddgiTrace.bvhMode,
      ddgiTrace.tlasNodeCount,
      ray,
      exclusiveMinT,
      launchSourceFeature,
    );
    if (
      event.status != OPTICAL_BOUNDARY_EVENT_CROSSING ||
      event.t != exactHit.t ||
      event.encodedBoundaryId != out.boundaryId ||
      event.representedPrimitiveInstanceId != out.representedId
    ) { return out; }
  }
  out.valid = 1u;
  out.hasTransmission = 1u;
  out.sourceFeature = sourceFeature;
  return out;
}

// Bounded full dielectric suffix for one correlated RGB channel. The random
// state intentionally starts from the same seed for R/G/B; dispersion changes
// the optical result, not the underlying random numbers.
fn ddgiTraceGlassChannel(
  firstHit: IntersectionResult,
  firstPos: vec3f,
  firstSegmentDistance: f32,
  incidentDirection: vec3f,
  firstShadingNormal: vec3f,
  firstMaterial: DdgiProbeHitMaterial,
  firstSurfaceSource: DdgiProbeSurfaceSource,
  firstSourceFeature: OpticalSourceFeature,
  firstContainingMedia: DdgiContainingMedia,
  channel: u32,
  seed: u32,
) -> f32 {
  if (firstContainingMedia.valid == 0u) { return 0.0; }
  var rng = ddgiPcgHashU32(seed ^ 0x44444749u);
  var prefixTransfer = ddgiDiagonalTransfer(vec3f(1.0));
  var accumulatedRadiance = 0.0;
  var rayDirection = ddgiNormalizeOr(
    incidentDirection, vec3f(0.0, 0.0, 1.0),
  );
  var currentHit = firstHit;
  var currentPos = firstPos;
  var currentShadingNormal = firstShadingNormal;
  var currentMaterial = firstMaterial;
  var currentSurfaceSource = firstSurfaceSource;
  var currentSourceFeature = firstSourceFeature;
  var mediumState = firstContainingMedia.state;
  var interfaceCount = 0u;

  if (mediumState.depth > 0u && firstSegmentDistance > 0.0) {
    prefixTransfer = prefixTransfer * ddgiConsumeTopMediumSegment(
      &mediumState, firstSegmentDistance,
    );
    if (!ddgiTransferFinite(prefixTransfer)) { return 0.0; }
  }

  // Eight interface events plus one terminal opaque/environment inspection.
  for (
    var depth = 0u;
    depth <= DDGI_GLASS_MAX_INTERFACES;
    depth = depth + 1u
  ) {
    if (currentHit.indices.w >= arrayLength(&bvh_materialId)) {
      return accumulatedRadiance;
    }
    let currentMaterialId = bvh_materialId[currentHit.indices.w];
    if (currentMaterialId >= DDGI_MAX_MATERIALS) {
      return accumulatedRadiance;
    }
    let currentEntry = materials[currentMaterialId];
    let hasTransmission = currentEntry.transmission > 0.0;
    if ((currentEntry.flags & MATERIAL_FLAG_UNLIT) != 0u) {
      return accumulatedRadiance + ddgiTransferredChannel(
        prefixTransfer, currentSurfaceSource.emissionLo, channel,
      );
    }
    if (!hasTransmission) {
      return accumulatedRadiance + ddgiTransferredChannel(
        prefixTransfer, currentSurfaceSource.terminalLo, channel,
      );
    }
    // Surface emission is deterministic and has no matching dielectric NEE in
    // this probe estimator. Accumulate it for reflected/TIR and transmitted
    // arrivals alike before choosing the current optical family.
    accumulatedRadiance = accumulatedRadiance + ddgiTransferredChannel(
      prefixTransfer, currentSurfaceSource.emissionLo, channel,
    );
    if (interfaceCount >= DDGI_GLASS_MAX_INTERFACES) {
      return accumulatedRadiance;
    }

    if (
      currentSourceFeature.kind == OPTICAL_SOURCE_FEATURE_INVALID ||
      currentSourceFeature.representedPrimitiveInstanceId == 0u
    ) { return accumulatedRadiance; }
    let boundaryId = currentSourceFeature.encodedBoundaryId;
    let representedId =
      currentSourceFeature.representedPrimitiveInstanceId;
    let hasBulkTopology = boundaryId != 0u;
    let entering = currentHit.side > 0.0;
    if (currentHit.side == 0.0) { return accumulatedRadiance; }
    var pairedExit = false;
    var pairedPaidExit = false;
    if (hasBulkTopology && !entering && mediumState.depth > 0u) {
      let top = mediumState.depth - 1u;
      pairedExit =
        mediumState.boundaryId[top] == boundaryId &&
        mediumState.representedId[top] == representedId;
      pairedPaidExit = pairedExit &&
        mediumState.transmissionPaid[top] != 0u;
    }
    if (hasBulkTopology && !entering && !pairedExit) {
      return accumulatedRadiance;
    }

    let alignedNormal = select(
      -currentShadingNormal,
      currentShadingNormal,
      dot(currentShadingNormal, currentHit.normal) >= 0.0,
    );
    let faceNormal = select(
      -alignedNormal,
      alignedNormal,
      dot(rayDirection, alignedNormal) < 0.0,
    );
    if (dot(rayDirection, faceNormal) >= -1e-6) {
      return accumulatedRadiance;
    }
    let tangentFrame = ddgiMaterialTangentFrameForHit(
      currentHit,
      faceNormal,
      DDGI_MATERIAL_MAP_ANISOTROPY_TEXEL_OFFSET,
    );
    var incidentIor = vec3f(1.0);
    if (mediumState.depth > 0u) {
      incidentIor = mediumState.ior[mediumState.depth - 1u];
    }
    var targetIor = currentMaterial.opticalIor;
    if (hasBulkTopology && !entering) {
      targetIor = vec3f(1.0);
      if (mediumState.depth > 1u) {
        targetIor = mediumState.ior[mediumState.depth - 2u];
      }
    }
    let layer = ddgiSampleFaceLayerControls(
      currentHit.indices.w, entering,
    );
    let xi = vec2f(
      ddgiGlassRandF32(&rng),
      ddgiGlassRandF32(&rng),
    );
    let interfaceSample = ddgiSampleDielectricInterface(
      currentHit,
      faceNormal,
      tangentFrame.tangent,
      tangentFrame.bitangent,
      -rayDirection,
      currentMaterial.roughness,
      currentMaterial.anisotropy.x,
      currentMaterial.anisotropy.y,
      incidentIor,
      targetIor,
      entering,
      layer,
      channel,
      xi,
    );
    interfaceCount = interfaceCount + 1u;
    if (interfaceSample.valid == 0u) { return accumulatedRadiance; }

    let mappedTransmission = clamp(
      currentMaterial.transmission * frameParams.glassMixScale,
      0.0,
      1.0,
    );
    let scalarTransmission = select(
      mappedTransmission,
      1.0,
      pairedPaidExit,
    );
    var transmissionPdf = 0.0;
    if (
      interfaceSample.transmissionValid != 0u &&
      scalarTransmission > 0.0
    ) {
      transmissionPdf = represented_bernoulli_probability_f32(
        scalarTransmission / (1.0 + scalarTransmission),
      );
    }
    let reflectionPdf = 1.0 - transmissionPdf;
    // Consume the family draw even for a channel-selective TIR event. The RGB
    // suffixes start from one seed, and retaining draw parity keeps lanes that
    // take the same reflected continuation correlated after dispersion makes
    // transmission invalid in only a subset of channels.
    let familyXi = ddgiGlassRandF32(&rng);
    let chooseTransmission =
      transmissionPdf > 0.0 && familyXi < transmissionPdf;
    // 0 = reflection or reciprocal thin sheet, 1 = bulk entry, 2 = bulk exit.
    var bulkTransmissionAction = 0u;
    if (!chooseTransmission) {
      let opaqueWeight = select(
        1.0 - mappedTransmission,
        0.0,
        pairedPaidExit,
      );
      accumulatedRadiance = accumulatedRadiance +
        ddgiTransferredChannel(
          prefixTransfer,
          currentSurfaceSource.opaqueLo * opaqueWeight +
            currentSurfaceSource.persistentReflectionLo,
          channel,
        ) / reflectionPdf;
      if (!(
        ddgiRgbChannel(interfaceSample.reflectionWeight, channel) > 0.0
      )) {
        return accumulatedRadiance;
      }
      prefixTransfer = prefixTransfer * ddgiDiagonalTransfer(
        interfaceSample.reflectionWeight / reflectionPdf,
      );
      rayDirection = interfaceSample.reflectedDirection;
    } else {
      prefixTransfer = prefixTransfer * ddgiDiagonalTransfer(
        interfaceSample.transmissionWeight *
          scalarTransmission / transmissionPdf,
      );
      rayDirection = interfaceSample.transmittedDirection;
      if (!hasBulkTopology) {
        // A zero-thickness sheet has a reciprocal virtual reverse boundary.
        // Internal Fresnel reflection remains in the same global interface
        // budget; no absorption distance or second scalar t is invented.
        var slabNormal = faceNormal;
        var slabFrontFacing = !entering;
        var slabExited = false;
        loop {
          if (interfaceCount >= DDGI_GLASS_MAX_INTERFACES) { break; }
          let slabLayer = ddgiSampleFaceLayerControls(
            currentHit.indices.w, slabFrontFacing,
          );
          let slabFrame = ddgiMaterialTangentFrameForHit(
            currentHit,
            slabNormal,
            DDGI_MATERIAL_MAP_ANISOTROPY_TEXEL_OFFSET,
          );
          let slabXi = vec2f(
            ddgiGlassRandF32(&rng),
            ddgiGlassRandF32(&rng),
          );
          let slabSample = ddgiSampleDielectricInterface(
            currentHit,
            slabNormal,
            slabFrame.tangent,
            slabFrame.bitangent,
            -rayDirection,
            ddgiFaceLayerRoughness(currentMaterial.baseRoughness, slabLayer),
            currentMaterial.anisotropy.x,
            currentMaterial.anisotropy.y,
            targetIor,
            incidentIor,
            slabFrontFacing,
            slabLayer,
            channel,
            slabXi,
          );
          interfaceCount = interfaceCount + 1u;
          if (slabSample.valid == 0u) { break; }
          let slabTransmissionPdf = select(
            0.0,
            represented_bernoulli_probability_f32(0.5),
            slabSample.transmissionValid != 0u,
          );
          let slabReflectionPdf = 1.0 - slabTransmissionPdf;
          let slabFamilyXi = ddgiGlassRandF32(&rng);
          let slabChooseTransmission =
            slabTransmissionPdf > 0.0 &&
            slabFamilyXi < slabTransmissionPdf;
          if (slabChooseTransmission) {
            prefixTransfer = prefixTransfer * ddgiDiagonalTransfer(
              slabSample.transmissionWeight / slabTransmissionPdf,
            );
            rayDirection = slabSample.transmittedDirection;
            slabExited = true;
            break;
          }
          if (!(
            ddgiRgbChannel(slabSample.reflectionWeight, channel) > 0.0
          )) { break; }
          prefixTransfer = prefixTransfer * ddgiDiagonalTransfer(
            slabSample.reflectionWeight / slabReflectionPdf,
          );
          rayDirection = slabSample.reflectedDirection;
          slabNormal = -slabNormal;
          slabFrontFacing = !slabFrontFacing;
        }
        if (!slabExited) { return accumulatedRadiance; }
      } else if (entering) {
        if (mediumState.depth >= DDGI_OPTICAL_MEDIUM_CAPACITY) {
          return accumulatedRadiance;
        }
        bulkTransmissionAction = 1u;
      } else {
        bulkTransmissionAction = 2u;
      }
    }
    if (!ddgiTransferFinite(prefixTransfer)) {
      return accumulatedRadiance;
    }

    let outgoingContaining = ddgiClassifyContainingMedia(
      currentPos,
      rayDirection,
      currentSourceFeature,
      false,
    );
    if (outgoingContaining.valid == 0u) { return accumulatedRadiance; }
    var nextMediumState = outgoingContaining.state;
    if (bulkTransmissionAction == 0u) {
      if (!ddgiOpticalStateIdentityEqual(
        mediumState, nextMediumState,
      )) { return accumulatedRadiance; }
      for (var i = 0u; i < mediumState.depth; i = i + 1u) {
        nextMediumState.ior[i] = mediumState.ior[i];
        nextMediumState.distance[i] = mediumState.distance[i];
        nextMediumState.transmissionPaid[i] =
          mediumState.transmissionPaid[i];
      }
    } else if (bulkTransmissionAction == 1u) {
      if (
        nextMediumState.depth != mediumState.depth + 1u ||
        !ddgiOpticalStatePrefixEqual(
          mediumState, nextMediumState, mediumState.depth,
        ) ||
        nextMediumState.boundaryId[mediumState.depth] != boundaryId ||
        nextMediumState.representedId[mediumState.depth] != representedId
      ) { return accumulatedRadiance; }
      for (var i = 0u; i < mediumState.depth; i = i + 1u) {
        nextMediumState.ior[i] = mediumState.ior[i];
        nextMediumState.distance[i] = mediumState.distance[i];
        nextMediumState.transmissionPaid[i] =
          mediumState.transmissionPaid[i];
      }
      let top = mediumState.depth;
      nextMediumState.ior[top] = targetIor;
      nextMediumState.distance[top] = 0.0;
      nextMediumState.transmissionPaid[top] = 1u;
    } else {
      if (
        mediumState.depth == 0u ||
        nextMediumState.depth + 1u != mediumState.depth ||
        !ddgiOpticalStatePrefixEqual(
          mediumState, nextMediumState, nextMediumState.depth,
        )
      ) { return accumulatedRadiance; }
      for (var i = 0u; i < nextMediumState.depth; i = i + 1u) {
        nextMediumState.ior[i] = mediumState.ior[i];
        nextMediumState.distance[i] = mediumState.distance[i];
        nextMediumState.transmissionPaid[i] =
          mediumState.transmissionPaid[i];
      }
    }
    mediumState = nextMediumState;

    let nextRay = Ray(currentPos, rayDirection);
    let sourceAware = ddgiTraceFirstHitAlphaMaskTexturedWithOpticalSource(
      nextRay, currentSourceFeature,
    );
    if (sourceAware.valid == 0u) { return accumulatedRadiance; }
    if (!sourceAware.hit.didHit) {
      if (mediumState.depth != 0u) { return accumulatedRadiance; }
      return accumulatedRadiance + ddgiTransferredChannel(
        prefixTransfer, sampleSkyColor(rayDirection), channel,
      );
    }
    let accepted = ddgiAcceptProbeHit(
      nextRay,
      sourceAware.hit,
      0.0,
      currentSourceFeature,
    );
    if (accepted.valid == 0u) { return accumulatedRadiance; }
    if (mediumState.depth > 0u) {
      prefixTransfer = prefixTransfer * ddgiConsumeTopMediumSegment(
        &mediumState, accepted.hit.dist,
      );
      if (!ddgiTransferFinite(prefixTransfer)) {
        return accumulatedRadiance;
      }
    }
    currentHit = accepted.hit;
    currentSourceFeature = accepted.sourceFeature;
    currentPos = nextRay.origin + rayDirection * currentHit.dist;
    let smoothNormal = ddgiSmoothShadingNormalForHit(
      currentHit, currentHit.normal,
    );
    currentShadingNormal = ddgiApplyBumpMapForHit(
      currentHit,
      ddgiApplyNormalMapForHit(currentHit, smoothNormal),
    );
    let nextMaterialId = bvh_materialId[currentHit.indices.w];
    if (nextMaterialId >= DDGI_MAX_MATERIALS) {
      return accumulatedRadiance;
    }
    let nextEntry = materials[nextMaterialId];
    currentMaterial = ddgiSampleProbeHitMaterial(
      currentHit,
      nextEntry.baseColor,
      nextEntry.roughness,
      nextEntry.metalness,
      nextEntry.transmission,
      nextEntry.ior,
      nextEntry.attenuationColor,
      nextEntry.attenuationDistance,
      smoothNormal,
      currentShadingNormal,
      -rayDirection,
    );
    currentSurfaceSource = ddgiEvaluateProbeSurfaceRadiance(
      currentHit,
      currentPos,
      rayDirection,
      nextEntry,
      currentMaterial,
      currentShadingNormal,
      ddgiGlassPcgNext(&rng),
    );
  }
  return accumulatedRadiance;
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

    let containingMedia = ddgiClassifyContainingMedia(
      probeOrigin,
      dir,
      opticalSourceFeatureInvalid(),
      false,
    );
    let ordinaryHit = ddgiTraceFirstHitAlphaMaskTextured(ray);

    var out: ProbeRay;
    out.direction = dir;
    // Reserve a transient classification lane for the blend passes. The
    // classifier overwrites the first record for each active probe; initialize
    // every record so malformed/early classifier exits stay conservative.
    out._pad0 = 0.0;

    if (containingMedia.valid == 0u) {
      out.hitRadiance = vec3f(0.0);
      out.hitDistance = select(
        BVH_INTERSECT_INFINITY, ordinaryHit.dist, ordinaryHit.didHit,
      );
    } else if (!ordinaryHit.didHit) {
      // A miss from a valid closed containing medium is topologically
      // impossible; an invalid containment scan or open shell fails closed.
      out.hitRadiance = select(
        vec3f(0.0),
        sampleSkyColor(dir),
        containingMedia.state.depth == 0u,
      );
      out.hitDistance  = BVH_INTERSECT_INFINITY;
    } else {
      let accepted = ddgiAcceptProbeHit(
        ray,
        ordinaryHit,
        ddgiProbeDistance(DDGI_TRACE_T_MIN_NORMALIZED),
        opticalSourceFeatureInvalid(),
      );
      if (accepted.valid == 0u) {
        out.hitRadiance = vec3f(0.0);
        out.hitDistance = ordinaryHit.dist;
        let resultIdx = probeIdx * RAYS_PER_PROBE + rayIdx;
        if (resultIdx < arrayLength(&rayResults)) {
          rayResults[resultIdx] = out;
        }
        continue;
      }
      let hit = accepted.hit;
      let matId = bvh_materialId[hit.indices.w];
      if (matId >= DDGI_MAX_MATERIALS) {
        out.hitRadiance = vec3f(0.0);
        out.hitDistance = hit.dist;
        let resultIdx = probeIdx * RAYS_PER_PROBE + rayIdx;
        if (resultIdx < arrayLength(&rayResults)) {
          rayResults[resultIdx] = out;
        }
        continue;
      }
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
          mat.ior, mat.attenuationColor, mat.attenuationDistance,
          smoothNormal, probeNormal, -dir,
        );

        let directSeed = frameParams.frameIndex ^
          (probeIdx * 0x9e3779b9u) ^ rayIdx;
        let surfaceSource = ddgiEvaluateProbeSurfaceRadiance(
          hit,
          hitWorldPos,
          dir,
          mat,
          probeMat,
          probeNormal,
          directSeed,
        );
        var radiance = surfaceSource.terminalLo;
        if (
          containingMedia.state.depth > 0u ||
          accepted.hasTransmission != 0u
        ) {
          let transportSeed = directSeed ^ (rayIdx * 0x85ebca6bu);
          // Containment is reconstructed along this ray's actual direction.
          // An opaque hit inside bulk applies the incoming segment and returns;
          // a transmissive hit continues from its exact represented feature.
          radiance = vec3f(
            ddgiTraceGlassChannel(
              hit, hitWorldPos, hit.dist, dir, probeNormal, probeMat,
              surfaceSource, accepted.sourceFeature, containingMedia,
              0u, transportSeed,
            ),
            ddgiTraceGlassChannel(
              hit, hitWorldPos, hit.dist, dir, probeNormal, probeMat,
              surfaceSource, accepted.sourceFeature, containingMedia,
              1u, transportSeed,
            ),
            ddgiTraceGlassChannel(
              hit, hitWorldPos, hit.dist, dir, probeNormal, probeMat,
              surfaceSource, accepted.sourceFeature, containingMedia,
              2u, transportSeed,
            ),
          );
        }

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
