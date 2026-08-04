/**
 * Probe ray-cast compute shader — assembled raw WGSL.
 *
 * Extracted from `_staging/legacy-source/src/rendering/shaders/walkaround/probeRayCast.wgsl.ts`.
 * The original used Three.js TSL `wgslFn()` / `wgsl()` nodes to assemble the shader
 * from composable pieces.  This file is the assembled equivalent: all constituent
 * WGSL strings concatenated in dependency order into a single module, suitable for
 * `device.createShaderModule({ code: PROBE_RAY_CAST_WGSL })`.
 *
 * Composition order:
 *   1. Canonical shared material, binary-BVH, and TLAS traversal modules.
 *   2. RC binding-loader seams, including the packed material/TLAS scene arena.
 *   3. Canonical octahedral and PCG sampling helpers.
 *   4. RC material-atlas, BRDF, and analytic-light evaluation modules.
 *   5. Cascade uniforms, binding declarations, and the workgroup-64 entry point.
 *
 * The original TSL `instanceIndex` built-in (global thread index) becomes
 * `@builtin(global_invocation_id) globalId: vec3u` with `let index = globalId.x;`.
 *
 * The first extraction preserved the TSL function bodies. The production
 * module now composes canonical Vitrum traversal/material/light kernels; the
 * live WGSL and cascadeDispatch binding layout are the executable source of
 * truth.
 *
 * See `../TSL_TO_RAW_MAPPING.md` for the conversion rationale and current ABI.
 */

import {
  BEER_LAMBERT_WGSL,
  BVH_INTERSECT_STACK_DEPTH,
  BVH_INTERSECT_CORE_WGSL,
  MATERIAL_ENTRY_WGSL,
  OPTICAL_WATERTIGHT_TRIANGLE_WGSL,
  TLAS_TRAVERSAL_STACK_DEPTH,
  TLAS_TRAVERSAL_CORE_WGSL,
} from '@vitrum/shared-bvh';
import { OCTAHEDRAL_CORE_WGSL, PCG_HASH_TO_F32_WGSL } from '@vitrum/shared-samplers';
import { RC_ENVIRONMENT_RADIANCE_SCALE_WGSL } from '../environmentRadianceScale.js';
import { RC_SUN_VISIBILITY_WGSL, RC_NEE_POINTSPOT_WGSL } from './rcLightEval.wgsl.js';
import { RC_MATERIAL_ATLAS_WGSL } from './rcMaterialAtlas.wgsl.js';
import { RC_BRDF_WGSL } from './rcBrdf.wgsl.js';
import { RC_OCTAHEDRAL_STRATIFIED_SAMPLING_WGSL } from './octahedralSampling.wgsl.js';

export const PROBE_RAY_CAST_WGSL = /* wgsl */`
${MATERIAL_ENTRY_WGSL}
${BEER_LAMBERT_WGSL}
${BVH_INTERSECT_CORE_WGSL}
${TLAS_TRAVERSAL_CORE_WGSL}
${OPTICAL_WATERTIGHT_TRIANGLE_WGSL}
${RC_ENVIRONMENT_RADIANCE_SCALE_WGSL}

// Portable value-return loader seam for RC's binding names.
fn bvhLoadNode(index: u32) -> BVHNode { return rc_bvh[index]; }
fn bvhLoadIndex(index: u32) -> vec4u { return rc_geom_index[index]; }
fn bvhLoadPosition(index: u32) -> vec4f { return rc_geom_position[index]; }

// rc_scene_arena header: nine (wordOffset, elementCount) pairs followed by
// one reserved pair. Packing these read-only arrays behind one binding keeps
// the complete cast pass at WebGPU's guaranteed eight-storage-buffer limit.
const RC_ARENA_MATERIAL_OFFSET: u32 = 0u;
const RC_ARENA_MATERIAL_COUNT: u32 = 1u;
const RC_ARENA_TRI_MATERIAL_OFFSET: u32 = 2u;
const RC_ARENA_TRI_MATERIAL_COUNT: u32 = 3u;
const RC_ARENA_TLAS_NODE_OFFSET: u32 = 4u;
const RC_ARENA_TLAS_NODE_COUNT: u32 = 5u;
const RC_ARENA_TLAS_INSTANCE_OFFSET: u32 = 6u;
const RC_ARENA_TLAS_INSTANCE_COUNT: u32 = 7u;
const RC_ARENA_TLAS_BLAS_OFFSET: u32 = 8u;
const RC_ARENA_TLAS_BLAS_COUNT: u32 = 9u;
const RC_ARENA_TLAS_W2L_OFFSET: u32 = 10u;
const RC_ARENA_TLAS_W2L_COUNT: u32 = 11u;
const RC_ARENA_TLAS_L2W_OFFSET: u32 = 12u;
const RC_ARENA_TLAS_L2W_COUNT: u32 = 13u;
const RC_ARENA_OPTICAL_TRI_IDENTITY_OFFSET: u32 = 14u;
const RC_ARENA_OPTICAL_TRI_IDENTITY_COUNT: u32 = 15u;
const RC_ARENA_OPTICAL_INSTANCE_BASE_OFFSET: u32 = 16u;
const RC_ARENA_OPTICAL_INSTANCE_BASE_COUNT: u32 = 17u;

fn rcArenaF32(wordIndex: u32) -> f32 {
  return bitcast<f32>(rc_scene_arena[wordIndex]);
}

fn rcArenaVec4(offsetWord: u32, index: u32) -> vec4f {
  let base = offsetWord + index * 4u;
  return vec4f(
    rcArenaF32(base),
    rcArenaF32(base + 1u),
    rcArenaF32(base + 2u),
    rcArenaF32(base + 3u),
  );
}

fn rcLoadMaterial(index: u32) -> MaterialEntry {
  let base = rc_scene_arena[RC_ARENA_MATERIAL_OFFSET] + index * 16u;
  return MaterialEntry(
    vec3f(rcArenaF32(base), rcArenaF32(base + 1u), rcArenaF32(base + 2u)),
    rcArenaF32(base + 3u),
    vec3f(rcArenaF32(base + 4u), rcArenaF32(base + 5u), rcArenaF32(base + 6u)),
    rcArenaF32(base + 7u),
    rcArenaF32(base + 8u),
    rcArenaF32(base + 9u),
    rcArenaF32(base + 10u),
    rcArenaF32(base + 11u),
    vec3f(rcArenaF32(base + 12u), rcArenaF32(base + 13u), rcArenaF32(base + 14u)),
    rc_scene_arena[base + 15u],
  );
}

fn rcLoadTriMaterialId(index: u32) -> u32 {
  return rc_scene_arena[rc_scene_arena[RC_ARENA_TRI_MATERIAL_OFFSET] + index];
}

fn tlasLoadNode(index: u32) -> BVHNode {
  let base = rc_scene_arena[RC_ARENA_TLAS_NODE_OFFSET] + index * 8u;
  return BVHNode(
    array<f32, 3>(rcArenaF32(base), rcArenaF32(base + 1u), rcArenaF32(base + 2u)),
    array<f32, 3>(rcArenaF32(base + 3u), rcArenaF32(base + 4u), rcArenaF32(base + 5u)),
    rc_scene_arena[base + 6u],
    rc_scene_arena[base + 7u],
  );
}
fn tlasNodeCapacity() -> u32 { return rc_scene_arena[RC_ARENA_TLAS_NODE_COUNT]; }
fn tlasLoadInstanceIndex(index: u32) -> u32 {
  return rc_scene_arena[rc_scene_arena[RC_ARENA_TLAS_INSTANCE_OFFSET] + index];
}
fn tlasInstanceIndexCount() -> u32 { return rc_scene_arena[RC_ARENA_TLAS_INSTANCE_COUNT]; }
fn tlasLoadBlasRoot(index: u32) -> u32 {
  return rc_scene_arena[rc_scene_arena[RC_ARENA_TLAS_BLAS_OFFSET] + index];
}
fn tlasBlasRootCount() -> u32 { return rc_scene_arena[RC_ARENA_TLAS_BLAS_COUNT]; }
fn tlasLoadWorldToLocalColumn(index: u32) -> vec4f {
  return rcArenaVec4(rc_scene_arena[RC_ARENA_TLAS_W2L_OFFSET], index);
}
fn tlasWorldToLocalColumnCount() -> u32 { return rc_scene_arena[RC_ARENA_TLAS_W2L_COUNT]; }
fn tlasLoadLocalToWorldColumn(index: u32) -> vec4f {
  return rcArenaVec4(rc_scene_arena[RC_ARENA_TLAS_L2W_OFFSET], index);
}
fn tlasLocalToWorldColumnCount() -> u32 { return rc_scene_arena[RC_ARENA_TLAS_L2W_COUNT]; }

fn rcLoadOpticalTriangleIdentity(index: u32) -> vec2u {
  let base = rc_scene_arena[RC_ARENA_OPTICAL_TRI_IDENTITY_OFFSET] + index * 2u;
  return vec2u(rc_scene_arena[base], rc_scene_arena[base + 1u]);
}
fn rcOpticalTriangleIdentityCount() -> u32 {
  return rc_scene_arena[RC_ARENA_OPTICAL_TRI_IDENTITY_COUNT];
}
fn rcLoadOpticalInstanceBoundaryIdBasePlusOne(index: u32) -> u32 {
  return rc_scene_arena[
    rc_scene_arena[RC_ARENA_OPTICAL_INSTANCE_BASE_OFFSET] + index
  ];
}
fn rcOpticalInstanceBoundaryIdBaseCount() -> u32 {
  return rc_scene_arena[RC_ARENA_OPTICAL_INSTANCE_BASE_COUNT];
}

// safe_normalize — zero-length-guarded normalize. Required by TLAS_TRAVERSAL_WGSL
// (transformDirToWorld / transformDirToLocal call it for the per-instance
// world↔local ray-direction transforms). The other two TLAS consumers — the DDGI
// probe shader (probeUpdateRays.wgsl) and the hybrid scene-traversal shaders
// (sharedPrimitives.wgsl) — each define this same helper alongside their
// TLAS_TRAVERSAL_WGSL include; RC was the lone consumer that included the TLAS
// traversal but never defined safe_normalize, so the assembled rc-probe-ray-cast
// module failed naga compilation (unknown identifier safe_normalize). Mirrors the
// canonical definition in those siblings. (Latent because RC's probe shader had no
// GPU-compile gate -- W8 pinned only the CPU packRCParams + wgslCompose order;
// surfaced by the RC core-BVH converged A/B.)
fn safe_normalize(v: vec3f) -> vec3f {
  let maxComponent = max(abs(v.x), max(abs(v.y), abs(v.z)));
  if (
    !(maxComponent > 0.0) ||
    !(maxComponent <= 3.402823e+38)
  ) {
    return vec3f(0.0, 1.0, 0.0);
  }
  let scaled = v / maxComponent;
  let scaledLen2 = dot(scaled, scaled);
  if (!(scaledLen2 > 0.0) || !(scaledLen2 <= 3.0)) {
    return vec3f(0.0, 1.0, 0.0);
  }
  return scaled * inverseSqrt(scaledLen2);
}

// C2 — merged world BVH vs TLAS+local BLAS (same traversal as ReSTIR / DDGI).
// ─── CascadeUniforms struct ───────────────────────────────────────────────────
// Must match buildCascadeUniformDataInto() layout in cascadeDispatch.ts
// (40-word / 160-byte WGSL layout and host allocation).
//
// E2 — triIntersectEpsilon: Möller–Trumbore coplanarity threshold (f32).
// Replaces the local const TRI_INTERSECT_EPSILON: f32 = 1e-5 that was
// previously hardcoded. The final ABI remains 40 f32/u32 words = 160 bytes.

struct CascadeUniforms {
  probeOriginWorld  : vec3f,
  _pad0             : f32,
  roomSize          : vec3f,
  _pad1             : f32,
  probeCount        : vec3u,
  raysPerProbe      : u32,
  rayGridSize       : u32,
  intervalNear      : f32,
  intervalFar       : f32,
  cascadeIndex      : u32,
  sunDirection      : vec3f,
  sunAngularRadius  : f32,
  sunColor          : vec3f,
  envIntensity      : f32,
  frameSeed         : u32,
  lastCascade       : u32,
  triIntersectEpsilon: f32,  // E2: UBO-plumbed (was local const 1e-5)
  bvhMode           : u32,   // C2: 0 merged, 1 TLAS+local BLAS
  tlasNodeCount     : u32,
  // RC emitter-NEE (2026-06-07): number of rect-area emitter triangles in
  // rc_emitters. 0 ⇒ the NEE loop is skipped and RC's light model stays
  // sun+emissive+env (the prior behaviour, byte-identical). Host packs this
  // at slot 29 (offset 116) in buildCascadeUniformDataInto.
  emitterCount      : u32,
  // Runtime analytic/directional light count in rc_lights.
  // 0 ⇒ analytic-light evaluation is skipped (byte-identical with prior).
  // Host packs this at slot 30 (offset 120) in buildCascadeUniformDataInto.
  lightCount        : u32,
  sunCastShadowDisabled: u32,
  emitterDataWordOffset: u32,
  emitterAliasWordOffset: u32,
  transmittedInterfaceBudget: u32,
  // H6: world direction is transformed by RY(-envRotationY) before lookup.
  // Appended so every previously published field offset stays stable.
  envRotationY      : f32,
  // Main/ReSTIR parity: a bindable placeholder must not suppress scalar sky.
  // These fields consume the final 16 bytes of the established allocation.
  scalarSkyRadiance : vec3f,
  hasDirectionalEnv : u32,
};

// ─── EmitterTri struct ────────────────────────────────────────────────────────
// 80-byte rect-area emitter triangle, layout-identical to the shade/ReSTIR-DI
// EmitterTri (reservoirDi.wgsl.ts) so RC can BIND THE SAME host buffer the main
// pipeline already builds (BvhBufferHost._emitterBuffer) — no second upload, no
// new dummy-buffer (the 32-byte-min bind class). World-space triangle + its
// front-face normal + radiance Le.
struct EmitterTri {
  vA:        vec3f,
  _padA:     f32,
  vB:        vec3f,
  _padB:     f32,
  vC:        vec3f,
  _padC:     f32,
  normal:    vec3f,
  area:      f32,
  Le:        vec3f,
  emitterFlags: f32,
};

const RC_EMITTER_CAST_SHADOW_DISABLED: u32 = 1u;
const RC_EMITTER_TWO_SIDED: u32 = 2u;

fn rcEmitterFlags(emitter: EmitterTri) -> u32 {
  let flags = emitter.emitterFlags;
  if (
    !rcMaterialAtlasFiniteF32(flags) ||
    flags < 0.0 ||
    flags > 3.0 ||
    floor(flags) != flags
  ) {
    return 0u;
  }
  return u32(flags);
}

fn rcEmitterCastShadowDisabled(emitter: EmitterTri) -> bool {
  return (
    rcEmitterFlags(emitter) & RC_EMITTER_CAST_SHADOW_DISABLED
  ) != 0u;
}

fn rcEmitterCosineTowardReceiver(
  emitter: EmitterTri,
  towardReceiver: vec3f,
) -> f32 {
  let signedCosine = dot(emitter.normal, towardReceiver);
  return select(
    max(signedCosine, 0.0),
    abs(signedCosine),
    (rcEmitterFlags(emitter) & RC_EMITTER_TWO_SIDED) != 0u,
  );
}

// ─── MaterialEntry struct ─────────────────────────────────────────────────────
// Canonical 16 × f32 = 64 bytes per entry — declared by
// @vitrum/shared-bvh/wgsl/materialEntry.wgsl.ts and injected at the top of
// this file via the MATERIAL_ENTRY_WGSL template substitution. Pre-W2-C5
// this file declared a flat-struct local copy (colorR/G/B/A, attenColorR/G/B,
// etc.); the canonical struct uses vec3<f32> for color triples and the
// field-rename collapses two drifted layouts (DDGI / RC) into one.
// (ReSTIR did not use this struct — it packs per-tri RGBA8 into bvhIndex.w.)

// ─── Octahedral helpers (canonical from @vitrum/shared-samplers) ─────────────
// Call sites use octDecode(uv * 2.0 - 1.0) to remap from [0,1] to [-1,1].
${OCTAHEDRAL_CORE_WGSL}

// ─── PCG hash utilities (canonical from @vitrum/shared-samplers) ─────────────
${PCG_HASH_TO_F32_WGSL}

// Exact semantic mirror of shared-samplers' stateful-PCG helper. RC composes
// only the stateless hash seam, whose output has the same 2^24-point lattice.
fn represented_bernoulli_probability_f32(probability: f32) -> f32 {
  if (!(probability > 0.0)) { return 0.0; }
  if (probability >= 1.0) { return 1.0; }
  let bucket = clamp(
    floor(probability * 16777216.0 + 0.5),
    1.0,
    16777215.0,
  );
  return bucket / 16777216.0;
}
${RC_OCTAHEDRAL_STRATIFIED_SAMPLING_WGSL}

// ─── Probe-ray helpers ────────────────────────────────────────────────────────
// Verbatim from probeRayHelpers wgslFn in probeRayCast.wgsl.ts.

fn dirToEquirectUV(d: vec3f) -> vec2f {
  let phi   = atan2(d.z, d.x);
  let theta = acos(clamp(d.y, -1.0, 1.0));
  return vec2f(phi / (2.0 * 3.14159265) + 0.5, theta / 3.14159265);
}

// H6 — RY(-rotY)·dir (world → unrotated-map lookup direction), identical
// to walkaround-hybrid's envRotateYNeg / DDGI environment convention.
fn rcEnvRotateYNeg(dir: vec3f, rotY: f32) -> vec3f {
  let c = cos(rotY);
  let s = sin(rotY);
  return vec3f(c * dir.x - s * dir.z, dir.y, s * dir.x + c * dir.z);
}

${RC_SUN_VISIBILITY_WGSL}

fn rcSoftSunDirection(
  sunBase: vec3f,
  angularRadius: f32,
  hitPos: vec3f,
  roomSize: vec3f,
  cascadeIndex: u32,
) -> vec3f {
  let radius = max(angularRadius, 0.0);
  if (radius <= 0.0) {
    return sunBase;
  }
  let roomScale = min(roomSize.x, min(roomSize.y, roomSize.z));
  if (!(roomScale > 0.0)) { return sunBase; }
  let quant = vec3i(floor(hitPos / roomScale * 1024.0));
  let seed =
    (bitcast<u32>(quant.x) * 0x9E3779B9u) ^
    (bitcast<u32>(quant.y) * 0x85EBCA6Bu) ^
    (bitcast<u32>(quant.z) * 0xC2B2AE35u) ^
    (cascadeIndex * 0x27D4EB2Du) ^
    0x52435355u;
  let xi = vec2f(
    pcgHashToF32(seed ^ 0x53474341u),
    pcgHashToF32(seed ^ 0x4f495431u),
  );
  let upRef = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), abs(sunBase.y) < 0.99);
  let tangent = normalize(cross(upRef, sunBase));
  let bitangent = cross(sunBase, tangent);
  let r = radius * sqrt(xi.x);
  let phi = 6.2831853 * xi.y;
  return normalize(sunBase + tangent * (r * cos(phi)) + bitangent * (r * sin(phi)));
}

// ─── RCLight struct ──────────────────────────────────────────────────────────
// 64-byte analytic punctual/directional light for RC probe rays.
// Layout mirrors DDGI's DDGILight (probeUpdateRays.wgsl.ts) exactly so the
// same DDGIProbeLights-style host buffer can be shared into RC via binding 15.
//
// Host packs runtime-sized entries followed by a represented-PMF alias table.
// lightCount == 0 keeps analytic evaluation a no-op.
//
// Struct layout (4 × u32/f32 = 16 floats = 64 bytes):
//   [0]       kind:       low bits 0 = skipped, 1 = point, 2 = spot;
//                         high bit set => source emitter castShadow:false
//   [1]       distance:   finite cutoff distance; 0 = no cutoff
//   [2]       decay:      falloff exponent; 0 = no falloff, 2 = inverse-square
//   [3]       _pad
//   [4..6]    position:   vec3f world-space position
//   [7]       intensity:  scalar (lux / cd equivalent — 1/r² applied below)
//   [8..10]   direction:  vec3f spot-cone forward beam axis (zero = point → no cone)
//   [11]      innerCone:  cosine of inner-cone angle (smooth = 1 for point)
//   [12..14]  color:      vec3f RGB
//   [15]      outerCone:  cosine of outer-cone angle (hard edge = 0 for point)

const RC_LIGHT_POINT: u32 = 1u;
const RC_LIGHT_SPOT:  u32 = 2u;
const RC_LIGHT_DIRECTIONAL: u32 = 3u;
const RC_LIGHT_KIND_MASK: u32 = 0x7fffffffu;
const RC_LIGHT_CAST_SHADOW_DISABLED: u32 = 0x80000000u;
const RC_LIGHTS_ABI_MAGIC: u32 = 0x31544352u;

struct RCLight {
  kind:      u32,
  distance:  f32,
  decay:     f32,
  _pad2:     f32,
  position:  vec3f,
  intensity: f32,
  direction: vec3f,
  innerCone: f32,
  color:     vec3f,
  outerCone: f32,
};

// ─── Bind group declarations ──────────────────────────────────────────────────
// Cast pass: @group(0) bindings 0-20.

@group(0) @binding(0) var<storage, read>       rc_bvh:                   array<BVHNode>;
@group(0) @binding(1) var<storage, read>       rc_geom_index:            array<vec4u>;
@group(0) @binding(2) var<storage, read>       rc_geom_position:         array<vec4f>;
@group(0) @binding(3) var<storage, read>       rc_scene_arena:           array<u32>;
@group(0) @binding(5) var<storage, read_write> rc_cascadeOut:            array<vec4f>;
@group(0) @binding(6) var                      rc_envMap:                texture_2d<f32>;
@group(0) @binding(7) var                      rc_envSampler:            sampler;
@group(0) @binding(8) var<uniform>             rc_u:                     CascadeUniforms;
@group(0) @binding(14) var<storage, read>      rc_emitters:              array<u32>;
// Runtime punctual/directional lights for probe-ray direct lighting.
// lightCount == 0 ⇒ loop is a no-op. Host binds the same DDGIProbeLights-style
// buffer (packRCLights) or a header-only zero placeholder when no lights exist.
@group(0) @binding(15) var<storage, read>      rc_lights:                array<u32>;
// RC material-backed emitter NEE (2026-06-16): optional material atlas views
// forwarded from the main pipeline. Placeholder metadata advertises zero
// logical layers, so helpers fall back to scalar EmitterTri.Le when omitted.
@group(0) @binding(16) var                      rc_materialTextureAtlas: texture_2d_array<u32>;
@group(0) @binding(17) var                      rc_materialMapMeta:      texture_2d<f32>;
@group(0) @binding(18) var<storage, read>       rc_geom_normal:           array<vec4f>;
@group(0) @binding(19) var                      rc_geom_tangent:          texture_2d<f32>;
@group(0) @binding(20) var                      rc_geom_vertex_color:     texture_2d<f32>;

// Match walkaround-hybrid envRadiance: directional payloads sample the map,
// while a bindable black placeholder selects the authored scalar sky.
fn rcEnvironmentRadiance(dir: vec3f) -> vec3f {
  if (rc_u.hasDirectionalEnv == 0u) {
    return rcScaleEnvironmentRadiance(rc_u.scalarSkyRadiance, 1.0);
  }
  let texel = textureSampleLevel(
    rc_envMap,
    rc_envSampler,
    dirToEquirectUV(rcEnvRotateYNeg(dir, rc_u.envRotationY)),
    0.0,
  );
  return rcScaleEnvironmentRadiance(texel.rgb, rc_u.envIntensity);
}

struct RCAliasDraw {
  index: u32,
  pmf: f32,
};

fn rcPcgHashU32(seed: u32) -> u32 {
  var state = seed * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

fn rcAliasColumn(seed: u32, count: u32) -> u32 {
  let threshold = ((0xffffffffu % count) + 1u) % count;
  var word = rcPcgHashU32(seed);
  loop {
    if (word >= threshold) { return word % count; }
    word = rcPcgHashU32(word ^ 0x27d4eb2du);
  }
  return 0u;
}

fn rcLoadEmitter(index: u32) -> EmitterTri {
  let base = rc_u.emitterDataWordOffset + index * 20u;
  var e: EmitterTri;
  e.vA = bitcast<vec3f>(vec3u(rc_emitters[base], rc_emitters[base + 1u], rc_emitters[base + 2u]));
  e._padA = bitcast<f32>(rc_emitters[base + 3u]);
  e.vB = bitcast<vec3f>(vec3u(rc_emitters[base + 4u], rc_emitters[base + 5u], rc_emitters[base + 6u]));
  e._padB = bitcast<f32>(rc_emitters[base + 7u]);
  e.vC = bitcast<vec3f>(vec3u(rc_emitters[base + 8u], rc_emitters[base + 9u], rc_emitters[base + 10u]));
  e._padC = bitcast<f32>(rc_emitters[base + 11u]);
  e.normal = bitcast<vec3f>(vec3u(rc_emitters[base + 12u], rc_emitters[base + 13u], rc_emitters[base + 14u]));
  e.area = bitcast<f32>(rc_emitters[base + 15u]);
  e.Le = bitcast<vec3f>(vec3u(rc_emitters[base + 16u], rc_emitters[base + 17u], rc_emitters[base + 18u]));
  e.emitterFlags = bitcast<f32>(rc_emitters[base + 19u]);
  return e;
}

fn rcEmitterAliasDraw(count: u32, seed: u32) -> RCAliasDraw {
  let column = rcAliasColumn(seed, count);
  let base = rc_u.emitterAliasWordOffset + column * 4u;
  let q = bitcast<f32>(rc_emitters[base]);
  let aliasEntry = rc_emitters[base + 1u];
  let selected = select(aliasEntry, column, pcgHashToF32(seed ^ 0x9e3779b9u) < q);
  var draw: RCAliasDraw;
  draw.index = selected;
  draw.pmf = bitcast<f32>(rc_emitters[rc_u.emitterAliasWordOffset + selected * 4u + 2u]);
  return draw;
}

fn rcLoadLight(index: u32) -> RCLight {
  let base = rc_lights[1u] + index * 16u;
  var light: RCLight;
  light.kind = rc_lights[base];
  light.distance = bitcast<f32>(rc_lights[base + 1u]);
  light.decay = bitcast<f32>(rc_lights[base + 2u]);
  light._pad2 = bitcast<f32>(rc_lights[base + 3u]);
  light.position = bitcast<vec3f>(vec3u(rc_lights[base + 4u], rc_lights[base + 5u], rc_lights[base + 6u]));
  light.intensity = bitcast<f32>(rc_lights[base + 7u]);
  light.direction = bitcast<vec3f>(vec3u(rc_lights[base + 8u], rc_lights[base + 9u], rc_lights[base + 10u]));
  light.innerCone = bitcast<f32>(rc_lights[base + 11u]);
  light.color = bitcast<vec3f>(vec3u(rc_lights[base + 12u], rc_lights[base + 13u], rc_lights[base + 14u]));
  light.outerCone = bitcast<f32>(rc_lights[base + 15u]);
  return light;
}

fn rcLightAliasDraw(count: u32, seed: u32) -> RCAliasDraw {
  let column = rcAliasColumn(seed, count);
  let aliasOffset = rc_lights[2u];
  let base = aliasOffset + column * 4u;
  let q = bitcast<f32>(rc_lights[base]);
  let aliasEntry = rc_lights[base + 1u];
  let selected = select(aliasEntry, column, pcgHashToF32(seed ^ 0x85ebca6bu) < q);
  var draw: RCAliasDraw;
  draw.index = selected;
  draw.pmf = bitcast<f32>(rc_lights[aliasOffset + selected * 4u + 2u]);
  return draw;
}

${RC_MATERIAL_ATLAS_WGSL}

${RC_BRDF_WGSL}

struct RCAlphaCoverage {
  mode: u32,
  coverage: f32,
  cutoff: f32,
};

fn rcMaterialAlphaCoverageForHit(hit: IntersectionResult) -> RCAlphaCoverage {
  var out: RCAlphaCoverage;
  out.mode = 0u;
  out.coverage = 1.0;
  out.cutoff = 0.0;

  if (!rcMaterialMetaAvailable(
    hit.indices.w,
    RC_MATERIAL_MAP_ALPHA_COVERAGE_TEXEL_OFFSET,
  )) {
    return out;
  }

  let coverageMeta = textureLoad(
    rc_materialMapMeta,
    rcMaterialMetaCoord(hit.indices.w, RC_MATERIAL_MAP_ALPHA_COVERAGE_TEXEL_OFFSET),
    0,
  );
  if (
    !rcMaterialAtlasFiniteF32(coverageMeta.x) ||
    coverageMeta.x < 0.0 ||
    coverageMeta.x > 2.0 ||
    floor(coverageMeta.x) != coverageMeta.x
  ) {
    return out;
  }
  out.mode = u32(coverageMeta.x);
  if (out.mode == 0u) {
    return out;
  }

  let uvs = rcHitMaterialUvs(hit);
  if (uvs.valid == 0u) {
    return out;
  }

  let baseColorTexel = rcSampleMaterialAtlasRaw(hit.indices.w, RC_MATERIAL_MAP_SLOT_BASE_COLOR, uvs.uv0, uvs.uv1);
  let baseColorAlpha = select(
    1.0,
    clamp(baseColorTexel.value.a, 0.0, 1.0),
    baseColorTexel.valid != 0u,
  );
  let alphaTexel = rcSampleMaterialAtlasRaw(hit.indices.w, RC_MATERIAL_MAP_SLOT_ALPHA, uvs.uv0, uvs.uv1);
  let alphaMapCoverage = select(
    1.0,
    clamp(alphaTexel.value.r, 0.0, 1.0),
    alphaTexel.valid != 0u,
  );
  let vertexColorAlpha = rcSampleVertexColorForHit(hit).a;
  let opacity = clamp(coverageMeta.y, 0.0, 1.0);
  out.cutoff = clamp(coverageMeta.z, 0.0, 1.0);
  out.coverage = clamp(opacity * vertexColorAlpha * baseColorAlpha * alphaMapCoverage, 0.0, 1.0);
  return out;
}

fn rcMaterialSideAdmittedForHit(hit: IntersectionResult) -> bool {
  if (hit.side >= 0.0) { return true; }
  let matId = rcLoadTriMaterialId(hit.indices.w);
  let mat = rcLoadMaterial(matId);
  return (mat.flags & MATERIAL_FLAG_DOUBLE_SIDED) != 0u ||
    (mat.flags & MATERIAL_FLAG_IS_GLASS) != 0u ||
    mat.transmission > 0.0;
}

fn rcAlphaShadowTransmittanceForHit(hit: IntersectionResult) -> f32 {
  if (!rcMaterialSideAdmittedForHit(hit)) {
    return 1.0;
  }
  let alpha = rcMaterialAlphaCoverageForHit(hit);
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

fn rcAlphaBlendCoverageHash(hit: IntersectionResult, ray: Ray, layer: u32) -> f32 {
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
    (rc_u.frameSeed * 0x85ebca6bu);
  seed = rcPcgHashU32(seed);
  return f32(seed >> 8u) / 16777216.0;
}

fn rcMaterialAlphaDiscardedForProbeHit(hit: IntersectionResult, ray: Ray, layer: u32) -> bool {
  if (!rcMaterialSideAdmittedForHit(hit)) {
    return true;
  }
  let alpha = rcMaterialAlphaCoverageForHit(hit);
  if (alpha.mode == 0u) { return false; }
  if (alpha.mode == 1u) { return alpha.coverage < alpha.cutoff; }
  if (alpha.mode == 2u) {
    let representedCoverage = represented_bernoulli_probability_f32(alpha.coverage);
    return representedCoverage < 1.0 &&
      rcAlphaBlendCoverageHash(hit, ray, layer) >= representedCoverage;
  }
  return alpha.coverage <= 0.0;
}

fn rcWorldSurfaceBudget() -> u32 {
  let triangleCount = arrayLength(&rc_geom_index);
  if (triangleCount == 0u) { return 1u; }
  var instanceCount = 1u;
  if (rc_u.bvhMode == 1u && rc_u.tlasNodeCount > 0u) {
    instanceCount = max(tlasBlasRootCount(), 1u);
  }
  if (triangleCount > 0xfffffffeu / instanceCount) {
    return 0xffffffffu;
  }
  return triangleCount * instanceCount + 1u;
}

const RC_CONTAINMENT_COVERAGE_HOLE: u32 = 0u;
const RC_CONTAINMENT_COVERAGE_SOLID: u32 = 1u;
const RC_CONTAINMENT_COVERAGE_FRACTIONAL: u32 = 2u;
const RC_MAX_FINITE_F32: f32 = 3.402823466e38;

fn rcContainmentFiniteVec3(value: vec3f) -> bool {
  return all(value == value) &&
    all(abs(value) <= vec3f(RC_MAX_FINITE_F32));
}

// Deterministic topology classification. A mask texel is either a boundary or
// a hole; fractional blend coverage has no unique closed interior and fails
// containment. Production hosts reject non-opaque bulk boundaries up front,
// while this classification keeps the raw-buffer entry point fail-closed.
fn rcContainmentCoverageStatus(hit: IntersectionResult) -> u32 {
  if (!rcMaterialSideAdmittedForHit(hit)) {
    return RC_CONTAINMENT_COVERAGE_HOLE;
  }
  let alpha = rcMaterialAlphaCoverageForHit(hit);
  if (alpha.mode == 0u) {
    return RC_CONTAINMENT_COVERAGE_SOLID;
  }
  if (alpha.mode == 1u) {
    return select(
      RC_CONTAINMENT_COVERAGE_HOLE,
      RC_CONTAINMENT_COVERAGE_SOLID,
      alpha.coverage >= alpha.cutoff,
    );
  }
  if (alpha.coverage <= 0.0) {
    return RC_CONTAINMENT_COVERAGE_HOLE;
  }
  if (alpha.coverage >= 1.0) {
    return RC_CONTAINMENT_COVERAGE_SOLID;
  }
  return RC_CONTAINMENT_COVERAGE_FRACTIONAL;
}

// Build the finite segment from a point proven outside the active world root
// to targetOrigin. Walking the same direction as the transport ray preserves
// the incident-side state for origins on a boundary. The caller replays exact
// LIFO entries/exits directly into its live medium stack, so the number of
// disjoint shells encountered on the segment consumes no fixed bookkeeping.
fn rcMaterialIsOptical(mat: MaterialEntry) -> bool {
  return (mat.flags & MATERIAL_FLAG_IS_GLASS) != 0u || mat.transmission > 0.0;
}

// --------------------------------------------------------------------------
// Exact represented-surface traversal
// --------------------------------------------------------------------------
// Ordinary Moller traversal remains available to the non-optical light helpers,
// but every probe suffix and containment query uses this inclusive watertight
// path. The ray origin is fixed and continuation advances only exclusiveMinT or
// an exact source-feature exclusion; no geometric epsilon step may skip a
// one-ULP nested boundary.

struct RCExactSceneTriangle {
  valid: u32,
  indices: vec3u,
  a: vec3f,
  b: vec3f,
  c: vec3f,
  uvA: vec2f,
  uvB: vec2f,
  uvC: vec2f,
  materialWord: u32,
};

struct RCExactSurfaceEvent {
  status: u32,
  t: f32,
  optical: u32,
  encodedBoundaryId: u32,
  representedPrimitiveInstanceId: u32,
  zeroEdgeMask: u32,
  point: vec3f,
  hit: IntersectionResult,
};

struct RCExactSurfaceAccumulator {
  hasCandidate: u32,
  invalidInput: u32,
  invalidTie: u32,
  t: f32,
  optical: u32,
  encodedBoundaryId: u32,
  representedPrimitiveInstanceId: u32,
  frontCount: u32,
  backCount: u32,
  selectedInstance: u32,
  selectedTriangle: u32,
};

fn rcExactSurfaceEventEmpty() -> RCExactSurfaceEvent {
  var out: RCExactSurfaceEvent;
  out.status = OPTICAL_BOUNDARY_EVENT_NONE;
  out.t = RC_MAX_FINITE_F32;
  out.optical = 0u;
  out.encodedBoundaryId = 0u;
  out.representedPrimitiveInstanceId = 0u;
  out.zeroEdgeMask = 0u;
  out.point = vec3f(0.0);
  out.hit = tlasEmptyFirstHit();
  return out;
}

fn rcExactSurfaceAccumulatorInit() -> RCExactSurfaceAccumulator {
  var out: RCExactSurfaceAccumulator;
  out.hasCandidate = 0u;
  out.invalidInput = 0u;
  out.invalidTie = 0u;
  out.t = RC_MAX_FINITE_F32;
  out.optical = 0u;
  out.encodedBoundaryId = 0u;
  out.representedPrimitiveInstanceId = 0u;
  out.frontCount = 0u;
  out.backCount = 0u;
  out.selectedInstance = 0xffffffffu;
  out.selectedTriangle = 0xffffffffu;
  return out;
}

fn rcExactUseTlas() -> bool {
  // Traversal mode is an ownership contract, not a heuristic. In particular,
  // an empty TLAS must remain empty; falling through to direct BLAS root zero
  // would resurrect stale geometry after the last live placement disappears.
  return rc_u.bvhMode == 1u;
}

fn rcExactEncodedBoundaryId(
  useTlas: bool,
  triangleIndex: u32,
  instanceIndex: u32,
) -> u32 {
  if (triangleIndex >= rcOpticalTriangleIdentityCount()) { return 0u; }
  let componentPlusOne = rcLoadOpticalTriangleIdentity(triangleIndex).x;
  if (componentPlusOne == 0u) { return 0u; }
  let baseIndex = select(0u, instanceIndex, useTlas);
  if (baseIndex >= rcOpticalInstanceBoundaryIdBaseCount()) { return 0u; }
  let basePlusOne = rcLoadOpticalInstanceBoundaryIdBasePlusOne(baseIndex);
  if (basePlusOne == 0u) { return 0u; }
  let componentOffset = componentPlusOne - 1u;
  if (componentOffset > 0xffffffffu - basePlusOne) { return 0u; }
  return basePlusOne + componentOffset;
}

fn rcExactRepresentedPrimitiveInstanceId(
  useTlas: bool,
  triangleIndex: u32,
  instanceIndex: u32,
) -> u32 {
  if (triangleIndex >= rcOpticalTriangleIdentityCount()) { return 0u; }
  if (useTlas) {
    if (
      instanceIndex == 0xffffffffu ||
      instanceIndex >= rcOpticalInstanceBoundaryIdBaseCount()
    ) { return 0u; }
    return instanceIndex + 1u;
  }
  return rcLoadOpticalTriangleIdentity(triangleIndex).y;
}

fn rcLoadExactWorldTriangle(
  useTlas: bool,
  triangleIndex: u32,
  instanceIndex: u32,
) -> RCExactSceneTriangle {
  var triangle: RCExactSceneTriangle;
  triangle.valid = 0u;
  if (
    triangleIndex >= arrayLength(&rc_geom_index) ||
    triangleIndex >= rcOpticalTriangleIdentityCount()
  ) { return triangle; }
  let indexEntry = bvhLoadIndex(triangleIndex);
  if (any(indexEntry.xyz >= vec3u(arrayLength(&rc_geom_position)))) {
    return triangle;
  }
  let pa = bvhLoadPosition(indexEntry.x);
  let pb = bvhLoadPosition(indexEntry.y);
  let pc = bvhLoadPosition(indexEntry.z);
  triangle.indices = indexEntry.xyz;
  triangle.a = pa.xyz;
  triangle.b = pb.xyz;
  triangle.c = pc.xyz;
  triangle.uvA = unpack2x16float(bitcast<u32>(pa.w));
  triangle.uvB = unpack2x16float(bitcast<u32>(pb.w));
  triangle.uvC = unpack2x16float(bitcast<u32>(pc.w));
  triangle.materialWord = indexEntry.w;
  if (useTlas) {
    if (instanceIndex > 0x3fffffffu) { return triangle; }
    let matrixBase = instanceIndex * 4u;
    if (matrixBase + 3u >= tlasLocalToWorldColumnCount()) {
      return triangle;
    }
    let c0 = tlasLoadLocalToWorldColumn(matrixBase);
    let c1 = tlasLoadLocalToWorldColumn(matrixBase + 1u);
    let c2 = tlasLoadLocalToWorldColumn(matrixBase + 2u);
    let c3 = tlasLoadLocalToWorldColumn(matrixBase + 3u);
    triangle.a = tlasTransformPointCols(c0, c1, c2, c3, triangle.a);
    triangle.b = tlasTransformPointCols(c0, c1, c2, c3, triangle.b);
    triangle.c = tlasTransformPointCols(c0, c1, c2, c3, triangle.c);
  }
  if (
    !rcContainmentFiniteVec3(triangle.a) ||
    !rcContainmentFiniteVec3(triangle.b) ||
    !rcContainmentFiniteVec3(triangle.c)
  ) { return triangle; }
  triangle.valid = 1u;
  return triangle;
}

fn rcExactCandidateComesFirst(
  instanceIndex: u32,
  triangleIndex: u32,
  accumulator: RCExactSurfaceAccumulator,
) -> bool {
  return instanceIndex < accumulator.selectedInstance ||
    (
      instanceIndex == accumulator.selectedInstance &&
      triangleIndex < accumulator.selectedTriangle
    );
}

fn rcStoreExactSelectedCandidate(
  selected: ptr<function, RCExactSurfaceEvent>,
  exact: OpticalWatertightHit,
  triangle: RCExactSceneTriangle,
  triangleIndex: u32,
  instanceIndex: u32,
  optical: bool,
  encodedBoundaryId: u32,
  representedId: u32,
) {
  (*selected).t = exact.t;
  (*selected).optical = select(0u, 1u, optical);
  (*selected).encodedBoundaryId = encodedBoundaryId;
  (*selected).representedPrimitiveInstanceId = representedId;
  (*selected).zeroEdgeMask = exact.zeroEdgeMask;
  (*selected).point = opticalCanonicalHitPoint(
    exact, triangle.a, triangle.b, triangle.c,
  );
  (*selected).hit.didHit = true;
  (*selected).hit.indices = vec4u(triangle.indices, triangleIndex);
  (*selected).hit.normal = exact.normal;
  (*selected).hit.barycoord = exact.bary;
  (*selected).hit.side = exact.side;
  (*selected).hit.dist = exact.t;
  (*selected).hit.matColorPacked = triangle.materialWord;
  (*selected).hit.uv = exact.bary.x * triangle.uvA +
    exact.bary.y * triangle.uvB + exact.bary.z * triangle.uvC;
  (*selected).hit.instanceIndex = instanceIndex;
}

fn rcAccumulateExactSurfaceTriangle(
  useTlas: bool,
  triangleIndex: u32,
  instanceIndex: u32,
  ray: Ray,
  exclusiveMinT: f32,
  sourceFeature: OpticalSourceFeature,
  bulkOnly: bool,
  stochasticAlpha: bool,
  accumulator: ptr<function, RCExactSurfaceAccumulator>,
  selected: ptr<function, RCExactSurfaceEvent>,
) {
  let triangle = rcLoadExactWorldTriangle(
    useTlas, triangleIndex, instanceIndex,
  );
  if (triangle.valid == 0u) {
    (*accumulator).invalidInput = 1u;
    return;
  }
  let encodedBoundaryId = rcExactEncodedBoundaryId(
    useTlas, triangleIndex, instanceIndex,
  );
  let representedId = rcExactRepresentedPrimitiveInstanceId(
    useTlas, triangleIndex, instanceIndex,
  );
  let matId = rcLoadTriMaterialId(triangleIndex);
  let mat = rcLoadMaterial(matId);
  let optical = rcMaterialIsOptical(mat);
  // Packed analyzed topology is authoritative. Authored thickness is a finite
  // distance cap and may legitimately be zero; it cannot classify a boundary
  // as thin versus closed bulk.
  let bulk = optical && encodedBoundaryId != 0u;
  if (bulkOnly && !bulk) { return; }
  if (optical && (representedId == 0u || (bulk && encodedBoundaryId == 0u))) {
    (*accumulator).invalidInput = 1u;
    return;
  }
  if (bulkOnly && encodedBoundaryId == 0u) {
    (*accumulator).invalidInput = 1u;
    return;
  }
  if (opticalSourceFeatureSuppressesTriangle(
    sourceFeature,
    encodedBoundaryId,
    representedId,
    triangleIndex,
    triangle.a,
    triangle.b,
    triangle.c,
  )) { return; }

  let exact = opticalWatertightTriangleIntersect(
    ray.origin,
    ray.direction,
    triangle.a,
    triangle.b,
    triangle.c,
    exclusiveMinT,
  );
  if (!exact.hit) { return; }

  var candidateHit = tlasEmptyFirstHit();
  candidateHit.didHit = true;
  candidateHit.indices = vec4u(triangle.indices, triangleIndex);
  candidateHit.normal = exact.normal;
  candidateHit.barycoord = exact.bary;
  candidateHit.side = exact.side;
  candidateHit.dist = exact.t;
  candidateHit.matColorPacked = triangle.materialWord;
  candidateHit.uv = exact.bary.x * triangle.uvA +
    exact.bary.y * triangle.uvB + exact.bary.z * triangle.uvC;
  candidateHit.instanceIndex = instanceIndex;
  if (
    !bulkOnly && stochasticAlpha &&
    rcMaterialAlphaDiscardedForProbeHit(
      candidateHit,
      ray,
      triangleIndex ^ (instanceIndex * 0x9e3779b9u),
    )
  ) { return; }
  if (bulkOnly) {
    let coverage = rcContainmentCoverageStatus(candidateHit);
    if (coverage == RC_CONTAINMENT_COVERAGE_FRACTIONAL) {
      (*accumulator).invalidInput = 1u;
      return;
    }
    if (coverage == RC_CONTAINMENT_COVERAGE_HOLE) { return; }
  }

  let nearer = (*accumulator).hasCandidate == 0u ||
    exact.t < (*accumulator).t;
  if (nearer) {
    (*accumulator).hasCandidate = 1u;
    (*accumulator).invalidTie = 0u;
    (*accumulator).t = exact.t;
    (*accumulator).optical = select(0u, 1u, optical);
    (*accumulator).encodedBoundaryId = encodedBoundaryId;
    (*accumulator).representedPrimitiveInstanceId = representedId;
    (*accumulator).frontCount = select(0u, 1u, optical && exact.side > 0.0);
    (*accumulator).backCount = select(0u, 1u, optical && exact.side < 0.0);
    (*accumulator).selectedInstance = instanceIndex;
    (*accumulator).selectedTriangle = triangleIndex;
    rcStoreExactSelectedCandidate(
      selected, exact, triangle, triangleIndex, instanceIndex,
      optical, encodedBoundaryId, representedId,
    );
    return;
  }
  if (exact.t != (*accumulator).t) { return; }

  if ((*accumulator).optical != 0u || optical) {
    if (
      (*accumulator).optical == 0u || !optical ||
      (*accumulator).encodedBoundaryId != encodedBoundaryId ||
      (*accumulator).representedPrimitiveInstanceId != representedId
    ) {
      // An optical surface coincident with an opaque surface, or with a
      // distinct represented optical range, has no unique transport ordering.
      (*accumulator).invalidTie = 1u;
      return;
    }
    (*accumulator).frontCount += select(0u, 1u, exact.side > 0.0);
    (*accumulator).backCount += select(0u, 1u, exact.side < 0.0);
  }
  if (rcExactCandidateComesFirst(
    instanceIndex, triangleIndex, *accumulator,
  )) {
    (*accumulator).selectedInstance = instanceIndex;
    (*accumulator).selectedTriangle = triangleIndex;
    rcStoreExactSelectedCandidate(
      selected, exact, triangle, triangleIndex, instanceIndex,
      optical, encodedBoundaryId, representedId,
    );
  }
}

fn rcTraverseExactBlasCandidates(
  useTlas: bool,
  instanceIndex: u32,
  rootNode: u32,
  ray: Ray,
  exclusiveMinT: f32,
  sourceFeature: OpticalSourceFeature,
  bulkOnly: bool,
  stochasticAlpha: bool,
  accumulator: ptr<function, RCExactSurfaceAccumulator>,
  selected: ptr<function, RCExactSurfaceEvent>,
) {
  var traversalRay = ray;
  if (useTlas) {
    if (instanceIndex > 0x3fffffffu) {
      (*accumulator).invalidInput = 1u;
      return;
    }
    let matrixBase = instanceIndex * 4u;
    if (matrixBase + 3u >= tlasWorldToLocalColumnCount()) {
      (*accumulator).invalidInput = 1u;
      return;
    }
    let c0 = tlasLoadWorldToLocalColumn(matrixBase);
    let c1 = tlasLoadWorldToLocalColumn(matrixBase + 1u);
    let c2 = tlasLoadWorldToLocalColumn(matrixBase + 2u);
    let c3 = tlasLoadWorldToLocalColumn(matrixBase + 3u);
    traversalRay.origin = tlasTransformPointCols(c0, c1, c2, c3, ray.origin);
    traversalRay.direction = tlasTransformDirectionCols(c0, c1, c2, ray.direction);
  }
  let invDirection = safeInvDir(traversalRay.direction);
  var stack: array<u32, ${BVH_INTERSECT_STACK_DEPTH}>;
  var stackPointer = 1u;
  stack[0] = rootNode;
  while (stackPointer > 0u) {
    stackPointer -= 1u;
    let nodeIndex = stack[stackPointer];
    if (nodeIndex >= arrayLength(&rc_bvh)) {
      (*accumulator).invalidInput = 1u;
      return;
    }
    let node = bvhLoadNode(nodeIndex);
    let boundsMin = vec3f(
      node.boundsMin[0], node.boundsMin[1], node.boundsMin[2],
    );
    let boundsMax = vec3f(
      node.boundsMax[0], node.boundsMax[1], node.boundsMax[2],
    );
    let t0 = (boundsMin - traversalRay.origin) * invDirection;
    let t1 = (boundsMax - traversalRay.origin) * invDirection;
    let tNear = max(
      max(min(t0.x, t1.x), min(t0.y, t1.y)), min(t0.z, t1.z),
    );
    let tFar = min(
      min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z),
    );
    if (tNear > tFar || tFar < 0.0) { continue; }

    let splitOrCount = node.splitAxisOrTriCount;
    if ((splitOrCount & BVH_LEAFNODE_FLAG) == BVH_LEAFNODE_FLAG) {
      let triangleCount = splitOrCount & 0x0000ffffu;
      let triangleOffset = node.rightChildOrTriOffset;
      for (var triangle = 0u; triangle < triangleCount; triangle += 1u) {
        rcAccumulateExactSurfaceTriangle(
          useTlas,
          triangleOffset + triangle,
          instanceIndex,
          ray,
          exclusiveMinT,
          sourceFeature,
          bulkOnly,
          stochasticAlpha,
          accumulator,
          selected,
        );
      }
    } else {
      if (stackPointer + 2u > ${BVH_INTERSECT_STACK_DEPTH}u) {
        (*accumulator).invalidInput = 1u;
        return;
      }
      stack[stackPointer] = nodeIndex + node.rightChildOrTriOffset;
      stackPointer += 1u;
      stack[stackPointer] = nodeIndex + 1u;
      stackPointer += 1u;
    }
  }
}

fn rcTraverseAllExactTlasInstances(
  ray: Ray,
  exclusiveMinT: f32,
  sourceFeature: OpticalSourceFeature,
  bulkOnly: bool,
  stochasticAlpha: bool,
  accumulator: ptr<function, RCExactSurfaceAccumulator>,
  selected: ptr<function, RCExactSurfaceEvent>,
) {
  for (
    var permutationIndex = 0u;
    permutationIndex < tlasInstanceIndexCount();
    permutationIndex += 1u
  ) {
    let instanceIndex = tlasLoadInstanceIndex(permutationIndex);
    if (instanceIndex >= tlasBlasRootCount()) {
      (*accumulator).invalidInput = 1u;
      return;
    }
    rcTraverseExactBlasCandidates(
      true,
      instanceIndex,
      tlasLoadBlasRoot(instanceIndex),
      ray,
      exclusiveMinT,
      sourceFeature,
      bulkOnly,
      stochasticAlpha,
      accumulator,
      selected,
    );
  }
}

fn rcTraceExactSurfaceEvent(
  ray: Ray,
  exclusiveMinT: f32,
  sourceFeature: OpticalSourceFeature,
  bulkOnly: bool,
  stochasticAlpha: bool,
) -> RCExactSurfaceEvent {
  var accumulator = rcExactSurfaceAccumulatorInit();
  var selected = rcExactSurfaceEventEmpty();
  let useTlas = rcExactUseTlas();
  if (useTlas) {
    if (
      rc_u.tlasNodeCount == 0u || tlasNodeCapacity() == 0u ||
      tlasInstanceIndexCount() == 0u || tlasBlasRootCount() == 0u
    ) {
      // Empty TLAS means empty scene. Never reinterpret the BLAS arena as a
      // merged/direct scene merely because every placement was removed.
      return selected;
    }
    var stack: array<u32, ${TLAS_TRAVERSAL_STACK_DEPTH}>;
    var stackPointer = 1u;
    stack[0] = 0u;
    var overflowed = false;
    while (stackPointer > 0u) {
      stackPointer -= 1u;
      let nodeIndex = stack[stackPointer];
      if (nodeIndex >= min(rc_u.tlasNodeCount, tlasNodeCapacity())) {
        accumulator.invalidInput = 1u;
        break;
      }
      let node = tlasLoadNode(nodeIndex);
      let boundsMin = vec3f(
        node.boundsMin[0], node.boundsMin[1], node.boundsMin[2],
      );
      let boundsMax = vec3f(
        node.boundsMax[0], node.boundsMax[1], node.boundsMax[2],
      );
      if (!tlasIntersectAabb(
        ray,
        boundsMin,
        boundsMax,
        max(exclusiveMinT, 0.0),
        accumulator.t,
      )) { continue; }
      let splitOrCount = node.splitAxisOrTriCount;
      if ((splitOrCount & BVH_LEAFNODE_FLAG) == BVH_LEAFNODE_FLAG) {
        let count = splitOrCount & 0x0000ffffu;
        let start = node.rightChildOrTriOffset;
        for (var entry = 0u; entry < count; entry += 1u) {
          let permutationIndex = start + entry;
          if (permutationIndex >= tlasInstanceIndexCount()) {
            accumulator.invalidInput = 1u;
            break;
          }
          let instanceIndex = tlasLoadInstanceIndex(permutationIndex);
          if (instanceIndex >= tlasBlasRootCount()) {
            accumulator.invalidInput = 1u;
            break;
          }
          rcTraverseExactBlasCandidates(
            true,
            instanceIndex,
            tlasLoadBlasRoot(instanceIndex),
            ray,
            exclusiveMinT,
            sourceFeature,
            bulkOnly,
            stochasticAlpha,
            &accumulator,
            &selected,
          );
        }
      } else {
        if (stackPointer + 2u > ${TLAS_TRAVERSAL_STACK_DEPTH}u) {
          overflowed = true;
          break;
        }
        stack[stackPointer] = nodeIndex + node.rightChildOrTriOffset;
        stackPointer += 1u;
        stack[stackPointer] = nodeIndex + 1u;
        stackPointer += 1u;
      }
    }
    if (overflowed) {
      accumulator = rcExactSurfaceAccumulatorInit();
      selected = rcExactSurfaceEventEmpty();
      rcTraverseAllExactTlasInstances(
        ray,
        exclusiveMinT,
        sourceFeature,
        bulkOnly,
        stochasticAlpha,
        &accumulator,
        &selected,
      );
    }
  } else {
    rcTraverseExactBlasCandidates(
      false,
      0u,
      0u,
      ray,
      exclusiveMinT,
      sourceFeature,
      bulkOnly,
      stochasticAlpha,
      &accumulator,
      &selected,
    );
  }

  if (accumulator.invalidInput != 0u || accumulator.invalidTie != 0u) {
    selected.status = OPTICAL_BOUNDARY_EVENT_INVALID;
    selected.hit.didHit = false;
    return selected;
  }
  if (accumulator.hasCandidate == 0u) { return selected; }
  if (accumulator.optical != 0u) {
    if (accumulator.frontCount > 0u && accumulator.backCount > 0u) {
      selected.status = select(
        OPTICAL_BOUNDARY_EVENT_INVALID,
        OPTICAL_BOUNDARY_EVENT_TANGENT,
        accumulator.frontCount == accumulator.backCount,
      );
      if (selected.status == OPTICAL_BOUNDARY_EVENT_INVALID) {
        selected.hit.didHit = false;
      }
      return selected;
    }
  }
  selected.status = OPTICAL_BOUNDARY_EVENT_CROSSING;
  return selected;
}

fn rcRetraceExactHit(
  ray: Ray,
  hit: IntersectionResult,
  exclusiveMinT: f32,
) -> OpticalWatertightHit {
  let triangle = rcLoadExactWorldTriangle(
    rcExactUseTlas(), hit.indices.w, hit.instanceIndex,
  );
  if (triangle.valid == 0u) { return opticalWatertightMiss(); }
  return opticalWatertightTriangleIntersect(
    ray.origin,
    ray.direction,
    triangle.a,
    triangle.b,
    triangle.c,
    exclusiveMinT,
  );
}

fn rcExactSourceFeatureForHit(
  hit: IntersectionResult,
  exact: OpticalWatertightHit,
) -> OpticalSourceFeature {
  if (!exact.hit) { return opticalSourceFeatureInvalid(); }
  let useTlas = rcExactUseTlas();
  let triangle = rcLoadExactWorldTriangle(
    useTlas, hit.indices.w, hit.instanceIndex,
  );
  if (triangle.valid == 0u) { return opticalSourceFeatureInvalid(); }
  return opticalCreateSourceFeature(
    rcExactEncodedBoundaryId(useTlas, hit.indices.w, hit.instanceIndex),
    rcExactRepresentedPrimitiveInstanceId(
      useTlas, hit.indices.w, hit.instanceIndex,
    ),
    hit.indices.w,
    exact.zeroEdgeMask,
    triangle.a,
    triangle.b,
    triangle.c,
  );
}

fn rcExactCanonicalPointForHit(
  hit: IntersectionResult,
  exact: OpticalWatertightHit,
) -> vec3f {
  let triangle = rcLoadExactWorldTriangle(
    rcExactUseTlas(), hit.indices.w, hit.instanceIndex,
  );
  if (triangle.valid == 0u || !exact.hit) { return vec3f(0.0); }
  return opticalCanonicalHitPoint(
    exact, triangle.a, triangle.b, triangle.c,
  );
}

fn rcSuffixDiagonalTransfer(value: vec3f) -> mat3x3f {
  return mat3x3f(
    vec3f(value.x, 0.0, 0.0),
    vec3f(0.0, value.y, 0.0),
    vec3f(0.0, 0.0, value.z),
  );
}

fn rcSuffixTransferFinite(transfer: mat3x3f) -> bool {
  let maximum = max(
    max(max(abs(transfer[0].x), abs(transfer[0].y)), abs(transfer[0].z)),
    max(
      max(max(abs(transfer[1].x), abs(transfer[1].y)), abs(transfer[1].z)),
      max(max(abs(transfer[2].x), abs(transfer[2].y)), abs(transfer[2].z)),
    ),
  );
  return maximum < 1e30;
}

fn rcSuffixTransferredChannel(
  transfer: mat3x3f,
  radiance: vec3f,
  channel: u32,
) -> f32 {
  return rcRgbChannel(transfer * radiance, channel);
}

fn rcMediumActiveDistance(
  hasFiniteCap: u32,
  remainingDistance: f32,
  segmentDistance: f32,
) -> f32 {
  let segment = max(segmentDistance, 0.0);
  return select(
    segment,
    min(segment, max(remainingDistance, 0.0)),
    hasFiniteCap != 0u,
  );
}

fn rcMediumRemainingDistanceAfterSegment(
  hasFiniteCap: u32,
  remainingDistance: f32,
  segmentDistance: f32,
) -> f32 {
  if (hasFiniteCap == 0u) {
    // Unbounded media use a separate tag. Never encode infinity and later form
    // the indeterminate Inf-Inf update.
    return 0.0;
  }
  return max(
    max(remainingDistance, 0.0) - max(segmentDistance, 0.0),
    0.0,
  );
}

// Entry-owned absorption over the already-resolved active medium distance.
// Finite KHR thickness is sampled once when the stack layer is created and is
// decremented by callers; this function must never resample a face texel.
fn rcMediumSegmentAbsorption(
  triIndex: u32,
  attenuationColor: vec3f,
  attenuationDistance: f32,
  activeDistance: f32,
) -> vec3f {
  let distance = max(activeDistance, 0.0);
  var absorption = vec3f(1.0);
  if (distance > 0.0) {
    let rgbBeer = beerLambertTransmittanceRgb(
      attenuationColor, attenuationDistance, distance,
    );
    absorption = materialSpectralAttenuation(
      triIndex, distance, rgbBeer,
    );
  }
  return clamp(absorption, vec3f(0.0), vec3f(1.0));
}

// Visibility owns extinction only. It must never add an in-scatter source.
fn rcMediumShadowExtinction(
  triIndex: u32,
  attenuationColor: vec3f,
  attenuationDistance: f32,
  scattering: vec3f,
  activeDistance: f32,
) -> vec3f {
  let distance = max(activeDistance, 0.0);
  let absorption = rcMediumSegmentAbsorption(
    triIndex,
    attenuationColor,
    attenuationDistance,
    distance,
  );
  return clamp(
    absorption * rcHomogeneousBeerTransmittanceRgb(
      max(scattering, vec3f(0.0)), distance,
    ),
    vec3f(0.0),
    vec3f(1.0),
  );
}

// Linear form of the directionally aggregated homogeneous single-scatter
// contract. The matrix is required because luminance-driven in-scatter mixes
// downstream RGB lanes while each represented dispersion lane traces its own
// geometry. This matches the ordinary ReSTIR/NRC dielectric suffix operator.
fn rcMediumRadianceSegmentTransfer(
  triIndex: u32,
  attenuationColor: vec3f,
  attenuationDistance: f32,
  scattering: vec4f,
  albedo: vec3f,
  activeDistance: f32,
) -> mat3x3f {
  let distance = max(activeDistance, 0.0);
  let absorption = rcMediumSegmentAbsorption(
    triIndex,
    attenuationColor,
    attenuationDistance,
    distance,
  );
  if (!(distance > 0.0)) {
    return rcSuffixDiagonalTransfer(absorption);
  }
  let sigmaS = max(scattering.rgb, vec3f(0.0));
  // A zero attenuation-colour lane is an exact +infinite absorption atom for
  // every positive path. Preserve it as infinity instead of flooring to 1e-30:
  // nonzero scattering must not leak an in-scatter source through that lane.
  let positiveInfinity = bitcast<f32>(0x7f800000u);
  var sigmaA = vec3f(0.0);
  sigmaA.x = select(
    positiveInfinity,
    -log(absorption.x) / distance,
    absorption.x > 0.0,
  );
  sigmaA.y = select(
    positiveInfinity,
    -log(absorption.y) / distance,
    absorption.y > 0.0,
  );
  sigmaA.z = select(
    positiveInfinity,
    -log(absorption.z) / distance,
    absorption.z > 0.0,
  );
  let sigmaT = sigmaA + sigmaS;
  let transmittance = absorption *
    rcHomogeneousBeerTransmittanceRgb(sigmaS, distance);
  var scatterAlbedo = vec3f(0.0);
  if (sigmaT.x > 0.0 && absorption.x > 0.0) {
    scatterAlbedo.x = sigmaS.x / sigmaT.x;
  }
  if (sigmaT.y > 0.0 && absorption.y > 0.0) {
    scatterAlbedo.y = sigmaS.y / sigmaT.y;
  }
  if (sigmaT.z > 0.0 && absorption.z > 0.0) {
    scatterAlbedo.z = sigmaS.z / sigmaT.z;
  }
  let sourceScale = max(albedo, vec3f(0.0)) * scatterAlbedo *
    (vec3f(1.0) - transmittance) *
    rcHenyeyGreensteinPhase(0.0, clamp(scattering.a, -0.99, 0.99));
  return mat3x3f(
    vec3f(transmittance.x, 0.0, 0.0) + sourceScale * 0.2126,
    vec3f(0.0, transmittance.y, 0.0) + sourceScale * 0.7152,
    vec3f(0.0, 0.0, transmittance.z) + sourceScale * 0.0722,
  );
}

fn rcShadowInterfaceTransmission(
  hit: IntersectionResult,
  faceNormal: vec3f,
  dir: vec3f,
  etaIncident: vec3f,
  etaTarget: vec3f,
  layerTransmission: vec3f,
) -> vec3f {
  let cosIncident = clamp(abs(dot(faceNormal, -dir)), 0.0, 1.0);
  var transfer = rcDielectricInterfaceTransmissionRgb(
    cosIncident, etaIncident, etaTarget,
  );
  let film = materialThinFilmResponse(
    hit.indices.w, hit.side >= 0.0, cosIncident,
  );
  if (film.present != 0u) { transfer = film.transmittance; }
  return clamp(
    transfer * clamp(layerTransmission, vec3f(0.0), vec3f(1.0)),
    vec3f(0.0),
    vec3f(1.0),
  );
}

fn rcTraceShadowTransmittance(
  origin: vec3f,
  dir: vec3f,
  tMax: f32,
  triEps: f32,
) -> vec3f {
  var visibility = vec3f(1.0);

  var mediumBoundary: array<u32, 16>;
  var mediumRepresented: array<u32, 16>;
  var mediumTri: array<u32, 16>;
  var mediumIor: array<vec3f, 16>;
  var mediumColor: array<vec3f, 16>;
  var mediumDistance: array<f32, 16>;
  var mediumFiniteCap: array<u32, 16>;
  var mediumInitialDistance: array<f32, 16>;
  var mediumRemainingDistance: array<f32, 16>;
  var mediumScattering: array<vec3f, 16>;
  var mediumTransmissionPaid: array<u32, 16>;
  var mediumDepth = 0u;

  let surfaceBudget = rcWorldSurfaceBudget();

  // Fixed-origin outward classification. Matching front/back events beyond
  // the origin cancel on a temporary LIFO. Unmatched back events are precisely
  // the media containing the origin, discovered inner-to-outer and reversed
  // below. Exact-t progression never moves the ray origin.
  let containmentRay = Ray(origin, dir);
  let noSourceFeature = opticalSourceFeatureInvalid();
  var containmentMinT = 0.0;
  var containmentComplete = false;
  var temporaryDepth = 0u;
  var temporaryBoundary: array<u32, 16>;
  var temporaryRepresented: array<u32, 16>;
  for (
    var surface = 0u;
    surface < surfaceBudget;
    surface = surface + 1u
  ) {
    if (containmentComplete) { break; }
    let containmentEvent = rcTraceExactSurfaceEvent(
      containmentRay, containmentMinT, noSourceFeature, true, false,
    );
    if (containmentEvent.status == OPTICAL_BOUNDARY_EVENT_INVALID) {
      return vec3f(0.0);
    }
    if (containmentEvent.status == OPTICAL_BOUNDARY_EVENT_NONE) {
      containmentComplete = true;
      break;
    }
    if (!(containmentEvent.t > containmentMinT)) {
      return vec3f(0.0);
    }
    containmentMinT = containmentEvent.t;
    if (containmentEvent.status == OPTICAL_BOUNDARY_EVENT_TANGENT) {
      continue;
    }
    if (
      containmentEvent.status != OPTICAL_BOUNDARY_EVENT_CROSSING ||
      containmentEvent.encodedBoundaryId == 0u ||
      containmentEvent.representedPrimitiveInstanceId == 0u ||
      !containmentEvent.hit.didHit
    ) { return vec3f(0.0); }
    let containmentHit = containmentEvent.hit;
    let containmentMatId = rcLoadTriMaterialId(containmentHit.indices.w);
    let containmentMat = rcLoadMaterial(containmentMatId);
    if ((containmentMat.flags & MATERIAL_FLAG_CAST_SHADOW_DISABLED) != 0u) {
      continue;
    }
    if (containmentHit.side > 0.0) {
      if (temporaryDepth >= 16u) { return vec3f(0.0); }
      temporaryBoundary[temporaryDepth] = containmentEvent.encodedBoundaryId;
      temporaryRepresented[temporaryDepth] =
        containmentEvent.representedPrimitiveInstanceId;
      temporaryDepth += 1u;
      continue;
    }
    if (containmentHit.side >= 0.0) { return vec3f(0.0); }
    if (temporaryDepth > 0u) {
      let top = temporaryDepth - 1u;
      if (
        temporaryBoundary[top] != containmentEvent.encodedBoundaryId ||
        temporaryRepresented[top] !=
          containmentEvent.representedPrimitiveInstanceId
      ) { return vec3f(0.0); }
      temporaryDepth = top;
      continue;
    }
    if (mediumDepth >= 16u) { return vec3f(0.0); }
    let smoothNormal = rcSmoothNormalForHit(
      containmentHit, containmentHit.normal,
    );
    let shadingNormal = rcApplyBumpMapForHit(
      containmentHit,
      rcApplyNormalMapForHit(containmentHit, smoothNormal),
    );
    let containmentProbeMat = rcSampleProbeHitMaterial(
      containmentHit,
      containmentMat.baseColor,
      containmentMat.roughness,
      containmentMat.metalness,
      smoothNormal,
      shadingNormal,
      containmentMat.transmission,
      containmentMat.ior,
      -containmentRay.direction,
    );
    mediumBoundary[mediumDepth] = containmentEvent.encodedBoundaryId;
    mediumRepresented[mediumDepth] =
      containmentEvent.representedPrimitiveInstanceId;
    mediumTri[mediumDepth] = containmentHit.indices.w;
    mediumIor[mediumDepth] = containmentProbeMat.opticalIor;
    mediumColor[mediumDepth] = clamp(
      containmentMat.attenuationColor, vec3f(0.0), vec3f(1.0),
    );
    mediumDistance[mediumDepth] = containmentMat.attenuationDistance;
    let finiteCap = materialOpticalHasAuthoredThickness(
      containmentHit.indices.w,
    );
    let cap = max(
      containmentProbeMat.bulkThickness *
        containmentProbeMat.thicknessMapScale,
      0.0,
    );
    mediumFiniteCap[mediumDepth] = select(0u, 1u, finiteCap);
    mediumInitialDistance[mediumDepth] = select(0.0, cap, finiteCap);
    mediumRemainingDistance[mediumDepth] = select(0.0, cap, finiteCap);
    mediumScattering[mediumDepth] =
      containmentProbeMat.volumeScattering.rgb;
    mediumTransmissionPaid[mediumDepth] = 0u;
    mediumDepth += 1u;
  }
  if (!containmentComplete || temporaryDepth != 0u) { return vec3f(0.0); }

  // Unmatched exits were encountered inner-to-outer; live medium state is
  // outer-to-inner so its top entry is the physically active nested medium.
  for (var lower = 0u; lower < mediumDepth / 2u; lower += 1u) {
    let upper = mediumDepth - 1u - lower;
    let swapBoundary = mediumBoundary[lower];
    mediumBoundary[lower] = mediumBoundary[upper];
    mediumBoundary[upper] = swapBoundary;
    let swapRepresented = mediumRepresented[lower];
    mediumRepresented[lower] = mediumRepresented[upper];
    mediumRepresented[upper] = swapRepresented;
    let swapTri = mediumTri[lower];
    mediumTri[lower] = mediumTri[upper];
    mediumTri[upper] = swapTri;
    let swapIor = mediumIor[lower];
    mediumIor[lower] = mediumIor[upper];
    mediumIor[upper] = swapIor;
    let swapColor = mediumColor[lower];
    mediumColor[lower] = mediumColor[upper];
    mediumColor[upper] = swapColor;
    let swapDistance = mediumDistance[lower];
    mediumDistance[lower] = mediumDistance[upper];
    mediumDistance[upper] = swapDistance;
    let swapFinite = mediumFiniteCap[lower];
    mediumFiniteCap[lower] = mediumFiniteCap[upper];
    mediumFiniteCap[upper] = swapFinite;
    let swapInitial = mediumInitialDistance[lower];
    mediumInitialDistance[lower] = mediumInitialDistance[upper];
    mediumInitialDistance[upper] = swapInitial;
    let swapRemaining = mediumRemainingDistance[lower];
    mediumRemainingDistance[lower] = mediumRemainingDistance[upper];
    mediumRemainingDistance[upper] = swapRemaining;
    let swapScattering = mediumScattering[lower];
    mediumScattering[lower] = mediumScattering[upper];
    mediumScattering[upper] = swapScattering;
    let swapPaid = mediumTransmissionPaid[lower];
    mediumTransmissionPaid[lower] = mediumTransmissionPaid[upper];
    mediumTransmissionPaid[upper] = swapPaid;
  }

  let fixedRay = Ray(origin, dir);
  let fixedNoSource = opticalSourceFeatureInvalid();
  var exclusiveMinT = 0.0;
  for (var layer = 0u; layer < surfaceBudget; layer = layer + 1u) {
    if (max(max(visibility.x, visibility.y), visibility.z) <= 0.0) {
      return vec3f(0.0);
    }
    if (!(tMax > exclusiveMinT)) {
      return clamp(visibility, vec3f(0.0), vec3f(1.0));
    }
    let event = rcTraceExactSurfaceEvent(
      fixedRay, exclusiveMinT, fixedNoSource, false, false,
    );
    if (event.status == OPTICAL_BOUNDARY_EVENT_INVALID) {
      return vec3f(0.0);
    }
    var acceptedT = tMax;
    if (event.status != OPTICAL_BOUNDARY_EVENT_NONE) {
      if (!(event.t > exclusiveMinT)) { return vec3f(0.0); }
      acceptedT = min(event.t, tMax);
    }
    let segmentDistance = acceptedT - exclusiveMinT;
    if (mediumDepth > 0u) {
      let top = mediumDepth - 1u;
      let activeDistance = rcMediumActiveDistance(
        mediumFiniteCap[top], mediumRemainingDistance[top], segmentDistance,
      );
      visibility = visibility * rcMediumShadowExtinction(
        mediumTri[top],
        mediumColor[top],
        mediumDistance[top],
        mediumScattering[top],
        activeDistance,
      );
      mediumRemainingDistance[top] = rcMediumRemainingDistanceAfterSegment(
        mediumFiniteCap[top], mediumRemainingDistance[top], segmentDistance,
      );
    }
    if (
      event.status == OPTICAL_BOUNDARY_EVENT_NONE ||
      event.t >= tMax
    ) {
      return clamp(visibility, vec3f(0.0), vec3f(1.0));
    }
    exclusiveMinT = event.t;
    if (event.status == OPTICAL_BOUNDARY_EVENT_TANGENT) { continue; }
    if (
      event.status != OPTICAL_BOUNDARY_EVENT_CROSSING ||
      !event.hit.didHit
    ) { return vec3f(0.0); }

    let hit = event.hit;
    let matId = rcLoadTriMaterialId(hit.indices.w);
    let mat = rcLoadMaterial(matId);
    if ((mat.flags & MATERIAL_FLAG_CAST_SHADOW_DISABLED) != 0u) {
      continue;
    }
    let alphaT = rcAlphaShadowTransmittanceForHit(hit);
    let isOptical = rcMaterialIsOptical(mat);
    if (!isOptical) {
      if (alphaT <= 0.0) { return vec3f(0.0); }
      visibility = visibility * alphaT;
      continue;
    }

    var coverage = clamp(1.0 - alphaT, 0.0, 1.0);
    let bulkMedium = event.encodedBoundaryId != 0u;
    if (bulkMedium) {
      let coverageStatus = rcContainmentCoverageStatus(hit);
      if (coverageStatus == RC_CONTAINMENT_COVERAGE_FRACTIONAL) {
        return vec3f(0.0);
      }
      coverage = select(
        0.0, 1.0,
        coverageStatus == RC_CONTAINMENT_COVERAGE_SOLID,
      );
    }
    if (coverage <= 0.0) { continue; }

    let smoothNormal = rcSmoothNormalForHit(hit, hit.normal);
    let shadingNormal = rcApplyBumpMapForHit(
      hit, rcApplyNormalMapForHit(hit, smoothNormal),
    );
    let probeMat = rcSampleProbeHitMaterial(
      hit,
      mat.baseColor,
      mat.roughness,
      mat.metalness,
      smoothNormal,
      shadingNormal,
      mat.transmission,
      mat.ior,
      -dir,
    );
    let alignedNormal = select(
      -shadingNormal,
      shadingNormal,
      dot(shadingNormal, hit.normal) >= 0.0,
    );
    let faceNormal = select(
      -alignedNormal,
      alignedNormal,
      dot(dir, alignedNormal) < 0.0,
    );
    var incidentIor = vec3f(1.0);
    if (mediumDepth > 0u) {
      incidentIor = mediumIor[mediumDepth - 1u];
    }

    if (!bulkMedium) {
      // Reciprocal virtual slab. Surface absorption is paid on both faces;
      // a preintegrated thin-film/TMM response represents the complete film
      // and is therefore paid once, never again at the virtual exit.
      let cosIncident = clamp(abs(dot(faceNormal, -dir)), 0.0, 1.0);
      let etaGreen = max(incidentIor.g, 1e-6) /
        max(probeMat.opticalIor.g, 1e-6);
      let sin2Inside = etaGreen * etaGreen *
        max(0.0, 1.0 - cosIncident * cosIncident);
      if (sin2Inside >= 1.0) { return vec3f(0.0); }
      let cosInside = sqrt(max(0.0, 1.0 - sin2Inside));
      let entryLayer = rcSampleFaceLayerControls(
        hit.indices.w, hit.side >= 0.0,
      );
      let exitLayer = rcSampleFaceLayerControls(
        hit.indices.w, hit.side < 0.0,
      );
      let film = materialThinFilmResponse(
        hit.indices.w, hit.side >= 0.0, cosIncident,
      );
      var opticalTransfer = rcDielectricInterfaceTransmissionRgb(
        cosIncident, incidentIor, probeMat.opticalIor,
      ) * rcDielectricInterfaceTransmissionRgb(
        cosInside, probeMat.opticalIor, incidentIor,
      );
      if (film.present != 0u) {
        opticalTransfer = film.transmittance;
      }
      let fullTransfer =
        clamp(entryLayer.rgb, vec3f(0.0), vec3f(1.0)) *
        clamp(exitLayer.rgb, vec3f(0.0), vec3f(1.0)) *
        clamp(opticalTransfer, vec3f(0.0), vec3f(1.0)) *
        vec3f(clamp(probeMat.transmission, 0.0, 1.0));
      visibility = visibility * mix(
        vec3f(1.0), fullTransfer, vec3f(coverage),
      );
      continue;
    }

    if (coverage != 1.0 || event.representedPrimitiveInstanceId == 0u) {
      return vec3f(0.0);
    }
    if (hit.side >= 0.0) {
      if (mediumDepth >= 16u) { return vec3f(0.0); }
      let interfaceTransmission = rcShadowInterfaceTransmission(
        hit,
        faceNormal,
        dir,
        incidentIor,
        probeMat.opticalIor,
        probeMat.dielectricLayerTransmission,
      );
      visibility = visibility * interfaceTransmission *
        vec3f(clamp(probeMat.transmission, 0.0, 1.0));
      mediumBoundary[mediumDepth] = event.encodedBoundaryId;
      mediumRepresented[mediumDepth] =
        event.representedPrimitiveInstanceId;
      mediumTri[mediumDepth] = hit.indices.w;
      mediumIor[mediumDepth] = probeMat.opticalIor;
      mediumColor[mediumDepth] = clamp(
        mat.attenuationColor, vec3f(0.0), vec3f(1.0),
      );
      mediumDistance[mediumDepth] = mat.attenuationDistance;
      let finiteCap = materialOpticalHasAuthoredThickness(hit.indices.w);
      let cap = max(
        probeMat.bulkThickness * probeMat.thicknessMapScale, 0.0,
      );
      mediumFiniteCap[mediumDepth] = select(0u, 1u, finiteCap);
      mediumInitialDistance[mediumDepth] = select(0.0, cap, finiteCap);
      mediumRemainingDistance[mediumDepth] = select(0.0, cap, finiteCap);
      mediumScattering[mediumDepth] = probeMat.volumeScattering.rgb;
      mediumTransmissionPaid[mediumDepth] = 1u;
      mediumDepth += 1u;
      continue;
    }

    if (mediumDepth == 0u) { return vec3f(0.0); }
    let top = mediumDepth - 1u;
    if (
      mediumBoundary[top] != event.encodedBoundaryId ||
      mediumRepresented[top] != event.representedPrimitiveInstanceId
    ) { return vec3f(0.0); }
    var targetIor = vec3f(1.0);
    if (mediumDepth > 1u) {
      targetIor = mediumIor[mediumDepth - 2u];
    }
    let interfaceTransmission = rcShadowInterfaceTransmission(
      hit,
      faceNormal,
      dir,
      mediumIor[top],
      targetIor,
      probeMat.dielectricLayerTransmission,
    );
    let scalarTransmission = select(
      clamp(probeMat.transmission, 0.0, 1.0),
      1.0,
      mediumTransmissionPaid[top] != 0u,
    );
    visibility = visibility * interfaceTransmission *
      vec3f(scalarTransmission);
    mediumDepth -= 1u;
  }

  return vec3f(0.0);
}

fn rcSampleEmitterLeAtBary(e: EmitterTri, localBary: vec3f, scalarEmission: vec3f) -> vec3f {
  if (
    !rcMaterialAtlasFiniteF32(e._padA) ||
    e._padA < -16777216.0 ||
    e._padA > 16777216.0 ||
    floor(e._padA) != e._padA
  ) {
    return scalarEmission;
  }
  let encodedSourceTri = i32(e._padA);
  if (encodedSourceTri == -1) {
    return scalarEmission;
  }
  let mirroredSourceTri = encodedSourceTri < -1;
  let sourceTri = select(encodedSourceTri, -encodedSourceTri - 2, mirroredSourceTri);
  if (sourceTri < 0) {
    return scalarEmission;
  }
  let triIndex = u32(sourceTri);
  if (triIndex >= arrayLength(&rc_geom_index)) {
    return scalarEmission;
  }
  let tri = rc_geom_index[triIndex].xyz;
  if (tri.x >= arrayLength(&rc_geom_position) || tri.y >= arrayLength(&rc_geom_position) || tri.z >= arrayLength(&rc_geom_position)) {
    return scalarEmission;
  }
  if (tri.x >= arrayLength(&rc_geom_normal) || tri.y >= arrayLength(&rc_geom_normal) || tri.z >= arrayLength(&rc_geom_normal)) {
    return scalarEmission;
  }

  if (
    !rcMaterialAtlasFiniteF32(e._padB) ||
    e._padB < 1.0 ||
    e._padB > 16.0 ||
    floor(e._padB) != e._padB ||
    !rcMaterialAtlasFiniteF32(e._padC) ||
    e._padC < 0.0 ||
    floor(e._padC) != e._padC
  ) {
    return scalarEmission;
  }
  let subdivLevel = u32(e._padB);
  let subdivOrdinalCount = subdivLevel * subdivLevel;
  if (e._padC >= f32(subdivOrdinalCount)) {
    return scalarEmission;
  }
  let subdivOrdinal = u32(e._padC);
  var bary = rcEmitterParentBarycentricFromLocal(
    localBary,
    subdivLevel,
    subdivOrdinal,
  );
  if (mirroredSourceTri) {
    bary = vec3f(bary.z, bary.y, bary.x);
  }

  let uv0a = rcPackedUvFromVec4(rc_geom_position[tri.x]);
  let uv0b = rcPackedUvFromVec4(rc_geom_position[tri.y]);
  let uv0c = rcPackedUvFromVec4(rc_geom_position[tri.z]);
  let uv0 = bary.x * uv0a + bary.y * uv0b + bary.z * uv0c;
  let uv1a = rcPackedUvFromVec4(rc_geom_normal[tri.x]);
  let uv1b = rcPackedUvFromVec4(rc_geom_normal[tri.y]);
  let uv1c = rcPackedUvFromVec4(rc_geom_normal[tri.z]);
  let uv1 = bary.x * uv1a + bary.y * uv1b + bary.z * uv1c;
  let texel = rcSampleMaterialAtlasRawAtOffset(
    triIndex,
    RC_MATERIAL_MAP_EMISSIVE_TEXEL_OFFSET,
    uv0,
    uv1,
  );
  if (texel.valid == 0u) {
    return scalarEmission;
  }
  return rcMaterialAtlasFiniteNonNegativeRadianceOrBlack(
    scalarEmission * texel.value.rgb,
  );
}

${RC_NEE_POINTSPOT_WGSL}

fn rcRgbChannel(value: vec3f, channel: u32) -> f32 {
  if (channel == 0u) { return value.r; }
  if (channel == 1u) { return value.g; }
  return value.b;
}

// Exact unpolarised dielectric interface transmission. This is the same
// per-lane Fresnel accounting used by the hybrid GI glass-prefix walk.
fn rcDielectricInterfaceTransmissionRgb(
  cosIncident: f32,
  etaIncident: vec3f,
  etaTarget: vec3f,
) -> vec3f {
  let ci = clamp(abs(cosIncident), 0.0, 1.0);
  let eta = max(etaIncident, vec3f(1e-6)) / max(etaTarget, vec3f(1e-6));
  let sin2Target = eta * eta * (1.0 - ci * ci);
  let ct = sqrt(max(vec3f(0.0), vec3f(1.0) - sin2Target));
  let rsNumerator = etaIncident * ci - etaTarget * ct;
  let rsDenominator = etaIncident * ci + etaTarget * ct;
  let rpNumerator = etaTarget * ci - etaIncident * ct;
  let rpDenominator = etaTarget * ci + etaIncident * ct;
  let rs = rsNumerator / max(abs(rsDenominator), vec3f(1e-6));
  let rp = rpNumerator / max(abs(rpDenominator), vec3f(1e-6));
  let transmission = max(vec3f(0.0), vec3f(1.0) - 0.5 * (rs * rs + rp * rp));
  return transmission * vec3f(
    select(0.0, 1.0, sin2Target.r < 1.0),
    select(0.0, 1.0, sin2Target.g < 1.0),
    select(0.0, 1.0, sin2Target.b < 1.0),
  );
}

struct RCTransmissionInterfaceSources {
  localSurface: vec3f,
  emission: vec3f,
  unlit: u32,
};

// Split an interface's local source from emission. The direct-response helper
// already applies (1-t) to diffuse while retaining specular/clearcoat/sheen at
// t=1. A paired bulk exit suppresses only the already-paid opaque substrate by
// evaluating that response at t=1; exact reflection and emission remain live.
fn rcShadeTransmissionInterfaceSources(
  hit: IntersectionResult,
  exactHitPoint: vec3f,
  incomingRay: Ray,
  raySeed: u32,
  triEps: f32,
  normalBias: f32,
  slabStep: f32,
  suppressOpaqueSubstrate: bool,
  explicitBulkSegment: bool,
) -> RCTransmissionInterfaceSources {
  var out: RCTransmissionInterfaceSources;
  out.localSurface = vec3f(0.0);
  out.emission = vec3f(0.0);
  out.unlit = 0u;
  let u = rc_u;
  let triIdx = hit.indices.w;
  let matId = rcLoadTriMaterialId(triIdx);
  let mat = rcLoadMaterial(matId);
  let receiverPos = exactHitPoint;
  let smoothNormal = rcSmoothNormalForHit(hit, hit.normal);
  let normalMapped = rcApplyNormalMapForHit(hit, smoothNormal);
  let n = rcApplyBumpMapForHit(hit, normalMapped);
  let wo = -incomingRay.direction;
  let probeMat = rcSampleProbeHitMaterial(
    hit, mat.baseColor, mat.roughness, mat.metalness, smoothNormal, n,
    mat.transmission, mat.ior, wo,
  );
  if ((mat.flags & MATERIAL_FLAG_UNLIT) != 0u) {
    out.emission = probeMat.albedo * probeMat.layerTransmission;
    out.unlit = 1u;
    return out;
  }
  var directMat = probeMat;
  if (suppressOpaqueSubstrate) {
    directMat.transmission = 1.0;
  }
  let toSun = rcSoftSunDirection(
    u.sunDirection, u.sunAngularRadius, receiverPos, u.roomSize, u.cascadeIndex,
  );
  var sunVis = vec3f(1.0);
  if (u.sunCastShadowDisabled == 0u) {
    sunVis = traceSunVisibility(
      receiverPos + n * normalBias, toSun, slabStep, triEps,
    );
  }
  let directSun = u.sunColor *
    rcEvaluateProbeDirectResponse(directMat, n, wo, toSun) * sunVis;
  let emitterNEE = rcEmitterNEE(
    receiverPos, n, wo, directMat, u.emitterCount, raySeed, triEps, normalBias,
  );
  let pointSpotLights = evalRCPointSpotLights(
    receiverPos, n, wo, directMat, normalBias, triEps, raySeed,
  );
  out.emission = select(
    vec3f(0.0),
    rcSampleSurfaceEmissiveMap(hit, mat.emissive),
    hit.side >= 0.0 ||
      (mat.flags & MATERIAL_FLAG_DOUBLE_SIDED) != 0u,
  ) * probeMat.layerTransmission;
  let bakedOutgoing = directMat.albedo *
    (1.0 - clamp(directMat.metalness, 0.0, 1.0)) *
    (1.0 - clamp(directMat.transmission, 0.0, 1.0)) * RC_INV_PI *
    rcSampleLightMapIrradiance(hit) * probeMat.layerTransmission;
  let radiance = directSun + emitterNEE + pointSpotLights + bakedOutgoing;
  out.localSurface = radiance;
  // Closed optical bulk is owned by the explicit suffix segment operator,
  // which uses the actual incoming distance exactly once. Keep its boundary
  // sources raw; the directionally aggregated proxy remains only for materials
  // that have no explicit bulk segment topology.
  if (!suppressOpaqueSubstrate && !explicitBulkSegment) {
    out.localSurface = rcApplyHomogeneousVolumeSingleScatter(
      radiance, directMat.albedo, probeMat.volumeScattering,
      probeMat.bulkThickness * probeMat.thicknessMapScale, n, wo,
    );
  }
  return out;
}

// Once an interval first hits a transmissive surface, that invocation owns the
// complete dielectric suffix. Every later scene-to-scene segment is bounded by
// the scene AABB diagonal. The runtime interface budget bounds those segments,
// and one additional segment reaches the final opaque receiver. The initial
// interval-local hit distance is accounted for separately.
fn rcCompleteDielectricSuffixMaxDistance(initialHitDistance: f32) -> f32 {
  let u = rc_u;
  let boundedSegments = min(u.transmittedInterfaceBudget, 8u) + 1u;
  return initialHitDistance + length(u.roomSize) * f32(boundedSegments);
}

fn rcSuffixRandom(raySeed: u32, eventIndex: u32, salt: u32) -> f32 {
  return pcgHashToF32(
    raySeed ^ (eventIndex * 0x9e3779b9u) ^ salt,
  );
}

// Trace one correlated spectral lane through a runtime-selected 1..8 exact
// dielectric-event budget. Local opaque/direct response is accumulated once;
// reflection has a unit envelope while scalar material transmission weights
// the transmitted family. This retains Fresnel reflection and TIR at t=1
// without paying t twice at a paired bulk exit. Medium state is entry-owned and
// supports alpha-aware starts-inside plus nested material+instance identities.
fn rcTraceDielectricSuffixChannel(
  initialRay: Ray,
  initialHit: IntersectionResult,
  maxDistance: f32,
  channel: u32,
  raySeed: u32,
  triEps: f32,
  normalBias: f32,
  slabStep: f32,
  firstLocalSurfaceSource: vec3f,
  firstEmissionSource: vec3f,
) -> f32 {
  let u = rc_u;
  const RC_GLASS_STATIC_MAX_INTERFACES: u32 = 8u;
  let interfaceBudget = min(
    u.transmittedInterfaceBudget, RC_GLASS_STATIC_MAX_INTERFACES,
  );
  var ray = initialRay;
  var hit = initialHit;
  var throughput = rcSuffixDiagonalTransfer(vec3f(1.0));
  var accumulatedRadiance = 0.0;
  var arrivedWithoutNeeOwner = true;
  var travelled = 0.0;
  var interfaceCount = 0u;
  var mediumDepth = 0u;
  var mediumIor: array<vec3f, 8>;
  var mediumTri: array<u32, 8>;
  var mediumBoundary: array<u32, 8>;
  var mediumRepresented: array<u32, 8>;
  var mediumAttenuationColor: array<vec3f, 8>;
  var mediumAttenuationDistance: array<f32, 8>;
  var mediumFiniteCap: array<u32, 8>;
  var mediumInitialDistance: array<f32, 8>;
  var mediumRemainingDistance: array<f32, 8>;
  var mediumScattering: array<vec4f, 8>;
  var mediumAlbedo: array<vec3f, 8>;
  var mediumTransmissionPaid: array<u32, 8>;

  // The initial hit came from exact equal-t grouping in the entry kernel. A
  // canonical retrace is still mandatory before it can become a continuation
  // source: this pins face/edge/vertex ownership to represented triangle data.
  let initialExact = rcRetraceExactHit(initialRay, initialHit, 0.0);
  if (!initialExact.hit) { return 0.0; }
  var currentSourceFeature = rcExactSourceFeatureForHit(
    initialHit, initialExact,
  );
  if (currentSourceFeature.kind == OPTICAL_SOURCE_FEATURE_INVALID) {
    return 0.0;
  }
  var currentHitPoint = rcExactCanonicalPointForHit(initialHit, initialExact);
  if (!rcContainmentFiniteVec3(currentHitPoint)) { return 0.0; }
  var currentBoundary = rcExactEncodedBoundaryId(
    rcExactUseTlas(), initialHit.indices.w, initialHit.instanceIndex,
  );
  var currentRepresented = rcExactRepresentedPrimitiveInstanceId(
    rcExactUseTlas(), initialHit.indices.w, initialHit.instanceIndex,
  );
  if (currentRepresented == 0u) { return 0.0; }

  // Fixed-origin outward classification. Matching entries/exits beyond the
  // origin cancel on a temporary LIFO. Unmatched exits are exactly the closed
  // media containing the origin; reverse them into outer-to-inner live order.
  let surfaceBudget = rcWorldSurfaceBudget();
  let containmentRay = Ray(initialRay.origin, initialRay.direction);
  let noSourceFeature = opticalSourceFeatureInvalid();
  var containmentMinT = 0.0;
  var containmentComplete = false;
  var temporaryDepth = 0u;
  var temporaryBoundary: array<u32, 8>;
  var temporaryRepresented: array<u32, 8>;
  for (
    var surface = 0u;
    surface < surfaceBudget;
    surface = surface + 1u
  ) {
    let containmentEvent = rcTraceExactSurfaceEvent(
      containmentRay, containmentMinT, noSourceFeature, true, false,
    );
    if (containmentEvent.status == OPTICAL_BOUNDARY_EVENT_INVALID) {
      return 0.0;
    }
    if (containmentEvent.status == OPTICAL_BOUNDARY_EVENT_NONE) {
      containmentComplete = true;
      break;
    }
    if (!(containmentEvent.t > containmentMinT)) { return 0.0; }
    containmentMinT = containmentEvent.t;
    if (containmentEvent.status == OPTICAL_BOUNDARY_EVENT_TANGENT) {
      continue;
    }
    if (
      containmentEvent.status != OPTICAL_BOUNDARY_EVENT_CROSSING ||
      containmentEvent.encodedBoundaryId == 0u ||
      containmentEvent.representedPrimitiveInstanceId == 0u ||
      !containmentEvent.hit.didHit
    ) { return 0.0; }
    let containmentHit = containmentEvent.hit;
    if (containmentHit.side > 0.0) {
      if (temporaryDepth >= RC_GLASS_STATIC_MAX_INTERFACES) {
        return 0.0;
      }
      temporaryBoundary[temporaryDepth] = containmentEvent.encodedBoundaryId;
      temporaryRepresented[temporaryDepth] =
        containmentEvent.representedPrimitiveInstanceId;
      temporaryDepth += 1u;
      continue;
    }
    if (containmentHit.side >= 0.0) { return 0.0; }
    if (temporaryDepth > 0u) {
      let top = temporaryDepth - 1u;
      if (
        temporaryBoundary[top] != containmentEvent.encodedBoundaryId ||
        temporaryRepresented[top] !=
          containmentEvent.representedPrimitiveInstanceId
      ) { return 0.0; }
      temporaryDepth = top;
      continue;
    }
    if (mediumDepth >= RC_GLASS_STATIC_MAX_INTERFACES) { return 0.0; }
    let containmentMatId = rcLoadTriMaterialId(containmentHit.indices.w);
    let containmentMat = rcLoadMaterial(containmentMatId);
    let smoothNormal = rcSmoothNormalForHit(
      containmentHit, containmentHit.normal,
    );
    let shadingNormal = rcApplyBumpMapForHit(
      containmentHit,
      rcApplyNormalMapForHit(containmentHit, smoothNormal),
    );
    let containmentProbeMat = rcSampleProbeHitMaterial(
      containmentHit,
      containmentMat.baseColor,
      containmentMat.roughness,
      containmentMat.metalness,
      smoothNormal,
      shadingNormal,
      containmentMat.transmission,
      containmentMat.ior,
      -containmentRay.direction,
    );
    mediumBoundary[mediumDepth] = containmentEvent.encodedBoundaryId;
    mediumRepresented[mediumDepth] =
      containmentEvent.representedPrimitiveInstanceId;
    mediumIor[mediumDepth] = containmentProbeMat.opticalIor;
    mediumTri[mediumDepth] = containmentHit.indices.w;
    mediumAttenuationColor[mediumDepth] = clamp(
      containmentMat.attenuationColor, vec3f(0.0), vec3f(1.0),
    );
    mediumAttenuationDistance[mediumDepth] =
      containmentMat.attenuationDistance;
    let finiteCap = materialOpticalHasAuthoredThickness(
      containmentHit.indices.w,
    );
    let cap = max(
      containmentProbeMat.bulkThickness *
        containmentProbeMat.thicknessMapScale,
      0.0,
    );
    mediumFiniteCap[mediumDepth] = select(0u, 1u, finiteCap);
    mediumInitialDistance[mediumDepth] = select(0.0, cap, finiteCap);
    mediumRemainingDistance[mediumDepth] = select(0.0, cap, finiteCap);
    mediumScattering[mediumDepth] = containmentProbeMat.volumeScattering;
    mediumAlbedo[mediumDepth] = containmentProbeMat.albedo;
    mediumTransmissionPaid[mediumDepth] = 0u;
    mediumDepth += 1u;
  }
  if (!containmentComplete || temporaryDepth != 0u) { return 0.0; }

  for (var lower = 0u; lower < mediumDepth / 2u; lower += 1u) {
    let upper = mediumDepth - 1u - lower;
    let swapBoundary = mediumBoundary[lower];
    mediumBoundary[lower] = mediumBoundary[upper];
    mediumBoundary[upper] = swapBoundary;
    let swapRepresented = mediumRepresented[lower];
    mediumRepresented[lower] = mediumRepresented[upper];
    mediumRepresented[upper] = swapRepresented;
    let swapTri = mediumTri[lower];
    mediumTri[lower] = mediumTri[upper];
    mediumTri[upper] = swapTri;
    let swapIor = mediumIor[lower];
    mediumIor[lower] = mediumIor[upper];
    mediumIor[upper] = swapIor;
    let swapColor = mediumAttenuationColor[lower];
    mediumAttenuationColor[lower] = mediumAttenuationColor[upper];
    mediumAttenuationColor[upper] = swapColor;
    let swapDistance = mediumAttenuationDistance[lower];
    mediumAttenuationDistance[lower] = mediumAttenuationDistance[upper];
    mediumAttenuationDistance[upper] = swapDistance;
    let swapFinite = mediumFiniteCap[lower];
    mediumFiniteCap[lower] = mediumFiniteCap[upper];
    mediumFiniteCap[upper] = swapFinite;
    let swapInitial = mediumInitialDistance[lower];
    mediumInitialDistance[lower] = mediumInitialDistance[upper];
    mediumInitialDistance[upper] = swapInitial;
    let swapRemaining = mediumRemainingDistance[lower];
    mediumRemainingDistance[lower] = mediumRemainingDistance[upper];
    mediumRemainingDistance[upper] = swapRemaining;
    let swapScattering = mediumScattering[lower];
    mediumScattering[lower] = mediumScattering[upper];
    mediumScattering[upper] = swapScattering;
    let swapAlbedo = mediumAlbedo[lower];
    mediumAlbedo[lower] = mediumAlbedo[upper];
    mediumAlbedo[upper] = swapAlbedo;
    let swapPaid = mediumTransmissionPaid[lower];
    mediumTransmissionPaid[lower] = mediumTransmissionPaid[upper];
    mediumTransmissionPaid[upper] = swapPaid;
  }

  // Eight statically bounded interfaces plus one final opaque receiver
  // inspection. The runtime budget can stop transmissive continuation earlier.
  for (
    var inspection = 0u;
    inspection <= RC_GLASS_STATIC_MAX_INTERFACES;
    inspection = inspection + 1u
  ) {
    travelled = travelled + hit.dist;
    if (travelled > maxDistance) { return accumulatedRadiance; }

    if (mediumDepth > 0u) {
      let top = mediumDepth - 1u;
      let activeDistance = rcMediumActiveDistance(
        mediumFiniteCap[top], mediumRemainingDistance[top], hit.dist,
      );
      let segmentTransfer = rcMediumRadianceSegmentTransfer(
        mediumTri[top],
        mediumAttenuationColor[top],
        mediumAttenuationDistance[top],
        mediumScattering[top],
        mediumAlbedo[top],
        activeDistance,
      );
      throughput = throughput * segmentTransfer;
      mediumRemainingDistance[top] = rcMediumRemainingDistanceAfterSegment(
        mediumFiniteCap[top], mediumRemainingDistance[top], hit.dist,
      );
      if (!rcSuffixTransferFinite(throughput)) {
        return accumulatedRadiance;
      }
    }

    let triIdx = hit.indices.w;
    let matId = rcLoadTriMaterialId(triIdx);
    let mat = rcLoadMaterial(matId);
    let smoothNormal = rcSmoothNormalForHit(hit, hit.normal);
    let shadingNormal = rcApplyBumpMapForHit(
      hit, rcApplyNormalMapForHit(hit, smoothNormal),
    );
    let probeMat = rcSampleProbeHitMaterial(
      hit, mat.baseColor, mat.roughness, mat.metalness,
      smoothNormal, shadingNormal, mat.transmission, mat.ior, -ray.direction,
    );

    let isOptical = rcMaterialIsOptical(mat);
    if (isOptical && currentRepresented == 0u) { return accumulatedRadiance; }
    let entering = hit.side > 0.0;
    let hasBulkTopology = isOptical && currentBoundary != 0u;
    var validBulkExit = false;
    var pairedPaidExit = false;
    var malformedBulkExit = false;
    if (isOptical && hasBulkTopology && !entering) {
      if (mediumDepth == 0u) {
        malformedBulkExit = true;
      } else {
        let top = mediumDepth - 1u;
        validBulkExit =
          mediumBoundary[top] == currentBoundary &&
          mediumRepresented[top] == currentRepresented;
        malformedBulkExit = !validBulkExit;
        if (validBulkExit) {
          pairedPaidExit = mediumTransmissionPaid[top] != 0u;
        }
      }
    }

    var localSurfaceSource = firstLocalSurfaceSource;
    var emissionSource = firstEmissionSource;
    var interfaceUnlit = 0u;
    if (inspection > 0u) {
      let sources = rcShadeTransmissionInterfaceSources(
        hit, currentHitPoint, ray, raySeed, triEps, normalBias, slabStep,
        pairedPaidExit, hasBulkTopology,
      );
      localSurfaceSource = sources.localSurface;
      emissionSource = sources.emission;
      interfaceUnlit = sources.unlit;
    }

    // Unlit is a terminal emission closure even when the material also carries
    // glass/transmission metadata. The first hit is handled by the kernel; this
    // closes the same contract for an unlit receiver reached later in a suffix.
    if (interfaceUnlit != 0u) {
      accumulatedRadiance = accumulatedRadiance +
        rcSuffixTransferredChannel(throughput, emissionSource, channel);
      return accumulatedRadiance;
    }

    // The camera/probe-visible first hit owns its emitted radiance. Later
    // emitter hits retain Le only when the sampled event had no explicit-light
    // NEE owner (transmission, a delta interface, or an empty emitter table), or
    // when the destination material explicitly opts out of emitter sampling.
    // Rough reflection/TIR paths otherwise suppress Le to avoid double counting.
    let includeEmitterEmission =
      inspection == 0u ||
      arrivedWithoutNeeOwner ||
      (mat.flags & MATERIAL_FLAG_SKIP_EMITTER) != 0u;
    if (includeEmitterEmission) {
      accumulatedRadiance = accumulatedRadiance +
        rcSuffixTransferredChannel(throughput, emissionSource, channel);
    }
    if (malformedBulkExit) { return accumulatedRadiance; }

    // Direct diffuse already carries (1-t), while explicit-light reflection
    // lobes remain live at t=1. A paid bulk exit was evaluated with t=1 above,
    // so its opaque substrate is not charged a second time.
    accumulatedRadiance = accumulatedRadiance +
      rcSuffixTransferredChannel(throughput, localSurfaceSource, channel);
    if (!isOptical) { return accumulatedRadiance; }

    if (interfaceCount >= interfaceBudget) {
      return accumulatedRadiance;
    }

    let alignedInterfaceNormal = select(
      -shadingNormal,
      shadingNormal,
      dot(shadingNormal, hit.normal) >= 0.0,
    );
    // Orient against the incident ray itself. Thin sheets have no enclosing
    // inside/outside volume, so their optical result must not depend on mesh
    // winding or which geometric side the ray sees first.
    let faceNormal = select(
      -alignedInterfaceNormal,
      alignedInterfaceNormal,
      dot(ray.direction, alignedInterfaceNormal) < 0.0,
    );
    if (dot(ray.direction, faceNormal) >= -1e-6) {
      return accumulatedRadiance;
    }
    let anisotropyFrame = rcMaterialTangentFrameForHit(
      hit, faceNormal, RC_MATERIAL_MAP_ANISOTROPY_TEXEL_OFFSET,
    );
    var incidentIor = vec3f(1.0);
    if (mediumDepth > 0u) {
      incidentIor = mediumIor[mediumDepth - 1u];
    }
    var targetIor = probeMat.opticalIor;
    if (validBulkExit) {
      targetIor = vec3f(1.0);
      if (mediumDepth > 1u) {
        targetIor = mediumIor[mediumDepth - 2u];
      }
    }

    let mappedTransmission = clamp(probeMat.transmission, 0.0, 1.0);
    let transmissionPhysicalWeight = select(
      mappedTransmission, 1.0, pairedPaidExit,
    );
    let transmissionBranchPdf = represented_bernoulli_probability_f32(
      transmissionPhysicalWeight / (1.0 + transmissionPhysicalWeight),
    );
    let reflectionBranchPdf = 1.0 - transmissionBranchPdf;
    let chooseTransmission =
      transmissionBranchPdf > 0.0 &&
      rcSuffixRandom(raySeed, interfaceCount, 0x4252414eu) <
        transmissionBranchPdf;

    let interfaceXi = vec2f(
      rcSuffixRandom(raySeed, interfaceCount, 0x564e4446u),
      rcSuffixRandom(raySeed, interfaceCount, 0x48454954u),
    );
    let interfaceLobe = rcSampleDielectricLobe(
      triIdx,
      hit.side >= 0.0,
      faceNormal,
      anisotropyFrame.tangent,
      anisotropyFrame.bitangent,
      -ray.direction,
      probeMat.roughness,
      probeMat.anisotropy.x,
      probeMat.anisotropy.y,
      incidentIor,
      targetIor,
      probeMat.dielectricLayerTransmission,
      channel,
      true,
      !chooseTransmission,
      interfaceXi,
    );
    interfaceCount = interfaceCount + 1u;
    if (interfaceLobe.valid == 0u) { return accumulatedRadiance; }
    if (chooseTransmission) {
      throughput = throughput * rcSuffixDiagonalTransfer(
        interfaceLobe.weightRgb *
          transmissionPhysicalWeight / transmissionBranchPdf,
      );
    } else {
      throughput = throughput * rcSuffixDiagonalTransfer(
        interfaceLobe.weightRgb / reflectionBranchPdf,
      );
    }
    if (!rcSuffixTransferFinite(throughput)) {
      return accumulatedRadiance;
    }
    var nextArrivesWithoutNeeOwner =
      interfaceLobe.kind == RC_DIELECTRIC_EVENT_TRANSMISSION ||
      probeMat.roughness <= 0.0 ||
      u.emitterCount == 0u;
    ray.direction = interfaceLobe.direction;

    if (chooseTransmission) {
      if (!hasBulkTopology) {
        // A thin sheet is one bounded reciprocal compound event: entry and a
        // forced reciprocal exit. It has no bulk stack and no unbounded
        // internal-reflection walk. The preintegrated thin-film/TMM response
        // was applied at entry and is explicitly disabled at the virtual exit.
        if (interfaceCount >= interfaceBudget) {
          return accumulatedRadiance;
        }
        let exitFrontFacing = hit.side < 0.0;
        let exitLayer = rcSampleFaceLayerControls(triIdx, exitFrontFacing);
        let exitRoughness = select(
          probeMat.roughness,
          clamp(exitLayer.a, 0.0, 1.0),
          exitLayer.a >= 0.0,
        );
        let exitFrame = rcMaterialTangentFrameForHit(
          hit, faceNormal, RC_MATERIAL_MAP_ANISOTROPY_TEXEL_OFFSET,
        );
        let exitXi = vec2f(
          rcSuffixRandom(raySeed, interfaceCount, 0x534c5631u),
          rcSuffixRandom(raySeed, interfaceCount, 0x534c5632u),
        );
        let exitLobe = rcSampleDielectricLobe(
          triIdx,
          exitFrontFacing,
          faceNormal,
          exitFrame.tangent,
          exitFrame.bitangent,
          -ray.direction,
          exitRoughness,
          probeMat.anisotropy.x,
          probeMat.anisotropy.y,
          targetIor,
          incidentIor,
          clamp(exitLayer.rgb, vec3f(0.0), vec3f(1.0)),
          channel,
          false,
          false,
          exitXi,
        );
        interfaceCount += 1u;
        if (
          exitLobe.valid == 0u ||
          exitLobe.kind != RC_DIELECTRIC_EVENT_TRANSMISSION
        ) { return accumulatedRadiance; }
        throughput = throughput * rcSuffixDiagonalTransfer(
          exitLobe.weightRgb,
        );
        if (!rcSuffixTransferFinite(throughput)) {
          return accumulatedRadiance;
        }
        nextArrivesWithoutNeeOwner = true;
        ray.direction = exitLobe.direction;
      } else if (entering) {
        if (mediumDepth >= RC_GLASS_STATIC_MAX_INTERFACES) {
          return accumulatedRadiance;
        }
        mediumIor[mediumDepth] = targetIor;
        mediumTri[mediumDepth] = triIdx;
        mediumBoundary[mediumDepth] = currentBoundary;
        mediumRepresented[mediumDepth] = currentRepresented;
        mediumAttenuationColor[mediumDepth] = clamp(
          mat.attenuationColor, vec3f(0.0), vec3f(1.0),
        );
        mediumAttenuationDistance[mediumDepth] = mat.attenuationDistance;
        let finiteCap = materialOpticalHasAuthoredThickness(triIdx);
        let cap = max(
          probeMat.bulkThickness * probeMat.thicknessMapScale, 0.0,
        );
        mediumFiniteCap[mediumDepth] = select(0u, 1u, finiteCap);
        mediumInitialDistance[mediumDepth] = select(0.0, cap, finiteCap);
        mediumRemainingDistance[mediumDepth] = select(0.0, cap, finiteCap);
        mediumScattering[mediumDepth] = probeMat.volumeScattering;
        mediumAlbedo[mediumDepth] = probeMat.albedo;
        mediumTransmissionPaid[mediumDepth] = 1u;
        mediumDepth = mediumDepth + 1u;
      } else if (validBulkExit) {
        mediumDepth = mediumDepth - 1u;
      }
    }

    // Continue from the exact represented point. The source-feature exclusion
    // removes only the just-departed face/edge/vertex; a one-ULP nested layer
    // remains visible because neither the origin nor distance is epsilon-shifted.
    ray.origin = currentHitPoint;
    var nextEvent = rcExactSurfaceEventEmpty();
    var nextResolved = false;
    var nextEscaped = false;
    var nextMinT = 0.0;
    for (var candidate = 0u; candidate < surfaceBudget; candidate += 1u) {
      nextEvent = rcTraceExactSurfaceEvent(
        ray, nextMinT, currentSourceFeature, false, true,
      );
      if (nextEvent.status == OPTICAL_BOUNDARY_EVENT_INVALID) {
        return accumulatedRadiance;
      }
      if (nextEvent.status == OPTICAL_BOUNDARY_EVENT_NONE) {
        nextEscaped = true;
        break;
      }
      if (!(nextEvent.t > nextMinT)) { return accumulatedRadiance; }
      nextMinT = nextEvent.t;
      if (nextEvent.status == OPTICAL_BOUNDARY_EVENT_TANGENT) { continue; }
      if (
        nextEvent.status != OPTICAL_BOUNDARY_EVENT_CROSSING ||
        !nextEvent.hit.didHit
      ) { return accumulatedRadiance; }
      nextResolved = true;
      break;
    }
    arrivedWithoutNeeOwner = nextArrivesWithoutNeeOwner;
    if (nextEscaped) {
      if (mediumDepth != 0u) { return accumulatedRadiance; }
      let env = rcEnvironmentRadiance(ray.direction);
      return accumulatedRadiance +
        rcSuffixTransferredChannel(throughput, env, channel);
    }
    if (!nextResolved) { return accumulatedRadiance; }

    hit = nextEvent.hit;
    let nextExact = rcRetraceExactHit(ray, hit, 0.0);
    if (!nextExact.hit || nextExact.t != nextEvent.t) {
      return accumulatedRadiance;
    }
    currentHitPoint = rcExactCanonicalPointForHit(hit, nextExact);
    if (!rcContainmentFiniteVec3(currentHitPoint)) {
      return accumulatedRadiance;
    }
    currentSourceFeature = rcExactSourceFeatureForHit(hit, nextExact);
    if (currentSourceFeature.kind == OPTICAL_SOURCE_FEATURE_INVALID) {
      return accumulatedRadiance;
    }
    currentBoundary = nextEvent.encodedBoundaryId;
    currentRepresented = nextEvent.representedPrimitiveInstanceId;
  }
  return accumulatedRadiance;
}

// ─── Entry point ─────────────────────────────────────────────────────────────
// Verbatim from probeRayCastKernel wgslFn body.
// TSL instanceIndex → @builtin(global_invocation_id) globalId, index = globalId.x.
// TSL storage ptr params → @group(0)/@binding(N) module-scope vars accessed by reference.

@compute @workgroup_size(64)
fn probeRayCastKernel(@builtin(global_invocation_id) globalId: vec3u) {
  let index = globalId.x;
  let u = rc_u;
  let totalProbes  = u.probeCount.x * u.probeCount.y * u.probeCount.z;
  let totalThreads = totalProbes * u.raysPerProbe;
  if (index >= totalThreads) { return; }

  let probeIdx = index / u.raysPerProbe;
  let rayIdx   = index % u.raysPerProbe;

  let pz = probeIdx / (u.probeCount.x * u.probeCount.y);
  let py = (probeIdx / u.probeCount.x) % u.probeCount.y;
  let px = probeIdx % u.probeCount.x;
  let probeUV  = (vec3f(f32(px), f32(py), f32(pz)) + 0.5) / vec3f(u.probeCount);
  let probePos = u.probeOriginWorld + probeUV * u.roomSize;

  // Shared with the hybrid receiver: both sides reconstruct the same
  // stratified UV sample, direction, and octahedral-Jacobian weight.
  let raySeed = rcStratifiedRaySeed(probeIdx, rayIdx, u.frameSeed);
  let rayUV   = rcStratifiedRayUV(probeIdx, rayIdx, u.rayGridSize, u.frameSeed);
  let rayDir  = octDecode(rayUV * 2.0 - 1.0);

  var ray = Ray();
  ray.origin    = probePos + rayDir * u.intervalNear;
  ray.direction = rayDir;
  let maxT = u.intervalFar - u.intervalNear;

  var radiance     = vec3f(0.0);
  var escaped      = true;

  // E2: read epsilon from CascadeUniforms (UBO-plumbed from HybridEngine.triIntersectEpsilon).
  let triEps = u.triIntersectEpsilon;

  // Initial transport uses the same inclusive watertight equal-t grouping as
  // every continuation. Distinct coincident optical ranges fail closed;
  // tangential balanced groups advance only the exact t lower bound.
  let initialNoSource = opticalSourceFeatureInvalid();
  let initialSurfaceBudget = rcWorldSurfaceBudget();
  var initialMinT = 0.0;
  var initialEvent = rcExactSurfaceEventEmpty();
  var initialResolved = false;
  var initialInvalid = false;
  for (
    var candidate = 0u;
    candidate < initialSurfaceBudget;
    candidate += 1u
  ) {
    initialEvent = rcTraceExactSurfaceEvent(
      ray, initialMinT, initialNoSource, false, true,
    );
    if (initialEvent.status == OPTICAL_BOUNDARY_EVENT_INVALID) {
      initialInvalid = true;
      break;
    }
    if (initialEvent.status == OPTICAL_BOUNDARY_EVENT_NONE) { break; }
    if (!(initialEvent.t > initialMinT)) {
      initialInvalid = true;
      break;
    }
    initialMinT = initialEvent.t;
    if (initialEvent.t > maxT) { break; }
    if (initialEvent.status == OPTICAL_BOUNDARY_EVENT_TANGENT) { continue; }
    if (
      initialEvent.status != OPTICAL_BOUNDARY_EVENT_CROSSING ||
      !initialEvent.hit.didHit
    ) {
      initialInvalid = true;
      break;
    }
    initialResolved = true;
    break;
  }
  var hit = initialEvent.hit;

  // M14: scene-scale-proportional step to clear glass slabs/faces.
  // Uses the smallest room axis * 0.001 so the offset is never
  // Cornell-tuned (0.5 units) but scales with actual scene extents.
  let slabStep = min(u.roomSize.x, min(u.roomSize.y, u.roomSize.z)) * 0.001;

  if (initialInvalid) {
    // Invalid represented topology is absorbing/fail-closed, not an escape
    // that may import radiance from a later cascade.
    escaped = false;
  } else if (!initialResolved || !hit.didHit || hit.dist > maxT) {
    if (u.cascadeIndex == u.lastCascade) {
      radiance = rcEnvironmentRadiance(rayDir);
      escaped = false;
    }
  } else {
    escaped = false;

    let triIdx = hit.indices.w;
    let matId  = rcLoadTriMaterialId(triIdx);
    let mat    = rcLoadMaterial(matId);

    // A7: scene-scale-proportional normal bias for all shadow rays.
    // min(roomSize) * 0.001 mirrors DDGI's gridParams.spacing * 0.001 (M13).
    let normalBias = min(u.roomSize.x, min(u.roomSize.y, u.roomSize.z)) * 0.001;
    let firstSources = rcShadeTransmissionInterfaceSources(
      hit, initialEvent.point, ray, raySeed, triEps, normalBias, slabStep,
      false, initialEvent.encodedBoundaryId != 0u,
    );

    if (firstSources.unlit != 0u) {
      radiance = firstSources.emission;
    } else if (rcMaterialIsOptical(mat)) {
      // One correlated RGB event stream owns the complete dielectric suffix:
      // local surface response, exact Fresnel reflection/TIR, transmitted
      // nested media, thin-sheet internal reflection, and terminal receiver.
      let dielectricMaxDistance =
        rcCompleteDielectricSuffixMaxDistance(hit.dist);
      radiance = vec3f(
        rcTraceDielectricSuffixChannel(
          ray, hit, dielectricMaxDistance, 0u,
          raySeed, triEps, normalBias, slabStep,
          firstSources.localSurface, firstSources.emission,
        ),
        rcTraceDielectricSuffixChannel(
          ray, hit, dielectricMaxDistance, 1u,
          raySeed, triEps, normalBias, slabStep,
          firstSources.localSurface, firstSources.emission,
        ),
        rcTraceDielectricSuffixChannel(
          ray, hit, dielectricMaxDistance, 2u,
          raySeed, triEps, normalBias, slabStep,
          firstSources.localSurface, firstSources.emission,
        ),
      );
    } else {
      radiance = firstSources.localSurface + firstSources.emission;
    }
  }

  let outIdx = probeIdx * u.raysPerProbe + rayIdx;
  let escapedF = select(1.0, 0.0, escaped);
  rc_cascadeOut[outIdx] = vec4f(radiance, escapedF);
}
`;
