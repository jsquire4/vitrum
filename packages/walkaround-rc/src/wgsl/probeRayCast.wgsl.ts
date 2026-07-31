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
  BVH_INTERSECT_CORE_WGSL,
  MATERIAL_ENTRY_WGSL,
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
${RC_ENVIRONMENT_RADIANCE_SCALE_WGSL}

// Portable value-return loader seam for RC's binding names.
fn bvhLoadNode(index: u32) -> BVHNode { return rc_bvh[index]; }
fn bvhLoadIndex(index: u32) -> vec4u { return rc_geom_index[index]; }
fn bvhLoadPosition(index: u32) -> vec4f { return rc_geom_position[index]; }

// rc_scene_arena header: seven (wordOffset, elementCount) pairs followed by
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
fn rcTraceFirstHit(ray: Ray, triEps: f32) -> IntersectionResult {
  let u = rc_u;
  if (u.bvhMode == 1u && u.tlasNodeCount > 0u) {
    return traceTlasFirstHit(
      u.tlasNodeCount,
      ray,
      triEps,
    );
  }
  return bvhIntersectFirstHit(ray, triEps);
}

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
  return u32(max(emitter.emitterFlags, 0.0));
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
  out.mode = u32(max(coverageMeta.x, 0.0) + 0.5);
  if (out.mode == 0u) {
    return out;
  }

  let uvs = rcHitMaterialUvs(hit);
  if (uvs.valid == 0u) {
    return out;
  }

  let baseColorTexel = rcSampleMaterialAtlasRaw(hit.indices.w, RC_MATERIAL_MAP_SLOT_BASE_COLOR, uvs.uv0, uvs.uv1);
  let hasBaseColorMap = rcMaterialAtlasMapAvailableAtOffset(
    hit.indices.w,
    RC_MATERIAL_MAP_SLOT_BASE_COLOR * 2u,
  );
  let baseColorAlpha = select(1.0, clamp(baseColorTexel.a, 0.0, 1.0), hasBaseColorMap);
  let alphaTexel = rcSampleMaterialAtlasRaw(hit.indices.w, RC_MATERIAL_MAP_SLOT_ALPHA, uvs.uv0, uvs.uv1);
  let hasAlphaMap = rcMaterialAtlasMapAvailableAtOffset(
    hit.indices.w,
    RC_MATERIAL_MAP_SLOT_ALPHA * 2u,
  );
  let alphaMapCoverage = select(1.0, clamp(alphaTexel.r, 0.0, 1.0), hasAlphaMap);
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
  return f32(seed) / 4294967296.0;
}

fn rcMaterialAlphaDiscardedForProbeHit(hit: IntersectionResult, ray: Ray, layer: u32) -> bool {
  if (!rcMaterialSideAdmittedForHit(hit)) {
    return true;
  }
  let alpha = rcMaterialAlphaCoverageForHit(hit);
  if (alpha.mode == 0u) { return false; }
  if (alpha.mode == 1u) { return alpha.coverage < alpha.cutoff; }
  if (alpha.mode == 2u) {
    return alpha.coverage < 1.0 && rcAlphaBlendCoverageHash(hit, ray, layer) >= alpha.coverage;
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

fn rcTraceFirstHitAlphaTextured(ray: Ray, triEps: f32) -> IntersectionResult {
  var walkRay = ray;
  var traveled = 0.0;
  let step = triEps * 4.0;
  let surfaceBudget = rcWorldSurfaceBudget();
  for (var layer = 0u; layer < surfaceBudget; layer = layer + 1u) {
    var hit = rcTraceFirstHit(walkRay, triEps);
    if (!hit.didHit) { return hit; }
    if (!rcMaterialAlphaDiscardedForProbeHit(hit, ray, layer)) {
      hit.dist = hit.dist + traveled;
      return hit;
    }
    traveled = traveled + hit.dist + step;
    walkRay.origin = ray.origin + ray.direction * traveled;
  }
  var exhausted = rcTraceFirstHit(walkRay, triEps);
  if (exhausted.didHit) { exhausted.dist = exhausted.dist + traveled; }
  return exhausted;
}

fn rcTraceShadowTransmittance(
  origin: vec3f,
  dir: vec3f,
  tMax: f32,
  triEps: f32,
) -> vec3f {
  var walkRay = Ray();
  walkRay.origin = origin;
  walkRay.direction = dir;
  var traveled = 0.0;
  var visibility = vec3f(1.0);
  let step = triEps * 4.0;

  var mediumMaterial: array<u32, 16>;
  var mediumTri: array<u32, 16>;
  var mediumInstance: array<u32, 16>;
  var mediumColor: array<vec3f, 16>;
  var mediumDistance: array<f32, 16>;
  var mediumDepth = 0u;

  let surfaceBudget = rcWorldSurfaceBudget();
  for (var layer = 0u; layer < surfaceBudget; layer = layer + 1u) {
    let remaining = tMax - traveled;
    if (
      remaining <= step ||
      max(max(visibility.x, visibility.y), visibility.z) <= 0.0
    ) {
      return clamp(visibility, vec3f(0.0), vec3f(1.0));
    }

    let hit = rcTraceFirstHit(walkRay, triEps);
    if (!hit.didHit || hit.dist >= remaining) {
      if (mediumDepth > 0u) {
        let top = mediumDepth - 1u;
        let rgbBeer = beerLambertTransmittanceRgb(
          mediumColor[top], mediumDistance[top], remaining,
        );
        visibility = visibility * materialSpectralAttenuation(
          mediumTri[top], remaining, rgbBeer,
        );
      }
      return clamp(visibility, vec3f(0.0), vec3f(1.0));
    }

    if (mediumDepth > 0u) {
      let top = mediumDepth - 1u;
      let rgbBeer = beerLambertTransmittanceRgb(
        mediumColor[top], mediumDistance[top], hit.dist,
      );
      visibility = visibility * materialSpectralAttenuation(
        mediumTri[top], hit.dist, rgbBeer,
      );
      if (max(max(visibility.x, visibility.y), visibility.z) <= 0.0) {
        return vec3f(0.0);
      }
    }

    let matId = rcLoadTriMaterialId(hit.indices.w);
    let mat = rcLoadMaterial(matId);
    if ((mat.flags & MATERIAL_FLAG_CAST_SHADOW_DISABLED) == 0u) {
      let alphaT = rcAlphaShadowTransmittanceForHit(hit);
      let isGlass = (mat.flags & MATERIAL_FLAG_IS_GLASS) != 0u;
      if (!isGlass) {
        if (alphaT <= 0.0) { return vec3f(0.0); }
        visibility = visibility * alphaT;
      } else if (
        hit.side < 0.0 &&
        mediumDepth > 0u &&
        mediumMaterial[mediumDepth - 1u] == matId &&
        mediumInstance[mediumDepth - 1u] == hit.instanceIndex
      ) {
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
        visibility = visibility * probeMat.layerTransmission;
        mediumDepth = mediumDepth - 1u;
      } else {
        let coverage = clamp(1.0 - alphaT, 0.0, 1.0);
        if (coverage > 0.0) {
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
          let interfaceTransmission =
            probeMat.layerTransmission *
            vec3f(clamp(probeMat.transmission, 0.0, 1.0));

          if (coverage < 1.0 || probeMat.bulkThickness <= 0.0) {
            visibility = visibility * mix(
              vec3f(1.0),
              interfaceTransmission,
              vec3f(coverage),
            );
          } else if (hit.side >= 0.0) {
            if (mediumDepth >= 16u) { return vec3f(0.0); }
            visibility = visibility * interfaceTransmission;
            mediumMaterial[mediumDepth] = matId;
            mediumTri[mediumDepth] = hit.indices.w;
            mediumInstance[mediumDepth] = hit.instanceIndex;
            mediumColor[mediumDepth] = clamp(
              mat.attenuationColor, vec3f(0.0), vec3f(1.0),
            );
            mediumDistance[mediumDepth] = mat.attenuationDistance;
            mediumDepth = mediumDepth + 1u;
          } else if (mediumDepth > 0u) {
            return vec3f(0.0);
          } else {
            let rgbBeer = beerLambertTransmittanceRgb(
              clamp(mat.attenuationColor, vec3f(0.0), vec3f(1.0)),
              mat.attenuationDistance,
              hit.dist,
            );
            visibility = visibility * interfaceTransmission *
              materialSpectralAttenuation(
                hit.indices.w, hit.dist, rgbBeer,
              );
          }
        }
      }
    }

    if (mediumDepth > 0u) {
      let top = mediumDepth - 1u;
      let stepBeer = beerLambertTransmittanceRgb(
        mediumColor[top], mediumDistance[top], step,
      );
      visibility = visibility * materialSpectralAttenuation(
        mediumTri[top], step, stepBeer,
      );
    }
    traveled = traveled + hit.dist + step;
    walkRay.origin = origin + dir * traveled;
  }

  return vec3f(0.0);
}

fn rcSampleEmitterLeAtBary(e: EmitterTri, localBary: vec3f, scalarEmission: vec3f) -> vec3f {
  let encodedSourceTri = i32(round(e._padA));
  if (encodedSourceTri == -1) {
    return scalarEmission;
  }
  let mirroredSourceTri = encodedSourceTri < -1;
  let sourceTri = select(encodedSourceTri, -encodedSourceTri - 2, mirroredSourceTri);
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

  var bary = rcEmitterParentBarycentricFromLocal(localBary, e._padB, e._padC);
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
  if (!rcMaterialAtlasMapAvailableAtOffset(
    triIndex,
    RC_MATERIAL_MAP_EMISSIVE_TEXEL_OFFSET,
  )) {
    return scalarEmission;
  }
  return scalarEmission * texel.rgb;
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

fn rcShadeOpaqueTransmissionReceiver(
  hit: IntersectionResult,
  incomingRay: Ray,
  raySeed: u32,
  triEps: f32,
  normalBias: f32,
  slabStep: f32,
) -> vec3f {
  let u = rc_u;
  let triIdx = hit.indices.w;
  let matId = rcLoadTriMaterialId(triIdx);
  let mat = rcLoadMaterial(matId);
  let receiverPos = incomingRay.origin + incomingRay.direction * hit.dist;
  let smoothNormal = rcSmoothNormalForHit(hit, hit.normal);
  let normalMapped = rcApplyNormalMapForHit(hit, smoothNormal);
  let n = rcApplyBumpMapForHit(hit, normalMapped);
  let wo = -incomingRay.direction;
  let probeMat = rcSampleProbeHitMaterial(
    hit, mat.baseColor, mat.roughness, mat.metalness, smoothNormal, n,
    mat.transmission, mat.ior, wo,
  );
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
    rcEvaluateProbeDirectResponse(probeMat, n, wo, toSun) * sunVis;
  let emitterNEE = rcEmitterNEE(
    receiverPos, n, wo, probeMat, u.emitterCount, raySeed, triEps, normalBias,
  );
  let pointSpotLights = evalRCPointSpotLights(
    receiverPos, n, wo, probeMat, normalBias, triEps, raySeed,
  );
  let emissive = select(
    vec3f(0.0),
    rcSampleSurfaceEmissiveMap(hit, mat.emissive),
    hit.side >= 0.0 ||
      (mat.flags & MATERIAL_FLAG_DOUBLE_SIDED) != 0u,
  );
  let bakedOutgoing = probeMat.albedo * RC_INV_PI *
    rcSampleLightMapIrradiance(hit);
  var radiance = (
    directSun + emitterNEE + pointSpotLights + emissive + bakedOutgoing
  ) * probeMat.layerTransmission;
  radiance = rcApplyHomogeneousVolumeSingleScatter(
    radiance, probeMat.albedo, probeMat.volumeScattering,
    probeMat.bulkThickness, n, wo,
  );
  return radiance;
}

// Once an interval first hits a transmissive surface, that invocation owns the
// complete dielectric suffix. Every later scene-to-scene segment is bounded by
// the scene AABB diagonal. The runtime interface budget bounds those segments,
// and one additional segment reaches the final opaque receiver. The initial
// interval-local hit distance is accounted for separately.
fn rcCompleteDielectricSuffixMaxDistance(initialHitDistance: f32) -> f32 {
  let u = rc_u;
  let boundedSegments = u.transmittedInterfaceBudget + 1u;
  return initialHitDistance + length(u.roomSize) * f32(boundedSegments);
}

// Trace one spectral channel through a runtime-selected 1..8 dielectric
// interface budget. Each
// channel owns its refracted direction, so dispersion changes the geometry it
// actually sees instead of only recolouring one green-channel path. The medium
// stack carries absolute RGB IOR and actual-segment Beer state; TIR, malformed
// open media, and interface-budget overflow fail closed.
fn rcTraceTransmittedChannel(
  initialRay: Ray,
  initialHit: IntersectionResult,
  maxDistance: f32,
  channel: u32,
  raySeed: u32,
  triEps: f32,
  normalBias: f32,
  slabStep: f32,
) -> f32 {
  let u = rc_u;
  const RC_GLASS_STATIC_MAX_INTERFACES: u32 = 8u;
  let interfaceBudget = u.transmittedInterfaceBudget;
  var ray = initialRay;
  var hit = initialHit;
  var throughput = 1.0;
  var travelled = 0.0;
  var interfaceCount = 0u;
  var mediumDepth = 0u;
  var mediumIor: array<vec3f, 8>;
  var mediumTri: array<u32, 8>;
  var mediumInstance: array<u32, 8>;
  var mediumAttenuationColor: array<vec3f, 8>;
  var mediumAttenuationDistance: array<f32, 8>;

  // Eight statically bounded interfaces plus one final opaque receiver
  // inspection. The runtime budget can stop transmissive continuation earlier.
  for (var step = 0u; step <= RC_GLASS_STATIC_MAX_INTERFACES; step = step + 1u) {
    travelled = travelled + hit.dist;
    if (travelled > maxDistance) { return 0.0; }

    if (mediumDepth > 0u) {
      let top = mediumDepth - 1u;
      let attenuationDistance = mediumAttenuationDistance[top];
      if (!(attenuationDistance > 0.0)) { return 0.0; }
      let rgbBeer = beerLambertTransmittanceRgb(
        mediumAttenuationColor[top],
        attenuationDistance,
        hit.dist,
      );
      let spectralBeer = materialSpectralAttenuation(
        mediumTri[top], hit.dist, rgbBeer,
      );
      throughput = throughput * rcRgbChannel(spectralBeer, channel);
      if (!(throughput > 0.0)) { return 0.0; }
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

    if (!(probeMat.transmission > 0.0)) {
      // Reconstruct the world-space receiver position for the local shading
      // helper by translating its direction-times-distance result.
      let receiverRadiance = rcShadeOpaqueTransmissionReceiver(
        hit, ray, raySeed, triEps, normalBias, slabStep,
      );
      return throughput * rcRgbChannel(receiverRadiance, channel);
    }
    let hitPos = ray.origin + ray.direction * hit.dist;
    let entering = hit.side >= 0.0;
    let thickness = max(probeMat.bulkThickness, 0.0);
    let thinSheet = thickness <= 0.0;
    // A bulk back face without a tracked entry is not a continuation from
    // air. A thin sheet is reciprocal and may be viewed from either side.
    if (thickness > 0.0 && !entering && mediumDepth == 0u) {
      return 0.0;
    }
    // Thin-sheet T² crosses two dielectric boundaries even though the sheet
    // is represented by one geometric hit, so it consumes two interface slots.
    let interfaceCost = select(1u, 2u, thinSheet);
    if (interfaceCount + interfaceCost > interfaceBudget) {
      return 0.0;
    }
    interfaceCount = interfaceCount + interfaceCost;
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
    if (dot(ray.direction, faceNormal) >= 0.0) { return 0.0; }
    // The incident medium is always the current enclosing medium (stack top or
    // air). In particular, a geometric backside hit on a thin sheet does NOT
    // mean the ray started inside the sheet: it still enters the sheet IOR and
    // immediately crosses the reciprocal virtual exit below.
    var incidentIor = vec3f(1.0);
    if (mediumDepth > 0u) {
      incidentIor = mediumIor[mediumDepth - 1u];
    }
    var targetIor = probeMat.opticalIor;
    if (!thinSheet && !entering) {
      let top = mediumDepth - 1u;
      // mediumTri identifies the entered optical medium through its material
      // record in merged mode. TLAS mode additionally requires the exact
      // instance that was pushed: two nested instances may legally share both
      // geometry and material, so material identity alone cannot pair exits.
      // Never pop a different nested medium merely because a back face was
      // encountered out of order: malformed/open geometry fails closed.
      if (rcLoadTriMaterialId(mediumTri[top]) != matId) { return 0.0; }
      if (u.bvhMode == 1u && mediumInstance[top] != hit.instanceIndex) {
        return 0.0;
      }
      targetIor = vec3f(1.0);
      if (mediumDepth > 1u) {
        targetIor = mediumIor[mediumDepth - 2u];
      }
    }

    let etaIncidentChannel = rcRgbChannel(incidentIor, channel);
    let etaTargetChannel = rcRgbChannel(targetIor, channel);
    let btdfXi = vec2f(
      pcgHashToF32(raySeed ^ (step * 0x9e3779b9u) ^ (channel * 0x85ebca6bu)),
      pcgHashToF32(raySeed ^ (step * 0xc2b2ae35u) ^ (channel * 0x27d4eb2fu)),
    );
    let interfaceBtdf = rcSampleGgxDielectricTransmission(
      faceNormal,
      -ray.direction,
      probeMat.roughness,
      etaIncidentChannel,
      etaTargetChannel,
      btdfXi,
    );
    if (interfaceBtdf.valid == 0u) { return 0.0; }
    let interfaceCos = interfaceBtdf.microfacetCos;
    var interfaceT = rcDielectricInterfaceTransmissionRgb(
      interfaceCos, incidentIor, targetIor,
    );
    let film = materialThinFilmResponse(
      triIdx, hit.side >= 0.0, interfaceCos,
    );
    if (film.present != 0u) { interfaceT = film.transmittance; }
    throughput = throughput * rcRgbChannel(interfaceT, channel) *
      (interfaceBtdf.weight / interfaceBtdf.transmission) *
      rcRgbChannel(probeMat.layerTransmission, channel);

    var nextDirection = interfaceBtdf.direction;
    if (thinSheet) {
      let exitXi = vec2f(
        pcgHashToF32(raySeed ^ (step * 0x165667b1u) ^ (channel * 0xd3a2646cu)),
        pcgHashToF32(raySeed ^ (step * 0xfd7046c5u) ^ (channel * 0xb55a4f09u)),
      );
      let exitLayer = rcSampleFaceLayerControls(triIdx, hit.side < 0.0);
      let exitRoughness = select(
        probeMat.roughness,
        clamp(exitLayer.a, 0.0, 1.0),
        exitLayer.a >= 0.0,
      );
      let exitBtdf = rcSampleGgxDielectricTransmission(
        faceNormal,
        -interfaceBtdf.direction,
        exitRoughness,
        etaTargetChannel,
        etaIncidentChannel,
        exitXi,
      );
      if (exitBtdf.valid == 0u) { return 0.0; }
      var exitT = rcDielectricInterfaceTransmissionRgb(
        exitBtdf.microfacetCos, targetIor, incidentIor,
      );
      let exitFilm = materialThinFilmResponse(
        triIdx, hit.side < 0.0, exitBtdf.microfacetCos,
      );
      if (exitFilm.present != 0u) { exitT = exitFilm.transmittance; }
      throughput = throughput * rcRgbChannel(exitT, channel) *
        (exitBtdf.weight / exitBtdf.transmission) *
        rcRgbChannel(clamp(exitLayer.rgb, vec3f(0.0), vec3f(1.0)), channel) *
        probeMat.transmission;
      nextDirection = exitBtdf.direction;
    } else {
      if (entering) { throughput = throughput * probeMat.transmission; }

      if (entering) {
        if (mediumDepth >= RC_GLASS_STATIC_MAX_INTERFACES) { return 0.0; }
        mediumIor[mediumDepth] = probeMat.opticalIor;
        mediumTri[mediumDepth] = triIdx;
        mediumInstance[mediumDepth] = hit.instanceIndex;
        mediumAttenuationColor[mediumDepth] = mat.attenuationColor;
        mediumAttenuationDistance[mediumDepth] = mat.attenuationDistance;
        mediumDepth = mediumDepth + 1u;
      } else if (mediumDepth > 0u) {
        mediumDepth = mediumDepth - 1u;
      }
    }
    if (!(throughput > 0.0)) { return 0.0; }

    let continuationStep = max(slabStep, triEps * 4.0);
    if (mediumDepth > 0u) {
      let top = mediumDepth - 1u;
      let stepBeer = beerLambertTransmittanceRgb(
        mediumAttenuationColor[top],
        mediumAttenuationDistance[top],
        continuationStep,
      );
      let stepSpectral = materialSpectralAttenuation(
        mediumTri[top], continuationStep, stepBeer,
      );
      throughput = throughput * rcRgbChannel(stepSpectral, channel);
      if (!(throughput > 0.0)) { return 0.0; }
    }
    travelled = travelled + continuationStep;
    ray.direction = nextDirection;
    ray.origin = hitPos + nextDirection * continuationStep;
    hit = rcTraceFirstHitAlphaTextured(ray, triEps);
    if (!hit.didHit) {
      if (mediumDepth != 0u) { return 0.0; }
      let env = rcEnvironmentRadiance(ray.direction);
      return throughput * rcRgbChannel(env, channel);
    }
  }
  return 0.0;
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

  let hit = rcTraceFirstHitAlphaTextured(ray, triEps);

  // M14: scene-scale-proportional step to clear glass slabs/faces.
  // Uses the smallest room axis * 0.001 so the offset is never
  // Cornell-tuned (0.5 units) but scales with actual scene extents.
  let slabStep = min(u.roomSize.x, min(u.roomSize.y, u.roomSize.z)) * 0.001;

  if (!hit.didHit || hit.dist > maxT) {
    if (u.cascadeIndex == u.lastCascade) {
      radiance = rcEnvironmentRadiance(rayDir);
      escaped = false;
    }
  } else {
    escaped = false;

    let triIdx = hit.indices.w;
    let matId  = rcLoadTriMaterialId(triIdx);
    let mat    = rcLoadMaterial(matId);

    let hitPos = ray.origin + ray.direction * hit.dist;
    let geoNormal = hit.normal;
    let smoothNormal = rcSmoothNormalForHit(hit, geoNormal);
    let normalMapped = rcApplyNormalMapForHit(hit, smoothNormal);
    let n = rcApplyBumpMapForHit(hit, normalMapped);
    let wo = -ray.direction;

    let probeMat = rcSampleProbeHitMaterial(
      hit, mat.baseColor, mat.roughness, mat.metalness, smoothNormal, n,
      mat.transmission, mat.ior, wo,
    );
    let matColor    = probeMat.albedo;
    let matAtten    = mat.attenuationColor;
    let matEmissive = mat.emissive;

    // A7: scene-scale-proportional normal bias for all shadow rays.
    // min(roomSize) * 0.001 mirrors DDGI's gridParams.spacing * 0.001 (M13).
    let normalBias = min(u.roomSize.x, min(u.roomSize.y, u.roomSize.z)) * 0.001;

    let toSun = rcSoftSunDirection(u.sunDirection, u.sunAngularRadius, hitPos, u.roomSize, u.cascadeIndex);
    var sunVis = vec3f(1.0);
    if (u.sunCastShadowDisabled == 0u) {
      sunVis = traceSunVisibility(hitPos + n * normalBias, toSun, slabStep, triEps);
    }
    let directSun = u.sunColor * rcEvaluateProbeDirectResponse(probeMat, n, wo, toSun) * sunVis;

    // Rect-area emitter NEE (2026-06-07): closes the regime gap where RC saw
    // sun + emissive geometry + env but NOT the abstract rect-area emitter
    // list. emitterCount==0 ⇒ no-op (RC's prior light model, byte-identical).
    let emitterNEE = rcEmitterNEE(hitPos, n, wo, probeMat, u.emitterCount, raySeed, triEps, normalBias);

    // One bounded-work punctual/directional alias sample. lightCount==0 ⇒ no-op.
    let pointSpotLights = evalRCPointSpotLights(hitPos, n, wo, probeMat, normalBias, triEps, raySeed);

    let emissive = select(
      vec3f(0.0),
      rcSampleSurfaceEmissiveMap(hit, matEmissive),
      hit.side >= 0.0 ||
        (mat.flags & MATERIAL_FLAG_DOUBLE_SIDED) != 0u,
    );
    let bakedOutgoing = probeMat.albedo * RC_INV_PI *
      rcSampleLightMapIrradiance(hit);

    // Bounded RGB-dispersive dielectric continuation. Each colour channel
    // traces its own Snell path through the configured 1..8 interface budget;
    // a glass exit is
    // never shaded as the receiver. Fresnel, mapped transmission, and actual-
    // segment Beer/spectral attenuation are paid inside the channel walk.
    var transContrib = vec3f(0.0);
    if (probeMat.transmission > 0.0) {
      let dielectricMaxDistance =
        rcCompleteDielectricSuffixMaxDistance(hit.dist);
      transContrib = vec3f(
        rcTraceTransmittedChannel(
          ray, hit, dielectricMaxDistance, 0u,
          raySeed, triEps, normalBias, slabStep,
        ),
        rcTraceTransmittedChannel(
          ray, hit, dielectricMaxDistance, 1u,
          raySeed, triEps, normalBias, slabStep,
        ),
        rcTraceTransmittedChannel(
          ray, hit, dielectricMaxDistance, 2u,
          raySeed, triEps, normalBias, slabStep,
        ),
      );
    }

    radiance = (
      directSun + emitterNEE + pointSpotLights + emissive + bakedOutgoing
    ) * probeMat.layerTransmission + transContrib;
    radiance = rcApplyHomogeneousVolumeSingleScatter(
      radiance, probeMat.albedo, probeMat.volumeScattering,
      probeMat.bulkThickness, n, wo,
    );
  }

  let outIdx = probeIdx * u.raysPerProbe + rayIdx;
  let escapedF = select(1.0, 0.0, escaped);
  rc_cascadeOut[outIdx] = vec4f(radiance, escapedF);
}
`;
