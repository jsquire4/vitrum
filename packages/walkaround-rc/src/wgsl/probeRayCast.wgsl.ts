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
 *   1. three-mesh-bvh constants (BVH_STACK_DEPTH, INFINITY, TRI_INTERSECT_EPSILON)
 *   2. three-mesh-bvh structs (Ray, BVHBoundingBox, BVHNode, IntersectionResult)
 *   3. three-mesh-bvh functions (intersectsBounds, intersectsTriangle, intersectTriangles,
 *      bvhIntersectFirstHit)
 *   4. CascadeUniforms struct (40 floats = 160 bytes; must match cascadeDispatch.ts layout)
 *   5. MaterialEntry struct (16 f32 fields = 64 bytes; must match @vitrum/shared-bvh layout)
 *   6. Octahedral helpers (octEncode, octDecode) — body stripped of file header
 *   7. PCG hash utilities (pcgHashToF32 from @vitrum/shared-samplers PCG_HASH_TO_F32_WGSL)
 *   8. Probe-ray helpers (dirToEquirectUV)
 *   9. Sun visibility helper (traceSunVisibility)
 *  10. Entry-point function with @compute @workgroup_size(64)
 *
 * The original TSL `instanceIndex` built-in (global thread index) becomes
 * `@builtin(global_invocation_id) globalId: vec3u` with `let index = globalId.x;`.
 *
 * Source-of-truth: the WGSL function bodies below are taken VERBATIM from the
 * `wgslFn()` source strings in `probeRayCast.wgsl.ts` (and from three-mesh-bvh's
 * own `.wgsl.js` files).  No semantic changes.
 *
 * See `src/rc/TSL_TO_RAW_MAPPING.md` for the full mapping rationale.
 */

import {
  BVH_CAST_SHADOW_PREDICATE_WGSL,
  BVH_INTERSECT_WGSL,
  MATERIAL_ENTRY_WGSL,
  TLAS_TRAVERSAL_WGSL,
} from '@vitrum/shared-bvh';
import { OCTAHEDRAL_CORE_WGSL, PCG_HASH_TO_F32_WGSL } from '@vitrum/shared-samplers';
import { RC_SUN_VISIBILITY_WGSL, RC_NEE_POINTSPOT_WGSL } from './rcLightEval.wgsl.js';

export const PROBE_RAY_CAST_WGSL = /* wgsl */`
${MATERIAL_ENTRY_WGSL}
${BVH_INTERSECT_WGSL}
${TLAS_TRAVERSAL_WGSL}

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
  let len2 = dot(v, v);
  if (len2 < 1e-20) { return vec3f(0.0, 1.0, 0.0); }
  return v * inverseSqrt(len2);
}

fn bvhCastShadowDisabledForTri(triIdx: u32) -> bool {
  let matId = rc_triMatId[triIdx];
  return (rc_materials[matId].flags & MATERIAL_FLAG_CAST_SHADOW_DISABLED) != 0u;
}

${BVH_CAST_SHADOW_PREDICATE_WGSL}

// C2 — merged world BVH vs TLAS+local BLAS (same traversal as ReSTIR / DDGI).
fn rcTraceFirstHit(ray: Ray, triEps: f32) -> IntersectionResult {
  let u = rc_u_arr[0];
  if (u.bvhMode == 1u && u.tlasNodeCount > 0u) {
    return traceTlasFirstHit(
      &rc_tlas_nodes,
      &rc_tlas_instance_indices,
      &rc_tlas_blas_roots,
      &rc_tlas_w2l,
      &rc_tlas_l2w,
      u.tlasNodeCount,
      &rc_geom_index,
      &rc_geom_position,
      &rc_bvh,
      ray,
      triEps,
    );
  }
  return bvhIntersectFirstHit(&rc_geom_index, &rc_geom_position, &rc_bvh, ray, triEps);
}

fn rcTraceAny(origin: vec3f, dir: vec3f, tMax: f32, triEps: f32, skipGlass: bool) -> bool {
  let u = rc_u_arr[0];
  if (u.bvhMode == 1u && u.tlasNodeCount > 0u) {
    return traceTlasAny(
      &rc_tlas_nodes,
      &rc_tlas_instance_indices,
      &rc_tlas_blas_roots,
      &rc_tlas_w2l,
      &rc_tlas_l2w,
      u.tlasNodeCount,
      &rc_geom_index,
      &rc_geom_position,
      &rc_bvh,
      origin,
      dir,
      tMax,
      triEps,
      skipGlass,
    );
  }
  return bvhIntersectAny(
    &rc_geom_index, &rc_geom_position, &rc_bvh, origin, dir, tMax, triEps, skipGlass,
  );
}

fn rcTraceAnyCastShadow(origin: vec3f, dir: vec3f, tMax: f32, triEps: f32, skipGlass: bool) -> bool {
  let u = rc_u_arr[0];
  if (u.bvhMode == 1u && u.tlasNodeCount > 0u) {
    return traceTlasAnyCastPredicate(
      &rc_tlas_nodes,
      &rc_tlas_instance_indices,
      &rc_tlas_blas_roots,
      &rc_tlas_w2l,
      &rc_tlas_l2w,
      u.tlasNodeCount,
      &rc_geom_index,
      &rc_geom_position,
      &rc_bvh,
      origin,
      dir,
      tMax,
      triEps,
      skipGlass,
    );
  }
  return bvhIntersectAnyAtRootCastPredicate(
    &rc_geom_index, &rc_geom_position, &rc_bvh, origin, dir, tMax, triEps, skipGlass, 0u,
  );
}

// ─── CascadeUniforms struct ───────────────────────────────────────────────────
// Must match buildCascadeUniformDataInto() layout in cascadeDispatch.ts
// (40 floats = 160 bytes).
//
// E2 — triIntersectEpsilon: Möller–Trumbore coplanarity threshold (f32).
// Replaces the local const TRI_INTERSECT_EPSILON: f32 = 1e-5 that was
// previously hardcoded.  Occupies the first of the two former _pad4 slots;
// the second becomes _pad4a.  Total size unchanged: 40 f32/u32 = 160 bytes.

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
  // A7 (2026-06-10): number of point/spot analytic lights in rc_lights.
  // 0 ⇒ analytic-light evaluation is skipped (byte-identical with prior).
  // Host packs this at slot 30 (offset 120) in buildCascadeUniformDataInto.
  lightCount        : u32,
  sunCastShadowDisabled: u32,
  _pad4             : u32,
  _pad5             : u32,
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
  castShadowDisabled: f32,
};

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

// ─── Probe-ray helpers ────────────────────────────────────────────────────────
// Verbatim from probeRayHelpers wgslFn in probeRayCast.wgsl.ts.

fn dirToEquirectUV(d: vec3f) -> vec2f {
  let phi   = atan2(d.z, d.x);
  let theta = acos(clamp(d.y, -1.0, 1.0));
  return vec2f(phi / (2.0 * 3.14159265) + 0.5, 1.0 - theta / 3.14159265);
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
  if (radius <= 1e-7) {
    return sunBase;
  }
  let quant = vec3i(floor(hitPos / max(min(roomSize.x, min(roomSize.y, roomSize.z)), 1e-4) * 1024.0));
  let seed =
    bitcast<u32>(quant.x) * 0x9E3779B9u ^
    bitcast<u32>(quant.y) * 0x85EBCA6Bu ^
    bitcast<u32>(quant.z) * 0xC2B2AE35u ^
    cascadeIndex * 0x27D4EB2Du ^
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
// 64-byte analytic point/spot light for RC probe rays (A7, 2026-06-10).
// Layout mirrors DDGI's DDGILight (probeUpdateRays.wgsl.ts) exactly so the
// same DDGIProbeLights-style host buffer can be shared into RC via binding 15.
//
// Host packs at most RC_MAX_LIGHTS entries (16, matching DDGI's cap) into
// rc_lights using packRCLights() in HybridEngineRC.ts.  lightCount == 0 ⇒
// the evalRCPointSpotLights loop is a no-op (byte-identical with prior).
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
const RC_LIGHT_KIND_MASK: u32 = 0x7fffffffu;
const RC_LIGHT_CAST_SHADOW_DISABLED: u32 = 0x80000000u;
const RC_MAX_LIGHTS:  u32 = 16u;

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

// Header: count (u32) + 3 × u32 pad = 16 bytes, then up to 16 × RCLight entries.
struct RCLightBuffer {
  count: u32,
  _h0: u32, _h1: u32, _h2: u32,
  items: array<RCLight, 16>,
};

// ─── Bind group declarations ──────────────────────────────────────────────────
// Cast pass: @group(0) bindings 0-20.

@group(0) @binding(0) var<storage, read>       rc_bvh:                   array<BVHNode>;
@group(0) @binding(1) var<storage, read>       rc_geom_index:            array<vec4u>;
@group(0) @binding(2) var<storage, read>       rc_geom_position:         array<vec4f>;
@group(0) @binding(3) var<storage, read>       rc_materials:             array<MaterialEntry>;
@group(0) @binding(4) var<storage, read>       rc_triMatId:              array<u32>;
@group(0) @binding(5) var<storage, read_write> rc_cascadeOut:            array<vec4f>;
@group(0) @binding(6) var                      rc_envMap:                texture_2d<f32>;
@group(0) @binding(7) var                      rc_envSampler:            sampler;
@group(0) @binding(8) var<storage, read>       rc_u_arr:                 array<CascadeUniforms>;
@group(0) @binding(9) var<storage, read>       rc_tlas_nodes:            array<BVHNode>;
@group(0) @binding(10) var<storage, read>      rc_tlas_instance_indices: array<u32>;
@group(0) @binding(11) var<storage, read>      rc_tlas_blas_roots:       array<u32>;
@group(0) @binding(12) var<storage, read>      rc_tlas_w2l:              array<vec4f>;
@group(0) @binding(13) var<storage, read>      rc_tlas_l2w:              array<vec4f>;
@group(0) @binding(14) var<storage, read>      rc_emitters:              array<EmitterTri>;
// A7 (2026-06-10): point/spot analytic lights for probe-ray direct lighting.
// lightCount == 0 ⇒ loop is a no-op. Host binds the same DDGIProbeLights-style
// buffer (packRCLights) or a 1040-byte zero placeholder when no fixtures exist.
@group(0) @binding(15) var<storage, read>      rc_lights:                RCLightBuffer;
// RC material-backed emitter NEE (2026-06-16): optional material atlas views
// forwarded from the main pipeline. Placeholder meta has layer=-1, so helper
// calls fall back to scalar EmitterTri.Le when the caller omits these bindings.
@group(0) @binding(16) var                      rc_materialTextureAtlas: texture_2d_array<f32>;
@group(0) @binding(17) var                      rc_materialMapMeta:      texture_2d<f32>;
@group(0) @binding(18) var<storage, read>       rc_geom_normal:           array<vec4f>;
@group(0) @binding(19) var                      rc_geom_tangent:          texture_2d<f32>;
@group(0) @binding(20) var                      rc_geom_vertex_color:     texture_2d<f32>;

const RC_MATERIAL_MAP_META_TEXELS_PER_TRI: u32 = 62u;
const RC_MATERIAL_MAP_SLOT_BASE_COLOR: u32 = 0u;
const RC_MATERIAL_MAP_SLOT_ROUGHNESS: u32 = 1u;
const RC_MATERIAL_MAP_SLOT_METALLIC: u32 = 2u;
const RC_MATERIAL_MAP_SLOT_ALPHA: u32 = 4u;
const RC_MATERIAL_MAP_ALPHA_COVERAGE_TEXEL_OFFSET: u32 = 10u;
const RC_MATERIAL_MAP_EMISSIVE_TEXEL_OFFSET: u32 = 11u;
const RC_MATERIAL_MAP_NORMAL_TEXEL_OFFSET: u32 = 15u;
const RC_MATERIAL_MAP_NORMAL_SCALE_TEXEL_OFFSET: u32 = 17u;
const RC_MATERIAL_MAP_SPECULAR_TEXEL_OFFSET: u32 = 21u;
const RC_MATERIAL_MAP_CLEARCOAT_TEXEL_OFFSET: u32 = 22u;
const RC_MATERIAL_MAP_SHEEN_COLOR_TEXEL_OFFSET: u32 = 23u;
const RC_MATERIAL_MAP_SPECULAR_COLOR_TEXEL_OFFSET: u32 = 24u;
const RC_MATERIAL_MAP_SPECULAR_INTENSITY_TEXEL_OFFSET: u32 = 26u;
const RC_MATERIAL_MAP_CLEARCOAT_FACTOR_TEXEL_OFFSET: u32 = 28u;
const RC_MATERIAL_MAP_CLEARCOAT_ROUGHNESS_TEXEL_OFFSET: u32 = 30u;
const RC_MATERIAL_MAP_SHEEN_COLOR_MAP_TEXEL_OFFSET: u32 = 32u;
const RC_MATERIAL_MAP_SHEEN_ROUGHNESS_TEXEL_OFFSET: u32 = 34u;
const RC_MATERIAL_MAP_CLEARCOAT_NORMAL_TEXEL_OFFSET: u32 = 36u;
const RC_MATERIAL_MAP_CLEARCOAT_NORMAL_SCALE_TEXEL_OFFSET: u32 = 38u;
const RC_MATERIAL_MAP_ANISOTROPY_TEXEL_OFFSET: u32 = 39u;
const RC_MATERIAL_MAP_ANISOTROPY_SCALAR_TEXEL_OFFSET: u32 = 41u;
const RC_MATERIAL_MAP_IRIDESCENCE_TEXEL_OFFSET: u32 = 42u;
const RC_MATERIAL_MAP_IRIDESCENCE_THICKNESS_TEXEL_OFFSET: u32 = 44u;
const RC_MATERIAL_MAP_IRIDESCENCE_SCALAR_TEXEL_OFFSET: u32 = 46u;
const RC_MATERIAL_MAP_BUMP_TEXEL_OFFSET: u32 = 49u;
const RC_MATERIAL_MAP_BUMP_SCALE_TEXEL_OFFSET: u32 = 51u;
const RC_MATERIAL_MAP_FRONT_LAYER_NORMAL_TEXEL_OFFSET: u32 = 56u;
const RC_MATERIAL_MAP_FRONT_LAYER_NORMAL_SCALE_TEXEL_OFFSET: u32 = 58u;
const RC_MATERIAL_MAP_BACK_LAYER_NORMAL_TEXEL_OFFSET: u32 = 59u;
const RC_MATERIAL_MAP_BACK_LAYER_NORMAL_SCALE_TEXEL_OFFSET: u32 = 61u;
const RC_PI: f32 = 3.14159265;
const RC_INV_PI: f32 = 0.31830988618;

fn rcMaterialMetaCoord(texel: u32) -> vec2i {
  let dims = textureDimensions(rc_materialMapMeta);
  let w = max(dims.x, 1u);
  return vec2i(i32(texel % w), i32(texel / w));
}

fn rcMaterialMetaAvailable(triIndex: u32, metaOffset: u32) -> bool {
  let dims = textureDimensions(rc_materialMapMeta);
  let texel = triIndex * RC_MATERIAL_MAP_META_TEXELS_PER_TRI + metaOffset;
  return texel < dims.x * dims.y;
}

fn rcMaterialMetaLoadOrZero(triIndex: u32, metaOffset: u32) -> vec4f {
  let texel = triIndex * RC_MATERIAL_MAP_META_TEXELS_PER_TRI + metaOffset;
  if (!rcMaterialMetaAvailable(triIndex, metaOffset)) {
    return vec4f(0.0);
  }
  return textureLoad(rc_materialMapMeta, rcMaterialMetaCoord(texel), 0);
}

fn rcWrapMaterialUv1(v: f32, mode: u32) -> f32 {
  if (mode == 1u) {
    return clamp(v, 0.0, 1.0);
  }
  if (mode == 2u) {
    return 1.0 - abs(fract(v * 0.5) * 2.0 - 1.0);
  }
  return fract(v);
}

fn rcWrapMaterialUv(uv: vec2f, wrapPacked: u32) -> vec2f {
  let wrapS = wrapPacked & 0x3u;
  let wrapT = (wrapPacked >> 2u) & 0x3u;
  return vec2f(rcWrapMaterialUv1(uv.x, wrapS), rcWrapMaterialUv1(uv.y, wrapT));
}

fn rcPackedUvFromVec4(v: vec4f) -> vec2f {
  return unpack2x16float(bitcast<u32>(v.w));
}

struct RCHitMaterialUvs {
  valid: u32,
  uv0: vec2f,
  uv1: vec2f,
};

fn rcHitMaterialUvs(hit: IntersectionResult) -> RCHitMaterialUvs {
  var out: RCHitMaterialUvs;
  out.valid = 0u;
  out.uv0 = vec2f(0.0);
  out.uv1 = vec2f(0.0);

  let i0 = hit.indices.x;
  let i1 = hit.indices.y;
  let i2 = hit.indices.z;
  if (
    hit.indices.w >= arrayLength(&rc_geom_index) ||
    i0 >= arrayLength(&rc_geom_position) || i1 >= arrayLength(&rc_geom_position) || i2 >= arrayLength(&rc_geom_position) ||
    i0 >= arrayLength(&rc_geom_normal) || i1 >= arrayLength(&rc_geom_normal) || i2 >= arrayLength(&rc_geom_normal)
  ) {
    return out;
  }

  out.valid = 1u;
  out.uv0 =
    hit.barycoord.x * rcPackedUvFromVec4(rc_geom_position[i0]) +
    hit.barycoord.y * rcPackedUvFromVec4(rc_geom_position[i1]) +
    hit.barycoord.z * rcPackedUvFromVec4(rc_geom_position[i2]);
  out.uv1 =
    hit.barycoord.x * rcPackedUvFromVec4(rc_geom_normal[i0]) +
    hit.barycoord.y * rcPackedUvFromVec4(rc_geom_normal[i1]) +
    hit.barycoord.z * rcPackedUvFromVec4(rc_geom_normal[i2]);
  return out;
}

fn rcEmitterSubdivWeightAt(i: u32, j: u32, level: u32) -> vec3f {
  let invLevel = 1.0 / f32(max(level, 1u));
  let u = f32(i) * invLevel;
  let v = f32(j) * invLevel;
  return vec3f(1.0 - u - v, u, v);
}

fn rcEmitterParentBarycentricFromLocal(localBary: vec3f, levelF: f32, ordinalF: f32) -> vec3f {
  let level = min(16u, max(1u, u32(round(max(levelF, 1.0)))));
  if (level <= 1u) {
    return localBary;
  }

  let ordinal = u32(round(max(ordinalF, 0.0)));
  var cursor = 0u;
  for (var i = 0u; i < level; i = i + 1u) {
    for (var j = 0u; j < level - i; j = j + 1u) {
      let a = rcEmitterSubdivWeightAt(i, j, level);
      let b = rcEmitterSubdivWeightAt(i + 1u, j, level);
      let c = rcEmitterSubdivWeightAt(i, j + 1u, level);
      if (cursor == ordinal) {
        return localBary.x * a + localBary.y * b + localBary.z * c;
      }
      cursor = cursor + 1u;

      if (i + j < level - 1u) {
        let d = rcEmitterSubdivWeightAt(i + 1u, j + 1u, level);
        if (cursor == ordinal) {
          return localBary.x * b + localBary.y * d + localBary.z * c;
        }
        cursor = cursor + 1u;
      }
    }
  }

  return localBary;
}

fn rcSampleMaterialAtlasRawAtOffsetDelta(
  triIndex: u32,
  metaOffset: u32,
  uv0: vec2f,
  uv1: vec2f,
  transformedDelta: vec2f,
) -> vec4f {
  let metaDims = textureDimensions(rc_materialMapMeta);
  let metaTexel = triIndex * RC_MATERIAL_MAP_META_TEXELS_PER_TRI + metaOffset;
  if (metaTexel + 1u >= metaDims.x * metaDims.y) {
    return vec4f(-1.0);
  }
  let meta0 = textureLoad(rc_materialMapMeta, rcMaterialMetaCoord(metaTexel), 0);
  let layer = i32(meta0.x);
  if (layer < 0 || u32(layer) >= textureNumLayers(rc_materialTextureAtlas)) {
    return vec4f(-1.0);
  }
  let wrapPacked = u32(max(meta0.y, 0.0) + 0.5);
  let texCoord = (wrapPacked >> 4u) & 0x3u;
  let uv = select(uv0, uv1, texCoord == 1u);
  let meta1 = textureLoad(rc_materialMapMeta, rcMaterialMetaCoord(metaTexel + 1u), 0);
  let scaled = uv * meta1.xy;
  let transformed = vec2f(
    scaled.x * meta1.z - scaled.y * meta1.w,
    scaled.x * meta1.w + scaled.y * meta1.z,
  ) + meta0.zw + transformedDelta;
  let wrapped = rcWrapMaterialUv(transformed, wrapPacked);
  let dims = textureDimensions(rc_materialTextureAtlas);
  let texel = vec2i(
    i32(min(u32(floor(wrapped.x * f32(dims.x))), dims.x - 1u)),
    i32(min(u32(floor(wrapped.y * f32(dims.y))), dims.y - 1u)),
  );
  return textureLoad(rc_materialTextureAtlas, texel, layer, 0);
}

fn rcSampleMaterialAtlasRawAtOffset(triIndex: u32, metaOffset: u32, uv0: vec2f, uv1: vec2f) -> vec4f {
  return rcSampleMaterialAtlasRawAtOffsetDelta(triIndex, metaOffset, uv0, uv1, vec2f(0.0));
}

fn rcSampleMaterialAtlasRaw(triIndex: u32, slot: u32, uv0: vec2f, uv1: vec2f) -> vec4f {
  return rcSampleMaterialAtlasRawAtOffset(triIndex, slot * 2u, uv0, uv1);
}

fn rcSampleSurfaceEmissiveMap(hit: IntersectionResult, scalarEmission: vec3f) -> vec3f {
  let uvs = rcHitMaterialUvs(hit);
  if (uvs.valid == 0u) {
    return scalarEmission;
  }
  let texel = rcSampleMaterialAtlasRawAtOffset(
    hit.indices.w,
    RC_MATERIAL_MAP_EMISSIVE_TEXEL_OFFSET,
    uvs.uv0,
    uvs.uv1,
  );
  if (texel.x < 0.0) {
    return scalarEmission;
  }
  return scalarEmission * texel.rgb;
}

fn rcMaterialMapChannel(v: vec4f, channel: u32) -> f32 {
  if (channel == 1u) { return v.g; }
  if (channel == 2u) { return v.b; }
  if (channel == 3u) { return v.a; }
  return v.r;
}

fn rcSampleMaterialScalarMap(
  triIndex: u32,
  slot: u32,
  channel: u32,
  uv0: vec2f,
  uv1: vec2f,
  fallback: f32,
) -> f32 {
  let texel = rcSampleMaterialAtlasRaw(triIndex, slot, uv0, uv1);
  if (texel.x < 0.0) {
    return fallback;
  }
  return clamp(fallback * rcMaterialMapChannel(texel, channel), 0.0, 1.0);
}

fn rcSampleSpecularControls(triIndex: u32, uv0: vec2f, uv1: vec2f) -> vec4f {
  var color = vec3f(1.0);
  var intensity = 1.0;
  if (rcMaterialMetaAvailable(triIndex, RC_MATERIAL_MAP_SPECULAR_TEXEL_OFFSET)) {
    let spec = rcMaterialMetaLoadOrZero(triIndex, RC_MATERIAL_MAP_SPECULAR_TEXEL_OFFSET);
    color = clamp(spec.rgb, vec3f(0.0), vec3f(1.0));
    intensity = clamp(spec.a, 0.0, 1.0);
  }

  let colorMap = rcSampleMaterialAtlasRawAtOffset(triIndex, RC_MATERIAL_MAP_SPECULAR_COLOR_TEXEL_OFFSET, uv0, uv1);
  if (colorMap.x >= 0.0) {
    color = clamp(color * colorMap.rgb, vec3f(0.0), vec3f(1.0));
  }
  let intensityMap = rcSampleMaterialAtlasRawAtOffset(triIndex, RC_MATERIAL_MAP_SPECULAR_INTENSITY_TEXEL_OFFSET, uv0, uv1);
  if (intensityMap.x >= 0.0) {
    intensity = clamp(intensity * intensityMap.a, 0.0, 1.0);
  }
  return vec4f(color, intensity);
}

fn rcSampleClearcoatControls(triIndex: u32, uv0: vec2f, uv1: vec2f) -> vec2f {
  let cc = rcMaterialMetaLoadOrZero(triIndex, RC_MATERIAL_MAP_CLEARCOAT_TEXEL_OFFSET);
  var factor = clamp(cc.x, 0.0, 1.0);
  var roughness = clamp(cc.y, 0.0, 1.0);

  let clearcoatMap = rcSampleMaterialAtlasRawAtOffset(triIndex, RC_MATERIAL_MAP_CLEARCOAT_FACTOR_TEXEL_OFFSET, uv0, uv1);
  if (clearcoatMap.x >= 0.0) {
    factor = clamp(factor * clearcoatMap.r, 0.0, 1.0);
  }
  let roughnessMap = rcSampleMaterialAtlasRawAtOffset(triIndex, RC_MATERIAL_MAP_CLEARCOAT_ROUGHNESS_TEXEL_OFFSET, uv0, uv1);
  if (roughnessMap.x >= 0.0) {
    roughness = clamp(roughness * roughnessMap.g, 0.0, 1.0);
  }
  return vec2f(factor, roughness);
}

fn rcSampleSheenControls(triIndex: u32, uv0: vec2f, uv1: vec2f) -> vec4f {
  let scalars = rcMaterialMetaLoadOrZero(triIndex, RC_MATERIAL_MAP_CLEARCOAT_TEXEL_OFFSET);
  let colorMeta = rcMaterialMetaLoadOrZero(triIndex, RC_MATERIAL_MAP_SHEEN_COLOR_TEXEL_OFFSET);
  var sheenColor = clamp(colorMeta.rgb, vec3f(0.0), vec3f(1.0));
  var sheen = clamp(scalars.z, 0.0, 1.0);

  let colorMap = rcSampleMaterialAtlasRawAtOffset(triIndex, RC_MATERIAL_MAP_SHEEN_COLOR_MAP_TEXEL_OFFSET, uv0, uv1);
  if (colorMap.x >= 0.0) {
    sheenColor = clamp(sheenColor * colorMap.rgb, vec3f(0.0), vec3f(1.0));
  }
  return vec4f(sheenColor, sheen);
}

fn rcSampleSheenRoughness(triIndex: u32, uv0: vec2f, uv1: vec2f) -> f32 {
  let scalars = rcMaterialMetaLoadOrZero(triIndex, RC_MATERIAL_MAP_CLEARCOAT_TEXEL_OFFSET);
  var roughness = clamp(scalars.w, 0.0, 1.0);
  let roughnessMap = rcSampleMaterialAtlasRawAtOffset(triIndex, RC_MATERIAL_MAP_SHEEN_ROUGHNESS_TEXEL_OFFSET, uv0, uv1);
  if (roughnessMap.x >= 0.0) {
    roughness = clamp(roughness * roughnessMap.a, 0.0, 1.0);
  }
  return roughness;
}

fn rcSampleAnisotropyControls(triIndex: u32, uv0: vec2f, uv1: vec2f) -> vec2f {
  let scalars = rcMaterialMetaLoadOrZero(triIndex, RC_MATERIAL_MAP_ANISOTROPY_SCALAR_TEXEL_OFFSET);
  var strength = clamp(scalars.x, 0.0, 1.0);
  var rotation = scalars.y;

  let anisoMap = rcSampleMaterialAtlasRawAtOffset(triIndex, RC_MATERIAL_MAP_ANISOTROPY_TEXEL_OFFSET, uv0, uv1);
  if (anisoMap.x >= 0.0) {
    strength = clamp(strength * anisoMap.b, 0.0, 1.0);
    let direction = anisoMap.rg * 2.0 - vec2f(1.0);
    if (dot(direction, direction) > 1e-6) {
      rotation += atan2(direction.y, direction.x);
    }
  }
  return vec2f(strength, rotation);
}

fn rcSampleIridescenceControls(triIndex: u32, uv0: vec2f, uv1: vec2f) -> vec4f {
  let scalars = rcMaterialMetaLoadOrZero(triIndex, RC_MATERIAL_MAP_IRIDESCENCE_SCALAR_TEXEL_OFFSET);
  var factor = clamp(scalars.x, 0.0, 1.0);
  let ior = max(1.0, scalars.y);
  var thicknessMin = max(0.0, scalars.z);
  var thicknessMax = max(0.0, scalars.w);

  let iridescenceMap = rcSampleMaterialAtlasRawAtOffset(triIndex, RC_MATERIAL_MAP_IRIDESCENCE_TEXEL_OFFSET, uv0, uv1);
  if (iridescenceMap.x >= 0.0) {
    factor = clamp(factor * iridescenceMap.r, 0.0, 1.0);
  }
  let thicknessMap = rcSampleMaterialAtlasRawAtOffset(triIndex, RC_MATERIAL_MAP_IRIDESCENCE_THICKNESS_TEXEL_OFFSET, uv0, uv1);
  if (thicknessMap.x >= 0.0) {
    let thickness = mix(thicknessMin, thicknessMax, clamp(thicknessMap.g, 0.0, 1.0));
    thicknessMin = thickness;
    thicknessMax = thickness;
    if (thickness <= 0.0) {
      factor = 0.0;
    }
  }
  return vec4f(factor, ior, thicknessMin, thicknessMax);
}

fn rcSmoothNormalForHit(hit: IntersectionResult, fallbackNormal: vec3f) -> vec3f {
  let i0 = hit.indices.x;
  let i1 = hit.indices.y;
  let i2 = hit.indices.z;
  if (i0 >= arrayLength(&rc_geom_normal) || i1 >= arrayLength(&rc_geom_normal) || i2 >= arrayLength(&rc_geom_normal)) {
    return fallbackNormal;
  }
  let n = normalize(
    hit.barycoord.x * rc_geom_normal[i0].xyz +
    hit.barycoord.y * rc_geom_normal[i1].xyz +
    hit.barycoord.z * rc_geom_normal[i2].xyz
  );
  return select(-n, n, dot(n, fallbackNormal) >= 0.0);
}

struct RCMaterialTangentFrame {
  tangent: vec3f,
  bitangent: vec3f,
}

fn rcBvhTangentTexel(vertexIndex: u32) -> vec4f {
  let dims = textureDimensions(rc_geom_tangent);
  let width = u32(dims.x);
  let height = u32(dims.y);
  if (width == 0u || height == 0u) {
    return vec4f(0.0);
  }
  let y = vertexIndex / width;
  if (y >= height) {
    return vec4f(0.0);
  }
  return textureLoad(rc_geom_tangent, vec2i(i32(vertexIndex % width), i32(y)), 0);
}

fn rcBvhVertexColorTexel(vertexIndex: u32) -> vec4f {
  let dims = textureDimensions(rc_geom_vertex_color);
  let width = u32(dims.x);
  let height = u32(dims.y);
  if (width == 0u || height == 0u) {
    return vec4f(1.0);
  }
  let y = vertexIndex / width;
  if (y >= height) {
    return vec4f(1.0);
  }
  return clamp(textureLoad(rc_geom_vertex_color, vec2i(i32(vertexIndex % width), i32(y)), 0), vec4f(0.0), vec4f(1.0));
}

fn rcSampleVertexColorForHit(hit: IntersectionResult) -> vec4f {
  let ca = rcBvhVertexColorTexel(hit.indices.x);
  let cb = rcBvhVertexColorTexel(hit.indices.y);
  let cc = rcBvhVertexColorTexel(hit.indices.z);
  return clamp(
    ca * hit.barycoord.x +
    cb * hit.barycoord.y +
    cc * hit.barycoord.z,
    vec4f(0.0),
    vec4f(1.0)
  );
}

fn rcTransformDirectionCols(l2w0: vec4f, l2w1: vec4f, l2w2: vec4f, v: vec3f) -> vec3f {
  return l2w0.xyz * v.x + l2w1.xyz * v.y + l2w2.xyz * v.z;
}

fn rcTangentHandednessForLocalToWorld(l2w0: vec4f, l2w1: vec4f, l2w2: vec4f) -> f32 {
  let det = dot(l2w0.xyz, cross(l2w1.xyz, l2w2.xyz));
  return select(-1.0, 1.0, det >= 0.0);
}

fn rcPreferAuthoredTangentFrameForHit(
  hit: IntersectionResult,
  frameNormal: vec3f,
  fallbackTangent: vec3f,
  fallbackBitangent: vec3f,
) -> RCMaterialTangentFrame {
  var tangent = fallbackTangent;
  var bitangent = fallbackBitangent;

  let ta = rcBvhTangentTexel(hit.indices.x);
  let tb = rcBvhTangentTexel(hit.indices.y);
  let tc = rcBvhTangentTexel(hit.indices.z);
  var authoredTangent =
    hit.barycoord.x * ta.xyz +
    hit.barycoord.y * tb.xyz +
    hit.barycoord.z * tc.xyz;
  var authoredHandedness =
    hit.barycoord.x * ta.w +
    hit.barycoord.y * tb.w +
    hit.barycoord.z * tc.w;

  if (length(authoredTangent) > 1e-8 && abs(authoredHandedness) > 0.5) {
    let isTlas = rc_u_arr[0].bvhMode == 1u;
    let tBase = hit.instanceIndex * 4u;
    let tOk = isTlas && tBase + 2u < arrayLength(&rc_tlas_l2w);
    if (tOk) {
      authoredTangent = rcTransformDirectionCols(
        rc_tlas_l2w[tBase],
        rc_tlas_l2w[tBase + 1u],
        rc_tlas_l2w[tBase + 2u],
        authoredTangent,
      );
      authoredHandedness = authoredHandedness * rcTangentHandednessForLocalToWorld(
        rc_tlas_l2w[tBase],
        rc_tlas_l2w[tBase + 1u],
        rc_tlas_l2w[tBase + 2u],
      );
    }

    authoredTangent = authoredTangent - frameNormal * dot(frameNormal, authoredTangent);
    let tLen2 = dot(authoredTangent, authoredTangent);
    if (tLen2 > 1e-8) {
      tangent = authoredTangent * inverseSqrt(tLen2);
      bitangent = normalize(cross(frameNormal, tangent)) * select(-1.0, 1.0, authoredHandedness >= 0.0);
    }
  }

  return RCMaterialTangentFrame(tangent, bitangent);
}

fn rcFallbackBitangentForNormal(n: vec3f, t: vec3f) -> vec3f {
  let b = cross(n, t);
  let len2 = dot(b, b);
  if (len2 < 1e-8) {
    let up = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(n.y) > 0.95);
    return normalize(cross(n, up));
  }
  return b * inverseSqrt(len2);
}

fn rcMaterialTangentFrameForHit(
  hit: IntersectionResult,
  frameNormal: vec3f,
  mapOffset: u32,
) -> RCMaterialTangentFrame {
  let triIndex = hit.indices.w;
  let metaTexel = triIndex * RC_MATERIAL_MAP_META_TEXELS_PER_TRI + mapOffset;
  let meta0 = textureLoad(rc_materialMapMeta, rcMaterialMetaCoord(metaTexel), 0);
  let flags = u32(max(meta0.y, 0.0) + 0.5);
  let useUv1 = ((flags >> 4u) & 0x3u) == 1u;

  let p0 = rc_geom_position[hit.indices.x];
  let p1 = rc_geom_position[hit.indices.y];
  let p2 = rc_geom_position[hit.indices.z];
  let n0 = rc_geom_normal[hit.indices.x];
  let n1 = rc_geom_normal[hit.indices.y];
  let n2 = rc_geom_normal[hit.indices.z];
  let uv0a = rcPackedUvFromVec4(p0);
  let uv0b = rcPackedUvFromVec4(p1);
  let uv0c = rcPackedUvFromVec4(p2);
  let uv1a = rcPackedUvFromVec4(n0);
  let uv1b = rcPackedUvFromVec4(n1);
  let uv1c = rcPackedUvFromVec4(n2);
  let ta = select(uv0a, uv1a, useUv1);
  let tb = select(uv0b, uv1b, useUv1);
  let tc = select(uv0c, uv1c, useUv1);

  let dp1 = p1.xyz - p0.xyz;
  let dp2 = p2.xyz - p0.xyz;
  let duv1 = tb - ta;
  let duv2 = tc - ta;
  let det = duv1.x * duv2.y - duv1.y * duv2.x;
  var tangent = dp1;
  var bitangent = rcFallbackBitangentForNormal(frameNormal, tangent);
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
    bitangent = rcFallbackBitangentForNormal(frameNormal, tangent);
  } else {
    bitangent = bitangent * inverseSqrt(bLen2);
  }

  return rcPreferAuthoredTangentFrameForHit(hit, frameNormal, tangent, bitangent);
}

fn rcApplyNormalMapAtOffsetForHit(
  hit: IntersectionResult,
  frameNormal: vec3f,
  fallbackNormal: vec3f,
  normalMapOffset: u32,
  normalScaleOffset: u32,
) -> vec3f {
  let triIndex = hit.indices.w;
  let metaTexel = triIndex * RC_MATERIAL_MAP_META_TEXELS_PER_TRI + normalMapOffset;
  let metaDims = textureDimensions(rc_materialMapMeta);
  if (metaTexel + 1u >= metaDims.x * metaDims.y) {
    return fallbackNormal;
  }
  let meta0 = textureLoad(rc_materialMapMeta, rcMaterialMetaCoord(metaTexel), 0);
  if (i32(meta0.x) < 0) {
    return fallbackNormal;
  }

  let uvs = rcHitMaterialUvs(hit);
  if (uvs.valid == 0u) {
    return fallbackNormal;
  }
  let texelColor = rcSampleMaterialAtlasRawAtOffset(triIndex, normalMapOffset, uvs.uv0, uvs.uv1);
  if (texelColor.x < 0.0) {
    return fallbackNormal;
  }

  let scaleMeta = textureLoad(
    rc_materialMapMeta,
    rcMaterialMetaCoord(triIndex * RC_MATERIAL_MAP_META_TEXELS_PER_TRI + normalScaleOffset),
    0,
  );
  let normalScale = max(scaleMeta.x, 0.0);
  let tangentSample = normalize(vec3f(
    (texelColor.r * 2.0 - 1.0) * normalScale,
    (texelColor.g * 2.0 - 1.0) * normalScale,
    texelColor.b * 2.0 - 1.0,
  ));

  let frame = rcMaterialTangentFrameForHit(hit, frameNormal, normalMapOffset);
  let perturbed = normalize(frame.tangent * tangentSample.x + frame.bitangent * tangentSample.y + frameNormal * tangentSample.z);
  return select(-perturbed, perturbed, dot(perturbed, frameNormal) >= 0.0);
}

fn rcApplyFaceLayerNormalMapForHit(hit: IntersectionResult, frameNormal: vec3f, fallbackNormal: vec3f) -> vec3f {
  let isFrontFace = hit.side >= 0.0;
  let normalMapOffset = select(
    RC_MATERIAL_MAP_BACK_LAYER_NORMAL_TEXEL_OFFSET,
    RC_MATERIAL_MAP_FRONT_LAYER_NORMAL_TEXEL_OFFSET,
    isFrontFace,
  );
  let normalScaleOffset = select(
    RC_MATERIAL_MAP_BACK_LAYER_NORMAL_SCALE_TEXEL_OFFSET,
    RC_MATERIAL_MAP_FRONT_LAYER_NORMAL_SCALE_TEXEL_OFFSET,
    isFrontFace,
  );
  return rcApplyNormalMapAtOffsetForHit(hit, frameNormal, fallbackNormal, normalMapOffset, normalScaleOffset);
}

fn rcApplyNormalMapForHit(hit: IntersectionResult, baseNormal: vec3f) -> vec3f {
  let baseMapped = rcApplyNormalMapAtOffsetForHit(
    hit,
    baseNormal,
    baseNormal,
    RC_MATERIAL_MAP_NORMAL_TEXEL_OFFSET,
    RC_MATERIAL_MAP_NORMAL_SCALE_TEXEL_OFFSET,
  );
  return rcApplyFaceLayerNormalMapForHit(hit, baseNormal, baseMapped);
}

fn rcApplyClearcoatNormalMapForHit(hit: IntersectionResult, frameNormal: vec3f, fallbackNormal: vec3f) -> vec3f {
  return rcApplyNormalMapAtOffsetForHit(
    hit,
    frameNormal,
    fallbackNormal,
    RC_MATERIAL_MAP_CLEARCOAT_NORMAL_TEXEL_OFFSET,
    RC_MATERIAL_MAP_CLEARCOAT_NORMAL_SCALE_TEXEL_OFFSET,
  );
}

fn rcApplyBumpMapForHit(hit: IntersectionResult, shadingNormal: vec3f) -> vec3f {
  let triIndex = hit.indices.w;
  let metaTexel = triIndex * RC_MATERIAL_MAP_META_TEXELS_PER_TRI + RC_MATERIAL_MAP_BUMP_TEXEL_OFFSET;
  let metaDims = textureDimensions(rc_materialMapMeta);
  if (metaTexel + 1u >= metaDims.x * metaDims.y) {
    return shadingNormal;
  }
  let meta0 = textureLoad(rc_materialMapMeta, rcMaterialMetaCoord(metaTexel), 0);
  if (i32(meta0.x) < 0) {
    return shadingNormal;
  }

  let scaleMeta = textureLoad(
    rc_materialMapMeta,
    rcMaterialMetaCoord(triIndex * RC_MATERIAL_MAP_META_TEXELS_PER_TRI + RC_MATERIAL_MAP_BUMP_SCALE_TEXEL_OFFSET),
    0,
  );
  let bumpScale = scaleMeta.x;
  if (abs(bumpScale) < 1e-8) {
    return shadingNormal;
  }

  let uvs = rcHitMaterialUvs(hit);
  if (uvs.valid == 0u) {
    return shadingNormal;
  }
  let hC = rcSampleMaterialAtlasRawAtOffset(triIndex, RC_MATERIAL_MAP_BUMP_TEXEL_OFFSET, uvs.uv0, uvs.uv1);
  if (hC.x < 0.0) {
    return shadingNormal;
  }

  let atlasDims = textureDimensions(rc_materialTextureAtlas);
  let atlasTexelStep = vec2f(
    1.0 / f32(max(atlasDims.x, 1u)),
    1.0 / f32(max(atlasDims.y, 1u)),
  );
  let bumpTexelStep = vec2f(
    1.0 / max(scaleMeta.y, 1.0),
    1.0 / max(scaleMeta.z, 1.0),
  );
  let texelStep = select(atlasTexelStep, bumpTexelStep, scaleMeta.y > 0.0 && scaleMeta.z > 0.0);
  let hU = rcSampleMaterialAtlasRawAtOffsetDelta(
    triIndex,
    RC_MATERIAL_MAP_BUMP_TEXEL_OFFSET,
    uvs.uv0,
    uvs.uv1,
    vec2f(texelStep.x, 0.0),
  ).r;
  let hV = rcSampleMaterialAtlasRawAtOffsetDelta(
    triIndex,
    RC_MATERIAL_MAP_BUMP_TEXEL_OFFSET,
    uvs.uv0,
    uvs.uv1,
    vec2f(0.0, texelStep.y),
  ).r;
  let dhdu = (hU - hC.r) / texelStep.x;
  let dhdv = (hV - hC.r) / texelStep.y;
  let frame = rcMaterialTangentFrameForHit(hit, shadingNormal, RC_MATERIAL_MAP_BUMP_TEXEL_OFFSET);
  let perturbed = shadingNormal - bumpScale * (dhdu * frame.tangent + dhdv * frame.bitangent);
  let plen = length(perturbed);
  let n = select(shadingNormal, perturbed / plen, plen > 1e-6);
  return select(-n, n, dot(n, shadingNormal) >= 0.0);
}

struct RCProbeHitMaterial {
  albedo: vec3f,
  roughness: f32,
  metalness: f32,
  specular: vec4f,
  clearcoat: vec2f,
  clearcoatNormal: vec3f,
  sheen: vec4f,
  sheenRoughness: f32,
  anisotropy: vec2f,
  anisotropyTangent: vec3f,
  anisotropyBitangent: vec3f,
  iridescence: vec4f,
}

fn rcSampleProbeHitMaterial(
  hit: IntersectionResult,
  scalarBaseColor: vec3f,
  scalarRoughness: f32,
  scalarMetalness: f32,
  frameNormal: vec3f,
  shadingNormal: vec3f,
) -> RCProbeHitMaterial {
  var out: RCProbeHitMaterial;
  out.albedo = scalarBaseColor;
  out.roughness = scalarRoughness;
  out.metalness = scalarMetalness;
  out.specular = vec4f(1.0);
  out.clearcoat = vec2f(0.0);
  out.clearcoatNormal = shadingNormal;
  out.sheen = vec4f(0.0);
  out.sheenRoughness = 0.0;
  out.anisotropy = vec2f(0.0);
  let defaultAnisotropyFrame = rcMaterialTangentFrameForHit(hit, shadingNormal, RC_MATERIAL_MAP_ANISOTROPY_TEXEL_OFFSET);
  out.anisotropyTangent = defaultAnisotropyFrame.tangent;
  out.anisotropyBitangent = defaultAnisotropyFrame.bitangent;
  out.iridescence = vec4f(0.0, 1.0, 0.0, 0.0);
  out.clearcoatNormal = rcApplyClearcoatNormalMapForHit(hit, frameNormal, shadingNormal);

  let uvs = rcHitMaterialUvs(hit);
  if (uvs.valid == 0u) {
    return out;
  }

  let baseColorTexel = rcSampleMaterialAtlasRaw(
    hit.indices.w,
    RC_MATERIAL_MAP_SLOT_BASE_COLOR,
    uvs.uv0,
    uvs.uv1,
  );
  if (baseColorTexel.x >= 0.0) {
    out.albedo = scalarBaseColor * baseColorTexel.rgb;
  }
  out.roughness = rcSampleMaterialScalarMap(
    hit.indices.w,
    RC_MATERIAL_MAP_SLOT_ROUGHNESS,
    1u,
    uvs.uv0,
    uvs.uv1,
    scalarRoughness,
  );
  out.metalness = rcSampleMaterialScalarMap(
    hit.indices.w,
    RC_MATERIAL_MAP_SLOT_METALLIC,
    2u,
    uvs.uv0,
    uvs.uv1,
    scalarMetalness,
  );
  out.specular = rcSampleSpecularControls(hit.indices.w, uvs.uv0, uvs.uv1);
  out.clearcoat = rcSampleClearcoatControls(hit.indices.w, uvs.uv0, uvs.uv1);
  out.sheen = rcSampleSheenControls(hit.indices.w, uvs.uv0, uvs.uv1);
  out.sheenRoughness = rcSampleSheenRoughness(hit.indices.w, uvs.uv0, uvs.uv1);
  out.anisotropy = rcSampleAnisotropyControls(hit.indices.w, uvs.uv0, uvs.uv1);
  let anisotropyFrame = rcMaterialTangentFrameForHit(hit, shadingNormal, RC_MATERIAL_MAP_ANISOTROPY_TEXEL_OFFSET);
  out.anisotropyTangent = anisotropyFrame.tangent;
  out.anisotropyBitangent = anisotropyFrame.bitangent;
  out.iridescence = rcSampleIridescenceControls(hit.indices.w, uvs.uv0, uvs.uv1);
  return out;
}

fn rcBaseMaterialF0(mat: RCProbeHitMaterial) -> vec3f {
  let dielectricF0 = vec3f(0.04) * mat.specular.rgb * mat.specular.a;
  return mix(dielectricF0, mat.albedo, clamp(mat.metalness, 0.0, 1.0));
}

fn rcFresnelSchlick(cosTheta: f32, f0: vec3f) -> vec3f {
  let f = pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
  return f0 + (vec3f(1.0) - f0) * f;
}

fn rcIridescenceModifiedF0(baseF0: vec3f, iridescence: vec4f, vDotH: f32) -> vec3f {
  let factor = clamp(iridescence.x, 0.0, 1.0);
  if (factor <= 1e-4) {
    return baseF0;
  }
  let thickness = max(0.0, (iridescence.z + iridescence.w) * 0.5);
  let iorShift = clamp(iridescence.y - 1.0, 0.0, 2.0) * 0.12;
  let phase = thickness * 0.012 + (1.0 - clamp(vDotH, 0.0, 1.0)) * RC_PI;
  let filmTint = clamp(
    0.5 + 0.5 * cos(vec3f(phase, phase + 2.0943951, phase + 4.1887902)) + vec3f(iorShift),
    vec3f(0.0),
    vec3f(1.0),
  );
  let filmF0 = mix(vec3f(0.04), filmTint, clamp(thickness / 1200.0, 0.0, 1.0));
  return clamp(mix(baseF0, filmF0, factor), vec3f(0.0), vec3f(1.0));
}

fn rcDistributionGGX(nDotH: f32, rough: f32) -> f32 {
  let a = rough * rough;
  let a2 = a * a;
  let d = nDotH * nDotH * (a2 - 1.0) + 1.0;
  return a2 / max(RC_PI * d * d, 1e-6);
}

fn rcGeometrySchlickGGX(nDotV: f32, rough: f32) -> f32 {
  let r = rough + 1.0;
  let k = r * r * 0.125;
  return nDotV / max(nDotV * (1.0 - k) + k, 1e-6);
}

fn rcGeometrySmith(nDotV: f32, nDotL: f32, rough: f32) -> f32 {
  return rcGeometrySchlickGGX(nDotV, rough) * rcGeometrySchlickGGX(nDotL, rough);
}

fn rcRotateTangentFrame(t: vec3f, b: vec3f, rotation: f32) -> mat2x3f {
  let c = cos(rotation);
  let s = sin(rotation);
  let rt = safe_normalize(t * c + b * s);
  let rb = safe_normalize(-t * s + b * c);
  return mat2x3f(rt, rb);
}

fn rcAnisotropyAxes(rough: f32, anisotropy: f32) -> vec2f {
  let a = max(0.01, rough * rough);
  let aspect = sqrt(max(0.1, 1.0 - 0.9 * clamp(anisotropy, 0.0, 1.0)));
  return vec2f(max(0.001, a / aspect), max(0.001, a * aspect));
}

fn rcDistributionGGXAnisotropic(n: vec3f, t: vec3f, b: vec3f, h: vec3f, ax: f32, ay: f32) -> f32 {
  let nDotH = max(0.0, dot(n, h));
  if (nDotH <= 1e-6) { return 0.0; }
  let tx = dot(t, h) / ax;
  let by = dot(b, h) / ay;
  let d = tx * tx + by * by + nDotH * nDotH;
  return 1.0 / max(RC_PI * ax * ay * d * d, 1e-6);
}

fn rcSmithG1Anisotropic(n: vec3f, t: vec3f, b: vec3f, v: vec3f, ax: f32, ay: f32) -> f32 {
  let nDotV = max(0.0, dot(n, v));
  if (nDotV <= 1e-6) { return 0.0; }
  let tx = dot(t, v) * ax;
  let by = dot(b, v) * ay;
  let lambda = (-1.0 + sqrt(1.0 + (tx * tx + by * by) / max(nDotV * nDotV, 1e-6))) * 0.5;
  return 1.0 / (1.0 + lambda);
}

fn rcEvalClearcoatLobe(clearcoat: vec2f, clearcoatNormal: vec3f, wo: vec3f, wi: vec3f) -> vec3f {
  let cc = clamp(clearcoat.x, 0.0, 1.0);
  if (cc <= 1e-4) { return vec3f(0.0); }
  let h = safe_normalize(wo + wi);
  let nDotL = max(0.0, dot(clearcoatNormal, wi));
  let nDotV = max(1e-4, dot(clearcoatNormal, wo));
  let nDotH = max(0.0, dot(clearcoatNormal, h));
  let vDotH = max(0.0, dot(wo, h));
  if (nDotL <= 1e-6 || nDotV <= 1e-6) { return vec3f(0.0); }
  let rough = clamp(clearcoat.y, 0.01, 1.0);
  let D = rcDistributionGGX(nDotH, rough);
  let G = rcGeometrySmith(nDotV, nDotL, rough);
  let F = rcFresnelSchlick(vDotH, vec3f(0.04));
  return cc * (D * G * F) / max(4.0 * nDotV * nDotL, 1e-6) * nDotL;
}

fn rcCharlieD(nDotH: f32, alpha: f32) -> f32 {
  let invAlpha = 1.0 / max(alpha, 1e-4);
  let sinThetaH = sqrt(max(0.0, 1.0 - nDotH * nDotH));
  return (2.0 + invAlpha) * pow(sinThetaH, invAlpha) / (2.0 * RC_PI);
}

fn rcSheenVisibility(nDotL: f32, nDotV: f32) -> f32 {
  return 1.0 / max(4.0 * (nDotL + nDotV - nDotL * nDotV), 1e-6);
}

fn rcEvalSheenLobe(sheen: vec4f, sheenRoughness: f32, n: vec3f, wo: vec3f, wi: vec3f) -> vec3f {
  let sh = clamp(sheen.a, 0.0, 1.0);
  if (sh <= 1e-4) { return vec3f(0.0); }
  let nDotL = max(0.0, dot(n, wi));
  let nDotV = max(0.0, dot(n, wo));
  if (nDotL <= 1e-6 || nDotV <= 1e-6) { return vec3f(0.0); }
  let h = safe_normalize(wo + wi);
  let nDotH = max(0.0, dot(n, h));
  let alpha = max(clamp(sheenRoughness, 0.0, 1.0) * clamp(sheenRoughness, 0.0, 1.0), 1e-3);
  return sh * clamp(sheen.rgb, vec3f(0.0), vec3f(1.0)) * rcCharlieD(nDotH, alpha) * rcSheenVisibility(nDotL, nDotV) * nDotL;
}

fn rcEvaluateProbeDirectResponse(mat: RCProbeHitMaterial, n: vec3f, wo: vec3f, wi: vec3f) -> vec3f {
  let nDotL = max(0.0, dot(n, wi));
  if (nDotL <= 1e-6) {
    return vec3f(0.0);
  }
  let v = safe_normalize(wo);
  let l = safe_normalize(wi);
  let h = safe_normalize(v + l);
  let nDotV = max(0.0, dot(n, v));
  let nDotH = max(0.0, dot(n, h));
  let vDotH = max(0.0, dot(v, h));
  let rough = clamp(mat.roughness, 0.04, 1.0);
  var D: f32;
  var G: f32;
  let aniso = clamp(mat.anisotropy.x, 0.0, 1.0);
  if (aniso > 1e-4) {
    let frame = rcRotateTangentFrame(mat.anisotropyTangent, mat.anisotropyBitangent, mat.anisotropy.y);
    let axes = rcAnisotropyAxes(rough, aniso);
    D = rcDistributionGGXAnisotropic(n, frame[0], frame[1], h, axes.x, axes.y);
    G = rcSmithG1Anisotropic(n, frame[0], frame[1], v, axes.x, axes.y) *
        rcSmithG1Anisotropic(n, frame[0], frame[1], l, axes.x, axes.y);
  } else {
    D = rcDistributionGGX(nDotH, rough);
    G = rcGeometrySmith(nDotV, nDotL, rough);
  }
  let F0 = rcIridescenceModifiedF0(rcBaseMaterialF0(mat), mat.iridescence, vDotH);
  let F = rcFresnelSchlick(vDotH, F0);
  let spec = (D * G) * F / max(4.0 * max(nDotV, 1e-6) * nDotL, 1e-6);
  let diffuse = mat.albedo * (1.0 - clamp(mat.metalness, 0.0, 1.0)) * RC_INV_PI;
  return (diffuse + spec) * nDotL
       + rcEvalClearcoatLobe(mat.clearcoat, mat.clearcoatNormal, v, l)
       + rcEvalSheenLobe(mat.sheen, mat.sheenRoughness, n, v, l);
}

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

  let metaDims = textureDimensions(rc_materialMapMeta);
  let metaTexel = hit.indices.w * RC_MATERIAL_MAP_META_TEXELS_PER_TRI + RC_MATERIAL_MAP_ALPHA_COVERAGE_TEXEL_OFFSET;
  if (metaTexel >= metaDims.x * metaDims.y) {
    return out;
  }

  let coverageMeta = textureLoad(rc_materialMapMeta, rcMaterialMetaCoord(metaTexel), 0);
  out.mode = u32(max(coverageMeta.x, 0.0) + 0.5);
  if (out.mode == 0u) {
    return out;
  }

  let uvs = rcHitMaterialUvs(hit);
  if (uvs.valid == 0u) {
    return out;
  }

  let baseColorTexel = rcSampleMaterialAtlasRaw(hit.indices.w, RC_MATERIAL_MAP_SLOT_BASE_COLOR, uvs.uv0, uvs.uv1);
  let baseColorAlpha = select(clamp(baseColorTexel.a, 0.0, 1.0), 1.0, baseColorTexel.x < 0.0);
  let alphaTexel = rcSampleMaterialAtlasRaw(hit.indices.w, RC_MATERIAL_MAP_SLOT_ALPHA, uvs.uv0, uvs.uv1);
  let alphaMapCoverage = select(clamp(alphaTexel.r, 0.0, 1.0), 1.0, alphaTexel.x < 0.0);
  let vertexColorAlpha = rcSampleVertexColorForHit(hit).a;
  let opacity = clamp(coverageMeta.y, 0.0, 1.0);
  out.cutoff = clamp(coverageMeta.z, 0.0, 1.0);
  out.coverage = clamp(opacity * vertexColorAlpha * baseColorAlpha * alphaMapCoverage, 0.0, 1.0);
  return out;
}

fn rcAlphaShadowTransmittanceForHit(hit: IntersectionResult) -> f32 {
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

fn rcTraceShadowTransmittance(origin: vec3f, dir: vec3f, tMax: f32, triEps: f32, skipGlass: bool) -> f32 {
  var walkRay = Ray();
  walkRay.origin = origin;
  walkRay.direction = dir;
  var traveled = 0.0;
  var tau = 1.0;
  let step = max(1e-4, triEps * 4.0);

  for (var layer = 0u; layer < 32u; layer = layer + 1u) {
    let remaining = tMax - traveled;
    if (remaining <= step || tau <= 0.001) {
      return clamp(tau, 0.0, 1.0);
    }

    let hit = rcTraceFirstHit(walkRay, triEps);
    if (!hit.didHit || hit.dist >= remaining) {
      return clamp(tau, 0.0, 1.0);
    }

    let matId = rc_triMatId[hit.indices.w];
    let mat = rc_materials[matId];
    if ((mat.flags & MATERIAL_FLAG_CAST_SHADOW_DISABLED) == 0u) {
      if (skipGlass && (mat.flags & MATERIAL_FLAG_IS_GLASS) != 0u) {
        // Preserve RC's coarse direct-light policy: scalar glass does not fully
        // occlude local probe lighting. Alpha-mapped non-glass still attenuates.
      } else {
        tau = tau * rcAlphaShadowTransmittanceForHit(hit);
        if (tau <= 0.001) {
          return 0.0;
        }
      }
    }

    traveled = traveled + hit.dist + step;
    walkRay.origin = origin + dir * traveled;
  }

  if (rcTraceAnyCastShadow(walkRay.origin, dir, max(0.0, tMax - traveled), triEps, skipGlass)) {
    return 0.0;
  }
  return clamp(tau, 0.0, 1.0);
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
  if (texel.x < 0.0) {
    return scalarEmission;
  }
  return scalarEmission * texel.rgb;
}

${RC_NEE_POINTSPOT_WGSL}

// ─── Entry point ─────────────────────────────────────────────────────────────
// Verbatim from probeRayCastKernel wgslFn body.
// TSL instanceIndex → @builtin(global_invocation_id) globalId, index = globalId.x.
// TSL storage ptr params → @group(0)/@binding(N) module-scope vars accessed by reference.

@compute @workgroup_size(64)
fn probeRayCastKernel(@builtin(global_invocation_id) globalId: vec3u) {
  let index = globalId.x;
  let u = rc_u_arr[0];
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

  let gx = f32(rayIdx % u.rayGridSize);
  let gy = f32(rayIdx / u.rayGridSize);
  let jSeed = (probeIdx * 0x9E3779B9u + rayIdx) ^ u.frameSeed;
  let jitter = vec2f(pcgHashToF32(jSeed), pcgHashToF32(jSeed * 7919u + 1u));
  let rayUV   = (vec2f(gx, gy) + jitter) / f32(u.rayGridSize);
  let rayDir  = octDecode(rayUV * 2.0 - 1.0);

  var ray = Ray();
  ray.origin    = probePos + rayDir * u.intervalNear;
  ray.direction = rayDir;
  let maxT = u.intervalFar - u.intervalNear;

  var radiance     = vec3f(0.0);
  var escaped      = true;

  // E2: read epsilon from CascadeUniforms (UBO-plumbed from HybridEngine.triIntersectEpsilon).
  let triEps = u.triIntersectEpsilon;

  let hit = rcTraceFirstHit(ray, triEps);

  // M14: scene-scale-proportional step to clear glass slabs/faces.
  // Uses the smallest room axis * 0.001 so the offset is never
  // Cornell-tuned (0.5 units) but scales with actual scene extents.
  let slabStep = min(u.roomSize.x, min(u.roomSize.y, u.roomSize.z)) * 0.001;

  if (!hit.didHit || hit.dist > maxT) {
    if (u.cascadeIndex == u.lastCascade) {
      let envUV  = dirToEquirectUV(rayDir);
      let sample = textureSampleLevel(rc_envMap, rc_envSampler, envUV, 0.0);
      radiance   = sample.rgb * u.envIntensity;
      escaped = false;
    }
  } else {
    escaped = false;

    let triIdx = hit.indices.w;
    let matId  = rc_triMatId[triIdx];
    let mat    = rc_materials[matId];

    let hitPos = ray.origin + ray.direction * hit.dist;
    let geoNormal = hit.normal;
    let smoothNormal = rcSmoothNormalForHit(hit, geoNormal);
    let normalMapped = rcApplyNormalMapForHit(hit, smoothNormal);
    let n = rcApplyBumpMapForHit(hit, normalMapped);
    let wo = -ray.direction;

    let probeMat    = rcSampleProbeHitMaterial(hit, mat.baseColor, mat.roughness, mat.metalness, smoothNormal, n);
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
    let emitterNEE = rcEmitterNEE(hitPos, n, wo, probeMat, u.emitterCount, jSeed, triEps, normalBias);

    // A7 (2026-06-10): point/spot analytic lights (fixtures). lightCount==0 ⇒ no-op.
    let pointSpotLights = evalRCPointSpotLights(hitPos, n, wo, probeMat, normalBias, triEps);

    let emissive = rcSampleSurfaceEmissiveMap(hit, matEmissive);

    var transContrib = vec3f(0.0);
    if (mat.transmission > 0.5) {
      let glassThickness = max(0.001, mat.thickness);
      let beerAttenColor = exp(-matAtten * (glassThickness / max(0.001, mat.attenuationDistance)));
      var refRay = Ray();
      // M14: step past the glass face proportionally rather than 0.5 units.
      refRay.origin    = hitPos + ray.direction * slabStep;
      refRay.direction = ray.direction;
      let secondHit = rcTraceFirstHit(refRay, triEps);
      if (!secondHit.didHit) {
        let envUV = dirToEquirectUV(refRay.direction);
        let envS  = textureSampleLevel(rc_envMap, rc_envSampler, envUV, 0.0);
        transContrib = envS.rgb * u.envIntensity * beerAttenColor * matColor;
      } else {
        let secondPos   = refRay.origin + refRay.direction * secondHit.dist;
        let secondMatId = rc_triMatId[secondHit.indices.w];
        let secondMat   = rc_materials[secondMatId];
        let secondSmoothNormal = rcSmoothNormalForHit(secondHit, secondHit.normal);
        let secondNormalMapped = rcApplyNormalMapForHit(secondHit, secondSmoothNormal);
        let secondNormal = rcApplyBumpMapForHit(secondHit, secondNormalMapped);
        let secondProbeMat = rcSampleProbeHitMaterial(secondHit, secondMat.baseColor, secondMat.roughness, secondMat.metalness, secondSmoothNormal, secondNormal);
        let secondColor = secondProbeMat.albedo;
        let secondToSun = rcSoftSunDirection(u.sunDirection, u.sunAngularRadius, secondPos, u.roomSize, u.cascadeIndex);
        var sunVis2 = vec3f(1.0);
        if (u.sunCastShadowDisabled == 0u) {
          sunVis2 = traceSunVisibility(
            secondPos + secondNormal * normalBias,
            secondToSun,
            slabStep,
            triEps,
          );
        }
        let nDotL2 = max(0.0, dot(secondNormal, secondToSun));
        transContrib = u.sunColor * secondColor * nDotL2 * 0.31831
                       * beerAttenColor * matColor * sunVis2;
      }
    }

    radiance = directSun + emitterNEE + pointSpotLights + emissive + transContrib;
  }

  let outIdx = probeIdx * u.raysPerProbe + rayIdx;
  let escapedF = select(1.0, 0.0, escaped);
  rc_cascadeOut[outIdx] = vec4f(radiance, escapedF);
}
`;
