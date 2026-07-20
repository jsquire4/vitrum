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
import { RC_MATERIAL_ATLAS_WGSL } from './rcMaterialAtlas.wgsl.js';
import { RC_BRDF_WGSL } from './rcBrdf.wgsl.js';

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
