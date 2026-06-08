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

import { BVH_INTERSECT_WGSL, MATERIAL_ENTRY_WGSL, TLAS_TRAVERSAL_WGSL } from '@vitrum/shared-bvh';
import { OCTAHEDRAL_CORE_WGSL, PCG_HASH_TO_F32_WGSL } from '@vitrum/shared-samplers';

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
  _pad2             : f32,
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
  intensity: f32,
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

// ─── Sun visibility helper ────────────────────────────────────────────────────
// Glass-aware sun shadow test.  Verbatim from sunVisibilityHelper wgslFn.

// M14 audit remediation: slabStepSize replaces the Cornell-specific 0.5-unit
// glass-slab step. Callers compute it from the scene extent
// (min(roomSize) * 0.001) so the step is proportional to the actual scene.
fn traceSunVisibility(
  origin:        vec3f,
  sunDir:        vec3f,
  slabStepSize:  f32,
  triEps:        f32,
) -> vec3f {
  var visibility = vec3f(1.0);
  var rayOrigin  = origin;
  for (var iter: u32 = 0u; iter < 3u; iter = iter + 1u) {
    var sRay = Ray();
    sRay.origin    = rayOrigin;
    sRay.direction = sunDir;
    let sHit = rcTraceFirstHit(sRay, triEps);
    if (!sHit.didHit) {
      return visibility;
    }
    let sMatId = rc_triMatId[sHit.indices.w];
    let sMat   = rc_materials[sMatId];
    if (sMat.transmission <= 0.5) {
      return vec3f(0.0);
    }
    let gThick    = max(0.001, sMat.thickness);
    let gAttenCol = sMat.attenuationColor;
    let gColor    = sMat.baseColor;
    let beerAtten = exp(-gAttenCol * (gThick / max(0.001, sMat.attenuationDistance)));
    visibility = visibility * gColor * beerAtten;
    let hitPos = rayOrigin + sunDir * sHit.dist;
    rayOrigin  = hitPos + sunDir * slabStepSize;
  }
  return vec3f(0.0);
}

// ─── Bind group declarations ──────────────────────────────────────────────────
// Cast pass: @group(0) bindings 0-8.

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

// ─── Rect-area emitter NEE ────────────────────────────────────────────────────
// RC's prior light model (radiance = directSun + emissive + envTransmission)
// could see emissive GEOMETRY a probe ray directly hit, but NOT the abstract
// rect-area emitter list — so a rect-area-only scene produced all-zero cascades
// (the 2026-06-07 "RC cascade-zero" regime gap). This adds one-sample-per-
// emitter next-event estimation at the probe-ray hit: for each emitter triangle
// sample a point, shadow-test through RC's own BVH, and add the Lambertian
// diffuse-reflected contribution. Summing one sample per emitter (rather than
// CDF-importance-sampling a single emitter) is unbiased and lower-variance for
// the handful of emitters a walkaround scene carries, and needs no CDF buffer.
//
// Estimator (area-form, pdf = 1/area ⇒ 1/pdf = area):
//   Lo += (albedo/π) · Le · (cosSurf · cosLight / dist²) · area · vis
// cosLight uses the emitter's front face only (one-sided), matching the
// shade/ReSTIR-DI convention. The shadow ray reuses traceSunVisibility's
// glass-aware semantics via rcTraceFirstHit: an OPAQUE hit before the light
// occludes; we ignore glass tint here (kept simple — RC is a coarse GI cache).
fn rcEmitterNEE(hitPos: vec3f, n: vec3f, albedo: vec3f, count: u32, seed0: u32, triEps: f32) -> vec3f {
  var Lo = vec3f(0.0);
  for (var ei: u32 = 0u; ei < count; ei = ei + 1u) {
    let e = rc_emitters[ei];
    // Per-emitter jittered area sample.
    let s0 = pcgHashToF32(seed0 ^ (ei * 0x9E3779B9u + 0x1u));
    let s1 = pcgHashToF32(seed0 * 7919u ^ (ei * 0x85EBCA6Bu + 0x2u));
    let su = sqrt(s0);
    let pos = (1.0 - su) * e.vA + (su * (1.0 - s1)) * e.vB + (su * s1) * e.vC;

    let toL    = pos - hitPos;
    let dist2  = max(dot(toL, toL), 1e-8);
    let dist   = sqrt(dist2);
    let wi     = toL / dist;
    let cosSurf  = dot(n, wi);
    let cosLight = dot(e.normal, -wi);   // emitter front face only
    if (cosSurf <= 0.0 || cosLight <= 0.0) { continue; }

    // Opaque shadow test toward the light sample (stop just short of it).
    var sRay = Ray();
    sRay.origin    = hitPos + n * 0.01;
    sRay.direction = wi;
    let sHit = rcTraceFirstHit(sRay, triEps);
    if (sHit.didHit && sHit.dist < dist - 0.02) { continue; }

    let G = (cosSurf * cosLight) / dist2;
    Lo = Lo + albedo * 0.31831 * e.Le * G * e.area;   // 0.31831 = 1/π
  }
  return Lo;
}

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
    let n      = hit.normal;

    let matColor    = mat.baseColor;
    let matAtten    = mat.attenuationColor;
    let matEmissive = mat.emissive;

    let sunVis = traceSunVisibility(hitPos + n * 0.01, u.sunDirection, slabStep, triEps);
    let nDotL  = max(0.0, dot(n, u.sunDirection));
    let directSun = u.sunColor * matColor * nDotL * 0.31831 * sunVis;

    // Rect-area emitter NEE (2026-06-07): closes the regime gap where RC saw
    // sun + emissive geometry + env but NOT the abstract rect-area emitter
    // list. emitterCount==0 ⇒ no-op (RC's prior light model, byte-identical).
    let emitterNEE = rcEmitterNEE(hitPos, n, matColor, u.emitterCount, jSeed, triEps);

    let emissive = matEmissive;

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
        let secondColor = secondMat.baseColor;
        let sunVis2 = traceSunVisibility(
          secondPos + secondHit.normal * 0.01,
          u.sunDirection,
          slabStep,
          triEps,
        );
        let nDotL2 = max(0.0, dot(secondHit.normal, u.sunDirection));
        transContrib = u.sunColor * secondColor * nDotL2 * 0.31831
                       * beerAttenColor * matColor * sunVis2;
      }
    }

    radiance = directSun + emitterNEE + emissive + transContrib;
  }

  let outIdx = probeIdx * u.raysPerProbe + rayIdx;
  let escapedF = select(1.0, 0.0, escaped);
  rc_cascadeOut[outIdx] = vec4f(radiance, escapedF);
}
`;
