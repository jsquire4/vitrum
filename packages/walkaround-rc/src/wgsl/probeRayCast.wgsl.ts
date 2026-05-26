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
 *   5. MaterialEntry struct (16 f32 fields = 64 bytes; must match bvhCompute.ts layout)
 *   6. Octahedral helpers (octEncode, octDecode) — body stripped of file header
 *   7. Probe-ray helpers (pcgHash, dirToEquirectUV)
 *   8. Sun visibility helper (traceSunVisibility)
 *   9. Entry-point function with @compute @workgroup_size(64)
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

import { MATERIAL_ENTRY_WGSL, BVH_INTERSECT_WGSL } from '@vitrum/shared-bvh';
import { OCTAHEDRAL_CORE_WGSL } from '@vitrum/shared-samplers';

export const PROBE_RAY_CAST_WGSL = /* wgsl */`
${MATERIAL_ENTRY_WGSL}
${BVH_INTERSECT_WGSL}

// ─── BVH structs + traversal ─────────────────────────────────────────────────
// Canonical from @vitrum/shared-bvh BVH_INTERSECT_WGSL (injected above).
// The pre-canonical local copies of: BVH_STACK_DEPTH / INFINITY constants;
// Ray / BVHBoundingBox / BVHNode (nested-bounds) / IntersectionResult structs;
// safeInvDir / intersectsBounds / intersectsTriangle / intersectTriangles /
// bvhIntersectFirstHit helpers — all lived here and have been removed.
//
// Call-site migration:
//   - bvhIntersectFirstHit(geom_index, geom_position, bvh, ray, triEps)
//     now resolves to the canonical V3 entry, bvhIntersectFirstHitV3 — the
//     algorithm is unchanged; the canonical uses flat BVHNode boundsMin /
//     boundsMax fields (rename-only; same memory layout). Below we re-export
//     the canonical V3 entry under the pre-canonical name so the four
//     call sites in this file (entry kernel, two glass-bounce traces,
//     traceSunVisibility) need no rename.
fn bvhIntersectFirstHit(
  bvh_index:    ptr<storage, array<vec3u>,   read>,
  bvh_position: ptr<storage, array<vec3f>,   read>,
  bvh:          ptr<storage, array<BVHNode>, read>,
  ray: Ray,
  triEps: f32,
) -> IntersectionResult {
  return bvhIntersectFirstHitV3(bvh_index, bvh_position, bvh, ray, triEps);
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
  _pad4a            : u32,
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

// ─── Probe-ray helpers ────────────────────────────────────────────────────────
// Verbatim from probeRayHelpers wgslFn in probeRayCast.wgsl.ts.

fn pcgHash(seed: u32) -> f32 {
  var s = seed * 747796405u + 2891336453u;
  let word = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return f32((word >> 22u) ^ word) / 4294967295.0;
}

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
  bvh:           ptr<storage, array<BVHNode>,        read>,
  geom_index:    ptr<storage, array<vec3u>,          read>,
  geom_position: ptr<storage, array<vec3f>,          read>,
  materials:     ptr<storage, array<MaterialEntry>,  read>,
  triMatId:      ptr<storage, array<u32>,            read>,
  origin:        vec3f,
  sunDir:        vec3f,
  slabStepSize:  f32,
  triEps:        f32,  // E2: threaded from CascadeUniforms.triIntersectEpsilon
) -> vec3f {
  var visibility = vec3f(1.0);
  var rayOrigin  = origin;
  for (var iter: u32 = 0u; iter < 3u; iter = iter + 1u) {
    var sRay = Ray();
    sRay.origin    = rayOrigin;
    sRay.direction = sunDir;
    let sHit = bvhIntersectFirstHit(geom_index, geom_position, bvh, sRay, triEps);
    if (!sHit.didHit) {
      return visibility;
    }
    let sMatId = (*triMatId)[sHit.indices.w];
    let sMat   = (*materials)[sMatId];
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

@group(0) @binding(0) var<storage, read>       rc_bvh:           array<BVHNode>;
@group(0) @binding(1) var<storage, read>       rc_geom_index:    array<vec3u>;
@group(0) @binding(2) var<storage, read>       rc_geom_position: array<vec3f>;
@group(0) @binding(3) var<storage, read>       rc_materials:     array<MaterialEntry>;
@group(0) @binding(4) var<storage, read>       rc_triMatId:      array<u32>;
@group(0) @binding(5) var<storage, read_write> rc_cascadeOut:    array<vec4f>;
@group(0) @binding(6) var                      rc_envMap:        texture_2d<f32>;
@group(0) @binding(7) var                      rc_envSampler:    sampler;
@group(0) @binding(8) var<storage, read>       rc_u_arr:         array<CascadeUniforms>;

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
  let jitter = vec2f(pcgHash(jSeed), pcgHash(jSeed * 7919u + 1u));
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

  let hit = bvhIntersectFirstHit(
    &rc_geom_index, &rc_geom_position, &rc_bvh, ray, triEps
  );

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

    let sunVis = traceSunVisibility(
      &rc_bvh, &rc_geom_index, &rc_geom_position, &rc_materials, &rc_triMatId,
      hitPos + n * 0.01,
      u.sunDirection,
      slabStep,
      triEps,
    );
    let nDotL  = max(0.0, dot(n, u.sunDirection));
    let directSun = u.sunColor * matColor * nDotL * 0.31831 * sunVis;

    let emissive = matEmissive;

    var transContrib = vec3f(0.0);
    if (mat.transmission > 0.5) {
      let glassThickness = max(0.001, mat.thickness);
      let beerAttenColor = exp(-matAtten * (glassThickness / max(0.001, mat.attenuationDistance)));
      var refRay = Ray();
      // M14: step past the glass face proportionally rather than 0.5 units.
      refRay.origin    = hitPos + ray.direction * slabStep;
      refRay.direction = ray.direction;
      let secondHit = bvhIntersectFirstHit(
        &rc_geom_index, &rc_geom_position, &rc_bvh, refRay, triEps
      );
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
          &rc_bvh, &rc_geom_index, &rc_geom_position, &rc_materials, &rc_triMatId,
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

    radiance = directSun + emissive + transContrib;
  }

  let outIdx = probeIdx * u.raysPerProbe + rayIdx;
  let escapedF = select(1.0, 0.0, escaped);
  rc_cascadeOut[outIdx] = vec4f(radiance, escapedF);
}
`;
