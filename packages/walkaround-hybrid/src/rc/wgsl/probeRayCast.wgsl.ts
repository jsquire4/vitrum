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

export const PROBE_RAY_CAST_WGSL = /* wgsl */`

// ─── three-mesh-bvh: constants ───────────────────────────────────────────────

const BVH_STACK_DEPTH = 60u;
const INFINITY = 1e20;
// E2: TRI_INTERSECT_EPSILON removed from module scope; value is now
// read from CascadeUniforms.triIntersectEpsilon (UBO-plumbed per M4.A).
// intersectsTriangle() receives it as a function parameter.

// ─── three-mesh-bvh: structs ─────────────────────────────────────────────────

struct Ray {
  origin: vec3f,
  direction: vec3f,
};

struct BVHBoundingBox {
  min: array<f32, 3>,
  max: array<f32, 3>,
}

struct BVHNode {
  bounds: BVHBoundingBox,
  rightChildOrTriangleOffset: u32,
  splitAxisOrTriangleCount: u32,
};

struct IntersectionResult {
  didHit: bool,
  indices: vec4u,
  normal: vec3f,
  barycoord: vec3f,
  side: f32,
  dist: f32,
};

// ─── safeInvDir helper ────────────────────────────────────────────────────────
// Williams 2005 §4 IEEE-safe inverse-direction: substitutes a finite large
// value when a component is near-zero to avoid NaN from 0 * ±Inf in the slab
// test.  WGSL sign(0)==0, so a zero component yields 0 * 1e30 == 0, which is
// correct (zero-direction axis contributes nothing to tNear/tFar).
fn safeInvDir(d: vec3f) -> vec3f {
  return vec3f(
    select(1.0 / d.x, sign(d.x) * 1e30, abs(d.x) < 1e-30),
    select(1.0 / d.y, sign(d.y) * 1e30, abs(d.y) < 1e-30),
    select(1.0 / d.z, sign(d.z) * 1e30, abs(d.z) < 1e-30),
  );
}

// ─── three-mesh-bvh: intersectsBounds ────────────────────────────────────────

fn intersectsBounds(
  ray: Ray,
  bounds: BVHBoundingBox,
  dist: ptr<function, f32>
) -> bool {

  let boundsMin = vec3( bounds.min[0], bounds.min[1], bounds.min[2] );
  let boundsMax = vec3( bounds.max[0], bounds.max[1], bounds.max[2] );

  let invDir = safeInvDir(ray.direction);
  let tMinPlane = ( boundsMin - ray.origin ) * invDir;
  let tMaxPlane = ( boundsMax - ray.origin ) * invDir;

  let tMinHit = vec3f(
    min( tMinPlane.x, tMaxPlane.x ),
    min( tMinPlane.y, tMaxPlane.y ),
    min( tMinPlane.z, tMaxPlane.z )
  );

  let tMaxHit = vec3f(
    max( tMinPlane.x, tMaxPlane.x ),
    max( tMinPlane.y, tMaxPlane.y ),
    max( tMinPlane.z, tMaxPlane.z )
  );

  let t0 = max( max( tMinHit.x, tMinHit.y ), tMinHit.z );
  let t1 = min( min( tMaxHit.x, tMaxHit.y ), tMaxHit.z );

  ( *dist ) = max( t0, 0.0 );

  return t1 >= ( *dist );
}

// ─── three-mesh-bvh: intersectsTriangle ──────────────────────────────────────

// E2: triEps (Möller–Trumbore coplanarity threshold) is passed as a parameter
// rather than read from a module-scope constant. All call sites thread the
// value from CascadeUniforms.triIntersectEpsilon.
fn intersectsTriangle( ray: Ray, a: vec3f, b: vec3f, c: vec3f, triEps: f32 ) -> IntersectionResult {

  var result: IntersectionResult;
  result.didHit = false;

  let edge1 = b - a;
  let edge2 = c - a;
  let n = cross( edge1, edge2 );

  let det = - dot( ray.direction, n );

  if ( abs( det ) < triEps ) {
    return result;
  }

  let invdet = 1.0 / det;

  let AO = ray.origin - a;
  let DAO = cross( AO, ray.direction );

  let u = dot( edge2, DAO ) * invdet;
  let v = -dot( edge1, DAO ) * invdet;
  let t = dot( AO, n ) * invdet;

  let w = 1.0 - u - v;

  if ( u < - triEps || v < - triEps || w < - triEps || t < triEps ) {
    return result;
  }

  result.didHit = true;
  result.barycoord = vec3f( w, u, v );
  result.dist = t;
  result.side = sign( det );
  result.normal = result.side * normalize( n );

  return result;
}

// ─── three-mesh-bvh: intersectTriangles ──────────────────────────────────────

fn intersectTriangles(
  bvh_position: ptr<storage, array<vec3f>, read>,
  bvh_index: ptr<storage, array<vec3u>, read>,
  offset: u32,
  count: u32,
  ray: Ray,
  triEps: f32,  // E2: UBO-plumbed epsilon
) -> IntersectionResult {

  var closestResult: IntersectionResult;
  closestResult.didHit = false;
  closestResult.dist = INFINITY;

  for ( var i = offset; i < offset + count; i = i + 1u ) {
    let indices = bvh_index[ i ];
    let a = bvh_position[ indices.x ];
    let b = bvh_position[ indices.y ];
    let c = bvh_position[ indices.z ];

    var triResult = intersectsTriangle( ray, a, b, c, triEps );

    if ( triResult.didHit && triResult.dist < closestResult.dist ) {
      closestResult = triResult;
      closestResult.indices = vec4u( indices.xyz, i );
    }
  }

  return closestResult;
}

// ─── three-mesh-bvh: bvhIntersectFirstHit ────────────────────────────────────

fn bvhIntersectFirstHit(
  bvh_index: ptr<storage, array<vec3u>, read>,
  bvh_position: ptr<storage, array<vec3f>, read>,
  bvh: ptr<storage, array<BVHNode>, read>,
  ray: Ray,
  triEps: f32,  // E2: UBO-plumbed epsilon threaded from CascadeUniforms
) -> IntersectionResult {

  var pointer = 0;
  var stack: array<u32, BVH_STACK_DEPTH>;
  stack[ 0 ] = 0u;

  var bestHit: IntersectionResult;
  bestHit.didHit = false;
  bestHit.dist = INFINITY;

  loop {
    if ( pointer < 0 || pointer >= i32( BVH_STACK_DEPTH ) ) {
      break;
    }

    let currNodeIndex = stack[ pointer ];
    let node = bvh[ currNodeIndex ];

    pointer = pointer - 1;

    var boundsHitDistance: f32 = 0.0;

    if ( ! intersectsBounds( ray, node.bounds, &boundsHitDistance ) || boundsHitDistance > bestHit.dist ) {
      continue;
    }

    let boundsInfox = node.splitAxisOrTriangleCount;
    let boundsInfoy = node.rightChildOrTriangleOffset;

    let isLeaf = ( boundsInfox & 0xffff0000u ) != 0u;

    if ( isLeaf ) {
      let count = boundsInfox & 0x0000ffffu;
      let offset = boundsInfoy;

      let localHit = intersectTriangles(
        bvh_position, bvh_index, offset, count, ray, triEps
      );

      if ( localHit.didHit && localHit.dist < bestHit.dist ) {
        bestHit = localHit;
      }

    } else {
      let leftIndex = currNodeIndex + 1u;
      let splitAxis = boundsInfox & 0x0000ffffu;
      let rightIndex = currNodeIndex + boundsInfoy;

      let leftToRight = ray.direction[splitAxis] >= 0.0;
      let c1 = select( rightIndex, leftIndex, leftToRight );
      let c2 = select( leftIndex, rightIndex, leftToRight );

      pointer = pointer + 1;
      stack[ pointer ] = c2;

      pointer = pointer + 1;
      stack[ pointer ] = c1;
    }
  }

  return bestHit;
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
// 16 × f32 = 64 bytes per entry.  Uses individual f32 members (not vec3f) to
// guarantee CPU/GPU layout identity.  Matches packCascadeMaterials() in bvhCompute.ts.

struct MaterialEntry {
  colorR      : f32,
  colorG      : f32,
  colorB      : f32,
  colorA      : f32,
  transmission: f32,
  ior         : f32,
  attenColorR : f32,
  attenColorG : f32,
  attenColorB : f32,
  attenDist   : f32,
  roughness   : f32,
  metalness   : f32,
  emissiveR   : f32,
  emissiveG   : f32,
  emissiveB   : f32,
  thickness   : f32,
};

// ─── Octahedral helpers (from @vitrum/shared-bvh octahedral.wgsl.ts) ─────────
// Call sites use octDecode(uv * 2.0 - 1.0) to remap from [0,1] to [-1,1].

fn octEncode(dir: vec3f) -> vec2f {
  // Zero-vector guard — see octahedralCore.wgsl for rationale.
  let n = dir / max(abs(dir.x) + abs(dir.y) + abs(dir.z), 1e-20);
  if (n.z >= 0.0) {
    return n.xy;
  }
  return (1.0 - abs(n.yx)) * vec2f(sign(n.x), sign(n.y));
}

fn octDecode(oct: vec2f) -> vec3f {
  let n = vec3f(oct, 1.0 - abs(oct.x) - abs(oct.y));
  if (n.z < 0.0) {
    let xy = (1.0 - abs(n.yx)) * vec2f(sign(n.x), sign(n.y));
    return normalize(vec3f(xy, n.z));
  }
  return normalize(n);
}

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
    let gAttenCol = vec3f(sMat.attenColorR, sMat.attenColorG, sMat.attenColorB);
    let gColor    = vec3f(sMat.colorR,      sMat.colorG,      sMat.colorB);
    let beerAtten = exp(-gAttenCol * (gThick / max(0.001, sMat.attenDist)));
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

    let matColor    = vec3f(mat.colorR,     mat.colorG,     mat.colorB);
    let matAtten    = vec3f(mat.attenColorR, mat.attenColorG, mat.attenColorB);
    let matEmissive = vec3f(mat.emissiveR,  mat.emissiveG,  mat.emissiveB);

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
      let beerAttenColor = exp(-matAtten * (glassThickness / max(0.001, mat.attenDist)));
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
        let secondColor = vec3f(secondMat.colorR, secondMat.colorG, secondMat.colorB);
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
