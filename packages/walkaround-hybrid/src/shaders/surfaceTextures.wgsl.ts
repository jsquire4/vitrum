/**
 * Scene-specific procedural surface textures + tinted-visibility BVH helper.
 *
 * Both items are consumed only by shade.wgsl (the BVH helper traces shadow
 * rays that respect glass tints, which calls `surfaceTextureMod` to weight
 * the per-hit attenuation by the cell's authored pattern).
 *
 * Kept out of COMMON_WGSL so the shared header stays focused on math; the
 * pipelineCompiler injects this string into shade's shader module only.
 */

import type { WgslModule } from '../pipeline/wgslComposer.js';

export const SURFACE_TEXTURES_WGSL = /* wgsl */ `

// ============================================================
// Procedural surface-texture pattern functions
// ============================================================
//
// One function per authored surface-texture name. Each takes the hit's
// interpolated UV (already in [0,1]) and returns a scalar modulation
// factor — multiplied into the cell's emission to produce visible
// per-pixel colour variation.
//
// These functions are faithful WGSL re-implementations of the GLSL
// surface bakers — the same trig + noise math, evaluated per-shader-
// invocation instead of per-texel-during-bake.  They DO NOT hardcode
// colours; they only produce the *pattern*, which the shade pass
// multiplies into the per-cell baseColor it decoded from bvhIndex.

fn _hash21(p: vec2f) -> f32 {
  // Quick deterministic 2D-to-1D hash, range ~[0,1).
  let h = sin(dot(p, vec2f(127.1, 311.7))) * 43758.5453;
  return fract(h);
}

fn _vnoise(p: vec2f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  let a = _hash21(i);
  let b = _hash21(i + vec2f(1.0, 0.0));
  let c = _hash21(i + vec2f(0.0, 1.0));
  let d = _hash21(i + vec2f(1.0, 1.0));
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn _fbm(p: vec2f) -> f32 {
  // 4-octave fractal Brownian motion.
  var v = 0.0;
  var a = 0.5;
  var pp = p;
  for (var k = 0; k < 4; k = k + 1) {
    v = v + a * _vnoise(pp);
    pp = pp * 2.03;
    a  = a  * 0.5;
  }
  return v;
}

fn _waterglassMod(uv: vec2f) -> f32 {
  let k = 12.0;
  let w1 = sin(uv.x * k * 4.0 + uv.y * 2.0) * 0.4;
  let w2 = sin(uv.y * k * 5.0 - uv.x * 1.5) * 0.3;
  let w3 = sin((uv.x + uv.y) * k * 3.0) * 0.2;
  let w4 = sin((uv.x - uv.y) * k * 6.0) * 0.1;
  return 0.95 + (w1 + w2 + w3 + w4) * 0.45;
}

fn _rippleMod(uv: vec2f) -> f32 {
  let p = uv - vec2f(0.5);
  let r = length(p) * 22.0;
  let s = sin(r);
  return 0.95 + s * 0.4;
}

fn _hammeredMod(uv: vec2f) -> f32 {
  let n = _vnoise(uv * 14.0);
  let n2 = _vnoise(uv * 28.0 + vec2f(11.0, 7.0));
  return 0.85 + (n - 0.5) * 0.7 + (n2 - 0.5) * 0.25;
}

fn _graniteMod(uv: vec2f) -> f32 {
  let n = _vnoise(uv * 32.0);
  let n2 = _vnoise(uv * 64.0 + vec2f(3.0, 9.0));
  return 0.85 + (n - 0.5) * 0.45 + (n2 - 0.5) * 0.2;
}

fn _baroqueMod(uv: vec2f) -> f32 {
  let warp = vec2f(_fbm(uv * 2.0), _fbm(uv * 2.0 + vec2f(7.3, 1.7)));
  let v = _fbm(uv * 3.5 + warp * 1.5);
  return 0.7 + v * 0.7;
}

fn _catspawMod(uv: vec2f) -> f32 {
  let lattice = sin(uv.x * 18.0) * sin(uv.y * 18.0);
  let n = _vnoise(uv * 8.0);
  return 0.95 + lattice * 0.25 + (n - 0.5) * 0.3;
}

fn _flemishMod(uv: vec2f) -> f32 {
  let stripes = sin(uv.y * 28.0 + sin(uv.x * 6.0) * 1.2);
  let n = _vnoise(uv * vec2f(20.0, 4.0));
  return 0.9 + stripes * 0.3 + (n - 0.5) * 0.25;
}

/**
 * Procedural surface modulation factor for a glass cell.
 * Returns a single scalar that the shade pass multiplies into the
 * cell's emitted radiance, producing visible per-pixel patterns.
 *
 * Result is clamped to [0.2, 1.8] so cells stay distinguishably
 * coloured (no full black-out, no over-bright NaN-prone values).
 */
fn surfaceTextureMod(uv: vec2f, texId: u32) -> f32 {
  var m: f32 = 1.0;
  switch (texId) {
    case 0u: { m = 1.0; }                   // smooth — flat
    case 1u: { m = _hammeredMod(uv); }
    case 2u: { m = _rippleMod(uv); }
    case 3u: { m = _graniteMod(uv); }
    case 4u: { m = _baroqueMod(uv); }
    case 5u: { m = _waterglassMod(uv); }
    case 6u: { m = _catspawMod(uv); }
    case 7u: { m = _flemishMod(uv); }
    default: { m = 1.0; }
  }
  // Wider clamp range [0.2, 1.8] for more dramatic per-pixel texture modulation.
  return clamp(m, 0.2, 1.8);
}

// ============================================================
// Per-channel visibility (vec3f) along a ray. Used for sun-aware shadow
// queries that must tint the sunlight by every glass slab the shadow
// ray passes through, instead of either:
//   (a) the bool bvhIntersectAny path, which skips ALL glass tris and
//       therefore hands the floor full white sunlight even when colored
//       panel cells are in the path; or
//   (b) the opaque-shadow path, which would treat glass as a wall and
//       black-out the floor caustic entirely.
//
// Algorithm:
//   visibility = vec3f(1.0)
//   for each tri the ray hits along [0, tMax):
//     if opaque  → return vec3f(0.0)   (fully shadowed)
//     if glass   → visibility *= attenuationColor * trans
//   return visibility
//
// tMax lets the caller cap the ray at e.g. the distance to a sampled
// emitter point. For directional-light queries pass a large value (INFINITY).
//
// §4.10 (road-to-100, 2026-06-10) — TLAS-aware: in bvhMode==1 the function
// traverses the TLAS, transforms the ray to each BLAS's local space, and
// performs the leaf tinted-visibility accumulation using LOCAL positions so
// world-vs-local coordinates are never conflated. The BLAS local t is
// converted back to a world-space distance (dot(worldHitPos−origin, dir))
// for the tMax guard — mirrors the approach in traceTlasAny/traceTlasFirstHit.
// ============================================================
// WS1 (2026-05-29) — bvh_beer is an r32uint TEXTURE (not a storage buffer):
// width matches host pipeline/bvhBeerTexture.ts. Declared here (the earliest
// scene-consuming module in the shade compose chain) so both shade and
// stainedGlassShade see it. Textures are passed to WGSL functions by handle
// (no ptr), so bvhTraceTintedVisibility takes the texture directly.
const BVH_BEER_TEX_WIDTH: u32 = 4096u;

// Per-triangle Beer-Lambert tint accumulation, shared by both the BLAS-leaf
// traversal helper and the merged-BVH leaf body in bvhTraceTintedVisibility.
//
// Parameters
//   triIdx       — absolute triangle index (for beer-texture coord lookup)
//   idxEntry     — raw bvh_index entry: .xyz = vertex indices, .w = packed material
//   t            — ray hit distance (from intersectTriangle, pre-computed at call site)
//   tMaxCmp      — upper-bound for t (local t for BLAS path, world t for merged path)
//   origin / dir — ray (needed for barycentric hit-point interpolation)
//   bvh_position — vertex position buffer (.w = packed UV)
//   bvh_beer     — packed Beer-Lambert texture (r32uint, width BVH_BEER_TEX_WIDTH)
//   visibility   — in/out tinted visibility accumulator (ptr)
//
// Returns true if traversal should continue; false on opaque hit (zeroes *visibility).
fn _bvhTintedTriAccumulate(
  triIdx:       u32,
  idxEntry:     vec4u,
  t:            f32,
  tMaxCmp:      f32,
  origin:       vec3f,
  dir:          vec3f,
  bvh_position: ptr<storage, array<vec4f>, read>,
  bvh_beer:     texture_2d<u32>,
  visibility:   ptr<function, vec3f>,
) -> bool {
  if (t <= 1e-4 || t >= tMaxCmp) { return true; }
  let trans4 = (idxEntry.w >> 4u) & 0xFu;
  if (trans4 > 4u) {
    // Glass hit — multiply visibility by sqrt(Beer x trans x texMod).
    let idx = idxEntry.xyz;
    let matCol = decodeMaterialColor(idxEntry.w);
    // WS1 — beer texel: triangle index -> vec2u(tri % W, tri / W).
    let beerCoord = vec2u(triIdx % BVH_BEER_TEX_WIDTH, triIdx / BVH_BEER_TEX_WIDTH);
    let beerPacked = textureLoad(bvh_beer, vec2i(beerCoord), 0).r;
    let beerColor = vec3f(
      f32((beerPacked >> 24u) & 0xFFu) / 255.0,
      f32((beerPacked >> 16u) & 0xFFu) / 255.0,
      f32((beerPacked >>  8u) & 0xFFu) / 255.0,
    );
    // Re-read full vec4f for .w (packed UV) — the .xyz was already used
    // by the caller for the intersection test; this second load is
    // intentional (matches the pre-extracted code).
    let pa4 = (*bvh_position)[idx.x];
    let pb4 = (*bvh_position)[idx.y];
    let pc4 = (*bvh_position)[idx.z];
    let a = pa4.xyz; let b = pb4.xyz; let c = pc4.xyz;
    let p = origin + dir * t;
    let ab = b - a; let ac = c - a; let ap = p - a;
    let d00 = dot(ab, ab); let d01 = dot(ab, ac); let d11 = dot(ac, ac);
    let d20 = dot(ap, ab); let d21 = dot(ap, ac);
    let denom = max(d00 * d11 - d01 * d01, 1e-8);
    var u = clamp((d11 * d20 - d01 * d21) / denom, 0.0, 1.0);
    var v = clamp((d00 * d21 - d01 * d20) / denom, 0.0, 1.0);
    let bw = 1.0 - u - v;
    let uvA = unpack2x16unorm(bitcast<u32>(pa4.w));
    let uvB = unpack2x16unorm(bitcast<u32>(pb4.w));
    let uvC = unpack2x16unorm(bitcast<u32>(pc4.w));
    let uvAt = bw * uvA + u * uvB + v * uvC;
    let texId = decodeSurfaceTextureId(idxEntry.w);
    let texMod = surfaceTextureMod(uvAt, texId);
    let perHitFactor = sqrt(max(vec3f(1e-8), beerColor * matCol.a * texMod));
    *visibility = (*visibility) * perHitFactor;
  } else {
    // Opaque hit — fully shadowed.
    *visibility = vec3f(0.0);
    return false;
  }
  return true;
}

// Inner helper: traverse one BLAS from blasRoot, accumulating tinted visibility.
// Positions in bvh_position are LOCAL-space; tMax is in LOCAL t units.
// Returns false early (sets visibility = 0) on opaque hit.
fn _bvhTraceTintedBlasLeaves(
  bvh_index:    ptr<storage, array<vec4u>,   read>,
  bvh_position: ptr<storage, array<vec4f>,   read>,
  bvh:          ptr<storage, array<BVHNode>, read>,
  bvh_beer:     texture_2d<u32>,
  origin:   vec3f,   // LOCAL-space ray origin
  dir:      vec3f,   // LOCAL-space ray direction (unit)
  tMaxLocal: f32,
  blasRoot:  u32,
  visibility: ptr<function, vec3f>,
) -> bool {
  // Returns true if we should keep going; false if opaque hit (caller should
  // abort and return vec3f(0)).
  var stack: array<u32, 64>;
  var stackPtr = 0u;
  stack[stackPtr] = blasRoot; stackPtr++;

  while (stackPtr > 0u) {
    stackPtr--;
    let nodeIdx = stack[stackPtr];
    if (nodeIdx >= arrayLength(bvh)) { continue; }
    let node = (*bvh)[nodeIdx];

    let nMin = vec3f(node.boundsMin[0], node.boundsMin[1], node.boundsMin[2]);
    let nMax = vec3f(node.boundsMax[0], node.boundsMax[1], node.boundsMax[2]);
    // Williams 2005 §4 IEEE-safe inverse-direction.
    let invDir = safeInvDir(dir);
    let t1 = (nMin - origin) * invDir;
    let t2 = (nMax - origin) * invDir;
    let tNear = max(max(min(t1.x, t2.x), min(t1.y, t2.y)), min(t1.z, t2.z));
    let tFar  = min(min(max(t1.x, t2.x), max(t1.y, t2.y)), max(t1.z, t2.z));
    if (tNear > tFar || tFar < 0.0 || tNear > tMaxLocal) { continue; }

    let splitOrCount = node.splitAxisOrTriCount;
    if ((splitOrCount & 0xFFFF0000u) == LEAFNODE_FLAG) {
      let count = splitOrCount & 0x0000FFFFu;
      let offset = node.rightChildOrTriOffset;
      for (var i = 0u; i < count; i++) {
        let triIdx = offset + i;
        if (triIdx >= arrayLength(bvh_index)) { continue; }
        let idxEntry = (*bvh_index)[triIdx];
        let idx = idxEntry.xyz;
        let a = (*bvh_position)[idx.x].xyz;
        let b = (*bvh_position)[idx.y].xyz;
        let c = (*bvh_position)[idx.z].xyz;
        // Canonical intersectTriangle returns IntersectionResult; unwrap .dist.
        let triRes = intersectTriangle(origin, dir, a, b, c, ubo.triIntersectEpsilon);
        let t = select(BVH_INTERSECT_INFINITY, triRes.dist, triRes.didHit);
        if (!_bvhTintedTriAccumulate(triIdx, idxEntry, t, tMaxLocal, origin, dir, bvh_position, bvh_beer, visibility)) {
          return false;
        }
      }
    } else {
      let rightChild = nodeIdx + node.rightChildOrTriOffset;
      if (stackPtr < 62u) {
        stack[stackPtr] = rightChild; stackPtr++;
        stack[stackPtr] = nodeIdx + 1u; stackPtr++;
      } else {
        // Stack overflow: conservatively treat as occluded.
        *visibility = vec3f(0.0);
        return false;
      }
    }
  }
  return true;
}

fn bvhTraceTintedVisibility(
  bvh_index:    ptr<storage, array<vec4u>,    read>,
  bvh_position: ptr<storage, array<vec4f>,    read>,
  bvh:          ptr<storage, array<BVHNode>,  read>,
  bvh_beer:     texture_2d<u32>,
  origin: vec3f,
  dir:    vec3f,
  tMax:   f32,
) -> vec3f {
  var visibility = vec3f(1.0);

  // §4.10 — TLAS mode: traverse the TLAS tree, transform the ray to each
  // instance's local space, and accumulate tinted visibility via
  // _bvhTraceTintedBlasLeaves. World-space t is recovered from the local hit
  // position by dot(worldHitPos − origin, dir) (matches traceTlasAny/FirstHit).
  if (ubo.bvhMode == 1u && ubo.tlasNodeCount > 0u &&
      arrayLength(&tlasNodes) > 0u && arrayLength(&tlasInstanceIndices) > 0u) {

    var tlasStack: array<u32, 64>;
    var tlasStackPtr = 0u;
    tlasStack[tlasStackPtr] = 0u; tlasStackPtr++;

    while (tlasStackPtr > 0u) {
      tlasStackPtr--;
      let nodeIdx = tlasStack[tlasStackPtr];
      if (nodeIdx >= min(ubo.tlasNodeCount, arrayLength(&tlasNodes))) { continue; }
      let node = tlasNodes[nodeIdx];

      let nMin = vec3f(node.boundsMin[0], node.boundsMin[1], node.boundsMin[2]);
      let nMax = vec3f(node.boundsMax[0], node.boundsMax[1], node.boundsMax[2]);
      let invDirT = safeInvDir(dir);
      let t1t = (nMin - origin) * invDirT;
      let t2t = (nMax - origin) * invDirT;
      let tNearT = max(max(min(t1t.x, t2t.x), min(t1t.y, t2t.y)), min(t1t.z, t2t.z));
      let tFarT  = min(min(max(t1t.x, t2t.x), max(t1t.y, t2t.y)), max(t1t.z, t2t.z));
      if (tNearT > tFarT || tFarT < 0.0 || tNearT > tMax) { continue; }

      let splitOrCount = node.splitAxisOrTriCount;
      if ((splitOrCount & BVH_LEAFNODE_FLAG) == BVH_LEAFNODE_FLAG) {
        let count = splitOrCount & 0x0000FFFFu;
        let start = node.rightChildOrTriOffset;
        for (var i = 0u; i < count; i++) {
          let permIdx = start + i;
          if (permIdx >= arrayLength(&tlasInstanceIndices)) { continue; }
          let instIdx = tlasInstanceIndices[permIdx];
          let m = instIdx * 4u;
          if (m + 3u >= arrayLength(&tlasInstanceWorldToLocal) ||
              m + 3u >= arrayLength(&tlasInstanceLocalToWorld)) { continue; }
          let w2l0 = tlasInstanceWorldToLocal[m];
          let w2l1 = tlasInstanceWorldToLocal[m + 1u];
          let w2l2 = tlasInstanceWorldToLocal[m + 2u];
          let w2l3 = tlasInstanceWorldToLocal[m + 3u];
          // Transform ray to local space (mirrors traceTlasFirstHit).
          let localOrigin = tlasTransformPointCols(w2l0, w2l1, w2l2, w2l3, origin);
          let localDir    = tlasTransformDirectionCols(w2l0, w2l1, w2l2, dir);
          // Use a conservative local tMax of 1e20 — the BLAS leaf test uses a
          // local t but the outer TLAS AABB already bounds the traversal to the
          // world segment [0, tMax]. This matches traceTlasAny's approach.
          let blasRoot = select(0u, tlasBlasRoots[instIdx], instIdx < arrayLength(&tlasBlasRoots));
          // Run tinted BLAS traversal; if an opaque hit fires, visibility → 0.
          let cont = _bvhTraceTintedBlasLeaves(
            bvh_index, bvh_position, bvh, bvh_beer,
            localOrigin, localDir, 1e20,
            blasRoot, &visibility,
          );
          if (!cont) { return vec3f(0.0); }
        }
      } else {
        let rightChild = nodeIdx + node.rightChildOrTriOffset;
        if (tlasStackPtr < 62u) {
          tlasStack[tlasStackPtr] = rightChild; tlasStackPtr++;
          tlasStack[tlasStackPtr] = nodeIdx + 1u; tlasStackPtr++;
        } else {
          return visibility;
        }
      }
    }
    return visibility;
  }

  // ── Merged world-BVH path (bvhMode == 0) ─────────────────────────────────
  // All positions in bvh_position are world-space; t is directly comparable
  // to tMax. Unchanged from the pre-§4.10 implementation.
  var stack: array<u32, 64>;
  var stackPtr = 0u;
  stack[stackPtr] = 0u; stackPtr++;

  while (stackPtr > 0u) {
    stackPtr--;
    let nodeIdx = stack[stackPtr];
    let node = (*bvh)[nodeIdx];

    let nMin = vec3f(node.boundsMin[0], node.boundsMin[1], node.boundsMin[2]);
    let nMax = vec3f(node.boundsMax[0], node.boundsMax[1], node.boundsMax[2]);
    // Williams 2005 §4 IEEE-safe inverse-direction. Plain \`vec3f(1.0)/dir\`
    // produces ±Inf when a component of \`dir\` is exactly zero (axis-aligned
    // sun-shadow rays, axis-aligned hit normals, sky-aperture taps along
    // ±X/Y/Z), and \`0 * Inf = NaN\` then poisons the slab test when the ray
    // origin sits on the corresponding AABB face. safeInvDir lives in
    // @vitrum/shared-bvh's bvhIntersect.wgsl and is in scope because composeWgsl
    // topo-sorts that module ahead of this one when building the shade module.
    let invDir = safeInvDir(dir);
    let t1 = (nMin - origin) * invDir;
    let t2 = (nMax - origin) * invDir;
    let tNear = max(max(min(t1.x, t2.x), min(t1.y, t2.y)), min(t1.z, t2.z));
    let tFar  = min(min(max(t1.x, t2.x), max(t1.y, t2.y)), max(t1.z, t2.z));
    if (tNear > tFar || tFar < 0.0 || tNear > tMax) { continue; }

    let splitOrCount = node.splitAxisOrTriCount;
    if ((splitOrCount & 0xFFFF0000u) == LEAFNODE_FLAG) {
      let count = splitOrCount & 0x0000FFFFu;
      let offset = node.rightChildOrTriOffset;
      for (var i = 0u; i < count; i++) {
        let triIdx = offset + i;
        let idxEntry = (*bvh_index)[triIdx];
        let idx = idxEntry.xyz;
        let a = (*bvh_position)[idx.x].xyz;
        let b = (*bvh_position)[idx.y].xyz;
        let c = (*bvh_position)[idx.z].xyz;
        // Canonical intersectTriangle now returns IntersectionResult; unwrap
        // .dist (or INFINITY when !didHit) so the rest of this helper continues
        // to operate on a plain f32 t-value.
        let triRes = intersectTriangle(origin, dir, a, b, c, ubo.triIntersectEpsilon);
        let t = select(BVH_INTERSECT_INFINITY, triRes.dist, triRes.didHit);
        if (!_bvhTintedTriAccumulate(triIdx, idxEntry, t, tMax, origin, dir, bvh_position, bvh_beer, &visibility)) {
          return vec3f(0.0);
        }
      }
    } else {
      let rightChild = nodeIdx + node.rightChildOrTriOffset;
      if (stackPtr < 62u) {
        stack[stackPtr] = rightChild; stackPtr++;
        stack[stackPtr] = nodeIdx + 1u; stackPtr++;
      } else {
        // Stack overflow: bail out with current accumulated visibility
        // rather than silently dropping the right subtree.  At depth 64
        // a balanced BVH spans 2^64 triangles, so this branch is
        // unreachable for any real scene; the guard exists for invariant
        // clarity.
        return visibility;
      }
    }
  }
  return visibility;
}
`;

/** W1-R6 — declarative include-graph entry. */
export const SURFACE_TEXTURES_MODULE: WgslModule = {
  name: 'surfaceTextures',
  source: SURFACE_TEXTURES_WGSL,
  requires: ['common'],
};
