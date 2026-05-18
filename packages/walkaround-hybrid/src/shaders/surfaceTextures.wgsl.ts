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
// Algorithm (matches RC's traceSunVisibility / probeRayCast.wgsl):
//   visibility = vec3f(1.0)
//   for each tri the ray hits along [0, tMax):
//     if opaque  → return vec3f(0.0)   (fully shadowed)
//     if glass   → visibility *= attenuationColor * trans
//   return visibility
//
// tMax lets the caller cap the ray at e.g. the distance to a sampled
// emitter point. For directional-light queries pass a large value (INFINITY).
// ============================================================
fn bvhTraceTintedVisibility(
  bvh_index:    ptr<storage, array<vec4u>,    read>,
  bvh_position: ptr<storage, array<vec4f>,    read>,
  bvh:          ptr<storage, array<BVHNode>,  read>,
  bvh_beer:     ptr<storage, array<u32>,      read>,
  origin: vec3f,
  dir:    vec3f,
  tMax:   f32,
) -> vec3f {
  var visibility = vec3f(1.0);
  var stack: array<u32, 64>;
  var stackPtr = 0u;
  stack[stackPtr] = 0u; stackPtr++;

  while (stackPtr > 0u) {
    stackPtr--;
    let nodeIdx = stack[stackPtr];
    let node = (*bvh)[nodeIdx];

    let nMin = vec3f(node.boundsMin[0], node.boundsMin[1], node.boundsMin[2]);
    let nMax = vec3f(node.boundsMax[0], node.boundsMax[1], node.boundsMax[2]);
    // Williams 2005 IEEE-safe inverse-direction (see common.wgsl.ts).
    // Raw 1/dir produces +/-Inf on axis-aligned rays; 0 * +/-Inf = NaN
    // poisons the slab test if the ray origin coincides with an AABB face.
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
        let t = intersectTriangle(origin, dir, a, b, c, ubo.triIntersectEpsilon);
        if (t > 1e-4 && t < tMax) {
          let trans4 = (idxEntry.w >> 4u) & 0xFu;
          if (trans4 > 4u) {
            // Glass hit — multiply visibility by sqrt(Beer-Lambert × trans × texMod).
            // Two hits per cell crossing → sqrt²= the full one-cell Beer-Lambert factor.
            let matCol = decodeMaterialColor(idxEntry.w);
            let beerPacked = (*bvh_beer)[triIdx];
            let beerColor = vec3f(
              f32((beerPacked >> 24u) & 0xFFu) / 255.0,
              f32((beerPacked >> 16u) & 0xFFu) / 255.0,
              f32((beerPacked >>  8u) & 0xFFu) / 255.0,
            );
            // Procedural surface modulation at the actual hit UV.
            let pa4 = (*bvh_position)[idx.x];
            let pb4 = (*bvh_position)[idx.y];
            let pc4 = (*bvh_position)[idx.z];
            let p = origin + dir * t;
            let ab = b - a; let ac = c - a; let ap = p - a;
            let d00 = dot(ab, ab); let d01 = dot(ab, ac); let d11 = dot(ac, ac);
            let d20 = dot(ap, ab); let d21 = dot(ap, ac);
            let denom = d00 * d11 - d01 * d01;
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
            visibility = visibility * perHitFactor;
          } else {
            // Opaque hit — fully shadowed.
            return vec3f(0.0);
          }
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
