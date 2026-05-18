/**
 * Library-general per-channel BVH tinted-visibility shadow-ray walker.
 *
 * W7-H6 — extracted from the former `surfaceTextures.wgsl.ts`. This is the
 * piece any glass+sunlight renderer needs: it walks the BVH along a ray
 * and returns a `vec3f` visibility that multiplies the sunlight by every
 * glass slab the shadow ray passes through, but returns vec3f(0.0) on
 * the first opaque hit.
 *
 * Composition contract:
 *   `bvhTraceTintedVisibility` calls `surfaceTextureMod(uv, texId)` to
 *   weight the per-hit attenuation by the cell's authored pattern. That
 *   function is provided by the HOST module `stained-glass/surfaceMods`
 *   (see sibling file). The include-graph composes them together so the
 *   final WGSL has `surfaceTextureMod` declared before the call site.
 *
 *   In a different host app (no stained-glass patterns), the consumer
 *   would compose `glassVisibility` with a stub that returns `1.0` for
 *   every texId. The texId field comes from `decodeSurfaceTextureId`
 *   (in `common.wgsl.ts`).
 *
 * Kept out of COMMON_WGSL so the shared header stays focused on math; the
 * include-graph injects this string into shade's shader module only.
 */

import type { WgslModule } from '../pipeline/wgslComposer.js';

export const GLASS_VISIBILITY_WGSL = /* wgsl */ `
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
    let invDir = vec3f(1.0) / dir;
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
      }
    }
  }
  return visibility;
}
`;

/** W1-R6 — declarative include-graph entry. Depends on `surfaceMods` for the
 *  `surfaceTextureMod` symbol (host-app stained-glass pattern catalogue);
 *  consumers running with a different texture catalogue must provide a
 *  drop-in surfaceMods module with the same function signature. */
export const GLASS_VISIBILITY_MODULE: WgslModule = {
  name: 'glassVisibility',
  source: GLASS_VISIBILITY_WGSL,
  requires: ['common', 'surfaceMods'],
};
