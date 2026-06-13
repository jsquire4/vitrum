/**
 * Cast-shadow-masked any-hit traversal variants (SHADOW-01, 2026-06-11).
 *
 * `MeshPrimitive.castShadow:false` must make geometry transparent to SHADOW
 * (occlusion) rays while staying visible to camera/radiance rays. walkaround's
 * per-triangle material lane (`bvh_material`, r32uint texture — see
 * walkaround-hybrid/shaders/materialDecode.wgsl.ts + restir/packingHelpers.ts
 * `packBVHRoughMetalFromCore`) carries the flag in material-flag BIT 0:
 * `(word & 1u) != 0u` ⟺ "does NOT cast shadows".
 *
 * These variants are DERIVED from the canonical traversal strings
 * (`BVH_INTERSECT_WGSL` / `TLAS_TRAVERSAL_WGSL`) by anchored string surgery —
 * the same single-source-of-truth pattern as pt-webgpu's
 * `tlasSceneHitTraversalWithInstanceIndex`. Every rewrite is guarded: if an
 * anchor in the source string changes, module evaluation throws instead of
 * silently shipping a stale clone. The transforms are:
 *   1. rename the function (`*CastMask` suffix),
 *   2. append `castMask: texture_2d<u32>, castMaskWidth: u32` parameters,
 *   3. insert the bit-0 skip immediately after the per-leaf `idxEntry` load,
 *   4. (TLAS only) rewire the inner BLAS calls to the masked variants.
 *
 * Exposed texture-mask functions:
 *   - `bvhIntersectAnyAtRootCastMask`     — masked any-hit (merged path / TLAS fallback)
 *   - `bvhIntersectFirstHitAtRootCastMask`— masked glass-aware closest-hit
 *                                           (the per-instance probe inside the TLAS
 *                                           any-hit; see H32 in tlasTraversal.wgsl.ts)
 *   - `traceTlasAnyCastMask`              — masked TLAS any-hit
 *
 * Exposed predicate-mask functions:
 *   - `bvhIntersectAnyAtRootCastPredicate`
 *   - `bvhIntersectFirstHitAtRootCastPredicate`
 *   - `traceTlasAnyCastPredicate`
 *
 * The predicate variants require the including shader to define:
 *
 *   fn bvhCastShadowDisabledForTri(triIdx: u32) -> bool
 *
 * before the derived functions are compiled. They are for passes that already
 * have material-id / material-entry buffers (DDGI / RC) instead of the main
 * walkaround `bvh_material` texture.
 *
 * Requires the including module to have composed `BVH_INTERSECT_WGSL` and
 * `TLAS_TRAVERSAL_WGSL` first (BVHNode / Ray / IntersectionResult / safeInvDir /
 * intersectTriangle / tlasTransform* / tlasIntersectAabb).
 *
 * The mask texel addressing matches walkaround's `bvh_material` upload
 * (BVH_MATERIAL_TEX_WIDTH-wide r32uint rows): texel = (tri % W, tri / W).
 * `triIdx` here is the GLOBAL triangle index in both merged and TLAS modes
 * (TLAS BLAS leaves carry concat-global tri offsets), so one addressing rule
 * covers both paths.
 */

import { BVH_INTERSECT_WGSL } from './bvhIntersect.wgsl.js';
import { TLAS_TRAVERSAL_WGSL } from './tlasTraversal.wgsl.js';

/** Replace `anchor` with `replacement` in `source`; throw when the anchor is
 *  missing (the canonical traversal text changed — update this module). */
function mustReplace(source: string, anchor: string, replacement: string, what: string): string {
  const out = source.replace(anchor, replacement);
  if (out === source) {
    throw new Error(
      `bvhCastShadowMask.wgsl.ts: anchor for "${what}" not found in the canonical ` +
      `traversal WGSL — the source string changed; update the cast-shadow-mask derivation.`,
    );
  }
  return out;
}

/** Extract one top-level `fn <header> ... \n}` block from a composed WGSL string.
 *  The canonical traversal functions close with a column-0 `}` and contain only
 *  indented inner braces, so the first `\n}` after the header is the function end. */
function extractFn(source: string, header: string): string {
  const start = source.indexOf(header);
  if (start < 0) {
    throw new Error(
      `bvhCastShadowMask.wgsl.ts: function header "${header}" not found in the canonical ` +
      `traversal WGSL — the source string changed; update the cast-shadow-mask derivation.`,
    );
  }
  const end = source.indexOf('\n}', start);
  if (end < 0) {
    throw new Error(`bvhCastShadowMask.wgsl.ts: unterminated function body for "${header}".`);
  }
  return source.slice(start, end + 2);
}

/** The bit-0 skip inserted into every masked leaf loop. */
const CAST_MASK_SKIP_LINE =
  `        // SHADOW-01 — skip castShadow:false triangles (bvh_material bit 0).\n` +
  `        if ((textureLoad(castMask, vec2i(i32(triIdx % castMaskWidth), i32(triIdx / castMaskWidth)), 0).r & 1u) != 0u) { continue; }\n`;

const CAST_PREDICATE_SKIP_LINE =
  `        // SHADOW-01 — skip castShadow:false triangles via includer predicate.\n` +
  `        if (bvhCastShadowDisabledForTri(triIdx)) { continue; }\n`;

function buildMaskedAnyAtRoot(): string {
  let fn = extractFn(BVH_INTERSECT_WGSL, 'fn bvhIntersectAnyAtRoot(');
  fn = mustReplace(
    fn,
    'fn bvhIntersectAnyAtRoot(',
    'fn bvhIntersectAnyAtRootCastMask(',
    'any-at-root rename',
  );
  fn = mustReplace(
    fn,
    '  skipGlass: bool,\n  rootNode: u32,\n) -> bool {',
    '  skipGlass: bool,\n  rootNode: u32,\n  castMask: texture_2d<u32>,\n  castMaskWidth: u32,\n) -> bool {',
    'any-at-root mask params',
  );
  fn = mustReplace(
    fn,
    '        let idxEntry = (*bvh_index)[triIdx];\n',
    '        let idxEntry = (*bvh_index)[triIdx];\n' + CAST_MASK_SKIP_LINE,
    'any-at-root leaf skip',
  );
  return fn;
}

function buildPredicateAnyAtRoot(): string {
  let fn = extractFn(BVH_INTERSECT_WGSL, 'fn bvhIntersectAnyAtRoot(');
  fn = mustReplace(
    fn,
    'fn bvhIntersectAnyAtRoot(',
    'fn bvhIntersectAnyAtRootCastPredicate(',
    'predicate any-at-root rename',
  );
  fn = mustReplace(
    fn,
    '        let idxEntry = (*bvh_index)[triIdx];\n',
    '        let idxEntry = (*bvh_index)[triIdx];\n' + CAST_PREDICATE_SKIP_LINE,
    'predicate any-at-root leaf skip',
  );
  return fn;
}

function buildMaskedFirstHitAtRoot(): string {
  let fn = extractFn(BVH_INTERSECT_WGSL, 'fn bvhIntersectFirstHitAtRoot(');
  fn = mustReplace(
    fn,
    'fn bvhIntersectFirstHitAtRoot(',
    'fn bvhIntersectFirstHitAtRootCastMask(',
    'first-hit-at-root rename',
  );
  fn = mustReplace(
    fn,
    '  rootNode: u32,\n  skipGlass: bool,\n) -> IntersectionResult {',
    '  rootNode: u32,\n  skipGlass: bool,\n  castMask: texture_2d<u32>,\n  castMaskWidth: u32,\n) -> IntersectionResult {',
    'first-hit-at-root mask params',
  );
  fn = mustReplace(
    fn,
    '        let idxEntry = (*bvh_index)[triIdx];\n',
    '        let idxEntry = (*bvh_index)[triIdx];\n' + CAST_MASK_SKIP_LINE,
    'first-hit-at-root leaf skip',
  );
  return fn;
}

function buildPredicateFirstHitAtRoot(): string {
  let fn = extractFn(BVH_INTERSECT_WGSL, 'fn bvhIntersectFirstHitAtRoot(');
  fn = mustReplace(
    fn,
    'fn bvhIntersectFirstHitAtRoot(',
    'fn bvhIntersectFirstHitAtRootCastPredicate(',
    'predicate first-hit-at-root rename',
  );
  fn = mustReplace(
    fn,
    '        let idxEntry = (*bvh_index)[triIdx];\n',
    '        let idxEntry = (*bvh_index)[triIdx];\n' + CAST_PREDICATE_SKIP_LINE,
    'predicate first-hit-at-root leaf skip',
  );
  return fn;
}

function buildMaskedTlasAny(): string {
  let fn = extractFn(TLAS_TRAVERSAL_WGSL, 'fn traceTlasAny(');
  fn = mustReplace(fn, 'fn traceTlasAny(', 'fn traceTlasAnyCastMask(', 'tlas-any rename');
  fn = mustReplace(
    fn,
    '  skipGlass: bool,\n) -> bool {',
    '  skipGlass: bool,\n  castMask: texture_2d<u32>,\n  castMaskWidth: u32,\n) -> bool {',
    'tlas-any mask params',
  );
  fn = mustReplace(
    fn,
    'return bvhIntersectAny(bvh_index, bvh_position, bvh, origin, dir, tMax, triEps, skipGlass);',
    'return bvhIntersectAnyAtRootCastMask(bvh_index, bvh_position, bvh, origin, dir, tMax, triEps, skipGlass, 0u, castMask, castMaskWidth);',
    'tlas-any merged fallback rewire',
  );
  fn = mustReplace(
    fn,
    'let localHit = bvhIntersectFirstHitAtRoot(\n          bvh_index, bvh_position, bvh, localRay, triEps, blasRoot, skipGlass,\n        );',
    'let localHit = bvhIntersectFirstHitAtRootCastMask(\n          bvh_index, bvh_position, bvh, localRay, triEps, blasRoot, skipGlass, castMask, castMaskWidth,\n        );',
    'tlas-any per-instance BLAS rewire',
  );
  return fn;
}

function buildPredicateTlasAny(): string {
  let fn = extractFn(TLAS_TRAVERSAL_WGSL, 'fn traceTlasAny(');
  fn = mustReplace(fn, 'fn traceTlasAny(', 'fn traceTlasAnyCastPredicate(', 'predicate tlas-any rename');
  fn = mustReplace(
    fn,
    'return bvhIntersectAny(bvh_index, bvh_position, bvh, origin, dir, tMax, triEps, skipGlass);',
    'return bvhIntersectAnyAtRootCastPredicate(bvh_index, bvh_position, bvh, origin, dir, tMax, triEps, skipGlass, 0u);',
    'predicate tlas-any merged fallback rewire',
  );
  fn = mustReplace(
    fn,
    'let localHit = bvhIntersectFirstHitAtRoot(\n          bvh_index, bvh_position, bvh, localRay, triEps, blasRoot, skipGlass,\n        );',
    'let localHit = bvhIntersectFirstHitAtRootCastPredicate(\n          bvh_index, bvh_position, bvh, localRay, triEps, blasRoot, skipGlass,\n        );',
    'predicate tlas-any per-instance BLAS rewire',
  );
  return fn;
}

/**
 * WGSL snippet providing the three cast-shadow-masked traversal entry points.
 * Compose AFTER `BVH_INTERSECT_WGSL` + `TLAS_TRAVERSAL_WGSL`.
 */
export const BVH_CAST_SHADOW_MASK_WGSL = /* wgsl */ `
// ─── SHADOW-01 — cast-shadow-masked occlusion traversal (derived; see
// shared-bvh/src/wgsl/bvhCastShadowMask.wgsl.ts for the derivation contract) ───

${buildMaskedAnyAtRoot()}

${buildMaskedFirstHitAtRoot()}

${buildMaskedTlasAny()}
`;

/**
 * Predicate-backed cast-shadow-masked traversal entry points. The includer must
 * provide `bvhCastShadowDisabledForTri(triIdx)` before using these functions.
 */
export const BVH_CAST_SHADOW_PREDICATE_WGSL = /* wgsl */ `
// ─── SHADOW-01 — predicate-backed cast-shadow-masked occlusion traversal ───

${buildPredicateAnyAtRoot()}

${buildPredicateFirstHitAtRoot()}

${buildPredicateTlasAny()}
`;
