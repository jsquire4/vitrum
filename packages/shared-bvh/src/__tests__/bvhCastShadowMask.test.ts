/**
 * SHADOW-01 — cast-shadow-masked traversal derivation pins.
 *
 * `BVH_CAST_SHADOW_MASK_WGSL` is DERIVED from the canonical traversal strings
 * by anchored string surgery (bvhCastShadowMask.wgsl.ts). These tests pin the
 * derivation contract:
 *   1. module evaluation itself is the anchor guard (a changed anchor throws at
 *      import time — the import at the top of this file IS the assertion);
 *   2. the three masked entry points exist with the mask parameters;
 *   3. the bit-0/bit-2 skip is present in BOTH BLAS leaf loops (any + first-hit);
 *   4. the TLAS clone is rewired onto the masked BLAS variants;
 *   5. the CANONICAL strings remain mask-free (no accidental in-place edit).
 */
import { describe, expect, it } from 'vitest';
import {
  BVH_CAST_SHADOW_MASK_WGSL,
  BVH_CAST_SHADOW_PREDICATE_WGSL,
  BVH_INTERSECT_WGSL,
  TLAS_TRAVERSAL_WGSL,
} from '../index.js';

const SKIP =
  'if ((textureLoad(castMask, vec2i(i32(triIdx % castMaskWidth), i32(triIdx / castMaskWidth)), 0).r & 5u) != 0u) { continue; }';

describe('BVH_CAST_SHADOW_MASK_WGSL (SHADOW-01)', () => {
  it('exposes the three masked entry points with mask parameters', () => {
    expect(BVH_CAST_SHADOW_MASK_WGSL).toContain('fn bvhIntersectAnyAtRootCastMask(');
    expect(BVH_CAST_SHADOW_MASK_WGSL).toContain('fn bvhIntersectFirstHitAtRootCastMask(');
    expect(BVH_CAST_SHADOW_MASK_WGSL).toContain('fn traceTlasAnyCastMask(');
    expect(BVH_CAST_SHADOW_MASK_WGSL).toContain('castMask: texture_2d<u32>');
    expect(BVH_CAST_SHADOW_MASK_WGSL).toContain('castMaskWidth: u32');
  });

  it('inserts the cast-shadow/scalar-alpha skip into BOTH BLAS leaf loops (any-hit + glass-aware first-hit)', () => {
    const occurrences = BVH_CAST_SHADOW_MASK_WGSL.split(SKIP).length - 1;
    expect(occurrences).toBe(2);
  });

  it('rewires the TLAS clone onto the masked BLAS variants (no unmasked leak)', () => {
    const tlasClone = BVH_CAST_SHADOW_MASK_WGSL.slice(
      BVH_CAST_SHADOW_MASK_WGSL.indexOf('fn tlasTraceInstanceAnyCastMask('),
    );
    // The merged fallback + per-instance BLAS probe both use the masked forms.
    expect(tlasClone).toContain('bvhIntersectAnyAtRootCastMask(');
    expect(tlasClone).toContain('bvhIntersectFirstHitAtRootCastMask(');
    expect(tlasClone).toContain('fn tlasAnyFallbackCastMask(');
    expect(tlasClone).toContain('return tlasAnyFallbackCastMask(');
    // No call to the UNMASKED functions remains inside the clone (word-boundary:
    // every masked name contains the unmasked name as a prefix, so check the
    // exact unmasked call forms).
    expect(tlasClone).not.toContain('bvhIntersectAny(bvh_index');
    expect(tlasClone).not.toContain('bvhIntersectFirstHitAtRoot(\n');
  });

  it('keeps the canonical traversal strings mask-free', () => {
    expect(BVH_INTERSECT_WGSL).not.toContain('castMask');
    expect(TLAS_TRAVERSAL_WGSL).not.toContain('castMask');
  });

  it('does not redeclare canonical symbols (clone names are all suffixed)', () => {
    expect(BVH_CAST_SHADOW_MASK_WGSL).not.toContain('fn bvhIntersectAnyAtRoot(');
    expect(BVH_CAST_SHADOW_MASK_WGSL).not.toContain('fn bvhIntersectFirstHitAtRoot(');
    expect(BVH_CAST_SHADOW_MASK_WGSL).not.toContain('fn traceTlasAny(');
  });
});

describe('BVH_CAST_SHADOW_PREDICATE_WGSL (SHADOW-01)', () => {
  it('exposes predicate-backed any-hit variants for DDGI/RC material-entry buffers', () => {
    expect(BVH_CAST_SHADOW_PREDICATE_WGSL).toContain('fn bvhIntersectAnyAtRootCastPredicate(');
    expect(BVH_CAST_SHADOW_PREDICATE_WGSL).toContain('fn bvhIntersectFirstHitAtRootCastPredicate(');
    expect(BVH_CAST_SHADOW_PREDICATE_WGSL).toContain('fn traceTlasAnyCastPredicate(');
    expect(BVH_CAST_SHADOW_PREDICATE_WGSL).toContain('fn tlasAnyFallbackCastPredicate(');
    expect(BVH_CAST_SHADOW_PREDICATE_WGSL).toContain('return tlasAnyFallbackCastPredicate(');
    expect(BVH_CAST_SHADOW_PREDICATE_WGSL).toContain('if (bvhCastShadowDisabledForTri(triIdx)) { continue; }');
  });
});

describe('BVH any-hit overflow policy', () => {
  it('fails closed in canonical and derived shadow traversals', () => {
    const overflowGuard =
      /if \(stackPtr \+ 1u >= \d+u\) \{\s*return true;\s*\}/;
    expect(BVH_INTERSECT_WGSL).toMatch(overflowGuard);
    expect(BVH_CAST_SHADOW_MASK_WGSL).toMatch(overflowGuard);
    expect(BVH_CAST_SHADOW_PREDICATE_WGSL).toMatch(overflowGuard);
    expect(BVH_INTERSECT_WGSL).not.toMatch(
      /if \(stackPtr \+ 1u >= \d+u\) \{\s*return false;\s*\}/,
    );
  });
});
