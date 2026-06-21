import { describe, expect, it } from 'vitest';
import { BVH_INTERSECT_WGSL } from '../wgsl/bvhIntersect.wgsl.js';
import { TLAS_SCENE_HIT_TRAVERSAL_WGSL } from '../wgsl/tlasSceneHitTraversal.wgsl.js';
import { TLAS_TRAVERSAL_WGSL } from '../wgsl/tlasTraversal.wgsl.js';

/**
 * C2 — pins that TLAS traverse-into-BLAS WGSL is exported for hybrid / RC / DDGI.
 */
describe('TLAS WGSL pipeline exports', () => {
  it('exports traceTlasFirstHit and bvhIntersectFirstHitAtRoot', () => {
    expect(TLAS_TRAVERSAL_WGSL).toContain('fn traceTlasFirstHit(');
    expect(TLAS_TRAVERSAL_WGSL).toContain('bvhIntersectFirstHitAtRoot');
    expect(BVH_INTERSECT_WGSL).toContain('fn bvhIntersectFirstHitAtRoot(');
  });

  it('uses 64-deep traversal stacks (W3.5 alignment)', () => {
    expect(TLAS_TRAVERSAL_WGSL).toContain('array<u32, 64>');
  });

  it('exports pt-webgpu SceneHit TLAS traversal (W3.6)', () => {
    expect(TLAS_SCENE_HIT_TRAVERSAL_WGSL).toContain('fn traceTlasClosest(');
    expect(TLAS_SCENE_HIT_TRAVERSAL_WGSL).toContain('fn traceTlasAny(');
    expect(TLAS_SCENE_HIT_TRAVERSAL_WGSL).toContain('array<u32, 64>');
  });

  it('threads skipGlass through TLAS closest-hit shadow traversal (H32)', () => {
    expect(TLAS_TRAVERSAL_WGSL).toContain('fn traceTlasAny(');
    expect(TLAS_TRAVERSAL_WGSL).toContain(
      'bvh_index, bvh_position, bvh, localRay, triEps, blasRoot, skipGlass,',
    );
    expect(BVH_INTERSECT_WGSL).toContain('if (skipGlass) {');
    expect(BVH_INTERSECT_WGSL).toContain('let trans4 = (idxEntry.w >> 4u) & 0xFu;');
    expect(BVH_INTERSECT_WGSL).toContain('if (trans4 > 4u) { continue; }');
  });

  it('decodes position.w UV payloads as packed f16 pairs', () => {
    expect(BVH_INTERSECT_WGSL).toContain('unpack2x16float(bitcast<u32>(pa4.w))');
    expect(BVH_INTERSECT_WGSL).not.toContain('unpack2x16unorm(bitcast<u32>(pa4.w))');
  });
});
