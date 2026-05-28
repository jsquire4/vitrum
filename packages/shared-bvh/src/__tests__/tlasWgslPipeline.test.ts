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
});
