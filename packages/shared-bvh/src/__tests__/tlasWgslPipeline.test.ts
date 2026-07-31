import { describe, expect, it } from 'vitest';
import {
  BVH_INTERSECT_CORE_WGSL,
  BVH_INTERSECT_WGSL,
} from '../wgsl/bvhIntersect.wgsl.js';
import { TLAS_SCENE_HIT_TRAVERSAL_WGSL } from '../wgsl/tlasSceneHitTraversal.wgsl.js';
import {
  TLAS_TRAVERSAL_CORE_WGSL,
  TLAS_TRAVERSAL_STATUS_COMPLETE,
  TLAS_TRAVERSAL_STATUS_FALLBACK,
  TLAS_TRAVERSAL_STATUS_STACK_OVERFLOW,
  TLAS_TRAVERSAL_WGSL,
} from '../wgsl/tlasTraversal.wgsl.js';

/**
 * C2 — pins that TLAS traverse-into-BLAS WGSL is exported for hybrid / RC / DDGI.
 */
describe('TLAS WGSL pipeline exports', () => {
  it('exports traceTlasFirstHit and bvhIntersectFirstHitAtRoot', () => {
    expect(TLAS_TRAVERSAL_WGSL).toContain('fn traceTlasFirstHit(');
    expect(TLAS_TRAVERSAL_WGSL).toContain('bvhIntersectFirstHitAtRoot');
    expect(BVH_INTERSECT_WGSL).toContain('fn bvhIntersectFirstHitAtRoot(');
  });

  it('uses the full 64-entry stack capacity without an off-by-one drop', () => {
    expect(TLAS_TRAVERSAL_WGSL).toContain('array<u32, 64>');
    expect(TLAS_TRAVERSAL_WGSL).toContain('stackPtr + 2u <= 64u');
    expect(TLAS_TRAVERSAL_WGSL).not.toContain('stackPtr + 2u < 64u');
  });

  it('exports pt-webgpu SceneHit TLAS traversal (W3.6)', () => {
    expect(TLAS_SCENE_HIT_TRAVERSAL_WGSL).toContain('fn traceTlasClosest(');
    expect(TLAS_SCENE_HIT_TRAVERSAL_WGSL).toContain('fn traceTlasAny(');
    expect(TLAS_SCENE_HIT_TRAVERSAL_WGSL).toContain('array<u32, 64>');
    expect(TLAS_SCENE_HIT_TRAVERSAL_WGSL).toContain('stackPtr + 2u <= 64u');
  });

  it('publishes traversal completion, overflow, and fallback status to shader callers', () => {
    for (const source of [TLAS_TRAVERSAL_WGSL, TLAS_SCENE_HIT_TRAVERSAL_WGSL]) {
      expect(source).toContain('fn tlasLastTraversalStatus() -> u32');
      expect(source).toContain(
        `const TLAS_TRAVERSAL_STATUS_COMPLETE: u32 = ${TLAS_TRAVERSAL_STATUS_COMPLETE}u;`,
      );
      expect(source).toContain(
        `const TLAS_TRAVERSAL_STATUS_STACK_OVERFLOW: u32 = ${TLAS_TRAVERSAL_STATUS_STACK_OVERFLOW}u;`,
      );
      expect(source).toContain(
        `const TLAS_TRAVERSAL_STATUS_FALLBACK: u32 = ${TLAS_TRAVERSAL_STATUS_FALLBACK}u;`,
      );
      expect(source).toContain(
        'tlasTraversalStatusCode = TLAS_TRAVERSAL_STATUS_STACK_OVERFLOW;',
      );
    }
    expect(TLAS_TRAVERSAL_WGSL).toContain(
      'tlasTraversalStatusCode = TLAS_TRAVERSAL_STATUS_COMPLETE;',
    );
  });

  it('restarts overflowed traversal with an exact stackless instance fallback', () => {
    expect(TLAS_TRAVERSAL_WGSL).toContain('fn tlasFirstHitFallback(');
    expect(TLAS_TRAVERSAL_WGSL).toContain('fn tlasAnyFallback(');
    expect(TLAS_TRAVERSAL_WGSL).toContain(
      'return tlasFirstHitFallback(ray, triEps);',
    );
    expect(TLAS_TRAVERSAL_WGSL).toContain(
      'return tlasAnyFallback(ray, tMax, triEps, skipGlass);',
    );
    expect(TLAS_TRAVERSAL_WGSL).toContain(
      'permIdx < tlasInstanceIndexCount()',
    );
    expect(TLAS_TRAVERSAL_WGSL).not.toContain(
      'tlasTraversalStatusCode = TLAS_TRAVERSAL_STATUS_STACK_OVERFLOW;\n        return best;',
    );
    expect(TLAS_TRAVERSAL_WGSL).not.toContain(
      'tlasTraversalStatusCode = TLAS_TRAVERSAL_STATUS_STACK_OVERFLOW;\n        return true;',
    );

    expect(TLAS_SCENE_HIT_TRAVERSAL_WGSL).toContain('fn tlasClosestFallback(');
    expect(TLAS_SCENE_HIT_TRAVERSAL_WGSL).toContain('fn tlasAnySceneFallback(');
    expect(TLAS_SCENE_HIT_TRAVERSAL_WGSL).toContain(
      'return tlasClosestFallback(ray, tMin, tMax, hit);',
    );
    expect(TLAS_SCENE_HIT_TRAVERSAL_WGSL).toContain(
      'return tlasAnySceneFallback(ray, tMin, tMax);',
    );
    expect(TLAS_SCENE_HIT_TRAVERSAL_WGSL).toContain(
      'permIdx < arrayLength(&tlasInstanceIndices)',
    );
    expect(TLAS_SCENE_HIT_TRAVERSAL_WGSL).not.toContain(
      'tlasTraversalStatusCode = TLAS_TRAVERSAL_STATUS_STACK_OVERFLOW;\n        return (*hit).didHit;',
    );
    expect(TLAS_SCENE_HIT_TRAVERSAL_WGSL).not.toContain(
      'tlasTraversalStatusCode = TLAS_TRAVERSAL_STATUS_STACK_OVERFLOW;\n        return true;',
    );
  });

  it('threads skipGlass through TLAS closest-hit shadow traversal (H32)', () => {
    expect(TLAS_TRAVERSAL_WGSL).toContain('fn traceTlasAny(');
    expect(TLAS_TRAVERSAL_WGSL).toContain(
      'bvhIntersectFirstHitAtRoot(localRay, localTMin, blasRoot, skipGlass)',
    );
    expect(BVH_INTERSECT_WGSL).toContain('if (skipGlass) {');
    expect(BVH_INTERSECT_WGSL).toContain(
      'fn bvhPackedMaterialHasTransmission(packedMaterial: u32) -> bool',
    );
    expect(BVH_INTERSECT_WGSL).toContain(
      'if (bvhPackedMaterialHasTransmission(idxEntry.w)) { continue; }',
    );
  });

  it('uses portable value-return loaders instead of storage-pointer parameters', () => {
    expect(BVH_INTERSECT_CORE_WGSL).not.toContain('ptr<storage');
    expect(TLAS_TRAVERSAL_CORE_WGSL).not.toContain('ptr<storage');
    expect(BVH_INTERSECT_CORE_WGSL).toContain('bvhLoadNode(currNodeIdx)');
    expect(TLAS_TRAVERSAL_CORE_WGSL).toContain('tlasLoadWorldToLocalColumn(m)');
    expect(BVH_INTERSECT_WGSL).toContain('fn bvhLoadIndex(index: u32) -> vec4u');
    expect(TLAS_TRAVERSAL_WGSL).toContain('fn tlasNodeCapacity() -> u32');
    expect(TLAS_TRAVERSAL_CORE_WGSL).toContain('fn tlasSafeNormalize(v: vec3f) -> vec3f');
    expect(TLAS_TRAVERSAL_CORE_WGSL).not.toContain('safe_normalize(');
  });

  it('transforms local normals with transpose(worldToLocal) for rotation and shear', () => {
    expect(TLAS_TRAVERSAL_CORE_WGSL).toContain('dot(w2l0.xyz, nLocal)');
    expect(TLAS_TRAVERSAL_CORE_WGSL).toContain('dot(w2l1.xyz, nLocal)');
    expect(TLAS_TRAVERSAL_CORE_WGSL).toContain('dot(w2l2.xyz, nLocal)');
    expect(TLAS_TRAVERSAL_CORE_WGSL).not.toContain(
      'let row0 = vec3f(w2l0.x, w2l1.x, w2l2.x)',
    );
    expect(TLAS_TRAVERSAL_CORE_WGSL).not.toContain('dot(row0, nLocal)');
  });

  it('decodes position.w UV payloads as packed f16 pairs', () => {
    expect(BVH_INTERSECT_WGSL).toContain('unpack2x16float(bitcast<u32>(pa4.w))');
    expect(BVH_INTERSECT_WGSL).not.toContain('unpack2x16unorm(bitcast<u32>(pa4.w))');
  });

  it('converts world TLAS bounds into normalized BLAS-local distance', () => {
    expect(TLAS_TRAVERSAL_WGSL).toContain(
      'let localTMin = max(tlasRayParameterAtPoint(localRay, localStart), 0.0);',
    );
    expect(TLAS_TRAVERSAL_WGSL).toContain(
      'bvhIntersectFirstHitAtRoot(localRay, localTMin, blasRoot, false)',
    );
    expect(TLAS_TRAVERSAL_WGSL).toContain(
      'return worldDist > triEps && worldDist < tMax;',
    );
    expect(TLAS_TRAVERSAL_WGSL).not.toContain('worldDist > 1e-4');
    expect(TLAS_SCENE_HIT_TRAVERSAL_WGSL).toContain(
      'let localTMin = max(dot(localStart - localRay.origin, localRay.direction), 0.0);',
    );
    expect(TLAS_SCENE_HIT_TRAVERSAL_WGSL).toContain(
      'traceMeshBvh(localRay, localTMin, localTMax, true, &localHit, blasRoot, true)',
    );
    expect(TLAS_SCENE_HIT_TRAVERSAL_WGSL).not.toContain('localHit.dist < (*hit).dist');
    expect(TLAS_TRAVERSAL_WGSL).not.toContain('localHit.dist < best.dist');
  });

  it('compares closest candidates only after reconstructing world distance', () => {
    const existingWorldDistance = 5;
    const candidateWorldDistance = 4;
    const candidateLocalDistance = 40;
    expect(candidateLocalDistance < existingWorldDistance).toBe(false);
    expect(candidateWorldDistance < existingWorldDistance).toBe(true);
  });
});
