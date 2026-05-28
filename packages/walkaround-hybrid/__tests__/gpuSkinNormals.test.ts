/**
 * Theme 2 — GPU normal-skinning WGSL structural pins.
 *
 * Runtime GPU execution is deferred (no device in CI — see report doc entry),
 * so these tests pin the structural properties the with-normals LBS kernel
 * must have:
 *   - it binds a skinned-normal output buffer (binding 7),
 *   - it reads the rest normals (binding 2) — closing the "binds restNormals
 *     but never skins normals" gap the old kernel had,
 *   - it transforms the normal via an inverse-transpose of the blended skin
 *     linear part (not the plain matrix), and composes the world matrix's
 *     inverse-transpose when world-applying,
 *   - the position math is unchanged from the position-only kernel.
 */

import { describe, expect, it } from 'vitest';

import {
  GPU_SKIN_BVH_WGSL,
  GPU_SKIN_BVH_WITH_NORMALS_WGSL,
} from '../src/skin/gpuSkinBvh.wgsl.js';

describe('GPU normal-skinning WGSL', () => {
  it('with-normals kernel binds a skinned-normal output at binding 7', () => {
    expect(GPU_SKIN_BVH_WITH_NORMALS_WGSL).toContain(
      '@group(0) @binding(7) var<storage, read_write> skinnedNormals: array<vec4f>',
    );
    // The old position-only kernel did NOT have a normal output.
    expect(GPU_SKIN_BVH_WGSL).not.toContain('skinnedNormals');
  });

  it('with-normals kernel actually reads the rest normals (binding 2)', () => {
    // The position-only kernel bound restNormals but never read it.
    expect(GPU_SKIN_BVH_WGSL).not.toContain('restNormals[vi]');
    expect(GPU_SKIN_BVH_WITH_NORMALS_WGSL).toContain('restNormals[vi]');
  });

  it('with-normals kernel transforms the normal by an inverse-transpose', () => {
    expect(GPU_SKIN_BVH_WITH_NORMALS_WGSL).toContain('fn mat3InverseTranspose');
    // Normal is produced by applying the inverse-transpose matrix to the rest
    // normal, not the plain blended matrix.
    expect(GPU_SKIN_BVH_WITH_NORMALS_WGSL).toContain('mat3InverseTranspose(col0, col1, col2)');
    expect(GPU_SKIN_BVH_WITH_NORMALS_WGSL).toMatch(/var outN = nt \* rn/);
  });

  it('with-normals kernel composes the world inverse-transpose when applyWorld', () => {
    expect(GPU_SKIN_BVH_WITH_NORMALS_WGSL).toContain('mat3InverseTranspose(w0, w1, w2)');
    expect(GPU_SKIN_BVH_WITH_NORMALS_WGSL).toMatch(/outN = wnt \* outN/);
  });

  it('with-normals kernel normalizes the output normal and guards degeneracy', () => {
    expect(GPU_SKIN_BVH_WITH_NORMALS_WGSL).toContain('let nlen = length(outN)');
    expect(GPU_SKIN_BVH_WITH_NORMALS_WGSL).toContain('nlen > 1e-12');
  });

  it('with-normals kernel skins normals mesh-local (vi), positions merged (outIdx)', () => {
    // Positions go to the shared merged buffer at baseVertex + vi.
    expect(GPU_SKIN_BVH_WITH_NORMALS_WGSL).toContain('bvhPositions[outIdx] = vec4f(outPos, uvPack)');
    // Normals go to the per-mesh buffer at the mesh-local index.
    expect(GPU_SKIN_BVH_WITH_NORMALS_WGSL).toContain('skinnedNormals[vi] = vec4f(safeN, 0.0)');
  });

  it('position math is preserved between the two kernels', () => {
    // Both accumulate sp via the weighted bone-matrix product and apply the
    // world matrix the same way.
    for (const src of [GPU_SKIN_BVH_WGSL, GPU_SKIN_BVH_WITH_NORMALS_WGSL]) {
      expect(src).toContain('sp = sp + wi * p4');
      expect(src).toContain('outPos = (skinParams.matrixWorld * sp).xyz');
    }
  });
});
