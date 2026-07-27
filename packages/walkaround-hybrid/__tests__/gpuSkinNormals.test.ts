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
  GPU_SKIN_BVH_WITH_NORMALS_WGSL,
} from '../src/skin/gpuSkinBvh.wgsl.js';

describe('GPU normal-skinning WGSL', () => {
  it('with-normals kernel binds a skinned-normal output at binding 7', () => {
    expect(GPU_SKIN_BVH_WITH_NORMALS_WGSL).toContain(
      '@group(0) @binding(7) var<storage, read_write> skinnedNormals: array<vec4f>',
    );
    expect(GPU_SKIN_BVH_WITH_NORMALS_WGSL).not.toContain('skinnedNormals[vi]');
  });

  it('with-normals kernel actually reads the rest normals (binding 2)', () => {
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

  it('WS1 — skins BOTH positions AND normals into the shared merged buffers at outIdx', () => {
    // Positions go to the shared merged buffer at baseVertex + vi.
    expect(GPU_SKIN_BVH_WITH_NORMALS_WGSL).toContain('bvhPositions[outIdx] = vec4f(outPos, uvPack)');
    // WS1 (2026-05-29) — normals now write into the SHARED merged bvh_normal
    // buffer at the SAME world-space slot (outIdx), so the smooth-shading-
    // normal blend consumes the skinned normal. The old mesh-local `vi` write
    // (dropped by applyGpuSkinnedRefit) is gone.
    expect(GPU_SKIN_BVH_WITH_NORMALS_WGSL).toContain(
      'let uv1Pack = skinnedNormals[outIdx].w',
    );
    expect(GPU_SKIN_BVH_WITH_NORMALS_WGSL).toContain(
      'skinnedNormals[outIdx] = vec4f(safeN, uv1Pack)',
    );
    expect(GPU_SKIN_BVH_WITH_NORMALS_WGSL).not.toContain(
      'skinnedNormals[vi] = vec4f(safeN, uv1Pack)',
    );
  });

  it('with-normals kernel still skins positions into bvhPositions', () => {
    expect(GPU_SKIN_BVH_WITH_NORMALS_WGSL).toContain('sp = sp + wi * p4');
    expect(GPU_SKIN_BVH_WITH_NORMALS_WGSL).toContain('outPos = (skinParams.matrixWorld * sp).xyz');
  });
});
