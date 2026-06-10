/**
 * H34-a unit tests — 16-bit leaf count guard
 *
 * Verifies that buildArrayBvh throws a clear error when a BLAS leaf's triangle
 * count exceeds 0xFFFF, and that buildTlas throws for an oversized instance leaf.
 *
 * The guard fires at the forced-leaf fallback path (when SAH cannot split
 * co-planar centroids), which is reachable without needing 65536+ real triangles:
 * we patch maxLeafTriangles to a huge number so the first (and only) leaf gets
 * the full set, then check the guard at the threshold boundary.
 */

import { describe, expect, it } from 'vitest';
import { buildArrayBvh } from '../buildArrayBvh.js';
import { buildTlas } from '../tlas.js';

// Build a minimal positions / indices pair with N triangles, all sharing the
// same centroid (all degenerate co-planar) so the SAH forced-leaf path is taken.
function makeCoplanarTriangles(n: number): { positions: Float32Array; indices: Uint32Array; matIds: Uint32Array } {
  // 3 unique vertices in the XY plane — all N triangles reference the same three.
  const positions = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]);  // stride 4
  const indices = new Uint32Array(n * 4);
  const matIds = new Uint32Array(n);
  for (let t = 0; t < n; t += 1) {
    indices[t * 4] = 0;
    indices[t * 4 + 1] = 1;
    indices[t * 4 + 2] = 2;
    indices[t * 4 + 3] = 0;
  }
  return { positions, indices, matIds };
}

describe('H34-a: buildArrayBvh 16-bit leaf count guard', () => {
  it('does NOT throw for a leaf of exactly 0xFFFF triangles', () => {
    // Build a leaf of count = 0xFFFF using a very high maxLeafTriangles so the
    // small-leaf path is taken without SAH splitting.
    const N = 0xffff;
    const { positions, indices, matIds } = makeCoplanarTriangles(N);
    expect(() =>
      buildArrayBvh(positions, indices, matIds, {
        positionStride: 4,
        indexStride: 4,
        maxLeafTriangles: N + 1,  // ensure leaf path (not SAH)
      }),
    ).not.toThrow();
  });

  it('throws a clear error for a leaf of 0x10000 (65536) triangles', () => {
    const N = 0x10000;
    const { positions, indices, matIds } = makeCoplanarTriangles(N);
    expect(() =>
      buildArrayBvh(positions, indices, matIds, {
        positionStride: 4,
        indexStride: 4,
        maxLeafTriangles: N + 1,
      }),
    ).toThrow(/16-bit limit/);
  });
});

// Helper: build a minimal TLAS with N instances all at the same location so
// the SAH degenerate forced-leaf path fires.
function makeFlatTlasInstances(n: number) {
  const wToL = new Float32Array(16);
  wToL[0] = 1; wToL[5] = 1; wToL[10] = 1; wToL[15] = 1;  // identity
  return Array.from({ length: n }, (_, i) => ({
    blasId: i,
    aabbMin: [0, 0, 0] as [number, number, number],
    aabbMax: [1, 1, 1] as [number, number, number],
    worldToLocal: wToL,
  }));
}

describe('H34-a: buildTlas 16-bit leaf count guard', () => {
  it('throws a clear error when a TLAS leaf instance count exceeds 0xFFFF', () => {
    const instances = makeFlatTlasInstances(0x10000);
    expect(() =>
      buildTlas(instances, { maxLeafInstances: 0x10000 + 1 }),
    ).toThrow(/16-bit limit/);
  });
});
