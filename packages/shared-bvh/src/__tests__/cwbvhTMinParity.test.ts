import { describe, expect, it } from 'vitest';
import {
  buildCompressedWideBvh,
  intersectCompressedWideBvhAnyHit,
  intersectCompressedWideBvhFirstHit,
  type CwbvhRay,
} from '../index.js';

/**
 * Pins the tMin / comparator parity invariant between the two CWBVH
 * traversals: anyHit must NEVER report a miss where firstHit reports a hit at
 * the same epsilon boundary. Before the 2026-07-20 reconcile, firstHit used
 * `tMin = triEps` with `hit.t >= tMin` while anyHit used `tMin = 1e-4` with
 * `hit.t > tMin`, so a hit exactly at tMin was a hit in firstHit but a miss in
 * anyHit — shadow rays under-occluded relative to the closest-hit geometry.
 */

// A single triangle lying in the z = zPlane plane, large enough that a
// +z ray from the origin strikes its interior at exactly t = zPlane.
function buildSingleTriangleAt(zPlane: number) {
  const positions = new Float32Array([
    -1, -1, zPlane, 0,
     3, -1, zPlane, 0,
    -1,  3, zPlane, 0,
  ]);
  const indices = new Uint32Array([0, 1, 2, 0]);
  const triMaterialIds = new Uint32Array([0]);
  const cwbvh = buildCompressedWideBvh(positions, indices, triMaterialIds);
  return { cwbvh, positions };
}

const rayAlongZ: CwbvhRay = { origin: [0, 0, 0], direction: [0, 0, 1] };

describe('CWBVH tMin/comparator parity (firstHit vs anyHit)', () => {
  it('a hit exactly at an explicit tMin is a hit in both traversals', () => {
    const zPlane = 5;
    const { cwbvh, positions } = buildSingleTriangleAt(zPlane);

    // tMin set exactly to the hit distance: the boundary case.
    const first = intersectCompressedWideBvhFirstHit(cwbvh, positions, rayAlongZ, { tMin: zPlane });
    const any = intersectCompressedWideBvhAnyHit(cwbvh, positions, rayAlongZ, { tMin: zPlane });

    expect(first.didHit).toBe(true);
    expect(first.dist).toBeCloseTo(zPlane, 6);
    // Invariant: firstHit-hit ⇒ anyHit-hit at the same boundary.
    expect(any).toBe(true);
  });

  it('a hit at the old anyHit default tMin (1e-4) no longer diverges at defaults', () => {
    // Triangle exactly at z = 1e-4, the OLD anyHit default tMin. With the pre-
    // reconcile code, firstHit (default tMin = triEps = 1e-5, `>=`) reported a
    // hit while anyHit (default tMin = 1e-4, `>`) reported a miss — the exact
    // divergence this fix closes. Both now derive tMin from triEps with `>=`.
    const zPlane = 1e-4;
    const { cwbvh, positions } = buildSingleTriangleAt(zPlane);

    const first = intersectCompressedWideBvhFirstHit(cwbvh, positions, rayAlongZ);
    const any = intersectCompressedWideBvhAnyHit(cwbvh, positions, rayAlongZ);

    expect(first.didHit).toBe(true);
    // Invariant: firstHit-hit ⇒ anyHit-hit at the same default boundary.
    expect(any).toBe(true);
  });
});
