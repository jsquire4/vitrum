import { describe, expect, it } from 'vitest';
import {
  buildCompressedWideBvh,
  intersectCompressedWideBvhAnyHit,
  intersectCompressedWideBvhFirstHit,
  type CwbvhRay,
} from '../index.js';

/**
 * Pins the canonical open ray interval `(tMin, tMax)` in both CPU CWBVH
 * traversals. The active pt-webgpu binary and wide WGSL paths use the same
 * strict comparisons, so exact-boundary results must not diverge by backend.
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
  it('rejects a hit exactly at tMin and accepts it immediately above tMin in both traversals', () => {
    const zPlane = 5;
    const { cwbvh, positions } = buildSingleTriangleAt(zPlane);

    const firstAtBoundary = intersectCompressedWideBvhFirstHit(
      cwbvh, positions, rayAlongZ, { tMin: zPlane },
    );
    const anyAtBoundary = intersectCompressedWideBvhAnyHit(
      cwbvh, positions, rayAlongZ, { tMin: zPlane },
    );
    expect(firstAtBoundary.didHit).toBe(false);
    expect(anyAtBoundary).toBe(false);

    const firstInside = intersectCompressedWideBvhFirstHit(
      cwbvh, positions, rayAlongZ, { tMin: zPlane - 1e-4 },
    );
    const anyInside = intersectCompressedWideBvhAnyHit(
      cwbvh, positions, rayAlongZ, { tMin: zPlane - 1e-4 },
    );
    expect(firstInside.didHit).toBe(true);
    expect(firstInside.dist).toBeCloseTo(zPlane, 6);
    expect(anyInside).toBe(true);
  });

  it('rejects a hit exactly at tMax in both traversals', () => {
    const zPlane = 5;
    const { cwbvh, positions } = buildSingleTriangleAt(zPlane);

    const first = intersectCompressedWideBvhFirstHit(cwbvh, positions, rayAlongZ, { tMax: zPlane });
    const any = intersectCompressedWideBvhAnyHit(cwbvh, positions, rayAlongZ, { tMax: zPlane });

    expect(first.didHit).toBe(false);
    expect(any).toBe(false);
  });
});
