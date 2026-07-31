/**
 * safeInvDir slab-test pin (mirror of the WGSL helper in
 * `src/wgsl/bvhIntersect.wgsl.ts:119`).
 *
 * The WGSL `safeInvDir` lives only as a string; this CPU mirror
 * exists to lock in the *intended* behavior so a future revision can't
 * silently regress it. Specifically the test "outside-slab axis-aligned
 * ray rejects" pins the bug that shipped pre-2026-05-19: the earlier
 * WGSL used `sign(d.x) * 1e30`, but `sign(0) == 0` in WGSL — so an
 * exact-zero direction yielded inv-component `0`, collapsing the X
 * slab's t0/t1 to 0 regardless of origin position. Slab tests passed
 * (tNear == tFar == 0) even when the ray's origin was outside the AABB
 * on the parallel axis. The current helper uses the largest finite f32 for
 * exact zero and saturates overflowing reciprocals to that same bound. This
 * keeps the definite sign while preserving the full finite reciprocal range.
 */

import { describe, expect, it } from 'vitest';

const SENTINEL = 3.402823e38;

/** Mirror of the WGSL safeInvDir post-fix. d.x = 0 → +SENTINEL. */
function safeInvDir(d: [number, number, number]): [number, number, number] {
  function safeRecip(v: number): number {
    if (v === 0) {
      return v >= 0 ? SENTINEL : -SENTINEL;
    }
    return Math.max(-SENTINEL, Math.min(SENTINEL, 1.0 / v));
  }
  return [safeRecip(d[0]), safeRecip(d[1]), safeRecip(d[2])];
}

function slabTest(
  origin: [number, number, number],
  invDir: [number, number, number],
  bMin: [number, number, number],
  bMax: [number, number, number],
): { tNear: number; tFar: number } {
  let tNear = -Infinity;
  let tFar = Infinity;
  for (let i = 0; i < 3; i++) {
    const t0 = (bMin[i]! - origin[i]!) * invDir[i]!;
    const t1 = (bMax[i]! - origin[i]!) * invDir[i]!;
    tNear = Math.max(tNear, Math.min(t0, t1));
    tFar = Math.min(tFar, Math.max(t0, t1));
  }
  return { tNear, tFar };
}

describe('safeInvDir mirror — post-fix WGSL semantics', () => {
  it('exact-zero direction component yields +SENTINEL (post-2026-05-19 fix)', () => {
    const inv = safeInvDir([0, 1, 0]);
    expect(inv[0]).toBe(SENTINEL);  // not 0 — the bug we just fixed
    expect(inv[1]).toBe(1);
    expect(inv[2]).toBe(SENTINEL);
  });

  it('negative reciprocal overflow saturates to -f32 max', () => {
    expect(safeInvDir([-1e-39, 1, 0])[0]).toBe(-SENTINEL);
  });

  it('finite tiny direction components retain their reciprocal magnitude', () => {
    const inv = safeInvDir([-1e-31, 1, 0]);
    expect(inv[0]).toBe(-1e31);
    expect(inv[1]).toBe(1);
    expect(inv[2]).toBe(SENTINEL);
  });

  it('axis-aligned ray with origin INSIDE the slab → slab test passes', () => {
    // ray (0, 0, 0) → +Y, AABB [0,1]^3, origin inside AABB.
    const inv = safeInvDir([0, 1, 0]);
    const { tNear, tFar } = slabTest([0, 0, 0], inv, [0, 0, 0], [1, 1, 1]);
    expect(Number.isNaN(tNear)).toBe(false);
    expect(Number.isNaN(tFar)).toBe(false);
    expect(tNear).toBeLessThanOrEqual(tFar);
    expect(tFar).toBeGreaterThanOrEqual(0);
  });

  it('REGRESSION: axis-aligned ray with origin OUTSIDE the slab on the parallel axis → slab test rejects', () => {
    // Ray origin (5, 0, 0), direction (0, 1, 0). x=5 is OUTSIDE x∈[0,1].
    // Pre-fix WGSL `sign(0)*1e30 = 0` would have given inv.x = 0 →
    // t0/t1 for X collapse to 0, slab test "passes" with tNear=tFar=0
    // (false positive — the traversal would descend into this node and
    // waste time on ray-triangle tests for a ray that geometrically
    // misses the AABB entirely). Post-fix: inv.x = SENTINEL → t0/t1 for
    // X are both ~-5e30, tFar < 0, slab correctly rejects.
    const inv = safeInvDir([0, 1, 0]);
    const { tFar } = slabTest([5, 0, 0], inv, [0, 0, 0], [1, 1, 1]);
    expect(tFar).toBeLessThan(0);  // slab rejects: AABB entirely behind effective ray entry
  });

  it('REGRESSION: outside-slab from below (origin.x < bMin.x) → slab test rejects', () => {
    // Origin (-5, 0, 0), direction (0, 1, 0). x=-5 below AABB's x range.
    const inv = safeInvDir([0, 1, 0]);
    const { tNear, tFar } = slabTest([-5, 0, 0], inv, [0, 0, 0], [1, 1, 1]);
    // tNear should push past tFar — both ends sit beyond the AABB on x.
    expect(tNear).toBeGreaterThan(tFar);
  });
});
