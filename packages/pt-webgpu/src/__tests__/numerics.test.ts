/**
 * 33-H Numerics tests: Beer-Lambert attenuation and safeInvDir slab-test math.
 *
 * These tests validate assumptions behind sweep-2026-05-11 Items 18 and 28:
 * - Item 18: removing the `min(hit.dist, 32.0)` Beer-Lambert clamp is numerically safe.
 * - Item 28: a safeInvDir helper prevents 0*Inf=NaN in axis-aligned ray–AABB slab tests.
 *
 * The `safeInvDir` helper below is a TypeScript model used to reason about
 * the slab-test math from JS — there is no shared TS implementation to
 * re-export (the production `safeInvDir` lives only as a WGSL string, the
 * `SAFE_INV_DIR_WGSL` export in `shared-bvh/src/wgsl/bvhIntersect.wgsl.ts`).
 * This mirror's
 * exact-zero handling now matches the WGSL post-fix-2026-05-19:
 * both use ±sentinel (this mirror uses ±1e20, WGSL uses ±1e30 — value
 * differs, sign convention matches). The canonical CPU mirror with
 * matching ±1e30 semantics lives in `@vitrum/shared-bvh`'s
 * `__tests__/safeInvDir.test.ts`; this file's mirror is older and
 * intentionally local.
 */

import { describe, expect, it } from 'vitest';

// ---------------------------------------------------------------------------
// Test-only mirror of the proposed WGSL safeInvDir (Item 28).
// Replaces a near-zero component with ±1e20 so slab tests never compute 0*Inf=NaN.
// ---------------------------------------------------------------------------
function safeInvDir(d: [number, number, number]): [number, number, number] {
  const EPS = 1e-30;
  function safeRecip(v: number): number {
    return Math.abs(v) < EPS ? Math.sign(v === 0 ? 1 : v) * 1e20 : 1.0 / v;
  }
  return [safeRecip(d[0]), safeRecip(d[1]), safeRecip(d[2])];
}

// Slab test for a unit AABB [0,1]^3, given ray origin + invDir.
// Returns { tNear, tFar } (may be ±Infinity; NaN indicates failure).
function slabTest(
  origin: [number, number, number],
  invDir: [number, number, number],
  bMin: [number, number, number] = [0, 0, 0],
  bMax: [number, number, number] = [1, 1, 1],
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

// ---------------------------------------------------------------------------
// 1. Beer-Lambert attenuation: exp(-sigmaT * d) for large d
// ---------------------------------------------------------------------------
describe('Beer-Lambert attenuation (Item 18)', () => {
  const sigmaT = 0.1;
  const distances = [1, 10, 32, 100, 10_000, 1e6];

  it.each(distances)(
    'exp(-sigmaT * %s) is finite and >= 0',
    (d) => {
      const T = Math.exp(-sigmaT * d);
      expect(Number.isFinite(T)).toBe(true);
      expect(T).toBeGreaterThanOrEqual(0);
    },
  );

  it('attenuation approaches 0 for d=1e6 (not NaN or Inf)', () => {
    const T = Math.exp(-sigmaT * 1e6);
    expect(Number.isFinite(T)).toBe(true);
    expect(T).toBeGreaterThanOrEqual(0);
    // Should be effectively 0 — much smaller than the clamped result
    const clampedT = Math.exp(-sigmaT * 32);
    expect(T).toBeLessThan(clampedT);
  });
});

// ---------------------------------------------------------------------------
// 2. safeInvDir handles axis-aligned rays without NaN
// ---------------------------------------------------------------------------
describe('safeInvDir (Item 28)', () => {
  it('axis-aligned +Y ray returns finite invDir', () => {
    const inv = safeInvDir([0, 1, 0]);
    expect(Number.isFinite(inv[0])).toBe(true);
    expect(Number.isFinite(inv[1])).toBe(true);
    expect(Number.isFinite(inv[2])).toBe(true);
    expect(Number.isNaN(inv[0])).toBe(false);
    expect(Number.isNaN(inv[2])).toBe(false);
  });

  it('near-zero x component returns finite invDir', () => {
    const inv = safeInvDir([1e-31, 1, 0]);
    for (const v of inv) {
      expect(Number.isFinite(v)).toBe(true);
      expect(Number.isNaN(v)).toBe(false);
    }
  });

  it('unit direction returns [1,1,1] unchanged', () => {
    const inv = safeInvDir([1, 1, 1]);
    expect(inv[0]).toBeCloseTo(1, 10);
    expect(inv[1]).toBeCloseTo(1, 10);
    expect(inv[2]).toBeCloseTo(1, 10);
  });

  // Slab-test math: ray along +Y from (0,0,0), bottom face of unit AABB.
  // origin.y == bMin.y == 0 and dir.y > 0, so tNear_y = (0-0)/1 = 0.
  // origin.x == 0 == bMin.x and dir.x == 0 → would produce 0*Inf=NaN without guard.
  it('axis-aligned +Y ray through unit AABB: tNear <= tFar, no NaN', () => {
    const origin: [number, number, number] = [0, 0, 0];
    const dir: [number, number, number] = [0, 1, 0];
    const inv = safeInvDir(dir);
    const { tNear, tFar } = slabTest(origin, inv);
    expect(Number.isNaN(tNear)).toBe(false);
    expect(Number.isNaN(tFar)).toBe(false);
    expect(tNear).toBeLessThanOrEqual(tFar);
  });

  it('near-zero x ray through unit AABB: tNear <= tFar, no NaN', () => {
    const origin: [number, number, number] = [0, 0, 0];
    const dir: [number, number, number] = [1e-31, 1, 0];
    const inv = safeInvDir(dir);
    const { tNear, tFar } = slabTest(origin, inv);
    expect(Number.isNaN(tNear)).toBe(false);
    expect(Number.isNaN(tFar)).toBe(false);
    expect(tNear).toBeLessThanOrEqual(tFar);
  });

  it('diagonal ray through unit AABB: tNear <= tFar, no NaN', () => {
    const origin: [number, number, number] = [-1, -1, -1];
    const dir: [number, number, number] = [1, 1, 1];
    const inv = safeInvDir(dir);
    const { tNear, tFar } = slabTest(origin, inv);
    expect(Number.isNaN(tNear)).toBe(false);
    expect(Number.isNaN(tFar)).toBe(false);
    expect(tNear).toBeLessThanOrEqual(tFar);
    expect(tFar).toBeGreaterThan(0);
  });
});
