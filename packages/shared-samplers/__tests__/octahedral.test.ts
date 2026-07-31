/**
 * Octahedral encode/decode round-trip tests — Item 33-E.
 *
 * TypeScript mirrors of:
 *   packages/shared-samplers/src/wgsl/octahedralCore.wgsl.ts  lines 7–22
 * (Cigolle et al. JCGT 2014 §3 eq. 2)
 *
 * Encoding convention: output is in [-1, 1]² (not [0, 1]²).
 *
 * South-pole singularity (Item #39) is fixed: the lower-hemisphere fold now
 * uses `select(-1.0, 1.0, n.x >= 0.0)` (WGSL) / `n[0] >= 0 ? 1 : -1` (TS)
 * so that sign(0) maps to +1 per Cigolle et al. 2014 §A.1.
 */

import { describe, it, expect } from 'vitest';

// ---------------------------------------------------------------------------
// TypeScript mirrors of octahedralCore.wgsl.ts:7–22
// ---------------------------------------------------------------------------

/** Mirror of octahedralCore.wgsl.ts:7–13 (octEncode).
 *  Cigolle et al. 2014 §3 eq. 2.
 *  Item #39: use n[0] >= 0 ? 1 : -1 (maps 0 → +1) to match the WGSL
 *  select(-1.0, 1.0, n.x >= 0.0) fix; Math.sign(0) = 0 is wrong here. */
function octEncodeTS(v: [number, number, number]): [number, number] {
  const scale = Math.max(Math.abs(v[0]), Math.abs(v[1]), Math.abs(v[2]));
  if (!(scale > 0) || !Number.isFinite(scale)) return [0, 0];
  const scaled: [number, number, number] = [
    v[0] / scale,
    v[1] / scale,
    v[2] / scale,
  ];
  const denom = Math.abs(scaled[0]) + Math.abs(scaled[1]) + Math.abs(scaled[2]);
  const n: [number, number, number] = [
    scaled[0] / denom,
    scaled[1] / denom,
    scaled[2] / denom,
  ];
  if (n[2] >= 0) {
    return [n[0], n[1]];
  }
  // Lower-hemisphere fold — Cigolle 2014 §A.1 / Item #39:
  // 0 must map to +1, not 0. Use ternary to match WGSL select() semantics.
  const sx = n[0] >= 0 ? 1 : -1;
  const sy = n[1] >= 0 ? 1 : -1;
  return [
    (1.0 - Math.abs(n[1])) * sx,
    (1.0 - Math.abs(n[0])) * sy,
  ];
}

/** Mirror of octahedralCore.wgsl.ts:15–22 (octDecode).
 *  Cigolle et al. 2014 §3 eq. 2 inverse. */
function octDecodeTS(oct: [number, number]): [number, number, number] {
  // n = vec3f(oct, 1.0 - abs(oct.x) - abs(oct.y))
  const nz = 1.0 - Math.abs(oct[0]) - Math.abs(oct[1]);
  let nx = oct[0];
  let ny = oct[1];
  if (nz < 0.0) {
    // Mirror WGSL select() semantics so 0 maps to +1.
    const origNx = nx;
    const sx = origNx >= 0 ? 1 : -1;
    const sy = ny >= 0 ? 1 : -1;
    nx = (1.0 - Math.abs(ny)) * sx;
    ny = (1.0 - Math.abs(origNx)) * sy;
  }
  const len = Math.hypot(nx, ny, nz);
  return [nx / len, ny / len, nz / len];
}

// ---------------------------------------------------------------------------
// Deterministic LCG RNG (Numerical Recipes parameters)
// ---------------------------------------------------------------------------

function makeLcg(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(1664525, s) + 1013904223) >>> 0;
    return s / 0x100000000;
  };
}

/** Marsaglia rejection method — uniform sphere sample. */
function uniformSphereSample(rand: () => number): [number, number, number] {
  while (true) {
    const x = rand() * 2.0 - 1.0;
    const y = rand() * 2.0 - 1.0;
    const r2 = x * x + y * y;
    if (r2 >= 1.0) continue;
    const z = 1.0 - 2.0 * r2;
    const scale = 2.0 * Math.sqrt(1.0 - r2);
    return [x * scale, y * scale, z];
  }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('octahedral encode/decode (33-E)', () => {

  // 1. Encode → decode round-trip identity for generic sphere samples
  it('round-trip identity for 1000 uniform sphere samples', () => {
    const rand = makeLcg(0xdeadbeef);
    let failures = 0;
    for (let i = 0; i < 1000; i++) {
      const v = uniformSphereSample(rand);
      const enc = octEncodeTS(v);
      const dec = octDecodeTS(enc);
      const dot = v[0] * dec[0] + v[1] * dec[1] + v[2] * dec[2];
      if (dot <= 0.9999) {
        failures++;
        expect(dot, `sample ${i}: dot=${dot}, v=[${v}], enc=[${enc}], dec=[${dec}]`).toBeGreaterThan(0.9999);
      }
    }
    expect(failures).toBe(0);
  });

  // 2. Bijection coverage — encoded values fall in [-1, 1]² and span the domain
  //    NOTE: Octahedral projection is NOT area-uniform over [-1,1]² — the sphere
  //    area element distorts (corners of the square map to the equatorial belt with
  //    higher density than the poles). A chi-squared uniformity test would always
  //    fail. The correct assertions are: (a) all values are in bounds, and (b) every
  //    cell of a coarse grid receives at least one sample (no dead zones = no missing
  //    regions in the bijection).
  it('bijection coverage: encoded values in [-1,1]² with no dead zones in 16×16 grid', () => {
    const GRID = 16;
    const N = 10_000;
    const rand = makeLcg(0xc0ffee42);
    const occupied = new Uint8Array(GRID * GRID);

    for (let i = 0; i < N; i++) {
      const v = uniformSphereSample(rand);
      const [ex, ey] = octEncodeTS(v);

      // Verify bounds — encoding outputs [-1, 1]²
      expect(ex, `sample ${i}: ex=${ex} out of bounds`).toBeGreaterThanOrEqual(-1.0 - 1e-9);
      expect(ex, `sample ${i}: ex=${ex} out of bounds`).toBeLessThanOrEqual(1.0 + 1e-9);
      expect(ey, `sample ${i}: ey=${ey} out of bounds`).toBeGreaterThanOrEqual(-1.0 - 1e-9);
      expect(ey, `sample ${i}: ey=${ey} out of bounds`).toBeLessThanOrEqual(1.0 + 1e-9);

      // Map [-1,1] → [0, GRID)
      const col = Math.min(GRID - 1, Math.floor(((ex + 1.0) / 2.0) * GRID));
      const row = Math.min(GRID - 1, Math.floor(((ey + 1.0) / 2.0) * GRID));
      occupied[row * GRID + col] = 1;
    }

    // Every cell should be reachable — no dead zones means no broken bijection regions
    const deadCells = occupied.filter(v => v === 0).length;
    expect(deadCells, `${deadCells} of ${GRID * GRID} grid cells had zero samples (dead zone in bijection)`).toBe(0);
  });

  // 3. Boundary handling
  describe('boundary cases', () => {
    const EPS = 1e-5;

    function roundTrip(v: [number, number, number]): [number, number, number] {
      return octDecodeTS(octEncodeTS(v));
    }

    function dotV(a: [number, number, number], b: [number, number, number]): number {
      return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    }

    function nearEq(a: [number, number, number], b: [number, number, number], tol = EPS): boolean {
      return (
        Math.abs(a[0] - b[0]) < tol &&
        Math.abs(a[1] - b[1]) < tol &&
        Math.abs(a[2] - b[2]) < tol
      );
    }

    it('pure +Z pole round-trips to (0,0,1)', () => {
      const dec = roundTrip([0, 0, 1]);
      expect(nearEq(dec, [0, 0, 1])).toBe(true);
    });

    // Item #39 fix landed: south pole now round-trips correctly.
    it('south pole (0,0,-1) round-trips to (0,0,-1)', () => {
      const dec = roundTrip([0, 0, -1]);
      const dot = dotV([0, 0, -1], dec);
      expect(dot).toBeGreaterThan(0.9999);
    });

    it('+X axis direction round-trips correctly', () => {
      const v: [number, number, number] = [1, 0, 0];
      expect(dotV(v, roundTrip(v))).toBeGreaterThan(0.9999);
    });

    it('encodes a tiny non-zero direction identically to its unit direction', () => {
      const tiny = octEncodeTS([1e-30, -2e-30, 3e-30]);
      const unitScale = octEncodeTS([1, -2, 3]);
      expect(tiny[0]).toBeCloseTo(unitScale[0], 15);
      expect(tiny[1]).toBeCloseTo(unitScale[1], 15);
      expect(octEncodeTS([0, 0, 0])).toEqual([0, 0]);
    });

    it('-X axis direction round-trips correctly', () => {
      const v: [number, number, number] = [-1, 0, 0];
      expect(dotV(v, roundTrip(v))).toBeGreaterThan(0.9999);
    });

    it('+Y axis direction round-trips correctly', () => {
      const v: [number, number, number] = [0, 1, 0];
      expect(dotV(v, roundTrip(v))).toBeGreaterThan(0.9999);
    });

    it('-Y axis direction round-trips correctly', () => {
      const v: [number, number, number] = [0, -1, 0];
      expect(dotV(v, roundTrip(v))).toBeGreaterThan(0.9999);
    });

    it('+Z axis direction round-trips correctly', () => {
      const v: [number, number, number] = [0, 0, 1];
      expect(dotV(v, roundTrip(v))).toBeGreaterThan(0.9999);
    });

    // -Z axis is the same case as the south-pole test above (Item #39).
    // Kept here so the axis-aligned suite is symmetric.
    it('-Z axis direction round-trips correctly', () => {
      const v: [number, number, number] = [0, 0, -1];
      expect(dotV(v, roundTrip(v))).toBeGreaterThan(0.9999);
    });

    it('z=0 boundary direction (1,0,0) round-trips correctly', () => {
      // (1,0,0) sits on the equatorial seam — z=0 branch, not the lower fold
      const v: [number, number, number] = [1, 0, 0];
      expect(dotV(v, roundTrip(v))).toBeGreaterThan(0.9999);
    });

    it('lower-hemisphere direction with non-zero x,y round-trips correctly', () => {
      // (1/√3, 1/√3, -1/√3) — lower hemisphere with x≠0, y≠0; sign() is well-defined
      const s = 1.0 / Math.sqrt(3);
      const v: [number, number, number] = [s, s, -s];
      expect(dotV(v, roundTrip(v))).toBeGreaterThan(0.9999);
    });
  });

  // 4. Determinism — same input → same encoded value
  it('determinism: encoding the same direction twice gives identical output', () => {
    const directions: Array<[number, number, number]> = [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
      [0.577350, 0.577350, 0.577350],
      [-0.5, 0.3, -0.812404],
    ];

    for (const v of directions) {
      const e1 = octEncodeTS(v);
      const e2 = octEncodeTS(v);
      expect(e1[0]).toBe(e2[0]);
      expect(e1[1]).toBe(e2[1]);
    }
  });

});
