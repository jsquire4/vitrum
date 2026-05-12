/**
 * Octahedral encode/decode round-trip tests — Item 33-E.
 *
 * TypeScript mirrors of:
 *   packages/shared-samplers/src/wgsl/octahedralCore.wgsl.ts  lines 7–22
 * (Cigolle et al. JCGT 2014 §3 eq. 2)
 *
 * WGSL sign() and JS Math.sign() both return 0 for input 0, so the
 * conventions are consistent across the seam.
 *
 * Encoding convention: output is in [-1, 1]² (not [0, 1]²).
 *
 * KNOWN BUG (surfaced by this test suite):
 *   The south pole (0, 0, -1) is degenerate. octEncode maps it to [0, 0]
 *   (via the lower-hemisphere fold: sign(0)*... = 0), which is the same
 *   encoding as the north pole (0, 0, 1). octDecode([0, 0]) reconstructs
 *   (0, 0, 1) — the wrong pole. Any direction with x=0, y=0, z<0 will
 *   exhibit this sign-collapse. Root cause: Math.sign(0) = 0 in both WGSL
 *   and JS, so the fold `(1-|n.y|)*sign(n.x)` collapses to 0 when n.x=0,
 *   regardless of the z-sign being negative. The fix (not applied here —
 *   do not touch source files) is to clamp sign(0) to +1 in the lower-
 *   hemisphere fold, e.g. `select(-1.0, 1.0, n.x >= 0.0)`.
 *   See: Cigolle et al. 2014 §A.1 — they note sign(0) must be treated as
 *   +1 to avoid this degenerate case.
 *
 *   The round-trip identity test (1000 Marsaglia samples) passes because the
 *   south-pole singularity has measure zero on the sphere and Marsaglia
 *   sampling never produces exactly (0, 0, -1).
 */

import { describe, it, expect } from 'vitest';

// ---------------------------------------------------------------------------
// TypeScript mirrors of octahedralCore.wgsl.ts:7–22
// ---------------------------------------------------------------------------

/** Mirror of octahedralCore.wgsl.ts:7–13 (octEncode).
 *  Cigolle et al. 2014 §3 eq. 2. */
function octEncodeTS(v: [number, number, number]): [number, number] {
  const denom = Math.abs(v[0]) + Math.abs(v[1]) + Math.abs(v[2]);
  const n: [number, number, number] = [v[0] / denom, v[1] / denom, v[2] / denom];
  if (n[2] >= 0) {
    return [n[0], n[1]];
  }
  // Fold lower hemisphere — mirrors WGSL: (1.0 - abs(n.yx)) * vec2f(sign(n.x), sign(n.y))
  return [
    (1.0 - Math.abs(n[1])) * Math.sign(n[0]),
    (1.0 - Math.abs(n[0])) * Math.sign(n[1]),
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
    // xy = (1.0 - abs(n.yx)) * vec2f(sign(n.x), sign(n.y))
    const origNx = nx;
    nx = (1.0 - Math.abs(ny)) * Math.sign(origNx);
    ny = (1.0 - Math.abs(origNx)) * Math.sign(ny);
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

    // TODO(M4-#39): Octahedral south-pole singularity.
    // (0,0,-1) currently encodes to [0,0] because sign(0)=0 collapses
    // the lower-hemisphere fold; [0,0] decodes to (0,0,+1). South pole
    // round-trips to north pole. Fix: in octEncode, replace `sign(n.x)`
    // with `select(-1.0, 1.0, n.x >= 0.0)` (and same for n.y).
    // Cigolle et al. 2014 §A.1 documents this sign(0) gotcha.
    // This `it.fails` will flip to passing once the source fix lands.
    it.fails('south pole (0,0,-1) round-trips to (0,0,-1) [pending #39 fix]', () => {
      const dec = roundTrip([0, 0, -1]);
      const dot = dotV([0, 0, -1], dec);
      expect(dot).toBeGreaterThan(0.9999);
    });

    it('+X axis direction round-trips correctly', () => {
      const v: [number, number, number] = [1, 0, 0];
      expect(dotV(v, roundTrip(v))).toBeGreaterThan(0.9999);
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

    // -Z axis is the same case as the south-pole `it.fails` above; covered
    // by the M4-#39 fix. Kept here so the axis-aligned suite is symmetric.
    it.fails('-Z axis direction round-trips correctly [pending #39 fix]', () => {
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
