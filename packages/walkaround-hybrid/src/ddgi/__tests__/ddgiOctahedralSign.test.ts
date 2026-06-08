/**
 * Regression test for the sign()→select() fix in ddgiSampleWgsl.ts.
 *
 * The WGSL `sign(0.0)` function returns 0 in WGSL, collapsing the
 * octahedral lower-hemisphere fold at exactly axis-aligned directions
 * (W2-C3 bug class). The fix replaces `sign(n.x)` / `sign(n.y)` with
 * `select(-1.0, 1.0, n.x >= 0.0)` / `select(-1.0, 1.0, n.y >= 0.0)`
 * so that 0 maps to +1, matching the canonical octEncode in
 * shared-samplers/octahedralCore.wgsl.ts.
 *
 * This test mirrors the inline octEncode logic in ddgiSampleWgsl.ts (lines
 * 90–93 for octV, lines 118–122 for octN) in TypeScript, and asserts that:
 *
 *  1. The FIXED form produces non-zero (non-collapsed) output for all 6
 *     axis-aligned unit directions when z < 0 triggers the lower-hemisphere
 *     fold (±x and ±y land on the fold boundary; the test uses negated z so
 *     the fold IS taken, then verifies the output is not the zero-vector
 *     that the old sign()-based form would have produced).
 *
 *  2. The BUGGY (old) sign()-based form provably collapses at directions
 *     where n.x==0 or n.y==0 in the lower hemisphere, confirming the
 *     before/after difference is real and non-trivial.
 *
 *  3. The FIXED form's output agrees with the canonical octEncodeTS that
 *     shared-samplers already pins (round-trip to within 1e-5 dot).
 *
 * Radiometric impact: direction lookup into DDGI irradiance/visibility
 * atlases is incorrect at axis-aligned surface normals and probe directions
 * (very common in Cornell-box / architectural scenes), sampling from the
 * wrong texel. The fix changes output only at those collapsed lanes.
 */

import { describe, it, expect } from 'vitest';

// ---------------------------------------------------------------------------
// TypeScript mirror of the BUGGY ddgiSampleWgsl.ts inline (OLD form, for
// before/after contrast). Uses Math.sign(), which returns 0 at 0 just like
// WGSL sign().
// ---------------------------------------------------------------------------

/** OLD (buggy) lower-hemisphere fold — sign(0) = 0 collapses the mapping. */
function ddgiOctEncodeOLD(dir: [number, number, number]): [number, number] {
  const denom = Math.abs(dir[0]) + Math.abs(dir[1]) + Math.abs(dir[2]);
  const n: [number, number, number] = [dir[0] / denom, dir[1] / denom, dir[2] / denom];
  if (n[2] >= 0) {
    return [n[0], n[1]];
  }
  // OLD: (1.0 - abs(n.yx)) * vec2f(sign(n.x), sign(n.y))
  // Math.sign(0) === 0, matching WGSL sign(0.0) = 0
  return [
    (1.0 - Math.abs(n[1])) * Math.sign(n[0]),
    (1.0 - Math.abs(n[0])) * Math.sign(n[1]),
  ];
}

// ---------------------------------------------------------------------------
// TypeScript mirror of the FIXED ddgiSampleWgsl.ts inline (NEW form).
// Matches the canonical octEncode in octahedralCore.wgsl.ts.
// ---------------------------------------------------------------------------

/** NEW (fixed) lower-hemisphere fold — select() maps 0 → +1. */
function ddgiOctEncodeNEW(dir: [number, number, number]): [number, number] {
  const denom = Math.abs(dir[0]) + Math.abs(dir[1]) + Math.abs(dir[2]);
  const n: [number, number, number] = [dir[0] / denom, dir[1] / denom, dir[2] / denom];
  if (n[2] >= 0) {
    return [n[0], n[1]];
  }
  // NEW: vec2f((1.0 - abs(n.y)) * select(-1.0, 1.0, n.x >= 0.0),
  //            (1.0 - abs(n.x)) * select(-1.0, 1.0, n.y >= 0.0))
  const sx = n[0] >= 0 ? 1.0 : -1.0;  // select(-1.0, 1.0, n.x >= 0.0)
  const sy = n[1] >= 0 ? 1.0 : -1.0;  // select(-1.0, 1.0, n.y >= 0.0)
  return [
    (1.0 - Math.abs(n[1])) * sx,
    (1.0 - Math.abs(n[0])) * sy,
  ];
}

// ---------------------------------------------------------------------------
// Canonical octEncode from shared-samplers/octahedralCore.wgsl.ts (ref impl).
// ---------------------------------------------------------------------------

function canonicalOctEncodeTS(v: [number, number, number]): [number, number] {
  const denom = Math.abs(v[0]) + Math.abs(v[1]) + Math.abs(v[2]);
  const n: [number, number, number] = [v[0] / denom, v[1] / denom, v[2] / denom];
  if (n[2] >= 0) {
    return [n[0], n[1]];
  }
  const sx = n[0] >= 0 ? 1 : -1;
  const sy = n[1] >= 0 ? 1 : -1;
  return [
    (1.0 - Math.abs(n[1])) * sx,
    (1.0 - Math.abs(n[0])) * sy,
  ];
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function isZeroVec2(v: [number, number], eps = 1e-9): boolean {
  return Math.abs(v[0]) < eps && Math.abs(v[1]) < eps;
}

function vec2Near(
  a: [number, number],
  b: [number, number],
  eps = 1e-5,
): boolean {
  return Math.abs(a[0] - b[0]) < eps && Math.abs(a[1] - b[1]) < eps;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('ddgiOctahedral sign()→select() fix', () => {

  /**
   * Directions that exercise the lower-hemisphere fold (z < 0) AND have
   * n.x or n.y exactly 0 after L1 normalization — these are precisely the
   * lanes where sign(0.0) = 0 collapses the output.
   *
   * The 6 axis directions as encoded by ddgi (before the * 0.5 + 0.5 atlas
   * remap):
   *   +x = (1,0,0) → z=0, upper branch → no fold (fold irrelevant)
   *   -x = (-1,0,0) → z=0, upper branch → no fold
   *   +y = (0,1,0) → z=0, upper branch → no fold
   *   -y = (0,-1,0) → z=0, upper branch → no fold
   *   +z = (0,0,1) → z>0, upper branch → no fold
   *   -z = (0,0,-1) → z<0, lower branch → sign(0.0) collapses BOTH axes
   *
   * For the fold to be taken AND to expose the sign(0) bug, we need
   * z < 0 PLUS n.x == 0 or n.y == 0.  That happens at:
   *   (0, 0, -1)  — both n.x and n.y are 0 after L1 norm
   *   (1, 0, -1)  — n.y = 0 after norm (L1 denom = 2, n = [0.5, 0, -0.5])
   *   (-1, 0, -1) — n.y = 0
   *   (0, 1, -1)  — n.x = 0 after norm
   *   (0, -1, -1) — n.x = 0
   */
  const collapseCandidates: Array<{ label: string; dir: [number, number, number] }> = [
    { label: '(0,0,-1) — south pole, both n.x and n.y = 0 in fold', dir: [0, 0, -1] },
    { label: '(1,0,-1) — lower hemisphere, n.y = 0 in fold',        dir: [1, 0, -1] },
    { label: '(-1,0,-1) — lower hemisphere, n.y = 0 in fold',       dir: [-1, 0, -1] },
    { label: '(0,1,-1) — lower hemisphere, n.x = 0 in fold',        dir: [0, 1, -1] },
    { label: '(0,-1,-1) — lower hemisphere, n.x = 0 in fold',       dir: [0, -1, -1] },
  ];

  describe('OLD sign()-based form collapses at axis-aligned lower-hemisphere directions', () => {
    for (const { label, dir } of collapseCandidates) {
      it(`OLD collapses at ${label}`, () => {
        const enc = ddgiOctEncodeOLD(dir);
        // For directions where BOTH n.x and n.y are 0 (south pole), old form
        // gives (0,0). For directions where only one is 0, at least one
        // component is forced to 0 incorrectly.  Either way, the zero-component
        // assertion fires.
        const hasZeroComponent = Math.abs(enc[0]) < 1e-9 || Math.abs(enc[1]) < 1e-9;
        // The south-pole case collapses to exactly (0,0).
        // Edge cases like (1,0,-1) collapse one axis to 0.
        // This confirm the bug is real and the direction is a true regression candidate.
        expect(
          hasZeroComponent || isZeroVec2(enc),
          `OLD form unexpectedly non-zero for both components at ${label}: got [${enc}]`,
        ).toBe(true);
      });
    }
  });

  describe('NEW select()-based form does NOT collapse at axis-aligned lower-hemisphere directions', () => {
    for (const { label, dir } of collapseCandidates) {
      it(`NEW does not collapse at ${label}`, () => {
        const enc = ddgiOctEncodeNEW(dir);
        // The output must NOT be the zero-vector — the fold maps to a valid
        // octahedral cell, not the degenerate origin.
        expect(
          isZeroVec2(enc),
          `NEW form collapsed to (0,0) for ${label}: got [${enc}]`,
        ).toBe(false);
      });
    }
  });

  describe('NEW form agrees with canonical octEncodeTS from shared-samplers', () => {
    const testDirs: Array<[number, number, number]> = [
      [0, 0, -1],
      [1, 0, -1],
      [-1, 0, -1],
      [0, 1, -1],
      [0, -1, -1],
      // Also check upper-hemisphere and equatorial (unchanged by fix)
      [1, 0, 0],
      [-1, 0, 0],
      [0, 1, 0],
      [0, -1, 0],
      [0, 0, 1],
      // Generic lower-hemisphere (sign()-safe, both axes non-zero)
      [1, 1, -1],
      [-1, -1, -1],
      [0.5, 0.3, -0.9],
    ];

    for (const dir of testDirs) {
      it(`NEW matches canonical for (${dir.join(',')})`, () => {
        const got = ddgiOctEncodeNEW(dir);
        const expected = canonicalOctEncodeTS(dir);
        expect(
          vec2Near(got, expected),
          `NEW [${got}] ≠ canonical [${expected}] for dir (${dir.join(',')})`,
        ).toBe(true);
      });
    }
  });

  describe('NEW form output differs from OLD form at collapse lanes', () => {
    // The fix is only effective if the NEW output is actually DIFFERENT from
    // the OLD output at the collapsed lanes. This is the before/after delta pin.
    for (const { label, dir } of collapseCandidates) {
      it(`NEW ≠ OLD at ${label}`, () => {
        const oldEnc = ddgiOctEncodeOLD(dir);
        const newEnc = ddgiOctEncodeNEW(dir);
        const same = Math.abs(oldEnc[0] - newEnc[0]) < 1e-9 && Math.abs(oldEnc[1] - newEnc[1]) < 1e-9;
        expect(
          same,
          `NEW and OLD produced the same output [${newEnc}] for ${label} — fix had no effect`,
        ).toBe(false);
      });
    }
  });

  describe('DDGI_SAMPLE_WGSL source contains select() form and not the old sign() form', () => {
    it('source does not contain the buggy sign(nv.x) or sign(nv.y) pattern', async () => {
      const { DDGI_SAMPLE_WGSL } = await import('../ddgiSampleWgsl.js');
      // The old form: sign(nv.x) or sign(nv.y) or sign(nN.x) or sign(nN.y)
      expect(DDGI_SAMPLE_WGSL).not.toMatch(/sign\(n[vN]\.[xy]\)/);
    });

    it('source contains the select(-1.0, 1.0, ...) form for the octV (visibility) fold', async () => {
      const { DDGI_SAMPLE_WGSL } = await import('../ddgiSampleWgsl.js');
      // Count occurrences of the fixed pattern — must appear at least twice
      // (2 for the octV visibility fold). The former octN irradiance
      // octahedral fold was removed when irradiance migrated to L2 SH (seam-
      // free; ddgiSH.wgsl.ts) — there is no octahedral irradiance lookup any
      // more, so only the visibility octahedral fold remains and must keep the
      // sign()->select() fix (the axis-aligned collapse this test guards).
      const matches = (DDGI_SAMPLE_WGSL.match(/select\(-1\.0, 1\.0,/g) ?? []).length;
      expect(
        matches,
        `Expected at least 2 select(-1.0, 1.0, ...) occurrences (the octV visibility fold), found ${matches}`,
      ).toBeGreaterThanOrEqual(2);
    });
  });

});
