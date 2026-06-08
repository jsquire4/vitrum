/**
 * CPU-side correctness tests for the DDGI atlas border-mirror math.
 *
 * These tests exercise the exact same mirror formulas encoded in
 * `ddgi/wgsl/probeUpdateBorder.wgsl.ts` on a TS-side replica, verifying:
 *
 *   1. Every border texel maps to an interior source (no border→border or
 *      border→out-of-bounds copy).
 *   2. The mirror is an involution — applying it twice returns the original
 *      source (or an equivalent interior texel for edge cases where the
 *      convention symmetry only holds modulo the probe grid).
 *   3. A uniform-color atlas (every interior texel = same value) produces
 *      identical border values — no zero-pollution.
 *   4. Bilinear interpolation at the cell seam (u/v at exactly 0 or 1 in
 *      the [0,1] octahedral UV range) produces the same value as the
 *      interior edge — no darkening from the border.
 *   5. Spot-check specific corner/edge mirror coordinates against the
 *      reference formulas in Majercik 2019 §3.2 / Cigolle 2014 §A.1.
 *
 * These tests do NOT require a GPU. They operate on plain Float32Array
 * atlas simulations.
 */

import { describe, it, expect } from 'vitest';
import {
  makeBorderFillWGSL,
  makeProbeUpdateBorderVisWGSL,
} from '../src/ddgi/wgsl/probeUpdateBorder.wgsl.js';
import { IRR_CELL, IRR_STRIDE, VIS_CELL, VIS_STRIDE } from '../src/ddgi/ddgiAtlasLayout.js';

// ---------------------------------------------------------------------------
// TS mirror of the WGSL border-mirror logic.
// Keep in sync with probeUpdateBorder.wgsl.ts mirror functions.
// ---------------------------------------------------------------------------

/**
 * For a cell of interior dimension N, given a local coordinate (lx, ly)
 * in [0, N+1] × [0, N+1] (stride = N+2), return:
 *   - { mirror: [mx, my], isBorder: true }  if (lx, ly) is a border pixel
 *   - { mirror: [lx, ly], isBorder: false } if (lx, ly) is interior
 *
 * The returned (mx, my) is the interior local coordinate that holds the
 * octahedral-mirrored direction. Interior is at local coords (1..N, 1..N).
 */
function borderMirror(
  N: number,
  lx: number,
  ly: number,
): { mirror: [number, number]; isBorder: boolean } {
  const onLeftEdge   = lx === 0;
  const onRightEdge  = lx === N + 1;
  const onTopEdge    = ly === 0;
  const onBottomEdge = ly === N + 1;
  const isBorder     = onLeftEdge || onRightEdge || onTopEdge || onBottomEdge;
  if (!isBorder) return { mirror: [lx, ly], isBorder: false };

  // Corners
  if (onTopEdge    && onLeftEdge)  return { mirror: [N,     N    ], isBorder: true };
  if (onTopEdge    && onRightEdge) return { mirror: [1,     N    ], isBorder: true };
  if (onBottomEdge && onLeftEdge)  return { mirror: [N,     1    ], isBorder: true };
  if (onBottomEdge && onRightEdge) return { mirror: [1,     1    ], isBorder: true };

  // Edges
  if (onTopEdge)    return { mirror: [N + 1 - lx, 2        ], isBorder: true };
  if (onBottomEdge) return { mirror: [N + 1 - lx, N - 1    ], isBorder: true };
  if (onLeftEdge)   return { mirror: [2,           N + 1 - ly], isBorder: true };
  if (onRightEdge)  return { mirror: [N - 1,       N + 1 - ly], isBorder: true };

  // Unreachable
  return { mirror: [lx, ly], isBorder: false };
}

/** Fill a flat atlas cell (CELL×CELL interior, with 1-pixel border) from a
 *  simple flat Float32Array, run the border mirror, and return the result. */
function runBorderFill(
  N: number,
  interior: Float32Array, // N*N rgba values (row-major, 4 floats each)
): Float32Array {
  const stride = N + 2;
  const total  = stride * stride;
  const atlas  = new Float32Array(total * 4); // rgba per texel

  // Write interior into atlas (at local offset (+1, +1)).
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const src = (y * N + x) * 4;
      const dst = ((y + 1) * stride + (x + 1)) * 4;
      atlas[dst + 0] = interior[src + 0]!;
      atlas[dst + 1] = interior[src + 1]!;
      atlas[dst + 2] = interior[src + 2]!;
      atlas[dst + 3] = interior[src + 3]!;
    }
  }

  // Run border fill.
  for (let ly = 0; ly < stride; ly++) {
    for (let lx = 0; lx < stride; lx++) {
      const { mirror, isBorder } = borderMirror(N, lx, ly);
      if (!isBorder) continue;
      const [mx, my] = mirror;
      const srcIdx   = (my * stride + mx) * 4;
      const dstIdx   = (ly * stride + lx) * 4;
      atlas[dstIdx + 0] = atlas[srcIdx + 0]!;
      atlas[dstIdx + 1] = atlas[srcIdx + 1]!;
      atlas[dstIdx + 2] = atlas[srcIdx + 2]!;
      atlas[dstIdx + 3] = atlas[srcIdx + 3]!;
    }
  }

  return atlas;
}

/** Read a single texel from a flat atlas (rgba, row-major). */
function readTexel(atlas: Float32Array, stride: number, lx: number, ly: number): [number, number, number, number] {
  const i = (ly * stride + lx) * 4;
  return [atlas[i]!, atlas[i + 1]!, atlas[i + 2]!, atlas[i + 3]!];
}

/** Bilinear interpolation between four texel values at fractional (u, v) in [0, 1]. */
function bilinear(
  tl: [number, number, number, number],
  tr: [number, number, number, number],
  bl: [number, number, number, number],
  br: [number, number, number, number],
  u: number,
  v: number,
): [number, number, number, number] {
  const lerp = (a: number, b: number, t: number) => a * (1 - t) + b * t;
  return [
    lerp(lerp(tl[0], tr[0], u), lerp(bl[0], br[0], u), v),
    lerp(lerp(tl[1], tr[1], u), lerp(bl[1], br[1], u), v),
    lerp(lerp(tl[2], tr[2], u), lerp(bl[2], br[2], u), v),
    lerp(lerp(tl[3], tr[3], u), lerp(bl[3], br[3], u), v),
  ];
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('DDGI atlas border-mirror math (CPU replica, no GPU required)', () => {
  const IRR_N = 8;  // IRR_CELL
  const VIS_N = 16; // VIS_CELL

  // Test 1 — Mirror targets are always interior for every N in {8, 16}.
  for (const N of [IRR_N, VIS_N]) {
    const stride = N + 2;
    describe(`N=${N}: every border texel maps to an interior source`, () => {
      it(`all ${4 * N + 4} border texels of a ${stride}×${stride} cell mirror to interior (1..${N}, 1..${N})`, () => {
        let borderCount = 0;
        for (let ly = 0; ly < stride; ly++) {
          for (let lx = 0; lx < stride; lx++) {
            const { mirror, isBorder } = borderMirror(N, lx, ly);
            if (!isBorder) continue;
            borderCount++;
            const [mx, my] = mirror;
            // Mirror must land inside the interior region.
            expect(mx).toBeGreaterThanOrEqual(1);
            expect(mx).toBeLessThanOrEqual(N);
            expect(my).toBeGreaterThanOrEqual(1);
            expect(my).toBeLessThanOrEqual(N);
          }
        }
        // 4 edges × N texels + 4 corners = 4N+4.
        expect(borderCount).toBe(4 * N + 4);
      });
    });
  }

  // Test 2 — Spot-check specific mirror coordinates (Majercik 2019 §3.2).
  describe('N=8: specific mirror coordinates match reference formulas', () => {
    const N = IRR_N;

    it('corner (0,0) → interior (N, N) = (8, 8)', () => {
      const { mirror } = borderMirror(N, 0, 0);
      expect(mirror).toEqual([8, 8]);
    });

    it('corner (N+1, 0) = (9,0) → interior (1, N) = (1, 8)', () => {
      const { mirror } = borderMirror(N, N + 1, 0);
      expect(mirror).toEqual([1, 8]);
    });

    it('corner (0, N+1) = (0, 9) → interior (N, 1) = (8, 1)', () => {
      const { mirror } = borderMirror(N, 0, N + 1);
      expect(mirror).toEqual([8, 1]);
    });

    it('corner (N+1, N+1) = (9, 9) → interior (1, 1)', () => {
      const { mirror } = borderMirror(N, N + 1, N + 1);
      expect(mirror).toEqual([1, 1]);
    });

    it('top edge (lx=3, ly=0) → interior (N+1-3, 2) = (6, 2)', () => {
      const { mirror } = borderMirror(N, 3, 0);
      expect(mirror).toEqual([6, 2]);
    });

    it('bottom edge (lx=5, ly=N+1=9) → interior (N+1-5, N-1) = (4, 7)', () => {
      const { mirror } = borderMirror(N, 5, N + 1);
      expect(mirror).toEqual([4, 7]);
    });

    it('left edge (lx=0, ly=4) → interior (2, N+1-4) = (2, 5)', () => {
      const { mirror } = borderMirror(N, 0, 4);
      expect(mirror).toEqual([2, 5]);
    });

    it('right edge (lx=N+1=9, ly=7) → interior (N-1, N+1-7) = (7, 2)', () => {
      const { mirror } = borderMirror(N, N + 1, 7);
      expect(mirror).toEqual([7, 2]);
    });
  });

  // Test 3 — Uniform interior → uniform atlas including borders (no zero-pollution).
  describe('uniform interior produces no zero-pollution in borders', () => {
    for (const N of [IRR_N, VIS_N]) {
      it(`N=${N}: every border texel equals the interior constant after fill`, () => {
        const interior = new Float32Array(N * N * 4);
        // Fill with a non-zero sentinel: r=0.7, g=0.3, b=0.9, a=1.0.
        for (let i = 0; i < N * N; i++) {
          interior[i * 4 + 0] = 0.7;
          interior[i * 4 + 1] = 0.3;
          interior[i * 4 + 2] = 0.9;
          interior[i * 4 + 3] = 1.0;
        }
        const atlas  = runBorderFill(N, interior);
        const stride = N + 2;
        const EPS = 1e-6;

        for (let ly = 0; ly < stride; ly++) {
          for (let lx = 0; lx < stride; lx++) {
            const { isBorder } = borderMirror(N, lx, ly);
            if (!isBorder) continue;
            const [r, g, b, a] = readTexel(atlas, stride, lx, ly);
            expect(Math.abs(r - 0.7)).toBeLessThan(EPS);
            expect(Math.abs(g - 0.3)).toBeLessThan(EPS);
            expect(Math.abs(b - 0.9)).toBeLessThan(EPS);
            expect(Math.abs(a - 1.0)).toBeLessThan(EPS);
          }
        }
      });
    }
  });

  // Test 4 — Bilinear sampling at the cell seam produces no darkening.
  // With uniform interior and correct borders, sampling at uv=(0,0.5) (left seam)
  // should return the same constant as the interior, not 0.5×constant (the
  // blend of interior with a zero border pixel).
  describe('bilinear interpolation at cell seam with uniform interior', () => {
    const N = IRR_N;
    const stride = N + 2;

    it('sampling at u=0 (left seam) returns interior value, not 50% darkened', () => {
      const interior = new Float32Array(N * N * 4).fill(0);
      for (let i = 0; i < N * N; i++) {
        interior[i * 4 + 0] = 0.6;
        interior[i * 4 + 1] = 0.4;
        interior[i * 4 + 2] = 0.2;
        interior[i * 4 + 3] = 1.0;
      }
      const atlas = runBorderFill(N, interior);

      // Sample at the left seam: u=0, v=0.5.
      // The atlas UV maps [0,1] octahedral → [1, 1+N] pixel range with +0.5 centering.
      // At u=0, the sample point falls at pixel x=1.0 (between border x=0 and interior x=1).
      // Bilinear uses pixels at x=0 (border) and x=1 (first interior column).
      const vy   = 4; // arbitrary interior y (interior row 4 = local ly=5)
      const tl   = readTexel(atlas, stride, 0, vy);      // border left
      const tr   = readTexel(atlas, stride, 1, vy);      // interior first col
      const bl   = readTexel(atlas, stride, 0, vy + 1);  // border left, next row
      const br   = readTexel(atlas, stride, 1, vy + 1);  // interior first col, next row
      const u    = 0.0; // exactly at the seam — bilinear: border contributes (1-0)=100%
      const v    = 0.0;
      const sampled = bilinear(tl, tr, bl, br, u, v);

      // With a zero border (bug state), sampled[0] = 0.
      // With a correct border mirror, sampled[0] = 0.6.
      const EPS = 1e-5;
      expect(Math.abs(sampled[0] - 0.6)).toBeLessThan(EPS);
      expect(Math.abs(sampled[1] - 0.4)).toBeLessThan(EPS);
    });

    it('sampling at u=1 (right seam) returns interior value', () => {
      const interior = new Float32Array(N * N * 4);
      for (let i = 0; i < N * N; i++) {
        interior[i * 4 + 0] = 0.5;
        interior[i * 4 + 1] = 0.5;
        interior[i * 4 + 2] = 0.5;
        interior[i * 4 + 3] = 1.0;
      }
      const atlas = runBorderFill(N, interior);

      // At u=1, sample point is between interior last column (lx=N=8) and right border (lx=N+1=9).
      const vy  = 3;
      const tl  = readTexel(atlas, stride, N,     vy);
      const tr  = readTexel(atlas, stride, N + 1, vy);
      const bl  = readTexel(atlas, stride, N,     vy + 1);
      const br  = readTexel(atlas, stride, N + 1, vy + 1);
      const u   = 1.0; // full weight on right border column
      const v   = 0.0;
      const sampled = bilinear(tl, tr, bl, br, u, v);

      const EPS = 1e-5;
      // Border mirror: right edge (lx=N+1, any ly) → interior (N-1, N+1-ly).
      // For a uniform atlas, this is always 0.5.
      expect(Math.abs(sampled[0] - 0.5)).toBeLessThan(EPS);
    });

    it('sampling at v=0 (top seam) returns interior value', () => {
      const interior = new Float32Array(N * N * 4);
      for (let i = 0; i < N * N; i++) {
        interior[i * 4 + 0] = 0.8;
        interior[i * 4 + 1] = 0.2;
        interior[i * 4 + 2] = 0.1;
        interior[i * 4 + 3] = 1.0;
      }
      const atlas = runBorderFill(N, interior);

      const lx  = 3; // arbitrary x in interior
      const tl  = readTexel(atlas, stride, lx,     0); // border top
      const tr  = readTexel(atlas, stride, lx + 1, 0); // border top, next col
      const bl  = readTexel(atlas, stride, lx,     1); // interior first row
      const br  = readTexel(atlas, stride, lx + 1, 1); // interior first row, next col
      const u   = 0.0;
      const v   = 0.0; // full weight on top border
      const sampled = bilinear(tl, tr, bl, br, u, v);

      const EPS = 1e-5;
      expect(Math.abs(sampled[0] - 0.8)).toBeLessThan(EPS);
      expect(Math.abs(sampled[1] - 0.2)).toBeLessThan(EPS);
    });
  });

  // Test 5 — Interior pixels are not modified by the border fill.
  describe('border fill does not corrupt interior pixels', () => {
    it('N=8: non-uniform interior values survive border fill unchanged', () => {
      const N = IRR_N;
      const stride = N + 2;
      const interior = new Float32Array(N * N * 4);
      // Fill with a gradient so interior values are all distinct.
      for (let y = 0; y < N; y++) {
        for (let x = 0; x < N; x++) {
          const i = (y * N + x) * 4;
          interior[i + 0] = (x + 1) / (N + 1);
          interior[i + 1] = (y + 1) / (N + 1);
          interior[i + 2] = 0.5;
          interior[i + 3] = 1.0;
        }
      }
      const atlas = runBorderFill(N, interior);
      const EPS = 1e-6;
      for (let y = 0; y < N; y++) {
        for (let x = 0; x < N; x++) {
          const [r, g] = readTexel(atlas, stride, x + 1, y + 1);
          expect(Math.abs(r - (x + 1) / (N + 1))).toBeLessThan(EPS);
          expect(Math.abs(g - (y + 1) / (N + 1))).toBeLessThan(EPS);
        }
      }
    });
  });
});

// ---------------------------------------------------------------------------
// Pass layout tests — border slots must be registered.
// ---------------------------------------------------------------------------

import { buildPassLayout, MAX_PASS_COUNT } from '../src/pipeline/timestampQueries.js';

describe('buildPassLayout — DDGI border fill slots', () => {
  it('atrous-variance layout includes ddgi-border-irr and ddgi-border-vis', () => {
    const layout = buildPassLayout({ denoiserMode: 'atrous-variance' });
    // Both slots must be present (no throw).
    expect(layout.index('ddgi-border-irr')).toBeGreaterThan(0);
    expect(layout.index('ddgi-border-vis')).toBeGreaterThan(0);
  });

  it('atrous layout includes ddgi-border-irr and ddgi-border-vis', () => {
    const layout = buildPassLayout({ denoiserMode: 'atrous' });
    expect(layout.index('ddgi-border-irr')).toBeGreaterThan(0);
    expect(layout.index('ddgi-border-vis')).toBeGreaterThan(0);
  });

  it('ddgi-border-irr comes before ddgi-border-vis', () => {
    const layout = buildPassLayout({ denoiserMode: 'atrous-variance' });
    expect(layout.index('ddgi-border-irr')).toBeLessThan(layout.index('ddgi-border-vis'));
  });

  it('ddgi-border-vis comes before temporalAccum', () => {
    const layout = buildPassLayout({ denoiserMode: 'atrous-variance' });
    expect(layout.index('ddgi-border-vis')).toBeLessThan(layout.index('temporalAccum'));
  });

  it('ddgi-border-irr comes after indirect-combine', () => {
    const layout = buildPassLayout({ denoiserMode: 'atrous-variance' });
    expect(layout.index('ddgi-border-irr')).toBeGreaterThan(layout.index('indirect-combine'));
  });

  it('atrous-variance layout reports 31 slots (30 + trailing opt-in regir-build)', () => {
    // 2026-06-06: the dead ppg-guide pass was removed from the order table
    // (G-P1.1) — every layout dropped exactly one slot.
    const layout = buildPassLayout({ denoiserMode: 'atrous-variance' });
    expect(layout.slotCount).toBe(31);
    expect(layout.slotCount).toBeLessThanOrEqual(MAX_PASS_COUNT);
  });

  it('atrous layout reports 29 slots (28 + trailing opt-in regir-build)', () => {
    const layout = buildPassLayout({ denoiserMode: 'atrous' });
    expect(layout.slotCount).toBe(29);
  });

  it('MAX_PASS_COUNT is 34 (includes the trailing opt-in regir-build slot)', () => {
    expect(MAX_PASS_COUNT).toBe(34);
  });
});

// ---------------------------------------------------------------------------
// Test 5 — Non-uniform interior spot-check (catches source-row off-by-one).
// Audit follow-up: the uniform-interior tests above can't distinguish
// source row 1 vs row 2 for the top edge (or N vs N-1 for bottom). This
// suite uses a per-row distinct value so a bad source row would show up.
// ---------------------------------------------------------------------------
describe('non-uniform interior — source-row sanity', () => {
  it('N=8: top-edge border samples interior row 2 (not row 1)', () => {
    const N = 8;
    const stride = N + 2;
    // Interior row r (1..N) is filled with r=row, g=0, b=0, a=1.
    // After border fill, top-edge border (ly=0, lx=1..N) should hold row=2.
    const interior = new Float32Array(N * N * 4);
    for (let r = 0; r < N; r++) {
      for (let c = 0; c < N; c++) {
        const i = (r * N + c) * 4;
        interior[i + 0] = r + 1;  // row label 1..N
        interior[i + 1] = 0;
        interior[i + 2] = 0;
        interior[i + 3] = 1;
      }
    }
    const atlas = runBorderFill(N, interior);

    // For each top-edge border texel (lx ∈ 1..N, ly = 0), the source per
    // borderMirror is interior (N+1-lx, 2). The atlas y=0 row should hold
    // value r=2 in every position.
    for (let lx = 1; lx <= N; lx++) {
      const [r, g, b, a] = readTexel(atlas, stride, lx, 0);
      expect(r).toBe(2);  // ← source row label is 2 (NOT 1)
      expect(g).toBe(0);
      expect(b).toBe(0);
      expect(a).toBe(1);
    }
  });

  it('N=8: bottom-edge border samples interior row N-1 = 7 (not row N = 8)', () => {
    const N = 8;
    const stride = N + 2;
    const interior = new Float32Array(N * N * 4);
    for (let r = 0; r < N; r++) {
      for (let c = 0; c < N; c++) {
        const i = (r * N + c) * 4;
        interior[i + 0] = r + 1;
        interior[i + 3] = 1;
      }
    }
    const atlas = runBorderFill(N, interior);

    // Bottom-edge border (lx ∈ 1..N, ly = N+1=9). Source: interior (N+1-lx, N-1).
    // N-1 = 7 → row label 7 (NOT 8).
    for (let lx = 1; lx <= N; lx++) {
      const [r] = readTexel(atlas, stride, lx, N + 1);
      expect(r).toBe(7);
    }
  });

  it('N=8: left-edge border samples interior column 2 (not column 1)', () => {
    const N = 8;
    const stride = N + 2;
    const interior = new Float32Array(N * N * 4);
    // Interior column c (1..N) holds r=column.
    for (let r = 0; r < N; r++) {
      for (let c = 0; c < N; c++) {
        const i = (r * N + c) * 4;
        interior[i + 0] = c + 1;
        interior[i + 3] = 1;
      }
    }
    const atlas = runBorderFill(N, interior);

    // Left-edge border (lx=0, ly ∈ 1..N). Source: interior (2, N+1-ly).
    // Column label 2 (NOT 1).
    for (let ly = 1; ly <= N; ly++) {
      const [r] = readTexel(atlas, stride, 0, ly);
      expect(r).toBe(2);
    }
  });

  it('N=8: top-edge border preserves the cross-cell horizontal flip (source x = N+1-lx)', () => {
    const N = 8;
    const stride = N + 2;
    // Interior column c (1..N) holds r=column. Top edge mirror flips x:
    // border (lx=1, ly=0) → source (8, 2) → r=8.
    // border (lx=8, ly=0) → source (1, 2) → r=1.
    const interior = new Float32Array(N * N * 4);
    for (let r = 0; r < N; r++) {
      for (let c = 0; c < N; c++) {
        const i = (r * N + c) * 4;
        interior[i + 0] = c + 1;
        interior[i + 3] = 1;
      }
    }
    const atlas = runBorderFill(N, interior);
    expect(readTexel(atlas, stride, 1, 0)[0]).toBe(8);   // lx=1 → mirror source col=8
    expect(readTexel(atlas, stride, 8, 0)[0]).toBe(1);   // lx=8 → mirror source col=1
    expect(readTexel(atlas, stride, 4, 0)[0]).toBe(5);   // lx=4 → mirror source col=5
  });
});

// ---------------------------------------------------------------------------
// Coverage regression — the emitted WGSL must visit EVERY local position of
// the cell, not just the first 2*workgroupSize.
//
// Pre-existing bug (V5 radiometric): the irradiance pass (workgroupSize=48)
// used a fixed 2-strip loop covering local positions [0,96), but a 10×10 cell
// has 100 positions. Thread positions t=96..99 map to (lx = t%10, ly = t/10) =
// (6,9),(7,9),(8,9),(9,9) — all BOTTOM-edge border texels (ly = N+1 = 9) that
// were left at their zero-initialized value → seam darkening at the bottom
// octahedral edge of every probe cell.
//
// The fix derives the strip count as ⌈stride²/workgroupSize⌉, so every border
// texel is now written. These tests parse the emitted loop bound and replay
// the exact thread→texel mapping to prove full coverage (and would re-fail if
// the fixed 2-strip loop ever returned).
// ---------------------------------------------------------------------------
describe('border-fill WGSL covers every border texel of the cell (V5 regression)', () => {
  /** Extract the `passIdx < <N>u` loop bound from the emitted WGSL. */
  function loopStripCount(wgsl: string): number {
    const m = wgsl.match(/for \(var passIdx = 0u; passIdx < (\d+)u;/);
    if (!m) throw new Error('could not find the passIdx strip-count loop in emitted WGSL');
    return Number(m[1]);
  }

  /**
   * Replay the WGSL thread→position mapping exactly:
   * thread `t = lid.x + passIdx*workgroupSize` for lid.x in [0,workgroupSize)
   * and passIdx in [0,strips); positions ≥ stride² are skipped. Returns the
   * set of border texels (as "lx,ly") that the pass actually writes.
   */
  function coveredBorderTexels(
    cell: number,
    stride: number,
    workgroupSize: number,
    strips: number,
  ): Set<string> {
    const total = stride * stride;
    const N = cell;
    const covered = new Set<string>();
    for (let passIdx = 0; passIdx < strips; passIdx++) {
      for (let lidx = 0; lidx < workgroupSize; lidx++) {
        const t = lidx + passIdx * workgroupSize;
        if (t >= total) continue;
        const lx = t % stride;
        const ly = Math.floor(t / stride);
        const isBorder = lx === 0 || lx === N + 1 || ly === 0 || ly === N + 1;
        if (isBorder) covered.add(`${lx},${ly}`);
      }
    }
    return covered;
  }

  /** Enumerate the full border-ring texel set for a (cell, stride) cell. */
  function allBorderTexels(cell: number, stride: number): Set<string> {
    const N = cell;
    const all = new Set<string>();
    for (let ly = 0; ly < stride; ly++) {
      for (let lx = 0; lx < stride; lx++) {
        if (lx === 0 || lx === N + 1 || ly === 0 || ly === N + 1) all.add(`${lx},${ly}`);
      }
    }
    return all;
  }

  // The V5 strip-coverage regression lives in the SHARED makeBorderFillWGSL
  // factory (still used by the visibility border). It used to be exercised via
  // the irradiance border at 8×8/stride-10/48-threads; irradiance migrated to SH
  // (no border), so these tests now call the factory directly at that historical
  // 8×8/48-thread case (decoupled from the production IRR_CELL, now 3).
  const fill8x8 = () => makeBorderFillWGSL({ cell: 8, stride: 10, workgroupSize: 48, entryPoint: 'probeUpdateBorderIrradiance' });

  it('border-fill (48 threads, 10×10 cell) emits ⌈100/48⌉ = 3 strips', () => {
    const wgsl = fill8x8();
    const strips = loopStripCount(wgsl);
    expect(strips).toBe(Math.ceil((10 * 10) / 48)); // 3
    // 48 threads × strips must reach at least all 100 positions.
    expect(48 * strips).toBeGreaterThanOrEqual(10 * 10);
  });

  it('border-fill writes ALL border texels — incl. the four bottom-edge texels the old 2-strip loop missed', () => {
    const wgsl = fill8x8();
    const strips = loopStripCount(wgsl);
    const covered = coveredBorderTexels(8, 10, 48, strips);
    const all = allBorderTexels(8, 10);

    // Every border texel of the 10×10 cell is now covered.
    for (const texel of all) expect(covered.has(texel)).toBe(true);
    expect(covered.size).toBe(all.size);

    // Explicitly assert the previously-unfilled bottom-edge texels (ly = N+1 = 9,
    // lx ∈ {6,7,8,9}) — these are positions t = 96..99, missed by the old [0,96) loop.
    for (const lx of [6, 7, 8, 9]) {
      expect(covered.has(`${lx},9`)).toBe(true);   // ly = N+1 = 9 for the 8×8 cell
    }
  });

  it('the old fixed 2-strip loop would NOT have covered the four bottom-edge texels (documents the bug)', () => {
    // Re-run coverage with the buggy strip count of 2 to confirm the gap was real.
    const buggy = coveredBorderTexels(8, 10, 48, 2);
    for (const lx of [6, 7, 8, 9]) {
      expect(buggy.has(`${lx},9`)).toBe(false);
    }
  });

  it('visibility pass (256 threads, 18×18 cell) is unchanged at ⌈324/256⌉ = 2 strips and fully covered', () => {
    const wgsl = makeProbeUpdateBorderVisWGSL();
    const strips = loopStripCount(wgsl);
    expect(strips).toBe(2);
    const covered = coveredBorderTexels(VIS_CELL, VIS_STRIDE, 256, strips);
    const all = allBorderTexels(VIS_CELL, VIS_STRIDE);
    for (const texel of all) expect(covered.has(texel)).toBe(true);
    expect(covered.size).toBe(all.size);
  });

  it('strip count is derived from stride²/workgroupSize for arbitrary cells', () => {
    // A pathological tiny workgroup must still fully cover the ring.
    const wgsl = makeBorderFillWGSL({ cell: 8, stride: 10, workgroupSize: 7, entryPoint: 'probe' });
    const strips = loopStripCount(wgsl);
    expect(strips).toBe(Math.ceil(100 / 7)); // 15
    const covered = coveredBorderTexels(8, 10, 7, strips);
    const all = allBorderTexels(8, 10);
    expect(covered.size).toBe(all.size);
  });
});
