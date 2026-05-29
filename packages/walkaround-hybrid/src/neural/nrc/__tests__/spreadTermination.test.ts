// spreadTermination.test.ts — CPU-oracle + WGSL-codegen tests for the NRC
// path-spread cache-termination heuristic (Müller et al. 2021 §5).
//
// The oracle (`../spreadTermination.ts`) is the reference; the WGSL predicate
// (`../wgsl/spreadTermination.wgsl.ts`) is hand-verified line-for-line. The
// predicate is pure arithmetic (sqrt + running sum + comparison) with no kinks,
// so the WGSL ↔ CPU match is exact to f32 epsilon — the codegen-shape pins below
// guard that the WGSL keeps emitting that same arithmetic.

import { describe, it, expect } from 'vitest';
import {
  segmentSpreadTerm, accumulatedSpread, primarySpread, evaluateSpreadTermination,
  type PathSegment,
} from '../spreadTermination.ts';
import { nrcSpreadTerminationWgsl } from '../wgsl/spreadTermination.wgsl.ts';

describe('NRC spread term — per segment', () => {
  it('equals sqrt(d² / (p·|cosθ|))', () => {
    const seg: PathSegment = { dist: 2, pdf: 0.5, cosTheta: 0.8 };
    expect(segmentSpreadTerm(seg)).toBeCloseTo(Math.sqrt((2 * 2) / (0.5 * 0.8)), 9);
  });

  it('clamps a degenerate (near-zero pdf·cos) denominator → huge spread (early termination)', () => {
    const seg: PathSegment = { dist: 1, pdf: 0, cosTheta: 0 };
    // denom clamped to 1e-12 → term = sqrt(1/1e-12) = 1e6.
    expect(segmentSpreadTerm(seg)).toBeCloseTo(1e6, 0);
  });
});

describe('NRC accumulated spread (Müller 2021 §5)', () => {
  it('is the SQUARE of the running sum of per-segment terms', () => {
    const segs: PathSegment[] = [
      { dist: 1, pdf: 0.4, cosTheta: 0.9 },
      { dist: 0.5, pdf: 0.6, cosTheta: 0.7 },
      { dist: 2, pdf: 0.3, cosTheta: 0.5 },
    ];
    const acc = accumulatedSpread(segs);
    let running = 0;
    for (let i = 0; i < segs.length; i++) {
      running += segmentSpreadTerm(segs[i]!);
      // acc is a Float32Array → compare to the JS-double reference at f32
      // precision (~1e-6 relative; the brief's stated tolerance).
      expect(acc[i]).toBeCloseTo(running * running, 4);
    }
  });

  it('is monotonically non-decreasing along the path (spread only grows)', () => {
    const segs: PathSegment[] = [
      { dist: 1, pdf: 0.4, cosTheta: 0.9 },
      { dist: 0.5, pdf: 0.6, cosTheta: 0.7 },
      { dist: 2, pdf: 0.3, cosTheta: 0.5 },
    ];
    const acc = accumulatedSpread(segs);
    for (let i = 1; i < acc.length; i++) expect(acc[i]).toBeGreaterThanOrEqual(acc[i - 1]!);
  });

  it('a0 = (first-segment term)²', () => {
    const segs: PathSegment[] = [{ dist: 1.5, pdf: 0.5, cosTheta: 0.8 }, { dist: 1, pdf: 0.5, cosTheta: 0.5 }];
    const t0 = segmentSpreadTerm(segs[0]!);
    expect(primarySpread(segs)).toBeCloseTo(t0 * t0, 9);
  });
});

describe('NRC cache-termination heuristic a(x) > c·a0', () => {
  it('never terminates at the primary vertex (k=0)', () => {
    // even with a tiny c, the loop starts at k=1: a single-segment path never
    // terminates (the primary hit is never a cache-query target).
    const r = evaluateSpreadTermination([{ dist: 1, pdf: 0.5, cosTheta: 0.8 }], 1e-9);
    expect(r.terminate).toBe(false);
    expect(r.terminateAtSegment).toBe(-1);
  });

  it('fires at the first vertex whose accumulated spread exceeds c·a0', () => {
    const segs: PathSegment[] = [
      { dist: 1, pdf: 0.5, cosTheta: 0.9 },   // a0 footprint
      { dist: 1, pdf: 0.5, cosTheta: 0.9 },   // spread grows
      { dist: 5, pdf: 0.05, cosTheta: 0.3 },  // big jump
    ];
    const acc = accumulatedSpread(segs);
    const a0 = primarySpread(segs);
    // choose c so the threshold lands between segment 1 and 2.
    const c = (acc[1]! / a0 + acc[2]! / a0) / 2;
    const r = evaluateSpreadTermination(segs, c);
    expect(r.terminate).toBe(true);
    expect(r.terminateAtSegment).toBe(2);
    expect(acc[2]!).toBeGreaterThan(c * a0);
    expect(acc[1]!).toBeLessThanOrEqual(c * a0);
  });

  it('larger c defers termination (less bias / longer paths)', () => {
    const segs: PathSegment[] = [
      { dist: 1, pdf: 0.5, cosTheta: 0.9 },
      { dist: 1, pdf: 0.4, cosTheta: 0.8 },
      { dist: 1, pdf: 0.3, cosTheta: 0.6 },
      { dist: 1, pdf: 0.2, cosTheta: 0.4 },
    ];
    const small = evaluateSpreadTermination(segs, 1.2);
    const large = evaluateSpreadTermination(segs, 8.0);
    expect(small.terminate).toBe(true);
    // a larger c either defers to a later segment or stops firing within the path.
    if (large.terminate) {
      expect(large.terminateAtSegment).toBeGreaterThanOrEqual(small.terminateAtSegment);
    } else {
      expect(large.terminateAtSegment).toBe(-1);
    }
  });
});

describe('NRC spread WGSL codegen — shape pins (oracle equivalence)', () => {
  it('emits sqrt(d²/(p·|cosθ|)) with the same 1e-12 denom clamp', () => {
    const wgsl = nrcSpreadTerminationWgsl();
    expect(wgsl).toContain('max(pdf * abs(cosTheta), 1e-12)');
    expect(wgsl).toContain('sqrt((dist * dist) / denom)');
  });

  it('accumulates the running sum and returns its square as a(x)', () => {
    const wgsl = nrcSpreadTerminationWgsl();
    expect(wgsl).toContain('(*runningSum_io) * (*runningSum_io)');
  });

  it('the termination test is a(x) > c·a0', () => {
    const wgsl = nrcSpreadTerminationWgsl();
    expect(wgsl).toContain('return aX > c * a0;');
  });
});
