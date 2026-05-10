/**
 * mixturePdf.test.ts — Unit tests for MIS heuristics and mixture PDF combiner.
 *
 * Standard MIS weight identities verified:
 *  - balanceHeuristic: monotonically increasing in pdf1 (for fixed pdf2 > 0)
 *  - balanceHeuristic + complementary weight sum to 1
 *  - powerHeuristic with β=1 matches balanceHeuristic
 *  - powerHeuristic with β=2 (default) outweighs balance for dominant strategy
 *  - mixturePdf: linear combination of strategy PDFs
 *  - mixturePdf: degenerate single-strategy case returns its own PDF
 *  - edge cases: both PDFs zero → returns 0.5 for heuristics
 */

import { describe, it, expect } from 'vitest';
import { balanceHeuristic, powerHeuristic, mixturePdf } from '../src/mixturePdf.js';

// ── balanceHeuristic ──────────────────────────────────────────────────────────

describe('balanceHeuristic', () => {
  it('weight + complement = 1 for any positive PDFs', () => {
    const pairs: [number, number][] = [
      [1, 1],
      [3, 7],
      [0.001, 999],
      [100, 0.001],
    ];
    for (const [p1, p2] of pairs) {
      const w1 = balanceHeuristic(p1, p2);
      const w2 = balanceHeuristic(p2, p1);
      expect(w1 + w2).toBeCloseTo(1.0);
    }
  });

  it('is monotonically increasing in pdf1 for fixed pdf2', () => {
    const pdf2 = 2.0;
    let prev = balanceHeuristic(0, pdf2);
    for (const pdf1 of [0.5, 1.0, 2.0, 5.0, 10.0]) {
      const w = balanceHeuristic(pdf1, pdf2);
      expect(w).toBeGreaterThanOrEqual(prev);
      prev = w;
    }
  });

  it('returns 0.5 when both PDFs are 0', () => {
    expect(balanceHeuristic(0, 0)).toBeCloseTo(0.5);
  });

  it('returns ~1 when pdf1 >> pdf2', () => {
    expect(balanceHeuristic(1e6, 1)).toBeCloseTo(1.0, 4);
  });

  it('returns ~0 when pdf1 << pdf2', () => {
    expect(balanceHeuristic(1, 1e6)).toBeCloseTo(0.0, 4);
  });
});

// ── powerHeuristic ────────────────────────────────────────────────────────────

describe('powerHeuristic', () => {
  it('with β=1 matches balanceHeuristic', () => {
    const cases: [number, number][] = [[2, 3], [5, 1], [0.5, 0.5], [100, 1]];
    for (const [p1, p2] of cases) {
      expect(powerHeuristic(p1, p2, 1)).toBeCloseTo(balanceHeuristic(p1, p2));
    }
  });

  it('with β=2 (default) amplifies dominant strategy more than balance', () => {
    const pdf1 = 5;
    const pdf2 = 1;
    const balance = balanceHeuristic(pdf1, pdf2);
    const power2 = powerHeuristic(pdf1, pdf2); // default β=2
    // Power heuristic with β=2 should give more weight to the dominant strategy
    expect(power2).toBeGreaterThan(balance);
  });

  it('returns 0.5 when both PDFs are 0', () => {
    expect(powerHeuristic(0, 0)).toBeCloseTo(0.5);
    expect(powerHeuristic(0, 0, 2)).toBeCloseTo(0.5);
  });

  it('weight + complement = 1 for β=2', () => {
    const [p1, p2] = [3.0, 4.0];
    const w1 = powerHeuristic(p1, p2, 2);
    const w2 = powerHeuristic(p2, p1, 2);
    expect(w1 + w2).toBeCloseTo(1.0);
  });

  it('is monotonically increasing in pdf1 for fixed pdf2 and β=2', () => {
    const pdf2 = 2.0;
    let prev = powerHeuristic(0, pdf2);
    for (const pdf1 of [0.5, 1.0, 2.0, 5.0, 10.0]) {
      const w = powerHeuristic(pdf1, pdf2);
      expect(w).toBeGreaterThanOrEqual(prev);
      prev = w;
    }
  });
});

// ── mixturePdf ────────────────────────────────────────────────────────────────

describe('mixturePdf', () => {
  it('single strategy: returns probability × pdf', () => {
    expect(mixturePdf([1.0], [3.5])).toBeCloseTo(3.5);
    expect(mixturePdf([0.5], [2.0])).toBeCloseTo(1.0);
  });

  it('two equal strategies: returns average of PDFs when probabilities are 0.5/0.5', () => {
    const result = mixturePdf([0.5, 0.5], [4.0, 2.0]);
    expect(result).toBeCloseTo(3.0); // 0.5*4 + 0.5*2
  });

  it('three strategies: linear combination', () => {
    // BSDF (40%), env (30%), light (30%)
    const probs = [0.4, 0.3, 0.3];
    const pdfs = [2.0, 1.0, 5.0];
    // 0.4*2 + 0.3*1 + 0.3*5 = 0.8 + 0.3 + 1.5 = 2.6
    expect(mixturePdf(probs, pdfs)).toBeCloseTo(2.6);
  });

  it('zero-probability strategy contributes nothing', () => {
    const result = mixturePdf([1.0, 0.0], [3.0, 999.0]);
    expect(result).toBeCloseTo(3.0);
  });

  it('throws on empty arrays', () => {
    expect(() => mixturePdf([], [])).toThrow();
  });

  it('throws on length mismatch', () => {
    expect(() => mixturePdf([0.5], [1.0, 2.0])).toThrow();
  });
});
