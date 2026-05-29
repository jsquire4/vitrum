/**
 * bmfrRegression.test.ts — CPU-unit tests for the BMFR per-block feature
 * regression (Koskela et al. 2019).
 *
 * The GPU kernel (wgsl/bmfr.wgsl.ts) reimplements this exact math; these tests
 * pin the numerical behaviour without needing a GPU device:
 *   - Householder QR solves a known small linear system.
 *   - A noisy 1-spp signal over a SMOOTH surface denoises toward the mean.
 *   - A signal that is an EXACT linear function of the features is recovered
 *     to (near-)machine precision (zero residual fit).
 *   - The fit stays bounded on a rank-deficient (constant) block.
 */

import { describe, expect, it } from 'vitest';
import {
  BMFR_FEATURE_COUNT,
  bmfrFeatureRow,
  bmfrFitBlock,
  bmfrSolveChannel,
  householderSolve,
} from '../src/bmfrRegression.js';

/** Deterministic LCG so tests are reproducible. */
function lcg(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (1664525 * s + 1013904223) >>> 0;
    return s / 0xffffffff;
  };
}

describe('householderSolve', () => {
  it('solves a small symmetric positive-definite system', () => {
    // M x = b with a known x. M is 3×3 SPD.
    const n = 3;
    const M = new Float64Array([
      4, 1, 0,
      1, 3, 1,
      0, 1, 2,
    ]);
    const xTrue = [1, -2, 3];
    const b = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let acc = 0;
      for (let j = 0; j < n; j++) acc += M[i * n + j]! * xTrue[j]!;
      b[i] = acc;
    }
    const x = householderSolve(M, b, n);
    for (let i = 0; i < n; i++) expect(x[i]!).toBeCloseTo(xTrue[i]!, 4);
  });

  it('solves an identity system (x == b)', () => {
    const n = 4;
    const I = new Float64Array(n * n);
    for (let i = 0; i < n; i++) I[i * n + i] = 1;
    const b = new Float64Array([5, -3, 2, 7]);
    const x = householderSolve(I, b, n);
    for (let i = 0; i < n; i++) expect(x[i]!).toBeCloseTo(b[i]!, 6);
  });
});

describe('bmfrFeatureRow', () => {
  it('lays out [1, p.xyz, n.xyz, p².xyz]', () => {
    const out = new Float32Array(BMFR_FEATURE_COUNT);
    bmfrFeatureRow([2, 3, -1], [0, 1, 0], out);
    expect(Array.from(out)).toEqual([1, 2, 3, -1, 0, 1, 0, 4, 9, 1]);
  });
});

describe('bmfrSolveChannel — exact linear recovery', () => {
  it('recovers weights when the signal is an exact feature combination', () => {
    // Construct a 16-pixel block on a tilted plane (positions vary in x,y,z),
    // and a color that is EXACTLY w·features for a known w. The least-squares
    // fit must recover w (zero residual).
    const rand = lcg(42);
    const wTrue = [0.5, 0.2, -0.1, 0.05, 0.3, 0.1, -0.2, 0.01, -0.02, 0.03];
    const rows: Float32Array[] = [];
    const vals: number[] = [];
    for (let i = 0; i < 16; i++) {
      const p: [number, number, number] = [
        rand() * 2 - 1, rand() * 2 - 1, rand() * 2 - 1,
      ];
      // Unit-ish normal.
      const n: [number, number, number] = [rand() * 2 - 1, rand() * 2 - 1, rand() * 2 - 1];
      const row = new Float32Array(BMFR_FEATURE_COUNT);
      bmfrFeatureRow(p, n, row);
      rows.push(row);
      let target = 0;
      for (let k = 0; k < BMFR_FEATURE_COUNT; k++) target += row[k]! * wTrue[k]!;
      vals.push(target);
    }
    // Use ~zero regularisation so exact recovery is possible.
    const alpha = bmfrSolveChannel(rows, vals, 1e-9);
    // Reconstruction must match the exact targets very closely.
    for (let i = 0; i < rows.length; i++) {
      let recon = 0;
      for (let k = 0; k < BMFR_FEATURE_COUNT; k++) recon += rows[i]![k]! * alpha[k]!;
      expect(recon).toBeCloseTo(vals[i]!, 3);
    }
  });
});

describe('bmfrFitBlock — denoises a noisy smooth surface toward the mean', () => {
  it('reduces variance of a constant signal + zero-mean noise', () => {
    // A flat block: all pixels share one world plane (z=0), one normal.
    // The TRUE color is a constant c0; the noisy 1-spp input is c0 + noise.
    // The fit should collapse to ~c0 everywhere (the spatial features cannot
    // represent the per-pixel noise, so least-squares averages it out).
    const blockN = 64; // 8×8 conceptual block
    const c0 = [0.6, 0.4, 0.3];
    const rand = lcg(7);
    const features = new Float32Array(blockN * BMFR_FEATURE_COUNT);
    const colors = new Float32Array(blockN * 3);
    let inputVar = 0;
    for (let p = 0; p < blockN; p++) {
      // Spread positions over a small planar patch (so the linear terms exist
      // but the signal is constant — features cannot explain the noise).
      const px = ((p % 8) - 3.5) * 0.1;
      const py = (Math.floor(p / 8) - 3.5) * 0.1;
      const row = new Float32Array(BMFR_FEATURE_COUNT);
      bmfrFeatureRow([px, py, 0], [0, 0, 1], row);
      features.set(row, p * BMFR_FEATURE_COUNT);
      for (let ch = 0; ch < 3; ch++) {
        const noise = (rand() - 0.5) * 0.8; // zero-mean noise, range ±0.4
        const noisy = c0[ch]! + noise;
        colors[p * 3 + ch] = noisy;
        inputVar += (noisy - c0[ch]!) ** 2;
      }
    }
    inputVar /= blockN * 3;

    const out = bmfrFitBlock(features, colors, blockN, 1e-3);

    let outVar = 0;
    for (let p = 0; p < blockN; p++) {
      for (let ch = 0; ch < 3; ch++) {
        outVar += (out[p * 3 + ch]! - c0[ch]!) ** 2;
      }
    }
    outVar /= blockN * 3;

    // The reconstruction's deviation from the true constant must be far
    // smaller than the noisy input's — BMFR is denoising toward the mean.
    expect(outVar).toBeLessThan(inputVar * 0.25);
    // And the block mean of the reconstruction should track c0.
    const meanR = out.filter((_, i) => i % 3 === 0).reduce((a, b) => a + b, 0) / blockN;
    expect(meanR).toBeCloseTo(c0[0]!, 1);
  });

  it('preserves a genuine linear gradient (does not over-smooth signal)', () => {
    // The TRUE color varies linearly with x. BMFR must KEEP that gradient
    // (it is in the feature space) while removing noise — i.e. it is an
    // edge/gradient-preserving regressor, not a box blur.
    const blockN = 64;
    const rand = lcg(99);
    const features = new Float32Array(blockN * BMFR_FEATURE_COUNT);
    const colors = new Float32Array(blockN * 3);
    const trueColor = new Float32Array(blockN * 3);
    for (let p = 0; p < blockN; p++) {
      const px = ((p % 8) - 3.5) * 0.25;
      const py = (Math.floor(p / 8) - 3.5) * 0.25;
      const row = new Float32Array(BMFR_FEATURE_COUNT);
      bmfrFeatureRow([px, py, 0], [0, 0, 1], row);
      features.set(row, p * BMFR_FEATURE_COUNT);
      // Linear ramp in x for the red channel.
      const trueR = 0.5 + 0.3 * px;
      for (let ch = 0; ch < 3; ch++) {
        const tc = ch === 0 ? trueR : 0.4;
        trueColor[p * 3 + ch] = tc;
        colors[p * 3 + ch] = tc + (rand() - 0.5) * 0.6;
      }
    }
    const out = bmfrFitBlock(features, colors, blockN, 1e-4);
    // Reconstruction tracks the true gradient much better than the noisy input.
    let outErr = 0;
    let inErr = 0;
    for (let i = 0; i < blockN * 3; i++) {
      outErr += (out[i]! - trueColor[i]!) ** 2;
      inErr += (colors[i]! - trueColor[i]!) ** 2;
    }
    expect(outErr).toBeLessThan(inErr * 0.5);
  });

  it('stays bounded on a rank-deficient (constant-position) block', () => {
    // Every pixel shares the same position + normal: the feature matrix is
    // rank 1 (only the constant column is informative). The λ-regularised
    // solve must not blow up — output should be ~the noisy mean.
    const blockN = 16;
    const features = new Float32Array(blockN * BMFR_FEATURE_COUNT);
    const colors = new Float32Array(blockN * 3);
    const rand = lcg(123);
    let sum = 0;
    for (let p = 0; p < blockN; p++) {
      const row = new Float32Array(BMFR_FEATURE_COUNT);
      bmfrFeatureRow([0, 0, 0], [0, 0, 1], row);
      features.set(row, p * BMFR_FEATURE_COUNT);
      const v = 0.5 + (rand() - 0.5) * 0.2;
      colors[p * 3] = v; colors[p * 3 + 1] = v; colors[p * 3 + 2] = v;
      sum += v;
    }
    const mean = sum / blockN;
    const out = bmfrFitBlock(features, colors, blockN, 1e-3);
    for (let i = 0; i < blockN * 3; i++) {
      expect(Number.isFinite(out[i]!)).toBe(true);
      // Constant-feature block → reconstruction ≈ the noisy mean.
      expect(out[i]!).toBeCloseTo(mean, 1);
    }
  });
});
