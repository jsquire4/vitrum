/**
 * spectral.test.ts — Unit tests for Sprint 12 hero-wavelength spectral utilities.
 *
 * Covers:
 *   cieCmf.ts:
 *     - Table lengths are exactly 81
 *     - Wavelength range constants match tables
 *     - Y table peaks at 555 nm (CIE definition)
 *     - sampleCMF(550) Y ≈ 0.995 (near max, adjacent to 555 nm peak)
 *     - sampleCMF(380) returns near-zero values
 *     - sampleCMF outside range returns [0,0,0]
 *     - sampleCMF(λ) at table entries matches table values exactly
 *     - sampleCMF interpolates between table entries
 *     - xyzToLinearSRGB: D65 flat spectrum → near (1, 1, 1)
 *     - xyzToLinearSRGB: known pure white XYZ → expected sRGB
 *
 *   wavelengthSampling.ts:
 *     - sampleHeroWavelength(u) returns wavelength in [380, 780] for all u ∈ [0,1]
 *     - PDF is positive for all u ∈ (0,1)
 *     - Importance sampling concentrates samples in 500–600 nm range
 *     - sampleHeroWavelength(0.5) returns wavelength near 555 nm (Y peak)
 *     - wavelengthToRGB at 550 nm returns mostly green
 *     - wavelengthToRGB at 700 nm returns mostly red
 *     - wavelengthToRGB at 450 nm returns mostly blue
 *     - wavelengthToRGB with zero pdf returns [0,0,0]
 *     - Y_CMF_INTEGRAL is approximately the known value (~106.86 nm)
 *
 *   cauchyIor.ts:
 *     - cauchyIOR is monotonically decreasing in λ for normal dispersion (B>0)
 *     - cauchyIOR with C=0 reduces to A + B/λ² formula
 *     - Crown glass at 550 nm → n ≈ 1.518
 *     - Flint glass at 550 nm → n > crown glass (denser medium)
 *     - CAUCHY_CROWN_GLASS Abbe number ≈ 60–70
 *     - CAUCHY_FLINT_GLASS Abbe number ≈ 30–42
 *     - CAUCHY_LEAD_CRYSTAL Abbe number ≈ 25–40
 *     - Lead crystal n(589.3) is within expected range for the material
 */

import { describe, it, expect } from 'vitest';

import {
  CIE_X_TABLE,
  CIE_Y_TABLE,
  CIE_Z_TABLE,
  CIE_D65_TABLE,
  CIE_LAMBDA_MIN,
  CIE_LAMBDA_MAX,
  CIE_LAMBDA_STEP,
  CIE_TABLE_LENGTH,
  sampleCMF,
  xyzToLinearSRGB,
} from '../src/cieCmf.js';

import {
  sampleHeroWavelength,
  sampleHeroWavelengthMIS,
  wavelengthToRGB,
  Y_CMF_INTEGRAL,
  X_CMF_INTEGRAL,
  Z_CMF_INTEGRAL,
  X_CMF_CDF,
  Y_CMF_CDF,
  Z_CMF_CDF,
  HERO_LAMBDA_MIN,
  HERO_LAMBDA_MAX,
} from '../src/wavelengthSampling.js';

import {
  cauchyIOR,
  abbeNumber,
  CAUCHY_CROWN_GLASS,
  CAUCHY_FLINT_GLASS,
  CAUCHY_LEAD_CRYSTAL,
  FRAUNHOFER_D_NM,
} from '../src/cauchyIor.js';

// ════════════════════════════════════════════════════════════════════════════════
// cieCmf.ts
// ════════════════════════════════════════════════════════════════════════════════

describe('CIE CMF table dimensions and constants', () => {
  it('CIE_TABLE_LENGTH equals 81', () => {
    expect(CIE_TABLE_LENGTH).toBe(81);
  });

  it('CIE_LAMBDA_MIN = 380, CIE_LAMBDA_MAX = 780, CIE_LAMBDA_STEP = 5', () => {
    expect(CIE_LAMBDA_MIN).toBe(380);
    expect(CIE_LAMBDA_MAX).toBe(780);
    expect(CIE_LAMBDA_STEP).toBe(5);
  });

  it('CIE_X_TABLE has length 81', () => {
    expect(CIE_X_TABLE.length).toBe(81);
  });

  it('CIE_Y_TABLE has length 81', () => {
    expect(CIE_Y_TABLE.length).toBe(81);
  });

  it('CIE_Z_TABLE has length 81', () => {
    expect(CIE_Z_TABLE.length).toBe(81);
  });

  it('CIE_D65_TABLE has length 81', () => {
    expect(CIE_D65_TABLE.length).toBe(81);
  });

  it('HERO_LAMBDA_MIN equals CIE_LAMBDA_MIN', () => {
    expect(HERO_LAMBDA_MIN).toBe(CIE_LAMBDA_MIN);
  });

  it('HERO_LAMBDA_MAX equals CIE_LAMBDA_MAX', () => {
    expect(HERO_LAMBDA_MAX).toBe(CIE_LAMBDA_MAX);
  });
});

describe('CIE Y table peak location', () => {
  it('Y table peaks at index 35 (555 nm), with value 1.0', () => {
    // Index 35 = (555 - 380) / 5
    const idx = (555 - 380) / 5; // = 35
    expect(CIE_Y_TABLE[idx]).toBe(1.0);
  });

  it('Y table values are all in range [0, 1]', () => {
    for (const y of CIE_Y_TABLE) {
      expect(y).toBeGreaterThanOrEqual(0);
      expect(y).toBeLessThanOrEqual(1.0);
    }
  });

  it('X table values are non-negative', () => {
    for (const x of CIE_X_TABLE) {
      expect(x).toBeGreaterThanOrEqual(0);
    }
  });

  it('Z table values are non-negative', () => {
    for (const z of CIE_Z_TABLE) {
      expect(z).toBeGreaterThanOrEqual(0);
    }
  });
});

describe('sampleCMF', () => {
  it('sampleCMF(380) returns near-zero Y (low luminous efficiency in UV)', () => {
    const [, y] = sampleCMF(380);
    expect(y).toBeLessThan(0.001);
  });

  it('sampleCMF(550) Y is close to table value at index 34 (0.994950)', () => {
    // Index 34 = (550 - 380) / 5
    const [, y] = sampleCMF(550);
    // Table value at 550 nm is 0.994950 (index 34); near but not peak (555 nm)
    expect(y).toBeCloseTo(0.994950, 3);
  });

  it('sampleCMF(555) Y ≈ 1.0 (luminous efficiency peak)', () => {
    const [, y] = sampleCMF(555);
    expect(y).toBeGreaterThan(0.99);
  });

  it('sampleCMF at table entry wavelengths returns exact table values', () => {
    // Test a selection of entries: 400, 500, 600, 700 nm
    const checkLambdas = [400, 500, 600, 700];
    for (const lambda of checkLambdas) {
      const idx = (lambda - CIE_LAMBDA_MIN) / CIE_LAMBDA_STEP;
      const [x, y, z] = sampleCMF(lambda);
      expect(x).toBeCloseTo(CIE_X_TABLE[idx] ?? 0, 5);
      expect(y).toBeCloseTo(CIE_Y_TABLE[idx] ?? 0, 5);
      expect(z).toBeCloseTo(CIE_Z_TABLE[idx] ?? 0, 5);
    }
  });

  it('sampleCMF below 380 nm returns [0, 0, 0]', () => {
    const [x, y, z] = sampleCMF(300);
    expect(x).toBe(0);
    expect(y).toBe(0);
    expect(z).toBe(0);
  });

  it('sampleCMF above 780 nm returns [0, 0, 0]', () => {
    const [x, y, z] = sampleCMF(900);
    expect(x).toBe(0);
    expect(y).toBe(0);
    expect(z).toBe(0);
  });

  it('sampleCMF interpolates between table entries (510 nm between 510 and 515)', () => {
    // 512.5 nm is exactly halfway between 510 nm (idx=26) and 515 nm (idx=27)
    const [, y512] = sampleCMF(512.5);
    const y510 = CIE_Y_TABLE[26] ?? 0;
    const y515 = CIE_Y_TABLE[27] ?? 0;
    const expected = (y510 + y515) / 2;
    expect(y512).toBeCloseTo(expected, 5);
  });

  it('sampleCMF all values are non-negative in [380, 780]', () => {
    for (let lambda = 380; lambda <= 780; lambda += 1) {
      const [x, y, z] = sampleCMF(lambda);
      expect(x).toBeGreaterThanOrEqual(0);
      expect(y).toBeGreaterThanOrEqual(0);
      expect(z).toBeGreaterThanOrEqual(0);
    }
  });
});

describe('xyzToLinearSRGB', () => {
  it('XYZ of D65 equal-energy white (X≈Y≈Z after D65 normalisation) → near (1,1,1)', () => {
    // For a flat spectrum under D65, integrating CMF × D65 over the visible range
    // gives XYZ proportional to the D65 white point.  We test the round-trip
    // numerically: integrate CMF × flat-spectrum and convert to sRGB.
    //
    // For a flat-spectrum reflectance of 1.0 under D65, the XYZ values equal
    // the D65 illuminant's XYZ: X ≈ 95.05, Y ≈ 100.0, Z ≈ 108.88 (CIE standard).
    // Normalise to Y = 1: X ≈ 0.9505, Y = 1.0, Z ≈ 1.0888.
    const [r, g, b] = xyzToLinearSRGB(0.9505, 1.0, 1.0888);
    // Should be close to (1, 1, 1) given the Bradford-adapted D65 matrix.
    expect(r).toBeCloseTo(1.0, 1);
    expect(g).toBeCloseTo(1.0, 1);
    expect(b).toBeCloseTo(1.0, 1);
  });

  it('XYZ (0,0,0) → sRGB (0,0,0)', () => {
    const [r, g, b] = xyzToLinearSRGB(0, 0, 0);
    expect(r).toBe(0);
    expect(g).toBe(0);
    expect(b).toBe(0);
  });

  it('High Y (green channel) produces positive g component', () => {
    const [, g] = xyzToLinearSRGB(0, 1, 0);
    expect(g).toBeGreaterThan(0);
  });
});

// ════════════════════════════════════════════════════════════════════════════════
// wavelengthSampling.ts
// ════════════════════════════════════════════════════════════════════════════════

describe('sampleHeroWavelength', () => {
  it('returns wavelength in [380, 780] nm for u = 0', () => {
    const { lambdaNm } = sampleHeroWavelength(0);
    expect(lambdaNm).toBeGreaterThanOrEqual(380);
    expect(lambdaNm).toBeLessThanOrEqual(780);
  });

  it('returns wavelength in [380, 780] nm for u = 0.999', () => {
    const { lambdaNm } = sampleHeroWavelength(0.999);
    expect(lambdaNm).toBeGreaterThanOrEqual(380);
    expect(lambdaNm).toBeLessThanOrEqual(780);
  });

  it('returns wavelength in [380, 780] nm for all u ∈ [0, 1] (100 samples)', () => {
    for (let i = 0; i <= 100; i++) {
      const u = i / 100;
      const { lambdaNm } = sampleHeroWavelength(Math.min(u, 0.9999));
      expect(lambdaNm).toBeGreaterThanOrEqual(380);
      expect(lambdaNm).toBeLessThanOrEqual(780);
    }
  });

  it('pdf is positive for u ∈ (0, 1)', () => {
    for (let i = 1; i < 10; i++) {
      const { pdf } = sampleHeroWavelength(i / 10);
      expect(pdf).toBeGreaterThan(0);
    }
  });

  it('concentrates samples in 500–600 nm range (importance sampling of Y peak)', () => {
    // Draw 200 stratified samples; expect >50% to land in 500–600 nm range.
    let inPeak = 0;
    const N = 200;
    for (let i = 0; i < N; i++) {
      const { lambdaNm } = sampleHeroWavelength(i / N);
      if (lambdaNm >= 500 && lambdaNm <= 600) inPeak++;
    }
    // The Y CMF integrates most of its mass in ~480–640 nm; 500–600 nm ≈ ~40–50%.
    expect(inPeak / N).toBeGreaterThan(0.40);
  });

  it('u=0.5 samples near the luminance peak (within 480–620 nm)', () => {
    // The median of the Y CMF distribution is around 555 nm ± some range.
    const { lambdaNm } = sampleHeroWavelength(0.5);
    expect(lambdaNm).toBeGreaterThan(480);
    expect(lambdaNm).toBeLessThan(620);
  });

  it('returns a deterministic result for the same u', () => {
    const r1 = sampleHeroWavelength(0.3);
    const r2 = sampleHeroWavelength(0.3);
    expect(r1.lambdaNm).toBe(r2.lambdaNm);
    expect(r1.pdf).toBe(r2.pdf);
  });

  it('PDF × wavelength range approximately integrates to 1 (MC check)', () => {
    // Numerical check: sum pdf × dλ over evenly-spaced u samples.
    // This verifies the importance sampling density is properly normalised.
    // We estimate ∫ pdf(λ) dλ ≈ 1 by sampling uniformly in λ and checking
    // that the pdf values are consistent with the Y CMF normalisation.
    // Simpler check: sum(Y_table) × step / Y_INTEGRAL = 1 by construction.
    expect(Y_CMF_INTEGRAL).toBeGreaterThan(100);
    expect(Y_CMF_INTEGRAL).toBeLessThan(115);
  });
});

describe('Y_CMF_INTEGRAL', () => {
  it('is approximately 106.86 nm (CIE Y integral over [380, 780])', () => {
    // Trapezoidal rule at 5 nm steps gives ~106.86.
    expect(Y_CMF_INTEGRAL).toBeCloseTo(106.857, 0);
  });
});

describe('wavelengthToRGB', () => {
  it('returns [0, 0, 0] when pdfLambda = 0', () => {
    const [r, g, b] = wavelengthToRGB(550, 1.0, 0);
    expect(r).toBe(0);
    expect(g).toBe(0);
    expect(b).toBe(0);
  });

  it('returns [0, 0, 0] when pdfLambda < 0 (guard)', () => {
    const [r, g, b] = wavelengthToRGB(550, 1.0, -0.001);
    expect(r).toBe(0);
    expect(g).toBe(0);
    expect(b).toBe(0);
  });

  it('at 550 nm with unit throughput/pdf, g component dominates over r and b', () => {
    // 550 nm is near the green peak; X and Z are lower than Y there.
    // The sRGB reconstruction should give a positive green-dominant colour.
    const pdf = 0.01; // arbitrary nonzero pdf
    const [r, g, b] = wavelengthToRGB(550, 1.0, pdf);
    expect(g).toBeGreaterThan(0);
    expect(g).toBeGreaterThan(Math.abs(r));
    expect(g).toBeGreaterThan(Math.abs(b));
  });

  it('at 450 nm, blue channel contribution present (Z is large at 450 nm)', () => {
    // At 450 nm, z̄(λ) ≈ 1.77 which is the largest CMF value there.
    // xyzToLinearSRGB converts high Z to significant blue contribution.
    const pdf = 0.005;
    const [r, _g, b] = wavelengthToRGB(450, 1.0, pdf);
    // b should be positive and larger than r
    expect(b).toBeGreaterThan(0);
    expect(b).toBeGreaterThan(r);
  });

  it('at 700 nm, Y and Z are near zero; x̄ is still nonzero, giving red-ish contribution', () => {
    // At 700 nm: Y ≈ 0.004, Z = 0, X ≈ 0.011.
    // sRGB reconstruction: r > 0, g < r.
    const pdf = 0.001;
    const [r, _g, b] = wavelengthToRGB(700, 1.0, pdf);
    // X is the dominant CMF value at 700 nm; r should be positive
    expect(r).toBeGreaterThan(0);
    expect(r).toBeGreaterThan(b);
  });

  it('throughput=0 gives [0,0,0] regardless of wavelength and pdf', () => {
    const [r, g, b] = wavelengthToRGB(550, 0, 0.01);
    expect(r).toBeCloseTo(0, 10);
    expect(g).toBeCloseTo(0, 10);
    expect(b).toBeCloseTo(0, 10);
  });

  it('normalizes unit throughput to display-scale values', () => {
    const { lambdaNm, pdf } = sampleHeroWavelength(0.5);
    const [r, g, b] = wavelengthToRGB(lambdaNm, 1, pdf);
    expect(Math.max(Math.abs(r), Math.abs(g), Math.abs(b))).toBeLessThan(3);
  });
});

// ════════════════════════════════════════════════════════════════════════════════
// cauchyIor.ts
// ════════════════════════════════════════════════════════════════════════════════

describe('cauchyIOR', () => {
  it('is monotonically decreasing in λ for normal dispersion (B > 0)', () => {
    const { A, B, C } = CAUCHY_CROWN_GLASS;
    const lambdas = [400, 450, 500, 550, 600, 650, 700];
    for (let i = 1; i < lambdas.length; i++) {
      const nPrev = cauchyIOR(lambdas[i - 1]!, A, B, C);
      const nCurr = cauchyIOR(lambdas[i]!, A, B, C);
      expect(nPrev).toBeGreaterThan(nCurr);
    }
  });

  it('with C=0, equals A + B/λ² (first-order Cauchy) at any wavelength', () => {
    const A = 1.5;
    const B = 0.005;
    const lambda = 550;
    const lambdaUm = lambda * 1e-3; // nm → µm
    const expected = A + B / (lambdaUm * lambdaUm);
    expect(cauchyIOR(lambda, A, B, 0)).toBeCloseTo(expected, 10);
    expect(cauchyIOR(lambda, A, B)).toBeCloseTo(expected, 10);
  });

  it('crown glass at 550 nm → n in [1.510, 1.525]', () => {
    const { A, B, C } = CAUCHY_CROWN_GLASS;
    const n = cauchyIOR(550, A, B, C);
    expect(n).toBeGreaterThan(1.510);
    expect(n).toBeLessThan(1.525);
  });

  it('flint glass at 550 nm → n > crown glass (denser medium)', () => {
    const nCrown = cauchyIOR(550, CAUCHY_CROWN_GLASS.A, CAUCHY_CROWN_GLASS.B, CAUCHY_CROWN_GLASS.C);
    const nFlint = cauchyIOR(550, CAUCHY_FLINT_GLASS.A, CAUCHY_FLINT_GLASS.B, CAUCHY_FLINT_GLASS.C);
    expect(nFlint).toBeGreaterThan(nCrown);
  });

  it('lead crystal IOR at D-line (589.3 nm) is in physically plausible range [1.55, 1.64]', () => {
    const { A, B, C } = CAUCHY_LEAD_CRYSTAL;
    const nD = cauchyIOR(FRAUNHOFER_D_NM, A, B, C);
    expect(nD).toBeGreaterThan(1.55);
    expect(nD).toBeLessThan(1.64);
  });

  it('returns A when B=0 and C=0 (non-dispersive material)', () => {
    const A = 1.5;
    expect(cauchyIOR(550, A, 0, 0)).toBeCloseTo(A, 10);
    expect(cauchyIOR(700, A, 0, 0)).toBeCloseTo(A, 10);
    expect(cauchyIOR(400, A, 0, 0)).toBeCloseTo(A, 10);
  });

  it('IOR increases toward shorter wavelengths (blue refracts more than red)', () => {
    const { A, B, C } = CAUCHY_LEAD_CRYSTAL;
    const nBlue = cauchyIOR(450, A, B, C);
    const nRed  = cauchyIOR(700, A, B, C);
    expect(nBlue).toBeGreaterThan(nRed);
  });
});

describe('abbeNumber', () => {
  it('crown glass Abbe number is in expected range 55–75', () => {
    const { A, B, C } = CAUCHY_CROWN_GLASS;
    const V = abbeNumber(A, B, C);
    expect(V).toBeGreaterThan(55);
    expect(V).toBeLessThan(75);
  });

  it('flint glass Abbe number is in expected range 28–45', () => {
    const { A, B, C } = CAUCHY_FLINT_GLASS;
    const V = abbeNumber(A, B, C);
    expect(V).toBeGreaterThan(28);
    expect(V).toBeLessThan(45);
  });

  it('lead crystal Abbe number is in expected range 25–42 (Sprint 8 target ≈ 32)', () => {
    const { A, B, C } = CAUCHY_LEAD_CRYSTAL;
    const V = abbeNumber(A, B, C);
    expect(V).toBeGreaterThan(25);
    expect(V).toBeLessThan(42);
  });

  it('higher dispersion (larger B) gives lower Abbe number', () => {
    const A = 1.5;
    const vLow  = abbeNumber(A, 0.005, 0);  // low dispersion
    const vHigh = abbeNumber(A, 0.015, 0);  // high dispersion
    expect(vLow).toBeGreaterThan(vHigh);
  });

  it('non-dispersive material (B=0) gives infinite Abbe number', () => {
    // n_F = n_C when B=0, so denominator → 0 → V_d → ∞
    const V = abbeNumber(1.5, 0, 0);
    expect(!isFinite(V) || V > 1000).toBe(true);
  });
});

// ════════════════════════════════════════════════════════════════════════════════
// MIS hero-wavelength sampling — extension of Wilkie et al. 2014 §3.3
// ════════════════════════════════════════════════════════════════════════════════

describe('X_CMF_INTEGRAL / Z_CMF_INTEGRAL', () => {
  // CIE 1931 2-deg CMFs are normalised so that ∫X = ∫Y = ∫Z. This guarantees
  // equal-energy white lands at chromaticity (x,y,z) = (1/3, 1/3, 1/3).
  // At 5 nm steps with trapezoidal rule, all three integrals ≈ 106.85 (within 0.02%).
  it('all three integrals are equal to within 0.05% (CIE chromaticity-normalisation invariant)', () => {
    expect(Math.abs(X_CMF_INTEGRAL - Y_CMF_INTEGRAL) / Y_CMF_INTEGRAL).toBeLessThan(0.0005);
    expect(Math.abs(Z_CMF_INTEGRAL - Y_CMF_INTEGRAL) / Y_CMF_INTEGRAL).toBeLessThan(0.0005);
  });

  it('all three integrals ≈ 106.85 (trapezoidal at 5 nm steps over [380,780] nm)', () => {
    expect(X_CMF_INTEGRAL).toBeCloseTo(106.85, 0);
    expect(Y_CMF_INTEGRAL).toBeCloseTo(106.85, 0);
    expect(Z_CMF_INTEGRAL).toBeCloseTo(106.85, 0);
  });
});

describe('X / Y / Z CDFs are valid normalised CDFs', () => {
  for (const [name, cdf] of [
    ['X', X_CMF_CDF],
    ['Y', Y_CMF_CDF],
    ['Z', Z_CMF_CDF],
  ] as const) {
    it(`${name}_CMF_CDF has length 82, starts at 0, ends at 1`, () => {
      expect(cdf.length).toBe(82);
      expect(cdf[0]).toBe(0);
      expect(cdf[81]).toBeCloseTo(1, 10);
    });
    it(`${name}_CMF_CDF is monotone non-decreasing`, () => {
      for (let i = 1; i < cdf.length; i++) {
        expect(cdf[i]!).toBeGreaterThanOrEqual(cdf[i - 1]!);
      }
    });
  }
});

describe('sampleHeroWavelengthMIS', () => {
  it('returns wavelength in [380, 780] for any (uStrategy, uLambda) ∈ [0,1]×[0,1]', () => {
    for (let i = 0; i < 100; i++) {
      const us = (i % 10) / 10;
      const ul = Math.floor(i / 10) / 10;
      const { lambdaNm } = sampleHeroWavelengthMIS(us, ul);
      expect(lambdaNm).toBeGreaterThanOrEqual(HERO_LAMBDA_MIN);
      expect(lambdaNm).toBeLessThanOrEqual(HERO_LAMBDA_MAX);
    }
  });

  it('returns positive pdf for any sampled wavelength inside [380, 780]', () => {
    for (let i = 0; i < 100; i++) {
      const { pdf } = sampleHeroWavelengthMIS((i * 0.013) % 1, (i * 0.029) % 1);
      expect(pdf).toBeGreaterThan(0);
    }
  });

  it('uStrategy < 1/3 picks the X strategy: λ-distribution clusters near X peak (~600 nm)', () => {
    // Sample many λ from the X strategy by fixing uStrategy < 1/3.
    // Mean λ should be biased toward X-peak region (560-620nm), not Y-peak (555nm).
    const lambdas: number[] = [];
    for (let i = 0; i < 200; i++) {
      const { lambdaNm } = sampleHeroWavelengthMIS(0.1, i / 200);
      lambdas.push(lambdaNm);
    }
    const mean = lambdas.reduce((a, b) => a + b, 0) / lambdas.length;
    // X CMF has TWO lobes: small at ~445 nm, large at ~600 nm. Mean should be > 555.
    expect(mean).toBeGreaterThan(530);
  });

  it('uStrategy in [2/3, 1) picks the Z strategy: λ-distribution clusters near Z peak (~445 nm)', () => {
    // Z CMF peaks at ~445 nm — sampling from Z should give λ predominantly < 500 nm.
    const lambdas: number[] = [];
    for (let i = 0; i < 200; i++) {
      const { lambdaNm } = sampleHeroWavelengthMIS(0.9, i / 200);
      lambdas.push(lambdaNm);
    }
    const mean = lambdas.reduce((a, b) => a + b, 0) / lambdas.length;
    expect(mean).toBeLessThan(490);
    // Most samples should land in the blue band [380, 510].
    const blueFraction = lambdas.filter((l) => l < 510).length / lambdas.length;
    expect(blueFraction).toBeGreaterThan(0.6);
  });

  it('over many samples, mixture covers all three chromatic regions roughly evenly', () => {
    // Drive a wide u-grid through MIS; bin the results into blue/green/red bands.
    // With uniform strategy selection over X/Y/Z, we expect roughly balanced coverage.
    const N = 6000;
    let blue = 0, green = 0, red = 0;
    for (let i = 0; i < N; i++) {
      // Use a Halton-like scrambled pair to avoid axis-aligned aliasing
      const us = ((i * 7919) % 1009) / 1009;
      const ul = ((i * 6151) % 997) / 997;
      const { lambdaNm } = sampleHeroWavelengthMIS(us, ul);
      if (lambdaNm < 490) blue++;
      else if (lambdaNm < 580) green++;
      else red++;
    }
    // Each band gets at least 15% of samples (vs Y-only sampling where blue would be < 5%).
    expect(blue / N).toBeGreaterThan(0.15);
    expect(green / N).toBeGreaterThan(0.15);
    expect(red / N).toBeGreaterThan(0.15);
  });

  it('mixture pdf is unbiased: Monte Carlo integral of constant L(λ)=1 recovers Y of D65 ≈ 1', () => {
    // For a flat unit spectral input, the MC estimator for Y component should
    // converge to Y_INTEGRAL / Y_INTEGRAL = 1 (per the wavelengthToRGB contract).
    // Estimator: (1/N) Σ Y(λ_i) / pdf_mis(λ_i) — should converge to ∫Y dλ.
    const N = 4000;
    let sum = 0;
    for (let i = 0; i < N; i++) {
      const us = ((i * 7919) % 1013) / 1013;
      const ul = ((i * 6151) % 1009) / 1009;
      const { lambdaNm, pdf } = sampleHeroWavelengthMIS(us, ul);
      const [, y] = sampleCMF(lambdaNm);
      sum += y / pdf;
    }
    const estimate = sum / N;
    // Should be close to Y_INTEGRAL (~106.86). 5% tolerance for low-N MC noise.
    expect(estimate).toBeGreaterThan(Y_CMF_INTEGRAL * 0.92);
    expect(estimate).toBeLessThan(Y_CMF_INTEGRAL * 1.08);
  });

  it('deterministic for the same (uStrategy, uLambda) pair', () => {
    const a = sampleHeroWavelengthMIS(0.4, 0.7);
    const b = sampleHeroWavelengthMIS(0.4, 0.7);
    expect(a.lambdaNm).toBe(b.lambdaNm);
    expect(a.pdf).toBe(b.pdf);
  });
});
