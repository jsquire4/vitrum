/**
 * A3 — CPU spectral-vs-RGB transport invariant harness.
 *
 * This is the SELF-VALIDATING math harness the A3 item is built around. It
 * mirrors the GPU's hero-wavelength spectral transport (kernel.wgsl.ts +
 * material.wgsl.ts + shadePrologue.wgsl.ts) on the CPU and proves the two
 * load-bearing invariants:
 *
 *   1. FLAT-SPECTRUM INVARIANT. For a material whose RGB→spectrum upsampling is
 *      (near-)flat — i.e. a NEUTRAL grey reflectance and a NEUTRAL emitter — the
 *      Monte-Carlo spectral estimator converges (in expectation, over many hero-λ
 *      samples) to the SAME RGB radiance as the plain RGB single-bounce estimator.
 *      This is the invariant that guarantees spectralEnabled=true does not shift
 *      the energy of an achromatic scene: it is the radiometric anchor.
 *
 *   2. DISPERSIVE / CHROMATIC DIVERGENCE. For a SATURATED (chromatic) reflectance
 *      the spectral and RGB estimators correctly DIFFER (the spectral path carries
 *      a genuinely wavelength-resolved reflectance, the RGB path a tristimulus
 *      product) — proving the spectral path is not a no-op tint.
 *
 * The harness reuses the EXACT primitives the GPU uses, via their TS mirrors in
 * @vitrum/shared-samplers: the Jakob & Hanika upsampling (rgbToSpectralCoefficients
 * / evaluateSpectrum), the hero-λ MIS sampler (sampleHeroWavelengthMIS), and the
 * CMF reconstruction (wavelengthToRGB). The on-surface emission/env upsampling
 * (spectralEmissionAtHero) is mirrored here byte-for-byte from material.wgsl.ts.
 */
import { describe, expect, it } from 'vitest';
import {
  rgbToSpectralCoefficients,
  evaluateSpectrum,
  sampleHeroWavelengthMIS,
  wavelengthToRGB,
  HERO_LAMBDA_MIN,
  Y_CMF_INTEGRAL,
  CIE_D65_TABLE,
  CIE_Y_TABLE,
  CIE_LAMBDA_MIN,
  CIE_LAMBDA_STEP,
} from '@vitrum/shared-samplers';
import { PT_WEBGPU_PATH_TRACE_MATERIAL_FUNCS_WGSL } from '../wgsl/pathTrace/material.wgsl.js';

// ── Mirrors of the WGSL spectral helpers (material.wgsl.ts) ──────────────────

/** heroLambdaTo01 — WGSL: clamp((λ − 380)/400, 0, 1). */
function heroLambdaTo01(lambdaNm: number): number {
  return Math.min(1, Math.max(0, (lambdaNm - HERO_LAMBDA_MIN) / 400));
}

// D65 luminance normaliser (== heroWavelengthTables HERO_D65_Y_INTEGRAL).
const D65_Y_INTEGRAL = (() => {
  let n = 0;
  for (let i = 0; i < CIE_Y_TABLE.length; i++) {
    n += (CIE_D65_TABLE[i] ?? 0) * (CIE_Y_TABLE[i] ?? 0) * CIE_LAMBDA_STEP;
  }
  return n;
})();

/** heroSampleD65Normalised — WGSL mirror: D65(λ)·Y_INTEGRAL/D65_Y_INTEGRAL. */
function heroSampleD65Normalised(lambdaNm: number): number {
  // Linear-interp the D65 table at λ.
  if (lambdaNm < CIE_LAMBDA_MIN || lambdaNm > 780) return 0;
  const f = (lambdaNm - CIE_LAMBDA_MIN) / CIE_LAMBDA_STEP;
  const lo = Math.floor(f);
  const hi = Math.min(lo + 1, CIE_D65_TABLE.length - 1);
  const t = f - lo;
  const d65 = (CIE_D65_TABLE[lo] ?? 0) + t * ((CIE_D65_TABLE[hi] ?? 0) - (CIE_D65_TABLE[lo] ?? 0));
  return d65 * (Y_CMF_INTEGRAL / Math.max(D65_Y_INTEGRAL, 1e-9));
}

/**
 * spectralEmissionAtHero — byte-for-byte mirror of material.wgsl.ts. Maps an
 * authored RGB emitter/env colour to a scalar hero-λ radiance via a chroma weight
 * across the RGB primaries (short λ→B, mid→G, long→R) × the D65-normalised SPD.
 * A neutral colour reconstructs (with a flat reflectance) to its RGB exactly.
 */
function spectralEmissionAtHero(rgb: readonly [number, number, number], lambdaNm: number): number {
  const lum = Math.max(0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2], 0);
  if (lum < 1e-8) return 0;
  const t = heroLambdaTo01(lambdaNm);
  const wB = Math.max(1 - Math.abs(t - 0.15) / 0.35, 0);
  const wG = Math.max(1 - Math.abs(t - 0.5) / 0.35, 0);
  const wR = Math.max(1 - Math.abs(t - 0.85) / 0.35, 0);
  const wSum = Math.max(wR + wG + wB, 1e-6);
  const chroma = (rgb[0] * wR + rgb[1] * wG + rgb[2] * wB) / wSum;
  return chroma * heroSampleD65Normalised(lambdaNm);
}

function spectralRgbFactorAtHero(
  rgb: readonly [number, number, number],
  lambdaNm: number,
): number {
  const value = rgb.map((channel) => Math.max(channel, 0)) as [number, number, number];
  const t = heroLambdaTo01(lambdaNm);
  const wB = Math.max(1 - Math.abs(t - 0.15) / 0.35, 0);
  const wG = Math.max(1 - Math.abs(t - 0.5) / 0.35, 0);
  const wR = Math.max(1 - Math.abs(t - 0.85) / 0.35, 0);
  const wSum = Math.max(wR + wG + wB, 1e-6);
  return Math.max((value[0] * wR + value[1] * wG + value[2] * wB) / wSum, 0);
}

function activeLayerWeightRgb(
  layerRgb: readonly [number, number, number],
  lambdaNm: number,
  spectralEnabled: boolean,
): [number, number, number] {
  if (!spectralEnabled) return [...layerRgb];
  const weight = spectralRgbFactorAtHero(layerRgb, lambdaNm);
  return [weight, weight, weight];
}

function extractFunctionBody(src: string, name: string): string {
  const start = src.indexOf(`fn ${name}`);
  expect(start, `${name} exists`).toBeGreaterThanOrEqual(0);
  const open = src.indexOf('{', start);
  expect(open, `${name} body opens`).toBeGreaterThanOrEqual(0);
  let depth = 0;
  for (let i = open; i < src.length; i += 1) {
    const ch = src[i];
    if (ch === '{') depth += 1;
    if (ch === '}') {
      depth -= 1;
      if (depth === 0) {
        return src.slice(open + 1, i);
      }
    }
  }
  throw new Error(`${name} body did not close`);
}

/** evalJakobHanikaSpectrum — WGSL mirror (== evaluateSpectrum in shared-samplers). */
function evalSpectralReflectance(
  coeffs: readonly [number, number, number],
  lambdaNm: number,
): number {
  return evaluateSpectrum(coeffs, lambdaNm);
}

// Deterministic PCG-ish stratified sampler over [0,1) for the harness MC.
function* stratified(n: number): Generator<number> {
  for (let i = 0; i < n; i++) yield (i + 0.5) / n;
}

/**
 * RGB single-bounce estimator: a Lambertian surface of `albedo` lit by a
 * directional light of radiance `Li`, geometry term `g` (= nDotL/π folded as a
 * scalar; identical on both sides so it cancels in the comparison). Returns the
 * RGB radiance per channel.
 */
function rgbSingleBounce(
  albedo: readonly [number, number, number],
  Li: readonly [number, number, number],
  g: number,
): [number, number, number] {
  return [albedo[0] * Li[0] * g, albedo[1] * Li[1] * g, albedo[2] * Li[2] * g];
}

/**
 * Spectral single-bounce estimator mirroring the GPU: for each hero-λ sample,
 * throughput = S(λ) (spectral reflectance of the albedo), the light radiance is
 * spectralised via spectralEmissionAtHero, the product · g is reconstructed
 * through the CMF. Averages over `samples` hero-λ draws (full 3-strategy MIS via
 * sampleHeroWavelengthMIS, stratified on both the strategy and λ dimensions).
 */
function spectralSingleBounce(
  albedo: readonly [number, number, number],
  Li: readonly [number, number, number],
  g: number,
  samples: number,
): [number, number, number] {
  const coeffs = rgbToSpectralCoefficients(albedo[0], albedo[1], albedo[2]);
  let r = 0;
  let gg = 0;
  let b = 0;
  let count = 0;
  // Stratify the strategy and λ dimensions independently for low-variance MC.
  const sStrata = Math.max(3, Math.round(Math.sqrt(samples)));
  const lStrata = Math.max(3, Math.round(samples / sStrata));
  for (const us of stratified(sStrata)) {
    for (const ul of stratified(lStrata)) {
      const { lambdaNm, pdf } = sampleHeroWavelengthMIS(us, ul);
      const reflS = evalSpectralReflectance(coeffs, lambdaNm);
      const liS = spectralEmissionAtHero(Li, lambdaNm);
      const scalarRadiance = reflS * liS * g;
      const [cr, cg, cb] = wavelengthToRGB(lambdaNm, scalarRadiance, pdf);
      r += cr;
      gg += cg;
      b += cb;
      count++;
    }
  }
  return [r / count, gg / count, b / count];
}

describe('A3 — spectral transport invariant harness (CPU mirror)', () => {
  const g = 0.7; // arbitrary shared geometry/normalisation factor (cancels)

  it('FLAT-SPECTRUM INVARIANT: neutral grey ⇒ spectral == RGB in expectation', () => {
    // Neutral grey reflectance, neutral white light — the achromatic anchor.
    const albedo: [number, number, number] = [0.5, 0.5, 0.5];
    const Li: [number, number, number] = [1, 1, 1];
    const rgb = rgbSingleBounce(albedo, Li, g);
    const spec = spectralSingleBounce(albedo, Li, g, 4096);
    // The spectral estimator must reconstruct the SAME RGB radiance (within MC +
    // upsampling-roundtrip tolerance). This is the load-bearing invariant.
    for (let c = 0; c < 3; c++) {
      expect(spec[c]!).toBeCloseTo(rgb[c]!, 1); // ~5% — MC + Jakob-Hanika round-trip
    }
    // And it must be genuinely non-trivial (not all zero).
    expect(spec[0] + spec[1] + spec[2]).toBeGreaterThan(0.1);
  });

  it('FLAT-SPECTRUM INVARIANT holds across grey levels (0.2, 0.5, 0.8)', () => {
    const Li: [number, number, number] = [1, 1, 1];
    for (const grey of [0.2, 0.5, 0.8]) {
      const albedo: [number, number, number] = [grey, grey, grey];
      const rgb = rgbSingleBounce(albedo, Li, g);
      const spec = spectralSingleBounce(albedo, Li, g, 4096);
      // Ratio of total energy spectral/RGB ≈ 1 (energy conservation at neutral).
      const sumRgb = rgb[0] + rgb[1] + rgb[2];
      const sumSpec = spec[0] + spec[1] + spec[2];
      expect(sumSpec / sumRgb).toBeGreaterThan(0.9);
      expect(sumSpec / sumRgb).toBeLessThan(1.1);
    }
  });

  it('DISPERSIVE/CHROMATIC: a TWO-bounce chromatic path makes spectral ≠ RGB', () => {
    // The single-bounce case is (correctly) near-identical: Jakob-Hanika fits the
    // reflectance to reproduce the RGB under D65, so red·white ≈ red on both
    // paths. The genuine spectral divergence appears under MULTIPLICATION of
    // reflectances — colour bleeding. For a saturated red surface seen via a
    // second red bounce, RGB computes albedo·albedo per channel, but the spectral
    // path computes ∫ S(λ)²·D65·CMF, which is NOT the same (S² narrows the band —
    // saturated colours darken/shift, the metameric/colour-bleed effect spectral
    // rendering captures and RGB cannot). This divergence is the proof the
    // transport is genuinely wavelength-resolved, not a luminance tint.
    const albedo: [number, number, number] = [0.9, 0.05, 0.05];
    const Li: [number, number, number] = [1, 1, 1];

    // RGB two-bounce: albedo² · Li · g (per channel).
    const rgb2: [number, number, number] = [
      albedo[0] * albedo[0] * Li[0] * g,
      albedo[1] * albedo[1] * Li[1] * g,
      albedo[2] * albedo[2] * Li[2] * g,
    ];

    // Spectral two-bounce: S(λ)² · D65(λ) · g reconstructed through the CMF.
    const coeffs = rgbToSpectralCoefficients(albedo[0], albedo[1], albedo[2]);
    let r = 0;
    let gg = 0;
    let b = 0;
    let count = 0;
    const M = 64;
    for (const us of stratified(M)) {
      for (const ul of stratified(M)) {
        const { lambdaNm, pdf } = sampleHeroWavelengthMIS(us, ul);
        const reflS = evaluateSpectrum(coeffs, lambdaNm);
        const liS = spectralEmissionAtHero(Li, lambdaNm);
        const scalar = reflS * reflS * liS * g; // TWO reflectance bounces
        const [cr, cg, cb] = wavelengthToRGB(lambdaNm, scalar, pdf);
        r += cr;
        gg += cg;
        b += cb;
        count++;
      }
    }
    const spec2: [number, number, number] = [r / count, gg / count, b / count];

    const maxDelta = Math.max(
      Math.abs(spec2[0] - rgb2[0]),
      Math.abs(spec2[1] - rgb2[1]),
      Math.abs(spec2[2] - rgb2[2]),
    );
    expect(maxDelta).toBeGreaterThan(0.02);
  });

  it('round-trip: Jakob-Hanika upsample → CMF integral recovers a neutral grey', () => {
    // Standalone check of the upsampling round-trip used by the transport: the
    // full CMF integral of the upsampled spectrum reconstructs the input grey.
    const grey = 0.5;
    const coeffs = rgbToSpectralCoefficients(grey, grey, grey);
    // Integrate S(λ) against the hero-λ MIS estimator with a constant unit "light".
    let lum = 0;
    let count = 0;
    const N = 64;
    for (const us of stratified(N)) {
      for (const ul of stratified(N)) {
        const { lambdaNm, pdf } = sampleHeroWavelengthMIS(us, ul);
        // Illuminate with the D65-normalised SPD (the reflectance is D65-relative).
        const reflS = evaluateSpectrum(coeffs, lambdaNm) * heroSampleD65Normalised(lambdaNm);
        const [r, gch, b] = wavelengthToRGB(lambdaNm, reflS, pdf);
        lum += 0.2126 * r + 0.7152 * gch + 0.0722 * b;
        count++;
      }
    }
    lum /= count;
    expect(lum).toBeCloseTo(grey, 1); // recovers ~0.5 within round-trip tolerance
  });

  it('activeLayerWeightRgb passes RGB through when spectral is disabled', () => {
    expect(activeLayerWeightRgb([0.8, 0.3, 0.1], 540, false)).toEqual([0.8, 0.3, 0.1]);
  });

  it('activeLayerWeightRgb evaluates chromatic layer transmission at the hero wavelength', () => {
    const lambdaNm = 540;
    const red: [number, number, number] = [1, 0, 0];
    const sameLumGreen: [number, number, number] = [0, 0.2126 / 0.7152, 0];

    const redWeight = activeLayerWeightRgb(red, lambdaNm, true);
    const greenWeight = activeLayerWeightRgb(sameLumGreen, lambdaNm, true);
    expect(redWeight[0]).toBeCloseTo(0, 12);
    expect(greenWeight[0]).toBeGreaterThan(0.1);
    expect(activeLayerWeightRgb([0.37, 0.37, 0.37], 630, true)).toEqual([0.37, 0.37, 0.37]);
  });

  it('activeLayerWeightRgb clamps negative spectral layer luminance to black', () => {
    expect(activeLayerWeightRgb([-1, -0.5, -0.25], 540, true)).toEqual([0, 0, 0]);
  });

  it('production WGSL keeps the active-layer helper on the oracle semantics', () => {
    const body = extractFunctionBody(PT_WEBGPU_PATH_TRACE_MATERIAL_FUNCS_WGSL, 'activeLayerWeightRgb');
    expect(body).toContain('if (!spectralEnabled)');
    expect(body).toContain('return layerRgb;');
    expect(body).toContain('spectralRgbFactorAtHero(layerRgb, heroLambda)');
    expect(body).not.toContain('heroWavelengthToRgb');
  });
});
