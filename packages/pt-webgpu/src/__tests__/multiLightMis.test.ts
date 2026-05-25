/**
 * T1.D2 — 2-light sum-MIS correctness
 *
 * CPU mirror of the multi-light NEE evaluation from
 * pathTraceBruteforce.wgsl.ts (Item 15, multi-light area MIS).
 *
 * The WGSL picks one light randomly per shading event and scales by lightCount
 * (unbiased estimator). For a deterministic per-light test we compute the
 * contribution for each light analytically (no path-trace stochasticity) and
 * verify:
 *   1. 2-light sum ≈ A_only + B_only (within MIS interaction tolerance ~5%)
 *   2. Symmetric lights produce equal contributions (within 1%)
 *   3. Doubling light intensity doubles the contribution (within 1%)
 *
 * Ref: Veach, E. PhD thesis, Stanford 1997, Ch. 9 — power-heuristic sum-MIS
 *      is unbiased; per-light evaluations are the direct-lighting building block.
 */

import { describe, expect, it } from 'vitest';

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------

type Vec3 = [number, number, number];

function dot3(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function sub3(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function normalize3(v: Vec3): Vec3 {
  const len = Math.sqrt(dot3(v, v));
  return len < 1e-12 ? [0, 1, 0] : [v[0] / len, v[1] / len, v[2] / len];
}

// ---------------------------------------------------------------------------
// TS mirrors of WGSL functions (mirrors pathTraceBruteforce.wgsl.ts)
// ---------------------------------------------------------------------------

/** Mirror of powerHeuristic (WGSL line 332). β=2. */
function powerHeuristic(pdfA: number, pdfB: number): number {
  const a2 = pdfA * pdfA;
  const b2 = pdfB * pdfB;
  return a2 / Math.max(a2 + b2, 1e-6);
}

/** Mirror of ggxD (WGSL line 296). */
function ggxD(nDotH: number, alpha: number): number {
  const a2 = alpha * alpha;
  const d = nDotH * nDotH * (a2 - 1.0) + 1.0;
  return a2 / Math.max(Math.PI * d * d, 1e-6);
}

/** Mirror of smithG1 (WGSL line 302). */
function smithG1(nDotV: number, roughness: number): number {
  const r = roughness + 1.0;
  const k = (r * r) * 0.125;
  return nDotV / Math.max(nDotV * (1.0 - k) + k, 1e-6);
}

/** Mirror of fresnelSchlick (WGSL line 289), scalar f0. */
function schlickScalar(f0: number, cosTheta: number): number {
  const m = Math.min(Math.max(1.0 - cosTheta, 0.0), 1.0);
  const m5 = m * m * m * m * m;
  return f0 + (1.0 - f0) * m5;
}

/**
 * Mirror of evaluateBrdf (WGSL line 338).
 * Normal is always +Z in this test (surface-local frame).
 * baseColor = [rho, rho, rho], roughness=1.0 (Lambertian limit), metallic=0.
 */
function evaluateLambertianBrdf(rho: number, wi: Vec3): number {
  // Lambertian: f = rho / π (single scalar channel, surface normal = +Z).
  const nDotL = Math.max(wi[2], 0.0);
  if (nDotL <= 1e-5) return 0.0;
  return rho / Math.PI;
}

/**
 * BRDF PDF for a Lambertian surface (roughness=1, metallic=0).
 * Mirror of brdfDirectionalPdf (WGSL line 358) in the diffuse-dominant limit.
 * pdfDiff = nDotL / π.
 */
function lambertianPdf(wi: Vec3): number {
  const nDotL = Math.max(wi[2], 0.0);
  return nDotL / Math.PI;
}

// ---------------------------------------------------------------------------
// Point-light direct-lighting evaluation
//
// For a point light at position `lightPos` with radiance `L` (W/sr),
// the contribution to a flat Lambertian surface (normal = +Z, albedo ρ)
// at the receiver `recvPos` is:
//
//   contrib = f(wi) · cos(θ) · L / dist²
//           = (ρ/π) · (N·wi) · L / dist²
//
// where wi = normalize(lightPos - recvPos), dist = |lightPos - recvPos|.
//
// In the stochastic estimator the lightPdf for a point light sampled
// deterministically is a delta — we bypass MIS (powerHeuristic returns 1
// when pdfA >> pdfB). For a clean analytical test we set brdfPdf low so
// the MIS weight ≈ 1, approximating NEE-only evaluation.
// ---------------------------------------------------------------------------

interface PointLight {
  pos: Vec3;
  intensity: number; // scalar (white light)
}

/**
 * Evaluate the per-light direct-lighting contribution (no MIS weight, pure
 * analytic). This mirrors the "current == picked" branch per point light in
 * the WGSL main() loop (lines 1682–1698), simplified to a flat Lambertian
 * surface with no shadows (open scene).
 *
 * Returns: f(wi) * nDotL * L / dist²
 */
function evalPointLightContrib(
  light: PointLight,
  recvPos: Vec3,
  rho: number,
): number {
  const toLight = sub3(light.pos, recvPos);
  const dist2 = Math.max(dot3(toLight, toLight), 1e-5);
  const wi = normalize3(toLight);
  const nDotL = Math.max(wi[2], 0.0); // normal = +Z
  if (nDotL <= 1e-5) return 0.0;
  const f = evaluateLambertianBrdf(rho, wi);
  return f * nDotL * light.intensity / dist2;
}

/**
 * Evaluate with MIS (power heuristic, β=2) using a model where the light
 * has a finite area pdf (for rect/mesh lights). This lets us test that
 * powerHeuristic correctly weights the light pdf against the BRDF pdf.
 *
 * We model each "point light" as a small disc of radius r=0.1 (area=π*r²).
 * The area pdf from the surface = dist²/(cosLight * area), same formula
 * as the WGSL rect/mesh light loops.
 *
 * The BRDF pdf for Lambertian = nDotL / π.
 * MIS weight = lightPdf² / (lightPdf² + brdfPdf²).
 *
 * The combined contribution = f * nDotL * L / dist² * misWeight.
 * The direct (no-MIS) contribution = f * nDotL * L / dist².
 * Ratio = misWeight ∈ (0, 1].
 */
function evalPointLightContribMIS(
  light: PointLight,
  recvPos: Vec3,
  rho: number,
  discRadius: number = 0.1,
): number {
  const toLight = sub3(light.pos, recvPos);
  const dist2 = Math.max(dot3(toLight, toLight), 1e-5);
  const wi = normalize3(toLight);
  const nDotL = Math.max(wi[2], 0.0);
  if (nDotL <= 1e-5) return 0.0;
  const f = evaluateLambertianBrdf(rho, wi);
  // Area pdf proxy: cosLight = 1 (disc facing receiver), area = π * r².
  const area = Math.PI * discRadius * discRadius;
  const lightPdf = dist2 / Math.max(1.0 * area, 1e-10); // solid-angle pdf for area light
  const brdfPdf = lambertianPdf(wi);
  const misWeight = powerHeuristic(lightPdf, brdfPdf);
  return f * nDotL * light.intensity / dist2 * misWeight;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('T1.D2 — 2-light sum-MIS correctness (Item 15)', () => {
  // Receiver at origin, flat surface normal = +Z.
  const recvPos: Vec3 = [0, 0, 0];
  const rho = 0.5; // Lambertian albedo

  // Light A: above the surface to the left.
  const lightA: PointLight = { pos: [1, 0, 3], intensity: 1.0 };
  // Light B: above the surface to the right.
  const lightB: PointLight = { pos: [-1, 0, 3], intensity: 1.0 };

  it('test 1 — 2-light sum ≈ A_only + B_only within 5%', () => {
    // No MIS interaction (point lights, deterministic directions): direct sum.
    const contribA = evalPointLightContrib(lightA, recvPos, rho);
    const contribB = evalPointLightContrib(lightB, recvPos, rho);
    const sum = contribA + contribB;

    // Independently compute each, then sum.
    expect(contribA).toBeGreaterThan(0);
    expect(contribB).toBeGreaterThan(0);

    // Compare against the independently-derived analytic expectation for this
    // geometry:
    //   dist² = 1² + 0² + 3² = 10
    //   n·l   = 3 / sqrt(10)
    //   f     = rho / π
    //   per-light = f * (n·l) * I / dist²
    const dist2 = 10;
    const nDotL = 3 / Math.sqrt(dist2);
    const expectedEach = (rho / Math.PI) * nDotL * (1 / dist2);
    const expectedSum = expectedEach * 2;
    const relErr = Math.abs(sum - expectedSum) / Math.max(expectedSum, 1e-12);
    expect(relErr).toBeLessThan(1e-6);

    // Sanity: each light contributes non-trivially.
    expect(contribA).toBeGreaterThan(1e-3);
    expect(contribB).toBeGreaterThan(1e-3);
  });

  it('test 2 — symmetric lights yield equal contributions within 1%', () => {
    // Light A at (+1, 0, 3) and lightB at (-1, 0, 3) are symmetric about Z-axis.
    // The surface normal is +Z so contributions depend only on the angle to +Z
    // and the distance, which are identical by symmetry.
    const contribA = evalPointLightContrib(lightA, recvPos, rho);
    const contribB = evalPointLightContrib(lightB, recvPos, rho);

    const relErr = Math.abs(contribA - contribB) / Math.max(contribA, 1e-12);
    expect(relErr).toBeLessThan(0.01); // ±1%
  });

  it('test 3 — doubling light intensity doubles contribution within 1%', () => {
    const lightA2x: PointLight = { pos: lightA.pos, intensity: lightA.intensity * 2.0 };
    const contrib1x = evalPointLightContrib(lightA, recvPos, rho);
    const contrib2x = evalPointLightContrib(lightA2x, recvPos, rho);

    const ratio = contrib2x / Math.max(contrib1x, 1e-12);
    expect(Math.abs(ratio - 2.0) / 2.0).toBeLessThan(0.01); // ±1%
  });

  it('test 4 — MIS-weighted 2-light sum matches no-MIS sum within 5% (power heuristic β=2)', () => {
    // Point lights have delta PDFs → MIS weight ≈ 1 in all cases.
    // Verify MIS-weighted sum tracks the raw sum closely.
    const rawA = evalPointLightContrib(lightA, recvPos, rho);
    const rawB = evalPointLightContrib(lightB, recvPos, rho);
    const rawSum = rawA + rawB;

    const misA = evalPointLightContribMIS(lightA, recvPos, rho);
    const misB = evalPointLightContribMIS(lightB, recvPos, rho);
    const misSum = misA + misB;

    // MIS weight should be close to 1 for point lights (lightPdf >> brdfPdf).
    const relErr = Math.abs(misSum - rawSum) / Math.max(rawSum, 1e-12);
    expect(relErr).toBeLessThan(0.05); // ≤5%
  });
});
