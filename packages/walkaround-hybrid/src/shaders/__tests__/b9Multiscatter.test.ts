/**
 * B9 (road-to-100) — Kulla-Conty GGX multiscatter energy compensation.
 *
 * CPU mirror of the WGSL `ggxMultiscatter` / `ggxDirectionalAlbedo` /
 * `ggxAverageAlbedo` fit in ggxBrdf.wgsl.ts. Two gates:
 *   1. STRUCTURAL — the WGSL exposes the multiscatter helper and folds it into
 *      evalGGXSpecularOnly (the gated glossy/metal GI lobe) but NOT into evalGGX
 *      (the direct-light lobe) — diffuse-default byte-identity invariant.
 *   2. WHITE-FURNACE — for a perfect conductor (F0 = 1) the directional albedo
 *      E(μ) PLUS the cosine-integrated multiscatter lobe recovers ≈ all incident
 *      energy at every roughness. Without compensation a rough conductor reads
 *      dark; with it the furnace residual → 0.
 */
import { describe, it, expect } from 'vitest';
import { GGX_BRDF_WGSL } from '../ggxBrdf.wgsl.js';

const PI = Math.PI;

// ── CPU mirror of the WGSL fit (must track ggxBrdf.wgsl.ts exactly) ──────────
function ggxDirectionalAlbedo(mu: number, rough: number): number {
  const a = Math.min(1, Math.max(0, rough));
  const c = Math.min(1, Math.max(0, mu));
  const a2 = a * a;
  return Math.min(1, Math.max(0, 1 - a2 * (1 - c) * (0.75 + 0.25 * c)));
}
function ggxAverageAlbedo(rough: number): number {
  const a = Math.min(1, Math.max(0, rough));
  const a2 = a * a;
  return Math.min(1, Math.max(0, 1 - a2 * (7 / 24)));
}
function ggxMultiscatterScalar(rough: number, NdotV: number, NdotL: number): number {
  // Scalar (F0 = 1, Favg = 1) form of ggxMultiscatter — perfect conductor.
  const Eo = ggxDirectionalAlbedo(NdotV, rough);
  const Ei = ggxDirectionalAlbedo(NdotL, rough);
  const Eavg = ggxAverageAlbedo(rough);
  const oneMinusEavg = 1 - Eavg;
  if (oneMinusEavg < 1e-4) return 0;
  const fms = ((1 - Eo) * (1 - Ei)) / (PI * oneMinusEavg);
  // Favg = 1 → Fms = (1·1·Eavg)/(1 − 1·(1−Eavg)) = Eavg/Eavg = 1.
  return fms;
}

describe('B9 — Kulla-Conty multiscatter (structural)', () => {
  it('ggxBrdf exposes the multiscatter helper + albedo fits', () => {
    expect(GGX_BRDF_WGSL).toContain('fn ggxMultiscatter(');
    expect(GGX_BRDF_WGSL).toContain('fn ggxDirectionalAlbedo(');
    expect(GGX_BRDF_WGSL).toContain('fn ggxAverageAlbedo(');
  });

  it('multiscatter is folded into evalGGXSpecularOnlyWithSpecular (the gated GI lobe)', () => {
    const start = GGX_BRDF_WGSL.indexOf('fn evalGGXSpecularOnlyWithSpecular(');
    const body = GGX_BRDF_WGSL.slice(start);
    expect(body).toContain('ggxMultiscatter(F0');
    expect(body).toContain('(specular + ms) * NdotL');
  });

  it('multiscatter is NOT folded into the direct-light evalGGX (byte-identity)', () => {
    // Slice the evalGGX body only (up to the next top-level `fn ` decl), so the
    // helper functions declared between evalGGX and evalGGXSpecularOnly do not
    // leak into the negative match.
    const start = GGX_BRDF_WGSL.indexOf('fn evalGGX(');
    const next = GGX_BRDF_WGSL.indexOf('\nfn ', start + 1);
    const body = GGX_BRDF_WGSL.slice(start, next);
    expect(body).not.toContain('ggxMultiscatter');
  });
});

describe('B9 — white-furnace energy conservation (perfect conductor)', () => {
  // For F0 = 1 the single-scatter directional albedo + the cosine-integrated
  // multiscatter lobe should recover ≈ 1 (no energy lost) at every roughness.
  //   recovered(μo) = E(μo) + ∫_hemisphere f_ms(μo,μi) · μi dωi
  //                 = E(μo) + 2π·∫₀¹ f_ms(μo,μi)·μi dμi
  function recoveredEnergy(NdotV: number, rough: number): number {
    const E = ggxDirectionalAlbedo(NdotV, rough);
    // Numerically integrate the multiscatter lobe over the hemisphere.
    let msIntegral = 0;
    const N = 2048;
    for (let i = 0; i < N; i += 1) {
      const mu = (i + 0.5) / N; // NdotL ∈ (0,1)
      const fms = ggxMultiscatterScalar(rough, NdotV, mu);
      msIntegral += fms * mu * (1 / N); // ∫₀¹ f_ms·μ dμ
    }
    msIntegral *= 2 * PI; // azimuthal integral (f_ms is isotropic in φ)
    return E + msIntegral;
  }

  for (const rough of [0.2, 0.4, 0.6, 0.8, 1.0]) {
    it(`furnace residual is small at rough=${rough}`, () => {
      for (const mu of [0.2, 0.5, 0.9]) {
        const recovered = recoveredEnergy(mu, rough);
        // Compensation must IMPROVE on single-scatter (which loses energy) and
        // never over-shoot egregiously. The analytic fit targets residual → 0;
        // a 12% band absorbs the closed-form fit error vs the true integral.
        expect(recovered).toBeGreaterThan(0.88);
        expect(recovered).toBeLessThan(1.12);
      }
    });
  }

  it('multiscatter vanishes at low roughness (single-scatter unchanged)', () => {
    // rough → 0 ⇒ Eavg → 1 ⇒ oneMinusEavg < 1e-4 ⇒ ms = 0.
    expect(ggxMultiscatterScalar(0.0, 0.5, 0.5)).toBe(0);
    // Smooth surfaces keep nearly all single-scatter energy already.
    expect(ggxDirectionalAlbedo(0.5, 0.05)).toBeGreaterThan(0.99);
  });
});
